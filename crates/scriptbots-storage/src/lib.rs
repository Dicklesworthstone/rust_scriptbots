//! FrankenSQLite-backed persistence layer for ScriptBots.

use arc_swap::ArcSwap;
use crossbeam_channel as xchan;
use fsqlite::{
    Connection, FrankenError, Row, SqliteValue,
    compat::{FromSqliteValue, OpenFlags, RowExt, Transaction, TransactionExt, open_with_flags},
    migrate::MigrationRunner,
};
use scriptbots_core::{
    AgentId, AgentState, BirthRecord, BrainBinding, DeathCause, DeathRecord,
    PersistenceAdmissionError, PersistenceBatch, PersistenceEventKind, ReplayAgentPhase,
    ReplayEvent, ReplayEventKind, ReplayRngScope, WorldPersistence,
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{self, Value, json};
use slotmap::{Key, KeyData};
use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    thread,
    time::Duration,
};
use thiserror::Error;
use tracing::{info, warn};

const DEFAULT_TICK_BUFFER: usize = 32;
const DEFAULT_AGENT_BUFFER: usize = 1024;
const DEFAULT_EVENT_BUFFER: usize = 256;
const DEFAULT_METRIC_BUFFER: usize = 256;
const DEFAULT_LIFECYCLE_BUFFER: usize = 512;
const DEFAULT_REPLAY_BUFFER: usize = 1024;
const DEFAULT_COMMAND_CAPACITY: usize = 8;
const MAX_TRANSACTION_ATTEMPTS: u8 = 4;

/// Files FrankenSQLite may create beside its primary database.
pub const STORAGE_SIDECAR_SUFFIXES: [&str; 7] = [
    "-wal",
    "-shm",
    "-journal",
    "-wal-fec",
    "-lock-shared",
    "-lock-reserved",
    "-lock-pending",
];

const SCRIPTBOTS_SCHEMA_V1: &str = "
    CREATE TABLE ticks (
        tick INTEGER PRIMARY KEY CHECK (tick >= 0),
        epoch INTEGER NOT NULL CHECK (epoch >= 0),
        closed INTEGER NOT NULL CHECK (closed IN (0, 1)),
        agent_count INTEGER NOT NULL CHECK (agent_count >= 0),
        births INTEGER NOT NULL CHECK (births >= 0),
        deaths INTEGER NOT NULL CHECK (deaths >= 0),
        total_energy REAL NOT NULL,
        average_energy REAL NOT NULL,
        average_health REAL NOT NULL
    );
    CREATE TABLE metrics (
        tick INTEGER NOT NULL CHECK (tick >= 0),
        name TEXT NOT NULL,
        value REAL NOT NULL,
        PRIMARY KEY (tick, name)
    );
    CREATE TABLE events (
        tick INTEGER NOT NULL CHECK (tick >= 0),
        kind TEXT NOT NULL,
        count INTEGER NOT NULL CHECK (count >= 0),
        PRIMARY KEY (tick, kind)
    );
    CREATE TABLE replay_events (
        tick INTEGER NOT NULL CHECK (tick >= 0),
        seq INTEGER NOT NULL CHECK (seq >= 0),
        agent_id INTEGER CHECK (agent_id IS NULL OR agent_id >= 0),
        scope TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        PRIMARY KEY (tick, seq)
    );
    CREATE TABLE agents (
        tick INTEGER NOT NULL CHECK (tick >= 0),
        agent_id INTEGER NOT NULL CHECK (agent_id >= 0),
        generation INTEGER NOT NULL CHECK (generation >= 0),
        age INTEGER NOT NULL CHECK (age >= 0),
        position_x REAL NOT NULL,
        position_y REAL NOT NULL,
        velocity_x REAL NOT NULL,
        velocity_y REAL NOT NULL,
        heading REAL NOT NULL,
        health REAL NOT NULL,
        energy REAL NOT NULL,
        color_r REAL NOT NULL,
        color_g REAL NOT NULL,
        color_b REAL NOT NULL,
        spike_length REAL NOT NULL,
        boost INTEGER NOT NULL CHECK (boost IN (0, 1)),
        herbivore_tendency REAL NOT NULL,
        sound_multiplier REAL NOT NULL,
        reproduction_counter REAL NOT NULL,
        mutation_rate_primary REAL NOT NULL,
        mutation_rate_secondary REAL NOT NULL,
        trait_smell REAL NOT NULL,
        trait_sound REAL NOT NULL,
        trait_hearing REAL NOT NULL,
        trait_eye REAL NOT NULL,
        trait_blood REAL NOT NULL,
        give_intent REAL NOT NULL,
        brain_binding TEXT NOT NULL,
        brain_key INTEGER CHECK (brain_key IS NULL OR brain_key >= 0),
        food_delta REAL NOT NULL,
        spiked INTEGER NOT NULL CHECK (spiked IN (0, 1)),
        hybrid INTEGER NOT NULL CHECK (hybrid IN (0, 1)),
        sound_output REAL NOT NULL,
        spike_attacker INTEGER NOT NULL CHECK (spike_attacker IN (0, 1)),
        spike_victim INTEGER NOT NULL CHECK (spike_victim IN (0, 1)),
        hit_carnivore INTEGER NOT NULL CHECK (hit_carnivore IN (0, 1)),
        hit_herbivore INTEGER NOT NULL CHECK (hit_herbivore IN (0, 1)),
        hit_by_carnivore INTEGER NOT NULL CHECK (hit_by_carnivore IN (0, 1)),
        hit_by_herbivore INTEGER NOT NULL CHECK (hit_by_herbivore IN (0, 1)),
        PRIMARY KEY (tick, agent_id)
    );
    CREATE TABLE births (
        tick INTEGER NOT NULL CHECK (tick >= 0),
        agent_id INTEGER NOT NULL CHECK (agent_id >= 0),
        parent_a INTEGER CHECK (parent_a IS NULL OR parent_a >= 0),
        parent_b INTEGER CHECK (parent_b IS NULL OR parent_b >= 0),
        brain_kind TEXT,
        brain_key INTEGER CHECK (brain_key IS NULL OR brain_key >= 0),
        herbivore_tendency REAL NOT NULL,
        generation INTEGER NOT NULL CHECK (generation >= 0),
        position_x REAL NOT NULL,
        position_y REAL NOT NULL,
        is_hybrid INTEGER NOT NULL CHECK (is_hybrid IN (0, 1)),
        PRIMARY KEY (tick, agent_id)
    );
    CREATE TABLE deaths (
        tick INTEGER NOT NULL CHECK (tick >= 0),
        agent_id INTEGER NOT NULL CHECK (agent_id >= 0),
        age INTEGER NOT NULL CHECK (age >= 0),
        generation INTEGER NOT NULL CHECK (generation >= 0),
        herbivore_tendency REAL NOT NULL,
        brain_kind TEXT,
        brain_key INTEGER CHECK (brain_key IS NULL OR brain_key >= 0),
        energy REAL NOT NULL,
        food_balance_total REAL NOT NULL,
        cause TEXT NOT NULL,
        was_hybrid INTEGER NOT NULL CHECK (was_hybrid IN (0, 1)),
        spike_attacker INTEGER NOT NULL CHECK (spike_attacker IN (0, 1)),
        spike_victim INTEGER NOT NULL CHECK (spike_victim IN (0, 1)),
        hit_carnivore INTEGER NOT NULL CHECK (hit_carnivore IN (0, 1)),
        hit_herbivore INTEGER NOT NULL CHECK (hit_herbivore IN (0, 1)),
        hit_by_carnivore INTEGER NOT NULL CHECK (hit_by_carnivore IN (0, 1)),
        hit_by_herbivore INTEGER NOT NULL CHECK (hit_by_herbivore IN (0, 1)),
        PRIMARY KEY (tick, agent_id)
    );
";

const AGENT_COLUMNS: &[&str] = &[
    "tick",
    "agent_id",
    "generation",
    "age",
    "position_x",
    "position_y",
    "velocity_x",
    "velocity_y",
    "heading",
    "health",
    "energy",
    "color_r",
    "color_g",
    "color_b",
    "spike_length",
    "boost",
    "herbivore_tendency",
    "sound_multiplier",
    "reproduction_counter",
    "mutation_rate_primary",
    "mutation_rate_secondary",
    "trait_smell",
    "trait_sound",
    "trait_hearing",
    "trait_eye",
    "trait_blood",
    "give_intent",
    "brain_binding",
    "brain_key",
    "food_delta",
    "spiked",
    "hybrid",
    "sound_output",
    "spike_attacker",
    "spike_victim",
    "hit_carnivore",
    "hit_herbivore",
    "hit_by_carnivore",
    "hit_by_herbivore",
];

/// Storage error wrapper.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("FrankenSQLite error: {0}")]
    Database(#[from] FrankenError),
    #[error(
        "FrankenSQLite transaction failed after {attempts} attempt(s) (transient={transient}, commit_state={commit_state:?}): {source}"
    )]
    Transaction {
        attempts: u8,
        transient: bool,
        commit_state: FailureCommitState,
        #[source]
        source: FrankenError,
    },
    #[error("storage is already closed")]
    Closed,
    #[error("storage transaction is terminally failed; buffered rows will not be replayed")]
    TerminallyFailed,
    #[error("invalid storage data in {context}: {reason}")]
    InvalidData {
        context: &'static str,
        reason: String,
    },
    #[error("invalid storage target {path:?}: {reason}")]
    InvalidTarget { path: String, reason: String },
    #[error("failed to {operation} storage path {path:?}: {source}")]
    Filesystem {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(transparent)]
    Worker(#[from] StorageWorkerError),
    #[error("invalid replay event at tick {tick}, seq {seq}: {reason}")]
    ReplayParse { tick: i64, seq: i64, reason: String },
}

#[derive(Debug)]
enum StorageTarget {
    Memory,
    CreateNewFile(String),
}

impl StorageTarget {
    fn path(&self) -> &str {
        match self {
            Self::Memory => ":memory:",
            Self::CreateNewFile(path) => path,
        }
    }

    const fn guarantee(&self) -> PersistenceGuarantee {
        match self {
            Self::Memory => PersistenceGuarantee::CommittedVolatile,
            Self::CreateNewFile(_) => PersistenceGuarantee::Durable,
        }
    }

    fn prepare_for_open(&self) -> Result<(), StorageError> {
        if let Self::CreateNewFile(path) = self {
            ensure_no_storage_sidecars(Path::new(path))?;
        }
        Ok(())
    }
}

fn validate_durable_storage_path(path: &str) -> Result<(), StorageError> {
    let trimmed = path.trim();
    let invalid = |reason: &str| StorageError::InvalidTarget {
        path: path.to_owned(),
        reason: reason.to_owned(),
    };
    if trimmed.is_empty() {
        return Err(invalid("file storage requires a non-empty path"));
    }
    if trimmed == ":memory:" {
        return Err(invalid(
            "the volatile :memory: engine is available only through Storage::memory or StoragePipeline::memory",
        ));
    }
    if trimmed
        .get(.."file:".len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("file:"))
    {
        return Err(invalid(
            "file: URI targets bypass the create-new filesystem contract",
        ));
    }
    Ok(())
}

fn storage_sidecar_paths(path: &Path) -> impl Iterator<Item = PathBuf> + '_ {
    STORAGE_SIDECAR_SUFFIXES.into_iter().map(|suffix| {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(suffix);
        PathBuf::from(sidecar)
    })
}

fn path_entry_exists(path: &Path) -> Result<bool, StorageError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(StorageError::Filesystem {
            operation: "inspect",
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn ensure_no_storage_sidecars(path: &Path) -> Result<(), StorageError> {
    for sidecar in storage_sidecar_paths(path) {
        if path_entry_exists(&sidecar)? {
            return Err(StorageError::InvalidTarget {
                path: path.display().to_string(),
                reason: format!("stale FrankenSQLite sidecar {} exists", sidecar.display()),
            });
        }
    }
    Ok(())
}

fn reserve_new_file_with_hook(
    path: &str,
    after_reservation: impl FnOnce(&Path),
) -> Result<StorageTarget, StorageError> {
    validate_durable_storage_path(path)?;
    let path = Path::new(path);
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|source| StorageError::Filesystem {
            operation: "create parent directory for",
            path: parent.to_path_buf(),
            source,
        })?;
    }
    ensure_no_storage_sidecars(path)?;
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(file) => drop(file),
        Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
            return Err(StorageError::InvalidTarget {
                path: path.display().to_string(),
                reason: "refusing to reuse an existing single-run database path".to_owned(),
            });
        }
        Err(source) => {
            return Err(StorageError::Filesystem {
                operation: "reserve new",
                path: path.to_path_buf(),
                source,
            });
        }
    }
    after_reservation(path);
    // Leave the reservation in place on failure. Without an identity-bound
    // descriptor, pathname cleanup could delete a file swapped in by a racer.
    ensure_no_storage_sidecars(path)?;
    Ok(StorageTarget::CreateNewFile(path.display().to_string()))
}

fn reserve_new_file(path: &str) -> Result<StorageTarget, StorageError> {
    reserve_new_file_with_hook(path, |_| {})
}

/// Worker operation associated with a structured persistence failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageOperation {
    Startup,
    Admit,
    Persist,
    Flush,
    Shutdown,
    Close,
    Join,
}

/// What is known about the affected batch when a worker operation fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureCommitState {
    NotAdmitted,
    RolledBack,
    Indeterminate,
    Committed,
}

/// Structured error crossing the storage worker boundary.
#[derive(Debug, Error)]
pub enum StorageWorkerError {
    #[error(
        "storage {operation:?} failed at {path} (tick={tick:?}, attempt={attempt}, transient={transient}, commit_state={commit_state:?}): {source}"
    )]
    Database {
        operation: StorageOperation,
        path: String,
        tick: Option<u64>,
        attempt: u8,
        transient: bool,
        commit_state: FailureCommitState,
        #[source]
        source: FrankenError,
    },
    #[error(
        "storage {operation:?} channel failed at {path} (tick={tick:?}, commit_state={commit_state:?}): {detail}"
    )]
    Channel {
        operation: StorageOperation,
        path: String,
        tick: Option<u64>,
        commit_state: FailureCommitState,
        detail: String,
    },
    #[error(
        "storage {operation:?} failed at {path} (tick={tick:?}, commit_state={commit_state:?}): {detail}"
    )]
    Internal {
        operation: StorageOperation,
        path: String,
        tick: Option<u64>,
        commit_state: FailureCommitState,
        detail: String,
    },
}

/// Cloneable structured failure state published to frontends and host supervision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageFailureStatus {
    pub kind: StorageFailureKind,
    pub operation: StorageOperation,
    pub path: Option<String>,
    pub tick: Option<u64>,
    pub attempt: u8,
    pub transient: bool,
    pub commit_state: FailureCommitState,
    pub detail: String,
}

/// Stable failure category used to preserve the most informative terminal cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StorageFailureKind {
    Channel,
    Internal,
    Database,
}

impl StorageWorkerError {
    #[must_use]
    pub fn status(&self) -> StorageFailureStatus {
        match self {
            Self::Database {
                operation,
                path,
                tick,
                attempt,
                transient,
                commit_state,
                source,
            } => StorageFailureStatus {
                kind: StorageFailureKind::Database,
                operation: *operation,
                path: Some(path.clone()),
                tick: *tick,
                attempt: *attempt,
                transient: *transient,
                commit_state: *commit_state,
                detail: source.to_string(),
            },
            Self::Channel {
                operation,
                path,
                tick,
                commit_state,
                detail,
            } => StorageFailureStatus {
                kind: StorageFailureKind::Channel,
                operation: *operation,
                path: Some(path.clone()),
                tick: *tick,
                attempt: 0,
                transient: false,
                commit_state: *commit_state,
                detail: detail.clone(),
            },
            Self::Internal {
                operation,
                path,
                tick,
                commit_state,
                detail,
            } => StorageFailureStatus {
                kind: StorageFailureKind::Internal,
                operation: *operation,
                path: Some(path.clone()),
                tick: *tick,
                attempt: 0,
                transient: false,
                commit_state: *commit_state,
                detail: detail.clone(),
            },
        }
    }
}

fn worker_error_from_storage(
    operation: StorageOperation,
    path: &str,
    tick: Option<u64>,
    default_commit_state: FailureCommitState,
    error: StorageError,
) -> StorageWorkerError {
    match error {
        StorageError::Database(source) => StorageWorkerError::Database {
            operation,
            path: path.to_owned(),
            tick,
            attempt: 1,
            transient: source.is_transient(),
            commit_state: default_commit_state,
            source,
        },
        StorageError::Transaction {
            attempts,
            transient,
            commit_state,
            source,
        } => StorageWorkerError::Database {
            operation,
            path: path.to_owned(),
            tick,
            attempt: attempts,
            transient,
            commit_state,
            source,
        },
        StorageError::Worker(error) => error,
        other => StorageWorkerError::Internal {
            operation,
            path: path.to_owned(),
            tick,
            commit_state: default_commit_state,
            detail: other.to_string(),
        },
    }
}

/// Rebuild an equivalent structured worker error so the terminal failure can be
/// both replied to the requester and returned from the worker thread join.
///
/// `FrankenError` is not `Clone`, so the `Database` variant carries the source's
/// rendered message instead of the original source value; every other field is
/// preserved verbatim.
fn duplicate_worker_error(error: &StorageWorkerError) -> StorageWorkerError {
    match error {
        StorageWorkerError::Database {
            operation,
            path,
            tick,
            attempt,
            transient,
            commit_state,
            source,
        } => StorageWorkerError::Database {
            operation: *operation,
            path: path.clone(),
            tick: *tick,
            attempt: *attempt,
            transient: *transient,
            commit_state: *commit_state,
            source: FrankenError::Internal(source.to_string()),
        },
        StorageWorkerError::Channel {
            operation,
            path,
            tick,
            commit_state,
            detail,
        } => StorageWorkerError::Channel {
            operation: *operation,
            path: path.clone(),
            tick: *tick,
            commit_state: *commit_state,
            detail: detail.clone(),
        },
        StorageWorkerError::Internal {
            operation,
            path,
            tick,
            commit_state,
            detail,
        } => StorageWorkerError::Internal {
            operation: *operation,
            path: path.clone(),
            tick: *tick,
            commit_state: *commit_state,
            detail: detail.clone(),
        },
    }
}

/// Summary row written to the `ticks` table.
#[derive(Debug, Clone)]
struct TickRow {
    tick: i64,
    epoch: i64,
    closed: bool,
    agent_count: i64,
    births: i64,
    deaths: i64,
    total_energy: f64,
    average_energy: f64,
    average_health: f64,
}

/// Metric row written to the `metrics` table.
#[derive(Debug, Clone)]
struct MetricRow {
    tick: i64,
    name: String,
    value: f64,
}

/// Event row persisted for analytics.
#[derive(Debug, Clone)]
struct EventRow {
    tick: i64,
    kind: String,
    count: i64,
}

/// Latest metric reading fetched for analytics displays.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricReading {
    pub tick: i64,
    pub name: String,
    pub value: f64,
}

/// Historical metric row returned by the read-only storage API.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistedMetric {
    pub tick: u64,
    pub name: String,
    pub value: f64,
}

/// Tick ledger row exposed to storage consumers without leaking SQL details.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistedTick {
    pub tick: u64,
    pub epoch: u64,
    pub closed: bool,
    pub agent_count: usize,
    pub births: usize,
    pub deaths: usize,
    pub total_energy: f64,
    pub average_energy: f64,
    pub average_health: f64,
}

/// Cross-table lifecycle totals for validating and summarizing a completed run.
#[derive(Debug, Clone, PartialEq)]
pub struct RunLedgerSummary {
    pub tick_count: u64,
    pub latest_tick: Option<PersistedTick>,
    pub birth_records: u64,
    pub death_records: u64,
    pub birth_events: u64,
    pub death_events: u64,
}

/// Immutable, lock-free read model published after successful storage commits.
#[derive(Debug, Clone)]
pub struct AnalyticsSnapshot {
    pub revision: u64,
    pub committed_tick: Option<u64>,
    pub committed_agent_count: Option<usize>,
    pub readings: Arc<[MetricReading]>,
    pub last_error: Option<Arc<str>>,
    pub last_failure: Option<Arc<StorageFailureStatus>>,
    pub stopped: bool,
}

impl Default for AnalyticsSnapshot {
    fn default() -> Self {
        Self {
            revision: 0,
            committed_tick: None,
            committed_agent_count: None,
            readings: Arc::from([]),
            last_error: None,
            last_failure: None,
            stopped: false,
        }
    }
}

/// Cloneable read-only handle for the latest committed analytics state.
#[derive(Clone)]
pub struct AnalyticsSnapshotProvider {
    inner: Arc<ArcSwap<AnalyticsSnapshot>>,
}

impl AnalyticsSnapshotProvider {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(AnalyticsSnapshot::default())),
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> Arc<AnalyticsSnapshot> {
        self.inner.load_full()
    }

    fn publish_committed(&self, pending: PendingAnalytics) {
        self.inner.rcu(|current| {
            if current.stopped {
                return Arc::clone(current);
            }
            Arc::new(AnalyticsSnapshot {
                revision: current.revision.saturating_add(1),
                committed_tick: Some(pending.tick),
                committed_agent_count: Some(pending.agent_count),
                readings: Arc::clone(&pending.readings),
                last_error: None,
                last_failure: None,
                stopped: false,
            })
        });
    }

    fn publish_worker_error(&self, error: &StorageWorkerError, stopped: bool) {
        let incoming = Arc::new(error.status());
        let error_text: Arc<str> = Arc::from(error.to_string());
        self.inner.rcu(|current| {
            let preserve_existing = current.stopped
                && current
                    .last_failure
                    .as_ref()
                    .is_some_and(|existing| existing.kind >= incoming.kind);
            if preserve_existing {
                return Arc::clone(current);
            }
            Arc::new(AnalyticsSnapshot {
                revision: current.revision.saturating_add(1),
                committed_tick: current.committed_tick,
                committed_agent_count: current.committed_agent_count,
                readings: Arc::clone(&current.readings),
                last_error: Some(Arc::clone(&error_text)),
                last_failure: Some(Arc::clone(&incoming)),
                stopped,
            })
        });
    }

    fn publish_stopped(&self) {
        self.inner.rcu(|current| {
            if current.stopped {
                return Arc::clone(current);
            }
            Arc::new(AnalyticsSnapshot {
                revision: current.revision.saturating_add(1),
                committed_tick: current.committed_tick,
                committed_agent_count: current.committed_agent_count,
                readings: Arc::clone(&current.readings),
                last_error: current.last_error.clone(),
                last_failure: current.last_failure.clone(),
                stopped: true,
            })
        });
    }
}

#[derive(Debug)]
struct PendingAnalytics {
    tick: u64,
    agent_count: usize,
    readings: Arc<[MetricReading]>,
}

impl PendingAnalytics {
    fn from_batch(batch: &PersistenceBatch) -> Result<Self, StorageError> {
        let tick = batch.summary.tick.0;
        let tick_column = encode_u64("metrics.tick", tick)?;
        let mut values = BTreeMap::new();
        for metric in &batch.metrics {
            values.insert(metric.name.to_string(), metric.value);
        }
        let readings = values
            .into_iter()
            .map(|(name, value)| MetricReading {
                tick: tick_column,
                name,
                value,
            })
            .collect::<Vec<_>>();
        Ok(Self {
            tick,
            agent_count: batch.summary.agent_count,
            readings: Arc::from(readings),
        })
    }
}

#[derive(Debug)]
struct PreparedPersistenceBatch {
    tick: u64,
    storage: StorageBuffer,
    analytics: PendingAnalytics,
}

impl PreparedPersistenceBatch {
    fn from_batch(batch: &PersistenceBatch) -> Result<Self, StorageError> {
        Ok(Self {
            tick: batch.summary.tick.0,
            storage: Storage::prepare_batch(batch)?,
            analytics: PendingAnalytics::from_batch(batch)?,
        })
    }
}

/// Agent snapshot row.
#[derive(Debug, Clone)]
struct AgentRow {
    tick: i64,
    agent_id: i64,
    generation: i64,
    age: i64,
    position_x: f64,
    position_y: f64,
    velocity_x: f64,
    velocity_y: f64,
    heading: f64,
    health: f64,
    energy: f64,
    color_r: f64,
    color_g: f64,
    color_b: f64,
    spike_length: f64,
    boost: bool,
    herbivore_tendency: f64,
    sound_multiplier: f64,
    reproduction_counter: f64,
    mutation_rate_primary: f64,
    mutation_rate_secondary: f64,
    trait_smell: f64,
    trait_sound: f64,
    trait_hearing: f64,
    trait_eye: f64,
    trait_blood: f64,
    give_intent: f64,
    brain_binding: String,
    brain_key: Option<i64>,
    food_delta: f64,
    spiked: bool,
    hybrid: bool,
    sound_output: f64,
    spike_attacker: bool,
    spike_victim: bool,
    hit_carnivore: bool,
    hit_herbivore: bool,
    hit_by_carnivore: bool,
    hit_by_herbivore: bool,
}

#[derive(Debug, Clone)]
struct BirthRow {
    tick: i64,
    agent_id: i64,
    parent_a: Option<i64>,
    parent_b: Option<i64>,
    brain_kind: Option<String>,
    brain_key: Option<i64>,
    herbivore_tendency: f64,
    generation: i64,
    position_x: f64,
    position_y: f64,
    is_hybrid: bool,
}

#[derive(Debug, Clone)]
struct DeathRow {
    tick: i64,
    agent_id: i64,
    age: i64,
    generation: i64,
    herbivore_tendency: f64,
    brain_kind: Option<String>,
    brain_key: Option<i64>,
    energy: f64,
    food_balance_total: f64,
    cause: String,
    was_hybrid: bool,
    spike_attacker: bool,
    spike_victim: bool,
    hit_carnivore: bool,
    hit_herbivore: bool,
    hit_by_carnivore: bool,
    hit_by_herbivore: bool,
}

#[derive(Debug, Clone)]
struct ReplayEventRow {
    tick: i64,
    seq: i64,
    agent_id: Option<i64>,
    scope: String,
    event_type: String,
    payload: String,
}

/// Aggregate event count grouped by replay event type.
#[derive(Debug, Clone)]
pub struct ReplayEventCount {
    pub event_type: String,
    pub count: u64,
}

/// Replay event reconstructed from persisted storage.
#[derive(Debug, Clone)]
pub struct PersistedReplayEvent {
    pub tick: u64,
    pub seq: u64,
    pub event: ReplayEvent,
}

#[derive(Debug, Default)]
struct StorageBuffer {
    ticks: Vec<TickRow>,
    metrics: Vec<MetricRow>,
    events: Vec<EventRow>,
    agents: Vec<AgentRow>,
    births: Vec<BirthRow>,
    deaths: Vec<DeathRow>,
    replay_events: Vec<ReplayEventRow>,
}

#[derive(Debug)]
struct FlushAttemptError {
    source: FrankenError,
    commit_state: FailureCommitState,
}

fn should_retry_transaction(error: &FlushAttemptError, attempt: u8) -> bool {
    error.source.is_transient()
        && error.commit_state == FailureCommitState::RolledBack
        && attempt < MAX_TRANSACTION_ATTEMPTS
}

impl StorageBuffer {
    fn is_empty(&self) -> bool {
        self.ticks.is_empty()
            && self.metrics.is_empty()
            && self.events.is_empty()
            && self.agents.is_empty()
            && self.births.is_empty()
            && self.deaths.is_empty()
            && self.replay_events.is_empty()
    }

    fn clear(&mut self) {
        self.ticks.clear();
        self.metrics.clear();
        self.events.clear();
        self.agents.clear();
        self.births.clear();
        self.deaths.clear();
        self.replay_events.clear();
    }

    fn append(&mut self, mut other: Self) {
        self.ticks.append(&mut other.ticks);
        self.metrics.append(&mut other.metrics);
        self.events.append(&mut other.events);
        self.agents.append(&mut other.agents);
        self.births.append(&mut other.births);
        self.deaths.append(&mut other.deaths);
        self.replay_events.append(&mut other.replay_events);
    }
}

fn sqlite_bool(value: bool) -> SqliteValue {
    SqliteValue::Integer(i64::from(value))
}

fn sqlite_optional_i64(value: Option<i64>) -> SqliteValue {
    value.map_or(SqliteValue::Null, SqliteValue::Integer)
}

fn sqlite_optional_text(value: Option<&str>) -> SqliteValue {
    value.map_or(SqliteValue::Null, SqliteValue::from)
}

fn checked_u64(context: &'static str, value: i64) -> Result<u64, StorageError> {
    u64::try_from(value).map_err(|error| StorageError::InvalidData {
        context,
        reason: error.to_string(),
    })
}

fn checked_usize(context: &'static str, value: i64) -> Result<usize, StorageError> {
    usize::try_from(value).map_err(|error| StorageError::InvalidData {
        context,
        reason: error.to_string(),
    })
}

fn checked_i64(context: &'static str, value: usize) -> Result<i64, StorageError> {
    i64::try_from(value).map_err(|error| StorageError::InvalidData {
        context,
        reason: error.to_string(),
    })
}

/// Checked `u64` -> `i64` conversion for values headed into SQLite INTEGER
/// columns. Values above `i64::MAX` would otherwise wrap negative on write
/// while the read side rejects negatives, so out-of-range input must fail
/// batch preparation (admission) instead of poisoning the durable file.
fn encode_u64(context: &'static str, value: u64) -> Result<i64, StorageError> {
    i64::try_from(value).map_err(|_| StorageError::InvalidData {
        context,
        reason: format!("value {value} exceeds the i64 range supported by storage"),
    })
}

fn decode<T: FromSqliteValue>(
    row: &Row,
    index: usize,
    context: &'static str,
) -> Result<T, StorageError> {
    row.get_typed(index)
        .map_err(|error| StorageError::InvalidData {
            context,
            reason: error.to_string(),
        })
}

/// Read-only view over an existing ScriptBots database.
pub struct StorageReader {
    conn: Option<Connection>,
}

impl StorageReader {
    /// Open an existing FrankenSQLite database without creating or migrating it.
    pub fn open(path: &str) -> Result<Self, StorageError> {
        validate_durable_storage_path(path)?;
        let conn = open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        Ok(Self { conn: Some(conn) })
    }

    fn connection(&self) -> Result<&Connection, StorageError> {
        self.conn.as_ref().ok_or(StorageError::Closed)
    }

    /// Close the read-only connection without attempting a WAL checkpoint.
    pub fn close(mut self) -> Result<(), StorageError> {
        let connection = self.conn.take().ok_or(StorageError::Closed)?;
        connection.close_without_checkpoint()?;
        Ok(())
    }

    /// Return the maximum durable tick, if the database contains tick rows.
    pub fn max_tick(&self) -> Result<Option<u64>, StorageError> {
        let row = self
            .connection()?
            .query_row("SELECT MAX(tick) FROM ticks")?;
        decode::<Option<i64>>(&row, 0, "ticks.max_tick")?
            .map(|tick| checked_u64("ticks.max_tick", tick))
            .transpose()
    }

    /// Load replay events in deterministic tick/sequence order.
    pub fn load_replay_events(&self) -> Result<Vec<PersistedReplayEvent>, StorageError> {
        let rows = self.connection()?.query(
            "SELECT tick, seq, agent_id, scope, event_type, payload
             FROM replay_events
             ORDER BY tick ASC, seq ASC",
        )?;
        let mut events = Vec::with_capacity(rows.len());
        for row in rows {
            let replay_row = ReplayEventRow {
                tick: decode(&row, 0, "replay_events.tick")?,
                seq: decode(&row, 1, "replay_events.seq")?,
                agent_id: decode(&row, 2, "replay_events.agent_id")?,
                scope: decode(&row, 3, "replay_events.scope")?,
                event_type: decode(&row, 4, "replay_events.event_type")?,
                payload: decode(&row, 5, "replay_events.payload")?,
            };
            let event = replay_event_from_row(&replay_row)?;
            events.push(PersistedReplayEvent {
                tick: checked_u64("replay_events.tick", replay_row.tick)?,
                seq: checked_u64("replay_events.seq", replay_row.seq)?,
                event,
            });
        }
        Ok(events)
    }

    /// Return replay-event counts grouped by stable event type.
    pub fn replay_event_counts(&self) -> Result<Vec<ReplayEventCount>, StorageError> {
        let rows = self.connection()?.query(
            "SELECT event_type, COUNT(*) AS total
             FROM replay_events
             GROUP BY event_type
             ORDER BY event_type",
        )?;
        let mut counts = Vec::with_capacity(rows.len());
        for row in rows {
            counts.push(ReplayEventCount {
                event_type: decode(&row, 0, "replay_events.event_type")?,
                count: checked_u64(
                    "replay_events.count",
                    decode(&row, 1, "replay_events.count")?,
                )?,
            });
        }
        Ok(counts)
    }

    /// Return agents ranked by average energy across all recorded ticks.
    pub fn top_predators(&self, limit: usize) -> Result<Vec<PredatorStats>, StorageError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let bound = checked_i64("top_predators.limit", limit)?;
        let rows = self.connection()?.query_with_params(
            "SELECT agent_id,
                    AVG(energy) AS avg_energy,
                    MAX(spike_length) AS max_spike_length,
                    MAX(tick) AS last_tick
             FROM agents
             GROUP BY agent_id
             ORDER BY avg_energy DESC
             LIMIT ?1",
            &[bound.into()],
        )?;
        let mut stats = Vec::with_capacity(limit.min(16));
        for row in rows {
            stats.push(PredatorStats {
                agent_id: checked_u64("agents.agent_id", decode(&row, 0, "agents.agent_id")?)?,
                avg_energy: decode(&row, 1, "agents.avg_energy")?,
                max_spike_length: decode(&row, 2, "agents.max_spike_length")?,
                last_tick: decode(&row, 3, "agents.last_tick")?,
            });
        }
        Ok(stats)
    }

    /// Load metric history in chronological order, optionally keeping only the newest rows.
    pub fn recent_metrics(
        &self,
        limit: Option<usize>,
    ) -> Result<Vec<PersistedMetric>, StorageError> {
        if matches!(limit, Some(0)) {
            return Ok(Vec::new());
        }

        let rows = if let Some(limit) = limit {
            let bound = checked_i64("recent_metrics.limit", limit)?;
            self.connection()?.query_with_params(
                "SELECT tick, name, value
                 FROM metrics
                 ORDER BY tick DESC, name DESC
                 LIMIT ?1",
                &[bound.into()],
            )?
        } else {
            self.connection()?.query(
                "SELECT tick, name, value
                 FROM metrics
                 ORDER BY tick DESC, name DESC",
            )?
        };

        let mut readings = Vec::with_capacity(rows.len());
        for row in rows {
            readings.push(PersistedMetric {
                tick: checked_u64("metrics.tick", decode(&row, 0, "metrics.tick")?)?,
                name: decode(&row, 1, "metrics.name")?,
                value: decode(&row, 2, "metrics.value")?,
            });
        }
        readings.reverse();
        Ok(readings)
    }

    /// Load tick history in chronological order, optionally keeping only the newest rows.
    pub fn recent_ticks(&self, limit: Option<usize>) -> Result<Vec<PersistedTick>, StorageError> {
        if matches!(limit, Some(0)) {
            return Ok(Vec::new());
        }

        let rows = if let Some(limit) = limit {
            let bound = checked_i64("recent_ticks.limit", limit)?;
            self.connection()?.query_with_params(
                "SELECT tick, epoch, closed, agent_count, births, deaths,
                        total_energy, average_energy, average_health
                 FROM ticks
                 ORDER BY tick DESC
                 LIMIT ?1",
                &[bound.into()],
            )?
        } else {
            self.connection()?.query(
                "SELECT tick, epoch, closed, agent_count, births, deaths,
                        total_energy, average_energy, average_health
                 FROM ticks
                 ORDER BY tick DESC",
            )?
        };

        let mut ticks = Vec::with_capacity(rows.len());
        for row in rows {
            ticks.push(PersistedTick {
                tick: checked_u64("ticks.tick", decode(&row, 0, "ticks.tick")?)?,
                epoch: checked_u64("ticks.epoch", decode(&row, 1, "ticks.epoch")?)?,
                closed: decode(&row, 2, "ticks.closed")?,
                agent_count: checked_usize(
                    "ticks.agent_count",
                    decode(&row, 3, "ticks.agent_count")?,
                )?,
                births: checked_usize("ticks.births", decode(&row, 4, "ticks.births")?)?,
                deaths: checked_usize("ticks.deaths", decode(&row, 5, "ticks.deaths")?)?,
                total_energy: decode(&row, 6, "ticks.total_energy")?,
                average_energy: decode(&row, 7, "ticks.average_energy")?,
                average_health: decode(&row, 8, "ticks.average_health")?,
            });
        }
        ticks.reverse();
        Ok(ticks)
    }

    /// Summarize the durable tick and lifecycle ledgers for a completed run.
    pub fn run_ledger_summary(&self) -> Result<RunLedgerSummary, StorageError> {
        let mut tx = self.connection()?.transaction()?;
        let query_result = (|| -> Result<RunLedgerSummary, StorageError> {
            let tick_count_row = tx.query_row("SELECT COUNT(*) FROM ticks")?;
            let tick_count =
                checked_u64("ticks.count", decode(&tick_count_row, 0, "ticks.count")?)?;
            let birth_count_row = tx.query_row("SELECT COUNT(*) FROM births")?;
            let birth_records =
                checked_u64("births.count", decode(&birth_count_row, 0, "births.count")?)?;
            let death_count_row = tx.query_row("SELECT COUNT(*) FROM deaths")?;
            let death_records =
                checked_u64("deaths.count", decode(&death_count_row, 0, "deaths.count")?)?;
            let birth_event_row =
                tx.query_row("SELECT COALESCE(SUM(count), 0) FROM events WHERE kind = 'births'")?;
            let birth_events = checked_u64(
                "events.births",
                decode(&birth_event_row, 0, "events.births")?,
            )?;
            let death_event_row =
                tx.query_row("SELECT COALESCE(SUM(count), 0) FROM events WHERE kind = 'deaths'")?;
            let death_events = checked_u64(
                "events.deaths",
                decode(&death_event_row, 0, "events.deaths")?,
            )?;
            let rows = tx.query(
                "SELECT tick, epoch, closed, agent_count, births, deaths,
                        total_energy, average_energy, average_health
                 FROM ticks
                 ORDER BY tick DESC
                 LIMIT 1",
            )?;
            let latest_tick = rows
                .first()
                .map(|row| -> Result<PersistedTick, StorageError> {
                    Ok(PersistedTick {
                        tick: checked_u64("ticks.tick", decode(row, 0, "ticks.tick")?)?,
                        epoch: checked_u64("ticks.epoch", decode(row, 1, "ticks.epoch")?)?,
                        closed: decode(row, 2, "ticks.closed")?,
                        agent_count: checked_usize(
                            "ticks.agent_count",
                            decode(row, 3, "ticks.agent_count")?,
                        )?,
                        births: checked_usize("ticks.births", decode(row, 4, "ticks.births")?)?,
                        deaths: checked_usize("ticks.deaths", decode(row, 5, "ticks.deaths")?)?,
                        total_energy: decode(row, 6, "ticks.total_energy")?,
                        average_energy: decode(row, 7, "ticks.average_energy")?,
                        average_health: decode(row, 8, "ticks.average_health")?,
                    })
                })
                .transpose()?;

            Ok(RunLedgerSummary {
                tick_count,
                latest_tick,
                birth_records,
                death_records,
                birth_events,
                death_events,
            })
        })();
        let rollback_result = tx.rollback().map_err(StorageError::from);
        let summary = query_result?;
        rollback_result?;
        Ok(summary)
    }
}

impl Drop for StorageReader {
    fn drop(&mut self) {
        if let Some(mut connection) = self.conn.take() {
            connection.close_best_effort_in_place();
        }
    }
}

/// FrankenSQLite-backed persistence sink with buffered writes.
pub struct Storage {
    path: String,
    conn: Option<Connection>,
    terminally_failed: bool,
    buffer: StorageBuffer,
    tick_flush_threshold: usize,
    agent_flush_threshold: usize,
    event_flush_threshold: usize,
    metric_flush_threshold: usize,
    birth_flush_threshold: usize,
    death_flush_threshold: usize,
    replay_flush_threshold: usize,
}

impl Storage {
    /// Atomically reserve and create a file-backed run database.
    pub fn create_new_file(path: &str) -> Result<Self, StorageError> {
        Self::create_new_file_with_thresholds(
            path,
            DEFAULT_TICK_BUFFER,
            DEFAULT_AGENT_BUFFER,
            DEFAULT_EVENT_BUFFER,
            DEFAULT_METRIC_BUFFER,
        )
    }

    /// Atomically reserve a file-backed run database with explicit flush thresholds.
    #[allow(dead_code)]
    pub fn create_new_file_with_thresholds(
        path: &str,
        tick: usize,
        agent: usize,
        event: usize,
        metric: usize,
    ) -> Result<Self, StorageError> {
        Self::with_target(reserve_new_file(path)?, tick, agent, event, metric)
    }

    /// Open an isolated volatile database with default buffering thresholds.
    pub fn memory() -> Result<Self, StorageError> {
        Self::memory_with_thresholds(
            DEFAULT_TICK_BUFFER,
            DEFAULT_AGENT_BUFFER,
            DEFAULT_EVENT_BUFFER,
            DEFAULT_METRIC_BUFFER,
        )
    }

    /// Open an isolated volatile database with explicit flush thresholds.
    pub fn memory_with_thresholds(
        tick: usize,
        agent: usize,
        event: usize,
        metric: usize,
    ) -> Result<Self, StorageError> {
        Self::with_target(StorageTarget::Memory, tick, agent, event, metric)
    }

    fn with_target(
        target: StorageTarget,
        tick: usize,
        agent: usize,
        event: usize,
        metric: usize,
    ) -> Result<Self, StorageError> {
        target.prepare_for_open()?;
        let path = target.path().to_owned();
        let conn = Connection::open(&path)?;
        conn.execute("PRAGMA synchronous = FULL;")?;
        let mut storage = Self {
            path,
            conn: Some(conn),
            terminally_failed: false,
            buffer: StorageBuffer::default(),
            tick_flush_threshold: tick,
            agent_flush_threshold: agent,
            event_flush_threshold: event,
            metric_flush_threshold: metric,
            birth_flush_threshold: DEFAULT_LIFECYCLE_BUFFER,
            death_flush_threshold: DEFAULT_LIFECYCLE_BUFFER,
            replay_flush_threshold: DEFAULT_REPLAY_BUFFER,
        };
        storage.initialize_schema()?;
        Ok(storage)
    }

    fn initialize_schema(&mut self) -> Result<(), StorageError> {
        MigrationRunner::new()
            .add(1, "create_scriptbots_schema", SCRIPTBOTS_SCHEMA_V1)
            .run(self.connection()?)?;
        Ok(())
    }

    fn connection(&self) -> Result<&Connection, StorageError> {
        self.conn.as_ref().ok_or(StorageError::Closed)
    }

    fn prepare_batch(payload: &PersistenceBatch) -> Result<StorageBuffer, StorageError> {
        let summary = &payload.summary;
        let tick = encode_u64("ticks.tick", summary.tick.0)?;
        let mut prepared = StorageBuffer::default();

        prepared.ticks.push(TickRow {
            tick,
            epoch: encode_u64("ticks.epoch", payload.epoch)?,
            closed: payload.closed,
            agent_count: checked_i64("ticks.agent_count", summary.agent_count)?,
            births: checked_i64("ticks.births", summary.births)?,
            deaths: checked_i64("ticks.deaths", summary.deaths)?,
            total_energy: f64::from(summary.total_energy),
            average_energy: f64::from(summary.average_energy),
            average_health: f64::from(summary.average_health),
        });

        for metric in &payload.metrics {
            prepared.metrics.push(MetricRow {
                tick,
                name: metric.name.to_string(),
                value: metric.value,
            });
        }

        for event in &payload.events {
            prepared.events.push(EventRow {
                tick,
                kind: match &event.kind {
                    PersistenceEventKind::Births => "births".to_string(),
                    PersistenceEventKind::Deaths => "deaths".to_string(),
                    PersistenceEventKind::Custom(name) => name.to_string(),
                },
                count: checked_i64("events.count", event.count)?,
            });
        }

        for agent in &payload.agents {
            prepared.agents.push(agent_row_from_snapshot(tick, agent)?);
        }

        for birth in &payload.births {
            prepared.births.push(birth_row_from_record(birth)?);
        }

        for death in &payload.deaths {
            prepared.deaths.push(death_row_from_record(death)?);
        }

        for (seq, event) in payload.replay_events.iter().enumerate() {
            prepared
                .replay_events
                .push(replay_row_from_event(event, tick, seq)?);
        }

        Ok(prepared)
    }

    fn enqueue_prepared(&mut self, prepared: StorageBuffer) -> Result<bool, StorageError> {
        if self.terminally_failed {
            return Err(StorageError::TerminallyFailed);
        }
        self.buffer.append(prepared);
        self.maybe_flush()
    }

    fn enqueue(&mut self, payload: &PersistenceBatch) -> Result<bool, StorageError> {
        let prepared = Self::prepare_batch(payload)?;
        self.enqueue_prepared(prepared)
    }

    /// Persist a simulation payload, buffering until thresholds are met.
    pub fn persist(&mut self, payload: &PersistenceBatch) -> Result<(), StorageError> {
        self.enqueue(payload).map(|_| ())
    }

    fn maybe_flush(&mut self) -> Result<bool, StorageError> {
        if self.buffer.ticks.len() >= self.tick_flush_threshold
            || self.buffer.metrics.len() >= self.metric_flush_threshold
            || self.buffer.events.len() >= self.event_flush_threshold
            || self.buffer.agents.len() >= self.agent_flush_threshold
            || self.buffer.births.len() >= self.birth_flush_threshold
            || self.buffer.deaths.len() >= self.death_flush_threshold
            || self.buffer.replay_events.len() >= self.replay_flush_threshold
        {
            self.flush()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn insert_ticks(tx: &Transaction<'_>, rows: &[TickRow]) -> Result<(), FrankenError> {
        if rows.is_empty() {
            return Ok(());
        }
        let sql = "insert or replace into ticks (
                tick, epoch, closed, agent_count, births, deaths,
                total_energy, average_energy, average_health
            ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)";
        for row in rows {
            tx.execute_with_params(
                sql,
                &[
                    row.tick.into(),
                    row.epoch.into(),
                    sqlite_bool(row.closed),
                    row.agent_count.into(),
                    row.births.into(),
                    row.deaths.into(),
                    row.total_energy.into(),
                    row.average_energy.into(),
                    row.average_health.into(),
                ],
            )?;
        }
        Ok(())
    }

    fn insert_metrics(tx: &Transaction<'_>, rows: &[MetricRow]) -> Result<(), FrankenError> {
        if rows.is_empty() {
            return Ok(());
        }
        let sql = "insert or replace into metrics (tick, name, value) values (?1, ?2, ?3)";
        for row in rows {
            tx.execute_with_params(
                sql,
                &[row.tick.into(), row.name.as_str().into(), row.value.into()],
            )?;
        }
        Ok(())
    }

    fn insert_events(tx: &Transaction<'_>, rows: &[EventRow]) -> Result<(), FrankenError> {
        if rows.is_empty() {
            return Ok(());
        }
        let sql = "insert or replace into events (tick, kind, count) values (?1, ?2, ?3)";
        for row in rows {
            tx.execute_with_params(
                sql,
                &[row.tick.into(), row.kind.as_str().into(), row.count.into()],
            )?;
        }
        Ok(())
    }

    fn insert_agents(tx: &Transaction<'_>, rows: &[AgentRow]) -> Result<(), FrankenError> {
        if rows.is_empty() {
            return Ok(());
        }
        for row in rows {
            tx.execute_with_params(
                Self::agent_insert_sql(),
                &[
                    row.tick.into(),
                    row.agent_id.into(),
                    row.generation.into(),
                    row.age.into(),
                    row.position_x.into(),
                    row.position_y.into(),
                    row.velocity_x.into(),
                    row.velocity_y.into(),
                    row.heading.into(),
                    row.health.into(),
                    row.energy.into(),
                    row.color_r.into(),
                    row.color_g.into(),
                    row.color_b.into(),
                    row.spike_length.into(),
                    sqlite_bool(row.boost),
                    row.herbivore_tendency.into(),
                    row.sound_multiplier.into(),
                    row.reproduction_counter.into(),
                    row.mutation_rate_primary.into(),
                    row.mutation_rate_secondary.into(),
                    row.trait_smell.into(),
                    row.trait_sound.into(),
                    row.trait_hearing.into(),
                    row.trait_eye.into(),
                    row.trait_blood.into(),
                    row.give_intent.into(),
                    row.brain_binding.as_str().into(),
                    sqlite_optional_i64(row.brain_key),
                    row.food_delta.into(),
                    sqlite_bool(row.spiked),
                    sqlite_bool(row.hybrid),
                    row.sound_output.into(),
                    sqlite_bool(row.spike_attacker),
                    sqlite_bool(row.spike_victim),
                    sqlite_bool(row.hit_carnivore),
                    sqlite_bool(row.hit_herbivore),
                    sqlite_bool(row.hit_by_carnivore),
                    sqlite_bool(row.hit_by_herbivore),
                ],
            )?;
        }
        Ok(())
    }

    fn agent_insert_sql() -> &'static str {
        static SQL: OnceLock<String> = OnceLock::new();
        SQL.get_or_init(|| {
            let columns = AGENT_COLUMNS.join(", ");
            let placeholders = (1..=AGENT_COLUMNS.len())
                .map(|index| format!("?{index}"))
                .collect::<Vec<String>>()
                .join(", ");
            format!("insert or replace into agents ({columns}) values ({placeholders})")
        })
    }

    fn insert_births(tx: &Transaction<'_>, rows: &[BirthRow]) -> Result<(), FrankenError> {
        if rows.is_empty() {
            return Ok(());
        }
        let sql = "insert or replace into births (
                tick, agent_id, parent_a, parent_b,
                brain_kind, brain_key, herbivore_tendency,
                generation, position_x, position_y, is_hybrid
            ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)";
        for row in rows {
            tx.execute_with_params(
                sql,
                &[
                    row.tick.into(),
                    row.agent_id.into(),
                    sqlite_optional_i64(row.parent_a),
                    sqlite_optional_i64(row.parent_b),
                    sqlite_optional_text(row.brain_kind.as_deref()),
                    sqlite_optional_i64(row.brain_key),
                    row.herbivore_tendency.into(),
                    row.generation.into(),
                    row.position_x.into(),
                    row.position_y.into(),
                    sqlite_bool(row.is_hybrid),
                ],
            )?;
        }
        Ok(())
    }

    fn insert_deaths(tx: &Transaction<'_>, rows: &[DeathRow]) -> Result<(), FrankenError> {
        if rows.is_empty() {
            return Ok(());
        }
        let sql = "insert or replace into deaths (
                tick, agent_id, age, generation,
                herbivore_tendency, brain_kind, brain_key,
                energy, food_balance_total, cause, was_hybrid,
                spike_attacker, spike_victim, hit_carnivore, hit_herbivore,
                hit_by_carnivore, hit_by_herbivore
            ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)";
        for row in rows {
            tx.execute_with_params(
                sql,
                &[
                    row.tick.into(),
                    row.agent_id.into(),
                    row.age.into(),
                    row.generation.into(),
                    row.herbivore_tendency.into(),
                    sqlite_optional_text(row.brain_kind.as_deref()),
                    sqlite_optional_i64(row.brain_key),
                    row.energy.into(),
                    row.food_balance_total.into(),
                    row.cause.as_str().into(),
                    sqlite_bool(row.was_hybrid),
                    sqlite_bool(row.spike_attacker),
                    sqlite_bool(row.spike_victim),
                    sqlite_bool(row.hit_carnivore),
                    sqlite_bool(row.hit_herbivore),
                    sqlite_bool(row.hit_by_carnivore),
                    sqlite_bool(row.hit_by_herbivore),
                ],
            )?;
        }
        Ok(())
    }

    fn insert_replay_events(
        tx: &Transaction<'_>,
        rows: &[ReplayEventRow],
    ) -> Result<(), FrankenError> {
        if rows.is_empty() {
            return Ok(());
        }
        let sql = "insert or replace into replay_events (
                tick, seq, agent_id, scope, event_type, payload
            ) values (?1, ?2, ?3, ?4, ?5, ?6)";
        for row in rows {
            tx.execute_with_params(
                sql,
                &[
                    row.tick.into(),
                    row.seq.into(),
                    sqlite_optional_i64(row.agent_id),
                    row.scope.as_str().into(),
                    row.event_type.as_str().into(),
                    row.payload.as_str().into(),
                ],
            )?;
        }
        Ok(())
    }

    fn flush_attempt(
        connection: &Connection,
        buffer: &StorageBuffer,
    ) -> Result<(), FlushAttemptError> {
        let mut tx = connection
            .transaction()
            .map_err(|source| FlushAttemptError {
                source,
                commit_state: FailureCommitState::RolledBack,
            })?;
        let transaction_result = (|| -> Result<(), FrankenError> {
            Self::insert_ticks(&tx, &buffer.ticks)?;
            Self::insert_metrics(&tx, &buffer.metrics)?;
            Self::insert_events(&tx, &buffer.events)?;
            Self::insert_agents(&tx, &buffer.agents)?;
            Self::insert_births(&tx, &buffer.births)?;
            Self::insert_deaths(&tx, &buffer.deaths)?;
            Self::insert_replay_events(&tx, &buffer.replay_events)?;
            tx.commit()?;
            Ok(())
        })();

        if let Err(source) = transaction_result {
            return match tx.rollback() {
                Ok(()) => Err(FlushAttemptError {
                    source,
                    commit_state: FailureCommitState::RolledBack,
                }),
                Err(rollback_error) => Err(FlushAttemptError {
                    source: FrankenError::Internal(format!(
                        "transaction failed ({source}); rollback also failed ({rollback_error})"
                    )),
                    commit_state: FailureCommitState::Indeterminate,
                }),
            };
        }

        Ok(())
    }

    /// Force flush buffered records, retrying only fully rolled-back transient transactions.
    pub fn flush(&mut self) -> Result<(), StorageError> {
        if self.terminally_failed {
            return Err(StorageError::TerminallyFailed);
        }
        if self.buffer.is_empty() {
            return Ok(());
        }

        let connection = self.connection()?;
        let mut attempt = 1_u8;
        loop {
            match Self::flush_attempt(connection, &self.buffer) {
                Ok(()) => {
                    info!(
                        path = %self.path,
                        attempt,
                        rows = self.buffer.ticks.len(),
                        "FrankenSQLite storage transaction committed"
                    );
                    self.buffer.clear();
                    return Ok(());
                }
                Err(error) if should_retry_transaction(&error, attempt) => {
                    warn!(
                        path = %self.path,
                        attempt,
                        transient = true,
                        commit_state = ?error.commit_state,
                        error = %error.source,
                        "retrying fully rolled-back FrankenSQLite transaction"
                    );
                    thread::sleep(Duration::from_millis(1_u64 << attempt));
                    attempt += 1;
                }
                Err(error) => {
                    let transient = error.source.is_transient();
                    self.terminally_failed = true;
                    return Err(StorageError::Transaction {
                        attempts: attempt,
                        transient,
                        commit_state: error.commit_state,
                        source: error.source,
                    });
                }
            }
        }
    }

    /// Compact storage after first durably flushing every buffered row.
    pub fn optimize(&mut self) -> Result<(), StorageError> {
        self.flush()?;
        self.connection()?.execute("VACUUM;")?;
        Ok(())
    }

    /// Flush, checkpoint, and explicitly close the FrankenSQLite connection.
    pub fn close(mut self) -> Result<(), StorageError> {
        self.flush()?;
        let connection = self.conn.take().ok_or(StorageError::Closed)?;
        connection.close()?;
        Ok(())
    }

    /// Dispose of a terminally failed worker without replaying its buffered transaction in Drop.
    fn abandon_after_error(mut self) {
        self.buffer.clear();
        if let Some(mut connection) = self.conn.take() {
            connection.close_best_effort_in_place();
        }
    }

    /// Return the maximum tick recorded in the `ticks` table, if any.
    pub fn max_tick(&mut self) -> Result<Option<u64>, StorageError> {
        self.flush()?;
        let row = self
            .connection()?
            .query_row("SELECT MAX(tick) FROM ticks")?;
        let value = decode::<Option<i64>>(&row, 0, "ticks.max_tick")?;
        value
            .map(|tick| checked_u64("ticks.max_tick", tick))
            .transpose()
    }

    /// Load all replay events ordered by tick/sequence and reconstruct their payloads.
    pub fn load_replay_events(&mut self) -> Result<Vec<PersistedReplayEvent>, StorageError> {
        self.flush()?;
        let rows = self.connection()?.query(
            "SELECT tick, seq, agent_id, scope, event_type, payload
             from replay_events
             ORDER BY tick ASC, seq ASC",
        )?;
        let mut events = Vec::with_capacity(rows.len());
        for row in rows {
            let replay_row = ReplayEventRow {
                tick: decode(&row, 0, "replay_events.tick")?,
                seq: decode(&row, 1, "replay_events.seq")?,
                agent_id: decode(&row, 2, "replay_events.agent_id")?,
                scope: decode(&row, 3, "replay_events.scope")?,
                event_type: decode(&row, 4, "replay_events.event_type")?,
                payload: decode(&row, 5, "replay_events.payload")?,
            };
            let event = replay_event_from_row(&replay_row)?;
            events.push(PersistedReplayEvent {
                tick: checked_u64("replay_events.tick", replay_row.tick)?,
                seq: checked_u64("replay_events.seq", replay_row.seq)?,
                event,
            });
        }
        Ok(events)
    }

    /// Return counts of replay events grouped by event type.
    pub fn replay_event_counts(&mut self) -> Result<Vec<ReplayEventCount>, StorageError> {
        self.flush()?;
        let rows = self.connection()?.query(
            "SELECT event_type, COUNT(*) AS total
             FROM replay_events
             GROUP BY event_type
             ORDER BY event_type",
        )?;
        let mut counts = Vec::with_capacity(rows.len());
        for row in rows {
            let count = decode::<i64>(&row, 1, "replay_events.count")?;
            counts.push(ReplayEventCount {
                event_type: decode(&row, 0, "replay_events.event_type")?,
                count: checked_u64("replay_events.count", count)?,
            });
        }
        Ok(counts)
    }

    /// Return agents ranked by average energy across all recorded ticks.
    pub fn top_predators(&mut self, limit: usize) -> Result<Vec<PredatorStats>, StorageError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        self.flush()?;
        let bound = checked_i64("top_predators.limit", limit)?;
        let rows = self.connection()?.query_with_params(
            "SELECT agent_id,
                    AVG(energy) AS avg_energy,
                    MAX(spike_length) AS max_spike_length,
                    MAX(tick) AS last_tick
             FROM agents
             GROUP BY agent_id
             ORDER BY avg_energy DESC
             LIMIT ?1",
            &[bound.into()],
        )?;
        let mut stats = Vec::with_capacity(limit.min(16));
        for row in rows {
            let agent_id = decode::<i64>(&row, 0, "agents.agent_id")?;
            stats.push(PredatorStats {
                agent_id: checked_u64("agents.agent_id", agent_id)?,
                avg_energy: decode(&row, 1, "agents.avg_energy")?,
                max_spike_length: decode(&row, 2, "agents.max_spike_length")?,
                last_tick: decode(&row, 3, "agents.last_tick")?,
            });
        }
        Ok(stats)
    }

    /// Fetch the latest recorded metrics (ordered by name) up to `limit`.
    pub fn latest_metrics(&mut self, limit: usize) -> Result<Vec<MetricReading>, StorageError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        self.flush()?;
        let row = self
            .connection()?
            .query_row("SELECT MAX(tick) FROM metrics")?;
        let latest_tick = decode::<Option<i64>>(&row, 0, "metrics.latest_tick")?;

        let Some(tick) = latest_tick else {
            return Ok(Vec::new());
        };

        let bound = checked_i64("latest_metrics.limit", limit)?;
        let rows = self.connection()?.query_with_params(
            "SELECT name, value
             FROM metrics
             WHERE tick = ?1
             ORDER BY name ASC
             LIMIT ?2",
            &[tick.into(), bound.into()],
        )?;
        let mut readings = Vec::with_capacity(rows.len());
        for row in rows {
            readings.push(MetricReading {
                tick,
                name: decode(&row, 0, "metrics.name")?,
                value: decode(&row, 1, "metrics.value")?,
            });
        }
        Ok(readings)
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if self.conn.is_none() {
            return;
        }
        if self.terminally_failed {
            self.buffer.clear();
        } else if let Err(err) = self.flush() {
            eprintln!("failed to flush persistence buffer on drop: {err}");
        }
        if let Some(mut connection) = self.conn.take() {
            connection.close_best_effort_in_place();
        }
    }
}

/// Aggregated predator statistics used for analytics.
#[derive(Debug, Clone)]
pub struct PredatorStats {
    pub agent_id: u64,
    pub avg_energy: f64,
    pub max_spike_length: f64,
    pub last_tick: i64,
}

#[derive(Debug)]
enum StorageCommand {
    Persist(Box<PreparedPersistenceBatch>),
    Flush {
        reply: xchan::Sender<Result<FlushReceipt, StorageWorkerError>>,
    },
    Shutdown {
        reply: xchan::Sender<Result<ShutdownReceipt, StorageWorkerError>>,
    },
    #[cfg(test)]
    PauseForAdmissionRace {
        entered: xchan::Sender<()>,
        release: xchan::Receiver<()>,
    },
    #[cfg(test)]
    DropMetricsTable {
        reply: xchan::Sender<Result<(), String>>,
    },
}

/// Persistence strength associated with an acknowledged commit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PersistenceGuarantee {
    /// Transaction committed in an in-memory database and will not survive close or process exit.
    #[default]
    CommittedVolatile,
    /// Transaction committed to a file-backed database under the configured durability policy.
    Durable,
}

/// Proof that every persistence command admitted before a flush has committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlushReceipt {
    pub committed_tick: Option<u64>,
    pub guarantee: PersistenceGuarantee,
    pub analytics_revision: u64,
}

/// Proof that the worker flushed, closed, and joined with an explicit persistence guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShutdownReceipt {
    pub committed_tick: Option<u64>,
    pub guarantee: PersistenceGuarantee,
    pub analytics_revision: u64,
}

#[derive(Default)]
struct WorkerState {
    admitted_tick: Option<u64>,
    committed_tick: Option<u64>,
    guarantee: PersistenceGuarantee,
    pending_analytics: Option<PendingAnalytics>,
}

#[derive(Clone, Copy)]
struct StorageThresholds {
    tick: usize,
    agent: usize,
    event: usize,
    metric: usize,
}

#[derive(Debug)]
struct AdmissionState {
    open: bool,
}

/// Cloneable persistence sink that never owns or transports a database connection.
#[derive(Clone)]
pub struct StorageSink {
    tx: xchan::Sender<StorageCommand>,
    analytics: AnalyticsSnapshotProvider,
    admission: Arc<Mutex<AdmissionState>>,
    path: Arc<str>,
}

impl StorageSink {
    /// Admit a persistence batch to the bounded worker queue.
    pub fn submit(&self, payload: &PersistenceBatch) -> Result<(), StorageError> {
        let prepared = PreparedPersistenceBatch::from_batch(payload).inspect_err(|error| {
            let worker_error = StorageWorkerError::Internal {
                operation: StorageOperation::Admit,
                path: self.path.to_string(),
                tick: Some(payload.summary.tick.0),
                commit_state: FailureCommitState::NotAdmitted,
                detail: error.to_string(),
            };
            self.analytics.publish_worker_error(&worker_error, false);
        })?;
        let admission = self.admission.lock().map_err(|error| {
            let worker_error = StorageWorkerError::Internal {
                operation: StorageOperation::Admit,
                path: self.path.to_string(),
                tick: Some(payload.summary.tick.0),
                commit_state: FailureCommitState::NotAdmitted,
                detail: format!("storage admission gate is poisoned: {error}"),
            };
            self.analytics.publish_worker_error(&worker_error, true);
            StorageError::Worker(worker_error)
        })?;
        if !admission.open {
            let worker_error = StorageWorkerError::Channel {
                operation: StorageOperation::Admit,
                path: self.path.to_string(),
                tick: Some(payload.summary.tick.0),
                commit_state: FailureCommitState::NotAdmitted,
                detail: "storage pipeline is closing or closed".to_owned(),
            };
            self.analytics.publish_worker_error(&worker_error, true);
            return Err(StorageError::Worker(worker_error));
        }

        let send_result = self.tx.send(StorageCommand::Persist(Box::new(prepared)));
        drop(admission);
        send_result.map_err(|error| {
            let worker_error = StorageWorkerError::Channel {
                operation: StorageOperation::Admit,
                path: self.path.to_string(),
                tick: Some(payload.summary.tick.0),
                commit_state: FailureCommitState::NotAdmitted,
                detail: error.to_string(),
            };
            self.analytics.publish_worker_error(&worker_error, true);
            StorageError::Worker(worker_error)
        })
    }
}

impl WorldPersistence for StorageSink {
    fn on_tick(&mut self, payload: &PersistenceBatch) -> Result<(), PersistenceAdmissionError> {
        self.submit(payload).map_err(|error| {
            PersistenceAdmissionError::new(payload.summary.tick.0, error.to_string())
        })
    }
}

/// Host-owned controller for flush receipts, shutdown acknowledgement, and worker join.
pub struct StoragePipeline {
    sink: StorageSink,
    handle: Option<thread::JoinHandle<Option<StorageWorkerError>>>,
}

impl StoragePipeline {
    /// Atomically reserve and create a file-backed asynchronous pipeline.
    pub fn create_new_file(path: &str) -> Result<Self, StorageError> {
        Self::create_new_file_with_thresholds(
            path,
            DEFAULT_TICK_BUFFER,
            DEFAULT_AGENT_BUFFER,
            DEFAULT_EVENT_BUFFER,
            DEFAULT_METRIC_BUFFER,
        )
    }

    /// Atomically reserve a file-backed pipeline with explicit thresholds.
    pub fn create_new_file_with_thresholds(
        path: &str,
        tick: usize,
        agent: usize,
        event: usize,
        metric: usize,
    ) -> Result<Self, StorageError> {
        Self::with_target(reserve_new_file(path)?, tick, agent, event, metric)
    }

    /// Create an isolated volatile pipeline with default thresholds.
    pub fn memory() -> Result<Self, StorageError> {
        Self::memory_with_thresholds(
            DEFAULT_TICK_BUFFER,
            DEFAULT_AGENT_BUFFER,
            DEFAULT_EVENT_BUFFER,
            DEFAULT_METRIC_BUFFER,
        )
    }

    /// Create an isolated volatile pipeline with explicit thresholds.
    pub fn memory_with_thresholds(
        tick: usize,
        agent: usize,
        event: usize,
        metric: usize,
    ) -> Result<Self, StorageError> {
        Self::with_target(StorageTarget::Memory, tick, agent, event, metric)
    }

    fn with_target(
        target: StorageTarget,
        tick: usize,
        agent: usize,
        event: usize,
        metric: usize,
    ) -> Result<Self, StorageError> {
        let (tx, rx) = xchan::bounded::<StorageCommand>(DEFAULT_COMMAND_CAPACITY);
        let (startup_tx, startup_rx) = xchan::bounded::<Result<(), StorageWorkerError>>(1);
        let analytics = AnalyticsSnapshotProvider::empty();
        let admission = Arc::new(Mutex::new(AdmissionState { open: true }));
        let storage_path: Arc<str> = Arc::from(target.path());
        let worker_analytics = analytics.clone();
        let thresholds = StorageThresholds {
            tick,
            agent,
            event,
            metric,
        };
        let handle = thread::Builder::new()
            .name("scriptbots-storage-worker".into())
            .spawn(move || storage_worker(target, thresholds, rx, startup_tx, worker_analytics))
            .map_err(|err| {
                StorageError::Worker(StorageWorkerError::Internal {
                    operation: StorageOperation::Startup,
                    path: storage_path.to_string(),
                    tick: None,
                    commit_state: FailureCommitState::NotAdmitted,
                    detail: format!("failed to spawn storage worker thread: {err}"),
                })
            })?;

        match startup_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sink: StorageSink {
                    tx,
                    analytics,
                    admission,
                    path: storage_path.clone(),
                },
                handle: Some(handle),
            }),
            Ok(Err(error)) => match handle.join() {
                Ok(Some(terminal_error)) => Err(StorageError::Worker(terminal_error)),
                Ok(None) => Err(StorageError::Worker(error)),
                Err(panic) => Err(StorageError::Worker(StorageWorkerError::Internal {
                    operation: StorageOperation::Join,
                    path: storage_path.to_string(),
                    tick: None,
                    commit_state: FailureCommitState::Indeterminate,
                    detail: format!("storage worker panicked during startup: {panic:?}"),
                })),
            },
            Err(error) => match handle.join() {
                Ok(Some(terminal_error)) => Err(StorageError::Worker(terminal_error)),
                Ok(None) => Err(StorageError::Worker(StorageWorkerError::Channel {
                    operation: StorageOperation::Startup,
                    path: storage_path.to_string(),
                    tick: None,
                    commit_state: FailureCommitState::NotAdmitted,
                    detail: format!(
                        "storage worker exited before startup acknowledgement: {error}"
                    ),
                })),
                Err(panic) => Err(StorageError::Worker(StorageWorkerError::Internal {
                    operation: StorageOperation::Join,
                    path: storage_path.to_string(),
                    tick: None,
                    commit_state: FailureCommitState::Indeterminate,
                    detail: format!("storage worker panicked during startup: {panic:?}"),
                })),
            },
        }
    }

    /// Return a lock-free read handle containing only committed analytics data.
    #[must_use]
    pub fn analytics_provider(&self) -> AnalyticsSnapshotProvider {
        self.sink.analytics.clone()
    }

    /// Return a clonable sink for `WorldState` while retaining host control of the worker.
    #[must_use]
    pub fn sink(&self) -> StorageSink {
        self.sink.clone()
    }

    /// Admit a persistence batch to the bounded worker queue.
    pub fn submit(&self, payload: &PersistenceBatch) -> Result<(), StorageError> {
        self.sink.submit(payload)
    }

    #[cfg(test)]
    fn drop_metrics_table_for_test(&self) -> Result<(), StorageError> {
        let (reply_tx, reply_rx) = xchan::bounded(1);
        let admission = self
            .sink
            .admission
            .lock()
            .map_err(|error| StorageError::InvalidData {
                context: "test.drop_metrics_table",
                reason: format!("storage admission gate is poisoned: {error}"),
            })?;
        if !admission.open {
            return Err(StorageError::Closed);
        }
        self.sink
            .tx
            .send(StorageCommand::DropMetricsTable { reply: reply_tx })
            .map_err(|error| StorageError::InvalidData {
                context: "test.drop_metrics_table",
                reason: error.to_string(),
            })?;
        drop(admission);
        reply_rx
            .recv()
            .map_err(|error| StorageError::InvalidData {
                context: "test.drop_metrics_table",
                reason: error.to_string(),
            })?
            .map_err(|reason| StorageError::InvalidData {
                context: "test.drop_metrics_table",
                reason,
            })
    }

    /// Flush all previously admitted batches and wait for a durability receipt.
    pub fn flush_and_wait(&self) -> Result<FlushReceipt, StorageError> {
        let (reply_tx, reply_rx) = xchan::bounded(1);
        let admission = self.sink.admission.lock().map_err(|error| {
            StorageError::Worker(StorageWorkerError::Internal {
                operation: StorageOperation::Flush,
                path: self.sink.path.to_string(),
                tick: None,
                commit_state: FailureCommitState::Indeterminate,
                detail: format!("storage admission gate is poisoned: {error}"),
            })
        })?;
        if !admission.open {
            return Err(StorageError::Worker(StorageWorkerError::Channel {
                operation: StorageOperation::Flush,
                path: self.sink.path.to_string(),
                tick: None,
                commit_state: FailureCommitState::Indeterminate,
                detail: "storage pipeline is closing or closed".to_owned(),
            }));
        }
        let send_result = self.sink.tx.send(StorageCommand::Flush { reply: reply_tx });
        drop(admission);
        send_result.map_err(|error| {
            StorageError::Worker(StorageWorkerError::Channel {
                operation: StorageOperation::Flush,
                path: self.sink.path.to_string(),
                tick: None,
                commit_state: FailureCommitState::Indeterminate,
                detail: format!("failed to request storage flush: {error}"),
            })
        })?;
        reply_rx
            .recv()
            .map_err(|error| {
                StorageError::Worker(StorageWorkerError::Channel {
                    operation: StorageOperation::Flush,
                    path: self.sink.path.to_string(),
                    tick: None,
                    commit_state: FailureCommitState::Indeterminate,
                    detail: format!("storage worker exited before flush acknowledgement: {error}"),
                })
            })?
            .map_err(StorageError::Worker)
    }

    /// Flush, close, and join the worker, returning an explicit shutdown receipt.
    pub fn shutdown(&mut self) -> Result<ShutdownReceipt, StorageError> {
        let Some(handle) = self.handle.take() else {
            return Err(StorageError::Worker(StorageWorkerError::Internal {
                operation: StorageOperation::Shutdown,
                path: self.sink.path.to_string(),
                tick: None,
                commit_state: FailureCommitState::Committed,
                detail: "storage worker has already been shut down".to_owned(),
            }));
        };

        let (reply_tx, reply_rx) = xchan::bounded(1);
        let mut admission = match self.sink.admission.lock() {
            Ok(admission) => admission,
            Err(poisoned) => {
                warn!("recovering poisoned storage admission gate during shutdown");
                poisoned.into_inner()
            }
        };
        admission.open = false;
        let send_result = self
            .sink
            .tx
            .send(StorageCommand::Shutdown { reply: reply_tx })
            .map_err(|error| {
                StorageError::Worker(StorageWorkerError::Channel {
                    operation: StorageOperation::Shutdown,
                    path: self.sink.path.to_string(),
                    tick: None,
                    commit_state: FailureCommitState::Indeterminate,
                    detail: format!("failed to request storage shutdown: {error}"),
                })
            });
        drop(admission);
        // Join before waiting for the reply. A terminal flush can acknowledge
        // its requester and then exit while this shutdown command is being
        // admitted. In that race the command remains buffered, and its embedded
        // reply sender stays alive until the command channel itself is dropped;
        // waiting for the reply first would therefore deadlock. The bounded
        // reply channel lets a healthy worker acknowledge shutdown before it
        // exits without waiting for this receiver.
        let joined = handle.join().map_err(|panic| {
            StorageError::Worker(StorageWorkerError::Internal {
                operation: StorageOperation::Join,
                path: self.sink.path.to_string(),
                tick: None,
                commit_state: FailureCommitState::Indeterminate,
                detail: format!("storage worker thread panicked: {panic:?}"),
            })
        });
        match joined {
            Err(error) => Err(error),
            Ok(Some(terminal_error)) => Err(StorageError::Worker(terminal_error)),
            Ok(None) => send_result.and_then(|()| {
                reply_rx
                    .recv()
                    .map_err(|error| {
                        StorageError::Worker(StorageWorkerError::Channel {
                            operation: StorageOperation::Shutdown,
                            path: self.sink.path.to_string(),
                            tick: None,
                            commit_state: FailureCommitState::Indeterminate,
                            detail: format!(
                                "storage worker exited before shutdown acknowledgement: {error}"
                            ),
                        })
                    })?
                    .map_err(StorageError::Worker)
            }),
        }
    }
}

impl Drop for StoragePipeline {
    fn drop(&mut self) {
        if self.handle.is_some()
            && let Err(error) = self.shutdown()
        {
            eprintln!("failed to shut down storage worker cleanly: {error}");
        }
    }
}

fn storage_worker(
    target: StorageTarget,
    thresholds: StorageThresholds,
    rx: xchan::Receiver<StorageCommand>,
    startup: xchan::Sender<Result<(), StorageWorkerError>>,
    analytics: AnalyticsSnapshotProvider,
) -> Option<StorageWorkerError> {
    let path = target.path().to_owned();
    let guarantee = target.guarantee();
    let mut storage = match Storage::with_target(
        target,
        thresholds.tick,
        thresholds.agent,
        thresholds.event,
        thresholds.metric,
    ) {
        Ok(storage) => storage,
        Err(error) => {
            let worker_error = worker_error_from_storage(
                StorageOperation::Startup,
                &path,
                None,
                FailureCommitState::NotAdmitted,
                error,
            );
            analytics.publish_worker_error(&worker_error, true);
            let _ = startup.send(Err(worker_error));
            return None;
        }
    };
    if startup.send(Ok(())).is_err() {
        let _ = storage.close();
        let error = StorageWorkerError::Channel {
            operation: StorageOperation::Startup,
            path: path.clone(),
            tick: None,
            commit_state: FailureCommitState::NotAdmitted,
            detail: "startup receiver disconnected".to_owned(),
        };
        analytics.publish_worker_error(&error, true);
        return Some(error);
    }

    let mut state = WorkerState {
        guarantee,
        ..WorkerState::default()
    };
    while let Ok(command) = rx.recv() {
        match command {
            StorageCommand::Persist(batch) => {
                let PreparedPersistenceBatch {
                    tick,
                    storage: prepared,
                    analytics: pending,
                } = *batch;
                match storage.enqueue_prepared(prepared) {
                    Ok(flushed) => {
                        state.admitted_tick = Some(tick);
                        state.pending_analytics = Some(pending);
                        if flushed {
                            publish_committed_state(&mut state, &analytics);
                        }
                    }
                    Err(error) => {
                        let worker_error = worker_error_from_storage(
                            StorageOperation::Persist,
                            &path,
                            Some(tick),
                            FailureCommitState::Indeterminate,
                            error,
                        );
                        analytics.publish_worker_error(&worker_error, true);
                        storage.abandon_after_error();
                        return Some(worker_error);
                    }
                }
            }
            StorageCommand::Flush { reply } => {
                match flush_worker_storage(&mut storage, &mut state, &analytics) {
                    Ok(receipt) => {
                        let _ = reply.send(Ok(receipt));
                    }
                    Err(error) => {
                        analytics.publish_worker_error(&error, true);
                        storage.abandon_after_error();
                        // Preserve the structured root cause for the worker join
                        // (StoragePipeline::shutdown prefers it) while still
                        // acknowledging the flush requester with the original.
                        let worker_error = duplicate_worker_error(&error);
                        let _ = reply.send(Err(error));
                        return Some(worker_error);
                    }
                }
            }
            StorageCommand::Shutdown { reply } => {
                let result = shutdown_worker_storage(storage, &mut state, &analytics);
                if let Err(error) = &result {
                    analytics.publish_worker_error(error, true);
                }
                let _ = reply.send(result);
                return None;
            }
            #[cfg(test)]
            StorageCommand::PauseForAdmissionRace { entered, release } => {
                let _ = entered.send(());
                let _ = release.recv();
            }
            #[cfg(test)]
            StorageCommand::DropMetricsTable { reply } => {
                let result = storage
                    .connection()
                    .and_then(|connection| {
                        connection
                            .execute("DROP TABLE metrics")
                            .map(|_| ())
                            .map_err(StorageError::from)
                    })
                    .map_err(|error| error.to_string());
                let _ = reply.send(result);
            }
        }
    }

    match shutdown_worker_storage(storage, &mut state, &analytics) {
        Ok(_) => None,
        Err(error) => {
            analytics.publish_worker_error(&error, true);
            Some(error)
        }
    }
}

fn publish_committed_state(state: &mut WorkerState, analytics: &AnalyticsSnapshotProvider) {
    state.committed_tick = state.admitted_tick;
    if let Some(pending) = state.pending_analytics.take() {
        analytics.publish_committed(pending);
    }
}

fn flush_worker_storage(
    storage: &mut Storage,
    state: &mut WorkerState,
    analytics: &AnalyticsSnapshotProvider,
) -> Result<FlushReceipt, StorageWorkerError> {
    storage.flush().map_err(|error| {
        worker_error_from_storage(
            StorageOperation::Flush,
            &storage.path,
            state.admitted_tick,
            FailureCommitState::Indeterminate,
            error,
        )
    })?;
    publish_committed_state(state, analytics);
    Ok(FlushReceipt {
        committed_tick: state.committed_tick,
        guarantee: state.guarantee,
        analytics_revision: analytics.snapshot().revision,
    })
}

fn shutdown_worker_storage(
    mut storage: Storage,
    state: &mut WorkerState,
    analytics: &AnalyticsSnapshotProvider,
) -> Result<ShutdownReceipt, StorageWorkerError> {
    if let Err(error) = flush_worker_storage(&mut storage, state, analytics) {
        storage.abandon_after_error();
        return Err(error);
    }
    let path = storage.path.clone();
    storage.close().map_err(|error| {
        worker_error_from_storage(
            StorageOperation::Close,
            &path,
            state.committed_tick,
            FailureCommitState::Committed,
            error,
        )
    })?;
    analytics.publish_stopped();
    Ok(ShutdownReceipt {
        committed_tick: state.committed_tick,
        guarantee: state.guarantee,
        analytics_revision: analytics.snapshot().revision,
    })
}

fn brain_binding_to_string(binding: &BrainBinding) -> String {
    binding.describe().into_owned()
}

fn agent_row_from_snapshot(tick: i64, agent: &AgentState) -> Result<AgentRow, StorageError> {
    let id = encode_u64("agents.agent_id", agent.id.data().as_ffi())?;
    let data = &agent.data;
    let runtime = &agent.runtime;
    Ok(AgentRow {
        tick,
        agent_id: id,
        generation: i64::from(data.generation.0),
        age: i64::from(data.age),
        position_x: f64::from(data.position.x),
        position_y: f64::from(data.position.y),
        velocity_x: f64::from(data.velocity.vx),
        velocity_y: f64::from(data.velocity.vy),
        heading: f64::from(data.heading),
        health: f64::from(data.health),
        energy: f64::from(runtime.energy),
        color_r: f64::from(data.color[0]),
        color_g: f64::from(data.color[1]),
        color_b: f64::from(data.color[2]),
        spike_length: f64::from(data.spike_length),
        boost: data.boost,
        herbivore_tendency: f64::from(runtime.herbivore_tendency),
        sound_multiplier: f64::from(runtime.sound_multiplier),
        reproduction_counter: f64::from(runtime.reproduction_counter),
        mutation_rate_primary: f64::from(runtime.mutation_rates.primary),
        mutation_rate_secondary: f64::from(runtime.mutation_rates.secondary),
        trait_smell: f64::from(runtime.trait_modifiers.smell),
        trait_sound: f64::from(runtime.trait_modifiers.sound),
        trait_hearing: f64::from(runtime.trait_modifiers.hearing),
        trait_eye: f64::from(runtime.trait_modifiers.eye),
        trait_blood: f64::from(runtime.trait_modifiers.blood),
        give_intent: f64::from(runtime.give_intent),
        brain_binding: brain_binding_to_string(&runtime.brain),
        brain_key: runtime
            .brain
            .registry_key()
            .map(|key| encode_u64("agents.brain_key", key))
            .transpose()?,
        food_delta: f64::from(runtime.food_delta),
        spiked: runtime.spiked,
        hybrid: runtime.hybrid,
        sound_output: f64::from(runtime.sound_output),
        spike_attacker: runtime.combat.spike_attacker,
        spike_victim: runtime.combat.spike_victim,
        hit_carnivore: runtime.combat.hit_carnivore,
        hit_herbivore: runtime.combat.hit_herbivore,
        hit_by_carnivore: runtime.combat.was_spiked_by_carnivore,
        hit_by_herbivore: runtime.combat.was_spiked_by_herbivore,
    })
}

fn optional_agent_id(
    context: &'static str,
    id: Option<AgentId>,
) -> Result<Option<i64>, StorageError> {
    id.map(|agent_id| encode_u64(context, agent_id.data().as_ffi()))
        .transpose()
}

fn phase_label(phase: ReplayAgentPhase) -> &'static str {
    match phase {
        ReplayAgentPhase::Movement => "movement",
        ReplayAgentPhase::Reproduction => "reproduction",
        ReplayAgentPhase::Mutation => "mutation",
        ReplayAgentPhase::Spawn => "spawn",
        ReplayAgentPhase::Selection => "selection",
        ReplayAgentPhase::Misc => "misc",
    }
}

fn scope_label(scope: ReplayRngScope) -> String {
    match scope {
        ReplayRngScope::World => "world".to_string(),
        ReplayRngScope::Agent { phase, .. } => {
            format!("agent:{}", phase_label(phase))
        }
    }
}

fn replay_row_from_event(
    event: &ReplayEvent,
    tick: i64,
    seq: usize,
) -> Result<ReplayEventRow, StorageError> {
    let invalid_non_finite = |context: &'static str| StorageError::InvalidData {
        context,
        reason: format!("non-finite replay value at tick {tick}, seq {seq}"),
    };
    let (scope, event_type, payload_value): (String, String, Value) = match &event.kind {
        ReplayEventKind::BrainOutputs { outputs } => {
            if outputs.iter().any(|value| !value.is_finite()) {
                return Err(invalid_non_finite("replay_events.brain_outputs"));
            }
            (
                if event.agent_id.is_some() {
                    "agent:brain"
                } else {
                    "world:brain"
                }
                .to_string(),
                "brain_outputs".to_string(),
                json!({ "outputs": outputs }),
            )
        }
        ReplayEventKind::Action {
            left_wheel,
            right_wheel,
            boost,
            spike_target,
            sound_level,
            give_intent,
        } => {
            if [*left_wheel, *right_wheel, *sound_level, *give_intent]
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(invalid_non_finite("replay_events.action"));
            }
            let spike_target = spike_target
                .map(|agent_id| {
                    encode_u64(
                        "replay_events.action.spike_target",
                        agent_id.data().as_ffi(),
                    )
                })
                .transpose()?;
            (
                if event.agent_id.is_some() {
                    "agent:action"
                } else {
                    "world:action"
                }
                .to_string(),
                "action".to_string(),
                json!({
                    "left_wheel": left_wheel,
                    "right_wheel": right_wheel,
                    "boost": boost,
                    "spike_target": spike_target,
                    "sound_level": sound_level,
                    "give_intent": give_intent,
                }),
            )
        }
        ReplayEventKind::RngSample {
            scope,
            range_min,
            range_max,
            value,
        } => {
            if [*range_min, *range_max, *value]
                .iter()
                .any(|sample| !sample.is_finite())
            {
                return Err(invalid_non_finite("replay_events.rng_sample"));
            }
            let scope_agent_id = match scope {
                ReplayRngScope::World => None,
                ReplayRngScope::Agent { agent_id, .. } => Some(encode_u64(
                    "replay_events.rng_sample.scope_agent_id",
                    agent_id.data().as_ffi(),
                )?),
            };
            (
                scope_label(*scope),
                "rng_sample".to_string(),
                json!({
                    "scope_agent_id": scope_agent_id,
                    "range_min": range_min,
                    "range_max": range_max,
                    "value": value,
                }),
            )
        }
    };

    Ok(ReplayEventRow {
        tick,
        seq: checked_i64("replay_events.seq", seq)?,
        agent_id: optional_agent_id("replay_events.agent_id", event.agent_id)?,
        scope,
        event_type,
        payload: payload_value.to_string(),
    })
}

fn decode_agent_id(raw: Option<i64>, tick: i64, seq: i64) -> Result<Option<AgentId>, StorageError> {
    match raw {
        Some(value) if value < 0 => Err(StorageError::ReplayParse {
            tick,
            seq,
            reason: format!("negative agent id {value}"),
        }),
        Some(value) => {
            let key = KeyData::from_ffi(value as u64);
            Ok(Some(AgentId::from(key)))
        }
        None => Ok(None),
    }
}

fn agent_id_from_u64(value: u64, tick: i64, seq: i64) -> Result<AgentId, StorageError> {
    if value > i64::MAX as u64 {
        return Err(StorageError::ReplayParse {
            tick,
            seq,
            reason: format!("agent id {value} exceeds supported range"),
        });
    }
    Ok(AgentId::from(KeyData::from_ffi(value)))
}

fn parse_payload<T>(row: &ReplayEventRow) -> Result<T, StorageError>
where
    T: DeserializeOwned,
{
    serde_json::from_str(&row.payload).map_err(|err| StorageError::ReplayParse {
        tick: row.tick,
        seq: row.seq,
        reason: format!("failed to deserialize payload: {err}"),
    })
}

fn parse_agent_phase(label: &str) -> Option<ReplayAgentPhase> {
    match label {
        "movement" => Some(ReplayAgentPhase::Movement),
        "reproduction" => Some(ReplayAgentPhase::Reproduction),
        "mutation" => Some(ReplayAgentPhase::Mutation),
        "spawn" => Some(ReplayAgentPhase::Spawn),
        "selection" => Some(ReplayAgentPhase::Selection),
        "misc" => Some(ReplayAgentPhase::Misc),
        _ => None,
    }
}

fn parse_rng_scope(
    scope: &str,
    agent_id: Option<AgentId>,
    row: &ReplayEventRow,
) -> Result<ReplayRngScope, StorageError> {
    if scope == "world" {
        return Ok(ReplayRngScope::World);
    }

    if let Some(phase_label) = scope.strip_prefix("agent:") {
        let agent_id = agent_id.ok_or_else(|| StorageError::ReplayParse {
            tick: row.tick,
            seq: row.seq,
            reason: "agent-scoped RNG sample missing agent_id".to_string(),
        })?;
        let phase = parse_agent_phase(phase_label).ok_or_else(|| StorageError::ReplayParse {
            tick: row.tick,
            seq: row.seq,
            reason: format!("unknown agent phase '{phase_label}'"),
        })?;
        return Ok(ReplayRngScope::Agent { agent_id, phase });
    }

    Err(StorageError::ReplayParse {
        tick: row.tick,
        seq: row.seq,
        reason: format!("unknown replay scope '{scope}'"),
    })
}

#[derive(Debug, Deserialize)]
struct BrainOutputsPayload {
    outputs: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ActionPayload {
    left_wheel: f32,
    right_wheel: f32,
    boost: bool,
    spike_target: Option<u64>,
    sound_level: f32,
    give_intent: f32,
}

#[derive(Debug, Deserialize)]
struct RngSamplePayload {
    scope_agent_id: Option<u64>,
    range_min: f32,
    range_max: f32,
    value: f32,
}

fn replay_event_from_row(row: &ReplayEventRow) -> Result<ReplayEvent, StorageError> {
    let agent_id = decode_agent_id(row.agent_id, row.tick, row.seq)?;
    let kind = match row.event_type.as_str() {
        "brain_outputs" => {
            if row.scope.starts_with("agent:") && agent_id.is_none() {
                return Err(StorageError::ReplayParse {
                    tick: row.tick,
                    seq: row.seq,
                    reason: "brain outputs missing agent_id".to_string(),
                });
            }
            let payload: BrainOutputsPayload = parse_payload(row)?;
            ReplayEventKind::BrainOutputs {
                outputs: payload.outputs,
            }
        }
        "action" => {
            if row.scope.starts_with("agent:") && agent_id.is_none() {
                return Err(StorageError::ReplayParse {
                    tick: row.tick,
                    seq: row.seq,
                    reason: "action event missing agent_id".to_string(),
                });
            }
            let payload: ActionPayload = parse_payload(row)?;
            let spike_target = match payload.spike_target {
                Some(raw) => Some(agent_id_from_u64(raw, row.tick, row.seq)?),
                None => None,
            };
            ReplayEventKind::Action {
                left_wheel: payload.left_wheel,
                right_wheel: payload.right_wheel,
                boost: payload.boost,
                spike_target,
                sound_level: payload.sound_level,
                give_intent: payload.give_intent,
            }
        }
        "rng_sample" => {
            let payload: RngSamplePayload = parse_payload(row)?;
            let scope_agent_id = payload
                .scope_agent_id
                .map(|raw| agent_id_from_u64(raw, row.tick, row.seq))
                .transpose()?;
            let scope = parse_rng_scope(&row.scope, scope_agent_id, row)?;
            ReplayEventKind::RngSample {
                scope,
                range_min: payload.range_min,
                range_max: payload.range_max,
                value: payload.value,
            }
        }
        other => {
            return Err(StorageError::ReplayParse {
                tick: row.tick,
                seq: row.seq,
                reason: format!("unknown event type '{other}'"),
            });
        }
    };

    Ok(ReplayEvent { agent_id, kind })
}

fn birth_row_from_record(record: &BirthRecord) -> Result<BirthRow, StorageError> {
    Ok(BirthRow {
        tick: encode_u64("births.tick", record.tick.0)?,
        agent_id: encode_u64("births.agent_id", record.agent_id.data().as_ffi())?,
        parent_a: optional_agent_id("births.parent_a", record.parent_a)?,
        parent_b: optional_agent_id("births.parent_b", record.parent_b)?,
        brain_kind: record.brain_kind.clone(),
        brain_key: record
            .brain_key
            .map(|key| encode_u64("births.brain_key", key))
            .transpose()?,
        herbivore_tendency: f64::from(record.herbivore_tendency),
        generation: i64::from(record.generation.0),
        position_x: f64::from(record.position.x),
        position_y: f64::from(record.position.y),
        is_hybrid: record.is_hybrid,
    })
}

fn death_cause_to_string(cause: DeathCause) -> &'static str {
    match cause {
        DeathCause::CombatCarnivore => "combat_carnivore",
        DeathCause::CombatHerbivore => "combat_herbivore",
        DeathCause::Starvation => "starvation",
        DeathCause::Aging => "aging",
        DeathCause::Unknown => "unknown",
    }
}

fn death_row_from_record(record: &DeathRecord) -> Result<DeathRow, StorageError> {
    Ok(DeathRow {
        tick: encode_u64("deaths.tick", record.tick.0)?,
        agent_id: encode_u64("deaths.agent_id", record.agent_id.data().as_ffi())?,
        age: i64::from(record.age),
        generation: i64::from(record.generation.0),
        herbivore_tendency: f64::from(record.herbivore_tendency),
        brain_kind: record.brain_kind.clone(),
        brain_key: record
            .brain_key
            .map(|key| encode_u64("deaths.brain_key", key))
            .transpose()?,
        energy: f64::from(record.energy),
        food_balance_total: f64::from(record.food_balance_total),
        cause: death_cause_to_string(record.cause).to_string(),
        was_hybrid: record.was_hybrid,
        spike_attacker: record.combat_flags.spike_attacker,
        spike_victim: record.combat_flags.spike_victim,
        hit_carnivore: record.combat_flags.hit_carnivore,
        hit_herbivore: record.combat_flags.hit_herbivore,
        hit_by_carnivore: record.combat_flags.was_spiked_by_carnivore,
        hit_by_herbivore: record.combat_flags.was_spiked_by_herbivore,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{
        AgentData, AgentRuntime, AgentState, MetricSample, PersistenceBatch, PersistenceEvent,
        PersistenceEventKind, Position, Tick, TickSummary,
    };
    use std::{
        fs,
        path::PathBuf,
        sync::TryLockError,
        time::{Instant, SystemTime, UNIX_EPOCH},
    };

    fn temp_db_path(prefix: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        path.push(format!(
            "{}-{}-{}.sqlite",
            prefix,
            std::process::id(),
            timestamp
        ));
        path
    }

    fn sample_agent(energy: f32) -> AgentState {
        let data = AgentData {
            position: Position::new(12.0, 34.0),
            health: energy,
            ..AgentData::default()
        };

        let runtime = AgentRuntime {
            energy,
            ..AgentRuntime::default()
        };

        AgentState {
            id: scriptbots_core::AgentId::default(),
            data,
            runtime,
        }
    }

    fn sample_batch(tick: u64, energy: f32) -> PersistenceBatch {
        PersistenceBatch {
            summary: TickSummary {
                tick: Tick(tick),
                agent_count: 1,
                births: 1,
                deaths: 0,
                total_energy: energy,
                average_energy: energy,
                average_health: 1.0,
                max_age: 0,
                spike_hits: 0,
            },
            epoch: 3,
            closed: false,
            metrics: vec![
                MetricSample::from_f32("total_energy", energy),
                MetricSample::from_f32("average_energy", energy),
                MetricSample::from_f32("average_health", 1.0),
            ],
            events: vec![PersistenceEvent::new(PersistenceEventKind::Births, 1)],
            agents: vec![sample_agent(energy)],
            births: Vec::new(),
            deaths: Vec::new(),
            replay_events: Vec::new(),
        }
    }

    fn assert_invalid_data_context<T: std::fmt::Debug>(
        result: Result<T, StorageError>,
        expected: &'static str,
    ) {
        let matches_expected = matches!(
            &result,
            Err(StorageError::InvalidData { context, .. }) if *context == expected
        );
        assert!(
            matches_expected,
            "expected InvalidData for {expected}, got {result:?}"
        );
    }

    #[test]
    fn persist_batch_writes_all_tables() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-persist");
        let path_string = path.to_string_lossy().to_string();
        let mut storage = Storage::create_new_file_with_thresholds(&path_string, 1, 1, 1, 1)?;

        let batch = sample_batch(42, 5.5);
        storage.persist(&batch)?;
        storage.flush()?;

        let tick_count: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM ticks")?
            .get_typed(0)?;
        assert_eq!(tick_count, 1);

        let metric_count: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM metrics")?
            .get_typed(0)?;
        assert_eq!(metric_count, batch.metrics.len() as i64);

        let event_count: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM events")?
            .get_typed(0)?;
        assert_eq!(event_count, batch.events.len() as i64);

        let agent_count: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM agents")?
            .get_typed(0)?;
        assert_eq!(agent_count, batch.agents.len() as i64);

        let latest = storage.latest_metrics(8)?;
        assert_eq!(latest.len(), batch.metrics.len());
        assert!(latest.iter().all(|m| m.tick == 42));

        storage.close()?;
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn replay_rng_scope_preserves_inner_and_outer_agent_ids()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-replay-rng-agent-ids");
        let path_string = path.to_string_lossy().to_string();
        let mut storage =
            Storage::create_new_file_with_thresholds(&path_string, 64, 4096, 1024, 1024)?;
        let outer_agent = AgentId::from(KeyData::from_ffi(0x0000_0001_0000_0001));
        let scope_agent = AgentId::from(KeyData::from_ffi(0x0000_0002_0000_0001));
        let mut batch = sample_batch(5, 1.0);
        batch.replay_events.push(ReplayEvent {
            agent_id: Some(outer_agent),
            kind: ReplayEventKind::RngSample {
                scope: ReplayRngScope::Agent {
                    agent_id: scope_agent,
                    phase: ReplayAgentPhase::Mutation,
                },
                range_min: -1.0,
                range_max: 1.0,
                value: 0.25,
            },
        });
        storage.persist(&batch)?;
        storage.flush()?;

        let replay = storage.load_replay_events()?;
        assert_eq!(replay.len(), 1);
        assert_eq!(replay[0].event, batch.replay_events[0]);
        storage.close()?;
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn replay_action_preserves_spike_target_agent_id() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-replay-action-spike-target");
        let path_string = path.to_string_lossy().to_string();
        let mut storage =
            Storage::create_new_file_with_thresholds(&path_string, 64, 4096, 1024, 1024)?;
        let actor = AgentId::from(KeyData::from_ffi(0x0000_0001_0000_0001));
        let target = AgentId::from(KeyData::from_ffi(0x0000_0002_0000_0001));
        let mut batch = sample_batch(6, 1.0);
        batch.replay_events.push(ReplayEvent {
            agent_id: Some(actor),
            kind: ReplayEventKind::Action {
                left_wheel: -0.25,
                right_wheel: 0.75,
                boost: true,
                spike_target: Some(target),
                sound_level: 0.5,
                give_intent: 0.125,
            },
        });
        storage.persist(&batch)?;
        storage.flush()?;

        let replay = storage.load_replay_events()?;
        assert_eq!(replay.len(), 1);
        assert_eq!(replay[0].event, batch.replay_events[0]);
        storage.close()?;
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn invalid_replay_batch_leaves_direct_storage_buffers_unchanged()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-invalid-replay-atomic");
        let path_string = path.to_string_lossy().to_string();
        let mut storage =
            Storage::create_new_file_with_thresholds(&path_string, 64, 4096, 1024, 1024)?;
        let mut invalid = sample_batch(1, 1.0);
        invalid.replay_events.push(ReplayEvent {
            agent_id: None,
            kind: ReplayEventKind::BrainOutputs {
                outputs: vec![f32::NAN],
            },
        });

        let error = storage
            .persist(&invalid)
            .expect_err("non-finite replay payload must be rejected");
        assert!(matches!(
            error,
            StorageError::InvalidData {
                context: "replay_events.brain_outputs",
                ..
            }
        ));
        assert!(storage.buffer.is_empty());

        storage.persist(&sample_batch(2, 2.0))?;
        storage.flush()?;
        let ticks = storage
            .connection()?
            .query("SELECT tick FROM ticks ORDER BY tick")?;
        assert_eq!(ticks.len(), 1);
        assert_eq!(decode::<i64>(&ticks[0], 0, "ticks.tick")?, 2);
        storage.close()?;
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn late_checked_conversion_failure_preserves_buffer_and_pipeline_usability()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-late-conversion-atomic");
        let path_string = path.to_string_lossy().to_string();
        let mut storage =
            Storage::create_new_file_with_thresholds(&path_string, 64, 4096, 1024, 1024)?;
        storage.persist(&sample_batch(1, 1.0))?;
        let lengths_before = (
            storage.buffer.ticks.len(),
            storage.buffer.metrics.len(),
            storage.buffer.events.len(),
            storage.buffer.agents.len(),
            storage.buffer.births.len(),
            storage.buffer.deaths.len(),
            storage.buffer.replay_events.len(),
        );

        let mut invalid = sample_batch(2, 2.0);
        invalid.deaths.push(DeathRecord {
            tick: Tick(2),
            agent_id: AgentId::default(),
            age: 1,
            generation: scriptbots_core::Generation(1),
            herbivore_tendency: 0.5,
            brain_kind: Some("test.invalid-key".to_owned()),
            brain_key: Some(u64::MAX),
            energy: 0.0,
            food_balance_total: 0.0,
            cause: DeathCause::Unknown,
            was_hybrid: false,
            combat_flags: scriptbots_core::CombatEventFlags::default(),
        });
        assert_invalid_data_context(storage.persist(&invalid), "deaths.brain_key");
        assert_eq!(
            (
                storage.buffer.ticks.len(),
                storage.buffer.metrics.len(),
                storage.buffer.events.len(),
                storage.buffer.agents.len(),
                storage.buffer.births.len(),
                storage.buffer.deaths.len(),
                storage.buffer.replay_events.len(),
            ),
            lengths_before,
            "failed late conversion must not partially append a prepared batch"
        );

        storage.persist(&sample_batch(3, 3.0))?;
        storage.flush()?;
        let ticks = storage
            .connection()?
            .query("SELECT tick FROM ticks ORDER BY tick")?;
        assert_eq!(
            ticks
                .iter()
                .map(|row| decode::<i64>(row, 0, "ticks.tick"))
                .collect::<Result<Vec<_>, _>>()?,
            vec![1, 3]
        );
        storage.close()?;
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn invalid_replay_is_rejected_before_worker_admission() -> Result<(), Box<dyn std::error::Error>>
    {
        let path = temp_db_path("storage-invalid-replay-admission");
        let path_string = path.to_string_lossy().to_string();
        let mut pipeline =
            StoragePipeline::create_new_file_with_thresholds(&path_string, 64, 4096, 1024, 1024)?;
        pipeline.submit(&sample_batch(1, 1.0))?;
        let mut invalid = sample_batch(2, 2.0);
        invalid.replay_events.push(ReplayEvent {
            agent_id: None,
            kind: ReplayEventKind::RngSample {
                scope: ReplayRngScope::World,
                range_min: 0.0,
                range_max: 1.0,
                value: f32::INFINITY,
            },
        });
        assert!(matches!(
            pipeline.submit(&invalid),
            Err(StorageError::InvalidData {
                context: "replay_events.rng_sample",
                ..
            })
        ));
        let validation_failure = pipeline.analytics_provider().snapshot();
        assert!(!validation_failure.stopped);
        assert!(
            validation_failure
                .last_error
                .as_deref()
                .is_some_and(|error| error.contains("non-finite replay value"))
        );
        assert_eq!(
            validation_failure
                .last_failure
                .as_ref()
                .and_then(|failure| failure.path.as_deref()),
            Some(path_string.as_str())
        );
        pipeline.submit(&sample_batch(3, 3.0))?;
        let receipt = pipeline.shutdown()?;
        assert_eq!(receipt.committed_tick, Some(3));

        let reader = StorageReader::open(&path_string)?;
        let ticks = reader.recent_ticks(None)?;
        assert_eq!(
            ticks.iter().map(|tick| tick.tick).collect::<Vec<_>>(),
            vec![1, 3]
        );
        reader.close()?;
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn top_predators_tracks_average_energy() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-predators");
        let path_string = path.to_string_lossy().to_string();
        let mut storage = Storage::create_new_file_with_thresholds(&path_string, 1, 1, 1, 1)?;

        let batch_one = sample_batch(1, 1.0);
        storage.persist(&batch_one)?;
        storage.flush()?;

        let mut batch_two = sample_batch(2, 3.0);
        if let Some(agent) = batch_two.agents.first_mut() {
            agent.data.spike_length = 2.5;
        }
        storage.persist(&batch_two)?;
        storage.flush()?;

        let metrics = storage.latest_metrics(4)?;
        assert_eq!(metrics.len(), 3);
        assert!(metrics.iter().all(|reading| reading.tick == 2));

        let predators = storage.top_predators(4)?;
        assert!(!predators.is_empty());
        let leader = &predators[0];
        assert!((leader.avg_energy - 2.0).abs() < 1e-6);
        assert_eq!(leader.last_tick, 2);

        storage.close()?;
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn production_schema_constraints_and_type_errors_are_observable()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-production-schema");
        let path_string = path.to_string_lossy().to_string();
        let mut storage = Storage::create_new_file(&path_string)?;

        let invalid_bool = storage.connection()?.execute(
            "INSERT INTO ticks (
                tick, epoch, closed, agent_count, births, deaths,
                total_energy, average_energy, average_health
             ) VALUES (1, 0, 2, 0, 0, 0, 0.0, 0.0, 0.0)",
        );
        assert!(
            invalid_bool.is_err(),
            "closed CHECK must reject values outside 0/1"
        );

        let invalid_null = storage.connection()?.execute(
            "INSERT INTO ticks (
                tick, epoch, closed, agent_count, births, deaths,
                total_energy, average_energy, average_health
             ) VALUES (2, 0, 0, 0, 0, 0, NULL, 0.0, 0.0)",
        );
        assert!(invalid_null.is_err(), "NOT NULL columns must reject NULL");

        let mut batch = sample_batch(11, 1.0);
        let agent_id = batch.agents[0].id;
        batch.births.push(BirthRecord {
            tick: Tick(11),
            agent_id,
            parent_a: None,
            parent_b: None,
            brain_kind: Some("schema-test".to_owned()),
            brain_key: Some(7),
            herbivore_tendency: 0.5,
            generation: scriptbots_core::Generation(0),
            position: Position::new(1.0, 2.0),
            is_hybrid: false,
        });
        batch.deaths.push(DeathRecord {
            tick: Tick(11),
            agent_id,
            age: 0,
            generation: scriptbots_core::Generation(0),
            herbivore_tendency: 0.5,
            brain_kind: Some("schema-test".to_owned()),
            brain_key: Some(7),
            energy: 0.0,
            food_balance_total: 0.0,
            cause: DeathCause::Unknown,
            was_hybrid: false,
            combat_flags: scriptbots_core::CombatEventFlags::default(),
        });
        batch.replay_events.push(ReplayEvent {
            agent_id: None,
            kind: ReplayEventKind::BrainOutputs {
                outputs: vec![0.25],
            },
        });
        storage.persist(&batch)?;
        storage.flush()?;

        let negative_updates = [
            ("ticks.tick", "UPDATE ticks SET tick = -1 WHERE tick = 11"),
            ("ticks.epoch", "UPDATE ticks SET epoch = -1 WHERE tick = 11"),
            (
                "ticks.agent_count",
                "UPDATE ticks SET agent_count = -1 WHERE tick = 11",
            ),
            (
                "ticks.births",
                "UPDATE ticks SET births = -1 WHERE tick = 11",
            ),
            (
                "ticks.deaths",
                "UPDATE ticks SET deaths = -1 WHERE tick = 11",
            ),
            (
                "metrics.tick",
                "UPDATE metrics SET tick = -1 WHERE tick = 11",
            ),
            ("events.tick", "UPDATE events SET tick = -1 WHERE tick = 11"),
            (
                "events.count",
                "UPDATE events SET count = -1 WHERE tick = 11",
            ),
            (
                "replay_events.tick",
                "UPDATE replay_events SET tick = -1 WHERE tick = 11",
            ),
            (
                "replay_events.seq",
                "UPDATE replay_events SET seq = -1 WHERE tick = 11",
            ),
            (
                "replay_events.agent_id",
                "UPDATE replay_events SET agent_id = -1 WHERE tick = 11",
            ),
            ("agents.tick", "UPDATE agents SET tick = -1 WHERE tick = 11"),
            (
                "agents.agent_id",
                "UPDATE agents SET agent_id = -1 WHERE tick = 11",
            ),
            (
                "agents.generation",
                "UPDATE agents SET generation = -1 WHERE tick = 11",
            ),
            ("agents.age", "UPDATE agents SET age = -1 WHERE tick = 11"),
            (
                "agents.brain_key",
                "UPDATE agents SET brain_key = -1 WHERE tick = 11",
            ),
            ("births.tick", "UPDATE births SET tick = -1 WHERE tick = 11"),
            (
                "births.agent_id",
                "UPDATE births SET agent_id = -1 WHERE tick = 11",
            ),
            (
                "births.parent_a",
                "UPDATE births SET parent_a = -1 WHERE tick = 11",
            ),
            (
                "births.parent_b",
                "UPDATE births SET parent_b = -1 WHERE tick = 11",
            ),
            (
                "births.brain_key",
                "UPDATE births SET brain_key = -1 WHERE tick = 11",
            ),
            (
                "births.generation",
                "UPDATE births SET generation = -1 WHERE tick = 11",
            ),
            ("deaths.tick", "UPDATE deaths SET tick = -1 WHERE tick = 11"),
            (
                "deaths.agent_id",
                "UPDATE deaths SET agent_id = -1 WHERE tick = 11",
            ),
            ("deaths.age", "UPDATE deaths SET age = -1 WHERE tick = 11"),
            (
                "deaths.generation",
                "UPDATE deaths SET generation = -1 WHERE tick = 11",
            ),
            (
                "deaths.brain_key",
                "UPDATE deaths SET brain_key = -1 WHERE tick = 11",
            ),
        ];
        for (context, sql) in negative_updates {
            assert!(
                storage.connection()?.execute(sql).is_err(),
                "{context} CHECK must reject negative values"
            );
        }

        storage.connection()?.execute(
            "INSERT INTO ticks (
                tick, epoch, closed, agent_count, births, deaths,
                total_energy, average_energy, average_health
             ) VALUES (3, 'invalid-epoch', 0, 0, 0, 0, 0.0, 0.0, 0.0)",
        )?;
        storage.close()?;

        let reader = StorageReader::open(&path_string)?;
        let decode_error = reader
            .recent_ticks(None)
            .expect_err("typed reader must reject a TEXT epoch in an INTEGER domain field");
        assert!(matches!(
            decode_error,
            StorageError::InvalidData {
                context: "ticks.epoch",
                ..
            }
        ));
        reader.close()?;

        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn failed_production_flush_rolls_back_and_is_never_replayed_from_drop()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-production-rollback");
        let path_string = path.to_string_lossy().to_string();
        let mut storage =
            Storage::create_new_file_with_thresholds(&path_string, 64, 4096, 1024, 1024)?;
        storage.connection()?.execute("DROP TABLE metrics")?;
        storage.persist(&sample_batch(42, 5.5))?;

        let error = storage
            .flush()
            .expect_err("missing production table must fail the whole transaction");
        assert!(matches!(
            error,
            StorageError::Transaction {
                attempts: 1,
                transient: false,
                commit_state: FailureCommitState::RolledBack,
                ..
            }
        ));
        let tick_count: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM ticks")?
            .get_typed(0)?;
        assert_eq!(tick_count, 0, "partial tick insert escaped rollback");
        assert!(matches!(
            storage.flush(),
            Err(StorageError::TerminallyFailed)
        ));
        let buffered_lengths = (
            storage.buffer.ticks.len(),
            storage.buffer.metrics.len(),
            storage.buffer.events.len(),
            storage.buffer.agents.len(),
            storage.buffer.births.len(),
            storage.buffer.deaths.len(),
            storage.buffer.replay_events.len(),
        );
        assert!(matches!(
            storage.persist(&sample_batch(43, 6.0)),
            Err(StorageError::TerminallyFailed)
        ));
        assert_eq!(
            (
                storage.buffer.ticks.len(),
                storage.buffer.metrics.len(),
                storage.buffer.events.len(),
                storage.buffer.agents.len(),
                storage.buffer.births.len(),
                storage.buffer.deaths.len(),
                storage.buffer.replay_events.len(),
            ),
            buffered_lengths,
            "terminal storage must reject before mutating any buffer"
        );
        storage.abandon_after_error();

        let reader = StorageReader::open(&path_string)?;
        assert!(reader.recent_ticks(None)?.is_empty());
        reader.close()?;
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn retry_policy_is_transient_rollback_only_and_bounded() {
        let transient = FlushAttemptError {
            source: FrankenError::Busy,
            commit_state: FailureCommitState::RolledBack,
        };
        assert!(should_retry_transaction(&transient, 1));
        assert!(!should_retry_transaction(
            &transient,
            MAX_TRANSACTION_ATTEMPTS
        ));

        let indeterminate = FlushAttemptError {
            source: FrankenError::Busy,
            commit_state: FailureCommitState::Indeterminate,
        };
        assert!(!should_retry_transaction(&indeterminate, 1));

        let permanent = FlushAttemptError {
            source: FrankenError::DatabaseFull,
            commit_state: FailureCommitState::RolledBack,
        };
        assert!(!should_retry_transaction(&permanent, 1));
    }

    #[test]
    fn file_pipeline_receipt_is_visible_and_shutdown_closes_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-file-receipt");
        let path_string = path.to_string_lossy().to_string();
        let mut pipeline =
            StoragePipeline::create_new_file_with_thresholds(&path_string, 64, 4096, 1024, 1024)?;
        let sink = pipeline.sink();
        let reader = StorageReader::open(&path_string)?;

        sink.submit(&sample_batch(7, 2.5))?;
        assert_eq!(reader.run_ledger_summary()?.tick_count, 0);

        let flush = pipeline.flush_and_wait()?;
        assert_eq!(flush.committed_tick, Some(7));
        assert_eq!(flush.guarantee, PersistenceGuarantee::Durable);
        assert_eq!(reader.run_ledger_summary()?.tick_count, 1);

        let shutdown = pipeline.shutdown()?;
        assert_eq!(shutdown.committed_tick, Some(7));
        assert_eq!(shutdown.guarantee, PersistenceGuarantee::Durable);
        let submit_error = sink
            .submit(&sample_batch(8, 3.0))
            .expect_err("closed admission gate must reject later batches");
        assert!(matches!(
            submit_error,
            StorageError::Worker(StorageWorkerError::Channel {
                operation: StorageOperation::Admit,
                commit_state: FailureCommitState::NotAdmitted,
                ..
            })
        ));

        reader.close()?;
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn shutdown_barrier_preserves_every_successful_racing_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-admission-race");
        let path_string = path.to_string_lossy().to_string();
        let pipeline =
            StoragePipeline::create_new_file_with_thresholds(&path_string, 64, 4096, 1024, 1024)?;
        let retained_sink = pipeline.sink();

        let (entered_tx, entered_rx) = xchan::bounded(1);
        let (release_tx, release_rx) = xchan::bounded(1);
        pipeline
            .sink
            .tx
            .send(StorageCommand::PauseForAdmissionRace {
                entered: entered_tx,
                release: release_rx,
            })?;
        entered_rx.recv_timeout(Duration::from_secs(2))?;

        for tick in 1..=DEFAULT_COMMAND_CAPACITY as u64 {
            pipeline.submit(&sample_batch(tick, tick as f32))?;
        }

        let racing_sink = pipeline.sink();
        let (submit_started_tx, submit_started_rx) = xchan::bounded(1);
        let submit_handle = thread::spawn(move || {
            let _ = submit_started_tx.send(());
            let result =
                racing_sink.submit(&sample_batch(DEFAULT_COMMAND_CAPACITY as u64 + 1, 99.0));
            (racing_sink, result)
        });
        submit_started_rx.recv_timeout(Duration::from_secs(2))?;

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match pipeline.sink.admission.try_lock() {
                Err(TryLockError::WouldBlock) => break,
                Err(TryLockError::Poisoned(error)) => {
                    return Err(format!("admission gate poisoned before race: {error}").into());
                }
                Ok(guard) => drop(guard),
            }
            assert!(
                Instant::now() < deadline,
                "racing submit never blocked while holding the admission gate"
            );
            thread::yield_now();
        }

        let shutdown_handle = thread::spawn(move || {
            let mut pipeline = pipeline;
            pipeline.shutdown()
        });
        release_tx.send(())?;

        let (racing_sink, submit_result) = submit_handle
            .join()
            .expect("racing submit thread must not panic");
        submit_result.expect("submit ordered before shutdown must be admitted");
        let receipt = shutdown_handle
            .join()
            .expect("shutdown thread must not panic")?;
        assert_eq!(
            receipt.committed_tick,
            Some(DEFAULT_COMMAND_CAPACITY as u64 + 1)
        );
        assert_eq!(receipt.guarantee, PersistenceGuarantee::Durable);

        for sink in [&retained_sink, &racing_sink] {
            let error = sink
                .submit(&sample_batch(DEFAULT_COMMAND_CAPACITY as u64 + 2, 100.0))
                .expect_err("post-shutdown submit must be rejected");
            assert!(matches!(
                error,
                StorageError::Worker(StorageWorkerError::Channel {
                    operation: StorageOperation::Admit,
                    commit_state: FailureCommitState::NotAdmitted,
                    ..
                })
            ));
        }

        let reader = StorageReader::open(&path_string)?;
        let ledger = reader.run_ledger_summary()?;
        assert_eq!(
            ledger.tick_count,
            DEFAULT_COMMAND_CAPACITY as u64 + 1,
            "every admission returning Ok must survive worker shutdown"
        );
        reader.close()?;
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn pipeline_acknowledges_startup_flush_and_shutdown() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut pipeline = StoragePipeline::memory_with_thresholds(1, 1, 1, 1)?;
        pipeline.submit(&sample_batch(7, 2.5))?;

        let flush = pipeline.flush_and_wait()?;
        assert_eq!(flush.committed_tick, Some(7));
        assert_eq!(flush.guarantee, PersistenceGuarantee::CommittedVolatile);
        let committed = pipeline.analytics_provider().snapshot();
        assert_eq!(committed.committed_tick, Some(7));
        assert!(!committed.readings.is_empty());
        assert!(!committed.stopped);

        let shutdown = pipeline.shutdown()?;
        assert_eq!(shutdown.committed_tick, Some(7));
        assert_eq!(shutdown.guarantee, PersistenceGuarantee::CommittedVolatile);
        let stopped = pipeline.analytics_provider().snapshot();
        assert!(stopped.stopped);
        assert!(shutdown.analytics_revision >= flush.analytics_revision);
        Ok(())
    }

    #[test]
    fn empty_metric_batch_still_advances_committed_snapshot()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut pipeline = StoragePipeline::memory_with_thresholds(64, 4096, 1024, 1024)?;
        let mut batch = sample_batch(11, 4.0);
        batch.metrics.clear();
        pipeline.submit(&batch)?;

        let receipt = pipeline.flush_and_wait()?;
        let snapshot = pipeline.analytics_provider().snapshot();
        assert_eq!(receipt.committed_tick, Some(11));
        assert_eq!(snapshot.committed_tick, receipt.committed_tick);
        assert_eq!(snapshot.committed_agent_count, Some(1));
        assert!(snapshot.readings.is_empty());
        pipeline.shutdown()?;
        Ok(())
    }

    #[test]
    fn pipeline_reports_worker_initialization_failure() {
        let error = StoragePipeline::create_new_file("")
            .err()
            .expect("empty storage paths must fail during the startup handshake");
        assert!(matches!(error, StorageError::InvalidTarget { .. }));
    }

    #[test]
    fn writer_constructors_require_explicit_file_or_memory_mode() {
        for invalid in [
            "",
            "   ",
            ":memory:",
            "file:test.sqlite",
            "FiLe:test.sqlite",
        ] {
            assert!(matches!(
                Storage::create_new_file(invalid),
                Err(StorageError::InvalidTarget { .. })
            ));
            assert!(matches!(
                StoragePipeline::create_new_file(invalid),
                Err(StorageError::InvalidTarget { .. })
            ));
        }

        let storage = Storage::memory().expect("explicit same-thread memory target");
        storage.close().expect("close memory storage");
        let mut pipeline = StoragePipeline::memory().expect("explicit pipeline memory target");
        let receipt = pipeline.shutdown().expect("shutdown memory pipeline");
        assert_eq!(receipt.guarantee, PersistenceGuarantee::CommittedVolatile);
    }

    #[test]
    fn create_new_file_refuses_existing_main_and_sidecars_without_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        let existing = temp_db_path("storage-existing-main");
        fs::write(&existing, b"sentinel-main")?;
        let existing_string = existing.to_string_lossy().to_string();
        assert!(matches!(
            Storage::create_new_file(&existing_string),
            Err(StorageError::InvalidTarget { .. })
        ));
        assert!(matches!(
            StoragePipeline::create_new_file(&existing_string),
            Err(StorageError::InvalidTarget { .. })
        ));
        assert_eq!(fs::read(&existing)?, b"sentinel-main");

        for (index, suffix) in STORAGE_SIDECAR_SUFFIXES.into_iter().enumerate() {
            let main = temp_db_path(&format!("storage-sidecar-{index}"));
            let sidecar = storage_sidecar_paths(&main)
                .find(|candidate| candidate.as_os_str().to_string_lossy().ends_with(suffix))
                .expect("requested sidecar suffix");
            fs::write(&sidecar, b"sentinel-sidecar")?;
            let main_string = main.to_string_lossy().to_string();
            assert!(matches!(
                StoragePipeline::create_new_file(&main_string),
                Err(StorageError::InvalidTarget { .. })
            ));
            assert!(!path_entry_exists(&main)?);
            assert_eq!(fs::read(&sidecar)?, b"sentinel-sidecar");
            fs::remove_file(sidecar)?;
        }

        fs::remove_file(existing)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn dangling_sidecar_symlink_blocks_new_file_creation() -> Result<(), Box<dyn std::error::Error>>
    {
        use std::os::unix::fs::symlink;

        let main = temp_db_path("storage-dangling-sidecar");
        let sidecar = storage_sidecar_paths(&main)
            .next()
            .expect("at least one sidecar suffix");
        let missing_target = temp_db_path("storage-missing-sidecar-target");
        symlink(&missing_target, &sidecar)?;

        let error = StoragePipeline::create_new_file(&main.to_string_lossy())
            .err()
            .expect("dangling sidecar symlink must be treated as an existing entry");
        assert!(matches!(error, StorageError::InvalidTarget { .. }));
        assert!(!path_entry_exists(&main)?);
        assert!(fs::symlink_metadata(&sidecar)?.file_type().is_symlink());

        fs::remove_file(sidecar)?;
        Ok(())
    }

    #[test]
    fn post_reservation_sidecar_race_fails_closed_and_retains_main_reservation()
    -> Result<(), Box<dyn std::error::Error>> {
        let main = temp_db_path("storage-post-reservation-race");
        let sidecar = storage_sidecar_paths(&main)
            .next()
            .expect("at least one sidecar suffix");
        let result = reserve_new_file_with_hook(&main.to_string_lossy(), |_| {
            fs::write(&sidecar, b"racing-sidecar").expect("inject sidecar race");
        });

        assert!(matches!(result, Err(StorageError::InvalidTarget { .. })));
        assert_eq!(fs::metadata(&main)?.len(), 0);
        assert_eq!(fs::read(&sidecar)?, b"racing-sidecar");

        fs::remove_file(sidecar)?;
        fs::remove_file(main)?;
        Ok(())
    }

    #[test]
    fn prepare_batch_rejects_out_of_range_tick() {
        let error = Storage::prepare_batch(&sample_batch(u64::MAX, 1.0))
            .expect_err("tick above i64::MAX must fail batch preparation");
        assert!(matches!(
            error,
            StorageError::InvalidData {
                context: "ticks.tick",
                ..
            }
        ));
    }

    #[test]
    fn prepare_batch_rejects_out_of_range_agent_id() {
        let mut batch = sample_batch(9, 1.0);
        batch.agents[0].id = AgentId::from(KeyData::from_ffi(u64::MAX));
        let error = Storage::prepare_batch(&batch)
            .expect_err("agent id above i64::MAX must fail batch preparation");
        assert!(matches!(
            error,
            StorageError::InvalidData {
                context: "agents.agent_id",
                ..
            }
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn prepare_batch_rejects_every_out_of_range_count() {
        let overflow = usize::try_from(i64::MAX).expect("64-bit usize") + 1;
        let assert_context = |batch: &PersistenceBatch, expected| {
            let error = Storage::prepare_batch(batch)
                .expect_err("count above i64::MAX must fail batch preparation");
            assert!(matches!(
                error,
                StorageError::InvalidData { context, .. } if context == expected
            ));
        };

        let mut batch = sample_batch(9, 1.0);
        batch.summary.agent_count = overflow;
        assert_context(&batch, "ticks.agent_count");

        let mut batch = sample_batch(9, 1.0);
        batch.summary.births = overflow;
        assert_context(&batch, "ticks.births");

        let mut batch = sample_batch(9, 1.0);
        batch.summary.deaths = overflow;
        assert_context(&batch, "ticks.deaths");

        let mut batch = sample_batch(9, 1.0);
        batch.events[0].count = overflow;
        assert_context(&batch, "events.count");
    }

    #[test]
    fn prepare_batch_rejects_out_of_range_nested_replay_agent_ids() {
        let invalid_id = AgentId::from(KeyData::from_ffi(u64::MAX));

        let mut action = sample_batch(9, 1.0);
        action.replay_events.push(ReplayEvent {
            agent_id: None,
            kind: ReplayEventKind::Action {
                left_wheel: 0.0,
                right_wheel: 0.0,
                boost: false,
                spike_target: Some(invalid_id),
                sound_level: 0.0,
                give_intent: 0.0,
            },
        });
        let error = Storage::prepare_batch(&action)
            .expect_err("nested spike target above i64::MAX must be rejected");
        assert!(matches!(
            error,
            StorageError::InvalidData {
                context: "replay_events.action.spike_target",
                ..
            }
        ));

        let mut rng = sample_batch(9, 1.0);
        rng.replay_events.push(ReplayEvent {
            agent_id: None,
            kind: ReplayEventKind::RngSample {
                scope: ReplayRngScope::Agent {
                    agent_id: invalid_id,
                    phase: ReplayAgentPhase::Mutation,
                },
                range_min: 0.0,
                range_max: 1.0,
                value: 0.5,
            },
        });
        let error = Storage::prepare_batch(&rng)
            .expect_err("nested RNG scope agent above i64::MAX must be rejected");
        assert!(matches!(
            error,
            StorageError::InvalidData {
                context: "replay_events.rng_sample.scope_agent_id",
                ..
            }
        ));
    }

    #[test]
    fn checked_encoding_covers_epoch_lifecycle_ids_and_brain_keys() {
        struct KeyedBrain;

        impl scriptbots_core::BrainRunner for KeyedBrain {
            fn kind(&self) -> &'static str {
                "test.keyed"
            }

            fn tick(
                &mut self,
                _inputs: &[f32; scriptbots_core::INPUT_SIZE],
            ) -> [f32; scriptbots_core::OUTPUT_SIZE] {
                [0.0; scriptbots_core::OUTPUT_SIZE]
            }
        }

        let invalid_id = AgentId::from(KeyData::from_ffi(u64::MAX));

        let mut epoch = sample_batch(9, 1.0);
        epoch.epoch = u64::MAX;
        assert_invalid_data_context(Storage::prepare_batch(&epoch), "ticks.epoch");

        let mut agent = sample_agent(1.0);
        agent.runtime.brain = BrainBinding::inherited(Box::new(KeyedBrain), Some(u64::MAX));
        assert_invalid_data_context(agent_row_from_snapshot(9, &agent), "agents.brain_key");

        let base_birth = BirthRecord {
            tick: Tick(9),
            agent_id: AgentId::default(),
            parent_a: None,
            parent_b: None,
            brain_kind: Some("test.keyed".to_owned()),
            brain_key: Some(7),
            herbivore_tendency: 0.5,
            generation: scriptbots_core::Generation(1),
            position: Position::new(1.0, 2.0),
            is_hybrid: false,
        };
        let mut birth = base_birth.clone();
        birth.tick = Tick(u64::MAX);
        assert_invalid_data_context(birth_row_from_record(&birth), "births.tick");
        let mut birth = base_birth.clone();
        birth.agent_id = invalid_id;
        assert_invalid_data_context(birth_row_from_record(&birth), "births.agent_id");
        let mut birth = base_birth.clone();
        birth.parent_a = Some(invalid_id);
        assert_invalid_data_context(birth_row_from_record(&birth), "births.parent_a");
        let mut birth = base_birth.clone();
        birth.parent_b = Some(invalid_id);
        assert_invalid_data_context(birth_row_from_record(&birth), "births.parent_b");
        let mut birth = base_birth;
        birth.brain_key = Some(u64::MAX);
        assert_invalid_data_context(birth_row_from_record(&birth), "births.brain_key");

        let base_death = DeathRecord {
            tick: Tick(9),
            agent_id: AgentId::default(),
            age: 1,
            generation: scriptbots_core::Generation(1),
            herbivore_tendency: 0.5,
            brain_kind: Some("test.keyed".to_owned()),
            brain_key: Some(7),
            energy: 0.0,
            food_balance_total: 0.0,
            cause: DeathCause::Unknown,
            was_hybrid: false,
            combat_flags: scriptbots_core::CombatEventFlags::default(),
        };
        let mut death = base_death.clone();
        death.tick = Tick(u64::MAX);
        assert_invalid_data_context(death_row_from_record(&death), "deaths.tick");
        let mut death = base_death.clone();
        death.agent_id = invalid_id;
        assert_invalid_data_context(death_row_from_record(&death), "deaths.agent_id");
        let mut death = base_death;
        death.brain_key = Some(u64::MAX);
        assert_invalid_data_context(death_row_from_record(&death), "deaths.brain_key");

        let replay = ReplayEvent {
            agent_id: Some(invalid_id),
            kind: ReplayEventKind::BrainOutputs {
                outputs: vec![0.25],
            },
        };
        assert_invalid_data_context(
            replay_row_from_event(&replay, 9, 0),
            "replay_events.agent_id",
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn replay_sequence_above_sql_integer_range_is_rejected() {
        let replay = ReplayEvent {
            agent_id: None,
            kind: ReplayEventKind::BrainOutputs {
                outputs: vec![0.25],
            },
        };
        let overflow = usize::try_from(i64::MAX).expect("64-bit usize") + 1;
        assert_invalid_data_context(
            replay_row_from_event(&replay, 9, overflow),
            "replay_events.seq",
        );
    }

    #[test]
    fn out_of_range_batch_fails_admission_without_poisoning_pipeline()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut pipeline = StoragePipeline::memory_with_thresholds(64, 4096, 1024, 1024)?;
        let mut bad = sample_batch(7, 1.0);
        bad.agents[0].id = AgentId::from(KeyData::from_ffi(u64::MAX));

        let error = pipeline
            .submit(&bad)
            .expect_err("out-of-range agent id must be rejected at admission");
        assert!(matches!(
            error,
            StorageError::InvalidData {
                context: "agents.agent_id",
                ..
            }
        ));

        pipeline.submit(&sample_batch(8, 2.0))?;
        let receipt = pipeline.flush_and_wait()?;
        assert_eq!(receipt.committed_tick, Some(8));
        pipeline.shutdown()?;
        Ok(())
    }

    #[test]
    fn terminal_flush_failure_root_cause_survives_worker_join()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-flush-terminal-join");
        let path_string = path.to_string_lossy().to_string();
        let mut pipeline =
            StoragePipeline::create_new_file_with_thresholds(&path_string, 64, 4096, 1024, 1024)?;
        pipeline.submit(&sample_batch(11, 1.5))?;

        // Apply the sabotage on the connection-owning worker thread. Opening a
        // second writer here can block indefinitely on FrankenSQLite's strict
        // writer exclusion and would test lock contention rather than error
        // propagation.
        pipeline.drop_metrics_table_for_test()?;

        let flush_error = pipeline
            .flush_and_wait()
            .expect_err("flush against a dropped table must fail");
        let StorageError::Worker(reply_error) = flush_error else {
            return Err("flush must surface a structured worker error".into());
        };
        let reply_status = reply_error.status();
        assert_eq!(reply_status.kind, StorageFailureKind::Database);
        assert_eq!(reply_status.operation, StorageOperation::Flush);

        let shutdown_error = pipeline
            .shutdown()
            .expect_err("shutdown after a terminal flush failure must report the root cause");
        let StorageError::Worker(join_error) = shutdown_error else {
            return Err("shutdown must surface a structured worker error".into());
        };
        let join_status = join_error.status();
        assert_eq!(join_status.kind, StorageFailureKind::Database);
        assert_eq!(join_status.operation, StorageOperation::Flush);
        assert!(
            join_status.detail.contains("metrics"),
            "join error must preserve the flush root cause, got: {}",
            join_status.detail
        );

        let _ = fs::remove_file(path);
        Ok(())
    }
}
