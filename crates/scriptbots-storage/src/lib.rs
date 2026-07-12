//! FrankenSQLite-backed persistence layer for ScriptBots.

use arc_swap::ArcSwap;
use crossbeam_channel as xchan;
use fsqlite::{
    Connection, FileIdentity, FrankenError, Row, SqliteValue,
    compat::{FromSqliteValue, OpenFlags, RowExt, Transaction, TransactionExt, open_with_flags},
    migrate::MigrationRunner,
};
use scriptbots_core::{
    AgentId, AgentState, BirthRecord, BrainBinding, DeathCause, DeathRecord,
    PersistenceAdmissionError, PersistenceAdmissionState, PersistenceBatch, PersistenceEventKind,
    ReplayAgentPhase, ReplayEvent, ReplayEventKind, ReplayRngScope, WorldPersistence,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{self, Value, json};
use slotmap::{Key, KeyData};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    thread,
    time::{Duration, Instant},
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
const DEFAULT_STARTUP_ACK_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_COMMAND_ENQUEUE_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_ADMISSION_ACK_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_FLUSH_ACK_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_SHUTDOWN_ACK_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_STORAGE_WAIT_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);
const MAX_TRANSACTION_ATTEMPTS: u8 = 4;
const OUTBOX_PAYLOAD_VERSION: u32 = 1;
const STORAGE_WRITER_LOCK_SUFFIX: &str = ".scriptbots-writer.lock";

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

const SCRIPTBOTS_SCHEMA_V2: &str = "
    CREATE TABLE storage_progress (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        admitted_batch_id INTEGER NOT NULL CHECK (admitted_batch_id >= 0),
        applied_batch_id INTEGER NOT NULL CHECK (
            applied_batch_id >= 0 AND applied_batch_id <= admitted_batch_id
        ),
        durable_batch_id INTEGER NOT NULL CHECK (
            durable_batch_id >= 0 AND durable_batch_id <= applied_batch_id
        )
    );
    INSERT INTO storage_progress (
        singleton, admitted_batch_id, applied_batch_id, durable_batch_id
    ) VALUES (1, 0, 0, 0);
    CREATE TABLE storage_batch_ledger (
        batch_id INTEGER PRIMARY KEY CHECK (batch_id > 0),
        tick INTEGER NOT NULL UNIQUE CHECK (tick >= 0),
        payload_digest TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('admitted', 'applied', 'durable'))
    );
    CREATE TABLE storage_outbox (
        batch_id INTEGER PRIMARY KEY CHECK (batch_id > 0),
        payload TEXT NOT NULL
    );
";

#[derive(Debug, PartialEq, Eq)]
struct SchemaObject {
    object_type: String,
    name: String,
    table_name: String,
    sql: Option<String>,
}

impl SchemaObject {
    fn summary(&self) -> String {
        let sql_fingerprint = self.sql.as_deref().map_or_else(
            || "none".to_owned(),
            |sql| blake3::hash(sql.as_bytes()).to_hex().to_string(),
        );
        format!(
            "type={:?}, name={:?}, table={:?}, sql=blake3:{sql_fingerprint}",
            self.object_type, self.name, self.table_name
        )
    }
}

fn install_scriptbots_schema(connection: &Connection) -> Result<(), StorageError> {
    MigrationRunner::new()
        .add(1, "create_scriptbots_schema", SCRIPTBOTS_SCHEMA_V1)
        .add(2, "create_durable_persistence_outbox", SCRIPTBOTS_SCHEMA_V2)
        .run(connection)?;
    Ok(())
}

fn read_schema_objects(connection: &Connection) -> Result<Vec<SchemaObject>, StorageError> {
    connection
        .query(
            "SELECT type, name, tbl_name, sql
             FROM sqlite_schema
             ORDER BY type ASC, name ASC, tbl_name ASC, sql ASC",
        )?
        .into_iter()
        .map(|row| {
            Ok(SchemaObject {
                object_type: decode(&row, 0, "sqlite_schema.type")?,
                name: decode(&row, 1, "sqlite_schema.name")?,
                table_name: decode(&row, 2, "sqlite_schema.tbl_name")?,
                sql: decode(&row, 3, "sqlite_schema.sql")?,
            })
        })
        .collect()
}

fn schema_fingerprint(objects: &[SchemaObject]) -> String {
    let mut hasher = blake3::Hasher::new();
    for object in objects {
        for field in [
            Some(object.object_type.as_str()),
            Some(object.name.as_str()),
            Some(object.table_name.as_str()),
            object.sql.as_deref(),
        ] {
            match field {
                Some(field) => {
                    hasher.update(b"S");
                    hasher.update(&field.len().to_le_bytes());
                    hasher.update(field.as_bytes());
                }
                None => {
                    hasher.update(b"N");
                }
            }
        }
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn canonical_schema_objects() -> Result<Vec<SchemaObject>, StorageError> {
    let connection = Connection::open(":memory:")?;
    let result = (|| {
        install_scriptbots_schema(&connection)?;
        read_schema_objects(&connection)
    })();
    let close_result = connection
        .close_without_checkpoint()
        .map_err(StorageError::from);
    let objects = result?;
    close_result?;
    Ok(objects)
}

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
    #[error("another ScriptBots writer owns database {path} through OS lease {lock_path}")]
    WriterLeaseHeld { path: String, lock_path: String },
    #[error(transparent)]
    Worker(#[from] StorageWorkerError),
    #[error("invalid replay event at tick {tick}, seq {seq}: {reason}")]
    ReplayParse { tick: i64, seq: i64, reason: String },
}

#[derive(Debug)]
enum StorageTarget {
    Memory,
    CreateNewFile(String),
    RecoverExisting(String),
}

impl StorageTarget {
    fn path(&self) -> &str {
        match self {
            Self::Memory => ":memory:",
            Self::CreateNewFile(path) | Self::RecoverExisting(path) => path,
        }
    }

    const fn guarantee(&self) -> PersistenceGuarantee {
        match self {
            Self::Memory => PersistenceGuarantee::CommittedVolatile,
            Self::CreateNewFile(_) | Self::RecoverExisting(_) => PersistenceGuarantee::Durable,
        }
    }

    fn prepare_for_open(&self) -> Result<(), StorageError> {
        match self {
            Self::Memory => {}
            Self::CreateNewFile(path) => ensure_no_storage_sidecars(Path::new(path))?,
            Self::RecoverExisting(path) => validate_durable_storage_path(path)?,
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
    Recovery,
    Admit,
    Persist,
    Flush,
    Durability,
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

/// Controller-side wait phase that exhausted its configured deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageWaitPhase {
    AdmissionGate,
    CommandEnqueue,
    Acknowledgement,
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
        source: Box<FrankenError>,
    },
    #[error(
        "storage {operation:?} timed out during {phase:?} at {path} after {waited:?} (tick={tick:?}, commit_state={commit_state:?})"
    )]
    Timeout {
        operation: StorageOperation,
        phase: StorageWaitPhase,
        path: String,
        tick: Option<u64>,
        waited: Duration,
        commit_state: FailureCommitState,
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
    #[error(
        "storage {operation:?} refused a second writer for {path}; OS lease {lock_path} is held (tick={tick:?}, commit_state={commit_state:?})"
    )]
    WriterLeaseHeld {
        operation: StorageOperation,
        path: String,
        lock_path: String,
        tick: Option<u64>,
        commit_state: FailureCommitState,
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
    Timeout,
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
            Self::Timeout {
                operation,
                phase,
                path,
                tick,
                waited,
                commit_state,
            } => StorageFailureStatus {
                kind: StorageFailureKind::Timeout,
                operation: *operation,
                path: Some(path.clone()),
                tick: *tick,
                attempt: 0,
                transient: true,
                commit_state: *commit_state,
                detail: format!("{phase:?} deadline exhausted after {waited:?}"),
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
            Self::WriterLeaseHeld {
                operation,
                path,
                lock_path,
                tick,
                commit_state,
            } => StorageFailureStatus {
                kind: StorageFailureKind::Internal,
                operation: *operation,
                path: Some(path.clone()),
                tick: *tick,
                attempt: 0,
                transient: true,
                commit_state: *commit_state,
                detail: format!("OS writer lease {lock_path} is held"),
            },
        }
    }

    fn with_commit_state(mut self, state: FailureCommitState) -> Self {
        match &mut self {
            Self::Database { commit_state, .. }
            | Self::Timeout { commit_state, .. }
            | Self::Channel { commit_state, .. }
            | Self::Internal { commit_state, .. }
            | Self::WriterLeaseHeld { commit_state, .. } => *commit_state = state,
        }
        self
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
            source: Box::new(source),
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
            source: Box::new(source),
        },
        StorageError::WriterLeaseHeld {
            path: lease_path,
            lock_path,
        } => StorageWorkerError::WriterLeaseHeld {
            operation,
            path: lease_path,
            lock_path,
            tick,
            commit_state: default_commit_state,
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
            source: Box::new(FrankenError::Internal(source.to_string())),
        },
        StorageWorkerError::Timeout {
            operation,
            phase,
            path,
            tick,
            waited,
            commit_state,
        } => StorageWorkerError::Timeout {
            operation: *operation,
            phase: *phase,
            path: path.clone(),
            tick: *tick,
            waited: *waited,
            commit_state: *commit_state,
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
        StorageWorkerError::WriterLeaseHeld {
            operation,
            path,
            lock_path,
            tick,
            commit_state,
        } => StorageWorkerError::WriterLeaseHeld {
            operation: *operation,
            path: path.clone(),
            lock_path: lock_path.clone(),
            tick: *tick,
            commit_state: *commit_state,
        },
    }
}

/// Summary row written to the `ticks` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetricRow {
    tick: i64,
    name: String,
    value: f64,
}

/// Event row persisted for analytics.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Monotonic identifier assigned when a lossless batch enters the durable outbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PersistenceBatchId(u64);

impl PersistenceBatchId {
    /// Construct an identifier from its persisted positive integer representation.
    fn new(value: u64) -> Result<Self, StorageError> {
        if value == 0 || value > i64::MAX as u64 {
            return Err(StorageError::InvalidData {
                context: "storage_batch_ledger.batch_id",
                reason: format!("batch id {value} is outside 1..=i64::MAX"),
            });
        }
        Ok(Self(value))
    }

    /// Return the stable integer representation.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    fn as_i64(self) -> i64 {
        self.0 as i64
    }
}

/// Monotonic prefixes proven at each persistence boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PersistenceWatermarks {
    /// Highest batch whose exact payload was admitted to the outbox.
    pub admitted: Option<PersistenceBatchId>,
    /// Highest contiguous batch atomically applied to the scientific tables.
    pub applied: Option<PersistenceBatchId>,
    /// Highest contiguous file-backed batch carrying a durable marker.
    pub durable: Option<PersistenceBatchId>,
}

impl PersistenceWatermarks {
    fn from_raw(admitted: i64, applied: i64, durable: i64) -> Result<Self, StorageError> {
        if admitted < 0 || applied < 0 || durable < 0 || durable > applied || applied > admitted {
            return Err(StorageError::InvalidData {
                context: "storage_progress",
                reason: format!(
                    "invalid watermark ordering durable={durable}, applied={applied}, admitted={admitted}"
                ),
            });
        }
        let decode = |value: i64| -> Result<Option<PersistenceBatchId>, StorageError> {
            if value == 0 {
                Ok(None)
            } else {
                PersistenceBatchId::new(value as u64).map(Some)
            }
        };
        Ok(Self {
            admitted: decode(admitted)?,
            applied: decode(applied)?,
            durable: decode(durable)?,
        })
    }

    fn admitted_raw(self) -> i64 {
        self.admitted.map_or(0, PersistenceBatchId::as_i64)
    }

    fn applied_raw(self) -> i64 {
        self.applied.map_or(0, PersistenceBatchId::as_i64)
    }

    fn durable_raw(self) -> i64 {
        self.durable.map_or(0, PersistenceBatchId::as_i64)
    }
}

/// Durable ledger state for one admitted persistence batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchPersistenceState {
    Admitted,
    Applied,
    Durable,
}

/// Query result for one batch in the compact persistence ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedBatchStatus {
    pub batch_id: PersistenceBatchId,
    pub tick: u64,
    pub payload_digest: String,
    pub state: BatchPersistenceState,
}

/// Synchronous proof that the exact batch payload entered the worker outbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionReceipt {
    pub batch_id: PersistenceBatchId,
    pub tick: u64,
    pub guarantee: PersistenceGuarantee,
    pub watermarks: PersistenceWatermarks,
}

/// Immutable, lock-free read model published after successful storage commits.
#[derive(Debug, Clone)]
pub struct AnalyticsSnapshot {
    pub revision: u64,
    pub committed_tick: Option<u64>,
    pub committed_agent_count: Option<usize>,
    pub watermarks: PersistenceWatermarks,
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
            watermarks: PersistenceWatermarks::default(),
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

    fn publish_progress(&self, watermarks: PersistenceWatermarks) {
        self.inner.rcu(|current| {
            if current.stopped || current.watermarks == watermarks {
                return Arc::clone(current);
            }
            Arc::new(AnalyticsSnapshot {
                revision: current.revision.saturating_add(1),
                committed_tick: current.committed_tick,
                committed_agent_count: current.committed_agent_count,
                watermarks,
                readings: Arc::clone(&current.readings),
                last_error: current.last_error.clone(),
                last_failure: current.last_failure.clone(),
                stopped: false,
            })
        });
    }

    fn publish_committed(&self, pending: PendingAnalytics, watermarks: PersistenceWatermarks) {
        self.inner.rcu(|current| {
            if current.stopped {
                return Arc::clone(current);
            }
            Arc::new(AnalyticsSnapshot {
                revision: current.revision.saturating_add(1),
                committed_tick: Some(pending.tick),
                committed_agent_count: Some(pending.agent_count),
                watermarks,
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
                watermarks: current.watermarks,
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
                watermarks: current.watermarks,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Default, Serialize, Deserialize)]
struct StorageBuffer {
    ticks: Vec<TickRow>,
    metrics: Vec<MetricRow>,
    events: Vec<EventRow>,
    agents: Vec<AgentRow>,
    births: Vec<BirthRow>,
    deaths: Vec<DeathRow>,
    replay_events: Vec<ReplayEventRow>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OutboxPayload {
    version: u32,
    tick: u64,
    storage: StorageBuffer,
}

#[derive(Serialize)]
struct OutboxPayloadRef<'a> {
    version: u32,
    tick: u64,
    storage: &'a StorageBuffer,
}

#[derive(Debug)]
struct RecoveredOutboxBatch {
    batch_id: PersistenceBatchId,
    tick: u64,
    payload_digest: String,
    storage: StorageBuffer,
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

fn execute_transaction_with_retry<F>(
    connection: &Connection,
    mut operation: F,
) -> Result<u8, StorageError>
where
    F: FnMut(&Transaction<'_>) -> Result<(), FrankenError>,
{
    let mut attempt = 1_u8;
    loop {
        let mut transaction = match connection.transaction() {
            Ok(transaction) => transaction,
            Err(source) => {
                let failure = FlushAttemptError {
                    source,
                    commit_state: FailureCommitState::RolledBack,
                };
                if should_retry_transaction(&failure, attempt) {
                    thread::sleep(Duration::from_millis(1_u64 << attempt));
                    attempt += 1;
                    continue;
                }
                return Err(StorageError::Transaction {
                    attempts: attempt,
                    transient: failure.source.is_transient(),
                    commit_state: failure.commit_state,
                    source: failure.source,
                });
            }
        };
        let result = operation(&transaction).and_then(|()| transaction.commit());
        match result {
            Ok(()) => return Ok(attempt),
            Err(source) => {
                let failure = match transaction.rollback() {
                    Ok(()) => FlushAttemptError {
                        source,
                        commit_state: FailureCommitState::RolledBack,
                    },
                    Err(rollback_error) => FlushAttemptError {
                        source: FrankenError::Internal(format!(
                            "transaction failed ({source}); rollback also failed ({rollback_error})"
                        )),
                        commit_state: FailureCommitState::Indeterminate,
                    },
                };
                if should_retry_transaction(&failure, attempt) {
                    thread::sleep(Duration::from_millis(1_u64 << attempt));
                    attempt += 1;
                    continue;
                }
                return Err(StorageError::Transaction {
                    attempts: attempt,
                    transient: failure.source.is_transient(),
                    commit_state: failure.commit_state,
                    source: failure.source,
                });
            }
        }
    }
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

    fn validate_finite(&self) -> Result<(), StorageError> {
        let invalid = |context: &'static str, value: f64| StorageError::InvalidData {
            context,
            reason: format!("non-finite value {value}"),
        };
        for row in &self.ticks {
            for (context, value) in [
                ("ticks.total_energy", row.total_energy),
                ("ticks.average_energy", row.average_energy),
                ("ticks.average_health", row.average_health),
            ] {
                if !value.is_finite() {
                    return Err(invalid(context, value));
                }
            }
        }
        for row in &self.metrics {
            if !row.value.is_finite() {
                return Err(invalid("metrics.value", row.value));
            }
        }
        for row in &self.agents {
            for (context, value) in [
                ("agents.position_x", row.position_x),
                ("agents.position_y", row.position_y),
                ("agents.velocity_x", row.velocity_x),
                ("agents.velocity_y", row.velocity_y),
                ("agents.heading", row.heading),
                ("agents.health", row.health),
                ("agents.energy", row.energy),
                ("agents.color_r", row.color_r),
                ("agents.color_g", row.color_g),
                ("agents.color_b", row.color_b),
                ("agents.spike_length", row.spike_length),
                ("agents.herbivore_tendency", row.herbivore_tendency),
                ("agents.sound_multiplier", row.sound_multiplier),
                ("agents.reproduction_counter", row.reproduction_counter),
                ("agents.mutation_rate_primary", row.mutation_rate_primary),
                (
                    "agents.mutation_rate_secondary",
                    row.mutation_rate_secondary,
                ),
                ("agents.trait_smell", row.trait_smell),
                ("agents.trait_sound", row.trait_sound),
                ("agents.trait_hearing", row.trait_hearing),
                ("agents.trait_eye", row.trait_eye),
                ("agents.trait_blood", row.trait_blood),
                ("agents.give_intent", row.give_intent),
                ("agents.food_delta", row.food_delta),
                ("agents.sound_output", row.sound_output),
            ] {
                if !value.is_finite() {
                    return Err(invalid(context, value));
                }
            }
        }
        for row in &self.births {
            for (context, value) in [
                ("births.herbivore_tendency", row.herbivore_tendency),
                ("births.position_x", row.position_x),
                ("births.position_y", row.position_y),
            ] {
                if !value.is_finite() {
                    return Err(invalid(context, value));
                }
            }
        }
        for row in &self.deaths {
            for (context, value) in [
                ("deaths.herbivore_tendency", row.herbivore_tendency),
                ("deaths.energy", row.energy),
                ("deaths.food_balance_total", row.food_balance_total),
            ] {
                if !value.is_finite() {
                    return Err(invalid(context, value));
                }
            }
        }
        Ok(())
    }

    fn encode_outbox(&self, tick: u64) -> Result<(String, String), StorageError> {
        self.validate_finite()?;
        let payload = serde_json::to_string(&OutboxPayloadRef {
            version: OUTBOX_PAYLOAD_VERSION,
            tick,
            storage: self,
        })
        .map_err(|error| StorageError::InvalidData {
            context: "storage_outbox.payload",
            reason: error.to_string(),
        })?;
        let digest = format!("blake3:{}", blake3::hash(payload.as_bytes()).to_hex());
        Ok((payload, digest))
    }

    fn decode_outbox(
        payload: &str,
        expected_tick: u64,
        expected_digest: &str,
    ) -> Result<Self, StorageError> {
        let actual_digest = format!("blake3:{}", blake3::hash(payload.as_bytes()).to_hex());
        if actual_digest != expected_digest {
            return Err(StorageError::InvalidData {
                context: "storage_outbox.payload_digest",
                reason: format!("expected {expected_digest}, computed {actual_digest}"),
            });
        }
        let decoded: OutboxPayload =
            serde_json::from_str(payload).map_err(|error| StorageError::InvalidData {
                context: "storage_outbox.payload",
                reason: error.to_string(),
            })?;
        if decoded.version != OUTBOX_PAYLOAD_VERSION {
            return Err(StorageError::InvalidData {
                context: "storage_outbox.payload.version",
                reason: format!(
                    "unsupported version {}, expected {}",
                    decoded.version, OUTBOX_PAYLOAD_VERSION
                ),
            });
        }
        if decoded.tick != expected_tick {
            return Err(StorageError::InvalidData {
                context: "storage_outbox.payload.tick",
                reason: format!("ledger tick {expected_tick}, payload tick {}", decoded.tick),
            });
        }
        decoded.storage.validate_finite()?;
        Ok(decoded.storage)
    }
}

fn decode_batch_state(value: &str) -> Result<BatchPersistenceState, StorageError> {
    match value {
        "admitted" => Ok(BatchPersistenceState::Admitted),
        "applied" => Ok(BatchPersistenceState::Applied),
        "durable" => Ok(BatchPersistenceState::Durable),
        other => Err(StorageError::InvalidData {
            context: "storage_batch_ledger.state",
            reason: format!("unknown state {other:?}"),
        }),
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

    /// Return durable outbox progress from an independent read connection.
    pub fn persistence_watermarks(&self) -> Result<PersistenceWatermarks, StorageError> {
        let row = self.connection()?.query_row(
            "SELECT admitted_batch_id, applied_batch_id, durable_batch_id
             FROM storage_progress
             WHERE singleton = 1",
        )?;
        PersistenceWatermarks::from_raw(
            decode(&row, 0, "storage_progress.admitted_batch_id")?,
            decode(&row, 1, "storage_progress.applied_batch_id")?,
            decode(&row, 2, "storage_progress.durable_batch_id")?,
        )
    }

    /// Query one batch's compact ledger status without loading its outbox payload.
    pub fn batch_status(
        &self,
        batch_id: PersistenceBatchId,
    ) -> Result<Option<PersistedBatchStatus>, StorageError> {
        let rows = self.connection()?.query_with_params(
            "SELECT tick, payload_digest, state
             FROM storage_batch_ledger
             WHERE batch_id = ?1",
            &[batch_id.as_i64().into()],
        )?;
        rows.first()
            .map(|row| {
                let tick = checked_u64(
                    "storage_batch_ledger.tick",
                    decode(row, 0, "storage_batch_ledger.tick")?,
                )?;
                let payload_digest = decode(row, 1, "storage_batch_ledger.payload_digest")?;
                let state_text: String = decode(row, 2, "storage_batch_ledger.state")?;
                Ok(PersistedBatchStatus {
                    batch_id,
                    tick,
                    payload_digest,
                    state: decode_batch_state(&state_text)?,
                })
            })
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

static STORAGE_PATH_LEASES: OnceLock<Mutex<BTreeSet<StorageLeaseKey>>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum StorageLeaseKey {
    File(StorageFileIdentity),
    Path(PathBuf),
}

struct StoragePathLease {
    keys: BTreeSet<StorageLeaseKey>,
}

impl StoragePathLease {
    fn acquire(path: &str) -> Result<Option<Self>, StorageError> {
        if path == ":memory:" {
            return Ok(None);
        }
        let path_ref = Path::new(path);
        let keys = match std::fs::symlink_metadata(path_ref) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_file() {
                    return Err(StorageError::InvalidData {
                        context: "storage.writer_path",
                        reason: format!("writer requires a non-symlink regular file at {path}"),
                    });
                }
                if storage_file_has_multiple_links(&metadata) {
                    return Err(StorageError::InvalidData {
                        context: "storage.writer_path",
                        reason: format!(
                            "refusing multiply linked database {path}; writer identity must have one path"
                        ),
                    });
                }
                let canonical =
                    std::fs::canonicalize(path_ref).map_err(|source| StorageError::Filesystem {
                        operation: "canonicalize writer",
                        path: path_ref.to_path_buf(),
                        source,
                    })?;
                BTreeSet::from([
                    StorageLeaseKey::File(StorageFileIdentity::from_metadata(&metadata)),
                    StorageLeaseKey::Path(canonical),
                ])
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                let absolute =
                    std::path::absolute(path_ref).map_err(|source| StorageError::Filesystem {
                        operation: "make writer path absolute",
                        path: path_ref.to_path_buf(),
                        source,
                    })?;
                let parent = absolute.parent().ok_or(StorageError::InvalidData {
                    context: "storage.writer_path",
                    reason: format!("storage path {path} has no parent"),
                })?;
                let name = absolute.file_name().ok_or(StorageError::InvalidData {
                    context: "storage.writer_path",
                    reason: format!("storage path {path} has no file name"),
                })?;
                let canonical_parent =
                    std::fs::canonicalize(parent).map_err(|source| StorageError::Filesystem {
                        operation: "canonicalize writer parent",
                        path: parent.to_path_buf(),
                        source,
                    })?;
                BTreeSet::from([StorageLeaseKey::Path(canonical_parent.join(name))])
            }
            Err(source) => {
                return Err(StorageError::Filesystem {
                    operation: "inspect writer",
                    path: path_ref.to_path_buf(),
                    source,
                });
            }
        };
        let leases = STORAGE_PATH_LEASES.get_or_init(|| Mutex::new(BTreeSet::new()));
        let mut leases = match leases.lock() {
            Ok(leases) => leases,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(conflict) = keys.iter().find(|key| leases.contains(*key)) {
            return Err(StorageError::InvalidData {
                context: "storage.path_lease",
                reason: format!("another ScriptBots writer still owns {conflict:?}"),
            });
        }
        leases.extend(keys.iter().cloned());
        Ok(Some(Self { keys }))
    }

    fn promote_existing(&mut self, path: &str) -> Result<(), StorageError> {
        if self
            .keys
            .iter()
            .any(|key| matches!(key, StorageLeaseKey::File(_)))
        {
            return Ok(());
        }
        let metadata =
            std::fs::symlink_metadata(path).map_err(|source| StorageError::Filesystem {
                operation: "inspect promoted writer",
                path: PathBuf::from(path),
                source,
            })?;
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || storage_file_has_multiple_links(&metadata)
        {
            return Err(StorageError::InvalidData {
                context: "storage.writer_path",
                reason: format!("writer path {path} changed during creation"),
            });
        }
        let promoted = StorageLeaseKey::File(StorageFileIdentity::from_metadata(&metadata));
        let leases = STORAGE_PATH_LEASES.get_or_init(|| Mutex::new(BTreeSet::new()));
        let mut leases = match leases.lock() {
            Ok(leases) => leases,
            Err(poisoned) => poisoned.into_inner(),
        };
        if leases.contains(&promoted) {
            return Err(StorageError::InvalidData {
                context: "storage.path_lease",
                reason: format!("another ScriptBots writer owns {promoted:?}"),
            });
        }
        leases.insert(promoted.clone());
        self.keys.insert(promoted);
        Ok(())
    }
}

fn storage_file_has_multiple_links(metadata: &std::fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        metadata.nlink() > 1
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.number_of_links().is_some_and(|links| links > 1)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = metadata;
        false
    }
}

impl Drop for StoragePathLease {
    fn drop(&mut self) {
        let leases = STORAGE_PATH_LEASES.get_or_init(|| Mutex::new(BTreeSet::new()));
        let mut leases = match leases.lock() {
            Ok(leases) => leases,
            Err(poisoned) => poisoned.into_inner(),
        };
        for key in &self.keys {
            leases.remove(key);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum StorageFileIdentity {
    #[cfg(unix)]
    Unix { device: u64, inode: u64 },
    #[cfg(windows)]
    Windows {
        volume: Option<u32>,
        index: Option<u64>,
    },
    #[cfg(not(any(unix, windows)))]
    Portable { length: u64 },
}

impl StorageFileIdentity {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self::Unix {
                device: metadata.dev(),
                inode: metadata.ino(),
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            Self::Windows {
                volume: metadata.volume_serial_number(),
                index: metadata.file_index(),
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            Self::Portable {
                length: metadata.len(),
            }
        }
    }
}

fn storage_writer_lock_path(path: &str) -> PathBuf {
    let mut lock_path = Path::new(path).as_os_str().to_os_string();
    lock_path.push(STORAGE_WRITER_LOCK_SUFFIX);
    PathBuf::from(lock_path)
}

struct StorageWriterLease {
    // The live descriptor is the lease. The companion path deliberately persists across runs.
    file: std::fs::File,
    lock_path: PathBuf,
}

impl StorageWriterLease {
    fn acquire(path: &str) -> Result<Option<Self>, StorageError> {
        if path == ":memory:" {
            return Ok(None);
        }

        let lock_path = storage_writer_lock_path(path);
        let lock_path_display = lock_path.display().to_string();
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let file = options
            .open(&lock_path)
            .map_err(|source| StorageError::Filesystem {
                operation: "open writer lease",
                path: lock_path.clone(),
                source,
            })?;
        let file_metadata = file.metadata().map_err(|source| StorageError::Filesystem {
            operation: "inspect opened writer lease",
            path: lock_path.clone(),
            source,
        })?;
        if !file_metadata.is_file() || storage_file_has_multiple_links(&file_metadata) {
            return Err(StorageError::InvalidData {
                context: "storage.writer_lease",
                reason: format!(
                    "writer lease must be a singly linked regular file at {lock_path_display}"
                ),
            });
        }
        let identity = StorageFileIdentity::from_metadata(&file_metadata);
        Self::verify_path(&lock_path, &lock_path_display, &identity)?;

        match file.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(StorageError::WriterLeaseHeld {
                    path: path.to_owned(),
                    lock_path: lock_path_display,
                });
            }
            Err(std::fs::TryLockError::Error(source)) => {
                return Err(StorageError::Filesystem {
                    operation: "lock writer lease",
                    path: lock_path,
                    source,
                });
            }
        }
        Self::verify_path(&lock_path, &lock_path_display, &identity)?;

        Ok(Some(Self { file, lock_path }))
    }

    fn verify_path(
        lock_path: &Path,
        lock_path_display: &str,
        identity: &StorageFileIdentity,
    ) -> Result<(), StorageError> {
        let metadata =
            std::fs::symlink_metadata(lock_path).map_err(|source| StorageError::Filesystem {
                operation: "verify writer lease",
                path: lock_path.to_path_buf(),
                source,
            })?;
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || storage_file_has_multiple_links(&metadata)
            || StorageFileIdentity::from_metadata(&metadata) != *identity
        {
            return Err(StorageError::InvalidData {
                context: "storage.writer_lease",
                reason: format!("writer lease path {lock_path_display} changed during locked open"),
            });
        }
        Ok(())
    }
}

impl Drop for StorageWriterLease {
    fn drop(&mut self) {
        if let Err(error) = self.file.unlock() {
            warn!(
                path = %self.lock_path.display(),
                %error,
                "failed to explicitly release storage writer lease; closing the file will release it"
            );
        }
    }
}

struct ExistingStorageLease {
    _file: std::fs::File,
    path_identity: StorageFileIdentity,
    connection_identity: Option<FileIdentity>,
}

impl ExistingStorageLease {
    fn open(path: &str) -> Result<Self, StorageError> {
        let metadata =
            std::fs::symlink_metadata(path).map_err(|source| StorageError::Filesystem {
                operation: "inspect recovery database",
                path: PathBuf::from(path),
                source,
            })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(StorageError::InvalidData {
                context: "storage.recovery_path",
                reason: format!("recovery requires a non-symlink regular file at {path}"),
            });
        }
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW);
        }
        let file = options
            .open(path)
            .map_err(|source| StorageError::Filesystem {
                operation: "open recovery identity",
                path: PathBuf::from(path),
                source,
            })?;
        let file_metadata = file.metadata().map_err(|source| StorageError::Filesystem {
            operation: "inspect recovery identity",
            path: PathBuf::from(path),
            source,
        })?;
        if !file_metadata.is_file() {
            return Err(StorageError::InvalidData {
                context: "storage.recovery_path",
                reason: format!("recovery path {path} changed away from a regular file"),
            });
        }
        let connection_identity =
            FileIdentity::from_file(&file).map_err(|source| StorageError::Filesystem {
                operation: "identify opened recovery database",
                path: PathBuf::from(path),
                source,
            })?;
        let lease = Self {
            _file: file,
            path_identity: StorageFileIdentity::from_metadata(&file_metadata),
            connection_identity,
        };
        lease.verify_path(path)?;
        Ok(lease)
    }

    fn bind_connection(&self, connection: &Connection, path: &str) -> Result<(), StorageError> {
        let leased_identity = self.required_connection_identity()?;
        let connection_identity = connection
            .file_identity()?
            .ok_or(StorageError::InvalidData {
                context: "storage.recovery_identity",
                reason:
                    "the FrankenSQLite VFS cannot prove the identity of its open database handle"
                        .to_owned(),
            })?;
        if connection_identity != leased_identity {
            return Err(StorageError::InvalidData {
                context: "storage.recovery_identity",
                reason: format!(
                    "FrankenSQLite opened a different filesystem object than the leased recovery database at {path}"
                ),
            });
        }
        self.verify_path(path)
    }

    fn required_connection_identity(&self) -> Result<FileIdentity, StorageError> {
        self.connection_identity.ok_or(StorageError::InvalidData {
            context: "storage.recovery_identity",
            reason: "the platform cannot prove the identity of the leased database descriptor"
                .to_owned(),
        })
    }

    fn verify_path(&self, path: &str) -> Result<(), StorageError> {
        let metadata =
            std::fs::symlink_metadata(path).map_err(|source| StorageError::Filesystem {
                operation: "verify recovery identity",
                path: PathBuf::from(path),
                source,
            })?;
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || storage_file_has_multiple_links(&metadata)
            || StorageFileIdentity::from_metadata(&metadata) != self.path_identity
        {
            return Err(StorageError::InvalidData {
                context: "storage.recovery_path",
                reason: format!("recovery path {path} changed during validated open"),
            });
        }
        Ok(())
    }
}

/// FrankenSQLite-backed persistence sink with buffered writes.
pub struct Storage {
    path: String,
    conn: Option<Connection>,
    _path_lease: Option<StoragePathLease>,
    _writer_lease: Option<StorageWriterLease>,
    _existing_lease: Option<ExistingStorageLease>,
    terminally_failed: bool,
    buffer: StorageBuffer,
    buffered_outbox_ids: Vec<PersistenceBatchId>,
    next_batch_id: u64,
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
        Self::with_target_before_recovery_writer_open(target, tick, agent, event, metric, |_| {})
    }

    fn with_target_before_recovery_writer_open(
        target: StorageTarget,
        tick: usize,
        agent: usize,
        event: usize,
        metric: usize,
        before_recovery_writer_open: impl FnOnce(&str),
    ) -> Result<Self, StorageError> {
        target.prepare_for_open()?;
        let path = target.path().to_owned();
        let recover_existing = matches!(target, StorageTarget::RecoverExisting(_));
        let initialize_schema = !recover_existing;
        let mut path_lease = StoragePathLease::acquire(&path)?;
        // This OS-backed lease is the cross-process authority. It must precede every recovery
        // inspection and every writable database open below.
        let writer_lease = StorageWriterLease::acquire(&path)?;
        let existing_lease = match target {
            StorageTarget::Memory => None,
            StorageTarget::CreateNewFile(_) | StorageTarget::RecoverExisting(_) => {
                Some(ExistingStorageLease::open(&path)?)
            }
        };
        let recovery_identity = if recover_existing {
            let validation_connection = open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            let validation = existing_lease
                .as_ref()
                .ok_or(StorageError::InvalidData {
                    context: "storage.recovery_identity",
                    reason: "recovery opened without an identity lease".to_owned(),
                })
                .and_then(|lease| lease.bind_connection(&validation_connection, &path))
                .and_then(|()| Self::validate_existing_scriptbots_database(&validation_connection));
            let close_result = validation_connection
                .close_without_checkpoint()
                .map_err(StorageError::from);
            validation?;
            close_result?;
            let identity_lease = existing_lease.as_ref().ok_or(StorageError::InvalidData {
                context: "storage.recovery_identity",
                reason: "recovery opened without an identity lease".to_owned(),
            })?;
            identity_lease.verify_path(&path)?;
            let expected_identity = identity_lease.required_connection_identity()?;
            before_recovery_writer_open(&path);
            Some(expected_identity)
        } else {
            None
        };
        let conn = if let Some(expected_identity) = recovery_identity {
            Connection::open_existing_with_expected_identity(&path, expected_identity)?
        } else {
            Connection::open(&path)?
        };
        if recover_existing {
            let validation = existing_lease
                .as_ref()
                .ok_or(StorageError::InvalidData {
                    context: "storage.recovery_identity",
                    reason: "recovery opened without an identity lease".to_owned(),
                })
                .and_then(|lease| lease.bind_connection(&conn, &path))
                .and_then(|()| Self::validate_existing_scriptbots_database(&conn));
            if let Err(error) = validation {
                if let Err(close_error) = conn.close_without_checkpoint() {
                    warn!(
                        path,
                        %close_error,
                        "failed to close refused recovery connection without checkpoint"
                    );
                }
                return Err(error);
            }
        }
        if let Some(lease) = path_lease.as_mut() {
            lease.promote_existing(&path)?;
        }
        if let Some(lease) = &existing_lease {
            lease.verify_path(&path)?;
        }
        conn.execute("PRAGMA synchronous = FULL;")?;
        let mut storage = Self {
            path,
            conn: Some(conn),
            _path_lease: path_lease,
            _writer_lease: writer_lease,
            _existing_lease: existing_lease,
            terminally_failed: false,
            buffer: StorageBuffer::default(),
            buffered_outbox_ids: Vec::new(),
            next_batch_id: 1,
            tick_flush_threshold: tick,
            agent_flush_threshold: agent,
            event_flush_threshold: event,
            metric_flush_threshold: metric,
            birth_flush_threshold: DEFAULT_LIFECYCLE_BUFFER,
            death_flush_threshold: DEFAULT_LIFECYCLE_BUFFER,
            replay_flush_threshold: DEFAULT_REPLAY_BUFFER,
        };
        if initialize_schema && let Err(error) = storage.initialize_schema() {
            storage.terminally_failed = true;
            return Err(error);
        }
        if let Err(error) = storage.validate_persistence_invariants() {
            storage.terminally_failed = true;
            return Err(error);
        }
        if let Err(error) = storage.recover_outbox() {
            storage.terminally_failed = true;
            return Err(error);
        }
        storage.next_batch_id =
            storage
                .persistence_watermarks()?
                .admitted
                .map_or(Ok(1), |batch_id| {
                    batch_id
                        .get()
                        .checked_add(1)
                        .ok_or(StorageError::InvalidData {
                            context: "storage_progress.admitted_batch_id",
                            reason: "batch id space exhausted".to_owned(),
                        })
                })?;
        Ok(storage)
    }

    fn validate_existing_scriptbots_database(connection: &Connection) -> Result<(), StorageError> {
        let migrations = connection.query(
            "SELECT version, name FROM _schema_migrations
             ORDER BY version ASC",
        )?;
        if migrations.len() != 2 {
            return Err(StorageError::InvalidData {
                context: "storage.recovery_schema",
                reason: format!(
                    "expected exactly two ScriptBots migrations, found {}",
                    migrations.len()
                ),
            });
        }
        let expected_migrations = [
            (1_i64, "create_scriptbots_schema"),
            (2_i64, "create_durable_persistence_outbox"),
        ];
        for (row, (version, name)) in migrations.iter().zip(expected_migrations) {
            let actual_version: i64 = decode(row, 0, "_schema_migrations.version")?;
            let actual_name: String = decode(row, 1, "_schema_migrations.name")?;
            if actual_version != version || actual_name != name {
                return Err(StorageError::InvalidData {
                    context: "storage.recovery_schema",
                    reason: format!(
                        "unexpected migration ({actual_version}, {actual_name:?}); expected ({version}, {name:?})"
                    ),
                });
            }
        }

        let expected_schema = canonical_schema_objects()?;
        let actual_schema = read_schema_objects(connection)?;
        if actual_schema != expected_schema {
            let first_difference = expected_schema
                .iter()
                .zip(&actual_schema)
                .position(|(expected, actual)| expected != actual);
            let difference = first_difference.map_or_else(
                || {
                    format!(
                        "object count differs: expected {}, found {}",
                        expected_schema.len(),
                        actual_schema.len()
                    )
                },
                |index| {
                    format!(
                        "first difference at object {index}: expected [{}], found [{}]",
                        expected_schema[index].summary(),
                        actual_schema[index].summary()
                    )
                },
            );
            return Err(StorageError::InvalidData {
                context: "storage.recovery_schema",
                reason: format!(
                    "schema fingerprint mismatch: expected {}, found {}; {difference}",
                    schema_fingerprint(&expected_schema),
                    schema_fingerprint(&actual_schema)
                ),
            });
        }

        let progress = connection.query_row(
            "SELECT admitted_batch_id, applied_batch_id, durable_batch_id
             FROM storage_progress
             WHERE singleton = 1",
        )?;
        PersistenceWatermarks::from_raw(
            decode(&progress, 0, "storage_progress.admitted_batch_id")?,
            decode(&progress, 1, "storage_progress.applied_batch_id")?,
            decode(&progress, 2, "storage_progress.durable_batch_id")?,
        )?;
        Ok(())
    }

    fn initialize_schema(&mut self) -> Result<(), StorageError> {
        install_scriptbots_schema(self.connection()?)
    }

    fn validate_persistence_invariants(&self) -> Result<(), StorageError> {
        let watermarks = self.persistence_watermarks()?;
        let admitted = watermarks.admitted_raw();
        let applied = watermarks.applied_raw();
        let durable = watermarks.durable_raw();
        let ledger = self.connection()?.query_row(
            "SELECT COUNT(*), MIN(batch_id), MAX(batch_id), COUNT(DISTINCT tick)
             FROM storage_batch_ledger",
        )?;
        let ledger_count: i64 = decode(&ledger, 0, "storage_batch_ledger.count")?;
        let ledger_min: Option<i64> = decode(&ledger, 1, "storage_batch_ledger.min_batch_id")?;
        let ledger_max: Option<i64> = decode(&ledger, 2, "storage_batch_ledger.max_batch_id")?;
        let distinct_ticks: i64 = decode(&ledger, 3, "storage_batch_ledger.distinct_ticks")?;
        let expected_min = (admitted > 0).then_some(1);
        let expected_max = (admitted > 0).then_some(admitted);
        if ledger_count != admitted
            || ledger_min != expected_min
            || ledger_max != expected_max
            || distinct_ticks != admitted
        {
            return Err(StorageError::InvalidData {
                context: "storage_batch_ledger",
                reason: format!(
                    "admitted={admitted} requires count/min/max/distinct_ticks={admitted}/{expected_min:?}/{expected_max:?}/{admitted}, found {ledger_count}/{ledger_min:?}/{ledger_max:?}/{distinct_ticks}"
                ),
            });
        }
        for (expected_state, lower_exclusive, upper_inclusive) in [
            ("durable", 0_i64, durable),
            ("applied", durable, applied),
            ("admitted", applied, admitted),
        ] {
            let mismatches = self.connection()?.query_with_params(
                "SELECT batch_id, state
                 FROM storage_batch_ledger
                 WHERE batch_id > ?1 AND batch_id <= ?2 AND state != ?3
                 ORDER BY batch_id ASC
                 LIMIT 1",
                &[
                    lower_exclusive.into(),
                    upper_inclusive.into(),
                    expected_state.into(),
                ],
            )?;
            if let Some(row) = mismatches.first() {
                let batch_id: i64 = decode(row, 0, "storage_batch_ledger.batch_id")?;
                let state: String = decode(row, 1, "storage_batch_ledger.state")?;
                return Err(StorageError::InvalidData {
                    context: "storage_batch_ledger.state",
                    reason: format!(
                        "batch {batch_id} is {state:?}; watermarks require {expected_state:?}"
                    ),
                });
            }
        }

        if self.file_backed() {
            let outbox = self.connection()?.query_row(
                "SELECT COUNT(*), MIN(batch_id), MAX(batch_id)
                 FROM storage_outbox",
            )?;
            let outbox_count: i64 = decode(&outbox, 0, "storage_outbox.count")?;
            let outbox_min: Option<i64> = decode(&outbox, 1, "storage_outbox.min_batch_id")?;
            let outbox_max: Option<i64> = decode(&outbox, 2, "storage_outbox.max_batch_id")?;
            let expected_count = admitted - durable;
            let expected_min = (expected_count > 0).then_some(durable + 1);
            let expected_max = (expected_count > 0).then_some(admitted);
            if outbox_count != expected_count
                || outbox_min != expected_min
                || outbox_max != expected_max
            {
                return Err(StorageError::InvalidData {
                    context: "storage_outbox.batch_id",
                    reason: format!(
                        "watermarks require outbox count/min/max={expected_count}/{expected_min:?}/{expected_max:?}, found {outbox_count}/{outbox_min:?}/{outbox_max:?}"
                    ),
                });
            }
        }
        Ok(())
    }

    fn connection(&self) -> Result<&Connection, StorageError> {
        self.conn.as_ref().ok_or(StorageError::Closed)
    }

    fn file_backed(&self) -> bool {
        self.path != ":memory:"
    }

    /// Return the persisted monotonic admission/application/durability prefixes.
    pub fn persistence_watermarks(&self) -> Result<PersistenceWatermarks, StorageError> {
        let row = self.connection()?.query_row(
            "SELECT admitted_batch_id, applied_batch_id, durable_batch_id
             FROM storage_progress
             WHERE singleton = 1",
        )?;
        PersistenceWatermarks::from_raw(
            decode(&row, 0, "storage_progress.admitted_batch_id")?,
            decode(&row, 1, "storage_progress.applied_batch_id")?,
            decode(&row, 2, "storage_progress.durable_batch_id")?,
        )
    }

    /// Query one compact batch-ledger entry without exposing the outbox payload.
    pub fn batch_status(
        &self,
        batch_id: PersistenceBatchId,
    ) -> Result<Option<PersistedBatchStatus>, StorageError> {
        let rows = self.connection()?.query_with_params(
            "SELECT tick, payload_digest, state
             FROM storage_batch_ledger
             WHERE batch_id = ?1",
            &[batch_id.as_i64().into()],
        )?;
        rows.first()
            .map(|row| {
                let tick = checked_u64(
                    "storage_batch_ledger.tick",
                    decode(row, 0, "storage_batch_ledger.tick")?,
                )?;
                let payload_digest = decode(row, 1, "storage_batch_ledger.payload_digest")?;
                let state_text: String = decode(row, 2, "storage_batch_ledger.state")?;
                Ok(PersistedBatchStatus {
                    batch_id,
                    tick,
                    payload_digest,
                    state: decode_batch_state(&state_text)?,
                })
            })
            .transpose()
    }

    fn stage_outbox(
        &mut self,
        tick: u64,
        prepared: &StorageBuffer,
    ) -> Result<(AdmissionReceipt, bool), StorageError> {
        let tick_i64 = i64::try_from(tick).map_err(|error| StorageError::InvalidData {
            context: "storage_batch_ledger.tick",
            reason: error.to_string(),
        })?;
        let (payload, payload_digest) = prepared.encode_outbox(tick)?;
        let before = self.persistence_watermarks()?;
        let existing = self.connection()?.query_with_params(
            "SELECT batch_id, payload_digest
             FROM storage_batch_ledger
             WHERE tick = ?1
             ORDER BY batch_id ASC",
            &[tick_i64.into()],
        )?;
        if let Some(row) = existing.first() {
            if existing.len() != 1 {
                return Err(StorageError::InvalidData {
                    context: "storage_batch_ledger.tick",
                    reason: format!("tick {tick} has {} ledger entries", existing.len()),
                });
            }
            let batch_id = PersistenceBatchId::new(checked_u64(
                "storage_batch_ledger.batch_id",
                decode(row, 0, "storage_batch_ledger.batch_id")?,
            )?)?;
            let existing_digest: String = decode(row, 1, "storage_batch_ledger.payload_digest")?;
            if existing_digest != payload_digest {
                return Err(StorageError::InvalidData {
                    context: "storage_batch_ledger.payload_digest",
                    reason: format!(
                        "tick {tick} was already admitted as batch {} with a different payload",
                        batch_id.get()
                    ),
                });
            }
            return Ok((
                AdmissionReceipt {
                    batch_id,
                    tick,
                    guarantee: if self.file_backed() {
                        PersistenceGuarantee::Durable
                    } else {
                        PersistenceGuarantee::CommittedVolatile
                    },
                    watermarks: before,
                },
                false,
            ));
        }

        let batch_id = PersistenceBatchId::new(self.next_batch_id)?;
        let expected_previous = batch_id.as_i64() - 1;
        if before.admitted_raw() != expected_previous {
            return Err(StorageError::InvalidData {
                context: "storage_progress.admitted_batch_id",
                reason: format!(
                    "next batch {} does not follow admitted watermark {}",
                    batch_id.get(),
                    before.admitted_raw()
                ),
            });
        }

        execute_transaction_with_retry(self.connection()?, |transaction| {
            let ledger_rows = transaction.execute_with_params(
                "INSERT INTO storage_batch_ledger (
                    batch_id, tick, payload_digest, state
                 ) VALUES (?1, ?2, ?3, 'admitted')",
                &[
                    batch_id.as_i64().into(),
                    tick_i64.into(),
                    payload_digest.as_str().into(),
                ],
            )?;
            if ledger_rows != 1 {
                return Err(FrankenError::Internal(format!(
                    "admission inserted {ledger_rows} ledger rows for batch {}",
                    batch_id.get()
                )));
            }
            let outbox_rows = transaction.execute_with_params(
                "INSERT INTO storage_outbox (batch_id, payload) VALUES (?1, ?2)",
                &[batch_id.as_i64().into(), payload.as_str().into()],
            )?;
            if outbox_rows != 1 {
                return Err(FrankenError::Internal(format!(
                    "admission inserted {outbox_rows} outbox rows for batch {}",
                    batch_id.get()
                )));
            }
            let progress_rows = transaction.execute_with_params(
                "UPDATE storage_progress
                 SET admitted_batch_id = ?1
                 WHERE singleton = 1 AND admitted_batch_id = ?2",
                &[batch_id.as_i64().into(), expected_previous.into()],
            )?;
            if progress_rows != 1 {
                return Err(FrankenError::Internal(format!(
                    "admission CAS updated {progress_rows} progress rows for batch {}",
                    batch_id.get()
                )));
            }
            Ok(())
        })?;

        self.next_batch_id = batch_id
            .get()
            .checked_add(1)
            .ok_or(StorageError::InvalidData {
                context: "storage_progress.admitted_batch_id",
                reason: "batch id space exhausted".to_owned(),
            })?;
        let watermarks = PersistenceWatermarks {
            admitted: Some(batch_id),
            ..before
        };
        Ok((
            AdmissionReceipt {
                batch_id,
                tick,
                guarantee: if self.file_backed() {
                    PersistenceGuarantee::Durable
                } else {
                    PersistenceGuarantee::CommittedVolatile
                },
                watermarks,
            },
            true,
        ))
    }

    fn load_outbox(&self) -> Result<Vec<RecoveredOutboxBatch>, StorageError> {
        let rows = self.connection()?.query(
            "SELECT outbox.batch_id, ledger.tick, ledger.payload_digest, outbox.payload
             FROM storage_outbox AS outbox
             JOIN storage_batch_ledger AS ledger ON ledger.batch_id = outbox.batch_id
             ORDER BY outbox.batch_id ASC",
        )?;
        let mut batches = Vec::with_capacity(rows.len());
        for row in rows {
            let batch_id = PersistenceBatchId::new(checked_u64(
                "storage_outbox.batch_id",
                decode(&row, 0, "storage_outbox.batch_id")?,
            )?)?;
            let tick = checked_u64(
                "storage_batch_ledger.tick",
                decode(&row, 1, "storage_batch_ledger.tick")?,
            )?;
            let payload_digest: String = decode(&row, 2, "storage_batch_ledger.payload_digest")?;
            let payload: String = decode(&row, 3, "storage_outbox.payload")?;
            let storage = StorageBuffer::decode_outbox(&payload, tick, &payload_digest)?;
            batches.push(RecoveredOutboxBatch {
                batch_id,
                tick,
                payload_digest,
                storage,
            });
        }
        Ok(batches)
    }

    fn recover_outbox(&mut self) -> Result<(), StorageError> {
        let before = self.persistence_watermarks()?;
        let batches = self.load_outbox()?;
        let applied = before.applied_raw();
        let admitted = before.admitted_raw();
        let mut next_unapplied = applied + 1;
        let durable = before.durable_raw();
        let mut highest_outbox = durable;

        for (offset, batch) in batches.into_iter().enumerate() {
            let offset = i64::try_from(offset).map_err(|error| StorageError::InvalidData {
                context: "storage_outbox.batch_id",
                reason: error.to_string(),
            })?;
            let next_outbox = durable
                .checked_add(offset)
                .and_then(|value| value.checked_add(1))
                .ok_or(StorageError::InvalidData {
                    context: "storage_outbox.batch_id",
                    reason: "outbox sequence overflow".to_owned(),
                })?;
            let raw_id = batch.batch_id.as_i64();
            if raw_id != next_outbox {
                return Err(StorageError::InvalidData {
                    context: "storage_outbox.batch_id",
                    reason: format!("outbox gap: expected batch {next_outbox}, found {raw_id}"),
                });
            }
            highest_outbox = highest_outbox.max(raw_id);
            if raw_id <= applied {
                continue;
            }
            if raw_id != next_unapplied {
                return Err(StorageError::InvalidData {
                    context: "storage_outbox.batch_id",
                    reason: format!("outbox gap: expected batch {next_unapplied}, found {raw_id}"),
                });
            }
            debug_assert!(!batch.payload_digest.is_empty());
            debug_assert_eq!(
                batch.storage.ticks.last().map(|row| row.tick as u64),
                Some(batch.tick)
            );
            self.buffer.append(batch.storage);
            self.buffered_outbox_ids.push(batch.batch_id);
            next_unapplied += 1;
        }

        if admitted > durable && highest_outbox != admitted {
            return Err(StorageError::InvalidData {
                context: "storage_outbox",
                reason: format!(
                    "admitted watermark {admitted} has no complete outbox prefix (highest {highest_outbox})"
                ),
            });
        }
        self.flush()?;
        self.finalize_applied_outbox()?;
        Ok(())
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

    fn enqueue_staged(
        &mut self,
        batch_id: PersistenceBatchId,
        prepared: StorageBuffer,
    ) -> Result<bool, StorageError> {
        if self.terminally_failed {
            return Err(StorageError::TerminallyFailed);
        }
        self.buffer.append(prepared);
        self.buffered_outbox_ids.push(batch_id);
        self.maybe_flush()
    }

    /// Durably admit and apply a simulation payload through the same outbox protocol as the
    /// asynchronous worker, buffering scientific rows until thresholds or an explicit flush.
    pub fn persist(&mut self, payload: &PersistenceBatch) -> Result<(), StorageError> {
        if self.terminally_failed {
            return Err(StorageError::TerminallyFailed);
        }
        let tick = payload.summary.tick.0;
        let prepared = Self::prepare_batch(payload)?;
        let (receipt, newly_admitted) = self.stage_outbox(tick, &prepared)?;
        if newly_admitted {
            self.enqueue_staged(receipt.batch_id, prepared)?;
        }
        Ok(())
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
        outbox_ids: &[PersistenceBatchId],
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
            if let Some(last) = outbox_ids.last().copied() {
                let first = outbox_ids[0];
                for pair in outbox_ids.windows(2) {
                    if pair[1].get() != pair[0].get() + 1 {
                        return Err(FrankenError::Internal(format!(
                            "non-contiguous outbox application {} then {}",
                            pair[0].get(),
                            pair[1].get()
                        )));
                    }
                }
                let progress = tx.query_row(
                    "SELECT admitted_batch_id, applied_batch_id FROM storage_progress
                     WHERE singleton = 1",
                )?;
                let admitted = progress
                    .get_typed::<i64>(0)
                    .map_err(|error| FrankenError::Internal(error.to_string()))?;
                let applied = progress
                    .get_typed::<i64>(1)
                    .map_err(|error| FrankenError::Internal(error.to_string()))?;
                if applied != first.as_i64() - 1 || admitted < last.as_i64() {
                    return Err(FrankenError::Internal(format!(
                        "outbox application prefix mismatch: admitted={admitted}, applied={applied}, applying={}..={}",
                        first.get(),
                        last.get()
                    )));
                }
                for batch_id in outbox_ids {
                    let ledger = tx.query_row_with_params(
                        "SELECT state FROM storage_batch_ledger WHERE batch_id = ?1",
                        &[batch_id.as_i64().into()],
                    )?;
                    let state = ledger
                        .get_typed::<String>(0)
                        .map_err(|error| FrankenError::Internal(error.to_string()))?;
                    if state != "admitted" {
                        return Err(FrankenError::Internal(format!(
                            "batch {} cannot apply from state {state:?}",
                            batch_id.get()
                        )));
                    }
                    let updated = tx.execute_with_params(
                        "UPDATE storage_batch_ledger SET state = 'applied' WHERE batch_id = ?1",
                        &[batch_id.as_i64().into()],
                    )?;
                    if updated != 1 {
                        return Err(FrankenError::Internal(format!(
                            "application updated {updated} ledger rows for batch {}",
                            batch_id.get()
                        )));
                    }
                }
                let progress_rows = tx.execute_with_params(
                    "UPDATE storage_progress SET applied_batch_id = ?1 WHERE singleton = 1",
                    &[last.as_i64().into()],
                )?;
                if progress_rows != 1 {
                    return Err(FrankenError::Internal(format!(
                        "application updated {progress_rows} progress rows through batch {}",
                        last.get()
                    )));
                }
            }
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
            match Self::flush_attempt(connection, &self.buffer, &self.buffered_outbox_ids) {
                Ok(()) => {
                    info!(
                        path = %self.path,
                        attempt,
                        rows = self.buffer.ticks.len(),
                        batches = self.buffered_outbox_ids.len(),
                        "FrankenSQLite storage transaction committed"
                    );
                    self.buffer.clear();
                    self.buffered_outbox_ids.clear();
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

    fn finalize_applied_outbox(&mut self) -> Result<PersistenceWatermarks, StorageError> {
        let before = self.persistence_watermarks()?;
        let Some(applied) = before.applied else {
            return Ok(before);
        };
        let durable_before = before.durable_raw();
        let target = applied.as_i64();
        let file_backed = self.file_backed();
        execute_transaction_with_retry(self.connection()?, |transaction| {
            if file_backed && target > durable_before {
                let rows = transaction.query_with_params(
                    "SELECT batch_id, state
                     FROM storage_batch_ledger
                     WHERE batch_id > ?1 AND batch_id <= ?2
                     ORDER BY batch_id ASC",
                    &[durable_before.into(), target.into()],
                )?;
                if rows.len() != usize::try_from(target - durable_before).unwrap_or(usize::MAX) {
                    return Err(FrankenError::Internal(format!(
                        "durability ledger gap for batches {}..={target}",
                        durable_before + 1
                    )));
                }
                for row in rows {
                    let batch_id = row
                        .get_typed::<i64>(0)
                        .map_err(|error| FrankenError::Internal(error.to_string()))?;
                    let state = row
                        .get_typed::<String>(1)
                        .map_err(|error| FrankenError::Internal(error.to_string()))?;
                    if state != "applied" {
                        return Err(FrankenError::Internal(format!(
                            "batch {batch_id} cannot become durable from state {state:?}"
                        )));
                    }
                }
                let expected = usize::try_from(target - durable_before).unwrap_or(usize::MAX);
                let ledger_rows = transaction.execute_with_params(
                    "UPDATE storage_batch_ledger
                     SET state = 'durable'
                     WHERE batch_id > ?1 AND batch_id <= ?2",
                    &[durable_before.into(), target.into()],
                )?;
                if ledger_rows != expected {
                    return Err(FrankenError::Internal(format!(
                        "durability updated {ledger_rows} ledger rows; expected {expected}"
                    )));
                }
                let progress_rows = transaction.execute_with_params(
                    "UPDATE storage_progress SET durable_batch_id = ?1 WHERE singleton = 1",
                    &[target.into()],
                )?;
                if progress_rows != 1 {
                    return Err(FrankenError::Internal(format!(
                        "durability updated {progress_rows} progress rows through batch {target}"
                    )));
                }
            }
            transaction.execute_with_params(
                "DELETE FROM storage_outbox WHERE batch_id <= ?1",
                &[target.into()],
            )?;
            Ok(())
        })?;
        self.persistence_watermarks()
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
        self.finalize_applied_outbox()?;
        let connection = self.conn.take().ok_or(StorageError::Closed)?;
        connection.close()?;
        Ok(())
    }

    /// Dispose of a terminally failed worker without replaying its buffered transaction in Drop.
    fn abandon_after_error(mut self) {
        self.buffer.clear();
        self.buffered_outbox_ids.clear();
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

    fn latest_pending_analytics(&self) -> Result<Option<PendingAnalytics>, StorageError> {
        let rows = self
            .connection()?
            .query("SELECT tick, agent_count FROM ticks ORDER BY tick DESC LIMIT 1")?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let tick = checked_u64("ticks.tick", decode(row, 0, "ticks.tick")?)?;
        let agent_count = checked_usize("ticks.agent_count", decode(row, 1, "ticks.agent_count")?)?;
        let metric_rows = self.connection()?.query_with_params(
            "SELECT name, value FROM metrics WHERE tick = ?1 ORDER BY name ASC",
            &[i64::try_from(tick)
                .map_err(|error| StorageError::InvalidData {
                    context: "ticks.tick",
                    reason: error.to_string(),
                })?
                .into()],
        )?;
        let mut readings = Vec::with_capacity(metric_rows.len());
        for row in metric_rows {
            readings.push(MetricReading {
                tick: tick as i64,
                name: decode(&row, 0, "metrics.name")?,
                value: decode(&row, 1, "metrics.value")?,
            });
        }
        Ok(Some(PendingAnalytics {
            tick,
            agent_count,
            readings: Arc::from(readings),
        }))
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if self.conn.is_none() {
            return;
        }
        if self.terminally_failed {
            self.buffer.clear();
            self.buffered_outbox_ids.clear();
        } else if let Err(err) = self
            .flush()
            .and_then(|()| self.finalize_applied_outbox().map(|_| ()))
        {
            eprintln!("failed to finalize persistence buffer on drop: {err}");
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
    Persist {
        batch: Box<PreparedPersistenceBatch>,
        reply: xchan::Sender<Result<AdmissionReceipt, StorageWorkerError>>,
    },
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

#[cfg(test)]
struct StartupPause {
    entered: xchan::Sender<()>,
    release: xchan::Receiver<()>,
}

#[cfg(test)]
static STARTUP_PAUSES: OnceLock<Mutex<BTreeMap<String, StartupPause>>> = OnceLock::new();

#[cfg(test)]
fn register_startup_pause(path: &str) -> (xchan::Receiver<()>, xchan::Sender<()>) {
    let (entered_tx, entered_rx) = xchan::bounded(1);
    let (release_tx, release_rx) = xchan::bounded(1);
    let pauses = STARTUP_PAUSES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut pauses = match pauses.lock() {
        Ok(pauses) => pauses,
        Err(poisoned) => poisoned.into_inner(),
    };
    pauses.insert(
        path.to_owned(),
        StartupPause {
            entered: entered_tx,
            release: release_rx,
        },
    );
    (entered_rx, release_tx)
}

#[cfg(test)]
fn maybe_pause_storage_startup(path: &str) {
    let pauses = STARTUP_PAUSES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let pause = {
        let mut pauses = match pauses.lock() {
            Ok(pauses) => pauses,
            Err(poisoned) => poisoned.into_inner(),
        };
        pauses.remove(path)
    };
    if let Some(pause) = pause {
        let _ = pause.entered.send(());
        let _ = pause.release.recv();
    }
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
    pub watermarks: PersistenceWatermarks,
    pub analytics_revision: u64,
}

/// Proof that the worker flushed, closed, and joined with an explicit persistence guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShutdownReceipt {
    pub committed_tick: Option<u64>,
    pub guarantee: PersistenceGuarantee,
    pub watermarks: PersistenceWatermarks,
    pub analytics_revision: u64,
}

#[derive(Default)]
struct WorkerState {
    admitted_tick: Option<u64>,
    committed_tick: Option<u64>,
    guarantee: PersistenceGuarantee,
    watermarks: PersistenceWatermarks,
    pending_analytics: Vec<(PersistenceBatchId, PendingAnalytics)>,
}

#[derive(Clone, Copy)]
struct StorageThresholds {
    tick: usize,
    agent: usize,
    event: usize,
    metric: usize,
}

/// Bounded controller waits around the synchronous storage worker.
///
/// A deadline bounds the caller; it cannot cancel a FrankenSQLite call already executing on the
/// connection-owning worker thread. Timed-out admissions therefore remain retryable with their
/// exact payload, while timed-out shutdowns retain worker ownership for retry or supervised reap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageDeadlines {
    pub startup_ack: Duration,
    pub command_enqueue: Duration,
    pub admission_ack: Duration,
    pub flush_ack: Duration,
    pub shutdown_ack: Duration,
}

impl Default for StorageDeadlines {
    fn default() -> Self {
        Self {
            startup_ack: DEFAULT_STARTUP_ACK_TIMEOUT,
            command_enqueue: DEFAULT_COMMAND_ENQUEUE_TIMEOUT,
            admission_ack: DEFAULT_ADMISSION_ACK_TIMEOUT,
            flush_ack: DEFAULT_FLUSH_ACK_TIMEOUT,
            shutdown_ack: DEFAULT_SHUTDOWN_ACK_TIMEOUT,
        }
    }
}

impl StorageDeadlines {
    fn validate(self) -> Result<(), StorageError> {
        for (name, timeout) in [
            ("startup_ack", self.startup_ack),
            ("command_enqueue", self.command_enqueue),
            ("admission_ack", self.admission_ack),
            ("flush_ack", self.flush_ack),
            ("shutdown_ack", self.shutdown_ack),
        ] {
            if timeout.is_zero() || timeout > MAX_STORAGE_WAIT_TIMEOUT {
                return Err(StorageError::InvalidData {
                    context: "storage.deadlines",
                    reason: format!(
                        "{name} must be within 1ns..={MAX_STORAGE_WAIT_TIMEOUT:?}, got {timeout:?}"
                    ),
                });
            }
        }
        Ok(())
    }
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
    deadlines: StorageDeadlines,
}

#[derive(Clone, Copy)]
struct AdmissionGateWait {
    operation: StorageOperation,
    tick: Option<u64>,
    commit_state: FailureCommitState,
    deadline: Instant,
    waited: Duration,
    recover_poison: bool,
}

fn lock_admission_gate_until<'a>(
    admission: &'a Mutex<AdmissionState>,
    path: &str,
    wait: AdmissionGateWait,
) -> Result<std::sync::MutexGuard<'a, AdmissionState>, StorageWorkerError> {
    loop {
        match admission.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(std::sync::TryLockError::Poisoned(poisoned)) if wait.recover_poison => {
                warn!(operation = ?wait.operation, "recovering poisoned storage admission gate");
                return Ok(poisoned.into_inner());
            }
            Err(std::sync::TryLockError::Poisoned(error)) => {
                return Err(StorageWorkerError::Internal {
                    operation: wait.operation,
                    path: path.to_owned(),
                    tick: wait.tick,
                    commit_state: wait.commit_state,
                    detail: format!("storage admission gate is poisoned: {error}"),
                });
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                let now = Instant::now();
                if now >= wait.deadline {
                    return Err(StorageWorkerError::Timeout {
                        operation: wait.operation,
                        phase: StorageWaitPhase::AdmissionGate,
                        path: path.to_owned(),
                        tick: wait.tick,
                        waited: wait.waited,
                        commit_state: wait.commit_state,
                    });
                }
                thread::sleep(Duration::from_millis(1).min(wait.deadline.duration_since(now)));
            }
        }
    }
}

impl StorageSink {
    /// Admit a persistence batch and wait until its exact payload is in the worker outbox.
    pub fn submit_with_receipt(
        &self,
        payload: &PersistenceBatch,
    ) -> Result<AdmissionReceipt, StorageError> {
        let tick = payload.summary.tick.0;
        let prepared = PreparedPersistenceBatch::from_batch(payload).inspect_err(|error| {
            let worker_error = StorageWorkerError::Internal {
                operation: StorageOperation::Admit,
                path: self.path.to_string(),
                tick: Some(tick),
                commit_state: FailureCommitState::NotAdmitted,
                detail: error.to_string(),
            };
            self.analytics.publish_worker_error(&worker_error, false);
        })?;
        let enqueue_deadline = Instant::now() + self.deadlines.command_enqueue;
        let admission = lock_admission_gate_until(
            &self.admission,
            &self.path,
            AdmissionGateWait {
                operation: StorageOperation::Admit,
                tick: Some(tick),
                commit_state: FailureCommitState::NotAdmitted,
                deadline: enqueue_deadline,
                waited: self.deadlines.command_enqueue,
                recover_poison: false,
            },
        )
        .map_err(|worker_error| {
            self.analytics.publish_worker_error(&worker_error, false);
            StorageError::Worker(worker_error)
        })?;
        if !admission.open {
            let worker_error = StorageWorkerError::Channel {
                operation: StorageOperation::Admit,
                path: self.path.to_string(),
                tick: Some(tick),
                commit_state: FailureCommitState::NotAdmitted,
                detail: "storage pipeline is closing or closed".to_owned(),
            };
            self.analytics.publish_worker_error(&worker_error, true);
            return Err(StorageError::Worker(worker_error));
        }

        let (reply_tx, reply_rx) = xchan::bounded(1);
        let send_result = self.tx.send_deadline(
            StorageCommand::Persist {
                batch: Box::new(prepared),
                reply: reply_tx,
            },
            enqueue_deadline,
        );
        drop(admission);
        match send_result {
            Ok(()) => {}
            Err(xchan::SendTimeoutError::Timeout(_)) => {
                let worker_error = StorageWorkerError::Timeout {
                    operation: StorageOperation::Admit,
                    phase: StorageWaitPhase::CommandEnqueue,
                    path: self.path.to_string(),
                    tick: Some(tick),
                    waited: self.deadlines.command_enqueue,
                    commit_state: FailureCommitState::NotAdmitted,
                };
                self.analytics.publish_worker_error(&worker_error, false);
                return Err(StorageError::Worker(worker_error));
            }
            Err(xchan::SendTimeoutError::Disconnected(_)) => {
                let worker_error = StorageWorkerError::Channel {
                    operation: StorageOperation::Admit,
                    path: self.path.to_string(),
                    tick: Some(tick),
                    commit_state: FailureCommitState::NotAdmitted,
                    detail: "storage worker command channel is disconnected".to_owned(),
                };
                self.analytics.publish_worker_error(&worker_error, true);
                return Err(StorageError::Worker(worker_error));
            }
        }
        reply_rx
            .recv_deadline(Instant::now() + self.deadlines.admission_ack)
            .map_err(|error| {
                let (worker_error, stopped) = match error {
                    xchan::RecvTimeoutError::Timeout => (
                        StorageWorkerError::Timeout {
                            operation: StorageOperation::Admit,
                            phase: StorageWaitPhase::Acknowledgement,
                            path: self.path.to_string(),
                            tick: Some(tick),
                            waited: self.deadlines.admission_ack,
                            commit_state: FailureCommitState::Indeterminate,
                        },
                        false,
                    ),
                    xchan::RecvTimeoutError::Disconnected => (
                        StorageWorkerError::Channel {
                            operation: StorageOperation::Admit,
                            path: self.path.to_string(),
                            tick: Some(tick),
                            commit_state: FailureCommitState::Indeterminate,
                            detail: "storage worker exited before durable outbox acknowledgement"
                                .to_owned(),
                        },
                        true,
                    ),
                };
                self.analytics.publish_worker_error(&worker_error, stopped);
                StorageError::Worker(worker_error)
            })?
            .map_err(StorageError::Worker)
    }

    /// Admit a persistence batch while discarding the returned batch identifier.
    pub fn submit(&self, payload: &PersistenceBatch) -> Result<(), StorageError> {
        self.submit_with_receipt(payload).map(|_| ())
    }
}

impl WorldPersistence for StorageSink {
    fn on_tick(&mut self, payload: &PersistenceBatch) -> Result<(), PersistenceAdmissionError> {
        self.submit(payload).map_err(|error| {
            let state = match &error {
                StorageError::Worker(worker_error)
                    if matches!(
                        worker_error.status().commit_state,
                        FailureCommitState::Indeterminate | FailureCommitState::Committed
                    ) =>
                {
                    PersistenceAdmissionState::Indeterminate
                }
                _ => PersistenceAdmissionState::NotAdmitted,
            };
            match state {
                PersistenceAdmissionState::NotAdmitted => {
                    PersistenceAdmissionError::new(payload.summary.tick.0, error.to_string())
                }
                PersistenceAdmissionState::Indeterminate => {
                    PersistenceAdmissionError::indeterminate(
                        payload.summary.tick.0,
                        error.to_string(),
                    )
                }
            }
        })
    }
}

type ShutdownReply = Result<ShutdownReceipt, StorageWorkerError>;
type ShutdownReplyReceiver = xchan::Receiver<ShutdownReply>;

enum StorageReapRequest {
    Pipeline {
        tx: xchan::Sender<StorageCommand>,
        admission: Arc<Mutex<AdmissionState>>,
        pending_shutdown: Option<ShutdownReplyReceiver>,
        handle: thread::JoinHandle<Option<StorageWorkerError>>,
        path: Arc<str>,
        analytics: AnalyticsSnapshotProvider,
    },
    JoinOnly {
        handle: thread::JoinHandle<Option<StorageWorkerError>>,
        path: Arc<str>,
        analytics: AnalyticsSnapshotProvider,
    },
}

fn join_reaped_worker(
    handle: thread::JoinHandle<Option<StorageWorkerError>>,
    path: &str,
    analytics: &AnalyticsSnapshotProvider,
    response: Option<ShutdownReply>,
) {
    match handle.join() {
        Err(panic) => analytics.publish_worker_error(
            &StorageWorkerError::Internal {
                operation: StorageOperation::Join,
                path: path.to_owned(),
                tick: None,
                commit_state: FailureCommitState::Indeterminate,
                detail: format!("storage worker panicked during supervised reap: {panic:?}"),
            },
            true,
        ),
        Ok(Some(terminal_error)) => analytics.publish_worker_error(&terminal_error, true),
        Ok(None) => {
            if let Some(Err(error)) = response {
                analytics.publish_worker_error(&error, true);
            }
        }
    }
}

fn reap_storage_request(request: StorageReapRequest) {
    match request {
        StorageReapRequest::JoinOnly {
            handle,
            path,
            analytics,
        } => join_reaped_worker(handle, &path, &analytics, None),
        StorageReapRequest::Pipeline {
            tx,
            admission,
            pending_shutdown,
            handle,
            path,
            analytics,
        } => {
            let receiver = pending_shutdown.map_or_else(
                || -> Result<ShutdownReplyReceiver, StorageWorkerError> {
                    let mut gate = match admission.lock() {
                        Ok(gate) => gate,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    gate.open = false;
                    let (reply, receiver) = xchan::bounded(1);
                    let send_result = tx.send(StorageCommand::Shutdown { reply });
                    drop(gate);
                    match send_result {
                        Ok(()) => Ok(receiver),
                        Err(error) => Err(StorageWorkerError::Channel {
                            operation: StorageOperation::Shutdown,
                            path: path.to_string(),
                            tick: None,
                            commit_state: FailureCommitState::Indeterminate,
                            detail: format!(
                                "failed to enqueue supervised storage shutdown: {error}"
                            ),
                        }),
                    }
                },
                Ok,
            );
            // The bounded reply channel lets a healthy worker acknowledge before exiting. Join
            // first so a command stranded behind a terminal worker cannot keep its own reply
            // sender alive forever inside the disconnected command queue.
            match handle.join() {
                Err(panic) => analytics.publish_worker_error(
                    &StorageWorkerError::Internal {
                        operation: StorageOperation::Join,
                        path: path.to_string(),
                        tick: None,
                        commit_state: FailureCommitState::Indeterminate,
                        detail: format!(
                            "storage worker panicked during supervised reap: {panic:?}"
                        ),
                    },
                    true,
                ),
                Ok(Some(terminal_error)) => {
                    analytics.publish_worker_error(&terminal_error, true);
                }
                Ok(None) => {
                    let response = receiver.and_then(|receiver| {
                        receiver.try_recv().map_err(|error| StorageWorkerError::Channel {
                            operation: StorageOperation::Shutdown,
                            path: path.to_string(),
                            tick: None,
                            commit_state: FailureCommitState::Indeterminate,
                            detail: format!(
                                "storage worker exited before supervised shutdown acknowledgement: {error}"
                            ),
                        })
                    });
                    if let Err(error) = response.and_then(|response| response) {
                        analytics.publish_worker_error(&error, true);
                    }
                }
            }
        }
    }
}

fn handoff_storage_reap(request: StorageReapRequest) {
    let request = Arc::new(Mutex::new(Some(request)));
    let background = Arc::clone(&request);
    let spawned = thread::Builder::new()
        .name("scriptbots-storage-reaper".into())
        .spawn(move || {
            let request = match background.lock() {
                Ok(mut request) => request.take(),
                Err(poisoned) => poisoned.into_inner().take(),
            };
            if let Some(request) = request {
                reap_storage_request(request);
            }
        });
    if spawned.is_err() {
        let request = match request.lock() {
            Ok(mut request) => request.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if let Some(request) = request {
            reap_storage_request(request);
        }
    }
}

/// Host-owned controller for flush receipts, shutdown acknowledgement, and worker join.
pub struct StoragePipeline {
    sink: StorageSink,
    handle: Option<thread::JoinHandle<Option<StorageWorkerError>>>,
    pending_shutdown: Option<ShutdownReplyReceiver>,
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

    /// Open a validated existing ScriptBots database, recover its outbox, and own it exclusively.
    pub fn recover_existing(path: &str) -> Result<Self, StorageError> {
        validate_durable_storage_path(path)?;
        Self::with_target_and_deadlines(
            StorageTarget::RecoverExisting(path.to_owned()),
            DEFAULT_TICK_BUFFER,
            DEFAULT_AGENT_BUFFER,
            DEFAULT_EVENT_BUFFER,
            DEFAULT_METRIC_BUFFER,
            StorageDeadlines::default(),
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
        Self::with_target_and_deadlines(
            reserve_new_file(path)?,
            tick,
            agent,
            event,
            metric,
            StorageDeadlines::default(),
        )
    }

    /// Atomically reserve a file-backed pipeline with explicit thresholds and wait deadlines.
    pub fn create_new_file_with_thresholds_and_deadlines(
        path: &str,
        tick: usize,
        agent: usize,
        event: usize,
        metric: usize,
        deadlines: StorageDeadlines,
    ) -> Result<Self, StorageError> {
        Self::with_target_and_deadlines(
            reserve_new_file(path)?,
            tick,
            agent,
            event,
            metric,
            deadlines,
        )
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
        Self::with_target_and_deadlines(
            StorageTarget::Memory,
            tick,
            agent,
            event,
            metric,
            StorageDeadlines::default(),
        )
    }

    /// Create an isolated volatile pipeline with explicit thresholds and wait deadlines.
    pub fn memory_with_thresholds_and_deadlines(
        tick: usize,
        agent: usize,
        event: usize,
        metric: usize,
        deadlines: StorageDeadlines,
    ) -> Result<Self, StorageError> {
        Self::with_target_and_deadlines(
            StorageTarget::Memory,
            tick,
            agent,
            event,
            metric,
            deadlines,
        )
    }

    fn with_target_and_deadlines(
        target: StorageTarget,
        tick: usize,
        agent: usize,
        event: usize,
        metric: usize,
        deadlines: StorageDeadlines,
    ) -> Result<Self, StorageError> {
        deadlines.validate()?;
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

        match startup_rx.recv_deadline(Instant::now() + deadlines.startup_ack) {
            Ok(Ok(())) => Ok(Self {
                sink: StorageSink {
                    tx,
                    analytics,
                    admission,
                    path: storage_path.clone(),
                    deadlines,
                },
                handle: Some(handle),
                pending_shutdown: None,
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
            Err(xchan::RecvTimeoutError::Disconnected) => match handle.join() {
                Ok(Some(terminal_error)) => Err(StorageError::Worker(terminal_error)),
                Ok(None) => Err(StorageError::Worker(StorageWorkerError::Channel {
                    operation: StorageOperation::Startup,
                    path: storage_path.to_string(),
                    tick: None,
                    commit_state: FailureCommitState::NotAdmitted,
                    detail: "storage worker exited before startup acknowledgement".to_owned(),
                })),
                Err(panic) => Err(StorageError::Worker(StorageWorkerError::Internal {
                    operation: StorageOperation::Join,
                    path: storage_path.to_string(),
                    tick: None,
                    commit_state: FailureCommitState::Indeterminate,
                    detail: format!("storage worker panicked during startup: {panic:?}"),
                })),
            },
            Err(xchan::RecvTimeoutError::Timeout) => {
                let error = StorageWorkerError::Timeout {
                    operation: StorageOperation::Startup,
                    phase: StorageWaitPhase::Acknowledgement,
                    path: storage_path.to_string(),
                    tick: None,
                    waited: deadlines.startup_ack,
                    commit_state: FailureCommitState::Indeterminate,
                };
                analytics.publish_worker_error(&error, false);
                drop(startup_rx);
                drop(tx);
                handoff_storage_reap(StorageReapRequest::JoinOnly {
                    handle,
                    path: storage_path,
                    analytics,
                });
                Err(StorageError::Worker(error))
            }
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

    /// Admit a persistence batch and return its durable-outbox identity.
    pub fn submit_with_receipt(
        &self,
        payload: &PersistenceBatch,
    ) -> Result<AdmissionReceipt, StorageError> {
        self.sink.submit_with_receipt(payload)
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
        let enqueue_deadline = Instant::now() + self.sink.deadlines.command_enqueue;
        let admission = lock_admission_gate_until(
            &self.sink.admission,
            &self.sink.path,
            AdmissionGateWait {
                operation: StorageOperation::Flush,
                tick: None,
                commit_state: FailureCommitState::Indeterminate,
                deadline: enqueue_deadline,
                waited: self.sink.deadlines.command_enqueue,
                recover_poison: false,
            },
        )
        .map_err(|worker_error| {
            self.sink
                .analytics
                .publish_worker_error(&worker_error, false);
            StorageError::Worker(worker_error)
        })?;
        if !admission.open {
            let error = StorageWorkerError::Channel {
                operation: StorageOperation::Flush,
                path: self.sink.path.to_string(),
                tick: None,
                commit_state: FailureCommitState::Indeterminate,
                detail: "storage pipeline is closing or closed".to_owned(),
            };
            self.sink.analytics.publish_worker_error(&error, true);
            return Err(StorageError::Worker(error));
        }
        let send_result = self
            .sink
            .tx
            .send_deadline(StorageCommand::Flush { reply: reply_tx }, enqueue_deadline);
        drop(admission);
        match send_result {
            Ok(()) => {}
            Err(xchan::SendTimeoutError::Timeout(_)) => {
                let error = StorageWorkerError::Timeout {
                    operation: StorageOperation::Flush,
                    phase: StorageWaitPhase::CommandEnqueue,
                    path: self.sink.path.to_string(),
                    tick: None,
                    waited: self.sink.deadlines.command_enqueue,
                    commit_state: FailureCommitState::Indeterminate,
                };
                self.sink.analytics.publish_worker_error(&error, false);
                return Err(StorageError::Worker(error));
            }
            Err(xchan::SendTimeoutError::Disconnected(_)) => {
                let error = StorageWorkerError::Channel {
                    operation: StorageOperation::Flush,
                    path: self.sink.path.to_string(),
                    tick: None,
                    commit_state: FailureCommitState::Indeterminate,
                    detail: "storage worker command channel is disconnected".to_owned(),
                };
                self.sink.analytics.publish_worker_error(&error, true);
                return Err(StorageError::Worker(error));
            }
        }
        reply_rx
            .recv_deadline(Instant::now() + self.sink.deadlines.flush_ack)
            .map_err(|error| {
                let (worker_error, stopped) = match error {
                    xchan::RecvTimeoutError::Timeout => (
                        StorageWorkerError::Timeout {
                            operation: StorageOperation::Flush,
                            phase: StorageWaitPhase::Acknowledgement,
                            path: self.sink.path.to_string(),
                            tick: None,
                            waited: self.sink.deadlines.flush_ack,
                            commit_state: FailureCommitState::Indeterminate,
                        },
                        false,
                    ),
                    xchan::RecvTimeoutError::Disconnected => (
                        StorageWorkerError::Channel {
                            operation: StorageOperation::Flush,
                            path: self.sink.path.to_string(),
                            tick: None,
                            commit_state: FailureCommitState::Indeterminate,
                            detail: "storage worker exited before flush acknowledgement".to_owned(),
                        },
                        true,
                    ),
                };
                self.sink
                    .analytics
                    .publish_worker_error(&worker_error, stopped);
                StorageError::Worker(worker_error)
            })?
            .map_err(StorageError::Worker)
    }

    fn join_shutdown_worker(
        &mut self,
        response: ShutdownReply,
    ) -> Result<ShutdownReceipt, StorageError> {
        let Some(handle) = self.handle.take() else {
            return Err(StorageError::Worker(StorageWorkerError::Internal {
                operation: StorageOperation::Join,
                path: self.sink.path.to_string(),
                tick: None,
                commit_state: FailureCommitState::Indeterminate,
                detail: "storage worker ownership was lost before join".to_owned(),
            }));
        };
        match handle.join() {
            Err(panic) => {
                let error = StorageWorkerError::Internal {
                    operation: StorageOperation::Join,
                    path: self.sink.path.to_string(),
                    tick: None,
                    commit_state: FailureCommitState::Indeterminate,
                    detail: format!("storage worker thread panicked: {panic:?}"),
                };
                self.sink.analytics.publish_worker_error(&error, true);
                Err(StorageError::Worker(error))
            }
            Ok(Some(terminal_error)) => {
                self.sink
                    .analytics
                    .publish_worker_error(&terminal_error, true);
                Err(StorageError::Worker(terminal_error))
            }
            Ok(None) => response.map_err(StorageError::Worker),
        }
    }

    /// Flush, close, and join the worker, returning an explicit shutdown receipt.
    pub fn shutdown(&mut self) -> Result<ShutdownReceipt, StorageError> {
        if self.handle.is_none() {
            return Err(StorageError::Worker(StorageWorkerError::Internal {
                operation: StorageOperation::Shutdown,
                path: self.sink.path.to_string(),
                tick: None,
                commit_state: FailureCommitState::Committed,
                detail: "storage worker has already been shut down".to_owned(),
            }));
        }

        if self.pending_shutdown.is_none() {
            let enqueue_deadline = Instant::now() + self.sink.deadlines.command_enqueue;
            let mut admission = lock_admission_gate_until(
                &self.sink.admission,
                &self.sink.path,
                AdmissionGateWait {
                    operation: StorageOperation::Shutdown,
                    tick: None,
                    commit_state: FailureCommitState::Indeterminate,
                    deadline: enqueue_deadline,
                    waited: self.sink.deadlines.command_enqueue,
                    recover_poison: true,
                },
            )
            .map_err(|worker_error| {
                self.sink
                    .analytics
                    .publish_worker_error(&worker_error, false);
                StorageError::Worker(worker_error)
            })?;
            admission.open = false;
            let (reply_tx, reply_rx) = xchan::bounded(1);
            let send_result = self.sink.tx.send_deadline(
                StorageCommand::Shutdown { reply: reply_tx },
                enqueue_deadline,
            );
            drop(admission);
            match send_result {
                Ok(()) => self.pending_shutdown = Some(reply_rx),
                Err(xchan::SendTimeoutError::Timeout(_)) => {
                    let error = StorageWorkerError::Timeout {
                        operation: StorageOperation::Shutdown,
                        phase: StorageWaitPhase::CommandEnqueue,
                        path: self.sink.path.to_string(),
                        tick: None,
                        waited: self.sink.deadlines.command_enqueue,
                        commit_state: FailureCommitState::Indeterminate,
                    };
                    self.sink.analytics.publish_worker_error(&error, false);
                    return Err(StorageError::Worker(error));
                }
                Err(xchan::SendTimeoutError::Disconnected(_)) => {
                    let error = StorageWorkerError::Channel {
                        operation: StorageOperation::Shutdown,
                        path: self.sink.path.to_string(),
                        tick: None,
                        commit_state: FailureCommitState::Indeterminate,
                        detail: "storage worker command channel disconnected before shutdown"
                            .to_owned(),
                    };
                    self.sink.analytics.publish_worker_error(&error, true);
                    return self.join_shutdown_worker(Err(error));
                }
            }
        }

        let deadline = Instant::now() + self.sink.deadlines.shutdown_ack;
        loop {
            let now = Instant::now();
            if now >= deadline {
                let error = StorageWorkerError::Timeout {
                    operation: StorageOperation::Shutdown,
                    phase: StorageWaitPhase::Acknowledgement,
                    path: self.sink.path.to_string(),
                    tick: None,
                    waited: self.sink.deadlines.shutdown_ack,
                    commit_state: FailureCommitState::Indeterminate,
                };
                self.sink.analytics.publish_worker_error(&error, false);
                return Err(StorageError::Worker(error));
            }

            let wait = Duration::from_millis(10).min(deadline.duration_since(now));
            let response = self
                .pending_shutdown
                .as_ref()
                .ok_or_else(|| {
                    StorageError::Worker(StorageWorkerError::Internal {
                        operation: StorageOperation::Shutdown,
                        path: self.sink.path.to_string(),
                        tick: None,
                        commit_state: FailureCommitState::Indeterminate,
                        detail: "shutdown command has no receipt receiver".to_owned(),
                    })
                })?
                .recv_timeout(wait);
            match response {
                Ok(response) => {
                    self.pending_shutdown.take();
                    return self.join_shutdown_worker(response);
                }
                Err(xchan::RecvTimeoutError::Disconnected) => {
                    self.pending_shutdown.take();
                    let error = StorageWorkerError::Channel {
                        operation: StorageOperation::Shutdown,
                        path: self.sink.path.to_string(),
                        tick: None,
                        commit_state: FailureCommitState::Indeterminate,
                        detail: "storage worker exited before shutdown acknowledgement".to_owned(),
                    };
                    self.sink.analytics.publish_worker_error(&error, true);
                    return self.join_shutdown_worker(Err(error));
                }
                Err(xchan::RecvTimeoutError::Timeout)
                    if self
                        .handle
                        .as_ref()
                        .is_some_and(thread::JoinHandle::is_finished) =>
                {
                    // A terminal worker can exit after another command is acknowledged while an
                    // unconsumed shutdown command still owns its reply sender in the channel.
                    // Once the worker has finished, never wait on that potentially orphaned sender:
                    // join first, allowing the structured terminal cause to take precedence.
                    let response = match self
                        .pending_shutdown
                        .as_ref()
                        .expect("pending shutdown checked above")
                        .try_recv()
                    {
                        Ok(response) => response,
                        Err(error) => Err(StorageWorkerError::Channel {
                            operation: StorageOperation::Shutdown,
                            path: self.sink.path.to_string(),
                            tick: None,
                            commit_state: FailureCommitState::Indeterminate,
                            detail: format!(
                                "storage worker finished before shutdown acknowledgement: {error}"
                            ),
                        }),
                    };
                    self.pending_shutdown.take();
                    return self.join_shutdown_worker(response);
                }
                Err(xchan::RecvTimeoutError::Timeout) => {}
            }
        }
    }
}

impl Drop for StoragePipeline {
    fn drop(&mut self) {
        if self.handle.is_some()
            && let Err(error) = self.shutdown()
        {
            eprintln!("failed to shut down storage worker cleanly: {error}");
            if let Some(handle) = self.handle.take() {
                handoff_storage_reap(StorageReapRequest::Pipeline {
                    tx: self.sink.tx.clone(),
                    admission: Arc::clone(&self.sink.admission),
                    pending_shutdown: self.pending_shutdown.take(),
                    handle,
                    path: Arc::clone(&self.sink.path),
                    analytics: self.sink.analytics.clone(),
                });
            }
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
    let watermarks = match storage.persistence_watermarks() {
        Ok(watermarks) => watermarks,
        Err(error) => {
            let worker_error = worker_error_from_storage(
                StorageOperation::Recovery,
                &path,
                None,
                FailureCommitState::Indeterminate,
                error,
            );
            analytics.publish_worker_error(&worker_error, true);
            let _ = startup.send(Err(worker_error));
            storage.abandon_after_error();
            return None;
        }
    };
    let recovered_analytics = match storage.latest_pending_analytics() {
        Ok(pending) => pending,
        Err(error) => {
            let worker_error = worker_error_from_storage(
                StorageOperation::Recovery,
                &path,
                None,
                FailureCommitState::Indeterminate,
                error,
            );
            analytics.publish_worker_error(&worker_error, true);
            let _ = startup.send(Err(worker_error));
            storage.abandon_after_error();
            return None;
        }
    };
    let mut state = WorkerState {
        committed_tick: recovered_analytics.as_ref().map(|pending| pending.tick),
        admitted_tick: recovered_analytics.as_ref().map(|pending| pending.tick),
        guarantee,
        watermarks,
        ..WorkerState::default()
    };
    if let Some(pending) = recovered_analytics {
        analytics.publish_committed(pending, watermarks);
    } else {
        analytics.publish_progress(watermarks);
    }
    #[cfg(test)]
    maybe_pause_storage_startup(&path);
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

    while let Ok(command) = rx.recv() {
        match command {
            StorageCommand::Persist { batch, reply } => {
                let PreparedPersistenceBatch {
                    tick,
                    storage: prepared,
                    analytics: pending,
                } = *batch;
                match storage.stage_outbox(tick, &prepared) {
                    Ok((receipt, newly_admitted)) => {
                        state.admitted_tick = Some(
                            state
                                .admitted_tick
                                .map_or(tick, |previous| previous.max(tick)),
                        );
                        state.watermarks = receipt.watermarks;
                        analytics.publish_progress(receipt.watermarks);
                        let _ = reply.send(Ok(receipt));
                        if !newly_admitted {
                            continue;
                        }
                        state.pending_analytics.push((receipt.batch_id, pending));
                        match storage.enqueue_staged(receipt.batch_id, prepared) {
                            Ok(true) => {
                                if let Err(error) =
                                    flush_worker_storage(&mut storage, &mut state, &analytics)
                                {
                                    analytics.publish_worker_error(&error, true);
                                    storage.abandon_after_error();
                                    return Some(error);
                                }
                            }
                            Ok(false) => {}
                            Err(error) => {
                                let worker_error = worker_error_from_storage(
                                    StorageOperation::Persist,
                                    &path,
                                    Some(tick),
                                    FailureCommitState::RolledBack,
                                    error,
                                );
                                analytics.publish_worker_error(&worker_error, true);
                                storage.abandon_after_error();
                                return Some(worker_error);
                            }
                        }
                    }
                    Err(error) => {
                        let worker_error = worker_error_from_storage(
                            StorageOperation::Admit,
                            &path,
                            Some(tick),
                            FailureCommitState::NotAdmitted,
                            error,
                        );
                        analytics.publish_worker_error(&worker_error, true);
                        let terminal_error = duplicate_worker_error(&worker_error);
                        let _ = reply.send(Err(worker_error));
                        storage.abandon_after_error();
                        return Some(terminal_error);
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
    let applied = state.watermarks.applied.map_or(0, PersistenceBatchId::get);
    let eligible = state
        .pending_analytics
        .partition_point(|(batch_id, _)| batch_id.get() <= applied);
    if eligible == 0 {
        analytics.publish_progress(state.watermarks);
        return;
    }
    let mut committed = None;
    for (_, pending) in state.pending_analytics.drain(..eligible) {
        committed = Some(pending);
    }
    if let Some(pending) = committed {
        state.committed_tick = Some(pending.tick);
        analytics.publish_committed(pending, state.watermarks);
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
    state.watermarks = storage.persistence_watermarks().map_err(|error| {
        worker_error_from_storage(
            StorageOperation::Flush,
            &storage.path,
            state.admitted_tick,
            FailureCommitState::Committed,
            error,
        )
    })?;
    publish_committed_state(state, analytics);
    state.watermarks = match storage.finalize_applied_outbox() {
        Ok(watermarks) => watermarks,
        Err(error) => {
            if let Ok(watermarks) = storage.persistence_watermarks() {
                state.watermarks = watermarks;
                publish_committed_state(state, analytics);
            }
            return Err(worker_error_from_storage(
                StorageOperation::Durability,
                &storage.path,
                state.admitted_tick,
                FailureCommitState::Committed,
                error,
            )
            .with_commit_state(FailureCommitState::Committed));
        }
    };
    publish_committed_state(state, analytics);
    Ok(FlushReceipt {
        committed_tick: state.committed_tick,
        guarantee: state.guarantee,
        watermarks: state.watermarks,
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
        watermarks: state.watermarks,
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
        io::{Read, Write},
        path::PathBuf,
        process::{Child, Command, ExitStatus, Stdio},
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

    fn create_file_storage(path: &str) -> Result<Storage, StorageError> {
        Storage::create_new_file_with_thresholds(path, 64, 4096, 1024, 1024)
    }

    fn recover_file_storage(path: &str) -> Result<Storage, StorageError> {
        Storage::with_target(
            StorageTarget::RecoverExisting(path.to_owned()),
            64,
            4096,
            1024,
            1024,
        )
    }

    fn short_deadlines() -> StorageDeadlines {
        StorageDeadlines {
            startup_ack: Duration::from_secs(2),
            command_enqueue: Duration::from_millis(100),
            admission_ack: Duration::from_millis(250),
            flush_ack: Duration::from_millis(250),
            shutdown_ack: Duration::from_millis(250),
        }
    }

    #[test]
    fn invalid_deadline_configuration_is_rejected_without_panicking()
    -> Result<(), Box<dyn std::error::Error>> {
        let deadlines = StorageDeadlines {
            startup_ack: Duration::MAX,
            ..StorageDeadlines::default()
        };
        let error = match StoragePipeline::memory_with_thresholds_and_deadlines(
            64, 4096, 1024, 1024, deadlines,
        ) {
            Ok(mut pipeline) => {
                pipeline.shutdown()?;
                return Err("invalid deadline unexpectedly created a pipeline".into());
            }
            Err(error) => error,
        };
        assert!(matches!(
            error,
            StorageError::InvalidData {
                context: "storage.deadlines",
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn startup_timeout_retains_path_ownership_until_supervised_join()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-startup-timeout-lease");
        let path_string = path.to_string_lossy().to_string();
        let (entered, release) = register_startup_pause(&path_string);
        let deadlines = StorageDeadlines {
            startup_ack: Duration::from_millis(100),
            ..short_deadlines()
        };
        let first_error = match StoragePipeline::create_new_file_with_thresholds_and_deadlines(
            &path_string,
            64,
            4096,
            1024,
            1024,
            deadlines,
        ) {
            Ok(mut pipeline) => {
                pipeline.shutdown()?;
                return Err("startup pause did not trigger its deadline".into());
            }
            Err(error) => error,
        };
        assert!(matches!(
            first_error,
            StorageError::Worker(StorageWorkerError::Timeout {
                operation: StorageOperation::Startup,
                phase: StorageWaitPhase::Acknowledgement,
                ..
            })
        ));
        entered.recv_timeout(Duration::from_secs(2))?;

        let second_error = match StoragePipeline::with_target_and_deadlines(
            StorageTarget::RecoverExisting(path_string.clone()),
            64,
            4096,
            1024,
            1024,
            deadlines,
        ) {
            Ok(mut pipeline) => {
                pipeline.shutdown()?;
                return Err("second writer bypassed the timed-out startup lease".into());
            }
            Err(error) => error,
        };
        assert!(
            second_error
                .to_string()
                .contains("another ScriptBots writer")
        );
        release.send(())?;

        let retry_deadline = Instant::now() + Duration::from_secs(5);
        let retry_deadlines = short_deadlines();
        loop {
            match StoragePipeline::with_target_and_deadlines(
                StorageTarget::RecoverExisting(path_string.clone()),
                64,
                4096,
                1024,
                1024,
                retry_deadlines,
            ) {
                Ok(mut pipeline) => {
                    pipeline.shutdown()?;
                    break;
                }
                Err(error) => {
                    if Instant::now() >= retry_deadline {
                        return Err(format!(
                            "supervised startup join never released the path lease: {error}"
                        )
                        .into());
                    }
                    thread::sleep(Duration::from_millis(10));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn writer_path_lease_rejects_a_second_live_writer_and_releases_on_shutdown()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-writer-path-lease");
        let path_string = path.to_string_lossy().to_string();
        let mut first = StoragePipeline::create_new_file(&path_string)?;
        let error = match StoragePipeline::recover_existing(&path_string) {
            Ok(mut second) => {
                second.shutdown()?;
                return Err("second writer unexpectedly acquired the same database".into());
            }
            Err(error) => error,
        };
        assert!(error.to_string().contains("another ScriptBots writer"));
        first.shutdown()?;

        let mut after_shutdown = StoragePipeline::recover_existing(&path_string)?;
        after_shutdown.shutdown()?;
        Ok(())
    }

    struct WriterLeaseChild {
        child: Child,
    }

    impl WriterLeaseChild {
        fn wait_until_exit(
            &mut self,
            context: &str,
        ) -> Result<ExitStatus, Box<dyn std::error::Error>> {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(status) = self.child.try_wait()? {
                    return Ok(status);
                }
                if Instant::now() >= deadline {
                    self.child.kill()?;
                    let status = self.child.wait()?;
                    return Err(format!(
                        "writer-lease child timed out during {context}; terminated with {status}"
                    )
                    .into());
                }
                thread::sleep(Duration::from_millis(10));
            }
        }

        fn force_terminate(&mut self) -> Result<ExitStatus, Box<dyn std::error::Error>> {
            if let Some(status) = self.child.try_wait()? {
                return Ok(status);
            }
            self.child.kill()?;
            Ok(self.child.wait()?)
        }
    }

    impl Drop for WriterLeaseChild {
        fn drop(&mut self) {
            if self.child.try_wait().ok().flatten().is_none() {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }

    fn spawn_writer_lease_child(
        path: &str,
        mode: &str,
        ready_path: Option<&Path>,
    ) -> Result<WriterLeaseChild, Box<dyn std::error::Error>> {
        let mut command = Command::new(std::env::current_exe()?);
        command
            .args([
                "--exact",
                "tests::storage_writer_lease_child",
                "--nocapture",
            ])
            .env("SCRIPTBOTS_STORAGE_WRITER_LEASE_CHILD", mode)
            .env("SCRIPTBOTS_STORAGE_WRITER_LEASE_PATH", path)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        if let Some(ready_path) = ready_path {
            command
                .env("SCRIPTBOTS_STORAGE_WRITER_LEASE_READY", ready_path)
                .stdin(Stdio::piped());
        } else {
            command.stdin(Stdio::null());
        }
        Ok(WriterLeaseChild {
            child: command.spawn()?,
        })
    }

    fn wait_for_writer_lease_child(
        child: &mut WriterLeaseChild,
        ready_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if ready_path.try_exists()? {
                return Ok(());
            }
            if let Some(status) = child.child.try_wait()? {
                return Err(format!(
                    "writer-lease owner exited before readiness with status {status}"
                )
                .into());
            }
            if Instant::now() >= deadline {
                let status = child.force_terminate()?;
                return Err(format!(
                    "writer-lease owner did not become ready; terminated with status {status}"
                )
                .into());
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn release_writer_lease_child(
        child: &mut WriterLeaseChild,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut stdin = child
            .child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("writer-lease child stdin was not piped"))?;
        stdin.write_all(b"\n")?;
        drop(stdin);
        let status = child.wait_until_exit("graceful shutdown")?;
        if !status.success() {
            return Err(format!("writer-lease owner failed during shutdown: {status}").into());
        }
        Ok(())
    }

    fn run_writer_lease_child(path: &str, mode: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut child = spawn_writer_lease_child(path, mode, None)?;
        let status = child.wait_until_exit(mode)?;
        if !status.success() {
            return Err(format!("writer-lease child mode {mode:?} failed: {status}").into());
        }
        Ok(())
    }

    #[test]
    fn storage_writer_lease_child() -> Result<(), Box<dyn std::error::Error>> {
        let Ok(mode) = std::env::var("SCRIPTBOTS_STORAGE_WRITER_LEASE_CHILD") else {
            return Ok(());
        };
        let path = std::env::var("SCRIPTBOTS_STORAGE_WRITER_LEASE_PATH")?;
        match mode.as_str() {
            "pipeline-owner" => {
                let mut pipeline = StoragePipeline::recover_existing(&path)?;
                let ready_path = std::env::var("SCRIPTBOTS_STORAGE_WRITER_LEASE_READY")?;
                fs::write(ready_path, b"ready")?;
                std::io::stdin().read_exact(&mut [0_u8])?;
                pipeline.shutdown()?;
            }
            "lease-only-owner" => {
                let _lease = StorageWriterLease::acquire(&path)?.ok_or_else(|| {
                    std::io::Error::other("file-backed path did not create a writer lease")
                })?;
                let ready_path = std::env::var("SCRIPTBOTS_STORAGE_WRITER_LEASE_READY")?;
                fs::write(ready_path, b"ready")?;
                std::io::stdin().read_exact(&mut [0_u8])?;
            }
            "expect-refusal" => {
                let expected_lock_path = storage_writer_lock_path(&path).display().to_string();
                let error = match StoragePipeline::recover_existing(&path) {
                    Ok(mut pipeline) => {
                        pipeline.shutdown()?;
                        return Err("second process unexpectedly acquired the writer lease".into());
                    }
                    Err(error) => error,
                };
                match error {
                    StorageError::Worker(StorageWorkerError::WriterLeaseHeld {
                        operation,
                        path: refused_path,
                        lock_path,
                        tick,
                        commit_state,
                    }) => {
                        assert_eq!(operation, StorageOperation::Startup);
                        assert_eq!(refused_path, path);
                        assert_eq!(lock_path, expected_lock_path);
                        assert_eq!(tick, None);
                        assert_eq!(commit_state, FailureCommitState::NotAdmitted);
                    }
                    other => {
                        return Err(format!(
                            "second writer returned an untyped or unexpected error: {other}"
                        )
                        .into());
                    }
                }
            }
            "expect-open" => {
                let mut pipeline = StoragePipeline::recover_existing(&path)?;
                pipeline.shutdown()?;
            }
            other => return Err(format!("unknown writer-lease child mode {other:?}").into()),
        }
        Ok(())
    }

    #[test]
    fn os_writer_lease_refuses_second_process_and_reopens_after_graceful_exit()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-os-writer-lease-graceful");
        let path_string = path.to_string_lossy().to_string();
        let lock_path = storage_writer_lock_path(&path_string);
        let mut fixture = StoragePipeline::create_new_file(&path_string)?;
        fixture.shutdown()?;
        assert!(lock_path.try_exists()?);

        let ready_path = temp_db_path("storage-os-writer-lease-graceful-ready");
        let mut owner =
            spawn_writer_lease_child(&path_string, "pipeline-owner", Some(&ready_path))?;
        wait_for_writer_lease_child(&mut owner, &ready_path)?;
        run_writer_lease_child(&path_string, "expect-refusal")?;
        release_writer_lease_child(&mut owner)?;

        assert!(lock_path.try_exists()?, "companion lease file was removed");
        run_writer_lease_child(&path_string, "expect-open")?;
        Ok(())
    }

    #[test]
    fn os_writer_lease_releases_after_forced_child_termination()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-os-writer-lease-forced");
        let path_string = path.to_string_lossy().to_string();
        let lock_path = storage_writer_lock_path(&path_string);
        let mut fixture = StoragePipeline::create_new_file(&path_string)?;
        fixture.shutdown()?;

        let ready_path = temp_db_path("storage-os-writer-lease-forced-ready");
        let mut owner =
            spawn_writer_lease_child(&path_string, "pipeline-owner", Some(&ready_path))?;
        wait_for_writer_lease_child(&mut owner, &ready_path)?;
        let killed = owner.force_terminate()?;
        assert!(
            !killed.success(),
            "forced child unexpectedly exited successfully"
        );

        assert!(lock_path.try_exists()?, "companion lease file was removed");
        run_writer_lease_child(&path_string, "expect-open")?;
        Ok(())
    }

    #[test]
    fn writer_lease_precedes_recovery_validation_and_writable_open()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-os-writer-lease-before-validation");
        fs::write(&path, b"not a ScriptBots database")?;
        let path_string = path.to_string_lossy().to_string();
        let ready_path = temp_db_path("storage-os-writer-lease-before-validation-ready");
        let mut owner =
            spawn_writer_lease_child(&path_string, "lease-only-owner", Some(&ready_path))?;
        wait_for_writer_lease_child(&mut owner, &ready_path)?;

        run_writer_lease_child(&path_string, "expect-refusal")?;
        release_writer_lease_child(&mut owner)?;
        assert!(StoragePipeline::recover_existing(&path_string).is_err());
        Ok(())
    }

    #[test]
    fn memory_storage_does_not_participate_in_file_writer_leases()
    -> Result<(), Box<dyn std::error::Error>> {
        assert!(StorageWriterLease::acquire(":memory:")?.is_none());
        let mut first = StoragePipeline::memory()?;
        let mut second = StoragePipeline::memory()?;
        first.shutdown()?;
        second.shutdown()?;
        Ok(())
    }

    #[test]
    fn recover_existing_refuses_empty_and_unrelated_files_without_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        let empty = temp_db_path("storage-recover-empty-refusal");
        fs::write(&empty, b"")?;
        let empty_string = empty.to_string_lossy().to_string();
        assert!(StoragePipeline::recover_existing(&empty_string).is_err());
        assert_eq!(fs::read(&empty)?, b"");

        let unrelated = temp_db_path("storage-recover-unrelated-refusal");
        let unrelated_string = unrelated.to_string_lossy().to_string();
        let connection = Connection::open_strict_multi_process(&unrelated_string)?;
        connection.execute("CREATE TABLE unrelated (value INTEGER NOT NULL)")?;
        connection.close()?;
        let before = fs::read(&unrelated)?;
        assert!(StoragePipeline::recover_existing(&unrelated_string).is_err());
        assert_eq!(fs::read(&unrelated)?, before);
        Ok(())
    }

    fn assert_recovery_refused_without_database_mutation(
        path: &Path,
        expected_error_fragment: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let before = fs::read(path)?;
        let path_string = path.to_string_lossy().to_string();
        let error = match recover_file_storage(&path_string) {
            Ok(storage) => {
                storage.close()?;
                return Err(format!(
                    "recovery unexpectedly accepted malformed database at {}",
                    path.display()
                )
                .into());
            }
            Err(error) => error,
        };
        assert!(
            error.to_string().contains(expected_error_fragment),
            "unexpected recovery error: {error}"
        );
        assert_eq!(
            fs::read(path)?,
            before,
            "refused recovery mutated {}",
            path.display()
        );
        Ok(())
    }

    fn create_valid_database(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let path_string = path.to_string_lossy().to_string();
        let mut storage = StoragePipeline::create_new_file(&path_string)?;
        storage.shutdown()?;
        Ok(())
    }

    fn add_schema_object(path: &Path, sql: &str) -> Result<(), Box<dyn std::error::Error>> {
        let connection = Connection::open(path.to_string_lossy().as_ref())?;
        connection.execute(sql)?;
        connection.close()?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn recovery_binds_validation_and_writes_to_the_leased_open_file_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let original = temp_db_path("storage-recovery-identity-original");
        let replacement = temp_db_path("storage-recovery-identity-replacement");
        let parked_original = temp_db_path("storage-recovery-identity-parked");
        create_valid_database(&original)?;
        create_valid_database(&replacement)?;
        let original_before = fs::read(&original)?;
        let replacement_before = fs::read(&replacement)?;
        let original_string = original.to_string_lossy().to_string();

        let error = match Storage::with_target_before_recovery_writer_open(
            StorageTarget::RecoverExisting(original_string),
            64,
            4096,
            1024,
            1024,
            |_| {
                fs::rename(&original, &parked_original).expect("park leased original");
                fs::rename(&replacement, &original).expect("swap replacement into pathname");
            },
        ) {
            Ok(storage) => {
                storage.close()?;
                return Err("path-swapped recovery unexpectedly succeeded".into());
            }
            Err(error) => error,
        };

        assert!(matches!(error, StorageError::Database(_)));
        assert_eq!(fs::read(&parked_original)?, original_before);
        assert_eq!(fs::read(&original)?, replacement_before);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn recovery_writer_open_does_not_initialize_a_swapped_empty_file()
    -> Result<(), Box<dyn std::error::Error>> {
        let original = temp_db_path("storage-recovery-empty-swap-original");
        let replacement = temp_db_path("storage-recovery-empty-swap-replacement");
        let parked_original = temp_db_path("storage-recovery-empty-swap-parked");
        create_valid_database(&original)?;
        fs::write(&replacement, b"")?;
        let original_before = fs::read(&original)?;
        let replacement_before = fs::read(&replacement)?;
        let original_string = original.to_string_lossy().to_string();

        let error = match Storage::with_target_before_recovery_writer_open(
            StorageTarget::RecoverExisting(original_string),
            64,
            4096,
            1024,
            1024,
            |_| {
                fs::rename(&original, &parked_original).expect("park leased original");
                fs::rename(&replacement, &original).expect("swap empty file into pathname");
            },
        ) {
            Ok(storage) => {
                storage.close()?;
                return Err("empty path-swapped recovery unexpectedly succeeded".into());
            }
            Err(error) => error,
        };

        assert!(
            matches!(error, StorageError::Database(_)),
            "unexpected swapped-empty recovery error: {error}"
        );
        assert_eq!(fs::read(&parked_original)?, original_before);
        assert_eq!(
            fs::read(&original)?,
            replacement_before,
            "recovery writer open initialized the swapped empty path"
        );
        Ok(())
    }

    #[test]
    fn recovery_requires_the_exact_supported_migration_set_without_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        let future = temp_db_path("storage-recovery-future-migration");
        create_valid_database(&future)?;
        add_schema_object(
            &future,
            "INSERT INTO _schema_migrations (version, name) VALUES (3, 'future_schema')",
        )?;
        assert_recovery_refused_without_database_mutation(&future, "exactly two")?;

        let weak_v1 = temp_db_path("storage-recovery-weak-v1");
        let weak_connection = Connection::open(weak_v1.to_string_lossy().as_ref())?;
        MigrationRunner::new()
            .add(1, "create_scriptbots_schema", SCRIPTBOTS_SCHEMA_V1)
            .run(&weak_connection)?;
        weak_connection.close()?;
        assert_recovery_refused_without_database_mutation(&weak_v1, "exactly two")?;
        Ok(())
    }

    #[test]
    fn recovery_rejects_forged_lookalike_schema_without_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        let forged = temp_db_path("storage-recovery-forged-lookalike");
        let connection = Connection::open(forged.to_string_lossy().as_ref())?;
        connection.execute(
            "CREATE TABLE _schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
             );
             INSERT INTO _schema_migrations (version, name, applied_at)
             VALUES (1, 'create_scriptbots_schema', 'forged');
             INSERT INTO _schema_migrations (version, name, applied_at)
             VALUES (2, 'create_durable_persistence_outbox', 'forged');
             CREATE TABLE storage_progress (
                singleton INTEGER PRIMARY KEY,
                admitted_batch_id INTEGER NOT NULL,
                applied_batch_id INTEGER NOT NULL,
                durable_batch_id INTEGER NOT NULL
             );
             INSERT INTO storage_progress VALUES (1, 0, 0, 0);",
        )?;
        connection.close()?;

        assert_recovery_refused_without_database_mutation(&forged, "schema fingerprint mismatch")?;

        let weakened_constraint = temp_db_path("storage-recovery-weakened-check");
        create_valid_database(&weakened_constraint)?;
        let connection = Connection::open(weakened_constraint.to_string_lossy().as_ref())?;
        connection.execute(
            "DROP TABLE metrics;
             CREATE TABLE metrics (
                tick INTEGER NOT NULL,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (tick, name)
             )",
        )?;
        connection.close()?;
        assert_recovery_refused_without_database_mutation(
            &weakened_constraint,
            "schema fingerprint mismatch",
        )
    }

    #[test]
    fn recovery_rejects_every_unexpected_schema_object_without_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        let cases = [
            (
                "extra-table",
                "CREATE TABLE unexpected_table (value INTEGER NOT NULL)",
            ),
            (
                "extra-index",
                "CREATE INDEX unexpected_metric_index ON metrics(name)",
            ),
            (
                "extra-view",
                "CREATE VIEW unexpected_progress_view AS
                 SELECT singleton FROM storage_progress",
            ),
            (
                "extra-trigger",
                "CREATE TRIGGER unexpected_tick_trigger AFTER INSERT ON ticks
                 BEGIN SELECT 1; END",
            ),
        ];
        for (label, sql) in cases {
            let path = temp_db_path(&format!("storage-recovery-{label}"));
            create_valid_database(&path)?;
            add_schema_object(&path, sql)?;
            assert_recovery_refused_without_database_mutation(
                &path,
                "schema fingerprint mismatch",
            )?;
        }
        Ok(())
    }

    #[test]
    fn exact_schema_database_recovers_through_the_identity_bound_connection()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-recovery-exact-schema-happy");
        create_valid_database(&path)?;
        let path_string = path.to_string_lossy().to_string();
        let storage = recover_file_storage(&path_string)?;
        assert_eq!(
            read_schema_objects(storage.connection()?)?,
            canonical_schema_objects()?
        );
        storage.close()?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn writer_refuses_symlink_and_hard_link_aliases() -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let original = temp_db_path("storage-writer-alias-original");
        let original_string = original.to_string_lossy().to_string();
        let mut pipeline = StoragePipeline::create_new_file(&original_string)?;
        pipeline.shutdown()?;

        let symlink_path = temp_db_path("storage-writer-alias-symlink");
        symlink(&original, &symlink_path)?;
        let symlink_error =
            match StoragePipeline::recover_existing(symlink_path.to_string_lossy().as_ref()) {
                Ok(mut pipeline) => {
                    pipeline.shutdown()?;
                    return Err("symlink writer path unexpectedly succeeded".into());
                }
                Err(error) => error,
            };
        assert!(symlink_error.to_string().contains("non-symlink"));

        let hard_link = temp_db_path("storage-writer-alias-hard-link");
        fs::hard_link(&original, &hard_link)?;
        let hard_link_error =
            match StoragePipeline::recover_existing(hard_link.to_string_lossy().as_ref()) {
                Ok(mut pipeline) => {
                    pipeline.shutdown()?;
                    return Err("hard-link writer path unexpectedly succeeded".into());
                }
                Err(error) => error,
            };
        assert!(hard_link_error.to_string().contains("multiply linked"));
        assert!(StoragePipeline::recover_existing(&original_string).is_err());
        Ok(())
    }

    fn assert_integrity(storage: &Storage) -> Result<(), Box<dyn std::error::Error>> {
        let result: String = storage
            .connection()?
            .query_row("PRAGMA integrity_check")?
            .get_typed(0)?;
        assert_eq!(result, "ok", "FrankenSQLite integrity check failed");
        Ok(())
    }

    #[test]
    fn storage_crash_child() -> Result<(), Box<dyn std::error::Error>> {
        let Ok(boundary) = std::env::var("SCRIPTBOTS_STORAGE_CRASH_CHILD") else {
            return Ok(());
        };
        let path = std::env::var("SCRIPTBOTS_STORAGE_CRASH_PATH")?;
        let tick = std::env::var("SCRIPTBOTS_STORAGE_CRASH_TICK")?.parse::<u64>()?;
        let prepared = PreparedPersistenceBatch::from_batch(&sample_batch(tick, tick as f32))?;
        let mut storage = create_file_storage(&path)?;
        let (admission, newly_admitted) = storage.stage_outbox(prepared.tick, &prepared.storage)?;
        assert!(newly_admitted);
        match boundary.as_str() {
            "admitted" => std::process::exit(86),
            "applied" => {
                assert!(!storage.enqueue_staged(admission.batch_id, prepared.storage)?);
                storage.flush()?;
                std::process::exit(87);
            }
            other => Err(format!("unknown storage crash boundary {other:?}").into()),
        }
    }

    #[test]
    fn abrupt_process_exit_recovers_admitted_and_applied_boundaries()
    -> Result<(), Box<dyn std::error::Error>> {
        for (boundary, exit_code, tick) in [("admitted", 86, 71_u64), ("applied", 87, 72)] {
            let path = temp_db_path(&format!("storage-process-exit-{boundary}"));
            let path_string = path.to_string_lossy().to_string();
            let status = Command::new(std::env::current_exe()?)
                .args(["--exact", "tests::storage_crash_child", "--nocapture"])
                .env("SCRIPTBOTS_STORAGE_CRASH_CHILD", boundary)
                .env("SCRIPTBOTS_STORAGE_CRASH_PATH", &path_string)
                .env("SCRIPTBOTS_STORAGE_CRASH_TICK", tick.to_string())
                .status()?;
            assert_eq!(
                status.code(),
                Some(exit_code),
                "child did not exit at the requested {boundary} boundary"
            );

            let mut recovered = StoragePipeline::recover_existing(&path_string)?;
            let shutdown = recovered.shutdown()?;
            assert_eq!(shutdown.committed_tick, Some(tick));
            assert_eq!(shutdown.watermarks.admitted, shutdown.watermarks.applied);
            assert_eq!(shutdown.watermarks.applied, shutdown.watermarks.durable);
            let reader = StorageReader::open(&path_string)?;
            assert_eq!(reader.run_ledger_summary()?.tick_count, 1);
            reader.close()?;
        }
        Ok(())
    }

    #[test]
    fn durable_outbox_recovers_an_admitted_batch_after_the_worker_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-outbox-admission-recovery");
        let path_string = path.to_string_lossy().to_string();
        let batch = sample_batch(21, 2.1);
        let prepared = PreparedPersistenceBatch::from_batch(&batch)?;
        let mut interrupted = create_file_storage(&path_string)?;
        let (admission, newly_admitted) =
            interrupted.stage_outbox(prepared.tick, &prepared.storage)?;
        assert!(newly_admitted);
        assert_eq!(
            interrupted.persistence_watermarks()?,
            PersistenceWatermarks {
                admitted: Some(admission.batch_id),
                applied: None,
                durable: None,
            }
        );
        interrupted.abandon_after_error();

        let mut recovered = StoragePipeline::recover_existing(&path_string)?;
        let snapshot = recovered.analytics_provider().snapshot();
        assert_eq!(snapshot.committed_tick, Some(21));
        assert_eq!(
            snapshot.watermarks,
            PersistenceWatermarks {
                admitted: Some(admission.batch_id),
                applied: Some(admission.batch_id),
                durable: Some(admission.batch_id),
            }
        );
        let shutdown = recovered.shutdown()?;
        assert_eq!(shutdown.watermarks, snapshot.watermarks);

        let durable = recover_file_storage(&path_string)?;
        assert_eq!(
            durable
                .batch_status(admission.batch_id)?
                .map(|status| status.state),
            Some(BatchPersistenceState::Durable)
        );
        let tick_count: i64 = durable
            .connection()?
            .query_row("SELECT COUNT(*) FROM ticks")?
            .get_typed(0)?;
        assert_eq!(tick_count, 1);
        assert_integrity(&durable)?;
        durable.close()?;
        Ok(())
    }

    #[test]
    fn recovery_after_application_before_durable_marker_is_idempotent()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-outbox-applied-recovery");
        let path_string = path.to_string_lossy().to_string();
        let batch = sample_batch(31, 3.1);
        let prepared = PreparedPersistenceBatch::from_batch(&batch)?;
        let mut interrupted = create_file_storage(&path_string)?;
        let (admission, newly_admitted) =
            interrupted.stage_outbox(prepared.tick, &prepared.storage)?;
        assert!(newly_admitted);
        assert!(!interrupted.enqueue_staged(admission.batch_id, prepared.storage)?);
        interrupted.flush()?;
        assert_eq!(
            interrupted.persistence_watermarks()?,
            PersistenceWatermarks {
                admitted: Some(admission.batch_id),
                applied: Some(admission.batch_id),
                durable: None,
            }
        );
        interrupted.abandon_after_error();

        let mut recovered = StoragePipeline::recover_existing(&path_string)?;
        let shutdown = recovered.shutdown()?;
        assert_eq!(shutdown.watermarks.durable, Some(admission.batch_id));
        let durable = recover_file_storage(&path_string)?;
        let tick_count: i64 = durable
            .connection()?
            .query_row("SELECT COUNT(*) FROM ticks")?
            .get_typed(0)?;
        let event_count: i64 = durable
            .connection()?
            .query_row("SELECT COUNT(*) FROM events")?
            .get_typed(0)?;
        assert_eq!(tick_count, 1, "recovery duplicated a tick row");
        assert_eq!(event_count, 1, "recovery duplicated an event row");
        assert_integrity(&durable)?;
        durable.close()?;
        Ok(())
    }

    #[test]
    fn applied_watermark_is_published_before_durable_finalization_failure()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-applied-before-finalize-failure");
        let path_string = path.to_string_lossy().to_string();
        let PreparedPersistenceBatch {
            tick,
            storage: prepared,
            analytics: pending,
        } = PreparedPersistenceBatch::from_batch(&sample_batch(32, 3.2))?;
        let mut storage = create_file_storage(&path_string)?;
        let (admission, newly_admitted) = storage.stage_outbox(tick, &prepared)?;
        assert!(newly_admitted);
        assert!(!storage.enqueue_staged(admission.batch_id, prepared)?);
        storage.connection()?.execute("DROP TABLE storage_outbox")?;
        let analytics = AnalyticsSnapshotProvider::empty();
        let mut state = WorkerState {
            admitted_tick: Some(tick),
            guarantee: PersistenceGuarantee::Durable,
            watermarks: admission.watermarks,
            pending_analytics: vec![(admission.batch_id, pending)],
            ..WorkerState::default()
        };

        let error = flush_worker_storage(&mut storage, &mut state, &analytics)
            .expect_err("missing outbox table must fail durable finalization");
        assert!(matches!(
            error,
            StorageWorkerError::Database {
                operation: StorageOperation::Durability,
                commit_state: FailureCommitState::Committed,
                ..
            }
        ));
        let snapshot = analytics.snapshot();
        assert_eq!(snapshot.committed_tick, Some(tick));
        assert_eq!(snapshot.watermarks.admitted, Some(admission.batch_id));
        assert_eq!(snapshot.watermarks.applied, Some(admission.batch_id));
        assert_eq!(snapshot.watermarks.durable, None);
        storage.abandon_after_error();
        Ok(())
    }

    #[test]
    fn recovery_applies_outbox_batches_in_admission_order() -> Result<(), Box<dyn std::error::Error>>
    {
        let path = temp_db_path("storage-outbox-order");
        let path_string = path.to_string_lossy().to_string();
        let mut interrupted = create_file_storage(&path_string)?;
        let mut ids = Vec::new();
        for tick in 1..=3 {
            let prepared = PreparedPersistenceBatch::from_batch(&sample_batch(tick, tick as f32))?;
            let (admission, newly_admitted) =
                interrupted.stage_outbox(prepared.tick, &prepared.storage)?;
            assert!(newly_admitted);
            assert!(!interrupted.enqueue_staged(admission.batch_id, prepared.storage)?);
            ids.push(admission.batch_id);
        }
        interrupted.abandon_after_error();

        let mut recovered = StoragePipeline::recover_existing(&path_string)?;
        let shutdown = recovered.shutdown()?;
        assert_eq!(shutdown.watermarks.durable, ids.last().copied());
        let reader = StorageReader::open(&path_string)?;
        assert_eq!(
            reader
                .recent_ticks(None)?
                .into_iter()
                .map(|tick| tick.tick)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        for batch_id in ids {
            assert_eq!(
                reader.batch_status(batch_id)?.map(|status| status.state),
                Some(BatchPersistenceState::Durable)
            );
        }
        reader.close()?;
        Ok(())
    }

    #[test]
    fn corrupt_outbox_gap_fails_startup_without_partial_replay()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-outbox-gap");
        let path_string = path.to_string_lossy().to_string();
        let mut interrupted = create_file_storage(&path_string)?;
        let mut ids = Vec::new();
        for tick in 61..=63 {
            let prepared = PreparedPersistenceBatch::from_batch(&sample_batch(tick, tick as f32))?;
            let (admission, newly_admitted) =
                interrupted.stage_outbox(prepared.tick, &prepared.storage)?;
            assert!(newly_admitted);
            assert!(!interrupted.enqueue_staged(admission.batch_id, prepared.storage)?);
            ids.push(admission.batch_id);
        }
        interrupted.abandon_after_error();

        let corruptor = Connection::open(&path_string)?;
        corruptor.execute_with_params(
            "DELETE FROM storage_outbox WHERE batch_id = ?1",
            &[ids[1].as_i64().into()],
        )?;
        corruptor.close()?;
        let error = recover_file_storage(&path_string)
            .err()
            .expect("a gap in the durable outbox must fail startup");
        assert!(matches!(
            error,
            StorageError::InvalidData {
                context: "storage_outbox.batch_id",
                ..
            }
        ));

        let inspector = Connection::open(&path_string)?;
        let tick_count: i64 = inspector
            .query_row("SELECT COUNT(*) FROM ticks")?
            .get_typed(0)?;
        let progress = inspector.query_row(
            "SELECT applied_batch_id, durable_batch_id FROM storage_progress WHERE singleton = 1",
        )?;
        assert_eq!(tick_count, 0, "failed recovery applied a later batch");
        assert_eq!(progress.get_typed::<i64>(0)?, 0);
        assert_eq!(progress.get_typed::<i64>(1)?, 0);
        inspector.close()?;
        Ok(())
    }

    #[test]
    fn recovery_rejects_ledger_state_that_contradicts_durable_watermark()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-ledger-state-corruption");
        let path_string = path.to_string_lossy().to_string();
        let mut pipeline = StoragePipeline::create_new_file(&path_string)?;
        let admission = pipeline.submit_with_receipt(&sample_batch(64, 6.4))?;
        pipeline.shutdown()?;

        let corruptor = Connection::open_strict_multi_process(&path_string)?;
        let updated = corruptor.execute_with_params(
            "UPDATE storage_batch_ledger SET state = 'applied' WHERE batch_id = ?1",
            &[admission.batch_id.as_i64().into()],
        )?;
        assert_eq!(updated, 1);
        corruptor.close()?;
        let error = match StoragePipeline::recover_existing(&path_string) {
            Ok(mut recovered) => {
                recovered.shutdown()?;
                return Err("contradictory durable ledger unexpectedly recovered".into());
            }
            Err(error) => error,
        };
        assert!(error.to_string().contains("watermarks require \"durable\""));
        Ok(())
    }

    #[test]
    fn rolled_back_application_keeps_the_exact_outbox_for_recovery()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-outbox-rollback-recovery");
        let path_string = path.to_string_lossy().to_string();
        let batch = sample_batch(41, 4.1);
        let prepared = PreparedPersistenceBatch::from_batch(&batch)?;
        let mut interrupted = create_file_storage(&path_string)?;
        let (admission, newly_admitted) =
            interrupted.stage_outbox(prepared.tick, &prepared.storage)?;
        assert!(newly_admitted);
        assert!(!interrupted.enqueue_staged(admission.batch_id, prepared.storage)?);
        let canonical_metrics_sql: String = interrupted
            .connection()?
            .query_row(
                "SELECT sql FROM sqlite_schema
                 WHERE type = 'table' AND name = 'metrics'",
            )?
            .get_typed(0)?;
        interrupted.connection()?.execute("DROP TABLE metrics")?;
        let failure = interrupted
            .flush()
            .expect_err("missing table must roll back the scientific-table transaction");
        assert!(matches!(
            failure,
            StorageError::Transaction {
                commit_state: FailureCommitState::RolledBack,
                ..
            }
        ));
        let tick_count: i64 = interrupted
            .connection()?
            .query_row("SELECT COUNT(*) FROM ticks")?
            .get_typed(0)?;
        assert_eq!(tick_count, 0);
        assert_eq!(
            interrupted.persistence_watermarks()?.applied,
            None,
            "rolled-back application advanced its watermark"
        );
        interrupted.abandon_after_error();

        let repair = Connection::open(&path_string)?;
        repair.execute(&canonical_metrics_sql)?;
        repair.close()?;
        let mut recovered = StoragePipeline::recover_existing(&path_string)?;
        let shutdown = recovered.shutdown()?;
        assert_eq!(shutdown.watermarks.durable, Some(admission.batch_id));
        let mut durable = recover_file_storage(&path_string)?;
        assert_eq!(durable.max_tick()?, Some(41));
        assert_integrity(&durable)?;
        durable.close()?;
        Ok(())
    }

    #[test]
    fn exact_duplicate_admission_reuses_one_batch_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-outbox-duplicate");
        let path_string = path.to_string_lossy().to_string();
        let batch = sample_batch(51, 5.1);
        let prepared = PreparedPersistenceBatch::from_batch(&batch)?;
        let mut storage = create_file_storage(&path_string)?;
        let (first, first_is_new) = storage.stage_outbox(prepared.tick, &prepared.storage)?;
        let (duplicate, duplicate_is_new) =
            storage.stage_outbox(prepared.tick, &prepared.storage)?;
        assert!(first_is_new);
        assert!(!duplicate_is_new);
        assert_eq!(duplicate.batch_id, first.batch_id);
        assert!(!storage.enqueue_staged(first.batch_id, prepared.storage)?);
        storage.flush()?;
        let watermarks = storage.finalize_applied_outbox()?;
        assert_eq!(watermarks.durable, Some(first.batch_id));
        let ledger_count: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM storage_batch_ledger")?
            .get_typed(0)?;
        assert_eq!(ledger_count, 1);

        assert_integrity(&storage)?;
        storage.close()?;
        let prepared = PreparedPersistenceBatch::from_batch(&batch)?;
        let mut reopened = recover_file_storage(&path_string)?;
        let (duplicate_after_reopen, duplicate_after_reopen_is_new) =
            reopened.stage_outbox(prepared.tick, &prepared.storage)?;
        assert!(!duplicate_after_reopen_is_new);
        assert_eq!(duplicate_after_reopen.batch_id, first.batch_id);

        let conflicting = Storage::prepare_batch(&sample_batch(51, 99.0))?;
        assert!(matches!(
            reopened.stage_outbox(51, &conflicting),
            Err(StorageError::InvalidData {
                context: "storage_batch_ledger.payload_digest",
                ..
            })
        ));
        let ledger_count: i64 = reopened
            .connection()?
            .query_row("SELECT COUNT(*) FROM storage_batch_ledger")?
            .get_typed(0)?;
        assert_eq!(ledger_count, 1);
        assert_integrity(&reopened)?;
        reopened.close()?;
        Ok(())
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
            storage.persist(&sample_batch(42, 5.5)),
            Err(StorageError::TerminallyFailed)
        ));
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

        let admission = sink.submit_with_receipt(&sample_batch(7, 2.5))?;
        assert_eq!(admission.guarantee, PersistenceGuarantee::Durable);
        assert_eq!(admission.watermarks.admitted, Some(admission.batch_id));
        assert_eq!(admission.watermarks.applied, None);
        assert_eq!(admission.watermarks.durable, None);
        assert_eq!(reader.run_ledger_summary()?.tick_count, 0);

        let flush = pipeline.flush_and_wait()?;
        assert_eq!(flush.committed_tick, Some(7));
        assert_eq!(flush.guarantee, PersistenceGuarantee::Durable);
        assert_eq!(flush.watermarks.applied, Some(admission.batch_id));
        assert_eq!(flush.watermarks.durable, Some(admission.batch_id));
        assert_eq!(reader.persistence_watermarks()?, flush.watermarks);
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
    fn world_persistence_preserves_indeterminate_acknowledgement_loss()
    -> Result<(), Box<dyn std::error::Error>> {
        let (tx, rx) = xchan::bounded(1);
        let mut sink = StorageSink {
            tx,
            analytics: AnalyticsSnapshotProvider::empty(),
            admission: Arc::new(Mutex::new(AdmissionState { open: true })),
            path: Arc::from(":memory:"),
            deadlines: StorageDeadlines::default(),
        };
        let worker = thread::spawn(move || -> Result<(), std::io::Error> {
            match rx
                .recv()
                .map_err(|error| std::io::Error::other(error.to_string()))?
            {
                StorageCommand::Persist { reply, .. } => drop(reply),
                _ => return Err(std::io::Error::other("unexpected storage command")),
            }
            Ok(())
        });

        let error = sink
            .on_tick(&sample_batch(81, 8.1))
            .expect_err("lost worker acknowledgement must remain typed");
        assert_eq!(error.tick(), 81);
        assert_eq!(error.state(), PersistenceAdmissionState::Indeterminate);
        worker
            .join()
            .map_err(|panic| std::io::Error::other(format!("worker panicked: {panic:?}")))??;
        Ok(())
    }

    #[test]
    fn full_queue_and_contended_gate_have_bounded_definite_non_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let (tx, rx) = xchan::bounded(1);
        let (dummy_reply, _dummy_receiver) = xchan::bounded(1);
        tx.send(StorageCommand::Flush { reply: dummy_reply })?;
        let sink = StorageSink {
            tx,
            analytics: AnalyticsSnapshotProvider::empty(),
            admission: Arc::new(Mutex::new(AdmissionState { open: true })),
            path: Arc::from(":memory:"),
            deadlines: short_deadlines(),
        };
        let started = Instant::now();
        let error = sink
            .submit(&sample_batch(82, 8.2))
            .expect_err("a full queue must hit its enqueue deadline");
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(matches!(
            error,
            StorageError::Worker(StorageWorkerError::Timeout {
                operation: StorageOperation::Admit,
                phase: StorageWaitPhase::CommandEnqueue,
                commit_state: FailureCommitState::NotAdmitted,
                ..
            })
        ));
        drop(rx);

        let (tx, _rx) = xchan::bounded(1);
        let mut sink = StorageSink {
            tx,
            analytics: AnalyticsSnapshotProvider::empty(),
            admission: Arc::new(Mutex::new(AdmissionState { open: true })),
            path: Arc::from(":memory:"),
            deadlines: short_deadlines(),
        };
        let admission = Arc::clone(&sink.admission);
        let guard = admission.lock().expect("test admission gate");
        let started = Instant::now();
        let error = sink
            .on_tick(&sample_batch(83, 8.3))
            .expect_err("a contended gate must hit its deadline");
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(error.state(), PersistenceAdmissionState::NotAdmitted);
        assert!(error.detail().contains("AdmissionGate"));
        drop(guard);
        Ok(())
    }

    #[test]
    fn admission_ack_timeout_is_indeterminate_and_exact_retry_is_idempotent()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-admission-timeout-retry");
        let path_string = path.to_string_lossy().to_string();
        let mut pipeline = StoragePipeline::create_new_file_with_thresholds_and_deadlines(
            &path_string,
            64,
            4096,
            1024,
            1024,
            short_deadlines(),
        )?;
        let sink = pipeline.sink();
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

        let error = sink
            .submit_with_receipt(&sample_batch(84, 8.4))
            .expect_err("paused worker must miss the admission acknowledgement deadline");
        assert!(matches!(
            error,
            StorageError::Worker(StorageWorkerError::Timeout {
                operation: StorageOperation::Admit,
                phase: StorageWaitPhase::Acknowledgement,
                commit_state: FailureCommitState::Indeterminate,
                ..
            })
        ));
        assert!(!pipeline.analytics_provider().snapshot().stopped);
        release_tx.send(())?;

        let retry = sink.submit_with_receipt(&sample_batch(84, 8.4))?;
        let flush = pipeline.flush_and_wait()?;
        assert_eq!(flush.watermarks.durable, Some(retry.batch_id));
        pipeline.shutdown()?;
        let reader = StorageReader::open(&path_string)?;
        assert_eq!(reader.run_ledger_summary()?.tick_count, 1);
        reader.close()?;
        Ok(())
    }

    #[test]
    fn flush_and_shutdown_timeouts_are_bounded_and_retry_the_original_barrier()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut flush_pipeline = StoragePipeline::memory_with_thresholds_and_deadlines(
            64,
            4096,
            1024,
            1024,
            short_deadlines(),
        )?;
        let (entered_tx, entered_rx) = xchan::bounded(1);
        let (release_tx, release_rx) = xchan::bounded(1);
        flush_pipeline
            .sink
            .tx
            .send(StorageCommand::PauseForAdmissionRace {
                entered: entered_tx,
                release: release_rx,
            })?;
        entered_rx.recv_timeout(Duration::from_secs(2))?;
        let error = flush_pipeline
            .flush_and_wait()
            .expect_err("paused worker must miss the flush deadline");
        assert!(matches!(
            error,
            StorageError::Worker(StorageWorkerError::Timeout {
                operation: StorageOperation::Flush,
                phase: StorageWaitPhase::Acknowledgement,
                ..
            })
        ));
        assert!(!flush_pipeline.analytics_provider().snapshot().stopped);
        release_tx.send(())?;
        flush_pipeline.flush_and_wait()?;
        flush_pipeline.shutdown()?;

        let mut shutdown_pipeline = StoragePipeline::memory_with_thresholds_and_deadlines(
            64,
            4096,
            1024,
            1024,
            short_deadlines(),
        )?;
        let retained_sink = shutdown_pipeline.sink();
        let (entered_tx, entered_rx) = xchan::bounded(1);
        let (release_tx, release_rx) = xchan::bounded(1);
        shutdown_pipeline
            .sink
            .tx
            .send(StorageCommand::PauseForAdmissionRace {
                entered: entered_tx,
                release: release_rx,
            })?;
        entered_rx.recv_timeout(Duration::from_secs(2))?;
        let error = shutdown_pipeline
            .shutdown()
            .expect_err("paused worker must miss the shutdown deadline");
        assert!(matches!(
            error,
            StorageError::Worker(StorageWorkerError::Timeout {
                operation: StorageOperation::Shutdown,
                phase: StorageWaitPhase::Acknowledgement,
                ..
            })
        ));
        assert!(shutdown_pipeline.handle.is_some());
        assert!(shutdown_pipeline.pending_shutdown.is_some());
        assert!(matches!(
            retained_sink.submit(&sample_batch(85, 8.5)),
            Err(StorageError::Worker(StorageWorkerError::Channel {
                operation: StorageOperation::Admit,
                commit_state: FailureCommitState::NotAdmitted,
                ..
            }))
        ));
        release_tx.send(())?;
        shutdown_pipeline.shutdown()?;
        Ok(())
    }

    #[test]
    fn timed_out_shutdown_drop_hands_worker_to_supervised_reaper()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-shutdown-reaper");
        let path_string = path.to_string_lossy().to_string();
        let mut pipeline = StoragePipeline::create_new_file_with_thresholds_and_deadlines(
            &path_string,
            64,
            4096,
            1024,
            1024,
            short_deadlines(),
        )?;
        let analytics = pipeline.analytics_provider();
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
        assert!(pipeline.shutdown().is_err());
        let started = Instant::now();
        drop(pipeline);
        assert!(started.elapsed() < Duration::from_secs(2));
        release_tx.send(())?;

        let deadline = Instant::now() + Duration::from_secs(2);
        while !analytics.snapshot().stopped {
            assert!(
                Instant::now() < deadline,
                "storage reaper never joined worker"
            );
            thread::sleep(Duration::from_millis(10));
        }
        let reader = StorageReader::open(&path_string)?;
        assert_eq!(reader.run_ledger_summary()?.tick_count, 0);
        reader.close()?;
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

        let mut queued_submits = Vec::with_capacity(DEFAULT_COMMAND_CAPACITY);
        for tick in 1..=DEFAULT_COMMAND_CAPACITY as u64 {
            let sink = pipeline.sink();
            queued_submits.push(thread::spawn(move || {
                sink.submit(&sample_batch(tick, tick as f32))
            }));
        }
        let queue_deadline = Instant::now() + Duration::from_secs(2);
        while pipeline.sink.tx.len() < DEFAULT_COMMAND_CAPACITY {
            assert!(
                Instant::now() < queue_deadline,
                "admission commands never filled the bounded worker queue"
            );
            thread::yield_now();
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
                    return Err(std::io::Error::other(format!(
                        "admission gate poisoned before race: {error}"
                    ))
                    .into());
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

        for submit in queued_submits {
            submit
                .join()
                .expect("queued submit thread must not panic")
                .expect("queued submit ordered before shutdown must be admitted");
        }

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
        let admission = pipeline.submit_with_receipt(&sample_batch(7, 2.5))?;
        assert_eq!(admission.guarantee, PersistenceGuarantee::CommittedVolatile);

        let flush = pipeline.flush_and_wait()?;
        assert_eq!(flush.committed_tick, Some(7));
        assert_eq!(flush.guarantee, PersistenceGuarantee::CommittedVolatile);
        assert_eq!(flush.watermarks.admitted, Some(admission.batch_id));
        assert_eq!(flush.watermarks.applied, Some(admission.batch_id));
        assert_eq!(flush.watermarks.durable, None);
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
