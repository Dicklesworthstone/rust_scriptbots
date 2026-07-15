#![cfg_attr(windows, feature(windows_by_handle))]

//! FrankenSQLite-backed persistence layer for ScriptBots.

use arc_swap::ArcSwap;
use crossbeam_channel as xchan;
use fsqlite::{
    Connection, FileIdentity, FrankenError, Row, SqliteValue,
    compat::{FromSqliteValue, OpenFlags, RowExt, Transaction, TransactionExt, open_with_flags},
    migrate::MigrationRunner,
};
use scriptbots_core::{
    AgentState, AgentUid, BirthOrigin, BirthRecord, BrainBinding, DeathCause, DeathRecord,
    Generation, PersistenceAdmissionError, PersistenceAdmissionState, PersistenceBatch,
    PersistenceEventKind, ReplayAgentPhase, ReplayEvent, ReplayEventKind, ReplayRngScope, Tick,
    WorldPersistence,
    ancestry::{AncestryError, AncestryGraph},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{self, Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
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
const OUTBOX_PAYLOAD_VERSION: u32 = 3;
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

/// Canonical one-shot DDL for the production ScriptBots scientific workload tables.
///
/// This export is versioned independently of the historical migration ledger. It exists for
/// conformance harnesses that need the exact current table constraints without copying a weaker
/// approximation of the production schema. Storage recovery continues to use the immutable
/// migration sequence below.
pub const SCRIPTBOTS_SCHEMA_V1: &str = r#"
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
        agent_uid INTEGER CHECK (agent_uid IS NULL OR agent_uid >= 0),
        scope TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        PRIMARY KEY (tick, seq)
    );
    CREATE TABLE agents (
        tick INTEGER NOT NULL CHECK (tick >= 0),
        agent_uid INTEGER NOT NULL CHECK (agent_uid >= 0),
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
        PRIMARY KEY (tick, agent_uid)
    );
    CREATE TABLE births (
        tick INTEGER NOT NULL,
        agent_uid INTEGER NOT NULL,
        spawn_ordinal INTEGER NOT NULL,
        birth_ordinal INTEGER,
        parent_a INTEGER,
        parent_b INTEGER,
        brain_kind TEXT,
        brain_key INTEGER,
        herbivore_tendency REAL NOT NULL,
        generation INTEGER NOT NULL,
        position_x REAL NOT NULL,
        position_y REAL NOT NULL,
        is_hybrid INTEGER NOT NULL,
        origin TEXT NOT NULL,
        PRIMARY KEY (tick, agent_uid),
        CHECK (tick >= 0),
        CHECK (agent_uid >= 0),
        CHECK (spawn_ordinal >= 0),
        CHECK ((birth_ordinal IS NULL) OR (birth_ordinal >= 0)),
        CHECK ((parent_a IS NULL) OR (parent_a >= 0)),
        CHECK ((parent_b IS NULL) OR (parent_b >= 0)),
        CHECK ((brain_key IS NULL) OR (brain_key >= 0)),
        CHECK (generation >= 0),
        CHECK (is_hybrid IN (0, 1)),
        CHECK (origin IN ('born', 'seeded', 'injected')),
        CHECK ((origin != 'seeded') OR (tick = 0)),
        CHECK (
            ((origin = 'born') AND (birth_ordinal IS NOT NULL))
            OR ((origin IN ('seeded', 'injected')) AND (birth_ordinal IS NULL))
        )
    );
    CREATE UNIQUE INDEX births_agent_uid_unique ON births (agent_uid);
    CREATE UNIQUE INDEX births_spawn_ordinal_unique ON births (spawn_ordinal);
    CREATE UNIQUE INDEX births_birth_ordinal_unique ON births (birth_ordinal);
    CREATE TABLE deaths (
        tick INTEGER NOT NULL CHECK (tick >= 0),
        agent_uid INTEGER NOT NULL CHECK (agent_uid >= 0),
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
        PRIMARY KEY (tick, agent_uid)
    );
    CREATE UNIQUE INDEX deaths_agent_uid_unique ON deaths (agent_uid);
"#;

const SCRIPTBOTS_SCHEMA_V3: &str = "
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
        agent_uid INTEGER CHECK (agent_uid IS NULL OR agent_uid >= 0),
        scope TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        PRIMARY KEY (tick, seq)
    );
    CREATE TABLE agents (
        tick INTEGER NOT NULL CHECK (tick >= 0),
        agent_uid INTEGER NOT NULL CHECK (agent_uid >= 0),
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
        PRIMARY KEY (tick, agent_uid)
    );
    CREATE TABLE births (
        tick INTEGER NOT NULL CHECK (tick >= 0),
        agent_uid INTEGER NOT NULL CHECK (agent_uid >= 0),
        spawn_ordinal INTEGER NOT NULL CHECK (spawn_ordinal >= 0),
        birth_ordinal INTEGER NOT NULL CHECK (birth_ordinal >= 0),
        parent_a INTEGER CHECK (parent_a IS NULL OR parent_a >= 0),
        parent_b INTEGER CHECK (parent_b IS NULL OR parent_b >= 0),
        brain_kind TEXT,
        brain_key INTEGER CHECK (brain_key IS NULL OR brain_key >= 0),
        herbivore_tendency REAL NOT NULL,
        generation INTEGER NOT NULL CHECK (generation >= 0),
        position_x REAL NOT NULL,
        position_y REAL NOT NULL,
        is_hybrid INTEGER NOT NULL CHECK (is_hybrid IN (0, 1)),
        PRIMARY KEY (tick, agent_uid)
    );
    CREATE TABLE deaths (
        tick INTEGER NOT NULL CHECK (tick >= 0),
        agent_uid INTEGER NOT NULL CHECK (agent_uid >= 0),
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
        PRIMARY KEY (tick, agent_uid)
    );
";

const SCRIPTBOTS_SCHEMA_V4: &str = "
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

const SCRIPTBOTS_SCHEMA_V5: &str = "
    CREATE TABLE births_v5 (
        tick INTEGER NOT NULL CHECK (tick >= 0),
        agent_uid INTEGER NOT NULL CHECK (agent_uid >= 0),
        spawn_ordinal INTEGER NOT NULL CHECK (spawn_ordinal >= 0),
        birth_ordinal INTEGER CHECK (birth_ordinal IS NULL OR birth_ordinal >= 0),
        parent_a INTEGER CHECK (parent_a IS NULL OR parent_a >= 0),
        parent_b INTEGER CHECK (parent_b IS NULL OR parent_b >= 0),
        brain_kind TEXT,
        brain_key INTEGER CHECK (brain_key IS NULL OR brain_key >= 0),
        herbivore_tendency REAL NOT NULL,
        generation INTEGER NOT NULL CHECK (generation >= 0),
        position_x REAL NOT NULL,
        position_y REAL NOT NULL,
        is_hybrid INTEGER NOT NULL CHECK (is_hybrid IN (0, 1)),
        origin TEXT NOT NULL CHECK (origin IN ('born', 'seeded', 'injected')),
        CHECK (origin <> 'seeded' OR tick = 0),
        CHECK (
            (origin = 'born' AND birth_ordinal IS NOT NULL)
            OR (origin IN ('seeded', 'injected') AND birth_ordinal IS NULL)
        ),
        PRIMARY KEY (tick, agent_uid)
    );
    INSERT INTO births_v5 (
        tick, agent_uid, spawn_ordinal, birth_ordinal, parent_a, parent_b,
        brain_kind, brain_key, herbivore_tendency, generation,
        position_x, position_y, is_hybrid, origin
    )
    SELECT
        tick, agent_uid, spawn_ordinal, birth_ordinal, parent_a, parent_b,
        brain_kind, brain_key, herbivore_tendency, generation,
        position_x, position_y, is_hybrid, 'born'
    FROM births;
    DROP TABLE births;
    ALTER TABLE births_v5 RENAME TO births;
    CREATE UNIQUE INDEX births_agent_uid_unique ON births (agent_uid);
    CREATE UNIQUE INDEX births_spawn_ordinal_unique ON births (spawn_ordinal);
    CREATE UNIQUE INDEX births_birth_ordinal_unique ON births (birth_ordinal);
    CREATE UNIQUE INDEX deaths_agent_uid_unique ON deaths (agent_uid);
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
        .add(3, "create_stable_agent_uid_schema", SCRIPTBOTS_SCHEMA_V3)
        .add(
            4,
            "create_stable_uid_persistence_outbox",
            SCRIPTBOTS_SCHEMA_V4,
        )
        .add(5, "record_birth_origin", SCRIPTBOTS_SCHEMA_V5)
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
    "agent_uid",
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

/// Number of values bound by [`scriptbots_agent_insert_sql`].
pub const SCRIPTBOTS_AGENT_COLUMN_COUNT: usize = AGENT_COLUMNS.len();

/// Canonical production insert statement for one full scientific agent snapshot.
///
/// The statement is generated from the same ordered column list used by [`Storage`], so external
/// conformance tests cannot silently drift to a different column order or placeholder count.
#[must_use]
pub fn scriptbots_agent_insert_sql() -> &'static str {
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

/// Storage error wrapper.
#[derive(Debug, Error)]
pub enum StorageError {
    /// The batch was refused BEFORE it was prepared, because it is too large.
    ///
    /// This is a `NotAdmitted` outcome in the strictest sense: nothing was
    /// allocated, nothing was enqueued, and the caller still holds the exact
    /// payload it tried to submit, so the retry semantics are untouched.
    #[error(
        "batch at tick {tick} is too large to admit: {bytes} bytes across {events} records exceeds the cap of {max_bytes} bytes / {max_events} records"
    )]
    PayloadTooLarge {
        /// Which tick was refused.
        tick: u64,
        /// Estimated size of the payload.
        bytes: usize,
        /// Record count across every vector in the batch.
        events: usize,
        /// The byte ceiling.
        max_bytes: usize,
        /// The record ceiling.
        max_events: usize,
    },
    /// Admitting this batch would push the in-flight buffer past its byte ceiling.
    #[error(
        "batch at tick {tick} would push in-flight persistence to {would_be} bytes, over the cap of {max_inflight}"
    )]
    InFlightBytesExhausted {
        /// Which tick was refused.
        tick: u64,
        /// What the in-flight total would have become.
        would_be: usize,
        /// The ceiling.
        max_inflight: usize,
    },
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
    /// Demographic reproduction rows only (`origin = 'born'`).
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
    agent_uid: i64,
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
    agent_uid: i64,
    spawn_ordinal: i64,
    birth_ordinal: Option<i64>,
    parent_a: Option<i64>,
    parent_b: Option<i64>,
    brain_kind: Option<String>,
    brain_key: Option<i64>,
    herbivore_tendency: f64,
    generation: i64,
    position_x: f64,
    position_y: f64,
    is_hybrid: bool,
    origin: BirthOrigin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeathRow {
    tick: i64,
    agent_uid: i64,
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
    agent_uid: Option<i64>,
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

/// One agent-arrival ancestry edge, read back out of the run database.
///
/// The physical `births` row is the edge for every origin: the arriving agent,
/// both optional parents, and everything the graph needs to reconstruct the node.
/// The historical type name follows that table name; it is not limited to
/// [`BirthOrigin::Born`] records.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistedAncestryBirth {
    /// When the agent entered the world.
    pub tick: Tick,
    /// The arriving agent's logical identity — never a slot handle.
    pub agent_uid: AgentUid,
    /// Monotonic ordinal among every successful agent insertion.
    pub spawn_ordinal: u64,
    /// Monotonic demographic-birth ordinal, present only for born agents.
    pub birth_ordinal: Option<u64>,
    /// First parent, if any.
    pub parent_a: Option<AgentUid>,
    /// Second parent, if any.
    pub parent_b: Option<AgentUid>,
    /// Generations since a root.
    pub generation: Generation,
    /// Which brain it ran.
    pub brain_key: Option<u64>,
    /// Whether it was a hybrid.
    pub is_hybrid: bool,
    /// How the agent entered the world.
    pub origin: BirthOrigin,
}

/// One death, read back out of the run database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistedAncestryDeath {
    /// When it died.
    pub tick: Tick,
    /// Who died.
    pub agent_uid: AgentUid,
    /// The exact cause recorded by the live run.
    pub cause: DeathCause,
}

/// Rebuild an ancestry graph from a run database alone.
///
/// THIS FUNCTION IS THE JUSTIFICATION FOR THE STORAGE LAYER. If a graph rebuilt
/// from nothing but the persisted rows is not identical to the one the live run
/// held, then the run database is not sufficient for offline science, and every
/// claim that a run can be analysed after the fact is a claim we cannot back.
/// The test that compares the two by `canonical_digest` is what turns that from
/// an aspiration into a checked property.
///
/// Each reconstructed [`BirthRecord`] carries the exact persisted insertion and
/// demographic ordinals. The graph does not currently hash those fields, but
/// fabricating placeholders here would make this record-rebuild boundary lossy.
///
/// Deaths are applied AFTER all arrival records, deliberately. An agent can die
/// on the same tick another enters the world, and the two tables are ordered
/// independently; if a death were applied before the arrival it terminates, the
/// rebuild would report an unknown-uid error for a log that is perfectly well
/// formed.
///
/// # Errors
///
/// [`AncestryError`] if the persisted log is not a well-formed ancestry — which
/// would itself be a finding: it would mean the writer emitted rows the graph's
/// invariants reject.
pub fn rebuild_ancestry(
    births: &[PersistedAncestryBirth],
    deaths: &[PersistedAncestryDeath],
) -> Result<AncestryGraph, AncestryError> {
    let mut graph = AncestryGraph::new();
    for birth in births {
        graph.apply_birth(&BirthRecord {
            tick: birth.tick,
            agent_uid: birth.agent_uid,
            spawn_ordinal: birth.spawn_ordinal,
            birth_ordinal: birth.birth_ordinal,
            parent_a: birth.parent_a,
            parent_b: birth.parent_b,
            brain_kind: None,
            brain_key: birth.brain_key,
            herbivore_tendency: 0.0,
            generation: birth.generation,
            position: scriptbots_core::Position::new(0.0, 0.0),
            is_hybrid: birth.is_hybrid,
            origin: birth.origin,
        })?;
    }
    for death in deaths {
        // A death for an agent this run never recorded an arrival for is a hole in
        // the log, not something to paper over.
        graph.apply_death(&DeathRecord {
            tick: death.tick,
            agent_uid: death.agent_uid,
            age: 0,
            generation: Generation(0),
            herbivore_tendency: 0.0,
            brain_kind: None,
            brain_key: None,
            energy: 0.0,
            food_balance_total: 0.0,
            cause: death.cause,
            was_hybrid: false,
            combat_flags: scriptbots_core::CombatEventFlags::default(),
        })?;
    }
    Ok(graph)
}

/// Replay event reconstructed from persisted storage.
#[derive(Debug, Clone)]
pub struct PersistedReplayEvent {
    pub tick: u64,
    pub seq: u64,
    pub event: ReplayEvent,
}

/// Holds a reservation on the in-flight byte counter and releases it on drop.
///
/// RAII rather than a manual decrement, because "released exactly once on commit,
/// refusal, timeout handoff, crash, and shutdown" is a requirement that a chain of
/// hand-written `fetch_sub` calls WILL eventually violate — one early return that
/// forgets it, and the counter creeps up until the sink refuses everything and
/// persistence dies quietly in a long run rather than loudly in a test.
struct InFlightPermit {
    counter: Arc<AtomicUsize>,
    bytes: usize,
}

impl Drop for InFlightPermit {
    fn drop(&mut self) {
        self.counter.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

/// Ceilings on what persistence will admit.
///
/// The queue used to be bounded by COUNT alone, which bounds nothing that matters:
/// a single batch carrying a hundred thousand agent rows is one command, sails
/// through a count-based gate, and is fully materialized by
/// `PreparedPersistenceBatch::from_batch` BEFORE any deadline or admission check
/// can refuse it. The memory is gone by the time anyone gets a say.
///
/// So the size is measured FIRST, from the batch's own shape, and an oversized
/// batch is refused before a single row is allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadBudget {
    /// Largest single batch, in estimated bytes.
    pub max_batch_bytes: usize,
    /// Largest single batch, in records across every vector.
    pub max_batch_events: usize,
    /// Largest total in-flight (buffered, not yet flushed) payload, in bytes.
    pub max_inflight_bytes: usize,
}

impl Default for PayloadBudget {
    fn default() -> Self {
        Self {
            // 64 MiB is generous for one tick and still bounded: a batch this
            // large is pathological, and pathological is exactly what must be
            // refused rather than allocated.
            max_batch_bytes: 64 << 20,
            max_batch_events: 1_000_000,
            // 256 MiB of buffered payload is the ceiling before back-pressure.
            max_inflight_bytes: 256 << 20,
        }
    }
}

/// Estimated bytes and record count for a batch, WITHOUT allocating it.
///
/// Deterministic and cheap: it reads the vector lengths and multiplies by each
/// row's in-memory size. It deliberately does NOT serialize — serializing to find
/// out whether something is too big to serialize is the bug, not the check.
///
/// The estimate is an approximation of the heap the prepared batch will occupy,
/// and it does not need to be exact. It needs to be MONOTONIC in the batch's size
/// and computable in constant time, which it is.
#[must_use]
pub fn estimate_batch_size(payload: &PersistenceBatch) -> (usize, usize) {
    let events = payload.metrics.len()
        + payload.events.len()
        + payload.agents.len()
        + payload.births.len()
        + payload.deaths.len()
        + payload.replay_events.len();

    // Per-record sizes, plus a per-record allowance for the strings and JSON each
    // row carries on the heap. Fixed constants keep this deterministic across
    // platforms and feature sets, which a `size_of` chain would not be.
    const METRIC_BYTES: usize = 96;
    const EVENT_BYTES: usize = 256;
    const AGENT_BYTES: usize = 512;
    const BIRTH_BYTES: usize = 192;
    const DEATH_BYTES: usize = 192;
    const REPLAY_BYTES: usize = 512;
    const SUMMARY_BYTES: usize = 256;

    let bytes = SUMMARY_BYTES
        + payload.metrics.len().saturating_mul(METRIC_BYTES)
        + payload.events.len().saturating_mul(EVENT_BYTES)
        + payload.agents.len().saturating_mul(AGENT_BYTES)
        + payload.births.len().saturating_mul(BIRTH_BYTES)
        + payload.deaths.len().saturating_mul(DEATH_BYTES)
        + payload.replay_events.len().saturating_mul(REPLAY_BYTES);

    (bytes, events)
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

#[derive(Debug, Deserialize)]
struct OutboxPayloadEnvelope {
    version: u32,
    tick: u64,
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

    fn validate_contents(&self, enclosing_tick: u64) -> Result<(), StorageError> {
        let invalid = |context: &'static str, value: f64| StorageError::InvalidData {
            context,
            reason: format!("non-finite value {value}"),
        };
        let [summary] = self.ticks.as_slice() else {
            return Err(StorageError::InvalidData {
                context: "ticks",
                reason: format!(
                    "an outbox batch must contain exactly one tick summary, found {}",
                    self.ticks.len()
                ),
            });
        };
        let summary_tick = checked_u64("ticks.tick", summary.tick)?;
        if summary_tick != enclosing_tick {
            return Err(StorageError::InvalidData {
                context: "ticks.tick",
                reason: format!(
                    "tick summary {summary_tick} does not match enclosing batch tick {enclosing_tick}"
                ),
            });
        }
        checked_u64("ticks.epoch", summary.epoch)?;
        checked_usize("ticks.agent_count", summary.agent_count)?;
        let summary_births = checked_usize("ticks.births", summary.births)?;
        let summary_deaths = checked_usize("ticks.deaths", summary.deaths)?;
        for (context, value) in [
            ("ticks.total_energy", summary.total_energy),
            ("ticks.average_energy", summary.average_energy),
            ("ticks.average_health", summary.average_health),
        ] {
            if !value.is_finite() {
                return Err(invalid(context, value));
            }
        }
        for row in &self.metrics {
            if !row.value.is_finite() {
                return Err(invalid("metrics.value", row.value));
            }
        }
        let mut birth_event_rows = 0usize;
        let mut birth_event_total = 0usize;
        let mut death_event_rows = 0usize;
        let mut death_event_total = 0usize;
        for row in &self.events {
            let row_tick = checked_u64("events.tick", row.tick)?;
            if row_tick != enclosing_tick {
                return Err(StorageError::InvalidData {
                    context: "events.tick",
                    reason: format!(
                        "event tick {row_tick} does not match enclosing batch tick {enclosing_tick}"
                    ),
                });
            }
            let count = checked_usize("events.count", row.count)?;
            match row.kind.as_str() {
                "births" => {
                    birth_event_rows += 1;
                    birth_event_total = birth_event_total.checked_add(count).ok_or_else(|| {
                        StorageError::InvalidData {
                            context: "events.births",
                            reason: "birth event total overflow".to_owned(),
                        }
                    })?;
                }
                "deaths" => {
                    death_event_rows += 1;
                    death_event_total = death_event_total.checked_add(count).ok_or_else(|| {
                        StorageError::InvalidData {
                            context: "events.deaths",
                            reason: "death event total overflow".to_owned(),
                        }
                    })?;
                }
                _ => {}
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
        let mut birth_agent_uids = BTreeSet::new();
        let mut birth_spawn_ordinals = BTreeSet::new();
        let mut birth_ordinals = BTreeSet::new();
        for row in &self.births {
            let row_tick = checked_u64("births.tick", row.tick)?;
            validate_lifecycle_record_tick("births.tick", row_tick, enclosing_tick)?;
            let agent_uid = checked_u64("births.agent_uid", row.agent_uid)?;
            validate_birth_origin_tick(row.origin, row_tick, agent_uid)?;
            checked_u64("births.spawn_ordinal", row.spawn_ordinal)?;
            if let Some(parent_a) = row.parent_a {
                checked_u64("births.parent_a", parent_a)?;
            }
            if let Some(parent_b) = row.parent_b {
                checked_u64("births.parent_b", parent_b)?;
            }
            checked_u32("births.generation", row.generation)?;
            if !birth_agent_uids.insert(row.agent_uid) {
                return Err(StorageError::InvalidData {
                    context: "births.agent_uid",
                    reason: format!(
                        "agent uid {} has more than one arrival in the enclosing batch",
                        row.agent_uid
                    ),
                });
            }
            if !birth_spawn_ordinals.insert(row.spawn_ordinal) {
                return Err(StorageError::InvalidData {
                    context: "births.spawn_ordinal",
                    reason: format!(
                        "spawn ordinal {} has more than one arrival in the enclosing batch",
                        row.spawn_ordinal
                    ),
                });
            }
            let birth_ordinal = row
                .birth_ordinal
                .map(|raw| checked_u64("births.birth_ordinal", raw))
                .transpose()?;
            validate_birth_origin_ordinal(row.origin, birth_ordinal)?;
            if let Some(birth_ordinal) = row.birth_ordinal
                && !birth_ordinals.insert(birth_ordinal)
            {
                return Err(StorageError::InvalidData {
                    context: "births.birth_ordinal",
                    reason: format!(
                        "birth ordinal {birth_ordinal} has more than one birth in the enclosing batch"
                    ),
                });
            }
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
        let mut death_uids = BTreeSet::new();
        for row in &self.deaths {
            let row_tick = checked_u64("deaths.tick", row.tick)?;
            validate_lifecycle_record_tick("deaths.tick", row_tick, enclosing_tick)?;
            checked_u64("deaths.agent_uid", row.agent_uid)?;
            checked_u32("deaths.age", row.age)?;
            checked_u32("deaths.generation", row.generation)?;
            if !death_uids.insert(row.agent_uid) {
                return Err(StorageError::InvalidData {
                    context: "deaths.agent_uid",
                    reason: format!(
                        "agent uid {} has more than one death in the enclosing batch",
                        row.agent_uid
                    ),
                });
            }
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
        let born_records = self
            .births
            .iter()
            .filter(|row| row.origin == BirthOrigin::Born)
            .count();
        if summary_births != born_records {
            return Err(StorageError::InvalidData {
                context: "ticks.births",
                reason: format!(
                    "tick summary reports {summary_births} demographic births, but the batch carries {born_records} Born origin rows"
                ),
            });
        }
        if summary_deaths != self.deaths.len() {
            return Err(StorageError::InvalidData {
                context: "ticks.deaths",
                reason: format!(
                    "tick summary reports {summary_deaths} deaths, but the batch carries {} death rows",
                    self.deaths.len()
                ),
            });
        }
        for (context, expected, rows, total) in [
            (
                "events.births",
                summary_births,
                birth_event_rows,
                birth_event_total,
            ),
            (
                "events.deaths",
                summary_deaths,
                death_event_rows,
                death_event_total,
            ),
        ] {
            let expected_rows = usize::from(expected > 0);
            if rows != expected_rows || total != expected {
                return Err(StorageError::InvalidData {
                    context,
                    reason: format!(
                        "expected {expected_rows} canonical event row(s) totaling {expected}, found {rows} row(s) totaling {total}"
                    ),
                });
            }
        }
        Ok(())
    }

    fn encode_outbox(&self, tick: u64) -> Result<(String, String), StorageError> {
        self.validate_contents(tick)?;
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
        let envelope: OutboxPayloadEnvelope =
            serde_json::from_str(payload).map_err(|error| StorageError::InvalidData {
                context: "storage_outbox.payload",
                reason: error.to_string(),
            })?;
        if envelope.version != OUTBOX_PAYLOAD_VERSION {
            return Err(StorageError::InvalidData {
                context: "storage_outbox.payload.version",
                reason: format!(
                    "unsupported version {}, expected {}",
                    envelope.version, OUTBOX_PAYLOAD_VERSION
                ),
            });
        }
        if envelope.tick != expected_tick {
            return Err(StorageError::InvalidData {
                context: "storage_outbox.payload.tick",
                reason: format!(
                    "ledger tick {expected_tick}, payload tick {}",
                    envelope.tick
                ),
            });
        }
        let decoded: OutboxPayload =
            serde_json::from_str(payload).map_err(|error| StorageError::InvalidData {
                context: "storage_outbox.payload",
                reason: error.to_string(),
            })?;
        debug_assert_eq!(decoded.version, envelope.version);
        debug_assert_eq!(decoded.tick, envelope.tick);
        decoded.storage.validate_contents(expected_tick)?;
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

fn decode_birth_origin(value: &str) -> Result<BirthOrigin, StorageError> {
    match value {
        "born" => Ok(BirthOrigin::Born),
        "seeded" => Ok(BirthOrigin::Seeded),
        "injected" => Ok(BirthOrigin::Injected),
        other => Err(StorageError::InvalidData {
            context: "births.origin",
            reason: format!("unknown birth origin {other:?}"),
        }),
    }
}

fn validate_birth_origin_ordinal(
    origin: BirthOrigin,
    birth_ordinal: Option<u64>,
) -> Result<(), StorageError> {
    match (origin, birth_ordinal) {
        (BirthOrigin::Born, None) => Err(StorageError::InvalidData {
            context: "births.birth_ordinal",
            reason: "born origin requires a demographic birth ordinal".to_owned(),
        }),
        (BirthOrigin::Seeded | BirthOrigin::Injected, Some(_)) => Err(StorageError::InvalidData {
            context: "births.birth_ordinal",
            reason: format!(
                "{} origin must not carry a demographic birth ordinal",
                origin.as_str()
            ),
        }),
        (BirthOrigin::Born, Some(_)) | (BirthOrigin::Seeded | BirthOrigin::Injected, None) => {
            Ok(())
        }
    }
}

fn validate_birth_origin_tick(
    origin: BirthOrigin,
    tick: u64,
    agent_uid: u64,
) -> Result<(), StorageError> {
    if origin == BirthOrigin::Seeded && tick != 0 {
        return Err(StorageError::InvalidData {
            context: "births.origin",
            reason: format!(
                "seeded founder uid {agent_uid} must arrive at tick zero, found tick {tick}"
            ),
        });
    }
    Ok(())
}

fn validate_lifecycle_record_tick(
    context: &'static str,
    record_tick: u64,
    enclosing_tick: u64,
) -> Result<(), StorageError> {
    if record_tick > enclosing_tick {
        return Err(StorageError::InvalidData {
            context,
            reason: format!(
                "lifecycle record tick {record_tick} exceeds enclosing batch tick {enclosing_tick}"
            ),
        });
    }
    Ok(())
}

fn checked_u64(context: &'static str, value: i64) -> Result<u64, StorageError> {
    u64::try_from(value).map_err(|error| StorageError::InvalidData {
        context,
        reason: error.to_string(),
    })
}

fn checked_u32(context: &'static str, value: i64) -> Result<u32, StorageError> {
    u32::try_from(value).map_err(|error| StorageError::InvalidData {
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

struct FinishedRunReaderLease {
    _path: StoragePathLease,
    _writer: StorageWriterLease,
    _identity: ExistingStorageLease,
}

/// Read-only view over an existing ScriptBots database.
pub struct StorageReader {
    conn: Option<Connection>,
    _finished_run_lease: Option<FinishedRunReaderLease>,
}

impl StorageReader {
    /// Open an existing ScriptBots database read-only without creating or migrating it.
    pub fn open(path: &str) -> Result<Self, StorageError> {
        validate_durable_storage_path(path)?;
        let conn = open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        Ok(Self {
            conn: Some(conn),
            _finished_run_lease: None,
        })
    }

    /// Open a finished ScriptBots run under an exclusive, identity-bound read lease.
    ///
    /// Unlike [`Self::open`], this rejects a live writer before validating the exact
    /// migration set, structural schema, and persistence invariants. The writer and
    /// identity leases remain held for the reader's lifetime, so validation and all
    /// later report queries observe an immutable finished-run database.
    pub fn open_finished(path: &str) -> Result<Self, StorageError> {
        validate_durable_storage_path(path)?;
        let path_lease = StoragePathLease::acquire(path)?.ok_or(StorageError::InvalidData {
            context: "storage.finished_reader_path",
            reason: "a finished-run reader requires file-backed storage".to_owned(),
        })?;
        let writer_lease = StorageWriterLease::acquire_existing(path)?;
        let existing_lease = ExistingStorageLease::open(path)?;
        let conn = open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let validation = existing_lease
            .bind_connection(&conn, path)
            .and_then(|()| Storage::validate_existing_scriptbots_database(&conn))
            .and_then(|()| Storage::validate_persistence_invariants_for_connection(&conn, true))
            .and_then(|()| existing_lease.verify_path(path));
        if let Err(error) = validation {
            if let Err(close_error) = conn.close_without_checkpoint() {
                warn!(
                    path,
                    %close_error,
                    "failed to close refused read-only storage connection"
                );
            }
            return Err(error);
        }
        Ok(Self {
            conn: Some(conn),
            _finished_run_lease: Some(FinishedRunReaderLease {
                _path: path_lease,
                _writer: writer_lease,
                _identity: existing_lease,
            }),
        })
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
            "SELECT tick, seq, agent_uid, scope, event_type, payload
             FROM replay_events
             ORDER BY tick ASC, seq ASC",
        )?;
        let mut events = Vec::with_capacity(rows.len());
        for row in rows {
            let replay_row = ReplayEventRow {
                tick: decode(&row, 0, "replay_events.tick")?,
                seq: decode(&row, 1, "replay_events.seq")?,
                agent_uid: decode(&row, 2, "replay_events.agent_uid")?,
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

    /// Load every agent-arrival ancestry edge recorded in this run for offline rebuild.
    ///
    /// THE PHYSICAL `births` ROW IS THE EDGE FOR EVERY ORIGIN. Every field the
    /// graph needs — the arriving agent's uid, exact insertion and demographic
    /// ordinals, both parents' uids, the arrival tick, the generation, the brain
    /// key, the hybrid flag, and the origin — is already present, so there is no
    /// second table to keep in sync and no join that could silently drop a parent.
    /// Despite this method's historical table-derived name, the returned rows
    /// include born, seeded, and injected arrivals.
    ///
    /// Ordered by `(tick, agent_uid)`, and that order is LOAD-BEARING rather than
    /// tidy: a rebuilt graph is checked against the live one by digest, and the
    /// digest only means something if the replay order is fixed. Monotonic UIDs
    /// also keep an existing parent before a same-tick injected crossover arrival.
    ///
    /// # Errors
    ///
    /// [`StorageError`] if the connection is unavailable, a row does not decode,
    /// a seeded founder is recorded after tick zero, or an origin carries an
    /// inconsistent demographic-birth ordinal.
    pub fn load_ancestry_births(&self) -> Result<Vec<PersistedAncestryBirth>, StorageError> {
        let rows = self.connection()?.query(
            "SELECT tick, agent_uid, spawn_ordinal, birth_ordinal,
                    parent_a, parent_b, generation, brain_key, is_hybrid, origin
             FROM births
             ORDER BY tick ASC, agent_uid ASC",
        )?;
        let mut births = Vec::with_capacity(rows.len());
        for row in rows {
            let tick = checked_u64("births.tick", decode::<i64>(&row, 0, "births.tick")?)?;
            let agent_uid = checked_u64(
                "births.agent_uid",
                decode::<i64>(&row, 1, "births.agent_uid")?,
            )?;
            let spawn_ordinal: i64 = decode(&row, 2, "births.spawn_ordinal")?;
            let birth_ordinal: Option<i64> = decode(&row, 3, "births.birth_ordinal")?;
            let parent_a: Option<i64> = decode(&row, 4, "births.parent_a")?;
            let parent_b: Option<i64> = decode(&row, 5, "births.parent_b")?;
            let generation: i64 = decode(&row, 6, "births.generation")?;
            let brain_key: Option<i64> = decode(&row, 7, "births.brain_key")?;
            let is_hybrid: i64 = decode(&row, 8, "births.is_hybrid")?;
            let origin: String = decode(&row, 9, "births.origin")?;
            let birth_ordinal = birth_ordinal
                .map(|raw| checked_u64("births.birth_ordinal", raw))
                .transpose()?;
            let origin = decode_birth_origin(&origin)?;
            validate_birth_origin_ordinal(origin, birth_ordinal)?;
            validate_birth_origin_tick(origin, tick, agent_uid)?;

            births.push(PersistedAncestryBirth {
                tick: Tick(tick),
                agent_uid: AgentUid(agent_uid),
                spawn_ordinal: checked_u64("births.spawn_ordinal", spawn_ordinal)?,
                birth_ordinal,
                parent_a: parent_a
                    .map(|raw| checked_u64("births.parent_a", raw).map(AgentUid))
                    .transpose()?,
                parent_b: parent_b
                    .map(|raw| checked_u64("births.parent_b", raw).map(AgentUid))
                    .transpose()?,
                generation: Generation(checked_u32("births.generation", generation)?),
                brain_key: brain_key
                    .map(|raw| checked_u64("births.brain_key", raw))
                    .transpose()?,
                is_hybrid: is_hybrid != 0,
                origin,
            });
        }
        Ok(births)
    }

    /// Load every death recorded in this run, ordered to match the arrival replay.
    ///
    /// # Errors
    ///
    /// [`StorageError`] if the connection is unavailable or a row does not decode.
    pub fn load_ancestry_deaths(&self) -> Result<Vec<PersistedAncestryDeath>, StorageError> {
        let rows = self.connection()?.query(
            "SELECT tick, agent_uid, cause
             FROM deaths
             ORDER BY tick ASC, agent_uid ASC",
        )?;
        let mut deaths = Vec::with_capacity(rows.len());
        for row in rows {
            let tick: i64 = decode(&row, 0, "deaths.tick")?;
            let agent_uid: i64 = decode(&row, 1, "deaths.agent_uid")?;
            let cause: String = decode(&row, 2, "deaths.cause")?;
            deaths.push(PersistedAncestryDeath {
                tick: Tick(checked_u64("deaths.tick", tick)?),
                agent_uid: AgentUid(checked_u64("deaths.agent_uid", agent_uid)?),
                cause: decode_death_cause(&cause)?,
            });
        }
        Ok(deaths)
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
            "SELECT agent_uid,
                    AVG(energy) AS avg_energy,
                    MAX(spike_length) AS max_spike_length,
                    MAX(tick) AS last_tick
             FROM agents
             GROUP BY agent_uid
             ORDER BY avg_energy DESC
             LIMIT ?1",
            &[bound.into()],
        )?;
        let mut stats = Vec::with_capacity(limit.min(16));
        for row in rows {
            stats.push(PredatorStats {
                agent_uid: checked_u64("agents.agent_uid", decode(&row, 0, "agents.agent_uid")?)?,
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
            let birth_count_row =
                tx.query_row("SELECT COUNT(*) FROM births WHERE origin = 'born'")?;
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

        Self::acquire_lock(path, true).map(Some)
    }

    fn acquire_existing(path: &str) -> Result<Self, StorageError> {
        if path == ":memory:" {
            return Err(StorageError::InvalidData {
                context: "storage.finished_reader_path",
                reason: "a finished-run reader requires file-backed storage".to_owned(),
            });
        }

        Self::acquire_lock(path, false)
    }

    fn acquire_lock(path: &str, create: bool) -> Result<Self, StorageError> {
        let lock_path = storage_writer_lock_path(path);
        let lock_path_display = lock_path.display().to_string();
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(create);
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

        Ok(Self { file, lock_path })
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
    /// Whether this filesystem can actually supply the stable (device, inode) identity the
    /// swapped-file check depends on. Decided ONCE, by probing the filesystem — see
    /// [`filesystem_has_stable_file_identity`].
    identity_is_enforceable: bool,
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
        let identity_is_enforceable = filesystem_has_stable_file_identity(Path::new(path));
        if !identity_is_enforceable {
            // Say it once, out loud, with the reason and the consequence. A security check that
            // quietly turns itself off is worse than one that fails noisily: the operator would go
            // on believing they had a guarantee they no longer have.
            warn!(
                path = %path,
                "this filesystem does not give files a stable identity across truncate-and-regrow \
                 (exFAT and FAT32 synthesize an inode from the file's starting cluster, so it \
                 MOVES when the file is rewritten). The swapped-file check is therefore SKIPPED \
                 for this database: storage cannot detect another process replacing it underneath \
                 us. The symlink, regular-file and hard-link checks still apply. Put the database \
                 on APFS/ext4 to get the full guarantee."
            );
        }

        let lease = Self {
            _file: file,
            path_identity: StorageFileIdentity::from_metadata(&file_metadata),
            connection_identity,
            identity_is_enforceable,
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

        // Each condition is reported SEPARATELY. These four failures have nothing to do with one
        // another — a symlink where a file should be is an attack, while a changed inode may be
        // nothing but the filesystem reshuffling clusters — and collapsing them into one message
        // ("changed during validated open") sent readers hunting for corruption that was not
        // there. It cost real time; see bd-15c8.
        if metadata.file_type().is_symlink() {
            return Err(StorageError::InvalidData {
                context: "storage.recovery_path",
                reason: format!(
                    "recovery path {path} is a SYMLINK. The database must be a regular file: a \
                     symlink can be repointed at another file between the check and the open."
                ),
            });
        }
        if !metadata.is_file() {
            return Err(StorageError::InvalidData {
                context: "storage.recovery_path",
                reason: format!("recovery path {path} is not a regular file"),
            });
        }
        if storage_file_has_multiple_links(&metadata) {
            return Err(StorageError::InvalidData {
                context: "storage.recovery_path",
                reason: format!(
                    "recovery path {path} has MULTIPLE HARD LINKS. Another name for this file \
                     exists, so writes made through it would bypass this lease entirely."
                ),
            });
        }

        // THE INODE CHECK IS ONLY MEANINGFUL WHERE THE FILESYSTEM SUPPLIES A STABLE INODE.
        //
        // `StorageFileIdentity` is (device, inode) on Unix. On APFS and ext4 that is a genuine
        // file identity and comparing it detects a swapped file. On exFAT it is NOT: exFAT has no
        // inodes, and the kernel synthesizes one from the file's STARTING CLUSTER — so truncating
        // and regrowing a file, which is exactly what creating and initialising a database does,
        // MOVES THE INODE while the file stays the same file. Measured, not assumed: a 64-byte
        // file rewritten to 1 MB on this exFAT volume went from inode 43648807 to 43648811, while
        // the identical operation on APFS did not move at all.
        //
        // The result was that storage refused to open ANY database on an exFAT volume, accusing
        // the user's own file of having "changed during validated open" — reading our own
        // initialisation as tampering. exFAT is what external drives and USB sticks are formatted
        // as, so pointing SCRIPTBOTS_STORAGE_PATH at an external disk (an entirely reasonable
        // thing to do with multi-gigabyte run databases) was simply broken.
        //
        // So the guard now asks whether the filesystem can ACTUALLY PROVIDE the property it
        // depends on, rather than assuming it. Where it can, the check is enforced exactly as
        // before. Where it cannot, the check is SKIPPED — and skipped LOUDLY: the weakening is
        // reported, once, with the reason. The other three checks above still apply, and they are
        // the ones that catch the attacks that matter. A silently weakened security check would be
        // worse than either the false positive or the honest downgrade.
        if self.identity_is_enforceable
            && StorageFileIdentity::from_metadata(&metadata) != self.path_identity
        {
            return Err(StorageError::InvalidData {
                context: "storage.recovery_path",
                reason: format!(
                    "recovery path {path} now refers to a DIFFERENT FILE than the one this lease \
                     opened (its device/inode changed). Another process replaced the database \
                     underneath us."
                ),
            });
        }
        Ok(())
    }
}

/// Does this filesystem give a file a STABLE identity across truncate-and-regrow?
///
/// Behavioural probe, not a filesystem-name lookup. We ask the filesystem to demonstrate the
/// property we are about to depend on, because a name tells you what something is called and a
/// probe tells you what it does — and the list of filesystems that synthesize inodes (exFAT,
/// FAT32, some SMB/network mounts, some FUSE layers) is not one we can enumerate correctly in
/// advance.
///
/// The probe creates a small temporary file in the SAME DIRECTORY as the database (identity
/// semantics are a property of the mount, not of the process), notes its inode, truncates and
/// regrows it — the exact operation a database create/initialise performs — and checks whether the
/// inode survived. On failure to probe at all we return `true`: a probe that cannot run is not
/// evidence that the filesystem is broken, and defaulting to the STRONGER check keeps the guard on
/// wherever we are unsure.
fn filesystem_has_stable_file_identity(database_path: &Path) -> bool {
    let Some(dir) = database_path.parent() else {
        return true;
    };
    let probe_path = dir.join(format!(
        ".scriptbots-identity-probe-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default()
    ));

    let stable = (|| -> std::io::Result<bool> {
        std::fs::write(&probe_path, [0u8; 64])?;
        let first = StorageFileIdentity::from_metadata(&std::fs::symlink_metadata(&probe_path)?);
        // Truncate and regrow past a cluster boundary — what initialising a database does, and
        // what moves a synthesized inode.
        std::fs::write(&probe_path, vec![1u8; 1 << 20])?;
        let second = StorageFileIdentity::from_metadata(&std::fs::symlink_metadata(&probe_path)?);
        Ok(first == second)
    })()
    .unwrap_or(true);

    let _ = std::fs::remove_file(&probe_path);
    stable
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
        if migrations.len() != 3 {
            return Err(StorageError::InvalidData {
                context: "storage.recovery_schema",
                reason: format!(
                    "expected exactly three ScriptBots migrations, found {}",
                    migrations.len()
                ),
            });
        }
        let expected_migrations = [
            (3_i64, "create_stable_agent_uid_schema"),
            (4_i64, "create_stable_uid_persistence_outbox"),
            (5_i64, "record_birth_origin"),
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
        Self::validate_persistence_invariants_for_connection(self.connection()?, self.file_backed())
    }

    fn validate_persistence_invariants_for_connection(
        connection: &Connection,
        file_backed: bool,
    ) -> Result<(), StorageError> {
        let progress = connection.query_row(
            "SELECT admitted_batch_id, applied_batch_id, durable_batch_id
             FROM storage_progress
             WHERE singleton = 1",
        )?;
        let watermarks = PersistenceWatermarks::from_raw(
            decode(&progress, 0, "storage_progress.admitted_batch_id")?,
            decode(&progress, 1, "storage_progress.applied_batch_id")?,
            decode(&progress, 2, "storage_progress.durable_batch_id")?,
        )?;
        let admitted = watermarks.admitted_raw();
        let applied = watermarks.applied_raw();
        let durable = watermarks.durable_raw();
        let ledger = connection.query_row(
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
            let mismatches = connection.query_with_params(
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

        if file_backed {
            let outbox = connection.query_row(
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

    fn validate_new_birth_identities(&self, prepared: &StorageBuffer) -> Result<(), StorageError> {
        if prepared.births.is_empty() {
            return Ok(());
        }

        let buffered_agent_uids = self
            .buffer
            .births
            .iter()
            .map(|row| row.agent_uid)
            .collect::<BTreeSet<_>>();
        let buffered_spawn_ordinals = self
            .buffer
            .births
            .iter()
            .map(|row| row.spawn_ordinal)
            .collect::<BTreeSet<_>>();
        let buffered_birth_ordinals = self
            .buffer
            .births
            .iter()
            .filter_map(|row| row.birth_ordinal)
            .collect::<BTreeSet<_>>();
        for row in &prepared.births {
            if buffered_agent_uids.contains(&row.agent_uid) {
                return Err(StorageError::InvalidData {
                    context: "births.agent_uid",
                    reason: format!("agent uid {} already has a staged arrival", row.agent_uid),
                });
            }
            if buffered_spawn_ordinals.contains(&row.spawn_ordinal) {
                return Err(StorageError::InvalidData {
                    context: "births.spawn_ordinal",
                    reason: format!(
                        "spawn ordinal {} already has a staged arrival",
                        row.spawn_ordinal
                    ),
                });
            }
            if let Some(ordinal) = row.birth_ordinal
                && buffered_birth_ordinals.contains(&ordinal)
            {
                return Err(StorageError::InvalidData {
                    context: "births.birth_ordinal",
                    reason: format!("birth ordinal {ordinal} already has a staged birth"),
                });
            }

            let existing = self.connection()?.query_with_params(
                "SELECT agent_uid, spawn_ordinal, birth_ordinal
                 FROM births
                 WHERE agent_uid = ?1 OR spawn_ordinal = ?2 OR birth_ordinal = ?3
                 LIMIT 1",
                &[
                    row.agent_uid.into(),
                    row.spawn_ordinal.into(),
                    sqlite_optional_i64(row.birth_ordinal),
                ],
            )?;
            if let Some(existing) = existing.first() {
                let existing_agent_uid =
                    checked_u64("births.agent_uid", decode(existing, 0, "births.agent_uid")?)?;
                let existing_spawn_ordinal = checked_u64(
                    "births.spawn_ordinal",
                    decode(existing, 1, "births.spawn_ordinal")?,
                )?;
                let existing_birth_ordinal =
                    decode::<Option<i64>>(existing, 2, "births.birth_ordinal")?
                        .map(|raw| checked_u64("births.birth_ordinal", raw))
                        .transpose()?;
                let new_agent_uid = checked_u64("births.agent_uid", row.agent_uid)?;
                let new_spawn_ordinal = checked_u64("births.spawn_ordinal", row.spawn_ordinal)?;
                let new_birth_ordinal = row
                    .birth_ordinal
                    .map(|raw| checked_u64("births.birth_ordinal", raw))
                    .transpose()?;
                let (context, detail) = if existing_agent_uid == new_agent_uid {
                    (
                        "births.agent_uid",
                        format!("agent uid {new_agent_uid} already has a persisted arrival"),
                    )
                } else if existing_spawn_ordinal == new_spawn_ordinal {
                    (
                        "births.spawn_ordinal",
                        format!(
                            "spawn ordinal {new_spawn_ordinal} already has a persisted arrival"
                        ),
                    )
                } else if let Some(ordinal) = new_birth_ordinal {
                    debug_assert_eq!(existing_birth_ordinal, new_birth_ordinal);
                    (
                        "births.birth_ordinal",
                        format!("birth ordinal {ordinal} already has a persisted birth"),
                    )
                } else {
                    return Err(StorageError::InvalidData {
                        context: "births.birth_ordinal",
                        reason: "a NULL birth ordinal matched an indexed persisted value"
                            .to_owned(),
                    });
                };
                return Err(StorageError::InvalidData {
                    context,
                    reason: detail,
                });
            }
        }
        Ok(())
    }

    fn validate_new_death_uids(&self, prepared: &StorageBuffer) -> Result<(), StorageError> {
        if prepared.deaths.is_empty() {
            return Ok(());
        }

        let buffered = self
            .buffer
            .deaths
            .iter()
            .map(|row| row.agent_uid)
            .collect::<BTreeSet<_>>();
        for row in &prepared.deaths {
            if buffered.contains(&row.agent_uid) {
                return Err(StorageError::InvalidData {
                    context: "deaths.agent_uid",
                    reason: format!("agent uid {} already has a staged death", row.agent_uid),
                });
            }
            let existing = self.connection()?.query_with_params(
                "SELECT agent_uid, tick FROM deaths WHERE agent_uid = ?1 LIMIT 1",
                &[row.agent_uid.into()],
            )?;
            if let Some(existing) = existing.first() {
                let existing_uid =
                    checked_u64("deaths.agent_uid", decode(existing, 0, "deaths.agent_uid")?)?;
                let new_uid = checked_u64("deaths.agent_uid", row.agent_uid)?;
                if existing_uid != new_uid {
                    return Err(StorageError::InvalidData {
                        context: "deaths.agent_uid",
                        reason: format!(
                            "death lookup for uid {new_uid} returned uid {existing_uid}"
                        ),
                    });
                }
                let existing_tick =
                    checked_u64("deaths.tick", decode(existing, 1, "deaths.tick")?)?;
                return Err(StorageError::InvalidData {
                    context: "deaths.agent_uid",
                    reason: format!(
                        "agent uid {new_uid} already has a persisted death at tick {existing_tick}"
                    ),
                });
            }
        }
        Ok(())
    }

    fn validate_new_ancestry_relationships(
        &self,
        prepared: &StorageBuffer,
    ) -> Result<(), StorageError> {
        if prepared.births.is_empty() && prepared.deaths.is_empty() {
            return Ok(());
        }

        // Unapplied admitted batches live in `buffer`; applied batches live in
        // `births`; and the current batch can contain a parent arrival followed by
        // a later child or death. Build one read-only view across all three
        // locations before the new outbox identity is assigned.
        let mut known_birth_ticks = BTreeMap::new();
        for row in self.buffer.births.iter().chain(&prepared.births) {
            known_birth_ticks.insert(
                checked_u64("births.agent_uid", row.agent_uid)?,
                checked_u64("births.tick", row.tick)?,
            );
        }
        let mut referenced_uids = BTreeSet::new();

        for row in &prepared.births {
            let parent_a = row
                .parent_a
                .map(|raw| checked_u64("births.parent_a", raw))
                .transpose()?;
            let parent_b = row
                .parent_b
                .map(|raw| checked_u64("births.parent_b", raw))
                .transpose()?;
            if let (Some(parent_a), Some(parent_b)) = (parent_a, parent_b)
                && parent_a == parent_b
            {
                return Err(StorageError::InvalidData {
                    context: "births.parent_b",
                    reason: format!(
                        "arrival uid {} names parent uid {parent_a} in both parent slots",
                        row.agent_uid
                    ),
                });
            }
            referenced_uids.extend([parent_a, parent_b].into_iter().flatten());
        }
        for row in &prepared.deaths {
            referenced_uids.insert(checked_u64("deaths.agent_uid", row.agent_uid)?);
        }

        for agent_uid in referenced_uids {
            let std::collections::btree_map::Entry::Vacant(entry) =
                known_birth_ticks.entry(agent_uid)
            else {
                continue;
            };
            let existing = self.connection()?.query_with_params(
                "SELECT agent_uid, tick FROM births WHERE agent_uid = ?1 LIMIT 1",
                &[encode_u64("births.agent_uid", agent_uid)?.into()],
            )?;
            if let Some(row) = existing.first() {
                let persisted_uid =
                    checked_u64("births.agent_uid", decode(row, 0, "births.agent_uid")?)?;
                if persisted_uid != agent_uid {
                    return Err(StorageError::InvalidData {
                        context: "births.agent_uid",
                        reason: format!(
                            "arrival lookup for uid {agent_uid} returned uid {persisted_uid}"
                        ),
                    });
                }
                entry.insert(checked_u64("births.tick", decode(row, 1, "births.tick")?)?);
            }
        }

        for row in &prepared.births {
            let child_uid = checked_u64("births.agent_uid", row.agent_uid)?;
            let child_tick = checked_u64("births.tick", row.tick)?;
            for (context, parent_uid) in [
                ("births.parent_a", row.parent_a),
                ("births.parent_b", row.parent_b),
            ] {
                let Some(parent_uid) = parent_uid else {
                    continue;
                };
                let parent_uid = checked_u64(context, parent_uid)?;
                let Some(parent_tick) = known_birth_ticks.get(&parent_uid).copied() else {
                    return Err(StorageError::InvalidData {
                        context,
                        reason: format!(
                            "arrival uid {child_uid} names parent uid {parent_uid}, whose arrival was not recorded"
                        ),
                    });
                };
                if child_tick <= parent_tick {
                    return Err(StorageError::InvalidData {
                        context,
                        reason: format!(
                            "arrival uid {child_uid} at tick {child_tick} does not follow parent uid {parent_uid} at tick {parent_tick}"
                        ),
                    });
                }
            }
        }

        for row in &prepared.deaths {
            let agent_uid = checked_u64("deaths.agent_uid", row.agent_uid)?;
            let death_tick = checked_u64("deaths.tick", row.tick)?;
            let Some(birth_tick) = known_birth_ticks.get(&agent_uid).copied() else {
                return Err(StorageError::InvalidData {
                    context: "deaths.agent_uid",
                    reason: format!("death uid {agent_uid} has no recorded arrival"),
                });
            };
            if death_tick <= birth_tick {
                return Err(StorageError::InvalidData {
                    context: "deaths.tick",
                    reason: format!(
                        "death uid {agent_uid} at tick {death_tick} does not follow its arrival at tick {birth_tick}"
                    ),
                });
            }
        }

        Ok(())
    }

    fn validate_new_lifecycle_identities(
        &self,
        prepared: &StorageBuffer,
    ) -> Result<(), StorageError> {
        self.validate_new_birth_identities(prepared)?;
        self.validate_new_death_uids(prepared)?;
        self.validate_new_ancestry_relationships(prepared)
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

        self.validate_new_lifecycle_identities(prepared)?;
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
            self.validate_new_lifecycle_identities(&batch.storage)?;
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

        prepared.validate_contents(summary.tick.0)?;
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
                scriptbots_agent_insert_sql(),
                &[
                    row.tick.into(),
                    row.agent_uid.into(),
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

    fn insert_births(tx: &Transaction<'_>, rows: &[BirthRow]) -> Result<(), FrankenError> {
        if rows.is_empty() {
            return Ok(());
        }
        let sql = "insert into births (
                tick, agent_uid, spawn_ordinal, birth_ordinal, parent_a, parent_b,
                brain_kind, brain_key, herbivore_tendency,
                generation, position_x, position_y, is_hybrid, origin
            ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)";
        for row in rows {
            tx.execute_with_params(
                sql,
                &[
                    row.tick.into(),
                    row.agent_uid.into(),
                    row.spawn_ordinal.into(),
                    sqlite_optional_i64(row.birth_ordinal),
                    sqlite_optional_i64(row.parent_a),
                    sqlite_optional_i64(row.parent_b),
                    sqlite_optional_text(row.brain_kind.as_deref()),
                    sqlite_optional_i64(row.brain_key),
                    row.herbivore_tendency.into(),
                    row.generation.into(),
                    row.position_x.into(),
                    row.position_y.into(),
                    sqlite_bool(row.is_hybrid),
                    row.origin.as_str().into(),
                ],
            )?;
        }
        Ok(())
    }

    fn insert_deaths(tx: &Transaction<'_>, rows: &[DeathRow]) -> Result<(), FrankenError> {
        if rows.is_empty() {
            return Ok(());
        }
        let sql = "insert into deaths (
                tick, agent_uid, age, generation,
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
                    row.agent_uid.into(),
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
                tick, seq, agent_uid, scope, event_type, payload
            ) values (?1, ?2, ?3, ?4, ?5, ?6)";
        for row in rows {
            tx.execute_with_params(
                sql,
                &[
                    row.tick.into(),
                    row.seq.into(),
                    sqlite_optional_i64(row.agent_uid),
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
            "SELECT tick, seq, agent_uid, scope, event_type, payload
             from replay_events
             ORDER BY tick ASC, seq ASC",
        )?;
        let mut events = Vec::with_capacity(rows.len());
        for row in rows {
            let replay_row = ReplayEventRow {
                tick: decode(&row, 0, "replay_events.tick")?,
                seq: decode(&row, 1, "replay_events.seq")?,
                agent_uid: decode(&row, 2, "replay_events.agent_uid")?,
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
            "SELECT agent_uid,
                    AVG(energy) AS avg_energy,
                    MAX(spike_length) AS max_spike_length,
                    MAX(tick) AS last_tick
             FROM agents
             GROUP BY agent_uid
             ORDER BY avg_energy DESC
             LIMIT ?1",
            &[bound.into()],
        )?;
        let mut stats = Vec::with_capacity(limit.min(16));
        for row in rows {
            let agent_uid = decode::<i64>(&row, 0, "agents.agent_uid")?;
            stats.push(PredatorStats {
                agent_uid: checked_u64("agents.agent_uid", agent_uid)?,
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
    pub agent_uid: u64,
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
    /// What this sink will admit. See [`PayloadBudget`].
    budget: PayloadBudget,
    /// Bytes admitted but not yet flushed.
    ///
    /// Incremented on admission and decremented when the worker reports the batch
    /// committed or failed — EXACTLY ONCE on every exit path. A permit that leaks
    /// on the error path is worse than no permit at all: the counter creeps up,
    /// the sink eventually refuses everything, and persistence dies quietly in a
    /// long run rather than loudly in a test.
    inflight_bytes: Arc<AtomicUsize>,
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

        // MEASURE BEFORE ALLOCATING. `from_batch` below materializes the entire
        // batch; if the batch is pathological, the memory is already gone by the
        // time any deadline or admission gate gets a say. So the size is computed
        // from the batch's own shape first — constant time, no serialization —
        // and an oversized batch is refused here, having allocated nothing.
        //
        // This is a NotAdmitted outcome in the strictest sense: the caller still
        // holds the exact payload it tried to submit, so the exact-retry semantics
        // the storage contract depends on are untouched.
        let (bytes, events) = estimate_batch_size(payload);
        if bytes > self.budget.max_batch_bytes || events > self.budget.max_batch_events {
            return Err(StorageError::PayloadTooLarge {
                tick,
                bytes,
                events,
                max_bytes: self.budget.max_batch_bytes,
                max_events: self.budget.max_batch_events,
            });
        }

        // Total in-flight back-pressure. A stream of individually-legal batches
        // can still exhaust memory if the writer falls behind, so the buffered
        // total is bounded too. Reserved BEFORE preparation, and released on every
        // exit below.
        let previous = self.inflight_bytes.fetch_add(bytes, Ordering::SeqCst);
        let would_be = previous.saturating_add(bytes);
        if would_be > self.budget.max_inflight_bytes {
            // Release immediately: this batch never became in-flight.
            self.inflight_bytes.fetch_sub(bytes, Ordering::SeqCst);
            return Err(StorageError::InFlightBytesExhausted {
                tick,
                would_be,
                max_inflight: self.budget.max_inflight_bytes,
            });
        }
        // From here on, `bytes` is reserved and MUST be released exactly once on
        // every path out of this function. `InFlightPermit` does that on drop, so
        // an early return, an error, or a panic cannot leak it.
        let _permit = InFlightPermit {
            counter: Arc::clone(&self.inflight_bytes),
            bytes,
        };

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

/// Most reaper threads that may exist at once, across every storage path.
///
/// The old handoff spawned ONE INDEPENDENT OS THREAD PER TIMEOUT, with no bound.
/// A slow or wedged disk does not time out once; it times out over and over, and
/// each timeout spawned another thread that then blocked on the same sick disk.
/// The failure mode is a thread-count explosion caused BY the thing that was
/// already failing — the process runs out of threads while trying to clean up
/// after a disk that stopped answering.
const MAX_CONCURRENT_REAPERS: usize = 4;

/// The supervisor registry: who is being reaped, and who is waiting.
///
/// Keyed by storage PATH, which is what makes coalescing correct. Two timeouts on
/// the same path do not need two reapers — the second is queued behind the first
/// and drained by it. Two timeouts on DIFFERENT paths must not block each other,
/// which is why the key is the path and not a single global lock on "reaping".
#[derive(Default)]
struct ReaperRegistry {
    /// Paths with a live reaper thread.
    active: BTreeSet<String>,
    /// Requests waiting behind an active reaper, per path.
    ///
    /// A queued request is NEVER dropped: the reaper that owns the path drains
    /// this before it retires. Dropping one would leak a `JoinHandle` and lose a
    /// receipt, which is the one thing the storage contract cannot tolerate.
    queued: BTreeMap<String, Vec<StorageReapRequest>>,
    /// Reaper threads started.
    started: u64,
    /// Handoffs folded into an existing reaper rather than spawning a new thread.
    coalesced: u64,
    /// Handoffs run on the CALLER's thread because the registry was saturated or
    /// the spawn failed.
    synchronous: u64,
}

/// A snapshot of the reaper registry, for operators and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReaperStats {
    /// Paths currently being reaped.
    pub active: usize,
    /// Requests waiting behind an active reaper.
    pub queued: usize,
    /// Reaper threads started since process start.
    pub started: u64,
    /// Handoffs coalesced onto an existing reaper.
    pub coalesced: u64,
    /// Handoffs that fell back to running synchronously.
    pub synchronous: u64,
}

fn reaper_registry() -> &'static Mutex<ReaperRegistry> {
    static REGISTRY: OnceLock<Mutex<ReaperRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(ReaperRegistry::default()))
}

fn lock_reaper_registry() -> std::sync::MutexGuard<'static, ReaperRegistry> {
    // A poisoned registry is recoverable: the contents are counters and queues, not
    // invariants a panic could have half-broken. Refusing to reap because a previous
    // reaper panicked would leak every subsequent worker.
    match reaper_registry().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Current state of the supervised-reap registry.
#[must_use]
pub fn storage_reaper_stats() -> ReaperStats {
    let registry = lock_reaper_registry();
    ReaperStats {
        active: registry.active.len(),
        queued: registry.queued.values().map(Vec::len).sum(),
        started: registry.started,
        coalesced: registry.coalesced,
        synchronous: registry.synchronous,
    }
}

fn request_path(request: &StorageReapRequest) -> String {
    match request {
        StorageReapRequest::Pipeline { path, .. } | StorageReapRequest::JoinOnly { path, .. } => {
            path.to_string()
        }
    }
}

fn handoff_storage_reap(request: StorageReapRequest) {
    let path = request_path(&request);

    // Decide under the lock, then act outside it: reaping JOINS a worker thread,
    // and joining while holding the registry lock would let one sick path stall
    // every other path's handoff — the exact cross-path blocking this bead forbids.
    let action = {
        let mut registry = lock_reaper_registry();
        if registry.active.contains(&path) {
            // COALESCE. A path already has a reaper; a second thread would just
            // queue behind the same worker. Hand the request to the running reaper,
            // which drains this before it retires.
            registry.coalesced += 1;
            registry
                .queued
                .entry(path.clone())
                .or_default()
                .push(request);
            return;
        }
        if registry.active.len() >= MAX_CONCURRENT_REAPERS {
            // SATURATED. Run on the caller's thread rather than spawning past the
            // bound. This is the same fallback the old code used when `spawn`
            // failed, and it is what makes the bound a real bound: the work still
            // happens, the handle is still joined, and no receipt is dropped — the
            // caller simply pays for it.
            registry.synchronous += 1;
            ReaperAction::Synchronous(request)
        } else {
            registry.active.insert(path.clone());
            registry.started += 1;
            ReaperAction::Spawn(request)
        }
    };

    let request = match action {
        ReaperAction::Synchronous(request) => {
            reap_storage_request(request);
            return;
        }
        ReaperAction::Spawn(request) => request,
    };

    let thread_path = path.clone();
    // The request travels in a shared slot rather than being moved into the
    // closure. `thread::Builder::spawn` DROPS the closure when it fails, and
    // dropping this request would drop the worker's JoinHandle with it — leaking
    // the very thread we are trying to reap. The slot lets the caller take it back.
    let slot = Arc::new(Mutex::new(Some(request)));
    let thread_slot = Arc::clone(&slot);
    let spawned = thread::Builder::new()
        .name("scriptbots-storage-reaper".into())
        .spawn(move || {
            let mut current = match thread_slot.lock() {
                Ok(mut slot) => slot.take(),
                Err(poisoned) => poisoned.into_inner().take(),
            };
            while let Some(request) = current.take() {
                reap_storage_request(request);
                // Drain anything that arrived for this path while we were joining.
                // Retiring with a queued request still in the registry would leak
                // its JoinHandle forever.
                let mut registry = lock_reaper_registry();
                current = registry
                    .queued
                    .get_mut(&thread_path)
                    .and_then(std::vec::Vec::pop);
                if current.is_none() {
                    registry.queued.remove(&thread_path);
                    registry.active.remove(&thread_path);
                }
            }
        });

    if let Err(error) = spawned {
        // The spawn failed, so nobody owns this path. Release it and do the work
        // here — losing the request because the OS would not give us a thread would
        // leak the very worker we were trying to clean up after.
        let mut registry = lock_reaper_registry();
        registry.active.remove(&path);
        registry.synchronous += 1;
        let queued = registry.queued.remove(&path).unwrap_or_default();
        drop(registry);
        tracing::warn!(
            path = %path,
            error = %error,
            "could not spawn a storage reaper; reaping synchronously on the caller"
        );
        let request = match slot.lock() {
            Ok(mut slot) => slot.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if let Some(request) = request {
            reap_storage_request(request);
        }
        for request in queued {
            reap_storage_request(request);
        }
    }
}

enum ReaperAction {
    Spawn(StorageReapRequest),
    Synchronous(StorageReapRequest),
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
                    budget: PayloadBudget::default(),
                    inflight_bytes: Arc::new(AtomicUsize::new(0)),
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

    /// Return a clonable sink for an external persistence session while retaining host control
    /// of the worker.
    #[must_use]
    pub fn sink(&self) -> StorageSink {
        self.sink.clone()
    }

    /// Admit a persistence batch to the bounded worker queue.
    /// Set the ceilings this pipeline will admit.
    ///
    /// Exposed so operators (and tests) can bound persistence to the machine they
    /// are actually on; the default is deliberately generous but finite.
    pub fn set_payload_budget(&mut self, budget: PayloadBudget) {
        self.sink.budget = budget;
    }

    /// Bytes admitted but not yet flushed.
    #[must_use]
    pub fn inflight_bytes(&self) -> usize {
        self.sink.inflight_bytes.load(Ordering::SeqCst)
    }

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
    let uid = encode_u64("agents.agent_uid", agent.identity.uid.get())?;
    let data = &agent.data;
    let runtime = &agent.runtime;
    Ok(AgentRow {
        tick,
        agent_uid: uid,
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

fn optional_agent_uid(
    context: &'static str,
    uid: Option<AgentUid>,
) -> Result<Option<i64>, StorageError> {
    uid.map(|agent_uid| encode_u64(context, agent_uid.get()))
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
                if event.agent_uid.is_some() {
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
                .map(|agent_uid| encode_u64("replay_events.action.spike_target", agent_uid.get()))
                .transpose()?;
            (
                if event.agent_uid.is_some() {
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
            let scope_agent_uid = match scope {
                ReplayRngScope::World => None,
                ReplayRngScope::Agent { agent_uid, .. } => Some(encode_u64(
                    "replay_events.rng_sample.scope_agent_uid",
                    agent_uid.get(),
                )?),
            };
            (
                scope_label(*scope),
                "rng_sample".to_string(),
                json!({
                    "scope_agent_uid": scope_agent_uid,
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
        agent_uid: optional_agent_uid("replay_events.agent_uid", event.agent_uid)?,
        scope,
        event_type,
        payload: payload_value.to_string(),
    })
}

fn decode_agent_uid(
    raw: Option<i64>,
    tick: i64,
    seq: i64,
) -> Result<Option<AgentUid>, StorageError> {
    match raw {
        Some(value) if value < 0 => Err(StorageError::ReplayParse {
            tick,
            seq,
            reason: format!("negative agent uid {value}"),
        }),
        Some(value) => Ok(Some(AgentUid(value as u64))),
        None => Ok(None),
    }
}

fn agent_uid_from_u64(value: u64, tick: i64, seq: i64) -> Result<AgentUid, StorageError> {
    if value > i64::MAX as u64 {
        return Err(StorageError::ReplayParse {
            tick,
            seq,
            reason: format!("agent uid {value} exceeds supported range"),
        });
    }
    Ok(AgentUid(value))
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
    agent_uid: Option<AgentUid>,
    row: &ReplayEventRow,
) -> Result<ReplayRngScope, StorageError> {
    if scope == "world" {
        return Ok(ReplayRngScope::World);
    }

    if let Some(phase_label) = scope.strip_prefix("agent:") {
        let agent_uid = agent_uid.ok_or_else(|| StorageError::ReplayParse {
            tick: row.tick,
            seq: row.seq,
            reason: "agent-scoped RNG sample missing agent_uid".to_string(),
        })?;
        let phase = parse_agent_phase(phase_label).ok_or_else(|| StorageError::ReplayParse {
            tick: row.tick,
            seq: row.seq,
            reason: format!("unknown agent phase '{phase_label}'"),
        })?;
        return Ok(ReplayRngScope::Agent { agent_uid, phase });
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
    scope_agent_uid: Option<u64>,
    range_min: f32,
    range_max: f32,
    value: f32,
}

fn replay_event_from_row(row: &ReplayEventRow) -> Result<ReplayEvent, StorageError> {
    let agent_uid = decode_agent_uid(row.agent_uid, row.tick, row.seq)?;
    let kind = match row.event_type.as_str() {
        "brain_outputs" => {
            if row.scope.starts_with("agent:") && agent_uid.is_none() {
                return Err(StorageError::ReplayParse {
                    tick: row.tick,
                    seq: row.seq,
                    reason: "brain outputs missing agent_uid".to_string(),
                });
            }
            let payload: BrainOutputsPayload = parse_payload(row)?;
            ReplayEventKind::BrainOutputs {
                outputs: payload.outputs,
            }
        }
        "action" => {
            if row.scope.starts_with("agent:") && agent_uid.is_none() {
                return Err(StorageError::ReplayParse {
                    tick: row.tick,
                    seq: row.seq,
                    reason: "action event missing agent_uid".to_string(),
                });
            }
            let payload: ActionPayload = parse_payload(row)?;
            let spike_target = match payload.spike_target {
                Some(raw) => Some(agent_uid_from_u64(raw, row.tick, row.seq)?),
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
            let scope_agent_uid = payload
                .scope_agent_uid
                .map(|raw| agent_uid_from_u64(raw, row.tick, row.seq))
                .transpose()?;
            let scope = parse_rng_scope(&row.scope, scope_agent_uid, row)?;
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

    Ok(ReplayEvent { agent_uid, kind })
}

fn birth_row_from_record(record: &BirthRecord) -> Result<BirthRow, StorageError> {
    validate_birth_origin_ordinal(record.origin, record.birth_ordinal)?;
    let birth_ordinal = record
        .birth_ordinal
        .map(|ordinal| encode_u64("births.birth_ordinal", ordinal))
        .transpose()?;
    Ok(BirthRow {
        tick: encode_u64("births.tick", record.tick.0)?,
        agent_uid: encode_u64("births.agent_uid", record.agent_uid.get())?,
        spawn_ordinal: encode_u64("births.spawn_ordinal", record.spawn_ordinal)?,
        birth_ordinal,
        parent_a: optional_agent_uid("births.parent_a", record.parent_a)?,
        parent_b: optional_agent_uid("births.parent_b", record.parent_b)?,
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
        origin: record.origin,
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

fn decode_death_cause(value: &str) -> Result<DeathCause, StorageError> {
    match value {
        "combat_carnivore" => Ok(DeathCause::CombatCarnivore),
        "combat_herbivore" => Ok(DeathCause::CombatHerbivore),
        "starvation" => Ok(DeathCause::Starvation),
        "aging" => Ok(DeathCause::Aging),
        "unknown" => Ok(DeathCause::Unknown),
        other => Err(StorageError::InvalidData {
            context: "deaths.cause",
            reason: format!("unknown death cause {other:?}"),
        }),
    }
}

fn death_row_from_record(record: &DeathRecord) -> Result<DeathRow, StorageError> {
    Ok(DeathRow {
        tick: encode_u64("deaths.tick", record.tick.0)?,
        agent_uid: encode_u64("deaths.agent_uid", record.agent_uid.get())?,
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

    fn normalized_scientific_schema(
        connection: &Connection,
    ) -> Result<Vec<(String, String, String, String)>, StorageError> {
        const INTERNAL_TABLES: &[&str] = &[
            "_schema_migrations",
            "storage_batch_ledger",
            "storage_outbox",
            "storage_progress",
        ];

        fn normalize_schema_sql(sql: &str) -> String {
            let mut normalized = String::with_capacity(sql.len());
            let mut characters = sql.chars().peekable();
            let mut in_string_literal = false;
            while let Some(character) = characters.next() {
                if in_string_literal {
                    normalized.push(character);
                    if character == '\'' {
                        if characters.peek() == Some(&'\'') {
                            normalized.push(characters.next().expect("peeked escaped quote"));
                        } else {
                            in_string_literal = false;
                        }
                    }
                } else if character == '\'' {
                    in_string_literal = true;
                    normalized.push(character);
                } else if !character.is_ascii_whitespace() && character != '"' {
                    normalized.push(character.to_ascii_lowercase());
                }
            }
            normalized
        }

        read_schema_objects(connection).map(|objects| {
            objects
                .into_iter()
                .filter(|object| {
                    !INTERNAL_TABLES.contains(&object.table_name.as_str()) && object.sql.is_some()
                })
                .map(|object| {
                    let normalized_sql = normalize_schema_sql(
                        &object.sql.expect("filtered schema objects have SQL"),
                    );
                    (
                        object.object_type,
                        object.name,
                        object.table_name,
                        normalized_sql,
                    )
                })
                .collect()
        })
    }

    #[test]
    fn exported_schema_v1_executes_with_canonical_workload_table_names()
    -> Result<(), Box<dyn std::error::Error>> {
        let connection = Connection::open(":memory:")?;
        connection.execute_batch(SCRIPTBOTS_SCHEMA_V1)?;

        let production_connection = Connection::open(":memory:")?;
        install_scriptbots_schema(&production_connection)?;
        assert_eq!(
            normalized_scientific_schema(&connection)?,
            normalized_scientific_schema(&production_connection)?,
            "exported scientific DDL drifted from the production migration result"
        );

        let table_names = connection
            .query(
                "SELECT name FROM sqlite_schema
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                 ORDER BY name ASC",
            )?
            .into_iter()
            .map(|row| row.get_typed::<String>(0))
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            table_names,
            vec![
                "agents",
                "births",
                "deaths",
                "events",
                "metrics",
                "replay_events",
                "ticks",
            ]
        );
        assert_eq!(
            scriptbots_agent_insert_sql()
                .bytes()
                .filter(|byte| *byte == b'?')
                .count(),
            SCRIPTBOTS_AGENT_COLUMN_COUNT,
            "canonical agent insert placeholder count drifted from its column list"
        );
        assert!(
            scriptbots_agent_insert_sql().ends_with("?39)"),
            "canonical agent insert no longer binds all production columns in order"
        );

        production_connection.close()?;
        connection.close()?;
        Ok(())
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
            identity: scriptbots_core::AgentIdentity {
                uid: AgentUid(1),
                spawn_ordinal: 0,
                birth_ordinal: None,
            },
            data,
            runtime,
        }
    }

    fn sample_batch(tick: u64, energy: f32) -> PersistenceBatch {
        PersistenceBatch {
            summary: TickSummary {
                tick: Tick(tick),
                agent_count: 1,
                births: 0,
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
            events: vec![PersistenceEvent::new(
                PersistenceEventKind::Custom("sample".into()),
                1,
            )],
            agents: vec![sample_agent(energy)],
            births: Vec::new(),
            deaths: Vec::new(),
            replay_events: Vec::new(),
        }
    }

    fn synchronize_lifecycle_counts(batch: &mut PersistenceBatch) {
        let births = batch
            .births
            .iter()
            .filter(|record| record.origin == BirthOrigin::Born)
            .count();
        let deaths = batch.deaths.len();
        batch.summary.births = births;
        batch.summary.deaths = deaths;
        batch.events.retain(|event| {
            !matches!(
                &event.kind,
                PersistenceEventKind::Births | PersistenceEventKind::Deaths
            )
        });
        if births > 0 {
            batch
                .events
                .push(PersistenceEvent::new(PersistenceEventKind::Births, births));
        }
        if deaths > 0 {
            batch
                .events
                .push(PersistenceEvent::new(PersistenceEventKind::Deaths, deaths));
        }
    }

    fn sample_birth(tick: u64, uid: u64, origin: BirthOrigin) -> BirthRecord {
        BirthRecord {
            tick: Tick(tick),
            agent_uid: AgentUid(uid),
            spawn_ordinal: uid.saturating_sub(1),
            birth_ordinal: (origin == BirthOrigin::Born).then_some(uid),
            origin,
            parent_a: None,
            parent_b: None,
            brain_kind: Some("test.origin".to_owned()),
            brain_key: None,
            herbivore_tendency: 0.5,
            generation: Generation(0),
            position: Position::new(1.0, 2.0),
            is_hybrid: false,
        }
    }

    fn sample_death(tick: u64, uid: u64, cause: DeathCause) -> DeathRecord {
        DeathRecord {
            tick: Tick(tick),
            agent_uid: AgentUid(uid),
            age: 1,
            generation: Generation(0),
            herbivore_tendency: 0.5,
            brain_kind: Some("test.death".to_owned()),
            brain_key: None,
            energy: 0.0,
            food_balance_total: 0.0,
            cause,
            was_hybrid: false,
            combat_flags: scriptbots_core::CombatEventFlags::default(),
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

    #[test]
    fn storage_reader_is_identity_bound_schema_verified_and_read_only()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-reader-verified-read-only");
        let path_string = path.to_string_lossy().to_string();
        let mut storage = create_file_storage(&path_string)?;
        storage.persist(&sample_batch(1, 10.0))?;
        storage.flush()?;
        storage.close()?;
        let before = fs::read(&path)?;

        let reader = StorageReader::open_finished(&path_string)?;
        let write_error = reader
            .connection()?
            .execute("CREATE TABLE reader_must_not_write (value INTEGER NOT NULL)")
            .expect_err("the verified analytics connection accepted a schema write");
        assert!(
            matches!(write_error, FrankenError::ReadOnly),
            "unexpected read-only write error: {write_error}"
        );
        reader.close()?;
        assert_eq!(
            fs::read(&path)?,
            before,
            "read-only open or refused write changed database bytes"
        );

        let unrelated = temp_db_path("storage-reader-unrelated-refusal");
        let unrelated_string = unrelated.to_string_lossy().to_string();
        let connection = Connection::open_strict_multi_process(&unrelated_string)?;
        connection.execute("CREATE TABLE unrelated (value INTEGER NOT NULL)")?;
        connection.close()?;
        let unrelated_before = fs::read(&unrelated)?;
        let unrelated_lock = storage_writer_lock_path(&unrelated_string);
        assert!(!unrelated_lock.try_exists()?);
        assert!(
            StorageReader::open_finished(&unrelated_string).is_err(),
            "verified reader accepted an unrelated FrankenSQLite database"
        );
        assert_eq!(
            fs::read(&unrelated)?,
            unrelated_before,
            "refused reader open mutated an unrelated database"
        );
        assert!(
            !unrelated_lock.try_exists()?,
            "refused reader open created a writer-lease sidecar"
        );
        Ok(())
    }

    #[test]
    fn finished_reader_rejects_live_writer_without_breaking_live_readers()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-finished-reader-live-writer");
        let path_string = path.to_string_lossy().to_string();
        let mut pipeline = StoragePipeline::create_new_file(&path_string)?;

        let live_reader = StorageReader::open(&path_string)?;
        let finished_error = match StorageReader::open_finished(&path_string) {
            Ok(reader) => {
                reader.close()?;
                return Err("finished-run reader admitted a live writer".into());
            }
            Err(error) => error,
        };
        assert!(
            matches!(finished_error, StorageError::InvalidData { .. }),
            "unexpected live-writer refusal: {finished_error}"
        );

        live_reader.close()?;
        pipeline.shutdown()?;
        let finished_reader = StorageReader::open_finished(&path_string)?;
        finished_reader.close()?;
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
    fn birth_origin_migration_backfills_historical_rows_without_a_default()
    -> Result<(), Box<dyn std::error::Error>> {
        let connection = Connection::open(":memory:")?;
        MigrationRunner::new()
            .add(3, "create_stable_agent_uid_schema", SCRIPTBOTS_SCHEMA_V3)
            .add(
                4,
                "create_stable_uid_persistence_outbox",
                SCRIPTBOTS_SCHEMA_V4,
            )
            .run(&connection)?;
        connection.execute(
            "INSERT INTO births (
                tick, agent_uid, spawn_ordinal, birth_ordinal, parent_a, parent_b,
                brain_kind, brain_key, herbivore_tendency, generation,
                position_x, position_y, is_hybrid
             ) VALUES (1, 2, 1, 0, NULL, NULL, NULL, NULL, 0.5, 0, 1.0, 2.0, 0)",
        )?;

        install_scriptbots_schema(&connection)?;

        let origin: String = connection
            .query_row("SELECT origin FROM births WHERE tick = 1 AND agent_uid = 2")?
            .get_typed(0)?;
        assert_eq!(origin, "born");
        let historical_ordinal: Option<i64> = connection
            .query_row("SELECT birth_ordinal FROM births WHERE tick = 1 AND agent_uid = 2")?
            .get_typed(0)?;
        assert_eq!(historical_ordinal, Some(0));
        let births_schema: String = connection
            .query_row("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'births'")?
            .get_typed(0)?;
        assert!(
            !births_schema.to_ascii_uppercase().contains("DEFAULT"),
            "the final origin column must not retain a DEFAULT: {births_schema}"
        );
        assert!(
            births_schema.contains("PRIMARY KEY (tick, agent_uid)"),
            "the table rebuild did not preserve the composite primary key: {births_schema}"
        );
        assert!(
            connection
                .execute("UPDATE births SET origin = 'unknown' WHERE tick = 1")
                .is_err(),
            "the V5 origin domain must remain strict after migrating a historical row"
        );
        assert!(
            connection
                .execute(
                    "INSERT INTO births (
                        tick, agent_uid, spawn_ordinal, birth_ordinal,
                        herbivore_tendency, generation, position_x, position_y, is_hybrid
                     ) VALUES (2, 3, 2, NULL, 0.5, 0, 3.0, 4.0, 0)",
                )
                .is_err(),
            "omitting origin must fail instead of silently defaulting to born"
        );
        connection.execute(
            "INSERT INTO births (
                tick, agent_uid, spawn_ordinal, birth_ordinal,
                herbivore_tendency, generation, position_x, position_y, is_hybrid, origin
             ) VALUES (0, 1, 0, NULL, 0.5, 0, 3.0, 4.0, 0, 'seeded')",
        )?;
        let seeded_ordinal: Option<i64> = connection
            .query_row("SELECT birth_ordinal FROM births WHERE agent_uid = 1")?
            .get_typed(0)?;
        assert_eq!(seeded_ordinal, None);
        assert!(
            connection
                .execute("UPDATE births SET tick = 2 WHERE agent_uid = 1")
                .is_err(),
            "the V5 schema must keep seeded founders at tick zero"
        );
        assert!(
            connection
                .execute(
                    "INSERT INTO births (
                        tick, agent_uid, spawn_ordinal, birth_ordinal,
                        herbivore_tendency, generation, position_x, position_y, is_hybrid, origin
                     ) VALUES (3, 3, 2, 2, 0.5, 0, 3.0, 4.0, 0, 'injected')",
                )
                .is_err(),
            "non-born origins must not persist a demographic birth ordinal"
        );
        connection.close()?;
        Ok(())
    }

    #[test]
    fn birth_origin_migration_rejects_historical_duplicate_uids()
    -> Result<(), Box<dyn std::error::Error>> {
        let connection = Connection::open(":memory:")?;
        MigrationRunner::new()
            .add(3, "create_stable_agent_uid_schema", SCRIPTBOTS_SCHEMA_V3)
            .add(
                4,
                "create_stable_uid_persistence_outbox",
                SCRIPTBOTS_SCHEMA_V4,
            )
            .run(&connection)?;
        connection.execute(
            "INSERT INTO births (
                tick, agent_uid, spawn_ordinal, birth_ordinal,
                herbivore_tendency, generation, position_x, position_y, is_hybrid
             ) VALUES (1, 7, 0, 0, 0.5, 0, 1.0, 2.0, 0)",
        )?;
        connection.execute(
            "INSERT INTO births (
                tick, agent_uid, spawn_ordinal, birth_ordinal,
                herbivore_tendency, generation, position_x, position_y, is_hybrid
             ) VALUES (2, 7, 1, 1, 0.5, 0, 3.0, 4.0, 0)",
        )?;

        install_scriptbots_schema(&connection)
            .expect_err("V5 must reject historical databases with duplicate origin UIDs");
        let v5_count: i64 = connection
            .query_row("SELECT COUNT(*) FROM _schema_migrations WHERE version = 5")?
            .get_typed(0)?;
        assert_eq!(v5_count, 0, "a failed uniqueness migration was recorded");

        let births_schema: String = connection
            .query_row("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'births'")?
            .get_typed(0)?;
        assert!(
            !births_schema.contains("origin"),
            "failed V5 leaked its replacement schema instead of rolling back: {births_schema}"
        );
        let replacement_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'births_v5'",
            )?
            .get_typed(0)?;
        assert_eq!(
            replacement_count, 0,
            "failed V5 left its rebuild table behind"
        );
        let historical_rows: i64 = connection
            .query_row("SELECT COUNT(*) FROM births WHERE agent_uid = 7")?
            .get_typed(0)?;
        assert_eq!(
            historical_rows, 2,
            "failed V5 did not restore both historical duplicate rows"
        );

        connection.execute("UPDATE births SET agent_uid = 8 WHERE tick = 2 AND agent_uid = 7")?;
        install_scriptbots_schema(&connection)?;
        let migrated_origins: i64 = connection
            .query_row("SELECT COUNT(*) FROM births WHERE origin = 'born'")?
            .get_typed(0)?;
        assert_eq!(migrated_origins, 2, "retry did not backfill every V4 row");
        let unique_index_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'index' AND name IN (
                     'births_agent_uid_unique',
                     'births_spawn_ordinal_unique',
                     'births_birth_ordinal_unique',
                     'deaths_agent_uid_unique'
                 )",
            )?
            .get_typed(0)?;
        assert_eq!(
            unique_index_count, 4,
            "retry did not finish every V5 lifecycle uniqueness index"
        );
        connection.close()?;
        Ok(())
    }

    #[test]
    fn birth_origin_migration_rejects_duplicate_ordinals_atomically_and_retries()
    -> Result<(), Box<dyn std::error::Error>> {
        for (duplicate_kind, rows, repair_sql) in [
            (
                "spawn",
                [(1_i64, 7_i64, 9_i64, 0_i64), (2, 8, 9, 1)],
                "UPDATE births SET spawn_ordinal = 10 WHERE tick = 2",
            ),
            (
                "birth",
                [(1_i64, 7_i64, 9_i64, 9_i64), (2, 8, 10, 9)],
                "UPDATE births SET birth_ordinal = 10 WHERE tick = 2",
            ),
        ] {
            let connection = Connection::open(":memory:")?;
            MigrationRunner::new()
                .add(3, "create_stable_agent_uid_schema", SCRIPTBOTS_SCHEMA_V3)
                .add(
                    4,
                    "create_stable_uid_persistence_outbox",
                    SCRIPTBOTS_SCHEMA_V4,
                )
                .run(&connection)?;
            for (tick, agent_uid, spawn_ordinal, birth_ordinal) in rows {
                connection.execute_with_params(
                    "INSERT INTO births (
                        tick, agent_uid, spawn_ordinal, birth_ordinal,
                        herbivore_tendency, generation, position_x, position_y, is_hybrid
                     ) VALUES (?1, ?2, ?3, ?4, 0.5, 0, 1.0, 2.0, 0)",
                    &[
                        tick.into(),
                        agent_uid.into(),
                        spawn_ordinal.into(),
                        birth_ordinal.into(),
                    ],
                )?;
            }

            let expected_failure = format!(
                "V5 must reject historical databases with duplicate {duplicate_kind} ordinals"
            );
            install_scriptbots_schema(&connection).expect_err(&expected_failure);
            let v5_count: i64 = connection
                .query_row("SELECT COUNT(*) FROM _schema_migrations WHERE version = 5")?
                .get_typed(0)?;
            assert_eq!(
                v5_count, 0,
                "a failed {duplicate_kind}-ordinal migration was recorded"
            );
            let births_schema: String = connection
                .query_row(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'births'",
                )?
                .get_typed(0)?;
            assert!(
                !births_schema.contains("origin"),
                "failed {duplicate_kind}-ordinal V5 leaked its replacement schema: {births_schema}"
            );
            let replacement_count: i64 = connection
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type = 'table' AND name = 'births_v5'",
                )?
                .get_typed(0)?;
            assert_eq!(
                replacement_count, 0,
                "failed {duplicate_kind}-ordinal V5 left its rebuild table behind"
            );
            let historical_rows: i64 = connection
                .query_row("SELECT COUNT(*) FROM births")?
                .get_typed(0)?;
            assert_eq!(
                historical_rows, 2,
                "failed {duplicate_kind}-ordinal V5 did not restore both historical rows"
            );

            connection.execute(repair_sql)?;
            install_scriptbots_schema(&connection)?;
            let index_count: i64 = connection
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type = 'index' AND name IN (
                         'births_agent_uid_unique',
                         'births_spawn_ordinal_unique',
                         'births_birth_ordinal_unique'
                     )",
                )?
                .get_typed(0)?;
            assert_eq!(
                index_count, 3,
                "retry after repairing duplicate {duplicate_kind} ordinals missed an index"
            );
            connection.execute(
                "INSERT INTO births (
                    tick, agent_uid, spawn_ordinal, birth_ordinal,
                    herbivore_tendency, generation, position_x, position_y, is_hybrid, origin
                 ) VALUES (0, 1, 0, NULL, 0.5, 0, 1.0, 2.0, 0, 'seeded')",
            )?;
            connection.execute(
                "INSERT INTO births (
                    tick, agent_uid, spawn_ordinal, birth_ordinal,
                    herbivore_tendency, generation, position_x, position_y, is_hybrid, origin
                 ) VALUES (21, 21, 21, NULL, 0.5, 0, 1.0, 2.0, 0, 'injected')",
            )?;
            connection.close()?;
        }
        Ok(())
    }

    #[test]
    fn birth_origin_migration_rejects_duplicate_death_uids_atomically_and_retries()
    -> Result<(), Box<dyn std::error::Error>> {
        let connection = Connection::open(":memory:")?;
        MigrationRunner::new()
            .add(3, "create_stable_agent_uid_schema", SCRIPTBOTS_SCHEMA_V3)
            .add(
                4,
                "create_stable_uid_persistence_outbox",
                SCRIPTBOTS_SCHEMA_V4,
            )
            .run(&connection)?;
        for (tick, cause) in [(1_i64, "starvation"), (2_i64, "aging")] {
            connection.execute_with_params(
                "INSERT INTO deaths (
                    tick, agent_uid, age, generation, herbivore_tendency,
                    energy, food_balance_total, cause, was_hybrid,
                    spike_attacker, spike_victim, hit_carnivore, hit_herbivore,
                    hit_by_carnivore, hit_by_herbivore
                 ) VALUES (?1, 7, 1, 0, 0.5, 0.0, 0.0, ?2, 0, 0, 0, 0, 0, 0, 0)",
                &[tick.into(), cause.into()],
            )?;
        }

        install_scriptbots_schema(&connection)
            .expect_err("V5 must reject historical databases with duplicate death UIDs");
        let v5_count: i64 = connection
            .query_row("SELECT COUNT(*) FROM _schema_migrations WHERE version = 5")?
            .get_typed(0)?;
        assert_eq!(v5_count, 0, "a failed death-UID migration was recorded");
        let births_schema: String = connection
            .query_row("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'births'")?
            .get_typed(0)?;
        assert!(
            !births_schema.contains("origin"),
            "failed death-UID V5 leaked the birth-table rebuild: {births_schema}"
        );
        let death_rows: i64 = connection
            .query_row("SELECT COUNT(*) FROM deaths WHERE agent_uid = 7")?
            .get_typed(0)?;
        assert_eq!(
            death_rows, 2,
            "failed death-UID V5 did not restore both historical rows"
        );

        connection.execute("UPDATE deaths SET agent_uid = 8 WHERE tick = 2")?;
        install_scriptbots_schema(&connection)?;
        let death_index_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'index' AND name = 'deaths_agent_uid_unique'",
            )?
            .get_typed(0)?;
        assert_eq!(
            death_index_count, 1,
            "retry did not finish the death UID uniqueness index"
        );
        connection.close()?;
        Ok(())
    }

    #[test]
    fn recovery_requires_the_exact_supported_migration_set_without_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        let future = temp_db_path("storage-recovery-future-migration");
        create_valid_database(&future)?;
        add_schema_object(
            &future,
            "INSERT INTO _schema_migrations (version, name) VALUES (6, 'future_schema')",
        )?;
        assert_recovery_refused_without_database_mutation(&future, "exactly three")?;

        let incomplete = temp_db_path("storage-recovery-incomplete-migration-set");
        let incomplete_connection = Connection::open(incomplete.to_string_lossy().as_ref())?;
        MigrationRunner::new()
            .add(3, "create_stable_agent_uid_schema", SCRIPTBOTS_SCHEMA_V3)
            .run(&incomplete_connection)?;
        incomplete_connection.close()?;
        assert_recovery_refused_without_database_mutation(&incomplete, "exactly three")?;
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
             VALUES (3, 'create_stable_agent_uid_schema', 'forged');
             INSERT INTO _schema_migrations (version, name, applied_at)
             VALUES (4, 'create_stable_uid_persistence_outbox', 'forged');
             INSERT INTO _schema_migrations (version, name, applied_at)
             VALUES (5, 'record_birth_origin', 'forged');
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
        match fs::hard_link(&original, &hard_link) {
            Ok(()) => {
                let hard_link_error =
                    match StoragePipeline::recover_existing(hard_link.to_string_lossy().as_ref()) {
                        Ok(mut pipeline) => {
                            pipeline.shutdown()?;
                            return Err("hard-link writer path unexpectedly succeeded".into());
                        }
                        Err(error) => error,
                    };
                assert!(hard_link_error.to_string().contains("multiply linked"));

                // AND THE ORIGINAL IS NOW REFUSED TOO. This is the part that makes the guard
                // worth having: once a second name for the file exists, writing through EITHER
                // name is unsafe, because the lease taken on one cannot see writes made through
                // the other. So the original path is refused as well.
                //
                // It is asserted HERE, inside the branch where the link was actually created,
                // because it is a CONSEQUENCE of the alias existing. Where the filesystem cannot
                // create hard links there is no alias, the file has one name, and the original is
                // rightly openable — asserting otherwise would be demanding a refusal with no
                // reason behind it.
                assert!(
                    StoragePipeline::recover_existing(&original_string).is_err(),
                    "a hard link to the database exists, but the ORIGINAL path still opened. \
                     Writes through the alias would bypass this lease entirely."
                );
            }
            // THE FILESYSTEM CANNOT CREATE HARD LINKS AT ALL (exFAT, FAT32).
            //
            // This is not a security check being skipped — it is an attack that CANNOT BE MOUNTED
            // here. There is no second name for the file to write through, because the filesystem
            // has no way to make one. The symlink half above ran, and still passed.
            //
            // The REASON is asserted rather than swallowed. A bare `is_err()` would let this test
            // quietly stop testing anything the day hard_link began failing for some entirely
            // different reason — a permission problem, a full disk — and a security test that has
            // silently disabled itself is worse than one that was never written.
            Err(error) if error.raw_os_error() == Some(libc::ENOTSUP) => {}
            Err(error) => {
                return Err(format!(
                    "could not create a hard link, and NOT because the filesystem lacks the \
                     feature: {error} (os error {:?}). The hard-link half of this security test \
                     did not run, and the reason is unexplained.",
                    error.raw_os_error()
                )
                .into());
            }
        }
        Ok(())
    }

    #[test]
    fn a_database_opens_on_a_filesystem_without_stable_inodes()
    -> Result<(), Box<dyn std::error::Error>> {
        // THE REGRESSION THIS BEAD EXISTS FOR (bd-15c8).
        //
        // Storage refused to open ANY database on exFAT, accusing the user's own file of having
        // "changed during validated open". The swapped-file guard compares (device, inode), and
        // exFAT has no inodes: the kernel synthesizes one from the file's STARTING CLUSTER, so
        // truncating and regrowing a file — exactly what creating and initialising a database does
        // — MOVES it. Our own initialisation was being read as tampering.
        //
        // exFAT is what external drives and USB sticks are formatted as, so putting a run database
        // on an external disk was simply broken, with an error that sent the reader hunting for a
        // corruption bug that did not exist.
        //
        // This test runs the real open path against whatever filesystem TMPDIR is on. It must
        // succeed on BOTH: where inodes are stable the guard is enforced, and where they are not it
        // is skipped (loudly) rather than firing falsely.
        let path = temp_db_path("storage-unstable-inode-open");
        let path_string = path.to_string_lossy().to_string();

        let mut pipeline = StoragePipeline::create_new_file(&path_string)?;
        pipeline.shutdown()?;

        // And it must REOPEN — the recovery path is where the identity check actually runs.
        let mut reopened = StoragePipeline::recover_existing(&path_string)?;
        reopened.shutdown()?;

        Ok(())
    }

    #[test]
    fn the_identity_probe_agrees_with_the_filesystem_it_probes() {
        // The probe decides whether a security check can be enforced, so it must not be a coin
        // flip. Ask it twice about the same directory: a probe that disagreed with itself would
        // enable the guard on one run and disable it on the next, and neither answer could be
        // trusted.
        let path = temp_db_path("storage-identity-probe");
        std::fs::write(&path, b"probe").expect("write probe database");

        let first = filesystem_has_stable_file_identity(&path);
        let second = filesystem_has_stable_file_identity(&path);
        assert_eq!(
            first, second,
            "the identity probe gave two different answers about the same filesystem. It decides \
             whether a security check is enforceable, so an unstable answer means the guard is on \
             or off depending on the run — and neither state could be trusted."
        );

        // And it must not leave litter behind in the user's database directory.
        let dir = path.parent().expect("temp dir");
        let leftovers = std::fs::read_dir(dir)
            .expect("read temp dir")
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".scriptbots-identity-probe-")
            })
            .count();
        assert_eq!(
            leftovers, 0,
            "the identity probe left its temporary files behind in the database directory"
        );
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
    fn durable_outbox_recovers_all_birth_origins_after_the_worker_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-outbox-admission-recovery");
        let path_string = path.to_string_lossy().to_string();
        let mut batch = sample_batch(21, 2.1);
        let mut born = sample_birth(21, 17, BirthOrigin::Born);
        born.spawn_ordinal = 41;
        born.birth_ordinal = Some(73);
        let mut seeded = sample_birth(0, 29, BirthOrigin::Seeded);
        seeded.spawn_ordinal = 0;
        let mut injected = sample_birth(21, 37, BirthOrigin::Injected);
        injected.spawn_ordinal = 64;
        batch.births = vec![born, seeded, injected];
        synchronize_lifecycle_counts(&mut batch);
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

        let reader = StorageReader::open(&path_string)?;
        let persisted_arrivals = reader
            .load_ancestry_births()?
            .into_iter()
            .map(|birth| {
                (
                    birth.agent_uid,
                    birth.spawn_ordinal,
                    birth.birth_ordinal,
                    birth.origin,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            persisted_arrivals,
            vec![
                (AgentUid(29), 0, None, BirthOrigin::Seeded),
                (AgentUid(17), 41, Some(73), BirthOrigin::Born),
                (AgentUid(37), 64, None, BirthOrigin::Injected),
            ],
            "durable outbox recovery changed an origin or persisted ordinal"
        );
        let ordinal_rows = reader.connection()?.query(
            "SELECT agent_uid, spawn_ordinal, birth_ordinal, origin
                 FROM births ORDER BY agent_uid ASC",
        )?;
        let persisted_ordinals = ordinal_rows
            .iter()
            .map(|row| {
                Ok::<_, StorageError>((
                    decode::<i64>(row, 0, "births.agent_uid")?,
                    decode::<i64>(row, 1, "births.spawn_ordinal")?,
                    decode::<Option<i64>>(row, 2, "births.birth_ordinal")?,
                    decode::<String>(row, 3, "births.origin")?,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            persisted_ordinals,
            vec![
                (17, 41, Some(73), "born".to_owned()),
                (29, 0, None, "seeded".to_owned()),
                (37, 64, None, "injected".to_owned()),
            ],
            "durable outbox recovery did not preserve exact raw identity and ordinal columns"
        );
        assert_eq!(reader.run_ledger_summary()?.birth_records, 1);
        reader.close()?;

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
    fn recovery_rejects_ancestry_incoherence_before_partial_replay()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-outbox-ancestry-corruption");
        let path_string = path.to_string_lossy().to_string();
        let mut interrupted = create_file_storage(&path_string)?;

        let mut root_batch = sample_batch(10, 1.0);
        root_batch.summary.births = 0;
        root_batch.births = vec![sample_birth(10, 200, BirthOrigin::Injected)];
        synchronize_lifecycle_counts(&mut root_batch);
        let root = PreparedPersistenceBatch::from_batch(&root_batch)?;
        let (root_admission, root_is_new) = interrupted.stage_outbox(root.tick, &root.storage)?;
        assert!(root_is_new);
        assert!(!interrupted.enqueue_staged(root_admission.batch_id, root.storage)?);

        let mut child_batch = sample_batch(11, 1.1);
        let mut child = sample_birth(11, 201, BirthOrigin::Born);
        child.parent_a = Some(AgentUid(200));
        child_batch.births = vec![child];
        synchronize_lifecycle_counts(&mut child_batch);
        let child = PreparedPersistenceBatch::from_batch(&child_batch)?;
        let (child_admission, child_is_new) =
            interrupted.stage_outbox(child.tick, &child.storage)?;
        assert!(child_is_new);
        assert!(!interrupted.enqueue_staged(child_admission.batch_id, child.storage)?);
        interrupted.abandon_after_error();

        // Model a correctly hashed but semantically corrupted durable outbox.
        // Decode-level shape checks must pass so the recovery-only relational
        // validator is the layer that refuses it.
        let corruptor = Connection::open(&path_string)?;
        let original_payload: String = corruptor
            .query_row_with_params(
                "SELECT payload FROM storage_outbox WHERE batch_id = ?1",
                &[child_admission.batch_id.as_i64().into()],
            )?
            .get_typed(0)?;
        let mut corrupted: Value = serde_json::from_str(&original_payload)?;
        corrupted["storage"]["births"][0]["parent_a"] = json!(999_i64);
        let corrupted_payload = serde_json::to_string(&corrupted)?;
        let corrupted_digest = format!(
            "blake3:{}",
            blake3::hash(corrupted_payload.as_bytes()).to_hex()
        );
        assert_eq!(
            corruptor.execute_with_params(
                "UPDATE storage_outbox SET payload = ?1 WHERE batch_id = ?2",
                &[
                    corrupted_payload.as_str().into(),
                    child_admission.batch_id.as_i64().into(),
                ],
            )?,
            1
        );
        assert_eq!(
            corruptor.execute_with_params(
                "UPDATE storage_batch_ledger SET payload_digest = ?1 WHERE batch_id = ?2",
                &[
                    corrupted_digest.as_str().into(),
                    child_admission.batch_id.as_i64().into(),
                ],
            )?,
            1
        );
        corruptor.close()?;

        let error = recover_file_storage(&path_string)
            .err()
            .expect("recovery accepted an outbox child whose parent was never recorded");
        assert!(matches!(
            error,
            StorageError::InvalidData {
                context: "births.parent_a",
                ..
            }
        ));

        let inspector = Connection::open(&path_string)?;
        let tick_count: i64 = inspector
            .query_row("SELECT COUNT(*) FROM ticks")?
            .get_typed(0)?;
        assert_eq!(
            tick_count, 0,
            "failed recovery partially replayed the root batch"
        );
        let progress = inspector.query_row(
            "SELECT admitted_batch_id, applied_batch_id, durable_batch_id
             FROM storage_progress WHERE singleton = 1",
        )?;
        assert_eq!(
            progress.get_typed::<i64>(0)?,
            child_admission.batch_id.as_i64()
        );
        assert_eq!(progress.get_typed::<i64>(1)?, 0);
        assert_eq!(progress.get_typed::<i64>(2)?, 0);
        inspector.close()?;
        Ok(())
    }

    #[test]
    fn recovery_rejects_rehashed_lifecycle_count_divergence_before_partial_replay()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-outbox-lifecycle-count-corruption");
        let path_string = path.to_string_lossy().to_string();
        let mut interrupted = create_file_storage(&path_string)?;

        let root = PreparedPersistenceBatch::from_batch(&sample_batch(10, 1.0))?;
        let (root_admission, root_is_new) = interrupted.stage_outbox(root.tick, &root.storage)?;
        assert!(root_is_new);
        assert!(!interrupted.enqueue_staged(root_admission.batch_id, root.storage)?);

        let mut born_batch = sample_batch(11, 1.1);
        born_batch.births = vec![sample_birth(11, 401, BirthOrigin::Born)];
        synchronize_lifecycle_counts(&mut born_batch);
        let born = PreparedPersistenceBatch::from_batch(&born_batch)?;
        let (born_admission, born_is_new) = interrupted.stage_outbox(born.tick, &born.storage)?;
        assert!(born_is_new);
        assert!(!interrupted.enqueue_staged(born_admission.batch_id, born.storage)?);
        interrupted.abandon_after_error();

        // Rehash both durable identities after deleting the origin row. Digest
        // integrity therefore passes; semantic cross-field validation must be
        // what refuses replay before the valid root batch is partially applied.
        let corruptor = Connection::open(&path_string)?;
        let original_payload: String = corruptor
            .query_row_with_params(
                "SELECT payload FROM storage_outbox WHERE batch_id = ?1",
                &[born_admission.batch_id.as_i64().into()],
            )?
            .get_typed(0)?;
        let mut corrupted: Value = serde_json::from_str(&original_payload)?;
        corrupted["storage"]["births"] = json!([]);
        let corrupted_payload = serde_json::to_string(&corrupted)?;
        let corrupted_digest = format!(
            "blake3:{}",
            blake3::hash(corrupted_payload.as_bytes()).to_hex()
        );
        assert_eq!(
            corruptor.execute_with_params(
                "UPDATE storage_outbox SET payload = ?1 WHERE batch_id = ?2",
                &[
                    corrupted_payload.as_str().into(),
                    born_admission.batch_id.as_i64().into(),
                ],
            )?,
            1
        );
        assert_eq!(
            corruptor.execute_with_params(
                "UPDATE storage_batch_ledger SET payload_digest = ?1 WHERE batch_id = ?2",
                &[
                    corrupted_digest.as_str().into(),
                    born_admission.batch_id.as_i64().into(),
                ],
            )?,
            1
        );
        corruptor.close()?;

        let error = recover_file_storage(&path_string)
            .err()
            .expect("recovery accepted a demographic birth with no Born origin row");
        assert!(matches!(
            error,
            StorageError::InvalidData {
                context: "ticks.births",
                ..
            }
        ));

        let inspector = Connection::open(&path_string)?;
        let tick_count: i64 = inspector
            .query_row("SELECT COUNT(*) FROM ticks")?
            .get_typed(0)?;
        assert_eq!(
            tick_count, 0,
            "failed recovery partially replayed the valid root batch"
        );
        let progress = inspector.query_row(
            "SELECT admitted_batch_id, applied_batch_id, durable_batch_id
             FROM storage_progress WHERE singleton = 1",
        )?;
        assert_eq!(
            progress.get_typed::<i64>(0)?,
            born_admission.batch_id.as_i64()
        );
        assert_eq!(progress.get_typed::<i64>(1)?, 0);
        assert_eq!(progress.get_typed::<i64>(2)?, 0);
        inspector.close()?;
        Ok(())
    }

    #[test]
    fn recovery_rejects_lifecycle_values_outside_core_u32_before_partial_replay()
    -> Result<(), Box<dyn std::error::Error>> {
        let overflow = i64::from(u32::MAX) + 1;
        for (label, collection, field, context) in [
            (
                "birth-generation",
                "births",
                "generation",
                "births.generation",
            ),
            ("death-age", "deaths", "age", "deaths.age"),
            (
                "death-generation",
                "deaths",
                "generation",
                "deaths.generation",
            ),
        ] {
            let path = temp_db_path(&format!("storage-outbox-{label}-overflow"));
            let path_string = path.to_string_lossy().to_string();
            let mut interrupted = create_file_storage(&path_string)?;

            let mut root_batch = sample_batch(10, 1.0);
            root_batch.summary.births = 0;
            root_batch.births = vec![sample_birth(10, 400, BirthOrigin::Injected)];
            synchronize_lifecycle_counts(&mut root_batch);
            let root = PreparedPersistenceBatch::from_batch(&root_batch)?;
            let (root_admission, root_is_new) =
                interrupted.stage_outbox(root.tick, &root.storage)?;
            assert!(root_is_new);
            assert!(!interrupted.enqueue_staged(root_admission.batch_id, root.storage)?);

            let mut second_batch = sample_batch(11, 1.1);
            if collection == "births" {
                let mut child = sample_birth(11, 401, BirthOrigin::Born);
                child.parent_a = Some(AgentUid(400));
                second_batch.births = vec![child];
            } else {
                second_batch.summary.births = 0;
                second_batch.summary.deaths = 1;
                second_batch.deaths = vec![sample_death(11, 400, DeathCause::Aging)];
            }
            synchronize_lifecycle_counts(&mut second_batch);
            let second = PreparedPersistenceBatch::from_batch(&second_batch)?;
            let (second_admission, second_is_new) =
                interrupted.stage_outbox(second.tick, &second.storage)?;
            assert!(second_is_new);
            assert!(!interrupted.enqueue_staged(second_admission.batch_id, second.storage)?);
            interrupted.abandon_after_error();

            let corruptor = Connection::open(&path_string)?;
            let original_payload: String = corruptor
                .query_row_with_params(
                    "SELECT payload FROM storage_outbox WHERE batch_id = ?1",
                    &[second_admission.batch_id.as_i64().into()],
                )?
                .get_typed(0)?;
            let mut corrupted: Value = serde_json::from_str(&original_payload)?;
            corrupted["storage"][collection][0][field] = json!(overflow);
            let corrupted_payload = serde_json::to_string(&corrupted)?;
            let corrupted_digest = format!(
                "blake3:{}",
                blake3::hash(corrupted_payload.as_bytes()).to_hex()
            );
            assert_eq!(
                corruptor.execute_with_params(
                    "UPDATE storage_outbox SET payload = ?1 WHERE batch_id = ?2",
                    &[
                        corrupted_payload.as_str().into(),
                        second_admission.batch_id.as_i64().into(),
                    ],
                )?,
                1
            );
            assert_eq!(
                corruptor.execute_with_params(
                    "UPDATE storage_batch_ledger SET payload_digest = ?1 WHERE batch_id = ?2",
                    &[
                        corrupted_digest.as_str().into(),
                        second_admission.batch_id.as_i64().into(),
                    ],
                )?,
                1
            );
            corruptor.close()?;

            let recovery = recover_file_storage(&path_string);
            assert!(recovery.is_err(), "recovery accepted overflowing {context}");
            let Err(error) = recovery else {
                continue;
            };
            assert!(matches!(
                error,
                StorageError::InvalidData {
                    context: actual,
                    ..
                } if actual == context
            ));

            let inspector = Connection::open(&path_string)?;
            let tick_count: i64 = inspector
                .query_row("SELECT COUNT(*) FROM ticks")?
                .get_typed(0)?;
            assert_eq!(
                tick_count, 0,
                "failed {context} recovery partially replayed its valid prefix"
            );
            let progress = inspector.query_row(
                "SELECT admitted_batch_id, applied_batch_id, durable_batch_id
                 FROM storage_progress WHERE singleton = 1",
            )?;
            assert_eq!(
                progress.get_typed::<i64>(0)?,
                second_admission.batch_id.as_i64()
            );
            assert_eq!(progress.get_typed::<i64>(1)?, 0);
            assert_eq!(progress.get_typed::<i64>(2)?, 0);
            inspector.close()?;
        }
        Ok(())
    }

    #[test]
    fn ancestry_reader_rejects_generation_outside_the_core_u32_domain()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-ancestry-generation-domain");
        let path_string = path.to_string_lossy().to_string();
        let mut batch = sample_batch(23, 2.3);
        batch.births = vec![sample_birth(23, 11, BirthOrigin::Injected)];
        synchronize_lifecycle_counts(&mut batch);

        let mut storage = create_file_storage(&path_string)?;
        storage.persist(&batch)?;
        storage.flush()?;
        storage.close()?;

        let connection = Connection::open(&path_string)?;
        connection.execute_with_params(
            "UPDATE births SET generation = ?1 WHERE agent_uid = 11",
            &[(i64::from(u32::MAX) + 1).into()],
        )?;
        connection.close()?;

        let reader = StorageReader::open(&path_string)?;
        assert_invalid_data_context(reader.load_ancestry_births(), "births.generation");
        reader.close()?;
        Ok(())
    }

    #[test]
    fn stale_or_tampered_birth_origin_outbox_payload_is_refused() -> Result<(), StorageError> {
        let mut batch = sample_batch(22, 2.2);
        batch.births = vec![
            sample_birth(22, 1, BirthOrigin::Born),
            sample_birth(0, 2, BirthOrigin::Seeded),
            sample_birth(22, 3, BirthOrigin::Injected),
        ];
        synchronize_lifecycle_counts(&mut batch);
        let prepared = PreparedPersistenceBatch::from_batch(&batch)?;
        let (payload, payload_digest) = prepared.storage.encode_outbox(prepared.tick)?;
        let stale_payload = r#"{"version":2,"tick":22,"storage":{"ticks":[],"metrics":[],"events":[],"agents":[],"births":[{"tick":22,"agent_uid":1,"spawn_ordinal":0,"birth_ordinal":0,"parent_a":null,"parent_b":null,"brain_kind":"legacy","brain_key":null,"herbivore_tendency":0.5,"generation":0,"position_x":1.0,"position_y":2.0,"is_hybrid":false}],"deaths":[],"replay_events":[]}}"#;
        assert!(
            !stale_payload.contains("\"origin\""),
            "the V2 fixture accidentally included the V3-only origin field"
        );
        let stale_digest = format!("blake3:{}", blake3::hash(stale_payload.as_bytes()).to_hex());
        let version_error =
            StorageBuffer::decode_outbox(stale_payload, prepared.tick, &stale_digest)
                .expect_err("a correctly hashed V2-shaped payload must still be refused");
        assert!(matches!(
            version_error,
            StorageError::InvalidData {
                context: "storage_outbox.payload.version",
                ..
            }
        ));

        let inconsistent_payload =
            payload.replacen("\"birth_ordinal\":1", "\"birth_ordinal\":null", 1);
        assert_ne!(
            inconsistent_payload, payload,
            "the test failed to remove the born arrival's ordinal"
        );
        let inconsistent_digest = format!(
            "blake3:{}",
            blake3::hash(inconsistent_payload.as_bytes()).to_hex()
        );
        let invariant_error = StorageBuffer::decode_outbox(
            &inconsistent_payload,
            prepared.tick,
            &inconsistent_digest,
        )
        .expect_err("a correctly hashed payload with an inconsistent ordinal must be refused");
        assert!(matches!(
            invariant_error,
            StorageError::InvalidData {
                context: "births.birth_ordinal",
                ..
            }
        ));

        let late_seeded_payload = payload.replacen(
            "\"tick\":0,\"agent_uid\":2",
            "\"tick\":22,\"agent_uid\":2",
            1,
        );
        assert_ne!(
            late_seeded_payload, payload,
            "the test failed to move the seeded founder after bootstrap"
        );
        let late_seeded_digest = format!(
            "blake3:{}",
            blake3::hash(late_seeded_payload.as_bytes()).to_hex()
        );
        let late_seeded_error =
            StorageBuffer::decode_outbox(&late_seeded_payload, prepared.tick, &late_seeded_digest)
                .expect_err("a correctly hashed nonzero seeded arrival must be refused");
        assert!(matches!(
            late_seeded_error,
            StorageError::InvalidData {
                context: "births.origin",
                ..
            }
        ));

        let tampered_payload = payload.replacen("\"tick\":22", "\"tick\":23", 1);
        assert_ne!(
            tampered_payload, payload,
            "the test failed to tamper with the current outbox payload"
        );
        let tamper_error =
            StorageBuffer::decode_outbox(&tampered_payload, prepared.tick, &payload_digest)
                .expect_err("changing the payload without its digest must be refused");
        assert!(matches!(
            tamper_error,
            StorageError::InvalidData {
                context: "storage_outbox.payload_digest",
                ..
            }
        ));
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
        let mut batch = sample_batch(51, 5.1);
        let mut birth = sample_birth(51, 17, BirthOrigin::Born);
        birth.spawn_ordinal = 41;
        birth.birth_ordinal = Some(73);
        batch.births.push(birth);
        synchronize_lifecycle_counts(&mut batch);
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
    fn replay_rng_scope_preserves_inner_and_outer_agent_uids()
    -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-replay-rng-agent-ids");
        let path_string = path.to_string_lossy().to_string();
        let mut storage =
            Storage::create_new_file_with_thresholds(&path_string, 64, 4096, 1024, 1024)?;
        let outer_agent = AgentUid(0x0000_0001_0000_0001);
        let scope_agent = AgentUid(0x0000_0002_0000_0001);
        let mut batch = sample_batch(5, 1.0);
        batch.replay_events.push(ReplayEvent {
            agent_uid: Some(outer_agent),
            kind: ReplayEventKind::RngSample {
                scope: ReplayRngScope::Agent {
                    agent_uid: scope_agent,
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
    fn replay_action_preserves_spike_target_agent_uid() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-replay-action-spike-target");
        let path_string = path.to_string_lossy().to_string();
        let mut storage =
            Storage::create_new_file_with_thresholds(&path_string, 64, 4096, 1024, 1024)?;
        let actor = AgentUid(0x0000_0001_0000_0001);
        let target = AgentUid(0x0000_0002_0000_0001);
        let mut batch = sample_batch(6, 1.0);
        batch.replay_events.push(ReplayEvent {
            agent_uid: Some(actor),
            kind: ReplayEventKind::Action {
                left_wheel: -0.25,
                right_wheel: 0.75,
                boost: true,
                spike_target: Some(target),
                sound_level: 0.5,
                give_intent: 0.125,
            },
        });
        let encoded_row = replay_row_from_event(&batch.replay_events[0], 6, 0)?;
        assert_eq!(encoded_row.scope, "agent:action");
        assert_eq!(encoded_row.event_type, "action");
        assert_eq!(
            encoded_row.payload,
            r#"{"left_wheel":-0.25,"right_wheel":0.75,"boost":true,"spike_target":8589934593,"sound_level":0.5,"give_intent":0.125}"#,
            "replay payload changes require a versioned schema boundary"
        );
        assert_eq!(replay_event_from_row(&encoded_row)?, batch.replay_events[0]);
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
            agent_uid: None,
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
            agent_uid: AgentUid(2),
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
        synchronize_lifecycle_counts(&mut invalid);
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
            agent_uid: None,
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
    fn production_birth_insert_rejects_duplicate_uid_across_ticks()
    -> Result<(), Box<dyn std::error::Error>> {
        let storage = Storage::memory()?;
        let rows = [
            birth_row_from_record(&sample_birth(0, 7, BirthOrigin::Seeded))?,
            birth_row_from_record(&sample_birth(12, 7, BirthOrigin::Injected))?,
        ];
        {
            let mut first = storage.connection()?.transaction()?;
            Storage::insert_births(&first, &rows[..1])?;
            first.commit()?;
        }
        {
            let mut conflicting = storage.connection()?.transaction()?;
            Storage::insert_births(&conflicting, &rows[1..])
                .expect_err("a second origin row for one uid must not replace the first");
            conflicting.rollback()?;
        }

        let persisted = storage.connection()?.query_row(
            "SELECT tick, agent_uid, spawn_ordinal, birth_ordinal,
                        position_x, position_y, origin
                 FROM births WHERE agent_uid = 7",
        )?;
        assert_eq!(persisted.get_typed::<i64>(0)?, 0);
        assert_eq!(persisted.get_typed::<i64>(1)?, 7);
        assert_eq!(persisted.get_typed::<i64>(2)?, 6);
        assert_eq!(persisted.get_typed::<Option<i64>>(3)?, None);
        assert_eq!(persisted.get_typed::<f64>(4)?, 1.0);
        assert_eq!(persisted.get_typed::<f64>(5)?, 2.0);
        assert_eq!(persisted.get_typed::<String>(6)?, "seeded");
        storage.close()?;
        Ok(())
    }

    #[test]
    fn production_birth_insert_enforces_unique_ordinals_and_nullable_non_birth_ordinals()
    -> Result<(), Box<dyn std::error::Error>> {
        let storage = Storage::memory()?;
        let mut canonical = sample_birth(11, 7, BirthOrigin::Born);
        canonical.spawn_ordinal = 20;
        canonical.birth_ordinal = Some(30);
        let canonical = birth_row_from_record(&canonical)?;
        {
            let mut transaction = storage.connection()?.transaction()?;
            Storage::insert_births(&transaction, std::slice::from_ref(&canonical))?;
            transaction.commit()?;
        }

        let mut duplicate_spawn = sample_birth(12, 8, BirthOrigin::Born);
        duplicate_spawn.spawn_ordinal = 20;
        duplicate_spawn.birth_ordinal = Some(31);
        let mut duplicate_birth = sample_birth(13, 9, BirthOrigin::Born);
        duplicate_birth.spawn_ordinal = 21;
        duplicate_birth.birth_ordinal = Some(30);
        for conflicting in [duplicate_spawn, duplicate_birth] {
            let conflicting = birth_row_from_record(&conflicting)?;
            let mut transaction = storage.connection()?.transaction()?;
            Storage::insert_births(&transaction, &[conflicting])
                .expect_err("a duplicate insertion or birth ordinal must not replace a row");
            transaction.rollback()?;
        }

        let mut seeded = sample_birth(0, 10, BirthOrigin::Seeded);
        seeded.spawn_ordinal = 22;
        let mut injected = sample_birth(15, 11, BirthOrigin::Injected);
        injected.spawn_ordinal = 23;
        let nullable_rows = [
            birth_row_from_record(&seeded)?,
            birth_row_from_record(&injected)?,
        ];
        {
            let mut transaction = storage.connection()?.transaction()?;
            Storage::insert_births(&transaction, &nullable_rows)?;
            transaction.commit()?;
        }

        let mut late_seeded = sample_birth(16, 12, BirthOrigin::Seeded);
        late_seeded.spawn_ordinal = 24;
        let late_seeded = birth_row_from_record(&late_seeded)?;
        {
            let mut transaction = storage.connection()?.transaction()?;
            Storage::insert_births(&transaction, &[late_seeded])
                .expect_err("the schema must reject a seeded founder after tick zero");
            transaction.rollback()?;
        }

        let persisted: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM births")?
            .get_typed(0)?;
        assert_eq!(
            persisted, 3,
            "unique optional birth ordinals must allow multiple NULL values"
        );
        storage.close()?;
        Ok(())
    }

    #[test]
    fn production_death_insert_never_replaces_an_existing_uid()
    -> Result<(), Box<dyn std::error::Error>> {
        let storage = Storage::memory()?;
        let original = death_row_from_record(&sample_death(11, 7, DeathCause::Starvation))?;
        {
            let mut transaction = storage.connection()?.transaction()?;
            Storage::insert_deaths(&transaction, std::slice::from_ref(&original))?;
            transaction.commit()?;
        }

        for conflicting in [
            sample_death(11, 7, DeathCause::Aging),
            sample_death(12, 7, DeathCause::CombatCarnivore),
        ] {
            let conflicting = death_row_from_record(&conflicting)?;
            let mut transaction = storage.connection()?.transaction()?;
            Storage::insert_deaths(&transaction, &[conflicting])
                .expect_err("a second death for one uid must fail instead of replacing it");
            transaction.rollback()?;
        }

        let persisted = storage
            .connection()?
            .query_row("SELECT tick, cause FROM deaths WHERE agent_uid = 7")?;
        assert_eq!(persisted.get_typed::<i64>(0)?, 11);
        assert_eq!(persisted.get_typed::<String>(1)?, "starvation");
        storage.close()?;
        Ok(())
    }

    #[test]
    fn birth_preparation_rejects_origin_ordinal_mismatches() {
        let mut born_without_ordinal = sample_birth(11, 1, BirthOrigin::Born);
        born_without_ordinal.birth_ordinal = None;
        assert!(matches!(
            birth_row_from_record(&born_without_ordinal),
            Err(StorageError::InvalidData {
                context: "births.birth_ordinal",
                ..
            })
        ));

        let mut seeded_with_ordinal = sample_birth(0, 2, BirthOrigin::Seeded);
        seeded_with_ordinal.birth_ordinal = Some(1);
        assert!(matches!(
            birth_row_from_record(&seeded_with_ordinal),
            Err(StorageError::InvalidData {
                context: "births.birth_ordinal",
                ..
            })
        ));

        let mut injected_with_ordinal = sample_birth(11, 3, BirthOrigin::Injected);
        injected_with_ordinal.birth_ordinal = Some(2);
        assert!(matches!(
            birth_row_from_record(&injected_with_ordinal),
            Err(StorageError::InvalidData {
                context: "births.birth_ordinal",
                ..
            })
        ));

        assert!(validate_birth_origin_ordinal(BirthOrigin::Born, Some(0)).is_ok());
        assert!(validate_birth_origin_ordinal(BirthOrigin::Seeded, None).is_ok());
        assert!(validate_birth_origin_ordinal(BirthOrigin::Injected, None).is_ok());
        assert!(validate_birth_origin_tick(BirthOrigin::Seeded, 0, 4).is_ok());
        assert!(validate_birth_origin_tick(BirthOrigin::Injected, 11, 5).is_ok());
        assert_invalid_data_context(
            validate_birth_origin_tick(BirthOrigin::Seeded, 11, 6),
            "births.origin",
        );
    }

    #[test]
    fn lifecycle_summary_rows_and_events_must_agree_before_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut missing_birth_row = sample_batch(35, 3.5);
        missing_birth_row.summary.births = 1;
        missing_birth_row
            .events
            .push(PersistenceEvent::new(PersistenceEventKind::Births, 1));

        let mut missing_death_row = sample_batch(36, 3.6);
        missing_death_row.summary.deaths = 1;
        missing_death_row
            .events
            .push(PersistenceEvent::new(PersistenceEventKind::Deaths, 1));

        let mut wrong_birth_event = sample_batch(37, 3.7);
        wrong_birth_event.births = vec![sample_birth(37, 370, BirthOrigin::Born)];
        synchronize_lifecycle_counts(&mut wrong_birth_event);
        wrong_birth_event
            .events
            .iter_mut()
            .find(|event| matches!(event.kind, PersistenceEventKind::Births))
            .expect("synchronized birth event")
            .count = 2;

        let mut missing_death_event = sample_batch(38, 3.8);
        missing_death_event.deaths = vec![sample_death(38, 380, DeathCause::Aging)];
        synchronize_lifecycle_counts(&mut missing_death_event);
        missing_death_event
            .events
            .retain(|event| !matches!(event.kind, PersistenceEventKind::Deaths));

        let mut late_seeded = sample_batch(39, 3.9);
        late_seeded.births = vec![sample_birth(39, 390, BirthOrigin::Seeded)];
        synchronize_lifecycle_counts(&mut late_seeded);

        let mut storage = Storage::memory()?;
        for (context, malformed) in [
            ("ticks.births", missing_birth_row),
            ("ticks.deaths", missing_death_row),
            ("events.births", wrong_birth_event),
            ("events.deaths", missing_death_event),
            ("births.origin", late_seeded),
        ] {
            assert_invalid_data_context(PreparedPersistenceBatch::from_batch(&malformed), context);
            assert_invalid_data_context(storage.persist(&malformed), context);
            assert!(storage.buffer.is_empty());
            assert_eq!(
                storage.persistence_watermarks()?,
                PersistenceWatermarks {
                    admitted: None,
                    applied: None,
                    durable: None,
                },
                "cross-field lifecycle mismatch received an outbox identity"
            );
        }

        storage.persist(&sample_batch(39, 3.9))?;
        storage.flush()?;
        let tick_count: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM ticks")?
            .get_typed(0)?;
        assert_eq!(tick_count, 1, "rejected mismatch poisoned the writer");
        storage.close()?;
        Ok(())
    }

    #[test]
    fn lifecycle_records_after_the_summary_tick_are_rejected_before_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut storage = Storage::memory()?;

        let mut future_birth = sample_batch(40, 4.0);
        future_birth
            .births
            .push(sample_birth(41, 7, BirthOrigin::Born));
        synchronize_lifecycle_counts(&mut future_birth);
        assert_invalid_data_context(
            PreparedPersistenceBatch::from_batch(&future_birth),
            "births.tick",
        );
        assert_invalid_data_context(storage.persist(&future_birth), "births.tick");
        assert!(storage.buffer.is_empty());

        let mut future_death = sample_batch(40, 4.0);
        future_death
            .deaths
            .push(sample_death(41, 7, DeathCause::Starvation));
        synchronize_lifecycle_counts(&mut future_death);
        assert_invalid_data_context(
            PreparedPersistenceBatch::from_batch(&future_death),
            "deaths.tick",
        );
        assert_invalid_data_context(storage.persist(&future_death), "deaths.tick");
        assert!(storage.buffer.is_empty());
        assert_eq!(
            storage.persistence_watermarks()?,
            PersistenceWatermarks {
                admitted: None,
                applied: None,
                durable: None,
            },
            "future lifecycle rows must fail before an outbox identity is assigned"
        );

        storage.persist(&sample_batch(42, 4.2))?;
        storage.flush()?;
        let tick_count: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM ticks")?
            .get_typed(0)?;
        assert_eq!(tick_count, 1, "rejected lifecycle rows poisoned the writer");
        storage.close()?;
        Ok(())
    }

    #[test]
    fn outbox_validation_rejects_lifecycle_rows_after_the_enclosing_tick()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut birth_batch = sample_batch(50, 5.0);
        birth_batch
            .births
            .push(sample_birth(50, 7, BirthOrigin::Born));
        synchronize_lifecycle_counts(&mut birth_batch);
        let mut future_birth = Storage::prepare_batch(&birth_batch)?;
        future_birth.births[0].tick = 51;

        let mut death_batch = sample_batch(50, 5.0);
        death_batch
            .deaths
            .push(sample_death(50, 8, DeathCause::Aging));
        synchronize_lifecycle_counts(&mut death_batch);
        let mut future_death = Storage::prepare_batch(&death_batch)?;
        future_death.deaths[0].tick = 51;

        for (context, invalid) in [("births.tick", future_birth), ("deaths.tick", future_death)] {
            assert_invalid_data_context(invalid.encode_outbox(50), context);

            let payload = serde_json::to_string(&OutboxPayloadRef {
                version: OUTBOX_PAYLOAD_VERSION,
                tick: 50,
                storage: &invalid,
            })?;
            let digest = format!("blake3:{}", blake3::hash(payload.as_bytes()).to_hex());
            assert_invalid_data_context(
                StorageBuffer::decode_outbox(&payload, 50, &digest),
                context,
            );
        }
        Ok(())
    }

    #[test]
    fn ancestry_incoherence_is_rejected_before_new_outbox_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut storage = Storage::memory()?;
        let no_watermarks = PersistenceWatermarks {
            admitted: None,
            applied: None,
            durable: None,
        };

        let mut unknown_parent = sample_batch(100, 10.0);
        let mut child = sample_birth(100, 20, BirthOrigin::Born);
        child.parent_a = Some(AgentUid(999));
        unknown_parent.births = vec![child];
        synchronize_lifecycle_counts(&mut unknown_parent);

        let mut duplicate_parent = sample_batch(101, 10.1);
        let parent = sample_birth(100, 30, BirthOrigin::Injected);
        let mut child = sample_birth(101, 31, BirthOrigin::Born);
        child.parent_a = Some(AgentUid(30));
        child.parent_b = Some(AgentUid(30));
        duplicate_parent.births = vec![parent, child];
        synchronize_lifecycle_counts(&mut duplicate_parent);

        let mut same_tick_parent = sample_batch(110, 11.0);
        let parent = sample_birth(110, 40, BirthOrigin::Injected);
        let mut child = sample_birth(110, 41, BirthOrigin::Born);
        child.parent_a = Some(AgentUid(40));
        same_tick_parent.births = vec![parent, child];
        synchronize_lifecycle_counts(&mut same_tick_parent);

        let mut unknown_death = sample_batch(120, 12.0);
        unknown_death.summary.births = 0;
        unknown_death.summary.deaths = 1;
        unknown_death.deaths = vec![sample_death(120, 50, DeathCause::Starvation)];
        synchronize_lifecycle_counts(&mut unknown_death);

        let mut same_tick_death = sample_batch(130, 13.0);
        same_tick_death.summary.deaths = 1;
        same_tick_death.births = vec![sample_birth(130, 60, BirthOrigin::Born)];
        same_tick_death.deaths = vec![sample_death(130, 60, DeathCause::Aging)];
        synchronize_lifecycle_counts(&mut same_tick_death);

        for (context, malformed) in [
            ("births.parent_a", unknown_parent),
            ("births.parent_b", duplicate_parent),
            ("births.parent_a", same_tick_parent),
            ("deaths.agent_uid", unknown_death),
            ("deaths.tick", same_tick_death),
        ] {
            assert_invalid_data_context(storage.persist(&malformed), context);
            assert!(
                storage.buffer.is_empty(),
                "rejected ancestry rows reached the scientific buffer"
            );
            assert_eq!(
                storage.persistence_watermarks()?,
                no_watermarks,
                "rejected ancestry rows received an outbox identity"
            );
        }
        let ledger_count: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM storage_batch_ledger")?
            .get_typed(0)?;
        assert_eq!(ledger_count, 0);

        // Prove the validator is relational rather than a blanket rejection: a
        // current batch may carry an earlier root, its later child, and the
        // root's later death.
        let mut valid = sample_batch(141, 14.1);
        valid.summary.deaths = 1;
        let root = sample_birth(140, 70, BirthOrigin::Injected);
        let mut child = sample_birth(141, 71, BirthOrigin::Born);
        child.parent_a = Some(AgentUid(70));
        valid.births = vec![root, child];
        valid.deaths = vec![sample_death(141, 70, DeathCause::Aging)];
        synchronize_lifecycle_counts(&mut valid);
        storage.persist(&valid)?;
        assert!(storage.persistence_watermarks()?.admitted.is_some());
        storage.close()?;
        Ok(())
    }

    #[test]
    fn ancestry_checks_cover_staged_and_persisted_arrivals_without_moving_watermarks()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut storage = Storage::memory()?;
        let mut root_batch = sample_batch(200, 20.0);
        root_batch.summary.births = 0;
        root_batch.births = vec![sample_birth(200, 80, BirthOrigin::Injected)];
        synchronize_lifecycle_counts(&mut root_batch);
        storage.persist(&root_batch)?;
        let staged_watermarks = storage.persistence_watermarks()?;

        let mut same_tick_child = sample_batch(201, 20.1);
        let mut child = sample_birth(200, 81, BirthOrigin::Born);
        child.parent_a = Some(AgentUid(80));
        same_tick_child.births = vec![child];
        synchronize_lifecycle_counts(&mut same_tick_child);
        assert_invalid_data_context(storage.persist(&same_tick_child), "births.parent_a");
        assert_eq!(storage.persistence_watermarks()?, staged_watermarks);

        storage.flush()?;
        let persisted_watermarks = storage.persistence_watermarks()?;
        let mut same_tick_death = sample_batch(202, 20.2);
        same_tick_death.summary.births = 0;
        same_tick_death.summary.deaths = 1;
        same_tick_death.deaths = vec![sample_death(200, 80, DeathCause::Unknown)];
        synchronize_lifecycle_counts(&mut same_tick_death);
        assert_invalid_data_context(storage.persist(&same_tick_death), "deaths.tick");
        assert_eq!(storage.persistence_watermarks()?, persisted_watermarks);

        let ledger_count: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM storage_batch_ledger")?
            .get_typed(0)?;
        assert_eq!(ledger_count, 1);

        let mut valid = sample_batch(202, 20.2);
        valid.summary.deaths = 1;
        let mut child = sample_birth(201, 81, BirthOrigin::Born);
        child.parent_a = Some(AgentUid(80));
        valid.births = vec![child];
        valid.deaths = vec![sample_death(201, 80, DeathCause::Unknown)];
        synchronize_lifecycle_counts(&mut valid);
        storage.persist(&valid)?;
        assert_ne!(storage.persistence_watermarks()?, persisted_watermarks);
        storage.close()?;
        Ok(())
    }

    #[test]
    fn negative_persisted_arrival_tick_is_rejected_before_new_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut storage = Storage::memory()?;
        let mut root_batch = sample_batch(300, 30.0);
        root_batch.summary.births = 0;
        root_batch.births = vec![sample_birth(300, 90, BirthOrigin::Injected)];
        synchronize_lifecycle_counts(&mut root_batch);
        storage.persist(&root_batch)?;
        storage.flush()?;
        let before = storage.persistence_watermarks()?;

        // Rebuild only this table without its CHECK constraints to model a
        // corrupted persisted row. The writer remains live so the next
        // admission must validate the decoded database value, not rely on the
        // schema having prevented corruption in the first place.
        storage.connection()?.execute(
            "CREATE TABLE births_unchecked AS SELECT * FROM births;
             DROP TABLE births;
             ALTER TABLE births_unchecked RENAME TO births;
             UPDATE births SET tick = -1 WHERE agent_uid = 90;",
        )?;

        let mut child_batch = sample_batch(301, 30.1);
        let mut child = sample_birth(301, 91, BirthOrigin::Born);
        child.parent_a = Some(AgentUid(90));
        child_batch.births = vec![child];
        synchronize_lifecycle_counts(&mut child_batch);
        assert_invalid_data_context(storage.persist(&child_batch), "births.tick");
        assert_eq!(storage.persistence_watermarks()?, before);
        assert!(storage.buffer.is_empty());
        let ledger_count: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM storage_batch_ledger")?
            .get_typed(0)?;
        assert_eq!(ledger_count, 1);
        storage.close()?;
        Ok(())
    }

    #[test]
    fn duplicate_birth_identities_are_rejected_before_new_outbox_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let make_birth = |tick: u64, uid: u64, spawn: u64, birth: u64| {
            let mut record = sample_birth(tick, uid, BirthOrigin::Born);
            record.spawn_ordinal = spawn;
            record.birth_ordinal = Some(birth);
            record
        };

        for (context, births) in [
            (
                "births.agent_uid",
                vec![make_birth(55, 40, 50, 60), make_birth(55, 40, 51, 61)],
            ),
            (
                "births.spawn_ordinal",
                vec![make_birth(55, 41, 52, 62), make_birth(55, 42, 52, 63)],
            ),
            (
                "births.birth_ordinal",
                vec![make_birth(55, 43, 53, 64), make_birth(55, 44, 54, 64)],
            ),
        ] {
            let mut duplicate_batch = sample_batch(55, 5.5);
            duplicate_batch.births = births;
            synchronize_lifecycle_counts(&mut duplicate_batch);
            assert_invalid_data_context(
                PreparedPersistenceBatch::from_batch(&duplicate_batch),
                context,
            );
        }

        let mut storage = Storage::memory()?;
        let mut first = sample_batch(70, 7.0);
        first.births = vec![make_birth(70, 7, 20, 30)];
        synchronize_lifecycle_counts(&mut first);
        storage.persist(&first)?;
        let admitted_once = storage.persistence_watermarks()?;

        for (tick, context, conflicting) in [
            (71, "births.agent_uid", make_birth(71, 7, 21, 31)),
            (72, "births.spawn_ordinal", make_birth(72, 8, 20, 32)),
            (73, "births.birth_ordinal", make_birth(73, 9, 22, 30)),
        ] {
            let mut batch = sample_batch(tick, 7.1);
            batch.births = vec![conflicting];
            synchronize_lifecycle_counts(&mut batch);
            assert_invalid_data_context(storage.persist(&batch), context);
            assert_eq!(storage.persistence_watermarks()?, admitted_once);
        }

        storage.flush()?;
        let durable_once = storage.persistence_watermarks()?;
        for (tick, context, conflicting) in [
            (81, "births.agent_uid", make_birth(81, 7, 23, 33)),
            (82, "births.spawn_ordinal", make_birth(82, 10, 20, 34)),
            (83, "births.birth_ordinal", make_birth(83, 11, 24, 30)),
        ] {
            let mut batch = sample_batch(tick, 8.1);
            batch.births = vec![conflicting];
            synchronize_lifecycle_counts(&mut batch);
            assert_invalid_data_context(storage.persist(&batch), context);
            assert_eq!(storage.persistence_watermarks()?, durable_once);
        }

        let admitted_batches: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM storage_batch_ledger")?
            .get_typed(0)?;
        assert_eq!(
            admitted_batches, 1,
            "duplicate birth identities leaked rejected batches into the admission ledger"
        );
        storage.close()?;
        Ok(())
    }

    #[test]
    fn duplicate_death_uids_are_rejected_before_new_outbox_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut storage = Storage::memory()?;
        let mut first = sample_batch(60, 6.0);
        first.births.push(sample_birth(59, 7, BirthOrigin::Born));
        first
            .deaths
            .push(sample_death(60, 7, DeathCause::Starvation));
        synchronize_lifecycle_counts(&mut first);
        storage.persist(&first)?;
        let admitted_once = storage.persistence_watermarks()?;

        let mut duplicate_staged = sample_batch(61, 6.1);
        duplicate_staged
            .deaths
            .push(sample_death(61, 7, DeathCause::Aging));
        synchronize_lifecycle_counts(&mut duplicate_staged);
        assert_invalid_data_context(storage.persist(&duplicate_staged), "deaths.agent_uid");
        assert_eq!(
            storage.persistence_watermarks()?,
            admitted_once,
            "a death conflicting with a staged row received an outbox identity"
        );

        storage.flush()?;
        let durable_once = storage.persistence_watermarks()?;
        let mut duplicate_persisted = sample_batch(62, 6.2);
        duplicate_persisted
            .deaths
            .push(sample_death(62, 7, DeathCause::CombatCarnivore));
        synchronize_lifecycle_counts(&mut duplicate_persisted);
        assert_invalid_data_context(storage.persist(&duplicate_persisted), "deaths.agent_uid");
        assert_eq!(
            storage.persistence_watermarks()?,
            durable_once,
            "a death conflicting with a persisted row received an outbox identity"
        );

        let mut same_batch = sample_batch(63, 6.3);
        same_batch.deaths = vec![
            sample_death(63, 8, DeathCause::Starvation),
            sample_death(63, 8, DeathCause::Aging),
        ];
        synchronize_lifecycle_counts(&mut same_batch);
        assert_invalid_data_context(
            PreparedPersistenceBatch::from_batch(&same_batch),
            "deaths.agent_uid",
        );
        assert_invalid_data_context(storage.persist(&same_batch), "deaths.agent_uid");
        assert_eq!(storage.persistence_watermarks()?, durable_once);

        let persisted: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM deaths WHERE agent_uid = 7")?
            .get_typed(0)?;
        assert_eq!(persisted, 1);
        let admitted_batches: i64 = storage
            .connection()?
            .query_row("SELECT COUNT(*) FROM storage_batch_ledger")?
            .get_typed(0)?;
        assert_eq!(
            admitted_batches, 1,
            "duplicate deaths leaked rejected batches into the admission ledger"
        );
        storage.close()?;
        Ok(())
    }

    #[test]
    fn run_ledger_counts_only_demographic_births() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-birth-origin-ledger");
        let path_string = path.to_string_lossy().to_string();
        let mut storage = Storage::create_new_file_with_thresholds(&path_string, 1, 1, 1, 1)?;
        let mut batch = sample_batch(11, 1.0);
        let base = BirthRecord {
            tick: Tick(11),
            agent_uid: AgentUid(2),
            spawn_ordinal: 1,
            birth_ordinal: Some(0),
            parent_a: None,
            parent_b: None,
            brain_kind: Some("origin-ledger".to_owned()),
            brain_key: None,
            herbivore_tendency: 0.5,
            generation: Generation(0),
            position: Position::new(1.0, 2.0),
            is_hybrid: false,
            origin: BirthOrigin::Born,
        };
        batch.births = vec![
            BirthRecord {
                tick: Tick::zero(),
                agent_uid: AgentUid(1),
                spawn_ordinal: 0,
                birth_ordinal: None,
                origin: BirthOrigin::Seeded,
                ..base.clone()
            },
            base.clone(),
            BirthRecord {
                agent_uid: AgentUid(3),
                spawn_ordinal: 2,
                birth_ordinal: None,
                origin: BirthOrigin::Injected,
                ..base
            },
        ];
        synchronize_lifecycle_counts(&mut batch);
        storage.persist(&batch)?;
        storage.flush()?;
        storage.close()?;

        let reader = StorageReader::open(&path_string)?;
        let ledger = reader.run_ledger_summary()?;
        assert_eq!(ledger.birth_records, 1);
        assert_eq!(reader.load_ancestry_births()?.len(), 3);
        reader.close()?;
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
        let agent_uid = batch.agents[0].identity.uid;
        batch.births.push(BirthRecord {
            // Lifecycle rows can trail the enclosing persistence cadence. Keep
            // this arrival strictly before the death below so the fixture is a
            // valid ancestry log as well as a schema-constraint probe.
            tick: Tick(10),
            agent_uid,
            spawn_ordinal: 0,
            birth_ordinal: Some(0),
            parent_a: None,
            parent_b: None,
            brain_kind: Some("schema-test".to_owned()),
            brain_key: Some(7),
            herbivore_tendency: 0.5,
            generation: scriptbots_core::Generation(0),
            position: Position::new(1.0, 2.0),
            is_hybrid: false,
            origin: BirthOrigin::Born,
        });
        batch.deaths.push(DeathRecord {
            tick: Tick(11),
            agent_uid,
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
        synchronize_lifecycle_counts(&mut batch);
        batch.replay_events.push(ReplayEvent {
            agent_uid: None,
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
                "replay_events.agent_uid",
                "UPDATE replay_events SET agent_uid = -1 WHERE tick = 11",
            ),
            ("agents.tick", "UPDATE agents SET tick = -1 WHERE tick = 11"),
            (
                "agents.agent_uid",
                "UPDATE agents SET agent_uid = -1 WHERE tick = 11",
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
            ("births.tick", "UPDATE births SET tick = -1 WHERE tick = 10"),
            (
                "births.agent_uid",
                "UPDATE births SET agent_uid = -1 WHERE tick = 10",
            ),
            (
                "births.spawn_ordinal",
                "UPDATE births SET spawn_ordinal = -1 WHERE tick = 10",
            ),
            (
                "births.birth_ordinal",
                "UPDATE births SET birth_ordinal = -1 WHERE tick = 10",
            ),
            (
                "births.parent_a",
                "UPDATE births SET parent_a = -1 WHERE tick = 10",
            ),
            (
                "births.parent_b",
                "UPDATE births SET parent_b = -1 WHERE tick = 10",
            ),
            (
                "births.brain_key",
                "UPDATE births SET brain_key = -1 WHERE tick = 10",
            ),
            (
                "births.generation",
                "UPDATE births SET generation = -1 WHERE tick = 10",
            ),
            ("deaths.tick", "UPDATE deaths SET tick = -1 WHERE tick = 11"),
            (
                "deaths.agent_uid",
                "UPDATE deaths SET agent_uid = -1 WHERE tick = 11",
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
        assert!(
            storage
                .connection()?
                .execute("UPDATE births SET origin = 'unknown' WHERE tick = 10")
                .is_err(),
            "births.origin CHECK must reject values outside the typed domain"
        );

        storage.connection()?.execute(
            "INSERT INTO ticks (
                tick, epoch, closed, agent_count, births, deaths,
                total_energy, average_energy, average_health
             ) VALUES (3, 'invalid-epoch', 0, 0, 0, 0, 0.0, 0.0, 0.0)",
        )?;
        storage
            .connection()?
            .execute("UPDATE deaths SET cause = 'not-a-cause' WHERE tick = 11")?;
        storage.close()?;

        let reader = StorageReader::open(&path_string)?;
        let cause_error = reader
            .load_ancestry_deaths()
            .expect_err("typed ancestry reader must reject an unknown death cause");
        assert!(matches!(
            cause_error,
            StorageError::InvalidData {
                context: "deaths.cause",
                ..
            }
        ));
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
    fn birth_origin_decode_is_fail_closed() {
        assert!(matches!(decode_birth_origin("born"), Ok(BirthOrigin::Born)));
        assert!(matches!(
            decode_birth_origin("seeded"),
            Ok(BirthOrigin::Seeded)
        ));
        assert!(matches!(
            decode_birth_origin("injected"),
            Ok(BirthOrigin::Injected)
        ));
        assert!(matches!(
            decode_birth_origin("unknown"),
            Err(StorageError::InvalidData {
                context: "births.origin",
                ..
            })
        ));
    }

    #[test]
    fn ancestry_death_decode_and_rebuild_preserve_non_starvation_cause() {
        for (encoded, expected) in [
            ("combat_carnivore", DeathCause::CombatCarnivore),
            ("combat_herbivore", DeathCause::CombatHerbivore),
            ("starvation", DeathCause::Starvation),
            ("aging", DeathCause::Aging),
            ("unknown", DeathCause::Unknown),
        ] {
            assert!(matches!(decode_death_cause(encoded), Ok(actual) if actual == expected));
        }
        assert!(matches!(
            decode_death_cause("not-a-cause"),
            Err(StorageError::InvalidData {
                context: "deaths.cause",
                ..
            })
        ));

        let births = [PersistedAncestryBirth {
            tick: Tick(0),
            agent_uid: AgentUid(1),
            spawn_ordinal: 0,
            birth_ordinal: None,
            parent_a: None,
            parent_b: None,
            generation: Generation(0),
            brain_key: None,
            is_hybrid: false,
            origin: BirthOrigin::Seeded,
        }];
        let aging_deaths = [PersistedAncestryDeath {
            tick: Tick(1),
            agent_uid: AgentUid(1),
            cause: DeathCause::Aging,
        }];
        let starvation_deaths = [PersistedAncestryDeath {
            cause: DeathCause::Starvation,
            ..aging_deaths[0]
        }];

        let aging = rebuild_ancestry(&births, &aging_deaths).expect("aging rebuild");
        let starvation = rebuild_ancestry(&births, &starvation_deaths).expect("starvation rebuild");
        assert_eq!(
            aging.node(AgentUid(1)).and_then(|node| node.death_cause),
            Some(DeathCause::Aging)
        );
        assert_ne!(
            aging.canonical_digest(),
            starvation.canonical_digest(),
            "substituting Starvation for the persisted cause must change the ancestry oracle"
        );
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
            budget: PayloadBudget::default(),
            inflight_bytes: Arc::new(AtomicUsize::new(0)),
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
            budget: PayloadBudget::default(),
            inflight_bytes: Arc::new(AtomicUsize::new(0)),
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
            budget: PayloadBudget::default(),
            inflight_bytes: Arc::new(AtomicUsize::new(0)),
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
    fn prepare_batch_rejects_out_of_range_agent_uid() {
        let mut batch = sample_batch(9, 1.0);
        batch.agents[0].identity.uid = AgentUid(u64::MAX);
        let error = Storage::prepare_batch(&batch)
            .expect_err("agent uid above i64::MAX must fail batch preparation");
        assert!(matches!(
            error,
            StorageError::InvalidData {
                context: "agents.agent_uid",
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
    fn prepare_batch_rejects_out_of_range_nested_replay_agent_uids() {
        let invalid_uid = AgentUid(u64::MAX);

        let mut action = sample_batch(9, 1.0);
        action.replay_events.push(ReplayEvent {
            agent_uid: None,
            kind: ReplayEventKind::Action {
                left_wheel: 0.0,
                right_wheel: 0.0,
                boost: false,
                spike_target: Some(invalid_uid),
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
            agent_uid: None,
            kind: ReplayEventKind::RngSample {
                scope: ReplayRngScope::Agent {
                    agent_uid: invalid_uid,
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
                context: "replay_events.rng_sample.scope_agent_uid",
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

        let invalid_uid = AgentUid(u64::MAX);

        let mut epoch = sample_batch(9, 1.0);
        epoch.epoch = u64::MAX;
        assert_invalid_data_context(Storage::prepare_batch(&epoch), "ticks.epoch");

        let mut agent = sample_agent(1.0);
        agent.runtime.brain = BrainBinding::inherited(Box::new(KeyedBrain), Some(u64::MAX));
        assert_invalid_data_context(agent_row_from_snapshot(9, &agent), "agents.brain_key");

        let base_birth = BirthRecord {
            tick: Tick(9),
            agent_uid: AgentUid(1),
            spawn_ordinal: 0,
            birth_ordinal: Some(0),
            parent_a: None,
            parent_b: None,
            brain_kind: Some("test.keyed".to_owned()),
            brain_key: Some(7),
            herbivore_tendency: 0.5,
            generation: scriptbots_core::Generation(1),
            position: Position::new(1.0, 2.0),
            is_hybrid: false,
            origin: BirthOrigin::Born,
        };
        let mut birth = base_birth.clone();
        birth.tick = Tick(u64::MAX);
        assert_invalid_data_context(birth_row_from_record(&birth), "births.tick");
        let mut birth = base_birth.clone();
        birth.agent_uid = invalid_uid;
        assert_invalid_data_context(birth_row_from_record(&birth), "births.agent_uid");
        let mut birth = base_birth.clone();
        birth.spawn_ordinal = u64::MAX;
        assert_invalid_data_context(birth_row_from_record(&birth), "births.spawn_ordinal");
        let mut birth = base_birth.clone();
        birth.birth_ordinal = Some(u64::MAX);
        assert_invalid_data_context(birth_row_from_record(&birth), "births.birth_ordinal");
        let mut birth = base_birth.clone();
        birth.parent_a = Some(invalid_uid);
        assert_invalid_data_context(birth_row_from_record(&birth), "births.parent_a");
        let mut birth = base_birth.clone();
        birth.parent_b = Some(invalid_uid);
        assert_invalid_data_context(birth_row_from_record(&birth), "births.parent_b");
        let mut birth = base_birth;
        birth.brain_key = Some(u64::MAX);
        assert_invalid_data_context(birth_row_from_record(&birth), "births.brain_key");

        let base_death = DeathRecord {
            tick: Tick(9),
            agent_uid: AgentUid(1),
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
        death.agent_uid = invalid_uid;
        assert_invalid_data_context(death_row_from_record(&death), "deaths.agent_uid");
        let mut death = base_death;
        death.brain_key = Some(u64::MAX);
        assert_invalid_data_context(death_row_from_record(&death), "deaths.brain_key");

        let replay = ReplayEvent {
            agent_uid: Some(invalid_uid),
            kind: ReplayEventKind::BrainOutputs {
                outputs: vec![0.25],
            },
        };
        assert_invalid_data_context(
            replay_row_from_event(&replay, 9, 0),
            "replay_events.agent_uid",
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn replay_sequence_above_sql_integer_range_is_rejected() {
        let replay = ReplayEvent {
            agent_uid: None,
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
        bad.agents[0].identity.uid = AgentUid(u64::MAX);

        let error = pipeline
            .submit(&bad)
            .expect_err("out-of-range agent uid must be rejected at admission");
        assert!(matches!(
            error,
            StorageError::InvalidData {
                context: "agents.agent_uid",
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
