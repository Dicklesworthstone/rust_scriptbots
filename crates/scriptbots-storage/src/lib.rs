//! DuckDB-backed persistence layer for ScriptBots.

#[cfg(target_os = "windows")]
#[link(name = "rstrtmgr")]
extern "system" {}

use duckdb::{Connection, Transaction, params};
use serde_json::{json, Value};
use scriptbots_core::{
    AgentId, AgentState, BirthRecord, BrainBinding, DeathCause, DeathRecord, PersistenceBatch,
    PersistenceEventKind, ReplayAgentPhase, ReplayEvent, ReplayEventKind, ReplayRngScope,
    WorldPersistence,
};
use slotmap::Key;
use std::{
    sync::{Arc, Mutex, OnceLock, mpsc},
    thread,
};
use thiserror::Error;

const DEFAULT_TICK_BUFFER: usize = 32;
const DEFAULT_AGENT_BUFFER: usize = 1024;
const DEFAULT_EVENT_BUFFER: usize = 256;
const DEFAULT_METRIC_BUFFER: usize = 256;
const DEFAULT_LIFECYCLE_BUFFER: usize = 512;
const DEFAULT_REPLAY_BUFFER: usize = 1024;

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
    #[error("duckdb error: {0}")]
    DuckDb(#[from] duckdb::Error),
    #[error("storage worker error: {0}")]
    Worker(String),
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
#[derive(Debug, Clone)]
pub struct MetricReading {
    pub tick: i64,
    pub name: String,
    pub value: f64,
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

#[derive(Default)]
struct StorageBuffer {
    ticks: Vec<TickRow>,
    metrics: Vec<MetricRow>,
    events: Vec<EventRow>,
    agents: Vec<AgentRow>,
    births: Vec<BirthRow>,
    deaths: Vec<DeathRow>,
    replay_events: Vec<ReplayEventRow>,
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
}

/// DuckDB-backed persistence sink with buffered writes.
pub struct Storage {
    conn: Connection,
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
    /// Open or create a DuckDB database at the provided path with default buffering thresholds.
    pub fn open(path: &str) -> Result<Self, StorageError> {
        let conn = Connection::open(path)?;
        let mut storage = Self {
            conn,
            buffer: StorageBuffer::default(),
            tick_flush_threshold: DEFAULT_TICK_BUFFER,
            agent_flush_threshold: DEFAULT_AGENT_BUFFER,
            event_flush_threshold: DEFAULT_EVENT_BUFFER,
            metric_flush_threshold: DEFAULT_METRIC_BUFFER,
            birth_flush_threshold: DEFAULT_LIFECYCLE_BUFFER,
            death_flush_threshold: DEFAULT_LIFECYCLE_BUFFER,
            replay_flush_threshold: DEFAULT_REPLAY_BUFFER,
        };
        storage.initialize_schema()?;
        Ok(storage)
    }

    /// Override flush thresholds for ticks, agents, events, and metrics respectively.
    #[allow(dead_code)]
    pub fn with_thresholds(
        path: &str,
        tick: usize,
        agent: usize,
        event: usize,
        metric: usize,
    ) -> Result<Self, StorageError> {
        let conn = Connection::open(path)?;
        let mut storage = Self {
            conn,
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
        self.conn.execute(
            "create table if not exists ticks (
                tick bigint primary key,
                epoch bigint,
                closed boolean,
                agent_count integer,
                births integer,
                deaths integer,
                total_energy double,
                average_energy double,
                average_health double
            )",
            [],
        )?;
        self.conn.execute(
            "create table if not exists metrics (
                tick bigint,
                name text,
                value double,
                primary key (tick, name)
            )",
            [],
        )?;
        self.conn.execute(
            "create table if not exists events (
                tick bigint,
                kind text,
                count integer,
                primary key (tick, kind)
            )",
            [],
        )?;
        self.conn.execute(
            "create table if not exists replay_events (
                tick bigint,
                seq bigint,
                agent_id bigint,
                scope text,
                event_type text,
                payload json,
                primary key (tick, seq)
            )",
            [],
        )?;
        self.conn.execute(
            "create table if not exists agents (
                tick bigint,
                agent_id bigint,
                generation integer,
                age integer,
                position_x double,
                position_y double,
                velocity_x double,
                velocity_y double,
                heading double,
                health double,
                energy double,
                color_r double,
                color_g double,
                color_b double,
                spike_length double,
                boost boolean,
                herbivore_tendency double,
                sound_multiplier double,
                reproduction_counter double,
                mutation_rate_primary double,
                mutation_rate_secondary double,
                trait_smell double,
                trait_sound double,
                trait_hearing double,
                trait_eye double,
                trait_blood double,
                give_intent double,
                brain_binding text,
                brain_key bigint,
                food_delta double,
                spiked boolean,
                hybrid boolean,
                sound_output double,
                spike_attacker boolean,
                spike_victim boolean,
                hit_carnivore boolean,
                hit_herbivore boolean,
                hit_by_carnivore boolean,
                hit_by_herbivore boolean,
                primary key (tick, agent_id)
            )",
            [],
        )?;
        let _ = self
            .conn
            .execute("alter table agents add column brain_key bigint", []);
        self.conn.execute(
            "create table if not exists births (
                tick bigint,
                agent_id bigint,
                parent_a bigint,
                parent_b bigint,
                brain_kind text,
                brain_key bigint,
                herbivore_tendency double,
                generation integer,
                position_x double,
                position_y double,
                is_hybrid boolean,
                primary key (tick, agent_id)
            )",
            [],
        )?;
        self.conn.execute(
            "create table if not exists deaths (
                tick bigint,
                agent_id bigint,
                age integer,
                generation integer,
                herbivore_tendency double,
                brain_kind text,
                brain_key bigint,
                energy double,
                food_balance_total double,
                cause text,
                was_hybrid boolean,
                spike_attacker boolean,
                spike_victim boolean,
                hit_carnivore boolean,
                hit_herbivore boolean,
                hit_by_carnivore boolean,
                hit_by_herbivore boolean,
                primary key (tick, agent_id)
            )",
            [],
        )?;
        let _ = self
            .conn
            .execute("alter table births add column brain_key bigint", []);
        let _ = self
            .conn
            .execute("alter table births add column is_hybrid boolean", []);
        let _ = self.conn.execute(
            "alter table deaths add column food_balance_total double",
            [],
        );
        let _ = self
            .conn
            .execute("alter table deaths add column spike_attacker boolean", []);
        let _ = self
            .conn
            .execute("alter table deaths add column spike_victim boolean", []);
        let _ = self
            .conn
            .execute("alter table deaths add column hit_carnivore boolean", []);
        let _ = self
            .conn
            .execute("alter table deaths add column hit_herbivore boolean", []);
        let _ = self
            .conn
            .execute("alter table deaths add column hit_by_carnivore boolean", []);
        let _ = self
            .conn
            .execute("alter table deaths add column hit_by_herbivore boolean", []);
        Ok(())
    }

    fn enqueue(&mut self, payload: &PersistenceBatch) -> Result<(), StorageError> {
        let summary = &payload.summary;
        let tick = summary.tick.0 as i64;

        self.buffer.ticks.push(TickRow {
            tick,
            epoch: payload.epoch as i64,
            closed: payload.closed,
            agent_count: summary.agent_count as i64,
            births: summary.births as i64,
            deaths: summary.deaths as i64,
            total_energy: f64::from(summary.total_energy),
            average_energy: f64::from(summary.average_energy),
            average_health: f64::from(summary.average_health),
        });

        for metric in &payload.metrics {
            self.buffer.metrics.push(MetricRow {
                tick,
                name: metric.name.to_string(),
                value: metric.value,
            });
        }

        for event in &payload.events {
            self.buffer.events.push(EventRow {
                tick,
                kind: match &event.kind {
                    PersistenceEventKind::Births => "births".to_string(),
                    PersistenceEventKind::Deaths => "deaths".to_string(),
                    PersistenceEventKind::Custom(name) => name.to_string(),
                },
                count: event.count as i64,
            });
        }

        for agent in &payload.agents {
            self.buffer
                .agents
                .push(agent_row_from_snapshot(tick, agent));
        }

        for birth in &payload.births {
            self.buffer.births.push(birth_row_from_record(birth));
        }

        for death in &payload.deaths {
            self.buffer.deaths.push(death_row_from_record(death));
        }

        for (seq, event) in payload.replay_events.iter().enumerate() {
            self.buffer
                .replay_events
                .push(replay_row_from_event(event, tick, seq));
        }

        self.maybe_flush()?;
        Ok(())
    }

    /// Persist a simulation payload, buffering until thresholds are met.
    pub fn persist(&mut self, payload: &PersistenceBatch) -> Result<(), StorageError> {
        self.enqueue(payload)
    }

    fn maybe_flush(&mut self) -> Result<(), StorageError> {
        if self.buffer.ticks.len() >= self.tick_flush_threshold
            || self.buffer.metrics.len() >= self.metric_flush_threshold
            || self.buffer.events.len() >= self.event_flush_threshold
            || self.buffer.agents.len() >= self.agent_flush_threshold
            || self.buffer.births.len() >= self.birth_flush_threshold
            || self.buffer.deaths.len() >= self.death_flush_threshold
            || self.buffer.replay_events.len() >= self.replay_flush_threshold
        {
            self.flush()?;
        }
        Ok(())
    }

    fn insert_ticks(tx: &Transaction<'_>, rows: &[TickRow]) -> Result<(), duckdb::Error> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut stmt = tx.prepare(
            "insert or replace into ticks (
                tick, epoch, closed, agent_count, births, deaths,
                total_energy, average_energy, average_health
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )?;
        for row in rows {
            stmt.execute(params![
                row.tick,
                row.epoch,
                row.closed,
                row.agent_count,
                row.births,
                row.deaths,
                row.total_energy,
                row.average_energy,
                row.average_health,
            ])?;
        }
        Ok(())
    }

    fn insert_metrics(tx: &Transaction<'_>, rows: &[MetricRow]) -> Result<(), duckdb::Error> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut stmt =
            tx.prepare("insert or replace into metrics (tick, name, value) values (?, ?, ?)")?;
        for row in rows {
            stmt.execute(params![row.tick, row.name, row.value])?;
        }
        Ok(())
    }

    fn insert_events(tx: &Transaction<'_>, rows: &[EventRow]) -> Result<(), duckdb::Error> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut stmt =
            tx.prepare("insert or replace into events (tick, kind, count) values (?, ?, ?)")?;
        for row in rows {
            stmt.execute(params![row.tick, row.kind, row.count])?;
        }
        Ok(())
    }

    fn insert_agents(tx: &Transaction<'_>, rows: &[AgentRow]) -> Result<(), duckdb::Error> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut stmt = tx.prepare(Self::agent_insert_sql())?;
        for row in rows {
            stmt.execute(params![
                row.tick,
                row.agent_id,
                row.generation,
                row.age,
                row.position_x,
                row.position_y,
                row.velocity_x,
                row.velocity_y,
                row.heading,
                row.health,
                row.energy,
                row.color_r,
                row.color_g,
                row.color_b,
                row.spike_length,
                row.boost,
                row.herbivore_tendency,
                row.sound_multiplier,
                row.reproduction_counter,
                row.mutation_rate_primary,
                row.mutation_rate_secondary,
                row.trait_smell,
                row.trait_sound,
                row.trait_hearing,
                row.trait_eye,
                row.trait_blood,
                row.give_intent,
                row.brain_binding,
                row.brain_key,
                row.food_delta,
                row.spiked,
                row.hybrid,
                row.sound_output,
                row.spike_attacker,
                row.spike_victim,
                row.hit_carnivore,
                row.hit_herbivore,
                row.hit_by_carnivore,
                row.hit_by_herbivore,
            ])?;
        }
        Ok(())
    }

    fn agent_insert_sql() -> &'static str {
        static SQL: OnceLock<String> = OnceLock::new();
        SQL.get_or_init(|| {
            let columns = AGENT_COLUMNS.join(", ");
            let placeholders = std::iter::repeat_n("?", AGENT_COLUMNS.len())
                .collect::<Vec<_>>()
                .join(", ");
            format!("insert or replace into agents ({columns}) values ({placeholders})")
        })
    }

    fn insert_births(tx: &Transaction<'_>, rows: &[BirthRow]) -> Result<(), duckdb::Error> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut stmt = tx.prepare(
            "insert or replace into births (
                tick, agent_id, parent_a, parent_b,
                brain_kind, brain_key, herbivore_tendency,
                generation, position_x, position_y, is_hybrid
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )?;
        for row in rows {
            stmt.execute(params![
                row.tick,
                row.agent_id,
                row.parent_a,
                row.parent_b,
                row.brain_kind.as_deref(),
                row.brain_key,
                row.herbivore_tendency,
                row.generation,
                row.position_x,
                row.position_y,
                row.is_hybrid,
            ])?;
        }
        Ok(())
    }

    fn insert_deaths(tx: &Transaction<'_>, rows: &[DeathRow]) -> Result<(), duckdb::Error> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut stmt = tx.prepare(
            "insert or replace into deaths (
                tick, agent_id, age, generation,
                herbivore_tendency, brain_kind, brain_key,
                energy, food_balance_total, cause, was_hybrid,
                spike_attacker, spike_victim, hit_carnivore, hit_herbivore,
                hit_by_carnivore, hit_by_herbivore
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )?;
        for row in rows {
            stmt.execute(params![
                row.tick,
                row.agent_id,
                row.age,
                row.generation,
                row.herbivore_tendency,
                row.brain_kind.as_deref(),
                row.brain_key,
                row.energy,
                row.food_balance_total,
                &row.cause,
                row.was_hybrid,
                row.spike_attacker,
                row.spike_victim,
                row.hit_carnivore,
                row.hit_herbivore,
                row.hit_by_carnivore,
                row.hit_by_herbivore,
            ])?;
        }
        Ok(())
    }

    fn insert_replay_events(
        tx: &Transaction<'_>,
        rows: &[ReplayEventRow],
    ) -> Result<(), duckdb::Error> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut stmt = tx.prepare(
            "insert or replace into replay_events (
                tick, seq, agent_id, scope, event_type, payload
            ) values (?, ?, ?, ?, ?, ?)",
        )?;
        for row in rows {
            stmt.execute(params![
                row.tick,
                row.seq,
                row.agent_id,
                &row.scope,
                &row.event_type,
                &row.payload,
            ])?;
        }
        Ok(())
    }

    /// Force flush buffered records to disk.
    pub fn flush(&mut self) -> Result<(), StorageError> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let tx = self.conn.transaction()?;
        Self::insert_ticks(&tx, &self.buffer.ticks)?;
        Self::insert_metrics(&tx, &self.buffer.metrics)?;
        Self::insert_events(&tx, &self.buffer.events)?;
        Self::insert_agents(&tx, &self.buffer.agents)?;
        Self::insert_births(&tx, &self.buffer.births)?;
        Self::insert_deaths(&tx, &self.buffer.deaths)?;
        Self::insert_replay_events(&tx, &self.buffer.replay_events)?;
        tx.commit()?;
        self.buffer.clear();
        Ok(())
    }

    /// Run database maintenance to optimize and compact storage.
    pub fn optimize(&mut self) -> Result<(), StorageError> {
        self.flush()?;
        self.conn.execute("PRAGMA optimize;", [])?;
        self.conn.execute("VACUUM;", [])?;
        Ok(())
    }

    /// Return agents ranked by average energy across all recorded ticks.
    pub fn top_predators(&mut self, limit: usize) -> Result<Vec<PredatorStats>, StorageError> {
        self.flush()?;
        let mut stmt = self.conn.prepare(
            "select agent_id,
                    avg(energy) as avg_energy,
                    max(spike_length) as max_spike_length,
                    max(tick) as last_tick
             from agents
             group by agent_id
             order by avg_energy desc
             limit ?",
        )?;
        let mut rows = stmt.query(params![limit as i64])?;
        let mut stats = Vec::with_capacity(limit.min(16));
        while let Some(row) = rows.next()? {
            stats.push(PredatorStats {
                agent_id: row.get::<_, i64>(0)? as u64,
                avg_energy: row.get::<_, f64>(1)?,
                max_spike_length: row.get::<_, f64>(2)?,
                last_tick: row.get::<_, i64>(3)?,
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
        let mut stmt = self.conn.prepare("select max(tick) from metrics")?;
        let mut rows = stmt.query([])?;
        let latest_tick = match rows.next()? {
            Some(row) => row.get::<_, Option<i64>>(0)?,
            None => None,
        };
        drop(rows);

        let Some(tick) = latest_tick else {
            return Ok(Vec::new());
        };

        let mut metrics_stmt = self.conn.prepare(
            "select name, value
             from metrics
             where tick = ?
             order by name asc
             limit ?",
        )?;
        let mut metrics_rows = metrics_stmt.query(params![tick, limit as i64])?;
        let mut readings = Vec::new();
        while let Some(row) = metrics_rows.next()? {
            readings.push(MetricReading {
                tick,
                name: row.get(0)?,
                value: row.get(1)?,
            });
        }
        Ok(readings)
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if let Err(err) = self.flush() {
            eprintln!("failed to flush persistence buffer on drop: {err}");
        }
    }
}

impl WorldPersistence for Storage {
    fn on_tick(&mut self, payload: &PersistenceBatch) {
        if let Err(err) = self.persist(payload) {
            eprintln!(
                "failed to enqueue persistence data for tick {}: {err}",
                payload.summary.tick.0
            );
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
    Persist(PersistenceBatch),
    Flush,
    Shutdown,
}

pub struct StoragePipeline {
    tx: mpsc::Sender<StorageCommand>,
    storage: Arc<Mutex<Storage>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl StoragePipeline {
    /// Create an asynchronous pipeline using default buffering thresholds.
    pub fn new(path: &str) -> Result<Self, StorageError> {
        Self::with_thresholds(
            path,
            DEFAULT_TICK_BUFFER,
            DEFAULT_AGENT_BUFFER,
            DEFAULT_EVENT_BUFFER,
            DEFAULT_METRIC_BUFFER,
        )
    }

    /// Create an asynchronous pipeline with explicit thresholds.
    pub fn with_thresholds(
        path: &str,
        tick: usize,
        agent: usize,
        event: usize,
        metric: usize,
    ) -> Result<Self, StorageError> {
        let storage = Storage::with_thresholds(path, tick, agent, event, metric)?;
        Self::from_storage(storage)
    }

    fn from_storage(storage: Storage) -> Result<Self, StorageError> {
        let shared = Arc::new(Mutex::new(storage));
        let (tx, rx) = mpsc::channel::<StorageCommand>();
        let worker_storage = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name("scriptbots-storage-worker".into())
            .spawn(move || {
                while let Ok(command) = rx.recv() {
                    match command {
                        StorageCommand::Persist(batch) => match worker_storage.lock() {
                            Ok(mut storage) => {
                                if let Err(err) = storage.persist(&batch) {
                                    eprintln!(
                                        "failed to persist tick {} asynchronously: {err}",
                                        batch.summary.tick.0
                                    );
                                }
                            }
                            Err(poisoned) => {
                                eprintln!(
                                    "storage mutex poisoned while persisting tick {}",
                                    batch.summary.tick.0
                                );
                                let mut storage = poisoned.into_inner();
                                if let Err(err) = storage.persist(&batch) {
                                    eprintln!(
                                        "failed to persist tick {} after poison: {err}",
                                        batch.summary.tick.0
                                    );
                                }
                            }
                        },
                        StorageCommand::Flush => {
                            if let Ok(mut storage) = worker_storage.lock()
                                && let Err(err) = storage.flush()
                            {
                                eprintln!("failed to flush storage: {err}");
                            }
                        }
                        StorageCommand::Shutdown => {
                            if let Ok(mut storage) = worker_storage.lock() {
                                let _ = storage.flush();
                            }
                            break;
                        }
                    }
                }
            })
            .map_err(|err| {
                StorageError::Worker(format!("failed to spawn storage worker thread: {err}"))
            })?;

        Ok(Self {
            tx,
            storage: shared,
            handle: Some(handle),
        })
    }

    /// Exposes shared access to the underlying storage for analytics queries.
    #[must_use]
    pub fn storage(&self) -> Arc<Mutex<Storage>> {
        Arc::clone(&self.storage)
    }

    /// Request an immediate flush of buffered records.
    pub fn flush(&self) {
        let _ = self.tx.send(StorageCommand::Flush);
    }
}

impl WorldPersistence for StoragePipeline {
    fn on_tick(&mut self, payload: &PersistenceBatch) {
        if self
            .tx
            .send(StorageCommand::Persist(payload.clone()))
            .is_err()
        {
            eprintln!(
                "storage worker channel closed; tick {} dropped",
                payload.summary.tick.0
            );
        }
    }
}

impl Drop for StoragePipeline {
    fn drop(&mut self) {
        let _ = self.tx.send(StorageCommand::Shutdown);
        if let Some(handle) = self.handle.take()
            && let Err(err) = handle.join()
        {
            eprintln!("storage worker thread panicked: {err:?}");
        }
    }
}

fn brain_binding_to_string(binding: &BrainBinding) -> String {
    binding.describe().into_owned()
}

fn agent_row_from_snapshot(tick: i64, agent: &AgentState) -> AgentRow {
    let id = agent.id.data().as_ffi() as i64;
    let data = &agent.data;
    let runtime = &agent.runtime;
    AgentRow {
        tick,
        agent_id: id,
        generation: data.generation.0 as i64,
        age: data.age as i64,
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
        brain_key: runtime.brain.registry_key().map(|key| key as i64),
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
    }
}

fn optional_agent_id(id: Option<AgentId>) -> Option<i64> {
    id.map(|agent_id| agent_id.data().as_ffi() as i64)
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

fn replay_row_from_event(event: &ReplayEvent, tick: i64, seq: usize) -> ReplayEventRow {
    let (scope, event_type, payload_value): (String, String, Value) = match &event.kind {
        ReplayEventKind::BrainOutputs { outputs } => (
            if event.agent_id.is_some() {
                "agent:brain"
            } else {
                "world:brain"
            }
            .to_string(),
            "brain_outputs".to_string(),
            json!({ "outputs": outputs }),
        ),
        ReplayEventKind::Action {
            left_wheel,
            right_wheel,
            boost,
            spike_target,
            sound_level,
            give_intent,
        } => (
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
        ),
        ReplayEventKind::RngSample {
            scope,
            range_min,
            range_max,
            value,
        } => (
            scope_label(*scope),
            "rng_sample".to_string(),
            json!({
                "range_min": range_min,
                "range_max": range_max,
                "value": value,
            }),
        ),
    };

    ReplayEventRow {
        tick,
        seq: seq as i64,
        agent_id: optional_agent_id(event.agent_id),
        scope,
        event_type,
        payload: payload_value.to_string(),
    }
}

fn birth_row_from_record(record: &BirthRecord) -> BirthRow {
    BirthRow {
        tick: record.tick.0 as i64,
        agent_id: record.agent_id.data().as_ffi() as i64,
        parent_a: optional_agent_id(record.parent_a),
        parent_b: optional_agent_id(record.parent_b),
        brain_kind: record.brain_kind.clone(),
        brain_key: record.brain_key.map(|key| key as i64),
        herbivore_tendency: f64::from(record.herbivore_tendency),
        generation: record.generation.0 as i64,
        position_x: f64::from(record.position.x),
        position_y: f64::from(record.position.y),
        is_hybrid: record.is_hybrid,
    }
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

fn death_row_from_record(record: &DeathRecord) -> DeathRow {
    DeathRow {
        tick: record.tick.0 as i64,
        agent_id: record.agent_id.data().as_ffi() as i64,
        age: record.age as i64,
        generation: record.generation.0 as i64,
        herbivore_tendency: f64::from(record.herbivore_tendency),
        brain_kind: record.brain_kind.clone(),
        brain_key: record.brain_key.map(|key| key as i64),
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
    }
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
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temp_db_path(prefix: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        path.push(format!(
            "{}-{}-{}.duckdb",
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

    #[test]
    fn persist_batch_writes_all_tables() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-persist");
        let path_string = path.to_string_lossy().to_string();
        let mut storage = Storage::with_thresholds(&path_string, 1, 1, 1, 1)?;

        let batch = sample_batch(42, 5.5);
        storage.persist(&batch)?;
        storage.flush()?;

        let tick_count: i64 = storage
            .conn
            .query_row("select count(*) from ticks", [], |row| row.get(0))?;
        assert_eq!(tick_count, 1);

        let metric_count: i64 =
            storage
                .conn
                .query_row("select count(*) from metrics", [], |row| row.get(0))?;
        assert_eq!(metric_count, batch.metrics.len() as i64);

        let event_count: i64 =
            storage
                .conn
                .query_row("select count(*) from events", [], |row| row.get(0))?;
        assert_eq!(event_count, batch.events.len() as i64);

        let agent_count: i64 =
            storage
                .conn
                .query_row("select count(*) from agents", [], |row| row.get(0))?;
        assert_eq!(agent_count, batch.agents.len() as i64);

        let latest = storage.latest_metrics(8)?;
        assert_eq!(latest.len(), batch.metrics.len());
        assert!(latest.iter().all(|m| m.tick == 42));

        drop(storage);
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn top_predators_tracks_average_energy() -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_db_path("storage-predators");
        let path_string = path.to_string_lossy().to_string();
        let mut storage = Storage::with_thresholds(&path_string, 1, 1, 1, 1)?;

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

        drop(storage);
        let _ = fs::remove_file(path);
        Ok(())
    }
}
