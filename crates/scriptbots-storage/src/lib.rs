//! DuckDB-backed persistence layer for ScriptBots.

use duckdb::{Connection, Transaction, params};
use scriptbots_core::{
    AgentState, BrainBinding, PersistenceBatch, PersistenceEventKind, WorldPersistence,
};
use slotmap::Key;
use std::sync::{Arc, Mutex};
use thiserror::Error;

const DEFAULT_TICK_BUFFER: usize = 32;
const DEFAULT_AGENT_BUFFER: usize = 1024;
const DEFAULT_EVENT_BUFFER: usize = 256;
const DEFAULT_METRIC_BUFFER: usize = 256;

/// Storage error wrapper.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("duckdb error: {0}")]
    DuckDb(#[from] duckdb::Error),
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
    food_delta: f64,
    spiked: bool,
    hybrid: bool,
    sound_output: f64,
}

#[derive(Default)]
struct StorageBuffer {
    ticks: Vec<TickRow>,
    metrics: Vec<MetricRow>,
    events: Vec<EventRow>,
    agents: Vec<AgentRow>,
}

impl StorageBuffer {
    fn is_empty(&self) -> bool {
        self.ticks.is_empty()
            && self.metrics.is_empty()
            && self.events.is_empty()
            && self.agents.is_empty()
    }

    fn clear(&mut self) {
        self.ticks.clear();
        self.metrics.clear();
        self.events.clear();
        self.agents.clear();
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
                food_delta double,
                spiked boolean,
                hybrid boolean,
                sound_output double,
                primary key (tick, agent_id)
            )",
            [],
        )?;
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
        let mut stmt = tx.prepare(
            "insert or replace into agents (
                tick, agent_id, generation, age,
                position_x, position_y,
                velocity_x, velocity_y,
                heading, health, energy,
                color_r, color_g, color_b,
                spike_length, boost,
                herbivore_tendency, sound_multiplier, reproduction_counter,
                mutation_rate_primary, mutation_rate_secondary,
                trait_smell, trait_sound, trait_hearing, trait_eye, trait_blood,
                give_intent, brain_binding,
                food_delta, spiked, hybrid, sound_output
            ) values (
                ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?
            )",
        )?;
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
                row.food_delta,
                row.spiked,
                row.hybrid,
                row.sound_output,
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
        tx.commit()?;
        self.buffer.clear();
        Ok(())
    }

    /// Run database maintenance to optimize and compact storage.
    pub fn optimize(&mut self) -> Result<(), StorageError> {
        self.flush()?;
        self.conn.execute("PRAGMA optimize_database;", [])?;
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

/// Shareable wrapper around `Storage` for use in persistence and UI layers.
#[derive(Clone)]
pub struct SharedStorage {
    inner: Arc<Mutex<Storage>>,
}

impl SharedStorage {
    /// Create a new shared handle from an `Arc<Mutex<Storage>>`.
    #[must_use]
    pub fn new(inner: Arc<Mutex<Storage>>) -> Self {
        Self { inner }
    }

    /// Borrow the underlying shared storage arc.
    #[must_use]
    pub fn inner(&self) -> Arc<Mutex<Storage>> {
        Arc::clone(&self.inner)
    }
}

impl WorldPersistence for SharedStorage {
    fn on_tick(&mut self, payload: &PersistenceBatch) {
        match self.inner.lock() {
            Ok(mut storage) => {
                if let Err(err) = storage.persist(payload) {
                    eprintln!(
                        "failed to enqueue persistence data for tick {}: {err}",
                        payload.summary.tick.0
                    );
                }
            }
            Err(poisoned) => {
                eprintln!(
                    "storage mutex poisoned during tick {}: attempting recovery",
                    payload.summary.tick.0
                );
                let mut storage = poisoned.into_inner();
                if let Err(err) = storage.persist(payload) {
                    eprintln!(
                        "failed to enqueue persistence data for tick {} after poison: {err}",
                        payload.summary.tick.0
                    );
                }
            }
        }
    }
}

fn brain_binding_to_string(binding: &BrainBinding) -> String {
    match binding {
        BrainBinding::Unbound => "unbound".to_string(),
        BrainBinding::Registry { key } => format!("registry:{key}"),
    }
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
        food_delta: f64::from(runtime.food_delta),
        spiked: runtime.spiked,
        hybrid: runtime.hybrid,
        sound_output: f64::from(runtime.sound_output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{AgentData, ScriptBotsConfig, WorldState};
    use std::fs;
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Default)]
    struct CapturePersistence {
        batches: Arc<Mutex<Vec<PersistenceBatch>>>,
    }

    impl WorldPersistence for CapturePersistence {
        fn on_tick(&mut self, payload: &PersistenceBatch) {
            self.batches.lock().unwrap().push(payload.clone());
        }
    }

    #[test]
    fn storage_flushes_persistence_batch() {
        let config = ScriptBotsConfig {
            persistence_interval: 1,
            history_capacity: 4,
            ..ScriptBotsConfig::default()
        };
        let capture = CapturePersistence::default();
        let batches = capture.batches.clone();
        let mut world =
            WorldState::with_persistence(config, Box::new(capture)).expect("world creation");
        world.spawn_agent(AgentData::default());
        world.step();

        let batch = {
            let guard = batches.lock().unwrap();
            guard.first().cloned().expect("captured batch")
        };

        let filename = format!(
            "scriptbots-storage-test-{}-{}.duckdb",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        );
        let path = std::env::temp_dir().join(filename);
        let path_str = path.to_string_lossy().to_string();

        let mut storage = Storage::with_thresholds(&path_str, 1, 1, 1, 1).expect("storage");
        storage.enqueue(&batch).expect("enqueue");
        storage.flush().expect("flush");

        let tick_count: i64 = storage
            .conn
            .query_row("select count(*) from ticks", [], |row| row.get(0))
            .expect("tick count");
        assert_eq!(tick_count, 1);

        let agent_count: i64 = storage
            .conn
            .query_row("select count(*) from agents", [], |row| row.get(0))
            .expect("agent count");
        assert_eq!(agent_count, batch.agents.len() as i64);

        let metric_count: i64 = storage
            .conn
            .query_row("select count(*) from metrics", [], |row| row.get(0))
            .expect("metric count");
        assert_eq!(metric_count, batch.metrics.len() as i64);

        let _ = fs::remove_file(path);
    }
}
