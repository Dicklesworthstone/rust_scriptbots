//! DuckDB-backed persistence layer for ScriptBots.

use duckdb::{Connection, params};
use scriptbots_core::{TickSummary, WorldPersistence};
use thiserror::Error;

/// Storage error wrapper.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("duckdb error: {0}")]
    DuckDb(#[from] duckdb::Error),
}

/// Thin wrapper that owns a DuckDB connection for logging metrics.
pub struct Storage {
    conn: Connection,
}

impl Storage {
    /// Open or create a DuckDB database at the provided path.
    pub fn open(path: &str) -> Result<Self, StorageError> {
        let conn = Connection::open(path)?;
        conn.execute(
            "create table if not exists tick_metrics (
                tick bigint primary key,
                agents integer,
                births integer,
                deaths integer,
                total_energy double,
                average_energy double,
                average_health double
            )",
            [],
        )?;
        Ok(Self { conn })
    }

    /// Persist aggregated metrics for a tick.
    pub fn record_tick(&self, summary: &TickSummary) -> Result<(), StorageError> {
        self.conn.execute(
            "insert or replace into tick_metrics values (?, ?, ?, ?, ?, ?, ?)",
            params![
                summary.tick.0,
                summary.agent_count as i64,
                summary.births as i64,
                summary.deaths as i64,
                summary.total_energy as f64,
                summary.average_energy as f64,
                summary.average_health as f64,
            ],
        )?;
        Ok(())
    }
}

impl WorldPersistence for Storage {
    fn on_tick(&mut self, summary: &TickSummary) {
        if let Err(err) = self.record_tick(summary) {
            eprintln!("failed to persist tick {}: {err}", summary.tick.0);
        }
    }
}
