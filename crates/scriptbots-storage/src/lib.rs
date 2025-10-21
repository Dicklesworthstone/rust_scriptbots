//! DuckDB-backed persistence layer for ScriptBots.

use duckdb::{params, Connection};
use scriptbots_core::Tick;
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
            "create table if not exists tick_metrics (tick bigint primary key, agents integer)",
            [],
        )?;
        Ok(Self { conn })
    }

    /// Persist aggregated metrics for a tick.
    pub fn record_tick(&self, tick: Tick, agent_count: usize) -> Result<(), StorageError> {
        self.conn
            .execute("insert or replace into tick_metrics values (?, ?)", params![tick.0, agent_count as i64])?;
        Ok(())
    }
}
