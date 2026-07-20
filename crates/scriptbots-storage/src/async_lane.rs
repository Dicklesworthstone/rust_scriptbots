//! Deadline-carrying, cancellation-safe read lane for control-plane queries (bd-2z0.8.9.12).
//!
//! The lane owns an `fsqlite::AsyncConnection` — its dedicated worker task owns the
//! underlying `!Send` engine connection — and drives every query on a caller-scoped
//! current-thread asupersync runtime with the caller's `fsqlite_types::cx::Cx`. Cancelling
//! that context surfaces `FrankenError::Interrupt` at VDBE opcode boundaries instead of an
//! unbounded wait, and an expired budget surfaces through the same typed interrupt family
//! instead of a hang. The lane is query-only by construction: it exposes no mutating
//! statement path, while `StoragePipeline` remains the sole writer on its own thread.
//! MVCC snapshot semantics mean lane readers observe only commit boundaries, as proven by
//! `tests/durability_proofs.rs`.

use asupersync::runtime::{Runtime, RuntimeBuilder};
use fsqlite::{AsyncConnection, FrankenError, compat::RowExt};
use fsqlite_types::cx::{Budget, Cx};
use std::time::Duration;

use crate::{PersistedMetric, PersistedTick, PredatorStats, RunLedgerSummary, StorageError};

/// Build a full-capability context carrying a deadline measured from when the query starts
/// executing. `Cx::new()` remains the no-deadline convenience constructor.
#[must_use]
pub fn cx_with_deadline(deadline: Duration) -> Cx {
    Cx::with_budget(Budget::INFINITE.with_deadline(deadline))
}

/// Query-only async read lane for control-plane consumers.
pub struct AsyncReadLane {
    connection: AsyncConnection,
    run_id: String,
}

impl AsyncReadLane {
    /// Open the lane over an existing run database.
    ///
    /// The lane never executes a mutating statement; its connection is a second,
    /// independent engine connection beside the writer's, so MVCC snapshot reads observe
    /// only commit boundaries and never stall the writer behind a read. Like
    /// `StorageReader::open`, the lane refuses databases with zero or multiple runs so a
    /// control-plane consumer can never silently read the wrong run.
    pub fn open(path: &str) -> Result<Self, StorageError> {
        let connection =
            AsyncConnection::open_sync(path).map_err(|source| StorageError::InvalidData {
                context: "async_read_lane.open",
                reason: format!("failed to open async read lane at {path}: {source}"),
            })?;
        let run_id = resolve_single_run_id(&connection, path)?;
        Ok(Self { connection, run_id })
    }

    fn runtime() -> Result<Runtime, StorageError> {
        RuntimeBuilder::current_thread()
            .blocking_threads(1, 1)
            .enable_time()
            .build()
            .map_err(|error| StorageError::InvalidData {
                context: "async_read_lane.runtime",
                reason: format!("failed to build the lane runtime: {error}"),
            })
    }

    /// Latest metric rows, newest-first, mirroring `StorageReader::recent_metrics`.
    pub fn recent_metrics(
        &self,
        limit: usize,
        cx: &Cx,
    ) -> Result<Vec<PersistedMetric>, StorageError> {
        let runtime = Self::runtime()?;
        let bound = checked_lane_limit("recent_metrics.limit", limit)?;
        let run_id = self.run_id.clone();
        runtime
            .block_on(async {
                let rows = self
                    .connection
                    .query_with_params(
                        cx,
                        "SELECT tick, name, value
                         FROM metrics
                         WHERE run_id = ?1
                         ORDER BY tick DESC, name DESC
                         LIMIT ?2",
                        &[run_id.into(), bound],
                    )
                    .await?;
                rows.iter()
                    .map(|row| {
                        Ok(PersistedMetric {
                            tick: u64::try_from(row.get_typed::<i64>(0)?).map_err(|_| {
                                FrankenError::Internal("negative metric tick".into())
                            })?,
                            name: row.get_typed(1)?,
                            value: row.get_typed(2)?,
                        })
                    })
                    .collect::<Result<Vec<_>, FrankenError>>()
            })
            .map_err(Into::into)
    }

    /// Top predators by average energy, mirroring `StorageReader::top_predators`.
    pub fn top_predators(&self, limit: usize, cx: &Cx) -> Result<Vec<PredatorStats>, StorageError> {
        let runtime = Self::runtime()?;
        let bound = checked_lane_limit("top_predators.limit", limit)?;
        let run_id = self.run_id.clone();
        runtime
            .block_on(async {
                let rows = self
                    .connection
                    .query_with_params(
                        cx,
                        "SELECT agent_uid,
                                AVG(energy) AS avg_energy,
                                MAX(spike_length) AS max_spike_length,
                                MAX(tick) AS last_tick
                         FROM agents
                         WHERE run_id = ?1
                         GROUP BY agent_uid
                         ORDER BY avg_energy DESC
                         LIMIT ?2",
                        &[run_id.into(), bound],
                    )
                    .await?;
                rows.iter()
                    .map(|row| {
                        Ok(PredatorStats {
                            agent_uid: u64::try_from(row.get_typed::<i64>(0)?)
                                .map_err(|_| FrankenError::Internal("negative agent uid".into()))?,
                            avg_energy: row.get_typed(1)?,
                            max_spike_length: row.get_typed(2)?,
                            last_tick: row.get_typed(3)?,
                        })
                    })
                    .collect::<Result<Vec<_>, FrankenError>>()
            })
            .map_err(Into::into)
    }

    /// Run-level row counts, mirroring `StorageReader::run_ledger_summary`.
    pub fn run_ledger_summary(&self, cx: &Cx) -> Result<RunLedgerSummary, StorageError> {
        let runtime = Self::runtime()?;
        let run_id = self.run_id.clone();
        runtime
            .block_on(async {
                let count_row = self
                    .connection
                    .query_row_with_params(
                        cx,
                        "SELECT
                             (SELECT COUNT(*) FROM tick_summaries WHERE run_id = ?1),
                             (SELECT COUNT(*) FROM births WHERE run_id = ?1 AND origin = 'born'),
                             (SELECT COUNT(*) FROM deaths WHERE run_id = ?1),
                             (SELECT COALESCE(SUM(count), 0) FROM events
                              WHERE run_id = ?1 AND kind = 'births'),
                             (SELECT COALESCE(SUM(count), 0) FROM events
                              WHERE run_id = ?1 AND kind = 'deaths')",
                        &[run_id.clone().into()],
                    )
                    .await?;
                let latest_tick = self
                    .connection
                    .query_with_params(
                        cx,
                        "SELECT tick, epoch, closed, agent_count, births, deaths,
                                total_energy, average_energy, average_health
                         FROM tick_summaries
                         WHERE run_id = ?1
                         ORDER BY tick DESC
                         LIMIT 1",
                        &[run_id.into()],
                    )
                    .await?
                    .into_iter()
                    .next()
                    .map(|row| {
                        Ok::<_, FrankenError>(PersistedTick {
                            tick: u64::try_from(row.get_typed::<i64>(0)?)
                                .map_err(|_| FrankenError::Internal("negative tick".into()))?,
                            epoch: u64::try_from(row.get_typed::<i64>(1)?)
                                .map_err(|_| FrankenError::Internal("negative epoch".into()))?,
                            closed: row.get_typed(2)?,
                            agent_count: usize::try_from(row.get_typed::<i64>(3)?).map_err(
                                |_| FrankenError::Internal("negative agent count".into()),
                            )?,
                            births: usize::try_from(row.get_typed::<i64>(4)?)
                                .map_err(|_| FrankenError::Internal("negative births".into()))?,
                            deaths: usize::try_from(row.get_typed::<i64>(5)?)
                                .map_err(|_| FrankenError::Internal("negative deaths".into()))?,
                            total_energy: row.get_typed(6)?,
                            average_energy: row.get_typed(7)?,
                            average_health: row.get_typed(8)?,
                        })
                    })
                    .transpose()?;
                Ok::<_, FrankenError>(RunLedgerSummary {
                    tick_count: u64::try_from(count_row.get_typed::<i64>(0)?)
                        .map_err(|_| FrankenError::Internal("negative tick count".into()))?,
                    latest_tick,
                    birth_records: u64::try_from(count_row.get_typed::<i64>(1)?)
                        .map_err(|_| FrankenError::Internal("negative birth count".into()))?,
                    death_records: u64::try_from(count_row.get_typed::<i64>(2)?)
                        .map_err(|_| FrankenError::Internal("negative death count".into()))?,
                    birth_events: u64::try_from(count_row.get_typed::<i64>(3)?)
                        .map_err(|_| FrankenError::Internal("negative birth events".into()))?,
                    death_events: u64::try_from(count_row.get_typed::<i64>(4)?)
                        .map_err(|_| FrankenError::Internal("negative death events".into()))?,
                })
            })
            .map_err(Into::into)
    }

    /// Run any read-only SQL the lane's consumer needs, with the same deadline/cancel
    /// semantics. This is the sanctioned escape hatch for control-plane queries that do not
    /// have a named lane method yet; write-capable statement kinds are not offered.
    pub fn query_rows(
        &self,
        sql: &str,
        params: &[fsqlite::SqliteValue],
        cx: &Cx,
    ) -> Result<Vec<fsqlite::Row>, StorageError> {
        let runtime = Self::runtime()?;
        runtime
            .block_on(async { self.connection.query_with_params(cx, sql, params).await })
            .map_err(Into::into)
    }

    /// Close the lane's connection with an explicit, error-checked shutdown.
    pub fn close(mut self) -> Result<(), StorageError> {
        let runtime = Self::runtime()?;
        runtime
            .block_on(async {
                let cx: Cx = Cx::with_budget(Budget::MINIMAL);
                self.connection.close(&cx).await
            })
            .map_err(Into::into)
    }
}

/// Resolve the single run recorded in a lane database, refusing zero- or multi-run files.
fn resolve_single_run_id(connection: &AsyncConnection, path: &str) -> Result<String, StorageError> {
    let runtime = AsyncReadLane::runtime()?;
    runtime
        .block_on(async {
            let cx: Cx = Cx::with_budget(Budget::INFINITE);
            connection
                .query(&cx, "SELECT run_id FROM runs ORDER BY run_id ASC")
                .await
        })
        .map_err(|source| StorageError::InvalidData {
            context: "async_read_lane.resolve_run_id",
            reason: format!("failed to list runs in {path}: {source}"),
        })
        .and_then(|rows| match rows.as_slice() {
            [row] => row
                .get_typed::<String>(0)
                .map_err(|source| StorageError::InvalidData {
                    context: "async_read_lane.resolve_run_id",
                    reason: format!("run id in {path} does not decode as text: {source}"),
                }),
            [] => Err(StorageError::InvalidData {
                context: "async_read_lane.resolve_run_id",
                reason: format!("{path} contains no registered runs"),
            }),
            [..] => Err(StorageError::InvalidData {
                context: "async_read_lane.resolve_run_id",
                reason: format!("{path} contains multiple runs; a lane must be run-scoped"),
            }),
        })
}

fn checked_lane_limit(
    context: &'static str,
    limit: usize,
) -> Result<fsqlite::SqliteValue, StorageError> {
    if limit == 0 {
        return Err(StorageError::InvalidData {
            context,
            reason: "limit must be greater than zero".to_owned(),
        });
    }
    let bound = i64::try_from(limit).map_err(|_| StorageError::InvalidData {
        context,
        reason: format!("limit {limit} exceeds the SQL integer domain"),
    })?;
    Ok(bound.into())
}
