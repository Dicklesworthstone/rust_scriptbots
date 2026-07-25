//! Query-only read lane for control-plane queries (bd-2z0.8.9.12).
//!
//! The lane owns an `fsqlite::AsyncConnection` — its dedicated worker task owns the
//! underlying `!Send` engine connection — and drives every query on a caller-scoped
//! current-thread asupersync runtime with the caller's `fsqlite_types::cx::Cx`. It is
//! query-only by construction: it exposes no mutating statement path, while
//! `StoragePipeline` remains the sole writer on its own thread. MVCC snapshot semantics
//! mean lane readers observe only commit boundaries, as proven by
//! `tests/durability_proofs.rs`.
//!
//! # The budget on a `Cx` is NOT enforced while a statement runs (`bd-aj12`)
//!
//! This module previously documented the opposite — that cancelling the context surfaces
//! `FrankenError::Interrupt` at VDBE opcode boundaries, and that an expired budget surfaces
//! instead of a hang. Measured against the pinned engine, neither holds. A ~6.3 s self-join
//! runs to completion and returns its full result under every budget dimension:
//!
//! | budget            | elapsed | outcome              |
//! |-------------------|---------|----------------------|
//! | deadline 1 ms     | 6.33 s  | `Ok`, complete result |
//! | `poll_quota` 100  | 6.19 s  | `Ok`, complete result |
//! | `poll_quota` 1    | 6.16 s  | `Ok`, complete result |
//! | `cost_quota` 1    | 6.21 s  | `Ok`, complete result |
//!
//! `fsqlite::async_api::query_with_params` checks the context once before dispatch, then
//! hands the statement to a worker thread that holds no reference to it. Nothing can stop
//! the statement once it starts, and the pinned engine exposes no interrupt handle or
//! progress callback to build one from — so this cannot be corrected here, only upstream.
//!
//! A caller-side timeout is deliberately NOT offered as a substitute. Returning early while
//! the worker keeps scanning would leak the work rather than bound it, and would report a
//! bound this lane does not have. Until the engine can interrupt a running statement, treat
//! every lane query as unbounded and size the query accordingly: `recent_metrics`,
//! `top_predators`, and `run_ledger_summary` bound their own inputs, while `query_rows`
//! takes arbitrary SQL and is therefore the surface where an expensive query can block for
//! as long as it takes.

use asupersync::runtime::{Runtime, RuntimeBuilder};
use fsqlite::{AsyncConnection, FrankenError, compat::RowExt};
use fsqlite_types::cx::{Budget, Cx};
use std::time::Duration;

use crate::{PersistedMetric, PersistedTick, PredatorStats, RunLedgerSummary, StorageError};

/// Build a full-capability context carrying a deadline.
///
/// # This deadline is not currently enforced (`bd-aj12`)
///
/// The budget reaches the engine intact, but nothing checks it while a statement executes:
/// a 1 ms deadline against a measured ~6.3 s query returns the complete result after the
/// full 6.3 s. See the module documentation for the measurements and why a caller-side
/// timeout is not offered in its place.
///
/// The constructor is kept rather than removed because it is the right primitive the moment
/// the engine can interrupt a running statement, and `tests/async_lane.rs` already contains
/// the test that will pass unchanged when it can. Do not treat a context built here as a
/// bound on anything today.
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

    /// The most recent `limit` metric rows, ordered oldest-first, exactly as
    /// `StorageReader::recent_metrics` returns them.
    ///
    /// The `ORDER BY tick DESC, name DESC` below selects the newest window; the reversal
    /// after decoding restores ascending order. The synchronous reader does the same thing
    /// with its own `readings.reverse()`, and this lane is documented as a drop-in mirror
    /// of it for GUI, TUI, and API consumers — omitting the reversal handed those callers
    /// the same rows in the opposite direction.
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
                    .map(|mut readings| {
                        readings.reverse();
                        readings
                    })
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

    /// Tear the lane down without attempting the writer-side work it has no stake in
    /// (`bd-qan3`).
    ///
    /// This deliberately does not call `AsyncConnection::close`. That path runs
    /// `Connection::close_in_place`, i.e. `close_internal(best_effort: false,
    /// checkpoint_on_close: true)`, which attempts a passive WAL checkpoint — a *write* —
    /// on behalf of a lane that has never written anything. With a `StoragePipeline`
    /// actively committing, that checkpoint returns `Busy`, and because `best_effort` is
    /// false the `?` propagates *before* `release_connection`, so the failed close also
    /// leaves the connection unreleased. The observable symptom was a read-only lane
    /// failing to close because some unrelated connection was mid-commit.
    ///
    /// The engine already skips the checkpoint for a handle where `pager.is_readonly()`,
    /// which is what this lane morally is, but the async API offers no read-only open —
    /// `ConnectionEnv` carries no such flag — so the handle cannot be opened that way.
    ///
    /// Dropping is the correct teardown instead of a workaround. `AsyncConnection::drop`
    /// sends `Command::Shutdown` and joins the worker, and the worker's `Connection::drop`
    /// runs `close_internal(best_effort: true, checkpoint_on_close: false)` — no checkpoint
    /// to contend over, and it still reaches `release_connection`. So teardown always
    /// completes, which the checkpointing path could not guarantee.
    ///
    /// Waiting for the writer instead was considered and rejected: it would make a
    /// read-only teardown block on an unrelated writer, and a blocking close needs a bound
    /// the engine cannot enforce (`bd-aj12`), which would rebuild that hang in the teardown
    /// path. Retrying the checkpoint would be worse still — it retries work this lane
    /// should never attempt.
    ///
    /// The `Result` is retained so callers need not change and so a future read-only open
    /// can restore explicit error reporting.
    pub fn close(self) -> Result<(), StorageError> {
        drop(self);
        Ok(())
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
