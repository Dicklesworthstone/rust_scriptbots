//! Durability, retry, rollback, and concurrent-read proofs for the FrankenSQLite storage boundary.
//!
//! Bead `bd-2z0.8.9.7`. These tests complement the crate's inline unit suite with real
//! file-backed evidence gathered through the public API only:
//!
//! * a compile-time proof that `fsqlite::Connection` stays `!Send + !Sync` (no single
//!   connection is ever shared between roles or threads);
//! * engine-level MVCC commit-boundary visibility for an independent live reader;
//! * an induced real transient write conflict driving bounded, transient-only retry with
//!   eventual recovery — contrasted with a non-transient constraint failure that is never
//!   retried;
//! * a forced constraint failure committing none of a multi-statement batch, verified after
//!   close/reopen;
//! * production `Storage::flush` contention: bounded terminal failure with captured log
//!   evidence (database path, attempt, transient classification, terminal commit state),
//!   followed by exact-once outbox recovery;
//! * child-process durability: acknowledged data survives process exit and reopen;
//! * a corrupt-header startup refusal that leaves database bytes untouched;
//! * `PRAGMA integrity_check` asserted `ok` after every scenario that leaves a valid database.

use fsqlite::{
    Connection, FrankenError, SqliteValue,
    compat::{OpenFlags, RowExt, open_with_flags},
};
use scriptbots_core::{MetricSample, PersistenceBatch, Tick, TickSummary};
use scriptbots_storage::{
    FailureCommitState, PersistenceWatermarks, Storage, StorageError, StoragePipeline,
    StorageReader,
};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::{Arc, LazyLock, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

// ---------------------------------------------------------------------------
// Compile-time thread-confinement proof.
//
// `Connection` must never cross a thread boundary: the storage worker owns its connection
// on one thread by contract. If `Connection` ever implements `Send`, both blanket impls
// below apply and the associated-function reference becomes ambiguous, failing the build.
// The `#[test]` below exists only to document and reference the proof at runtime.
// ---------------------------------------------------------------------------

macro_rules! assert_not_impl_any {
    ($ty:ty: $($trait_:path),+) => {
        const _: fn() = || {
            trait AmbiguousIfImpl<A> {
                fn some_item() {}
            }
            impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
            #[allow(dead_code)]
            struct Invalid;
            $(impl<T: ?Sized + $trait_> AmbiguousIfImpl<Invalid> for T {})+
            let _ = <$ty as AmbiguousIfImpl<_>>::some_item;
        };
    };
}

assert_not_impl_any!(Connection: Send);
assert_not_impl_any!(Connection: Sync);

#[test]
fn connection_is_statically_thread_confined() {
    // The const block above is the proof: this test binary compiles only while
    // `fsqlite::Connection` is `!Send + !Sync`. Every scenario in this file therefore
    // uses an independent `Connection` per role (writer, reader, contender, child).
}

// ---------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------

fn test_path(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "scriptbots-durability-{tag}-{}-{nanos}.sqlite",
        std::process::id()
    ))
}

/// Probe once whether the temp directory can host a FrankenSQLite database. The pinned
/// engine's VFS refuses exFAT-class volumes at open (the same reason production run
/// databases must stay on POSIX filesystems), so DSR profiles that point TMPDIR at such a
/// volume must skip these file-backed proofs rather than fail them.
fn engine_capable_temp_dir() -> bool {
    static CAPABLE: LazyLock<bool> = LazyLock::new(|| {
        let probe = test_path("engine-capability");
        let probe_string = probe.to_string_lossy().to_string();
        let capable = Connection::open(&probe_string)
            .and_then(Connection::close)
            .is_ok();
        let _ = fs::remove_file(&probe);
        if !capable {
            eprintln!(
                "skipping: the temp filesystem cannot host a FrankenSQLite database \
                 (exFAT-class volume)"
            );
        }
        capable
    });
    *CAPABLE
}

fn cleanup(path: &PathBuf) {
    let path_string = path.to_string_lossy().to_string();
    let _ = fs::remove_file(path);
    for suffix in ["-wal", "-shm", "-journal", "-wal-fec", ".lock", "-lock"] {
        let _ = fs::remove_file(format!("{path_string}{suffix}"));
    }
}

fn sample_batch(tick: u64, metric_value: f64) -> PersistenceBatch {
    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(tick),
            agent_count: 0,
            births: 0,
            deaths: 0,
            total_energy: 0.0,
            average_energy: 0.0,
            average_health: 0.0,
            max_age: 0,
            spike_hits: 0,
        },
        epoch: 0,
        closed: false,
        metrics: vec![MetricSample::new("energy", metric_value)],
        events: Vec::new(),
        agents: Vec::new(),
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events: Vec::new(),
    }
}

/// Read `PRAGMA integrity_check` through an independent read-only connection and require `ok`.
fn assert_integrity_ok(path: &str) {
    let connection = open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("integrity-check reader should open");
    let result: String = connection
        .query_row("PRAGMA integrity_check")
        .expect("integrity_check query should run")
        .get_typed(0)
        .expect("integrity_check result should be TEXT");
    connection.close().expect("integrity reader closes");
    assert_eq!(result, "ok", "PRAGMA integrity_check failed for {path}");
}

/// Shared in-memory log capture so retry/commit evidence can be asserted on the storage
/// crate's structured `warn!`/`info!` records. Assertions always filter by the unique
/// per-test database path, so interleaved records from other tests cannot confuse them.
#[derive(Clone)]
struct LogBuffer(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for LogBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .expect("log buffer poisoned")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn log_buffer() -> LogBuffer {
    static BUFFER: LazyLock<LogBuffer> = LazyLock::new(|| {
        let buffer = LogBuffer(Arc::new(Mutex::new(Vec::new())));
        let writer = buffer.clone();
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .try_init();
        buffer
    });
    BUFFER.clone()
}

impl LogBuffer {
    fn records_for(&self, needle: &str) -> Vec<String> {
        let text =
            String::from_utf8_lossy(&self.0.lock().expect("log buffer poisoned")).into_owned();
        text.lines()
            .filter(|line| line.contains(needle))
            .map(str::to_owned)
            .collect()
    }
}

/// Count rows in a table through an independent read-only connection.
fn read_only_count(path: &str, sql: &str, params: &[SqliteValue]) -> i64 {
    let reader = open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("read-only counter should open");
    let count = reader
        .query_row_with_params(sql, params)
        .expect("count query should run")
        .get_typed::<i64>(0)
        .expect("COUNT(*) should be INTEGER");
    reader.close().expect("read-only counter closes");
    count
}

// ---------------------------------------------------------------------------
// Concurrent reads: an independent live reader observes only commit boundaries.
// ---------------------------------------------------------------------------

#[test]
fn concurrent_reader_observes_only_commit_boundaries() {
    if !engine_capable_temp_dir() {
        return;
    }
    let path = test_path("mvcc-commit-boundary");
    let path_string = path.to_string_lossy().to_string();

    let writer = Connection::open(&path_string).expect("writer opens the probe database");
    writer
        .execute("CREATE TABLE probe (id INTEGER PRIMARY KEY, note TEXT NOT NULL)")
        .expect("writer creates the probe table");
    writer
        .execute_with_params(
            "INSERT INTO probe (id, note) VALUES (?1, ?2)",
            &[1_i64.into(), "committed-before-reader".into()],
        )
        .expect("baseline committed row inserts");
    writer
        .execute_with_params(
            "INSERT INTO probe (id, note) VALUES (?1, ?2)",
            &[2_i64.into(), "committed-before-txn".into()],
        )
        .expect("second committed row inserts");

    // The reader is an independent connection: no single connection is shared between roles.
    let reader = open_with_flags(&path_string, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("independent live reader opens beside the writer");

    let committed_rows = |reader: &Connection| -> Vec<(i64, String)> {
        reader
            .query_with_params("SELECT id, note FROM probe ORDER BY id ASC", &[])
            .expect("reader snapshot query should run")
            .iter()
            .map(|row| {
                (
                    row.get_typed::<i64>(0).expect("id is INTEGER"),
                    row.get_typed::<String>(1).expect("note is TEXT"),
                )
            })
            .collect()
    };

    let baseline = committed_rows(&reader);
    assert_eq!(
        baseline,
        vec![
            (1, "committed-before-reader".to_owned()),
            (2, "committed-before-txn".to_owned())
        ],
        "reader must see exactly the committed prefix before the writer transaction"
    );

    // The writer stages a multi-statement batch but does not commit it.
    writer
        .begin_transaction()
        .expect("writer batch transaction begins");
    writer
        .execute_with_params(
            "INSERT INTO probe (id, note) VALUES (?1, ?2)",
            &[3_i64.into(), "uncommitted-staged".into()],
        )
        .expect("uncommitted row stages inside the transaction");
    writer
        .execute_with_params("DELETE FROM probe WHERE id = ?1", &[1_i64.into()])
        .expect("uncommitted delete stages inside the transaction");

    let during = committed_rows(&reader);
    assert_eq!(
        during, baseline,
        "reader observed uncommitted writer state: MVCC must expose only commit boundaries"
    );

    writer
        .commit_transaction()
        .expect("writer batch commits as one boundary");

    let after = committed_rows(&reader);
    assert_eq!(
        after,
        vec![
            (2, "committed-before-txn".to_owned()),
            (3, "uncommitted-staged".to_owned())
        ],
        "after the commit the reader must observe the whole batch atomically \
         (insert visible, delete visible, nothing half-applied)"
    );

    reader.close().expect("reader closes explicitly");
    writer.close().expect("writer closes explicitly");
    assert_integrity_ok(&path_string);
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// Induced real transient conflict: bounded transient-only retry with recovery, and a
// non-transient constraint failure that must never be retried.
// ---------------------------------------------------------------------------

/// Mirror of the production policy in `should_retry_transaction`: retry only transient
/// errors, at most `MAX_ATTEMPTS` total attempts.
const MAX_ATTEMPTS: u8 = 4;

fn commit_with_bounded_retry(
    connection: &Connection,
    stage: &dyn Fn(&Connection) -> Result<(), FrankenError>,
) -> Result<u8, (u8, FrankenError)> {
    let mut attempt = 1_u8;
    loop {
        let result = (|| -> Result<(), FrankenError> {
            connection.begin_transaction()?;
            stage(connection)?;
            connection.commit_transaction()
        })();
        match result {
            Ok(()) => return Ok(attempt),
            Err(error) => {
                let _ = connection.rollback_transaction();
                if error.is_transient() && attempt < MAX_ATTEMPTS {
                    attempt += 1;
                    continue;
                }
                return Err((attempt, error));
            }
        }
    }
}

#[test]
fn induced_real_transient_conflict_retries_bounded_and_recovers() {
    if !engine_capable_temp_dir() {
        return;
    }
    let path = test_path("transient-conflict");
    let path_string = path.to_string_lossy().to_string();

    let holder = Connection::open(&path_string).expect("lock holder opens");
    holder
        .execute("CREATE TABLE contested (id INTEGER PRIMARY KEY, val INTEGER NOT NULL)")
        .expect("contended table is created");

    let retrier = Connection::open(&path_string).expect("retrying writer opens beside holder");

    // Hold a real write transaction open so every conflicting commit attempt fails with a
    // genuine engine-raised transient error (Busy/WriteConflict family).
    holder
        .begin_transaction()
        .expect("holder transaction begins");
    holder
        .execute_with_params(
            "INSERT INTO contested (id, val) VALUES (?1, ?2)",
            &[100_i64.into(), 1_i64.into()],
        )
        .expect("holder stages its locking write");

    let stage = |connection: &Connection| -> Result<(), FrankenError> {
        connection.execute_with_params(
            "INSERT INTO contested (id, val) VALUES (?1, ?2)",
            &[200_i64.into(), 2_i64.into()],
        )?;
        Ok(())
    };

    let (attempts_used, terminal_error) = commit_with_bounded_retry(&retrier, &stage)
        .expect_err("conflicting commit must exhaust the bounded retry while the lock is held");
    assert_eq!(
        attempts_used, MAX_ATTEMPTS,
        "transient retry must be bounded at {MAX_ATTEMPTS} attempts"
    );
    assert!(
        terminal_error.is_transient(),
        "the induced contention error must be classified transient, got {terminal_error}"
    );
    eprintln!(
        "durability proof: induced transient conflict exhausted {attempts_used} attempts \
         with engine error: {terminal_error}"
    );

    // Once the holder commits, the identical retry succeeds without changing the policy.
    holder
        .commit_transaction()
        .expect("holder releases the write lock");
    let recovered_attempt = commit_with_bounded_retry(&retrier, &stage)
        .expect("the same bounded retry must succeed after the lock releases");
    assert_eq!(
        recovered_attempt, 1,
        "post-release commit should succeed on the first attempt"
    );

    // Contrast: a NOT NULL constraint violation is not transient and must stop at attempt 1.
    let bad_stage = |connection: &Connection| -> Result<(), FrankenError> {
        connection.execute_with_params(
            "INSERT INTO contested (id, val) VALUES (?1, NULL)",
            &[300_i64.into()],
        )?;
        Ok(())
    };
    let (bad_attempts, bad_error) = commit_with_bounded_retry(&retrier, &bad_stage)
        .expect_err("constraint violation must fail");
    assert_eq!(
        bad_attempts, 1,
        "a non-transient constraint failure must never be retried"
    );
    assert!(
        !bad_error.is_transient(),
        "NOT NULL violation must not be classified transient, got {bad_error}"
    );

    // Exactly the committed rows exist: holder's row plus the recovered retry's row.
    let rows = retrier
        .query_with_params("SELECT COUNT(*) FROM contested", &[])
        .expect("final row count query")
        .first()
        .expect("COUNT returns one row")
        .get_typed::<i64>(0)
        .expect("COUNT is INTEGER");
    assert_eq!(
        rows, 2,
        "bounded retry must apply the staged write exactly once"
    );

    retrier.close().expect("retrying writer closes");
    holder.close().expect("holder closes");
    assert_integrity_ok(&path_string);
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// Forced constraint failure: none of a multi-statement batch commits, proven across reopen.
// ---------------------------------------------------------------------------

#[test]
fn forced_constraint_failure_rolls_back_entire_batch_and_survives_reopen() {
    if !engine_capable_temp_dir() {
        return;
    }
    let path = test_path("rollback-atomicity");
    let path_string = path.to_string_lossy().to_string();

    let connection = Connection::open(&path_string).expect("batch writer opens");
    connection
        .execute("CREATE TABLE batch_a (id INTEGER PRIMARY KEY, val INTEGER NOT NULL)")
        .expect("batch table A is created");
    connection
        .execute("CREATE TABLE batch_b (id INTEGER PRIMARY KEY, note TEXT NOT NULL)")
        .expect("batch table B is created");
    connection
        .execute_with_params(
            "INSERT INTO batch_a (id, val) VALUES (?1, ?2)",
            &[1_i64.into(), 1_i64.into()],
        )
        .expect("pre-existing committed row");

    connection
        .begin_transaction()
        .expect("doomed batch transaction begins");
    connection
        .execute_with_params(
            "INSERT INTO batch_a (id, val) VALUES (?1, ?2)",
            &[2_i64.into(), 2_i64.into()],
        )
        .expect("batch statement one stages");
    connection
        .execute_with_params(
            "INSERT INTO batch_b (id, note) VALUES (?1, ?2)",
            &[2_i64.into(), "staged".into()],
        )
        .expect("batch statement two stages");
    let violation = connection.execute_with_params(
        "INSERT INTO batch_a (id, val) VALUES (?1, NULL)",
        &[3_i64.into()],
    );
    let violation = violation.expect_err("the forced NOT NULL violation must fail the statement");
    assert!(
        !violation.is_transient(),
        "constraint violation is a permanent error, got {violation}"
    );
    // A statement error must not implicitly commit or abort the transaction: another
    // statement still stages, and the explicit rollback decides the outcome.
    connection
        .execute_with_params(
            "INSERT INTO batch_b (id, note) VALUES (?1, ?2)",
            &[4_i64.into(), "also-staged".into()],
        )
        .expect("transaction stays live after a statement error");
    connection
        .rollback_transaction()
        .expect("the failed batch remains explicitly rollbackable");

    assert_eq!(
        read_only_count(&path_string, "SELECT COUNT(*) FROM batch_a", &[]),
        1,
        "rollback left batch rows in table A"
    );
    assert_eq!(
        read_only_count(&path_string, "SELECT COUNT(*) FROM batch_b", &[]),
        0,
        "rollback left batch rows in table B"
    );

    connection.close().expect("batch writer closes");

    // Reopen independently: the forced failure committed none of the batch, durably.
    let reopened = Connection::open_existing(&path_string).expect("database reopens after failure");
    for (table, expected) in [("batch_a", 1_i64), ("batch_b", 0)] {
        let count = reopened
            .query_row_with_params(&format!("SELECT COUNT(*) FROM {table}"), &[])
            .expect("reopened count query runs")
            .get_typed::<i64>(0)
            .expect("COUNT is INTEGER");
        assert_eq!(
            count, expected,
            "reopened database shows rolled-back rows in {table}"
        );
    }
    reopened.close().expect("reopened writer closes");
    assert_integrity_ok(&path_string);
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// Production contention: `Storage::flush` under a real external write transaction is
// bounded, transient, logged with path/attempt/commit-state evidence, terminally typed,
// and the admitted outbox batch recovers exactly once afterwards.
// ---------------------------------------------------------------------------

#[test]
fn storage_flush_under_real_contention_is_bounded_logged_and_recovers_exactly_once() {
    if !engine_capable_temp_dir() {
        return;
    }
    let logs = log_buffer();
    let path = test_path("flush-contention");
    let path_string = path.to_string_lossy().to_string();

    // Large thresholds keep the batch buffered until the explicit flush below.
    let mut storage =
        Storage::create_unattributed_file_with_thresholds(&path_string, 1_000, 1_000, 1_000, 1_000)
            .expect("file-backed storage opens");
    storage
        .persist(&sample_batch(1, 41.5))
        .expect("batch admission stages the outbox payload");

    // Discover the run id the unattributed storage created so the contender can stage a
    // schema-clean locking write it will roll back.
    let run_id: String = {
        let reader = open_with_flags(&path_string, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .expect("run-id probe reader opens");
        let run_id = reader
            .query_row_with_params("SELECT run_id FROM runs LIMIT 1", &[])
            .expect("run id query runs")
            .get_typed::<String>(0)
            .expect("run id is TEXT");
        reader.close().expect("run-id probe closes");
        run_id
    };

    // An independent contending connection holds a real write transaction on the same file.
    let contender = Connection::open(&path_string).expect("contender opens the same database");
    contender
        .begin_transaction()
        .expect("contender transaction begins");
    contender
        .execute_with_params(
            "INSERT INTO metrics (run_id, tick, name, value) VALUES (?1, ?2, ?3, ?4)",
            &[
                run_id.clone().into(),
                999_999_i64.into(),
                "contention-probe".into(),
                0.0_f64.into(),
            ],
        )
        .expect("contender stages its locking write");

    let terminal = storage
        .flush()
        .expect_err("flush under real contention must exhaust the bounded retry");
    assert!(
        matches!(
            &terminal,
            StorageError::Transaction {
                attempts: 4,
                transient: true,
                commit_state: FailureCommitState::RolledBack,
                ..
            }
        ),
        "contended flush must fail after exactly 4 bounded attempts, every one transient and \
         fully rolled back, got {terminal}"
    );
    eprintln!("durability proof: contended flush terminal result: {terminal}");

    // Log evidence: retries name the database path, the attempt, and the transient rollback.
    let retry_records = logs.records_for(&path_string);
    let retry_warnings: Vec<&String> = retry_records
        .iter()
        .filter(|line| line.contains("retrying fully rolled-back FrankenSQLite transaction"))
        .collect();
    assert_eq!(
        retry_warnings.len(),
        3,
        "three rolled-back transient retries should each log a warning: {retry_records:?}"
    );
    for record in &retry_warnings {
        assert!(
            record.contains("transient=true"),
            "retry warning must identify the transient classification: {record}"
        );
        assert!(
            record.contains("RolledBack"),
            "retry warning must identify the transaction commit state: {record}"
        );
        assert!(
            record.contains("attempt"),
            "retry warning must identify the attempt: {record}"
        );
    }
    // The typed terminal result identifies the transaction, the attempt count, and the cause.
    let terminal_text = terminal.to_string();
    assert!(
        terminal_text.contains("attempt(s)"),
        "terminal result must identify the attempts: {terminal_text}"
    );
    assert!(
        terminal_text.contains("transient=true"),
        "terminal result must identify the transient classification: {terminal_text}"
    );

    // Roll back the contender's probe write: the database is left schema- and row-clean.
    contender
        .rollback_transaction()
        .expect("contender rolls its probe write back");
    contender.close().expect("contender closes");

    // The terminally failed storage cannot flush again; its drop must not replay the buffer.
    let terminal_again = storage
        .flush()
        .expect_err("a terminally failed storage stays failed");
    assert!(
        matches!(terminal_again, StorageError::TerminallyFailed),
        "expected TerminallyFailed, got {terminal_again}"
    );
    drop(storage);

    // Recovery replays the admitted outbox batch exactly once.
    let mut recovered =
        StoragePipeline::recover_existing(&path_string).expect("recovery reopens the run");
    let shutdown = recovered
        .shutdown()
        .expect("recovered pipeline shuts down cleanly");
    assert_eq!(
        shutdown.committed_tick,
        Some(1),
        "recovery must commit the admitted tick-1 batch"
    );

    let reader = StorageReader::open(&path_string).expect("reader opens the recovered run");
    let metrics = reader.recent_metrics(8).expect("metrics query runs");
    let energy_rows: Vec<_> = metrics.iter().filter(|row| row.name == "energy").collect();
    assert_eq!(
        energy_rows.len(),
        1,
        "the admitted batch must be applied exactly once, got {energy_rows:?}"
    );
    assert_eq!(
        reader.max_tick().expect("max tick query runs"),
        Some(1),
        "the recovered run must expose tick 1"
    );
    let watermarks: PersistenceWatermarks = reader
        .persistence_watermarks()
        .expect("watermark query runs");
    assert!(
        watermarks.admitted.is_some()
            && watermarks.admitted == watermarks.applied
            && watermarks.applied == watermarks.durable,
        "recovery must converge admitted/applied/durable watermarks, got {watermarks:?}"
    );
    reader.close().expect("reader closes");

    // The contender's probe row was rolled back: it must not exist.
    assert_eq!(
        read_only_count(
            &path_string,
            "SELECT COUNT(*) FROM metrics WHERE name = ?1",
            &["contention-probe".into()],
        ),
        0,
        "the contender's rolled-back probe row leaked into the database"
    );

    // Commit evidence: the recovered flush logged the committed transaction with its path.
    let commit_records = logs.records_for(&path_string);
    assert!(
        commit_records
            .iter()
            .any(|line| line.contains("transaction committed")),
        "a committed-transaction record naming the database path must exist: {commit_records:?}"
    );

    assert_integrity_ok(&path_string);
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// Child-process durability: acknowledged data survives process exit and reopen.
// ---------------------------------------------------------------------------

/// Child role: create a file-backed run, persist + flush + close, print the receipt
/// evidence, and exit successfully. No-op when the guard variable is absent so the parent
/// suite skips it.
#[test]
fn durability_child_clean_exit() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(path_string) = std::env::var("SCRIPTBOTS_DURABILITY_CHILD_PATH") else {
        return Ok(());
    };
    let tick: u64 = std::env::var("SCRIPTBOTS_DURABILITY_CHILD_TICK")?.parse()?;
    let mut storage = Storage::create_unattributed_file_with_thresholds(&path_string, 1, 1, 1, 1)?;
    storage.persist(&sample_batch(tick, 77.25))?;
    storage.flush()?;
    let watermarks = storage.persistence_watermarks()?;
    storage.close()?;
    // Diagnostic evidence on stdout for the parent to verify.
    println!(
        "durability-child: database={path_string} tick={tick} \
         admitted={:?} applied={:?} durable={:?} terminal=clean-close",
        watermarks.admitted, watermarks.applied, watermarks.durable
    );
    Ok(())
}

#[test]
fn acknowledged_batch_survives_child_process_exit_and_reopen()
-> Result<(), Box<dyn std::error::Error>> {
    if !engine_capable_temp_dir() {
        return Ok(());
    }
    let path = test_path("child-durability");
    let path_string = path.to_string_lossy().to_string();

    let output = Command::new(std::env::current_exe()?)
        .args(["--exact", "durability_child_clean_exit", "--nocapture"])
        .env("SCRIPTBOTS_DURABILITY_CHILD_PATH", &path_string)
        .env("SCRIPTBOTS_DURABILITY_CHILD_TICK", "7")
        .output()?;
    assert!(
        output.status.success(),
        "child writer failed: status={:?} stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(&format!("database={path_string}")),
        "child log evidence must identify the database path: {stdout}"
    );
    assert!(
        stdout.contains("terminal=clean-close"),
        "child log evidence must identify the terminal result: {stdout}"
    );

    // Independently reopen: the acknowledged batch is fully visible and durable.
    let reader = StorageReader::open(&path_string)?;
    assert_eq!(
        reader.max_tick()?,
        Some(7),
        "acknowledged tick lost across process exit"
    );
    let metrics = reader.recent_metrics(8)?;
    assert!(
        metrics
            .iter()
            .any(|row| row.name == "energy" && (row.value - 77.25).abs() < f64::EPSILON),
        "acknowledged metric payload lost across process exit: {metrics:?}"
    );
    let watermarks = reader.persistence_watermarks()?;
    assert!(
        watermarks.admitted.is_some()
            && watermarks.admitted == watermarks.applied
            && watermarks.applied == watermarks.durable,
        "clean child exit must leave converged watermarks, got {watermarks:?}"
    );
    reader.close()?;

    assert_integrity_ok(&path_string);
    cleanup(&path);
    Ok(())
}

// ---------------------------------------------------------------------------
// Startup failure matrix: a corrupt header is refused without mutating database bytes.
// ---------------------------------------------------------------------------

#[test]
fn corrupt_header_startup_refusal_leaves_database_bytes_untouched() {
    if !engine_capable_temp_dir() {
        return;
    }
    let path = test_path("corrupt-header");
    let path_string = path.to_string_lossy().to_string();

    let mut storage = Storage::create_unattributed_file_with_thresholds(&path_string, 1, 1, 1, 1)
        .expect("storage opens for the corruption fixture");
    storage
        .persist(&sample_batch(1, 5.5))
        .expect("fixture batch persists");
    storage.close().expect("fixture storage closes");
    assert_integrity_ok(&path_string);

    let mut bytes = fs::read(&path).expect("fixture database reads");
    assert!(
        bytes.len() >= 100,
        "fixture database should have a full header page"
    );
    // Destroy the database magic and header shape; leave the file length intact.
    for (index, byte) in bytes.iter_mut().take(32).enumerate() {
        *byte = b'X'.wrapping_add(index as u8);
    }
    fs::write(&path, &bytes).expect("corrupted fixture writes");

    let reader_outcome = StorageReader::open(&path_string);
    assert!(
        reader_outcome.is_err(),
        "a corrupt-header database must be refused by the verified reader"
    );
    let engine_outcome = Connection::open_existing(&path_string);
    assert!(
        engine_outcome.is_err(),
        "a corrupt-header database must be refused by the engine"
    );

    assert_eq!(
        fs::read(&path).expect("post-refusal database reads"),
        bytes,
        "a refused startup mutated database bytes"
    );
    cleanup(&path);
}
