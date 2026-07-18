//! Proofs for the deadline/cancellation async read lane (bd-2z0.8.9.12).
//!
//! Every scenario uses a real file-backed run database: the lane must return the same data
//! as the synchronous `StorageReader`, surface typed interrupts on external cancellation,
//! expire tight budgets on genuinely slow queries, and coexist with an actively writing
//! `StoragePipeline` on MVCC commit boundaries.

use fsqlite::FrankenError;
use fsqlite_types::cx::Cx;
use scriptbots_core::{MetricSample, PersistenceBatch, Tick, TickSummary};
use scriptbots_storage::{
    Storage, StorageError, StoragePipeline, StorageReader,
    async_lane::{AsyncReadLane, cx_with_deadline},
};
use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

fn test_path(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "scriptbots-async-lane-{tag}-{}-{nanos}.sqlite",
        std::process::id()
    ))
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
        metrics: vec![
            MetricSample::new("energy", metric_value),
            MetricSample::new("population", 100.0 + tick as f64),
        ],
        events: Vec::new(),
        agents: Vec::new(),
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events: Vec::new(),
    }
}

fn write_fixture(path: &str, ticks: u64) {
    let mut storage = Storage::create_unattributed_file_with_thresholds(path, 1, 1, 1, 1)
        .expect("fixture storage opens");
    for tick in 1..=ticks {
        storage
            .persist(&sample_batch(tick, tick as f64 * 1.5))
            .expect("fixture batch persists");
    }
    storage.close().expect("fixture storage closes");
}

#[test]
fn lane_reads_match_the_sync_reader() {
    let path = test_path("parity");
    let path_string = path.to_string_lossy().to_string();
    write_fixture(&path_string, 5);

    let cx = Cx::new();
    let lane = AsyncReadLane::open(&path_string).expect("lane opens");
    let reader = StorageReader::open(&path_string).expect("sync reader opens");

    let lane_metrics = lane.recent_metrics(6, &cx).expect("lane metrics");
    let sync_metrics = reader.recent_metrics(6).expect("sync metrics");
    assert_eq!(
        lane_metrics, sync_metrics,
        "lane and sync reader must agree on recent metrics"
    );

    let lane_predators = lane.top_predators(4, &cx).expect("lane predators");
    let sync_predators = reader.top_predators(4).expect("sync predators");
    assert_eq!(
        lane_predators, sync_predators,
        "lane and sync reader must agree on top predators"
    );

    let lane_ledger = lane.run_ledger_summary(&cx).expect("lane ledger");
    let sync_ledger = reader.run_ledger_summary().expect("sync ledger");
    assert_eq!(
        lane_ledger, sync_ledger,
        "lane and sync reader must agree on the run ledger"
    );

    reader.close().expect("sync reader closes");
    lane.close().expect("lane closes");
    cleanup(&path);
}

#[test]
fn cancelled_query_surfaces_typed_interrupt_and_lane_recovers() {
    let path = test_path("cancel");
    let path_string = path.to_string_lossy().to_string();
    write_fixture(&path_string, 3);

    let cx = Cx::new();
    let canceller = cx.clone();
    let slow_sql = "WITH RECURSIVE cnt(x) AS (
                        SELECT 1
                        UNION ALL
                        SELECT x + 1 FROM cnt LIMIT 5000000
                    )
                    SELECT COUNT(*) FROM cnt";

    let lane_thread = std::thread::spawn({
        let path_string = path_string.clone();
        let slow_sql = slow_sql.to_owned();
        move || {
            let lane = AsyncReadLane::open(&path_string).expect("lane opens");
            let started = Instant::now();
            let result = lane.query_rows(&slow_sql, &[], &cx);
            (result, started.elapsed(), lane)
        }
    });
    std::thread::sleep(Duration::from_millis(50));
    canceller.cancel();

    let (result, elapsed, lane) = lane_thread.join().expect("lane thread joins");
    let error = result.expect_err("a cancelled slow query must not complete");
    assert!(
        matches!(error, StorageError::Database(FrankenError::Interrupt)),
        "cancellation must surface a typed interrupt, got {error} after {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(30),
        "cancellation took {elapsed:?}; the whole scan would take far longer"
    );

    // The lane is not poisoned: a fresh context must read normally.
    let recovered = lane
        .recent_metrics(2, &Cx::new())
        .expect("lane accepts queries after a cancellation");
    assert!(!recovered.is_empty(), "lane reads after cancellation");
    lane.close().expect("lane closes");
    cleanup(&path);
}

#[test]
fn sub_ms_deadline_expires_a_slow_query() {
    let path = test_path("deadline");
    let path_string = path.to_string_lossy().to_string();
    write_fixture(&path_string, 2);

    let lane = AsyncReadLane::open(&path_string).expect("lane opens");
    let slow_sql = "WITH RECURSIVE cnt(x) AS (
                        SELECT 1
                        UNION ALL
                        SELECT x + 1 FROM cnt LIMIT 8000000
                    )
                    SELECT COUNT(*) FROM cnt";
    let started = Instant::now();
    let result = lane.query_rows(slow_sql, &[], &cx_with_deadline(Duration::from_millis(1)));
    let elapsed = started.elapsed();

    let error = result.expect_err("a sub-millisecond deadline must expire the slow query");
    assert!(
        matches!(error, StorageError::Database(FrankenError::Interrupt)),
        "an expired deadline must surface a typed interrupt, got {error}"
    );
    assert!(
        elapsed < Duration::from_secs(30),
        "the deadline fired after {elapsed:?}; an unchecked scan would run far longer"
    );
    lane.close().expect("lane closes");
    cleanup(&path);
}

#[test]
fn concurrent_lane_readers_observe_commit_boundaries_while_writer_applies_batches() {
    let path = test_path("concurrent");
    let path_string = path.to_string_lossy().to_string();
    write_fixture(&path_string, 3);

    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(
        &path_string,
        1,
        1,
        1,
        1,
    )
    .expect("writer pipeline opens");

    let mut readers = Vec::new();
    for _reader_index in 0..4 {
        readers.push(std::thread::spawn({
            let path_string = path_string.clone();
            move || {
                let lane = AsyncReadLane::open(&path_string).expect("reader lane opens");
                let cx = Cx::new();
                let mut max_seen = 0_u64;
                for _ in 0..40 {
                    let metrics = lane.recent_metrics(2, &cx).expect("concurrent read");
                    if let Some(newest) = metrics.first() {
                        max_seen = max_seen.max(newest.tick);
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                lane.close().expect("reader lane closes");
                max_seen
            }
        }));
    }

    for tick in 4..=12 {
        pipeline
            .submit(&sample_batch(tick, tick as f64 * 2.5))
            .expect("writer batch admitted");
        std::thread::sleep(Duration::from_millis(5));
    }

    let mut peaks = Vec::new();
    for reader in readers {
        peaks.push(reader.join().expect("reader joins"));
    }
    let shutdown = pipeline.shutdown().expect("writer shuts down");
    assert!(
        shutdown.committed_tick.is_some_and(|tick| tick >= 12),
        "writer committed every batch: {shutdown:?}"
    );
    assert!(
        peaks.iter().all(|peak| *peak >= 3),
        "every lane reader observed the pre-existing commit boundary: {peaks:?}"
    );
    eprintln!("durability proof: concurrent lane reader peaks {peaks:?}");

    cleanup(&path);
}
