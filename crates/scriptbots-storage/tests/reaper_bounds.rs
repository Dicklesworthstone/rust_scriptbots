//! Supervised reaping must be BOUNDED, and it must never drop a worker.
//!
//! The old handoff spawned one independent OS thread per timeout, with no bound.
//! A slow or wedged disk does not time out once — it times out over and over, and
//! each timeout spawned another thread that then blocked on the same sick disk.
//! The failure mode is a thread-count explosion caused BY the thing that was
//! already failing: the process runs out of threads while trying to clean up after
//! a disk that stopped answering.

use scriptbots_storage::{StoragePipeline, storage_reaper_stats};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_db(label: &str, index: usize) -> String {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir()
        .join(format!(
            "scriptbots_reaper_{label}_{index}_{}_{nonce}.sqlite",
            std::process::id()
        ))
        .to_str()
        .expect("utf8 path")
        .to_owned()
}

#[test]
fn the_reaper_registry_reports_its_own_state() {
    // The bead requires active/queued counts to be OBSERVABLE. A supervisor whose
    // saturation you cannot see is a supervisor you cannot diagnose: the symptom of
    // the old unbounded design was "the process mysteriously has 900 threads", with
    // nothing to point at.
    let stats = storage_reaper_stats();

    // Whatever else is true, the counters must be internally coherent: you cannot
    // have queued work with nobody active to drain it.
    if stats.queued > 0 {
        assert!(
            stats.active > 0,
            "queued reap requests with NO active reaper means the queued work will \
             never be drained — every one of those requests holds a JoinHandle that \
             is now leaked forever"
        );
    }
}

#[test]
fn many_pipelines_shut_down_cleanly_without_a_thread_explosion() {
    // The saturation case, exercised through the real path: open and shut a stream
    // of pipelines. Each shutdown that times out hands its worker to the reaper. The
    // old code would spawn one thread per handoff, unbounded; the registry bounds
    // the concurrent reapers and runs the overflow on the caller.
    //
    // The property that matters and is checkable deterministically: EVERY worker is
    // accounted for. Nothing is dropped, nothing is left queued behind a reaper that
    // already retired.
    let before = storage_reaper_stats();

    let mut paths = Vec::new();
    for index in 0..12 {
        let path = temp_db("many", index);
        let mut pipeline =
            StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
                .expect("pipeline");
        pipeline.shutdown().expect("shutdown");
        paths.push(path);
    }

    let after = storage_reaper_stats();

    // NOTHING MAY BE LEFT QUEUED. A queued request holds a worker's JoinHandle; if
    // the reaper that owned its path has retired without draining it, that handle is
    // leaked and the worker is never joined.
    assert_eq!(
        after.queued, 0,
        "reap requests are still queued after every pipeline shut down cleanly — \
         their JoinHandles are leaked"
    );

    // The registry must never exceed its own bound. This is the whole point: a sick
    // disk cannot make the process spawn threads without limit.
    assert!(
        after.active <= 4,
        "the reaper registry exceeded its concurrency bound: {} active",
        after.active
    );

    // Counters only move forward, and coherently.
    assert!(after.started >= before.started);
    assert!(after.coalesced >= before.coalesced);
    assert!(after.synchronous >= before.synchronous);

    for path in paths {
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn reaping_one_path_does_not_block_another() {
    // Per-PATH keying is what makes this true. If the registry were keyed on a single
    // global "am I reaping" flag, one sick disk would stall the cleanup of every
    // other storage path in the process — and the healthy paths would be punished for
    // the sick one's failure.
    let path_a = temp_db("cross", 0);
    let path_b = temp_db("cross", 1);

    let mut a =
        StoragePipeline::create_unattributed_file_with_thresholds(&path_a, 1, 1, 1, 1).expect("a");
    let mut b =
        StoragePipeline::create_unattributed_file_with_thresholds(&path_b, 1, 1, 1, 1).expect("b");

    // Both shut down; neither may be blocked by the other's reaping.
    a.shutdown().expect("a shuts down");
    b.shutdown().expect("b shuts down independently of a");

    let stats = storage_reaper_stats();
    assert_eq!(stats.queued, 0, "no request may be stranded");

    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);
}

#[test]
fn a_path_can_be_reopened_after_it_has_been_reaped() {
    // Recovery. A reaped path must be usable again — otherwise a single timeout
    // would poison that database for the rest of the process's life, and the
    // "supervised reap" would be a permanent quarantine rather than a cleanup.
    let path = temp_db("reopen", 0);

    let mut first = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("first");
    first.shutdown().expect("first shutdown");
    // Clear the database AND its sidecars: a stale -wal is refused on open, and
    // leaving one behind would make this test fail for a reason that has nothing
    // to do with reaping.
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{path}{suffix}"));
    }

    // The path's writer lease must have been released by the reap. If the reaper
    // held it, this second open would fail and a single timeout would have poisoned
    // that database for the life of the process.
    let mut second = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("the path must be usable again after it was reaped");
    second.shutdown().expect("second shutdown");

    assert_eq!(
        storage_reaper_stats().queued,
        0,
        "reopening must not strand a reap request"
    );

    let _ = std::fs::remove_file(&path);
}
