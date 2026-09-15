//! Supervised reaping must be BOUNDED, and it must never drop a worker.
//!
//! Completion proof for bd-2z0.5.14:
//! 1. Typed reaper snapshot with active, queued, started, coalesced, synchronous,
//!    hung, and oldest-active-age fields, with checked invariants.
//! 2. Deterministic test seams: held worker, duplicate same-path timeouts, registry
//!    saturation cap, thread spawn failure, and admission fallback.
//! 3. Exactly-once receipt and JoinHandle accounting through coalescing, queued FIFO
//!    drain, synchronous fallback, panic/poison recovery, and final retirement.
//! 4. Cross-path independence (hung path never blocks healthy path) and safe reopen.
//! 5. Negative controls for skipped drain and double consumption.
//! 6. Mock-free E2E timeout handoff, held worker, and eventual recovery with structured logs.

use scriptbots_storage::{
    DEFAULT_HUNG_REAPER_TIMEOUT, MAX_CONCURRENT_REAPERS, ReaperAccountingError,
    ReaperFallbackReason, ReaperFaultPoint, ReaperJoinOutcome, ReaperReceiptState, ReaperStats,
    StorageDeadlines, StoragePipeline, arm_reaper_fault, clear_all_reaper_faults,
    clear_reaper_fault, handoff_join_only_for_test, next_reap_request_id,
    poison_reaper_registry_for_test, reaper_accounting_receipts, reset_reaper_registry_for_test,
    set_reaper_hung_threshold_for_test, simulate_negative_double_consumption_for_test,
    simulate_negative_skipped_drain_for_test, storage_reaper_stats, verify_reaper_accounting,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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

fn cleanup_temp_db(path: &str) {
    for suffix in ["", "-wal", "-shm", ".lock"] {
        let _ = std::fs::remove_file(format!("{path}{suffix}"));
    }
}

#[test]
fn the_reaper_registry_reports_its_own_state() {
    let stats = storage_reaper_stats();
    stats.check_invariants().expect("invariants must hold");
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
fn zero_paths_invariants_and_error_detection() {
    reset_reaper_registry_for_test();
    let stats = storage_reaper_stats();
    assert_eq!(stats.active, 0);
    assert_eq!(stats.queued, 0);
    assert_eq!(stats.hung, 0);
    assert_eq!(stats.oldest_active_age, None);
    assert_eq!(stats.synchronous_fallback(), 0);
    assert!(stats.check_invariants().is_ok());

    // Negative invariant cases:
    // 1. Queued with active == 0
    let bad_queued = ReaperStats {
        active: 0,
        queued: 1,
        ..Default::default()
    };
    assert!(bad_queued.check_invariants().is_err());

    // 2. Hung > active
    let bad_hung = ReaperStats {
        active: 1,
        hung: 2,
        oldest_active_age: Some(Duration::from_secs(1)),
        ..Default::default()
    };
    assert!(bad_hung.check_invariants().is_err());

    // 3. Active > MAX_CONCURRENT_REAPERS
    let bad_active = ReaperStats {
        active: MAX_CONCURRENT_REAPERS + 1,
        oldest_active_age: Some(Duration::from_millis(10)),
        ..Default::default()
    };
    assert!(bad_active.check_invariants().is_err());

    // 4. Active > 0 without oldest_active_age
    let bad_age = ReaperStats {
        active: 1,
        oldest_active_age: None,
        ..Default::default()
    };
    assert!(bad_age.check_invariants().is_err());
}

#[test]
fn many_pipelines_shut_down_cleanly_without_a_thread_explosion() {
    reset_reaper_registry_for_test();
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
    after.check_invariants().expect("invariants must hold");

    assert_eq!(
        after.queued, 0,
        "reap requests are still queued after every pipeline shut down cleanly — \
         their JoinHandles are leaked"
    );

    assert!(
        after.active <= MAX_CONCURRENT_REAPERS,
        "the reaper registry exceeded its concurrency bound: {} active",
        after.active
    );

    assert!(after.started >= before.started);
    assert!(after.coalesced >= before.coalesced);
    assert!(after.synchronous >= before.synchronous);

    for path in paths {
        cleanup_temp_db(&path);
    }
}

#[test]
fn held_worker_and_hung_observability() {
    reset_reaper_registry_for_test();
    set_reaper_hung_threshold_for_test(Duration::from_millis(30));

    let path = temp_db("held", 0);
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");

    let pause_guard = pipeline.pause_worker_for_test().expect("pause worker");
    let req_id = pipeline.handoff_to_reaper().expect("handoff to reaper");
    drop(pipeline);

    // Initial state: reaper is active, not yet hung
    let stats = storage_reaper_stats();
    assert_eq!(stats.active, 1);
    assert_eq!(stats.queued, 0);
    assert!(stats.oldest_active_age.is_some());

    // Wait past hung threshold
    thread::sleep(Duration::from_millis(50));
    let stats_hung = storage_reaper_stats();
    assert_eq!(stats_hung.active, 1);
    assert_eq!(stats_hung.hung, 1);
    assert!(stats_hung.oldest_active_age.unwrap() >= Duration::from_millis(30));
    stats_hung
        .check_invariants()
        .expect("invariants hold when hung");

    // Release held worker
    pause_guard.release();

    // Wait for reaper to complete
    let deadline = Instant::now() + Duration::from_secs(3);
    while storage_reaper_stats().active > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }

    let stats_final = storage_reaper_stats();
    assert_eq!(stats_final.active, 0);
    assert_eq!(stats_final.hung, 0);
    assert_eq!(stats_final.queued, 0);
    assert_eq!(stats_final.oldest_active_age, None);

    let receipts = reaper_accounting_receipts();
    let req_receipt = receipts
        .iter()
        .find(|r| r.request_id == req_id)
        .expect("receipt exists");
    assert_eq!(req_receipt.state, ReaperReceiptState::Retired);
    assert_eq!(req_receipt.join_outcome, Some(ReaperJoinOutcome::Clean));

    cleanup_temp_db(&path);
}

#[test]
fn duplicate_same_path_timeouts_coalesce_and_drain_fifo() {
    reset_reaper_registry_for_test();
    let path = temp_db("coalesce", 0);
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");

    let pause_guard = pipeline.pause_worker_for_test().expect("pause worker");
    let req_id_1 = pipeline.handoff_to_reaper().expect("handoff 1");
    drop(pipeline);

    // Path is active. Now hand off a second and third request for the SAME path.
    let path_arc: Arc<str> = path.as_str().into();
    let dummy_worker_2 = thread::spawn(|| None);
    let req_id_2 = handoff_join_only_for_test(
        dummy_worker_2,
        Arc::clone(&path_arc),
        scriptbots_storage::AnalyticsSnapshotProvider::default(),
    );

    let dummy_worker_3 = thread::spawn(|| None);
    let req_id_3 = handoff_join_only_for_test(
        dummy_worker_3,
        Arc::clone(&path_arc),
        scriptbots_storage::AnalyticsSnapshotProvider::default(),
    );

    let stats = storage_reaper_stats();
    assert_eq!(stats.active, 1);
    assert_eq!(stats.queued, 2);
    assert!(stats.coalesced >= 2);
    stats.check_invariants().expect("invariants hold");

    // Release worker 1
    pause_guard.release();

    // Wait for all to complete
    let deadline = Instant::now() + Duration::from_secs(3);
    while storage_reaper_stats().active > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }

    assert_eq!(storage_reaper_stats().active, 0);
    assert_eq!(storage_reaper_stats().queued, 0);

    // Verify all 3 receipts retired
    let receipts = reaper_accounting_receipts();
    for id in [req_id_1, req_id_2, req_id_3] {
        let r = receipts
            .iter()
            .find(|r| r.request_id == id)
            .unwrap_or_else(|| panic!("receipt {id} must exist"));
        assert_eq!(r.state, ReaperReceiptState::Retired);
        assert!(r.join_outcome.is_some());
    }

    verify_reaper_accounting().expect("exact accounting verified");

    // Reopen exact path to verify lease was cleanly released
    cleanup_temp_db(&path);
    let mut reopened = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("reopen after coalesce");
    reopened.shutdown().expect("reopened shutdown");
    cleanup_temp_db(&path);
}

#[test]
fn reaping_one_path_does_not_block_another() {
    reset_reaper_registry_for_test();
    let path_a = temp_db("cross_a", 0);
    let path_b = temp_db("cross_b", 1);

    let mut a =
        StoragePipeline::create_unattributed_file_with_thresholds(&path_a, 1, 1, 1, 1).expect("a");
    let pause_guard_a = a.pause_worker_for_test().expect("pause a");
    let req_a = a.handoff_to_reaper().expect("handoff a");
    drop(a);

    // While Path A is actively held in the reaper, Path B operates independently!
    let mut b =
        StoragePipeline::create_unattributed_file_with_thresholds(&path_b, 1, 1, 1, 1).expect("b");
    b.shutdown()
        .expect("b shuts down cleanly while a is reaped");
    cleanup_temp_db(&path_b);

    let mut b2 =
        StoragePipeline::create_unattributed_file_with_thresholds(&path_b, 1, 1, 1, 1).expect("b2");
    b2.shutdown()
        .expect("b2 shuts down cleanly while a is still held");
    cleanup_temp_db(&path_b);

    // Now release Path A
    pause_guard_a.release();
    let deadline = Instant::now() + Duration::from_secs(3);
    while storage_reaper_stats().active > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }

    let stats = storage_reaper_stats();
    assert_eq!(stats.active, 0);
    assert_eq!(stats.queued, 0);

    let receipts = reaper_accounting_receipts();
    let rec_a = receipts
        .iter()
        .find(|r| r.request_id == req_a)
        .expect("receipt a");
    assert_eq!(rec_a.state, ReaperReceiptState::Retired);

    cleanup_temp_db(&path_a);
}

#[test]
fn registry_saturation_and_cap_plus_one_fallback() {
    reset_reaper_registry_for_test();

    let mut pipelines = Vec::new();
    let mut guards = Vec::new();
    let mut paths = Vec::new();

    // Saturate up to MAX_CONCURRENT_REAPERS (4)
    for i in 0..MAX_CONCURRENT_REAPERS {
        let p = temp_db("sat", i);
        let mut pipe = StoragePipeline::create_unattributed_file_with_thresholds(&p, 1, 1, 1, 1)
            .expect("pipe");
        let guard = pipe.pause_worker_for_test().expect("pause");
        pipe.handoff_to_reaper().expect("handoff");
        guards.push(guard);
        paths.push(p);
        pipelines.push(pipe);
    }

    let stats = storage_reaper_stats();
    assert_eq!(stats.active, MAX_CONCURRENT_REAPERS);

    // 5th distinct path must hit synchronous saturation fallback
    let p5 = temp_db("sat", MAX_CONCURRENT_REAPERS);
    let mut pipe5 =
        StoragePipeline::create_unattributed_file_with_thresholds(&p5, 1, 1, 1, 1).expect("pipe5");
    let before_sync = storage_reaper_stats().synchronous;
    let req5_id = pipe5.handoff_to_reaper().expect("handoff 5");
    drop(pipe5);

    let after_sync = storage_reaper_stats().synchronous;
    assert_eq!(
        after_sync,
        before_sync + 1,
        "5th path must increment synchronous count"
    );

    let receipts = reaper_accounting_receipts();
    let r5 = receipts
        .iter()
        .find(|r| r.request_id == req5_id)
        .expect("receipt 5");
    assert_eq!(r5.state, ReaperReceiptState::Retired);
    assert_eq!(r5.fallback_reason, Some(ReaperFallbackReason::Saturation));

    // Release all held workers
    for g in guards {
        g.release();
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    while storage_reaper_stats().active > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }

    assert_eq!(storage_reaper_stats().active, 0);
    verify_reaper_accounting().expect("all accounted");

    for p in paths {
        cleanup_temp_db(&p);
    }
    cleanup_temp_db(&p5);
}

#[test]
fn spawn_failure_seam_falls_back_synchronously_without_leaking_handle() {
    reset_reaper_registry_for_test();
    arm_reaper_fault(ReaperFaultPoint::ForceSpawnFailure);

    let path = temp_db("spawn_fail", 0);
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");

    let before_sync = storage_reaper_stats().synchronous;
    let req_id = pipeline.handoff_to_reaper().expect("handoff");
    drop(pipeline);

    let after_sync = storage_reaper_stats().synchronous;
    assert_eq!(
        after_sync,
        before_sync + 1,
        "spawn failure must run synchronously"
    );
    assert_eq!(storage_reaper_stats().active, 0);

    let receipts = reaper_accounting_receipts();
    let rec = receipts
        .iter()
        .find(|r| r.request_id == req_id)
        .expect("receipt");
    assert_eq!(rec.state, ReaperReceiptState::Retired);
    assert_eq!(
        rec.fallback_reason,
        Some(ReaperFallbackReason::SpawnFailure)
    );
    assert!(rec.join_outcome.is_some());

    clear_reaper_fault(ReaperFaultPoint::ForceSpawnFailure);

    // Reopen exact path
    cleanup_temp_db(&path);
    let mut reopened = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("reopen");
    reopened.shutdown().expect("reopened shutdown");
    cleanup_temp_db(&path);
}

#[test]
fn registry_admission_fallback_seam() {
    reset_reaper_registry_for_test();
    arm_reaper_fault(ReaperFaultPoint::ForceRegistryAdmissionFallback);

    let path = temp_db("adm_fail", 0);
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");

    let before_sync = storage_reaper_stats().synchronous;
    let req_id = pipeline.handoff_to_reaper().expect("handoff");
    drop(pipeline);

    let after_sync = storage_reaper_stats().synchronous;
    assert_eq!(after_sync, before_sync + 1);

    let receipts = reaper_accounting_receipts();
    let rec = receipts
        .iter()
        .find(|r| r.request_id == req_id)
        .expect("receipt");
    assert_eq!(rec.state, ReaperReceiptState::Retired);
    assert_eq!(
        rec.fallback_reason,
        Some(ReaperFallbackReason::AdmissionRefusal)
    );

    clear_reaper_fault(ReaperFaultPoint::ForceRegistryAdmissionFallback);
    cleanup_temp_db(&path);
}

#[test]
fn poisoned_registry_mutex_recovery() {
    reset_reaper_registry_for_test();
    poison_reaper_registry_for_test();

    // Lock recovery must succeed and return coherent stats without panicking
    let stats = storage_reaper_stats();
    assert!(stats.check_invariants().is_ok());

    // Normal handoff must also succeed after poison
    let path = temp_db("poison_rec", 0);
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");
    pipeline.handoff_to_reaper().expect("handoff after poison");
    drop(pipeline);

    let deadline = Instant::now() + Duration::from_secs(3);
    while storage_reaper_stats().active > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(storage_reaper_stats().active, 0);
    cleanup_temp_db(&path);
}

#[test]
fn worker_panic_is_caught_and_accounted() {
    reset_reaper_registry_for_test();
    let path: Arc<str> = "panic_path.sqlite".into();

    let panic_handle = thread::spawn(|| -> Option<scriptbots_storage::StorageWorkerError> {
        panic!("simulated worker panic for test");
    });

    let req_id = handoff_join_only_for_test(
        panic_handle,
        Arc::clone(&path),
        scriptbots_storage::AnalyticsSnapshotProvider::default(),
    );

    let deadline = Instant::now() + Duration::from_secs(3);
    while storage_reaper_stats().active > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }

    assert_eq!(storage_reaper_stats().active, 0);

    let receipts = reaper_accounting_receipts();
    let rec = receipts
        .iter()
        .find(|r| r.request_id == req_id)
        .expect("receipt");
    assert_eq!(rec.state, ReaperReceiptState::Retired);
    match &rec.join_outcome {
        Some(ReaperJoinOutcome::Panicked(msg)) => {
            assert!(msg.contains("simulated worker panic"));
        }
        other => panic!("expected Panicked join outcome, got {other:?}"),
    }
}

#[test]
fn negative_control_skipped_drain_detected() {
    reset_reaper_registry_for_test();
    simulate_negative_skipped_drain_for_test(8888, "stranded_db.sqlite");

    let audit = verify_reaper_accounting();
    match audit {
        Err(ReaperAccountingError::UnretiredRequest {
            request_id,
            path,
            state,
        }) => {
            assert_eq!(request_id, 8888);
            assert_eq!(path, "stranded_db.sqlite");
            assert_eq!(state, ReaperReceiptState::Queued);
        }
        other => panic!("expected UnretiredRequest diagnostic, got {other:?}"),
    }
    reset_reaper_registry_for_test();
}

#[test]
fn negative_control_double_consumption_detected() {
    reset_reaper_registry_for_test();
    let path = temp_db("dbl_cons", 0);
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");
    let req_id = pipeline.handoff_to_reaper().expect("handoff");
    drop(pipeline);

    let deadline = Instant::now() + Duration::from_secs(3);
    while storage_reaper_stats().active > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }

    // Now attempt to consume it a second time
    let err = simulate_negative_double_consumption_for_test(req_id, &path);
    match err {
        Err(ReaperAccountingError::DoubleConsumption { request_id, .. }) => {
            assert_eq!(request_id, req_id);
        }
        other => panic!("expected DoubleConsumption diagnostic, got {other:?}"),
    }

    cleanup_temp_db(&path);
}

#[test]
fn reaper_bounds_mock_free_timeout_handoff_and_eventual_recovery_e2e() {
    reset_reaper_registry_for_test();
    set_reaper_hung_threshold_for_test(Duration::from_millis(40));

    let path_a = temp_db("e2e_a", 1);
    let path_b = temp_db("e2e_b", 2);

    println!(
        "{{\"schema\":\"scriptbots.storage-reaper.evidence.v1\",\"phase\":\"start\",\"path_a\":\"{}\",\"path_b\":\"{}\"}}",
        path_a, path_b
    );

    // 1. Open Pipeline A with short shutdown deadline
    let mut deadlines_a = StorageDeadlines::default();
    deadlines_a.shutdown_ack = Duration::from_millis(50);
    let mut pipeline_a = StoragePipeline::create_unattributed_file_with_thresholds_and_deadlines(
        &path_a,
        1,
        1,
        1,
        1,
        deadlines_a,
    )
    .expect("pipeline a");

    // 2. Pause Worker A to deterministically force timeout handoff
    let guard_a = pipeline_a.pause_worker_for_test().expect("pause worker A");
    let req_id_a1 = pipeline_a.handoff_to_reaper().expect("handoff a1");
    drop(pipeline_a);

    let stats_1 = storage_reaper_stats();
    assert_eq!(stats_1.active, 1);
    assert_eq!(stats_1.queued, 0);

    // Wait for hung threshold
    thread::sleep(Duration::from_millis(60));
    let stats_hung = storage_reaper_stats();
    assert_eq!(stats_hung.active, 1);
    assert_eq!(stats_hung.hung, 1);
    let inv_err = stats_hung.check_invariants().err();

    println!(
        "{{\"schema\":\"scriptbots.storage-reaper.evidence.v1\",\"phase\":\"held_worker\",\"path\":\"{}\",\"request_id\":{},\"active\":{},\"queued\":{},\"hung\":{},\"oldest_active_age_ms\":{},\"fallback_reason\":null,\"receipt_state\":\"Started\",\"join_outcome\":null,\"lease_released\":false,\"first_invariant_failure\":{:?}}}",
        path_a,
        req_id_a1,
        stats_hung.active,
        stats_hung.queued,
        stats_hung.hung,
        stats_hung.oldest_active_age.unwrap().as_millis(),
        inv_err
    );

    // 3. Duplicate timeout on Path A coalesces
    let path_a_arc: Arc<str> = path_a.as_str().into();
    let worker_a2 = thread::spawn(|| None);
    let req_id_a2 = handoff_join_only_for_test(
        worker_a2,
        Arc::clone(&path_a_arc),
        scriptbots_storage::AnalyticsSnapshotProvider::default(),
    );

    let stats_dup = storage_reaper_stats();
    assert_eq!(stats_dup.active, 1);
    assert_eq!(stats_dup.queued, 1);
    assert!(stats_dup.coalesced >= 1);

    println!(
        "{{\"schema\":\"scriptbots.storage-reaper.evidence.v1\",\"phase\":\"duplicate_coalesced\",\"path\":\"{}\",\"request_id\":{},\"active\":{},\"queued\":{},\"hung\":{},\"receipt_state\":\"Coalesced\"}}",
        path_a, req_id_a2, stats_dup.active, stats_dup.queued, stats_dup.hung
    );

    // 4. Cross-path independence on Path B
    let mut pipeline_b =
        StoragePipeline::create_unattributed_file_with_thresholds(&path_b, 1, 1, 1, 1)
            .expect("pipeline b");
    pipeline_b.shutdown().expect("b shuts down cleanly");
    cleanup_temp_db(&path_b);

    println!(
        "{{\"schema\":\"scriptbots.storage-reaper.evidence.v1\",\"phase\":\"cross_path_progress\",\"path\":\"{}\",\"status\":\"healthy_and_independent\"}}",
        path_b
    );

    // 5. Release held worker on Path A
    guard_a.release();

    let deadline = Instant::now() + Duration::from_secs(3);
    while storage_reaper_stats().active > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }

    let stats_final = storage_reaper_stats();
    assert_eq!(stats_final.active, 0);
    assert_eq!(stats_final.queued, 0);
    assert_eq!(stats_final.hung, 0);
    let inv_err_final = stats_final.check_invariants().err();

    verify_reaper_accounting().expect("all receipts accounted");

    // 6. Safe reopen of exact path A
    cleanup_temp_db(&path_a);
    let mut reopened_a =
        StoragePipeline::create_unattributed_file_with_thresholds(&path_a, 1, 1, 1, 1)
            .expect("exact path A reopens safely after reap");
    reopened_a
        .shutdown()
        .expect("reopened a shuts down cleanly");

    println!(
        "{{\"schema\":\"scriptbots.storage-reaper.evidence.v1\",\"phase\":\"recovery_and_reopen\",\"path\":\"{}\",\"lease_released\":true,\"reopen_success\":true,\"first_invariant_failure\":{:?}}}",
        path_a, inv_err_final
    );

    cleanup_temp_db(&path_a);
}
