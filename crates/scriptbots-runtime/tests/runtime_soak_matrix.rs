//! Runtime lifecycle, contention, journal fault-injection, and soak matrix (bd-2z0.4.11).
//!
//! Verifies `HostCore`, `FixedDeadlineHost` native driver, `HostClient`, `SnapshotHub` subscribers,
//! `EventHub` cursors, and fault-injected storage adapters under multi-threaded contention,
//! lifecycle transitions, backpressure/fault injection, and deterministic soak invariance.

use scriptbots_core::{
    AgentData, NullPersistence, ScriptBotsConfig, SelectionMode, SelectionState, SelectionUpdate,
    Tick, WorldDigestV1, WorldState,
};
use scriptbots_runtime::channel::{
    ChannelDriveError, ChannelHostDriver, ChannelHostOptions, ChannelHostPort, ChannelRunOutcome,
    ChannelRunReceipt, ChannelStepReport,
};
use scriptbots_runtime::{
    ApplicationState, CommandEnvelope, CommandId, CommandStatus, EventPoll, FixedDeadlineHost,
    HostBlocker, HostClient, HostCommand, HostCore, HostCoreOptions, HostHealth, HostPort,
    HostSessionId, JournalAdmission, JournalBatch, JournalFailure, JournalPort, JournalReceipt,
    JournalReceiptState, ManualInstant, PlaybackSnapshot, ShutdownCommitRequirement,
    SnapshotSubscription, VolatileJournal,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

/// Structured retained telemetry evidence for bd-2z0.4.11.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSoakEvidence {
    /// Name of the executed soak test.
    pub test_name: String,
    /// Bead issue identity.
    pub bead_id: String,
    /// Machine architecture class.
    pub machine_arch: String,
    /// Operating system class.
    pub machine_os: String,
    /// Duration of this soak run and its reference comparison in milliseconds.
    pub duration_ms: u64,
    /// One-based index of this independently observed soak run.
    pub run_index: u8,
    /// Requested minimum completed tick.
    pub target_tick: u64,
    /// Actual owner-world tick after shutdown, starting from tick zero.
    pub total_ticks_advanced: u64,
    /// Total commands submitted by client threads.
    pub commands_submitted: u64,
    /// Driver-reported admissions, including the owner-disconnect shutdown command.
    pub commands_admitted: u64,
    /// Client commands whose terminal status was observed as applied.
    pub commands_applied: u64,
    /// Total snapshot frames polled by subscribers.
    pub snapshots_polled: u64,
    /// Event-cursor count when observed; the digest soak does not poll events.
    pub events_consumed: Option<u64>,
    /// Canonical digest captured on the owner thread before its first drive.
    pub initial_digest: String,
    /// Canonical world digest after soak completion.
    pub final_digest: String,
    /// Digest of an identically initialized reference advanced to the actual final tick.
    pub reference_digest: String,
    /// Peak command queue depth observed under contention.
    pub peak_queue_depth: usize,
}

fn build_test_world(seed: u64) -> WorldState {
    let config = ScriptBotsConfig {
        rng_seed: Some(seed),
        population_minimum: 20,
        population_spawn_interval: 1,
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("deterministic test world");
    for _ in 0..20 {
        world
            .try_spawn_agent(AgentData::default())
            .expect("seed actual soak/lifecycle agents");
    }
    world
}

/// Helper to build a deterministic test host with live agents.
fn build_test_host(seed: u64, paused: bool) -> FixedDeadlineHost {
    let world = build_test_world(seed);
    let options = HostCoreOptions {
        initial_playback: PlaybackSnapshot {
            paused,
            ..PlaybackSnapshot::default()
        },
        tick_period_nanos: 5_000_000, // 5ms per tick
        snapshot_interval_ticks: 1,
        capture_agent_visuals: true,
        ..HostCoreOptions::default()
    };
    let core = HostCore::new(HostSessionId::new(seed), world, options).expect("host core builds");
    FixedDeadlineHost::new(core)
}

/// Fast options for cross-thread channel ports in tests.
const fn fast_channel_options() -> ChannelHostOptions {
    ChannelHostOptions {
        ingress_capacity: 256,
        ingress_drain_budget: 128,
        status_board_capacity: 1024,
        protocol_event_capacity: 1024,
        submit_deadline: Duration::from_secs(5),
        maintenance_period: Duration::from_millis(2),
    }
}

fn elapsed_instant(start: Instant) -> ManualInstant {
    ManualInstant::from_nanos(
        u64::try_from(start.elapsed().as_nanos()).expect("test elapsed nanoseconds fit u64"),
    )
}

/// Helper to spawn the driver on a dedicated simulation owner thread.
fn spawn_driver(
    seed: u64,
    paused: bool,
    options: ChannelHostOptions,
) -> (
    thread::JoinHandle<Result<ChannelRunReceipt, ChannelDriveError>>,
    ChannelHostPort,
) {
    let (port_tx, port_rx) = std::sync::mpsc::channel();
    let worker = thread::spawn(move || {
        let host = build_test_host(seed, paused);
        let (mut driver, port) = ChannelHostDriver::new(host, options).expect("driver builds");
        port_tx.send(port).expect("port handoff");
        let start = Instant::now();
        driver.run(move || elapsed_instant(start))
    });
    let port = port_rx.recv().expect("port handoff");
    (worker, port)
}

/// Wait until a submitted command resolves to a terminal application state.
fn wait_resolved<P: HostPort>(client: &mut HostClient<P>, command_id: CommandId) -> CommandStatus {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = client.command_status(command_id).expect("status lookup")
            && !matches!(status.application(), ApplicationState::Admitted)
        {
            return status;
        }
        assert!(
            Instant::now() <= deadline,
            "timed out waiting for command {command_id:?} to resolve"
        );
        thread::sleep(Duration::from_millis(2));
    }
}

fn assert_clean_shutdown(receipt: &ChannelRunReceipt) {
    assert!(
        matches!(
            receipt.outcome,
            ChannelRunOutcome::ControllerDisconnected | ChannelRunOutcome::Stopped
        ),
        "expected clean termination, got {:?}",
        receipt.outcome
    );
}

// ---------------------------------------------------------------------------
// 1. Integrated Lifecycle Matrix Test
// ---------------------------------------------------------------------------

#[test]
fn test_runtime_lifecycle_matrix() {
    let start_time = Instant::now();
    let (driver_handle, port) = spawn_driver(0x11fe_0001, true, fast_channel_options());
    let mut client = HostClient::new(port);

    // 1. Startup verification: host starts paused at tick 0
    let mut sub = client.subscribe_snapshots();
    let initial_snap = client
        .poll_snapshot(&mut sub)
        .expect("poll initial snapshot")
        .expect("initial snapshot present");
    assert!(initial_snap.playback.paused);
    assert_eq!(initial_snap.completed_summary, None); // no completed ticks yet

    // 2. Single-step execution: Step advances exactly one tick per command
    let step_cmd_1 = CommandId::from_client_sequence(0x101, 1);
    client
        .submit(CommandEnvelope::new(step_cmd_1, HostCommand::Step))
        .expect("submit step 1");
    wait_resolved(&mut client, step_cmd_1);

    let snap_step_1 = client
        .poll_snapshot(&mut sub)
        .expect("poll snapshot after step 1")
        .expect("snapshot step 1 present");
    assert_eq!(
        snap_step_1.completed_summary.as_ref().map(|s| s.tick),
        Some(Tick(1))
    );
    assert!(snap_step_1.playback.paused);

    let step_cmd_2 = CommandId::from_client_sequence(0x101, 2);
    client
        .submit(CommandEnvelope::new(step_cmd_2, HostCommand::Step))
        .expect("submit step 2");
    wait_resolved(&mut client, step_cmd_2);

    let snap_step_2 = client
        .poll_snapshot(&mut sub)
        .expect("poll snapshot after step 2")
        .expect("snapshot step 2 present");
    assert_eq!(
        snap_step_2.completed_summary.as_ref().map(|s| s.tick),
        Some(Tick(2))
    );

    // 3. Resume: automatic ticking
    let resume_cmd = CommandId::from_client_sequence(0x101, 3);
    client
        .submit(CommandEnvelope::new(resume_cmd, HostCommand::Resume))
        .expect("submit resume");
    wait_resolved(&mut client, resume_cmd);

    // Wait briefly for automatic steps to accumulate
    thread::sleep(Duration::from_millis(50));
    let snap_resumed = client
        .poll_snapshot(&mut sub)
        .expect("poll snapshot after resume")
        .expect("snapshot resumed present");
    assert!(!snap_resumed.playback.paused);
    let current_tick = snap_resumed
        .completed_summary
        .as_ref()
        .map_or(0, |s| s.tick.0);
    assert!(
        current_tick > 2,
        "resumed world must advance beyond tick 2, tick={current_tick}"
    );

    // 4. SetSpeed dynamic reconfiguration
    let speed_cmd = CommandId::from_client_sequence(0x101, 4);
    client
        .submit(CommandEnvelope::new(speed_cmd, HostCommand::SetSpeed(3.0)))
        .expect("submit set speed");
    wait_resolved(&mut client, speed_cmd);

    let snap_speed = client
        .poll_snapshot(&mut sub)
        .expect("poll snapshot after speed")
        .expect("snapshot speed present");
    assert!((snap_speed.playback.speed_multiplier - 3.0).abs() < 1e-4);

    // 5. UpdateSelection command
    let select_cmd = CommandId::from_client_sequence(0x101, 5);
    client
        .submit(CommandEnvelope::new(
            select_cmd,
            HostCommand::UpdateSelection(SelectionUpdate {
                mode: SelectionMode::Replace,
                agent_ids: vec![1, 2],
                state: SelectionState::Selected,
            }),
        ))
        .expect("submit selection update");
    wait_resolved(&mut client, select_cmd);

    // 6. Pause again
    let pause_cmd = CommandId::from_client_sequence(0x101, 6);
    client
        .submit(CommandEnvelope::new(pause_cmd, HostCommand::Pause))
        .expect("submit pause");
    wait_resolved(&mut client, pause_cmd);

    let snap_paused = client
        .poll_snapshot(&mut sub)
        .expect("poll snapshot after pause")
        .expect("snapshot paused present");
    assert!(snap_paused.playback.paused);

    // 7. Structured shutdown: drop client port, driver completes gracefully
    drop(client);
    let receipt = driver_handle
        .join()
        .expect("driver thread joins cleanly")
        .expect("driver run succeeds");
    assert_clean_shutdown(&receipt);
    assert!(receipt.commands_admitted >= 6);
    assert!(receipt.drives > 0);

    eprintln!(
        "[bd-2z0.4.11] test_runtime_lifecycle_matrix passed in {:?}",
        start_time.elapsed()
    );
}

// ---------------------------------------------------------------------------
// 2. High-Throughput Contention Storm Matrix Test
// ---------------------------------------------------------------------------

fn spawn_snapshot_subscriber(
    port: ChannelHostPort,
    running: Arc<AtomicBool>,
    snap_counter: Arc<AtomicU64>,
) -> thread::JoinHandle<u64> {
    thread::spawn(move || {
        let mut client = HostClient::new(port);
        let mut sub = client.subscribe_snapshots();
        let mut prev_scientific_rev = 0_u64;
        let mut polled_count = 0_u64;

        while running.load(Ordering::Relaxed) {
            if let Ok(Some(snap)) = client.poll_snapshot(&mut sub) {
                let rev = snap.revision.get();
                assert!(
                    rev >= prev_scientific_rev,
                    "snapshot revision regressed: prev={prev_scientific_rev}, new={rev}"
                );
                prev_scientific_rev = rev;
                polled_count += 1;
                snap_counter.fetch_add(1, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_millis(1));
        }
        polled_count
    })
}

fn spawn_event_consumer(
    port: ChannelHostPort,
    running: Arc<AtomicBool>,
    event_counter: Arc<AtomicU64>,
) -> thread::JoinHandle<u64> {
    thread::spawn(move || {
        let mut client = HostClient::new(port);
        let mut cursor = client.event_cursor();
        let mut consumed = 0_u64;

        while running.load(Ordering::Relaxed) {
            if let Ok(EventPoll::Contiguous(page)) = client.read_events(&mut cursor, 16) {
                let count = page.events.len() as u64;
                consumed += count;
                event_counter.fetch_add(count, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_millis(2));
        }
        consumed
    })
}

#[test]
fn test_runtime_contention_storm_matrix() {
    let start_time = Instant::now();
    let (driver_handle, main_port) = spawn_driver(0x2024_cafe, false, fast_channel_options());

    let running = Arc::new(AtomicBool::new(true));
    let total_submitted = Arc::new(AtomicU64::new(0));
    let total_snapshots = Arc::new(AtomicU64::new(0));
    let total_events = Arc::new(AtomicU64::new(0));

    // Spawn 4 concurrent command worker threads
    let mut client_handles = Vec::new();
    for thread_idx in 0..4_u64 {
        let port_clone = main_port.clone();
        let running_clone = Arc::clone(&running);
        let submitted_counter = Arc::clone(&total_submitted);

        let handle = thread::spawn(move || {
            let mut client = HostClient::new(port_clone);
            let mut seq = 0_u64;
            while running_clone.load(Ordering::Relaxed) && seq < 300 {
                seq += 1;
                let command_id = CommandId::from_client_sequence(0x5700_0000 + thread_idx, seq);
                let cmd = match (thread_idx + seq) % 5 {
                    0 => HostCommand::Step,
                    1 => HostCommand::Pause,
                    2 => HostCommand::Resume,
                    3 => HostCommand::SetSpeed(
                        0.5 + f32::from(u8::try_from(seq % 4).expect("remainder is in 0..4")),
                    ),
                    _ => HostCommand::UpdateSelection(SelectionUpdate {
                        mode: SelectionMode::Clear,
                        agent_ids: vec![],
                        state: SelectionState::Selected,
                    }),
                };
                if client.submit(CommandEnvelope::new(command_id, cmd)).is_ok() {
                    submitted_counter.fetch_add(1, Ordering::Relaxed);
                }
                thread::sleep(Duration::from_micros(500));
            }
            seq
        });
        client_handles.push(handle);
    }

    // Spawn 2 concurrent snapshot subscriber threads
    let mut subscriber_handles = Vec::new();
    for _ in 0..2 {
        subscriber_handles.push(spawn_snapshot_subscriber(
            main_port.clone(),
            Arc::clone(&running),
            Arc::clone(&total_snapshots),
        ));
    }

    // Spawn 2 concurrent event consumer threads
    let mut event_handles = Vec::new();
    for _ in 0..2 {
        event_handles.push(spawn_event_consumer(
            main_port.clone(),
            Arc::clone(&running),
            Arc::clone(&total_events),
        ));
    }

    // Allow contention storm to execute for 1.5 seconds
    thread::sleep(Duration::from_millis(1500));

    // Signal all workers to conclude
    running.store(false, Ordering::Relaxed);

    // Join client workers
    for handle in client_handles {
        let _ = handle.join().expect("client worker thread joins cleanly");
    }

    // Join snapshot subscribers
    for handle in subscriber_handles {
        let _ = handle.join().expect("subscriber thread joins cleanly");
    }

    // Join event consumers
    for handle in event_handles {
        let _ = handle.join().expect("event consumer thread joins cleanly");
    }

    // Drop main port to trigger graceful driver disconnect and shutdown
    drop(main_port);

    let receipt = driver_handle
        .join()
        .expect("driver thread joins cleanly")
        .expect("driver runs successfully");

    let submitted = total_submitted.load(Ordering::Relaxed);
    let snapshots = total_snapshots.load(Ordering::Relaxed);
    let events = total_events.load(Ordering::Relaxed);

    assert!(
        submitted > 100,
        "expected >100 commands submitted, got {submitted}"
    );
    assert!(
        receipt.commands_admitted > 50,
        "expected >50 admitted, got {}",
        receipt.commands_admitted
    );
    assert!(
        snapshots > 10,
        "expected >10 snapshots polled, got {snapshots}"
    );

    eprintln!(
        "[bd-2z0.4.11] Contention storm verified: submitted={submitted}, admitted={}, snapshots={snapshots}, events={events} in {:?}",
        receipt.commands_admitted,
        start_time.elapsed()
    );
}

// ---------------------------------------------------------------------------
// 3. Storage & Journal Fault-Injection Matrix Test
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct FaultInjectedJournal {
    inner: VolatileJournal,
    mode: Arc<AtomicU8>, // 0: Normal, 1: RejectAdmissionFull, 2: RejectAdmissionClosed, 3: DelayReceipts, 4: FailReceipts
    delayed_receipts: Mutex<VecDeque<JournalReceipt>>,
}

impl FaultInjectedJournal {
    fn new(mode: Arc<AtomicU8>) -> Self {
        Self {
            inner: VolatileJournal::default(),
            mode,
            delayed_receipts: Mutex::new(VecDeque::new()),
        }
    }
}

impl JournalPort for FaultInjectedJournal {
    fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
        match self.mode.load(Ordering::SeqCst) {
            1 => JournalAdmission::Full {
                batch_id: batch.id(),
                capacity: 0,
            },
            2 => JournalAdmission::Closed {
                batch_id: batch.id(),
            },
            _ => self.inner.try_admit(batch),
        }
    }

    fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
        match self.mode.load(Ordering::SeqCst) {
            3 => {
                // Delay receipts: pull from inner and hold back
                let inner_receipts = self.inner.poll_receipts(limit);
                if let Ok(mut delayed) = self.delayed_receipts.lock() {
                    delayed.extend(inner_receipts);
                }
                Vec::new()
            }
            4 => {
                // Fail receipts: emit terminal failed state
                let inner_receipts = self.inner.poll_receipts(limit);
                inner_receipts
                    .into_iter()
                    .map(|r| {
                        JournalReceipt::new(
                            r.batch_id(),
                            JournalReceiptState::Failed(JournalFailure {
                                code: "synthetic_fault".to_owned(),
                                message: "injected disk failure".to_owned(),
                            }),
                        )
                    })
                    .collect()
            }
            _ => {
                let mut out = Vec::new();
                if let Ok(mut delayed) = self.delayed_receipts.lock() {
                    while out.len() < limit && !delayed.is_empty() {
                        if let Some(r) = delayed.pop_front() {
                            out.push(r);
                        }
                    }
                }
                if out.len() < limit {
                    let remaining = limit - out.len();
                    out.extend(self.inner.poll_receipts(remaining));
                }
                out
            }
        }
    }

    fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
        ShutdownCommitRequirement::CommittedVolatile
    }
}

fn assert_drive_completed_at_tick(
    report: &ChannelStepReport,
    driver: &ChannelHostDriver,
    tick: Tick,
) {
    assert!(report.drove);
    assert_eq!(driver.host().core().world_tick(), tick);
}

fn build_fault_test_host(fault_mode: Arc<AtomicU8>) -> FixedDeadlineHost {
    let journal = Box::new(FaultInjectedJournal::new(fault_mode));
    let config = ScriptBotsConfig {
        rng_seed: Some(0xfa01_7001),
        population_minimum: 10,
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("deterministic world");
    for _ in 0..10 {
        world
            .try_spawn_agent(AgentData::default())
            .expect("seed actual fault-matrix agents");
    }

    let options = HostCoreOptions {
        initial_playback: PlaybackSnapshot {
            paused: false,
            ..PlaybackSnapshot::default()
        },
        tick_period_nanos: 10_000_000,
        snapshot_interval_ticks: 1,
        ..HostCoreOptions::default()
    };

    let core = HostCore::with_journal(HostSessionId::new(0xfa01_7001), world, options, journal)
        .expect("host core with fault journal builds");
    FixedDeadlineHost::new(core)
}

#[test]
fn test_runtime_journal_fault_injection_matrix() {
    let start_time = Instant::now();
    let fault_mode = Arc::new(AtomicU8::new(0)); // Start Normal
    let host = build_fault_test_host(Arc::clone(&fault_mode));

    let (mut driver, port) =
        ChannelHostDriver::new(host, fast_channel_options()).expect("driver builds");
    let client = HostClient::new(port);

    let mut verified_faults = Vec::new();

    // 0. Establish epoch at t=0
    let _ = driver
        .step(ManualInstant::from_nanos(0))
        .expect("epoch step");
    assert_eq!(driver.host().core().world_tick(), Tick(0));

    // 1. Clean step crossing the first deadline at t=10_000_000
    let step_1 = driver
        .step(ManualInstant::from_nanos(10_000_000))
        .expect("initial step");
    assert_drive_completed_at_tick(&step_1, &driver, Tick(1));

    // 2. FAULT INJECTION A: RejectAdmissionFull (Queue backpressure)
    fault_mode.store(1, Ordering::SeqCst);
    let _ = driver.step(ManualInstant::from_nanos(20_000_000));
    // The world stepped to Tick(2), but its batch hit JournalFull, so it became retained and blocked
    assert_eq!(driver.host().core().world_tick(), Tick(2));
    assert!(matches!(
        driver.host().core().health(),
        HostHealth::Blocked(HostBlocker::JournalFull { .. })
    ));
    verified_faults.push("admission_full_backpressure".to_owned());
    let retained = driver
        .host()
        .core()
        .pending_journal_batch()
        .expect("retained batch");

    // While blocked, elapsed time cannot bypass retained journal work:
    let _ = driver.step(ManualInstant::from_nanos(25_000_000));
    assert_eq!(driver.host().core().world_tick(), Tick(2));
    assert!(Arc::ptr_eq(
        &retained,
        &driver
            .host()
            .core()
            .pending_journal_batch()
            .expect("a refused automatic retry keeps the same allocation")
    ));

    // Restore admission, then let the production channel perform its bounded retry.
    // HostCore discards elapsed time at blocked boundaries. The recovery boundary
    // therefore admits the retained tick, and a full later period earns the next tick.
    fault_mode.store(0, Ordering::SeqCst);
    let retry = driver
        .step(ManualInstant::from_nanos(30_000_000))
        .expect("automatic retry");
    assert_drive_completed_at_tick(&retry, &driver, Tick(2));
    assert!(driver.host().core().pending_journal_batch().is_none());
    assert!(!matches!(
        driver.host().core().health(),
        HostHealth::Blocked(HostBlocker::JournalFull { .. })
    ));

    // One full automatic period after recovery advances exactly one tick.
    let step_recovered = driver
        .step(ManualInstant::from_nanos(40_000_000))
        .expect("step after recovery");
    assert_drive_completed_at_tick(&step_recovered, &driver, Tick(3));
    verified_faults.push("admission_recovery".to_owned());

    // 3. FAULT INJECTION B: DelayReceipts
    fault_mode.store(3, Ordering::SeqCst);
    let step_delayed = driver
        .step(ManualInstant::from_nanos(50_000_000))
        .expect("step with delayed receipts");
    assert_drive_completed_at_tick(&step_delayed, &driver, Tick(4));
    // The host advances science even while receipts are delayed
    verified_faults.push("delayed_receipts_tolerance".to_owned());

    // Release delayed receipts
    fault_mode.store(0, Ordering::SeqCst);
    let step_caught_up = driver
        .step(ManualInstant::from_nanos(60_000_000))
        .expect("step catching up receipts");
    assert_drive_completed_at_tick(&step_caught_up, &driver, Tick(5));
    verified_faults.push("delayed_receipts_catchup".to_owned());

    // 4. FAULT INJECTION C: FailReceipts (Terminal Storage Failure)
    fault_mode.store(4, Ordering::SeqCst);
    let _ = driver.step(ManualInstant::from_nanos(70_000_000));
    // Host transitions to Faulted health when journal receipt fails
    let health = driver.host().core().health();
    assert!(
        matches!(health, HostHealth::Faulted(_)),
        "expected faulted health, got {health:?}"
    );
    verified_faults.push("terminal_receipt_failure_latch".to_owned());

    drop(client);
    eprintln!(
        "[bd-2z0.4.11] Fault injection matrix verified: {verified_faults:?} in {:?}",
        start_time.elapsed()
    );
}

// ---------------------------------------------------------------------------
// 4. Deterministic Soak & Bit-Exact Digest Invariance Test
// ---------------------------------------------------------------------------

struct OwnerSoakOutcome {
    initial_digest: WorldDigestV1,
    final_digest: WorldDigestV1,
    world_tick: Tick,
    receipt: ChannelRunReceipt,
}

#[derive(Default)]
struct SoakProgress {
    snapshots_seen: u64,
    max_queue_depth: usize,
    observed_tick: u64,
    commands_submitted: u64,
    commands_applied: u64,
}

fn spawn_observed_soak(seed: u64) -> (thread::JoinHandle<OwnerSoakOutcome>, ChannelHostPort) {
    let (port_tx, port_rx) = std::sync::mpsc::channel();
    let worker = thread::spawn(move || {
        let host = build_test_host(seed, false);
        let initial_digest = host
            .core()
            .world()
            .world_digest_v1()
            .expect("initial owner digest");
        assert_eq!(initial_digest.tick, Tick(0));
        let (mut driver, port) =
            ChannelHostDriver::new(host, fast_channel_options()).expect("soak driver builds");
        port_tx.send(port).expect("soak port handoff");
        let start = Instant::now();
        let receipt = driver
            .run(move || elapsed_instant(start))
            .expect("soak driver run");
        let world = driver.host().core().world();
        OwnerSoakOutcome {
            initial_digest,
            final_digest: world.world_digest_v1().expect("stopped owner digest"),
            world_tick: world.tick(),
            receipt,
        }
    });
    let port = port_rx.recv().expect("soak port handoff");
    (worker, port)
}

fn poll_soak_snapshot(
    client: &mut HostClient<ChannelHostPort>,
    sub: &mut SnapshotSubscription,
    progress: &mut SoakProgress,
) {
    if let Some(snap) = client.poll_snapshot(sub).expect("soak snapshot lookup") {
        progress.snapshots_seen += 1;
        progress.max_queue_depth = progress.max_queue_depth.max(snap.command_queue_depth);
        if let Some(summary) = &snap.completed_summary {
            assert!(
                summary.tick.0 >= progress.observed_tick,
                "soak tick regressed"
            );
            progress.observed_tick = summary.tick.0;
        }
    }
}

fn submit_soak_command(
    client: &mut HostClient<ChannelHostPort>,
    sequence: u64,
    command: HostCommand,
    progress: &mut SoakProgress,
) -> CommandId {
    let command_id = CommandId::from_client_sequence(0x50a4, sequence);
    progress.commands_submitted += 1;
    let status = client
        .submit(CommandEnvelope::new(command_id, command))
        .expect("soak command submission");
    assert!(
        status.admission_sequence().is_some(),
        "soak command was refused: {status:?}"
    );
    command_id
}

fn wait_soak_applied(
    client: &mut HostClient<ChannelHostPort>,
    command_id: CommandId,
    progress: &mut SoakProgress,
) {
    let status = wait_resolved(client, command_id);
    assert!(
        matches!(status.application(), ApplicationState::Applied(_)),
        "soak command did not apply: {status:?}"
    );
    progress.commands_applied += 1;
}

fn run_observed_soak(seed: u64, target_tick: u64) -> (OwnerSoakOutcome, SoakProgress) {
    let (driver_handle, port) = spawn_observed_soak(seed);
    let mut client = HostClient::new(port);
    let mut sub = client.subscribe_snapshots();
    let mut progress = SoakProgress::default();
    let mut submitted = Vec::new();

    // Keep the original command sequence and spacing, including its five Steps.
    for i in 1..=20_u64 {
        let command = match i % 4 {
            0 => HostCommand::SetSpeed(1.2),
            1 => HostCommand::SetSpeed(1.0),
            2 => HostCommand::UpdateSelection(SelectionUpdate {
                mode: SelectionMode::Clear,
                agent_ids: vec![],
                state: SelectionState::Selected,
            }),
            _ => HostCommand::Step,
        };
        submitted.push(submit_soak_command(&mut client, i, command, &mut progress));
        thread::sleep(Duration::from_millis(2));
        poll_soak_snapshot(&mut client, &mut sub, &mut progress);
    }
    for command_id in submitted {
        wait_soak_applied(&mut client, command_id, &mut progress);
    }

    // Step pauses playback. Explicitly resume automatic ticking before the target wait.
    let resume = submit_soak_command(&mut client, 998, HostCommand::Resume, &mut progress);
    wait_soak_applied(&mut client, resume, &mut progress);
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        poll_soak_snapshot(&mut client, &mut sub, &mut progress);
        if progress.observed_tick >= target_tick {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }

    let pause = submit_soak_command(&mut client, 999, HostCommand::Pause, &mut progress);
    wait_soak_applied(&mut client, pause, &mut progress);
    drop(client);
    let outcome = driver_handle.join().expect("soak owner joins");
    assert_clean_shutdown(&outcome.receipt);
    assert_eq!(
        outcome.receipt.outcome,
        ChannelRunOutcome::ControllerDisconnected
    );
    assert_eq!(progress.commands_applied, progress.commands_submitted);
    assert_eq!(
        outcome.receipt.commands_admitted,
        progress.commands_submitted + 1,
        "driver admissions include every applied client command and the disconnect shutdown"
    );
    (outcome, progress)
}

fn validate_soak_progress(
    initial: &WorldDigestV1,
    final_digest: &WorldDigestV1,
    owner_tick: Tick,
    observed_tick: u64,
    target_tick: u64,
) -> Result<(), &'static str> {
    if observed_tick < target_tick {
        return Err("snapshot target was not reached within the soak deadline");
    }
    if owner_tick.0 < observed_tick {
        return Err("owner tick precedes the observed snapshot tick");
    }
    if final_digest.tick != owner_tick {
        return Err("final digest does not describe the observed owner tick");
    }
    if final_digest.overall == initial.overall {
        return Err("final digest still describes the initial world");
    }
    Ok(())
}

fn reference_digest_at_tick(seed: u64, tick: Tick) -> WorldDigestV1 {
    let mut world = build_test_world(seed);
    let mut session = world
        .bind_persistence(Box::new(NullPersistence))
        .expect("reference session");
    for _ in 0..tick.0 {
        session.step(&mut world).expect("reference scientific step");
    }
    session
        .finalize(&mut world)
        .expect("reference shutdown boundary");
    assert_eq!(world.tick(), tick);
    world.world_digest_v1().expect("advanced reference digest")
}

fn assert_soak_digest_matches_reference(digest_1: &WorldDigestV1, digest_2: &WorldDigestV1) {
    assert_eq!(
        digest_1.overall, digest_2.overall,
        "soaked owner and same-tick reference must have bit-exact overall digest: {} vs {}",
        digest_1.overall, digest_2.overall
    );
    assert_eq!(
        digest_1.agents, digest_2.agents,
        "soaked owner and same-tick reference must have bit-exact agent digest: {} vs {}",
        digest_1.agents, digest_2.agents
    );
    assert_eq!(
        digest_1.food, digest_2.food,
        "soaked owner and same-tick reference must have bit-exact food digest: {} vs {}",
        digest_1.food, digest_2.food
    );
    assert_eq!(
        digest_1.hydrology, digest_2.hydrology,
        "soaked owner and same-tick reference must have bit-exact hydrology digest: {:?} vs {:?}",
        digest_1.hydrology, digest_2.hydrology
    );
    assert_eq!(
        digest_1, digest_2,
        "every scientific digest lane must match"
    );
}

#[test]
fn test_runtime_soak_and_digest_invariance() {
    let soak_seed = 0x50a4_c0de_u64;
    let soak_ticks = 100_u64;
    for run_index in 1..=2 {
        let start_time = Instant::now();
        let (outcome, progress) = run_observed_soak(soak_seed, soak_ticks);
        validate_soak_progress(
            &outcome.initial_digest,
            &outcome.final_digest,
            outcome.world_tick,
            progress.observed_tick,
            soak_ticks,
        )
        .expect("soak reached its target and captured the evolved owner world");

        // Wall-clock runs may stop at different ticks. Compare each actual world
        // against identical initialization and scientific steps to its observed tick.
        let reference = reference_digest_at_tick(soak_seed, outcome.world_tick);
        assert_soak_digest_matches_reference(&outcome.final_digest, &reference);
        let evidence = RuntimeSoakEvidence {
            test_name: "test_runtime_soak_and_digest_invariance".to_owned(),
            bead_id: "bd-2z0.4.11".to_owned(),
            machine_arch: std::env::consts::ARCH.to_owned(),
            machine_os: std::env::consts::OS.to_owned(),
            duration_ms: u64::try_from(start_time.elapsed().as_millis())
                .expect("test milliseconds fit u64"),
            run_index,
            target_tick: soak_ticks,
            total_ticks_advanced: outcome.world_tick.0,
            commands_submitted: progress.commands_submitted,
            commands_admitted: outcome.receipt.commands_admitted,
            commands_applied: progress.commands_applied,
            snapshots_polled: progress.snapshots_seen,
            events_consumed: None,
            initial_digest: outcome.initial_digest.overall,
            final_digest: outcome.final_digest.overall,
            reference_digest: reference.overall,
            peak_queue_depth: progress.max_queue_depth,
        };
        let evidence_json = serde_json::to_string_pretty(&evidence).expect("serialize evidence");
        eprintln!("[bd-2z0.4.11] RETAINED SOAK EVIDENCE:\n{evidence_json}");
    }
}

#[test]
fn soak_progress_rejects_unticked_digest_and_unreached_target() {
    let seed = 0x50a4_c0de_u64;
    let initial = build_test_world(seed)
        .world_digest_v1()
        .expect("initial digest");
    let evolved = reference_digest_at_tick(seed, Tick(2));
    assert_eq!(
        validate_soak_progress(&initial, &evolved, Tick(2), 2, 2),
        Ok(())
    );
    assert_eq!(
        validate_soak_progress(&initial, &initial, Tick(2), 2, 2),
        Err("final digest does not describe the observed owner tick"),
        "substituting the old unticked reference must fail the live soak's validator"
    );
    assert_eq!(
        validate_soak_progress(&initial, &initial, Tick(0), 0, 0),
        Err("final digest still describes the initial world")
    );
    assert_eq!(
        validate_soak_progress(&initial, &evolved, Tick(2), 2, 3),
        Err("snapshot target was not reached within the soak deadline")
    );
    assert_eq!(
        validate_soak_progress(&initial, &evolved, Tick(1), 2, 2),
        Err("owner tick precedes the observed snapshot tick")
    );
}
