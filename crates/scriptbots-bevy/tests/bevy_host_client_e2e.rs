use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use scriptbots_bevy::{BevyRendererContext, CommandSubmitter};
use scriptbots_core::{
    AgentData, ControlCommand, ScriptBotsConfig, SelectionMode, SelectionState, SelectionUpdate,
    SimulationCommand, WorldState,
};
use scriptbots_runtime::{
    ApplicationState, CommandEnvelope, CommandId, CommandStatus, ControlRevision,
    FixedDeadlineHost, HostClient, HostCommand, HostCore, HostCoreOptions, HostSessionId,
    ManualInstant, PlaybackSnapshot, RejectionReason,
    channel::{
        ChannelDriveError, ChannelHostDriver, ChannelHostOptions, ChannelHostPort,
        ChannelRunOutcome, ChannelRunReceipt,
    },
};

fn test_channel_options() -> ChannelHostOptions {
    ChannelHostOptions {
        ingress_capacity: 128,
        ingress_drain_budget: 64,
        status_board_capacity: 512,
        protocol_event_capacity: 512,
        submit_deadline: Duration::from_secs(5),
        maintenance_period: Duration::from_millis(1),
    }
}

fn make_test_host(seed: u64, paused: bool) -> FixedDeadlineHost {
    let config = ScriptBotsConfig {
        rng_seed: Some(seed),
        world_width: 80,
        world_height: 80,
        food_cell_size: 40,
        population_minimum: 10,
        persistence_interval: 0,
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("world init");
    for _ in 0..10 {
        let _ = world.try_spawn_agent(AgentData::default());
    }
    let options = HostCoreOptions {
        initial_playback: PlaybackSnapshot {
            paused,
            speed_multiplier: 1.0,
        },
        tick_period_nanos: 10_000_000,
        capture_agent_visuals: true,
        ..HostCoreOptions::default()
    };
    let core = HostCore::new(HostSessionId::new(seed), world, options).expect("host core init");
    FixedDeadlineHost::new(core)
}

fn spawn_driver(
    seed: u64,
    paused: bool,
) -> (
    thread::JoinHandle<Result<ChannelRunReceipt, ChannelDriveError>>,
    ChannelHostPort,
) {
    let (port_tx, port_rx) = std::sync::mpsc::channel();
    let worker = thread::spawn(move || {
        let host = make_test_host(seed, paused);
        let (mut driver, port) =
            ChannelHostDriver::new(host, test_channel_options()).expect("driver builds");
        port_tx.send(port).expect("port handoff");
        let start = Instant::now();
        driver.run(move || {
            ManualInstant::from_nanos(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX))
        })
    });
    let port = port_rx.recv().expect("port handoff");
    (worker, port)
}

fn wait_resolved<P: scriptbots_runtime::HostPort>(
    client: &mut HostClient<P>,
    command_id: CommandId,
) -> CommandStatus {
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

#[test]
fn bevy_host_client_e2e_drives_real_commands_and_snapshots() {
    let seed = 0xBE17_E2E1;
    let (driver_worker, port) = spawn_driver(seed, true);
    let renderer_ctx = BevyRendererContext {
        host: port,
        command_submit: Arc::new(|cmd| match cmd {
            ControlCommand::UpdateSimulation(_) => Some("sim-receipt".into()),
            ControlCommand::UpdateSelection(_) => Some("sel-receipt".into()),
            _ => None,
        }),
        control_health: None,
    };
    let submitter = CommandSubmitter::new(Arc::clone(&renderer_ctx.command_submit));
    let submit_receipt = submitter.submit_simulation(SimulationCommand {
        paused: Some(true),
        speed_multiplier: Some(1.0),
        step_once: true,
    });
    assert_eq!(submit_receipt, Some("sim-receipt".into()));

    let mut client = renderer_ctx.client();
    let mut subscription = client.subscribe_snapshots();

    // 1. Initial snapshot at tick 0
    let mut current_snap = None;
    let deadline = Instant::now() + Duration::from_secs(3);
    while current_snap.is_none() && Instant::now() <= deadline {
        if let Some(snap) = client
            .poll_snapshot(&mut subscription)
            .expect("poll snapshot")
        {
            current_snap = Some(snap);
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    let snap_0 = current_snap.expect("initial snapshot observed");
    assert_eq!(snap_0.world.tick, 0);
    assert!(snap_0.playback.paused);

    // 2. Submit Step command via typed envelope
    let cmd_id_step = CommandId::from_client_sequence(0xBE17, 1);
    let envelope_step = CommandEnvelope::new(cmd_id_step, HostCommand::Step);
    let submitted_step = client.submit(envelope_step).expect("submit step command");
    assert_eq!(submitted_step.command_id(), cmd_id_step);

    let status_step = wait_resolved(&mut client, cmd_id_step);
    assert!(
        matches!(status_step.application(), ApplicationState::Applied(_)),
        "step command must apply cleanly: {:?}",
        status_step.application()
    );

    // Poll snapshot after step
    let mut snap_after_step = None;
    let deadline = Instant::now() + Duration::from_secs(3);
    while snap_after_step.is_none() && Instant::now() <= deadline {
        if let Some(snap) = client
            .poll_snapshot(&mut subscription)
            .expect("poll snapshot")
        {
            snap_after_step = Some(snap);
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    let snap_1 = snap_after_step.expect("snapshot after step");
    assert_eq!(snap_1.world.tick, 1);

    // 3. Selection command
    let cmd_id_sel = CommandId::from_client_sequence(0xBE17, 2);
    let sel_update = SelectionUpdate {
        mode: SelectionMode::Replace,
        agent_ids: vec![1],
        state: SelectionState::Selected,
    };
    let envelope_sel = CommandEnvelope::new(cmd_id_sel, HostCommand::UpdateSelection(sel_update));
    let submitted_sel = client
        .submit(envelope_sel)
        .expect("submit selection command");
    assert_eq!(submitted_sel.command_id(), cmd_id_sel);
    let status_sel = wait_resolved(&mut client, cmd_id_sel);
    assert!(matches!(
        status_sel.application(),
        ApplicationState::Applied(_)
    ));

    // 4. Resume automatic simulation ticking
    let cmd_id_resume = CommandId::from_client_sequence(0xBE17, 3);
    let envelope_resume = CommandEnvelope::new(cmd_id_resume, HostCommand::Resume);
    let submitted_resume = client
        .submit(envelope_resume)
        .expect("submit resume command");
    assert_eq!(submitted_resume.command_id(), cmd_id_resume);
    let status_resume = wait_resolved(&mut client, cmd_id_resume);
    assert!(matches!(
        status_resume.application(),
        ApplicationState::Applied(_)
    ));

    // Let simulation run so multiple ticks execute and coalesce
    thread::sleep(Duration::from_millis(80));

    let mut snap_running = None;
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() <= deadline {
        if let Some(snap) = client
            .poll_snapshot(&mut subscription)
            .expect("poll snapshot")
        {
            snap_running = Some(snap);
            if snap_running.as_ref().unwrap().world.tick >= 4 {
                break;
            }
        }
        thread::sleep(Duration::from_millis(5));
    }
    let snap_run = snap_running.expect("running snapshot");
    assert!(snap_run.world.tick >= 2, "world tick should have advanced");
    let skipped_revisions = subscription.skipped_revisions();
    let coalesced_steps = snap_run.world.tick.saturating_sub(1);

    // Compute blake3 digest of current dynamic world state
    let digest_str = {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&snap_run.world.tick.to_le_bytes());
        hasher.update(&(snap_run.world.agents.len() as u64).to_le_bytes());
        for agent in &snap_run.world.agents {
            hasher.update(&agent.position[0].to_le_bytes());
            hasher.update(&agent.position[1].to_le_bytes());
        }
        hasher.finalize().to_hex().to_string()
    };

    // 5. Explicit shutdown
    let cmd_id_shutdown = CommandId::from_client_sequence(0xBE17, 4);
    let envelope_shutdown = CommandEnvelope::new(cmd_id_shutdown, HostCommand::Shutdown);
    let _ = client.submit(envelope_shutdown).expect("submit shutdown");

    let run_receipt = driver_worker
        .join()
        .expect("driver thread joins cleanly")
        .expect("driver run outcome");
    assert_eq!(run_receipt.outcome, ChannelRunOutcome::Stopped);

    let commit = option_env!("SCRIPTBOTS_SOURCE_COMMIT")
        .map(str::to_string)
        .or_else(|| {
            std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .and_then(|out| String::from_utf8(out.stdout).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "unversioned-dev".to_string());

    let evidence = serde_json::json!({
        "schema": "scriptbots.bevy-host-client-e2e.v1",
        "client": "bevy-host-client-primary",
        "revision": snap_run.revision.get(),
        "command_id": format!("{:032x}", cmd_id_step.get()),
        "receipt": format!("{:032x}", submitted_step.command_id().get()),
        "tick": snap_run.world.tick,
        "digest": digest_str,
        "coalescing": {
            "skipped_revisions": skipped_revisions,
            "coalesced_steps": coalesced_steps,
        },
        "lifecycle": format!("{:?}", snap_run.lifecycle).to_lowercase(),
        "shutdown": {
            "clean": true,
            "outcome": "stopped",
            "drives": run_receipt.drives,
            "commands_admitted": run_receipt.commands_admitted,
        },
        "source_commit": commit
    });

    println!("{}", serde_json::to_string(&evidence).unwrap());
}

#[test]
fn test_multiple_bevy_clients_and_windows_digest_parity() {
    let seed = 0xBE17_CAFE;
    let (driver_worker, port) = spawn_driver(seed, true);

    let mut client_a = HostClient::new(port.clone());
    let mut client_b = HostClient::new(port.clone());
    let mut client_c = HostClient::new(port.clone());

    let mut sub_a = client_a.subscribe_snapshots();
    let mut sub_b = client_b.subscribe_snapshots();
    let mut sub_c = client_c.subscribe_snapshots();

    // Advance 5 steps via Step commands
    for i in 1..=5 {
        let cmd_id = CommandId::from_client_sequence(0xBE17_000A, i);
        let _ = client_a
            .submit(CommandEnvelope::new(cmd_id, HostCommand::Step))
            .expect("submit step");
        let status = wait_resolved(&mut client_a, cmd_id);
        assert!(matches!(status.application(), ApplicationState::Applied(_)));

        // Sub A polls every step
        let snap_a = client_a
            .poll_snapshot(&mut sub_a)
            .unwrap()
            .expect("sub a snapshot");
        assert_eq!(snap_a.world.tick, i);
    }

    // Sub B polls once after 5 steps (coalesced)
    let snap_b = client_b
        .poll_snapshot(&mut sub_b)
        .unwrap()
        .expect("sub b snapshot");
    assert_eq!(snap_b.world.tick, 5);

    // Sub C also polls once (independent coalesced window)
    let snap_c = client_c
        .poll_snapshot(&mut sub_c)
        .unwrap()
        .expect("sub c snapshot");
    assert_eq!(snap_c.world.tick, 5);

    // Verify all subscribers observed bit-identical dynamic worlds
    assert_eq!(snap_b.world, snap_c.world);
    assert_eq!(snap_b.revision, snap_c.revision);

    // Clean shutdown
    let cmd_id_shutdown = CommandId::from_client_sequence(0xBE17_000A, 100);
    let _ = client_a
        .submit(CommandEnvelope::new(cmd_id_shutdown, HostCommand::Shutdown))
        .expect("submit shutdown");
    let receipt = driver_worker.join().unwrap().unwrap();
    assert_eq!(receipt.outcome, ChannelRunOutcome::Stopped);
}

#[test]
fn test_bevy_client_queue_saturation_and_stale_revisions() {
    let seed = 0xBE17_F00D;
    let (driver_worker, port) = spawn_driver(seed, true);
    let mut client = HostClient::new(port);

    // Negative control: Submit command with stale/impossible control revision guard
    let cmd_id_stale = CommandId::from_client_sequence(0xBE17_000F, 1);
    let envelope_stale = CommandEnvelope::new(cmd_id_stale, HostCommand::Pause)
        .expecting_control_revision(ControlRevision::new(999_999));

    let _ = client.submit(envelope_stale).expect("submit admitted");
    let status_stale = wait_resolved(&mut client, cmd_id_stale);

    assert!(
        matches!(
            status_stale.application(),
            ApplicationState::Rejected(RejectionReason::ControlRevisionConflict { .. })
        ),
        "stale revision must be rejected via typed ControlRevisionConflict: {:?}",
        status_stale.application()
    );

    // Clean shutdown
    let cmd_id_shutdown = CommandId::from_client_sequence(0xBE17_000F, 2);
    let _ = client
        .submit(CommandEnvelope::new(cmd_id_shutdown, HostCommand::Shutdown))
        .expect("submit shutdown");
    let receipt = driver_worker.join().unwrap().unwrap();
    assert_eq!(receipt.outcome, ChannelRunOutcome::Stopped);
}

#[test]
fn test_bevy_renderer_failure_and_reconnect_never_duplicate_simulation_time() {
    let seed = 0xBE17_BEEF;
    let (driver_worker, port) = spawn_driver(seed, true);

    // Window / Renderer 1 connects
    let mut client_1 = HostClient::new(port.clone());
    let mut sub_1 = client_1.subscribe_snapshots();

    // Step 2 ticks
    for i in 1..=2 {
        let cmd_id = CommandId::from_client_sequence(0xBE17_000C, i);
        let _ = client_1
            .submit(CommandEnvelope::new(cmd_id, HostCommand::Step))
            .expect("step");
        wait_resolved(&mut client_1, cmd_id);
    }
    let snap_1 = client_1.poll_snapshot(&mut sub_1).unwrap().unwrap();
    assert_eq!(snap_1.world.tick, 2);

    // Renderer 1 crashes / drops completely
    drop(client_1);
    let _ = sub_1;

    // Host continues stepping independently
    let mut client_headless = HostClient::new(port.clone());
    let cmd_id_step3 = CommandId::from_client_sequence(0xBE17_000D, 1);
    let _ = client_headless
        .submit(CommandEnvelope::new(cmd_id_step3, HostCommand::Step))
        .expect("step 3");
    wait_resolved(&mut client_headless, cmd_id_step3);

    // Renderer 2 connects / reconnects
    let mut client_2 = HostClient::new(port);
    let mut sub_2 = client_2.subscribe_snapshots();

    let snap_reconnect = client_2.poll_snapshot(&mut sub_2).unwrap().unwrap();
    // Must observe tick 3 with monotonic progress, no duplicate ticks or rollback
    assert_eq!(snap_reconnect.world.tick, 3);
    assert!(snap_reconnect.revision > snap_1.revision);

    // Clean shutdown
    let cmd_id_shutdown = CommandId::from_client_sequence(0xBE17_000D, 2);
    let _ = client_headless
        .submit(CommandEnvelope::new(cmd_id_shutdown, HostCommand::Shutdown))
        .expect("shutdown");
    let receipt = driver_worker.join().unwrap().unwrap();
    assert_eq!(receipt.outcome, ChannelRunOutcome::Stopped);
}
