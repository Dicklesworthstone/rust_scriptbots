//! Public-boundary coverage for the HostCore storage-journal adapter.

use fsqlite::Connection;
use scriptbots_core::{
    AgentData, AgentUid, BrainRunner, ControlCommand, INPUT_SIZE, OUTPUT_SIZE, Position,
    ReplayEventKind, ScriptBotsConfig, SelectionMode, SelectionState, SelectionUpdate, Tick,
    WorldDigestV1, WorldState, channels::OutputChannel,
};
use scriptbots_runtime::{
    ApplicationState, CommandAuthorityLookupFailure, CommandEnvelope, CommandId, CommandStatus,
    ControlRevision, EventCatchUp, EventCatchUpGuarantee, EventCatchUpState, EventCommitment,
    EventJournalReader, EventPageSource, EventPoll, EventSequence, FixedDeadlineHost,
    HostAccessError, HostBlocker, HostCommand, HostCore, HostCoreOptions, HostLifecycle, HostPort,
    HostSessionId, JournalAdmission, JournalBatchId, JournalState, LocalHostPort, ManualInstant,
    NullFrontend, NullFrontendSubmissionError, PlaybackSnapshot, RejectionReason,
    channel::{ChannelHostDriver, ChannelHostOptions, ChannelHostPort, ChannelRunOutcome},
};
use scriptbots_storage::{
    CommandJournalCursor, CommandStorageTransitionKind, DomainEventExpectation, DomainEventPayload,
    HostJournalPrefixes, HostJournalRecordState, HostJournalSessionPage, PersistenceGuarantee,
    StorageError, StorageEventJournalReader, StorageIntegrityCheckResult, StorageJournalOptions,
    StoragePipeline, StorageReader,
};
use std::{
    fs,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const WORKER_RETRY_LIMIT: usize = 10_000;

fn compact_world() -> WorldState {
    WorldState::new(ScriptBotsConfig {
        world_width: 64,
        world_height: 64,
        food_cell_size: 16,
        rng_seed: Some(0x5eed),
        closed: true,
        history_capacity: 8,
        persistence_interval: 1,
        ..ScriptBotsConfig::default()
    })
    .expect("compact deterministic journal world")
}

fn command_replay_world() -> WorldState {
    WorldState::new(ScriptBotsConfig {
        world_width: 100,
        world_height: 100,
        food_cell_size: 20,
        population_minimum: 0,
        population_spawn_interval: 0,
        reproduction_attempt_chance: 0.0,
        // The first Step makes the injected parents eligible for crossover, while
        // the second Step still exercises a real sealed persistence boundary.
        // Sealing every tick would correctly reject the intervening external GUI
        // edits rather than testing their journal/replay ordering.
        persistence_interval: 2,
        rng_seed: Some(0x37C0_1100),
        closed: false,
        history_capacity: 8,
        ..ScriptBotsConfig::default()
    })
    .expect("deterministic GUI command replay world")
}

#[derive(Debug)]
struct AlwaysSpikeBrain;

impl BrainRunner for AlwaysSpikeBrain {
    fn kind(&self) -> &'static str {
        "storage-journal-always-spike"
    }

    fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        let mut outputs = [0.0; OUTPUT_SIZE];
        outputs[OutputChannel::SpikeTarget.index()] = 1.0;
        outputs
    }
}

fn eventful_cadence_world() -> WorldState {
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 64,
        world_height: 64,
        food_cell_size: 16,
        initial_food: 0.0,
        food_respawn_interval: 0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        rng_seed: Some(0xd0_5e_52),
        closed: false,
        population_minimum: 0,
        population_spawn_interval: 1,
        population_spawn_count: 1,
        population_crossover_chance: 0.0,
        reproduction_attempt_chance: 0.0,
        spike_radius: 100.0,
        spike_damage: 1_000.0,
        spike_energy_cost: 0.0,
        spike_growth_rate: 1.0,
        spike_min_length: 0.1,
        spike_alignment_cosine: 0.5,
        spike_speed_damage_bonus: 0.0,
        spike_length_damage_bonus: 0.0,
        carnivore_threshold: 0.5,
        history_capacity: 8,
        persistence_interval: 3,
        ..ScriptBotsConfig::default()
    })
    .expect("deterministic lifecycle/combat journal world");
    let attacker_brain = world
        .brain_registry_mut()
        .expect("fresh world permits brain registration")
        .register("storage-journal-always-spike", |_| {
            Ok(Box::new(AlwaysSpikeBrain))
        });
    let attacker = world
        .try_spawn_agent_with(
            AgentData {
                position: Position::new(10.0, 10.0),
                heading: 0.0,
                health: 2.0,
                spike_length: 1.0,
                ..AgentData::default()
            },
            |runtime| runtime.herbivore_tendency = 0.0,
        )
        .expect("deterministic seeded combat attacker");
    world
        .try_spawn_agent_with(
            AgentData {
                position: Position::new(12.0, 10.0),
                heading: 0.0,
                health: 0.1,
                ..AgentData::default()
            },
            |runtime| runtime.herbivore_tendency = 1.0,
        )
        .expect("deterministic seeded combat victim");
    assert!(
        world
            .bind_agent_brain(attacker, attacker_brain)
            .expect("bind deterministic combat attacker")
    );
    world
}

fn host_options(scientific_event_capacity: usize) -> HostCoreOptions {
    HostCoreOptions {
        initial_playback: PlaybackSnapshot {
            paused: true,
            speed_multiplier: 1.0,
        },
        scientific_event_capacity,
        volatile_event_history_capacity: scientific_event_capacity.saturating_add(8),
        ..HostCoreOptions::default()
    }
}

fn unique_database_path(label: &str) -> String {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock follows the Unix epoch")
        .as_nanos();
    std::env::temp_dir()
        .join(format!(
            "scriptbots_storage_journal_{label}_{}_{nonce}.sqlite",
            std::process::id()
        ))
        .to_str()
        .expect("temporary path is valid UTF-8")
        .to_owned()
}

struct CommandAuthorityEvidence<'a> {
    envelope: &'a CommandEnvelope,
    authority_phase: &'static str,
    cache_result: &'static str,
    durable_lookup: &'static str,
    status: Option<&'a CommandStatus>,
    disposition: &'static str,
    tick: Tick,
    world_digest: &'a str,
    recovery: bool,
    host_lifecycle: HostLifecycle,
}

fn emit_command_authority_evidence(evidence: CommandAuthorityEvidence<'_>) {
    let encoded =
        postcard::to_allocvec(evidence.envelope).expect("canonical command envelope encoding");
    let envelope_digest = blake3::hash(&encoded).to_hex().to_string();
    let admission_sequence = evidence
        .status
        .and_then(CommandStatus::admission_sequence)
        .map(|sequence| sequence.get());
    let application = evidence.status.map(CommandStatus::application);
    let journal = evidence.status.map(CommandStatus::journal);
    println!(
        "{}",
        serde_json::json!({
            "schema": "scriptbots.command-authority.evidence.v1",
            "command_id": evidence.envelope.command_id.to_string(),
            "envelope_digest": envelope_digest,
            "authority_phase": evidence.authority_phase,
            "cache_result": evidence.cache_result,
            "durable_lookup": evidence.durable_lookup,
            "application_status": application,
            "journal_status": journal,
            "admission_sequence": admission_sequence,
            "host_lifecycle": evidence.host_lifecycle,
            "disposition": evidence.disposition,
            "tick": evidence.tick.0,
            "world_digest": evidence.world_digest,
            "recovery": evidence.recovery,
        })
    );
}

fn drive_until_journal_state(
    frontend: &mut NullFrontend<LocalHostPort>,
    core: &mut HostCore,
    command_id: CommandId,
    expected: &JournalState,
    next_nanos: &mut u64,
) -> CommandStatus {
    let mut last_status = None;
    let mut last_drive = None;
    for _ in 0..WORKER_RETRY_LIMIT {
        last_drive = Some(
            frontend
                .drive_at(core, ManualInstant::from_nanos(*next_nanos))
                .expect("public frontend drives the matching host"),
        );
        *next_nanos = next_nanos
            .checked_add(1)
            .expect("test manual clock does not overflow");

        let status = frontend
            .command_status(command_id)
            .expect("public command-status query succeeds")
            .expect("submitted command remains queryable");
        if status.journal() == expected {
            return status;
        }
        last_status = Some(status);
        thread::sleep(Duration::from_millis(1));
    }

    let status = last_status.expect("nonzero journal polling budget observes a command status");
    assert_eq!(
        status.journal(),
        expected,
        "command {command_id:?} did not reach journal state {expected:?}; \
         status={status:?}; last drive={last_drive:?}; interest={:?}; health={:?}",
        core.drive_interest(),
        core.health()
    );
    status
}

fn submit_envelope_durably(
    frontend: &mut NullFrontend<LocalHostPort>,
    core: &mut HostCore,
    envelope: &CommandEnvelope,
    next_nanos: &mut u64,
) -> (CommandStatus, WorldDigestV1) {
    let submitted = submit_envelope_with_authority(frontend, core, envelope.clone(), next_nanos);
    assert_eq!(submitted.command_id(), envelope.command_id);
    let status = drive_until_journal_state(
        frontend,
        core,
        envelope.command_id,
        &JournalState::Durable,
        next_nanos,
    );
    assert!(
        matches!(status.application(), ApplicationState::Applied(_)),
        "command {:?} must apply before its durable journal receipt",
        envelope.command_id
    );
    let digest = core
        .world()
        .world_digest_v1()
        .expect("command leaves a canonical world digest");
    (status, digest)
}

fn retryable_authority_envelope(error: NullFrontendSubmissionError) -> CommandEnvelope {
    match error {
        NullFrontendSubmissionError::HostAccess {
            envelope,
            source:
                HostAccessError::CommandAuthorityLookup {
                    failure:
                        CommandAuthorityLookupFailure::Pending
                        | CommandAuthorityLookupFailure::Busy
                        | CommandAuthorityLookupFailure::Capacity { .. },
                    ..
                },
        } => envelope,
        other => panic!("fresh durable command returned a terminal submission error: {other:?}"),
    }
}

fn resolve_authority_submission(
    frontend: &mut NullFrontend<LocalHostPort>,
    core: &mut HostCore,
    initial: Result<CommandStatus, NullFrontendSubmissionError>,
    next_nanos: &mut u64,
) -> CommandStatus {
    let mut envelope = match initial {
        Ok(status) => return status,
        Err(error) => retryable_authority_envelope(error),
    };
    for _ in 0..WORKER_RETRY_LIMIT {
        frontend
            .drive_at(core, ManualInstant::from_nanos(*next_nanos))
            .expect("authority retry drives the matching production host");
        *next_nanos = next_nanos
            .checked_add(1)
            .expect("authority retry clock does not overflow");
        match frontend.submit_envelope(envelope.clone()) {
            Ok(status) => return status,
            Err(error) => envelope = retryable_authority_envelope(error),
        }
        thread::sleep(Duration::from_millis(1));
    }
    panic!(
        "durable command authority did not resolve within {WORKER_RETRY_LIMIT} nonblocking polls; \
         envelope={envelope:?}; tick={:?}; interest={:?}; health={:?}",
        core.world_tick(),
        core.drive_interest(),
        core.health()
    );
}

fn resolve_authority_status(
    frontend: &mut NullFrontend<LocalHostPort>,
    core: &mut HostCore,
    command_id: CommandId,
    initial: Result<Option<CommandStatus>, HostAccessError>,
    next_nanos: &mut u64,
) -> CommandStatus {
    let mut attempt = initial;
    for _ in 0..WORKER_RETRY_LIMIT {
        match attempt {
            Ok(Some(status)) => return status,
            Ok(None) => panic!("durable command authority reported {command_id:?} absent"),
            Err(HostAccessError::CommandAuthorityLookup {
                command_id: pending_id,
                failure:
                    CommandAuthorityLookupFailure::Pending
                    | CommandAuthorityLookupFailure::Busy
                    | CommandAuthorityLookupFailure::Capacity { .. },
            }) if pending_id == command_id => {
                frontend
                    .drive_at(core, ManualInstant::from_nanos(*next_nanos))
                    .expect("status authority retry drives the matching production host");
                *next_nanos = next_nanos
                    .checked_add(1)
                    .expect("status authority retry clock does not overflow");
                thread::sleep(Duration::from_millis(1));
                attempt = frontend.command_status(command_id);
            }
            Err(error) => panic!("durable status authority failed for {command_id:?}: {error}"),
        }
    }
    panic!(
        "durable status authority did not resolve within {WORKER_RETRY_LIMIT} nonblocking polls"
    );
}

fn resolve_authority_collision(
    frontend: &mut NullFrontend<LocalHostPort>,
    core: &mut HostCore,
    command_id: CommandId,
    initial: Result<CommandStatus, NullFrontendSubmissionError>,
    next_nanos: &mut u64,
) {
    let mut attempt = initial;
    for _ in 0..WORKER_RETRY_LIMIT {
        let envelope = match attempt {
            Err(NullFrontendSubmissionError::HostAccess {
                source:
                    HostAccessError::CommandIdCollision {
                        command_id: collision_id,
                    },
                ..
            }) if collision_id == command_id => return,
            Err(error) => retryable_authority_envelope(error),
            Ok(status) => {
                panic!("changed envelope unexpectedly returned authoritative status {status:?}")
            }
        };
        frontend
            .drive_at(core, ManualInstant::from_nanos(*next_nanos))
            .expect("collision authority retry drives the matching production host");
        *next_nanos = next_nanos
            .checked_add(1)
            .expect("collision authority retry clock does not overflow");
        thread::sleep(Duration::from_millis(1));
        attempt = frontend.submit_envelope(envelope);
    }
    panic!(
        "durable collision authority did not resolve within {WORKER_RETRY_LIMIT} nonblocking polls"
    );
}

fn submit_envelope_with_authority(
    frontend: &mut NullFrontend<LocalHostPort>,
    core: &mut HostCore,
    envelope: CommandEnvelope,
    next_nanos: &mut u64,
) -> CommandStatus {
    let initial = frontend.submit_envelope(envelope);
    resolve_authority_submission(frontend, core, initial, next_nanos)
}

fn submit_command_with_authority(
    frontend: &mut NullFrontend<LocalHostPort>,
    core: &mut HostCore,
    command: HostCommand,
    expected_control_revision: Option<ControlRevision>,
    next_nanos: &mut u64,
) -> CommandStatus {
    let initial = frontend.submit(command, expected_control_revision);
    resolve_authority_submission(frontend, core, initial, next_nanos)
}

fn submit_command_with_authority_before_owner_drive(
    frontend: &mut NullFrontend<LocalHostPort>,
    command: HostCommand,
    expected_control_revision: Option<ControlRevision>,
) -> CommandStatus {
    let initial = frontend.submit(command, expected_control_revision);
    let mut envelope = match initial {
        Ok(status) => return status,
        Err(error) => retryable_authority_envelope(error),
    };
    for _ in 0..WORKER_RETRY_LIMIT {
        thread::sleep(Duration::from_millis(1));
        match frontend.submit_envelope(envelope.clone()) {
            Ok(status) => return status,
            Err(error) => envelope = retryable_authority_envelope(error),
        }
    }
    panic!(
        "durable command authority did not resolve before the owner drive within \
         {WORKER_RETRY_LIMIT} nonblocking polls"
    );
}

fn wait_channel_command_durable(
    port: &mut ChannelHostPort,
    command_id: CommandId,
) -> CommandStatus {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let observed = port
            .command_status(command_id)
            .expect("channel command-status lookup");
        if let Some(status) = &observed
            && !matches!(status.application(), ApplicationState::Admitted)
            && status.journal() == &JournalState::Durable
        {
            return status.clone();
        }
        assert!(
            Instant::now() < deadline,
            "channel command {command_id:?} did not become durable; last status={observed:?}"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

fn sorted_agent_identities(core: &HostCore) -> Vec<(AgentUid, u64)> {
    let mut identities = core
        .world()
        .agents()
        .iter_handles()
        .map(|agent_id| {
            (
                core.world()
                    .agent_uid(agent_id)
                    .expect("live agent has a stable UID"),
                agent_id.raw(),
            )
        })
        .collect::<Vec<_>>();
    identities.sort_unstable_by_key(|(uid, _)| *uid);
    identities
}

fn map_gui_command(command: ControlCommand) -> HostCommand {
    HostCommand::try_from(command).expect("GUI ControlCommand maps to one atomic HostCommand")
}

#[test]
fn memory_journal_accepts_zero_session_and_repairs_a_hot_ring_gap() {
    let mut pipeline = StoragePipeline::unattributed_memory().expect("memory storage pipeline");
    let session_id = HostSessionId::new(0);
    let journal = pipeline
        .journal_port(session_id, StorageJournalOptions::default())
        .expect("memory journal port");
    let mut core = HostCore::with_journal(
        session_id,
        compact_world(),
        host_options(1),
        Box::new(journal),
    )
    .expect("host backed by the memory storage journal");
    let mut frontend = NullFrontend::new(core.local_port(), 0x2001);
    let mut next_nanos = 0;

    let first = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::Step,
        None,
        &mut next_nanos,
    );
    let first = drive_until_journal_state(
        &mut frontend,
        &mut core,
        first.command_id(),
        &JournalState::CommittedVolatile,
        &mut next_nanos,
    );
    assert_eq!(first.journal(), &JournalState::CommittedVolatile);

    let second = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::Step,
        None,
        &mut next_nanos,
    );
    let second = drive_until_journal_state(
        &mut frontend,
        &mut core,
        second.command_id(),
        &JournalState::CommittedVolatile,
        &mut next_nanos,
    );
    assert_eq!(second.journal(), &JournalState::CommittedVolatile);
    assert_eq!(core.world_tick(), Tick(2));

    let poll = frontend
        .read_events(usize::MAX)
        .expect("read the one-entry hot ring");
    assert!(
        matches!(&poll, EventPoll::Gap(_)),
        "two events behind a one-entry hot ring must produce a gap, got {poll:?}"
    );
    let EventPoll::Gap(gap) = poll else {
        return;
    };
    let catch_up = gap.catch_up;
    assert!(
        matches!(&catch_up, EventCatchUpState::Available(_)),
        "memory journal must advertise catch-up, got {catch_up:?}"
    );
    let EventCatchUpState::Available(locator) = catch_up else {
        return;
    };
    assert_eq!(locator.guarantee(), EventCatchUpGuarantee::LiveMemory);

    let catch_up = frontend
        .catch_up_events(locator, 1)
        .expect("resolve the exact missing prefix");
    assert!(
        matches!(&catch_up, EventCatchUp::Contiguous(_)),
        "retained memory event must repair the hot-ring gap, got {catch_up:?}"
    );
    let EventCatchUp::Contiguous(caught_up) = catch_up else {
        return;
    };
    assert_eq!(caught_up.source, EventPageSource::LiveMemory);
    assert_eq!(caught_up.events.len(), 1);
    assert_eq!(caught_up.events[0].event.sequence, EventSequence::new(1));
    assert_eq!(
        caught_up.events[0].commitment,
        EventCommitment::CommittedVolatile
    );

    let hot_poll = frontend
        .read_events(1)
        .expect("resume through the current hot suffix");
    assert!(
        matches!(&hot_poll, EventPoll::Contiguous(_)),
        "successful catch-up must rejoin the hot ring, got {hot_poll:?}"
    );
    let EventPoll::Contiguous(hot_suffix) = hot_poll else {
        return;
    };
    assert_eq!(hot_suffix.events.len(), 1);
    assert_eq!(hot_suffix.events[0].event.sequence, EventSequence::new(2));

    let shutdown = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::Shutdown,
        None,
        &mut next_nanos,
    );
    drive_until_journal_state(
        &mut frontend,
        &mut core,
        shutdown.command_id(),
        &JournalState::CommittedVolatile,
        &mut next_nanos,
    );
    assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);
    assert_eq!(
        pipeline.shutdown().expect("close memory storage").guarantee,
        PersistenceGuarantee::CommittedVolatile
    );
}

#[test]
#[allow(
    clippy::drop_non_drop,
    clippy::too_many_lines,
    reason = "the public durable lifecycle deliberately releases all nested file handles before independently reopening the reader"
)]
fn file_journal_orders_durable_shutdown_and_reopens_a_detached_reader() {
    let path = unique_database_path("durable_reopen");
    let mut pipeline = StoragePipeline::create_unattributed_file(&path)
        .expect("new uniquely named file storage pipeline");
    let run_id = pipeline.run_id();
    let session_id = HostSessionId::new(0x1002);
    let options = StorageJournalOptions::default();
    let journal = pipeline
        .journal_port(session_id, options)
        .expect("file journal port");
    let mut core = HostCore::with_journal(
        session_id,
        compact_world(),
        host_options(1),
        Box::new(journal),
    )
    .expect("host backed by the durable storage journal");
    let mut frontend = NullFrontend::new(core.local_port(), 0x2002);
    let mut next_nanos = 0;

    let first_submission = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::Step,
        None,
        &mut next_nanos,
    );
    let first_command_id = first_submission.command_id();
    let first_status = drive_until_journal_state(
        &mut frontend,
        &mut core,
        first_command_id,
        &JournalState::Durable,
        &mut next_nanos,
    );
    assert!(
        matches!(first_status.application(), ApplicationState::Applied(_)),
        "first durable step must retain its exact applied boundary"
    );
    let ApplicationState::Applied(first_applied) = first_status.application() else {
        return;
    };
    let first_applied = *first_applied;

    let second_submission = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::Step,
        None,
        &mut next_nanos,
    );
    let second_command_id = second_submission.command_id();
    let second_status = drive_until_journal_state(
        &mut frontend,
        &mut core,
        second_command_id,
        &JournalState::Durable,
        &mut next_nanos,
    );
    assert!(
        matches!(second_status.application(), ApplicationState::Applied(_)),
        "second durable step must retain its exact applied boundary"
    );
    let ApplicationState::Applied(second_applied) = second_status.application() else {
        return;
    };
    let second_applied = *second_applied;

    let poll = frontend
        .read_events(usize::MAX)
        .expect("durable hot-ring gap");
    assert!(
        matches!(&poll, EventPoll::Gap(_)),
        "two durable events behind a one-entry hot ring must gap, got {poll:?}"
    );
    let EventPoll::Gap(gap) = poll else {
        return;
    };
    let catch_up = gap.catch_up;
    assert!(
        matches!(&catch_up, EventCatchUpState::Available(_)),
        "durable journal must advertise catch-up, got {catch_up:?}"
    );
    let EventCatchUpState::Available(locator) = catch_up else {
        return;
    };
    assert_eq!(locator.guarantee(), EventCatchUpGuarantee::CrashDurable);
    assert_eq!(
        locator.range(),
        scriptbots_runtime::EventSequenceRange {
            first: EventSequence::new(1),
            last: EventSequence::new(1),
        }
    );

    let live_catch_up = frontend
        .catch_up_events(locator, 1)
        .expect("read the first exact durable event before closing the writer");
    assert!(
        matches!(&live_catch_up, EventCatchUp::Contiguous(_)),
        "live durable catch-up must be contiguous, got {live_catch_up:?}"
    );
    let EventCatchUp::Contiguous(live_catch_up) = live_catch_up else {
        return;
    };
    assert_eq!(live_catch_up.source, EventPageSource::Durable);
    assert_eq!(live_catch_up.events.len(), 1);
    assert_eq!(live_catch_up.events[0].commitment, EventCommitment::Durable);
    let expected_first_event = live_catch_up.events[0].clone();

    let live_hot_suffix = frontend
        .read_events(1)
        .expect("read the second exact event from the durable hot suffix");
    assert!(
        matches!(&live_hot_suffix, EventPoll::Contiguous(_)),
        "live durable hot suffix must be contiguous, got {live_hot_suffix:?}"
    );
    let EventPoll::Contiguous(live_hot_suffix) = live_hot_suffix else {
        return;
    };
    assert_eq!(live_hot_suffix.source, EventPageSource::Hot);
    assert_eq!(live_hot_suffix.events.len(), 1);
    assert_eq!(
        live_hot_suffix.events[0].commitment,
        EventCommitment::Durable
    );
    let expected_second_event = live_hot_suffix.events[0].clone();

    let shutdown_submission = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::Shutdown,
        None,
        &mut next_nanos,
    );
    let shutdown_command_id = shutdown_submission.command_id();
    let shutdown_status = drive_until_journal_state(
        &mut frontend,
        &mut core,
        shutdown_command_id,
        &JournalState::Durable,
        &mut next_nanos,
    );
    assert_eq!(shutdown_status.journal(), &JournalState::Durable);
    assert!(
        matches!(shutdown_status.application(), ApplicationState::Applied(_)),
        "durable shutdown must retain its exact applied boundary"
    );
    let ApplicationState::Applied(shutdown_applied) = shutdown_status.application() else {
        return;
    };
    let shutdown_applied = *shutdown_applied;
    assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);
    assert_eq!(
        pipeline
            .shutdown()
            .expect("close durable storage")
            .guarantee,
        PersistenceGuarantee::Durable
    );

    drop(frontend);
    drop(core);
    drop(pipeline);

    let reader = StorageEventJournalReader::open_file(&path, run_id, session_id, options)
        .expect("reopen a detached durable reader after writer shutdown");
    assert_eq!(reader.guarantee(), EventCatchUpGuarantee::CrashDurable);
    let complete_event_range = scriptbots_runtime::EventSequenceRange {
        first: EventSequence::new(1),
        last: EventSequence::new(2),
    };
    assert_eq!(reader.available_range(), Some(complete_event_range));
    let retention = reader
        .retention_snapshot()
        .expect("reopened reader retains its complete durable event range");
    assert_eq!(retention.session_id(), session_id);
    assert_eq!(retention.guarantee(), EventCatchUpGuarantee::CrashDurable);
    assert_eq!(retention.range(), complete_event_range);
    assert!(
        reader.contains_event_identity(EventSequence::new(1), JournalBatchId::new(session_id, 1))
    );
    assert!(
        reader.contains_event_identity(EventSequence::new(2), JournalBatchId::new(session_id, 2))
    );

    let catch_up = reader
        .read(locator, 1)
        .expect("read the original durable catch-up locator after reopen");
    assert!(
        matches!(&catch_up, EventCatchUp::Contiguous(_)),
        "reopened reader must retain the committed missing prefix, got {catch_up:?}"
    );
    let EventCatchUp::Contiguous(page) = catch_up else {
        return;
    };
    assert_eq!(page.source, EventPageSource::Durable);
    assert_eq!(page.events.len(), 1);
    assert_eq!(page.events[0].event.sequence, EventSequence::new(1));
    assert_eq!(page.events[0].commitment, EventCommitment::Durable);
    assert_eq!(page.events[0], expected_first_event);

    let finished_reader = StorageReader::open_finished_for_run(&path, run_id)
        .expect("open the immutable finished run for typed journal conformance");
    let journal_page = finished_reader
        .host_journal_session_conformance_page(session_id, None, 16, options.max_event_page_bytes)
        .expect("read one bounded canonical page covering the complete journal session");
    assert_eq!(journal_page.run_id, run_id);
    assert_eq!(journal_page.session_id, session_id);
    assert_eq!(
        journal_page.integrity_check,
        StorageIntegrityCheckResult::Ok
    );
    assert_eq!(
        journal_page.progress.journal,
        HostJournalPrefixes {
            admitted: 3,
            applied: 3,
            committed_volatile: 3,
            durable: 3,
        }
    );
    assert_eq!(
        journal_page.progress.events,
        HostJournalPrefixes {
            admitted: 2,
            applied: 2,
            committed_volatile: 2,
            durable: 2,
        }
    );
    assert_eq!(
        journal_page.progress.shutdown,
        Some(JournalBatchId::new(session_id, 3))
    );
    assert_eq!(journal_page.next_after, None);
    assert_eq!(journal_page.records.len(), 3);
    let [first_record, second_record, shutdown_record] = journal_page.records.as_slice() else {
        return;
    };

    assert_eq!(first_record.batch_id, JournalBatchId::new(session_id, 1));
    assert_eq!(second_record.batch_id, JournalBatchId::new(session_id, 2));
    assert_eq!(shutdown_record.batch_id, JournalBatchId::new(session_id, 3));
    for record in [first_record, second_record, shutdown_record] {
        assert_eq!(record.state, HostJournalRecordState::Durable);
    }

    let Some(first_lifecycle) = first_record.command_lifecycle.as_ref() else {
        return;
    };
    let first_command = first_lifecycle.envelope();
    assert_eq!(first_command.command_id, first_command_id);
    assert_eq!(first_command.expected_control_revision, None);
    assert!(matches!(&first_command.command, HostCommand::Step));
    assert_eq!(first_record.applied, first_applied);
    assert_eq!(first_record.event.as_ref(), Some(&expected_first_event));

    let Some(second_lifecycle) = second_record.command_lifecycle.as_ref() else {
        return;
    };
    let second_command = second_lifecycle.envelope();
    assert_eq!(second_command.command_id, second_command_id);
    assert_eq!(second_command.expected_control_revision, None);
    assert!(matches!(&second_command.command, HostCommand::Step));
    assert_eq!(second_record.applied, second_applied);
    assert_eq!(second_record.event.as_ref(), Some(&expected_second_event));

    let Some(shutdown_lifecycle) = shutdown_record.command_lifecycle.as_ref() else {
        return;
    };
    let shutdown_command = shutdown_lifecycle.envelope();
    assert_eq!(shutdown_command.command_id, shutdown_command_id);
    assert_eq!(shutdown_command.expected_control_revision, None);
    assert!(matches!(&shutdown_command.command, HostCommand::Shutdown));
    assert!(shutdown_lifecycle.is_applied_shutdown());
    assert_eq!(shutdown_record.applied, shutdown_applied);
    assert_eq!(shutdown_record.event, None);

    let command_evidence = finished_reader
        .command_journal_evidence(session_id)
        .expect("finished Step/Step/Shutdown session has non-vacuous command evidence");
    assert_eq!(command_evidence.command_count, 3);
    assert_eq!(command_evidence.application_transition_count, 6);
    assert_eq!(command_evidence.storage_transition_count, 6);
    let exact_first = finished_reader
        .command_journal_record(session_id, first_command_id)
        .expect("exact typed command lookup");
    assert_eq!(exact_first.batch_id, JournalBatchId::new(session_id, 1));
    assert_eq!(exact_first.lifecycle, first_lifecycle.clone());
    assert_eq!(exact_first.terminal_boundary, first_applied);
    assert_eq!(
        exact_first.scientific_event_sequence,
        Some(EventSequence::new(1))
    );
    assert_eq!(exact_first.archive_payload_digest.len(), 64);
    assert_eq!(exact_first.storage_transitions.len(), 2);
    assert_eq!(
        exact_first.storage_transitions[0].kind,
        CommandStorageTransitionKind::CommittedVolatile
    );
    assert_eq!(
        exact_first.storage_transitions[1].kind,
        CommandStorageTransitionKind::Durable
    );
    let command_first_page = finished_reader
        .command_journal_page(session_id, None, 1, options.max_event_page_bytes)
        .expect("first exact command page");
    assert_eq!(command_first_page.commands.len(), 1);
    assert_eq!(command_first_page.evidence, command_evidence);
    let first_command_cursor = command_first_page
        .next_after
        .expect("Step/Step/Shutdown has a second command");
    assert_eq!(
        first_command_cursor,
        command_first_page.commands[0].cursor()
    );
    let command_second_page = finished_reader
        .command_journal_page(
            session_id,
            Some(first_command_cursor),
            2,
            options.max_event_page_bytes,
        )
        .expect("resume after exact command cursor");
    assert_eq!(command_second_page.commands.len(), 2);
    assert_eq!(command_second_page.next_after, None);
    let fabricated_command_cursor = CommandJournalCursor {
        command_id: shutdown_command_id,
        ..first_command_cursor
    };
    let fabricated_error = finished_reader
        .command_journal_page(
            session_id,
            Some(fabricated_command_cursor),
            1,
            options.max_event_page_bytes,
        )
        .expect_err("cursor command id must match its exact journal sequence");
    assert!(matches!(
        fabricated_error,
        StorageError::InvalidData {
            context: "host_command_records.after",
            ..
        }
    ));

    let first_page = finished_reader
        .host_journal_session_conformance_page(session_id, None, 1, options.max_event_page_bytes)
        .expect("read the first single-record conformance page");
    assert_eq!(first_page.records.len(), 1);
    assert_eq!(
        first_page.records[0].batch_id,
        JournalBatchId::new(session_id, 1)
    );
    assert_eq!(
        first_page.next_after,
        Some(JournalBatchId::new(session_id, 1))
    );

    let second_page = finished_reader
        .host_journal_session_conformance_page(
            session_id,
            first_page.next_after,
            1,
            options.max_event_page_bytes,
        )
        .expect("read the second single-record conformance page");
    assert_eq!(second_page.records.len(), 1);
    assert_eq!(
        second_page.records[0].batch_id,
        JournalBatchId::new(session_id, 2)
    );
    assert_eq!(
        second_page.next_after,
        Some(JournalBatchId::new(session_id, 2))
    );

    let third_page = finished_reader
        .host_journal_session_conformance_page(
            session_id,
            second_page.next_after,
            1,
            options.max_event_page_bytes,
        )
        .expect("read the final single-record conformance page");
    assert_eq!(third_page.records.len(), 1);
    assert_eq!(
        third_page.records[0].batch_id,
        JournalBatchId::new(session_id, 3)
    );
    assert_eq!(third_page.next_after, None);

    let empty_tip_page = finished_reader
        .host_journal_session_conformance_page(
            session_id,
            Some(JournalBatchId::new(session_id, 3)),
            1,
            options.max_event_page_bytes,
        )
        .expect("a cursor at the durable tip returns an empty terminal page");
    assert!(empty_tip_page.records.is_empty());
    assert_eq!(empty_tip_page.next_after, None);

    let assert_invalid_data_context =
        |result: Result<HostJournalSessionPage, StorageError>, expected_context: &'static str| {
            let error = result.expect_err("invalid conformance query must be rejected");
            assert!(
                matches!(
                    &error,
                    StorageError::InvalidData { context, .. }
                        if *context == expected_context
                ),
                "expected InvalidData context {expected_context:?}, got {error:?}"
            );
        };

    assert_invalid_data_context(
        finished_reader.host_journal_session_conformance_page(
            session_id,
            None,
            0,
            options.max_event_page_bytes,
        ),
        "host_journal_conformance.record_limit",
    );
    assert_invalid_data_context(
        finished_reader.host_journal_session_conformance_page(
            session_id,
            None,
            4_097,
            options.max_event_page_bytes,
        ),
        "host_journal_conformance.record_limit",
    );
    assert_invalid_data_context(
        finished_reader.host_journal_session_conformance_page(session_id, None, 1, 0),
        "host_journal_conformance.page_payload_byte_limit",
    );
    assert_invalid_data_context(
        finished_reader.host_journal_session_conformance_page(
            session_id,
            None,
            1,
            (256_usize << 20) + 1,
        ),
        "host_journal_conformance.page_payload_byte_limit",
    );
    assert_invalid_data_context(
        finished_reader.host_journal_session_conformance_page(session_id, None, 1, 1),
        "host_journal_conformance.page_payload_byte_limit",
    );
    assert_invalid_data_context(
        finished_reader.host_journal_session_conformance_page(
            session_id,
            Some(JournalBatchId::new(HostSessionId::new(0xdead), 1)),
            1,
            options.max_event_page_bytes,
        ),
        "host_journal_conformance.after",
    );
    assert_invalid_data_context(
        finished_reader.host_journal_session_conformance_page(
            session_id,
            Some(JournalBatchId::new(session_id, 0)),
            1,
            options.max_event_page_bytes,
        ),
        "host_journal_conformance.after",
    );
    assert_invalid_data_context(
        finished_reader.host_journal_session_conformance_page(
            session_id,
            Some(JournalBatchId::new(session_id, 4)),
            1,
            options.max_event_page_bytes,
        ),
        "host_journal_conformance.after",
    );

    let empty_domain_evidence = finished_reader
        .domain_event_evidence(session_id, DomainEventExpectation::AllowEmpty)
        .expect("two empty scientific boundaries still have explicit projection coverage");
    assert_eq!(empty_domain_evidence.scientific_event_count, 2);
    assert_eq!(empty_domain_evidence.domain_event_count, 0);
    let empty_domain_page = finished_reader
        .domain_event_page(session_id, None, 16, options.max_event_page_bytes)
        .expect("honestly empty normalized domain page");
    assert!(empty_domain_page.events.is_empty());
    assert_eq!(empty_domain_page.next_after, None);
    assert_eq!(empty_domain_page.evidence, empty_domain_evidence);
    let no_evidence = finished_reader
        .domain_event_evidence(session_id, DomainEventExpectation::RequireNonEmpty)
        .expect_err("a scenario-declared event requirement must fail closed on zero rows");
    assert!(
        matches!(
            no_evidence,
            StorageError::NoEvidence {
                context: "host_domain_events",
                ..
            }
        ),
        "non-vacuity must remain a typed NoEvidence outcome"
    );

    finished_reader
        .close()
        .expect("close the immutable finished-run conformance reader");

    // The uniquely named database is intentionally retained; this test performs no file deletion.
}

#[test]
#[allow(
    clippy::drop_non_drop,
    clippy::too_many_lines,
    reason = "one mock-free proof keeps cache eviction, durable authority, restart, and exactly-once progress observable together"
)]
fn file_command_authority_survives_cache_eviction_and_restart() {
    const CLIENT_NAMESPACE: u64 = 0x2a_75_74_68;
    const FIRST_SESSION: HostSessionId = HostSessionId::new(0x2a_01);
    const STATUS_SESSION: HostSessionId = HostSessionId::new(0x2a_02);
    const EXACT_SESSION: HostSessionId = HostSessionId::new(0x2a_03);
    const RECOVERED_SESSION: HostSessionId = HostSessionId::new(0x2a_04);

    let path = unique_database_path("command_authority_eviction_restart");
    let journal_options = StorageJournalOptions::default();
    let host_options = HostCoreOptions {
        archived_command_capacity: 2,
        ..host_options(8)
    };
    let mut pipeline = StoragePipeline::create_unattributed_file(&path)
        .expect("new file-backed command-authority journal");
    let run_id = pipeline.run_id();
    let journal = pipeline
        .journal_port(FIRST_SESSION, journal_options)
        .expect("first command-authority journal port");
    let mut core = HostCore::with_journal(
        FIRST_SESSION,
        compact_world(),
        host_options,
        Box::new(journal),
    )
    .expect("first command-authority host");
    let mut frontend = NullFrontend::new(core.local_port(), CLIENT_NAMESPACE);
    let mut next_nanos = 0;

    let settled_envelopes = [1, 2, 3].map(|sequence| {
        CommandEnvelope::new(
            CommandId::from_client_sequence(CLIENT_NAMESPACE, sequence),
            HostCommand::Step,
        )
    });
    let mut settled_statuses = Vec::with_capacity(settled_envelopes.len());
    for envelope in &settled_envelopes {
        let (status, _) =
            submit_envelope_durably(&mut frontend, &mut core, envelope, &mut next_nanos);
        settled_statuses.push(status);
    }
    assert_eq!(core.world_tick(), Tick(3));

    let first_envelope = settled_envelopes[0].clone();
    let first_status = settled_statuses[0].clone();
    let first_id = first_envelope.command_id;
    let before_evicted_retry_tick = core.world_tick();
    let before_evicted_retry_digest = core
        .world()
        .world_digest_v1()
        .expect("world digest before evicted authority probes");

    let exact_initial = frontend.submit_envelope(first_envelope.clone());
    assert!(
        matches!(
            &exact_initial,
            Err(NullFrontendSubmissionError::HostAccess {
                source:
                    HostAccessError::CommandAuthorityLookup {
                        command_id,
                        failure: CommandAuthorityLookupFailure::Pending,
                    },
                ..
            }) if *command_id == first_id
        ),
        "the exact evicted retry must consult durable authority"
    );
    let exact_replay =
        resolve_authority_submission(&mut frontend, &mut core, exact_initial, &mut next_nanos);
    assert_eq!(exact_replay, first_status);
    assert_eq!(core.world_tick(), before_evicted_retry_tick);
    assert_eq!(
        core.world()
            .world_digest_v1()
            .expect("world digest after exact evicted replay"),
        before_evicted_retry_digest
    );
    emit_command_authority_evidence(CommandAuthorityEvidence {
        envelope: &first_envelope,
        authority_phase: "evicted_durable_lookup",
        cache_result: "miss_evicted",
        durable_lookup: "resolved_exact",
        status: Some(&exact_replay),
        disposition: "authoritative_replay",
        tick: core.world_tick(),
        world_digest: &before_evicted_retry_digest.overall,
        recovery: false,
        host_lifecycle: core.latest_snapshot().lifecycle,
    });

    let changed_envelope = CommandEnvelope::new(first_id, HostCommand::Pause);
    let changed = frontend.submit_envelope(changed_envelope.clone());
    assert!(
        matches!(
            changed,
            Err(NullFrontendSubmissionError::HostAccess {
                source: HostAccessError::CommandIdCollision { command_id },
                ..
            }) if command_id == first_id
        ),
        "the authority cache warmed by the exact durable replay must reject a changed payload"
    );
    assert_eq!(
        frontend
            .command_status(first_id)
            .expect("cached authoritative status lookup"),
        Some(first_status.clone())
    );
    assert_eq!(core.world_tick(), before_evicted_retry_tick);
    assert_eq!(
        core.world()
            .world_digest_v1()
            .expect("world digest after evicted authority replay and collision"),
        before_evicted_retry_digest
    );
    emit_command_authority_evidence(CommandAuthorityEvidence {
        envelope: &changed_envelope,
        authority_phase: "evicted_cache",
        cache_result: "hit_collision",
        durable_lookup: "not_used",
        status: Some(&first_status),
        disposition: "command_id_collision",
        tick: core.world_tick(),
        world_digest: &before_evicted_retry_digest.overall,
        recovery: false,
        host_lifecycle: core.latest_snapshot().lifecycle,
    });

    let shutdown_envelope = CommandEnvelope::new(
        CommandId::from_client_sequence(CLIENT_NAMESPACE, 4),
        HostCommand::Shutdown,
    );
    submit_envelope_durably(
        &mut frontend,
        &mut core,
        &shutdown_envelope,
        &mut next_nanos,
    );
    assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);
    assert_eq!(
        pipeline
            .shutdown()
            .expect("first command-authority writer shuts down")
            .guarantee,
        PersistenceGuarantee::Durable
    );
    drop(frontend);
    drop(core);
    drop(pipeline);

    let mut status_pipeline =
        StoragePipeline::recover_existing(&path).expect("recover the same authority database");
    assert_eq!(status_pipeline.run_id(), run_id);
    let status_journal = status_pipeline
        .journal_port(STATUS_SESSION, journal_options)
        .expect("cold recovered status journal port");
    let mut status_core = HostCore::with_journal(
        STATUS_SESSION,
        compact_world(),
        host_options,
        Box::new(status_journal),
    )
    .expect("fresh status host over recovered durable authority");
    let mut status_frontend = NullFrontend::new(status_core.local_port(), CLIENT_NAMESPACE);
    let mut status_next_nanos = 0;
    let status_before_tick = status_core.world_tick();
    let status_before_digest = status_core
        .world()
        .world_digest_v1()
        .expect("fresh world digest before cold recovered status lookup");

    let recovered_status_initial = status_frontend.command_status(first_id);
    assert!(
        matches!(
            &recovered_status_initial,
            Err(HostAccessError::CommandAuthorityLookup {
                command_id,
                failure: CommandAuthorityLookupFailure::Pending,
            }) if *command_id == first_id
        ),
        "a fresh host status lookup must consult recovered durable authority"
    );
    let recovered_status = resolve_authority_status(
        &mut status_frontend,
        &mut status_core,
        first_id,
        recovered_status_initial,
        &mut status_next_nanos,
    );
    assert_eq!(recovered_status, first_status);
    assert_eq!(status_core.world_tick(), status_before_tick);
    assert_eq!(
        status_core
            .world()
            .world_digest_v1()
            .expect("fresh world digest after cold recovered status lookup"),
        status_before_digest
    );
    emit_command_authority_evidence(CommandAuthorityEvidence {
        envelope: &first_envelope,
        authority_phase: "recovered_durable_status",
        cache_result: "miss_cold",
        durable_lookup: "resolved_status",
        status: Some(&recovered_status),
        disposition: "authoritative_status",
        tick: status_core.world_tick(),
        world_digest: &status_before_digest.overall,
        recovery: true,
        host_lifecycle: status_core.latest_snapshot().lifecycle,
    });
    drop(status_frontend);
    drop(status_core);
    assert_eq!(
        status_pipeline
            .shutdown()
            .expect("close cold status authority writer")
            .guarantee,
        PersistenceGuarantee::Durable
    );
    drop(status_pipeline);

    let mut exact_pipeline =
        StoragePipeline::recover_existing(&path).expect("reopen authority for cold exact retry");
    assert_eq!(exact_pipeline.run_id(), run_id);
    let exact_journal = exact_pipeline
        .journal_port(EXACT_SESSION, journal_options)
        .expect("cold exact-retry journal port");
    let mut exact_core = HostCore::with_journal(
        EXACT_SESSION,
        compact_world(),
        host_options,
        Box::new(exact_journal),
    )
    .expect("fresh exact-retry host over recovered durable authority");
    let mut exact_frontend = NullFrontend::new(exact_core.local_port(), CLIENT_NAMESPACE);
    let mut exact_next_nanos = 0;
    let exact_before_tick = exact_core.world_tick();
    let exact_before_digest = exact_core
        .world()
        .world_digest_v1()
        .expect("fresh world digest before cold exact retry");

    let recovered_exact_initial = exact_frontend.submit_envelope(first_envelope.clone());
    assert!(
        matches!(
            &recovered_exact_initial,
            Err(NullFrontendSubmissionError::HostAccess {
                source:
                    HostAccessError::CommandAuthorityLookup {
                        command_id,
                        failure: CommandAuthorityLookupFailure::Pending,
                    },
                ..
            }) if *command_id == first_id
        ),
        "a fresh host exact retry must consult recovered durable authority"
    );
    let recovered_exact = resolve_authority_submission(
        &mut exact_frontend,
        &mut exact_core,
        recovered_exact_initial,
        &mut exact_next_nanos,
    );
    assert_eq!(recovered_exact, first_status);
    assert_eq!(exact_core.world_tick(), exact_before_tick);
    assert_eq!(
        exact_core
            .world()
            .world_digest_v1()
            .expect("fresh world digest after cold recovered exact replay"),
        exact_before_digest
    );
    emit_command_authority_evidence(CommandAuthorityEvidence {
        envelope: &first_envelope,
        authority_phase: "recovered_durable_exact",
        cache_result: "miss_cold",
        durable_lookup: "resolved_exact",
        status: Some(&recovered_exact),
        disposition: "authoritative_replay",
        tick: exact_core.world_tick(),
        world_digest: &exact_before_digest.overall,
        recovery: true,
        host_lifecycle: exact_core.latest_snapshot().lifecycle,
    });
    drop(exact_frontend);
    drop(exact_core);
    assert_eq!(
        exact_pipeline
            .shutdown()
            .expect("close cold exact-retry authority writer")
            .guarantee,
        PersistenceGuarantee::Durable
    );
    drop(exact_pipeline);

    let mut recovered =
        StoragePipeline::recover_existing(&path).expect("reopen authority for cold collision");
    assert_eq!(recovered.run_id(), run_id);
    let recovered_journal = recovered
        .journal_port(RECOVERED_SESSION, journal_options)
        .expect("cold collision journal port");
    let mut recovered_core = HostCore::with_journal(
        RECOVERED_SESSION,
        compact_world(),
        host_options,
        Box::new(recovered_journal),
    )
    .expect("fresh collision host over recovered durable authority");
    let mut recovered_frontend = NullFrontend::new(recovered_core.local_port(), CLIENT_NAMESPACE);
    let mut recovered_next_nanos = 0;
    let recovered_before_tick = recovered_core.world_tick();
    let recovered_before_digest = recovered_core
        .world()
        .world_digest_v1()
        .expect("fresh world digest before cold recovered collision");

    let recovered_changed_envelope = CommandEnvelope::new(first_id, HostCommand::Pause);
    let recovered_changed = recovered_frontend.submit_envelope(recovered_changed_envelope.clone());
    assert!(
        matches!(
            &recovered_changed,
            Err(NullFrontendSubmissionError::HostAccess {
                source:
                    HostAccessError::CommandAuthorityLookup {
                        command_id,
                        failure: CommandAuthorityLookupFailure::Pending,
                    },
                ..
            }) if *command_id == first_id
        ),
        "a fresh host changed envelope must consult recovered durable authority"
    );
    resolve_authority_collision(
        &mut recovered_frontend,
        &mut recovered_core,
        first_id,
        recovered_changed,
        &mut recovered_next_nanos,
    );
    let initial_status = recovered_frontend.command_status(first_id);
    let status_after_collision = resolve_authority_status(
        &mut recovered_frontend,
        &mut recovered_core,
        first_id,
        initial_status,
        &mut recovered_next_nanos,
    );
    assert_eq!(status_after_collision, first_status);
    assert_eq!(recovered_core.world_tick(), recovered_before_tick);
    assert_eq!(
        recovered_core
            .world()
            .world_digest_v1()
            .expect("fresh world digest after recovered replay and collision"),
        recovered_before_digest
    );
    emit_command_authority_evidence(CommandAuthorityEvidence {
        envelope: &recovered_changed_envelope,
        authority_phase: "recovered_durable_collision",
        cache_result: "miss_cold",
        durable_lookup: "resolved_collision",
        status: Some(&first_status),
        disposition: "command_id_collision",
        tick: recovered_core.world_tick(),
        world_digest: &recovered_before_digest.overall,
        recovery: true,
        host_lifecycle: recovered_core.latest_snapshot().lifecycle,
    });

    let recovered_exact = recovered_frontend
        .submit_envelope(first_envelope.clone())
        .expect("exact retry uses authority warmed by cold collision resolution");
    assert_eq!(recovered_exact, first_status);
    assert_eq!(recovered_core.world_tick(), recovered_before_tick);
    assert_eq!(
        recovered_core
            .world()
            .world_digest_v1()
            .expect("fresh world digest after recovered collision and exact replay"),
        recovered_before_digest
    );
    emit_command_authority_evidence(CommandAuthorityEvidence {
        envelope: &first_envelope,
        authority_phase: "recovered_cache",
        cache_result: "hit_exact",
        durable_lookup: "not_used",
        status: Some(&recovered_exact),
        disposition: "authoritative_replay",
        tick: recovered_core.world_tick(),
        world_digest: &recovered_before_digest.overall,
        recovery: true,
        host_lifecycle: recovered_core.latest_snapshot().lifecycle,
    });

    let before_fresh_revisions = recovered_core.latest_snapshot().revisions;
    let fresh_envelope = CommandEnvelope::new(
        CommandId::from_client_sequence(CLIENT_NAMESPACE, 5),
        HostCommand::Pause,
    );
    let (fresh_status, fresh_digest) = submit_envelope_durably(
        &mut recovered_frontend,
        &mut recovered_core,
        &fresh_envelope,
        &mut recovered_next_nanos,
    );
    let after_fresh_revisions = recovered_core.latest_snapshot().revisions;
    assert_ne!(
        after_fresh_revisions.control, before_fresh_revisions.control,
        "the genuinely fresh control command applies exactly one control boundary"
    );
    assert_eq!(
        after_fresh_revisions.scientific, before_fresh_revisions.scientific,
        "recovery does not pretend to reconstruct or resume the old scientific world"
    );
    assert_eq!(recovered_core.world_tick(), recovered_before_tick);
    let fresh_replay = recovered_frontend
        .submit_envelope(fresh_envelope.clone())
        .expect("fresh durable command exact retry");
    assert_eq!(fresh_replay, fresh_status);
    assert_eq!(
        recovered_core.latest_snapshot().revisions,
        after_fresh_revisions,
        "the exact retry must not apply a second control boundary"
    );
    assert_eq!(recovered_core.world_tick(), recovered_before_tick);
    assert_eq!(
        recovered_core
            .world()
            .world_digest_v1()
            .expect("fresh command exact retry digest"),
        fresh_digest
    );
    emit_command_authority_evidence(CommandAuthorityEvidence {
        envelope: &fresh_envelope,
        authority_phase: "recovered_fresh_durable",
        cache_result: "hit_exact",
        durable_lookup: "not_used",
        status: Some(&fresh_status),
        disposition: "applied_once",
        tick: recovered_core.world_tick(),
        world_digest: &fresh_digest.overall,
        recovery: true,
        host_lifecycle: recovered_core.latest_snapshot().lifecycle,
    });

    let recovered_shutdown = CommandEnvelope::new(
        CommandId::from_client_sequence(CLIENT_NAMESPACE, 6),
        HostCommand::Shutdown,
    );
    submit_envelope_durably(
        &mut recovered_frontend,
        &mut recovered_core,
        &recovered_shutdown,
        &mut recovered_next_nanos,
    );
    assert_eq!(
        recovered
            .shutdown()
            .expect("recovered command-authority writer shuts down")
            .guarantee,
        PersistenceGuarantee::Durable
    );

    // The uniquely named database is intentionally retained; this test performs no file deletion.
}

#[test]
#[allow(
    clippy::drop_non_drop,
    clippy::too_many_lines,
    reason = "one mock-free proof keeps the concurrent clients, sole owner, durable authority, and independent science oracle visible together"
)]
fn file_channel_concurrent_exact_duplicate_clients_apply_once_and_persist_authority() {
    const CLIENT_NAMESPACE: u64 = 0x2a_63_68_61;
    const SESSION: HostSessionId = HostSessionId::new(0x2a_11);
    const ORACLE_SESSION: HostSessionId = HostSessionId::new(0x2a_12);

    let path = unique_database_path("channel_concurrent_exact_duplicate");
    let journal_options = StorageJournalOptions::default();
    let command_id = CommandId::from_client_sequence(CLIENT_NAMESPACE, 1);
    let envelope = CommandEnvelope::new(command_id, HostCommand::Step);

    let mut oracle = HostCore::new(ORACLE_SESSION, compact_world(), host_options(8))
        .expect("independent persistence-free science oracle");
    let mut oracle_frontend = NullFrontend::new(oracle.local_port(), CLIENT_NAMESPACE);
    let mut oracle_nanos = 0;
    submit_envelope_with_authority(
        &mut oracle_frontend,
        &mut oracle,
        envelope.clone(),
        &mut oracle_nanos,
    );
    let oracle_status = drive_until_journal_state(
        &mut oracle_frontend,
        &mut oracle,
        command_id,
        &JournalState::CommittedVolatile,
        &mut oracle_nanos,
    );
    let oracle_digest = oracle
        .scientific_digest_v1()
        .expect("independent post-Step science digest");
    let ApplicationState::Applied(oracle_applied) = oracle_status.application() else {
        panic!("independent Step oracle did not apply");
    };
    assert_eq!(oracle_applied.tick, Tick(1));
    assert_eq!(oracle_digest.tick, Tick(1));

    let (handoff_tx, handoff_rx) = std::sync::mpsc::sync_channel(1);
    let owner_path = path.clone();
    let owner = thread::spawn(move || {
        let mut world = compact_world();
        world.request_replay_world_digest();
        let mut pipeline = StoragePipeline::create_unattributed_file(&owner_path)
            .expect("new file-backed channel command journal");
        let run_id = pipeline.run_id();
        let journal = pipeline
            .journal_port(SESSION, journal_options)
            .expect("channel owner journal port");
        let core = HostCore::with_journal(SESSION, world, host_options(8), Box::new(journal))
            .expect("file-backed channel owner host");
        let (mut driver, port) = ChannelHostDriver::new(
            FixedDeadlineHost::new(core),
            ChannelHostOptions {
                ingress_capacity: 8,
                ingress_drain_budget: 8,
                status_board_capacity: 8,
                protocol_event_capacity: 16,
                submit_deadline: Duration::from_secs(10),
                maintenance_period: Duration::from_millis(1),
            },
        )
        .expect("bounded channel driver");
        handoff_tx
            .send((port, run_id))
            .expect("hand channel port to concurrent clients");
        let started = Instant::now();
        let run = driver
            .run(move || {
                ManualInstant::from_nanos(
                    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                )
            })
            .expect("channel owner reaches ordered shutdown");
        let guarantee = pipeline
            .shutdown()
            .expect("channel storage pipeline shuts down")
            .guarantee;
        (run, guarantee)
    });
    let (mut coordinator, run_id) = handoff_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("receive channel client from owner thread");

    let barrier = Arc::new(Barrier::new(3));
    let spawn_submitter =
        |mut client: ChannelHostPort, barrier: Arc<Barrier>, envelope: CommandEnvelope| {
            thread::spawn(move || {
                barrier.wait();
                let submitted = client
                    .submit(envelope.clone())
                    .expect("concurrent exact envelope reaches command authority");
                let terminal = wait_channel_command_durable(&mut client, envelope.command_id);
                (submitted, terminal)
            })
        };
    let left = spawn_submitter(coordinator.clone(), Arc::clone(&barrier), envelope.clone());
    let right = spawn_submitter(coordinator.clone(), Arc::clone(&barrier), envelope.clone());
    barrier.wait();
    let (left_submission, left_terminal) = left.join().expect("left client joins");
    let (right_submission, right_terminal) = right.join().expect("right client joins");

    assert_eq!(left_submission.command_id(), command_id);
    assert_eq!(right_submission.command_id(), command_id);
    assert_eq!(
        left_submission.admission_sequence(),
        right_submission.admission_sequence(),
        "both exact submissions must resolve to one admission identity"
    );
    assert_eq!(
        left_terminal, right_terminal,
        "both clients must observe one identical authoritative terminal status"
    );
    assert_eq!(left_terminal.journal(), &JournalState::Durable);
    assert_eq!(
        left_terminal.admission_sequence().map(|value| value.get()),
        Some(1)
    );
    let ApplicationState::Applied(applied) = left_terminal.application() else {
        panic!("concurrent exact Step did not apply");
    };
    let applied = *applied;
    assert_eq!(applied.tick, Tick(1));
    assert_eq!(applied, *oracle_applied);

    let before_collision = coordinator
        .snapshot_after(None)
        .expect("channel snapshot lookup")
        .expect("channel owner published a snapshot");
    assert_eq!(before_collision.last_applied_command, Some(command_id));
    assert_eq!(
        before_collision
            .completed_summary
            .as_ref()
            .map(|summary| summary.tick),
        Some(Tick(1))
    );
    assert_eq!(before_collision.revisions, applied.revisions);

    let changed_envelope = CommandEnvelope::new(command_id, HostCommand::Pause);
    let collision = coordinator
        .submit(changed_envelope.clone())
        .expect_err("changed payload under the same id must collide");
    assert!(
        matches!(
            collision,
            HostAccessError::CommandIdCollision {
                command_id: collision_id,
            } if collision_id == command_id
        ),
        "changed payload must retain the exact colliding command id"
    );
    let after_collision = coordinator
        .snapshot_after(None)
        .expect("post-collision channel snapshot lookup")
        .expect("post-collision snapshot remains available");
    assert_eq!(
        after_collision.as_ref(),
        before_collision.as_ref(),
        "a collision must not apply or publish a second world boundary"
    );

    let shutdown_envelope = CommandEnvelope::new(
        CommandId::from_client_sequence(CLIENT_NAMESPACE, 2),
        HostCommand::Shutdown,
    );
    coordinator
        .submit(shutdown_envelope)
        .expect("ordered channel shutdown admission");
    let (run, guarantee) = owner.join().expect("channel owner thread joins");
    assert_eq!(run.outcome, ChannelRunOutcome::Stopped);
    assert_eq!(
        run.commands_admitted, 2,
        "only the Step and Shutdown envelopes may be admitted"
    );
    assert_eq!(guarantee, PersistenceGuarantee::Durable);

    let reader = StorageReader::open_finished_for_run(&path, run_id)
        .expect("open immutable concurrent-command run");
    let evidence = reader
        .command_journal_evidence(SESSION)
        .expect("concurrent run has normalized command evidence");
    assert_eq!(evidence.command_count, 2);
    assert_eq!(evidence.application_transition_count, 4);
    assert_eq!(evidence.storage_transition_count, 4);

    let record = reader
        .command_journal_record(SESSION, command_id)
        .expect("read exact durable Step authority");
    assert_eq!(record.batch_id, JournalBatchId::new(SESSION, 1));
    assert_eq!(record.lifecycle.envelope(), &envelope);
    assert_eq!(
        record.lifecycle.admission_sequence(),
        left_terminal.admission_sequence()
    );
    assert_eq!(record.terminal_boundary, applied);
    assert_eq!(
        record
            .lifecycle
            .terminal()
            .map(|transition| transition.application()),
        Some(left_terminal.application())
    );
    assert_eq!(
        record.scientific_event_sequence,
        Some(EventSequence::new(1))
    );
    assert_eq!(record.storage_transitions.len(), 2);
    assert_eq!(
        record.storage_transitions[0].kind,
        CommandStorageTransitionKind::CommittedVolatile
    );
    assert_eq!(
        record.storage_transitions[1].kind,
        CommandStorageTransitionKind::Durable
    );

    let page = reader
        .host_journal_session_conformance_page(
            SESSION,
            None,
            4,
            journal_options.max_event_page_bytes,
        )
        .expect("read complete bounded concurrent-command session");
    assert_eq!(
        page.progress.journal,
        HostJournalPrefixes {
            admitted: 2,
            applied: 2,
            committed_volatile: 2,
            durable: 2,
        }
    );
    assert_eq!(
        page.progress.events,
        HostJournalPrefixes {
            admitted: 1,
            applied: 1,
            committed_volatile: 1,
            durable: 1,
        }
    );
    assert_eq!(
        page.progress.shutdown,
        Some(JournalBatchId::new(SESSION, 2))
    );
    assert_eq!(page.records.len(), 2);
    assert_eq!(page.next_after, None);
    assert!(
        page.records
            .iter()
            .all(|record| record.state == HostJournalRecordState::Durable)
    );

    let persisted = reader
        .load_replay_events()
        .expect("load durable concurrent-command replay events");
    let world_digests = persisted
        .iter()
        .filter_map(|entry| match &entry.event.kind {
            ReplayEventKind::WorldDigest { overall } => Some((entry.tick, overall.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(world_digests.len(), 1);
    let (persisted_tick, persisted_digest) = world_digests[0];
    assert_eq!(persisted_tick, applied.tick.0);
    assert_eq!(
        persisted_digest, oracle_digest.overall,
        "durable world anchor must match the independent single-Step oracle"
    );

    emit_command_authority_evidence(CommandAuthorityEvidence {
        envelope: &envelope,
        authority_phase: "concurrent_client_left",
        cache_result: "miss_or_live_exact_race",
        durable_lookup: "claimed_or_not_consulted_race",
        status: Some(&left_terminal),
        disposition: "authoritative_terminal",
        tick: applied.tick,
        world_digest: persisted_digest,
        recovery: false,
        host_lifecycle: HostLifecycle::Running,
    });
    emit_command_authority_evidence(CommandAuthorityEvidence {
        envelope: &envelope,
        authority_phase: "concurrent_client_right",
        cache_result: "miss_or_live_exact_race",
        durable_lookup: "claimed_or_not_consulted_race",
        status: Some(&right_terminal),
        disposition: "authoritative_terminal",
        tick: applied.tick,
        world_digest: persisted_digest,
        recovery: false,
        host_lifecycle: HostLifecycle::Running,
    });
    emit_command_authority_evidence(CommandAuthorityEvidence {
        envelope: &changed_envelope,
        authority_phase: "concurrent_cache_collision",
        cache_result: "hit_collision",
        durable_lookup: "not_used",
        status: Some(&left_terminal),
        disposition: "command_id_collision",
        tick: applied.tick,
        world_digest: persisted_digest,
        recovery: false,
        host_lifecycle: HostLifecycle::Running,
    });
    emit_command_authority_evidence(CommandAuthorityEvidence {
        envelope: &envelope,
        authority_phase: "finished_durable_database",
        cache_result: "not_observed",
        durable_lookup: "finished_reader_exact",
        status: Some(&left_terminal),
        disposition: "single_admission_single_application",
        tick: applied.tick,
        world_digest: persisted_digest,
        recovery: false,
        host_lifecycle: HostLifecycle::Stopped,
    });
    reader
        .close()
        .expect("close immutable concurrent-command reader");

    // The uniquely named database is intentionally retained; this test performs no file deletion.
}

#[test]
#[allow(
    clippy::drop_non_drop,
    clippy::too_many_lines,
    reason = "the mock-free two-run proof keeps command order, durable readback, replay, and per-boundary science digests visible in one audit"
)]
fn file_journal_replays_gui_world_edits_with_identical_digests_and_receipts() {
    const GUI_CLIENT_NAMESPACE: u64 = 0x37_6f_70_75_69;
    const FIRST_SESSION: HostSessionId = HostSessionId::new(0x37_01);
    const REPLAY_SESSION: HostSessionId = HostSessionId::new(0x37_02);

    let first_path = unique_database_path("gui_world_edits_first");
    let replay_path = unique_database_path("gui_world_edits_replay");
    let options = StorageJournalOptions::default();
    let mut first_pipeline = StoragePipeline::create_unattributed_file(&first_path)
        .expect("new file-backed GUI command journal");
    let first_run_id = first_pipeline.run_id();
    let first_journal = first_pipeline
        .journal_port(FIRST_SESSION, options)
        .expect("first durable GUI command journal port");
    let mut first_core = HostCore::with_journal(
        FIRST_SESSION,
        command_replay_world(),
        host_options(8),
        Box::new(first_journal),
    )
    .expect("first production HostCore boundary");
    let mut first_frontend = NullFrontend::new(first_core.local_port(), GUI_CLIENT_NAMESPACE);
    let mut first_next_nanos = 0;
    let mut original_envelopes = Vec::new();
    let mut original_statuses = Vec::new();
    let mut original_digests = Vec::new();

    for (sequence, command) in [
        (
            1,
            ControlCommand::SpawnAgent {
                herbivore_tendency: 1.0,
            },
        ),
        (
            2,
            ControlCommand::SpawnAgent {
                herbivore_tendency: 0.0,
            },
        ),
        (3, ControlCommand::Step),
    ] {
        let envelope = CommandEnvelope::new(
            CommandId::from_client_sequence(GUI_CLIENT_NAMESPACE, sequence),
            map_gui_command(command),
        );
        let (status, digest) = submit_envelope_durably(
            &mut first_frontend,
            &mut first_core,
            &envelope,
            &mut first_next_nanos,
        );
        original_envelopes.push(envelope);
        original_statuses.push(status);
        original_digests.push(digest);
    }

    let parents = sorted_agent_identities(&first_core);
    assert_eq!(
        parents.len(),
        2,
        "the controlled fixture must expose exactly the two injected parents"
    );
    let selection = SelectionUpdate {
        mode: SelectionMode::Replace,
        agent_ids: parents.iter().map(|(_, raw_id)| *raw_id).collect(),
        state: SelectionState::Selected,
    };
    let mut closed_config = first_core.world().config().clone();
    closed_config.closed = true;
    for (sequence, command) in [
        (4, ControlCommand::UpdateSelection(selection)),
        (
            5,
            ControlCommand::AdjustAgentMutationRates {
                agent_uid: parents[0].0,
                delta_primary: 0.002,
                delta_secondary: -0.01,
            },
        ),
        (6, ControlCommand::UpdateConfig(Box::new(closed_config))),
        (
            7,
            ControlCommand::SpawnCrossover {
                parent_a: parents[0].0,
                parent_b: parents[1].0,
            },
        ),
        (8, ControlCommand::Step),
        (9, ControlCommand::Shutdown),
    ] {
        let envelope = CommandEnvelope::new(
            CommandId::from_client_sequence(GUI_CLIENT_NAMESPACE, sequence),
            map_gui_command(command),
        );
        let (status, digest) = submit_envelope_durably(
            &mut first_frontend,
            &mut first_core,
            &envelope,
            &mut first_next_nanos,
        );
        original_envelopes.push(envelope);
        original_statuses.push(status);
        original_digests.push(digest);
    }
    assert_eq!(
        first_core.latest_snapshot().lifecycle,
        HostLifecycle::Stopped
    );
    assert_eq!(
        first_core.world().agent_count(),
        3,
        "the crossover command must add exactly one child"
    );
    assert!(
        first_core.world().config().closed,
        "the GUI-equivalent closed-world config command must survive the interleaved step"
    );
    let first_final_digest = first_core
        .world()
        .world_digest_v1()
        .expect("first final world digest");
    let first_shutdown_receipt = first_pipeline
        .shutdown()
        .expect("first file-backed pipeline shuts down cleanly");
    assert_eq!(
        first_shutdown_receipt.guarantee,
        PersistenceGuarantee::Durable
    );
    drop(first_frontend);
    drop(first_core);
    drop(first_pipeline);

    let first_reader = StorageReader::open_finished_for_run(&first_path, first_run_id)
        .expect("open immutable first command journal");
    let command_count =
        u64::try_from(original_envelopes.len()).expect("bounded command count fits u64");
    let first_evidence = first_reader
        .command_journal_evidence(FIRST_SESSION)
        .expect("first run has complete command evidence");
    assert_eq!(first_evidence.command_count, command_count);
    assert_eq!(
        first_evidence.application_transition_count,
        command_count * 2
    );
    assert_eq!(first_evidence.storage_transition_count, command_count * 2);
    let first_page = first_reader
        .command_journal_page(
            FIRST_SESSION,
            None,
            original_envelopes.len() + 1,
            options.max_event_page_bytes,
        )
        .expect("read all first-run GUI commands in journal order");
    assert_eq!(first_page.next_after, None);
    assert_eq!(first_page.commands.len(), original_envelopes.len());
    assert_eq!(first_page.evidence, first_evidence);
    let readback_envelopes = first_page
        .commands
        .iter()
        .map(|record| record.lifecycle.envelope().clone())
        .collect::<Vec<_>>();
    assert_eq!(
        readback_envelopes, original_envelopes,
        "durable decoding must preserve exact GUI envelope order and payloads"
    );
    for (index, record) in first_page.commands.iter().enumerate() {
        assert_eq!(
            record.batch_id.sequence(),
            u64::try_from(index + 1).expect("bounded journal sequence")
        );
        assert_eq!(
            record.lifecycle.admission_sequence(),
            original_statuses[index].admission_sequence()
        );
        assert_eq!(record.lifecycle.transitions().len(), 2);
        assert_eq!(
            record
                .lifecycle
                .terminal()
                .map(|transition| transition.application()),
            Some(original_statuses[index].application())
        );
        assert_eq!(record.storage_transitions.len(), 2);
        assert_eq!(record.storage_transitions[0].ordinal, 0);
        assert_eq!(
            record.storage_transitions[0].kind,
            CommandStorageTransitionKind::CommittedVolatile
        );
        assert_eq!(record.storage_transitions[1].ordinal, 1);
        assert_eq!(
            record.storage_transitions[1].kind,
            CommandStorageTransitionKind::Durable
        );
        assert_eq!(record.archive_payload_digest.len(), 64);
    }
    first_reader
        .close()
        .expect("close immutable first command journal");

    let mut replay_pipeline = StoragePipeline::create_unattributed_file(&replay_path)
        .expect("new file-backed replay command journal");
    let replay_run_id = replay_pipeline.run_id();
    let replay_journal = replay_pipeline
        .journal_port(REPLAY_SESSION, options)
        .expect("replay durable GUI command journal port");
    let mut replay_core = HostCore::with_journal(
        REPLAY_SESSION,
        command_replay_world(),
        host_options(8),
        Box::new(replay_journal),
    )
    .expect("replay production HostCore boundary");
    let mut replay_frontend = NullFrontend::new(replay_core.local_port(), GUI_CLIENT_NAMESPACE);
    let mut replay_next_nanos = 0;
    let mut replay_statuses = Vec::with_capacity(readback_envelopes.len());
    let mut replay_digests = Vec::with_capacity(readback_envelopes.len());
    for envelope in &readback_envelopes {
        let (status, digest) = submit_envelope_durably(
            &mut replay_frontend,
            &mut replay_core,
            envelope,
            &mut replay_next_nanos,
        );
        replay_statuses.push(status);
        replay_digests.push(digest);
    }
    assert_eq!(
        replay_statuses, original_statuses,
        "replayed envelopes must retain exact admission and terminal boundaries"
    );
    assert_eq!(
        replay_digests, original_digests,
        "every replayed command boundary must retain WorldDigestV1"
    );
    assert_eq!(
        replay_core
            .world()
            .world_digest_v1()
            .expect("replay final world digest"),
        first_final_digest
    );
    assert_eq!(
        replay_core.latest_snapshot().lifecycle,
        HostLifecycle::Stopped
    );
    let replay_shutdown_receipt = replay_pipeline
        .shutdown()
        .expect("replay file-backed pipeline shuts down cleanly");
    assert_eq!(
        replay_shutdown_receipt, first_shutdown_receipt,
        "identical command/science work must produce identical persistence receipts"
    );
    drop(replay_frontend);
    drop(replay_core);
    drop(replay_pipeline);

    let replay_reader = StorageReader::open_finished_for_run(&replay_path, replay_run_id)
        .expect("open immutable replay command journal");
    let replay_page = replay_reader
        .command_journal_page(
            REPLAY_SESSION,
            None,
            readback_envelopes.len() + 1,
            options.max_event_page_bytes,
        )
        .expect("read all replayed GUI commands in journal order");
    assert_eq!(replay_page.next_after, None);
    assert_eq!(replay_page.evidence, first_evidence);
    assert_eq!(replay_page.commands.len(), first_page.commands.len());
    for (first, replayed) in first_page.commands.iter().zip(&replay_page.commands) {
        assert_eq!(replayed.batch_id.sequence(), first.batch_id.sequence());
        assert_eq!(replayed.lifecycle, first.lifecycle);
        assert_eq!(replayed.terminal_boundary, first.terminal_boundary);
        assert_eq!(
            replayed.scientific_event_sequence,
            first.scientific_event_sequence
        );
        assert_eq!(replayed.storage_transitions, first.storage_transitions);
        assert_eq!(replayed.archive_payload_digest.len(), 64);
    }
    replay_reader
        .close()
        .expect("close immutable replay command journal");

    // The uniquely named databases are intentionally retained; this test performs no file deletion.
}

#[test]
#[allow(
    clippy::drop_non_drop,
    clippy::too_many_lines,
    reason = "one durable public-boundary matrix proves rejected, failed, control-only, configuration, scientific, and shutdown command lifecycles plus both transition axes"
)]
fn finished_command_reader_preserves_complete_lifecycle_and_storage_evidence() {
    let path = unique_database_path("command_lifecycle_evidence");
    let mut pipeline = StoragePipeline::create_unattributed_file(&path)
        .expect("new uniquely named command-evidence database");
    let run_id = pipeline.run_id();
    let session_id = HostSessionId::new(0x1052);
    let options = StorageJournalOptions::default();
    let journal = pipeline
        .journal_port(session_id, options)
        .expect("file command-evidence journal port");
    let mut core = HostCore::with_journal(
        session_id,
        compact_world(),
        host_options(8),
        Box::new(journal),
    )
    .expect("host backed by durable command evidence");
    let client_namespace = 0x2052;
    let mut frontend = NullFrontend::new(core.local_port(), client_namespace);
    let mut next_nanos = 0;

    let nan_bits = 0x7fc0_5252_u32;
    let pre_admission_rejection = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::SetSpeed(f32::from_bits(nan_bits)),
        None,
        &mut next_nanos,
    );
    let pre_admission_id = pre_admission_rejection.command_id();
    let pre_admission_status = drive_until_journal_state(
        &mut frontend,
        &mut core,
        pre_admission_id,
        &JournalState::Durable,
        &mut next_nanos,
    );
    assert!(matches!(
        pre_admission_status.application(),
        ApplicationState::Rejected(RejectionReason::Validation { .. })
    ));

    let rejected_shutdown = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::Shutdown,
        Some(ControlRevision::new(u64::MAX)),
        &mut next_nanos,
    );
    let rejected_shutdown_id = rejected_shutdown.command_id();
    let rejected_shutdown_status = drive_until_journal_state(
        &mut frontend,
        &mut core,
        rejected_shutdown_id,
        &JournalState::Durable,
        &mut next_nanos,
    );
    assert!(matches!(
        rejected_shutdown_status.application(),
        ApplicationState::Rejected(RejectionReason::ControlRevisionConflict { .. })
    ));
    assert_ne!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);

    let pause = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::Pause,
        None,
        &mut next_nanos,
    );
    let pause_id = pause.command_id();
    let pause_status = drive_until_journal_state(
        &mut frontend,
        &mut core,
        pause_id,
        &JournalState::Durable,
        &mut next_nanos,
    );
    let ApplicationState::Applied(pause_boundary) = pause_status.application() else {
        panic!("pause did not retain its applied boundary");
    };
    let pause_boundary = *pause_boundary;

    let updated_config = ScriptBotsConfig {
        world_width: 64,
        world_height: 64,
        food_cell_size: 16,
        rng_seed: Some(0x5eed),
        closed: true,
        history_capacity: 8,
        persistence_interval: 1,
        food_growth_rate: 0.0025,
        ..ScriptBotsConfig::default()
    };
    let expected_updated_config = updated_config.clone();
    let update_id = CommandId::from_client_sequence(client_namespace, 10_004);
    let update_envelope = CommandEnvelope::new(
        update_id,
        HostCommand::UpdateConfig(Box::new(updated_config)),
    )
    .expecting_control_revision(pause_boundary.revisions.control)
    .expecting_scientific_revision(pause_boundary.revisions.scientific)
    .expecting_config_revision(pause_boundary.revisions.config);
    submit_envelope_with_authority(&mut frontend, &mut core, update_envelope, &mut next_nanos);
    let update_status = drive_until_journal_state(
        &mut frontend,
        &mut core,
        update_id,
        &JournalState::Durable,
        &mut next_nanos,
    );
    let ApplicationState::Applied(update_boundary) = update_status.application() else {
        panic!("update-config did not retain its applied boundary");
    };
    let update_boundary = *update_boundary;

    let mut failing_config = expected_updated_config.clone();
    failing_config.world_width = 80;
    let expected_failing_config = failing_config.clone();
    let failed_config_id = CommandId::from_client_sequence(client_namespace, 10_005);
    let failed_config_envelope = CommandEnvelope::new(
        failed_config_id,
        HostCommand::UpdateConfig(Box::new(failing_config)),
    )
    .expecting_control_revision(update_boundary.revisions.control)
    .expecting_scientific_revision(update_boundary.revisions.scientific)
    .expecting_config_revision(update_boundary.revisions.config);
    submit_envelope_with_authority(
        &mut frontend,
        &mut core,
        failed_config_envelope,
        &mut next_nanos,
    );
    let failed_config_status = drive_until_journal_state(
        &mut frontend,
        &mut core,
        failed_config_id,
        &JournalState::Durable,
        &mut next_nanos,
    );
    let ApplicationState::Failed(failed_config_application) = failed_config_status.application()
    else {
        panic!("live-geometry config update did not retain its application failure");
    };
    assert_eq!(failed_config_application.code, "config_application");
    assert_eq!(
        failed_config_application.message,
        "invalid configuration: changing world dimensions at runtime is not supported; restart with the new configuration"
    );

    let step_id = CommandId::from_client_sequence(client_namespace, 10_006);
    let step_envelope = CommandEnvelope::new(step_id, HostCommand::Step)
        .expecting_control_revision(update_boundary.revisions.control)
        .expecting_scientific_revision(update_boundary.revisions.scientific)
        .expecting_config_revision(update_boundary.revisions.config);
    submit_envelope_with_authority(&mut frontend, &mut core, step_envelope, &mut next_nanos);
    let step_status = drive_until_journal_state(
        &mut frontend,
        &mut core,
        step_id,
        &JournalState::Durable,
        &mut next_nanos,
    );
    let ApplicationState::Applied(step_boundary) = step_status.application() else {
        panic!("step did not retain its applied boundary");
    };
    let step_boundary = *step_boundary;
    assert_eq!(core.world_tick(), Tick(1));

    let shutdown_id = CommandId::from_client_sequence(client_namespace, 10_007);
    let shutdown_envelope = CommandEnvelope::new(shutdown_id, HostCommand::Shutdown)
        .expecting_control_revision(step_boundary.revisions.control)
        .expecting_scientific_revision(step_boundary.revisions.scientific)
        .expecting_config_revision(step_boundary.revisions.config);
    submit_envelope_with_authority(&mut frontend, &mut core, shutdown_envelope, &mut next_nanos);
    drive_until_journal_state(
        &mut frontend,
        &mut core,
        shutdown_id,
        &JournalState::Durable,
        &mut next_nanos,
    );
    assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);
    assert_eq!(
        pipeline
            .shutdown()
            .expect("close durable command-evidence storage")
            .guarantee,
        PersistenceGuarantee::Durable
    );
    drop(frontend);
    drop(core);
    drop(pipeline);

    let reader = StorageReader::open_finished_for_run(&path, run_id)
        .expect("reopen immutable command evidence");
    let evidence = reader
        .command_journal_evidence(session_id)
        .expect("finished session contains non-vacuous command evidence");
    assert_eq!(evidence.command_count, 7);
    assert_eq!(evidence.application_transition_count, 13);
    assert_eq!(evidence.storage_transition_count, 14);

    let pre_admission = reader
        .command_journal_record(session_id, pre_admission_id)
        .expect("exact pre-admission rejection lookup");
    assert_eq!(pre_admission.batch_id.sequence(), 1);
    assert_eq!(
        pre_admission.lifecycle.source_client_namespace(),
        client_namespace
    );
    assert_eq!(pre_admission.lifecycle.admission_sequence(), None);
    assert_eq!(pre_admission.lifecycle.transitions().len(), 1);
    assert!(matches!(
        pre_admission
            .lifecycle
            .terminal()
            .map(|transition| transition.application()),
        Some(ApplicationState::Rejected(
            RejectionReason::Validation { .. }
        ))
    ));
    let HostCommand::SetSpeed(stored_speed) = &pre_admission.lifecycle.envelope().command else {
        panic!("pre-admission lifecycle changed the rejected command kind");
    };
    assert_eq!(stored_speed.to_bits(), nan_bits);
    assert_eq!(pre_admission.scientific_event_sequence, None);

    let rejected_shutdown_record = reader
        .command_journal_record(session_id, rejected_shutdown_id)
        .expect("exact admitted rejection lookup");
    assert_eq!(rejected_shutdown_record.batch_id.sequence(), 2);
    assert!(
        rejected_shutdown_record
            .lifecycle
            .admission_sequence()
            .is_some()
    );
    assert_eq!(rejected_shutdown_record.lifecycle.transitions().len(), 2);
    assert!(matches!(
        rejected_shutdown_record
            .lifecycle
            .terminal()
            .map(|transition| transition.application()),
        Some(ApplicationState::Rejected(
            RejectionReason::ControlRevisionConflict { .. }
        ))
    ));
    assert!(!rejected_shutdown_record.lifecycle.is_applied_shutdown());
    assert_eq!(rejected_shutdown_record.scientific_event_sequence, None);
    assert_eq!(
        rejected_shutdown_record
            .lifecycle
            .envelope()
            .expected_control_revision,
        Some(ControlRevision::new(u64::MAX))
    );
    assert_eq!(
        rejected_shutdown_record
            .lifecycle
            .envelope()
            .expected_scientific_revision,
        None
    );
    assert_eq!(
        rejected_shutdown_record
            .lifecycle
            .envelope()
            .expected_config_revision,
        None
    );

    let pause_record = reader
        .command_journal_record(session_id, pause_id)
        .expect("exact applied control lookup");
    assert!(matches!(
        &pause_record.lifecycle.envelope().command,
        HostCommand::Pause
    ));
    assert!(matches!(
        pause_record
            .lifecycle
            .terminal()
            .map(|transition| transition.application()),
        Some(ApplicationState::Applied(_))
    ));
    assert_eq!(pause_record.scientific_event_sequence, None);

    let update_record = reader
        .command_journal_record(session_id, update_id)
        .expect("exact applied configuration lookup");
    let HostCommand::UpdateConfig(stored_config) = &update_record.lifecycle.envelope().command
    else {
        panic!("configuration lifecycle changed the command kind");
    };
    assert_eq!(stored_config.as_ref(), &expected_updated_config);
    assert_eq!(
        stored_config.food_growth_rate.to_bits(),
        expected_updated_config.food_growth_rate.to_bits(),
        "postcard command projection must preserve exact configuration float bits"
    );
    assert!(matches!(
        update_record
            .lifecycle
            .terminal()
            .map(|transition| transition.application()),
        Some(ApplicationState::Applied(_))
    ));
    assert_eq!(
        update_record.lifecycle.envelope().expected_control_revision,
        Some(pause_boundary.revisions.control)
    );
    assert_eq!(
        update_record
            .lifecycle
            .envelope()
            .expected_scientific_revision,
        Some(pause_boundary.revisions.scientific)
    );
    assert_eq!(
        update_record.lifecycle.envelope().expected_config_revision,
        Some(pause_boundary.revisions.config)
    );
    assert_eq!(update_record.scientific_event_sequence, None);

    let failed_config_record = reader
        .command_journal_record(session_id, failed_config_id)
        .expect("exact failed configuration lookup");
    assert_eq!(failed_config_record.batch_id.sequence(), 5);
    assert_eq!(
        failed_config_record.lifecycle.source_client_namespace(),
        client_namespace
    );
    assert!(
        failed_config_record
            .lifecycle
            .admission_sequence()
            .is_some()
    );
    let HostCommand::UpdateConfig(stored_failing_config) =
        &failed_config_record.lifecycle.envelope().command
    else {
        panic!("failed configuration lifecycle changed the command kind");
    };
    assert_eq!(stored_failing_config.as_ref(), &expected_failing_config);
    assert_eq!(
        failed_config_record
            .lifecycle
            .envelope()
            .expected_control_revision,
        Some(update_boundary.revisions.control)
    );
    assert_eq!(
        failed_config_record
            .lifecycle
            .envelope()
            .expected_scientific_revision,
        Some(update_boundary.revisions.scientific)
    );
    assert_eq!(
        failed_config_record
            .lifecycle
            .envelope()
            .expected_config_revision,
        Some(update_boundary.revisions.config)
    );
    assert_eq!(failed_config_record.lifecycle.transitions().len(), 2);
    assert_eq!(failed_config_record.lifecycle.transitions()[0].ordinal(), 0);
    assert_eq!(
        failed_config_record.lifecycle.transitions()[0].boundary(),
        update_boundary
    );
    assert_eq!(
        failed_config_record.lifecycle.transitions()[0].application(),
        &ApplicationState::Admitted
    );
    let failed_terminal = &failed_config_record.lifecycle.transitions()[1];
    assert_eq!(failed_terminal.ordinal(), 1);
    assert_eq!(failed_terminal.boundary(), update_boundary);
    let ApplicationState::Failed(stored_failure) = failed_terminal.application() else {
        panic!("finished reader changed the failed application state");
    };
    assert_eq!(stored_failure.code, "config_application");
    assert_eq!(
        stored_failure.message,
        "invalid configuration: changing world dimensions at runtime is not supported; restart with the new configuration"
    );
    assert_eq!(failed_config_record.terminal_boundary, update_boundary);
    assert_eq!(failed_config_record.scientific_event_sequence, None);
    assert_eq!(failed_config_record.archive_payload_digest.len(), 64);

    let step_record = reader
        .command_journal_record(session_id, step_id)
        .expect("exact applied scientific lookup");
    assert!(matches!(
        &step_record.lifecycle.envelope().command,
        HostCommand::Step
    ));
    assert_eq!(
        step_record.lifecycle.envelope().expected_control_revision,
        Some(update_boundary.revisions.control)
    );
    assert_eq!(
        step_record
            .lifecycle
            .envelope()
            .expected_scientific_revision,
        Some(update_boundary.revisions.scientific)
    );
    assert_eq!(
        step_record.lifecycle.envelope().expected_config_revision,
        Some(update_boundary.revisions.config)
    );
    assert_eq!(
        step_record.scientific_event_sequence,
        Some(EventSequence::new(1))
    );

    let shutdown_record = reader
        .command_journal_record(session_id, shutdown_id)
        .expect("exact applied shutdown lookup");
    assert!(shutdown_record.lifecycle.is_applied_shutdown());
    assert_eq!(shutdown_record.batch_id.sequence(), 7);
    assert_eq!(
        shutdown_record
            .lifecycle
            .envelope()
            .expected_control_revision,
        Some(step_boundary.revisions.control)
    );
    assert_eq!(
        shutdown_record
            .lifecycle
            .envelope()
            .expected_scientific_revision,
        Some(step_boundary.revisions.scientific)
    );
    assert_eq!(
        shutdown_record
            .lifecycle
            .envelope()
            .expected_config_revision,
        Some(step_boundary.revisions.config)
    );
    for record in [
        &pre_admission,
        &rejected_shutdown_record,
        &pause_record,
        &update_record,
        &failed_config_record,
        &step_record,
        &shutdown_record,
    ] {
        assert_eq!(record.storage_transitions.len(), 2);
        assert_eq!(record.storage_transitions[0].ordinal, 0);
        assert_eq!(
            record.storage_transitions[0].kind,
            CommandStorageTransitionKind::CommittedVolatile
        );
        assert_eq!(record.storage_transitions[1].ordinal, 1);
        assert_eq!(
            record.storage_transitions[1].kind,
            CommandStorageTransitionKind::Durable
        );
        assert_eq!(record.archive_payload_digest.len(), 64);
    }

    let first_page = reader
        .command_journal_page(session_id, None, 2, options.max_event_page_bytes)
        .expect("first bounded command page");
    assert_eq!(first_page.commands.len(), 2);
    assert_eq!(first_page.evidence, evidence);
    let cursor = first_page
        .next_after
        .expect("seven commands require a second page");
    assert_eq!(cursor, first_page.commands[1].cursor());
    let second_page = reader
        .command_journal_page(session_id, Some(cursor), 4, options.max_event_page_bytes)
        .expect("resume at exact normalized command cursor");
    assert_eq!(second_page.commands.len(), 4);
    assert_eq!(second_page.evidence, evidence);
    assert_eq!(second_page.commands[2], failed_config_record);
    let final_cursor = second_page
        .next_after
        .expect("the bounded second page leaves the shutdown record");
    assert_eq!(final_cursor, second_page.commands[3].cursor());
    let final_page = reader
        .command_journal_page(
            session_id,
            Some(final_cursor),
            1,
            options.max_event_page_bytes,
        )
        .expect("resume at the exact final command cursor");
    assert_eq!(final_page.commands.len(), 1);
    assert_eq!(final_page.commands[0], shutdown_record);
    assert_eq!(final_page.evidence, evidence);
    assert_eq!(final_page.next_after, None);

    let fabricated = CommandJournalCursor {
        command_id: shutdown_id,
        ..cursor
    };
    let fabricated_error = reader
        .command_journal_page(
            session_id,
            Some(fabricated),
            1,
            options.max_event_page_bytes,
        )
        .expect_err("fabricated command cursor cannot skip evidence");
    assert!(matches!(
        fabricated_error,
        StorageError::InvalidData {
            context: "host_command_records.after",
            ..
        }
    ));
    let missing = reader
        .command_journal_record(
            session_id,
            CommandId::from_client_sequence(client_namespace, 10_000),
        )
        .expect_err("exact missing command lookup is non-vacuous");
    assert!(matches!(
        missing,
        StorageError::NoEvidence {
            context: "host_command_records.command_id",
            ..
        }
    ));

    reader
        .close()
        .expect("close immutable command-evidence reader");

    let corrupted_path = unique_database_path("command_transition_ordinal_corruption");
    fs::copy(&path, &corrupted_path).expect("copy finished command database for corruption proof");
    let connection = Connection::open(&corrupted_path)
        .expect("open copied command database for deliberate corruption");
    let changed = connection
        .execute_with_params(
            "UPDATE host_command_application_transitions
             SET transition_ordinal = 2
             WHERE command_id = ?1 AND transition_ordinal = 1",
            &[pause_id.to_string().into()],
        )
        .expect("tamper one application transition ordinal");
    assert_eq!(changed, 1);
    connection
        .close()
        .expect("close deliberately corrupted command database");
    let corruption = match StorageReader::open_finished_for_run(&corrupted_path, run_id) {
        Err(error) => error,
        Ok(reader) => {
            reader
                .close()
                .expect("close unexpectedly admitted corrupted command reader");
            panic!("finished reader admitted a command transition ordinal gap");
        }
    };
    assert!(
        matches!(
            corruption,
            StorageError::InvalidData {
                context: "host_command_application_transitions",
                ..
            } | StorageError::InvalidData {
                context: "host_command_application_transitions.transition_ordinal",
                ..
            }
        ),
        "unexpected command-corruption refusal: {corruption}"
    );

    // The uniquely named database is intentionally retained; this test performs no file deletion.
}

#[test]
#[allow(
    clippy::drop_non_drop,
    clippy::too_many_lines,
    reason = "one public-boundary test proves per-tick lifecycle/combat projection, deferred persistence cadence, durable reopen, typed non-vacuity, and bounded cursor paging together"
)]
fn file_domain_journal_preserves_lifecycle_and_combat_across_deferred_persistence() {
    let path = unique_database_path("domain_event_cadence");
    let mut pipeline = StoragePipeline::create_unattributed_file(&path)
        .expect("new uniquely named file storage pipeline");
    let run_id = pipeline.run_id();
    let session_id = HostSessionId::new(0x1004);
    let options = StorageJournalOptions::default();
    let journal = pipeline
        .journal_port(session_id, options)
        .expect("file domain journal port");
    let mut core = HostCore::with_journal(
        session_id,
        eventful_cadence_world(),
        host_options(8),
        Box::new(journal),
    )
    .expect("host backed by the durable domain journal");
    let mut frontend = NullFrontend::new(core.local_port(), 0x2004);
    let mut next_nanos = 0;

    let first = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::Step,
        None,
        &mut next_nanos,
    );
    drive_until_journal_state(
        &mut frontend,
        &mut core,
        first.command_id(),
        &JournalState::Durable,
        &mut next_nanos,
    );
    let second = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::Step,
        None,
        &mut next_nanos,
    );
    drive_until_journal_state(
        &mut frontend,
        &mut core,
        second.command_id(),
        &JournalState::Durable,
        &mut next_nanos,
    );
    assert_eq!(core.world_tick(), Tick(2));

    let shutdown = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::Shutdown,
        None,
        &mut next_nanos,
    );
    drive_until_journal_state(
        &mut frontend,
        &mut core,
        shutdown.command_id(),
        &JournalState::Durable,
        &mut next_nanos,
    );
    assert_eq!(
        pipeline
            .shutdown()
            .expect("close durable domain storage")
            .guarantee,
        PersistenceGuarantee::Durable
    );
    drop(frontend);
    drop(core);
    drop(pipeline);

    let reader = StorageReader::open_finished_for_run(&path, run_id)
        .expect("reopen immutable domain-event evidence");
    let journal_page = reader
        .host_journal_session_conformance_page(session_id, None, 16, options.max_event_page_bytes)
        .expect("load exact Step/Step/Shutdown canonical archives");
    assert_eq!(journal_page.records.len(), 3);
    assert_eq!(journal_page.progress.events.durable, 2);

    let mut expected_payloads = Vec::new();
    for record in &journal_page.records {
        let Some(event) = &record.event else {
            continue;
        };
        expected_payloads.extend(
            event
                .event
                .boundary
                .births()
                .iter()
                .cloned()
                .map(DomainEventPayload::Birth),
        );
        expected_payloads.extend(
            event
                .event
                .boundary
                .deaths()
                .iter()
                .cloned()
                .map(DomainEventPayload::Death),
        );
        let combat = event.event.boundary.combat();
        if combat.spike_attempts != 0 || combat.spike_hits != 0 {
            expected_payloads.push(DomainEventPayload::Combat(combat));
        }
    }
    assert!(
        expected_payloads
            .iter()
            .any(|payload| matches!(payload, DomainEventPayload::Birth(_))),
        "the first two pre-cadence scientific boundaries must retain scheduled arrivals"
    );
    assert!(
        expected_payloads
            .iter()
            .any(|payload| matches!(payload, DomainEventPayload::Death(_))),
        "the first two pre-cadence scientific boundaries must retain combat deaths"
    );
    assert!(
        expected_payloads.iter().any(|payload| matches!(
            payload,
            DomainEventPayload::Combat(combat)
                if combat.spike_attempts != 0 && combat.spike_hits != 0
        )),
        "the first two pre-cadence scientific boundaries must retain nonzero aggregate combat"
    );

    let evidence = reader
        .domain_event_evidence(session_id, DomainEventExpectation::RequireNonEmpty)
        .expect("scenario-declared lifecycle/combat evidence is non-vacuous");
    assert_eq!(evidence.scientific_event_count, 2);
    assert_eq!(
        evidence.domain_event_count,
        u64::try_from(expected_payloads.len()).expect("bounded test evidence count fits u64")
    );
    let domain_page = reader
        .domain_event_page(session_id, None, 4_096, options.max_event_page_bytes)
        .expect("load the complete bounded typed domain-event page");
    assert_eq!(domain_page.next_after, None);
    assert_eq!(domain_page.evidence, evidence);
    assert_eq!(
        domain_page
            .events
            .iter()
            .map(|event| event.payload.clone())
            .collect::<Vec<_>>(),
        expected_payloads,
        "normalized rows must reproduce the complete canonical boundary payloads"
    );
    for event in &domain_page.events {
        assert_eq!(event.archive_payload_digest.len(), 64);
        assert!(
            event
                .archive_payload_digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        );
        assert!(matches!(event.journal_batch_id.sequence(), 1 | 2));
    }

    let first_page = reader
        .domain_event_page(session_id, None, 1, options.max_event_page_bytes)
        .expect("read first single-row domain page");
    assert_eq!(first_page.events.len(), 1);
    let first_cursor = first_page
        .next_after
        .expect("nontrivial evidence has a second domain row");
    assert_eq!(first_cursor, first_page.events[0].cursor());
    let second_page = reader
        .domain_event_page(
            session_id,
            Some(first_cursor),
            1,
            options.max_event_page_bytes,
        )
        .expect("resume after the exact typed domain cursor");
    assert_eq!(second_page.events.len(), 1);
    assert_ne!(second_page.events[0].cursor(), first_cursor);

    let fabricated_cursor = scriptbots_storage::DomainEventCursor {
        event_ordinal: first_cursor.event_ordinal.saturating_add(10_000),
        ..first_cursor
    };
    let cursor_error = reader
        .domain_event_page(
            session_id,
            Some(fabricated_cursor),
            1,
            options.max_event_page_bytes,
        )
        .expect_err("a fabricated cursor cannot skip normalized evidence");
    assert!(matches!(
        cursor_error,
        StorageError::InvalidData {
            context: "host_domain_events.after",
            ..
        }
    ));

    reader.close().expect("close immutable domain-event reader");

    let domain_corruptions = [
        (
            "domain_sequence_corruption",
            "UPDATE host_domain_event_batches
             SET scientific_event_sequence = 'ffffffffffffffff'
             WHERE scientific_event_sequence = '0000000000000001'",
        ),
        (
            "domain_ordinal_corruption",
            "UPDATE host_domain_events
             SET event_ordinal = event_ordinal + 10000
             WHERE scientific_event_sequence = '0000000000000001'
               AND event_ordinal = 0",
        ),
        (
            "domain_digest_corruption",
            "UPDATE host_domain_event_batches
             SET archive_payload_digest =
                 '0000000000000000000000000000000000000000000000000000000000000000'
             WHERE scientific_event_sequence = '0000000000000001'",
        ),
    ];
    for (label, mutation) in domain_corruptions {
        let corrupted_path = unique_database_path(label);
        fs::copy(&path, &corrupted_path)
            .expect("copy finished domain database for corruption proof");
        let connection = Connection::open(&corrupted_path)
            .expect("open copied domain database for deliberate corruption");
        connection
            .execute("PRAGMA foreign_keys = OFF")
            .expect("isolate deliberate projection corruption from foreign-key enforcement");
        let changed = connection
            .execute(mutation)
            .expect("tamper normalized domain projection");
        assert!(changed > 0, "domain corruption fixture must change a row");
        connection
            .close()
            .expect("close deliberately corrupted domain database");
        let corruption = match StorageReader::open_finished_for_run(&corrupted_path, run_id) {
            Err(error) => error,
            Ok(reader) => {
                reader
                    .close()
                    .expect("close unexpectedly admitted corrupted domain reader");
                panic!("finished reader admitted {label}");
            }
        };
        assert!(
            matches!(corruption, StorageError::InvalidData { .. }),
            "unexpected {label} refusal: {corruption}"
        );
    }

    // The uniquely named database is intentionally retained; this test performs no file deletion.
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one public-boundary test keeps backpressure, exact-Arc retry, recovery, and orderly shutdown in one observable lifecycle"
)]
fn capacity_one_backpressure_retries_the_exact_retained_batch() {
    let mut pipeline = StoragePipeline::unattributed_memory().expect("memory storage pipeline");
    let session_id = HostSessionId::new(0x1003);
    let options = StorageJournalOptions {
        admission_capacity: 1,
        ..StorageJournalOptions::default()
    };
    let journal = pipeline
        .journal_port(session_id, options)
        .expect("capacity-one journal port");
    let mut core = HostCore::with_journal(
        session_id,
        compact_world(),
        host_options(4),
        Box::new(journal),
    )
    .expect("host backed by the bounded storage journal");
    let mut frontend = NullFrontend::new(core.local_port(), 0x2003);

    let first =
        submit_command_with_authority_before_owner_drive(&mut frontend, HostCommand::Step, None);
    let second =
        submit_command_with_authority_before_owner_drive(&mut frontend, HostCommand::Step, None);
    let blocked = frontend
        .drive_at(&mut core, ManualInstant::from_nanos(0))
        .expect("both steps finish science before the second journal admission blocks");
    assert_eq!(blocked.scientific_steps, 2);
    assert!(matches!(
        blocked.blocker,
        Some(HostBlocker::JournalFull { capacity: 1, .. })
    ));
    assert_eq!(core.world_tick(), Tick(2));

    let retained = core
        .pending_journal_batch()
        .expect("second exact batch remains retained");
    let immediate_retry = core
        .retry_retained_journal()
        .expect("retry preserves host invariants")
        .expect("retained batch exists");
    assert!(matches!(
        immediate_retry,
        JournalAdmission::Full { capacity: 1, .. }
    ));
    assert!(Arc::ptr_eq(
        &retained,
        &core
            .pending_journal_batch()
            .expect("the same allocation remains retained")
    ));

    let mut next_nanos = 1;
    let mut accepted = false;
    for _ in 0..WORKER_RETRY_LIMIT {
        frontend
            .drive_at(&mut core, ManualInstant::from_nanos(next_nanos))
            .expect("poll the first bounded receipt");
        next_nanos = next_nanos
            .checked_add(1)
            .expect("test manual clock does not overflow");
        let admission = core
            .retry_retained_journal()
            .expect("retry the exact retained allocation")
            .expect("batch remains retained until accepted");
        assert!(
            !matches!(admission, JournalAdmission::Closed { .. }),
            "bounded journal closed while retrying {:?}",
            admission.batch_id()
        );
        match admission {
            JournalAdmission::Accepted { .. } => {
                accepted = true;
                break;
            }
            JournalAdmission::Full { capacity, .. } => {
                assert_eq!(
                    capacity, 1,
                    "bounded journal reported unexpected capacity {capacity}"
                );
                assert!(Arc::ptr_eq(
                    &retained,
                    &core
                        .pending_journal_batch()
                        .expect("backpressure retains the exact allocation")
                ));
                thread::sleep(Duration::from_millis(1));
            }
            JournalAdmission::Closed { .. } => continue,
        }
    }
    assert!(
        accepted,
        "capacity-one journal never accepted the retained retry"
    );
    assert!(core.pending_journal_batch().is_none());

    drive_until_journal_state(
        &mut frontend,
        &mut core,
        first.command_id(),
        &JournalState::CommittedVolatile,
        &mut next_nanos,
    );
    drive_until_journal_state(
        &mut frontend,
        &mut core,
        second.command_id(),
        &JournalState::CommittedVolatile,
        &mut next_nanos,
    );

    let shutdown = submit_command_with_authority(
        &mut frontend,
        &mut core,
        HostCommand::Shutdown,
        None,
        &mut next_nanos,
    );
    drive_until_journal_state(
        &mut frontend,
        &mut core,
        shutdown.command_id(),
        &JournalState::CommittedVolatile,
        &mut next_nanos,
    );
    assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);
    pipeline.shutdown().expect("close bounded memory storage");
}
