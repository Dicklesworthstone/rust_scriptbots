//! Public-boundary coverage for the HostCore storage-journal adapter.

use fsqlite::Connection;
use scriptbots_core::{
    AgentData, BrainRunner, INPUT_SIZE, OUTPUT_SIZE, Position, ScriptBotsConfig, Tick, WorldState,
    channels::OutputChannel,
};
use scriptbots_runtime::{
    ApplicationState, CommandEnvelope, CommandId, CommandStatus, ControlRevision, EventCatchUp,
    EventCatchUpGuarantee, EventCatchUpState, EventCommitment, EventJournalReader, EventPageSource,
    EventPoll, EventSequence, HostBlocker, HostCommand, HostCore, HostCoreOptions, HostLifecycle,
    HostSessionId, JournalAdmission, JournalBatchId, JournalState, LocalHostPort, ManualInstant,
    NullFrontend, PlaybackSnapshot, RejectionReason,
};
use scriptbots_storage::{
    CommandJournalCursor, CommandStorageTransitionKind, DomainEventExpectation, DomainEventPayload,
    HostJournalPrefixes, HostJournalRecordState, HostJournalSessionPage, PersistenceGuarantee,
    StorageError, StorageEventJournalReader, StorageIntegrityCheckResult, StorageJournalOptions,
    StoragePipeline, StorageReader,
};
use std::{
    fs,
    sync::Arc,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const WORKER_RETRY_LIMIT: usize = 2_000;

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
        spike_alignment_cosine: 0.0,
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

fn drive_until_journal_state(
    frontend: &mut NullFrontend<LocalHostPort>,
    core: &mut HostCore,
    command_id: CommandId,
    expected: &JournalState,
    next_nanos: &mut u64,
) -> CommandStatus {
    let mut last_status = None;
    for _ in 0..WORKER_RETRY_LIMIT {
        frontend
            .drive_at(core, ManualInstant::from_nanos(*next_nanos))
            .expect("public frontend drives the matching host");
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
        "command {command_id:?} did not reach journal state {expected:?}"
    );
    status
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

    let first = frontend.step().expect("first explicit step");
    let first = drive_until_journal_state(
        &mut frontend,
        &mut core,
        first.command_id(),
        &JournalState::CommittedVolatile,
        &mut next_nanos,
    );
    assert_eq!(first.journal(), &JournalState::CommittedVolatile);

    let second = frontend.step().expect("second explicit step");
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

    let shutdown = frontend.shutdown().expect("ordered memory shutdown");
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

    let first_submission = frontend.step().expect("first durable step");
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

    let second_submission = frontend.step().expect("second durable step");
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

    let shutdown_submission = frontend.shutdown().expect("ordered durable shutdown");
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
    let pre_admission_rejection = frontend
        .set_speed(f32::from_bits(nan_bits))
        .expect("invalid nonfinite speed remains a queryable rejected command");
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

    let rejected_shutdown = frontend
        .submit(HostCommand::Shutdown, Some(ControlRevision::new(u64::MAX)))
        .expect("revision-conflicting shutdown enters ordered command audit");
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

    let pause = frontend.pause().expect("applied control-only pause");
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
    frontend
        .submit_envelope(update_envelope)
        .expect("applied command-only configuration replacement");
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
    frontend
        .submit_envelope(failed_config_envelope)
        .expect("valid live-geometry change is admitted before application failure");
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
    frontend
        .submit_envelope(step_envelope)
        .expect("applied scientific step");
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
    frontend
        .submit_envelope(shutdown_envelope)
        .expect("final applied shutdown");
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

    let first = frontend
        .step()
        .expect("first pre-cadence scientific boundary");
    drive_until_journal_state(
        &mut frontend,
        &mut core,
        first.command_id(),
        &JournalState::Durable,
        &mut next_nanos,
    );
    let second = frontend
        .step()
        .expect("second pre-cadence scientific boundary");
    drive_until_journal_state(
        &mut frontend,
        &mut core,
        second.command_id(),
        &JournalState::Durable,
        &mut next_nanos,
    );
    assert_eq!(core.world_tick(), Tick(2));

    let shutdown = frontend.shutdown().expect("ordered durable shutdown");
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

    let first = frontend.step().expect("first bounded step");
    let second = frontend.step().expect("second bounded step");
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

    let shutdown = frontend.shutdown().expect("bounded ordered shutdown");
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
