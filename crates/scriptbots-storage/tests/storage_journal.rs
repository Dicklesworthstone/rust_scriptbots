//! Public-boundary coverage for the HostCore storage-journal adapter.

use scriptbots_core::{ScriptBotsConfig, Tick, WorldState};
use scriptbots_runtime::{
    CommandId, CommandStatus, EventCatchUp, EventCatchUpGuarantee, EventCatchUpState,
    EventCommitment, EventJournalReader, EventPageSource, EventPoll, EventSequence, HostBlocker,
    HostCore, HostCoreOptions, HostLifecycle, HostSessionId, JournalAdmission, JournalBatchId,
    JournalState, LocalHostPort, ManualInstant, NullFrontend, PlaybackSnapshot,
};
use scriptbots_storage::{
    PersistenceGuarantee, StorageEventJournalReader, StorageJournalOptions, StoragePipeline,
};
use std::{
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

    let first = frontend.step().expect("first durable step");
    drive_until_journal_state(
        &mut frontend,
        &mut core,
        first.command_id(),
        &JournalState::Durable,
        &mut next_nanos,
    );
    let second = frontend.step().expect("second durable step");
    drive_until_journal_state(
        &mut frontend,
        &mut core,
        second.command_id(),
        &JournalState::Durable,
        &mut next_nanos,
    );

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

    let shutdown = frontend.shutdown().expect("ordered durable shutdown");
    let shutdown = drive_until_journal_state(
        &mut frontend,
        &mut core,
        shutdown.command_id(),
        &JournalState::Durable,
        &mut next_nanos,
    );
    assert_eq!(shutdown.journal(), &JournalState::Durable);
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
    assert_eq!(
        reader.available_range(),
        Some(scriptbots_runtime::EventSequenceRange {
            first: EventSequence::new(1),
            last: EventSequence::new(2),
        })
    );
    assert!(
        reader.contains_event_identity(EventSequence::new(1), JournalBatchId::new(session_id, 1))
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
