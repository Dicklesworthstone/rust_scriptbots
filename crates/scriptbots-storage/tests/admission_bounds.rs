//! Persistence admission must be bounded by BYTES, not merely by command count.
//!
//! A count-based bound bounds nothing that matters: a single batch carrying a
//! hundred thousand agent rows is ONE command, sails through a count gate, and is
//! fully materialized before any deadline or admission check can refuse it. The
//! memory is gone by the time anyone gets a say.

use scriptbots_core::{
    AgentData, AgentIdentity, AgentRuntime, AgentState, AgentUid, MetricSample, PersistenceBatch,
    PersistenceEvent, PersistenceEventKind, Position, ReplayEvent, ReplayEventKind,
    ReplayInteractionKind, Tick, TickSummary,
};
use scriptbots_storage::{
    AnalyticsSnapshotProvider, FailureCommitState, PayloadBudget, PreparationFaultPoint, Storage,
    StorageDeadlines, StorageError, StorageOperation, StoragePipeline, StorageReader,
    StorageWorkerError, arm_preparation_fault, cleanup_handoff_stats, clear_all_preparation_faults,
    clear_preparation_fault, drain_cleanup_handoffs_for_test, estimate_batch_size,
    estimate_narrative_size, handoff_cleanup,
};
use serde_json::json;
use std::borrow::Cow;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn temp_db(label: &str) -> String {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir()
        .join(format!(
            "scriptbots_admission_{label}_{}_{nonce}.sqlite",
            std::process::id()
        ))
        .to_str()
        .expect("utf8 path")
        .to_owned()
}

struct PreparationFaultGuard(PreparationFaultPoint);

impl Drop for PreparationFaultGuard {
    fn drop(&mut self) {
        clear_preparation_fault(self.0);
    }
}

fn scoped_preparation_fault(point: PreparationFaultPoint) -> PreparationFaultGuard {
    arm_preparation_fault(point);
    PreparationFaultGuard(point)
}

fn batch(tick: u64, metrics: usize) -> PersistenceBatch {
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
        metrics: (0..metrics)
            .map(|i| scriptbots_core::MetricSample {
                name: std::borrow::Cow::Borrowed("probe"),
                value: i as f64,
            })
            .collect(),
        events: Vec::new(),
        agents: Vec::new(),
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events: Vec::new(),
        narrative_events: Vec::new(),
        genomes: Vec::new(),
    }
}

/// One narrative event, sized so a burst of them is unmistakably large.
fn narrative_event(tick: u64) -> scriptbots_core::narrative::EventRecord {
    scriptbots_core::narrative::EventRecord {
        schema_version: 1,
        tick: Tick(tick),
        kind: scriptbots_core::narrative::EventKind::PopulationCrash,
        severity: 0.5,
        magnitude: 1.0,
        window: (tick.saturating_sub(1), tick),
        metric: "population".to_owned(),
        before: 100.0,
        after: 10.0,
        score: 1.0,
        subject: None,
        human_text: "x".repeat(4096),
    }
}

/// A burst of narration must not be able to refuse a batch of scientific rows (`bd-erff`).
///
/// Narrative events used to be folded into `estimate_batch_size`, so they counted against
/// `max_batch_events` and `max_batch_bytes` alongside metrics, agents, births, deaths and
/// replay rows. Exceeding either cap refuses the *whole* batch as a definite `NotAdmitted`,
/// which latches the world and blocks later science ticks — so derived commentary about the
/// simulation could stop the simulation being recorded. That inverts what the budget is for.
///
/// Narration is now estimated separately, against its own pool. This asserts the property
/// that separation buys: identical scientific content is admitted identically whether or not
/// a large amount of commentary rides along.
#[test]
fn a_narration_burst_cannot_refuse_a_batch_of_science() {
    let science_only = batch(1, 64);
    let mut with_narration = batch(1, 64);
    with_narration.narrative_events = (0..512).map(narrative_event).collect();

    let (quiet_bytes, quiet_events) = estimate_batch_size(&science_only);
    let (loud_bytes, loud_events) = estimate_batch_size(&with_narration);
    assert_eq!(
        (quiet_bytes, quiet_events),
        (loud_bytes, loud_events),
        "narration changed the scientific estimate, so it can still consume the budget \
         that protects science"
    );

    // And the commentary is still accounted for, in its own pool rather than nowhere:
    // separating the budgets must not become a way of charging nothing at all.
    let (narrative_bytes, narrative_events) = estimate_narrative_size(&with_narration);
    assert_eq!(narrative_events, 512);
    assert!(
        narrative_bytes > 512 * 4096,
        "the narrative estimate must cover the human text it carries, got {narrative_bytes}"
    );
    assert_eq!(
        estimate_narrative_size(&science_only),
        (0, 0),
        "a batch with no commentary must cost nothing in the narrative pool"
    );
}

/// The budgets must be independent *under pressure*, not merely in the estimator (`bd-erff`).
///
/// The test above proves narration does not change the scientific estimate. This proves the
/// consequence that actually matters: a batch whose commentary vastly exceeds the scientific
/// caps is still admitted, because it is the scientific content that those caps govern.
///
/// This is the failure the miscount was hiding. A refusal here is `PayloadTooLarge`, a
/// definite `NotAdmitted` that latches the world and blocks later science ticks — so before
/// the separation, a run generating a lot of commentary could stop recording simulation data
/// entirely, and the commentary that cost it was not even stored.
#[test]
fn a_narration_burst_cannot_starve_simulation_admission_through_the_real_path() {
    let path = temp_db("narration-pressure");
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");

    // Caps sized so the science below fits comfortably while the narration riding along is
    // orders of magnitude past them. If the two pools were still shared, this is precisely
    // the shape that would be refused.
    pipeline.set_payload_budget(PayloadBudget {
        max_batch_bytes: 64 << 10,
        max_batch_events: 128,
        max_inflight_bytes: 1 << 20,
        ..PayloadBudget::default()
    });

    let mut loud = batch(1, 8);
    // Exercise budget pressure with valid identities. Repeating one identity tests the
    // independent corruption guard, which must continue to refuse duplicate commentary.
    loud.narrative_events = (0..512)
        .map(|index| {
            let mut event = narrative_event(1);
            event.metric = format!("population.cohort.{index}");
            event
        })
        .collect();
    assert_eq!(
        loud.narrative_events
            .iter()
            .map(|event| (event.tick.0, event.kind, &event.metric))
            .collect::<std::collections::HashSet<_>>()
            .len(),
        loud.narrative_events.len(),
        "budget pressure must not be confounded with duplicate identities"
    );
    let (narrative_bytes, narrative_events) = estimate_narrative_size(&loud);
    assert!(
        narrative_events > 128 && narrative_bytes > (64 << 10),
        "the fixture must exceed both scientific caps to be a real pressure test, got \
         {narrative_events} events / {narrative_bytes} bytes"
    );

    pipeline
        .submit(&loud)
        .expect("commentary must not be able to refuse a batch of scientific rows");

    // And the pipeline is not left degraded: ordinary science still admits afterwards.
    pipeline
        .submit(&batch(2, 8))
        .expect("a later scientific batch must still be admitted");

    pipeline.shutdown().expect("shutdown");
    let _ = std::fs::remove_file(&path);
}

/// Narrative events must never be discarded without surfacing (`bd-erff`).
///
/// This pins the defect that mattered most in bd-erff, and it was not the miscount.
/// `StorageBuffer` declared `run_events` but never wired it into `append`, so rows built
/// per batch were dropped the instant batches merged into the flush buffer. Nothing failed:
/// admission succeeded, the flush committed, the run reported complete, and the narrative
/// was simply absent with no error, no counter and no log to say it had ever existed. A
/// silent drop is worse than a wrong count — a wrong count is visible in the arithmetic,
/// whereas this looked like a run that generated no commentary.
///
/// Deliberately exercises the *merge* path rather than a single batch: buffering several
/// batches before a flush is exactly what the broken `append` destroyed, and a one-batch
/// test would have passed against the bug.
#[test]
fn buffered_narrative_events_are_never_dropped_between_batches() {
    let path = temp_db("narrative-no-drop");
    // Thresholds high enough that batches accumulate in the buffer and must be merged,
    // rather than each one flushing on its own.
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(
        &path, 1_000, 1_000, 1_000, 1_000,
    )
    .expect("pipeline");

    let mut expected = Vec::new();
    for tick in 1..=6_u64 {
        let mut batch = batch(tick, 1);
        let event = narrative_event(tick);
        expected.push((tick, event.human_text.clone()));
        batch.narrative_events = vec![event];
        pipeline.submit(&batch).expect("batch admitted");
    }
    let shutdown = pipeline.shutdown().expect("shutdown");
    assert!(
        shutdown.committed_tick.is_some(),
        "expected a committed tick"
    );

    let reader = StorageReader::open(&path).expect("reader opens");
    let stored = reader.recent_run_events(64).expect("run events query");
    reader.close().expect("reader closes");

    let actual = stored
        .iter()
        .map(|event| (event.record().tick.0, event.record().human_text.clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "narrative events were lost between admission and the database — every submitted \
         event must be readable back, and a shortfall here means rows were discarded with \
         no error to reveal it"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_size_estimate_is_deterministic_monotonic_and_allocates_nothing() {
    // The estimate must be computable WITHOUT serializing the batch. Serializing
    // to find out whether something is too big to serialize is the bug, not the
    // check.
    let small = batch(1, 10);
    let large = batch(1, 1_000);

    let (small_bytes, small_events) = estimate_batch_size(&small);
    let (large_bytes, large_events) = estimate_batch_size(&large);

    assert_eq!(
        estimate_batch_size(&small),
        (small_bytes, small_events),
        "the estimate must be deterministic — a size that varies between calls \
         could admit a batch and then refuse the identical retry"
    );
    assert!(
        large_bytes > small_bytes && large_events > small_events,
        "the estimate must be MONOTONIC in the batch's size, or a bigger batch \
         could slip under a cap a smaller one hit"
    );
    assert_eq!(small_events, 10);
    assert_eq!(large_events, 1_000);
}

#[test]
fn derived_interaction_rows_are_charged_to_the_scientific_budget() {
    let mut ordinary = batch(1, 0);
    ordinary.replay_events.push(ReplayEvent {
        agent_uid: Some(AgentUid(7)),
        position: Some(Position::new(1.0, 2.0)),
        counterpart: None,
        counterpart_position: None,
        kind: ReplayEventKind::BrainOutputs {
            outputs: Vec::new(),
        },
    });

    let mut pairwise = batch(1, 0);
    pairwise.replay_events.push(ReplayEvent {
        agent_uid: Some(AgentUid(7)),
        position: Some(Position::new(1.0, 2.0)),
        counterpart: Some(AgentUid(11)),
        counterpart_position: Some(Position::new(3.0, 4.0)),
        kind: ReplayEventKind::Interaction {
            tick: Tick(1),
            ordinal: 0,
            kind: ReplayInteractionKind::Combat,
            magnitude: 0.25,
        },
    });

    let (ordinary_bytes, ordinary_events) = estimate_batch_size(&ordinary);
    let (pairwise_bytes, pairwise_events) = estimate_batch_size(&pairwise);
    assert_eq!(ordinary_events, 1);
    assert_eq!(
        pairwise_events, 2,
        "one typed interaction creates both a replay row and a derived interaction row"
    );
    assert!(
        pairwise_bytes > ordinary_bytes,
        "the derived SQL row must consume a conservative byte allowance"
    );

    let mut pipeline = StoragePipeline::unattributed_memory_with_thresholds(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    )
    .expect("pipeline");
    pipeline.set_payload_budget(PayloadBudget {
        max_batch_bytes: ordinary_bytes,
        max_batch_events: ordinary_events,
        max_inflight_bytes: usize::MAX,
        ..PayloadBudget::default()
    });
    pipeline
        .submit(&ordinary)
        .expect("the one-row baseline fits its exact budget");
    assert!(matches!(
        pipeline.submit(&pairwise),
        Err(StorageError::PayloadTooLarge { .. })
    ));
    pipeline.shutdown().expect("shutdown");
}

#[test]
fn five_k_agent_interaction_window_fits_and_flushes_under_default_budget() {
    const AGENT_COUNT: usize = 5_000;
    const WINDOW_EDGES: usize = scriptbots_core::DEFAULT_INTERACTION_EVENT_TICK_CAP;

    let mut five_k = batch(1, 0);
    five_k.summary.agent_count = AGENT_COUNT;
    five_k.agents = (0..AGENT_COUNT)
        .map(|index| {
            let ordinal = u64::try_from(index).expect("5k agent ordinal fits u64");
            AgentState {
                id: scriptbots_core::AgentId::default(),
                identity: AgentIdentity {
                    uid: AgentUid(ordinal + 1),
                    spawn_ordinal: ordinal,
                    birth_ordinal: None,
                },
                data: AgentData {
                    position: Position::new(
                        f32::from(u16::try_from(index % 1_000).expect("x fits u16")),
                        f32::from(u16::try_from(index / 1_000).expect("y fits u16")),
                    ),
                    ..AgentData::default()
                },
                runtime: AgentRuntime::default(),
            }
        })
        .collect();
    five_k.events.extend([
        PersistenceEvent::new(
            PersistenceEventKind::Custom(Cow::Borrowed(
                scriptbots_core::INTERACTION_EVENTS_OBSERVED_KIND,
            )),
            WINDOW_EDGES,
        ),
        PersistenceEvent::new(
            PersistenceEventKind::Custom(Cow::Borrowed(
                scriptbots_core::INTERACTION_EVENTS_PERSISTED_KIND,
            )),
            WINDOW_EDGES,
        ),
    ]);
    five_k.replay_events = (0..WINDOW_EDGES)
        .map(|ordinal| {
            let ordinal_u16 = u16::try_from(ordinal).expect("window ordinal fits u16");
            let ordinal_u64 = u64::from(ordinal_u16);
            ReplayEvent {
                agent_uid: Some(AgentUid(ordinal_u64 + 1)),
                position: Some(Position::new(f32::from(ordinal_u16), 1.0)),
                counterpart: Some(AgentUid(
                    ordinal_u64 + u64::try_from(WINDOW_EDGES).expect("window fits u64") + 1,
                )),
                counterpart_position: Some(Position::new(f32::from(ordinal_u16), 2.0)),
                kind: ReplayEventKind::Interaction {
                    tick: Tick(1),
                    ordinal: ordinal_u64,
                    kind: if ordinal.is_multiple_of(2) {
                        ReplayInteractionKind::Combat
                    } else {
                        ReplayInteractionKind::FoodShare
                    },
                    magnitude: 0.125,
                },
            }
        })
        .collect();
    five_k
        .replay_events
        .extend(
            (0..scriptbots_core::DEFAULT_REPLAY_EVENT_TICK_CAP).map(|ordinal| {
                let ordinal_u16 = u16::try_from(ordinal).expect("action ordinal fits u16");
                ReplayEvent {
                    agent_uid: Some(AgentUid(u64::from(ordinal_u16) + 1)),
                    position: Some(Position::new(f32::from(ordinal_u16), 3.0)),
                    counterpart: None,
                    counterpart_position: None,
                    kind: ReplayEventKind::Action {
                        tick: None,
                        left_wheel: 0.25,
                        right_wheel: -0.25,
                        boost: false,
                        spike_target: None,
                        sound_level: 0.0,
                        give_intent: 0.5,
                    },
                }
            }),
        );

    let default_budget = PayloadBudget::default();
    let (estimated_bytes, estimated_events) = estimate_batch_size(&five_k);
    assert!(
        estimated_bytes <= default_budget.max_batch_bytes,
        "the documented 5k/default-window shape exceeds the default byte budget: \
         {estimated_bytes} > {}",
        default_budget.max_batch_bytes
    );
    assert!(
        estimated_events <= default_budget.max_batch_events,
        "the documented 5k/default-window shape exceeds the default event budget: \
         {estimated_events} > {}",
        default_budget.max_batch_events
    );

    // The proof writes 5,000 complete agent snapshots under the unoptimized test profile.
    // Keep that evidence lane explicitly bounded without changing the 120-second production
    // default; the measured flush duration is printed below for the bead record.
    let proof_ack_deadline = Duration::from_secs(10 * 60);
    let mut pipeline = StoragePipeline::unattributed_memory_with_thresholds_and_deadlines(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        StorageDeadlines {
            flush_ack: proof_ack_deadline,
            shutdown_ack: proof_ack_deadline,
            ..StorageDeadlines::default()
        },
    )
    .expect("memory pipeline");
    pipeline.set_payload_budget(default_budget);
    pipeline
        .submit(&five_k)
        .expect("the 5k/default-window batch must be admitted");
    let flush_started = Instant::now();
    pipeline
        .flush_and_wait()
        .expect("the 5k/default-window batch must flush");
    let flush_elapsed = flush_started.elapsed();
    eprintln!(
        "bd-2z0.5.9 5k window: edges={WINDOW_EDGES} estimated_bytes={estimated_bytes} \
         estimated_events={estimated_events} flush_ms={}",
        flush_elapsed.as_millis()
    );
    pipeline.shutdown().expect("shutdown");
}

#[test]
fn long_dynamic_strings_and_nested_brain_outputs_cross_the_byte_cap() {
    let baseline = batch(1, 0);
    let (baseline_bytes, _) = estimate_batch_size(&baseline);

    let mut long_metric = batch(2, 0);
    long_metric
        .metrics
        .push(MetricSample::new("m".repeat(16_384), 1.0));
    let (long_metric_bytes, long_metric_events) = estimate_batch_size(&long_metric);
    assert!(
        long_metric_bytes > baseline_bytes.saturating_add(16_384),
        "the estimator must charge both prepared copies and the escaped outbox form of a metric name"
    );

    let mut long_custom_event = batch(3, 0);
    long_custom_event.events.push(PersistenceEvent::new(
        PersistenceEventKind::Custom(Cow::Owned("event".repeat(4_096))),
        1,
    ));
    let (long_event_bytes, long_event_events) = estimate_batch_size(&long_custom_event);
    assert!(
        long_event_bytes > baseline_bytes.saturating_add(16_384),
        "the estimator must charge dynamic custom-event strings"
    );

    let mut empty_outputs = batch(4, 0);
    empty_outputs.replay_events.push(ReplayEvent {
        agent_uid: None,
        position: None,
        counterpart: None,
        counterpart_position: None,
        kind: ReplayEventKind::BrainOutputs {
            outputs: Vec::new(),
        },
    });
    let (empty_output_bytes, _) = estimate_batch_size(&empty_outputs);

    let mut nested_outputs = batch(5, 0);
    nested_outputs.replay_events.push(ReplayEvent {
        agent_uid: None,
        position: None,
        counterpart: None,
        counterpart_position: None,
        kind: ReplayEventKind::BrainOutputs {
            outputs: vec![0.25; 4_096],
        },
    });
    let (nested_output_bytes, nested_output_events) = estimate_batch_size(&nested_outputs);
    assert!(
        nested_output_bytes > empty_output_bytes.saturating_add(4_096),
        "one replay row with a large nested output vector must not look like one fixed-size event"
    );

    let mut pipeline = StoragePipeline::unattributed_memory_with_thresholds(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    )
    .expect("pipeline");

    pipeline.set_payload_budget(PayloadBudget {
        max_batch_bytes: baseline_bytes.saturating_add(1_024),
        max_batch_events: long_metric_events.max(long_event_events),
        max_inflight_bytes: usize::MAX,
        ..PayloadBudget::default()
    });
    assert!(matches!(
        pipeline.submit(&long_metric),
        Err(StorageError::PayloadTooLarge { .. })
    ));
    assert!(matches!(
        pipeline.submit(&long_custom_event),
        Err(StorageError::PayloadTooLarge { .. })
    ));

    pipeline.set_payload_budget(PayloadBudget {
        max_batch_bytes: empty_output_bytes,
        max_batch_events: nested_output_events,
        max_inflight_bytes: usize::MAX,
        ..PayloadBudget::default()
    });
    assert!(matches!(
        pipeline.submit(&nested_outputs),
        Err(StorageError::PayloadTooLarge { .. })
    ));

    pipeline.shutdown().expect("shutdown");
}

#[test]
fn an_oversized_batch_is_refused_before_it_is_ever_allocated() {
    let path = temp_db("oversize");
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");

    // A budget small enough that a modest batch is already over it.
    pipeline.set_payload_budget(PayloadBudget {
        max_batch_bytes: 1_024,
        max_batch_events: 4,
        max_inflight_bytes: 1 << 20,
        ..PayloadBudget::default()
    });

    let oversized = batch(1, 64);
    let (oversized_bytes, oversized_events) = estimate_batch_size(&oversized);
    for attempt in 1..=2 {
        let error = pipeline
            .submit(&oversized)
            .expect_err("a batch over the cap must be refused");

        // The caller must be able to TELL that this was a size refusal, and by how
        // much — a generic error would leave them unable to shrink the batch and retry.
        assert!(
            matches!(
                &error,
                StorageError::PayloadTooLarge {
                    tick: 1,
                    bytes,
                    events,
                    max_bytes: 1_024,
                    max_events: 4,
                } if (*bytes, *events) == (oversized_bytes, oversized_events)
            ),
            "oversized attempt {attempt} returned an untyped or unstable refusal: {error}"
        );
        assert_eq!(
            pipeline.inflight_bytes(),
            0,
            "oversized attempt {attempt} acquired or leaked an in-flight permit"
        );
    }

    let reader = StorageReader::open(&path).expect("pre-admission reader");
    assert_eq!(
        reader
            .persistence_watermarks()
            .expect("pre-admission watermarks"),
        Default::default(),
        "an oversized retry must not mint a batch identity or advance an outbox watermark"
    );
    assert_eq!(
        reader
            .run_ledger_summary()
            .expect("pre-admission ledger")
            .tick_count,
        0,
        "an oversized retry must not mutate scientific storage"
    );
    reader.close().expect("pre-admission reader closes");

    // AND THE RETRY DATA IS INTACT: nothing was consumed, so the caller still
    // holds the exact payload. Submitting a batch that FITS must still work — the
    // refusal must not have poisoned the pipeline.
    let receipt = pipeline
        .submit_with_receipt(&batch(2, 2))
        .expect("a batch inside the budget must still be admitted after a refusal");
    assert_eq!(
        receipt.batch_id.get(),
        1,
        "refused oversized attempts must not consume persistence identities"
    );

    let final_oversized_error = pipeline
        .submit(&oversized)
        .expect_err("an identical oversized retry must remain refused after later traffic");
    assert!(
        matches!(
            &final_oversized_error,
            StorageError::PayloadTooLarge {
                tick: 1,
                bytes,
                events,
                max_bytes: 1_024,
                max_events: 4,
            } if (*bytes, *events) == (oversized_bytes, oversized_events)
        ),
        "the post-admission oversized retry changed disposition or estimate: \
         {final_oversized_error}"
    );

    pipeline.shutdown().expect("shutdown");
    let reader = StorageReader::open(&path).expect("post-shutdown reader");
    let final_watermarks = reader
        .persistence_watermarks()
        .expect("post-shutdown watermarks");
    assert_eq!(final_watermarks.admitted, Some(receipt.batch_id));
    assert_eq!(final_watermarks.applied, Some(receipt.batch_id));
    assert_eq!(final_watermarks.durable, Some(receipt.batch_id));
    let ledger = reader.run_ledger_summary().expect("post-shutdown ledger");
    assert_eq!(
        ledger.tick_count, 1,
        "only the fitting batch may reach durable storage"
    );
    assert_eq!(
        ledger.latest_tick.map(|tick| tick.tick),
        Some(2),
        "the only durable row must be the fitting tick"
    );
    reader.close().expect("post-shutdown reader closes");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_boundary_is_exact_at_the_cap_and_one_record_past_it() {
    let path = temp_db("boundary");
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");

    // Size the cap to EXACTLY what an 8-record batch estimates.
    let at_cap = batch(1, 8);
    let (bytes, events) = estimate_batch_size(&at_cap);
    pipeline.set_payload_budget(PayloadBudget {
        max_batch_bytes: bytes,
        max_batch_events: events,
        max_inflight_bytes: 1 << 20,
        ..PayloadBudget::default()
    });

    // Exactly at the cap: ADMITTED. An off-by-one here would refuse a batch the
    // operator explicitly sized to fit.
    pipeline
        .submit(&at_cap)
        .expect("a batch exactly at the cap must be admitted");

    // One record past it: REFUSED.
    let over = batch(2, 9);
    assert!(
        pipeline.submit(&over).is_err(),
        "one record past the cap must be refused"
    );

    pipeline.shutdown().expect("shutdown");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_in_flight_permit_is_released_on_every_path_including_the_refusal_path() -> Result<(), String>
{
    // THE LEAK TEST. "Released exactly once on commit, refusal, timeout handoff,
    // crash, and shutdown" is a requirement that a chain of hand-written
    // decrements WILL eventually violate: one early return that forgets it, and
    // the counter creeps up until the sink refuses everything and persistence dies
    // quietly in a long run rather than loudly here.
    //
    // If the permit leaked on ANY path, the in-flight total would ratchet upward
    // and this loop would start failing partway through. It does not.
    let path = temp_db("permit");
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");

    let small = batch(0, 2);
    let (small_bytes, _) = estimate_batch_size(&small);
    // An in-flight ceiling that admits only about two such batches at once. If the
    // permit leaked, the third submission would be refused — and the two hundredth
    // certainly would be.
    pipeline.set_payload_budget(PayloadBudget {
        max_batch_bytes: 1 << 20,
        max_batch_events: 1_000,
        max_inflight_bytes: small_bytes * 2,
        ..PayloadBudget::default()
    });

    for tick in 0..200u64 {
        pipeline.submit(&batch(tick, 2)).map_err(|error| {
            format!(
                "submission {tick} was refused: {error}. The in-flight byte permit \
                 has LEAKED — every batch reserved bytes it never gave back, so the \
                 counter ratcheted up until the sink refused everything. In a long \
                 run this is persistence dying silently."
            )
        })?;
    }

    // Interleave refusals: a refused batch must give its reservation back too,
    // otherwise a run that gets a few oversized batches slowly strangles itself.
    for tick in 200..260u64 {
        let _ = pipeline.submit(&batch(tick, 5_000)); // refused: over max_batch_events
        pipeline
            .submit(&batch(tick, 2))
            .expect("a refusal must not consume in-flight budget");
    }

    pipeline.shutdown().expect("shutdown");
    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn a_buffered_batch_holds_its_permit_until_flush_or_shutdown() {
    // Maximal row thresholds deterministically keep an admitted batch buffered:
    // the worker acknowledges its durable-outbox admission but cannot release the
    // byte reservation until an explicit finalization barrier.
    let mut pipeline = StoragePipeline::unattributed_memory_with_thresholds(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    )
    .expect("pipeline");
    let first = batch(1, 2);
    let (bytes, events) = estimate_batch_size(&first);
    let max_inflight = bytes.saturating_mul(2).saturating_sub(1);
    pipeline.set_payload_budget(PayloadBudget {
        max_batch_bytes: bytes,
        max_batch_events: events,
        max_inflight_bytes: max_inflight,
        ..PayloadBudget::default()
    });

    pipeline.submit(&first).expect("first admission");
    assert_eq!(
        pipeline.inflight_bytes(),
        bytes,
        "an admission acknowledgement must not release a still-buffered payload"
    );

    let error = pipeline
        .submit(&batch(2, 2))
        .expect_err("a second same-sized batch must exceed the buffered byte ceiling");
    assert!(matches!(
        error,
        StorageError::InFlightBytesExhausted {
            would_be,
            max_inflight: observed_max,
            ..
        } if would_be == bytes.saturating_mul(2) && observed_max == max_inflight
    ));
    assert_eq!(
        pipeline.inflight_bytes(),
        bytes,
        "a refused reservation must leave the first permit intact"
    );

    pipeline.flush_and_wait().expect("flush");
    assert_eq!(
        pipeline.inflight_bytes(),
        0,
        "successful flush and finalization must release the buffered permit"
    );

    pipeline.submit(&batch(3, 2)).expect("post-flush admission");
    assert_eq!(pipeline.inflight_bytes(), bytes);
    pipeline.shutdown().expect("shutdown");
    assert_eq!(
        pipeline.inflight_bytes(),
        0,
        "shutdown finalization must release the final buffered permit"
    );
}

#[test]
fn tiny_scientific_with_oversized_narrative_refused_without_scientific_leak() {
    clear_all_preparation_faults();
    let mut pipeline = StoragePipeline::unattributed_memory_with_thresholds(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    )
    .expect("pipeline");

    // Sized with large scientific budget, but small narrative budget.
    pipeline.set_payload_budget(PayloadBudget {
        max_batch_bytes: 1 << 20,
        max_batch_events: 1_000,
        max_inflight_bytes: 1 << 20,
        max_narrative_batch_bytes: 8_192,
        max_narrative_batch_events: 2,
        max_narrative_inflight_bytes: 16_384,
    });

    let mut loud = batch(1, 4);
    loud.narrative_events = (0..5).map(narrative_event).collect();
    let (_loud_bytes, _loud_events) = estimate_batch_size(&loud);
    let (loud_narrative_bytes, loud_narrative_events) = estimate_narrative_size(&loud);
    assert!(loud_narrative_bytes > 8_192);
    assert!(loud_narrative_events > 2);

    let error = pipeline
        .submit(&loud)
        .expect_err("oversized narrative must be refused before allocation");
    assert!(
        matches!(
            error,
            StorageError::NarrativePayloadTooLarge {
                bytes,
                events,
                max_bytes,
                max_events,
                ..
            } if bytes == loud_narrative_bytes
                && max_bytes == 8_192
                && events == loud_narrative_events
                && max_events == 2
        ),
        "expected NarrativePayloadTooLarge, got: {error:?}"
    );

    // Assert zero permit leak:
    assert_eq!(
        pipeline.inflight_bytes(),
        0,
        "scientific permit leaked on narrative refusal"
    );
    assert_eq!(
        pipeline.inflight_narrative_bytes(),
        0,
        "narrative permit leaked on refusal"
    );

    // Exact retry contract: the rejected payload was not modified
    assert_eq!(loud.summary.tick.0, 1);
    assert_eq!(loud.metrics.len(), 4);
    assert_eq!(loud.narrative_events.len(), 5);

    // A valid scientific batch with narrative within cap admits cleanly
    let mut modest = batch(2, 4);
    let mut modest_ev = narrative_event(2);
    modest_ev.human_text = "x".repeat(100);
    modest.narrative_events = vec![modest_ev];
    let (modest_bytes, _) = estimate_batch_size(&modest);
    let (modest_narrative_bytes, _) = estimate_narrative_size(&modest);
    pipeline
        .submit(&modest)
        .expect("modest narrative batch must be admitted");

    assert_eq!(pipeline.inflight_bytes(), modest_bytes);
    assert_eq!(pipeline.inflight_narrative_bytes(), modest_narrative_bytes);

    pipeline.flush_and_wait().expect("flush");
    assert_eq!(pipeline.inflight_bytes(), 0);
    assert_eq!(pipeline.inflight_narrative_bytes(), 0);

    pipeline.shutdown().expect("shutdown");
    assert_eq!(pipeline.inflight_bytes(), 0);
    assert_eq!(pipeline.inflight_narrative_bytes(), 0);
}

#[test]
fn narrative_inflight_saturation_backpressure_and_permit_release() {
    clear_all_preparation_faults();
    let mut pipeline = StoragePipeline::unattributed_memory_with_thresholds(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    )
    .expect("pipeline");

    let first = {
        let mut b = batch(1, 2);
        b.narrative_events = vec![narrative_event(1)];
        b
    };
    let (first_bytes, _) = estimate_batch_size(&first);
    let (narrative_bytes, _) = estimate_narrative_size(&first);

    // Sized so exactly one narrative event fits in-flight.
    let max_narrative_inflight = narrative_bytes.saturating_mul(2).saturating_sub(1);
    pipeline.set_payload_budget(PayloadBudget {
        max_batch_bytes: 1 << 20,
        max_batch_events: 1_000,
        max_inflight_bytes: 1 << 20,
        max_narrative_batch_bytes: narrative_bytes.saturating_mul(2),
        max_narrative_batch_events: 10,
        max_narrative_inflight_bytes: max_narrative_inflight,
    });

    pipeline.submit(&first).expect("first batch admission");
    assert_eq!(pipeline.inflight_bytes(), first_bytes);
    assert_eq!(pipeline.inflight_narrative_bytes(), narrative_bytes);

    let second = {
        let mut b = batch(2, 2);
        b.narrative_events = vec![narrative_event(2)];
        b
    };
    let error = pipeline
        .submit(&second)
        .expect_err("second narrative batch must exceed in-flight narrative ceiling");

    assert!(
        matches!(
            error,
            StorageError::InFlightNarrativeBytesExhausted {
                tick: 2,
                would_be,
                max_inflight,
            } if would_be == narrative_bytes.saturating_mul(2) && max_inflight == max_narrative_inflight
        ),
        "expected InFlightNarrativeBytesExhausted, got: {error:?}"
    );

    // Refusal of second batch did not disturb first batch's permits nor leak anything:
    assert_eq!(pipeline.inflight_bytes(), first_bytes);
    assert_eq!(pipeline.inflight_narrative_bytes(), narrative_bytes);

    // Flush releases narrative permits:
    pipeline.flush_and_wait().expect("flush");
    assert_eq!(pipeline.inflight_bytes(), 0);
    assert_eq!(pipeline.inflight_narrative_bytes(), 0);

    // After release, exact retry of the second batch succeeds:
    pipeline
        .submit(&second)
        .expect("retry after flush succeeds");
    assert_eq!(pipeline.inflight_bytes(), first_bytes);
    assert_eq!(pipeline.inflight_narrative_bytes(), narrative_bytes);

    pipeline.shutdown().expect("shutdown");
    assert_eq!(pipeline.inflight_bytes(), 0);
    assert_eq!(pipeline.inflight_narrative_bytes(), 0);
}

#[test]
fn fallible_reservation_failure_returns_typed_not_admitted_without_panic() {
    clear_all_preparation_faults();
    let mut pipeline = StoragePipeline::unattributed_memory_with_thresholds(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    )
    .expect("pipeline");

    let test_batch = batch(10, 8);
    let fault = scoped_preparation_fault(PreparationFaultPoint::ForceReservationFailure);

    let error = pipeline
        .submit(&test_batch)
        .expect_err("reservation failure must return typed error without panic");

    assert!(
        matches!(
            error,
            StorageError::ReservationFailed { context, .. } if context.starts_with("storage.")
        ),
        "expected ReservationFailed error, got: {error:?}"
    );

    // In-flight permits were not leaked:
    assert_eq!(pipeline.inflight_bytes(), 0);
    assert_eq!(pipeline.inflight_narrative_bytes(), 0);

    drop(fault);

    // Retry of the exact same batch succeeds cleanly:
    pipeline
        .submit(&test_batch)
        .expect("retry after clearing fault must succeed");

    pipeline.shutdown().expect("shutdown");
}

#[test]
fn caller_return_and_cleanup_handoff_are_bounded_independently() {
    clear_all_preparation_faults();
    drain_cleanup_handoffs_for_test();

    let stats_before = cleanup_handoff_stats();

    // Create a large batch to simulate deallocation overhead
    let mut large_batch = batch(1, 100);
    large_batch.narrative_events = (0..50).map(narrative_event).collect();

    let started = Instant::now();
    handoff_cleanup(large_batch);
    let elapsed = started.elapsed();

    // Caller must return immediately without waiting for synchronous drop
    assert!(
        elapsed < Duration::from_millis(100),
        "handoff_cleanup took too long: {elapsed:?}"
    );

    let stats_mid = cleanup_handoff_stats();
    assert!(
        stats_mid.handed_off_count > stats_before.handed_off_count,
        "handed_off_count did not increment"
    );

    drain_cleanup_handoffs_for_test();
    let stats_after = cleanup_handoff_stats();
    assert_eq!(stats_after.active_count, 0);
    assert!(stats_after.completed_count >= stats_mid.handed_off_count);
}

#[test]
fn bounded_error_publication_under_forced_contention() {
    clear_all_preparation_faults();
    let analytics = AnalyticsSnapshotProvider::empty();
    let worker_error = StorageWorkerError::Internal {
        operation: StorageOperation::Admit,
        path: ":memory:".to_string(),
        tick: Some(42),
        commit_state: FailureCommitState::NotAdmitted,
        detail: "simulated error".to_string(),
    };

    let _fault = scoped_preparation_fault(PreparationFaultPoint::ForceErrorPublicationContention);

    let started = Instant::now();
    // Bounded publication with 8 max attempts must terminate immediately
    analytics.publish_worker_error_bounded(&worker_error, false, 8);
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_millis(50),
        "publish_worker_error_bounded under contention took too long: {elapsed:?}"
    );
}

#[test]
fn direct_same_thread_persistence_boundary_policy_and_fallible_reservation() {
    clear_all_preparation_faults();
    let mut storage = Storage::unattributed_memory().expect("in-memory storage");

    let valid_batch = batch(1, 4);

    // 1. Same-thread API persists directly without background threads or timeouts:
    storage
        .persist(&valid_batch)
        .expect("direct same-thread persistence must succeed");
    assert_eq!(
        storage
            .persistence_watermarks()
            .unwrap()
            .admitted
            .map(|b| b.get()),
        Some(1)
    );
    storage.flush().expect("flush");
    assert_eq!(
        storage
            .persistence_watermarks()
            .unwrap()
            .applied
            .map(|b| b.get()),
        Some(1)
    );

    // 2. Same-thread API respects fallible container reservations:
    {
        let _fault = scoped_preparation_fault(PreparationFaultPoint::ForceReservationFailure);
        let fail_batch = batch(2, 4);
        let error = storage
            .persist(&fail_batch)
            .expect_err("forced reservation failure must be caught and returned");
        assert!(
            matches!(error, StorageError::ReservationFailed { .. }),
            "expected ReservationFailed, got: {error:?}"
        );
    }

    // 3. Exact retry after reservation failure succeeds:
    let fail_batch = batch(2, 4);
    storage
        .persist(&fail_batch)
        .expect("exact retry after cleared reservation fault must succeed");
    assert_eq!(
        storage
            .persistence_watermarks()
            .unwrap()
            .admitted
            .map(|b| b.get()),
        Some(2)
    );
    storage.flush().expect("flush");
    assert_eq!(
        storage
            .persistence_watermarks()
            .unwrap()
            .applied
            .map(|b| b.get()),
        Some(2)
    );

    storage.close().expect("close");
}

#[test]
fn narrative_preparation_bounds_and_timeout_cleanup_e2e() -> Result<(), Box<dyn std::error::Error>>
{
    clear_all_preparation_faults();
    drain_cleanup_handoffs_for_test();

    let path_string = temp_db("storage-narrative-preparation-e2e");
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(
        &path_string,
        64,
        4_096,
        1_024,
        1_024,
    )?;

    // Phase 1: Oversized Narrative Refusal
    // Configure finite narrative budget (8 KiB narrative cap, while science allows 1 MiB)
    pipeline.set_payload_budget(PayloadBudget {
        max_batch_bytes: 1 << 20,
        max_batch_events: 1_000,
        max_inflight_bytes: 1 << 20,
        max_narrative_batch_bytes: 8_192,
        max_narrative_batch_events: 2,
        max_narrative_inflight_bytes: 16_384,
    });

    let mut oversized_narrative_batch = batch(101, 4);
    let mut ev1 = narrative_event(101);
    ev1.human_text = "x".repeat(5_000);
    let mut ev2 = narrative_event(101);
    ev2.metric = "population.b".to_string();
    ev2.human_text = "x".repeat(5_000);
    let mut ev3 = narrative_event(101);
    ev3.metric = "population.c".to_string();
    ev3.human_text = "x".repeat(5_000);
    oversized_narrative_batch.narrative_events = vec![ev1, ev2, ev3];

    let (scientific_bytes, scientific_records) = estimate_batch_size(&oversized_narrative_batch);
    let (narrative_bytes, narrative_records) = estimate_narrative_size(&oversized_narrative_batch);
    assert!(narrative_bytes > 8_192);

    let refusal = pipeline
        .submit(&oversized_narrative_batch)
        .expect_err("oversized narrative must be refused");
    assert!(matches!(
        refusal,
        StorageError::NarrativePayloadTooLarge { .. }
    ));
    assert_eq!(pipeline.inflight_bytes(), 0);
    assert_eq!(pipeline.inflight_narrative_bytes(), 0);

    println!(
        "{}",
        json!({
            "schema": "scriptbots.narrative-preparation.evidence.v1",
            "phase": "oversized_narrative_refusal",
            "stage": "narrative_measure",
            "disposition": "not_admitted",
            "error": "NarrativePayloadTooLarge",
            "narrative_bytes": narrative_bytes,
            "narrative_records": narrative_records,
            "scientific_bytes": scientific_bytes,
            "scientific_records": scientific_records,
            "inflight_bytes_before": 0,
            "inflight_bytes_after": 0,
            "inflight_narrative_bytes_before": 0,
            "inflight_narrative_bytes_after": 0,
            "tick": 101,
            "durable_tick_count": 0
        })
    );

    // Phase 2: Fallible Reservation Refusal
    {
        let _fault = scoped_preparation_fault(PreparationFaultPoint::ForceReservationFailure);
        let mut fail_batch = batch(102, 4);
        let mut fail_ev = narrative_event(102);
        fail_ev.human_text = "x".repeat(100);
        fail_batch.narrative_events = vec![fail_ev];
        let res_error = pipeline
            .submit(&fail_batch)
            .expect_err("reservation failure must return typed error without panic");
        assert!(matches!(res_error, StorageError::ReservationFailed { .. }));
        assert_eq!(pipeline.inflight_bytes(), 0);
        assert_eq!(pipeline.inflight_narrative_bytes(), 0);

        println!(
            "{}",
            json!({
                "schema": "scriptbots.narrative-preparation.evidence.v1",
                "phase": "fallible_reservation_refusal",
                "stage": "prepare_buffer",
                "disposition": "not_admitted",
                "error": "ReservationFailed",
                "tick": 102,
                "durable_tick_count": 0
            })
        );
    }

    // Phase 3: Cleanup Handoff
    let stats_before = cleanup_handoff_stats();
    let large_payload = Box::new(batch(103, 100));
    let started = Instant::now();
    handoff_cleanup(large_payload);
    let handoff_elapsed = started.elapsed();
    assert!(handoff_elapsed < Duration::from_millis(100));
    let stats_mid = cleanup_handoff_stats();
    assert!(stats_mid.handed_off_count > stats_before.handed_off_count);
    drain_cleanup_handoffs_for_test();
    let stats_after = cleanup_handoff_stats();

    println!(
        "{}",
        json!({
            "schema": "scriptbots.narrative-preparation.evidence.v1",
            "phase": "cleanup_handoff",
            "stage": "background_worker",
            "disposition": "async_cleanup",
            "handed_off_count": stats_after.handed_off_count,
            "completed_count": stats_after.completed_count,
            "active_count": stats_after.active_count,
            "tick": 103
        })
    );

    // Phase 4: Healthy Admission and Exact Retry
    let mut healthy_batch = batch(104, 4);
    let mut healthy_ev = narrative_event(104);
    healthy_ev.human_text = "x".repeat(100);
    healthy_batch.narrative_events = vec![healthy_ev];
    let (h_sci_bytes, _) = estimate_batch_size(&healthy_batch);
    let (h_narr_bytes, _) = estimate_narrative_size(&healthy_batch);

    let receipt = pipeline.submit_with_receipt(&healthy_batch)?;
    let duplicate = pipeline.submit_with_receipt(&healthy_batch)?;
    assert_eq!(receipt.batch_id, duplicate.batch_id);
    assert_eq!(pipeline.inflight_bytes(), h_sci_bytes);
    assert_eq!(pipeline.inflight_narrative_bytes(), h_narr_bytes);

    let flush = pipeline.flush_and_wait()?;
    assert_eq!(flush.watermarks.durable, Some(receipt.batch_id));
    assert_eq!(pipeline.inflight_bytes(), 0);
    assert_eq!(pipeline.inflight_narrative_bytes(), 0);

    pipeline.shutdown()?;

    let reader = StorageReader::open(&path_string)?;
    let ledger = reader.run_ledger_summary()?;
    assert_eq!(ledger.tick_count, 1);
    assert_eq!(ledger.latest_tick.map(|t| t.tick), Some(104));
    reader.close()?;

    println!(
        "{}",
        json!({
            "schema": "scriptbots.narrative-preparation.evidence.v1",
            "phase": "healthy_admission_and_retry",
            "stage": "outbox_admission",
            "disposition": "admitted",
            "receipt": "durable",
            "batch_id": receipt.batch_id.get(),
            "tick": 104,
            "durable_tick_count": ledger.tick_count
        })
    );

    let _ = std::fs::remove_file(&path_string);
    Ok(())
}
