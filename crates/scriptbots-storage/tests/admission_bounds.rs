//! Persistence admission must be bounded by BYTES, not merely by command count.
//!
//! A count-based bound bounds nothing that matters: a single batch carrying a
//! hundred thousand agent rows is ONE command, sails through a count gate, and is
//! fully materialized before any deadline or admission check can refuse it. The
//! memory is gone by the time anyone gets a say.

use scriptbots_core::{
    MetricSample, PersistenceBatch, PersistenceEvent, PersistenceEventKind, ReplayEvent,
    ReplayEventKind, Tick, TickSummary,
};
use scriptbots_storage::{
    PayloadBudget, StorageError, StoragePipeline, StorageReader, estimate_batch_size,
    estimate_narrative_size,
};
use std::borrow::Cow;
use std::time::{SystemTime, UNIX_EPOCH};

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
    });

    let mut loud = batch(1, 8);
    loud.narrative_events = (0..512).map(narrative_event).collect();
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
        .map(|event| (event.tick, event.human_text.clone()))
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
        kind: ReplayEventKind::BrainOutputs {
            outputs: Vec::new(),
        },
    });
    let (empty_output_bytes, _) = estimate_batch_size(&empty_outputs);

    let mut nested_outputs = batch(5, 0);
    nested_outputs.replay_events.push(ReplayEvent {
        agent_uid: None,
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
    });

    let oversized = batch(1, 64);
    let error = pipeline
        .submit(&oversized)
        .expect_err("a batch over the cap must be refused");

    // The caller must be able to TELL that this was a size refusal, and by how
    // much — a generic error would leave them unable to shrink the batch and retry.
    let text = error.to_string();
    assert!(
        text.contains("too large to admit"),
        "the refusal must name the reason; got: {text}"
    );

    // AND THE RETRY DATA IS INTACT: nothing was consumed, so the caller still
    // holds the exact payload. Submitting a batch that FITS must still work — the
    // refusal must not have poisoned the pipeline.
    pipeline
        .submit(&batch(2, 2))
        .expect("a batch inside the budget must still be admitted after a refusal");

    pipeline.shutdown().expect("shutdown");
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
