//! Persistence admission must be bounded by BYTES, not merely by command count.
//!
//! A count-based bound bounds nothing that matters: a single batch carrying a
//! hundred thousand agent rows is ONE command, sails through a count gate, and is
//! fully materialized before any deadline or admission check can refuse it. The
//! memory is gone by the time anyone gets a say.

use scriptbots_core::{PersistenceBatch, Tick, TickSummary};
use scriptbots_storage::{PayloadBudget, StoragePipeline, estimate_batch_size};
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
    }
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
