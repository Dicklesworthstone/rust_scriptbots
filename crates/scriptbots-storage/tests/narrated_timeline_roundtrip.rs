//! The NARRATED timeline survives the run database byte-identically (bd-ji3a).
//!
//! `persistence_integration.rs` already proves that hand-authored [`EventRecord`] values
//! recover byte-identically across a crash boundary. This file proves the thing that test
//! cannot: that the records the PRODUCTION BRIDGE actually emits -- built by running real
//! detectors over a real series and mapping the resulting evidence through
//! `event_record_from_evidence` -- survive the same trip.
//!
//! The distinction is the whole point. A fixture round-trips whatever a test author typed;
//! it says nothing about whether the prose a reader sees in the timeline is the prose the
//! database keeps. That is the property bd-ji3a is actually asking for, because the
//! timeline is templated precisely so two runs of a seed diff line by line.

use scriptbots_core::{
    PersistenceBatch, Tick, TickSummary,
    detect::{CusumParams, DetectionEvidence, Sample, change_points_cusum},
    narrative::{EventRecord, event_record_from_evidence},
};
use scriptbots_runtime::RunId;
use scriptbots_storage::{RunManifestRecord, StoragePipeline, StorageReader};
use std::time::{SystemTime, UNIX_EPOCH};

fn test_path(label: &str) -> std::path::PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "scriptbots_narrated_{label}_{}_{}.sqlite",
        std::process::id(),
        timestamp
    ))
}

/// A population that runs flat, crashes hard, then runs flat again.
///
/// Deliberately built to be detectable rather than realistic: this file is testing the
/// persistence of a narration, not the sensitivity of a detector. The detector's own
/// calibration is bd-16g.2's business and is tested there.
fn crashing_population(seed_offset: f64) -> Vec<Sample> {
    let mut samples = Vec::new();
    for tick in 0..150u64 {
        samples.push(Sample {
            tick,
            value: 1000.0 + seed_offset + f64::from((tick % 7) as u32),
        });
    }
    for tick in 150..300u64 {
        samples.push(Sample {
            tick,
            value: 300.0 + seed_offset + f64::from((tick % 5) as u32),
        });
    }
    samples
}

/// Run the real detector and bridge, exactly as production would.
fn bridged_records(seed_offset: f64) -> (Vec<DetectionEvidence>, Vec<EventRecord>) {
    let samples = crashing_population(seed_offset);
    let params = CusumParams::default();
    let changes = change_points_cusum(&samples, params).expect("well-formed series");
    let evidence: Vec<DetectionEvidence> = changes
        .iter()
        .map(|change| change.evidence("population", samples.len(), params))
        .collect();
    let records = evidence
        .iter()
        .filter_map(event_record_from_evidence)
        .collect();
    (evidence, records)
}

fn batch(events: Vec<EventRecord>) -> PersistenceBatch {
    let tick = events
        .iter()
        .map(|event| event.tick.0)
        .max()
        .expect("the detector produced at least one event");
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
        metrics: Vec::new(),
        events: Vec::new(),
        agents: Vec::new(),
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events: Vec::new(),
        narrative_events: events,
    }
}

/// Records built by the production bridge reload byte-identically, prose included.
#[test]
fn bridged_detector_evidence_round_trips_byte_identically() {
    for (index, seed_offset) in [0.0_f64, 17.0, 41.0].into_iter().enumerate() {
        let (evidence, records) = bridged_records(seed_offset);

        // A vacuous pass is the failure mode this guard exists for: persisting an empty
        // vector round-trips perfectly and proves nothing whatsoever.
        assert!(
            !records.is_empty(),
            "offset {seed_offset}: the detector found nothing to narrate, so this case \
             would have asserted on an empty set"
        );

        let path = test_path(&format!("bridge{index}"));
        let path_str = path.to_str().expect("utf8 path");
        let run_id = RunId::new(0x0BD1_0000_u128 + index as u128);
        let mut manifest = RunManifestRecord::unattributed(run_id);
        manifest.variant_id = Some(format!("bridged-{index}"));

        let mut pipeline = StoragePipeline::create_new_file_for_run_with_thresholds(
            path_str, manifest, 1, 1, 1, 1,
        )
        .expect("pipeline opens");
        pipeline
            .submit(&batch(records.clone()))
            .expect("narrative batch is admitted");
        pipeline.shutdown().expect("pipeline drains and closes");

        let reader = StorageReader::open(path_str).expect("typed reader opens");
        let reloaded: Vec<EventRecord> = reader
            .recent_run_events(256)
            .expect("narrative rows decode")
            .into_iter()
            .map(|event| event.into_parts().1)
            .collect();
        reader.close().expect("typed reader closes");

        assert_eq!(
            reloaded.len(),
            records.len(),
            "offset {seed_offset}: every bridged record must come back"
        );

        // Sort both sides identically: this asserts CONTENT survives, not read order.
        // Read ordering is `persistence_integration.rs`'s property, not this file's.
        let key = |record: &EventRecord| (record.tick.0, format!("{:?}", record.kind));
        let mut expected = records.clone();
        let mut actual = reloaded;
        expected.sort_by_key(key);
        actual.sort_by_key(key);

        assert_eq!(
            serde_json::to_vec(&actual).expect("reloaded serializes"),
            serde_json::to_vec(&expected).expect("expected serializes"),
            "offset {seed_offset}: bridged records did not reload byte-identically"
        );

        // THE PROPERTY THIS FILE EXISTS FOR: the prose in the database is the prose the
        // detector generated. A round trip that preserved every number but re-templated
        // the text would pass every assertion above and still destroy the run-to-run diff.
        let narrated: Vec<String> = evidence
            .iter()
            .filter(|item| event_record_from_evidence(item).is_some())
            .map(DetectionEvidence::narrate)
            .collect();
        let mut narrated_sorted = narrated;
        narrated_sorted.sort();
        let mut persisted_sorted: Vec<String> =
            actual.iter().map(|r| r.human_text.clone()).collect();
        persisted_sorted.sort();
        assert_eq!(
            persisted_sorted, narrated_sorted,
            "offset {seed_offset}: the persisted prose diverged from the narrated prose"
        );

        // Row count is also the collision check. RunEventIdentity is (tick, kind, metric),
        // so two detections sharing all three would silently collapse to one row on the
        // way in -- the `reloaded.len() == records.len()` assertion above is what catches
        // that, and it is the reason this test persists a real detector's output rather
        // than a handful of hand-spaced fixtures.

        let _ = std::fs::remove_file(&path);
    }
}
