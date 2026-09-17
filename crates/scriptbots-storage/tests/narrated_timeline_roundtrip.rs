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
    MetricSample, PersistenceBatch, Tick, TickSummary,
    detect::{
        BimodalityParams, CrossDirection, CusumParams, DetectionEvidence, DetectionKind, Direction,
        EvidenceClass, EvidenceSide, Regime, RegimeParams, Sample, Threshold, bimodality_with_work,
        change_points_cusum, regimes, threshold_crossings,
    },
    narrative::{EventKind, EventRecord, event_record_from_evidence},
};
use scriptbots_runtime::RunId;
use scriptbots_storage::{RunManifestRecord, StoragePipeline, StorageReader};
use serde_json::{Value, json};
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
        genomes: Vec::new(),
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

// This is a test-local consumer of the leaf detectors, not a production online feeder.
// Floats include their bits so the canonical envelope cannot hide signed zero or rounding.
fn exact_float(value: f64) -> Value {
    json!({ "value": value, "bits": value.to_bits() })
}

fn direction_json(direction: Direction) -> &'static str {
    match direction {
        Direction::Up => "up",
        Direction::Down => "down",
    }
}

fn evidence_json(evidence: &DetectionEvidence) -> Value {
    // No `..`: adding an envelope field must force this consumer to account for it.
    let DetectionEvidence {
        metric,
        kind,
        start_tick,
        end_tick,
        samples,
        score,
        class,
        before,
        after,
        params,
        finite,
    } = evidence;
    let kind = match kind {
        DetectionKind::ChangePoint => "change_point",
        DetectionKind::ThresholdCrossing => "threshold_crossing",
        DetectionKind::Regime => "regime",
        DetectionKind::Bimodality => "bimodality",
        DetectionKind::Speciation => "speciation",
        DetectionKind::Extinction => "extinction",
        DetectionKind::Radiation => "radiation",
    };
    let class = match *class {
        EvidenceClass::Shift(direction) => json!({ "shift": direction_json(direction) }),
        EvidenceClass::Crossing(direction) => json!({ "crossing": direction_json(direction) }),
        EvidenceClass::Regime(regime) => json!({ "regime": match regime {
            Regime::Growth => "growth",
            Regime::Equilibrium => "equilibrium",
            Regime::Oscillation => "oscillation",
            Regime::Collapse => "collapse",
        }}),
        EvidenceClass::Bimodal(value) => json!({ "bimodal": value }),
        EvidenceClass::Lineage(value) => json!({ "lineage": value }),
    };
    let side = |side: &Option<EvidenceSide>| {
        side.map(|EvidenceSide { samples, mean }| {
        json!({ "samples": samples, "mean": exact_float(mean) })
    })
    };
    json!({
        "metric": metric, "kind": kind, "start_tick": start_tick, "end_tick": end_tick,
        "samples": samples, "score": exact_float(*score), "class": class,
        "before": side(before), "after": side(after),
        "params": params.iter().map(|(name, value)| json!({
            "name": name, "value": exact_float(*value),
        })).collect::<Vec<_>>(),
        "finite": finite,
    })
}

fn sample_json(series: &[Sample]) -> Vec<Value> {
    series
        .iter()
        .map(|sample| {
            json!({
                "tick": sample.tick, "value": exact_float(sample.value),
            })
        })
        .collect()
}

fn first_evidence_divergence(expected: &[Value], actual: &[Value]) -> Option<Value> {
    (0..expected.len().max(actual.len())).find_map(|index| {
        let expected_bytes = expected
            .get(index)
            .map(|item| serde_json::to_vec(item).unwrap());
        let actual_bytes = actual
            .get(index)
            .map(|item| serde_json::to_vec(item).unwrap());
        (expected_bytes != actual_bytes).then(|| {
            json!({
                "index": index, "expected": expected.get(index), "actual": actual.get(index),
            })
        })
    })
}

fn compare_evidence(path: &str, expected: &[Value], actual: &[Value], equal: bool) {
    let expected_bytes = serde_json::to_vec(expected).expect("expected evidence serializes");
    let actual_bytes = serde_json::to_vec(actual).expect("actual evidence serializes");
    let divergence = first_evidence_divergence(expected, actual);
    println!(
        "{}",
        json!({
            "comparison": path,
            "expected_blake3": blake3::hash(&expected_bytes).to_hex().to_string(),
            "actual_blake3": blake3::hash(&actual_bytes).to_hex().to_string(),
            "first_divergence": divergence,
        })
    );
    assert_eq!(
        expected_bytes == actual_bytes,
        equal,
        "{path}: {divergence:?}"
    );
    assert_eq!(
        divergence.is_none(),
        equal,
        "{path}: divergence must include length changes"
    );
}

fn detector_window(path: &str, series: &[Sample]) -> (Vec<DetectionEvidence>, Vec<Value>) {
    let cusum = CusumParams::default();
    let thresholds = [Threshold {
        name: "population_floor",
        level: 600.0,
        direction: CrossDirection::Falling,
    }];
    let regime = RegimeParams {
        window: 50,
        ..RegimeParams::default()
    };
    let bimodal = BimodalityParams::default();
    let changes = change_points_cusum(series, cusum).expect("CUSUM input");
    let crossings = threshold_crossings(series, &thresholds).expect("crossing input");
    let windows = regimes(series, regime).expect("regime input");
    let values: Vec<_> = series.iter().map(|sample| sample.value).collect();
    let (split, work) = bimodality_with_work(&values, bimodal).expect("bimodality input");
    let mut evidence: Vec<_> = changes
        .iter()
        .map(|change| change.evidence("population", series.len(), cusum))
        .collect();
    evidence.extend(
        crossings
            .iter()
            .map(|crossing| crossing.evidence("population", series.len())),
    );
    evidence.extend(
        windows
            .iter()
            .map(|window| window.evidence("population", regime.window)),
    );
    evidence.push(split.evidence(
        "population",
        series[0].tick,
        series[series.len() - 1].tick,
        bimodal,
    ));
    let envelopes: Vec<_> = evidence.iter().map(evidence_json).collect();
    let bytes = serde_json::to_vec(&envelopes).expect("complete evidence serializes");
    println!(
        "{}",
        json!({
            "path": path, "metric": "population", "samples": sample_json(series),
            "input_window": [series[0].tick, series[series.len() - 1].tick],
            "sample_count": series.len(),
            "configuration": {
                "cusum": { "warmup": cusum.warmup, "k": exact_float(cusum.k),
                    "h": exact_float(cusum.h), "min_sigma": exact_float(cusum.min_sigma),
                    "max_detections": cusum.max_detections },
                "thresholds": thresholds.iter().map(|threshold| json!({
                    "name": threshold.name, "level": exact_float(threshold.level),
                    "direction": match threshold.direction {
                        CrossDirection::Falling => "falling", CrossDirection::Rising => "rising",
                        CrossDirection::Either => "either",
                    },
                })).collect::<Vec<_>>(),
                "regime": { "window": regime.window, "growth_slope": exact_float(regime.growth_slope),
                    "collapse_slope": exact_float(regime.collapse_slope),
                    "oscillation_autocorrelation": exact_float(regime.oscillation_autocorrelation),
                    "oscillation_spread": exact_float(regime.oscillation_spread),
                    "oscillation_min_crossings": regime.oscillation_min_crossings,
                    "min_scale": exact_float(regime.min_scale) },
                "bimodality": { "min_score": exact_float(bimodal.min_score),
                    "min_separation": exact_float(bimodal.min_separation),
                    "min_cluster_fraction": exact_float(bimodal.min_cluster_fraction),
                    "min_sigma": exact_float(bimodal.min_sigma) },
            },
            "work": { "cusum": null, "threshold_crossings": null, "regimes": null,
                "unavailable_reason": "public APIs expose no measured work for these three kernels",
                "bimodality": { "value_visits": work.value_visits, "bin_visits": work.bin_visits,
                    "heap_allocations": work.heap_allocations } },
            "evidence": envelopes, "blake3": blake3::hash(&bytes).to_hex().to_string(),
            "first_divergence": null,
        })
    );
    (evidence, envelopes)
}

/// Independent metric recovery, all-four complete evidence bytes, and the populated bridge.
#[test]
fn persisted_inputs_reproduce_all_detector_evidence_and_bridge() {
    let original = crashing_population(0.0);
    let path = test_path("detector_inputs");
    let path_str = path.to_str().expect("utf8 path");
    let manifest = RunManifestRecord::unattributed(RunId::new(0x0BD1_0211));
    let mut pipeline = StoragePipeline::create_new_file_for_run_with_thresholds(
        path_str, manifest, 64, 64, 64, 64,
    )
    .expect("metric pipeline opens");
    for sample in &original {
        let payload = PersistenceBatch {
            summary: TickSummary {
                tick: Tick(sample.tick),
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
            metrics: vec![MetricSample::new("population", sample.value)],
            events: Vec::new(),
            agents: Vec::new(),
            births: Vec::new(),
            deaths: Vec::new(),
            replay_events: Vec::new(),
            narrative_events: Vec::new(),
            genomes: Vec::new(),
        };
        pipeline.submit(&payload).expect("metric input admitted");
    }
    pipeline
        .shutdown()
        .expect("all metric inputs durable and writer closed");
    let reader = StorageReader::open(path_str).expect("independent metric reader opens");
    let rows = reader
        .recent_metrics(original.len() + 1)
        .expect("all metric rows recover");
    reader.close().expect("metric reader closes");
    assert_eq!(
        rows.len(),
        original.len(),
        "missing or extra persisted metric row"
    );
    for (index, (expected, actual)) in original.iter().zip(&rows).enumerate() {
        assert_eq!(actual.name, "population", "metric identity at row {index}");
        assert_eq!(
            (actual.tick, actual.value.to_bits()),
            (expected.tick, expected.value.to_bits()),
            "first differing persisted input at row {index}"
        );
    }
    let recovered: Vec<_> = rows
        .into_iter()
        .map(|row| Sample::new(row.tick, row.value))
        .collect();

    // Fixed complete windows: chunking changes assembly, never the detector's input window.
    // In particular, a partial bimodality assessment is not compared to the full-set verdict.
    for end in [200, 300] {
        let (memory_evidence, expected) = detector_window("in_memory", &original[..end]);
        let (recovered_evidence, actual) = detector_window("recovered", &recovered[..end]);
        for kind in [
            DetectionKind::ChangePoint,
            DetectionKind::ThresholdCrossing,
            DetectionKind::Regime,
            DetectionKind::Bimodality,
        ] {
            assert!(
                recovered_evidence.iter().any(|item| item.kind == kind),
                "fixture window {end} must exercise {kind:?}"
            );
        }
        compare_evidence("persisted_input_recovery", &expected, &actual, true);
        for chunk_size in [1, 7, 64, 151] {
            let mut assembled = Vec::with_capacity(end);
            for chunk in recovered[..end].chunks(chunk_size) {
                assembled.extend_from_slice(chunk);
            }
            let label = format!("recovered_chunks_{chunk_size}_window_{end}");
            let (_, chunked) = detector_window(&label, &assembled);
            compare_evidence(&label, &expected, &chunked, true);
        }

        let expected_records: Vec<_> = memory_evidence
            .iter()
            .filter_map(event_record_from_evidence)
            .collect();
        let records: Vec<_> = recovered_evidence
            .iter()
            .filter_map(event_record_from_evidence)
            .collect();
        assert_eq!(
            records, expected_records,
            "recovered inputs must preserve every bridged field and prose"
        );
        let crash = records
            .iter()
            .find(|record| record.kind == EventKind::PopulationCrash)
            .expect("the planted population crash must reach the narrative consumer");
        assert_eq!(crash.metric, "population");
        assert_eq!(crash.tick, Tick(150));
        assert_eq!(crash.window, (150, 150));
        assert!(
            crash.before > 1000.0 && crash.after == 300.0 && crash.magnitude > 700.0,
            "bridge must describe the planted loss, not merely emit a populated record"
        );
        println!(
            "{}",
            json!({ "path": "recovered_bridge", "input_end": end, "records": records })
        );

        // A plausible lost row must change the complete envelope stream, not pass because
        // the surviving detections happen to retain their ticks or their narrative prose.
        let mut missing = recovered[..end].to_vec();
        missing.remove(150);
        let (_, missing_evidence) = detector_window("negative_missing_crash_row", &missing);
        compare_evidence(
            "negative_missing_crash_row",
            &expected,
            &missing_evidence,
            false,
        );
        assert!(
            missing_evidence
                .iter()
                .any(|item| item["kind"] == "threshold_crossing" && item["start_tick"] == 151),
            "missing transition row must move the crossing"
        );

        // Exercise the comparison with every envelope field omitted in turn, including
        // fields absent from EventRecord (samples, typed class, params and finite).
        for field in [
            "metric",
            "kind",
            "start_tick",
            "end_tick",
            "samples",
            "score",
            "class",
            "before",
            "after",
            "params",
            "finite",
        ] {
            let mut omitted = expected.clone();
            assert!(
                omitted[0]
                    .as_object_mut()
                    .expect("envelope object")
                    .remove(field)
                    .is_some()
            );
            compare_evidence(
                &format!("negative_omitted_{field}"),
                &expected,
                &omitted,
                false,
            );
        }
        let mut truncated = expected.clone();
        truncated.pop();
        compare_evidence("negative_truncated_evidence", &expected, &truncated, false);
    }
}
