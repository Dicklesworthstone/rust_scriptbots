//! Finished-database matched-seed endpoint analysis and refusal tests.
//!
//! Attributed fixtures carry real core-generated seed checkpoints and synthetic endpoint
//! outcomes with a planted shift/null. They test the report's pairing and statistics, not a
//! claim that those outcomes were measured in an independently executed ecological experiment.
//! Legacy time-series fixtures verify that additional ticks cannot manufacture replication.

use std::fs;
use std::path::Path;

use scriptbots_analytics::{
    AnalyticsError, MAX_COMPARISON_COHORT_BYTES, MAX_COMPARISON_METRICS, MAX_COMPARISON_PAIRS,
    RUN_COMPARISON_COHORT_SCHEMA_V1, RUN_COMPARISON_SCHEMA_V2, ReaderCtx, Registry, ReportOutput,
    ReportParams,
};
use scriptbots_core::narrative::NarrativeInputRecordV1;
use scriptbots_core::{
    MetricSample, PersistenceBatch, ReplayEvent, ReplayEventKind, ScriptBotsConfig, Tick,
    TickSummary, WorldState,
};
use scriptbots_runtime::RunId;
use scriptbots_storage::{RunManifestRecord, Storage, StorageReader};

const fn batch(tick: u64, metrics: Vec<MetricSample>) -> PersistenceBatch {
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
        epoch: 1,
        closed: false,
        metrics,
        events: Vec::new(),
        agents: Vec::new(),
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events: Vec::new(),
        narrative_events: Vec::new(),
        genomes: Vec::new(),
    }
}

/// Small deterministic pseudo-noise so no metric's paired differences are perfectly constant (a
/// constant difference gives an infinite standardized effect size and a degenerate test).
fn wobble(tick: u64, salt: u64) -> f64 {
    let bucket = u32::try_from(tick.wrapping_mul(7).wrapping_add(salt) % 11)
        .expect("modulo 11 always fits in u32");
    (f64::from(bucket) - 5.0) * 0.1
}

fn run_db(
    dir: &tempfile::TempDir,
    file: &str,
    ticks: &[u64],
    shifting_base: f64,
    salt: u64,
) -> String {
    let path = dir.path().join(file).display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create run db");
    for &tick in ticks {
        let metrics = vec![
            MetricSample::new("shifting", shifting_base + wobble(tick, salt)),
            MetricSample::new("flat", 7.0 + wobble(tick, salt + 100)),
        ];
        storage.persist(&batch(tick, metrics)).expect("persist");
    }
    storage.flush().expect("flush");
    storage.close().expect("close");
    path
}

fn compare(control: &str, params: &[String]) -> Result<ReportOutput, AnalyticsError> {
    let cx = ReaderCtx::open(control)?;
    let params = ReportParams::from_pairs(params.iter().cloned())?;
    let result = Registry::builtin().run("compare-runs", &cx, &params);
    cx.reader.close()?;
    result
}

fn compare_selected(
    db: &str,
    run_id: RunId,
    params: &[String],
) -> Result<ReportOutput, AnalyticsError> {
    let cx = ReaderCtx::open_for_run(db, run_id)?;
    let params = ReportParams::from_pairs(params.iter().cloned())?;
    let result = Registry::builtin().run("compare-runs", &cx, &params);
    cx.reader.close()?;
    result
}

fn metric<'a>(machine: &'a serde_json::Value, name: &str) -> Option<&'a serde_json::Value> {
    machine["metrics"]
        .as_array()
        .expect("metrics array")
        .iter()
        .find(|m| m["metric"] == name)
}

#[expect(
    clippy::too_many_lines,
    reason = "complete attributed fixture includes storage-required seed, build and manifest projections"
)]
fn manifest(run_id: RunId, seed: u64, arm: &str, budget: u64) -> RunManifestRecord {
    // The real matched-seed runner assigns caller_seeded(run_id) to each run. Display IDs
    // necessarily differ within an arm even when its executable condition is identical.
    let scenario_id = format!("fixture-run-{run_id}");
    let config = ScriptBotsConfig {
        world_width: 40,
        world_height: 40,
        food_cell_size: 10,
        initial_food: 0.0,
        food_respawn_interval: 0,
        population_minimum: 0,
        population_spawn_interval: 0,
        rng_seed: Some(seed),
        ..ScriptBotsConfig::default()
    };
    let world = WorldState::new(config.clone()).expect("seeded manifest fixture");
    let mut normalized_config = serde_json::to_value(&config).expect("config");
    normalized_config.sort_all_objects();
    let normalized_config_json = serde_json::to_string(&normalized_config).expect("config JSON");
    let config_digest = format!(
        "blake3:{}",
        blake3::hash(normalized_config_json.as_bytes()).to_hex()
    );
    let (next_agent_uid, next_spawn_ordinal, next_birth_ordinal) = world.identity_sequence_state();
    let manifest_json = serde_json::json!({
        "schema": "scriptbots.run-manifest.v3.3",
        "schema_version": 3,
        "purpose": "characterization_only",
        "identity": {
            "run_id": run_id,
            "experiment_id": "synthetic-endpoint-study",
            "variant_id": arm,
            "started_at_unix_ms": 1,
            "requested_tick_budget": budget,
            "live_run_policy": null
        },
        "root_seed": seed,
        "thread_policy": {"threads": 4, "source": "test-fixture", "overridden": null},
        "random_streams": world.random_streams_checkpoint(),
        "agent_substream_protocol": world.agent_substream_protocol_v1(),
        "agent_rng_counters": world.ordered_agent_rng_counters_v1().expect("agent counters"),
        "next_agent_uid": next_agent_uid,
        "next_spawn_ordinal": next_spawn_ordinal,
        "next_birth_ordinal": next_birth_ordinal,
        "scenario": {
            "id": scenario_id,
            "schema_version": 0,
            "ordered_config_layer_digests": [],
            "population_recipe": "caller_seeded_world_v0",
            "bootstrap_ticks": 0
        },
        "normalized_config": normalized_config,
        "config_digest": config_digest,
        "config_digest_encoding": "blake3-canonical-json-v1",
        "brain_roster": [],
        "build": {
            "package_name": "scriptbots-analytics-fixture",
            "package_version": "0.1.0",
            "source_revision": "0123456789abcdef",
            "source_branch": "main",
            "source_tree_clean": true,
            "source_status_digest": "blake3:status-fixture",
            "source_diff_digest": "blake3:source-fixture",
            "declared_toolchain": "nightly-test-toolchain",
            "compiler_toolchain": null,
            "rustc_vv": "rustc test\nhost: aarch64-apple-darwin",
            "toolchain_file_digest": "blake3:toolchain-fixture",
            "lockfile_digest": "blake3:lock-fixture",
            "compiled_features": ["storage"],
            "core": {
                "parallel": true,
                "simd_wide": true,
                "rayon_threads": 4,
                "target_arch": "aarch64",
                "target_os": "macos",
                "target_family": "unix",
                "target_endian": "little",
                "pointer_width": 64
            },
            "rustflags": null,
            "rayon_num_threads": null,
            "scriptbots_max_threads": null,
            "provenance_complete": true,
            "warnings": []
        },
        "reproducible": true,
        "warnings": [],
        "limitations": {
            "purpose": "characterization_only",
            "agent_identity": "empty synthetic population",
            "source_identity": "synthetic schema fixture, not live experiment evidence",
            "evaluator_state_covered": false,
            "rng_state_restorable": true,
            "checkpoint_replay_guarantee": false,
            "comparison_lane": "same pinned test lane",
            "superseded_by": "WorldDigestV1"
        }
    })
    .to_string();
    RunManifestRecord {
        run_id,
        manifest_schema_version: 3,
        experiment_id: Some("synthetic-endpoint-study".to_owned()),
        variant_id: Some(arm.to_owned()),
        scenario_id,
        scenario_version: 0,
        normalized_config_json,
        config_digest,
        root_seed: seed,
        rng_algorithm: "scriptbots.rng-domains.v1".to_owned(),
        rng_version: 1,
        brain_roster_json: "[]".to_owned(),
        source_revision: Some("0123456789abcdef".to_owned()),
        source_tree_digest: Some("blake3:source-fixture".to_owned()),
        source_tree_dirty: Some(false),
        source_bundle_digest: None,
        rust_toolchain: "nightly-test-toolchain".to_owned(),
        cargo_lock_digest: "blake3:lock-fixture".to_owned(),
        target_triple: "aarch64-apple-darwin".to_owned(),
        started_at_unix_ms: 1,
        requested_tick_budget: Some(budget),
        live_run_policy: None,
        reproducible: true,
        features: vec!["storage".to_owned()],
        manifest_json,
    }
}

fn attributed_run(
    dir: &Path,
    file: &str,
    record: RunManifestRecord,
    observations: &[(u64, f64, f64)],
) -> String {
    let path = dir.join(file).display().to_string();
    let mut storage = Storage::create_new_file_for_run(&path, record).expect("attributed run");
    persist_endpoints(&mut storage, observations);
    storage.close().expect("close attributed run");
    path
}

fn append_attributed_run(path: &str, record: RunManifestRecord, observations: &[(u64, f64, f64)]) {
    let mut storage = Storage::append_run(path, record).expect("append attributed run");
    persist_endpoints(&mut storage, observations);
    storage.close().expect("close appended run");
}

fn shared_pair(dir: &Path) -> String {
    let path = attributed_run(
        dir,
        "shared-pair.sqlite",
        manifest(RunId::new(107), 7, "control", 2),
        &[(1, -1000.0, 7.0), (2, 10.0, 7.0)],
    );
    append_attributed_run(
        &path,
        manifest(RunId::new(207), 7, "treatment", 2),
        &[(1, 1000.0, 7.0), (2, 15.0, 7.0)],
    );
    path
}

fn persist_endpoints(storage: &mut Storage, observations: &[(u64, f64, f64)]) {
    let mut previous = 0;
    for &(tick, shifting, flat) in observations {
        let mut endpoint = batch(
            tick,
            vec![
                MetricSample::new("shifting", shifting),
                MetricSample::new("flat", flat),
            ],
        );
        // Attributed persistence requires the complete canonical per-tick narrative stream,
        // including ticks whose generic metrics are intentionally absent in endpoint tests.
        for input_tick in previous + 1..=tick {
            let input_summary = batch(input_tick, Vec::new()).summary;
            let record = NarrativeInputRecordV1::from_summary(
                &input_summary,
                0,
                &ScriptBotsConfig::default(),
            )
            .expect("canonical input");
            endpoint.replay_events.push(ReplayEvent {
                agent_uid: None,
                position: None,
                counterpart: None,
                counterpart_position: None,
                kind: ReplayEventKind::NarrativeInputV1 { record },
            });
        }
        storage.persist(&endpoint).expect("persist endpoint");
        previous = tick;
    }
    storage.flush().expect("flush endpoints");
}

fn cohort_value(endpoint: u64, control: &[String], treatment: &[String]) -> serde_json::Value {
    let entries = |paths: &[String]| {
        paths.iter().map(|path| {
        serde_json::json!({"db": Path::new(path).file_name().unwrap().to_str().unwrap()})
    }).collect::<Vec<_>>()
    };
    serde_json::json!({
        "schema": RUN_COMPARISON_COHORT_SCHEMA_V1,
        "endpoint_tick": endpoint,
        "metrics": ["shifting", "flat"],
        "control": entries(control),
        "treatment": entries(treatment)
    })
}

fn write_cohort(dir: &Path, file: &str, value: &serde_json::Value) -> String {
    let path = dir.join(file);
    fs::write(&path, serde_json::to_vec(value).unwrap()).expect("write cohort");
    path.display().to_string()
}

fn compare_cohort(control: &str, cohort: &str) -> Result<ReportOutput, AnalyticsError> {
    compare(control, &[format!("cohort_file={cohort}")])
}

fn assert_refused(control: &str, cohort: &str, reason: &str) {
    let error = compare_cohort(control, cohort).expect_err("invalid cohort must be refused");
    assert!(
        error.to_string().contains(reason),
        "unexpected refusal: {error}"
    );
}

/// Repack an established independent-seed oracle into one database with overlapping ticks.
/// Only each run's manifested endpoint is copied; earlier trajectory rows cannot affect inference.
fn assert_shared_database_cohort_matches(dir: &Path, machine: &serde_json::Value) {
    let file = "shared-cohort.sqlite";
    let shared = fs::canonicalize(dir)
        .unwrap()
        .join(file)
        .display()
        .to_string();
    let endpoint = machine["endpoint_tick"].as_u64().unwrap();
    let mut expected_pairs = machine["pairs"].clone();
    let mut cohort = serde_json::json!({
        "schema": RUN_COMPARISON_COHORT_SCHEMA_V1,
        "endpoint_tick": endpoint,
        "metrics": ["shifting", "flat"],
        "control": [],
        "treatment": [],
    });
    let pairs = expected_pairs.as_array_mut().unwrap();
    for (index, pair) in pairs.iter_mut().enumerate() {
        for arm in ["control", "treatment"] {
            let source = &mut pair[arm];
            let reader = StorageReader::open_finished(source["db"].as_str().unwrap()).unwrap();
            let record = reader.run_manifest().unwrap();
            reader.close().unwrap();
            let observations = [(
                endpoint,
                source["values"]["shifting"].as_f64().unwrap(),
                source["values"]["flat"].as_f64().unwrap(),
            )];
            if index == 0 && arm == "control" {
                attributed_run(dir, file, record, &observations);
            } else {
                append_attributed_run(&shared, record, &observations);
            }
            cohort[arm].as_array_mut().unwrap().push(serde_json::json!({
                "db": file,
                "run_id": source["run_id"],
            }));
            source["db"] = serde_json::json!(shared);
        }
    }
    cohort["control"].as_array_mut().unwrap().reverse();
    let path = write_cohort(dir, "shared-cohort.json", &cohort);
    let selected: RunId = pairs[0]["control"]["run_id"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    let cx = ReaderCtx::open_for_run(&shared, selected).unwrap();
    let params = ReportParams::from_pairs([format!("cohort_file={path}")]).unwrap();
    let result = Registry::builtin()
        .run("compare-runs", &cx, &params)
        .unwrap();
    assert_eq!(result.machine["mode"], "matched_seed_cohort");
    assert_eq!(result.machine["inference_status"], "computed");
    assert_eq!(result.machine["independent_pairs"], 12);
    assert_eq!(result.machine["metrics"], machine["metrics"]);
    assert_eq!(result.machine["pairs"], expected_pairs);
    assert_eq!(cx.reader.run_id(), selected);
    assert_eq!(cx.reader.run_manifest().unwrap().run_id, selected);
    assert_eq!(
        cx.reader.metrics_at_tick(endpoint, 2).unwrap()[1].value,
        25.0
    );
    cx.reader.close().unwrap();
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "end-to-end test constructs paired control and treatment arms over twelve seeds and asserts both the planted effect and the preserved null"
)]
fn independent_seed_endpoints_detect_the_planted_effect_and_preserve_the_null() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut control = Vec::new();
    let mut treatment = Vec::new();
    for seed in 1..=12u32 {
        let baseline = 25.0 * f64::from(seed);
        let null = 7.0 + f64::from(seed);
        let mut control_manifest = manifest(
            RunId::new(100 + u128::from(seed)),
            u64::from(seed),
            "control",
            2,
        );
        if seed == 1 {
            // Object insertion order and an explicit empty intervention schedule must not
            // make a different scientific condition from the other runs in this arm.
            let mut value: serde_json::Value =
                serde_json::from_str(&control_manifest.manifest_json).unwrap();
            value["scenario"]["interventions"] = serde_json::json!([]);
            value.sort_all_objects();
            control_manifest.manifest_json = value.to_string();
        }
        control.push(attributed_run(
            dir.path(),
            &format!("control,seed-{seed}.sqlite"),
            control_manifest,
            &[(1, -1000.0, null), (2, baseline, null)],
        ));
        treatment.push(attributed_run(
            dir.path(),
            &format!("treatment,seed-{seed}.sqlite"),
            manifest(
                RunId::new(200 + u128::from(seed)),
                u64::from(seed),
                "treatment",
                2,
            ),
            &[
                (1, 1000.0, null),
                (2, baseline + 5.0 + wobble(u64::from(seed), 3), null),
            ],
        ));
    }
    // Array position is deliberately the wrong pairing; only the manifested root seeds agree.
    treatment.reverse();
    let value = cohort_value(2, &control, &treatment);
    let path = write_cohort(dir.path(), "study.json", &value);
    let report = compare_cohort(&control[0], &path).expect("matched-seed comparison");
    assert_eq!(report.schema_version, 2);
    let machine = report.machine;
    assert_eq!(machine["schema"], RUN_COMPARISON_SCHEMA_V2);
    assert_eq!(machine["mode"], "matched_seed_cohort");
    assert_eq!(machine["inference_status"], "computed");
    assert_eq!(machine["independent_pairs"], control.len());
    assert_eq!(machine["endpoint_tick"], 2);
    let shifting = metric(&machine, "shifting").expect("shifting must be compared");
    assert_eq!(
        shifting["n_pairs"],
        control.len(),
        "each independently seeded run contributes exactly one endpoint"
    );
    assert!(
        (shifting["mean_difference"].as_f64().unwrap() - 5.0).abs() < 1.0,
        "the treatment effect should be ~+5; got {}",
        shifting["mean_difference"]
    );
    assert!(
        shifting["significant_fdr"].as_bool().unwrap(),
        "the planted +5 effect must survive across-metric FDR (p={})",
        shifting["p_value"]
    );
    assert!(shifting["p_value"].as_f64().unwrap() < 0.05);
    assert!(
        shifting["ci_lower"].as_f64().unwrap() > 0.0,
        "the CI on the effect must exclude zero: [{}, {}]",
        shifting["ci_lower"],
        shifting["ci_upper"]
    );

    let flat = metric(&machine, "flat").expect("flat must be compared");
    assert!(
        !flat["significant_fdr"].as_bool().unwrap(),
        "the unchanged `flat` metric was certified as a treatment effect (p={}); false positive",
        flat["p_value"]
    );

    assert_eq!(
        machine["significant"], 1,
        "exactly one metric genuinely shifted"
    );
    assert_eq!(machine["pairs"][0]["seed_hex"], "0000000000000001");
    assert_eq!(
        machine["pairs"][0]["control"]["values"]["shifting"], 25.0,
        "source observations are retained, not just an inferential claim"
    );
    let reader = StorageReader::open_finished(&control[0]).unwrap();
    let durable_digest = reader.run_manifest().unwrap().manifest_digest().unwrap();
    reader.close().unwrap();
    assert_eq!(
        machine["pairs"][0]["control"]["manifest_digest"],
        durable_digest
    );

    let mut reordered = value;
    reordered["control"].as_array_mut().unwrap().reverse();
    reordered["treatment"].as_array_mut().unwrap().reverse();
    let reordered_path = write_cohort(dir.path(), "reordered.json", &reordered);
    let again = compare_cohort(&control[0], &reordered_path).expect("reordered cohort");
    assert_eq!(machine["metrics"], again.machine["metrics"]);
    assert_eq!(machine["pairs"], again.machine["pairs"]);
    assert_shared_database_cohort_matches(dir.path(), &machine);
}

#[test]
fn arbitrarily_many_ticks_in_one_pair_remain_descriptive() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ticks: Vec<u64> = (1..=200).collect();
    let control = run_db(&dir, "control.sqlite", &ticks, 10.0, 1);
    let treatment = run_db(&dir, "treatment.sqlite", &ticks, 15.0, 2);
    let output = compare(&control, &[format!("treatment_db={treatment}")]).unwrap();
    assert_eq!(output.schema_version, 2);
    assert!(
        output
            .human_md
            .contains("insufficient independent seed pairs")
    );
    let machine = output.machine;
    assert_eq!(machine["mode"], "descriptive_pair");
    assert_eq!(
        machine["inference_status"],
        "insufficient_independent_pairs"
    );
    assert_eq!(machine["seed_pairing"], "unknown");
    assert_eq!(machine["independent_pairs"], 0);
    assert_eq!(machine["declared_pairs"], 1);
    assert_eq!(machine["endpoint_tick"], 200);
    assert_eq!(machine["significant"], 0);
    let shifting = metric(&machine, "shifting").unwrap();
    assert_eq!(shifting["n_pairs"], 1);
    assert!(shifting["p_value"].is_null());
    assert!(shifting["ci_lower"].is_null());
    assert!(shifting["ci_upper"].is_null());
    assert!(!shifting["significant_fdr"].as_bool().unwrap());

    // The exact same endpoint with no earlier metric history must give the same results.
    let short_control = run_db(&dir, "short-control.sqlite", &[200], 10.0, 1);
    let short_treatment = run_db(&dir, "short-treatment.sqlite", &[200], 15.0, 2);
    let short = compare(&short_control, &[format!("treatment_db={short_treatment}")]).unwrap();
    assert_eq!(machine["metrics"], short.machine["metrics"]);
}

#[test]
fn an_explicit_earlier_endpoint_is_exact_and_a_single_seed_has_no_inference() {
    let dir = tempfile::tempdir().expect("tempdir");
    let control = attributed_run(
        dir.path(),
        "control.sqlite",
        manifest(RunId::new(101), 7, "control", 4),
        &[(1, 10.0, 7.0), (4, 90.0, 7.0)],
    );
    let treatment = attributed_run(
        dir.path(),
        "treatment.sqlite",
        manifest(RunId::new(201), 7, "treatment", 4),
        &[(1, 15.0, 7.0), (4, -90.0, 7.0)],
    );
    let value = cohort_value(
        1,
        std::slice::from_ref(&control),
        std::slice::from_ref(&treatment),
    );
    let path = write_cohort(dir.path(), "earlier.json", &value);
    let output = compare_cohort(&control, &path).unwrap();
    assert_eq!(output.machine["independent_pairs"], 1);
    assert_eq!(output.machine["seed_pairing"], "matched");
    assert_eq!(
        output.machine["inference_status"],
        "insufficient_independent_pairs"
    );
    assert_eq!(output.machine["pairs"][0]["control"]["terminal_tick"], 4);
    let shifting = metric(&output.machine, "shifting").unwrap();
    assert_eq!(shifting["mean_difference"], 5.0);
    assert!(shifting["p_value"].is_null());
    assert!(shifting["ci_lower"].is_null());

    let mut absent_tick = value.clone();
    absent_tick["endpoint_tick"] = 2.into();
    let path = write_cohort(dir.path(), "absent-tick.json", &absent_tick);
    assert_refused(&control, &path, "no exact value");
    let mut absent_metric = value;
    absent_metric["metrics"] = serde_json::json!(["shifting", "not-recorded"]);
    let path = write_cohort(dir.path(), "absent-metric.json", &absent_metric);
    assert_refused(&control, &path, "not-recorded");
}

#[test]
fn incomplete_budgets_and_unattributed_seed_claims_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let control = attributed_run(
        dir.path(),
        "control.sqlite",
        manifest(RunId::new(101), 7, "control", 4),
        &[(2, 10.0, 7.0), (4, 10.0, 7.0)],
    );
    let unfinished = attributed_run(
        dir.path(),
        "unfinished.sqlite",
        manifest(RunId::new(201), 7, "treatment", 4),
        &[(2, 15.0, 7.0), (3, 15.0, 7.0)],
    );
    let value = cohort_value(2, std::slice::from_ref(&control), &[unfinished]);
    let path = write_cohort(dir.path(), "unfinished.json", &value);
    assert_refused(&control, &path, "has not completed");

    let legacy = run_db(&dir, "unattributed.sqlite", &[2], 10.0, 1);
    let value = cohort_value(2, std::slice::from_ref(&legacy), &[control]);
    let path = write_cohort(dir.path(), "unattributed.json", &value);
    assert_refused(&legacy, &path, "lacks attributable seed evidence");
}

#[test]
fn duplicate_unmatched_reused_and_unrelated_sources_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let c7 = attributed_run(
        dir.path(),
        "c7.sqlite",
        manifest(RunId::new(107), 7, "control", 2),
        &[(2, 10.0, 7.0)],
    );
    let c8 = attributed_run(
        dir.path(),
        "c8.sqlite",
        manifest(RunId::new(108), 8, "control", 2),
        &[(2, 11.0, 7.0)],
    );
    let c7_duplicate = attributed_run(
        dir.path(),
        "c7-duplicate.sqlite",
        manifest(RunId::new(117), 7, "control", 2),
        &[(2, 10.0, 7.0)],
    );
    let t7 = attributed_run(
        dir.path(),
        "t7.sqlite",
        manifest(RunId::new(207), 7, "treatment", 2),
        &[(2, 15.0, 7.0)],
    );
    let t8 = attributed_run(
        dir.path(),
        "t8.sqlite",
        manifest(RunId::new(208), 8, "treatment", 2),
        &[(2, 16.0, 7.0)],
    );
    let t9 = attributed_run(
        dir.path(),
        "t9.sqlite",
        manifest(RunId::new(209), 9, "treatment", 2),
        &[(2, 17.0, 7.0)],
    );
    let control = vec![c7.clone(), c8.clone()];
    for (name, controls, treatments, reason) in [
        (
            "duplicate",
            vec![c7.clone(), c7_duplicate.clone()],
            vec![t7.clone(), t8.clone()],
            "duplicate root seed",
        ),
        (
            "unmatched",
            control.clone(),
            vec![t7.clone(), t9],
            "unmatched root seed sets",
        ),
        (
            "reused",
            control.clone(),
            vec![c7.clone(), t8.clone()],
            "reused run identity",
        ),
    ] {
        let value = cohort_value(2, &controls, &treatments);
        let path = write_cohort(dir.path(), &format!("{name}.json"), &value);
        assert_refused(&c7, &path, reason);
    }
    let value = cohort_value(2, &control, &[t7, t8]);
    let path = write_cohort(dir.path(), "unrelated-context.json", &value);
    assert_refused(&c7_duplicate, &path, "must be a declared control");

    let descriptive = compare(&c7, &[format!("treatment_db={c8}")]).unwrap();
    assert_eq!(descriptive.machine["seed_pairing"], "mismatched");
    assert_eq!(descriptive.machine["independent_pairs"], 0);
    assert_eq!(descriptive.machine["significant"], 0);
}

#[test]
fn scheduled_interventions_are_part_of_the_executable_arm_condition() {
    let dir = tempfile::tempdir().unwrap();
    let c7 = attributed_run(
        dir.path(),
        "c7.sqlite",
        manifest(RunId::new(107), 7, "control", 2),
        &[(2, 10.0, 7.0)],
    );
    let mut changed = manifest(RunId::new(108), 8, "control", 2);
    let mut value: serde_json::Value = serde_json::from_str(&changed.manifest_json).unwrap();
    value["scenario"]["interventions"] = serde_json::json!([
        {"tick": 1, "set": {"food_respawn_interval": 3}}
    ]);
    changed.manifest_json = value.to_string();
    let c8 = attributed_run(dir.path(), "c8.sqlite", changed, &[(2, 11.0, 7.0)]);
    let t7 = attributed_run(
        dir.path(),
        "t7.sqlite",
        manifest(RunId::new(207), 7, "treatment", 2),
        &[(2, 15.0, 7.0)],
    );
    let t8 = attributed_run(
        dir.path(),
        "t8.sqlite",
        manifest(RunId::new(208), 8, "treatment", 2),
        &[(2, 16.0, 7.0)],
    );
    let value = cohort_value(2, &[c7.clone(), c8], &[t7, t8]);
    let path = write_cohort(dir.path(), "mixed-condition.json", &value);
    assert_refused(&c7, &path, "control arm mixes conditions");
}

#[test]
fn descriptive_pairs_disclose_conditional_metrics_without_claiming_replication() {
    let dir = tempfile::tempdir().unwrap();
    let mut paths = Vec::new();
    for (file, value, extra) in [
        ("control.sqlite", 10.0, "only_control"),
        ("treatment.sqlite", 15.0, "only_treatment"),
    ] {
        let path = dir.path().join(file).display().to_string();
        let mut storage = Storage::create_unattributed_file(&path).unwrap();
        storage
            .persist(&batch(
                1,
                vec![
                    MetricSample::new("shifting", value),
                    MetricSample::new(extra, 1.0),
                ],
            ))
            .unwrap();
        storage.close().unwrap();
        paths.push(path);
    }
    let output = compare(&paths[0], &[format!("treatment_db={}", paths[1])]).unwrap();
    assert_eq!(
        output.machine["control_only_metrics"],
        serde_json::json!(["only_control"])
    );
    assert_eq!(
        output.machine["treatment_only_metrics"],
        serde_json::json!(["only_treatment"])
    );
    assert_eq!(output.machine["metrics_compared"], 1);
    assert_eq!(output.machine["significant"], 0);
    assert_eq!(
        metric(&output.machine, "shifting").unwrap()["mean_difference"],
        5.0
    );
    assert!(metric(&output.machine, "shifting").unwrap()["p_value"].is_null());
    assert!(output.human_md.contains("only_control"));
    assert!(output.human_md.contains("only_treatment"));
}

#[test]
fn cohort_schema_parameter_and_resource_bounds_fail_before_analysis() {
    let dir = tempfile::tempdir().unwrap();
    let control = run_db(&dir, "control.sqlite", &[1], 10.0, 1);
    let treatment = run_db(&dir, "treatment.sqlite", &[1], 15.0, 2);
    let base = cohort_value(
        1,
        std::slice::from_ref(&control),
        std::slice::from_ref(&treatment),
    );
    let cases = [
        (
            "schema",
            "schema",
            serde_json::json!("unknown-schema"),
            "schema must be",
        ),
        (
            "duplicate-metrics",
            "metrics",
            serde_json::json!(["shifting", "shifting"]),
            "metric names must be unique",
        ),
        (
            "too-many-metrics",
            "metrics",
            serde_json::json!(vec!["shifting"; MAX_COMPARISON_METRICS + 1]),
            "endpoint metrics",
        ),
        (
            "unknown-field",
            "seed",
            serde_json::json!(7),
            "unknown field",
        ),
    ];
    for (name, key, value, reason) in cases {
        let mut invalid = base.clone();
        invalid[key] = value;
        let path = write_cohort(dir.path(), &format!("{name}.json"), &invalid);
        assert_refused(&control, &path, reason);
    }
    let mut many = base.clone();
    many["control"] = serde_json::json!(vec![base["control"][0].clone(); MAX_COMPARISON_PAIRS + 1]);
    many["treatment"] =
        serde_json::json!(vec![base["treatment"][0].clone(); MAX_COMPARISON_PAIRS + 1]);
    let path = write_cohort(dir.path(), "too-many-runs.json", &many);
    assert_refused(&control, &path, "equal lengths");
    let oversized = dir.path().join("oversized.json");
    fs::write(&oversized, vec![b' '; MAX_COMPARISON_COHORT_BYTES + 1]).unwrap();
    assert_refused(&control, oversized.to_str().unwrap(), "no larger than");

    for params in [
        Vec::new(),
        vec![format!("treatment_db={treatment}"), "fdr=NaN".to_owned()],
        vec![format!("treatment_db={treatment}"), "fdr=1".to_owned()],
        vec![format!("treatment_db={treatment}"), "unknown=1".to_owned()],
        vec![
            format!("treatment_db={treatment}"),
            "cohort_file=unused.json".to_owned(),
        ],
    ] {
        assert!(matches!(
            compare(&control, &params),
            Err(AnalyticsError::BadParam { .. })
        ));
    }
}

#[test]
fn explicit_run_ids_select_the_correct_run_in_a_multi_run_input() {
    let dir = tempfile::tempdir().unwrap();
    let control = attributed_run(
        dir.path(),
        "control.sqlite",
        manifest(RunId::new(107), 7, "control", 2),
        &[(2, 10.0, 7.0)],
    );
    let treatment = attributed_run(
        dir.path(),
        "treatment.sqlite",
        manifest(RunId::new(207), 7, "treatment", 2),
        &[(2, 15.0, 7.0)],
    );
    let mut storage =
        Storage::append_run(&treatment, manifest(RunId::new(208), 8, "treatment", 2)).unwrap();
    persist_endpoints(&mut storage, &[(2, 1000.0, 7.0)]);
    storage.close().unwrap();
    let mut value = cohort_value(2, std::slice::from_ref(&control), &[treatment]);
    let path = write_cohort(dir.path(), "ambiguous.json", &value);
    assert_refused(&control, &path, "multiple runs");
    value["treatment"][0]["run_id"] = serde_json::to_value(RunId::new(207)).unwrap();
    let path = write_cohort(dir.path(), "selected.json", &value);
    let output = compare_cohort(&control, &path).unwrap();
    assert_eq!(
        output.machine["pairs"][0]["treatment"]["run_id"],
        RunId::new(207).to_string()
    );
    assert_eq!(
        metric(&output.machine, "shifting").unwrap()["mean_difference"],
        5.0
    );
}

#[test]
fn descriptive_pairs_can_select_two_runs_in_the_context_database() {
    let dir = tempfile::tempdir().unwrap();
    let db = shared_pair(dir.path());
    let control_id = RunId::new(107);
    let treatment_id = RunId::new(207);
    let params = vec![
        format!("treatment_db={db}"),
        format!("treatment_run_id={treatment_id}"),
    ];
    let output = compare_selected(&db, control_id, &params).unwrap();
    assert_eq!(output.machine["mode"], "descriptive_pair");
    assert_eq!(output.machine["seed_pairing"], "matched");
    assert_eq!(
        output.machine["inference_status"],
        "insufficient_independent_pairs"
    );
    assert_eq!(
        output.machine["pairs"][0]["control"]["run_id"],
        control_id.to_string()
    );
    assert_eq!(
        output.machine["pairs"][0]["treatment"]["run_id"],
        treatment_id.to_string()
    );
    let shifting = metric(&output.machine, "shifting").unwrap();
    assert_eq!(shifting["mean_difference"], 5.0);
    assert!(shifting["p_value"].is_null());
    assert!(shifting["ci_lower"].is_null());

    let mut earlier_params = params;
    earlier_params.push("endpoint_tick=1".to_owned());
    let earlier = compare_selected(&db, control_id, &earlier_params).unwrap();
    assert_eq!(earlier.machine["endpoint_tick"], 1);
    assert_eq!(
        metric(&earlier.machine, "shifting").unwrap()["mean_difference"],
        2000.0
    );

    for (selection, reason) in [
        (None, "multiple runs"),
        (Some(control_id), "distinct run identities"),
        (Some(RunId::new(999)), "does not exist"),
    ] {
        let mut params = vec![format!("treatment_db={db}")];
        if let Some(run_id) = selection {
            params.push(format!("treatment_run_id={run_id}"));
        }
        let error = compare_selected(&db, control_id, &params).unwrap_err();
        assert!(
            error.to_string().contains(reason),
            "unexpected refusal: {error}"
        );
    }
}

#[test]
fn cohort_selection_is_explicit_and_preserves_the_context_run_binding() {
    let dir = tempfile::tempdir().unwrap();
    let db = shared_pair(dir.path());
    let control_id = RunId::new(107);
    let treatment_id = RunId::new(207);
    let mut value = cohort_value(2, std::slice::from_ref(&db), std::slice::from_ref(&db));
    value["control"][0]["run_id"] = serde_json::to_value(control_id).unwrap();
    value["treatment"][0]["run_id"] = serde_json::to_value(treatment_id).unwrap();
    let path = write_cohort(dir.path(), "selected-shared.json", &value);
    let params = [format!("cohort_file={path}")];
    let output = compare_selected(&db, control_id, &params).unwrap();
    assert_eq!(
        metric(&output.machine, "shifting").unwrap()["mean_difference"],
        5.0
    );
    let error = compare_selected(&db, treatment_id, &params).unwrap_err();
    assert!(error.to_string().contains("must be a declared control"));

    for (file, arm, run_id, reason) in [
        ("ambiguous-shared.json", "control", None, "multiple runs"),
        (
            "absent-shared.json",
            "treatment",
            Some(RunId::new(999)),
            "does not exist",
        ),
        (
            "reused-shared.json",
            "treatment",
            Some(control_id),
            "reused run identity",
        ),
    ] {
        let mut invalid = value.clone();
        invalid[arm][0]["run_id"] = serde_json::to_value(run_id).unwrap();
        let path = write_cohort(dir.path(), file, &invalid);
        let error =
            compare_selected(&db, control_id, &[format!("cohort_file={path}")]).unwrap_err();
        assert!(
            error.to_string().contains(reason),
            "unexpected refusal: {error}"
        );
    }
}

#[cfg(unix)]
#[test]
fn same_database_aliases_preserve_finished_reader_path_refusals() {
    let dir = tempfile::tempdir().unwrap();
    let db = shared_pair(dir.path());
    let alias = dir.path().join("shared-alias.sqlite");
    std::os::unix::fs::symlink(&db, &alias).unwrap();
    let alias = alias.display().to_string();
    let control_id = RunId::new(107);
    let treatment_id = RunId::new(207);
    let error = compare_selected(
        &db,
        control_id,
        &[
            format!("treatment_db={alias}"),
            format!("treatment_run_id={treatment_id}"),
        ],
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("SYMLINK"),
        "unexpected refusal: {error}"
    );

    let mut value = cohort_value(2, std::slice::from_ref(&db), &[alias]);
    value["control"][0]["run_id"] = serde_json::to_value(control_id).unwrap();
    value["treatment"][0]["run_id"] = serde_json::to_value(treatment_id).unwrap();
    let path = write_cohort(dir.path(), "alias-shared.json", &value);
    let error = compare_selected(&db, control_id, &[format!("cohort_file={path}")]).unwrap_err();
    assert!(
        error.to_string().contains("SYMLINK"),
        "unexpected refusal: {error}"
    );
}

#[test]
fn the_report_is_registered() {
    assert!(
        Registry::builtin()
            .list()
            .iter()
            .any(|(name, _)| *name == "compare-runs")
    );
}
