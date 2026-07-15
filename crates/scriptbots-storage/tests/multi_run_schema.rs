//! End-to-end proof that one FrankenSQLite file can hold independent runs with overlapping keys.

use fsqlite::{Connection, compat::RowExt};
use scriptbots_core::{
    AgentData, AgentIdentity, AgentRuntime, AgentState, AgentUid, BirthOrigin, BirthRecord,
    Generation, MetricSample, PersistenceBatch, Position, Tick, TickSummary,
};
use scriptbots_runtime::RunId;
use scriptbots_storage::{
    BatchPersistenceState, RunManifestRecord, StoragePipeline, StorageReader,
};
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

fn temp_db_path(label: &str) -> String {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir()
        .join(format!(
            "scriptbots_multi_run_{label}_{}_{nonce}.sqlite",
            std::process::id()
        ))
        .to_str()
        .expect("temporary database path is UTF-8")
        .to_owned()
}

fn manifest(run_id: RunId, variant_id: &str, started_at_unix_ms: u64) -> RunManifestRecord {
    let normalized_config = serde_json::json!({
        "agent_count": 1,
        "rng_seed": u64::MAX,
        "tick_budget": 1
    });
    let normalized_config_json = serde_json::to_string(&normalized_config)
        .expect("fixture normalized configuration is serializable");
    let config_digest = format!(
        "blake3:{}",
        blake3::hash(normalized_config_json.as_bytes()).to_hex()
    );
    let brain_roster = serde_json::json!([{"registry_key": 7, "kind": "mlp"}]);
    let brain_roster_json =
        serde_json::to_string(&brain_roster).expect("fixture brain roster is serializable");
    let manifest_json = serde_json::json!({
        "schema": "scriptbots.run-manifest.v2",
        "schema_version": 2,
        "purpose": "characterization_only",
        "identity": {
            "run_id": run_id,
            "experiment_id": "overlapping-identities",
            "variant_id": variant_id,
            "started_at_unix_ms": started_at_unix_ms,
            "requested_tick_budget": 1,
            "live_run_policy": null
        },
        "root_seed": u64::MAX,
        "thread_policy": {
            "threads": 4,
            "source": "test-fixture",
            "overridden": null
        },
        "random_stream": {
            "algorithm": "test-small-rng",
            "version": 1,
            "codec_version": 1,
            "state": [1, 2, 3, 4]
        },
        "next_agent_uid": 2,
        "next_spawn_ordinal": 1,
        "next_birth_ordinal": 0,
        "scenario": {
            "id": "multi-run-schema-proof",
            "schema_version": 1,
            "ordered_config_layer_digests": [],
            "population_recipe": "single-fixture-agent-v1",
            "bootstrap_ticks": 0
        },
        "normalized_config": normalized_config,
        "config_digest": config_digest,
        "config_digest_encoding": "blake3-canonical-json-v1",
        "brain_roster": brain_roster,
        "build": {
            "package_name": "scriptbots-storage-fixture",
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
            "compiled_features": ["brain-mlp", "storage"],
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
            "agent_identity": "stable AgentUid fixture identity",
            "source_identity": "clean revision plus status and diff digests",
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
        manifest_schema_version: 2,
        experiment_id: Some("overlapping-identities".to_owned()),
        variant_id: Some(variant_id.to_owned()),
        scenario_id: "multi-run-schema-proof".to_owned(),
        scenario_version: 1,
        normalized_config_json,
        config_digest,
        root_seed: u64::MAX,
        rng_algorithm: "test-small-rng".to_owned(),
        rng_version: 1,
        brain_roster_json,
        source_revision: Some("0123456789abcdef".to_owned()),
        source_tree_digest: Some("blake3:source-fixture".to_owned()),
        source_tree_dirty: Some(false),
        source_bundle_digest: None,
        rust_toolchain: "nightly-test-toolchain".to_owned(),
        cargo_lock_digest: "blake3:lock-fixture".to_owned(),
        target_triple: "aarch64-apple-darwin".to_owned(),
        started_at_unix_ms,
        requested_tick_budget: Some(1),
        live_run_policy: None,
        reproducible: true,
        features: vec![
            "storage".to_owned(),
            "brain-mlp".to_owned(),
            "storage".to_owned(),
        ],
        manifest_json,
    }
}

fn mutate_manifest(record: &mut RunManifestRecord, mutate: impl FnOnce(&mut serde_json::Value)) {
    let mut value: serde_json::Value =
        serde_json::from_str(&record.manifest_json).expect("fixture manifest is valid JSON");
    mutate(&mut value);
    record.manifest_json = serde_json::to_string(&value).expect("mutated manifest is valid JSON");
}

fn manifest_validation_error(record: RunManifestRecord) -> String {
    StoragePipeline::memory_for_run(record)
        .map(|mut unexpected| {
            unexpected
                .shutdown()
                .expect("unexpected accepted fixture still shuts down");
        })
        .expect_err("storage unexpectedly accepted an invalid V2 manifest")
        .to_string()
}

fn overlapping_batch(epoch: u64, energy: f32) -> PersistenceBatch {
    let position = Position::new(12.0, 34.0);
    let agent = AgentState {
        id: scriptbots_core::AgentId::default(),
        identity: AgentIdentity {
            uid: AgentUid(1),
            spawn_ordinal: 0,
            birth_ordinal: None,
        },
        data: AgentData {
            position,
            generation: Generation(0),
            health: energy,
            ..AgentData::default()
        },
        runtime: AgentRuntime {
            energy,
            ..AgentRuntime::default()
        },
    };
    let founder = BirthRecord {
        tick: Tick(0),
        agent_uid: AgentUid(1),
        spawn_ordinal: 0,
        birth_ordinal: None,
        origin: BirthOrigin::Seeded,
        parent_a: None,
        parent_b: None,
        brain_kind: Some("mlp".to_owned()),
        brain_key: None,
        herbivore_tendency: 0.5,
        generation: Generation(0),
        position,
        is_hybrid: false,
    };

    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(0),
            agent_count: 1,
            births: 0,
            deaths: 0,
            total_energy: energy,
            average_energy: energy,
            average_health: energy,
            max_age: 0,
            spike_hits: 0,
        },
        epoch,
        closed: false,
        metrics: vec![MetricSample::from_f32("run_energy", energy)],
        events: Vec::new(),
        agents: vec![agent],
        births: vec![founder],
        deaths: Vec::new(),
        replay_events: Vec::new(),
    }
}

#[test]
fn two_runs_with_overlapping_scientific_and_operational_keys_remain_isolated()
-> Result<(), Box<dyn std::error::Error>> {
    let path = temp_db_path("overlapping_keys");
    let run_a = RunId::from_namespace_sequence(0xa11c_e001, 1);
    let run_b = RunId::from_namespace_sequence(0xb22d_e002, 1);

    let mut first = StoragePipeline::create_new_file_for_run(
        &path,
        manifest(run_a, "variant-a", 1_700_000_000_001),
    )?;
    let admission_a = first.submit_with_receipt(&overlapping_batch(11, 10.0))?;
    let shutdown_a = first.shutdown()?;

    let mut second =
        StoragePipeline::append_run(&path, manifest(run_b, "variant-b", 1_700_000_000_002))?;
    let admission_b = second.submit_with_receipt(&overlapping_batch(22, 20.0))?;
    let shutdown_b = second.shutdown()?;

    assert_eq!(admission_a.run_id, run_a);
    assert_eq!(admission_b.run_id, run_b);
    assert_eq!(admission_a.tick, 0);
    assert_eq!(admission_b.tick, 0);
    assert_eq!(admission_a.batch_id.get(), 1);
    assert_eq!(admission_b.batch_id.get(), 1);
    assert_eq!(shutdown_a.watermarks.durable, Some(admission_a.batch_id));
    assert_eq!(shutdown_b.watermarks.durable, Some(admission_b.batch_id));

    let ambiguous = match StorageReader::open(&path) {
        Err(error) => error,
        Ok(_unexpected) => {
            return Err("an unscoped reader must refuse a database containing two runs".into());
        }
    };
    assert!(
        ambiguous
            .to_string()
            .contains("database contains multiple runs; select one with open_for_run"),
        "unexpected ambiguity error: {ambiguous}"
    );
    let ambiguous_recovery = match StoragePipeline::recover_existing(&path) {
        Err(error) => error,
        Ok(mut unexpected) => {
            unexpected.shutdown()?;
            return Err("unscoped recovery must refuse a multi-run database".into());
        }
    };
    assert!(
        ambiguous_recovery
            .to_string()
            .contains("database contains multiple runs; select one with open_for_run"),
        "unexpected recovery ambiguity error: {ambiguous_recovery}"
    );

    let newest = StorageReader::catalog_page(&path, 0, 1)?;
    assert_eq!(newest.len(), 1);
    assert_eq!(newest[0].run_id, run_b);
    assert_eq!(newest[0].variant_id.as_deref(), Some("variant-b"));
    assert_eq!(newest[0].started_at_unix_ms, 1_700_000_000_002);
    let older = StorageReader::catalog_page(&path, 1, 1)?;
    assert_eq!(older.len(), 1);
    assert_eq!(older[0].run_id, run_a);
    let oversized_catalog = StorageReader::catalog_page(&path, 0, 4_097)
        .expect_err("run discovery must enforce the shared bounded-page ceiling");
    assert!(
        oversized_catalog
            .to_string()
            .contains("bounded maximum 4096")
    );

    for (run_id, admission, expected_epoch, expected_energy, expected_variant) in [
        (run_a, admission_a, 11, 10.0, "variant-a"),
        (run_b, admission_b, 22, 20.0, "variant-b"),
    ] {
        let reader = StorageReader::open_for_run(&path, run_id)?;
        assert_eq!(reader.run_id(), run_id);
        assert_eq!(reader.max_tick()?, Some(0));

        let ticks = reader.recent_ticks(8)?;
        assert_eq!(ticks.len(), 1);
        assert_eq!(ticks[0].tick, 0);
        assert_eq!(ticks[0].epoch, expected_epoch);
        assert_eq!(ticks[0].total_energy, f64::from(expected_energy));

        let metrics = reader.recent_metrics(8)?;
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].tick, 0);
        assert_eq!(metrics[0].name, "run_energy");
        assert_eq!(metrics[0].value, f64::from(expected_energy));

        let agents = reader.top_predators(8)?;
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].agent_uid, 1);
        assert_eq!(agents[0].avg_energy, f64::from(expected_energy));
        assert_eq!(agents[0].last_tick, 0);

        let ancestry = reader.load_ancestry_births()?;
        assert_eq!(ancestry.len(), 1);
        assert_eq!(ancestry[0].tick, Tick(0));
        assert_eq!(ancestry[0].agent_uid, AgentUid(1));
        assert_eq!(ancestry[0].spawn_ordinal, 0);
        assert_eq!(ancestry[0].origin, BirthOrigin::Seeded);

        let watermarks = reader.persistence_watermarks()?;
        assert_eq!(watermarks.admitted, Some(admission.batch_id));
        assert_eq!(watermarks.applied, Some(admission.batch_id));
        assert_eq!(watermarks.durable, Some(admission.batch_id));
        let status = reader
            .batch_status(admission.batch_id)?
            .expect("the admitted batch remains in the compact ledger");
        assert_eq!(status.run_id, run_id);
        assert_eq!(status.batch_id.get(), 1);
        assert_eq!(status.tick, 0);
        assert_eq!(status.state, BatchPersistenceState::Durable);

        let persisted_manifest = reader.run_manifest()?;
        assert_eq!(persisted_manifest.run_id, run_id);
        assert_eq!(
            persisted_manifest.variant_id.as_deref(),
            Some(expected_variant)
        );
        assert_eq!(persisted_manifest.root_seed, u64::MAX);
        assert_eq!(persisted_manifest.requested_tick_budget, Some(1));
        assert_eq!(persisted_manifest.live_run_policy, None);
        assert_eq!(
            persisted_manifest.features,
            vec!["brain-mlp".to_owned(), "storage".to_owned()]
        );
        let manifest_json: serde_json::Value =
            serde_json::from_str(&persisted_manifest.manifest_json)?;
        assert_eq!(manifest_json["identity"]["run_id"], run_id.to_string());
        assert_eq!(manifest_json["identity"]["variant_id"], expected_variant);
        assert_eq!(
            persisted_manifest.source_revision.as_deref(),
            Some("0123456789abcdef")
        );
        assert_eq!(
            persisted_manifest.source_tree_digest.as_deref(),
            Some("blake3:source-fixture")
        );
        assert_eq!(persisted_manifest.source_tree_dirty, Some(false));
        assert_eq!(persisted_manifest.rust_toolchain, "nightly-test-toolchain");
        assert_eq!(persisted_manifest.cargo_lock_digest, "blake3:lock-fixture");
        assert!(persisted_manifest.reproducible);
        reader.close()?;
    }

    let connection = Connection::open(&path)?;
    let command_id = "0000000000000000000000000000002a";
    for run_id in [run_a, run_b] {
        connection.execute_with_params(
            "INSERT INTO commands (
                 run_id, command_id, issued_at_tick, issued_ordinal, command_type,
                 source, payload_json, requested_at_utc
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            &[
                run_id.to_string().into(),
                command_id.into(),
                0_i64.into(),
                0_i64.into(),
                "set_speed".into(),
                "multi-run-test".into(),
                r#"{"speed":1}"#.into(),
                "2026-07-15T00:00:00Z".into(),
            ],
        )?;
        connection.execute_with_params(
            "INSERT INTO state_digests (
                 run_id, tick, digest_kind, digest, canonicalization_version
             ) VALUES (?1, ?2, ?3, ?4, ?5)",
            &[
                run_id.to_string().into(),
                0_i64.into(),
                "scientific-state".into(),
                "blake3:shared-digest".into(),
                1_i64.into(),
            ],
        )?;
    }

    for run_id in [run_a, run_b] {
        let command_count = connection
            .query_row_with_params(
                "SELECT COUNT(*) FROM commands
                 WHERE run_id = ?1 AND command_id = ?2
                   AND issued_at_tick = 0 AND issued_ordinal = 0",
                &[run_id.to_string().into(), command_id.into()],
            )?
            .get_typed::<i64>(0)?;
        assert_eq!(command_count, 1);

        let digest = connection
            .query_row_with_params(
                "SELECT digest FROM state_digests
                 WHERE run_id = ?1 AND tick = 0 AND digest_kind = 'scientific-state'",
                &[run_id.to_string().into()],
            )?
            .get_typed::<String>(0)?;
        assert_eq!(digest, "blake3:shared-digest");
    }
    let command_total = connection
        .query_row_with_params(
            "SELECT COUNT(*) FROM commands WHERE command_id = ?1",
            &[command_id.into()],
        )?
        .get_typed::<i64>(0)?;
    let digest_total = connection
        .query_row_with_params(
            "SELECT COUNT(*) FROM state_digests
             WHERE tick = 0 AND digest_kind = 'scientific-state'",
            &[],
        )?
        .get_typed::<i64>(0)?;
    assert_eq!(command_total, 2);
    assert_eq!(digest_total, 2);
    connection.close()?;

    Ok(())
}

#[test]
fn registration_and_recovery_reject_unverifiable_manifest_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    let contradictory_path = temp_db_path("contradictory_manifest");
    let run_id = RunId::from_namespace_sequence(0xc011_1de0, 1);
    let mut contradictory = manifest(run_id, "manifest-variant", 1_700_000_000_010);
    contradictory.variant_id = Some("projected-variant".to_owned());
    let contradiction =
        match StoragePipeline::create_new_file_for_run(&contradictory_path, contradictory) {
            Err(error) => error,
            Ok(mut unexpected) => {
                unexpected.shutdown()?;
                return Err(
                    "registration must reject a scalar projection that contradicts manifest JSON"
                        .into(),
                );
            }
        };
    assert!(
        contradiction.to_string().contains("/identity/variant_id"),
        "unexpected projection error: {contradiction}"
    );

    let tampered_path = temp_db_path("tampered_manifest_digest");
    let mut pipeline = StoragePipeline::create_new_file_for_run(
        &tampered_path,
        manifest(run_id, "digest-variant", 1_700_000_000_011),
    )?;
    pipeline.shutdown()?;
    let connection = Connection::open(&tampered_path)?;
    connection.execute_with_params(
        "UPDATE runs SET manifest_digest = ?1 WHERE run_id = ?2",
        &["blake3:tampered".into(), run_id.to_string().into()],
    )?;
    connection.close()?;

    let tampering = match StoragePipeline::recover_existing(&tampered_path) {
        Err(error) => error,
        Ok(mut unexpected) => {
            unexpected.shutdown()?;
            return Err("recovery must recompute every stored manifest digest".into());
        }
    };
    assert!(
        tampering.to_string().contains("runs.manifest_digest"),
        "unexpected digest-tampering error: {tampering}"
    );

    Ok(())
}

#[test]
fn v2_manifest_validation_rejects_missing_and_wrongly_typed_structure() {
    let run_id = RunId::from_namespace_sequence(0xc011_1de0, 2);

    let mut missing_purpose = manifest(run_id, "missing-purpose", 1_700_000_000_012);
    mutate_manifest(&mut missing_purpose, |value| {
        value
            .as_object_mut()
            .expect("manifest object")
            .remove("purpose");
    });
    let error = manifest_validation_error(missing_purpose);
    assert!(
        error.contains("/purpose is required"),
        "unexpected error: {error}"
    );

    let mut missing_codec = manifest(run_id, "missing-codec", 1_700_000_000_013);
    mutate_manifest(&mut missing_codec, |value| {
        value["random_stream"]
            .as_object_mut()
            .expect("random stream object")
            .remove("codec_version");
    });
    let error = manifest_validation_error(missing_codec);
    assert!(
        error.contains("/random_stream/codec_version is required"),
        "unexpected error: {error}"
    );

    let mut missing_target = manifest(run_id, "missing-target", 1_700_000_000_014);
    mutate_manifest(&mut missing_target, |value| {
        value["build"]["core"]
            .as_object_mut()
            .expect("core build object")
            .remove("target_family");
    });
    let error = manifest_validation_error(missing_target);
    assert!(
        error.contains("/build/core/target_family is required"),
        "unexpected error: {error}"
    );

    let mut wrong_limitation = manifest(run_id, "wrong-limitation", 1_700_000_000_015);
    mutate_manifest(&mut wrong_limitation, |value| {
        value["limitations"]["rng_state_restorable"] = serde_json::json!("yes");
    });
    let error = manifest_validation_error(wrong_limitation);
    assert!(
        error.contains("/limitations/rng_state_restorable must be a boolean"),
        "unexpected error: {error}"
    );
}

#[test]
fn v2_manifest_validation_recomputes_config_digest_and_binds_root_seed() {
    let run_id = RunId::from_namespace_sequence(0xc011_1de0, 3);

    let mut unknown_encoding = manifest(run_id, "unknown-encoding", 1_700_000_000_016);
    mutate_manifest(&mut unknown_encoding, |value| {
        value["config_digest_encoding"] = serde_json::json!("unversioned-json");
    });
    let error = manifest_validation_error(unknown_encoding);
    assert!(
        error.contains("/config_digest_encoding"),
        "unexpected error: {error}"
    );

    let mut forged_digest = manifest(run_id, "forged-digest", 1_700_000_000_017);
    forged_digest.config_digest = format!("blake3:{}", "0".repeat(64));
    let forged_digest_value = forged_digest.config_digest.clone();
    mutate_manifest(&mut forged_digest, |value| {
        value["config_digest"] = serde_json::json!(forged_digest_value);
    });
    let error = manifest_validation_error(forged_digest);
    assert!(
        error.contains("recomputed") && error.contains("blake3-canonical-json-v1"),
        "unexpected error: {error}"
    );

    let mut mismatched_seed = manifest(run_id, "seed-mismatch", 1_700_000_000_018);
    let changed_config = serde_json::json!({
        "agent_count": 1,
        "rng_seed": 7,
        "tick_budget": 1
    });
    mismatched_seed.normalized_config_json =
        serde_json::to_string(&changed_config).expect("changed config is serializable");
    mismatched_seed.config_digest = format!(
        "blake3:{}",
        blake3::hash(mismatched_seed.normalized_config_json.as_bytes()).to_hex()
    );
    let changed_digest = mismatched_seed.config_digest.clone();
    mutate_manifest(&mut mismatched_seed, |value| {
        value["normalized_config"] = changed_config;
        value["config_digest"] = serde_json::json!(changed_digest);
    });
    let error = manifest_validation_error(mismatched_seed);
    assert!(
        error.contains("/normalized_config/rng_seed") && error.contains("root seed"),
        "unexpected error: {error}"
    );
}

#[test]
fn v2_manifest_validation_rejects_incomplete_rng_and_provenance() {
    let run_id = RunId::from_namespace_sequence(0xc011_1de0, 4);

    let mut invalid_state = manifest(run_id, "invalid-rng-state", 1_700_000_000_019);
    mutate_manifest(&mut invalid_state, |value| {
        value["random_stream"]["state"] = serde_json::json!([256]);
    });
    let error = manifest_validation_error(invalid_state);
    assert!(
        error.contains("/random_stream/state/0 must be an integer byte"),
        "unexpected error: {error}"
    );

    let mut false_completeness = manifest(run_id, "false-completeness", 1_700_000_000_020);
    mutate_manifest(&mut false_completeness, |value| {
        value["build"]["provenance_complete"] = serde_json::json!(false);
    });
    let error = manifest_validation_error(false_completeness);
    assert!(
        error.contains("/build/provenance_complete") && error.contains("reproducible=true"),
        "unexpected error: {error}"
    );

    let mut missing_rustc = manifest(run_id, "missing-rustc", 1_700_000_000_021);
    mutate_manifest(&mut missing_rustc, |value| {
        value["build"]["rustc_vv"] = serde_json::Value::Null;
    });
    let error = manifest_validation_error(missing_rustc);
    assert!(
        error.contains("embedded evidence derives false"),
        "unexpected error: {error}"
    );

    let mut wrong_tree_projection = manifest(run_id, "wrong-tree-projection", 1_700_000_000_022);
    mutate_manifest(&mut wrong_tree_projection, |value| {
        value["build"]["source_diff_digest"] = serde_json::json!("blake3:other-tree");
    });
    let error = manifest_validation_error(wrong_tree_projection);
    assert!(
        error.contains("source-tree digest projection"),
        "unexpected error: {error}"
    );
}

#[test]
fn v21_manifest_requires_explicit_bootstrap_evidence() {
    let run_id = RunId::from_namespace_sequence(0xc011_1de0, 5);
    let mut missing_evidence = manifest(run_id, "v21-missing-evidence", 1_700_000_000_023);
    mutate_manifest(&mut missing_evidence, |value| {
        value["schema"] = serde_json::json!("scriptbots.run-manifest.v2.1");
    });
    let error = manifest_validation_error(missing_evidence);
    assert!(
        error.contains("/bootstrap_evidence is required"),
        "unexpected error: {error}"
    );
}

#[test]
fn append_refuses_to_strand_an_earlier_runs_pending_outbox()
-> Result<(), Box<dyn std::error::Error>> {
    let path = temp_db_path("pending_prior_run");
    let run_a = RunId::from_namespace_sequence(0xa11c_0a7b, 1);
    let run_b = RunId::from_namespace_sequence(0xb22d_0a7b, 1);
    let mut first = StoragePipeline::create_new_file_for_run(
        &path,
        manifest(run_a, "prior", 1_700_000_000_020),
    )?;
    first.shutdown()?;

    let connection = Connection::open(&path)?;
    connection.execute_with_params(
        "INSERT INTO storage_batch_ledger (
             run_id, batch_id, tick, payload_digest, state
         ) VALUES (?1, 1, 99, 'blake3:pending', 'admitted')",
        &[run_a.to_string().into()],
    )?;
    connection.execute_with_params(
        "INSERT INTO storage_outbox (run_id, batch_id, payload) VALUES (?1, 1, '{}')",
        &[run_a.to_string().into()],
    )?;
    connection.execute_with_params(
        "UPDATE storage_progress SET admitted_batch_id = 1
         WHERE run_id = ?1 AND singleton = 1",
        &[run_a.to_string().into()],
    )?;
    connection.close()?;

    let refusal = match StoragePipeline::append_run(
        &path,
        manifest(run_b, "must-not-append", 1_700_000_000_021),
    ) {
        Err(error) => error,
        Ok(mut unexpected) => {
            unexpected.shutdown()?;
            return Err("append must not strand a prior run's admitted payload".into());
        }
    };
    assert!(
        refusal
            .to_string()
            .contains("recover it before appending another run"),
        "unexpected pending-run refusal: {refusal}"
    );
    let catalog = StorageReader::catalog_page(&path, 0, 8)?;
    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog[0].run_id, run_a);

    Ok(())
}

#[test]
fn legacy_v5_database_is_refused_before_its_primary_file_is_mutated()
-> Result<(), Box<dyn std::error::Error>> {
    let path = temp_db_path("legacy_v5_refusal");
    let legacy = Connection::open(&path)?;
    legacy.execute_batch(
        "CREATE TABLE _schema_migrations (
             version INTEGER PRIMARY KEY,
             name TEXT NOT NULL,
             applied_at INTEGER NOT NULL
         );
         INSERT INTO _schema_migrations (version, name, applied_at) VALUES
             (3, 'create_stable_agent_uid_schema', 1),
             (4, 'create_stable_uid_persistence_outbox', 2),
             (5, 'record_birth_origin', 3);
         CREATE TABLE legacy_sentinel (value TEXT PRIMARY KEY);
         INSERT INTO legacy_sentinel (value) VALUES ('must-survive-refusal');
         PRAGMA user_version = 5;",
    )?;
    legacy.close()?;

    let before = fs::read(&path)?;
    let new_run = RunId::from_namespace_sequence(0xc33e_e003, 1);
    let refusal = StoragePipeline::append_run(
        &path,
        manifest(new_run, "must-not-register", 1_700_000_000_003),
    );
    let error = match refusal {
        Err(error) => error,
        Ok(mut unexpected) => {
            unexpected.shutdown()?;
            return Err("a legacy v5 database must not be upgraded implicitly".into());
        }
    };
    assert!(
        error
            .to_string()
            .contains("expected exactly one ScriptBots v6 migration, found 3"),
        "unexpected legacy-schema refusal: {error}"
    );
    let after = fs::read(&path)?;
    assert_eq!(
        after, before,
        "refusing an unsupported legacy schema must not rewrite the primary database file"
    );

    let verification = Connection::open(&path)?;
    let sentinel = verification
        .query_row("SELECT value FROM legacy_sentinel")?
        .get_typed::<String>(0)?;
    let user_version = verification
        .query_row("PRAGMA user_version")?
        .get_typed::<i64>(0)?;
    assert_eq!(sentinel, "must-survive-refusal");
    assert_eq!(user_version, 5);
    verification.close()?;

    Ok(())
}
