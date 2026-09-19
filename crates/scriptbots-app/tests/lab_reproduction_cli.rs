//! Mock-free CLI reproduction E2E for bd-16g.1.7 acceptance item 6.
//!
//! A real matched-seed cohort runs through the canonical `MatchedSeedExecutor`
//! (real worlds, real bundles, real statistics — no provider), the notebook is
//! materialized by the real renderer, and the emitted `reproduce.sh` is executed
//! through the real binary. Each tamper negative then makes the same script fail.

use scriptbots_app::BuildProvenanceV0;
use scriptbots_app::lab::llm::{PROPOSE_EXPERIMENT_TOOL_NAME, ScriptedClient, ScriptedTurn};
use scriptbots_app::lab::notebook::{
    NotebookContext, NotebookRenderer, claims_from_analysis, run_refs,
};
use scriptbots_app::lab_assistant::{LabBudget, LabPhase, LabStateMachine};

const EXE: &str = env!("CARGO_BIN_EXE_scriptbots-app");

fn spec_input() -> serde_json::Value {
    serde_json::json!({
        "hypothesis": "faster food growth raises the final population",
        "falsifier": "matched seeds show no increase in alive agents",
        "factors": [{
            "knob_path": "food_growth_rate",
            "values": [0.01, 0.02]
        }],
        "seeds": {"base": 41, "count": 2},
        "ticks_per_run": 2,
        "metrics": ["alive_agents"],
        "budget": {"runs": 4, "ticks": 8}
    })
}

/// Runs the REAL pipeline end-to-end: validated spec -> real worlds -> real
/// analysis -> materialized notebook. No provider and no mocks anywhere.
fn real_notebook(root: &std::path::Path) -> std::path::PathBuf {
    materialize_real_notebook(root, false)
}

fn materialize_real_notebook(
    root: &std::path::Path,
    shuffled_rerender: bool,
) -> std::path::PathBuf {
    let turn = ScriptedTurn {
        body: serde_json::json!({
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 4, "output_tokens": 6},
            "content": [{
                "type": "tool_use",
                "name": PROPOSE_EXPERIMENT_TOOL_NAME,
                "input": spec_input()
            }]
        }),
    };
    let mut lab = LabStateMachine::new(
        Box::new(ScriptedClient::new("offline-scripted", vec![turn])),
        LabBudget {
            max_runs: 4,
            max_ticks: 8,
            max_tokens: 1_000,
            max_iterations: 10,
        },
        root.to_path_buf(),
    );
    for expected in [
        LabPhase::Validate,
        LabPhase::Execute,
        LabPhase::Analyze,
        LabPhase::Report,
        LabPhase::Finished,
    ] {
        assert_eq!(lab.step().expect("lab transition"), expected);
    }
    let summaries = lab.run_summaries.clone();
    let analysis = lab.analysis.clone().expect("canonical analysis");
    let claims = claims_from_analysis(&analysis, &summaries, "hypothesis", "falsifier")
        .expect("claims rederive");
    let context = NotebookContext {
        model_id: "offline-scripted".to_owned(),
        build: BuildProvenanceV0::current(),
        max_runs: 4,
        runs_charged: 4,
        max_ticks: 8,
        ticks_charged: 8,
        max_tokens: 1_000,
        tokens_charged: 10,
        max_iterations: 10,
        iterations: 4,
        failure_reason: None,
    };
    let out = root.join("notebook");
    NotebookRenderer::render_notebook(
        "e2e-reproduction",
        "faster food growth raises the final population",
        &claims,
        &run_refs(&summaries),
        &out,
        &context,
    )
    .expect("materialized notebook");
    if shuffled_rerender {
        let retained = retained_files(&out);
        let mut shuffled_runs = run_refs(&summaries);
        shuffled_runs.reverse();
        let mut shuffled_claims = claims.clone();
        shuffled_claims.reverse();
        NotebookRenderer::render_notebook(
            "e2e-reproduction",
            "faster food growth raises the final population",
            &shuffled_claims,
            &shuffled_runs,
            &out,
            &context,
        )
        .expect("shuffled identical cohort rerenders at the same immutable destination");
        assert_eq!(
            retained_files(&out),
            retained,
            "rerender changed retained evidence"
        );
    }
    out
}

fn run_script(notebook: &std::path::Path) -> std::process::Output {
    std::process::Command::new("bash")
        .arg(notebook.join("reproduce.sh"))
        .env("SCRIPTBOTS_BIN", EXE)
        .env_remove("SCRIPTBOTS_DET_RUN")
        .output()
        .expect("spawn reproduce.sh")
}

fn log(output: &std::process::Output) -> String {
    format!(
        "exit={:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn retained_files(
    root: &std::path::Path,
) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    let mut files = std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(root).expect("read evidence directory") {
        let entry = entry.expect("evidence entry");
        let path = entry.path();
        if entry.file_type().expect("evidence type").is_dir() {
            files.extend(retained_files(&path));
        } else {
            files.insert(
                path.clone(),
                std::fs::read(path).expect("retained evidence bytes"),
            );
        }
    }
    files
}

fn read_json(path: &std::path::Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).expect("read JSON artifact"))
        .expect("valid JSON artifact")
}

// Preserve valid JSON and align the retained tables and emitted script's input
// pin, so these tests reach semantic verification rather than the outer hash.
fn repin_input(notebook: &std::path::Path, input: &serde_json::Value) {
    let path = notebook.join("reproduction.json");
    let old_digest = blake3::hash(&std::fs::read(&path).expect("original input"))
        .to_hex()
        .to_string();
    let bytes = serde_json::to_vec_pretty(input).expect("serialize tampered input");
    let new_digest = blake3::hash(&bytes).to_hex().to_string();
    std::fs::write(path, bytes).expect("retain tampered input");
    for (name, value) in [
        ("summaries.json", &input["runs"]),
        ("analysis.json", &input["analysis"]),
    ] {
        std::fs::write(
            notebook.join(name),
            serde_json::to_vec_pretty(value).unwrap(),
        )
        .expect("align retained table");
    }
    let script = notebook.join("reproduce.sh");
    let emitted = std::fs::read_to_string(&script).expect("emitted script");
    std::fs::write(script, emitted.replace(&old_digest, &new_digest))
        .expect("repin real emitted script");
}

fn assert_semantic_refusal(mutate: impl FnOnce(&mut serde_json::Value), diagnostic: &str) {
    let temp = tempfile::tempdir().expect("temp root");
    let notebook = real_notebook(temp.path());
    let mut input = read_json(&notebook.join("reproduction.json"));
    mutate(&mut input);
    repin_input(&notebook, &input);
    // Snapshot the entire cohort, including bundle manifests, CSV exports and
    // run evidence, not just the particular JSON field under attack.
    let retained = retained_files(temp.path());
    let output = run_script(&notebook);
    assert_ne!(output.status.code(), Some(0), "{}", log(&output));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(diagnostic),
        "expected {diagnostic}: {}",
        log(&output)
    );
    assert!(
        !stderr.contains("reproduction input BLAKE3 mismatch"),
        "{}",
        log(&output)
    );
    assert_eq!(
        retained_files(temp.path()),
        retained,
        "refusal rewrote evidence or began a rerun"
    );
}

#[test]
fn tampered_seed_is_refused_by_normalized_config_verification() {
    assert_semantic_refusal(
        |input| {
            let seed = input["runs"][0]["reference"]["seed"].as_u64().unwrap();
            input["runs"][0]["reference"]["seed"] = (seed + 100).into();
        },
        "normalized config mismatch",
    );
}

#[test]
fn tampered_world_digest_is_refused_by_run_evidence_verification() {
    assert_semantic_refusal(
        |input| {
            let digest = input["runs"][0]["reference"]["digest"].as_str().unwrap();
            let mut tampered = digest.to_owned();
            tampered.replace_range(..1, if digest.starts_with('0') { "1" } else { "0" });
            input["runs"][0]["reference"]["digest"] = tampered.into();
        },
        "evidence mismatch run=",
    );
}

#[test]
fn tampered_summary_artifact_hash_is_refused_by_scientific_summary_verification() {
    assert_semantic_refusal(
        |input| {
            let digest = input["runs"][0]["reference"]["summary_artifact_digest"]
                .as_str()
                .unwrap();
            let mut tampered = digest.to_owned();
            tampered.replace_range(..1, if digest.starts_with('0') { "1" } else { "0" });
            input["runs"][0]["reference"]["summary_artifact_digest"] = tampered.into();
        },
        "scientific summary mismatch run=",
    );
}

#[test]
fn tampered_adjusted_p_value_is_refused_by_canonical_analysis_verification() {
    assert_semantic_refusal(
        |input| {
            let adjusted = &mut input["analysis"]["effects"][0]["adjusted"]["adjusted_p_value"];
            let original = adjusted.as_f64().expect("real adjusted p value");
            *adjusted = if original == 0.0 { 0.5 } else { 0.0 }.into();
        },
        "retained canonical analysis differs",
    );
}

#[test]
fn tampered_build_identity_is_refused_before_execution() {
    assert_semantic_refusal(
        |input| {
            let version = input["context"]["build"]["package_version"]
                .as_str()
                .unwrap();
            input["context"]["build"]["package_version"] = format!("{version}-tampered").into();
        },
        "expected source/build provenance mismatch",
    );
}

#[test]
fn shuffled_cohort_rerender_preserves_retained_artifacts() {
    let temp = tempfile::tempdir().expect("temp root");
    let notebook = materialize_real_notebook(temp.path(), true);
    let output = run_script(&notebook);
    assert_eq!(output.status.code(), Some(0), "{}", log(&output));
}

#[test]
fn clean_cohort_reproduces_every_arm_and_seed() {
    let temp = tempfile::tempdir().expect("temp root");
    let notebook = real_notebook(temp.path());
    let output = run_script(&notebook);
    assert_eq!(
        output.status.code(),
        Some(0),
        "clean reproduction must pass:\n{}",
        log(&output)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Detailed provenance logging, no secrets/prompts.
    for token in ["[REPRODUCE]", "[CHILD]", "[VERIFY]", "e2e-reproduction"] {
        assert!(stderr.contains(token), "missing {token} in:\n{stderr}");
    }
    let regenerated = notebook.join("rerun-000000");
    assert_eq!(
        read_json(&regenerated.join("analysis.json")),
        read_json(&notebook.join("analysis.json")),
        "complete canonical raw/adjusted analysis must reproduce"
    );
    let mut expected = read_json(&notebook.join("summaries.json"));
    let mut actual = read_json(&regenerated.join("summaries.json"));
    let expected_rows = expected.as_array_mut().expect("retained summary rows");
    let actual_rows = actual.as_array_mut().expect("regenerated summary rows");
    let expected_keys = [(0, 41), (0, 42), (1, 41), (1, 42)];
    assert_eq!(actual_rows.len(), expected_keys.len());
    for (index, (arm, seed)) in expected_keys.into_iter().enumerate() {
        let row = &actual_rows[index]["reference"];
        assert_eq!(row["arm_id"], arm);
        assert_eq!(row["seed"], seed);
        let record = read_json(&regenerated.join(format!("record-{index:04}.json")));
        assert_eq!(record["state"], "Completed");
        assert_eq!(record["run_id"], row["run_id"]);
        assert_eq!(record["variant_id"], format!("arm-{arm:03}"));
        assert_eq!(record["seed"], seed);
        assert_eq!(record["total_ticks"], 2);
        assert_eq!(record["final_digest"], row["digest"]);
    }
    // Only the retained CSV's location changes; hashes, metrics, seeds, configs
    // and all other scientific provenance must remain byte-for-byte equivalent.
    for rows in [expected_rows, actual_rows] {
        for row in rows {
            let path = row["reference"]["summary_path"]
                .as_str()
                .expect("summary path");
            assert!(std::path::Path::new(path).is_file());
            row["reference"]["summary_path"] = serde_json::Value::Null;
        }
    }
    assert_eq!(
        actual, expected,
        "canonical regenerated scientific summaries differ"
    );
}

#[test]
fn tampered_input_digest_fails_closed() {
    let temp = tempfile::tempdir().expect("temp root");
    let notebook = real_notebook(temp.path());
    // Flip one byte in the pinned input; the script pins the ORIGINAL digest.
    let input = notebook.join("reproduction.json");
    let mut bytes = std::fs::read(&input).expect("read input");
    let pivot = bytes.iter().position(|b| *b == b'4').expect("pivot byte");
    bytes[pivot] = b'5';
    std::fs::write(&input, &bytes).expect("rewrite input");
    let output = run_script(&notebook);
    assert_ne!(
        output.status.code(),
        Some(0),
        "tampered input must fail:\n{}",
        log(&output)
    );
}

#[test]
fn tampered_summary_and_analysis_tables_fail_closed() {
    let temp = tempfile::tempdir().expect("temp root");
    let notebook = real_notebook(temp.path());
    // Retained tables are immutable artifacts: any byte change must fail the
    // run, and the offending bytes must never be "repaired" by the script.
    for name in ["summaries.json", "analysis.json"] {
        let path = notebook.join(name);
        let original = std::fs::read(&path).expect("read table");
        let mut tampered = original.clone();
        let last = tampered.len() - 1;
        tampered[last] = tampered[last].wrapping_add(1);
        std::fs::write(&path, &tampered).expect("tamper table");
        let output = run_script(&notebook);
        assert_ne!(
            output.status.code(),
            Some(0),
            "tampered {name} must fail:\n{}",
            log(&output)
        );
        assert_eq!(
            std::fs::read(&path).expect("table bytes"),
            tampered,
            "tampered table must not be rewritten by the script"
        );
        std::fs::write(&path, &original).expect("restore table");
    }
}

#[test]
fn tampered_retained_config_is_refused_without_overwrite() {
    let temp = tempfile::tempdir().expect("temp root");
    let notebook = real_notebook(temp.path());
    let config = notebook.join("config-0000.toml");
    let tampered = b"rng_seed = 999\n";
    std::fs::write(&config, tampered).expect("tamper config");
    let output = run_script(&notebook);
    assert_ne!(
        output.status.code(),
        Some(0),
        "tampered config must fail:\n{}",
        log(&output)
    );
    assert_eq!(
        std::fs::read(&config).expect("config bytes"),
        tampered,
        "config must not be overwritten"
    );
}

#[test]
fn symlink_artifact_escape_is_refused() {
    let temp = tempfile::tempdir().expect("temp root");
    let notebook = real_notebook(temp.path());
    let outside = temp.path().join("outside.toml");
    std::fs::write(&outside, b"rng_seed = 1\n").expect("outside file");
    let config = notebook.join("config-0000.toml");
    std::fs::remove_file(&config).expect("remove retained config");
    std::os::unix::fs::symlink(&outside, &config).expect("plant symlink");
    let output = run_script(&notebook);
    assert_ne!(
        output.status.code(),
        Some(0),
        "symlink must be refused:\n{}",
        log(&output)
    );
    assert_eq!(
        std::fs::read(&outside).expect("outside bytes"),
        b"rng_seed = 1\n",
        "symlink target must never be written through"
    );
}

#[test]
fn rerun_is_rerunnable_with_unique_output_dirs() {
    let temp = tempfile::tempdir().expect("temp root");
    let notebook = real_notebook(temp.path());
    for run in 0..2 {
        let output = run_script(&notebook);
        assert_eq!(
            output.status.code(),
            Some(0),
            "rerun {run} must pass:\n{}",
            log(&output)
        );
        assert!(
            notebook.join(format!("rerun-{run:06}")).is_dir(),
            "unique dir per run"
        );
    }
}

#[test]
fn failing_executable_exit_fails_closed() {
    let temp = tempfile::tempdir().expect("temp root");
    let notebook = real_notebook(temp.path());
    let output = std::process::Command::new("bash")
        .arg(notebook.join("reproduce.sh"))
        .env("SCRIPTBOTS_BIN", "/bin/false")
        .output()
        .expect("spawn");
    assert_ne!(
        output.status.code(),
        Some(0),
        "failing executable must fail closed"
    );
}

#[test]
fn autonomous_lab_subcommand_produces_notebook_and_passes_reproduction() {
    let temp = tempfile::tempdir().expect("temp root");
    let out_dir = temp.path().join("lab_out");

    let status = std::process::Command::new(EXE)
        .arg("lab")
        .arg("--goal")
        .arg("characterize how food_regrowth_rate affects carnivore/herbivore equilibrium")
        .arg("--budget")
        .arg("4-runs")
        .arg("--ticks")
        .arg("16")
        .arg("--offline-fixture")
        .arg("--out")
        .arg(&out_dir)
        .status()
        .expect("spawn scriptbots-app lab");

    assert!(
        status.success(),
        "scriptbots-app lab exited with status {status}"
    );

    let mut notebook_dir = None;
    for entry in std::fs::read_dir(&out_dir).expect("read out_dir") {
        let entry = entry.expect("entry");
        let path = entry.path();
        if path.is_dir() && path.join("notebook").is_dir() {
            notebook_dir = Some(path.join("notebook"));
            break;
        }
    }
    let notebook = notebook_dir.expect("found generated notebook dir");
    assert!(notebook.join("notebook.md").is_file(), "notebook.md exists");
    assert!(
        notebook.join("reproduce.sh").is_file(),
        "reproduce.sh exists"
    );
    assert!(
        notebook.join("reproduction.json").is_file(),
        "reproduction.json exists"
    );
    assert!(
        notebook.join("summaries.json").is_file(),
        "summaries.json exists"
    );
    assert!(
        notebook.join("analysis.json").is_file(),
        "analysis.json exists"
    );

    let notebook_content =
        std::fs::read_to_string(notebook.join("notebook.md")).expect("read notebook.md");
    assert!(
        notebook_content.contains("carnivore/herbivore equilibrium"),
        "notebook should contain research goal: {notebook_content}"
    );
    assert!(
        notebook_content.contains("reproduce.sh"),
        "notebook should contain reproduction command"
    );

    let repro_output = run_script(&notebook);
    assert_eq!(
        repro_output.status.code(),
        Some(0),
        "reproduce.sh must pass:\n{}",
        log(&repro_output)
    );
}
