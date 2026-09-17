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
    // The rerun landed in a unique confined child directory with regenerated tables.
    assert!(
        notebook.join("rerun-000000").is_dir(),
        "rerun output retained"
    );
    assert!(notebook.join("rerun-000000/summaries.json").is_file());
    assert!(notebook.join("rerun-000000/analysis.json").is_file());
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
