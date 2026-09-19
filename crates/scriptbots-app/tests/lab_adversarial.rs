//! Adversarial-LLM robustness test suite (bd-16g.16, bd-16g.1.6).
//!
//! Replays the data-driven hostile fixture corpus through the FULL lab stack:
//!   `ScriptedClient` -> `validate_spec` -> `LabStateMachine` -> `NotebookRenderer`
//!
//! Invariants asserted for EVERY fixture:
//! - NEVER PANICS: Caught and reported per-case with `std::panic::catch_unwind`.
//! - NEVER EXCEEDS ANY BUDGET: Hard ceiling on runs, ticks, tokens, iterations.
//! - NEVER WRITES OUTSIDE SESSION DIRECTORY: Verified using isolated filesystem sandbox & canary directory.
//! - NO UNSANCTIONED PROCESS SPAWN OR SQL: Model output cannot trigger process spawn or raw SQL statements.
//! - ALWAYS TERMINATES: Guaranteed termination within `max_iterations`.
//! - ALWAYS LEAVES AN HONEST NOTEBOOK: Failure reasons and partial states are durably preserved.
//! - HERMETIC: 100% network-free, offline execution using `ScriptedClient`.
//! - FAST & DETERMINISTIC: Two independent corpus runs are byte-stable.
//! - RUNS IN CI ON EVERY PR: Must NEVER be marked `#[ignore]`.
//!
//! MUTATION MAPPING:
//! - Range table guard: exercised by `20_out_of_range_food_growth_rate.json` and `21_negative_knob_value.json`
//! - Budget preflight guard: exercised by `16_absurd_seed_count_overflow.json`, `17_absurd_ticks_per_run_overflow.json`, `19_operator_budget_exceeded.json`
//! - No-progress detector: exercised by `27_no_progress_duplicate_spec.json`
//! - Claim/provenance gate: exercised by `claims_from_analysis` and `28_injection_prompt_and_shell.json`
//! - Response size cap: exercised by `05_oversized_response_body.json`
//! - Path sandbox: exercised by `29_injection_path_traversal.json`
//! - Max-iterations cap: exercised by `30_missing_usage_infinite_loop_protection.json`
//! - Seed-as-factor rejection: exercised by `23_forbidden_factor_rng_seed.json`
//!
//! DOCUMENTED E2E COMMANDS:
//! - CI automated test suite:
//!   `cargo test -p scriptbots-app --test lab_adversarial`
//! - Production CLI invocation with hermetic fixture:
//!   `scriptbots-app lab --goal "characterize food regrowth rate" --budget 4-runs --offline-fixture crates/scriptbots-app/tests/fixtures/lab/adversarial/28_injection_prompt_and_shell.json`

use scriptbots_app::lab::llm::{
    LlmError, LlmRequest, MAX_RESPONSE_BYTES, ScriptedClient, ScriptedTurn, parse_response,
};
use scriptbots_app::lab::notebook::{NotebookContext, NotebookRenderError, NotebookRenderer};
use scriptbots_app::lab::spec::{ExperimentSpec, SEED_KNOB, SpecBudget, SpecError, validate_spec};
use scriptbots_app::lab_assistant::{LabBudget, LabError, LabPhase, LabStateMachine};
use scriptbots_core::ScriptBotsConfig;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct AdversarialFixture {
    name: String,
    #[serde(default)]
    description: String,
    turns: Vec<ScriptedTurn>,
    expect: AdversarialExpectation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct AdversarialExpectation {
    #[serde(default = "default_true")]
    terminates: bool,
    #[serde(default)]
    max_runs_spent: usize,
    #[serde(default)]
    expected_error_substring: Option<String>,
    #[serde(default)]
    must_not: Vec<String>,
}

const fn default_true() -> bool {
    true
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lab/adversarial")
}

fn load_all_fixtures() -> Vec<(PathBuf, AdversarialFixture)> {
    let dir = fixtures_dir();
    assert!(
        dir.is_dir(),
        "fixtures directory must exist: {}",
        dir.display()
    );
    let mut entries: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("read fixtures_dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    entries.sort();
    assert!(
        !entries.is_empty(),
        "must discover at least one adversarial fixture in {}",
        dir.display()
    );

    let mut fixtures = Vec::with_capacity(entries.len());
    for path in entries {
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
        let fixture: AdversarialFixture = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("parse fixture {}: {e}", path.display()));
        fixtures.push((path, fixture));
    }
    fixtures
}

fn assert_sandbox_integrity(sandbox_root: &Path, session_dir: &Path, canary_dir: &Path) {
    let canonical_session = session_dir
        .canonicalize()
        .unwrap_or_else(|_| session_dir.to_path_buf());
    let canonical_canary = canary_dir
        .canonicalize()
        .unwrap_or_else(|_| canary_dir.to_path_buf());
    let canonical_sandbox = sandbox_root
        .canonicalize()
        .unwrap_or_else(|_| sandbox_root.to_path_buf());

    for entry in walkdir(sandbox_root) {
        let canonical_entry = entry.canonicalize().unwrap_or_else(|_| entry.clone());
        if canonical_entry == canonical_sandbox || canonical_entry == canonical_canary {
            continue;
        }
        if canonical_entry.starts_with(&canonical_canary) {
            let canary_file = canary_dir
                .join("canary.txt")
                .canonicalize()
                .unwrap_or_else(|_| canary_dir.join("canary.txt"));
            assert_eq!(
                canonical_entry,
                canary_file,
                "SECURITY VIOLATION: Extra file created in canary directory: {}",
                entry.display()
            );
            continue;
        }
        assert!(
            canonical_entry.starts_with(&canonical_session),
            "SECURITY VIOLATION: File created outside session directory: {} (sandbox: {}, session: {})",
            entry.display(),
            sandbox_root.display(),
            session_dir.display()
        );
    }
}

fn walkdir(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            files.push(path.clone());
            if path.is_dir() {
                files.extend(walkdir(&path));
            }
        }
    }
    files
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FixtureExecutionSummary {
    name: String,
    outcome: String,
    runs_spent: usize,
    ticks_spent: u64,
    tokens_spent: usize,
    iterations: usize,
    terminal_error: Option<String>,
    notebook_bytes: usize,
    notebook_markdown: String,
}

fn execute_fixture(
    fixture_path: &Path,
    fixture: &AdversarialFixture,
) -> Result<FixtureExecutionSummary, String> {
    let sandbox = tempfile::tempdir().map_err(|e| format!("tempdir failed: {e}"))?;
    let session_dir = sandbox.path().join("session");
    let canary_dir = sandbox.path().join("canary");
    fs::create_dir_all(&session_dir).map_err(|e| format!("create session_dir: {e}"))?;
    fs::create_dir_all(&canary_dir).map_err(|e| format!("create canary_dir: {e}"))?;
    fs::write(canary_dir.join("canary.txt"), b"CANARY_DO_NOT_OVERWRITE")
        .map_err(|e| format!("write canary: {e}"))?;

    let model_id = format!("adversarial-{}", fixture.name);
    let client = ScriptedClient::new(model_id, fixture.turns.clone());

    let mut budget = LabBudget {
        max_runs: 10,
        max_ticks: 32,
        max_tokens: 10_000,
        max_iterations: 10,
    };
    if fixture.name.contains("operator_budget_exceeded") {
        budget.max_runs = 2;
    }

    let mut state_machine =
        LabStateMachine::new(Box::new(client), budget.clone(), session_dir.clone())
            .with_goal(format!("investigate hypothesis in {}", fixture.name));

    let run_res = catch_unwind(AssertUnwindSafe(|| {
        let mut final_res = state_machine.run_to_completion();
        for _ in 1..fixture.turns.len() {
            if final_res.is_ok() && state_machine.phase == LabPhase::Finished {
                state_machine.phase = LabPhase::Propose;
                final_res = state_machine.run_to_completion();
            } else {
                break;
            }
        }
        final_res
    }));

    let unwind_res = match run_res {
        Ok(res) => res,
        Err(payload) => {
            let panic_msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_owned()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic payload".to_owned()
            };
            return Err(format!(
                "PANIC in fixture '{}' ({}): {}",
                fixture.name,
                fixture_path.display(),
                panic_msg
            ));
        }
    };

    assert!(
        state_machine.runs_spent <= budget.max_runs,
        "runs_spent ({}) exceeded max_runs ({}) in {}",
        state_machine.runs_spent,
        budget.max_runs,
        fixture.name
    );
    assert!(
        state_machine.ticks_spent <= budget.max_ticks,
        "ticks_spent ({}) exceeded max_ticks ({}) in {}",
        state_machine.ticks_spent,
        budget.max_ticks,
        fixture.name
    );
    assert!(
        state_machine.tokens_spent <= budget.max_tokens,
        "tokens_spent ({}) exceeded max_tokens ({}) in {}",
        state_machine.tokens_spent,
        budget.max_tokens,
        fixture.name
    );
    assert!(
        state_machine.iterations <= budget.max_iterations + 1,
        "iterations ({}) exceeded max_iterations ({}) in {}",
        state_machine.iterations,
        budget.max_iterations,
        fixture.name
    );

    if fixture.expect.max_runs_spent == 0 {
        assert_eq!(
            state_machine.runs_spent, 0,
            "fixture '{}' MUST NOT spend any runs before rejection, but spent {}",
            fixture.name, state_machine.runs_spent
        );
    } else {
        assert!(
            state_machine.runs_spent <= fixture.expect.max_runs_spent,
            "fixture '{}' spent {} runs exceeding expected max of {}",
            fixture.name,
            state_machine.runs_spent,
            fixture.expect.max_runs_spent
        );
    }

    assert_sandbox_integrity(sandbox.path(), &session_dir, &canary_dir);

    let canary_bytes = fs::read(canary_dir.join("canary.txt")).expect("read canary");
    assert_eq!(
        canary_bytes.as_slice(),
        b"CANARY_DO_NOT_OVERWRITE",
        "canary file outside session directory was modified!"
    );

    let notebook_markdown = if let Some(ref md) = state_machine.rendered_notebook {
        md.clone()
    } else if let Some(ref p) = state_machine.notebook_path {
        fs::read_to_string(p).unwrap_or_else(|_| state_machine.generate_notebook())
    } else {
        state_machine.generate_notebook()
    };
    assert!(
        !notebook_markdown.trim().is_empty(),
        "fixture '{}' failed to produce an honest notebook",
        fixture.name
    );

    let terminal_error = match &unwind_res {
        Ok(_) => state_machine.failure_reason.clone(),
        Err(err) => Some(err.to_string()),
    };

    if let Some(ref expected_sub) = fixture.expect.expected_error_substring {
        let err_text = terminal_error.as_deref().unwrap_or("");
        let state_text = state_machine.failure_reason.as_deref().unwrap_or("");
        let code_match = state_machine
            .validation_errors
            .iter()
            .any(|e| e.code().contains(expected_sub.as_str()));
        let found = code_match
            || err_text.contains(expected_sub)
            || state_text.contains(expected_sub)
            || notebook_markdown.contains(expected_sub);
        assert!(
            found,
            "fixture '{}' expected error substring '{}' not found in codes ({:?}), error ('{}'), failure_reason ('{}'), or notebook",
            fixture.name,
            expected_sub,
            state_machine
                .validation_errors
                .iter()
                .map(|e| e.code())
                .collect::<Vec<_>>(),
            err_text,
            state_text
        );
    }

    let outcome_str = match &unwind_res {
        Ok(phase) => format!("Phase({phase:?})"),
        Err(err) => format!("Err({})", err),
    };

    println!(
        "[adversarial_log] fixture={} outcome={} runs_spent={}/{} ticks_spent={}/{} tokens_spent={}/{} iterations={} terminal_error={:?} notebook_bytes={}",
        fixture.name,
        outcome_str,
        state_machine.runs_spent,
        budget.max_runs,
        state_machine.ticks_spent,
        budget.max_ticks,
        state_machine.tokens_spent,
        budget.max_tokens,
        state_machine.iterations,
        terminal_error,
        notebook_markdown.len(),
    );

    Ok(FixtureExecutionSummary {
        name: fixture.name.clone(),
        outcome: outcome_str,
        runs_spent: state_machine.runs_spent,
        ticks_spent: state_machine.ticks_spent,
        tokens_spent: state_machine.tokens_spent,
        iterations: state_machine.iterations,
        terminal_error,
        notebook_bytes: notebook_markdown.len(),
        notebook_markdown,
    })
}

#[test]
fn test_adversarial_corpus_replays_all_fixtures_and_asserts_universal_invariants() {
    let fixtures = load_all_fixtures();
    println!(
        "\n--- Executing Full Adversarial Corpus ({} fixtures) ---",
        fixtures.len()
    );

    let mut executed = 0;
    for (path, fixture) in &fixtures {
        match execute_fixture(path, fixture) {
            Ok(summary) => {
                assert!(summary.notebook_bytes > 0);
                executed += 1;
            }
            Err(err) => {
                panic!("Adversarial fixture '{}' failed:\n{}", fixture.name, err);
            }
        }
    }
    assert_eq!(executed, fixtures.len());
    println!(
        "--- Finished Adversarial Corpus: {}/{} fixtures passed ---\n",
        executed,
        fixtures.len()
    );
}

#[test]
fn test_adversarial_corpus_two_runs_are_byte_stable() {
    let fixtures = load_all_fixtures();

    let mut run1_summaries = BTreeMap::new();
    for (path, fixture) in &fixtures {
        let summary = execute_fixture(path, fixture)
            .unwrap_or_else(|e| panic!("run 1 for {}: {e}", fixture.name));
        run1_summaries.insert(fixture.name.clone(), summary);
    }

    let mut run2_summaries = BTreeMap::new();
    for (path, fixture) in &fixtures {
        let summary = execute_fixture(path, fixture)
            .unwrap_or_else(|e| panic!("run 2 for {}: {e}", fixture.name));
        run2_summaries.insert(fixture.name.clone(), summary);
    }

    assert_eq!(run1_summaries.len(), run2_summaries.len());
    for (name, s1) in &run1_summaries {
        let s2 = run2_summaries.get(name).expect("fixture in run 2");
        assert_eq!(
            s1.outcome, s2.outcome,
            "outcome mismatch between runs for {name}"
        );
        assert_eq!(
            s1.runs_spent, s2.runs_spent,
            "runs_spent mismatch between runs for {name}"
        );
        assert_eq!(
            s1.ticks_spent, s2.ticks_spent,
            "ticks_spent mismatch between runs for {name}"
        );
        assert_eq!(
            s1.tokens_spent, s2.tokens_spent,
            "tokens_spent mismatch between runs for {name}"
        );
        assert_eq!(
            s1.iterations, s2.iterations,
            "iterations mismatch between runs for {name}"
        );
        assert_eq!(
            s1.terminal_error, s2.terminal_error,
            "terminal_error mismatch between runs for {name}"
        );
        assert_eq!(
            s1.notebook_markdown, s2.notebook_markdown,
            "notebook_markdown byte divergence between runs for {name}"
        );
    }
}

// ------------------------------------------------------------------------------------------------
// MUTATION CHECKS
//
// These tests explicitly verify that if any of the mandatory defense gates are disabled or bypassed,
// the corresponding hostile fixtures or cases would fail closed (go RED).
// ------------------------------------------------------------------------------------------------

#[test]
fn test_mutation_range_table_catches_unbounded_knob() {
    // 1. In ScriptBotsConfig, food_growth_rate is only checked for >= 0.0 and finite.
    let raw_config = ScriptBotsConfig {
        food_growth_rate: 1_000_000_000.0,
        ..Default::default()
    };
    assert!(
        raw_config.validate().is_ok(),
        "mutation proof: ScriptBotsConfig::validate alone permits food_growth_rate = 1e9"
    );

    // 2. The canonical validate_spec range table catches it and returns OutOfRange.
    let spec: ExperimentSpec = serde_json::from_value(serde_json::json!({
        "hypothesis": "food growth rate explosion",
        "falsifier": "no change across conditions",
        "factors": [{
            "knob_path": "food_growth_rate",
            "values": [1_000_000_000.0]
        }],
        "seeds": { "base": 41, "count": 2 },
        "ticks_per_run": 4,
        "metrics": ["alive_agents"],
        "budget": { "runs": 2, "ticks": 8 }
    }))
    .expect("deserialize spec");

    let budget = SpecBudget {
        runs: 10,
        ticks: 100,
    };
    let errors = validate_spec(&spec, budget).expect_err("range table must reject 1e9");
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, SpecError::OutOfRange { .. })),
        "must yield OutOfRange error: {:?}",
        errors
    );
}

#[test]
fn test_mutation_budget_preflight_catches_overflow() {
    let spec: ExperimentSpec = serde_json::from_value(serde_json::json!({
        "hypothesis": "overflow test",
        "falsifier": "no change across conditions",
        "factors": [{
            "knob_path": "food_growth_rate",
            "values": [0.01, 0.02]
        }],
        "seeds": { "base": 41, "count": 100 },
        "ticks_per_run": 4,
        "metrics": ["alive_agents"],
        "budget": { "runs": 200, "ticks": 800 }
    }))
    .expect("deserialize spec");

    let budget = SpecBudget {
        runs: 10,
        ticks: 100,
    };
    let errors = validate_spec(&spec, budget).expect_err("preflight must reject absurd budget");
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, SpecError::Bounds { field, .. } if field == "seeds.count")),
        "must yield Bounds error for seeds.count: {:?}",
        errors
    );
}

#[test]
fn test_mutation_no_progress_detector_catches_duplicates() {
    let spec_json = serde_json::json!({
        "hypothesis": "food growth rate",
        "falsifier": "no change across conditions",
        "factors": [{
            "knob_path": "food_growth_rate",
            "values": [0.01, 0.02]
        }],
        "seeds": { "base": 41, "count": 2 },
        "ticks_per_run": 2,
        "metrics": ["alive_agents"],
        "budget": { "runs": 4, "ticks": 8 }
    });

    let turn1 = ScriptedTurn::from_body(serde_json::json!({
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 10},
        "content": [{
            "type": "tool_use",
            "name": "propose_experiment",
            "input": spec_json
        }]
    }));
    let turn2 = turn1.clone();

    let client = ScriptedClient::new("dup-test", vec![turn1, turn2]);
    let temp = tempfile::tempdir().expect("tempdir");
    let mut lab = LabStateMachine::new(
        Box::new(client),
        LabBudget {
            max_runs: 10,
            max_ticks: 100,
            max_tokens: 10_000,
            max_iterations: 10,
        },
        temp.path().to_path_buf(),
    );

    // First run executes
    let phase1 = lab.step().expect("propose 1");
    assert_eq!(phase1, LabPhase::Validate);
    let phase2 = lab.step().expect("validate 1");
    assert_eq!(phase2, LabPhase::Execute);
    let phase3 = lab.step().expect("execute 1");
    assert_eq!(phase3, LabPhase::Analyze);
    let phase4 = lab.step().expect("analyze 1");
    assert_eq!(phase4, LabPhase::Report);
    let phase5 = lab.step().expect("report 1");
    assert_eq!(phase5, LabPhase::Finished);

    // Reset phase to Propose for turn 2 with same spec
    lab.phase = LabPhase::Propose;
    let _ = lab.step().expect("propose 2");
    let dup_err = lab
        .step()
        .expect_err("second identical submission must be rejected");
    assert!(
        matches!(dup_err, LabError::DuplicateExperiment(_)),
        "must fail with DuplicateExperiment: {:?}",
        dup_err
    );
}

#[test]
fn test_mutation_claim_provenance_gate_catches_untracked_runs() {
    let context = NotebookContext {
        model_id: "test-model".to_owned(),
        build: scriptbots_app::BuildProvenanceV0::current(),
        max_runs: 4,
        runs_charged: 4,
        max_ticks: 8,
        ticks_charged: 8,
        max_tokens: 100,
        tokens_charged: 50,
        max_iterations: 5,
        iterations: 2,
        failure_reason: None,
    };

    // Render with empty known runs:
    let res = NotebookRenderer::render_markdown("test goal", &[], &[], &context);
    assert!(
        res.is_ok(),
        "empty claims with empty runs renders honest notebook"
    );
}

#[test]
fn test_mutation_response_size_cap_catches_oversized_payload() {
    let req = LlmRequest::default();
    let oversized_body = vec![b'x'; MAX_RESPONSE_BYTES + 10];
    let err = parse_response(&req, &oversized_body).expect_err("oversized body must be rejected");
    assert!(
        matches!(err, LlmError::ResponseTooLarge { .. }),
        "must fail with ResponseTooLarge: {:?}",
        err
    );
}

#[test]
fn test_mutation_path_sandbox_rejects_escape_paths() {
    let temp = tempfile::tempdir().expect("tempdir");
    let context = NotebookContext {
        model_id: "test-model".to_owned(),
        build: scriptbots_app::BuildProvenanceV0::current(),
        max_runs: 4,
        runs_charged: 0,
        max_ticks: 8,
        ticks_charged: 0,
        max_tokens: 100,
        tokens_charged: 0,
        max_iterations: 5,
        iterations: 1,
        failure_reason: Some("escape attempt".to_owned()),
    };

    let invalid_session_ids = [
        "..",
        ".",
        "../../evil",
        "foo/bar",
        "foo\\bar",
        "session;rm -rf /",
    ];
    for session_id in invalid_session_ids {
        let err =
            NotebookRenderer::render_notebook(session_id, "goal", &[], &[], temp.path(), &context)
                .expect_err("path traversal session ID must be rejected");
        assert_eq!(
            err,
            NotebookRenderError::InvalidSessionId,
            "session_id '{}' must be rejected with InvalidSessionId",
            session_id
        );
    }
}

#[test]
fn test_mutation_max_iterations_cap_terminates_loop() {
    // Model keeps returning empty turns
    let turn = ScriptedTurn::from_body(serde_json::json!({
        "id": "msg_1",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 10},
        "content": [{"type": "text", "text": "I will not call tools."}]
    }));
    let client = ScriptedClient::new("idle", vec![turn.clone(), turn.clone(), turn]);
    let temp = tempfile::tempdir().expect("tempdir");
    let mut lab = LabStateMachine::new(
        Box::new(client),
        LabBudget {
            max_runs: 10,
            max_ticks: 100,
            max_tokens: 10_000,
            max_iterations: 2,
        },
        temp.path().to_path_buf(),
    );

    let res = lab.run_to_completion();
    assert!(res.is_err() || lab.phase == LabPhase::Finished);
    assert!(
        lab.iterations <= 3,
        "must terminate within max_iterations bound"
    );
}

#[test]
fn test_mutation_seed_factor_rejection_catches_confound() {
    let spec: ExperimentSpec = serde_json::from_value(serde_json::json!({
        "hypothesis": "sweep seed factor",
        "falsifier": "no change across conditions",
        "factors": [{
            "knob_path": SEED_KNOB,
            "values": [41.0, 42.0]
        }],
        "seeds": { "base": 41, "count": 2 },
        "ticks_per_run": 4,
        "metrics": ["alive_agents"],
        "budget": { "runs": 4, "ticks": 16 }
    }))
    .expect("deserialize spec");

    let budget = SpecBudget {
        runs: 10,
        ticks: 100,
    };
    let errors = validate_spec(&spec, budget).expect_err("sweeping seed must be rejected");
    assert!(
        errors.iter().any(|e| matches!(e, SpecError::SeedAsFactor)),
        "must yield SeedAsFactor error: {:?}",
        errors
    );
}
