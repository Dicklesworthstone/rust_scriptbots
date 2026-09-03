use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn app_binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_scriptbots-app"))
}

#[test]
fn economy_audit_cli_subcommand_runs_and_emits_deterministic_artifacts() {
    let nonce = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_root = std::env::temp_dir().join(format!(
        "economy_audit_e2e_{}_{}",
        std::process::id(),
        nonce
    ));
    let dir_a = temp_root.join("run_a");
    let dir_b = temp_root.join("run_b");
    let _ = std::fs::remove_dir_all(&temp_root);

    // Invocation 1
    let output_a = app_binary()
        .arg("economy-audit")
        .arg("--seeds")
        .arg("1")
        .arg("--ticks")
        .arg("32")
        .arg("--out")
        .arg(&dir_a)
        .output()
        .expect("spawn app binary run a");

    assert!(
        output_a.status.success(),
        "first invocation failed: stderr:\n{}",
        String::from_utf8_lossy(&output_a.stderr)
    );

    let stdout_a = String::from_utf8_lossy(&output_a.stdout);
    assert!(
        stdout_a.contains("economy_gate verdict=pass seeds=1 breaches=0"),
        "stdout must contain summary line: {stdout_a}"
    );

    let verdict_a_path = dir_a.join("verdict.json");
    let csv_a_path = dir_a.join("residual_1786642433.csv");
    assert!(verdict_a_path.exists(), "verdict.json must be written");
    assert!(csv_a_path.exists(), "residual CSV must be written");

    let verdict_a_bytes = std::fs::read(&verdict_a_path).expect("read verdict a");
    let csv_a_bytes = std::fs::read(&csv_a_path).expect("read csv a");

    let val_a: serde_json::Value =
        serde_json::from_slice(&verdict_a_bytes).expect("parse verdict JSON");
    assert_eq!(val_a["pass"], true);
    assert_eq!(val_a["tolerance_overridden"], false);
    assert!(val_a["config_digest"].is_string());

    // Invocation 2: identical inputs must produce byte-identical artifacts
    let output_b = app_binary()
        .arg("economy-audit")
        .arg("--seeds")
        .arg("1")
        .arg("--ticks")
        .arg("32")
        .arg("--out")
        .arg(&dir_b)
        .output()
        .expect("spawn app binary run b");

    assert!(
        output_b.status.success(),
        "second invocation failed: stderr:\n{}",
        String::from_utf8_lossy(&output_b.stderr)
    );

    let verdict_b_bytes = std::fs::read(dir_b.join("verdict.json")).expect("read verdict b");
    let csv_b_bytes = std::fs::read(dir_b.join("residual_1786642433.csv")).expect("read csv b");

    assert_eq!(
        verdict_a_bytes, verdict_b_bytes,
        "verdict JSON must be byte-identical across identical invocations"
    );
    assert_eq!(
        csv_a_bytes, csv_b_bytes,
        "residual CSV must be byte-identical across identical invocations"
    );

    let _ = std::fs::remove_dir_all(&temp_root);
}

#[test]
fn economy_audit_cli_tolerance_flag_sets_override_in_artifact() {
    let nonce = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_dir = std::env::temp_dir().join(format!(
        "economy_audit_tol_{}_{}",
        std::process::id(),
        nonce
    ));
    let _ = std::fs::remove_dir_all(&temp_dir);

    let output = app_binary()
        .arg("economy-audit")
        .arg("--seeds")
        .arg("1")
        .arg("--ticks")
        .arg("16")
        .arg("--out")
        .arg(&temp_dir)
        .arg("--tolerance")
        .arg("0.0005")
        .output()
        .expect("spawn app binary with tolerance override");

    assert!(
        output.status.success(),
        "tolerance override run failed: stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let verdict_bytes = std::fs::read(temp_dir.join("verdict.json")).expect("read verdict");
    let val: serde_json::Value =
        serde_json::from_slice(&verdict_bytes).expect("parse verdict JSON");
    assert_eq!(val["tolerance_overridden"], true);
    assert_eq!(val["tolerances"]["per_tick_relative"], 0.0005);
    assert_eq!(val["tolerances"]["cumulative_relative"], 0.0005);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("tolerance_overridden=true"),
        "stdout summary line must record tolerance_overridden=true: {stdout}"
    );

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn legacy_flag_based_invocations_continue_working() {
    let output = app_binary()
        .arg("--characterize-v0")
        .arg("4")
        .output()
        .expect("spawn app binary with legacy flags");

    assert!(
        output.status.success(),
        "legacy flag invocation failed: stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("characterization-trace"));
}
