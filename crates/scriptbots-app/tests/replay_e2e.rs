//! Mock-free terminal → FrankenSQLite → CSV export → replay E2E (bd-2z0.8.9.8).
//!
//! Drives the shipped terminal application headlessly with a fixed seed into a real
//! file-backed run database, then verifies the artifact, the CSV export boundary, and
//! replay/digest behavior with zero mocks: baseline verification must succeed and a
//! perturbed candidate must report its first divergence.

use scriptbots_storage::StorageReader;
use std::{
    env,
    ffi::OsString,
    fs,
    path::Path,
    process::{Command, Output},
};
use tempfile::tempdir;

const SEED: u64 = 42;
const BOOTSTRAP_TICKS: u64 = 24;

fn clear_scriptbots_environment(command: &mut Command, names: impl IntoIterator<Item = OsString>) {
    for name in names {
        let encoded = name.as_encoded_bytes();
        if encoded.starts_with(b"SCRIPTBOTS_") || encoded.starts_with(b"SB_") {
            command.env_remove(name);
        }
    }
}

fn base_command(bin: &str) -> Command {
    let mut cmd = Command::new(bin);
    clear_scriptbots_environment(&mut cmd, env::vars_os().map(|(name, _)| name));
    cmd.env("SCRIPTBOTS_MODE", "terminal")
        .env("SCRIPTBOTS_CONTROL_REST_ENABLED", "0")
        .env("SCRIPTBOTS_CONTROL_MCP", "disabled")
        .env("TERM", "xterm-256color")
        .env("RUST_LOG", "info")
        .env("RUST_LOG_STYLE", "never");
    cmd
}

/// Produce one real file-backed run through the shipped terminal application.
fn produce_run(database: &Path, extra_set: &[String]) -> Output {
    let mut cmd = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    cmd.env("SCRIPTBOTS_TERMINAL_HEADLESS", "1")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", "2")
        .env("SCRIPTBOTS_STORAGE_PATH", database);
    cmd.arg("--storage")
        .arg("file")
        .arg("--threads")
        .arg("1")
        .arg("--bootstrap-ticks")
        .arg(BOOTSTRAP_TICKS.to_string())
        .arg("--set")
        .arg(format!("rng_seed={SEED}"))
        .arg("--set")
        .arg("persistence_interval=1");
    for update in extra_set {
        cmd.arg("--set").arg(update);
    }
    cmd.output().expect("failed to run scriptbots-app binary")
}

fn export_csv(database: &Path, kind: &str, out: &Path) -> Output {
    let mut cmd = base_command(env!("CARGO_BIN_EXE_control_cli"));
    cmd.arg("export")
        .arg(kind)
        .arg("--db")
        .arg(database)
        .arg("--last")
        .arg("4096")
        .arg("--out")
        .arg(out);
    cmd.output().expect("failed to run control_cli export")
}

fn stderr_text(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout_text(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn strip_ansi(input: &str) -> String {
    let mut cleaned = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
        }
        cleaned.push(ch);
    }
    cleaned
}

fn csv_lines(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .lines()
        .map(str::to_owned)
        .collect()
}

#[test]
#[ignore = "pending bd-2z0.8.9.8 production replay instrumentation (core emission + verify plumb)"]
fn mock_free_terminal_to_sqlite_export_and_replay_e2e() {
    let temp_dir = tempdir().expect("temp run directory");
    let baseline_db = temp_dir.path().join("baseline.sqlite");
    let perturbed_db = temp_dir.path().join("perturbed.sqlite");

    // ------------------------------------------------------------------
    // Phase 1: produce the baseline run through the shipped terminal app.
    // ------------------------------------------------------------------
    let produced = produce_run(&baseline_db, &[]);
    assert!(
        produced.status.success(),
        "baseline terminal run failed: {}",
        stderr_text(&produced)
    );
    let produced_logs = strip_ansi(&format!(
        "{}{}",
        stdout_text(&produced),
        stderr_text(&produced)
    ));
    assert!(
        produced_logs.contains("Selected unique FrankenSQLite run database")
            && produced_logs.contains(&baseline_db.display().to_string()),
        "production logs must name the database path: {produced_logs}"
    );
    assert!(
        produced_logs.contains("shut down with an explicit persistence receipt"),
        "production logs must name the durability receipt: {produced_logs}"
    );

    // ------------------------------------------------------------------
    // Phase 2: the fresh artifact holds nonzero ticks, metrics, agents,
    // lifecycle rows, and replay evidence.
    // ------------------------------------------------------------------
    let db_display = baseline_db.display().to_string();
    let reader = StorageReader::open(&db_display).expect("baseline database opens");
    let max_tick = reader.max_tick().expect("max tick query");
    assert!(
        max_tick.unwrap_or(0) >= BOOTSTRAP_TICKS,
        "expected at least {BOOTSTRAP_TICKS} committed ticks, got {max_tick:?}"
    );
    let ledger = reader.run_ledger_summary().expect("ledger summary");
    assert!(ledger.tick_count >= BOOTSTRAP_TICKS, "ledger tick rows");
    let metrics = reader.recent_metrics(8).expect("metrics query");
    assert!(!metrics.is_empty(), "expected nonzero metric rows");
    let predators = reader.top_predators(4).expect("agents query");
    assert!(!predators.is_empty(), "expected nonzero agent rows");
    let births = reader.load_ancestry_births().expect("births query");
    assert!(!births.is_empty(), "expected nonzero lifecycle rows");
    let recorded_replay = reader.load_replay_events().expect("replay query");
    assert!(
        !recorded_replay.is_empty(),
        "expected nonempty production replay evidence in the artifact"
    );
    reader.close().expect("reader closes");

    // ------------------------------------------------------------------
    // Phase 3: CSV export boundary — headers, order, and counts.
    // ------------------------------------------------------------------
    let metrics_csv = temp_dir.path().join("metrics.csv");
    let ticks_csv = temp_dir.path().join("ticks.csv");
    let metrics_export = export_csv(&baseline_db, "metrics", &metrics_csv);
    assert!(
        metrics_export.status.success(),
        "metrics export failed: {}",
        stderr_text(&metrics_export)
    );
    let ticks_export = export_csv(&baseline_db, "ticks", &ticks_csv);
    assert!(
        ticks_export.status.success(),
        "ticks export failed: {}",
        stderr_text(&ticks_export)
    );

    let metric_rows = csv_lines(&metrics_csv);
    assert_eq!(metric_rows[0], "tick,name,value", "metrics CSV header");
    assert!(metric_rows.len() > 1, "metrics CSV must carry rows");
    let mut previous: Option<(u64, String)> = None;
    for row in &metric_rows[1..] {
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields.len(), 3, "metrics CSV row shape: {row}");
        let tick: u64 = fields[0].parse().expect("metrics tick is an integer");
        fields[2]
            .parse::<f64>()
            .expect("metrics value is a fixed-precision float");
        if let Some((prev_tick, prev_name)) = previous {
            assert!(
                tick < prev_tick || (tick == prev_tick && fields[1] <= prev_name.as_str()),
                "metrics CSV must stay ordered tick DESC, name DESC: {row}"
            );
        }
        previous = Some((tick, fields[1].to_owned()));
    }

    let tick_rows = csv_lines(&ticks_csv);
    assert_eq!(
        tick_rows[0],
        "tick,epoch,closed,agent_count,births,deaths,total_energy,average_energy,average_health",
        "ticks CSV header"
    );
    assert!(tick_rows.len() > 1, "ticks CSV must carry rows");
    let mut previous_tick: Option<u64> = None;
    for row in &tick_rows[1..] {
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields.len(), 9, "ticks CSV row shape: {row}");
        let tick: u64 = fields[0].parse().expect("ticks tick is an integer");
        if let Some(prev) = previous_tick {
            assert!(tick < prev, "ticks CSV must stay ordered tick DESC: {row}");
        }
        previous_tick = Some(tick);
    }

    // ------------------------------------------------------------------
    // Phase 4: replay verification succeeds for the exact baseline config.
    // ------------------------------------------------------------------
    let mut verify = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    verify
        .arg("--replay-db")
        .arg(&baseline_db)
        .arg("--threads")
        .arg("1")
        .arg("--set")
        .arg(format!("rng_seed={SEED}"))
        .arg("--set")
        .arg("persistence_interval=1");
    let verified = verify.output().expect("failed to run replay verification");
    let verify_out = strip_ansi(&format!(
        "{}{}",
        stdout_text(&verified),
        stderr_text(&verified)
    ));
    assert!(
        verified.status.success(),
        "baseline replay verification must succeed: {verify_out}"
    );
    assert!(
        verify_out.contains("Replay matched"),
        "baseline verification must report the matched stream: {verify_out}"
    );
    assert!(
        verify_out.contains(&format!("{SEED}")),
        "replay logs must name the seed: {verify_out}"
    );

    // ------------------------------------------------------------------
    // Phase 5: a perturbed candidate reports its first divergence.
    // ------------------------------------------------------------------
    let perturbed = produce_run(
        &perturbed_db,
        &["food_max=0.05".to_owned(), "food_respawn_amount=0.01".to_owned()],
    );
    assert!(
        perturbed.status.success(),
        "perturbed terminal run failed: {}",
        stderr_text(&perturbed)
    );

    let mut compare = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    compare
        .arg("--replay-db")
        .arg(&baseline_db)
        .arg("--compare-db")
        .arg(&perturbed_db)
        .arg("--threads")
        .arg("1")
        .arg("--set")
        .arg(format!("rng_seed={SEED}"))
        .arg("--set")
        .arg("persistence_interval=1");
    let compared = compare.output().expect("failed to run replay comparison");
    let compare_out = strip_ansi(&format!(
        "{}{}",
        stdout_text(&compared),
        stderr_text(&compared)
    ));
    assert!(
        !compared.status.success(),
        "a perturbed candidate must fail replay comparison: {compare_out}"
    );
    assert!(
        compare_out.contains("baseline") && compare_out.contains("candidate"),
        "comparison logs must name both databases/roles: {compare_out}"
    );
    assert!(
        compare_out.contains("mismatch") || compare_out.contains("divergence"),
        "comparison must report the first divergence: {compare_out}"
    );
}
