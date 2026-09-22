//! Mock-free terminal → FrankenSQLite → CSV export → replay E2E (bd-2z0.8.9.8).
//!
//! Drives the shipped terminal application headlessly with a fixed seed into a real
//! file-backed run database, then verifies the artifact, the CSV export boundary, and
//! replay/digest behavior with zero mocks: baseline verification must succeed and a
//! perturbed candidate must report its first divergence.

use scriptbots_storage::{Connection, RowExt, StorageReader};
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
        .arg("persistence_interval=1")
        .arg("--set")
        .arg("replay_event_tick_cap=65536");
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
        if ch == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }
        cleaned.push(ch);
    }
    cleaned
}

fn csv_lines(path: &Path) -> std::io::Result<Vec<String>> {
    let contents = fs::read_to_string(path)?;
    Ok(contents.lines().map(str::to_owned).collect())
}

#[test]
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

    let metric_rows = csv_lines(&metrics_csv).expect("read exported metrics CSV");
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
                tick > prev_tick || (tick == prev_tick && fields[1] >= prev_name.as_str()),
                "metrics CSV must stay chronological (tick ASC, name ASC): {row}"
            );
        }
        previous = Some((tick, fields[1].to_owned()));
    }

    let tick_rows = csv_lines(&ticks_csv).expect("read exported ticks CSV");
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
            assert!(
                tick > prev,
                "ticks CSV must stay chronological (tick ASC): {row}"
            );
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
        .arg("persistence_interval=1")
        .arg("--set")
        .arg("replay_event_tick_cap=65536");
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
        &[
            "food_max=0.05".to_owned(),
            "food_respawn_amount=0.01".to_owned(),
        ],
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
        .arg("persistence_interval=1")
        .arg("--set")
        .arg("replay_event_tick_cap=65536");
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

#[test]
fn mock_free_checkpoint_start_replay_e2e() {
    let temp_dir = tempdir().expect("temp run directory");
    let database = temp_dir.path().join("checkpoint_run.sqlite");
    let db_str = database.display().to_string();

    // 1. Produce a real file-backed baseline run.
    let produced = produce_run(&database, &[]);
    assert!(
        produced.status.success(),
        "baseline terminal run failed: {}",
        stderr_text(&produced)
    );

    // 2. Step a matching world headlessly to tick 12 and capture a valid WorldCheckpointV1.
    let config = scriptbots_core::ScriptBotsConfig {
        rng_seed: Some(SEED),
        persistence_interval: 0,
        replay_event_tick_cap: 65536,
        history_capacity: 600,
        ..scriptbots_core::ScriptBotsConfig::default()
    };
    let mut world = scriptbots_core::WorldState::new(config).expect("build world");
    let brain_keys = scriptbots_app::install_brains(&mut world, scriptbots_app::BrainPreset::Mixed)
        .expect("install brains")
        .population;
    scriptbots_app::seed_founding_population(&mut world, &brain_keys).expect("seed founders");
    for _ in 0..12 {
        world.step().expect("step world");
    }
    let checkpoint = world.checkpoint_v1().expect("capture tick 12 checkpoint");
    assert_eq!(checkpoint.tick().0, 12);

    // 3. Persist the checkpoint into the existing SQLite database using Connection.
    let encoded = checkpoint.encode().expect("encode checkpoint");
    let payload = encoded
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let payload_digest = format!("blake3:{}", blake3::hash(&encoded).to_hex());
    let schema_format = format!(
        "{}+postcard_hex",
        scriptbots_core::WORLD_CHECKPOINT_V1_SCHEMA
    );

    let connection = Connection::open(&db_str).expect("open connection");
    let run_id: String = connection
        .query_row("SELECT run_id FROM runs LIMIT 1")
        .expect("query run_id")
        .get_typed(0)
        .expect("get run_id");
    connection
        .execute_with_params(
            "INSERT INTO checkpoints (run_id, checkpoint_id, tick, checkpoint_ordinal, format, payload, payload_digest, metadata_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            &[
                run_id.into(),
                "cp-tick-12".into(),
                12i64.into(),
                0i64.into(),
                schema_format.into(),
                payload.into(),
                payload_digest.into(),
                "{}".into(),
            ],
        )
        .expect("insert checkpoint");
    connection.close().expect("close connection");

    // 4. Verify replay WITHOUT --checkpoint-start: runs from tick 0.
    let mut verify_full = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    verify_full
        .arg("--replay-db")
        .arg(&database)
        .arg("--threads")
        .arg("1")
        .arg("--set")
        .arg(format!("rng_seed={SEED}"))
        .arg("--set")
        .arg("persistence_interval=1")
        .arg("--set")
        .arg("replay_event_tick_cap=65536");
    let full_out = verify_full.output().expect("run full replay");
    let full_text = strip_ansi(&format!(
        "{}{}",
        stdout_text(&full_out),
        stderr_text(&full_out)
    ));
    assert!(full_out.status.success(), "full replay failed: {full_text}");
    assert!(
        full_text.contains("Replay matched"),
        "full replay must match: {full_text}"
    );

    // 5. Verify replay WITH --checkpoint-start: resumes from tick 12.
    let mut verify_cp = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    verify_cp
        .arg("--replay-db")
        .arg(&database)
        .arg("--checkpoint-start")
        .arg("--threads")
        .arg("1")
        .arg("--set")
        .arg(format!("rng_seed={SEED}"))
        .arg("--set")
        .arg("persistence_interval=1")
        .arg("--set")
        .arg("replay_event_tick_cap=65536");
    let cp_out = verify_cp.output().expect("run checkpoint-start replay");
    let cp_text = strip_ansi(&format!("{}{}", stdout_text(&cp_out), stderr_text(&cp_out)));
    assert!(
        cp_out.status.success(),
        "checkpoint replay failed: {cp_text}"
    );
    assert!(
        cp_text.contains("from tick 12"),
        "checkpoint replay must state starting tick 12: {cp_text}"
    );
    assert!(
        cp_text.contains("Replay matched"),
        "checkpoint replay must match: {cp_text}"
    );

    // 6. Negative control: corrupt the checkpoint payload in the database.
    // Replay WITH --checkpoint-start must fail closed rather than falling back to tick zero.
    let connection = Connection::open(&db_str).expect("open connection for corruption");
    connection
        .execute("UPDATE checkpoints SET payload = 'aabbccdd'")
        .expect("corrupt payload");
    connection.close().expect("close connection");

    let mut verify_corrupt = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    verify_corrupt
        .arg("--replay-db")
        .arg(&database)
        .arg("--checkpoint-start")
        .arg("--threads")
        .arg("1")
        .arg("--set")
        .arg(format!("rng_seed={SEED}"))
        .arg("--set")
        .arg("persistence_interval=1")
        .arg("--set")
        .arg("replay_event_tick_cap=65536");
    let corrupt_out = verify_corrupt
        .output()
        .expect("run corrupt checkpoint replay");
    let corrupt_text = strip_ansi(&format!(
        "{}{}",
        stdout_text(&corrupt_out),
        stderr_text(&corrupt_out)
    ));
    assert!(
        !corrupt_out.status.success(),
        "corrupt checkpoint must fail closed, but succeeded: {corrupt_text}"
    );
    assert!(
        corrupt_text.contains("corrupt")
            || corrupt_text.contains("payload")
            || corrupt_text.contains("failed")
            || corrupt_text.contains("error"),
        "expected corruption diagnostic, got: {corrupt_text}"
    );
}
