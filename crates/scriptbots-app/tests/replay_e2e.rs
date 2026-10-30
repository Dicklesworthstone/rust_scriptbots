//! Mock-free terminal → FrankenSQLite → CSV export → replay E2E (bd-2z0.8.9.8).
//!
//! Drives the shipped terminal application headlessly with a fixed seed into a real
//! file-backed run database, then verifies the artifact, the CSV export boundary, and
//! replay/digest behavior with zero mocks: baseline verification must succeed and a
//! perturbed candidate must report its first divergence.

use scriptbots_storage::{Connection, OpenFlags, PersistedCheckpointRecord, RowExt, StorageReader};
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

const HOST_CONTINUATION_TICKS: u64 = 18;
const HOST_HANDOFF_TICK: u64 = 7;
const HOST_MINIMUM_POPULATION: usize = 20;
const HOST_THREAD_BUDGET: usize = 1;
const HOST_INHERITED_RAYON_THREADS: &str = "8";

/// Keep the subprocess streams beside the closed databases, including failed invocations.
/// DSR uses a retained TMPDIR; RCH can select an external SCRIPTBOTS_TEST_ARTIFACT_ROOT.
fn retained_host_output(directory: &Path, name: &str, command: &mut Command) -> Output {
    let stdout_path = directory.join(format!("{name}.stdout"));
    let stderr_path = directory.join(format!("{name}.stderr"));
    let stdout = fs::File::options()
        .write(true)
        .create_new(true)
        .open(&stdout_path)
        .expect("create unique retained stdout");
    let stderr = fs::File::options()
        .write(true)
        .create_new(true)
        .open(&stderr_path)
        .expect("create unique retained stderr");
    let arguments = command
        .get_args()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    command
        .env("RAYON_NUM_THREADS", HOST_INHERITED_RAYON_THREADS)
        .env("RUST_LOG", "info,scriptbots_core::parallelism=debug")
        .env("NO_COLOR", "1")
        .env(
            "RCH_BUILD_SOURCE",
            "runtime-source-must-not-replace-embedded-identity",
        )
        .current_dir(directory)
        .stdout(stdout)
        .stderr(stderr);
    let status = command.status().expect("run actual host checkpoint CLI");
    fs::write(
        directory.join(format!("{name}.process.json")),
        serde_json::to_vec_pretty(&serde_json::json!({
            "executable": command.get_program().to_string_lossy(),
            "arguments": arguments,
            "exit_code": status.code(),
            "success": status.success(),
        }))
        .expect("encode observed process disposition"),
    )
    .expect("retain process disposition");
    Output {
        status,
        stdout: fs::read(stdout_path).expect("read actual retained stdout"),
        stderr: fs::read(stderr_path).expect("read actual retained stderr"),
    }
}

fn produce_host_handoff(
    directory: &Path,
    name: &str,
    database: &Path,
    ticks: u64,
    checkpoint_id: &str,
    food_growth_rate: &str,
    brain: &str,
) -> Output {
    let mut command = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    command
        .env("SCRIPTBOTS_STORAGE_PATH", database)
        .args(["--storage", "file", "--threads"])
        .arg(HOST_THREAD_BUDGET.to_string())
        .args(["--brain", brain])
        .arg("--bootstrap-ticks")
        .arg(ticks.to_string())
        .arg("--checkpoint-handoff")
        .arg(checkpoint_id)
        .arg("--set")
        .arg(format!("rng_seed={SEED}"))
        .args([
            "--set",
            "persistence_interval=1",
            "--set",
            "replay_event_tick_cap=65536",
        ])
        .arg("--set")
        .arg(format!("population_minimum={HOST_MINIMUM_POPULATION}"))
        .arg("--set")
        .arg(format!("food_growth_rate={food_growth_rate}"));
    if brain == "mixed" {
        command.args(["--set", "neuroflow.enabled=true"]);
    }
    let output = retained_host_output(directory, name, &mut command);
    assert_host_science_pool(&output, ticks);
    output
}

fn continue_host_handoff(
    directory: &Path,
    name: &str,
    database: &Path,
    checkpoint_id: &str,
    ticks: u64,
    brain: &str,
) -> Output {
    let mut command = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    command
        .env_remove("SCRIPTBOTS_MODE")
        .arg("--continue-host-db")
        .arg(database)
        .arg("--host-checkpoint-id")
        .arg(checkpoint_id)
        .arg("--continue-ticks")
        .arg(ticks.to_string())
        .args(["--brain", brain]);
    let output = retained_host_output(directory, name, &mut command);
    assert_host_science_pool(&output, ticks);
    output
}

/// Observe the actual production step context, including steps after restoration.
/// Failure paths retain their existing exit and unchanged-prefix assertions below.
fn assert_host_science_pool(output: &Output, ticks: u64) {
    if !output.status.success() {
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let widths = stderr
        .lines()
        .filter(|line| line.contains("observed science-step Rayon pool"))
        .map(|line| {
            line.split_once("rayon_threads=")
                .expect("observed science pool field")
                .1
                .split_whitespace()
                .next()
                .expect("actual width")
                .parse::<usize>()
                .expect("numeric actual science pool width")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        u64::try_from(widths.len()).expect("bounded observations"),
        ticks,
        "each actual fresh/continued science transition must be observed"
    );
    assert!(
        widths.iter().all(|width| *width == HOST_THREAD_BUDGET),
        "actual fresh/continued science widths {widths:?} must retain budget {HOST_THREAD_BUDGET}"
    );
}

fn compare_host_handoffs(
    directory: &Path,
    name: &str,
    expected_database: &Path,
    expected_id: &str,
    actual_database: &Path,
    actual_id: &str,
) -> Output {
    let mut command = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    command
        .env_remove("SCRIPTBOTS_MODE")
        .arg("--compare-host-db")
        .arg(expected_database)
        .arg("--compare-host-checkpoint-id")
        .arg(expected_id)
        .arg("--host-candidate-db")
        .arg(actual_database)
        .arg("--host-candidate-checkpoint-id")
        .arg(actual_id);
    retained_host_output(directory, name, &mut command)
}

fn host_cli_report(output: &Output, schema: &str) -> serde_json::Value {
    let reports = output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter_map(|line| serde_json::from_slice::<serde_json::Value>(line).ok())
        .filter(|report| report["schema"].as_str() == Some(schema))
        .collect::<Vec<_>>();
    assert_eq!(
        reports.len(),
        1,
        "expected one actual {schema} report; stdout={} stderr={}",
        stdout_text(output),
        stderr_text(output)
    );
    reports.into_iter().next().expect("single observed report")
}

fn assert_host_comparison_source(source: &serde_json::Value, handoff: &serde_json::Value) {
    for field in [
        "artifact_path",
        "run_id",
        "session_id",
        "checkpoint_id",
        "checkpoint_tick",
        "payload_digest",
        "source_commit",
        "root_seed",
        "brain_roster",
        "watermarks",
        "journal_cut",
    ] {
        assert_eq!(
            source[field], handoff[field],
            "production comparison must bind actual {field} provenance"
        );
    }
}

fn host_watermarks_json(
    watermarks: scriptbots_storage::PersistenceWatermarks,
) -> serde_json::Value {
    serde_json::json!({
        "admitted": watermarks.admitted.map(|id| id.get()),
        "applied": watermarks.applied.map(|id| id.get()),
        "durable": watermarks.durable.map(|id| id.get()),
    })
}

fn copy_closed_host_database(source: &Path, destination: &Path) {
    let reader = StorageReader::open_finished(source.to_str().expect("UTF-8 source database"))
        .expect("hold the actual finished source lease");
    let copied = reader
        .materialize_finished_database(source, destination)
        .expect("copy the closed image with its own destination lease");
    copied.close().expect("close the leased destination reader");
    reader.close().expect("close the leased source reader");
}

/// Observe science and journal prefixes independently of the CLI's presentation.
fn host_database_prefix(database: &Path) -> serde_json::Value {
    let reader = StorageReader::open(database.to_str().expect("UTF-8 host database"))
        .expect("open actual host database");
    let ticks = reader.summary_ticks().expect("read actual science ticks");
    let watermarks = reader
        .persistence_watermarks()
        .expect("read actual three science watermarks");
    let replay = reader
        .load_replay_events()
        .expect("read actual replay rows");
    reader.close().expect("close host science reader");
    let connection =
        scriptbots_storage::open_with_flags(database, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .expect("open read-only FrankenSQLite journal connection");
    let prefixes = connection
        .query(
            "SELECT host_session_id, admitted_journal_prefix, applied_journal_prefix,
                    committed_volatile_journal_prefix, durable_journal_prefix,
                    admitted_event_prefix, applied_event_prefix,
                    committed_volatile_event_prefix, durable_event_prefix, shutdown_sequence
             FROM host_journal_progress ORDER BY host_session_id",
        )
        .expect("read actual host journal prefixes")
        .iter()
        .map(|row| {
            let fields = (0..9)
                .map(|index| {
                    row.get_typed::<String>(index)
                        .expect("journal prefix field")
                })
                .collect::<Vec<_>>();
            let shutdown: Option<String> = row.get_typed(9).expect("shutdown sequence field");
            serde_json::json!({"prefixes": fields, "shutdown_sequence": shutdown})
        })
        .collect::<Vec<_>>();
    let tables = [
        "tick_summaries",
        "agents",
        "metrics",
        "replay_events",
        "host_journal_archive",
        "host_journal_batch_ledger",
        "host_command_records",
        "checkpoints",
    ];
    let mut row_counts = serde_json::Map::new();
    for table in tables {
        let count: i64 = connection
            .query_row(&format!("SELECT COUNT(*) FROM {table}"))
            .expect("read actual retained row count")
            .get_typed(0)
            .expect("row count field");
        row_counts.insert(table.to_owned(), count.into());
    }
    connection
        .close_without_checkpoint()
        .expect("close read-only journal connection");
    serde_json::json!({
        "ticks": ticks, "watermarks": host_watermarks_json(watermarks),
        "replay": replay, "journal": prefixes, "row_counts": row_counts,
    })
}

fn assert_host_handoff_readback(
    database: &Path,
    report: &serde_json::Value,
    tick: u64,
) -> PersistedCheckpointRecord {
    assert_eq!(report["status"], "handed_off");
    assert_eq!(report["checkpoint_tick"], tick);
    assert_eq!(
        report["artifact_path"],
        database.to_str().expect("UTF-8 path")
    );
    assert_eq!(report["root_seed"], SEED);
    let reader = StorageReader::open(database.to_str().expect("UTF-8 path"))
        .expect("open independent host artifact reader");
    let record = reader
        .load_checkpoints()
        .expect("validate actual retained checkpoint rows")
        .into_iter()
        .find(|record| Some(record.checkpoint_id.as_str()) == report["checkpoint_id"].as_str())
        .expect("reported full host artifact exists");
    let checkpoint = record.host_checkpoint().expect("decode full host artifact");
    assert_eq!(checkpoint.tick().0, tick);
    assert_eq!(report["session_id"], checkpoint.session_id().get());
    assert_eq!(
        report["checkpoint_schema"],
        scriptbots_runtime::HOST_CHECKPOINT_V1_SCHEMA
    );
    assert_eq!(report["checkpoint_format"], record.format);
    assert_eq!(report["payload_digest"], record.payload_digest);
    let encoded = checkpoint.encode().expect("encode validated retained host");
    assert_eq!(
        record.payload,
        encoded
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    assert_eq!(
        record.payload_digest,
        format!("blake3:{}", blake3::hash(&encoded).to_hex())
    );
    let digest = checkpoint.world_checkpoint().source_digest();
    digest
        .validate_contract()
        .expect("complete science digest contract");
    assert!(digest.evaluator_state_covered && digest.factory_state_covered);
    let limitations: scriptbots_app::CharacterizationLimitationsV0 =
        serde_json::from_value(report["characterization_limitations"].clone())
            .expect("typed artifact-scoped continuation claim");
    assert_eq!(
        limitations.purpose,
        "retained_full_host_checkpoint_continuation"
    );
    assert!(limitations.evaluator_state_covered);
    assert!(limitations.rng_state_restorable);
    assert!(limitations.checkpoint_replay_guarantee);
    assert!(report["continuation_refusal"].is_null());
    assert_eq!(
        report["boundary_digest"],
        serde_json::to_value(digest).expect("digest JSON")
    );
    assert_eq!(
        report["config"],
        serde_json::to_value(checkpoint.config()).expect("config JSON")
    );
    let traces = checkpoint.step_traces();
    assert_eq!(
        traces.iter().map(|trace| trace.tick.0).collect::<Vec<_>>(),
        (1..=tick).collect::<Vec<_>>(),
        "the actual owner must retain every measured transition"
    );
    for trace in traces {
        trace
            .validate_contract()
            .expect("actual six-point trace contract");
        for stage in &trace.stages {
            stage
                .capture
                .as_ref()
                .expect("actual complete stage observation");
        }
    }
    assert_eq!(
        report["step_traces"],
        serde_json::to_value(traces).expect("trace JSON")
    );
    let requirements = checkpoint.world_checkpoint().required_brain_registry();
    assert_eq!(requirements.entries.len(), 1, "one actual MLP family");
    assert_eq!(
        requirements.entries[0].kind,
        scriptbots_brain::MlpBrain::KIND.as_str()
    );
    let manifest = reader.run_manifest().expect("actual stored provenance");
    let launch: scriptbots_app::RunManifestV3 = serde_json::from_str(&manifest.manifest_json)
        .expect("independent immutable launch provenance");
    assert_eq!(
        launch.build.rayon_num_threads.as_deref(),
        Some(HOST_INHERITED_RAYON_THREADS)
    );
    assert_eq!(
        launch
            .thread_policy
            .as_ref()
            .and_then(|policy| policy.threads),
        Some(HOST_THREAD_BUDGET)
    );
    let built = scriptbots_app::BuildProvenanceV0::current();
    assert_eq!(
        manifest.source_revision, built.source_revision,
        "stored source must retain the embedded identity despite conflicting runtime input"
    );
    assert_eq!(launch.build.source_revision, built.source_revision);
    assert!(!launch.limitations.checkpoint_replay_guarantee);
    assert_eq!(report["run_id"], manifest.run_id.to_string());
    assert_eq!(manifest.root_seed, SEED);
    assert_eq!(report["brain_roster"], manifest.brain_roster_json);
    assert_eq!(
        report["source_commit"],
        serde_json::json!(manifest.source_revision)
    );
    let ticks = reader.summary_ticks().expect("actual persisted tick rows");
    assert_eq!(ticks, (1..=tick).collect::<Vec<_>>());
    assert_eq!(report["persisted_summary_ticks"], serde_json::json!(ticks));
    assert!(
        reader.recent_ticks(1).expect("actual final population")[0].agent_count
            >= HOST_MINIMUM_POPULATION,
        "nonempty production MLP population must survive the tested boundary"
    );
    let watermarks = reader
        .persistence_watermarks()
        .expect("actual persistence prefixes");
    assert!(watermarks.admitted.is_some());
    assert_eq!(watermarks.admitted, watermarks.applied);
    assert_eq!(watermarks.applied, watermarks.durable);
    assert_eq!(report["watermarks"], host_watermarks_json(watermarks));
    let metadata: serde_json::Value =
        serde_json::from_str(&record.metadata_json).expect("actual artifact metadata");
    for field in ["admitted", "applied", "durable"] {
        assert_eq!(
            metadata["host_storage_cut"][field],
            report["watermarks"][field]
        );
    }
    assert_eq!(metadata["host_storage_cut"]["last_admitted_tick"], tick);
    assert_eq!(metadata["host_storage_cut"]["run_id"], report["run_id"]);
    let final_batch = reader
        .batch_status(watermarks.durable.expect("durable prefix"))
        .expect("actual final batch")
        .expect("durable ledger row exists");
    assert_eq!(final_batch.tick, tick);
    assert_eq!(
        final_batch.state,
        scriptbots_storage::BatchPersistenceState::Durable
    );
    assert_eq!(
        checkpoint.persistence_last_admitted_tick(),
        Some(scriptbots_core::Tick(tick))
    );
    assert_eq!(
        reader.host_journal_sessions().expect("actual sessions"),
        vec![checkpoint.session_id()]
    );
    reader.close().expect("close artifact reader");
    let prefix = host_database_prefix(database);
    assert!(
        !prefix["replay"]
            .as_array()
            .expect("replay array")
            .is_empty()
    );
    let journal = &prefix["journal"].as_array().expect("journal array")[0];
    assert!(
        journal["shutdown_sequence"].is_null(),
        "handoff must not fabricate shutdown"
    );
    let cut = checkpoint.journal_cut();
    assert_eq!(
        report["journal_cut"],
        serde_json::to_value(cut).expect("actual cut JSON")
    );
    let fields = journal["prefixes"].as_array().expect("prefix fields");
    assert_eq!(fields[0], format!("{:016x}", cut.session_id.get()));
    for field in &fields[1..5] {
        assert_eq!(*field, format!("{:016x}", cut.last_journal_sequence));
    }
    for field in &fields[5..9] {
        assert_eq!(
            *field,
            format!("{:016x}", cut.last_scientific_event_sequence)
        );
    }
    record
}

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

/// Export a run's journal domain events and check the CSV against the storage layer's own
/// evidence: same row count, journal order, known kinds, JSON payloads. Returns the number of
/// exported rows as (reproductive births with origin "born", all births, deaths, combat).
fn verify_domain_event_export(database: &Path, out: &Path) -> (u64, u64, u64, u64) {
    let export = export_csv(database, "domain-events", out);
    assert!(
        export.status.success(),
        "domain-events export failed: {}",
        stderr_text(&export)
    );
    let finished =
        scriptbots_storage::StorageReader::open_finished(database.to_str().expect("utf-8 path"))
            .expect("finished reader opens the run");
    let mut durable_events = 0_u64;
    for session in finished.host_journal_sessions().expect("journal sessions") {
        durable_events += finished
            .domain_event_evidence(
                session,
                scriptbots_storage::DomainEventExpectation::AllowEmpty,
            )
            .expect("domain-event evidence")
            .domain_event_count;
    }
    finished.close().expect("finished reader closes");
    let mut reader = csv::Reader::from_path(out).expect("read domain-events CSV");
    assert_eq!(
        reader.headers().expect("domain-events header"),
        vec![
            "session_id",
            "scientific_event_sequence",
            "event_ordinal",
            "tick",
            "kind",
            "agent_uid",
            "payload_json",
        ],
        "domain-events CSV header"
    );
    let (mut born, mut births, mut deaths, mut combat) = (0_u64, 0_u64, 0_u64, 0_u64);
    let mut previous_key: Option<(u64, u64, u64)> = None;
    for record in reader.records() {
        let record = record.expect("domain-events CSV row");
        let key: (u64, u64, u64) = (
            record[0].parse().expect("session id"),
            record[1].parse().expect("scientific sequence"),
            record[2].parse().expect("event ordinal"),
        );
        if let Some(previous) = previous_key {
            assert!(
                key > previous,
                "domain events must stay in journal order: {record:?}"
            );
        }
        previous_key = Some(key);
        record[3].parse::<u64>().expect("tick");
        match &record[4] {
            "birth" => births += 1,
            "death" => deaths += 1,
            "combat" => combat += 1,
            other => panic!("unknown domain-event kind {other:?}: {record:?}"),
        }
        assert_eq!(
            record[5].is_empty(),
            &record[4] == "combat",
            "only aggregate combat rows lack an agent uid: {record:?}"
        );
        let payload: serde_json::Value = serde_json::from_str(&record[6]).expect("payload is JSON");
        if &record[4] == "birth" && payload["origin"] == "born" {
            born += 1;
        }
    }
    assert_eq!(
        births + deaths + combat,
        durable_events,
        "the export must carry exactly the durable domain events"
    );
    (born, births, deaths, combat)
}

#[test]
fn domain_event_export_carries_a_meadow_runs_births() {
    // The meadow scenario's checked-in envelope requires at least 5 reproductive births
    // within 360 ticks on its declared seed 42, so this run cannot be vacuous.
    let temp_dir = tempdir().expect("temp run directory");
    let database = temp_dir.path().join("meadow.sqlite");
    let scenario =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scenarios/meadow.scenario.toml");
    let mut cmd = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    cmd.env("SCRIPTBOTS_TERMINAL_HEADLESS", "1")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", "2")
        .env("SCRIPTBOTS_STORAGE_PATH", &database)
        .arg("--storage")
        .arg("file")
        .arg("--threads")
        .arg("1")
        .arg("--scenario")
        .arg(&scenario)
        .arg("--bootstrap-ticks")
        .arg("360")
        .arg("--set")
        .arg("rng_seed=42");
    let produced = cmd.output().expect("run meadow");
    assert!(
        produced.status.success(),
        "meadow run failed: {}",
        stderr_text(&produced)
    );
    let (born, births, deaths, combat) =
        verify_domain_event_export(&database, &temp_dir.path().join("meadow_events.csv"));
    assert!(
        born >= 5,
        "meadow's envelope promises >= 5 reproductive births by tick 360; exported {born} \
         (births {births}, deaths {deaths}, combat {combat})"
    );
}

fn stderr_text(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout_text(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Live owner publication, durable typed rows and offline CLI must tell the same story.
#[test]
fn curated_headless_narrative_seeds_persist_real_events_and_full_info_tables() {
    use scriptbots_core::narrative::{EventKind, EventRecord};

    const BOOTSTRAP: u64 = 94;
    const FRAMES: u64 = 2;
    const TICKS: u64 = BOOTSTRAP + FRAMES;
    let artifact_root = env::var_os("SCRIPTBOTS_TEST_ARTIFACT_ROOT")
        .map_or_else(env::temp_dir, std::path::PathBuf::from);
    fs::create_dir_all(&artifact_root).expect("external narrative artifact root");
    let directory = tempfile::Builder::new()
        .prefix("headless-narrative-seeds-")
        .tempdir_in(artifact_root)
        .expect("unique retained narrative proof")
        .keep();
    eprintln!(
        "headless narrative evidence retained at {}",
        directory.display()
    );
    let cases: [(u64, &str, &str, usize, &[EventKind]); 4] = [
        (42, "quiet", "", 64, &[]),
        (
            137,
            "boom",
            "\n[[interventions]]\ntick = 40\n[interventions.set]\nclosed = false\npopulation_minimum = 48\n",
            64,
            &[EventKind::PopulationBoom],
        ),
        (
            20_260_717,
            "extinction",
            "\n[[interventions]]\ntick = 40\n[interventions.set]\nmetabolism_drain = 1.0\n",
            64,
            &[EventKind::PopulationCrash, EventKind::Extinction],
        ),
        (
            20_260_717,
            "extinction-clipped",
            "\n[[interventions]]\ntick = 40\n[interventions.set]\nmetabolism_drain = 1.0\n",
            1,
            &[EventKind::PopulationCrash, EventKind::Extinction],
        ),
    ];
    let mut observed_kinds = Vec::new();
    for (seed, label, intervention, capacity, required_kinds) in cases {
        let database = directory.join(format!("{label}.sqlite"));
        let scenario = directory.join(format!("{label}.scenario.toml"));
        let report_path = directory.join(format!("{label}.report.json"));
        // Freeze non-intervention ecology rather than relaxing detector policy.
        // Real MLP agents, real scheduled config commands and real owner ticks run below.
        fs::write(
            &scenario,
            format!(
                r#"schema = "scriptbots.scenario.v1"
schema_version = 1
id = "narrative-{label}"
[config]
rng_seed = {seed}
world_width = 120
world_height = 120
food_cell_size = 20
initial_food = 0.0
food_respawn_interval = 0
food_growth_rate = 0.0
food_intake_rate = 0.0
metabolism_drain = 0.0
metabolism_ramp_rate = 0.0
metabolism_boost_penalty = 0.0
movement_drain = 0.0
temperature_discomfort_rate = 0.0
topography_enabled = false
bot_speed = 0.0
spike_damage = 0.0
spike_energy_cost = 0.0
food_sharing_rate = 0.0
food_transfer_rate = 0.0
reproduction_energy_threshold = 0.0
aging_health_decay_rate = 0.0
closed = true
population_minimum = 0
population_spawn_interval = 0
chart_flush_interval = 0
history_capacity = 128
narrative_interval = 1
narrative_capacity = {capacity}
persistence_interval = 32
{intervention}"#
            ),
        )
        .expect("write actual scenario input");
        let mut command = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
        command
            .env("SCRIPTBOTS_TERMINAL_HEADLESS", "1")
            .env("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", FRAMES.to_string())
            .env("SCRIPTBOTS_TERMINAL_HEADLESS_REPORT", &report_path)
            .env("SCRIPTBOTS_STORAGE_PATH", &database)
            .args(["--storage", "file", "--threads", "1", "--brain", "mlp"])
            .arg("--scenario")
            .arg(&scenario)
            .arg("--bootstrap-ticks")
            .arg(BOOTSTRAP.to_string());
        let output = retained_host_output(&directory, label, &mut command);
        assert!(
            output.status.success(),
            "{label} seed {seed} failed: stdout={} stderr={}",
            stdout_text(&output),
            stderr_text(&output)
        );
        assert_host_science_pool(&output, TICKS);
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(&report_path).expect("actual headless report"))
                .expect("decode actual headless report");
        assert_eq!(report["summary"]["final_tick"], TICKS);
        let dropped = report["narrative_dropped_events"]
            .as_u64()
            .expect("actual dropped count");
        if capacity == 1 {
            assert!(
                dropped > 0,
                "the clipped control must actually evict records"
            );
        } else {
            assert_eq!(dropped, 0);
        }
        let mut published: Vec<EventRecord> =
            serde_json::from_value(report["narrative_events"].clone())
                .expect("complete typed owner narrative publication");
        let emitted_json = serde_json::to_string(&published).expect("published event table");
        let stderr = strip_ansi(&stderr_text(&output));
        let table_lines = stderr
            .lines()
            .filter(|line| line.contains("observed headless narrative ring"))
            .collect::<Vec<_>>();
        assert_eq!(table_lines.len(), 1, "one complete observed table per run");
        assert!(table_lines[0].contains("INFO"));
        assert!(table_lines[0].contains(&format!("events_json={emitted_json}")));
        if required_kinds.is_empty() {
            assert!(
                published.is_empty(),
                "quiet runs must have no synthetic event rows"
            );
        } else if dropped == 0 {
            for kind in required_kinds {
                assert!(
                    published.iter().any(|event| event.kind == *kind),
                    "{label} seed {seed} must actually emit {kind:?}: {emitted_json}"
                );
            }
        }
        published.sort_by(|a, b| {
            (a.tick.0, a.kind.as_str(), &a.metric).cmp(&(b.tick.0, b.kind.as_str(), &b.metric))
        });
        let mut kinds = published
            .iter()
            .map(|event| event.kind.as_str())
            .collect::<Vec<_>>();
        kinds.sort_unstable();
        kinds.dedup();
        if dropped == 0 {
            observed_kinds.push(kinds);
        }
        let before = fs::read(&database).expect("closed database preimage");
        let reader = StorageReader::open_finished(database.to_str().expect("UTF-8 path"))
            .expect("validate actual finished run");
        let manifest = reader.run_manifest().expect("actual persisted provenance");
        let config: scriptbots_core::ScriptBotsConfig =
            serde_json::from_str(&manifest.normalized_config_json).expect("recorded config");
        assert_eq!(config.rng_seed, Some(seed));
        assert_eq!(reader.max_tick().expect("last persisted tick"), Some(TICKS));
        let watermarks = reader.persistence_watermarks().expect("durable watermarks");
        assert!(watermarks.durable.is_some());
        assert_eq!(watermarks.admitted, watermarks.applied);
        assert_eq!(watermarks.applied, watermarks.durable);
        let stored = reader
            .recent_run_events(
                published.len() + usize::try_from(dropped).expect("bounded drops") + 1,
            )
            .expect("bounded complete typed rows")
            .into_iter()
            .map(|event| event.into_parts().1)
            .collect::<Vec<_>>();
        for kind in required_kinds {
            assert!(stored.iter().any(|event| event.kind == *kind));
        }
        if dropped == 0 {
            assert_eq!(
                serde_json::to_vec(&stored).expect("stored full event records"),
                serde_json::to_vec(&published).expect("published full event records"),
                "{label} must round-trip every field and prose byte, not only the event count"
            );
        } else {
            assert_eq!(published.len(), capacity);
            assert_eq!(
                u64::try_from(stored.len() - published.len()).unwrap(),
                dropped
            );
            for event in &published {
                let bytes = serde_json::to_vec(event).unwrap();
                assert!(
                    stored
                        .iter()
                        .any(|row| serde_json::to_vec(row).unwrap() == bytes)
                );
            }
        }
        let persisted_logs = stderr
            .lines()
            .filter(|line| line.contains("narrative event persisted"))
            .collect::<Vec<_>>();
        assert_eq!(persisted_logs.len(), stored.len());
        for event in &stored {
            assert!(
                persisted_logs.iter().any(|line| {
                    line.contains("INFO")
                        && line.contains(&format!("event_tick={} ", event.tick.0))
                        && line.contains(&format!("event_kind={} ", event.kind.as_str()))
                        && line.contains(&event.human_text)
                }),
                "every durable narrative row must have its actual info-level evidence"
            );
        }
        reader.close().expect("close actual reader");
        let mut around = base_command(env!("CARGO_BIN_EXE_control_cli"));
        around
            .args(["narrative", "around", "0", "--window"])
            .arg(TICKS.to_string())
            .arg("--db")
            .arg(&database)
            .arg("--json");
        let queried = retained_host_output(&directory, &format!("{label}-readback"), &mut around);
        assert!(
            queried.status.success(),
            "offline readback: {}",
            stderr_text(&queried)
        );
        let hits: Vec<scriptbots_app::narrative_search::NarrativeSearchHitDto> =
            serde_json::from_slice(&queried.stdout).expect("actual offline CLI hits");
        assert_eq!(hits.len(), stored.len());
        for (hit, event) in hits.iter().zip(&stored) {
            assert_eq!(hit.tick, event.tick.0);
            assert_eq!(hit.kind, event.kind.as_str());
            assert_eq!(hit.severity, event.severity);
            assert_eq!(hit.human_text.as_bytes(), event.human_text.as_bytes());
        }
        assert_eq!(fs::read(&database).expect("readback database"), before);
    }
    for (i, first) in observed_kinds.iter().enumerate() {
        for second in observed_kinds.iter().skip(i + 1) {
            assert_ne!(
                first, second,
                "the three real trajectories must not collapse"
            );
        }
    }
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

    // Export and verify through separate shipped application processes. Then read the
    // copied database independently, including the actual science rows and durability.
    let bundle_dir = temp_dir.path().join("portable-run");
    let created = base_command(env!("CARGO_BIN_EXE_scriptbots-app"))
        .arg("--create-bundle")
        .arg(&baseline_db)
        .arg("--bundle-output")
        .arg(&bundle_dir)
        .output()
        .expect("bundle export process runs");
    println!("BUNDLE_CREATE_STDOUT:\n{}", stdout_text(&created));
    println!("BUNDLE_CREATE_STDERR:\n{}", stderr_text(&created));
    assert!(
        created.status.success(),
        "bundle create: {}",
        stderr_text(&created)
    );
    let verified = base_command(env!("CARGO_BIN_EXE_scriptbots-app"))
        .arg("--verify-bundle")
        .arg(&bundle_dir)
        .output()
        .expect("bundle verification process runs");
    println!("BUNDLE_VERIFY_STDOUT:\n{}", stdout_text(&verified));
    println!("BUNDLE_VERIFY_STDERR:\n{}", stderr_text(&verified));
    assert!(
        verified.status.success(),
        "bundle verify: {}",
        stderr_text(&verified)
    );
    let source = StorageReader::open_finished(&db_display).expect("finished source");
    let copied = StorageReader::open_finished(&bundle_dir.join("run.db").to_string_lossy())
        .expect("independent portable database opens without source sidecars");
    assert_eq!(
        copied.run_manifest().expect("copied manifest"),
        source.run_manifest().expect("source manifest")
    );
    assert_eq!(
        copied.persistence_watermarks().expect("copied watermarks"),
        source.persistence_watermarks().expect("source watermarks")
    );
    assert_eq!(
        copied
            .run_ledger_summary()
            .expect("copied ledger")
            .tick_count,
        ledger.tick_count
    );
    assert_eq!(
        copied.load_replay_events().expect("copied replay"),
        recorded_replay
    );
    assert_eq!(
        copied.load_checkpoints().expect("copied checkpoints"),
        source.load_checkpoints().expect("source checkpoints")
    );
    assert_eq!(
        copied.recent_metrics(8).expect("copied metrics").len(),
        metrics.len()
    );
    copied.close().expect("copied reader closes");
    source.close().expect("source reader closes");

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

    // Domain events: a short launch-ecology run may have none yet, so this checks the export
    // against the storage evidence; `domain_event_export_carries_a_meadow_runs_births` checks
    // a run that is guaranteed to have some.
    verify_domain_event_export(&baseline_db, &temp_dir.path().join("domain_events.csv"));

    // ------------------------------------------------------------------
    // Phase 4: replay verification succeeds for the exact baseline config.
    // ------------------------------------------------------------------
    let mut verify = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    verify
        .env("RAYON_NUM_THREADS", HOST_INHERITED_RAYON_THREADS)
        .env("RUST_LOG", "info,scriptbots_core::parallelism=debug")
        .env("NO_COLOR", "1")
        .arg("--replay-db")
        .arg(&baseline_db)
        .arg("--threads")
        .arg(HOST_THREAD_BUDGET.to_string())
        .arg("--set")
        .arg(format!("rng_seed={SEED}"))
        .arg("--set")
        .arg("persistence_interval=1")
        .arg("--set")
        .arg("replay_event_tick_cap=65536");
    let verified = verify.output().expect("failed to run replay verification");
    assert_host_science_pool(&verified, ledger.tick_count);
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

/// The production interval recorder (not a planted row) supplies the checkpoint and the
/// canonical digest that `--checkpoint-start` replay verifies against.
#[test]
fn interval_checkpoints_recorded_by_a_real_run_support_checkpoint_start_replay() {
    let temp_dir = tempdir().expect("temp run directory");
    let database = temp_dir.path().join("interval_checkpoint_run.sqlite");

    let mut cmd = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    // Bootstrap ticks run without per-step command round trips, so the recorder finds a
    // due tick as soon as the frontend starts; the few rendered frames give it time to
    // capture and to place its digest before the final-frame digest.
    cmd.env("SCRIPTBOTS_TERMINAL_HEADLESS", "1")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", "8")
        .env("SCRIPTBOTS_STORAGE_PATH", &database)
        .args([
            "--storage",
            "file",
            "--threads",
            "1",
            "--bootstrap-ticks",
            "40",
            "--checkpoint-interval",
            "20",
        ])
        .arg("--set")
        .arg(format!("rng_seed={SEED}"))
        .args([
            "--set",
            "persistence_interval=1",
            "--set",
            "replay_event_tick_cap=65536",
        ]);
    let produced = cmd.output().expect("run headless terminal");
    assert!(
        produced.status.success(),
        "recorded run failed: {}",
        stderr_text(&produced)
    );

    let reader = StorageReader::open(&database.display().to_string()).expect("open run");
    let checkpoints = reader.load_checkpoints().expect("load checkpoints");
    reader.close().expect("close reader");
    assert!(
        !checkpoints.is_empty(),
        "the interval recorder stored no checkpoint in a 48-tick run at interval 20: {}",
        stderr_text(&produced)
    );
    // The recorder captures whatever quiescent tick it reaches once one is due.
    for record in &checkpoints {
        assert!(
            record.checkpoint_id == format!("auto-t{}", record.tick) && record.tick >= 20,
            "unexpected interval checkpoint {} at tick {}",
            record.checkpoint_id,
            record.tick
        );
        record
            .world_checkpoint()
            .expect("stored checkpoint decodes");
    }

    let mut verify = base_command(env!("CARGO_BIN_EXE_scriptbots-app"));
    verify
        .arg("--replay-db")
        .arg(&database)
        .arg("--checkpoint-start")
        .args(["--threads", "1"])
        .arg("--set")
        .arg(format!("rng_seed={SEED}"))
        .args([
            "--set",
            "persistence_interval=1",
            "--set",
            "replay_event_tick_cap=65536",
        ]);
    let out = verify.output().expect("run checkpoint-start replay");
    let text = strip_ansi(&format!("{}{}", stdout_text(&out), stderr_text(&out)));
    assert!(
        out.status.success(),
        "checkpoint-start replay failed: {text}"
    );
    assert!(
        text.contains("Replay matched"),
        "replay from a recorded interval checkpoint must match: {text}"
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
    // The binary ran without a scenario document, so it used the app launch ecology
    // (compose_config_with_scenario); this world must use the same science fields.
    let config = scriptbots_core::ScriptBotsConfig {
        rng_seed: Some(SEED),
        persistence_interval: 0,
        replay_event_tick_cap: 65536,
        history_capacity: 600,
        reproduction_rate_herbivore: 0.0001,
        reproduction_rate_carnivore: 0.0001,
        reproduction_cooldown: 7,
        food_growth_rate: 0.0005,
        population_minimum: 20,
        ..scriptbots_core::ScriptBotsConfig::default()
    };
    let mut world = scriptbots_core::WorldState::new(config).expect("build world");
    let brain_keys = scriptbots_app::install_brains(&mut world, scriptbots_app::BrainPreset::Mixed)
        .expect("install brains")
        .population;
    // The binary ran without a scenario document, so it used the spread founder recipe.
    scriptbots_app::seed_founders(
        &mut world,
        &brain_keys,
        scriptbots_app::FounderRecipe::Spread,
    )
    .expect("seed founders");
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

/// Exercise the production full-host handoff and same-session continuation with real SQL.
/// Every database and subprocess stream remains beneath the unique external proof directory.
#[test]
fn mock_free_full_host_checkpoint_continuation_e2e() {
    let artifact_root = env::var_os("SCRIPTBOTS_TEST_ARTIFACT_ROOT")
        .map_or_else(env::temp_dir, std::path::PathBuf::from);
    fs::create_dir_all(&artifact_root).expect("create external host continuation artifact root");
    let directory = tempfile::Builder::new()
        .prefix("host-checkpoint-continuation-")
        .tempdir_in(artifact_root)
        .expect("unique external host continuation proof")
        .keep();
    eprintln!(
        "host continuation evidence retained at {}",
        directory.display()
    );
    let baseline_db = directory.join("uninterrupted.sqlite");
    let split_db = directory.join("continued.sqlite");
    let source_db = directory.join("closed-source-cut.sqlite");
    let baseline_output = produce_host_handoff(
        &directory,
        "uninterrupted",
        &baseline_db,
        HOST_CONTINUATION_TICKS,
        "uninterrupted-final",
        "0.0005",
        "mlp",
    );
    assert!(
        baseline_output.status.success(),
        "actual uninterrupted handoff failed: {}",
        stderr_text(&baseline_output)
    );
    let baseline_report =
        host_cli_report(&baseline_output, "scriptbots.host-checkpoint-handoff.v1");
    let baseline_record =
        assert_host_handoff_readback(&baseline_db, &baseline_report, HOST_CONTINUATION_TICKS);
    let cut_output = produce_host_handoff(
        &directory,
        "initial-cut",
        &split_db,
        HOST_HANDOFF_TICK,
        "first-cut",
        "0.0005",
        "mlp",
    );
    assert!(
        cut_output.status.success(),
        "actual initial handoff failed: {}",
        stderr_text(&cut_output)
    );
    let cut_report = host_cli_report(&cut_output, "scriptbots.host-checkpoint-handoff.v1");
    let cut_record = assert_host_handoff_readback(&split_db, &cut_report, HOST_HANDOFF_TICK);
    copy_closed_host_database(&split_db, &source_db);
    assert_eq!(
        fs::read(&source_db).expect("closed source bytes"),
        fs::read(&split_db).expect("initial database bytes")
    );
    let source_bytes = fs::read(&source_db).expect("retained source bytes");
    let source_prefix = host_database_prefix(&source_db);
    assert_eq!(source_prefix, host_database_prefix(&split_db));
    let continued_output = continue_host_handoff(
        &directory,
        "continuation",
        &split_db,
        "first-cut",
        HOST_CONTINUATION_TICKS - HOST_HANDOFF_TICK,
        "mlp",
    );
    assert!(
        continued_output.status.success(),
        "actual same-session continuation failed: {}",
        stderr_text(&continued_output)
    );
    let continued_report =
        host_cli_report(&continued_output, "scriptbots.host-checkpoint-handoff.v1");
    let continued_record =
        assert_host_handoff_readback(&split_db, &continued_report, HOST_CONTINUATION_TICKS);
    assert_eq!(continued_report["session_id"], cut_report["session_id"]);
    assert_eq!(continued_report["run_id"], cut_report["run_id"]);
    assert_ne!(
        continued_report["checkpoint_id"],
        cut_report["checkpoint_id"]
    );
    assert_eq!(
        continued_report["metadata"]["continued_from_checkpoint"],
        "first-cut"
    );
    assert_eq!(
        continued_report["metadata"]["source_tick"],
        HOST_HANDOFF_TICK
    );
    assert_eq!(
        continued_report["metadata"]["requested_steps"],
        HOST_CONTINUATION_TICKS - HOST_HANDOFF_TICK
    );
    let uninterrupted = baseline_record
        .host_checkpoint()
        .expect("uninterrupted host");
    let continued = continued_record.host_checkpoint().expect("continued host");
    assert_eq!(
        uninterrupted.world_checkpoint().source_digest(),
        continued.world_checkpoint().source_digest(),
        "every actual science digest lane must equal N versus K + (N-K)"
    );
    assert_eq!(uninterrupted.step_traces(), continued.step_traces());
    assert_eq!(baseline_report["config"], continued_report["config"]);
    assert_eq!(
        baseline_report["brain_roster"],
        continued_report["brain_roster"]
    );
    assert_eq!(
        baseline_report["source_commit"],
        continued_report["source_commit"]
    );
    let reader = StorageReader::open(split_db.to_str().expect("UTF-8 split database"))
        .expect("read retained initial artifact after continuation");
    let checkpoints = reader
        .load_checkpoints()
        .expect("both actual host artifacts");
    assert_eq!(checkpoints.len(), 2);
    assert_eq!(
        checkpoints[0], cut_record,
        "continuation must preserve its source artifact"
    );
    assert_eq!(checkpoints[1], continued_record);
    reader.close().expect("close continued artifact reader");

    let matched_output = compare_host_handoffs(
        &directory,
        "matched-comparison",
        &baseline_db,
        "uninterrupted-final",
        &split_db,
        continued_record.checkpoint_id.as_str(),
    );
    assert!(
        matched_output.status.success(),
        "production comparison of actual continued traces failed: {}",
        stderr_text(&matched_output)
    );
    let matched = host_cli_report(&matched_output, "scriptbots.host-checkpoint-comparison.v1");
    assert_eq!(matched["status"], "match");
    assert_eq!(
        matched["basis"],
        "complete_observed_step_trace_prefix_and_final_boundary"
    );
    assert_eq!(matched["compared_tick"], HOST_CONTINUATION_TICKS);
    assert_eq!(
        matched["compared_trace_ticks"],
        serde_json::json!(uninterrupted.step_traces().len())
    );
    assert!(matched["first_observed_divergence"].is_null());
    assert_eq!(
        matched["expected_boundary_digest"],
        serde_json::to_value(uninterrupted.world_checkpoint().source_digest())
            .expect("actual uninterrupted boundary JSON")
    );
    assert_eq!(
        matched["actual_boundary_digest"],
        serde_json::to_value(continued.world_checkpoint().source_digest())
            .expect("actual continued boundary JSON")
    );
    assert_host_comparison_source(&matched["expected"], &baseline_report);
    assert_host_comparison_source(&matched["actual"], &continued_report);
    assert!(matched["expected"]["continuation_source"].is_null());
    assert_eq!(
        matched["actual"]["continuation_source"],
        serde_json::json!({
            "checkpoint_id": cut_record.checkpoint_id, "checkpoint_tick": cut_record.tick,
            "payload_digest": cut_record.payload_digest,
        })
    );

    let perturbed_db = directory.join("perturbed.sqlite");
    let perturbed_output = produce_host_handoff(
        &directory,
        "perturbed",
        &perturbed_db,
        HOST_CONTINUATION_TICKS,
        "perturbed-final",
        "0.4",
        "mlp",
    );
    assert!(
        perturbed_output.status.success(),
        "actual perturbed MLP handoff failed: {}",
        stderr_text(&perturbed_output)
    );
    let perturbed_report =
        host_cli_report(&perturbed_output, "scriptbots.host-checkpoint-handoff.v1");
    let perturbed_record =
        assert_host_handoff_readback(&perturbed_db, &perturbed_report, HOST_CONTINUATION_TICKS);
    let divergent_output = compare_host_handoffs(
        &directory,
        "divergent-comparison",
        &baseline_db,
        "uninterrupted-final",
        &perturbed_db,
        "perturbed-final",
    );
    assert!(
        !divergent_output.status.success(),
        "real altered configuration must diverge"
    );
    let divergent = host_cli_report(
        &divergent_output,
        "scriptbots.host-checkpoint-comparison.v1",
    );
    assert_eq!(divergent["status"], "diverged");
    assert_eq!(divergent["compared_tick"], HOST_CONTINUATION_TICKS);
    assert_eq!(
        divergent["compared_trace_ticks"],
        serde_json::json!(uninterrupted.step_traces().len())
    );
    assert_host_comparison_source(&divergent["expected"], &baseline_report);
    assert_host_comparison_source(&divergent["actual"], &perturbed_report);
    let perturbed = perturbed_record
        .host_checkpoint()
        .expect("actual perturbed host");
    // The altered configuration is present at the first Sense observation, before actuation.
    let first = &divergent["first_observed_divergence"];
    assert_eq!(first["tick"], 1);
    assert_eq!(
        first["stage"],
        scriptbots_core::WorldStepTracePoint::Sense.as_str()
    );
    let expected_stage = uninterrupted.step_traces()[0].stages[0]
        .capture
        .as_ref()
        .expect("actual first baseline Sense capture");
    let actual_stage = perturbed.step_traces()[0].stages[0]
        .capture
        .as_ref()
        .expect("actual first perturbed Sense capture");
    assert_ne!(expected_stage.world.config, actual_stage.world.config);
    assert_ne!(expected_stage.overall, actual_stage.overall);
    assert_eq!(first["expected_digest"], expected_stage.overall);
    assert_eq!(first["actual_digest"], actual_stage.overall);

    // Distinct closed copies exercise the outer checksum guard and the inner codec guard.
    // Recompute only the truncated copy's outer checksum so it reaches actual wire decoding.
    for name in ["corrupt", "truncated"] {
        let database = directory.join(format!("{name}.sqlite"));
        copy_closed_host_database(&source_db, &database);
        let checkpoint = cut_record
            .host_checkpoint()
            .expect("validated source artifact");
        let mut bytes = checkpoint.encode().expect("actual host artifact bytes");
        let original_digest = cut_record.payload_digest.clone();
        let digest = if name == "corrupt" {
            bytes[0] ^= 1;
            original_digest
        } else {
            bytes.truncate(bytes.len() / 2);
            format!("blake3:{}", blake3::hash(&bytes).to_hex())
        };
        let payload = bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let connection = Connection::open_existing(database.to_str().expect("UTF-8 copy"))
            .expect("open only the negative closed copy for real SQL mutation");
        assert_eq!(
            connection
                .execute_with_params(
                    "UPDATE checkpoints SET payload = ?1, payload_digest = ?2 WHERE checkpoint_id = ?3",
                    &[payload.into(), digest.into(), "first-cut".into()],
                )
                .expect("mutate the copied actual artifact"),
            1
        );
        connection
            .close()
            .expect("close mutated copy before the real CLI");
        let before = host_database_prefix(&database);
        let output = continue_host_handoff(&directory, name, &database, "first-cut", 1, "mlp");
        assert!(!output.status.success(), "{name} artifact must fail closed");
        let text = strip_ansi(&format!("{}{}", stdout_text(&output), stderr_text(&output)));
        assert!(
            text.contains("first-cut") && text.contains("checkpoints.payload"),
            "refusal must name the actual source artifact: {text}"
        );
        if name == "corrupt" {
            assert!(
                text.contains("payload digest mismatch"),
                "outer checksum refusal: {text}"
            );
        } else {
            assert!(
                text.contains("wire.decode") && !text.contains("payload digest mismatch"),
                "a valid outer checksum must reach the actual truncated host codec: {text}"
            );
        }
        let after = host_database_prefix(&database);
        fs::write(
            directory.join(format!("{name}.prefixes.json")),
            serde_json::to_vec_pretty(&serde_json::json!({"before": before, "after": after}))
                .expect("encode actual negative prefix observations"),
        )
        .expect("retain actual negative prefix observations");
        assert_eq!(
            after, before,
            "{name} refusal must leave every observed science/journal prefix and row count unchanged"
        );
        assert!(!stdout_text(&output).contains("\"status\":\"handed_off\""));
    }
    assert_eq!(
        fs::read(&source_db).expect("original retained source"),
        source_bytes
    );
    assert_eq!(host_database_prefix(&source_db), source_prefix);
    #[cfg(feature = "neuro")]
    {
        let database = directory.join("uncovered-factory.sqlite");
        let output = produce_host_handoff(
            &directory,
            "uncovered-factory",
            &database,
            1,
            "uncovered-factory-cut",
            "0.0005",
            "mixed",
        );
        assert!(
            output.status.success(),
            "a retained artifact must report its actual continuation limitation: {}",
            stderr_text(&output)
        );
        let report = host_cli_report(&output, "scriptbots.host-checkpoint-handoff.v1");
        assert_eq!(report["status"], "handed_off");
        let limitations: scriptbots_app::CharacterizationLimitationsV0 =
            serde_json::from_value(report["characterization_limitations"].clone())
                .expect("typed real uncovered-factory claim");
        assert_eq!(
            limitations.purpose,
            "retained_full_host_checkpoint_continuation"
        );
        assert!(!limitations.checkpoint_replay_guarantee);
        assert!(
            report["continuation_refusal"]
                .as_str()
                .expect("actual uncovered-factory refusal")
                .contains("complete evaluator and factory state")
        );
        let reader = StorageReader::open(database.to_str().expect("UTF-8 negative database"))
            .expect("independently read the real uncovered-factory artifact");
        let record = reader
            .load_checkpoints()
            .expect("real retained uncovered-factory checkpoint")
            .into_iter()
            .find(|record| record.checkpoint_id == "uncovered-factory-cut")
            .expect("exact reported checkpoint exists");
        let checkpoint = record
            .host_checkpoint()
            .expect("decode real mixed artifact");
        assert_eq!(checkpoint.tick().0, 1);
        assert!(checkpoint.world_checkpoint().agent_count() >= HOST_MINIMUM_POPULATION);
        let digest = checkpoint.world_checkpoint().source_digest();
        assert!(digest.evaluator_state_covered);
        assert!(limitations.evaluator_state_covered);
        assert!(!digest.factory_state_covered);
        let uncovered_family = scriptbots_brain_neuro::NeuroflowBrain::KIND.as_str();
        assert_eq!(
            digest.uncovered_factory_families,
            [uncovered_family.to_owned()]
        );
        assert_eq!(
            report["boundary_digest"],
            serde_json::to_value(digest).expect("actual uncovered-factory digest JSON")
        );
        assert_eq!(report["payload_digest"], record.payload_digest);
        let watermarks = reader
            .persistence_watermarks()
            .expect("actual durable prefixes");
        assert!(watermarks.durable.is_some());
        assert_eq!(watermarks.admitted, watermarks.applied);
        assert_eq!(watermarks.applied, watermarks.durable);
        assert_eq!(report["watermarks"], host_watermarks_json(watermarks));
        reader
            .close()
            .expect("close independent uncovered-factory reader");
        let before = host_database_prefix(&database);
        let refused = continue_host_handoff(
            &directory,
            "uncovered-factory-continuation",
            &database,
            "uncovered-factory-cut",
            1,
            "mixed",
        );
        assert!(!refused.status.success());
        let text = strip_ansi(&format!(
            "{}{}",
            stdout_text(&refused),
            stderr_text(&refused)
        ));
        assert!(text.contains("uncovered-factory-cut"));
        assert!(text.contains("complete evaluator and factory state"));
        let after = host_database_prefix(&database);
        assert_eq!(
            after, before,
            "uncovered factory refusal cannot advance the real run"
        );
        fs::write(
            directory.join("uncovered-factory.observations.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "handoff": report, "before": before, "after": after,
            }))
            .expect("encode real uncovered-factory observations"),
        )
        .expect("retain real uncovered-factory observations");
    }
    // Resume the returned artifact again, as the documented repeated-handoff workflow does.
    // The same next transition must agree after one handoff versus two handoffs.
    let chained_tick = HOST_CONTINUATION_TICKS + 1;
    let baseline_next = continue_host_handoff(
        &directory,
        "uninterrupted-next",
        &baseline_db,
        &baseline_record.checkpoint_id,
        1,
        "mlp",
    );
    assert!(
        baseline_next.status.success(),
        "{}",
        stderr_text(&baseline_next)
    );
    let baseline_next_report =
        host_cli_report(&baseline_next, "scriptbots.host-checkpoint-handoff.v1");
    let baseline_next_record =
        assert_host_handoff_readback(&baseline_db, &baseline_next_report, chained_tick);
    let continued_next = continue_host_handoff(
        &directory,
        "continuation-next",
        &split_db,
        &continued_record.checkpoint_id,
        1,
        "mlp",
    );
    assert!(
        continued_next.status.success(),
        "{}",
        stderr_text(&continued_next)
    );
    let continued_next_report =
        host_cli_report(&continued_next, "scriptbots.host-checkpoint-handoff.v1");
    let continued_next_record =
        assert_host_handoff_readback(&split_db, &continued_next_report, chained_tick);
    let baseline_next_host = baseline_next_record
        .host_checkpoint()
        .expect("next baseline host");
    let continued_next_host = continued_next_record
        .host_checkpoint()
        .expect("next continued host");
    assert_eq!(
        baseline_next_host.world_checkpoint().source_digest(),
        continued_next_host.world_checkpoint().source_digest()
    );
    assert_eq!(
        baseline_next_host.step_traces(),
        continued_next_host.step_traces()
    );
    assert_eq!(
        baseline_next_report["session_id"],
        baseline_report["session_id"]
    );
    assert_eq!(
        continued_next_report["session_id"],
        cut_report["session_id"]
    );
    assert_eq!(baseline_next_report["run_id"], baseline_report["run_id"]);
    assert_eq!(continued_next_report["run_id"], cut_report["run_id"]);

    // A valid artifact from an older cut must be attributable and must not rewind the run.
    let stale_before = host_database_prefix(&split_db);
    let stale = continue_host_handoff(
        &directory,
        "stale-cut",
        &split_db,
        &cut_record.checkpoint_id,
        1,
        "mlp",
    );
    let stale_after = host_database_prefix(&split_db);
    fs::write(
        directory.join("stale-cut.observations.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "checkpoint_id": cut_record.checkpoint_id, "source_tick": HOST_HANDOFF_TICK,
            "before": stale_before, "after": stale_after,
        }))
        .expect("encode actual stale-cut observations"),
    )
    .expect("retain actual stale-cut observations");
    assert!(
        !stale.status.success(),
        "an obsolete handoff must refuse continuation"
    );
    assert_eq!(
        stale_after, stale_before,
        "stale-cut refusal cannot rewind or append science/journal rows"
    );
    let stale_text = strip_ansi(&format!("{}{}", stdout_text(&stale), stderr_text(&stale)));
    assert!(
        stale_text.contains(&cut_record.checkpoint_id)
            && stale_text.contains(&format!("source tick {HOST_HANDOFF_TICK}"))
            && stale_text.contains(split_db.to_str().expect("UTF-8 actual database")),
        "stale-cut refusal must name the actual checkpoint, source tick and database: {stale_text}"
    );
    assert!(stale_text.contains("checkpoints.host_continuation_cut"));
    fs::write(
        directory.join("observed-handoffs.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "uninterrupted": baseline_report, "initial_cut": cut_report,
            "continued": continued_report, "perturbed": perturbed_report,
            "matched_comparison": matched, "divergent_comparison": divergent,
            "source_prefix": source_prefix,
            "next_uninterrupted": baseline_next_report,
            "next_continued": continued_next_report,
        }))
        .expect("encode actual handoff observations"),
    )
    .expect("retain actual handoff observations");
}
