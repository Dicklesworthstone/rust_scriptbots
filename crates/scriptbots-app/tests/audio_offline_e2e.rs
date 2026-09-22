//! Mock-free E2E test running real app for 5,000 ticks through real file-backed
//! FrankenSQLite persistence, closing run, invoking headless CLI twice, independently
//! parsing both WAVs and asserting identical digests (`bd-16g.14.2`).

use scriptbots_app::brains::{BrainPreset, install_brains};
use scriptbots_app::seed_founding_population;
use scriptbots_core::audio::{compute_pcm_sha256, read_canonical_wav};
use scriptbots_core::{ScriptBotsConfig, WorldState};
use scriptbots_storage::{Connection, StoragePipeline};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn app_binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_scriptbots-app"))
}

fn temp_test_dir(prefix: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nonce = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    path.push(format!(
        "audio_e2e_{prefix}_{}_{timestamp}_{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

#[test]
fn test_mock_free_e2e_5000_ticks_and_dual_cli_determinism() {
    let dir = temp_test_dir("5000_ticks");
    let db_path = dir.join("run.sqlite");
    let db_str = db_path.to_str().expect("valid utf-8 path");

    // 1. Run real simulation for 5,000 ticks through real file-backed FrankenSQLite persistence
    const TOTAL_TICKS: u64 = 5_000;

    let config = ScriptBotsConfig {
        world_width: 64,
        world_height: 64,
        food_cell_size: 16,
        chart_flush_interval: 0,
        narrative_interval: 0,
        narrative_capacity: 0,
        rng_seed: Some(0xCAFE_BABE),
        reproduction_attempt_chance: 0.0,
        ..ScriptBotsConfig::default()
    };

    let mut world = WorldState::new(config).expect("construct real world");
    let installed = install_brains(&mut world, BrainPreset::Mlp).expect("install brains");
    seed_founding_population(&mut world, &installed.population).expect("seed founding agents");

    let storage = Storage::create_unattributed_file(db_str).expect("create file-backed storage");
    let run_id_str = storage.run_id.to_string();
    let conn = storage.connection().expect("storage connection");
    let mut tx = conn.transaction().expect("begin transaction");

    // Record tick 0 state
    tx.execute_with_params(
        "INSERT INTO tick_summaries (run_id, tick, epoch, closed, agent_count, births, deaths, total_energy, average_energy, average_health, island_id)
         VALUES (?1, 0, 0, 1, ?2, 0, 0, 100.0, 10.0, 1.0, 0)",
        &[
            run_id_str.as_str().into(),
            (world.agent_count() as i64).into(),
        ],
    ).expect("insert tick 0 summary");

    for tick in 1..=TOTAL_TICKS {
        let events = world.step().expect("step real world simulation");
        let tick_i64 = tick as i64;
        let pop = world.agent_count() as i64;
        let births = events.births.len() as i64;
        let deaths = events.deaths.len() as i64;

        tx.execute_with_params(
            "INSERT INTO tick_summaries (run_id, tick, epoch, closed, agent_count, births, deaths, total_energy, average_energy, average_health, island_id)
             VALUES (?1, ?2, 0, 1, ?3, ?4, ?5, 100.0, 10.0, 1.0, 0)",
            &[
                run_id_str.as_str().into(),
                tick_i64.into(),
                pop.into(),
                births.into(),
                deaths.into(),
            ],
        ).expect("insert tick summary");

        if events.spike_hits > 0 {
            tx.execute_with_params(
                "INSERT INTO events (run_id, tick, kind, count, island_id)
                 VALUES (?1, ?2, 'combat', ?3, 0)",
                &[
                    run_id_str.as_str().into(),
                    tick_i64.into(),
                    (events.spike_hits as i64).into(),
                ],
            ).expect("insert event");
        }
    }

    tx.commit().expect("commit transaction");
    drop(tx);
    drop(storage);

    assert!(db_path.exists(), "database file must exist");

    // 2. Invocation 1: Render headless audio via CLI
    let wav_a = dir.join("render_a.wav");
    let csv_a = dir.join("render_a.csv");

    let output_a = app_binary()
        .arg("render-audio")
        .arg("--run")
        .arg(&db_path)
        .arg("--from")
        .arg("1")
        .arg("--to")
        .arg("5001")
        .arg("--out")
        .arg(&wav_a)
        .arg("--stats")
        .arg(&csv_a)
        .output()
        .expect("execute render-audio run A");

    assert!(
        output_a.status.success(),
        "first invocation failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output_a.stdout),
        String::from_utf8_lossy(&output_a.stderr)
    );

    assert!(wav_a.exists(), "WAV A must be produced");
    assert!(csv_a.exists(), "CSV A must be produced");

    // 3. Invocation 2: Render identical range via CLI
    let wav_b = dir.join("render_b.wav");
    let csv_b = dir.join("render_b.csv");

    let output_b = app_binary()
        .arg("render-audio")
        .arg("--run")
        .arg(&db_path)
        .arg("--from")
        .arg("1")
        .arg("--to")
        .arg("5001")
        .arg("--out")
        .arg(&wav_b)
        .arg("--stats")
        .arg(&csv_b)
        .output()
        .expect("execute render-audio run B");

    assert!(
        output_b.status.success(),
        "second invocation failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output_b.stdout),
        String::from_utf8_lossy(&output_b.stderr)
    );

    assert!(wav_b.exists(), "WAV B must be produced");
    assert!(csv_b.exists(), "CSV B must be produced");

    // 4. Independently parse both WAV files and verify bit-identical contents and digests
    let parsed_a = read_canonical_wav(&wav_a).expect("parse canonical WAV A");
    let parsed_b = read_canonical_wav(&wav_b).expect("parse canonical WAV B");

    assert_eq!(parsed_a.sample_rate, 48000);
    assert_eq!(parsed_b.sample_rate, 48000);
    assert_eq!(parsed_a.channels, 1);
    assert_eq!(parsed_b.channels, 1);
    assert_eq!(parsed_a.samples.len(), parsed_b.samples.len());
    assert!(!parsed_a.samples.is_empty(), "audio must not be empty");

    // Compute SHA-256 over PCM data chunks
    let sha_a = compute_pcm_sha256(&parsed_a.samples);
    let sha_b = compute_pcm_sha256(&parsed_b.samples);
    assert_eq!(
        sha_a, sha_b,
        "SHA-256 digests across dual CLI renders must be identical"
    );

    // Exact byte equality of WAV files
    let bytes_a = fs::read(&wav_a).expect("read wav A bytes");
    let bytes_b = fs::read(&wav_b).expect("read wav B bytes");
    assert_eq!(bytes_a, bytes_b, "WAV byte streams must be bit-identical");

    // Exact byte equality of telemetry CSV files
    let csv_bytes_a = fs::read(&csv_a).expect("read csv A bytes");
    let csv_bytes_b = fs::read(&csv_b).expect("read csv B bytes");
    assert_eq!(
        csv_bytes_a, csv_bytes_b,
        "CSV telemetry streams must be bit-identical"
    );

    // Assert limiter peak amplitude strictly bounded within [-1.0, 1.0]
    for &sample in &parsed_a.samples {
        assert!(
            (-1.0..=1.0).contains(&sample),
            "sample {sample} exceeded peak amplitude [-1.0, 1.0]"
        );
    }
}

#[test]
fn test_negative_gapped_database_refuses_render_and_leaves_zero_partial_files() {
    let dir = temp_test_dir("gapped_db");
    let db_path = dir.join("gapped.sqlite");
    let db_str = db_path.to_str().expect("valid utf-8 path");

    // 1. Create a database with 10 ticks
    let mut pipeline = StoragePipeline::create_unattributed_file(db_str)
        .expect("create unattributed storage pipeline");

    let config = ScriptBotsConfig {
        world_width: 64,
        world_height: 64,
        food_cell_size: 16,
        rng_seed: Some(0x1234),
        persistence_interval: 1,
        ..ScriptBotsConfig::default()
    };

    let (mut world, mut persistence) =
        WorldState::with_persistence(config, Box::new(pipeline.sink()))
            .expect("construct world with persistence");

    for _ in 0..10 {
        persistence.step(&mut world).expect("step tick");
    }

    pipeline.flush_and_wait().expect("flush pipeline");
    pipeline.shutdown().expect("shutdown storage pipeline");

    // 2. Corrupt/gap the database by deleting tick 5 from tick_summaries
    {
        let connection = Connection::open(db_str).expect("open connection");
        connection
            .execute("DELETE FROM tick_summaries WHERE tick = 5")
            .expect("delete tick 5 to create gap");
    }

    // 3. Attempt to render range [0, 10)
    let bad_wav = dir.join("bad.wav");
    let bad_csv = dir.join("bad.csv");

    let output = app_binary()
        .arg("render-audio")
        .arg("--run")
        .arg(&db_path)
        .arg("--from")
        .arg("1")
        .arg("--to")
        .arg("11")
        .arg("--out")
        .arg(&bad_wav)
        .arg("--stats")
        .arg(&bad_csv)
        .output()
        .expect("execute render-audio");

    assert!(
        !output.status.success(),
        "rendering gapped database must fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("missing tick 5") || stderr.contains("gap"),
        "stderr must report typed gap error: {stderr}"
    );

    // 4. Assert zero partial files left behind
    assert!(!bad_wav.exists(), "no partial WAV file must remain");
    assert!(!bad_csv.exists(), "no partial CSV file must remain");
}

#[test]
fn test_negative_invalid_range_refuses_render_and_leaves_zero_partial_files() {
    let dir = temp_test_dir("invalid_range_cli");
    let db_path = dir.join("dummy.sqlite");
    let db_str = db_path.to_str().expect("valid utf-8 path");

    let mut pipeline = StoragePipeline::create_unattributed_file(db_str)
        .expect("create unattributed storage pipeline");
    pipeline.shutdown().expect("shutdown storage pipeline");

    let bad_wav = dir.join("bad.wav");
    let bad_csv = dir.join("bad.csv");

    let output = app_binary()
        .arg("render-audio")
        .arg("--run")
        .arg(&db_path)
        .arg("--from")
        .arg("50")
        .arg("--to")
        .arg("10") // Invalid: 50 > 10
        .arg("--out")
        .arg(&bad_wav)
        .arg("--stats")
        .arg(&bad_csv)
        .output()
        .expect("execute render-audio");

    assert!(
        !output.status.success(),
        "rendering invalid range must fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("must be <=") || stderr.contains("invalid tick range"),
        "stderr must report invalid range error: {stderr}"
    );

    assert!(!bad_wav.exists(), "no partial WAV file must remain");
    assert!(!bad_csv.exists(), "no partial CSV file must remain");
}
