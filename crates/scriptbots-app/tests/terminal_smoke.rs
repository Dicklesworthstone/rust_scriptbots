use scriptbots_app::STORAGE_SIDECAR_SUFFIXES;
use std::{
    env,
    ffi::{OsStr, OsString},
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::tempdir;

fn clear_scriptbots_environment(command: &mut Command, names: impl IntoIterator<Item = OsString>) {
    for name in names {
        let encoded = name.as_encoded_bytes();
        if encoded.starts_with(b"SCRIPTBOTS_") || encoded.starts_with(b"SB_") {
            command.env_remove(name);
        }
    }
}

fn headless_command() -> Command {
    let bin = env!("CARGO_BIN_EXE_scriptbots-app");
    let mut cmd = Command::new(bin);
    // A developer shell may carry recovery, profiling, renderer, storage, or
    // malformed control settings. Clear the whole application namespace so
    // every subprocess test proves only the branch it declares below.
    clear_scriptbots_environment(&mut cmd, env::vars_os().map(|(name, _)| name));
    cmd.env("SCRIPTBOTS_MODE", "terminal")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS", "1")
        .env("SCRIPTBOTS_CONTROL_REST_ENABLED", "0")
        .env("SCRIPTBOTS_CONTROL_MCP", "disabled")
        .env("TERM", "xterm-256color")
        .env("RUST_LOG", "off");
    cmd
}

#[test]
fn headless_environment_sanitizer_removes_only_scriptbots_namespaces() {
    let mut command = Command::new("unused-test-command");
    clear_scriptbots_environment(
        &mut command,
        [
            OsString::from("SCRIPTBOTS_RECOVER_STORAGE"),
            OsString::from("SB_WGPU_DUMP"),
            OsString::from("HOME"),
        ],
    );

    for removed in ["SCRIPTBOTS_RECOVER_STORAGE", "SB_WGPU_DUMP"] {
        assert_eq!(
            command
                .get_envs()
                .find(|(name, _)| *name == OsStr::new(removed))
                .map(|(_, value)| value),
            Some(None),
            "{removed} was not explicitly removed"
        );
    }
    assert!(
        command
            .get_envs()
            .all(|(name, _)| name != OsStr::new("HOME")),
        "unrelated environment names must remain inherited"
    );
}

fn assert_no_startup_artifacts(storage_path: &Path, config_path: &Path, tuning_dir: &Path) {
    assert!(
        !storage_path.exists(),
        "control preflight failure must not reserve FrankenSQLite storage"
    );
    let writer_lock = PathBuf::from(format!(
        "{}{}",
        storage_path.display(),
        ".scriptbots-writer.lock"
    ));
    assert!(
        !writer_lock.exists(),
        "control preflight failure created the persistent writer-lock companion"
    );
    assert!(
        !config_path.exists(),
        "control preflight failure wrote configuration"
    );
    for suffix in STORAGE_SIDECAR_SUFFIXES {
        let sidecar = PathBuf::from(format!("{}{suffix}", storage_path.display()));
        assert!(
            !sidecar.exists(),
            "control preflight failure created unexpected sidecar {}",
            sidecar.display()
        );
    }
    assert_eq!(
        fs::read_dir(tuning_dir)
            .expect("auto-tune scratch directory")
            .count(),
        0,
        "control preflight failure left auto-tune artifacts"
    );
}

#[test]
fn actual_binary_terminal_test_backend_path_exits_successfully() {
    let temp_dir = tempdir().expect("temp storage directory");
    let storage_path = temp_dir.path().join("scriptbots_headless.sqlite");

    let mut cmd = headless_command();
    cmd.env("SCRIPTBOTS_STORAGE_PATH", &storage_path);
    let status = cmd.status().expect("failed to run scriptbots-app binary");
    assert!(status.success(), "terminal headless run failed");
}

#[test]
fn actual_binary_terminal_test_backend_path_reports_rendered_tick_budget() {
    let temp_dir = tempdir().expect("temp storage directory");
    let storage_path = temp_dir.path().join("scriptbots_headless_report.sqlite");

    let mut cmd = headless_command();
    cmd.env("RUST_LOG", "info")
        .env("RUST_LOG_STYLE", "never")
        .env("SCRIPTBOTS_STORAGE_PATH", &storage_path);

    let output = cmd.output().expect("failed to run scriptbots-app binary");
    assert!(
        output.status.success(),
        "terminal headless run failed: status={:?}",
        output.status
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    let clean = strip_ansi(&stderr);
    assert!(
        clean.contains("Starting ScriptBots simulation shell"),
        "expected startup log; stderr:\n{clean}"
    );
    assert!(
        clean.contains("renderer=\"terminal\""),
        "expected renderer selection log; stderr:\n{clean}"
    );
    assert!(
        clean.contains("Initialized seeded world at tick zero without bootstrap advancement"),
        "expected explicit zero-bootstrap diagnostic; stderr:\n{clean}"
    );
    assert!(
        clean.contains("Terminal headless run completed"),
        "expected terminal completion log; stderr:\n{clean}"
    );
    assert!(
        clean.contains("final_tick=12"),
        "expected 0 bootstrap ticks plus 12 headless-renderer ticks; stderr:\n{clean}"
    );
}

#[derive(Clone, Copy)]
enum SchedulePerturbation {
    None,
    SkipZero,
    ReverseSameTick,
    DelayRecovery,
}

// Explicit assignments form the oracle, independent of the runtime scheduler,
// scenario resolver, and its sort. Each perturbation must change the result.
fn schedule_reference(
    perturbation: SchedulePerturbation,
) -> (Vec<serde_json::Value>, Vec<scriptbots_core::WorldDigestV1>) {
    use scriptbots_app::{
        brains::{BrainPreset, install_brains},
        seed_founding_population,
    };
    use scriptbots_core::{ScriptBotsConfig, WorldState};
    let config = ScriptBotsConfig {
        rng_seed: Some(4242),
        world_width: 600,
        world_height: 600,
        food_cell_size: 50,
        population_minimum: 0,
        population_spawn_interval: 0,
        // The oracle advances science directly. WorldDigestV1 explicitly
        // excludes persistence cadence from its scientific config lane.
        persistence_interval: 0,
        history_capacity: 600,
        metabolism_drain: 0.001,
        food_growth_rate: 0.01,
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("reference world");
    let brains = install_brains(&mut world, BrainPreset::Mlp).expect("production MLP installer");
    seed_founding_population(&mut world, brains.population()).expect("production founders");
    let mut digests = vec![world.world_digest_v1().expect("reference launch digest")];
    let mut frames = Vec::new();
    for boundary in 0..5 {
        if boundary == 0 && !matches!(perturbation, SchedulePerturbation::SkipZero) {
            let mut config = world.config().clone();
            config.metabolism_drain = 0.01;
            world
                .apply_config_update(config)
                .expect("manual tick-zero patch");
        }
        if boundary == 1 {
            let rates = if matches!(perturbation, SchedulePerturbation::ReverseSameTick) {
                [0.03, 0.02]
            } else {
                [0.02, 0.03]
            };
            for rate in rates {
                let mut config = world.config().clone();
                config.metabolism_drain = rate;
                config.food_growth_rate = 0.005;
                world
                    .apply_config_update(config)
                    .expect("manual ordered tick-one patch");
            }
        }
        if boundary
            == if matches!(perturbation, SchedulePerturbation::DelayRecovery) {
                4
            } else {
                3
            }
        {
            let mut config = world.config().clone();
            config.food_growth_rate = 0.03;
            config.metabolism_drain = 0.007;
            world
                .apply_config_update(config)
                .expect("manual later patch");
        }
        world.step().expect("manual scientific transition");
        frames.push(serde_json::json!({
            "tick": world.tick().0,
            "epoch": world.epoch(),
            "agent_count": world.agents().len(),
            "spike_hits": world.last_spike_hits(),
        }));
        digests.push(world.world_digest_v1().expect("reference boundary digest"));
    }
    assert_eq!(
        world.config().food_growth_rate.to_bits(),
        0.03_f32.to_bits()
    );
    assert_eq!(
        world.config().metabolism_drain.to_bits(),
        0.007_f32.to_bits()
    );
    (frames, digests)
}

fn write_unsorted_schedule(scenario: &Path) {
    fs::write(
        scenario,
        r#"
schema = "scriptbots.scenario.v1"
schema_version = 1
id = "bootstrap-schedule-parity"
description = "Unsorted boundary patches with ordered tick-one overrides."
seeds = [4242]
[config]
rng_seed = 4242
world_width = 600
world_height = 600
food_cell_size = 50
population_minimum = 0
population_spawn_interval = 0
persistence_interval = 1
history_capacity = 600
metabolism_drain = 0.001
food_growth_rate = 0.01
[[interventions]]
tick = 3
set = { food_growth_rate = 0.03, metabolism_drain = 0.007 }
[[interventions]]
tick = 1
set = { metabolism_drain = 0.02, food_growth_rate = 0.005 }
[[interventions]]
tick = 0
set = { metabolism_drain = 0.01 }
[[interventions]]
tick = 1
set = { metabolism_drain = 0.03, food_growth_rate = 0.005 }
"#,
    )
    .expect("write unsorted scenario");
}

fn launch_scheduled_terminal(
    directory: &Path,
    scenario: &Path,
) -> (serde_json::Value, serde_json::Value, String) {
    let database = directory.join("scheduled.sqlite");
    let report_path = directory.join("headless.json");
    let output = headless_command()
        .env("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", "3")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS_REPORT", &report_path)
        .env("SCRIPTBOTS_STORAGE_PATH", &database)
        .env("RUST_LOG", "info")
        .args([
            "--storage",
            "file",
            "--threads",
            "1",
            "--brain",
            "mlp",
            "--bootstrap-ticks",
            "2",
            "--scenario",
        ])
        .arg(scenario)
        .output()
        .expect("launch scheduled headless binary");
    let stderr = strip_ansi(&String::from_utf8_lossy(&output.stderr));
    eprintln!(
        "scheduled CLI status={}\nstdout:\n{}\nstderr:\n{stderr}",
        output.status,
        String::from_utf8_lossy(&output.stdout)
    );
    fs::write(directory.join("terminal.stdout.log"), &output.stdout)
        .expect("retain terminal stdout");
    fs::write(directory.join("terminal.stderr.log"), &output.stderr)
        .expect("retain terminal stderr");
    assert!(output.status.success(), "scheduled CLI failed:\n{stderr}");
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(&report_path).expect("actual headless report"))
            .expect("report JSON");
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(database.with_extension("manifest.json")).expect("actual bootstrap manifest"),
    )
    .expect("manifest JSON");
    (report, manifest, stderr)
}

fn assert_scheduled_terminal_trace(
    report: &serde_json::Value,
    manifest: &serde_json::Value,
    expected_frames: &[serde_json::Value],
    expected_digests: &[scriptbots_core::WorldDigestV1],
) {
    assert_eq!(report["initial"]["tick"], 2);
    assert_eq!(report["summary"]["ticks_simulated"], 3);
    assert_eq!(report["summary"]["final_tick"], 5);
    let frames = report["frames"].as_array().expect("actual frame trace");
    assert_eq!(frames.len(), 3);
    for (actual, expected) in frames.iter().zip(&expected_frames[2..]) {
        for field in ["tick", "epoch", "agent_count", "spike_hits"] {
            assert_eq!(actual[field], expected[field], "frame field {field}");
        }
    }
    let bootstrap = &manifest["bootstrap_evidence"];
    assert_eq!(bootstrap["completed"], 2);
    assert_eq!(
        bootstrap["start"],
        serde_json::to_value(&expected_digests[0]).expect("start digest JSON")
    );
    assert_eq!(
        bootstrap["end"],
        serde_json::to_value(&expected_digests[2]).expect("bootstrap digest JSON")
    );
    assert_eq!(
        report["summary"]["world_digest"],
        expected_digests[5].overall
    );
}

fn assert_scheduled_determinism_trace(
    directory: &Path,
    scenario: &Path,
    expected_frames: &[serde_json::Value],
    expected_digests: &[scriptbots_core::WorldDigestV1],
) {
    // The separate verification entry point must execute the same schedule too;
    // its output is another subject, never the oracle for the production run.
    let det_output = headless_command()
        .env("SCRIPTBOTS_DET_RUN", "1")
        .env("SCRIPTBOTS_DET_TICKS", "5")
        .env("RUST_LOG", "info")
        .args(["--threads", "1", "--brain", "mlp", "--scenario"])
        .arg(scenario)
        .output()
        .expect("launch actual determinism child entry point");
    eprintln!(
        "determinism CLI status={}\nstdout:\n{}\nstderr:\n{}",
        det_output.status,
        String::from_utf8_lossy(&det_output.stdout),
        String::from_utf8_lossy(&det_output.stderr)
    );
    fs::write(directory.join("det.stdout.log"), &det_output.stdout)
        .expect("retain determinism stdout");
    fs::write(directory.join("det.stderr.log"), &det_output.stderr)
        .expect("retain determinism stderr");
    assert!(
        det_output.status.success(),
        "determinism entry point failed"
    );
    let det: serde_json::Value =
        serde_json::from_slice(&det_output.stdout).expect("actual determinism JSON");
    assert_eq!(det["ticks"], 5);
    assert_eq!(det["last_tick"], 5);
    assert_eq!(
        det["world_digest"],
        serde_json::to_value(&expected_digests[5]).expect("final digest JSON")
    );
    let summaries = det["summaries"]
        .as_array()
        .expect("actual deterministic summaries");
    assert_eq!(summaries.len(), expected_frames.len());
    for (actual, expected) in summaries.iter().zip(expected_frames) {
        for field in ["tick", "agent_count", "spike_hits"] {
            assert_eq!(
                actual[field], expected[field],
                "determinism trace field {field}"
            );
        }
    }
}

fn assert_scheduled_patch_logs(stderr: &str) {
    let applied: Vec<_> = stderr
        .lines()
        .filter(|line| line.contains("scheduled patch applied"))
        .collect();
    assert_eq!(
        applied.len(),
        4,
        "each actual patch must be reported exactly once:\n{stderr}"
    );
    for (line, (sequence, tick, changed)) in applied.iter().zip([
        (1, 0, "metabolism_drain"),
        (2, 1, "food_growth_rate, metabolism_drain"),
        (3, 1, "metabolism_drain"),
        (4, 3, "food_growth_rate, metabolism_drain"),
    ]) {
        assert!(
            line.contains(&format!("sequence={sequence}"))
                && line.contains(&format!("tick={tick}")),
            "wrong applied boundary: {line}"
        );
        assert!(
            line.contains(&format!("changed={changed}")),
            "wrong observed changed paths: {line}"
        );
    }
}

#[test]
fn headless_scenario_schedule_matches_manual_science_across_bootstrap() {
    let directory = tempdir().expect("scheduled CLI artifacts").keep();
    eprintln!(
        "scheduled CLI artifacts retained at {}",
        directory.display()
    );
    let scenario = directory.join("scheduled.scenario.toml");
    write_unsorted_schedule(&scenario);
    let (report, manifest, stderr) = launch_scheduled_terminal(&directory, &scenario);
    let (expected_frames, expected_digests) = schedule_reference(SchedulePerturbation::None);
    assert_scheduled_terminal_trace(&report, &manifest, &expected_frames, &expected_digests);
    assert_scheduled_determinism_trace(&directory, &scenario, &expected_frames, &expected_digests);
    for (label, perturbation) in [
        ("omitted tick zero", SchedulePerturbation::SkipZero),
        (
            "reversed same-tick order",
            SchedulePerturbation::ReverseSameTick,
        ),
        ("late recovery", SchedulePerturbation::DelayRecovery),
    ] {
        let (_, wrong) = schedule_reference(perturbation);
        assert_ne!(
            wrong[5].overall, expected_digests[5].overall,
            "the digest oracle must detect {label}"
        );
    }
    assert_scheduled_patch_logs(&stderr);
}

#[test]
fn occupied_rest_port_refuses_before_config_tuning_or_storage() {
    let occupied =
        TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).expect("occupy REST control port");
    let address = occupied.local_addr().expect("occupied REST address");
    let temp_dir = tempdir().expect("startup preflight directory");
    let storage_path = temp_dir.path().join("must-not-create-rest.sqlite");
    let config_path = temp_dir.path().join("must-not-write-rest.json");
    let tuning_dir = temp_dir.path().join("rest-tuning");
    fs::create_dir(&tuning_dir).expect("auto-tune scratch directory");

    let output = headless_command()
        .env("SCRIPTBOTS_CONTROL_REST_ENABLED", "1")
        .env("SCRIPTBOTS_CONTROL_REST_ADDR", address.to_string())
        .env("SCRIPTBOTS_CONTROL_MCP", "disabled")
        .env("SCRIPTBOTS_STORAGE_PATH", &storage_path)
        .env("TMPDIR", &tuning_dir)
        .args([
            "--auto-tune",
            "1",
            "--write-config",
            config_path.to_str().expect("UTF-8 config path"),
        ])
        .output()
        .expect("launch with occupied REST port");

    assert!(!output.status.success(), "occupied REST port must fail");
    let stderr = strip_ansi(&String::from_utf8_lossy(&output.stderr));
    assert!(
        stderr.contains("failed to reserve REST address") && stderr.contains(&address.to_string()),
        "unexpected REST preflight error:\n{stderr}"
    );
    assert_no_startup_artifacts(&storage_path, &config_path, &tuning_dir);
}

#[test]
fn occupied_mcp_port_refuses_before_config_tuning_or_storage() {
    let occupied =
        TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).expect("occupy MCP control port");
    let address = occupied.local_addr().expect("occupied MCP address");
    let temp_dir = tempdir().expect("startup preflight directory");
    let storage_path = temp_dir.path().join("must-not-create-mcp.sqlite");
    let config_path = temp_dir.path().join("must-not-write-mcp.json");
    let tuning_dir = temp_dir.path().join("mcp-tuning");
    fs::create_dir(&tuning_dir).expect("auto-tune scratch directory");

    let output = headless_command()
        .env("SCRIPTBOTS_CONTROL_REST_ENABLED", "0")
        .env("SCRIPTBOTS_CONTROL_MCP", "http")
        .env("SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR", address.to_string())
        .env("SCRIPTBOTS_STORAGE_PATH", &storage_path)
        .env("TMPDIR", &tuning_dir)
        .args([
            "--auto-tune",
            "1",
            "--write-config",
            config_path.to_str().expect("UTF-8 config path"),
        ])
        .output()
        .expect("launch with occupied MCP port");

    assert!(!output.status.success(), "occupied MCP port must fail");
    let stderr = strip_ansi(&String::from_utf8_lossy(&output.stderr));
    assert!(
        stderr.contains("failed to reserve MCP HTTP address")
            && stderr.contains(&address.to_string()),
        "unexpected MCP preflight error:\n{stderr}"
    );
    assert_no_startup_artifacts(&storage_path, &config_path, &tuning_dir);
}

#[cfg(any(not(feature = "gui"), not(feature = "bevy_render")))]
fn assert_uncompiled_renderer_refuses_before_storage_reservation(mode: &str, expected_error: &str) {
    let temp_dir = tempdir().expect("temp storage directory");
    let storage_path = temp_dir.path().join("must-not-be-created.sqlite");
    let tuning_dir = temp_dir.path().join("auto-tune-temp");
    let config_path = temp_dir.path().join("must-not-be-written.json");
    fs::create_dir(&tuning_dir).expect("auto-tune temp directory");

    let mut cmd = headless_command();
    let output = cmd
        .env("SCRIPTBOTS_STORAGE_PATH", &storage_path)
        .env("TMPDIR", &tuning_dir)
        .args([
            "--mode",
            mode,
            "--bootstrap-ticks",
            "0",
            "--auto-tune",
            "1",
            "--write-config",
            config_path.to_str().expect("UTF-8 test config path"),
        ])
        .output()
        .expect("failed to run scriptbots-app binary");

    assert!(
        !output.status.success(),
        "uncompiled {mode} request must fail"
    );
    let stderr = strip_ansi(&String::from_utf8_lossy(&output.stderr));
    assert!(
        stderr.contains(expected_error),
        "expected precise unavailable-feature error; stderr:\n{stderr}"
    );
    assert!(
        !storage_path.exists(),
        "renderer preflight must fail before reserving FrankenSQLite storage"
    );
    let writer_lock = PathBuf::from(format!(
        "{}{}",
        storage_path.display(),
        ".scriptbots-writer.lock"
    ));
    assert!(
        !writer_lock.exists(),
        "renderer preflight must not create the persistent writer-lock companion"
    );
    assert!(
        !config_path.exists(),
        "renderer preflight must reject the request before writing configuration"
    );
    for suffix in STORAGE_SIDECAR_SUFFIXES {
        let sidecar = PathBuf::from(format!("{}{suffix}", storage_path.display()));
        assert!(
            !sidecar.exists(),
            "renderer preflight created unexpected sidecar {}",
            sidecar.display()
        );
    }
    assert!(
        fs::read_dir(&tuning_dir)
            .expect("read auto-tune temp directory")
            .next()
            .is_none(),
        "renderer preflight must reject the request before the auto-tuning sweep"
    );
}

#[cfg(not(feature = "gui"))]
#[test]
fn explicit_uncompiled_gui_refuses_before_storage_reservation() {
    assert_uncompiled_renderer_refuses_before_storage_reservation(
        "gui",
        "--mode gui requires a binary built with --features gui",
    );
}

#[cfg(not(feature = "bevy_render"))]
#[test]
fn explicit_uncompiled_bevy_refuses_before_storage_reservation() {
    assert_uncompiled_renderer_refuses_before_storage_reservation(
        "bevy",
        "--mode bevy requires a binary built with --features bevy_render",
    );
}

fn strip_ansi(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            if let Some('[') = chars.next() {
                for code in chars.by_ref() {
                    if ('@'..='~').contains(&code) {
                        break;
                    }
                }
            }
            continue;
        }
        result.push(ch);
    }
    result
}
