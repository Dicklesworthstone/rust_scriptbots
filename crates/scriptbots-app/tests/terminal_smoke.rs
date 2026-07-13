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
        clean.contains("Primed world and persisted initial summary"),
        "expected bootstrap summary; stderr:\n{clean}"
    );
    assert!(
        clean.contains("Terminal headless run completed"),
        "expected terminal completion log; stderr:\n{clean}"
    );
    assert!(
        clean.contains("final_tick=132"),
        "expected 120 bootstrap ticks plus 12 headless-renderer ticks; stderr:\n{clean}"
    );
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

#[cfg(not(feature = "gui"))]
#[test]
fn explicit_uncompiled_gui_refuses_before_storage_reservation() {
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
            "gui",
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
        "uncompiled GPUI request must fail"
    );
    let stderr = strip_ansi(&String::from_utf8_lossy(&output.stderr));
    assert!(
        stderr.contains("--mode gui requires a binary built with --features gui"),
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
