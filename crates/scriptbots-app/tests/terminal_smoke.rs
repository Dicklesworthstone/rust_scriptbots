use std::{env, process::Command};
use tempfile::tempdir;

fn headless_command() -> Command {
    let bin = env!("CARGO_BIN_EXE_scriptbots-app");
    let mut cmd = Command::new(bin);
    cmd.env("SCRIPTBOTS_MODE", "terminal")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS", "1")
        .env("SCRIPTBOTS_CONTROL_REST_ENABLED", "0")
        .env("SCRIPTBOTS_CONTROL_MCP", "disabled")
        .env("TERM", "xterm-256color")
        .env("RUST_LOG", "off");
    cmd
}

#[test]
fn terminal_headless_smoke() {
    let temp_dir = tempdir().expect("temp storage directory");
    let storage_path = temp_dir.path().join("scriptbots_headless.duckdb");

    let mut cmd = headless_command();
    cmd.env("SCRIPTBOTS_STORAGE_PATH", &storage_path);
    let status = cmd.status().expect("failed to run scriptbots-app binary");
    assert!(status.success(), "terminal headless run failed");
}

#[test]
fn terminal_headless_emits_bootstrap_metrics() {
    let temp_dir = tempdir().expect("temp storage directory");
    let storage_path = temp_dir.path().join("scriptbots_headless_report.duckdb");

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
        clean.contains("final_tick=120"),
        "expected final tick=120 in summary; stderr:\n{clean}"
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
