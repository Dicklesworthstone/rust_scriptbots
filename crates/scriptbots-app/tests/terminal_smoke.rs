use std::env;
use std::process::Command;

#[test]
fn terminal_headless_smoke() {
    let bin = env!("CARGO_BIN_EXE_scriptbots-app");
    let mut cmd = Command::new(bin);
    cmd.env("SCRIPTBOTS_MODE", "terminal")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS", "1")
        .env("SCRIPTBOTS_CONTROL_REST_ENABLED", "0")
        .env("SCRIPTBOTS_CONTROL_MCP", "disabled")
        .env("TERM", "xterm-256color")
        .env("RUST_LOG", "off");

    let status = cmd.status().expect("failed to run scriptbots-app binary");
    assert!(status.success(), "terminal headless run failed");
}
