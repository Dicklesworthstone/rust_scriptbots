//! Real-process control-plane E2E for bd-0n87 (criterion 8 of bd-88yj).
//!
//! Every other control-plane proof in this workspace builds a `ControlRuntime`
//! in-process. That is the adjacent thing: it skips `main.rs` wiring, CLI and env
//! config parsing, renderer-mode resolution, and process lifecycle — which is
//! precisely the surface a user meets first. This drives the SHIPPED BINARY.
//!
//! NOTHING HERE IS STUBBED. A real child process, its real REST listener on an
//! OS-assigned port, real HTTP over a real socket, and child kill/reap at the end.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serial_test::serial;
use tempfile::tempdir;

/// The shipped binary, following the house convention.
fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_scriptbots-app"))
}

/// Minimal HTTP/1.1 with optional body over a real socket.
fn http_with_body(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body_bytes: &[u8],
    content_type: Option<&str>,
) -> Result<(u16, String)> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(20)))?;
    let ct_header = match content_type {
        Some(ct) => format!("Content-Type: {ct}\r\n"),
        None => String::new(),
    };
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\n{ct_header}Content-Length: {}\r\nConnection: close\r\n\r\n",
        body_bytes.len()
    )?;
    if !body_bytes.is_empty() {
        stream.write_all(body_bytes)?;
    }
    stream.flush()?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| anyhow!("malformed HTTP response: {text:?}"))?;
    let status: u16 = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1).map(str::to_string))
        .ok_or_else(|| anyhow!("no status line in {head:?}"))?
        .parse()?;
    Ok((status, body.to_string()))
}

fn http(addr: SocketAddr, method: &str, path: &str) -> Result<(u16, String)> {
    http_with_body(addr, method, path, &[], None)
}

/// Pull a string field out of a small JSON object without a parser dependency.
fn json_str(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":\"");
    let start = body.find(&needle)? + needle.len();
    let rest = &body[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extract the bound address from the listener announcement.
///
/// Deliberately NOT a `split_whitespace().strip_prefix("address=")` match. The
/// tracing formatter colours field NAMES independently, so on a colour-enabled
/// child the token is `\x1b[..maddress\x1b[0m=127.0.0.1:38835` and a prefix match
/// silently never fires — the harness then reads the child's entire log without
/// recognising the line it was waiting for, and reports a timeout that looks like
/// "the server never started" when the server started in 0.4s. Strip ANSI first,
/// then search anywhere in the line.
fn parse_announced_address(line: &str) -> Option<SocketAddr> {
    let plain: String = {
        let mut out = String::with_capacity(line.len());
        let mut chars = line.chars();
        while let Some(ch) = chars.next() {
            if ch == '\u{1b}' {
                for esc in chars.by_ref() {
                    if esc.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(ch);
            }
        }
        out
    };
    let start = plain.find("address=")? + "address=".len();
    let rest = &plain[start..];
    let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
    rest[..end].parse::<SocketAddr>().ok()
}

/// Kill the child and reap it, so a failing assertion cannot leak a process that
/// keeps a port bound and wedges the next run.
struct ChildGuard(Option<Child>);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Read the child's stderr until both REST and MCP listeners announce their bound addresses.
fn wait_for_control_addresses(
    child: &mut Child,
    timeout: Duration,
) -> Result<(SocketAddr, SocketAddr, Vec<String>)> {
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("child stderr was not captured"))?;

    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(std::result::Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    let deadline = Instant::now() + timeout;
    let mut log = Vec::new();
    let mut rest_addr: Option<SocketAddr> = None;
    let mut mcp_addr: Option<SocketAddr> = None;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!(
                "timed out after {timeout:?} waiting for REST/MCP listeners; child alive={}; \
                 rest={:?}, mcp={:?}; last {} stderr lines:\n{}",
                child.try_wait().ok().flatten().is_none(),
                rest_addr,
                mcp_addr,
                log.len().min(40),
                log.iter()
                    .rev()
                    .take(40)
                    .rev()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
        match rx.recv_timeout(remaining) {
            Ok(line) => {
                let trimmed = line.trim_end().to_string();
                log.push(trimmed.clone());
                if trimmed.contains("REST control server listening")
                    && let Some(parsed) = parse_announced_address(&trimmed)
                {
                    rest_addr = Some(parsed);
                }
                if trimmed.contains("MCP HTTP server listening")
                    && let Some(parsed) = parse_announced_address(&trimmed)
                {
                    mcp_addr = Some(parsed);
                }
                if let (Some(rest), Some(mcp)) = (rest_addr, mcp_addr) {
                    return Ok((rest, mcp, log));
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => bail!(
                "child stderr closed before announcing listeners; child alive={}; \
                 rest={:?}, mcp={:?}; last {} lines:\n{}",
                child.try_wait().ok().flatten().is_none(),
                rest_addr,
                mcp_addr,
                log.len().min(40),
                log.iter()
                    .rev()
                    .take(40)
                    .rev()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
        }
    }
}

/// THE REAL-PROCESS CONTROL-PLANE ACCEPTANCE PROBE (bd-6mus, bd-0n87).
///
/// Asserts wire-level contracts against the real SHIPPED BINARY:
///  1. `--mode server --storage memory` boots and binds OS-assigned REST and MCP ports.
///  2. GET /api-docs/openapi.json exposes >=29 paths; GET /api/knobs >=100 entries;
///     GET /api/status returns live tick/founding population.
///  3. POST /api/control/pause reaches `applied` and freezes ticks (f2 == f1).
///  4. POST /api/control/step {count:1} reaches `applied`, advances tick by exactly +1,
///     and leaves the world paused.
///  5. POST /api/control/resume reaches `applied` and unfreezes ticks.
///  6. Negative paths: malformed step body -> 400; unknown command status -> 404.
///  7. GET /api/screenshot/ascii refuses with 409 in unpresented server mode.
///  8. FastMCP HTTP endpoint: GET /health -> 200; POST /mcp initialize -> 200 with negotiated
///     protocolVersion; POST /mcp notifications/initialized -> 202; POST /mcp tools/list -> 200
///     with all 13 tools; POST /mcp tools/call get_status -> 200 with live tick;
///     POST /mcp tools/call unknown -> JSON-RPC error.
///  9. Process lifecycle: process remains alive throughout, then the test kills and reaps it.
#[test]
#[serial]
fn real_process_server_mode_applies_commands_and_refuses_an_unpresented_screenshot() -> Result<()> {
    let run_dir = tempdir()?;

    let mut child = Command::new(binary())
        .args(["--mode", "server", "--storage", "memory"])
        .env("SCRIPTBOTS_CONTROL_REST_ENABLED", "1")
        // Port 0: the OS assigns and the process announces what it bound.
        .env("SCRIPTBOTS_CONTROL_REST_ADDR", "127.0.0.1:0")
        .env("SCRIPTBOTS_CONTROL_MCP", "http")
        .env("SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR", "127.0.0.1:0")
        // Narrow the filter deliberately: at bare `info` the fsqlite statement-reuse
        // telemetry emits thousands of lines and the listener announcement is
        // drowned in them, which is how the first run of this test timed out.
        .env("RUST_LOG", "warn,scriptbots_app=info")
        .current_dir(run_dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn the shipped binary")?;

    let (rest_addr, mcp_addr, boot_log) =
        match wait_for_control_addresses(&mut child, Duration::from_secs(90)) {
            Ok(found) => found,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };
    let mut guard = ChildGuard(Some(child));

    // (1) REST /api/status is reachable and reports live world state.
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut status_val: serde_json::Value = serde_json::Value::Null;
    let mut status_code = 0;
    while Instant::now() < deadline {
        if let Ok((code, body)) = http(rest_addr, "GET", "/api/status") {
            status_code = code;
            if code == 200
                && let Ok(v) = serde_json::from_str(&body)
            {
                status_val = v;
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(
        status_code, 200,
        "the shipped binary must serve /api/status in --mode server, got {status_code}"
    );
    assert!(
        status_val["agent_count"].as_u64().unwrap_or(0) >= 1,
        "world must report a live founding population; got: {status_val:?}"
    );

    // (2) REST metadata reads: OpenAPI and Knobs
    let (open_code, open_body) = http(rest_addr, "GET", "/api-docs/openapi.json")?;
    assert_eq!(open_code, 200, "openapi endpoint must return 200");
    let open_json: serde_json::Value = serde_json::from_str(&open_body)?;
    let paths_count = open_json["paths"].as_object().map_or(0, |p| p.len());
    assert!(
        paths_count >= 29,
        "openapi must publish >= 29 paths, found {paths_count}"
    );

    let (knobs_code, knobs_body) = http(rest_addr, "GET", "/api/knobs")?;
    assert_eq!(knobs_code, 200, "knobs endpoint must return 200");
    let knobs_json: serde_json::Value = serde_json::from_str(&knobs_body)?;
    let knobs_count = knobs_json.as_array().map_or(0, |a| a.len());
    assert!(
        knobs_count >= 100,
        "knobs roster must publish >= 100 knobs, found {knobs_count}"
    );

    // (3) Two-axis playback semantics: Pause
    let (pause_code, pause_body) = http(rest_addr, "POST", "/api/control/pause")?;
    assert_eq!(
        pause_code, 200,
        "pause must be accepted, got {pause_code}: {pause_body}"
    );
    let pause_id = json_str(&pause_body, "command_id")
        .ok_or_else(|| anyhow!("pause response carried no command_id: {pause_body}"))?;

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut pause_app_state = String::new();
    let mut journal_state = String::new();
    let mut receipt_body = String::new();
    while Instant::now() < deadline {
        let (code, body) = http(rest_addr, "GET", &format!("/api/control/status/{pause_id}"))?;
        if code == 200 {
            receipt_body = body.clone();
            pause_app_state = json_str(&body, "application_state").unwrap_or_default();
            journal_state = json_str(&body, "journal_state").unwrap_or_default();
            if pause_app_state == "applied" {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    assert_eq!(
        pause_app_state, "applied",
        "the pause command must reach the WORLD, not merely the queue; receipt: {receipt_body}"
    );
    assert!(
        !journal_state.is_empty(),
        "the receipt must report a journal state, even if it is not_required: {receipt_body}"
    );

    // Verify ticks are frozen
    std::thread::sleep(Duration::from_millis(150));
    let (_, b1) = http(rest_addr, "GET", "/api/status")?;
    let f1: serde_json::Value = serde_json::from_str(&b1)?;
    let tick1 = f1["tick"].as_u64().expect("tick u64");
    std::thread::sleep(Duration::from_millis(150));
    let (_, b2) = http(rest_addr, "GET", "/api/status")?;
    let f2: serde_json::Value = serde_json::from_str(&b2)?;
    let tick2 = f2["tick"].as_u64().expect("tick u64");
    assert_eq!(
        tick2, tick1,
        "pause must freeze world ticks; got {tick1} then {tick2}"
    );

    // (4) Single step: exactly one tick, remains paused
    let (step_code, step_body) = http_with_body(
        rest_addr,
        "POST",
        "/api/control/step",
        b"{\"count\":1}",
        Some("application/json"),
    )?;
    assert_eq!(step_code, 200, "step must be accepted: {step_body}");
    let step_id = json_str(&step_body, "command_id")
        .ok_or_else(|| anyhow!("step response carried no command_id: {step_body}"))?;

    let step_deadline = Instant::now() + Duration::from_secs(30);
    let mut step_app_state = String::new();
    while Instant::now() < step_deadline {
        let (code, body) = http(rest_addr, "GET", &format!("/api/control/status/{step_id}"))?;
        if code == 200 {
            step_app_state = json_str(&body, "application_state").unwrap_or_default();
            if step_app_state == "applied" {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    assert_eq!(
        step_app_state, "applied",
        "step command must reach applied state"
    );

    let (_, b_step) = http(rest_addr, "GET", "/api/status")?;
    let v_step: serde_json::Value = serde_json::from_str(&b_step)?;
    let tick_step = v_step["tick"].as_u64().expect("tick u64");
    assert_eq!(
        tick_step,
        tick2 + 1,
        "step count 1 must advance tick by exactly 1"
    );

    std::thread::sleep(Duration::from_millis(150));
    let (_, b_step2) = http(rest_addr, "GET", "/api/status")?;
    let v_step2: serde_json::Value = serde_json::from_str(&b_step2)?;
    let tick_step2 = v_step2["tick"].as_u64().expect("tick u64");
    assert_eq!(tick_step2, tick_step, "stepped world must remain paused");

    // (5) Resume: unfreezes the world
    let (resume_code, resume_body) = http(rest_addr, "POST", "/api/control/resume")?;
    assert_eq!(resume_code, 200, "resume must be accepted: {resume_body}");
    let resume_id = json_str(&resume_body, "command_id")
        .ok_or_else(|| anyhow!("resume response carried no command_id: {resume_body}"))?;

    let res_deadline = Instant::now() + Duration::from_secs(30);
    let mut res_app_state = String::new();
    while Instant::now() < res_deadline {
        let (code, body) = http(
            rest_addr,
            "GET",
            &format!("/api/control/status/{resume_id}"),
        )?;
        if code == 200 {
            res_app_state = json_str(&body, "application_state").unwrap_or_default();
            if res_app_state == "applied" {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    assert_eq!(
        res_app_state, "applied",
        "resume command must reach applied state"
    );

    let advance_deadline = Instant::now() + Duration::from_secs(30);
    let mut advanced = false;
    while Instant::now() < advance_deadline {
        if let Ok((code, body)) = http(rest_addr, "GET", "/api/status") {
            if code == 200
                && let Ok(v) = serde_json::from_str::<serde_json::Value>(&body)
            {
                if let Some(t) = v["tick"].as_u64() {
                    if t > tick_step {
                        advanced = true;
                        break;
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        advanced,
        "resume must unfreeze world ticks past {tick_step}"
    );

    // (6) Negative paths: malformed step and unknown command status
    let (bad_step_code, _) = http_with_body(
        rest_addr,
        "POST",
        "/api/control/step",
        b"not-json",
        Some("application/json"),
    )?;
    assert_eq!(bad_step_code, 400, "malformed step payload must return 400");

    let (not_found_code, _) = http(rest_addr, "GET", "/api/control/status/no-such-command-xyz")?;
    assert_eq!(
        not_found_code, 404,
        "unknown command status must return 404"
    );

    // (7) Terminal unpresented screenshot refusal
    let (shot_code, shot_body) = http(rest_addr, "GET", "/api/screenshot/ascii")?;
    assert_eq!(
        shot_code, 409,
        "--mode server presents no terminal frame, so the endpoint must refuse \
         rather than substitute a re-rasterized world map; got {shot_code}: {shot_body}"
    );
    assert!(
        shot_body.contains("no terminal frame has been presented"),
        "the refusal must explain itself: {shot_body}"
    );
    assert!(
        shot_body.contains("re-rasterized"),
        "and must name what it is deliberately not doing, so the fallback is not \
         restored as a convenience: {shot_body}"
    );

    // (8) FastMCP HTTP protocol verification
    let (health_code, health_body) = http(mcp_addr, "GET", "/health")?;
    assert_eq!(health_code, 200, "MCP /health must return 200");
    assert!(
        health_body.contains("healthy"),
        "MCP /health body: {health_body}"
    );

    // MCP initialize (protocolVersion 2024-11-05)
    let init_payload = br#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"e2e-control-plane","version":"0"}}}"#;
    let (init_code, init_body) = http_with_body(
        mcp_addr,
        "POST",
        "/mcp",
        init_payload,
        Some("application/json"),
    )?;
    assert_eq!(
        init_code, 200,
        "MCP initialize must return 200: {init_body}"
    );
    let init_json: serde_json::Value = serde_json::from_str(&init_body)?;
    assert_eq!(
        init_json["result"]["protocolVersion"], "2024-11-05",
        "MCP must negotiate protocolVersion: {init_body}"
    );

    // MCP 2024-11-05 lifecycle, Version Negotiation: an unsupported requested version
    // receives a server-supported alternative, not an echo of the unsupported value.
    // https://modelcontextprotocol.io/specification/2024-11-05/basic/lifecycle
    let future_version_payload = br#"{"jsonrpc":"2.0","id":99,"method":"initialize","params":{"protocolVersion":"2099-01-01","capabilities":{},"clientInfo":{"name":"e2e-control-plane","version":"0"}}}"#;
    let (future_version_code, future_version_body) = http_with_body(
        mcp_addr,
        "POST",
        "/mcp",
        future_version_payload,
        Some("application/json"),
    )?;
    assert_eq!(
        future_version_code, 200,
        "MCP version negotiation must return a response: {future_version_body}"
    );
    let future_version: serde_json::Value = serde_json::from_str(&future_version_body)?;
    assert_eq!(future_version["id"], 99);
    assert!(future_version.get("error").is_none());
    assert_eq!(
        future_version["result"]["protocolVersion"], "2024-11-05",
        "MCP must choose its supported version: {future_version_body}"
    );

    // A malformed version TYPE is invalid initialize input, regardless of negotiation.
    let invalid_version_payload = br#"{"jsonrpc":"2.0","id":100,"method":"initialize","params":{"protocolVersion":42,"capabilities":{},"clientInfo":{"name":"e2e-control-plane","version":"0"}}}"#;
    let (invalid_version_code, invalid_version_body) = http_with_body(
        mcp_addr,
        "POST",
        "/mcp",
        invalid_version_payload,
        Some("application/json"),
    )?;
    assert_eq!(invalid_version_code, 200);
    let invalid_version: serde_json::Value = serde_json::from_str(&invalid_version_body)?;
    assert_eq!(invalid_version["id"], 100);
    assert!(invalid_version.get("result").is_none());
    assert_eq!(
        invalid_version["error"]["code"], -32602,
        "MCP malformed initialize parameters must be rejected: {invalid_version_body}"
    );
    assert!(
        !invalid_version["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .is_empty(),
        "MCP malformed-input refusal must include a diagnostic: {invalid_version_body}"
    );

    // MCP notifications/initialized
    let notify_payload = br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
    let (notify_code, _) = http_with_body(
        mcp_addr,
        "POST",
        "/mcp",
        notify_payload,
        Some("application/json"),
    )?;
    assert_eq!(
        notify_code, 202,
        "MCP notifications/initialized must return 202"
    );

    // MCP tools/list
    let list_payload = br#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;
    let (list_code, list_body) = http_with_body(
        mcp_addr,
        "POST",
        "/mcp",
        list_payload,
        Some("application/json"),
    )?;
    assert_eq!(
        list_code, 200,
        "MCP tools/list must return 200: {list_body}"
    );
    let list_json: serde_json::Value = serde_json::from_str(&list_body)?;
    let tools = list_json["result"]["tools"]
        .as_array()
        .expect("tools array");
    assert_eq!(tools.len(), 13, "MCP tools/list must return all 13 tools");

    // MCP tools/call get_status
    let status_payload = br#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_status","arguments":{}}}"#;
    let (status_call_code, status_call_body) = http_with_body(
        mcp_addr,
        "POST",
        "/mcp",
        status_payload,
        Some("application/json"),
    )?;
    assert_eq!(
        status_call_code, 200,
        "MCP tools/call get_status must return 200: {status_call_body}"
    );
    assert!(
        status_call_body.contains("tick"),
        "MCP get_status must report tick: {status_call_body}"
    );

    // MCP tools/call unknown tool -> JSON-RPC error
    let unknown_payload = br#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"no_such_tool","arguments":{}}}"#;
    let (unknown_code, unknown_body) = http_with_body(
        mcp_addr,
        "POST",
        "/mcp",
        unknown_payload,
        Some("application/json"),
    )?;
    assert_eq!(
        unknown_code, 200,
        "MCP unknown tool call returns JSON-RPC error response"
    );
    let unknown_json: serde_json::Value = serde_json::from_str(&unknown_body)?;
    assert!(
        unknown_json["error"].is_object(),
        "MCP unknown tool must return JSON-RPC error object: {unknown_body}"
    );

    // (9) Lifecycle: observe a live child, then deliberately kill and reap it.
    let child = guard.0.as_mut().expect("child still held");
    assert!(
        child.try_wait()?.is_none(),
        "--mode server must still be running after the assertions; it exited early"
    );
    child.kill()?;
    let exit = child.wait()?;
    guard.0 = None;

    let commit = Command::new("git")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!(
        "{{\"schema\":\"scriptbots.real-process-e2e.v2\",\"binary\":\"{}\",\"mode\":\"server\",\
         \"storage\":\"memory\",\"rest_address\":\"{rest_addr}\",\"mcp_address\":\"{mcp_addr}\",\
         \"boot_log_lines\":{},\"status_code\":{status_code},\"pause_code\":{pause_code},\
         \"pause_id\":\"{pause_id}\",\"step_id\":\"{step_id}\",\"resume_id\":\"{resume_id}\",\
         \"application_state\":\"applied\",\"journal_state\":\"{journal_state}\",\
         \"proved_level\":\"applied\",\"screenshot_code\":{shot_code},\
         \"tools_count\":{},\"child_exit\":\"{}\",\"source_commit\":\"{commit}\"}}",
        binary().display(),
        boot_log.len(),
        tools.len(),
        exit.code()
            .map_or_else(|| "signalled".to_string(), |code| code.to_string()),
    );

    Ok(())
}

/// The original bd-w1oi command must keep advancing for the full 600-second
/// reproduction window, beyond both observed failure ticks (240 and 420).
/// Retain stderr and every REST observation, including on failure. This replaces
/// the former diagnostic whose success meant that the bug had reproduced.
#[test]
#[serial]
#[ignore = "bd-w1oi live server regression: 600 seconds per storage backend. Run explicitly: cargo test -p scriptbots-app \
            --test real_process_control_e2e -- --ignored bd_w1oi --nocapture"]
fn bd_w1oi_server_mode_keeps_advancing_with_memory_storage() -> Result<()> {
    verify_server_progress("memory")
}

/// Exercise the same progress requirement against the production file backend.
#[test]
#[serial]
#[ignore = "bd-w1oi live server regression: 600 seconds per storage backend. Run explicitly: cargo test -p scriptbots-app \
            --test real_process_control_e2e -- --ignored bd_w1oi --nocapture"]
fn bd_w1oi_server_mode_keeps_advancing_with_file_storage() -> Result<()> {
    verify_server_progress("file")
}

/// Observe actual tick progress, not merely a live process or HTTP 200 responses.
fn verify_server_progress(storage: &str) -> Result<()> {
    let run_dir = tempfile::Builder::new()
        .prefix(&format!("server-progress-{storage}-"))
        .tempdir()?
        .keep();
    println!("server progress artifacts: {}", run_dir.display());
    let stderr_path = run_dir.join("server.stderr");
    let mut observations = std::fs::File::create(run_dir.join("status.jsonl"))?;
    let mut command = Command::new(binary());
    for (name, _) in std::env::vars_os() {
        if name.as_encoded_bytes().starts_with(b"SCRIPTBOTS_")
            || name.as_encoded_bytes().starts_with(b"SB_")
        {
            command.env_remove(name);
        }
    }
    command
        .args(["--mode", "server", "--storage", storage])
        .env("SCRIPTBOTS_CONTROL_REST_ENABLED", "1")
        .env("SCRIPTBOTS_CONTROL_REST_ADDR", "127.0.0.1:0")
        .env("SCRIPTBOTS_CONTROL_MCP", "disabled")
        .env("RUST_LOG", "warn,scriptbots_app=info")
        .current_dir(&run_dir)
        .stdout(std::fs::File::create(run_dir.join("server.stdout"))?)
        .stderr(std::fs::File::create(&stderr_path)?);
    std::fs::write(run_dir.join("command.txt"), format!("{command:?}\n"))?;
    let child = command
        .spawn()
        .context("failed to spawn the shipped binary")?;
    let mut guard = ChildGuard(Some(child));
    let started = Instant::now();
    let duration = Duration::from_secs(600);
    let progress_deadline = scriptbots_storage::StorageDeadlines::default().admission_ack;
    let mut address = None;
    let mut last_tick = 0;
    let mut first_tick = None;
    let mut last_advance = started;
    let mut samples = 0_u64;
    loop {
        assert!(
            guard.0.as_mut().expect("child held").try_wait()?.is_none(),
            "server exited early; artifacts: {}",
            run_dir.display()
        );
        let stderr = std::fs::read_to_string(&stderr_path)?;
        assert!(
            !stderr.contains("Simulation step failed in server mode"),
            "server stopped simulating; artifacts: {}\n{stderr}",
            run_dir.display()
        );
        if address.is_none() {
            address = stderr
                .lines()
                .filter(|line| line.contains("REST control server listening"))
                .find_map(parse_announced_address);
        }
        if let Some(address) = address {
            let (code, body) = http(address, "GET", "/api/status")?;
            let status: serde_json::Value = serde_json::from_str(&body)?;
            writeln!(
                observations,
                "{}",
                serde_json::json!({
                    "elapsed_ms": started.elapsed().as_millis(),
                    "http_status": code, "status": status,
                })
            )?;
            observations.flush()?;
            assert_eq!(code, 200, "status failed: {body}");
            let tick = status["tick"].as_u64().context("missing live tick")?;
            assert!(tick >= last_tick, "tick regressed: {last_tick} -> {tick}");
            first_tick.get_or_insert(tick);
            if tick > last_tick {
                last_tick = tick;
                last_advance = Instant::now();
            }
            samples += 1;
        } else {
            assert!(
                started.elapsed() < Duration::from_secs(90),
                "REST did not bind"
            );
        }
        assert!(
            last_advance.elapsed() < progress_deadline,
            "no tick progress for {progress_deadline:?} at tick {last_tick}; artifacts: {}",
            run_dir.display()
        );
        if started.elapsed() >= duration {
            break;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    assert!(
        last_tick > 420,
        "did not pass the prior failure tick: {last_tick}"
    );
    assert!(samples >= 2 && Some(last_tick) > first_tick);
    let child = guard.0.as_mut().expect("child held");
    child.kill()?;
    let exit = child.wait()?;
    guard.0 = None;
    let stderr = std::fs::read_to_string(&stderr_path)?;
    assert!(
        !stderr.contains("Simulation step failed in server mode"),
        "server stopped simulating before termination: {stderr}"
    );
    let result = serde_json::json!({
        "storage": storage, "elapsed_ms": started.elapsed().as_millis(),
        "first_tick": first_tick, "last_tick": last_tick, "samples": samples,
        "termination": "killed and reaped by test", "child_exit": exit.to_string(),
    });
    std::fs::write(
        run_dir.join("result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!(
        "server progress observed: {result}; artifacts: {}",
        run_dir.display()
    );
    Ok(())
}
