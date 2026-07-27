//! Real-process control-plane E2E for bd-0n87 (criterion 8 of bd-88yj).
//!
//! Every other control-plane proof in this workspace builds a `ControlRuntime`
//! in-process. That is the adjacent thing: it skips `main.rs` wiring, CLI and env
//! config parsing, renderer-mode resolution, and process lifecycle — which is
//! precisely the surface a user meets first. This drives the SHIPPED BINARY.
//!
//! NOTHING HERE IS STUBBED. A real child process, its real REST listener on an
//! OS-assigned port, real HTTP over a real socket, and a real SIGTERM at the end.

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
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_scriptbots-app") {
        return PathBuf::from(path);
    }
    let mut path = std::env::current_exe().expect("test exe");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.join("scriptbots-app")
}

/// Minimal HTTP/1.1 over a real socket.
///
/// Hand-rolled rather than adding an HTTP client dependency: these are bare GETs
/// and one bodyless POST, and a new dev-dependency is an operator decision this
/// proof does not need.
fn http(addr: SocketAddr, method: &str, path: &str) -> Result<(u16, String)> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(20)))?;
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )?;
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

/// Read the child's stderr until the REST listener announces its bound address.
///
/// Port 0 means the OS assigns, so the address must be READ BACK rather than
/// assumed — a fixed port would collide with whatever else is running on a shared
/// machine, which is the failure mode that makes an E2E flaky rather than wrong.
fn wait_for_rest_address(
    child: &mut Child,
    timeout: Duration,
) -> Result<(SocketAddr, Vec<String>)> {
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("child stderr was not captured"))?;

    // Read on a THREAD and hand lines over a channel, rather than calling
    // read_line on this one. BufRead::read_line BLOCKS until a line or EOF, so a
    // deadline checked between reads cannot fire while the child is merely quiet —
    // the loop would wait forever instead of failing with a diagnosis. recv_timeout
    // makes the deadline real.
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
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!(
                "timed out after {timeout:?} waiting for the REST listener; child alive={}; \
                 last {} stderr lines:\n{}",
                child.try_wait().ok().flatten().is_none(),
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
                    return Ok((parsed, log));
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => bail!(
                "child stderr closed before announcing a REST listener; child alive={}; \
                 last {} lines:\n{}",
                child.try_wait().ok().flatten().is_none(),
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

/// THE REAL-PROCESS CONTROL-PLANE PROOF.
///
/// Asserts three things the in-process harnesses structurally cannot:
///
///  1. `--mode server` resolves and stays alive without a TTY, with its REST
///     listener bound and announced — i.e. main.rs wiring and mode resolution work.
///  2. A command submitted over a real socket reaches the WORLD. Server mode drains
///     and applies the bus, so the receipt advances past admission on its own.
///  3. `/api/screenshot/ascii` REFUSES with 409 in this mode. `ServerRenderer`
///     publishes no presented frame, so the endpoint has nothing to serve. This is
///     the exact situation where the old handler silently re-rasterized the world
///     and returned 200 with a synthesized map, so proving the refusal against the
///     SHIPPED BINARY is the end-to-end evidence that the substitution is gone —
///     not merely gone in a unit test's view of the code.
///
/// The 200 from `/api/status` is paired with the 409 deliberately: without it, a
/// 409 would be indistinguishable from "the server never came up".
#[test]
#[serial]
#[ignore = "bd-0n87: real-process lifecycle exceeds the rch 9m40s client timeout when it \
            carries the build too, so this has not yet been OBSERVED green. Run it \
            explicitly against a warm target: cargo test -p scriptbots-app --test \
            real_process_control_e2e -- --ignored --nocapture. Ignored rather than left \
            in the default lane because an unverified test that fails costs every other \
            pane, and asserting a pass I have not seen is the defect this bead family \
            exists to prevent."]
fn real_process_server_mode_applies_commands_and_refuses_an_unpresented_screenshot() -> Result<()> {
    let run_dir = tempdir()?;

    let mut child = Command::new(binary())
        .args(["--mode", "server", "--storage", "memory"])
        .env("SCRIPTBOTS_CONTROL_REST_ENABLED", "1")
        // Port 0: the OS assigns and the process announces what it bound.
        .env("SCRIPTBOTS_CONTROL_REST_ADDR", "127.0.0.1:0")
        .env("SCRIPTBOTS_CONTROL_MCP", "disabled")
        // Narrow the filter deliberately: at bare `info` the fsqlite statement-reuse
        // telemetry emits thousands of lines and the listener announcement is
        // drowned in them, which is how the first run of this test timed out.
        .env("RUST_LOG", "warn,scriptbots_app=info")
        .current_dir(run_dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn the shipped binary")?;

    let (addr, boot_log) = match wait_for_rest_address(&mut child, Duration::from_secs(90)) {
        Ok(found) => found,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
    };
    let mut guard = ChildGuard(Some(child));

    // (1) The control plane is genuinely up. Polled rather than slept on: the
    // listener is announced before the first tick, so a fixed sleep would either
    // be flaky or wastefully long.
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut status_body = String::new();
    let mut status_code = 0;
    while Instant::now() < deadline {
        if let Ok((code, body)) = http(addr, "GET", "/api/status") {
            status_code = code;
            status_body = body;
            if code == 200 {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(
        status_code, 200,
        "the shipped binary must serve /api/status in --mode server, got {status_code} \
         with body {status_body:?}"
    );

    // (2) A command submitted over the socket must reach the world.
    let (pause_code, pause_body) = http(addr, "POST", "/api/control/pause")?;
    assert_eq!(
        pause_code, 200,
        "pause must be accepted, got {pause_code}: {pause_body}"
    );
    let command_id = json_str(&pause_body, "command_id")
        .ok_or_else(|| anyhow!("pause response carried no command_id: {pause_body}"))?;

    // Server mode drains at a 16ms cadence, so application follows admission
    // quickly — but poll for it rather than assuming, because asserting the
    // SUBMITTED state and calling it applied is the exact defect class bd-0oro
    // records.
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut application_state = String::new();
    let mut journal_state = String::new();
    let mut receipt_body = String::new();
    while Instant::now() < deadline {
        let (code, body) = http(addr, "GET", &format!("/api/control/status/{command_id}"))?;
        if code == 200 {
            receipt_body = body.clone();
            application_state = json_str(&body, "application_state").unwrap_or_default();
            journal_state = json_str(&body, "journal_state").unwrap_or_default();
            if application_state == "applied" {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    assert_eq!(
        application_state, "applied",
        "the command must reach the WORLD, not merely the queue; receipt: {receipt_body}"
    );

    // WHICH LEVEL THIS PROVES, stated rather than left for the reader to assume.
    // `application_state == applied` is the applier's own observation: the world
    // took the command. The JOURNAL axis is a separate, strictly stronger claim
    // about durability, and it is reported here as OBSERVED rather than asserted at
    // a level this path does not reach. bd-88yj records that the legacy bus — which
    // server mode drains — writes no lifecycle record, so a `not_required` journal
    // state here is the honest current answer, not a failure. If a later change
    // advances it, this assertion will still pass and the evidence line below will
    // show the stronger value, which is the point of recording it separately.
    assert!(
        !journal_state.is_empty(),
        "the receipt must report a journal state, even if it is not_required: {receipt_body}"
    );

    // (3) The screenshot endpoint must REFUSE, because no frame was presented.
    let (shot_code, shot_body) = http(addr, "GET", "/api/screenshot/ascii")?;
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

    // (4) Lifecycle: the child must still be running (this is a long-lived server,
    // not a batch job that happened to answer), and must then terminate.
    let child = guard.0.as_mut().expect("child still held");
    assert!(
        child.try_wait()?.is_none(),
        "--mode server must still be running after the assertions; it exited early"
    );
    child.kill()?;
    let exit = child.wait()?;
    guard.0 = None;

    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!(
        "{{\"schema\":\"scriptbots.real-process-e2e.v1\",\"binary\":\"{}\",\"mode\":\"server\",\
         \"storage\":\"memory\",\"rest_address\":\"{addr}\",\"boot_log_lines\":{},\
         \"status_code\":{status_code},\"pause_code\":{pause_code},\"command_id\":\"{command_id}\",\
         \"application_state\":\"{application_state}\",\"journal_state\":\"{journal_state}\",\
         \"proved_level\":\"applied\",\"screenshot_code\":{shot_code},\
         \"screenshot_bytes\":{},\"terminated\":{},\"source_commit\":\"{commit}\"}}",
        binary().display(),
        boot_log.len(),
        shot_body.len(),
        !exit.success() || exit.code().is_some(),
    );

    Ok(())
}
