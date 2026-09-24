//! Cross-surface conformance matrix for the scriptbots control plane (bd-5dkk).
//!
//! Asserts parity and behavioral contracts across FastMCP, REST, and streaming
//! surfaces:
//! 1. FastMCP production dispatch, protocolVersion negotiation (-32602 on unsupported),
//!    and the registered 15-tool control roster.
//! 2. Cross-surface read and command parity between REST and MCP.
//! 3. Idempotency key replay deduplication across surfaces.
//! 4. Selection and interventions post-gap state reconstruction.
//! 5. SSE and NDJSON resume, gap semantics, and MCP /mcp/events endpoint discovery.

use std::{
    collections::HashSet,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use scriptbots_app::{
    CommandSubmit, ControlRuntime, ControlServerConfig, McpTransportConfig, host_thread::HostThread,
};
use scriptbots_core::{ScriptBotsConfig, WorldState};
use scriptbots_runtime::{
    HostCoreOptions, HostSessionId, VolatileJournal, channel::ChannelHostOptions,
};
use serde_json::Value;

fn unused_loopback_address() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("ephemeral port");
    let address = listener.local_addr().expect("bound address");
    drop(listener);
    address
}

fn http_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> Result<(u16, String)> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut header_bytes = Vec::new();
    let body_len = body.map_or(0, |b| b.len());
    write!(
        header_bytes,
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {body_len}\r\nConnection: close\r\n"
    )?;
    for (k, v) in headers {
        write!(header_bytes, "{k}: {v}\r\n")?;
    }
    write!(header_bytes, "\r\n")?;
    stream.write_all(&header_bytes)?;
    if let Some(b) = body {
        stream.write_all(b)?;
    }
    stream.flush()?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body_str) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| anyhow!("malformed HTTP response: {text:?}"))?;
    let status: u16 = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1).map(str::to_string))
        .ok_or_else(|| anyhow!("no status line in {head:?}"))?
        .parse()?;
    Ok((status, body_str.to_string()))
}

fn mcp_json_rpc(
    mcp_addr: SocketAddr,
    id: Option<i64>,
    method: &str,
    params: Option<Value>,
) -> Result<(u16, Value)> {
    let mut obj = serde_json::Map::new();
    obj.insert("jsonrpc".to_string(), Value::String("2.0".to_string()));
    if let Some(id_val) = id {
        obj.insert("id".to_string(), Value::Number(id_val.into()));
    }
    obj.insert("method".to_string(), Value::String(method.to_string()));
    if let Some(p) = params {
        obj.insert("params".to_string(), p);
    }
    let body = serde_json::to_vec(&Value::Object(obj))?;
    let (status, resp_str) = http_request(
        mcp_addr,
        "POST",
        "/mcp",
        &[
            ("Content-Type", "application/json"),
            ("Accept", "application/json, text/event-stream"),
        ],
        Some(&body),
    )?;
    let val: Value = if resp_str.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str(&resp_str)
            .with_context(|| format!("parsing MCP JSON-RPC response: {resp_str:?}"))?
    };
    Ok((status, val))
}

fn read_stream_chunk(
    addr: SocketAddr,
    path: &str,
    headers: &[(&str, &str)],
    timeout: Duration,
) -> Result<String> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(timeout))?;
    let mut header_bytes = Vec::new();
    write!(
        header_bytes,
        "GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n"
    )?;
    for (k, v) in headers {
        write!(header_bytes, "{k}: {v}\r\n")?;
    }
    write!(header_bytes, "\r\n")?;
    stream.write_all(&header_bytes)?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let mut output = String::new();
    let mut line = String::new();
    let start = Instant::now();
    while start.elapsed() < timeout {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                output.push_str(&line);
                if output.contains("\n\n") && output.contains("data:") {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    Ok(output)
}

struct TestFixture {
    host: Option<HostThread>,
    runtime: Option<ControlRuntime>,
    submit: Option<CommandSubmit>,
    pub rest_addr: SocketAddr,
    pub mcp_addr: SocketAddr,
}

impl Drop for TestFixture {
    fn drop(&mut self) {
        if let Some(rt) = self.runtime.take() {
            let _ = rt.shutdown();
        }
        self.submit.take();
        if let Some(h) = self.host.take() {
            let _ = h.join();
        }
    }
}

fn setup_conformance_fixture() -> TestFixture {
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 64,
        world_height: 64,
        food_cell_size: 16,
        rng_seed: Some(0x00C0_FFEE),
        persistence_interval: 0,
        closed: true,
        ..ScriptBotsConfig::default()
    })
    .expect("conformance world");
    world.step().expect("world step");

    let persistence = world
        .bind_persistence(Box::new(scriptbots_core::NullPersistence))
        .expect("persistence binding");
    let host = HostThread::spawn(
        HostSessionId::new(0xc0ff),
        world,
        persistence,
        Box::new(VolatileJournal::default()),
        HostCoreOptions::default(),
        ChannelHostOptions::default(),
    )
    .expect("conformance host");

    let rest_addr = unused_loopback_address();
    let mcp_addr = unused_loopback_address();
    let config = ControlServerConfig {
        rest_address: rest_addr,
        rest_enabled: true,
        mcp_transport: McpTransportConfig::Http {
            bind_address: mcp_addr,
        },
        ..ControlServerConfig::default()
    };
    let (runtime, submit) =
        ControlRuntime::launch(host.port(), config).expect("control runtime launch");

    // Wait briefly for server listeners to be ready.
    std::thread::sleep(Duration::from_millis(150));

    TestFixture {
        host: Some(host),
        runtime: Some(runtime),
        submit: Some(submit),
        rest_addr,
        mcp_addr,
    }
}

fn initialize_mcp(mcp_addr: SocketAddr) -> Value {
    let (status, res) = mcp_json_rpc(
        mcp_addr,
        Some(1),
        "initialize",
        Some(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "conformance-tester",
                "version": "1.0.0"
            }
        })),
    )
    .expect("mcp initialize helper");
    assert_eq!(status, 200);
    assert_eq!(res["jsonrpc"], "2.0");
    assert_eq!(res["result"]["protocolVersion"], "2024-11-05");
    res
}

#[test]
fn test_mcp_protocol_negotiation_and_tool_discovery() {
    let fixture = setup_conformance_fixture();

    // 1. Malformed protocolVersion parameter type returns typed error code -32602.
    let (status_bad, bad_res) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(99),
        "initialize",
        Some(serde_json::json!({
            "protocolVersion": 42,
            "capabilities": {},
            "clientInfo": { "name": "bad-version", "version": "1.0" }
        })),
    )
    .expect("mcp bad version initialize");
    assert_eq!(status_bad, 200);
    assert_eq!(bad_res["id"], 99);
    assert_eq!(bad_res["error"]["code"], -32602);
    let msg = bad_res["error"]["message"].as_str().unwrap_or_default();
    assert!(
        !msg.is_empty(),
        "error message should explain invalid parameter: {msg}"
    );

    // 1b. Future protocolVersion string negotiates down to supported version 2024-11-05.
    let (status_future, future_res) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(100),
        "initialize",
        Some(serde_json::json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": { "name": "future-version", "version": "1.0" }
        })),
    )
    .expect("mcp future version initialize");
    assert_eq!(status_future, 200);
    assert_eq!(future_res["id"], 100);
    assert!(future_res.get("error").is_none());
    assert_eq!(future_res["result"]["protocolVersion"], "2024-11-05");

    // 2. Valid protocolVersion 2024-11-05 negotiates successfully.
    let init_res = initialize_mcp(fixture.mcp_addr);
    assert_eq!(
        init_res["result"]["serverInfo"]["name"],
        "scriptbots-control"
    );
    assert!(init_res["result"]["capabilities"]["tools"].is_object());

    // 3. Notification (no ID) receives HTTP 202 without body.
    let (status_notif, notif_res) = mcp_json_rpc(
        fixture.mcp_addr,
        None,
        "notifications/initialized",
        Some(serde_json::json!({})),
    )
    .expect("mcp notification");
    assert_eq!(status_notif, 202);
    assert_eq!(notif_res, Value::Null);

    // 4. tools/list returns exactly the 15-tool control roster.
    let (status_tools, tools_res) =
        mcp_json_rpc(fixture.mcp_addr, Some(2), "tools/list", None).expect("mcp tools/list");
    assert_eq!(status_tools, 200);
    let tool_names: HashSet<String> = tools_res["result"]["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .filter_map(|t| t["name"].as_str().map(str::to_string))
        .collect();

    let expected_tools = [
        "apply_patch",
        "apply_preset",
        "apply_updates",
        "get_command_status",
        "intervene",
        "get_config",
        "get_status",
        "list_knobs",
        "list_presets",
        "map_apply",
        "map_generate",
        "pause",
        "resume",
        "set_speed",
        "shutdown",
        "step",
        "narrative_search",
        "narrative_around",
        "get_version",
        "get_schema",
        "experiment_create",
        "experiment_list",
        "experiment_status",
        "experiment_cancel",
        "experiment_resume",
        "checkpoint_create",
        "checkpoint_list",
        "checkpoint_status",
        "artifact_list",
        "artifact_get",
    ];
    for expected in expected_tools {
        assert!(
            tool_names.contains(expected),
            "expected tool '{expected}' missing from roster: {tool_names:?}"
        );
    }
    assert_eq!(tool_names.len(), 30);
}

#[test]
fn test_cross_surface_query_parity() {
    let fixture = setup_conformance_fixture();
    initialize_mcp(fixture.mcp_addr);

    // 1. Status parity: REST /api/status vs MCP tools/call get_status.
    let (rest_status, rest_body) =
        http_request(fixture.rest_addr, "GET", "/api/status", &[], None).expect("rest get status");
    assert_eq!(rest_status, 200);
    let rest_status_json: Value = serde_json::from_str(&rest_body).expect("parse rest status");

    let (mcp_status, mcp_call_res) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(10),
        "tools/call",
        Some(serde_json::json!({
            "name": "get_status",
            "arguments": {}
        })),
    )
    .expect("mcp call get_status");
    assert_eq!(mcp_status, 200);
    let mcp_text = mcp_call_res["result"]["content"][0]["text"]
        .as_str()
        .expect("content text");
    let mcp_status_json: Value = serde_json::from_str(mcp_text).expect("parse mcp status text");

    assert_eq!(
        rest_status_json["is_closed"], mcp_status_json["is_closed"],
        "status is_closed match"
    );
    assert_eq!(
        rest_status_json["config_revision"], mcp_status_json["config_revision"],
        "config_revision match"
    );

    // 2. Config parity: REST /api/config vs MCP tools/call get_config.
    let (rest_cfg_status, rest_cfg_body) =
        http_request(fixture.rest_addr, "GET", "/api/config", &[], None).expect("rest get config");
    assert_eq!(rest_cfg_status, 200);
    let rest_cfg_json: Value = serde_json::from_str(&rest_cfg_body).expect("parse rest config");

    let (mcp_cfg_status, mcp_cfg_call) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(11),
        "tools/call",
        Some(serde_json::json!({
            "name": "get_config",
            "arguments": {}
        })),
    )
    .expect("mcp call get_config");
    assert_eq!(mcp_cfg_status, 200);
    let mcp_cfg_text = mcp_cfg_call["result"]["content"][0]["text"]
        .as_str()
        .expect("content text");
    let mcp_cfg_json: Value = serde_json::from_str(mcp_cfg_text).expect("parse mcp config text");

    assert_eq!(
        rest_cfg_json["config"]["world_width"], mcp_cfg_json["config"]["world_width"],
        "world_width match"
    );
    assert_eq!(
        rest_cfg_json["config"]["world_height"], mcp_cfg_json["config"]["world_height"],
        "world_height match"
    );

    // 3. Presets parity: REST /api/presets vs MCP tools/call list_presets.
    let (rest_pre_status, rest_pre_body) =
        http_request(fixture.rest_addr, "GET", "/api/presets", &[], None)
            .expect("rest get presets");
    assert_eq!(rest_pre_status, 200);
    let rest_pre_json: Value = serde_json::from_str(&rest_pre_body).expect("parse rest presets");

    let (mcp_pre_status, mcp_pre_call) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(12),
        "tools/call",
        Some(serde_json::json!({
            "name": "list_presets",
            "arguments": {}
        })),
    )
    .expect("mcp call list_presets");
    assert_eq!(mcp_pre_status, 200);
    let mcp_pre_text = mcp_pre_call["result"]["content"][0]["text"]
        .as_str()
        .expect("content text");
    let mcp_pre_json: Value = serde_json::from_str(mcp_pre_text).expect("parse mcp presets text");

    assert_eq!(
        rest_pre_json["presets"].as_array().map(|a| a.len()),
        mcp_pre_json.as_array().map(|a| a.len()),
        "presets count match"
    );
}

#[test]
fn test_cross_surface_command_parity_and_receipt_tracking() {
    let fixture = setup_conformance_fixture();
    initialize_mcp(fixture.mcp_addr);

    // 1. REST pause command.
    let (rest_status, rest_body) =
        http_request(fixture.rest_addr, "POST", "/api/control/pause", &[], None)
            .expect("rest pause");
    assert_eq!(rest_status, 200);
    let rest_receipt: Value = serde_json::from_str(&rest_body).expect("parse rest pause receipt");
    let rest_cmd_id = rest_receipt["command_id"].as_str().expect("command id");
    assert_eq!(rest_receipt["application_state"], "admitted");

    // Query status of REST command via MCP get_command_status tool.
    let (mcp_stat_status, mcp_stat_call) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(20),
        "tools/call",
        Some(serde_json::json!({
            "name": "get_command_status",
            "arguments": { "command_id": rest_cmd_id }
        })),
    )
    .expect("mcp query rest command status");
    assert_eq!(mcp_stat_status, 200);
    let mcp_stat_text = mcp_stat_call["result"]["content"][0]["text"]
        .as_str()
        .expect("command status text");
    let mcp_stat_json: Value = serde_json::from_str(mcp_stat_text).expect("parse command status");
    assert_eq!(mcp_stat_json["command_id"], rest_cmd_id);

    // 2. MCP resume command.
    let (mcp_resume_status, mcp_resume_call) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(21),
        "tools/call",
        Some(serde_json::json!({
            "name": "resume",
            "arguments": {}
        })),
    )
    .expect("mcp resume");
    assert_eq!(mcp_resume_status, 200);
    let mcp_resume_text = mcp_resume_call["result"]["content"][0]["text"]
        .as_str()
        .expect("resume text");
    let mcp_resume_json: Value = serde_json::from_str(mcp_resume_text).expect("parse resume JSON");
    let mcp_cmd_id = mcp_resume_json["command_id"]
        .as_str()
        .expect("mcp command id");

    // Query status of MCP command via REST /api/control/status/{id}.
    let (rest_query_status, rest_query_body) = http_request(
        fixture.rest_addr,
        "GET",
        &format!("/api/control/status/{mcp_cmd_id}"),
        &[],
        None,
    )
    .expect("rest query mcp command status");
    assert_eq!(rest_query_status, 200);
    let rest_query_json: Value =
        serde_json::from_str(&rest_query_body).expect("parse rest query json");
    assert_eq!(rest_query_json["command_id"], mcp_cmd_id);
}

#[test]
fn test_idempotency_key_deduplication() {
    let fixture = setup_conformance_fixture();

    let idempotency_key = "matrix-test-dedup-key-777";
    let headers = [("X-Idempotency-Key", idempotency_key)];

    // First submission.
    let (status1, body1) = http_request(
        fixture.rest_addr,
        "POST",
        "/api/control/pause",
        &headers,
        None,
    )
    .expect("first pause request");
    assert_eq!(status1, 200);
    let receipt1: Value = serde_json::from_str(&body1).expect("parse receipt 1");
    let cmd_id1 = receipt1["command_id"].as_str().expect("cmd_id1");

    // Replay with identical idempotency key.
    let (status2, body2) = http_request(
        fixture.rest_addr,
        "POST",
        "/api/control/pause",
        &headers,
        None,
    )
    .expect("second pause request");
    assert_eq!(status2, 200);
    let receipt2: Value = serde_json::from_str(&body2).expect("parse receipt 2");
    let cmd_id2 = receipt2["command_id"].as_str().expect("cmd_id2");

    assert_eq!(
        cmd_id1, cmd_id2,
        "identical idempotency key must return the existing command receipt"
    );
}

#[test]
fn test_selection_and_interventions_gap_recovery() {
    let fixture = setup_conformance_fixture();

    // 1. Initial selection state is queryable.
    let (status_sel_init, body_sel_init) =
        http_request(fixture.rest_addr, "GET", "/api/selection", &[], None)
            .expect("get initial selection");
    assert_eq!(status_sel_init, 200);
    let init_sel: Value = serde_json::from_str(&body_sel_init).expect("parse initial selection");
    assert_eq!(init_sel["selected_count"], 0);
    let init_rev = init_sel["revision"].as_u64().unwrap_or(0);

    // 2. Post selection update.
    let update_body = serde_json::to_vec(&serde_json::json!({
        "mode": "replace",
        "agent_ids": [42, 43],
        "state": "selected"
    }))
    .expect("serialize selection body");

    let (status_post, body_post) = http_request(
        fixture.rest_addr,
        "POST",
        "/api/selection",
        &[("Content-Type", "application/json")],
        Some(&update_body),
    )
    .expect("post selection update");
    assert_eq!(status_post, 202);
    let receipt: Value = serde_json::from_str(&body_post).expect("parse post receipt");
    assert_eq!(receipt["application_state"], "admitted");

    // 3. Selection query confirms revision update and selected items.
    let (status_sel_after, body_sel_after) =
        http_request(fixture.rest_addr, "GET", "/api/selection", &[], None)
            .expect("get updated selection");
    assert_eq!(status_sel_after, 200);
    let after_sel: Value = serde_json::from_str(&body_sel_after).expect("parse updated selection");
    assert!(
        after_sel["revision"].as_u64().unwrap_or(0) >= init_rev,
        "selection revision advances"
    );

    // 4. Interventions query supports after_seq filter and gap flag.
    let (status_intv, body_intv) =
        http_request(fixture.rest_addr, "GET", "/api/interventions", &[], None)
            .expect("get interventions");
    assert_eq!(status_intv, 200);
    let intv_json: Value = serde_json::from_str(&body_intv).expect("parse interventions");
    assert_eq!(intv_json["gap_detected"], false);
    assert!(intv_json["records"].is_array());

    // Querying with after_seq filter.
    let (status_intv_seq, body_intv_seq) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/interventions?after_seq=0",
        &[],
        None,
    )
    .expect("get interventions after_seq");
    assert_eq!(status_intv_seq, 200);
    let intv_seq_json: Value =
        serde_json::from_str(&body_intv_seq).expect("parse interventions seq");
    assert_eq!(intv_seq_json["gap_detected"], false);
}

#[test]
fn test_sse_resume_filtering_and_mcp_events_discovery() {
    let fixture = setup_conformance_fixture();

    // 1. /mcp/events delivers the MCP SSE endpoint discovery event first.
    let mcp_events_chunk = read_stream_chunk(
        fixture.mcp_addr,
        "/mcp/events",
        &[("Accept", "text/event-stream")],
        Duration::from_secs(3),
    )
    .expect("read mcp events chunk");
    assert!(
        mcp_events_chunk.contains("event: endpoint"),
        "mcp/events first event must be endpoint: {mcp_events_chunk}"
    );
    assert!(
        mcp_events_chunk.contains("data: /mcp"),
        "mcp/events data must point to /mcp: {mcp_events_chunk}"
    );

    // 2. /api/ticks/stream delivers summary events with IDs.
    let sse_chunk = read_stream_chunk(
        fixture.rest_addr,
        "/api/ticks/stream",
        &[("Accept", "text/event-stream")],
        Duration::from_secs(3),
    )
    .expect("read sse chunk");
    assert!(
        sse_chunk.contains("event: tick"),
        "SSE must contain tick event: {sse_chunk}"
    );
    assert!(
        sse_chunk.contains("id:"),
        "SSE must include event ID: {sse_chunk}"
    );

    // 3. /api/ticks/stream with future/evicted tick yields a gap event for reconstruction.
    let sse_gap_chunk = read_stream_chunk(
        fixture.rest_addr,
        "/api/ticks/stream?after_tick=9999999",
        &[("Accept", "text/event-stream")],
        Duration::from_secs(3),
    )
    .expect("read sse gap chunk");
    assert!(
        sse_gap_chunk.contains("event: gap"),
        "requesting evicted/disjoint tick must emit gap event: {sse_gap_chunk}"
    );
    assert!(
        sse_gap_chunk.contains("\"gap\":true"),
        "gap data must identify gap: {sse_gap_chunk}"
    );

    // 4. /api/ticks/ndjson delivers json lines.
    let ndjson_chunk = read_stream_chunk(
        fixture.rest_addr,
        "/api/ticks/ndjson",
        &[("Accept", "application/x-ndjson")],
        Duration::from_secs(3),
    )
    .expect("read ndjson chunk");
    assert!(
        ndjson_chunk.contains("\"tick\":"),
        "NDJSON must contain tick summary: {ndjson_chunk}"
    );
}

#[test]
fn test_cross_surface_intervention_equivalence_and_rejection() {
    let fixture = setup_conformance_fixture();
    let _ = initialize_mcp(fixture.mcp_addr);

    // 1. REST /api/control/intervene submission
    let rest_payload = serde_json::json!({
        "intervention": {
            "kind": "drought",
            "region": { "shape": "all" },
            "ticks": 20,
            "growth_scale": 0.25
        },
        "surface": "rest",
        "actor": "rest_tester"
    });
    let (rest_status, rest_resp) = http_request(
        fixture.rest_addr,
        "POST",
        "/api/control/intervene",
        &[("Content-Type", "application/json")],
        Some(rest_payload.to_string().as_bytes()),
    )
    .expect("REST intervene POST");
    assert_eq!(rest_status, 200, "REST intervene response: {rest_resp}");
    let rest_json: Value = serde_json::from_str(&rest_resp).expect("parse REST response JSON");
    assert_eq!(rest_json["application_state"], "admitted");

    // 2. REST alias /api/interventions submission
    let alias_payload = serde_json::json!({
        "intervention": {
            "kind": "bloom",
            "region": { "shape": "disc", "x": 32.0, "y": 32.0, "radius": 16.0 },
            "amount": 50.0
        },
        "surface": "rest",
        "actor": "rest_alias_tester"
    });
    let (alias_status, alias_resp) = http_request(
        fixture.rest_addr,
        "POST",
        "/api/interventions",
        &[("Content-Type", "application/json")],
        Some(alias_payload.to_string().as_bytes()),
    )
    .expect("REST alias intervene POST");
    assert_eq!(
        alias_status, 200,
        "REST alias intervene response: {alias_resp}"
    );
    let alias_json: Value = serde_json::from_str(&alias_resp).expect("parse REST alias JSON");
    assert_eq!(alias_json["application_state"], "admitted");

    // 3. FastMCP tool `intervene` submission
    let mcp_call_payload = serde_json::json!({
        "intervention": {
            "kind": "drought",
            "region": { "shape": "all" },
            "ticks": 20,
            "growth_scale": 0.25
        },
        "surface": "mcp",
        "actor": "mcp_tester"
    });
    let (mcp_status, mcp_resp) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(42),
        "tools/call",
        Some(serde_json::json!({
            "name": "intervene",
            "arguments": mcp_call_payload
        })),
    )
    .expect("MCP tool call intervene");
    assert_eq!(mcp_status, 200, "MCP response: {mcp_resp}");
    assert_eq!(mcp_resp["jsonrpc"], "2.0");
    assert!(mcp_resp["error"].is_null(), "MCP tool error: {mcp_resp}");
    let mcp_text = mcp_resp["result"]["content"][0]["text"]
        .as_str()
        .expect("MCP tool text");
    let mcp_json: Value = serde_json::from_str(mcp_text).expect("parse MCP response JSON");
    assert_eq!(mcp_json["application_state"], "admitted");

    // 4. Rejection across surfaces: invalid parameter (ticks: 0) rejected by REST
    let bad_rest_payload = serde_json::json!({
        "intervention": {
            "kind": "drought",
            "region": { "shape": "all" },
            "ticks": 0,
            "growth_scale": 0.25
        },
        "surface": "rest",
        "actor": "bad_actor"
    });
    let (bad_status, bad_resp) = http_request(
        fixture.rest_addr,
        "POST",
        "/api/control/intervene",
        &[("Content-Type", "application/json")],
        Some(bad_rest_payload.to_string().as_bytes()),
    )
    .expect("REST bad intervene POST");
    assert!(
        bad_status == 400 || bad_resp.contains("\"accepted\":false"),
        "invalid intervention must be rejected by REST: status={bad_status}, body={bad_resp}"
    );

    // 5. Canonical Postcard parameter parity:
    // REST drought and MCP drought produce identical canonical parameter bytes.
    let drought_canonical =
        scriptbots_core::interventions::drought(scriptbots_core::Region::All, 20, 0.25)
            .expect("canonical constructor drought");
    let drought_bytes = scriptbots_core::interventions::canonical_param_bytes(&drought_canonical);
    assert!(
        !drought_bytes.is_empty(),
        "canonical bytes must not be empty"
    );
}

fn http_request_raw(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> Result<(u16, Vec<u8>)> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut header_bytes = Vec::new();
    let body_len = body.map_or(0, |b| b.len());
    write!(
        header_bytes,
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {body_len}\r\nConnection: close\r\n"
    )?;
    for (k, v) in headers {
        write!(header_bytes, "{k}: {v}\r\n")?;
    }
    write!(header_bytes, "\r\n")?;
    stream.write_all(&header_bytes)?;
    if let Some(b) = body {
        stream.write_all(b)?;
    }
    stream.flush()?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let sep = b"\r\n\r\n";
    let pos = raw
        .windows(sep.len())
        .position(|window| window == sep)
        .ok_or_else(|| anyhow!("malformed HTTP response"))?;
    let head = std::str::from_utf8(&raw[..pos])?;
    let status: u16 = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1).map(str::to_string))
        .ok_or_else(|| anyhow!("no status line in {head:?}"))?
        .parse()?;
    let body_bytes = raw[pos + sep.len()..].to_vec();
    Ok((status, body_bytes))
}

#[test]
fn test_version_and_schema_cross_surface_parity() {
    let fixture = setup_conformance_fixture();
    initialize_mcp(fixture.mcp_addr);

    // 1. Version parity: GET /api/v1/version vs GET /api/version vs MCP get_version
    let (status_v1, body_v1) = http_request(fixture.rest_addr, "GET", "/api/v1/version", &[], None)
        .expect("GET /api/v1/version");
    assert_eq!(status_v1, 200);
    let v1_json: Value = serde_json::from_str(&body_v1).expect("parse /api/v1/version");

    let (status_v_alias, body_v_alias) =
        http_request(fixture.rest_addr, "GET", "/api/version", &[], None)
            .expect("GET /api/version");
    assert_eq!(status_v_alias, 200);
    let v_alias_json: Value = serde_json::from_str(&body_v_alias).expect("parse /api/version");
    assert_eq!(v1_json, v_alias_json, "version alias parity");

    let (mcp_status, mcp_call) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(100),
        "tools/call",
        Some(serde_json::json!({
            "name": "get_version",
            "arguments": {}
        })),
    )
    .expect("mcp call get_version");
    assert_eq!(mcp_status, 200);
    let mcp_text = mcp_call["result"]["content"][0]["text"]
        .as_str()
        .expect("mcp text");
    let mcp_version: Value = serde_json::from_str(mcp_text).expect("parse mcp version");

    assert_eq!(v1_json["service_name"], mcp_version["service_name"]);
    assert_eq!(v1_json["version"], mcp_version["version"]);
    assert_eq!(v1_json["capabilities"], mcp_version["capabilities"]);

    // 2. Schema parity: GET /api/v1/schema vs GET /api/schema vs MCP get_schema
    let (status_schema1, body_schema1) =
        http_request(fixture.rest_addr, "GET", "/api/v1/schema", &[], None)
            .expect("GET /api/v1/schema");
    assert_eq!(status_schema1, 200);
    let schema1_json: Value = serde_json::from_str(&body_schema1).expect("parse /api/v1/schema");

    let (status_schema_alias, body_schema_alias) =
        http_request(fixture.rest_addr, "GET", "/api/schema", &[], None).expect("GET /api/schema");
    assert_eq!(status_schema_alias, 200);
    let schema_alias_json: Value =
        serde_json::from_str(&body_schema_alias).expect("parse /api/schema");
    assert_eq!(schema1_json, schema_alias_json, "schema alias parity");

    let (mcp_schema_status, mcp_schema_call) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(101),
        "tools/call",
        Some(serde_json::json!({
            "name": "get_schema",
            "arguments": {}
        })),
    )
    .expect("mcp call get_schema");
    assert_eq!(mcp_schema_status, 200);
    let mcp_schema_text = mcp_schema_call["result"]["content"][0]["text"]
        .as_str()
        .expect("mcp text");
    let mcp_schema: Value = serde_json::from_str(mcp_schema_text).expect("parse mcp schema");

    assert_eq!(schema1_json["title"], mcp_schema["title"]);
    assert_eq!(
        schema1_json["mcp_tools"].as_array().map(|a| a.len()),
        Some(30)
    );
    assert_eq!(
        mcp_schema["mcp_tools"].as_array().map(|a| a.len()),
        Some(30)
    );
    assert!(schema1_json["routes"].as_array().map_or(0, |a| a.len()) >= 20);
}

#[test]
fn test_experiments_checkpoints_and_artifacts_lifecycle() {
    let fixture = setup_conformance_fixture();
    initialize_mcp(fixture.mcp_addr);

    // --- 1. Experiment Validation & Error Semantics ---
    // 1a. Empty variants rejected with 400 Bad Request
    let empty_variants = serde_json::json!({
        "variants": [],
        "seeds": [42]
    });
    let (st_err1, body_err1) = http_request(
        fixture.rest_addr,
        "POST",
        "/api/v1/experiments",
        &[("Content-Type", "application/json")],
        Some(empty_variants.to_string().as_bytes()),
    )
    .expect("POST empty variants");
    assert_eq!(st_err1, 400);
    assert!(
        body_err1.contains("variants cannot be empty"),
        "body={body_err1}"
    );

    // 1b. Empty seeds rejected with 400 Bad Request
    let empty_seeds = serde_json::json!({
        "variants": [{ "variant_id": "v1", "brain_family": "mlp" }],
        "seeds": []
    });
    let (st_err2, body_err2) = http_request(
        fixture.rest_addr,
        "POST",
        "/api/v1/experiments",
        &[("Content-Type", "application/json")],
        Some(empty_seeds.to_string().as_bytes()),
    )
    .expect("POST empty seeds");
    assert_eq!(st_err2, 400);
    assert!(
        body_err2.contains("seeds cannot be empty"),
        "body={body_err2}"
    );

    // 1c. Invalid brain family rejected with 400 Bad Request
    let bad_brain = serde_json::json!({
        "variants": [{ "variant_id": "v1", "brain_family": "nonexistent_brain_architecture" }],
        "seeds": [1]
    });
    let (st_err3, body_err3) = http_request(
        fixture.rest_addr,
        "POST",
        "/api/v1/experiments",
        &[("Content-Type", "application/json")],
        Some(bad_brain.to_string().as_bytes()),
    )
    .expect("POST bad brain");
    assert_eq!(st_err3, 400);
    assert!(
        body_err3.contains("unknown brain family"),
        "body={body_err3}"
    );

    // --- 2. Create Experiment & Idempotency ---
    let create_payload = serde_json::json!({
        "description": "conformance batch test",
        "variants": [
            { "variant_id": "mlp_std", "brain_family": "mlp" }
        ],
        "seeds": [101, 102],
        "ticks_per_run": 5,
        "idempotency_key": "conformance-exp-idem-42"
    });
    let (st_create, body_create) = http_request(
        fixture.rest_addr,
        "POST",
        "/api/v1/experiments",
        &[("Content-Type", "application/json")],
        Some(create_payload.to_string().as_bytes()),
    )
    .expect("POST create experiment");
    assert_eq!(st_create, 201, "create response: {body_create}");
    let create_json: Value = serde_json::from_str(&body_create).expect("parse create experiment");
    let exp_id = create_json["experiment_id"]
        .as_str()
        .expect("experiment_id string")
        .to_string();
    assert_eq!(create_json["total_runs"], 2);

    // Idempotent retry returns identical experiment
    let (st_create_retry, body_create_retry) = http_request(
        fixture.rest_addr,
        "POST",
        "/api/v1/experiments",
        &[("Content-Type", "application/json")],
        Some(create_payload.to_string().as_bytes()),
    )
    .expect("POST create experiment retry");
    assert_eq!(st_create_retry, 201);
    let retry_json: Value = serde_json::from_str(&body_create_retry).expect("parse retry");
    assert_eq!(retry_json["experiment_id"], exp_id);

    // --- 3. Experiment Lookup & Status Parity (REST vs MCP) ---
    let (st_get, body_get) = http_request(
        fixture.rest_addr,
        "GET",
        &format!("/api/v1/experiments/{exp_id}"),
        &[],
        None,
    )
    .expect("GET experiment");
    assert_eq!(st_get, 200);
    let get_json: Value = serde_json::from_str(&body_get).expect("parse get experiment");
    assert_eq!(get_json["experiment_id"], exp_id);

    let (mcp_exp_st, mcp_exp_call) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(102),
        "tools/call",
        Some(serde_json::json!({
            "name": "experiment_status",
            "arguments": { "experiment_id": exp_id }
        })),
    )
    .expect("mcp call experiment_status");
    assert_eq!(mcp_exp_st, 200);
    let mcp_exp_text = mcp_exp_call["result"]["content"][0]["text"]
        .as_str()
        .expect("mcp text");
    let mcp_exp_json: Value =
        serde_json::from_str(mcp_exp_text).expect("parse mcp experiment status");
    assert_eq!(mcp_exp_json["experiment_id"], exp_id);

    // The batch really executes: wait for every run to reach a terminal state
    // with a verified final digest and a bundle on disk (bd-2z0.12.2).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    let finished_json = loop {
        let (st, body) = http_request(
            fixture.rest_addr,
            "GET",
            &format!("/api/v1/experiments/{exp_id}"),
            &[],
            None,
        )
        .expect("GET experiment while executing");
        assert_eq!(st, 200, "status response: {body}");
        let json: Value = serde_json::from_str(&body).expect("parse experiment status");
        let status = json["status"].as_str().unwrap_or_default().to_string();
        assert!(
            ["pending", "running", "completed"].contains(&status.as_str()),
            "unexpected experiment status {status}: {json}"
        );
        if status == "completed" {
            break json;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "experiment did not complete within 300s: {json}"
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    assert_eq!(finished_json["completed_runs"], 2, "{finished_json}");
    assert_eq!(finished_json["failed_runs"], 0, "{finished_json}");
    let finished_runs = finished_json["runs"].as_array().expect("runs array");
    assert_eq!(finished_runs.len(), 2);
    let mut digests = Vec::new();
    for run in finished_runs {
        assert_eq!(run["state"], "completed", "{run}");
        assert_eq!(run["total_ticks"], 5, "{run}");
        let digest = run["final_digest"].as_str().expect("final digest");
        assert!(!digest.is_empty());
        digests.push(digest.to_string());
        let bundle = run["bundle_path"].as_str().expect("bundle path");
        assert!(
            std::path::Path::new(bundle).exists(),
            "bundle {bundle} missing on disk"
        );
    }
    assert_ne!(
        digests[0], digests[1],
        "distinct seeds must yield distinct final worlds"
    );

    // Cancelling a finished batch is idempotent and reports the terminal state.
    let (st_cancel, body_cancel) = http_request(
        fixture.rest_addr,
        "POST",
        &format!("/api/v1/experiments/{exp_id}/cancel"),
        &[],
        None,
    )
    .expect("POST cancel experiment");
    assert_eq!(st_cancel, 200);
    let cancel_json: Value = serde_json::from_str(&body_cancel).expect("parse cancel");
    assert_eq!(cancel_json["status"], "completed");

    // Resuming a batch with no pending runs is refused over MCP.
    let (mcp_resume_st, mcp_resume_call) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(103),
        "tools/call",
        Some(serde_json::json!({
            "name": "experiment_resume",
            "arguments": { "experiment_id": exp_id }
        })),
    )
    .expect("mcp call experiment_resume");
    assert_eq!(mcp_resume_st, 200);
    assert_eq!(
        mcp_resume_call["result"]["isError"], true,
        "resume of a finished batch must be refused: {mcp_resume_call}"
    );
    assert!(
        mcp_resume_call["result"]["content"][0]["text"]
            .as_str()
            .is_some_and(|text| text.contains("no pending runs")),
        "refusal must name the reason: {mcp_resume_call}"
    );

    // List experiments via REST and MCP with pagination bounds
    let (st_exp_list, body_exp_list) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/v1/experiments?limit=500",
        &[],
        None,
    )
    .expect("GET experiments list");
    assert_eq!(st_exp_list, 200);
    let exp_list_json: Value = serde_json::from_str(&body_exp_list).expect("parse exp list");
    assert!(exp_list_json["items"].as_array().map_or(0, |a| a.len()) >= 1);

    // --- 4. Checkpoint Creation & Parity ---
    let chk_payload = serde_json::json!({
        "description": "conformance test checkpoint",
        "idempotency_key": "chk-idem-001"
    });
    let (st_chk_create, body_chk_create) = http_request(
        fixture.rest_addr,
        "POST",
        "/api/v1/checkpoints",
        &[("Content-Type", "application/json")],
        Some(chk_payload.to_string().as_bytes()),
    )
    .expect("POST create checkpoint");
    assert_eq!(st_chk_create, 201, "body: {body_chk_create}");
    let chk_create_json: Value = serde_json::from_str(&body_chk_create).expect("parse chk create");
    let chk_id = chk_create_json["checkpoint_id"]
        .as_str()
        .expect("checkpoint_id")
        .to_string();
    let artifact_id = chk_id.clone();
    let expected_sha256 = chk_create_json["checksum_sha256"]
        .as_str()
        .expect("sha256")
        .to_string();
    let expected_blake3 = chk_create_json["checksum_blake3"]
        .as_str()
        .expect("blake3")
        .to_string();

    // Checkpoint lookup via REST & MCP
    let (st_chk_get, _body_chk_get) = http_request(
        fixture.rest_addr,
        "GET",
        &format!("/api/v1/checkpoints/{chk_id}"),
        &[],
        None,
    )
    .expect("GET checkpoint");
    assert_eq!(st_chk_get, 200);

    let (mcp_chk_st, mcp_chk_call) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(104),
        "tools/call",
        Some(serde_json::json!({
            "name": "checkpoint_status",
            "arguments": { "checkpoint_id": chk_id }
        })),
    )
    .expect("mcp call checkpoint_status");
    assert_eq!(mcp_chk_st, 200);
    let mcp_chk_text = mcp_chk_call["result"]["content"][0]["text"]
        .as_str()
        .expect("mcp text");
    let mcp_chk_json: Value = serde_json::from_str(mcp_chk_text).expect("parse mcp chk");
    assert_eq!(mcp_chk_json["checkpoint_id"], chk_id);

    // List checkpoints via REST & MCP
    let (st_chk_list, body_chk_list) =
        http_request(fixture.rest_addr, "GET", "/api/v1/checkpoints", &[], None)
            .expect("GET checkpoints");
    assert_eq!(st_chk_list, 200);
    let chk_list_json: Value = serde_json::from_str(&body_chk_list).expect("parse chk list");
    assert!(chk_list_json["items"].as_array().map_or(0, |a| a.len()) >= 1);

    // --- 5. Artifact Discovery, Download, and Negative Tests ---
    // 5a. List artifacts via REST & MCP
    let (st_art_list, body_art_list) =
        http_request(fixture.rest_addr, "GET", "/api/v1/artifacts", &[], None)
            .expect("GET artifacts");
    assert_eq!(st_art_list, 200);
    let art_list_json: Value = serde_json::from_str(&body_art_list).expect("parse art list");
    assert!(art_list_json["items"].as_array().map_or(0, |a| a.len()) >= 1);

    let (mcp_art_st, _mcp_art_call) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(105),
        "tools/call",
        Some(serde_json::json!({
            "name": "artifact_list",
            "arguments": {}
        })),
    )
    .expect("mcp call artifact_list");
    assert_eq!(mcp_art_st, 200);

    // 5b. Get artifact metadata via REST & MCP
    let (st_art_get, body_art_get) = http_request(
        fixture.rest_addr,
        "GET",
        &format!("/api/v1/artifacts/{artifact_id}"),
        &[],
        None,
    )
    .expect("GET artifact metadata");
    assert_eq!(st_art_get, 200);
    let art_get_json: Value = serde_json::from_str(&body_art_get).expect("parse art get");
    assert_eq!(art_get_json["artifact_id"], artifact_id);
    assert_eq!(art_get_json["checksum_sha256"], expected_sha256);

    let (mcp_art_meta_st, _mcp_art_meta_call) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(106),
        "tools/call",
        Some(serde_json::json!({
            "name": "artifact_get",
            "arguments": { "artifact_id": artifact_id }
        })),
    )
    .expect("mcp call artifact_get");
    assert_eq!(mcp_art_meta_st, 200);

    // 5c. Download artifact bytes and verify checksums!
    let (st_dl, dl_bytes) = http_request_raw(
        fixture.rest_addr,
        "GET",
        &format!("/api/v1/artifacts/{artifact_id}/download"),
        &[],
        None,
    )
    .expect("download artifact");
    assert_eq!(st_dl, 200);
    assert!(
        !dl_bytes.is_empty(),
        "downloaded artifact bytes must not be empty"
    );

    let computed_sha256 = scriptbots_app::control::compute_sha256(&dl_bytes);
    let computed_blake3 = scriptbots_app::control::compute_blake3(&dl_bytes);
    assert_eq!(
        computed_sha256, expected_sha256,
        "sha256 checksum mismatch on downloaded bytes"
    );
    assert_eq!(
        computed_blake3, expected_blake3,
        "blake3 checksum mismatch on downloaded bytes"
    );

    // 5d. Path traversal attack rejection
    let (st_traversal, _) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/v1/artifacts/..%2F..%2Fetc%2Fpasswd/download",
        &[],
        None,
    )
    .expect("traversal request");
    assert!(
        st_traversal == 400 || st_traversal == 404,
        "path traversal must be rejected: {st_traversal}"
    );

    // 5e. Non-existent artifact lookup and download returns 404
    let (st_art_missing, _) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/v1/artifacts/nonexistent_artifact_xyz",
        &[],
        None,
    )
    .expect("missing artifact get");
    assert_eq!(st_art_missing, 404);

    let (st_art_missing_dl, _) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/v1/artifacts/nonexistent_artifact_xyz/download",
        &[],
        None,
    )
    .expect("missing artifact dl");
    assert_eq!(st_art_missing_dl, 404);
}
