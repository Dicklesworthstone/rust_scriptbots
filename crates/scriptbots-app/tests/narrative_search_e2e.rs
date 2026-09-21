//! End-to-end acceptance tests for narrative search surfaces (bd-16g.2.7).
//!
//! Validates:
//! 1. Direct storage facade (`execute_narrative_search` / `execute_narrative_around`).
//! 2. REST endpoints `GET /api/narrative/search` and `GET /api/narrative/around/{tick}`.
//! 3. FastMCP tool execution for `narrative_search` and `narrative_around`.
//! 4. Cross-surface parity (identical hit count, ordering, and fields).
//! 5. Input validation error handling (empty query, too long, bounds, window).
//! 6. Human table formatting and JSON serialization.

use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::PathBuf,
    time::Duration,
};

use anyhow::{Context, Result, anyhow};
use scriptbots_app::{
    CommandSubmit, ControlRuntime, ControlServerConfig,
    host_thread::HostThread,
    narrative_search::{
        MAX_NARRATIVE_PAGE_LIMIT, MAX_NARRATIVE_QUERY_BYTES, NarrativeAroundQuery,
        NarrativeSearchHitDto, NarrativeSearchQuery, execute_narrative_around,
        execute_narrative_search, format_hits_table,
    },
};
use scriptbots_core::{
    PersistenceBatch, ScriptBotsConfig, Tick, TickSummary, WorldState,
    narrative::{EventKind, EventRecord},
};
use scriptbots_runtime::{
    HostCoreOptions, HostSessionId, VolatileJournal, channel::ChannelHostOptions,
};
use scriptbots_storage::StoragePipeline;
use serde_json::Value;
use serial_test::serial;

fn unused_loopback_address() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("ephemeral port");
    let address = listener.local_addr().expect("bound address");
    drop(listener);
    address
}

fn control_cli_cmd() -> std::process::Command {
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_control_cli"));
    for (name, _) in std::env::vars_os() {
        if name.as_encoded_bytes().starts_with(b"SCRIPTBOTS_")
            || name.as_encoded_bytes().starts_with(b"SB_")
        {
            cmd.env_remove(name);
        }
    }
    cmd
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

fn initialize_mcp(mcp_addr: SocketAddr) {
    let (status, res) = mcp_json_rpc(
        mcp_addr,
        Some(1),
        "initialize",
        Some(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "narrative-e2e-test",
                "version": "1.0.0"
            }
        })),
    )
    .expect("mcp initialize helper");
    assert_eq!(status, 200);
    assert_eq!(res["jsonrpc"], "2.0");
    assert_eq!(res["result"]["protocolVersion"], "2024-11-05");
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

struct NarrativeFixture {
    _tempdir: tempfile::TempDir,
    db_path: PathBuf,
    host: Option<HostThread>,
    runtime: Option<ControlRuntime>,
    _submit: Option<CommandSubmit>,
    pub rest_addr: SocketAddr,
    pub mcp_addr: SocketAddr,
}

impl Drop for NarrativeFixture {
    fn drop(&mut self) {
        if let Some(rt) = self.runtime.take() {
            let _ = rt.shutdown();
        }
        self._submit.take();
        if let Some(h) = self.host.take() {
            let _ = h.join();
        }
    }
}

fn make_event(tick: u64, kind: EventKind, severity: f32, text: &str) -> EventRecord {
    EventRecord {
        schema_version: 1,
        tick: Tick(tick),
        kind,
        severity,
        magnitude: 1.0,
        window: (tick.saturating_sub(10), tick),
        metric: "population".to_string(),
        before: 100.0,
        after: 50.0,
        score: 1.0,
        subject: None,
        human_text: text.to_string(),
    }
}

fn setup_narrative_fixture() -> NarrativeFixture {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db_path = tempdir.path().join("narrative_e2e.sqlite");

    // Pre-populate FrankenSQLite database with seeded narrative events
    {
        let db_str = db_path.to_str().expect("db path utf8");
        let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(
            db_str, 1_000, 1_000, 1_000, 1_000,
        )
        .expect("storage pipeline creation");

        let events = vec![
            make_event(
                10,
                EventKind::PopulationCrash,
                0.8,
                "Catastrophic population crash: colony dwindled rapidly",
            ),
            make_event(
                20,
                EventKind::Extinction,
                0.95,
                "Sudden extinction event: lineage alpha perished completely",
            ),
            make_event(
                50,
                EventKind::PopulationBoom,
                0.7,
                "Population boom: abundant food resources enabled rapid expansion",
            ),
            make_event(
                100,
                EventKind::SpeciationHint,
                0.6,
                "Novel speciation detected: divergence in lineage beta",
            ),
            make_event(
                150,
                EventKind::ResourceCollapse,
                0.85,
                "Severe drought in northern quadrant caused starvation",
            ),
        ];

        for event in events {
            let tick = event.tick.0;
            let batch = PersistenceBatch {
                summary: TickSummary {
                    tick: Tick(tick),
                    agent_count: 50,
                    births: 0,
                    deaths: 0,
                    total_energy: 500.0,
                    average_energy: 10.0,
                    average_health: 1.0,
                    max_age: 100,
                    spike_hits: 0,
                },
                epoch: 0,
                closed: false,
                metrics: Vec::new(),
                events: Vec::new(),
                agents: Vec::new(),
                births: Vec::new(),
                deaths: Vec::new(),
                replay_events: Vec::new(),
                narrative_events: vec![event],
                genomes: Vec::new(),
            };
            pipeline.submit(&batch).expect("batch admitted");
        }

        let receipt = pipeline.shutdown().expect("pipeline shutdown");
        assert!(receipt.committed_tick.is_some(), "database must be flushed");
    }

    // Now start the control server attached to this database
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 64,
        world_height: 64,
        food_cell_size: 16,
        rng_seed: Some(42),
        persistence_interval: 0,
        closed: true,
        ..ScriptBotsConfig::default()
    })
    .expect("world state");
    world.step().expect("step world");

    let persistence = world
        .bind_persistence(Box::new(scriptbots_core::NullPersistence))
        .expect("persistence binding");
    let host = HostThread::spawn(
        HostSessionId::new(0x4242),
        world,
        persistence,
        Box::new(VolatileJournal::default()),
        HostCoreOptions::default(),
        ChannelHostOptions::default(),
    )
    .expect("host thread");

    let rest_addr = unused_loopback_address();
    let mcp_addr = unused_loopback_address();
    let config = ControlServerConfig {
        rest_address: rest_addr,
        rest_enabled: true,
        mcp_transport: scriptbots_app::McpTransportConfig::Http {
            bind_address: mcp_addr,
        },
        database_path: Some(db_path.clone()),
        ..ControlServerConfig::default()
    };

    let (runtime, submit) =
        ControlRuntime::launch(host.port().clone(), config).expect("control runtime launch");

    std::thread::sleep(Duration::from_millis(150));
    initialize_mcp(mcp_addr);

    NarrativeFixture {
        _tempdir: tempdir,
        db_path,
        host: Some(host),
        runtime: Some(runtime),
        _submit: Some(submit),
        rest_addr,
        mcp_addr,
    }
}

#[test]
#[serial]
fn test_narrative_search_cross_surface_parity_and_ordering() {
    let fixture = setup_narrative_fixture();

    // 1. Direct Facade search for "population"
    let direct_hits = execute_narrative_search(
        Some(&fixture.db_path),
        None,
        NarrativeSearchQuery {
            query: "population".into(),
            from_tick: None,
            to_tick: None,
            limit: None,
        },
    )
    .expect("direct facade search");
    assert_eq!(direct_hits.len(), 2, "expected 2 hits for 'population'");
    assert_eq!(direct_hits[0].tick, 10);
    assert_eq!(direct_hits[0].kind, "population_crash");
    assert_eq!(direct_hits[1].tick, 50);
    assert_eq!(direct_hits[1].kind, "population_boom");

    // 2. REST search for "population"
    let (status, body) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/narrative/search?q=population",
        &[],
        None,
    )
    .expect("rest narrative search");
    assert_eq!(status, 200);
    let rest_hits: Vec<NarrativeSearchHitDto> =
        serde_json::from_str(&body).expect("parse rest hits");
    assert_eq!(rest_hits.len(), 2);

    // Cross-surface bit parity: direct vs REST
    assert_eq!(
        direct_hits, rest_hits,
        "direct and REST hits must be bit-identical"
    );

    // 3. FastMCP search for "population"
    let (mcp_status, mcp_resp) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(100),
        "tools/call",
        Some(serde_json::json!({
            "name": "narrative_search",
            "arguments": {
                "query": "population"
            }
        })),
    )
    .expect("mcp narrative search");
    assert_eq!(mcp_status, 200);

    let content_text = mcp_resp["result"]["content"][0]["text"]
        .as_str()
        .expect("mcp text content");
    let mcp_hits: Vec<NarrativeSearchHitDto> =
        serde_json::from_str(content_text).expect("parse mcp hits");
    assert_eq!(mcp_hits.len(), 2);

    // Cross-surface bit parity: direct vs MCP
    assert_eq!(
        direct_hits, mcp_hits,
        "direct and MCP hits must be bit-identical"
    );

    // 4. control_cli narrative search --db (offline storage facade)
    let mut cli_db = control_cli_cmd();
    cli_db.args([
        "narrative",
        "search",
        "population",
        "--db",
        fixture.db_path.to_str().expect("utf8 path"),
        "--json",
    ]);
    let cli_db_out = cli_db.output().expect("control_cli search --db output");
    assert!(
        cli_db_out.status.success(),
        "cli --db search failed (status={:?}):\n--- STDOUT ---\n{}\n--- STDERR ---\n{}",
        cli_db_out.status.code(),
        String::from_utf8_lossy(&cli_db_out.stdout),
        String::from_utf8_lossy(&cli_db_out.stderr)
    );
    let cli_db_hits: Vec<NarrativeSearchHitDto> =
        serde_json::from_slice(&cli_db_out.stdout).expect("parse cli db hits");
    assert_eq!(
        direct_hits, cli_db_hits,
        "direct and CLI --db hits must be bit-identical"
    );

    // 5. control_cli narrative search --base-url (REST surface)
    let mut cli_rest = control_cli_cmd();
    cli_rest.args([
        "--base-url",
        &format!("http://{}", fixture.rest_addr),
        "narrative",
        "search",
        "population",
        "--json",
    ]);
    let cli_rest_out = cli_rest
        .output()
        .expect("control_cli search --base-url output");
    assert!(
        cli_rest_out.status.success(),
        "cli --base-url search failed: {}",
        String::from_utf8_lossy(&cli_rest_out.stderr)
    );
    let cli_rest_hits: Vec<NarrativeSearchHitDto> =
        serde_json::from_slice(&cli_rest_out.stdout).expect("parse cli rest hits");
    assert_eq!(
        direct_hits, cli_rest_hits,
        "direct and CLI --base-url hits must be bit-identical"
    );

    // 6. control_cli narrative search table output
    let mut cli_table = control_cli_cmd();
    cli_table.args([
        "narrative",
        "search",
        "population",
        "--db",
        fixture.db_path.to_str().expect("utf8 path"),
    ]);
    let cli_table_out = cli_table.output().expect("control_cli table output");
    assert!(cli_table_out.status.success());
    let table_str = String::from_utf8_lossy(&cli_table_out.stdout);
    assert!(table_str.contains("population_crash"));
    assert!(table_str.contains("population_boom"));

    println!(
        "E2E_EVIDENCE: {}",
        serde_json::json!({
            "schema": "scriptbots.narrative-search.e2e-evidence.v1",
            "phase": "cross_surface_parity_confirmed",
            "surfaces": ["direct_facade", "rest", "mcp", "cli_db", "cli_rest"],
            "query": "population",
            "hit_count": direct_hits.len(),
            "first_hit_tick": direct_hits.first().map(|h| h.tick),
            "status": "pass"
        })
    );
}

#[test]
#[serial]
fn test_narrative_around_cross_surface_parity() {
    let fixture = setup_narrative_fixture();

    // Center tick 20, window 15 => ticks [5, 35]
    // Should capture tick 10 (population crash) and tick 20 (extinction)

    // 1. Direct Facade around query
    let direct_hits = execute_narrative_around(
        Some(&fixture.db_path),
        None,
        NarrativeAroundQuery {
            tick: 20,
            window: Some(15),
        },
    )
    .expect("direct facade around");
    assert_eq!(direct_hits.len(), 2);
    assert_eq!(direct_hits[0].tick, 10);
    assert_eq!(direct_hits[1].tick, 20);

    // 2. REST around query
    let (status, body) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/narrative/around/20?window=15",
        &[],
        None,
    )
    .expect("rest narrative around");
    assert_eq!(status, 200);
    let rest_hits: Vec<NarrativeSearchHitDto> =
        serde_json::from_str(&body).expect("parse rest hits");
    assert_eq!(direct_hits, rest_hits);

    // 3. FastMCP around query
    let (mcp_status, mcp_resp) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(200),
        "tools/call",
        Some(serde_json::json!({
            "name": "narrative_around",
            "arguments": {
                "tick": 20,
                "window": 15
            }
        })),
    )
    .expect("mcp narrative around");
    assert_eq!(mcp_status, 200);

    let content_text = mcp_resp["result"]["content"][0]["text"]
        .as_str()
        .expect("mcp text content");
    let mcp_hits: Vec<NarrativeSearchHitDto> =
        serde_json::from_str(content_text).expect("parse mcp hits");
    assert_eq!(direct_hits, mcp_hits);

    // 4. control_cli narrative around --db
    let mut cli_around_db = control_cli_cmd();
    cli_around_db.args([
        "narrative",
        "around",
        "20",
        "--window",
        "15",
        "--db",
        fixture.db_path.to_str().expect("utf8 path"),
        "--json",
    ]);
    let cli_around_db_out = cli_around_db
        .output()
        .expect("control_cli around --db output");
    assert!(
        cli_around_db_out.status.success(),
        "cli --db around failed: {}",
        String::from_utf8_lossy(&cli_around_db_out.stderr)
    );
    let cli_around_db_hits: Vec<NarrativeSearchHitDto> =
        serde_json::from_slice(&cli_around_db_out.stdout).expect("parse cli around db hits");
    assert_eq!(
        direct_hits, cli_around_db_hits,
        "direct and CLI around --db hits must be bit-identical"
    );

    // 5. control_cli narrative around --base-url
    let mut cli_around_rest = control_cli_cmd();
    cli_around_rest.args([
        "--base-url",
        &format!("http://{}", fixture.rest_addr),
        "narrative",
        "around",
        "20",
        "--window",
        "15",
        "--json",
    ]);
    let cli_around_rest_out = cli_around_rest
        .output()
        .expect("control_cli around --base-url output");
    assert!(
        cli_around_rest_out.status.success(),
        "cli --base-url around failed: {}",
        String::from_utf8_lossy(&cli_around_rest_out.stderr)
    );
    let cli_around_rest_hits: Vec<NarrativeSearchHitDto> =
        serde_json::from_slice(&cli_around_rest_out.stdout).expect("parse cli around rest hits");
    assert_eq!(
        direct_hits, cli_around_rest_hits,
        "direct and CLI around --base-url hits must be bit-identical"
    );

    println!(
        "E2E_EVIDENCE: {}",
        serde_json::json!({
            "schema": "scriptbots.narrative-search.e2e-evidence.v1",
            "phase": "around_cross_surface_parity_confirmed",
            "surfaces": ["direct_facade", "rest", "mcp", "cli_db", "cli_rest"],
            "around_tick": 20,
            "window": 15,
            "hit_count": direct_hits.len(),
            "status": "pass"
        })
    );
}

#[test]
#[serial]
fn test_narrative_search_tick_bounds_and_limit() {
    let fixture = setup_narrative_fixture();

    // Search "population" with from_tick=20
    let (status, body) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/narrative/search?q=population&from=20",
        &[],
        None,
    )
    .expect("rest query");
    assert_eq!(status, 200);
    let hits: Vec<NarrativeSearchHitDto> = serde_json::from_str(&body).expect("parse hits");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].tick, 50);

    // Search with limit=1
    let (status, body) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/narrative/search?q=population&limit=1",
        &[],
        None,
    )
    .expect("rest query");
    assert_eq!(status, 200);
    let hits: Vec<NarrativeSearchHitDto> = serde_json::from_str(&body).expect("parse hits");
    assert_eq!(hits.len(), 1);
}

#[test]
#[serial]
fn test_narrative_validation_errors_consistent_envelope() {
    let fixture = setup_narrative_fixture();

    // 1. Empty query
    let (status, body) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/narrative/search?q=",
        &[],
        None,
    )
    .expect("rest empty query");
    assert_eq!(status, 400);
    assert!(body.contains("empty") || body.contains("whitespace"));

    let (_, mcp_res) = mcp_json_rpc(
        fixture.mcp_addr,
        Some(301),
        "tools/call",
        Some(serde_json::json!({
            "name": "narrative_search",
            "arguments": {"query": ""}
        })),
    )
    .expect("mcp empty query");
    let is_err = mcp_res["error"]["code"] == -32602
        || mcp_res["result"]["isError"] == true
        || mcp_res["result"]["is_error"] == true;
    assert!(is_err, "empty query must be rejected: {mcp_res:?}");
    let err_str = serde_json::to_string(&mcp_res).unwrap_or_default();
    assert!(
        err_str.contains("empty") || err_str.contains("whitespace"),
        "error message should cite empty query: {err_str}"
    );

    // 2. Whitespace query
    let (status, _) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/narrative/search?q=%20%20%20",
        &[],
        None,
    )
    .expect("rest whitespace query");
    assert_eq!(status, 400);

    // 3. Query too long
    let long_query = "a".repeat(MAX_NARRATIVE_QUERY_BYTES + 1);
    let (status, body) = http_request(
        fixture.rest_addr,
        "GET",
        &format!("/api/narrative/search?q={long_query}"),
        &[],
        None,
    )
    .expect("rest long query");
    assert_eq!(status, 400);
    assert!(body.contains("exceeds maximum"));

    // 4. Invalid tick range (from > to)
    let (status, body) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/narrative/search?q=population&from=100&to=10",
        &[],
        None,
    )
    .expect("rest invalid range");
    assert_eq!(status, 400);
    assert!(body.contains("less than or equal to"));

    // 5. Limit too large (> 4096)
    let (status, body) = http_request(
        fixture.rest_addr,
        "GET",
        &format!(
            "/api/narrative/search?q=population&limit={}",
            MAX_NARRATIVE_PAGE_LIMIT + 1
        ),
        &[],
        None,
    )
    .expect("rest limit too large");
    assert_eq!(status, 400);
    assert!(body.contains("exceeds maximum allowable"));

    // 6. Window too large (> 10000)
    let (status, body) = http_request(
        fixture.rest_addr,
        "GET",
        "/api/narrative/around/50?window=10001",
        &[],
        None,
    )
    .expect("rest window too large");
    assert_eq!(status, 400);
    assert!(body.contains("exceeds maximum allowable"));
}

#[test]
fn test_formatters_table_and_json() {
    let hits = vec![
        NarrativeSearchHitDto {
            tick: 10,
            kind: "population_crash".into(),
            severity: 0.8,
            human_text: "Severe crash in colony alpha".into(),
            score: Some(-2.5),
        },
        NarrativeSearchHitDto {
            tick: 50,
            kind: "boom".into(),
            severity: 0.7,
            human_text: "Resource boom".into(),
            score: None,
        },
    ];

    // Table formatting
    let table = format_hits_table(&hits);
    assert!(table.contains("TICK"));
    assert!(table.contains("KIND"));
    assert!(table.contains("SEVERITY"));
    assert!(table.contains("SCORE"));
    assert!(table.contains("TEXT"));
    assert!(table.contains("10"));
    assert!(table.contains("population_crash"));
    assert!(table.contains("Severe crash in colony alpha"));
    assert!(table.contains("-2.500"));
    assert!(table.contains("50"));
    assert!(table.contains("boom"));

    // Empty table
    let empty_table = format_hits_table(&[]);
    assert!(empty_table.contains("No narrative events found"));

    // JSON serialization
    let json_str = serde_json::to_string(&hits).expect("serialize json");
    let deserialized: Vec<NarrativeSearchHitDto> =
        serde_json::from_str(&json_str).expect("deserialize json");
    assert_eq!(deserialized, hits);
}
