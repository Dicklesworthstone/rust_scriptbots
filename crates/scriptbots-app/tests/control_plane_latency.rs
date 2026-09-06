//! bd-134 acceptance: the control plane must stay responsive while the
//! simulation is stepping at full speed.
//!
//! The failure this guards against was measured in the bd-134 audit: every
//! REST/MCP handler parked a tokio worker on the world mutex, so `num_cpus`
//! concurrent SSE clients froze the whole control plane. The migrated harness
//! runs a real sole-owner host, a real REST server, saturating SSE clients and
//! a measured request loop. The parked-owner test below checks immutable status
//! reads separately; neither fixture is evidence of an in-flight database stall.
//!
//! DSR lane only: `#[ignore]` keeps it out of the fast suite; the centralized
//! DSR profile runs it explicitly on controlled hardware where the acceptance
//! numbers are meaningful.

use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use scriptbots_app::{
    ControlRuntime, ControlServerConfig, McpTransportConfig, host_thread::HostThread,
};
use scriptbots_core::{AgentData, Position, ScriptBotsConfig, WorldState};
use scriptbots_runtime::{
    FixedDeadlineHost, HostCore, HostCoreOptions, HostSessionId, ManualInstant, PlaybackSnapshot,
    VolatileJournal,
    channel::{ChannelHostDriver, ChannelHostOptions},
};

const AGENT_COUNT: usize = 1_000;
const MEASURED_REQUESTS: usize = 200;
const PHASE_SECONDS: u64 = 3;
const LATENCY_P95_BUDGET: Duration = Duration::from_millis(50);
/// The bead's acceptance bound: simulation throughput under client load must
/// stay within 10% of the unloaded baseline.
const MAX_TICK_DEGRADATION: f64 = 0.10;

fn unused_loopback_address() -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("ephemeral port");
    let address = listener.local_addr().expect("bound address");
    drop(listener);
    address
}

fn populated_world() -> WorldState {
    let mut world = WorldState::new(ScriptBotsConfig {
        rng_seed: Some(0xB134_1A7E),
        persistence_interval: 0,
        ..ScriptBotsConfig::default()
    })
    .expect("latency world");
    let (width, height) = (
        world.config().world_width as f32,
        world.config().world_height as f32,
    );
    for index in 0..AGENT_COUNT {
        // Deterministic scatter; no RNG needed for a load fixture.
        let fraction = index as f32 / AGENT_COUNT as f32;
        let agent = AgentData {
            position: Position {
                x: (fraction * 0.9).mul_add(width, width * 0.05),
                y: ((fraction * 7.0).fract() * 0.9).mul_add(height, height * 0.05),
            },
            ..AgentData::default()
        };
        world.try_spawn_agent(agent).expect("spawn load agent");
    }
    world
}

/// Minimal blocking HTTP GET over a fresh connection; returns the whole
/// response (headers + body). Good enough for latency measurement of small
/// JSON bodies served with `content-length`.
fn http_get(address: std::net::SocketAddr, path: &str) -> std::io::Result<String> {
    let mut stream = TcpStream::connect(address)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

fn percentile(sorted: &[Duration], quantile: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

#[test]
fn status_http_returns_the_published_boundary_before_the_parked_owner_is_released() {
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 64,
        world_height: 64,
        food_cell_size: 16,
        rng_seed: Some(0x513B),
        persistence_interval: 0,
        closed: true,
        ..ScriptBotsConfig::default()
    })
    .expect("real status world");
    world.step().expect("actual published tick");
    let (ready, port_rx) = std::sync::mpsc::sync_channel(1);
    let (release, wait_for_release) = std::sync::mpsc::channel();
    let owner = std::thread::spawn(move || {
        let core = HostCore::new(
            HostSessionId::new(0x513b),
            world,
            HostCoreOptions {
                initial_playback: PlaybackSnapshot {
                    paused: true,
                    speed_multiplier: 1.0,
                },
                ..HostCoreOptions::default()
            },
        )
        .expect("status owner");
        let (mut driver, port) =
            ChannelHostDriver::new(FixedDeadlineHost::new(core), ChannelHostOptions::default())
                .expect("channel driver");
        ready.send(port).expect("publish actual owner snapshot");
        wait_for_release
            .recv()
            .expect("release owner after HTTP read");
        let epoch = Instant::now();
        driver
            .run(|| ManualInstant::from_nanos(u64::try_from(epoch.elapsed().as_nanos()).unwrap()))
            .expect("owner completes shutdown");
    });
    let rest_address = unused_loopback_address();
    let (runtime, submit) = ControlRuntime::launch(
        port_rx.recv().expect("ready owner"),
        ControlServerConfig {
            rest_address,
            rest_enabled: true,
            mcp_transport: McpTransportConfig::Disabled,
            ..ControlServerConfig::default()
        },
    )
    .expect("real REST startup");
    let response = http_get(rest_address, "/api/status");
    // Release before asserting so a regressed handler can finish and teardown can join.
    release.send(()).expect("release owner");
    runtime.shutdown().expect("REST shutdown");
    drop(submit);
    owner.join().expect("owner joined");
    let response = response.expect("status returns before the owner is released");
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let (_, body) = response.split_once("\r\n\r\n").expect("HTTP body");
    let status: scriptbots_app::control::SimulationStatusDto =
        serde_json::from_str(body).expect("actual status JSON");
    assert_eq!(status.tick, 1);
    assert_eq!(status.agent_count, 0);
    assert!(status.is_closed);
    assert_eq!(status.config_revision, 0);
}

#[test]
#[ignore = "DSR latency lane (bd-134): 1k agents, saturating SSE clients, wall-clock acceptance numbers"]
fn control_plane_latency_holds_under_stepping_and_sse_load() {
    let world = populated_world();
    let persistence = world
        .bind_persistence(Box::new(scriptbots_core::NullPersistence))
        .expect("persistence binding");
    let host = HostThread::spawn(
        HostSessionId::new(0xb134),
        world,
        persistence,
        Box::new(VolatileJournal::default()),
        HostCoreOptions {
            // A one-nanosecond requested period keeps the real owner under load;
            // its bounded catch-up policy still controls actual science work.
            tick_period_nanos: 1,
            ..HostCoreOptions::default()
        },
        ChannelHostOptions::default(),
    )
    .expect("full-speed owner");
    let snapshots = host.port().snapshot_hub();

    let rest_address = unused_loopback_address();
    let config = ControlServerConfig {
        rest_address,
        rest_enabled: true,
        mcp_transport: McpTransportConfig::Disabled,
        ..ControlServerConfig::default()
    };
    let (runtime, submit) =
        ControlRuntime::launch(host.port(), config).expect("REST startup for latency harness");

    let stop = Arc::new(AtomicBool::new(false));

    // Phase A: unloaded baseline throughput.
    let baseline_start = snapshots.latest().world.tick;
    std::thread::sleep(Duration::from_secs(PHASE_SECONDS));
    let baseline_ticks = snapshots.latest().world.tick - baseline_start;

    // Phase B: saturate with SSE clients, then measure request latencies.
    let client_count = 2 * std::thread::available_parallelism().map_or(4, usize::from);
    let sse_events = Arc::new(AtomicU64::new(0));
    let mut sse_clients = Vec::new();
    for _ in 0..client_count {
        let stop = Arc::clone(&stop);
        let sse_events = Arc::clone(&sse_events);
        sse_clients.push(std::thread::spawn(move || {
            let Ok(mut stream) = TcpStream::connect(rest_address) else {
                return;
            };
            let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
            let request = format!(
                "GET /api/ticks/stream HTTP/1.1\r\nHost: {rest_address}\r\nAccept: text/event-stream\r\n\r\n"
            );
            if stream.write_all(request.as_bytes()).is_err() {
                return;
            }
            let mut buffer = [0_u8; 4096];
            while !stop.load(Ordering::Relaxed) {
                match stream.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(read) => {
                        let chunk = String::from_utf8_lossy(&buffer[..read]);
                        sse_events
                            .fetch_add(chunk.matches("data:").count() as u64, Ordering::Relaxed);
                    }
                    // Timeouts just mean no event inside the poll window.
                    Err(_) => {}
                }
            }
        }));
    }

    // Let the SSE herd attach before measuring.
    std::thread::sleep(Duration::from_millis(750));

    let loaded_start_ticks = snapshots.latest().world.tick;
    let loaded_start = Instant::now();
    let mut latencies = Vec::with_capacity(MEASURED_REQUESTS);
    let mut failures = 0_usize;
    for _ in 0..MEASURED_REQUESTS {
        let begin = Instant::now();
        match http_get(rest_address, "/api/ticks/latest") {
            Ok(response) if response.starts_with("HTTP/1.1 200") => {
                latencies.push(begin.elapsed());
            }
            Ok(_) | Err(_) => failures += 1,
        }
    }
    let loaded_elapsed = loaded_start.elapsed().max(Duration::from_millis(1));
    let loaded_ticks = snapshots.latest().world.tick - loaded_start_ticks;

    stop.store(true, Ordering::Relaxed);
    for client in sse_clients {
        client.join().expect("SSE client joins");
    }
    runtime.shutdown().expect("REST shutdown");
    drop(submit);
    host.join().expect("owner shutdown");

    latencies.sort_unstable();
    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    let p99 = percentile(&latencies, 0.99);
    let baseline_tps = baseline_ticks as f64 / PHASE_SECONDS as f64;
    let loaded_tps = loaded_ticks as f64 / loaded_elapsed.as_secs_f64();
    let degradation = if baseline_tps > 0.0 {
        1.0 - (loaded_tps / baseline_tps)
    } else {
        0.0
    };

    // The bead requires the histogram table in the failure output; println!
    // is captured by the harness and replayed exactly when an assert fires.
    println!("control-plane latency harness (bd-134)");
    println!("  agents: {AGENT_COUNT}  sse-clients: {client_count}");
    println!("  endpoint                requests  fail  p50        p95        p99");
    println!(
        "  /api/ticks/latest       {:>8}  {failures:>4}  {p50:>9.2?}  {p95:>9.2?}  {p99:>9.2?}",
        latencies.len(),
    );
    println!(
        "  tick throughput: baseline {baseline_tps:.1} tps -> loaded {loaded_tps:.1} tps ({:.1}% degradation)",
        degradation * 100.0
    );
    println!(
        "  sse events observed across clients: {}",
        sse_events.load(Ordering::Relaxed)
    );

    assert_eq!(
        failures, 0,
        "every latest-summary request must succeed under load"
    );
    assert!(
        baseline_ticks > 0 && loaded_ticks > 0,
        "both phases must observe actual science progress"
    );
    assert!(
        sse_events.load(Ordering::Relaxed) > 0,
        "the SSE streams must stay live while the simulation steps"
    );
    assert!(
        p95 < LATENCY_P95_BUDGET,
        "latest-summary p95 {p95:?} exceeded the {LATENCY_P95_BUDGET:?} budget"
    );
    assert!(
        degradation < MAX_TICK_DEGRADATION,
        "client load degraded simulation throughput by {:.1}% (budget {:.0}%)",
        degradation * 100.0,
        MAX_TICK_DEGRADATION * 100.0
    );
}
