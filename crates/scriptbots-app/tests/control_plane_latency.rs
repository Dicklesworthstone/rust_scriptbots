//! bd-134 acceptance: the control plane must stay responsive while the
//! simulation is stepping at full speed.
//!
//! The failure this guards against was measured in the bd-134 audit: every
//! REST/MCP handler parked a tokio worker on the world mutex, so `num_cpus`
//! concurrent SSE clients froze the whole control plane. The fixes were
//! (1) blocking-pool wraps for every world-locking handler, (2) a published
//! lock-free latest-summary slot, and (3) capture-then-rasterize screenshots.
//! This harness drives all of that end to end: a real REST server, a real
//! stepping loop contending on the real world mutex, saturating SSE clients,
//! and a latency-measured request loop — then prints the histogram table the
//! bead requires and asserts the acceptance numbers.
//!
//! DSR lane only: `#[ignore]` keeps it out of the fast suite; the centralized
//! DSR profile runs it explicitly on controlled hardware where the acceptance
//! numbers are meaningful.

use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use scriptbots_app::{
    ControlRuntime, ControlServerConfig, McpTransportConfig, WorldStepDriver,
    control::empty_latest_summary,
};
use scriptbots_core::{AgentData, Position, ScriptBotsConfig, WorldState};

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

/// Production-shaped driver: locks the world per tick and publishes the
/// completed summary into the lock-free slot, exactly like
/// `persistence_step_driver` in `main.rs`.
fn publishing_step_driver(
    world: &Arc<Mutex<WorldState>>,
    latest: &scriptbots_app::control::SharedLatestSummary,
) -> WorldStepDriver {
    let world = Arc::clone(world);
    let latest = Arc::clone(latest);
    Arc::new(move || {
        let mut world = world.lock().expect("latency world mutex");
        let events = world.step()?;
        if let Some(summary) = world.history().next_back() {
            latest.store(Some(Arc::new(summary.clone())));
        }
        Ok(events)
    })
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
#[ignore = "DSR latency lane (bd-134): 1k agents, saturating SSE clients, wall-clock acceptance numbers"]
fn control_plane_latency_holds_under_stepping_and_sse_load() {
    let world = Arc::new(Mutex::new(populated_world()));
    let latest = empty_latest_summary();
    let driver = publishing_step_driver(&world, &latest);

    let rest_address = unused_loopback_address();
    let config = ControlServerConfig {
        rest_address,
        rest_enabled: true,
        mcp_transport: McpTransportConfig::Disabled,
        ..ControlServerConfig::default()
    };
    let (runtime, _drain, _submit) =
        ControlRuntime::launch(Arc::clone(&world), Arc::clone(&latest), config)
            .expect("REST startup for latency harness");

    // Stepping loop: full speed, production lock pattern, no frame pacing —
    // the worst realistic contention the control plane can face.
    let stop = Arc::new(AtomicBool::new(false));
    let ticks = Arc::new(AtomicU64::new(0));
    let stepper = {
        let stop = Arc::clone(&stop);
        let ticks = Arc::clone(&ticks);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                driver().expect("latency world step");
                ticks.fetch_add(1, Ordering::Relaxed);
            }
        })
    };

    // Phase A: unloaded baseline throughput.
    let baseline_start = ticks.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_secs(PHASE_SECONDS));
    let baseline_ticks = ticks.load(Ordering::Relaxed) - baseline_start;

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

    let loaded_start_ticks = ticks.load(Ordering::Relaxed);
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
    let loaded_ticks = ticks.load(Ordering::Relaxed) - loaded_start_ticks;

    stop.store(true, Ordering::Relaxed);
    stepper.join().expect("stepper joins");
    for client in sse_clients {
        client.join().expect("SSE client joins");
    }
    runtime.shutdown().expect("REST shutdown");

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
