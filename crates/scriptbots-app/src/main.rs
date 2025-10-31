use anyhow::{Context, Result, bail};
use clap::{ArgAction, Parser, ValueEnum};
use owo_colors::OwoColorize;
use ron::ser::PrettyConfig as RonPrettyConfig;
use scriptbots_app::{
    ControlRuntime, ControlServerConfig, SharedStorage, SharedWorld,
    renderer::{Renderer, RendererContext},
    terminal::TerminalRenderer,
};
#[cfg(feature = "bevy_render")]
use scriptbots_bevy::{BevyRendererContext, render_png_offscreen as render_bevy_png};
use scriptbots_brain::MlpBrain;
use scriptbots_core::{
    AgentData, NeuroflowActivationKind, ReplayEventKind, ScriptBotsConfig, TickSummary,
    WorldPersistence, WorldState,
};
#[cfg(feature = "gui")]
use scriptbots_render::{render_png_offscreen, run_demo};
use scriptbots_storage::{PersistedReplayEvent, Storage, StorageError, StoragePipeline};
use serde_json::{self, Value as JsonValue};
use std::process::{Command, Stdio};
use std::{
    collections::HashMap,
    env, fmt, fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};
use tracing::{debug, info, warn};

#[cfg(feature = "fast-alloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<()> {
    let cli = AppCli::parse();
    init_tracing();

    // Determinism check child mode: run headless and emit JSON, then exit.
    if let Ok(flag) = env::var("SCRIPTBOTS_DET_RUN")
        && matches!(parse_bool(&flag), Some(true))
    {
        let ticks_env = env::var("SCRIPTBOTS_DET_TICKS").ok();
        let tick_limit = ticks_env.and_then(|s| s.parse::<u64>().ok()).unwrap_or(500);
        let config = compose_config(&cli)?;
        run_det_child(&config, tick_limit)?;
        return Ok(());
    }
    let config = compose_config(&cli)?;
    if let Some(outcome) = maybe_emit_config(&cli, &config)?
        && matches!(outcome, ConfigEmitOutcome::Exit)
    {
        return Ok(());
    }

    if cli.replay_db.is_some() {
        run_replay_cli(&cli, &config)?;
        return Ok(());
    }

    if let Some(ticks) = cli.det_check {
        run_det_check(&cli, ticks)?;
        return Ok(());
    }

    // Optional: profiling runs (headless). Execute and exit if specified.
    let thresholds = thresholds_from_cli(&cli);
    if cli.profile_steps.is_some() || cli.profile_storage_steps.is_some() {
        if let Some(ticks) = cli.profile_steps {
            profile_world_steps(&config, ticks)?;
        }
        if let Some(ticks) = cli.profile_storage_steps {
            profile_world_steps_with_storage(&config, ticks, cli.storage, thresholds)?;
        }
        return Ok(());
    }

    // Automated profiling sweep (child-process based)
    if let Some(ticks) = cli.profile_sweep {
        run_profile_sweep(&config, ticks, &cli)?;
        return Ok(());
    }

    // Auto-tune: run a quick sweep for the chosen storage mode, apply best settings, then continue.
    let mut thresholds = thresholds;
    if let Some(ticks) = cli.auto_tune {
        if let Some(best) =
            pick_best_for_storage(&config, ticks, cli.storage, cli.threads, cli.low_power)?
        {
            // Apply threads if not explicitly set
            if cli.threads.is_none() {
                unsafe {
                    std::env::set_var("SCRIPTBOTS_MAX_THREADS", best.threads.to_string());
                }
            }
            // Apply thresholds if not provided via CLI
            if cli.storage_thresholds.is_none() {
                thresholds = ThresholdsOverride {
                    tick: Some(best.tick),
                    agent: Some(best.agent),
                    event: Some(best.event),
                    metric: Some(best.metric),
                };
            }
            println!(
                "{} Auto-tune selected: threads={} storage={} thresholds={},{},{},{} ({:.0} tps)",
                "✔".green().bold(),
                best.threads,
                match cli.storage {
                    StorageMode::DuckDb => "duckdb",
                    StorageMode::Memory => "memory",
                },
                best.tick,
                best.agent,
                best.event,
                best.metric,
                best.tps
            );
        }
    }

    // Configure low-power / thread budget before world creation so the Rayon pool is capped.
    if let Some(threads) = cli.threads {
        unsafe {
            std::env::set_var("SCRIPTBOTS_MAX_THREADS", threads.to_string());
        }
    } else if cli.low_power {
        // Conservative default: 2 worker threads unless explicitly overridden by --threads
        unsafe {
            std::env::set_var("SCRIPTBOTS_MAX_THREADS", "2");
        }
    }

    // Apply OS-level priority niceness where supported.
    apply_process_niceness(cli.low_power)?;

    // Prefer high-performance adapter on Windows for wgpu
    #[cfg(windows)]
    unsafe {
        if std::env::var("WGPU_POWER_PREFERENCE").is_err() {
            std::env::set_var("WGPU_POWER_PREFERENCE", "high_performance");
        }
    }

    // Renderer debug toggles
    if cli.debug_watermark {
        unsafe {
            std::env::set_var("SCRIPTBOTS_RENDER_WATERMARK", "1");
        }
    }
    if cli.renderer_safe || cli.low_power {
        unsafe {
            std::env::set_var("SCRIPTBOTS_RENDER_SAFE", "1");
        }
    }
    // Prefer terminal renderer in low-power auto mode unless FORCE_GUI is explicitly set
    if cli.low_power
        && matches!(cli.mode, RendererMode::Auto)
        && !matches!(
            env::var("SCRIPTBOTS_FORCE_GUI")
                .ok()
                .and_then(|v| parse_bool(&v)),
            Some(true)
        )
    {
        unsafe {
            std::env::set_var("SCRIPTBOTS_FORCE_TERMINAL", "1");
        }
    }

    let (world, storage) = bootstrap_world(config, cli.storage, thresholds)?;

    // Optional: dump a PNG snapshot and exit (no UI launched).
    if let Some(path) = cli.dump_png.as_ref() {
        #[cfg(feature = "gui")]
        {
            let (w, h) = cli
                .png_size
                .as_deref()
                .and_then(parse_png_size)
                .unwrap_or((1600, 900));

            let bytes = {
                let guard = world.lock().expect("world mutex poisoned");
                // Prefer wgpu compositor path if requested via env; otherwise fallback CPU raster
                let bytes = if matches!(
                    std::env::var("SB_WGPU_DUMP").ok().as_deref(),
                    Some("1" | "true" | "yes" | "on")
                ) {
                    scriptbots_render::world_compositor::render_wgpu_png_offscreen(&guard, w, h)
                } else {
                    render_png_offscreen(&guard, w, h)
                };
                bytes
            };
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, &bytes)?;
            println!(
                "{} Wrote snapshot {} ({}x{})",
                "✔".green().bold(),
                path.display(),
                w,
                h
            );
            return Ok(());
        }
        #[cfg(not(feature = "gui"))]
        {
            // Avoid unused-variable warning when GUI is not enabled
            let _ = path;
            bail!("--dump-png requires GUI feature; recompile with --features gui");
        }
    }
    #[cfg(feature = "bevy_render")]
    if let Some(path) = cli.dump_bevy_png.as_ref() {
        let (w, h) = cli
            .png_size
            .as_deref()
            .and_then(parse_png_size)
            .unwrap_or((1600, 900));
        let bytes = {
            let guard = world.lock().expect("world mutex poisoned");
            render_bevy_png(&guard, w, h)?
        };
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &bytes)?;
        println!(
            "{} Wrote Bevy snapshot {} ({}x{})",
            "✔".green().bold(),
            path.display(),
            w,
            h
        );
        return Ok(());
    }
    let control_config = ControlServerConfig::from_env();
    let (control_runtime, command_drain, command_submit) =
        ControlRuntime::launch(world.clone(), control_config)?;
    let (active_mode, renderer) = resolve_renderer(cli.mode)?;
    info!(
        requested_mode = cli.mode.as_str(),
        active_mode = active_mode.as_str(),
        renderer = renderer.name(),
        "Starting ScriptBots simulation shell"
    );
    let context = RendererContext {
        world: Arc::clone(&world),
        storage: Arc::clone(&storage),
        control_runtime: &control_runtime,
        command_drain,
        command_submit,
    };
    renderer.run(context)?;
    control_runtime.shutdown()?;
    Ok(())
}

#[cfg(unix)]
fn apply_process_niceness(low_power: bool) -> Result<()> {
    use libc::{PRIO_PROCESS, id_t, setpriority};
    // Always reduce CPU priority a bit when low_power; otherwise keep default niceness.
    if low_power {
        unsafe {
            // niceness +10 (lower priority); ignore errors on restricted environments
            let _ = setpriority(PRIO_PROCESS as _, 0 as id_t, 10);
        }
    }
    // Best-effort I/O niceness via ionice class 3 (idle) where available.
    // There is no stable libc wrapper; attempt calling the ionice syscall number is fragile.
    // We intentionally skip ionice here and rely on OS tools if needed.
    Ok(())
}

#[cfg(windows)]
fn apply_process_niceness(low_power: bool) -> Result<()> {
    use windows_sys::Win32::System::Threading::{
        BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, SetPriorityClass,
    };
    unsafe {
        let handle = GetCurrentProcess();
        let class = if low_power {
            BELOW_NORMAL_PRIORITY_CLASS
        } else {
            0
        };
        if class != 0 {
            let _ = SetPriorityClass(handle, class);
        }
    }
    Ok(())
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}

fn run_det_child(config: &ScriptBotsConfig, tick_limit: u64) -> Result<()> {
    let run = run_headless_simulation(config, tick_limit)?;
    #[derive(serde::Serialize)]
    struct DetOut {
        events: usize,
        ticks: usize,
        last_tick: u64,
        summaries: Vec<TickSummary>,
    }
    let last_tick = run.summaries.last().map(|s| s.tick.0).unwrap_or(0);
    let out = DetOut {
        events: run.events.len(),
        ticks: run.summaries.len(),
        last_tick,
        summaries: run.summaries,
    };
    let json = serde_json::to_string(&out)?;
    println!("{}", json);
    Ok(())
}

fn run_det_check(_cli: &AppCli, ticks: u64) -> Result<()> {
    let exe = std::env::current_exe().context("failed to get current exe path")?;
    // Child 1: single-thread (force RAYON_NUM_THREADS=1)
    let mut child1 = Command::new(&exe);
    child1.arg("--config-only"); // avoid launching UI
    child1.env("SCRIPTBOTS_DET_RUN", "1");
    child1.env("SCRIPTBOTS_DET_TICKS", ticks.to_string());
    child1.env("RAYON_NUM_THREADS", "1");
    child1.env("RUST_LOG", "error");
    if let Ok(seed) = std::env::var("SCRIPTBOTS_DET_SEED") {
        child1.env("SCRIPTBOTS_RNG_SEED", seed);
    }
    child1.stdout(Stdio::piped());
    child1.stderr(Stdio::null());
    let handle1 = child1.spawn().context("failed to spawn det child 1")?;

    // Child N: default thread budget
    let mut childn = Command::new(&exe);
    childn.arg("--config-only");
    childn.env("SCRIPTBOTS_DET_RUN", "1");
    childn.env("SCRIPTBOTS_DET_TICKS", ticks.to_string());
    childn.env("RUST_LOG", "error");
    childn.stdout(Stdio::piped());
    childn.stderr(Stdio::null());
    let handlen = childn.spawn().context("failed to spawn det child N")?;

    // Wait for both to complete (they run concurrently)
    let out1 = handle1
        .wait_with_output()
        .context("failed to wait for det child 1")?;
    let outn = handlen
        .wait_with_output()
        .context("failed to wait for det child N")?;
    if !out1.status.success() {
        bail!("det child 1 failed: status {:?}", out1.status);
    }
    if !outn.status.success() {
        bail!("det child N failed: status {:?}", outn.status);
    }

    #[derive(serde::Deserialize)]
    struct DetOutIn {
        events: usize,
        ticks: usize,
        last_tick: u64,
        summaries: Vec<TickSummary>,
    }
    let left: DetOutIn =
        serde_json::from_slice(&out1.stdout).context("failed to parse child 1 JSON output")?;
    let right: DetOutIn =
        serde_json::from_slice(&outn.stdout).context("failed to parse child N JSON output")?;

    if left.ticks != right.ticks || left.last_tick != right.last_tick {
        bail!(
            "tick count/last tick mismatch: 1t {:?} vs Nt {:?}",
            left.ticks,
            right.ticks
        );
    }
    for (idx, (a, b)) in left
        .summaries
        .iter()
        .zip(right.summaries.iter())
        .enumerate()
    {
        if a != b {
            println!(
                "{} divergence at idx {} tick {}",
                "✖".red().bold(),
                idx,
                a.tick.0
            );
            println!(
                "    1t: agents={} births={} deaths={} avgE={:.4}",
                a.agent_count, a.births, a.deaths, a.average_energy
            );
            println!(
                "    Nt: agents={} births={} deaths={} avgE={:.4}",
                b.agent_count, b.births, b.deaths, b.average_energy
            );
            bail!("determinism self-check failed");
        }
    }
    println!(
        "{} Determinism self-check passed for {} ticks (events: 1t={}, Nt={})",
        "✔".green().bold(),
        ticks,
        left.events,
        right.events
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct ThresholdsOverride {
    tick: Option<usize>,
    agent: Option<usize>,
    event: Option<usize>,
    metric: Option<usize>,
}

fn thresholds_from_cli(cli: &AppCli) -> ThresholdsOverride {
    if let Some(raw) = cli.storage_thresholds.as_ref() {
        let mut parts = raw.split(',').map(|s| s.trim());
        let tick = parts.next().and_then(|p| p.parse::<usize>().ok());
        let agent = parts.next().and_then(|p| p.parse::<usize>().ok());
        let event = parts.next().and_then(|p| p.parse::<usize>().ok());
        let metric = parts.next().and_then(|p| p.parse::<usize>().ok());
        return ThresholdsOverride {
            tick,
            agent,
            event,
            metric,
        };
    }
    ThresholdsOverride::default()
}

fn bootstrap_world(
    config: ScriptBotsConfig,
    storage_mode: StorageMode,
    thresholds: ThresholdsOverride,
) -> Result<(SharedWorld, SharedStorage)> {
    let storage_path =
        env::var("SCRIPTBOTS_STORAGE_PATH").unwrap_or_else(|_| "scriptbots.db".to_string());
    if let Some(parent) = Path::new(&storage_path)
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    // Choose persistence strategy
    let (storage, mut world) = match storage_mode {
        StorageMode::DuckDb => {
            // Helper to try opening a pipeline with current thresholds
            let try_open = |path: &str| -> std::result::Result<StoragePipeline, StorageError> {
                match (
                    thresholds.tick,
                    thresholds.agent,
                    thresholds.event,
                    thresholds.metric,
                ) {
                    (Some(t), Some(a), Some(e), Some(m)) => {
                        StoragePipeline::with_thresholds(path, t, a, e, m)
                    }
                    _ => StoragePipeline::new(path),
                }
            };
            // First attempt
            let pipeline = match try_open(&storage_path) {
                Ok(p) => p,
                Err(err) => {
                    let msg = err.to_string();
                    let lock_error = msg.to_ascii_lowercase().contains("could not set lock")
                        || msg.to_ascii_lowercase().contains("conflicting lock");
                    if lock_error {
                        // Generate a safe fallback path alongside the requested path
                        let ts = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        let pid = std::process::id();
                        let fallback_path = {
                            let p = std::path::Path::new(&storage_path);
                            let parent = p.parent();
                            let (stem, ext) = (
                                p.file_stem()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("scriptbots"),
                                p.extension().and_then(|e| e.to_str()).unwrap_or("duckdb"),
                            );
                            let file = format!("{}.run-{}-{}.{}", stem, pid, ts, ext);
                            if let Some(dir) = parent {
                                dir.join(file)
                            } else {
                                std::path::PathBuf::from(file)
                            }
                        };
                        if let Some(parent) =
                            fallback_path.parent().filter(|d| !d.as_os_str().is_empty())
                        {
                            std::fs::create_dir_all(parent)?;
                        }
                        println!(
                            "{} {} is locked ({}). Falling back to {}",
                            "⚠".yellow().bold(),
                            storage_path.cyan(),
                            msg,
                            fallback_path.display().to_string().magenta().bold()
                        );
                        try_open(&fallback_path.to_string_lossy())?
                    } else {
                        return Err(err.into());
                    }
                }
            };
            let storage: SharedStorage = pipeline.storage();
            let world = WorldState::with_persistence(config, Box::new(pipeline))?;
            (storage, world)
        }
        StorageMode::Memory => {
            // In-memory DuckDB for analytics, avoids disk I/O
            let pipeline = StoragePipeline::with_thresholds(
                ":memory:",
                thresholds.tick.unwrap_or(64),
                thresholds.agent.unwrap_or(2048),
                thresholds.event.unwrap_or(512),
                thresholds.metric.unwrap_or(512),
            )?;
            let storage: SharedStorage = pipeline.storage();
            let world = WorldState::with_persistence(config, Box::new(pipeline))?;
            (storage, world)
        }
    };
    let brain_keys = install_brains(&mut world);

    seed_agents(&mut world, &brain_keys);

    for _ in 0..120 {
        world.step();
    }

    if let Some(summary) = world.history().last() {
        info!(
            tick = summary.tick.0,
            agents = summary.agent_count,
            births = summary.births,
            deaths = summary.deaths,
            avg_energy = summary.average_energy,
            "Primed world and persisted initial summary",
        );
    } else {
        warn!("World bootstrap completed without persistence summaries");
    }

    Ok((Arc::new(Mutex::new(world)), storage))
}

fn compose_config(cli: &AppCli) -> Result<ScriptBotsConfig> {
    let mut config = ScriptBotsConfig {
        persistence_interval: 60,
        history_capacity: 600,
        ..ScriptBotsConfig::default()
    };
    config = apply_config_layers(config, &cli.config_layers)?;
    apply_env_overrides(&mut config);
    if let Some(seed) = cli.rng_seed {
        config.rng_seed = Some(seed);
    }
    if let Some(limit) = cli.auto_pause_below {
        config.control.auto_pause_population_below = Some(limit);
    }
    if let Some(age) = cli.auto_pause_age_above {
        config.control.auto_pause_age_above = Some(age);
    }
    if cli.auto_pause_on_spike {
        config.control.auto_pause_on_spike_hit = true;
    }
    Ok(config)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConfigEmitOutcome {
    Continue,
    Exit,
}

fn maybe_emit_config(cli: &AppCli, config: &ScriptBotsConfig) -> Result<Option<ConfigEmitOutcome>> {
    if !cli.print_config && cli.write_config.is_none() {
        return Ok(None);
    }

    let rendered = match cli.config_format {
        ConfigFormat::Json => serde_json::to_string_pretty(config)?,
        ConfigFormat::Toml => toml::to_string_pretty(config)?,
        ConfigFormat::Ron => ron::ser::to_string_pretty(config, RonPrettyConfig::new())?,
    };

    if cli.print_config {
        println!("{}", rendered);
    }

    if let Some(path) = cli.write_config.as_ref() {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, rendered.as_bytes())?;
        info!(path = %path.display(), "Wrote composed configuration to file");
    }

    let outcome = if cli.config_only {
        ConfigEmitOutcome::Exit
    } else {
        ConfigEmitOutcome::Continue
    };

    Ok(Some(outcome))
}

#[derive(Parser, Debug)]
#[command(
    name = "scriptbots-app",
    version,
    about = "ScriptBots simulation shell"
)]
struct AppCli {
    /// Rendering mode for the simulation shell (auto detects GPUI fallback).
    #[arg(
        long,
        value_enum,
        env = "SCRIPTBOTS_MODE",
        default_value_t = RendererMode::Auto
    )]
    mode: RendererMode,
    /// Layered configuration files (TOML or RON) applied in order.
    #[arg(
        long = "config",
        value_name = "FILE",
        action = ArgAction::Append,
        env = "SCRIPTBOTS_CONFIG",
        value_delimiter = ';'
    )]
    config_layers: Vec<PathBuf>,
    /// RNG seed override for deterministic runs.
    #[arg(long = "rng-seed", value_name = "SEED", env = "SCRIPTBOTS_RNG_SEED")]
    rng_seed: Option<u64>,
    /// Path to a DuckDB run to verify via headless deterministic replay.
    #[arg(long = "replay-db", value_name = "FILE", env = "SCRIPTBOTS_REPLAY_DB")]
    replay_db: Option<PathBuf>,
    /// Optional comparison database for divergence analysis.
    #[arg(long = "compare-db", value_name = "FILE", requires = "replay_db")]
    compare_db: Option<PathBuf>,
    /// Limit the number of ticks simulated during replay verification.
    #[arg(long = "tick-limit", value_name = "TICKS", requires = "replay_db")]
    tick_limit: Option<u64>,
    /// Auto-pause when population is at or below this count.
    #[arg(
        long = "auto-pause-below",
        value_name = "COUNT",
        env = "SCRIPTBOTS_AUTO_PAUSE_BELOW"
    )]
    auto_pause_below: Option<u32>,
    /// Auto-pause when any agent's age meets or exceeds this value.
    #[arg(
        long = "auto-pause-age-above",
        value_name = "AGE",
        env = "SCRIPTBOTS_AUTO_PAUSE_AGE_ABOVE"
    )]
    auto_pause_age_above: Option<u32>,
    /// Auto-pause after a spike hit is recorded.
    #[arg(
        long = "auto-pause-on-spike",
        action = ArgAction::SetTrue,
        env = "SCRIPTBOTS_AUTO_PAUSE_ON_SPIKE"
    )]
    auto_pause_on_spike: bool,
    /// Print the composed configuration in the selected format.
    #[arg(long = "print-config", action = ArgAction::SetTrue)]
    print_config: bool,
    /// Write the composed configuration to the provided path (directories created as needed).
    #[arg(long = "write-config", value_name = "FILE")]
    write_config: Option<PathBuf>,
    /// Output format for `--print-config` / `--write-config`.
    #[arg(long = "config-format", value_enum, default_value_t = ConfigFormat::Json)]
    config_format: ConfigFormat,
    /// Exit immediately after emitting configuration output.
    #[arg(long = "config-only", action = ArgAction::SetTrue)]
    config_only: bool,
    /// Run determinism self-check comparing 1-thread vs N-threads for the given number of ticks.
    #[arg(long = "det-check", value_name = "TICKS")]
    det_check: Option<u64>,
    /// Overlay a tiny debug watermark in the render canvas (diagnostics).
    #[arg(long = "debug-watermark", action = ArgAction::SetTrue)]
    debug_watermark: bool,
    /// Force a conservative canvas paint path (diagnostics on Windows black canvas).
    #[arg(long = "renderer-safe", action = ArgAction::SetTrue)]
    renderer_safe: bool,
    /// Cap simulation worker threads (overrides low-power default).
    #[arg(long = "threads", value_name = "N")]
    threads: Option<usize>,
    /// Prefer lower CPU usage (equivalent to --threads 2 unless --threads is provided).
    #[arg(long = "low-power", action = ArgAction::SetTrue)]
    low_power: bool,
    /// Write an offscreen PNG snapshot and exit (no UI).
    #[arg(long = "dump-png", value_name = "FILE")]
    dump_png: Option<PathBuf>,
    /// Write a Bevy offscreen PNG (requires bevy_render feature) and exit (no UI).
    #[cfg(feature = "bevy_render")]
    #[arg(long = "dump-bevy-png", value_name = "FILE")]
    dump_bevy_png: Option<PathBuf>,
    /// Snapshot size for --dump-png, formatted as WIDTHxHEIGHT (e.g., 1280x720).
    #[arg(long = "png-size", value_name = "WxH")]
    png_size: Option<String>,
    /// Storage mode: duckdb (default) or memory (disable persistence).
    #[arg(long = "storage", value_enum, default_value_t = StorageMode::DuckDb)]
    storage: StorageMode,
    /// Profile headless `world.step()` without persistence for N ticks, then exit.
    #[arg(long = "profile-steps", value_name = "TICKS")]
    profile_steps: Option<u64>,
    /// Profile headless `world.step()` with selected storage mode for N ticks, then exit.
    #[arg(long = "profile-storage-steps", value_name = "TICKS")]
    profile_storage_steps: Option<u64>,
    /// Override storage flush thresholds: tick,agent,event,metric (e.g., 64,4096,1024,1024).
    #[arg(long = "storage-thresholds", value_name = "t,a,e,m")]
    storage_thresholds: Option<String>,
    /// Automated profiling sweep: runs multiple configurations for N ticks and summarizes.
    #[arg(long = "profile-sweep", value_name = "TICKS")]
    profile_sweep: Option<u64>,
    /// Auto-tune: quick sweep to pick threads/thresholds for current storage, then continue.
    #[arg(long = "auto-tune", value_name = "TICKS")]
    auto_tune: Option<u64>,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum StorageMode {
    DuckDb,
    Memory,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum ConfigFormat {
    Json,
    Toml,
    Ron,
}

fn apply_config_layers(base: ScriptBotsConfig, layers: &[PathBuf]) -> Result<ScriptBotsConfig> {
    if layers.is_empty() {
        return Ok(base);
    }

    let mut merged = serde_json::to_value(&base).expect("serialize base config");
    for path in layers {
        let layer_value = load_config_layer(path)?;
        info!(
            layer = %path.display(),
            "Applying configuration layer"
        );
        merge_layer(&mut merged, layer_value);
    }

    serde_json::from_value(merged)
        .map_err(|err| anyhow::anyhow!("failed to deserialize merged configuration: {err}"))
}

fn load_config_layer(path: &Path) -> Result<JsonValue> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration layer {}", path.display()))?;

    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("ron") => ron::from_str(&contents)
            .with_context(|| format!("failed to parse RON config layer {}", path.display())),
        _ => toml::from_str(&contents)
            .with_context(|| format!("failed to parse TOML config layer {}", path.display())),
    }
}

fn merge_layer(base: &mut JsonValue, layer: JsonValue) {
    match (base, layer) {
        (JsonValue::Object(base_map), JsonValue::Object(layer_map)) => {
            for (key, value) in layer_map {
                if let Some(existing) = base_map.get_mut(&key) {
                    merge_layer(existing, value);
                } else {
                    base_map.insert(key, value);
                }
            }
        }
        (target, value) => {
            *target = value;
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum RendererMode {
    Auto,
    Gui,
    Bevy,
    Terminal,
}

impl RendererMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gui => "gui",
            Self::Bevy => "bevy",
            Self::Terminal => "terminal",
        }
    }
}

impl fmt::Display for RendererMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

fn resolve_renderer(mode: RendererMode) -> Result<(RendererMode, Box<dyn Renderer>)> {
    match mode {
        RendererMode::Gui => {
            #[cfg(not(feature = "gui"))]
            {
                warn!("GUI feature not enabled; falling back to terminal renderer");
                return Ok((
                    RendererMode::Terminal,
                    Box::new(TerminalRenderer::default()),
                ));
            }
            #[cfg(feature = "gui")]
            {
                if should_use_terminal_mode() {
                    // Headless environment (no DISPLAY/WAYLAND) or forced terminal mode
                    println!(
                        "{} GUI unavailable or disabled. Falling back to terminal. Set {} to force GPUI, or run with {}",
                        "⚠".yellow().bold(),
                        "SCRIPTBOTS_FORCE_GUI=1".cyan(),
                        "--mode terminal".cyan()
                    );
                    return Ok((
                        RendererMode::Terminal,
                        Box::new(TerminalRenderer::default()),
                    ));
                }
                Ok((RendererMode::Gui, Box::new(GuiRenderer)))
            }
        }
        RendererMode::Bevy => {
            #[cfg(feature = "bevy_render")]
            {
                return Ok((RendererMode::Bevy, Box::new(BevyRenderer::default())));
            }
            #[cfg(not(feature = "bevy_render"))]
            {
                warn!(
                    "Bevy renderer requested, but binary was built without bevy_render. Falling back to terminal UI."
                );
                return Ok((
                    RendererMode::Terminal,
                    Box::new(TerminalRenderer::default()),
                ));
            }
        }
        RendererMode::Terminal => Ok((
            RendererMode::Terminal,
            Box::new(TerminalRenderer::default()),
        )),
        RendererMode::Auto => {
            if should_use_terminal_mode() {
                debug!("Auto-selected terminal renderer due to headless environment");
                Ok((
                    RendererMode::Terminal,
                    Box::new(TerminalRenderer::default()),
                ))
            } else {
                Ok((RendererMode::Gui, Box::new(GuiRenderer)))
            }
        }
    }
}

#[derive(Default)]
struct GuiRenderer;

impl Renderer for GuiRenderer {
    fn name(&self) -> &'static str {
        "gpui"
    }

    fn run(&self, ctx: RendererContext<'_>) -> Result<()> {
        #[cfg(feature = "gui")]
        {
            prepare_linux_gui_backend();
            run_demo(
                Arc::clone(&ctx.world),
                Some(Arc::clone(&ctx.storage)),
                Arc::clone(&ctx.command_drain),
                Arc::clone(&ctx.command_submit),
            );
            return Ok(());
        }
        #[cfg(not(feature = "gui"))]
        {
            // Avoid unused-parameter warning when GUI is not enabled
            let _ = ctx;
            bail!("GUI feature not enabled; recompile with --features gui or use --mode terminal");
        }
    }
}

#[cfg(feature = "bevy_render")]
#[derive(Default)]
struct BevyRenderer;

#[cfg(feature = "bevy_render")]
impl Renderer for BevyRenderer {
    fn name(&self) -> &'static str {
        "bevy"
    }

    fn run(&self, ctx: RendererContext<'_>) -> Result<()> {
        prepare_linux_gui_backend();
        let bevy_ctx = BevyRendererContext {
            world: Arc::clone(&ctx.world),
            command_submit: Arc::clone(&ctx.command_submit),
            command_drain: Arc::clone(&ctx.command_drain),
        };
        scriptbots_bevy::run_renderer(bevy_ctx)
    }
}

fn should_use_terminal_mode() -> bool {
    if let Ok(value) = env::var("SCRIPTBOTS_FORCE_TERMINAL")
        && let Some(flag) = parse_bool(&value)
    {
        return flag;
    }

    if let Ok(value) = env::var("SCRIPTBOTS_FORCE_GUI")
        && matches!(parse_bool(&value), Some(true))
    {
        return false;
    }

    #[cfg(target_family = "unix")]
    {
        let has_display =
            env::var_os("DISPLAY").is_some() || env::var_os("WAYLAND_DISPLAY").is_some();
        if !has_display {
            return true;
        }
    }

    false
}

#[cfg(target_os = "linux")]
fn prepare_linux_gui_backend() {
    use std::sync::Once;

    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if let Err(error) = maybe_force_x11_for_legacy_wayland() {
            tracing::debug!(%error, "Wayland backend probe failed; leaving backend selection unchanged");
        }
    });
}

#[cfg(not(target_os = "linux"))]
fn prepare_linux_gui_backend() {}

#[cfg(target_os = "linux")]
fn maybe_force_x11_for_legacy_wayland() -> Result<()> {
    use std::env;
    use wayland_client::{
        Connection, Dispatch, Proxy, QueueHandle,
        globals::{GlobalListContents, registry_queue_init},
        protocol::{wl_compositor, wl_registry, wl_surface},
    };

    struct RegistryProbe;

    impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for RegistryProbe {
        fn event(
            _state: &mut Self,
            _proxy: &wl_registry::WlRegistry,
            _event: wl_registry::Event,
            _data: &GlobalListContents,
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    // Respect explicit backend overrides or GUI forcing.
    if env::var_os("WINIT_UNIX_BACKEND").is_some() {
        return Ok(());
    }
    if matches!(
        env::var("SCRIPTBOTS_FORCE_GUI")
            .ok()
            .and_then(|value| parse_bool(&value)),
        Some(true)
    ) {
        return Ok(());
    }

    let Some(display) = env::var_os("WAYLAND_DISPLAY") else {
        return Ok(());
    };
    if display.is_empty() {
        return Ok(());
    }

    let connection = Connection::connect_to_env()?;
    let (globals, _queue) = registry_queue_init::<RegistryProbe>(&connection)?;
    let required = wl_surface::REQ_SET_BUFFER_SCALE_SINCE;

    let compositor_version = globals.contents().with_list(|globals| {
        globals
            .iter()
            .find(|global| global.interface == wl_compositor::WlCompositor::interface().name)
            .map(|global| global.version)
    });

    if let Some(version) = compositor_version {
        if version < required {
            // SAFETY: Modifying the process environment before spawning GUI worker threads.
            unsafe {
                env::set_var("WINIT_UNIX_BACKEND", "x11");
            }
            tracing::warn!(
                version,
                required,
                "Wayland compositor version too old (v{version}); forcing WINIT_UNIX_BACKEND=x11. Set SCRIPTBOTS_FORCE_GUI=1 to override."
            );
        }
    }

    Ok(())
}

fn run_replay_cli(cli: &AppCli, config: &ScriptBotsConfig) -> Result<()> {
    let db_path = cli
        .replay_db
        .as_ref()
        .expect("replay_db required to enter replay mode");
    let db_display = db_path.display().to_string();

    let mut storage = Storage::open(&db_display)
        .with_context(|| format!("failed to open replay database {db_display}"))?;
    let recorded_max_tick = storage.max_tick()?.unwrap_or(0);
    let persisted_events = storage.load_replay_events()?;
    let recorded_counts = storage.replay_event_counts()?;
    drop(storage);

    let events_max_tick = persisted_events.iter().map(|e| e.tick).max().unwrap_or(0);
    let tick_limit = cli
        .tick_limit
        .unwrap_or(recorded_max_tick.max(events_max_tick));

    if config.rng_seed.is_none() {
        warn!(
            "config_rng_seed" = false,
            "Replay config has no rng_seed; deterministic verification may fail"
        );
    }

    let replay_run = run_headless_simulation(config, tick_limit)?;
    let simulated_tick_count = replay_run.summaries.len() as u64;
    debug!(
        simulated_ticks = simulated_tick_count,
        simulated_events = replay_run.events.len(),
        "Headless replay complete"
    );
    if simulated_tick_count != tick_limit {
        warn!(
            requested_ticks = tick_limit,
            simulated_ticks = simulated_tick_count,
            "Simulated tick count differs from requested limit"
        );
    }
    let diff = diff_event_stream(&persisted_events, &replay_run.events);

    let recorded_map = recorded_counts
        .into_iter()
        .map(|entry| (entry.event_type, entry.count))
        .collect::<HashMap<_, _>>();
    let simulated_counts = count_event_kinds(&replay_run.events)
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect::<HashMap<_, _>>();

    // Deterministic key ordering for printed output
    let mut recorded_sorted: Vec<_> = recorded_map.iter().collect();
    recorded_sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut simulated_sorted: Vec<_> = simulated_counts.iter().collect();
    simulated_sorted.sort_by(|a, b| a.0.cmp(b.0));

    println!(
        "{} Replaying {} ticks ({} recorded events) against {}",
        "▶".bright_blue().bold(),
        tick_limit,
        persisted_events.len(),
        db_display.cyan()
    );
    print_event_counts("recorded", &recorded_map, None);
    print_event_counts("simulated", &simulated_counts, Some(&recorded_map));

    if let Some(divergence) = diff {
        report_divergence("recorded", "simulated", divergence)?;
    } else {
        println!(
            "{} Replay matched {} events across {} ticks",
            "✔".green().bold(),
            replay_run.events.len().green(),
            simulated_tick_count.green()
        );
    }

    if let Some(compare_path) = cli.compare_db.as_ref() {
        let compare_display = compare_path.display().to_string();
        let mut other = Storage::open(&compare_display)
            .with_context(|| format!("failed to open comparison database {compare_display}"))?;
        let other_events = other.load_replay_events()?;
        let other_counts = other.replay_event_counts()?;
        drop(other);

        println!(
            "{} Comparing {} against {}",
            "▶".bright_blue().bold(),
            db_display.cyan(),
            compare_display.cyan()
        );
        let compare_diff = diff_event_stream(&persisted_events, &other_events);
        let other_map = other_counts
            .into_iter()
            .map(|entry| (entry.event_type, entry.count))
            .collect::<HashMap<_, _>>();
        print_event_counts("baseline", &recorded_map, None);
        print_event_counts("candidate", &other_map, Some(&recorded_map));

        if let Some(divergence) = compare_diff {
            report_divergence("baseline", "candidate", divergence)?;
        } else {
            println!(
                "{} Event streams are identical ({} events)",
                "✔".green().bold(),
                other_events.len().green()
            );
        }
    }

    Ok(())
}

struct ReplayCollector {
    ticks: Arc<Mutex<Vec<ReplayTickRecord>>>,
}

impl ReplayCollector {
    fn with_capacity(capacity: usize) -> (Self, Arc<Mutex<Vec<ReplayTickRecord>>>) {
        let ticks = Arc::new(Mutex::new(Vec::with_capacity(capacity)));
        (
            Self {
                ticks: Arc::clone(&ticks),
            },
            ticks,
        )
    }

    fn new() -> (Self, Arc<Mutex<Vec<ReplayTickRecord>>>) {
        Self::with_capacity(0)
    }
}

impl WorldPersistence for ReplayCollector {
    fn on_tick(&mut self, payload: &scriptbots_core::PersistenceBatch) {
        if let Ok(mut guard) = self.ticks.lock() {
            guard.push(ReplayTickRecord {
                tick: payload.summary.tick.0,
                events: payload.replay_events.clone(),
                summary: payload.summary.clone(),
            });
        }
    }
}

struct ReplayTickRecord {
    tick: u64,
    events: Vec<scriptbots_core::ReplayEvent>,
    summary: TickSummary,
}

struct ReplayRun {
    events: Vec<PersistedReplayEvent>,
    summaries: Vec<TickSummary>,
}

fn run_headless_simulation(config: &ScriptBotsConfig, tick_limit: u64) -> Result<ReplayRun> {
    let (collector, handle) = ReplayCollector::with_capacity(tick_limit as usize);
    let mut world = WorldState::with_persistence(config.clone(), Box::new(collector))?;
    let brain_keys = install_brains(&mut world);
    seed_agents(&mut world, &brain_keys);

    for _ in 0..tick_limit {
        world.step();
    }

    drop(world);

    let records = Arc::try_unwrap(handle)
        .map_err(|_| anyhow::anyhow!("replay collector still in use"))?
        .into_inner()
        .map_err(|err| anyhow::anyhow!("replay collector poisoned: {err}"))?;

    let mut events = Vec::new();
    let mut summaries = Vec::with_capacity(records.len());
    for record in records {
        summaries.push(record.summary);
        for (seq, event) in record.events.into_iter().enumerate() {
            events.push(PersistedReplayEvent {
                tick: record.tick,
                seq: seq as u64,
                event,
            });
        }
    }

    Ok(ReplayRun { events, summaries })
}

fn profile_world_steps(config: &ScriptBotsConfig, tick_limit: u64) -> Result<()> {
    let (collector, _handle) = ReplayCollector::new();
    let mut world = WorldState::with_persistence(config.clone(), Box::new(collector))?;
    let brain_keys = install_brains(&mut world);
    seed_agents(&mut world, &brain_keys);

    let start = Instant::now();
    for _ in 0..tick_limit {
        world.step();
    }
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64().max(1e-9);
    let tps = tick_limit as f64 / secs;
    println!(
        "{} Headless no-storage: {} ticks in {:.3}s ({:.0} tps)",
        "✔".green().bold(),
        tick_limit,
        secs,
        tps
    );
    Ok(())
}

fn profile_world_steps_with_storage(
    config: &ScriptBotsConfig,
    tick_limit: u64,
    storage_mode: StorageMode,
    thresholds: ThresholdsOverride,
) -> Result<()> {
    let storage_path =
        env::var("SCRIPTBOTS_STORAGE_PATH").unwrap_or_else(|_| "scriptbots.db".to_string());
    let pipeline = match storage_mode {
        StorageMode::DuckDb => {
            let try_open = |path: &str| -> std::result::Result<StoragePipeline, StorageError> {
                match (
                    thresholds.tick,
                    thresholds.agent,
                    thresholds.event,
                    thresholds.metric,
                ) {
                    (Some(t), Some(a), Some(e), Some(m)) => {
                        StoragePipeline::with_thresholds(path, t, a, e, m)
                    }
                    _ => StoragePipeline::new(path),
                }
            };
            match try_open(&storage_path) {
                Ok(p) => p,
                Err(err) => {
                    let msg = err.to_string();
                    let lock_error = msg.to_ascii_lowercase().contains("could not set lock")
                        || msg.to_ascii_lowercase().contains("conflicting lock");
                    if lock_error {
                        let ts = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        let pid = std::process::id();
                        let fallback_path = {
                            let p = std::path::Path::new(&storage_path);
                            let parent = p.parent();
                            let (stem, ext) = (
                                p.file_stem()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("scriptbots"),
                                p.extension().and_then(|e| e.to_str()).unwrap_or("duckdb"),
                            );
                            let file = format!("{}.run-{}-{}.{}", stem, pid, ts, ext);
                            if let Some(dir) = parent {
                                dir.join(file)
                            } else {
                                std::path::PathBuf::from(file)
                            }
                        };
                        if let Some(parent) =
                            fallback_path.parent().filter(|d| !d.as_os_str().is_empty())
                        {
                            std::fs::create_dir_all(parent)?;
                        }
                        println!(
                            "{} {} is locked ({}). Falling back to {}",
                            "⚠".yellow().bold(),
                            storage_path.cyan(),
                            msg,
                            fallback_path.display().to_string().magenta().bold()
                        );
                        try_open(&fallback_path.to_string_lossy())?
                    } else {
                        return Err(err.into());
                    }
                }
            }
        }
        StorageMode::Memory => StoragePipeline::with_thresholds(
            ":memory:",
            thresholds.tick.unwrap_or(64),
            thresholds.agent.unwrap_or(2048),
            thresholds.event.unwrap_or(512),
            thresholds.metric.unwrap_or(512),
        )?,
    };

    let mut world = WorldState::with_persistence(config.clone(), Box::new(pipeline))?;
    let brain_keys = install_brains(&mut world);
    seed_agents(&mut world, &brain_keys);

    let start = Instant::now();
    for _ in 0..tick_limit {
        world.step();
    }
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64().max(1e-9);
    let tps = tick_limit as f64 / secs;
    println!(
        "{} Headless with-storage({}): {} ticks in {:.3}s ({:.0} tps)",
        "✔".green().bold(),
        match storage_mode {
            StorageMode::DuckDb => "duckdb",
            StorageMode::Memory => "memory",
        },
        tick_limit,
        secs,
        tps
    );
    Ok(())
}

fn parse_tps_from_stdout(stdout: &[u8]) -> Option<f64> {
    let s = std::str::from_utf8(stdout).ok()?;
    // Expect a line ending with "(NNN tps)"; grab the last number before " tps)"
    for line in s.lines().rev() {
        if let Some(idx) = line.rfind(" tps)") {
            let start = line[..idx].rfind('(')? + 1;
            let num_str = &line[start..idx];
            if let Ok(val) = num_str.trim().parse::<f64>() {
                return Some(val);
            }
        }
    }
    None
}

fn run_profile_sweep(_config: &ScriptBotsConfig, ticks: u64, cli: &AppCli) -> Result<()> {
    let exe = std::env::current_exe().context("failed to get current exe path")?;

    // Candidate configurations
    let thread_candidates: Vec<usize> = if let Some(t) = cli.threads {
        vec![t]
    } else {
        vec![1, 2, 4, 8]
    };
    let storage_candidates = [StorageMode::Memory, StorageMode::DuckDb];
    let threshold_candidates: Vec<&str> = vec![
        "64,2048,512,512",
        "128,4096,1024,1024",
        "256,4096,2048,1024",
    ];

    #[derive(Clone)]
    struct ResultRow {
        threads: usize,
        storage: StorageMode,
        thresholds: &'static str,
        tps: f64,
    }

    let mut results: Vec<ResultRow> = Vec::new();

    for threads in thread_candidates {
        for &storage in &storage_candidates {
            let threshold_list: Vec<&str> = match storage {
                StorageMode::Memory => threshold_candidates.clone(),
                StorageMode::DuckDb => threshold_candidates.clone(),
            };
            for thresholds in threshold_list {
                let mut cmd = Command::new(&exe);
                cmd.env("SCRIPTBOTS_DET_RUN", "0");
                cmd.env("RUST_LOG", "error");
                cmd.arg("--profile-storage-steps").arg(ticks.to_string());
                cmd.arg("--storage").arg(match storage {
                    StorageMode::DuckDb => "duckdb",
                    StorageMode::Memory => "memory",
                });
                cmd.arg("--storage-thresholds").arg(thresholds);
                cmd.arg("--threads").arg(threads.to_string());
                if cli.low_power {
                    cmd.arg("--low-power");
                }
                cmd.stdout(Stdio::piped());
                cmd.stderr(Stdio::null());
                let out = cmd.output().with_context(|| {
                    format!(
                        "sweep run failed (thr={threads}, storage={storage:?}, thres={thresholds})"
                    )
                })?;
                if !out.status.success() {
                    continue;
                }
                if let Some(tps) = parse_tps_from_stdout(&out.stdout) {
                    // thresholds is &'static str via const literals
                    let thresholds_static: &'static str = match thresholds {
                        "64,2048,512,512" => "64,2048,512,512",
                        "128,4096,1024,1024" => "128,4096,1024,1024",
                        _ => "256,4096,2048,1024",
                    };
                    results.push(ResultRow {
                        threads,
                        storage,
                        thresholds: thresholds_static,
                        tps,
                    });
                }
            }
        }
    }

    results.sort_by(|a, b| {
        b.tps
            .partial_cmp(&a.tps)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!(
        "{} Automated profile sweep ({} ticks):",
        "▶".bright_blue().bold(),
        ticks
    );
    for row in results.iter().take(8) {
        println!(
            "    threads={:<2} storage={:<6} thresholds={:<20} {:>8.0} tps",
            row.threads,
            match row.storage {
                StorageMode::DuckDb => "duckdb",
                StorageMode::Memory => "memory",
            },
            row.thresholds,
            row.tps
        );
    }

    if let Some(best) = results.first() {
        println!(
            "{} Best: threads={} storage={} thresholds={} ({:.0} tps)",
            "✔".green().bold(),
            best.threads,
            match best.storage {
                StorageMode::DuckDb => "duckdb",
                StorageMode::Memory => "memory",
            },
            best.thresholds,
            best.tps
        );
    } else {
        println!("{} No successful sweep results", "✖".red().bold());
    }
    Ok(())
}

struct BestPick {
    threads: usize,
    tick: usize,
    agent: usize,
    event: usize,
    metric: usize,
    tps: f64,
}

fn pick_best_for_storage(
    _config: &ScriptBotsConfig,
    ticks: u64,
    storage: StorageMode,
    pinned_threads: Option<usize>,
    low_power: bool,
) -> Result<Option<BestPick>> {
    let exe = std::env::current_exe().context("failed to get current exe path")?;

    let thread_candidates: Vec<usize> = if let Some(t) = pinned_threads {
        vec![t]
    } else {
        vec![1, 2, 4, 8]
    };
    let threshold_candidates: Vec<&str> = vec![
        "64,2048,512,512",
        "128,4096,1024,1024",
        "256,4096,2048,1024",
    ];
    let mut best: Option<BestPick> = None;

    for threads in thread_candidates {
        for thresholds in &threshold_candidates {
            let mut cmd = Command::new(&exe);
            cmd.env("SCRIPTBOTS_DET_RUN", "0");
            cmd.env("RUST_LOG", "error");
            cmd.arg("--profile-storage-steps").arg(ticks.to_string());
            cmd.arg("--storage").arg(match storage {
                StorageMode::DuckDb => "duckdb",
                StorageMode::Memory => "memory",
            });
            cmd.arg("--storage-thresholds").arg(thresholds);
            cmd.arg("--threads").arg(threads.to_string());
            if low_power {
                cmd.arg("--low-power");
            }
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());
            let out = cmd.output()?;
            if !out.status.success() {
                continue;
            }
            if let Some(tps) = parse_tps_from_stdout(&out.stdout) {
                let parts: Vec<_> = thresholds.split(',').collect();
                if parts.len() == 4 {
                    if let (Ok(tk), Ok(ag), Ok(ev), Ok(me)) = (
                        parts[0].parse(),
                        parts[1].parse(),
                        parts[2].parse(),
                        parts[3].parse(),
                    ) {
                        let candidate = BestPick {
                            threads,
                            tick: tk,
                            agent: ag,
                            event: ev,
                            metric: me,
                            tps,
                        };
                        match &best {
                            Some(b) if b.tps >= candidate.tps => {}
                            _ => {
                                best = Some(candidate);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(best)
}

#[derive(Debug)]
struct Divergence {
    kind: DivergenceKind,
    expected: Option<PersistedReplayEvent>,
    actual: Option<PersistedReplayEvent>,
}

#[derive(Debug)]
enum DivergenceKind {
    TickMismatch,
    SequenceMismatch,
    EventMismatch,
    MissingActual,
    ExtraActual,
}

fn diff_event_stream(
    expected: &[PersistedReplayEvent],
    actual: &[PersistedReplayEvent],
) -> Option<Divergence> {
    let mut idx = 0;
    loop {
        match (expected.get(idx), actual.get(idx)) {
            (Some(left), Some(right)) => {
                if left.tick != right.tick {
                    return Some(Divergence {
                        kind: DivergenceKind::TickMismatch,
                        expected: Some(left.clone()),
                        actual: Some(right.clone()),
                    });
                }
                if left.seq != right.seq {
                    return Some(Divergence {
                        kind: DivergenceKind::SequenceMismatch,
                        expected: Some(left.clone()),
                        actual: Some(right.clone()),
                    });
                }
                if left.event != right.event {
                    return Some(Divergence {
                        kind: DivergenceKind::EventMismatch,
                        expected: Some(left.clone()),
                        actual: Some(right.clone()),
                    });
                }
            }
            (Some(left), None) => {
                return Some(Divergence {
                    kind: DivergenceKind::MissingActual,
                    expected: Some(left.clone()),
                    actual: None,
                });
            }
            (None, Some(right)) => {
                return Some(Divergence {
                    kind: DivergenceKind::ExtraActual,
                    expected: None,
                    actual: Some(right.clone()),
                });
            }
            (None, None) => return None,
        }
        idx += 1;
    }
}

fn count_event_kinds(events: &[PersistedReplayEvent]) -> HashMap<&'static str, u64> {
    let mut counts = HashMap::new();
    for entry in events {
        let key = match entry.event.kind {
            ReplayEventKind::BrainOutputs { .. } => "brain_outputs",
            ReplayEventKind::Action { .. } => "action",
            ReplayEventKind::RngSample { .. } => "rng_sample",
        };
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn report_divergence(left_label: &str, right_label: &str, divergence: Divergence) -> Result<()> {
    let marker = format!("{}", "✖".red().bold());
    match divergence.kind {
        DivergenceKind::TickMismatch => {
            if let (Some(exp), Some(act)) = (&divergence.expected, &divergence.actual) {
                println!(
                    "{marker} Tick mismatch: {left_label} tick {} vs {right_label} tick {}",
                    exp.tick.red(),
                    act.tick.red()
                );
            }
        }
        DivergenceKind::SequenceMismatch => {
            if let (Some(exp), Some(act)) = (&divergence.expected, &divergence.actual) {
                println!(
                    "{marker} Sequence mismatch at tick {}: {left_label} seq {} vs {right_label} seq {}",
                    exp.tick.red(),
                    exp.seq.red(),
                    act.seq.red()
                );
            }
        }
        DivergenceKind::EventMismatch => {
            if let (Some(exp), Some(act)) = (&divergence.expected, &divergence.actual) {
                println!(
                    "{marker} Event mismatch at tick {} seq {}",
                    exp.tick.red(),
                    exp.seq.red()
                );
                println!("    expected: {}", format_replay_event(&exp.event).yellow());
                println!("    actual:   {}", format_replay_event(&act.event).yellow());
            }
        }
        DivergenceKind::MissingActual => {
            if let Some(exp) = divergence.expected {
                println!(
                    "{marker} {right_label} stream ended before event tick {} seq {}",
                    exp.tick.red(),
                    exp.seq.red()
                );
            }
        }
        DivergenceKind::ExtraActual => {
            if let Some(act) = divergence.actual {
                println!(
                    "{marker} {right_label} has extra event at tick {} seq {}",
                    act.tick.red(),
                    act.seq.red()
                );
            }
        }
    }

    bail!("replay divergence detected")
}

fn format_replay_event(event: &scriptbots_core::ReplayEvent) -> String {
    match &event.kind {
        ReplayEventKind::BrainOutputs { outputs } => format!(
            "BrainOutputs(agent={:?}, len={})",
            event.agent_id,
            outputs.len()
        ),
        ReplayEventKind::Action {
            left_wheel,
            right_wheel,
            boost,
            spike_target,
            sound_level,
            give_intent,
        } => format!(
            "Action(agent={:?}, lw={:.3}, rw={:.3}, boost={}, spike={:?}, sound={:.3}, give={:.3})",
            event.agent_id, left_wheel, right_wheel, boost, spike_target, sound_level, give_intent
        ),
        ReplayEventKind::RngSample {
            scope,
            range_min,
            range_max,
            value,
        } => format!(
            "RngSample(scope={:?}, min={:.3}, max={:.3}, value={:.3})",
            scope, range_min, range_max, value
        ),
    }
}

fn print_event_counts(
    label: &str,
    counts: &HashMap<String, u64>,
    reference: Option<&HashMap<String, u64>>,
) {
    let keys = ["brain_outputs", "action", "rng_sample"];
    println!("  {}", label.cyan().bold());
    for key in keys {
        let value = counts.get(key).copied().unwrap_or(0);
        if let Some(baseline) = reference {
            let baseline_value = baseline.get(key).copied().unwrap_or(0);
            let delta = value as i64 - baseline_value as i64;
            let delta_fmt = format!("Δ {delta:+}");
            let delta_colored = if delta == 0 {
                format!("{}", delta_fmt.yellow())
            } else if delta > 0 {
                format!("{}", delta_fmt.green())
            } else {
                format!("{}", delta_fmt.red())
            };
            println!("    {:<14} {:>8} ({delta_colored})", key, value);
        } else {
            println!("    {:<14} {:>8}", key, value);
        }
    }
}

fn install_brains(world: &mut WorldState) -> Vec<u64> {
    let mut keys = Vec::new();

    let mlp_key = world
        .brain_registry_mut()
        .register(MlpBrain::KIND.as_str(), |seed_rng| {
            MlpBrain::runner(seed_rng)
        });
    keys.push(mlp_key);

    #[cfg(feature = "ml")]
    {
        let label = {
            let prototype = scriptbots_brain_ml::runner();
            prototype.kind().to_string()
        };
        let key = world
            .brain_registry_mut()
            .register(label, |_seed_rng| scriptbots_brain_ml::runner());
        keys.push(key);
    }

    #[cfg(feature = "neuro")]
    {
        use scriptbots_brain_neuro::{NeuroflowBrain, NeuroflowBrainConfig};
        let settings = world.config().neuroflow.clone();
        if settings.enabled {
            let config = NeuroflowBrainConfig::from_settings(&settings);
            let key = NeuroflowBrain::register(world, config);
            keys.push(key);
        }
    }

    keys
}

fn apply_env_overrides(config: &mut ScriptBotsConfig) {
    if let Ok(value) = env::var("SCRIPTBOTS_NEUROFLOW_ENABLED") {
        match parse_bool(&value) {
            Some(flag) => config.neuroflow.enabled = flag,
            None => {
                warn!(value = %value, "Invalid SCRIPTBOTS_NEUROFLOW_ENABLED value; expected true/false")
            }
        }
    }

    if let Ok(value) = env::var("SCRIPTBOTS_NEUROFLOW_HIDDEN") {
        match parse_layers(&value) {
            Some(layers) => config.neuroflow.hidden_layers = layers,
            None => {
                warn!(value = %value, "Invalid SCRIPTBOTS_NEUROFLOW_HIDDEN value; expected comma-separated integers")
            }
        }
    }

    if let Ok(value) = env::var("SCRIPTBOTS_NEUROFLOW_ACTIVATION") {
        match parse_activation(&value) {
            Some(activation) => config.neuroflow.activation = activation,
            None => {
                warn!(value = %value, "Invalid SCRIPTBOTS_NEUROFLOW_ACTIVATION value; expected tanh|sigmoid|relu")
            }
        }
    }
}

fn parse_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_layers(raw: &str) -> Option<Vec<usize>> {
    let mut layers = Vec::new();
    for token in raw.split(',') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed.parse::<usize>() {
            Ok(value) if value > 0 => layers.push(value),
            _ => return None,
        }
    }
    Some(layers)
}

fn parse_activation(raw: &str) -> Option<NeuroflowActivationKind> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "tanh" => Some(NeuroflowActivationKind::Tanh),
        "sigmoid" => Some(NeuroflowActivationKind::Sigmoid),
        "relu" => Some(NeuroflowActivationKind::Relu),
        _ => None,
    }
}

#[cfg(any(feature = "gui", feature = "bevy_render"))]
fn parse_png_size(raw: &str) -> Option<(u32, u32)> {
    let lower = raw.trim().to_ascii_lowercase();
    let mut parts = lower.split('x');
    let w = parts.next()?.trim().parse::<u32>().ok()?;
    let h = parts.next()?.trim().parse::<u32>().ok()?;
    if parts.next().is_some() || w == 0 || h == 0 {
        return None;
    }
    Some((w, h))
}

fn seed_agents(world: &mut WorldState, brain_keys: &[u64]) {
    let mut agent = AgentData::default();
    let spacing = 120.0;
    for row in 0..4 {
        for col in 0..4 {
            agent.position.x = col as f32 * spacing + spacing * 0.5;
            agent.position.y = row as f32 * spacing + spacing * 0.5;
            agent.heading = 0.0;
            agent.spike_length = 10.0;
            let id = world.spawn_agent(agent);
            if let Some(&key) = brain_keys.get((row * 4 + col) % brain_keys.len())
                && !world.bind_agent_brain(id, key)
            {
                warn!(agent = ?id, key, "Failed to bind brain to seeded agent");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_storage::{Storage, StoragePipeline};
    use serial_test::serial;
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    fn default_cli() -> AppCli {
        AppCli::parse_from(["scriptbots-app"])
    }

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn with_env_lock<F: FnOnce()>(f: F) {
        let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock.lock().expect("env mutex poisoned");
        f();
    }

    #[test]
    #[serial]
    fn layered_configs_apply_in_order() {
        let dir = tempdir().expect("tempdir");
        let base_path = dir.path().join("base.toml");
        fs::write(
            &base_path,
            r#"
persistence_interval = 120
rng_seed = 1337

[neuroflow]
enabled = true
hidden_layers = [64, 32]
activation = "Tanh"
"#,
        )
        .expect("write base layer");

        let overlay_path = dir.path().join("overlay.toml");
        fs::write(
            &overlay_path,
            r#"
history_capacity = 1024
world_width = 2048

[neuroflow]
hidden_layers = [8, 4]
activation = "Sigmoid"
"#,
        )
        .expect("write overlay layer");

        let base_config = ScriptBotsConfig {
            persistence_interval: 60,
            history_capacity: 600,
            ..ScriptBotsConfig::default()
        };

        let layered = apply_config_layers(base_config, &[base_path, overlay_path])
            .expect("apply config layers");

        assert_eq!(layered.persistence_interval, 120);
        assert_eq!(layered.history_capacity, 1024);
        assert_eq!(layered.world_width, 2048);
        assert_eq!(layered.rng_seed, Some(1337));
        assert!(layered.neuroflow.enabled);
        assert_eq!(layered.neuroflow.hidden_layers, vec![8, 4]);
        assert_eq!(
            layered.neuroflow.activation,
            NeuroflowActivationKind::Sigmoid
        );
    }

    #[test]
    #[serial]
    fn headless_replay_matches_storage() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("replay.duckdb");
        let db_str = db_path.to_string_lossy().to_string();

        let config = ScriptBotsConfig {
            world_width: 600,
            world_height: 600,
            food_cell_size: 60,
            persistence_interval: 1,
            history_capacity: 128,
            rng_seed: Some(0xA1B2C3D4),
            ..ScriptBotsConfig::default()
        };

        {
            let pipeline = StoragePipeline::with_thresholds(&db_str, 1, 1, 1, 1).expect("pipeline");
            let mut world =
                WorldState::with_persistence(config.clone(), Box::new(pipeline)).expect("world");
            let keys = install_brains(&mut world);
            seed_agents(&mut world, &keys);
            for _ in 0..16 {
                world.step();
            }
        }

        let mut storage = Storage::open(&db_str).expect("open storage");
        let recorded_events = storage.load_replay_events().expect("load events");
        let max_tick = storage.max_tick().expect("max tick").unwrap_or(0);
        drop(storage);

        let replay = run_headless_simulation(&config, max_tick).expect("replay run");
        let diff = diff_event_stream(&recorded_events, &replay.events);
        assert!(diff.is_none(), "expected replay to match persisted events");
    }

    #[test]
    #[serial]
    fn write_config_honors_format_and_exit_flag() {
        let dir = tempdir().expect("tempdir");
        let output = dir.path().join("effective.toml");
        let mut cli = default_cli();
        cli.write_config = Some(output.clone());
        cli.config_format = ConfigFormat::Toml;
        cli.config_only = true;

        let config = ScriptBotsConfig {
            world_width: 1234,
            world_height: 5678,
            rng_seed: Some(42),
            ..ScriptBotsConfig::default()
        };

        let outcome = maybe_emit_config(&cli, &config)
            .expect("emit config")
            .expect("expected emit outcome");
        assert_eq!(outcome, ConfigEmitOutcome::Exit);

        let written = fs::read_to_string(&output).expect("read output");
        assert!(written.contains("world_width = 1234"));
        assert!(written.contains("rng_seed = 42"));
    }

    #[test]
    #[serial]
    fn rng_seed_cli_override_applies() {
        let mut cli = default_cli();
        cli.rng_seed = Some(2025);
        let config = compose_config(&cli).expect("compose config");
        assert_eq!(config.rng_seed, Some(2025));
    }

    #[test]
    #[serial]
    fn emit_config_continue_when_not_config_only() {
        let dir = tempdir().expect("tempdir");
        let output = dir.path().join("effective.ron");
        let mut cli = default_cli();
        cli.write_config = Some(output.clone());
        cli.config_format = ConfigFormat::Ron;
        cli.config_only = false;

        let config = ScriptBotsConfig::default();

        let outcome = maybe_emit_config(&cli, &config)
            .expect("emit config")
            .expect("expected emit outcome");
        assert_eq!(outcome, ConfigEmitOutcome::Continue);

        let written = fs::read_to_string(&output).expect("read output");
        assert!(written.contains("world_width"));
    }

    fn restore_env(var: &str, previous: Option<String>) {
        unsafe {
            if let Some(value) = previous {
                std::env::set_var(var, value);
            } else {
                std::env::remove_var(var);
            }
        }
    }

    #[test]
    fn env_overrides_apply_expected_settings() {
        with_env_lock(|| {
            let prev_enabled = std::env::var("SCRIPTBOTS_NEUROFLOW_ENABLED").ok();
            let prev_hidden = std::env::var("SCRIPTBOTS_NEUROFLOW_HIDDEN").ok();
            let prev_activation = std::env::var("SCRIPTBOTS_NEUROFLOW_ACTIVATION").ok();

            unsafe {
                std::env::set_var("SCRIPTBOTS_NEUROFLOW_ENABLED", "true");
                std::env::set_var("SCRIPTBOTS_NEUROFLOW_HIDDEN", "64, 32 ,16");
                std::env::set_var("SCRIPTBOTS_NEUROFLOW_ACTIVATION", "relu");
            }

            let mut config = ScriptBotsConfig::default();
            apply_env_overrides(&mut config);

            assert!(config.neuroflow.enabled);
            assert_eq!(config.neuroflow.hidden_layers, vec![64, 32, 16]);
            assert_eq!(config.neuroflow.activation, NeuroflowActivationKind::Relu);

            restore_env("SCRIPTBOTS_NEUROFLOW_ENABLED", prev_enabled);
            restore_env("SCRIPTBOTS_NEUROFLOW_HIDDEN", prev_hidden);
            restore_env("SCRIPTBOTS_NEUROFLOW_ACTIVATION", prev_activation);
        });
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn neuroflow_installation_respects_toggle() {
        let expected_base = if cfg!(feature = "ml") { 2 } else { 1 };
        let mut config = ScriptBotsConfig::default();
        config.neuroflow.enabled = false;
        let mut world = WorldState::new(config).expect("world");
        let keys = install_brains(&mut world);
        assert_eq!(
            keys.len(),
            expected_base,
            "NeuroFlow brain should not register when disabled"
        );

        let mut config_enabled = ScriptBotsConfig::default();
        config_enabled.neuroflow.enabled = true;
        config_enabled.neuroflow.hidden_layers = vec![12, 6];
        config_enabled.neuroflow.activation = NeuroflowActivationKind::Sigmoid;
        config_enabled.rng_seed = Some(99);
        let mut world_enabled = WorldState::new(config_enabled.clone()).expect("world");
        let keys_enabled = install_brains(&mut world_enabled);
        assert_eq!(
            keys_enabled.len(),
            expected_base + 1,
            "Expected baseline brains plus NeuroFlow"
        );

        let neuro_key = *keys_enabled.last().expect("neuro key");
        let agent_id = world_enabled.spawn_agent(AgentData::default());
        assert!(world_enabled.bind_agent_brain(agent_id, neuro_key));
        world_enabled.step();
        let outputs_one = world_enabled.agent_runtime(agent_id).unwrap().outputs;

        let mut world_repeat = WorldState::new(config_enabled).expect("world");
        let keys_repeat = install_brains(&mut world_repeat);
        assert_eq!(keys_repeat.len(), expected_base + 1);
        let neuro_repeat = *keys_repeat.last().unwrap();
        let agent_repeat = world_repeat.spawn_agent(AgentData::default());
        assert!(world_repeat.bind_agent_brain(agent_repeat, neuro_repeat));
        world_repeat.step();
        let outputs_two = world_repeat.agent_runtime(agent_repeat).unwrap().outputs;

        assert_eq!(
            outputs_one, outputs_two,
            "NeuroFlow outputs should be deterministic for same seed"
        );
    }
}
