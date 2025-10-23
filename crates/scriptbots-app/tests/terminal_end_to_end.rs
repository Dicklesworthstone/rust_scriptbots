use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
use duckdb::Connection;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use scriptbots_app::{
    ControlCommand, ControlRuntime, ControlServerConfig, McpTransportConfig,
    renderer::{Renderer, RendererContext},
    terminal::TerminalRenderer,
};
use scriptbots_core::{AgentData, Generation, Position, ScriptBotsConfig, Velocity, WorldState};
use scriptbots_storage::{Storage, StoragePipeline};
use serde::Deserialize;
use tempfile::tempdir;
use tracing::Level;

static ENV_GUARD: OnceLock<Mutex<()>> = OnceLock::new();

struct EnvCleanup {
    keys: Vec<String>,
}

impl EnvCleanup {
    fn new() -> Self {
        Self { keys: Vec::new() }
    }

    fn set(&mut self, key: &str, value: &str) {
        unsafe {
            std::env::set_var(key, value);
        }
        self.keys.push(key.to_string());
    }
}

impl Drop for EnvCleanup {
    fn drop(&mut self) {
        for key in &self.keys {
            unsafe {
                std::env::remove_var(key);
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct FrameStatsDto {
    tick: u64,
    epoch: u64,
    agent_count: usize,
    births: usize,
    deaths: usize,
    avg_energy: f32,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ReportSummaryDto {
    frame_count: usize,
    ticks_simulated: u64,
    final_tick: u64,
    final_epoch: u64,
    final_agent_count: usize,
    total_births: usize,
    total_deaths: usize,
    avg_energy_mean: f32,
    avg_energy_min: f32,
    avg_energy_max: f32,
}

#[derive(Debug, Deserialize)]
struct HeadlessReportDto {
    initial: FrameStatsDto,
    frames: Vec<FrameStatsDto>,
    summary: ReportSummaryDto,
}

#[test]
fn terminal_headless_generates_report() -> Result<()> {
    let _env_guard = ENV_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env guard");

    let _ = tracing_subscriber::fmt()
        .with_env_filter("scriptbots_app=info,scriptbots_core=warn")
        .with_max_level(Level::INFO)
        .with_test_writer()
        .try_init();

    let frames = 160usize;

    let report_dir = tempdir()?;
    let report_path = report_dir.path().join("terminal_report.json");

    let storage_dir = tempdir()?;
    let storage_path = storage_dir.path().join("scriptbots_test.duckdb");

    let mut env = EnvCleanup::new();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS", "1");
    let frames_env = frames.to_string();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", &frames_env);
    let report_env = report_path.to_string_lossy().into_owned();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS_REPORT", &report_env);

    let mut config = ScriptBotsConfig {
        food_cell_size: 32,
        world_width: 160,
        world_height: 96,
        population_minimum: 0,
        population_spawn_interval: 0,
        history_capacity: 512,
        rng_seed: Some(0xDEC0_DEAD),
        initial_food: 0.35,
        food_respawn_interval: 6,
        food_respawn_amount: 0.45,
        food_max: 1.0,
        food_growth_rate: 0.18,
        food_decay_rate: 0.0008,
        food_diffusion_rate: 0.18,
        reproduction_cooldown: 12,
        reproduction_rate_herbivore: 160.0,
        reproduction_rate_carnivore: 160.0,
        reproduction_energy_cost: 0.12,
        reproduction_child_energy: 0.9,
        reproduction_spawn_jitter: 12.0,
        reproduction_spawn_back_distance: 6.0,
        reproduction_partner_chance: 0.4,
        metabolism_drain: 0.006,
        movement_drain: 0.012,
        temperature_discomfort_rate: 0.0015,
        food_intake_rate: 0.008,
        food_waste_rate: 0.0005,
        ..ScriptBotsConfig::default()
    };
    config.food_sharing_rate = 0.15;
    config.food_transfer_rate = 0.0025;

    let mut world = WorldState::new(config.clone())?;
    let mut rng = SmallRng::seed_from_u64(0xBAD5_EED5);
    for _ in 0..32 {
        let position = Position::new(
            rng.random_range(0.0..config.world_width as f32),
            rng.random_range(0.0..config.world_height as f32),
        );
        let heading = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
        let color = [
            rng.random_range(0.0..1.0),
            rng.random_range(0.0..1.0),
            rng.random_range(0.0..1.0),
        ];
        let agent = AgentData::new(
            position,
            Velocity::default(),
            heading,
            1.0,
            color,
            0.1,
            false,
            0,
            Generation::default(),
        );
        world.spawn_agent(agent);
    }
    let shared_world = Arc::new(Mutex::new(world));

    let storage = Storage::open(
        storage_path
            .to_str()
            .expect("temporary storage path should be utf-8"),
    )?;
    let shared_storage = Arc::new(Mutex::new(storage));

    let control_config = ControlServerConfig {
        rest_enabled: false,
        mcp_transport: McpTransportConfig::Disabled,
        ..ControlServerConfig::default()
    };

    let (control_runtime, command_drain, command_submit) =
        ControlRuntime::launch(Arc::clone(&shared_world), control_config)?;

    let renderer = TerminalRenderer::default();
    {
        let context = RendererContext {
            world: Arc::clone(&shared_world),
            storage: Arc::clone(&shared_storage),
            control_runtime: &control_runtime,
            command_drain,
            command_submit,
        };
        renderer.run(context)?;
    }

    control_runtime.shutdown()?;

    let report_contents = std::fs::read_to_string(&report_path)?;
    let report: HeadlessReportDto = serde_json::from_str(&report_contents)?;
    let summary = &report.summary;

    assert_eq!(
        summary.frame_count, frames,
        "headless renderer should honour requested frame budget"
    );
    assert_eq!(
        summary.ticks_simulated,
        summary.final_tick.saturating_sub(report.initial.tick),
        "tick delta should align with simulated frames"
    );
    assert!(
        summary.final_agent_count > 0,
        "simulation should retain surviving agents"
    );
    assert!(
        summary.avg_energy_max > summary.avg_energy_min,
        "energy extrema should be well ordered and non-degenerate"
    );

    let total_births: usize = report.frames.iter().map(|frame| frame.births).sum();
    let total_deaths: usize = report.frames.iter().map(|frame| frame.deaths).sum();
    assert_eq!(
        total_births, summary.total_births,
        "summary births should match frame-wise birth totals"
    );
    assert_eq!(
        total_deaths, summary.total_deaths,
        "summary deaths should match frame-wise death totals"
    );
    assert!(
        summary.total_births > 8,
        "simulation should exhibit meaningful reproduction activity (births={})",
        summary.total_births
    );
    assert!(
        summary.total_deaths > 0,
        "simulation should register at least one death"
    );

    let frames_with_births = report
        .frames
        .iter()
        .filter(|frame| frame.births > 0)
        .count();
    assert!(
        frames_with_births >= 6,
        "births should occur across multiple frames (frames_with_births={frames_with_births})"
    );

    let frames_with_deaths = report
        .frames
        .iter()
        .filter(|frame| frame.deaths > 0)
        .count();
    assert!(
        frames_with_deaths >= 3,
        "deaths should appear in several frames (frames_with_deaths={frames_with_deaths})"
    );

    let agent_counts: Vec<usize> = report
        .frames
        .iter()
        .map(|frame| frame.agent_count)
        .collect();
    let min_agents = *agent_counts.iter().min().expect("min agent count");
    let max_agents = *agent_counts.iter().max().expect("max agent count");
    assert!(
        max_agents > min_agents,
        "agent count should vary over the run (min={min_agents}, max={max_agents})"
    );

    assert_eq!(
        report.initial.agent_count + summary.total_births - summary.total_deaths,
        summary.final_agent_count,
        "agent conservation should hold (initial + births - deaths = final)"
    );

    {
        let guard = shared_world.lock().expect("world mutex");
        assert_eq!(
            guard.tick().0,
            summary.final_tick,
            "world tick should advance to the reported final tick"
        );
        let history: Vec<_> = guard.history().cloned().collect();
        assert!(
            history.len() >= frames,
            "world history should retain per-tick summaries (len={})",
            history.len()
        );
        assert!(
            history.iter().any(|entry| entry.births > 0),
            "world history should record at least one birth"
        );
        assert!(
            history.iter().any(|entry| entry.deaths > 0),
            "world history should record at least one death"
        );
    }

    Ok(())
}

#[test]
fn terminal_headless_applies_control_updates() -> Result<()> {
    let _env_guard = ENV_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env guard");

    let _ = tracing_subscriber::fmt()
        .with_env_filter("scriptbots_app=info,scriptbots_core=warn")
        .with_max_level(Level::INFO)
        .with_test_writer()
        .try_init();

    let frames = 180usize;

    let report_dir = tempdir()?;
    let report_path = report_dir.path().join("terminal_control_report.json");

    let storage_dir = tempdir()?;
    let storage_path = storage_dir.path().join("terminal_control.duckdb");

    let mut env = EnvCleanup::new();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS", "1");
    let frames_env = frames.to_string();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", &frames_env);
    let report_env = report_path.to_string_lossy().into_owned();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS_REPORT", &report_env);
    env.set("RUST_LOG", "info");
    env.set("RUST_LOG_STYLE", "never");

    let mut config = ScriptBotsConfig {
        world_width: 200,
        world_height: 140,
        food_cell_size: 20,
        population_minimum: 0,
        population_spawn_interval: 0,
        persistence_interval: 1,
        history_capacity: 640,
        rng_seed: Some(0x51EED5),
        initial_food: 0.3,
        food_max: 1.0,
        food_respawn_interval: 8,
        food_respawn_amount: 0.4,
        food_growth_rate: 0.16,
        food_decay_rate: 0.0008,
        food_diffusion_rate: 0.16,
        reproduction_cooldown: 14,
        reproduction_rate_herbivore: 140.0,
        reproduction_rate_carnivore: 140.0,
        reproduction_energy_cost: 0.1,
        reproduction_child_energy: 0.88,
        reproduction_spawn_jitter: 10.0,
        reproduction_spawn_back_distance: 5.0,
        reproduction_partner_chance: 0.35,
        metabolism_drain: 0.007,
        movement_drain: 0.014,
        temperature_discomfort_rate: 0.001,
        food_intake_rate: 0.009,
        food_waste_rate: 0.0006,
        chart_flush_interval: 240,
        ..ScriptBotsConfig::default()
    };
    config.analytics_stride.behavior_metrics = 24;
    config.analytics_stride.lifecycle_events = 12;
    config.food_sharing_rate = 0.16;
    config.food_transfer_rate = 0.002;

    let mut world = WorldState::new(config.clone())?;
    let pipeline = StoragePipeline::with_thresholds(
        storage_path
            .to_str()
            .expect("temporary storage path should be utf-8"),
        1,
        1,
        1,
        1,
    )?;
    let shared_storage = pipeline.storage();
    world.set_persistence(Box::new(pipeline));

    let mut rng = SmallRng::seed_from_u64(0xDECAF00D);
    for _ in 0..48 {
        let position = Position::new(
            rng.random_range(0.0..config.world_width as f32),
            rng.random_range(0.0..config.world_height as f32),
        );
        let heading = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
        let color = [
            rng.random_range(0.0..1.0),
            rng.random_range(0.0..1.0),
            rng.random_range(0.0..1.0),
        ];
        let agent = AgentData::new(
            position,
            Velocity::default(),
            heading,
            1.0,
            color,
            0.05,
            false,
            0,
            Generation::default(),
        );
        world.spawn_agent(agent);
    }
    let shared_world = Arc::new(Mutex::new(world));

    let control_config = ControlServerConfig {
        rest_enabled: false,
        mcp_transport: McpTransportConfig::Disabled,
        ..ControlServerConfig::default()
    };

    let (control_runtime, command_drain, command_submit) =
        ControlRuntime::launch(Arc::clone(&shared_world), control_config)?;

    let mut updated_config = config.clone();
    updated_config.food_growth_rate = 0.24;
    updated_config.food_decay_rate = 0.0004;
    updated_config.food_respawn_amount = 0.5;
    updated_config.metabolism_drain = 0.004;
    updated_config.reproduction_cooldown = 8;
    updated_config.reproduction_rate_herbivore = 220.0;
    updated_config.reproduction_rate_carnivore = 220.0;
    updated_config.reproduction_energy_cost = 0.08;
    updated_config.chart_flush_interval = 90;
    let submit_ok = command_submit(ControlCommand::UpdateConfig(updated_config.clone()));
    assert!(submit_ok, "control queue rejected config update");

    let renderer = TerminalRenderer::default();
    {
        let context = RendererContext {
            world: Arc::clone(&shared_world),
            storage: Arc::clone(&shared_storage),
            control_runtime: &control_runtime,
            command_drain,
            command_submit,
        };
        renderer.run(context)?;
    }
    control_runtime.shutdown()?;

    let report_contents = std::fs::read_to_string(&report_path)?;
    let report: HeadlessReportDto = serde_json::from_str(&report_contents)?;
    let summary = &report.summary;

    assert_eq!(summary.frame_count, frames);
    assert_eq!(
        summary.final_tick,
        report.initial.tick + frames as u64,
        "final tick should equal initial tick plus simulated frames"
    );
    assert!(
        summary.total_births > 20,
        "integration run should yield substantial reproduction (births={})",
        summary.total_births
    );
    assert!(
        summary.total_deaths > 6,
        "integration run should produce observable mortality (deaths={})",
        summary.total_deaths
    );
    assert!(
        summary.final_agent_count > report.initial.agent_count,
        "population should grow over the run (initial={}, final={})",
        report.initial.agent_count,
        summary.final_agent_count
    );
    assert!(
        summary.avg_energy_mean > report.initial.avg_energy,
        "mean energy should increase once control updates take effect"
    );

    let total_births: usize = report.frames.iter().map(|frame| frame.births).sum();
    let total_deaths: usize = report.frames.iter().map(|frame| frame.deaths).sum();
    assert_eq!(
        total_births, summary.total_births,
        "frame-wise birth totals should match summary"
    );
    assert_eq!(
        total_deaths, summary.total_deaths,
        "frame-wise death totals should match summary"
    );

    let frames_with_births = report
        .frames
        .iter()
        .filter(|frame| frame.births > 0)
        .count();
    assert!(
        frames_with_births >= 12,
        "birth activity should span many frames (frames_with_births={frames_with_births})"
    );

    let frames_with_deaths = report
        .frames
        .iter()
        .filter(|frame| frame.deaths > 0)
        .count();
    assert!(
        frames_with_deaths >= 6,
        "deaths should appear in multiple frames (frames_with_deaths={frames_with_deaths})"
    );

    let agent_counts: Vec<usize> = report
        .frames
        .iter()
        .map(|frame| frame.agent_count)
        .collect();
    let min_agents = *agent_counts.iter().min().expect("min agent count");
    let max_agents = *agent_counts.iter().max().expect("max agent count");
    assert!(
        max_agents > min_agents,
        "agent count should vary over the run (min={min_agents}, max={max_agents})"
    );
    assert_eq!(
        report.initial.agent_count + summary.total_births - summary.total_deaths,
        summary.final_agent_count,
        "agent conservation should match persistence totals"
    );

    {
        let guard = shared_world.lock().expect("world mutex");
        let world_config = guard.config();
        assert!(
            (world_config.food_growth_rate - updated_config.food_growth_rate).abs() < f32::EPSILON
        );
        assert!(
            (world_config.food_decay_rate - updated_config.food_decay_rate).abs() < f32::EPSILON
        );
        assert!(
            (world_config.metabolism_drain - updated_config.metabolism_drain).abs() < f32::EPSILON
        );
        assert!(
            (world_config.reproduction_rate_herbivore - updated_config.reproduction_rate_herbivore)
                .abs()
                < f32::EPSILON
        );
        assert_eq!(
            world_config.chart_flush_interval, updated_config.chart_flush_interval,
            "chart flush interval should reflect control update"
        );

        let history: Vec<_> = guard.history().cloned().collect();
        assert!(
            history.len() >= frames,
            "history should capture each simulated tick (len={})",
            history.len()
        );
        let births_in_history = history.iter().filter(|entry| entry.births > 0).count();
        let deaths_in_history = history.iter().filter(|entry| entry.deaths > 0).count();
        assert!(
            births_in_history >= 10,
            "history should record repeated birth activity (birth_ticks={births_in_history})"
        );
        assert!(
            deaths_in_history >= 5,
            "history should record repeated death activity (death_ticks={deaths_in_history})"
        );
    }

    // Allow asynchronous storage worker to flush outstanding batches.
    std::thread::sleep(std::time::Duration::from_millis(150));

    let conn = Connection::open(storage_path)?;
    let tick_count: i64 = conn.query_row("select count(*) from ticks", [], |row| row.get(0))?;
    assert!(
        tick_count as usize >= frames,
        "storage should persist all ticks (frames={}, rows={tick_count})",
        frames
    );

    let (stored_tick, stored_agents): (i64, i64) = conn.query_row(
        "select tick, agent_count from ticks order by tick desc limit 1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    assert_eq!(
        stored_tick as u64, summary.final_tick,
        "tick ledger should align with headless summary"
    );
    assert_eq!(
        stored_agents as usize, summary.final_agent_count,
        "tick ledger should capture final population size"
    );

    let births_records: i64 =
        conn.query_row("select count(*) from births", [], |row| row.get(0))?;
    assert_eq!(
        births_records as usize, summary.total_births,
        "birth records should match reported total"
    );

    let deaths_records: i64 =
        conn.query_row("select count(*) from deaths", [], |row| row.get(0))?;
    assert_eq!(
        deaths_records as usize, summary.total_deaths,
        "death records should match reported total"
    );

    let births_events: i64 = conn.query_row(
        "select coalesce(sum(count), 0) from events where kind = 'births'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(
        births_events as usize, summary.total_births,
        "birth events should sum to reported total"
    );

    let deaths_events: i64 = conn.query_row(
        "select coalesce(sum(count), 0) from events where kind = 'deaths'",
        [],
        |row| row.get(0),
    )?;
    assert_eq!(
        deaths_events as usize, summary.total_deaths,
        "death events should sum to reported total"
    );

    let births_metric: f64 = conn.query_row(
        "select value from metrics where name = 'births.total.count' order by tick desc limit 1",
        [],
        |row| row.get(0),
    )?;
    assert!(
        (births_metric - summary.total_births as f64).abs() < f64::EPSILON,
        "birth metrics should mirror totals"
    );

    let mortality_metric: f64 = conn.query_row(
        "select value from metrics where name = 'mortality.total.count' order by tick desc limit 1",
        [],
        |row| row.get(0),
    )?;
    assert!(
        (mortality_metric - summary.total_deaths as f64).abs() < f64::EPSILON,
        "mortality metrics should mirror totals"
    );

    Ok(())
}
