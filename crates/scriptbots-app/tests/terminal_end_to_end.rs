use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use scriptbots_app::{
    ControlCommand, ControlRuntime, ControlServerConfig, McpTransportConfig,
    renderer::{Renderer, RendererContext},
    terminal::TerminalRenderer,
};
use scriptbots_core::{AgentData, Generation, Position, ScriptBotsConfig, Velocity, WorldState};
use scriptbots_storage::{AnalyticsSnapshotProvider, StoragePipeline, StorageReader};
use serde::Deserialize;
use serial_test::serial;
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
    buffer: Option<HeadlessBufferEvidenceDto>,
}

#[derive(Debug, Deserialize)]
struct HeadlessBufferEvidenceDto {
    backend: String,
    capability_profile: String,
    viewport_width: u16,
    viewport_height: u16,
    current_tick: u64,
    non_blank_cells: usize,
    styled_cells: usize,
    skipped_cells: usize,
    forced_width_cells: usize,
    empty_symbol_cells: usize,
    full_cell_fnv1a64: String,
    semantic_regions: Vec<String>,
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

fn assert_test_backend_buffer_evidence(report: &HeadlessReportDto) {
    assert!(
        report.initial.buffer.is_none(),
        "the initial measurement precedes the first rendered frame"
    );
    for frame in &report.frames {
        let buffer = frame
            .buffer
            .as_ref()
            .expect("every TestBackend frame must retain rendered-buffer evidence");
        assert_eq!(buffer.backend, "ratatui_test_backend");
        assert_eq!(
            buffer.capability_profile, "ascii_natural_fixed_80x36",
            "headless evidence must identify its deterministic non-live capability profile"
        );
        assert_eq!((buffer.viewport_width, buffer.viewport_height), (80, 36));
        assert_eq!(
            buffer.current_tick, frame.tick,
            "rendered tick text must describe the same snapshot as the frame report"
        );
        assert!(
            buffer.non_blank_cells > 100,
            "the TestBackend frame must contain substantive rendered content"
        );
        assert!(
            buffer.styled_cells > 0,
            "the full-cell proof must cover styles"
        );
        assert_eq!(
            buffer.skipped_cells, 0,
            "the ASCII TestBackend profile must not skip cells"
        );
        assert_eq!(
            buffer.forced_width_cells, 0,
            "the ASCII TestBackend profile must not require forced-width cells"
        );
        assert_eq!(
            buffer.empty_symbol_cells, 0,
            "the ASCII TestBackend profile must fill each cell symbol"
        );
        assert_eq!(
            buffer.full_cell_fnv1a64.len(),
            16,
            "buffer evidence must include the full deterministic FNV-1a digest"
        );
        assert_eq!(
            buffer.semantic_regions,
            ["terminal_hud", "current_tick", "world_map", "vital_stats"],
            "buffer evidence must come from the expected terminal HUD regions"
        );
    }
}

#[test]
#[serial]
fn terminal_test_backend_generates_semantic_buffer_report() -> Result<()> {
    let _env_guard = ENV_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env guard");

    let _ = tracing_subscriber::fmt()
        .with_env_filter("warn,scriptbots_app=info")
        .with_max_level(Level::INFO)
        .with_test_writer()
        .try_init();

    let frames = 32usize;

    let report_dir = tempdir()?;
    let report_path = report_dir.path().join("terminal_report.json");

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
        reproduction_rate_herbivore: 2.5,
        reproduction_rate_carnivore: 2.5,
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
        reproduction_attempt_interval: 1,
        reproduction_attempt_chance: 1.0,
        ..ScriptBotsConfig::default()
    };
    config.food_sharing_rate = 0.15;
    config.food_transfer_rate = 0.0025;

    let mut world = WorldState::new(config.clone())?;
    let mut rng = SmallRng::seed_from_u64(0xBAD5_EED5);
    for index in 0..32 {
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
            if index < 4 { -0.01 } else { 1.0 },
            color,
            0.1,
            false,
            0,
            Generation::default(),
        );
        world
            .try_spawn_agent(agent)
            .expect("terminal fixture agent is finite");
    }
    let shared_world = Arc::new(Mutex::new(world));

    let analytics = AnalyticsSnapshotProvider::empty();

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
            analytics: analytics.clone(),
            control_runtime: &control_runtime,
            command_drain,
            command_submit,
        };
        renderer.run(context)?;
    }

    control_runtime.shutdown()?;

    let report_contents = std::fs::read_to_string(&report_path)?;
    let report: HeadlessReportDto = serde_json::from_str(&report_contents)?;
    assert_test_backend_buffer_evidence(&report);
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
        frames_with_deaths >= 1,
        "deaths should appear in the run (frames_with_deaths={frames_with_deaths})"
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
#[serial]
fn terminal_test_backend_applies_control_updates_and_renders_receipts() -> Result<()> {
    let _env_guard = ENV_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env guard");

    let _ = tracing_subscriber::fmt()
        .with_env_filter("warn,scriptbots_app=info")
        .with_max_level(Level::INFO)
        .with_test_writer()
        .try_init();

    let frames = 37usize;

    let report_dir = tempdir()?;
    let report_path = report_dir.path().join("terminal_control_report.json");

    let storage_dir = tempdir()?;
    let storage_path = storage_dir.path().join("terminal_control.sqlite");

    let mut env = EnvCleanup::new();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS", "1");
    let frames_env = frames.to_string();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", &frames_env);
    let report_env = report_path.to_string_lossy().into_owned();
    env.set("SCRIPTBOTS_TERMINAL_HEADLESS_REPORT", &report_env);
    env.set("RUST_LOG", "warn,scriptbots_app=info");
    env.set("RUST_LOG_STYLE", "never");

    let mut config = ScriptBotsConfig {
        world_width: 200,
        world_height: 140,
        food_cell_size: 20,
        population_minimum: 0,
        population_spawn_interval: 0,
        persistence_interval: 5,
        history_capacity: 640,
        rng_seed: Some(0x51EED5),
        initial_food: 0.3,
        food_max: 1.0,
        food_respawn_interval: 8,
        food_respawn_amount: 0.4,
        food_growth_rate: 0.16,
        food_decay_rate: 0.0008,
        food_diffusion_rate: 0.16,
        reproduction_cooldown: 12,
        reproduction_rate_herbivore: 1.0,
        reproduction_rate_carnivore: 1.0,
        reproduction_energy_cost: 0.2,
        reproduction_child_energy: 0.6,
        reproduction_spawn_jitter: 10.0,
        reproduction_spawn_back_distance: 5.0,
        reproduction_partner_chance: 0.35,
        metabolism_drain: 0.007,
        movement_drain: 0.014,
        temperature_discomfort_rate: 0.001,
        food_intake_rate: 0.009,
        food_waste_rate: 0.0006,
        chart_flush_interval: 240,
        reproduction_attempt_interval: 1,
        reproduction_attempt_chance: 0.5,
        ..ScriptBotsConfig::default()
    };
    config.analytics_stride.behavior_metrics = 24;
    config.analytics_stride.lifecycle_events = 12;
    config.food_sharing_rate = 0.16;
    config.food_transfer_rate = 0.002;

    let mut world = WorldState::new(config.clone())?;
    let mut pipeline = StoragePipeline::create_new_file_with_thresholds(
        storage_path
            .to_str()
            .expect("temporary storage path should be utf-8"),
        1,
        1,
        1,
        1,
    )?;
    let analytics = pipeline.analytics_provider();
    world.set_persistence(Box::new(pipeline.sink()));

    let mut rng = SmallRng::seed_from_u64(0xDECAF00D);
    for index in 0..24 {
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
        let health = if index < 4 { -0.01 } else { 1.0 };
        let agent = AgentData::new(
            position,
            Velocity::default(),
            heading,
            health,
            color,
            0.05,
            false,
            0,
            Generation::default(),
        );
        world
            .try_spawn_agent(agent)
            .expect("terminal fixture agent is finite");
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
    updated_config.food_growth_rate = 0.36;
    updated_config.food_decay_rate = 0.00025;
    updated_config.food_respawn_amount = 0.72;
    updated_config.metabolism_drain = 0.01;
    updated_config.reproduction_cooldown = 10;
    updated_config.reproduction_rate_herbivore = 1.0;
    updated_config.reproduction_rate_carnivore = 1.0;
    updated_config.reproduction_energy_cost = 0.2;
    updated_config.reproduction_child_energy = 0.6;
    updated_config.chart_flush_interval = 90;
    updated_config.reproduction_attempt_interval = 1;
    updated_config.reproduction_attempt_chance = 0.5;
    let submit_ok = command_submit(ControlCommand::UpdateConfig(Box::new(
        updated_config.clone(),
    )));
    assert!(submit_ok, "control queue rejected config update");

    let renderer = TerminalRenderer::default();
    {
        let context = RendererContext {
            world: Arc::clone(&shared_world),
            analytics: analytics.clone(),
            control_runtime: &control_runtime,
            command_drain,
            command_submit,
        };
        renderer.run(context)?;
    }
    control_runtime.shutdown()?;
    let finalized_tail = shared_world
        .lock()
        .expect("world mutex")
        .finalize_persistence()?;
    assert!(
        finalized_tail,
        "a 37-tick run with a five-tick cadence must admit its partial tail"
    );
    let shutdown = pipeline.shutdown()?;

    let report_contents = std::fs::read_to_string(&report_path)?;
    let report: HeadlessReportDto = serde_json::from_str(&report_contents)?;
    assert_test_backend_buffer_evidence(&report);
    let summary = &report.summary;

    assert_eq!(summary.frame_count, frames);
    assert_eq!(
        summary.final_tick,
        report.initial.tick + frames as u64,
        "final tick should equal initial tick plus simulated frames"
    );
    assert!(
        summary.total_births > 10,
        "integration run should yield bounded reproduction (births={})",
        summary.total_births
    );
    assert!(
        summary.total_deaths >= 4,
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
        summary.avg_energy_max > summary.avg_energy_min,
        "mean energy should vary once control updates take effect"
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
        frames_with_births >= 3,
        "birth activity should span many frames (frames_with_births={frames_with_births})"
    );

    let frames_with_deaths = report
        .frames
        .iter()
        .filter(|frame| frame.deaths > 0)
        .count();
    assert!(
        frames_with_deaths >= 1,
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
            births_in_history >= 3,
            "history should record repeated birth activity (birth_ticks={births_in_history})"
        );
        assert!(
            deaths_in_history >= 1,
            "history should record repeated death activity (death_ticks={deaths_in_history})"
        );
    }

    drop(shared_world);
    assert!(
        analytics.snapshot().stopped,
        "explicit pipeline shutdown must be visible to frontend readers"
    );
    assert_eq!(shutdown.committed_tick, Some(summary.final_tick));
    assert_eq!(
        shutdown.guarantee,
        scriptbots_storage::PersistenceGuarantee::Durable
    );

    let reader = StorageReader::open(&storage_path.to_string_lossy())?;
    let ledger = reader.run_ledger_summary()?;
    let expected_tick_rows = u64::try_from(frames.div_ceil(5)).expect("frame budget fits in u64");
    assert_eq!(
        ledger.tick_count, expected_tick_rows,
        "storage should contain every cadence boundary plus one final partial batch"
    );

    let latest_tick = ledger
        .latest_tick
        .as_ref()
        .expect("completed run should have a durable tick row");
    assert_eq!(
        latest_tick.tick, summary.final_tick,
        "tick ledger should align with headless summary"
    );
    assert_eq!(
        latest_tick.agent_count, summary.final_agent_count,
        "tick ledger should capture final population size"
    );

    assert_eq!(
        ledger.birth_records,
        u64::try_from(summary.total_births).expect("birth total fits in u64"),
        "birth records should match reported total"
    );

    assert_eq!(
        ledger.death_records,
        u64::try_from(summary.total_deaths).expect("death total fits in u64"),
        "death records should match reported total"
    );

    assert_eq!(
        ledger.birth_events,
        u64::try_from(summary.total_births).expect("birth total fits in u64"),
        "birth events should sum to reported total"
    );

    assert_eq!(
        ledger.death_events,
        u64::try_from(summary.total_deaths).expect("death total fits in u64"),
        "death events should sum to reported total"
    );

    let metrics = reader.recent_metrics(None)?;
    let births_metric: f64 = metrics
        .iter()
        .filter(|reading| reading.name == "births.total.count")
        .map(|reading| reading.value)
        .sum();
    assert!(
        (births_metric - summary.total_births as f64).abs() < f64::EPSILON,
        "birth metric samples should sum to reported totals"
    );

    let mortality_metric: f64 = metrics
        .iter()
        .filter(|reading| reading.name == "mortality.total.count")
        .map(|reading| reading.value)
        .sum();
    assert!(
        (mortality_metric - summary.total_deaths as f64).abs() < f64::EPSILON,
        "mortality metric samples should sum to reported totals"
    );

    reader.close()?;
    Ok(())
}
