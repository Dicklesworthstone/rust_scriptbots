use std::sync::{Arc, Mutex};

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
use serde_json::Value;
use tempfile::tempdir;
use tracing::Level;

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

#[test]
fn terminal_headless_generates_report() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("scriptbots_app=info,scriptbots_core=warn")
        .with_max_level(Level::INFO)
        .with_test_writer()
        .try_init();

    let frames = 24usize;

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

    let mut config = ScriptBotsConfig::default();
    config.food_cell_size = 32;
    config.world_width = 160;
    config.world_height = 96;
    config.population_minimum = 0;
    config.population_spawn_interval = 0;
    config.history_capacity = 512;
    config.rng_seed = Some(0xDEC0_DEAD);

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
    let report: Value = serde_json::from_str(&report_contents)?;

    let summary = report
        .get("summary")
        .and_then(Value::as_object)
        .expect("report summary object");

    assert_eq!(
        summary
            .get("frame_count")
            .and_then(Value::as_u64)
            .expect("frame_count"),
        frames as u64,
        "headless renderer should honour requested frame budget"
    );

    assert!(
        summary
            .get("ticks_simulated")
            .and_then(Value::as_u64)
            .unwrap_or_default()
            >= frames as u64,
        "tick progress should match simulated frames"
    );

    assert!(
        summary
            .get("final_agent_count")
            .and_then(Value::as_u64)
            .unwrap_or_default()
            > 0,
        "simulation should retain surviving agents"
    );

    let energy_min = summary
        .get("avg_energy_min")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    let energy_max = summary
        .get("avg_energy_max")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    assert!(
        energy_max >= energy_min,
        "energy extrema should be well ordered"
    );

    let frames_array = report
        .get("frames")
        .and_then(Value::as_array)
        .expect("frames array");
    assert_eq!(
        frames_array.len(),
        frames,
        "per-frame stats should match recorded frame count"
    );

    {
        let guard = shared_world.lock().expect("world mutex");
        let final_tick = summary
            .get("final_tick")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        assert!(
            guard.tick().0 >= final_tick,
            "world tick should advance to the reported final tick"
        );
    }

    Ok(())
}

#[test]
fn terminal_headless_applies_control_updates() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("scriptbots_app=info,scriptbots_core=warn")
        .with_max_level(Level::INFO)
        .with_test_writer()
        .try_init();

    let frames = 48usize;

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

    let mut config = ScriptBotsConfig::default();
    config.world_width = 200;
    config.world_height = 140;
    config.food_cell_size = 20;
    config.population_minimum = 0;
    config.population_spawn_interval = 0;
    config.persistence_interval = 1;
    config.analytics_stride.behavior_metrics = 24;
    config.rng_seed = Some(0x51EED5);

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
    updated_config.food_growth_rate = 0.02;
    updated_config.food_decay_rate = 0.0015;
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
    let report: Value = serde_json::from_str(&report_contents)?;
    let summary = report
        .get("summary")
        .and_then(Value::as_object)
        .expect("report summary object");

    assert_eq!(
        summary
            .get("frame_count")
            .and_then(Value::as_u64)
            .expect("frame count"),
        frames as u64
    );
    assert!(
        summary
            .get("ticks_simulated")
            .and_then(Value::as_u64)
            .unwrap_or_default()
            >= frames as u64,
        "tick progress should advance with frames"
    );

    let history_len;
    {
        let guard = shared_world.lock().expect("world mutex");
        let world_config = guard.config();
        assert!(
            (world_config.food_growth_rate - updated_config.food_growth_rate).abs() < f32::EPSILON
        );
        assert!(
            (world_config.food_decay_rate - updated_config.food_decay_rate).abs() < f32::EPSILON
        );
        assert_eq!(
            world_config.chart_flush_interval, updated_config.chart_flush_interval,
            "chart flush interval should reflect control update"
        );
        history_len = guard.history().count();
        assert!(
            history_len > 0,
            "headless run should record tick history entries"
        );
    }

    // Allow asynchronous storage worker to flush outstanding batches.
    std::thread::sleep(std::time::Duration::from_millis(100));

    let conn = Connection::open(storage_path)?;
    let tick_count: i64 = conn.query_row("select count(*) from ticks", [], |row| row.get(0))?;
    assert!(
        tick_count > 0,
        "storage should persist at least one tick (got {tick_count})"
    );

    let metric_count: i64 = conn.query_row("select count(*) from metrics", [], |row| row.get(0))?;
    assert!(
        metric_count > 0,
        "storage should persist metrics for analytics (got {metric_count})"
    );

    Ok(())
}
