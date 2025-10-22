use scriptbots_core::{AgentData, ScriptBotsConfig, WorldState};
use scriptbots_storage::StoragePipeline;
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn storage_persists_metrics_roundtrip() {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_micros();
    let path = std::env::temp_dir().join(format!(
        "scriptbots_storage_test_{}_{}.duckdb",
        std::process::id(),
        timestamp
    ));

    let path_str = path.to_str().expect("utf8 path");
    let pipeline = StoragePipeline::with_thresholds(path_str, 1, 1, 1, 1).expect("pipeline");
    let storage_arc = pipeline.storage();

    let config = ScriptBotsConfig {
        world_width: 128,
        world_height: 128,
        food_cell_size: 16,
        initial_food: 0.25,
        food_max: 1.0,
        persistence_interval: 1,
        history_capacity: 32,
        ..ScriptBotsConfig::default()
    };

    {
        let mut world = WorldState::with_persistence(config, Box::new(pipeline)).expect("world");
        world.spawn_agent(AgentData::default());

        for _ in 0..5 {
            world.step();
        }
    }

    let mut guard = storage_arc.lock().expect("storage lock");
    guard.flush().expect("flush");
    let metrics = guard.latest_metrics(8).expect("latest metrics");
    assert!(!metrics.is_empty(), "expected persisted metrics");

    let predators = guard.top_predators(4).expect("top predators query");
    assert!(
        predators.len() <= 4,
        "top predators should not exceed requested limit"
    );

    drop(guard);
    let _ = fs::remove_file(&path);
}
