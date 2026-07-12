use scriptbots_core::{
    AgentData, PersistenceBatch, ReplayEvent, ReplayEventKind, ScriptBotsConfig, Tick, TickSummary,
    WorldState,
};
use scriptbots_storage::{StoragePipeline, StorageReader};
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
        "scriptbots_storage_test_{}_{}.sqlite",
        std::process::id(),
        timestamp
    ));

    let path_str = path.to_str().expect("utf8 path");
    let mut pipeline =
        StoragePipeline::create_new_file_with_thresholds(path_str, 1, 1, 1, 1).expect("pipeline");
    let analytics = pipeline.analytics_provider();
    pipeline
        .submit(&PersistenceBatch {
            summary: TickSummary {
                tick: Tick(0),
                agent_count: 0,
                births: 0,
                deaths: 0,
                total_energy: 0.0,
                average_energy: 0.0,
                average_health: 0.0,
                max_age: 0,
                spike_hits: 0,
            },
            epoch: 0,
            closed: false,
            metrics: Vec::new(),
            events: Vec::new(),
            agents: Vec::new(),
            births: Vec::new(),
            deaths: Vec::new(),
            replay_events: vec![ReplayEvent {
                agent_id: None,
                kind: ReplayEventKind::BrainOutputs {
                    outputs: vec![0.25, 0.75],
                },
            }],
        })
        .expect("explicit replay fixture should enter the bounded queue");

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
        let mut world =
            WorldState::with_persistence(config, Box::new(pipeline.sink())).expect("world");
        world.spawn_agent(AgentData::default());

        for _ in 0..5 {
            world.step().expect("file-backed persistence step");
        }
    }
    let shutdown = pipeline.shutdown().expect("durable pipeline shutdown");
    assert!(
        shutdown.committed_tick.is_some(),
        "expected a committed tick receipt"
    );
    assert_eq!(
        shutdown.guarantee,
        scriptbots_storage::PersistenceGuarantee::Durable
    );

    let snapshot = analytics.snapshot();
    assert!(
        !snapshot.readings.is_empty(),
        "expected published analytics readings"
    );
    assert!(
        snapshot.committed_tick.is_some(),
        "expected a committed analytics tick"
    );
    assert!(snapshot.stopped, "shutdown should be visible to readers");

    let storage = StorageReader::open(path_str).expect("open storage after pipeline shutdown");

    let predators = storage.top_predators(4).expect("top predators query");
    assert!(
        predators.len() <= 4,
        "top predators should not exceed requested limit"
    );

    let max_tick = storage.max_tick().expect("max tick");
    assert!(max_tick.is_some(), "expected ticks recorded");

    let replay_events = storage.load_replay_events().expect("replay events");
    assert!(
        !replay_events.is_empty(),
        "expected at least one replay event"
    );

    let counts = storage.replay_event_counts().expect("replay event counts");
    assert!(
        !counts.is_empty(),
        "expected replay event counts to be populated"
    );

    storage.close().expect("close storage reader explicitly");
    let _ = fs::remove_file(&path);
}
