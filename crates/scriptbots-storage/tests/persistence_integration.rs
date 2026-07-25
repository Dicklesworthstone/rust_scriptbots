use fsqlite::compat::{OpenFlags, RowExt, open_with_flags};
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
        StoragePipeline::create_unattributed_file_with_thresholds(path_str, 1, 1, 1, 1)
            .expect("pipeline");
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
                agent_uid: None,
                kind: ReplayEventKind::BrainOutputs {
                    outputs: vec![0.25, 0.75],
                },
            }],
            narrative_events: Vec::new(),
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
        let (mut world, mut persistence) =
            WorldState::with_persistence(config, Box::new(pipeline.sink())).expect("world");
        world
            .try_spawn_agent(AgentData::default())
            .expect("default agent is finite");

        for _ in 0..5 {
            persistence
                .step(&mut world)
                .expect("file-backed persistence step");
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

#[test]
/// Every narrative event emitted online must be readable back from the run database.
///
/// This is a real parity proof again (`bd-erff`). It could not be one for most of its life:
/// storage charged `PersistenceBatch.narrative_events` against the admission budget and then
/// discarded it, so there was no offline side to compare against. An earlier session made
/// the file compile by replacing a call to the nonexistent `StorageReader::search_narrative`
/// with a `max_tick` probe, which left the parity claim in the name while asserting nothing
/// about it.
///
/// The offline half now exists, and asserting it immediately earned its keep: it caught that
/// `StorageBuffer::append` silently dropped `run_events`, so rows were built per batch and
/// then lost when batches merged into the flush buffer. The writer looked correct and
/// persisted nothing.
fn narrative_events_persisted_online_are_readable_offline() {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_micros();
    let path = std::env::temp_dir().join(format!(
        "scriptbots_narrative_parity_{}_{}.sqlite",
        std::process::id(),
        timestamp
    ));
    let path_str = path.to_str().expect("utf8 path");
    let mut pipeline =
        StoragePipeline::create_unattributed_file_with_thresholds(path_str, 1, 1, 1, 1)
            .expect("pipeline");

    let config = ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        rng_seed: Some(0xCA1F),
        persistence_interval: 1,
        ..ScriptBotsConfig::default()
    };

    let online_events = {
        let (mut world, mut persistence) =
            WorldState::with_persistence(config, Box::new(pipeline.sink())).expect("world");

        for seed in 0..24 {
            world
                .try_spawn_agent(AgentData {
                    position: scriptbots_core::Position::new(
                        (seed * 37 % 190) as f32,
                        (seed * 53 % 190) as f32,
                    ),
                    health: 1.0,
                    ..AgentData::default()
                })
                .expect("valid agent");
        }

        for _ in 0..500 {
            persistence.step(&mut world).expect("persistence step");
        }

        world
            .narrative_events()
            .iter()
            .map(|e| (e.tick.0, e.human_text.clone()))
            .collect::<Vec<_>>()
    };

    pipeline.shutdown().expect("clean shutdown");

    let storage = StorageReader::open(path_str).expect("open storage");
    let max_tick = storage.max_tick().expect("read max tick");
    assert!(max_tick.is_some());
    assert!(
        !online_events.is_empty(),
        "expected online events generated during run"
    );
    storage.close().expect("close storage reader");

    // The offline half, restored (bd-erff). This test was named for online/offline parity
    // but could not check it: narrative events reached the persistence boundary and were
    // discarded, because nothing wrote `StorageBuffer.run_events`. With the writer in place
    // every event the world emitted must now be readable back from the run database.
    let reader = open_with_flags(path_str, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("independent read-only reader opens");
    let rows = reader
        .query("SELECT tick, human_text FROM run_events ORDER BY tick ASC, human_text ASC")
        .expect("run_events query runs");
    let mut offline_events = rows
        .iter()
        .map(|row| {
            let tick: i64 = row.get_typed(0).expect("tick is INTEGER");
            let text: String = row.get_typed(1).expect("human_text is TEXT");
            (u64::try_from(tick).expect("tick is non-negative"), text)
        })
        .collect::<Vec<_>>();
    reader.close().expect("read-only reader closes");

    let mut expected = online_events.clone();
    expected.sort();
    offline_events.sort();
    assert_eq!(
        offline_events, expected,
        "every narrative event the world emitted online must be readable back from the run \
         database; this is the parity the test is named for"
    );

    let _ = fs::remove_file(&path);
}
