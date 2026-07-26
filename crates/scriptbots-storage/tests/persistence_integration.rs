use fsqlite::compat::{OpenFlags, RowExt, open_with_flags};
use scriptbots_core::{
    AgentData, BrainRunner, INPUT_SIZE, OUTPUT_SIZE, PersistenceBatch, Position, ReplayEvent,
    ReplayEventKind, ReplayInteractionKind, ScriptBotsConfig, Tick, TickSummary, WorldState,
    channels::OutputChannel,
};
use scriptbots_storage::{StorageDeadlines, StoragePipeline, StorageReader};
use std::{
    fs,
    time::{Duration, SystemTime, UNIX_EPOCH},
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
                position: None,
                counterpart: None,
                counterpart_position: None,
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

/// Build a batch carrying exactly the supplied replay events at `tick`.
fn batch_with_replay_events(tick: u64, replay_events: Vec<ReplayEvent>) -> PersistenceBatch {
    let interaction_count = replay_events
        .iter()
        .filter(|event| matches!(event.kind, ReplayEventKind::Interaction { .. }))
        .count();
    let events = if interaction_count == 0 {
        Vec::new()
    } else {
        vec![
            scriptbots_core::PersistenceEvent::new(
                scriptbots_core::PersistenceEventKind::Custom(std::borrow::Cow::Borrowed(
                    scriptbots_core::INTERACTION_EVENTS_OBSERVED_KIND,
                )),
                interaction_count,
            ),
            scriptbots_core::PersistenceEvent::new(
                scriptbots_core::PersistenceEventKind::Custom(std::borrow::Cow::Borrowed(
                    scriptbots_core::INTERACTION_EVENTS_PERSISTED_KIND,
                )),
                interaction_count,
            ),
        ]
    };
    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(tick),
            agent_count: 2,
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
        events,
        agents: Vec::new(),
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events,
        narrative_events: Vec::new(),
    }
}

fn temp_run_path(label: &str) -> std::path::PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_micros();
    std::env::temp_dir().join(format!(
        "scriptbots_storage_{label}_{}_{}.sqlite",
        std::process::id(),
        timestamp
    ))
}

#[derive(Debug)]
struct GiveIntentBrain {
    give: bool,
}

impl BrainRunner for GiveIntentBrain {
    fn kind(&self) -> &'static str {
        if self.give {
            "test.storage.give"
        } else {
            "test.storage.receive"
        }
    }

    fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        let mut outputs = [0.0; OUTPUT_SIZE];
        outputs[OutputChannel::GiveIntent.index()] = f32::from(u8::from(self.give));
        outputs
    }
}

/// A seeded 2k world must preserve the exact core interaction count through durable SQL.
#[test]
fn seeded_2k_world_interaction_count_matches_durable_rows() {
    const AGENTS: usize = 2_000;
    const PAIRS: usize = AGENTS / 2;

    let path = temp_run_path("seeded_2k_interactions");
    let path_str = path.to_str().expect("utf8 path");
    let proof_deadline = Duration::from_secs(10 * 60);
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds_and_deadlines(
        path_str,
        1,
        1,
        1,
        1,
        StorageDeadlines {
            flush_ack: proof_deadline,
            shutdown_ack: proof_deadline,
            ..StorageDeadlines::default()
        },
    )
    .expect("2k durable pipeline");
    let (mut world, mut persistence) = WorldState::with_persistence(
        ScriptBotsConfig {
            world_width: 5_100,
            world_height: 2_100,
            food_cell_size: 50,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_growth_rate: 0.0,
            food_decay_rate: 0.0,
            food_diffusion_rate: 0.0,
            food_intake_rate: 0.0,
            food_waste_rate: 0.0,
            food_transfer_rate: 0.01,
            food_sharing_distance: 2.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            reproduction_attempt_chance: 0.0,
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 1,
            interaction_event_tick_cap: PAIRS,
            interaction_event_tick_stride: 1,
            rng_seed: Some(0x2000_5EED),
            ..ScriptBotsConfig::default()
        },
        Box::new(pipeline.sink()),
    )
    .expect("2k seeded world");

    let giver_brain = world
        .brain_registry_mut()
        .expect("giver brain registry")
        .register("test.storage.give", |_rng| {
            Ok(Box::new(GiveIntentBrain { give: true }))
        });
    let receiver_brain = world
        .brain_registry_mut()
        .expect("receiver brain registry")
        .register("test.storage.receive", |_rng| {
            Ok(Box::new(GiveIntentBrain { give: false }))
        });

    for pair in 0..PAIRS {
        let grid_x = u16::try_from(pair % 50).expect("2k grid x fits u16");
        let grid_y = u16::try_from(pair / 50).expect("2k grid y fits u16");
        let x = 50.0 + f32::from(grid_x) * 100.0;
        let y = 50.0 + f32::from(grid_y) * 100.0;
        let giver = world
            .try_spawn_agent(AgentData {
                position: Position::new(x, y),
                ..AgentData::default()
            })
            .expect("seed giver");
        let receiver = world
            .try_spawn_agent(AgentData {
                position: Position::new(x + 1.0, y),
                ..AgentData::default()
            })
            .expect("seed receiver");
        assert!(
            world
                .bind_agent_brain(giver, giver_brain)
                .expect("bind giver brain")
        );
        assert!(
            world
                .bind_agent_brain(receiver, receiver_brain)
                .expect("bind receiver brain")
        );
    }

    let completion = persistence
        .step_outcome(&mut world)
        .expect("run the seeded interaction tick");
    assert!(
        completion.fault.is_none(),
        "the seeded science boundary must complete without a contained fault"
    );
    let batch = persistence
        .pending_batch()
        .expect("the interval-one boundary stages a persistence batch");
    let core_edges = batch
        .replay_events
        .iter()
        .filter(|event| matches!(event.kind, ReplayEventKind::Interaction { .. }))
        .count();
    let counter = |kind: &str| {
        batch
            .events
            .iter()
            .find_map(|event| match &event.kind {
                scriptbots_core::PersistenceEventKind::Custom(name) if name == kind => {
                    Some(event.count)
                }
                _ => None,
            })
            .unwrap_or(0)
    };
    let core_observed = counter(scriptbots_core::INTERACTION_EVENTS_OBSERVED_KIND);
    let core_persisted = counter(scriptbots_core::INTERACTION_EVENTS_PERSISTED_KIND);
    let core_sampled_out = counter(scriptbots_core::INTERACTION_EVENTS_SAMPLED_OUT_KIND);
    let core_truncated = counter(scriptbots_core::INTERACTION_EVENTS_TRUNCATED_KIND);
    assert_eq!(batch.summary.agent_count, AGENTS);
    assert_eq!(core_edges, PAIRS);
    assert_eq!(
        (
            core_observed,
            core_persisted,
            core_sampled_out,
            core_truncated
        ),
        (PAIRS, PAIRS, 0, 0)
    );
    assert!(
        persistence
            .admit_pending(&mut world)
            .expect("admit the exact staged 2k batch")
    );
    pipeline
        .flush_and_wait()
        .expect("durably flush the 2k batch");
    pipeline.shutdown().expect("durable pipeline shutdown");

    let reader = open_with_flags(path_str, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("independent read-only reader opens");
    let durable = reader
        .query(
            "SELECT
               (SELECT COUNT(*) FROM interactions),
               COALESCE(SUM(CASE WHEN kind = 'interaction_events_observed' THEN count ELSE 0 END), 0),
               COALESCE(SUM(CASE WHEN kind = 'interaction_events_persisted' THEN count ELSE 0 END), 0)
             FROM events",
        )
        .expect("durable accounting query runs");
    let durable_edges: i64 = durable[0].get_typed(0).expect("edge count");
    let durable_observed: i64 = durable[0].get_typed(1).expect("observed count");
    let durable_persisted: i64 = durable[0].get_typed(2).expect("persisted count");
    assert_eq!(
        usize::try_from(durable_edges).expect("edge count is non-negative"),
        core_edges,
        "the durable interaction graph must contain every world-emitted edge exactly once"
    );
    assert_eq!(
        usize::try_from(durable_observed).expect("observed count is non-negative"),
        core_observed
    );
    assert_eq!(
        usize::try_from(durable_persisted).expect("persisted count is non-negative"),
        core_persisted
    );
    reader.close().expect("read-only reader closes");

    let _ = fs::remove_file(&path);
}

/// Typed combat and food-share facts must retain their source tick, stable participants,
/// positions, kind, and magnitude through the durable projection.
#[test]
fn typed_pairwise_events_persist_as_queryable_interaction_edges() {
    let path = temp_run_path("interaction_edge");
    let path_str = path.to_str().expect("utf8 path");

    let actor = scriptbots_core::AgentUid(7);
    let target = scriptbots_core::AgentUid(11);
    let actor_position = scriptbots_core::Position::new(12.5, -3.25);
    let target_position = scriptbots_core::Position::new(14.0, -2.5);

    {
        let mut pipeline =
            StoragePipeline::create_unattributed_file_with_thresholds(path_str, 1, 1, 1, 1)
                .expect("pipeline");
        pipeline
            .submit(&batch_with_replay_events(
                4,
                vec![
                    ReplayEvent {
                        agent_uid: Some(actor),
                        position: Some(actor_position),
                        counterpart: Some(target),
                        counterpart_position: Some(target_position),
                        kind: ReplayEventKind::Interaction {
                            tick: Tick(2),
                            ordinal: 0,
                            kind: ReplayInteractionKind::Combat,
                            magnitude: 0.375,
                        },
                    },
                    ReplayEvent {
                        agent_uid: Some(target),
                        position: Some(target_position),
                        counterpart: Some(actor),
                        counterpart_position: Some(actor_position),
                        kind: ReplayEventKind::Interaction {
                            tick: Tick(4),
                            ordinal: 0,
                            kind: ReplayInteractionKind::FoodShare,
                            magnitude: 0.125,
                        },
                    },
                    ReplayEvent {
                        agent_uid: Some(actor),
                        position: Some(actor_position),
                        counterpart: None,
                        counterpart_position: None,
                        kind: ReplayEventKind::BrainOutputs {
                            outputs: vec![0.5, 0.5],
                        },
                    },
                ],
            ))
            .expect("hand-built replay fixture enters the bounded queue");
        pipeline.flush_and_wait().expect("flush the staged batch");
        pipeline.shutdown().expect("durable pipeline shutdown");
    }

    let storage = StorageReader::open(path_str).expect("open storage after shutdown");

    // Premise: all events were actually written. Without this the exclusion assertion below
    // would be satisfied by a run that persisted nothing at all.
    let replayed = storage.load_replay_events().expect("replay events");
    assert_eq!(
        replayed.len(),
        3,
        "both interactions and the single-agent control must reach the database"
    );
    let edge_events = replayed
        .iter()
        .filter(|persisted| persisted.event.counterpart.is_some())
        .collect::<Vec<_>>();
    assert_eq!(edge_events.len(), 2);
    assert_eq!(edge_events[0].tick, 2);
    assert_eq!(edge_events[0].event.agent_uid, Some(actor));
    assert_eq!(edge_events[0].event.counterpart, Some(target));
    assert_eq!(
        edge_events[0].event.position,
        Some(actor_position),
        "the emission-time actor position must survive the write path"
    );
    assert_eq!(
        edge_events[0].event.counterpart_position,
        Some(target_position),
        "the emission-time counterpart position must survive the write path"
    );
    assert!(matches!(
        edge_events[0].event.kind,
        ReplayEventKind::Interaction {
            tick: Tick(2),
            ordinal: 0,
            kind: ReplayInteractionKind::Combat,
            magnitude,
        } if magnitude.to_bits() == 0.375_f32.to_bits()
    ));
    assert!(matches!(
        edge_events[1].event.kind,
        ReplayEventKind::Interaction {
            tick: Tick(4),
            ordinal: 0,
            kind: ReplayInteractionKind::FoodShare,
            magnitude,
        } if magnitude.to_bits() == 0.125_f32.to_bits()
    ));

    let interactions = storage.recent_interactions(16).expect("interaction edges");
    assert_eq!(
        interactions.len(),
        2,
        "exactly the pairwise events are interactions; the single-agent event is not: \
         {interactions:?}"
    );
    let combat = &interactions[0];
    assert_eq!(combat.tick, 2);
    assert_eq!(combat.actor, actor);
    assert_eq!(combat.target, target);
    assert_eq!(combat.kind, "combat");
    assert_eq!(combat.actor_position, Some(actor_position));
    assert_eq!(combat.target_position, Some(target_position));
    assert_eq!(combat.value, Some(0.375));
    let food_share = &interactions[1];
    assert_eq!(food_share.tick, 4);
    assert_eq!(food_share.actor, target);
    assert_eq!(food_share.target, actor);
    assert_eq!(food_share.kind, "food_share");
    assert_eq!(food_share.value, Some(0.125));

    storage.close().expect("close storage reader");

    // The edge is answerable in SQL by an offline consumer that never links this crate --
    // the property bd-2z0.5.9 was filed for, and the one a JSON payload could not provide.
    let reader = open_with_flags(path_str, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("independent read-only reader opens");
    let rows = reader
        .query(
            "SELECT tick, actor_agent_uid, target_agent_uid, kind, value FROM interactions
             ORDER BY tick ASC, seq ASC",
        )
        .expect("interactions table is queryable");
    assert_eq!(rows.len(), 2, "exactly the two edges must be recorded");
    let source_tick: i64 = rows[0].get_typed(0).expect("tick is INTEGER");
    let actor_uid: i64 = rows[0].get_typed(1).expect("actor_agent_uid is INTEGER");
    let target_uid: i64 = rows[0].get_typed(2).expect("target_agent_uid is INTEGER");
    let kind: String = rows[0].get_typed(3).expect("kind is TEXT");
    let value: f64 = rows[0].get_typed(4).expect("value is REAL");
    assert_eq!(source_tick, 2);
    assert_eq!(actor_uid, 7);
    assert_eq!(target_uid, 11);
    assert_eq!(kind, "combat");
    assert_eq!(value, 0.375);

    let completeness = reader
        .query(&format!(
            "SELECT
               COALESCE(SUM(CASE WHEN kind = '{}' THEN count ELSE 0 END), 0),
               COALESCE(SUM(CASE WHEN kind = '{}' THEN count ELSE 0 END), 0),
               COALESCE(SUM(CASE WHEN kind = '{}' THEN count ELSE 0 END), 0),
               COALESCE(SUM(CASE WHEN kind = '{}' THEN count ELSE 0 END), 0)
             FROM events",
            scriptbots_core::INTERACTION_EVENTS_OBSERVED_KIND,
            scriptbots_core::INTERACTION_EVENTS_PERSISTED_KIND,
            scriptbots_core::INTERACTION_EVENTS_SAMPLED_OUT_KIND,
            scriptbots_core::INTERACTION_EVENTS_TRUNCATED_KIND,
        ))
        .expect("persisted interaction completeness counters are queryable");
    let observed: i64 = completeness[0].get_typed(0).expect("observed count");
    let projected: i64 = completeness[0].get_typed(1).expect("projected count");
    let sampled_out: i64 = completeness[0].get_typed(2).expect("sampled count");
    let truncated: i64 = completeness[0].get_typed(3).expect("truncated count");
    assert_eq!((observed, projected, sampled_out, truncated), (2, 2, 0, 0));

    // The accounting identity bd-2z0.5.9 asks for: an interaction row exists for exactly the
    // replay events that name two participants -- no edge without an event, no pairwise event
    // without an edge. Expressed as SQL over both tables rather than as two Rust counts, so it
    // also proves the shared (run_id, tick, seq) key really does join them.
    let orphans = reader
        .query(
            "SELECT
               (SELECT COUNT(*) FROM interactions i
                  LEFT JOIN replay_events e
                    ON e.run_id = i.run_id AND e.tick = i.tick AND e.seq = i.seq
                 WHERE e.run_id IS NULL),
               (SELECT COUNT(*) FROM replay_events e
                  LEFT JOIN interactions i
                    ON i.run_id = e.run_id AND i.tick = e.tick AND i.seq = e.seq
                 WHERE e.agent_uid IS NOT NULL
                   AND e.counterpart_uid IS NOT NULL
                   AND i.run_id IS NULL)",
        )
        .expect("accounting identity query runs");
    let edges_without_events: i64 = orphans[0].get_typed(0).expect("count is INTEGER");
    let pairwise_events_without_edges: i64 = orphans[0].get_typed(1).expect("count is INTEGER");
    assert_eq!(
        edges_without_events, 0,
        "an edge exists with no source event"
    );
    assert_eq!(
        pairwise_events_without_edges, 0,
        "a pairwise event was persisted without its interaction edge"
    );
    assert_eq!(
        observed,
        projected + sampled_out + truncated,
        "persisted completeness counters must account for every observed interaction"
    );
    assert_eq!(
        projected,
        i64::try_from(rows.len()).expect("row count fits i64"),
        "the projected counter must equal the durable SQL edge count"
    );

    reader.close().expect("read-only reader closes");

    let _ = fs::remove_file(&path);
}

/// A run whose events name no counterpart must yield an empty interaction set, not an error
/// and not a row with an invented participant.
///
/// The negative half of the guard above. Exercised in both directions because a writer with a
/// wrong pairwise filter -- say, one that required only an `agent_uid` -- would still satisfy
/// the positive test while turning every ordinary brain-output event into a fictional edge
/// between an agent and itself.
#[test]
fn events_without_a_counterpart_produce_no_interaction_edges() {
    let path = temp_run_path("no_interaction_edges");
    let path_str = path.to_str().expect("utf8 path");

    {
        let mut pipeline =
            StoragePipeline::create_unattributed_file_with_thresholds(path_str, 1, 1, 1, 1)
                .expect("pipeline");
        pipeline
            .submit(&batch_with_replay_events(
                1,
                vec![ReplayEvent {
                    agent_uid: Some(scriptbots_core::AgentUid(3)),
                    position: Some(scriptbots_core::Position::new(1.0, 2.0)),
                    counterpart: None,
                    counterpart_position: None,
                    kind: ReplayEventKind::BrainOutputs {
                        outputs: vec![0.1, 0.2],
                    },
                }],
            ))
            .expect("single-agent fixture enters the bounded queue");
        pipeline.flush_and_wait().expect("flush the staged batch");
        pipeline.shutdown().expect("durable pipeline shutdown");
    }

    let storage = StorageReader::open(path_str).expect("open storage after shutdown");
    assert_eq!(
        storage.load_replay_events().expect("replay events").len(),
        1,
        "premise: the event was persisted, so an empty interaction set is a real exclusion \
         rather than an empty database"
    );
    assert!(
        storage
            .recent_interactions(16)
            .expect("interaction edges")
            .is_empty(),
        "an event with no counterpart is not an interaction"
    );
    storage.close().expect("close storage reader");
    let _ = fs::remove_file(&path);
}
