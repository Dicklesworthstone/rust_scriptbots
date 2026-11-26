//! Scaffold integration tests for scriptbots-analytics (bd-2z0.11.5).
//!
//! Deliberate design improvement over the original bead spec: instead of a
//! checked-in binary fixture DB (+ regeneration script that inevitably
//! drifts), every test synthesizes its own run database in a tempdir via the
//! real storage write path (`Storage::create_unattributed_file` → `persist` → `flush`
//! → `close`) and then reads it back through the real read path
//! (`StorageReader`). No binary blobs in git, no fixture drift, and the
//! storage crate's own admission machinery is exercised end-to-end.

use std::process::Command;

use scriptbots_analytics::{
    AnalyticsError, NARRATIVE_RECONSTRUCT_SCHEMA_VERSION, NARRATIVE_TIMELINE_SCHEMA_VERSION,
    NARRATIVE_VERIFICATION_SCHEMA_VERSION, NarrativeDifferenceKind, NarrativeVerificationError,
    NarrativeVerificationV1, REPLAY_TIMELINE_SCHEMA_VERSION, REPORT_SCHEMA_VERSION, ReaderCtx,
    Registry, ReportParams, verify_island_narrative, verify_narrative,
};
use scriptbots_core::narrative::{EventKind, EventRecord, NarrativeInputRecordV1, SubjectRef};
use scriptbots_core::{
    AgentData, AgentUid, Intervention, MetricSample, PersistenceBatch, Position, Region,
    ReplayEvent, ReplayEventKind, ReplayRngScope, ScriptBotsConfig, Tick, TickSummary, WorldState,
};
use scriptbots_storage::{
    Connection, NarrativeInputStreamError, RowExt, RunEventDecodeError, RunManifestRecord,
    SqliteValue, Storage, StorageError, StoragePipeline,
};
use serde_json::json;

fn batch(tick: u64, agent_count: usize, energy: f32) -> PersistenceBatch {
    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(tick),
            agent_count,
            births: 0,
            deaths: 0,
            total_energy: energy,
            average_energy: if agent_count == 0 {
                0.0
            } else {
                #[allow(clippy::cast_precision_loss)]
                let n = agent_count as f32;
                energy / n
            },
            average_health: 1.0,
            max_age: 0,
            spike_hits: 0,
        },
        epoch: 1,
        closed: false,
        metrics: vec![MetricSample::new("total_energy", f64::from(energy))],
        events: Vec::new(),
        agents: Vec::new(),
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events: Vec::new(),
        narrative_events: Vec::new(),
        genomes: Vec::new(),
    }
}

/// Builds a three-tick run database and returns its path.
fn synth_db(dir: &tempfile::TempDir) -> String {
    synth_db_with_events(dir, false, false)
}

fn narrative_event(tick: u64, kind: EventKind, metric: &str) -> EventRecord {
    EventRecord {
        schema_version: 1,
        tick: Tick(tick),
        kind,
        severity: 0.75,
        magnitude: 20.0,
        window: (0, tick),
        metric: metric.to_owned(),
        before: 10.0,
        after: 30.0,
        score: 2.5,
        subject: Some(SubjectRef::Agent(AgentUid(9_007_199_254_740_993))),
        human_text: "recovered | after <rain>\nagain".to_owned(),
    }
}

fn synth_db_with_events(
    dir: &tempfile::TempDir,
    include_replay: bool,
    include_narrative: bool,
) -> String {
    let path = dir.path().join("run.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create synth run db");
    for (tick, pop, energy) in [(1u64, 10usize, 100.0f32), (2, 12, 130.0), (3, 8, 90.0)] {
        let mut tick_batch = batch(tick, pop, energy);
        if include_replay && tick == 1 {
            tick_batch.replay_events.push(ReplayEvent {
                agent_uid: None,
                position: None,
                counterpart: None,
                counterpart_position: None,
                kind: ReplayEventKind::RngSample {
                    scope: ReplayRngScope::World,
                    range_min: 0.0,
                    range_max: 1.0,
                    value: 0.25,
                },
            });
        }
        if include_replay && tick == 3 {
            tick_batch.replay_events.push(ReplayEvent {
                agent_uid: None,
                position: None,
                counterpart: None,
                counterpart_position: None,
                kind: ReplayEventKind::BrainOutputs {
                    outputs: vec![0.5, -0.25],
                },
            });
        }
        if include_narrative && (tick == 1 || tick == 3) {
            tick_batch.narrative_events.push(narrative_event(
                tick,
                EventKind::PopulationBoom,
                "population",
            ));
        }
        if include_narrative && tick == 3 {
            // Persist reverse kind order; the report must use canonical identity
            // order, not the producer vector order or the replay-event sequence.
            tick_batch.narrative_events.push(narrative_event(
                tick,
                EventKind::EnergyRecovery,
                "average_energy",
            ));
        }
        storage.persist(&tick_batch).expect("persist batch");
    }
    storage.flush().expect("flush");
    storage.close().expect("close");
    path
}

/// A real production run whose complete input stream spans more than one report page.
fn real_narrative_db(dir: &tempfile::TempDir) -> (String, u64) {
    real_narrative_db_seeded(dir, 0x16_020_901)
}

fn real_narrative_db_seeded(dir: &tempfile::TempDir, seed: u64) -> (String, u64) {
    let path = dir.path().join("narrative.sqlite").display().to_string();
    let mut pipeline = StoragePipeline::create_unattributed_file(&path).expect("create run");
    let config = ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        closed: true,
        population_spawn_interval: 0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        aging_health_decay_rate: 0.0,
        spike_damage: 0.0,
        spike_energy_cost: 0.0,
        reproduction_energy_threshold: f32::MAX,
        persistence_interval: 37,
        narrative_interval: 16,
        history_capacity: 128,
        narrative_capacity: 1,
        rng_seed: Some(seed),
        ..ScriptBotsConfig::default()
    };
    let (mut world, mut session) =
        WorldState::with_persistence(config, Box::new(pipeline.sink())).expect("world");
    for index in 0..12_u16 {
        world
            .try_spawn_agent(AgentData {
                position: Position::new(f32::from(index).mul_add(10.0, 20.0), 20.0),
                health: 2.0,
                ..AgentData::default()
            })
            .expect("spawn founder");
    }
    for tick in 1..=320 {
        if tick == 97 {
            world
                .enqueue_intervention(Intervention::Meteor {
                    region: Region::All,
                    lethality: 10.0,
                    scorch: 0.0,
                })
                .expect("extinction intervention");
        }
        session
            .step(&mut world)
            .expect("production step and persistence admission");
    }
    session
        .finalize(&mut world)
        .expect("finalize partial batch");
    let dropped_events = world.narrative_dropped_events();
    drop(world);
    pipeline.shutdown().expect("durable storage shutdown");
    (path, dropped_events)
}

fn synth_narrative_input_db(dir: &tempfile::TempDir, interval: u32) -> String {
    synth_narrative_input_db_ticks(dir, interval, 4)
}

/// Two actual worlds emit distinct narratives into real fused island persistence batches.
/// The null session sinks acknowledge only after the explicit shared storage admission;
/// they are test orchestration, not a replacement detector or durable owner-journal proof.
fn real_island_narrative_db(
    dir: &tempfile::TempDir,
) -> (String, std::collections::BTreeMap<u32, Vec<EventRecord>>) {
    use scriptbots_core::NullPersistence;
    use scriptbots_core::rng_domains::IslandId;
    use scriptbots_runtime::{
        Archipelago, ArchipelagoConfig, HostCoreOptions, IslandSpec, Topology,
    };

    let path = dir.path().join("island-narrative.sqlite").display().to_string();
    let config = ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        closed: true,
        population_spawn_interval: 0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        aging_health_decay_rate: 0.0,
        spike_damage: 0.0,
        spike_energy_cost: 0.0,
        reproduction_energy_threshold: f32::MAX,
        persistence_interval: 1,
        narrative_interval: 16,
        history_capacity: 128,
        narrative_capacity: 1,
        ..ScriptBotsConfig::default()
    };
    let construction = Archipelago::new(ArchipelagoConfig {
        islands: vec![
            IslandSpec::new(IslandId(0), "early extinction", config.clone()),
            IslandSpec::new(IslandId(7), "late extinction", config),
        ],
        topology: Topology::Ring,
        barrier_interval: std::num::NonZeroU64::MIN,
        master_seed: 0x16_29_05,
        host_options: HostCoreOptions::default(),
        migration: None,
    })
    .expect("island construction metadata");
    let metadata: Vec<_> = construction.islands().cloned().collect();
    let mut storage = Storage::create_unattributed_file(&path).expect("island run file");
    storage.persist_islands(&metadata).expect("persist construction identity");
    let mut worlds = Vec::new();
    for meta in &metadata {
        let (mut world, session) =
            WorldState::with_persistence(meta.effective_config.clone(), Box::new(NullPersistence))
                .expect("actual world and admission session");
        for index in 0..12_u16 {
            world
                .try_spawn_agent(AgentData {
                    position: Position::new(f32::from(index).mul_add(10.0, 20.0), 20.0),
                    health: 2.0,
                    ..AgentData::default()
                })
                .expect("actual founder");
        }
        worlds.push((meta.id, world, session));
    }
    let mut online = std::collections::BTreeMap::<u32, Vec<EventRecord>>::new();
    for tick in 1..=320 {
        let mut batches = Vec::new();
        for (island, world, session) in &mut worlds {
            if tick == if island.0 == 0 { 97 } else { 161 } {
                world
                    .enqueue_intervention(Intervention::Meteor {
                        region: Region::All,
                        lethality: 10.0,
                        scorch: 0.0,
                    })
                    .expect("island-specific extinction");
            }
            let completion = session.step_outcome(world).expect("real scientific transition");
            assert!(completion.fault.is_none());
            let batch = completion
                .outcome
                .persistence
                .into_batch()
                .expect("every-tick projection");
            online.entry(island.0).or_default().extend(batch.narrative_events.clone());
            batches.push((*island, batch));
        }
        let borrowed: Vec<_> = batches.iter().map(|(island, batch)| (*island, batch.as_ref())).collect();
        storage.persist_barrier(&borrowed).expect("real shared outbox admission");
        for (_, world, session) in &mut worlds {
            session.admit_pending(world).expect("acknowledge admitted exact batch");
        }
    }
    for (_, world, _) in worlds {
        assert!(world.narrative_dropped_events() > 0, "display rings must wrap");
    }
    storage.close().expect("durable island shutdown");
    (path, online)
}

fn synth_narrative_input_db_ticks(dir: &tempfile::TempDir, interval: u32, ticks: u64) -> String {
    let path = dir.path().join("inputs.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create input run");
    let config = ScriptBotsConfig {
        narrative_interval: interval,
        history_capacity: 128,
        ..ScriptBotsConfig::default()
    };
    for tick in 1..=ticks {
        let mut tick_batch = batch(tick, 10, 10.0);
        let record = NarrativeInputRecordV1::from_summary(&tick_batch.summary, 0, &config)
            .expect("canonical input");
        tick_batch.replay_events.push(ReplayEvent {
            agent_uid: None,
            position: None,
            counterpart: None,
            counterpart_position: None,
            kind: ReplayEventKind::NarrativeInputV1 { record },
        });
        storage.persist(&tick_batch).expect("persist input");
    }
    storage.close().expect("close input run");
    path
}

/// Keep the independently maintained search projection consistent while changing only
/// saved narrative output. Inputs, scientific rows, and manifests remain untouched.
fn insert_verification_event(
    connection: &Connection,
    run_id: scriptbots_runtime::RunId,
    event: &EventRecord,
) {
    let run_id = run_id.to_string();
    let tick = i64::try_from(event.tick.0).expect("fixture tick");
    let subject = event.subject.map(SubjectRef::to_db_string);
    connection.execute("BEGIN").expect("begin event insertion");
    connection
        .execute_with_params(
            "INSERT INTO run_events (
                run_id, tick, kind, severity, magnitude, window_start, window_end,
                metric, before_value, after_value, score, subject_ref, human_text,
                schema_version, island_id
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, 0)",
            &[
                run_id.as_str().into(),
                tick.into(),
                event.kind.as_str().into(),
                f64::from(event.severity).into(),
                event.magnitude.into(),
                i64::try_from(event.window.0)
                    .expect("fixture window")
                    .into(),
                i64::try_from(event.window.1)
                    .expect("fixture window")
                    .into(),
                event.metric.as_str().into(),
                event.before.into(),
                event.after.into(),
                event.score.into(),
                subject.as_deref().map_or(SqliteValue::Null, Into::into),
                event.human_text.as_str().into(),
                i64::from(event.schema_version).into(),
            ],
        )
        .expect("insert complete saved event");
    connection
        .execute_with_params(
            "INSERT INTO run_events_fts (run_id, tick, kind, metric, human_text, island_id)
             VALUES (?1, ?2, ?3, ?4, ?5, 0)",
            &[
                run_id.as_str().into(),
                tick.into(),
                event.kind.as_str().into(),
                event.metric.as_str().into(),
                event.human_text.as_str().into(),
            ],
        )
        .expect("insert matching search projection");
    connection
        .execute("COMMIT")
        .expect("commit event insertion");
}

fn remove_verification_event(
    connection: &Connection,
    run_id: scriptbots_runtime::RunId,
    event: &EventRecord,
) {
    let run_id = run_id.to_string();
    let parameters = [
        run_id.as_str().into(),
        i64::try_from(event.tick.0).expect("fixture tick").into(),
        event.kind.as_str().into(),
        event.metric.as_str().into(),
    ];
    connection.execute("BEGIN").expect("begin event removal");
    assert_eq!(
        connection
            .execute_with_params(
                "DELETE FROM run_events_fts WHERE run_id = ?1 AND CAST(tick AS INTEGER) = ?2
                 AND kind = ?3 AND metric = ?4 AND CAST(island_id AS INTEGER) = 0",
                &parameters,
            )
            .expect("remove exact search projection"),
        1
    );
    assert_eq!(
        connection
            .execute_with_params(
                "DELETE FROM run_events WHERE run_id = ?1 AND tick = ?2
                 AND kind = ?3 AND metric = ?4 AND island_id = 0",
                &parameters,
            )
            .expect("remove exact saved event"),
        1
    );
    connection.execute("COMMIT").expect("commit event removal");
}

#[test]
fn run_summary_reports_trajectory_and_ledger() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_db(&dir);

    let cx = ReaderCtx::open(&path).expect("open reader");
    let out = Registry::builtin()
        .run("run-summary", &cx, &ReportParams::default())
        .expect("run-summary");

    assert_eq!(out.schema_version, REPORT_SCHEMA_VERSION);
    assert_eq!(out.report, "run-summary");
    assert_eq!(out.latest_tick, Some(3));
    assert_eq!(out.row_count, 3);

    let m = &out.machine;
    assert_eq!(m["tick_count"], 3, "three ticks persisted: {m}");
    assert_eq!(
        m["population_first"], 10,
        "chronological order, not reader order: {m}"
    );
    assert_eq!(m["population_last"], 8);
    assert_eq!(m["population_min"], 8);
    assert_eq!(m["population_max"], 12);
    assert_eq!(m["total_energy_first"], 100.0);
    assert_eq!(m["total_energy_last"], 90.0);
    assert!(
        out.human_md.contains("Run summary"),
        "markdown rendered: {}",
        out.human_md
    );

    let serialized = serde_json::to_value(&out).expect("serialize stable report envelope");
    assert_eq!(
        serialized,
        json!({
            "schema_version": 1,
            "report": "run-summary",
            "db_path": path,
            "latest_tick": 3,
            "row_count": 3,
            "machine": {
                "tick_count": 3,
                "birth_records": 0,
                "death_records": 0,
                "population_first": 10,
                "population_last": 8,
                "population_min": 8,
                "population_max": 12,
                "population_mean": 10.0,
                "total_energy_first": 100.0,
                "total_energy_last": 90.0,
                "watermarks": {
                    "admitted": 3,
                    "applied": 3,
                    "durable": 3
                }
            }
        }),
        "machine-readable schema changed without a version bump"
    );
}

#[test]
fn replay_timeline_handles_empty_stream_and_limit_param() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_db(&dir);

    let cx = ReaderCtx::open(&path).expect("open reader");
    let registry = Registry::builtin();

    let out = registry
        .run("replay-timeline", &cx, &ReportParams::default())
        .expect("timeline");
    assert_eq!(out.schema_version, REPLAY_TIMELINE_SCHEMA_VERSION);
    assert_eq!(out.row_count, 0);
    assert_eq!(out.machine["events"].as_array().map(Vec::len), Some(0));
    assert!(
        out.human_md.contains("No replay events"),
        "{}",
        out.human_md
    );
    assert_eq!(
        serde_json::to_value(&out).expect("serialize timeline envelope"),
        json!({
            "schema_version": 2,
            "report": "replay-timeline",
            "db_path": path,
            "latest_tick": 3,
            "row_count": 0,
            "machine": {
                "run_id": cx.reader.run_id().to_string(),
                "event_counts": [],
                "events": [],
                "truncated_to": null
            }
        }),
        "timeline schema changed without a version bump"
    );

    // limit param parses; bad param is a typed error, not a panic.
    let params = ReportParams::from_pairs(["limit=5".to_owned()]).expect("params");
    registry
        .run("replay-timeline", &cx, &params)
        .expect("timeline with limit");
    let bad = ReportParams::from_pairs(["limit=banana".to_owned()])
        .and_then(|p| registry.run("replay-timeline", &cx, &p));
    assert!(
        matches!(bad, Err(AnalyticsError::BadParam { .. })),
        "typed error: {bad:?}"
    );
    assert!(matches!(
        ReportParams::from_pairs(["limit=1".to_owned(), "limit=2".to_owned()]),
        Err(AnalyticsError::BadParam { .. })
    ));
    assert!(matches!(
        ReportParams::from_pairs(["=value".to_owned()]),
        Err(AnalyticsError::BadParam { .. })
    ));
}

#[test]
fn replay_timeline_orders_serializes_and_limits_replay_events() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_db_with_events(&dir, true, true);
    let cx = ReaderCtx::open(&path).expect("open reader");
    let registry = Registry::builtin();

    let out = registry
        .run("replay-timeline", &cx, &ReportParams::default())
        .expect("full timeline");
    assert_eq!(out.row_count, 2);
    assert_eq!(
        out.machine["event_counts"],
        json!([["brain_outputs", 1], ["rng_sample", 1]])
    );
    assert_eq!(out.machine["events"][0]["tick"], 1);
    assert_eq!(out.machine["events"][0]["island_id"], 0);
    assert_eq!(out.machine["events"][0]["seq"], 0);
    assert_eq!(
        out.machine["events"][0]["event"],
        // The three positional keys are asserted EXPLICITLY, not tolerated. `ReplayEvent`
        // gained `position`, `counterpart` and `counterpart_position`, and this report
        // serializes the whole event, so they are now part of its published JSON shape --
        // an offline consumer parsing this report will see them. Pinning them as null here
        // is the assertion that the timeline reports what the event actually carries: if a
        // pairwise kind ever populates them and this report silently kept emitting null,
        // that is a drop this test now catches rather than a shape it happens to ignore.
        json!({
            "agent_uid": null,
            "position": null,
            "counterpart": null,
            "counterpart_position": null,
            "kind": {
                "RngSample": {
                    "scope": "World",
                    "range_min": 0.0,
                    "range_max": 1.0,
                    "value": 0.25
                }
            }
        })
    );
    assert_eq!(out.machine["events"][1]["tick"], 3);
    assert_eq!(out.machine["events"][1]["island_id"], 0);
    assert_eq!(
        out.machine["events"][1]["event"],
        json!({
            "agent_uid": null,
            "position": null,
            "counterpart": null,
            "counterpart_position": null,
            "kind": {"BrainOutputs": {"outputs": [0.5, -0.25]}}
        })
    );

    let limited = registry
        .run(
            "replay-timeline",
            &cx,
            &ReportParams::from_pairs(["limit=1".to_owned()]).expect("limit param"),
        )
        .expect("limited timeline");
    assert_eq!(limited.row_count, 1);
    assert_eq!(limited.machine["events"].as_array().map(Vec::len), Some(1));
    assert_eq!(
        limited.machine["events"][0]["tick"], 3,
        "a bounded timeline page must retain the newest event, not the oldest prefix"
    );
    assert_eq!(limited.machine["truncated_to"], 1);
}

#[test]
fn replay_timeline_preserves_colliding_island_identities_in_bounded_pages() {
    use scriptbots_core::rng_domains::IslandId;

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir
        .path()
        .join("island-replay.sqlite")
        .display()
        .to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create island run");
    for tick in [1, 2] {
        let mut island_batch = batch(tick, 1, 1.0);
        island_batch.replay_events.push(ReplayEvent {
            agent_uid: Some(AgentUid(7)),
            position: None,
            counterpart: None,
            counterpart_position: None,
            kind: ReplayEventKind::BrainOutputs {
                outputs: vec![0.25],
            },
        });
        // Both worlds deliberately emit identical tick/sequence/agent/payload
        // values. Only their retained island identity can distinguish them.
        storage
            .persist_barrier(&[(IslandId(7), &island_batch), (IslandId(0), &island_batch)])
            .expect("persist the actual multi-island barrier");
    }
    storage.flush().expect("flush island run");
    storage.close().expect("close island run");

    let cx = ReaderCtx::open(&path).expect("cold reader");
    let registry = Registry::builtin();
    let report = registry
        .run("replay-timeline", &cx, &ReportParams::default())
        .expect("complete island timeline");
    let keys = |output: &scriptbots_analytics::ReportOutput| {
        output.machine["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                (
                    row["tick"].as_u64().unwrap(),
                    row["island_id"].as_u64().unwrap(),
                    row["seq"].as_u64().unwrap(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(report.schema_version, REPLAY_TIMELINE_SCHEMA_VERSION);
    assert_eq!(report.machine["run_id"], cx.reader.run_id().to_string());
    assert_eq!(keys(&report), [(1, 0, 0), (1, 7, 0), (2, 0, 0), (2, 7, 0)]);
    assert_eq!(
        report.machine["event_counts"],
        json!([["brain_outputs", 4]])
    );
    assert!(report.human_md.contains("| tick | island | seq | event |"));
    assert!(report.human_md.contains("| 2 | 7 | 0 |"));
    for limit in [0, 1, 3] {
        let limited = registry
            .run(
                "replay-timeline",
                &cx,
                &ReportParams::from_pairs([format!("limit={limit}")]).unwrap(),
            )
            .expect("bounded island timeline");
        let expected = keys(&report);
        assert_eq!(keys(&limited), expected[expected.len() - limit..]);
        assert_eq!(
            limited.machine["event_counts"],
            report.machine["event_counts"]
        );
        assert_eq!(limited.machine["truncated_to"], limit);
    }
}

#[test]
fn narrative_timeline_preserves_complete_typed_evidence_and_canonical_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_db_with_events(&dir, true, true);
    let cx = ReaderCtx::open(&path).expect("open reader");
    let registry = Registry::builtin();
    let output = registry
        .run("narrative-timeline", &cx, &ReportParams::default())
        .expect("typed narrative timeline");
    assert_eq!(output.schema_version, NARRATIVE_TIMELINE_SCHEMA_VERSION);
    assert_eq!(output.row_count, 3, "replay records are a separate stream");
    assert_eq!(output.machine["run_id"], cx.reader.run_id().to_string());
    assert_eq!(output.machine["event_counts_scope"], "returned_page");
    assert_eq!(
        output.machine["event_counts"],
        json!([["energy_recovery", 1], ["population_boom", 2]])
    );
    let expected = [
        narrative_event(1, EventKind::PopulationBoom, "population"),
        narrative_event(3, EventKind::EnergyRecovery, "average_energy"),
        narrative_event(3, EventKind::PopulationBoom, "population"),
    ];
    for (row, record) in output.machine["events"]
        .as_array()
        .expect("event array")
        .iter()
        .zip(&expected)
    {
        assert_eq!(row["record"], serde_json::to_value(record).expect("record"));
        assert_eq!(
            row["identity"],
            json!({
                "tick": record.tick.0,
                "kind": record.kind.as_str(),
                "metric": record.metric,
                "island": 0
            })
        );
    }
    assert_eq!(output.machine["truncated_to"], serde_json::Value::Null);
    assert!(
        output
            .human_md
            .contains("recovered \\| after &lt;rain&gt; again")
    );

    let limited = registry
        .run(
            "narrative-timeline",
            &cx,
            &ReportParams::from_pairs(["limit=1".to_owned()]).expect("params"),
        )
        .expect("limited narrative");
    assert_eq!(limited.row_count, 1);
    assert_eq!(limited.machine["events"][0]["identity"]["tick"], 3);
    assert_eq!(
        limited.machine["events"][0]["identity"]["kind"],
        "population_boom"
    );
    assert_eq!(
        limited.machine["event_counts"],
        json!([["population_boom", 1]])
    );
    assert_eq!(limited.machine["truncated_to"], 1);

    let exact = registry
        .run(
            "narrative-timeline",
            &cx,
            &ReportParams::from_pairs(["limit=3".to_owned()]).expect("params"),
        )
        .expect("exact boundary");
    assert_eq!(exact.row_count, 3);
    assert!(exact.machine["truncated_to"].is_null());

    let zero = registry
        .run(
            "narrative-timeline",
            &cx,
            &ReportParams::from_pairs(["limit=0".to_owned()]).expect("params"),
        )
        .expect("zero requested");
    assert_eq!(zero.row_count, 0);
    assert_eq!(zero.machine["truncated_to"], 0);
    assert!(zero.human_md.contains("events are present"));
}

#[test]
fn narrative_timeline_distinguishes_empty_evidence_and_rejects_bad_limits() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_db_with_events(&dir, true, false);
    let cx = ReaderCtx::open(&path).expect("open reader");
    let registry = Registry::builtin();
    let output = registry
        .run("narrative-timeline", &cx, &ReportParams::default())
        .expect("empty narrative stream");
    assert_eq!(
        serde_json::to_value(&output).expect("schema"),
        json!({
            "schema_version": 2,
            "report": "narrative-timeline",
            "db_path": path,
            "latest_tick": 3,
            "row_count": 0,
            "machine": {
                "run_id": cx.reader.run_id().to_string(),
                "event_counts_scope": "returned_page",
                "event_counts": [],
                "events": [],
                "truncated_to": null
            }
        })
    );
    assert!(output.human_md.contains("No narrative events"));
    for invalid in [
        "limit=4096",
        "limit=18446744073709551615",
        "limit=banana",
        "limt=1",
    ] {
        let params = ReportParams::from_pairs([invalid.to_owned()]).expect("parameter pair");
        assert!(
            matches!(
                registry.run("narrative-timeline", &cx, &params),
                Err(AnalyticsError::BadParam { .. })
            ),
            "invalid parameter was accepted: {invalid}"
        );
    }
}

#[test]
fn narrative_reconstruct_matches_online_events_across_pages_and_output_limits() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (path, dropped_events) = real_narrative_db(&dir);
    let cx = ReaderCtx::open(&path).expect("finished reader");
    let expected: Vec<_> = cx
        .reader
        .recent_run_events(4096)
        .expect("persisted online events")
        .into_iter()
        .map(|event| event.into_parts().1)
        .collect();
    assert!(
        expected
            .iter()
            .any(|event| event.kind == EventKind::Extinction)
    );
    assert!(
        expected.len() > 1,
        "nonempty parity must outlive the one-event ring"
    );
    assert!(dropped_events > 0, "the production event ring must wrap");

    let registry = Registry::builtin();
    let full = registry
        .run("narrative-reconstruct", &cx, &ReportParams::default())
        .expect("independent reconstruction");
    assert_eq!(full.schema_version, NARRATIVE_RECONSTRUCT_SCHEMA_VERSION);
    assert_eq!(full.latest_tick, Some(320));
    assert_eq!(full.row_count, expected.len());
    assert_eq!(
        full.machine["events"],
        serde_json::to_value(&expected).unwrap()
    );
    assert_eq!(full.machine["inputs_replayed"], 320);
    assert_eq!(full.machine["input_pages"], 2);
    assert_eq!(full.machine["input_binding"]["input_count"], 320);
    assert_eq!(full.machine["policy_changes"], 0);
    assert_eq!(
        full.machine["final_policy"],
        full.machine["input_binding"]["policy"]
    );
    assert_eq!(
        full.machine["input_binding"]["policy"]["history_capacity"],
        128
    );
    assert_eq!(full.machine["history_inputs_evicted"], 192);
    assert_eq!(full.machine["detector_events_evicted"], dropped_events);
    assert_eq!(full.machine["event_counts_scope"], "complete_run");
    assert_eq!(full.machine["truncated_to"], serde_json::Value::Null);

    for limit in [1, 0] {
        let params = ReportParams::from_pairs([format!("limit={limit}")]).expect("limit");
        let limited = registry
            .run("narrative-reconstruct", &cx, &params)
            .expect("limited report");
        assert_eq!(limited.row_count, limit);
        assert_eq!(
            limited.machine["inputs_replayed"],
            full.machine["inputs_replayed"]
        );
        assert_eq!(limited.machine["input_pages"], full.machine["input_pages"]);
        assert_eq!(limited.machine["events_reconstructed"], expected.len());
        assert_eq!(
            limited.machine["event_counts"],
            full.machine["event_counts"]
        );
        assert_eq!(limited.machine["truncated_to"], limit);
        assert_eq!(
            limited.machine["events"],
            serde_json::to_value(&expected[expected.len() - limit..]).unwrap()
        );
    }

    drop(cx);
    let connection = Connection::open_existing(path.clone()).expect("open negative control");
    connection.execute("BEGIN").expect("begin output removal");
    // The finished reader requires the independently maintained FTS mirror to
    // match relational output rows. Remove both while leaving all inputs intact.
    connection
        .execute("DELETE FROM run_events_fts")
        .expect("remove indexed output rows");
    assert!(
        connection
            .execute("DELETE FROM run_events")
            .expect("remove only output rows")
            > 0
    );
    connection.execute("COMMIT").expect("commit output removal");
    connection.close().expect("close negative control");
    let cx = ReaderCtx::open(&path).expect("inputs remain complete");
    let independent = registry
        .run("narrative-reconstruct", &cx, &ReportParams::default())
        .expect("reconstruct without saved event outputs");
    assert_eq!(
        independent.machine, full.machine,
        "output rows cannot supply reconstruction"
    );
}

/// Exercise the actual owner, durable command authority and journal adapter.
/// Retry an admission with its original identity; never turn a retry into a
/// second step. A completed application alone is not a durable observation.
fn narrative_host_command(
    core: &mut scriptbots_runtime::HostCore,
    command: scriptbots_runtime::HostCommand,
    sequence: &mut u64,
    now: &mut u64,
) {
    use scriptbots_runtime::{
        ApplicationState, CommandAuthorityLookupFailure, CommandEnvelope, CommandId,
        HostAccessError, HostPort, JournalState, ManualHostDriver, ManualInstant,
    };
    let envelope =
        CommandEnvelope::new(CommandId::from_client_sequence(0x16_29, *sequence), command);
    *sequence += 1;
    let mut port = core.local_port();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut admitted = false;
    loop {
        let observation = if admitted {
            port.command_status(envelope.command_id)
        } else {
            port.submit(envelope.clone()).map(Some)
        };
        match observation {
            Ok(Some(status)) => {
                admitted = true;
                assert!(
                    matches!(
                        status.application(),
                        ApplicationState::Admitted | ApplicationState::Applied(_)
                    ),
                    "owner refused the actual command: {status:?}"
                );
                assert!(
                    !matches!(status.journal(), JournalState::Failed(_)),
                    "{status:?}"
                );
                if status.journal() == &JournalState::Durable {
                    assert!(matches!(status.application(), ApplicationState::Applied(_)));
                    return;
                }
            }
            Err(HostAccessError::CommandAuthorityLookup {
                failure:
                    CommandAuthorityLookupFailure::Pending
                    | CommandAuthorityLookupFailure::Busy
                    | CommandAuthorityLookupFailure::Capacity { .. },
                ..
            }) => {}
            other => panic!("unexpected command observation: {other:?}"),
        }
        assert!(
            std::time::Instant::now() < deadline,
            "durable command timed out"
        );
        core.drive(ManualInstant::from_nanos(*now))
            .expect("drive actual owner");
        *now += 1;
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "complete owner-journal fixture covers policy transitions, persisted evidence, paging and bounded output"
)]
fn narrative_reconstruct_applies_journal_authenticated_policy_changes_across_pages() {
    use scriptbots_runtime::{
        HostCommand, HostCore, HostCoreOptions, HostSessionId, PlaybackSnapshot,
        ScheduledConfigPatch,
    };
    use scriptbots_storage::StorageJournalOptions;

    fn resolve(
        config: &ScriptBotsConfig,
        patch: &serde_json::Value,
    ) -> Result<ScriptBotsConfig, String> {
        let mut value = serde_json::to_value(config).map_err(|error| error.to_string())?;
        let object = value.as_object_mut().ok_or("config object required")?;
        for (key, field) in patch.as_object().ok_or("patch object required")? {
            object.insert(key.clone(), field.clone());
        }
        serde_json::from_value(value).map_err(|error| error.to_string())
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir
        .path()
        .join("changing-policy.sqlite")
        .display()
        .to_string();
    let mut pipeline = StoragePipeline::create_unattributed_file(&path).expect("file run");
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        closed: true,
        population_spawn_interval: 0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        aging_health_decay_rate: 0.0,
        spike_damage: 0.0,
        spike_energy_cost: 0.0,
        reproduction_energy_threshold: f32::MAX,
        persistence_interval: 37,
        narrative_interval: 16,
        history_capacity: 128,
        narrative_capacity: 8,
        rng_seed: Some(0x16_020_902),
        ..ScriptBotsConfig::default()
    })
    .expect("world");
    for index in 0..12_u16 {
        world
            .try_spawn_agent(AgentData {
                position: Position::new(f32::from(index).mul_add(10.0, 20.0), 20.0),
                health: 2.0,
                ..AgentData::default()
            })
            .expect("founder");
    }
    let session = HostSessionId::new(0x16_29_02);
    let journal = pipeline
        .journal_port(session, StorageJournalOptions::default())
        .expect("journal");
    let mut core = HostCore::with_journal(
        session,
        world,
        HostCoreOptions {
            initial_playback: PlaybackSnapshot {
                paused: true,
                speed_multiplier: 1.0,
            },
            ..HostCoreOptions::default()
        },
        Box::new(journal),
    )
    .expect("owner");
    core.install_schedule(
        vec![
            ScheduledConfigPatch::new(1, Tick(64), json!({"narrative_interval": 8})).unwrap(),
            // A real ecological change with an unchanged narrative policy must not
            // be counted as a detector-policy transition.
            ScheduledConfigPatch::new(
                2,
                Tick(96),
                json!({
                    "aging_health_decay_start": 0,
                    "aging_health_decay_rate": 10.0,
                    "aging_health_decay_max": 10.0
                }),
            )
            .unwrap(),
            ScheduledConfigPatch::new(
                3,
                Tick(128),
                json!({
                    "history_capacity": 64, "narrative_capacity": 1
                }),
            )
            .unwrap(),
            ScheduledConfigPatch::new(
                4,
                Tick(192),
                json!({
                    "narrative_interval": 0, "history_capacity": 32, "narrative_capacity": 0
                }),
            )
            .unwrap(),
            ScheduledConfigPatch::new(
                5,
                Tick(256),
                json!({
                    "narrative_interval": 16, "history_capacity": 128, "narrative_capacity": 4
                }),
            )
            .unwrap(),
        ],
        resolve,
    )
    .expect("authenticated owner schedule");
    let mut sequence = 1;
    let mut now = 0;
    for tick in 1..=320 {
        narrative_host_command(&mut core, HostCommand::Step, &mut sequence, &mut now);
        assert_eq!(core.world_tick(), Tick(tick));
        assert!(core.world().history().count() <= core.world().config().history_capacity);
        assert!(core.world().narrative_events().len() <= core.world().config().narrative_capacity);
    }
    let dropped = core.world().narrative_dropped_events();
    assert!(
        dropped > 0,
        "capacity changes must actually discard display events"
    );
    narrative_host_command(&mut core, HostCommand::Shutdown, &mut sequence, &mut now);
    drop(core);
    pipeline.shutdown().expect("durable shutdown");

    let cx = ReaderCtx::open(&path).expect("finished reader");
    let expected: Vec<_> = cx
        .reader
        .recent_run_events(4096)
        .unwrap()
        .into_iter()
        .map(|event| event.into_parts().1)
        .collect();
    assert!(
        expected
            .iter()
            .any(|event| event.kind == EventKind::Extinction)
    );
    assert!(
        expected.len() > 1,
        "comparison must include events discarded by the display ring"
    );
    let registry = Registry::builtin();
    let full = registry
        .run("narrative-reconstruct", &cx, &ReportParams::default())
        .expect("reconstruct authenticated live policy changes");
    assert_eq!(full.schema_version, NARRATIVE_RECONSTRUCT_SCHEMA_VERSION);
    assert_eq!(full.machine["inputs_replayed"], 320);
    assert_eq!(full.machine["input_pages"], 2);
    assert_eq!(full.machine["policy_changes"], 4);
    assert_eq!(
        full.machine["final_policy"],
        json!({
            "narrative_interval": 16, "history_capacity": 128, "event_capacity": 4
        })
    );
    assert_eq!(full.machine["detector_events_evicted"], dropped);
    assert_eq!(
        full.machine["events"],
        serde_json::to_value(&expected).unwrap()
    );
    let limited = registry
        .run(
            "narrative-reconstruct",
            &cx,
            &ReportParams::from_pairs(["limit=0".to_owned()]).unwrap(),
        )
        .unwrap();
    assert_eq!(limited.row_count, 0);
    assert_eq!(limited.machine["policy_changes"], 4);
    assert_eq!(limited.machine["events_reconstructed"], expected.len());
    assert_eq!(
        limited.machine["event_counts"],
        full.machine["event_counts"]
    );
}

#[test]
fn narrative_reconstruct_distinguishes_quiet_inputs_from_missing_or_corrupt_evidence() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_narrative_input_db(&dir, 0);
    let registry = Registry::builtin();
    let cx = ReaderCtx::open(&path).expect("finished reader");
    let output = registry
        .run("narrative-reconstruct", &cx, &ReportParams::default())
        .expect("disabled detector with a complete short stream");
    assert_eq!(output.row_count, 0);
    assert_eq!(output.machine["inputs_replayed"], 4);
    assert_eq!(output.machine["events_reconstructed"], 0);
    assert_eq!(
        output.machine["input_binding"]["policy"]["narrative_interval"],
        0
    );
    assert!(output.human_md.contains("complete recorded input stream"));
    drop(cx);

    let connection = Connection::open_existing(path.clone()).expect("open corruption control");
    assert_eq!(
        connection
            .execute(
                "DELETE FROM replay_events WHERE event_type = 'narrative_input_v1' AND tick = 3"
            )
            .expect("remove one detector input"),
        1
    );
    connection.close().expect("close corruption control");
    let cx = ReaderCtx::open(&path).expect("open finished corrupt stream");
    let params = ReportParams::from_pairs(["limit=0".to_owned()]).expect("zero output limit");
    assert!(matches!(
        registry.run("narrative-reconstruct", &cx, &params),
        Err(AnalyticsError::Storage(StorageError::NarrativeInputStream(
            NarrativeInputStreamError::TickGap {
                expected: 3,
                actual: 4,
                ..
            }
        )))
    ));

    let missing_dir = tempfile::tempdir().expect("tempdir");
    let missing = ReaderCtx::open(&synth_db(&missing_dir)).expect("legacy run without inputs");
    assert!(matches!(
        registry.run("narrative-reconstruct", &missing, &params),
        Err(AnalyticsError::Storage(StorageError::NarrativeInputStream(
            NarrativeInputStreamError::MissingEvidence {
                first_offending_tick: 1
            }
        )))
    ));
    for invalid in ["limit=4097", "limit=banana", "limt=1"] {
        let params = ReportParams::from_pairs([invalid.to_owned()]).expect("parameter pair");
        assert!(matches!(
            registry.run("narrative-reconstruct", &missing, &params),
            Err(AnalyticsError::BadParam { .. })
        ));
    }
}

#[test]
fn narrative_verification_matches_three_real_scoped_runs_after_display_ring_wrap() {
    // These are bounded 320-tick integration controls, not the scenario budget suite.
    for seed in [0x16_020_901, 0x16_020_902, 0x16_020_903] {
        let dir = tempfile::tempdir().expect("tempdir");
        let (path, dropped_events) = real_narrative_db_seeded(&dir, seed);
        let cx = ReaderCtx::open(&path).expect("finished real run");
        let saved = cx
            .reader
            .run_event_page_v1(None, 4096)
            .expect("complete event page");
        assert!(
            saved.next_after.is_none(),
            "scoped fixture must fit one event page"
        );
        assert!(saved.events.len() > 1);
        assert!(
            saved
                .events
                .iter()
                .any(|event| event.record().kind == EventKind::Extinction)
        );
        assert!(dropped_events > 0, "seed {seed}: display ring must wrap");

        let report = verify_narrative(&cx.reader).expect("complete independent comparison");
        report
            .require_equivalent()
            .expect("saved events equal reconstruction");
        assert_eq!(report.schema_version, NARRATIVE_VERIFICATION_SCHEMA_VERSION);
        assert!(report.equivalent);
        assert!(report.first_difference.is_none());
        assert_eq!(report.inputs_replayed, 320);
        assert_eq!(report.input_pages, 2);
        assert_eq!(report.input_binding.terminal_tick, 320);
        assert_eq!(report.input_binding.policy.event_capacity, 1);
        assert_eq!(report.events_reconstructed, saved.events.len() as u64);
        assert_eq!(report.events_persisted, report.events_reconstructed);
        assert_eq!(report.events_matched, report.events_persisted);
        assert_eq!(
            (
                report.events_missing,
                report.events_unexpected,
                report.events_changed
            ),
            (0, 0, 0)
        );
        assert_eq!(report.reconstructed_digest, report.persisted_digest);
        assert_eq!(report.input_digest.len(), 64);
        let repeated = verify_narrative(&cx.reader).expect("repeat on the immutable source");
        assert_eq!(
            report, repeated,
            "seed {seed}: verdict and digests must be deterministic"
        );
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one immutable two-world recording proves selected reconstruction, complete comparison, CLI output, and cross-island negative controls"
)]
fn island_narrative_reconstruction_and_verification_follow_independent_real_worlds() {
    use scriptbots_analytics::{
        ArchipelagoNarrativeVerificationV1, IslandNarrativeVerificationV1,
        verify_archipelago_narrative,
    };
    use scriptbots_core::rng_domains::IslandId;
    use scriptbots_storage::RunEventIdentity;

    let dir = tempfile::tempdir().expect("tempdir");
    let (path, mut online) = real_island_narrative_db(&dir);
    let cx = ReaderCtx::open(&path).expect("finished island recording");
    assert!(matches!(
        verify_narrative(&cx.reader),
        Err(AnalyticsError::NarrativeVerification(
            NarrativeVerificationError::UnsupportedIsland { island: 7 }
        ))
    ));
    let registry = Registry::builtin();
    assert!(registry.run("narrative-reconstruct", &cx, &ReportParams::default()).is_err());
    let all = verify_archipelago_narrative(&cx.reader).expect("verify the full archipelago");
    all.require_equivalent().expect("both independent detectors matched");
    assert_eq!(
        all.islands.iter().map(|i| i.input_binding.island_id).collect::<Vec<_>>(),
        [IslandId(0), IslandId(7)]
    );
    let mut input_digests = Vec::new();
    for id in [0, 7] {
        let expected = online.get_mut(&id).expect("online detector records");
        expected.sort_by_key(RunEventIdentity::from_record);
        assert!(expected.len() > 1, "complete output must exceed the one-event display ring");
        assert!(expected.iter().any(|event| event.kind == EventKind::Extinction));
        let report = verify_island_narrative(&cx.reader, IslandId(id))
            .expect("complete selected-island comparison");
        let member = all.islands.iter().find(|i| i.input_binding.island_id == IslandId(id)).unwrap();
        assert_eq!(member.verification.input_digest, report.verification.input_digest);
        assert_eq!(member.verification.persisted_digest, report.verification.persisted_digest);
        assert_eq!(member.verification.events_matched, report.verification.events_matched);
        report.require_equivalent().expect("independent reconstruction matches live output");
        assert_eq!(report.input_binding.island_id, IslandId(id));
        assert_eq!(report.input_binding.stream.terminal_tick, 320);
        assert_eq!(report.verification.inputs_replayed, 320);
        assert_eq!(report.verification.input_pages, 2);
        assert_eq!(report.verification.events_persisted, expected.len() as u64);
        assert_eq!(report.verification.events_reconstructed, expected.len() as u64);
        input_digests.push(report.verification.input_digest.clone());
        for limit in [4096, 1, 0] {
            let params = ReportParams::from_pairs([
                format!("island_id={id}"),
                format!("limit={limit}"),
            ])
            .expect("island reconstruction parameters");
            let reconstructed = registry.run("narrative-reconstruct", &cx, &params)
                .expect("selected reconstruction");
            assert_eq!(reconstructed.machine["island_input_binding"]["island_id"], id);
            assert_eq!(reconstructed.machine["inputs_replayed"], 320);
            assert_eq!(reconstructed.machine["event_counts_scope"], "complete_island");
            assert_eq!(reconstructed.machine["events_reconstructed"], expected.len());
            let retained = limit.min(expected.len());
            assert_eq!(
                reconstructed.machine["events"],
                serde_json::to_value(&expected[expected.len() - retained..]).unwrap()
            );
        }
    }
    assert_ne!(input_digests[0], input_digests[1], "islands must have distinct histories");
    for invalid in ["island_id=4294967296", "island_id=-1", "island_id=oops"] {
        let params = ReportParams::from_pairs([invalid.to_owned()]).expect("parameter pair");
        assert!(matches!(
            registry.run("narrative-reconstruct", &cx, &params),
            Err(AnalyticsError::BadParam { .. })
        ));
    }
    assert!(matches!(
        verify_island_narrative(&cx.reader, IslandId(99)),
        Err(AnalyticsError::Storage(StorageError::NarrativeInputStream(
            NarrativeInputStreamError::IslandUnavailable { .. }
        )))
    ));
    drop(cx);

    let executable = env!("CARGO_BIN_EXE_sb-analyze");
    let json_path = dir.path().join("island-equivalent.json");
    let md_path = dir.path().join("island-equivalent.md");
    let command = Command::new(executable)
        .args([&path, "verify-narrative", "--island-id", "7", "--json"])
        .arg(&json_path)
        .arg("--md")
        .arg(&md_path)
        .output()
        .expect("selected-island CLI");
    assert!(command.status.success(), "{}", String::from_utf8_lossy(&command.stderr));
    let cli_report: IslandNarrativeVerificationV1 =
        serde_json::from_slice(&std::fs::read(&json_path).expect("scoped JSON"))
            .expect("scoped result envelope");
    cli_report.require_equivalent().expect("CLI exact island comparison");
    assert_eq!(cli_report.input_binding.island_id, IslandId(7));
    assert_eq!(std::fs::read_to_string(&md_path).unwrap(), cli_report.markdown());

    let connection = Connection::open_existing(&path).expect("negative output control");
    assert!(connection
        .execute("UPDATE run_events SET magnitude = magnitude + 1.0 WHERE island_id = 7")
        .expect("change only island seven outputs") > 0);
    connection.close().expect("close changed output");
    let cx = ReaderCtx::open(&path).expect("typed changed narrative evidence");
    verify_island_narrative(&cx.reader, IslandId(0)).expect("unmodified island").require_equivalent()
        .expect("another island's changed events cannot contaminate this comparison");
    let changed = verify_island_narrative(&cx.reader, IslandId(7)).expect("complete changed comparison");
    assert!(!changed.verification.equivalent);
    let difference = changed.verification.first_difference.as_ref().expect("typed first difference");
    assert_eq!(difference.identity.island(), IslandId(7));
    assert_eq!(difference.kind, NarrativeDifferenceKind::ChangedRecord);
    assert_eq!(difference.field.as_deref(), Some("magnitude"));
    drop(cx);
    let changed_path = dir.path().join("island-diverged.json");
    let command = Command::new(executable)
        .args([&path, "verify-narrative", "--island-id", "7", "--json"])
        .arg(&changed_path)
        .output()
        .expect("divergent selected-island CLI");
    assert_eq!(command.status.code(), Some(1));
    let retained: IslandNarrativeVerificationV1 =
        serde_json::from_slice(&std::fs::read(changed_path).expect("report retained before failure"))
            .expect("retained scoped divergence");
    assert_eq!(retained, changed);

    let all_path = dir.path().join("archipelago-diverged.json");
    let cli = Command::new(executable)
        .args([&path, "verify-narrative", "--all-islands", "--json"])
        .arg(&all_path)
        .output()
        .expect("complete archipelago CLI");
    assert_eq!(cli.status.code(), Some(1));
    let all: ArchipelagoNarrativeVerificationV1 =
        serde_json::from_slice(&std::fs::read(all_path).unwrap()).unwrap();
    assert!(!all.equivalent);
    all.islands[0].require_equivalent().expect("island zero still matches");
    assert_eq!(all.islands[1].verification.first_difference, changed.verification.first_difference);
    assert!(matches!(
        all.require_equivalent(),
        Err(NarrativeVerificationError::Diverged { identity, .. }) if identity.island() == IslandId(7)
    ));

    let connection = Connection::open_existing(&path).expect("negative input control");
    assert_eq!(connection
        .execute("DELETE FROM replay_events WHERE island_id = 7 AND tick = 3 AND event_type = 'narrative_input_v1'")
        .expect("remove one selected input"), 1);
    connection.close().expect("close missing input");
    let cx = ReaderCtx::open(&path).expect("finished structurally valid source");
    verify_island_narrative(&cx.reader, IslandId(0)).expect("healthy independent stream").require_equivalent()
        .expect("unmodified island remains complete");
    assert!(matches!(
        verify_archipelago_narrative(&cx.reader),
        Err(AnalyticsError::IslandEvidence { island: 7, source })
            if matches!(*source, AnalyticsError::Storage(StorageError::NarrativeInputStream(
                NarrativeInputStreamError::TickGap { expected: 3, actual: 4, .. }
            )))
    ));
    assert!(matches!(
        verify_island_narrative(&cx.reader, IslandId(7)),
        Err(AnalyticsError::Storage(StorageError::NarrativeInputStream(
            NarrativeInputStreamError::TickGap { expected: 3, actual: 4, .. }
        )))
    ));
    drop(cx);

    // Selection must not conceal malformed persisted records encountered while
    // traversing the complete saved-event stream.
    let connection = Connection::open_existing(&path).expect("corrupt other-island output");
    assert!(connection
        .execute("UPDATE run_events SET schema_version = 999 WHERE island_id = 7")
        .expect("invalidate other-island event schema") > 0);
    connection.close().expect("close corrupt event control");
    let cx = ReaderCtx::open(&path).expect("finished typed-event source");
    assert!(matches!(
        verify_island_narrative(&cx.reader, IslandId(0)),
        Err(AnalyticsError::Storage(StorageError::RunEvent(
            RunEventDecodeError::UnsupportedSchemaVersion { found: 999, .. }
        )))
    ));
    drop(cx);
    let connection = Connection::open_existing(&path).expect("orphan an island");
    assert_eq!(connection.execute("DELETE FROM islands WHERE island_id = 7").unwrap(), 1);
    connection.close().unwrap();
    let cx = ReaderCtx::open(&path).unwrap();
    assert!(matches!(
        verify_archipelago_narrative(&cx.reader),
        Err(AnalyticsError::Storage(StorageError::NarrativeInputStream(
            NarrativeInputStreamError::IslandUnavailable { island: IslandId(7), .. }
        )))
    ));
}

#[test]
fn narrative_verification_distinguishes_active_quiet_history_from_missing_inputs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_narrative_input_db_ticks(&dir, 1, 320);
    let cx = ReaderCtx::open(&path).expect("quiet complete input stream");
    let report = verify_narrative(&cx.reader).expect("active detector on constant inputs");
    report.require_equivalent().expect("true quiet equivalence");
    assert_eq!(report.input_binding.policy.narrative_interval, 1);
    assert_eq!(report.inputs_replayed, 320);
    assert_eq!(report.input_pages, 2);
    assert_eq!(report.event_pages, 1);
    assert_eq!(
        (report.events_reconstructed, report.events_persisted),
        (0, 0)
    );
    assert_eq!(report.reconstructed_digest, report.persisted_digest);
    assert!(report.first_difference.is_none());
    drop(cx);

    let connection = Connection::open_existing(&path).expect("open missing input control");
    assert_eq!(
        connection
            .execute(
                "DELETE FROM replay_events WHERE event_type = 'narrative_input_v1' AND tick = 3"
            )
            .expect("remove one exact input"),
        1
    );
    connection.close().expect("close missing input control");
    let cx = ReaderCtx::open(&path).expect("structurally valid finished source");
    assert!(matches!(
        verify_narrative(&cx.reader),
        Err(AnalyticsError::Storage(StorageError::NarrativeInputStream(
            NarrativeInputStreamError::TickGap {
                expected: 3,
                actual: 4,
                ..
            }
        )))
    ));
    let missing_dir = tempfile::tempdir().expect("tempdir");
    let missing = ReaderCtx::open(&synth_db(&missing_dir)).expect("legacy source without inputs");
    assert!(matches!(
        verify_narrative(&missing.reader),
        Err(AnalyticsError::Storage(StorageError::NarrativeInputStream(
            NarrativeInputStreamError::MissingEvidence {
                first_offending_tick: 1
            }
        )))
    ));
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one real run isolates missing, extra, changed, malformed output and exact-input controls without repeating production setup"
)]
fn narrative_verification_detects_old_output_divergence_and_independent_input_changes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (path, _) = real_narrative_db(&dir);
    let cx = ReaderCtx::open(&path).expect("finished real run");
    let baseline = verify_narrative(&cx.reader).expect("baseline comparison");
    baseline.require_equivalent().expect("baseline parity");
    let all = cx
        .reader
        .run_event_page_v1(None, 4096)
        .expect("small complete fixture");
    assert!(all.next_after.is_none());
    let oldest = all.events.first().expect("old saved event").clone();
    let newest = cx.reader.recent_run_events(1).expect("display tail");
    assert_ne!(oldest.identity(), newest[0].identity());
    let run_id = cx.reader.run_id();
    drop(cx);

    for (kind, field) in [
        (NarrativeDifferenceKind::MissingStoredEvent, None),
        (NarrativeDifferenceKind::UnexpectedStoredEvent, None),
        (NarrativeDifferenceKind::ChangedRecord, Some("magnitude")),
        (NarrativeDifferenceKind::ChangedRecord, Some("subject")),
    ] {
        let connection = Connection::open_existing(&path).expect("open output control");
        let mut modified = oldest.record().clone();
        match kind {
            NarrativeDifferenceKind::MissingStoredEvent => {
                remove_verification_event(&connection, run_id, oldest.record());
            }
            NarrativeDifferenceKind::UnexpectedStoredEvent => {
                modified.metric = "!unexpected-old-event".to_owned();
                insert_verification_event(&connection, run_id, &modified);
            }
            NarrativeDifferenceKind::ChangedRecord => {
                if field == Some("subject") {
                    assert!(modified.subject.is_none());
                    modified.subject = Some(SubjectRef::Agent(AgentUid(9_007_199_254_740_993)));
                } else {
                    modified.magnitude += 1.0;
                }
                remove_verification_event(&connection, run_id, oldest.record());
                insert_verification_event(&connection, run_id, &modified);
            }
        }
        connection.close().expect("close output control");
        let cx = ReaderCtx::open(&path).expect("input and search projections remain valid");
        assert_eq!(
            cx.reader
                .recent_run_events(1)
                .expect("unchanged display tail"),
            newest
        );
        let report = verify_narrative(&cx.reader).expect("complete divergent comparison");
        assert!(!report.equivalent);
        assert_eq!(report.inputs_replayed, 320);
        assert_eq!(report.input_digest, baseline.input_digest);
        assert_eq!(report.reconstructed_digest, baseline.reconstructed_digest);
        assert_ne!(report.persisted_digest, baseline.persisted_digest);
        assert_eq!(report.events_reconstructed, baseline.events_reconstructed);
        let difference = report.first_difference.as_ref().expect("first divergence");
        assert_eq!(difference.index, 0);
        assert_eq!(difference.kind, kind);
        assert!(matches!(
            report.require_equivalent(),
            Err(NarrativeVerificationError::Diverged { index: 0, .. })
        ));
        match kind {
            NarrativeDifferenceKind::MissingStoredEvent => {
                assert_eq!(report.events_persisted + 1, baseline.events_persisted);
                assert_eq!(report.events_matched + 1, baseline.events_matched);
                assert_eq!(
                    (
                        report.events_missing,
                        report.events_unexpected,
                        report.events_changed
                    ),
                    (1, 0, 0)
                );
                assert_eq!(&difference.identity, oldest.identity());
                assert_eq!(difference.reconstructed.as_ref(), Some(oldest.record()));
                assert!(difference.persisted.is_none());
                assert!(difference.field.is_none());
            }
            NarrativeDifferenceKind::UnexpectedStoredEvent => {
                assert_eq!(report.events_persisted, baseline.events_persisted + 1);
                assert_eq!(report.events_matched, baseline.events_matched);
                assert_eq!(
                    (
                        report.events_missing,
                        report.events_unexpected,
                        report.events_changed
                    ),
                    (0, 1, 0)
                );
                assert_eq!(difference.identity.metric(), modified.metric);
                assert!(difference.reconstructed.is_none());
                assert_eq!(difference.persisted.as_ref(), Some(&modified));
                assert!(difference.field.is_none());
            }
            NarrativeDifferenceKind::ChangedRecord => {
                assert_eq!(report.events_persisted, baseline.events_persisted);
                assert_eq!(report.events_matched + 1, baseline.events_matched);
                assert_eq!(
                    (
                        report.events_missing,
                        report.events_unexpected,
                        report.events_changed
                    ),
                    (0, 0, 1)
                );
                assert_eq!(&difference.identity, oldest.identity());
                assert_eq!(difference.field.as_deref(), field);
                assert_eq!(difference.reconstructed.as_ref(), Some(oldest.record()));
                assert_eq!(difference.persisted.as_ref(), Some(&modified));
            }
        }
        drop(cx);
        let connection = Connection::open_existing(&path).expect("restore output control");
        if kind != NarrativeDifferenceKind::MissingStoredEvent {
            remove_verification_event(&connection, run_id, &modified);
        }
        if kind != NarrativeDifferenceKind::UnexpectedStoredEvent {
            insert_verification_event(&connection, run_id, oldest.record());
        }
        connection.close().expect("close restored output");
    }

    // Move the exact population-zero crossing by one tick while leaving saved output intact.
    let connection = Connection::open_existing(&path).expect("open exact input control");
    let original: String = connection.query_row(
        "SELECT payload FROM replay_events WHERE event_type = 'narrative_input_v1' AND tick = 97"
    ).expect("recorded extinction input").get_typed(0).expect("exact input payload");
    let mut altered: serde_json::Value =
        serde_json::from_str(&original).expect("input payload JSON");
    assert_eq!(altered["agent_count_hex"], "0000000000000000");
    altered["agent_count_hex"] = json!("000000000000000c");
    connection.execute_with_params(
        "UPDATE replay_events SET payload = ?1 WHERE event_type = 'narrative_input_v1' AND tick = 97",
        &[serde_json::to_string(&altered).expect("changed input JSON").into()],
    ).expect("write exact changed input");
    connection.close().expect("close input control");
    let cx = ReaderCtx::open(&path).expect("complete changed input stream");
    let report = verify_narrative(&cx.reader).expect("independent input reconstruction");
    assert!(!report.equivalent);
    assert!(report.first_difference.is_some());
    assert_eq!(report.inputs_replayed, 320);
    assert_ne!(report.input_digest, baseline.input_digest);
    assert_ne!(report.reconstructed_digest, baseline.reconstructed_digest);
    assert_eq!(report.persisted_digest, baseline.persisted_digest);
    drop(cx);

    let connection = Connection::open_existing(&path).expect("open malformed row control");
    connection.execute_with_params(
        "UPDATE replay_events SET payload = ?1 WHERE event_type = 'narrative_input_v1' AND tick = 97",
        &[original.into()],
    ).expect("restore exact original input");
    connection
        .execute("UPDATE run_events SET schema_version = 999")
        .expect("corrupt output schema");
    connection.close().expect("close malformed row control");
    let cx = ReaderCtx::open(&path).expect("structurally valid malformed output source");
    assert!(matches!(
        verify_narrative(&cx.reader),
        Err(AnalyticsError::Storage(StorageError::RunEvent(
            RunEventDecodeError::UnsupportedSchemaVersion { found: 999, .. }
        )))
    ));
}

#[test]
fn narrative_verification_cli_writes_both_reports_before_failing_on_divergence() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_narrative_input_db(&dir, 1);
    let executable = env!("CARGO_BIN_EXE_sb-analyze");
    let json_path = dir.path().join("equivalent.json");
    let md_path = dir.path().join("equivalent.md");
    let run = Command::new(executable)
        .args([&path, "verify-narrative", "--json"])
        .arg(&json_path)
        .arg("--md")
        .arg(&md_path)
        .output()
        .expect("run equivalent verification CLI");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let report: NarrativeVerificationV1 =
        serde_json::from_slice(&std::fs::read(&json_path).expect("read equivalent JSON"))
            .expect("parse typed verification report");
    report.require_equivalent().expect("equivalent CLI report");
    assert!(String::from_utf8_lossy(&run.stdout).contains("Equivalent."));
    assert_eq!(
        std::fs::read_to_string(&md_path).expect("read equivalent markdown"),
        report.markdown()
    );

    let connection = Connection::open_existing(&path).expect("open CLI divergence control");
    let unexpected = narrative_event(1, EventKind::PopulationBoom, "population");
    insert_verification_event(&connection, report.input_binding.run_id, &unexpected);
    connection.close().expect("close CLI divergence control");
    let json_path = dir.path().join("diverged.json");
    let md_path = dir.path().join("diverged.md");
    let run = Command::new(executable)
        .args([&path, "verify-narrative", "--json"])
        .arg(&json_path)
        .arg("--md")
        .arg(&md_path)
        .output()
        .expect("run divergent verification CLI");
    assert_eq!(run.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&run.stdout).contains("Diverged."));
    assert!(String::from_utf8_lossy(&run.stderr).contains("saved narrative differs"));
    let report: NarrativeVerificationV1 = serde_json::from_slice(
        &std::fs::read(&json_path).expect("divergent JSON must exist before failure"),
    )
    .expect("parse divergent report");
    assert!(!report.equivalent);
    assert_eq!(report.inputs_replayed, 4);
    assert_eq!(report.events_reconstructed, 0);
    assert_eq!(report.events_persisted, 1);
    assert_eq!(report.events_unexpected, 1);
    let difference = report
        .first_difference
        .as_ref()
        .expect("first extra saved event");
    assert_eq!(
        difference.kind,
        NarrativeDifferenceKind::UnexpectedStoredEvent
    );
    assert_eq!(difference.persisted.as_ref(), Some(&unexpected));
    assert_eq!(
        std::fs::read_to_string(&md_path).expect("divergent markdown must exist before failure"),
        report.markdown()
    );
}

#[test]
fn narrative_verification_isolates_explicitly_selected_runs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_narrative_input_db(&dir, 1);
    let first = ReaderCtx::open(&path).expect("first run");
    let first_id = first.reader.run_id();
    drop(first);
    let second_id = scriptbots_runtime::RunId::new(0x16_020_999);
    let mut storage = Storage::append_run(&path, RunManifestRecord::unattributed(second_id))
        .expect("append second run");
    let config = ScriptBotsConfig {
        narrative_interval: 1,
        history_capacity: 128,
        ..ScriptBotsConfig::default()
    };
    let mut second_batch = batch(1, 10, 10.0);
    let record = NarrativeInputRecordV1::from_summary(&second_batch.summary, 0, &config)
        .expect("second run input");
    second_batch.replay_events.push(ReplayEvent {
        agent_uid: None,
        position: None,
        counterpart: None,
        counterpart_position: None,
        kind: ReplayEventKind::NarrativeInputV1 { record },
    });
    second_batch
        .narrative_events
        .push(narrative_event(1, EventKind::PopulationBoom, "population"));
    storage.persist(&second_batch).expect("second run evidence");
    storage.close().expect("close appended run");
    assert!(matches!(
        ReaderCtx::open(&path),
        Err(AnalyticsError::Storage(StorageError::InvalidData {
            context: "runs.run_id",
            ..
        }))
    ));
    let first = ReaderCtx::open_for_run(&path, first_id).expect("select first run");
    let first_report = verify_narrative(&first.reader).expect("first selected comparison");
    first_report
        .require_equivalent()
        .expect("second run output cannot contaminate first");
    assert_eq!(first_report.inputs_replayed, 4);
    drop(first);
    let second = ReaderCtx::open_for_run(&path, second_id).expect("select second run");
    let second_report = verify_narrative(&second.reader).expect("second selected comparison");
    assert_eq!(second_report.input_binding.run_id, second_id);
    assert_eq!(second_report.inputs_replayed, 1);
    assert_eq!(second_report.events_unexpected, 1);
    assert!(!second_report.equivalent);
    drop(second);

    let run = Command::new(env!("CARGO_BIN_EXE_sb-analyze"))
        .args([&path, "--run-id", &first_id.to_string(), "verify-narrative"])
        .output()
        .expect("CLI explicit run selection");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains(&first_id.to_string()));
}

#[test]
fn unknown_report_is_a_typed_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_db(&dir);
    let cx = ReaderCtx::open(&path).expect("open reader");
    let err = Registry::builtin().run("no-such-report", &cx, &ReportParams::default());
    assert!(
        matches!(err, Err(AnalyticsError::UnknownReport(_))),
        "{err:?}"
    );
}

#[test]
fn reader_is_verified_read_only_and_never_creates_databases() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("does-not-exist.sqlite");
    let missing_str = missing.display().to_string();

    let err = ReaderCtx::open(&missing_str);
    assert!(
        err.is_err(),
        "opening a missing run DB must fail, not create one"
    );
    assert!(
        !missing.exists(),
        "read path must never create a database file"
    );

    #[cfg(unix)]
    {
        let path = synth_db(&dir);
        let symlink = dir.path().join("run-alias.sqlite");
        std::os::unix::fs::symlink(&path, &symlink).expect("create symlink alias");
        assert!(
            ReaderCtx::open(&symlink.display().to_string()).is_err(),
            "verified reader accepted a symlink alias"
        );

        let hardlink = dir.path().join("run-hardlink.sqlite");
        match std::fs::hard_link(&path, &hardlink) {
            Ok(()) => assert!(
                ReaderCtx::open(&hardlink.display().to_string()).is_err(),
                "verified reader accepted a multiply linked database"
            ),
            // On filesystems without hard-link support there is no second path
            // through which the alias attack can be mounted. Accept only the
            // precise capability error; permissions, disk pressure, and other
            // failures must not silently disable this security assertion.
            Err(error) => assert_eq!(
                error.raw_os_error(),
                Some(libc::ENOTSUP),
                "hard-link security check did not run for an unexplained reason: {error}"
            ),
        }
    }
}

#[test]
fn registry_lists_builtin_reports_with_descriptions() {
    let listed = Registry::builtin().list();
    let names: Vec<&str> = listed.iter().map(|(n, _)| *n).collect();
    assert_eq!(
        names,
        vec![
            "run-summary",
            "narrative-timeline",
            "narrative-reconstruct",
            "replay-timeline",
            "metric-summary",
            "metric-changepoints",
            "compare-runs",
            "metric-distribution",
            "phenotype-interactions",
            "lineage-fitness",
            "lineage-structure",
            "dynasty-communities",
            "interaction-centrality",
            "narrative-validate",
        ]
    );
    assert!(listed.iter().all(|(_, d)| !d.is_empty()));
}

#[test]
fn sb_analyze_list_and_run_use_the_real_fixture_database() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_db(&dir);
    let executable = env!("CARGO_BIN_EXE_sb-analyze");

    let list = Command::new(executable)
        .arg(&path)
        .arg("list")
        .output()
        .expect("run sb-analyze list");
    assert!(
        list.status.success(),
        "list stderr: {}",
        String::from_utf8_lossy(&list.stderr)
    );
    let list_stdout = String::from_utf8(list.stdout).expect("list stdout utf8");
    assert!(list_stdout.contains("run-summary"));
    assert!(list_stdout.contains("narrative-timeline"));
    assert!(list_stdout.contains("narrative-reconstruct"));

    let json_path = dir.path().join("summary.json");
    let md_path = dir.path().join("summary.md");
    let run = Command::new(executable)
        .arg(&path)
        .arg("run")
        .arg("run-summary")
        .arg("--json")
        .arg(&json_path)
        .arg("--md")
        .arg(&md_path)
        .output()
        .expect("run sb-analyze report");
    assert!(
        run.status.success(),
        "run stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let machine: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&json_path).expect("read machine report"))
            .expect("parse machine report");
    assert_eq!(machine["schema_version"], REPORT_SCHEMA_VERSION);
    assert_eq!(machine["report"], "run-summary");
    assert_eq!(machine["row_count"], 3);
    let markdown = std::fs::read_to_string(md_path).expect("read markdown report");
    assert!(markdown.contains("# Run summary"));

    let params_run = Command::new(executable)
        .arg(&path)
        .arg("run")
        .arg("narrative-timeline")
        .arg("--params")
        .arg("limit=0")
        .output()
        .expect("run sb-analyze with repeatable params syntax");
    assert!(
        params_run.status.success(),
        "params run stderr: {}",
        String::from_utf8_lossy(&params_run.stderr)
    );
}

fn phenotype_agents() -> [scriptbots_core::AgentState; 2] {
    use scriptbots_core::{
        AgentData, AgentId, AgentIdentity, AgentRuntime, AgentState, AgentUid, Position,
        TraitModifiers, Velocity,
    };

    let agent1 = AgentState {
        id: AgentId::default(),
        identity: AgentIdentity {
            uid: AgentUid(1),
            spawn_ordinal: 0,
            birth_ordinal: None,
        },
        data: AgentData {
            position: Position::new(10.0, 20.0),
            velocity: Velocity::new(3.0, 4.0),
            health: 10.0,
            ..AgentData::default()
        },
        runtime: AgentRuntime {
            energy: 10.0,
            herbivore_tendency: 0.8,
            trait_modifiers: TraitModifiers {
                smell: 1.0,
                sound: 1.0,
                hearing: 1.0,
                eye: 1.0,
                blood: 1.0,
            },
            ..AgentRuntime::default()
        },
    };

    let agent2 = AgentState {
        id: AgentId::default(),
        identity: AgentIdentity {
            uid: AgentUid(2),
            spawn_ordinal: 1,
            birth_ordinal: None,
        },
        data: AgentData {
            position: Position::new(30.0, 40.0),
            velocity: Velocity::new(1.0, 0.0),
            health: 10.0,
            ..AgentData::default()
        },
        runtime: AgentRuntime {
            energy: 10.0,
            herbivore_tendency: 0.2,
            trait_modifiers: TraitModifiers {
                smell: 2.0,
                sound: 2.0,
                hearing: 2.0,
                eye: 2.0,
                blood: 2.0,
            },
            ..AgentRuntime::default()
        },
    };

    [agent1, agent2]
}

fn synth_db_with_phenotypes(dir: &tempfile::TempDir) -> String {
    use scriptbots_core::{
        AgentUid, BirthOrigin, BirthRecord, Generation, INTERACTION_EVENTS_OBSERVED_KIND,
        INTERACTION_EVENTS_PERSISTED_KIND, MetricSample, PersistenceBatch, PersistenceEvent,
        PersistenceEventKind, Position, ReplayEvent, ReplayEventKind, ReplayInteractionKind, Tick,
        TickSummary,
    };

    let path = dir.path().join("phenotypes.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create synth phenotype db");
    let [agent1, agent2] = phenotype_agents();

    let birth1 = BirthRecord {
        tick: Tick(0),
        agent_uid: AgentUid(1),
        spawn_ordinal: 0,
        birth_ordinal: None,
        origin: BirthOrigin::Seeded,
        parent_a: None,
        parent_b: None,
        brain_kind: Some("mlp".to_owned()),
        brain_key: None,
        herbivore_tendency: 0.8,
        generation: Generation(0),
        is_hybrid: false,
        position: Position::new(10.0, 20.0),
    };

    let birth2 = BirthRecord {
        tick: Tick(0),
        agent_uid: AgentUid(2),
        spawn_ordinal: 1,
        birth_ordinal: None,
        origin: BirthOrigin::Seeded,
        parent_a: None,
        parent_b: None,
        brain_kind: Some("mlp".to_owned()),
        brain_key: None,
        herbivore_tendency: 0.2,
        generation: Generation(0),
        is_hybrid: false,
        position: Position::new(30.0, 40.0),
    };

    let replay_interaction = ReplayEvent {
        agent_uid: Some(AgentUid(1)),
        position: Some(Position::new(10.0, 20.0)),
        counterpart: Some(AgentUid(2)),
        counterpart_position: Some(Position::new(30.0, 40.0)),
        kind: ReplayEventKind::Interaction {
            tick: Tick(0),
            ordinal: 0,
            kind: ReplayInteractionKind::Combat,
            magnitude: 2.5,
        },
    };

    let batch = PersistenceBatch {
        summary: TickSummary {
            tick: Tick(0),
            agent_count: 2,
            births: 0,
            deaths: 0,
            total_energy: 20.0,
            average_energy: 10.0,
            average_health: 10.0,
            max_age: 1,
            spike_hits: 0,
        },
        epoch: 1,
        closed: false,
        metrics: vec![MetricSample::new("total_energy", 20.0)],
        events: vec![
            PersistenceEvent::new(
                PersistenceEventKind::Custom(INTERACTION_EVENTS_OBSERVED_KIND.into()),
                1,
            ),
            PersistenceEvent::new(
                PersistenceEventKind::Custom(INTERACTION_EVENTS_PERSISTED_KIND.into()),
                1,
            ),
        ],
        agents: vec![agent1, agent2],
        births: vec![birth1, birth2],
        deaths: Vec::new(),
        replay_events: vec![replay_interaction],
        narrative_events: Vec::new(),
        genomes: Vec::new(),
    };

    storage.persist(&batch).expect("persist batch");
    storage.flush().expect("flush");
    storage.close().expect("close");
    path
}

#[test]
fn phenotype_interactions_report_runs_and_serializes_with_provenance() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_db_with_phenotypes(&dir);
    let executable = env!("CARGO_BIN_EXE_sb-analyze");

    let json_path = dir.path().join("phenotype.json");
    let md_path = dir.path().join("phenotype.md");
    let run = Command::new(executable)
        .arg(&path)
        .arg("run")
        .arg("phenotype-interactions")
        .arg("--json")
        .arg(&json_path)
        .arg("--md")
        .arg(&md_path)
        .output()
        .expect("run sb-analyze phenotype-interactions");
    assert!(
        run.status.success(),
        "run stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );

    let machine: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&json_path).expect("read machine report"))
            .expect("parse machine report");
    assert_eq!(machine["schema_version"], REPORT_SCHEMA_VERSION);
    assert_eq!(machine["report"], "phenotype-interactions");
    assert_eq!(machine["row_count"], 2);
    assert_eq!(
        machine["machine"]["features"]
            .as_array()
            .expect("features array")
            .len(),
        2
    );
    assert_eq!(
        machine["machine"]["interaction_graph"]["nodes"]
            .as_array()
            .expect("nodes array")
            .len(),
        2
    );
    assert_eq!(
        machine["machine"]["interaction_graph"]["edges"]
            .as_array()
            .expect("edges array")
            .len(),
        1
    );
    assert!(machine["machine"]["canonical_digest"].is_string());

    let markdown = std::fs::read_to_string(md_path).expect("read markdown report");
    assert!(markdown.contains("# Phenotype and interaction analysis"));
    assert!(markdown.contains("| uid | observations | speed |"));
}
