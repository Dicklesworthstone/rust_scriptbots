//! Certification regressions over exact persisted detector inputs.
//!
//! Hand-built fixtures isolate statistical behavior and corruption controls. A
//! separate production-world test proves that ordinary sparse persistence still
//! supplies complete tick windows, without changing the persistence interval.

use scriptbots_analytics::{
    AnalyticsError, NARRATIVE_VALIDATE_SCHEMA_VERSION, ReaderCtx, Registry, ReportParams,
};
use scriptbots_core::narrative::{
    EventKind, EventRecord, METRIC_COMBAT, METRIC_ENERGY, METRIC_POPULATION,
    NarrativeInputRecordV1,
};
use scriptbots_core::{
    AgentData, Intervention, MetricSample, PersistenceBatch, Position, Region, ReplayEvent,
    ReplayEventKind, ScriptBotsConfig, Tick, TickSummary, WorldState,
};
use scriptbots_storage::{
    Connection, NarrativeInputStreamError, Storage, StorageError, StoragePipeline,
};

const TERMINAL_TICK: u64 = 600;

fn synthetic_summary(tick: u64) -> TickSummary {
    let phase = u16::try_from(tick % 5).expect("bounded phase");
    let population = if tick < 100 { 10_u16 } else { 40_u16 } + phase;
    let energy_phase = i16::try_from(tick % 7).expect("bounded energy phase");
    let average_energy = 50.0 + f32::from(energy_phase - 3) * 0.5;
    let spike_hits = if tick < 100 { 2 } else { 12 } + u32::from(phase);
    TickSummary {
        tick: Tick(tick),
        agent_count: usize::from(population),
        births: 0,
        deaths: 0,
        total_energy: average_energy * f32::from(population),
        average_energy,
        average_health: 1.0,
        max_age: 0,
        spike_hits,
    }
}

fn event(tick: u64, kind: EventKind, metric: &str) -> EventRecord {
    EventRecord {
        schema_version: 1,
        tick: Tick(tick),
        kind,
        severity: 0.9,
        magnitude: 30.0,
        window: (tick.saturating_sub(30), tick.saturating_add(30)),
        metric: metric.to_owned(),
        before: 10.0,
        after: 40.0,
        score: 15.0,
        subject: None,
        human_text: format!("synthetic {metric} event at {tick}"),
    }
}

/// A synthetic statistics fixture, not a claim of simulated lifecycle counts.
///
/// Summaries and canonical inputs agree exactly. Batches use the production
/// default persistence cadence; generic dashboard metrics deliberately carry
/// no evidence of the planted shift.
fn create_fixture(
    dir: &tempfile::TempDir,
    events: &[EventRecord],
    include_inputs: bool,
) -> String {
    let path = dir.path().join("series.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create fixture");
    let config = ScriptBotsConfig::default();
    let interval = u64::from(config.persistence_interval);
    assert!(interval > 1, "the fixture must exercise sparse persistence");
    let mut replay_events = Vec::new();
    let mut previous_boundary = 0;
    for tick in 1..=TERMINAL_TICK {
        let summary = synthetic_summary(tick);
        if include_inputs {
            let record = NarrativeInputRecordV1::from_summary(&summary, 0, &config)
                .expect("exact detector input");
            replay_events.push(ReplayEvent {
                agent_uid: None,
                position: None,
                counterpart: None,
                counterpart_position: None,
                kind: ReplayEventKind::NarrativeInputV1 { record },
            });
        }
        if !tick.is_multiple_of(interval) && tick != TERMINAL_TICK {
            continue;
        }
        let batch = PersistenceBatch {
            summary,
            epoch: 1,
            closed: false,
            metrics: vec![
                MetricSample::new(METRIC_POPULATION, 5.0),
                MetricSample::new(METRIC_ENERGY, 5.0),
                MetricSample::new(METRIC_COMBAT, 5.0),
            ],
            events: Vec::new(),
            agents: Vec::new(),
            births: Vec::new(),
            deaths: Vec::new(),
            replay_events: std::mem::take(&mut replay_events),
            narrative_events: events
                .iter()
                .filter(|event| previous_boundary < event.tick.0 && event.tick.0 <= tick)
                .cloned()
                .collect(),
            genomes: Vec::new(),
        };
        storage.persist(&batch).expect("persist synthetic series");
        previous_boundary = tick;
    }
    storage.flush().expect("flush fixture");
    storage.close().expect("close fixture");
    path
}

fn certification_params() -> ReportParams {
    ReportParams::from_pairs([
        "window=30".to_owned(),
        "fdr=0.05".to_owned(),
        "resamples=200".to_owned(),
        "permutations=200".to_owned(),
    ])
    .expect("params")
}

#[test]
fn planted_shift_is_certified_from_complete_inputs_across_pages() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = create_fixture(
        &dir,
        &[event(100, EventKind::PopulationBoom, METRIC_POPULATION)],
        true,
    );
    let cx = ReaderCtx::open(&path).expect("open reader");
    let output = Registry::builtin()
        .run("narrative-validate", &cx, &certification_params())
        .expect("run report");

    assert_eq!(output.schema_version, NARRATIVE_VALIDATE_SCHEMA_VERSION);
    assert_eq!(output.machine["events_examined"], 1);
    assert_eq!(output.machine["significant"], 1);
    assert_eq!(output.machine["skipped"], 0);
    assert_eq!(output.machine["inputs_read"], TERMINAL_TICK);
    assert_eq!(output.machine["input_pages"], 3);
    assert_eq!(output.machine["input_binding"]["first_tick"], 1);
    assert_eq!(
        output.machine["input_binding"]["terminal_tick"],
        TERMINAL_TICK
    );
    assert_eq!(output.machine["truncated"], false);

    let rows = output.machine["events"].as_array().expect("events");
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row["tick"], 100);
    assert_eq!(row["metric"], METRIC_POPULATION);
    assert_eq!(row["significant_fdr"], true);
    assert!(row["p_value"].as_f64().expect("p-value") < 0.05);
    assert!(
        row["ci_lower"].as_f64().expect("CI lower") > 20.0,
        "canonical inputs must retain the planted +30 shift"
    );
    assert!(output.human_md.contains("consecutive simulation ticks"));
    assert!(output.human_md.contains("yes"));
}

#[test]
fn stationary_canonical_energy_is_not_certified_significant() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = create_fixture(
        &dir,
        &[event(100, EventKind::EnergyRecovery, METRIC_ENERGY)],
        true,
    );
    let cx = ReaderCtx::open(&path).expect("open reader");
    let output = Registry::builtin()
        .run("narrative-validate", &cx, &certification_params())
        .expect("run report");

    assert_eq!(output.machine["events_examined"], 1);
    assert_eq!(output.machine["significant"], 0);
    assert_eq!(output.machine["skipped"], 0);
    let row = &output.machine["events"][0];
    assert_eq!(row["metric"], METRIC_ENERGY);
    assert_eq!(row["significant_fdr"], false);
}

#[test]
fn combat_certification_uses_exact_spike_hits_missing_from_dashboard_metrics() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = create_fixture(
        &dir,
        &[event(100, EventKind::CombatSurge, METRIC_COMBAT)],
        true,
    );
    let cx = ReaderCtx::open(&path).expect("open reader");
    let output = Registry::builtin()
        .run("narrative-validate", &cx, &certification_params())
        .expect("run report");

    assert_eq!(output.machine["events_examined"], 1);
    assert_eq!(output.machine["significant"], 1);
    assert_eq!(output.machine["events"][0]["metric"], METRIC_COMBAT);
    assert!(
        output.machine["events"][0]["ci_lower"]
            .as_f64()
            .expect("CI")
            > 5.0
    );
}

#[test]
fn incomplete_tick_windows_and_unsupported_metrics_are_skipped_without_substitution() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = create_fixture(
        &dir,
        &[
            event(1, EventKind::PopulationBoom, METRIC_POPULATION),
            event(100, EventKind::PopulationBoom, METRIC_POPULATION),
            event(200, EventKind::ResourceCollapse, "food.total"),
            event(TERMINAL_TICK, EventKind::PopulationBoom, METRIC_POPULATION),
        ],
        true,
    );
    let cx = ReaderCtx::open(&path).expect("open reader");
    let registry = Registry::builtin();
    let output = registry
        .run("narrative-validate", &cx, &certification_params())
        .expect("run report");
    assert_eq!(output.machine["events_examined"], 1);
    assert_eq!(output.machine["skipped"], 3);
    assert_eq!(output.machine["significant"], 1);
    assert_eq!(output.machine["inputs_read"], TERMINAL_TICK);

    let empty_params = ReportParams::from_pairs(["limit=0".to_owned()]).expect("params");
    let empty = registry
        .run("narrative-validate", &cx, &empty_params)
        .expect("explicit empty selection");
    assert_eq!(empty.machine["events_examined"], 0);
    assert_eq!(empty.machine["truncated"], true);
    assert_eq!(empty.machine["inputs_read"], 0);
    assert_eq!(empty.machine["input_binding"], serde_json::Value::Null);
}

#[test]
fn missing_complete_input_stream_is_a_typed_refusal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = create_fixture(
        &dir,
        &[event(100, EventKind::PopulationBoom, METRIC_POPULATION)],
        false,
    );
    let cx = ReaderCtx::open(&path).expect("open older metric-only fixture");
    assert!(matches!(
        Registry::builtin().run("narrative-validate", &cx, &certification_params()),
        Err(AnalyticsError::Storage(StorageError::NarrativeInputStream(
            NarrativeInputStreamError::MissingEvidence {
                first_offending_tick: 1
            }
        )))
    ));
}

#[test]
fn missing_input_inside_or_outside_the_event_window_refuses_the_report() {
    for missing_tick in [99_u64, 257] {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = create_fixture(
            &dir,
            &[event(100, EventKind::PopulationBoom, METRIC_POPULATION)],
            true,
        );
        let connection = Connection::open_existing(path.clone()).expect("open corruption control");
        assert_eq!(
            connection
                .execute(&format!(
                    "DELETE FROM replay_events WHERE event_type = 'narrative_input_v1' AND tick = {missing_tick}"
                ))
                .expect("remove one exact input"),
            1
        );
        connection.close().expect("close corruption control");
        let cx = ReaderCtx::open(&path).expect("open gapped detector stream");
        assert!(matches!(
            Registry::builtin().run("narrative-validate", &cx, &certification_params()),
            Err(AnalyticsError::Storage(StorageError::NarrativeInputStream(
                NarrativeInputStreamError::TickGap {
                    expected,
                    actual,
                    ..
                }
            ))) if expected == missing_tick && actual == missing_tick + 1
        ));
    }
}

#[test]
fn truncated_canonical_tail_is_not_reported_as_a_complete_quiet_window() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = create_fixture(
        &dir,
        &[event(100, EventKind::PopulationBoom, METRIC_POPULATION)],
        true,
    );
    let connection = Connection::open_existing(path.clone()).expect("open corruption control");
    assert_eq!(
        connection
            .execute(&format!(
                "DELETE FROM replay_events WHERE event_type = 'narrative_input_v1' AND tick = {TERMINAL_TICK}"
            ))
            .expect("remove terminal input"),
        1
    );
    connection.close().expect("close corruption control");
    let cx = ReaderCtx::open(&path).expect("open truncated stream");
    assert!(matches!(
        Registry::builtin().run("narrative-validate", &cx, &certification_params()),
        Err(AnalyticsError::Storage(StorageError::NarrativeInputStream(
            NarrativeInputStreamError::Truncated {
                first_offending_tick: TERMINAL_TICK,
                ..
            }
        )))
    ));
}

#[test]
fn real_world_default_persistence_certifies_an_event_between_batch_boundaries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("production.sqlite").display().to_string();
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
        rng_seed: Some(0x16_020_901),
        ..ScriptBotsConfig::default()
    };
    let persistence_interval = u64::from(config.persistence_interval);
    assert!(persistence_interval > 1);
    assert!(!97_u64.is_multiple_of(persistence_interval));
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
    session.finalize(&mut world).expect("finalize partial batch");
    drop(world);
    pipeline.shutdown().expect("durable shutdown");

    let cx = ReaderCtx::open(&path).expect("finished reader");
    assert!(
        !cx.reader.summary_ticks().expect("batch boundaries").contains(&97),
        "the extinction tick must not be a dashboard summary boundary"
    );
    let output = Registry::builtin()
        .run("narrative-validate", &cx, &certification_params())
        .expect("certify production inputs");
    assert_eq!(output.machine["inputs_read"], 320);
    assert_eq!(output.machine["input_pages"], 2);
    let rows = output.machine["events"].as_array().expect("certifications");
    let extinction = rows
        .iter()
        .find(|row| row["kind"] == "Extinction")
        .expect("the production detector must produce a certified extinction row");
    assert_eq!(extinction["tick"], 97);
    assert_eq!(extinction["metric"], METRIC_POPULATION);
    assert_eq!(extinction["significant_fdr"], true);
    assert!(extinction["ci_upper"].as_f64().expect("CI") < 0.0);
}

#[test]
fn empty_database_handles_gracefully() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("empty.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create db");
    storage.flush().expect("flush");
    storage.close().expect("close");

    let cx = ReaderCtx::open(&path).expect("open reader");
    let output = Registry::builtin()
        .run("narrative-validate", &cx, &ReportParams::default())
        .expect("run on empty db");
    assert_eq!(output.schema_version, NARRATIVE_VALIDATE_SCHEMA_VERSION);
    assert_eq!(output.machine["events_examined"], 0);
    assert_eq!(output.machine["significant"], 0);
    assert_eq!(output.machine["inputs_read"], 0);
    assert_eq!(output.machine["input_binding"], serde_json::Value::Null);
    assert!(
        output
            .human_md
            .contains("No narrative events could be certified")
    );
}

#[test]
fn registry_contains_narrative_validate() {
    assert!(
        Registry::builtin()
            .list()
            .iter()
            .any(|(name, _)| *name == "narrative-validate")
    );
}
