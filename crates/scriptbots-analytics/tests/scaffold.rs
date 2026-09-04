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
    AnalyticsError, REPORT_SCHEMA_VERSION, ReaderCtx, Registry, ReportParams,
};
use scriptbots_core::{
    MetricSample, PersistenceBatch, ReplayEvent, ReplayEventKind, ReplayRngScope, Tick, TickSummary,
};
use scriptbots_storage::Storage;
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
    synth_db_with_replay(dir, false)
}

fn synth_db_with_replay(dir: &tempfile::TempDir, include_replay: bool) -> String {
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
        storage.persist(&tick_batch).expect("persist batch");
    }
    storage.flush().expect("flush");
    storage.close().expect("close");
    path
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
fn narrative_timeline_handles_empty_stream_and_limit_param() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_db(&dir);

    let cx = ReaderCtx::open(&path).expect("open reader");
    let registry = Registry::builtin();

    let out = registry
        .run("narrative-timeline", &cx, &ReportParams::default())
        .expect("timeline");
    assert_eq!(out.schema_version, REPORT_SCHEMA_VERSION);
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
            "schema_version": 1,
            "report": "narrative-timeline",
            "db_path": path,
            "latest_tick": 3,
            "row_count": 0,
            "machine": {
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
        .run("narrative-timeline", &cx, &params)
        .expect("timeline with limit");
    let bad = ReportParams::from_pairs(["limit=banana".to_owned()])
        .and_then(|p| registry.run("narrative-timeline", &cx, &p));
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
fn narrative_timeline_orders_serializes_and_limits_replay_events() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_db_with_replay(&dir, true);
    let cx = ReaderCtx::open(&path).expect("open reader");
    let registry = Registry::builtin();

    let out = registry
        .run("narrative-timeline", &cx, &ReportParams::default())
        .expect("full timeline");
    assert_eq!(out.row_count, 2);
    assert_eq!(
        out.machine["event_counts"],
        json!([["brain_outputs", 1], ["rng_sample", 1]])
    );
    assert_eq!(out.machine["events"][0]["tick"], 1);
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
            "narrative-timeline",
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
            "metric-summary",
            "metric-changepoints",
            "compare-runs",
            "metric-distribution",
            "phenotype-interactions",
            "lineage-fitness",
            "lineage-structure",
            "dynasty-communities",
            "interaction-centrality",
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

fn synth_db_with_phenotypes(dir: &tempfile::TempDir) -> String {
    use scriptbots_core::{
        AgentData, AgentId, AgentIdentity, AgentRuntime, AgentState, AgentUid, BirthOrigin,
        BirthRecord, Generation, INTERACTION_EVENTS_OBSERVED_KIND,
        INTERACTION_EVENTS_PERSISTED_KIND, MetricSample, PersistenceBatch, PersistenceEvent,
        PersistenceEventKind, Position, ReplayEvent, ReplayEventKind, ReplayInteractionKind, Tick,
        TickSummary, TraitModifiers, Velocity,
    };

    let path = dir.path().join("phenotypes.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create synth phenotype db");

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
