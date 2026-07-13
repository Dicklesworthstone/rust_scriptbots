//! Scaffold integration tests for scriptbots-analytics (bd-2z0.11.5).
//!
//! Deliberate design improvement over the original bead spec: instead of a
//! checked-in binary fixture DB (+ regeneration script that inevitably
//! drifts), every test synthesizes its own run database in a tempdir via the
//! real storage write path (`Storage::create_new_file` → `persist` → `flush`
//! → `close`) and then reads it back through the real read path
//! (`StorageReader`). No binary blobs in git, no fixture drift, and the
//! storage crate's own admission machinery is exercised end-to-end.

use scriptbots_analytics::{AnalyticsError, ReaderCtx, Registry, ReportParams, REPORT_SCHEMA_VERSION};
use scriptbots_core::{MetricSample, PersistenceBatch, Tick, TickSummary};
use scriptbots_storage::Storage;

fn batch(tick: u64, agent_count: usize, energy: f32) -> PersistenceBatch {
    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(tick),
            agent_count,
            births: 1,
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
    }
}

/// Builds a three-tick run database and returns its path.
fn synth_db(dir: &tempfile::TempDir) -> String {
    let path = dir.path().join("run.sqlite").display().to_string();
    let mut storage = Storage::create_new_file(&path).expect("create synth run db");
    for (tick, pop, energy) in [(1u64, 10usize, 100.0f32), (2, 12, 130.0), (3, 8, 90.0)] {
        storage.persist(&batch(tick, pop, energy)).expect("persist batch");
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

    let m = &out.machine;
    assert_eq!(m["tick_count"], 3, "three ticks persisted: {m}");
    assert_eq!(m["population_first"], 10, "chronological order, not reader order: {m}");
    assert_eq!(m["population_last"], 8);
    assert_eq!(m["population_min"], 8);
    assert_eq!(m["population_max"], 12);
    assert_eq!(m["total_energy_first"], 100.0);
    assert_eq!(m["total_energy_last"], 90.0);
    assert!(out.human_md.contains("Run summary"), "markdown rendered: {}", out.human_md);
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
    assert_eq!(out.machine["events"].as_array().map(Vec::len), Some(0));
    assert!(out.human_md.contains("No replay events"), "{}", out.human_md);

    // limit param parses; bad param is a typed error, not a panic.
    let params = ReportParams::from_pairs(["limit=5".to_owned()]).expect("params");
    registry.run("narrative-timeline", &cx, &params).expect("timeline with limit");
    let bad = ReportParams::from_pairs(["limit=banana".to_owned()])
        .and_then(|p| registry.run("narrative-timeline", &cx, &p));
    assert!(matches!(bad, Err(AnalyticsError::BadParam { .. })), "typed error: {bad:?}");
}

#[test]
fn unknown_report_is_a_typed_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = synth_db(&dir);
    let cx = ReaderCtx::open(&path).expect("open reader");
    let err = Registry::builtin().run("no-such-report", &cx, &ReportParams::default());
    assert!(matches!(err, Err(AnalyticsError::UnknownReport(_))), "{err:?}");
}

#[test]
fn reader_is_read_only_and_never_creates_databases() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("does-not-exist.sqlite");
    let missing_str = missing.display().to_string();

    let err = ReaderCtx::open(&missing_str);
    assert!(err.is_err(), "opening a missing run DB must fail, not create one");
    assert!(!missing.exists(), "read path must never create a database file");
}

#[test]
fn registry_lists_builtin_reports_with_descriptions() {
    let listed = Registry::builtin().list();
    let names: Vec<&str> = listed.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, vec!["run-summary", "narrative-timeline"]);
    assert!(listed.iter().all(|(_, d)| !d.is_empty()));
}
