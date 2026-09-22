//! End-to-end integration tests for `narrative-validate` report (`bd-2z0.11.6`).
//!
//! Validates:
//! 1. A real detected regime shift is certified significant under FDR control with positive CI.
//! 2. Stationary null noise is NOT falsely certified significant.
//! 3. Empty database edge-case handles cleanly without errors.
//! 4. The report is registered in `Registry::builtin()` and reachable by name.

use scriptbots_analytics::{ReaderCtx, Registry, ReportParams};
use scriptbots_core::narrative::{EventKind, EventRecord};
use scriptbots_core::{MetricSample, PersistenceBatch, Tick, TickSummary};
use scriptbots_storage::Storage;

const fn empty_summary(tick: u64) -> TickSummary {
    TickSummary {
        tick: Tick(tick),
        agent_count: 0,
        births: 0,
        deaths: 0,
        total_energy: 0.0,
        average_energy: 0.0,
        average_health: 0.0,
        max_age: 0,
        spike_hits: 0,
    }
}

fn create_shift_fixture(dir: &tempfile::TempDir) -> String {
    let path = dir.path().join("shift_run.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create test db");

    let wobble = |tick: u64| -> f64 {
        let phase = u32::try_from(tick % 5).expect("fit");
        (f64::from(phase) - 2.0) * 0.2
    };

    for tick in 1..=200u64 {
        let pop = if tick < 100 { 10.0 } else { 40.0 } + wobble(tick);
        let mut batch = PersistenceBatch {
            summary: empty_summary(tick),
            epoch: 1,
            closed: false,
            metrics: vec![MetricSample::new("population", pop)],
            events: Vec::new(),
            agents: Vec::new(),
            births: Vec::new(),
            deaths: Vec::new(),
            replay_events: Vec::new(),
            narrative_events: Vec::new(),
            genomes: Vec::new(),
        };

        if tick == 100 {
            batch.narrative_events.push(EventRecord {
                schema_version: 1,
                tick: Tick(100),
                kind: EventKind::PopulationBoom,
                severity: 0.9,
                magnitude: 30.0,
                window: (70, 130),
                metric: "population".to_owned(),
                before: 10.0,
                after: 40.0,
                score: 15.0,
                subject: None,
                human_text: "population boomed after environmental shift".to_owned(),
            });
        }

        storage.persist(&batch).expect("persist batch");
    }

    storage.flush().expect("flush");
    storage.close().expect("close");
    path
}

fn create_stationary_noise_fixture(dir: &tempfile::TempDir) -> String {
    let path = dir.path().join("noise_run.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create test db");

    let noise = |tick: u64| -> f64 {
        let phase = u32::try_from(tick % 7).expect("fit");
        (f64::from(phase) - 3.0) * 0.5
    };

    for tick in 1..=200u64 {
        let val = 50.0 + noise(tick);
        let mut batch = PersistenceBatch {
            summary: empty_summary(tick),
            epoch: 1,
            closed: false,
            metrics: vec![MetricSample::new("energy", val)],
            events: Vec::new(),
            agents: Vec::new(),
            births: Vec::new(),
            deaths: Vec::new(),
            replay_events: Vec::new(),
            narrative_events: Vec::new(),
            genomes: Vec::new(),
        };

        if tick == 100 {
            batch.narrative_events.push(EventRecord {
                schema_version: 1,
                tick: Tick(100),
                kind: EventKind::EnergyRecovery,
                severity: 0.5,
                magnitude: 0.0,
                window: (70, 130),
                metric: "energy".to_owned(),
                before: 50.0,
                after: 50.0,
                score: 0.1,
                subject: None,
                human_text: "false alarm fluctuation".to_owned(),
            });
        }

        storage.persist(&batch).expect("persist batch");
    }

    storage.flush().expect("flush");
    storage.close().expect("close");
    path
}

#[test]
fn planted_shift_is_certified_significant() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = create_shift_fixture(&dir);
    let cx = ReaderCtx::open(&path).expect("open reader");

    let params = ReportParams::from_pairs([
        "window=30".to_string(),
        "fdr=0.05".to_string(),
        "resamples=200".to_string(),
        "permutations=200".to_string(),
    ])
    .expect("params");

    let output = Registry::builtin()
        .run("narrative-validate", &cx, &params)
        .expect("run report");

    let machine = output.machine;
    assert_eq!(machine["events_examined"].as_u64(), Some(1));
    assert_eq!(machine["significant"].as_u64(), Some(1));

    let events = machine["events"].as_array().expect("events array");
    assert_eq!(events.len(), 1);
    let ev = &events[0];
    assert_eq!(ev["metric"].as_str(), Some("population"));
    assert_eq!(ev["significant_fdr"].as_bool(), Some(true));
    assert!(
        ev["p_value"].as_f64().unwrap() < 0.05,
        "p-value must be < 0.05, got {}",
        ev["p_value"]
    );
    assert!(
        ev["ci_lower"].as_f64().unwrap() > 20.0,
        "CI lower bound should reflect the ~+30 jump, got {}",
        ev["ci_lower"]
    );

    assert!(
        output
            .human_md
            .contains("Narrative event statistical certification")
    );
    assert!(output.human_md.contains("population"));
    assert!(output.human_md.contains("yes"));
}

#[test]
fn stationary_noise_is_not_certified_significant() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = create_stationary_noise_fixture(&dir);
    let cx = ReaderCtx::open(&path).expect("open reader");

    let params = ReportParams::from_pairs([
        "window=30".to_string(),
        "fdr=0.05".to_string(),
        "resamples=200".to_string(),
        "permutations=200".to_string(),
    ])
    .expect("params");

    let output = Registry::builtin()
        .run("narrative-validate", &cx, &params)
        .expect("run report");

    let machine = output.machine;
    assert_eq!(machine["events_examined"].as_u64(), Some(1));
    assert_eq!(
        machine["significant"].as_u64(),
        Some(0),
        "stationary noise must not be certified"
    );

    let events = machine["events"].as_array().expect("events array");
    assert_eq!(events.len(), 1);
    let ev = &events[0];
    assert_eq!(ev["metric"].as_str(), Some("energy"));
    assert_eq!(ev["significant_fdr"].as_bool(), Some(false));
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

    let machine = output.machine;
    assert_eq!(machine["events_examined"].as_u64(), Some(0));
    assert_eq!(machine["significant"].as_u64(), Some(0));
    assert!(
        output
            .human_md
            .contains("No narrative events could be certified")
    );
}

#[test]
fn registry_contains_narrative_validate() {
    let reg = Registry::builtin();
    let list = reg.list();
    assert!(
        list.iter().any(|(name, _)| *name == "narrative-validate"),
        "narrative-validate must be in registered reports: {list:?}"
    );
}
