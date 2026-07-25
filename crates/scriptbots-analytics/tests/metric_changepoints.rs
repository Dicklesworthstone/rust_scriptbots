//! End-to-end proof that the `metric-changepoints` report locates and certifies a regime shift in
//! a real run database (bd-2z0.11.6).
//!
//! The change-point finder and the certification are unit-tested in isolation; this proves the
//! REPORT wires them to persisted data correctly — a metric that genuinely shifts is found at the
//! right tick and certified real, while a flat metric is not.

use scriptbots_analytics::{ReaderCtx, Registry, ReportParams};
use scriptbots_core::{MetricSample, PersistenceBatch, Tick, TickSummary};
use scriptbots_storage::Storage;

fn batch(tick: u64, metrics: Vec<MetricSample>) -> PersistenceBatch {
    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(tick),
            agent_count: 0,
            births: 0,
            deaths: 0,
            total_energy: 0.0,
            average_energy: 0.0,
            average_health: 0.0,
            max_age: 0,
            spike_hits: 0,
        },
        epoch: 1,
        closed: false,
        metrics,
        events: Vec::new(),
        agents: Vec::new(),
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events: Vec::new(),
        narrative_events: Vec::new(),
    }
}

/// A run whose `shifting` metric jumps from ~10 to ~40 at tick 100, and whose `flat` metric is
/// constant. 200 ticks so a window-30 certification has room on both sides of the shift.
fn fixture(dir: &tempfile::TempDir) -> String {
    let path = dir.path().join("run.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create fixture run db");
    // A tiny deterministic wobble so the segments are not perfectly constant (a real permutation
    // test needs some within-segment variation; a constant series is a degenerate edge case).
    let wobble = |tick: u64| -> f64 {
        let phase = u32::try_from(tick % 5).expect("tick modulo five always fits in u32");
        (f64::from(phase) - 2.0) * 0.3
    };
    for tick in 1..=200u64 {
        let base = if tick < 100 { 10.0 } else { 40.0 };
        let metrics = vec![
            MetricSample::new("shifting", base + wobble(tick)),
            MetricSample::new("flat", 7.0 + wobble(tick)),
        ];
        storage.persist(&batch(tick, metrics)).expect("persist");
    }
    storage.flush().expect("flush");
    storage.close().expect("close");
    path
}

fn run_report(path: &str) -> serde_json::Value {
    let cx = ReaderCtx::open(path).expect("open reader");
    let out = Registry::builtin()
        .run("metric-changepoints", &cx, &ReportParams::default())
        .expect("metric-changepoints runs");
    out.machine
}

fn row<'a>(machine: &'a serde_json::Value, metric: &str) -> Option<&'a serde_json::Value> {
    machine["changepoints"]
        .as_array()
        .expect("changepoints array")
        .iter()
        .find(|c| c["metric"] == metric)
}

#[test]
fn a_real_regime_shift_is_located_and_certified() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir);
    let machine = run_report(&path);

    let shifting = row(&machine, "shifting").expect("the shifting metric must be examined");
    // The shift is at tick 100 (metric jumps 10 -> 40 there). The change tick should be at or very
    // near 100.
    let change_tick = shifting["change_tick"].as_u64().unwrap();
    assert!(
        (95..=105).contains(&change_tick),
        "the change-point should be located at the tick-100 boundary; got {change_tick}"
    );
    assert!(
        (shifting["shift"].as_f64().unwrap() - 30.0).abs() < 2.0,
        "the shift should be ~+30 (10 -> 40); got {}",
        shifting["shift"]
    );
    assert!(
        shifting["significant_fdr"].as_bool().unwrap(),
        "a 30-unit shift over 200 ticks must be certified real (p={})",
        shifting["p_value"]
    );
    assert!(
        shifting["ci_lower"].as_f64().unwrap() > 0.0,
        "the CI on the shift must exclude zero: [{}, {}]",
        shifting["ci_lower"],
        shifting["ci_upper"]
    );

    // The flat metric may or may not have a nominal "largest" split, but it must NOT be certified
    // as a real regime shift.
    if let Some(flat) = row(&machine, "flat") {
        assert!(
            !flat["significant_fdr"].as_bool().unwrap(),
            "the flat metric was certified as a regime shift (p={}); false positive",
            flat["p_value"]
        );
    }

    // Exactly the shifting metric is a real discovery.
    assert_eq!(
        machine["significant"], 1,
        "exactly one metric genuinely shifted"
    );
}

#[test]
fn the_report_is_deterministic_and_registered() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir);
    assert_eq!(
        run_report(&path),
        run_report(&path),
        "the report must be deterministic"
    );

    assert!(
        Registry::builtin()
            .list()
            .iter()
            .any(|(name, _)| *name == "metric-changepoints"),
        "metric-changepoints must be registered and discoverable"
    );
}

#[test]
fn a_window_larger_than_any_series_examines_nothing_without_erroring() {
    // A window bigger than the whole series admits no change-point. The report must handle that as
    // "nothing to certify", not a crash — an empty result is a valid answer.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir);
    let cx = ReaderCtx::open(&path).expect("open reader");
    let params = ReportParams::from_pairs(["window=500".to_owned()]).expect("params");
    let out = Registry::builtin()
        .run("metric-changepoints", &cx, &params)
        .expect("runs even when nothing is examinable");
    assert_eq!(out.machine["metrics_examined"], 0);
    assert_eq!(out.machine["significant"], 0);
}
