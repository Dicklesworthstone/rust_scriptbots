//! End-to-end proof that the `metric-summary` report reads a real run database and computes the
//! native statistics correctly (bd-2z0.11.6 item 2 foundation).
//!
//! The stats module is unit-tested in isolation; this test proves the REPORT — load metrics from a
//! `FrankenSQLite` run database through the real write→read path, group by metric, and render the
//! summary — produces the hand-computed answers. A report that computed the right statistics over
//! the wrong rows, or the wrong statistics over the right rows, would pass neither.

use scriptbots_analytics::{ReaderCtx, Registry, ReportParams};
use scriptbots_core::{MetricSample, PersistenceBatch, Tick, TickSummary};
use scriptbots_storage::Storage;

/// A tick batch carrying a set of named metric samples.
const fn batch(tick: u64, metrics: Vec<MetricSample>) -> PersistenceBatch {
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
        genomes: Vec::new(),
    }
}

/// Build a run database whose metric `probe` takes the values 1..=5 across five ticks, and whose
/// metric `flat` is constant at 7.0. Both have hand-computable summaries.
fn fixture(dir: &tempfile::TempDir) -> String {
    let path = dir.path().join("run.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create fixture run db");
    for value in 1_u32..=5 {
        let metrics = vec![
            MetricSample::new("probe", f64::from(value)),
            MetricSample::new("flat", 7.0),
        ];
        storage
            .persist(&batch(u64::from(value), metrics))
            .expect("persist");
    }
    storage.flush().expect("flush");
    storage.close().expect("close");
    path
}

fn run_report(path: &str) -> serde_json::Value {
    let cx = ReaderCtx::open(path).expect("open reader");
    let out = Registry::builtin()
        .run("metric-summary", &cx, &ReportParams::default())
        .expect("metric-summary runs");
    out.machine
}

fn metric<'a>(machine: &'a serde_json::Value, name: &str) -> Result<&'a serde_json::Value, String> {
    machine["metrics"]
        .as_array()
        .expect("metrics array")
        .iter()
        .find(|m| m["name"] == name)
        .ok_or_else(|| format!("metric `{name}` missing from the report"))
}

#[test]
fn the_report_computes_the_hand_verified_summary_of_a_real_run() -> Result<(), String> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir);
    let machine = run_report(&path);

    // `probe` = {1,2,3,4,5}: mean 3, median 3, min 1, max 5, sd = sqrt(2.5) ≈ 1.5811 (n-1).
    let probe = metric(&machine, "probe")?;
    assert_eq!(
        probe["n"], 5,
        "the report must see all five persisted values"
    );
    assert!(
        (probe["mean"].as_f64().unwrap() - 3.0).abs() < 1e-9,
        "mean of 1..5 is 3"
    );
    assert!(
        (probe["median"].as_f64().unwrap() - 3.0).abs() < 1e-9,
        "median of 1..5 is 3"
    );
    assert!((probe["min"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    assert!((probe["max"].as_f64().unwrap() - 5.0).abs() < 1e-9);
    assert!(
        (probe["std_dev"].as_f64().unwrap() - 2.5_f64.sqrt()).abs() < 1e-9,
        "sample sd of 1..5 is sqrt(2.5); got {}",
        probe["std_dev"]
    );
    // CV = sd / |mean| = sqrt(2.5)/3.
    assert!(
        (probe["coefficient_of_variation"].as_f64().unwrap() - 2.5_f64.sqrt() / 3.0).abs() < 1e-9,
        "CV must be sd/mean"
    );

    // `flat` = constant 7.0: zero spread, and the CV is reported as present-but-zero (the mean is
    // far from zero), not omitted.
    let flat = metric(&machine, "flat")?;
    assert_eq!(flat["n"], 5);
    assert!((flat["mean"].as_f64().unwrap() - 7.0).abs() < 1e-9);
    assert_eq!(
        flat["std_dev"].as_f64().unwrap().to_bits(),
        0.0_f64.to_bits(),
        "a constant metric has zero spread"
    );
    assert_eq!(
        flat["coefficient_of_variation"].as_f64().unwrap().to_bits(),
        0.0_f64.to_bits(),
        "a constant non-zero metric has CV 0, not null"
    );
    Ok(())
}

#[test]
fn the_report_lists_metrics_in_a_stable_sorted_order() {
    // Two runs of the report over the same database must render byte-identical, so the machine
    // payload can be diffed. The BTreeMap grouping guarantees name-sorted order.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir);
    let first = run_report(&path);
    let second = run_report(&path);
    assert_eq!(
        first, second,
        "the report is not deterministic over a fixed database"
    );

    let names: Vec<&str> = first["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(
        names, sorted,
        "metrics must be rendered in name-sorted order: {names:?}"
    );
    // `flat` sorts before `probe`.
    assert_eq!(names, vec!["flat", "probe"]);
}

#[test]
fn the_report_is_registered_and_listed() {
    // A report nobody can discover is a report nobody runs. It must appear in the registry list
    // with its description, next to the reports that were already there.
    let listed = Registry::builtin().list();
    assert!(
        listed.iter().any(|(name, _)| *name == "metric-summary"),
        "metric-summary is missing from the report registry: {listed:?}"
    );
}
