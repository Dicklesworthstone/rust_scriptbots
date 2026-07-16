//! End-to-end proof that the `compare-runs` report certifies a treatment effect between two run
//! databases (bd-2z0.11.6 item 3; serves bd-16g.1.4).
//!
//! The paired comparison is unit-tested in isolation. This proves the REPORT opens a second run
//! database, tick-aligns each shared metric into matched pairs, and certifies the effect — a
//! metric the treatment genuinely shifted is found significant, a metric it did not is not.

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
    }
}

/// Small deterministic pseudo-noise so no metric's paired differences are perfectly constant (a
/// constant difference gives an infinite standardized effect size and a degenerate test).
fn wobble(tick: u64, salt: u64) -> f64 {
    ((tick.wrapping_mul(7).wrapping_add(salt) % 11) as f64 - 5.0) * 0.1
}

/// Build a run database. `shifting_base` sets the level of the `shifting` metric; both runs use the
/// same ticks so the report can tick-align them.
fn run_db(dir: &tempfile::TempDir, file: &str, shifting_base: f64, salt: u64) -> String {
    let path = dir.path().join(file).display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create run db");
    for tick in 1..=200u64 {
        let metrics = vec![
            MetricSample::new("shifting", shifting_base + wobble(tick, salt)),
            MetricSample::new("flat", 7.0 + wobble(tick, salt + 100)),
        ];
        storage.persist(&batch(tick, metrics)).expect("persist");
    }
    storage.flush().expect("flush");
    storage.close().expect("close");
    path
}

fn compare(control: &str, treatment: &str) -> serde_json::Value {
    let cx = ReaderCtx::open(control).expect("open control");
    let params =
        ReportParams::from_pairs([format!("treatment_db={treatment}")]).expect("params");
    let out = Registry::builtin()
        .run("compare-runs", &cx, &params)
        .expect("compare-runs runs");
    out.machine
}

fn metric<'a>(machine: &'a serde_json::Value, name: &str) -> Option<&'a serde_json::Value> {
    machine["metrics"]
        .as_array()
        .expect("metrics array")
        .iter()
        .find(|m| m["metric"] == name)
}

#[test]
fn a_treatment_effect_between_two_runs_is_certified() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Control `shifting` sits at 10; treatment at 15 — a +5 effect. `flat` is 7 in both.
    let control = run_db(&dir, "control.sqlite", 10.0, 1);
    let treatment = run_db(&dir, "treatment.sqlite", 15.0, 2);

    let machine = compare(&control, &treatment);

    let shifting = metric(&machine, "shifting").expect("shifting must be compared");
    assert_eq!(shifting["n_pairs"], 200, "all 200 ticks are shared and should pair");
    assert!(
        (shifting["mean_difference"].as_f64().unwrap() - 5.0).abs() < 1.0,
        "the treatment effect should be ~+5; got {}",
        shifting["mean_difference"]
    );
    assert!(
        shifting["significant_fdr"].as_bool().unwrap(),
        "a +5 effect over 200 matched pairs must be certified (p={})",
        shifting["p_value"]
    );
    assert!(
        shifting["ci_lower"].as_f64().unwrap() > 0.0,
        "the CI on the effect must exclude zero: [{}, {}]",
        shifting["ci_lower"],
        shifting["ci_upper"]
    );

    // `flat` was not changed by the treatment; it must NOT be certified.
    let flat = metric(&machine, "flat").expect("flat must be compared");
    assert!(
        !flat["significant_fdr"].as_bool().unwrap(),
        "the unchanged `flat` metric was certified as a treatment effect (p={}); false positive",
        flat["p_value"]
    );

    assert_eq!(machine["significant"], 1, "exactly one metric genuinely shifted");
    assert_eq!(
        machine["treatment_db"].as_str(),
        Some(treatment.as_str()),
        "provenance records the treatment db"
    );
}

#[test]
fn identical_runs_show_no_treatment_effect() {
    // Comparing a run to a byte-identical twin must certify NOTHING — there is no effect. This is
    // the false-positive guard: if this reported an effect, the whole comparison would be noise.
    let dir = tempfile::tempdir().expect("tempdir");
    let control = run_db(&dir, "a.sqlite", 10.0, 7);
    let twin = run_db(&dir, "b.sqlite", 10.0, 7); // same base AND same salt → identical series

    let machine = compare(&control, &twin);
    assert_eq!(
        machine["significant"], 0,
        "two identical runs must show no treatment effect; got {} 'significant' metrics",
        machine["significant"]
    );
}

#[test]
fn a_missing_treatment_db_param_is_a_typed_error_not_a_panic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let control = run_db(&dir, "only.sqlite", 10.0, 1);
    let cx = ReaderCtx::open(&control).expect("open control");
    let out = Registry::builtin().run("compare-runs", &cx, &ReportParams::default());
    assert!(
        matches!(out, Err(scriptbots_analytics::AnalyticsError::BadParam { .. })),
        "compare-runs without treatment_db must be a typed BadParam, got {out:?}"
    );
}

#[test]
fn the_report_is_registered() {
    assert!(
        Registry::builtin()
            .list()
            .iter()
            .any(|(name, _)| *name == "compare-runs"),
        "compare-runs must be registered and discoverable"
    );
}
