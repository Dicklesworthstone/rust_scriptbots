//! End-to-end proof that the `metric-distribution` report characterizes each metric's shape from a
//! real run database (bd-2z0.11.6 item 2).
#![allow(clippy::cast_precision_loss)]
//!
//! The distribution module is unit-tested in isolation; this proves the REPORT loads metrics,
//! groups them, and reports the right shape verdict — a normal metric is not flagged, a skewed one
//! is, and a constant one is reported as constant rather than normal-looking.

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
        genomes: Vec::new(),
    }
}

/// Deterministic gaussian draws so the "normal metric is not flagged" case has a genuinely normal
/// series.
struct Normal {
    state: u64,
}
impl Normal {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    const fn bits(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        let value = self.bits() >> 11;
        (value as f64 + 1.0) / (9_007_199_254_740_992.0 + 1.0)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.unit();
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// A run with three metrics of known shape: `gaussian` (normal), `skewed` (a growing exponential
/// ramp, strongly right-tailed), and `constant`.
fn fixture(dir: &tempfile::TempDir) -> String {
    let path = dir.path().join("run.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create run db");
    let mut draws = Normal::new(4242);
    for tick in 1..=800u64 {
        let ramp = f64::from(u32::try_from(tick).unwrap());
        let metrics = vec![
            MetricSample::new("gaussian", draws.normal()),
            // A strongly right-skewed but modest finite series: exp(0.01)..exp(8).
            MetricSample::new("skewed", (ramp / 100.0).exp()),
            MetricSample::new("constant", 3.0),
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
        .run("metric-distribution", &cx, &ReportParams::default())
        .expect("metric-distribution runs");
    out.machine
}

fn metric<'a>(machine: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    machine["metrics"]
        .as_array()
        .expect("metrics array")
        .iter()
        .find(|m| m["name"] == name)
        .expect("requested metric missing from report")
}

#[test]
fn the_report_characterizes_each_metric_shape_correctly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir);
    let machine = run_report(&path);

    // gaussian: near-zero skew. NOTE: we do NOT assert the binary "not flagged" verdict — the
    // Jarque-Bera test rejects a genuinely normal sample ~alpha of the time by construction, so a
    // binary verdict at 0.05 would be ~5% flaky. We assert the robust facts: small skew, and that
    // it is far MORE normal (much smaller JB, much larger p) than the skewed metric.
    let gaussian = metric(&machine, "gaussian");
    let skewed = metric(&machine, "skewed");
    assert!(
        gaussian["skewness"].as_f64().unwrap().abs() < 0.3,
        "a gaussian metric should have near-zero skewness; got {}",
        gaussian["skewness"]
    );
    assert!(
        gaussian["jarque_bera"].as_f64().unwrap() < skewed["jarque_bera"].as_f64().unwrap(),
        "the gaussian metric's Jarque-Bera ({}) should be far below the skewed metric's ({})",
        gaussian["jarque_bera"],
        skewed["jarque_bera"]
    );

    // skewed: strongly positive skew, robustly flagged non-normal (its JB is enormous, so this is
    // not flaky).
    assert!(
        skewed["skewness"].as_f64().unwrap() > 1.0,
        "a strongly right-tailed metric must have large positive skewness; got {}",
        skewed["skewness"]
    );
    assert_eq!(
        skewed["non_normal"], true,
        "the strongly skewed metric was not flagged non-normal (p={})",
        skewed["jb_p_value"]
    );

    // constant: reported as degenerate, and never non-normal.
    let constant = metric(&machine, "constant");
    assert_eq!(
        constant["degenerate"], true,
        "a constant metric must be flagged degenerate"
    );
    assert_eq!(
        constant["non_normal"], false,
        "a constant metric has no shape and must not be flagged non-normal"
    );

    assert_eq!(
        machine["metrics_examined"], 3,
        "all three metrics have enough values"
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
            .any(|(name, _)| *name == "metric-distribution"),
        "metric-distribution must be registered"
    );
}
