//! Offline science layer for `ScriptBots` (bd-2z0.11.5, program bd-2js6).
//!
//! This crate is the ONE blessed offline reader of finished run databases:
//! a report framework plus the `sb-analyze` CLI. Boundary rules it exists to
//! uphold (`docs/franken_integration.md` §4):
//!
//! - **Read-only**: all access goes through [`scriptbots_storage::StorageReader`],
//!   which exposes no mutating API. This crate never opens a writable
//!   connection and never competes with a live run's storage worker.
//! - **Native-only**: never a dependency of the app binaries and never part
//!   of any wasm graph (`ci/check_wasm_graph.sh` guard B enforces the
//!   reverse boundary).
//! - **Franken analytics adapters land here** (fsci-stats: bd-2z0.11.6,
//!   fnx graphs: bd-2z0.11.7, frankenpandas exports: bd-2z0.11.8) behind
//!   this crate's report registry — never in the tick path, never in core.
//! - **Export successor**: report coverage lands here before the app's direct-DB
//!   `control_cli Export` path is retired under bd-2z0.8.9.5.
//!
//! Every report execution is wrapped in a tracing span carrying the report
//! name, parameter set, row counts, and wall time, so detailed logging is a
//! property of the framework rather than a per-report afterthought.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Instant;

use scriptbots_storage::{PersistedMetric, PersistenceBatchId, StorageError, StorageReader};
use serde::Serialize;

/// Native, dependency-free statistics for offline detector certification (bd-2z0.11.6).
///
/// Bootstrap confidence intervals, permutation tests, and effect sizes as pure functions over
/// `&[f64]`. Implemented natively rather than via `fsci-stats` because the latter is git-only and
/// nightly-only, and the four estimators the certification actually needs require neither — the
/// module's calibration tests demonstrate the native path is sufficient, which is the evidence
/// bd-2z0.11.3's adapter decision consumes. Never enters core or any tick path.
pub mod stats;

/// Statistical certification of narrative events (bd-2z0.11.6, item 1).
///
/// Answers "is this detected event real, or the tail of noise?" for the events the detector
/// fires, with Benjamini-Hochberg false-discovery-rate control across a whole run — the
/// principled replacement for eyeballed per-event thresholds that bd-16g.2.3's false-positive
/// budget needs. Pure functions over a metric series; the report that reads real `EventRecord`s
/// from a database is a thin adapter on top.
pub mod certify;

/// Matched-seed treatment-effect analysis (bd-2z0.11.6 item 3; serves bd-16g.1.4).
///
/// Given the same seeds run under a control and a treatment, measures whether the treatment
/// changed each metric — with a paired design (sign-flip permutation, paired bootstrap CI,
/// Cohen's `d_z`) that exploits the matched seeds, and Benjamini-Hochberg across metrics so a
/// study measuring many outcomes cannot report a chance "effect". Pure functions; the DB glue
/// that pulls per-seed outcomes from two run databases is a thin adapter on top.
pub mod compare;

/// Single change-point detection over a metric series (bd-2z0.11.6).
///
/// Finds the split that maximizes the absolute mean shift in a metric — the "if this run had one
/// regime shift, where was it?" question — which the `metric-changepoints` report then certifies
/// via [`certify`]. Pure; the certification that consumes it is where the resampling lives.
pub mod changepoint;

/// Native distribution characterization (bd-2z0.11.6 item 2).
///
/// Moment-based shape summary — skewness, kurtosis, and the Jarque-Bera normality test with an
/// exact chi-square(2) p-value — so "is this metric normal, and how is it shaped?" is answered
/// natively, with no `erf` and no `fsci` dependency. Full distribution fitting (lognormal/gamma +
/// KS) is left for the adapter decision (bd-2z0.11.3). Pure functions over a slice.
pub mod distribution;

/// Schema version stamped into every machine-readable report payload.
///
/// Bump ONLY with a migration note in the owning Bead/release evidence. Full
/// envelope goldens assert the value so an accidental schema change is loud.
pub const REPORT_SCHEMA_VERSION: u32 = 1;

/// Maximum metric rows sampled by `metric-summary` in one bounded SQL page.
///
/// Keeping report reads capped prevents a finished multi-run database from being
/// materialized wholesale merely to compute an interactive summary.
const METRIC_SUMMARY_ROW_LIMIT: usize = 4_096;

/// Maximum recent tick summaries sampled by `run-summary` in one bounded SQL page.
const RUN_SUMMARY_TICK_LIMIT: usize = 4_096;

/// Default number of recent replay events rendered by `narrative-timeline`.
const NARRATIVE_TIMELINE_DEFAULT_LIMIT: usize = 1_024;

/// Hard ceiling for a caller-selected `narrative-timeline` SQL page.
const NARRATIVE_TIMELINE_MAX_LIMIT: usize = 4_096;

/// Errors surfaced by the analytics layer.
#[derive(Debug, thiserror::Error)]
pub enum AnalyticsError {
    /// The underlying read-only storage access failed.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
    /// The requested report is not registered.
    #[error("unknown report '{0}' (run `sb-analyze <db> list` for the registry)")]
    UnknownReport(String),
    /// A parameter failed to parse or validate.
    #[error("bad parameter '{name}': {reason}")]
    BadParam {
        /// Parameter key as supplied by the caller.
        name: String,
        /// Human-readable validation failure.
        reason: String,
    },
    /// Serialization of the machine payload failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Writing a requested report artifact failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Read-only context handed to every report.
pub struct ReaderCtx {
    /// Open read-only handle over the finished run database.
    pub reader: StorageReader,
    /// Path of the database, for provenance stamping in outputs.
    pub db_path: String,
}

impl ReaderCtx {
    /// Opens a finished run database read-only.
    ///
    /// Fails (rather than creating anything) when the path does not exist —
    /// asserted by the scaffold tests as the read-only contract. The finished-run
    /// lease rejects a live writer and remains held for every report query.
    pub fn open(db_path: &str) -> Result<Self, AnalyticsError> {
        let reader = StorageReader::open_finished(db_path)?;
        Ok(Self {
            reader,
            db_path: db_path.to_owned(),
        })
    }
}

/// String-keyed report parameters with typed accessors.
#[derive(Debug, Default, Clone)]
pub struct ReportParams(BTreeMap<String, String>);

impl ReportParams {
    /// Builds parameters from `key=value` pairs, rejecting malformed input.
    pub fn from_pairs<I: IntoIterator<Item = String>>(pairs: I) -> Result<Self, AnalyticsError> {
        let mut map = BTreeMap::new();
        for pair in pairs {
            let Some((k, v)) = pair.split_once('=') else {
                return Err(AnalyticsError::BadParam {
                    name: pair,
                    reason: "expected key=value".into(),
                });
            };
            let key = k.trim();
            if key.is_empty() {
                return Err(AnalyticsError::BadParam {
                    name: pair,
                    reason: "parameter name must not be empty".into(),
                });
            }
            if map.insert(key.to_owned(), v.trim().to_owned()).is_some() {
                return Err(AnalyticsError::BadParam {
                    name: key.to_owned(),
                    reason: "parameter was supplied more than once".into(),
                });
            }
        }
        Ok(Self(map))
    }

    /// Raw string lookup.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    /// Parses an optional `usize` parameter.
    pub fn get_usize(&self, key: &str) -> Result<Option<usize>, AnalyticsError> {
        self.get(key)
            .map(|raw| {
                raw.parse::<usize>().map_err(|e| AnalyticsError::BadParam {
                    name: key.to_owned(),
                    reason: e.to_string(),
                })
            })
            .transpose()
    }

    /// Iterates the raw pairs (stable order) for logging.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

/// A finished report: stable machine payload plus human-readable markdown.
#[derive(Debug, Serialize)]
pub struct ReportOutput {
    /// Machine payload schema version ([`REPORT_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Registered report name.
    pub report: String,
    /// Database path the report ran against (provenance).
    pub db_path: String,
    /// Latest tick present in the database when the report ran, if any.
    pub latest_tick: Option<u64>,
    /// Number of primary rows rendered by this report.
    pub row_count: usize,
    /// Machine-readable payload (stable per `schema_version`).
    pub machine: serde_json::Value,
    /// Human-readable markdown rendering of the same content.
    #[serde(skip)]
    pub human_md: String,
}

/// A single offline report over a finished run database.
pub trait Report {
    /// Stable registry name (kebab-case).
    fn name(&self) -> &'static str;
    /// One-line description shown by `sb-analyze list`.
    fn description(&self) -> &'static str;
    /// Executes the report read-only.
    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError>;
}

/// Registry of available reports.
pub struct Registry {
    reports: Vec<Box<dyn Report>>,
}

impl Registry {
    /// Builds the built-in registry.
    ///
    /// Franken-adapter reports (fsci/fnx/frankenpandas) register here as
    /// their beads land (bd-2z0.11.6/.7/.8).
    #[must_use]
    pub fn builtin() -> Self {
        Self {
            reports: vec![
                Box::new(RunSummary),
                Box::new(NarrativeTimeline),
                Box::new(MetricSummary),
                Box::new(MetricChangepoints),
                Box::new(RunComparison),
                Box::new(MetricDistribution),
            ],
        }
    }

    /// Lists `(name, description)` pairs in registration order.
    #[must_use]
    pub fn list(&self) -> Vec<(&'static str, &'static str)> {
        self.reports
            .iter()
            .map(|r| (r.name(), r.description()))
            .collect()
    }

    /// Runs a report by name with framework-level tracing.
    pub fn run(
        &self,
        name: &str,
        cx: &ReaderCtx,
        params: &ReportParams,
    ) -> Result<ReportOutput, AnalyticsError> {
        let report = self
            .reports
            .iter()
            .find(|r| r.name() == name)
            .ok_or_else(|| AnalyticsError::UnknownReport(name.to_owned()))?;
        let span = tracing::info_span!("report", name = %name, db = %cx.db_path);
        let _guard = span.enter();
        for (k, v) in params.iter() {
            tracing::debug!(param = %k, value = %v, "report parameter");
        }
        let started = Instant::now();
        tracing::info!("report started");
        let result = report.run(cx, params);
        match &result {
            Ok(out) => tracing::info!(
                elapsed_ms = elapsed_millis(&started),
                latest_tick = ?out.latest_tick,
                rows = out.row_count,
                "report completed"
            ),
            Err(err) => tracing::error!(
                elapsed_ms = elapsed_millis(&started),
                error = %err,
                "report failed"
            ),
        }
        result
    }
}

fn elapsed_millis(started: &Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn log_report_stage(stage: &'static str, started: &Instant, rows: usize) {
    tracing::debug!(
        stage,
        elapsed_ms = elapsed_millis(started),
        rows,
        "report stage completed"
    );
}

/// `metric-summary`: a per-metric distribution summary over a finished run.
///
/// This is the first report to put the native [`stats`] module (bd-2z0.11.6) to work on real
/// persisted data: for every metric present in the newest bounded SQL page, it reports n, mean,
/// standard deviation, the 5/50/95 quantiles, min/max, and the coefficient of variation — the
/// foundation of the `distribution-report` (bd-2z0.11.6 item 2). Distribution FITTING (the
/// candidate normal/lognormal/gamma fits + KS test) is the piece where `fsci`'s distribution zoo
/// would earn its keep and is left for the adapter decision (bd-2z0.11.3); the summary itself
/// needs nothing beyond the native estimators.
struct MetricSummary;

#[derive(Debug, Serialize)]
struct MetricSummaryMachine {
    metrics: Vec<MetricSummaryRow>,
}

#[derive(Debug, Serialize)]
struct MetricSummaryRow {
    name: String,
    n: usize,
    mean: f64,
    std_dev: f64,
    min: f64,
    q05: f64,
    median: f64,
    q95: f64,
    max: f64,
    /// `std_dev / |mean|` — a scale-free measure of spread. `None` when the mean is within
    /// `f64::EPSILON` of zero, where the ratio is meaningless rather than merely large.
    coefficient_of_variation: Option<f64>,
}

impl Report for MetricSummary {
    fn name(&self) -> &'static str {
        "metric-summary"
    }

    fn description(&self) -> &'static str {
        "Per-metric distribution summary over the newest bounded row page of a finished run"
    }

    fn run(&self, cx: &ReaderCtx, _params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let read_started = Instant::now();
        let readings = cx.reader.recent_metrics(METRIC_SUMMARY_ROW_LIMIT)?;
        log_report_stage("read", &read_started, readings.len());

        let render_started = Instant::now();
        // Group values by metric name. BTreeMap keeps the output in a stable, name-sorted order so
        // two runs of the report over the same data render identically — a report whose row order
        // wobbled could not be diffed across runs.
        let mut by_metric: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for PersistedMetric { name, value, .. } in readings {
            by_metric.entry(name).or_default().push(value);
        }

        let mut rows = Vec::with_capacity(by_metric.len());
        for (name, values) in by_metric {
            // A metric with non-finite values is a real problem worth surfacing, but the stats
            // functions already reject it; map that to a report-level error rather than a panic.
            let mean = stats::mean(&values).map_err(|error| metric_stats_error(&error))?;
            let std_dev = stats::std_dev(&values).map_err(|error| metric_stats_error(&error))?;
            let q05 = stats::quantile(&values, 0.05).map_err(|error| metric_stats_error(&error))?;
            let median =
                stats::quantile(&values, 0.50).map_err(|error| metric_stats_error(&error))?;
            let q95 = stats::quantile(&values, 0.95).map_err(|error| metric_stats_error(&error))?;
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let coefficient_of_variation =
                (mean.abs() > f64::EPSILON).then(|| std_dev / mean.abs());
            rows.push(MetricSummaryRow {
                name,
                n: values.len(),
                mean,
                std_dev,
                min,
                q05,
                median,
                q95,
                max,
                coefficient_of_variation,
            });
        }

        let machine = MetricSummaryMachine { metrics: rows };

        let mut md = String::new();
        let _ = writeln!(md, "# Metric summary\n");
        if machine.metrics.is_empty() {
            let _ = writeln!(md, "_No metrics persisted in this run._");
        } else {
            let _ = writeln!(
                md,
                "| metric | n | mean | sd | min | p05 | median | p95 | max | CV |"
            );
            let _ = writeln!(md, "|---|---|---|---|---|---|---|---|---|---|");
            for row in &machine.metrics {
                let _ = writeln!(
                    md,
                    "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {} |",
                    row.name,
                    row.n,
                    row.mean,
                    row.std_dev,
                    row.min,
                    row.q05,
                    row.median,
                    row.q95,
                    row.max,
                    row.coefficient_of_variation
                        .map_or_else(|| "-".to_owned(), |cv| format!("{cv:.4}")),
                );
            }
        }

        let output = base_output(
            self.name(),
            cx,
            machine.metrics.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

/// Map a statistics error to a report error. The stats module only errors on genuinely bad data
/// (empty or non-finite), which for persisted metrics means the run wrote something impossible —
/// a report-level failure, not a panic.
fn metric_stats_error(error: &crate::stats::StatsError) -> AnalyticsError {
    AnalyticsError::Storage(StorageError::InvalidData {
        context: "analytics.metric_summary",
        reason: error.to_string(),
    })
}

/// `metric-changepoints`: find and CERTIFY the most prominent regime shift in each metric.
///
/// This is the certification pipeline (bd-2z0.11.6) run over real persisted data. For every metric
/// the run recorded, it locates the single largest mean shift ([`changepoint::largest_shift`]),
/// certifies it — permutation test, bootstrap CI on the shift, effect sizes — and applies
/// Benjamini-Hochberg across all metrics, so a run with a dozen metrics cannot report a chance
/// "regime change" it never had. It answers "which metrics genuinely shifted in this run, and
/// when?" with statistics rather than an eyeballed threshold.
///
/// Distinct from `scriptbots-core::detect`, which is the ONLINE detector: this is the offline
/// certification of shifts, over the finished series.
struct MetricChangepoints;

#[derive(Debug, Serialize)]
struct ChangepointsMachine {
    /// Certification window (samples on each side of the shift).
    window: usize,
    /// Target false-discovery rate for the across-metrics correction.
    fdr: f64,
    /// Metrics whose series was long enough to admit a certified change-point.
    metrics_examined: usize,
    /// How many of those hold up under FDR control — the honest count of real regime shifts.
    significant: usize,
    /// True when the bounded metric read hit its cap, so early history was not analysed and a
    /// change-point in it would be invisible. A truncated analysis must not read as a complete one.
    truncated: bool,
    changepoints: Vec<ChangepointRow>,
}

#[derive(Debug, Serialize)]
struct ChangepointRow {
    metric: String,
    /// The tick at which the new regime begins.
    change_tick: u64,
    shift: f64,
    before_mean: f64,
    after_mean: f64,
    p_value: f64,
    ci_lower: f64,
    ci_upper: f64,
    cohens_d: f64,
    cliffs_delta: f64,
    /// Survives Benjamini-Hochberg across the run's metrics. The field to act on.
    significant_fdr: bool,
}

struct ChangepointCandidate {
    metric: String,
    change_tick: u64,
    shift: f64,
    before_mean: f64,
    after_mean: f64,
    certification: certify::EventCertification,
}

fn certify_metric_changepoints(
    by_metric: BTreeMap<String, Vec<(u64, f64)>>,
    window: usize,
    cert_params: &certify::CertificationParams,
) -> Result<Vec<ChangepointCandidate>, AnalyticsError> {
    let mut candidates = Vec::new();
    for (metric, mut points) in by_metric {
        points.sort_by_key(|(tick, _)| *tick);
        let series: Vec<f64> = points.iter().map(|(_, value)| *value).collect();
        let Some(cp) = changepoint::largest_shift(&series, window) else {
            continue;
        };
        let certification = certify::certify_event(&series, cp.index, cert_params)
            .map_err(|error| metric_stats_error(&error))?;
        candidates.push(ChangepointCandidate {
            metric,
            change_tick: points[cp.index].0,
            shift: cp.shift,
            before_mean: cp.before_mean,
            after_mean: cp.after_mean,
            certification,
        });
    }
    Ok(candidates)
}

fn render_changepoints_markdown(machine: &ChangepointsMachine) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Metric change-points\n");
    if machine.truncated {
        let _ = writeln!(
            md,
            "> **Note:** the metric read hit its {METRIC_SUMMARY_ROW_LIMIT}-row cap; early \
             history was not analysed and a shift in it is not reported here.\n"
        );
    }
    let _ = writeln!(
        md,
        "_window={}, FDR={}, {} of {} metrics show a certified regime shift._\n",
        machine.window, machine.fdr, machine.significant, machine.metrics_examined
    );
    if machine.changepoints.is_empty() {
        let _ = writeln!(
            md,
            "_No metric series was long enough to certify a change-point._"
        );
        return md;
    }

    let _ = writeln!(md, "| metric | tick | shift | p | 95% CI | d | δ | real? |");
    let _ = writeln!(md, "|---|---|---|---|---|---|---|---|");
    for row in &machine.changepoints {
        let _ = writeln!(
            md,
            "| {} | {} | {:+.4} | {:.4} | [{:.3}, {:.3}] | {:.3} | {:.3} | {} |",
            row.metric,
            row.change_tick,
            row.shift,
            row.p_value,
            row.ci_lower,
            row.ci_upper,
            row.cohens_d,
            row.cliffs_delta,
            if row.significant_fdr { "yes" } else { "no" },
        );
    }
    md
}

impl Report for MetricChangepoints {
    fn name(&self) -> &'static str {
        "metric-changepoints"
    }

    fn description(&self) -> &'static str {
        "Find and statistically certify the largest regime shift in each metric (FDR-controlled)"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let window = params.get_usize("window")?.unwrap_or(30);
        if window == 0 {
            return Err(AnalyticsError::BadParam {
                name: "window".to_owned(),
                reason: "must be at least 1".to_owned(),
            });
        }

        let read_started = Instant::now();
        let readings = cx.reader.recent_metrics(METRIC_SUMMARY_ROW_LIMIT)?;
        // `recent_metrics` is a bounded most-recent-N read. If it came back full, earlier history
        // was dropped and a change-point in it is invisible — say so rather than let a truncated
        // analysis read as a complete one.
        let truncated = readings.len() >= METRIC_SUMMARY_ROW_LIMIT;
        log_report_stage("read", &read_started, readings.len());

        let render_started = Instant::now();
        // Group into ordered (tick, value) series per metric. recent_metrics order is not
        // contractually chronological here, so we sort by tick explicitly — a change-point over a
        // mis-ordered series would be meaningless.
        let mut by_metric: BTreeMap<String, Vec<(u64, f64)>> = BTreeMap::new();
        for PersistedMetric { tick, name, value } in readings {
            by_metric.entry(name).or_default().push((tick, value));
        }

        let cert_params = certify::CertificationParams {
            window,
            fdr: params
                .get("fdr")
                .map(str::parse::<f64>)
                .transpose()
                .map_err(|e| AnalyticsError::BadParam {
                    name: "fdr".to_owned(),
                    reason: e.to_string(),
                })?
                .unwrap_or(0.05),
            ..certify::CertificationParams::default()
        };

        // The window doubles as `min_segment`, so every located shift leaves a full certification
        // window on both sides and `certify_event` cannot see an out-of-range window.
        let candidates = certify_metric_changepoints(by_metric, window, &cert_params)?;

        // Second pass: Benjamini-Hochberg across every metric's p-value at once.
        let p_values: Vec<f64> = candidates.iter().map(|c| c.certification.p_value).collect();
        let rejected = certify::benjamini_hochberg(&p_values, cert_params.fdr);

        let mut rows = Vec::with_capacity(candidates.len());
        let mut significant = 0usize;
        for (candidate, &is_rejected) in candidates.into_iter().zip(&rejected) {
            if is_rejected {
                significant += 1;
            }
            rows.push(ChangepointRow {
                metric: candidate.metric,
                change_tick: candidate.change_tick,
                shift: candidate.shift,
                before_mean: candidate.before_mean,
                after_mean: candidate.after_mean,
                p_value: candidate.certification.p_value,
                ci_lower: candidate.certification.shift_ci.lower,
                ci_upper: candidate.certification.shift_ci.upper,
                cohens_d: candidate.certification.cohens_d,
                cliffs_delta: candidate.certification.cliffs_delta,
                significant_fdr: is_rejected,
            });
        }

        let machine = ChangepointsMachine {
            window,
            fdr: cert_params.fdr,
            metrics_examined: rows.len(),
            significant,
            truncated,
            changepoints: rows,
        };

        let md = render_changepoints_markdown(&machine);

        let output = base_output(
            self.name(),
            cx,
            machine.changepoints.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

/// `compare-runs`: paired treatment-effect comparison of two run databases (serves bd-16g.1.4).
///
/// Given a control run (the database this report runs against) and a `treatment_db=<path>`, it
/// measures whether the treatment shifted each metric the two runs share. The runs are assumed to
/// share seeds, so each metric is compared TICK-ALIGNED — the control and treatment values at the
/// same tick form a matched pair — and the pairing is fed to [`compare`], which applies a
/// sign-flip permutation test, a paired-bootstrap CI, Cohen's `d_z`, and Benjamini-Hochberg across
/// the metrics. It is the DB-facing glue for the matched-seed statistics; the pure analysis was
/// proven in isolation, this wires it to two real databases.
struct RunComparison;

#[derive(Debug, Serialize)]
struct RunComparisonMachine {
    /// The treatment database this control run was compared against (provenance).
    treatment_db: String,
    /// Target false-discovery rate for the across-metrics correction.
    fdr: f64,
    /// Metrics present in BOTH runs with enough tick-aligned pairs to compare.
    metrics_compared: usize,
    /// How many hold up under FDR control — the honest count of real treatment effects.
    significant: usize,
    /// True when either bounded metric read hit its cap, so the comparison is over recent history
    /// rather than the whole run.
    truncated: bool,
    metrics: Vec<RunComparisonRow>,
}

#[derive(Debug, Serialize)]
struct RunComparisonRow {
    metric: String,
    /// Number of tick-aligned matched pairs.
    n_pairs: usize,
    /// Mean of `treatment - control` over the matched pairs. The treatment-effect estimate.
    mean_difference: f64,
    ci_lower: f64,
    ci_upper: f64,
    p_value: f64,
    /// Paired standardized effect size (`d_z`).
    cohens_dz: f64,
    /// Fraction of pairs where treatment exceeded control.
    fraction_positive: f64,
    /// Survives Benjamini-Hochberg across the run's shared metrics. The field to act on.
    significant_fdr: bool,
}

impl Report for RunComparison {
    fn name(&self) -> &'static str {
        "compare-runs"
    }

    fn description(&self) -> &'static str {
        "Paired treatment-effect comparison of two run databases (tick-aligned, FDR-controlled)"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let treatment_path =
            params
                .get("treatment_db")
                .ok_or_else(|| AnalyticsError::BadParam {
                    name: "treatment_db".to_owned(),
                    reason:
                        "compare-runs requires treatment_db=<path> (the treatment run database)"
                            .to_owned(),
                })?;
        let fdr = params
            .get("fdr")
            .map(str::parse::<f64>)
            .transpose()
            .map_err(|e| AnalyticsError::BadParam {
                name: "fdr".to_owned(),
                reason: e.to_string(),
            })?
            .unwrap_or(0.05);

        let read_started = Instant::now();
        let control_readings = cx.reader.recent_metrics(METRIC_SUMMARY_ROW_LIMIT)?;
        let treatment_reader = StorageReader::open_finished(treatment_path)?;
        let treatment_readings = treatment_reader.recent_metrics(METRIC_SUMMARY_ROW_LIMIT)?;
        let truncated = control_readings.len() >= METRIC_SUMMARY_ROW_LIMIT
            || treatment_readings.len() >= METRIC_SUMMARY_ROW_LIMIT;
        log_report_stage(
            "read",
            &read_started,
            control_readings.len() + treatment_readings.len(),
        );

        let render_started = Instant::now();
        // metric -> (tick -> value), for each run. BTreeMap so tick order and metric order are
        // stable and the tick intersection below is straightforward.
        let mut control: BTreeMap<String, BTreeMap<u64, f64>> = BTreeMap::new();
        for PersistedMetric { tick, name, value } in control_readings {
            control.entry(name).or_default().insert(tick, value);
        }
        let mut treatment: BTreeMap<String, BTreeMap<u64, f64>> = BTreeMap::new();
        for PersistedMetric { tick, name, value } in treatment_readings {
            treatment.entry(name).or_default().insert(tick, value);
        }

        // For every metric present in BOTH runs, pair the values at ticks the two runs share. At
        // least three pairs are required — a paired test on one or two points is noise.
        struct Paired {
            name: String,
            control: Vec<f64>,
            treatment: Vec<f64>,
        }
        let mut paired: Vec<Paired> = Vec::new();
        for (name, control_ticks) in &control {
            let Some(treatment_ticks) = treatment.get(name) else {
                continue; // metric only present in one run — nothing to compare
            };
            let mut control_values = Vec::new();
            let mut treatment_values = Vec::new();
            for (tick, control_value) in control_ticks {
                if let Some(treatment_value) = treatment_ticks.get(tick) {
                    control_values.push(*control_value);
                    treatment_values.push(*treatment_value);
                }
            }
            if control_values.len() >= 3 {
                paired.push(Paired {
                    name: name.clone(),
                    control: control_values,
                    treatment: treatment_values,
                });
            }
        }

        let compare_params = compare::CompareParams {
            fdr,
            ..compare::CompareParams::default()
        };
        // `series` borrows `paired`, which outlives it and the compare_metrics call below.
        // `as_str`/`as_slice` are explicit rather than relying on `&String`/`&Vec` coercion at the
        // struct-field site.
        let series: Vec<compare::MetricSeries<'_>> = paired
            .iter()
            .map(|p| compare::MetricSeries {
                name: p.name.as_str(),
                control: p.control.as_slice(),
                treatment: p.treatment.as_slice(),
            })
            .collect();
        let study = compare::compare_metrics(&series, &compare_params)
            .map_err(|e| metric_stats_error(&e))?;

        let mut rows = Vec::with_capacity(study.metrics.len());
        let mut significant = 0usize;
        for named in &study.metrics {
            let c = &named.comparison;
            if c.significant_fdr {
                significant += 1;
            }
            rows.push(RunComparisonRow {
                metric: named.metric.clone(),
                n_pairs: c.n_pairs,
                mean_difference: c.mean_difference,
                ci_lower: c.difference_ci.lower,
                ci_upper: c.difference_ci.upper,
                p_value: c.p_value,
                cohens_dz: c.cohens_dz,
                fraction_positive: c.fraction_positive,
                significant_fdr: c.significant_fdr,
            });
        }

        let machine = RunComparisonMachine {
            treatment_db: treatment_path.to_owned(),
            fdr,
            metrics_compared: rows.len(),
            significant,
            truncated,
            metrics: rows,
        };

        let mut md = String::new();
        let _ = writeln!(md, "# Run comparison\n");
        let _ = writeln!(
            md,
            "_treatment=`{}`, FDR={}, {} of {} shared metrics show a certified treatment effect._\n",
            machine.treatment_db, machine.fdr, machine.significant, machine.metrics_compared
        );
        if machine.truncated {
            let _ = writeln!(
                md,
                "> **Note:** a metric read hit its {METRIC_SUMMARY_ROW_LIMIT}-row cap; the \
                 comparison is over recent history, not the whole run.\n"
            );
        }
        if machine.metrics.is_empty() {
            let _ = writeln!(
                md,
                "_No metric was present in both runs with enough matched ticks._"
            );
        } else {
            let _ = writeln!(
                md,
                "| metric | pairs | Δ (treat−ctrl) | 95% CI | p | d_z | +frac | real? |"
            );
            let _ = writeln!(md, "|---|---|---|---|---|---|---|---|");
            for row in &machine.metrics {
                let _ = writeln!(
                    md,
                    "| {} | {} | {:+.4} | [{:.3}, {:.3}] | {:.4} | {:.3} | {:.2} | {} |",
                    row.metric,
                    row.n_pairs,
                    row.mean_difference,
                    row.ci_lower,
                    row.ci_upper,
                    row.p_value,
                    row.cohens_dz,
                    row.fraction_positive,
                    if row.significant_fdr { "yes" } else { "no" },
                );
            }
        }

        let output = base_output(
            self.name(),
            cx,
            machine.metrics.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

/// `metric-distribution`: per-metric shape and normality (bd-2z0.11.6 item 2).
///
/// For every metric the run recorded, reports its skewness and excess kurtosis and runs a
/// Jarque-Bera normality test ([`distribution`]) — a native, `erf`-free assessment of "is this
/// metric normal, and how is it shaped?". A skewed or heavy-tailed metric is exactly the case
/// where a mean-and-SD summary (the `metric-summary` report) understates the story, so this is its
/// companion. Full distribution FITTING (candidate lognormal/gamma) stays with the `fsci` adapter
/// decision (bd-2z0.11.3).
struct MetricDistribution;

#[derive(Debug, Serialize)]
struct MetricDistributionMachine {
    /// Significance level for the normality verdict.
    alpha: f64,
    /// True when the bounded metric read hit its cap.
    truncated: bool,
    /// Metrics with at least four values (the minimum for a shape test).
    metrics_examined: usize,
    /// How many were flagged non-normal at `alpha`.
    non_normal: usize,
    metrics: Vec<MetricDistributionRow>,
}

#[derive(Debug, Serialize)]
struct MetricDistributionRow {
    name: String,
    n: usize,
    mean: f64,
    std_dev: f64,
    skewness: f64,
    excess_kurtosis: f64,
    jarque_bera: f64,
    jb_p_value: f64,
    /// A constant metric: no shape, reported as such rather than as "looks normal".
    degenerate: bool,
    /// Jarque-Bera rejects normality at `alpha`. Never true for a degenerate metric.
    non_normal: bool,
}

impl Report for MetricDistribution {
    fn name(&self) -> &'static str {
        "metric-distribution"
    }

    fn description(&self) -> &'static str {
        "Per-metric shape (skewness, kurtosis) and a Jarque-Bera normality test over a finished run"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let alpha = params
            .get("alpha")
            .map(str::parse::<f64>)
            .transpose()
            .map_err(|e| AnalyticsError::BadParam {
                name: "alpha".to_owned(),
                reason: e.to_string(),
            })?
            .unwrap_or(0.05);

        let read_started = Instant::now();
        let readings = cx.reader.recent_metrics(METRIC_SUMMARY_ROW_LIMIT)?;
        let truncated = readings.len() >= METRIC_SUMMARY_ROW_LIMIT;
        log_report_stage("read", &read_started, readings.len());

        let render_started = Instant::now();
        let mut by_metric: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for PersistedMetric { name, value, .. } in readings {
            by_metric.entry(name).or_default().push(value);
        }

        let mut rows = Vec::with_capacity(by_metric.len());
        let mut non_normal = 0usize;
        for (name, values) in by_metric {
            // The shape test needs at least four points; a shorter series is skipped rather than
            // reported with a meaningless statistic.
            if values.len() < 4 {
                continue;
            }
            let summary = distribution::summarize(&values).map_err(|e| metric_stats_error(&e))?;
            let is_non_normal = summary.rejects_normality(alpha);
            if is_non_normal {
                non_normal += 1;
            }
            rows.push(MetricDistributionRow {
                name,
                n: summary.n,
                mean: summary.mean,
                std_dev: summary.std_dev,
                skewness: summary.skewness,
                excess_kurtosis: summary.excess_kurtosis,
                jarque_bera: summary.jarque_bera,
                jb_p_value: summary.jb_p_value,
                degenerate: summary.degenerate,
                non_normal: is_non_normal,
            });
        }

        let machine = MetricDistributionMachine {
            alpha,
            truncated,
            metrics_examined: rows.len(),
            non_normal,
            metrics: rows,
        };

        let mut md = String::new();
        let _ = writeln!(md, "# Metric distributions\n");
        let _ = writeln!(
            md,
            "_alpha={}, {} of {} metrics flagged non-normal (Jarque-Bera)._\n",
            machine.alpha, machine.non_normal, machine.metrics_examined
        );
        if machine.truncated {
            let _ = writeln!(
                md,
                "> **Note:** the metric read hit its {METRIC_SUMMARY_ROW_LIMIT}-row cap; the shape \
                 is over recent history, not the whole run.\n"
            );
        }
        if machine.metrics.is_empty() {
            let _ = writeln!(md, "_No metric had at least four values to characterize._");
        } else {
            let _ = writeln!(
                md,
                "| metric | n | mean | sd | skew | ex.kurt | JB | p | normal? |"
            );
            let _ = writeln!(md, "|---|---|---|---|---|---|---|---|---|");
            for row in &machine.metrics {
                let verdict = if row.degenerate {
                    "constant"
                } else if row.non_normal {
                    "no"
                } else {
                    "yes"
                };
                let _ = writeln!(
                    md,
                    "| {} | {} | {:.4} | {:.4} | {:+.3} | {:+.3} | {:.2} | {:.4} | {} |",
                    row.name,
                    row.n,
                    row.mean,
                    row.std_dev,
                    row.skewness,
                    row.excess_kurtosis,
                    row.jarque_bera,
                    row.jb_p_value,
                    verdict,
                );
            }
        }

        let output = base_output(
            self.name(),
            cx,
            machine.metrics.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

fn base_output(
    name: &str,
    cx: &ReaderCtx,
    row_count: usize,
    machine: serde_json::Value,
    human_md: String,
) -> Result<ReportOutput, AnalyticsError> {
    Ok(ReportOutput {
        schema_version: REPORT_SCHEMA_VERSION,
        report: name.to_owned(),
        db_path: cx.db_path.clone(),
        latest_tick: cx.reader.max_tick()?,
        row_count,
        machine,
        human_md,
    })
}

/// `run-summary`: lifecycle totals and bounded recent population trajectory statistics.
struct RunSummary;

#[derive(Debug, Serialize)]
struct RunSummaryMachine {
    tick_count: u64,
    birth_records: u64,
    death_records: u64,
    population_first: Option<usize>,
    population_last: Option<usize>,
    population_min: Option<usize>,
    population_max: Option<usize>,
    population_mean: Option<f64>,
    total_energy_first: Option<f64>,
    total_energy_last: Option<f64>,
    watermarks: WatermarksMachine,
}

#[derive(Debug, Serialize)]
struct WatermarksMachine {
    admitted: Option<u64>,
    applied: Option<u64>,
    durable: Option<u64>,
}

impl Report for RunSummary {
    fn name(&self) -> &'static str {
        "run-summary"
    }

    fn description(&self) -> &'static str {
        "Lifecycle totals, recent bounded population trajectory stats, and persistence watermarks"
    }

    fn run(&self, cx: &ReaderCtx, _params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let read_started = Instant::now();
        let ledger = cx.reader.run_ledger_summary()?;
        // StorageReader returns the newest bounded page in chronological order.
        let ticks = cx.reader.recent_ticks(RUN_SUMMARY_TICK_LIMIT)?;
        let watermarks = cx.reader.persistence_watermarks()?;
        log_report_stage("read", &read_started, ticks.len());

        let render_started = Instant::now();
        let mut population_first = None;
        let mut population_last = None;
        let mut population_min = None;
        let mut population_max = None;
        let mut population_mean = 0.0_f64;
        let mut population_count = 0_u64;
        for tick in &ticks {
            population_first.get_or_insert(tick.agent_count);
            population_last = Some(tick.agent_count);
            population_min = Some(
                population_min.map_or(tick.agent_count, |value: usize| value.min(tick.agent_count)),
            );
            population_max = Some(
                population_max.map_or(tick.agent_count, |value: usize| value.max(tick.agent_count)),
            );
            population_count += 1;
            #[allow(clippy::cast_precision_loss)]
            let observation = tick.agent_count as f64;
            #[allow(clippy::cast_precision_loss)]
            let count = population_count as f64;
            population_mean += (observation - population_mean) / count;
        }
        let machine = RunSummaryMachine {
            tick_count: ledger.tick_count,
            birth_records: ledger.birth_records,
            death_records: ledger.death_records,
            population_first,
            population_last,
            population_min,
            population_max,
            population_mean: (population_count > 0).then_some(population_mean),
            total_energy_first: ticks.first().map(|t| t.total_energy),
            total_energy_last: ticks.last().map(|t| t.total_energy),
            watermarks: WatermarksMachine {
                admitted: watermarks.admitted.map(PersistenceBatchId::get),
                applied: watermarks.applied.map(PersistenceBatchId::get),
                durable: watermarks.durable.map(PersistenceBatchId::get),
            },
        };

        let mut md = String::new();
        let _ = writeln!(md, "# Run summary\n");
        let _ = writeln!(md, "| field | value |");
        let _ = writeln!(md, "|---|---|");
        let _ = writeln!(md, "| ticks persisted | {} |", machine.tick_count);
        let _ = writeln!(
            md,
            "| births / deaths | {} / {} |",
            machine.birth_records, machine.death_records
        );
        let _ = writeln!(
            md,
            "| recent-window population first→last (min/mean/max) | {:?}→{:?} ({:?}/{}/{:?}) |",
            machine.population_first,
            machine.population_last,
            machine.population_min,
            machine
                .population_mean
                .map_or_else(|| "-".into(), |m| format!("{m:.1}")),
            machine.population_max,
        );
        let _ = writeln!(
            md,
            "| total energy first→last | {:?}→{:?} |",
            machine.total_energy_first, machine.total_energy_last
        );

        let output = base_output(
            self.name(),
            cx,
            ticks.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

/// `narrative-timeline`: bounded page of the latest events for a finished run.
///
/// v1 renders aggregate kind counts plus a newest-first SQL page returned in
/// chronological order. When the
/// typed narrative tables land (bd-16g.2.2) and FTS search (bd-16g.2.6),
/// this report upgrades to the `run_events` stream + BM25 search parameters
/// WITHOUT changing its registry name; the machine schema will bump
/// [`REPORT_SCHEMA_VERSION`] per the documented migration policy.
struct NarrativeTimeline;

#[derive(Debug, Serialize)]
struct TimelineMachine {
    event_counts: Vec<(String, u64)>,
    events: Vec<TimelineRow>,
    truncated_to: Option<usize>,
}

#[derive(Debug, Serialize)]
struct TimelineRow {
    tick: u64,
    seq: u64,
    event: serde_json::Value,
}

impl Report for NarrativeTimeline {
    fn name(&self) -> &'static str {
        "narrative-timeline"
    }

    fn description(&self) -> &'static str {
        "Chronological replay-event timeline (upgrades to typed narrative events when bd-16g.2.2 lands)"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let limit = narrative_timeline_limit(params)?;
        let read_started = Instant::now();
        let counts = cx.reader.replay_event_counts()?;
        let events = cx.reader.recent_replay_events(limit)?;
        let total = counts
            .iter()
            .fold(0_u64, |sum, count| sum.saturating_add(count.count));
        log_report_stage("read", &read_started, events.len());

        let render_started = Instant::now();
        let machine = TimelineMachine {
            event_counts: counts
                .iter()
                .map(|c| (c.event_type.clone(), c.count))
                .collect(),
            events: events
                .iter()
                .map(|e| {
                    Ok(TimelineRow {
                        tick: e.tick,
                        seq: e.seq,
                        event: serde_json::to_value(&e.event)?,
                    })
                })
                .collect::<Result<Vec<_>, AnalyticsError>>()?,
            truncated_to: (total > u64::try_from(limit).unwrap_or(u64::MAX)).then_some(limit),
        };

        let mut md = String::new();
        let _ = writeln!(
            md,
            "# Narrative timeline (latest bounded replay-event page, v1)\n"
        );
        if machine.events.is_empty() {
            let _ = writeln!(md, "_No replay events persisted in this run._");
        } else {
            let _ = writeln!(md, "| tick | seq | event |");
            let _ = writeln!(md, "|---|---|---|");
            for row in &machine.events {
                let _ = writeln!(md, "| {} | {} | `{}` |", row.tick, row.seq, row.event);
            }
            if let Some(t) = machine.truncated_to {
                let _ = writeln!(md, "\n_…showing {t} of {total} events (bounded SQL page)._");
            }
        }

        let output = base_output(
            self.name(),
            cx,
            machine.events.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

fn narrative_timeline_limit(params: &ReportParams) -> Result<usize, AnalyticsError> {
    let limit = params
        .get_usize("limit")?
        .unwrap_or(NARRATIVE_TIMELINE_DEFAULT_LIMIT);
    if limit > NARRATIVE_TIMELINE_MAX_LIMIT {
        return Err(AnalyticsError::BadParam {
            name: "limit".to_owned(),
            reason: format!("must be at most {NARRATIVE_TIMELINE_MAX_LIMIT}"),
        });
    }
    Ok(limit)
}
