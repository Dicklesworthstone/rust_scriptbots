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

/// Schema version stamped into every machine-readable report payload.
///
/// Bump ONLY with a migration note in the owning Bead/release evidence. Full
/// envelope goldens assert the value so an accidental schema change is loud.
pub const REPORT_SCHEMA_VERSION: u32 = 1;

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
/// persisted data: for every metric the run recorded, it loads the full value series and reports
/// n, mean, standard deviation, the 5/50/95 quantiles, min/max, and the coefficient of variation
/// — the foundation of the `distribution-report` (bd-2z0.11.6 item 2). Distribution FITTING (the
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
        "Per-metric distribution summary (n, mean, sd, quantiles, CV) over a finished run"
    }

    fn run(&self, cx: &ReaderCtx, _params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let read_started = Instant::now();
        let readings = cx.reader.recent_metrics(None)?;
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

/// `run-summary`: lifecycle totals and population trajectory statistics.
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
        "Lifecycle totals, population trajectory stats, and persistence watermarks"
    }

    fn run(&self, cx: &ReaderCtx, _params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let read_started = Instant::now();
        let ledger = cx.reader.run_ledger_summary()?;
        // StorageReader guarantees chronological order.
        let ticks = cx.reader.recent_ticks(None)?;
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
            "| population first→last (min/mean/max) | {:?}→{:?} ({:?}/{}/{:?}) |",
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

/// `narrative-timeline`: chronological event dump for a finished run.
///
/// v1 renders the replay-event stream (kind counts + ordered rows). When the
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
        let limit = params.get_usize("limit")?;
        let read_started = Instant::now();
        let counts = cx.reader.replay_event_counts()?;
        let mut events = cx.reader.load_replay_events()?;
        events.sort_by_key(|e| (e.tick, e.seq));
        let total = events.len();
        if let Some(limit) = limit {
            events.truncate(limit);
        }
        log_report_stage("read", &read_started, total);

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
            truncated_to: limit.filter(|l| *l < total),
        };

        let mut md = String::new();
        let _ = writeln!(md, "# Narrative timeline (replay events, v1)\n");
        if machine.events.is_empty() {
            let _ = writeln!(md, "_No replay events persisted in this run._");
        } else {
            let _ = writeln!(md, "| tick | seq | event |");
            let _ = writeln!(md, "|---|---|---|");
            for row in &machine.events {
                let _ = writeln!(md, "| {} | {} | `{}` |", row.tick, row.seq, row.event);
            }
            if let Some(t) = machine.truncated_to {
                let _ = writeln!(md, "\n_…truncated to {t} of {total} events (limit param)._");
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
