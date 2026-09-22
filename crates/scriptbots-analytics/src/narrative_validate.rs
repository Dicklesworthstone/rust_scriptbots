//! Statistical certification of narrative events from a run database (`bd-2z0.11.6`, item 1).
//!
//! Provides the `narrative-validate` report: for each persisted [`scriptbots_core::narrative::EventRecord`],
//! it extracts the metric series surrounding the event, computes moving-block bootstrap confidence intervals
//! and permutation significance tests against a stationary null, applies Benjamini-Hochberg false discovery rate
//! (FDR) control across all evaluated events, and emits typed machine JSON and Markdown summary tables.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Instant;

use scriptbots_storage::PersistedMetric;
use serde::{Deserialize, Serialize};

use crate::certify::{self, CertificationParams};
use crate::stats::{EffectSizeEstimator, StatsError};
use crate::{
    AnalyticsError, ReaderCtx, Report, ReportOutput, ReportParams, base_output, log_report_stage,
    metric_stats_error,
};

/// Maximum rows read from the metrics table when validating narrative events.
const METRIC_READ_ROW_LIMIT: usize = 4_096;

/// Default event limit when not explicitly overridden via parameters.
const DEFAULT_EVENT_LIMIT: usize = 1_000;

/// `narrative-validate`: offline statistical certification of detected narrative events.
#[derive(Debug, Default)]
pub struct NarrativeValidate;

/// Machine-readable report payload for `narrative-validate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NarrativeValidateMachine {
    /// Analysis window (number of ticks before and after the event).
    pub window: usize,
    /// Target False Discovery Rate for Benjamini-Hochberg multiple-testing control.
    pub fdr: f64,
    /// Moving-block length used in block bootstrap resampling.
    pub block_len: usize,
    /// Total events examined and statistically tested.
    pub events_examined: usize,
    /// Number of events certified significant under FDR control.
    pub significant: usize,
    /// Number of events skipped because metric data was unavailable or window was out-of-range.
    pub skipped: usize,
    /// True if the bounded metric read hit the maximum row limit.
    pub truncated: bool,
    /// Detailed certification rows for tested events.
    pub events: Vec<NarrativeValidateRow>,
}

/// Detailed certification results for a single evaluated narrative event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NarrativeValidateRow {
    /// Simulation tick at which the detector fired.
    pub tick: u64,
    /// Classification of the narrative event.
    pub kind: String,
    /// Metric series evaluated (e.g., "population", "`average_energy`").
    pub metric: String,
    /// Deterministic templated event narrative description.
    pub human_text: String,
    /// Mean difference (`after - before`) across the evaluation windows.
    pub shift: f64,
    /// Lower bound of the bootstrap confidence interval on the shift.
    pub ci_lower: f64,
    /// Upper bound of the bootstrap confidence interval on the shift.
    pub ci_upper: f64,
    /// Two-sided permutation test p-value against the stationary null.
    pub p_value: f64,
    /// Parametric standardized effect size (Cohen's d).
    pub cohens_d: f64,
    /// Nonparametric rank effect size (Cliff's delta) in [-1, 1].
    pub cliffs_delta: f64,
    /// Which effect size estimator was used.
    pub effect_estimator: String,
    /// True if p-value < alpha prior to multiple testing correction.
    pub significant_uncorrected: bool,
    /// True if the event survives Benjamini-Hochberg FDR control across all tested events.
    pub significant_fdr: bool,
}

struct CandidateValidation {
    tick: u64,
    kind: String,
    metric: String,
    human_text: String,
    shift: f64,
    ci_lower: f64,
    ci_upper: f64,
    p_value: f64,
    cohens_d: f64,
    cliffs_delta: f64,
    effect_estimator: EffectSizeEstimator,
    significant_uncorrected: bool,
}

fn render_narrative_validate_markdown(machine: &NarrativeValidateMachine) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Narrative event statistical certification\n");
    if machine.truncated {
        let _ = writeln!(
            md,
            "> **Note:** metric history read reached the {METRIC_READ_ROW_LIMIT}-row limit; early events may lack surrounding data.\n"
        );
    }
    let _ = writeln!(
        md,
        "_window={}, FDR={}, {} of {} events certified significant ({} skipped)._\n",
        machine.window, machine.fdr, machine.significant, machine.events_examined, machine.skipped
    );

    if machine.events.is_empty() {
        let _ = writeln!(md, "_No narrative events could be certified in this run._");
        return md;
    }

    let _ = writeln!(
        md,
        "| tick | kind | metric | shift | 95% CI | p | d | δ | certified? | narrative |"
    );
    let _ = writeln!(md, "|---|---|---|---|---|---|---|---|---|---|");

    for row in &machine.events {
        let _ = writeln!(
            md,
            "| {} | {} | {} | {:+.4} | [{:.3}, {:.3}] | {:.4} | {:+.3} | {:+.3} | {} | {} |",
            row.tick,
            row.kind,
            row.metric,
            row.shift,
            row.ci_lower,
            row.ci_upper,
            row.p_value,
            row.cohens_d,
            row.cliffs_delta,
            if row.significant_fdr { "yes" } else { "no" },
            row.human_text,
        );
    }

    md
}

fn parse_params(params: &ReportParams) -> Result<(CertificationParams, usize), AnalyticsError> {
    let window = params.get_usize("window")?.unwrap_or(30);
    if window == 0 {
        return Err(AnalyticsError::BadParam {
            name: "window".to_owned(),
            reason: "must be at least 1".to_owned(),
        });
    }

    let fdr = params
        .get("fdr")
        .map(str::parse::<f64>)
        .transpose()
        .map_err(|e| AnalyticsError::BadParam {
            name: "fdr".to_owned(),
            reason: e.to_string(),
        })?
        .unwrap_or(0.05);

    let n_resamples = params.get_usize("resamples")?.unwrap_or(2000);
    let n_permutations = params.get_usize("permutations")?.unwrap_or(2000);
    let block_len = params.get_usize("block_len")?.unwrap_or(5);
    let limit = params.get_usize("limit")?.unwrap_or(DEFAULT_EVENT_LIMIT);

    Ok((
        CertificationParams {
            window,
            n_resamples,
            n_permutations,
            confidence: 0.95,
            fdr,
            block_len,
            seed: 0x5EED,
        },
        limit,
    ))
}

fn group_metrics_by_name(readings: Vec<PersistedMetric>) -> BTreeMap<String, Vec<(u64, f64)>> {
    let mut by_metric: BTreeMap<String, Vec<(u64, f64)>> = BTreeMap::new();
    for PersistedMetric { tick, name, value } in readings {
        by_metric.entry(name).or_default().push((tick, value));
    }
    for series in by_metric.values_mut() {
        series.sort_by_key(|entry| entry.0);
    }
    by_metric
}

fn evaluate_events(
    persisted_events: &[scriptbots_storage::PersistedRunEvent],
    by_metric: &BTreeMap<String, Vec<(u64, f64)>>,
    cert_params: &CertificationParams,
) -> Result<(Vec<CandidateValidation>, usize), AnalyticsError> {
    let mut candidates = Vec::new();
    let mut skipped = 0usize;

    for run_event in persisted_events {
        let event = run_event.record();
        let event_tick = event.tick.0;
        let metric_name = &event.metric;

        let Some(series) = by_metric.get(metric_name) else {
            skipped += 1;
            continue;
        };

        let event_idx = series
            .iter()
            .position(|entry| entry.0 == event_tick)
            .or_else(|| {
                series
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, entry)| entry.0.abs_diff(event_tick))
                    .map(|(i, _)| i)
            });

        let Some(idx) = event_idx else {
            skipped += 1;
            continue;
        };

        let values: Vec<f64> = series.iter().map(|entry| entry.1).collect();

        match certify::certify_event(&values, idx, cert_params) {
            Ok(certification) => {
                candidates.push(CandidateValidation {
                    tick: event_tick,
                    kind: format!("{:?}", event.kind),
                    metric: metric_name.clone(),
                    human_text: event.human_text.clone(),
                    shift: certification.shift_ci.point,
                    ci_lower: certification.shift_ci.lower,
                    ci_upper: certification.shift_ci.upper,
                    p_value: certification.p_value,
                    cohens_d: certification.cohens_d,
                    cliffs_delta: certification.cliffs_delta,
                    effect_estimator: certification.effect_estimator,
                    significant_uncorrected: certification.significant_uncorrected,
                });
            }
            Err(StatsError::EmptySample { .. }) => {
                skipped += 1;
            }
            Err(error) => {
                return Err(metric_stats_error(&error));
            }
        }
    }

    Ok((candidates, skipped))
}

impl Report for NarrativeValidate {
    fn name(&self) -> &'static str {
        "narrative-validate"
    }

    fn description(&self) -> &'static str {
        "Statistical certification of persisted narrative events (block-bootstrap CI + permutation test + FDR control)"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let (cert_params, limit) = parse_params(params)?;

        let read_started = Instant::now();
        let persisted_events = cx.reader.recent_run_events(limit)?;
        let metric_readings = cx.reader.recent_metrics(METRIC_READ_ROW_LIMIT)?;
        let truncated = metric_readings.len() >= METRIC_READ_ROW_LIMIT;
        log_report_stage("read", &read_started, persisted_events.len());

        let render_started = Instant::now();
        let by_metric = group_metrics_by_name(metric_readings);
        let (candidates, skipped) = evaluate_events(&persisted_events, &by_metric, &cert_params)?;

        let p_values: Vec<f64> = candidates.iter().map(|c| c.p_value).collect();
        let rejected = certify::benjamini_hochberg(&p_values, cert_params.fdr);

        let mut rows = Vec::with_capacity(candidates.len());
        let mut significant = 0usize;

        for (candidate, &is_significant) in candidates.into_iter().zip(&rejected) {
            if is_significant {
                significant += 1;
            }
            rows.push(NarrativeValidateRow {
                tick: candidate.tick,
                kind: candidate.kind,
                metric: candidate.metric,
                human_text: candidate.human_text,
                shift: candidate.shift,
                ci_lower: candidate.ci_lower,
                ci_upper: candidate.ci_upper,
                p_value: candidate.p_value,
                cohens_d: candidate.cohens_d,
                cliffs_delta: candidate.cliffs_delta,
                effect_estimator: format!("{:?}", candidate.effect_estimator),
                significant_uncorrected: candidate.significant_uncorrected,
                significant_fdr: is_significant,
            });
        }

        let machine = NarrativeValidateMachine {
            window: cert_params.window,
            fdr: cert_params.fdr,
            block_len: cert_params.block_len,
            events_examined: rows.len(),
            significant,
            skipped,
            truncated,
            events: rows,
        };

        let md = render_narrative_validate_markdown(&machine);

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
