//! Statistical certification of narrative events from a run database (`bd-2z0.11.6`, item 1).
//!
//! The report reads the complete canonical per-tick detector input stream in bounded
//! pages. Certification windows are exact simulation ticks, independently of the
//! cadence and retention of dashboard metrics. Missing or corrupt detector evidence
//! is a typed storage refusal, never a substitute window of nearby metric samples.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::time::Instant;

use scriptbots_core::narrative::{
    METRIC_COMBAT, METRIC_ENERGY, METRIC_POPULATION, NarrativeInputV1,
};
use scriptbots_storage::{NarrativeInputBindingV1, PersistedRunEvent, StorageReader};
use serde::{Deserialize, Serialize};

use crate::certify::{self, CertificationParams};
use crate::stats::{EffectSizeEstimator, StatsError};
use crate::{
    AnalyticsError, ReaderCtx, Report, ReportOutput, ReportParams, base_output, log_report_stage,
    metric_stats_error, timeline_markdown_cell,
};

/// V2 certifies exact canonical tick windows and reports their complete source binding.
///
/// V1 sampled a capped dashboard-metric tail and used sample counts as ticks.
/// The `truncated` field now describes event selection, not detector-input coverage.
pub const NARRATIVE_VALIDATE_SCHEMA_VERSION: u32 = 2;

/// Rows fetched at a time; the complete run is traversed without retaining it.
const INPUT_READ_PAGE_SIZE: usize = 256;

/// Bound the retained before/after window independently of the run length.
const MAX_WINDOW_TICKS: usize = 4_096;

/// Default event limit when not explicitly overridden via parameters.
const DEFAULT_EVENT_LIMIT: usize = 1_000;

/// Reserve one row of the storage page to observe event-selection truncation.
const MAX_EVENT_LIMIT: usize = 4_095;

/// Offline statistical certification of persisted narrative events.
#[derive(Debug, Default)]
pub struct NarrativeValidate;

/// Machine-readable report payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NarrativeValidateMachine {
    /// Ticks per exact window: before `[tick-window, tick)`, after `[tick, tick+window)`.
    pub window: usize,
    /// Target False Discovery Rate for Benjamini-Hochberg multiple-testing control.
    pub fdr: f64,
    /// Moving-block length used in block bootstrap resampling.
    pub block_len: usize,
    /// Total events examined and statistically tested.
    pub events_examined: usize,
    /// Number of events certified significant under FDR control.
    pub significant: usize,
    /// Selected events with unsupported island/metric evidence or incomplete tick windows.
    pub skipped: usize,
    /// True when older events were excluded by the requested event limit.
    ///
    /// Detector inputs are never truncated to a newest-only metric window.
    pub truncated: bool,
    /// Canonical source binding, absent only when no events were selected.
    pub input_binding: Option<NarrativeInputBindingV1>,
    /// Exact number of canonical inputs read across the complete finished run.
    pub inputs_read: u64,
    /// Number of bounded detector-input pages consumed.
    pub input_pages: u64,
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
    /// Detector metric: `population`, `agents.energy`, `average_energy`, or `spike_hits`.
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
    /// True if the event survives Benjamini-Hochberg FDR control across tested events.
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

#[derive(Default)]
struct ValidationEvidence {
    candidates: Vec<CandidateValidation>,
    skipped: usize,
    input_binding: Option<NarrativeInputBindingV1>,
    inputs_read: u64,
    input_pages: u64,
}

fn render_narrative_validate_markdown(machine: &NarrativeValidateMachine) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Narrative event statistical certification\n");
    if machine.truncated {
        let _ = writeln!(
            md,
            "> **Note:** the requested event limit excludes older events. FDR control covers the selected, tested events.\n"
        );
    }
    if let Some(binding) = &machine.input_binding {
        let _ = writeln!(
            md,
            "Read all {} canonical detector inputs from tick {} through {} in {} bounded pages. Each side of a certification window contains {} consecutive simulation ticks.\n",
            machine.inputs_read,
            binding.first_tick,
            binding.terminal_tick,
            machine.input_pages,
            machine.window,
        );
    }
    let _ = writeln!(
        md,
        "_window={}, FDR={}, {} of {} events certified significant ({} skipped)._\n",
        machine.window, machine.fdr, machine.significant, machine.events_examined, machine.skipped
    );
    if machine.skipped > 0 {
        let _ = writeln!(
            md,
            "_Skipped events lack a complete exact-tick window, belong to another island, or use a metric outside population, agents.energy, average_energy, and spike_hits._\n"
        );
    }

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
            timeline_markdown_cell(&row.metric),
            row.shift,
            row.ci_lower,
            row.ci_upper,
            row.p_value,
            row.cohens_d,
            row.cliffs_delta,
            if row.significant_fdr { "yes" } else { "no" },
            timeline_markdown_cell(&row.human_text),
        );
    }

    md
}

fn parse_params(params: &ReportParams) -> Result<(CertificationParams, usize), AnalyticsError> {
    params.validate_keys(&[
        "window",
        "fdr",
        "resamples",
        "permutations",
        "block_len",
        "limit",
    ])?;

    let window = params.get_usize("window")?.unwrap_or(30);
    if window == 0 || window > MAX_WINDOW_TICKS {
        return Err(AnalyticsError::BadParam {
            name: "window".to_owned(),
            reason: format!("must be between 1 and {MAX_WINDOW_TICKS} ticks"),
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
    if !(fdr.is_finite() && 0.0 < fdr && fdr <= 1.0) {
        return Err(AnalyticsError::BadParam {
            name: "fdr".to_owned(),
            reason: "must be finite and in (0, 1]".to_owned(),
        });
    }

    let n_resamples = params.get_usize("resamples")?.unwrap_or(2000);
    if n_resamples == 0 {
        return Err(AnalyticsError::BadParam {
            name: "resamples".to_owned(),
            reason: "must be at least 1".to_owned(),
        });
    }
    let n_permutations = params.get_usize("permutations")?.unwrap_or(2000);
    if n_permutations == 0 {
        return Err(AnalyticsError::BadParam {
            name: "permutations".to_owned(),
            reason: "must be at least 1".to_owned(),
        });
    }
    let block_len = params.get_usize("block_len")?.unwrap_or(5);
    if block_len == 0 {
        return Err(AnalyticsError::BadParam {
            name: "block_len".to_owned(),
            reason: "must be at least 1".to_owned(),
        });
    }
    let limit = params.get_usize("limit")?.unwrap_or(DEFAULT_EVENT_LIMIT);
    if limit > MAX_EVENT_LIMIT {
        return Err(AnalyticsError::BadParam {
            name: "limit".to_owned(),
            reason: format!("must be at most {MAX_EVENT_LIMIT}"),
        });
    }

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

/// A sample's position can stand for a tick only after exact adjacency is observed.
#[allow(
    clippy::cast_precision_loss,
    reason = "population uses the same u64-to-f64 series projection as the production detector"
)]
fn exact_window_values(
    inputs: &VecDeque<NarrativeInputV1>,
    event_tick: u64,
    window: usize,
    metric: &str,
) -> Option<Vec<f64>> {
    let length = window.checked_mul(2)?;
    if window == 0 || inputs.len() != length {
        return None;
    }
    let first_tick = event_tick.checked_sub(u64::try_from(window).ok()?)?;
    let mut values = Vec::with_capacity(length);
    for (offset, input) in inputs.iter().enumerate() {
        let expected_tick = first_tick.checked_add(u64::try_from(offset).ok()?)?;
        if input.tick.0 != expected_tick {
            return None;
        }
        let value = match metric {
            METRIC_POPULATION => input.agent_count as f64,
            // The evidence bridge and live kernel currently name this exact
            // same input lane differently. Preserve the producer's identity.
            METRIC_ENERGY | "average_energy" => f64::from(input.average_energy),
            METRIC_COMBAT => f64::from(input.spike_hits),
            _ => return None,
        };
        values.push(value);
    }
    Some(values)
}

fn evaluate_window(
    run_event: &PersistedRunEvent,
    inputs: &VecDeque<NarrativeInputV1>,
    params: &CertificationParams,
) -> Result<Option<CandidateValidation>, AnalyticsError> {
    // The V1 input stream has no island dimension. Never attach its evidence to
    // an event from another world merely because the tick and metric match.
    if run_event.identity().island().0 != 0 {
        return Ok(None);
    }
    let event = run_event.record();
    let Some(values) = exact_window_values(inputs, event.tick.0, params.window, &event.metric)
    else {
        return Ok(None);
    };
    match certify::certify_event(&values, params.window, params) {
        Ok(certification) => Ok(Some(CandidateValidation {
            tick: event.tick.0,
            kind: format!("{:?}", event.kind),
            metric: event.metric.clone(),
            human_text: event.human_text.clone(),
            shift: certification.shift_ci.point,
            ci_lower: certification.shift_ci.lower,
            ci_upper: certification.shift_ci.upper,
            p_value: certification.p_value,
            cohens_d: certification.cohens_d,
            cliffs_delta: certification.cliffs_delta,
            effect_estimator: certification.effect_estimator,
            significant_uncorrected: certification.significant_uncorrected,
        })),
        Err(StatsError::EmptySample { .. }) => Ok(None),
        Err(error) => Err(metric_stats_error(&error)),
    }
}

fn evaluate_events(
    reader: &StorageReader,
    persisted_events: &[PersistedRunEvent],
    cert_params: &CertificationParams,
) -> Result<ValidationEvidence, AnalyticsError> {
    if persisted_events.is_empty() {
        return Ok(ValidationEvidence::default());
    }
    let window = u64::try_from(cert_params.window).map_err(|error| AnalyticsError::BadParam {
        name: "window".to_owned(),
        reason: error.to_string(),
    })?;
    // This first read validates complete-run continuity and source identity. A
    // missing stream, missing tick, duplicate, or corrupt contract must refuse
    // the report even when the affected tick lies outside a selected window.
    // Live detector history capacity does not limit the persisted evidence.
    let mut page = reader.narrative_input_page_v1(None, INPUT_READ_PAGE_SIZE, 0)?;
    let mut evidence = ValidationEvidence {
        input_binding: Some(page.binding.clone()),
        ..ValidationEvidence::default()
    };
    let input_capacity = cert_params.window * 2;
    let mut inputs = VecDeque::with_capacity(input_capacity);
    let mut event_index = 0;
    loop {
        evidence.input_pages = evidence.input_pages.saturating_add(1);
        for record in page.inputs {
            let tick = record.input.tick.0;
            if inputs.len() == input_capacity {
                inputs.pop_front();
            }
            inputs.push_back(record.input);
            evidence.inputs_read = evidence.inputs_read.saturating_add(1);

            while let Some(run_event) = persisted_events.get(event_index) {
                let Some(last_tick) = run_event
                    .record()
                    .tick
                    .0
                    .checked_add(window)
                    .and_then(|end| end.checked_sub(1))
                else {
                    evidence.skipped += 1;
                    event_index += 1;
                    continue;
                };
                if last_tick > tick {
                    break;
                }
                event_index += 1;
                if last_tick == tick
                    && let Some(candidate) = evaluate_window(run_event, &inputs, cert_params)?
                {
                    evidence.candidates.push(candidate);
                } else {
                    evidence.skipped += 1;
                }
            }
        }
        let Some(cursor) = page.next_after else {
            break;
        };
        page = reader.narrative_input_page_v1(Some(&cursor), INPUT_READ_PAGE_SIZE, 0)?;
    }
    // Events at the end of a run have no complete after window; never shorten it.
    evidence.skipped += persisted_events.len() - event_index;
    Ok(evidence)
}

impl Report for NarrativeValidate {
    fn name(&self) -> &'static str {
        "narrative-validate"
    }

    fn description(&self) -> &'static str {
        "Statistical certification of persisted narrative events from exact per-tick detector evidence"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let (cert_params, limit) = parse_params(params)?;

        let read_started = Instant::now();
        let mut persisted_events = cx.reader.recent_run_events(limit + 1)?;
        let truncated = persisted_events.len() > limit;
        if truncated {
            persisted_events.remove(0);
        }
        let evidence = evaluate_events(&cx.reader, &persisted_events, &cert_params)?;
        log_report_stage("read", &read_started, persisted_events.len());

        let render_started = Instant::now();
        let p_values: Vec<f64> = evidence.candidates.iter().map(|c| c.p_value).collect();
        let rejected = certify::benjamini_hochberg(&p_values, cert_params.fdr);

        let mut rows = Vec::with_capacity(evidence.candidates.len());
        let mut significant = 0usize;

        for (candidate, &is_significant) in evidence.candidates.into_iter().zip(&rejected) {
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
            skipped: evidence.skipped,
            truncated,
            input_binding: evidence.input_binding,
            inputs_read: evidence.inputs_read,
            input_pages: evidence.input_pages,
            events: rows,
        };

        let md = render_narrative_validate_markdown(&machine);
        let mut output = base_output(
            self.name(),
            cx,
            machine.events.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        output.schema_version = NARRATIVE_VALIDATE_SCHEMA_VERSION;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::Tick;
    use scriptbots_core::narrative::{EventKind, EventRecord};

    fn inputs_at(ticks: &[u64]) -> VecDeque<NarrativeInputV1> {
        ticks
            .iter()
            .map(|&tick| NarrativeInputV1::new(Tick(tick), tick, 1.5, 2).expect("valid input"))
            .collect()
    }

    #[test]
    fn exact_tick_windows_preserve_metric_values_and_half_open_boundaries() {
        let inputs = inputs_at(&[8, 9, 10, 11]);
        assert_eq!(
            exact_window_values(&inputs, 10, 2, METRIC_POPULATION),
            Some(vec![8.0, 9.0, 10.0, 11.0])
        );
        assert_eq!(
            exact_window_values(&inputs, 10, 2, METRIC_ENERGY),
            Some(vec![1.5; 4])
        );
        assert_eq!(
            exact_window_values(&inputs, 10, 2, METRIC_COMBAT),
            Some(vec![2.0; 4])
        );
        assert_eq!(
            exact_window_values(&inputs, 10, 2, "average_energy"),
            Some(vec![1.5; 4])
        );
        assert_eq!(exact_window_values(&inputs, 10, 2, "energy"), None);
    }

    #[test]
    fn exact_event_tick_does_not_authorize_sparse_duplicate_or_reordered_windows() {
        for ticks in [
            [6, 8, 10, 12],
            [8, 8, 10, 11],
            [9, 8, 10, 11],
            [8, 9, 10, 12],
        ] {
            assert_eq!(
                exact_window_values(&inputs_at(&ticks), 10, 2, METRIC_POPULATION),
                None,
                "the event tick exists but its surrounding window is not exact: {ticks:?}"
            );
        }
        let inputs = inputs_at(&[8, 9, 10, 11]);
        assert_eq!(exact_window_values(&inputs, 10, 3, METRIC_POPULATION), None);
        assert_eq!(exact_window_values(&inputs, 1, 2, METRIC_POPULATION), None);
        assert_eq!(
            exact_window_values(&inputs, 10, usize::MAX, METRIC_POPULATION),
            None
        );
    }

    #[test]
    fn another_islands_event_cannot_borrow_single_world_input_evidence() {
        let record = EventRecord {
            schema_version: 1,
            tick: Tick(10),
            kind: EventKind::PopulationBoom,
            severity: 0.5,
            magnitude: 2.0,
            window: (8, 12),
            metric: METRIC_POPULATION.to_owned(),
            before: 8.5,
            after: 10.5,
            score: 1.0,
            subject: None,
            human_text: "population rose".to_owned(),
        };
        let mut payload = serde_json::json!({
            "identity": {
                "tick": 10, "kind": "population_boom",
                "metric": METRIC_POPULATION, "island": 0
            },
            "record": record
        });
        let params = CertificationParams {
            window: 2,
            n_resamples: 20,
            n_permutations: 20,
            ..CertificationParams::default()
        };
        let local: PersistedRunEvent = serde_json::from_value(payload.clone()).expect("local");
        let inputs = inputs_at(&[8, 9, 10, 11]);
        assert!(
            evaluate_window(&local, &inputs, &params)
                .expect("local evidence")
                .is_some()
        );
        payload["identity"]["island"] = serde_json::json!(1);
        let foreign = serde_json::from_value(payload).expect("another island");
        assert!(
            evaluate_window(&foreign, &inputs, &params)
                .expect("unsupported island is skipped")
                .is_none()
        );
    }

    #[test]
    fn certification_params_fail_closed_on_invalid_statistical_controls() {
        for raw in [
            "window=0",
            "window=4097",
            "fdr=0",
            "fdr=1.1",
            "fdr=NaN",
            "resamples=0",
            "permutations=0",
            "block_len=0",
            "limit=4096",
            "windwo=30",
        ] {
            let params = ReportParams::from_pairs([raw.to_owned()]).expect("key=value syntax");
            assert!(
                matches!(parse_params(&params), Err(AnalyticsError::BadParam { .. })),
                "invalid control was accepted: {raw}"
            );
        }
    }

    #[test]
    fn certification_params_accept_the_bounded_public_surface() {
        let params = ReportParams::from_pairs([
            "window=4096".to_owned(),
            "fdr=0.1".to_owned(),
            "resamples=100".to_owned(),
            "permutations=100".to_owned(),
            "block_len=4".to_owned(),
            "limit=4095".to_owned(),
        ])
        .expect("params");
        let (cert, limit) = parse_params(&params).expect("valid controls");
        assert_eq!(cert.window, MAX_WINDOW_TICKS);
        assert_eq!(cert.fdr, 0.1);
        assert_eq!(cert.n_resamples, 100);
        assert_eq!(cert.n_permutations, 100);
        assert_eq!(cert.block_len, 4);
        assert_eq!(limit, 4095);
    }
}
