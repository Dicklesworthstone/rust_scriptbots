//! Analysis layer: effect sizes with CIs over matched-seed run summaries (bd-16g.1.4).

use rand::Rng;
use scriptbots_core::SmallRngStream;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};
use thiserror::Error;

/// Individual run summary row produced by a verified run export.
///
/// The private analysis-input digest is computed by the constructor and rechecked by the
/// statistics authority. Callers may inspect the public evidence, but mutating any scientific
/// field makes the row fail closed instead of silently detaching values from provenance.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RunSummary {
    pub run_id: String,
    pub arm_id: u16,
    pub seed: u64,
    pub config_digest: String,
    pub digest: String,
    pub ticks: u64,
    pub metrics: BTreeMap<String, f64>,
    /// BLAKE3 digest of the exact retained summary artifact bytes.
    pub summary_artifact_digest: String,
    /// Retained artifact path, when a production runner materialized one.
    pub summary_path: Option<String>,
    analysis_input_digest: String,
}

impl RunSummary {
    /// Construct one typed row after its source artifact and run provenance have been verified.
    #[must_use]
    pub(crate) fn from_verified_parts(
        run_id: String,
        arm_id: u16,
        seed: u64,
        config_digest: String,
        digest: String,
        ticks: u64,
        metrics: BTreeMap<String, f64>,
        summary_artifact_digest: String,
        summary_path: Option<String>,
    ) -> Self {
        let analysis_input_digest = analysis_input_digest(
            &run_id,
            arm_id,
            seed,
            &config_digest,
            &digest,
            ticks,
            &metrics,
            &summary_artifact_digest,
        );
        Self {
            run_id,
            arm_id,
            seed,
            config_digest,
            digest,
            ticks,
            metrics,
            summary_artifact_digest,
            summary_path,
            analysis_input_digest,
        }
    }

    /// Digest binding every typed analysis input to the exact retained summary artifact.
    #[must_use]
    pub fn analysis_input_digest(&self) -> &str {
        &self.analysis_input_digest
    }

    fn has_valid_analysis_input_digest(&self) -> bool {
        self.analysis_input_digest
            == analysis_input_digest(
                &self.run_id,
                self.arm_id,
                self.seed,
                &self.config_digest,
                &self.digest,
                self.ticks,
                &self.metrics,
                &self.summary_artifact_digest,
            )
    }
}

fn hash_len_prefixed(hasher: &mut blake3::Hasher, value: &[u8]) {
    let length = u64::try_from(value.len()).expect("slice length fits u64 on supported targets");
    hasher.update(&length.to_le_bytes());
    hasher.update(value);
}

fn analysis_input_digest(
    run_id: &str,
    arm_id: u16,
    seed: u64,
    config_digest: &str,
    digest: &str,
    ticks: u64,
    metrics: &BTreeMap<String, f64>,
    summary_artifact_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"scriptbots.lab-run-summary.v1\0");
    hash_len_prefixed(&mut hasher, run_id.as_bytes());
    hasher.update(&arm_id.to_le_bytes());
    hasher.update(&seed.to_le_bytes());
    hash_len_prefixed(&mut hasher, config_digest.as_bytes());
    hash_len_prefixed(&mut hasher, digest.as_bytes());
    hasher.update(&ticks.to_le_bytes());
    let metric_count =
        u64::try_from(metrics.len()).expect("map length fits u64 on supported targets");
    hasher.update(&metric_count.to_le_bytes());
    for (name, value) in metrics {
        hash_len_prefixed(&mut hasher, name.as_bytes());
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hash_len_prefixed(&mut hasher, summary_artifact_digest.as_bytes());
    hasher.finalize().to_hex().to_string()
}

/// Multiple comparison adjustment correction methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Correction {
    None,
    HolmBonferroni,
    BenjaminiHochberg,
}

impl Correction {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::HolmBonferroni => "holm_bonferroni",
            Self::BenjaminiHochberg => "benjamini_hochberg",
        }
    }
}

/// Statistical test identifiers for audit trails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestName {
    PairedDifference,
    HedgesG,
    SpearmanRank,
}

impl TestName {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PairedDifference => "paired_difference",
            Self::HedgesG => "hedges_g",
            Self::SpearmanRank => "spearman_rank",
        }
    }
}

/// Procedure used to obtain a p-value for each matched effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PValueProcedure {
    /// Monte Carlo sign-flip test over the paired treatment-minus-control differences.
    PairedSignFlipMonteCarlo,
}

impl PValueProcedure {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PairedSignFlipMonteCarlo => "paired_sign_flip_monte_carlo",
        }
    }
}

/// Procedure used to obtain an interval for each matched mean difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceIntervalProcedure {
    /// Empirical 2.5th and 97.5th percentiles of paired bootstrap means.
    PercentileBootstrap,
}

impl ConfidenceIntervalProcedure {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PercentileBootstrap => "percentile_bootstrap",
        }
    }
}

/// Direction declared before a hypothesis test is run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlternativeHypothesis {
    /// Either positive or negative changes count as evidence.
    TwoSided,
    /// Only treatment values above control count as evidence.
    TreatmentGreater,
    /// Only treatment values below control count as evidence.
    TreatmentLess,
}

impl AlternativeHypothesis {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TwoSided => "two_sided",
            Self::TreatmentGreater => "treatment_greater",
            Self::TreatmentLess => "treatment_less",
        }
    }
}

/// Why a standardized effect could not be reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum UndefinedReason {
    /// Fewer than two matched pairs cannot estimate the spread of paired differences.
    InsufficientPairs { have: usize, need: usize },
    /// The paired differences have no measurable spread, so a finite standardized effect
    /// does not exist. The mean difference, interval, and permutation p-value remain valid.
    ZeroVariance,
}

impl UndefinedReason {
    #[must_use]
    pub fn description(self) -> String {
        match self {
            Self::InsufficientPairs { have, need } => {
                format!("insufficient_pairs(have={have}, need={need})")
            }
            Self::ZeroVariance => "zero_variance".to_owned(),
        }
    }
}

/// Why a defined result should not be read as a well-powered estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum UnderpoweredReason {
    /// The cohort is valid but smaller than the lab's documented interpretation floor.
    FewerThanRecommendedPairs { have: usize, recommended: usize },
}

impl UnderpoweredReason {
    #[must_use]
    pub fn description(self) -> String {
        match self {
            Self::FewerThanRecommendedPairs { have, recommended } => {
                format!("fewer_than_recommended_pairs(have={have}, recommended={recommended})")
            }
        }
    }
}

/// Reproducible choices governing a matched-seed analysis.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AnalysisParams {
    pub alpha: f64,
    pub correction: Correction,
    pub alternative: AlternativeHypothesis,
    pub bootstrap_iterations: usize,
    pub permutation_iterations: u32,
    pub resampling_seed: u64,
    pub recommended_pairs: usize,
}

impl Default for AnalysisParams {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            correction: Correction::BenjaminiHochberg,
            alternative: AlternativeHypothesis::TwoSided,
            bootstrap_iterations: 2_000,
            permutation_iterations: 4_000,
            resampling_seed: 0x7453,
            recommended_pairs: 10,
        }
    }
}

/// Statistical computation errors.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum StatsError {
    #[error("No samples provided")]
    NoSamples,
    #[error("At least two samples per cohort are required")]
    InsufficientSamples,
    #[error("Unmatched seeds between cohorts")]
    UnmatchedSeeds,
    #[error("Metric {0} missing from run summary")]
    MissingMetric(String),
    #[error("Zero variance in metric values")]
    ZeroVariance,
    #[error("Non-finite metric value encountered")]
    NonFiniteValue,
    #[error("At least two arms are required for a matched-seed comparison")]
    InsufficientArms,
    #[error("Matched-seed reports require control arm 0")]
    MissingControlArm,
    #[error("Metric {0} appears more than once in the requested report")]
    DuplicateMetric(String),
    #[error("Arm {arm_id} contains duplicate seed {seed}")]
    DuplicateSeed { arm_id: u16, seed: u64 },
    #[error("Bootstrap and permutation iteration counts must both be non-zero")]
    ZeroIterations,
    #[error("Recommended matched-pair threshold must be non-zero")]
    InvalidRecommendedPairs,
    #[error("Run summary {run_id} no longer matches its analysis-input digest")]
    RunSummaryIntegrity { run_id: String },
    /// The caller supplied a non-finite significance level or one outside `(0, 1)`.
    #[error("Significance level must be finite and strictly between zero and one")]
    InvalidSignificanceLevel,
    /// A raw p-value was non-finite or outside `[0, 1]`.
    #[error("P-value at index {index} must be finite and between zero and one")]
    InvalidPValue {
        /// Zero-based position of the rejected value in the caller's input.
        index: usize,
    },
    /// The input family cannot be ranked without losing integer precision.
    #[error("Too many simultaneous comparisons to rank")]
    TooManyComparisons,
}

/// Detailed paired effect size and confidence interval for matched-seed runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PairedEffect {
    pub metric: String,
    pub n_pairs: usize,
    pub mean_diff: f64,
    pub sd_diff: f64,
    pub cohens_dz: f64,
    pub ci_95: (f64, f64),
    pub test: TestName,
    pub alternative: AlternativeHypothesis,
    pub p_value: f64,
}

/// Generic effect size record carrying strict provenance and honesty metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Effect {
    pub metric: String,
    pub n: usize,
    pub test: TestName,
    pub alternative: AlternativeHypothesis,
    pub statistic: f64,
    pub ci_95: (f64, f64),
    pub correction: Option<Correction>,
    pub underpowered: bool,
}

/// One control-versus-treatment result produced by the canonical lab analysis.
///
/// `mean_difference` and permutation evidence remain meaningful when `standardized_effect`
/// is undefined. The bootstrap interval is separately optional for a one-pair cohort. This
/// prevents zero-spread or undersampled evidence from being replaced with invented values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MatchedEffect {
    pub metric: String,
    pub control_arm: u16,
    pub treatment_arm: u16,
    pub n_pairs: usize,
    pub estimator: TestName,
    pub alternative: AlternativeHypothesis,
    pub mean_difference: f64,
    pub standardized_effect: Option<f64>,
    pub undefined_reason: Option<UndefinedReason>,
    pub ci_95: Option<(f64, f64)>,
    pub ci_undefined_reason: Option<UndefinedReason>,
    pub raw_p_value: f64,
    pub adjusted: AdjustedComparison,
    pub underpowered_reason: Option<UnderpoweredReason>,
}

/// Complete, ordered output of one matched-seed report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MatchedSeedAnalysis {
    /// Exact ordered metric family requested by the validated experiment.
    pub metrics: Vec<String>,
    /// Every analysis and resampling choice needed to reproduce this result.
    pub params: AnalysisParams,
    /// Typed identity of the p-value procedure used for every effect.
    pub p_value_procedure: PValueProcedure,
    /// Typed identity of the confidence-interval procedure used for every effect.
    pub confidence_interval_procedure: ConfidenceIntervalProcedure,
    pub effects: Vec<MatchedEffect>,
}

/// One hypothesis after a multiple-comparison adjustment.
///
/// Records stay in the caller's original order. `rank` is one-based and describes the
/// hypothesis's position after sorting by raw p-value, with the original index breaking
/// ties deterministically. Holm-Bonferroni controls family-wise error; Benjamini-Hochberg
/// controls the false-discovery rate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdjustedComparison {
    /// Zero-based position in the caller's raw p-value slice.
    pub original_index: usize,
    /// One-based position after sorting by raw p-value and original index.
    pub rank: usize,
    /// Unmodified caller-supplied p-value.
    pub raw_p_value: f64,
    /// Multiplicity-adjusted p-value for the selected correction.
    pub adjusted_p_value: f64,
    /// Rank-specific critical value against which the raw p-value is compared.
    pub adjusted_alpha: f64,
    /// Whether the selected procedure rejects this hypothesis.
    pub rejected: bool,
    /// Multiple-comparison procedure that produced this record.
    pub correction: Correction,
}

/// Paired Monte Carlo sign-flip p-value for the mean difference (bd-h189).
///
/// Replaces a hardcoded `p_value: 0.05`. Under the null "the pairing carries no signal",
/// each paired difference's sign is exchangeable, so sampled sign assignments build a
/// distribution-free Monte Carlo null directly from the observed data.
///
/// Deterministic by construction. The generator is an inline xorshift64 seeded by the
/// caller, so the same cohorts always yield the same p-value — this module's other
/// randomised routine, `bootstrap_ci`, is deterministic for the same reason.
///
/// Uses the `(count + 1) / (iterations + 1)` correction, so the result is never exactly
/// zero: a permutation test can bound a p-value from above but cannot prove it is zero, and
/// reporting 0.0 would claim more than the procedure supports.
fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for value in values {
        let corrected = value - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
    }
    sum
}

fn paired_permutation_p_value(
    diffs: &[f64],
    iterations: u32,
    seed: u64,
    alternative: AlternativeHypothesis,
) -> f64 {
    debug_assert!(
        !diffs.is_empty(),
        "caller rejects empty cohorts before this point"
    );
    let n = diffs.len() as f64;
    let observed = compensated_sum(diffs.iter().copied()) / n;

    // Never zero: xorshift64 has a fixed point at zero and would emit a constant stream.
    let mut state = if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    };
    let mut extreme_count = 0_u64;
    for _ in 0..iterations {
        let mut sum = 0.0;
        let mut compensation = 0.0;
        for diff in diffs {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let signed = if state & 1 == 0 { *diff } else { -*diff };
            let corrected = signed - compensation;
            let next = sum + corrected;
            compensation = (next - sum) - corrected;
            sum = next;
        }
        let permuted = sum / n;
        let is_at_least_as_extreme = match alternative {
            AlternativeHypothesis::TwoSided => permuted.abs() >= observed.abs(),
            AlternativeHypothesis::TreatmentGreater => permuted >= observed,
            AlternativeHypothesis::TreatmentLess => permuted <= observed,
        };
        if is_at_least_as_extreme {
            extreme_count += 1;
        }
    }
    (extreme_count + 1) as f64 / (u64::from(iterations) + 1) as f64
}

#[derive(Debug)]
struct PairedComponents {
    n_pairs: usize,
    mean_difference: f64,
    sd_difference: Option<f64>,
    standardized_effect: Option<f64>,
    ci_95: Option<(f64, f64)>,
    p_value: f64,
}

fn paired_components(
    a: &[RunSummary],
    b: &[RunSummary],
    metric: &str,
    params: &AnalysisParams,
) -> Result<PairedComponents, StatsError> {
    if a.is_empty() || b.is_empty() {
        return Err(StatsError::NoSamples);
    }
    if params.bootstrap_iterations == 0 || params.permutation_iterations == 0 {
        return Err(StatsError::ZeroIterations);
    }

    let mut sorted_a = a.to_vec();
    sorted_a.sort_by_key(|run| run.seed);
    let mut sorted_b = b.to_vec();
    sorted_b.sort_by_key(|run| run.seed);
    for run in sorted_a.iter().chain(&sorted_b) {
        if !run.has_valid_analysis_input_digest() {
            return Err(StatsError::RunSummaryIntegrity {
                run_id: run.run_id.clone(),
            });
        }
    }
    for runs in [&sorted_a, &sorted_b] {
        if let Some(duplicate) = runs.windows(2).find(|pair| pair[0].seed == pair[1].seed) {
            return Err(StatsError::DuplicateSeed {
                arm_id: duplicate[0].arm_id,
                seed: duplicate[0].seed,
            });
        }
    }
    if sorted_a.len() != sorted_b.len() {
        return Err(StatsError::UnmatchedSeeds);
    }

    let mut diffs = Vec::with_capacity(sorted_a.len());
    for (control, treatment) in sorted_a.iter().zip(sorted_b.iter()) {
        if control.seed != treatment.seed {
            return Err(StatsError::UnmatchedSeeds);
        }
        let control_value = control
            .metrics
            .get(metric)
            .copied()
            .ok_or_else(|| StatsError::MissingMetric(metric.to_owned()))?;
        let treatment_value = treatment
            .metrics
            .get(metric)
            .copied()
            .ok_or_else(|| StatsError::MissingMetric(metric.to_owned()))?;
        if !control_value.is_finite() || !treatment_value.is_finite() {
            return Err(StatsError::NonFiniteValue);
        }
        diffs.push(treatment_value - control_value);
    }

    let n = diffs.len() as f64;
    let mean_difference = compensated_sum(diffs.iter().copied()) / n;
    let variance = (diffs.len() >= 2).then(|| {
        compensated_sum(
            diffs
                .iter()
                .map(|difference| (difference - mean_difference).powi(2)),
        ) / (n - 1.0)
    });
    let difference_range = diffs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(*value), max.max(*value))
        });
    let degenerate_tolerance = 1e-12 * (difference_range.1 - difference_range.0);
    let sd_difference = variance
        .map(f64::sqrt)
        .filter(|spread| *spread > degenerate_tolerance);
    let standardized_effect = sd_difference.map(|spread| mean_difference / spread);
    let ci_95 = if diffs.len() < 2 {
        None
    } else {
        Some(bootstrap_ci(
            &diffs,
            params.bootstrap_iterations,
            params.resampling_seed,
        )?)
    };
    let p_value = paired_permutation_p_value(
        &diffs,
        params.permutation_iterations,
        params.resampling_seed ^ 0x00F1_1900,
        params.alternative,
    );

    Ok(PairedComponents {
        n_pairs: diffs.len(),
        mean_difference,
        sd_difference,
        standardized_effect,
        ci_95,
        p_value,
    })
}

/// Computes paired difference effect sizes strictly matched by seed across cohorts.
pub fn paired_diff(
    a: &[RunSummary],
    b: &[RunSummary],
    metric: &str,
) -> Result<PairedEffect, StatsError> {
    let params = AnalysisParams {
        bootstrap_iterations: 1_000,
        permutation_iterations: 1_000,
        resampling_seed: 42,
        ..AnalysisParams::default()
    };
    let components = paired_components(a, b, metric, &params)?;

    // bd-7453: a degenerate spread has no standardized effect size, and reporting one is
    // worse than refusing. This previously fell back to `cohens_dz = 0.0`, which reads as
    // "no effect" — the strongest possible WRONG answer, because zero variance with a
    // nonzero mean difference is a PERFECTLY CONSISTENT difference, i.e. an unbounded
    // effect rather than an absent one.
    //
    // `StatsError::ZeroVariance` already existed for exactly this case and was constructed
    // nowhere. The variant was right and this caller was wrong; the fix is to construct it,
    // not to delete it.
    //
    // Note this also refuses a single pair, where `(n - 1).max(1.0)` makes the variance
    // structurally zero: one pair cannot support a standardized effect size either.
    let sd_diff = components.sd_difference.ok_or(StatsError::ZeroVariance)?;
    let cohens_dz = components
        .standardized_effect
        .ok_or(StatsError::ZeroVariance)?;

    Ok(PairedEffect {
        metric: metric.to_string(),
        n_pairs: components.n_pairs,
        mean_diff: components.mean_difference,
        sd_diff,
        cohens_dz,
        ci_95: components.ci_95.ok_or(StatsError::InsufficientSamples)?,
        test: TestName::PairedDifference,
        alternative: params.alternative,
        p_value: components.p_value,
    })
}

/// Computes independent-cohort Hedges' g as treatment minus control with small-sample correction.
///
/// The matched-seed report deliberately uses paired Cohen's dz instead: applying this
/// independent-sample estimator there would discard the validated pairing.
pub fn hedges_g(
    control: &[RunSummary],
    treatment: &[RunSummary],
    metric: &str,
) -> Result<Effect, StatsError> {
    if control.is_empty() || treatment.is_empty() {
        return Err(StatsError::NoSamples);
    }
    if control.len() < 2 || treatment.len() < 2 {
        return Err(StatsError::InsufficientSamples);
    }
    for run in control.iter().chain(treatment) {
        if !run.has_valid_analysis_input_digest() {
            return Err(StatsError::RunSummaryIntegrity {
                run_id: run.run_id.clone(),
            });
        }
    }

    let mut vals_a: Vec<f64> = control
        .iter()
        .map(|r| {
            r.metrics
                .get(metric)
                .copied()
                .ok_or_else(|| StatsError::MissingMetric(metric.to_string()))
        })
        .collect::<Result<_, _>>()?;

    let mut vals_b: Vec<f64> = treatment
        .iter()
        .map(|r| {
            r.metrics
                .get(metric)
                .copied()
                .ok_or_else(|| StatsError::MissingMetric(metric.to_string()))
        })
        .collect::<Result<_, _>>()?;

    if vals_a
        .iter()
        .chain(vals_b.iter())
        .any(|value| !value.is_finite())
    {
        return Err(StatsError::NonFiniteValue);
    }
    vals_a.sort_by(f64::total_cmp);
    vals_b.sort_by(f64::total_cmp);

    let na = vals_a.len() as f64;
    let nb = vals_b.len() as f64;

    let mean_a = compensated_sum(vals_a.iter().copied()) / na;
    let mean_b = compensated_sum(vals_b.iter().copied()) / nb;

    let var_a =
        compensated_sum(vals_a.iter().map(|value| (value - mean_a).powi(2))) / (na - 1.0).max(1.0);
    let var_b =
        compensated_sum(vals_b.iter().map(|value| (value - mean_b).powi(2))) / (nb - 1.0).max(1.0);

    let df = na + nb - 2.0;
    let pooled_variance = ((na - 1.0) * var_a + (nb - 1.0) * var_b) / df;
    if !pooled_variance.is_finite() {
        return Err(StatsError::NonFiniteValue);
    }
    if pooled_variance <= 0.0 {
        return Err(StatsError::ZeroVariance);
    }
    let pooled_sd = pooled_variance.sqrt();
    let (data_min, data_max) = vals_a
        .iter()
        .chain(vals_b.iter())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(*value), max.max(*value))
        });
    let data_range = data_max - data_min;
    if !data_range.is_finite() {
        return Err(StatsError::NonFiniteValue);
    }
    // Preserve paired_diff's 1e-12 degeneracy boundary without making it depend on
    // the unit or additive origin: the total observed range scales with the values
    // but is unchanged by translating every observation.
    let degenerate_tolerance = 1e-12 * data_range;
    if pooled_sd <= degenerate_tolerance {
        return Err(StatsError::ZeroVariance);
    }
    let cohens_d = (mean_b - mean_a) / pooled_sd;

    // Small-sample correction factor for Hedges' g
    let correction_factor = 1.0 - (3.0 / (4.0 * df - 1.0).max(1.0));
    let g = cohens_d * correction_factor;
    if !g.is_finite() {
        return Err(StatsError::NonFiniteValue);
    }

    // Large-sample standard error for Hedges' g. Unlike the previous hardcoded 0.1,
    // this uncertainty responds to both cohort size and the observed standardized effect:
    // https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/hedges_g.htm
    let standard_error = ((na + nb) / (na * nb) + g.powi(2) / (2.0 * (na + nb))).sqrt();
    if !standard_error.is_finite() {
        return Err(StatsError::NonFiniteValue);
    }
    let margin_95 = 1.959_963_984_540_054 * standard_error;
    let ci_95 = (g - margin_95, g + margin_95);

    Ok(Effect {
        metric: metric.to_string(),
        n: vals_a.len() + vals_b.len(),
        test: TestName::HedgesG,
        alternative: AlternativeHypothesis::TwoSided,
        statistic: g,
        ci_95,
        correction: None,
        underpowered: (vals_a.len() + vals_b.len()) < 10,
    })
}

/// Deterministic, input-order-invariant bootstrap 95% confidence interval.
///
/// # Errors
///
/// Refuses empty/non-finite input and zero iterations instead of manufacturing `(0, 0)` or
/// indexing an empty bootstrap distribution.
pub fn bootstrap_ci(values: &[f64], iters: usize, seed: u64) -> Result<(f64, f64), StatsError> {
    if values.is_empty() {
        return Err(StatsError::NoSamples);
    }
    if iters == 0 {
        return Err(StatsError::ZeroIterations);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(StatsError::NonFiniteValue);
    }
    if values.len() < 2 {
        return Err(StatsError::InsufficientSamples);
    }

    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let mut means = Vec::with_capacity(iters);
    let mut rng = SmallRngStream::seed_from_u64(seed);

    for _ in 0..iters {
        let sample_sum = compensated_sum((0..ordered.len()).map(|_| {
            // Use the project's portable scientific stream. A previous bespoke LCG
            // selected with `% len`; for power-of-two sample sizes its short low-bit
            // cycle could make every replicate the same multiset and collapse a
            // genuinely data-derived interval.
            ordered[rng.random_range(0..ordered.len())]
        }));
        means.push(sample_sum / ordered.len() as f64);
    }

    means.sort_by(|x, y| x.total_cmp(y));
    // Nearest-rank empirical quantiles: ceil(Bp) - 1 in zero-based indexing.
    let lower_idx = ((iters as f64 * 0.025).ceil() as usize).saturating_sub(1);
    let upper_idx = ((iters as f64 * 0.975).ceil() as usize).saturating_sub(1);

    Ok((
        means[lower_idx.min(iters - 1)],
        means[upper_idx.min(iters - 1)],
    ))
}

fn average_ranks(values: &[f64]) -> Result<Vec<f64>, StatsError> {
    if values.is_empty() {
        return Err(StatsError::NoSamples);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(StatsError::NonFiniteValue);
    }

    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        let value_ordering = if values[*left] < values[*right] {
            Ordering::Less
        } else if values[*left] > values[*right] {
            Ordering::Greater
        } else {
            Ordering::Equal
        };
        value_ordering.then_with(|| left.cmp(right))
    });

    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < indices.len() {
        let mut end = start + 1;
        while end < indices.len() && values[indices[end]] == values[indices[start]] {
            end += 1;
        }
        // Ranks are one-based. A tie occupying positions start..end receives their average.
        let average = (start + 1 + end) as f64 / 2.0;
        for &index in &indices[start..end] {
            ranks[index] = average;
        }
        start = end;
    }
    Ok(ranks)
}

/// Spearman's rank correlation with deterministic average-rank tie handling.
///
/// Callers must supply two variables whose monotonic association has been declared in advance.
/// The matched-seed report does not misuse control/treatment outcome concordance as evidence of
/// dose response; factor-level trend analysis needs the validated numeric factor values.
///
/// # Errors
///
/// Refuses empty, mismatched, non-finite, and zero-variance inputs. A constant series has no
/// rank correlation; returning zero would incorrectly claim evidence of no monotonic relation.
pub fn spearman_rank_correlation(left: &[f64], right: &[f64]) -> Result<f64, StatsError> {
    if left.is_empty() || right.is_empty() {
        return Err(StatsError::NoSamples);
    }
    if left.len() != right.len() {
        return Err(StatsError::UnmatchedSeeds);
    }
    if left.len() < 2 {
        return Err(StatsError::InsufficientSamples);
    }

    let left_ranks = average_ranks(left)?;
    let right_ranks = average_ranks(right)?;
    let count = left_ranks.len() as f64;
    let left_mean = compensated_sum(left_ranks.iter().copied()) / count;
    let right_mean = compensated_sum(right_ranks.iter().copied()) / count;
    let covariance = compensated_sum(
        left_ranks
            .iter()
            .zip(&right_ranks)
            .map(|(left, right)| (left - left_mean) * (right - right_mean)),
    );
    let left_ss = compensated_sum(left_ranks.iter().map(|rank| (rank - left_mean).powi(2)));
    let right_ss = compensated_sum(right_ranks.iter().map(|rank| (rank - right_mean).powi(2)));
    if left_ss <= 0.0 || right_ss <= 0.0 {
        return Err(StatsError::ZeroVariance);
    }
    Ok(covariance / (left_ss * right_ss).sqrt())
}

/// Adjusts a family of raw p-values and returns value-level correction evidence.
///
/// Holm-Bonferroni uses a true step-down decision: after the first failed rank, no later
/// hypothesis is rejected even if its own raw p-value is below that rank's alpha. Its
/// adjusted p-values are the prefix maximum of `(m - rank + 1) * p`.
///
/// Benjamini-Hochberg uses the complementary step-up decision and the reverse cumulative
/// minimum of `m / rank * p`. `Correction::None` is an explicit pass-through that still
/// returns the raw decision and rank provenance.
///
/// The returned records do not mutate effect estimates or confidence intervals. A reporting
/// caller must persist these results explicitly; merely labelling an `Effect` as corrected
/// without carrying the adjusted values would recreate bd-7vdu.
///
/// # Errors
///
/// Returns [`StatsError::NoSamples`] for an empty family, rejects a non-finite or out-of-range
/// significance level or p-value, and refuses a family whose size cannot be represented exactly
/// by the ranking implementation.
pub fn adjust_multiple_comparisons(
    p_values: &[f64],
    alpha: f64,
    method: Correction,
) -> Result<Vec<AdjustedComparison>, StatsError> {
    if p_values.is_empty() {
        return Err(StatsError::NoSamples);
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(StatsError::InvalidSignificanceLevel);
    }

    let family_size = u32::try_from(p_values.len()).map_err(|_| StatsError::TooManyComparisons)?;
    let family_size_f64 = f64::from(family_size);
    let mut ranked = Vec::with_capacity(p_values.len());
    for (original_index, &p_value) in p_values.iter().enumerate() {
        if !p_value.is_finite() || !(0.0..=1.0).contains(&p_value) {
            return Err(StatsError::InvalidPValue {
                index: original_index,
            });
        }
        ranked.push((original_index, p_value));
    }
    ranked.sort_by(|(left_index, left_p), (right_index, right_p)| {
        let value_ordering = if left_p < right_p {
            Ordering::Less
        } else if left_p > right_p {
            Ordering::Greater
        } else {
            Ordering::Equal
        };
        value_ordering.then_with(|| left_index.cmp(right_index))
    });

    let mut adjusted_p_values = vec![0.0; ranked.len()];
    let mut adjusted_alphas = vec![alpha; ranked.len()];
    let mut rejected = vec![false; ranked.len()];

    match method {
        Correction::None => {
            for (position, (_, p_value)) in ranked.iter().enumerate() {
                adjusted_p_values[position] = *p_value;
                rejected[position] = *p_value <= alpha;
            }
        }
        Correction::HolmBonferroni => {
            let mut running_adjusted = 0.0_f64;
            let mut still_rejecting = true;
            for (position, (_, p_value)) in ranked.iter().enumerate() {
                let rank =
                    u32::try_from(position + 1).map_err(|_| StatsError::TooManyComparisons)?;
                let remaining = family_size - rank + 1;
                let remaining_f64 = f64::from(remaining);
                let adjusted_alpha = alpha / remaining_f64;

                running_adjusted = running_adjusted.max((remaining_f64 * *p_value).min(1.0));
                adjusted_p_values[position] = running_adjusted;
                adjusted_alphas[position] = adjusted_alpha;

                rejected[position] = still_rejecting && *p_value <= adjusted_alpha;
                still_rejecting = rejected[position];
            }
        }
        Correction::BenjaminiHochberg => {
            let mut running_adjusted = 1.0_f64;
            for position in (0..ranked.len()).rev() {
                let rank =
                    u32::try_from(position + 1).map_err(|_| StatsError::TooManyComparisons)?;
                let rank_f64 = f64::from(rank);
                let candidate = (family_size_f64 * ranked[position].1 / rank_f64).min(1.0);
                running_adjusted = running_adjusted.min(candidate);
                adjusted_p_values[position] = running_adjusted;
                adjusted_alphas[position] = alpha * rank_f64 / family_size_f64;
            }

            let last_rejected = ranked
                .iter()
                .enumerate()
                .rev()
                .find(|(position, (_, p_value))| *p_value <= adjusted_alphas[*position])
                .map(|(position, _)| position);
            if let Some(last_rejected) = last_rejected {
                rejected[..=last_rejected].fill(true);
            }
        }
    }

    let mut comparisons = ranked
        .iter()
        .enumerate()
        .map(
            |(position, (original_index, raw_p_value))| AdjustedComparison {
                original_index: *original_index,
                rank: position + 1,
                raw_p_value: *raw_p_value,
                adjusted_p_value: adjusted_p_values[position],
                adjusted_alpha: adjusted_alphas[position],
                rejected: rejected[position],
                correction: method,
            },
        )
        .collect::<Vec<_>>();
    comparisons.sort_by_key(|comparison| comparison.original_index);
    Ok(comparisons)
}

/// Analyze every requested metric for arm 0 versus each later arm.
///
/// The input may arrive in any order. Arms are ordered numerically, metrics retain the
/// validated specification's order, and runs within each arm are paired by seed. All raw
/// p-values enter one correction family, so a report cannot quietly correct each metric in
/// isolation. Zero-spread differences retain their mean/CI/permutation evidence with a typed
/// undefined standardized effect.
///
/// # Errors
///
/// Refuses malformed cohorts, duplicate metrics or seeds, missing control/treatment pairs,
/// non-finite values, invalid analysis parameters, and zero resampling iterations.
pub fn analyze_matched_seed_runs(
    summaries: &[RunSummary],
    metrics: &[String],
    params: AnalysisParams,
) -> Result<MatchedSeedAnalysis, StatsError> {
    if summaries.is_empty() || metrics.is_empty() {
        return Err(StatsError::NoSamples);
    }
    if params.bootstrap_iterations == 0 || params.permutation_iterations == 0 {
        return Err(StatsError::ZeroIterations);
    }
    if params.recommended_pairs == 0 {
        return Err(StatsError::InvalidRecommendedPairs);
    }

    let mut unique_metrics = BTreeSet::new();
    for metric in metrics {
        if !unique_metrics.insert(metric.as_str()) {
            return Err(StatsError::DuplicateMetric(metric.clone()));
        }
    }

    let mut arms = BTreeMap::<u16, Vec<RunSummary>>::new();
    for summary in summaries {
        if !summary.has_valid_analysis_input_digest() {
            return Err(StatsError::RunSummaryIntegrity {
                run_id: summary.run_id.clone(),
            });
        }
        let runs = arms.entry(summary.arm_id).or_default();
        if runs.iter().any(|run| run.seed == summary.seed) {
            return Err(StatsError::DuplicateSeed {
                arm_id: summary.arm_id,
                seed: summary.seed,
            });
        }
        runs.push(summary.clone());
    }
    if arms.len() < 2 {
        return Err(StatsError::InsufficientArms);
    }
    let control = arms.get(&0).ok_or(StatsError::MissingControlArm)?;

    struct Provisional {
        metric: String,
        treatment_arm: u16,
        components: PairedComponents,
    }

    let mut provisional = Vec::new();
    for (&treatment_arm, treatment) in arms.range(1..) {
        for metric in metrics {
            provisional.push(Provisional {
                metric: metric.clone(),
                treatment_arm,
                components: paired_components(control, treatment, metric, &params)?,
            });
        }
    }
    if provisional.is_empty() {
        return Err(StatsError::InsufficientArms);
    }

    let adjusted = adjust_multiple_comparisons(
        &provisional
            .iter()
            .map(|result| result.components.p_value)
            .collect::<Vec<_>>(),
        params.alpha,
        params.correction,
    )?;

    let effects = provisional
        .into_iter()
        .zip(adjusted)
        .map(|(result, adjusted)| {
            let undefined_reason = if result.components.n_pairs < 2 {
                Some(UndefinedReason::InsufficientPairs {
                    have: result.components.n_pairs,
                    need: 2,
                })
            } else if result.components.standardized_effect.is_none() {
                Some(UndefinedReason::ZeroVariance)
            } else {
                None
            };
            let underpowered_reason = (result.components.n_pairs < params.recommended_pairs)
                .then_some(UnderpoweredReason::FewerThanRecommendedPairs {
                    have: result.components.n_pairs,
                    recommended: params.recommended_pairs,
                });
            MatchedEffect {
                metric: result.metric,
                control_arm: 0,
                treatment_arm: result.treatment_arm,
                n_pairs: result.components.n_pairs,
                estimator: TestName::PairedDifference,
                alternative: params.alternative,
                mean_difference: result.components.mean_difference,
                standardized_effect: result.components.standardized_effect,
                undefined_reason,
                ci_95: result.components.ci_95,
                ci_undefined_reason: result.components.ci_95.is_none().then_some(
                    UndefinedReason::InsufficientPairs {
                        have: result.components.n_pairs,
                        need: 2,
                    },
                ),
                raw_p_value: result.components.p_value,
                adjusted,
                underpowered_reason,
            }
        })
        .collect();

    Ok(MatchedSeedAnalysis {
        metrics: metrics.to_vec(),
        params,
        p_value_procedure: PValueProcedure::PairedSignFlipMonteCarlo,
        confidence_interval_procedure: ConfidenceIntervalProcedure::PercentileBootstrap,
        effects,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    fn report_summary(
        run_id: &str,
        arm_id: u16,
        seed: u64,
        metrics: impl IntoIterator<Item = (&'static str, f64)>,
    ) -> RunSummary {
        RunSummary::from_verified_parts(
            run_id.to_owned(),
            arm_id,
            seed,
            format!("config-{arm_id}"),
            format!("digest-{run_id}"),
            100,
            metrics
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value))
                .collect(),
            format!("summary-{run_id}"),
            None,
        )
    }

    #[test]
    fn test_paired_diff_matched_seeds() {
        let run_a = report_summary("run-1", 0, 42, [("pop", 100.0)]);
        let run_b = report_summary("run-2", 1, 42, [("pop", 110.0)]);

        // bd-7453: a SECOND pair with a different difference. The original single pair had
        // structurally zero variance -- `(n - 1).max(1.0)` -- so it now returns
        // ZeroVariance. Widened rather than inverted, to keep this test's actual subject:
        // that matched seeds pair up and produce the right mean difference.
        let run_c = report_summary("run-3", 0, 43, [("pop", 100.0)]);
        let run_d = report_summary("run-4", 1, 43, [("pop", 120.0)]);

        let effect = paired_diff(&[run_a, run_c], &[run_b, run_d], "pop").unwrap();
        assert_eq!(effect.n_pairs, 2);
        assert_eq!(effect.mean_diff, 15.0);
        // The standardized effect must be a real quotient now, never the old 0.0 fallback.
        assert!(effect.cohens_dz.is_finite() && effect.cohens_dz != 0.0);
    }

    /// bd-7453: a degenerate spread must be REFUSED, not reported as "no effect".
    ///
    /// Two matched pairs whose differences are identical: the mean difference is a large
    /// +10.0 and the spread is exactly zero. Under the old fallback this returned
    /// `cohens_dz = 0.0` — "no effect" for a perfectly consistent one. It must now return
    /// the error variant that was written for this case and never constructed.
    #[test]
    fn paired_diff_refuses_a_degenerate_spread_instead_of_reporting_no_effect() {
        let summary = |run_id: u64, seed: u64, value: f64| {
            report_summary(&format!("run-{run_id}"), 0, seed, [("pop", value)])
        };

        let control = [summary(1, 42, 100.0), summary(2, 43, 200.0)];
        let treatment = [summary(3, 42, 110.0), summary(4, 43, 210.0)];

        assert_eq!(
            paired_diff(&control, &treatment, "pop"),
            Err(StatsError::ZeroVariance),
            "every pair differs by exactly +10.0, so the spread is zero and no standardized \
             effect size exists; reporting 0.0 would claim no effect for a perfectly \
             consistent one"
        );
    }

    /// bd-h189: the p-value must be COMPUTED FROM THE DATA, never a constant.
    ///
    /// This is the anti-recurrence assertion, and it is the reusable part of the fix. The
    /// defect was `p_value: 0.05` returned on every call — a plausible number sitting exactly
    /// on the conventional significance threshold, indistinguishable from a computed one at
    /// every call site. Fixing the value alone would not stop it coming back.
    ///
    /// The property asserted is RESPONSIVENESS, not any particular number: two cohorts whose
    /// effects differ in strength must produce DIFFERENT p-values, and the more consistent
    /// effect must produce the SMALLER one. Any constant fails both halves, whatever constant
    /// someone picks — which is what a test pinning a specific p-value would not achieve.
    ///
    /// Deliberately does NOT assert `p_value != 0.05`. That would guard only the one literal
    /// that happened to be there and would pass for any other hardcoded value.
    #[test]
    fn p_value_is_computed_from_the_data_and_not_a_constant() {
        let cohort = |ids: [u64; 4], values: [f64; 4]| -> Vec<RunSummary> {
            ids.iter()
                .zip(values.iter())
                .enumerate()
                .map(|(index, (run_id, value))| {
                    report_summary(
                        &format!("run-{run_id}"),
                        0,
                        100 + index as u64,
                        [("pop", *value)],
                    )
                })
                .collect()
        };

        let control = cohort([1, 2, 3, 4], [100.0, 100.0, 100.0, 100.0]);
        // Strong: every pair moves the same way by a large, near-consistent amount.
        let strong = cohort([5, 6, 7, 8], [140.0, 141.0, 139.0, 140.0]);
        // Weak: the pairs disagree in sign, so the mean difference is near zero.
        let weak = cohort([9, 10, 11, 12], [101.0, 99.0, 102.0, 98.0]);

        let p_strong = paired_diff(&control, &strong, "pop")
            .expect("strong cohort has nonzero spread")
            .p_value;
        let p_weak = paired_diff(&control, &weak, "pop")
            .expect("weak cohort has nonzero spread")
            .p_value;

        assert!(
            (p_strong - p_weak).abs() > f64::EPSILON,
            "identical p-values ({p_strong}) for cohorts with different effects means the \
             value is not computed from the data — this is exactly the bd-h189 defect"
        );
        assert!(
            p_strong < p_weak,
            "a large consistent effect must be MORE significant than a near-zero one, got \
             strong={p_strong} weak={p_weak}"
        );
        for p in [p_strong, p_weak] {
            assert!(
                p > 0.0 && p <= 1.0,
                "a permutation p-value must lie in (0, 1], got {p}"
            );
        }
    }

    #[test]
    fn paired_test_distinguishes_a_balanced_null_from_an_injected_shift() {
        let control = (1_u64..=6)
            .map(|seed| report_summary(&format!("c{seed}"), 0, seed, [("score", 100.0)]))
            .collect::<Vec<_>>();
        let balanced_null = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
            .into_iter()
            .zip(1_u64..=6)
            .map(|(difference, seed)| {
                report_summary(
                    &format!("n{seed}"),
                    1,
                    seed,
                    [("score", 100.0 + difference)],
                )
            })
            .collect::<Vec<_>>();
        let injected_shift = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .zip(1_u64..=6)
            .map(|(difference, seed)| {
                report_summary(
                    &format!("s{seed}"),
                    1,
                    seed,
                    [("score", 100.0 + difference)],
                )
            })
            .collect::<Vec<_>>();

        let null_p = paired_diff(&control, &balanced_null, "score")
            .expect("balanced null has measurable spread")
            .p_value;
        let shifted_p = paired_diff(&control, &injected_shift, "score")
            .expect("injected shift has measurable spread")
            .p_value;
        assert!(
            null_p > 0.5,
            "a balanced sign-symmetric null must not be reported as evidence: {null_p}"
        );
        assert!(
            shifted_p < 0.1,
            "an injected all-positive shift must be distinguishable from the null: {shifted_p}"
        );
    }

    /// bd-h189: the same cohorts must always yield the same p-value.
    ///
    /// A permutation test resamples, so it would be easy to make it irreproducible. This
    /// module's other randomised routine is deterministic for the same reason.
    #[test]
    fn p_value_is_deterministic_across_repeated_calls() {
        let run = |run_id: u64, seed: u64, value: f64| {
            report_summary(&format!("run-{run_id}"), 0, seed, [("pop", value)])
        };
        let control = [run(1, 7, 100.0), run(2, 8, 100.0)];
        let treatment = [run(3, 7, 130.0), run(4, 8, 120.0)];

        let first = paired_diff(&control, &treatment, "pop")
            .expect("effect")
            .p_value;
        let second = paired_diff(&control, &treatment, "pop")
            .expect("effect")
            .p_value;
        assert_eq!(first, second, "permutation p-value must be reproducible");
    }

    /// bd-7453: one pair cannot support a standardized effect size either.
    #[test]
    fn paired_diff_refuses_a_single_pair() {
        let one = |run_id: u64, value: f64| {
            report_summary(&format!("run-{run_id}"), 0, 42, [("pop", value)])
        };
        assert_eq!(
            paired_diff(&[one(1, 100.0)], &[one(2, 110.0)], "pop"),
            Err(StatsError::ZeroVariance)
        );
    }

    #[test]
    fn paired_effect_is_scale_invariant_and_rejects_duplicate_or_detached_rows() {
        let tiny_control = [
            report_summary("tc1", 0, 1, [("score", 0.0)]),
            report_summary("tc2", 0, 2, [("score", 0.0)]),
        ];
        let tiny_treatment = [
            report_summary("tt1", 1, 1, [("score", 1e-13)]),
            report_summary("tt2", 1, 2, [("score", 2e-13)]),
        ];
        let unit_control = [
            report_summary("uc1", 0, 1, [("score", 0.0)]),
            report_summary("uc2", 0, 2, [("score", 0.0)]),
        ];
        let unit_treatment = [
            report_summary("ut1", 1, 1, [("score", 1.0)]),
            report_summary("ut2", 1, 2, [("score", 2.0)]),
        ];
        let tiny = paired_diff(&tiny_control, &tiny_treatment, "score").unwrap();
        let unit = paired_diff(&unit_control, &unit_treatment, "score").unwrap();
        assert!((tiny.cohens_dz - unit.cohens_dz).abs() < 1e-12);

        assert_eq!(
            paired_diff(
                &[
                    report_summary("dc1", 0, 1, [("score", 1.0)]),
                    report_summary("dc2", 0, 1, [("score", 2.0)]),
                ],
                &unit_treatment,
                "score",
            ),
            Err(StatsError::DuplicateSeed { arm_id: 0, seed: 1 })
        );

        let mut detached = report_summary("detached", 1, 1, [("score", 1.0)]);
        detached.metrics.insert("score".to_owned(), 99.0);
        assert_eq!(
            analyze_matched_seed_runs(
                &[report_summary("control", 0, 1, [("score", 0.0)]), detached,],
                &["score".to_owned()],
                AnalysisParams {
                    bootstrap_iterations: 10,
                    permutation_iterations: 10,
                    ..AnalysisParams::default()
                },
            ),
            Err(StatsError::RunSummaryIntegrity {
                run_id: "detached".to_owned(),
            })
        );
    }

    #[test]
    fn test_hedges_g_bias_correction() {
        let cohort = |first_run_id: u64, arm_id: u16, values: &[f64]| {
            values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    report_summary(
                        &format!("run-{}", first_run_id + index as u64),
                        arm_id,
                        100 + index as u64,
                        [("score", *value)],
                    )
                })
                .collect::<Vec<_>>()
        };
        let a = cohort(1, 0, &[-1.0, 0.0, 1.0, 2.0, 3.0]);
        let b = cohort(6, 1, &[-2.0, -1.0, 0.0, 1.0, 2.0]);

        let effect = hedges_g(&a, &b, "score").unwrap();
        assert!((effect.statistic - (-0.571_250_157_965_900_7)).abs() < 1e-12);
        assert!((effect.ci_95.0 - (-1.835_869_494_185_041_5)).abs() < 1e-12);
        assert!((effect.ci_95.1 - 0.693_369_178_253_24).abs() < 1e-12);
        assert_eq!(effect.n, 10);
        assert!(!effect.underpowered);
    }

    #[test]
    fn hedges_g_refuses_degenerate_pooled_spread() {
        let summary = |run_id: u64, arm_id: u16, value: f64| {
            report_summary(&format!("run-{run_id}"), arm_id, run_id, [("score", value)])
        };
        let control = [summary(1, 0, 10.0), summary(2, 0, 10.0)];
        let treatment = [summary(3, 1, 20.0), summary(4, 1, 20.0)];

        assert_eq!(
            hedges_g(&control, &treatment, "score"),
            Err(StatsError::ZeroVariance),
            "constant cohorts with different means have an unbounded standardized effect, \
             not a zero effect"
        );

        let equal_treatment = [summary(5, 1, 10.0), summary(6, 1, 10.0)];
        assert_eq!(
            hedges_g(&control, &equal_treatment, "score"),
            Err(StatsError::ZeroVariance),
            "equal constant cohorts still cannot support a standardized effect estimate"
        );

        assert_eq!(
            hedges_g(&control[..1], &treatment, "score"),
            Err(StatsError::InsufficientSamples),
            "each independent cohort needs at least two observations"
        );

        let near_constant_control = [summary(7, 0, 0.0), summary(8, 0, 1e-13)];
        let near_constant_treatment = [summary(9, 1, 1.0), summary(10, 1, 1.0 + 1e-13)];
        assert_eq!(
            hedges_g(&near_constant_control, &near_constant_treatment, "score"),
            Err(StatsError::ZeroVariance),
            "spread below the scale-relative degeneracy boundary must be refused"
        );

        let ordinary_control = [summary(11, 0, 0.0), summary(12, 0, 1.0)];
        let ordinary_treatment = [summary(13, 1, 2.0), summary(14, 1, 3.0)];
        let ordinary = hedges_g(&ordinary_control, &ordinary_treatment, "score").unwrap();

        let translated_control = [
            summary(15, 0, 1_000_000_000_000.0),
            summary(16, 0, 1_000_000_000_001.0),
        ];
        let translated_treatment = [
            summary(17, 1, 1_000_000_000_002.0),
            summary(18, 1, 1_000_000_000_003.0),
        ];
        let translated = hedges_g(&translated_control, &translated_treatment, "score").unwrap();
        assert_eq!(
            ordinary.statistic, translated.statistic,
            "adding a common origin must not change a standardized effect"
        );
        assert_eq!(
            ordinary.ci_95, translated.ci_95,
            "adding a common origin must not change its interval"
        );

        let scaled_control = [summary(19, 0, 0.0), summary(20, 0, 1e-13)];
        let scaled_treatment = [summary(21, 1, 2e-13), summary(22, 1, 3e-13)];
        let scaled = hedges_g(&scaled_control, &scaled_treatment, "score").unwrap();
        assert!(
            (ordinary.statistic - scaled.statistic).abs() < 1e-12,
            "uniform scaling must not change a standardized effect"
        );
        assert!(
            (ordinary.ci_95.0 - scaled.ci_95.0).abs() < 1e-12
                && (ordinary.ci_95.1 - scaled.ci_95.1).abs() < 1e-12,
            "uniform scaling must not change its interval"
        );
    }

    #[test]
    fn hedges_g_ci_width_responds_to_dispersion_and_sample_size() {
        let cohort = |first_run_id: u64, arm_id: u16, values: &[f64]| {
            values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    report_summary(
                        &format!("run-{}", first_run_id + index as u64),
                        arm_id,
                        index as u64,
                        [("score", *value)],
                    )
                })
                .collect::<Vec<_>>()
        };
        let control_values = [0.0, 1.0, 2.0, 3.0, 4.0];
        let narrow_values = [10.0, 11.0, 12.0, 13.0, 14.0];
        let diffuse_control_values = [-18.0, -8.0, 2.0, 12.0, 22.0];
        let diffuse_values = [-8.0, 2.0, 12.0, 22.0, 32.0];

        let control = cohort(1, 0, &control_values);
        let narrow = cohort(10, 1, &narrow_values);
        let diffuse_control = cohort(20, 0, &diffuse_control_values);
        let diffuse = cohort(30, 1, &diffuse_values);
        let narrow_effect = hedges_g(&control, &narrow, "score").unwrap();
        let diffuse_effect = hedges_g(&diffuse_control, &diffuse, "score").unwrap();
        let narrow_width = narrow_effect.ci_95.1 - narrow_effect.ci_95.0;
        let diffuse_width = diffuse_effect.ci_95.1 - diffuse_effect.ci_95.0;

        assert!(
            (narrow_width - 5.587_284_583_568_884).abs() < 1e-12,
            "tight-cohort CI width drifted: {narrow_width}"
        );
        assert!(
            (diffuse_width - 2.529_238_672_438_281_4).abs() < 1e-12,
            "diffuse-cohort CI width drifted: {diffuse_width}"
        );
        assert!(
            (narrow_width - diffuse_width).abs() > 0.5,
            "cohorts with the same means and sample sizes but different dispersion must not \
             receive a constant-width interval: narrow={narrow_width}, diffuse={diffuse_width}"
        );

        let repeated_control = control_values.repeat(10);
        let repeated_narrow = narrow_values.repeat(10);
        let large_control = cohort(100, 0, &repeated_control);
        let large_narrow = cohort(200, 1, &repeated_narrow);
        let large_effect = hedges_g(&large_control, &large_narrow, "score").unwrap();
        let large_width = large_effect.ci_95.1 - large_effect.ci_95.0;

        assert!(
            large_width < narrow_width,
            "more observations of the same cohort pattern must narrow the interval: \
             small={narrow_width}, large={large_width}"
        );
    }

    #[test]
    fn holm_bonferroni_reports_ranked_values_in_original_order() {
        let comparisons = adjust_multiple_comparisons(
            &[0.04, 0.001, 0.03, 0.20],
            0.05,
            Correction::HolmBonferroni,
        )
        .unwrap();

        assert_eq!(
            comparisons
                .iter()
                .map(|comparison| comparison.original_index)
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert_eq!(
            comparisons
                .iter()
                .map(|comparison| comparison.rank)
                .collect::<Vec<_>>(),
            [3, 1, 2, 4]
        );
        for (comparison, expected) in comparisons.iter().zip([0.09, 0.004, 0.09, 0.20]) {
            assert_close(comparison.adjusted_p_value, expected);
        }
        for (comparison, expected) in comparisons.iter().zip([0.025, 0.0125, 0.05 / 3.0, 0.05]) {
            assert_close(comparison.adjusted_alpha, expected);
        }
        assert_eq!(
            comparisons
                .iter()
                .map(|comparison| comparison.rejected)
                .collect::<Vec<_>>(),
            [false, true, false, false]
        );
        assert!(
            comparisons
                .iter()
                .all(|comparison| comparison.correction == Correction::HolmBonferroni)
        );
    }

    #[test]
    fn holm_bonferroni_stops_after_the_first_failed_rank() {
        let comparisons = adjust_multiple_comparisons(
            &[0.01, 0.02, 0.021, 0.022],
            0.05,
            Correction::HolmBonferroni,
        )
        .unwrap();

        for (comparison, expected) in comparisons.iter().zip([0.04, 0.06, 0.06, 0.06]) {
            assert_close(comparison.adjusted_p_value, expected);
        }
        assert_eq!(
            comparisons
                .iter()
                .map(|comparison| comparison.rejected)
                .collect::<Vec<_>>(),
            [true, false, false, false],
            "ranks three and four pass their individual alpha thresholds but must remain \
             unrejected after rank two fails"
        );
    }

    #[test]
    fn holm_bonferroni_uses_inclusive_thresholds_and_stable_ties() {
        let boundary = adjust_multiple_comparisons(
            &[0.05, 0.025, 1.0 / 60.0, 0.0125],
            0.05,
            Correction::HolmBonferroni,
        )
        .unwrap();
        assert!(boundary.iter().all(|comparison| comparison.rejected));
        for comparison in &boundary {
            assert_close(comparison.adjusted_p_value, 0.05);
            assert_close(comparison.raw_p_value, comparison.adjusted_alpha);
        }

        let ties =
            adjust_multiple_comparisons(&[0.04, 0.01, 0.01], 0.05, Correction::HolmBonferroni)
                .unwrap();
        assert_eq!(
            ties.iter()
                .map(|comparison| comparison.rank)
                .collect::<Vec<_>>(),
            [3, 1, 2],
            "equal p-values must be ranked by original index"
        );
        for (comparison, expected) in ties.iter().zip([0.04, 0.03, 0.03]) {
            assert_close(comparison.adjusted_p_value, expected);
        }

        let signed_zero_ties =
            adjust_multiple_comparisons(&[0.0, -0.0], 0.05, Correction::HolmBonferroni).unwrap();
        assert_eq!(
            signed_zero_ties
                .iter()
                .map(|comparison| comparison.rank)
                .collect::<Vec<_>>(),
            [1, 2],
            "numerically equal signed zeros must use the original-index tie break"
        );
    }

    #[test]
    fn benjamini_hochberg_is_a_value_level_step_up_adjustment() {
        let comparisons = adjust_multiple_comparisons(
            &[0.001, 0.03, 0.04, 0.20],
            0.05,
            Correction::BenjaminiHochberg,
        )
        .unwrap();

        for (comparison, expected) in comparisons
            .iter()
            .zip([0.004, 0.16 / 3.0, 0.16 / 3.0, 0.20])
        {
            assert_close(comparison.adjusted_p_value, expected);
        }
        for (comparison, expected) in comparisons.iter().zip([0.0125, 0.025, 0.0375, 0.05]) {
            assert_close(comparison.adjusted_alpha, expected);
        }
        assert_eq!(
            comparisons
                .iter()
                .map(|comparison| comparison.rejected)
                .collect::<Vec<_>>(),
            [true, false, false, false]
        );

        let step_up =
            adjust_multiple_comparisons(&[0.03, 0.04], 0.05, Correction::BenjaminiHochberg)
                .unwrap();
        assert!(
            step_up.iter().all(|comparison| comparison.rejected),
            "the largest passing rank rejects the whole prefix even when rank one misses its \
             individual critical value"
        );
        for comparison in step_up {
            assert_close(comparison.adjusted_p_value, 0.04);
        }
    }

    #[test]
    fn unadjusted_comparisons_pass_values_through() {
        let comparisons =
            adjust_multiple_comparisons(&[0.20, 0.01], 0.05, Correction::None).unwrap();

        assert_eq!(
            comparisons
                .iter()
                .map(|comparison| comparison.rank)
                .collect::<Vec<_>>(),
            [2, 1]
        );
        for (comparison, expected) in comparisons.iter().zip([0.20, 0.01]) {
            assert_close(comparison.raw_p_value, expected);
            assert_close(comparison.adjusted_p_value, expected);
            assert_close(comparison.adjusted_alpha, 0.05);
        }
        assert_eq!(
            comparisons
                .iter()
                .map(|comparison| comparison.rejected)
                .collect::<Vec<_>>(),
            [false, true]
        );
    }

    #[test]
    fn multiple_comparison_inputs_are_validated() {
        assert_eq!(
            adjust_multiple_comparisons(&[], 0.05, Correction::HolmBonferroni),
            Err(StatsError::NoSamples)
        );
        for alpha in [0.0, -0.0, 1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                adjust_multiple_comparisons(&[0.01], alpha, Correction::HolmBonferroni),
                Err(StatsError::InvalidSignificanceLevel)
            );
        }
        for p_value in [-0.01, 1.01, f64::NAN, f64::INFINITY] {
            assert_eq!(
                adjust_multiple_comparisons(&[0.01, p_value], 0.05, Correction::HolmBonferroni,),
                Err(StatsError::InvalidPValue { index: 1 })
            );
        }
    }

    #[test]
    fn bootstrap_refuses_fabricated_boundaries_and_is_order_invariant() {
        assert_eq!(bootstrap_ci(&[], 10, 7), Err(StatsError::NoSamples));
        assert_eq!(
            bootstrap_ci(&[1.0, 2.0], 0, 7),
            Err(StatsError::ZeroIterations)
        );
        assert_eq!(
            bootstrap_ci(&[1.0], 10, 7),
            Err(StatsError::InsufficientSamples)
        );
        assert_eq!(
            bootstrap_ci(&[1.0, f64::NAN], 10, 7),
            Err(StatsError::NonFiniteValue)
        );

        let forward = bootstrap_ci(&[1.0, 3.0, 8.0, 13.0], 2_000, 7).unwrap();
        let reversed = bootstrap_ci(&[13.0, 8.0, 3.0, 1.0], 2_000, 7).unwrap();
        assert_eq!(
            forward, reversed,
            "the same empirical distribution must not change because rows arrived in another order"
        );
        assert!(
            forward.0 < forward.1,
            "non-degenerate data must produce a data-derived interval"
        );
    }

    #[test]
    fn spearman_uses_average_ranks_for_ties_and_refuses_undefined_inputs() {
        let correlation =
            spearman_rank_correlation(&[1.0, 2.0, 2.0, 4.0], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_close(correlation, 0.948_683_298_050_513_8);

        let permuted =
            spearman_rank_correlation(&[2.0, 4.0, 1.0, 2.0], &[30.0, 40.0, 10.0, 20.0]).unwrap();
        assert_eq!(
            correlation, permuted,
            "reordering matched rows must preserve the rank correlation"
        );
        assert_eq!(
            spearman_rank_correlation(&[1.0, 1.0], &[2.0, 3.0]),
            Err(StatsError::ZeroVariance)
        );
        assert_eq!(
            spearman_rank_correlation(&[1.0], &[2.0, 3.0]),
            Err(StatsError::UnmatchedSeeds)
        );
        assert_eq!(
            spearman_rank_correlation(&[1.0, f64::INFINITY], &[2.0, 3.0]),
            Err(StatsError::NonFiniteValue)
        );
    }

    #[test]
    fn matched_seed_report_is_corrected_typed_and_input_order_invariant() {
        let mut summaries = vec![
            report_summary("c1", 0, 1, [("energy", 10.0), ("population", 1.0)]),
            report_summary("c2", 0, 2, [("energy", 20.0), ("population", 2.0)]),
            report_summary("c3", 0, 3, [("energy", 30.0), ("population", 3.0)]),
            report_summary("c4", 0, 4, [("energy", 40.0), ("population", 4.0)]),
            report_summary("a1", 1, 1, [("energy", 12.0), ("population", 4.0)]),
            report_summary("a2", 1, 2, [("energy", 23.0), ("population", 4.0)]),
            report_summary("a3", 1, 3, [("energy", 34.0), ("population", 3.0)]),
            report_summary("a4", 1, 4, [("energy", 45.0), ("population", 5.0)]),
            report_summary("b1", 2, 1, [("energy", 20.0), ("population", 2.0)]),
            report_summary("b2", 2, 2, [("energy", 30.0), ("population", 4.0)]),
            report_summary("b3", 2, 3, [("energy", 40.0), ("population", 5.0)]),
            report_summary("b4", 2, 4, [("energy", 50.0), ("population", 8.0)]),
        ];
        let metrics = vec!["population".to_owned(), "energy".to_owned()];
        let params = AnalysisParams {
            correction: Correction::HolmBonferroni,
            alternative: AlternativeHypothesis::TreatmentGreater,
            bootstrap_iterations: 500,
            permutation_iterations: 1_000,
            resampling_seed: 91,
            ..AnalysisParams::default()
        };

        let expected = analyze_matched_seed_runs(&summaries, &metrics, params).unwrap();
        assert_eq!(expected.effects.len(), 4);
        assert_eq!(
            expected
                .effects
                .iter()
                .map(|effect| (effect.treatment_arm, effect.metric.as_str()))
                .collect::<Vec<_>>(),
            [
                (1, "population"),
                (1, "energy"),
                (2, "population"),
                (2, "energy"),
            ]
        );
        assert!(
            expected
                .effects
                .iter()
                .all(|effect| effect.adjusted.correction == Correction::HolmBonferroni)
        );
        assert!(expected.effects.iter().all(|effect| {
            effect.alternative == AlternativeHypothesis::TreatmentGreater
                && effect.n_pairs == 4
                && effect.underpowered_reason
                    == Some(UnderpoweredReason::FewerThanRecommendedPairs {
                        have: 4,
                        recommended: 10,
                    })
        }));

        let constant_difference = &expected.effects[3];
        assert_eq!(constant_difference.mean_difference, 10.0);
        assert_eq!(constant_difference.standardized_effect, None);
        assert_eq!(
            constant_difference.undefined_reason,
            Some(UndefinedReason::ZeroVariance)
        );
        assert!(constant_difference.ci_95.is_some());
        assert_eq!(constant_difference.ci_undefined_reason, None);

        summaries.reverse();
        let reordered = analyze_matched_seed_runs(&summaries, &metrics, params).unwrap();
        assert_eq!(
            expected, reordered,
            "report ordering and values must be independent of summary arrival order"
        );
    }

    #[test]
    fn matched_seed_report_rejects_malformed_cohorts_and_changes_with_data() {
        let metrics = vec!["alive_agents".to_owned()];
        let params = AnalysisParams {
            bootstrap_iterations: 100,
            permutation_iterations: 100,
            ..AnalysisParams::default()
        };
        let control = report_summary("c1", 0, 1, [("alive_agents", 10.0)]);
        let treatment = report_summary("t1", 1, 1, [("alive_agents", 12.0)]);
        let original =
            analyze_matched_seed_runs(&[control.clone(), treatment.clone()], &metrics, params)
                .unwrap();
        let changed = analyze_matched_seed_runs(
            &[
                control.clone(),
                report_summary("t1", 1, 1, [("alive_agents", 99.0)]),
            ],
            &metrics,
            params,
        )
        .unwrap();
        assert_ne!(
            original, changed,
            "changing a measured value must change the report; constants cannot pass this control"
        );
        assert_eq!(
            analyze_matched_seed_runs(
                &[control.clone(), treatment.clone()],
                &metrics,
                AnalysisParams {
                    recommended_pairs: 0,
                    ..params
                },
            ),
            Err(StatsError::InvalidRecommendedPairs)
        );

        assert_eq!(
            analyze_matched_seed_runs(&[control.clone()], &metrics, params),
            Err(StatsError::InsufficientArms)
        );
        assert_eq!(
            analyze_matched_seed_runs(
                &[control.clone(), control.clone(), treatment.clone()],
                &metrics,
                params,
            ),
            Err(StatsError::DuplicateSeed { arm_id: 0, seed: 1 })
        );
        assert_eq!(
            analyze_matched_seed_runs(
                &[control, treatment],
                &["alive_agents".to_owned(), "alive_agents".to_owned()],
                params,
            ),
            Err(StatsError::DuplicateMetric("alive_agents".to_owned()))
        );
        assert_eq!(
            analyze_matched_seed_runs(
                &[
                    report_summary("c1", 0, 1, [("alive_agents", 10.0)]),
                    report_summary("t1", 1, 2, [("alive_agents", 12.0)]),
                ],
                &metrics,
                params,
            ),
            Err(StatsError::UnmatchedSeeds)
        );
    }

    #[test]
    fn declared_alternative_changes_the_tail_that_is_tested() {
        let summaries = [
            report_summary("c1", 0, 1, [("score", 0.0)]),
            report_summary("c2", 0, 2, [("score", 0.0)]),
            report_summary("c3", 0, 3, [("score", 0.0)]),
            report_summary("c4", 0, 4, [("score", 0.0)]),
            report_summary("t1", 1, 1, [("score", 3.0)]),
            report_summary("t2", 1, 2, [("score", 4.0)]),
            report_summary("t3", 1, 3, [("score", 5.0)]),
            report_summary("t4", 1, 4, [("score", 6.0)]),
        ];
        let metrics = ["score".to_owned()];
        let base = AnalysisParams {
            bootstrap_iterations: 100,
            permutation_iterations: 4_000,
            resampling_seed: 73,
            ..AnalysisParams::default()
        };
        let greater = analyze_matched_seed_runs(
            &summaries,
            &metrics,
            AnalysisParams {
                alternative: AlternativeHypothesis::TreatmentGreater,
                ..base
            },
        )
        .unwrap();
        let less = analyze_matched_seed_runs(
            &summaries,
            &metrics,
            AnalysisParams {
                alternative: AlternativeHypothesis::TreatmentLess,
                ..base
            },
        )
        .unwrap();
        assert!(
            greater.effects[0].raw_p_value < less.effects[0].raw_p_value,
            "a positive shift must be evidence for the greater tail, not the less tail"
        );
        assert_eq!(
            greater.effects[0].alternative,
            AlternativeHypothesis::TreatmentGreater
        );
        assert_eq!(
            less.effects[0].alternative,
            AlternativeHypothesis::TreatmentLess
        );
    }
}
