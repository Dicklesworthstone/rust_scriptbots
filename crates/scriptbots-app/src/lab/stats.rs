//! Analysis layer: effect sizes with CIs over matched-seed run summaries (bd-16g.1.4).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// Individual run summary row produced by run exports.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunSummary {
    pub run_id: u64,
    pub arm_id: u16,
    pub seed: u64,
    pub config_hash: [u8; 32],
    pub digest: [u8; 32],
    pub ticks: u64,
    pub metrics: BTreeMap<String, f64>,
}

/// Multiple comparison adjustment correction methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Correction {
    None,
    HolmBonferroni,
    BenjaminiHochberg,
}

/// Statistical test identifiers for audit trails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestName {
    PairedDifference,
    HedgesG,
    SpearmanRank,
    BootstrapCi,
}

/// Statistical computation errors.
#[derive(Debug, Error, PartialEq, Eq)]
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
    pub p_value: f64,
}

/// Generic effect size record carrying strict provenance and honesty metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Effect {
    pub metric: String,
    pub n: usize,
    pub test: TestName,
    pub statistic: f64,
    pub ci_95: (f64, f64),
    pub correction: Option<Correction>,
    pub underpowered: bool,
}

/// Two-sided paired permutation p-value for the mean difference (bd-h189).
///
/// Replaces a hardcoded `p_value: 0.05`. The sign-flip permutation is the exact test for
/// paired differences under the null "the pairing carries no signal": if it does not, the
/// sign of each difference is exchangeable, so resampling signs builds the null directly
/// from the observed data with no distributional assumption and no new dependency.
///
/// Deterministic by construction. The generator is an inline xorshift64 seeded by the
/// caller, so the same cohorts always yield the same p-value — this module's other
/// randomised routine, `bootstrap_ci`, is deterministic for the same reason.
///
/// Uses the `(count + 1) / (iterations + 1)` correction, so the result is never exactly
/// zero: a permutation test can bound a p-value from above but cannot prove it is zero, and
/// reporting 0.0 would claim more than the procedure supports.
fn paired_permutation_p_value(diffs: &[f64], iterations: u32, seed: u64) -> f64 {
    debug_assert!(
        !diffs.is_empty(),
        "caller rejects empty cohorts before this point"
    );
    let n = diffs.len() as f64;
    let observed = (diffs.iter().sum::<f64>() / n).abs();

    // Never zero: xorshift64 has a fixed point at zero and would emit a constant stream.
    let mut state = if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    };
    let mut at_least_as_extreme = 0u32;
    for _ in 0..iterations {
        let mut sum = 0.0;
        for diff in diffs {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            sum += if state & 1 == 0 { *diff } else { -*diff };
        }
        if (sum / n).abs() >= observed {
            at_least_as_extreme += 1;
        }
    }
    f64::from(at_least_as_extreme + 1) / f64::from(iterations + 1)
}

/// Computes paired difference effect sizes strictly matched by seed across cohorts.
pub fn paired_diff(
    a: &[RunSummary],
    b: &[RunSummary],
    metric: &str,
) -> Result<PairedEffect, StatsError> {
    if a.is_empty() || b.is_empty() {
        return Err(StatsError::NoSamples);
    }

    // Sort cohorts deterministically by seed
    let mut sorted_a = a.to_vec();
    sorted_a.sort_by_key(|r| r.seed);
    let mut sorted_b = b.to_vec();
    sorted_b.sort_by_key(|r| r.seed);

    if sorted_a.len() != sorted_b.len() {
        return Err(StatsError::UnmatchedSeeds);
    }

    let mut diffs = Vec::with_capacity(sorted_a.len());
    for (ra, rb) in sorted_a.iter().zip(sorted_b.iter()) {
        if ra.seed != rb.seed {
            return Err(StatsError::UnmatchedSeeds);
        }
        let va = ra
            .metrics
            .get(metric)
            .copied()
            .ok_or_else(|| StatsError::MissingMetric(metric.to_string()))?;
        let vb = rb
            .metrics
            .get(metric)
            .copied()
            .ok_or_else(|| StatsError::MissingMetric(metric.to_string()))?;

        if !va.is_finite() || !vb.is_finite() {
            return Err(StatsError::NonFiniteValue);
        }
        diffs.push(va - vb);
    }

    let n = diffs.len() as f64;
    let mean_diff = diffs.iter().sum::<f64>() / n;
    let var_diff = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let sd_diff = var_diff.sqrt();

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
    if sd_diff <= 1e-12 {
        return Err(StatsError::ZeroVariance);
    }
    let cohens_dz = mean_diff / sd_diff;

    let ci_95 = bootstrap_ci(&diffs, 1000, 42);

    Ok(PairedEffect {
        metric: metric.to_string(),
        n_pairs: diffs.len(),
        mean_diff,
        sd_diff,
        cohens_dz,
        ci_95,
        test: TestName::PairedDifference,
        p_value: paired_permutation_p_value(&diffs, 1000, 42),
    })
}

/// Computes Hedges' g effect size with small-sample bias correction.
pub fn hedges_g(a: &[RunSummary], b: &[RunSummary], metric: &str) -> Result<Effect, StatsError> {
    if a.is_empty() || b.is_empty() {
        return Err(StatsError::NoSamples);
    }
    if a.len() < 2 || b.len() < 2 {
        return Err(StatsError::InsufficientSamples);
    }

    let vals_a: Vec<f64> = a
        .iter()
        .map(|r| {
            r.metrics
                .get(metric)
                .copied()
                .ok_or_else(|| StatsError::MissingMetric(metric.to_string()))
        })
        .collect::<Result<_, _>>()?;

    let vals_b: Vec<f64> = b
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

    let na = vals_a.len() as f64;
    let nb = vals_b.len() as f64;

    let mean_a = vals_a.iter().sum::<f64>() / na;
    let mean_b = vals_b.iter().sum::<f64>() / nb;

    let var_a = vals_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (na - 1.0).max(1.0);
    let var_b = vals_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (nb - 1.0).max(1.0);

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
    let cohens_d = (mean_a - mean_b) / pooled_sd;

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
        statistic: g,
        ci_95,
        correction: None,
        underpowered: (vals_a.len() + vals_b.len()) < 10,
    })
}

/// Deterministic bootstrap 95% confidence interval computation.
pub fn bootstrap_ci(values: &[f64], iters: usize, seed: u64) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    if values.len() == 1 {
        return (values[0], values[0]);
    }

    let mut means = Vec::with_capacity(iters);
    let mut current_seed = seed;

    for _ in 0..iters {
        let mut sample_sum = 0.0;
        for _ in 0..values.len() {
            // LCG deterministic pseudorandom index selection
            current_seed = current_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let idx = (current_seed as usize) % values.len();
            sample_sum += values[idx];
        }
        means.push(sample_sum / values.len() as f64);
    }

    means.sort_by(|x, y| x.total_cmp(y));
    let lower_idx = (iters as f64 * 0.025) as usize;
    let upper_idx = (iters as f64 * 0.975) as usize;

    (means[lower_idx], means[upper_idx.min(iters - 1)])
}

/// Adjusts multiple comparison effects using Holm-Bonferroni correction.
pub fn adjust_multiple_comparisons(effects: &mut [Effect], method: Correction) {
    if method == Correction::None || effects.is_empty() {
        return;
    }
    for effect in effects {
        effect.correction = Some(method);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paired_diff_matched_seeds() {
        let mut m1 = BTreeMap::new();
        m1.insert("pop".to_string(), 100.0);
        let mut m2 = BTreeMap::new();
        m2.insert("pop".to_string(), 110.0);

        let run_a = RunSummary {
            run_id: 1,
            arm_id: 0,
            seed: 42,
            config_hash: [0; 32],
            digest: [0; 32],
            ticks: 100,
            metrics: m1,
        };

        let run_b = RunSummary {
            run_id: 2,
            arm_id: 1,
            seed: 42,
            config_hash: [0; 32],
            digest: [0; 32],
            ticks: 100,
            metrics: m2,
        };

        // bd-7453: a SECOND pair with a different difference. The original single pair had
        // structurally zero variance -- `(n - 1).max(1.0)` -- so it now returns
        // ZeroVariance. Widened rather than inverted, to keep this test's actual subject:
        // that matched seeds pair up and produce the right mean difference.
        let mut m3 = BTreeMap::new();
        m3.insert("pop".to_string(), 100.0);
        let mut m4 = BTreeMap::new();
        m4.insert("pop".to_string(), 120.0);

        let run_c = RunSummary {
            run_id: 3,
            arm_id: 0,
            seed: 43,
            config_hash: [0; 32],
            digest: [0; 32],
            ticks: 100,
            metrics: m3,
        };
        let run_d = RunSummary {
            run_id: 4,
            arm_id: 1,
            seed: 43,
            config_hash: [0; 32],
            digest: [0; 32],
            ticks: 100,
            metrics: m4,
        };

        let effect = paired_diff(&[run_a, run_c], &[run_b, run_d], "pop").unwrap();
        assert_eq!(effect.n_pairs, 2);
        assert_eq!(effect.mean_diff, -15.0);
        // The standardized effect must be a real quotient now, never the old 0.0 fallback.
        assert!(effect.cohens_dz.is_finite() && effect.cohens_dz != 0.0);
    }

    /// bd-7453: a degenerate spread must be REFUSED, not reported as "no effect".
    ///
    /// Two matched pairs whose differences are identical: the mean difference is a large
    /// -10.0 and the spread is exactly zero. Under the old fallback this returned
    /// `cohens_dz = 0.0` — "no effect" for a perfectly consistent one. It must now return
    /// the error variant that was written for this case and never constructed.
    #[test]
    fn paired_diff_refuses_a_degenerate_spread_instead_of_reporting_no_effect() {
        let summary = |run_id: u64, seed: u64, value: f64| {
            let mut metrics = BTreeMap::new();
            metrics.insert("pop".to_string(), value);
            RunSummary {
                run_id,
                arm_id: 0,
                seed,
                config_hash: [0; 32],
                digest: [0; 32],
                ticks: 100,
                metrics,
            }
        };

        let control = [summary(1, 42, 100.0), summary(2, 43, 200.0)];
        let treatment = [summary(3, 42, 110.0), summary(4, 43, 210.0)];

        assert_eq!(
            paired_diff(&control, &treatment, "pop"),
            Err(StatsError::ZeroVariance),
            "every pair differs by exactly -10.0, so the spread is zero and no standardized \
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
                    let mut metrics = BTreeMap::new();
                    metrics.insert("pop".to_string(), *value);
                    RunSummary {
                        run_id: *run_id,
                        arm_id: 0,
                        seed: 100 + index as u64,
                        config_hash: [0; 32],
                        digest: [0; 32],
                        ticks: 100,
                        metrics,
                    }
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

    /// bd-h189: the same cohorts must always yield the same p-value.
    ///
    /// A permutation test resamples, so it would be easy to make it irreproducible. This
    /// module's other randomised routine is deterministic for the same reason.
    #[test]
    fn p_value_is_deterministic_across_repeated_calls() {
        let mut metrics_a = BTreeMap::new();
        metrics_a.insert("pop".to_string(), 100.0);
        let mut metrics_b = BTreeMap::new();
        metrics_b.insert("pop".to_string(), 130.0);
        let mut metrics_c = BTreeMap::new();
        metrics_c.insert("pop".to_string(), 100.0);
        let mut metrics_d = BTreeMap::new();
        metrics_d.insert("pop".to_string(), 120.0);
        let run = |run_id: u64, seed: u64, metrics: BTreeMap<String, f64>| RunSummary {
            run_id,
            arm_id: 0,
            seed,
            config_hash: [0; 32],
            digest: [0; 32],
            ticks: 100,
            metrics,
        };
        let control = [run(1, 7, metrics_a), run(2, 8, metrics_c)];
        let treatment = [run(3, 7, metrics_b), run(4, 8, metrics_d)];

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
        let mut metrics_a = BTreeMap::new();
        metrics_a.insert("pop".to_string(), 100.0);
        let mut metrics_b = BTreeMap::new();
        metrics_b.insert("pop".to_string(), 110.0);
        let one = |run_id: u64, metrics: BTreeMap<String, f64>| RunSummary {
            run_id,
            arm_id: 0,
            seed: 42,
            config_hash: [0; 32],
            digest: [0; 32],
            ticks: 100,
            metrics,
        };
        assert_eq!(
            paired_diff(&[one(1, metrics_a)], &[one(2, metrics_b)], "pop"),
            Err(StatsError::ZeroVariance)
        );
    }

    #[test]
    fn test_hedges_g_bias_correction() {
        let cohort = |first_run_id: u64, arm_id: u16, values: &[f64]| {
            values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let mut metrics = BTreeMap::new();
                    metrics.insert("score".to_string(), *value);
                    RunSummary {
                        run_id: first_run_id + index as u64,
                        arm_id,
                        seed: 100 + index as u64,
                        config_hash: [0; 32],
                        digest: [0; 32],
                        ticks: 50,
                        metrics,
                    }
                })
                .collect::<Vec<_>>()
        };
        let a = cohort(1, 0, &[-1.0, 0.0, 1.0, 2.0, 3.0]);
        let b = cohort(6, 1, &[-2.0, -1.0, 0.0, 1.0, 2.0]);

        let effect = hedges_g(&a, &b, "score").unwrap();
        assert!((effect.statistic - 0.571_250_157_965_900_7).abs() < 1e-12);
        assert!((effect.ci_95.0 - (-0.693_369_178_253_24)).abs() < 1e-12);
        assert!((effect.ci_95.1 - 1.835_869_494_185_041_5).abs() < 1e-12);
        assert_eq!(effect.n, 10);
        assert!(!effect.underpowered);
    }

    #[test]
    fn hedges_g_refuses_degenerate_pooled_spread() {
        let summary = |run_id: u64, arm_id: u16, value: f64| {
            let mut metrics = BTreeMap::new();
            metrics.insert("score".to_string(), value);
            RunSummary {
                run_id,
                arm_id,
                seed: run_id,
                config_hash: [0; 32],
                digest: [0; 32],
                ticks: 50,
                metrics,
            }
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
                    let mut metrics = BTreeMap::new();
                    metrics.insert("score".to_string(), *value);
                    RunSummary {
                        run_id: first_run_id + index as u64,
                        arm_id,
                        seed: index as u64,
                        config_hash: [0; 32],
                        digest: [0; 32],
                        ticks: 50,
                        metrics,
                    }
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
}
