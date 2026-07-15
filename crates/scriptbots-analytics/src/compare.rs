//! Matched-seed treatment-effect analysis (bd-2z0.11.6 item 3; serves bd-16g.1.4).
//!
//! # What a matched-seed study is, and why the pairing matters
//!
//! The lab assistant (bd-16g.1.4) tests a hypothesis by running the SAME seeds under two
//! conditions — a control and a treatment (say, mutation rate doubled) — and asking whether the
//! treatment changed an outcome. Because the seeds are matched, the data is PAIRED: for each seed
//! there is one control value and one treatment value, and they share everything except the
//! treatment. That pairing is not a nicety; it is the whole point. A paired analysis removes the
//! seed-to-seed variance that an unpaired comparison would drown in, so it detects a real effect
//! with far fewer replicates. Throwing the pairing away — pooling all control values against all
//! treatment values — would be answering an easier, wrong question.
//!
//! # The right null for paired data
//!
//! Under the null that the treatment does nothing, each seed's control and treatment values are
//! exchangeable, so the SIGN of each paired difference is equally likely to be `+` or `-`. That
//! gives an exact, assumption-free significance test — the sign-flip permutation test in
//! [`paired_comparison`] — rather than leaning on a normality assumption a 20-seed study cannot
//! justify.
//!
//! # Many metrics ⇒ the same multiple-testing trap as many events
//!
//! A study rarely asks about one outcome. Test population, energy, diversity, lifespan and a
//! dozen others each at α = 0.05 and you manufacture false "effects" exactly as a long event
//! stream does. So [`compare_metrics`] runs every metric and then applies the SAME
//! Benjamini-Hochberg control the event certifier uses ([`crate::certify::benjamini_hochberg`]) —
//! one honest false-discovery bar across the whole study.
//!
//! # Purity
//!
//! Everything is a pure function of two equal-length slices of matched values and a seed. The
//! DB glue that pulls per-seed metric outcomes from two run databases is a thin adapter a report
//! adds on top; the analysis — the part that is subtle — is proven here on synthetic data with a
//! known treatment effect.

// Resampling casts small counts to f64 (means over pairs, rank thresholds); exact in f64 for any
// realistic study. Allowed module-wide as in `stats`/`certify`.
#![allow(clippy::cast_precision_loss)]

use crate::certify::benjamini_hochberg;
use crate::stats::{ConfidenceInterval, StatsError, mean, quantile, std_dev};

/// A deterministic `SplitMix64`, local to this module.
///
/// `stats` has its own `DeterministicRng` but keeps its draw private, so rather than couple to
/// another module's internals this module carries its own identical stepper. Same algorithm, same
/// reproducibility guarantee: every result is a pure function of the seed.
#[derive(Debug, Clone)]
struct LocalRng {
    state: u64,
}

impl LocalRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `0..len` via Lemire multiply-shift (no modulo bias). `len` must be > 0.
    fn index(&mut self, len: usize) -> usize {
        let m = u128::from(self.next_u64()) * (len as u128);
        usize::try_from(m >> 64)
            .expect("multiply-shift quotient is strictly below its usize input bound")
    }

    /// A fair coin.
    const fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

/// Parameters for a matched-seed comparison.
#[derive(Debug, Clone, Copy)]
pub struct CompareParams {
    /// Bootstrap resamples for the paired-difference confidence interval.
    pub n_resamples: usize,
    /// Sign-flip permutations for the significance test. With `n` pairs there are `2^n` sign
    /// assignments; for small `n` we could enumerate them exactly, but a seeded random subset is
    /// simpler and adequate, and it degrades gracefully as `n` grows.
    pub n_permutations: usize,
    /// Confidence level for the paired-difference CI, e.g. `0.95`.
    pub confidence: f64,
    /// Target false-discovery rate for the across-metrics Benjamini-Hochberg pass.
    pub fdr: f64,
    /// Seed for resampling and sign-flipping. Fixed ⇒ reproducible.
    pub seed: u64,
}

impl Default for CompareParams {
    fn default() -> Self {
        Self {
            n_resamples: 2000,
            n_permutations: 4000,
            confidence: 0.95,
            fdr: 0.05,
            seed: 0x00C0_FFEE,
        }
    }
}

/// The result of comparing one metric across matched seeds.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedComparison {
    /// Mean of the per-seed differences (`treatment - control`). The treatment effect estimate.
    pub mean_difference: f64,
    /// Bootstrap CI on the mean paired difference. If it excludes zero, the direction of the
    /// effect is established at this confidence level.
    pub difference_ci: ConfidenceInterval,
    /// Sign-flip permutation p-value for "the treatment changed nothing".
    pub p_value: f64,
    /// Cohen's `d_z`: the mean paired difference divided by the SD of the paired differences —
    /// the standardized effect size for a paired design. Distinct from the two-sample Cohen's d,
    /// which would ignore the pairing.
    pub cohens_dz: f64,
    /// Fraction of seeds for which treatment exceeded control, in `[0, 1]`. A blunt but
    /// assumption-free companion to the p-value: a real effect usually moves most pairs the same
    /// way, and a "significant" result where only half the pairs agree deserves a second look.
    pub fraction_positive: f64,
    /// Number of matched pairs the comparison used.
    pub n_pairs: usize,
    /// Survives Benjamini-Hochberg across the study's metrics. For a single isolated metric it
    /// equals `p_value < fdr`. Set by [`compare_metrics`]; for a lone [`paired_comparison`] there
    /// is nothing to correct against.
    pub significant_fdr: bool,
}

/// Compare one metric's matched control and treatment outcomes.
///
/// `control[i]` and `treatment[i]` are the same seed under the two conditions, so the slices must
/// have equal length — a length mismatch is a category error (unpaired data handed to a paired
/// analysis), not a recoverable input, and it errors rather than silently truncating.
pub fn paired_comparison(
    control: &[f64],
    treatment: &[f64],
    params: &CompareParams,
) -> Result<PairedComparison, StatsError> {
    if control.len() != treatment.len() {
        return Err(StatsError::EmptySample {
            what: "compare.mismatched_pair_lengths",
        });
    }
    if control.is_empty() {
        return Err(StatsError::EmptySample {
            what: "compare.empty",
        });
    }
    finite(control, "compare.control")?;
    finite(treatment, "compare.treatment")?;
    if params.n_resamples == 0 || params.n_permutations == 0 {
        return Err(StatsError::ZeroResamples);
    }
    if !(params.confidence > 0.0 && params.confidence < 1.0) {
        return Err(StatsError::InvalidConfidence {
            level: params.confidence,
        });
    }

    let diffs: Vec<f64> = treatment.iter().zip(control).map(|(t, c)| t - c).collect();
    let mean_difference = mean(&diffs)?;

    // Bootstrap CI on the mean paired difference: resample the PAIRS (i.e. the differences) with
    // replacement, preserving the pairing.
    let mut rng = LocalRng::new(params.seed);
    let mut boot_means = Vec::with_capacity(params.n_resamples);
    for _ in 0..params.n_resamples {
        let mut sum = 0.0;
        for _ in 0..diffs.len() {
            sum += diffs[rng.index(diffs.len())];
        }
        boot_means.push(sum / diffs.len() as f64);
    }
    let alpha = 1.0 - params.confidence;
    let difference_ci = ConfidenceInterval {
        point: mean_difference,
        lower: quantile(&boot_means, alpha / 2.0)?,
        upper: quantile(&boot_means, 1.0 - alpha / 2.0)?,
        confidence: params.confidence,
        resamples: params.n_resamples,
    };

    // Sign-flip permutation test: under the null the sign of each difference is exchangeable.
    let observed = mean_difference.abs();
    let mut sign_rng = LocalRng::new(params.seed ^ 0x00F1_1900);
    let mut at_least_as_extreme = 0usize;
    for _ in 0..params.n_permutations {
        let mut sum = 0.0;
        for &d in &diffs {
            // A fresh fair coin per pair per permutation.
            if sign_rng.coin() {
                sum += d;
            } else {
                sum -= d;
            }
        }
        if (sum / diffs.len() as f64).abs() >= observed {
            at_least_as_extreme += 1;
        }
    }
    let p_value = (at_least_as_extreme as f64 + 1.0) / (params.n_permutations as f64 + 1.0);

    let sd_diff = std_dev(&diffs)?;
    let cohens_dz = if sd_diff == 0.0 {
        if mean_difference == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        mean_difference / sd_diff
    };
    let positive = diffs.iter().filter(|&&d| d > 0.0).count();
    let fraction_positive = positive as f64 / diffs.len() as f64;

    Ok(PairedComparison {
        mean_difference,
        difference_ci,
        p_value,
        cohens_dz,
        fraction_positive,
        n_pairs: diffs.len(),
        significant_fdr: p_value < params.fdr,
    })
}

/// One named metric's matched outcomes.
#[derive(Debug, Clone)]
pub struct MetricSeries<'a> {
    /// Stable metric name included in the comparison result.
    pub name: &'a str,
    /// Control-arm outcomes ordered by matched seed.
    pub control: &'a [f64],
    /// Treatment-arm outcomes in the same matched-seed order.
    pub treatment: &'a [f64],
}

/// A metric's comparison, tagged with its name.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedComparison {
    /// Stable metric name copied from the input series.
    pub metric: String,
    /// Matched-pair effect estimate and corrected significance decision.
    pub comparison: PairedComparison,
}

/// Compare many metrics across a matched-seed study, with Benjamini-Hochberg across them.
///
/// Each metric is compared independently, then the whole set of p-values goes through the SAME
/// FDR control the event certifier uses, and `significant_fdr` is set from the corrected decision.
/// This is what stops a study that measured twenty outcomes from reporting one "effect" that is
/// pure chance.
pub fn compare_metrics(
    metrics: &[MetricSeries<'_>],
    params: &CompareParams,
) -> Result<StudyComparison, StatsError> {
    let mut named = Vec::with_capacity(metrics.len());
    for metric in metrics {
        let comparison = paired_comparison(metric.control, metric.treatment, params)?;
        named.push(NamedComparison {
            metric: metric.name.to_owned(),
            comparison,
        });
    }

    let p_values: Vec<f64> = named.iter().map(|n| n.comparison.p_value).collect();
    let rejected = benjamini_hochberg(&p_values, params.fdr);
    for (n, &is_rejected) in named.iter_mut().zip(&rejected) {
        n.comparison.significant_fdr = is_rejected;
    }

    let discoveries = named
        .iter()
        .filter(|n| n.comparison.significant_fdr)
        .count();
    Ok(StudyComparison {
        metrics: named,
        target_fdr: params.fdr,
        discoveries,
    })
}

/// The comparison of a whole matched-seed study across every metric measured.
#[derive(Debug, Clone, PartialEq)]
pub struct StudyComparison {
    /// One comparison per metric, in input order.
    pub metrics: Vec<NamedComparison>,
    /// The false-discovery rate the across-metrics pass targeted.
    pub target_fdr: f64,
    /// How many metrics show a real treatment effect after FDR control — the number a report
    /// should headline, not the raw count of metrics whose uncorrected p-value fell below α.
    pub discoveries: usize,
}

// --- internal --------------------------------------------------------------------------------

fn finite(sample: &[f64], what: &'static str) -> Result<(), StatsError> {
    if sample.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(StatsError::NonFinite { what })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic normal draws for building matched fixtures.
    struct Normal {
        state: u64,
    }
    impl Normal {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn bits(&mut self) -> u64 {
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
        fn normal(&mut self, mean: f64, sd: f64) -> f64 {
            let u1 = self.unit();
            let u2 = self.unit();
            sd.mul_add(
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos(),
                mean,
            )
        }
    }

    /// A matched control/treatment pair of `n` seeds: control is N(base, sd), treatment is the
    /// SAME control value plus a per-seed treatment effect `N(effect, effect_sd)`.
    fn matched(
        n: usize,
        base: f64,
        sd: f64,
        effect: f64,
        effect_sd: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut draws = Normal::new(seed);
        let mut control = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        for _ in 0..n {
            let c = draws.normal(base, sd);
            control.push(c);
            treatment.push(c + draws.normal(effect, effect_sd));
        }
        (control, treatment)
    }

    #[test]
    fn a_real_treatment_effect_is_detected_with_its_direction() {
        let (control, treatment) = matched(30, 100.0, 20.0, 5.0, 1.0, 1);
        let c = paired_comparison(&control, &treatment, &CompareParams::default()).unwrap();
        assert!(
            c.significant_fdr,
            "a +5 paired effect was not detected (p={:.4})",
            c.p_value
        );
        assert!(c.mean_difference > 0.0);
        assert!(
            c.difference_ci.lower > 0.0,
            "the CI on the treatment effect should exclude zero: {:?}",
            c.difference_ci
        );
        assert!(
            c.cohens_dz > 2.0,
            "d_z for a 5-unit effect with sd~1 should be large: {}",
            c.cohens_dz
        );
        assert!(
            c.fraction_positive > 0.9,
            "most pairs should move up: {}",
            c.fraction_positive
        );
    }

    #[test]
    fn the_pairing_is_what_makes_a_small_effect_detectable() {
        // THE POINT OF A PAIRED DESIGN. Seed-to-seed spread (sd=20) dwarfs the treatment effect
        // (5). A paired analysis removes the shared seed variance and finds the effect; an
        // UNPAIRED pooling of the same numbers is swamped by it. Demonstrate both on identical data.
        let (control, treatment) = matched(30, 100.0, 20.0, 5.0, 1.0, 2);
        let paired = paired_comparison(&control, &treatment, &CompareParams::default()).unwrap();
        assert!(
            paired.significant_fdr,
            "paired analysis missed the effect (p={:.4})",
            paired.p_value
        );

        // Unpaired: pool control vs treatment ignoring which seed is which. With sd=20 noise and a
        // 5-unit shift over 30 samples, the unpaired permutation test should be far less certain.
        let unpaired =
            crate::stats::permutation_test_mean_difference(&control, &treatment, 4000, 9).unwrap();
        println!(
            "paired p={:.4} vs unpaired p={:.4} on the same data",
            paired.p_value, unpaired.p_value
        );
        assert!(
            paired.p_value < unpaired.p_value,
            "the paired test ({:.4}) should be more powerful than the unpaired one ({:.4}); if not, \
             the pairing is being wasted",
            paired.p_value,
            unpaired.p_value
        );
    }

    #[test]
    fn no_treatment_effect_is_not_certified() {
        // Treatment = control + zero-mean noise: no real effect. Must not be significant.
        let (control, treatment) = matched(40, 50.0, 10.0, 0.0, 2.0, 3);
        let c = paired_comparison(&control, &treatment, &CompareParams::default()).unwrap();
        assert!(
            !c.significant_fdr,
            "a null treatment was certified as an effect (p={:.4}); false positive",
            c.p_value
        );
    }

    #[test]
    fn fdr_keeps_the_real_metrics_and_drops_the_null_ones() {
        // A study measuring many outcomes: two have a real effect, several are null. FDR must keep
        // the real ones and suppress the flood of nulls — the multiple-testing protection.
        let (c_pop, t_pop) = matched(30, 200.0, 15.0, 6.0, 1.5, 10); // real
        let (c_energy, t_energy) = matched(30, 1.0, 0.2, 0.15, 0.03, 11); // real
        let (c_a, t_a) = matched(30, 10.0, 3.0, 0.0, 1.0, 12); // null
        let (c_b, t_b) = matched(30, 5.0, 2.0, 0.0, 1.0, 13); // null
        let (c_d, t_d) = matched(30, 7.0, 1.0, 0.0, 0.5, 14); // null
        let metrics = vec![
            MetricSeries {
                name: "population",
                control: &c_pop,
                treatment: &t_pop,
            },
            MetricSeries {
                name: "energy",
                control: &c_energy,
                treatment: &t_energy,
            },
            MetricSeries {
                name: "null_a",
                control: &c_a,
                treatment: &t_a,
            },
            MetricSeries {
                name: "null_b",
                control: &c_b,
                treatment: &t_b,
            },
            MetricSeries {
                name: "null_d",
                control: &c_d,
                treatment: &t_d,
            },
        ];
        let study = compare_metrics(&metrics, &CompareParams::default()).unwrap();

        let real: Vec<&str> = study
            .metrics
            .iter()
            .filter(|m| m.comparison.significant_fdr)
            .map(|m| m.metric.as_str())
            .collect();
        println!("study discoveries after FDR: {real:?}");
        assert!(
            real.contains(&"population"),
            "the real population effect was suppressed"
        );
        assert!(
            real.contains(&"energy"),
            "the real energy effect was suppressed"
        );
        for null in ["null_a", "null_b", "null_d"] {
            assert!(
                !real.contains(&null),
                "a null metric `{null}` was reported as a real effect"
            );
        }
        assert_eq!(
            study.discoveries, 2,
            "exactly the two real metrics should survive FDR"
        );
    }

    #[test]
    fn mismatched_pair_lengths_error() {
        let err = paired_comparison(&[1.0, 2.0, 3.0], &[1.0, 2.0], &CompareParams::default());
        assert!(matches!(err, Err(StatsError::EmptySample { .. })));
    }

    #[test]
    fn the_comparison_is_reproducible() {
        let (control, treatment) = matched(25, 100.0, 12.0, 3.0, 1.0, 55);
        let a = paired_comparison(&control, &treatment, &CompareParams::default()).unwrap();
        let b = paired_comparison(&control, &treatment, &CompareParams::default()).unwrap();
        assert_eq!(
            a, b,
            "same data and params must give a bit-identical comparison"
        );
    }
}
