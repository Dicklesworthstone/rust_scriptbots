//! Native distribution characterization for offline analysis (bd-2z0.11.6 item 2).
//!
//! # What this is, and why it avoids special functions
//!
//! The distribution-report needs to say whether a metric's values look normal, and how they
//! depart from it. The heavy way is to fit candidate distributions and run a Kolmogorov-Smirnov
//! test — which needs the normal CDF, i.e. `erf`, i.e. a transcribed numerical approximation. This
//! module takes the moment-based route instead: skewness, kurtosis, and the **Jarque-Bera** test,
//! whose statistic is built from those moments and whose null distribution is exactly chi-square
//! with two degrees of freedom. Chi-square(2) has a closed-form survival function —
//! `P(X ≥ x) = exp(-x/2)` — so the p-value needs no special function at all. Every number here is
//! a sum, a power, or a single `exp`; there is nothing to get subtly wrong in an approximation.
//!
//! That is deliberate: it is a portable, dependency-free normality assessment that needs neither
//! `fsci`'s distribution zoo nor a hand-rolled `erf`. Full distribution fitting (lognormal/gamma +
//! KS) is where `fsci` would earn its keep and is left for the adapter decision (bd-2z0.11.3);
//! this covers the "is it normal, and how is it shaped" question natively.
//!
//! # Purity
//!
//! A pure function of a slice. No RNG, no I/O.

// Moment code casts the sample size to f64 and compares variance to an exact-zero sentinel (a
// constant sample has variance exactly 0) by construction. Allowed module-wide, as in `stats`.
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::float_cmp)]

use crate::stats::StatsError;

/// Shape summary of a sample: its first four moments and a normality test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistributionSummary {
    pub n: usize,
    pub mean: f64,
    /// Population variance (divided by `n`), matching the moments the skewness/kurtosis use.
    pub variance: f64,
    pub std_dev: f64,
    /// Fisher-Pearson skewness `g1 = m3 / m2^1.5`. Zero for a symmetric distribution, positive for
    /// a right tail.
    pub skewness: f64,
    /// EXCESS kurtosis `m4 / m2^2 - 3`. Zero for a normal distribution, positive for heavy tails,
    /// negative for light ones (a uniform is about `-1.2`).
    pub excess_kurtosis: f64,
    /// Jarque-Bera statistic `(n/6)(S^2 + (K-3)^2/4)`. Large when the sample departs from normality
    /// in skew or tail weight.
    pub jarque_bera: f64,
    /// The test's p-value against the null that the sample is normal, from the exact chi-square(2)
    /// survival function `exp(-JB/2)`. Small ⇒ reject normality. A degenerate constant sample gets
    /// `1.0` — there is no shape to reject.
    pub jb_p_value: f64,
    /// True for a constant sample, whose shape moments are undefined and are reported as zero
    /// rather than `NaN`. A reader must not read "skewness 0, excess_kurtosis 0" as "looks normal"
    /// when it actually means "there is nothing here".
    pub degenerate: bool,
}

impl DistributionSummary {
    /// Whether the sample departs from normality at the given significance level.
    ///
    /// Convenience over `jb_p_value < alpha`, named so a report reads clearly. A degenerate
    /// constant sample is never "non-normal" — it has no shape to depart.
    #[must_use]
    pub fn rejects_normality(&self, alpha: f64) -> bool {
        !self.degenerate && self.jb_p_value < alpha
    }
}

/// Characterize a sample's distribution via its moments and the Jarque-Bera normality test.
///
/// Requires at least four values: the test is built on the third and fourth moments, and it is an
/// asymptotic (large-sample) test — below four points it is meaningless, and it errors rather than
/// return a number nobody should trust.
///
/// A constant sample (zero variance) has no defined skewness or kurtosis; rather than divide by
/// zero it returns those as `0.0`, `jb_p_value = 1.0`, and `degenerate = true`.
pub fn summarize(sample: &[f64]) -> Result<DistributionSummary, StatsError> {
    if sample.len() < 4 {
        return Err(StatsError::EmptySample {
            what: "distribution.summarize (needs at least 4 values)",
        });
    }
    if !sample.iter().all(|value| value.is_finite()) {
        return Err(StatsError::NonFinite {
            what: "distribution.summarize",
        });
    }

    let n = sample.len() as f64;
    let mean = sample.iter().sum::<f64>() / n;

    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &value in sample {
        let d = value - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;

    let std_dev = m2.sqrt();

    // A constant sample: no shape. Report zeros and flag it rather than divide by zero.
    if m2 == 0.0 {
        return Ok(DistributionSummary {
            n: sample.len(),
            mean,
            variance: 0.0,
            std_dev: 0.0,
            skewness: 0.0,
            excess_kurtosis: 0.0,
            jarque_bera: 0.0,
            jb_p_value: 1.0,
            degenerate: true,
        });
    }

    let skewness = m3 / m2.powf(1.5);
    let excess_kurtosis = m4 / (m2 * m2) - 3.0;
    let jarque_bera = (n / 6.0) * excess_kurtosis.mul_add(excess_kurtosis / 4.0, skewness * skewness);
    // Chi-square(2) survival function: exact, closed form, no special function.
    let jb_p_value = (-jarque_bera / 2.0).exp();

    Ok(DistributionSummary {
        n: sample.len(),
        mean,
        variance: m2,
        std_dev,
        skewness,
        excess_kurtosis,
        jarque_bera,
        jb_p_value,
        degenerate: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic normal draws for the large-sample normality checks.
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
        fn normal(&mut self) -> f64 {
            let u1 = self.unit();
            let u2 = self.unit();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    #[test]
    fn the_moments_match_a_hand_computed_example() {
        // [1,2,3,4,5]: mean 3, m2 = (4+1+0+1+4)/5 = 2, m3 = 0 (symmetric), m4 = (16+1+0+1+16)/5 =
        // 6.8. skewness = 0/2^1.5 = 0. excess_kurtosis = 6.8/4 - 3 = 1.7 - 3 = -1.3.
        let s = summarize(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!((s.mean - 3.0).abs() < 1e-12);
        assert!((s.variance - 2.0).abs() < 1e-12, "population variance is 2; got {}", s.variance);
        assert!(s.skewness.abs() < 1e-12, "a symmetric sample has zero skewness; got {}", s.skewness);
        assert!(
            (s.excess_kurtosis - (-1.3)).abs() < 1e-12,
            "excess kurtosis of 1..5 is -1.3; got {}",
            s.excess_kurtosis
        );
        assert!(!s.degenerate);
    }

    #[test]
    fn a_right_skewed_sample_has_positive_skewness() {
        // Doubling steps: a long right tail.
        let s = summarize(&[0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]).unwrap();
        assert!(s.skewness > 0.5, "a right-tailed sample must have positive skewness; got {}", s.skewness);
    }

    #[test]
    fn a_left_skewed_sample_has_negative_skewness() {
        let s = summarize(&[-32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 0.0]).unwrap();
        assert!(s.skewness < -0.5, "a left-tailed sample must have negative skewness; got {}", s.skewness);
    }

    #[test]
    fn the_jb_p_value_is_the_exact_chi_square_2_survival() {
        // The p-value must be exactly exp(-JB/2). Check the relationship holds on a real sample,
        // and the two endpoints: JB=0 -> p=1.
        let s = summarize(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(
            (s.jb_p_value - (-s.jarque_bera / 2.0).exp()).abs() < 1e-12,
            "the p-value must equal exp(-JB/2)"
        );
        assert!(s.jb_p_value > 0.0 && s.jb_p_value <= 1.0, "a p-value must be in (0, 1]");
    }

    #[test]
    fn a_normal_sample_is_not_flagged_as_non_normal() {
        // A large gaussian sample: skew ~0, excess kurtosis ~0, so JB is small and normality is not
        // rejected at alpha=0.05.
        let mut draws = Normal::new(2024);
        let sample: Vec<f64> = (0..3000).map(|_| draws.normal()).collect();
        let s = summarize(&sample).unwrap();
        println!(
            "normal sample: skew={:.3} exkurt={:.3} JB={:.2} p={:.3}",
            s.skewness, s.excess_kurtosis, s.jarque_bera, s.jb_p_value
        );
        assert!(s.skewness.abs() < 0.2, "a normal sample has near-zero skewness; got {}", s.skewness);
        assert!(
            s.excess_kurtosis.abs() < 0.3,
            "a normal sample has near-zero excess kurtosis; got {}",
            s.excess_kurtosis
        );
        assert!(
            !s.rejects_normality(0.05),
            "a genuinely normal sample was flagged non-normal (p={:.4})",
            s.jb_p_value
        );
    }

    #[test]
    fn a_uniform_sample_is_flagged_as_non_normal() {
        // A uniform distribution has excess kurtosis about -1.2 (light tails). Over a large sample
        // Jarque-Bera detects that departure and rejects normality.
        let mut draws = Normal::new(99);
        // Reuse the unit() uniform draw from the helper by pulling raw uniforms.
        let sample: Vec<f64> = (0..3000).map(|_| draws.unit()).collect();
        let s = summarize(&sample).unwrap();
        println!(
            "uniform sample: skew={:.3} exkurt={:.3} JB={:.2} p={:.4}",
            s.skewness, s.excess_kurtosis, s.jarque_bera, s.jb_p_value
        );
        assert!(
            s.excess_kurtosis < -0.5,
            "a uniform sample has strongly negative excess kurtosis; got {}",
            s.excess_kurtosis
        );
        assert!(
            s.rejects_normality(0.05),
            "a uniform sample should be flagged non-normal, but p={:.4}",
            s.jb_p_value
        );
    }

    #[test]
    fn a_constant_sample_is_degenerate_not_normal_looking() {
        // A constant sample has no shape. It must not report "skewness 0, excess_kurtosis 0" as if
        // that meant "looks normal" — it is flagged degenerate, and it never rejects normality.
        let s = summarize(&[5.0, 5.0, 5.0, 5.0, 5.0]).unwrap();
        assert!(s.degenerate, "a constant sample must be flagged degenerate");
        assert_eq!(s.variance, 0.0);
        assert_eq!(s.jb_p_value, 1.0);
        assert!(!s.rejects_normality(0.05), "a constant sample has no shape to reject");
    }

    #[test]
    fn too_few_values_error_rather_than_return_a_meaningless_number() {
        assert!(matches!(
            summarize(&[1.0, 2.0, 3.0]),
            Err(StatsError::EmptySample { .. })
        ));
        assert!(matches!(
            summarize(&[1.0, 2.0, f64::NAN, 4.0]),
            Err(StatsError::NonFinite { .. })
        ));
    }

    #[test]
    fn the_summary_is_deterministic() {
        let sample = [1.0, 3.0, 3.0, 7.0, 2.0, 9.0, 4.0, 1.0];
        assert_eq!(summarize(&sample), summarize(&sample));
    }
}
