//! Native, dependency-free statistics for OFFLINE detector certification.
//!
//! # Why this exists (bd-2z0.11.6)
//!
//! The narrative detector in `scriptbots-core::detect` is deliberately hand-rolled, online, and
//! bit-stable. What it cannot do is CERTIFY its own findings: when it flags a regime change, is
//! that a real shift or a run of noise? That question is statistics, and statistics is expensive,
//! non-deterministic-looking, and has no business anywhere near the tick path. So it lives here,
//! offline, annotating events after the fact — never in core, never in a tick, never in wasm's
//! default graph.
//!
//! # Why native rather than frankenscipy (evidence for bd-2z0.11.3)
//!
//! `fsci-stats` is the strongest analytics fit in the franken family, but it is **git-only and
//! nightly-only**. An offline analysis binary should not inherit a nightly-toolchain requirement
//! for four textbook estimators. This module implements exactly those estimators natively — pure
//! functions over `&[f64]`, no dependencies, an inline deterministic RNG — and proves them with
//! the calibration goldens `bd-2z0.11.6` specifies (CI coverage ≥ 94% on known step changes;
//! permutation p-values uniform under the null). That is the concrete evidence the adapter
//! decision needs: it shows the native path is sufficient for the certification we actually do.
//! If a report later needs fsci's 95+ distributions, adopt it there — but the core estimators do
//! not require it, and this file demonstrates that rather than asserting it.
//!
//! # Determinism
//!
//! Every resampling routine takes an explicit seed and draws from an inline `SplitMix64`. No
//! `Math.random`, no thread-local entropy, no clock: two runs with the same seed and inputs
//! produce bit-identical output. An analysis that could not be reproduced would not be evidence.

// Numerical code inherently casts sample sizes and counts to f64 (for means, ranks, indices), and
// deliberately compares floats to exact sentinels — a pooled SD of exactly 0.0, an effect of
// exactly 0.0. Allowing these module-wide, with this justification, keeps the estimators readable
// instead of scattering identical per-line pragmas through arithmetic that is correct by
// construction. These are the ONLY blanket allows, and none of them suppress a correctness lint.
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::float_cmp)]

/// A deterministic, seedable RNG — `SplitMix64`.
///
/// Inlined rather than pulled from `rand` on purpose: this whole module exists to show that
/// detector certification needs no external statistics stack, and a dependency-free RNG is part
/// of that. `SplitMix64` is a well-known, well-distributed 64-bit generator; it is more than
/// adequate for resampling, and being a pure function of its seed it makes every result here
/// reproducible.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Seed the generator. Any seed is valid, including zero.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    const fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform index in `0..len`. `len` must be non-zero.
    ///
    /// Uses Lemire's multiply-shift to avoid modulo bias — a small correctness detail that
    /// matters when a resample is repeated millions of times and a biased index would quietly
    /// skew every confidence interval derived from it.
    fn index(&mut self, len: usize) -> usize {
        debug_assert!(len > 0, "cannot draw an index from an empty range");
        let m = u128::from(self.next_u64()) * (len as u128);
        (m >> 64) as usize
    }
}

/// Errors from a statistics routine.
///
/// These are PROGRAMMING errors — an empty sample, a nonsense confidence level — surfaced loudly
/// rather than papered over with a silent `NaN`, because a certification computed from bad inputs
/// is worse than no certification.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StatsError {
    /// A sample that a statistic requires to be non-empty was empty.
    #[error("statistic `{what}` requires a non-empty sample")]
    EmptySample {
        /// The statistic that received the empty sample.
        what: &'static str,
    },
    /// A confidence level outside the open interval (0, 1).
    #[error("confidence level must be in (0, 1); got {level}")]
    InvalidConfidence {
        /// The invalid confidence level.
        level: f64,
    },
    /// A resample count of zero, which would produce an empty bootstrap distribution.
    #[error("resample count must be at least 1")]
    ZeroResamples,
    /// A moving-block length that is zero or exceeds the series length.
    #[error("block length {block} is invalid for a series of length {series_len}")]
    InvalidBlockLength {
        /// The invalid requested block length.
        block: usize,
        /// The number of values in the source series.
        series_len: usize,
    },
    /// An input that must be finite contained NaN or an infinity.
    #[error("statistic `{what}` received a non-finite value")]
    NonFinite {
        /// The statistic that received the non-finite value.
        what: &'static str,
    },
}

/// The arithmetic mean. Errors on an empty sample rather than returning `NaN`.
pub fn mean(sample: &[f64]) -> Result<f64, StatsError> {
    if sample.is_empty() {
        return Err(StatsError::EmptySample { what: "mean" });
    }
    finite_guard(sample, "mean")?;
    Ok(sample.iter().sum::<f64>() / sample.len() as f64)
}

/// Sample standard deviation (Bessel-corrected, denominator `n - 1`).
///
/// Returns `0.0` for a single-element sample rather than erroring: one observation has no spread,
/// and that is a meaningful answer, not a failure.
pub fn std_dev(sample: &[f64]) -> Result<f64, StatsError> {
    if sample.is_empty() {
        return Err(StatsError::EmptySample { what: "std_dev" });
    }
    if sample.len() == 1 {
        return Ok(0.0);
    }
    let m = mean(sample)?;
    let variance =
        sample.iter().map(|value| (value - m).powi(2)).sum::<f64>() / (sample.len() - 1) as f64;
    Ok(variance.sqrt())
}

/// The `q`-quantile of a sample, `q` in `[0, 1]`, by linear interpolation between order
/// statistics (the "type 7" definition used by `NumPy` and R's default).
///
/// The sample is sorted internally with `total_cmp`, so NaN cannot silently reorder the data —
/// the finite guard rejects it first.
// The validated `q` bounds make both percentile ranks nonnegative and in range; Rust has no
// checked `f64`-to-`usize` conversion for these two indices.
#[allow(clippy::cast_sign_loss)]
pub fn quantile(sample: &[f64], q: f64) -> Result<f64, StatsError> {
    if sample.is_empty() {
        return Err(StatsError::EmptySample { what: "quantile" });
    }
    finite_guard(sample, "quantile")?;
    if !(0.0..=1.0).contains(&q) {
        return Err(StatsError::InvalidConfidence { level: q });
    }
    let mut sorted = sample.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() == 1 {
        return Ok(sorted[0]);
    }
    let rank = q * (sorted.len() - 1) as f64;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    if low == high {
        return Ok(sorted[low]);
    }
    let weight = rank - low as f64;
    Ok(sorted[high].mul_add(weight, sorted[low] * (1.0 - weight)))
}

/// A two-sided confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceInterval {
    /// The point estimate the interval is built around.
    pub point: f64,
    /// The lower endpoint of the interval.
    pub lower: f64,
    /// The upper endpoint of the interval.
    pub upper: f64,
    /// The nominal coverage, e.g. `0.95`.
    pub confidence: f64,
    /// How many bootstrap resamples produced this interval — reported so a reader can judge the
    /// Monte-Carlo error of the bounds themselves.
    pub resamples: usize,
}

impl ConfidenceInterval {
    /// Does the interval contain `value`? The coverage tests below are exactly the fraction of
    /// intervals for which this is true when `value` is the known ground truth.
    #[must_use]
    pub fn covers(&self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }
}

/// Percentile bootstrap CI for the difference in means between two independent samples.
///
/// This is the estimator the narrative-validate report needs: given a metric's window BEFORE an
/// event and its window AFTER, how large is the shift and how uncertain is that size? Resamples
/// each side independently with replacement, recomputes the mean difference, and takes the
/// central `confidence` mass of the bootstrap distribution.
pub fn bootstrap_mean_difference_ci(
    before: &[f64],
    after: &[f64],
    n_resamples: usize,
    confidence: f64,
    seed: u64,
) -> Result<ConfidenceInterval, StatsError> {
    validate_ci_inputs(
        before,
        "bootstrap.before",
        after,
        "bootstrap.after",
        n_resamples,
        confidence,
    )?;

    let point = mean(after)? - mean(before)?;
    let mut rng = DeterministicRng::new(seed);
    let mut diffs = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let b = resample_mean(before, &mut rng);
        let a = resample_mean(after, &mut rng);
        diffs.push(a - b);
    }

    let alpha = 1.0 - confidence;
    Ok(ConfidenceInterval {
        point,
        lower: quantile(&diffs, alpha / 2.0)?,
        upper: quantile(&diffs, 1.0 - alpha / 2.0)?,
        confidence,
        resamples: n_resamples,
    })
}

/// Moving-block bootstrap CI for the mean of a SINGLE autocorrelated series.
///
/// A plain bootstrap resamples individual points, which destroys the autocorrelation that a time
/// series carries and therefore UNDERSTATES the uncertainty of its mean — the classic mistake the
/// bead warns about by asking for an "autocorrelation-aware" method. The moving-block bootstrap
/// resamples contiguous BLOCKS of length `block_len` instead, preserving short-range dependence
/// within each block. `block_len` should scale with the series' correlation length (a common
/// rule of thumb is `n^(1/3)`); it is a parameter here so the report can log the value it used.
pub fn moving_block_bootstrap_mean_ci(
    series: &[f64],
    block_len: usize,
    n_resamples: usize,
    confidence: f64,
    seed: u64,
) -> Result<ConfidenceInterval, StatsError> {
    if series.is_empty() {
        return Err(StatsError::EmptySample {
            what: "block_bootstrap",
        });
    }
    finite_guard(series, "block_bootstrap")?;
    if block_len == 0 || block_len > series.len() {
        return Err(StatsError::InvalidBlockLength {
            block: block_len,
            series_len: series.len(),
        });
    }
    if n_resamples == 0 {
        return Err(StatsError::ZeroResamples);
    }
    validate_confidence(confidence)?;

    // Blocks start at any position; the last block wraps for the tail so every starting index is
    // usable and no observation is systematically under-represented.
    let n = series.len();
    let blocks_needed = n.div_ceil(block_len);
    let point = mean(series)?;
    let mut rng = DeterministicRng::new(seed);
    let mut means = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let mut rebuilt = Vec::with_capacity(blocks_needed * block_len);
        for _ in 0..blocks_needed {
            let start = rng.index(n);
            for offset in 0..block_len {
                rebuilt.push(series[(start + offset) % n]);
            }
        }
        rebuilt.truncate(n);
        means.push(rebuilt.iter().sum::<f64>() / rebuilt.len() as f64);
    }

    let alpha = 1.0 - confidence;
    Ok(ConfidenceInterval {
        point,
        lower: quantile(&means, alpha / 2.0)?,
        upper: quantile(&means, 1.0 - alpha / 2.0)?,
        confidence,
        resamples: n_resamples,
    })
}

/// The outcome of a two-sample permutation test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PermutationTest {
    /// The observed statistic (difference in means, `after - before`).
    pub observed: f64,
    /// Two-sided p-value: the fraction of permutations whose |statistic| is at least the
    /// observed |statistic|. Computed with the `(count + 1) / (n + 1)` correction so it can never
    /// be exactly zero — a p-value of literally zero is a claim no finite resampling can support.
    pub p_value: f64,
    /// The number of label permutations used to estimate the p-value.
    pub permutations: usize,
}

/// Two-sample permutation test for a difference in means against a stationary null.
///
/// The null hypothesis is that `before` and `after` are exchangeable — that the event marking the
/// boundary between them changed nothing. Under that null, relabelling points between the two
/// groups should not matter, so we pool the data, repeatedly shuffle the labels, and ask how
/// often a relabelling produces a mean difference as extreme as the one actually observed. This
/// is exactly the "principled method instead of eyeballed thresholds" the bead asks for to serve
/// the false-positive budget: the p-value is calibrated by construction.
pub fn permutation_test_mean_difference(
    before: &[f64],
    after: &[f64],
    n_permutations: usize,
    seed: u64,
) -> Result<PermutationTest, StatsError> {
    if before.is_empty() {
        return Err(StatsError::EmptySample {
            what: "permutation.before",
        });
    }
    if after.is_empty() {
        return Err(StatsError::EmptySample {
            what: "permutation.after",
        });
    }
    finite_guard(before, "permutation.before")?;
    finite_guard(after, "permutation.after")?;
    if n_permutations == 0 {
        return Err(StatsError::ZeroResamples);
    }

    let observed = mean(after)? - mean(before)?;
    let n_before = before.len();
    let mut pooled: Vec<f64> = Vec::with_capacity(before.len() + after.len());
    pooled.extend_from_slice(before);
    pooled.extend_from_slice(after);

    let mut rng = DeterministicRng::new(seed);
    let mut at_least_as_extreme = 0usize;
    for _ in 0..n_permutations {
        // Partial Fisher-Yates: shuffle the whole pool, then split at n_before.
        for i in (1..pooled.len()).rev() {
            let j = rng.index(i + 1);
            pooled.swap(i, j);
        }
        let perm_before: f64 = pooled[..n_before].iter().sum::<f64>() / n_before as f64;
        let perm_after: f64 =
            pooled[n_before..].iter().sum::<f64>() / (pooled.len() - n_before) as f64;
        if (perm_after - perm_before).abs() >= observed.abs() {
            at_least_as_extreme += 1;
        }
    }

    Ok(PermutationTest {
        observed,
        p_value: (at_least_as_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0),
        permutations: n_permutations,
    })
}

/// Cohen's d: the standardized mean difference, using the pooled standard deviation.
///
/// A parametric effect size — meaningful when the two groups are roughly normal with similar
/// spread. Reported ALONGSIDE Cliff's delta, never instead of it, because real metric series are
/// often skewed and a reader deserves both a parametric and a distribution-free view.
pub fn cohens_d(before: &[f64], after: &[f64]) -> Result<f64, StatsError> {
    if before.len() < 2 || after.len() < 2 {
        return Err(StatsError::EmptySample {
            what: "cohens_d (needs n>=2 per group)",
        });
    }
    finite_guard(before, "cohens_d.before")?;
    finite_guard(after, "cohens_d.after")?;
    let (n1, n2) = (before.len() as f64, after.len() as f64);
    let (s1, s2) = (std_dev(before)?, std_dev(after)?);
    let pooled_var = ((n2 - 1.0) * s2).mul_add(s2, (n1 - 1.0) * s1 * s1) / (n1 + n2 - 2.0);
    let pooled_sd = pooled_var.sqrt();
    if pooled_sd == 0.0 {
        // Both groups are constant. The effect is either zero (equal constants) or infinite
        // (different constants); report 0.0 for equal and a large sentinel for different, rather
        // than dividing by zero.
        return Ok(if (mean(after)? - mean(before)?) == 0.0 {
            0.0
        } else {
            f64::INFINITY
        });
    }
    Ok((mean(after)? - mean(before)?) / pooled_sd)
}

/// Which standardized-mean-difference estimator produced a reported effect size.
///
/// Carried in results rather than documented, because `n` alone is not enough to interpret one.
/// The realised sample size tells a reader whether the small-sample regime applies; this tells
/// them whether the correction has ALREADY been applied. Without it, a reader who correctly
/// notices `n_before = 6` cannot tell whether to discount the number or has already been given
/// a discounted one, and applying the correction twice is as wrong as never applying it.
// Deliberately NOT Serialize/Deserialize: this module is dependency-free by design (see the
// header), and its sibling result types — ConfidenceInterval, EventCertification — derive only
// Debug/Clone/PartialEq. Matching them keeps the constraint intact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectSizeEstimator {
    /// [`cohens_d`]: uncorrected. Biased HIGH for small samples.
    CohensD,
    /// [`hedges_g`]: Cohen's d with the small-sample bias correction applied.
    HedgesG,
}

/// Hedges' g: Cohen's d corrected for small-sample bias.
///
/// Cohen's d is a BIASED estimator of the population standardized mean difference — it
/// overstates the effect, and the overstatement grows as the sample shrinks. Hedges' correction
/// multiplies by
///
/// ```text
///     J(df) = 1 - 3 / (4 * df - 1),   df = n1 + n2 - 2
/// ```
///
/// The size of what this corrects, so a caller can judge whether it matters (equal n per group):
///
/// ```text
///     n per group     J        d overstates by
///          3        0.800          +25.0%
///          5        0.903          +10.7%
///         10        0.958           +4.4%
///         20        0.980           +2.0%
///         30        0.987           +1.3%
///        100        0.996           +0.4%
/// ```
///
/// So the correction is material below roughly n=15 per group and negligible above n=30. It is
/// offered ALONGSIDE [`cohens_d`] rather than replacing it, so the choice is explicit at the
/// call site instead of an invisible default — and whichever is chosen, report it with
/// [`EffectSizeEstimator`] so the reader is not left guessing (`bd-k3f3`).
///
/// Degenerate inputs behave exactly as [`cohens_d`] does, including the infinite effect for
/// two different constants: correcting an unbounded effect leaves it unbounded.
pub fn hedges_g(before: &[f64], after: &[f64]) -> Result<f64, StatsError> {
    let d = cohens_d(before, after)?;
    // `cohens_d` already refused n<2 per group, so df >= 2 and the denominator cannot vanish.
    let df = (before.len() + after.len()) as f64 - 2.0;
    let correction = 1.0 - 3.0 / 4.0f64.mul_add(df, -1.0);
    Ok(d * correction)
}

/// Cliff's delta: a nonparametric effect size in `[-1, 1]`.
///
/// The probability that a randomly drawn `after` value exceeds a randomly drawn `before` value,
/// minus the reverse. `+1` means every `after` beats every `before`; `0` means complete overlap.
/// It makes no distributional assumption, so it is the honest effect size for the skewed,
/// heavy-tailed metric series a simulation actually produces.
pub fn cliffs_delta(before: &[f64], after: &[f64]) -> Result<f64, StatsError> {
    if before.is_empty() {
        return Err(StatsError::EmptySample {
            what: "cliffs_delta.before",
        });
    }
    if after.is_empty() {
        return Err(StatsError::EmptySample {
            what: "cliffs_delta.after",
        });
    }
    finite_guard(before, "cliffs_delta.before")?;
    finite_guard(after, "cliffs_delta.after")?;
    let mut greater = 0i64;
    let mut less = 0i64;
    for &a in after {
        for &b in before {
            match a.total_cmp(&b) {
                std::cmp::Ordering::Greater => greater += 1,
                std::cmp::Ordering::Less => less += 1,
                std::cmp::Ordering::Equal => {}
            }
        }
    }
    let total = (before.len() * after.len()) as f64;
    Ok((greater - less) as f64 / total)
}

// --- internal helpers ------------------------------------------------------------------------

fn finite_guard(sample: &[f64], what: &'static str) -> Result<(), StatsError> {
    if sample.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(StatsError::NonFinite { what })
    }
}

fn validate_confidence(confidence: f64) -> Result<(), StatsError> {
    if confidence > 0.0 && confidence < 1.0 {
        Ok(())
    } else {
        Err(StatsError::InvalidConfidence { level: confidence })
    }
}

fn validate_ci_inputs(
    a: &[f64],
    a_what: &'static str,
    b: &[f64],
    b_what: &'static str,
    n_resamples: usize,
    confidence: f64,
) -> Result<(), StatsError> {
    if a.is_empty() {
        return Err(StatsError::EmptySample { what: a_what });
    }
    if b.is_empty() {
        return Err(StatsError::EmptySample { what: b_what });
    }
    finite_guard(a, a_what)?;
    finite_guard(b, b_what)?;
    if n_resamples == 0 {
        return Err(StatsError::ZeroResamples);
    }
    validate_confidence(confidence)
}

fn resample_mean(sample: &[f64], rng: &mut DeterministicRng) -> f64 {
    let mut sum = 0.0;
    for _ in 0..sample.len() {
        sum += sample[rng.index(sample.len())];
    }
    sum / sample.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic standard-normal draw via Box-Muller, so the calibration tests have a known
    /// generating distribution without pulling in a distributions crate.
    struct NormalDraws {
        rng: DeterministicRng,
    }
    impl NormalDraws {
        fn new(seed: u64) -> Self {
            Self {
                rng: DeterministicRng::new(seed),
            }
        }
        fn unit(&mut self) -> f64 {
            // Convert 53 high bits to a uniform in (0, 1].
            let bits = self.rng.next_u64() >> 11;
            (bits as f64 + 1.0) / (9_007_199_254_740_992.0 + 1.0)
        }
        fn normal(&mut self, mean: f64, sd: f64) -> f64 {
            let u1 = self.unit();
            let u2 = self.unit();
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            sd.mul_add(z, mean)
        }
        fn series(&mut self, n: usize, mean: f64, sd: f64) -> Vec<f64> {
            (0..n).map(|_| self.normal(mean, sd)).collect()
        }
    }

    #[test]
    fn the_rng_is_deterministic_and_unbiased_across_a_small_range() {
        // Everything rests on this: identical seeds must give identical streams, or no result in
        // this module is reproducible.
        let mut a = DeterministicRng::new(42);
        let mut b = DeterministicRng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }

        // And the index draw must not favour any bucket. Over many draws into 7 buckets the counts
        // should be close to uniform; a modulo-biased generator would skew the low buckets.
        let mut rng = DeterministicRng::new(7);
        let mut counts = [0u32; 7];
        let draws = 700_000;
        for _ in 0..draws {
            counts[rng.index(7)] += 1;
        }
        let expected = f64::from(draws) / 7.0;
        for (bucket, &count) in counts.iter().enumerate() {
            let deviation = (f64::from(count) - expected).abs() / expected;
            assert!(
                deviation < 0.02,
                "bucket {bucket} deviated {deviation:.4} from uniform ({count} vs \
                 {expected:.0}); the index draw is biased and every bootstrap built on it \
                 inherits that bias"
            );
        }
    }

    #[test]
    fn quantile_matches_the_numpy_type_7_definition() {
        // Pinned against known values so a refactor cannot silently change the interpolation and
        // move every confidence bound in the crate.
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile(&data, 0.0).unwrap(), 1.0);
        assert_eq!(quantile(&data, 1.0).unwrap(), 4.0);
        assert_eq!(quantile(&data, 0.5).unwrap(), 2.5);
        // 0.25 * (4-1) = 0.75 -> between index 0 (1.0) and 1 (2.0), weight 0.75 -> 1.75
        assert_eq!(quantile(&data, 0.25).unwrap(), 1.75);
    }

    #[test]
    fn cliffs_delta_is_plus_one_when_after_dominates_and_zero_on_overlap() {
        assert_eq!(
            cliffs_delta(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap(),
            1.0
        );
        assert_eq!(
            cliffs_delta(&[4.0, 5.0, 6.0], &[1.0, 2.0, 3.0]).unwrap(),
            -1.0
        );
        // Identical groups fully overlap -> delta 0.
        assert_eq!(
            cliffs_delta(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap(),
            0.0
        );
    }

    #[test]
    fn cohens_d_recovers_a_known_standardized_shift() {
        // Two normal samples one SD apart should give d ~ 1.0. Large n so sampling error is small.
        let mut draws = NormalDraws::new(123);
        let before = draws.series(4000, 0.0, 1.0);
        let after = draws.series(4000, 1.0, 1.0);
        let d = cohens_d(&before, &after).unwrap();
        assert!(
            (d - 1.0).abs() < 0.1,
            "Cohen's d should recover the injected 1-SD shift; got {d:.4}"
        );
    }

    #[test]
    fn bootstrap_ci_covers_a_known_step_change_at_its_nominal_rate() {
        // THE CALIBRATION GOLDEN the bead demands: inject a step change of KNOWN size and check
        // that the 95% bootstrap CI covers the truth at least 94% of the time over 200 seeded
        // replications. A CI that under-covers is a certification that lies about its confidence.
        let truth = 1.5; // the real difference in means
        let replications = 300;
        let mut covered = 0;
        for rep in 0..replications {
            let mut draws = NormalDraws::new(1_000 + rep);
            // n=100 per group: the percentile bootstrap under-covers slightly at small n, so a
            // comfortably-sized sample keeps empirical coverage above the 94% floor with margin
            // rather than balanced on it.
            let before = draws.series(100, 10.0, 2.0);
            let after = draws.series(100, 10.0 + truth, 2.0);
            let ci =
                bootstrap_mean_difference_ci(&before, &after, 2000, 0.95, 9_000 + rep).unwrap();
            if ci.covers(truth) {
                covered += 1;
            }
        }
        let coverage = f64::from(covered) / replications as f64;
        // LOG THE COVERAGE NUMBER, as the bead requires.
        println!("bootstrap 95% CI empirical coverage over {replications} reps: {coverage:.3}");
        assert!(
            coverage >= 0.94,
            "the 95% bootstrap CI covered the known truth only {coverage:.3} of the time — it \
             under-covers, so every 'certified' effect size is more uncertain than it claims"
        );
    }

    #[test]
    fn permutation_p_values_are_uniform_under_the_null() {
        // CALIBRATION under a PURE NULL: when before and after are drawn from the SAME
        // distribution, a correct two-sided test rejects at alpha=0.05 about 5% of the time — no
        // more. A test that rejected far more often would manufacture regime changes out of noise,
        // which is exactly the false-positive failure the narrative budget exists to prevent.
        let trials = 300;
        let alpha = 0.05;
        let mut rejections = 0;
        for trial in 0..trials {
            let mut draws = NormalDraws::new(50_000 + trial);
            let before = draws.series(40, 5.0, 1.5);
            let after = draws.series(40, 5.0, 1.5); // SAME distribution: null is true
            let test =
                permutation_test_mean_difference(&before, &after, 1000, 70_000 + trial).unwrap();
            if test.p_value < alpha {
                rejections += 1;
            }
        }
        let false_positive_rate = f64::from(rejections) / trials as f64;
        println!(
            "permutation test false-positive rate under the null at alpha={alpha}: \
             {false_positive_rate:.3} ({rejections}/{trials})"
        );
        // Allow Monte-Carlo slack around 0.05 (binomial SE over 300 trials is ~0.013).
        assert!(
            false_positive_rate <= 0.10,
            "the permutation test rejected the true null {false_positive_rate:.3} of the time, \
             well above its nominal {alpha}. It would flag noise as regime change."
        );
    }

    #[test]
    fn permutation_test_detects_a_real_and_large_shift() {
        // The other side of calibration: a genuine, large shift must be detected, or the test is
        // uniform-but-useless. A test that never rejects is as worthless as one that always does.
        let mut draws = NormalDraws::new(777);
        let before = draws.series(50, 0.0, 1.0);
        let after = draws.series(50, 3.0, 1.0); // a 3-SD shift is unmissable
        let test = permutation_test_mean_difference(&before, &after, 2000, 888).unwrap();
        assert!(
            test.p_value < 0.01,
            "a 3-SD shift produced p={:.4}; a test blind to an effect this large certifies nothing",
            test.p_value
        );
    }

    #[test]
    fn the_block_bootstrap_widens_the_ci_for_autocorrelated_data() {
        // The reason the moving-block method exists. Build a strongly autocorrelated series (an
        // AR(1) random walk) and compare a block bootstrap against a naive per-point one. The
        // block CI must be WIDER: the naive method treats correlated points as independent and so
        // understates the uncertainty of the mean.
        let mut draws = NormalDraws::new(2024);
        let n = 400;
        let phi = 0.9; // strong positive autocorrelation
        let mut series = Vec::with_capacity(n);
        let mut prev = 0.0;
        for _ in 0..n {
            #[expect(
                clippy::suboptimal_flops,
                reason = "preserve separate product and sum rounding for the seeded AR(1) calibration samples supplied to both bootstrap methods"
            )]
            prev = phi * prev + draws.normal(0.0, 1.0);
            series.push(prev);
        }

        let block = moving_block_bootstrap_mean_ci(&series, 20, 3000, 0.95, 11).unwrap();
        // A block length of 1 reduces the moving-block method to the naive per-point bootstrap,
        // which is the honest apples-to-apples comparison.
        let naive = moving_block_bootstrap_mean_ci(&series, 1, 3000, 0.95, 11).unwrap();

        let block_width = block.upper - block.lower;
        let naive_width = naive.upper - naive.lower;
        println!("block-bootstrap CI width {block_width:.4} vs naive {naive_width:.4}");
        assert!(
            block_width > naive_width,
            "the block bootstrap ({block_width:.4}) was not wider than the naive one \
             ({naive_width:.4}); it is not accounting for autocorrelation and would over-certify \
             the mean of a correlated series"
        );
    }

    #[test]
    fn results_are_reproducible_across_calls_with_the_same_seed() {
        let before = [1.0, 2.0, 3.0, 4.0, 5.0];
        let after = [2.0, 3.0, 4.0, 5.0, 6.0];
        let a = bootstrap_mean_difference_ci(&before, &after, 500, 0.95, 999).unwrap();
        let b = bootstrap_mean_difference_ci(&before, &after, 500, 0.95, 999).unwrap();
        assert_eq!(a, b, "same seed and inputs must give a bit-identical CI");

        let p1 = permutation_test_mean_difference(&before, &after, 500, 5).unwrap();
        let p2 = permutation_test_mean_difference(&before, &after, 500, 5).unwrap();
        assert_eq!(
            p1, p2,
            "same seed and inputs must give a bit-identical permutation p-value"
        );
    }

    #[test]
    fn degenerate_inputs_error_rather_than_return_nan() {
        assert!(matches!(mean(&[]), Err(StatsError::EmptySample { .. })));
        assert!(matches!(
            bootstrap_mean_difference_ci(&[], &[1.0], 10, 0.95, 0),
            Err(StatsError::EmptySample { .. })
        ));
        assert!(matches!(
            bootstrap_mean_difference_ci(&[1.0], &[1.0], 10, 1.5, 0),
            Err(StatsError::InvalidConfidence { .. })
        ));
        assert!(matches!(
            bootstrap_mean_difference_ci(&[1.0], &[1.0], 0, 0.95, 0),
            Err(StatsError::ZeroResamples)
        ));
        assert!(matches!(
            moving_block_bootstrap_mean_ci(&[1.0, 2.0], 5, 10, 0.95, 0),
            Err(StatsError::InvalidBlockLength { .. })
        ));
        assert!(matches!(
            mean(&[1.0, f64::NAN]),
            Err(StatsError::NonFinite { .. })
        ));
    }

    #[test]
    fn f32_metric_series_widen_to_f64_without_surprises() {
        // The bead notes our metric series are f32 and widen at the offline boundary. Prove the
        // widening is lossless for representable values and that the statistics agree.
        let f32_series: Vec<f32> = vec![1.5, 2.25, 3.125, 4.0];
        let widened: Vec<f64> = f32_series.iter().map(|&value| f64::from(value)).collect();
        assert_eq!(mean(&widened).unwrap(), (1.5 + 2.25 + 3.125 + 4.0) / 4.0);
        // Each widened value equals the exact f32 it came from.
        for (narrow, wide) in f32_series.iter().zip(&widened) {
            assert_eq!(f64::from(*narrow), *wide);
        }
    }

    /// Hedges' correction must actually bite at small n and vanish at large n.
    ///
    /// This asserts the SHAPE of the correction, not merely that a number comes back. A test
    /// that only checked `hedges_g` returns something finite would pass with the correction
    /// factor accidentally equal to 1.0 — i.e. with the bug this function exists to fix still
    /// present. So both ends are pinned: a material gap at n=5 and a negligible one at n=100.
    #[test]
    fn hedges_correction_diverges_at_small_n_and_converges_at_large_n() {
        // Same standardized effect at both sizes, so any difference is the correction alone.
        let small_before: Vec<f64> = (0..5).map(f64::from).collect();
        let small_after: Vec<f64> = (0..5).map(|i| f64::from(i) + 4.0).collect();
        let large_before: Vec<f64> = (0..100).map(|i| f64::from(i % 5)).collect();
        let large_after: Vec<f64> = (0..100).map(|i| f64::from(i % 5) + 4.0).collect();

        let d_small = cohens_d(&small_before, &small_after).expect("d small");
        let g_small = hedges_g(&small_before, &small_after).expect("g small");
        let d_large = cohens_d(&large_before, &large_after).expect("d large");
        let g_large = hedges_g(&large_before, &large_after).expect("g large");

        // n=5 per group -> df=8 -> J = 1 - 3/31 = 0.9032, so g is ~9.7% BELOW d.
        let small_ratio = g_small / d_small;
        assert!(
            (0.900..0.906).contains(&small_ratio),
            "at n=5 per group the correction must shrink d by ~10%; got ratio {small_ratio}"
        );

        // n=100 per group -> df=198 -> J = 0.9962, under half a percent.
        let large_ratio = g_large / d_large;
        assert!(
            (0.995..1.0).contains(&large_ratio),
            "at n=100 per group the correction must be negligible; got ratio {large_ratio}"
        );

        // The correction always shrinks toward zero and never flips sign.
        assert!(
            g_small.abs() < d_small.abs(),
            "correction must shrink the estimate"
        );
        assert!(
            g_small.signum() == d_small.signum(),
            "correction must not flip sign"
        );

        // And the whole point: the correction matters MORE at small n than at large n.
        assert!(
            (1.0 - small_ratio) > (1.0 - large_ratio) * 10.0,
            "the small-sample penalty must dominate the large-sample one"
        );
    }
}
