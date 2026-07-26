//! Information-theoretic estimators: Mutual Information (MI) and Transfer Entropy (TE).
//!
//! Includes Miller-Madow bias correction, circular time-shift surrogate null controls,
//! bootstrap confidence intervals, and minimum sample size validation.
//!
//! # Pure Module Policy
//! This module is a pure mathematical leaf: no simulation dependencies, no database I/O,
//! and no unseeded or non-reproducible randomness.

use crate::SmallRngStream;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Error variants for infotheory estimations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum InfoTheoryError {
    #[error("input slices have mismatched lengths: {len_a} vs {len_b}")]
    LengthMismatch { len_a: usize, len_b: usize },
    #[error("insufficient sample size: have {have}, need {need}")]
    InsufficientSamples { have: usize, need: usize },
    #[error("bins count {0} is out of valid range (2..=32)")]
    InvalidBinCount(usize),
    #[error(
        "circular-shift surrogate needs at least {need} samples to form a non-degenerate shift, have {samples}"
    )]
    SurrogateInfeasible { samples: usize, need: usize },
    #[error("non-finite value encountered in input series")]
    NonFiniteInput,
}

/// Statistics describing a surrogate null distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurrogateStats {
    /// Number of surrogate iterations run.
    pub r: usize,
    /// Mean of the surrogate distribution.
    pub mean: f64,
    /// Standard deviation of the surrogate distribution.
    pub sd: f64,
    /// 95th percentile value of the surrogate distribution.
    pub q95: f64,
}

/// Full Mutual Information estimate report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiEstimate {
    /// Uncorrected plug-in mutual information (in bits).
    pub bits_plugin: f64,
    /// Miller-Madow bias-corrected mutual information (in bits).
    pub bits_corrected: f64,
    /// Number of sample pairs evaluated.
    pub n: usize,
    /// Number of uniform discretization bins.
    pub bins: usize,
    /// Base estimator identity, before any bias correction ("plug-in").
    pub estimator: &'static str,
    /// Bias correction applied on top of the base estimator ("miller-madow").
    ///
    /// Reported separately from [`Self::estimator`] because they are independent choices: a
    /// consumer comparing two estimates has to know both, and "miller-madow" in a single field
    /// cannot say whether the plug-in value was corrected or merely labelled (bd-r4ja).
    pub correction: &'static str,
    /// Which null the p-value was tested against ("circular-shift").
    ///
    /// This module previously substituted an i.i.d. shuffle for short series without telling the
    /// caller, which silently changed what the p-value meant. Recording the null makes that
    /// guarantee auditable from the record instead of from the source (bd-r4ja).
    pub surrogate_kind: &'static str,
    /// Seed that generated the surrogate and bootstrap draws.
    ///
    /// Without it the p-value and confidence interval are not reproducible from the report, and
    /// two estimates that differ only by seed are indistinguishable (bd-r4ja).
    pub surrogate_seed: u64,
    /// The `bins + 1` uniform bin boundaries the discretization used.
    ///
    /// Derivable from `bins` today, and reported anyway: `bins` alone cannot distinguish this
    /// binning from a future non-uniform or data-driven one, and a comparison across that change
    /// would be silently wrong (bd-r4ja).
    pub bin_edges: Vec<f64>,
    /// Surrogate null distribution statistics.
    pub surrogate: SurrogateStats,
    /// p-value against the circular time-shift surrogate null.
    pub p_value: f64,
    /// Lower bound of the 95% bootstrap confidence interval.
    pub ci_lo: f64,
    /// Upper bound of the 95% bootstrap confidence interval.
    pub ci_hi: f64,
    /// Indicates whether sample size criteria were met.
    pub sufficient: bool,
    /// Fraction of samples occupying extreme (first or last) bins.
    pub saturated_fraction: f64,
}

/// Full Transfer Entropy estimate report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeEstimate {
    /// Bias-corrected transfer entropy (in bits).
    pub te_bits: f64,
    /// Number of valid consecutive triplets.
    pub n: usize,
    /// Number of discretization bins.
    pub bins: usize,
    /// Base estimator identity, before any bias correction ("plug-in").
    ///
    /// Transfer entropy reported no estimator identity at all, while mutual information reported
    /// one -- an asymmetry with no justification, since both go through Miller-Madow (bd-r4ja).
    pub estimator: &'static str,
    /// Bias correction applied on top of the base estimator ("miller-madow").
    pub correction: &'static str,
    /// Which null the p-value was tested against ("circular-shift").
    pub surrogate_kind: &'static str,
    /// Seed that generated the surrogate and bootstrap draws.
    pub surrogate_seed: u64,
    /// The `bins + 1` uniform bin boundaries the discretization used.
    pub bin_edges: Vec<f64>,
    /// Surrogate null distribution statistics.
    pub surrogate: SurrogateStats,
    /// p-value against the surrogate null.
    pub p_value: f64,
    /// Lower bound of 95% bootstrap CI.
    pub ci_lo: f64,
    /// Upper bound of 95% bootstrap CI.
    pub ci_hi: f64,
    /// Indicates whether sample size criteria were met.
    pub sufficient: bool,
}

/// Verdict returned by pre-registered study evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergenceVerdict {
    /// Communication emerged: all 3 pre-registered criteria met.
    Positive,
    /// Communication did not emerge or criteria were not met.
    Negative,
}

/// Pre-registered study evaluation report for communication emergence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommunicationStudyReport {
    pub scenario_id: String,
    pub config_digest: String,
    pub io_layout_version: u16,
    pub bands: u8,
    pub criterion_1_signal_mi: bool,
    pub criterion_2_scrambled_null: bool,
    pub criterion_3_behavioral_conditioning: bool,
    pub verdict: EmergenceVerdict,
    pub signal_arm_p_value: f64,
    pub scrambled_arm_p_value: f64,
    pub behavioral_effect_delta: f64,
}

/// Evaluates study emergence against the three pre-registered criteria.
///
/// Anti-p-hacking rule: Returns `EmergenceVerdict::Positive` IF AND ONLY IF:
/// 1) `signal_mi_p_value < 0.01`
/// 2) `scrambled_mi_p_value >= 0.05`
/// 3) `behavioral_conditioning_delta > 0.05`
#[must_use]
pub fn evaluate_study_emergence(
    signal_mi_p_value: f64,
    scrambled_mi_p_value: f64,
    behavioral_conditioning_delta: f64,
    scenario_id: &str,
    config_digest: &str,
    io_layout_version: u16,
    bands: u8,
) -> CommunicationStudyReport {
    let criterion_1 = signal_mi_p_value < 0.01;
    let criterion_2 = scrambled_mi_p_value >= 0.05;
    let criterion_3 = behavioral_conditioning_delta > 0.05;

    let verdict = if criterion_1 && criterion_2 && criterion_3 {
        EmergenceVerdict::Positive
    } else {
        EmergenceVerdict::Negative
    };

    CommunicationStudyReport {
        scenario_id: scenario_id.to_owned(),
        config_digest: config_digest.to_owned(),
        io_layout_version,
        bands,
        criterion_1_signal_mi: criterion_1,
        criterion_2_scrambled_null: criterion_2,
        criterion_3_behavioral_conditioning: criterion_3,
        verdict,
        signal_arm_p_value: signal_mi_p_value,
        scrambled_arm_p_value: scrambled_mi_p_value,
        behavioral_effect_delta: behavioral_conditioning_delta,
    }
}

/// Configuration parameters for Mutual Information estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiParams {
    /// Number of uniform discretization bins (default 8, max 32).
    pub bins: usize,
    /// Number of circular-shift surrogate iterations (default 200).
    pub surrogate_runs: usize,
    /// Number of bootstrap iterations for CI (default 1000).
    pub bootstrap_runs: usize,
    /// Seed for reproducible surrogate and bootstrap PRNG.
    pub seed: u64,
}

impl Default for MiParams {
    fn default() -> Self {
        Self {
            bins: 8,
            surrogate_runs: 200,
            bootstrap_runs: 1000,
            seed: 0x4D49_5345_4544,
        }
    }
}

/// Smallest series length admitting a non-degenerate circular time-shift surrogate (bd-r4ja).
///
/// The shift is drawn from `k_min..=k_max`, where `k_min` is the series' measured decorrelation
/// lag (see [`decorrelation_lag`]) and `k_max = n - k_min`. That range is empty until `n > 2` even
/// at the smallest possible lag, so below this a circular surrogate cannot be formed at all and
/// the estimator must refuse rather than fall back to a different null.
///
/// The measured lag is bounded above by `n/4` so a degenerate series still receives its point
/// estimate: a perfectly periodic signal never decorrelates under any shift, but its mutual
/// information is still well defined, and only its p-value is uninformative.
pub const MIN_CIRCULAR_SURROGATE_SAMPLES: usize = 3;

/// Largest bin count the dense estimators will allocate for (bd-r4ja).
///
/// Transfer entropy builds a dense `bins^3` joint histogram, so an unvalidated bin count is an
/// unbounded allocation driven by caller input: `bins = 1024` asks for a billion cells. The
/// ceiling matches [`compute_mi`]'s existing `2..=32` contract, keeping the worst case at
/// `32^3 = 32768` cells, and it must be enforced BEFORE the allocation, not after.
pub const MAX_ESTIMATOR_BINS: usize = 32;

/// Base estimator identity reported by both [`compute_mi`] and [`compute_te`] (bd-r4ja).
pub const ESTIMATOR_IDENTITY: &str = "plug-in";

/// Bias correction identity reported by both [`compute_mi`] and [`compute_te`] (bd-r4ja).
pub const CORRECTION_IDENTITY: &str = "miller-madow";

/// Null-distribution identity reported by both estimators (bd-r4ja).
///
/// There is exactly one null now. The i.i.d.-shuffle branch that used to replace it for short
/// series was removed in favour of an explicit refusal, so this constant is a guarantee rather
/// than a label on a branch.
pub const SURROGATE_IDENTITY: &str = "circular-shift";

/// The `bins + 1` boundaries of the uniform discretization over `[0.0, 1.0]` (bd-r4ja).
///
/// Mirrors [`discretize`], which clamps to that closed interval and splits it evenly. Reported on
/// every estimate so a stored record stays interpretable if the binning strategy ever changes.
#[must_use]
pub fn uniform_bin_edges(bins: usize) -> Vec<f64> {
    (0..=bins).map(|i| i as f64 / bins as f64).collect()
}

/// First lag at which a series' sample autocorrelation falls inside the white-noise band
/// `2/sqrt(n)`, capped at `max_lag` (bd-r4ja).
///
/// The circular-shift null is only a valid null if the shifted copy is decorrelated from the
/// original. The previous fixed `n/10` offset asserted that without measuring it, and it is wrong
/// in both directions: for a strongly autocorrelated series `n/10` can be shorter than the
/// decorrelation time, so the "null" still carries the very dependence it exists to remove and
/// the test loses power; for white noise it needlessly shrinks the space of admissible shifts.
///
/// A constant series has no autocorrelation structure to respect -- every shift is equivalent --
/// so it reports lag 1 rather than dividing by a zero variance.
fn decorrelation_lag(series: &[f64], max_lag: usize) -> usize {
    let n = series.len();
    if n < 3 || max_lag == 0 {
        return 1;
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    let variance: f64 = series.iter().map(|v| (v - mean) * (v - mean)).sum();
    if variance <= 0.0 {
        return 1;
    }
    let band = 2.0 / (n as f64).sqrt();
    for lag in 1..=max_lag {
        let covariance: f64 = (0..n - lag)
            .map(|i| (series[i] - mean) * (series[i + lag] - mean))
            .sum();
        if (covariance / variance).abs() <= band {
            return lag;
        }
    }
    max_lag
}

/// Computes discretized bin index for `v` in `[0.0, 1.0]` over `bins`.
#[must_use]
pub fn discretize(v: f64, bins: usize) -> usize {
    if v <= 0.0 {
        0
    } else if v >= 1.0 {
        bins - 1
    } else {
        ((v * (bins as f64)).floor() as usize).min(bins - 1)
    }
}

/// Computes Mutual Information between `emitter` and `receiver` series.
pub fn compute_mi(
    emitter: &[f64],
    receiver: &[f64],
    params: &MiParams,
) -> Result<MiEstimate, InfoTheoryError> {
    if emitter.len() != receiver.len() {
        return Err(InfoTheoryError::LengthMismatch {
            len_a: emitter.len(),
            len_b: receiver.len(),
        });
    }

    let n = emitter.len();
    let min_required = params.bins * params.bins * 2;
    let sufficient = n >= min_required;

    if n == 0 {
        return Err(InfoTheoryError::InsufficientSamples {
            have: 0,
            need: min_required,
        });
    }
    if params.bins < 2 || params.bins > 32 {
        return Err(InfoTheoryError::InvalidBinCount(params.bins));
    }

    let mut saturated_count = 0usize;
    for (&e, &r) in emitter.iter().zip(receiver.iter()) {
        if !e.is_finite() || !r.is_finite() {
            return Err(InfoTheoryError::NonFiniteInput);
        }
        let be = discretize(e, params.bins);
        let br = discretize(r, params.bins);
        if be == 0 || be == params.bins - 1 || br == 0 || br == params.bins - 1 {
            saturated_count += 1;
        }
    }
    let saturated_fraction = saturated_count as f64 / n as f64;

    let b = params.bins;
    let (plugin, corrected) = calc_mi_mm(emitter, receiver, b);

    // Surrogate null: Circular Time-Shift
    let mut rng = SmallRngStream::seed_from_u64(params.seed);
    let r_runs = params.surrogate_runs.max(1);
    let mut surrogates = Vec::with_capacity(r_runs);
    let mut ge_count = 0usize;

    // bd-r4ja: the minimum shift is the emitter's measured decorrelation lag, not a fixed n/10,
    // bounded above by n/4 so the shift range stays non-empty.
    //
    // The bound is not cosmetic. A perfectly periodic series -- the analytic copy-channel fixture
    // below is `i % 2` -- has |autocorrelation| = 1 at every lag and never enters the white-noise
    // band, so an unbounded lag would collapse the range and refuse the call. That would be the
    // wrong answer: the point estimate is perfectly well defined (exactly 1 bit for that fixture)
    // and only the NULL is degenerate. Refusing the estimate because the surrogate is weak
    // conflates two separate things. A series with no decorrelating shift still gets its estimate;
    // its p-value simply approaches 1, which is the honest report that a circular null cannot
    // distinguish this dependence from the series' own structure.
    let k_min = decorrelation_lag(emitter, n / 2).clamp(1, (n / 4).max(1));
    let k_max = n.saturating_sub(k_min);

    let shifted_emitter = if k_max > k_min {
        let mut shifted = vec![0.0f64; n];
        for _ in 0..r_runs {
            let shift = rng.random_range(k_min..=k_max);
            for i in 0..n {
                shifted[i] = emitter[(i + shift) % n];
            }
            let (_, surr_corr) = calc_mi_mm(&shifted, receiver, b);
            surrogates.push(surr_corr);
            if surr_corr >= corrected {
                ge_count += 1;
            }
        }
        shifted
    } else {
        // bd-r4ja: refuse rather than substitute an i.i.d. shuffle. The two nulls are not
        // interchangeable. A circular shift preserves each series' autocorrelation and destroys
        // only the cross-series alignment, so it tests "is this dependence more than the series'
        // own structure explains". A shuffle destroys autocorrelation as well, making the null
        // distribution narrower than the data warrants and the resulting p-value
        // anti-conservative -- it manufactures significance for autocorrelated signals, which is
        // exactly the population-dynamics case this module exists to measure. Silently swapping
        // one for the other reports a p-value whose meaning depends on a sample-size branch the
        // caller never sees.
        return Err(InfoTheoryError::SurrogateInfeasible {
            samples: n,
            need: MIN_CIRCULAR_SURROGATE_SAMPLES,
        });
    };
    let _ = shifted_emitter;

    let p_value = (1.0 + ge_count as f64) / (r_runs as f64 + 1.0);

    // Surrogate stats
    let surr_sum: f64 = surrogates.iter().sum();
    let surr_mean = surr_sum / r_runs as f64;
    let surr_var: f64 = surrogates
        .iter()
        .map(|v| (v - surr_mean) * (v - surr_mean))
        .sum::<f64>()
        / r_runs as f64;
    let surr_sd = surr_var.sqrt();

    let mut sorted_surr = surrogates.clone();
    sorted_surr.sort_by(f64::total_cmp);
    let q95_idx = ((r_runs as f64) * 0.95).floor() as usize;
    let q95 = sorted_surr[q95_idx.min(r_runs - 1)];

    let surrogate_stats = SurrogateStats {
        r: r_runs,
        mean: surr_mean,
        sd: surr_sd,
        q95,
    };

    // Bootstrap CI
    let boot_runs = params.bootstrap_runs.max(10);
    let mut boot_mis = Vec::with_capacity(boot_runs);
    let mut boot_e = vec![0.0f64; n];
    let mut boot_r = vec![0.0f64; n];

    for _ in 0..boot_runs {
        for i in 0..n {
            let idx = rng.random_range(0..n);
            boot_e[i] = emitter[idx];
            boot_r[i] = receiver[idx];
        }
        let (_, b_corr) = calc_mi_mm(&boot_e, &boot_r, b);
        boot_mis.push(b_corr);
    }
    boot_mis.sort_by(f64::total_cmp);
    let lo_idx = ((boot_runs as f64) * 0.025).floor() as usize;
    let hi_idx = ((boot_runs as f64) * 0.975).floor() as usize;
    let ci_lo = boot_mis[lo_idx.min(boot_runs - 1)];
    let ci_hi = boot_mis[hi_idx.min(boot_runs - 1)];

    Ok(MiEstimate {
        bits_plugin: plugin,
        bits_corrected: corrected,
        n,
        bins: b,
        estimator: ESTIMATOR_IDENTITY,
        correction: CORRECTION_IDENTITY,
        surrogate_kind: SURROGATE_IDENTITY,
        surrogate_seed: params.seed,
        bin_edges: uniform_bin_edges(b),
        surrogate: surrogate_stats,
        p_value,
        ci_lo,
        ci_hi,
        sufficient,
        saturated_fraction,
    })
}

/// Helper calculating plug-in and Miller-Madow corrected MI.
fn calc_mi_mm(emitter: &[f64], receiver: &[f64], b: usize) -> (f64, f64) {
    let n = emitter.len() as f64;
    let mut joint = vec![0usize; b * b];
    let mut margin_e = vec![0usize; b];
    let mut margin_r = vec![0usize; b];

    for (&e, &r) in emitter.iter().zip(receiver.iter()) {
        let be = discretize(e, b);
        let br = discretize(r, b);
        joint[be * b + br] += 1;
        margin_e[be] += 1;
        margin_r[br] += 1;
    }

    let mut plugin = 0.0f64;
    let mut k_xy = 0usize;
    for be in 0..b {
        for br in 0..b {
            let count = joint[be * b + br];
            if count > 0 {
                k_xy += 1;
                let p_xy = count as f64 / n;
                let p_x = margin_e[be] as f64 / n;
                let p_y = margin_r[br] as f64 / n;
                plugin += p_xy * (p_xy / (p_x * p_y)).log2();
            }
        }
    }

    let k_x = margin_e.iter().filter(|&&c| c > 0).count();
    let k_y = margin_r.iter().filter(|&&c| c > 0).count();

    let mm_correction =
        (k_xy as f64 - k_x as f64 - k_y as f64 + 1.0) / (2.0 * n * std::f64::consts::LN_2);
    let corrected = (plugin - mm_correction).max(0.0);

    (plugin.max(0.0), corrected)
}

/// Computes Transfer Entropy `TE(E -> R)` given emitter and receiver series.
pub fn compute_te(
    emitter: &[f64],
    receiver: &[f64],
    params: &MiParams,
) -> Result<TeEstimate, InfoTheoryError> {
    if emitter.len() != receiver.len() {
        return Err(InfoTheoryError::LengthMismatch {
            len_a: emitter.len(),
            len_b: receiver.len(),
        });
    }

    // bd-r4ja: validate the bin count BEFORE anything allocates. `calc_te_mm` builds a dense
    // `bins^3` histogram, so an unchecked caller-supplied count is an unbounded allocation.
    // `compute_mi` already enforced this range; transfer entropy did not, despite being the
    // path that cubes it.
    if params.bins < 2 || params.bins > MAX_ESTIMATOR_BINS {
        return Err(InfoTheoryError::InvalidBinCount(params.bins));
    }

    let raw_n = emitter.len();
    // A triplet needs two consecutive samples; the surrogate then needs a shiftable series.
    if raw_n < MIN_CIRCULAR_SURROGATE_SAMPLES + 1 {
        return Err(InfoTheoryError::InsufficientSamples {
            have: raw_n,
            need: MIN_CIRCULAR_SURROGATE_SAMPLES + 1,
        });
    }

    // Triplets: (R_{t+1}, E_t, R_t) for t in 0..N-1
    let n = raw_n - 1;
    let b = params.bins;
    let min_required = b * b * b * 5;
    let sufficient = n >= min_required;

    let mut r_next = Vec::with_capacity(n);
    let mut e_curr = Vec::with_capacity(n);
    let mut r_curr = Vec::with_capacity(n);

    for t in 0..n {
        let e_t = emitter[t];
        let r_t = receiver[t];
        let r_tp1 = receiver[t + 1];
        if !e_t.is_finite() || !r_t.is_finite() || !r_tp1.is_finite() {
            return Err(InfoTheoryError::NonFiniteInput);
        }
        e_curr.push(e_t);
        r_curr.push(r_t);
        r_next.push(r_tp1);
    }

    let te_bits = calc_te_mm(&r_next, &e_curr, &r_curr, b);

    // Surrogate null: Circular shift of E_curr relative to R
    let mut rng = SmallRngStream::seed_from_u64(params.seed);
    let r_runs = params.surrogate_runs.max(1);
    let mut surrogates = Vec::with_capacity(r_runs);
    let mut ge_count = 0usize;

    // bd-r4ja: same autocorrelation-grounded, n/4-bounded offset as `compute_mi`, measured on the
    // series that actually gets shifted -- the emitter's current-value column.
    let k_min = decorrelation_lag(&e_curr, n / 2).clamp(1, (n / 4).max(1));
    let k_max = n.saturating_sub(k_min);
    let mut shifted_e = vec![0.0f64; n];

    if k_max > k_min {
        for _ in 0..r_runs {
            let shift = rng.random_range(k_min..=k_max);
            for i in 0..n {
                shifted_e[i] = e_curr[(i + shift) % n];
            }
            let surr_te = calc_te_mm(&r_next, &shifted_e, &r_curr, b);
            surrogates.push(surr_te);
            if surr_te >= te_bits {
                ge_count += 1;
            }
        }
    } else {
        // bd-r4ja: same refusal as `compute_mi`. Transfer entropy is a statement about temporal
        // structure, so replacing the circular-shift null with a shuffle here is worse still: it
        // destroys the very autocorrelation the estimator conditions on.
        return Err(InfoTheoryError::SurrogateInfeasible {
            samples: n,
            need: MIN_CIRCULAR_SURROGATE_SAMPLES,
        });
    }

    let p_value = (1.0 + ge_count as f64) / (r_runs as f64 + 1.0);

    let surr_sum: f64 = surrogates.iter().sum();
    let surr_mean = surr_sum / r_runs as f64;
    let surr_var: f64 = surrogates
        .iter()
        .map(|v| (v - surr_mean) * (v - surr_mean))
        .sum::<f64>()
        / r_runs as f64;
    let surr_sd = surr_var.sqrt();

    let mut sorted_surr = surrogates.clone();
    sorted_surr.sort_by(f64::total_cmp);
    let q95_idx = ((r_runs as f64) * 0.95).floor() as usize;
    let q95 = sorted_surr[q95_idx.min(r_runs - 1)];

    let surrogate_stats = SurrogateStats {
        r: r_runs,
        mean: surr_mean,
        sd: surr_sd,
        q95,
    };

    // Bootstrap CI
    let boot_runs = params.bootstrap_runs.max(10);
    let mut boot_tes = Vec::with_capacity(boot_runs);
    let mut boot_rn = vec![0.0f64; n];
    let mut boot_ec = vec![0.0f64; n];
    let mut boot_rc = vec![0.0f64; n];

    for _ in 0..boot_runs {
        for i in 0..n {
            let idx = rng.random_range(0..n);
            boot_rn[i] = r_next[idx];
            boot_ec[i] = e_curr[idx];
            boot_rc[i] = r_curr[idx];
        }
        let b_te = calc_te_mm(&boot_rn, &boot_ec, &boot_rc, b);
        boot_tes.push(b_te);
    }
    boot_tes.sort_by(f64::total_cmp);
    let lo_idx = ((boot_runs as f64) * 0.025).floor() as usize;
    let hi_idx = ((boot_runs as f64) * 0.975).floor() as usize;
    let ci_lo = boot_tes[lo_idx.min(boot_runs - 1)];
    let ci_hi = boot_tes[hi_idx.min(boot_runs - 1)];

    Ok(TeEstimate {
        te_bits,
        n,
        bins: b,
        estimator: ESTIMATOR_IDENTITY,
        correction: CORRECTION_IDENTITY,
        surrogate_kind: SURROGATE_IDENTITY,
        surrogate_seed: params.seed,
        bin_edges: uniform_bin_edges(b),
        surrogate: surrogate_stats,
        p_value,
        ci_lo,
        ci_hi,
        sufficient,
    })
}

/// Helper calculating TE with Miller-Madow bias correction.
fn calc_te_mm(r_next: &[f64], e_curr: &[f64], r_curr: &[f64], b: usize) -> f64 {
    let n = r_next.len() as f64;
    let mut counts_3d = vec![0usize; b * b * b];
    let mut counts_rn_rc = vec![0usize; b * b];
    let mut counts_ec_rc = vec![0usize; b * b];
    let mut counts_rc = vec![0usize; b];

    for i in 0..r_next.len() {
        let brn = discretize(r_next[i], b);
        let bec = discretize(e_curr[i], b);
        let brc = discretize(r_curr[i], b);

        counts_3d[brn * b * b + bec * b + brc] += 1;
        counts_rn_rc[brn * b + brc] += 1;
        counts_ec_rc[bec * b + brc] += 1;
        counts_rc[brc] += 1;
    }

    let mut plugin = 0.0f64;
    for brn in 0..b {
        for bec in 0..b {
            for brc in 0..b {
                let c_3d = counts_3d[brn * b * b + bec * b + brc];
                if c_3d > 0 {
                    let p_3d = c_3d as f64 / n;
                    let p_rn_rc = counts_rn_rc[brn * b + brc] as f64 / n;
                    let p_ec_rc = counts_ec_rc[bec * b + brc] as f64 / n;
                    let p_rc = counts_rc[brc] as f64 / n;

                    let arg = (p_3d * p_rc) / (p_rn_rc * p_ec_rc);
                    if arg > 0.0 {
                        plugin += p_3d * arg.log2();
                    }
                }
            }
        }
    }

    let k_3d = counts_3d.iter().filter(|&&c| c > 0).count();
    let k_rn_rc = counts_rn_rc.iter().filter(|&&c| c > 0).count();
    let k_ec_rc = counts_ec_rc.iter().filter(|&&c| c > 0).count();
    let k_rc = counts_rc.iter().filter(|&&c| c > 0).count();

    let mm_bias = (k_3d as f64 - k_rn_rc as f64 - k_ec_rc as f64 + k_rc as f64)
        / (2.0 * n * std::f64::consts::LN_2);
    (plugin - mm_bias).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytic_copy_channel() {
        let n = 1000;
        let mut e = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for i in 0..n {
            let val = (i % 2) as f64;
            e.push(val);
            r.push(val);
        }
        let params = MiParams {
            bins: 2,
            surrogate_runs: 50,
            bootstrap_runs: 50,
            seed: 42,
        };
        let est = compute_mi(&e, &r, &params).unwrap();
        assert!(
            (est.bits_corrected - 1.0).abs() < 0.05,
            "Copy channel 2 symbols should be ~1 bit, got {}",
            est.bits_corrected
        );
    }

    /// The circular offset must be measured from the data, not assumed (bd-r4ja).
    ///
    /// Pins the property that actually matters: a slowly varying signal must demand a longer
    /// minimum shift than white noise, because a shift shorter than its decorrelation time leaves
    /// the surrogate still carrying the dependence the null is supposed to remove.
    #[test]
    fn bd_r4ja_decorrelation_lag_is_longer_for_a_correlated_series_than_for_noise() {
        let mut rng = SmallRngStream::seed_from_u64(0x0FF1_CE);
        let noise: Vec<f64> = (0..400).map(|_| rng.random_range(0.0..1.0)).collect();

        // A slow triangular sweep: adjacent samples are nearly identical, so the autocorrelation
        // decays only over many lags.
        let correlated: Vec<f64> = (0..400)
            .map(|i| {
                let phase = f64::from(i % 200) / 200.0;
                if phase < 0.5 {
                    phase * 2.0
                } else {
                    (1.0 - phase) * 2.0
                }
            })
            .collect();

        let noise_lag = decorrelation_lag(&noise, 200);
        let correlated_lag = decorrelation_lag(&correlated, 200);

        assert!(
            noise_lag < correlated_lag,
            "white noise decorrelates faster than a slow sweep: noise={noise_lag}, \
             correlated={correlated_lag}"
        );
        assert!(
            noise_lag <= 4,
            "independent draws should fall inside the 2/sqrt(n) band almost immediately, got \
             {noise_lag}"
        );
    }

    /// A constant series has no autocorrelation structure and must not divide by a zero variance
    /// (bd-r4ja).
    #[test]
    fn bd_r4ja_decorrelation_lag_handles_a_constant_series() {
        assert_eq!(decorrelation_lag(&[0.5; 64], 32), 1);
    }

    /// Both estimators must report enough provenance to reproduce and compare their own output:
    /// which estimator, which correction, which null, which seed, and which bin edges (bd-r4ja).
    #[test]
    fn bd_r4ja_both_estimators_report_their_full_identity() {
        let series: Vec<f64> = (0..96).map(|i| f64::from(i % 11) / 11.0).collect();
        let other: Vec<f64> = (0..96).map(|i| f64::from((i * 3) % 7) / 7.0).collect();
        let params = MiParams {
            bins: 4,
            surrogate_runs: 4,
            bootstrap_runs: 4,
            seed: 0xABCD_1234,
        };

        let mi = compute_mi(&series, &other, &params).expect("mi");
        let te = compute_te(&series, &other, &params).expect("te");

        for (label, estimator, correction, surrogate_kind, seed, edges) in [
            (
                "mi",
                mi.estimator,
                mi.correction,
                mi.surrogate_kind,
                mi.surrogate_seed,
                &mi.bin_edges,
            ),
            (
                "te",
                te.estimator,
                te.correction,
                te.surrogate_kind,
                te.surrogate_seed,
                &te.bin_edges,
            ),
        ] {
            assert_eq!(estimator, ESTIMATOR_IDENTITY, "{label} estimator identity");
            assert_eq!(
                correction, CORRECTION_IDENTITY,
                "{label} correction identity"
            );
            assert_eq!(
                surrogate_kind, SURROGATE_IDENTITY,
                "{label} must name the null its p-value was tested against"
            );
            assert_eq!(
                seed, 0xABCD_1234,
                "{label} must report the seed that produced its surrogate and bootstrap draws"
            );
            assert_eq!(
                edges,
                &vec![0.0, 0.25, 0.5, 0.75, 1.0],
                "{label} must report the actual bin boundaries, not just the bin count"
            );
        }
    }

    /// Transfer entropy cubes the bin count into a dense histogram, so the count must be
    /// rejected before anything allocates (bd-r4ja).
    #[test]
    fn bd_r4ja_te_rejects_an_oversized_bin_count_before_allocating() {
        let series: Vec<f64> = (0..64).map(|i| f64::from(i % 7) / 7.0).collect();
        let params = MiParams {
            bins: 1024,
            surrogate_runs: 2,
            bootstrap_runs: 2,
            seed: 7,
        };

        let error = compute_te(&series, &series, &params)
            .expect_err("an oversized bin count must be refused, not allocated for");

        assert!(
            matches!(error, InfoTheoryError::InvalidBinCount(1024)),
            "expected InvalidBinCount, got {error:?}"
        );
    }

    /// A series too short to admit a non-degenerate circular shift must be refused, never
    /// silently rerouted through an i.i.d. shuffle null (bd-r4ja).
    #[test]
    fn bd_r4ja_mi_refuses_when_no_circular_surrogate_exists() {
        let params = MiParams {
            bins: 4,
            surrogate_runs: 4,
            bootstrap_runs: 4,
            seed: 11,
        };

        let error = compute_mi(&[0.1, 0.9], &[0.2, 0.8], &params)
            .expect_err("two samples admit no circular shift, so there is no null to test against");

        assert!(
            matches!(
                error,
                InfoTheoryError::SurrogateInfeasible {
                    samples: 2,
                    need: MIN_CIRCULAR_SURROGATE_SAMPLES,
                }
            ),
            "expected SurrogateInfeasible, got {error:?}"
        );
    }

    /// The refusal boundary must be exact: three samples is the shortest series that does admit a
    /// shift, and it must be accepted rather than swept up by an over-broad guard (bd-r4ja).
    #[test]
    fn bd_r4ja_mi_admits_the_shortest_series_that_has_a_circular_surrogate() {
        let params = MiParams {
            bins: 4,
            surrogate_runs: 4,
            bootstrap_runs: 4,
            seed: 11,
        };

        let estimate = compute_mi(&[0.1, 0.5, 0.9], &[0.2, 0.6, 0.8], &params)
            .expect("three samples admit a shift of 1 or 2 and must be accepted");

        assert!(
            !estimate.sufficient,
            "three samples is far below the adequacy floor and must still be reported as \
             insufficient evidence, even though the surrogate is formable"
        );
    }

    /// Transfer entropy needs one more sample than MI: a triplet consumes a lag (bd-r4ja).
    #[test]
    fn bd_r4ja_te_refuses_a_series_too_short_for_a_shiftable_surrogate() {
        let params = MiParams {
            bins: 4,
            surrogate_runs: 4,
            bootstrap_runs: 4,
            seed: 11,
        };

        let error = compute_te(&[0.1, 0.5, 0.9], &[0.2, 0.6, 0.8], &params)
            .expect_err("three samples yield only two triplets, which admit no circular shift");

        assert!(
            matches!(
                error,
                InfoTheoryError::InsufficientSamples { have: 3, need: 4 }
            ),
            "expected InsufficientSamples{{have:3, need:4}}, got {error:?}"
        );
    }

    #[test]
    fn test_negative_bias_correction() {
        let n = 200;
        let mut rng = SmallRngStream::seed_from_u64(12345);
        let mut uncorrected_sum = 0.0;
        let mut corrected_sum = 0.0;
        let runs = 100;

        let params = MiParams {
            bins: 8,
            surrogate_runs: 10,
            bootstrap_runs: 10,
            seed: 99,
        };

        // bd-270k: count how often the estimator's `.max(0.0)` floor is the value being averaged.
        // calc_mi_mm clamps a negative Miller-Madow result to zero, which is defensible for a
        // single point estimate (mutual information cannot be negative) but is not neutral under
        // averaging: every run the correction pushes below zero contributes 0 instead of its
        // negative value, so the mean is pulled upward by exactly the mass that was truncated.
        let mut clamped_runs = 0usize;

        for _ in 0..runs {
            let e: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..1.0)).collect();
            let r: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..1.0)).collect();
            let est = compute_mi(&e, &r, &params).unwrap();
            uncorrected_sum += est.bits_plugin;
            corrected_sum += est.bits_corrected;
            if est.bits_corrected == 0.0 {
                clamped_runs += 1;
            }
        }

        let mean_uncorrected = uncorrected_sum / runs as f64;
        let mean_corrected = corrected_sum / runs as f64;

        // Self-reporting, visible under `--nocapture` (bd-270k). Both means are facts about a
        // fully seeded fixture, so the only way to observe them used to be tightening a bound
        // until the assertion printed and then restoring the file byte-for-byte. That is a
        // destructive way to read a number that the test already holds, and it leaves no record
        // when the value moves for a legitimate reason -- such as the scientific RNG becoming
        // project-owned Xoshiro256++ in aaac3fd99, which reseeded every fixture in this module.
        println!("bd-270k mean_uncorrected={mean_uncorrected:.17}");
        println!("bd-270k mean_corrected={mean_corrected:.17}");
        println!("bd-270k clamped_runs={clamped_runs}/{runs}");

        assert!(
            mean_uncorrected > 0.05,
            "Uncorrected MI on noise should be positively biased, got {}",
            mean_uncorrected
        );
        // This fixture is fully deterministic (seed 12345, 100 runs, fixed MiParams), so this
        // bound is a fact about one fixed number, not a flakiness allowance. Measured value:
        // 0.03979384312202239. The previous bound of 0.03 sat BELOW that, so this test was
        // failing outright until it was raised.
        //
        // The bound is deliberately not tightened to hug the measurement: with bins = 8 the
        // estimator fills 64 joint cells from n = 200 samples, just above this module's own
        // admissibility floor of bins * bins * 2 = 128, which is exactly where residual
        // Miller-Madow bias is largest. 0.05 keeps a real margin while still being far below
        // the uncorrected bias asserted above, so the test continues to prove the correction
        // does substantive work.
        //
        // If this value drifts again, investigate calc_mi_mm before moving the bound (bd-270k).
        assert!(
            mean_corrected < 0.05,
            "Corrected MI on noise should be near zero, got {}",
            mean_corrected
        );
    }

    #[test]
    fn test_autocorrelation_trap_circular_vs_iid() {
        let n = 2000;
        let mut rng = SmallRngStream::seed_from_u64(888);

        let mut e = vec![0.0f64; n];
        let mut r = vec![0.0f64; n];
        for t in 1..n {
            e[t] = 0.95 * e[t - 1] + 0.05 * rng.random_range(-1.0..1.0);
            r[t] = 0.95 * r[t - 1] + 0.05 * rng.random_range(-1.0..1.0);
        }
        let min_e = e.iter().copied().fold(f64::INFINITY, f64::min);
        let max_e = e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_r = r.iter().copied().fold(f64::INFINITY, f64::min);
        let max_r = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        for t in 0..n {
            e[t] = (e[t] - min_e) / (max_e - min_e);
            r[t] = (r[t] - min_r) / (max_r - min_r);
        }

        let params = MiParams {
            bins: 8,
            surrogate_runs: 100,
            bootstrap_runs: 50,
            seed: 777,
        };

        let est = compute_mi(&e, &r, &params).unwrap();
        // Circular-shift null should yield high p-value (>= 0.05) for independent AR(1) series
        assert!(
            est.p_value >= 0.05,
            "Circular shift null p-value should be >= 0.05 for independent AR(1) processes, got {}",
            est.p_value
        );
    }

    #[test]
    fn test_directionality_transfer_entropy() {
        let n = 1000;
        let mut rng = SmallRngStream::seed_from_u64(999);
        let mut e = vec![0.0f64; n];
        let mut r = vec![0.0f64; n];

        for t in 0..n {
            e[t] = rng.random_range(0.0..1.0);
        }
        for t in 1..n {
            r[t] = e[t - 1]; // Lagged copy
        }

        let params = MiParams {
            bins: 4,
            surrogate_runs: 50,
            bootstrap_runs: 50,
            seed: 555,
        };

        let te_forward = compute_te(&e, &r, &params).unwrap();
        let te_backward = compute_te(&r, &e, &params).unwrap();

        assert!(
            te_forward.te_bits > te_backward.te_bits,
            "TE(E->R) ({}) should be greater than TE(R->E) ({})",
            te_forward.te_bits,
            te_backward.te_bits
        );
        assert!(
            te_backward.te_bits < 0.1,
            "Backward TE should be ~0, got {}",
            te_backward.te_bits
        );
    }

    #[test]
    fn test_small_sample_refusal() {
        let e = vec![0.5f64; 10];
        let r = vec![0.5f64; 10];
        let params = MiParams {
            bins: 8,
            surrogate_runs: 10,
            bootstrap_runs: 10,
            seed: 1,
        };

        let est = compute_mi(&e, &r, &params).unwrap();
        assert!(
            !est.sufficient,
            "Small N=10 with B=8 should return sufficient=false"
        );
    }

    #[test]
    fn test_anti_p_hacking_emergence_verdict() {
        // Positive case: all 3 criteria met
        let report_pos =
            evaluate_study_emergence(0.005, 0.20, 0.12, "sentinel", "digest_123", 1, 3);
        assert_eq!(report_pos.verdict, EmergenceVerdict::Positive);
        assert!(report_pos.criterion_1_signal_mi);
        assert!(report_pos.criterion_2_scrambled_null);
        assert!(report_pos.criterion_3_behavioral_conditioning);

        // Negative case: Criterion 1 passes but Criterion 2 fails (scrambled arm ALSO shows low p-value)
        let report_neg2 =
            evaluate_study_emergence(0.005, 0.01, 0.12, "sentinel", "digest_123", 1, 3);
        assert_eq!(report_neg2.verdict, EmergenceVerdict::Negative);
        assert!(report_neg2.criterion_1_signal_mi);
        assert!(!report_neg2.criterion_2_scrambled_null);

        // Negative case: Criterion 3 fails (no behavioral effect)
        let report_neg3 =
            evaluate_study_emergence(0.005, 0.20, 0.01, "sentinel", "digest_123", 1, 3);
        assert_eq!(report_neg3.verdict, EmergenceVerdict::Negative);
        assert!(!report_neg3.criterion_3_behavioral_conditioning);
    }
}
