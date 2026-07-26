//! Detector kernel: pure statistics over metric series (`bd-16g.2.1`).
//!
//! This module turns a series of numbers into typed *detections*. It is a leaf:
//! it knows nothing about [`crate::WorldState`], storage, or any frontend, and it
//! must stay that way. Several consumers (the narrated timeline, speciation
//! cross-validation, highlight-reel selection, sonification, and the lab
//! assistant) all read from these primitives, and three half-detectors invented
//! independently by three consumers is the failure mode this module prevents.
//!
//! # Determinism
//!
//! Every function here is a pure function of its inputs: no wall clock, no RNG,
//! no map iteration, and no parallel reductions. Sums run in index order.
//! Sorting uses [`f64::total_cmp`]. The same series therefore yields the same
//! detections, bit for bit, on every platform.
//!
//! [`change_points_cusum`] is additionally *prefix-determined*: a detection at
//! index `i` depends only on `series[..=i]`. Feeding a series incrementally and
//! feeding it all at once produce identical output, which is what makes the
//! online (live HUD) and offline (post-hoc analysis) paths agree.
//!
//! # Non-finite input
//!
//! `NaN`/`inf` are rejected with a typed error rather than silently propagated.
//! A detector that reports `NaN` magnitudes is worse than one that reports
//! nothing, because it looks like it worked.

// bd-tqpj: deterministic-simulation policy — pinned floating-point evaluation
// order and fixed-width casts are part of the science contract; fma fusion,
// reassociation, or width changes alter world digests. Function lengths mirror
// the legacy C++ parity layout and are reviewed as units.
#![allow(clippy::suboptimal_flops, clippy::imprecise_flops)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use thiserror::Error;

/// One point of a metric series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// Simulation tick this value was observed at.
    pub tick: u64,
    /// Observed value. Must be finite.
    pub value: f64,
}

impl Sample {
    /// Convenience constructor.
    #[must_use]
    pub const fn new(tick: u64, value: f64) -> Self {
        Self { tick, value }
    }
}

/// Errors produced when a series or parameter set is unusable.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DetectError {
    /// The series contains `NaN` or an infinity.
    #[error("series contains a non-finite value at index {index}")]
    NonFinite {
        /// Index of the offending sample.
        index: usize,
    },
    /// Ticks must be strictly increasing so that windows are meaningful.
    #[error("series ticks must be strictly increasing (violated at index {index})")]
    UnorderedTicks {
        /// Index of the sample whose tick did not advance.
        index: usize,
    },
    /// A parameter was outside its documented domain.
    #[error("parameter `{name}` is invalid: {reason}")]
    InvalidParam {
        /// Parameter name.
        name: &'static str,
        /// Why it was rejected.
        reason: &'static str,
    },
}

/// Direction of a detected change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// The series stepped up relative to its baseline.
    Up,
    /// The series stepped down relative to its baseline.
    Down,
}

/// A detected change point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChangePoint {
    /// Tick at which the detector's statistic crossed its threshold.
    ///
    /// CUSUM is a *sequential* test: it fires once evidence has accumulated, so
    /// this tick lags the true onset of the change. Consumers that need the
    /// onset should search backwards from here; do not present this as the exact
    /// instant the world changed.
    pub tick: u64,
    /// Index into the input series.
    pub index: usize,
    /// Whether the series rose or fell.
    pub direction: Direction,
    /// Raw (unstandardized) deviation from the baseline mean at the firing sample.
    pub magnitude: f64,
    /// Mean of the baseline window this change was measured against.
    pub baseline_mean: f64,
    /// Value of the CUSUM statistic when it fired (in standard deviations).
    pub score: f64,
}

/// Parameters for [`change_points_cusum`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CusumParams {
    /// Number of leading samples used to estimate the baseline mean and sigma.
    ///
    /// After a detection the baseline is re-estimated from the samples that
    /// follow it, which is what lets one series contain several changes.
    pub warmup: usize,
    /// Slack, in baseline standard deviations. Drift smaller than `k` is ignored.
    pub k: f64,
    /// Decision threshold, in standard deviations of accumulated evidence.
    ///
    /// The default is deliberately conservative: a detector that cries wolf gets
    /// ignored, and an ignored detector is worth nothing. See the false-positive
    /// budget in `bd-16g.2.3`.
    pub h: f64,
    /// Floor applied to the baseline sigma so a perfectly flat baseline cannot
    /// divide by zero. Absolute (not relative) so that shift/scale invariance
    /// hold exactly whenever the real sigma exceeds this floor.
    pub min_sigma: f64,
    /// Hard cap on emitted detections; bounds memory on pathological input.
    pub max_detections: usize,
}

impl Default for CusumParams {
    fn default() -> Self {
        Self {
            warmup: 64,
            k: 0.5,
            h: 8.0,
            min_sigma: 1e-9,
            max_detections: 1024,
        }
    }
}

impl CusumParams {
    const fn validate(self) -> Result<(), DetectError> {
        if self.warmup < 2 {
            return Err(DetectError::InvalidParam {
                name: "warmup",
                reason: "must be at least 2 samples",
            });
        }
        if !self.k.is_finite() || self.k < 0.0 {
            return Err(DetectError::InvalidParam {
                name: "k",
                reason: "must be finite and non-negative",
            });
        }
        if !self.h.is_finite() || self.h <= 0.0 {
            return Err(DetectError::InvalidParam {
                name: "h",
                reason: "must be finite and positive",
            });
        }
        if !self.min_sigma.is_finite() || self.min_sigma <= 0.0 {
            return Err(DetectError::InvalidParam {
                name: "min_sigma",
                reason: "must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Detect level shifts with a two-sided CUSUM.
///
/// The series is standardized against a baseline estimated from `warmup`
/// samples; evidence accumulates while the standardized deviation exceeds `k`,
/// and a change is reported when the accumulated evidence exceeds `h`. After a
/// detection the baseline is re-estimated from the following samples, so a
/// series may contain many changes.
///
/// # Panics
///
/// Never: the warmup slice is guarded by the `base + params.warmup < series.len()` loop
/// condition, so every slice and index stays in bounds.
///
/// # Errors
///
/// Returns [`DetectError`] when the series contains non-finite values, its ticks
/// are not strictly increasing, or a parameter is outside its domain.
pub fn change_points_cusum(
    series: &[Sample],
    params: CusumParams,
) -> Result<Vec<ChangePoint>, DetectError> {
    params.validate()?;
    validate_series(series)?;

    let mut out = Vec::new();
    let mut base = 0usize;

    while base + params.warmup < series.len() {
        let (mean, sigma) = mean_sigma(&series[base..base + params.warmup]);
        let sigma = sigma.max(params.min_sigma);

        let mut s_hi = 0.0f64;
        let mut s_lo = 0.0f64;
        let mut fired = None;

        for (offset, sample) in series[base + params.warmup..].iter().enumerate() {
            let z = (sample.value - mean) / sigma;
            s_hi = (s_hi + z - params.k).max(0.0);
            s_lo = (s_lo - z - params.k).max(0.0);

            let up = s_hi > params.h;
            let down = s_lo > params.h;
            if up || down {
                let index = base + params.warmup + offset;
                let (direction, score) = if up && (!down || s_hi >= s_lo) {
                    (Direction::Up, s_hi)
                } else {
                    (Direction::Down, s_lo)
                };
                out.push(ChangePoint {
                    tick: sample.tick,
                    index,
                    direction,
                    magnitude: sample.value - mean,
                    baseline_mean: mean,
                    score,
                });
                fired = Some(index);
                break;
            }
        }

        match fired {
            // Re-baseline from just after the change so the next regime is
            // measured against itself rather than against the old one.
            Some(index) if out.len() < params.max_detections => base = index + 1,
            _ => break,
        }
    }

    Ok(out)
}

/// Which way a series must move through a level for a crossing to count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossDirection {
    /// Only report movement from above the level to at-or-below it.
    Falling,
    /// Only report movement from below the level to at-or-above it.
    Rising,
    /// Report movement in either direction.
    Either,
}

/// A configured level of interest (extinction at zero, a population floor, ...).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Threshold {
    /// Stable identifier carried into the emitted crossing.
    pub name: &'static str,
    /// The level itself.
    pub level: f64,
    /// Which transitions count.
    pub direction: CrossDirection,
}

/// A series moving through a [`Threshold`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crossing {
    /// Tick of the first sample on the far side of the level.
    pub tick: u64,
    /// Index into the input series.
    pub index: usize,
    /// Name of the threshold that was crossed.
    pub name: &'static str,
    /// The level that was crossed.
    pub level: f64,
    /// Direction of travel through the level.
    pub direction: Direction,
    /// Value immediately before the crossing.
    pub from: f64,
    /// Value at the crossing sample.
    pub to: f64,
}

/// Report every transition of `series` through each configured [`Threshold`].
///
/// Only *transitions* are reported: a series that sits below a level forever
/// produces one crossing (when it first arrives), not one per sample. This is
/// the difference between an event stream and a stuck alarm.
///
/// # Panics
///
/// Never: `windows(2)` yields slices of length exactly two, so `window[0]` and `window[1]` are
/// always in bounds.
///
/// # Errors
///
/// Returns [`DetectError`] when the series contains non-finite values, its ticks
/// are not strictly increasing, or a threshold level is non-finite.
pub fn threshold_crossings(
    series: &[Sample],
    thresholds: &[Threshold],
) -> Result<Vec<Crossing>, DetectError> {
    validate_series(series)?;
    for threshold in thresholds {
        if !threshold.level.is_finite() {
            return Err(DetectError::InvalidParam {
                name: "threshold.level",
                reason: "must be finite",
            });
        }
    }

    let mut out = Vec::new();
    for (i, window) in series.windows(2).enumerate() {
        let (previous, current) = (window[0], window[1]);
        for threshold in thresholds {
            let was_above = previous.value > threshold.level;
            let is_above = current.value > threshold.level;
            if was_above == is_above {
                continue;
            }
            let direction = if was_above {
                Direction::Down
            } else {
                Direction::Up
            };
            let wanted = match threshold.direction {
                CrossDirection::Falling => direction == Direction::Down,
                CrossDirection::Rising => direction == Direction::Up,
                CrossDirection::Either => true,
            };
            if !wanted {
                continue;
            }
            let index = i + 1;
            out.push(Crossing {
                tick: current.tick,
                index,
                name: threshold.name,
                level: threshold.level,
                direction,
                from: previous.value,
                to: current.value,
            });
        }
    }
    Ok(out)
}

/// Coarse classification of what a stretch of a series is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Sustained rise.
    Growth,
    /// Flat within tolerance.
    Equilibrium,
    /// Repeatedly crossing its own trend.
    Oscillation,
    /// Sustained fall.
    Collapse,
}

/// One classified window of a series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegimeWindow {
    /// First tick of the window.
    pub start_tick: u64,
    /// Last tick of the window.
    pub end_tick: u64,
    /// Classification.
    pub regime: Regime,
    /// Least-squares slope per sample, relative to the window mean.
    pub relative_slope: f64,
    /// Rate at which the detrended series crosses zero, in `[0, 1]`.
    pub crossing_rate: f64,
    /// Standard deviation of the detrended series, relative to the window mean.
    pub relative_spread: f64,
    /// Lag-1 autocorrelation of the detrended series, in `[-1, 1]`.
    ///
    /// This is what separates a *signal* from *noise*: a sampled periodic
    /// series is smooth (neighbouring residuals agree, so this is high), while
    /// white noise has no lag-1 structure (this is ~0). Zero-crossing rate
    /// cannot make that distinction — noise crosses its own mean *more* often
    /// than a slow oscillation does, so thresholding on crossings alone
    /// classifies a noisy equilibrium as an oscillation and misses the real one.
    pub autocorrelation: f64,
}

/// Parameters for [`regimes`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegimeParams {
    /// Samples per (non-overlapping) window.
    pub window: usize,
    /// Relative slope at or above which a window counts as growth.
    pub growth_slope: f64,
    /// Relative slope at or below which a window counts as collapse.
    pub collapse_slope: f64,
    /// Minimum lag-1 autocorrelation of the detrended series for oscillation.
    ///
    /// White noise sits at ~0; any adequately sampled periodic signal sits well
    /// above (a sinusoid's lag-1 autocorrelation is `cos(w)` for angular
    /// frequency `w`). The default admits periods down to roughly 5 samples
    /// while still rejecting noise.
    pub oscillation_autocorrelation: f64,
    /// Minimum relative spread for oscillation (keeps quiet noise out).
    pub oscillation_spread: f64,
    /// Minimum number of detrended zero crossings (at least one full cycle).
    pub oscillation_min_crossings: usize,
    /// Floor for the window mean when normalizing (avoids dividing by zero).
    pub min_scale: f64,
}

impl Default for RegimeParams {
    fn default() -> Self {
        Self {
            window: 64,
            growth_slope: 0.002,
            collapse_slope: -0.002,
            oscillation_autocorrelation: 0.30,
            oscillation_spread: 0.05,
            oscillation_min_crossings: 2,
            min_scale: 1e-9,
        }
    }
}

impl RegimeParams {
    const fn validate(self) -> Result<(), DetectError> {
        if self.window < 4 {
            return Err(DetectError::InvalidParam {
                name: "window",
                reason: "must be at least 4 samples",
            });
        }
        if !self.min_scale.is_finite() || self.min_scale <= 0.0 {
            return Err(DetectError::InvalidParam {
                name: "min_scale",
                reason: "must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Classify a series into a timeline of regimes, one per non-overlapping window.
///
/// A run yields a *sequence* of regimes rather than a single label, because
/// "this run collapsed" is far less useful than "this run grew, equilibrated,
/// then collapsed at t=9,000".
///
/// # Panics
///
/// Never: `chunks_exact` yields only full windows, so `chunk[0]` and `chunk[chunk.len() - 1]`
/// are always in bounds.
///
/// # Errors
///
/// Returns [`DetectError`] when the series is invalid or a parameter is outside
/// its domain.
pub fn regimes(series: &[Sample], params: RegimeParams) -> Result<Vec<RegimeWindow>, DetectError> {
    params.validate()?;
    validate_series(series)?;

    let mut out = Vec::with_capacity(series.len() / params.window);
    for chunk in series.chunks_exact(params.window) {
        let n = chunk.len() as f64;
        let (mean, _) = mean_sigma(chunk);
        let scale = mean.abs().max(params.min_scale);

        // Least-squares slope against the sample index (ticks are strictly
        // increasing but may be unevenly spaced; index keeps this scale-free).
        let mean_x = (n - 1.0) / 2.0;
        let mut sum_dx_dy = 0.0f64;
        let mut sum_dx_sq = 0.0f64;
        for (i, sample) in chunk.iter().enumerate() {
            let dx = i as f64 - mean_x;
            sum_dx_dy += dx * (sample.value - mean);
            sum_dx_sq += dx * dx;
        }
        let slope = if sum_dx_sq > 0.0 {
            sum_dx_dy / sum_dx_sq
        } else {
            0.0
        };

        // Detrended residuals: spread and zero-crossing rate.
        let mut residuals = Vec::with_capacity(chunk.len());
        for (i, sample) in chunk.iter().enumerate() {
            let fit = mean + slope * (i as f64 - mean_x);
            residuals.push(sample.value - fit);
        }
        let mut sum_sq = 0.0f64;
        for residual in &residuals {
            sum_sq += residual * residual;
        }
        let spread = (sum_sq / n).sqrt();
        let mut crossings = 0usize;
        for pair in residuals.windows(2) {
            if (pair[0] > 0.0 && pair[1] < 0.0) || (pair[0] < 0.0 && pair[1] > 0.0) {
                crossings += 1;
            }
        }
        let crossing_rate = crossings as f64 / (n - 1.0);

        // Lag-1 autocorrelation of the residuals: high for a smooth periodic
        // signal, ~0 for white noise. This is the discriminator; the crossing
        // count only confirms that at least one full cycle is present.
        let mut lag_product = 0.0f64;
        for pair in residuals.windows(2) {
            lag_product += pair[0] * pair[1];
        }
        let autocorrelation = if sum_sq > 0.0 {
            (lag_product / sum_sq).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let relative_slope = slope / scale;
        let relative_spread = spread / scale;
        let regime = if relative_slope <= params.collapse_slope {
            Regime::Collapse
        } else if relative_slope >= params.growth_slope {
            Regime::Growth
        } else if autocorrelation >= params.oscillation_autocorrelation
            && relative_spread >= params.oscillation_spread
            && crossings >= params.oscillation_min_crossings
        {
            Regime::Oscillation
        } else {
            Regime::Equilibrium
        };

        out.push(RegimeWindow {
            start_tick: chunk[0].tick,
            end_tick: chunk[chunk.len() - 1].tick,
            regime,
            relative_slope,
            crossing_rate,
            relative_spread,
            autocorrelation,
        });
    }
    Ok(out)
}

/// Evidence that a population's phenotype distribution has split in two.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BimodalityScore {
    /// Between-class variance as a fraction of total variance, in `[0, 1]`.
    pub score: f64,
    /// Distance between cluster means, in total standard deviations.
    pub separation: f64,
    /// Value separating the two clusters.
    pub split: f64,
    /// Mean of the lower cluster.
    pub lower_mean: f64,
    /// Mean of the upper cluster.
    pub upper_mean: f64,
    /// Population of the lower cluster.
    pub lower_count: usize,
    /// Population of the upper cluster.
    pub upper_count: usize,
    /// Whether the configured bimodality criteria were met.
    ///
    /// This is a *hint* only. Speciation is confirmed against ancestry in
    /// `bd-16g.3.3`; a bimodal histogram alone never justifies the claim.
    pub is_bimodal: bool,
}

/// Parameters for [`bimodality`].
///
/// # Calibrating these thresholds
///
/// The defaults are not taste; they sit between two analytically known points,
/// and moving them without re-deriving these numbers will produce a detector
/// that calls every population a new species.
///
/// * A **perfectly separated** pair of equal clusters scores `1.0` (all variance
///   is between-cluster) with a separation of **exactly** `2.0` total sigma.
///   A threshold *at* `2.0` therefore sits on a floating-point knife edge and
///   rejects the textbook case roughly half the time.
/// * A **unimodal** Gaussian, split at its own mean, still yields two halves
///   whose means are ~`1.6` sigma apart, giving an Otsu score of ~`0.64`. Any
///   `min_score` at or below that certifies pure noise as speciation.
///
/// The defaults (`min_score = 0.80`, `min_separation = 1.8`) fall strictly
/// between the unimodal ceiling and the perfect-split floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BimodalityParams {
    /// Minimum between/total variance ratio for the hint to fire.
    pub min_score: f64,
    /// Minimum separation between cluster means, in total standard deviations.
    pub min_separation: f64,
    /// Minimum share of the population each cluster must hold, in `[0, 0.5]`.
    pub min_cluster_fraction: f64,
    /// Floor for the total standard deviation when normalizing.
    pub min_sigma: f64,
}

impl Default for BimodalityParams {
    fn default() -> Self {
        Self {
            min_score: 0.80,
            min_separation: 1.8,
            min_cluster_fraction: 0.15,
            min_sigma: 1e-9,
        }
    }
}

/// Score a set of per-agent values (one phenotype dimension) for bimodality.
///
/// Otsu's method over a fixed-width histogram: deterministic, no iteration, no
/// seeded initialization, and therefore no chance of two runs disagreeing about
/// whether a population had split.
///
/// # Complexity (bd-16g.2.11)
///
/// **O(n) time, O(1) working memory.** Two passes over the input plus one pass
/// over [`BIMODALITY_BINS`] bins, which is independent of `n`. This previously
/// sorted a full copy of the sample — O(n log n) time and O(n) memory — which
/// contradicted the leaf contract this primitive advertises.
///
/// # Exactness and the approximation envelope
///
/// The quantisation is confined to the CANDIDATE SET, not to the answer:
///
/// * **Exact** for the chosen split — counts, means, between-class variance,
///   score, and separation are computed from true per-bin sums of the original
///   values, never from bin centres. For any given threshold the reported class
///   statistics are precisely what a sorted implementation would report.
/// * **Approximate** only in *which* thresholds are considered. The sorted
///   version could split between any two adjacent order statistics; this version
///   considers bin boundaries. The selected split therefore lies within one bin
///   width, `(max - min) / BIMODALITY_BINS`, of the exhaustive optimum.
///
/// Consequence worth knowing before trusting a borderline result: on a sample
/// whose score sits within rounding distance of `min_score`, the two versions can
/// disagree about `is_bimodal`. That is inherent to any bounded-memory formulation
/// and is why the envelope is stated rather than implied. Well-separated
/// populations — the case this detector exists for — are unaffected, since their
/// optimum is far from any threshold boundary.
///
/// Ties resolve to the lowest boundary (`>` on the running best), matching the
/// previous rank scan, so equal-scoring splits pick the same side as before.
///
/// # Panics
///
/// Never: bin indices are clamped into range and the split search only accepts
/// boundaries where both classes are non-empty.
///
/// # Errors
///
/// Returns [`DetectError::NonFinite`] when any value is non-finite.
pub fn bimodality(
    values: &[f64],
    params: BimodalityParams,
) -> Result<BimodalityScore, DetectError> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(DetectError::NonFinite { index });
        }
    }
    if values.len() < 2 {
        return Ok(BimodalityScore {
            score: 0.0,
            separation: 0.0,
            split: values.first().copied().unwrap_or_default(),
            lower_mean: values.first().copied().unwrap_or_default(),
            upper_mean: values.first().copied().unwrap_or_default(),
            lower_count: values.len(),
            upper_count: 0,
            is_bimodal: false,
        });
    }

    // PASS 1: extent only. `min`/`max` are order-independent by construction.
    let n = values.len() as f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        if *value < min {
            min = *value;
        }
        if *value > max {
            max = *value;
        }
    }

    if min == max {
        // Every value identical: no split exists. Detected from the extent rather
        // than from a failed search, so it costs nothing and cannot be confused
        // with "the search found no positive between-class variance".
        return Ok(BimodalityScore {
            score: 0.0,
            separation: 0.0,
            split: min,
            lower_mean: min,
            upper_mean: min,
            lower_count: values.len(),
            upper_count: 0,
            is_bimodal: false,
        });
    }

    // PASS 2: the histogram.
    let mut counts = [0usize; BIMODALITY_BINS];
    let mut sums = [0.0f64; BIMODALITY_BINS];
    let mut mins = [f64::INFINITY; BIMODALITY_BINS];
    let span = max - min;
    for value in values {
        let bin = bimodality_bin(*value, min, span);
        counts[bin] += 1;
        sums[bin] += *value;
        if *value < mins[bin] {
            mins[bin] = *value;
        }
    }

    // EVERY aggregate below is folded in BIN order, never input order. That is what
    // preserves permutation invariance: floating-point addition is not associative,
    // so summing the same multiset in a different sequence changes the last bits.
    // The sorted implementation got this for free by summing in sorted order; a
    // histogram has to reach it deliberately. Bin order is canonical for a given
    // multiset, so the only residual order dependence is between DISTINCT values
    // that share a bin, bounded by one ulp of that bin's sum.
    let mut total = 0.0f64;
    for bin_sum in &sums {
        total += *bin_sum;
    }
    let mean = total / n;

    // PASS 3: dispersion about the canonical mean, also folded in bin order.
    let mut sq_sums = [0.0f64; BIMODALITY_BINS];
    for value in values {
        let bin = bimodality_bin(*value, min, span);
        let d = *value - mean;
        sq_sums[bin] += d * d;
    }
    let mut sum_sq = 0.0f64;
    for bin_sq in &sq_sums {
        sum_sq += *bin_sq;
    }
    let total_variance = sum_sq / n;
    let total_sigma = total_variance.sqrt().max(params.min_sigma);

    // PASS 3: Otsu over BIN BOUNDARIES, O(BIMODALITY_BINS) and independent of n.
    // `>` keeps the lowest-scoring-tie boundary, matching the previous rank scan.
    let mut best_between = 0.0f64;
    let mut best_boundary = 0usize;
    let mut prefix_count = 0usize;
    let mut prefix_sum = 0.0f64;
    for boundary in 1..BIMODALITY_BINS {
        prefix_count += counts[boundary - 1];
        prefix_sum += sums[boundary - 1];
        if prefix_count == 0 || prefix_count == values.len() {
            continue;
        }
        let lower_n = prefix_count as f64;
        let upper_n = n - lower_n;
        let w0 = lower_n / n;
        let w1 = 1.0 - w0;
        let mu0 = prefix_sum / lower_n;
        let mu1 = (total - prefix_sum) / upper_n;
        let between = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
        if between > best_between {
            best_between = between;
            best_boundary = boundary;
        }
    }

    if best_boundary == 0 {
        return Ok(BimodalityScore {
            score: 0.0,
            separation: 0.0,
            split: mean,
            lower_mean: mean,
            upper_mean: mean,
            lower_count: values.len(),
            upper_count: 0,
            is_bimodal: false,
        });
    }

    // Class statistics for the chosen boundary are EXACT in the sense that matters:
    // `sums` holds true values, never bin centres, so only the candidate SET was
    // quantised, not the reported answer.
    //
    // `upper_sum` is folded from its own bins rather than taken as `total -
    // lower_sum`. The subtraction would be algebraically identical and numerically
    // not: it inherits `total`'s rounding, which made the result depend on input
    // order and broke permutation invariance.
    let mut lower_count = 0usize;
    let mut lower_sum = 0.0f64;
    for bin in 0..best_boundary {
        lower_count += counts[bin];
        lower_sum += sums[bin];
    }
    let mut upper_count = 0usize;
    let mut upper_sum = 0.0f64;
    for bin in best_boundary..BIMODALITY_BINS {
        upper_count += counts[bin];
        upper_sum += sums[bin];
    }
    let lower_mean = lower_sum / lower_count as f64;
    let upper_mean = upper_sum / upper_count as f64;

    // Report an actual observed value, as the sorted implementation did: the
    // smallest member of the upper class.
    let mut split = max;
    for bin in best_boundary..BIMODALITY_BINS {
        if counts[bin] > 0 {
            split = mins[bin];
            break;
        }
    }

    let score = if total_variance > 0.0 {
        (best_between / total_variance).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let separation = (upper_mean - lower_mean).abs() / total_sigma;
    let smaller = lower_count.min(upper_count) as f64 / n;
    let is_bimodal = score >= params.min_score
        && separation >= params.min_separation
        && smaller >= params.min_cluster_fraction;

    Ok(BimodalityScore {
        score,
        separation,
        split,
        lower_mean,
        upper_mean,
        lower_count,
        upper_count,
        is_bimodal,
    })
}

/// Bin count for the linear-time Otsu search (bd-16g.2.11).
///
/// Fixed, so working memory is O(1) in the sample count: three arrays of this
/// length, ~20 KiB total, regardless of whether the caller passes 100 values or
/// 10 million. That bound is the contract; the value itself is a resolution
/// choice and can be raised without changing any documented semantics.
const BIMODALITY_BINS: usize = 1024;

/// Map a value onto its histogram bin. `span` must be strictly positive.
///
/// Clamped rather than asserted: floating-point rounding can put `max` itself at
/// index `BIMODALITY_BINS`, and a detector that panics on its own maximum would
/// be worse than one that puts it in the top bin where it belongs.
fn bimodality_bin(value: f64, min: f64, span: f64) -> usize {
    let scaled = (value - min) / span * BIMODALITY_BINS as f64;
    if scaled <= 0.0 {
        return 0;
    }
    let index = scaled as usize;
    index.min(BIMODALITY_BINS - 1)
}

fn validate_series(series: &[Sample]) -> Result<(), DetectError> {
    for (index, sample) in series.iter().enumerate() {
        if !sample.value.is_finite() {
            return Err(DetectError::NonFinite { index });
        }
        if index > 0 && sample.tick <= series[index - 1].tick {
            return Err(DetectError::UnorderedTicks { index });
        }
    }
    Ok(())
}

/// Mean and population standard deviation, summed in index order for bitwise
/// stability across platforms and thread counts.
fn mean_sigma(series: &[Sample]) -> (f64, f64) {
    if series.is_empty() {
        return (0.0, 0.0);
    }
    let n = series.len() as f64;
    let mut sum = 0.0f64;
    for sample in series {
        sum += sample.value;
    }
    let mean = sum / n;
    let mut sum_sq = 0.0f64;
    for sample in series {
        let d = sample.value - mean;
        sum_sq += d * d;
    }
    (mean, (sum_sq / n).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    fn series_from(values: &[f64]) -> Vec<Sample> {
        values
            .iter()
            .enumerate()
            .map(|(i, v)| Sample::new(i as u64, *v))
            .collect()
    }

    fn step(before: f64, after: f64, at: usize, len: usize) -> Vec<Sample> {
        let values: Vec<f64> = (0..len)
            .map(|i| if i < at { before } else { after })
            .collect();
        series_from(&values)
    }

    /// Deterministic pseudo-noise: fixed seed, so this test is a regression pin
    /// on the default sensitivity rather than a probabilistic assertion.
    fn noise(len: usize, mean: f64, sigma: f64, seed: u64) -> Vec<Sample> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let values: Vec<f64> = (0..len)
            .map(|_| {
                let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
                let u2: f64 = rng.random::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                mean + z * sigma
            })
            .collect();
        series_from(&values)
    }

    #[test]
    fn cusum_detects_step_down_with_direction_and_magnitude() {
        let series = step(1000.0, 400.0, 300, 600);
        let found = change_points_cusum(&series, CusumParams::default()).expect("valid series");
        assert_eq!(found.len(), 1, "one step should yield one change point");
        let cp = found[0];
        assert_eq!(cp.direction, Direction::Down);
        // CUSUM is sequential: it fires shortly after the true onset, never before.
        assert!(cp.index >= 300, "must not fire before the change");
        assert!(
            cp.index <= 320,
            "should fire promptly after the change, got {}",
            cp.index
        );
        assert!((cp.baseline_mean - 1000.0).abs() < 1e-9);
        assert!((cp.magnitude + 600.0).abs() < 1e-9);
    }

    #[test]
    fn cusum_detects_step_up() {
        let series = step(50.0, 90.0, 200, 500);
        let found = change_points_cusum(&series, CusumParams::default()).expect("valid series");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].direction, Direction::Up);
        assert!(found[0].index >= 200 && found[0].index <= 220);
    }

    #[test]
    fn cusum_detects_multiple_changes_by_rebaselining() {
        let mut values = vec![100.0; 200];
        values.extend(std::iter::repeat_n(300.0, 200));
        values.extend(std::iter::repeat_n(120.0, 200));
        let series = series_from(&values);
        let found = change_points_cusum(&series, CusumParams::default()).expect("valid series");
        assert_eq!(found.len(), 2, "two steps should yield two change points");
        assert_eq!(found[0].direction, Direction::Up);
        assert_eq!(found[1].direction, Direction::Down);
    }

    #[test]
    fn cusum_is_silent_on_a_flat_line() {
        let series = series_from(&[42.0; 1000]);
        let found = change_points_cusum(&series, CusumParams::default()).expect("valid series");
        assert!(found.is_empty(), "a flat line has no changes: {found:?}");
    }

    #[test]
    fn cusum_respects_the_false_positive_budget_on_noise() {
        // A detector that fires on noise gets ignored, and an ignored detector is
        // worthless. The budget is asserted, not hoped for.
        for seed in 0..8u64 {
            let series = noise(2000, 500.0, 25.0, seed);
            let found = change_points_cusum(&series, CusumParams::default()).expect("valid");
            assert!(
                found.len() <= 1,
                "seed {seed}: {} false alarms over 2000 noise samples (budget 1)",
                found.len()
            );
        }
    }

    #[test]
    fn cusum_judges_drift_relative_to_noise_not_in_absolute_units() {
        // "Gentle" is only meaningful relative to the baseline's own variability.
        // On a *noiseless* ramp the baseline sigma is nearly zero, so even a
        // slow drift is genuinely significant and SHOULD fire — a detector that
        // stayed silent there would be broken, not polite. Noise is what makes a
        // small slope insignificant, so that is what these fixtures vary.
        let noisy = noise(1200, 500.0, 25.0, 11);
        let gentle: Vec<Sample> = noisy
            .iter()
            .enumerate()
            .map(|(i, s)| Sample::new(s.tick, s.value + i as f64 * 0.001))
            .collect();
        let found = change_points_cusum(&gentle, CusumParams::default()).expect("valid");
        assert!(
            found.is_empty(),
            "drift far below the noise floor must not fire: {found:?}"
        );

        let steep: Vec<Sample> = noisy
            .iter()
            .enumerate()
            .map(|(i, s)| Sample::new(s.tick, s.value + i as f64 * 0.5))
            .collect();
        let found = change_points_cusum(&steep, CusumParams::default()).expect("valid");
        assert!(!found.is_empty(), "drift far above the noise floor is real");
        assert_eq!(found[0].direction, Direction::Up);

        // And the noiseless ramp: sigma ~ 0, so this is a real change.
        let noiseless: Vec<f64> = (0..800).map(|i| 100.0 + f64::from(i) * 0.001).collect();
        let found =
            change_points_cusum(&series_from(&noiseless), CusumParams::default()).expect("valid");
        assert!(
            !found.is_empty(),
            "a perfectly clean drift off a flat baseline is significant"
        );
    }

    #[test]
    fn cusum_is_prefix_determined_so_online_equals_offline() {
        // This property is what makes the live HUD and the post-hoc analysis
        // agree; without it the timeline you watched would disagree with the
        // paper you wrote.
        let series = step(80.0, 20.0, 250, 500);
        let batch = change_points_cusum(&series, CusumParams::default()).expect("valid");

        let mut incremental: Vec<ChangePoint> = Vec::new();
        for len in 1..=series.len() {
            let partial =
                change_points_cusum(&series[..len], CusumParams::default()).expect("valid");
            if partial.len() > incremental.len() {
                incremental = partial;
            }
        }
        assert_eq!(batch, incremental);
    }

    #[test]
    fn cusum_is_shift_and_scale_invariant() {
        let base = step(200.0, 60.0, 300, 600);
        let expected = change_points_cusum(&base, CusumParams::default()).expect("valid");

        let shifted: Vec<Sample> = base
            .iter()
            .map(|s| Sample::new(s.tick, s.value + 1234.5))
            .collect();
        let shifted_found = change_points_cusum(&shifted, CusumParams::default()).expect("valid");
        assert_eq!(
            expected.iter().map(|c| c.index).collect::<Vec<_>>(),
            shifted_found.iter().map(|c| c.index).collect::<Vec<_>>(),
            "adding a constant must not move a change point"
        );

        let scaled: Vec<Sample> = base
            .iter()
            .map(|s| Sample::new(s.tick, s.value * 100.0))
            .collect();
        let scaled_found = change_points_cusum(&scaled, CusumParams::default()).expect("valid");
        assert_eq!(
            expected.iter().map(|c| c.index).collect::<Vec<_>>(),
            scaled_found.iter().map(|c| c.index).collect::<Vec<_>>(),
            "scaling must not move a change point"
        );
    }

    #[test]
    fn cusum_is_deterministic_across_repeated_calls() {
        let series = step(10.0, 40.0, 150, 400);
        let a = change_points_cusum(&series, CusumParams::default()).expect("valid");
        let b = change_points_cusum(&series, CusumParams::default()).expect("valid");
        assert_eq!(a, b);
    }

    #[test]
    fn detectors_reject_non_finite_and_unordered_input() {
        let bad = [Sample::new(0, 1.0), Sample::new(1, f64::NAN)];
        assert_eq!(
            change_points_cusum(&bad, CusumParams::default()),
            Err(DetectError::NonFinite { index: 1 })
        );

        let unordered = [Sample::new(5, 1.0), Sample::new(5, 2.0)];
        assert_eq!(
            change_points_cusum(&unordered, CusumParams::default()),
            Err(DetectError::UnorderedTicks { index: 1 })
        );

        assert_eq!(
            bimodality(&[1.0, f64::INFINITY], BimodalityParams::default()),
            Err(DetectError::NonFinite { index: 1 })
        );
    }

    #[test]
    fn detectors_survive_degenerate_series() {
        for len in 0..3usize {
            let series: Vec<Sample> = (0..len).map(|i| Sample::new(i as u64, 1.0)).collect();
            assert!(
                change_points_cusum(&series, CusumParams::default())
                    .expect("degenerate series is valid, just uninteresting")
                    .is_empty()
            );
            assert!(
                regimes(&series, RegimeParams::default())
                    .expect("valid")
                    .is_empty()
            );
            assert!(threshold_crossings(&series, &[]).expect("valid").is_empty());
        }
    }

    #[test]
    fn cusum_rejects_invalid_params() {
        let series = series_from(&[1.0, 2.0, 3.0]);
        let params = CusumParams {
            h: 0.0,
            ..CusumParams::default()
        };
        assert_eq!(
            change_points_cusum(&series, params),
            Err(DetectError::InvalidParam {
                name: "h",
                reason: "must be finite and positive"
            })
        );
    }

    #[test]
    fn threshold_crossings_report_transitions_not_states() {
        // Extinction: population reaches zero and stays there. One event, not 500.
        let mut values = vec![10.0, 6.0, 3.0];
        values.extend(std::iter::repeat_n(0.0, 500));
        let series = series_from(&values);
        let extinction = Threshold {
            name: "extinction",
            level: 0.0,
            direction: CrossDirection::Falling,
        };
        let found = threshold_crossings(&series, &[extinction]).expect("valid");
        assert_eq!(found.len(), 1, "a stuck alarm is not an event stream");
        assert_eq!(found[0].tick, 3);
        assert_eq!(found[0].direction, Direction::Down);
        assert_eq!(found[0].name, "extinction");
    }

    #[test]
    fn threshold_crossings_honor_direction_filters() {
        let series = series_from(&[0.0, 5.0, 0.0, 5.0]);
        let rising = Threshold {
            name: "floor",
            level: 1.0,
            direction: CrossDirection::Rising,
        };
        let found = threshold_crossings(&series, &[rising]).expect("valid");
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|c| c.direction == Direction::Up));

        let either = Threshold {
            direction: CrossDirection::Either,
            ..rising
        };
        assert_eq!(
            threshold_crossings(&series, &[either])
                .expect("valid")
                .len(),
            3
        );
    }

    #[test]
    fn regimes_classify_growth_equilibrium_and_collapse() {
        let growth: Vec<f64> = (0..64).map(|i| 100.0 + f64::from(i) * 2.0).collect();
        let windows = regimes(&series_from(&growth), RegimeParams::default()).expect("valid");
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].regime, Regime::Growth);

        let flat = [100.0; 64];
        let windows = regimes(&series_from(&flat), RegimeParams::default()).expect("valid");
        assert_eq!(windows[0].regime, Regime::Equilibrium);

        let collapse: Vec<f64> = (0..64).map(|i| 500.0 - f64::from(i) * 4.0).collect();
        let windows = regimes(&series_from(&collapse), RegimeParams::default()).expect("valid");
        assert_eq!(windows[0].regime, Regime::Collapse);
    }

    #[test]
    fn regimes_classify_oscillation() {
        let osc: Vec<f64> = (0..128)
            .map(|i| 200.0 + 30.0 * (f64::from(i) * 0.9).sin())
            .collect();
        let windows = regimes(&series_from(&osc), RegimeParams::default()).expect("valid");
        assert!(
            windows.iter().all(|w| w.regime == Regime::Oscillation),
            "a clean sinusoid is an oscillation: {windows:?}"
        );
    }

    #[test]
    fn regimes_do_not_mistake_noise_for_oscillation() {
        // This is the test that forced the classifier onto lag-1 autocorrelation.
        // White noise crosses its own mean MORE often (~0.5) than a sampled
        // sinusoid does (~0.29), so any crossing-rate threshold gets it exactly
        // backwards: it would flag a noisy equilibrium and miss the real signal.
        let noisy = noise(256, 400.0, 40.0, 3);
        let windows = regimes(&noisy, RegimeParams::default()).expect("valid");
        assert!(
            windows.iter().all(|w| w.regime != Regime::Oscillation),
            "noise is not an oscillation: {windows:?}"
        );
        for window in &windows {
            assert!(
                window.crossing_rate > 0.35,
                "noise really does cross often ({}); the rate alone cannot be the test",
                window.crossing_rate
            );
            assert!(
                window.autocorrelation < 0.30,
                "noise has no lag-1 structure ({})",
                window.autocorrelation
            );
        }
    }

    #[test]
    fn regimes_produce_a_timeline_not_a_single_label() {
        let mut values: Vec<f64> = (0..64).map(|i| 100.0 + f64::from(i) * 3.0).collect();
        values.extend(std::iter::repeat_n(292.0, 64));
        let windows = regimes(&series_from(&values), RegimeParams::default()).expect("valid");
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].regime, Regime::Growth);
        assert_eq!(windows[1].regime, Regime::Equilibrium);
    }

    #[test]
    fn bimodality_separates_two_clusters_and_stays_quiet_on_one() {
        let mut split: Vec<f64> = vec![0.1; 60];
        split.extend(std::iter::repeat_n(0.9, 60));
        let score = bimodality(&split, BimodalityParams::default()).expect("valid");
        assert!(
            score.is_bimodal,
            "two tight clusters are bimodal: {score:?}"
        );
        assert_eq!(score.lower_count, 60);
        assert_eq!(score.upper_count, 60);
        assert!((score.lower_mean - 0.1).abs() < 1e-9);
        assert!((score.upper_mean - 0.9).abs() < 1e-9);

        let unimodal = noise(200, 0.5, 0.05, 7)
            .iter()
            .map(|s| s.value)
            .collect::<Vec<_>>();
        let score = bimodality(&unimodal, BimodalityParams::default()).expect("valid");
        assert!(
            !score.is_bimodal,
            "a single noisy cluster is not speciation: {score:?}"
        );
    }

    #[test]
    fn bimodality_thresholds_sit_between_the_two_analytic_reference_points() {
        // These two numbers are why the defaults are what they are. Pin them: a
        // future "harmless" tweak that drops min_score to 0.6 would certify
        // every noisy population as a new species, and a min_separation of
        // exactly 2.0 would reject the textbook perfect split on a float
        // knife-edge.

        // Reference 1: a perfectly separated equal pair. Score is exactly 1.0
        // and separation is exactly 2.0 sigma — no threshold may sit AT 2.0.
        let mut perfect: Vec<f64> = vec![0.0; 100];
        perfect.extend(std::iter::repeat_n(1.0, 100));
        let score = bimodality(&perfect, BimodalityParams::default()).expect("valid");
        assert!((score.score - 1.0).abs() < 1e-9, "score {}", score.score);
        assert!(
            (score.separation - 2.0).abs() < 1e-9,
            "separation {}",
            score.separation
        );
        assert!(score.is_bimodal);

        // Reference 2: a unimodal Gaussian. Otsu still finds *a* split, and it
        // still looks impressive (~0.64 score, ~1.6 sigma apart) — which is
        // exactly the trap. It must not be called bimodal.
        let gaussian: Vec<f64> = noise(4000, 0.0, 1.0, 99).iter().map(|s| s.value).collect();
        let score = bimodality(&gaussian, BimodalityParams::default()).expect("valid");
        assert!(
            score.score > 0.55 && score.score < 0.75,
            "a unimodal Gaussian's Otsu ceiling should land near 0.64, got {}",
            score.score
        );
        assert!(
            !score.is_bimodal,
            "the defaults must reject the unimodal ceiling: {score:?}"
        );
    }

    #[test]
    fn bimodality_ignores_a_tiny_outlier_cluster() {
        // Three oddballs are not a new species; the minimum-cluster-fraction rule
        // is what keeps the speciation hint from firing on outliers.
        let mut values = vec![0.5; 200];
        values.extend_from_slice(&[5.0, 5.1, 5.2]);
        let score = bimodality(&values, BimodalityParams::default()).expect("valid");
        assert!(!score.is_bimodal, "outliers are not a clade: {score:?}");
    }

    #[test]
    fn bimodality_is_deterministic_and_order_independent() {
        let mut values: Vec<f64> = vec![0.2; 50];
        values.extend(std::iter::repeat_n(0.8, 50));
        let forward = bimodality(&values, BimodalityParams::default()).expect("valid");
        values.reverse();
        let reversed = bimodality(&values, BimodalityParams::default()).expect("valid");
        assert_eq!(forward, reversed);
    }

    #[test]
    fn bimodality_handles_all_equal_values() {
        let score = bimodality(&[3.0; 100], BimodalityParams::default()).expect("valid");
        assert!(!score.is_bimodal);
        assert!((score.score).abs() < 1e-12);
    }

    // ---- bd-16g.2.11: linear-time bimodality vs a sorting oracle ----

    /// The pre-bd-16g.2.11 implementation, kept verbatim as an independent oracle.
    ///
    /// Its whole value is that it is the OTHER algorithm: a full sort with an
    /// exhaustive scan over every rank. A test that re-derived the histogram logic
    /// would only prove the code agrees with itself.
    fn bimodality_sorting_oracle(values: &[f64], params: BimodalityParams) -> BimodalityScore {
        let mut sorted = values.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        let n = sorted.len() as f64;
        let total: f64 = sorted.iter().sum();
        let mean = total / n;
        let sum_sq: f64 = sorted.iter().map(|v| (v - mean) * (v - mean)).sum();
        let total_variance = sum_sq / n;
        let total_sigma = total_variance.sqrt().max(params.min_sigma);
        let mut best = (0.0f64, 0usize);
        let mut prefix_sum = 0.0f64;
        for i in 1..sorted.len() {
            prefix_sum += sorted[i - 1];
            let w0 = i as f64 / n;
            let w1 = 1.0 - w0;
            let mu0 = prefix_sum / i as f64;
            let mu1 = (total - prefix_sum) / (n - i as f64);
            let between = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
            if between > best.0 {
                best = (between, i);
            }
        }
        let (between_variance, split_index) = best;
        if split_index == 0 {
            return BimodalityScore {
                score: 0.0,
                separation: 0.0,
                split: mean,
                lower_mean: mean,
                upper_mean: mean,
                lower_count: sorted.len(),
                upper_count: 0,
                is_bimodal: false,
            };
        }
        let lower = &sorted[..split_index];
        let upper = &sorted[split_index..];
        let lower_mean = lower.iter().sum::<f64>() / lower.len() as f64;
        let upper_mean = upper.iter().sum::<f64>() / upper.len() as f64;
        let score = if total_variance > 0.0 {
            (between_variance / total_variance).clamp(0.0, 1.0)
        } else {
            0.0
        };
        BimodalityScore {
            score,
            separation: (upper_mean - lower_mean).abs() / total_sigma,
            split: sorted[split_index],
            lower_mean,
            upper_mean,
            lower_count: lower.len(),
            upper_count: upper.len(),
            is_bimodal: score >= params.min_score
                && (upper_mean - lower_mean).abs() / total_sigma >= params.min_separation
                && lower.len().min(upper.len()) as f64 / n >= params.min_cluster_fraction,
        }
    }

    /// Deterministic well-separated two-cluster sample.
    fn two_clusters(lower_n: usize, upper_n: usize, gap: f64) -> Vec<f64> {
        let mut out = Vec::with_capacity(lower_n + upper_n);
        for i in 0..lower_n {
            out.push((i % 7) as f64 * 0.01);
        }
        for i in 0..upper_n {
            out.push(gap + (i % 5) as f64 * 0.01);
        }
        out
    }

    /// On separated populations the linear version must agree with the oracle.
    ///
    /// This is the case the detector exists for, and the case where the histogram
    /// envelope is irrelevant because the optimum sits far from any bin boundary.
    #[test]
    fn bd_16g_2_11_linear_bimodality_matches_the_sorting_oracle() {
        let params = BimodalityParams::default();
        for (lower_n, upper_n, gap) in [(50, 50, 5.0), (20, 80, 3.0), (80, 20, 9.0), (2, 2, 4.0)] {
            let values = two_clusters(lower_n, upper_n, gap);
            let fast = bimodality(&values, params).expect("finite");
            let oracle = bimodality_sorting_oracle(&values, params);
            assert_eq!(
                fast.is_bimodal, oracle.is_bimodal,
                "verdict must agree for {lower_n}/{upper_n} gap {gap}"
            );
            assert_eq!(fast.lower_count, oracle.lower_count, "lower_count");
            assert_eq!(fast.upper_count, oracle.upper_count, "upper_count");
            assert!(
                (fast.lower_mean - oracle.lower_mean).abs() < 1e-9,
                "class means are computed from true sums and must be exact"
            );
            assert!(
                (fast.upper_mean - oracle.upper_mean).abs() < 1e-9,
                "upper_mean"
            );
            assert!((fast.score - oracle.score).abs() < 1e-9, "score");
        }
    }

    /// Input ORDER must not change the answer, including adversarial orders.
    ///
    /// Sorted, reverse and duplicate-heavy inputs are exactly the shapes that make a
    /// comparison-based implementation look linear on average while being O(n log n),
    /// so they are also the shapes most worth pinning behaviourally.
    #[test]
    fn bd_16g_2_11_bimodality_is_invariant_to_adversarial_input_order() {
        let params = BimodalityParams::default();
        let base = two_clusters(40, 60, 6.0);
        let reference = bimodality(&base, params).expect("finite");

        let mut ascending = base.clone();
        ascending.sort_unstable_by(f64::total_cmp);
        let mut descending = ascending.clone();
        descending.reverse();
        let mut interleaved = Vec::with_capacity(base.len());
        let (lo, hi) = ascending.split_at(base.len() / 2);
        for i in 0..lo.len().max(hi.len()) {
            if let Some(v) = lo.get(i) {
                interleaved.push(*v);
            }
            if let Some(v) = hi.get(i) {
                interleaved.push(*v);
            }
        }

        for (name, permuted) in [
            ("ascending", ascending),
            ("descending", descending),
            ("interleaved", interleaved),
        ] {
            let got = bimodality(&permuted, params).expect("finite");
            assert_eq!(got.is_bimodal, reference.is_bimodal, "{name}: verdict");
            assert_eq!(
                got.lower_count, reference.lower_count,
                "{name}: lower_count"
            );
            assert!((got.score - reference.score).abs() < 1e-9, "{name}: score");
            assert!((got.split - reference.split).abs() < 1e-9, "{name}: split");
        }
    }

    /// Degenerate inputs must not depend on the search finding anything.
    #[test]
    fn bd_16g_2_11_bimodality_handles_degenerate_inputs() {
        let params = BimodalityParams::default();

        let empty = bimodality(&[], params).expect("empty is not an error");
        assert!(!empty.is_bimodal);
        assert_eq!(empty.lower_count, 0);

        let single = bimodality(&[4.25], params).expect("single");
        assert!(!single.is_bimodal);
        assert_eq!(single.lower_count, 1);
        assert_eq!(single.upper_count, 0);

        let identical = bimodality(&[2.5; 64], params).expect("identical");
        assert!(
            !identical.is_bimodal,
            "no split exists in a constant sample"
        );
        assert_eq!(identical.lower_count, 64);
        assert_eq!(identical.upper_count, 0);
        assert!((identical.split - 2.5).abs() < 1e-12);

        assert!(matches!(
            bimodality(&[1.0, f64::NAN], params),
            Err(DetectError::NonFinite { index: 1 })
        ));
        assert!(matches!(
            bimodality(&[f64::INFINITY, 1.0], params),
            Err(DetectError::NonFinite { index: 0 })
        ));
    }

    /// The reported split must be an OBSERVED value and must actually separate.
    ///
    /// Reporting a bin edge instead would be a silent semantic change: callers use
    /// `split` as a threshold against real data, and an edge can sit where no sample
    /// lies.
    #[test]
    fn bd_16g_2_11_reported_split_is_an_observed_value_of_the_upper_class() {
        let params = BimodalityParams::default();
        let values = two_clusters(30, 30, 7.0);
        let got = bimodality(&values, params).expect("finite");
        assert!(
            values.iter().any(|v| (v - got.split).abs() < 1e-12),
            "split must be a value that actually occurs in the sample"
        );
        let below = values.iter().filter(|v| **v < got.split).count();
        assert_eq!(
            below, got.lower_count,
            "everything below the split is exactly the lower class"
        );
    }

    /// Scale invariance: a positive affine rescale must not change the verdict.
    #[test]
    fn bd_16g_2_11_bimodality_is_invariant_under_positive_affine_rescaling() {
        let params = BimodalityParams::default();
        let values = two_clusters(35, 45, 4.0);
        let reference = bimodality(&values, params).expect("finite");
        for (scale, shift) in [(2.0, 0.0), (0.5, 100.0), (10.0, -50.0)] {
            let scaled: Vec<f64> = values.iter().map(|v| v * scale + shift).collect();
            let got = bimodality(&scaled, params).expect("finite");
            assert_eq!(
                got.is_bimodal, reference.is_bimodal,
                "verdict must survive scale {scale} shift {shift}"
            );
            assert!(
                (got.score - reference.score).abs() < 1e-9,
                "score is a variance ratio and must be scale-free"
            );
            assert!(
                (got.separation - reference.separation).abs() < 1e-9,
                "separation is sigma-normalised and must be scale-free"
            );
        }
    }

    /// Large n must not grow working memory, and must still agree with the oracle.
    ///
    /// The bound this pins is the one the bead cares about: bins are fixed, so the
    /// histogram cost is identical at n = 100 and n = 200_000.
    #[test]
    fn bd_16g_2_11_large_sample_stays_bounded_and_correct() {
        let params = BimodalityParams::default();
        let values = two_clusters(100_000, 100_000, 8.0);
        let fast = bimodality(&values, params).expect("finite");
        let oracle = bimodality_sorting_oracle(&values, params);
        assert!(fast.is_bimodal, "a clean 8-sigma split must register");
        assert_eq!(fast.is_bimodal, oracle.is_bimodal);
        assert_eq!(fast.lower_count, oracle.lower_count);
        assert!((fast.score - oracle.score).abs() < 1e-9);
        assert_eq!(
            BIMODALITY_BINS, 1024,
            "working memory is O(BIMODALITY_BINS) and must stay independent of n"
        );
    }
}
