//! Exploratory statistical assessment of narrative events (bd-2z0.11.6).
//!
//! The detector selects a boundary from a metric series. This module describes the mean shift
//! around that boundary with a circular moving-block bootstrap CI and a whole-block permutation
//! test. Contiguous ticks are not independent replicates. Both inferential results use the
//! declared block length; the after-window mean CI is additional context, not the only place
//! serial dependence is considered.
//!
//! These are approximate time-series methods. Windows must be approximately stationary within
//! each segment; the block length must capture the relevant short-range dependence; the test
//! assumes exchangeable equal-length blocks under its null. Dependence across block boundaries
//! and selection of a boundary from the same observations are not corrected.
//!
//! Benjamini-Hochberg is applied to the reported p-values as an explicit thresholding rule.
//! Its usual FDR guarantee requires valid p-values and the relevant hypothesis-dependence
//! conditions. It does not repair invalid exchangeability or data-dependent event selection.
//! Consequently reports on detected events are exploratory, not confirmatory discoveries.
//!
//! Inputs, resampling controls, methods and seeds are explicit. No database, clock, or
//! simulation state is read here; reports bind these computations to their observed evidence.

// The Benjamini-Hochberg rank threshold `(k / m) · q` casts small integer counts to f64 — sizes
// exact in f64 for any realistic run. Allowed module-wide, as in `stats`, rather than scattering
// identical pragmas through the correction.
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

use crate::stats::{
    self, ConfidenceInterval, EffectSizeEstimator, StatsError,
    block_permutation_test_mean_difference, cliffs_delta, cohens_d,
    moving_block_bootstrap_mean_ci, moving_block_bootstrap_mean_difference_ci,
};

/// Versioned interval method recorded by persisted reports.
pub const SHIFT_CI_METHOD: &str = "circular_moving_block_bootstrap_percentile_v1";
/// Versioned significance method recorded by persisted reports.
pub const SIGNIFICANCE_METHOD: &str = "length_stratified_whole_block_permutation_v1";

/// Exact controls and interpretation of a reported time-series assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificationMethod {
    /// Versioned confidence-interval algorithm.
    pub confidence_interval: String,
    /// Versioned p-value algorithm.
    pub p_value: String,
    /// Contiguous observations per full resampling block.
    pub block_len: usize,
    /// Bootstrap draws used for the percentile interval.
    pub resamples: usize,
    /// Whole-block label permutations used for the p-value.
    pub permutations: usize,
    /// Nominal interval coverage under the method's assumptions.
    pub confidence: f64,
    /// Target used by the Benjamini-Hochberg thresholding rule.
    pub fdr: f64,
    /// Root seed; individual estimators derive their documented streams from it.
    pub seed: u64,
    /// Declares the data-selected, exploratory scope of detector reports.
    pub interpretation: String,
    /// Assumptions and limitations that the reported thresholds do not establish.
    pub assumptions: String,
}

/// Parameters governing a certification pass.
///
/// Every knob is explicit so a report can log the exact configuration it certified under — a
/// p-value means nothing without the window and resample counts that produced it.
#[derive(Debug, Clone, Copy)]
pub struct CertificationParams {
    /// How many samples on each side of the event to compare. The before window is
    /// `[event - window, event)`, the after window is `[event, event + window)`.
    pub window: usize,
    /// Bootstrap resamples for the shift's confidence interval.
    pub n_resamples: usize,
    /// Whole-block label permutations for the significance test.
    pub n_permutations: usize,
    /// Confidence level for the shift CI, e.g. `0.95`.
    pub confidence: f64,
    /// Target for Benjamini-Hochberg thresholding across the tested events.
    /// This alone does not establish FDR control for data-selected boundaries.
    pub fdr: f64,
    /// Contiguous block length for both the shift CI and significance test.
    /// Each window must hold at least two complete blocks; the declared value is never clamped.
    /// Choosing an adequate correlation length remains the caller's responsibility.
    pub block_len: usize,
    /// Seed for all resampling. Fixed ⇒ the certification is reproducible.
    pub seed: u64,
}

impl Default for CertificationParams {
    fn default() -> Self {
        Self {
            window: 30,
            n_resamples: 2000,
            n_permutations: 2000,
            confidence: 0.95,
            fdr: 0.05,
            block_len: 5,
            seed: 0x5EED,
        }
    }
}

impl CertificationParams {
    /// Validate statistical controls before attempting or skipping any event windows.
    ///
    /// Invalid controls are errors even for an empty candidate list. The minimum of two
    /// complete blocks per side prevents a full-window circular resample from being reported
    /// as a zero-uncertainty estimate; it does not certify the block-length choice.
    ///
    /// # Errors
    ///
    /// Refuses windows smaller than two observations, unsupported block lengths, zero
    /// draw counts, and invalid confidence or false-discovery-rate targets.
    pub fn validate(&self) -> Result<(), StatsError> {
        if self.window < 2 {
            return Err(StatsError::EmptySample {
                what: "certify.window (needs at least two observations per side)",
            });
        }
        stats::validate_time_series_block_len(self.window, self.block_len)?;
        if self.n_resamples == 0 || self.n_permutations == 0 {
            return Err(StatsError::ZeroResamples);
        }
        if !(0.0 < self.confidence && self.confidence < 1.0) {
            return Err(StatsError::InvalidConfidence {
                level: self.confidence,
            });
        }
        if !(self.fdr.is_finite() && 0.0 < self.fdr && self.fdr <= 1.0) {
            return Err(StatsError::InvalidFalseDiscoveryRate { rate: self.fdr });
        }
        Ok(())
    }

    /// Describe the configured estimators without claiming their assumptions were verified.
    ///
    /// Call [`Self::validate`] before publishing numerical results. The root seed drives
    /// the shift CI directly, the contextual after mean uses `seed ^ 0xA5A5`, and the
    /// significance test uses `seed ^ 0x1234`.
    #[must_use]
    pub fn method(&self) -> CertificationMethod {
        CertificationMethod {
            confidence_interval: SHIFT_CI_METHOD.to_owned(),
            p_value: SIGNIFICANCE_METHOD.to_owned(),
            block_len: self.block_len,
            resamples: self.n_resamples,
            permutations: self.n_permutations,
            confidence: self.confidence,
            fdr: self.fdr,
            seed: self.seed,
            interpretation: "exploratory_selected_events".to_owned(),
            assumptions: "Approximately stationary short-range dependence within each window; \
                exchangeable equal-length blocks under the null. Dependence across block \
                boundaries and event selection from the same data are not corrected. \
                Benjamini-Hochberg flags do not establish unconditional false-discovery control."
                .to_owned(),
        }
    }
}

/// The exploratory assessment of a single event.
#[derive(Debug, Clone, PartialEq)]
pub struct EventCertification {
    /// Index into the metric series where the event fired.
    pub event_index: usize,
    /// Circular moving-block percentile interval on the mean shift (`after - before`).
    pub shift_ci: ConfidenceInterval,
    /// Autocorrelation-aware CI on the after-window mean alone, for context.
    pub after_mean_ci: ConfidenceInterval,
    /// Two-sided whole-block permutation p-value under block exchangeability.
    /// It does not correct for selecting the event from these same observations.
    pub p_value: f64,
    /// Parametric effect size (standardized mean difference).
    pub cohens_d: f64,
    /// Which estimator produced [`Self::cohens_d`] — corrected for small-sample bias, or not.
    ///
    /// Reported because `n_before` / `n_after` are necessary but NOT sufficient to interpret the
    /// effect size. They tell a reader whether the small-sample regime applies; only this tells
    /// them whether the correction was already applied. A reader who sees `n_before = 6` and
    /// discounts a number that had already been discounted is as wrong as one who does not
    /// discount at all (`bd-k3f3`).
    pub effect_estimator: EffectSizeEstimator,
    /// Distribution-free effect size in `[-1, 1]`.
    pub cliffs_delta: f64,
    /// Number of samples in the before window that was tested.
    pub n_before: usize,
    /// Number of samples in the after window that was tested.
    pub n_after: usize,
    /// `p_value <= fdr` before multiple-testing thresholding.
    /// This is an exploratory flag, not independent confirmation of a selected event.
    pub significant_uncorrected: bool,
    /// Survives the Benjamini-Hochberg thresholding rule at the configured target.
    /// FDR control additionally requires valid p-values and hypothesis-dependence assumptions;
    /// this flag does not establish them. A single event has the uncorrected flag.
    pub significant_fdr: bool,
}

/// Certify one event in isolation.
///
/// With a single event there is no multiple-testing to correct, so `significant_fdr` equals
/// `significant_uncorrected`. Use [`certify_run`] for the realistic case of many events, where
/// the same thresholding rule is applied across all tested events.
pub fn certify_event(
    series: &[f64],
    event_index: usize,
    params: &CertificationParams,
) -> Result<EventCertification, StatsError> {
    params.validate()?;
    let (before, after) = windows(series, event_index, params.window)?;

    let shift_ci = moving_block_bootstrap_mean_difference_ci(
        before,
        after,
        params.block_len,
        params.n_resamples,
        params.confidence,
        params.seed,
    )?;
    let after_mean_ci = moving_block_bootstrap_mean_ci(
        after,
        params.block_len,
        params.n_resamples,
        params.confidence,
        params.seed ^ 0xA5A5,
    )?;
    let test = block_permutation_test_mean_difference(
        before,
        after,
        params.block_len,
        params.n_permutations,
        params.seed ^ 0x1234,
    )?;
    let d = cohens_d(before, after)?;
    let delta = cliffs_delta(before, after)?;

    let uncorrected = test.p_value <= params.fdr;
    Ok(EventCertification {
        event_index,
        shift_ci,
        after_mean_ci,
        p_value: test.p_value,
        cohens_d: d,
        // Declared, not switched. Certification keeps reporting the UNCORRECTED estimator so
        // this change alters no number anyone has already published; the correction is now
        // available as `stats::hedges_g` and adopting it is a deliberate one-line decision
        // rather than a silent shift in reported effect sizes (`bd-k3f3`).
        effect_estimator: EffectSizeEstimator::CohensD,
        cliffs_delta: delta,
        n_before: before.len(),
        n_after: after.len(),
        significant_uncorrected: uncorrected,
        // Nothing to correct against for a lone event.
        significant_fdr: uncorrected,
    })
}

/// Certify every event in a run, then apply Benjamini-Hochberg across all of them.
///
/// Each event receives a block-based assessment; then the vector of p-values is passed through
/// [`benjamini_hochberg`] at `params.fdr`. This applies the thresholding rule, without claiming
/// that overlapping windows are independent or that data-selected events have valid
/// confirmatory p-values. The rule cannot repair either assumption.
///
/// Events whose windows fall outside the series (too close to the start or end) are DROPPED with
/// their index recorded in the returned `skipped` list, never silently ignored: an event the
/// certifier could not evaluate must not read as an event it evaluated and cleared.
pub fn certify_run(
    series: &[f64],
    event_indices: &[usize],
    params: &CertificationParams,
) -> Result<RunCertification, StatsError> {
    params.validate()?;
    let mut certifications = Vec::new();
    let mut skipped = Vec::new();
    for &event_index in event_indices {
        match certify_event(series, event_index, params) {
            Ok(certification) => certifications.push(certification),
            // An out-of-range window is the one "error" that is a property of the event's
            // position, not of bad data — record it and carry on. Any other error is real and
            // propagates.
            Err(StatsError::EmptySample { .. }) => skipped.push(event_index),
            Err(other) => return Err(other),
        }
    }

    let p_values: Vec<f64> = certifications.iter().map(|c| c.p_value).collect();
    let rejected = benjamini_hochberg(&p_values, params.fdr);
    for (certification, &is_rejected) in certifications.iter_mut().zip(&rejected) {
        certification.significant_fdr = is_rejected;
    }

    let discoveries = certifications.iter().filter(|c| c.significant_fdr).count();
    Ok(RunCertification {
        certifications,
        skipped,
        target_fdr: params.fdr,
        discoveries,
    })
}

/// The certification of a whole run's events.
#[derive(Debug, Clone, PartialEq)]
pub struct RunCertification {
    /// One entry per event whose window fit inside the series, in input order.
    pub certifications: Vec<EventCertification>,
    /// Indices of events dropped because their window fell outside the series.
    pub skipped: Vec<usize>,
    /// The false-discovery rate the Benjamini-Hochberg pass targeted.
    pub target_fdr: f64,
    /// How many events survived the exploratory Benjamini-Hochberg thresholding rule.
    pub discoveries: usize,
}

/// Benjamini-Hochberg step-up procedure for controlling the false discovery rate.
///
/// Given m p-values and a target FDR `q`, sort them ascending `p_(1) ≤ … ≤ p_(m)`, find the
/// largest `k` with `p_(k) ≤ (k / m) · q`, and reject every hypothesis with `p ≤ p_(k)`. The
/// returned vector is in the SAME ORDER as the input, so the caller does not have to track the
/// sort permutation.
///
/// It is a pure, standalone function specifically so it can be tested against hand-computed
/// examples: the correction is the part most likely to be subtly wrong, and a wrong correction
/// silently reintroduces the false positives it was meant to remove.
#[must_use]
pub fn benjamini_hochberg(p_values: &[f64], q: f64) -> Vec<bool> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }

    // Sort indices by p-value ascending.
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]));

    // Largest k (1-based) with p_(k) ≤ (k/m)·q.
    let mut largest_k = 0usize;
    for (rank_zero_based, &idx) in order.iter().enumerate() {
        let k = rank_zero_based + 1;
        let threshold = (k as f64 / m as f64) * q;
        if p_values[idx] <= threshold {
            largest_k = k;
        }
    }

    // Reject the largest_k smallest p-values.
    let mut rejected = vec![false; m];
    for &idx in order.iter().take(largest_k) {
        rejected[idx] = true;
    }
    rejected
}

// --- internal --------------------------------------------------------------------------------

/// Extract the before/after windows around an event, or `EmptySample` if either falls outside the
/// series. The `EmptySample` error is the signal `certify_run` uses to *skip* an event whose
/// window does not fit, rather than fabricate a verdict from a truncated window.
fn windows(
    series: &[f64],
    event_index: usize,
    window: usize,
) -> Result<(&[f64], &[f64]), StatsError> {
    if window == 0 {
        return Err(StatsError::EmptySample {
            what: "certify.window",
        });
    }
    let Some(end) = event_index.checked_add(window) else {
        return Err(StatsError::EmptySample {
            what: "certify.out_of_range",
        });
    };
    if event_index < window || end > series.len() {
        return Err(StatsError::EmptySample {
            what: "certify.out_of_range",
        });
    }
    let before = &series[event_index - window..event_index];
    let after = &series[event_index..end];
    Ok((before, after))
}

/// The mean shift a certification measured, `after - before`.
///
/// Re-exported through the CI's point estimate but named here for readability at call sites.
#[must_use]
pub const fn shift(certification: &EventCertification) -> f64 {
    certification.shift_ci.point
}

/// Widen an `f32` metric series to the `f64` the statistics operate on.
///
/// The detector records metrics as `f32`; the offline certification works in `f64`. Kept here so
/// the widening happens in exactly one place.
#[must_use]
pub fn widen(series: &[f32]) -> Vec<f64> {
    series.iter().map(|&value| f64::from(value)).collect()
}

/// Re-export so callers can name the statistics error without a second `use`.
pub use stats::StatsError as CertificationError;

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic normal draws for building fixtures with known structure. Self-contained
    /// (its own inline `SplitMix64`) rather than reaching into the stats module's private RNG —
    /// the test's generating process should not depend on another module's internals.
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
            // 53 high bits → a uniform in (0, 1].
            let value = self.bits() >> 11;
            (value as f64 + 1.0) / (9_007_199_254_740_992.0 + 1.0)
        }
        fn normal(&mut self, mean: f64, sd: f64) -> f64 {
            let u1 = self.unit();
            let u2 = self.unit();
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            sd.mul_add(z, mean)
        }
    }

    /// Build a series that is flat at `base` for `n` samples, then steps by `step` for another
    /// `n`, with gaussian noise. The event is exactly at index `n`.
    fn stepped_series(n: usize, base: f64, step: f64, sd: f64, seed: u64) -> Vec<f64> {
        let mut draws = Normal::new(seed);
        let mut series = Vec::with_capacity(2 * n);
        for _ in 0..n {
            series.push(draws.normal(base, sd));
        }
        for _ in 0..n {
            series.push(draws.normal(base + step, sd));
        }
        series
    }

    #[test]
    fn benjamini_hochberg_matches_a_hand_computed_example() {
        // The canonical worked example: m=5 p-values, q=0.05. Sorted: 0.005, 0.011, 0.02, 0.04,
        // 0.13. Thresholds k/m·q: 0.01, 0.02, 0.03, 0.04, 0.05. Compare:
        //   0.005 ≤ 0.01 ✓ (k=1)
        //   0.011 ≤ 0.02 ✓ (k=2)
        //   0.02  ≤ 0.03 ✓ (k=3)
        //   0.04  ≤ 0.04 ✓ (k=4)
        //   0.13  ≤ 0.05 ✗
        // Largest k with success is 4 ⇒ reject the four smallest.
        let p = [0.04, 0.13, 0.005, 0.02, 0.011];
        let rejected = benjamini_hochberg(&p, 0.05);
        assert_eq!(rejected, vec![true, false, true, true, true]);
    }

    #[test]
    fn benjamini_hochberg_is_step_up_not_naive_thresholding() {
        // The step-up property that distinguishes BH from a naive "reject each p ≤ its own
        // threshold" rule: a p-value can be REJECTED even though it individually exceeds its own
        // k/m·q threshold, because a LARGER-ranked p-value cleared its threshold and BH rejects
        // everything up to the largest passing rank.
        // m=3, q=0.05. Sorted: 0.001, 0.04, 0.045. Thresholds: 0.0167, 0.0333, 0.05.
        //   0.001 ≤ 0.0167 ✓
        //   0.04  ≤ 0.0333 ✗   (naive would NOT reject this)
        //   0.045 ≤ 0.05   ✓   (k=3) ⇒ BH rejects ALL THREE, including the 0.04 that failed naively
        let p = [0.001, 0.04, 0.045];
        let rejected = benjamini_hochberg(&p, 0.05);
        assert_eq!(
            rejected,
            vec![true, true, true],
            "BH must reject the middle p-value via step-up even though it exceeds its own threshold"
        );
    }

    #[test]
    fn benjamini_hochberg_rejects_nothing_when_all_null() {
        let p = [0.4, 0.6, 0.8, 0.55, 0.9];
        assert_eq!(benjamini_hochberg(&p, 0.05), vec![false; 5]);
        assert_eq!(benjamini_hochberg(&[], 0.05), [] as [bool; 0]);
    }

    #[test]
    fn a_real_step_change_is_certified_significant() {
        // A clear 4-SD step at the event must be flagged, with a positive shift, a small p-value,
        // and a large effect size. A certifier blind to a step this obvious certifies nothing.
        let series = stepped_series(80, 10.0, 4.0, 1.0, 42);
        let params = CertificationParams {
            window: 40,
            ..CertificationParams::default()
        };
        let c = certify_event(&series, 80, &params).unwrap();

        assert!(
            c.significant_fdr,
            "a 4-SD step was not certified (p={:.4})",
            c.p_value
        );
        assert!(
            c.shift_ci.point > 0.0,
            "the shift should be positive; got {}",
            c.shift_ci.point
        );
        assert!(
            c.shift_ci.covers(4.0),
            "the CI should cover the true 4.0 shift: {:?}",
            c.shift_ci
        );
        assert!(
            c.cohens_d > 2.0,
            "Cohen's d for a 4-SD shift should be large; got {}",
            c.cohens_d
        );
        assert!(
            c.cliffs_delta > 0.9,
            "Cliff's delta should be near +1; got {}",
            c.cliffs_delta
        );
    }

    #[test]
    fn a_flat_series_is_not_certified_as_an_event() {
        // No real change at the "event": before and after are the same distribution. It must NOT
        // be certified. This is the false-positive the whole module exists to suppress.
        let series = stepped_series(80, 5.0, 0.0, 1.5, 7);
        let params = CertificationParams {
            window: 40,
            ..CertificationParams::default()
        };
        let c = certify_event(&series, 80, &params).unwrap();
        assert!(
            !c.significant_fdr,
            "a flat series was certified as a real event (p={:.4}); this is a false positive",
            c.p_value
        );
    }

    #[test]
    fn fdr_control_suppresses_the_flood_of_null_events() {
        // THE HEADLINE, and the reason bd-16g.2.3's false-positive budget needs this. Build a
        // long flat series and mark MANY events on it — all null, none real. Testing each at
        // α=0.05 would flag ~5% of them by chance; Benjamini-Hochberg must drive the number of
        // FDR-significant "discoveries" close to zero.
        let series = stepped_series(2000, 5.0, 0.0, 1.0, 100); // 4000 samples, all null
        let window = 40;
        // 60 candidate events spread across the interior, none corresponding to a real change.
        let events: Vec<usize> = (0..60).map(|i| window + 20 + i * 60).collect();
        let params = CertificationParams {
            window,
            ..CertificationParams::default()
        };

        let run = certify_run(&series, &events, &params).unwrap();
        let uncorrected = run
            .certifications
            .iter()
            .filter(|c| c.significant_uncorrected)
            .count();

        println!(
            "null run: {} events, {} uncorrected significant (α=0.05), {} FDR discoveries",
            run.certifications.len(),
            uncorrected,
            run.discoveries
        );
        // With 60 null tests at α=0.05, we expect ~3 uncorrected false positives — and FDR control
        // should reduce the discoveries well below that. Allow a small margin for Monte-Carlo.
        assert!(
            run.discoveries <= 1,
            "Benjamini-Hochberg let {} null events through as discoveries; the false-positive \
             budget is not being protected",
            run.discoveries
        );
    }

    #[test]
    fn fdr_control_keeps_the_real_events_and_drops_the_null_ones() {
        // The other side of the headline: FDR must not be so aggressive that it discards genuine
        // events, and not so lax that it keeps null ones. Build a series whose level steps by
        // +5 SD at ticks n, 3n and 5n, giving CONSTANT regions [n,3n) and [3n,5n). Real events sit
        // on the boundaries; genuine null events sit deep inside a constant region so their whole
        // ±n window is flat.
        let n = 60;
        let sd = 1.0;
        let mut draws = Normal::new(2024);
        let mut series = Vec::new();
        let mut level = 10.0;
        let real_events = [n, 3 * n, 5 * n];
        for seg in 0..6 {
            if seg == 1 || seg == 3 || seg == 5 {
                level += 5.0;
            }
            for _ in 0..n {
                series.push(draws.normal(level, sd));
            }
        }
        // Null candidates: 2n has window [n,3n) entirely at one level; 4n has window [3n,5n)
        // entirely at the next. Neither straddles a real step, so both are genuine nulls.
        let null_events = [2 * n, 4 * n];
        let events = vec![n, 2 * n, 3 * n, 4 * n, 5 * n];
        let params = CertificationParams {
            window: n,
            ..CertificationParams::default()
        };

        let run = certify_run(&series, &events, &params).unwrap();
        for &real in &real_events {
            let c = run
                .certifications
                .iter()
                .find(|c| c.event_index == real)
                .unwrap();
            assert!(
                c.significant_fdr,
                "a real +5-SD event at index {real} was suppressed by FDR (p={:.4})",
                c.p_value
            );
        }
        for &null in &null_events {
            let c = run
                .certifications
                .iter()
                .find(|c| c.event_index == null)
                .unwrap();
            assert!(
                !c.significant_fdr,
                "a genuine null event at index {null} (flat window) was certified as real \
                 (p={:.4}); FDR is passing noise",
                c.p_value
            );
        }
        println!(
            "mixed run: {} discoveries out of {} events",
            run.discoveries,
            events.len()
        );
        assert_eq!(
            run.discoveries, 3,
            "exactly the three real events should survive FDR; got {} discoveries",
            run.discoveries
        );
    }

    #[test]
    fn events_too_close_to_the_edge_are_skipped_not_faked() {
        // An event whose window runs off the end of the series cannot be certified. It must be
        // reported as skipped, never certified from a truncated window that would silently bias
        // the result.
        let series = stepped_series(50, 5.0, 3.0, 1.0, 9);
        let params = CertificationParams {
            window: 40,
            ..CertificationParams::default()
        };
        // Index 10 has only 10 samples before it (< window 40): out of range.
        let run = certify_run(&series, &[10, 50], &params).unwrap();
        assert!(
            run.skipped.contains(&10),
            "the edge event should be skipped"
        );
        assert!(
            run.certifications.iter().any(|c| c.event_index == 50),
            "the interior event should still be certified"
        );
    }

    #[test]
    fn the_certification_is_reproducible() {
        let series = stepped_series(80, 10.0, 3.0, 1.0, 55);
        let params = CertificationParams {
            window: 40,
            ..CertificationParams::default()
        };
        let a = certify_event(&series, 80, &params).unwrap();
        let b = certify_event(&series, 80, &params).unwrap();
        assert_eq!(
            a, b,
            "same series, index and params must give a bit-identical certification"
        );
    }

    #[test]
    fn reported_shift_and_significance_use_the_declared_blocks() {
        let before: Vec<f64> = [0.0, 2.0]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, 16))
            .collect();
        let after: Vec<f64> = [1.0, 3.0]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, 16))
            .collect();
        let series: Vec<f64> = before.iter().chain(&after).copied().collect();
        let params = CertificationParams {
            window: before.len(),
            block_len: 16,
            n_resamples: 256,
            n_permutations: 256,
            seed: 42,
            ..CertificationParams::default()
        };
        let blocked = certify_event(&series, before.len(), &params).unwrap();
        let independent = certify_event(
            &series,
            before.len(),
            &CertificationParams {
                block_len: 1,
                ..params
            },
        )
        .unwrap();

        assert_eq!(
            blocked.shift_ci,
            stats::moving_block_bootstrap_mean_difference_ci(
                &before,
                &after,
                params.block_len,
                params.n_resamples,
                params.confidence,
                params.seed,
            )
            .unwrap()
        );
        assert_eq!(
            blocked.p_value,
            stats::block_permutation_test_mean_difference(
                &before,
                &after,
                params.block_len,
                params.n_permutations,
                params.seed ^ 0x1234,
            )
            .unwrap()
            .p_value
        );
        assert!(
            independent.significant_uncorrected,
            "the IID control must expose the pseudoreplication defect"
        );
        assert!(
            !blocked.significant_uncorrected,
            "four correlated plateaus are insufficient independent evidence for this shift"
        );
        assert_ne!(blocked.shift_ci, independent.shift_ci);
        assert_ne!(blocked.p_value, independent.p_value);
        let run = certify_run(&series, &[before.len()], &params).unwrap();
        assert_eq!(run.discoveries, 0);
    }

    #[test]
    fn invalid_controls_refuse_empty_and_out_of_range_event_sets() {
        let defaults = CertificationParams::default();
        let invalid = [
            CertificationParams {
                window: 0,
                ..defaults
            },
            CertificationParams {
                window: 1,
                block_len: 1,
                ..defaults
            },
            CertificationParams {
                block_len: 0,
                ..defaults
            },
            CertificationParams {
                block_len: defaults.window,
                ..defaults
            },
            CertificationParams {
                block_len: 16,
                ..defaults
            },
            CertificationParams {
                n_resamples: 0,
                ..defaults
            },
            CertificationParams {
                n_permutations: 0,
                ..defaults
            },
            CertificationParams {
                confidence: 0.0,
                ..defaults
            },
            CertificationParams {
                confidence: 1.0,
                ..defaults
            },
            CertificationParams {
                confidence: f64::NAN,
                ..defaults
            },
            CertificationParams {
                fdr: 0.0,
                ..defaults
            },
            CertificationParams {
                fdr: -0.1,
                ..defaults
            },
            CertificationParams {
                fdr: 1.1,
                ..defaults
            },
            CertificationParams {
                fdr: f64::INFINITY,
                ..defaults
            },
            CertificationParams {
                fdr: f64::NAN,
                ..defaults
            },
        ];
        for params in invalid {
            assert!(params.validate().is_err());
            assert!(certify_run(&[], &[], &params).is_err());
            assert!(certify_run(&[1.0; 8], &[usize::MAX], &params).is_err());
            assert!(certify_event(&[1.0; 8], usize::MAX, &params).is_err());
        }
    }

    #[test]
    fn overflowing_event_windows_are_skipped_without_partial_evidence() {
        let params = CertificationParams {
            window: 2,
            block_len: 1,
            n_resamples: 32,
            n_permutations: 32,
            ..CertificationParams::default()
        };
        let run = certify_run(&[1.0; 8], &[usize::MAX, 0, 4], &params).unwrap();
        assert_eq!(run.skipped, vec![usize::MAX, 0]);
        assert_eq!(run.certifications.len(), 1);
        assert_eq!(run.certifications[0].event_index, 4);
        assert_eq!(run.discoveries, 0);
    }

    #[test]
    fn method_metadata_binds_actual_controls_and_exploratory_interpretation() {
        let params = CertificationParams {
            window: 24,
            block_len: 6,
            n_resamples: 128,
            n_permutations: 256,
            confidence: 0.9,
            fdr: 0.1,
            seed: 123,
        };
        params.validate().unwrap();
        let method = params.method();
        assert_eq!(method.confidence_interval, SHIFT_CI_METHOD);
        assert_eq!(method.p_value, SIGNIFICANCE_METHOD);
        assert_eq!(method.block_len, 6);
        assert_eq!(method.resamples, 128);
        assert_eq!(method.permutations, 256);
        assert_eq!(method.confidence, 0.9);
        assert_eq!(method.fdr, 0.1);
        assert_eq!(method.seed, 123);
        assert_eq!(method.interpretation, "exploratory_selected_events");
        assert!(method.assumptions.contains("exchangeable equal-length blocks"));
        assert!(method.assumptions.contains("event selection"));
        assert_eq!(
            serde_json::from_str::<CertificationMethod>(
                &serde_json::to_string(&method).unwrap()
            )
            .unwrap(),
            method
        );
    }

    #[test]
    fn f32_widening_is_lossless_for_representable_values() {
        let f32_series: Vec<f32> = vec![1.5, 2.25, -3.0, 0.0];
        let wide = widen(&f32_series);
        assert_eq!(wide, vec![1.5, 2.25, -3.0, 0.0]);
    }
}
