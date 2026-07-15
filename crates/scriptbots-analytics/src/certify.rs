//! Statistical certification of narrative events (bd-2z0.11.6, item 1).
//!
//! # The question this answers
//!
//! The detector in `scriptbots-core::detect` fires an `EventRecord` whenever a metric shifts.
//! It is deliberately generous — and its own materiality note records that a purely statistical
//! stream produced **853 events per 10k ticks** on a real run, which is not a story, it is
//! static. Materiality (bd-16g.2.3) filters events by whether a human would *care*. This module
//! answers the orthogonal question: of the events that survive materiality, which ones are
//! statistically **real**, and which are the tail of noise?
//!
//! # Why false-discovery-rate control is the whole point
//!
//! Certifying one event is just [`crate::stats`]: permutation test on the before/after windows,
//! a bootstrap CI on the shift, effect sizes. The trap is that a run flags MANY events, and if
//! you test each at α = 0.05 you *expect* one false positive in every twenty by construction —
//! so a long run manufactures "significant" events out of pure noise no matter how clean each
//! individual test is. That is exactly the false-positive budget bd-16g.2.3 exists to protect,
//! and eyeballed per-event thresholds cannot protect it.
//!
//! The fix is Benjamini-Hochberg: control the *false discovery rate* across the whole set of
//! events at once — the expected fraction of flagged events that are false. It is the principled
//! replacement for a per-event threshold, and [`benjamini_hochberg`] implements it as a pure,
//! separately-tested function so the correction can never quietly diverge from the textbook
//! procedure.
//!
//! # Purity and determinism
//!
//! Everything here is a pure function of a metric series, a set of event positions, and a seed.
//! No database, no clock, no I/O. The DB-reading glue that loads real `EventRecord`s and their
//! metric windows is a thin adapter a report adds on top; the certification logic — the part that
//! is easy to get subtly wrong — is proven here on synthetic series with known ground truth.

// The Benjamini-Hochberg rank threshold `(k / m) · q` casts small integer counts to f64 — sizes
// exact in f64 for any realistic run. Allowed module-wide, as in `stats`, rather than scattering
// identical pragmas through the correction.
#![allow(clippy::cast_precision_loss)]

use crate::stats::{
    self, ConfidenceInterval, StatsError, bootstrap_mean_difference_ci, cliffs_delta, cohens_d,
    moving_block_bootstrap_mean_ci, permutation_test_mean_difference,
};

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
    /// Permutations for the significance test.
    pub n_permutations: usize,
    /// Confidence level for the shift CI, e.g. `0.95`.
    pub confidence: f64,
    /// Target false-discovery rate for the Benjamini-Hochberg pass across a run's events.
    pub fdr: f64,
    /// Moving-block length for the block bootstrap. Metric series are autocorrelated, so a
    /// per-point bootstrap would understate the uncertainty; this should scale with the series'
    /// correlation length.
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

/// The certification verdict for a single event.
#[derive(Debug, Clone, PartialEq)]
pub struct EventCertification {
    /// Index into the metric series where the event fired.
    pub event_index: usize,
    /// Confidence interval on the mean shift (`after - before`).
    pub shift_ci: ConfidenceInterval,
    /// Autocorrelation-aware CI on the after-window mean alone, for context.
    pub after_mean_ci: ConfidenceInterval,
    /// Two-sided permutation p-value against the null that the event changed nothing.
    pub p_value: f64,
    /// Parametric effect size (standardized mean difference).
    pub cohens_d: f64,
    /// Distribution-free effect size in `[-1, 1]`.
    pub cliffs_delta: f64,
    /// Number of samples in the before window that was tested.
    pub n_before: usize,
    /// Number of samples in the after window that was tested.
    pub n_after: usize,
    /// `p_value < alpha` with NO multiple-testing correction. Reported for transparency, but it
    /// is NOT the field to act on across a run — that is what `significant_fdr` is for.
    pub significant_uncorrected: bool,
    /// Survives Benjamini-Hochberg control at the run's target FDR. This is the honest verdict
    /// for an event that is one of many. For a single isolated event it equals the uncorrected
    /// result, because there is nothing to correct against.
    pub significant_fdr: bool,
}

/// Certify one event in isolation.
///
/// With a single event there is no multiple-testing to correct, so `significant_fdr` equals
/// `significant_uncorrected`. Use [`certify_run`] for the realistic case of many events, where
/// the FDR control is the entire value.
pub fn certify_event(
    series: &[f64],
    event_index: usize,
    params: &CertificationParams,
) -> Result<EventCertification, StatsError> {
    let (before, after) = windows(series, event_index, params.window)?;

    let shift_ci = bootstrap_mean_difference_ci(
        before,
        after,
        params.n_resamples,
        params.confidence,
        params.seed,
    )?;
    let after_mean_ci = moving_block_bootstrap_mean_ci(
        after,
        params.block_len.min(after.len()).max(1),
        params.n_resamples,
        params.confidence,
        params.seed ^ 0xA5A5,
    )?;
    let test = permutation_test_mean_difference(
        before,
        after,
        params.n_permutations,
        params.seed ^ 0x1234,
    )?;
    let d = cohens_d(before, after)?;
    let delta = cliffs_delta(before, after)?;

    let uncorrected = test.p_value < params.fdr;
    Ok(EventCertification {
        event_index,
        shift_ci,
        after_mean_ci,
        p_value: test.p_value,
        cohens_d: d,
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
/// This is the realistic path. Each event is tested independently; then the whole vector of
/// p-values is passed through [`benjamini_hochberg`] at `params.fdr`, and `significant_fdr` is set
/// from the corrected decision. An event that would clear an isolated α = 0.05 test can fail here
/// — correctly — because it is competing against every other event flagged in the same run.
///
/// Events whose windows fall outside the series (too close to the start or end) are DROPPED with
/// their index recorded in the returned `skipped` list, never silently ignored: an event the
/// certifier could not evaluate must not read as an event it evaluated and cleared.
pub fn certify_run(
    series: &[f64],
    event_indices: &[usize],
    params: &CertificationParams,
) -> Result<RunCertification, StatsError> {
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
    /// How many events survived FDR control — the count a report should headline, not the raw
    /// number of events the detector fired.
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
    if event_index < window || event_index + window > series.len() {
        return Err(StatsError::EmptySample {
            what: "certify.out_of_range",
        });
    }
    let before = &series[event_index - window..event_index];
    let after = &series[event_index..event_index + window];
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
            mean + sd * z
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
        assert!(benjamini_hochberg(&[], 0.05).is_empty());
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
    fn f32_widening_is_lossless_for_representable_values() {
        let f32_series: Vec<f32> = vec![1.5, 2.25, -3.0, 0.0];
        let wide = widen(&f32_series);
        assert_eq!(wide, vec![1.5, 2.25, -3.0, 0.0]);
    }
}
