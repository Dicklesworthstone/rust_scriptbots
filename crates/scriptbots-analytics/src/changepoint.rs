//! Single change-point detection over a metric series (bd-2z0.11.6).
//!
//! # What this finds, and what it does NOT
//!
//! Given a metric's value series, [`largest_shift`] finds the ONE split point that maximizes the
//! absolute difference in mean between the segment before it and the segment after it. It is a
//! single-change-point finder, not a multi-change-point segmentation: it answers "if this run had
//! one regime shift in this metric, where was it and how big?" — which is exactly what the
//! certification pipeline then tests for reality.
//!
//! It is deliberately separate from the online detector in `scriptbots-core::detect`. That detector
//! runs live, bit-stable, and streaming. This runs offline over a finished series, is O(n) via
//! prefix sums, and feeds [`crate::certify`] so the shift it finds can be certified rather than
//! merely reported. Finding a shift is easy; the hard, valuable part is deciding whether it is
//! real, and that is certification's job.
//!
//! # Purity
//!
//! A pure function of a slice and a minimum segment size. No RNG, no I/O — the certification that
//! consumes its output is where the resampling lives.

/// The most prominent change-point in a series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChangePoint {
    /// The split index: the "before" segment is `series[..index]`, the "after" is `series[index..]`.
    /// Also the position of the first sample that belongs to the new regime.
    pub index: usize,
    /// `mean(after) - mean(before)`. Signed, so its direction is meaningful.
    pub shift: f64,
    /// Mean of the before segment.
    pub before_mean: f64,
    /// Mean of the after segment.
    pub after_mean: f64,
}

/// Find the single change-point that maximizes the absolute mean shift.
///
/// Only splits with at least `min_segment` samples on EACH side are considered — a "shift" measured
/// against one or two points is noise wearing a large number, and certification could not test it
/// anyway. Returns `None` when the series is shorter than `2 * min_segment` (there is no admissible
/// split) or when `min_segment` is zero.
///
/// O(n): one prefix-sum pass, then one scan over admissible split points.
#[must_use]
pub fn largest_shift(series: &[f64], min_segment: usize) -> Option<ChangePoint> {
    let n = series.len();
    if min_segment == 0 || n < 2 * min_segment {
        return None;
    }

    // Prefix sums so every segment mean is O(1). prefix[i] = sum of series[..i].
    let mut prefix = vec![0.0_f64; n + 1];
    for (i, &value) in series.iter().enumerate() {
        prefix[i + 1] = prefix[i] + value;
    }
    let total = prefix[n];

    let mut best: Option<ChangePoint> = None;
    for k in min_segment..=(n - min_segment) {
        #[allow(clippy::cast_precision_loss)]
        let before_mean = prefix[k] / k as f64;
        #[allow(clippy::cast_precision_loss)]
        let after_mean = (total - prefix[k]) / (n - k) as f64;
        let shift = after_mean - before_mean;
        // Strictly-greater keeps the EARLIEST split among ties, which is deterministic and the
        // conventional choice (report the onset, not a later equivalent split).
        if best.is_none_or(|b| shift.abs() > b.shift.abs()) {
            best = Some(ChangePoint {
                index: k,
                shift,
                before_mean,
                after_mean,
            });
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_step_is_located_at_its_boundary() {
        // 10 samples at 0, then 10 at 100. The split must land exactly at index 10, with a shift
        // of +100.
        let mut series = vec![0.0; 10];
        series.extend(std::iter::repeat_n(100.0, 10));
        let cp = largest_shift(&series, 3).expect("a step exists");
        assert_eq!(
            cp.index, 10,
            "the change-point must be at the regime boundary"
        );
        assert!(
            (cp.shift - 100.0).abs() < 1e-9,
            "the shift is the full step height"
        );
        assert!((cp.before_mean - 0.0).abs() < 1e-9);
        assert!((cp.after_mean - 100.0).abs() < 1e-9);
    }

    #[test]
    fn a_downward_step_reports_a_negative_shift() {
        let mut series = vec![50.0; 8];
        series.extend(std::iter::repeat_n(10.0, 8));
        let cp = largest_shift(&series, 2).expect("a step exists");
        assert_eq!(cp.index, 8);
        assert!(
            cp.shift < 0.0,
            "a drop must report a negative shift; got {}",
            cp.shift
        );
        assert!((cp.shift + 40.0).abs() < 1e-9);
    }

    #[test]
    fn a_flat_series_yields_a_negligible_shift() {
        // No regime change: the best split still exists, but its shift is ~0.
        let series = vec![7.0; 20];
        let cp = largest_shift(&series, 5).expect("admissible splits exist");
        assert!(
            cp.shift.abs() < 1e-9,
            "a flat series has no real shift; got {}",
            cp.shift
        );
    }

    #[test]
    fn the_largest_of_two_steps_wins() {
        // Two steps: a small +5 at index 10, a large +50 at index 20. The finder reports the LARGER.
        let mut series = vec![0.0; 10];
        series.extend(std::iter::repeat_n(5.0, 10));
        series.extend(std::iter::repeat_n(55.0, 10));
        let cp = largest_shift(&series, 3).expect("steps exist");
        assert_eq!(
            cp.index, 20,
            "the larger step at index 20 must win over the smaller at 10"
        );
    }

    #[test]
    fn a_series_too_short_for_the_minimum_segment_returns_none() {
        assert_eq!(
            largest_shift(&[1.0, 2.0, 3.0], 2),
            None,
            "n=3 < 2*min_segment=4"
        );
        assert_eq!(
            largest_shift(&[1.0, 2.0], 0),
            None,
            "min_segment 0 is rejected"
        );
        assert!(
            largest_shift(&[1.0, 2.0, 3.0, 4.0], 2).is_some(),
            "n=4 == 2*min_segment is admissible"
        );
    }

    #[test]
    fn the_finder_is_deterministic() {
        let series: Vec<f64> = (0..50).map(|i| if i < 25 { 1.0 } else { 4.0 }).collect();
        assert_eq!(largest_shift(&series, 5), largest_shift(&series, 5));
    }
}
