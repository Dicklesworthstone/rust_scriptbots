//! Deterministic, byte-stable number formatting for narrative event text (bd-16g.2.2).
//!
//! The narrative event stream is a HARD-diffable artifact: `human_text` is templated and must be
//! byte-identical across runs and builds so an offline analyst — and the LLM lab assistant of
//! bd-16g.1, which must READ these events without having authored them — can diff two runs and see
//! exactly which story changed. These primitives are the number formatters those templates share.
//!
//! Rust's `format!` is already locale-independent (always `.`, never a locale's `,` decimal), so
//! the hazard here is not locale drift but INCONSISTENT grouping and precision across call sites:
//! one template writing `1842` and another `1,842` for the same count silently corrupts a diff.
//! Centralizing the formatting here makes every count group thousands identically and every ratio
//! carry the same precision, by construction.
//!
//! The target format is the one bd-16g.2.2 specifies: `"population fell 63% (1,842 -> 681)"` —
//! grouped counts, integer percent magnitude, fixed-precision ratios.
//!
//! # Purity
//!
//! Every function is a pure function of its arguments: no RNG, no I/O, no globals, no locale. The
//! same inputs always produce the same bytes.

// `count()` deliberately rounds an f64 count to the nearest integer and casts it to `u64` for digit
// grouping. Counts are conceptually non-negative integers carried as f64; the saturating cast and
// the round are intentional, not lossy accidents.
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// Format a count (population, agents, spike hits) as a grouped integer: `1842.0 -> "1,842"`.
///
/// Counts are conceptually non-negative integers stored as `f64`; the value is rounded to the
/// nearest integer and its digits are grouped in threes with `,`. Deterministic and
/// locale-independent — the grouping is done by hand, not by a locale-aware formatter, so it can
/// never vary with the host's locale. Non-finite input (never produced for a real count) formats as
/// `"0"` rather than panicking.
#[must_use]
pub fn count(value: f64) -> String {
    let rounded = value.round();
    let negative = rounded < 0.0;
    // Saturating f64->u64 cast (NaN -> 0, +inf -> u64::MAX, negatives -> 0 after abs): total by
    // construction. `to_string` on the u64 yields ASCII digits only, so byte iteration is safe.
    let magnitude = rounded.abs() as u64;
    let digits = magnitude.to_string();
    let bytes = digits.as_bytes();
    let len = bytes.len();

    let mut out = String::with_capacity(len + len / 3 + 1);
    if negative && magnitude != 0 {
        out.push('-');
    }
    for (index, byte) in bytes.iter().enumerate() {
        if index > 0 && (len - index) % 3 == 0 {
            out.push(',');
        }
        out.push(char::from(*byte));
    }
    out
}

/// Format a `before -> after` count transition: `(1842.0, 681.0) -> "1,842 -> 681"`.
///
/// Both endpoints are grouped by [`count`]. This is the exact shape bd-16g.2.2's templates wrap in
/// a verb, e.g. `format!("population fell {} ({})", pct_magnitude(b, a), count_transition(b, a))`.
#[must_use]
pub fn count_transition(before: f64, after: f64) -> String {
    format!("{} -> {}", count(before), count(after))
}

/// Integer magnitude of a percentage change: `(1000.0, 300.0) -> "70"` (a 70% fall).
///
/// The sign is dropped — narrative templates encode direction in the verb ("fell"/"rose"), so the
/// number is always the magnitude. Rounded to a whole percent. When `before` is ~zero the percent
/// is undefined (any change is "infinite %"), so it reports `"0"` and the template must lead with
/// the absolute transition instead.
#[must_use]
pub fn pct_magnitude(before: f64, after: f64) -> String {
    let pct = if before.abs() > f64::EPSILON {
        ((after - before) / before * 100.0).abs()
    } else {
        0.0
    };
    format!("{pct:.0}")
}

/// Fixed-precision ratio formatting, centralized so every 0..1/0..2 quantity in the stream carries
/// the same number of decimals: `(0.994, 2) -> "0.99"`.
///
/// A thin, deliberate wrapper over `format!("{value:.decimals$}")` — its value is that the
/// precision lives in one place, so energy/severity templates cannot drift to different decimal
/// counts. Locale-independent like all of `format!`.
#[must_use]
pub fn fixed(value: f64, decimals: usize) -> String {
    format!("{value:.decimals$}")
}

#[cfg(test)]
mod tests {
    use super::{count, count_transition, fixed, pct_magnitude};

    #[test]
    fn counts_group_thousands_deterministically() {
        assert_eq!(count(0.0), "0");
        assert_eq!(count(12.0), "12");
        assert_eq!(count(100.0), "100");
        assert_eq!(count(999.0), "999");
        assert_eq!(count(1000.0), "1,000");
        assert_eq!(count(1842.0), "1,842");
        assert_eq!(count(12_345.0), "12,345");
        assert_eq!(count(1_234_567.0), "1,234,567");
        assert_eq!(count(1_000_000.0), "1,000,000");
    }

    #[test]
    fn counts_round_to_the_nearest_integer() {
        assert_eq!(count(1841.4), "1,841");
        assert_eq!(count(1841.6), "1,842");
        assert_eq!(count(0.4), "0");
        assert_eq!(count(0.6), "1");
    }

    #[test]
    fn counts_carry_a_sign_only_for_true_negatives() {
        assert_eq!(count(-1234.0), "-1,234");
        assert_eq!(count(-1.0), "-1");
        // -0.0 and values that round to zero are not "negative zero" in the text.
        assert_eq!(count(-0.0), "0");
        assert_eq!(count(-0.4), "0");
    }

    #[test]
    fn non_finite_counts_degrade_to_zero_not_panic() {
        assert_eq!(count(f64::NAN), "0");
    }

    #[test]
    fn transitions_group_both_endpoints() {
        assert_eq!(count_transition(1842.0, 681.0), "1,842 -> 681");
        assert_eq!(count_transition(1000.0, 300.0), "1,000 -> 300");
        assert_eq!(count_transition(23.0, 22.0), "23 -> 22");
    }

    #[test]
    fn percent_magnitude_is_unsigned_and_whole() {
        assert_eq!(pct_magnitude(1000.0, 300.0), "70");
        // (681-1842)/1842*100 = -63.03..., magnitude rounds to 63.
        assert_eq!(pct_magnitude(1842.0, 681.0), "63");
        // A rise reports the same way; the verb carries direction.
        assert_eq!(pct_magnitude(300.0, 1000.0), "233");
        assert_eq!(pct_magnitude(100.0, 100.0), "0");
    }

    #[test]
    fn percent_magnitude_guards_a_zero_baseline() {
        // From zero, any change is an undefined percent; report 0 rather than inf/NaN text.
        assert_eq!(pct_magnitude(0.0, 5.0), "0");
    }

    #[test]
    fn fixed_precision_is_centralized_and_stable() {
        assert_eq!(fixed(0.994, 2), "0.99");
        assert_eq!(fixed(0.99, 2), "0.99");
        assert_eq!(fixed(3.14159, 2), "3.14");
        assert_eq!(fixed(1.0, 2), "1.00");
        assert_eq!(fixed(2.0, 0), "2");
    }

    #[test]
    fn formatting_is_deterministic() {
        for value in [0.0_f64, 42.0, 1842.0, 1_000_000.0, -7.0] {
            assert_eq!(count(value), count(value), "count must be a pure function");
        }
    }
}
