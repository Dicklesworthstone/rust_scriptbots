//! Order-independent sensor accumulation, in fixed point.
//!
//! # Why this exists
//!
//! Naive GPU float reductions are NONDETERMINISTIC across vendors and drivers:
//! workgroup accumulation order varies, FMA contraction varies, and `sin`,
//! `cos`, `atan2` and `acos` are implementation-defined. Shipping GPU sensing on
//! `f32` accumulation would silently convert a bit-reproducible simulator into an
//! irreproducible one — and it would still LOOK right, which makes it worse than
//! any of the bugs this program has already fixed.
//!
//! Integer addition is associative and commutative. That is not a performance
//! property; it is THE property. It is why fixed point here is a design, not a
//! workaround.
//!
//! # The two halves of a contribution
//!
//! Per-neighbor TERMS are still computed in `f32` — `dx`, `dist`, `intensity`.
//! That is unavoidable and acceptable, because both lanes compute them from the
//! same inputs with the same operations. It is the ACCUMULATION, the only place
//! where visit order can enter the answer, that is integer.
//!
//! # Geometry
//!
//! `atan2` is gone from the per-neighbor path entirely. The eye falloff depends
//! on `|Δangle| = acos(dot(eye_unit, neighbor_unit))`, so each agent's eye unit
//! vectors are built ONCE (four transcendentals per agent, rather than four per
//! agent-neighbor PAIR — which is also a speedup), and the inner loop needs only
//! a dot product and a shared polynomial `acos` whose coefficients and evaluation
//! order are the contract between this module and the shader.
//!
//! # What this module is not
//!
//! It knows nothing about wgpu, rendering, or storage, and it does not own the
//! sensor wire format — [`crate::channels::SENSOR_LAYOUT`] does. It produces
//! CHANNELS; the caller places them. The GPU lane must never reimplement
//! [`SenseAccum::finalize`]: a reimplementation is a second accountant with a
//! floating-point opinion.

use crate::NUM_EYES;

/// Fractional bits in the fixed-point accumulator.
pub const SENSE_FRAC_BITS: u32 = 20;

/// Geometry contract named in lifecycle diagnostics and run evidence.
pub const SENSE_GEOMETRY: &str = "poly_acos";

/// One unit in the accumulator's last place.
const ONE: i64 = 1 << SENSE_FRAC_BITS;

/// The largest magnitude any single per-neighbor term may contribute.
///
/// Every factor in a contribution is bounded: the field-of-view factor and the
/// distance factor are both in `[0, 1]`, colours are in `[0, 1]`, and
/// `dist / radius <= 1`. So a term is bounded by the agent's eye sensitivity.
/// Four is a generous ceiling on that — legacy trait values live well below it —
/// and a term that exceeds it saturates rather than being trusted.
pub const MAX_TERM: f32 = 4.0;

/// How many neighbours a single agent's accumulator is sized for.
///
/// The neighbour count is NOT bounded in principle — agents pile up — so the
/// accumulator is sized for a deliberately pessimistic crowd and SATURATES past
/// it. See [`SenseAccum::saturations`].
pub const MAX_NEIGHBORS_ASSUMED: i64 = 4_096;

/// Ceiling on any one channel, in fixed-point units.
///
/// RANGE DERIVATION (this is a load-bearing calculation, not a footnote):
///
/// ```text
///   MAX_TERM * 2^FRAC = 4 * 2^20            = 4_194_304
///   ceiling = 4_096 neighbours * 4_194_304  = 17_179_869_184  (~2^34)
///   i64::MAX                                = ~2^63
///   headroom                                = ~2^29
/// ```
///
/// That is far more than the 2^10 of headroom required, so the accumulator
/// cannot overflow even if every one of the assumed neighbours contributes the
/// maximum term. Overflow past the ceiling SATURATES: a wrapping i64 would
/// produce a NEGATIVE sensor value — a silently insane input to a brain, which
/// is the worst failure available here.
pub const ACCUM_CEILING: i64 = MAX_NEIGHBORS_ASSUMED * (MAX_TERM as i64) * ONE;

/// Whole binary orders of headroom between [`ACCUM_CEILING`] and `i64::MAX`.
///
/// This is emitted in the lifecycle numeric-contract line. Its test derives the
/// value from the ceiling so the diagnostic cannot silently outlive the range
/// analysis it claims to summarize.
pub const SENSE_HEADROOM_BITS: u32 = 29;

/// Convert to fixed point: round-to-nearest-even, saturating.
///
/// Saturating rather than wrapping, for the reason in [`ACCUM_CEILING`]. A NaN
/// term contributes zero rather than poisoning the accumulator — a NaN that
/// reached a brain would propagate through every downstream reduction.
#[must_use]
#[inline]
pub fn to_fixed(value: f32) -> i64 {
    to_fixed_with_saturation(value).0
}

#[inline]
fn to_fixed_with_saturation(value: f32) -> (i64, bool) {
    if !value.is_finite() {
        return (0, false);
    }
    let scaled = f64::from(value) * f64::from(ONE as u32);
    // round-ties-to-even, matching IEEE's default and the shader's rounding.
    let rounded = round_half_to_even(scaled);
    if rounded >= ACCUM_CEILING as f64 {
        return (ACCUM_CEILING, true);
    }
    if rounded <= -(ACCUM_CEILING as f64) {
        return (-ACCUM_CEILING, true);
    }
    (rounded as i64, false)
}

fn round_half_to_even(value: f64) -> f64 {
    let floor = value.floor();
    let frac = value - floor;
    // Ties go to the even neighbour, matching IEEE's default rounding mode and
    // therefore the shader's. A tie broken differently on the two lanes is a
    // one-ULP disagreement that the accumulator would then carry forever.
    let round_up = frac > 0.5 || (frac == 0.5 && (floor as i64) % 2 != 0);
    if round_up { floor + 1.0 } else { floor }
}

/// Convert back to `f32`.
#[must_use]
#[inline]
pub fn from_fixed(value: i64) -> f32 {
    (value as f64 / f64::from(ONE as u32)) as f32
}

/// Coefficients of the shared `acos` polynomial (Abramowitz & Stegun 4.4.45).
///
/// These coefficients — and the evaluation ORDER below — are the contract
/// between this module and the WGSL shader. Not "the same approximation": the
/// same operations, in the same sequence. A Horner-versus-Estrin divergence, or
/// an `fma` on one side and a `mul`+`add` on the other, is enough to make the
/// two lanes disagree.
// The leading coefficient is NEAR pi/2 but is not pi/2: it is a fitted value
// from the reference table, and substituting `FRAC_PI_2` would perturb the
// polynomial and invalidate the error bound that the test proves.
#[allow(clippy::approx_constant)]
const ACOS_COEFFS: [f32; 8] = [
    1.570_796_3,
    -0.214_598_8,
    0.088_978_99,
    -0.050_174_3,
    0.030_891_88,
    -0.017_088_126,
    0.006_670_09,
    -0.001_262_491_1,
];

/// Maximum absolute error of [`poly_acos`] over `[-1, 1]`, proven by test.
pub const ACOS_MAX_ERROR: f32 = 1e-6;

/// `acos`, without the vendor-defined builtin.
///
/// Evaluated with explicit `a + b * x` (never `mul_add`): Rust does not contract
/// to FMA without fast-math, and the shader must not either, or the two lanes
/// round differently and the whole exercise is pointless.
#[must_use]
#[inline]
pub fn poly_acos(x: f32) -> f32 {
    let x = x.clamp(-1.0, 1.0);
    let negative = x < 0.0;
    let a = x.abs();

    // Horner, in this exact order.
    let mut poly = ACOS_COEFFS[7];
    poly = ACOS_COEFFS[6] + poly * a;
    poly = ACOS_COEFFS[5] + poly * a;
    poly = ACOS_COEFFS[4] + poly * a;
    poly = ACOS_COEFFS[3] + poly * a;
    poly = ACOS_COEFFS[2] + poly * a;
    poly = ACOS_COEFFS[1] + poly * a;
    poly = ACOS_COEFFS[0] + poly * a;

    let result = poly * (1.0 - a).sqrt();
    if negative {
        std::f32::consts::PI - result
    } else {
        result
    }
}

/// What one neighbour contributes to one agent's senses.
///
/// The caller computes these `f32` terms (identically on both lanes); this
/// module is responsible only for adding them up in an order-free way.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NeighborContribution {
    /// Proximity, per eye.
    pub density: [f32; NUM_EYES],
    /// Red, per eye.
    pub red: [f32; NUM_EYES],
    /// Green, per eye.
    pub green: [f32; NUM_EYES],
    /// Blue, per eye.
    pub blue: [f32; NUM_EYES],
    /// Smell.
    pub smell: f32,
    /// Sound.
    pub sound: f32,
    /// Hearing.
    pub hearing: f32,
    /// Blood.
    pub blood: f32,
}

/// The accumulator. Every field is an integer, so every field is order-free.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SenseAccum {
    /// Proximity, per eye.
    pub density: [i64; NUM_EYES],
    /// Red, per eye.
    pub red: [i64; NUM_EYES],
    /// Green, per eye.
    pub green: [i64; NUM_EYES],
    /// Blue, per eye.
    pub blue: [i64; NUM_EYES],
    /// Smell.
    pub smell: i64,
    /// Sound.
    pub sound: i64,
    /// Hearing.
    pub hearing: i64,
    /// Blood.
    pub blood: i64,
    /// How many times a channel hit [`ACCUM_CEILING`].
    ///
    /// A run with a non-zero count is SUSPECT and must say so: a sensor that hit
    /// the ceiling is a sensor that lied to a brain, and the run's conclusions
    /// inherit that. The struct carries the count so the caller can emit a
    /// complete log line without reaching back into the kernel.
    pub saturations: u32,
}

/// The finalized, clamped channels — what a brain actually reads.
///
/// This module does NOT map these into `INPUT_SIZE`: [`crate::channels::SENSOR_LAYOUT`]
/// owns the wire format, and duplicating it here would create a second place for
/// the sensor order to drift.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SenseChannels {
    /// Proximity, per eye, in `[0, 1]`.
    pub density: [f32; NUM_EYES],
    /// Red, per eye, in `[0, 1]`.
    pub red: [f32; NUM_EYES],
    /// Green, per eye, in `[0, 1]`.
    pub green: [f32; NUM_EYES],
    /// Blue, per eye, in `[0, 1]`.
    pub blue: [f32; NUM_EYES],
    /// Smell, in `[0, 1]`.
    pub smell: f32,
    /// Sound, in `[0, 1]`.
    pub sound: f32,
    /// Hearing, in `[0, 1]`.
    pub hearing: f32,
    /// Blood, in `[0, 1]`.
    pub blood: f32,
}

#[inline]
fn add_saturating(slot: &mut i64, term: f32, saturations: &mut u32) {
    let (fixed, conversion_saturated) = to_fixed_with_saturation(term);
    let sum = slot.saturating_add(fixed);
    let accumulation_saturated = if sum > ACCUM_CEILING {
        *slot = ACCUM_CEILING;
        true
    } else if sum < -ACCUM_CEILING {
        *slot = -ACCUM_CEILING;
        true
    } else {
        *slot = sum;
        false
    };
    if conversion_saturated || accumulation_saturated {
        *saturations = saturations.saturating_add(1);
    }
}

impl SenseAccum {
    /// Fold one neighbour in.
    ///
    /// Pure integer adds: associative and commutative, therefore order-free. Two
    /// lanes that visit the same neighbours in different orders reach a
    /// BIT-IDENTICAL accumulator, which is the entire point of this module.
    pub fn contribute(&mut self, contribution: &NeighborContribution) {
        for eye in 0..NUM_EYES {
            add_saturating(
                &mut self.density[eye],
                contribution.density[eye],
                &mut self.saturations,
            );
            add_saturating(
                &mut self.red[eye],
                contribution.red[eye],
                &mut self.saturations,
            );
            add_saturating(
                &mut self.green[eye],
                contribution.green[eye],
                &mut self.saturations,
            );
            add_saturating(
                &mut self.blue[eye],
                contribution.blue[eye],
                &mut self.saturations,
            );
        }
        add_saturating(&mut self.smell, contribution.smell, &mut self.saturations);
        add_saturating(&mut self.sound, contribution.sound, &mut self.saturations);
        add_saturating(
            &mut self.hearing,
            contribution.hearing,
            &mut self.saturations,
        );
        add_saturating(&mut self.blood, contribution.blood, &mut self.saturations);
    }

    /// Convert to the clamped channels a brain reads.
    ///
    /// The clamp to `[0, 1]` is why an accumulation-order bug here would be
    /// INTERMITTENT rather than obvious: most of the time the difference
    /// disappears into the clamp, and then one day it does not.
    #[must_use]
    pub fn finalize(&self) -> SenseChannels {
        self.finalize_with_multipliers(1.0, 1.0, 1.0, 1.0)
    }

    /// Convert to clamped channels after applying the non-eye trait modifiers.
    ///
    /// Smell, sound, hearing, and blood are accumulated without their agent
    /// traits and scaled exactly once after the complete neighbour reduction,
    /// matching the production sensing contract. Eye sensitivity is already
    /// part of each eye contribution and is deliberately not applied here.
    #[must_use]
    pub fn finalize_with_multipliers(
        &self,
        smell: f32,
        sound: f32,
        hearing: f32,
        blood: f32,
    ) -> SenseChannels {
        let channel = |value: i64, multiplier: f32| {
            (from_fixed(value) * multiplier).clamp(0.0, 1.0)
        };
        let mut out = SenseChannels {
            smell: channel(self.smell, smell),
            sound: channel(self.sound, sound),
            hearing: channel(self.hearing, hearing),
            blood: channel(self.blood, blood),
            ..SenseChannels::default()
        };
        for eye in 0..NUM_EYES {
            out.density[eye] = channel(self.density[eye], 1.0);
            out.red[eye] = channel(self.red[eye], 1.0);
            out.green[eye] = channel(self.green[eye], 1.0);
            out.blue[eye] = channel(self.blue[eye], 1.0);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;

    #[test]
    fn the_fixed_point_round_trip_stays_within_half_an_ulp() {
        let bound = 1.0 / (2.0 * (ONE as f32));
        for step in -4_000..=4_000 {
            let value = step as f32 * 0.001;
            let round_tripped = from_fixed(to_fixed(value));
            assert!(
                (round_tripped - value).abs() <= bound,
                "round trip of {value} lost {} > {bound}",
                (round_tripped - value).abs()
            );
        }
        // A NaN term must contribute nothing rather than poisoning every
        // downstream reduction it touches.
        assert_eq!(to_fixed(f32::NAN), 0);
        assert_eq!(to_fixed(f32::INFINITY), 0);
    }

    #[test]
    fn conversion_saturates_and_never_wraps() {
        // A wrapping i64 would turn an overflowing sensor NEGATIVE — a silently
        // insane input to a brain.
        assert_eq!(to_fixed(1e30), ACCUM_CEILING);
        assert_eq!(to_fixed(-1e30), -ACCUM_CEILING);
        assert!(to_fixed(1e30) > 0, "saturation must not wrap negative");
    }

    #[test]
    fn advertised_headroom_matches_the_accumulator_range_derivation() {
        let derived = (i64::MAX as u64 / ACCUM_CEILING as u64).ilog2();
        assert_eq!(SENSE_HEADROOM_BITS, derived);
        assert_eq!(SENSE_GEOMETRY, "poly_acos");
    }

    #[test]
    fn a_single_oversized_term_counts_its_conversion_saturation() {
        let mut accum = SenseAccum::default();
        accum.contribute(&NeighborContribution {
            smell: 1e30,
            sound: -1e30,
            ..NeighborContribution::default()
        });

        assert_eq!(accum.smell, ACCUM_CEILING);
        assert_eq!(accum.sound, -ACCUM_CEILING);
        assert_eq!(
            accum.saturations, 2,
            "each term clamped during fixed-point conversion must make the run suspect"
        );
    }

    #[test]
    fn trait_scaled_finalize_applies_non_eye_traits_after_accumulation() {
        let mut accum = SenseAccum::default();
        accum.contribute(&NeighborContribution {
            density: [0.5, 0.0, 0.0, 0.0],
            smell: 0.5,
            sound: 0.25,
            hearing: 0.75,
            blood: 0.5,
            ..NeighborContribution::default()
        });

        let channels = accum.finalize_with_multipliers(0.5, 2.0, 2.0, 0.25);

        assert_eq!(channels.density[0].to_bits(), 0.5f32.to_bits());
        assert_eq!(channels.smell.to_bits(), 0.25f32.to_bits());
        assert_eq!(channels.sound.to_bits(), 0.5f32.to_bits());
        assert_eq!(channels.hearing.to_bits(), 1.0f32.to_bits());
        assert_eq!(channels.blood.to_bits(), 0.125f32.to_bits());
    }

    fn contribution(seed: f32) -> NeighborContribution {
        NeighborContribution {
            density: [seed * 0.013, seed * 0.007, seed * 0.019, seed * 0.003],
            red: [seed * 0.011; NUM_EYES],
            green: [seed * 0.005; NUM_EYES],
            blue: [seed * 0.017; NUM_EYES],
            smell: seed * 0.009,
            sound: seed * 0.021,
            hearing: seed * 0.001,
            blood: seed * 0.015,
        }
    }

    #[test]
    fn accumulation_is_bit_identical_under_every_permutation() {
        // THE test. The acceptance property is not "close" — it is IDENTICAL.
        let neighbors: Vec<NeighborContribution> = (0..64)
            .map(|i| contribution(0.001 + (i as f32) * 0.000_173))
            .collect();

        let accumulate = |order: &[usize]| {
            let mut accum = SenseAccum::default();
            for &index in order {
                accum.contribute(&neighbors[index]);
            }
            accum
        };

        let mut order: Vec<usize> = (0..neighbors.len()).collect();
        let reference = accumulate(&order);

        let mut rng = SmallRng::seed_from_u64(0xF1_1ED);
        for shuffle in 0..1_000 {
            order.shuffle(&mut rng);
            assert_eq!(
                accumulate(&order),
                reference,
                "shuffle {shuffle} changed the integer accumulator — the whole \
                 premise of this module is that it cannot"
            );
        }
    }

    #[test]
    fn the_legacy_f32_accumulation_really_does_diverge_under_reordering() {
        // Without this test, "fixed point for determinism" is a story we tell
        // ourselves. This is what proves the problem is real and the fix is
        // load-bearing: the SAME neighbours, summed in a different order, in
        // f32, produce a different number.
        let neighbors: Vec<f32> = (0..64).map(|i| 0.001 + (i as f32) * 0.000_173_31).collect();

        let sum_f32 = |order: &[usize]| {
            let mut total = 0.0f32;
            for &index in order {
                total += neighbors[index];
            }
            total
        };

        let mut order: Vec<usize> = (0..neighbors.len()).collect();
        let reference = sum_f32(&order);

        let mut rng = SmallRng::seed_from_u64(99);
        let mut diverged = false;
        for _ in 0..1_000 {
            order.shuffle(&mut rng);
            if sum_f32(&order).to_bits() != reference.to_bits() {
                diverged = true;
                break;
            }
        }
        assert!(
            diverged,
            "if f32 accumulation were order-independent, this module would be \
             unnecessary — the fact that it diverges is why it exists"
        );
    }

    #[test]
    fn a_pathological_pile_up_saturates_loudly_and_never_goes_negative() {
        let mut accum = SenseAccum::default();
        let huge = NeighborContribution {
            density: [MAX_TERM; NUM_EYES],
            red: [MAX_TERM; NUM_EYES],
            green: [MAX_TERM; NUM_EYES],
            blue: [MAX_TERM; NUM_EYES],
            smell: MAX_TERM,
            sound: MAX_TERM,
            hearing: MAX_TERM,
            blood: MAX_TERM,
        };
        // Well past MAX_NEIGHBORS_ASSUMED.
        for _ in 0..(MAX_NEIGHBORS_ASSUMED + 64) {
            accum.contribute(&huge);
        }

        assert!(
            accum.saturations > 0,
            "a crowd past the assumed maximum must be reported, not swallowed"
        );
        assert_eq!(accum.density[0], ACCUM_CEILING);
        assert!(
            accum.density[0] > 0 && accum.smell > 0 && accum.blood > 0,
            "saturation must never wrap an accumulator negative"
        );

        let channels = accum.finalize();
        for eye in 0..NUM_EYES {
            for value in [
                channels.density[eye],
                channels.red[eye],
                channels.green[eye],
                channels.blue[eye],
            ] {
                assert!(
                    (0.0..=1.0).contains(&value),
                    "a saturated channel must still finalize inside [0, 1], got {value}"
                );
            }
        }
        for value in [
            channels.smell,
            channels.sound,
            channels.hearing,
            channels.blood,
        ] {
            assert!((0.0..=1.0).contains(&value), "got {value}");
        }
    }

    #[test]
    fn the_shared_acos_polynomial_meets_its_stated_error_bound() {
        // Including the endpoints, where naive polynomials blow up.
        let mut worst = 0.0f32;
        let mut worst_at = 0.0f32;
        for step in 0..=20_000 {
            let x = -1.0 + (step as f32) * (2.0 / 20_000.0);
            let x = x.clamp(-1.0, 1.0);
            let error = (poly_acos(x) - x.acos()).abs();
            if error > worst {
                worst = error;
                worst_at = x;
            }
        }
        assert!(
            worst <= ACOS_MAX_ERROR,
            "poly_acos error {worst} at x = {worst_at} exceeds the declared \
             bound {ACOS_MAX_ERROR}; the bound is part of the model's numeric \
             contract and cannot be quietly widened"
        );
        // The endpoints are exact enough to be usable as boundaries.
        assert!((poly_acos(1.0) - 0.0).abs() <= ACOS_MAX_ERROR);
        assert!((poly_acos(-1.0) - std::f32::consts::PI).abs() <= ACOS_MAX_ERROR);
        // Out-of-range inputs (which a dot product can produce by a rounding
        // whisker) must clamp, not produce NaN.
        assert!(poly_acos(1.000_001).is_finite());
        assert!(poly_acos(-1.000_001).is_finite());
    }

    #[test]
    fn an_empty_accumulator_finalizes_to_silence() {
        let channels = SenseAccum::default().finalize();
        assert_eq!(channels.density, [0.0; NUM_EYES]);
        assert_eq!(channels.smell, 0.0);
        assert_eq!(channels.blood, 0.0);
    }
}
