//! Sub-cell heading encoding for the `FrankenTUI` world canvas (bd-2z0.14.2.1).
//!
//! # What is deliberately NOT here
//!
//! This module previously also carried `hillshade_multiplier`,
//! `water_shimmer_alpha`, `food_pulse_glow`, and `day_night_tint_factor`. Each
//! was a terminal-local re-derivation of a curve that
//! [`scriptbots_core::visual`] already defines for every surface:
//!
//! | removed here            | shared definition                            |
//! |-------------------------|----------------------------------------------|
//! | `hillshade_multiplier`  | [`visual::terrain_normal_light_factor`]      |
//! | `water_shimmer_alpha`   | [`visual::shimmer`]                          |
//! | `food_pulse_glow`       | [`visual::shimmer`]                          |
//! | `day_night_tint_factor` | [`visual::daylight_factor`]                  |
//!
//! Keeping both sets meant the terminal could report a different time of day, or
//! a different point in a cell's pulse, than Bevy or GPUI showed for the same
//! tick — which is exactly the cross-surface equivalence this canvas is supposed
//! to demonstrate. The canvas now calls the shared functions directly, so the
//! only thing left in this module is the piece that has no shared equivalent:
//! how a heading becomes a neighbouring SUB-PIXEL, which is a fact about the
//! braille grid and about nothing else.
//!
//! [`visual::terrain_normal_light_factor`]: scriptbots_core::visual::terrain_normal_light_factor
//! [`visual::shimmer`]: scriptbots_core::visual::shimmer
//! [`visual::daylight_factor`]: scriptbots_core::visual::daylight_factor
//! [`scriptbots_core::visual`]: scriptbots_core::visual

use std::f32::consts::{PI, TAU};

/// One of the eight compass sectors a heading is quantized into.
///
/// The sector index matches the terminal's arrow glyphs exactly (`→ ↗ ↑ ↖ ← ↙ ↓
/// ↘` for 0..=7), so the canvas whisker and the flat map's arrow can never
/// disagree about which way an agent is facing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadingSector {
    East = 0,
    NorthEast = 1,
    North = 2,
    NorthWest = 3,
    West = 4,
    SouthWest = 5,
    South = 6,
    SouthEast = 7,
}

impl HeadingSector {
    /// Quantize a world heading in radians (0 = east, increasing counterclockwise).
    ///
    /// A non-finite heading is reported as [`Self::East`] rather than panicking:
    /// one corrupt agent must not be able to take down the frame loop.
    #[must_use]
    pub fn from_angle(radians: f32) -> Self {
        if !radians.is_finite() {
            return Self::East;
        }
        let sector = ((radians.rem_euclid(TAU) / (PI / 4.0)).round() as i32) & 7;
        match sector {
            1 => Self::NorthEast,
            2 => Self::North,
            3 => Self::NorthWest,
            4 => Self::West,
            5 => Self::SouthWest,
            6 => Self::South,
            7 => Self::SouthEast,
            _ => Self::East,
        }
    }

    /// The neighbouring sub-pixel offset `(dx, dy)` in the direction of travel.
    ///
    /// Screen space, so `dy` grows DOWNWARD: north is `(0, -1)`. All eight
    /// offsets are distinct, which is what makes the whisker readable — an
    /// encoding that collapsed two sectors onto the same dot would render two
    /// different headings identically.
    #[must_use]
    pub const fn whisker_offset(self) -> (i32, i32) {
        match self {
            Self::East => (1, 0),
            Self::NorthEast => (1, -1),
            Self::North => (0, -1),
            Self::NorthWest => (-1, -1),
            Self::West => (-1, 0),
            Self::SouthWest => (-1, 1),
            Self::South => (0, 1),
            Self::SouthEast => (1, 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn cardinal_angles_map_to_their_sectors() {
        assert_eq!(HeadingSector::from_angle(0.0), HeadingSector::East);
        assert_eq!(HeadingSector::from_angle(PI / 2.0), HeadingSector::North);
        assert_eq!(HeadingSector::from_angle(PI), HeadingSector::West);
        assert_eq!(
            HeadingSector::from_angle(3.0 * PI / 2.0),
            HeadingSector::South
        );
        // Wrapping and negative headings resolve to the same sectors.
        assert_eq!(HeadingSector::from_angle(TAU), HeadingSector::East);
        assert_eq!(HeadingSector::from_angle(-PI / 2.0), HeadingSector::South);
    }

    #[test]
    fn diagonal_angles_map_to_their_sectors() {
        for (index, expected) in [
            HeadingSector::East,
            HeadingSector::NorthEast,
            HeadingSector::North,
            HeadingSector::NorthWest,
            HeadingSector::West,
            HeadingSector::SouthWest,
            HeadingSector::South,
            HeadingSector::SouthEast,
        ]
        .into_iter()
        .enumerate()
        {
            let angle = index as f32 * (PI / 4.0);
            assert_eq!(
                HeadingSector::from_angle(angle),
                expected,
                "sector {index} at {angle} rad"
            );
        }
    }

    /// The whole point of the whisker: eight headings must produce eight
    /// distinguishable dots. A collision would silently render two directions
    /// the same.
    #[test]
    fn every_sector_has_a_distinct_unit_offset() {
        let sectors = [
            HeadingSector::East,
            HeadingSector::NorthEast,
            HeadingSector::North,
            HeadingSector::NorthWest,
            HeadingSector::West,
            HeadingSector::SouthWest,
            HeadingSector::South,
            HeadingSector::SouthEast,
        ];
        let offsets: BTreeSet<(i32, i32)> = sectors.iter().map(|s| s.whisker_offset()).collect();
        assert_eq!(offsets.len(), 8, "all eight offsets distinct: {offsets:?}");
        for sector in sectors {
            let (dx, dy) = sector.whisker_offset();
            assert!(
                dx.abs() <= 1 && dy.abs() <= 1 && (dx != 0 || dy != 0),
                "{sector:?} must be a nonzero unit step, got ({dx},{dy})"
            );
        }
    }

    /// The offset must point the same way the sector's angle does: `dx` follows
    /// `cos`, and `dy` follows `-sin` because screen rows grow downward.
    #[test]
    fn offsets_agree_with_the_angle_they_came_from() {
        for step in 0..8 {
            let angle = step as f32 * (PI / 4.0);
            let (dx, dy) = HeadingSector::from_angle(angle).whisker_offset();
            assert_eq!(
                dx,
                angle.cos().round() as i32,
                "dx follows cos at {angle} rad"
            );
            assert_eq!(
                dy,
                (-angle.sin()).round() as i32,
                "dy follows -sin at {angle} rad"
            );
        }
    }

    #[test]
    fn non_finite_headings_do_not_panic() {
        assert_eq!(HeadingSector::from_angle(f32::NAN), HeadingSector::East);
        assert_eq!(
            HeadingSector::from_angle(f32::INFINITY),
            HeadingSector::East
        );
    }
}
