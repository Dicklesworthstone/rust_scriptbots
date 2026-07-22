//! Canvas ramps, heading whiskers, hillshading, water shimmer, and day/night tint (bd-2z0.14.2.1.2).

use std::f32::consts::PI;

/// Heading whisker sector (8 directional sectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadingSector {
    North = 0,
    NorthEast = 1,
    East = 2,
    SouthEast = 3,
    South = 4,
    SouthWest = 5,
    West = 6,
    NorthWest = 7,
}

impl HeadingSector {
    pub fn from_angle(radians: f32) -> Self {
        let normalized = (radians % (2.0 * PI) + 2.0 * PI) % (2.0 * PI);
        let sector = ((normalized + PI / 8.0) / (PI / 4.0)).floor() as usize % 8;
        match sector {
            0 => Self::East,
            1 => Self::NorthEast,
            2 => Self::North,
            3 => Self::NorthWest,
            4 => Self::West,
            5 => Self::SouthWest,
            6 => Self::South,
            7 => Self::SouthEast,
            _ => Self::North,
        }
    }

    /// Braille dot pattern mask for heading whisker display.
    pub const fn braille_dot_mask(self) -> u8 {
        match self {
            Self::North => 0b0000_0001,
            Self::NorthEast => 0b0000_1001,
            Self::East => 0b0000_1000,
            Self::SouthEast => 0b0100_1000,
            Self::South => 0b0100_0000,
            Self::SouthWest => 0b0100_0000,
            Self::West => 0b0000_0010,
            Self::NorthWest => 0b0000_0011,
        }
    }
}

/// Compute hillshading intensity multiplier from elevation gradients (dz_dx, dz_dy).
pub fn hillshade_multiplier(dz_dx: f32, dz_dy: f32) -> f32 {
    let light_x = -0.707;
    let light_y = 0.707;
    let dot = -(dz_dx * light_x + dz_dy * light_y);
    (1.0 + dot * 0.4).clamp(0.4, 1.6)
}

/// Compute water shimmer alpha offset based on tick phase and position hash.
pub fn water_shimmer_alpha(tick: u64, cell_x: u32, cell_y: u32) -> f32 {
    let phase = (tick as f32 * 0.1) + (cell_x as f32 * 0.3) + (cell_y as f32 * 0.5);
    (phase.sin() * 0.15 + 0.85).clamp(0.7, 1.0)
}

/// Compute food pulse glow intensity from tick phase.
pub fn food_pulse_glow(tick: u64) -> f32 {
    let phase = tick as f32 * 0.05;
    (phase.sin() * 0.2 + 0.8).clamp(0.6, 1.0)
}

/// Compute day/night brightness tint factor from time tick.
pub fn day_night_tint_factor(tick: u64, day_cycle_ticks: u64) -> f32 {
    if day_cycle_ticks == 0 {
        return 1.0;
    }
    let cycle_pos = (tick % day_cycle_ticks) as f32 / day_cycle_ticks as f32;
    let sun_angle = cycle_pos * 2.0 * PI;
    (sun_angle.sin() * 0.4 + 0.6).clamp(0.2, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heading_sector_encoding() {
        assert_eq!(HeadingSector::from_angle(0.0), HeadingSector::East);
        assert_eq!(HeadingSector::from_angle(PI / 2.0), HeadingSector::North);
        assert_eq!(HeadingSector::from_angle(PI), HeadingSector::West);
        assert_eq!(HeadingSector::from_angle(3.0 * PI / 2.0), HeadingSector::South);

        assert!(HeadingSector::North.braille_dot_mask() > 0);
    }

    #[test]
    fn test_hillshading_and_tint_curves() {
        let flat = hillshade_multiplier(0.0, 0.0);
        assert_eq!(flat, 1.0);

        let steep = hillshade_multiplier(1.0, -1.0);
        assert!(steep > 1.0);

        let tint = day_night_tint_factor(500, 1000);
        assert!(tint >= 0.2 && tint <= 1.0);

        let glow = food_pulse_glow(10);
        assert!(glow >= 0.6 && glow <= 1.0);
    }
}
