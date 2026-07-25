//! Replayable, toroidal-exact intervention toolkit (bd-16g.10).

use crate::{Position as Vec2, Tick, toroidal_delta};
use serde::{Deserialize, Serialize};

/// Toroidal-aware spatial region specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToroidalRegion {
    All,
    Disc { center: Vec2, radius: f32 },
    Rect { min: Vec2, max: Vec2 },
}

impl ToroidalRegion {
    pub fn contains(&self, point: Vec2, world_size: Vec2) -> bool {
        match self {
            Self::All => true,
            Self::Disc { center, radius } => {
                let dx = toroidal_delta(point.x, center.x, world_size.x);
                let dy = toroidal_delta(point.y, center.y, world_size.y);
                (dx * dx + dy * dy) <= (radius * radius)
            }
            Self::Rect { min, max } => {
                point.x >= min.x && point.x <= max.x && point.y >= min.y && point.y <= max.y
            }
        }
    }
}

/// Intervention actions that can be journaled and replayed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InterventionAction {
    /// Suppress food growth in a region for a specified duration.
    Drought {
        region: ToroidalRegion,
        duration_ticks: u64,
    },
    /// Kill all agents and scorch food in a target disc.
    Meteor { center: Vec2, radius: f32 },
    /// Inject a cohort of predator agents with specified brain/genome parameters.
    PredatorInjection { count: usize, position: Vec2 },
    /// Paint terrain in a specified region.
    TerrainPaint {
        region: ToroidalRegion,
        terrain_kind: u8,
    },
    /// Freeze food diffusion across the entire world for T ticks.
    FoodEmbargo { duration_ticks: u64 },
}

/// Record of an issued intervention for replay and science provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionRecord {
    pub tick: Tick,
    pub action: InterventionAction,
    pub issued_by: String,
}

impl InterventionRecord {
    pub fn validate(&self) -> Result<(), String> {
        match &self.action {
            InterventionAction::Drought { duration_ticks, .. } => {
                if *duration_ticks == 0 {
                    return Err("drought duration must be > 0".into());
                }
            }
            InterventionAction::Meteor { radius, .. } => {
                if *radius <= 0.0 || !radius.is_finite() {
                    return Err("meteor radius must be finite and > 0".into());
                }
            }
            InterventionAction::PredatorInjection { count, .. } => {
                if *count == 0 || *count > 1000 {
                    return Err("predator injection count must be between 1 and 1000".into());
                }
            }
            InterventionAction::TerrainPaint { .. } => {}
            InterventionAction::FoodEmbargo { duration_ticks } => {
                if *duration_ticks == 0 {
                    return Err("food embargo duration must be > 0".into());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toroidal_region_wrapping() {
        let world_size = Vec2::new(1000.0, 1000.0);
        let disc = ToroidalRegion::Disc {
            center: Vec2::new(5.0, 500.0),
            radius: 20.0,
        };

        // Point near opposite edge across wrap seam (995, 500)
        assert!(disc.contains(Vec2::new(995.0, 500.0), world_size));
        // Point far away (500, 500)
        assert!(!disc.contains(Vec2::new(500.0, 500.0), world_size));
    }

    #[test]
    fn toroidal_disc_membership_accepts_arbitrary_point_representatives() {
        let world_size = Vec2::new(100.0, 100.0);
        let disc = ToroidalRegion::Disc {
            center: Vec2::new(1.0, 1.0),
            radius: 3.0,
        };

        assert!(disc.contains(Vec2::new(2.0, 3.0), world_size));
        assert!(
            disc.contains(Vec2::new(202.0, -397.0), world_size),
            "whole-extent translations on both axes must not change intervention membership"
        );
        assert!(
            !disc.contains(Vec2::new(250.0, -350.0), world_size),
            "an antipodal point must remain outside the small disc"
        );
    }

    #[test]
    fn test_intervention_validation() {
        let record = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::Meteor {
                center: Vec2::new(100.0, 100.0),
                radius: 50.0,
            },
            issued_by: "REST".into(),
        };
        assert!(record.validate().is_ok());

        let invalid = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::Meteor {
                center: Vec2::new(100.0, 100.0),
                radius: -10.0,
            },
            issued_by: "REST".into(),
        };
        assert!(invalid.validate().is_err());
    }
}
