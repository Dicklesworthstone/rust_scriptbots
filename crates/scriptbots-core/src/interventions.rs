//! Replayable, toroidal-exact intervention toolkit (bd-16g.10).

use crate::{Position as Vec2, Tick, toroidal_delta};
use serde::{Deserialize, Serialize};

/// Toroidal-aware spatial region specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToroidalRegion {
    /// Every point in the world.
    All,
    /// Closed disc measured using the shortest toroidal displacement.
    Disc {
        /// Disc center in world coordinates.
        center: Vec2,
        /// Disc radius in world units; validation requires a finite positive value.
        radius: f32,
    },
    /// Closed axis-aligned rectangle without wraparound across world seams.
    Rect {
        /// Inclusive lower coordinate bounds.
        min: Vec2,
        /// Inclusive upper coordinate bounds.
        max: Vec2,
    },
}

impl ToroidalRegion {
    /// Reject the malformed forms that would silently produce wrong behaviour
    /// in [`Self::contains`].
    ///
    /// In particular a negative or non-finite `Disc` radius computes
    /// `radius * radius` as a positive value, so `contains` would treat
    /// `-5.0` as if the operator had asked for a radius-5 disc. Catching
    /// it here is cheaper than diagnosing a mysterious region membership
    /// at apply-time.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::All => Ok(()),
            Self::Disc { radius, .. } => {
                if !radius.is_finite() || *radius <= 0.0 {
                    return Err("disc radius must be finite and > 0".into());
                }
                Ok(())
            }
            Self::Rect { min, max } => {
                if !min.x.is_finite()
                    || !min.y.is_finite()
                    || !max.x.is_finite()
                    || !max.y.is_finite()
                {
                    return Err("rect bounds must be finite".into());
                }
                if min.x > max.x || min.y > max.y {
                    return Err(
                        "rect bounds must be ordered (min.x <= max.x and min.y <= max.y); \
                         toroidal wraparound is not supported by this region type"
                            .into(),
                    );
                }
                Ok(())
            }
        }
    }

    /// Test membership, using `world_size` for disc wrapping and direct rectangle bounds.
    #[must_use]
    pub fn contains(&self, point: Vec2, world_size: Vec2) -> bool {
        match self {
            Self::All => true,
            Self::Disc { center, radius } => {
                let dx = toroidal_delta(point.x, center.x, world_size.x);
                let dy = toroidal_delta(point.y, center.y, world_size.y);
                #[expect(
                    clippy::suboptimal_flops,
                    reason = "Separate f32 products and addition preserve intervention membership at replay boundary points; fused rounding can change which agents are affected"
                )]
                let distance_squared = dx * dx + dy * dy;
                distance_squared <= (radius * radius)
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
        /// Region whose food growth is suppressed.
        region: ToroidalRegion,
        /// Number of simulation ticks to retain the drought effect.
        duration_ticks: u64,
    },
    /// Kill all agents and scorch food in a target disc.
    Meteor {
        /// Center of the affected toroidal disc in world coordinates.
        center: Vec2,
        /// Radius of the affected disc in world units.
        radius: f32,
    },
    /// Inject a cohort of predator agents with specified brain/genome parameters.
    PredatorInjection {
        /// Number of predators requested for injection.
        count: usize,
        /// Requested injection position in world coordinates.
        position: Vec2,
    },
    /// Paint terrain in a specified region.
    TerrainPaint {
        /// Region whose terrain cells are painted.
        region: ToroidalRegion,
        /// Encoded terrain kind to apply.
        terrain_kind: u8,
    },
    /// Freeze food diffusion across the entire world for T ticks.
    FoodEmbargo {
        /// Number of simulation ticks to retain the diffusion embargo.
        duration_ticks: u64,
    },
}

/// Record of an issued intervention for replay and science provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionRecord {
    /// Simulation tick attached to the intervention.
    pub tick: Tick,
    /// Intervention parameters to apply and record.
    pub action: InterventionAction,
    /// Caller-supplied provenance label identifying the issuer.
    pub issued_by: String,
}

impl InterventionRecord {
    /// Validate action-specific durations, counts, radii, and region bounds.
    pub fn validate(&self) -> Result<(), String> {
        match &self.action {
            InterventionAction::Drought {
                region,
                duration_ticks,
            } => {
                if *duration_ticks == 0 {
                    return Err("drought duration must be > 0".into());
                }
                region.validate()?;
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
            InterventionAction::TerrainPaint { region, .. } => {
                region.validate()?;
            }
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

    #[test]
    fn test_negative_disc_radius_is_rejected() {
        // A negative radius computes `radius * radius` as a positive value, so
        // `contains` would silently treat `-5.0` as if the operator had asked
        // for a radius-5 disc. Validation must reject it up-front.
        let record = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::Drought {
                region: ToroidalRegion::Disc {
                    center: Vec2::new(100.0, 100.0),
                    radius: -5.0,
                },
                duration_ticks: 10,
            },
            issued_by: "REST".into(),
        };
        let error = record
            .validate()
            .expect_err("negative disc radius must be rejected");
        assert!(
            error.contains("disc radius"),
            "unexpected error message: {error}"
        );
    }

    #[test]
    fn test_non_finite_disc_radius_is_rejected() {
        let record = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::TerrainPaint {
                region: ToroidalRegion::Disc {
                    center: Vec2::new(0.0, 0.0),
                    radius: f32::NAN,
                },
                terrain_kind: 1,
            },
            issued_by: "REST".into(),
        };
        assert!(record.validate().is_err());
    }

    #[test]
    fn test_inverted_rect_bounds_are_rejected() {
        // `min > max` silently turns the rect into the empty set, which would
        // be indistinguishable from "no terrain paint target" downstream.
        let record = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::TerrainPaint {
                region: ToroidalRegion::Rect {
                    min: Vec2::new(900.0, 500.0),
                    max: Vec2::new(100.0, 500.0),
                },
                terrain_kind: 1,
            },
            issued_by: "REST".into(),
        };
        let error = record
            .validate()
            .expect_err("inverted rect must be rejected");
        assert!(
            error.contains("rect bounds"),
            "unexpected error message: {error}"
        );
    }

    #[test]
    fn test_well_formed_regions_pass_validation() {
        let ok = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::Drought {
                region: ToroidalRegion::Disc {
                    center: Vec2::new(500.0, 500.0),
                    radius: 200.0,
                },
                duration_ticks: 50,
            },
            issued_by: "REST".into(),
        };
        assert!(ok.validate().is_ok());
    }
}
