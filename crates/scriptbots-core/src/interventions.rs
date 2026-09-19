pub use crate::{
    CellSelector, CohortSource, CoreBuildIdentityV0, Intervention, Placement, Position as Vec2,
    Region,
};
use crate::{TerrainKind, Tick, WorldState, WorldStateError, toroidal_delta};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::path::Path;

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

    /// Convert this legacy record into a canonical intervention command.
    #[must_use]
    pub fn into_command(self, surface: InterventionSurface) -> InterventionCommand {
        InterventionCommand {
            intervention: self.action.into(),
            surface,
            actor: self.issued_by,
        }
    }
}

impl From<ToroidalRegion> for Region {
    fn from(region: ToroidalRegion) -> Self {
        match region {
            ToroidalRegion::All => Self::All,
            ToroidalRegion::Disc { center, radius } => Self::Disc {
                x: center.x,
                y: center.y,
                radius,
            },
            ToroidalRegion::Rect { min, max } => Self::Rect {
                x: min.x,
                y: min.y,
                w: (max.x - min.x).max(0.0),
                h: (max.y - min.y).max(0.0),
            },
        }
    }
}

impl From<Region> for ToroidalRegion {
    fn from(region: Region) -> Self {
        match region {
            Region::All => Self::All,
            Region::Disc { x, y, radius } => Self::Disc {
                center: Vec2::new(x, y),
                radius,
            },
            Region::Rect { x, y, w, h } => Self::Rect {
                min: Vec2::new(x, y),
                max: Vec2::new(x + w, y + h),
            },
        }
    }
}

impl From<InterventionAction> for Intervention {
    fn from(action: InterventionAction) -> Self {
        match action {
            InterventionAction::Drought {
                region,
                duration_ticks,
            } => Self::Drought {
                region: region.into(),
                ticks: u32::try_from(duration_ticks).unwrap_or(u32::MAX),
                growth_scale: 0.0,
            },
            InterventionAction::Meteor { center, radius } => Self::Meteor {
                region: Region::Disc {
                    x: center.x,
                    y: center.y,
                    radius,
                },
                lethality: 1.0,
                scorch: 0.5,
            },
            InterventionAction::PredatorInjection { count, position } => Self::InjectCohort {
                count: u16::try_from(count).unwrap_or(u16::MAX),
                genome: CohortSource::RegisteredBrain { key: 0 },
                placement: Placement::Seeded {
                    region: Region::Disc {
                        x: position.x,
                        y: position.y,
                        radius: 10.0,
                    },
                    seed: 0,
                },
                herbivore_tendency: 0.05,
            },
            InterventionAction::TerrainPaint {
                region,
                terrain_kind,
            } => Self::PaintTerrain {
                region: region.into(),
                terrain: match terrain_kind {
                    0 => TerrainKind::Grass,
                    1 => TerrainKind::Sand,
                    2 => TerrainKind::DeepWater,
                    3 => TerrainKind::ShallowWater,
                    4 => TerrainKind::Bloom,
                    _ => TerrainKind::Rock,
                },
                fertility_bias: None,
            },
            InterventionAction::FoodEmbargo { duration_ticks } => Self::Embargo {
                region: Region::All,
                ticks: u32::try_from(duration_ticks).unwrap_or(u32::MAX),
            },
        }
    }
}

/// External control surface that initiated an intervention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionSurface {
    /// Native GPU rendering frontend / GPUI laboratory.
    Gpu,
    /// Interactive terminal UI palette / keyboard shortcut.
    Tui,
    /// HTTP REST control endpoint.
    Rest,
    /// Model Context Protocol tool call.
    Mcp,
    /// Headless control CLI utility.
    Cli,
    /// Replayable journal or scripted protocol file.
    Script,
}

impl fmt::Display for InterventionSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu => write!(f, "gpu"),
            Self::Tui => write!(f, "tui"),
            Self::Rest => write!(f, "rest"),
            Self::Mcp => write!(f, "mcp"),
            Self::Cli => write!(f, "cli"),
            Self::Script => write!(f, "script"),
        }
    }
}

/// A structured intervention request carrying provenance before admission into the world or journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterventionCommand {
    /// Canonical intervention specification.
    pub intervention: Intervention,
    /// Ingress control surface.
    pub surface: InterventionSurface,
    /// Human or automated actor identity.
    pub actor: String,
}

impl InterventionCommand {
    /// Canonical constructor binding an intervention to an actor and surface.
    #[must_use]
    pub fn new(
        intervention: Intervention,
        surface: InterventionSurface,
        actor: impl Into<String>,
    ) -> Self {
        Self {
            intervention,
            surface,
            actor: actor.into(),
        }
    }
}

/// Canonical constructor for a drought intervention.
pub fn drought(
    region: Region,
    ticks: u32,
    growth_scale: f32,
) -> Result<Intervention, WorldStateError> {
    let intervention = Intervention::Drought {
        region,
        ticks,
        growth_scale,
    };
    intervention.validate()?;
    Ok(intervention)
}

/// Canonical constructor for an embargo intervention.
pub fn embargo(region: Region, ticks: u32) -> Result<Intervention, WorldStateError> {
    let intervention = Intervention::Embargo { region, ticks };
    intervention.validate()?;
    Ok(intervention)
}

/// Canonical constructor for a food bloom intervention.
pub fn bloom(region: Region, amount: f32) -> Result<Intervention, WorldStateError> {
    let intervention = Intervention::Bloom { region, amount };
    intervention.validate()?;
    Ok(intervention)
}

/// Canonical constructor for a meteor strike intervention.
pub fn meteor(
    region: Region,
    lethality: f32,
    scorch: f32,
) -> Result<Intervention, WorldStateError> {
    let intervention = Intervention::Meteor {
        region,
        lethality,
        scorch,
    };
    intervention.validate()?;
    Ok(intervention)
}

/// Canonical constructor for a terrain paint intervention.
pub fn paint_terrain(
    region: Region,
    terrain: TerrainKind,
    fertility_bias: Option<f32>,
) -> Result<Intervention, WorldStateError> {
    let intervention = Intervention::PaintTerrain {
        region,
        terrain,
        fertility_bias,
    };
    intervention.validate()?;
    Ok(intervention)
}

/// Canonical constructor for a cohort injection intervention.
pub fn inject_cohort(
    count: u16,
    genome: CohortSource,
    placement: Placement,
    herbivore_tendency: f32,
) -> Result<Intervention, WorldStateError> {
    let intervention = Intervention::InjectCohort {
        count,
        genome,
        placement,
        herbivore_tendency,
    };
    intervention.validate()?;
    Ok(intervention)
}

/// Canonical constructor for world boundary closing/opening.
pub fn set_closed_world(closed: bool) -> Result<Intervention, WorldStateError> {
    let intervention = Intervention::SetClosedWorld { closed };
    intervention.validate()?;
    Ok(intervention)
}

/// Canonical constructor for MAP-Elites resurrection intervention.
pub fn spawn_from_archive(
    selector: CellSelector,
    placement: Placement,
    headroom: Option<u32>,
    herbivore_tendency: Option<f32>,
) -> Result<Intervention, WorldStateError> {
    let intervention = Intervention::SpawnFromArchive {
        selector,
        placement,
        headroom,
        herbivore_tendency,
    };
    intervention.validate()?;
    Ok(intervention)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
enum CanonicalRegionPostcard {
    All,
    Disc { x: f32, y: f32, radius: f32 },
    Rect { x: f32, y: f32, w: f32, h: f32 },
}

impl From<Region> for CanonicalRegionPostcard {
    fn from(r: Region) -> Self {
        match r {
            Region::All => Self::All,
            Region::Disc { x, y, radius } => Self::Disc { x, y, radius },
            Region::Rect { x, y, w, h } => Self::Rect { x, y, w, h },
        }
    }
}

impl From<CanonicalRegionPostcard> for Region {
    fn from(c: CanonicalRegionPostcard) -> Self {
        match c {
            CanonicalRegionPostcard::All => Self::All,
            CanonicalRegionPostcard::Disc { x, y, radius } => Self::Disc { x, y, radius },
            CanonicalRegionPostcard::Rect { x, y, w, h } => Self::Rect { x, y, w, h },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum CanonicalCohortSourcePostcard {
    RegisteredBrain { key: u64 },
}

impl From<CohortSource> for CanonicalCohortSourcePostcard {
    fn from(s: CohortSource) -> Self {
        match s {
            CohortSource::RegisteredBrain { key } => Self::RegisteredBrain { key },
        }
    }
}

impl From<CanonicalCohortSourcePostcard> for CohortSource {
    fn from(c: CanonicalCohortSourcePostcard) -> Self {
        match c {
            CanonicalCohortSourcePostcard::RegisteredBrain { key } => Self::RegisteredBrain { key },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
enum CanonicalPlacementPostcard {
    Seeded {
        region: CanonicalRegionPostcard,
        seed: u64,
    },
}

impl From<Placement> for CanonicalPlacementPostcard {
    fn from(p: Placement) -> Self {
        match p {
            Placement::Seeded { region, seed } => Self::Seeded {
                region: region.into(),
                seed,
            },
        }
    }
}

impl From<CanonicalPlacementPostcard> for Placement {
    fn from(c: CanonicalPlacementPostcard) -> Self {
        match c {
            CanonicalPlacementPostcard::Seeded { region, seed } => Self::Seeded {
                region: region.into(),
                seed,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum CanonicalInterventionPostcard {
    Drought {
        region: CanonicalRegionPostcard,
        ticks: u32,
        growth_scale: f32,
    },
    Embargo {
        region: CanonicalRegionPostcard,
        ticks: u32,
    },
    Bloom {
        region: CanonicalRegionPostcard,
        amount: f32,
    },
    Meteor {
        region: CanonicalRegionPostcard,
        lethality: f32,
        scorch: f32,
    },
    PaintTerrain {
        region: CanonicalRegionPostcard,
        terrain: TerrainKind,
        fertility_bias: Option<f32>,
    },
    InjectCohort {
        count: u16,
        genome: CanonicalCohortSourcePostcard,
        placement: CanonicalPlacementPostcard,
        herbivore_tendency: f32,
    },
    SetClosedWorld {
        closed: bool,
    },
    SpawnFromArchive {
        selector: CellSelector,
        placement: CanonicalPlacementPostcard,
        headroom: Option<u32>,
        herbivore_tendency: Option<f32>,
    },
}

impl From<&Intervention> for CanonicalInterventionPostcard {
    fn from(i: &Intervention) -> Self {
        match i {
            Intervention::Drought {
                region,
                ticks,
                growth_scale,
            } => Self::Drought {
                region: (*region).into(),
                ticks: *ticks,
                growth_scale: *growth_scale,
            },
            Intervention::Embargo { region, ticks } => Self::Embargo {
                region: (*region).into(),
                ticks: *ticks,
            },
            Intervention::Bloom { region, amount } => Self::Bloom {
                region: (*region).into(),
                amount: *amount,
            },
            Intervention::Meteor {
                region,
                lethality,
                scorch,
            } => Self::Meteor {
                region: (*region).into(),
                lethality: *lethality,
                scorch: *scorch,
            },
            Intervention::PaintTerrain {
                region,
                terrain,
                fertility_bias,
            } => Self::PaintTerrain {
                region: (*region).into(),
                terrain: *terrain,
                fertility_bias: *fertility_bias,
            },
            Intervention::InjectCohort {
                count,
                genome,
                placement,
                herbivore_tendency,
            } => Self::InjectCohort {
                count: *count,
                genome: (*genome).into(),
                placement: (*placement).into(),
                herbivore_tendency: *herbivore_tendency,
            },
            Intervention::SetClosedWorld { closed } => Self::SetClosedWorld { closed: *closed },
            Intervention::SpawnFromArchive {
                selector,
                placement,
                headroom,
                herbivore_tendency,
            } => Self::SpawnFromArchive {
                selector: selector.clone(),
                placement: (*placement).into(),
                headroom: *headroom,
                herbivore_tendency: *herbivore_tendency,
            },
        }
    }
}

impl From<CanonicalInterventionPostcard> for Intervention {
    fn from(c: CanonicalInterventionPostcard) -> Self {
        match c {
            CanonicalInterventionPostcard::Drought {
                region,
                ticks,
                growth_scale,
            } => Self::Drought {
                region: region.into(),
                ticks,
                growth_scale,
            },
            CanonicalInterventionPostcard::Embargo { region, ticks } => Self::Embargo {
                region: region.into(),
                ticks,
            },
            CanonicalInterventionPostcard::Bloom { region, amount } => Self::Bloom {
                region: region.into(),
                amount,
            },
            CanonicalInterventionPostcard::Meteor {
                region,
                lethality,
                scorch,
            } => Self::Meteor {
                region: region.into(),
                lethality,
                scorch,
            },
            CanonicalInterventionPostcard::PaintTerrain {
                region,
                terrain,
                fertility_bias,
            } => Self::PaintTerrain {
                region: region.into(),
                terrain,
                fertility_bias,
            },
            CanonicalInterventionPostcard::InjectCohort {
                count,
                genome,
                placement,
                herbivore_tendency,
            } => Self::InjectCohort {
                count,
                genome: genome.into(),
                placement: placement.into(),
                herbivore_tendency,
            },
            CanonicalInterventionPostcard::SetClosedWorld { closed } => {
                Self::SetClosedWorld { closed }
            }
            CanonicalInterventionPostcard::SpawnFromArchive {
                selector,
                placement,
                headroom,
                herbivore_tendency,
            } => Self::SpawnFromArchive {
                selector,
                placement: placement.into(),
                headroom,
                herbivore_tendency,
            },
        }
    }
}

/// Canonical serialization of intervention parameters via Postcard.
///
/// Serialized externally tagged (no `#[serde(tag)]`): internal tagging cannot
/// be decoded by `postcard` at runtime without self-describing field names.
///
/// # Panics
///
/// Panics if postcard serialization of the canonical `Intervention` enum fails.
#[must_use]
pub fn canonical_param_bytes(intervention: &Intervention) -> Vec<u8> {
    let canonical = CanonicalInterventionPostcard::from(intervention);
    postcard::to_stdvec(&canonical).expect("postcard serialization for canonical Intervention enum")
}

/// Deserialization of canonical Postcard parameter bytes back to an Intervention.
pub fn intervention_from_param_bytes(bytes: &[u8]) -> Result<Intervention, postcard::Error> {
    postcard::from_bytes::<CanonicalInterventionPostcard>(bytes).map(Into::into)
}

/// Hex serialization helper for canonical binary parameter bytes.
pub mod hex_bytes {
    use super::{Deserializer, Serializer};
    use serde::Deserialize;

    /// Serialize raw parameter bytes as a lowercase hex string.
    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut hex = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            use std::fmt::Write;
            let _ = write!(hex, "{b:02x}");
        }
        serializer.serialize_str(&hex)
    }

    /// Deserialize a lowercase hex string into raw parameter bytes.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.len() % 2 != 0 {
            return Err(serde::de::Error::custom("hex string has odd length"));
        }
        let mut bytes = Vec::with_capacity(s.len() / 2);
        for i in (0..s.len()).step_by(2) {
            let byte = u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|e| serde::de::Error::custom(format!("invalid hex character: {e}")))?;
            bytes.push(byte);
        }
        Ok(bytes)
    }
}

/// Durable journal row for an intervention request, outcome, and effect (bd-16g.10.2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterventionJournalRow {
    /// Monotonic total-ordered intervention sequence.
    pub seq: u64,
    /// Simulation tick at which the intervention was processed.
    pub tick_applied: u64,
    /// Originating control surface.
    pub surface: InterventionSurface,
    /// Submitting actor identity.
    pub actor: String,
    /// Canonical `snake_case` kind label.
    pub kind: String,
    /// Canonical Postcard serialized bytes of the Intervention.
    #[serde(with = "hex_bytes")]
    pub params: Vec<u8>,
    /// Whether the command was accepted and queued for application.
    pub accepted: bool,
    /// Actionable refusal reason if rejected.
    #[serde(default)]
    pub rejection_reason: Option<String>,
    /// Agents affected by the intervention at application.
    pub agents_affected: u32,
    /// Food or terrain cells affected by the intervention.
    pub cells_affected: u32,
    /// Active timed effect identifier, if one was instantiated.
    #[serde(default)]
    pub effect_id: Option<u64>,
}

impl InterventionJournalRow {
    /// Construct a journal row for an intervention, writing structured audit logs.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        seq: u64,
        tick_applied: u64,
        surface: InterventionSurface,
        actor: impl Into<String>,
        intervention: &Intervention,
        accepted: bool,
        rejection_reason: Option<String>,
        agents_affected: u32,
        cells_affected: u32,
        effect_id: Option<u64>,
    ) -> Self {
        let actor = actor.into();
        let kind = intervention.kind_label().to_string();
        let params = canonical_param_bytes(intervention);

        diag_info!(
            target: "scriptbots::intervention::journal",
            seq,
            tick = tick_applied,
            kind = kind.as_str(),
            surface = %surface,
            actor = actor.as_str(),
            accepted,
            rejection_reason = rejection_reason.as_deref(),
            "intervention journal write"
        );

        Self {
            seq,
            tick_applied,
            surface,
            actor,
            kind,
            params,
            accepted,
            rejection_reason,
            agents_affected,
            cells_affected,
            effect_id,
        }
    }

    /// Decode the canonical intervention from the Postcard params bytes.
    pub fn decode_intervention(&self) -> Result<Intervention, postcard::Error> {
        intervention_from_param_bytes(&self.params)
    }
}

/// Header metadata for an intervention journal or script file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterventionJournalHeader {
    /// Schema version for the journal file format (currently 1).
    pub version: u32,
    /// World random seed.
    pub seed: u64,
    /// Characterization or configuration digest representing world setup.
    pub config_digest: String,
    /// Compile-lane build identity.
    pub build_identity: String,
}

impl InterventionJournalHeader {
    /// Construct a header capturing seed, config digest, and the current build identity.
    #[must_use]
    pub fn new(seed: u64, config_digest: impl Into<String>) -> Self {
        let build = CoreBuildIdentityV0::current();
        let build_identity = format!(
            "{}-{}-{}-p{}",
            build.target_arch, build.target_os, build.target_endian, build.pointer_width
        );
        Self {
            version: 1,
            seed,
            config_digest: config_digest.into(),
            build_identity,
        }
    }
}

/// An intervention journal or executable script artifact.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterventionJournal {
    /// Header metadata.
    pub header: Option<InterventionJournalHeader>,
    /// Chronologically ordered journal rows.
    pub rows: Vec<InterventionJournalRow>,
}

/// Typed errors on malformed, backwards, unknown, or diverging intervention journals.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InterventionJournalError {
    /// Journal file or string contained no readable content.
    #[error("missing or empty journal content")]
    EmptyJournal,
    /// Row JSON could not be parsed.
    #[error("row {row_index}: malformed JSON row in field `{field}`: {detail}")]
    MalformedRow {
        /// 1-based row index in journal document.
        row_index: usize,
        /// Named field failing decode.
        field: String,
        /// Underlying error description.
        detail: String,
    },
    /// Intervention kind is unrecognized.
    #[error("row {row_index}: unknown intervention kind `{kind}`")]
    UnknownKind {
        /// 1-based row index in journal document.
        row_index: usize,
        /// Unrecognized kind label.
        kind: String,
    },
    /// Simulation tick ran backwards between rows.
    #[error("row {row_index}: tick_applied ran backwards from {prev_tick} to {current_tick}")]
    NonMonotonicTick {
        /// 1-based row index in journal document.
        row_index: usize,
        /// Preceding tick.
        prev_tick: u64,
        /// Erroneous earlier tick.
        current_tick: u64,
    },
    /// Postcard parameter bytes failed to decode into an Intervention.
    #[error("row {row_index}: invalid param postcard payload: {detail}")]
    InvalidParams {
        /// 1-based row index in journal document.
        row_index: usize,
        /// Underlying error description.
        detail: String,
    },
    /// Header line failed validation.
    #[error("header error: {detail}")]
    InvalidHeader {
        /// Error description.
        detail: String,
    },
    /// Random seed mismatch between journal and world.
    #[error(
        "seed mismatch: journal recorded seed {expected}, but target world configured with seed {actual}"
    )]
    SeedMismatch {
        /// Expected seed from journal header.
        expected: u64,
        /// Actual configured world seed.
        actual: u64,
    },
    /// Build identity mismatch between recorded artifact and runner.
    #[error(
        "build identity mismatch: recorded on build `{recorded}`, current build is `{current}`"
    )]
    BuildMismatch {
        /// Recorded build identity string.
        recorded: String,
        /// Running build identity string.
        current: String,
    },
    /// State divergence between original run and replay.
    #[error(
        "first divergence at tick {tick}: expected digest {expected_digest}, actual digest {actual_digest}, last applied seq {last_applied_seq:?}, last applied kind {last_applied_kind:?}"
    )]
    Divergence {
        /// Tick where divergence was observed.
        tick: u64,
        /// Expected characterization digest.
        expected_digest: String,
        /// Observed characterization digest.
        actual_digest: String,
        /// Sequence of last applied intervention before divergence.
        last_applied_seq: Option<u64>,
        /// Kind of last applied intervention before divergence.
        last_applied_kind: Option<String>,
    },
}

/// Failures detected by the run-level journal completeness check.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JournalCompletenessError {
    /// The number of applied interventions in the world differs from the accepted rows in the journal.
    #[error(
        "count mismatch: world applied {applied_in_world} interventions, but journal contains {journaled_accepted} accepted rows"
    )]
    CountMismatch {
        /// Count recorded in world's applied audit ring.
        applied_in_world: usize,
        /// Count of accepted rows in journal.
        journaled_accepted: usize,
    },
    /// An applied intervention in the world has no corresponding journal row.
    #[error(
        "applied intervention seq {seq} at tick {tick} of kind `{kind}` has no corresponding journal row"
    )]
    MissingJournalRow {
        /// Sequence number of applied intervention.
        seq: u64,
        /// Tick of applied intervention.
        tick: u64,
        /// Kind of applied intervention.
        kind: String,
    },
    /// An applied intervention in the world mismatches its journal row attributes.
    #[error(
        "mismatch on world seq {seq} vs journal seq {journal_seq}: world has kind `{world_kind}` affecting {world_agents} agents and {world_cells} cells, but journal has kind `{journal_kind}` affecting {journal_agents} agents and {journal_cells} cells"
    )]
    RowMismatch {
        /// Sequence number with mismatch.
        seq: u64,
        /// Sequence number in journal.
        journal_seq: u64,
        /// World kind label.
        world_kind: String,
        /// World agents count.
        world_agents: usize,
        /// World cells count.
        world_cells: usize,
        /// Journal kind label.
        journal_kind: String,
        /// Journal agents count.
        journal_agents: u32,
        /// Journal cells count.
        journal_cells: u32,
    },
    /// An intervention marked rejected in the journal was found in the world's applied records.
    #[error("rejected journal row seq {seq} of kind `{kind}` was applied to the world")]
    RejectedRowWasApplied {
        /// Sequence number of rejected row.
        seq: u64,
        /// Kind of rejected row.
        kind: String,
    },
}

impl InterventionJournal {
    /// Construct a fresh journal for a simulation run.
    #[must_use]
    pub fn new(seed: u64, config_digest: impl Into<String>) -> Self {
        Self {
            header: Some(InterventionJournalHeader::new(seed, config_digest)),
            rows: Vec::new(),
        }
    }

    /// Append one journal row.
    pub fn push(&mut self, row: InterventionJournalRow) {
        self.rows.push(row);
    }

    /// Serialize this journal into line-delimited JSON (JSONL).
    pub fn to_json_lines(&self) -> Result<String, serde_json::Error> {
        let mut lines = Vec::new();
        if let Some(header) = &self.header {
            #[derive(Serialize)]
            struct HeaderLine<'a> {
                #[serde(rename = "type")]
                line_type: &'static str,
                #[serde(flatten)]
                header: &'a InterventionJournalHeader,
            }
            let hl = HeaderLine {
                line_type: "header",
                header,
            };
            lines.push(serde_json::to_string(&hl)?);
        }
        for row in &self.rows {
            #[derive(Serialize)]
            struct RowLine<'a> {
                #[serde(rename = "type")]
                line_type: &'static str,
                #[serde(flatten)]
                row: &'a InterventionJournalRow,
            }
            let rl = RowLine {
                line_type: "row",
                row,
            };
            lines.push(serde_json::to_string(&rl)?);
        }
        Ok(lines.join("\n"))
    }

    /// Parse a journal from line-delimited JSON with strict typing and diagnostics.
    pub fn from_json_lines(text: &str) -> Result<Self, InterventionJournalError> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(InterventionJournalError::EmptyJournal);
        }

        let mut header = None;
        let mut rows = Vec::new();
        let mut prev_tick = 0_u64;

        for (line_idx, line) in text.lines().enumerate() {
            let row_index = line_idx + 1;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let value: serde_json::Value =
                serde_json::from_str(line).map_err(|e| InterventionJournalError::MalformedRow {
                    row_index,
                    field: "json".to_owned(),
                    detail: e.to_string(),
                })?;

            let obj = value
                .as_object()
                .ok_or_else(|| InterventionJournalError::MalformedRow {
                    row_index,
                    field: "root".to_owned(),
                    detail: "expected JSON object".to_owned(),
                })?;

            let line_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or_else(|| {
                if obj.contains_key("version") && obj.contains_key("seed") {
                    "header"
                } else {
                    "row"
                }
            });
            if line_type == "header" {
                let h: InterventionJournalHeader = serde_json::from_value(value).map_err(|e| {
                    InterventionJournalError::InvalidHeader {
                        detail: e.to_string(),
                    }
                })?;
                header = Some(h);
            } else {
                let row: InterventionJournalRow = serde_json::from_value(value).map_err(|e| {
                    InterventionJournalError::MalformedRow {
                        row_index,
                        field: "row".to_owned(),
                        detail: e.to_string(),
                    }
                })?;

                match row.kind.as_str() {
                    "drought" | "embargo" | "bloom" | "meteor" | "paint_terrain"
                    | "set_closed_world" | "inject_cohort" | "spawn_from_archive" => {}
                    _ => {
                        return Err(InterventionJournalError::UnknownKind {
                            row_index,
                            kind: row.kind.clone(),
                        });
                    }
                }

                if !rows.is_empty() && row.tick_applied < prev_tick {
                    return Err(InterventionJournalError::NonMonotonicTick {
                        row_index,
                        prev_tick,
                        current_tick: row.tick_applied,
                    });
                }
                prev_tick = row.tick_applied;

                if let Err(e) = row.decode_intervention() {
                    return Err(InterventionJournalError::InvalidParams {
                        row_index,
                        detail: e.to_string(),
                    });
                }

                rows.push(row);
            }
        }

        Ok(Self { header, rows })
    }

    /// Write journal content to a file.
    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let content = self
            .to_json_lines()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)
    }

    /// Read and decode a journal from a file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, InterventionJournalError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| InterventionJournalError::InvalidHeader {
                detail: format!("failed to read file: {e}"),
            })?;
        Self::from_json_lines(&content)
    }
}

/// Order simultaneous interventions arriving at the same tick deterministically.
pub fn sort_interventions_deterministically(items: &mut [InterventionCommand]) {
    items.sort_by(|a, b| {
        let kind_a = a.intervention.kind_label();
        let kind_b = b.intervention.kind_label();
        let params_a = canonical_param_bytes(&a.intervention);
        let params_b = canonical_param_bytes(&b.intervention);
        kind_a
            .cmp(kind_b)
            .then_with(|| params_a.cmp(&params_b))
            .then_with(|| a.surface.cmp(&b.surface))
            .then_with(|| a.actor.cmp(&b.actor))
    });
}

/// Run-level invariant verifying that every applied intervention has exactly one
/// journal row, and every accepted journal row corresponds to an applied intervention.
pub fn verify_journal_completeness(
    world: &WorldState,
    journal: &InterventionJournal,
) -> Result<(), JournalCompletenessError> {
    let applied: Vec<_> = world
        .applied_interventions()
        .iter()
        .filter(|r| !r.expired)
        .collect();
    let accepted_rows: Vec<&InterventionJournalRow> =
        journal.rows.iter().filter(|r| r.accepted).collect();

    if applied.len() != accepted_rows.len() {
        return Err(JournalCompletenessError::CountMismatch {
            applied_in_world: applied.len(),
            journaled_accepted: accepted_rows.len(),
        });
    }

    for (record, row) in applied.iter().zip(&accepted_rows) {
        if record.seq != row.seq {
            return Err(JournalCompletenessError::RowMismatch {
                seq: record.seq,
                journal_seq: row.seq,
                world_kind: record.kind.to_string(),
                world_agents: record.agents_affected,
                world_cells: record.cells_affected,
                journal_kind: row.kind.clone(),
                journal_agents: row.agents_affected,
                journal_cells: row.cells_affected,
            });
        }
        if record.kind != row.kind {
            return Err(JournalCompletenessError::RowMismatch {
                seq: record.seq,
                journal_seq: row.seq,
                world_kind: record.kind.to_string(),
                world_agents: record.agents_affected,
                world_cells: record.cells_affected,
                journal_kind: row.kind.clone(),
                journal_agents: row.agents_affected,
                journal_cells: row.cells_affected,
            });
        }
    }

    let applied_seqs: std::collections::HashSet<u64> = applied.iter().map(|r| r.seq).collect();
    for row in &journal.rows {
        if !row.accepted && applied_seqs.contains(&row.seq) {
            return Err(JournalCompletenessError::RejectedRowWasApplied {
                seq: row.seq,
                kind: row.kind.clone(),
            });
        }
    }

    Ok(())
}

/// Summary of a successful journal replay run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplaySummary {
    /// Final tick reached.
    pub ticks: u64,
    /// Number of intervention commands applied.
    pub commands_applied: usize,
    /// Number of intervention commands rejected.
    pub commands_rejected: usize,
    /// Final characterization digest string.
    pub final_digest: String,
}

/// Replay an intervention journal against a deterministic world.
///
/// Fails loudly on first divergence with expected digest, actual digest, and last applied command.
#[allow(clippy::too_many_lines)]
pub fn replay_journal_against_world(
    world: &mut WorldState,
    journal: &InterventionJournal,
    target_ticks: u64,
    journal_path_display: &str,
) -> Result<ReplaySummary, InterventionJournalError> {
    let header = journal
        .header
        .as_ref()
        .ok_or(InterventionJournalError::EmptyJournal)?;

    if let Some(configured_seed) = world.config().rng_seed
        && configured_seed != header.seed
    {
        return Err(InterventionJournalError::SeedMismatch {
            expected: header.seed,
            actual: configured_seed,
        });
    }

    let current_build = CoreBuildIdentityV0::current();
    let current_build_str = format!(
        "{}-{}-{}-p{}",
        current_build.target_arch,
        current_build.target_os,
        current_build.target_endian,
        current_build.pointer_width
    );
    if header.build_identity != current_build_str {
        diag_warn!(
            target: "scriptbots::intervention::journal",
            recorded = header.build_identity.as_str(),
            current = current_build_str.as_str(),
            "Replay build mismatch: recorded build differs from current binary"
        );
    }

    diag_info!(
        target: "scriptbots::intervention::journal",
        journal_path = journal_path_display,
        rows = journal.rows.len(),
        seed = header.seed,
        config_digest = header.config_digest.as_str(),
        build = header.build_identity.as_str(),
        effects_pending = world.active_effects().len(),
        "intervention replay start"
    );

    let mut rows_by_tick: std::collections::BTreeMap<u64, Vec<&InterventionJournalRow>> =
        std::collections::BTreeMap::new();
    for row in &journal.rows {
        rows_by_tick.entry(row.tick_applied).or_default().push(row);
    }

    let mut commands_applied = 0_usize;
    let mut commands_rejected = 0_usize;
    let mut last_applied_seq = None;
    let mut last_applied_kind = None;

    let start_tick = world.tick().0;
    for tick in start_tick..target_ticks {
        if let Some(rows) = rows_by_tick.get(&tick) {
            for row in rows {
                let intervention = row.decode_intervention().map_err(|e| {
                    InterventionJournalError::InvalidParams {
                        row_index: usize::try_from(row.seq).unwrap_or(0),
                        detail: e.to_string(),
                    }
                })?;

                if row.accepted {
                    world.enqueue_intervention(intervention).map_err(|e| {
                        InterventionJournalError::Divergence {
                            tick,
                            expected_digest: "accepted".to_owned(),
                            actual_digest: format!("enqueue error: {e}"),
                            last_applied_seq,
                            last_applied_kind: last_applied_kind.clone(),
                        }
                    })?;
                    commands_applied += 1;
                    last_applied_seq = Some(row.seq);
                    last_applied_kind = Some(row.kind.clone());
                } else {
                    let validation = intervention.validate();
                    if validation.is_ok() {
                        let actual_digest = world
                            .characterization_digest_v0()
                            .map_or_else(|e| format!("err:{e}"), |d| d.overall);
                        diag_error!(
                            target: "scriptbots::intervention::journal",
                            tick,
                            expected_digest = "rejection",
                            actual_digest = actual_digest.as_str(),
                            last_applied_seq,
                            last_applied_kind = last_applied_kind.as_deref(),
                            "intervention replay divergence: expected rejection was accepted"
                        );
                        return Err(InterventionJournalError::Divergence {
                            tick,
                            expected_digest: "rejection".to_owned(),
                            actual_digest,
                            last_applied_seq,
                            last_applied_kind,
                        });
                    }
                    commands_rejected += 1;
                }
            }
        }

        world
            .step()
            .map_err(|e| InterventionJournalError::Divergence {
                tick,
                expected_digest: "step ok".to_owned(),
                actual_digest: format!("step failed: {e}"),
                last_applied_seq,
                last_applied_kind: last_applied_kind.clone(),
            })?;
    }

    let final_digest = world
        .characterization_digest_v0()
        .map_or_else(|e| format!("digest_err:{e}"), |d| d.overall);

    diag_info!(
        target: "scriptbots::intervention::journal",
        ticks = world.tick().0,
        commands_applied,
        commands_rejected,
        final_digest = final_digest.as_str(),
        "intervention replay success"
    );

    Ok(ReplaySummary {
        ticks: world.tick().0,
        commands_applied,
        commands_rejected,
        final_digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ScriptBotsConfig, WorldState};

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

    #[test]
    fn test_drought_duration_zero_is_rejected() {
        let record = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::Drought {
                region: ToroidalRegion::All,
                duration_ticks: 0,
            },
            issued_by: "REST".into(),
        };
        let err = record
            .validate()
            .expect_err("drought with duration 0 must be rejected");
        assert!(err.contains("drought duration must be > 0"));
    }

    #[test]
    fn test_predator_injection_count_bounds_are_rejected() {
        let zero_count = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::PredatorInjection {
                count: 0,
                position: Vec2::new(10.0, 10.0),
            },
            issued_by: "REST".into(),
        };
        let err_zero = zero_count
            .validate()
            .expect_err("predator injection count 0 must be rejected");
        assert!(err_zero.contains("predator injection count must be between 1 and 1000"));

        let excess_count = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::PredatorInjection {
                count: 1001,
                position: Vec2::new(10.0, 10.0),
            },
            issued_by: "REST".into(),
        };
        let err_excess = excess_count
            .validate()
            .expect_err("predator injection count 1001 must be rejected");
        assert!(err_excess.contains("predator injection count must be between 1 and 1000"));

        let boundary_low = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::PredatorInjection {
                count: 1,
                position: Vec2::new(10.0, 10.0),
            },
            issued_by: "REST".into(),
        };
        assert!(boundary_low.validate().is_ok());

        let boundary_high = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::PredatorInjection {
                count: 1000,
                position: Vec2::new(10.0, 10.0),
            },
            issued_by: "REST".into(),
        };
        assert!(boundary_high.validate().is_ok());
    }

    #[test]
    fn test_food_embargo_duration_zero_is_rejected() {
        let zero_embargo = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::FoodEmbargo { duration_ticks: 0 },
            issued_by: "REST".into(),
        };
        let err = zero_embargo
            .validate()
            .expect_err("food embargo with duration 0 must be rejected");
        assert!(err.contains("food embargo duration must be > 0"));

        let valid_embargo = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::FoodEmbargo { duration_ticks: 20 },
            issued_by: "REST".into(),
        };
        assert!(valid_embargo.validate().is_ok());
    }

    #[test]
    fn test_non_finite_rect_bounds_are_rejected() {
        let nan_min = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::TerrainPaint {
                region: ToroidalRegion::Rect {
                    min: Vec2::new(f32::NAN, 0.0),
                    max: Vec2::new(100.0, 100.0),
                },
                terrain_kind: 1,
            },
            issued_by: "REST".into(),
        };
        let err_nan = nan_min
            .validate()
            .expect_err("NaN rect bound must be rejected");
        assert!(err_nan.contains("rect bounds must be finite"));

        let inf_max = InterventionRecord {
            tick: Tick(100),
            action: InterventionAction::TerrainPaint {
                region: ToroidalRegion::Rect {
                    min: Vec2::new(0.0, 0.0),
                    max: Vec2::new(100.0, f32::INFINITY),
                },
                terrain_kind: 1,
            },
            issued_by: "REST".into(),
        };
        let err_inf = inf_max
            .validate()
            .expect_err("infinite rect bound must be rejected");
        assert!(err_inf.contains("rect bounds must be finite"));
    }

    #[test]
    fn test_canonical_postcard_serialization_and_roundtrip() {
        let cases = vec![
            drought(Region::All, 100, 0.5).unwrap(),
            embargo(
                Region::Disc {
                    x: 50.0,
                    y: 50.0,
                    radius: 25.0,
                },
                50,
            )
            .unwrap(),
            bloom(
                Region::Rect {
                    x: 10.0,
                    y: 10.0,
                    w: 20.0,
                    h: 30.0,
                },
                1000.0,
            )
            .unwrap(),
            meteor(
                Region::Disc {
                    x: 100.0,
                    y: 100.0,
                    radius: 40.0,
                },
                500.0,
                0.8,
            )
            .unwrap(),
            paint_terrain(Region::All, TerrainKind::Grass, Some(0.8)).unwrap(),
            inject_cohort(
                10,
                CohortSource::RegisteredBrain { key: 0 },
                Placement::Seeded {
                    region: Region::All,
                    seed: 123,
                },
                0.75,
            )
            .unwrap(),
            set_closed_world(true).unwrap(),
            spawn_from_archive(
                CellSelector::TopKByQuality(5),
                Placement::Seeded {
                    region: Region::All,
                    seed: 456,
                },
                Some(20),
                Some(0.2),
            )
            .unwrap(),
        ];

        for orig in cases {
            let bytes = canonical_param_bytes(&orig);
            assert!(!bytes.is_empty(), "canonical bytes must not be empty");
            let restored = intervention_from_param_bytes(&bytes)
                .expect("canonical bytes must deserialize to identical intervention");
            assert_eq!(orig, restored);

            // Verify hex roundtrip
            let mut hex_buf = Vec::new();
            let mut serializer = serde_json::Serializer::new(&mut hex_buf);
            hex_bytes::serialize(&bytes, &mut serializer).unwrap();
            let hex_json = String::from_utf8(hex_buf).unwrap();

            let mut deserializer = serde_json::Deserializer::from_str(&hex_json);
            let restored_bytes = hex_bytes::deserialize(&mut deserializer).unwrap();
            assert_eq!(bytes, restored_bytes);
        }
    }

    #[test]
    fn test_simultaneous_ingress_ordering_100_randomized_interleavings() {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;

        let baseline_commands = vec![
            InterventionCommand {
                intervention: drought(Region::All, 50, 0.2).unwrap(),
                surface: InterventionSurface::Rest,
                actor: "rest_client".into(),
            },
            InterventionCommand {
                intervention: drought(Region::All, 50, 0.2).unwrap(),
                surface: InterventionSurface::Mcp,
                actor: "mcp_client".into(),
            },
            InterventionCommand {
                intervention: embargo(Region::All, 10).unwrap(),
                surface: InterventionSurface::Cli,
                actor: "operator".into(),
            },
            InterventionCommand {
                intervention: bloom(Region::All, 100.0).unwrap(),
                surface: InterventionSurface::Gpu,
                actor: "user_click".into(),
            },
            InterventionCommand {
                intervention: meteor(
                    Region::Disc {
                        x: 50.0,
                        y: 50.0,
                        radius: 10.0,
                    },
                    100.0,
                    0.5,
                )
                .unwrap(),
                surface: InterventionSurface::Tui,
                actor: "key_press".into(),
            },
            InterventionCommand {
                intervention: set_closed_world(true).unwrap(),
                surface: InterventionSurface::Script,
                actor: "batch_run".into(),
            },
        ];

        let mut canonical_order = baseline_commands.clone();
        sort_interventions_deterministically(&mut canonical_order);

        // 100 randomized interleavings must sort to the EXACT same canonical order
        for seed in 0..100 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut shuffled = baseline_commands.clone();
            shuffled.shuffle(&mut rng);

            sort_interventions_deterministically(&mut shuffled);
            assert_eq!(
                shuffled, canonical_order,
                "interleaving with seed {seed} failed deterministic sort equality"
            );
        }
    }

    #[test]
    fn test_verify_journal_completeness_positive_and_negative() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).unwrap();
        let cmd = drought(Region::All, 10, 0.5).unwrap();
        world.enqueue_intervention(cmd.clone()).unwrap();
        world.step().unwrap();

        let applied_record = world.applied_interventions().front().unwrap();
        let row = InterventionJournalRow {
            seq: applied_record.seq,
            tick_applied: applied_record.tick.0,
            surface: InterventionSurface::Rest,
            actor: "test".into(),
            kind: applied_record.kind.to_string(),
            params: canonical_param_bytes(&cmd),
            accepted: true,
            rejection_reason: None,
            agents_affected: u32::try_from(applied_record.agents_affected).unwrap(),
            cells_affected: u32::try_from(applied_record.cells_affected).unwrap(),
            effect_id: None,
        };

        let journal = InterventionJournal {
            header: Some(InterventionJournalHeader {
                version: 1,
                seed: 42,
                config_digest: "test".into(),
                build_identity: "test".into(),
            }),
            rows: vec![row.clone()],
        };

        // Positive check
        assert!(verify_journal_completeness(&world, &journal).is_ok());

        // Negative check: count mismatch (extra row in journal)
        let mut bad_journal = journal.clone();
        bad_journal.rows.push(row);
        assert!(matches!(
            verify_journal_completeness(&world, &bad_journal),
            Err(JournalCompletenessError::CountMismatch { .. })
        ));

        // Negative check: row mismatch (kind changed)
        let mut bad_kind_journal = journal;
        bad_kind_journal.rows[0].kind = "meteor".into();
        assert!(matches!(
            verify_journal_completeness(&world, &bad_kind_journal),
            Err(JournalCompletenessError::RowMismatch { .. })
        ));
    }

    #[test]
    fn test_hostile_and_malformed_journals() {
        // Truncated row JSON line
        let malformed_json = "{\"version\":1,\"seed\":42,\"config_digest\":\"a\",\"build_identity\":\"b\"}\n{\"seq\":0,\"tick\":1";
        assert!(matches!(
            InterventionJournal::from_json_lines(malformed_json),
            Err(InterventionJournalError::MalformedRow { .. })
        ));

        // Unknown kind
        let unknown_kind = "{\"version\":1,\"seed\":42,\"config_digest\":\"a\",\"build_identity\":\"b\"}\n{\"seq\":0,\"tick_applied\":1,\"surface\":\"rest\",\"actor\":\"a\",\"kind\":\"teleport_to_mars\",\"params\":\"\",\"accepted\":true,\"agents_affected\":0,\"cells_affected\":0}";
        assert!(matches!(
            InterventionJournal::from_json_lines(unknown_kind),
            Err(InterventionJournalError::UnknownKind { .. })
        ));

        // Non-monotonic tick
        let drought_cmd = drought(Region::All, 10, 0.5).unwrap();
        let valid_params_bytes = canonical_param_bytes(&drought_cmd);
        let mut hex_params = String::new();
        for b in &valid_params_bytes {
            use std::fmt::Write;
            let _ = write!(hex_params, "{b:02x}");
        }
        let row1 = format!(
            "{{\"seq\":0,\"tick_applied\":10,\"surface\":\"rest\",\"actor\":\"a\",\"kind\":\"drought\",\"params\":\"{hex_params}\",\"accepted\":true,\"agents_affected\":0,\"cells_affected\":0}}"
        );
        let row2 = format!(
            "{{\"seq\":1,\"tick_applied\":5,\"surface\":\"rest\",\"actor\":\"a\",\"kind\":\"drought\",\"params\":\"{hex_params}\",\"accepted\":true,\"agents_affected\":0,\"cells_affected\":0}}"
        );
        let non_monotonic = format!(
            "{{\"version\":1,\"seed\":42,\"config_digest\":\"a\",\"build_identity\":\"b\"}}\n{row1}\n{row2}"
        );
        assert!(matches!(
            InterventionJournal::from_json_lines(&non_monotonic),
            Err(InterventionJournalError::NonMonotonicTick { .. })
        ));

        // Invalid params
        let bad_params_row = "{\"seq\":0,\"tick_applied\":1,\"surface\":\"rest\",\"actor\":\"a\",\"kind\":\"drought\",\"params\":\"ffff\",\"accepted\":true,\"agents_affected\":0,\"cells_affected\":0}";
        let bad_params = format!(
            "{{\"version\":1,\"seed\":42,\"config_digest\":\"a\",\"build_identity\":\"b\"}}\n{bad_params_row}"
        );
        assert!(matches!(
            InterventionJournal::from_json_lines(&bad_params),
            Err(InterventionJournalError::InvalidParams { .. })
        ));
    }

    #[test]
    fn test_replay_with_rejection_and_divergence() {
        let config = ScriptBotsConfig {
            rng_seed: Some(42),
            ..Default::default()
        };
        let mut world = WorldState::new(config.clone()).unwrap();

        // Enqueue an accepted intervention
        let valid_cmd = drought(Region::All, 10, 0.5).unwrap();
        world.enqueue_intervention(valid_cmd.clone()).unwrap();
        world.step().unwrap();

        let mut journal = InterventionJournal::new(42, "config_hash");
        journal.push(InterventionJournalRow::record(
            0,
            0,
            InterventionSurface::Rest,
            "test_actor",
            &valid_cmd,
            true,
            None,
            0,
            0,
            None,
        ));

        // Record a rejected intervention (e.g. invalid duration 0)
        let invalid_cmd = Intervention::Drought {
            region: Region::All,
            ticks: 0,
            growth_scale: 0.5,
        };
        journal.push(InterventionJournalRow::record(
            1,
            0,
            InterventionSurface::Cli,
            "operator",
            &invalid_cmd,
            false,
            Some("drought duration must be positive".to_string()),
            0,
            0,
            None,
        ));

        // Replay against fresh world with same seed
        let mut replay_world = WorldState::new(config).unwrap();
        let summary = replay_journal_against_world(&mut replay_world, &journal, 2, "test_journal")
            .expect("replay must succeed with identical seed and valid rejections");
        assert_eq!(summary.commands_applied, 1);
        assert_eq!(summary.commands_rejected, 1);

        // Negative check: wrong seed fails loudly
        let wrong_seed_config = ScriptBotsConfig {
            rng_seed: Some(999),
            ..Default::default()
        };
        let mut wrong_seed_world = WorldState::new(wrong_seed_config).unwrap();
        assert!(matches!(
            replay_journal_against_world(&mut wrong_seed_world, &journal, 2, "test_journal"),
            Err(InterventionJournalError::SeedMismatch { .. })
        ));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_5000_ticks_replay_proof_with_expired_and_active_effects_and_negative_control() {
        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 20,
            initial_food: 0.5,
            food_growth_rate: 0.05,
            rng_seed: Some(0x0016_1002),
            ..Default::default()
        };

        let mut world = WorldState::new(config.clone()).unwrap();
        let mut journal = InterventionJournal::new(
            0x0016_1002,
            world
                .characterization_digest_v0()
                .expect("initial digest")
                .overall,
        );

        let interventions: Vec<(u64, Intervention, InterventionSurface, &'static str)> = vec![
            (
                100,
                drought(Region::All, 400, 0.1).unwrap(),
                InterventionSurface::Rest,
                "operator_rest",
            ),
            (
                800,
                meteor(
                    Region::Disc {
                        x: 50.0,
                        y: 50.0,
                        radius: 20.0,
                    },
                    25.0,
                    0.6,
                )
                .unwrap(),
                InterventionSurface::Mcp,
                "claude_mcp",
            ),
            (
                1500,
                bloom(
                    Region::Rect {
                        x: 20.0,
                        y: 20.0,
                        w: 40.0,
                        h: 40.0,
                    },
                    50.0,
                )
                .unwrap(),
                InterventionSurface::Cli,
                "cli_batch",
            ),
            (
                4200,
                drought(Region::All, 1200, 0.0).unwrap(),
                InterventionSurface::Gpu,
                "gui_user",
            ),
            (
                4600,
                embargo(
                    Region::Disc {
                        x: 30.0,
                        y: 30.0,
                        radius: 15.0,
                    },
                    600,
                )
                .unwrap(),
                InterventionSurface::Tui,
                "tui_operator",
            ),
        ];

        let mut next_intervention_idx = 0;

        for tick in 0..5000 {
            let mut applied_this_tick = Vec::new();
            while next_intervention_idx < interventions.len()
                && interventions[next_intervention_idx].0 == tick
            {
                let (_, ref intervention, surface, actor) = interventions[next_intervention_idx];
                world.enqueue_intervention(intervention.clone()).unwrap();
                applied_this_tick.push((intervention.clone(), surface, actor));
                next_intervention_idx += 1;
            }

            world.step().unwrap();

            for (intervention, surface, actor) in applied_this_tick {
                let applied = world
                    .applied_interventions()
                    .iter()
                    .rev()
                    .find(|r| !r.expired && r.kind == intervention.kind_label())
                    .unwrap();
                journal.push(InterventionJournalRow::record(
                    applied.seq,
                    tick,
                    surface,
                    actor,
                    &intervention,
                    true,
                    None,
                    u32::try_from(applied.agents_affected).unwrap(),
                    u32::try_from(applied.cells_affected).unwrap(),
                    None,
                ));
            }
        }

        // Add a rejected command to the journal (seq from world counter, rejected before submission)
        let rejected_cmd = Intervention::Drought {
            region: Region::All,
            ticks: 0,
            growth_scale: 0.5,
        };
        journal.push(InterventionJournalRow::record(
            world.next_intervention_seq(),
            4900,
            InterventionSurface::Rest,
            "bad_actor",
            &rejected_cmd,
            false,
            Some("drought duration must be positive".to_string()),
            0,
            0,
            None,
        ));

        // 1. Verify journal completeness on original run
        verify_journal_completeness(&world, &journal)
            .expect("original run journal must be complete");

        // 2. Verify active effects count at tick 5000
        // Interventions at 100 (400 ticks -> ends 500) lapsed.
        // Interventions at 4200 (1200 ticks -> ends 5400) and 4600 (600 ticks -> ends 5200) are ACTIVE.
        assert_eq!(
            world.active_effects().len(),
            2,
            "exactly 2 timed effects must remain active at tick 5000"
        );

        let original_digest = world
            .characterization_digest_v0()
            .expect("original final digest");

        // 3. Positive replay against fresh world from journal
        let mut replay_world = WorldState::new(config.clone()).unwrap();
        let summary = replay_journal_against_world(
            &mut replay_world,
            &journal,
            world.tick().0,
            "5000_tick_proof_journal",
        )
        .expect("replay must succeed bit-identically");

        assert_eq!(summary.commands_applied, 5);
        assert_eq!(summary.commands_rejected, 1);
        assert_eq!(replay_world.active_effects().len(), 2);

        let replay_digest = replay_world
            .characterization_digest_v0()
            .expect("replay final digest");
        assert_eq!(
            original_digest, replay_digest,
            "replayed world characterization digest must match original bit-identically"
        );

        // 4. Negative Control A: Tamper with parameter byte -> turns replay RED
        let mut tampered_journal = journal.clone();
        let last_byte_idx = tampered_journal.rows[0].params.len() - 1;
        tampered_journal.rows[0].params[last_byte_idx] ^= 0xFF;
        let mut tampered_world = WorldState::new(config.clone()).unwrap();
        let tamper_res = replay_journal_against_world(
            &mut tampered_world,
            &tampered_journal,
            world.tick().0,
            "tampered_journal",
        );
        assert!(
            tamper_res.is_err(),
            "tampered parameter bytes must cause replay failure"
        );

        // 5. Negative Control B: Tamper with seed -> turns replay RED
        let bad_seed_config = ScriptBotsConfig {
            rng_seed: Some(0xDEAD),
            ..config
        };
        let mut bad_seed_world = WorldState::new(bad_seed_config).unwrap();
        let seed_res = replay_journal_against_world(
            &mut bad_seed_world,
            &journal,
            world.tick().0,
            "seed_mismatch_journal",
        );
        assert!(matches!(
            seed_res,
            Err(InterventionJournalError::SeedMismatch { .. })
        ));
    }
}
