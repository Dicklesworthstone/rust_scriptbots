//! MAP-Elites behavioral archive, Quality-Diversity (QD) metrics, and novelty search (bd-16g.6.1).
//!
//! This module provides:
//! - Versioned behavior space definition ([`BehaviorSpaceV0`]) and axes ([`Axis`]).
//! - Discretization into half-open bins and checked mixed-radix [`CellId`] packing.
//! - Behavior descriptor extraction from accumulated lifetime statistics ([`BehaviorDescriptor`], [`AgentAccumulatedStats`]).
//! - Strict deterministic [`MapElitesArchive`] backed by [`std::collections::BTreeMap`].
//! - Lifetime eligibility filtering to prevent newborn carpet-bombing.
//! - Finite-quality validation, strictly-better / lower-UID tie-breaking replacement rules.
//! - Configurable cell and byte capacity limits with explicit error returns (no silent eviction).
//! - Zero RNG draws: the archive is purely an observer.

use crate::{AgentData, AgentRuntime, AgentUid, BrainGenomeEnvelope, Generation, Tick};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Canonical schema version for behavior space definition V0.
pub const BEHAVIOR_SPACE_SCHEMA_VERSION_V0: u16 = 0;

/// Maximum number of behavioral dimensions packed into [`CellId`].
pub const MAX_BEHAVIOR_DIMENSIONS: usize = 8;

/// Hard ceiling on total cells permitted in a single behavior space (1,000,000 cells).
pub const MAX_ARCHIVE_CELLS: u64 = 1_000_000;

/// Default minimum lifetime ticks an agent must survive to be eligible for the archive.
pub const DEFAULT_MIN_LIFETIME_TICKS: u32 = 200;

/// Default cadence interval in simulation ticks for archive evaluation.
pub const DEFAULT_ARCHIVE_INTERVAL: u32 = 100;

/// Default memory capacity budget in bytes for stored archive entries (64 MiB).
pub const DEFAULT_MAX_ARCHIVE_BYTES: usize = 64 * 1024 * 1024;

/// Errors arising from Quality-Diversity behavior space and archive operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum QdError {
    /// Behavior space has zero axes.
    #[error("behavior space has no axes (dimension D = 0)")]
    EmptySpace,
    /// Dimension exceeds the maximum allowed.
    #[error("behavior space dimension {dim} exceeds maximum allowed dimension {max}")]
    DimensionExceeded {
        /// Configured dimension.
        dim: usize,
        /// Maximum allowed dimension.
        max: usize,
    },
    /// Axis domain is invalid (e.g. non-finite or `lo >= hi`).
    #[error(
        "axis '{name}' has invalid domain [{lo}, {hi}]: lower bound must be strictly less than upper bound and finite"
    )]
    InvalidDomain {
        /// Axis name.
        name: String,
        /// Lower bound.
        lo: f32,
        /// Upper bound.
        hi: f32,
    },
    /// Axis has 0 bins.
    #[error("axis '{name}' has 0 bins")]
    ZeroBins {
        /// Axis name.
        name: String,
    },
    /// Input descriptor dimension does not match behavior space.
    #[error("descriptor dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Descriptor contains a non-finite value.
    #[error("descriptor value for axis '{name}' (index {index}) is non-finite: {value}")]
    NonFiniteValue {
        /// Axis name.
        name: String,
        /// Axis index.
        index: usize,
        /// Value observed.
        value: f32,
    },
    /// Candidate quality is non-finite.
    #[error("candidate quality is non-finite: {value}")]
    NonFiniteQuality {
        /// Quality value.
        value: f32,
    },
    /// Mixed-radix calculation overflowed integer representation.
    #[error("mixed-radix calculation overflowed for cell index")]
    MixedRadixOverflow,
    /// Total grid capacity exceeds configured maximum.
    #[error("total grid cells {total_cells} exceeds maximum allowed capacity {max_cells}")]
    CellCapacityExceeded {
        /// Total cell count calculated from axis bin counts.
        total_cells: u64,
        /// Maximum allowable cell count.
        max_cells: u64,
    },
    /// Memory byte cap would be exceeded by this insertion.
    #[error(
        "archive byte size {current_bytes} + {entry_bytes} exceeds configured byte cap {cap_bytes}"
    )]
    ByteCapExceeded {
        /// Current bytes tracked.
        current_bytes: usize,
        /// Entry bytes to insert.
        entry_bytes: usize,
        /// Configured cap.
        cap_bytes: usize,
    },
    /// Agent is ineligible for archive evaluation due to insufficient lifetime.
    #[error(
        "agent {agent_uid:?} lifetime {lifetime_ticks} is less than minimum eligible ticks {min_lifetime_ticks}"
    )]
    IneligibleAgent {
        /// UID of agent.
        agent_uid: AgentUid,
        /// Observed lifetime in simulation ticks.
        lifetime_ticks: u32,
        /// Minimum lifetime in ticks required.
        min_lifetime_ticks: u32,
    },
    /// Novelty k-NN parameter k must be strictly positive.
    #[error("novelty k-NN parameter k must be non-zero")]
    ZeroK,
    /// Curiosity weight w is invalid (must be finite and in [0.0, 1.0]).
    #[error("curiosity weight w must be finite and within [0.0, 1.0], got {w}")]
    InvalidCuriosityWeight {
        /// Configured weight.
        w: f32,
    },
    /// Serialization or deserialization error.
    #[error("archive serialization error: {0}")]
    Serialization(String),
}

/// Errors occurring when diffing two MAP-Elites archives.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QdDiffError {
    /// Behavior space schema versions differ between archives.
    #[error(
        "behavior space version mismatch: archive A has version {a}, archive B has version {b}"
    )]
    SpaceVersionMismatch {
        /// Version in archive A.
        a: u16,
        /// Version in archive B.
        b: u16,
    },
    /// Quality metrics differ between archives.
    #[error("quality metric mismatch: archive A uses {a:?}, archive B uses {b:?}")]
    QualityMetricMismatch {
        /// Metric in archive A.
        a: QualityMetric,
        /// Metric in archive B.
        b: QualityMetric,
    },
    /// Axis counts differ between archives.
    #[error("behavior space axis count mismatch: archive A has {a} axes, archive B has {b} axes")]
    AxisCountMismatch {
        /// Count in archive A.
        a: usize,
        /// Count in archive B.
        b: usize,
    },
    /// Specific axis configuration differs between archives.
    #[error(
        "behavior space axis mismatch at index {index}: archive A has '{a_name}', archive B has '{b_name}'"
    )]
    AxisMismatch {
        /// Mismatched index.
        index: usize,
        /// Name in archive A.
        a_name: String,
        /// Name in archive B.
        b_name: String,
    },
    /// Archive deserialization error encountered during diff comparison.
    #[error("archive deserialization error: {0}")]
    Serialization(String),
}

/// Canonical phenotype features used as behavioral axes (bd-2z0.11.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhenotypeFeature {
    /// Agent dietary tendency (carnivore 0.0 .. herbivore 1.0).
    DietTendency,
    /// Mean locomotion velocity magnitude.
    MeanSpeed,
    /// Rate of combat spike extension.
    SpikeUsageRate,
    /// Rate of food sharing / altruistic giving.
    GiveRate,
    /// Rate of sound emission.
    SoundUsage,
    /// Mean absolute heading change per tick.
    TurnRate,
    /// Mean sensory radius modifier.
    SensingMean,
    /// Monotonic offspring production rate.
    OffspringRate,
}

impl PhenotypeFeature {
    /// Canonical stable schema identifier for this feature (bd-2z0.11.2).
    #[must_use]
    pub const fn canonical_id(&self) -> &'static str {
        match self {
            Self::DietTendency => "diet.herbivore_trait.mean",
            Self::MeanSpeed => "movement.speed.mean",
            Self::SpikeUsageRate => "interaction.combat.spike_rate",
            Self::GiveRate => "interaction.share.actor_rate",
            Self::SoundUsage => "sensing.sound.usage_rate",
            Self::TurnRate => "movement.turn.rate",
            Self::SensingMean => "sensing.trait_modifier.mean",
            Self::OffspringRate => "lineage.offspring.parent_rate",
        }
    }

    /// Physical or mathematical unit for this feature.
    #[must_use]
    pub const fn unit(&self) -> &'static str {
        match self {
            Self::DietTendency => "ratio",
            Self::MeanSpeed => "world_unit_per_tick",
            Self::SpikeUsageRate | Self::GiveRate => "event_per_tick",
            Self::SoundUsage => "unit_per_tick",
            Self::TurnRate => "radian_per_tick",
            Self::SensingMean => "trait_multiplier",
            Self::OffspringRate => "edge_per_tick",
        }
    }
}

/// Definition of a single behavioral dimension / axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Axis {
    /// Descriptive name of the behavioral axis.
    pub name: String,
    /// Underlying canonical phenotype feature.
    pub feature: PhenotypeFeature,
    /// Valid domain `(lo, hi)` with `lo < hi`.
    pub domain: (f32, f32),
    /// Number of uniform discrete bins along this axis (`1..=u8::MAX`).
    pub bins: u8,
}

impl Axis {
    /// Construct a new axis definition and validate its parameters.
    pub fn new(
        name: impl Into<String>,
        feature: PhenotypeFeature,
        domain: (f32, f32),
        bins: u8,
    ) -> Result<Self, QdError> {
        let name = name.into();
        if bins == 0 {
            return Err(QdError::ZeroBins { name });
        }
        if !domain.0.is_finite() || !domain.1.is_finite() || domain.0 >= domain.1 {
            return Err(QdError::InvalidDomain {
                name,
                lo: domain.0,
                hi: domain.1,
            });
        }
        Ok(Self {
            name,
            feature,
            domain,
            bins,
        })
    }

    /// Validate the axis invariants.
    pub fn validate(&self) -> Result<(), QdError> {
        if self.bins == 0 {
            return Err(QdError::ZeroBins {
                name: self.name.clone(),
            });
        }
        if !self.domain.0.is_finite()
            || !self.domain.1.is_finite()
            || self.domain.0 >= self.domain.1
        {
            return Err(QdError::InvalidDomain {
                name: self.name.clone(),
                lo: self.domain.0,
                hi: self.domain.1,
            });
        }
        Ok(())
    }

    /// Discretize a scalar value into a 0-indexed bin in `[0, bins)`.
    ///
    /// Values equal to `domain.lo` map to bin `0`.
    /// Values equal to `domain.hi` map to the last bin (`bins - 1`).
    /// Values below `domain.lo` are clamped to `0`.
    /// Values above `domain.hi` are clamped to `bins - 1`.
    /// Non-finite values return [`QdError::NonFiniteValue`].
    pub fn discretize(&self, value: f32, axis_index: usize) -> Result<u8, QdError> {
        if !value.is_finite() {
            return Err(QdError::NonFiniteValue {
                name: self.name.clone(),
                index: axis_index,
                value,
            });
        }
        let lo = self.domain.0;
        let hi = self.domain.1;
        let bins = u32::from(self.bins);

        let clamped = value.clamp(lo, hi);
        if clamped >= hi {
            Ok(self.bins.saturating_sub(1))
        } else {
            let span = hi - lo;
            let frac = ((clamped - lo) / span).clamp(0.0, 1.0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let bin = (frac * f32::from(self.bins)).floor() as u32;
            let capped = bin.min(bins.saturating_sub(1));
            #[allow(clippy::cast_possible_truncation)]
            Ok(capped as u8)
        }
    }
}

/// Versioned behavior space definition holding an ordered list of axes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehaviorSpaceV0 {
    /// Schema version (fixed at [`BEHAVIOR_SPACE_SCHEMA_VERSION_V0`]).
    pub version: u16,
    /// Ordered behavioral axes (up to [`MAX_BEHAVIOR_DIMENSIONS`]).
    pub axes: Vec<Axis>,
}

impl Default for BehaviorSpaceV0 {
    /// Default 6-axis behavioral space using canonical phenotype features.
    fn default() -> Self {
        Self {
            version: BEHAVIOR_SPACE_SCHEMA_VERSION_V0,
            axes: vec![
                Axis {
                    name: "diet_tendency".to_string(),
                    feature: PhenotypeFeature::DietTendency,
                    domain: (0.0, 1.0),
                    bins: 5,
                },
                Axis {
                    name: "mean_speed".to_string(),
                    feature: PhenotypeFeature::MeanSpeed,
                    domain: (0.0, 5.0),
                    bins: 5,
                },
                Axis {
                    name: "spike_usage_rate".to_string(),
                    feature: PhenotypeFeature::SpikeUsageRate,
                    domain: (0.0, 1.0),
                    bins: 5,
                },
                Axis {
                    name: "give_rate".to_string(),
                    feature: PhenotypeFeature::GiveRate,
                    domain: (0.0, 1.0),
                    bins: 5,
                },
                Axis {
                    name: "sound_usage".to_string(),
                    feature: PhenotypeFeature::SoundUsage,
                    domain: (0.0, 1.0),
                    bins: 5,
                },
                Axis {
                    name: "turn_rate".to_string(),
                    feature: PhenotypeFeature::TurnRate,
                    domain: (0.0, std::f32::consts::PI),
                    bins: 5,
                },
            ],
        }
    }
}

impl BehaviorSpaceV0 {
    /// Create a new behavior space with explicit axes.
    #[must_use]
    pub const fn new(version: u16, axes: Vec<Axis>) -> Self {
        Self { version, axes }
    }

    /// Validate the behavior space against the default cell capacity ([`MAX_ARCHIVE_CELLS`]).
    pub fn validate(&self) -> Result<(), QdError> {
        self.validate_with_cap(MAX_ARCHIVE_CELLS)
    }

    /// Validate the behavior space against a specified cell capacity.
    pub fn validate_with_cap(&self, max_cells: u64) -> Result<(), QdError> {
        if self.axes.is_empty() {
            return Err(QdError::EmptySpace);
        }
        if self.axes.len() > MAX_BEHAVIOR_DIMENSIONS {
            return Err(QdError::DimensionExceeded {
                dim: self.axes.len(),
                max: MAX_BEHAVIOR_DIMENSIONS,
            });
        }
        for axis in &self.axes {
            axis.validate()?;
        }
        let total = self.total_cells()?;
        if total > max_cells {
            return Err(QdError::CellCapacityExceeded {
                total_cells: total,
                max_cells,
            });
        }
        Ok(())
    }

    /// Compute the total number of discrete cells in the behavior grid ($\prod \text{bins}_i$).
    pub fn total_cells(&self) -> Result<u64, QdError> {
        if self.axes.is_empty() {
            return Err(QdError::EmptySpace);
        }
        let mut total = 1u64;
        for axis in &self.axes {
            total = total
                .checked_mul(u64::from(axis.bins))
                .ok_or(QdError::MixedRadixOverflow)?;
        }
        Ok(total)
    }

    /// Compute the packed mixed-radix [`CellId`] for a given behavior descriptor.
    pub fn cell_index(&self, descriptor: &BehaviorDescriptor) -> Result<CellId, QdError> {
        if self.axes.is_empty() {
            return Err(QdError::EmptySpace);
        }
        if descriptor.0.len() != self.axes.len() {
            return Err(QdError::DimensionMismatch {
                expected: self.axes.len(),
                actual: descriptor.0.len(),
            });
        }

        let mut cell_id = 0u64;
        let mut multiplier = 1u64;
        for (i, (axis, &val)) in self.axes.iter().zip(&descriptor.0).enumerate() {
            let bin = u64::from(axis.discretize(val, i)?);
            let term = bin
                .checked_mul(multiplier)
                .ok_or(QdError::MixedRadixOverflow)?;
            cell_id = cell_id
                .checked_add(term)
                .ok_or(QdError::MixedRadixOverflow)?;
            multiplier = multiplier
                .checked_mul(u64::from(axis.bins))
                .ok_or(QdError::MixedRadixOverflow)?;
        }
        Ok(CellId(cell_id))
    }

    /// Unpack a [`CellId`] back into per-axis 0-indexed bin coordinates.
    pub fn decode_cell_coords(&self, cell_id: CellId) -> Result<Vec<u8>, QdError> {
        if self.axes.is_empty() {
            return Err(QdError::EmptySpace);
        }
        let mut remaining = cell_id.0;
        let mut coords = Vec::with_capacity(self.axes.len());
        for axis in &self.axes {
            let bins = u64::from(axis.bins);
            #[allow(clippy::cast_possible_truncation)]
            let coord = (remaining % bins) as u8;
            coords.push(coord);
            remaining /= bins;
        }
        Ok(coords)
    }
}

/// N-dimensional behavioral descriptor vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehaviorDescriptor(pub Vec<f32>);

impl BehaviorDescriptor {
    /// Construct a new behavior descriptor from a vector of continuous feature values.
    #[must_use]
    pub const fn new(values: Vec<f32>) -> Self {
        Self(values)
    }

    /// Construct a behavior descriptor from a borrowed slice.
    #[must_use]
    pub fn from_slice(slice: &[f32]) -> Self {
        Self(slice.to_vec())
    }

    /// Borrow the underlying feature values as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Dimension of this descriptor vector.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether this descriptor has zero dimensions.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Packed 64-bit integer identifier for a discrete cell in the behavior space.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(transparent)]
pub struct CellId(pub u64);

impl CellId {
    /// Return the raw integer value of this cell identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Provenance metadata for an individual elite in the archive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveProvenance {
    /// Run identifier under which this elite lived.
    pub run_id: String,
    /// UID of the primary parent agent, if any.
    pub parent_uid: Option<AgentUid>,
    /// Heritable generation of the agent.
    pub generation: Generation,
}

/// Record for an elite individual stored in a MAP-Elites grid cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchiveEntry {
    /// Stable logical identity of the agent.
    pub uid: AgentUid,
    /// Simulation tick when this elite was evaluated and inserted.
    pub tick_inserted: Tick,
    /// Behavior descriptor that placed this agent in this cell.
    pub descriptor: BehaviorDescriptor,
    /// Quality metric value evaluated for this agent.
    pub quality: f32,
    /// Versioned, bounded brain genome envelope.
    pub genome: BrainGenomeEnvelope,
    /// Historical provenance of this individual.
    pub provenance: ArchiveProvenance,
}

impl ArchiveEntry {
    /// Approximate memory consumption of this entry in bytes for memory budgeting.
    #[must_use]
    pub fn approximate_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.descriptor.0.len() * std::mem::size_of::<f32>())
            .saturating_add(self.genome.payload().len())
            .saturating_add(self.provenance.run_id.len())
    }
}

/// Configuration choice for how agent quality is measured in the archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum QualityMetric {
    /// Net lifetime food intake (`runtime.food_balance_total`). Default choice.
    #[default]
    LifetimeIntake,
    /// Total completed simulation ticks lived at evaluation time (`data.age`).
    AgeAtEvaluation,
    /// Number of offspring produced over lifetime.
    OffspringCount,
}

impl QualityMetric {
    /// Compute the quality value for an agent according to this metric.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "Archive quality is f32 for all metrics; preserving u32-to-f32 rounding retains the existing ranking and UID tie breaks above the exact-integer range"
    )]
    pub const fn compute(
        self,
        runtime: &AgentRuntime,
        data: &AgentData,
        stats: &AgentAccumulatedStats,
    ) -> f32 {
        match self {
            Self::LifetimeIntake => runtime.food_balance_total,
            Self::AgeAtEvaluation => data.age as f32,
            Self::OffspringCount => stats.offspring_count as f32,
        }
    }

    /// Return the canonical string identifier for this quality metric.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LifetimeIntake => "lifetime_intake",
            Self::AgeAtEvaluation => "age_at_evaluation",
            Self::OffspringCount => "offspring_count",
        }
    }
}

/// Result of attempting to insert a candidate into the MAP-Elites archive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InsertionResult {
    /// Candidate occupied a previously empty cell.
    InsertedNew,
    /// Candidate displaced an incumbent with strictly higher quality.
    ReplacedBetter {
        /// UID of the displaced incumbent.
        displaced_uid: AgentUid,
        /// Quality score of the displaced incumbent.
        displaced_quality: f32,
    },
    /// Candidate displaced an incumbent on an exact tie because it had a lower UID.
    ReplacedTieBreak {
        /// UID of the displaced incumbent.
        displaced_uid: AgentUid,
        /// Quality score of the displaced incumbent.
        displaced_quality: f32,
    },
    /// Candidate was rejected because the incumbent has higher or equal quality with lower UID.
    RejectedWorseOrEqual,
}

/// Compensated summation using Neumaier's variant of Kahan summation.
///
/// Handles large numbers added to small numbers without losing low-order bits,
/// ensuring high numerical stability and exact reproducibility across floating-point orders.
#[inline]
#[must_use]
pub fn neumaier_sum<I>(iter: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for val in iter {
        let t = sum + val;
        if sum.abs() >= val.abs() {
            c += (sum - t) + val;
        } else {
            c += (val - t) + sum;
        }
        sum = t;
    }
    sum + c
}

/// MAP-Elites behavioral grid archive.
///
/// Backed by [`BTreeMap<CellId, ArchiveEntry>`] to guarantee deterministic, sorted iteration
/// order across all machines and platforms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapElitesArchive {
    /// Behavior space defining the axes, resolution, and domains.
    pub space: BehaviorSpaceV0,
    /// Metric used to evaluate elite quality.
    pub quality_metric: QualityMetric,
    /// Minimum lifetime in ticks an agent must have survived to be eligible.
    pub min_lifetime_ticks: u32,
    /// Maximum allowed grid cells.
    pub max_archive_cells: u64,
    /// Maximum total memory bytes allowed for the archive.
    pub max_archive_bytes: usize,
    /// Current estimated memory byte consumption of stored entries.
    pub current_bytes: usize,
    /// Map of discretized cell coordinates to elite individual records.
    pub cells: BTreeMap<CellId, ArchiveEntry>,
    /// Internal flag to latch a one-time warning if >90% of agents are ineligible.
    #[serde(skip)]
    pub logged_eligibility_warning: bool,
}

impl MapElitesArchive {
    /// Construct a new MAP-Elites archive and validate its configuration.
    pub fn new(
        space: BehaviorSpaceV0,
        quality_metric: QualityMetric,
        min_lifetime_ticks: u32,
        max_archive_bytes: usize,
    ) -> Result<Self, QdError> {
        space.validate_with_cap(MAX_ARCHIVE_CELLS)?;
        let max_archive_cells = space.total_cells()?;
        Ok(Self {
            space,
            quality_metric,
            min_lifetime_ticks,
            max_archive_cells,
            max_archive_bytes,
            current_bytes: 0,
            cells: BTreeMap::new(),
            logged_eligibility_warning: false,
        })
    }

    /// Try inserting an elite candidate into the archive.
    ///
    /// Returns:
    /// - `Ok(InsertionResult::InsertedNew)` if cell was empty.
    /// - `Ok(InsertionResult::ReplacedBetter)` if candidate had strictly higher quality.
    /// - `Ok(InsertionResult::ReplacedTieBreak)` if candidate had equal quality but strictly lower UID.
    /// - `Ok(InsertionResult::RejectedWorseOrEqual)` if incumbent was retained.
    /// - `Err(QdError)` on validation failure, dimension mismatch, or byte capacity breach.
    pub fn insert(&mut self, entry: ArchiveEntry) -> Result<InsertionResult, QdError> {
        if !entry.quality.is_finite() {
            return Err(QdError::NonFiniteQuality {
                value: entry.quality,
            });
        }
        let cell_id = self.space.cell_index(&entry.descriptor)?;
        let entry_bytes = entry.approximate_bytes();

        match self.cells.get(&cell_id) {
            None => {
                if self.current_bytes.saturating_add(entry_bytes) > self.max_archive_bytes {
                    return Err(QdError::ByteCapExceeded {
                        current_bytes: self.current_bytes,
                        entry_bytes,
                        cap_bytes: self.max_archive_bytes,
                    });
                }
                let uid = entry.uid.get();
                let quality = entry.quality;
                let tick_inserted = entry.tick_inserted.0;
                self.current_bytes = self.current_bytes.saturating_add(entry_bytes);
                self.cells.insert(cell_id, entry);
                let binned_axes = self.space.decode_cell_coords(cell_id).unwrap_or_default();
                tracing::debug!(
                    target: "scriptbots::qd::archive",
                    tick = %tick_inserted,
                    cell_id = %cell_id.0,
                    binned_axes = ?binned_axes,
                    uid = %uid,
                    quality = %quality,
                    displaced_uid = tracing::field::Empty,
                    displaced_quality = tracing::field::Empty,
                    "archive inserted new elite"
                );
                Ok(InsertionResult::InsertedNew)
            }
            Some(incumbent) => {
                let ordering = entry.quality.total_cmp(&incumbent.quality);
                match ordering {
                    std::cmp::Ordering::Greater => {
                        let incumbent_bytes = incumbent.approximate_bytes();
                        let displaced_uid = incumbent.uid;
                        let displaced_quality = incumbent.quality;
                        let uid = entry.uid;
                        let quality = entry.quality;
                        let tick_inserted = entry.tick_inserted.0;
                        self.replace_incumbent(cell_id, entry, entry_bytes, incumbent_bytes)?;
                        let binned_axes =
                            self.space.decode_cell_coords(cell_id).unwrap_or_default();
                        tracing::debug!(
                            target: "scriptbots::qd::archive",
                            tick = %tick_inserted,
                            cell_id = %cell_id.0,
                            binned_axes = ?binned_axes,
                            uid = %uid.get(),
                            quality = %quality,
                            displaced_uid = %displaced_uid.get(),
                            displaced_quality = %displaced_quality,
                            "archive displaced elite with higher quality"
                        );
                        Ok(InsertionResult::ReplacedBetter {
                            displaced_uid,
                            displaced_quality,
                        })
                    }
                    std::cmp::Ordering::Equal if entry.uid < incumbent.uid => {
                        let incumbent_bytes = incumbent.approximate_bytes();
                        let displaced_uid = incumbent.uid;
                        let displaced_quality = incumbent.quality;
                        let uid = entry.uid;
                        let quality = entry.quality;
                        let tick_inserted = entry.tick_inserted.0;
                        self.replace_incumbent(cell_id, entry, entry_bytes, incumbent_bytes)?;
                        let binned_axes =
                            self.space.decode_cell_coords(cell_id).unwrap_or_default();
                        tracing::debug!(
                            target: "scriptbots::qd::archive",
                            tick = %tick_inserted,
                            cell_id = %cell_id.0,
                            binned_axes = ?binned_axes,
                            uid = %uid.get(),
                            quality = %quality,
                            displaced_uid = %displaced_uid.get(),
                            displaced_quality = %displaced_quality,
                            "archive replaced elite on tie with lower uid"
                        );
                        Ok(InsertionResult::ReplacedTieBreak {
                            displaced_uid,
                            displaced_quality,
                        })
                    }
                    _ => Ok(InsertionResult::RejectedWorseOrEqual),
                }
            }
        }
    }

    fn replace_incumbent(
        &mut self,
        cell_id: CellId,
        entry: ArchiveEntry,
        entry_bytes: usize,
        incumbent_bytes: usize,
    ) -> Result<(), QdError> {
        let net_delta = entry_bytes.saturating_sub(incumbent_bytes);
        if self.current_bytes.saturating_add(net_delta) > self.max_archive_bytes {
            return Err(QdError::ByteCapExceeded {
                current_bytes: self.current_bytes,
                entry_bytes: net_delta,
                cap_bytes: self.max_archive_bytes,
            });
        }
        if entry_bytes >= incumbent_bytes {
            self.current_bytes = self.current_bytes.saturating_add(net_delta);
        } else {
            self.current_bytes = self
                .current_bytes
                .saturating_sub(incumbent_bytes.saturating_sub(entry_bytes));
        }
        self.cells.insert(cell_id, entry);
        Ok(())
    }

    /// Compute raw Quality-Diversity (QD) score (compensated sum of all elite qualities).
    ///
    /// Summation is evaluated in strictly sorted [`CellId`] order using Neumaier compensated
    /// summation to eliminate numerical drift.
    #[must_use]
    pub fn qd_score_raw(&self) -> f64 {
        neumaier_sum(self.cells.values().map(|r| f64::from(r.quality)))
    }

    /// Compute Quality-Diversity (QD) score (sum of all elite qualities).
    ///
    /// Delegates to [`Self::qd_score_raw`] using Neumaier compensated summation in sorted [`CellId`] order.
    #[must_use]
    pub fn qd_score(&self) -> f64 {
        self.qd_score_raw()
    }

    /// Compute normalized Quality-Diversity (QD) score in `[0.0, 1.0]` (or non-negative).
    ///
    /// Defined as `qd_score_raw / total_cells`. Returns `0.0` if `total_cells == 0` or if the
    /// archive is empty, never NaN. For a full archive where every elite has quality 1.0,
    /// this evaluates to exactly 1.0.
    #[must_use]
    pub fn qd_score_norm(&self) -> f64 {
        let total = self.total_cells();
        if total == 0 || self.cells.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let norm = self.qd_score_raw() / total as f64;
            if norm.is_finite() { norm.max(0.0) } else { 0.0 }
        }
    }

    /// Number of distinct occupied cells in the archive.
    #[must_use]
    pub fn coverage_count(&self) -> usize {
        self.cells.len()
    }

    /// Number of occupied cells in the archive.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// True if no cells are occupied in the archive.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Retrieve an entry by cell ID if present.
    #[must_use]
    pub fn get(&self, cell_id: CellId) -> Option<&ArchiveEntry> {
        self.cells.get(&cell_id)
    }

    /// Total potential cells defined by the behavior space.
    #[must_use]
    pub fn total_cells(&self) -> u64 {
        self.space.total_cells().unwrap_or(0)
    }

    /// Percentage of total grid cells filled in `[0.0, 1.0]`.
    #[must_use]
    pub fn coverage_ratio(&self) -> f32 {
        let total = self.total_cells();
        if total == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let count = self.cells.len() as f64;
            #[allow(clippy::cast_precision_loss)]
            #[expect(
                clippy::cast_possible_truncation,
                reason = "The public coverage ratio is f32; retain the existing f64 division followed by narrowing instead of changing its rounding"
            )]
            let ratio = (count / total as f64) as f32;
            ratio.clamp(0.0, 1.0)
        }
    }

    /// Compute binary Shannon entropy of cell occupancy in bits `[0.0, 1.0]`.
    ///
    /// Given occupancy probability `p = coverage_count / total_cells`:
    /// `H(p) = -p * log2(p) - (1 - p) * log2(1 - p)`.
    ///
    /// Returns `0.0` for empty (`p = 0.0`) and full (`p = 1.0`) archives (zero uncertainty).
    /// Returns `1.0` for a half-full archive (`p = 0.5`, maximum uncertainty).
    /// Never produces NaN.
    #[must_use]
    pub fn occupancy_entropy(&self) -> f64 {
        let total = self.total_cells();
        if total == 0 || self.cells.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let count = self.cells.len() as f64;
        #[allow(clippy::cast_precision_loss)]
        let p = (count / total as f64).clamp(0.0, 1.0);
        if p <= 0.0 || p >= 1.0 {
            0.0
        } else {
            #[expect(
                clippy::suboptimal_flops,
                reason = "Explicit separate operations preserve exact binary entropy evaluation without architecture-dependent FMA variation"
            )]
            let h = -p * p.log2() - (1.0 - p) * (1.0 - p).log2();
            if h.is_finite() {
                h.clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
    }

    /// Arithmetic mean quality across all occupied cells.
    #[must_use]
    pub fn mean_quality(&self) -> f32 {
        if self.cells.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let mean = self.qd_score_raw() / self.cells.len() as f64;
            #[allow(clippy::cast_possible_truncation)]
            {
                mean as f32
            }
        }
    }

    /// Maximum quality observed in any cell.
    #[must_use]
    pub fn max_quality(&self) -> Option<f32> {
        self.cells
            .values()
            .map(|r| r.quality)
            .max_by(f32::total_cmp)
    }

    /// Sorted list of all occupied [`CellId`] keys.
    #[must_use]
    pub fn cell_ids_sorted(&self) -> Vec<CellId> {
        self.cells.keys().copied().collect()
    }

    /// Generate the comprehensive Quality-Diversity metrics report.
    #[must_use]
    pub fn metrics(&self) -> QdMetrics {
        QdMetrics {
            space_version: self.space.version,
            quality_metric: self.quality_metric,
            total_cells: self.total_cells(),
            occupied_cells: self.cells.len(),
            coverage: self.coverage_ratio(),
            qd_score_raw: self.qd_score_raw(),
            qd_score_norm: self.qd_score_norm(),
            occupancy_entropy: self.occupancy_entropy(),
            mean_quality: self.mean_quality(),
            max_quality: self.max_quality(),
        }
    }

    /// Compute the structured difference between this archive and another archive.
    pub fn diff(&self, other: &Self) -> Result<ArchiveDiff, QdDiffError> {
        archive_diff(self, other)
    }

    /// Select cell IDs according to a [`CellSelector`].
    ///
    /// The returned list contains only cell IDs actually present in the archive,
    /// sorted in deterministic order.
    #[must_use]
    pub fn select_cells(&self, selector: &CellSelector) -> Vec<CellId> {
        match selector {
            CellSelector::All => self.cell_ids_sorted(),
            CellSelector::TopKByQuality(k) => {
                let mut entries: Vec<(&CellId, &ArchiveEntry)> = self.cells.iter().collect();
                entries.sort_unstable_by(|(id_a, a), (id_b, b)| {
                    b.quality
                        .total_cmp(&a.quality)
                        .then_with(|| a.uid.cmp(&b.uid))
                        .then_with(|| id_a.cmp(id_b))
                });
                entries
                    .into_iter()
                    .take(*k as usize)
                    .map(|(&id, _)| id)
                    .collect()
            }
            CellSelector::AxisRange(ranges) => {
                let mut matched = Vec::new();
                for (&cell_id, entry) in &self.cells {
                    let mut in_range = true;
                    for &(axis_idx, lo, hi) in ranges {
                        if let Some(&val) = entry.descriptor.0.get(axis_idx) {
                            if val < lo || val > hi {
                                in_range = false;
                                break;
                            }
                        } else {
                            in_range = false;
                            break;
                        }
                    }
                    if in_range {
                        matched.push(cell_id);
                    }
                }
                matched
            }
            CellSelector::Explicit(ids) => ids
                .iter()
                .copied()
                .filter(|id| self.cells.contains_key(id))
                .collect(),
        }
    }

    /// Select archive entries according to a [`CellSelector`].
    #[must_use]
    pub fn select_entries(&self, selector: &CellSelector) -> Vec<&ArchiveEntry> {
        self.select_cells(selector)
            .into_iter()
            .filter_map(|id| self.get(id))
            .collect()
    }
}

/// Comprehensive Quality-Diversity (QD) metrics report for an archive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QdMetrics {
    /// Behavior space schema version.
    pub space_version: u16,
    /// Quality metric evaluated by this archive.
    pub quality_metric: QualityMetric,
    /// Total potential cells defined by the behavior space.
    pub total_cells: u64,
    /// Number of occupied cells in the archive.
    pub occupied_cells: usize,
    /// Ratio of occupied cells to total cells in `[0.0, 1.0]`.
    pub coverage: f32,
    /// Raw QD score (compensated sum of all elite qualities).
    pub qd_score_raw: f64,
    /// Normalized QD score (`qd_score_raw / total_cells`).
    pub qd_score_norm: f64,
    /// Binary Shannon occupancy entropy in bits `[0.0, 1.0]`.
    pub occupancy_entropy: f64,
    /// Arithmetic mean quality across occupied cells (`0.0` if empty).
    pub mean_quality: f32,
    /// Maximum quality observed across occupied cells (`None` if empty).
    pub max_quality: Option<f32>,
}

/// Selector specifying which archive cells are targeted for inspection or resurrection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CellSelector {
    /// Select every occupied cell in the archive.
    All,
    /// Select the top K cells ordered by quality descending (tie-broken by lower UID).
    TopKByQuality(u16),
    /// Select cells whose continuous feature coordinates fall within `[lo, hi]` on specified axes.
    AxisRange(Vec<(usize, f32, f32)>),
    /// Select specific cells by explicit [`CellId`] list.
    Explicit(Vec<CellId>),
}

/// Comparison detail for a single cell present in both archives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellComparison {
    /// Discretized cell identifier.
    pub cell_id: CellId,
    /// Quality of elite in archive A.
    pub quality_a: f32,
    /// Quality of elite in archive B.
    pub quality_b: f32,
    /// Quality delta (`quality_b - quality_a`).
    pub delta: f32,
}

/// Difference report between two MAP-Elites archives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchiveDiff {
    /// Cell IDs present only in archive A.
    pub only_in_a: Vec<CellId>,
    /// Cell IDs present only in archive B.
    pub only_in_b: Vec<CellId>,
    /// Cells present in both where archive B has strictly higher quality.
    pub improved_in_b: Vec<CellComparison>,
    /// Cells present in both where archive B has strictly lower quality.
    pub regressed_in_b: Vec<CellComparison>,
    /// Cells present in both where quality is exactly equal.
    pub unchanged: Vec<CellId>,
}

/// Compute the structured difference between two MAP-Elites archives.
///
/// Requires identical behavior space versions, axis definitions, and quality metrics.
/// Comparing archives across differing behavior spaces or quality metrics is a category error
/// and returns a typed [`QdDiffError`] rather than performing a silent join.
pub fn archive_diff(
    a: &MapElitesArchive,
    b: &MapElitesArchive,
) -> Result<ArchiveDiff, QdDiffError> {
    if a.space.version != b.space.version {
        return Err(QdDiffError::SpaceVersionMismatch {
            a: a.space.version,
            b: b.space.version,
        });
    }
    if a.quality_metric != b.quality_metric {
        return Err(QdDiffError::QualityMetricMismatch {
            a: a.quality_metric,
            b: b.quality_metric,
        });
    }
    if a.space.axes.len() != b.space.axes.len() {
        return Err(QdDiffError::AxisCountMismatch {
            a: a.space.axes.len(),
            b: b.space.axes.len(),
        });
    }
    for (i, (axis_a, axis_b)) in a.space.axes.iter().zip(&b.space.axes).enumerate() {
        if axis_a != axis_b {
            return Err(QdDiffError::AxisMismatch {
                index: i,
                a_name: axis_a.name.clone(),
                b_name: axis_b.name.clone(),
            });
        }
    }

    let mut only_in_a = Vec::new();
    let mut only_in_b = Vec::new();
    let mut improved_in_b = Vec::new();
    let mut regressed_in_b = Vec::new();
    let mut unchanged = Vec::new();

    let mut iter_a = a.cells.iter();
    let mut iter_b = b.cells.iter();
    let mut item_a = iter_a.next();
    let mut item_b = iter_b.next();

    while let (Some((&id_a, entry_a)), Some((&id_b, entry_b))) = (item_a, item_b) {
        match id_a.cmp(&id_b) {
            std::cmp::Ordering::Less => {
                only_in_a.push(id_a);
                item_a = iter_a.next();
            }
            std::cmp::Ordering::Greater => {
                only_in_b.push(id_b);
                item_b = iter_b.next();
            }
            std::cmp::Ordering::Equal => {
                let qa = entry_a.quality;
                let qb = entry_b.quality;
                let delta = qb - qa;
                match qb.total_cmp(&qa) {
                    std::cmp::Ordering::Greater => {
                        improved_in_b.push(CellComparison {
                            cell_id: id_a,
                            quality_a: qa,
                            quality_b: qb,
                            delta,
                        });
                    }
                    std::cmp::Ordering::Less => {
                        regressed_in_b.push(CellComparison {
                            cell_id: id_a,
                            quality_a: qa,
                            quality_b: qb,
                            delta,
                        });
                    }
                    std::cmp::Ordering::Equal => {
                        unchanged.push(id_a);
                    }
                }
                item_a = iter_a.next();
                item_b = iter_b.next();
            }
        }
    }

    while let Some((&id_a, _)) = item_a {
        only_in_a.push(id_a);
        item_a = iter_a.next();
    }
    while let Some((&id_b, _)) = item_b {
        only_in_b.push(id_b);
        item_b = iter_b.next();
    }

    Ok(ArchiveDiff {
        only_in_a,
        only_in_b,
        improved_in_b,
        regressed_in_b,
        unchanged,
    })
}

/// Format a clean, self-describing human-readable and structured text table of QD metrics.
#[must_use]
pub fn format_stats_report(archive: &MapElitesArchive, run_id: &str) -> String {
    let m = archive.metrics();
    let max_q_str = m
        .max_quality
        .map_or_else(|| "N/A".to_string(), |q| format!("{q:.4}"));
    format!(
        "=== MAP-Elites Behavioral Archive Metrics ===\n\
         Run ID:             {run_id}\n\
         Space Version:      {}\n\
         Quality Metric:     {}\n\
         Min Lifetime Ticks: {}\n\
         Occupied Cells:     {}\n\
         Total Cells:        {}\n\
         Coverage:           {:.4} ({:.2}%)\n\
         Raw QD Score:       {:.4}\n\
         Normalized QD:      {:.6}\n\
         Mean Quality:       {:.4}\n\
         Max Quality:        {max_q_str}\n\
         Occupancy Entropy:  {:.6} bits\n\
         Approx Bytes:       {} / {} bytes\n",
        m.space_version,
        m.quality_metric.as_str(),
        archive.min_lifetime_ticks,
        m.occupied_cells,
        m.total_cells,
        m.coverage,
        m.coverage * 100.0,
        m.qd_score_raw,
        m.qd_score_norm,
        m.mean_quality,
        m.occupancy_entropy,
        archive.current_bytes,
        archive.max_archive_bytes,
    )
}

/// Compute the structured difference between two serialized CSV archives.
pub fn diff_csv<R1: std::io::BufRead, R2: std::io::BufRead>(
    reader_a: R1,
    reader_b: R2,
    byte_cap: usize,
) -> Result<ArchiveDiff, QdDiffError> {
    let (_, archive_a) = MapElitesArchive::import_csv(reader_a, byte_cap)
        .map_err(|e| QdDiffError::Serialization(e.to_string()))?;
    let (_, archive_b) = MapElitesArchive::import_csv(reader_b, byte_cap)
        .map_err(|e| QdDiffError::Serialization(e.to_string()))?;
    archive_diff(&archive_a, &archive_b)
}

/// Compute the structured difference between two serialized JSON archives.
pub fn diff_json(json_a: &str, json_b: &str) -> Result<ArchiveDiff, QdDiffError> {
    let (_, archive_a) = MapElitesArchive::import_json(json_a)
        .map_err(|e| QdDiffError::Serialization(e.to_string()))?;
    let (_, archive_b) = MapElitesArchive::import_json(json_b)
        .map_err(|e| QdDiffError::Serialization(e.to_string()))?;
    archive_diff(&archive_a, &archive_b)
}

/// Accumulator tracking lifetime statistics for an agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentAccumulatedStats {
    /// Number of simulation ticks this agent was observed alive.
    pub ticks_observed: u32,
    /// Cumulative locomotion speed across observed ticks.
    pub speed_sum: f32,
    /// Number of ticks the combat spike was deployed.
    pub spiked_ticks: u32,
    /// Cumulative food sharing / altruistic intent.
    pub give_intent_sum: f32,
    /// Cumulative sound output emitted.
    pub sound_output_sum: f32,
    /// Cumulative heading turning angle in radians.
    pub turn_angle_sum: f32,
    /// Last observed heading angle.
    pub last_heading: Option<f32>,
    /// Cumulative food intake.
    pub food_intake_sum: f32,
    /// Total offspring produced.
    pub offspring_count: u32,
}

impl Default for AgentAccumulatedStats {
    fn default() -> Self {
        Self {
            ticks_observed: 0,
            speed_sum: 0.0,
            spiked_ticks: 0,
            give_intent_sum: 0.0,
            sound_output_sum: 0.0,
            turn_angle_sum: 0.0,
            last_heading: None,
            food_intake_sum: 0.0,
            offspring_count: 0,
        }
    }
}

impl AgentAccumulatedStats {
    /// Construct a fresh accumulator for a newly spawned or born agent.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record observations for one simulation step.
    pub fn record_tick(
        &mut self,
        speed: f32,
        heading: f32,
        spiked: bool,
        give_intent: f32,
        sound_output: f32,
        food_delta: f32,
    ) {
        self.ticks_observed = self.ticks_observed.saturating_add(1);
        if speed.is_finite() {
            self.speed_sum += speed.max(0.0);
        }
        if spiked {
            self.spiked_ticks = self.spiked_ticks.saturating_add(1);
        }
        if give_intent.is_finite() {
            self.give_intent_sum += give_intent.max(0.0);
        }
        if sound_output.is_finite() {
            self.sound_output_sum += sound_output.max(0.0);
        }
        if food_delta.is_finite() && food_delta > 0.0 {
            self.food_intake_sum += food_delta;
        }
        if heading.is_finite() {
            if let Some(prev) = self.last_heading {
                let diff = (heading - prev).abs();
                let wrapped = if diff > std::f32::consts::PI {
                    #[expect(
                        clippy::suboptimal_flops,
                        reason = "Retain the existing multiplication-then-subtraction form used for lifetime turn descriptors; lint cleanup preserves the scientific evaluation order"
                    )]
                    let remaining_turn = 2.0 * std::f32::consts::PI - diff;
                    remaining_turn.max(0.0)
                } else {
                    diff
                };
                self.turn_angle_sum += wrapped;
            }
            self.last_heading = Some(heading);
        }
    }

    /// Record that this agent successfully produced an offspring.
    pub const fn record_offspring(&mut self) {
        self.offspring_count = self.offspring_count.saturating_add(1);
    }

    /// Extract a single feature's accumulated average for behavior space mapping.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "Lifetime descriptors and sums use f32; keeping counter rounding before division preserves existing rate values and archive bin assignment"
    )]
    pub fn feature_value(&self, feature: PhenotypeFeature, runtime: &AgentRuntime) -> f32 {
        let obs = (self.ticks_observed as f32).max(1.0);
        match feature {
            PhenotypeFeature::DietTendency => runtime.herbivore_tendency,
            PhenotypeFeature::MeanSpeed => self.speed_sum / obs,
            PhenotypeFeature::SpikeUsageRate => self.spiked_ticks as f32 / obs,
            PhenotypeFeature::GiveRate => self.give_intent_sum / obs,
            PhenotypeFeature::SoundUsage => self.sound_output_sum / obs,
            PhenotypeFeature::TurnRate => self.turn_angle_sum / obs,
            PhenotypeFeature::SensingMean => {
                (runtime.trait_modifiers.smell
                    + runtime.trait_modifiers.sound
                    + runtime.trait_modifiers.hearing
                    + runtime.trait_modifiers.eye
                    + runtime.trait_modifiers.blood)
                    / 5.0
            }
            PhenotypeFeature::OffspringRate => self.offspring_count as f32 / obs,
        }
    }

    /// Compute the complete behavioral descriptor vector according to the configured space.
    pub fn compute_descriptor(
        &self,
        space: &BehaviorSpaceV0,
        runtime: &AgentRuntime,
    ) -> Result<BehaviorDescriptor, QdError> {
        let mut values = Vec::with_capacity(space.axes.len());
        for (i, axis) in space.axes.iter().enumerate() {
            let val = self.feature_value(axis.feature, runtime);
            if !val.is_finite() {
                return Err(QdError::NonFiniteValue {
                    name: axis.name.clone(),
                    index: i,
                    value: val,
                });
            }
            values.push(val);
        }
        Ok(BehaviorDescriptor(values))
    }
}

/// Compute k-NN novelty score for a candidate descriptor against the archive.
#[must_use]
pub fn compute_novelty_score(
    candidate: &BehaviorDescriptor,
    archive: &MapElitesArchive,
    k: usize,
) -> f32 {
    if archive.cells.is_empty() || k == 0 {
        return 0.0;
    }

    let mut distances: Vec<f32> = archive
        .cells
        .values()
        .map(|entry| {
            candidate
                .0
                .iter()
                .zip(&entry.descriptor.0)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .collect();

    distances.sort_by(f32::total_cmp);
    let take_k = k.min(distances.len());
    let sum: f32 = distances.iter().take(take_k).sum();
    #[allow(clippy::cast_precision_loss)]
    {
        sum / take_k as f32
    }
}

/// Evolution selection mode governing reproduction probability modulation (bd-16g.6.2).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum EvolutionSelectionMode {
    /// Standard Darwinian fitness-based selection (energy/food balance drives reproduction directly).
    /// Reproduction probability modulator is identically 1.0.
    #[default]
    Fitness,
    /// Novelty search selection: reproduction probability is modulated by normalized k-NN novelty
    /// in behavior space against the combined population and MAP-Elites behavioral archive.
    Novelty,
    /// Curiosity-driven selection: combines normalized novelty and normalized fitness with weight `w`:
    /// `score = w * novelty_norm + (1.0 - w) * fitness_norm` where `w` is in `[0.0, 1.0]`.
    Curiosity {
        /// Weight placed on novelty vs fitness in `[0.0, 1.0]`.
        w: f32,
    },
}

impl EvolutionSelectionMode {
    /// Validate evolution selection mode invariants.
    ///
    /// # Errors
    /// Returns [`QdError::InvalidCuriosityWeight`] if `w` in `Curiosity` is non-finite or outside `[0.0, 1.0]`.
    pub fn validate(&self) -> Result<(), QdError> {
        match self {
            Self::Fitness | Self::Novelty => Ok(()),
            Self::Curiosity { w } => {
                if !w.is_finite() || !(0.0..=1.0).contains(w) {
                    Err(QdError::InvalidCuriosityWeight { w: *w })
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Compute domain-normalized Euclidean distance between two behavior descriptors.
///
/// For each dimension `i`, the distance contribution is `((a[i] - b[i]) / (hi[i] - lo[i]))^2`.
/// Returns `Ok(distance)` or [`QdError`] on dimension mismatch or non-finite values.
///
/// # Errors
/// Returns [`QdError::DimensionMismatch`] if descriptor lengths do not match `space.axes.len()`,
/// or [`QdError::NonFiniteValue`] if any descriptor element is non-finite.
#[allow(clippy::suboptimal_flops)]
pub fn normalized_distance(
    a: &BehaviorDescriptor,
    b: &BehaviorDescriptor,
    space: &BehaviorSpaceV0,
) -> Result<f32, QdError> {
    if a.0.len() != space.axes.len() {
        return Err(QdError::DimensionMismatch {
            expected: space.axes.len(),
            actual: a.0.len(),
        });
    }
    if b.0.len() != space.axes.len() {
        return Err(QdError::DimensionMismatch {
            expected: space.axes.len(),
            actual: b.0.len(),
        });
    }
    let mut sum_sq = 0.0_f32;
    for (i, (axis, (&val_a, &val_b))) in space
        .axes
        .iter()
        .zip(a.0.iter().zip(b.0.iter()))
        .enumerate()
    {
        if !val_a.is_finite() {
            return Err(QdError::NonFiniteValue {
                name: axis.name.clone(),
                index: i,
                value: val_a,
            });
        }
        if !val_b.is_finite() {
            return Err(QdError::NonFiniteValue {
                name: axis.name.clone(),
                index: i,
                value: val_b,
            });
        }
        let span = axis.domain.1 - axis.domain.0;
        let diff = (val_a - val_b) / span;
        sum_sq += diff * diff;
    }
    Ok(sum_sq.sqrt())
}

/// Candidate agent behavior descriptor for population-wide novelty evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateDescriptor {
    /// Stable logical identity of the agent.
    pub uid: AgentUid,
    /// Behavior descriptor vector.
    pub descriptor: BehaviorDescriptor,
}

impl CandidateDescriptor {
    /// Construct a new candidate descriptor record.
    #[must_use]
    pub const fn new(uid: AgentUid, descriptor: BehaviorDescriptor) -> Self {
        Self { uid, descriptor }
    }
}

struct NeighborCandidate {
    distance: f32,
    uid: AgentUid,
    is_archive: bool,
    original_index: usize,
}

/// Compute k-NN novelty score in normalized behavior space for an entire population against
/// both other population members (self-excluded) and all archive entries (bd-16g.6.2).
///
/// Ties in distance are broken deterministically by neighbor UID, archive provenance,
/// and index order.
///
/// # Errors
/// Returns [`QdError::ZeroK`] if `k == 0`, [`QdError::EmptySpace`] if space has no axes,
/// or descriptor errors on dimension mismatch / non-finite values.
pub fn compute_population_novelty(
    population: &[CandidateDescriptor],
    archive: Option<&MapElitesArchive>,
    space: &BehaviorSpaceV0,
    k: usize,
) -> Result<Vec<(AgentUid, f32)>, QdError> {
    if k == 0 {
        return Err(QdError::ZeroK);
    }
    if space.axes.is_empty() {
        return Err(QdError::EmptySpace);
    }
    if population.is_empty() {
        return Ok(Vec::new());
    }

    // Pre-validate all population descriptors
    for cand in population {
        if cand.descriptor.0.len() != space.axes.len() {
            return Err(QdError::DimensionMismatch {
                expected: space.axes.len(),
                actual: cand.descriptor.0.len(),
            });
        }
        for (i, (axis, &val)) in space.axes.iter().zip(cand.descriptor.0.iter()).enumerate() {
            if !val.is_finite() {
                return Err(QdError::NonFiniteValue {
                    name: axis.name.clone(),
                    index: i,
                    value: val,
                });
            }
        }
    }

    let mut results = Vec::with_capacity(population.len());

    for cand in population {
        let mut neighbors = Vec::new();

        // 1. Other population members (self-excluded by UID)
        for (other_idx, other) in population.iter().enumerate() {
            if other.uid == cand.uid {
                continue;
            }
            let dist = normalized_distance(&cand.descriptor, &other.descriptor, space)?;
            neighbors.push(NeighborCandidate {
                distance: dist,
                uid: other.uid,
                is_archive: false,
                original_index: other_idx,
            });
        }

        // 2. Archive entries (self-excluded if an archive entry has the same UID)
        if let Some(arch) = archive {
            for (cell_idx, entry) in arch.cells.values().enumerate() {
                if entry.uid == cand.uid {
                    continue;
                }
                let dist = normalized_distance(&cand.descriptor, &entry.descriptor, space)?;
                neighbors.push(NeighborCandidate {
                    distance: dist,
                    uid: entry.uid,
                    is_archive: true,
                    original_index: cell_idx,
                });
            }
        }

        if neighbors.is_empty() {
            // A single agent with an empty archive scores 0.0 with no division by zero
            results.push((cand.uid, 0.0_f32));
            continue;
        }

        // Deterministic sorting with stable tie-breaking
        neighbors.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.uid.cmp(&b.uid))
                .then_with(|| a.is_archive.cmp(&b.is_archive))
                .then_with(|| a.original_index.cmp(&b.original_index))
        });

        // Clamp k to available neighbors (k > population-1 clamps to what exists)
        let take_k = k.min(neighbors.len());
        let sum: f32 = neighbors.iter().take(take_k).map(|n| n.distance).sum();
        #[expect(
            clippy::cast_precision_loss,
            reason = "take_k is bounded by population size which easily fits in f32 exact integer range"
        )]
        let mean = sum / (take_k as f32);
        results.push((cand.uid, mean));
    }

    Ok(results)
}

/// Normalize scores into `[0.0, 1.0]` across a population with documented degenerate semantics.
///
/// If `max > min`: `(score - min) / (max - min)`.
/// If `max == min`:
/// - if `max == 0.0`: returns `0.0` for all individuals (e.g. all-identical population or empty archive).
/// - if `max > 0.0`: returns `1.0` for all individuals.
#[must_use]
pub fn normalize_scores(scores: &[(AgentUid, f32)]) -> Vec<(AgentUid, f32)> {
    if scores.is_empty() {
        return Vec::new();
    }
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &(_, s) in scores {
        if s < min_val {
            min_val = s;
        }
        if s > max_val {
            max_val = s;
        }
    }
    if !min_val.is_finite() || !max_val.is_finite() {
        return scores.iter().map(|&(uid, _)| (uid, 0.0)).collect();
    }
    let range = max_val - min_val;
    if range > f32::EPSILON {
        scores
            .iter()
            .map(|&(uid, s)| {
                let norm = ((s - min_val) / range).clamp(0.0, 1.0);
                (uid, norm)
            })
            .collect()
    } else {
        let default_val = if max_val > 0.0 { 1.0 } else { 0.0 };
        scores.iter().map(|&(uid, _)| (uid, default_val)).collect()
    }
}

/// Combine normalized novelty and normalized fitness with weight `w` in `[0.0, 1.0]`.
///
/// `curiosity = w * novelty_norm + (1.0 - w) * fitness_norm`.
/// Returns [`QdError::InvalidCuriosityWeight`] if `w` is non-finite or outside `0.0..=1.0`.
///
/// # Errors
/// Returns [`QdError::InvalidCuriosityWeight`] if `w` is non-finite or outside `0.0..=1.0`.
#[allow(clippy::suboptimal_flops)]
pub fn combine_curiosity(
    w: f32,
    novelty_norm: &[(AgentUid, f32)],
    fitness_norm: &[(AgentUid, f32)],
) -> Result<Vec<(AgentUid, f32)>, QdError> {
    if !w.is_finite() || !(0.0..=1.0).contains(&w) {
        return Err(QdError::InvalidCuriosityWeight { w });
    }
    let fitness_map: BTreeMap<AgentUid, f32> = fitness_norm.iter().copied().collect();
    let mut results = Vec::with_capacity(novelty_norm.len());
    for &(uid, n) in novelty_norm {
        let f = fitness_map.get(&uid).copied().unwrap_or(0.0);
        let score = (w * n + (1.0 - w) * f).clamp(0.0, 1.0);
        results.push((uid, score));
    }
    Ok(results)
}

/// Cached novelty state across the live population (bd-16g.6.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoveltyState {
    /// Simulation tick when this novelty state was evaluated.
    pub last_recompute_tick: Tick,
    /// Raw novelty scores mapped by `AgentUid`.
    pub scores: BTreeMap<AgentUid, f32>,
    /// Normalized scores in `[0.0, 1.0]` mapped by `AgentUid`.
    pub normalized_scores: BTreeMap<AgentUid, f32>,
    /// Arithmetic mean novelty across the population.
    pub mean_novelty: f32,
    /// 50th percentile (median) novelty score.
    pub p50_novelty: f32,
    /// 95th percentile novelty score.
    pub p95_novelty: f32,
    /// Maximum novelty score observed.
    pub max_novelty: f32,
    /// Number of consecutive recompute samples where `mean_novelty == 0.0`.
    pub consecutive_zero_samples: u32,
}

impl NoveltyState {
    /// Construct a new `NoveltyState` from raw scores and normalized scores.
    #[must_use]
    pub fn new(
        tick: Tick,
        raw_scores: &[(AgentUid, f32)],
        normalized: &[(AgentUid, f32)],
        prev_consecutive_zeros: u32,
    ) -> Self {
        let mut scores_map = BTreeMap::new();
        let mut vals = Vec::with_capacity(raw_scores.len());
        for &(uid, s) in raw_scores {
            scores_map.insert(uid, s);
            vals.push(s);
        }
        vals.sort_by(f32::total_cmp);

        let (mean, p50, p95, max) = if vals.is_empty() {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let sum: f32 = vals.iter().sum();
            #[expect(clippy::cast_precision_loss)]
            let mean = sum / (vals.len() as f32);
            let max = vals[vals.len() - 1];

            let mid = vals.len() / 2;
            let p50 = if vals.len() % 2 == 1 {
                vals[mid]
            } else {
                f32::midpoint(vals[mid - 1], vals[mid])
            };

            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let p95_idx = (((vals.len() as f64) * 0.95).ceil() as usize)
                .saturating_sub(1)
                .min(vals.len() - 1);
            let p95 = vals[p95_idx];
            (mean, p50, p95, max)
        };

        let consecutive_zeros = if mean == 0.0 {
            prev_consecutive_zeros.saturating_add(1)
        } else {
            0
        };

        let normalized_map: BTreeMap<AgentUid, f32> = normalized.iter().copied().collect();

        Self {
            last_recompute_tick: tick,
            scores: scores_map,
            normalized_scores: normalized_map,
            mean_novelty: mean,
            p50_novelty: p50,
            p95_novelty: p95,
            max_novelty: max,
            consecutive_zero_samples: consecutive_zeros,
        }
    }
}

/// Row representation of the behavior space configuration for database persistence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveSpaceRow {
    /// Run identifier.
    pub run_id: String,
    /// Schema version of the behavior space.
    pub space_version: u16,
    /// Serialized JSON string of the axis definitions.
    pub axes_json: String,
}

/// Row representation of an individual archive cell for database persistence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchiveCellRow {
    /// Run identifier.
    pub run_id: String,
    /// Packed mixed-radix cell coordinate.
    pub cell_id: u64,
    /// Stable logical agent identity.
    pub uid: u64,
    /// Simulation tick when elite was inserted.
    pub tick_inserted: u64,
    /// Quality metric value.
    pub quality: f64,
    /// Serialized behavior descriptor bytes (JSON UTF-8).
    pub descriptor: Vec<u8>,
    /// Serialized genome envelope bytes (JSON UTF-8).
    pub genome: Vec<u8>,
    /// Genome schema version.
    pub genome_version: u32,
    /// Primary parent UID, if any.
    pub parent_uid: Option<u64>,
    /// Heritable generation of the agent.
    pub generation: u32,
}

impl ArchiveEntry {
    /// Convert this entry to a persistence database row.
    pub fn to_cell_row(&self, run_id: &str, cell_id: CellId) -> Result<ArchiveCellRow, QdError> {
        let descriptor_bytes = serde_json::to_vec(&self.descriptor)
            .map_err(|e| QdError::Serialization(e.to_string()))?;
        let genome_bytes =
            serde_json::to_vec(&self.genome).map_err(|e| QdError::Serialization(e.to_string()))?;
        Ok(ArchiveCellRow {
            run_id: run_id.to_string(),
            cell_id: cell_id.0,
            uid: self.uid.get(),
            tick_inserted: self.tick_inserted.0,
            quality: f64::from(self.quality),
            descriptor: descriptor_bytes,
            genome: genome_bytes,
            genome_version: self.genome.schema_version(),
            parent_uid: self.provenance.parent_uid.map(AgentUid::get),
            generation: self.provenance.generation.0,
        })
    }
}

impl ArchiveCellRow {
    /// Convert this persistence row back into a cell ID and archive entry.
    ///
    /// Narrows the persisted `f64` quality to the archive's `f32` representation.
    pub fn to_entry(&self) -> Result<(CellId, ArchiveEntry), QdError> {
        let descriptor: BehaviorDescriptor = serde_json::from_slice(&self.descriptor)
            .map_err(|e| QdError::Serialization(e.to_string()))?;
        let genome: BrainGenomeEnvelope = serde_json::from_slice(&self.genome)
            .map_err(|e| QdError::Serialization(e.to_string()))?;
        let provenance = ArchiveProvenance {
            run_id: self.run_id.clone(),
            parent_uid: self.parent_uid.map(AgentUid),
            generation: Generation(self.generation),
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "ArchiveEntry persists its f32 quality widened to f64; decoding restores that representation and retains the existing narrowing behavior for externally supplied rows"
        )]
        let quality = self.quality as f32;
        let entry = ArchiveEntry {
            uid: AgentUid(self.uid),
            tick_inserted: Tick(self.tick_inserted),
            descriptor,
            quality,
            genome,
            provenance,
        };
        Ok((CellId(self.cell_id), entry))
    }
}

impl BehaviorSpaceV0 {
    /// Export this behavior space as a persistence row.
    pub fn to_space_row(&self, run_id: &str) -> Result<ArchiveSpaceRow, QdError> {
        let axes_json =
            serde_json::to_string(&self.axes).map_err(|e| QdError::Serialization(e.to_string()))?;
        Ok(ArchiveSpaceRow {
            run_id: run_id.to_string(),
            space_version: self.version,
            axes_json,
        })
    }

    /// Reconstitute a behavior space from a persistence row.
    pub fn from_space_row(row: &ArchiveSpaceRow) -> Result<Self, QdError> {
        if row.space_version != BEHAVIOR_SPACE_SCHEMA_VERSION_V0 {
            return Err(QdError::Serialization(format!(
                "unsupported space version: expected {}, got {}",
                BEHAVIOR_SPACE_SCHEMA_VERSION_V0, row.space_version
            )));
        }
        let axes: Vec<Axis> = serde_json::from_str(&row.axes_json)
            .map_err(|e| QdError::Serialization(e.to_string()))?;
        let space = Self::new(row.space_version, axes);
        space.validate()?;
        Ok(space)
    }
}

impl MapElitesArchive {
    /// Export the archive behavior space definition as a persistence row.
    pub fn to_space_row(&self, run_id: &str) -> Result<ArchiveSpaceRow, QdError> {
        self.space.to_space_row(run_id)
    }

    /// Export all current elite cells as persistence rows.
    pub fn to_cell_rows(&self, run_id: &str) -> Result<Vec<ArchiveCellRow>, QdError> {
        let mut rows = Vec::with_capacity(self.cells.len());
        for (&cell_id, entry) in &self.cells {
            rows.push(entry.to_cell_row(run_id, cell_id)?);
        }
        Ok(rows)
    }

    /// Reconstitute an archive from space and cell rows.
    pub fn from_rows(
        space_row: &ArchiveSpaceRow,
        cell_rows: &[ArchiveCellRow],
        byte_cap: usize,
    ) -> Result<Self, QdError> {
        let space = BehaviorSpaceV0::from_space_row(space_row)?;
        let mut archive = Self::new(space, QualityMetric::default(), 0, byte_cap)?;
        for row in cell_rows {
            let (cell_id, entry) = row.to_entry()?;
            archive.current_bytes = archive
                .current_bytes
                .saturating_add(entry.approximate_bytes());
            archive.cells.insert(cell_id, entry);
        }
        Ok(archive)
    }

    /// Export the archive and its configuration as a self-describing bundle.
    #[must_use]
    pub fn export_bundle(&self, run_id: &str) -> ArchiveExportBundle {
        let cells: Vec<(CellId, ArchiveEntry)> = self
            .cells
            .iter()
            .map(|(&id, entry)| (id, entry.clone()))
            .collect();
        ArchiveExportBundle {
            run_id: run_id.to_string(),
            space: self.space.clone(),
            quality_metric: self.quality_metric,
            min_lifetime_ticks: self.min_lifetime_ticks,
            max_bytes: self.max_archive_bytes,
            cells,
        }
    }

    /// Reconstitute an archive from an export bundle.
    pub fn from_bundle(bundle: ArchiveExportBundle) -> Result<Self, QdError> {
        let mut archive = Self::new(
            bundle.space,
            bundle.quality_metric,
            bundle.min_lifetime_ticks,
            bundle.max_bytes,
        )?;
        for (cell_id, entry) in bundle.cells {
            archive.current_bytes = archive
                .current_bytes
                .saturating_add(entry.approximate_bytes());
            archive.cells.insert(cell_id, entry);
        }
        Ok(archive)
    }

    /// Emit structured log of all current QD metrics under target `scriptbots::qd`.
    pub fn log_metrics(&self, run_id: &str) {
        let m = self.metrics();
        diag_info!(
            target: "scriptbots::qd",
            run_id,
            occupied_cells = m.occupied_cells,
            total_cells = m.total_cells,
            coverage = m.coverage,
            qd_score_raw = m.qd_score_raw,
            qd_score_norm = m.qd_score_norm,
            mean_quality = m.mean_quality,
            max_quality = m.max_quality,
            occupancy_entropy = m.occupancy_entropy,
            "archive stats"
        );
    }

    /// Serialize the archive bundle to a deterministic, indented JSON string.
    pub fn export_json(&self, run_id: &str) -> Result<String, QdError> {
        let bundle = self.export_bundle(run_id);
        let s = serde_json::to_string_pretty(&bundle)
            .map_err(|e| QdError::Serialization(e.to_string()))?;
        let m = self.metrics();
        diag_info!(
            target: "scriptbots::qd",
            run_id,
            rows = self.cells.len(),
            bytes = s.len(),
            coverage = m.coverage,
            qd_score_raw = m.qd_score_raw,
            qd_score_norm = m.qd_score_norm,
            space_version = self.space.version,
            quality_version = self.quality_metric.as_str(),
            "archive exported json"
        );
        Ok(s)
    }

    /// Reconstitute an archive from a JSON export bundle string.
    pub fn import_json(json_str: &str) -> Result<(String, Self), QdError> {
        let bundle: ArchiveExportBundle =
            serde_json::from_str(json_str).map_err(|e| QdError::Serialization(e.to_string()))?;
        let run_id = bundle.run_id.clone();
        let archive = Self::from_bundle(bundle)?;
        let m = archive.metrics();
        diag_info!(
            target: "scriptbots::qd",
            run_id = %run_id,
            rows = archive.cells.len(),
            coverage = m.coverage,
            qd_score_raw = m.qd_score_raw,
            qd_score_norm = m.qd_score_norm,
            space_version = archive.space.version,
            quality_version = archive.quality_metric.as_str(),
            "archive imported json"
        );
        Ok((run_id, archive))
    }

    /// Export the archive to a self-describing, canonically ordered RFC 4180 CSV stream.
    ///
    /// The first line contains a structured `# PROVENANCE:` header encoding metadata,
    /// space definition, and quality metric. Subsequent lines contain elite rows ordered
    /// strictly by ascending [`CellId`].
    pub fn export_csv<W: std::io::Write>(
        &self,
        run_id: &str,
        writer: &mut W,
    ) -> Result<usize, QdError> {
        let header = CsvProvenanceHeader {
            run_id: run_id.to_string(),
            space_version: self.space.version,
            quality_metric: self.quality_metric,
            min_lifetime_ticks: self.min_lifetime_ticks,
            space: self.space.clone(),
        };
        let prov_json =
            serde_json::to_string(&header).map_err(|e| QdError::Serialization(e.to_string()))?;
        writeln!(writer, "# PROVENANCE: {prov_json}")
            .map_err(|e| QdError::Serialization(e.to_string()))?;
        writeln!(
            writer,
            "cell_id,uid,tick_inserted,quality,genome_version,parent_uid,generation,descriptor,genome"
        )
        .map_err(|e| QdError::Serialization(e.to_string()))?;

        let mut rows = 0_usize;
        for (&cell_id, entry) in &self.cells {
            let descriptor_json = serde_json::to_string(&entry.descriptor)
                .map_err(|e| QdError::Serialization(e.to_string()))?;
            let genome_json = serde_json::to_string(&entry.genome)
                .map_err(|e| QdError::Serialization(e.to_string()))?;
            let parent_uid_str = entry
                .provenance
                .parent_uid
                .map_or_else(String::new, |u| u.get().to_string());
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{}",
                cell_id.0,
                entry.uid.get(),
                entry.tick_inserted.0,
                entry.quality,
                entry.genome.schema_version(),
                parent_uid_str,
                entry.provenance.generation.0,
                escape_csv_field(&descriptor_json),
                escape_csv_field(&genome_json)
            )
            .map_err(|e| QdError::Serialization(e.to_string()))?;
            rows += 1;
        }
        writer
            .flush()
            .map_err(|e| QdError::Serialization(e.to_string()))?;
        let m = self.metrics();
        diag_info!(
            target: "scriptbots::qd",
            run_id,
            rows,
            coverage = m.coverage,
            qd_score_raw = m.qd_score_raw,
            qd_score_norm = m.qd_score_norm,
            space_version = self.space.version,
            quality_version = self.quality_metric.as_str(),
            "archive exported csv"
        );
        Ok(rows)
    }

    /// Reconstitute an archive from an RFC 4180 CSV reader written by [`Self::export_csv`].
    #[allow(clippy::too_many_lines)]
    pub fn import_csv<R: std::io::BufRead>(
        reader: R,
        byte_cap: usize,
    ) -> Result<(String, Self), QdError> {
        let mut lines = reader.lines();
        let mut first_line = None;
        for line in &mut lines {
            let l = line.map_err(|e| QdError::Serialization(e.to_string()))?;
            let trimmed = l.trim();
            if !trimmed.is_empty() {
                first_line = Some(trimmed.to_string());
                break;
            }
        }
        let Some(first_line) = first_line else {
            return Err(QdError::Serialization("empty CSV input".to_string()));
        };
        if !first_line.starts_with("# PROVENANCE: ") {
            return Err(QdError::Serialization(format!(
                "missing '# PROVENANCE: ' header line, found: {first_line:?}"
            )));
        }
        let prov_str = &first_line["# PROVENANCE: ".len()..];
        let header: CsvProvenanceHeader = serde_json::from_str(prov_str)
            .map_err(|e| QdError::Serialization(format!("invalid provenance JSON: {e}")))?;

        let mut header_line = None;
        for line in &mut lines {
            let l = line.map_err(|e| QdError::Serialization(e.to_string()))?;
            let trimmed = l.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                header_line = Some(l);
                break;
            }
        }
        let Some(header_line) = header_line else {
            return Err(QdError::Serialization(
                "missing CSV column header line".to_string(),
            ));
        };
        let col_headers = parse_csv_line(&header_line);
        if col_headers.len() < 9 {
            return Err(QdError::Serialization(format!(
                "expected at least 9 columns, found {}: {col_headers:?}",
                col_headers.len()
            )));
        }

        let mut archive = Self::new(
            header.space,
            header.quality_metric,
            header.min_lifetime_ticks,
            byte_cap,
        )?;

        for line_res in lines {
            let line = line_res.map_err(|e| QdError::Serialization(e.to_string()))?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let fields = parse_csv_line(&line);
            if fields.len() < 9 {
                return Err(QdError::Serialization(format!(
                    "row has fewer than 9 fields: {fields:?}"
                )));
            }
            let cell_id: u64 = fields[0]
                .parse()
                .map_err(|e| QdError::Serialization(format!("invalid cell_id: {e}")))?;
            let uid: u64 = fields[1]
                .parse()
                .map_err(|e| QdError::Serialization(format!("invalid uid: {e}")))?;
            let tick_inserted: u64 = fields[2]
                .parse()
                .map_err(|e| QdError::Serialization(format!("invalid tick_inserted: {e}")))?;
            let quality: f32 = fields[3]
                .parse()
                .map_err(|e| QdError::Serialization(format!("invalid quality: {e}")))?;
            let parent_uid = if fields[5].is_empty() {
                None
            } else {
                Some(AgentUid(fields[5].parse().map_err(|e| {
                    QdError::Serialization(format!("invalid parent_uid: {e}"))
                })?))
            };
            let generation: u32 = fields[6]
                .parse()
                .map_err(|e| QdError::Serialization(format!("invalid generation: {e}")))?;
            let descriptor: BehaviorDescriptor = serde_json::from_str(&fields[7])
                .map_err(|e| QdError::Serialization(format!("invalid descriptor JSON: {e}")))?;
            let genome: BrainGenomeEnvelope = serde_json::from_str(&fields[8])
                .map_err(|e| QdError::Serialization(format!("invalid genome JSON: {e}")))?;

            let entry = ArchiveEntry {
                uid: AgentUid(uid),
                tick_inserted: Tick(tick_inserted),
                descriptor,
                quality,
                genome,
                provenance: ArchiveProvenance {
                    run_id: header.run_id.clone(),
                    parent_uid,
                    generation: Generation(generation),
                },
            };
            archive.current_bytes = archive
                .current_bytes
                .saturating_add(entry.approximate_bytes());
            archive.cells.insert(CellId(cell_id), entry);
        }

        let m = archive.metrics();
        diag_info!(
            target: "scriptbots::qd",
            run_id = %header.run_id,
            rows = archive.cells.len(),
            coverage = m.coverage,
            qd_score_raw = m.qd_score_raw,
            qd_score_norm = m.qd_score_norm,
            space_version = archive.space.version,
            quality_version = archive.quality_metric.as_str(),
            "archive imported csv"
        );

        Ok((header.run_id, archive))
    }
}

/// Self-describing export bundle for a complete MAP-Elites archive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArchiveExportBundle {
    /// Identifier of the simulation run that generated the archive.
    pub run_id: String,
    /// Behavior space defining the axes and discretization bins.
    pub space: BehaviorSpaceV0,
    /// Quality metric function used to rank candidates.
    pub quality_metric: QualityMetric,
    /// Minimum lifetime ticks filter applied to candidates.
    pub min_lifetime_ticks: u32,
    /// Maximum allowed memory footprint in bytes.
    pub max_bytes: usize,
    /// Vector of (`cell_id`, entry) pairs in ascending `CellId` order.
    pub cells: Vec<(CellId, ArchiveEntry)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CsvProvenanceHeader {
    run_id: String,
    space_version: u16,
    quality_metric: QualityMetric,
    min_lifetime_ticks: u32,
    space: BehaviorSpaceV0,
}

fn escape_csv_field(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == ',' {
            fields.push(current);
            current = String::new();
        } else {
            current.push(c);
        }
    }
    fields.push(current);
    fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BrainFamilyId, BrainProvenance};

    fn make_test_genome(seed: u8) -> BrainGenomeEnvelope {
        let family_id = BrainFamilyId::new("mlp").expect("family_id");
        BrainGenomeEnvelope::new(family_id, 1, 1, vec![seed; 16], BrainProvenance::default())
            .expect("genome envelope")
    }

    #[test]
    fn test_binning_table_boundaries_and_clamping() {
        let axis = Axis::new("speed", PhenotypeFeature::MeanSpeed, (0.0, 10.0), 5).expect("axis");

        // value == domain.lo -> bin 0
        assert_eq!(axis.discretize(0.0, 0).expect("bin"), 0);

        // value == domain.hi -> LAST bin (4), not out-of-range
        assert_eq!(axis.discretize(10.0, 0).expect("bin"), 4);

        // Below domain -> clamp to 0
        assert_eq!(axis.discretize(-5.0, 0).expect("bin"), 0);

        // Above domain -> clamp to last bin (4)
        assert_eq!(axis.discretize(100.0, 0).expect("bin"), 4);

        // Intermediate values
        assert_eq!(axis.discretize(1.9, 0).expect("bin"), 0);
        assert_eq!(axis.discretize(2.0, 0).expect("bin"), 1);
        assert_eq!(axis.discretize(4.0, 0).expect("bin"), 2);
        assert_eq!(axis.discretize(6.0, 0).expect("bin"), 3);
        assert_eq!(axis.discretize(8.0, 0).expect("bin"), 4);

        // NaN -> typed error
        assert!(matches!(
            axis.discretize(f32::NAN, 0),
            Err(QdError::NonFiniteValue { .. })
        ));

        // Infs -> typed error
        assert!(matches!(
            axis.discretize(f32::INFINITY, 0),
            Err(QdError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            axis.discretize(f32::NEG_INFINITY, 0),
            Err(QdError::NonFiniteValue { .. })
        ));

        // Zero-width domain -> typed error
        assert!(matches!(
            Axis::new("zero", PhenotypeFeature::MeanSpeed, (5.0, 5.0), 5),
            Err(QdError::InvalidDomain { .. })
        ));

        // Inverted domain -> typed error
        assert!(matches!(
            Axis::new("inverted", PhenotypeFeature::MeanSpeed, (10.0, 5.0), 5),
            Err(QdError::InvalidDomain { .. })
        ));
    }

    #[test]
    fn test_mixed_radix_and_invertibility() {
        let space = BehaviorSpaceV0::new(
            0,
            vec![
                Axis::new("x", PhenotypeFeature::MeanSpeed, (0.0, 10.0), 4).expect("axis"),
                Axis::new("y", PhenotypeFeature::DietTendency, (0.0, 1.0), 5).expect("axis"),
                Axis::new("z", PhenotypeFeature::SpikeUsageRate, (0.0, 1.0), 3).expect("axis"),
            ],
        );
        space.validate().expect("valid space");
        assert_eq!(space.total_cells().expect("total"), 4 * 5 * 3);

        // (x=0, y=0, z=0) -> 0
        let desc_0 = BehaviorDescriptor::new(vec![0.0, 0.0, 0.0]);
        let id_0 = space.cell_index(&desc_0).expect("id");
        assert_eq!(id_0.0, 0);
        assert_eq!(
            space.decode_cell_coords(id_0).expect("coords"),
            vec![0, 0, 0]
        );

        // (x=1, y=2, z=2) -> 1 + 2*4 + 2*(4*5) = 1 + 8 + 40 = 49
        // x in [2.5, 5.0) -> bin 1
        // y in [0.4, 0.6) -> bin 2
        // z in [0.66, 1.0) -> bin 2
        let desc_49 = BehaviorDescriptor::new(vec![3.0, 0.5, 0.8]);
        let id_49 = space.cell_index(&desc_49).expect("id");
        assert_eq!(id_49.0, 49);
        assert_eq!(
            space.decode_cell_coords(id_49).expect("coords"),
            vec![1, 2, 2]
        );

        // Max cell -> (3, 4, 2) -> 3 + 4*4 + 2*20 = 3 + 16 + 40 = 59
        let desc_max = BehaviorDescriptor::new(vec![10.0, 1.0, 1.0]);
        let id_max = space.cell_index(&desc_max).expect("id");
        assert_eq!(id_max.0, 59);
        assert_eq!(
            space.decode_cell_coords(id_max).expect("coords"),
            vec![3, 4, 2]
        );
    }

    #[test]
    fn test_cell_cap_named_at_validate() {
        // Space with 10^7 = 10,000,000 cells exceeds 1,000,000 cap
        let axes = (0..7)
            .map(|i| {
                Axis::new(
                    format!("axis_{i}"),
                    PhenotypeFeature::MeanSpeed,
                    (0.0, 10.0),
                    10,
                )
                .expect("axis")
            })
            .collect();
        let space = BehaviorSpaceV0::new(0, axes);
        let err = space.validate().expect_err("should reject > 1,000,000");
        match err {
            QdError::CellCapacityExceeded {
                total_cells,
                max_cells,
            } => {
                assert_eq!(total_cells, 10_000_000);
                assert_eq!(max_cells, 1_000_000);
            }
            other => panic!("expected CellCapacityExceeded, got {other:?}"),
        }
    }

    #[test]
    fn test_insertion_replacement_and_uid_tie_breaking() {
        let space = BehaviorSpaceV0::new(
            0,
            vec![
                Axis::new("x", PhenotypeFeature::MeanSpeed, (0.0, 1.0), 2).expect("axis"),
                Axis::new("y", PhenotypeFeature::DietTendency, (0.0, 1.0), 2).expect("axis"),
            ],
        );
        let mut archive = MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 100, 100_000)
            .expect("archive");

        let entry1 = ArchiveEntry {
            uid: AgentUid(10),
            tick_inserted: Tick(100),
            descriptor: BehaviorDescriptor::new(vec![0.1, 0.1]),
            quality: 50.0,
            genome: make_test_genome(1),
            provenance: ArchiveProvenance {
                run_id: "test".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        };
        // Initial insert -> New
        let res = archive.insert(entry1).expect("insert");
        assert_eq!(res, InsertionResult::InsertedNew);
        assert_eq!(archive.cells.len(), 1);
        assert_eq!(archive.cells.values().next().unwrap().uid, AgentUid(10));

        // Worse quality -> Rejected
        let entry_worse = ArchiveEntry {
            uid: AgentUid(5),
            tick_inserted: Tick(101),
            descriptor: BehaviorDescriptor::new(vec![0.1, 0.1]),
            quality: 40.0,
            genome: make_test_genome(2),
            provenance: ArchiveProvenance {
                run_id: "test".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        };
        let res = archive.insert(entry_worse).expect("insert");
        assert_eq!(res, InsertionResult::RejectedWorseOrEqual);
        assert_eq!(archive.cells.values().next().unwrap().uid, AgentUid(10));

        // Equal quality, HIGHER UID -> Rejected
        let entry_equal_higher_uid = ArchiveEntry {
            uid: AgentUid(20),
            tick_inserted: Tick(102),
            descriptor: BehaviorDescriptor::new(vec![0.1, 0.1]),
            quality: 50.0,
            genome: make_test_genome(3),
            provenance: ArchiveProvenance {
                run_id: "test".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        };
        let res = archive.insert(entry_equal_higher_uid).expect("insert");
        assert_eq!(res, InsertionResult::RejectedWorseOrEqual);
        assert_eq!(archive.cells.values().next().unwrap().uid, AgentUid(10));

        // Equal quality, LOWER UID (5 < 10) -> ReplacedTieBreak
        let entry_equal_lower_uid = ArchiveEntry {
            uid: AgentUid(5),
            tick_inserted: Tick(103),
            descriptor: BehaviorDescriptor::new(vec![0.1, 0.1]),
            quality: 50.0,
            genome: make_test_genome(4),
            provenance: ArchiveProvenance {
                run_id: "test".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        };
        let res = archive.insert(entry_equal_lower_uid).expect("insert");
        assert_eq!(
            res,
            InsertionResult::ReplacedTieBreak {
                displaced_uid: AgentUid(10),
                displaced_quality: 50.0,
            }
        );
        assert_eq!(archive.cells.values().next().unwrap().uid, AgentUid(5));

        // Strictly higher quality -> ReplacedBetter
        let entry_better = ArchiveEntry {
            uid: AgentUid(99),
            tick_inserted: Tick(104),
            descriptor: BehaviorDescriptor::new(vec![0.1, 0.1]),
            quality: 80.0,
            genome: make_test_genome(5),
            provenance: ArchiveProvenance {
                run_id: "test".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        };
        let res = archive.insert(entry_better).expect("insert");
        assert_eq!(
            res,
            InsertionResult::ReplacedBetter {
                displaced_uid: AgentUid(5),
                displaced_quality: 50.0,
            }
        );
        assert_eq!(archive.cells.values().next().unwrap().uid, AgentUid(99));
        assert!((archive.qd_score() - 80.0).abs() < 1e-6);
    }

    #[test]
    fn test_byte_cap_enforced_explicitly() {
        let space = BehaviorSpaceV0::new(
            0,
            vec![Axis::new("x", PhenotypeFeature::MeanSpeed, (0.0, 1.0), 10).expect("axis")],
        );
        // Set cap very low (e.g. 50 bytes)
        let mut archive =
            MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 100, 50).expect("archive");

        let entry = ArchiveEntry {
            uid: AgentUid(1),
            tick_inserted: Tick(100),
            descriptor: BehaviorDescriptor::new(vec![0.5]),
            quality: 10.0,
            genome: make_test_genome(1),
            provenance: ArchiveProvenance {
                run_id: "run".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        };
        let err = archive.insert(entry).expect_err("should exceed byte cap");
        assert!(matches!(err, QdError::ByteCapExceeded { .. }));
    }

    #[test]
    fn test_determinism_and_sorted_iteration() {
        let space = BehaviorSpaceV0::new(
            0,
            vec![
                Axis::new("x", PhenotypeFeature::MeanSpeed, (0.0, 10.0), 5).expect("axis"),
                Axis::new("y", PhenotypeFeature::DietTendency, (0.0, 1.0), 5).expect("axis"),
            ],
        );
        let mut archive1 =
            MapElitesArchive::new(space.clone(), QualityMetric::LifetimeIntake, 100, 1_000_000)
                .expect("archive");
        let mut archive2 =
            MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 100, 1_000_000)
                .expect("archive");

        let entries = vec![
            ArchiveEntry {
                uid: AgentUid(3),
                tick_inserted: Tick(10),
                descriptor: BehaviorDescriptor::new(vec![8.0, 0.8]),
                quality: 30.0,
                genome: make_test_genome(3),
                provenance: ArchiveProvenance {
                    run_id: "test".to_string(),
                    parent_uid: None,
                    generation: Generation(0),
                },
            },
            ArchiveEntry {
                uid: AgentUid(1),
                tick_inserted: Tick(10),
                descriptor: BehaviorDescriptor::new(vec![1.0, 0.1]),
                quality: 10.0,
                genome: make_test_genome(1),
                provenance: ArchiveProvenance {
                    run_id: "test".to_string(),
                    parent_uid: None,
                    generation: Generation(0),
                },
            },
            ArchiveEntry {
                uid: AgentUid(2),
                tick_inserted: Tick(10),
                descriptor: BehaviorDescriptor::new(vec![5.0, 0.5]),
                quality: 20.0,
                genome: make_test_genome(2),
                provenance: ArchiveProvenance {
                    run_id: "test".to_string(),
                    parent_uid: None,
                    generation: Generation(0),
                },
            },
        ];

        // Insert in order
        for e in entries.clone() {
            archive1.insert(e).expect("insert");
        }
        // Insert in reverse order
        for e in entries.into_iter().rev() {
            archive2.insert(e).expect("insert");
        }

        // Iteration order must be strictly ascending CellId
        let keys1 = archive1.cell_ids_sorted();
        let keys2 = archive2.cell_ids_sorted();
        assert_eq!(keys1, keys2);
        for pair in keys1.windows(2) {
            assert!(pair[0] < pair[1], "keys must be strictly ascending");
        }

        // Serialized bytes must be identical
        let bytes1 = serde_json::to_vec(&archive1).expect("serialize");
        let bytes2 = serde_json::to_vec(&archive2).expect("serialize");
        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn test_archive_persistence_row_roundtrip() {
        let space = BehaviorSpaceV0::default();
        let mut archive = MapElitesArchive::new(
            space,
            QualityMetric::default(),
            100,
            DEFAULT_MAX_ARCHIVE_BYTES,
        )
        .expect("archive");
        let entry = ArchiveEntry {
            uid: AgentUid(42),
            tick_inserted: Tick(100),
            descriptor: BehaviorDescriptor::new(vec![0.5, 2.5, 0.1, 0.2, 0.3, 0.4]),
            quality: 75.5,
            genome: make_test_genome(7),
            provenance: ArchiveProvenance {
                run_id: "run-qd-test".to_string(),
                parent_uid: Some(AgentUid(10)),
                generation: Generation(3),
            },
        };
        archive.insert(entry.clone()).expect("insert");

        let space_row = archive.to_space_row("run-qd-test").expect("space row");
        let cell_rows = archive.to_cell_rows("run-qd-test").expect("cell rows");
        assert_eq!(space_row.run_id, "run-qd-test");
        assert_eq!(cell_rows.len(), 1);
        assert_eq!(cell_rows[0].uid, 42);
        assert_eq!(cell_rows[0].quality, 75.5);

        let restored =
            MapElitesArchive::from_rows(&space_row, &cell_rows, DEFAULT_MAX_ARCHIVE_BYTES)
                .expect("restored archive");
        assert_eq!(restored.len(), archive.len());
        assert_eq!(restored.cell_ids_sorted(), archive.cell_ids_sorted());
        let restored_entry = restored.get(restored.cell_ids_sorted()[0]).expect("entry");
        assert_eq!(restored_entry.uid, entry.uid);
        assert_eq!(restored_entry.quality, entry.quality);
        assert_eq!(restored_entry.descriptor, entry.descriptor);
        assert_eq!(restored_entry.genome, entry.genome);
    }

    #[test]
    fn test_knn_1d_fixture_exact_distances() {
        let space = BehaviorSpaceV0::new(
            0,
            vec![Axis::new("x", PhenotypeFeature::MeanSpeed, (0.0, 10.0), 10).expect("axis")],
        );
        let points = [1.0_f32, 2.0, 4.0, 7.0, 9.0];
        let population: Vec<CandidateDescriptor> = points
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                CandidateDescriptor::new(AgentUid(i as u64 + 1), BehaviorDescriptor::new(vec![x]))
            })
            .collect();

        let scores = compute_population_novelty(&population, None, &space, 2).expect("scores");
        assert_eq!(scores.len(), 5);

        // Expected hand-computed mean distances for k=2 with self-exclusion:
        // x=1: nearest {2, 4}, normalized distances {0.1, 0.3} -> mean 0.20
        // x=2: nearest {1, 4}, normalized distances {0.1, 0.2} -> mean 0.15
        // x=4: nearest {2, 1}, normalized distances {0.2, 0.3} -> mean 0.25
        // x=7: nearest {9, 4}, normalized distances {0.2, 0.3} -> mean 0.25
        // x=9: nearest {7, 4}, normalized distances {0.2, 0.5} -> mean 0.35
        let expected = [0.20_f32, 0.15, 0.25, 0.25, 0.35];
        for (i, &(_, score)) in scores.iter().enumerate() {
            assert!(
                (score - expected[i]).abs() < 1e-5,
                "index {i} expected {}, got {score}",
                expected[i]
            );
        }
    }

    #[test]
    fn test_knn_3d_fixture_exact_distances() {
        let space = BehaviorSpaceV0::new(
            0,
            vec![
                Axis::new("x", PhenotypeFeature::MeanSpeed, (0.0, 10.0), 10).expect("axis 0"),
                Axis::new("y", PhenotypeFeature::TurnRate, (0.0, 20.0), 10).expect("axis 1"),
                Axis::new("z", PhenotypeFeature::SensingMean, (0.0, 5.0), 10).expect("axis 2"),
            ],
        );
        let p1 =
            CandidateDescriptor::new(AgentUid(1), BehaviorDescriptor::new(vec![0.0, 0.0, 0.0]));
        let p2 =
            CandidateDescriptor::new(AgentUid(2), BehaviorDescriptor::new(vec![10.0, 0.0, 0.0]));
        let p3 =
            CandidateDescriptor::new(AgentUid(3), BehaviorDescriptor::new(vec![0.0, 20.0, 0.0]));
        let p4 =
            CandidateDescriptor::new(AgentUid(4), BehaviorDescriptor::new(vec![0.0, 0.0, 5.0]));
        let population = vec![p1, p2, p3, p4];

        let scores = compute_population_novelty(&population, None, &space, 2).expect("scores");
        assert_eq!(scores.len(), 4);

        // Every neighbor of p1 is at normalized distance sqrt(1.0^2 + 0 + 0) = 1.0
        // With k=2, nearest 2 neighbors have mean distance 1.0 exactly
        assert!((scores[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_domain_normalization_inverts_raw_euclidean_ranking() {
        let space = BehaviorSpaceV0::new(
            0,
            vec![
                Axis::new("large_axis", PhenotypeFeature::MeanSpeed, (0.0, 1000.0), 10)
                    .expect("axis 0"),
                Axis::new("small_axis", PhenotypeFeature::DietTendency, (0.0, 1.0), 10)
                    .expect("axis 1"),
            ],
        );
        let target = BehaviorDescriptor::new(vec![500.0, 0.5]);
        let cand_a = BehaviorDescriptor::new(vec![510.0, 0.5]);
        let cand_b = BehaviorDescriptor::new(vec![500.0, 0.9]);

        // Raw Euclidean distances:
        // dist(T, A) = sqrt(10^2 + 0^2) = 10.0
        // dist(T, B) = sqrt(0^2 + 0.4^2) = 0.4
        // Raw ranking: B is MUCH closer than A (0.4 < 10.0).
        let raw_a = (target.0[0] - cand_a.0[0]).hypot(target.0[1] - cand_a.0[1]);
        let raw_b = (target.0[0] - cand_b.0[0]).hypot(target.0[1] - cand_b.0[1]);
        assert!(raw_b < raw_a, "raw Euclidean must rank B closer than A");

        // Domain-normalized distances:
        // dist_norm(T, A) = sqrt((10/1000)^2 + 0) = 0.01
        // dist_norm(T, B) = sqrt(0 + (0.4/1.0)^2) = 0.40
        // Normalized ranking: A is MUCH closer than B (0.01 < 0.40).
        let norm_a = normalized_distance(&target, &cand_a, &space).expect("norm_a");
        let norm_b = normalized_distance(&target, &cand_b, &space).expect("norm_b");
        assert!(
            norm_a < norm_b,
            "domain-normalized distance must invert ranking: A ({norm_a}) is closer than B ({norm_b})"
        );
    }

    #[test]
    fn test_population_plus_archive_membership_and_self_exclusion() {
        let space = BehaviorSpaceV0::new(
            0,
            vec![Axis::new("x", PhenotypeFeature::MeanSpeed, (0.0, 10.0), 10).expect("axis")],
        );
        let mut archive = MapElitesArchive::new(
            space.clone(),
            QualityMetric::default(),
            0,
            DEFAULT_MAX_ARCHIVE_BYTES,
        )
        .expect("archive");

        // Insert an elite with UID 100 at x = 5.0
        let elite = ArchiveEntry {
            uid: AgentUid(100),
            tick_inserted: Tick(10),
            descriptor: BehaviorDescriptor::new(vec![5.0]),
            quality: 50.0,
            genome: make_test_genome(1),
            provenance: ArchiveProvenance {
                run_id: "test".to_string(),
                parent_uid: None,
                generation: Generation(1),
            },
        };
        archive.insert(elite).expect("insert elite");

        // Population: Agent 1 at x=5.1, Agent 2 at x=9.0
        let pop = vec![
            CandidateDescriptor::new(AgentUid(1), BehaviorDescriptor::new(vec![5.1])),
            CandidateDescriptor::new(AgentUid(2), BehaviorDescriptor::new(vec![9.0])),
        ];

        // For Agent 1, neighbors are Agent 2 (dist |5.1-9.0|/10 = 0.39) and Elite 100 (dist |5.1-5.0|/10 = 0.01).
        // With k=1, Agent 1's nearest neighbor is Elite 100 with distance 0.01.
        let scores = compute_population_novelty(&pop, Some(&archive), &space, 1).expect("scores");
        assert!((scores[0].1 - 0.01).abs() < 1e-4);

        // Self-exclusion test: if an agent in population has UID 100 (same as archive entry),
        // it must NOT compare against itself in the archive (distance 0.0 to self must be excluded).
        let pop_with_same_uid = vec![CandidateDescriptor::new(
            AgentUid(100),
            BehaviorDescriptor::new(vec![5.0]),
        )];
        let self_excluded_scores =
            compute_population_novelty(&pop_with_same_uid, Some(&archive), &space, 1)
                .expect("scores");
        // Only one agent, and archive has only UID 100, which is self -> pool is empty -> score 0.0
        assert_eq!(self_excluded_scores[0].1, 0.0);
    }

    #[test]
    fn test_degenerate_cases_exhaustive() {
        let space = BehaviorSpaceV0::new(
            0,
            vec![Axis::new("x", PhenotypeFeature::MeanSpeed, (0.0, 10.0), 10).expect("axis")],
        );

        // 1. k = 0 -> typed error
        let pop = vec![CandidateDescriptor::new(
            AgentUid(1),
            BehaviorDescriptor::new(vec![1.0]),
        )];
        assert_eq!(
            compute_population_novelty(&pop, None, &space, 0),
            Err(QdError::ZeroK)
        );

        // 2. Empty population -> empty result
        let empty_scores =
            compute_population_novelty(&[], None, &space, 5).expect("empty pop scores");
        assert_eq!(empty_scores, Vec::new());

        // 3. Singleton population with empty archive -> 0.0, no division by zero
        let single_scores = compute_population_novelty(&pop, None, &space, 5).expect("single");
        assert_eq!(single_scores, vec![(AgentUid(1), 0.0)]);

        // 4. k > N -> clamps to N available neighbors without panic
        let pop3 = vec![
            CandidateDescriptor::new(AgentUid(1), BehaviorDescriptor::new(vec![1.0])),
            CandidateDescriptor::new(AgentUid(2), BehaviorDescriptor::new(vec![2.0])),
            CandidateDescriptor::new(AgentUid(3), BehaviorDescriptor::new(vec![3.0])),
        ];
        let clamped = compute_population_novelty(&pop3, None, &space, 100).expect("clamped");
        assert_eq!(clamped.len(), 3);
        // Agent 1: neighbors 2 (0.1) and 3 (0.2) -> mean 0.15
        assert!((clamped[0].1 - 0.15).abs() < 1e-5);

        // 5. All-identical descriptors -> all zeros, no NaN
        let identical_pop = vec![
            CandidateDescriptor::new(AgentUid(1), BehaviorDescriptor::new(vec![4.0])),
            CandidateDescriptor::new(AgentUid(2), BehaviorDescriptor::new(vec![4.0])),
            CandidateDescriptor::new(AgentUid(3), BehaviorDescriptor::new(vec![4.0])),
        ];
        let zeros = compute_population_novelty(&identical_pop, None, &space, 2).expect("identical");
        for (_, score) in zeros {
            assert_eq!(score, 0.0);
            assert!(!score.is_nan());
        }

        // 6. Dimension mismatch -> typed error
        let wrong_dim = vec![CandidateDescriptor::new(
            AgentUid(1),
            BehaviorDescriptor::new(vec![1.0, 2.0]),
        )];
        assert!(matches!(
            compute_population_novelty(&wrong_dim, None, &space, 1),
            Err(QdError::DimensionMismatch { .. })
        ));

        // 7. Non-finite descriptor value -> typed error
        let nan_val = vec![CandidateDescriptor::new(
            AgentUid(1),
            BehaviorDescriptor::new(vec![f32::NAN]),
        )];
        assert!(matches!(
            compute_population_novelty(&nan_val, None, &space, 1),
            Err(QdError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_curiosity_weights_and_normalization() {
        let novelty_raw = vec![(AgentUid(1), 0.2_f32), (AgentUid(2), 0.8_f32)];
        let fitness_raw = vec![(AgentUid(1), 10.0_f32), (AgentUid(2), 30.0_f32)];

        let novelty_norm = normalize_scores(&novelty_raw);
        let fitness_norm = normalize_scores(&fitness_raw);

        // Min-max normalization:
        // Novelty: (0.2 -> 0.0, 0.8 -> 1.0)
        assert!((novelty_norm[0].1 - 0.0).abs() < 1e-5);
        assert!((novelty_norm[1].1 - 1.0).abs() < 1e-5);
        // Fitness: (10 -> 0.0, 30 -> 1.0)
        assert!((fitness_norm[0].1 - 0.0).abs() < 1e-5);
        assert!((fitness_norm[1].1 - 1.0).abs() < 1e-5);

        // w = 1.0 -> pure novelty
        let pure_novelty =
            combine_curiosity(1.0, &novelty_norm, &fitness_norm).expect("pure novelty");
        assert_eq!(pure_novelty, novelty_norm);

        // w = 0.0 -> pure fitness
        let pure_fitness =
            combine_curiosity(0.0, &novelty_norm, &fitness_norm).expect("pure fitness");
        assert_eq!(pure_fitness, fitness_norm);

        // w = 0.5 -> average
        let mixed = combine_curiosity(0.5, &novelty_norm, &fitness_norm).expect("mixed");
        assert!((mixed[0].1 - 0.0).abs() < 1e-5);
        assert!((mixed[1].1 - 1.0).abs() < 1e-5);

        // Out-of-range weights:
        assert!(matches!(
            combine_curiosity(-0.1, &novelty_norm, &fitness_norm),
            Err(QdError::InvalidCuriosityWeight { .. })
        ));
        assert!(matches!(
            combine_curiosity(1.1, &novelty_norm, &fitness_norm),
            Err(QdError::InvalidCuriosityWeight { .. })
        ));
        assert!(matches!(
            combine_curiosity(f32::NAN, &novelty_norm, &fitness_norm),
            Err(QdError::InvalidCuriosityWeight { .. })
        ));
    }

    #[test]
    fn test_evolution_selection_mode_validation() {
        assert_eq!(EvolutionSelectionMode::Fitness.validate(), Ok(()));
        assert_eq!(EvolutionSelectionMode::Novelty.validate(), Ok(()));
        assert_eq!(
            EvolutionSelectionMode::Curiosity { w: 0.0 }.validate(),
            Ok(())
        );
        assert_eq!(
            EvolutionSelectionMode::Curiosity { w: 0.5 }.validate(),
            Ok(())
        );
        assert_eq!(
            EvolutionSelectionMode::Curiosity { w: 1.0 }.validate(),
            Ok(())
        );
        assert!(matches!(
            EvolutionSelectionMode::Curiosity { w: -0.01 }.validate(),
            Err(QdError::InvalidCuriosityWeight { .. })
        ));
        assert!(matches!(
            EvolutionSelectionMode::Curiosity { w: 1.01 }.validate(),
            Err(QdError::InvalidCuriosityWeight { .. })
        ));
        assert!(matches!(
            EvolutionSelectionMode::Curiosity { w: f32::NAN }.validate(),
            Err(QdError::InvalidCuriosityWeight { .. })
        ));
    }

    fn make_test_entry(uid: u64, quality: f32, descriptor: Vec<f32>) -> ArchiveEntry {
        ArchiveEntry {
            uid: AgentUid(uid),
            tick_inserted: Tick(100),
            descriptor: BehaviorDescriptor::new(descriptor),
            quality,
            #[allow(clippy::cast_possible_truncation)]
            genome: make_test_genome((uid & 0xff) as u8),
            provenance: ArchiveProvenance {
                run_id: "test".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        }
    }

    #[test]
    fn test_qd_metrics_empty_full_partial_and_zero_qualities() {
        let axis = Axis::new("speed", PhenotypeFeature::MeanSpeed, (0.0, 4.0), 4).expect("axis");
        let space = BehaviorSpaceV0 {
            version: BEHAVIOR_SPACE_SCHEMA_VERSION_V0,
            axes: vec![axis],
        };
        let mut archive = MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 100, 100_000)
            .expect("archive");

        // 1. Empty archive metrics: coverage 0, qd 0, no NaN
        let empty_metrics = archive.metrics();
        assert_eq!(empty_metrics.total_cells, 4);
        assert_eq!(empty_metrics.occupied_cells, 0);
        assert_eq!(empty_metrics.coverage, 0.0);
        assert_eq!(empty_metrics.qd_score_raw, 0.0);
        assert_eq!(empty_metrics.qd_score_norm, 0.0);
        assert_eq!(empty_metrics.occupancy_entropy, 0.0);
        assert_eq!(empty_metrics.mean_quality, 0.0);
        assert_eq!(empty_metrics.max_quality, None);
        assert!(!empty_metrics.qd_score_norm.is_nan());
        assert!(!empty_metrics.occupancy_entropy.is_nan());

        // 2. Full archive with quality 1.0
        // Axis (0..4) with 4 bins: centers 0.5, 1.5, 2.5, 3.5
        for (i, &center) in [0.5, 1.5, 2.5, 3.5].iter().enumerate() {
            let entry = make_test_entry((i + 1) as u64, 1.0, vec![center]);
            archive.insert(entry).expect("insert full");
        }
        let full_metrics = archive.metrics();
        assert_eq!(full_metrics.total_cells, 4);
        assert_eq!(full_metrics.occupied_cells, 4);
        assert_eq!(full_metrics.coverage, 1.0);
        assert_eq!(full_metrics.qd_score_raw, 4.0);
        assert_eq!(full_metrics.qd_score_norm, 1.0);
        assert_eq!(full_metrics.occupancy_entropy, 0.0); // full has zero uncertainty
        assert_eq!(full_metrics.mean_quality, 1.0);
        assert_eq!(full_metrics.max_quality, Some(1.0));

        // 3. Half-full archive (2 of 4 cells occupied with quality 1.0)
        let mut half_archive = MapElitesArchive::new(
            archive.space.clone(),
            QualityMetric::LifetimeIntake,
            100,
            100_000,
        )
        .expect("archive");
        half_archive
            .insert(make_test_entry(1, 1.0, vec![0.5]))
            .expect("insert");
        half_archive
            .insert(make_test_entry(2, 1.0, vec![1.5]))
            .expect("insert");

        let half_metrics = half_archive.metrics();
        assert_eq!(half_metrics.total_cells, 4);
        assert_eq!(half_metrics.occupied_cells, 2);
        assert_eq!(half_metrics.coverage, 0.5);
        assert_eq!(half_metrics.qd_score_raw, 2.0);
        assert_eq!(half_metrics.qd_score_norm, 0.5);
        assert_eq!(half_metrics.occupancy_entropy, 1.0); // binary entropy of p=0.5 is 1.0 bit
        assert_eq!(half_metrics.mean_quality, 1.0);
        assert_eq!(half_metrics.max_quality, Some(1.0));

        // 4. All-zero archive (2 of 4 cells occupied with quality 0.0)
        let mut zero_archive = MapElitesArchive::new(
            archive.space.clone(),
            QualityMetric::LifetimeIntake,
            100,
            100_000,
        )
        .expect("archive");
        zero_archive
            .insert(make_test_entry(1, 0.0, vec![0.5]))
            .expect("insert");
        zero_archive
            .insert(make_test_entry(2, 0.0, vec![1.5]))
            .expect("insert");

        let zero_metrics = zero_archive.metrics();
        assert_eq!(zero_metrics.total_cells, 4);
        assert_eq!(zero_metrics.occupied_cells, 2);
        assert_eq!(zero_metrics.coverage, 0.5);
        assert_eq!(zero_metrics.qd_score_raw, 0.0);
        assert_eq!(zero_metrics.qd_score_norm, 0.0);
        assert_eq!(zero_metrics.occupancy_entropy, 1.0); // same occupancy -> same entropy
        assert_eq!(zero_metrics.mean_quality, 0.0);
        assert_eq!(zero_metrics.max_quality, Some(0.0));
        assert!(!zero_metrics.qd_score_norm.is_nan());
        assert!(!zero_metrics.occupancy_entropy.is_nan());
    }

    #[test]
    fn test_compensated_sum_permutation_invariance() {
        let axis = Axis::new("speed", PhenotypeFeature::MeanSpeed, (0.0, 30.0), 30).expect("axis");
        let space = BehaviorSpaceV0 {
            version: BEHAVIOR_SPACE_SCHEMA_VERSION_V0,
            axes: vec![axis],
        };

        // 20 diverse floats across orders of magnitude
        let qualities = [
            1e-7_f32,
            2.5,
            1000.0,
            0.003,
            42.125,
            0.00005,
            99.9,
            0.1,
            777.7,
            13.333,
            0.000_000_1,
            5.55,
            300.0,
            0.04,
            12.0,
            0.0008,
            55.5,
            1.11,
            888.8,
            7.77,
        ];
        let entries: Vec<ArchiveEntry> = qualities
            .iter()
            .enumerate()
            .map(|(i, &q)| {
                #[allow(clippy::cast_precision_loss)]
                let pos = i as f32 + 0.5;
                make_test_entry((i + 1) as u64, q, vec![pos])
            })
            .collect();

        // Compute baseline qd_score_raw
        let mut baseline_archive =
            MapElitesArchive::new(space.clone(), QualityMetric::LifetimeIntake, 100, 100_000)
                .expect("archive");
        for entry in &entries {
            baseline_archive.insert(entry.clone()).expect("insert");
        }
        let expected_raw = baseline_archive.qd_score_raw();
        let expected_bits = expected_raw.to_bits();

        // Test 20 different insertion permutations
        for seed_perm in 1..=20u64 {
            let mut permuted = entries.clone();
            // Deterministic Fisher-Yates shuffle using simple LCG
            let mut state = seed_perm
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            for i in (1..permuted.len()).rev() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let j = (state >> 33) as usize % (i + 1);
                permuted.swap(i, j);
            }

            let mut test_archive =
                MapElitesArchive::new(space.clone(), QualityMetric::LifetimeIntake, 100, 100_000)
                    .expect("archive");
            for entry in permuted {
                test_archive.insert(entry).expect("insert permuted");
            }

            let test_raw = test_archive.qd_score_raw();
            assert_eq!(
                test_raw.to_bits(),
                expected_bits,
                "permutation {seed_perm} produced different bits for qd_score_raw: {test_raw} vs {expected_raw}"
            );
        }
    }

    #[test]
    fn test_archive_diff_and_typed_refusal() {
        let axis = Axis::new("speed", PhenotypeFeature::MeanSpeed, (0.0, 4.0), 4).expect("axis");
        let space = BehaviorSpaceV0 {
            version: BEHAVIOR_SPACE_SCHEMA_VERSION_V0,
            axes: vec![axis],
        };

        // Archive A: cells 0 (q=1.0), 1 (q=2.0), 2 (q=3.0)
        let mut archive_a =
            MapElitesArchive::new(space.clone(), QualityMetric::LifetimeIntake, 100, 100_000)
                .expect("archive a");
        archive_a
            .insert(make_test_entry(1, 1.0, vec![0.5]))
            .expect("insert");
        archive_a
            .insert(make_test_entry(2, 2.0, vec![1.5]))
            .expect("insert");
        archive_a
            .insert(make_test_entry(3, 3.0, vec![2.5]))
            .expect("insert");

        // Archive B: cells 1 (q=2.5, improved), 2 (q=1.5, regressed), 3 (q=4.0, only in B)
        let mut archive_b =
            MapElitesArchive::new(space.clone(), QualityMetric::LifetimeIntake, 100, 100_000)
                .expect("archive b");
        archive_b
            .insert(make_test_entry(20, 2.5, vec![1.5]))
            .expect("insert");
        archive_b
            .insert(make_test_entry(30, 1.5, vec![2.5]))
            .expect("insert");
        archive_b
            .insert(make_test_entry(40, 4.0, vec![3.5]))
            .expect("insert");

        let diff = archive_a.diff(&archive_b).expect("diff");
        assert_eq!(diff.only_in_a, vec![CellId(0)]);
        assert_eq!(diff.only_in_b, vec![CellId(3)]);
        assert_eq!(diff.improved_in_b.len(), 1);
        assert_eq!(diff.improved_in_b[0].cell_id, CellId(1));
        assert_eq!(diff.improved_in_b[0].quality_a, 2.0);
        assert_eq!(diff.improved_in_b[0].quality_b, 2.5);
        assert!((diff.improved_in_b[0].delta - 0.5).abs() < 1e-5);

        assert_eq!(diff.regressed_in_b.len(), 1);
        assert_eq!(diff.regressed_in_b[0].cell_id, CellId(2));
        assert_eq!(diff.regressed_in_b[0].quality_a, 3.0);
        assert_eq!(diff.regressed_in_b[0].quality_b, 1.5);
        assert!((diff.regressed_in_b[0].delta - (-1.5)).abs() < 1e-5);

        assert_eq!(diff.unchanged, Vec::new());

        // Identical diff
        let self_diff = archive_a.diff(&archive_a).expect("self diff");
        assert_eq!(self_diff.only_in_a, Vec::new());
        assert_eq!(self_diff.only_in_b, Vec::new());
        assert_eq!(self_diff.improved_in_b, Vec::new());
        assert_eq!(self_diff.regressed_in_b, Vec::new());
        assert_eq!(self_diff.unchanged.len(), 3);

        // Version mismatch refusal
        let mut wrong_version_space = space.clone();
        wrong_version_space.version = 1;
        let archive_wrong_version = MapElitesArchive::new(
            wrong_version_space,
            QualityMetric::LifetimeIntake,
            100,
            100_000,
        )
        .expect("archive");
        let version_err = archive_a
            .diff(&archive_wrong_version)
            .expect_err("version mismatch");
        assert_eq!(
            version_err,
            QdDiffError::SpaceVersionMismatch { a: 0, b: 1 }
        );

        // Metric mismatch refusal
        let archive_wrong_metric =
            MapElitesArchive::new(space, QualityMetric::OffspringCount, 100, 100_000)
                .expect("archive");
        let metric_err = archive_a
            .diff(&archive_wrong_metric)
            .expect_err("metric mismatch");
        assert_eq!(
            metric_err,
            QdDiffError::QualityMetricMismatch {
                a: QualityMetric::LifetimeIntake,
                b: QualityMetric::OffspringCount,
            }
        );

        // Axis mismatch refusal
        let different_axis =
            Axis::new("diet", PhenotypeFeature::DietTendency, (0.0, 1.0), 4).expect("axis");
        let different_space = BehaviorSpaceV0 {
            version: BEHAVIOR_SPACE_SCHEMA_VERSION_V0,
            axes: vec![different_axis],
        };
        let archive_different_axis =
            MapElitesArchive::new(different_space, QualityMetric::LifetimeIntake, 100, 100_000)
                .expect("archive");
        let axis_err = archive_a
            .diff(&archive_different_axis)
            .expect_err("axis mismatch");
        assert!(matches!(axis_err, QdDiffError::AxisMismatch { .. }));
    }

    #[test]
    fn test_cell_selectors() {
        let axis = Axis::new("speed", PhenotypeFeature::MeanSpeed, (0.0, 4.0), 4).expect("axis");
        let space = BehaviorSpaceV0 {
            version: BEHAVIOR_SPACE_SCHEMA_VERSION_V0,
            axes: vec![axis],
        };
        let mut archive = MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 100, 100_000)
            .expect("archive");

        // Cell 0: q=10.0, center=0.5, UID=1
        // Cell 1: q=50.0, center=1.5, UID=2
        // Cell 2: q=30.0, center=2.5, UID=3
        archive
            .insert(make_test_entry(1, 10.0, vec![0.5]))
            .expect("insert");
        archive
            .insert(make_test_entry(2, 50.0, vec![1.5]))
            .expect("insert");
        archive
            .insert(make_test_entry(3, 30.0, vec![2.5]))
            .expect("insert");

        // 1. Selector::All
        assert_eq!(
            archive.select_cells(&CellSelector::All),
            vec![CellId(0), CellId(1), CellId(2)]
        );

        // 2. Selector::TopKByQuality(2) -> cells 1 (q=50) and 2 (q=30)
        assert_eq!(
            archive.select_cells(&CellSelector::TopKByQuality(2)),
            vec![CellId(1), CellId(2)]
        );

        // 3. Selector::AxisRange -> range [1.0, 3.0] matches cells 1 and 2
        assert_eq!(
            archive.select_cells(&CellSelector::AxisRange(vec![(0, 1.0, 3.0)])),
            vec![CellId(1), CellId(2)]
        );

        // 4. Selector::Explicit -> matches existing cells, filters non-existent cells without error
        assert_eq!(
            archive.select_cells(&CellSelector::Explicit(vec![
                CellId(0),
                CellId(999), // does not exist
                CellId(2)
            ])),
            vec![CellId(0), CellId(2)]
        );

        // 5. select_entries
        let entries = archive.select_entries(&CellSelector::TopKByQuality(1));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].uid, AgentUid(2));
        assert_eq!(entries[0].quality, 50.0);
    }

    #[test]
    fn test_archive_export_import_bundle_and_json_roundtrip() {
        let axis0 = Axis::new("speed", PhenotypeFeature::MeanSpeed, (0.0, 4.0), 4).expect("axis0");
        let axis1 =
            Axis::new("diet", PhenotypeFeature::DietTendency, (0.0, 1.0), 2).expect("axis1");
        let space = BehaviorSpaceV0 {
            version: BEHAVIOR_SPACE_SCHEMA_VERSION_V0,
            axes: vec![axis0, axis1],
        };
        let mut archive = MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 100, 100_000)
            .expect("archive");

        archive
            .insert(make_test_entry(1, 15.5, vec![0.5, 0.25]))
            .expect("insert 1");
        archive
            .insert(make_test_entry(2, 42.0, vec![2.5, 0.75]))
            .expect("insert 2");
        archive
            .insert(make_test_entry(3, 8.0, vec![3.5, 0.1]))
            .expect("insert 3");

        let metrics_orig = archive.metrics();
        assert_eq!(metrics_orig.occupied_cells, 3);

        // 1. Bundle roundtrip
        let bundle = archive.export_bundle("run_alpha");
        assert_eq!(bundle.run_id, "run_alpha");
        let restored = MapElitesArchive::from_bundle(bundle).expect("from_bundle");
        assert_eq!(archive.cells, restored.cells);
        assert_eq!(metrics_orig, restored.metrics());

        // 2. JSON roundtrip
        let json_str = archive.export_json("run_alpha").expect("export_json");
        let (run_id_json, json_restored) =
            MapElitesArchive::import_json(&json_str).expect("import_json");
        assert_eq!(run_id_json, "run_alpha");
        assert_eq!(archive.cells, json_restored.cells);
        assert_eq!(metrics_orig, json_restored.metrics());
    }

    #[test]
    fn test_archive_export_import_csv_roundtrip() {
        let axis = Axis::new("speed", PhenotypeFeature::MeanSpeed, (0.0, 4.0), 4).expect("axis");
        let space = BehaviorSpaceV0 {
            version: BEHAVIOR_SPACE_SCHEMA_VERSION_V0,
            axes: vec![axis],
        };
        let mut archive = MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 50, 50_000)
            .expect("archive");

        archive
            .insert(make_test_entry(10, 20.0, vec![0.5]))
            .expect("insert");
        archive
            .insert(make_test_entry(20, 60.0, vec![1.5]))
            .expect("insert");

        let metrics_orig = archive.metrics();

        let mut csv_buf = Vec::new();
        let rows = archive
            .export_csv("test", &mut csv_buf)
            .expect("export_csv");
        assert_eq!(rows, 2);

        let csv_text = String::from_utf8(csv_buf.clone()).expect("valid utf8");
        assert!(csv_text.starts_with("# PROVENANCE: "));
        assert!(csv_text.contains("\"run_id\":\"test\""));

        let (run_id, imported) =
            MapElitesArchive::import_csv(csv_buf.as_slice(), 50_000).expect("import_csv");
        assert_eq!(run_id, "test");
        assert_eq!(archive.cells, imported.cells);
        assert_eq!(metrics_orig, imported.metrics());
    }

    #[test]
    fn test_archive_import_csv_malformed_errors() {
        // Missing provenance line
        let invalid_csv = "cell_id,uid,tick_inserted,quality,genome_version,parent_uid,generation,descriptor,genome\n";
        let err = MapElitesArchive::import_csv(invalid_csv.as_bytes(), 50_000).unwrap_err();
        assert!(matches!(err, QdError::Serialization(_)));

        // Empty CSV
        let empty_csv = "";
        let err = MapElitesArchive::import_csv(empty_csv.as_bytes(), 50_000).unwrap_err();
        assert!(matches!(err, QdError::Serialization(_)));
    }

    #[test]
    fn test_format_stats_report_and_logging() {
        let axis = Axis::new("speed", PhenotypeFeature::MeanSpeed, (0.0, 4.0), 4).expect("axis");
        let space = BehaviorSpaceV0 {
            version: BEHAVIOR_SPACE_SCHEMA_VERSION_V0,
            axes: vec![axis],
        };
        let mut archive = MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 10, 50_000)
            .expect("archive");
        archive
            .insert(make_test_entry(1, 15.0, vec![1.0]))
            .expect("insert");

        let report = format_stats_report(&archive, "test_run_123");
        assert!(report.contains("test_run_123"));
        assert!(report.contains("Coverage:"));
        assert!(report.contains("Normalized QD:"));
        assert!(report.contains("lifetime_intake"));

        // Logging invocation shouldn't panic
        archive.log_metrics("test_run_123");
    }

    #[test]
    fn test_diff_csv_and_diff_json_success_and_failures() {
        let axis = Axis::new("speed", PhenotypeFeature::MeanSpeed, (0.0, 4.0), 4).expect("axis");
        let space = BehaviorSpaceV0 {
            version: BEHAVIOR_SPACE_SCHEMA_VERSION_V0,
            axes: vec![axis],
        };
        let mut archive_a =
            MapElitesArchive::new(space.clone(), QualityMetric::LifetimeIntake, 10, 50_000)
                .expect("archive");
        let mut archive_b = MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 10, 50_000)
            .expect("archive");

        archive_a
            .insert(make_test_entry(10, 20.0, vec![0.5]))
            .expect("insert");
        archive_a
            .insert(make_test_entry(20, 50.0, vec![1.5]))
            .expect("insert");

        // B has cell 0 with higher quality (improved), lacks cell 1 (only_in_a), and has cell 2 (only_in_b)
        archive_b
            .insert(make_test_entry(10, 35.0, vec![0.5]))
            .expect("insert");
        archive_b
            .insert(make_test_entry(30, 40.0, vec![2.5]))
            .expect("insert");

        // 1. CSV diff
        let mut csv_a = Vec::new();
        let mut csv_b = Vec::new();
        archive_a.export_csv("run_a", &mut csv_a).expect("csv a");
        archive_b.export_csv("run_b", &mut csv_b).expect("csv b");

        let diff_csv_res = diff_csv(csv_a.as_slice(), csv_b.as_slice(), 50_000).expect("diff csv");
        assert_eq!(diff_csv_res.improved_in_b.len(), 1);
        assert_eq!(diff_csv_res.improved_in_b[0].cell_id, CellId(0));
        assert_eq!(diff_csv_res.only_in_a.len(), 1);
        assert_eq!(diff_csv_res.only_in_a[0], CellId(1));
        assert_eq!(diff_csv_res.only_in_b.len(), 1);
        assert_eq!(diff_csv_res.only_in_b[0], CellId(2));

        // 2. JSON diff
        let json_a = archive_a.export_json("run_a").expect("json a");
        let json_b = archive_b.export_json("run_b").expect("json b");
        let diff_json_res = diff_json(&json_a, &json_b).expect("diff json");
        assert_eq!(diff_json_res, diff_csv_res);

        // 3. Version mismatch failure
        let diff_version_err = diff_json(&json_a, "{\"run_id\":\"b\",\"space\":{\"version\":999,\"axes\":[]},\"quality_metric\":\"LifetimeIntake\",\"min_lifetime_ticks\":10,\"max_bytes\":50000,\"cells\":[]}").unwrap_err();
        assert!(matches!(diff_version_err, QdDiffError::Serialization(_)));

        // 4. Missing run / malformed CSV
        let malformed_err = diff_csv(b"".as_slice(), csv_b.as_slice(), 50_000).unwrap_err();
        assert!(matches!(malformed_err, QdDiffError::Serialization(_)));
    }
}
