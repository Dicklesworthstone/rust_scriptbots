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
    /// Serialization or deserialization error.
    #[error("archive serialization error: {0}")]
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
            Self::SpikeUsageRate => "event_per_tick",
            Self::GiveRate => "event_per_tick",
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
    /// Number of uniform discrete bins along this axis (1..=256).
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
        let bins = self.bins as u32;

        let clamped = value.clamp(lo, hi);
        if clamped >= hi {
            Ok(self.bins.saturating_sub(1))
        } else {
            let span = hi - lo;
            let frac = ((clamped - lo) / span).clamp(0.0, 1.0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let bin = (frac * bins as f32).floor() as u32;
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
    pub fn new(version: u16, axes: Vec<Axis>) -> Self {
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
                .checked_mul(axis.bins as u64)
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
            let bin = axis.discretize(val, i)? as u64;
            let term = bin
                .checked_mul(multiplier)
                .ok_or(QdError::MixedRadixOverflow)?;
            cell_id = cell_id
                .checked_add(term)
                .ok_or(QdError::MixedRadixOverflow)?;
            multiplier = multiplier
                .checked_mul(axis.bins as u64)
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
            let bins = axis.bins as u64;
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
    pub fn new(values: Vec<f32>) -> Self {
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
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether this descriptor has zero dimensions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
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
    pub fn compute(
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
                        let net_delta = entry_bytes.saturating_sub(incumbent_bytes);
                        if self.current_bytes.saturating_add(net_delta) > self.max_archive_bytes {
                            return Err(QdError::ByteCapExceeded {
                                current_bytes: self.current_bytes,
                                entry_bytes: net_delta,
                                cap_bytes: self.max_archive_bytes,
                            });
                        }
                        let displaced_uid = incumbent.uid;
                        let displaced_quality = incumbent.quality;
                        let uid = entry.uid;
                        let quality = entry.quality;
                        let tick_inserted = entry.tick_inserted.0;
                        if entry_bytes >= incumbent_bytes {
                            self.current_bytes = self.current_bytes.saturating_add(net_delta);
                        } else {
                            self.current_bytes = self
                                .current_bytes
                                .saturating_sub(incumbent_bytes.saturating_sub(entry_bytes));
                        }
                        self.cells.insert(cell_id, entry);
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
                        let net_delta = entry_bytes.saturating_sub(incumbent_bytes);
                        if self.current_bytes.saturating_add(net_delta) > self.max_archive_bytes {
                            return Err(QdError::ByteCapExceeded {
                                current_bytes: self.current_bytes,
                                entry_bytes: net_delta,
                                cap_bytes: self.max_archive_bytes,
                            });
                        }
                        let displaced_uid = incumbent.uid;
                        let displaced_quality = incumbent.quality;
                        let uid = entry.uid;
                        let quality = entry.quality;
                        let tick_inserted = entry.tick_inserted.0;
                        if entry_bytes >= incumbent_bytes {
                            self.current_bytes = self.current_bytes.saturating_add(net_delta);
                        } else {
                            self.current_bytes = self
                                .current_bytes
                                .saturating_sub(incumbent_bytes.saturating_sub(entry_bytes));
                        }
                        self.cells.insert(cell_id, entry);
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

    /// Compute Quality-Diversity (QD) score (sum of all elite qualities).
    ///
    /// Summation is done in sorted [`CellId`] iteration order.
    #[must_use]
    pub fn qd_score(&self) -> f64 {
        self.cells.values().map(|r| r.quality as f64).sum()
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
            let ratio = (count / total as f64) as f32;
            ratio.clamp(0.0, 1.0)
        }
    }

    /// Arithmetic mean quality across all occupied cells.
    #[must_use]
    pub fn mean_quality(&self) -> f32 {
        if self.cells.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let mean = self.qd_score() / self.cells.len() as f64;
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
                    (2.0 * std::f32::consts::PI - diff).max(0.0)
                } else {
                    diff
                };
                self.turn_angle_sum += wrapped;
            }
            self.last_heading = Some(heading);
        }
    }

    /// Record that this agent successfully produced an offspring.
    pub fn record_offspring(&mut self) {
        self.offspring_count = self.offspring_count.saturating_add(1);
    }

    /// Extract a single feature's accumulated average for behavior space mapping.
    #[must_use]
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
            parent_uid: self.provenance.parent_uid.map(|u| u.get()),
            generation: self.provenance.generation.0,
        })
    }
}

impl ArchiveCellRow {
    /// Convert this persistence row back into a cell ID and archive entry.
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
        let entry = ArchiveEntry {
            uid: AgentUid(self.uid),
            tick_inserted: Tick(self.tick_inserted),
            descriptor,
            quality: self.quality as f32,
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
}
