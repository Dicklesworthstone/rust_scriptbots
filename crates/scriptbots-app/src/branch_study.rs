//! Paired checkpoint-branch intervention studies for evolutionary simulations (bd-2z0.11.4).
//!
//! # Scientific Design (Plan Phase 4.5)
//!
//! A paired checkpoint-branch study establishes causal attribution for ecological and evolutionary
//! interventions by eliminating founding-state and pre-intervention divergence:
//!
//! 1. **Common Pre-Intervention Baseline**: A simulation world is seeded and stepped until
//!    an exact boundary `checkpoint_tick`, or an existing saved scientific checkpoint is
//!    validated without warmup. Its canonical [`scriptbots_core::WorldCheckpointV1`] establishes
//!    an authoritative `pre_intervention_digest` for new isolated futures.
//! 2. **Matched Branch Restoration**: The Control branch and every intervention branch are restored
//!    from that identical checkpoint using a freshly prepared, bit-identical [`scriptbots_core::BrainRegistry`].
//!    Every branch asserts and verifies `restored_digest == pre_intervention_digest` before any
//!    intervention occurs.
//! 3. **Exact Command Timing**: Interventions are applied after the completed scheduled tick
//!    boundary (defaulting to `checkpoint_tick`), before the transition to the next tick.
//! 4. **First Divergence Tracing**: For every intervention branch, simulation transitions are traced
//!    tick-by-tick against the control branch to locate the first observed tick and differing digest lane
//!    (e.g., `effects`, `config`, `food`, `agents`) where divergence first occurred.
//! 5. **Quantitative Outcome Measures**:
//!    - **Extinction**: Detects population collapse, records time to extinction, and minimum population.
//!    - **Recovery**: Evaluates post-perturbation rebound against pre-intervention baseline and trough.
//!    - **Hysteresis**: Evaluates path dependency, integrated trajectory deficit, and persistent terminal gap.
//!    - **Descriptive Trajectory Comparisons**: Computes paired mean differences and their temporal
//!      standard deviation across the recorded horizon. One seed provides one independent pair;
//!      repeated ticks do not justify confidence intervals or significance tests. The separate
//!      [`ReplicatedBranchStudyOrchestrator`] runs independently initialized matched seed pairs
//!      and uses [`compute_paired_effect`] only on one terminal endpoint per seed and branch.
//! 6. **Verifiable Artifact Bundles**: Every branch packages its metrics and evidence, then checks
//!    bounded manifest/artifact integrity ([`scriptbots_storage::verify_run_bundle_bounded`]). This
//!    does not establish that the bundle contains a restorable simulation owner checkpoint.
//!
//! The existing `branch-study --plan PLAN --out DIRECTORY` command accepts an optional
//! `source_checkpoint` path in its JSON plan. Relative source paths use the invocation's
//! working directory; omitting the field retains independently seeded warmup.
//!
//! Raw scientific checkpoints omit selection history, map temperature and the scrambling RNG.
//! Sources and treatment configurations therefore require Fitness selection and disabled
//! communication scrambling. Imported map sources and archive-dependent spawning treatments
//! are refused. These restrictions apply to seeded and imported checkpoint studies.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use rand::{Rng, SeedableRng};
use scriptbots_core::{
    BrainRegistry, ControlCommand, DynamicWorldSnapshot, EvolutionSelectionMode, Intervention,
    Region, ScriptBotsConfig, WorldCheckpointError, WorldCheckpointV1, WorldDigestV1, WorldState,
    WorldStateError, apply_control_command,
};
use scriptbots_runtime::RunId;
use scriptbots_storage::{
    RunBundleV1, RunManifestRecord, bundle::RunBundleVerificationLimits,
    bundle::verify_run_bundle_bounded, create_run_bundle_from_artifacts,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::brains::{BrainPreset, install_brains};
use crate::lab::stats::bootstrap_ci;
use crate::{resolve_scheduled_config_patch, seed_founding_population};

/// Report revision identifying seeded warmup versus an exact retained source checkpoint.
pub const BRANCH_STUDY_REPORT_SCHEMA_VERSION: u16 = 4;
/// Replicated study report contract: one terminal endpoint per independently initialized seed.
pub const REPLICATED_BRANCH_STUDY_REPORT_SCHEMA_VERSION: u16 = 1;

/// Maximum branches retained in one study, including its control.
pub const MAX_STUDY_BRANCHES: usize = 32;
/// Maximum independently initialized seed pairs in one replicated study.
pub const MAX_STUDY_REPLICATES: usize = 32;
/// Maximum recorded transitions per branch.
pub const MAX_STUDY_HORIZON_TICKS: u64 = 10_000;
/// Maximum baseline plus branch transitions admitted by one study.
pub const MAX_STUDY_TOTAL_TRANSITIONS: u64 = 1_000_000;
/// Maximum food/terrain grid cells, using the core constructor's validated dimensions.
pub const MAX_STUDY_GRID_CELLS: u64 = 262_144;
/// Maximum population floor, per-tick injection count, or observed live population.
pub const MAX_STUDY_POPULATION: u64 = 10_000;
const MAX_STUDY_HISTORY_ENTRIES: u64 = 10_000;
const MAX_STUDY_IDENTIFIER_BYTES: usize = 128;
const MAX_STUDY_DESCRIPTION_BYTES: usize = 4096;
const MAX_STUDY_ARTIFACT_BYTES: u64 = 50 * 1024 * 1024;
const SOURCE_CHECKPOINT_ARTIFACT: &str = "source_checkpoint.bin";

/// The producer and managed download catalog verify the same bundle contract.
pub(crate) const STUDY_BUNDLE_VERIFICATION_LIMITS: RunBundleVerificationLimits =
    RunBundleVerificationLimits {
        max_manifest_bytes: 10 * 1024 * 1024,
        max_artifacts: 100,
        max_artifact_bytes: MAX_STUDY_ARTIFACT_BYTES,
        max_total_artifact_bytes: 100 * 1024 * 1024,
    };

/// Specific intervention perturbations for branch studies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "params", deny_unknown_fields)]
pub enum StudyIntervention {
    /// Suppress food regrowth in a region for a specified duration.
    Drought {
        /// Affected region (`None` defaults to entire world).
        region: Option<Region>,
        /// Number of simulation ticks to retain the drought.
        duration_ticks: u32,
        /// Growth multiplier (0.0 = total halt).
        growth_scale: f32,
    },
    /// Resource shock: sudden influx (bloom) or severe burn/depletion (meteor/scorch).
    ResourceShock {
        /// Affected region (`None` defaults to entire world).
        region: Option<Region>,
        /// Food amount added per cell in bloom.
        bloom_amount: f32,
        /// Fraction of cell food destroyed in scorch `[0.0, 1.0]`.
        scorch: f32,
    },
    /// Environmental temperature shift and discomfort penalty.
    TemperatureShift {
        /// Health drain multiplier applied when discomfort occurs.
        discomfort_rate: f32,
        /// Comfort band around preferred temperature.
        comfort_band: f32,
    },
    /// Enforce a closed world (no newcomer spawns) or reopen it.
    ClosedWorld {
        /// The closed world flag.
        closed: bool,
    },
    /// Cooperation economics parameter adjustments.
    Cooperation(CooperationParams),
    /// Direct core intervention command; archive-dependent spawning is unsupported.
    Core(Intervention),
    /// Arbitrary configuration patch resolved via pure precedence.
    ConfigPatch(serde_json::Value),
}

/// Parameters for cooperation and altruistic sharing studies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CooperationParams {
    /// Fraction of energy shared per neighbor when donating `[0.0, 1.0]`.
    pub food_sharing_rate: f32,
    /// Constant amount of energy transferred during altruistic sharing.
    pub food_transfer_rate: f32,
    /// Distance threshold for altruistic sharing interactions.
    pub food_sharing_distance: f32,
    /// Optional partner chance for crossover reproduction `[0.0, 1.0]`.
    pub reproduction_partner_chance: Option<f32>,
}

impl StudyIntervention {
    /// Validate the intervention parameters against basic physical bounds and world dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`BranchStudyError::InvalidIntervention`] if any parameter is malformed or out of range.
    pub fn validate(&self, world_width: u32, world_height: u32) -> Result<(), BranchStudyError> {
        match self {
            Self::Drought {
                region,
                duration_ticks,
                growth_scale,
            } => {
                if *duration_ticks == 0 {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "drought".to_string(),
                        reason: "duration_ticks must be > 0".to_string(),
                    });
                }
                if !growth_scale.is_finite() || *growth_scale < 0.0 {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "drought".to_string(),
                        reason: "growth_scale must be finite and >= 0.0".to_string(),
                    });
                }
                if let Some(r) = region {
                    let core_intervention = Intervention::Drought {
                        region: *r,
                        ticks: *duration_ticks,
                        growth_scale: *growth_scale,
                    };
                    core_intervention
                        .validate_for_world(world_width, world_height)
                        .map_err(|e| BranchStudyError::InvalidIntervention {
                            branch_id: "drought".to_string(),
                            reason: e.to_string(),
                        })?;
                }
            }
            Self::ResourceShock {
                region,
                bloom_amount,
                scorch,
            } => {
                if !bloom_amount.is_finite() || *bloom_amount < 0.0 {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "resource_shock".to_string(),
                        reason: "bloom_amount must be finite and >= 0.0".to_string(),
                    });
                }
                if !scorch.is_finite() || *scorch < 0.0 || *scorch > 1.0 {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "resource_shock".to_string(),
                        reason: "scorch must be finite and within [0.0, 1.0]".to_string(),
                    });
                }
                if *bloom_amount == 0.0 && *scorch == 0.0 {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "resource_shock".to_string(),
                        reason: "at least one of bloom_amount or scorch must be > 0.0".to_string(),
                    });
                }
                if let Some(r) = region {
                    if *bloom_amount > 0.0 {
                        Intervention::Bloom {
                            region: *r,
                            amount: *bloom_amount,
                        }
                        .validate_for_world(world_width, world_height)
                        .map_err(|e| {
                            BranchStudyError::InvalidIntervention {
                                branch_id: "resource_shock".to_string(),
                                reason: e.to_string(),
                            }
                        })?;
                    }
                    if *scorch > 0.0 {
                        Intervention::Meteor {
                            region: *r,
                            lethality: 0.0,
                            scorch: *scorch,
                        }
                        .validate_for_world(world_width, world_height)
                        .map_err(|e| {
                            BranchStudyError::InvalidIntervention {
                                branch_id: "resource_shock".to_string(),
                                reason: e.to_string(),
                            }
                        })?;
                    }
                }
            }
            Self::TemperatureShift {
                discomfort_rate,
                comfort_band,
            } => {
                if !discomfort_rate.is_finite() || *discomfort_rate < 0.0 {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "temperature_shift".to_string(),
                        reason: "discomfort_rate must be finite and >= 0.0".to_string(),
                    });
                }
                if !comfort_band.is_finite() || *comfort_band < 0.0 {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "temperature_shift".to_string(),
                        reason: "comfort_band must be finite and >= 0.0".to_string(),
                    });
                }
            }
            Self::ClosedWorld { .. } => {}
            Self::Cooperation(params) => {
                if !params.food_sharing_rate.is_finite()
                    || params.food_sharing_rate < 0.0
                    || params.food_sharing_rate > 1.0
                {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "cooperation".to_string(),
                        reason: "food_sharing_rate must be finite and within [0.0, 1.0]"
                            .to_string(),
                    });
                }
                if !params.food_transfer_rate.is_finite() || params.food_transfer_rate < 0.0 {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "cooperation".to_string(),
                        reason: "food_transfer_rate must be finite and >= 0.0".to_string(),
                    });
                }
                if !params.food_sharing_distance.is_finite() || params.food_sharing_distance < 0.0 {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "cooperation".to_string(),
                        reason: "food_sharing_distance must be finite and >= 0.0".to_string(),
                    });
                }
                if let Some(partner) = params.reproduction_partner_chance
                    && (!partner.is_finite() || partner < 0.0 || partner > 1.0)
                {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "cooperation".to_string(),
                        reason: "reproduction_partner_chance must be finite and within [0.0, 1.0]"
                            .to_string(),
                    });
                }
            }
            Self::Core(intervention) => {
                if matches!(intervention, Intervention::SpawnFromArchive { .. }) {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "core".to_string(),
                        reason: "raw checkpoints omit the behavioral archive required by SpawnFromArchive".to_owned(),
                    });
                }
                intervention
                    .validate_for_world(world_width, world_height)
                    .map_err(|e| BranchStudyError::InvalidIntervention {
                        branch_id: "core".to_string(),
                        reason: e.to_string(),
                    })?;
            }
            Self::ConfigPatch(patch) => {
                if !patch.is_object() {
                    return Err(BranchStudyError::InvalidIntervention {
                        branch_id: "config_patch".to_string(),
                        reason: "config patch must be a JSON object".to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Specification for one branch in a paired checkpoint study.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyBranchSpec {
    /// Unique identifier for this branch (e.g. "control", "drought_severe").
    pub branch_id: String,
    /// Human-readable description of the branch condition.
    pub description: String,
    /// Intervention to apply. `None` defines an unperturbed Control branch.
    pub intervention: Option<StudyIntervention>,
    /// Completed boundary after which the intervention is applied, before the next tick.
    /// Defaults to `plan.checkpoint_tick`; must precede the final observed tick.
    /// Unperturbed control branches cannot declare a scheduled tick.
    pub scheduled_tick: Option<u64>,
}

/// Complete plan for a paired checkpoint-branch study.
///
/// The existing `branch-study --plan PLAN --out DIRECTORY` command accepts
/// `source_checkpoint` in this JSON plan. When present, the path names an existing
/// scientific checkpoint; relative paths are resolved from the invocation's working
/// directory. The plan's config, tick and root seed must match that cut exactly.
/// Omitting the field retains independently seeded warmup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BranchStudyPlan {
    /// Unique identifier for this study.
    pub study_id: String,
    /// Research question or scientific hypothesis being tested.
    pub description: String,
    /// Founding RNG seed for warmup, or the saved world's actual root RNG seed for import.
    pub base_seed: u64,
    /// Simulation tick at which the shared baseline is snapshotted.
    pub checkpoint_tick: u64,
    /// Number of simulation ticks to run each branch post-checkpoint.
    pub horizon_ticks: u64,
    /// Architecture preset for agent brains.
    pub brain_preset: BrainPreset,
    /// Base simulation configuration; imported cuts require its exact saved value.
    pub config: ScriptBotsConfig,
    /// Optional saved scientific checkpoint, relative to the invocation's working directory.
    /// Its canonical bytes are retained once at the study root; the declared path is preserved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_checkpoint: Option<PathBuf>,
    /// Set of branches to evaluate (must include at least one control branch).
    pub branches: Vec<StudyBranchSpec>,
}

impl BranchStudyPlan {
    fn effective_config(&self) -> ScriptBotsConfig {
        let mut config = self.config.clone();
        config.persistence_interval = 0;
        config.rng_seed = Some(self.base_seed);
        config
    }

    fn final_tick(&self) -> Result<u64, BranchStudyError> {
        self.checkpoint_tick
            .checked_add(self.horizon_ticks)
            .ok_or_else(|| {
                BranchStudyError::InvalidPlan(
                    "checkpoint_tick + horizon_ticks exceeds the u64 tick range".to_string(),
                )
            })
    }

    fn sample_count(&self) -> Result<usize, BranchStudyError> {
        usize::try_from(self.horizon_ticks)
            .ok()
            .and_then(|ticks| ticks.checked_add(1))
            .ok_or_else(|| {
                BranchStudyError::InvalidPlan(
                    "the study horizon exceeds the addressable trajectory length".to_string(),
                )
            })
    }

    fn total_transitions(&self) -> Result<u64, BranchStudyError> {
        let warmup = if self.source_checkpoint.is_some() {
            0
        } else {
            self.checkpoint_tick
        };
        self.branch_transitions()?
            .checked_add(warmup)
            .ok_or_else(|| BranchStudyError::InvalidPlan("study work count overflows".to_owned()))
    }

    fn branch_transitions(&self) -> Result<u64, BranchStudyError> {
        self.horizon_ticks
            .checked_mul(self.branches.len() as u64)
            .ok_or_else(|| BranchStudyError::InvalidPlan("study work count overflows".to_owned()))
    }

    /// Create a standard 5-branch study plus control covering:
    /// Drought, ResourceShock, TemperatureShift, ClosedWorld, and Cooperation.
    #[must_use]
    pub fn standard_five_branch(
        study_id: &str,
        base_seed: u64,
        checkpoint_tick: u64,
        horizon_ticks: u64,
    ) -> Self {
        let config = ScriptBotsConfig {
            persistence_interval: 0,
            rng_seed: Some(base_seed),
            ..Default::default()
        };

        Self {
            study_id: study_id.to_string(),
            description: "Standard 5-branch paired checkpoint intervention study".to_string(),
            base_seed,
            checkpoint_tick,
            horizon_ticks,
            brain_preset: BrainPreset::Mlp,
            config,
            source_checkpoint: None,
            branches: vec![
                StudyBranchSpec {
                    branch_id: "control".to_string(),
                    description: "Unperturbed baseline control".to_string(),
                    intervention: None,
                    scheduled_tick: None,
                },
                StudyBranchSpec {
                    branch_id: "drought".to_string(),
                    description: "Halt food regrowth for 40 ticks".to_string(),
                    intervention: Some(StudyIntervention::Drought {
                        region: None,
                        duration_ticks: 40,
                        growth_scale: 0.0,
                    }),
                    scheduled_tick: None,
                },
                StudyBranchSpec {
                    branch_id: "resource_shock".to_string(),
                    description: "Instantaneous food bloom (+3.0 per cell)".to_string(),
                    intervention: Some(StudyIntervention::ResourceShock {
                        region: None,
                        bloom_amount: 3.0,
                        scorch: 0.0,
                    }),
                    scheduled_tick: None,
                },
                StudyBranchSpec {
                    branch_id: "temperature_shift".to_string(),
                    description: "High temperature discomfort penalty rate (0.75)".to_string(),
                    intervention: Some(StudyIntervention::TemperatureShift {
                        discomfort_rate: 0.75,
                        comfort_band: 0.05,
                    }),
                    scheduled_tick: None,
                },
                StudyBranchSpec {
                    branch_id: "closed_world".to_string(),
                    description: "Enforce closed world with zero external newcomer injection"
                        .to_string(),
                    intervention: Some(StudyIntervention::ClosedWorld { closed: true }),
                    scheduled_tick: None,
                },
                StudyBranchSpec {
                    branch_id: "cooperation".to_string(),
                    description: "Enhanced food sharing rate (0.8) and transfer (0.4)".to_string(),
                    intervention: Some(StudyIntervention::Cooperation(CooperationParams {
                        food_sharing_rate: 0.8,
                        food_transfer_rate: 0.4,
                        food_sharing_distance: 50.0,
                        reproduction_partner_chance: Some(0.8),
                    })),
                    scheduled_tick: None,
                },
            ],
        }
    }
}

/// Metric sample recorded at a simulation tick boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchTickSample {
    /// Completed simulation tick.
    pub tick: u64,
    /// Live agent population.
    pub population: usize,
    /// Total ground food remaining across the world.
    pub food_total: f64,
    /// Offspring born since the preceding study sample; zero at the checkpoint sample.
    pub births: usize,
    /// Agent deaths since the preceding study sample; zero at the checkpoint sample.
    pub deaths: usize,
    /// Sum of all agent energy.
    pub total_energy: f32,
    /// Mean health across living agents (zero for an empty population, as in core summaries).
    pub average_health: f32,
    /// Spike hits since the preceding study sample; zero at the checkpoint sample.
    pub spike_hits: u32,
    /// Canonical overall world digest under the core's declared digest protocol.
    pub digest_overall: String,
}

/// Keep lane evidence attached to its observed metrics without expanding the CSV schema.
#[derive(Debug, Clone)]
struct BranchObservation {
    sample: BranchTickSample,
    digest: WorldDigestV1,
}

/// Record of the first observed tick and differing digest lane against Control at that same tick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FirstDivergence {
    /// Identifier of the diverged branch.
    pub branch_id: String,
    /// First simulation tick where the world digest differed from Control.
    pub tick: u64,
    /// First differing digest lane in diagnostic order (`config`, `effects`, `food`, `agents`, etc.).
    /// This does not claim to locate the first internal transition within that tick.
    pub stage: String,
    /// Control world lane hash at divergence.
    pub control_digest: String,
    /// Branch world lane hash at divergence.
    pub branch_digest: String,
    /// Tick at which the intervention was applied.
    pub command_applied_tick: u64,
    /// Explanatory diagnostic details.
    pub details: String,
}

/// Extinction outcome assessment for a branch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtinctionMeasure {
    /// Whether the population was 0 at or after the recovery baseline boundary.
    pub extinct: bool,
    /// First tick where population reached 0 (if extinct).
    pub extinction_tick: Option<u64>,
    /// Minimum population from the intervention boundary through the final tick (checkpoint for controls).
    pub min_population: usize,
    /// Final population at the end of the horizon.
    pub final_population: usize,
}

/// Population recovery assessment for a branch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveryMeasure {
    /// Intervention boundary before the command was applied, or checkpoint boundary for controls.
    pub baseline_tick: u64,
    /// Observed population at `baseline_tick` before intervention.
    pub pre_intervention_population: usize,
    /// Minimum population from `baseline_tick` through the final tick.
    pub trough_population: usize,
    /// Simulation tick at which trough occurred.
    pub trough_tick: u64,
    /// Final population at the end of the horizon.
    pub final_population: usize,
    /// Whether final population recovered to at least 80% of pre-intervention level.
    pub recovered: bool,
    /// Ratio of final population to pre-intervention population.
    pub recovery_ratio: f64,
    /// Number of ticks from trough until population reached >= 80% of baseline (if recovered).
    pub recovery_ticks: Option<u64>,
}

/// Hysteresis and dynamical path-dependency assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HysteresisMeasure {
    /// Integrated population deficit: sum(control_pop - branch_pop).
    pub integrated_trajectory_deficit: f64,
    /// Normalized trajectory difference: sum(|control_pop - branch_pop|) / sum(control_pop).
    pub normalized_trajectory_difference: f64,
    /// Terminal trajectory gap: (control_final - branch_final) / control_final.
    pub terminal_trajectory_gap: f64,
    /// Ratio of terminal gap to peak gap during horizon.
    pub hysteresis_index: f64,
    /// Whether the absolute terminal gap exceeded the descriptive 5% threshold (not a significance test).
    pub path_dependence_detected: bool,
}

/// Statistical unit represented by a paired comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BranchAnalysisBasis {
    /// Repeated observations of one matched seed pair; no replicated inference is supported.
    SingleSeedTrajectory {
        /// The sole independent simulation seed in this comparison.
        seed: u64,
    },
    /// Independently sampled matched replicates supplied by the caller.
    IndependentPairedReplicates {
        /// Number of independent pairs, not the number of observations within a run.
        pairs: usize,
    },
}

/// Paired comparison against Control with an explicit distinction between description and inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchEffectSize {
    /// Name of the metric (e.g. "population", "food_total").
    pub metric: String,
    /// Statistical unit and independence assumptions behind these observations.
    pub analysis_basis: BranchAnalysisBasis,
    /// Number of paired observations. For trajectories this includes the checkpoint and all recorded
    /// timepoints, including any before a delayed intervention; it is not an independent replicate count.
    pub n_samples: usize,
    /// Mean difference (branch minus control).
    pub mean_diff: f64,
    /// Sample standard deviation of differences.
    pub sd_diff: f64,
    /// Cohen's d for independent pairs (absent for trajectories or zero variance).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cohens_d: Option<f64>,
    /// Bias-corrected Hedges' g for independent pairs (absent for trajectories or zero variance).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hedges_g: Option<f64>,
    /// 95% bootstrap interval of the mean difference across independent pairs; absent for trajectories.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ci_95: Option<(f64, f64)>,
    /// Monte Carlo sign-flip p-value for independent pairs; absent for trajectories or fewer than two pairs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_value: Option<f64>,
}

/// Detailed outcome report for a single branch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchOutcomeReport {
    /// Branch identifier.
    pub branch_id: String,
    /// Branch description.
    pub description: String,
    /// Tick at which intervention was applied (if any).
    pub command_applied_tick: Option<u64>,
    /// Exact first divergence from Control (None if control or bit-identical).
    pub first_divergence: Option<FirstDivergence>,
    /// Extinction measure.
    pub extinction: ExtinctionMeasure,
    /// Recovery measure.
    pub recovery: RecoveryMeasure,
    /// Hysteresis measure.
    pub hysteresis: HysteresisMeasure,
    /// Descriptive metric comparisons against Control over the entire recorded trajectory.
    pub effect_sizes: BTreeMap<String, BranchEffectSize>,
    /// Final world digest overall hash.
    pub final_digest: String,
    /// Actual last recorded boundary, retained without substituting a trajectory average.
    pub terminal_sample: BranchTickSample,
    /// Path to the metrics/evidence artifact bundle directory.
    pub bundle_path: Option<String>,
    /// Whether the bundle passed bounded manifest/artifact integrity verification.
    /// This does not establish executable checkpoint reconstruction.
    pub bundle_verified: bool,
}

/// Starting-state provenance for a new isolated scientific study.
///
/// Saved checkpoints continue supported scientific state only. The source host's session,
/// scheduled commands, storage ownership and retained analytics are not transferred.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BranchStudySource {
    /// The study initialized its own population and advanced it to the requested boundary.
    SeededWarmup,
    /// The study restored the exact saved scientific state without reseeding or warmup.
    SavedCheckpoint {
        /// Exact source envelope, relative to the study output directory.
        artifact_path: String,
        /// Byte length of that retained canonical envelope.
        byte_size: u64,
        /// BLAKE3 checksum of the complete retained envelope.
        checksum_blake3: String,
    },
}

/// Master study report aggregating all branches, proofs, and summaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchStudyReport {
    /// Study identifier.
    pub study_id: String,
    /// Schema version ([`BRANCH_STUDY_REPORT_SCHEMA_VERSION`]).
    pub schema_version: u16,
    /// Base seed used.
    pub base_seed: u64,
    /// How the shared scientific state was obtained; imported cuts retain their exact bytes.
    pub source: BranchStudySource,
    /// Simulation tick of the shared checkpoint.
    pub checkpoint_tick: u64,
    /// Horizon ticks stepped post-checkpoint.
    pub horizon_ticks: u64,
    /// World digest overall hash at checkpoint boundary.
    pub pre_intervention_digest: String,
    /// Proof that every branch verified the identical pre-intervention digest.
    pub pre_intervention_digest_verified_for_all_branches: bool,
    /// Identifier of the control branch.
    pub control_branch_id: String,
    /// Reports for all branches (including Control).
    pub branch_reports: Vec<BranchOutcomeReport>,
    /// Human-readable Markdown summary.
    pub summary_markdown: String,
}

/// Explicit independent-seed extension of an existing checkpoint-branch study.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicatedBranchStudyPlan {
    /// Common configuration, branches, and timing. Its `base_seed` seeds statistical resampling;
    /// each replicate replaces both founding seed fields with its own entry in `seeds`.
    /// A declared source checkpoint cannot provide independent founding seeds.
    pub study: BranchStudyPlan,
    /// Unique founding seeds, in the declared execution and reduction order.
    pub seeds: Vec<u64>,
}

/// One independently initialized seed pair and its complete per-branch evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchStudyReplicate {
    /// Exact founding seed supplied to the per-seed executor.
    pub seed: u64,
    /// Per-seed report artifact relative to the replicated study directory.
    pub report_path: String,
    /// Exact per-seed result, including baseline digest, terminal endpoints and bundle receipts.
    pub report: BranchStudyReport,
}

/// Terminal endpoint comparisons from independently initialized matched seed pairs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicatedBranchStudyReport {
    /// Version of this replicated report contract.
    pub schema_version: u16,
    /// Exact retained inputs, including declared seed order and resampling seed.
    pub plan: ReplicatedBranchStudyPlan,
    /// The common terminal tick used for every paired endpoint.
    pub endpoint_tick: u64,
    /// Baseline plus branch transitions across all seeds, admitted before output reservation.
    pub total_transitions: u64,
    /// Complete reports in the declared seed order; incomplete studies publish no aggregate.
    pub replicates: Vec<BranchStudyReplicate>,
    /// Branch id -> metric -> paired terminal effect; the unperturbed control is omitted.
    pub endpoint_effects: BTreeMap<String, BTreeMap<String, BranchEffectSize>>,
    /// Human-readable endpoint results and their statistical limits.
    pub summary_markdown: String,
}

/// Typed error returned during branch study planning, execution, or verification.
#[derive(Debug, Error)]
pub enum BranchStudyError {
    /// Plan validation error.
    #[error("study plan validation failed: {0}")]
    InvalidPlan(String),
    /// Requested work or retained evidence exceeds the supported study envelope.
    #[error("study {resource} requests {requested}, exceeding the limit of {limit}")]
    CapacityExceeded {
        /// Resource whose limit was exceeded.
        resource: &'static str,
        /// Requested count or byte length.
        requested: u64,
        /// Maximum supported count or byte length.
        limit: u64,
    },
    /// Paired observations are incomplete, unaligned, or non-finite.
    #[error("invalid paired comparison: {0}")]
    InvalidComparison(String),
    /// An imported scientific cut does not match the declared study identity or supported state.
    #[error("saved checkpoint does not match the study plan: {0}")]
    CheckpointMismatch(String),
    /// Intervention parameter error.
    #[error("branch `{branch_id}` intervention is invalid: {reason}")]
    InvalidIntervention {
        /// Branch identifier.
        branch_id: String,
        /// Reason for failure.
        reason: String,
    },
    /// Missing control branch.
    #[error("control branch not found in plan")]
    MissingControlBranch,
    /// Pre-intervention digest mismatch upon restoration.
    #[error(
        "pre-intervention digest mismatch for branch `{branch_id}`: expected `{expected}`, found `{actual}`"
    )]
    PreInterventionDigestMismatch {
        /// Branch identifier.
        branch_id: String,
        /// Expected checkpoint source digest.
        expected: String,
        /// Actual restored world digest.
        actual: String,
    },
    /// Simulation step failure.
    #[error("simulation step failed at tick {tick}: {reason}")]
    SimulationStep {
        /// Simulation tick.
        tick: u64,
        /// Failure details.
        reason: String,
    },
    /// Checkpoint capture or restore failure.
    #[error("checkpoint operation failed: {0}")]
    Checkpoint(#[from] WorldCheckpointError),
    /// Brain registry setup failure.
    #[error("brain registry setup failed: {0}")]
    Registry(String),
    /// World state error.
    #[error("world state error: {0}")]
    WorldState(#[from] WorldStateError),
    /// Bundle assembly or verification failure.
    #[error("bundle creation or verification failed: {0}")]
    Bundle(String),
    /// File system I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// An existing output must be retained rather than replaced by another study.
    #[error("study output {path} is already occupied; choose a new or empty directory")]
    OutputAlreadyExists {
        /// Path whose existing contents were preserved.
        path: PathBuf,
    },
    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Study was cancelled by caller.
    #[error("study execution was cancelled")]
    Cancelled,
    /// A specific independent replicate failed; retained earlier results are not a complete study.
    #[error("replicate seed {seed} failed: {source}")]
    ReplicateFailed {
        /// Founding seed of the failed replicate.
        seed: u64,
        /// Original typed failure.
        #[source]
        source: Box<BranchStudyError>,
    },
}

fn study_output_error(path: &Path, error: std::io::Error) -> BranchStudyError {
    if error.kind() == std::io::ErrorKind::AlreadyExists {
        BranchStudyError::OutputAlreadyExists {
            path: path.to_path_buf(),
        }
    } else {
        BranchStudyError::Io(error)
    }
}

fn enforce_study_limit(
    resource: &'static str,
    requested: u64,
    limit: u64,
) -> Result<(), BranchStudyError> {
    if requested > limit {
        return Err(BranchStudyError::CapacityExceeded {
            resource,
            requested,
            limit,
        });
    }
    Ok(())
}

pub(crate) fn validate_study_config(config: &ScriptBotsConfig) -> Result<(), BranchStudyError> {
    // These auxiliary states are retained by host checkpoints, not raw science cuts.
    // Apply this restriction to both sources and resolved future ConfigPatch treatments.
    if config.selection_mode != EvolutionSelectionMode::Fitness {
        return Err(BranchStudyError::InvalidPlan(
            "checkpoint branches require Fitness selection; raw checkpoints omit novelty and curiosity selection history".to_owned(),
        ));
    }
    if config.communication_scramble {
        return Err(BranchStudyError::InvalidPlan(
            "checkpoint branches require communication_scramble=false; raw checkpoints omit the scrambling RNG state".to_owned(),
        ));
    }
    let (width, height) = config.food_dimensions()?;
    enforce_study_limit(
        "grid cells",
        u64::from(width) * u64::from(height),
        MAX_STUDY_GRID_CELLS,
    )?;
    enforce_study_limit(
        "history entries",
        config.history_capacity as u64,
        MAX_STUDY_HISTORY_ENTRIES,
    )?;
    enforce_study_limit(
        "population floor",
        config.population_minimum as u64,
        MAX_STUDY_POPULATION,
    )?;
    enforce_study_limit(
        "population injection count",
        u64::from(config.population_spawn_count),
        MAX_STUDY_POPULATION,
    )?;
    if config.archive_enabled {
        enforce_study_limit(
            "archive bytes",
            config.archive_max_bytes as u64,
            scriptbots_core::map_elites::DEFAULT_MAX_ARCHIVE_BYTES as u64,
        )?;
    }
    Ok(())
}

fn enforce_study_population(world: &WorldState) -> Result<(), BranchStudyError> {
    enforce_study_limit(
        "live population",
        world.agents().len() as u64,
        MAX_STUDY_POPULATION,
    )
}

fn read_study_checkpoint(path: &Path) -> Result<WorldCheckpointV1, BranchStudyError> {
    let maximum = scriptbots_core::MAX_WORLD_CHECKPOINT_BYTES as u64;
    if !fs::symlink_metadata(path)?.is_file() {
        return Err(BranchStudyError::InvalidPlan(format!(
            "source_checkpoint `{}` must be a regular file",
            path.display(),
        )));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        // Reject links and avoid blocking on a pipe substituted between metadata and open.
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(BranchStudyError::InvalidPlan(format!(
            "source_checkpoint `{}` must be a regular file",
            path.display(),
        )));
    }
    enforce_study_limit("source checkpoint bytes", metadata.len(), maximum)?;
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    enforce_study_limit("source checkpoint bytes", bytes.len() as u64, maximum)?;
    Ok(WorldCheckpointV1::decode(&bytes)?)
}

fn write_new_study_artifact(path: &Path, bytes: &[u8]) -> Result<(), BranchStudyError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| study_output_error(path, error))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn reserve_study_output(output: &Path, plan_json: &[u8]) -> Result<(), BranchStudyError> {
    match fs::symlink_metadata(output) {
        Ok(metadata) => {
            if !metadata.is_dir()
                || metadata.file_type().is_symlink()
                || fs::read_dir(output)?.next().transpose()?.is_some()
            {
                return Err(BranchStudyError::OutputAlreadyExists {
                    path: output.to_path_buf(),
                });
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(output)?;
        }
        Err(error) => return Err(error.into()),
    }
    // Both single-seed and replicated studies use the same create_new writer claim.
    // Concurrent invocations cannot each reserve a different kind of plan in one directory.
    write_new_study_artifact(&output.join("study_plan.json"), plan_json)
}

/// Create a fresh [`BrainRegistry`] matching the given preset for checkpoint restoration.
///
/// # Errors
///
/// Returns [`BranchStudyError::Registry`] if any family adapter cannot be registered.
pub fn create_brain_registry(preset: BrainPreset) -> Result<BrainRegistry, BranchStudyError> {
    crate::brains::create_brain_registry(preset)
        .map_err(|e| BranchStudyError::Registry(e.to_string()))
}

/// Compare two world digests and return the first differing lane in fixed diagnostic order.
#[must_use]
pub fn find_first_divergence_stage(
    control: &WorldDigestV1,
    branch: &WorldDigestV1,
) -> Option<(&'static str, String, String)> {
    if control.overall == branch.overall {
        return None;
    }
    if control.config != branch.config {
        return Some(("config", control.config.clone(), branch.config.clone()));
    }
    if control.effects != branch.effects {
        return Some(("effects", control.effects.clone(), branch.effects.clone()));
    }
    if control.food != branch.food {
        return Some(("food", control.food.clone(), branch.food.clone()));
    }
    if control.terrain != branch.terrain {
        return Some(("terrain", control.terrain.clone(), branch.terrain.clone()));
    }
    if control.hydrology != branch.hydrology {
        return Some((
            "hydrology",
            control
                .hydrology
                .clone()
                .unwrap_or_else(|| "absent".to_string()),
            branch
                .hydrology
                .clone()
                .unwrap_or_else(|| "absent".to_string()),
        ));
    }
    if control.agents != branch.agents {
        return Some(("agents", control.agents.clone(), branch.agents.clone()));
    }
    if control.brains != branch.brains {
        return Some(("brains", control.brains.clone(), branch.brains.clone()));
    }
    if control.brain_registry != branch.brain_registry {
        return Some((
            "brain_registry",
            control.brain_registry.clone(),
            branch.brain_registry.clone(),
        ));
    }
    if control.counters != branch.counters {
        return Some((
            "counters",
            control.counters.clone(),
            branch.counters.clone(),
        ));
    }
    if control.derived_transition != branch.derived_transition {
        return Some((
            "derived_transition",
            control.derived_transition.clone(),
            branch.derived_transition.clone(),
        ));
    }
    if control.rng != branch.rng {
        return Some((
            "rng",
            format!("{:?}", control.rng),
            format!("{:?}", branch.rng),
        ));
    }
    if control.origins != branch.origins {
        return Some(("origins", control.origins.clone(), branch.origins.clone()));
    }
    Some(("overall", control.overall.clone(), branch.overall.clone()))
}

/// Compute effect size and uncertainty across independent matched replicates.
///
/// Each paired entry must summarize a separate independent experimental replicate, for example
/// the final endpoint from one matched seed. Consecutive ticks of one trajectory are not replicates.
///
/// # Errors
///
/// Returns [`BranchStudyError::InvalidComparison`] for unequal lengths or non-finite observations
/// or reductions. Fewer than two pairs produce descriptive values with no inferential statistics.
pub fn compute_paired_effect(
    control: &[f64],
    branch: &[f64],
    metric_name: &str,
    seed: u64,
) -> Result<BranchEffectSize, BranchStudyError> {
    let (mut effect, diffs) = describe_paired_effect(
        control,
        branch,
        metric_name,
        BranchAnalysisBasis::IndependentPairedReplicates {
            pairs: control.len(),
        },
    )?;
    if effect.n_samples < 2 {
        return Ok(effect);
    }

    if effect.sd_diff > 1e-12 {
        let d = effect.mean_diff / effect.sd_diff;
        #[allow(clippy::cast_precision_loss)]
        let df = (effect.n_samples - 1) as f64;
        let j = 1.0 - 3.0 / (4.0 * df - 1.0).max(1.0);
        effect.cohens_d = Some(d);
        effect.hedges_g = Some(d * j);
    }
    effect.ci_95 = Some(bootstrap_ci(&diffs, 1_000, seed).map_err(|error| {
        BranchStudyError::InvalidComparison(format!("{metric_name} bootstrap failed: {error}"))
    })?);
    effect.p_value = Some(monte_carlo_sign_flip_p_value(&diffs, 1_000, seed));
    Ok(effect)
}

fn describe_paired_effect(
    control: &[f64],
    branch: &[f64],
    metric_name: &str,
    analysis_basis: BranchAnalysisBasis,
) -> Result<(BranchEffectSize, Vec<f64>), BranchStudyError> {
    if control.len() != branch.len() {
        return Err(BranchStudyError::InvalidComparison(format!(
            "metric `{metric_name}` has {} control observations and {} branch observations",
            control.len(),
            branch.len()
        )));
    }
    let diffs = control
        .iter()
        .zip(branch)
        .enumerate()
        .map(|(index, (&control, &branch))| {
            let difference = branch - control;
            if !control.is_finite() || !branch.is_finite() || !difference.is_finite() {
                return Err(BranchStudyError::InvalidComparison(format!(
                    "metric `{metric_name}` has a non-finite pair at index {index}"
                )));
            }
            Ok(difference)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let n_samples = diffs.len();
    let mean_diff = if n_samples == 0 {
        0.0
    } else {
        diffs.iter().sum::<f64>() / n_samples as f64
    };
    let sd_diff = if n_samples < 2 {
        0.0
    } else {
        (diffs
            .iter()
            .map(|&diff| (diff - mean_diff).powi(2))
            .sum::<f64>()
            / (n_samples - 1) as f64)
            .sqrt()
    };
    if !mean_diff.is_finite() || !sd_diff.is_finite() {
        return Err(BranchStudyError::InvalidComparison(format!(
            "metric `{metric_name}` produced a non-finite paired reduction"
        )));
    }
    Ok((
        BranchEffectSize {
            metric: metric_name.to_string(),
            analysis_basis,
            n_samples,
            mean_diff,
            sd_diff,
            cohens_d: None,
            hedges_g: None,
            ci_95: None,
            p_value: None,
        },
        diffs,
    ))
}

/// Compute a two-sided Monte Carlo sign-flip permutation p-value for the paired difference mean.
#[must_use]
pub fn monte_carlo_sign_flip_p_value(diffs: &[f64], iters: usize, seed: u64) -> f64 {
    if diffs.is_empty() || iters == 0 {
        return 1.0;
    }
    let observed_mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    if observed_mean.abs() < 1e-14 {
        return 1.0;
    }

    let mut count_extreme = 0usize;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

    for _ in 0..iters {
        let mut sim_sum = 0.0;
        for &d in diffs {
            let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            sim_sum += sign * d;
        }
        let sim_mean = sim_sum / diffs.len() as f64;
        if sim_mean.abs() >= observed_mean.abs() - 1e-14 {
            count_extreme += 1;
        }
    }

    (count_extreme as f64 + 1.0) / (iters as f64 + 1.0)
}

/// Orchestrator for executing paired checkpoint-branch studies.
pub struct BranchStudyOrchestrator {
    /// Validated plan.
    pub plan: BranchStudyPlan,
    /// Destination directory for study outputs, bundles, and reports.
    pub output_dir: PathBuf,
}

impl BranchStudyOrchestrator {
    /// Construct a new orchestrator.
    #[must_use]
    pub fn new(plan: BranchStudyPlan, output_dir: PathBuf) -> Self {
        Self { plan, output_dir }
    }

    /// Validate the study plan before execution without reading a declared source file.
    ///
    /// # Errors
    ///
    /// Returns a typed [`BranchStudyError`] for malformed plans, unsupported brains,
    /// invalid configurations/interventions, or exceeded resource limits.
    pub fn validate(&self) -> Result<(), BranchStudyError> {
        if self.plan.source_checkpoint.is_some() {
            self.validate_with_config(&self.plan.config, self.plan.total_transitions()?, true)
        } else {
            self.validate_with_config(
                &self.plan.effective_config(),
                self.plan.total_transitions()?,
                false,
            )
        }
    }

    fn validate_with_config(
        &self,
        config: &ScriptBotsConfig,
        total_transitions: u64,
        saved_source: bool,
    ) -> Result<(), BranchStudyError> {
        if self.plan.study_id.trim().is_empty() {
            return Err(BranchStudyError::InvalidPlan(
                "study_id cannot be empty".to_string(),
            ));
        }
        if self.plan.checkpoint_tick == 0 && !saved_source {
            return Err(BranchStudyError::InvalidPlan(
                "seeded warmup checkpoint_tick must be >= 1".to_string(),
            ));
        }
        if self.plan.horizon_ticks == 0 {
            return Err(BranchStudyError::InvalidPlan(
                "horizon_ticks must be >= 1".to_string(),
            ));
        }
        let final_tick = self.plan.final_tick()?;
        self.plan.sample_count()?;
        enforce_study_limit(
            "branch count",
            self.plan.branches.len() as u64,
            MAX_STUDY_BRANCHES as u64,
        )?;
        enforce_study_limit(
            "horizon ticks",
            self.plan.horizon_ticks,
            MAX_STUDY_HORIZON_TICKS,
        )?;
        enforce_study_limit(
            "total transitions",
            total_transitions,
            MAX_STUDY_TOTAL_TRANSITIONS,
        )?;
        enforce_study_limit(
            "study identifier bytes",
            self.plan.study_id.len() as u64,
            MAX_STUDY_IDENTIFIER_BYTES as u64,
        )?;
        enforce_study_limit(
            "study description bytes",
            self.plan.description.len() as u64,
            MAX_STUDY_DESCRIPTION_BYTES as u64,
        )?;
        validate_study_config(config)?;
        crate::brains::create_brain_registry_for_config(self.plan.brain_preset, config)
            .map_err(|error| BranchStudyError::Registry(error.to_string()))?;
        if !self.plan.branches.iter().any(|b| b.intervention.is_none()) {
            return Err(BranchStudyError::MissingControlBranch);
        }

        let mut branch_ids = std::collections::BTreeSet::new();
        for branch in &self.plan.branches {
            enforce_study_limit(
                "branch identifier bytes",
                branch.branch_id.len() as u64,
                MAX_STUDY_IDENTIFIER_BYTES as u64,
            )?;
            enforce_study_limit(
                "branch description bytes",
                branch.description.len() as u64,
                MAX_STUDY_DESCRIPTION_BYTES as u64,
            )?;
            if branch.branch_id.trim().is_empty() {
                return Err(BranchStudyError::InvalidPlan(
                    "branch_id cannot be empty".to_string(),
                ));
            }
            // The id names an output directory (`bundle_<branch_id>`), so it
            // must not be able to escape `output_dir`.
            if !branch
                .branch_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
            {
                return Err(BranchStudyError::InvalidPlan(format!(
                    "branch_id `{}` may contain only ASCII letters, digits, '_' and '-'",
                    branch.branch_id
                )));
            }
            if !branch_ids.insert(&branch.branch_id) {
                return Err(BranchStudyError::InvalidPlan(format!(
                    "duplicate branch_id `{}` in plan",
                    branch.branch_id
                )));
            }
            if let Some(intervention) = &branch.intervention {
                let due_tick = branch.scheduled_tick.unwrap_or(self.plan.checkpoint_tick);
                if due_tick < self.plan.checkpoint_tick || due_tick >= final_tick {
                    return Err(BranchStudyError::InvalidPlan(format!(
                        "branch `{}` scheduled boundary {due_tick} must be in [{}, {final_tick}) so its next transition is observed",
                        branch.branch_id, self.plan.checkpoint_tick
                    )));
                }
                intervention
                    .validate(self.plan.config.world_width, self.plan.config.world_height)?;
                if let StudyIntervention::ConfigPatch(patch) = intervention {
                    let patched =
                        resolve_scheduled_config_patch(config, patch).map_err(|error| {
                            BranchStudyError::InvalidIntervention {
                                branch_id: branch.branch_id.clone(),
                                reason: error.to_string(),
                            }
                        })?;
                    patched
                        .validate()
                        .map_err(|error| BranchStudyError::InvalidIntervention {
                            branch_id: branch.branch_id.clone(),
                            reason: error.to_string(),
                        })?;
                    validate_study_config(&patched)?;
                    if (
                        patched.world_width,
                        patched.world_height,
                        patched.food_cell_size,
                    ) != (
                        config.world_width,
                        config.world_height,
                        config.food_cell_size,
                    ) || patched.persistence_interval != 0
                    {
                        return Err(BranchStudyError::InvalidIntervention {
                            branch_id: branch.branch_id.clone(),
                            reason: "checkpoint branches require unchanged world geometry and disabled persistence".to_owned(),
                        });
                    }
                }
            } else if branch.scheduled_tick.is_some() {
                return Err(BranchStudyError::InvalidPlan(format!(
                    "control branch `{}` cannot schedule an absent intervention",
                    branch.branch_id
                )));
            }
        }

        Ok(())
    }

    /// Execute the complete paired study from the declared saved cut or a seeded warmup.
    ///
    /// A declared source is read as a bounded regular file and decoded strictly before
    /// reserving output. Relative paths use the process working directory.
    ///
    /// # Errors
    ///
    /// Returns [`BranchStudyError`] if execution fails, is cancelled, or fails verification.
    pub fn execute_study(
        &self,
        cancellation: Option<&AtomicBool>,
    ) -> Result<BranchStudyReport, BranchStudyError> {
        self.validate()?;
        if cancellation.is_some_and(|cancel| cancel.load(Ordering::Relaxed)) {
            return Err(BranchStudyError::Cancelled);
        }
        if let Some(path) = &self.plan.source_checkpoint {
            let checkpoint = read_study_checkpoint(path)?;
            return self.execute_saved_checkpoint(&checkpoint, cancellation);
        }
        self.reserve_output_directory()?;

        // 1. Pre-intervention baseline execution to checkpoint_tick.
        let mut world = WorldState::new(self.plan.effective_config())?;
        let installed = install_brains(&mut world, self.plan.brain_preset)
            .map_err(|e| BranchStudyError::Registry(e.to_string()))?;
        seed_founding_population(&mut world, installed.population())
            .map_err(|e| BranchStudyError::Registry(e.to_string()))?;

        for tick in 1..=self.plan.checkpoint_tick {
            if let Some(cancel) = cancellation
                && cancel.load(Ordering::Relaxed)
            {
                return Err(BranchStudyError::Cancelled);
            }
            world.step().map_err(|e| BranchStudyError::SimulationStep {
                tick,
                reason: e.to_string(),
            })?;
            enforce_study_population(&world)?;
        }

        let checkpoint = world.checkpoint_v1()?;
        self.execute_checkpoint_branches(&checkpoint, BranchStudySource::SeededWarmup, cancellation)
    }

    /// Run new isolated scientific futures from an already decoded saved world.
    ///
    /// The plan must name the exact saved configuration, completed tick, root RNG seed and
    /// executable brain registry, and must not also declare a source path. No population is
    /// seeded and no warmup is performed. The canonical source envelope and its checksum
    /// are retained once at the study root. The source host's session, schedules, storage
    /// ownership and retained analytics are not continued.
    ///
    /// Sources and future configurations require Fitness selection and disabled communication
    /// scrambling because raw checkpoints omit those auxiliary histories. Imported maps are
    /// refused because raw checkpoints omit map temperature. Configuration-dependent brain
    /// factories must also match their original constructor identity: a raw cut cannot infer
    /// an earlier configuration from its hash if live settings changed that identity.
    ///
    /// # Errors
    ///
    /// Refuses invalid, unsupported or mismatched checkpoints, exceeded study bounds,
    /// cancellation and occupied output before running a branch. Partial output is preserved.
    pub fn execute_from_checkpoint(
        &self,
        checkpoint: &WorldCheckpointV1,
        cancellation: Option<&AtomicBool>,
    ) -> Result<BranchStudyReport, BranchStudyError> {
        if self.plan.source_checkpoint.is_some() {
            return Err(BranchStudyError::InvalidPlan(
                "execute_from_checkpoint cannot also declare source_checkpoint; use execute_study for the declared file".to_owned(),
            ));
        }
        self.execute_saved_checkpoint(checkpoint, cancellation)
    }

    fn execute_saved_checkpoint(
        &self,
        checkpoint: &WorldCheckpointV1,
        cancellation: Option<&AtomicBool>,
    ) -> Result<BranchStudyReport, BranchStudyError> {
        if cancellation.is_some_and(|cancel| cancel.load(Ordering::Relaxed)) {
            return Err(BranchStudyError::Cancelled);
        }
        let source_bytes = self.validate_from_checkpoint(checkpoint)?;
        let source = BranchStudySource::SavedCheckpoint {
            artifact_path: SOURCE_CHECKPOINT_ARTIFACT.to_owned(),
            byte_size: source_bytes.len() as u64,
            checksum_blake3: blake3::hash(&source_bytes).to_hex().to_string(),
        };
        if cancellation.is_some_and(|cancel| cancel.load(Ordering::Relaxed)) {
            return Err(BranchStudyError::Cancelled);
        }
        self.reserve_output_directory()?;
        write_new_study_artifact(
            &self.output_dir.join(SOURCE_CHECKPOINT_ARTIFACT),
            &source_bytes,
        )?;
        drop(source_bytes);
        self.execute_checkpoint_branches(checkpoint, source, cancellation)
    }

    /// Validate a decoded source before managed execution is admitted or output reserved.
    /// Returns the validated canonical source bytes, retaining no reconstructed world.
    pub(crate) fn validate_from_checkpoint(
        &self,
        checkpoint: &WorldCheckpointV1,
    ) -> Result<Vec<u8>, BranchStudyError> {
        self.validate_with_config(checkpoint.config(), self.plan.branch_transitions()?, true)?;
        // Compare canonical typed encodings, preserving distinctions such as signed zero
        // that PartialEq would treat as equal. No seed or persistence field is normalized.
        if serde_json::to_vec(&self.plan.config)? != serde_json::to_vec(checkpoint.config())? {
            return Err(BranchStudyError::CheckpointMismatch(
                "config must exactly equal the saved scientific configuration".to_owned(),
            ));
        }
        if self.plan.checkpoint_tick != checkpoint.tick().0 {
            return Err(BranchStudyError::CheckpointMismatch(format!(
                "checkpoint_tick is {}, but the saved boundary is {}",
                self.plan.checkpoint_tick,
                checkpoint.tick().0,
            )));
        }
        enforce_study_limit(
            "live population",
            checkpoint.agent_count() as u64,
            MAX_STUDY_POPULATION,
        )?;
        // Encoding validates the complete strict envelope. Canonical decode/encode retains
        // every source byte. Restore checks family codecs, supported state and the complete
        // science digest before output is claimed; the preflight owner never takes a step.
        let source_bytes = checkpoint.encode()?;
        drop(self.restore_study_checkpoint(checkpoint)?);
        Ok(source_bytes)
    }

    fn restore_study_checkpoint(
        &self,
        checkpoint: &WorldCheckpointV1,
    ) -> Result<WorldState, BranchStudyError> {
        let registry = crate::brains::create_brain_registry_for_config(
            self.plan.brain_preset,
            checkpoint.config(),
        )
        .map_err(|error| BranchStudyError::Registry(error.to_string()))?;
        let world = WorldState::restore_checkpoint_v1(checkpoint, registry)?;
        enforce_study_population(&world)?;
        if world.map_metadata().is_some() {
            return Err(BranchStudyError::CheckpointMismatch(
                "raw checkpoints with imported map metadata are unsupported because map temperature is not retained".to_owned(),
            ));
        }
        let root_seed = world.agent_substream_protocol_v1().root_seed();
        if self.plan.base_seed != root_seed {
            return Err(BranchStudyError::CheckpointMismatch(format!(
                "base_seed is {}, but the saved RNG root seed is {root_seed}",
                self.plan.base_seed,
            )));
        }
        let restored_digest = world.world_digest_v1().map_err(|error| {
            BranchStudyError::CheckpointMismatch(format!(
                "cannot inspect restored scientific digest: {error}"
            ))
        })?;
        if &restored_digest != checkpoint.source_digest() {
            return Err(BranchStudyError::CheckpointMismatch(
                "restored scientific digest differs from the saved source".to_owned(),
            ));
        }
        Ok(world)
    }

    fn execute_checkpoint_branches(
        &self,
        checkpoint: &WorldCheckpointV1,
        source: BranchStudySource,
        cancellation: Option<&AtomicBool>,
    ) -> Result<BranchStudyReport, BranchStudyError> {
        let pre_intervention_digest = checkpoint.source_digest().overall.clone();
        let pre_intervention_tick = checkpoint.tick().0;
        if pre_intervention_tick != self.plan.checkpoint_tick {
            return Err(BranchStudyError::SimulationStep {
                tick: pre_intervention_tick,
                reason: format!(
                    "captured checkpoint tick {} does not match requested {}",
                    pre_intervention_tick, self.plan.checkpoint_tick
                ),
            });
        }
        write_new_study_artifact(
            &self.output_dir.join("study_source.json"),
            &serde_json::to_vec_pretty(&source)?,
        )?;

        // 2. Identify Control branch and run it first to establish baseline trajectory.
        let control_spec = self
            .plan
            .branches
            .iter()
            .find(|b| b.intervention.is_none())
            .ok_or(BranchStudyError::MissingControlBranch)?;
        let control_branch_id = control_spec.branch_id.clone();

        let (control_samples, control_report) = self.run_branch(
            control_spec,
            checkpoint,
            &pre_intervention_digest,
            &source,
            None,
            cancellation,
        )?;

        // 3. Run all other branches and compare against control trajectory.
        let mut branch_reports = Vec::with_capacity(self.plan.branches.len());
        for branch_spec in &self.plan.branches {
            if branch_spec.branch_id == control_branch_id {
                branch_reports.push(control_report.clone());
            } else {
                let (_, report) = self.run_branch(
                    branch_spec,
                    checkpoint,
                    &pre_intervention_digest,
                    &source,
                    Some(&control_samples),
                    cancellation,
                )?;
                branch_reports.push(report);
            }
        }

        // 4. Generate summary Markdown.
        let summary_markdown = self.generate_summary_markdown(
            &pre_intervention_digest,
            &source,
            &control_branch_id,
            &branch_reports,
        );

        let report = BranchStudyReport {
            study_id: self.plan.study_id.clone(),
            schema_version: BRANCH_STUDY_REPORT_SCHEMA_VERSION,
            base_seed: self.plan.base_seed,
            source,
            checkpoint_tick: self.plan.checkpoint_tick,
            horizon_ticks: self.plan.horizon_ticks,
            pre_intervention_digest,
            pre_intervention_digest_verified_for_all_branches: true,
            control_branch_id,
            branch_reports,
            summary_markdown,
        };

        // 5. Publish the final report only after all branch bundles and the summary.
        // The plan was reserved before execution, so failed or interrupted studies
        // retain their inputs and cannot be silently replaced by a later invocation.
        let report_json = serde_json::to_string_pretty(&report)?;
        write_new_study_artifact(
            &self.output_dir.join("study_summary.md"),
            report.summary_markdown.as_bytes(),
        )?;
        write_new_study_artifact(
            &self.output_dir.join("study_report.json"),
            report_json.as_bytes(),
        )?;

        Ok(report)
    }

    fn reserve_output_directory(&self) -> Result<(), BranchStudyError> {
        let plan_json = serde_json::to_vec_pretty(&self.plan)?;
        reserve_study_output(&self.output_dir, &plan_json)
    }

    fn validate_control_observations(
        &self,
        observations: &[BranchObservation],
        expected_pre_digest: &str,
    ) -> Result<(), BranchStudyError> {
        let expected_count = self.plan.sample_count()?;
        if observations.len() != expected_count {
            return Err(BranchStudyError::InvalidComparison(format!(
                "control trajectory contains {} observations; expected {expected_count}",
                observations.len()
            )));
        }
        for (offset, observation) in observations.iter().enumerate() {
            let expected_tick = self.plan.checkpoint_tick + offset as u64;
            let digest = &observation.digest;
            if observation.sample.tick != expected_tick
                || digest.tick.0 != expected_tick
                || observation.sample.digest_overall != digest.overall
            {
                return Err(BranchStudyError::InvalidComparison(format!(
                    "control observation {offset} does not describe the exact boundary {expected_tick}"
                )));
            }
            digest.validate_contract().map_err(|error| {
                BranchStudyError::InvalidComparison(format!(
                    "control digest at tick {expected_tick} is invalid: {error}"
                ))
            })?;
            if !digest.evaluator_state_covered || !digest.factory_state_covered {
                return Err(BranchStudyError::InvalidComparison(format!(
                    "control digest at tick {expected_tick} does not cover the complete brain state"
                )));
            }
        }
        if observations[0].digest.overall != expected_pre_digest {
            return Err(BranchStudyError::InvalidComparison(
                "control trajectory does not begin at the shared checkpoint digest".to_string(),
            ));
        }
        Ok(())
    }

    fn run_branch(
        &self,
        branch: &StudyBranchSpec,
        checkpoint: &WorldCheckpointV1,
        expected_pre_digest: &str,
        source: &BranchStudySource,
        control_samples: Option<&[BranchObservation]>,
        cancellation: Option<&AtomicBool>,
    ) -> Result<(Vec<BranchObservation>, BranchOutcomeReport), BranchStudyError> {
        self.plan.final_tick()?;
        if let Some(control) = control_samples {
            self.validate_control_observations(control, expected_pre_digest)?;
        }
        // Restore from shared checkpoint.
        let mut branch_world = self.restore_study_checkpoint(checkpoint)?;

        let restored_digest =
            branch_world
                .world_digest_v1()
                .map_err(|e| BranchStudyError::SimulationStep {
                    tick: self.plan.checkpoint_tick,
                    reason: format!("failed to read restored digest: {e}"),
                })?;
        if restored_digest.overall != expected_pre_digest {
            return Err(BranchStudyError::PreInterventionDigestMismatch {
                branch_id: branch.branch_id.clone(),
                expected: expected_pre_digest.to_string(),
                actual: restored_digest.overall,
            });
        }

        let mut samples = Vec::new();
        samples
            .try_reserve_exact(self.plan.sample_count()?)
            .map_err(|error| {
                BranchStudyError::InvalidPlan(format!(
                    "cannot retain the study trajectory: {error}"
                ))
            })?;

        // Record sample at checkpoint_tick.
        let initial_food = branch_world
            .food()
            .cells()
            .iter()
            .map(|&c| f64::from(c))
            .sum::<f64>();
        let initial_summary = DynamicWorldSnapshot::from_world(&branch_world)
            .map_err(|error| BranchStudyError::SimulationStep {
                tick: self.plan.checkpoint_tick,
                reason: format!("failed to observe checkpoint state: {error}"),
            })?
            .summary;
        samples.push(BranchObservation {
            sample: BranchTickSample {
                tick: self.plan.checkpoint_tick,
                population: initial_summary.agent_count,
                food_total: initial_food,
                births: 0,
                deaths: 0,
                total_energy: initial_summary.total_energy,
                average_health: initial_summary.average_health,
                spike_hits: 0,
                digest_overall: restored_digest.overall.clone(),
            },
            digest: restored_digest,
        });

        let due_tick = branch.scheduled_tick.unwrap_or(self.plan.checkpoint_tick);
        let mut command_applied_tick = None;

        // Step through the horizon.
        let mut first_divergence: Option<FirstDivergence> = None;
        for step in 1..=self.plan.horizon_ticks {
            if let Some(cancel) = cancellation
                && cancel.load(Ordering::Relaxed)
            {
                return Err(BranchStudyError::Cancelled);
            }
            let boundary_tick = branch_world.tick().0;
            if boundary_tick == due_tick
                && let Some(intervention) = &branch.intervention
            {
                self.apply_intervention(&mut branch_world, intervention)?;
                // Record the boundary observed from the world, only after successful application.
                command_applied_tick = Some(boundary_tick);
            }
            let current_tick = self.plan.checkpoint_tick + step;
            let outcome =
                branch_world
                    .step_outcome()
                    .map_err(|e| BranchStudyError::SimulationStep {
                        tick: current_tick,
                        reason: e.to_string(),
                    })?;
            enforce_study_population(&branch_world)?;

            let digest =
                branch_world
                    .world_digest_v1()
                    .map_err(|e| BranchStudyError::SimulationStep {
                        tick: current_tick,
                        reason: format!("digest failure: {e}"),
                    })?;

            let food_total = branch_world
                .food()
                .cells()
                .iter()
                .map(|&c| f64::from(c))
                .sum::<f64>();
            let summary = &outcome.outcome.summary;
            let pop = summary.agent_count;

            // Check first divergence against control if control samples are available.
            if let Some(ctrl_list) = control_samples
                && first_divergence.is_none()
                && let Some((stage, ctrl_lane, br_lane)) =
                    find_first_divergence_stage(&ctrl_list[step as usize].digest, &digest)
            {
                let applied_tick = command_applied_tick.ok_or_else(|| {
                    BranchStudyError::InvalidComparison(format!(
                        "branch `{}` diverged from control at tick {current_tick} before any intervention was applied",
                        branch.branch_id
                    ))
                })?;
                first_divergence = Some(FirstDivergence {
                    branch_id: branch.branch_id.clone(),
                    tick: current_tick,
                    stage: stage.to_string(),
                    control_digest: ctrl_lane,
                    branch_digest: br_lane,
                    command_applied_tick: applied_tick,
                    details: format!(
                        "First observed divergence in digest lane `{stage}` at tick {current_tick}, comparing control and branch at the same completed boundary"
                    ),
                });
            }

            samples.push(BranchObservation {
                sample: BranchTickSample {
                    tick: current_tick,
                    population: pop,
                    food_total,
                    births: summary.births,
                    deaths: summary.deaths,
                    total_energy: summary.total_energy,
                    average_health: summary.average_health,
                    spike_hits: summary.spike_hits,
                    digest_overall: digest.overall.clone(),
                },
                digest,
            });
        }

        if branch.intervention.is_some() && command_applied_tick.is_none() {
            return Err(BranchStudyError::InvalidPlan(format!(
                "branch `{}` intervention was never applied within the observed horizon",
                branch.branch_id
            )));
        }

        let terminal_sample = samples
            .last()
            .ok_or_else(|| {
                BranchStudyError::InvalidComparison("branch has no terminal observation".to_owned())
            })?
            .sample
            .clone();
        let final_pop = terminal_sample.population;
        let final_digest = terminal_sample.digest_overall.clone();

        // Assess population outcomes from the actual pre-command boundary, not from an earlier
        // checkpoint when the intervention was delayed. Control branches use the checkpoint.
        let baseline_tick = command_applied_tick.unwrap_or(self.plan.checkpoint_tick);
        let baseline_index = (baseline_tick - self.plan.checkpoint_tick) as usize;
        let outcome_samples = &samples[baseline_index..];
        let pre_intervention_pop = outcome_samples[0].sample.population;
        let mut trough_population = pre_intervention_pop;
        let mut trough_tick = baseline_tick;
        let mut extinction_tick = None;
        for observation in outcome_samples {
            let sample = &observation.sample;
            if sample.population < trough_population {
                trough_population = sample.population;
                trough_tick = sample.tick;
            }
            if sample.population == 0 && extinction_tick.is_none() {
                extinction_tick = Some(sample.tick);
            }
        }

        // Extinction assessment.
        let extinction = ExtinctionMeasure {
            extinct: trough_population == 0,
            extinction_tick,
            min_population: trough_population,
            final_population: final_pop,
        };

        // Recovery assessment.
        #[allow(clippy::cast_precision_loss)]
        let target_recovery_pop = (0.8 * pre_intervention_pop as f64).round() as usize;
        let recovered = final_pop >= target_recovery_pop.max(1);
        #[allow(clippy::cast_precision_loss)]
        let recovery_ratio = final_pop as f64 / (pre_intervention_pop.max(1) as f64);

        let mut recovery_ticks = None;
        if recovered {
            for observation in outcome_samples {
                let sample = &observation.sample;
                if sample.tick > trough_tick && sample.population >= target_recovery_pop {
                    recovery_ticks = Some(sample.tick - trough_tick);
                    break;
                }
            }
        }
        let recovery = RecoveryMeasure {
            baseline_tick,
            pre_intervention_population: pre_intervention_pop,
            trough_population,
            trough_tick,
            final_population: final_pop,
            recovered,
            recovery_ratio,
            recovery_ticks,
        };

        // Hysteresis & Effect size assessments.
        let (hysteresis, effect_sizes) = if let Some(ctrl_list) = control_samples {
            let mut integrated_deficit = 0.0;
            let mut sum_abs_diff = 0.0;
            let mut sum_ctrl_pop = 0.0;
            let mut max_pop_gap = 0.0;

            let ctrl_pops: Vec<f64> = ctrl_list
                .iter()
                .map(|s| s.sample.population as f64)
                .collect();
            let br_pops: Vec<f64> = samples.iter().map(|s| s.sample.population as f64).collect();

            let ctrl_foods: Vec<f64> = ctrl_list.iter().map(|s| s.sample.food_total).collect();
            let br_foods: Vec<f64> = samples.iter().map(|s| s.sample.food_total).collect();

            let ctrl_energies: Vec<f64> = ctrl_list
                .iter()
                .map(|s| f64::from(s.sample.total_energy))
                .collect();
            let br_energies: Vec<f64> = samples
                .iter()
                .map(|s| f64::from(s.sample.total_energy))
                .collect();

            for (c, b) in ctrl_pops.iter().zip(&br_pops) {
                let diff = c - b;
                integrated_deficit += diff;
                sum_abs_diff += diff.abs();
                sum_ctrl_pop += c;
                if diff > max_pop_gap {
                    max_pop_gap = diff;
                }
            }

            let ctrl_final = ctrl_pops.last().copied().unwrap_or(1.0);
            let br_final = br_pops.last().copied().unwrap_or(0.0);
            let terminal_gap = (ctrl_final - br_final) / ctrl_final.max(1.0);
            let normalized_diff = sum_abs_diff / sum_ctrl_pop.max(1.0);

            let hysteresis_index = if max_pop_gap > 1e-6 {
                (ctrl_final - br_final).max(0.0) / max_pop_gap
            } else {
                0.0
            };

            let path_dependence_detected = terminal_gap.abs() > 0.05;

            let hyst = HysteresisMeasure {
                integrated_trajectory_deficit: integrated_deficit,
                normalized_trajectory_difference: normalized_diff,
                terminal_trajectory_gap: terminal_gap,
                hysteresis_index,
                path_dependence_detected,
            };

            let mut effects = BTreeMap::new();
            let analysis_basis = BranchAnalysisBasis::SingleSeedTrajectory {
                seed: self.plan.base_seed,
            };
            effects.insert(
                "population".to_string(),
                describe_paired_effect(&ctrl_pops, &br_pops, "population", analysis_basis)?.0,
            );
            effects.insert(
                "food_total".to_string(),
                describe_paired_effect(&ctrl_foods, &br_foods, "food_total", analysis_basis)?.0,
            );
            effects.insert(
                "total_energy".to_string(),
                describe_paired_effect(
                    &ctrl_energies,
                    &br_energies,
                    "total_energy",
                    analysis_basis,
                )?
                .0,
            );

            (hyst, effects)
        } else {
            // Control branch against itself: 0 deficit, no effect
            (
                HysteresisMeasure {
                    integrated_trajectory_deficit: 0.0,
                    normalized_trajectory_difference: 0.0,
                    terminal_trajectory_gap: 0.0,
                    hysteresis_index: 0.0,
                    path_dependence_detected: false,
                },
                BTreeMap::new(),
            )
        };

        // Create and verify run bundle for branch.
        let (bundle_path, bundle_verified) = self.create_and_verify_bundle(
            branch,
            source,
            &samples,
            command_applied_tick,
            &first_divergence,
            &extinction,
            &recovery,
            &hysteresis,
            &effect_sizes,
        )?;

        let report = BranchOutcomeReport {
            branch_id: branch.branch_id.clone(),
            description: branch.description.clone(),
            command_applied_tick,
            first_divergence,
            extinction,
            recovery,
            hysteresis,
            effect_sizes,
            final_digest,
            terminal_sample,
            bundle_path: Some(bundle_path.to_string_lossy().into_owned()),
            bundle_verified,
        };

        Ok((samples, report))
    }

    fn apply_intervention(
        &self,
        world: &mut WorldState,
        intervention: &StudyIntervention,
    ) -> Result<(), BranchStudyError> {
        match intervention {
            StudyIntervention::Drought {
                region,
                duration_ticks,
                growth_scale,
            } => {
                world
                    .enqueue_intervention(Intervention::Drought {
                        region: region.unwrap_or(Region::All),
                        ticks: *duration_ticks,
                        growth_scale: *growth_scale,
                    })
                    .map_err(BranchStudyError::WorldState)?;
            }
            StudyIntervention::ResourceShock {
                region,
                bloom_amount,
                scorch,
            } => {
                let reg = region.unwrap_or(Region::All);
                if *bloom_amount > 0.0 {
                    world
                        .enqueue_intervention(Intervention::Bloom {
                            region: reg,
                            amount: *bloom_amount,
                        })
                        .map_err(BranchStudyError::WorldState)?;
                }
                if *scorch > 0.0 {
                    world
                        .enqueue_intervention(Intervention::Meteor {
                            region: reg,
                            lethality: 0.0,
                            scorch: *scorch,
                        })
                        .map_err(BranchStudyError::WorldState)?;
                }
            }
            StudyIntervention::TemperatureShift {
                discomfort_rate,
                comfort_band,
            } => {
                let mut config = world.config().clone();
                config.temperature_discomfort_rate = *discomfort_rate;
                config.temperature_comfort_band = *comfort_band;
                apply_control_command(world, ControlCommand::UpdateConfig(Box::new(config)))
                    .map_err(BranchStudyError::WorldState)?;
            }
            StudyIntervention::ClosedWorld { closed } => {
                world
                    .enqueue_intervention(Intervention::SetClosedWorld { closed: *closed })
                    .map_err(BranchStudyError::WorldState)?;
            }
            StudyIntervention::Cooperation(params) => {
                let mut config = world.config().clone();
                config.food_sharing_rate = params.food_sharing_rate;
                config.food_transfer_rate = params.food_transfer_rate;
                config.food_sharing_distance = params.food_sharing_distance;
                if let Some(partner) = params.reproduction_partner_chance {
                    config.reproduction_partner_chance = partner;
                }
                apply_control_command(world, ControlCommand::UpdateConfig(Box::new(config)))
                    .map_err(BranchStudyError::WorldState)?;
            }
            StudyIntervention::Core(it) => {
                world
                    .enqueue_intervention(it.clone())
                    .map_err(BranchStudyError::WorldState)?;
            }
            StudyIntervention::ConfigPatch(patch) => {
                let new_config = resolve_scheduled_config_patch(world.config(), patch)
                    .map_err(BranchStudyError::InvalidPlan)?;
                apply_control_command(world, ControlCommand::UpdateConfig(Box::new(new_config)))
                    .map_err(BranchStudyError::WorldState)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn create_and_verify_bundle(
        &self,
        branch: &StudyBranchSpec,
        source: &BranchStudySource,
        samples: &[BranchObservation],
        command_applied_tick: Option<u64>,
        first_divergence: &Option<FirstDivergence>,
        extinction: &ExtinctionMeasure,
        recovery: &RecoveryMeasure,
        hysteresis: &HysteresisMeasure,
        effect_sizes: &BTreeMap<String, BranchEffectSize>,
    ) -> Result<(PathBuf, bool), BranchStudyError> {
        // 1. Summary CSV
        let mut csv_buf = Vec::new();
        writeln!(
            &mut csv_buf,
            "tick,population,food_total,births,deaths,total_energy,average_health,spike_hits,digest"
        )?;
        for observation in samples {
            let s = &observation.sample;
            writeln!(
                &mut csv_buf,
                "{},{},{:.4},{},{},{:.4},{:.4},{},{}",
                s.tick,
                s.population,
                s.food_total,
                s.births,
                s.deaths,
                s.total_energy,
                s.average_health,
                s.spike_hits,
                s.digest_overall
            )?;
        }

        // 2. Branch evidence JSON
        let terminal_sample = &samples
            .last()
            .ok_or_else(|| {
                BranchStudyError::InvalidComparison("branch has no terminal observation".to_owned())
            })?
            .sample;
        let evidence_json = serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": BRANCH_STUDY_REPORT_SCHEMA_VERSION,
            "branch_id": branch.branch_id,
            "description": branch.description,
            "base_seed": self.plan.base_seed,
            "source": source,
            "checkpoint_tick": self.plan.checkpoint_tick,
            "horizon_ticks": self.plan.horizon_ticks,
            "intervention": branch.intervention,
            "scheduled_tick": branch.scheduled_tick,
            "command_applied_tick": command_applied_tick,
            "first_divergence": first_divergence,
            "extinction": extinction,
            "recovery": recovery,
            "hysteresis": hysteresis,
            "effect_sizes": effect_sizes,
            "terminal_sample": terminal_sample,
        }))?;

        enforce_study_limit(
            "trajectory artifact bytes",
            csv_buf.len() as u64,
            MAX_STUDY_ARTIFACT_BYTES,
        )?;
        enforce_study_limit(
            "evidence artifact bytes",
            evidence_json.len() as u64,
            MAX_STUDY_ARTIFACT_BYTES,
        )?;
        let bundle_dir = self.output_dir.join(format!("bundle_{}", branch.branch_id));
        fs::create_dir(&bundle_dir).map_err(|error| study_output_error(&bundle_dir, error))?;

        // 3. Manifest
        let mut manifest = RunManifestRecord::unattributed(RunId::new(u128::from(
            self.plan.base_seed.wrapping_add(100),
        )));
        manifest.scenario_id = format!("branch_study_{}", branch.branch_id);
        manifest.experiment_id = Some(self.plan.study_id.clone());
        manifest.variant_id = Some(branch.branch_id.clone());

        let max_tick = self.plan.final_tick()?;

        let _bundle: RunBundleV1 = create_run_bundle_from_artifacts(
            &bundle_dir,
            manifest,
            max_tick,
            &[
                ("exports/summary.csv", "export", &csv_buf),
                ("evidence/branch_evidence.json", "evidence", &evidence_json),
            ],
        )
        .map_err(|e| BranchStudyError::Bundle(e.to_string()))?;

        // 4. Verify bundle bounded
        verify_run_bundle_bounded(&bundle_dir, STUDY_BUNDLE_VERIFICATION_LIMITS)
            .map_err(|e| BranchStudyError::Bundle(e.to_string()))?;

        Ok((bundle_dir, true))
    }

    fn generate_summary_markdown(
        &self,
        pre_intervention_digest: &str,
        source: &BranchStudySource,
        control_branch_id: &str,
        reports: &[BranchOutcomeReport],
    ) -> String {
        let mut md = String::new();
        md.push_str(&format!(
            "# Paired Checkpoint Study: {}\n\n",
            self.plan.study_id
        ));
        md.push_str(&format!("**Description**: {}\n\n", self.plan.description));
        md.push_str(&format!("- **Base Seed**: `{}`\n", self.plan.base_seed));
        match source {
            BranchStudySource::SeededWarmup => {
                md.push_str("- **Starting State**: Independently seeded population and warmup\n");
            }
            BranchStudySource::SavedCheckpoint {
                artifact_path,
                byte_size,
                checksum_blake3,
            } => {
                md.push_str(&format!(
                    "- **Starting State**: Saved scientific checkpoint [{artifact_path}]({artifact_path}), {byte_size} bytes, BLAKE3 `{checksum_blake3}`\n"
                ));
                md.push_str("- **Ownership**: New isolated scientific futures; the source host's session, scheduled commands and storage ownership are not continued\n");
                md.push_str("- **Source Retention**: One canonical checkpoint at the study root. To reproduce from this retained copy, select its path as source_checkpoint in a new study plan. Individual branch bundles contain metrics and evidence.\n");
            }
        }
        md.push_str(&format!(
            "- **Checkpoint Boundary Tick**: `{}`\n",
            self.plan.checkpoint_tick
        ));
        md.push_str(&format!(
            "- **Horizon Duration**: `{}` ticks\n",
            self.plan.horizon_ticks
        ));
        md.push_str(&format!(
            "- **Pre-Intervention Digest**: `{}`\n",
            pre_intervention_digest
        ));
        md.push_str("- **Pre-Intervention Digest Verified Across All Branches**: `true`\n\n");
        md.push_str("**Analysis basis**: One independent matched seed pair. Metric comparisons describe all recorded paired timepoints, including the shared checkpoint and any timepoints before a delayed intervention. Repeated ticks are not independent replicates; confidence intervals, standardized inferential effects, and p-values are omitted. Population recovery and extinction are assessed from the actual intervention boundary (the checkpoint for controls).\n\n");

        md.push_str("## Branch Outcomes Matrix\n\n");
        md.push_str("| Branch ID | Applied Tick | First Divergence | Extinct | Final Pop | Recovered (Ratio) | Path Dependence | Pop Trajectory Mean Diff | Artifact Integrity |\n");
        md.push_str("|---|---|---|---|---|---|---|---|---|\n");

        for r in reports {
            let app_tick = r
                .command_applied_tick
                .map_or("N/A".to_string(), |t| t.to_string());
            let div_str = r.first_divergence.as_ref().map_or_else(
                || "None".to_string(),
                |d| format!("t{} ({})", d.tick, d.stage),
            );
            let ext_str = if r.extinction.extinct {
                format!("Yes (t{})", r.extinction.extinction_tick.unwrap_or(0))
            } else {
                "No".to_string()
            };
            let rec_str = format!(
                "{} ({:.2}x)",
                if r.recovery.recovered { "Yes" } else { "No" },
                r.recovery.recovery_ratio
            );
            let path_str = if r.hysteresis.path_dependence_detected {
                format!(
                    "Yes (gap {:.1}%)",
                    r.hysteresis.terminal_trajectory_gap * 100.0
                )
            } else {
                "No".to_string()
            };
            let pop_diff_str = if let Some(eff) = r.effect_sizes.get("population") {
                format!("{:.2}", eff.mean_diff)
            } else {
                "0.00 (Control)".to_string()
            };

            md.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                r.branch_id,
                app_tick,
                div_str,
                ext_str,
                r.extinction.final_population,
                rec_str,
                path_str,
                pop_diff_str,
                if r.bundle_verified { "PASS" } else { "FAIL" }
            ));
        }

        md.push_str("\n## Causal Attribution and Divergence Analysis\n\n");
        for r in reports {
            if r.branch_id == control_branch_id {
                continue;
            }
            md.push_str(&format!(
                "### Branch `{}`: {}\n\n",
                r.branch_id, r.description
            ));
            if let Some(div) = &r.first_divergence {
                md.push_str(&format!(
                    "- **First Divergence**: Tick `{}`, Digest Lane `{}`\n",
                    div.tick, div.stage
                ));
                md.push_str(&format!("- **Details**: {}\n", div.details));
            }
            md.push_str(&format!(
                "- **Recovery Baseline**: `{}` agents at completed tick `{}`, before the intervention\n",
                r.recovery.pre_intervention_population, r.recovery.baseline_tick
            ));
            md.push_str(&format!(
                "- **Integrated Trajectory Deficit**: `{:.2}` agent-ticks\n",
                r.hysteresis.integrated_trajectory_deficit
            ));
            md.push_str(&format!(
                "- **Normalized Trajectory Difference**: `{:.4}`\n",
                r.hysteresis.normalized_trajectory_difference
            ));
            md.push_str(&format!(
                "- **Hysteresis Index**: `{:.4}`\n",
                r.hysteresis.hysteresis_index
            ));
            for (metric, eff) in &r.effect_sizes {
                md.push_str(&format!(
                    "- **Metric `{}`**: Trajectory Mean Diff `{:.4}`, Temporal SD of Paired Differences `{:.4}`, Paired Timepoints `{}` (one independent seed pair)\n",
                    metric,
                    eff.mean_diff,
                    eff.sd_diff,
                    eff.n_samples
                ));
            }
            md.push('\n');
        }

        md
    }
}

/// Execute independently initialized checkpoint pairs without retaining multiple live worlds.
pub struct ReplicatedBranchStudyOrchestrator {
    /// Common study and explicit independent founding seeds.
    pub plan: ReplicatedBranchStudyPlan,
    /// New or empty destination for the aggregate and per-seed evidence.
    pub output_dir: PathBuf,
}

impl ReplicatedBranchStudyOrchestrator {
    /// Compose replication around the existing exact per-seed executor.
    #[must_use]
    pub fn new(plan: ReplicatedBranchStudyPlan, output_dir: PathBuf) -> Self {
        Self { plan, output_dir }
    }

    /// Validate all common inputs, independent seeds and aggregate work before writing artifacts.
    ///
    /// # Errors
    ///
    /// Refuses a shared saved source, empty/duplicate seed lists, unsupported study inputs, or
    /// total work exceeding the single-study budget. Replication never multiplies that budget.
    pub fn validate(&self) -> Result<(), BranchStudyError> {
        if self.plan.study.source_checkpoint.is_some() {
            return Err(BranchStudyError::InvalidPlan(
                "source_checkpoint cannot be combined with independent --seeds replicates; one saved world is one matched pair".to_owned(),
            ));
        }
        if self.plan.seeds.is_empty() {
            return Err(BranchStudyError::InvalidPlan(
                "replicated study requires at least one explicit seed".to_owned(),
            ));
        }
        enforce_study_limit(
            "replicate count",
            self.plan.seeds.len() as u64,
            MAX_STUDY_REPLICATES as u64,
        )?;
        let mut unique = BTreeSet::new();
        for &seed in &self.plan.seeds {
            if !unique.insert(seed) {
                return Err(BranchStudyError::InvalidPlan(format!(
                    "duplicate replicate seed {seed}; repeated seeds are not independent pairs"
                )));
            }
        }
        enforce_study_limit(
            "replicated total transitions",
            self.total_transitions()?,
            MAX_STUDY_TOTAL_TRANSITIONS,
        )?;
        // Seed values themselves have no restricted range. All geometry, branches, schedules,
        // brain registration and per-world limits are validated by the same production preflight.
        BranchStudyOrchestrator::new(self.plan.study.clone(), self.output_dir.clone()).validate()
    }

    fn total_transitions(&self) -> Result<u64, BranchStudyError> {
        self.plan
            .study
            .total_transitions()?
            .checked_mul(self.plan.seeds.len() as u64)
            .ok_or_else(|| {
                BranchStudyError::InvalidPlan("replicated study work count overflows".to_owned())
            })
    }

    fn replicate_plan(&self, seed: u64) -> BranchStudyPlan {
        let mut plan = self.plan.study.clone();
        plan.base_seed = seed;
        plan.config.rng_seed = Some(seed);
        plan
    }

    /// Run all declared pairs and publish endpoint inference only after complete success.
    ///
    /// # Errors
    ///
    /// Retains the exact plan and completed/partial seed outputs on refusal, failure or
    /// cancellation. No aggregate success report is written for an incomplete seed set.
    pub fn execute_study(
        &self,
        cancellation: Option<&AtomicBool>,
    ) -> Result<ReplicatedBranchStudyReport, BranchStudyError> {
        self.validate()?;
        check_study_cancellation(cancellation)?;
        reserve_study_output(&self.output_dir, &serde_json::to_vec_pretty(&self.plan)?)?;

        let mut replicates = Vec::with_capacity(self.plan.seeds.len());
        for &seed in &self.plan.seeds {
            check_study_cancellation(cancellation)?;
            let directory = format!("seed_{seed:016x}");
            // Each call constructs its own founders, captures its own checkpoint, and verifies
            // restoration separately for every branch. No checkpoint is shared across seeds.
            let report = BranchStudyOrchestrator::new(
                self.replicate_plan(seed),
                self.output_dir.join(&directory),
            )
            .execute_study(cancellation)
            .map_err(|source| BranchStudyError::ReplicateFailed {
                seed,
                source: Box::new(source),
            })?;
            replicates.push(BranchStudyReplicate {
                seed,
                report_path: format!("{directory}/study_report.json"),
                report,
            });
        }
        check_study_cancellation(cancellation)?;
        let endpoint_effects = self.aggregate_endpoints(&replicates)?;
        let mut report = ReplicatedBranchStudyReport {
            schema_version: REPLICATED_BRANCH_STUDY_REPORT_SCHEMA_VERSION,
            plan: self.plan.clone(),
            endpoint_tick: self.plan.study.final_tick()?,
            total_transitions: self.total_transitions()?,
            replicates,
            endpoint_effects,
            summary_markdown: String::new(),
        };
        report.summary_markdown = replicated_summary(&report);
        let report_json = serde_json::to_vec_pretty(&report)?;
        enforce_study_limit(
            "replicated report bytes",
            report_json.len() as u64,
            MAX_STUDY_ARTIFACT_BYTES,
        )?;
        check_study_cancellation(cancellation)?;
        write_new_study_artifact(
            &self.output_dir.join("replicated_summary.md"),
            report.summary_markdown.as_bytes(),
        )?;
        check_study_cancellation(cancellation)?;
        write_new_study_artifact(
            &self.output_dir.join("replicated_report.json"),
            &report_json,
        )?;
        Ok(report)
    }

    fn aggregate_endpoints(
        &self,
        replicates: &[BranchStudyReplicate],
    ) -> Result<BTreeMap<String, BTreeMap<String, BranchEffectSize>>, BranchStudyError> {
        let control_id = self
            .plan
            .study
            .branches
            .iter()
            .find(|branch| branch.intervention.is_none())
            .ok_or(BranchStudyError::MissingControlBranch)?
            .branch_id
            .as_str();
        self.validate_replicate_results(replicates, control_id)?;
        let mut effects = BTreeMap::new();
        for branch in &self.plan.study.branches {
            if branch.branch_id == control_id {
                continue;
            }
            let mut control_values = [Vec::new(), Vec::new(), Vec::new()];
            let mut branch_values = [Vec::new(), Vec::new(), Vec::new()];
            for replicate in replicates {
                let endpoints = &replicate.report.branch_reports;
                let control = endpoints
                    .iter()
                    .find(|result| result.branch_id == control_id)
                    .ok_or(BranchStudyError::MissingControlBranch)?;
                let treatment = endpoints
                    .iter()
                    .find(|result| result.branch_id == branch.branch_id)
                    .ok_or_else(|| {
                        BranchStudyError::InvalidComparison(format!(
                            "seed {} is missing branch {}",
                            replicate.seed, branch.branch_id
                        ))
                    })?;
                for (values, sample) in [
                    (&mut control_values, &control.terminal_sample),
                    (&mut branch_values, &treatment.terminal_sample),
                ] {
                    // Population has already passed the 10,000-agent product limit.
                    let population = u32::try_from(sample.population).map_err(|_| {
                        BranchStudyError::InvalidComparison(
                            "endpoint population overflows".to_owned(),
                        )
                    })?;
                    values[0].push(f64::from(population));
                    values[1].push(sample.food_total);
                    values[2].push(f64::from(sample.total_energy));
                }
            }
            let mut metrics = BTreeMap::new();
            for (index, metric) in ["population", "food_total", "total_energy"]
                .iter()
                .enumerate()
            {
                metrics.insert(
                    (*metric).to_owned(),
                    compute_paired_effect(
                        &control_values[index],
                        &branch_values[index],
                        metric,
                        self.plan.study.base_seed,
                    )?,
                );
            }
            effects.insert(branch.branch_id.clone(), metrics);
        }
        Ok(effects)
    }

    fn validate_replicate_results(
        &self,
        replicates: &[BranchStudyReplicate],
        control_id: &str,
    ) -> Result<(), BranchStudyError> {
        if replicates.len() != self.plan.seeds.len() || replicates.is_empty() {
            return Err(BranchStudyError::InvalidComparison(
                "replicate set is incomplete; no partial inference is published".to_owned(),
            ));
        }
        let endpoint_tick = self.plan.study.final_tick()?;
        let expected_branches: BTreeSet<_> = self
            .plan
            .study
            .branches
            .iter()
            .map(|branch| branch.branch_id.as_str())
            .collect();
        for (&seed, replicate) in self.plan.seeds.iter().zip(replicates) {
            let report = &replicate.report;
            if replicate.seed != seed
                || report.base_seed != seed
                || report.source != BranchStudySource::SeededWarmup
                || report.schema_version != BRANCH_STUDY_REPORT_SCHEMA_VERSION
                || report.study_id != self.plan.study.study_id
                || report.checkpoint_tick != self.plan.study.checkpoint_tick
                || report.horizon_ticks != self.plan.study.horizon_ticks
                || report.control_branch_id != control_id
                || !report.pre_intervention_digest_verified_for_all_branches
                || report.pre_intervention_digest.is_empty()
                || report.branch_reports.len() != expected_branches.len()
            {
                return Err(BranchStudyError::InvalidComparison(format!(
                    "replicate seed {seed} does not match its declared study inputs"
                )));
            }
            let mut found = BTreeSet::new();
            for branch in &report.branch_reports {
                let sample = &branch.terminal_sample;
                if !expected_branches.contains(branch.branch_id.as_str())
                    || !found.insert(branch.branch_id.as_str())
                    || !branch.bundle_verified
                    || branch.bundle_path.is_none()
                    || sample.tick != endpoint_tick
                    || sample.digest_overall.is_empty()
                    || sample.digest_overall != branch.final_digest
                    || sample.population != branch.extinction.final_population
                    || sample.population as u64 > MAX_STUDY_POPULATION
                    || !sample.food_total.is_finite()
                    || !sample.total_energy.is_finite()
                {
                    return Err(BranchStudyError::InvalidComparison(format!(
                        "replicate seed {seed} branch {} lacks a verified terminal endpoint",
                        branch.branch_id
                    )));
                }
            }
        }
        Ok(())
    }
}

fn check_study_cancellation(cancellation: Option<&AtomicBool>) -> Result<(), BranchStudyError> {
    if cancellation.is_some_and(|cancel| cancel.load(Ordering::Relaxed)) {
        Err(BranchStudyError::Cancelled)
    } else {
        Ok(())
    }
}

fn replicated_summary(report: &ReplicatedBranchStudyReport) -> String {
    let mut markdown = format!(
        "# Replicated Checkpoint Study: {}\n\n{}\n\n\
         **Analysis basis**: {} independently initialized matched seed pairs. Each pair contributes \
         exactly one terminal observation at tick {} for population, food total and total energy. \
         Repeated ticks are not independent replicates.\n\n\
         Seeds, in execution order: `{:?}`. Statistical resampling seed: `{}`. \
         Total admitted simulation transitions: `{}`.\n\n\
         Intervals are 95% paired bootstrap intervals; p-values use the two-sided paired sign-flip \
         test. Both use 1,000 seeded resamples and are unadjusted for multiple endpoint or branch \
         comparisons. Seed choices define the sampled simulation ensemble; no population-wide \
         biological claim or automatic winner decision is made. One seed remains descriptive \
         and has no interval or p-value.\n\n\
         | Branch | Terminal metric | Independent pairs | Mean treatment − control | 95% interval | p-value |\n\
         |---|---|---:|---:|---|---:|\n",
        report.plan.study.study_id,
        report.plan.study.description,
        report.replicates.len(),
        report.endpoint_tick,
        report.plan.seeds,
        report.plan.study.base_seed,
        report.total_transitions,
    );
    for (branch, metrics) in &report.endpoint_effects {
        for (metric, effect) in metrics {
            let interval = effect.ci_95.map_or_else(
                || "unavailable".to_owned(),
                |(lower, upper)| format!("[{lower:.6}, {upper:.6}]"),
            );
            let p_value = effect
                .p_value
                .map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.6}"));
            markdown.push_str(&format!(
                "| {branch} | {metric} | {} | {:.6} | {interval} | {p_value} |\n",
                effect.n_samples, effect.mean_diff,
            ));
        }
    }
    markdown.push_str("\n## Independent checkpoint evidence\n\n| Seed | Shared checkpoint digest | Per-seed report |\n|---:|---|---|\n");
    for replicate in &report.replicates {
        markdown.push_str(&format!(
            "| {} | `{}` | [{}]({}) |\n",
            replicate.seed,
            replicate.report.pre_intervention_digest,
            replicate.report_path,
            replicate.report_path,
        ));
    }
    markdown
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn world_at_checkpoint(plan: &BranchStudyPlan) -> WorldState {
        let mut config = plan.config.clone();
        config.persistence_interval = 0;
        config.rng_seed = Some(plan.base_seed);
        let mut world = WorldState::new(config).expect("valid study world");
        let installed = install_brains(&mut world, plan.brain_preset).expect("register brains");
        seed_founding_population(&mut world, installed.population()).expect("seed founders");
        for _ in 0..plan.checkpoint_tick {
            world.step().expect("advance to checkpoint");
        }
        world
    }

    fn replicated_fixture() -> ReplicatedBranchStudyPlan {
        let mut study = BranchStudyPlan::standard_five_branch("replicated-endpoints", 19, 2, 4);
        study.brain_preset = BrainPreset::Mlp;
        study.config.world_width = 480;
        study.config.world_height = 480;
        study.config.food_cell_size = 40;
        study.config.initial_food = study.config.food_max;
        study.config.population_minimum = 0;
        study.config.population_spawn_interval = 0;
        study.config.reproduction_attempt_chance = 0.0;
        study
            .branches
            .retain(|branch| matches!(branch.branch_id.as_str(), "control" | "resource_shock"));
        let shock = study
            .branches
            .iter_mut()
            .find(|branch| branch.branch_id == "resource_shock")
            .unwrap();
        shock.intervention = Some(StudyIntervention::ResourceShock {
            region: None,
            bloom_amount: 0.0,
            scorch: 0.5,
        });
        shock.scheduled_tick = Some(4);
        ReplicatedBranchStudyPlan {
            study,
            seeds: vec![42, 137],
        }
    }

    #[test]
    fn saved_checkpoint_identity_and_cancellation_refuse_before_claiming_output() {
        let directory = tempdir().unwrap();
        let plan = replicated_fixture().study;
        let source_world = world_at_checkpoint(&plan);
        let checkpoint = source_world.checkpoint_v1().unwrap();
        let original_bytes = checkpoint.encode().unwrap();
        let mut wrong_config = plan.clone();
        wrong_config.config.food_growth_rate *= 0.5;
        let mut wrong_tick = plan.clone();
        wrong_tick.checkpoint_tick += 1;
        let mut wrong_root = plan.clone();
        wrong_root.base_seed ^= 1;
        for (label, invalid, expected) in [
            ("configuration", wrong_config, "config"),
            ("boundary", wrong_tick, "checkpoint_tick"),
            ("root", wrong_root, "RNG root seed"),
        ] {
            let output = directory.path().join(label);
            let error = BranchStudyOrchestrator::new(invalid, output.clone())
                .execute_from_checkpoint(&checkpoint, None)
                .unwrap_err();
            assert!(
                matches!(error, BranchStudyError::CheckpointMismatch(ref message) if message.contains(expected))
            );
            assert!(!output.exists());
        }
        let mut wrong_registry = plan.clone();
        wrong_registry.brain_preset = BrainPreset::Dwraon;
        let output = directory.path().join("registry");
        assert!(matches!(
            BranchStudyOrchestrator::new(wrong_registry, output.clone())
                .execute_from_checkpoint(&checkpoint, None),
            Err(BranchStudyError::Checkpoint(
                WorldCheckpointError::RegistryMismatch { .. }
            ))
        ));
        assert!(!output.exists());
        let output = directory.path().join("cancelled");
        assert!(matches!(
            BranchStudyOrchestrator::new(plan, output.clone())
                .execute_from_checkpoint(&checkpoint, Some(&AtomicBool::new(true))),
            Err(BranchStudyError::Cancelled)
        ));
        assert!(!output.exists());
        assert_eq!(checkpoint.encode().unwrap(), original_bytes);
        assert_eq!(
            source_world.world_digest_v1().unwrap(),
            *checkpoint.source_digest()
        );
    }

    #[test]
    fn declared_source_executes_the_same_future_and_retains_exact_inputs() {
        let directory = tempdir().unwrap();
        let mut plan = replicated_fixture().study;
        let checkpoint = world_at_checkpoint(&plan).checkpoint_v1().unwrap();
        let original = checkpoint.encode().unwrap();
        let source_path = directory.path().join("evolved-source.bin");
        fs::write(&source_path, &original).unwrap();
        let supplied =
            BranchStudyOrchestrator::new(plan.clone(), directory.path().join("supplied"))
                .execute_from_checkpoint(&checkpoint, None)
                .unwrap();
        plan.source_checkpoint = Some(source_path.clone());
        let output = directory.path().join("declared");
        let declared = BranchStudyOrchestrator::new(plan.clone(), output.clone())
            .execute_study(None)
            .unwrap();
        assert_eq!(
            declared.pre_intervention_digest,
            supplied.pre_intervention_digest
        );
        assert_eq!(declared.source, supplied.source);
        for (declared_branch, supplied_branch) in
            declared.branch_reports.iter().zip(&supplied.branch_reports)
        {
            assert_eq!(declared_branch.final_digest, supplied_branch.final_digest);
            assert_eq!(
                declared_branch.terminal_sample,
                supplied_branch.terminal_sample
            );
            assert_eq!(
                declared_branch.first_divergence,
                supplied_branch.first_divergence
            );
        }
        let retained_plan: BranchStudyPlan =
            serde_json::from_slice(&fs::read(output.join("study_plan.json")).unwrap()).unwrap();
        assert_eq!(retained_plan, plan);
        assert_eq!(fs::read(&source_path).unwrap(), original);
        assert_eq!(
            fs::read(output.join(SOURCE_CHECKPOINT_ARTIFACT)).unwrap(),
            original
        );
        let source: BranchStudySource =
            serde_json::from_slice(&fs::read(output.join("study_source.json")).unwrap()).unwrap();
        assert_eq!(
            source,
            BranchStudySource::SavedCheckpoint {
                artifact_path: SOURCE_CHECKPOINT_ARTIFACT.to_owned(),
                byte_size: original.len() as u64,
                checksum_blake3: blake3::hash(&original).to_hex().to_string(),
            }
        );
        assert_eq!(source, declared.source);
        assert!(matches!(
            BranchStudyOrchestrator::new(plan, output.clone()).execute_study(None),
            Err(BranchStudyError::OutputAlreadyExists { .. })
        ));
        assert_eq!(
            fs::read(output.join(SOURCE_CHECKPOINT_ARTIFACT)).unwrap(),
            original
        );
    }

    #[test]
    fn declared_sources_refuse_replication_and_ambiguous_api_before_file_reads() {
        let directory = tempdir().unwrap();
        let mut replicated = replicated_fixture();
        let checkpoint = world_at_checkpoint(&replicated.study)
            .checkpoint_v1()
            .unwrap();
        replicated.study.source_checkpoint = Some(directory.path().join("missing-source.bin"));
        let output = directory.path().join("replicated");
        assert!(matches!(
            ReplicatedBranchStudyOrchestrator::new(replicated.clone(), output.clone())
                .execute_study(None),
            Err(BranchStudyError::InvalidPlan(ref reason)) if reason.contains("--seeds")
        ));
        assert!(!output.exists());
        let output = directory.path().join("ambiguous");
        assert!(matches!(
            BranchStudyOrchestrator::new(replicated.study.clone(), output.clone())
                .execute_from_checkpoint(&checkpoint, None),
            Err(BranchStudyError::InvalidPlan(ref reason)) if reason.contains("also declare")
        ));
        assert!(!output.exists());
        let output = directory.path().join("cancelled");
        assert!(matches!(
            BranchStudyOrchestrator::new(replicated.study, output.clone())
                .execute_study(Some(&AtomicBool::new(true))),
            Err(BranchStudyError::Cancelled)
        ));
        assert!(!output.exists());
    }

    #[test]
    fn imported_work_counts_future_transitions_without_replaying_old_ticks() {
        let mut plan = replicated_fixture().study;
        plan.checkpoint_tick = MAX_STUDY_TOTAL_TRANSITIONS + 1;
        for branch in &mut plan.branches {
            branch.scheduled_tick = None;
        }
        let mut runner = BranchStudyOrchestrator::new(plan, PathBuf::from("unused"));
        assert!(matches!(
            runner.validate(),
            Err(BranchStudyError::CapacityExceeded {
                resource: "total transitions",
                ..
            })
        ));
        runner
            .validate_with_config(
                &runner.plan.config,
                runner.plan.branch_transitions().unwrap(),
                true,
            )
            .expect("a supplied cut performs only its bounded future transitions");
        runner.plan.source_checkpoint = Some(PathBuf::from("old-source.bin"));
        runner
            .validate()
            .expect("a declared cut has no replayed warmup and validation performs no file reads");
        assert_eq!(runner.plan.total_transitions().unwrap(), 8);
    }

    #[test]
    fn checkpoint_studies_refuse_missing_selection_history_in_sources_and_treatments() {
        let directory = tempdir().unwrap();
        let base = replicated_fixture().study;
        let fitness_source = world_at_checkpoint(&base).checkpoint_v1().unwrap();
        for (index, mode) in [
            EvolutionSelectionMode::Novelty,
            EvolutionSelectionMode::Curiosity { w: 0.5 },
        ]
        .into_iter()
        .enumerate()
        {
            let patch = serde_json::json!({"selection_mode": mode});
            let mut source_plan = base.clone();
            source_plan.config.selection_mode = mode;
            let checkpoint = world_at_checkpoint(&source_plan).checkpoint_v1().unwrap();
            let original = checkpoint.encode().unwrap();
            for imported in [false, true] {
                let output = directory.path().join(format!("mode-{index}-{imported}"));
                let runner = BranchStudyOrchestrator::new(source_plan.clone(), output.clone());
                let result = if imported {
                    runner.execute_from_checkpoint(&checkpoint, None)
                } else {
                    runner.execute_study(None)
                };
                assert!(matches!(
                    result,
                    Err(BranchStudyError::InvalidPlan(ref reason)) if reason.contains("selection history")
                ));
                assert!(!output.exists());
            }
            assert_eq!(checkpoint.encode().unwrap(), original);

            let mut treatment_plan = base.clone();
            treatment_plan.branches[1].intervention = Some(StudyIntervention::ConfigPatch(patch));
            for imported in [false, true] {
                let output = directory.path().join(format!("patch-{index}-{imported}"));
                let runner = BranchStudyOrchestrator::new(treatment_plan.clone(), output.clone());
                let result = if imported {
                    runner.execute_from_checkpoint(&fitness_source, None)
                } else {
                    runner.execute_study(None)
                };
                assert!(matches!(
                    result,
                    Err(BranchStudyError::InvalidPlan(ref reason)) if reason.contains("selection history")
                ));
                assert!(!output.exists());
            }
        }
    }

    #[test]
    fn checkpoint_studies_refuse_missing_scramble_rng_in_sources_and_treatments() {
        let directory = tempdir().unwrap();
        let base = replicated_fixture().study;
        let fitness_source = world_at_checkpoint(&base).checkpoint_v1().unwrap();
        let mut source_plan = base.clone();
        source_plan.config.communication_scramble = true;
        let checkpoint = world_at_checkpoint(&source_plan).checkpoint_v1().unwrap();
        let original = checkpoint.encode().unwrap();
        for imported in [false, true] {
            let output = directory.path().join(format!("source-{imported}"));
            let runner = BranchStudyOrchestrator::new(source_plan.clone(), output.clone());
            let result = if imported {
                runner.execute_from_checkpoint(&checkpoint, None)
            } else {
                runner.execute_study(None)
            };
            assert!(matches!(
                result,
                Err(BranchStudyError::InvalidPlan(ref reason)) if reason.contains("scrambling RNG")
            ));
            assert!(!output.exists());
        }
        let mut treatment_plan = base;
        treatment_plan.branches[1].intervention = Some(StudyIntervention::ConfigPatch(
            serde_json::json!({"communication_scramble": true}),
        ));
        for imported in [false, true] {
            let output = directory.path().join(format!("patch-{imported}"));
            let runner = BranchStudyOrchestrator::new(treatment_plan.clone(), output.clone());
            let result = if imported {
                runner.execute_from_checkpoint(&fitness_source, None)
            } else {
                runner.execute_study(None)
            };
            assert!(matches!(
                result,
                Err(BranchStudyError::InvalidPlan(ref reason)) if reason.contains("scrambling RNG")
            ));
            assert!(!output.exists());
        }
        assert_eq!(checkpoint.encode().unwrap(), original);
    }

    #[test]
    fn checkpoint_studies_refuse_archive_dependent_treatments_before_output() {
        let directory = tempdir().unwrap();
        let mut plan = replicated_fixture().study;
        plan.config.archive_enabled = true;
        plan.branches[1].intervention =
            Some(StudyIntervention::Core(Intervention::SpawnFromArchive {
                selector: scriptbots_core::CellSelector::TopKByQuality(1),
                placement: scriptbots_core::Placement::Seeded {
                    region: Region::All,
                    seed: 1111,
                },
                headroom: None,
                herbivore_tendency: None,
            }));
        let checkpoint = world_at_checkpoint(&plan).checkpoint_v1().unwrap();
        for imported in [false, true] {
            let output = directory.path().join(format!("archive-{imported}"));
            let runner = BranchStudyOrchestrator::new(plan.clone(), output.clone());
            let result = if imported {
                runner.execute_from_checkpoint(&checkpoint, None)
            } else {
                runner.execute_study(None)
            };
            assert!(matches!(
                result,
                Err(BranchStudyError::InvalidIntervention { ref reason, .. })
                    if reason.contains("archive")
            ));
            assert!(!output.exists());
        }
    }

    #[test]
    fn saved_map_temperature_refuses_even_when_raw_restore_digest_matches() {
        let directory = tempdir().unwrap();
        let output = directory.path().join("map-source");
        let plan = replicated_fixture().study;
        let mut world = world_at_checkpoint(&plan);
        let width = world.terrain().width();
        let height = world.terrain().height();
        let cells = world.terrain().tiles().len();
        let artifact = scriptbots_core::MapArtifact::new(
            world.terrain().clone(),
            None,
            Some(scriptbots_core::ScalarField::new(width, height, vec![0.9; cells]).unwrap()),
            None,
            None,
            scriptbots_core::MapArtifactMetadata {
                generator: scriptbots_core::MapGeneratorKind::RuleBased,
                tileset_id: "saved-cut-temperature".to_owned(),
                tileset_hash: 1,
                seed: plan.base_seed,
                width,
                height,
                attempt_count: 1,
                succeeded_on: 1,
                generated_at_epoch_ms: 0,
            },
        )
        .unwrap();
        world.apply_map_artifact(&artifact).unwrap();
        let checkpoint = world.checkpoint_v1().unwrap();
        let original = checkpoint.encode().unwrap();
        let registry =
            crate::brains::create_brain_registry_for_config(plan.brain_preset, checkpoint.config())
                .unwrap();
        let raw_restored = WorldState::restore_checkpoint_v1(&checkpoint, registry).unwrap();
        assert_eq!(
            raw_restored.world_digest_v1().unwrap(),
            *checkpoint.source_digest()
        );
        assert!(raw_restored.map_metadata().is_some());
        assert!(matches!(
            BranchStudyOrchestrator::new(plan, output.clone())
                .execute_from_checkpoint(&checkpoint, None),
            Err(BranchStudyError::CheckpointMismatch(ref reason)) if reason.contains("map temperature")
        ));
        assert!(!output.exists());
        assert_eq!(checkpoint.encode().unwrap(), original);
        assert_eq!(
            world.world_digest_v1().unwrap(),
            *checkpoint.source_digest()
        );
    }

    #[test]
    fn supported_fitness_configuration_treatments_keep_real_matched_futures() {
        let directory = tempdir().unwrap();
        let mut plan = replicated_fixture().study;
        plan.branches[1].intervention = Some(StudyIntervention::ConfigPatch(
            serde_json::json!({"food_growth_rate": plan.config.food_growth_rate * 0.5}),
        ));
        let checkpoint = world_at_checkpoint(&plan).checkpoint_v1().unwrap();
        let seeded = BranchStudyOrchestrator::new(plan.clone(), directory.path().join("seeded"))
            .execute_study(None)
            .unwrap();
        let imported = BranchStudyOrchestrator::new(plan, directory.path().join("imported"))
            .execute_from_checkpoint(&checkpoint, None)
            .unwrap();
        for (seeded_branch, imported_branch) in
            seeded.branch_reports.iter().zip(&imported.branch_reports)
        {
            assert_eq!(seeded_branch.final_digest, imported_branch.final_digest);
            assert_eq!(
                seeded_branch.terminal_sample,
                imported_branch.terminal_sample
            );
            assert_eq!(
                seeded_branch.first_divergence,
                imported_branch.first_divergence
            );
        }
        assert_eq!(
            imported.branch_reports[1]
                .first_divergence
                .as_ref()
                .unwrap()
                .stage,
            "config"
        );
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn neuro_studies_restore_the_exact_configured_protocol_family() {
        let directory = tempdir().unwrap();
        let mut plan = replicated_fixture().study;
        plan.brain_preset = BrainPreset::Neuro;
        plan.config.neuroflow.hidden_layers = vec![7, 3];
        plan.config.neuroflow.activation = scriptbots_core::NeuroflowActivationKind::Sigmoid;
        let checkpoint = world_at_checkpoint(&plan).checkpoint_v1().unwrap();
        assert!(checkpoint.source_digest().evaluator_state_covered);
        assert!(checkpoint.source_digest().factory_state_covered);
        let seeded = BranchStudyOrchestrator::new(plan.clone(), directory.path().join("seeded"))
            .execute_study(None)
            .expect("explicit NeuroFlow seeded study");
        let imported = BranchStudyOrchestrator::new(plan, directory.path().join("imported"))
            .execute_from_checkpoint(&checkpoint, None)
            .expect("configured NeuroFlow checkpoint study");
        assert_eq!(seeded.source, BranchStudySource::SeededWarmup);
        assert_eq!(
            seeded.pre_intervention_digest,
            imported.pre_intervention_digest
        );
        for (seeded_branch, imported_branch) in
            seeded.branch_reports.iter().zip(&imported.branch_reports)
        {
            assert_eq!(seeded_branch.final_digest, imported_branch.final_digest);
            assert_eq!(
                seeded_branch.terminal_sample,
                imported_branch.terminal_sample
            );
            assert_eq!(
                seeded_branch.first_divergence,
                imported_branch.first_divergence
            );
        }
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn saved_neuro_checkpoint_refuses_constructor_drift_even_without_agents() {
        let directory = tempdir().unwrap();
        let output = directory.path().join("constructor-drift");
        let mut plan = replicated_fixture().study;
        plan.brain_preset = BrainPreset::Neuro;
        plan.config.neuroflow.hidden_layers = vec![7, 3];
        let mut world = WorldState::new(plan.config.clone()).unwrap();
        let installed = install_brains(&mut world, plan.brain_preset).unwrap();
        assert_eq!(installed.population().len(), 1);
        assert!(
            world
                .brain_registry()
                .is_protocol_family(installed.population()[0])
        );
        // No genomes are available to guess the old constructor settings. The retained
        // registry identity remains authoritative for an empty source world.
        for _ in 0..plan.checkpoint_tick {
            world.step().unwrap();
        }
        assert!(world.agents().is_empty());
        plan.config.neuroflow.hidden_layers = vec![11, 5];
        apply_control_command(
            &mut world,
            ControlCommand::UpdateConfig(Box::new(plan.config.clone())),
        )
        .unwrap();
        let checkpoint = world.checkpoint_v1().unwrap();
        let original = checkpoint.encode().unwrap();
        assert_eq!(checkpoint.config(), &plan.config);
        assert_eq!(checkpoint.tick().0, plan.checkpoint_tick);
        assert_eq!(
            world.agent_substream_protocol_v1().root_seed(),
            plan.base_seed
        );
        assert!(matches!(
            BranchStudyOrchestrator::new(plan, output.clone())
                .execute_from_checkpoint(&checkpoint, None),
            Err(BranchStudyError::Checkpoint(
                WorldCheckpointError::RegistryMismatch { .. }
            ))
        ));
        assert!(!output.exists());
        assert_eq!(checkpoint.encode().unwrap(), original);
        assert_eq!(
            world.world_digest_v1().unwrap(),
            *checkpoint.source_digest()
        );
    }

    #[test]
    fn saved_source_requires_exact_config_encoding_including_signed_zero() {
        let directory = tempdir().unwrap();
        let mut plan = replicated_fixture().study;
        plan.config.food_growth_rate = 0.0;
        let checkpoint = world_at_checkpoint(&plan).checkpoint_v1().unwrap();
        let original = checkpoint.encode().unwrap();
        let source_path = directory.path().join("positive-zero-source.bin");
        fs::write(&source_path, &original).unwrap();
        let mut mismatched = plan.clone();
        mismatched.config.food_growth_rate = -0.0;
        assert_eq!(mismatched.config, plan.config);
        assert_ne!(
            serde_json::to_vec(&mismatched.config).unwrap(),
            serde_json::to_vec(checkpoint.config()).unwrap(),
        );
        for declared in [false, true] {
            let output = directory.path().join(format!("signed-zero-{declared}"));
            let mut candidate = mismatched.clone();
            if declared {
                candidate.source_checkpoint = Some(source_path.clone());
            }
            let runner = BranchStudyOrchestrator::new(candidate, output.clone());
            let result = if declared {
                runner.execute_study(None)
            } else {
                runner.execute_from_checkpoint(&checkpoint, None)
            };
            assert!(matches!(
                result,
                Err(BranchStudyError::CheckpointMismatch(ref reason)) if reason.contains("config")
            ));
            assert!(!output.exists());
        }
        assert_eq!(fs::read(source_path).unwrap(), original);
        assert_eq!(checkpoint.encode().unwrap(), original);
    }

    #[test]
    fn replicated_study_refuses_duplicate_seeds_and_bounds_total_work_before_output() {
        let directory = tempdir().unwrap();
        let output = directory.path().join("not-reserved");
        let plan = replicated_fixture();
        for seeds in [vec![], vec![42, 42]] {
            let invalid = ReplicatedBranchStudyPlan {
                seeds,
                ..plan.clone()
            };
            assert!(matches!(
                ReplicatedBranchStudyOrchestrator::new(invalid, output.clone()).execute_study(None),
                Err(BranchStudyError::InvalidPlan(_))
            ));
            assert!(!output.exists());
        }
        let too_many = ReplicatedBranchStudyPlan {
            seeds: (0..=MAX_STUDY_REPLICATES as u64).collect(),
            ..plan.clone()
        };
        assert!(matches!(
            ReplicatedBranchStudyOrchestrator::new(too_many, output.clone()).execute_study(None),
            Err(BranchStudyError::CapacityExceeded {
                resource: "replicate count",
                ..
            })
        ));
        let mut at_limit = plan;
        at_limit.study.checkpoint_tick = MAX_STUDY_TOTAL_TRANSITIONS / 2 - 8;
        at_limit.study.branches[1].scheduled_tick = None;
        ReplicatedBranchStudyOrchestrator::new(at_limit.clone(), output.clone())
            .validate()
            .expect("aggregate exact transition limit");
        at_limit.study.checkpoint_tick += 1;
        assert!(matches!(
            ReplicatedBranchStudyOrchestrator::new(at_limit, output.clone()).execute_study(None),
            Err(BranchStudyError::CapacityExceeded {
                resource: "replicated total transitions",
                ..
            })
        ));
        assert!(
            !output.exists(),
            "no rejected plan reserves its destination"
        );
    }

    #[test]
    fn replicated_study_uses_real_independent_endpoints_and_rejects_incomplete_evidence() {
        let directory = tempdir().unwrap();
        let plan = replicated_fixture();
        let orchestrator =
            ReplicatedBranchStudyOrchestrator::new(plan.clone(), directory.path().join("study"));
        let report = orchestrator.execute_study(None).unwrap();
        assert_eq!(report.endpoint_tick, 6);
        assert_eq!(report.total_transitions, 20);
        assert_eq!(report.replicates.len(), 2);
        assert_ne!(
            report.replicates[0].report.pre_intervention_digest,
            report.replicates[1].report.pre_intervention_digest,
            "independent founders must not reuse one checkpoint"
        );
        let mut observed_food_difference = 0.0;
        for (&seed, replicate) in plan.seeds.iter().zip(&report.replicates) {
            assert_eq!(replicate.seed, seed);
            assert_eq!(replicate.report.base_seed, seed);
            let branches = &replicate.report.branch_reports;
            let control = branches
                .iter()
                .find(|branch| branch.branch_id == "control")
                .unwrap();
            let shock = branches
                .iter()
                .find(|branch| branch.branch_id == "resource_shock")
                .unwrap();
            observed_food_difference +=
                shock.terminal_sample.food_total - control.terminal_sample.food_total;
            for branch in branches {
                let evidence: serde_json::Value = serde_json::from_slice(
                    &fs::read(
                        Path::new(branch.bundle_path.as_deref().unwrap())
                            .join("evidence/branch_evidence.json"),
                    )
                    .unwrap(),
                )
                .unwrap();
                assert_eq!(
                    evidence["terminal_sample"],
                    serde_json::to_value(&branch.terminal_sample).unwrap()
                );
                assert_eq!(branch.terminal_sample.tick, report.endpoint_tick);
                assert_eq!(branch.terminal_sample.digest_overall, branch.final_digest);
                for effect in branch.effect_sizes.values() {
                    assert_eq!(effect.n_samples, 5);
                    assert!(effect.ci_95.is_none() && effect.p_value.is_none());
                }
            }
        }
        let effects = &report.endpoint_effects["resource_shock"];
        assert_eq!(effects.len(), 3);
        assert_eq!(
            effects["food_total"].mean_diff,
            observed_food_difference / 2.0
        );
        assert!(
            effects["food_total"].mean_diff < 0.0,
            "observed shock must deplete food"
        );
        for effect in effects.values() {
            assert_eq!(effect.n_samples, 2);
            assert_eq!(
                effect.analysis_basis,
                BranchAnalysisBasis::IndependentPairedReplicates { pairs: 2 }
            );
            assert!(effect.ci_95.is_some() && effect.p_value.is_some());
        }
        assert!(
            orchestrator
                .aggregate_endpoints(&report.replicates[..1])
                .is_err()
        );
        let mut altered = report.replicates.clone();
        altered[1].report.base_seed = altered[0].report.base_seed;
        assert!(orchestrator.aggregate_endpoints(&altered).is_err());
        let mut altered = report.replicates.clone();
        altered[0].report.source = BranchStudySource::SavedCheckpoint {
            artifact_path: SOURCE_CHECKPOINT_ARTIFACT.to_owned(),
            byte_size: 1,
            checksum_blake3: "not-an-independent-seed".to_owned(),
        };
        assert!(orchestrator.aggregate_endpoints(&altered).is_err());
        let mut altered = report.replicates.clone();
        altered[0].report.branch_reports[1].terminal_sample.tick -= 1;
        assert!(orchestrator.aggregate_endpoints(&altered).is_err());
        let mut altered = report.replicates.clone();
        altered[0].report.branch_reports[1]
            .terminal_sample
            .total_energy = f32::NAN;
        assert!(orchestrator.aggregate_endpoints(&altered).is_err());
        let mut altered = report.replicates.clone();
        altered[0].report.branch_reports[1].bundle_verified = false;
        assert!(orchestrator.aggregate_endpoints(&altered).is_err());

        let retained = fs::read(orchestrator.output_dir.join("replicated_report.json")).unwrap();
        assert!(matches!(
            orchestrator.execute_study(None),
            Err(BranchStudyError::OutputAlreadyExists { .. })
        ));
        assert_eq!(
            fs::read(orchestrator.output_dir.join("replicated_report.json")).unwrap(),
            retained
        );
        let singleton = ReplicatedBranchStudyOrchestrator::new(
            ReplicatedBranchStudyPlan {
                seeds: vec![42],
                ..plan
            },
            directory.path().join("not-executed"),
        );
        let effects = singleton
            .aggregate_endpoints(&report.replicates[..1])
            .unwrap();
        for effect in effects["resource_shock"].values() {
            assert_eq!(effect.n_samples, 1);
            assert!(effect.ci_95.is_none() && effect.p_value.is_none());
        }
    }

    #[test]
    fn replicated_study_cancellation_and_partial_destination_never_publish_aggregate_success() {
        let directory = tempdir().unwrap();
        let output = directory.path().join("cancelled");
        let plan = replicated_fixture();
        let orchestrator = ReplicatedBranchStudyOrchestrator::new(plan.clone(), output.clone());
        assert!(matches!(
            orchestrator.execute_study(Some(&AtomicBool::new(true))),
            Err(BranchStudyError::Cancelled)
        ));
        assert!(!output.exists());

        // A retained first seed from an interrupted/partial study is never silently resumed or
        // promoted to complete inference. Its exact outputs survive the refusal.
        let first = BranchStudyOrchestrator::new(
            orchestrator.replicate_plan(42),
            output.join("seed_000000000000002a"),
        );
        first.execute_study(None).unwrap();
        let retained = fs::read(first.output_dir.join("study_report.json")).unwrap();
        assert!(matches!(
            orchestrator.execute_study(None),
            Err(BranchStudyError::OutputAlreadyExists { .. })
        ));
        assert_eq!(
            fs::read(first.output_dir.join("study_report.json")).unwrap(),
            retained
        );
        assert!(!output.join("replicated_report.json").exists());
    }

    #[test]
    fn test_plan_validation_rejects_empty_study_id_and_missing_control() {
        let mut plan = BranchStudyPlan::standard_five_branch("test_val", 42, 10, 10);

        // Empty study_id
        plan.study_id = "".to_string();
        let orchestrator = BranchStudyOrchestrator::new(plan.clone(), PathBuf::from("/tmp"));
        assert!(matches!(
            orchestrator.validate(),
            Err(BranchStudyError::InvalidPlan(_))
        ));

        // Missing control
        plan.study_id = "test_val".to_string();
        plan.branches.retain(|b| b.intervention.is_some());
        let orchestrator2 = BranchStudyOrchestrator::new(plan, PathBuf::from("/tmp"));
        assert!(matches!(
            orchestrator2.validate(),
            Err(BranchStudyError::MissingControlBranch)
        ));
    }

    #[test]
    fn study_preflight_enforces_work_and_grid_limits_before_creating_outputs() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("never-created");
        let base = BranchStudyPlan::standard_five_branch("bounded-study", 42, 2, 3);
        let mut too_many_branches = base.clone();
        too_many_branches.branches = (0..=MAX_STUDY_BRANCHES)
            .map(|index| StudyBranchSpec {
                branch_id: format!("control-{index}"),
                ..base.branches[0].clone()
            })
            .collect();
        let mut too_long = base.clone();
        too_long.horizon_ticks = MAX_STUDY_HORIZON_TICKS + 1;
        let mut too_much_work = base.clone();
        too_much_work.checkpoint_tick = MAX_STUDY_TOTAL_TRANSITIONS;
        let mut too_large_grid = base.clone();
        too_large_grid.config.world_width = 513;
        too_large_grid.config.world_height = 512;
        too_large_grid.config.food_cell_size = 1;
        for (plan, resource) in [
            (too_many_branches, "branch count"),
            (too_long, "horizon ticks"),
            (too_much_work, "total transitions"),
            (too_large_grid, "grid cells"),
        ] {
            assert!(matches!(
                BranchStudyOrchestrator::new(plan, output.clone()).execute_study(None),
                Err(BranchStudyError::CapacityExceeded { resource: actual, .. }) if actual == resource
            ));
            assert!(
                !output.exists(),
                "preflight must not reserve output on refusal"
            );
        }

        // Boundary admission only: do not allocate or simulate this large valid plan.
        let mut at_limits = base;
        at_limits.branches = (0..MAX_STUDY_BRANCHES)
            .map(|index| StudyBranchSpec {
                branch_id: format!("control-{index}"),
                description: "Unperturbed control".to_owned(),
                intervention: None,
                scheduled_tick: None,
            })
            .collect();
        at_limits.horizon_ticks = MAX_STUDY_HORIZON_TICKS;
        at_limits.checkpoint_tick =
            MAX_STUDY_TOTAL_TRANSITIONS - MAX_STUDY_BRANCHES as u64 * MAX_STUDY_HORIZON_TICKS;
        at_limits.config.world_width = 512;
        at_limits.config.world_height = 512;
        at_limits.config.food_cell_size = 1;
        BranchStudyOrchestrator::new(at_limits, output.clone())
            .validate()
            .expect("exact supported limits admitted without constructing a world");
        assert!(!output.exists());
    }

    #[test]
    fn study_preflight_rejects_invalid_config_patches_and_unsupported_brains() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("never-created");
        let base = BranchStudyPlan::standard_five_branch("invalid-input", 42, 2, 3);
        let mut invalid_config = base.clone();
        invalid_config.config.food_cell_size = 0;
        assert!(matches!(
            BranchStudyOrchestrator::new(invalid_config, output.clone()).execute_study(None),
            Err(BranchStudyError::WorldState(_))
        ));
        #[cfg(not(feature = "neuro"))]
        {
            let mut unsupported = base.clone();
            unsupported.brain_preset = BrainPreset::Neuro;
            assert!(matches!(
                BranchStudyOrchestrator::new(unsupported, output.clone()).execute_study(None),
                Err(BranchStudyError::Registry(_))
            ));
        }
        for patch in [
            serde_json::json!({"food_cell_size": 0}),
            serde_json::json!({"world_width": 400}),
            serde_json::json!({"persistence_interval": 1}),
            serde_json::json!({"food_growht_rate": 0.25}),
        ] {
            let mut invalid = base.clone();
            invalid.branches[1].intervention = Some(StudyIntervention::ConfigPatch(patch));
            assert!(matches!(
                BranchStudyOrchestrator::new(invalid, output.clone()).execute_study(None),
                Err(BranchStudyError::InvalidIntervention { .. })
            ));
        }
        assert!(!output.exists());
    }

    #[test]
    fn json_study_plans_reject_misspelled_fields_instead_of_ignoring_them() {
        let plan = BranchStudyPlan::standard_five_branch("strict-json", 42, 2, 3);
        let value = serde_json::to_value(&plan).unwrap();
        assert!(value.get("source_checkpoint").is_none());
        assert_eq!(
            serde_json::from_value::<BranchStudyPlan>(value.clone()).unwrap(),
            plan
        );
        let mut bad_plan = value.clone();
        bad_plan["horizion_ticks"] = serde_json::json!(30);
        let mut bad_branch = value.clone();
        bad_branch["branches"][1]["scheduld_tick"] = serde_json::json!(3);
        let mut bad_intervention = value.clone();
        bad_intervention["branches"][1]["intervention"]["params"]["growth_sacle"] =
            serde_json::json!(0.5);
        let mut bad_cooperation = value;
        bad_cooperation["branches"][5]["intervention"]["params"]["food_sharing_distnace"] =
            serde_json::json!(10.0);
        for malformed in [bad_plan, bad_branch, bad_intervention, bad_cooperation] {
            assert!(serde_json::from_value::<BranchStudyPlan>(malformed).is_err());
        }
    }

    #[test]
    fn scalar_allocation_and_injection_limits_apply_to_base_and_patched_configs() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("never-created");
        let base = BranchStudyPlan::standard_five_branch("bounded-config", 42, 2, 3);
        for (patch, resource) in [
            (
                serde_json::json!({"history_capacity": usize::MAX}),
                "history entries",
            ),
            (
                serde_json::json!({"population_minimum": usize::MAX}),
                "population floor",
            ),
            (
                serde_json::json!({"population_spawn_count": u32::MAX}),
                "population injection count",
            ),
            (
                serde_json::json!({"archive_enabled": true, "archive_max_bytes": usize::MAX}),
                "archive bytes",
            ),
        ] {
            let mut invalid_base = base.clone();
            invalid_base.config = resolve_scheduled_config_patch(&base.config, &patch).unwrap();
            let mut invalid_patch = base.clone();
            invalid_patch.branches[1].intervention = Some(StudyIntervention::ConfigPatch(patch));
            for plan in [invalid_base, invalid_patch] {
                assert!(matches!(
                    BranchStudyOrchestrator::new(plan, output.clone()).execute_study(None),
                    Err(BranchStudyError::CapacityExceeded { resource: actual, .. }) if actual == resource
                ));
                assert!(!output.exists());
            }
        }
    }

    #[test]
    fn scheduled_boundaries_must_have_an_observed_next_transition_without_tick_overflow() {
        let plan = BranchStudyPlan::standard_five_branch("schedule_bounds", 42, 6, 6);
        for due_tick in [5, 12, 13, u64::MAX] {
            let mut invalid = plan.clone();
            invalid.branches[1].scheduled_tick = Some(due_tick);
            let orchestrator = BranchStudyOrchestrator::new(invalid, PathBuf::from("unused"));
            assert!(
                matches!(
                    orchestrator.validate(),
                    Err(BranchStudyError::InvalidPlan(_))
                ),
                "invalid due boundary {due_tick} was admitted"
            );
        }
        for due_tick in [6, 11] {
            let mut valid = plan.clone();
            valid.branches[1].scheduled_tick = Some(due_tick);
            BranchStudyOrchestrator::new(valid, PathBuf::from("unused"))
                .validate()
                .expect("both the first and last observable command boundaries are allowed");
        }

        let mut scheduled_control = plan.clone();
        scheduled_control.branches[0].scheduled_tick = Some(6);
        assert!(matches!(
            BranchStudyOrchestrator::new(scheduled_control, PathBuf::from("unused")).validate(),
            Err(BranchStudyError::InvalidPlan(_))
        ));
        for (checkpoint_tick, horizon_ticks) in [(u64::MAX, 1), (u64::MAX - 2, 3)] {
            let mut overflowing = plan.clone();
            overflowing.checkpoint_tick = checkpoint_tick;
            overflowing.horizon_ticks = horizon_ticks;
            assert!(matches!(
                BranchStudyOrchestrator::new(overflowing, PathBuf::from("unused")).validate(),
                Err(BranchStudyError::InvalidPlan(_))
            ));
        }
    }

    #[test]
    fn independent_pair_inference_is_explicit_and_refuses_incomplete_or_nonfinite_pairs() {
        let effect = compute_paired_effect(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], "energy", 42)
            .expect("three independent matched replicates");
        assert_eq!(
            effect.analysis_basis,
            BranchAnalysisBasis::IndependentPairedReplicates { pairs: 3 }
        );
        assert_eq!(effect.n_samples, 3);
        assert_eq!(effect.mean_diff, 2.0);
        assert_eq!(effect.sd_diff, 1.0);
        assert_eq!(effect.cohens_d, Some(2.0));
        let (low, high) = effect.ci_95.expect("replicated bootstrap interval");
        assert!(low <= 2.0 && high >= 2.0);
        assert!((0.0..=1.0).contains(&effect.p_value.expect("replicated sign-flip test")));

        let singleton = compute_paired_effect(&[1.0], &[2.0], "energy", 42).unwrap();
        assert_eq!(singleton.mean_diff, 1.0);
        assert!(singleton.ci_95.is_none());
        assert!(singleton.p_value.is_none());
        for (control, branch) in [
            (vec![1.0, 2.0], vec![3.0]),
            (vec![1.0], vec![f64::NAN]),
            (vec![f64::INFINITY], vec![1.0]),
            (vec![-f64::MAX], vec![f64::MAX]),
        ] {
            assert!(matches!(
                compute_paired_effect(&control, &branch, "energy", 42),
                Err(BranchStudyError::InvalidComparison(_))
            ));
        }
    }

    #[test]
    fn delayed_intervention_preserves_state_until_due_and_reports_same_tick_lane_hashes() {
        let dir = tempdir().expect("tempdir");
        let mut plan = BranchStudyPlan::standard_five_branch("delayed_evidence", 54321, 2, 5);
        plan.config.population_minimum = 0;
        plan.config.population_spawn_interval = 1;
        plan.config.population_spawn_count = 2;
        plan.config.population_crossover_chance = 0.0;
        plan.config.reproduction_attempt_chance = 0.0;
        plan.branches.truncate(1);
        plan.branches.push(StudyBranchSpec {
            branch_id: "delayed_scorch".to_string(),
            description: "Remove half of ground food after completed tick four".to_string(),
            intervention: Some(StudyIntervention::ResourceShock {
                region: None,
                bloom_amount: 0.0,
                scorch: 0.5,
            }),
            scheduled_tick: Some(4),
        });
        let mut world = world_at_checkpoint(&plan);
        let agents = world.agents().iter_handles().collect::<Vec<_>>();
        for id in agents {
            assert!(
                world
                    .try_update_agent(id, |data, runtime| {
                        data.health = 0.75;
                        runtime.energy = 0.75;
                    })
                    .expect("edit observable checkpoint state")
            );
        }
        let checkpoint = world.checkpoint_v1().expect("capture real checkpoint");
        let source_digest = &checkpoint.source_digest().overall;
        let orchestrator = BranchStudyOrchestrator::new(plan, dir.path().to_path_buf());
        orchestrator.validate().expect("valid delayed study");
        let (control, _) = orchestrator
            .run_branch(
                &orchestrator.plan.branches[0],
                &checkpoint,
                source_digest,
                &BranchStudySource::SeededWarmup,
                None,
                None,
            )
            .expect("run control");
        let (branch, report) = orchestrator
            .run_branch(
                &orchestrator.plan.branches[1],
                &checkpoint,
                source_digest,
                &BranchStudySource::SeededWarmup,
                Some(&control),
                None,
            )
            .expect("run delayed branch");

        assert_eq!(
            control[0].sample.total_energy,
            0.75 * world.agent_count() as f32
        );
        assert_eq!(control[0].sample.average_health, 0.75);
        assert_eq!(branch[0].sample, control[0].sample);
        for index in 0..=2 {
            assert_eq!(
                control[index].digest, branch[index].digest,
                "completed tick {} must still match before the delayed command",
                control[index].sample.tick
            );
        }
        assert_eq!(report.command_applied_tick, Some(4));
        let divergence = report
            .first_divergence
            .as_ref()
            .expect("scorch changes food");
        assert_eq!(divergence.tick, 5);
        assert_eq!(divergence.command_applied_tick, 4);
        assert_eq!(divergence.stage, "food");
        assert_eq!(divergence.control_digest, control[3].digest.food);
        assert_eq!(divergence.branch_digest, branch[3].digest.food);
        assert_ne!(
            divergence.control_digest,
            checkpoint.source_digest().food,
            "the control lane must be observed at divergence, not at the checkpoint"
        );
        assert_ne!(divergence.control_digest, divergence.branch_digest);

        assert!(
            branch[2].sample.population > branch[0].sample.population,
            "newcomers must make the delayed baseline observably different from the checkpoint"
        );
        assert_eq!(report.recovery.baseline_tick, 4);
        assert_eq!(
            report.recovery.pre_intervention_population,
            branch[2].sample.population
        );
        assert_eq!(report.recovery.trough_tick, 4);
        assert_eq!(
            report.recovery.trough_population,
            branch[2].sample.population
        );
        assert_eq!(
            report.extinction.min_population,
            branch[2].sample.population
        );
        for effect in report.effect_sizes.values() {
            assert_eq!(
                effect.analysis_basis,
                BranchAnalysisBasis::SingleSeedTrajectory { seed: 54321 }
            );
            assert_eq!(effect.n_samples, 6);
            assert!(effect.ci_95.is_none());
            assert!(effect.p_value.is_none());
            assert!(effect.cohens_d.is_none());
            assert!(effect.hedges_g.is_none());
        }

        // A missing or shifted control observation must fail instead of dropping a comparison.
        assert!(matches!(
            orchestrator.run_branch(
                &orchestrator.plan.branches[1],
                &checkpoint,
                source_digest,
                &BranchStudySource::SeededWarmup,
                Some(&control[..control.len() - 1]),
                None,
            ),
            Err(BranchStudyError::InvalidComparison(_))
        ));
        let mut misaligned = control.clone();
        misaligned[3].sample.tick += 1;
        assert!(matches!(
            orchestrator.run_branch(
                &orchestrator.plan.branches[1],
                &checkpoint,
                source_digest,
                &BranchStudySource::SeededWarmup,
                Some(&misaligned),
                None,
            ),
            Err(BranchStudyError::InvalidComparison(_))
        ));
    }

    #[test]
    fn test_invalid_intervention_parameters_rejected() {
        let invalid_drought = StudyIntervention::Drought {
            region: None,
            duration_ticks: 0, // invalid
            growth_scale: 0.0,
        };
        assert!(invalid_drought.validate(1600, 900).is_err());

        let invalid_shock = StudyIntervention::ResourceShock {
            region: None,
            bloom_amount: -1.0, // invalid
            scorch: 0.5,
        };
        assert!(invalid_shock.validate(1600, 900).is_err());

        let invalid_temp = StudyIntervention::TemperatureShift {
            discomfort_rate: -0.5, // invalid
            comfort_band: 0.1,
        };
        assert!(invalid_temp.validate(1600, 900).is_err());

        let invalid_coop = StudyIntervention::Cooperation(CooperationParams {
            food_sharing_rate: 1.5, // invalid (> 1.0)
            food_transfer_rate: 0.5,
            food_sharing_distance: 20.0,
            reproduction_partner_chance: None,
        });
        assert!(invalid_coop.validate(1600, 900).is_err());
    }

    #[test]
    fn test_pre_intervention_digest_and_tick_identity() {
        let dir = tempdir().expect("tempdir");
        let mut plan = BranchStudyPlan::standard_five_branch("test_pre_digest", 12345, 5, 5);
        plan.branches = vec![
            StudyBranchSpec {
                branch_id: "control".to_string(),
                description: "Control".to_string(),
                intervention: None,
                scheduled_tick: None,
            },
            StudyBranchSpec {
                branch_id: "control_clone".to_string(),
                description: "Second control to verify identical trajectory".to_string(),
                intervention: None,
                scheduled_tick: None,
            },
        ];

        let orchestrator = BranchStudyOrchestrator::new(plan, dir.path().to_path_buf());
        let report = orchestrator
            .execute_study(None)
            .expect("study execution succeeds");

        assert_eq!(report.checkpoint_tick, 5);
        assert!(report.pre_intervention_digest_verified_for_all_branches);
        assert_eq!(report.branch_reports.len(), 2);
        assert_eq!(
            report.branch_reports[0].final_digest, report.branch_reports[1].final_digest,
            "two unperturbed branches restored from the same checkpoint must produce bit-identical final digests"
        );
    }

    #[test]
    fn test_command_timing_and_first_divergence_tracing() {
        let dir = tempdir().expect("tempdir");
        let mut plan = BranchStudyPlan::standard_five_branch("test_timing_div", 54321, 6, 6);
        plan.branches = vec![
            StudyBranchSpec {
                branch_id: "control".to_string(),
                description: "Control".to_string(),
                intervention: None,
                scheduled_tick: None,
            },
            StudyBranchSpec {
                branch_id: "drought_immediate".to_string(),
                description: "Halt food regrowth immediately".to_string(),
                intervention: Some(StudyIntervention::Drought {
                    region: None,
                    duration_ticks: 20,
                    growth_scale: 0.0,
                }),
                scheduled_tick: Some(6),
            },
        ];

        let orchestrator = BranchStudyOrchestrator::new(plan, dir.path().to_path_buf());
        let report = orchestrator
            .execute_study(None)
            .expect("study execution succeeds");

        let drought_report = report
            .branch_reports
            .iter()
            .find(|b| b.branch_id == "drought_immediate")
            .expect("drought report present");

        assert_eq!(drought_report.command_applied_tick, Some(6));
        let divergence = drought_report
            .first_divergence
            .as_ref()
            .expect("divergence must be detected");
        assert_eq!(
            divergence.tick, 7,
            "divergence must first manifest on the first step following the intervention at tick 6"
        );
        assert!(
            divergence.stage == "effects"
                || divergence.stage == "food"
                || divergence.stage == "overall",
            "divergence stage must name the affected lane (got {})",
            divergence.stage
        );
    }

    #[test]
    fn test_bundle_creation_and_verification() {
        let dir = tempdir().expect("tempdir");
        let mut plan = BranchStudyPlan::standard_five_branch("test_bundle_verif", 99999, 4, 4);
        plan.branches = vec![
            StudyBranchSpec {
                branch_id: "control".to_string(),
                description: "Control".to_string(),
                intervention: None,
                scheduled_tick: None,
            },
            StudyBranchSpec {
                branch_id: "bloom".to_string(),
                description: "Food bloom shock".to_string(),
                intervention: Some(StudyIntervention::ResourceShock {
                    region: None,
                    bloom_amount: 5.0,
                    scorch: 0.0,
                }),
                scheduled_tick: None,
            },
        ];

        let orchestrator = BranchStudyOrchestrator::new(plan, dir.path().to_path_buf());
        let report = orchestrator
            .execute_study(None)
            .expect("study execution succeeds");

        let mut retained = BTreeMap::new();
        for name in ["study_plan.json", "study_report.json", "study_summary.md"] {
            let path = dir.path().join(name);
            retained.insert(path.clone(), fs::read(path).expect("study artifact"));
        }
        for branch_rep in &report.branch_reports {
            assert!(
                branch_rep.bundle_verified,
                "bundle for `{}` must pass verification",
                branch_rep.branch_id
            );
            let bundle_path_str = branch_rep
                .bundle_path
                .as_ref()
                .expect("bundle path present");
            let bundle_path = PathBuf::from(bundle_path_str);
            assert!(bundle_path.exists());
            assert!(bundle_path.join("bundle_manifest.json").exists());
            assert!(bundle_path.join("exports/summary.csv").exists());
            assert!(bundle_path.join("evidence/branch_evidence.json").exists());
            for name in [
                "bundle_manifest.json",
                "exports/summary.csv",
                "evidence/branch_evidence.json",
            ] {
                let path = bundle_path.join(name);
                retained.insert(path.clone(), fs::read(path).expect("branch artifact"));
            }
        }

        assert!(matches!(
            orchestrator.execute_study(None),
            Err(BranchStudyError::OutputAlreadyExists { .. })
        ));
        for (path, original) in retained {
            assert_eq!(fs::read(&path).expect("retained output"), original);
        }
    }

    #[test]
    fn occupied_or_interrupted_study_outputs_are_preserved_before_execution() {
        let dir = tempdir().expect("tempdir");
        let bundle = dir.path().join("bundle_control");
        fs::create_dir(&bundle).expect("prior partial bundle");
        let evidence = bundle.join("prior-evidence.bin");
        fs::write(&evidence, b"irreplaceable interrupted study").expect("prior evidence");
        let plan = BranchStudyPlan::standard_five_branch("occupied", 42, 1, 1);
        let orchestrator = BranchStudyOrchestrator::new(plan, dir.path().to_path_buf());

        assert!(matches!(
            orchestrator.execute_study(None),
            Err(BranchStudyError::OutputAlreadyExists { .. })
        ));
        assert_eq!(
            fs::read(evidence).expect("preserved evidence"),
            b"irreplaceable interrupted study"
        );
        assert!(!dir.path().join("study_plan.json").exists());
        assert!(!dir.path().join("study_report.json").exists());
    }

    #[test]
    fn concurrent_studies_cannot_claim_the_same_empty_output() {
        let dir = tempdir().expect("tempdir");
        let plan = BranchStudyPlan::standard_five_branch("concurrent", 42, 1, 1);
        let orchestrator = BranchStudyOrchestrator::new(plan.clone(), dir.path().to_path_buf());
        let barrier = std::sync::Barrier::new(2);
        let outcomes = std::thread::scope(|scope| {
            let attempt = || {
                barrier.wait();
                orchestrator.reserve_output_directory()
            };
            let first = scope.spawn(attempt);
            let second = scope.spawn(attempt);
            [
                first.join().expect("first writer"),
                second.join().expect("second writer"),
            ]
        });
        assert_eq!(outcomes.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            outcomes
                .iter()
                .filter(|result| matches!(
                    result,
                    Err(BranchStudyError::OutputAlreadyExists { .. })
                ))
                .count(),
            1
        );
        let retained: BranchStudyPlan = serde_json::from_slice(
            &fs::read(dir.path().join("study_plan.json")).expect("winning plan"),
        )
        .expect("complete retained plan");
        assert_eq!(retained, plan);
    }

    #[cfg(unix)]
    #[test]
    fn study_output_symlink_is_refused_without_writing_the_target() {
        let dir = tempdir().expect("tempdir");
        let target = dir.path().join("target");
        fs::create_dir(&target).expect("empty target");
        let alias = dir.path().join("alias");
        std::os::unix::fs::symlink(&target, &alias).expect("output alias");
        let plan = BranchStudyPlan::standard_five_branch("symlink", 42, 1, 1);
        let orchestrator = BranchStudyOrchestrator::new(plan, alias);
        assert!(matches!(
            orchestrator.execute_study(None),
            Err(BranchStudyError::OutputAlreadyExists { .. })
        ));
        assert!(
            fs::read_dir(target)
                .expect("target retained")
                .next()
                .is_none()
        );
    }

    #[test]
    fn test_cancellation_stops_cleanly() {
        let dir = tempdir().expect("tempdir");
        let plan = BranchStudyPlan::standard_five_branch("test_cancel", 777, 10, 10);
        let orchestrator = BranchStudyOrchestrator::new(plan, dir.path().to_path_buf());

        let cancel_flag = AtomicBool::new(true); // already cancelled
        let result = orchestrator.execute_study(Some(&cancel_flag));
        assert!(matches!(result, Err(BranchStudyError::Cancelled)));
        assert!(
            fs::read_dir(dir.path())
                .expect("empty output")
                .next()
                .is_none()
        );
    }

    #[test]
    fn test_extinction_and_recovery_metrics_computation() {
        let dir = tempdir().expect("tempdir");
        let mut plan = BranchStudyPlan::standard_five_branch("test_metrics", u64::MAX, 5, 10);
        plan.branches = vec![
            StudyBranchSpec {
                branch_id: "control".to_string(),
                description: "Control".to_string(),
                intervention: None,
                scheduled_tick: None,
            },
            StudyBranchSpec {
                branch_id: "bloom".to_string(),
                description: "Food bloom".to_string(),
                intervention: Some(StudyIntervention::ResourceShock {
                    region: None,
                    bloom_amount: 4.0,
                    scorch: 0.0,
                }),
                scheduled_tick: None,
            },
        ];

        let orchestrator = BranchStudyOrchestrator::new(plan, dir.path().to_path_buf());
        let report = orchestrator
            .execute_study(None)
            .expect("study execution succeeds");

        let bloom_rep = report
            .branch_reports
            .iter()
            .find(|b| b.branch_id == "bloom")
            .expect("bloom report");

        assert!(bloom_rep.recovery.recovery_ratio > 0.0);
        assert!(!bloom_rep.extinction.extinct);
        assert!(bloom_rep.effect_sizes.contains_key("food_total"));
        let food_eff = &bloom_rep.effect_sizes["food_total"];
        assert!(
            food_eff.mean_diff > 0.0,
            "bloom must increase mean food compared to control"
        );
    }
}
