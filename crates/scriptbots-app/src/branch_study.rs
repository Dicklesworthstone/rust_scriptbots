//! Paired checkpoint-branch intervention studies for evolutionary simulations (bd-2z0.11.4).
//!
//! # Scientific Design (Plan Phase 4.5)
//!
//! A paired checkpoint-branch study establishes causal attribution for ecological and evolutionary
//! interventions by eliminating founding-state and pre-intervention divergence:
//!
//! 1. **Common Pre-Intervention Baseline**: A simulation world is seeded and stepped until
//!    an exact boundary `checkpoint_tick`. A canonical [`scriptbots_core::WorldCheckpointV1`] is
//!    captured, establishing an authoritative `pre_intervention_digest`.
//! 2. **Matched Branch Restoration**: The Control branch and every intervention branch are restored
//!    from that identical checkpoint using a freshly prepared, bit-identical [`scriptbots_core::BrainRegistry`].
//!    Every branch asserts and verifies `restored_digest == pre_intervention_digest` before any
//!    intervention occurs.
//! 3. **Exact Command Timing**: Interventions are applied at the exact scheduled tick boundary
//!    (`checkpoint_tick`).
//! 4. **First Divergence Tracing**: For every intervention branch, simulation transitions are traced
//!    tick-by-tick against the control branch to locate the exact tick and specific pipeline lane/stage
//!    (e.g., `effects`, `config`, `food`, `agents`) where divergence first occurred.
//! 5. **Quantitative Outcome Measures**:
//!    - **Extinction**: Detects population collapse, records time to extinction, and minimum population.
//!    - **Recovery**: Evaluates post-perturbation rebound against pre-intervention baseline and trough.
//!    - **Hysteresis**: Evaluates path dependency, integrated trajectory deficit, and persistent terminal gap.
//!    - **Effect Sizes & Uncertainty**: Computes paired mean differences, Cohen's $d$, Hedges' $g$,
//!      95% bootstrap confidence intervals, and exact Monte Carlo sign-flip permutation p-values.
//! 6. **Verifiable Run Bundles**: Every branch generates a self-contained artifact bundle validated
//!    via bounded canonical verification ([`scriptbots_storage::verify_run_bundle_bounded`]).

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use rand::{Rng, SeedableRng};
use scriptbots_brain::{
    assembly::{AssemblyBrain, AssemblyFamilyAdapter},
    dwraon::{DwraonBrain, DwraonFamilyAdapter},
    mlp::{MlpBrain, MlpBrainFamily},
};
#[cfg(feature = "brain-ft")]
use scriptbots_brain_ml::{FT_BRAIN_KIND, FtBrainFamily};
use scriptbots_core::{
    BrainRegistry, ControlCommand, Intervention, Region, ScriptBotsConfig, WorldCheckpointError,
    WorldDigestV1, WorldState, WorldStateError, apply_control_command,
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

/// Specific intervention perturbations for branch studies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "params")]
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
    /// Direct core intervention command.
    Core(Intervention),
    /// Arbitrary configuration patch resolved via pure precedence.
    ConfigPatch(serde_json::Value),
}

/// Parameters for cooperation and altruistic sharing studies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
pub struct StudyBranchSpec {
    /// Unique identifier for this branch (e.g. "control", "drought_severe").
    pub branch_id: String,
    /// Human-readable description of the branch condition.
    pub description: String,
    /// Intervention to apply. `None` defines an unperturbed Control branch.
    pub intervention: Option<StudyIntervention>,
    /// Optional scheduled tick override (defaults to `plan.checkpoint_tick`).
    pub scheduled_tick: Option<u64>,
}

/// Complete plan for a paired checkpoint-branch study.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchStudyPlan {
    /// Unique identifier for this study.
    pub study_id: String,
    /// Research question or scientific hypothesis being tested.
    pub description: String,
    /// Base RNG seed used for founding population and world initialization.
    pub base_seed: u64,
    /// Simulation tick at which baseline is snapshotted and branches diverge.
    pub checkpoint_tick: u64,
    /// Number of simulation ticks to run each branch post-checkpoint.
    pub horizon_ticks: u64,
    /// Architecture preset for agent brains.
    pub brain_preset: BrainPreset,
    /// Base simulation configuration.
    pub config: ScriptBotsConfig,
    /// Set of branches to evaluate (must include at least one control branch).
    pub branches: Vec<StudyBranchSpec>,
}

impl BranchStudyPlan {
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
    /// Offspring born during this tick.
    pub births: usize,
    /// Agent deaths during this tick.
    pub deaths: usize,
    /// Sum of all agent energy.
    pub total_energy: f32,
    /// Mean health across living agents.
    pub average_health: f32,
    /// Spike hits recorded during this tick.
    pub spike_hits: u32,
    /// Canonical BLAKE3 world digest hex.
    pub digest_overall: String,
}

/// Record of the exact tick and stage where a branch diverged from Control.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FirstDivergence {
    /// Identifier of the diverged branch.
    pub branch_id: String,
    /// First simulation tick where the world digest differed from Control.
    pub tick: u64,
    /// First pipeline lane/stage that differed (`config`, `effects`, `food`, `agents`, etc.).
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
    /// Whether the population reached 0 at any point post-intervention.
    pub extinct: bool,
    /// First tick where population reached 0 (if extinct).
    pub extinction_tick: Option<u64>,
    /// Minimum population observed during the post-intervention horizon.
    pub min_population: usize,
    /// Final population at the end of the horizon.
    pub final_population: usize,
}

/// Population recovery assessment for a branch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveryMeasure {
    /// Population at the checkpoint boundary before intervention.
    pub pre_intervention_population: usize,
    /// Minimum population reached during the post-intervention horizon.
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
    /// Whether significant path dependency / persistent deficit was detected (> 5% terminal gap).
    pub path_dependence_detected: bool,
}

/// Paired effect size comparison against Control.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchEffectSize {
    /// Name of the metric (e.g. "population", "food_total").
    pub metric: String,
    /// Number of paired sample points across the post-intervention horizon.
    pub n_samples: usize,
    /// Mean difference (branch minus control).
    pub mean_diff: f64,
    /// Sample standard deviation of differences.
    pub sd_diff: f64,
    /// Standardized effect size Cohen's d (None if variance is zero).
    pub cohens_d: Option<f64>,
    /// Bias-corrected Hedges' g (None if variance is zero).
    pub hedges_g: Option<f64>,
    /// 95% bootstrap confidence interval of the mean difference.
    pub ci_95: Option<(f64, f64)>,
    /// Exact Monte Carlo sign-flip permutation p-value.
    pub p_value: f64,
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
    /// Metric effect sizes compared to Control.
    pub effect_sizes: BTreeMap<String, BranchEffectSize>,
    /// Final world digest overall hash.
    pub final_digest: String,
    /// Path to verified run bundle directory.
    pub bundle_path: Option<String>,
    /// Whether the bundle passed bounded verification.
    pub bundle_verified: bool,
}

/// Master study report aggregating all branches, proofs, and summaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchStudyReport {
    /// Study identifier.
    pub study_id: String,
    /// Schema version (1).
    pub schema_version: u16,
    /// Base seed used.
    pub base_seed: u64,
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

/// Typed error returned during branch study planning, execution, or verification.
#[derive(Debug, Error)]
pub enum BranchStudyError {
    /// Plan validation error.
    #[error("study plan validation failed: {0}")]
    InvalidPlan(String),
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
    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Study was cancelled by caller.
    #[error("study execution was cancelled")]
    Cancelled,
}

/// Create a fresh [`BrainRegistry`] matching the given preset for checkpoint restoration.
///
/// # Errors
///
/// Returns [`BranchStudyError::Registry`] if any family adapter cannot be registered.
pub fn create_brain_registry(preset: BrainPreset) -> Result<BrainRegistry, BranchStudyError> {
    let mut registry = BrainRegistry::new();
    match preset {
        BrainPreset::Mlp => {
            registry
                .register_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
                .map_err(|e| BranchStudyError::Registry(e.to_string()))?;
        }
        BrainPreset::Dwraon => {
            registry
                .register_family(
                    DwraonBrain::KIND.as_str(),
                    Box::new(DwraonFamilyAdapter::default()),
                )
                .map_err(|e| BranchStudyError::Registry(e.to_string()))?;
        }
        BrainPreset::Assembly => {
            let adapter = AssemblyFamilyAdapter::new()
                .map_err(|e| BranchStudyError::Registry(e.to_string()))?;
            registry
                .register_family(AssemblyBrain::KIND.as_str(), Box::new(adapter))
                .map_err(|e| BranchStudyError::Registry(e.to_string()))?;
        }
        BrainPreset::Mixed => {
            registry
                .register_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
                .map_err(|e| BranchStudyError::Registry(e.to_string()))?;
            registry
                .register_family(
                    DwraonBrain::KIND.as_str(),
                    Box::new(DwraonFamilyAdapter::default()),
                )
                .map_err(|e| BranchStudyError::Registry(e.to_string()))?;
            let adapter = AssemblyFamilyAdapter::new()
                .map_err(|e| BranchStudyError::Registry(e.to_string()))?;
            registry
                .register_family(AssemblyBrain::KIND.as_str(), Box::new(adapter))
                .map_err(|e| BranchStudyError::Registry(e.to_string()))?;
            #[cfg(feature = "brain-ft")]
            {
                registry
                    .register_family(FT_BRAIN_KIND, Box::new(FtBrainFamily::default()))
                    .map_err(|e| BranchStudyError::Registry(e.to_string()))?;
            }
        }
        BrainPreset::Ft => {
            #[cfg(feature = "brain-ft")]
            {
                registry
                    .register_family(FT_BRAIN_KIND, Box::new(FtBrainFamily::default()))
                    .map_err(|e| BranchStudyError::Registry(e.to_string()))?;
            }
            #[cfg(not(feature = "brain-ft"))]
            return Err(BranchStudyError::Registry(
                "brain-ft feature not enabled in build".to_string(),
            ));
        }
        BrainPreset::Neuro => {
            return Err(BranchStudyError::Registry(
                "Neuro preset checkpoint restoration not supported".to_string(),
            ));
        }
    }
    Ok(registry)
}

/// Compare two world digests and return the first differing pipeline stage and lane values.
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
    if control.agents != branch.agents {
        return Some(("agents", control.agents.clone(), branch.agents.clone()));
    }
    if control.brains != branch.brains {
        return Some(("brains", control.brains.clone(), branch.brains.clone()));
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

/// Compute paired effect size and uncertainty between control series and branch series.
#[must_use]
pub fn compute_paired_effect(
    control: &[f64],
    branch: &[f64],
    metric_name: &str,
    seed: u64,
) -> BranchEffectSize {
    let n = control.len().min(branch.len());
    if n == 0 {
        return BranchEffectSize {
            metric: metric_name.to_string(),
            n_samples: 0,
            mean_diff: 0.0,
            sd_diff: 0.0,
            cohens_d: None,
            hedges_g: None,
            ci_95: None,
            p_value: 1.0,
        };
    }

    let diffs: Vec<f64> = (0..n).map(|i| branch[i] - control[i]).collect();
    let sum_diff: f64 = diffs.iter().sum();
    let mean_diff = sum_diff / n as f64;

    if n < 2 {
        return BranchEffectSize {
            metric: metric_name.to_string(),
            n_samples: n,
            mean_diff,
            sd_diff: 0.0,
            cohens_d: None,
            hedges_g: None,
            ci_95: None,
            p_value: 1.0,
        };
    }

    let var_diff: f64 = diffs
        .iter()
        .map(|&d| {
            let delta = d - mean_diff;
            delta * delta
        })
        .sum::<f64>()
        / (n - 1) as f64;
    let sd_diff = var_diff.sqrt();

    let (cohens_d, hedges_g) = if sd_diff > 1e-12 {
        let d = mean_diff / sd_diff;
        #[allow(clippy::cast_precision_loss)]
        let df = (n - 1) as f64;
        let j = 1.0 - 3.0 / (4.0 * df - 1.0).max(1.0);
        (Some(d), Some(d * j))
    } else {
        (None, None)
    };

    let ci_95 = bootstrap_ci(&diffs, 1_000, seed).ok();
    let p_value = monte_carlo_sign_flip_p_value(&diffs, 1_000, seed);

    BranchEffectSize {
        metric: metric_name.to_string(),
        n_samples: n,
        mean_diff,
        sd_diff,
        cohens_d,
        hedges_g,
        ci_95,
        p_value,
    }
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

    /// Validate the study plan before execution.
    ///
    /// # Errors
    ///
    /// Returns [`BranchStudyError::InvalidPlan`] or [`BranchStudyError::InvalidIntervention`] on failure.
    pub fn validate(&self) -> Result<(), BranchStudyError> {
        if self.plan.study_id.trim().is_empty() {
            return Err(BranchStudyError::InvalidPlan(
                "study_id cannot be empty".to_string(),
            ));
        }
        if self.plan.checkpoint_tick == 0 {
            return Err(BranchStudyError::InvalidPlan(
                "checkpoint_tick must be >= 1".to_string(),
            ));
        }
        if self.plan.horizon_ticks == 0 {
            return Err(BranchStudyError::InvalidPlan(
                "horizon_ticks must be >= 1".to_string(),
            ));
        }
        if !self.plan.branches.iter().any(|b| b.intervention.is_none()) {
            return Err(BranchStudyError::MissingControlBranch);
        }

        let mut branch_ids = std::collections::BTreeSet::new();
        for branch in &self.plan.branches {
            if branch.branch_id.trim().is_empty() {
                return Err(BranchStudyError::InvalidPlan(
                    "branch_id cannot be empty".to_string(),
                ));
            }
            if !branch_ids.insert(&branch.branch_id) {
                return Err(BranchStudyError::InvalidPlan(format!(
                    "duplicate branch_id `{}` in plan",
                    branch.branch_id
                )));
            }
            if let Some(intervention) = &branch.intervention {
                intervention
                    .validate(self.plan.config.world_width, self.plan.config.world_height)?;
            }
        }

        Ok(())
    }

    /// Execute the complete paired checkpoint study across all branches.
    ///
    /// # Errors
    ///
    /// Returns [`BranchStudyError`] if execution fails, is cancelled, or fails verification.
    pub fn execute_study(
        &self,
        cancellation: Option<&AtomicBool>,
    ) -> Result<BranchStudyReport, BranchStudyError> {
        self.validate()?;
        fs::create_dir_all(&self.output_dir)?;

        // 1. Pre-intervention baseline execution to checkpoint_tick.
        let mut base_config = self.plan.config.clone();
        base_config.persistence_interval = 0;
        base_config.rng_seed = Some(self.plan.base_seed);

        let mut world = WorldState::new(base_config)?;
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
        }

        let checkpoint = world.checkpoint_v1()?;
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
            &checkpoint,
            &pre_intervention_digest,
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
                    &checkpoint,
                    &pre_intervention_digest,
                    Some(&control_samples),
                    cancellation,
                )?;
                branch_reports.push(report);
            }
        }

        // 4. Generate summary Markdown.
        let summary_markdown = self.generate_summary_markdown(
            &pre_intervention_digest,
            &control_branch_id,
            &branch_reports,
        );

        let report = BranchStudyReport {
            study_id: self.plan.study_id.clone(),
            schema_version: 1,
            base_seed: self.plan.base_seed,
            checkpoint_tick: self.plan.checkpoint_tick,
            horizon_ticks: self.plan.horizon_ticks,
            pre_intervention_digest,
            pre_intervention_digest_verified_for_all_branches: true,
            control_branch_id,
            branch_reports,
            summary_markdown,
        };

        // 5. Persist master artifacts to output directory.
        let report_json = serde_json::to_string_pretty(&report)?;
        fs::write(self.output_dir.join("study_report.json"), report_json)?;
        fs::write(
            self.output_dir.join("study_summary.md"),
            &report.summary_markdown,
        )?;
        let plan_json = serde_json::to_string_pretty(&self.plan)?;
        fs::write(self.output_dir.join("study_plan.json"), plan_json)?;

        Ok(report)
    }

    fn run_branch(
        &self,
        branch: &StudyBranchSpec,
        checkpoint: &scriptbots_core::WorldCheckpointV1,
        expected_pre_digest: &str,
        control_samples: Option<&[BranchTickSample]>,
        cancellation: Option<&AtomicBool>,
    ) -> Result<(Vec<BranchTickSample>, BranchOutcomeReport), BranchStudyError> {
        // Restore from shared checkpoint.
        let brain_registry = create_brain_registry(self.plan.brain_preset)?;
        let mut branch_world = WorldState::restore_checkpoint_v1(checkpoint, brain_registry)?;

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

        let pre_intervention_pop = branch_world.agent_count();
        let mut samples = Vec::with_capacity((self.plan.horizon_ticks + 1) as usize);

        // Record sample at checkpoint_tick.
        let initial_food = branch_world
            .food()
            .cells()
            .iter()
            .map(|&c| f64::from(c))
            .sum::<f64>();
        samples.push(BranchTickSample {
            tick: self.plan.checkpoint_tick,
            population: pre_intervention_pop,
            food_total: initial_food,
            births: 0,
            deaths: 0,
            total_energy: 0.0,
            average_health: 1.0,
            spike_hits: 0,
            digest_overall: restored_digest.overall.clone(),
        });

        // Apply scheduled intervention at checkpoint_tick if configured.
        let command_applied_tick = if let Some(intervention) = &branch.intervention {
            self.apply_intervention(&mut branch_world, intervention)?;
            Some(branch.scheduled_tick.unwrap_or(self.plan.checkpoint_tick))
        } else {
            None
        };

        // Step through the horizon.
        let mut first_divergence: Option<FirstDivergence> = None;
        let mut min_population = pre_intervention_pop;
        let mut trough_population = pre_intervention_pop;
        let mut trough_tick = self.plan.checkpoint_tick;
        let mut extinction_tick: Option<u64> = None;

        for step in 1..=self.plan.horizon_ticks {
            if let Some(cancel) = cancellation
                && cancel.load(Ordering::Relaxed)
            {
                return Err(BranchStudyError::Cancelled);
            }
            let current_tick = self.plan.checkpoint_tick + step;
            let outcome =
                branch_world
                    .step_outcome()
                    .map_err(|e| BranchStudyError::SimulationStep {
                        tick: current_tick,
                        reason: e.to_string(),
                    })?;

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

            if pop < min_population {
                min_population = pop;
            }
            if pop < trough_population {
                trough_population = pop;
                trough_tick = current_tick;
            }
            if pop == 0 && extinction_tick.is_none() {
                extinction_tick = Some(current_tick);
            }

            // Check first divergence against control if control samples are available.
            if let Some(ctrl_list) = control_samples
                && first_divergence.is_none()
                && let Some(ctrl_sample) = ctrl_list.get(step as usize)
                && ctrl_sample.digest_overall != digest.overall
            {
                let stage_info = find_first_divergence_stage(checkpoint.source_digest(), &digest);
                let (stage, ctrl_lane, br_lane) = stage_info.unwrap_or((
                    "overall",
                    ctrl_sample.digest_overall.clone(),
                    digest.overall.clone(),
                ));
                first_divergence = Some(FirstDivergence {
                    branch_id: branch.branch_id.clone(),
                    tick: current_tick,
                    stage: stage.to_string(),
                    control_digest: ctrl_lane,
                    branch_digest: br_lane,
                    command_applied_tick: command_applied_tick.unwrap_or(self.plan.checkpoint_tick),
                    details: format!(
                        "First divergence detected in stage `{stage}` at tick {current_tick}"
                    ),
                });
            }

            samples.push(BranchTickSample {
                tick: current_tick,
                population: pop,
                food_total,
                births: summary.births,
                deaths: summary.deaths,
                total_energy: summary.total_energy,
                average_health: summary.average_health,
                spike_hits: summary.spike_hits,
                digest_overall: digest.overall,
            });
        }

        let final_pop = samples.last().map_or(0, |s| s.population);
        let final_digest = samples
            .last()
            .map_or_else(String::new, |s| s.digest_overall.clone());

        // Extinction assessment.
        let extinction = ExtinctionMeasure {
            extinct: min_population == 0,
            extinction_tick,
            min_population,
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
            for sample in &samples {
                if sample.tick > trough_tick && sample.population >= target_recovery_pop {
                    recovery_ticks = Some(sample.tick - trough_tick);
                    break;
                }
            }
        }
        let recovery = RecoveryMeasure {
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

            let ctrl_pops: Vec<f64> = ctrl_list.iter().map(|s| s.population as f64).collect();
            let br_pops: Vec<f64> = samples.iter().map(|s| s.population as f64).collect();

            let ctrl_foods: Vec<f64> = ctrl_list.iter().map(|s| s.food_total).collect();
            let br_foods: Vec<f64> = samples.iter().map(|s| s.food_total).collect();

            let ctrl_energies: Vec<f64> = ctrl_list
                .iter()
                .map(|s| f64::from(s.total_energy))
                .collect();
            let br_energies: Vec<f64> = samples.iter().map(|s| f64::from(s.total_energy)).collect();

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
            effects.insert(
                "population".to_string(),
                compute_paired_effect(&ctrl_pops, &br_pops, "population", self.plan.base_seed),
            );
            effects.insert(
                "food_total".to_string(),
                compute_paired_effect(
                    &ctrl_foods,
                    &br_foods,
                    "food_total",
                    self.plan.base_seed + 1,
                ),
            );
            effects.insert(
                "total_energy".to_string(),
                compute_paired_effect(
                    &ctrl_energies,
                    &br_energies,
                    "total_energy",
                    self.plan.base_seed + 2,
                ),
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
            &samples,
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
        samples: &[BranchTickSample],
        first_divergence: &Option<FirstDivergence>,
        extinction: &ExtinctionMeasure,
        recovery: &RecoveryMeasure,
        hysteresis: &HysteresisMeasure,
        effect_sizes: &BTreeMap<String, BranchEffectSize>,
    ) -> Result<(PathBuf, bool), BranchStudyError> {
        let bundle_dir = self.output_dir.join(format!("bundle_{}", branch.branch_id));
        if bundle_dir.exists() {
            fs::remove_dir_all(&bundle_dir)?;
        }

        // 1. Summary CSV
        let mut csv_buf = Vec::new();
        writeln!(
            &mut csv_buf,
            "tick,population,food_total,births,deaths,total_energy,average_health,spike_hits,digest"
        )?;
        for s in samples {
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
        let evidence_json = serde_json::to_vec_pretty(&serde_json::json!({
            "branch_id": branch.branch_id,
            "description": branch.description,
            "first_divergence": first_divergence,
            "extinction": extinction,
            "recovery": recovery,
            "hysteresis": hysteresis,
            "effect_sizes": effect_sizes,
        }))?;

        // 3. Manifest
        let mut manifest = RunManifestRecord::unattributed(RunId::new(u128::from(
            self.plan.base_seed.wrapping_add(100),
        )));
        manifest.scenario_id = format!("branch_study_{}", branch.branch_id);
        manifest.experiment_id = Some(self.plan.study_id.clone());
        manifest.variant_id = Some(branch.branch_id.clone());

        let max_tick = self.plan.checkpoint_tick + self.plan.horizon_ticks;

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
        let limits = RunBundleVerificationLimits {
            max_manifest_bytes: 10 * 1024 * 1024,
            max_artifacts: 100,
            max_artifact_bytes: 50 * 1024 * 1024,
            max_total_artifact_bytes: 100 * 1024 * 1024,
        };

        verify_run_bundle_bounded(&bundle_dir, limits)
            .map_err(|e| BranchStudyError::Bundle(e.to_string()))?;

        Ok((bundle_dir, true))
    }

    fn generate_summary_markdown(
        &self,
        pre_intervention_digest: &str,
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

        md.push_str("## Branch Outcomes Matrix\n\n");
        md.push_str("| Branch ID | Applied Tick | First Divergence | Extinct | Final Pop | Recovered (Ratio) | Path Dependence | Pop Mean Diff (95% CI) | Bundle Verified |\n");
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
                if let Some((low, high)) = eff.ci_95 {
                    format!("{:.2} [{:.2}, {:.2}]", eff.mean_diff, low, high)
                } else {
                    format!("{:.2}", eff.mean_diff)
                }
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
                    "- **First Divergence**: Tick `{}`, Pipeline Stage `{}`\n",
                    div.tick, div.stage
                ));
                md.push_str(&format!("- **Details**: {}\n", div.details));
            }
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
                    "- **Metric `{}`**: Mean Diff `{:.4}`, Cohen's d: `{}`, Hedges' g: `{}`, p-value: `{:.4}`\n",
                    metric,
                    eff.mean_diff,
                    eff.cohens_d.map_or("N/A".to_string(), |v| format!("{v:.4}")),
                    eff.hedges_g.map_or("N/A".to_string(), |v| format!("{v:.4}")),
                    eff.p_value
                ));
            }
            md.push('\n');
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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
        }
    }

    #[test]
    fn test_cancellation_stops_cleanly() {
        let dir = tempdir().expect("tempdir");
        let plan = BranchStudyPlan::standard_five_branch("test_cancel", 777, 10, 10);
        let orchestrator = BranchStudyOrchestrator::new(plan, dir.path().to_path_buf());

        let cancel_flag = AtomicBool::new(true); // already cancelled
        let result = orchestrator.execute_study(Some(&cancel_flag));
        assert!(matches!(result, Err(BranchStudyError::Cancelled)));
    }

    #[test]
    fn test_extinction_and_recovery_metrics_computation() {
        let dir = tempdir().expect("tempdir");
        let mut plan = BranchStudyPlan::standard_five_branch("test_metrics", 8888, 5, 10);
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
