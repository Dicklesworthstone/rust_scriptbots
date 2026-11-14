//! Deterministic matched-seed experiment runner and batch execution engine (bd-2z0.5.11).

use crate::{
    BuildProvenanceV0, RunIdentityV1, RunManifestV3, ScenarioIdentityV0, ScenarioInterventionV1,
    ThreadPolicyV0, resolve_scheduled_config_patch, seed_founding_population,
};
use scriptbots_brain::{
    AssemblyBrain, DwraonBrain, MlpBrain, assembly::AssemblyFamilyAdapter,
    dwraon::DwraonFamilyAdapter, mlp::MlpBrainFamily,
};
#[cfg(feature = "brain-ft")]
use scriptbots_brain_ml::{FT_BRAIN_KIND, FtBrainFamily};
#[cfg(feature = "neuro")]
use scriptbots_brain_neuro::{NeuroflowBrain, NeuroflowBrainConfig, NeuroflowFamilyAdapter};
use scriptbots_core::{
    BrainFamilyAdapter, BrainRegistry, NullPersistence, ScriptBotsConfig, Tick, WorldState,
    knob_range,
};
use scriptbots_runtime::{
    ApplicationState, CommandEnvelope, CommandId, HostCheckpointV1, HostCommand, HostCore,
    HostCoreOptions, HostLifecycle, HostSessionId, JournalState, ManualInstant, NullFrontend,
    PlaybackSnapshot, RunId, ScheduledConfigPatch, ScheduledPatchOutcome, VolatileHostCheckpointV1,
};
use scriptbots_storage::{
    RunBundleV1, RunBundleVerificationResult, bundle::RunBundleVerificationLimits,
    bundle::verify_run_bundle_bounded, create_run_bundle_from_artifacts,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tracing::{error, info};

const STATUS_SCHEMA_VERSION: u16 = 2;
pub(crate) const MAX_STATUS_BYTES: u64 = 16 * 1024 * 1024;
pub(crate) const MAX_BUNDLE_ARTIFACTS: usize = 32;
pub(crate) const MAX_BUNDLE_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024;
pub(crate) const MAX_BUNDLE_TOTAL_ARTIFACT_BYTES: u64 = 32 * 1024 * 1024;
pub(crate) const MAX_EXPERIMENT_SUMMARY_BYTES: usize = 4_096;
const EXPERIMENT_SUMMARY_SCHEMA_VERSION: u16 = 2;
const MAX_BATCH_RUNS: usize = 4_096;
pub(crate) const MAX_RUN_COMPONENT_BYTES: usize = 128;
pub(crate) const MAX_TICKS_PER_RUN: u64 = 10_000_000;
pub(crate) const MAX_CONCURRENT_RUNS: usize = 64;
// Admission limits for the shared dense food/terrain grid dimensions. Each cell
// also feeds profiles, tick scratch and checkpoint state, so bounding only each
// world dimension admits prohibitive products. These are cardinality policies,
// not measurements or guarantees of total memory, population or artifact bytes.
const MAX_RUN_GRID_CELLS: u64 = 262_144;
pub(crate) const MAX_CONCURRENT_GRID_CELLS: u64 = 1_048_576;
// Match the managed branch-study population policy. These are per-owner
// cardinality limits, not a total heap bound: a tick can cross the population
// ceiling before the runner observes it. Refuse further work and checkpoint
// capture at that boundary rather than truncating or changing the science.
const MAX_RUN_POPULATION: u64 = crate::branch_study::MAX_STUDY_POPULATION;
// WorldState preallocates its history ring from this value. The same ceiling
// also bounds retained narrative events; both are copied into owner cuts.
const MAX_RUN_HISTORY_ENTRIES: u64 = 10_000;
pub(crate) const MAX_EXPERIMENT_INTERVENTIONS: usize = 64;
pub(crate) const MAX_EXPERIMENT_INTERVENTION_PATCH_BYTES: usize = 4_096;
const HOST_RECEIPT_DRIVE_LIMIT: usize = 16;
const RUN_CHECKPOINT_MAGIC: &[u8] = b"SBRUNCH1";
const MAX_RUN_CHECKPOINT_BYTES: u64 = MAX_BUNDLE_ARTIFACT_BYTES;
const FINAL_CHECKPOINT_ARTIFACT: (&str, &str) = (
    "checkpoints/final.checkpoint",
    "experiment-final-checkpoint",
);
const REQUIRED_EXPERIMENT_ARTIFACTS: &[(&str, &str)] = &[
    ("exports/summary.csv", "experiment-summary"),
    ("evidence/run.json", "experiment-run-evidence"),
    FINAL_CHECKPOINT_ARTIFACT,
];

// Public entry points always choose MAX_RUN_POPULATION. Carrying the bound
// alongside cancellation lets the real execution path exercise a small crossing
// population in regression tests without allocating 10,001 complete brains.
#[derive(Clone, Copy)]
struct RunExecutionBudget<'a> {
    cancel: &'a std::sync::atomic::AtomicBool,
    population_limit: u64,
}

/// Typed failure from matched-seed planning, status recovery, execution, or bundle verification.
#[derive(Debug, Error)]
pub enum ExperimentRunnerError {
    /// A declarative experiment cannot produce an unambiguous bounded plan.
    #[error("invalid experiment plan: {reason}")]
    InvalidPlan { reason: String },
    /// One normalized world would allocate too many dense grid cells.
    #[error(
        "variant `{variant_id}` requires {cells} dense grid cells, exceeding the per-run limit {limit}; reduce world dimensions or increase food_cell_size"
    )]
    RunGridCapacityExceeded {
        variant_id: String,
        cells: u64,
        limit: u64,
    },
    /// Concurrent owners in one plan would exceed the shared dense-grid budget.
    #[error(
        "experiment's largest {concurrent_runs} concurrent runs require {cells} dense grid cells, exceeding the limit {limit}; reduce max_concurrency or grid dimensions, or increase food_cell_size"
    )]
    ConcurrentGridCapacityExceeded {
        concurrent_runs: usize,
        cells: u64,
        limit: u64,
    },
    /// An initial configuration or resolved intervention exceeds an owner budget.
    #[error(
        "variant `{variant_id}` configuration step {configuration_step} requests {requested} {resource}, exceeding the per-run limit {limit}; step 0 is the initial configuration"
    )]
    RunConfigCapacityExceeded {
        variant_id: String,
        configuration_step: u64,
        resource: &'static str,
        requested: u64,
        limit: u64,
    },
    /// A completed boundary exceeded the population budget before checkpoint capture.
    #[error(
        "run `{run_id}` reached population {population} at tick {tick}, exceeding the per-run limit {limit}; last retained checkpoint tick: {retained_checkpoint_tick:?}; no successor checkpoint was published"
    )]
    RunPopulationCapacityExceeded {
        run_id: String,
        tick: u64,
        population: u64,
        limit: u64,
        retained_checkpoint_tick: Option<u64>,
    },
    /// A variant names no registered protocol family supported by this runner.
    #[error("variant `{variant_id}` requests unknown brain family `{brain_family}`")]
    UnknownBrainFamily {
        variant_id: String,
        brain_family: String,
    },
    /// A known family cannot satisfy checkpointable experiment-runner semantics in this build.
    #[error("variant `{variant_id}` requests unavailable brain family `{brain_family}`: {reason}")]
    UnavailableBrainFamily {
        variant_id: String,
        brain_family: String,
        reason: String,
    },
    /// A previously written status file is malformed or contradicts the current plan.
    #[error("invalid experiment status at {path}: {reason}")]
    InvalidStatus { path: PathBuf, reason: String },
    /// A prior write left a temporary status file that cannot be safely promoted.
    #[error(
        "interrupted experiment status write left {temporary}; inspect it against {committed} before recovery"
    )]
    InterruptedStatusWrite {
        temporary: PathBuf,
        committed: PathBuf,
    },
    /// Another process or runner instance owns the status writer lease.
    #[error("experiment status writer lease is already held at {0}")]
    StatusWriterLeaseHeld(PathBuf),
    /// The caller tried to write a snapshot derived from an older committed generation.
    #[error(
        "stale experiment status generation at {path}: proposed {proposed}, committed {committed}"
    )]
    StaleStatusGeneration {
        path: PathBuf,
        proposed: u64,
        committed: u64,
    },
    /// An unfinished running record has no validated joint continuation artifact.
    #[error(
        "run `{run_id}` stopped in `running` state at status tick {completed_ticks} without a validated continuation artifact; automatic tick-zero replay is refused"
    )]
    ResumeCheckpointUnavailable {
        run_id: String,
        completed_ticks: u64,
    },
    /// A continuation artifact failed framing, identity, or atomic-write validation.
    #[error("invalid experiment checkpoint at {path}: {reason}")]
    InvalidCheckpoint { path: PathBuf, reason: String },
    /// Filesystem work failed and was not ignored.
    #[error("{operation} failed at {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    /// A canonical JSON boundary failed.
    #[error("{operation} JSON failed: {source}")]
    Json {
        operation: &'static str,
        #[source]
        source: serde_json::Error,
    },
    /// World, brain-family, or manifest construction failed for one run.
    #[error("run `{run_id}` construction failed: {reason}")]
    Construction { run_id: String, reason: String },
    /// A canonical HostCore/HostClient operation failed or returned an impossible receipt.
    #[error("run `{run_id}` host operation `{operation}` failed: {reason}")]
    Host {
        run_id: String,
        operation: &'static str,
        reason: String,
    },
    /// A bundle path already exists for a run that is not recorded complete.
    #[error(
        "run `{run_id}` refuses to overwrite existing bundle path {path}; reconcile or relocate the artifact explicitly"
    )]
    BundleAlreadyExists { run_id: String, path: PathBuf },
    /// Bundle construction or verification failed.
    #[error("run `{run_id}` bundle operation failed: {reason}")]
    Bundle { run_id: String, reason: String },
    /// A worker panicked; the run is recorded failed rather than reported complete.
    #[error("run `{run_id}` worker panicked")]
    WorkerPanicked { run_id: String },
}

/// Matched-seed cohort specifying the identical seed schedule shared across scenario variants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchedSeedCohort {
    pub cohort_id: String,
    pub seeds: Vec<u64>,
}

/// Scenario variant specification defining a brain family and configuration tuning overrides.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioVariant {
    pub variant_id: String,
    pub brain_family: String,
    /// Canonical validated arm. JSON values preserve integer-vs-float types
    /// until strict `ScriptBotsConfig` deserialization.
    pub config_overrides: BTreeMap<String, serde_json::Value>,
    /// Ordered live numeric config treatments, applied by the owner before the next tick.
    /// Ticks must be nondecreasing; statements at the same tick retain document order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interventions: Vec<ScenarioInterventionV1>,
}

/// Execution status of an individual experiment run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunState {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Detailed run record tracking progress, final digest, and bundle location.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RunRecord {
    pub run_id: String,
    pub variant_id: String,
    /// Canonical registered family identifier, never the caller's unchecked alias.
    pub brain_family: String,
    pub seed: u64,
    pub state: RunState,
    /// Number of scientific transitions durably represented by this status.
    pub total_ticks: u64,
    pub final_digest: Option<String>,
    pub bundle_path: Option<String>,
    pub error_reason: Option<String>,
}

/// How the last complete checkpoint was actually observed. Neither observation
/// continuously re-verifies the current file or certifies the completed run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointProgressSource {
    /// The runner successfully wrote, synced and published this exact cut.
    Published,
    /// An existing committed cut was read, validated and restored to an owner.
    Recovered,
}

/// An absent progress observation is distinct from an observed tick zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointProgressUnavailable {
    NotStarted,
    AwaitingPublication,
    RecoveryPending,
    NoCommittedCheckpoint,
}

/// Typed refusal of a checkpoint progress observation, without inventing ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointProgressRefusal {
    InvalidCheckpoint,
    UnreadableCheckpoint,
    ConflictingObservation,
}

/// Last successfully observed checkpoint for the enclosing source-bound plan/run.
///
/// This is observational progress, separate from terminal `RunRecord::total_ticks`.
/// Its digest identifies the complete encoded checkpoint bytes. Cached observations
/// describe that exact historical cut; downloads and resume still perform their own
/// complete validation. A stopped reader recovers at most one cut per status read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ExperimentCheckpointProgress {
    Observed {
        tick: u64,
        checkpoint_digest: String,
        #[schema(inline)]
        source: CheckpointProgressSource,
    },
    Unavailable {
        #[schema(inline)]
        reason: CheckpointProgressUnavailable,
    },
    Refused {
        #[schema(inline)]
        reason: CheckpointProgressRefusal,
    },
}

impl ExperimentCheckpointProgress {
    #[must_use]
    pub const fn observed_tick(&self) -> Option<u64> {
        match self {
            Self::Observed { tick, .. } => Some(*tick),
            Self::Unavailable { .. } | Self::Refused { .. } => None,
        }
    }
}

#[derive(Default)]
struct CheckpointProgressCache {
    plan_digest: String,
    output_dir: PathBuf,
    /// Exactly the admitted run IDs, bounded by `MAX_BATCH_RUNS`; no history.
    runs: BTreeMap<String, Option<ExperimentCheckpointProgress>>,
    /// Across concurrent status readers, at most one bounded recovery is active.
    recovering_run: Option<String>,
}

/// Complete experiment batch status report supporting atomic checkpointing and resume.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExperimentBatchStatus {
    /// Exact status wire version.
    pub schema_version: u16,
    /// Monotonic compare-and-swap generation for stale-writer rejection.
    pub generation: u64,
    /// BLAKE3 of the complete scientific plan and compiled-build identity.
    pub plan_digest: String,
    pub experiment_id: String,
    pub total_runs: usize,
    pub completed_runs: usize,
    pub failed_runs: usize,
    pub runs: Vec<RunRecord>,
}

impl ExperimentBatchStatus {
    pub fn is_finished(&self) -> bool {
        self.completed_runs + self.failed_runs == self.total_runs
    }
}

/// Canonical family used when an experiment proposal omits its family axis.
pub(crate) const DEFAULT_EXPERIMENT_BRAIN_FAMILY: &str = MlpBrain::KIND.as_str();

/// Canonical experiment families backed by executable adapters in this build.
/// Proposal schemas and arm validation consume the same installed-family list.
pub(crate) const EXPERIMENT_BRAIN_FAMILIES: &[&str] = &[
    DEFAULT_EXPERIMENT_BRAIN_FAMILY,
    DwraonBrain::KIND.as_str(),
    AssemblyBrain::KIND.as_str(),
    #[cfg(feature = "brain-ft")]
    FT_BRAIN_KIND,
    #[cfg(feature = "neuro")]
    NeuroflowBrain::KIND.as_str(),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ResolvedBrainFamily {
    Mlp,
    Dwraon,
    Assembly,
    #[cfg(feature = "brain-ft")]
    FrankenTorch,
    #[cfg(feature = "neuro")]
    Neuroflow,
}

impl ResolvedBrainFamily {
    fn resolve(variant: &ScenarioVariant) -> Result<Self, ExperimentRunnerError> {
        match variant.brain_family.as_str() {
            "mlp" | "mlp.baseline" => Ok(Self::Mlp),
            "dwraon" | "dwraon.baseline" => Ok(Self::Dwraon),
            "assembly" | "assembly.experimental" => Ok(Self::Assembly),
            #[cfg(feature = "brain-ft")]
            "ft" | "frankentorch" => Ok(Self::FrankenTorch),
            #[cfg(not(feature = "brain-ft"))]
            "ft" | "frankentorch" => Err(ExperimentRunnerError::UnavailableBrainFamily {
                variant_id: variant.variant_id.clone(),
                brain_family: variant.brain_family.clone(),
                reason: "FrankenTorch experiments require a build with the brain-ft feature"
                    .to_owned(),
            }),
            #[cfg(feature = "neuro")]
            "neuro" | "neuroflow" | "ml.neuroflow" => Ok(Self::Neuroflow),
            #[cfg(not(feature = "neuro"))]
            "neuro" | "neuroflow" | "ml.neuroflow" => {
                Err(ExperimentRunnerError::UnavailableBrainFamily {
                    variant_id: variant.variant_id.clone(),
                    brain_family: variant.brain_family.clone(),
                    reason: "NeuroFlow experiments require a build with the neuro feature"
                        .to_owned(),
                })
            }
            _ => Err(ExperimentRunnerError::UnknownBrainFamily {
                variant_id: variant.variant_id.clone(),
                brain_family: variant.brain_family.clone(),
            }),
        }
    }

    const fn canonical_name(self) -> &'static str {
        match self {
            Self::Mlp => MlpBrain::KIND.as_str(),
            Self::Dwraon => DwraonBrain::KIND.as_str(),
            Self::Assembly => AssemblyBrain::KIND.as_str(),
            #[cfg(feature = "brain-ft")]
            Self::FrankenTorch => FT_BRAIN_KIND,
            #[cfg(feature = "neuro")]
            Self::Neuroflow => NeuroflowBrain::KIND.as_str(),
        }
    }

    fn adapter(
        self,
        constructor_config: &ScriptBotsConfig,
        run_id: &str,
    ) -> Result<Box<dyn BrainFamilyAdapter>, ExperimentRunnerError> {
        #[cfg(not(feature = "neuro"))]
        let _ = constructor_config;
        Ok(match self {
            Self::Mlp => Box::new(MlpBrainFamily::new()),
            Self::Dwraon => Box::new(DwraonFamilyAdapter::default()),
            Self::Assembly => {
                let adapter = AssemblyFamilyAdapter::new().map_err(|source| {
                    ExperimentRunnerError::Construction {
                        run_id: run_id.to_owned(),
                        reason: format!("Assembly family construction failed: {source}"),
                    }
                })?;
                Box::new(adapter)
            }
            #[cfg(feature = "brain-ft")]
            Self::FrankenTorch => Box::new(FtBrainFamily::default()),
            #[cfg(feature = "neuro")]
            Self::Neuroflow => Box::new(
                NeuroflowFamilyAdapter::new(NeuroflowBrainConfig::from_settings(
                    &constructor_config.neuroflow,
                ))
                .map_err(|source| ExperimentRunnerError::Construction {
                    run_id: run_id.to_owned(),
                    reason: format!("configured NeuroFlow family construction failed: {source}"),
                })?,
            ),
        })
    }

    fn register(self, world: &mut WorldState, run_id: &str) -> Result<u64, ExperimentRunnerError> {
        let adapter = self.adapter(world.config(), run_id)?;
        world
            .register_brain_family(self.canonical_name(), adapter)
            .map_err(|source| ExperimentRunnerError::Construction {
                run_id: run_id.to_owned(),
                reason: format!(
                    "brain family `{}` registration failed: {source}",
                    self.canonical_name()
                ),
            })
    }
}

#[derive(Debug)]
struct ValidatedPlan {
    digest: String,
    runs: Vec<RunRecord>,
    variants: BTreeMap<String, (ScenarioVariant, ResolvedBrainFamily)>,
    concurrent_grid_cells: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunCheckpointMetadataV1 {
    plan_digest: String,
    run_id: String,
    parent_digest: Option<String>,
    manifest: RunManifestV3,
}

struct RunCheckpointV1 {
    metadata: RunCheckpointMetadataV1,
    owner: VolatileHostCheckpointV1,
}

impl RunCheckpointV1 {
    fn encode(&self, path: &Path) -> Result<Vec<u8>, ExperimentRunnerError> {
        let metadata =
            serde_json::to_vec(&self.metadata).map_err(|source| ExperimentRunnerError::Json {
                operation: "encode run checkpoint metadata",
                source,
            })?;
        let owner = self
            .owner
            .encode()
            .map_err(|error| invalid_checkpoint(path, error))?;
        let length = RUN_CHECKPOINT_MAGIC
            .len()
            .checked_add(size_of::<[u64; 2]>())
            .and_then(|n| n.checked_add(metadata.len()))
            .and_then(|n| n.checked_add(owner.len()))
            .and_then(|n| n.checked_add(blake3::OUT_LEN))
            .filter(|n| u64::try_from(*n).is_ok_and(|n| n <= MAX_RUN_CHECKPOINT_BYTES))
            .ok_or_else(|| {
                invalid_checkpoint(path, "encoded run checkpoint exceeds its byte cap")
            })?;
        let mut bytes = Vec::with_capacity(length);
        bytes.extend_from_slice(RUN_CHECKPOINT_MAGIC);
        for n in [metadata.len(), owner.len()] {
            bytes.extend_from_slice(
                &u64::try_from(n)
                    .map_err(|error| invalid_checkpoint(path, error))?
                    .to_le_bytes(),
            );
        }
        bytes.extend_from_slice(&metadata);
        bytes.extend_from_slice(&owner);
        let digest = *blake3::hash(&bytes).as_bytes();
        bytes.extend_from_slice(&digest);
        Ok(bytes)
    }

    fn decode(bytes: &[u8], path: &Path) -> Result<Self, ExperimentRunnerError> {
        let header = RUN_CHECKPOINT_MAGIC.len() + size_of::<[u64; 2]>();
        if bytes.len() < header + blake3::OUT_LEN
            || u64::try_from(bytes.len()).map_or(true, |n| n > MAX_RUN_CHECKPOINT_BYTES)
            || !bytes.starts_with(RUN_CHECKPOINT_MAGIC)
        {
            return Err(invalid_checkpoint(path, "invalid frame or byte cap"));
        }
        let end = bytes.len() - blake3::OUT_LEN;
        if blake3::hash(&bytes[..end]).as_bytes() != &bytes[end..] {
            return Err(invalid_checkpoint(path, "frame checksum differs"));
        }
        let mut lengths = [0_usize; 2];
        for (index, length) in lengths.iter_mut().enumerate() {
            let begin = RUN_CHECKPOINT_MAGIC.len() + index * size_of::<u64>();
            let raw = u64::from_le_bytes(
                bytes[begin..begin + size_of::<u64>()]
                    .try_into()
                    .map_err(|error| invalid_checkpoint(path, error))?,
            );
            *length = usize::try_from(raw).map_err(|error| invalid_checkpoint(path, error))?;
        }
        let middle = header
            .checked_add(lengths[0])
            .filter(|n| *n <= end)
            .ok_or_else(|| invalid_checkpoint(path, "metadata length exceeds frame"))?;
        if middle.checked_add(lengths[1]) != Some(end) {
            return Err(invalid_checkpoint(path, "owner length differs from frame"));
        }
        let checkpoint = Self {
            metadata: serde_json::from_slice(&bytes[header..middle])
                .map_err(|error| invalid_checkpoint(path, error))?,
            owner: VolatileHostCheckpointV1::decode(&bytes[middle..end])
                .map_err(|error| invalid_checkpoint(path, error))?,
        };
        if checkpoint.encode(path)? != bytes {
            return Err(invalid_checkpoint(path, "noncanonical run checkpoint"));
        }
        Ok(checkpoint)
    }
}

fn invalid_checkpoint(path: &Path, reason: impl std::fmt::Display) -> ExperimentRunnerError {
    ExperimentRunnerError::InvalidCheckpoint {
        path: path.to_path_buf(),
        reason: reason.to_string(),
    }
}

fn validate_run_config_capacity(
    variant_id: &str,
    config: &ScriptBotsConfig,
    configuration_step: u64,
) -> Result<(), ExperimentRunnerError> {
    for (resource, requested, limit) in [
        (
            "history entries",
            u64::try_from(config.history_capacity).unwrap_or(u64::MAX),
            MAX_RUN_HISTORY_ENTRIES,
        ),
        (
            "narrative events",
            u64::try_from(config.narrative_capacity).unwrap_or(u64::MAX),
            MAX_RUN_HISTORY_ENTRIES,
        ),
        (
            "population floor",
            u64::try_from(config.population_minimum).unwrap_or(u64::MAX),
            MAX_RUN_POPULATION,
        ),
        (
            "population injection count",
            u64::from(config.population_spawn_count),
            MAX_RUN_POPULATION,
        ),
    ] {
        if requested > limit {
            return Err(ExperimentRunnerError::RunConfigCapacityExceeded {
                variant_id: variant_id.to_owned(),
                configuration_step,
                resource,
                requested,
                limit,
            });
        }
    }
    Ok(())
}

fn enforce_run_population(
    world: &WorldState,
    run_id: &str,
    limit: u64,
    retained_checkpoint_tick: Option<u64>,
) -> Result<(), ExperimentRunnerError> {
    let population = u64::try_from(world.agents().len()).unwrap_or(u64::MAX);
    if population > limit {
        return Err(ExperimentRunnerError::RunPopulationCapacityExceeded {
            run_id: run_id.to_owned(),
            tick: world.tick().0,
            population,
            limit,
            retained_checkpoint_tick,
        });
    }
    Ok(())
}

fn validate_run_checkpoint_capacity(
    checkpoint: &RunCheckpointV1,
    path: &Path,
    population_limit: u64,
) -> Result<(), ExperimentRunnerError> {
    // Apply the same per-owner policy before executable restore and before a
    // recovered temporary cut can replace the preceding admitted checkpoint.
    let host = checkpoint.owner.host_checkpoint();
    validate_run_config_capacity(
        checkpoint
            .metadata
            .manifest
            .identity
            .variant_id
            .as_deref()
            .unwrap_or(&checkpoint.metadata.run_id),
        host.config(),
        host.completed_schedule_sequence().unwrap_or(0),
    )
    .map_err(|error| invalid_checkpoint(path, error))?;
    let population = u64::try_from(host.world_checkpoint().agent_count()).unwrap_or(u64::MAX);
    if population > population_limit {
        return Err(invalid_checkpoint(
            path,
            format!(
                "checkpoint population {population} exceeds the per-run limit {population_limit}"
            ),
        ));
    }
    Ok(())
}

fn restore_run_owner(
    checkpoint: &RunCheckpointV1,
    family: ResolvedBrainFamily,
    run_id: &str,
    path: &Path,
) -> Result<HostCore, ExperimentRunnerError> {
    validate_run_checkpoint_capacity(checkpoint, path, MAX_RUN_POPULATION)?;
    // Adapter identity belongs to the original constructor recipe. Current raw
    // world settings can change without replacing the installed genome family.
    let constructor_config: ScriptBotsConfig =
        serde_json::from_value(checkpoint.metadata.manifest.normalized_config.clone())
            .map_err(|error| invalid_checkpoint(path, error))?;
    constructor_config
        .validate()
        .map_err(|error| invalid_checkpoint(path, error))?;
    let mut registry = BrainRegistry::new();
    registry
        .register_family(
            family.canonical_name(),
            family
                .adapter(&constructor_config, run_id)
                .map_err(|error| invalid_checkpoint(path, error))?,
        )
        .map_err(|error| invalid_checkpoint(path, error))?;
    let world = checkpoint
        .owner
        .host_checkpoint()
        .restore_world(registry)
        .map_err(|error| invalid_checkpoint(path, error))?;
    let persistence = world
        .bind_persistence(Box::new(NullPersistence))
        .map_err(|error| invalid_checkpoint(path, error))?;
    HostCore::restore_volatile_host_checkpoint_v1(
        &checkpoint.owner,
        world,
        persistence,
        checkpoint.owner.host_checkpoint().has_schedule().then_some(
            resolve_scheduled_config_patch as scriptbots_runtime::PreparedConfigPatchResolver,
        ),
        ManualInstant::from_nanos(0),
    )
    .map_err(|error| invalid_checkpoint(path, error))
}

fn seal_run_checkpoint(
    run_id: &str,
    frontend: &mut NullFrontend<scriptbots_runtime::LocalHostPort>,
    core: &mut HostCore,
    next_nanos: &mut u64,
) -> Result<(), ExperimentRunnerError> {
    let tick = core.world_tick().0;
    let submitted =
        core.request_persistence_flush()
            .map_err(|source| ExperimentRunnerError::Host {
                run_id: run_id.to_owned(),
                operation: "submit checkpoint flush",
                reason: source.to_string(),
            })?;
    drive_command_to_volatile_commit(
        run_id,
        "checkpoint flush",
        frontend,
        core,
        submitted.command_id(),
        tick,
        0,
        next_nanos,
    )
}

pub(crate) fn read_experiment_file_bounded(
    path: &Path,
    max_bytes: u64,
) -> Result<Vec<u8>, ExperimentRunnerError> {
    let before = fs::symlink_metadata(path).map_err(|source| ExperimentRunnerError::Io {
        operation: "inspect experiment artifact",
        path: path.to_path_buf(),
        source,
    })?;
    if before.file_type().is_symlink() || !before.is_file() {
        return Err(invalid_checkpoint(
            path,
            "experiment artifact must be a regular file",
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|source| ExperimentRunnerError::Io {
            operation: "open experiment artifact",
            path: path.to_path_buf(),
            source,
        })?;
    let opened = file
        .metadata()
        .map_err(|source| ExperimentRunnerError::Io {
            operation: "inspect opened experiment artifact",
            path: path.to_path_buf(),
            source,
        })?;
    if !opened.is_file() || opened.len() > max_bytes || !same_status_file_identity(&before, &opened)
    {
        return Err(invalid_checkpoint(
            path,
            "opened experiment artifact identity, type or byte cap differs",
        ));
    }
    let mut bytes = Vec::with_capacity(
        usize::try_from(opened.len()).map_err(|error| invalid_checkpoint(path, error))?,
    );
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| ExperimentRunnerError::Io {
            operation: "read experiment artifact",
            path: path.to_path_buf(),
            source,
        })?;
    if u64::try_from(bytes.len()).map_or(true, |length| length > max_bytes) {
        return Err(invalid_checkpoint(
            path,
            "experiment artifact exceeded its read cap",
        ));
    }
    let after = fs::symlink_metadata(path).map_err(|source| ExperimentRunnerError::Io {
        operation: "reinspect experiment artifact",
        path: path.to_path_buf(),
        source,
    })?;
    if after.file_type().is_symlink() || !same_status_file_identity(&before, &after) {
        return Err(invalid_checkpoint(
            path,
            "experiment artifact identity changed while reading",
        ));
    }
    Ok(bytes)
}

fn read_optional_run_checkpoint(
    path: &Path,
) -> Result<Option<RunCheckpointV1>, ExperimentRunnerError> {
    match fs::symlink_metadata(path) {
        Ok(_) => RunCheckpointV1::decode(
            &read_experiment_file_bounded(path, MAX_RUN_CHECKPOINT_BYTES)?,
            path,
        )
        .map(Some),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(ExperimentRunnerError::Io {
            operation: "inspect checkpoint",
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn write_run_checkpoint(path: &Path, bytes: &[u8]) -> Result<(), ExperimentRunnerError> {
    let temporary = status_temporary_path(path)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = options
        .open(&temporary)
        .map_err(|source| ExperimentRunnerError::Io {
            operation: "create temporary checkpoint",
            path: temporary.clone(),
            source,
        })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|source| ExperimentRunnerError::Io {
            operation: "write and sync checkpoint",
            path: temporary.clone(),
            source,
        })?;
    fs::rename(&temporary, path).map_err(|source| ExperimentRunnerError::Io {
        operation: "commit checkpoint",
        path: path.to_path_buf(),
        source,
    })?;
    sync_parent_directory(path.parent().unwrap_or_else(|| Path::new(".")))
}

struct ExperimentStatusWriterLease {
    _file: File,
}

impl ExperimentStatusWriterLease {
    fn acquire(state_file: &Path) -> Result<Self, ExperimentRunnerError> {
        let parent = state_file
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|source| ExperimentRunnerError::Io {
            operation: "create status parent",
            path: parent.to_path_buf(),
            source,
        })?;
        let lock_path = status_lock_path(state_file)?;
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let file = options
            .open(&lock_path)
            .map_err(|source| ExperimentRunnerError::Io {
                operation: "open status writer lease",
                path: lock_path.clone(),
                source,
            })?;
        let opened_metadata = file
            .metadata()
            .map_err(|source| ExperimentRunnerError::Io {
                operation: "inspect status writer lease",
                path: lock_path.clone(),
                source,
            })?;
        let path_metadata =
            fs::symlink_metadata(&lock_path).map_err(|source| ExperimentRunnerError::Io {
                operation: "verify status writer lease",
                path: lock_path.clone(),
                source,
            })?;
        if !opened_metadata.is_file()
            || path_metadata.file_type().is_symlink()
            || !path_metadata.is_file()
            || !same_status_file_identity(&opened_metadata, &path_metadata)
        {
            return Err(ExperimentRunnerError::InvalidStatus {
                path: lock_path,
                reason: "writer lease must remain the same regular file across open".to_owned(),
            });
        }
        match file.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(ExperimentRunnerError::StatusWriterLeaseHeld(lock_path));
            }
            Err(std::fs::TryLockError::Error(source)) => {
                return Err(ExperimentRunnerError::Io {
                    operation: "lock status writer lease",
                    path: lock_path,
                    source,
                });
            }
        }
        let locked_metadata =
            fs::symlink_metadata(&lock_path).map_err(|source| ExperimentRunnerError::Io {
                operation: "reverify status writer lease",
                path: lock_path.clone(),
                source,
            })?;
        if locked_metadata.file_type().is_symlink()
            || !locked_metadata.is_file()
            || !same_status_file_identity(&opened_metadata, &locked_metadata)
        {
            return Err(ExperimentRunnerError::InvalidStatus {
                path: lock_path,
                reason: "writer lease path changed while its descriptor was locked".to_owned(),
            });
        }
        Ok(Self { _file: file })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RunEvidenceV2 {
    schema: String,
    summary_schema_version: u16,
    plan_digest: String,
    experiment_id: String,
    cohort_id: String,
    run_id: String,
    variant_id: String,
    brain_family: String,
    seed: u64,
    total_ticks: u64,
    final_digest: String,
    manifest_digest: String,
}

/// Final-state metrics emitted by the runner and accepted by the Lab proposal schema.
///
/// These describe one completed boundary. They are not time averages, cumulative
/// lifecycle counts, or observations over the run's unretained history.
pub const EXPERIMENT_ENDPOINT_METRICS: [&str; 8] = [
    "alive_agents",
    "total_energy",
    "average_energy",
    "average_health",
    "mean_age",
    "max_age",
    "max_generation",
    "total_food",
];

/// Canonical V2 experiment summary, recomputable from the final checkpoint alone.
///
/// Agent values are reduced in ascending stable `AgentUid` order using `f64`;
/// food cells are reduced in row-major order. No renderer snapshot or retained
/// tick history participates. For an extinct population, the population and
/// total energy are zero while all population means and maxima are unavailable.
/// Those `None` values become empty CSV cells and are omitted from analysis,
/// causing an explicitly requested unavailable endpoint to fail closed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExperimentEndpointSummaryV2 {
    /// Exact completed simulation boundary.
    pub tick: u64,
    /// Number of living agents at that boundary.
    pub alive_agents: u32,
    /// The matched-seed axis that produced the run.
    pub seed: u64,
    /// Canonical registered family identifier.
    pub brain_family: String,
    /// Scientific digest of the same final owner state.
    pub final_digest: String,
    /// Summary wire and reduction semantics version, currently 2.
    pub schema_version: u16,
    /// Sum of current live-agent energy reserves.
    pub total_energy: f64,
    /// Mean current live-agent energy; unavailable when the population is empty.
    pub average_energy: Option<f64>,
    /// Mean current live-agent health; unavailable when the population is empty.
    pub average_health: Option<f64>,
    /// Mean live-agent age counter, advanced at the configured aging cadence.
    /// Unavailable when the population is empty.
    pub mean_age: Option<f64>,
    /// Largest live-agent age counter; unavailable when the population is empty.
    pub max_age: Option<u32>,
    /// Largest living lineage generation; unavailable when the population is empty.
    pub max_generation: Option<u32>,
    /// Sum of the current food grid's cell energy values.
    pub total_food: f64,
}

impl ExperimentEndpointSummaryV2 {
    fn from_world(
        world: &WorldState,
        seed: u64,
        brain_family: &str,
        final_digest: String,
    ) -> Result<Self, String> {
        let arena = world.agents();
        let alive_agents = u32::try_from(arena.len())
            .map_err(|_| "final population exceeds the u32 endpoint representation".to_owned())?;
        let columns = arena.columns();
        let mut ordered = arena
            .iter_handles()
            .enumerate()
            .map(|(index, id)| {
                world
                    .agent_uid(id)
                    .map(|uid| (uid, index, id))
                    .ok_or_else(|| {
                        format!("endpoint agent at dense index {index} has no stable UID")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        ordered.sort_unstable_by_key(|(uid, _, _)| *uid);
        if ordered.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err("endpoint population contains duplicate stable UIDs".to_owned());
        }
        let mut total_energy = 0.0;
        let mut total_health = 0.0;
        let mut total_age = 0.0;
        let mut max_age = None;
        let mut max_generation = None;
        for (uid, index, id) in ordered {
            let runtime = world
                .agent_runtime(id)
                .ok_or_else(|| format!("endpoint agent UID {uid} has no runtime state"))?;
            let energy = f64::from(runtime.energy);
            let health = f64::from(columns.health()[index]);
            if !energy.is_finite() || !health.is_finite() {
                return Err(format!(
                    "endpoint agent UID {uid} has non-finite energy or health"
                ));
            }
            let age = columns.ages()[index];
            let generation = columns.generations()[index].0;
            total_energy += energy;
            total_health += health;
            total_age += f64::from(age);
            max_age = Some(max_age.map_or(age, |previous: u32| previous.max(age)));
            max_generation =
                Some(max_generation.map_or(generation, |previous: u32| previous.max(generation)));
        }
        let mut total_food = 0.0;
        for (index, &food) in world.food().cells().iter().enumerate() {
            if !food.is_finite() {
                return Err(format!("endpoint food cell {index} has non-finite energy"));
            }
            total_food += f64::from(food);
        }
        if [total_energy, total_health, total_age, total_food]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err("final endpoint reduction produced a non-finite total".to_owned());
        }
        let population = (alive_agents != 0).then_some(f64::from(alive_agents));
        Ok(Self {
            tick: world.tick().0,
            alive_agents,
            seed,
            brain_family: brain_family.to_owned(),
            final_digest,
            schema_version: EXPERIMENT_SUMMARY_SCHEMA_VERSION,
            total_energy,
            average_energy: population.map(|count| total_energy / count),
            average_health: population.map(|count| total_health / count),
            mean_age: population.map(|count| total_age / count),
            max_age,
            max_generation,
            total_food,
        })
    }

    /// Available observed endpoints, under the same names offered by the Lab schema.
    #[must_use]
    pub fn available_metrics(&self) -> BTreeMap<String, f64> {
        let values: [Option<f64>; EXPERIMENT_ENDPOINT_METRICS.len()] = [
            Some(f64::from(self.alive_agents)),
            Some(self.total_energy),
            self.average_energy,
            self.average_health,
            self.mean_age,
            self.max_age.map(f64::from),
            self.max_generation.map(f64::from),
            Some(self.total_food),
        ];
        EXPERIMENT_ENDPOINT_METRICS
            .iter()
            .zip(values)
            .filter_map(|(name, value)| value.map(|value| ((*name).to_owned(), value)))
            .collect()
    }

    fn canonical_csv_bytes(&self) -> Result<Vec<u8>, String> {
        let mut writer = csv::WriterBuilder::new()
            .terminator(csv::Terminator::Any(b'\n'))
            .from_writer(Vec::new());
        writer
            .serialize(self)
            .map_err(|error| format!("serialize V2 endpoint summary: {error}"))?;
        let bytes = writer
            .into_inner()
            .map_err(|error| format!("finish V2 endpoint summary: {error}"))?;
        if bytes.len() > MAX_EXPERIMENT_SUMMARY_BYTES {
            return Err("V2 endpoint summary exceeds the 4096-byte artifact bound".to_owned());
        }
        Ok(bytes)
    }
}

/// Scientific observations computed from a restored final experiment checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedExperimentFinalState {
    /// Experiment-local identity read from the verified evidence and cut.
    pub run_id: String,
    /// Exact scientific boundary observed after restoration.
    pub tick: u64,
    /// Actual restored population, not the reported CSV value.
    pub alive_agents: usize,
    /// Scientific digest recomputed by the restored owner.
    pub scientific_digest: String,
    /// Canonical launch configuration digest from the verified run manifest.
    pub config_digest: String,
    /// Typed endpoints recomputed by that owner and matched to the exact CSV bytes.
    pub summary: ExperimentEndpointSummaryV2,
    /// BLAKE3 of the summary bytes actually checked against the restored state.
    pub summary_artifact_digest: String,
}

/// Artifact integrity and, when declared, the separately observed final science.
#[derive(Debug)]
pub struct PortableRunBundleVerification {
    /// Canonical byte/database verification result.
    pub artifacts: RunBundleVerificationResult,
    /// BLAKE3 of the exact manifest bytes selected for these semantic checks.
    pub bundle_manifest_digest: String,
    /// Restored observations only for a declared matched-seed experiment run.
    pub final_state: Option<VerifiedExperimentFinalState>,
}

/// Verify a portable bundle, restoring declared experiment final states offline.
///
/// This checks final-state consistency, not the whole cohort's plan, checkpoint
/// chain, source authenticity or completeness of its build provenance. Ordinary
/// database bundles retain the storage verifier's database/replay checks.
///
/// # Errors
/// Refuses invalid artifact bytes or experiment metadata, unsupported checkpoint
/// codecs/families, and reports that disagree with the restored scientific state.
pub fn verify_portable_run_bundle(
    bundle_dir: &Path,
) -> Result<PortableRunBundleVerification, ExperimentRunnerError> {
    let bytes =
        read_experiment_file_bounded(&bundle_dir.join("bundle_manifest.json"), MAX_STATUS_BYTES)?;
    let bundle: RunBundleV1 =
        serde_json::from_slice(&bytes).map_err(|source| ExperimentRunnerError::Json {
            operation: "decode portable bundle manifest",
            source,
        })?;
    let bundle_manifest_digest = blake3::hash(&bytes).to_hex().to_string();
    let manifest_value: serde_json::Value = serde_json::from_str(&bundle.manifest.manifest_json)
        .map_err(|source| ExperimentRunnerError::Json {
            operation: "decode embedded portable manifest",
            source,
        })?;
    let declares_experiment = bundle.artifacts.iter().any(|entry| {
        matches!(
            entry.artifact_type.as_str(),
            "experiment-summary" | "experiment-run-evidence" | "experiment-final-checkpoint"
        ) || matches!(
            entry.relative_path.as_str(),
            "evidence/run.json" | "checkpoints/final.checkpoint"
        )
    }) || manifest_value
        .pointer("/scenario/population_recipe")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|recipe| recipe.starts_with("fixed-4x4-registered-brain-grid-v1;brain="));
    let artifacts = if declares_experiment {
        verify_run_bundle_bounded(
            bundle_dir,
            RunBundleVerificationLimits {
                max_manifest_bytes: MAX_STATUS_BYTES,
                max_artifacts: MAX_BUNDLE_ARTIFACTS,
                max_artifact_bytes: MAX_BUNDLE_ARTIFACT_BYTES,
                max_total_artifact_bytes: MAX_BUNDLE_TOTAL_ARTIFACT_BYTES,
            },
        )
    } else {
        scriptbots_storage::verify_run_bundle(bundle_dir)
    }
    .map_err(|error| ExperimentRunnerError::Bundle {
        run_id: bundle.digests.run_id.clone(),
        reason: format!("canonical artifact verification failed: {error}"),
    })?;
    if !declares_experiment {
        return Ok(PortableRunBundleVerification {
            artifacts,
            bundle_manifest_digest,
            final_state: None,
        });
    }
    for &(path, role) in REQUIRED_EXPERIMENT_ARTIFACTS {
        if bundle
            .artifacts
            .iter()
            .filter(|entry| entry.relative_path == path && entry.artifact_type == role)
            .count()
            != 1
        {
            return Err(ExperimentRunnerError::Bundle {
                run_id: bundle.digests.run_id.clone(),
                reason: format!("experiment bundle must index `{path}` exactly once as `{role}`"),
            });
        }
    }
    let manifest: RunManifestV3 =
        serde_json::from_value(manifest_value).map_err(|source| ExperimentRunnerError::Json {
            operation: "decode experiment run manifest",
            source,
        })?;
    let evidence: RunEvidenceV2 = serde_json::from_slice(&read_experiment_file_bounded(
        &bundle_dir.join("evidence/run.json"),
        MAX_STATUS_BYTES,
    )?)
    .map_err(|source| ExperimentRunnerError::Json {
        operation: "decode experiment run evidence",
        source,
    })?;
    let selector = ScenarioVariant {
        variant_id: evidence.variant_id.clone(),
        brain_family: evidence.brain_family.clone(),
        config_overrides: BTreeMap::new(),
        interventions: Vec::new(),
    };
    let family = ResolvedBrainFamily::resolve(&selector)?;
    let fail = |reason: &str| ExperimentRunnerError::Bundle {
        run_id: evidence.run_id.clone(),
        reason: reason.to_owned(),
    };
    if evidence.schema != "scriptbots.experiment-run-evidence.v2"
        || evidence.summary_schema_version != EXPERIMENT_SUMMARY_SCHEMA_VERSION
        || manifest.identity.run_id != MatchedSeedExperimentRunner::bundle_run_id(&evidence.run_id)
        || manifest.identity.experiment_id.as_deref() != Some(evidence.experiment_id.as_str())
        || manifest.identity.variant_id.as_deref() != Some(evidence.variant_id.as_str())
        || manifest.identity.requested_tick_budget != Some(evidence.total_ticks)
        || manifest.identity.live_run_policy.is_some()
        || manifest.root_seed != evidence.seed
        || manifest.scenario.id != evidence.run_id
        || manifest.scenario.population_recipe
            != format!(
                "fixed-4x4-registered-brain-grid-v1;brain={}",
                family.canonical_name()
            )
        || manifest.brain_roster.len() != 1
        || manifest.brain_roster[0].kind != family.canonical_name()
        || evidence.brain_family != family.canonical_name()
        || bundle.digests.max_tick != evidence.total_ticks
    {
        return Err(fail(
            "experiment evidence differs from its run manifest or exact boundary",
        ));
    }
    let projected = manifest
        .to_storage_record()
        .map_err(|error| fail(&error.to_string()))?;
    let canonical = manifest
        .canonical_json_bytes()
        .map_err(|error| fail(&error.to_string()))?;
    if projected != bundle.manifest
        || evidence.manifest_digest != blake3::hash(&canonical).to_hex().as_str()
    {
        return Err(fail("experiment manifest projection or digest differs"));
    }
    let path = bundle_dir.join(FINAL_CHECKPOINT_ARTIFACT.0);
    let checkpoint = RunCheckpointV1::decode(
        &read_experiment_file_bounded(&path, MAX_RUN_CHECKPOINT_BYTES)?,
        &path,
    )?;
    let final_state =
        verify_restored_final_checkpoint(bundle_dir, &checkpoint, &manifest, &evidence, family)?;
    Ok(PortableRunBundleVerification {
        artifacts,
        bundle_manifest_digest,
        final_state: Some(final_state),
    })
}

fn verify_restored_final_checkpoint(
    bundle_dir: &Path,
    checkpoint: &RunCheckpointV1,
    manifest: &RunManifestV3,
    evidence: &RunEvidenceV2,
    family: ResolvedBrainFamily,
) -> Result<VerifiedExperimentFinalState, ExperimentRunnerError> {
    let path = bundle_dir.join(FINAL_CHECKPOINT_ARTIFACT.0);
    let host = checkpoint.owner.host_checkpoint();
    let initial_config: ScriptBotsConfig =
        serde_json::from_value(manifest.normalized_config.clone())
            .map_err(|error| invalid_checkpoint(&path, error))?;
    initial_config
        .validate()
        .map_err(|error| invalid_checkpoint(&path, error))?;
    let interventions = PreparedInterventions::new(
        &evidence.variant_id,
        &initial_config,
        &manifest.scenario.interventions,
        evidence.total_ticks,
    )
    .map_err(|error| invalid_checkpoint(&path, error))?;
    if checkpoint.metadata.manifest != *manifest
        || checkpoint.metadata.run_id != evidence.run_id
        || checkpoint.metadata.plan_digest != evidence.plan_digest
        || host.tick().0 != evidence.total_ticks
        || host.session_id() != host_identifiers(&evidence.run_id).0
        || initial_config.rng_seed != Some(evidence.seed)
        || initial_config.persistence_interval != 1
        || host.playback()
            != (PlaybackSnapshot {
                paused: true,
                speed_multiplier: 1.0,
            })
        || (host.tick().0 == 0) != checkpoint.metadata.parent_digest.is_none()
        || checkpoint
            .metadata
            .parent_digest
            .as_ref()
            .is_some_and(|digest| {
                digest.len() != blake3::OUT_LEN * 2
                    || !digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            })
    {
        return Err(invalid_checkpoint(
            &path,
            "final cut differs from the bundle manifest, evidence, session or exact completion tick",
        ));
    }
    validate_checkpoint_interventions(&path, host, &interventions)?;
    let core = restore_run_owner(checkpoint, family, &evidence.run_id, &path)?;
    let digest = core
        .scientific_digest_v1()
        .map_err(|error| invalid_checkpoint(&path, error))?;
    if digest.overall != evidence.final_digest {
        return Err(invalid_checkpoint(
            &path,
            "computed scientific digest differs from the reported result",
        ));
    }
    let summary_path = bundle_dir.join("exports/summary.csv");
    let summary_bytes = read_experiment_file_bounded(
        &summary_path,
        u64::try_from(MAX_EXPERIMENT_SUMMARY_BYTES).unwrap_or(u64::MAX),
    )?;
    let summary = ExperimentEndpointSummaryV2::from_world(
        core.world(),
        evidence.seed,
        family.canonical_name(),
        digest.overall.clone(),
    )
    .map_err(|error| invalid_checkpoint(&summary_path, error))?;
    let expected_summary = summary
        .canonical_csv_bytes()
        .map_err(|error| invalid_checkpoint(&summary_path, error))?;
    if summary_bytes != expected_summary {
        return Err(invalid_checkpoint(
            &summary_path,
            "V2 endpoint summary differs from the restored final state, schema, family, seed or digest",
        ));
    }
    Ok(VerifiedExperimentFinalState {
        run_id: evidence.run_id.clone(),
        tick: core.world_tick().0,
        alive_agents: core.world().agents().len(),
        scientific_digest: digest.overall,
        config_digest: manifest.config_digest.clone(),
        summary,
        summary_artifact_digest: blake3::hash(&summary_bytes).to_hex().to_string(),
    })
}

/// Deterministic matched-seed experiment runner.
pub struct MatchedSeedExperimentRunner {
    pub experiment_id: String,
    pub cohort: MatchedSeedCohort,
    pub variants: Vec<ScenarioVariant>,
    pub max_ticks: u64,
    pub max_concurrency: usize,
    pub output_dir: PathBuf,
    checkpoint_progress: Mutex<CheckpointProgressCache>,
}

impl MatchedSeedExperimentRunner {
    pub fn new(
        experiment_id: impl Into<String>,
        cohort: MatchedSeedCohort,
        variants: Vec<ScenarioVariant>,
        max_ticks: u64,
        max_concurrency: usize,
        output_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            experiment_id: experiment_id.into(),
            cohort,
            variants,
            max_ticks,
            max_concurrency,
            output_dir: output_dir.into(),
            checkpoint_progress: Mutex::new(CheckpointProgressCache::default()),
        }
    }

    fn progress_cache(&self, plan: &ValidatedPlan) -> MutexGuard<'_, CheckpointProgressCache> {
        let mut cache = self
            .checkpoint_progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cache.plan_digest != plan.digest || cache.output_dir != self.output_dir {
            *cache = CheckpointProgressCache {
                plan_digest: plan.digest.clone(),
                output_dir: self.output_dir.clone(),
                runs: plan
                    .runs
                    .iter()
                    .map(|run| (run.run_id.clone(), None))
                    .collect(),
                recovering_run: None,
            };
        }
        cache
    }

    fn prepare_progress_for_execution(&self, plan: &ValidatedPlan) {
        let mut cache = self.progress_cache(plan);
        // An explicit resume retries unavailable/refused observations. Preserve
        // earlier observed cuts until the owner publishes a verified successor.
        for progress in cache.runs.values_mut() {
            if !matches!(
                progress,
                Some(ExperimentCheckpointProgress::Observed { .. })
            ) {
                *progress = None;
            }
        }
    }

    fn observe_checkpoint_progress(
        &self,
        plan_digest: &str,
        run_id: &str,
        tick: u64,
        checkpoint_digest: String,
        source: CheckpointProgressSource,
    ) {
        let mut cache = self
            .checkpoint_progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cache.plan_digest != plan_digest || cache.output_dir != self.output_dir {
            return;
        }
        let Some(progress) = cache.runs.get_mut(run_id) else {
            return;
        };
        if matches!(
            progress,
            Some(ExperimentCheckpointProgress::Refused {
                reason: CheckpointProgressRefusal::ConflictingObservation,
            })
        ) {
            // A contradictory same-tick observation cannot erase the previous
            // high-water mark and then admit an older cut. Explicit resume must
            // revalidate the committed owner before observations can restart.
            return;
        }
        if let Some(ExperimentCheckpointProgress::Observed {
            tick: previous,
            checkpoint_digest: previous_digest,
            ..
        }) = progress
        {
            if tick < *previous {
                return;
            }
            if tick == *previous {
                if checkpoint_digest != *previous_digest {
                    *progress = Some(ExperimentCheckpointProgress::Refused {
                        reason: CheckpointProgressRefusal::ConflictingObservation,
                    });
                }
                return;
            }
        }
        *progress = Some(ExperimentCheckpointProgress::Observed {
            tick,
            checkpoint_digest,
            source,
        });
    }

    /// Observe bounded checkpoint progress without changing status or science.
    /// Active polls only read the latest-value cache. Cold stopped jobs recover
    /// one committed cut per call, with one recovery in flight across readers;
    /// valid observations and typed refusals are cached until explicit resume.
    pub(crate) fn checkpoint_progress_snapshot(
        &self,
        state_file: &Path,
        status: &ExperimentBatchStatus,
        running: bool,
    ) -> Result<BTreeMap<String, ExperimentCheckpointProgress>, ExperimentRunnerError> {
        let plan = self.validated_plan()?;
        self.validate_status(state_file, status, &plan)?;
        let recover = {
            let mut cache = self.progress_cache(&plan);
            let candidate = if !running && cache.recovering_run.is_none() {
                // Suspended runs precede terminal ones so large completed
                // cohorts cannot hide the remaining live work after restart.
                [RunState::Running, RunState::Failed, RunState::Completed]
                    .into_iter()
                    .find_map(|state| {
                        status
                            .runs
                            .iter()
                            .find(|run| run.state == state && cache.runs[&run.run_id].is_none())
                    })
                    .cloned()
            } else {
                None
            };
            if let Some(run) = &candidate {
                cache.recovering_run = Some(run.run_id.clone());
                cache.runs.insert(
                    run.run_id.clone(),
                    Some(ExperimentCheckpointProgress::Unavailable {
                        reason: CheckpointProgressUnavailable::RecoveryPending,
                    }),
                );
            }
            candidate
        };
        if let Some(run) = recover {
            let path = self.run_checkpoint_path(&run.run_id);
            let recovered = self.recover_checkpoint_progress(&path, &run, &plan);
            let mut cache = self.progress_cache(&plan);
            if cache.recovering_run.as_deref() == Some(run.run_id.as_str()) {
                cache.recovering_run = None;
                let entry = cache.runs.get_mut(&run.run_id).expect("admitted run");
                // A runner may have started and published a newer observation
                // while this read was in flight. Never replace that publication.
                if matches!(
                    entry,
                    Some(ExperimentCheckpointProgress::Unavailable { .. }) | None
                ) {
                    *entry = Some(recovered);
                }
            }
        }
        let cache = self.progress_cache(&plan);
        Ok(status
            .runs
            .iter()
            .map(|run| {
                let observation = if run.state == RunState::Pending {
                    ExperimentCheckpointProgress::Unavailable {
                        reason: CheckpointProgressUnavailable::NotStarted,
                    }
                } else {
                    cache.runs[&run.run_id].clone().unwrap_or(
                        ExperimentCheckpointProgress::Unavailable {
                            reason: if running {
                                CheckpointProgressUnavailable::AwaitingPublication
                            } else {
                                CheckpointProgressUnavailable::RecoveryPending
                            },
                        },
                    )
                };
                (run.run_id.clone(), observation)
            })
            .collect())
    }

    fn recover_checkpoint_progress(
        &self,
        path: &Path,
        run: &RunRecord,
        plan: &ValidatedPlan,
    ) -> ExperimentCheckpointProgress {
        let read = || -> Result<Option<ExperimentCheckpointProgress>, ExperimentRunnerError> {
            let Some(checkpoint) = read_optional_run_checkpoint(path)? else {
                return Ok(None);
            };
            let (variant, family) = &plan.variants[&run.variant_id];
            self.validate_run_checkpoint(
                path,
                &checkpoint,
                variant,
                run.seed,
                &run.run_id,
                &plan.digest,
            )?;
            // Framing and metadata are insufficient: verify the actual executable
            // world/owner/journal cut without advancing a tick or promoting a tmp.
            let owner = restore_run_owner(&checkpoint, *family, &run.run_id, path)?;
            let tick = owner.world_tick().0;
            if run.state == RunState::Completed && tick != run.total_ticks {
                return Err(invalid_checkpoint(
                    path,
                    "completed status and cut ticks differ",
                ));
            }
            Ok(Some(ExperimentCheckpointProgress::Observed {
                tick,
                checkpoint_digest: blake3::hash(&checkpoint.encode(path)?).to_hex().to_string(),
                source: CheckpointProgressSource::Recovered,
            }))
        };
        match read() {
            Ok(Some(observation)) => observation,
            Ok(None) => ExperimentCheckpointProgress::Unavailable {
                reason: CheckpointProgressUnavailable::NoCommittedCheckpoint,
            },
            Err(error) => {
                tracing::warn!(
                    experiment_id = %self.experiment_id,
                    run_id = %run.run_id,
                    %error,
                    "experiment checkpoint progress refused"
                );
                ExperimentCheckpointProgress::Refused {
                    reason: if matches!(error, ExperimentRunnerError::Io { .. }) {
                        CheckpointProgressRefusal::UnreadableCheckpoint
                    } else {
                        CheckpointProgressRefusal::InvalidCheckpoint
                    },
                }
            }
        }
    }

    fn publish_run_checkpoint(
        &self,
        path: &Path,
        checkpoint: &RunCheckpointV1,
        bytes: &[u8],
    ) -> Result<String, ExperimentRunnerError> {
        write_run_checkpoint(path, bytes)?;
        let digest = blake3::hash(bytes).to_hex().to_string();
        self.observe_checkpoint_progress(
            &checkpoint.metadata.plan_digest,
            &checkpoint.metadata.run_id,
            checkpoint.owner.host_checkpoint().tick().0,
            digest.clone(),
            CheckpointProgressSource::Published,
        );
        Ok(digest)
    }

    /// Run a canonical CLI worker with a wall-clock bound and reap every exit path.
    /// Output is inherited, never buffered without a size bound.
    ///
    /// # Errors
    /// Returns spawn, wait, timeout, or unsuccessful-exit errors.
    pub fn run_bounded_child(
        command: &mut std::process::Command,
        timeout: std::time::Duration,
    ) -> std::io::Result<()> {
        use std::io::{Error, ErrorKind};
        command.stdin(std::process::Stdio::null());
        let mut child = command.spawn()?;
        let started = std::time::Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    return if status.success() {
                        Ok(())
                    } else {
                        Err(Error::other(format!(
                            "canonical child exit={status} timeout=false"
                        )))
                    };
                }
                Ok(None) if started.elapsed() < timeout => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                result => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(match result {
                        Err(error) => error,
                        _ => Error::new(
                            ErrorKind::TimedOut,
                            "canonical child timeout=true; killed and reaped",
                        ),
                    });
                }
            }
        }
    }

    /// Generates initial pending run records for every variant and seed combination.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal before any filesystem mutation when the plan is
    /// ambiguous, unsupported, or exceeds the runner's explicit resource bounds.
    pub fn plan_batch(&self) -> Result<Vec<RunRecord>, ExperimentRunnerError> {
        self.validated_plan().map(|plan| plan.runs)
    }

    /// Dense grid cells reserved by the largest executable concurrent run set.
    ///
    /// The same preflight guards initial admission, status recovery and resume.
    /// Service callers reserve this amount across batches as well as run slots.
    pub(crate) fn concurrent_grid_cells(&self) -> Result<u64, ExperimentRunnerError> {
        self.validated_plan().map(|plan| plan.concurrent_grid_cells)
    }

    /// Bind a saved definition to the same source-aware plan as status and cuts.
    pub(crate) fn plan_digest(&self) -> Result<String, ExperimentRunnerError> {
        self.validated_plan().map(|plan| plan.digest)
    }

    /// Loads existing status file if present, or initializes a new batch plan.
    ///
    /// # Errors
    ///
    /// Recovers a complete interrupted publication only when it is the exact
    /// initial plan or a valid immediate successor of the committed generation.
    /// Malformed, stale, or plan-incompatible state never permits a tick-zero restart.
    pub fn load_or_create_status(
        &self,
        state_file: &Path,
    ) -> Result<ExperimentBatchStatus, ExperimentRunnerError> {
        let plan = self.validated_plan()?;
        let _lease = ExperimentStatusWriterLease::acquire(state_file)?;
        self.load_or_create_status_locked(state_file, &plan)
    }

    /// Read one bounded, plan-validated snapshot without writing or taking the
    /// batch's writer lease. Before the first atomic publication, expose the
    /// actual initial plan rather than an invented schema or empty identity.
    pub(crate) fn status_snapshot(
        &self,
        state_file: &Path,
    ) -> Result<ExperimentBatchStatus, ExperimentRunnerError> {
        let plan = self.validated_plan()?;
        match self.load_existing_status(state_file, &plan) {
            Ok(status) => Ok(status),
            Err(ExperimentRunnerError::Io { source, .. })
                if source.kind() == io::ErrorKind::NotFound =>
            {
                Ok(self.initial_status(&plan))
            }
            Err(error) => Err(error),
        }
    }

    /// Verify every completed member of an already-read status before exposing
    /// its artifacts. This is read-only and retains the runner's complete
    /// plan/family/config/evidence checks rather than only bundle byte hashes.
    pub(crate) fn verify_completed_status(
        &self,
        state_file: &Path,
        status: &ExperimentBatchStatus,
    ) -> Result<(), ExperimentRunnerError> {
        let plan = self.validated_plan()?;
        self.validate_status(state_file, status, &plan)?;
        for record in status
            .runs
            .iter()
            .filter(|run| run.state == RunState::Completed)
        {
            let (variant, family) = &plan.variants[&record.variant_id];
            self.verify_completed_bundle(record, variant, *family, &plan.digest)?;
        }
        Ok(())
    }

    /// Restore both complete variant cohorts from one plan-validated status.
    ///
    /// Every call rechecks the bounded bundle bytes, run/plan/family/config
    /// identities and final owner state. Published artifact-cache membership is
    /// not evidence for a later analysis. Results are ordered by the exact seed
    /// schedule, independently of the order in which workers completed.
    pub(crate) fn verified_comparison_endpoints(
        &self,
        state_file: &Path,
        status: &ExperimentBatchStatus,
        baseline_variant: &str,
        challenger_variant: &str,
    ) -> Result<[Vec<VerifiedExperimentFinalState>; 2], ExperimentRunnerError> {
        if baseline_variant == challenger_variant {
            return Err(ExperimentRunnerError::InvalidPlan {
                reason: "comparison requires two distinct variants".to_owned(),
            });
        }
        let plan = self.validated_plan()?;
        self.validate_status(state_file, status, &plan)?;
        let mut expected_seeds = self.cohort.seeds.clone();
        expected_seeds.sort_unstable();
        let mut endpoints = [Vec::new(), Vec::new()];
        for (arm, variant_id) in [baseline_variant, challenger_variant]
            .into_iter()
            .enumerate()
        {
            let (variant, family) = plan.variants.get(variant_id).ok_or_else(|| {
                ExperimentRunnerError::InvalidPlan {
                    reason: format!("comparison variant `{variant_id}` is not in this batch"),
                }
            })?;
            let records = status
                .runs
                .iter()
                .filter(|run| run.variant_id == variant_id)
                .collect::<Vec<_>>();
            if let Some(run) = records.iter().find(|run| run.state != RunState::Completed) {
                return Err(ExperimentRunnerError::InvalidStatus {
                    path: state_file.to_path_buf(),
                    reason: format!(
                        "comparison run `{}` is {:?}; every declared seed must be completed",
                        run.run_id, run.state
                    ),
                });
            }
            for record in records {
                endpoints[arm].push(self.verify_completed_bundle(
                    record,
                    variant,
                    *family,
                    &plan.digest,
                )?);
            }
            endpoints[arm].sort_by_key(|state| state.summary.seed);
            if endpoints[arm]
                .iter()
                .map(|state| state.summary.seed)
                .ne(expected_seeds.iter().copied())
            {
                return Err(ExperimentRunnerError::InvalidStatus {
                    path: state_file.to_path_buf(),
                    reason: format!(
                        "comparison variant `{variant_id}` does not contain the exact declared seed cohort"
                    ),
                });
            }
        }
        Ok(endpoints)
    }

    fn initial_status(&self, plan: &ValidatedPlan) -> ExperimentBatchStatus {
        ExperimentBatchStatus {
            schema_version: STATUS_SCHEMA_VERSION,
            generation: 0,
            plan_digest: plan.digest.clone(),
            experiment_id: self.experiment_id.clone(),
            total_runs: plan.runs.len(),
            completed_runs: 0,
            failed_runs: 0,
            runs: plan.runs.clone(),
        }
    }

    fn load_or_create_status_locked(
        &self,
        state_file: &Path,
        plan: &ValidatedPlan,
    ) -> Result<ExperimentBatchStatus, ExperimentRunnerError> {
        let temporary = status_temporary_path(state_file)?;
        match fs::symlink_metadata(&temporary) {
            Ok(_) => return self.recover_status_publication_locked(state_file, &temporary, plan),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(ExperimentRunnerError::Io {
                    operation: "inspect interrupted status publication",
                    path: temporary,
                    source,
                });
            }
        }

        match fs::symlink_metadata(state_file) {
            Ok(_) => return self.load_existing_status(state_file, plan),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(ExperimentRunnerError::Io {
                    operation: "inspect experiment status",
                    path: state_file.to_path_buf(),
                    source,
                });
            }
        }

        let status = self.initial_status(plan);
        self.write_status_atomically(state_file, &status, plan)?;
        Ok(status)
    }

    /// Recover only a complete writer-produced successor while holding the status lease.
    fn recover_status_publication_locked(
        &self,
        state_file: &Path,
        temporary: &Path,
        plan: &ValidatedPlan,
    ) -> Result<ExperimentBatchStatus, ExperimentRunnerError> {
        let interrupted = || ExperimentRunnerError::InterruptedStatusWrite {
            temporary: temporary.to_path_buf(),
            committed: state_file.to_path_buf(),
        };
        let bytes = read_experiment_file_bounded(temporary, MAX_STATUS_BYTES)?;
        let candidate =
            serde_json::from_slice::<ExperimentBatchStatus>(&bytes).map_err(|_| interrupted())?;
        self.validate_status(temporary, &candidate, plan)?;
        if canonical_status_bytes(&candidate)? != bytes {
            return Err(interrupted());
        }

        let committed_bytes = match fs::symlink_metadata(state_file) {
            Ok(_) => Some(read_experiment_file_bounded(state_file, MAX_STATUS_BYTES)?),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(source) => {
                return Err(ExperimentRunnerError::Io {
                    operation: "inspect committed status for recovery",
                    path: state_file.to_path_buf(),
                    source,
                });
            }
        };
        if let Some(bytes) = &committed_bytes {
            let committed = self.decode_status(state_file, bytes, plan)?;
            if committed.generation.checked_add(1) != Some(candidate.generation) {
                return Err(interrupted());
            }
            self.validate_status_transition(state_file, &committed, &candidate)?;
        } else if candidate != self.initial_status(plan) {
            return Err(interrupted());
        }

        // Completion is scientific evidence, not merely a monotonic enum transition.
        // Verify the actual bundle and restored owner cut before publishing success.
        for record in candidate
            .runs
            .iter()
            .filter(|record| record.state == RunState::Completed)
        {
            let (variant, family) = &plan.variants[&record.variant_id];
            self.verify_completed_bundle(record, variant, *family, &plan.digest)?;
        }

        // A cooperating writer cannot change these paths under our lease. Refuse
        // external replacement too, rather than overwriting evidence read above.
        if read_experiment_file_bounded(temporary, MAX_STATUS_BYTES)? != bytes {
            return Err(interrupted());
        }
        match (&committed_bytes, fs::symlink_metadata(state_file)) {
            (Some(expected), Ok(_)) => {
                if read_experiment_file_bounded(state_file, MAX_STATUS_BYTES)? != *expected {
                    return Err(interrupted());
                }
            }
            (None, Err(error)) if error.kind() == io::ErrorKind::NotFound => {}
            _ => return Err(interrupted()),
        }
        let mut options = OpenOptions::new();
        options.read(true).write(cfg!(windows));
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        options
            .open(temporary)
            .and_then(|file| file.sync_all())
            .map_err(|source| ExperimentRunnerError::Io {
                operation: "sync recovered experiment status",
                path: temporary.to_path_buf(),
                source,
            })?;
        fs::rename(temporary, state_file).map_err(|source| ExperimentRunnerError::Io {
            operation: "promote recovered experiment status",
            path: state_file.to_path_buf(),
            source,
        })?;
        sync_parent_directory(
            state_file
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new(".")),
        )?;
        Ok(candidate)
    }

    /// Atomically and durably flushes a validated status update to disk.
    ///
    /// # Errors
    ///
    /// Refuses state that is inconsistent with the current plan or an unresolved
    /// temporary file from an interrupted prior write.
    pub fn save_status(
        &self,
        state_file: &Path,
        status: &mut ExperimentBatchStatus,
    ) -> Result<(), ExperimentRunnerError> {
        let plan = self.validated_plan()?;
        let _lease = ExperimentStatusWriterLease::acquire(state_file)?;
        self.save_status_transition_locked(state_file, status, &plan)
    }

    /// Derive the bundle's stable `RunId` from the runner's human-readable run identity.
    ///
    /// The experiment runner keys runs by `<experiment>-<variant>-seed<N>`, while the run
    /// bundle manifest is keyed by `RunId`. BLAKE3 over that exact string keeps the
    /// mapping deterministic and collision-resistant, and the readable identity is still
    /// carried in the manifest's `experiment_id`, `variant_id`, and `scenario_id`.
    pub(crate) fn bundle_run_id(run_id: &str) -> RunId {
        let digest = blake3::hash(run_id.as_bytes());
        let mut leading = [0_u8; 16];
        leading.copy_from_slice(&digest.as_bytes()[..16]);
        RunId::new(u128::from_be_bytes(leading).max(1))
    }

    /// Executes one validated family arm through the canonical HostCore/HostClient boundary.
    ///
    /// # Errors
    ///
    /// Returns a typed construction, host-receipt, artifact, or verification failure.
    pub fn execute_single_run(
        &self,
        variant: &ScenarioVariant,
        seed: u64,
    ) -> Result<RunRecord, ExperimentRunnerError> {
        validate_path_component("experiment_id", &self.experiment_id)?;
        validate_path_component("variant_id", &variant.variant_id)?;
        let family = ResolvedBrainFamily::resolve(variant)?;
        let plan = self.validated_plan()?;
        let run_id = format!("{}-{}-seed{}", self.experiment_id, variant.variant_id, seed);
        validate_path_component("run_id", &run_id)?;
        let expected = plan
            .runs
            .iter()
            .find(|record| record.run_id == run_id)
            .ok_or_else(|| ExperimentRunnerError::InvalidPlan {
                reason: format!(
                    "run `{run_id}` is not present in the validated matched-seed schedule"
                ),
            })?;
        let (planned_variant, planned_family) =
            plan.variants.get(&variant.variant_id).ok_or_else(|| {
                ExperimentRunnerError::InvalidPlan {
                    reason: format!(
                        "variant `{}` is not present in the validated experiment plan",
                        variant.variant_id
                    ),
                }
            })?;
        if planned_variant != variant
            || *planned_family != family
            || expected.brain_family != family.canonical_name()
        {
            return Err(ExperimentRunnerError::InvalidPlan {
                reason: format!(
                    "run `{run_id}` does not exactly match its validated variant and family `{}`",
                    family.canonical_name(),
                ),
            });
        }
        self.prepare_progress_for_execution(&plan);
        self.prepare_run_checkpoint(
            planned_variant,
            *planned_family,
            seed,
            &run_id,
            &plan.digest,
        )?;
        self.execute_validated_run(
            planned_variant,
            *planned_family,
            seed,
            &run_id,
            &plan.digest,
            RunExecutionBudget {
                cancel: &std::sync::atomic::AtomicBool::new(false),
                population_limit: MAX_RUN_POPULATION,
            },
        )
    }

    /// Executes the full batch using deterministic bounded scheduling and durable status.
    ///
    /// Failed runs become terminal status records so the caller can inspect the
    /// complete partial-failure set. An interrupted `Running` record can recover
    /// an already completed, verified bundle, or continue the next science tick
    /// from its atomically persisted owner/journal cut. Missing cuts are refused.
    /// Verified companions are saved even if another bundle fails verification;
    /// that failure is still returned before any pending runs are admitted.
    ///
    /// # Errors
    ///
    /// Returns a typed plan, recovery, durable-status, or bundle-verification failure.
    pub fn execute_batch(
        &self,
        state_file: &Path,
    ) -> Result<ExperimentBatchStatus, ExperimentRunnerError> {
        self.execute_batch_until_cancelled(state_file, &std::sync::atomic::AtomicBool::new(false))
    }

    /// [`Self::execute_batch`] with cooperative cancellation at durable tick boundaries.
    ///
    /// `cancel` is checked before admitting a wave and before each scientific tick.
    /// Active owners finish their current tick/checkpoint and shut down; their
    /// records remain `Running` with progress in the validated continuation cut.
    /// Unadmitted runs remain `Pending`. A later call resumes those exact cuts.
    /// This does not interrupt an executing host or filesystem operation.
    ///
    /// # Errors
    ///
    /// Same as [`Self::execute_batch`].
    pub fn execute_batch_until_cancelled(
        &self,
        state_file: &Path,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<ExperimentBatchStatus, ExperimentRunnerError> {
        self.execute_batch_with_budget(
            state_file,
            RunExecutionBudget {
                cancel,
                population_limit: MAX_RUN_POPULATION,
            },
        )
    }

    fn execute_batch_with_budget(
        &self,
        state_file: &Path,
        budget: RunExecutionBudget<'_>,
    ) -> Result<ExperimentBatchStatus, ExperimentRunnerError> {
        let plan = self.validated_plan()?;
        let _lease = ExperimentStatusWriterLease::acquire(state_file)?;
        let mut status = self.load_or_create_status_locked(state_file, &plan)?;
        self.prepare_progress_for_execution(&plan);

        let mut recovered = false;
        let mut recovery_error = None;
        for record in &mut status.runs {
            match record.state {
                RunState::Running => {
                    let (variant, family) = plan
                        .variants
                        .get(&record.variant_id)
                        .expect("validated status variant exists");
                    let completed =
                        self.recover_completed_run(record, variant, *family, &plan.digest);
                    let completed = match completed {
                        Ok(completed) => completed,
                        Err(ExperimentRunnerError::ResumeCheckpointUnavailable { .. }) => {
                            let resumed = ExperimentStatusWriterLease::acquire(
                                &self.run_checkpoint_path(&record.run_id),
                            )
                            .and_then(|_lease| {
                                self.load_run_checkpoint(
                                    variant,
                                    record.seed,
                                    &record.run_id,
                                    &plan.digest,
                                    budget.population_limit,
                                )
                            });
                            match resumed {
                                Ok(Some(_)) => continue,
                                Ok(None) => {
                                    recovery_error.get_or_insert(
                                        ExperimentRunnerError::ResumeCheckpointUnavailable {
                                            run_id: record.run_id.clone(),
                                            completed_ticks: record.total_ticks,
                                        },
                                    );
                                }
                                Err(error) => {
                                    recovery_error.get_or_insert(error);
                                }
                            }
                            continue;
                        }
                        Err(error) => {
                            recovery_error.get_or_insert(error);
                            continue;
                        }
                    };
                    *record = completed;
                    recovered = true;
                    info!(
                        experiment_id = %self.experiment_id,
                        run_id = %record.run_id,
                        tick = record.total_ticks,
                        digest = ?record.final_digest,
                        bundle_path = ?record.bundle_path,
                        "verified completed bundle for status recovery"
                    );
                }
                RunState::Completed => {
                    let (variant, family) =
                        plan.variants.get(&record.variant_id).ok_or_else(|| {
                            ExperimentRunnerError::InvalidStatus {
                                path: state_file.to_path_buf(),
                                reason: format!(
                                    "completed run `{}` names absent variant `{}`",
                                    record.run_id, record.variant_id
                                ),
                            }
                        })?;
                    if let Err(error) =
                        self.verify_completed_bundle(record, variant, *family, &plan.digest)
                    {
                        recovery_error.get_or_insert(error);
                    }
                }
                RunState::Pending | RunState::Failed => {}
            }
        }
        if recovered {
            recompute_status_counts(&mut status);
            self.save_status_transition_locked(state_file, &mut status, &plan)?;
        }
        if let Some(error) = recovery_error {
            return Err(error);
        }

        let pending = status
            .runs
            .iter()
            .enumerate()
            .filter_map(|(index, record)| {
                matches!(record.state, RunState::Pending | RunState::Running).then_some(index)
            })
            .collect::<Vec<_>>();

        for chunk in pending.chunks(self.max_concurrency) {
            if budget.cancel.load(std::sync::atomic::Ordering::Acquire) {
                info!(
                    experiment_id = %self.experiment_id,
                    "matched-seed batch cancelled before admitting the next wave"
                );
                break;
            }
            for &index in chunk {
                let record = &status.runs[index];
                if record.state == RunState::Pending {
                    let (variant, family) = plan
                        .variants
                        .get(&record.variant_id)
                        .expect("validated status variant exists");
                    self.prepare_run_checkpoint(
                        variant,
                        *family,
                        record.seed,
                        &record.run_id,
                        &plan.digest,
                    )?;
                }
                status.runs[index].state = RunState::Running;
                info!(
                    experiment_id = %self.experiment_id,
                    run_id = %status.runs[index].run_id,
                    variant_id = %status.runs[index].variant_id,
                    brain_family = %status.runs[index].brain_family,
                    seed = status.runs[index].seed,
                    scheduler_slot = index,
                    "matched-seed run admitted to bounded execution wave"
                );
            }
            recompute_status_counts(&mut status);
            self.save_status_transition_locked(state_file, &mut status, &plan)?;

            let results = std::thread::scope(|scope| {
                let mut workers = Vec::with_capacity(chunk.len());
                for &index in chunk {
                    let record = &status.runs[index];
                    let (variant, family) = plan
                        .variants
                        .get(&record.variant_id)
                        .expect("validated status variant exists");
                    let variant = variant.clone();
                    let family = *family;
                    let run_id = record.run_id.clone();
                    let plan_digest = plan.digest.clone();
                    let seed = record.seed;
                    workers.push((
                        index,
                        scope.spawn(move || {
                            self.execute_validated_run(
                                &variant,
                                family,
                                seed,
                                &run_id,
                                &plan_digest,
                                budget,
                            )
                        }),
                    ));
                }

                workers
                    .into_iter()
                    .map(|(index, worker)| {
                        let run_id = status.runs[index].run_id.clone();
                        let result = worker
                            .join()
                            .unwrap_or(Err(ExperimentRunnerError::WorkerPanicked { run_id }));
                        (index, result)
                    })
                    .collect::<Vec<_>>()
            });

            for (index, result) in results {
                match result {
                    Ok(record) => {
                        if record.state == RunState::Completed {
                            info!(
                                experiment_id = %self.experiment_id,
                                run_id = %record.run_id,
                                variant_id = %record.variant_id,
                                brain_family = %record.brain_family,
                                seed = record.seed,
                                tick = record.total_ticks,
                                digest = record.final_digest.as_deref().unwrap_or(""),
                                bundle_path = record.bundle_path.as_deref().unwrap_or(""),
                                scheduler_slot = index,
                                "matched-seed run completed and bundle reopened"
                            );
                        } else {
                            info!(
                                experiment_id = %self.experiment_id,
                                run_id = %record.run_id,
                                scheduler_slot = index,
                                "matched-seed run suspended with its durable continuation cut"
                            );
                        }
                        status.runs[index] = record;
                    }
                    Err(source) => {
                        error!(
                            experiment_id = %self.experiment_id,
                            run_id = %status.runs[index].run_id,
                            variant_id = %status.runs[index].variant_id,
                            brain_family = %status.runs[index].brain_family,
                            seed = status.runs[index].seed,
                            scheduler_slot = index,
                            error = %source,
                            "matched-seed run failed"
                        );
                        status.runs[index].state = RunState::Failed;
                        status.runs[index].total_ticks = 0;
                        status.runs[index].final_digest = None;
                        status.runs[index].bundle_path = None;
                        status.runs[index].error_reason = Some(source.to_string());
                    }
                }
            }

            recompute_status_counts(&mut status);
            self.save_status_transition_locked(state_file, &mut status, &plan)?;
        }

        Ok(status)
    }

    fn validated_plan(&self) -> Result<ValidatedPlan, ExperimentRunnerError> {
        validate_path_component("experiment_id", &self.experiment_id)?;
        validate_path_component("cohort_id", &self.cohort.cohort_id)?;
        if self.cohort.seeds.is_empty() {
            return Err(ExperimentRunnerError::InvalidPlan {
                reason: "the matched-seed cohort is empty".to_owned(),
            });
        }
        if self.variants.is_empty() {
            return Err(ExperimentRunnerError::InvalidPlan {
                reason: "the scenario-variant list is empty".to_owned(),
            });
        }
        if self.max_ticks > MAX_TICKS_PER_RUN {
            return Err(ExperimentRunnerError::InvalidPlan {
                reason: format!(
                    "max_ticks {} exceeds the per-run cap {MAX_TICKS_PER_RUN}",
                    self.max_ticks
                ),
            });
        }
        if self.max_concurrency == 0 || self.max_concurrency > MAX_CONCURRENT_RUNS {
            return Err(ExperimentRunnerError::InvalidPlan {
                reason: format!(
                    "max_concurrency must be in 1..={MAX_CONCURRENT_RUNS}, found {}",
                    self.max_concurrency
                ),
            });
        }
        if self
            .cohort
            .seeds
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != self.cohort.seeds.len()
        {
            return Err(ExperimentRunnerError::InvalidPlan {
                reason: "the matched-seed schedule contains a duplicate seed".to_owned(),
            });
        }

        let total_runs = self
            .cohort
            .seeds
            .len()
            .checked_mul(self.variants.len())
            .filter(|count| *count <= MAX_BATCH_RUNS)
            .ok_or_else(|| ExperimentRunnerError::InvalidPlan {
                reason: format!(
                    "variant x seed expansion exceeds the bounded {MAX_BATCH_RUNS}-run capacity"
                ),
            })?;

        let mut variants = BTreeMap::new();
        let mut run_grid_cells = Vec::with_capacity(total_runs);
        let mut semantic_arms = BTreeSet::new();
        for variant in &self.variants {
            validate_path_component("variant_id", &variant.variant_id)?;
            if variants.contains_key(&variant.variant_id) {
                return Err(ExperimentRunnerError::InvalidPlan {
                    reason: format!("duplicate variant_id `{}`", variant.variant_id),
                });
            }
            let family = ResolvedBrainFamily::resolve(variant)?;
            let normalized_config =
                config_for_run(&variant.config_overrides, 0).map_err(|reason| {
                    ExperimentRunnerError::InvalidPlan {
                        reason: format!(
                            "variant `{}` is not executable: {reason}",
                            variant.variant_id
                        ),
                    }
                })?;
            // Check the configured executable family before any owner, output or
            // worker exists. NeuroFlow's versioned envelope bounds are stricter
            // than the general world configuration's architecture bounds.
            drop(
                family
                    .adapter(&normalized_config, &variant.variant_id)
                    .map_err(|error| ExperimentRunnerError::InvalidPlan {
                        reason: format!(
                            "variant `{}` has an invalid configured brain family: {error}",
                            variant.variant_id
                        ),
                    })?,
            );
            // Use the constructor's validated dimensions, not a copied rounding
            // rule. WorldState::build uses this exact grid for food and terrain.
            let (width, height) = normalized_config.food_dimensions().map_err(|error| {
                ExperimentRunnerError::InvalidPlan {
                    reason: format!(
                        "variant `{}` has invalid grid dimensions: {error}",
                        variant.variant_id
                    ),
                }
            })?;
            let cells = u64::from(width)
                .checked_mul(u64::from(height))
                .ok_or_else(|| ExperimentRunnerError::InvalidPlan {
                    reason: format!("variant `{}` grid cell count overflows", variant.variant_id),
                })?;
            if cells > MAX_RUN_GRID_CELLS {
                return Err(ExperimentRunnerError::RunGridCapacityExceeded {
                    variant_id: variant.variant_id.clone(),
                    cells,
                    limit: MAX_RUN_GRID_CELLS,
                });
            }
            run_grid_cells.extend(std::iter::repeat_n(cells, self.cohort.seeds.len()));
            let interventions = PreparedInterventions::new(
                &variant.variant_id,
                &normalized_config,
                &variant.interventions,
                self.max_ticks,
            )?;
            let config_bytes = serde_json::to_vec(&(&normalized_config, &interventions.patches))
                .map_err(|source| ExperimentRunnerError::Json {
                    operation: "serialize normalized scenario arm",
                    source,
                })?;
            let semantic_key = (family, blake3::hash(&config_bytes).to_hex().to_string());
            if !semantic_arms.insert(semantic_key) {
                return Err(ExperimentRunnerError::InvalidPlan {
                    reason: format!(
                        "variant `{}` duplicates an existing `{}` arm after canonical config and intervention normalization",
                        variant.variant_id,
                        family.canonical_name()
                    ),
                });
            }
            variants.insert(variant.variant_id.clone(), (variant.clone(), family));
        }

        // Completed runs disappear from the resume queue, so its new waves can
        // combine arms that occupied different initial waves. Reserve the largest
        // possible concurrent subset instead of trusting the original chunking.
        run_grid_cells.sort_unstable();
        let concurrent_runs = self.max_concurrency.min(total_runs);
        let concurrent_grid_cells = run_grid_cells
            .iter()
            .rev()
            .take(concurrent_runs)
            .try_fold(0_u64, |total, cells| total.checked_add(*cells))
            .ok_or_else(|| ExperimentRunnerError::InvalidPlan {
                reason: "concurrent grid cell count overflows".to_owned(),
            })?;
        if concurrent_grid_cells > MAX_CONCURRENT_GRID_CELLS {
            return Err(ExperimentRunnerError::ConcurrentGridCapacityExceeded {
                concurrent_runs,
                cells: concurrent_grid_cells,
                limit: MAX_CONCURRENT_GRID_CELLS,
            });
        }

        let mut runs = Vec::with_capacity(total_runs);
        for variant in &self.variants {
            let (_, family) = variants
                .get(&variant.variant_id)
                .expect("validated variant was inserted");
            for &seed in &self.cohort.seeds {
                let run_id = format!("{}-{}-seed{}", self.experiment_id, variant.variant_id, seed);
                validate_path_component("run_id", &run_id)?;
                runs.push(RunRecord {
                    run_id,
                    variant_id: variant.variant_id.clone(),
                    brain_family: family.canonical_name().to_owned(),
                    seed,
                    state: RunState::Pending,
                    total_ticks: 0,
                    final_digest: None,
                    bundle_path: None,
                    error_reason: None,
                });
            }
        }

        let digest_input = serde_json::json!({
            "schema": "scriptbots.experiment-plan.v1",
            "experiment_id": self.experiment_id,
            "cohort": self.cohort,
            "variants": self.variants,
            "max_ticks": self.max_ticks,
            "build": BuildProvenanceV0::current(),
        });
        let digest_bytes =
            serde_json::to_vec(&digest_input).map_err(|source| ExperimentRunnerError::Json {
                operation: "serialize experiment plan",
                source,
            })?;

        Ok(ValidatedPlan {
            digest: blake3::hash(&digest_bytes).to_hex().to_string(),
            runs,
            variants,
            concurrent_grid_cells,
        })
    }

    fn validate_status(
        &self,
        state_file: &Path,
        status: &ExperimentBatchStatus,
        plan: &ValidatedPlan,
    ) -> Result<(), ExperimentRunnerError> {
        let invalid = |reason: String| ExperimentRunnerError::InvalidStatus {
            path: state_file.to_path_buf(),
            reason,
        };
        if status.schema_version != STATUS_SCHEMA_VERSION {
            return Err(invalid(format!(
                "schema_version {} is not supported version {STATUS_SCHEMA_VERSION}",
                status.schema_version
            )));
        }
        if status.plan_digest != plan.digest {
            return Err(invalid(format!(
                "plan digest {} does not match current {} (source, scenario, family, seed, or tick budget changed)",
                status.plan_digest, plan.digest
            )));
        }
        if status.experiment_id != self.experiment_id {
            return Err(invalid(format!(
                "experiment_id `{}` does not match `{}`",
                status.experiment_id, self.experiment_id
            )));
        }
        if status.total_runs != plan.runs.len() || status.runs.len() != plan.runs.len() {
            return Err(invalid(format!(
                "run cardinality {}/{} does not match planned {}",
                status.total_runs,
                status.runs.len(),
                plan.runs.len()
            )));
        }

        for (index, (actual, expected)) in status.runs.iter().zip(&plan.runs).enumerate() {
            if actual.run_id != expected.run_id
                || actual.variant_id != expected.variant_id
                || actual.brain_family != expected.brain_family
                || actual.seed != expected.seed
            {
                return Err(invalid(format!(
                    "run index {index} changed immutable run/variant/family/seed identity"
                )));
            }
            match actual.state {
                RunState::Pending | RunState::Running => {
                    if actual.total_ticks != 0
                        || actual.final_digest.is_some()
                        || actual.bundle_path.is_some()
                        || actual.error_reason.is_some()
                    {
                        return Err(invalid(format!(
                            "{} run `{}` carries terminal evidence",
                            if actual.state == RunState::Pending {
                                "pending"
                            } else {
                                "running"
                            },
                            actual.run_id
                        )));
                    }
                }
                RunState::Completed => {
                    let expected_bundle = self.output_dir.join(&actual.run_id);
                    if actual.total_ticks != self.max_ticks
                        || !actual
                            .final_digest
                            .as_deref()
                            .is_some_and(is_lower_hex_digest)
                        || actual.bundle_path.as_deref()
                            != Some(expected_bundle.to_string_lossy().as_ref())
                        || actual.error_reason.is_some()
                    {
                        return Err(invalid(format!(
                            "completed run `{}` has inconsistent tick, digest, bundle, or error evidence",
                            actual.run_id
                        )));
                    }
                }
                RunState::Failed => {
                    if actual.total_ticks != 0
                        || actual.final_digest.is_some()
                        || actual.bundle_path.is_some()
                        || !actual
                            .error_reason
                            .as_deref()
                            .is_some_and(|reason| !reason.trim().is_empty())
                    {
                        return Err(invalid(format!(
                            "failed run `{}` has inconsistent terminal evidence",
                            actual.run_id
                        )));
                    }
                }
            }
        }

        let completed = status
            .runs
            .iter()
            .filter(|record| record.state == RunState::Completed)
            .count();
        let failed = status
            .runs
            .iter()
            .filter(|record| record.state == RunState::Failed)
            .count();
        if status.completed_runs != completed || status.failed_runs != failed {
            return Err(invalid(format!(
                "derived counters completed={completed}, failed={failed} do not match recorded completed={}, failed={}",
                status.completed_runs, status.failed_runs
            )));
        }
        Ok(())
    }

    fn load_existing_status(
        &self,
        state_file: &Path,
        plan: &ValidatedPlan,
    ) -> Result<ExperimentBatchStatus, ExperimentRunnerError> {
        let bytes = read_bounded_file(state_file, MAX_STATUS_BYTES, "read experiment status")?;
        self.decode_status(state_file, &bytes, plan)
    }

    fn decode_status(
        &self,
        state_file: &Path,
        bytes: &[u8],
        plan: &ValidatedPlan,
    ) -> Result<ExperimentBatchStatus, ExperimentRunnerError> {
        let status = serde_json::from_slice::<ExperimentBatchStatus>(bytes).map_err(|source| {
            ExperimentRunnerError::InvalidStatus {
                path: state_file.to_path_buf(),
                reason: format!("status JSON does not match the versioned schema: {source}"),
            }
        })?;
        self.validate_status(state_file, &status, plan)?;
        Ok(status)
    }

    fn save_status_transition_locked(
        &self,
        state_file: &Path,
        status: &mut ExperimentBatchStatus,
        plan: &ValidatedPlan,
    ) -> Result<(), ExperimentRunnerError> {
        let committed = self.load_existing_status(state_file, plan)?;
        if status.generation != committed.generation {
            return Err(ExperimentRunnerError::StaleStatusGeneration {
                path: state_file.to_path_buf(),
                proposed: status.generation,
                committed: committed.generation,
            });
        }
        self.validate_status_transition(state_file, &committed, status)?;
        let mut candidate = status.clone();
        candidate.generation = candidate.generation.checked_add(1).ok_or_else(|| {
            ExperimentRunnerError::InvalidStatus {
                path: state_file.to_path_buf(),
                reason: "status generation exhausted u64".to_owned(),
            }
        })?;
        self.write_status_atomically(state_file, &candidate, plan)?;
        *status = candidate;
        Ok(())
    }

    fn validate_status_transition(
        &self,
        state_file: &Path,
        committed: &ExperimentBatchStatus,
        proposed: &ExperimentBatchStatus,
    ) -> Result<(), ExperimentRunnerError> {
        for (old, new) in committed.runs.iter().zip(&proposed.runs) {
            let allowed = match (old.state, new.state) {
                (RunState::Pending, RunState::Running)
                | (RunState::Running, RunState::Completed | RunState::Failed) => true,
                _ if old == new => true,
                _ => false,
            };
            if !allowed {
                return Err(ExperimentRunnerError::InvalidStatus {
                    path: state_file.to_path_buf(),
                    reason: format!(
                        "run `{}` attempted non-monotonic {:?} -> {:?} transition",
                        old.run_id, old.state, new.state
                    ),
                });
            }
        }
        Ok(())
    }

    fn write_status_atomically(
        &self,
        state_file: &Path,
        status: &ExperimentBatchStatus,
        plan: &ValidatedPlan,
    ) -> Result<(), ExperimentRunnerError> {
        self.validate_status(state_file, status, plan)?;
        let temporary = status_temporary_path(state_file)?;
        if temporary.exists() {
            return Err(ExperimentRunnerError::InterruptedStatusWrite {
                temporary,
                committed: state_file.to_path_buf(),
            });
        }
        let parent = state_file
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|source| ExperimentRunnerError::Io {
            operation: "create status parent",
            path: parent.to_path_buf(),
            source,
        })?;
        let bytes = canonical_status_bytes(status)?;
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_STATUS_BYTES {
            return Err(ExperimentRunnerError::InvalidStatus {
                path: state_file.to_path_buf(),
                reason: format!("serialized status exceeds the {MAX_STATUS_BYTES}-byte cap"),
            });
        }
        let mut temporary_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|source| ExperimentRunnerError::Io {
                operation: "create temporary status",
                path: temporary.clone(),
                source,
            })?;
        temporary_file
            .write_all(&bytes)
            .map_err(|source| ExperimentRunnerError::Io {
                operation: "write temporary status",
                path: temporary.clone(),
                source,
            })?;
        temporary_file
            .sync_all()
            .map_err(|source| ExperimentRunnerError::Io {
                operation: "sync temporary status",
                path: temporary.clone(),
                source,
            })?;
        drop(temporary_file);
        fs::rename(&temporary, state_file).map_err(|source| ExperimentRunnerError::Io {
            operation: "commit experiment status",
            path: state_file.to_path_buf(),
            source,
        })?;
        sync_parent_directory(parent)?;
        Ok(())
    }

    fn run_checkpoint_path(&self, run_id: &str) -> PathBuf {
        self.output_dir.join(format!("{run_id}.checkpoint"))
    }

    fn validate_run_checkpoint(
        &self,
        path: &Path,
        checkpoint: &RunCheckpointV1,
        variant: &ScenarioVariant,
        seed: u64,
        run_id: &str,
        plan_digest: &str,
    ) -> Result<(), ExperimentRunnerError> {
        validate_run_checkpoint_capacity(checkpoint, path, MAX_RUN_POPULATION)?;
        let family = ResolvedBrainFamily::resolve(variant)?;
        let metadata = &checkpoint.metadata;
        let manifest = &metadata.manifest;
        let host = checkpoint.owner.host_checkpoint();
        let expected_config = config_for_run(&variant.config_overrides, seed)
            .map_err(|error| invalid_checkpoint(path, error))?;
        let interventions = PreparedInterventions::new(
            &variant.variant_id,
            &expected_config,
            &variant.interventions,
            self.max_ticks,
        )
        .map_err(|error| invalid_checkpoint(path, error))?;
        let expected_config = serde_json::to_value(&expected_config)
            .map_err(|error| invalid_checkpoint(path, error))?;
        if metadata.plan_digest != plan_digest
            || metadata.run_id != run_id
            || manifest.identity.run_id != Self::bundle_run_id(run_id)
            || manifest.identity.experiment_id.as_deref() != Some(self.experiment_id.as_str())
            || manifest.identity.variant_id.as_deref() != Some(variant.variant_id.as_str())
            || manifest.identity.requested_tick_budget != Some(self.max_ticks)
            || manifest.identity.live_run_policy.is_some()
            || manifest.root_seed != seed
            || manifest.build != BuildProvenanceV0::current()
            || manifest.normalized_config != expected_config
            || manifest.scenario.interventions != variant.interventions
            || manifest.brain_roster.len() != 1
            || manifest.brain_roster[0].kind != family.canonical_name()
            || manifest.scenario.id != run_id
            || manifest.scenario.population_recipe
                != format!(
                    "fixed-4x4-registered-brain-grid-v1;brain={}",
                    family.canonical_name()
                )
            || host.session_id() != host_identifiers(run_id).0
            || host.tick().0 > self.max_ticks
            || host.playback()
                != (PlaybackSnapshot {
                    paused: true,
                    speed_multiplier: 1.0,
                })
            || (host.tick().0 == 0) != metadata.parent_digest.is_none()
            || metadata.parent_digest.as_ref().is_some_and(|digest| {
                digest.len() != blake3::OUT_LEN * 2
                    || !digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            })
        {
            return Err(invalid_checkpoint(
                path,
                "checkpoint differs from the validated plan, build, manifest, family, config, session or execution boundary",
            ));
        }
        validate_checkpoint_interventions(path, host, &interventions)?;
        manifest
            .to_storage_record()
            .map_err(|error| invalid_checkpoint(path, error))?;
        Ok(())
    }

    /// Caller owns the per-run OS writer lease while inspecting or promoting temporary cuts.
    fn load_run_checkpoint(
        &self,
        variant: &ScenarioVariant,
        seed: u64,
        run_id: &str,
        plan_digest: &str,
        population_limit: u64,
    ) -> Result<Option<RunCheckpointV1>, ExperimentRunnerError> {
        let path = self.run_checkpoint_path(run_id);
        let committed = read_optional_run_checkpoint(&path)?;
        if let Some(checkpoint) = &committed {
            validate_run_checkpoint_capacity(checkpoint, &path, population_limit)?;
            self.validate_run_checkpoint(&path, checkpoint, variant, seed, run_id, plan_digest)?;
        }
        let temporary = status_temporary_path(&path)?;
        let candidate = match fs::symlink_metadata(&temporary) {
            Ok(_) => {
                let bytes = read_experiment_file_bounded(&temporary, MAX_RUN_CHECKPOINT_BYTES)?;
                match RunCheckpointV1::decode(&bytes, &temporary) {
                    Ok(candidate) => candidate,
                    Err(error) => {
                        if committed.is_none() {
                            return Err(error);
                        }
                        // Preserve one bounded interrupted frame for inspection. Never delete it,
                        // overwrite an earlier incident, or promote any part of this frame.
                        let interrupted = path.with_extension("checkpoint.interrupted");
                        match fs::symlink_metadata(&interrupted) {
                            Ok(_) => {
                                return Err(invalid_checkpoint(
                                    &temporary,
                                    "an interrupted frame is already retained; inspection is required before another recovery",
                                ));
                            }
                            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                            Err(source) => {
                                return Err(ExperimentRunnerError::Io {
                                    operation: "inspect interrupted checkpoint",
                                    path: interrupted,
                                    source,
                                });
                            }
                        }
                        fs::rename(&temporary, &interrupted).map_err(|source| {
                            ExperimentRunnerError::Io {
                                operation: "retain interrupted checkpoint",
                                path: temporary.clone(),
                                source,
                            }
                        })?;
                        sync_parent_directory(&self.output_dir)?;
                        return Ok(committed);
                    }
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(committed),
            Err(source) => {
                return Err(ExperimentRunnerError::Io {
                    operation: "inspect temporary checkpoint",
                    path: temporary,
                    source,
                });
            }
        };
        validate_run_checkpoint_capacity(&candidate, &temporary, population_limit)?;
        self.validate_run_checkpoint(&temporary, &candidate, variant, seed, run_id, plan_digest)?;
        let expected_parent = committed
            .as_ref()
            .map(|checkpoint| {
                checkpoint
                    .encode(&path)
                    .map(|bytes| blake3::hash(&bytes).to_hex().to_string())
            })
            .transpose()?;
        let expected_tick = committed.as_ref().map_or(0, |checkpoint| {
            checkpoint.owner.host_checkpoint().tick().0 + 1
        });
        if candidate.metadata.parent_digest != expected_parent
            || candidate.owner.host_checkpoint().tick().0 != expected_tick
        {
            return Err(invalid_checkpoint(
                &temporary,
                "temporary cut does not extend the exact committed parent by one tick",
            ));
        }
        // Decoding validates the data envelope, but executable adapters and the
        // recomputed scientific digest are checked only by restoring the owner.
        // Refuse an unrestorable successor while the exact committed parent and
        // temporary evidence still exist. Ordinary reads without a temporary cut
        // keep their existing single-restore path.
        let family = ResolvedBrainFamily::resolve(variant)?;
        drop(restore_run_owner(&candidate, family, run_id, &temporary)?);
        File::open(&temporary)
            .and_then(|file| file.sync_all())
            .map_err(|source| ExperimentRunnerError::Io {
                operation: "sync recovered checkpoint",
                path: temporary.clone(),
                source,
            })?;
        fs::rename(&temporary, &path).map_err(|source| ExperimentRunnerError::Io {
            operation: "promote validated checkpoint",
            path: temporary,
            source,
        })?;
        sync_parent_directory(&self.output_dir)?;
        Ok(Some(candidate))
    }

    fn prepare_run_checkpoint(
        &self,
        variant: &ScenarioVariant,
        family: ResolvedBrainFamily,
        seed: u64,
        run_id: &str,
        plan_digest: &str,
    ) -> Result<(), ExperimentRunnerError> {
        let path = self.run_checkpoint_path(run_id);
        let _lease = ExperimentStatusWriterLease::acquire(&path)?;
        let bundle = self.output_dir.join(run_id);
        if bundle
            .try_exists()
            .map_err(|source| ExperimentRunnerError::Io {
                operation: "inspect existing bundle",
                path: bundle.clone(),
                source,
            })?
        {
            return Err(ExperimentRunnerError::BundleAlreadyExists {
                run_id: run_id.to_owned(),
                path: bundle,
            });
        }
        if let Some(checkpoint) =
            self.load_run_checkpoint(variant, seed, run_id, plan_digest, MAX_RUN_POPULATION)?
        {
            if checkpoint.owner.host_checkpoint().tick().0 != 0 {
                return Err(invalid_checkpoint(
                    &path,
                    "Pending run already has a nonzero cut",
                ));
            }
            return Ok(());
        }
        let checkpoint = self.initial_run_checkpoint(variant, family, seed, run_id, plan_digest)?;
        self.validate_run_checkpoint(&path, &checkpoint, variant, seed, run_id, plan_digest)?;
        self.publish_run_checkpoint(&path, &checkpoint, &checkpoint.encode(&path)?)
            .map(|_| ())
    }

    fn initial_run_checkpoint(
        &self,
        variant: &ScenarioVariant,
        family: ResolvedBrainFamily,
        seed: u64,
        run_id: &str,
        plan_digest: &str,
    ) -> Result<RunCheckpointV1, ExperimentRunnerError> {
        let config = config_for_run(&variant.config_overrides, seed).map_err(|reason| {
            ExperimentRunnerError::Construction {
                run_id: run_id.to_owned(),
                reason,
            }
        })?;
        let interventions = PreparedInterventions::new(
            &variant.variant_id,
            &config,
            &variant.interventions,
            self.max_ticks,
        )?;
        let mut world =
            WorldState::new(config).map_err(|source| ExperimentRunnerError::Construction {
                run_id: run_id.to_owned(),
                reason: source.to_string(),
            })?;
        let brain_key = family.register(&mut world, run_id)?;
        seed_founding_population(&mut world, &[brain_key]).map_err(|source| {
            ExperimentRunnerError::Construction {
                run_id: run_id.to_owned(),
                reason: format!("founder seeding failed: {source}"),
            }
        })?;
        enforce_run_population(&world, run_id, MAX_RUN_POPULATION, None)?;

        let stable_run_id = Self::bundle_run_id(run_id);
        let mut identity = RunIdentityV1::new(
            stable_run_id,
            current_unix_millis(run_id)?,
            Some(self.max_ticks),
            None,
        );
        identity.experiment_id = Some(self.experiment_id.clone());
        identity.variant_id = Some(variant.variant_id.clone());
        let mut scenario = ScenarioIdentityV0::caller_seeded(run_id);
        scenario.interventions = variant.interventions.clone();
        for intervention in &mut scenario.interventions {
            intervention.set.sort_all_objects();
        }
        scenario.population_recipe = format!(
            "fixed-4x4-registered-brain-grid-v1;brain={}",
            family.canonical_name()
        );
        let build = BuildProvenanceV0::current();
        let actual_rayon_threads = build.core.rayon_threads;
        let manifest = RunManifestV3::from_world_with_provenance(identity, scenario, &world, build)
            .map_err(|source| ExperimentRunnerError::Construction {
                run_id: run_id.to_owned(),
                reason: format!("canonical run manifest failed: {source}"),
            })?
            .with_thread_policy(ThreadPolicyV0 {
                threads: Some(actual_rayon_threads),
                source: "process-global-rayon-pool".to_owned(),
                overridden: None,
            });
        let (session_id, client_namespace) = host_identifiers(run_id);
        let options = HostCoreOptions {
            initial_playback: PlaybackSnapshot {
                paused: true,
                speed_multiplier: 1.0,
            },
            ..HostCoreOptions::default()
        };
        let mut core = HostCore::new(session_id, world, options).map_err(|source| {
            ExperimentRunnerError::Construction {
                run_id: run_id.to_owned(),
                reason: format!("HostCore construction failed: {source}"),
            }
        })?;
        if !interventions.patches.is_empty() {
            core.install_schedule(interventions.patches, resolve_scheduled_config_patch)
                .map_err(|source| ExperimentRunnerError::Construction {
                    run_id: run_id.to_owned(),
                    reason: format!("owner intervention schedule installation failed: {source}"),
                })?;
        }
        let mut frontend = NullFrontend::new(core.local_port(), client_namespace);
        let mut next_nanos = 0_u64;
        seal_run_checkpoint(run_id, &mut frontend, &mut core, &mut next_nanos)?;
        Ok(RunCheckpointV1 {
            metadata: RunCheckpointMetadataV1 {
                plan_digest: plan_digest.to_owned(),
                run_id: run_id.to_owned(),
                parent_digest: None,
                manifest,
            },
            owner: core
                .capture_volatile_host_checkpoint_v1()
                .map_err(|error| invalid_checkpoint(&self.run_checkpoint_path(run_id), error))?,
        })
    }

    fn execute_validated_run(
        &self,
        variant: &ScenarioVariant,
        family: ResolvedBrainFamily,
        seed: u64,
        run_id: &str,
        plan_digest: &str,
        budget: RunExecutionBudget<'_>,
    ) -> Result<RunRecord, ExperimentRunnerError> {
        let path = self.run_checkpoint_path(run_id);
        let _lease = ExperimentStatusWriterLease::acquire(&path)?;
        let mut checkpoint = self
            .load_run_checkpoint(variant, seed, run_id, plan_digest, budget.population_limit)?
            .ok_or_else(|| ExperimentRunnerError::ResumeCheckpointUnavailable {
                run_id: run_id.to_owned(),
                completed_ticks: 0,
            })?;
        let initial_config = config_for_run(&variant.config_overrides, seed)
            .map_err(|error| invalid_checkpoint(&path, error))?;
        let interventions = PreparedInterventions::new(
            &variant.variant_id,
            &initial_config,
            &variant.interventions,
            self.max_ticks,
        )
        .map_err(|error| invalid_checkpoint(&path, error))?;
        if experiment_path_exists(
            &self.staged_bundle_path(run_id),
            "inspect bundle staging before continuation",
        )? && checkpoint.owner.host_checkpoint().tick().0 != self.max_ticks
        {
            return Err(invalid_checkpoint(
                &path,
                "bundle staging exists without a sealed final owner cut; replaying ticks is refused",
            ));
        }
        let mut parent_digest = blake3::hash(&checkpoint.encode(&path)?)
            .to_hex()
            .to_string();
        let mut core = restore_run_owner(&checkpoint, family, run_id, &path)?;
        enforce_run_population(
            core.world(),
            run_id,
            budget.population_limit,
            Some(checkpoint.owner.host_checkpoint().tick().0),
        )?;
        self.observe_checkpoint_progress(
            plan_digest,
            run_id,
            core.world_tick().0,
            parent_digest.clone(),
            CheckpointProgressSource::Recovered,
        );
        let (_, client_namespace) = host_identifiers(run_id);
        let mut frontend = NullFrontend::new(core.local_port(), client_namespace);
        let mut next_nanos = 0_u64;
        info!(
            run_id,
            checkpoint_tick = core.world_tick().0,
            "restored durable experiment continuation cut"
        );

        let mut cancelled = false;
        for expected_tick in core.world_tick().0 + 1..=self.max_ticks {
            if budget.cancel.load(std::sync::atomic::Ordering::Acquire) {
                cancelled = true;
                break;
            }
            let submitted = frontend
                .submit_envelope(CommandEnvelope::new(
                    CommandId::from_client_sequence(client_namespace, expected_tick),
                    HostCommand::Step,
                ))
                .map_err(|source| ExperimentRunnerError::Host {
                    run_id: run_id.to_owned(),
                    operation: "submit step",
                    reason: source.to_string(),
                })?;
            drive_command_to_volatile_commit(
                run_id,
                "step",
                &mut frontend,
                &mut core,
                submitted.command_id(),
                expected_tick,
                interventions.due_at(expected_tick - 1),
                &mut next_nanos,
            )?;
            // Observe the actual completed science before any new flush/capture
            // or encoding. A crossing tick remains a failed, uncheckpointed
            // attempt; the previous committed cut is left byte-for-byte intact.
            enforce_run_population(
                core.world(),
                run_id,
                budget.population_limit,
                Some(checkpoint.owner.host_checkpoint().tick().0),
            )?;
            seal_run_checkpoint(run_id, &mut frontend, &mut core, &mut next_nanos)?;
            checkpoint.owner = core
                .capture_volatile_host_checkpoint_v1()
                .map_err(|error| invalid_checkpoint(&path, error))?;
            validate_checkpoint_interventions(
                &path,
                checkpoint.owner.host_checkpoint(),
                &interventions,
            )?;
            checkpoint.metadata.parent_digest = Some(parent_digest);
            let bytes = checkpoint.encode(&path)?;
            parent_digest = self.publish_run_checkpoint(&path, &checkpoint, &bytes)?;
        }

        let completed_ticks = core.world_tick().0;
        if !cancelled && completed_ticks != self.max_ticks {
            return Err(ExperimentRunnerError::Host {
                run_id: run_id.to_owned(),
                operation: "observe final tick",
                reason: format!(
                    "HostCore reached tick {} instead of {}",
                    core.world_tick().0,
                    self.max_ticks
                ),
            });
        }
        let digest = core
            .scientific_digest_v1()
            .map_err(|source| ExperimentRunnerError::Host {
                run_id: run_id.to_owned(),
                operation: "capture final digest",
                reason: source.to_string(),
            })?;
        let digest_hex = digest.overall;
        let summary = if cancelled {
            None
        } else {
            Some(
                ExperimentEndpointSummaryV2::from_world(
                    core.world(),
                    seed,
                    family.canonical_name(),
                    digest_hex.clone(),
                )
                .map_err(|reason| ExperimentRunnerError::Host {
                    run_id: run_id.to_owned(),
                    operation: "capture final endpoints",
                    reason,
                })?,
            )
        };

        let submitted = frontend
            .submit_envelope(CommandEnvelope::new(
                CommandId::from_client_sequence(client_namespace, completed_ticks + 1),
                HostCommand::Shutdown,
            ))
            .map_err(|source| ExperimentRunnerError::Host {
                run_id: run_id.to_owned(),
                operation: "submit shutdown",
                reason: source.to_string(),
            })?;
        drive_command_to_volatile_commit(
            run_id,
            "shutdown",
            &mut frontend,
            &mut core,
            submitted.command_id(),
            completed_ticks,
            0,
            &mut next_nanos,
        )?;
        if core.latest_snapshot().lifecycle != HostLifecycle::Stopped {
            return Err(ExperimentRunnerError::Host {
                run_id: run_id.to_owned(),
                operation: "observe shutdown",
                reason: format!(
                    "latest host lifecycle is {:?}, expected Stopped",
                    core.latest_snapshot().lifecycle
                ),
            });
        }

        if cancelled {
            info!(
                experiment_id = %self.experiment_id,
                run_id,
                checkpoint_tick = completed_ticks,
                "experiment owner stopped after cancellation; durable cut retained"
            );
            return Ok(RunRecord {
                run_id: run_id.to_owned(),
                variant_id: variant.variant_id.clone(),
                brain_family: family.canonical_name().to_owned(),
                seed,
                state: RunState::Running,
                // The unchanged status schema carries tick/digest evidence only
                // for terminal records. Actual progress stays in the joint cut.
                total_ticks: 0,
                final_digest: None,
                bundle_path: None,
                error_reason: None,
            });
        }

        let manifest = &checkpoint.metadata.manifest;
        let manifest_bytes =
            manifest
                .canonical_json_bytes()
                .map_err(|source| ExperimentRunnerError::Json {
                    operation: "serialize original run manifest",
                    source,
                })?;
        let manifest_record = manifest
            .to_storage_record()
            .map_err(|error| invalid_checkpoint(&path, error))?;
        let manifest_digest = blake3::hash(&manifest_bytes).to_hex().to_string();
        let evidence = RunEvidenceV2 {
            schema: "scriptbots.experiment-run-evidence.v2".to_owned(),
            summary_schema_version: EXPERIMENT_SUMMARY_SCHEMA_VERSION,
            plan_digest: plan_digest.to_owned(),
            experiment_id: self.experiment_id.clone(),
            cohort_id: self.cohort.cohort_id.clone(),
            run_id: run_id.to_owned(),
            variant_id: variant.variant_id.clone(),
            brain_family: family.canonical_name().to_owned(),
            seed,
            total_ticks: self.max_ticks,
            final_digest: digest_hex.clone(),
            manifest_digest,
        };
        let evidence_bytes =
            serde_json::to_vec_pretty(&evidence).map_err(|source| ExperimentRunnerError::Json {
                operation: "serialize run evidence",
                source,
            })?;
        let summary_csv = summary
            .ok_or_else(|| ExperimentRunnerError::Host {
                run_id: run_id.to_owned(),
                operation: "publish final endpoints",
                reason: "completed run has no observed endpoint summary".to_owned(),
            })?
            .canonical_csv_bytes()
            .map_err(|reason| ExperimentRunnerError::Bundle {
                run_id: run_id.to_owned(),
                reason,
            })?;
        // Export the exact sealed pre-shutdown cut, including evaluator/RNG and
        // owner-journal state. Reporting its digest alone is not portable state.
        let final_checkpoint_bytes = checkpoint.encode(&path)?;
        let bundle_dir = self.output_dir.join(run_id);
        let completed = RunRecord {
            run_id: run_id.to_owned(),
            variant_id: variant.variant_id.clone(),
            brain_family: family.canonical_name().to_owned(),
            seed,
            state: RunState::Completed,
            total_ticks: self.max_ticks,
            final_digest: Some(digest_hex),
            bundle_path: Some(bundle_dir.to_string_lossy().into_owned()),
            error_reason: None,
        };
        self.publish_completed_bundle(
            &completed,
            variant,
            family,
            plan_digest,
            manifest_record,
            &[
                ("exports/summary.csv", "experiment-summary", &summary_csv),
                (
                    "evidence/run.json",
                    "experiment-run-evidence",
                    &evidence_bytes,
                ),
                (
                    FINAL_CHECKPOINT_ARTIFACT.0,
                    FINAL_CHECKPOINT_ARTIFACT.1,
                    &final_checkpoint_bytes,
                ),
            ],
        )?;
        self.verify_completed_bundle(&completed, variant, family, plan_digest)?;
        Ok(completed)
    }

    fn staged_bundle_path(&self, run_id: &str) -> PathBuf {
        self.output_dir.join(format!("{run_id}.bundle.tmp"))
    }

    /// Publish only verified output from the final sealed owner cut. The caller holds
    /// the run's writer lease across restore, execution and this directory handoff.
    fn publish_completed_bundle(
        &self,
        record: &RunRecord,
        variant: &ScenarioVariant,
        family: ResolvedBrainFamily,
        plan_digest: &str,
        manifest: scriptbots_storage::RunManifestRecord,
        artifacts: &[(&str, &str, &[u8])],
    ) -> Result<(), ExperimentRunnerError> {
        let bundle_dir = self.output_dir.join(&record.run_id);
        if experiment_path_exists(&bundle_dir, "inspect final experiment bundle")? {
            return Err(ExperimentRunnerError::BundleAlreadyExists {
                run_id: record.run_id.clone(),
                path: bundle_dir,
            });
        }
        let staged = self.staged_bundle_path(&record.run_id);
        let verify_staged = || -> Result<(), ExperimentRunnerError> {
            self.verify_completed_bundle_at(&staged, record, variant, family, plan_digest)?;
            for (relative, _, expected) in artifacts {
                let path = staged.join(relative);
                if read_experiment_file_bounded(&path, MAX_BUNDLE_ARTIFACT_BYTES)?.as_slice()
                    != *expected
                {
                    return Err(ExperimentRunnerError::Bundle {
                        run_id: record.run_id.clone(),
                        reason: format!(
                            "staged artifact {} differs from the exact sealed final owner cut",
                            path.display()
                        ),
                    });
                }
            }
            Ok(())
        };
        let mut ready = false;
        if experiment_path_exists(&staged, "inspect staged experiment bundle")? {
            validate_staged_bundle_layout(&staged, &record.run_id)?;
            match verify_staged() {
                Ok(()) => ready = true,
                Err(error) => {
                    // Only uncommitted staging can be regenerated. Preserve one
                    // bounded incident instead of deleting or overwriting evidence.
                    let retained = self
                        .output_dir
                        .join(format!("{}.bundle.interrupted", record.run_id));
                    if experiment_path_exists(&retained, "inspect retained bundle incident")? {
                        return Err(ExperimentRunnerError::Bundle {
                            run_id: record.run_id.clone(),
                            reason: format!(
                                "staging is incomplete or invalid ({error}); prior incident at {} must be inspected before another recovery",
                                retained.display()
                            ),
                        });
                    }
                    fs::rename(&staged, &retained).map_err(|source| ExperimentRunnerError::Io {
                        operation: "retain interrupted bundle staging",
                        path: staged.clone(),
                        source,
                    })?;
                    sync_parent_directory(&self.output_dir)?;
                    info!(
                        run_id = %record.run_id,
                        retained_path = %retained.display(),
                        error = %error,
                        "retained interrupted staging; rebuilding from the verified final owner cut"
                    );
                }
            }
        }
        if !ready {
            fs::create_dir(&staged).map_err(|source| ExperimentRunnerError::Io {
                operation: "reserve experiment bundle staging",
                path: staged.clone(),
                source,
            })?;
            create_run_bundle_from_artifacts(&staged, manifest, self.max_ticks, artifacts)
                .map_err(|source| ExperimentRunnerError::Bundle {
                    run_id: record.run_id.clone(),
                    reason: format!("staged assembly failed: {source}"),
                })?;
            validate_staged_bundle_layout(&staged, &record.run_id)?;
            verify_staged()?;
        }

        // File contents and nested directory entries must precede publication of
        // the final directory name, including recovery of a pre-rename crash.
        for relative in REQUIRED_EXPERIMENT_ARTIFACTS
            .iter()
            .map(|(path, _)| *path)
            .chain(std::iter::once("bundle_manifest.json"))
        {
            let path = staged.join(relative);
            let mut options = OpenOptions::new();
            options.read(true).write(cfg!(windows));
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
            }
            options
                .open(&path)
                .and_then(|file| file.sync_all())
                .map_err(|source| ExperimentRunnerError::Io {
                    operation: "sync staged experiment artifact",
                    path: path.clone(),
                    source,
                })?;
            if let Some(parent) = path.parent() {
                sync_parent_directory(parent)?;
            }
        }
        sync_parent_directory(&staged)?;
        if experiment_path_exists(&bundle_dir, "reinspect final experiment bundle")? {
            return Err(ExperimentRunnerError::BundleAlreadyExists {
                run_id: record.run_id.clone(),
                path: bundle_dir,
            });
        }
        fs::rename(&staged, &bundle_dir).map_err(|source| ExperimentRunnerError::Io {
            operation: "publish verified experiment bundle",
            path: bundle_dir,
            source,
        })?;
        sync_parent_directory(&self.output_dir)
    }

    /// Recover only the bundle/status commit gap; never replay an unfinished run.
    fn recover_completed_run(
        &self,
        record: &RunRecord,
        variant: &ScenarioVariant,
        family: ResolvedBrainFamily,
        plan_digest: &str,
    ) -> Result<RunRecord, ExperimentRunnerError> {
        let bundle_dir = self.output_dir.join(&record.run_id);
        let exists = bundle_dir
            .try_exists()
            .map_err(|source| ExperimentRunnerError::Io {
                operation: "inspect interrupted run bundle",
                path: bundle_dir.clone(),
                source,
            })?;
        if !exists {
            return Err(ExperimentRunnerError::ResumeCheckpointUnavailable {
                run_id: record.run_id.clone(),
                completed_ticks: record.total_ticks,
            });
        }

        let evidence_path = bundle_dir.join("evidence/run.json");
        let evidence_bytes =
            read_bounded_file(&evidence_path, MAX_STATUS_BYTES, "read recovery evidence")?;
        let evidence: RunEvidenceV2 =
            serde_json::from_slice(&evidence_bytes).map_err(|source| {
                ExperimentRunnerError::Bundle {
                    run_id: record.run_id.clone(),
                    reason: format!("recovery evidence JSON is invalid: {source}"),
                }
            })?;
        let mut completed = record.clone();
        completed.state = RunState::Completed;
        completed.total_ticks = self.max_ticks;
        completed.final_digest = Some(evidence.final_digest);
        completed.bundle_path = Some(bundle_dir.to_string_lossy().into_owned());
        // The parsed evidence is untrusted until both byte integrity and every
        // run/plan/family/config identity pass the existing canonical verifier.
        self.verify_completed_bundle(&completed, variant, family, plan_digest)?;
        Ok(completed)
    }

    fn verify_completed_bundle(
        &self,
        record: &RunRecord,
        variant: &ScenarioVariant,
        family: ResolvedBrainFamily,
        plan_digest: &str,
    ) -> Result<VerifiedExperimentFinalState, ExperimentRunnerError> {
        let bundle_dir = self.output_dir.join(&record.run_id);
        self.verify_completed_bundle_at(&bundle_dir, record, variant, family, plan_digest)
    }

    fn verify_completed_bundle_at(
        &self,
        bundle_dir: &Path,
        record: &RunRecord,
        variant: &ScenarioVariant,
        family: ResolvedBrainFamily,
        plan_digest: &str,
    ) -> Result<VerifiedExperimentFinalState, ExperimentRunnerError> {
        verify_run_bundle_bounded(
            bundle_dir,
            RunBundleVerificationLimits {
                max_manifest_bytes: MAX_STATUS_BYTES,
                max_artifacts: MAX_BUNDLE_ARTIFACTS,
                max_artifact_bytes: MAX_BUNDLE_ARTIFACT_BYTES,
                max_total_artifact_bytes: MAX_BUNDLE_TOTAL_ARTIFACT_BYTES,
            },
        )
        .map_err(|source| ExperimentRunnerError::Bundle {
            run_id: record.run_id.clone(),
            reason: format!("bounded canonical verifier rejected the bundle: {source}"),
        })?;
        let bundle_path = bundle_dir.join("bundle_manifest.json");
        let bytes = read_bounded_file(
            &bundle_path,
            MAX_STATUS_BYTES,
            "read bundle manifest for semantic verification",
        )?;
        let bundle: RunBundleV1 =
            serde_json::from_slice(&bytes).map_err(|source| ExperimentRunnerError::Bundle {
                run_id: record.run_id.clone(),
                reason: format!("bundle manifest JSON is invalid: {source}"),
            })?;
        for &(required_path, required_type) in REQUIRED_EXPERIMENT_ARTIFACTS {
            let matches = bundle
                .artifacts
                .iter()
                .filter(|entry| {
                    entry.relative_path == required_path && entry.artifact_type == required_type
                })
                .count();
            if matches != 1 {
                return Err(ExperimentRunnerError::Bundle {
                    run_id: record.run_id.clone(),
                    reason: format!(
                        "bundle must index `{required_path}` exactly once as `{required_type}`, found {matches}"
                    ),
                });
            }
        }
        let stable_run_id = Self::bundle_run_id(&record.run_id);
        if bundle.manifest.run_id != stable_run_id
            || bundle.digests.run_id != stable_run_id.to_string()
            || bundle.digests.max_tick != self.max_ticks
            || bundle.manifest.experiment_id.as_deref() != Some(self.experiment_id.as_str())
            || bundle.manifest.variant_id.as_deref() != Some(variant.variant_id.as_str())
            || bundle.manifest.root_seed != record.seed
            || bundle.manifest.requested_tick_budget != Some(self.max_ticks)
            || bundle.manifest.live_run_policy.is_some()
        {
            return Err(ExperimentRunnerError::Bundle {
                run_id: record.run_id.clone(),
                reason: "bundle manifest identity, seed, or execution boundary does not match the validated run".to_owned(),
            });
        }

        let full_manifest: RunManifestV3 = serde_json::from_str(&bundle.manifest.manifest_json)
            .map_err(|source| ExperimentRunnerError::Bundle {
                run_id: record.run_id.clone(),
                reason: format!("embedded RunManifestV3 is invalid: {source}"),
            })?;
        let expected_config =
            config_for_run(&variant.config_overrides, record.seed).map_err(|reason| {
                ExperimentRunnerError::Bundle {
                    run_id: record.run_id.clone(),
                    reason: format!("current arm no longer produces a config: {reason}"),
                }
            })?;
        let expected_config_value = serde_json::to_value(expected_config).map_err(|source| {
            ExperimentRunnerError::Json {
                operation: "serialize expected bundle config",
                source,
            }
        })?;
        if full_manifest.identity.run_id != stable_run_id
            || full_manifest.identity.experiment_id.as_deref() != Some(self.experiment_id.as_str())
            || full_manifest.identity.variant_id.as_deref() != Some(variant.variant_id.as_str())
            || full_manifest.root_seed != record.seed
            || full_manifest.normalized_config != expected_config_value
            || full_manifest.scenario.interventions != variant.interventions
            || full_manifest.brain_roster.len() != 1
            || full_manifest.brain_roster[0].kind != family.canonical_name()
            || full_manifest.scenario.population_recipe
                != format!(
                    "fixed-4x4-registered-brain-grid-v1;brain={}",
                    family.canonical_name()
                )
        {
            return Err(ExperimentRunnerError::Bundle {
                run_id: record.run_id.clone(),
                reason: "embedded manifest does not bind the validated family, config, scenario, or identity".to_owned(),
            });
        }
        let manifest_bytes =
            full_manifest
                .canonical_json_bytes()
                .map_err(|source| ExperimentRunnerError::Json {
                    operation: "re-serialize embedded run manifest",
                    source,
                })?;
        let evidence_path = bundle_dir.join("evidence/run.json");
        let evidence_bytes =
            read_bounded_file(&evidence_path, MAX_STATUS_BYTES, "read run evidence")?;
        let evidence: RunEvidenceV2 =
            serde_json::from_slice(&evidence_bytes).map_err(|source| {
                ExperimentRunnerError::Bundle {
                    run_id: record.run_id.clone(),
                    reason: format!("run evidence JSON is invalid: {source}"),
                }
            })?;
        let expected_evidence = RunEvidenceV2 {
            schema: "scriptbots.experiment-run-evidence.v2".to_owned(),
            summary_schema_version: EXPERIMENT_SUMMARY_SCHEMA_VERSION,
            plan_digest: plan_digest.to_owned(),
            experiment_id: self.experiment_id.clone(),
            cohort_id: self.cohort.cohort_id.clone(),
            run_id: record.run_id.clone(),
            variant_id: record.variant_id.clone(),
            brain_family: family.canonical_name().to_owned(),
            seed: record.seed,
            total_ticks: self.max_ticks,
            final_digest: record.final_digest.clone().ok_or_else(|| {
                ExperimentRunnerError::Bundle {
                    run_id: record.run_id.clone(),
                    reason: "completed status has no final digest".to_owned(),
                }
            })?,
            manifest_digest: blake3::hash(&manifest_bytes).to_hex().to_string(),
        };
        if evidence != expected_evidence {
            return Err(ExperimentRunnerError::Bundle {
                run_id: record.run_id.clone(),
                reason: "run evidence does not match status, plan, family, manifest, or digest"
                    .to_owned(),
            });
        }
        let checkpoint_path = bundle_dir.join(FINAL_CHECKPOINT_ARTIFACT.0);
        let verify_final_checkpoint =
            || -> Result<VerifiedExperimentFinalState, ExperimentRunnerError> {
                let bytes =
                    read_experiment_file_bounded(&checkpoint_path, MAX_RUN_CHECKPOINT_BYTES)?;
                let checkpoint = RunCheckpointV1::decode(&bytes, &checkpoint_path)?;
                self.validate_run_checkpoint(
                    &checkpoint_path,
                    &checkpoint,
                    variant,
                    record.seed,
                    &record.run_id,
                    plan_digest,
                )?;
                verify_restored_final_checkpoint(
                    bundle_dir,
                    &checkpoint,
                    &full_manifest,
                    &evidence,
                    family,
                )
            };
        verify_final_checkpoint().map_err(|error| ExperimentRunnerError::Bundle {
            run_id: record.run_id.clone(),
            reason: format!("restored final checkpoint verification failed: {error}"),
        })
    }
}

fn experiment_path_exists(
    path: &Path,
    operation: &'static str,
) -> Result<bool, ExperimentRunnerError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(ExperimentRunnerError::Io {
            operation,
            path: path.to_path_buf(),
            source,
        }),
    }
}

/// Inspect the bounded writer-owned layout before any staged file is opened or moved.
/// Missing files are expected after an interruption; aliases and unrelated files are not.
fn validate_staged_bundle_layout(staged: &Path, run_id: &str) -> Result<(), ExperimentRunnerError> {
    let invalid = |path: &Path, reason: &str| ExperimentRunnerError::Bundle {
        run_id: run_id.to_owned(),
        reason: format!("invalid bundle staging at {}: {reason}", path.display()),
    };
    let files = REQUIRED_EXPERIMENT_ARTIFACTS
        .iter()
        .map(|(path, _)| PathBuf::from(path))
        .chain(std::iter::once(PathBuf::from("bundle_manifest.json")))
        .collect::<BTreeSet<_>>();
    let directories = files
        .iter()
        .flat_map(|path| path.ancestors().skip(1))
        .filter(|path| !path.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .collect::<BTreeSet<_>>();
    let mut pending = vec![PathBuf::new()];
    let mut artifact_bytes = 0_u64;
    while let Some(relative) = pending.pop() {
        let directory = staged.join(&relative);
        let metadata =
            fs::symlink_metadata(&directory).map_err(|source| ExperimentRunnerError::Io {
                operation: "inspect bundle staging directory",
                path: directory.clone(),
                source,
            })?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(invalid(&directory, "expected a non-symlink directory"));
        }
        let entries = fs::read_dir(&directory).map_err(|source| ExperimentRunnerError::Io {
            operation: "list bundle staging directory",
            path: directory.clone(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| ExperimentRunnerError::Io {
                operation: "read bundle staging directory entry",
                path: directory.clone(),
                source,
            })?;
            let relative_path = relative.join(entry.file_name());
            let path = entry.path();
            let metadata =
                fs::symlink_metadata(&path).map_err(|source| ExperimentRunnerError::Io {
                    operation: "inspect bundle staging entry",
                    path: path.clone(),
                    source,
                })?;
            if metadata.file_type().is_symlink() {
                return Err(invalid(&path, "symlinks are refused"));
            }
            if directories.contains(&relative_path) && metadata.is_dir() {
                pending.push(relative_path);
            } else if files.contains(&relative_path) && metadata.is_file() {
                if !same_status_file_identity(&metadata, &metadata) {
                    return Err(invalid(&path, "multiply linked files are refused"));
                }
                if relative_path == Path::new("bundle_manifest.json") {
                    if metadata.len() > MAX_STATUS_BYTES {
                        return Err(invalid(&path, "manifest exceeds its byte cap"));
                    }
                } else {
                    if metadata.len() > MAX_BUNDLE_ARTIFACT_BYTES {
                        return Err(invalid(&path, "artifact exceeds its byte cap"));
                    }
                    artifact_bytes = artifact_bytes
                        .checked_add(metadata.len())
                        .filter(|total| *total <= MAX_BUNDLE_TOTAL_ARTIFACT_BYTES)
                        .ok_or_else(|| invalid(staged, "artifacts exceed their total byte cap"))?;
                }
            } else {
                return Err(invalid(&path, "unexpected path or non-regular artifact"));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_path_component(
    label: &str,
    value: &str,
) -> Result<(), ExperimentRunnerError> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.len() > MAX_RUN_COMPONENT_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(ExperimentRunnerError::InvalidPlan {
            reason: format!(
                "{label} `{value}` must be 1..={MAX_RUN_COMPONENT_BYTES} ASCII bytes using only letters, digits, '.', '_', or '-' and cannot be '.' or '..'"
            ),
        });
    }
    Ok(())
}

fn status_temporary_path(state_file: &Path) -> Result<PathBuf, ExperimentRunnerError> {
    let file_name = state_file
        .file_name()
        .ok_or_else(|| ExperimentRunnerError::InvalidPlan {
            reason: format!(
                "status path {} has no file name",
                state_file.to_string_lossy()
            ),
        })?;
    let mut temporary_name = file_name.to_os_string();
    temporary_name.push(".tmp");
    Ok(state_file.with_file_name(temporary_name))
}

fn canonical_status_bytes(
    status: &ExperimentBatchStatus,
) -> Result<Vec<u8>, ExperimentRunnerError> {
    let mut bytes =
        serde_json::to_vec_pretty(status).map_err(|source| ExperimentRunnerError::Json {
            operation: "serialize experiment status",
            source,
        })?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn status_lock_path(state_file: &Path) -> Result<PathBuf, ExperimentRunnerError> {
    let file_name = state_file
        .file_name()
        .ok_or_else(|| ExperimentRunnerError::InvalidPlan {
            reason: format!(
                "status path {} has no file name",
                state_file.to_string_lossy()
            ),
        })?;
    let mut lock_name = file_name.to_os_string();
    lock_name.push(".lock");
    Ok(state_file.with_file_name(lock_name))
}

#[cfg(unix)]
fn same_status_file_identity(opened: &std::fs::Metadata, path: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    opened.dev() == path.dev()
        && opened.ino() == path.ino()
        && opened.nlink() == 1
        && path.nlink() == 1
}

#[cfg(not(unix))]
fn same_status_file_identity(_opened: &std::fs::Metadata, _path: &std::fs::Metadata) -> bool {
    true
}

fn read_bounded_file(
    path: &Path,
    max_bytes: u64,
    operation: &'static str,
) -> Result<Vec<u8>, ExperimentRunnerError> {
    let before = fs::symlink_metadata(path).map_err(|source| ExperimentRunnerError::Io {
        operation,
        path: path.to_path_buf(),
        source,
    })?;
    if before.file_type().is_symlink() || !before.is_file() {
        return Err(ExperimentRunnerError::InvalidStatus {
            path: path.to_path_buf(),
            reason: "expected a regular file without a symlink alias".to_owned(),
        });
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // A path can change after inspection. In particular, opening a FIFO
        // without NONBLOCK could stall every API caller holding the job registry.
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|source| ExperimentRunnerError::Io {
            operation,
            path: path.to_path_buf(),
            source,
        })?;
    let metadata = file
        .metadata()
        .map_err(|source| ExperimentRunnerError::Io {
            operation,
            path: path.to_path_buf(),
            source,
        })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.nlink() > 1 {
            return Err(ExperimentRunnerError::InvalidStatus {
                path: path.to_path_buf(),
                reason: "multiply linked status or evidence files are refused".to_owned(),
            });
        }
    }
    if !metadata.is_file() || metadata.len() > max_bytes {
        return Err(ExperimentRunnerError::InvalidStatus {
            path: path.to_path_buf(),
            reason: format!(
                "expected a regular file no larger than {max_bytes} bytes, found {} bytes",
                metadata.len()
            ),
        });
    }
    let mut bytes =
        Vec::with_capacity(usize::try_from(metadata.len().min(max_bytes)).unwrap_or(usize::MAX));
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| ExperimentRunnerError::Io {
            operation,
            path: path.to_path_buf(),
            source,
        })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max_bytes {
        return Err(ExperimentRunnerError::InvalidStatus {
            path: path.to_path_buf(),
            reason: format!("file exceeded the hard {max_bytes}-byte read cap"),
        });
    }
    // Status writers publish by atomic rename while readers hold this descriptor.
    // The old inode may now have zero links and the pathname may name a newer
    // generation. Return the complete opened snapshot; requiring the pathname
    // to remain unchanged would reject healthy concurrent status publication.
    Ok(bytes)
}

fn sync_parent_directory(parent: &Path) -> Result<(), ExperimentRunnerError> {
    #[cfg(unix)]
    {
        let directory = File::open(parent).map_err(|source| ExperimentRunnerError::Io {
            operation: "open status parent for sync",
            path: parent.to_path_buf(),
            source,
        })?;
        directory
            .sync_all()
            .map_err(|source| ExperimentRunnerError::Io {
                operation: "sync status parent",
                path: parent.to_path_buf(),
                source,
            })?;
    }
    // Windows cannot open a directory as a File to fsync it; the rename is the durability step.
    #[cfg(not(unix))]
    let _ = parent;
    Ok(())
}

fn recompute_status_counts(status: &mut ExperimentBatchStatus) {
    status.completed_runs = status
        .runs
        .iter()
        .filter(|record| record.state == RunState::Completed)
        .count();
    status.failed_runs = status
        .runs
        .iter()
        .filter(|record| record.state == RunState::Failed)
        .count();
}

fn current_unix_millis(run_id: &str) -> Result<u64, ExperimentRunnerError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|source| ExperimentRunnerError::Construction {
            run_id: run_id.to_owned(),
            reason: format!("system clock precedes Unix epoch: {source}"),
        })?;
    u64::try_from(duration.as_millis()).map_err(|_| ExperimentRunnerError::Construction {
        run_id: run_id.to_owned(),
        reason: "current Unix millisecond timestamp does not fit u64".to_owned(),
    })
}

fn host_identifiers(run_id: &str) -> (HostSessionId, u64) {
    let digest = blake3::hash(run_id.as_bytes());
    let mut session = [0_u8; 8];
    let mut client = [0_u8; 8];
    session.copy_from_slice(&digest.as_bytes()[..8]);
    client.copy_from_slice(&digest.as_bytes()[8..16]);
    (
        HostSessionId::new(u64::from_be_bytes(session).max(1)),
        u64::from_be_bytes(client).max(1),
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the bounded declared intervention work accompanies the exact command and tick receipt"
)]
fn drive_command_to_volatile_commit(
    run_id: &str,
    operation: &'static str,
    frontend: &mut NullFrontend<scriptbots_runtime::LocalHostPort>,
    core: &mut HostCore,
    command_id: scriptbots_runtime::CommandId,
    expected_tick: u64,
    due_interventions: usize,
    next_nanos: &mut u64,
) -> Result<(), ExperimentRunnerError> {
    if due_interventions > MAX_EXPERIMENT_INTERVENTIONS {
        return Err(ExperimentRunnerError::Host {
            run_id: run_id.to_owned(),
            operation,
            reason: "command exceeds bounded intervention work allowance".to_owned(),
        });
    }
    // The real VolatileJournal acknowledges each patch on the next drive. The
    // owner then clears that barrier and starts the next patch or original Step,
    // so N due inputs add N drives; unrelated commands keep the original bound.
    let drive_limit = HOST_RECEIPT_DRIVE_LIMIT + due_interventions;
    let mut last_status = None;
    for _ in 0..drive_limit {
        frontend
            .drive_at(core, ManualInstant::from_nanos(*next_nanos))
            .map_err(|source| ExperimentRunnerError::Host {
                run_id: run_id.to_owned(),
                operation,
                reason: format!("manual drive failed: {source}"),
            })?;
        *next_nanos = next_nanos
            .checked_add(1)
            .ok_or_else(|| ExperimentRunnerError::Host {
                run_id: run_id.to_owned(),
                operation,
                reason: "manual clock exhausted".to_owned(),
            })?;
        let status = frontend
            .command_status(command_id)
            .map_err(|source| ExperimentRunnerError::Host {
                run_id: run_id.to_owned(),
                operation,
                reason: format!("command status query failed: {source}"),
            })?
            .ok_or_else(|| ExperimentRunnerError::Host {
                run_id: run_id.to_owned(),
                operation,
                reason: "submitted command disappeared from queryable status".to_owned(),
            })?;
        match (status.application(), status.journal()) {
            (ApplicationState::Applied(applied), JournalState::CommittedVolatile) => {
                if applied.tick.0 != expected_tick {
                    return Err(ExperimentRunnerError::Host {
                        run_id: run_id.to_owned(),
                        operation,
                        reason: format!(
                            "receipt applied at tick {}, expected {expected_tick}",
                            applied.tick.0
                        ),
                    });
                }
                return Ok(());
            }
            (ApplicationState::Rejected(reason), _) => {
                return Err(ExperimentRunnerError::Host {
                    run_id: run_id.to_owned(),
                    operation,
                    reason: format!("command was rejected: {reason:?}"),
                });
            }
            (ApplicationState::Failed(failure), _) => {
                return Err(ExperimentRunnerError::Host {
                    run_id: run_id.to_owned(),
                    operation,
                    reason: format!("command application failed: {failure:?}"),
                });
            }
            (_, JournalState::Failed(failure)) => {
                return Err(ExperimentRunnerError::Host {
                    run_id: run_id.to_owned(),
                    operation,
                    reason: format!("command journal failed: {failure:?}"),
                });
            }
            _ => last_status = Some(status),
        }
    }
    Err(ExperimentRunnerError::Host {
        run_id: run_id.to_owned(),
        operation,
        reason: format!(
            "command did not reach Applied + CommittedVolatile within {drive_limit} drives; last status: {last_status:?}"
        ),
    })
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 16
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Prove that one canonical arm becomes a valid fresh-world configuration
/// without creating a world or writing an artifact.
///
/// # Errors
///
/// Returns the exact unknown-path, type, decode, or config-invariant failure.
pub fn validate_scenario_arm(
    overrides: &BTreeMap<String, serde_json::Value>,
) -> Result<(), String> {
    config_for_run(overrides, 0).map(|_| ())
}

/// Validate a requested experiment family against the adapters in this build.
///
/// Control-plane admission and checkpoint/bundle reconstruction use the same
/// resolver so a compiled optional family cannot be admitted by only one path.
pub(crate) fn validate_experiment_brain_family(
    variant_id: &str,
    brain_family: &str,
) -> Result<(), ExperimentRunnerError> {
    canonical_experiment_brain_family(variant_id, brain_family).map(|_| ())
}

/// Resolve an accepted experiment-family alias to its installed canonical identity.
/// This is pure validation: it constructs no world, adapter, worker or artifact.
pub(crate) fn canonical_experiment_brain_family(
    variant_id: &str,
    requested: &str,
) -> Result<&'static str, ExperimentRunnerError> {
    ResolvedBrainFamily::resolve(&ScenarioVariant {
        variant_id: variant_id.to_owned(),
        brain_family: requested.to_owned(),
        config_overrides: BTreeMap::new(),
        interventions: Vec::new(),
    })
    .map(ResolvedBrainFamily::canonical_name)
}

pub(crate) fn config_for_run(
    overrides: &BTreeMap<String, serde_json::Value>,
    seed: u64,
) -> Result<ScriptBotsConfig, String> {
    // Experiments own an enabled recording cadence. Founders and every science
    // boundary must cross the canonical journal/admission path before a complete
    // owner cut can be captured. This raw policy is also bound by each manifest.
    let recorded = ScriptBotsConfig {
        persistence_interval: 1,
        ..ScriptBotsConfig::default()
    };
    let mut encoded = serde_json::to_value(recorded)
        .map_err(|error| format!("default config serialization failed: {error}"))?;
    let root = encoded
        .as_object_mut()
        .ok_or_else(|| "default config did not serialize as an object".to_owned())?;

    for (path, value) in overrides {
        let Some(range) = knob_range(path) else {
            return Err(format!(
                "validated arm contains unknown or unsupported knob `{path}`"
            ));
        };
        // Enforce the same bounds as a live config patch: `ScriptBotsConfig::validate`
        // alone admits worlds large enough to exhaust memory when a run starts.
        if let Some(number) = value.as_f64()
            && (!number.is_finite() || number < range.min || number > range.max)
        {
            return Err(format!(
                "knob `{path}` value {number} is outside its accepted range [{}, {}]",
                range.min, range.max
            ));
        }
        insert_dotted_value(root, path, value.clone())?;
    }

    let mut config: ScriptBotsConfig = serde_json::from_value(encoded)
        .map_err(|error| format!("validated arm does not decode as ScriptBotsConfig: {error}"))?;
    // The matched-seed axis is owned by the cohort and can never be displaced by
    // a treatment arm.
    config.rng_seed = Some(seed);
    config
        .validate()
        .map_err(|error| format!("validated arm produces an invalid config: {error}"))?;
    Ok(config)
}

/// Pure preparation of the immutable inputs and expected configuration at every
/// schedule prefix. Only small hashes and changed paths are retained, so checking
/// each committed tick does not repeatedly resolve the whole treatment schedule.
struct PreparedInterventions {
    patches: Vec<ScheduledConfigPatch>,
    config_digests: Vec<blake3::Hash>,
    changed_paths: Vec<Vec<String>>,
}

impl PreparedInterventions {
    fn new(
        variant_id: &str,
        initial: &ScriptBotsConfig,
        interventions: &[ScenarioInterventionV1],
        max_ticks: u64,
    ) -> Result<Self, ExperimentRunnerError> {
        let invalid = |reason: String| ExperimentRunnerError::InvalidPlan {
            reason: format!("variant `{variant_id}` is not executable: {reason}"),
        };
        validate_run_config_capacity(variant_id, initial, 0)?;
        if interventions.len() > MAX_EXPERIMENT_INTERVENTIONS {
            return Err(invalid(format!(
                "intervention count exceeds the {MAX_EXPERIMENT_INTERVENTIONS}-statement cap"
            )));
        }
        if interventions.len() > HostCoreOptions::default().archived_command_capacity {
            return Err(invalid(
                "intervention schedule exceeds complete owner receipt retention".to_owned(),
            ));
        }
        let mut current = initial.clone();
        let mut current_value =
            serde_json::to_value(&current).map_err(|error| invalid(error.to_string()))?;
        let mut prepared = Self {
            patches: Vec::with_capacity(interventions.len()),
            config_digests: vec![config_value_digest(&current_value).map_err(invalid)?],
            changed_paths: Vec::with_capacity(interventions.len()),
        };
        let mut previous_tick = 0;
        for (index, intervention) in interventions.iter().enumerate() {
            if intervention.tick >= max_ticks {
                return Err(invalid(format!(
                    "intervention {} tick {} must precede the {max_ticks}-tick completion boundary",
                    index + 1,
                    intervention.tick
                )));
            }
            if intervention.tick < previous_tick {
                return Err(invalid(
                    "intervention ticks must be nondecreasing; same-tick order is preserved"
                        .to_owned(),
                ));
            }
            let sequence = u64::try_from(index + 1).map_err(|error| invalid(error.to_string()))?;
            let patch = ScheduledConfigPatch::new(
                sequence,
                Tick(intervention.tick),
                intervention.set.clone(),
            )
            .map_err(invalid)?;
            if patch.patch.len() > MAX_EXPERIMENT_INTERVENTION_PATCH_BYTES {
                return Err(invalid(format!(
                    "intervention {sequence} exceeds the {MAX_EXPERIMENT_INTERVENTION_PATCH_BYTES}-byte canonical patch cap"
                )));
            }
            validate_live_intervention_patch(&intervention.set, "").map_err(invalid)?;
            let next =
                resolve_scheduled_config_patch(&current, &intervention.set).map_err(|error| {
                    invalid(format!("intervention {sequence} cannot resolve: {error}"))
                })?;
            next.validate().map_err(|error| {
                invalid(format!(
                    "intervention {sequence} produces an invalid config: {error}"
                ))
            })?;
            validate_run_config_capacity(variant_id, &next, sequence)?;
            if next.rng_seed != initial.rng_seed
                || next.persistence_interval != initial.persistence_interval
            {
                return Err(invalid(
                    "interventions cannot change the matched seed or recording policy".to_owned(),
                ));
            }
            let next_value =
                serde_json::to_value(&next).map_err(|error| invalid(error.to_string()))?;
            prepared
                .config_digests
                .push(config_value_digest(&next_value).map_err(invalid)?);
            prepared
                .changed_paths
                .push(scriptbots_runtime::host_core::config_changed_paths(
                    &current_value,
                    &next_value,
                ));
            prepared.patches.push(patch);
            current = next;
            current_value = next_value;
            previous_tick = intervention.tick;
        }
        Ok(prepared)
    }

    fn completed_at(&self, tick: u64) -> usize {
        self.patches
            .partition_point(|patch| patch.due_tick.0 < tick)
    }

    fn due_at(&self, tick: u64) -> usize {
        let completed = self.completed_at(tick);
        self.patches[completed..]
            .iter()
            .take_while(|patch| patch.due_tick.0 == tick)
            .count()
    }
}

fn config_value_digest(value: &serde_json::Value) -> Result<blake3::Hash, String> {
    serde_json::to_vec(value)
        .map(|bytes| blake3::hash(&bytes))
        .map_err(|error| error.to_string())
}

/// Interventions use nested config-layer objects, not dotted override keys. Only
/// the registered live numeric knobs are executable; launch-only geometry, seed,
/// recording policy and unknown fields cannot be changed by a scheduled patch.
fn validate_live_intervention_patch(value: &serde_json::Value, path: &str) -> Result<(), String> {
    if let Some(object) = value.as_object() {
        if object.is_empty() {
            return Err("intervention patch objects must not be empty".to_owned());
        }
        for (key, child) in object {
            if key.is_empty() || key.contains('.') {
                return Err(
                    "intervention patches require nested objects with ordinary config keys"
                        .to_owned(),
                );
            }
            let child_path = if path.is_empty() {
                key.clone()
            } else {
                format!("{path}.{key}")
            };
            validate_live_intervention_patch(child, &child_path)?;
        }
        return Ok(());
    }
    let range = knob_range(path)
        .ok_or_else(|| format!("intervention contains unknown or unsupported knob `{path}`"))?;
    if range.fresh_world_only {
        return Err(format!("intervention knob `{path}` requires a fresh world"));
    }
    let number = value
        .as_f64()
        .ok_or_else(|| format!("intervention knob `{path}` requires a finite numeric value"))?;
    if !number.is_finite() || number < range.min || number > range.max {
        return Err(format!(
            "intervention knob `{path}` value {number} is outside its accepted range [{}, {}]",
            range.min, range.max
        ));
    }
    Ok(())
}

/// Bind the complete observed prefix and pending suffix to the declared inputs.
/// A final config alone cannot establish that noncommuting same-tick statements
/// actually ran in their declared order, and a cursor alone proves neither half.
fn validate_checkpoint_interventions(
    path: &Path,
    host: &HostCheckpointV1,
    prepared: &PreparedInterventions,
) -> Result<(), ExperimentRunnerError> {
    let completed = prepared.completed_at(host.tick().0);
    let completed_sequence =
        u64::try_from(completed).map_err(|error| invalid_checkpoint(path, error))?;
    let history = host.scheduled_patch_history();
    let expected_remaining = if prepared.patches.is_empty() {
        None
    } else {
        Some(&prepared.patches[completed..])
    };
    let expected_sequence = expected_remaining.map(|_| completed_sequence);
    let actual_config =
        serde_json::to_value(host.config()).map_err(|error| invalid_checkpoint(path, error))?;
    if prepared.patches.len() > host.scheduled_patch_history_capacity()
        || host.remaining_schedule() != expected_remaining
        || host.completed_schedule_sequence() != expected_sequence
        || history.len() != completed
        || host.config_revision().get() != completed_sequence
        || config_value_digest(&actual_config).map_err(|error| invalid_checkpoint(path, error))?
            != prepared.config_digests[completed]
    {
        return Err(invalid_checkpoint(
            path,
            "owner schedule cursor, retained receipts, pending inputs or current config differ from the declared intervention prefix",
        ));
    }
    let mut previous_journal_sequence = 0;
    for (index, status) in history.iter().enumerate() {
        let expected = &prepared.patches[index];
        let outcome_matches = matches!(
            &status.evidence.outcome,
            ScheduledPatchOutcome::Applied { changed_paths }
                if changed_paths == &prepared.changed_paths[index]
        );
        if status.evidence.scheduled != *expected
            || status.evidence.boundary.tick != expected.due_tick
            || status.evidence.boundary.revisions.scientific.get() != expected.due_tick.0
            || status.evidence.boundary.revisions.config.get() != expected.sequence
            || status.batch_id.session_id() != host.session_id()
            || status.batch_id.sequence() <= previous_journal_sequence
            || status.batch_id.sequence() > host.journal_cut().last_journal_sequence
            || status.journal != JournalState::CommittedVolatile
            || !outcome_matches
        {
            return Err(invalid_checkpoint(
                path,
                format!(
                    "intervention {} lacks its exact ordered applied and committed owner receipt",
                    expected.sequence
                ),
            ));
        }
        previous_journal_sequence = status.batch_id.sequence();
    }
    Ok(())
}

pub(crate) fn insert_dotted_value(
    root: &mut serde_json::Map<String, serde_json::Value>,
    path: &str,
    value: serde_json::Value,
) -> Result<(), String> {
    let segments = path.split('.').collect::<Vec<_>>();
    if segments.is_empty() || segments.iter().any(|segment| segment.is_empty()) {
        return Err(format!("invalid dotted knob path `{path}`"));
    }

    let mut object = root;
    for segment in &segments[..segments.len() - 1] {
        let entry = object
            .get_mut(*segment)
            .ok_or_else(|| format!("knob path `{path}` has unknown segment `{segment}`"))?;
        if entry.is_null() {
            *entry = serde_json::Value::Object(serde_json::Map::new());
        }
        object = entry
            .as_object_mut()
            .ok_or_else(|| format!("knob path `{path}` crosses non-object segment `{segment}`"))?;
    }
    let leaf = segments[segments.len() - 1];
    if !object.contains_key(leaf) {
        return Err(format!("knob path `{path}` has unknown leaf `{leaf}`"));
    }
    object.insert(leaf.to_owned(), value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress_runner(directory: &Path, seeds: Vec<u64>) -> MatchedSeedExperimentRunner {
        MatchedSeedExperimentRunner::new(
            "checkpoint-progress",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds,
            },
            vec![ScenarioVariant {
                variant_id: "mlp".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::from([
                    ("world_width".into(), serde_json::json!(40)),
                    ("world_height".into(), serde_json::json!(40)),
                    ("food_cell_size".into(), serde_json::json!(10)),
                ]),
                interventions: Vec::new(),
            }],
            3,
            2,
            directory.join("bundles"),
        )
    }

    #[test]
    fn checkpoint_progress_follows_successful_publication_not_attempted_science() {
        let directory = tempfile::tempdir().unwrap().keep();
        let runner = progress_runner(&directory, vec![7]);
        let state_file = directory.join("status.json");
        let mut status = runner.load_or_create_status(&state_file).unwrap();
        let plan = runner.validated_plan().unwrap();
        let record = &plan.runs[0];
        let (variant, family) = &plan.variants[&record.variant_id];
        let pending = runner
            .checkpoint_progress_snapshot(&state_file, &status, true)
            .unwrap();
        assert_eq!(
            pending[&record.run_id],
            ExperimentCheckpointProgress::Unavailable {
                reason: CheckpointProgressUnavailable::NotStarted,
            }
        );
        runner.prepare_progress_for_execution(&plan);
        runner
            .prepare_run_checkpoint(variant, *family, record.seed, &record.run_id, &plan.digest)
            .unwrap();
        status.runs[0].state = RunState::Running;
        runner.save_status(&state_file, &mut status).unwrap();
        let status_bytes = fs::read(&state_file).unwrap();
        let path = runner.run_checkpoint_path(&record.run_id);
        let initial_bytes = fs::read(&path).unwrap();
        let initial = runner
            .checkpoint_progress_snapshot(&state_file, &status, true)
            .unwrap();
        assert_eq!(
            initial[&record.run_id],
            ExperimentCheckpointProgress::Observed {
                tick: 0,
                checkpoint_digest: blake3::hash(&initial_bytes).to_hex().to_string(),
                source: CheckpointProgressSource::Published,
            }
        );

        let mut checkpoint = RunCheckpointV1::decode(&initial_bytes, &path).unwrap();
        let mut owner = restore_run_owner(&checkpoint, *family, &record.run_id, &path).unwrap();
        let (_, namespace) = host_identifiers(&record.run_id);
        let mut frontend = NullFrontend::new(owner.local_port(), namespace);
        let command = frontend
            .submit_envelope(CommandEnvelope::new(
                CommandId::from_client_sequence(namespace, 1),
                HostCommand::Step,
            ))
            .unwrap();
        let mut now = 0;
        drive_command_to_volatile_commit(
            &record.run_id,
            "progress test step",
            &mut frontend,
            &mut owner,
            command.command_id(),
            1,
            0,
            &mut now,
        )
        .unwrap();
        seal_run_checkpoint(&record.run_id, &mut frontend, &mut owner, &mut now).unwrap();
        checkpoint.owner = owner.capture_volatile_host_checkpoint_v1().unwrap();
        checkpoint.metadata.parent_digest = Some(blake3::hash(&initial_bytes).to_hex().to_string());
        let advanced_bytes = checkpoint.encode(&path).unwrap();
        let temporary = status_temporary_path(&path).unwrap();
        fs::write(&temporary, b"injected incomplete checkpoint write").unwrap();
        assert!(
            runner
                .publish_run_checkpoint(&path, &checkpoint, &advanced_bytes)
                .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), initial_bytes);
        assert_eq!(
            runner
                .checkpoint_progress_snapshot(&state_file, &status, true)
                .unwrap(),
            initial,
            "an advanced owner and failed write cannot publish progress"
        );
        fs::rename(&temporary, path.with_extension("preserved-write-fault")).unwrap();
        runner
            .publish_run_checkpoint(&path, &checkpoint, &advanced_bytes)
            .unwrap();
        let observed = runner
            .checkpoint_progress_snapshot(&state_file, &status, true)
            .unwrap();
        assert_eq!(
            observed[&record.run_id],
            ExperimentCheckpointProgress::Observed {
                tick: 1,
                checkpoint_digest: blake3::hash(&advanced_bytes).to_hex().to_string(),
                source: CheckpointProgressSource::Published,
            }
        );
        runner.observe_checkpoint_progress(
            &plan.digest,
            &record.run_id,
            0,
            blake3::hash(&initial_bytes).to_hex().to_string(),
            CheckpointProgressSource::Recovered,
        );
        assert_eq!(
            runner
                .checkpoint_progress_snapshot(&state_file, &status, true)
                .unwrap(),
            observed
        );
        assert_eq!(fs::read(&state_file).unwrap(), status_bytes);
        assert_eq!(status.runs[0].total_ticks, 0);
        assert_eq!(status.runs[0].state, RunState::Running);
        runner.observe_checkpoint_progress(
            &plan.digest,
            &record.run_id,
            1,
            blake3::hash(&initial_bytes).to_hex().to_string(),
            CheckpointProgressSource::Recovered,
        );
        runner.observe_checkpoint_progress(
            &plan.digest,
            &record.run_id,
            0,
            blake3::hash(&initial_bytes).to_hex().to_string(),
            CheckpointProgressSource::Recovered,
        );
        assert_eq!(
            runner
                .checkpoint_progress_snapshot(&state_file, &status, true)
                .unwrap()[&record.run_id],
            ExperimentCheckpointProgress::Refused {
                reason: CheckpointProgressRefusal::ConflictingObservation,
            },
            "conflicting evidence cannot reset the monotonicity guard"
        );
    }

    #[test]
    fn checkpoint_progress_cold_recovery_is_bounded_read_only_and_rejects_foreign_cuts() {
        let directory = tempfile::tempdir().unwrap().keep();
        let runner = progress_runner(&directory, vec![7, 9]);
        let state_file = directory.join("status.json");
        let mut status = runner.load_or_create_status(&state_file).unwrap();
        let plan = runner.validated_plan().unwrap();
        for record in &mut status.runs {
            let (variant, family) = &plan.variants[&record.variant_id];
            runner
                .prepare_run_checkpoint(variant, *family, record.seed, &record.run_id, &plan.digest)
                .unwrap();
            record.state = RunState::Running;
        }
        runner.save_status(&state_file, &mut status).unwrap();
        let status_bytes = fs::read(&state_file).unwrap();
        let paths = status
            .runs
            .iter()
            .map(|run| runner.run_checkpoint_path(&run.run_id))
            .collect::<Vec<_>>();
        let cuts = paths
            .iter()
            .map(|path| fs::read(path).unwrap())
            .collect::<Vec<_>>();
        let names = fs::read_dir(&runner.output_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<BTreeSet<_>>();
        let cold = progress_runner(&directory, vec![7, 9]);
        let active = cold
            .checkpoint_progress_snapshot(&state_file, &status, true)
            .unwrap();
        assert!(
            active.values().all(|progress| matches!(
                progress,
                ExperimentCheckpointProgress::Unavailable {
                    reason: CheckpointProgressUnavailable::AwaitingPublication,
                }
            )),
            "active polling must not decode existing files"
        );
        let first = cold
            .checkpoint_progress_snapshot(&state_file, &status, false)
            .unwrap();
        assert_eq!(
            first
                .values()
                .filter(|progress| progress.observed_tick().is_some())
                .count(),
            1
        );
        let second = cold
            .checkpoint_progress_snapshot(&state_file, &status, false)
            .unwrap();
        for (index, record) in status.runs.iter().enumerate() {
            assert_eq!(
                second[&record.run_id],
                ExperimentCheckpointProgress::Observed {
                    tick: 0,
                    checkpoint_digest: blake3::hash(&cuts[index]).to_hex().to_string(),
                    source: CheckpointProgressSource::Recovered,
                }
            );
        }
        assert_eq!(fs::read(&state_file).unwrap(), status_bytes);
        assert_eq!(
            names,
            fs::read_dir(&runner.output_dir)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect()
        );
        for (path, bytes) in paths.iter().zip(&cuts) {
            assert_eq!(fs::read(path).unwrap(), *bytes);
        }

        // Cached progress is explicitly an observation of a past cut, not a
        // substitute for current readback or resume validation.
        fs::copy(&paths[0], paths[0].with_extension("preserved-original")).unwrap();
        fs::write(&paths[0], b"corrupt checkpoint").unwrap();
        assert_eq!(
            cold.checkpoint_progress_snapshot(&state_file, &status, false)
                .unwrap(),
            second
        );
        let refused = progress_runner(&directory, vec![7, 9]);
        let corrupt = refused
            .checkpoint_progress_snapshot(&state_file, &status, false)
            .unwrap();
        assert_eq!(
            corrupt[&status.runs[0].run_id],
            ExperimentCheckpointProgress::Refused {
                reason: CheckpointProgressRefusal::InvalidCheckpoint,
            }
        );
        fs::copy(&paths[1], paths[1].with_extension("preserved-original")).unwrap();
        fs::write(&paths[1], &cuts[0]).unwrap();
        let foreign = refused
            .checkpoint_progress_snapshot(&state_file, &status, false)
            .unwrap();
        assert!(
            foreign.values().all(|progress| matches!(
                progress,
                ExperimentCheckpointProgress::Refused {
                    reason: CheckpointProgressRefusal::InvalidCheckpoint,
                }
            )),
            "a byte-valid cut for another seed must refuse"
        );
        for (path, bytes) in paths.iter().zip(&cuts) {
            fs::write(path, bytes).unwrap();
        }
        assert_eq!(
            refused
                .checkpoint_progress_snapshot(&state_file, &status, false)
                .unwrap(),
            foreign
        );
        let completed = refused.execute_batch(&state_file).unwrap();
        assert_eq!(
            completed.completed_runs, 2,
            "explicit resume retries repaired cuts"
        );
        let resumed = refused
            .checkpoint_progress_snapshot(&state_file, &completed, false)
            .unwrap();
        assert!(
            resumed
                .values()
                .all(|progress| progress.observed_tick() == Some(3))
        );
    }

    #[test]
    fn endpoint_summary_reports_observed_state_and_preserves_unavailable_empty_statistics() {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 40,
            world_height: 40,
            food_cell_size: 20,
            initial_food: 0.0,
            rng_seed: Some(7),
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        world
            .try_update_food(|cells| cells.copy_from_slice(&[1.0, 2.0, 4.0, 8.0]))
            .unwrap();
        let empty = ExperimentEndpointSummaryV2::from_world(
            &world,
            7,
            "mlp.baseline",
            "observed-empty-state".into(),
        )
        .unwrap();
        assert_eq!(
            empty.available_metrics(),
            BTreeMap::from([
                ("alive_agents".into(), 0.0),
                ("total_energy".into(), 0.0),
                ("total_food".into(), 15.0),
            ])
        );
        assert_eq!(empty.average_energy, None);
        assert_eq!(empty.average_health, None);
        assert_eq!(empty.mean_age, None);
        assert_eq!(empty.max_age, None);
        assert_eq!(empty.max_generation, None);

        for (energy, health, age, generation) in [(2.0, 1.0, 3, 1), (6.0, 3.0, 9, 4)] {
            world
                .try_spawn_agent_with(
                    scriptbots_core::AgentData {
                        health,
                        age,
                        generation: scriptbots_core::Generation(generation),
                        ..scriptbots_core::AgentData::default()
                    },
                    |runtime| runtime.energy = energy,
                )
                .unwrap();
        }
        let populated = ExperimentEndpointSummaryV2::from_world(
            &world,
            7,
            "mlp.baseline",
            "observed-populated-state".into(),
        )
        .unwrap();
        assert_eq!(
            populated.available_metrics(),
            BTreeMap::from([
                ("alive_agents".into(), 2.0),
                ("total_energy".into(), 8.0),
                ("average_energy".into(), 4.0),
                ("average_health".into(), 2.0),
                ("mean_age".into(), 6.0),
                ("max_age".into(), 9.0),
                ("max_generation".into(), 4.0),
                ("total_food".into(), 15.0),
            ])
        );
        for summary in [empty, populated] {
            let bytes = summary.canonical_csv_bytes().unwrap();
            assert!(bytes.len() <= MAX_EXPERIMENT_SUMMARY_BYTES);
            let mut reader = csv::Reader::from_reader(bytes.as_slice());
            let mut rows = reader.deserialize::<ExperimentEndpointSummaryV2>();
            assert_eq!(rows.next().unwrap().unwrap(), summary);
            assert!(rows.next().is_none());
        }
    }

    #[test]
    fn experiment_status_snapshot_is_plan_bound_and_read_only() {
        let directory = tempfile::tempdir().unwrap().keep();
        let state_file = directory.join("status.json");
        let runner = MatchedSeedExperimentRunner::new(
            "status-reader",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![7],
            },
            vec![ScenarioVariant {
                variant_id: "baseline".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            }],
            2,
            1,
            directory.join("bundles"),
        );
        let planned = runner.status_snapshot(&state_file).expect("initial plan");
        assert_eq!(planned.schema_version, STATUS_SCHEMA_VERSION);
        assert_eq!(planned.generation, 0);
        assert_eq!(planned.total_runs, 1);
        assert_eq!(
            fs::read_dir(&directory).unwrap().count(),
            0,
            "read created files"
        );
        let persisted = runner
            .load_or_create_status(&state_file)
            .expect("real writer");
        assert_eq!(
            planned, persisted,
            "initial reader must bind the writer's exact plan"
        );
        let original = fs::read(&state_file).unwrap();
        let _lease = ExperimentStatusWriterLease::acquire(&state_file).expect("hold writer lease");
        let names_before: BTreeSet<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        let observed = runner
            .status_snapshot(&state_file)
            .expect("read alongside writer lease");
        assert_eq!(observed, persisted);
        assert_eq!(fs::read(&state_file).unwrap(), original);
        let names_after: BTreeSet<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(
            names_before, names_after,
            "read must not create recovery or lock files"
        );
    }

    #[cfg(unix)]
    #[test]
    fn experiment_status_reads_refuse_aliases_without_reinitializing_them() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let runner = interrupted_status_runner(directory.path());
        let original_path = directory.path().join("original.json");
        let original = runner.load_or_create_status(&original_path).unwrap();
        let bytes = fs::read(&original_path).unwrap();
        let linked = directory.path().join("linked.json");
        symlink(&original_path, &linked).unwrap();
        let missing_target = directory.path().join("missing.json");
        let dangling = directory.path().join("dangling.json");
        symlink(&missing_target, &dangling).unwrap();
        let hardlinked = directory.path().join("hardlinked.json");
        let aliased_target = directory.path().join("aliased-target.json");
        fs::write(&aliased_target, &bytes).unwrap();
        fs::hard_link(&aliased_target, &hardlinked).unwrap();
        let non_file = directory.path().join("directory.json");
        fs::create_dir(&non_file).unwrap();

        for path in [&linked, &dangling, &hardlinked, &non_file] {
            assert!(
                matches!(
                    runner.status_snapshot(path),
                    Err(ExperimentRunnerError::InvalidStatus { .. })
                ),
                "reader admitted {}",
                path.display()
            );
            assert!(
                matches!(
                    runner.load_or_create_status(path),
                    Err(ExperimentRunnerError::InvalidStatus { .. })
                ),
                "writer admitted or reinitialized {}",
                path.display()
            );
            assert!(!status_temporary_path(path).unwrap().exists());
        }
        assert_eq!(fs::read_link(&linked).unwrap(), original_path);
        assert_eq!(fs::read_link(&dangling).unwrap(), missing_target);
        assert!(!missing_target.exists());
        assert_eq!(fs::read(&hardlinked).unwrap(), bytes);
        assert_eq!(fs::read(&original_path).unwrap(), bytes);
        assert_eq!(runner.status_snapshot(&original_path).unwrap(), original);
        assert!(!runner.output_dir.exists());
    }

    #[cfg(unix)]
    #[test]
    fn experiment_status_fifo_refusal_does_not_wait_for_a_writer() {
        use std::os::unix::fs::FileTypeExt;
        use std::process::Command;
        use std::time::Duration;

        const CHILD_DIRECTORY: &str = "SCRIPTBOTS_STATUS_FIFO_TEST_DIRECTORY";
        if let Some(directory) = std::env::var_os(CHILD_DIRECTORY) {
            let directory = PathBuf::from(directory);
            let runner = interrupted_status_runner(&directory);
            let state_file = directory.join("status.json");
            assert!(matches!(
                runner.status_snapshot(&state_file),
                Err(ExperimentRunnerError::InvalidStatus { .. })
            ));
            assert!(matches!(
                runner.load_or_create_status(&state_file),
                Err(ExperimentRunnerError::InvalidStatus { .. })
            ));
            assert!(!runner.output_dir.exists());
            fs::write(
                directory.join("refused.txt"),
                b"reader and writer refused FIFO",
            )
            .unwrap();
            return;
        }

        let directory = tempfile::tempdir().unwrap();
        let state_file = directory.path().join("status.json");
        assert!(
            Command::new("mkfifo")
                .arg(&state_file)
                .status()
                .unwrap()
                .success()
        );
        // Isolate the actual reader so restoring the blocking-open defect fails
        // with a reaped timeout instead of hanging the test harness indefinitely.
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "experiment_runner::tests::experiment_status_fifo_refusal_does_not_wait_for_a_writer",
                "--nocapture",
            ])
            .env(CHILD_DIRECTORY, directory.path());
        MatchedSeedExperimentRunner::run_bounded_child(&mut command, Duration::from_secs(10))
            .expect("status reads must refuse a FIFO without an external writer");
        assert_eq!(
            fs::read(directory.path().join("refused.txt")).unwrap(),
            b"reader and writer refused FIFO",
            "the child must execute both refusals, not merely match zero tests"
        );
        assert!(
            fs::symlink_metadata(&state_file)
                .unwrap()
                .file_type()
                .is_fifo()
        );
    }

    #[test]
    fn experiment_status_reads_accept_complete_atomic_replacements() {
        const REPLACEMENTS: u64 = 128;

        let directory = tempfile::tempdir().unwrap();
        let runner = interrupted_status_runner(directory.path());
        let state_file = directory.path().join("status.json");
        let initial = runner.load_or_create_status(&state_file).unwrap();
        let mut running = initial.clone();
        running.runs[0].state = RunState::Running;
        runner.save_status(&state_file, &mut running).unwrap();
        let mut failed = running.clone();
        failed.runs[0].state = RunState::Failed;
        failed.runs[0].error_reason = Some("retained test failure".to_owned());
        recompute_status_counts(&mut failed);
        let last_generation = running.generation + REPLACEMENTS;

        let start = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let writer = scope.spawn(|| {
                let mut status = failed.clone();
                start.wait();
                for _ in 0..REPLACEMENTS {
                    // Exercise actual atomic publication while the reader opens
                    // either complete generation. No partial in-place writes.
                    runner.save_status(&state_file, &mut status).unwrap();
                }
                status
            });
            start.wait();
            for _ in 0..REPLACEMENTS {
                let observed = runner.status_snapshot(&state_file).unwrap();
                if observed.generation == running.generation {
                    assert_eq!(observed, running);
                } else {
                    assert!(
                        (running.generation + 1..=last_generation).contains(&observed.generation),
                        "{observed:?}"
                    );
                    let mut expected = failed.clone();
                    expected.generation = observed.generation;
                    assert_eq!(observed, expected);
                }
            }
            let committed = writer.join().unwrap();
            assert_eq!(committed.generation, last_generation);
            assert_eq!(runner.status_snapshot(&state_file).unwrap(), committed);
        });
        assert!(!runner.output_dir.exists());
    }

    #[test]
    fn canonical_arm_application_is_strict_and_type_preserving() {
        let overrides = BTreeMap::from([
            ("world_width".to_owned(), serde_json::json!(5_000)),
            ("food_growth_rate".to_owned(), serde_json::json!(0.2)),
            ("food_transfer_rate".to_owned(), serde_json::json!(0.25)),
            ("population_minimum".to_owned(), serde_json::json!(2)),
        ]);
        let config = config_for_run(&overrides, 99).expect("canonical arm applies");
        assert_eq!(config.world_width, 5_000);
        assert!((config.food_growth_rate - 0.2).abs() < f32::EPSILON);
        assert!((config.food_transfer_rate - 0.25).abs() < f32::EPSILON);
        assert_eq!(config.population_minimum, 2);
        assert_eq!(config.rng_seed, Some(99));
        assert_eq!(config.persistence_interval, 1);

        let unknown = BTreeMap::from([("food.regrowth".to_owned(), serde_json::json!(0.1))]);
        assert!(
            config_for_run(&unknown, 1)
                .expect_err("unknown overrides must never disappear")
                .contains("unknown or unsupported knob")
        );

        let default_json =
            serde_json::to_value(ScriptBotsConfig::default()).expect("default config schema");
        for range in scriptbots_core::KNOB_RANGES {
            if range.path.starts_with("mutation.") || range.path.starts_with("render.") {
                continue;
            }
            let pointer = format!("/{}", range.path.replace('.', "/"));
            assert!(
                default_json.pointer(&pointer).is_some(),
                "canonical validator accepts `{}`, but the strict runner config has no such field",
                range.path
            );
        }
    }

    #[test]
    fn experiment_runner_planning_binds_real_families_and_matched_seed_order() {
        let cohort = MatchedSeedCohort {
            cohort_id: "cohort-1".into(),
            seeds: vec![100, 200, 300],
        };
        let variants = vec![
            ScenarioVariant {
                variant_id: "mlp_base".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            },
            ScenarioVariant {
                variant_id: "dwraon_base".into(),
                brain_family: "dwraon".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            },
        ];

        let runner =
            MatchedSeedExperimentRunner::new("exp-001", cohort, variants, 50, 2, "/tmp/exp_test");

        let plan = runner.plan_batch().expect("valid matched-seed plan");
        assert_eq!(plan.len(), 6);
        assert_eq!(plan[0].run_id, "exp-001-mlp_base-seed100");
        assert_eq!(plan[0].brain_family, "mlp.baseline");
        assert_eq!(plan[3].run_id, "exp-001-dwraon_base-seed100");
        assert_eq!(plan[3].brain_family, "dwraon.baseline");
    }

    #[test]
    fn frankentorch_arm_admission_matches_the_compiled_family() {
        let directory = tempfile::tempdir().unwrap();
        for requested_family in ["ft", "frankentorch"] {
            let variant = ScenarioVariant {
                variant_id: "torch".into(),
                brain_family: requested_family.into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            };
            let runner = MatchedSeedExperimentRunner::new(
                "torch-plan",
                MatchedSeedCohort {
                    cohort_id: "cohort".into(),
                    seeds: vec![71],
                },
                vec![variant.clone()],
                2,
                1,
                directory.path().join("bundles"),
            );
            #[cfg(feature = "brain-ft")]
            {
                validate_experiment_brain_family(&variant.variant_id, requested_family)
                    .expect("compiled family is admitted by the control plane");
                let plan = runner.plan_batch().expect("compiled family is executable");
                assert_eq!(plan[0].brain_family, "ft");

                let duplicate = MatchedSeedExperimentRunner {
                    variants: vec![
                        variant,
                        ScenarioVariant {
                            variant_id: "same-science".into(),
                            brain_family: "ft".into(),
                            config_overrides: BTreeMap::new(),
                            interventions: Vec::new(),
                        },
                    ],
                    ..runner
                };
                assert!(matches!(
                    duplicate.plan_batch(),
                    Err(ExperimentRunnerError::InvalidPlan { .. })
                ));
            }
            #[cfg(not(feature = "brain-ft"))]
            {
                for result in [
                    validate_experiment_brain_family(&variant.variant_id, requested_family),
                    runner.plan_batch().map(|_| ()),
                ] {
                    let error = result.expect_err("missing family must not silently fall back");
                    assert!(matches!(
                        error,
                        ExperimentRunnerError::UnavailableBrainFamily { .. }
                    ));
                    assert!(error.to_string().contains("brain-ft"));
                }
            }
        }
        assert!(directory.path().read_dir().unwrap().next().is_none());
    }

    #[test]
    fn neuroflow_arm_admission_matches_the_compiled_family() {
        let directory = tempfile::tempdir().unwrap();
        for requested_family in ["neuro", "neuroflow", "ml.neuroflow"] {
            let variant = ScenarioVariant {
                variant_id: "neuro".into(),
                brain_family: requested_family.into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            };
            let runner = MatchedSeedExperimentRunner::new(
                "neuro-plan",
                MatchedSeedCohort {
                    cohort_id: "cohort".into(),
                    seeds: vec![71],
                },
                vec![variant.clone()],
                2,
                1,
                directory.path().join("bundles"),
            );
            #[cfg(feature = "neuro")]
            {
                assert_eq!(
                    canonical_experiment_brain_family(&variant.variant_id, requested_family)
                        .unwrap(),
                    "ml.neuroflow"
                );
                assert!(EXPERIMENT_BRAIN_FAMILIES.contains(&"ml.neuroflow"));
                let plan = runner.plan_batch().expect("compiled family is executable");
                assert_eq!(plan[0].brain_family, "ml.neuroflow");

                let duplicate = MatchedSeedExperimentRunner {
                    variants: vec![
                        variant,
                        ScenarioVariant {
                            variant_id: "same-science".into(),
                            brain_family: "ml.neuroflow".into(),
                            config_overrides: BTreeMap::new(),
                            interventions: Vec::new(),
                        },
                    ],
                    ..runner
                };
                assert!(matches!(
                    duplicate.plan_batch(),
                    Err(ExperimentRunnerError::InvalidPlan { .. })
                ));
            }
            #[cfg(not(feature = "neuro"))]
            {
                assert!(!EXPERIMENT_BRAIN_FAMILIES.contains(&"ml.neuroflow"));
                for result in [
                    validate_experiment_brain_family(&variant.variant_id, requested_family),
                    runner.plan_batch().map(|_| ()),
                    runner
                        .load_or_create_status(&directory.path().join("status.json"))
                        .map(|_| ()),
                ] {
                    let error = result.expect_err("missing family must not silently fall back");
                    assert!(matches!(
                        error,
                        ExperimentRunnerError::UnavailableBrainFamily { .. }
                    ));
                    assert!(error.to_string().contains("neuro feature"));
                }
            }
        }
        assert!(directory.path().read_dir().unwrap().next().is_none());
    }

    #[test]
    fn single_run_refuses_an_arm_changed_after_plan_validation() {
        let planned_variant = ScenarioVariant {
            variant_id: "mlp_base".into(),
            brain_family: "mlp".into(),
            config_overrides: BTreeMap::new(),
            interventions: Vec::new(),
        };
        let runner = MatchedSeedExperimentRunner::new(
            "exp-immutable-arm",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![100],
            },
            vec![planned_variant.clone()],
            1,
            1,
            "/tmp/unused-immutable-arm-output",
        );
        let mut changed_variant = planned_variant;
        changed_variant
            .config_overrides
            .insert("world_width".into(), serde_json::json!(2_000));

        assert!(matches!(
            runner.execute_single_run(&changed_variant, 100),
            Err(ExperimentRunnerError::InvalidPlan { .. })
        ));
    }

    #[test]
    fn plan_preflight_refuses_ambiguous_or_unavailable_execution() {
        let base_variant = ScenarioVariant {
            variant_id: "baseline".into(),
            brain_family: "mlp".into(),
            config_overrides: BTreeMap::new(),
            interventions: Vec::new(),
        };
        let output = PathBuf::from("/tmp/unused-experiment-output");

        let zero_concurrency = MatchedSeedExperimentRunner::new(
            "exp-zero",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![1],
            },
            vec![base_variant.clone()],
            1,
            0,
            &output,
        );
        assert!(matches!(
            zero_concurrency.plan_batch(),
            Err(ExperimentRunnerError::InvalidPlan { .. })
        ));

        let duplicate_seed = MatchedSeedExperimentRunner::new(
            "exp-duplicate-seed",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![1, 1],
            },
            vec![base_variant.clone()],
            1,
            1,
            &output,
        );
        assert!(matches!(
            duplicate_seed.plan_batch(),
            Err(ExperimentRunnerError::InvalidPlan { .. })
        ));

        let duplicate_arm = MatchedSeedExperimentRunner::new(
            "exp-duplicate-arm",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![1],
            },
            vec![
                base_variant.clone(),
                ScenarioVariant {
                    variant_id: "same-science".into(),
                    ..base_variant.clone()
                },
            ],
            1,
            1,
            &output,
        );
        assert!(matches!(
            duplicate_arm.plan_batch(),
            Err(ExperimentRunnerError::InvalidPlan { .. })
        ));

        let unknown = MatchedSeedExperimentRunner::new(
            "exp-unknown",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![1],
            },
            vec![ScenarioVariant {
                brain_family: "imaginary".into(),
                ..base_variant.clone()
            }],
            1,
            1,
            &output,
        );
        assert!(matches!(
            unknown.plan_batch(),
            Err(ExperimentRunnerError::UnknownBrainFamily { .. })
        ));

        #[cfg(not(feature = "neuro"))]
        {
            let unavailable = MatchedSeedExperimentRunner::new(
                "exp-unavailable",
                MatchedSeedCohort {
                    cohort_id: "cohort".into(),
                    seeds: vec![1],
                },
                vec![ScenarioVariant {
                    brain_family: "neuroflow".into(),
                    ..base_variant
                }],
                1,
                1,
                output,
            );
            assert!(matches!(
                unavailable.plan_batch(),
                Err(ExperimentRunnerError::UnavailableBrainFamily { .. })
            ));
        }
    }

    fn grid_test_variant(
        variant_id: &str,
        width: u32,
        height: u32,
        cell_size: u32,
    ) -> ScenarioVariant {
        ScenarioVariant {
            variant_id: variant_id.to_owned(),
            brain_family: "mlp".to_owned(),
            config_overrides: BTreeMap::from([
                ("world_width".to_owned(), serde_json::json!(width)),
                ("world_height".to_owned(), serde_json::json!(height)),
                ("food_cell_size".to_owned(), serde_json::json!(cell_size)),
            ]),
            interventions: Vec::new(),
        }
    }

    #[test]
    fn experiment_grid_preflight_uses_actual_dimensions_and_rejects_one_cell_over() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("unallocated");
        let boundary = grid_test_variant("boundary", 16_384, 16_384, 32);
        let config = config_for_run(&boundary.config_overrides, 1).unwrap();
        let (width, height) = config.food_dimensions().unwrap();
        assert_eq!((width, height), (512, 512));
        assert_eq!(u64::from(width) * u64::from(height), MAX_RUN_GRID_CELLS);
        let mut runner = MatchedSeedExperimentRunner::new(
            "grid-boundary",
            MatchedSeedCohort {
                cohort_id: "grid".to_owned(),
                seeds: vec![1],
            },
            vec![boundary],
            1,
            MAX_CONCURRENT_RUNS,
            &output,
        );
        assert_eq!(runner.plan_batch().unwrap().len(), 1);
        assert_eq!(runner.concurrent_grid_cells().unwrap(), MAX_RUN_GRID_CELLS);

        // Both extents remain within the existing knob ranges; only their
        // normalized product exceeds the new owner allocation limit.
        runner.variants = vec![grid_test_variant("one-over", 65, 4_033, 1)];
        assert!(matches!(
            runner.plan_batch(),
            Err(ExperimentRunnerError::RunGridCapacityExceeded { variant_id, cells, limit })
                if variant_id == "one-over"
                    && cells == MAX_RUN_GRID_CELLS + 1
                    && limit == MAX_RUN_GRID_CELLS
        ));
        runner.variants = vec![grid_test_variant("huge", 20_000, 20_000, 1)];
        assert!(matches!(
            runner.plan_batch(),
            Err(ExperimentRunnerError::RunGridCapacityExceeded {
                cells: 400_000_000,
                ..
            })
        ));
        assert!(
            !output.exists(),
            "preflight must never construct a world or write a cut"
        );
    }

    #[test]
    fn experiment_grid_preflight_preserves_default_and_small_mixed_cohorts() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("unallocated");
        let mut runner = MatchedSeedExperimentRunner::new(
            "grid-default",
            MatchedSeedCohort {
                cohort_id: "grid".to_owned(),
                seeds: (0..u64::try_from(MAX_CONCURRENT_RUNS).unwrap()).collect(),
            },
            vec![ScenarioVariant {
                variant_id: "default".to_owned(),
                brain_family: "mlp".to_owned(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            }],
            1,
            MAX_CONCURRENT_RUNS,
            &output,
        );
        let (width, height) = ScriptBotsConfig::default().food_dimensions().unwrap();
        let expected =
            u64::from(width) * u64::from(height) * u64::try_from(MAX_CONCURRENT_RUNS).unwrap();
        assert_eq!(runner.plan_batch().unwrap().len(), MAX_CONCURRENT_RUNS);
        assert_eq!(runner.concurrent_grid_cells().unwrap(), expected);
        assert!(expected <= MAX_CONCURRENT_GRID_CELLS);

        runner.cohort.seeds = vec![1];
        runner.variants = vec![
            grid_test_variant("large", 512, 512, 1),
            grid_test_variant("small", 32, 32, 32),
        ];
        assert_eq!(runner.plan_batch().unwrap().len(), 2);
        assert_eq!(
            runner.concurrent_grid_cells().unwrap(),
            MAX_RUN_GRID_CELLS + 1
        );
        assert!(
            !output.exists(),
            "capacity discovery must not allocate owners or touch their artifact directory"
        );
    }

    #[test]
    fn experiment_grid_preflight_counts_matched_seeds_and_resume_wave_reordering() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("unallocated");
        let mut runner = MatchedSeedExperimentRunner::new(
            "grid-waves",
            MatchedSeedCohort {
                cohort_id: "grid".to_owned(),
                seeds: vec![1, 2, 3, 4],
            },
            vec![
                grid_test_variant("large", 512, 512, 1),
                grid_test_variant("small", 32, 32, 32),
            ],
            1,
            4,
            &output,
        );
        assert_eq!(runner.plan_batch().unwrap().len(), 8);
        assert_eq!(
            runner.concurrent_grid_cells().unwrap(),
            MAX_CONCURRENT_GRID_CELLS
        );
        let digest = runner.plan_digest().unwrap();
        runner.max_concurrency = 3;
        assert_eq!(
            runner.concurrent_grid_cells().unwrap(),
            3 * MAX_RUN_GRID_CELLS
        );
        assert_eq!(
            runner.plan_digest().unwrap(),
            digest,
            "capacity does not change the science plan"
        );

        runner.max_concurrency = 5;
        for _ in 0..2 {
            assert!(matches!(
                runner.plan_batch(),
                Err(ExperimentRunnerError::ConcurrentGridCapacityExceeded {
                    concurrent_runs: 5, cells, limit,
                }) if cells == MAX_CONCURRENT_GRID_CELLS + 1
                    && limit == MAX_CONCURRENT_GRID_CELLS
            ));
            // When the small arm comes first, the original five-run wave holds
            // only one large world. Filtering completed companions on resume
            // can pack all four large worlds with a small one into the same wave.
            runner.variants.reverse();
        }
        assert!(!output.exists());
    }

    #[test]
    fn experiment_grid_preflight_guards_status_recovery_and_execution_before_io() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("unallocated");
        let state_file = directory.path().join("status.json");
        let mut runner = MatchedSeedExperimentRunner::new(
            "grid-recovery",
            MatchedSeedCohort {
                cohort_id: "grid".to_owned(),
                seeds: vec![1, 2, 3, 4],
            },
            vec![
                grid_test_variant("large", 512, 512, 1),
                grid_test_variant("small", 32, 32, 32),
            ],
            1,
            4,
            &output,
        );
        let mut status = runner.load_or_create_status(&state_file).unwrap();
        status.runs[0].state = RunState::Running;
        runner.save_status(&state_file, &mut status).unwrap();
        let committed = fs::read(&state_file).unwrap();
        runner.max_concurrency = 5;
        let outcomes = [
            runner.plan_digest().map(|_| ()),
            runner.status_snapshot(&state_file).map(|_| ()),
            runner.load_or_create_status(&state_file).map(|_| ()),
            runner.save_status(&state_file, &mut status),
            runner.verify_completed_status(&state_file, &status),
            runner.execute_batch(&state_file).map(|_| ()),
            runner
                .execute_single_run(&runner.variants[0], 1)
                .map(|_| ()),
        ];
        for result in outcomes {
            assert!(matches!(
                result,
                Err(ExperimentRunnerError::ConcurrentGridCapacityExceeded { .. })
            ));
        }
        assert_eq!(fs::read(&state_file).unwrap(), committed);
        assert!(!status_temporary_path(&state_file).unwrap().exists());
        assert!(
            !output.exists(),
            "oversized resume must not allocate a world or write a cut"
        );

        let unopened_state = directory.path().join("unopened").join("status.json");
        assert!(matches!(
            runner.load_or_create_status(&unopened_state),
            Err(ExperimentRunnerError::ConcurrentGridCapacityExceeded { .. })
        ));
        assert!(!unopened_state.parent().unwrap().exists());
    }

    #[test]
    fn experiment_population_preflight_checks_initial_and_every_scheduled_prefix_before_io() {
        let directory = tempfile::tempdir().unwrap().keep();
        let mut runner = progress_runner(&directory, vec![7]);
        let state_file = directory.join("unopened").join("status.json");
        for floor in [MAX_RUN_POPULATION - 1, MAX_RUN_POPULATION] {
            runner.variants[0]
                .config_overrides
                .insert("population_minimum".into(), serde_json::json!(floor));
            assert_eq!(runner.plan_batch().unwrap().len(), 1);
        }
        runner.variants[0].config_overrides.insert(
            "population_minimum".into(),
            serde_json::json!(MAX_RUN_POPULATION + 1),
        );
        let outcomes = [
            runner.plan_batch().map(|_| ()),
            runner.load_or_create_status(&state_file).map(|_| ()),
            runner.execute_batch(&state_file).map(|_| ()),
            runner
                .execute_single_run(&runner.variants[0], 7)
                .map(|_| ()),
        ];
        for outcome in outcomes {
            assert!(matches!(
                outcome,
                Err(ExperimentRunnerError::RunConfigCapacityExceeded {
                    variant_id,
                    configuration_step: 0,
                    resource: "population floor",
                    requested,
                    limit,
                }) if variant_id == "mlp"
                    && requested == MAX_RUN_POPULATION + 1
                    && limit == MAX_RUN_POPULATION
            ));
        }
        runner.variants[0]
            .config_overrides
            .insert("population_minimum".into(), serde_json::json!(0));
        runner.variants[0].interventions = vec![
            ScenarioInterventionV1 {
                tick: 0,
                set: serde_json::json!({"population_minimum": MAX_RUN_POPULATION}),
            },
            ScenarioInterventionV1 {
                tick: 1,
                set: serde_json::json!({"population_minimum": MAX_RUN_POPULATION + 1}),
            },
            ScenarioInterventionV1 {
                tick: 2,
                set: serde_json::json!({"population_minimum": 0}),
            },
        ];
        for outcome in [
            runner.plan_batch().map(|_| ()),
            runner.execute_batch(&state_file).map(|_| ()),
        ] {
            assert!(matches!(
                outcome,
                Err(ExperimentRunnerError::RunConfigCapacityExceeded {
                    configuration_step: 2,
                    resource: "population floor",
                    requested,
                    ..
                }) if requested == MAX_RUN_POPULATION + 1
            ));
        }
        assert!(!runner.output_dir.exists());
        assert!(!state_file.parent().unwrap().exists());
    }

    #[test]
    fn experiment_config_capacity_checks_the_allocating_history_and_population_fields() {
        let history_limit = usize::try_from(MAX_RUN_HISTORY_ENTRIES).unwrap();
        let population_limit = usize::try_from(MAX_RUN_POPULATION).unwrap();
        let accepted = ScriptBotsConfig {
            history_capacity: history_limit,
            narrative_capacity: history_limit,
            population_minimum: population_limit,
            population_spawn_count: u32::try_from(MAX_RUN_POPULATION).unwrap(),
            ..ScriptBotsConfig::default()
        };
        PreparedInterventions::new("boundary", &accepted, &[], 1).unwrap();
        for (resource, config) in [
            (
                "history entries",
                ScriptBotsConfig {
                    history_capacity: history_limit + 1,
                    ..accepted.clone()
                },
            ),
            (
                "narrative events",
                ScriptBotsConfig {
                    narrative_capacity: history_limit + 1,
                    ..accepted.clone()
                },
            ),
            (
                "population injection count",
                ScriptBotsConfig {
                    population_spawn_count: u32::try_from(MAX_RUN_POPULATION + 1).unwrap(),
                    ..accepted.clone()
                },
            ),
        ] {
            assert!(matches!(
                PreparedInterventions::new("oversized", &config, &[], 1),
                Err(ExperimentRunnerError::RunConfigCapacityExceeded {
                    resource: observed,
                    configuration_step: 0,
                    requested,
                    limit,
                    ..
                }) if observed == resource && requested == limit + 1
            ));
        }
    }

    fn population_budget_runner(directory: &Path) -> MatchedSeedExperimentRunner {
        let mut runner = progress_runner(directory, vec![7]);
        runner.variants[0].config_overrides.extend([
            ("population_spawn_interval".into(), serde_json::json!(1)),
            ("reproduction_attempt_chance".into(), serde_json::json!(0.0)),
            ("spike_damage".into(), serde_json::json!(0.0)),
            ("metabolism_drain".into(), serde_json::json!(0.0)),
            ("movement_drain".into(), serde_json::json!(0.0)),
        ]);
        runner
    }

    #[test]
    fn experiment_population_crossing_retains_the_last_cut_and_terminal_failure() {
        let directory = tempfile::tempdir().unwrap().keep();
        let runner = population_budget_runner(&directory);
        let state_file = directory.join("status.json");
        runner.load_or_create_status(&state_file).unwrap();
        let plan = runner.validated_plan().unwrap();
        let record = &plan.runs[0];
        let (variant, family) = &plan.variants[&record.variant_id];
        runner
            .prepare_run_checkpoint(variant, *family, record.seed, &record.run_id, &plan.digest)
            .unwrap();
        let path = runner.run_checkpoint_path(&record.run_id);
        let retained = fs::read(&path).unwrap();
        let checkpoint = RunCheckpointV1::decode(&retained, &path).unwrap();
        let population_limit = u64::try_from(
            checkpoint
                .owner
                .host_checkpoint()
                .world_checkpoint()
                .agent_count(),
        )
        .unwrap();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let budget = RunExecutionBudget {
            cancel: &cancel,
            population_limit,
        };
        let error = runner
            .execute_validated_run(
                variant,
                *family,
                record.seed,
                &record.run_id,
                &plan.digest,
                budget,
            )
            .unwrap_err();
        assert!(
            matches!(
                &error,
                ExperimentRunnerError::RunPopulationCapacityExceeded {
                    run_id,
                    tick: 1,
                    population,
                    limit,
                    retained_checkpoint_tick: Some(0),
                } if run_id == &record.run_id
                    && *population == population_limit + 1
                    && *limit == population_limit
            ),
            "{error}"
        );
        assert_eq!(fs::read(&path).unwrap(), retained);
        assert!(!status_temporary_path(&path).unwrap().exists());
        assert!(!runner.output_dir.join(&record.run_id).exists());

        let failed = runner
            .execute_batch_with_budget(&state_file, budget)
            .unwrap();
        assert_eq!(failed.completed_runs, 0);
        assert_eq!(failed.failed_runs, 1);
        assert_eq!(failed.runs[0].state, RunState::Failed);
        assert_eq!(failed.runs[0].total_ticks, 0);
        assert_eq!(
            failed.runs[0].error_reason.as_deref(),
            Some(error.to_string().as_str())
        );
        assert_eq!(fs::read(&path).unwrap(), retained);
        assert_eq!(
            runner
                .checkpoint_progress_snapshot(&state_file, &failed, false)
                .unwrap()[&record.run_id]
                .observed_tick(),
            Some(0),
        );
        let status_bytes = fs::read(&state_file).unwrap();
        assert_eq!(
            runner
                .execute_batch_with_budget(&state_file, budget)
                .unwrap(),
            failed
        );
        assert_eq!(fs::read(&state_file).unwrap(), status_bytes);
        assert_eq!(fs::read(&path).unwrap(), retained);
        assert!(!runner.output_dir.join(&record.run_id).exists());
    }

    #[test]
    fn experiment_recovery_refuses_population_growth_before_promoting_the_next_cut() {
        let directory = tempfile::tempdir().unwrap().keep();
        let runner = population_budget_runner(&directory);
        let plan = runner.validated_plan().unwrap();
        let record = &plan.runs[0];
        let (variant, family) = &plan.variants[&record.variant_id];
        runner
            .prepare_run_checkpoint(variant, *family, record.seed, &record.run_id, &plan.digest)
            .unwrap();
        let path = runner.run_checkpoint_path(&record.run_id);
        let temporary = status_temporary_path(&path).unwrap();
        let retained = fs::read(&path).unwrap();
        let mut checkpoint = RunCheckpointV1::decode(&retained, &path).unwrap();
        let mut owner = restore_run_owner(&checkpoint, *family, &record.run_id, &path).unwrap();
        let population_limit = u64::try_from(owner.world().agents().len()).unwrap();
        let (_, namespace) = host_identifiers(&record.run_id);
        let mut frontend = NullFrontend::new(owner.local_port(), namespace);
        let submitted = frontend
            .submit_envelope(CommandEnvelope::new(
                CommandId::from_client_sequence(namespace, 1),
                HostCommand::Step,
            ))
            .unwrap();
        let mut now = 0;
        drive_command_to_volatile_commit(
            &record.run_id,
            "over-cap temporary checkpoint step",
            &mut frontend,
            &mut owner,
            submitted.command_id(),
            1,
            0,
            &mut now,
        )
        .unwrap();
        seal_run_checkpoint(&record.run_id, &mut frontend, &mut owner, &mut now).unwrap();
        checkpoint.owner = owner.capture_volatile_host_checkpoint_v1().unwrap();
        checkpoint.metadata.parent_digest = Some(blake3::hash(&retained).to_hex().to_string());
        assert_eq!(checkpoint.owner.host_checkpoint().tick().0, 1);
        assert_eq!(
            u64::try_from(
                checkpoint
                    .owner
                    .host_checkpoint()
                    .world_checkpoint()
                    .agent_count()
            )
            .unwrap(),
            population_limit + 1,
        );
        // This is a real, canonical next-tick cut with the exact parent. Only
        // the smaller execution budget should prevent its atomic promotion.
        runner
            .validate_run_checkpoint(
                &temporary,
                &checkpoint,
                variant,
                record.seed,
                &record.run_id,
                &plan.digest,
            )
            .unwrap();
        let candidate = checkpoint.encode(&temporary).unwrap();
        fs::write(&temporary, &candidate).unwrap();
        let _lease = ExperimentStatusWriterLease::acquire(&path).unwrap();
        for _ in 0..2 {
            assert!(matches!(
                runner.load_run_checkpoint(
                    variant,
                    record.seed,
                    &record.run_id,
                    &plan.digest,
                    population_limit,
                ),
                Err(ExperimentRunnerError::InvalidCheckpoint { path: rejected, reason })
                    if rejected == temporary && reason.contains("checkpoint population")
            ));
            assert_eq!(fs::read(&path).unwrap(), retained);
            assert_eq!(fs::read(&temporary).unwrap(), candidate);
        }
    }

    #[test]
    fn experiment_population_budget_preserves_exact_resume_and_admits_the_boundary() {
        let uninterrupted_dir = tempfile::tempdir().unwrap().keep();
        let uninterrupted = population_budget_runner(&uninterrupted_dir);
        let expected = uninterrupted
            .execute_single_run(&uninterrupted.variants[0], 7)
            .unwrap();

        let directory = tempfile::tempdir().unwrap().keep();
        let runner = population_budget_runner(&directory);
        let state_file = directory.join("status.json");
        let mut status = runner.load_or_create_status(&state_file).unwrap();
        let plan = runner.validated_plan().unwrap();
        let record = &plan.runs[0];
        let (variant, family) = &plan.variants[&record.variant_id];
        runner
            .prepare_run_checkpoint(variant, *family, record.seed, &record.run_id, &plan.digest)
            .unwrap();
        let path = runner.run_checkpoint_path(&record.run_id);
        let initial_bytes = fs::read(&path).unwrap();
        let mut checkpoint = RunCheckpointV1::decode(&initial_bytes, &path).unwrap();
        let mut owner = restore_run_owner(&checkpoint, *family, &record.run_id, &path).unwrap();
        let population_limit =
            u64::try_from(owner.world().agents().len()).unwrap() + runner.max_ticks;
        let (_, namespace) = host_identifiers(&record.run_id);
        let mut frontend = NullFrontend::new(owner.local_port(), namespace);
        let command = frontend
            .submit_envelope(CommandEnvelope::new(
                CommandId::from_client_sequence(namespace, 1),
                HostCommand::Step,
            ))
            .unwrap();
        let mut now = 0;
        drive_command_to_volatile_commit(
            &record.run_id,
            "population budget oracle step",
            &mut frontend,
            &mut owner,
            command.command_id(),
            1,
            0,
            &mut now,
        )
        .unwrap();
        seal_run_checkpoint(&record.run_id, &mut frontend, &mut owner, &mut now).unwrap();
        checkpoint.owner = owner.capture_volatile_host_checkpoint_v1().unwrap();
        checkpoint.metadata.parent_digest = Some(blake3::hash(&initial_bytes).to_hex().to_string());
        runner
            .publish_run_checkpoint(&path, &checkpoint, &checkpoint.encode(&path).unwrap())
            .unwrap();
        status.runs[0].state = RunState::Running;
        runner.save_status(&state_file, &mut status).unwrap();
        drop(frontend);
        drop(owner);

        let actual = runner
            .execute_batch_with_budget(
                &state_file,
                RunExecutionBudget {
                    cancel: &std::sync::atomic::AtomicBool::new(false),
                    population_limit,
                },
            )
            .unwrap();
        assert_eq!(actual.completed_runs, 1, "{actual:?}");
        assert_eq!(actual.failed_runs, 0);
        assert_eq!(actual.runs[0].final_digest, expected.final_digest);
        assert_eq!(actual.runs[0].total_ticks, expected.total_ticks);
        let final_cut = RunCheckpointV1::decode(&fs::read(&path).unwrap(), &path).unwrap();
        assert_eq!(final_cut.owner.host_checkpoint().tick().0, runner.max_ticks);
        assert_eq!(
            u64::try_from(
                final_cut
                    .owner
                    .host_checkpoint()
                    .world_checkpoint()
                    .agent_count()
            )
            .unwrap(),
            population_limit,
        );
        runner
            .verify_completed_status(&state_file, &actual)
            .unwrap();
    }

    #[test]
    fn scheduled_intervention_preflight_preserves_identity_and_refuses_invalid_work() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("not-created");
        let control = intervention_test_variant("control", Vec::new());
        let treatment = intervention_test_variant(
            "treatment",
            vec![
                ScenarioInterventionV1 {
                    tick: 0,
                    set: serde_json::json!({"food_max": 0.5, "initial_food": 0.5}),
                },
                ScenarioInterventionV1 {
                    tick: 1,
                    set: serde_json::json!({"food_max": 10.0}),
                },
            ],
        );
        let mut runner = MatchedSeedExperimentRunner::new(
            "intervention-preflight",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![7],
            },
            vec![control, treatment.clone()],
            2,
            1,
            &output,
        );
        assert_eq!(runner.plan_batch().unwrap().len(), 2);
        let original_digest = runner.plan_digest().unwrap();
        runner.variants[1].interventions[1].tick = 0;
        assert_ne!(runner.plan_digest().unwrap(), original_digest);
        let same_tick_digest = runner.plan_digest().unwrap();
        runner.variants[1].interventions.swap(0, 1);
        assert_ne!(runner.plan_digest().unwrap(), same_tick_digest);
        runner.variants.push(ScenarioVariant {
            variant_id: "duplicate-treatment".into(),
            ..runner.variants[1].clone()
        });
        assert!(matches!(
            runner.plan_batch(),
            Err(ExperimentRunnerError::InvalidPlan { .. })
        ));

        let at = |tick, set| ScenarioInterventionV1 { tick, set };
        for (interventions, diagnostic) in [
            (
                vec![at(2, serde_json::json!({"food_growth_rate": 0.01}))],
                "must precede",
            ),
            (
                vec![
                    at(1, serde_json::json!({"food_growth_rate": 0.01})),
                    at(0, serde_json::json!({"food_growth_rate": 0.02})),
                ],
                "nondecreasing",
            ),
        ] {
            runner.variants = vec![intervention_test_variant("invalid-timing", interventions)];
            let error = runner
                .plan_batch()
                .expect_err("valid patches have invalid scheduling");
            assert!(error.to_string().contains(diagnostic), "{error}");
            assert!(!output.exists());
        }
        let invalid_schedules = [
            vec![at(0, serde_json::json!({"world_width": 80}))],
            vec![at(0, serde_json::json!({"rng_seed": 9}))],
            vec![at(0, serde_json::json!({"persistence_interval": 0}))],
            vec![at(0, serde_json::json!({"unknown_knob": 1}))],
            vec![at(0, serde_json::json!({"food_max": true}))],
            vec![at(0, serde_json::json!({"food_max": 101}))],
            vec![at(0, serde_json::json!({"population_minimum": 1.5}))],
            vec![at(
                0,
                serde_json::json!({"render.post.bloom.intensity": 0.5}),
            )],
            vec![at(0, serde_json::json!({}))],
            vec![at(0, serde_json::json!([]))],
            vec![at(
                0,
                serde_json::json!({
                    "food_max": "x".repeat(MAX_EXPERIMENT_INTERVENTION_PATCH_BYTES),
                }),
            )],
            vec![
                at(0, serde_json::json!({"food_growth_rate": 0.01}));
                MAX_EXPERIMENT_INTERVENTIONS + 1
            ],
        ];
        for interventions in invalid_schedules {
            runner.variants = vec![intervention_test_variant("invalid", interventions)];
            assert!(matches!(
                runner.plan_batch(),
                Err(ExperimentRunnerError::InvalidPlan { .. })
            ));
            assert!(
                !output.exists(),
                "refused planning must not create execution artifacts"
            );
        }
        runner.variants = vec![treatment];
        runner.max_ticks = 0;
        assert!(runner.plan_batch().is_err());
        assert!(!output.exists());
    }

    fn intervention_test_variant(
        variant_id: &str,
        interventions: Vec<ScenarioInterventionV1>,
    ) -> ScenarioVariant {
        ScenarioVariant {
            variant_id: variant_id.into(),
            brain_family: "mlp".into(),
            config_overrides: BTreeMap::from([
                ("world_width".into(), serde_json::json!(40)),
                ("world_height".into(), serde_json::json!(40)),
                ("food_cell_size".into(), serde_json::json!(10)),
                ("food_max".into(), serde_json::json!(10.0)),
                ("initial_food".into(), serde_json::json!(2.0)),
                ("food_growth_rate".into(), serde_json::json!(0.0)),
                ("food_decay_rate".into(), serde_json::json!(0.0)),
                ("food_intake_rate".into(), serde_json::json!(0.0)),
                ("population_minimum".into(), serde_json::json!(0)),
            ]),
            interventions,
        }
    }

    #[test]
    fn scheduled_interventions_execute_ordered_owner_barriers_and_preserve_real_effects() {
        let directory = tempfile::tempdir().unwrap();
        let patch = |value| ScenarioInterventionV1 {
            tick: 0,
            set: serde_json::json!({"food_max": value, "initial_food": 0.5}),
        };
        let variants = vec![
            intervention_test_variant("clamp-restore", vec![patch(0.5), patch(10.0)]),
            intervention_test_variant("restore-clamp", vec![patch(10.0), patch(0.5)]),
            intervention_test_variant("no-clamp", vec![patch(10.0), patch(10.0)]),
            intervention_test_variant(
                "nested-render",
                vec![
                    ScenarioInterventionV1 {
                        tick: 0,
                        set: serde_json::json!({"render": {"post": {"bloom": {"intensity": 0.5}}}}),
                    },
                    ScenarioInterventionV1 {
                        tick: 0,
                        set: serde_json::json!({"render": {"post": {"bloom": {"intensity": 0.75}}}}),
                    },
                ],
            ),
            intervention_test_variant(
                "maximum-schedule",
                (0..MAX_EXPERIMENT_INTERVENTIONS)
                    .map(|index| patch(if index % 2 == 0 { 0.5 } else { 10.0 }))
                    .collect(),
            ),
        ];
        let runner = MatchedSeedExperimentRunner::new(
            "owner-interventions",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![23],
            },
            variants,
            2,
            2,
            directory.path().join("bundles"),
        );
        let completed = runner
            .execute_batch(&directory.path().join("status.json"))
            .unwrap();
        assert_eq!(completed.completed_runs, 5, "{completed:?}");
        let mut observed = BTreeMap::new();
        for record in &completed.runs {
            let bundle_dir = runner.output_dir.join(&record.run_id);
            let verified = verify_portable_run_bundle(&bundle_dir).unwrap();
            let final_state = verified.final_state.unwrap();
            let checkpoint =
                read_optional_run_checkpoint(&runner.run_checkpoint_path(&record.run_id))
                    .unwrap()
                    .unwrap();
            let declared = &checkpoint.metadata.manifest.scenario.interventions;
            let host = checkpoint.owner.host_checkpoint();
            let expected_count = u64::try_from(declared.len()).unwrap();
            assert_eq!(host.completed_schedule_sequence(), Some(expected_count));
            assert_eq!(host.scheduled_patch_history().len(), declared.len());
            assert_eq!(host.remaining_schedule(), Some([].as_slice()));
            assert_eq!(host.config_revision().get(), expected_count);
            for (index, receipt) in host.scheduled_patch_history().iter().enumerate() {
                assert_eq!(
                    receipt.evidence.scheduled.decode_patch().unwrap(),
                    declared[index].set
                );
                assert_eq!(
                    receipt.evidence.scheduled.sequence,
                    u64::try_from(index + 1).unwrap()
                );
            }
            if record.variant_id == "nested-render" {
                let history = host.scheduled_patch_history();
                assert_eq!(
                    history[0].evidence.outcome,
                    ScheduledPatchOutcome::Applied {
                        changed_paths: vec!["render.post".into()],
                    }
                );
                assert_eq!(
                    history[1].evidence.outcome,
                    ScheduledPatchOutcome::Applied {
                        changed_paths: vec!["render.post.bloom.intensity".into()],
                    }
                );
            }
            observed.insert(
                record.variant_id.clone(),
                (host.config().clone(), final_state),
            );
        }
        // Clamping food then restoring capacity changes the real food state even
        // though both arms finish with identical configs and config revisions.
        assert_eq!(observed["clamp-restore"].0, observed["no-clamp"].0);
        assert!(
            observed["clamp-restore"].1.summary.total_food
                < observed["no-clamp"].1.summary.total_food
        );
        assert_ne!(
            observed["clamp-restore"].1.scientific_digest,
            observed["no-clamp"].1.scientific_digest
        );
        assert_ne!(observed["clamp-restore"].0, observed["restore-clamp"].0);
        assert_ne!(
            observed["clamp-restore"].1.scientific_digest,
            observed["restore-clamp"].1.scientific_digest
        );
    }

    #[test]
    fn scheduled_interventions_resume_before_and_after_exact_due_boundaries() {
        for (family_name, completed_ticks) in [
            "mlp",
            #[cfg(feature = "neuro")]
            "neuroflow",
        ]
        .into_iter()
        .flat_map(|family| [0_u64, 1, 2, 3, 4].map(|tick| (family, tick)))
        {
            let directory = tempfile::tempdir().unwrap();
            let mut variant = intervention_test_variant(
                "drought-recovery",
                vec![
                    ScenarioInterventionV1 {
                        tick: 0,
                        set: serde_json::json!({"food_max": 0.5, "initial_food": 0.5}),
                    },
                    ScenarioInterventionV1 {
                        tick: 2,
                        set: serde_json::json!({"food_max": 10.0}),
                    },
                    ScenarioInterventionV1 {
                        tick: 4,
                        set: serde_json::json!({"food_growth_rate": 0.01}),
                    },
                ],
            );
            variant.brain_family = family_name.into();
            let mut runner = MatchedSeedExperimentRunner::new(
                "scheduled-continuation",
                MatchedSeedCohort {
                    cohort_id: "cohort".into(),
                    seeds: vec![71],
                },
                vec![variant],
                5,
                1,
                directory.path().join("bundles"),
            );
            let plan = runner.validated_plan().unwrap();
            let record = &plan.runs[0];
            let (variant, family) = &plan.variants[&record.variant_id];
            let path = runner.run_checkpoint_path(&record.run_id);
            runner
                .prepare_run_checkpoint(variant, *family, record.seed, &record.run_id, &plan.digest)
                .unwrap();
            let mut checkpoint = read_optional_run_checkpoint(&path).unwrap().unwrap();
            let original_manifest = checkpoint.metadata.manifest.clone();
            let prepared = PreparedInterventions::new(
                &variant.variant_id,
                &config_for_run(&variant.config_overrides, record.seed).unwrap(),
                &variant.interventions,
                runner.max_ticks,
            )
            .unwrap();
            let mut oracle =
                restore_run_owner(&checkpoint, *family, &record.run_id, &path).unwrap();
            let (_, namespace) = host_identifiers(&record.run_id);
            let mut frontend = NullFrontend::new(oracle.local_port(), namespace);
            let mut now = 0;
            for tick in 1..=runner.max_ticks {
                let submitted = frontend
                    .submit_envelope(CommandEnvelope::new(
                        CommandId::from_client_sequence(namespace, tick),
                        HostCommand::Step,
                    ))
                    .unwrap();
                drive_command_to_volatile_commit(
                    &record.run_id,
                    "scheduled oracle step",
                    &mut frontend,
                    &mut oracle,
                    submitted.command_id(),
                    tick,
                    prepared.due_at(tick - 1),
                    &mut now,
                )
                .unwrap();
                seal_run_checkpoint(&record.run_id, &mut frontend, &mut oracle, &mut now).unwrap();
                if tick <= completed_ticks {
                    checkpoint.metadata.parent_digest =
                        Some(blake3::hash(&fs::read(&path).unwrap()).to_hex().to_string());
                    checkpoint.owner = oracle.capture_volatile_host_checkpoint_v1().unwrap();
                    runner
                        .validate_run_checkpoint(
                            &path,
                            &checkpoint,
                            variant,
                            record.seed,
                            &record.run_id,
                            &plan.digest,
                        )
                        .unwrap();
                    write_run_checkpoint(&path, &checkpoint.encode(&path).unwrap()).unwrap();
                }
            }
            let expected_digest = oracle.scientific_digest_v1().unwrap().overall;
            let expected_summary = ExperimentEndpointSummaryV2::from_world(
                oracle.world(),
                record.seed,
                family.canonical_name(),
                expected_digest.clone(),
            )
            .unwrap();
            let expected_journal = oracle
                .capture_volatile_host_checkpoint_v1()
                .unwrap()
                .host_checkpoint()
                .journal_cut();
            drop(frontend);
            drop(oracle);
            let saved = read_optional_run_checkpoint(&path).unwrap().unwrap();
            let saved_host = saved.owner.host_checkpoint();
            assert_eq!(saved_host.tick().0, completed_ticks);
            assert_eq!(
                saved_host.completed_schedule_sequence(),
                Some(u64::try_from(prepared.completed_at(completed_ticks)).unwrap())
            );
            if let Some(pending) = saved_host.remaining_schedule().unwrap().first() {
                assert!(pending.due_tick.0 >= completed_ticks);
            }
            let state_file = directory.path().join("status.json");
            let mut status = runner.load_or_create_status(&state_file).unwrap();
            status.runs[0].state = RunState::Running;
            runner.save_status(&state_file, &mut status).unwrap();
            runner.max_concurrency = 2;
            let resumed = runner.execute_batch(&state_file).unwrap();
            assert_eq!(
                resumed.completed_runs, 1,
                "cut {completed_ticks}: {resumed:?}"
            );
            assert_eq!(
                resumed.runs[0].final_digest.as_deref(),
                Some(expected_digest.as_str())
            );
            let final_cut = read_optional_run_checkpoint(&path).unwrap().unwrap();
            assert_eq!(final_cut.metadata.manifest, original_manifest);
            assert_eq!(
                final_cut.owner.host_checkpoint().journal_cut(),
                expected_journal
            );
            let verified =
                verify_portable_run_bundle(&runner.output_dir.join(&record.run_id)).unwrap();
            assert_eq!(verified.final_state.unwrap().summary, expected_summary);
        }
    }

    #[test]
    fn checksummed_intervention_forgeries_fail_pending_and_completed_schedule_binding() {
        let directory = tempfile::tempdir().unwrap();
        let variant = intervention_test_variant(
            "ordered-treatment",
            vec![
                ScenarioInterventionV1 {
                    tick: 0,
                    set: serde_json::json!({"food_max": 0.5, "initial_food": 0.5}),
                },
                ScenarioInterventionV1 {
                    tick: 0,
                    set: serde_json::json!({"food_max": 10.0}),
                },
            ],
        );
        let runner = MatchedSeedExperimentRunner::new(
            "schedule-forgery",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![23],
            },
            vec![variant.clone()],
            2,
            1,
            directory.path().join("bundles"),
        );
        let plan = runner.validated_plan().unwrap();
        let record = &plan.runs[0];
        let family = ResolvedBrainFamily::resolve(&variant).unwrap();
        let path = runner.run_checkpoint_path(&record.run_id);
        runner
            .prepare_run_checkpoint(&variant, family, record.seed, &record.run_id, &plan.digest)
            .unwrap();
        let initial = read_optional_run_checkpoint(&path).unwrap().unwrap();
        let mut declared_other = variant.clone();
        declared_other.interventions[0].set =
            serde_json::json!({"food_max": 0.75, "initial_food": 0.5});
        let altered_runner = MatchedSeedExperimentRunner::new(
            runner.experiment_id.clone(),
            runner.cohort.clone(),
            vec![declared_other.clone()],
            runner.max_ticks,
            runner.max_concurrency,
            &runner.output_dir,
        );
        let altered_plan = altered_runner.validated_plan().unwrap();
        let mut changed = RunCheckpointV1 {
            metadata: initial.metadata.clone(),
            owner: initial.owner.clone(),
        };
        changed
            .metadata
            .plan_digest
            .clone_from(&altered_plan.digest);
        changed
            .metadata
            .manifest
            .scenario
            .interventions
            .clone_from(&declared_other.interventions);
        let framed = RunCheckpointV1::decode(&changed.encode(&path).unwrap(), &path).unwrap();
        assert!(
            altered_runner
                .validate_run_checkpoint(
                    &path,
                    &framed,
                    &declared_other,
                    record.seed,
                    &record.run_id,
                    &altered_plan.digest,
                )
                .is_err(),
            "tick-zero future inputs must be bound before any patch executes"
        );

        let status = runner
            .execute_batch(&directory.path().join("status.json"))
            .unwrap();
        assert_eq!(status.completed_runs, 1, "{status:?}");
        let bundle_dir = runner.output_dir.join(&record.run_id);
        let checkpoint_path = bundle_dir.join(FINAL_CHECKPOINT_ARTIFACT.0);
        let final_checkpoint = read_optional_run_checkpoint(&checkpoint_path)
            .unwrap()
            .unwrap();
        changed.owner = final_checkpoint.owner.clone();
        changed.metadata = final_checkpoint.metadata.clone();
        changed
            .metadata
            .manifest
            .scenario
            .interventions
            .clone_from(&declared_other.interventions);
        let forged_manifest = changed.metadata.manifest.clone();
        fs::write(&checkpoint_path, changed.encode(&checkpoint_path).unwrap()).unwrap();
        let evidence_path = bundle_dir.join("evidence/run.json");
        let mut evidence: RunEvidenceV2 =
            serde_json::from_slice(&fs::read(&evidence_path).unwrap()).unwrap();
        evidence.manifest_digest = blake3::hash(&forged_manifest.canonical_json_bytes().unwrap())
            .to_hex()
            .to_string();
        fs::write(&evidence_path, serde_json::to_vec(&evidence).unwrap()).unwrap();
        let manifest_path = bundle_dir.join("bundle_manifest.json");
        let mut bundle: RunBundleV1 =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        bundle.manifest = forged_manifest.to_storage_record().unwrap();
        for entry in &mut bundle.artifacts {
            let bytes = fs::read(bundle_dir.join(&entry.relative_path)).unwrap();
            entry.bytes_len = u64::try_from(bytes.len()).unwrap();
            entry.blake3_hex = blake3::hash(&bytes).to_hex().to_string();
        }
        fs::write(&manifest_path, serde_json::to_vec(&bundle).unwrap()).unwrap();
        scriptbots_storage::verify_run_bundle(&bundle_dir)
            .expect("coherent metadata and fresh artifact checksums are byte-valid");
        let error = verify_portable_run_bundle(&bundle_dir)
            .expect_err("same final config does not justify forged completed treatment inputs");
        assert!(
            error
                .to_string()
                .contains("ordered applied and committed owner receipt"),
            "{error}"
        );
    }

    #[test]
    fn malformed_and_interrupted_status_fail_closed() {
        let temp_dir = tempfile::tempdir().expect("temporary status directory");
        let state_file = temp_dir.path().join("status.json");
        let runner = MatchedSeedExperimentRunner::new(
            "exp-status",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![7],
            },
            vec![ScenarioVariant {
                variant_id: "mlp".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            }],
            1,
            1,
            temp_dir.path().join("bundles"),
        );

        fs::write(&state_file, b"{truncated")
            .expect("test can write a deliberately malformed status");
        assert!(matches!(
            runner.load_or_create_status(&state_file),
            Err(ExperimentRunnerError::InvalidStatus { .. })
        ));
        assert_eq!(
            fs::read(&state_file).expect("malformed evidence remains available"),
            b"{truncated"
        );

        let interrupted_state = temp_dir.path().join("interrupted.json");
        let temporary = status_temporary_path(&interrupted_state).expect("temporary path");
        fs::write(&temporary, b"partial").expect("test can model an interrupted atomic write");
        assert!(matches!(
            runner.load_or_create_status(&interrupted_state),
            Err(ExperimentRunnerError::InterruptedStatusWrite { .. })
        ));
    }

    fn interrupted_status_runner(directory: &Path) -> MatchedSeedExperimentRunner {
        MatchedSeedExperimentRunner::new(
            "exp-status-recovery",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![7],
            },
            vec![ScenarioVariant {
                variant_id: "mlp".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            }],
            3,
            1,
            directory.join("bundles"),
        )
    }

    #[test]
    fn interrupted_status_publication_recovers_initial_and_running_generations() {
        let directory = tempfile::tempdir().unwrap();
        let runner = interrupted_status_runner(directory.path());
        let state_file = directory.path().join("status.json");
        let temporary = status_temporary_path(&state_file).unwrap();
        let plan = runner.validated_plan().unwrap();
        let initial = runner.initial_status(&plan);
        let initial_bytes = canonical_status_bytes(&initial).unwrap();
        fs::write(&temporary, &initial_bytes).unwrap();

        assert_eq!(runner.load_or_create_status(&state_file).unwrap(), initial);
        assert_eq!(fs::read(&state_file).unwrap(), initial_bytes);
        assert!(!temporary.exists());

        let record = &initial.runs[0];
        let (variant, family) = &plan.variants[&record.variant_id];
        runner
            .prepare_run_checkpoint(variant, *family, record.seed, &record.run_id, &plan.digest)
            .unwrap();
        let checkpoint_path = runner.run_checkpoint_path(&record.run_id);
        let initial_cut = fs::read(&checkpoint_path).unwrap();
        let mut running = initial;
        running.generation += 1;
        running.runs[0].state = RunState::Running;
        let running_bytes = canonical_status_bytes(&running).unwrap();
        fs::write(&temporary, &running_bytes).unwrap();

        assert_eq!(runner.load_or_create_status(&state_file).unwrap(), running);
        assert_eq!(fs::read(&state_file).unwrap(), running_bytes);
        assert_eq!(fs::read(&checkpoint_path).unwrap(), initial_cut);
        assert!(!temporary.exists());

        let completed = runner.execute_batch(&state_file).unwrap();
        assert_eq!(completed.completed_runs, 1, "{completed:?}");
        assert_eq!(completed.failed_runs, 0);
        assert_eq!(completed.runs[0].total_ticks, runner.max_ticks);
        let final_cut = read_optional_run_checkpoint(&checkpoint_path)
            .unwrap()
            .unwrap();
        assert_eq!(final_cut.owner.host_checkpoint().tick().0, runner.max_ticks);
        assert_ne!(fs::read(&checkpoint_path).unwrap(), initial_cut);
    }

    #[test]
    fn interrupted_status_publication_verifies_completion_before_promoting_it() {
        let directory = tempfile::tempdir().unwrap();
        let runner = interrupted_status_runner(directory.path());
        let state_file = directory.path().join("status.json");
        let temporary = status_temporary_path(&state_file).unwrap();
        let mut running = runner.load_or_create_status(&state_file).unwrap();
        running.runs[0].state = RunState::Running;
        runner.save_status(&state_file, &mut running).unwrap();
        let record = runner
            .execute_single_run(&runner.variants[0], runner.cohort.seeds[0])
            .unwrap();
        let checkpoint_path = runner.run_checkpoint_path(&record.run_id);
        let sealed_cut = fs::read(&checkpoint_path).unwrap();
        let bundle_dir = runner.output_dir.join(&record.run_id);
        let summary_path = bundle_dir.join("exports/summary.csv");
        let summary = fs::read(&summary_path).unwrap();
        let mut completed = running;
        completed.generation += 1;
        completed.runs[0] = record;
        recompute_status_counts(&mut completed);
        let completed_bytes = canonical_status_bytes(&completed).unwrap();
        fs::write(&temporary, &completed_bytes).unwrap();
        let committed_bytes = fs::read(&state_file).unwrap();

        fs::write(&summary_path, b"forged summary\n").unwrap();
        assert!(matches!(
            runner.load_or_create_status(&state_file),
            Err(ExperimentRunnerError::Bundle { .. })
        ));
        assert_eq!(fs::read(&state_file).unwrap(), committed_bytes);
        assert_eq!(fs::read(&temporary).unwrap(), completed_bytes);
        assert_eq!(fs::read(&checkpoint_path).unwrap(), sealed_cut);

        fs::write(&summary_path, &summary).unwrap();
        // An already finished run is recovered without another owner execution
        // or status generation. Its original continuation artifact stays exact.
        assert_eq!(runner.execute_batch(&state_file).unwrap(), completed);
        assert_eq!(fs::read(&state_file).unwrap(), completed_bytes);
        assert_eq!(fs::read(&checkpoint_path).unwrap(), sealed_cut);
        assert_eq!(fs::read(&summary_path).unwrap(), summary);
        assert!(!temporary.exists());
    }

    #[test]
    fn interrupted_status_publication_refuses_conflicts_and_retains_both_files() {
        for case in [
            "stale",
            "skipped",
            "changed-plan",
            "changed-run",
            "regressed",
            "forged-success",
            "partial",
            "noncanonical",
            "generation-overflow",
            "missing-initial",
            "nonzero-initial",
            "corrupt-committed",
        ] {
            let directory = tempfile::tempdir().unwrap();
            let runner = interrupted_status_runner(directory.path());
            let state_file = directory.path().join("status.json");
            let temporary = status_temporary_path(&state_file).unwrap();
            let plan = runner.validated_plan().unwrap();
            let mut committed = runner.initial_status(&plan);
            committed.generation = 1;
            committed.runs[0].state = RunState::Running;
            let mut candidate = committed.clone();
            candidate.generation = 2;
            let has_committed = !matches!(case, "missing-initial" | "nonzero-initial");
            match case {
                "stale" => candidate.generation = committed.generation,
                "skipped" => candidate.generation += 1,
                "changed-plan" => candidate.plan_digest = "0".repeat(64),
                "changed-run" => candidate.runs[0].seed += 1,
                "regressed" => candidate.runs[0].state = RunState::Pending,
                "forged-success" => {
                    let record = &mut candidate.runs[0];
                    record.state = RunState::Completed;
                    record.total_ticks = runner.max_ticks;
                    record.final_digest = Some("0".repeat(16));
                    record.bundle_path = Some(
                        runner
                            .output_dir
                            .join(&record.run_id)
                            .to_string_lossy()
                            .into_owned(),
                    );
                    recompute_status_counts(&mut candidate);
                }
                "generation-overflow" => {
                    committed.generation = u64::MAX;
                    candidate.generation = 0;
                }
                "missing-initial" => candidate.generation = 0,
                "nonzero-initial" => {
                    candidate = runner.initial_status(&plan);
                    candidate.generation = 1;
                }
                _ => {}
            }
            let mut committed_bytes = canonical_status_bytes(&committed).unwrap();
            if case == "corrupt-committed" {
                committed_bytes.truncate(committed_bytes.len() / 2);
            }
            if has_committed {
                fs::write(&state_file, &committed_bytes).unwrap();
            }
            let mut candidate_bytes = canonical_status_bytes(&candidate).unwrap();
            if case == "partial" {
                candidate_bytes.truncate(candidate_bytes.len() / 2);
            } else if case == "noncanonical" {
                candidate_bytes.pop();
            }
            fs::write(&temporary, &candidate_bytes).unwrap();

            assert!(runner.load_or_create_status(&state_file).is_err(), "{case}");
            if has_committed {
                assert_eq!(fs::read(&state_file).unwrap(), committed_bytes, "{case}");
            } else {
                assert!(!state_file.exists(), "{case}");
            }
            assert_eq!(fs::read(&temporary).unwrap(), candidate_bytes, "{case}");
            assert!(!runner.output_dir.exists(), "{case}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn interrupted_status_publication_refuses_linked_files() {
        use std::os::unix::fs::symlink;

        for (link_temporary, hard_link) in [(true, false), (false, false), (true, true)] {
            let directory = tempfile::tempdir().unwrap();
            let runner = interrupted_status_runner(directory.path());
            let state_file = directory.path().join("status.json");
            let temporary = status_temporary_path(&state_file).unwrap();
            let target = directory.path().join("linked-status.json");
            let plan = runner.validated_plan().unwrap();
            let committed = runner.initial_status(&plan);
            let mut candidate = committed.clone();
            candidate.generation = 1;
            candidate.runs[0].state = RunState::Running;
            let committed_bytes = canonical_status_bytes(&committed).unwrap();
            let candidate_bytes = canonical_status_bytes(&candidate).unwrap();
            let (link_path, target_bytes) = if link_temporary {
                fs::write(&state_file, &committed_bytes).unwrap();
                (&temporary, &candidate_bytes)
            } else {
                fs::write(&temporary, &candidate_bytes).unwrap();
                (&state_file, &committed_bytes)
            };
            fs::write(&target, target_bytes).unwrap();
            if hard_link {
                fs::hard_link(&target, link_path).unwrap();
            } else {
                symlink(&target, link_path).unwrap();
            }

            assert!(runner.load_or_create_status(&state_file).is_err());
            assert_eq!(fs::read(&state_file).unwrap(), committed_bytes);
            assert_eq!(fs::read(&temporary).unwrap(), candidate_bytes);
            assert_eq!(fs::read(&target).unwrap(), *target_bytes);
        }
    }

    #[test]
    fn staged_bundle_publication_recovers_a_complete_bundle_without_replaying_ticks() {
        let directory = tempfile::tempdir().unwrap();
        let runner = interrupted_status_runner(directory.path());
        let state_file = directory.path().join("status.json");
        let mut running = runner.load_or_create_status(&state_file).unwrap();
        running.runs[0].state = RunState::Running;
        runner.save_status(&state_file, &mut running).unwrap();
        let completed = runner
            .execute_single_run(&runner.variants[0], runner.cohort.seeds[0])
            .unwrap();
        let checkpoint_path = runner.run_checkpoint_path(&completed.run_id);
        let original_cut = fs::read(&checkpoint_path).unwrap();
        let bundle_dir = runner.output_dir.join(&completed.run_id);
        let original_bundle = fs::read(bundle_dir.join("bundle_manifest.json")).unwrap();
        let staged = runner.staged_bundle_path(&completed.run_id);
        // A complete real bundle models the crash after staging verification,
        // before the single directory rename and final status publication.
        fs::rename(&bundle_dir, &staged).unwrap();

        let recovered = runner.execute_batch(&state_file).unwrap();
        assert_eq!(recovered.completed_runs, 1, "{recovered:?}");
        assert_eq!(recovered.failed_runs, 0);
        assert_eq!(recovered.runs[0], completed);
        assert_eq!(fs::read(&checkpoint_path).unwrap(), original_cut);
        assert_eq!(
            fs::read(bundle_dir.join("bundle_manifest.json")).unwrap(),
            original_bundle
        );
        assert!(!staged.exists());
        assert!(
            !runner
                .output_dir
                .join(format!("{}.bundle.interrupted", completed.run_id))
                .exists()
        );
    }

    #[test]
    fn staged_bundle_publication_retains_each_partial_write_and_uses_the_final_cut() {
        let reference_dir = tempfile::tempdir().unwrap();
        let reference = interrupted_status_runner(reference_dir.path());
        let expected = reference
            .execute_single_run(&reference.variants[0], reference.cohort.seeds[0])
            .unwrap();
        let original_cut = fs::read(reference.run_checkpoint_path(&expected.run_id)).unwrap();
        let bundle_dir = reference.output_dir.join(&expected.run_id);
        let artifacts = REQUIRED_EXPERIMENT_ARTIFACTS
            .iter()
            .map(|(path, _)| *path)
            .chain(std::iter::once("bundle_manifest.json"))
            .map(|path| (path, fs::read(bundle_dir.join(path)).unwrap()))
            .collect::<Vec<_>>();

        // Empty staging, then a power loss during each actual file's write.
        for partial_file in std::iter::once(None).chain((0..artifacts.len()).map(Some)) {
            let directory = tempfile::tempdir().unwrap();
            let runner = interrupted_status_runner(directory.path());
            let state_file = directory.path().join("status.json");
            let mut running = runner.load_or_create_status(&state_file).unwrap();
            running.runs[0].state = RunState::Running;
            runner.save_status(&state_file, &mut running).unwrap();
            fs::create_dir_all(&runner.output_dir).unwrap();
            let checkpoint_path = runner.run_checkpoint_path(&expected.run_id);
            fs::write(&checkpoint_path, &original_cut).unwrap();
            let staged = runner.staged_bundle_path(&expected.run_id);
            fs::create_dir(&staged).unwrap();
            let mut interrupted_files = Vec::new();
            if let Some(last) = partial_file {
                for (index, (path, bytes)) in artifacts.iter().enumerate().take(last + 1) {
                    let bytes = if index == last {
                        &bytes[..bytes.len() / 2]
                    } else {
                        bytes.as_slice()
                    };
                    let destination = staged.join(path);
                    fs::create_dir_all(destination.parent().unwrap()).unwrap();
                    fs::write(&destination, bytes).unwrap();
                    interrupted_files.push((*path, bytes.to_vec()));
                }
            }

            let recovered = runner.execute_batch(&state_file).unwrap();
            assert_eq!(
                recovered.completed_runs, 1,
                "{partial_file:?}: {recovered:?}"
            );
            assert_eq!(recovered.failed_runs, 0);
            assert_eq!(recovered.runs[0].final_digest, expected.final_digest);
            assert_eq!(recovered.runs[0].total_ticks, expected.total_ticks);
            assert_eq!(fs::read(&checkpoint_path).unwrap(), original_cut);
            assert!(!staged.exists());
            let retained = runner
                .output_dir
                .join(format!("{}.bundle.interrupted", expected.run_id));
            assert!(retained.is_dir());
            for (path, bytes) in &interrupted_files {
                assert_eq!(fs::read(retained.join(path)).unwrap(), *bytes);
            }
            for (path, _) in artifacts.iter().skip(interrupted_files.len()) {
                assert!(!retained.join(path).exists());
            }
            for (path, bytes) in &artifacts[..REQUIRED_EXPERIMENT_ARTIFACTS.len()] {
                assert_eq!(
                    fs::read(runner.output_dir.join(&expected.run_id).join(path)).unwrap(),
                    *bytes
                );
            }
        }
    }

    #[test]
    fn staged_bundle_publication_refuses_unsafe_recovery_and_preserves_evidence() {
        let reference_dir = tempfile::tempdir().unwrap();
        let reference = interrupted_status_runner(reference_dir.path());
        let expected = reference
            .execute_single_run(&reference.variants[0], reference.cohort.seeds[0])
            .unwrap();
        let final_cut = fs::read(reference.run_checkpoint_path(&expected.run_id)).unwrap();
        for case in [
            "prior-incident",
            "nonfinal-cut",
            "corrupt-cut",
            "oversized",
            "unrelated-file",
        ] {
            let directory = tempfile::tempdir().unwrap();
            let runner = interrupted_status_runner(directory.path());
            let state_file = directory.path().join("status.json");
            let mut running = runner.load_or_create_status(&state_file).unwrap();
            running.runs[0].state = RunState::Running;
            runner.save_status(&state_file, &mut running).unwrap();
            let plan = runner.validated_plan().unwrap();
            let record = &plan.runs[0];
            let (variant, family) = &plan.variants[&record.variant_id];
            runner
                .prepare_run_checkpoint(variant, *family, record.seed, &record.run_id, &plan.digest)
                .unwrap();
            let checkpoint_path = runner.run_checkpoint_path(&record.run_id);
            if case != "nonfinal-cut" {
                let mut bytes = final_cut.clone();
                if case == "corrupt-cut" {
                    bytes[0] ^= 1;
                }
                fs::write(&checkpoint_path, bytes).unwrap();
            }
            let original_cut = fs::read(&checkpoint_path).unwrap();
            let staged = runner.staged_bundle_path(&record.run_id);
            fs::create_dir(&staged).unwrap();
            let staged_file = if case == "unrelated-file" {
                staged.join("unrelated.txt")
            } else {
                let exports = staged.join("exports");
                fs::create_dir(&exports).unwrap();
                exports.join("summary.csv")
            };
            fs::write(&staged_file, b"partial").unwrap();
            if case == "oversized" {
                OpenOptions::new()
                    .write(true)
                    .open(&staged_file)
                    .unwrap()
                    .set_len(MAX_BUNDLE_ARTIFACT_BYTES + 1)
                    .unwrap();
            }
            let retained = runner
                .output_dir
                .join(format!("{}.bundle.interrupted", record.run_id));
            if case == "prior-incident" {
                fs::create_dir(&retained).unwrap();
                fs::write(retained.join("original-incident"), b"keep this evidence").unwrap();
            }

            let result = runner.execute_batch(&state_file);
            if case == "corrupt-cut" {
                assert!(matches!(
                    result,
                    Err(ExperimentRunnerError::InvalidCheckpoint { .. })
                ));
            } else {
                let status = result.unwrap();
                assert_eq!(status.completed_runs, 0, "{case}: {status:?}");
                assert_eq!(status.failed_runs, 1, "{case}: {status:?}");
                assert!(status.runs[0].error_reason.is_some());
            }
            assert_eq!(fs::read(&checkpoint_path).unwrap(), original_cut, "{case}");
            assert!(!runner.output_dir.join(&record.run_id).exists(), "{case}");
            assert!(staged.is_dir(), "{case}");
            if case == "oversized" {
                assert_eq!(
                    fs::metadata(&staged_file).unwrap().len(),
                    MAX_BUNDLE_ARTIFACT_BYTES + 1
                );
            } else {
                assert_eq!(fs::read(&staged_file).unwrap(), b"partial", "{case}");
            }
            if case == "prior-incident" {
                assert_eq!(
                    fs::read(retained.join("original-incident")).unwrap(),
                    b"keep this evidence"
                );
            } else {
                assert!(!retained.exists(), "{case}");
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn staged_bundle_publication_refuses_symlink_and_hard_link_aliases() {
        use std::os::unix::fs::symlink;

        let reference_dir = tempfile::tempdir().unwrap();
        let reference = interrupted_status_runner(reference_dir.path());
        let expected = reference
            .execute_single_run(&reference.variants[0], reference.cohort.seeds[0])
            .unwrap();
        let final_cut = fs::read(reference.run_checkpoint_path(&expected.run_id)).unwrap();
        for case in [
            "root-symlink",
            "directory-symlink",
            "file-symlink",
            "file-hard-link",
        ] {
            let directory = tempfile::tempdir().unwrap();
            let runner = interrupted_status_runner(directory.path());
            let state_file = directory.path().join("status.json");
            let mut running = runner.load_or_create_status(&state_file).unwrap();
            running.runs[0].state = RunState::Running;
            runner.save_status(&state_file, &mut running).unwrap();
            fs::create_dir_all(&runner.output_dir).unwrap();
            let checkpoint_path = runner.run_checkpoint_path(&expected.run_id);
            fs::write(&checkpoint_path, &final_cut).unwrap();
            let staged = runner.staged_bundle_path(&expected.run_id);
            let external = directory.path().join("external");
            fs::create_dir(&external).unwrap();
            fs::write(external.join("summary.csv"), b"external evidence").unwrap();
            if case == "root-symlink" {
                symlink(&external, &staged).unwrap();
            } else {
                fs::create_dir(&staged).unwrap();
                let exports = staged.join("exports");
                if case == "directory-symlink" {
                    symlink(&external, &exports).unwrap();
                } else {
                    fs::create_dir(&exports).unwrap();
                    if case == "file-symlink" {
                        symlink(external.join("summary.csv"), exports.join("summary.csv")).unwrap();
                    } else {
                        fs::hard_link(external.join("summary.csv"), exports.join("summary.csv"))
                            .unwrap();
                    }
                }
            }

            let status = runner.execute_batch(&state_file).unwrap();
            assert_eq!(status.completed_runs, 0, "{case}: {status:?}");
            assert_eq!(status.failed_runs, 1, "{case}: {status:?}");
            assert_eq!(fs::read(&checkpoint_path).unwrap(), final_cut);
            assert_eq!(
                fs::read(external.join("summary.csv")).unwrap(),
                b"external evidence"
            );
            assert!(fs::symlink_metadata(&staged).is_ok());
            assert!(!runner.output_dir.join(&expected.run_id).exists());
            assert!(
                !runner
                    .output_dir
                    .join(format!("{}.bundle.interrupted", expected.run_id))
                    .exists()
            );
        }
    }

    #[test]
    fn running_status_refuses_tick_zero_replay_without_composite_checkpoint() {
        let temp_dir = tempfile::tempdir().expect("temporary status directory");
        let state_file = temp_dir.path().join("status.json");
        let runner = MatchedSeedExperimentRunner::new(
            "exp-running",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![7],
            },
            vec![ScenarioVariant {
                variant_id: "mlp".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            }],
            1,
            1,
            temp_dir.path().join("bundles"),
        );
        let mut status = runner
            .load_or_create_status(&state_file)
            .expect("fresh status");
        status.runs[0].state = RunState::Running;
        runner
            .save_status(&state_file, &mut status)
            .expect("running boundary is durable");

        assert!(matches!(
            runner.execute_batch(&state_file),
            Err(ExperimentRunnerError::ResumeCheckpointUnavailable {
                completed_ticks: 0,
                ..
            })
        ));
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn neuroflow_checkpoint_restore_preserves_the_original_constructor_recipe() {
        let directory = tempfile::tempdir().unwrap();
        let mut runner = progress_runner(directory.path(), vec![71]);
        runner.variants[0].variant_id = "neuro".into();
        runner.variants[0].brain_family = "neuroflow".into();
        let plan = runner.validated_plan().unwrap();
        let record = &plan.runs[0];
        let (variant, family) = &plan.variants[&record.variant_id];
        let path = runner.run_checkpoint_path(&record.run_id);
        let mut checkpoint = runner
            .initial_run_checkpoint(
                variant,
                *family,
                record.seed,
                &record.run_id,
                &plan.digest,
            )
            .unwrap();
        let original_manifest = checkpoint.metadata.manifest.canonical_json_bytes().unwrap();
        let mut oracle = restore_run_owner(&checkpoint, *family, &record.run_id, &path).unwrap();
        let original_registry = checkpoint
            .owner
            .host_checkpoint()
            .world_checkpoint()
            .required_brain_registry();
        assert!(oracle.world().agent_count() > 0);
        let (_, namespace) = host_identifiers(&record.run_id);
        let mut frontend = NullFrontend::new(oracle.local_port(), namespace);
        let mut now = 0;

        // A live owner accepts complete settings edits without replacing its
        // already installed family. This tests restore identity directly; it
        // does not add new architecture knobs to the experiment proposal API.
        let mut edited = oracle.world().config().clone();
        edited.neuroflow.hidden_layers = vec![11, 5];
        edited.neuroflow.activation = scriptbots_core::NeuroflowActivationKind::Relu;
        edited.validate().unwrap();
        let submitted = frontend
            .submit_envelope(CommandEnvelope::new(
                CommandId::from_client_sequence(namespace, 1),
                HostCommand::UpdateConfig(Box::new(edited.clone())),
            ))
            .unwrap();
        drive_command_to_volatile_commit(
            &record.run_id,
            "update raw NeuroFlow settings",
            &mut frontend,
            &mut oracle,
            submitted.command_id(),
            0,
            0,
            &mut now,
        )
        .unwrap();
        seal_run_checkpoint(&record.run_id, &mut frontend, &mut oracle, &mut now).unwrap();
        checkpoint.owner = oracle.capture_volatile_host_checkpoint_v1().unwrap();
        assert_eq!(checkpoint.owner.host_checkpoint().config(), &edited);
        assert_eq!(
            checkpoint.metadata.manifest.canonical_json_bytes().unwrap(),
            original_manifest
        );
        assert_eq!(
            checkpoint
                .owner
                .host_checkpoint()
                .world_checkpoint()
                .required_brain_registry(),
            original_registry
        );
        let encoded = checkpoint.encode(&path).unwrap();
        let mut decoded = RunCheckpointV1::decode(&encoded, &path).unwrap();
        let mut restored = restore_run_owner(&decoded, *family, &record.run_id, &path)
            .expect("the launch manifest retains the installed NeuroFlow constructor");
        assert_eq!(
            restored.scientific_digest_v1().unwrap(),
            oracle.scientific_digest_v1().unwrap()
        );
        assert_eq!(restored.world().config(), &edited);

        // Substituting the current raw settings supplies a different valid
        // adapter, whose identity must be rejected before any continued science.
        let constructor = decoded.metadata.manifest.normalized_config.clone();
        decoded.metadata.manifest.normalized_config = serde_json::to_value(&edited).unwrap();
        let error = restore_run_owner(&decoded, *family, &record.run_id, &path)
            .err()
            .expect("current settings cannot replace the original constructor recipe");
        assert!(matches!(
            error,
            ExperimentRunnerError::InvalidCheckpoint { .. }
        ));
        assert!(error.to_string().contains("prepared brain registry"));
        decoded.metadata.manifest.normalized_config = constructor;
        assert_eq!(
            decoded.metadata.manifest.canonical_json_bytes().unwrap(),
            original_manifest
        );

        let mut resumed_frontend = NullFrontend::new(restored.local_port(), namespace);
        let mut resumed_now = 0;
        for tick in 1..=2 {
            let command = CommandEnvelope::new(
                CommandId::from_client_sequence(namespace, tick + 1),
                HostCommand::Step,
            );
            for (owner, client, clock) in [
                (&mut oracle, &mut frontend, &mut now),
                (&mut restored, &mut resumed_frontend, &mut resumed_now),
            ] {
                let submitted = client.submit_envelope(command.clone()).unwrap();
                drive_command_to_volatile_commit(
                    &record.run_id,
                    "continue configured NeuroFlow owner",
                    client,
                    owner,
                    submitted.command_id(),
                    tick,
                    0,
                    clock,
                )
                .unwrap();
                seal_run_checkpoint(&record.run_id, client, owner, clock).unwrap();
            }
            assert_eq!(
                restored.scientific_digest_v1().unwrap(),
                oracle.scientific_digest_v1().unwrap()
            );
        }
    }

    #[test]
    fn running_checkpoint_continues_real_families_and_retains_original_owner_cut() {
        for family_name in [
            "mlp",
            "dwraon",
            #[cfg(feature = "brain-ft")]
            "ft",
            #[cfg(feature = "neuro")]
            "neuroflow",
        ] {
            for completed_ticks in [0_u64, 2, 4] {
                let directory = tempfile::tempdir().expect("continuation directory");
                let runner = MatchedSeedExperimentRunner::new(
                    "continue-owner",
                    MatchedSeedCohort {
                        cohort_id: "cohort".into(),
                        seeds: vec![71],
                    },
                    vec![ScenarioVariant {
                        variant_id: family_name.into(),
                        brain_family: family_name.into(),
                        config_overrides: BTreeMap::new(),
                        interventions: Vec::new(),
                    }],
                    5,
                    1,
                    directory.path().join("bundles"),
                );
                let plan = runner.validated_plan().unwrap();
                let record = &plan.runs[0];
                let (variant, family) = plan.variants.get(&record.variant_id).unwrap();
                let path = runner.run_checkpoint_path(&record.run_id);
                runner
                    .prepare_run_checkpoint(
                        variant,
                        *family,
                        record.seed,
                        &record.run_id,
                        &plan.digest,
                    )
                    .unwrap();
                let mut checkpoint = read_optional_run_checkpoint(&path).unwrap().unwrap();
                let original_manifest =
                    checkpoint.metadata.manifest.canonical_json_bytes().unwrap();
                let mut oracle =
                    restore_run_owner(&checkpoint, *family, &record.run_id, &path).unwrap();
                let (_, namespace) = host_identifiers(&record.run_id);
                let mut frontend = NullFrontend::new(oracle.local_port(), namespace);
                let mut now = 0;
                for tick in 1..=5 {
                    let submitted = frontend
                        .submit_envelope(CommandEnvelope::new(
                            CommandId::from_client_sequence(namespace, tick),
                            HostCommand::Step,
                        ))
                        .unwrap();
                    drive_command_to_volatile_commit(
                        &record.run_id,
                        "oracle step",
                        &mut frontend,
                        &mut oracle,
                        submitted.command_id(),
                        tick,
                        0,
                        &mut now,
                    )
                    .unwrap();
                    seal_run_checkpoint(&record.run_id, &mut frontend, &mut oracle, &mut now)
                        .unwrap();
                    if tick == completed_ticks {
                        // A genuine same-tick owner command distinguishes this cut from a
                        // deterministic tick-zero rerun, without changing the science.
                        seal_run_checkpoint(&record.run_id, &mut frontend, &mut oracle, &mut now)
                            .unwrap();
                        checkpoint.metadata.parent_digest =
                            Some(blake3::hash(&fs::read(&path).unwrap()).to_hex().to_string());
                        checkpoint.owner = oracle.capture_volatile_host_checkpoint_v1().unwrap();
                        write_run_checkpoint(&path, &checkpoint.encode(&path).unwrap()).unwrap();
                    }
                }
                let expected_digest = oracle.scientific_digest_v1().unwrap().overall;
                let expected_journal = oracle
                    .capture_volatile_host_checkpoint_v1()
                    .unwrap()
                    .host_checkpoint()
                    .journal_cut();
                drop(frontend);
                drop(oracle);
                let state_file = directory.path().join("status.json");
                let mut status = runner.load_or_create_status(&state_file).unwrap();
                status.runs[0].state = RunState::Running;
                runner.save_status(&state_file, &mut status).unwrap();
                let resumed = runner.execute_batch(&state_file).unwrap();
                assert_eq!(
                    resumed.completed_runs, 1,
                    "{family_name} cut {completed_ticks}: {resumed:?}"
                );
                assert_eq!(
                    resumed.runs[0].final_digest.as_deref(),
                    Some(expected_digest.as_str())
                );
                let final_cut = read_optional_run_checkpoint(&path).unwrap().unwrap();
                assert_eq!(
                    final_cut.owner.host_checkpoint().journal_cut(),
                    expected_journal
                );
                assert_eq!(
                    final_cut.metadata.manifest.canonical_json_bytes().unwrap(),
                    original_manifest
                );
            }
        }
    }

    #[test]
    fn running_checkpoint_survives_process_kill_and_matches_uninterrupted_process() {
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};
        const CHILD_DIRECTORY: &str = "SCRIPTBOTS_RUNNER_RESTART_TEST_DIRECTORY";
        const CHILD_FAMILY: &str = "SCRIPTBOTS_RUNNER_RESTART_TEST_FAMILY";
        const CHILD_THREADS: &str = "SCRIPTBOTS_RUNNER_RESTART_TEST_THREADS";
        const TICKS: u64 = 40;
        struct RunnerChild(std::process::Child);
        impl Drop for RunnerChild {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        let make_runner = |directory: &Path, family: &str| {
            MatchedSeedExperimentRunner::new(
                "process-continuation",
                MatchedSeedCohort {
                    cohort_id: "cohort".into(),
                    seeds: vec![92],
                },
                vec![ScenarioVariant {
                    variant_id: family.into(),
                    brain_family: family.into(),
                    config_overrides: BTreeMap::new(),
                    interventions: Vec::new(),
                }],
                TICKS,
                1,
                directory.join("bundles"),
            )
        };
        if let Some(directory) = std::env::var_os(CHILD_DIRECTORY) {
            // Earlier tests may initialize the parent's pool through WorldState,
            // while a fresh child's first plan would initialize Rayon's default.
            // Match the actual execution class without altering launch provenance.
            let threads = std::env::var(CHILD_THREADS).unwrap().parse().unwrap();
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();
            assert_eq!(rayon::current_num_threads(), threads);
            let family = std::env::var(CHILD_FAMILY).unwrap();
            let directory = PathBuf::from(directory);
            let status = make_runner(&directory, &family)
                .execute_batch(&directory.join("status.json"))
                .unwrap();
            assert_eq!(status.completed_runs, 1, "{status:?}");
            assert_eq!(status.failed_runs, 0);
            return;
        }
        let spawn = |directory: &Path, family: &str, label: &str| {
            fs::create_dir_all(directory).unwrap();
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "experiment_runner::tests::running_checkpoint_survives_process_kill_and_matches_uninterrupted_process", "--nocapture"])
                .env(CHILD_DIRECTORY, directory).env(CHILD_FAMILY, family)
                .env(CHILD_THREADS, rayon::current_num_threads().to_string())
                .stdout(Stdio::from(File::create(directory.join(format!("{label}.stdout.log"))).unwrap()))
                .stderr(Stdio::from(File::create(directory.join(format!("{label}.stderr.log"))).unwrap()))
                .spawn().map(RunnerChild).unwrap()
        };
        let wait = |mut child: RunnerChild| {
            let deadline = Instant::now() + Duration::from_secs(120);
            loop {
                if let Some(status) = child.0.try_wait().unwrap() {
                    assert!(status.success(), "runner subprocess failed: {status}");
                    break;
                }
                if Instant::now() >= deadline {
                    child.0.kill().unwrap();
                    child.0.wait().unwrap();
                    panic!("runner subprocess exceeded its 120-second test deadline");
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        };
        for family in ["mlp", "dwraon"] {
            let directory = tempfile::tempdir().unwrap().keep();
            println!("runner process restart evidence: {}", directory.display());
            let interrupted = directory.join("interrupted");
            let oracle = directory.join("oracle");
            let runner = make_runner(&interrupted, family);
            let record = runner.plan_batch().unwrap().remove(0);
            let path = runner.run_checkpoint_path(&record.run_id);
            let mut child = spawn(&interrupted, family, "first");
            let deadline = Instant::now() + Duration::from_secs(60);
            let observed = loop {
                if let Ok(Some(cut)) = read_optional_run_checkpoint(&path)
                    && (2..TICKS).contains(&cut.owner.host_checkpoint().tick().0)
                {
                    break cut.owner.host_checkpoint().tick().0;
                }
                if let Some(status) = child.0.try_wait().unwrap() {
                    panic!(
                        "runner exited before a nonzero cut was observed: {status}; {}",
                        interrupted.display()
                    );
                }
                if Instant::now() >= deadline {
                    child.0.kill().unwrap();
                    child.0.wait().unwrap();
                    panic!("no nonzero cut appeared within 60 seconds");
                }
                std::thread::sleep(Duration::from_millis(2));
            };
            child.0.kill().unwrap();
            let killed = child.0.wait().unwrap();
            assert!(!killed.success());
            let cut = read_optional_run_checkpoint(&path).unwrap().unwrap();
            let killed_tick = cut.owner.host_checkpoint().tick().0;
            assert!((observed..TICKS).contains(&killed_tick));
            let original_manifest = cut.metadata.manifest.canonical_json_bytes().unwrap();
            let interrupted_status: ExperimentBatchStatus =
                serde_json::from_slice(&fs::read(interrupted.join("status.json")).unwrap())
                    .unwrap();
            assert_eq!(interrupted_status.runs[0].state, RunState::Running);
            wait(spawn(&interrupted, family, "resumed"));
            wait(spawn(&oracle, family, "oracle"));
            let completed = runner
                .load_or_create_status(&interrupted.join("status.json"))
                .unwrap();
            let expected = make_runner(&oracle, family)
                .load_or_create_status(&oracle.join("status.json"))
                .unwrap();
            assert_eq!(
                completed.runs[0].final_digest,
                expected.runs[0].final_digest
            );
            assert_eq!(completed.runs[0].total_ticks, TICKS);
            let resumed_cut = read_optional_run_checkpoint(&path).unwrap().unwrap();
            let oracle_cut = read_optional_run_checkpoint(
                &make_runner(&oracle, family).run_checkpoint_path(&record.run_id),
            )
            .unwrap()
            .unwrap();
            assert_eq!(
                resumed_cut.owner.host_checkpoint().journal_cut(),
                oracle_cut.owner.host_checkpoint().journal_cut()
            );
            assert_eq!(
                resumed_cut
                    .metadata
                    .manifest
                    .canonical_json_bytes()
                    .unwrap(),
                original_manifest
            );
            println!(
                "{family}: killed actual process at durable tick {killed_tick}, resumed to {TICKS}, digest {:?}",
                completed.runs[0].final_digest
            );
        }
    }

    #[test]
    fn recorded_matched_cohort_keeps_exact_science_at_concurrency_one_and_two() {
        let directory = tempfile::tempdir().unwrap();
        let seeds = vec![53, 71];
        let variants = [
            "mlp",
            "dwraon",
            #[cfg(feature = "brain-ft")]
            "ft",
            #[cfg(feature = "neuro")]
            "neuroflow",
        ]
        .map(|family| ScenarioVariant {
            variant_id: family.into(),
            brain_family: family.into(),
            config_overrides: BTreeMap::new(),
            interventions: Vec::new(),
        })
        .to_vec();
        let mut outcomes = Vec::new();
        for concurrency in [1, 2] {
            let output = directory.path().join(format!("concurrency-{concurrency}"));
            let runner = MatchedSeedExperimentRunner::new(
                "concurrency-cohort",
                MatchedSeedCohort {
                    cohort_id: "cohort".into(),
                    seeds: seeds.clone(),
                },
                variants.clone(),
                3,
                concurrency,
                output.join("bundles"),
            );
            let status = runner.execute_batch(&output.join("status.json")).unwrap();
            assert_eq!(
                status.completed_runs,
                runner.plan_batch().unwrap().len(),
                "{status:?}"
            );
            assert_eq!(status.failed_runs, 0);
            assert_eq!(
                runner.execute_batch(&output.join("status.json")).unwrap(),
                status
            );
            outcomes.push(
                status
                    .runs
                    .into_iter()
                    .map(|record| {
                        assert_eq!(record.total_ticks, runner.max_ticks);
                        (
                            (record.brain_family, record.seed),
                            record.final_digest.unwrap(),
                        )
                    })
                    .collect::<BTreeMap<_, _>>(),
            );
        }
        assert_eq!(outcomes[0], outcomes[1]);
        // All treatment/seed cells must exist and differ. A collapsed
        // registry or seed schedule cannot masquerade as schedule independence.
        assert_eq!(outcomes[0].len(), variants.len() * seeds.len());
        assert_eq!(
            outcomes[0].values().collect::<BTreeSet<_>>().len(),
            outcomes[0].len()
        );
    }

    #[test]
    fn checkpoint_recovery_restores_candidate_before_replacing_the_committed_cut() {
        let directory = tempfile::tempdir().unwrap();
        let runner = progress_runner(directory.path(), vec![7]);
        let plan = runner.validated_plan().unwrap();
        let record = &plan.runs[0];
        let (variant, family) = &plan.variants[&record.variant_id];
        runner
            .prepare_run_checkpoint(
                variant,
                *family,
                record.seed,
                &record.run_id,
                &plan.digest,
            )
            .unwrap();
        let path = runner.run_checkpoint_path(&record.run_id);
        let temporary = status_temporary_path(&path).unwrap();
        let parent_bytes = fs::read(&path).unwrap();
        let parent = RunCheckpointV1::decode(&parent_bytes, &path).unwrap();
        let parent_digest = blake3::hash(&parent_bytes).to_hex().to_string();
        let _lease = ExperimentStatusWriterLease::acquire(&path).unwrap();

        // Capture actual one-tick owners from both executable families. Their
        // metadata and outer checksums cannot prove the expected family was run.
        let next_owner = |checkpoint: &RunCheckpointV1, chosen: ResolvedBrainFamily| {
            let mut owner = restore_run_owner(checkpoint, chosen, &record.run_id, &path).unwrap();
            let (_, namespace) = host_identifiers(&record.run_id);
            let mut frontend = NullFrontend::new(owner.local_port(), namespace);
            let mut now = 0;
            let submitted = frontend
                .submit_envelope(CommandEnvelope::new(
                    CommandId::from_client_sequence(namespace, 1),
                    HostCommand::Step,
                ))
                .unwrap();
            drive_command_to_volatile_commit(
                &record.run_id,
                "candidate owner step",
                &mut frontend,
                &mut owner,
                submitted.command_id(),
                1,
                0,
                &mut now,
            )
            .unwrap();
            seal_run_checkpoint(&record.run_id, &mut frontend, &mut owner, &mut now).unwrap();
            owner.capture_volatile_host_checkpoint_v1().unwrap()
        };
        let mut valid = RunCheckpointV1 {
            metadata: parent.metadata.clone(),
            owner: next_owner(&parent, *family),
        };
        valid.metadata.parent_digest = Some(parent_digest.clone());
        let other_family = runner
            .initial_run_checkpoint(
                variant,
                ResolvedBrainFamily::Dwraon,
                record.seed,
                &record.run_id,
                &plan.digest,
            )
            .unwrap();
        let mut wrong_family = RunCheckpointV1 {
            metadata: parent.metadata.clone(),
            owner: next_owner(&other_family, ResolvedBrainFamily::Dwraon),
        };
        wrong_family.metadata.parent_digest = Some(parent_digest);
        let wrong_bytes = wrong_family.encode(&temporary).unwrap();
        let decoded = RunCheckpointV1::decode(&wrong_bytes, &temporary).unwrap();
        runner
            .validate_run_checkpoint(
                &temporary,
                &decoded,
                variant,
                record.seed,
                &record.run_id,
                &plan.digest,
            )
            .expect("the authentic owner passes framing, metadata, capacity and schedule checks");
        assert_ne!(
            decoded
                .owner
                .host_checkpoint()
                .world_checkpoint()
                .required_brain_registry(),
            parent
                .owner
                .host_checkpoint()
                .world_checkpoint()
                .required_brain_registry()
        );
        assert!(
            restore_run_owner(&decoded, *family, &record.run_id, &temporary).is_err(),
            "the declared MLP executable cannot restore the real DWRAON owner"
        );
        fs::write(&temporary, &wrong_bytes).unwrap();

        for _ in 0..2 {
            let error = runner
                .load_run_checkpoint(
                    variant,
                    record.seed,
                    &record.run_id,
                    &plan.digest,
                    MAX_RUN_POPULATION,
                )
                .err()
                .expect("an unrestorable successor must not replace the recoverable parent");
            assert!(
                matches!(
                    &error,
                    ExperimentRunnerError::InvalidCheckpoint { path: refused, reason }
                        if refused == &temporary && reason.contains("prepared brain registry")
                ),
                "{error}"
            );
            assert_eq!(fs::read(&path).unwrap(), parent_bytes);
            assert_eq!(fs::read(&temporary).unwrap(), wrong_bytes);
            assert!(!path.with_extension("checkpoint.interrupted").exists());
            assert_eq!(
                restore_run_owner(&parent, *family, &record.run_id, &path)
                    .unwrap()
                    .world_tick()
                    .0,
                0
            );
        }

        // The same recovery path still admits a genuine, executable successor.
        let valid_bytes = valid.encode(&temporary).unwrap();
        let expected = restore_run_owner(&valid, *family, &record.run_id, &temporary)
            .unwrap()
            .scientific_digest_v1()
            .unwrap();
        fs::write(&temporary, &valid_bytes).unwrap();
        let recovered = runner
            .load_run_checkpoint(
                variant,
                record.seed,
                &record.run_id,
                &plan.digest,
                MAX_RUN_POPULATION,
            )
            .unwrap()
            .unwrap();
        assert_eq!(recovered.owner.host_checkpoint().tick().0, 1);
        assert_eq!(fs::read(&path).unwrap(), valid_bytes);
        assert!(!temporary.exists());
        assert_eq!(
            restore_run_owner(&recovered, *family, &record.run_id, &path)
                .unwrap()
                .scientific_digest_v1()
                .unwrap(),
            expected
        );
    }

    #[test]
    fn running_checkpoint_recovers_only_exact_complete_temporary_cuts_and_retains_partial_frames() {
        let directory = tempfile::tempdir().unwrap();
        let runner = MatchedSeedExperimentRunner::new(
            "temporary-cut",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![12, 13],
            },
            vec![ScenarioVariant {
                variant_id: "mlp".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            }],
            4,
            1,
            directory.path().join("bundles"),
        );
        let plan = runner.validated_plan().unwrap();
        let first = &plan.runs[0];
        let second = &plan.runs[1];
        let (variant, family) = plan.variants.get(&first.variant_id).unwrap();
        for record in &plan.runs {
            runner
                .prepare_run_checkpoint(variant, *family, record.seed, &record.run_id, &plan.digest)
                .unwrap();
        }
        let path = runner.run_checkpoint_path(&first.run_id);
        let temporary = status_temporary_path(&path).unwrap();
        let original = fs::read(&path).unwrap();
        let _lease = ExperimentStatusWriterLease::acquire(&path).unwrap();
        assert!(matches!(
            ExperimentStatusWriterLease::acquire(&path),
            Err(ExperimentRunnerError::StatusWriterLeaseHeld(_))
        ));
        let mut checkpoint = RunCheckpointV1::decode(&original, &path).unwrap();
        // These are parser/identity controls over a real captured cut, not live
        // fault observations. Every forged metadata frame has a fresh valid checksum.
        for field in ["build", "scenario", "seed", "family", "config", "plan"] {
            let mut changed = RunCheckpointV1 {
                metadata: checkpoint.metadata.clone(),
                owner: checkpoint.owner.clone(),
            };
            match field {
                "build" => {
                    changed.metadata.manifest.build.source_revision =
                        Some("0".repeat(blake3::OUT_LEN * 2))
                }
                "scenario" => changed.metadata.manifest.scenario.id = "foreign-scenario".into(),
                "seed" => changed.metadata.manifest.root_seed += 1,
                "family" => {
                    changed.metadata.manifest.brain_roster[0].kind = "dwraon.baseline".into()
                }
                "config" => {
                    changed.metadata.manifest.normalized_config["food_max"] =
                        serde_json::json!(0.777)
                }
                _ => changed.metadata.plan_digest = "0".repeat(blake3::OUT_LEN * 2),
            }
            fs::write(&temporary, changed.encode(&path).unwrap()).unwrap();
            assert!(
                matches!(runner.load_run_checkpoint(variant, first.seed, &first.run_id, &plan.digest, MAX_RUN_POPULATION),
                Err(ExperimentRunnerError::InvalidCheckpoint { reason, .. }) if reason.contains("validated plan")),
                "{field}"
            );
            assert_eq!(fs::read(&path).unwrap(), original);
        }
        for malformed in [
            Vec::new(),
            original[..original.len() - 1].to_vec(),
            {
                let mut bytes = original.clone();
                bytes[RUN_CHECKPOINT_MAGIC.len() - 1] = b'2';
                bytes
            },
            {
                let mut bytes = original.clone();
                let begin = RUN_CHECKPOINT_MAGIC.len();
                bytes[begin..begin + size_of::<u64>()].copy_from_slice(&u64::MAX.to_le_bytes());
                let end = bytes.len() - blake3::OUT_LEN;
                let digest = *blake3::hash(&bytes[..end]).as_bytes();
                bytes[end..].copy_from_slice(&digest);
                bytes
            },
        ] {
            assert!(matches!(
                RunCheckpointV1::decode(&malformed, &path),
                Err(ExperimentRunnerError::InvalidCheckpoint { .. })
            ));
        }
        File::create(&temporary)
            .unwrap()
            .set_len(MAX_RUN_CHECKPOINT_BYTES + 1)
            .unwrap();
        assert!(matches!(
            read_experiment_file_bounded(&temporary, MAX_RUN_CHECKPOINT_BYTES),
            Err(ExperimentRunnerError::InvalidCheckpoint { .. })
        ));
        #[cfg(unix)]
        {
            let symlink = directory.path().join("symlink.checkpoint");
            std::os::unix::fs::symlink(&path, &symlink).unwrap();
            assert!(matches!(
                read_experiment_file_bounded(&symlink, MAX_RUN_CHECKPOINT_BYTES),
                Err(ExperimentRunnerError::InvalidCheckpoint { .. })
            ));
            let hardlinked = directory.path().join("hardlinked.checkpoint");
            fs::write(&hardlinked, &original).unwrap();
            let alias = directory.path().join("alias.checkpoint");
            fs::hard_link(&hardlinked, &alias).unwrap();
            assert!(matches!(
                read_experiment_file_bounded(&alias, MAX_RUN_CHECKPOINT_BYTES),
                Err(ExperimentRunnerError::InvalidCheckpoint { .. })
            ));
            assert_eq!(fs::read(&path).unwrap(), original);
        }
        let mut core = restore_run_owner(&checkpoint, *family, &first.run_id, &path).unwrap();
        let (_, namespace) = host_identifiers(&first.run_id);
        let mut frontend = NullFrontend::new(core.local_port(), namespace);
        let mut now = 0;
        let submitted = frontend
            .submit_envelope(CommandEnvelope::new(
                CommandId::from_client_sequence(namespace, 1),
                HostCommand::Step,
            ))
            .unwrap();
        drive_command_to_volatile_commit(
            &first.run_id,
            "temporary candidate step",
            &mut frontend,
            &mut core,
            submitted.command_id(),
            1,
            0,
            &mut now,
        )
        .unwrap();
        seal_run_checkpoint(&first.run_id, &mut frontend, &mut core, &mut now).unwrap();
        checkpoint.owner = core.capture_volatile_host_checkpoint_v1().unwrap();
        checkpoint.metadata.parent_digest = Some(blake3::hash(&original).to_hex().to_string());
        let candidate = checkpoint.encode(&path).unwrap();
        // A complete, checksummed frame from another seed must remain a refusal.
        fs::write(
            &temporary,
            fs::read(runner.run_checkpoint_path(&second.run_id)).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            runner.load_run_checkpoint(
                variant,
                first.seed,
                &first.run_id,
                &plan.digest,
                MAX_RUN_POPULATION
            ),
            Err(ExperimentRunnerError::InvalidCheckpoint { .. })
        ));
        assert_eq!(fs::read(&path).unwrap(), original);
        // Correct run but wrong parent is a separate refusal, even with a valid checksum.
        checkpoint.metadata.parent_digest = Some("0".repeat(blake3::OUT_LEN * 2));
        fs::write(&temporary, checkpoint.encode(&path).unwrap()).unwrap();
        assert!(matches!(
            runner.load_run_checkpoint(
                variant,
                first.seed,
                &first.run_id,
                &plan.digest,
                MAX_RUN_POPULATION
            ),
            Err(ExperimentRunnerError::InvalidCheckpoint { .. })
        ));
        assert_eq!(fs::read(&path).unwrap(), original);
        fs::write(&temporary, &candidate).unwrap();
        let promoted = runner
            .load_run_checkpoint(
                variant,
                first.seed,
                &first.run_id,
                &plan.digest,
                MAX_RUN_POPULATION,
            )
            .unwrap()
            .unwrap();
        assert_eq!(promoted.owner.host_checkpoint().tick().0, 1);
        assert_eq!(fs::read(&path).unwrap(), candidate);
        assert!(!temporary.exists());
        let partial = &candidate[..candidate.len() / 2];
        fs::write(&temporary, partial).unwrap();
        assert_eq!(
            runner
                .load_run_checkpoint(
                    variant,
                    first.seed,
                    &first.run_id,
                    &plan.digest,
                    MAX_RUN_POPULATION
                )
                .unwrap()
                .unwrap()
                .owner
                .host_checkpoint()
                .tick()
                .0,
            1
        );
        assert_eq!(
            fs::read(path.with_extension("checkpoint.interrupted")).unwrap(),
            partial
        );
        assert_eq!(fs::read(&path).unwrap(), candidate);
        // The retained incident is a finite slot. A second interruption refuses;
        // it cannot silently overwrite evidence or create unbounded forensic files.
        fs::write(&temporary, partial).unwrap();
        assert!(matches!(
            runner.load_run_checkpoint(
                variant,
                first.seed,
                &first.run_id,
                &plan.digest,
                MAX_RUN_POPULATION
            ),
            Err(ExperimentRunnerError::InvalidCheckpoint { .. })
        ));
        assert_eq!(fs::read(&temporary).unwrap(), partial);
        assert_eq!(fs::read(&path).unwrap(), candidate);
        // Corruption of the authoritative committed frame never falls back to a fresh world.
        let mut corrupt = candidate;
        corrupt[RUN_CHECKPOINT_MAGIC.len()] ^= 1;
        fs::write(&path, &corrupt).unwrap();
        assert!(matches!(
            runner.load_run_checkpoint(
                variant,
                first.seed,
                &first.run_id,
                &plan.digest,
                MAX_RUN_POPULATION
            ),
            Err(ExperimentRunnerError::InvalidCheckpoint { .. })
        ));
        assert_eq!(fs::read(&path).unwrap(), corrupt);
    }

    #[test]
    fn completed_bundles_recover_the_status_commit_gap_without_reexecution() {
        let (runner, state_file, running, completed) = completed_bundle_gap_fixture(&[7, 9]);
        let before = completed
            .iter()
            .map(|record| {
                fs::read(
                    runner
                        .output_dir
                        .join(&record.run_id)
                        .join("bundle_manifest.json"),
                )
                .expect("completed manifest")
            })
            .collect::<Vec<_>>();

        // Cancellation prevents admitting new work, but must not hide completed
        // work that already passed the canonical verifier.
        let recovered = runner
            .execute_batch_until_cancelled(&state_file, &std::sync::atomic::AtomicBool::new(true))
            .expect("recover completed bundles");
        assert_eq!(recovered.runs, completed);
        assert_eq!(recovered.completed_runs, 2);
        assert_eq!(recovered.failed_runs, 0);
        assert_eq!(recovered.generation, running.generation + 1);
        for (record, bytes) in recovered.runs.iter().zip(before) {
            assert_eq!(
                fs::read(
                    runner
                        .output_dir
                        .join(&record.run_id)
                        .join("bundle_manifest.json")
                )
                .expect("retained manifest"),
                bytes
            );
        }
        let status_bytes = fs::read(&state_file).expect("recovered status");
        assert_eq!(
            runner.execute_batch(&state_file).expect("repeat recovery"),
            recovered
        );
        assert_eq!(
            fs::read(&state_file).expect("unchanged status"),
            status_bytes
        );
    }

    #[test]
    fn completed_bundle_recovery_refuses_corruption_and_cross_run_swaps() {
        let (runner, state_file, mut expected, completed) = completed_bundle_gap_fixture(&[7, 9]);
        let first = runner.output_dir.join(&completed[0].run_id);
        let second = runner.output_dir.join(&completed[1].run_id);
        let paths = [
            "bundle_manifest.json",
            "evidence/run.json",
            "exports/summary.csv",
            FINAL_CHECKPOINT_ARTIFACT.0,
        ];
        let original = paths.map(|path| fs::read(first.join(path)).expect("original artifact"));

        fs::write(first.join(paths[2]), b"corrupt summary\n").expect("plant corrupt artifact");
        assert!(matches!(
            runner.execute_batch(&state_file),
            Err(ExperimentRunnerError::Bundle { .. })
        ));
        // The corrupt run remains Running, while its independently verified
        // companion must survive the failed batch recovery.
        expected.runs[1] = completed[1].clone();
        expected.generation += 1;
        recompute_status_counts(&mut expected);
        assert_eq!(
            runner
                .load_or_create_status(&state_file)
                .expect("partial recovery"),
            expected
        );
        let status_bytes = fs::read(&state_file).expect("partially recovered status");
        fs::write(first.join(paths[2]), &original[2]).expect("restore summary");

        // All swapped bytes remain mutually consistent. The semantic run/seed
        // binding must reject them even though their artifact hashes still pass.
        for path in paths {
            fs::write(
                first.join(path),
                fs::read(second.join(path)).expect("other run bytes"),
            )
            .expect("plant complete cross-run swap");
        }
        scriptbots_storage::verify_run_bundle(&first)
            .expect("the complete cross-run swap remains byte-valid");
        assert!(matches!(
            runner.execute_batch(&state_file),
            Err(ExperimentRunnerError::Bundle { .. })
        ));
        assert_eq!(
            fs::read(&state_file).expect("refused swap status"),
            status_bytes
        );
        for (path, bytes) in paths.into_iter().zip(original) {
            fs::write(first.join(path), bytes).expect("restore original bundle");
        }
        assert_eq!(
            runner
                .execute_batch(&state_file)
                .expect("restored success")
                .runs,
            completed
        );
    }

    #[test]
    fn completed_bundle_verification_recomputes_the_reported_final_digest() {
        let (runner, _state_file, _running, completed) = completed_bundle_gap_fixture(&[7, 9]);
        assert_ne!(completed[0].final_digest, completed[1].final_digest);
        let mut forged = completed[1].clone();
        let actual_digest = forged.final_digest.clone().unwrap();
        forged.final_digest = completed[0].final_digest.clone();
        let false_digest = forged.final_digest.as_ref().unwrap();
        let bundle_dir = runner.output_dir.join(&forged.run_id);
        let evidence_path = bundle_dir.join("evidence/run.json");
        let mut evidence: RunEvidenceV2 =
            serde_json::from_slice(&fs::read(&evidence_path).unwrap()).unwrap();
        evidence.final_digest.clone_from(false_digest);
        fs::write(&evidence_path, serde_json::to_vec(&evidence).unwrap()).unwrap();
        let summary_path = bundle_dir.join("exports/summary.csv");
        let summary = fs::read_to_string(&summary_path).unwrap();
        assert!(summary.contains(&actual_digest));
        fs::write(&summary_path, summary.replace(&actual_digest, false_digest)).unwrap();
        let manifest_path = bundle_dir.join("bundle_manifest.json");
        let mut bundle: RunBundleV1 =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        for entry in &mut bundle.artifacts {
            let bytes = fs::read(bundle_dir.join(&entry.relative_path)).unwrap();
            entry.bytes_len = u64::try_from(bytes.len()).unwrap();
            entry.blake3_hex = blake3::hash(&bytes).to_hex().to_string();
        }
        fs::write(&manifest_path, serde_json::to_vec(&bundle).unwrap()).unwrap();
        scriptbots_storage::verify_run_bundle(&bundle_dir)
            .expect("all forged reports remain mutually consistent and byte-valid");

        let plan = runner.validated_plan().unwrap();
        let (variant, family) = &plan.variants[&forged.variant_id];
        let error = runner
            .verify_completed_bundle(&forged, variant, *family, &plan.digest)
            .expect_err("reported digest must match the actual restored final checkpoint");
        assert!(
            error.to_string().contains("restored final checkpoint"),
            "{error}"
        );
        let error = verify_portable_run_bundle(&bundle_dir)
            .expect_err("offline verification must reject the same byte-valid false science");
        assert!(
            error.to_string().contains("computed scientific digest"),
            "{error}"
        );
    }

    #[test]
    fn offline_verification_restores_relocated_bundles_and_requires_experiment_roles() {
        let (runner, _state_file, _running, completed) = completed_bundle_gap_fixture(&[7]);
        let record = &completed[0];
        let original = runner.output_dir.join(&record.run_id);
        let relocated = runner.output_dir.join("renamed-portable-bundle");
        fs::rename(&original, &relocated).unwrap();
        let verified = verify_portable_run_bundle(&relocated).unwrap();
        let final_state = verified.final_state.unwrap();
        assert_eq!(final_state.run_id, record.run_id);
        assert_eq!(final_state.tick, record.total_ticks);
        assert_eq!(Some(final_state.scientific_digest), record.final_digest);
        assert_eq!(verified.artifacts.max_tick, final_state.tick);

        let path = relocated.join("bundle_manifest.json");
        let original_manifest = fs::read(&path).unwrap();
        let bundle: RunBundleV1 = serde_json::from_slice(&original_manifest).unwrap();
        let mut absent = bundle.clone();
        absent
            .artifacts
            .retain(|entry| entry.relative_path != FINAL_CHECKPOINT_ARTIFACT.0);
        let mut mislabelled = bundle.clone();
        mislabelled
            .artifacts
            .iter_mut()
            .find(|entry| entry.relative_path == FINAL_CHECKPOINT_ARTIFACT.0)
            .unwrap()
            .artifact_type = "ordinary-binary".into();
        let mut stripped = bundle;
        stripped.artifacts.clear();
        let cases = [absent, mislabelled, stripped];
        let expected_cells = cases.len();
        let mut refused = 0;
        for changed in cases {
            fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
            scriptbots_storage::verify_run_bundle(&relocated)
                .expect("these index changes preserve byte integrity");
            let error = verify_portable_run_bundle(&relocated)
                .expect_err("declared experiment cannot fall back to byte-only verification");
            assert!(error.to_string().contains("exactly once"), "{error}");
            refused += 1;
        }
        assert_eq!(refused, expected_cells);
        fs::write(&path, original_manifest).unwrap();
        assert!(
            verify_portable_run_bundle(&relocated)
                .unwrap()
                .final_state
                .is_some()
        );
    }

    #[test]
    fn zero_tick_completed_bundle_restores_its_initial_owner_cut() {
        let directory = tempfile::tempdir().unwrap();
        let runner = MatchedSeedExperimentRunner::new(
            "zero-tick",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![17],
            },
            vec![ScenarioVariant {
                variant_id: "mlp".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            }],
            0,
            1,
            directory.path().join("bundles"),
        );
        let state_file = directory.path().join("status.json");
        let status = runner.execute_batch(&state_file).unwrap();
        assert_eq!(status.completed_runs, status.total_runs);
        assert_eq!(status.failed_runs, 0);
        for run in &status.runs {
            let path = runner
                .output_dir
                .join(&run.run_id)
                .join(FINAL_CHECKPOINT_ARTIFACT.0);
            let checkpoint = RunCheckpointV1::decode(&fs::read(&path).unwrap(), &path).unwrap();
            assert!(checkpoint.metadata.parent_digest.is_none());
            let restored =
                restore_run_owner(&checkpoint, ResolvedBrainFamily::Mlp, &run.run_id, &path)
                    .unwrap();
            assert_eq!(restored.world_tick().0, runner.max_ticks);
            assert_eq!(
                Some(restored.scientific_digest_v1().unwrap().overall),
                run.final_digest
            );
            let offline = verify_portable_run_bundle(&runner.output_dir.join(&run.run_id))
                .unwrap()
                .final_state
                .unwrap();
            assert_eq!(offline.tick, 0);
            assert_eq!(offline.alive_agents, restored.world().agents().len());
            assert_eq!(Some(offline.scientific_digest), run.final_digest);
        }
        assert_eq!(runner.execute_batch(&state_file).unwrap(), status);
    }

    #[test]
    fn completed_bundle_final_checkpoint_guards_observe_distinct_defects() {
        let (runner, _state_file, _running, completed) = completed_bundle_gap_fixture(&[7, 9]);
        let record = &completed[0];
        let plan = runner.validated_plan().unwrap();
        let (variant, family) = &plan.variants[&record.variant_id];
        let bundle_dir = runner.output_dir.join(&record.run_id);
        let checkpoint_path = bundle_dir.join(FINAL_CHECKPOINT_ARTIFACT.0);
        let manifest_path = bundle_dir.join("bundle_manifest.json");
        let original_checkpoint = fs::read(&checkpoint_path).unwrap();
        let original_manifest = fs::read(&manifest_path).unwrap();
        let other = fs::read(
            runner
                .output_dir
                .join(&completed[1].run_id)
                .join(FINAL_CHECKPOINT_ARTIFACT.0),
        )
        .unwrap();
        assert_ne!(original_checkpoint, other);
        let initial = runner
            .initial_run_checkpoint(variant, *family, record.seed, &record.run_id, &plan.digest)
            .unwrap()
            .encode(&checkpoint_path)
            .unwrap();
        let mut corrupt = original_checkpoint.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        let mut observed = BTreeSet::new();
        let cases = [
            ("corrupt", corrupt, "frame checksum differs"),
            ("wrong-tick", initial, "exact completion tick"),
            ("cross-run", other, "validated plan"),
        ];
        let expected_cells = cases.len();
        for (label, bytes, expected_reason) in cases {
            fs::write(&checkpoint_path, &bytes).unwrap();
            let mut bundle: RunBundleV1 = serde_json::from_slice(&original_manifest).unwrap();
            let entry = bundle
                .artifacts
                .iter_mut()
                .find(|entry| entry.relative_path == FINAL_CHECKPOINT_ARTIFACT.0)
                .unwrap();
            entry.bytes_len = u64::try_from(bytes.len()).unwrap();
            entry.blake3_hex = blake3::hash(&bytes).to_hex().to_string();
            fs::write(&manifest_path, serde_json::to_vec(&bundle).unwrap()).unwrap();
            scriptbots_storage::verify_run_bundle(&bundle_dir)
                .expect("outer artifact hashes remain valid for each distinct defect");
            let error = runner
                .verify_completed_bundle(record, variant, *family, &plan.digest)
                .expect_err(label);
            let reason = error.to_string();
            assert!(reason.contains(expected_reason), "{label}: {reason}");
            assert!(
                reason.contains(&checkpoint_path.display().to_string()),
                "failure must name the bundled cut it actually read: {reason}"
            );
            observed.insert(reason);
            assert!(
                verify_portable_run_bundle(&bundle_dir).is_err(),
                "offline consumer must also reject {label}"
            );
        }
        assert_eq!(
            observed.len(),
            expected_cells,
            "all declared defect cells must differ"
        );
        fs::write(&checkpoint_path, &original_checkpoint).unwrap();
        fs::write(&manifest_path, &original_manifest).unwrap();
        runner
            .verify_completed_bundle(record, variant, *family, &plan.digest)
            .expect("the exact original cut still verifies");

        let summary_path = bundle_dir.join("exports/summary.csv");
        let original_summary = fs::read(&summary_path).unwrap();
        let mut summary = verify_portable_run_bundle(&bundle_dir)
            .unwrap()
            .final_state
            .unwrap()
            .summary;
        summary.alive_agents = u32::MAX;
        let summary = summary.canonical_csv_bytes().unwrap();
        fs::write(&summary_path, &summary).unwrap();
        let mut changed: RunBundleV1 = serde_json::from_slice(&original_manifest).unwrap();
        let entry = changed
            .artifacts
            .iter_mut()
            .find(|entry| entry.relative_path == "exports/summary.csv")
            .unwrap();
        entry.bytes_len = u64::try_from(summary.len()).unwrap();
        entry.blake3_hex = blake3::hash(&summary).to_hex().to_string();
        fs::write(&manifest_path, serde_json::to_vec(&changed).unwrap()).unwrap();
        scriptbots_storage::verify_run_bundle(&bundle_dir)
            .expect("a forged population still has valid artifact hashes");
        let error = runner
            .verify_completed_bundle(record, variant, *family, &plan.digest)
            .expect_err("reported population must match the restored final state");
        assert!(error.to_string().contains("summary differs"), "{error}");
        let error = verify_portable_run_bundle(&bundle_dir)
            .expect_err("offline population must come from the actual owner");
        assert!(error.to_string().contains("summary differs"), "{error}");
        fs::write(&summary_path, &original_summary).unwrap();
        fs::write(&manifest_path, &original_manifest).unwrap();

        let mut missing_index: RunBundleV1 = serde_json::from_slice(&original_manifest).unwrap();
        missing_index
            .artifacts
            .retain(|entry| entry.relative_path != FINAL_CHECKPOINT_ARTIFACT.0);
        fs::write(&manifest_path, serde_json::to_vec(&missing_index).unwrap()).unwrap();
        scriptbots_storage::verify_run_bundle(&bundle_dir)
            .expect("an unindexed file does not prove the required role");
        let error = runner
            .verify_completed_bundle(record, variant, *family, &plan.digest)
            .expect_err("the final cut must be indexed with its actual artifact role");
        assert!(error.to_string().contains("exactly once"), "{error}");
        fs::write(&manifest_path, &original_manifest).unwrap();

        let preserved = runner.output_dir.join("preserved-final-checkpoint");
        fs::rename(&checkpoint_path, &preserved).unwrap();
        assert!(
            runner
                .verify_completed_bundle(record, variant, *family, &plan.digest)
                .is_err(),
            "an absent indexed cut must refuse verification"
        );
        fs::rename(&preserved, &checkpoint_path).unwrap();
        runner
            .verify_completed_bundle(record, variant, *family, &plan.digest)
            .expect("restoring the same physical cut repairs the bundle");
    }

    #[test]
    fn completed_bundle_recovery_preserves_finished_work_after_prior_bundle_corruption() {
        let (runner, state_file, mut expected, completed) = completed_bundle_gap_fixture(&[7, 9]);
        expected.runs[0] = completed[0].clone();
        recompute_status_counts(&mut expected);
        runner
            .save_status(&state_file, &mut expected)
            .expect("prior completion published");
        let summary = runner
            .output_dir
            .join(&completed[0].run_id)
            .join("exports/summary.csv");
        let original = fs::read(&summary).expect("original summary");
        fs::write(&summary, b"corrupt prior completion\n").expect("plant corruption");

        assert!(matches!(
            runner.execute_batch(&state_file),
            Err(ExperimentRunnerError::Bundle { run_id, .. }) if run_id == completed[0].run_id
        ));
        expected.runs[1] = completed[1].clone();
        expected.generation += 1;
        recompute_status_counts(&mut expected);
        assert_eq!(
            runner
                .load_or_create_status(&state_file)
                .expect("verified companion retained"),
            expected
        );
        let bytes = fs::read(&state_file).expect("recovered status");
        assert!(matches!(
            runner.execute_batch(&state_file),
            Err(ExperimentRunnerError::Bundle { run_id, .. }) if run_id == completed[0].run_id
        ));
        assert_eq!(fs::read(&state_file).expect("unchanged refusal"), bytes);
        fs::write(&summary, original).expect("restore prior completion");
        assert_eq!(
            runner
                .execute_batch(&state_file)
                .expect("restored bundles verify")
                .runs,
            completed
        );
    }

    #[test]
    fn completed_bundle_recovery_publishes_finished_runs_before_refusing_unfinished_work() {
        for seed in [7, 9] {
            let (runner, state_file, running, completed) = completed_bundle_gap_fixture(&[seed]);
            let unfinished = running
                .runs
                .iter()
                .find(|record| record.seed != seed)
                .expect("one unfinished run");
            assert!(matches!(
                runner.execute_batch(&state_file),
                Err(ExperimentRunnerError::ResumeCheckpointUnavailable { run_id, .. })
                    if run_id == unfinished.run_id
            ));
            let recovered = runner
                .load_or_create_status(&state_file)
                .expect("saved recovery");
            assert_eq!(recovered.completed_runs, completed.len());
            assert_eq!(recovered.generation, running.generation + 1);
            assert!(recovered.runs.contains(&completed[0]));
            assert!(recovered.runs.contains(unfinished));
            let bytes = fs::read(&state_file).expect("partially recovered status");
            assert!(matches!(
                runner.execute_batch(&state_file),
                Err(ExperimentRunnerError::ResumeCheckpointUnavailable { .. })
            ));
            assert_eq!(
                fs::read(&state_file).expect("unchanged retry status"),
                bytes
            );
        }
    }

    fn completed_bundle_gap_fixture(
        seeds_to_complete: &[u64],
    ) -> (
        MatchedSeedExperimentRunner,
        PathBuf,
        ExperimentBatchStatus,
        Vec<RunRecord>,
    ) {
        let directory = tempfile::tempdir()
            .expect("completion gap directory")
            .keep();
        let state_file = directory.join("status.json");
        let runner = MatchedSeedExperimentRunner::new(
            "completion-gap",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![7, 9],
            },
            vec![ScenarioVariant {
                variant_id: "mlp".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            }],
            2,
            2,
            directory.join("bundles"),
        );
        let mut running = runner
            .load_or_create_status(&state_file)
            .expect("fresh batch");
        for record in &mut running.runs {
            record.state = RunState::Running;
        }
        runner
            .save_status(&state_file, &mut running)
            .expect("durable running boundary");
        let completed = seeds_to_complete
            .iter()
            .map(|&seed| {
                runner
                    .execute_single_run(&runner.variants[0], seed)
                    .expect("real completed HostCore run")
            })
            .collect();
        (runner, state_file, running, completed)
    }

    #[test]
    fn active_wave_cancellation_retains_nonzero_cuts_and_resumes_exact_science() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, Instant};

        let families = ["mlp", "dwraon"];
        let concurrency_levels = [1, 2];
        let mut observed_cells = BTreeSet::new();
        for family in families {
            for concurrency in concurrency_levels {
                let directory = tempfile::tempdir().unwrap().keep();
                println!(
                    "active cancellation evidence: {} family={family} concurrency={concurrency}",
                    directory.display()
                );
                let runner = MatchedSeedExperimentRunner::new(
                    "active-cancel",
                    MatchedSeedCohort {
                        cohort_id: "cohort".into(),
                        seeds: vec![3, 4, 5, 6],
                    },
                    vec![ScenarioVariant {
                        variant_id: family.into(),
                        brain_family: family.into(),
                        config_overrides: BTreeMap::from([
                            ("world_width".into(), serde_json::json!(40)),
                            ("world_height".into(), serde_json::json!(40)),
                            ("food_cell_size".into(), serde_json::json!(10)),
                        ]),
                        interventions: Vec::new(),
                    }],
                    40,
                    concurrency,
                    directory.join("interrupted"),
                );
                let state_file = directory.join("status.json");
                let mut initial = runner.load_or_create_status(&state_file).unwrap();
                initial.runs[0].state = RunState::Running;
                runner.save_status(&state_file, &mut initial).unwrap();
                initial.runs[0] = runner
                    .execute_single_run(&runner.variants[0], runner.cohort.seeds[0])
                    .unwrap();
                recompute_status_counts(&mut initial);
                runner.save_status(&state_file, &mut initial).unwrap();
                let completed_bundle = runner.output_dir.join(&initial.runs[0].run_id);
                let original_manifest =
                    fs::read(completed_bundle.join("bundle_manifest.json")).unwrap();
                let active_ids = initial
                    .runs
                    .iter()
                    .skip(1)
                    .take(concurrency)
                    .map(|record| record.run_id.clone())
                    .collect::<Vec<_>>();
                let cancel = AtomicBool::new(false);
                let stopped = std::thread::scope(|scope| {
                    let worker =
                        scope.spawn(|| runner.execute_batch_until_cancelled(&state_file, &cancel));
                    let deadline = Instant::now() + Duration::from_secs(30);
                    loop {
                        let all_started = active_ids.iter().all(|id| {
                            read_optional_run_checkpoint(&runner.run_checkpoint_path(id))
                                .ok()
                                .flatten()
                                .is_some_and(|cut| cut.owner.host_checkpoint().tick().0 > 0)
                        });
                        if all_started {
                            let status = runner.status_snapshot(&state_file).unwrap();
                            let progress = runner
                                .checkpoint_progress_snapshot(&state_file, &status, true)
                                .unwrap();
                            if !active_ids
                                .iter()
                                .all(|id| progress[id].observed_tick().is_some_and(|tick| tick > 0))
                            {
                                assert!(
                                    Instant::now() < deadline,
                                    "published progress never appeared"
                                );
                                std::thread::yield_now();
                                continue;
                            }
                            assert!(
                                status
                                    .runs
                                    .iter()
                                    .filter(|run| run.state == RunState::Running)
                                    .all(|run| run.total_ticks == 0)
                            );
                            cancel.store(true, Ordering::Release);
                            break;
                        }
                        assert!(Instant::now() < deadline, "active cuts never appeared");
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    worker.join().unwrap().unwrap()
                });
                assert_eq!(stopped.completed_runs, 1);
                assert_eq!(stopped.failed_runs, 0);
                assert!(!stopped.is_finished());
                assert_eq!(stopped.runs[0], initial.runs[0]);
                let active_count = stopped
                    .runs
                    .iter()
                    .filter(|record| record.state == RunState::Running)
                    .count();
                assert_eq!(active_count, concurrency);
                observed_cells.insert((stopped.runs[1].brain_family.clone(), active_count));
                let progress = runner
                    .checkpoint_progress_snapshot(&state_file, &stopped, false)
                    .unwrap();
                let mut saved_cuts = Vec::new();
                for (index, record) in stopped.runs.iter().enumerate().skip(1) {
                    if index <= concurrency {
                        assert_eq!(record.state, RunState::Running, "active run must suspend");
                        assert_eq!(record.total_ticks, 0, "status retains its existing schema");
                        assert!(record.final_digest.is_none());
                        assert!(record.bundle_path.is_none());
                        assert!(record.error_reason.is_none());
                        let path = runner.run_checkpoint_path(&record.run_id);
                        let bytes = fs::read(&path).unwrap();
                        let cut = RunCheckpointV1::decode(&bytes, &path).unwrap();
                        let tick = cut.owner.host_checkpoint().tick().0;
                        assert!(tick > 0 && tick < runner.max_ticks, "stopped tick {tick}");
                        assert_eq!(
                            progress[&record.run_id],
                            ExperimentCheckpointProgress::Observed {
                                tick,
                                checkpoint_digest: blake3::hash(&bytes).to_hex().to_string(),
                                source: CheckpointProgressSource::Published,
                            }
                        );
                        let restored = restore_run_owner(
                            &cut,
                            ResolvedBrainFamily::resolve(&runner.variants[0]).unwrap(),
                            &record.run_id,
                            &path,
                        )
                        .unwrap();
                        assert_eq!(restored.latest_snapshot().lifecycle, HostLifecycle::Running);
                        assert!(!runner.output_dir.join(&record.run_id).exists());
                        fs::write(
                            directory.join(format!("stopped-{}.checkpoint", record.run_id)),
                            &bytes,
                        )
                        .unwrap();
                        saved_cuts.push((path, bytes));
                    } else {
                        assert_eq!(record.state, RunState::Pending);
                        assert!(!runner.run_checkpoint_path(&record.run_id).exists());
                    }
                }
                let status_bytes = fs::read(&state_file).unwrap();
                fs::write(directory.join("cancelled-status.json"), &status_bytes).unwrap();
                assert_eq!(
                    runner
                        .execute_batch_until_cancelled(&state_file, &cancel)
                        .unwrap(),
                    stopped,
                    "repeated cancellation must not admit another wave"
                );
                assert_eq!(fs::read(&state_file).unwrap(), status_bytes);
                for (path, bytes) in saved_cuts {
                    assert_eq!(fs::read(path).unwrap(), bytes);
                }
                let mut invalid = stopped.clone();
                invalid.runs[1].total_ticks = 1;
                assert!(matches!(
                    runner.save_status(&state_file, &mut invalid),
                    Err(ExperimentRunnerError::InvalidStatus { .. })
                ));
                assert_eq!(fs::read(&state_file).unwrap(), status_bytes);
                let resumed = runner.execute_batch(&state_file).unwrap();
                assert!(resumed.is_finished());
                assert_eq!(resumed.failed_runs, 0);
                assert_eq!(resumed.completed_runs, resumed.total_runs);
                assert!(
                    runner
                        .checkpoint_progress_snapshot(&state_file, &resumed, false)
                        .unwrap()
                        .values()
                        .all(|progress| progress.observed_tick() == Some(runner.max_ticks))
                );
                assert_eq!(
                    fs::read(completed_bundle.join("bundle_manifest.json")).unwrap(),
                    original_manifest
                );
                let oracle = MatchedSeedExperimentRunner::new(
                    &runner.experiment_id,
                    runner.cohort.clone(),
                    runner.variants.clone(),
                    runner.max_ticks,
                    concurrency,
                    directory.join("oracle"),
                );
                let expected = oracle
                    .execute_batch(&directory.join("oracle.json"))
                    .unwrap();
                for (actual, expected) in resumed.runs.iter().zip(expected.runs) {
                    assert_eq!(actual.run_id, expected.run_id);
                    assert_eq!(actual.final_digest, expected.final_digest);
                    assert_eq!(actual.total_ticks, expected.total_ticks);
                    scriptbots_storage::verify_run_bundle(Path::new(
                        actual.bundle_path.as_ref().unwrap(),
                    ))
                    .unwrap();
                }
            }
        }
        assert_eq!(
            observed_cells.len(),
            families.len() * concurrency_levels.len()
        );
    }

    #[test]
    fn cancellation_before_a_wave_leaves_runs_pending_and_resume_completes_them() {
        let temp_dir = tempfile::tempdir().expect("temporary status directory");
        let state_file = temp_dir.path().join("status.json");
        let runner = MatchedSeedExperimentRunner::new(
            "exp-cancel",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![3, 4],
            },
            vec![ScenarioVariant {
                variant_id: "mlp".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            }],
            3,
            1,
            temp_dir.path().join("bundles"),
        );

        let cancelled = runner
            .execute_batch_until_cancelled(&state_file, &std::sync::atomic::AtomicBool::new(true))
            .expect("a pre-cancelled batch stops cleanly");
        assert_eq!(cancelled.completed_runs, 0);
        assert!(!cancelled.is_finished());
        assert!(
            cancelled
                .runs
                .iter()
                .all(|run| run.state == RunState::Pending)
        );

        let resumed = runner
            .execute_batch(&state_file)
            .expect("the same state file resumes the pending runs");
        assert!(resumed.is_finished());
        assert_eq!(resumed.completed_runs, 2);
        assert_eq!(resumed.failed_runs, 0);
        assert!(
            resumed
                .runs
                .iter()
                .all(|run| run.state == RunState::Completed
                    && run.total_ticks == 3
                    && run.final_digest.is_some())
        );
    }

    #[test]
    fn status_writer_lease_and_generation_reject_concurrent_or_regressive_writes() {
        let temp_dir = tempfile::tempdir().expect("temporary status directory");
        let state_file = temp_dir.path().join("status.json");
        let runner = MatchedSeedExperimentRunner::new(
            "exp-status-cas",
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![7],
            },
            vec![ScenarioVariant {
                variant_id: "mlp".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            }],
            1,
            1,
            temp_dir.path().join("bundles"),
        );
        let mut status = runner
            .load_or_create_status(&state_file)
            .expect("fresh status");
        assert_eq!(status.generation, 0);
        let mut stale = status.clone();
        status.runs[0].state = RunState::Running;
        runner
            .save_status(&state_file, &mut status)
            .expect("pending to running is monotonic");
        assert_eq!(status.generation, 1);

        stale.runs[0].state = RunState::Running;
        assert!(matches!(
            runner.save_status(&state_file, &mut stale),
            Err(ExperimentRunnerError::StaleStatusGeneration {
                proposed: 0,
                committed: 1,
                ..
            })
        ));

        let mut regression = status.clone();
        regression.runs[0].state = RunState::Pending;
        assert!(matches!(
            runner.save_status(&state_file, &mut regression),
            Err(ExperimentRunnerError::InvalidStatus { .. })
        ));

        let lease =
            ExperimentStatusWriterLease::acquire(&state_file).expect("first writer owns lease");
        let competing_path = state_file.clone();
        let competing =
            std::thread::spawn(move || ExperimentStatusWriterLease::acquire(&competing_path))
                .join()
                .expect("lease probe thread returns");
        assert!(matches!(
            competing,
            Err(ExperimentRunnerError::StatusWriterLeaseHeld(_))
        ));
        drop(lease);
    }

    #[test]
    fn matched_seed_execution_uses_real_host_families_and_reopens_bundles() {
        let temp_dir = tempfile::tempdir().unwrap();
        let state_file = temp_dir.path().join("experiment_state.json");
        let output_dir = temp_dir.path().join("bundles");

        let cohort = MatchedSeedCohort {
            cohort_id: "cohort-test".into(),
            seeds: vec![1001],
        };
        let variants = vec![
            ScenarioVariant {
                variant_id: "variant_a".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            },
            ScenarioVariant {
                variant_id: "variant_b".into(),
                brain_family: "dwraon".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            },
            ScenarioVariant {
                variant_id: "variant_c".into(),
                brain_family: "assembly".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            },
            #[cfg(feature = "brain-ft")]
            ScenarioVariant {
                variant_id: "variant_d".into(),
                brain_family: "frankentorch".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            },
            #[cfg(feature = "neuro")]
            ScenarioVariant {
                variant_id: "variant_neuro".into(),
                brain_family: "neuro".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            },
        ];

        let expected_runs = variants.len() * cohort.seeds.len();

        let runner =
            MatchedSeedExperimentRunner::new("exp-matched", cohort, variants, 2, 2, &output_dir);

        let status = runner.execute_batch(&state_file).unwrap();
        assert_eq!(status.completed_runs, expected_runs);
        assert_eq!(status.failed_runs, 0);
        assert!(status.is_finished());
        assert_eq!(status.runs[0].brain_family, "mlp.baseline");
        assert_eq!(status.runs[1].brain_family, "dwraon.baseline");
        assert_eq!(status.runs[2].brain_family, "assembly.experimental");
        #[cfg(feature = "brain-ft")]
        assert_eq!(status.runs[3].brain_family, "ft");
        #[cfg(feature = "neuro")]
        assert_eq!(
            status
                .runs
                .iter()
                .find(|run| run.variant_id == "variant_neuro")
                .unwrap()
                .brain_family,
            "ml.neuroflow"
        );
        assert_ne!(
            status.runs[0].final_digest, status.runs[1].final_digest,
            "real distinct family registries must produce distinct full world digests"
        );
        assert_ne!(
            status.runs[0].final_digest, status.runs[2].final_digest,
            "real distinct family registries must produce distinct full world digests"
        );
        assert_eq!(
            status
                .runs
                .iter()
                .map(|run| run.final_digest.as_ref().unwrap())
                .collect::<BTreeSet<_>>()
                .len(),
            expected_runs,
            "every requested family must execute its own scientific state"
        );

        for run in &status.runs {
            let bundle_bytes = fs::read(output_dir.join(&run.run_id).join("bundle_manifest.json"))
                .expect("verified bundle manifest exists");
            let bundle: RunBundleV1 = serde_json::from_slice(&bundle_bytes).expect("bundle schema");
            let manifest: RunManifestV3 = serde_json::from_str(&bundle.manifest.manifest_json)
                .expect("embedded canonical manifest");
            assert_eq!(manifest.brain_roster.len(), 1);
            assert_eq!(manifest.brain_roster[0].kind, run.brain_family);
            assert_eq!(manifest.identity.requested_tick_budget, Some(2));
            assert_eq!(manifest.root_seed, 1001);
            let checkpoint_path = output_dir
                .join(&run.run_id)
                .join(FINAL_CHECKPOINT_ARTIFACT.0);
            let bytes = fs::read(&checkpoint_path).expect("portable final state");
            let standalone = runner.run_checkpoint_path(&run.run_id);
            assert_eq!(bytes, fs::read(&standalone).unwrap(), "exact sealed cut");
            let checkpoint = RunCheckpointV1::decode(&bytes, &checkpoint_path).unwrap();
            let plan = runner.validated_plan().unwrap();
            let (_, family) = &plan.variants[&run.variant_id];
            let restored =
                restore_run_owner(&checkpoint, *family, &run.run_id, &checkpoint_path).unwrap();
            assert_eq!(restored.world_tick().0, run.total_ticks);
            assert_eq!(
                Some(restored.scientific_digest_v1().unwrap().overall),
                run.final_digest
            );
            let offline = verify_portable_run_bundle(&output_dir.join(&run.run_id))
                .unwrap()
                .final_state
                .unwrap();
            assert_eq!(offline.tick, run.total_ticks);
            assert_eq!(offline.alive_agents, restored.world().agents().len());
            assert_eq!(Some(offline.scientific_digest), run.final_digest);
            fs::rename(
                &standalone,
                temp_dir
                    .path()
                    .join(format!("preserved-{}.checkpoint", run.run_id)),
            )
            .unwrap();
        }

        // A completed resume reopens and semantically verifies all bundles rather
        // than trusting status JSON, depending on the external continuation cuts,
        // or executing duplicate runs.
        let resumed_status = runner.execute_batch(&state_file).unwrap();
        assert_eq!(resumed_status.completed_runs, expected_runs);
        assert_eq!(resumed_status.failed_runs, 0);
        assert_eq!(resumed_status, status);

        let first_manifest_path = output_dir
            .join(&status.runs[0].run_id)
            .join("bundle_manifest.json");
        let mut incomplete: RunBundleV1 = serde_json::from_slice(
            &fs::read(&first_manifest_path).expect("bundle manifest remains readable"),
        )
        .expect("bundle manifest schema");
        incomplete
            .artifacts
            .retain(|entry| entry.relative_path != "evidence/run.json");
        fs::write(
            &first_manifest_path,
            serde_json::to_vec_pretty(&incomplete).expect("serialize incomplete manifest"),
        )
        .expect("write deliberate attribution omission");
        assert!(matches!(
            runner.execute_batch(&state_file),
            Err(ExperimentRunnerError::Bundle { .. })
        ));
    }

    #[test]
    fn arm_overrides_outside_knob_ranges_are_refused() {
        let mut overrides = BTreeMap::new();
        overrides.insert("world_width".to_owned(), serde_json::json!(16_777_216));
        let err = config_for_run(&overrides, 1).expect_err("out-of-range world width");
        assert!(err.contains("outside its accepted range"), "{err}");
        let default_width = ScriptBotsConfig::default().world_width;
        overrides.insert("world_width".to_owned(), serde_json::json!(default_width));
        let accepted = config_for_run(&overrides, 1);
        assert!(accepted.is_ok(), "{accepted:?}");
    }
}
