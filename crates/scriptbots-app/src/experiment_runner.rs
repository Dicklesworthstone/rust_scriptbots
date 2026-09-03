//! Deterministic matched-seed experiment runner and batch execution engine (bd-2z0.5.11).

use crate::{
    BuildProvenanceV0, RunIdentityV1, RunManifestV3, ScenarioIdentityV0, ThreadPolicyV0,
    seed_founding_population,
};
use scriptbots_brain::{
    AssemblyBrain, DwraonBrain, MlpBrain, assembly::AssemblyFamilyAdapter,
    dwraon::DwraonFamilyAdapter, mlp::MlpBrainFamily,
};
use scriptbots_core::{ScriptBotsConfig, WorldState, knob_range};
use scriptbots_runtime::{
    ApplicationState, HostCore, HostCoreOptions, HostLifecycle, HostSessionId, JournalState,
    ManualInstant, NullFrontend, PlaybackSnapshot, RunId,
};
use scriptbots_storage::{
    RunBundleV1, bundle::RunBundleVerificationLimits, bundle::verify_run_bundle_bounded,
    create_run_bundle_from_artifacts,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tracing::{error, info};

const STATUS_SCHEMA_VERSION: u16 = 2;
const MAX_STATUS_BYTES: u64 = 16 * 1024 * 1024;
const MAX_BUNDLE_ARTIFACTS: usize = 32;
const MAX_BUNDLE_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_BUNDLE_TOTAL_ARTIFACT_BYTES: u64 = 32 * 1024 * 1024;
const MAX_BATCH_RUNS: usize = 4_096;
const MAX_RUN_COMPONENT_BYTES: usize = 128;
const MAX_TICKS_PER_RUN: u64 = 10_000_000;
const MAX_CONCURRENT_RUNS: usize = 64;
const HOST_RECEIPT_DRIVE_LIMIT: usize = 16;

/// Typed failure from matched-seed planning, status recovery, execution, or bundle verification.
#[derive(Debug, Error)]
pub enum ExperimentRunnerError {
    /// A declarative experiment cannot produce an unambiguous bounded plan.
    #[error("invalid experiment plan: {reason}")]
    InvalidPlan { reason: String },
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
    /// A prior write stopped after creating the temporary status file.
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
    /// This first execution slice refuses to pretend that a running record can resume at tick zero.
    #[error(
        "run `{run_id}` stopped in `running` state at tick {completed_ticks}; composite HostCore/checkpoint/storage resume is not yet available, so automatic tick-zero replay is refused"
    )]
    ResumeCheckpointUnavailable {
        run_id: String,
        completed_ticks: u64,
    },
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
pub struct ScenarioVariant {
    pub variant_id: String,
    pub brain_family: String,
    /// Canonical validated arm. JSON values preserve integer-vs-float types
    /// until strict `ScriptBotsConfig` deserialization.
    pub config_overrides: BTreeMap<String, serde_json::Value>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ResolvedBrainFamily {
    Mlp,
    Dwraon,
    Assembly,
}

impl ResolvedBrainFamily {
    fn resolve(variant: &ScenarioVariant) -> Result<Self, ExperimentRunnerError> {
        match variant.brain_family.as_str() {
            "mlp" | "mlp.baseline" => Ok(Self::Mlp),
            "dwraon" | "dwraon.baseline" => Ok(Self::Dwraon),
            "assembly" | "assembly.experimental" => Ok(Self::Assembly),
            "neuroflow" | "ft" | "frankentorch" => {
                Err(ExperimentRunnerError::UnavailableBrainFamily {
                    variant_id: variant.variant_id.clone(),
                    brain_family: variant.brain_family.clone(),
                    reason: "this runner build has no canonical family installation for that feature; execution is refused instead of substituting another family".to_owned(),
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
            Self::Mlp => "mlp.baseline",
            Self::Dwraon => "dwraon.baseline",
            Self::Assembly => "assembly.experimental",
        }
    }

    fn register(self, world: &mut WorldState, run_id: &str) -> Result<u64, ExperimentRunnerError> {
        let result = match self {
            Self::Mlp => world
                .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new())),
            Self::Dwraon => world.register_brain_family(
                DwraonBrain::KIND.as_str(),
                Box::new(DwraonFamilyAdapter::default()),
            ),
            Self::Assembly => {
                let adapter = AssemblyFamilyAdapter::new().map_err(|source| {
                    ExperimentRunnerError::Construction {
                        run_id: run_id.to_owned(),
                        reason: format!("Assembly family construction failed: {source}"),
                    }
                })?;
                world.register_brain_family(AssemblyBrain::KIND.as_str(), Box::new(adapter))
            }
        };
        result.map_err(|source| ExperimentRunnerError::Construction {
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
struct RunEvidenceV1 {
    schema: String,
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

/// Deterministic matched-seed experiment runner.
pub struct MatchedSeedExperimentRunner {
    pub experiment_id: String,
    pub cohort: MatchedSeedCohort,
    pub variants: Vec<ScenarioVariant>,
    pub max_ticks: u64,
    pub max_concurrency: usize,
    pub output_dir: PathBuf,
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

    /// Loads existing status file if present, or initializes a new batch plan.
    ///
    /// # Errors
    ///
    /// Refuses malformed, stale, interrupted, or plan-incompatible state. A
    /// malformed status is never treated as permission to start again at tick zero.
    pub fn load_or_create_status(
        &self,
        state_file: &Path,
    ) -> Result<ExperimentBatchStatus, ExperimentRunnerError> {
        let plan = self.validated_plan()?;
        let _lease = ExperimentStatusWriterLease::acquire(state_file)?;
        self.load_or_create_status_locked(state_file, &plan)
    }

    fn load_or_create_status_locked(
        &self,
        state_file: &Path,
        plan: &ValidatedPlan,
    ) -> Result<ExperimentBatchStatus, ExperimentRunnerError> {
        let temporary = status_temporary_path(state_file)?;
        if temporary.exists() {
            return Err(ExperimentRunnerError::InterruptedStatusWrite {
                temporary,
                committed: state_file.to_path_buf(),
            });
        }

        if state_file.exists() {
            return self.load_existing_status(state_file, plan);
        }

        let status = ExperimentBatchStatus {
            schema_version: STATUS_SCHEMA_VERSION,
            generation: 0,
            plan_digest: plan.digest.clone(),
            experiment_id: self.experiment_id.clone(),
            total_runs: plan.runs.len(),
            completed_runs: 0,
            failed_runs: 0,
            runs: plan.runs.clone(),
        };
        self.write_status_atomically(state_file, &status, plan)?;
        Ok(status)
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
    fn bundle_run_id(run_id: &str) -> RunId {
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
        self.execute_validated_run(
            planned_variant,
            *planned_family,
            seed,
            &run_id,
            &plan.digest,
        )
    }

    /// Executes the full batch using deterministic bounded scheduling and durable status.
    ///
    /// Failed runs become terminal status records so the caller can inspect the
    /// complete partial-failure set. An interrupted `Running` record is instead a
    /// typed refusal until composite checkpoint resume exists.
    ///
    /// # Errors
    ///
    /// Returns a typed plan, recovery, durable-status, or bundle-verification failure.
    pub fn execute_batch(
        &self,
        state_file: &Path,
    ) -> Result<ExperimentBatchStatus, ExperimentRunnerError> {
        let plan = self.validated_plan()?;
        let _lease = ExperimentStatusWriterLease::acquire(state_file)?;
        let mut status = self.load_or_create_status_locked(state_file, &plan)?;

        for record in &status.runs {
            match record.state {
                RunState::Running => {
                    return Err(ExperimentRunnerError::ResumeCheckpointUnavailable {
                        run_id: record.run_id.clone(),
                        completed_ticks: record.total_ticks,
                    });
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
                    self.verify_completed_bundle(record, variant, *family, &plan.digest)?;
                }
                RunState::Pending | RunState::Failed => {}
            }
        }

        let pending = status
            .runs
            .iter()
            .enumerate()
            .filter_map(|(index, record)| (record.state == RunState::Pending).then_some(index))
            .collect::<Vec<_>>();

        for chunk in pending.chunks(self.max_concurrency) {
            for &index in chunk {
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
                    Ok(completed) => {
                        info!(
                            experiment_id = %self.experiment_id,
                            run_id = %completed.run_id,
                            variant_id = %completed.variant_id,
                            brain_family = %completed.brain_family,
                            seed = completed.seed,
                            tick = completed.total_ticks,
                            digest = completed.final_digest.as_deref().unwrap_or(""),
                            bundle_path = completed.bundle_path.as_deref().unwrap_or(""),
                            scheduler_slot = index,
                            "matched-seed run completed and bundle reopened"
                        );
                        status.runs[index] = completed;
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
            let config_bytes = serde_json::to_vec(&normalized_config).map_err(|source| {
                ExperimentRunnerError::Json {
                    operation: "serialize normalized scenario arm",
                    source,
                }
            })?;
            let semantic_key = (family, blake3::hash(&config_bytes).to_hex().to_string());
            if !semantic_arms.insert(semantic_key) {
                return Err(ExperimentRunnerError::InvalidPlan {
                    reason: format!(
                        "variant `{}` duplicates an existing `{}` arm after canonical config normalization",
                        variant.variant_id,
                        family.canonical_name()
                    ),
                });
            }
            variants.insert(variant.variant_id.clone(), (variant.clone(), family));
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
        let status = serde_json::from_slice::<ExperimentBatchStatus>(&bytes).map_err(|source| {
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
        let mut bytes =
            serde_json::to_vec_pretty(status).map_err(|source| ExperimentRunnerError::Json {
                operation: "serialize experiment status",
                source,
            })?;
        bytes.push(b'\n');
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

    fn execute_validated_run(
        &self,
        variant: &ScenarioVariant,
        family: ResolvedBrainFamily,
        seed: u64,
        run_id: &str,
        plan_digest: &str,
    ) -> Result<RunRecord, ExperimentRunnerError> {
        let config = config_for_run(&variant.config_overrides, seed).map_err(|reason| {
            ExperimentRunnerError::Construction {
                run_id: run_id.to_owned(),
                reason,
            }
        })?;
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
        let manifest_bytes =
            manifest
                .canonical_json_bytes()
                .map_err(|source| ExperimentRunnerError::Json {
                    operation: "serialize canonical run manifest",
                    source,
                })?;
        let manifest_digest = blake3::hash(&manifest_bytes).to_hex().to_string();
        let manifest_record =
            manifest
                .to_storage_record()
                .map_err(|source| ExperimentRunnerError::Construction {
                    run_id: run_id.to_owned(),
                    reason: format!("storage manifest projection failed: {source}"),
                })?;

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
        let mut frontend = NullFrontend::new(core.local_port(), client_namespace);
        let mut next_nanos = 0_u64;

        for expected_tick in 1..=self.max_ticks {
            let submitted = frontend
                .step()
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
                &mut next_nanos,
            )?;
        }

        if core.world_tick().0 != self.max_ticks {
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
        let alive_agents = core.world().agents().len();

        let submitted = frontend
            .shutdown()
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
            self.max_ticks,
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

        let evidence = RunEvidenceV1 {
            schema: "scriptbots.experiment-run-evidence.v1".to_owned(),
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
        let summary_csv = format!(
            "tick,alive_agents,seed,brain_family,final_digest\n{},{},{},{},{}\n",
            self.max_ticks,
            alive_agents,
            seed,
            family.canonical_name(),
            digest_hex
        );
        let bundle_dir = self.output_dir.join(run_id);
        if bundle_dir.exists() {
            return Err(ExperimentRunnerError::BundleAlreadyExists {
                run_id: run_id.to_owned(),
                path: bundle_dir,
            });
        }
        create_run_bundle_from_artifacts(
            &bundle_dir,
            manifest_record,
            self.max_ticks,
            &[
                (
                    "exports/summary.csv",
                    "experiment-summary",
                    summary_csv.as_bytes(),
                ),
                (
                    "evidence/run.json",
                    "experiment-run-evidence",
                    &evidence_bytes,
                ),
            ],
        )
        .map_err(|source| ExperimentRunnerError::Bundle {
            run_id: run_id.to_owned(),
            reason: format!("assembly failed: {source}"),
        })?;

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
        self.verify_completed_bundle(&completed, variant, family, plan_digest)?;
        Ok(completed)
    }

    fn verify_completed_bundle(
        &self,
        record: &RunRecord,
        variant: &ScenarioVariant,
        family: ResolvedBrainFamily,
        plan_digest: &str,
    ) -> Result<(), ExperimentRunnerError> {
        let bundle_dir = self.output_dir.join(&record.run_id);
        verify_run_bundle_bounded(
            &bundle_dir,
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
        for (required_path, required_type) in [
            ("exports/summary.csv", "experiment-summary"),
            ("evidence/run.json", "experiment-run-evidence"),
        ] {
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
        let evidence: RunEvidenceV1 =
            serde_json::from_slice(&evidence_bytes).map_err(|source| {
                ExperimentRunnerError::Bundle {
                    run_id: record.run_id.clone(),
                    reason: format!("run evidence JSON is invalid: {source}"),
                }
            })?;
        let expected_evidence = RunEvidenceV1 {
            schema: "scriptbots.experiment-run-evidence.v1".to_owned(),
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
        Ok(())
    }
}

fn validate_path_component(label: &str, value: &str) -> Result<(), ExperimentRunnerError> {
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
    let file = File::open(path).map_err(|source| ExperimentRunnerError::Io {
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

fn drive_command_to_volatile_commit(
    run_id: &str,
    operation: &'static str,
    frontend: &mut NullFrontend<scriptbots_runtime::LocalHostPort>,
    core: &mut HostCore,
    command_id: scriptbots_runtime::CommandId,
    expected_tick: u64,
    next_nanos: &mut u64,
) -> Result<(), ExperimentRunnerError> {
    let mut last_status = None;
    for _ in 0..HOST_RECEIPT_DRIVE_LIMIT {
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
            "command did not reach Applied + CommittedVolatile within {HOST_RECEIPT_DRIVE_LIMIT} drives; last status: {last_status:?}"
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

fn config_for_run(
    overrides: &BTreeMap<String, serde_json::Value>,
    seed: u64,
) -> Result<ScriptBotsConfig, String> {
    let mut encoded = serde_json::to_value(ScriptBotsConfig::default())
        .map_err(|error| format!("default config serialization failed: {error}"))?;
    let root = encoded
        .as_object_mut()
        .ok_or_else(|| "default config did not serialize as an object".to_owned())?;

    for (path, value) in overrides {
        if knob_range(path).is_none() {
            return Err(format!(
                "validated arm contains unknown or unsupported knob `{path}`"
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

fn insert_dotted_value(
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
            },
            ScenarioVariant {
                variant_id: "dwraon_base".into(),
                brain_family: "dwraon".into(),
                config_overrides: BTreeMap::new(),
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
    fn single_run_refuses_an_arm_changed_after_plan_validation() {
        let planned_variant = ScenarioVariant {
            variant_id: "mlp_base".into(),
            brain_family: "mlp".into(),
            config_overrides: BTreeMap::new(),
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
            },
            ScenarioVariant {
                variant_id: "variant_b".into(),
                brain_family: "dwraon".into(),
                config_overrides: BTreeMap::new(),
            },
            ScenarioVariant {
                variant_id: "variant_c".into(),
                brain_family: "assembly".into(),
                config_overrides: BTreeMap::new(),
            },
        ];

        let runner =
            MatchedSeedExperimentRunner::new("exp-matched", cohort, variants, 2, 2, &output_dir);

        let status = runner.execute_batch(&state_file).unwrap();
        assert_eq!(status.completed_runs, 3);
        assert_eq!(status.failed_runs, 0);
        assert!(status.is_finished());
        assert_eq!(status.runs[0].brain_family, "mlp.baseline");
        assert_eq!(status.runs[1].brain_family, "dwraon.baseline");
        assert_eq!(status.runs[2].brain_family, "assembly.experimental");
        assert_ne!(
            status.runs[0].final_digest, status.runs[1].final_digest,
            "real distinct family registries must produce distinct full world digests"
        );
        assert_ne!(
            status.runs[0].final_digest, status.runs[2].final_digest,
            "real distinct family registries must produce distinct full world digests"
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
        }

        // A completed resume reopens and semantically verifies all bundles rather
        // than trusting the status JSON or executing duplicate runs.
        let resumed_status = runner.execute_batch(&state_file).unwrap();
        assert_eq!(resumed_status.completed_runs, 3);
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
}
