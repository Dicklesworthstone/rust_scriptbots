//! Renderer-independent consumers of application-owned Lab services.
//!
//! A single bounded worker performs application reads and requests. Frontends
//! render these observations and apply returned checkpoint or event frames only
//! to their presentation; this module never owns a world, database or experiment runner.

use crate::{HostSessionId, RenderSnapshot};
use std::{
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, TryRecvError},
    },
    time::{Duration, Instant},
};

/// Maximum rows in an application catalog page.
pub const LAB_PAGE_SIZE: usize = 12;
/// Maximum retained variants, independently of seed/run cardinality.
pub const LAB_MAX_VARIANTS: usize = 64;
/// Variant rows presented together by native frontends.
pub const VARIANTS_PER_PAGE: usize = 8;
/// Hard bound on compiled family choices retained by the native composer.
pub const LAB_MAX_EXPERIMENT_FAMILIES: usize = 8;
/// Hard bound on the seed vector before any capability-driven allocation.
pub const LAB_MAX_EXPERIMENT_SEEDS: usize = 128;
// Run IDs append "-challenger-seed" and up to twenty decimal seed digits to
// the experiment identity. Reserve that suffix within the runner's 128-byte bound.
const MAX_EXPERIMENT_REQUEST_ID_BYTES: usize = 92;
const REFRESH_INTERVAL: Duration = Duration::from_secs(2);
// Application verification can read FrankenSQLite on this worker. Match the
// control worker's stack, including its existing reader recursion requirements.
const LAB_WORKER_STACK_BYTES: usize = 32 * 1024 * 1024;
static CHECKPOINT_CLIENT_NAMESPACE: OnceLock<u128> = OnceLock::new();
static NEXT_CHECKPOINT_REQUEST: AtomicU64 = AtomicU64::new(1);

/// Final-state observations supported by the experiment runner and native Lab.
///
/// These describe one completed boundary, not time averages or cumulative
/// lifecycle counts. Population means and maxima require a living population;
/// the application service refuses an unavailable endpoint instead of zero-filling it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LabMetric {
    /// Number of living agents.
    #[default]
    AliveAgents,
    /// Sum of current live-agent energy reserves.
    TotalEnergy,
    /// Mean current live-agent energy reserve.
    AverageEnergy,
    /// Mean current live-agent health.
    AverageHealth,
    /// Mean live-agent age counter, advanced at the configured aging cadence.
    MeanAge,
    /// Largest live-agent age counter, not elapsed simulation ticks.
    MaxAge,
    /// Largest living lineage generation.
    MaxGeneration,
    /// Sum of current food-grid cell energy values.
    TotalFood,
}

impl LabMetric {
    /// Canonical endpoint order shared with runner summaries and API schemas.
    pub const ALL: [Self; 8] = [
        Self::AliveAgents,
        Self::TotalEnergy,
        Self::AverageEnergy,
        Self::AverageHealth,
        Self::MeanAge,
        Self::MaxAge,
        Self::MaxGeneration,
        Self::TotalFood,
    ];

    /// Exact wire key used by the runner and application comparison service.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::AliveAgents => "alive_agents",
            Self::TotalEnergy => "total_energy",
            Self::AverageEnergy => "average_energy",
            Self::AverageHealth => "average_health",
            Self::MeanAge => "mean_age",
            Self::MaxAge => "max_age",
            Self::MaxGeneration => "max_generation",
            Self::TotalFood => "total_food",
        }
    }

    /// Human-readable observation named by this endpoint.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::AliveAgents => "Population",
            Self::TotalEnergy => "Total live-agent energy",
            Self::AverageEnergy => "Mean live-agent energy",
            Self::AverageHealth => "Mean live-agent health",
            Self::MeanAge => "Mean age counter",
            Self::MaxAge => "Maximum age counter",
            Self::MaxGeneration => "Maximum living generation",
            Self::TotalFood => "Total food",
        }
    }

    /// Scientific units of the observed value and its unstandardized difference.
    #[must_use]
    pub const fn unit(self) -> &'static str {
        match self {
            Self::AliveAgents => "agents",
            Self::TotalEnergy => "energy units",
            Self::AverageEnergy => "energy units per agent",
            Self::AverageHealth => "health units",
            Self::MeanAge | Self::MaxAge => "age increments",
            Self::MaxGeneration => "generations",
            Self::TotalFood => "food energy units",
        }
    }

    /// Canonical wire vocabulary, derived from the same catalog as UI choices.
    #[must_use]
    pub const fn keys() -> [&'static str; Self::ALL.len()] {
        let mut keys = [""; Self::ALL.len()];
        let mut index = 0;
        while index < Self::ALL.len() {
            keys[index] = Self::ALL[index].key();
            index += 1;
        }
        keys
    }
}

/// Executable choices and admission limits supplied by the application runner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabExperimentCapabilities {
    /// Canonical installed families that support experiment execution and continuation.
    pub families: Vec<String>,
    /// Maximum independent matched seed pairs admitted by the runner.
    pub max_seed_count: usize,
    /// Maximum scientific transitions in each arm's run.
    pub max_ticks_per_run: u64,
    /// Maximum total transitions across both complete seed cohorts.
    pub max_total_ticks: u64,
    /// Actual fresh-world configuration description supplied by the application.
    pub launch_description: String,
}

impl LabExperimentCapabilities {
    /// Validate the bounded native capability envelope before using its limits.
    ///
    /// # Errors
    ///
    /// Rejects excessive or duplicate family choices, invalid limits, or an
    /// empty, overlong or control-character-containing launch description.
    pub fn validate(&self) -> Result<(), String> {
        if self.families.len() > LAB_MAX_EXPERIMENT_FAMILIES
            || self.families.iter().any(|family| {
                family.is_empty()
                    || family.len() > 128
                    || !family.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')
                    })
            })
            || self
                .families
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.families.len()
        {
            return Err("Experiment family capabilities are invalid or exceed their bounds.".into());
        }
        if self.max_seed_count == 0
            || self.max_seed_count > LAB_MAX_EXPERIMENT_SEEDS
            || self.max_ticks_per_run == 0
            || self.max_total_ticks == 0
        {
            return Err("Experiment execution capabilities contain invalid limits.".into());
        }
        if self.launch_description.trim().is_empty()
            || self.launch_description.len() > 2048
            || self.launch_description.chars().any(char::is_control)
        {
            return Err("Experiment launch description is missing or exceeds its bounds.".into());
        }
        Ok(())
    }
}

/// One explicit two-arm family comparison in independent fresh runner worlds.
///
/// The two arms use the same checked consecutive seed cohort and horizon.
/// Scientific execution is sequential; this draft never inherits the live owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabExperimentDraft {
    /// Reference family's canonical installed identity.
    pub baseline_family: String,
    /// Distinct comparison family's canonical installed identity.
    pub challenger_family: String,
    /// First disclosed seed; later seeds increment by one without wrapping.
    pub first_seed: u64,
    /// Independent matched seed pairs, not the total number of runs.
    pub seed_count: usize,
    /// Exact scientific transitions requested for every run.
    pub ticks_per_run: u64,
}

impl Default for LabExperimentDraft {
    fn default() -> Self {
        Self {
            baseline_family: String::new(),
            challenger_family: String::new(),
            first_seed: 42,
            seed_count: 3,
            ticks_per_run: 100,
        }
    }
}

impl LabExperimentDraft {
    /// Materialize the exact bounded cohort without wrapping its last seed.
    ///
    /// # Errors
    ///
    /// Rejects zero or excessive seed counts and a cohort that exceeds u64.
    pub fn seeds(&self) -> Result<Vec<u64>, String> {
        if self.seed_count == 0 || self.seed_count > LAB_MAX_EXPERIMENT_SEEDS {
            return Err(format!(
                "Seed count must be in 1..={LAB_MAX_EXPERIMENT_SEEDS}."
            ));
        }
        let span = u64::try_from(self.seed_count - 1)
            .map_err(|_| "Seed count cannot be represented.".to_owned())?;
        let last_seed = self
            .first_seed
            .checked_add(span)
            .ok_or_else(|| "The seed cohort would overflow u64; choose an earlier first seed.".to_owned())?;
        Ok((self.first_seed..=last_seed).collect())
    }

    /// Check the complete two-arm request against the observed application limits.
    ///
    /// # Errors
    ///
    /// Rejects unavailable or identical families, unsupported seed/horizon
    /// choices, seed overflow, or excessive/overflowing total scientific work.
    pub fn validate(&self, capabilities: &LabExperimentCapabilities) -> Result<(), String> {
        capabilities.validate()?;
        if self.baseline_family == self.challenger_family
            || !capabilities.families.contains(&self.baseline_family)
            || !capabilities.families.contains(&self.challenger_family)
        {
            return Err("Choose two distinct executable brain families.".into());
        }
        if self.seed_count == 0 || self.seed_count > capabilities.max_seed_count {
            return Err(format!(
                "Seed count must be in 1..={}.",
                capabilities.max_seed_count,
            ));
        }
        if self.ticks_per_run == 0 || self.ticks_per_run > capabilities.max_ticks_per_run {
            return Err(format!(
                "Ticks per run must be in 1..={}.",
                capabilities.max_ticks_per_run,
            ));
        }
        let total_ticks = u64::try_from(self.seed_count)
            .ok()
            .and_then(|count| count.checked_mul(2))
            .and_then(|runs| runs.checked_mul(self.ticks_per_run))
            .ok_or_else(|| "The two-arm scientific work count overflows u64.".to_owned())?;
        if total_ticks > capabilities.max_total_ticks {
            return Err(format!(
                "The two-arm cohort needs {total_ticks} transitions, exceeding {}.",
                capabilities.max_total_ticks,
            ));
        }
        self.seeds().map(|_| ())
    }
}

/// Exact request retained until creation is observed or explicitly abandoned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabExperimentCreation {
    /// Explicit experiment identity and idempotency key.
    pub request_id: String,
    /// Application source from which this creation was requested.
    pub session_id: HostSessionId,
    /// Frozen scientific inputs, reused unchanged on every retry.
    pub draft: LabExperimentDraft,
    /// Latest observed admission result.
    pub state: LabExperimentCreationState,
}

/// A creation request is not a running or completed scientific experiment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LabExperimentCreationState {
    /// The sole worker is awaiting the application response.
    Pending,
    /// Admission is unconfirmed; a retry must preserve every original input.
    Retryable {
        /// Bounded error or invalidation explanation.
        reason: String,
    },
    /// The application's bounded batch identity and shape have been checked.
    Created {
        /// Verified retained experiment identity.
        experiment_id: String,
    },
}

/// One canonical narrative event within a publishing owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabEventIdentity {
    /// Owner whose bounded narrative ring supplied the event.
    pub session_id: HostSessionId,
    /// One-based absolute ordinal, including events evicted before the ring.
    pub ordinal: u64,
    /// Exact historical tick named by the canonical event.
    pub tick: u64,
}

/// A reconstructed event observation with its actual saved source.
pub struct LabEventReview {
    /// Event requested by the user and checked against the owner's ring.
    pub event: LabEventIdentity,
    /// Complete canonical evidence captured before reconstruction.
    pub record: scriptbots_core::narrative::EventRecord,
    /// Retained checkpoint from which the application reconstructed the target.
    pub checkpoint_id: String,
    /// Actual tick of that source checkpoint.
    pub base_tick: u64,
    /// Historical presentation, never a replacement for the live owner.
    pub snapshot: Arc<RenderSnapshot>,
}

struct SelectedEvent {
    identity: LabEventIdentity,
    record: scriptbots_core::narrative::EventRecord,
}

/// A request to application data services, never a direct world operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LabWorkflowRequest {
    /// Read compiled experiment families and actual fresh-world admission limits.
    ExperimentCapabilities,
    /// Explicitly create, or retry, one immutable two-arm experiment request.
    CreateExperiment {
        /// Explicit experiment identity reused as the application idempotency key.
        request_id: String,
        /// Application source observed when the draft was submitted.
        session_id: HostSessionId,
        /// Complete frozen scientific choices for this creation identity.
        draft: LabExperimentDraft,
    },
    /// Read one experiment catalog page.
    ListExperiments {
        /// Application page cursor.
        cursor: Option<String>,
    },
    /// Read one complete bounded batch projection.
    GetExperiment {
        /// Stable experiment identity.
        experiment_id: String,
    },
    /// Request cancellation through the existing experiment runner.
    CancelExperiment {
        /// Stable experiment identity.
        experiment_id: String,
    },
    /// Ask the application to validate and resume a stopped batch.
    ResumeExperiment {
        /// Stable experiment identity.
        experiment_id: String,
    },
    /// Compare verified observations from two distinct arms.
    Compare {
        /// Stable experiment identity.
        experiment_id: String,
        /// Reference variant identity.
        baseline_variant: String,
        /// Variant whose observations have the reference subtracted from them.
        challenger_variant: String,
        /// Final-state observation to compare across the matched seed pairs.
        metric: LabMetric,
    },
    /// Read one checkpoint catalog page.
    ListCheckpoints {
        /// Application page cursor.
        cursor: Option<String>,
    },
    /// Save a live owner cut through the existing idempotent checkpoint service.
    CreateCheckpoint {
        /// Stable identity reused after any unconfirmed outcome.
        request_id: String,
        /// Actual owner session observed when the user requested the save.
        session_id: HostSessionId,
    },
    /// Reconstruct a saved cut without changing the live owner.
    ReviewCheckpoint {
        /// Stable retained checkpoint identity.
        checkpoint_id: String,
    },
    /// Reconstruct a retained canonical event through the existing replay service.
    ReviewEvent {
        /// Observed owner, absolute event ordinal and exact historical tick.
        event: LabEventIdentity,
    },
}

/// Bounded experiment list row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabExperimentSummary {
    /// Stable experiment identity.
    pub experiment_id: String,
    /// Observed runner state, not a command acknowledgement.
    pub status: String,
    /// Planned run count.
    pub total_runs: usize,
    /// Runs with terminal completed evidence.
    pub completed_runs: usize,
    /// Runs with terminal failure evidence.
    pub failed_runs: usize,
}

/// A bounded experiment page and its observed execution capacity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabExperimentPage {
    /// Visible catalog rows.
    pub items: Vec<LabExperimentSummary>,
    /// Total catalog rows.
    pub total: usize,
    /// Offset of the first visible row.
    pub offset: usize,
    /// Cursor for the next page, if any.
    pub next_cursor: Option<String>,
    /// Reserved slots; this is not simultaneous science activity.
    pub reserved_run_slots: usize,
    /// Configured slot ceiling.
    pub max_run_slots: usize,
}

/// Run-state counts for one experimental arm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabVariantProgress {
    /// Stable arm identity.
    pub variant_id: String,
    /// Actual configured brain family.
    pub brain_family: String,
    /// Planned independent seed runs.
    pub total_runs: usize,
    /// Completed runs.
    pub completed_runs: usize,
    /// Failed runs.
    pub failed_runs: usize,
    /// Runs currently marked running.
    pub running_runs: usize,
}

/// Complete bounded batch projection with source identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabExperiment {
    /// Batch-level progress.
    pub summary: LabExperimentSummary,
    /// Digest of the admitted source-aware plan.
    pub plan_digest: String,
    /// Observed status generation.
    pub generation: u64,
    /// Every arm, independently of the number of seed/run rows.
    pub variants: Vec<LabVariantProgress>,
    /// Batch-level verification or execution failure.
    pub error_reason: Option<String>,
}

/// Source-bound paired final-state endpoint comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct LabComparison {
    /// Stable experiment identity.
    pub experiment_id: String,
    /// Reference arm.
    pub baseline_variant: String,
    /// Arm whose observations are compared with the reference.
    pub challenger_variant: String,
    /// Verified final-state observation, including its scientific units.
    pub metric: LabMetric,
    /// Common observed endpoint.
    pub endpoint_tick: u64,
    /// Ordered unique independent seed pairs.
    pub matched_seeds: Vec<u64>,
    /// Challenger minus baseline mean, in the selected metric's units.
    pub mean_difference: f64,
    /// Optional confidence interval.
    pub ci_95: Option<(f64, f64)>,
    /// Optional paired Cohen's `d_z`.
    pub standardized_effect: Option<f64>,
    /// Optional raw inferential p value.
    pub p_value: Option<f64>,
    /// Optional multiplicity-adjusted p value.
    pub adjusted_p_value: Option<f64>,
    /// Whether inference is unavailable for this cohort.
    pub descriptive_only: bool,
    /// Explanation of insufficient inference support.
    pub underpowered_reason: Option<String>,
    /// Explanation of an undefined standardized effect.
    pub undefined_effect_reason: Option<String>,
    /// Source plan digest used by the comparison.
    pub plan_digest: String,
    /// Exact source status generation.
    pub status_generation: u64,
}

/// Retained checkpoint catalog row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabCheckpoint {
    /// Stable checkpoint identity.
    pub checkpoint_id: String,
    /// Scientific tick of the saved cut.
    pub tick: u64,
    /// Saved living-agent count.
    pub agent_count: usize,
    /// Encoded artifact size.
    pub byte_size: u64,
    /// Scientific state digest, distinct from an artifact checksum.
    pub world_digest: String,
}

/// A bounded checkpoint page and actual retention usage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabCheckpointPage {
    /// Visible saved cuts.
    pub items: Vec<LabCheckpoint>,
    /// Total retained cuts.
    pub total: usize,
    /// Offset of the first row.
    pub offset: usize,
    /// Cursor for the next page.
    pub next_cursor: Option<String>,
    /// Actual retained file bytes.
    pub retained_bytes: u64,
    /// Configured retention ceiling.
    pub max_bytes: u64,
}

/// Latest checkpoint save, including the identity needed for an exact retry.
///
/// Only one record is retained. A deliberate save of a newly attached source
/// replaces an older record after its worker has drained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabCheckpointSave {
    /// Stable application idempotency identity.
    pub request_id: String,
    /// Owner session to which the request belongs.
    pub session_id: HostSessionId,
    /// Observed result, distinct from merely submitting a request.
    pub state: LabCheckpointSaveState,
}

/// Bounded progress for the most recent checkpoint save.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LabCheckpointSaveState {
    /// A worker is awaiting the service's verified receipt.
    Pending,
    /// The outcome is unconfirmed; retry the same identity on the same source.
    Retryable {
        /// Bounded refusal, disconnection or invalidation explanation.
        reason: String,
    },
    /// Exact saved metadata returned by the existing checkpoint service.
    Saved(LabCheckpoint),
}

/// Application evidence returned by the shared native worker.
pub enum LabWorkflowResponse {
    /// Bounded compiled-family and fresh-world capabilities.
    ExperimentCapabilities(LabExperimentCapabilities),
    /// Source-bound creation response after application admission.
    ExperimentCreated {
        /// Explicit experiment identity and retry key.
        request_id: String,
        /// Application source checked by the adapter.
        session_id: HostSessionId,
        /// Actual bounded batch observation, not a completed-science claim.
        experiment: LabExperiment,
    },
    /// Experiment catalog page.
    Experiments(LabExperimentPage),
    /// Batch details after a read or mutation request.
    Experiment(LabExperiment),
    /// Verified endpoint comparison.
    Comparison(LabComparison),
    /// Retained checkpoint catalog page.
    Checkpoints(LabCheckpointPage),
    /// Verified service receipt for one source-bound save request.
    CheckpointCreated {
        /// Stable request identity, unchanged by retry.
        request_id: String,
        /// Owner session checked by the application provider.
        session_id: HostSessionId,
        /// Authoritative saved cut; it may precede the current live tick.
        checkpoint: LabCheckpoint,
    },
    /// Reconstructed immutable frame, never a live-owner replacement.
    CheckpointFrame {
        /// Requested checkpoint identity.
        checkpoint_id: String,
        /// Reconstructed scientific presentation.
        snapshot: Arc<RenderSnapshot>,
    },
    /// Verified event target and its retained checkpoint provenance.
    EventFrame(LabEventReview),
}

/// Application adapter invoked on the model's bounded background worker.
pub type LabWorkflowSubmit =
    Arc<dyn Fn(LabWorkflowRequest) -> Result<LabWorkflowResponse, String> + Send + Sync>;

/// Active native panel; inactive panels do not schedule refreshes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LabPanel {
    /// No Lab panel is visible.
    #[default]
    Inactive,
    /// Experiment catalog, progress and comparisons.
    Experiments,
    /// Saved scientific cuts.
    Checkpoints,
    /// Canonical events read from immutable live publications.
    Narrative,
}

/// Frontend intent reduced by the shared model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LabAction {
    /// Open the fresh-world experiment composer and refresh its capabilities.
    NewExperiment,
    /// Leave the composer without cancelling any submitted creation.
    BackToExperiments,
    /// Select an offered reference family in the editable draft.
    DraftBaseline(String),
    /// Select an offered comparison family in the editable draft.
    DraftChallenger(String),
    /// Set the exact first seed without silently changing the cohort size.
    FirstSeed(u64),
    /// Set the exact number of independent matched seed pairs.
    SeedCount(usize),
    /// Set the exact scientific horizon for each run.
    TicksPerRun(u64),
    /// Explicitly submit or retry the frozen two-arm creation.
    CreateExperiment,
    /// Forget an unconfirmed idle request; its admitted experiment may still exist.
    AbandonExperimentRequest,
    /// Refresh visible application evidence.
    Refresh,
    /// Move to another experiment page.
    ExperimentsPage(Option<String>),
    /// Inspect a catalog experiment.
    SelectExperiment(String),
    /// Return to the experiment catalog.
    AllExperiments,
    /// Choose the reference arm.
    Baseline(String),
    /// Choose the comparison arm.
    Challenger(String),
    /// Choose the final-state observation used by the next comparison.
    Metric(LabMetric),
    /// Choose a bounded variant-page offset.
    VariantsPage(usize),
    /// Request a verified comparison.
    Compare,
    /// Request cancellation of the selected batch.
    CancelExperiment,
    /// Request validated resume of the selected batch.
    ResumeExperiment,
    /// Move to another checkpoint catalog page.
    CheckpointsPage(Option<String>),
    /// Select a retained cut.
    SelectCheckpoint(String),
    /// Save the live source, or retry its latest unconfirmed save identity.
    CreateCheckpoint,
    /// Request a presentation-only reconstruction.
    ReviewCheckpoint,
    /// Review exactly the event that was selected when this action was rendered.
    ReviewEvent(LabEventIdentity),
    /// Discard a pending result, retaining its occupied worker slot.
    CancelRequest,
    /// Invalidate pending review; the frontend restores its own live view.
    ReturnLive,
}

struct LabJob {
    request: LabWorkflowRequest,
    generation: u64,
    context: u64,
    session: HostSessionId,
    result: Receiver<Result<LabWorkflowResponse, String>>,
}

/// One application-service worker and bounded, source-validated UI observations.
///
/// Frontends read the public projections and route intents through dispatch.
/// Changing session, selection or tab invalidates a result without releasing its
/// worker slot. A slow cancelled request therefore cannot spawn parallel reads.
#[derive(Default)]
pub struct LabWorkflows {
    /// Optional application data-service adapter.
    pub submit: Option<LabWorkflowSubmit>,
    job: Option<LabJob>,
    generation: u64,
    context: u64,
    session: Option<HostSessionId>,
    panel: LabPanel,
    /// Whether the fresh-world composer replaces the experiment catalog.
    pub composing_experiment: bool,
    /// Last bounded capabilities observed from this application source.
    pub experiment_capabilities: Option<LabExperimentCapabilities>,
    /// Editable choices, distinct from a frozen pending creation request.
    pub experiment_draft: LabExperimentDraft,
    /// One immutable creation request retained through invalidation and retry.
    pub latest_experiment_creation: Option<LabExperimentCreation>,
    /// Last accepted experiment page.
    pub experiments: Option<LabExperimentPage>,
    experiment_cursor: Option<String>,
    /// Selected experiment identity.
    pub selected_experiment: Option<String>,
    /// Last accepted complete detail projection.
    pub batch: Option<LabExperiment>,
    /// Selected reference arm.
    pub baseline: Option<String>,
    /// Selected comparison arm.
    pub challenger: Option<String>,
    /// Selected final-state endpoint, independent of the chosen arms.
    pub metric: LabMetric,
    /// First visible variant row.
    pub variant_offset: usize,
    /// Last accepted comparison for the current plan and status generation.
    pub comparison: Option<LabComparison>,
    /// Last accepted checkpoint page.
    pub checkpoints: Option<LabCheckpointPage>,
    checkpoint_cursor: Option<String>,
    /// Selected retained cut.
    pub selected_checkpoint: Option<String>,
    /// One bounded save record; an old-source record is informational only.
    pub latest_checkpoint_save: Option<LabCheckpointSave>,
    selected_event: Option<SelectedEvent>,
    event_review: Option<LabEventReview>,
    /// Bounded user-facing operation state or failure.
    pub notice: Option<String>,
    last_refresh: Option<Instant>,
}

fn bounded_message(message: &str) -> String {
    message.chars().take(2048).collect()
}

fn identity_is_bounded(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}

/// Whether a save identity can cross the native application boundary.
#[must_use]
pub fn checkpoint_request_identity_is_valid(request_id: &str) -> bool {
    identity_is_bounded(request_id)
}

/// Whether an explicit experiment identity leaves room for the longest run suffix.
#[must_use]
pub fn experiment_request_identity_is_valid(request_id: &str) -> bool {
    !request_id.is_empty()
        && request_id.len() <= MAX_EXPERIMENT_REQUEST_ID_BYTES
        && request_id != "."
        && !request_id.contains("..")
        && request_id.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')
        })
}

fn reserve_checkpoint_sequence(counter: &AtomicU64) -> Option<u64> {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let next = current.checked_add(1)?;
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(reserved) => return Some(reserved),
            Err(observed) => current = observed,
        }
    }
}

fn next_checkpoint_request(session: HostSessionId) -> Option<String> {
    let sequence = reserve_checkpoint_sequence(&NEXT_CHECKPOINT_REQUEST)?;
    // This client namespace uses independent thread entropy, never the world's
    // scientific RNG. The checked process-wide sequence prevents reuse across
    // frontend instances and source resets even if they share a namespace.
    let namespace = CHECKPOINT_CLIENT_NAMESPACE.get_or_init(rand::random::<u128>);
    Some(format!(
        "native-lab-{:016x}-{namespace:032x}-{sequence:016x}",
        session.get(),
    ))
}

fn checkpoint_receipt_is_bounded(checkpoint: &LabCheckpoint) -> bool {
    identity_is_bounded(&checkpoint.checkpoint_id)
        && checkpoint.byte_size > 0
        && checkpoint.byte_size <= scriptbots_core::MAX_WORLD_CHECKPOINT_BYTES as u64
        && u64::try_from(checkpoint.agent_count).is_ok_and(|count| count <= checkpoint.byte_size)
        // WorldDigestV1 is FNV-1a64; artifact BLAKE3/SHA-256 checksums are
        // separate fields and must never stand in for this scientific digest.
        && checkpoint.world_digest.len() == std::mem::size_of::<u64>() * 2
        && checkpoint
            .world_digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn page_is_bounded<T>(items: &[T], offset: usize, total: usize, next: Option<&str>) -> bool {
    items.len() <= LAB_PAGE_SIZE
        && offset <= total
        && items.len() <= total.saturating_sub(offset)
        && next.map_or(offset + items.len() == total, |cursor| {
            cursor.len() <= 20
                && cursor
                    .parse::<usize>()
                    .is_ok_and(|next| next == offset + items.len() && next > offset && next < total)
        })
}

fn summary_is_bounded(summary: &LabExperimentSummary) -> bool {
    identity_is_bounded(&summary.experiment_id)
        && summary.status.len() <= 64
        && summary.total_runs <= 4096
        && summary.completed_runs <= summary.total_runs
        && summary.failed_runs <= summary.total_runs - summary.completed_runs
}

fn batch_is_bounded(batch: &LabExperiment) -> bool {
    summary_is_bounded(&batch.summary)
        && identity_is_bounded(&batch.plan_digest)
        && !batch.variants.is_empty()
        && batch.variants.len() <= LAB_MAX_VARIANTS
        && batch.variants.iter().all(|row| {
            identity_is_bounded(&row.variant_id)
                && row.brain_family.len() <= 256
                && row.total_runs <= 128
                && row.completed_runs <= row.total_runs
                && row.failed_runs <= row.total_runs - row.completed_runs
                && row.running_runs <= row.total_runs - row.completed_runs - row.failed_runs
        })
        && batch
            .variants
            .iter()
            .map(|row| row.total_runs)
            .sum::<usize>()
            == batch.summary.total_runs
        && batch
            .variants
            .iter()
            .map(|row| row.completed_runs)
            .sum::<usize>()
            == batch.summary.completed_runs
        && batch
            .variants
            .iter()
            .map(|row| row.failed_runs)
            .sum::<usize>()
            == batch.summary.failed_runs
        && batch
            .variants
            .iter()
            .map(|row| &row.variant_id)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == batch.variants.len()
}

fn created_experiment_matches_draft(
    experiment: &LabExperiment,
    request_id: &str,
    draft: &LabExperimentDraft,
) -> bool {
    if !batch_is_bounded(experiment)
        || experiment.summary.experiment_id != request_id
        || experiment.variants.len() != 2
        || draft.seed_count.checked_mul(2) != Some(experiment.summary.total_runs)
    {
        return false;
    }
    [
        ("baseline", &draft.baseline_family),
        ("challenger", &draft.challenger_family),
    ]
    .into_iter()
    .all(|(id, family)| {
        experiment.variants.iter().any(|variant| {
            variant.variant_id == id
                && &variant.brain_family == family
                && variant.total_runs == draft.seed_count
        })
    })
}

fn comparison_matches_batch(comparison: &LabComparison, batch: &LabExperiment) -> bool {
    let same_generation = comparison.status_generation == batch.generation;
    comparison.plan_digest == batch.plan_digest && same_generation
}

impl LabWorkflows {
    /// Invalidate pending results without releasing an occupied worker slot.
    pub fn invalidate(&mut self) {
        // Saturation cannot resurrect an old generation: request admission also
        // refuses when this counter is exhausted.
        self.context = self.context.saturating_add(1);
        self.last_refresh = None;
        self.event_review = None;
        if let Some(creation) = &mut self.latest_experiment_creation
            && matches!(creation.state, LabExperimentCreationState::Pending)
        {
            creation.state = LabExperimentCreationState::Retryable {
                reason: "Creation result discarded; the application may still admit it. Retry the exact request on its original source.".into(),
            };
        }
        if let Some(save) = &mut self.latest_checkpoint_save
            && matches!(save.state, LabCheckpointSaveState::Pending)
        {
            save.state = LabCheckpointSaveState::Retryable {
                reason: "Save result discarded; the service may still complete. Retry the same identity only on the original source.".into(),
            };
        }
    }

    /// Bind observations and future work to the actual owner and visible panel.
    pub fn synchronize(&mut self, session: HostSessionId, panel: LabPanel) {
        if self.session != Some(session) {
            self.reset_observations();
            self.session = Some(session);
            self.note_prior_checkpoint_save();
            self.note_prior_experiment_creation();
        }
        if self.panel != panel {
            self.invalidate();
            self.panel = panel;
        }
    }

    /// Bind a selection to its complete canonical evidence, without cloning on
    /// every live publication. Eviction or changed content invalidates pending
    /// narrative work, without disturbing services in another visible panel.
    pub fn select_event(
        &mut self,
        selection: Option<(LabEventIdentity, &scriptbots_core::narrative::EventRecord)>,
    ) {
        let selection = selection.filter(|(event, record)| {
            self.session == Some(event.session_id)
                && event.ordinal > 0
                && event.tick == record.tick.0
        });
        let unchanged = match (&self.selected_event, &selection) {
            (None, None) => true,
            (Some(previous), Some((event, record))) => {
                previous.identity == *event && &previous.record == *record
            }
            _ => false,
        };
        if unchanged {
            return;
        }
        // Live retention still changes while another panel is open. Its
        // checkpoint save or experiment read does not depend on this selection;
        // the panel transition already invalidated any older event request.
        if self.panel == LabPanel::Narrative {
            self.invalidate();
        }
        self.selected_event = selection.map(|(identity, record)| SelectedEvent {
            identity,
            record: record.clone(),
        });
    }

    /// Take one accepted event frame after polling the bounded worker.
    /// Any intervening selection, source or panel change clears this result.
    pub fn take_event_review(&mut self) -> Option<LabEventReview> {
        self.event_review.take()
    }

    fn note_prior_experiment_creation(&mut self) {
        if let Some(creation) = &self.latest_experiment_creation
            && self.session != Some(creation.session_id)
            && !matches!(creation.state, LabExperimentCreationState::Created { .. })
        {
            self.notice = Some(format!(
                "Previous source creation {} is unconfirmed. Return to that source to retry, or explicitly abandon its local retry identity before composing a new request.",
                creation.request_id,
            ));
        }
    }

    fn has_unconfirmed_experiment_creation(&self) -> bool {
        self.latest_experiment_creation.as_ref().is_some_and(|creation| {
            !matches!(creation.state, LabExperimentCreationState::Created { .. })
        })
    }

    /// Whether the next creation retries an unconfirmed request on this exact source.
    #[must_use]
    pub fn will_retry_experiment_creation(&self) -> bool {
        self.latest_experiment_creation.as_ref().is_some_and(|creation| {
            self.session == Some(creation.session_id)
                && !matches!(creation.state, LabExperimentCreationState::Created { .. })
        })
    }

    /// Whether the composer may change scientific choices before explicit submission.
    #[must_use]
    pub fn can_edit_experiment_draft(&self) -> bool {
        self.composing_experiment
            && self.panel == LabPanel::Experiments
            && self.session.is_some()
            && self.experiment_capabilities.is_some()
            && !self.is_busy()
            && !self.has_unconfirmed_experiment_creation()
    }

    /// Whether one bounded new creation or exact same-source retry can be submitted.
    #[must_use]
    pub fn can_create_experiment(&self) -> bool {
        if !self.composing_experiment
            || self.panel != LabPanel::Experiments
            || self.session.is_none()
            || self.submit.is_none()
            || self.is_busy()
            || (self.has_unconfirmed_experiment_creation()
                && !self.will_retry_experiment_creation())
        {
            return false;
        }
        let Some(capabilities) = &self.experiment_capabilities else {
            return false;
        };
        let draft = self
            .latest_experiment_creation
            .as_ref()
            .filter(|_| self.will_retry_experiment_creation())
            .map_or(&self.experiment_draft, |creation| &creation.draft);
        draft.validate(capabilities).is_ok()
    }

    /// Whether an idle unconfirmed request can be forgotten locally.
    ///
    /// Forgetting the identity does not cancel an experiment that was admitted.
    #[must_use]
    pub fn can_abandon_experiment_request(&self) -> bool {
        !self.is_busy() && self.has_unconfirmed_experiment_creation()
    }

    fn accept_experiment_capabilities(&mut self, capabilities: LabExperimentCapabilities) -> bool {
        if !self.composing_experiment || self.panel != LabPanel::Experiments {
            return false;
        }
        if capabilities.validate().is_err() {
            self.experiment_capabilities = None;
            return false;
        }
        if self.experiment_draft.baseline_family.is_empty()
            && self.experiment_draft.challenger_family.is_empty()
            && !self.has_unconfirmed_experiment_creation()
        {
            self.experiment_draft.baseline_family = capabilities
                .families
                .iter()
                .find(|family| family.as_str() == "mlp.baseline")
                .or_else(|| capabilities.families.first())
                .cloned()
                .unwrap_or_default();
            self.experiment_draft.challenger_family = capabilities
                .families
                .iter()
                .find(|family| {
                    family.as_str() == "dwraon.baseline"
                        && *family != &self.experiment_draft.baseline_family
                })
                .or_else(|| {
                    capabilities
                        .families
                        .iter()
                        .find(|family| *family != &self.experiment_draft.baseline_family)
                })
                .cloned()
                .unwrap_or_default();
        }
        self.experiment_capabilities = Some(capabilities);
        true
    }

    fn create_experiment(&mut self) {
        if !self.can_create_experiment() {
            self.set_notice("Creation requires a valid disclosed draft, current application capabilities, the Experiments composer and a free worker slot.");
            return;
        }
        let Some(session_id) = self.session else {
            return;
        };
        let (request_id, draft) = if self.will_retry_experiment_creation() {
            let Some(creation) = &self.latest_experiment_creation else {
                return;
            };
            (creation.request_id.clone(), creation.draft.clone())
        } else {
            let Some(request_id) = next_checkpoint_request(session_id)
                .filter(|identity| experiment_request_identity_is_valid(identity))
            else {
                self.set_notice("Experiment request identities are exhausted.");
                return;
            };
            (request_id, self.experiment_draft.clone())
        };
        let request = LabWorkflowRequest::CreateExperiment {
            request_id: request_id.clone(),
            session_id,
            draft: draft.clone(),
        };
        self.latest_experiment_creation = Some(LabExperimentCreation {
            request_id,
            session_id,
            draft,
            state: LabExperimentCreationState::Pending,
        });
        self.start(request.clone());
        if self.is_busy() {
            self.set_notice("Submitting the frozen experiment request; awaiting application admission.");
        } else {
            let reason = self
                .notice
                .clone()
                .unwrap_or_else(|| "Experiment worker did not start.".into());
            self.experiment_creation_unconfirmed(&request, &reason);
        }
    }

    fn experiment_creation_unconfirmed(&mut self, request: &LabWorkflowRequest, reason: &str) {
        let LabWorkflowRequest::CreateExperiment {
            request_id,
            session_id,
            draft,
        } = request
        else {
            return;
        };
        if let Some(creation) = &mut self.latest_experiment_creation
            && creation.request_id == *request_id
            && creation.session_id == *session_id
            && creation.draft == *draft
            && !matches!(creation.state, LabExperimentCreationState::Created { .. })
        {
            creation.state = LabExperimentCreationState::Retryable {
                reason: bounded_message(reason),
            };
        }
    }

    fn finish_experiment_creation(
        &mut self,
        request: &LabWorkflowRequest,
        result: Result<LabWorkflowResponse, String>,
    ) {
        let LabWorkflowRequest::CreateExperiment {
            request_id,
            session_id,
            draft,
        } = request
        else {
            return;
        };
        let reason = match result {
            Ok(LabWorkflowResponse::ExperimentCreated {
                request_id: returned_id,
                session_id: returned_session,
                experiment,
            }) if returned_id == *request_id
                && returned_session == *session_id
                && self.session == Some(*session_id)
                && self.panel == LabPanel::Experiments
                && self.composing_experiment
                && self.latest_experiment_creation.as_ref().is_some_and(|creation| {
                    creation.request_id == *request_id
                        && creation.session_id == *session_id
                        && creation.draft == *draft
                })
                && experiment_request_identity_is_valid(request_id)
                && created_experiment_matches_draft(&experiment, request_id, draft) =>
            {
                self.notice = Some(format!(
                    "Experiment {} admitted; observed status: {}.",
                    experiment.summary.experiment_id, experiment.summary.status,
                ));
                self.latest_experiment_creation = Some(LabExperimentCreation {
                    request_id: returned_id,
                    session_id: returned_session,
                    draft: draft.clone(),
                    state: LabExperimentCreationState::Created {
                        experiment_id: experiment.summary.experiment_id.clone(),
                    },
                });
                self.composing_experiment = false;
                self.experiments = None;
                self.experiment_cursor = None;
                self.selected_experiment = Some(experiment.summary.experiment_id.clone());
                self.baseline = Some("baseline".into());
                self.challenger = Some("challenger".into());
                self.comparison = None;
                self.variant_offset = 0;
                self.accept_batch(experiment);
                self.last_refresh = Some(Instant::now());
                return;
            }
            Ok(_) => {
                "Experiment receipt did not match the frozen request, source or two-arm cohort.".into()
            }
            Err(error) => bounded_message(&error),
        };
        self.experiment_creation_unconfirmed(request, &reason);
        self.set_notice(format!(
            "Experiment creation is unconfirmed: {reason}. Retry preserves the exact identity and inputs."
        ));
    }

    fn note_prior_checkpoint_save(&mut self) {
        if let Some(save) = &self.latest_checkpoint_save
            && self.session != Some(save.session_id)
            && !matches!(save.state, LabCheckpointSaveState::Saved(_))
        {
            self.notice = Some(format!(
                "Previous source save {} is unconfirmed. After its worker drains, a save of this source receives a fresh identity.",
                save.request_id,
            ));
        }
    }

    /// Whether the current source can admit one save request.
    ///
    /// The frontend must additionally require its live presentation, because
    /// historical presentation is deliberately not owned by this model.
    #[must_use]
    pub fn can_create_checkpoint(&self) -> bool {
        self.panel == LabPanel::Checkpoints
            && self.session.is_some()
            && self.submit.is_some()
            && !self.is_busy()
    }

    /// Whether the next save would retry an unconfirmed request for this exact
    /// session, rather than capture a newly requested cut.
    #[must_use]
    pub fn will_retry_checkpoint_save(&self) -> bool {
        self.latest_checkpoint_save.as_ref().is_some_and(|save| {
            self.session == Some(save.session_id)
                && !matches!(save.state, LabCheckpointSaveState::Saved(_))
        })
    }

    fn create_checkpoint(&mut self) {
        if !self.can_create_checkpoint() {
            self.set_notice("Saving requires the Checkpoints panel, an identified live source, application services and a free worker slot.");
            return;
        }
        let Some(session_id) = self.session else {
            return;
        };
        let request_id = if self.will_retry_checkpoint_save() {
            self.latest_checkpoint_save
                .as_ref()
                .map(|save| save.request_id.clone())
        } else {
            next_checkpoint_request(session_id)
        };
        let Some(request_id) = request_id else {
            self.set_notice("Checkpoint request identities are exhausted.");
            return;
        };
        let request = LabWorkflowRequest::CreateCheckpoint {
            request_id: request_id.clone(),
            session_id,
        };
        self.latest_checkpoint_save = Some(LabCheckpointSave {
            request_id,
            session_id,
            state: LabCheckpointSaveState::Pending,
        });
        self.start(request.clone());
        if self.is_busy() {
            self.set_notice("Saving checkpoint; awaiting the application's verified receipt.");
        } else {
            let reason = self
                .notice
                .clone()
                .unwrap_or_else(|| "Checkpoint worker did not start.".into());
            self.checkpoint_save_unconfirmed(&request, &reason);
        }
    }

    fn checkpoint_save_unconfirmed(&mut self, request: &LabWorkflowRequest, reason: &str) {
        let LabWorkflowRequest::CreateCheckpoint {
            request_id,
            session_id,
        } = request
        else {
            return;
        };
        if let Some(save) = &mut self.latest_checkpoint_save
            && save.request_id == *request_id
            && save.session_id == *session_id
            && !matches!(save.state, LabCheckpointSaveState::Saved(_))
        {
            save.state = LabCheckpointSaveState::Retryable {
                reason: bounded_message(reason),
            };
        }
    }

    fn finish_checkpoint_save(
        &mut self,
        request: &LabWorkflowRequest,
        result: Result<LabWorkflowResponse, String>,
    ) {
        let LabWorkflowRequest::CreateCheckpoint {
            request_id,
            session_id,
        } = request
        else {
            return;
        };
        let reason = match result {
            Ok(LabWorkflowResponse::CheckpointCreated {
                request_id: returned_id,
                session_id: returned_session,
                checkpoint,
            }) if returned_id == *request_id
                && returned_session == *session_id
                && self.session == Some(*session_id)
                && self.latest_checkpoint_save.as_ref().is_some_and(|save| {
                    save.request_id == *request_id && save.session_id == *session_id
                })
                && checkpoint_receipt_is_bounded(&checkpoint) =>
            {
                self.notice = Some(format!(
                    "Saved checkpoint {} at tick {} ({} agents).",
                    checkpoint.checkpoint_id, checkpoint.tick, checkpoint.agent_count,
                ));
                self.selected_checkpoint = Some(checkpoint.checkpoint_id.clone());
                self.latest_checkpoint_save = Some(LabCheckpointSave {
                    request_id: returned_id,
                    session_id: returned_session,
                    state: LabCheckpointSaveState::Saved(checkpoint),
                });
                self.checkpoints = None;
                self.checkpoint_cursor = None;
                self.last_refresh = None;
                return;
            }
            Ok(_) => {
                "Checkpoint receipt did not match the request, source or saved-cut bounds.".into()
            }
            Err(error) => bounded_message(&error),
        };
        self.checkpoint_save_unconfirmed(request, &reason);
        self.set_notice(format!(
            "Checkpoint save is unconfirmed: {reason}. Retry preserves the same request identity."
        ));
    }

    fn start(&mut self, request: LabWorkflowRequest) {
        if self.job.is_some() {
            self.notice =
                Some("A Lab request is still running; its worker slot remains occupied.".into());
            return;
        }
        let Some(submit) = self.submit.clone() else {
            self.notice =
                Some("Application experiment and checkpoint services are unavailable.".into());
            self.last_refresh = Some(Instant::now());
            return;
        };
        let Some(session) = self.session else { return };
        let Some(generation) = self
            .generation
            .checked_add(1)
            .filter(|_| self.context < u64::MAX)
        else {
            self.notice = Some("Lab request identities are exhausted.".into());
            return;
        };
        let (send, result) = std::sync::mpsc::sync_channel(1);
        let worker_request = request.clone();
        match std::thread::Builder::new()
            .name("scriptbots-native-lab".into())
            .stack_size(LAB_WORKER_STACK_BYTES)
            .spawn(move || {
                let _ = send.send(submit(worker_request));
            }) {
            Ok(_) => {
                self.generation = generation;
                self.job = Some(LabJob {
                    request,
                    generation,
                    context: self.context,
                    session,
                    result,
                });
                self.notice = Some("Loading application evidence…".into());
                self.last_refresh = Some(Instant::now());
            }
            Err(error) => {
                self.notice = Some(format!("Lab worker unavailable: {error}"));
                self.last_refresh = Some(Instant::now());
            }
        }
    }

    /// Selected metadata from a verified catalog page or the latest save receipt.
    #[must_use]
    pub fn selected_checkpoint_metadata(&self) -> Option<&LabCheckpoint> {
        let selected = self.selected_checkpoint.as_deref()?;
        self.checkpoints
            .as_ref()
            .and_then(|page| page.items.iter().find(|row| row.checkpoint_id == selected))
            .or_else(|| {
                self.saved_checkpoint_metadata()
                    .filter(|row| row.checkpoint_id == selected)
            })
    }

    fn saved_checkpoint_metadata(&self) -> Option<&LabCheckpoint> {
        let save = self.latest_checkpoint_save.as_ref()?;
        if self.session != Some(save.session_id) {
            return None;
        }
        match &save.state {
            LabCheckpointSaveState::Saved(checkpoint) => Some(checkpoint),
            LabCheckpointSaveState::Pending | LabCheckpointSaveState::Retryable { .. } => None,
        }
    }

    /// Current request for two distinct admitted variants.
    #[must_use]
    pub fn comparison_request(&self) -> Option<LabWorkflowRequest> {
        if self.composing_experiment {
            return None;
        }
        let batch = self.batch.as_ref()?;
        let baseline = self.baseline.as_ref()?;
        let challenger = self.challenger.as_ref()?;
        if baseline == challenger
            || !batch.variants.iter().any(|row| &row.variant_id == baseline)
            || !batch
                .variants
                .iter()
                .any(|row| &row.variant_id == challenger)
        {
            return None;
        }
        Some(LabWorkflowRequest::Compare {
            experiment_id: batch.summary.experiment_id.clone(),
            baseline_variant: baseline.clone(),
            challenger_variant: challenger.clone(),
            metric: self.metric,
        })
    }

    /// Consume one worker result without blocking the frontend.
    ///
    /// An accepted, still-selected checkpoint returns a frame. An accepted event
    /// is available through [`Self::take_event_review`]. The frontend must honor
    /// any newer local live-view intent before displaying either result.
    pub fn poll(&mut self) -> Option<(String, Arc<RenderSnapshot>)> {
        let job = self.job.as_ref()?;
        let result = match job.result.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => Err("Lab worker disconnected".to_owned()),
        };
        let job = self.job.take()?;
        if job.generation != self.generation
            || job.context != self.context
            || self.session != Some(job.session)
        {
            self.experiment_creation_unconfirmed(
                &job.request,
                "Creation finished after its selection or source changed; its receipt was discarded.",
            );
            self.checkpoint_save_unconfirmed(
                &job.request,
                "Save finished after the selection or session changed; its receipt was discarded.",
            );
            self.notice = Some(
                "Lab request finished after its selection or session changed; result discarded."
                    .into(),
            );
            return None;
        }
        if matches!(&job.request, LabWorkflowRequest::CreateExperiment { .. }) {
            self.finish_experiment_creation(&job.request, result);
            return None;
        }
        if matches!(&job.request, LabWorkflowRequest::CreateCheckpoint { .. }) {
            self.finish_checkpoint_save(&job.request, result);
            return None;
        }
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                if matches!(&job.request, LabWorkflowRequest::ExperimentCapabilities) {
                    self.experiment_capabilities = None;
                }
                if matches!(
                    &job.request,
                    LabWorkflowRequest::GetExperiment { .. }
                        | LabWorkflowRequest::CancelExperiment { .. }
                        | LabWorkflowRequest::ResumeExperiment { .. }
                        | LabWorkflowRequest::Compare { .. }
                ) {
                    self.comparison = None;
                }
                self.notice = Some(format!("Lab request failed: {}", bounded_message(&error)));
                return None;
            }
        };
        self.accept_response(&job.request, response)
    }

    fn accept_response(
        &mut self,
        request: &LabWorkflowRequest,
        response: LabWorkflowResponse,
    ) -> Option<(String, Arc<RenderSnapshot>)> {
        let accepted = match (request, response) {
            (
                LabWorkflowRequest::ExperimentCapabilities,
                LabWorkflowResponse::ExperimentCapabilities(capabilities),
            ) => self.accept_experiment_capabilities(capabilities),
            (
                LabWorkflowRequest::ListExperiments { cursor },
                LabWorkflowResponse::Experiments(page),
            ) => self.accept_experiment_page(cursor.as_deref(), page),
            (
                LabWorkflowRequest::GetExperiment { experiment_id }
                | LabWorkflowRequest::CancelExperiment { experiment_id }
                | LabWorkflowRequest::ResumeExperiment { experiment_id },
                LabWorkflowResponse::Experiment(batch),
            ) if self.selected_experiment.as_ref() == Some(experiment_id)
                && batch.summary.experiment_id == *experiment_id
                && batch_is_bounded(&batch) =>
            {
                self.accept_batch(batch);
                true
            }
            (
                LabWorkflowRequest::Compare {
                    experiment_id,
                    baseline_variant,
                    challenger_variant,
                    metric,
                },
                LabWorkflowResponse::Comparison(comparison),
            ) if self.comparison_request().as_ref() == Some(request)
                && comparison.experiment_id == *experiment_id
                && comparison.baseline_variant == *baseline_variant
                && comparison.challenger_variant == *challenger_variant
                && comparison.metric == *metric
                && self
                    .batch
                    .as_ref()
                    .is_some_and(|batch| comparison_matches_batch(&comparison, batch))
                && comparison_is_bounded(&comparison) =>
            {
                self.comparison = Some(comparison);
                true
            }
            (
                LabWorkflowRequest::ListCheckpoints { cursor },
                LabWorkflowResponse::Checkpoints(page),
            ) => self.accept_checkpoint_page(cursor.as_deref(), page),
            (
                LabWorkflowRequest::ReviewCheckpoint { checkpoint_id },
                LabWorkflowResponse::CheckpointFrame {
                    checkpoint_id: returned_id,
                    snapshot,
                },
            ) if self.panel == LabPanel::Checkpoints
                && checkpoint_id == &returned_id
                && self.selected_checkpoint_metadata().is_some_and(|row| {
                    row.checkpoint_id == returned_id
                        && row.tick == snapshot.world.tick
                        && row.agent_count == snapshot.world.agents.len()
                }) =>
            {
                self.notice = Some(format!(
                    "Verified checkpoint {returned_id} at tick {}.",
                    snapshot.world.tick
                ));
                return Some((returned_id, snapshot));
            }
            (
                LabWorkflowRequest::ReviewEvent { event },
                LabWorkflowResponse::EventFrame(review),
            ) if self.panel == LabPanel::Narrative
                && self.session == Some(event.session_id)
                && self.selected_event.as_ref().is_some_and(|selected| {
                    selected.identity == *event && selected.record == review.record
                })
                && review.event == *event
                && review.record.tick.0 == event.tick
                && review.snapshot.session_id == event.session_id
                && review.snapshot.world.tick == event.tick
                && review.base_tick <= event.tick
                && identity_is_bounded(&review.checkpoint_id) =>
            {
                self.notice = Some(format!(
                    "Reconstructed event #{} at tick {} from checkpoint {} at tick {}.",
                    event.ordinal, event.tick, review.checkpoint_id, review.base_tick,
                ));
                self.event_review = Some(review);
                return None;
            }
            _ => false,
        };
        if !accepted && matches!(request, LabWorkflowRequest::ExperimentCapabilities) {
            self.experiment_capabilities = None;
        }
        if !accepted
            && matches!(
                request,
                LabWorkflowRequest::GetExperiment { .. }
                    | LabWorkflowRequest::CancelExperiment { .. }
                    | LabWorkflowRequest::ResumeExperiment { .. }
                    | LabWorkflowRequest::Compare { .. }
            )
        {
            self.comparison = None;
        }
        self.notice = Some(if accepted {
            "Application evidence refreshed.".into()
        } else {
            "Lab response changed, exceeded its bounds, or did not match the request; refresh before retrying.".into()
        });
        if accepted && matches!(request, LabWorkflowRequest::ExperimentCapabilities) {
            self.note_prior_experiment_creation();
        }
        None
    }

    fn accept_experiment_page(&mut self, cursor: Option<&str>, page: LabExperimentPage) -> bool {
        if cursor != self.experiment_cursor.as_deref()
            || page.offset != cursor.and_then(|value| value.parse().ok()).unwrap_or(0)
            || !page_is_bounded(
                &page.items,
                page.offset,
                page.total,
                page.next_cursor.as_deref(),
            )
            || !page.items.iter().all(summary_is_bounded)
        {
            return false;
        }
        self.experiments = Some(page);
        true
    }

    fn accept_checkpoint_page(&mut self, cursor: Option<&str>, page: LabCheckpointPage) -> bool {
        if cursor != self.checkpoint_cursor.as_deref()
            || page.offset != cursor.and_then(|value| value.parse().ok()).unwrap_or(0)
            || !page_is_bounded(
                &page.items,
                page.offset,
                page.total,
                page.next_cursor.as_deref(),
            )
            || !page
                .items
                .iter()
                .all(|row| identity_is_bounded(&row.checkpoint_id) && row.world_digest.len() <= 128)
        {
            return false;
        }
        if self.selected_checkpoint.as_ref().is_some_and(|id| {
            !page.items.iter().any(|row| &row.checkpoint_id == id)
                && self
                    .saved_checkpoint_metadata()
                    .is_none_or(|row| &row.checkpoint_id != id)
        }) {
            self.selected_checkpoint = None;
        }
        self.checkpoints = Some(page);
        true
    }

    fn refresh_request(&self) -> Option<LabWorkflowRequest> {
        match self.panel {
            LabPanel::Inactive | LabPanel::Narrative => None,
            LabPanel::Experiments if self.composing_experiment => {
                Some(LabWorkflowRequest::ExperimentCapabilities)
            }
            LabPanel::Experiments => Some(self.selected_experiment.as_ref().map_or_else(
                || LabWorkflowRequest::ListExperiments {
                    cursor: self.experiment_cursor.clone(),
                },
                |id| LabWorkflowRequest::GetExperiment {
                    experiment_id: id.clone(),
                },
            )),
            LabPanel::Checkpoints => Some(LabWorkflowRequest::ListCheckpoints {
                cursor: self.checkpoint_cursor.clone(),
            }),
        }
    }

    /// Refresh visible changing evidence at most once every two seconds.
    pub fn maybe_refresh(&mut self) {
        if self.job.is_some()
            || self.submit.is_none()
            || self
                .last_refresh
                .is_some_and(|last| last.elapsed() < REFRESH_INTERVAL)
        {
            return;
        }
        let needs_refresh = match self.panel {
            LabPanel::Inactive | LabPanel::Narrative => false,
            LabPanel::Checkpoints => self.checkpoints.is_none(),
            LabPanel::Experiments if self.composing_experiment => {
                self.experiment_capabilities.is_none()
            }
            LabPanel::Experiments => {
                self.selected_experiment.is_none()
                    || self.batch.as_ref().is_none_or(|batch| {
                        matches!(
                            batch.summary.status.as_str(),
                            "running" | "pending" | "cancelling"
                        )
                    })
            }
        };
        if needs_refresh && let Some(request) = self.refresh_request() {
            self.start(request);
        }
    }

    fn accept_batch(&mut self, mut batch: LabExperiment) {
        batch.error_reason = batch.error_reason.as_deref().map(bounded_message);
        if self
            .comparison
            .as_ref()
            .is_some_and(|comparison| !comparison_matches_batch(comparison, &batch))
        {
            self.comparison = None;
        }
        if self
            .baseline
            .as_ref()
            .is_none_or(|id| !batch.variants.iter().any(|row| &row.variant_id == id))
        {
            self.baseline = batch.variants.first().map(|row| row.variant_id.clone());
        }
        if self
            .challenger
            .as_ref()
            .is_none_or(|id| !batch.variants.iter().any(|row| &row.variant_id == id))
        {
            self.challenger = batch
                .variants
                .iter()
                .find(|row| Some(&row.variant_id) != self.baseline.as_ref())
                .map(|row| row.variant_id.clone());
        }
        self.variant_offset = self
            .variant_offset
            .min(batch.variants.len().saturating_sub(1) / VARIANTS_PER_PAGE * VARIANTS_PER_PAGE);
        self.batch = Some(batch);
    }
    /// Whether the sole worker slot is occupied, including cancelled work.
    #[must_use]
    pub const fn is_busy(&self) -> bool {
        self.job.is_some()
    }

    /// Set a bounded notice after a frontend applies a returned frame.
    pub fn set_notice(&mut self, message: impl AsRef<str>) {
        self.notice = Some(bounded_message(message.as_ref()));
    }

    /// Clear presentation observations, preserving the occupied worker slot,
    /// application adapter, actual session and any unconfirmed save identity.
    pub fn reset_observations(&mut self) {
        self.invalidate();
        self.composing_experiment = false;
        self.experiment_capabilities = None;
        self.experiment_draft = self
            .latest_experiment_creation
            .as_ref()
            .filter(|_| self.has_unconfirmed_experiment_creation())
            .map_or_else(LabExperimentDraft::default, |creation| creation.draft.clone());
        self.experiments = None;
        self.experiment_cursor = None;
        self.selected_experiment = None;
        self.batch = None;
        self.baseline = None;
        self.challenger = None;
        self.comparison = None;
        self.checkpoints = None;
        self.checkpoint_cursor = None;
        self.selected_checkpoint = None;
        self.selected_event = None;
        self.notice = None;
        self.variant_offset = 0;
    }

    fn allow_draft_edit(&mut self) -> bool {
        if self.can_edit_experiment_draft() {
            true
        } else {
            self.set_notice("Draft inputs are locked while work is pending or a creation is unconfirmed. Retry its exact inputs, or explicitly abandon it after the worker finishes.");
            false
        }
    }

    fn report_draft_validation(&mut self) {
        let message = self.experiment_capabilities.as_ref().map_or_else(
            || "Load current experiment capabilities before submitting.".to_owned(),
            |capabilities| {
                self.experiment_draft.validate(capabilities).map_or_else(
                    |reason| format!("Draft cannot be submitted: {reason}"),
                    |()| "Draft ready. Creation starts only when requested explicitly.".into(),
                )
            },
        );
        self.set_notice(message);
    }

    /// Apply one native UI intent, preserving request identity and bounded state.
    pub fn dispatch(&mut self, action: LabAction) {
        match action {
            LabAction::NewExperiment => {
                if self.panel != LabPanel::Experiments {
                    self.set_notice("Open the Experiments panel before creating a batch.");
                    return;
                }
                self.invalidate();
                self.composing_experiment = true;
                if let Some(creation) = &self.latest_experiment_creation
                    && !matches!(creation.state, LabExperimentCreationState::Created { .. })
                {
                    self.experiment_draft = creation.draft.clone();
                }
                self.experiment_capabilities = None;
                self.comparison = None;
                if !self.is_busy() {
                    self.start(LabWorkflowRequest::ExperimentCapabilities);
                }
            }
            LabAction::DraftBaseline(ref family) | LabAction::DraftChallenger(ref family) => {
                if !self.allow_draft_edit() {
                    return;
                }
                if !self
                    .experiment_capabilities
                    .as_ref()
                    .is_some_and(|capabilities| capabilities.families.contains(family))
                {
                    self.set_notice("This brain family was not offered by the current runner.");
                    return;
                }
                if matches!(&action, LabAction::DraftBaseline(_)) {
                    self.experiment_draft.baseline_family = family.clone();
                } else {
                    self.experiment_draft.challenger_family = family.clone();
                }
                self.report_draft_validation();
            }
            LabAction::FirstSeed(seed) => {
                if self.allow_draft_edit() {
                    self.experiment_draft.first_seed = seed;
                    self.report_draft_validation();
                }
            }
            LabAction::SeedCount(count) => {
                if self.allow_draft_edit() {
                    self.experiment_draft.seed_count = count;
                    self.report_draft_validation();
                }
            }
            LabAction::TicksPerRun(ticks) => {
                if self.allow_draft_edit() {
                    self.experiment_draft.ticks_per_run = ticks;
                    self.report_draft_validation();
                }
            }
            LabAction::CreateExperiment => self.create_experiment(),
            LabAction::AbandonExperimentRequest => {
                if !self.can_abandon_experiment_request() {
                    self.set_notice("Only an unconfirmed creation whose worker has finished can be abandoned locally.");
                    return;
                }
                if let Some(creation) = self.latest_experiment_creation.take() {
                    self.invalidate();
                    self.set_notice(format!(
                        "Forgot local retry identity {}. Its experiment may already exist; inspect the catalog before submitting another batch.",
                        creation.request_id,
                    ));
                }
            }
            LabAction::ReturnLive => self.invalidate(),
            LabAction::CancelRequest => {
                self.invalidate();
                self.set_notice("Result cancelled. The worker remains occupied until it finishes; submitted actions may still complete. Unconfirmed checkpoint saves retain their retry identity.");
            }
            LabAction::SelectExperiment(id) => self.select_experiment(id),
            LabAction::AllExperiments | LabAction::BackToExperiments => {
                self.invalidate();
                self.composing_experiment = false;
                self.selected_experiment = None;
                self.batch = None;
                self.comparison = None;
                self.experiments = None;
                self.experiment_cursor = None;
                self.set_notice("Returned to the experiment catalog. Any admitted experiment continues independently.");
            }
            LabAction::ExperimentsPage(cursor) => {
                self.invalidate();
                self.experiment_cursor = cursor;
                self.experiments = None;
            }
            LabAction::CheckpointsPage(cursor) => {
                self.invalidate();
                self.checkpoint_cursor = cursor;
                self.checkpoints = None;
                self.selected_checkpoint = None;
            }
            LabAction::SelectCheckpoint(id) => {
                if self
                    .checkpoints
                    .as_ref()
                    .is_some_and(|page| page.items.iter().any(|row| row.checkpoint_id == id))
                    || self
                        .saved_checkpoint_metadata()
                        .is_some_and(|row| row.checkpoint_id == id)
                {
                    self.invalidate();
                    self.selected_checkpoint = Some(id);
                }
            }
            LabAction::Baseline(ref id) | LabAction::Challenger(ref id) => {
                let baseline = matches!(&action, LabAction::Baseline(_));
                if self
                    .batch
                    .as_ref()
                    .is_some_and(|batch| batch.variants.iter().any(|row| &row.variant_id == id))
                {
                    self.invalidate();
                    if baseline {
                        self.baseline = Some(id.clone());
                    } else {
                        self.challenger = Some(id.clone());
                    }
                    self.comparison = None;
                }
            }
            LabAction::Metric(metric) => {
                if self.metric != metric {
                    self.invalidate();
                    self.metric = metric;
                    self.comparison = None;
                    self.set_notice(format!(
                        "Selected {} ({}). Request a comparison for this endpoint.",
                        metric.label(),
                        metric.unit(),
                    ));
                }
            }
            LabAction::VariantsPage(offset) => {
                let maximum = self.batch.as_ref().map_or(0, |batch| {
                    batch.variants.len().saturating_sub(1) / VARIANTS_PER_PAGE * VARIANTS_PER_PAGE
                });
                self.variant_offset = (offset / VARIANTS_PER_PAGE * VARIANTS_PER_PAGE).min(maximum);
            }
            LabAction::Compare => {
                if let Some(request) = self.comparison_request() {
                    self.comparison = None;
                    self.start(request);
                }
            }
            LabAction::CancelExperiment | LabAction::ResumeExperiment => {
                if let Some(id) = self.selected_experiment.clone() {
                    let request = if matches!(action, LabAction::CancelExperiment) {
                        LabWorkflowRequest::CancelExperiment { experiment_id: id }
                    } else {
                        LabWorkflowRequest::ResumeExperiment { experiment_id: id }
                    };
                    self.start(request);
                }
            }
            LabAction::ReviewCheckpoint => {
                if let Some(row) = self.selected_checkpoint_metadata() {
                    self.start(LabWorkflowRequest::ReviewCheckpoint {
                        checkpoint_id: row.checkpoint_id.clone(),
                    });
                }
            }
            LabAction::ReviewEvent(event) => {
                if self.panel == LabPanel::Narrative
                    && self.session == Some(event.session_id)
                    && self
                        .selected_event
                        .as_ref()
                        .is_some_and(|selected| selected.identity == event)
                {
                    self.event_review = None;
                    self.start(LabWorkflowRequest::ReviewEvent { event });
                } else {
                    self.set_notice("Select the same retained event in the current owner's timeline before reviewing it.");
                }
            }
            LabAction::CreateCheckpoint => self.create_checkpoint(),
            LabAction::Refresh => {
                if let Some(request) = self.refresh_request() {
                    self.start(request);
                }
            }
        }
    }

    fn select_experiment(&mut self, id: String) {
        if !identity_is_bounded(&id) {
            self.set_notice("Invalid experiment identity.");
            return;
        }
        self.invalidate();
        self.composing_experiment = false;
        self.selected_experiment = Some(id.clone());
        self.batch = None;
        self.baseline = None;
        self.challenger = None;
        self.comparison = None;
        self.variant_offset = 0;
        if !self.is_busy() {
            self.start(LabWorkflowRequest::GetExperiment { experiment_id: id });
        }
    }
}

fn comparison_is_bounded(comparison: &LabComparison) -> bool {
    !comparison.matched_seeds.is_empty()
        && comparison.matched_seeds.len() <= 128
        && comparison
            .matched_seeds
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        && comparison.mean_difference.is_finite()
        && comparison
            .ci_95
            .is_none_or(|(low, high)| low.is_finite() && high.is_finite() && low <= high)
        && comparison.standardized_effect.is_none_or(f64::is_finite)
        && comparison
            .p_value
            .is_none_or(|p| p.is_finite() && (0.0..=1.0).contains(&p))
        && comparison
            .adjusted_p_value
            .is_none_or(|p| p.is_finite() && (0.0..=1.0).contains(&p))
        && (comparison.matched_seeds.len() > 1
            || (comparison.descriptive_only
                && comparison.ci_95.is_none()
                && comparison.standardized_effect.is_none()
                && comparison.p_value.is_none()
                && comparison.adjusted_p_value.is_none()))
}

/// Human-readable statistical observations shared by native Lab frontends.
#[must_use]
pub fn comparison_lines(comparison: &LabComparison) -> Vec<String> {
    let optional = |value: Option<f64>| {
        value.map_or_else(|| "unavailable".into(), |value| format!("{value:.6}"))
    };
    let mut lines = vec![
        format!(
            "Verified {} ({}), endpoint tick {}, {} matched seed pairs",
            comparison.metric.label(),
            comparison.metric.unit(),
            comparison.endpoint_tick,
            comparison.matched_seeds.len()
        ),
        format!(
            "Challenger − baseline: {:.6} {}",
            comparison.mean_difference,
            comparison.metric.unit(),
        ),
        format!(
            "95% CI: {}",
            comparison.ci_95.map_or_else(
                || "unavailable".into(),
                |(low, high)| format!("[{low:.6}, {high:.6}] {}", comparison.metric.unit())
            )
        ),
        format!(
            "p: {} · adjusted p: {} · paired Cohen's d_z: {}",
            optional(comparison.p_value),
            optional(comparison.adjusted_p_value),
            optional(comparison.standardized_effect)
        ),
        format!(
            "Verified source generation {} · plan {}",
            comparison.status_generation, comparison.plan_digest
        ),
    ];
    if comparison.descriptive_only {
        lines.push("Descriptive only: one independent seed pair cannot support inference.".into());
    }
    if let Some(reason) = &comparison.underpowered_reason {
        lines.push(bounded_message(reason));
    }
    if let Some(reason) = &comparison.undefined_effect_reason {
        lines.push(bounded_message(reason));
    }
    lines.push("Values describe the selected final-state endpoint across the matched seeds.".into());
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    };

    fn batch(variants: usize) -> LabExperiment {
        LabExperiment {
            summary: LabExperimentSummary {
                experiment_id: "actual-batch".into(),
                status: "completed".into(),
                total_runs: variants,
                completed_runs: variants,
                failed_runs: 0,
            },
            plan_digest: "recorded-plan".into(),
            generation: 9,
            variants: (0..variants)
                .map(|index| LabVariantProgress {
                    variant_id: format!("arm-{index:02}"),
                    brain_family: "mlp".into(),
                    total_runs: 1,
                    completed_runs: 1,
                    failed_runs: 0,
                    running_runs: 0,
                })
                .collect(),
            error_reason: None,
        }
    }

    fn comparison(challenger: &str) -> LabComparison {
        LabComparison {
            experiment_id: "actual-batch".into(),
            baseline_variant: "arm-00".into(),
            challenger_variant: challenger.into(),
            metric: LabMetric::AliveAgents,
            endpoint_tick: 2,
            matched_seeds: vec![7],
            mean_difference: 3.0,
            ci_95: None,
            standardized_effect: None,
            p_value: None,
            adjusted_p_value: None,
            descriptive_only: true,
            underpowered_reason: Some("one seed pair".into()),
            undefined_effect_reason: Some("one observation".into()),
            plan_digest: "recorded-plan".into(),
            status_generation: 9,
        }
    }

    fn deliver(
        state: &mut LabWorkflows,
        request: LabWorkflowRequest,
        response: LabWorkflowResponse,
    ) -> Option<(String, Arc<RenderSnapshot>)> {
        let (send, result) = std::sync::mpsc::sync_channel(1);
        assert!(send.send(Ok(response)).is_ok());
        state.generation += 1;
        state.job = Some(LabJob {
            request,
            generation: state.generation,
            context: state.context,
            session: state.session.unwrap(),
            result,
        });
        state.poll()
    }

    #[test]
    fn lab_worker_retains_one_slot_through_cancellation_and_rejects_old_sessions() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        let (started, observed_start) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        let released = Mutex::new(released);
        let mut state = LabWorkflows {
            submit: Some(Arc::new(move |_| {
                observed_calls.fetch_add(1, AtomicOrdering::SeqCst);
                started.send(()).unwrap();
                released
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap();
                Ok(LabWorkflowResponse::Experiments(LabExperimentPage {
                    items: vec![],
                    total: 0,
                    offset: 0,
                    next_cursor: None,
                    reserved_run_slots: 0,
                    max_run_slots: 1,
                }))
            })),
            ..LabWorkflows::default()
        };
        state.synchronize(HostSessionId::new(1), LabPanel::Experiments);
        let request = LabWorkflowRequest::ListExperiments { cursor: None };
        state.start(request.clone());
        observed_start.recv_timeout(Duration::from_secs(5)).unwrap();
        state.invalidate();
        state.start(request.clone());
        assert_eq!(
            calls.load(AtomicOrdering::SeqCst),
            1,
            "discarding a result must not create a second worker"
        );
        state.synchronize(HostSessionId::new(2), LabPanel::Experiments);
        state.reset_observations();
        assert!(
            state.is_busy(),
            "epoch reset retains the occupied worker slot"
        );
        release.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while state.job.is_some() {
            assert!(Instant::now() < deadline);
            assert!(state.poll().is_none());
            std::thread::yield_now();
        }
        assert!(
            state.experiments.is_none(),
            "an old-session catalog must not become the new session's catalog"
        );
        assert!(state.notice.as_deref().unwrap().contains("discarded"));
        state.start(request);
        observed_start.recv_timeout(Duration::from_secs(5)).unwrap();
        release.send(()).unwrap();
        while state.job.is_some() {
            assert!(Instant::now() < deadline);
            state.poll();
            std::thread::yield_now();
        }
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
        assert!(
            state.experiments.is_some(),
            "the slot must reopen after the original worker exits"
        );
    }

    fn composer_test_capabilities() -> LabExperimentCapabilities {
        LabExperimentCapabilities {
            families: vec!["mlp.baseline".into(), "dwraon.baseline".into()],
            max_seed_count: LAB_MAX_EXPERIMENT_SEEDS,
            max_ticks_per_run: 1000,
            max_total_ticks: 100_000,
            launch_description: "Fresh runner defaults; sequential independent worlds.".into(),
        }
    }

    fn composer_test_response(request: &LabWorkflowRequest) -> LabWorkflowResponse {
        let LabWorkflowRequest::CreateExperiment {
            request_id,
            session_id,
            draft,
        } = request
        else {
            panic!("expected creation request");
        };
        LabWorkflowResponse::ExperimentCreated {
            request_id: request_id.clone(),
            session_id: *session_id,
            experiment: LabExperiment {
                summary: LabExperimentSummary {
                    experiment_id: request_id.clone(),
                    status: "pending".into(),
                    total_runs: draft.seed_count * 2,
                    completed_runs: 0,
                    failed_runs: 0,
                },
                plan_digest: "admitted-plan".into(),
                generation: 1,
                variants: [
                    ("baseline", &draft.baseline_family),
                    ("challenger", &draft.challenger_family),
                ]
                .into_iter()
                .map(|(id, family)| LabVariantProgress {
                    variant_id: id.into(),
                    brain_family: family.clone(),
                    total_runs: draft.seed_count,
                    completed_runs: 0,
                    failed_runs: 0,
                    running_runs: 0,
                })
                .collect(),
                error_reason: None,
            },
        }
    }

    fn open_test_composer(submit: LabWorkflowSubmit) -> LabWorkflows {
        let mut state = LabWorkflows {
            submit: Some(submit),
            ..LabWorkflows::default()
        };
        state.synchronize(HostSessionId::new(1), LabPanel::Experiments);
        state.dispatch(LabAction::NewExperiment);
        drain_worker(&mut state);
        assert!(state.experiment_capabilities.is_some());
        assert!(state.latest_experiment_creation.is_none());
        state
    }

    #[test]
    fn experiment_draft_refuses_seed_wrap_and_total_work_before_submission() {
        let capabilities = composer_test_capabilities();
        let mut draft = LabExperimentDraft {
            baseline_family: "mlp.baseline".into(),
            challenger_family: "dwraon.baseline".into(),
            ..LabExperimentDraft::default()
        };
        assert_eq!(draft.seeds().unwrap(), vec![42, 43, 44]);
        draft.validate(&capabilities).unwrap();
        let mut exact_work = capabilities.clone();
        exact_work.max_total_ticks = 600;
        draft.validate(&exact_work).unwrap();
        exact_work.max_total_ticks -= 1;
        assert!(draft.validate(&exact_work).is_err());

        draft.first_seed = u64::MAX - 1;
        draft.seed_count = 2;
        assert_eq!(draft.seeds().unwrap(), vec![u64::MAX - 1, u64::MAX]);
        draft.validate(&capabilities).unwrap();
        draft.seed_count = 3;
        assert!(draft.seeds().is_err());
        assert!(draft.validate(&capabilities).is_err());
        draft.seed_count = usize::MAX;
        assert!(draft.seeds().is_err(), "reject before allocating from the count");
        draft.seed_count = 0;
        assert!(draft.seeds().is_err());

        draft.first_seed = 42;
        draft.seed_count = LAB_MAX_EXPERIMENT_SEEDS;
        draft.ticks_per_run = u64::MAX;
        let mut overflowing_work = capabilities.clone();
        overflowing_work.max_ticks_per_run = u64::MAX;
        overflowing_work.max_total_ticks = u64::MAX;
        assert!(draft.validate(&overflowing_work).unwrap_err().contains("overflows"));
        draft.seed_count = 1;
        draft.ticks_per_run = 100;
        draft.challenger_family = draft.baseline_family.clone();
        assert!(draft.validate(&capabilities).is_err());
        draft.challenger_family = "unavailable.family".into();
        assert!(draft.validate(&capabilities).is_err());
    }

    #[test]
    fn composer_refuses_unbounded_capabilities_and_never_clamps_its_disclosed_defaults() {
        let mut state = LabWorkflows::default();
        state.synchronize(HostSessionId::new(1), LabPanel::Experiments);
        state.composing_experiment = true;
        for case in 0..10 {
            let mut capabilities = composer_test_capabilities();
            match case {
                0 => capabilities.families = vec!["same".into(); 2],
                1 => {
                    capabilities.families = (0..=LAB_MAX_EXPERIMENT_FAMILIES)
                        .map(|index| format!("family{index}"))
                        .collect();
                }
                2 => capabilities.families[0] = "x".repeat(129),
                3 => capabilities.families[0] = "invalid family".into(),
                4 => capabilities.max_seed_count = usize::MAX,
                5 => capabilities.max_seed_count = 0,
                6 => capabilities.max_ticks_per_run = 0,
                7 => capabilities.max_total_ticks = 0,
                8 => capabilities.launch_description = "x".repeat(2049),
                9 => capabilities.launch_description = "invalid\nmultiline".into(),
                _ => unreachable!(),
            }
            assert!(capabilities.validate().is_err(), "case {case}");
            assert!(!state.accept_experiment_capabilities(capabilities), "case {case}");
            assert!(state.experiment_capabilities.is_none());
        }
        let mut capabilities = composer_test_capabilities();
        capabilities.families.insert(0, "assembly.baseline".into());
        capabilities.max_seed_count = 1;
        capabilities.max_ticks_per_run = 1;
        capabilities.max_total_ticks = 2;
        assert!(state.accept_experiment_capabilities(capabilities));
        assert_eq!(state.experiment_draft.baseline_family, "mlp.baseline");
        assert_eq!(state.experiment_draft.challenger_family, "dwraon.baseline");
        assert_eq!(state.experiment_draft.first_seed, 42);
        assert_eq!(state.experiment_draft.seed_count, 3);
        assert_eq!(state.experiment_draft.ticks_per_run, 100);
        assert!(state
            .experiment_draft
            .validate(state.experiment_capabilities.as_ref().unwrap())
            .is_err());
        assert!(state.latest_experiment_creation.is_none());
        assert!(!state.is_busy(), "capability observation starts no experiment");
    }

    #[test]
    fn experiment_identity_leaves_room_for_the_full_width_run_seed() {
        let identity = next_checkpoint_request(HostSessionId::new(u64::MAX)).unwrap();
        assert!(experiment_request_identity_is_valid(&identity));
        assert!(format!("{identity}-challenger-seed{}", u64::MAX).len() <= 128);
        assert!(experiment_request_identity_is_valid(&"x".repeat(92)));
        for invalid in [
            String::new(),
            ".".into(),
            "contains..traversal".into(),
            " leading".into(),
            "trailing ".into(),
            "a/b".into(),
            "a\\b".into(),
            "x".repeat(93),
        ] {
            assert!(!experiment_request_identity_is_valid(&invalid), "{invalid:?}");
        }
    }

    #[test]
    fn experiment_creation_retries_frozen_inputs_after_an_unconfirmed_admission() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let observed_requests = Arc::clone(&requests);
        let mut state = open_test_composer(Arc::new(move |request| {
            if request == LabWorkflowRequest::ExperimentCapabilities {
                return Ok(LabWorkflowResponse::ExperimentCapabilities(composer_test_capabilities()));
            }
            let mut requests = observed_requests.lock().unwrap();
            requests.push(request.clone());
            if requests.len() == 1 {
                Err("admission receipt was lost".into())
            } else {
                Ok(composer_test_response(&request))
            }
        }));
        assert!(requests.lock().unwrap().is_empty(), "opening is read-only");
        state.dispatch(LabAction::FirstSeed(u64::MAX - 1));
        state.dispatch(LabAction::SeedCount(2));
        state.dispatch(LabAction::TicksPerRun(1));
        assert!(state.can_create_experiment());
        state.dispatch(LabAction::CreateExperiment);
        drain_worker(&mut state);
        let frozen = state.latest_experiment_creation.clone().unwrap();
        assert!(state.will_retry_experiment_creation());
        assert!(!state.can_edit_experiment_draft());
        for edit in [
            LabAction::FirstSeed(99),
            LabAction::SeedCount(1),
            LabAction::TicksPerRun(500),
            LabAction::DraftBaseline("dwraon.baseline".into()),
            LabAction::DraftChallenger("mlp.baseline".into()),
        ] {
            state.dispatch(edit);
            assert_eq!(state.experiment_draft, frozen.draft);
        }
        state.dispatch(LabAction::NewExperiment);
        drain_worker(&mut state);
        assert_eq!(state.experiment_draft, frozen.draft);
        state.dispatch(LabAction::CreateExperiment);
        drain_worker(&mut state);
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0], requests[1]);
        assert!(!state.composing_experiment);
        assert_eq!(state.selected_experiment.as_ref(), Some(&frozen.request_id));
        assert_eq!(state.batch.as_ref().unwrap().summary.status, "pending");
        assert_eq!(state.batch.as_ref().unwrap().summary.total_runs, 4);
        assert!(matches!(
            state.latest_experiment_creation.as_ref().unwrap().state,
            LabExperimentCreationState::Created { .. }
        ));
    }

    #[test]
    fn unconfirmed_creation_survives_source_switches_until_exact_retry_or_explicit_abandon() {
        for abandon_on_other_source in [false, true] {
            let requests = Arc::new(Mutex::new(Vec::new()));
            let observed_requests = Arc::clone(&requests);
            let mut state = open_test_composer(Arc::new(move |request| {
                if request == LabWorkflowRequest::ExperimentCapabilities {
                    return Ok(LabWorkflowResponse::ExperimentCapabilities(composer_test_capabilities()));
                }
                let mut requests = observed_requests.lock().unwrap();
                requests.push(request.clone());
                if requests.len() == 1 {
                    Err("response unavailable after submission".into())
                } else {
                    Ok(composer_test_response(&request))
                }
            }));
            state.dispatch(LabAction::FirstSeed(137));
            state.dispatch(LabAction::SeedCount(8));
            state.dispatch(LabAction::CreateExperiment);
            drain_worker(&mut state);
            let frozen = state.latest_experiment_creation.clone().unwrap();

            state.synchronize(HostSessionId::new(2), LabPanel::Experiments);
            state.dispatch(LabAction::NewExperiment);
            drain_worker(&mut state);
            assert_eq!(state.experiment_draft, frozen.draft);
            assert_eq!(state.latest_experiment_creation.as_ref().unwrap().request_id, frozen.request_id);
            assert!(!state.will_retry_experiment_creation());
            assert!(!state.can_edit_experiment_draft());
            assert!(!state.can_create_experiment());
            state.dispatch(LabAction::FirstSeed(11));
            state.dispatch(LabAction::CreateExperiment);
            assert_eq!(requests.lock().unwrap().len(), 1);
            assert_eq!(state.experiment_draft, frozen.draft);

            if abandon_on_other_source {
                assert!(state.can_abandon_experiment_request());
                state.dispatch(LabAction::AbandonExperimentRequest);
                assert!(state.notice.as_deref().unwrap().contains("may already exist"));
                assert!(state.latest_experiment_creation.is_none());
                assert!(state.can_edit_experiment_draft());
                state.dispatch(LabAction::FirstSeed(11));
            } else {
                state.synchronize(HostSessionId::new(1), LabPanel::Experiments);
                state.dispatch(LabAction::NewExperiment);
                drain_worker(&mut state);
                assert_eq!(state.experiment_draft, frozen.draft);
                assert!(state.will_retry_experiment_creation());
            }
            state.dispatch(LabAction::CreateExperiment);
            drain_worker(&mut state);
            let requests = requests.lock().unwrap();
            assert_eq!(requests.len(), 2);
            if abandon_on_other_source {
                assert_ne!(requests[0], requests[1]);
                assert_ne!(state.selected_experiment.as_ref(), Some(&frozen.request_id));
            } else {
                assert_eq!(requests[0], requests[1]);
                assert_eq!(state.selected_experiment.as_ref(), Some(&frozen.request_id));
            }
        }
    }

    #[test]
    fn creation_context_changes_retain_the_worker_and_discard_late_admission_receipts() {
        for change in 0..5 {
            let calls = Arc::new(AtomicUsize::new(0));
            let observed_calls = Arc::clone(&calls);
            let (started, observed_start) = std::sync::mpsc::channel();
            let (release, released) = std::sync::mpsc::channel();
            let released = Mutex::new(released);
            let mut state = open_test_composer(Arc::new(move |request| {
                if request == LabWorkflowRequest::ExperimentCapabilities {
                    return Ok(LabWorkflowResponse::ExperimentCapabilities(composer_test_capabilities()));
                }
                observed_calls.fetch_add(1, AtomicOrdering::SeqCst);
                started.send(request.clone()).unwrap();
                released.lock().unwrap().recv_timeout(Duration::from_secs(5)).unwrap();
                Ok(composer_test_response(&request))
            }));
            state.dispatch(LabAction::CreateExperiment);
            let original = observed_start.recv_timeout(Duration::from_secs(5)).unwrap();
            let frozen = state.latest_experiment_creation.clone().unwrap();
            match change {
                0 => state.dispatch(LabAction::CancelRequest),
                1 => state.dispatch(LabAction::BackToExperiments),
                2 => state.dispatch(LabAction::NewExperiment),
                3 => state.synchronize(HostSessionId::new(1), LabPanel::Checkpoints),
                4 => state.synchronize(HostSessionId::new(2), LabPanel::Experiments),
                _ => unreachable!(),
            }
            assert!(state.is_busy(), "change {change} must retain the worker");
            assert!(!state.can_abandon_experiment_request());
            state.dispatch(LabAction::AbandonExperimentRequest);
            state.dispatch(LabAction::CreateExperiment);
            assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
            assert_eq!(state.latest_experiment_creation.as_ref().unwrap().request_id, frozen.request_id);
            assert_eq!(state.experiment_draft, frozen.draft);
            assert_eq!(state.job.as_ref().unwrap().request, original);
            release.send(()).unwrap();
            drain_worker(&mut state);
            assert!(state.selected_experiment.is_none());
            assert!(matches!(
                state.latest_experiment_creation.as_ref().unwrap().state,
                LabExperimentCreationState::Retryable { .. }
            ));
            assert!(state.can_abandon_experiment_request());
            state.dispatch(LabAction::AbandonExperimentRequest);
            assert!(state.latest_experiment_creation.is_none());
            assert!(state.notice.as_deref().unwrap().contains("may already exist"));
        }
    }

    #[test]
    fn creation_receipts_bind_request_source_and_both_declared_cohorts() {
        for mismatch in 0..8 {
            let mut state = LabWorkflows::default();
            let session = HostSessionId::new(1);
            state.synchronize(session, LabPanel::Experiments);
            state.composing_experiment = true;
            assert!(state.accept_experiment_capabilities(composer_test_capabilities()));
            let request = LabWorkflowRequest::CreateExperiment {
                request_id: "native-comparison".into(),
                session_id: session,
                draft: state.experiment_draft.clone(),
            };
            state.latest_experiment_creation = Some(LabExperimentCreation {
                request_id: "native-comparison".into(),
                session_id: session,
                draft: state.experiment_draft.clone(),
                state: LabExperimentCreationState::Pending,
            });
            let mut response = composer_test_response(&request);
            let LabWorkflowResponse::ExperimentCreated {
                request_id,
                session_id,
                experiment,
            } = &mut response
            else {
                unreachable!();
            };
            match mismatch {
                0 => *request_id = "another-request".into(),
                1 => *session_id = HostSessionId::new(2),
                2 => experiment.summary.experiment_id = "another-experiment".into(),
                3 => experiment.variants[0].brain_family = "assembly.baseline".into(),
                4 => {
                    experiment.summary.total_runs += 2;
                    for variant in &mut experiment.variants {
                        variant.total_runs += 1;
                    }
                }
                5 => experiment.variants[0].variant_id = "another-arm".into(),
                6 => {
                    let _ = experiment.variants.pop();
                }
                7 => experiment.variants.swap(0, 1),
                _ => unreachable!(),
            }
            deliver(&mut state, request, response);
            if mismatch == 7 {
                assert_eq!(state.selected_experiment.as_deref(), Some("native-comparison"));
                assert!(!state.composing_experiment, "valid cohort order is immaterial");
                assert_eq!(state.baseline.as_deref(), Some("baseline"));
                assert_eq!(state.challenger.as_deref(), Some("challenger"));
            } else {
                assert!(state.selected_experiment.is_none(), "mismatch {mismatch}");
                assert!(state.composing_experiment);
                assert!(matches!(
                    state.latest_experiment_creation.as_ref().unwrap().state,
                    LabExperimentCreationState::Retryable { .. }
                ));
            }
        }
    }

    #[test]
    fn endpoint_catalog_preserves_wire_keys_and_renders_the_observed_units() {
        const KEYS: [&str; LabMetric::ALL.len()] = LabMetric::keys();
        assert_eq!(
            KEYS,
            [
                "alive_agents",
                "total_energy",
                "average_energy",
                "average_health",
                "mean_age",
                "max_age",
                "max_generation",
                "total_food",
            ]
        );
        assert_eq!(
            KEYS.iter().collect::<std::collections::BTreeSet<_>>().len(),
            KEYS.len(),
            "each supported endpoint has a distinct wire identity"
        );
        assert_eq!(LabMetric::default(), LabMetric::AliveAgents);
        let expected = [
            ("Population", "agents"),
            ("Total live-agent energy", "energy units"),
            ("Mean live-agent energy", "energy units per agent"),
            ("Mean live-agent health", "health units"),
            ("Mean age counter", "age increments"),
            ("Maximum age counter", "age increments"),
            ("Maximum living generation", "generations"),
            ("Total food", "food energy units"),
        ];
        for (metric, (label, unit)) in LabMetric::ALL.into_iter().zip(expected) {
            let mut observed = comparison("arm-01");
            observed.metric = metric;
            observed.matched_seeds = vec![7, 11, 13];
            observed.descriptive_only = false;
            observed.ci_95 = Some((-1.5, 7.5));
            observed.underpowered_reason = None;
            observed.undefined_effect_reason = None;
            let lines = comparison_lines(&observed);
            assert_eq!(
                lines[0],
                format!("Verified {label} ({unit}), endpoint tick 2, 3 matched seed pairs")
            );
            assert_eq!(lines[1], format!("Challenger − baseline: 3.000000 {unit}"));
            assert_eq!(lines[2], format!("95% CI: [-1.500000, 7.500000] {unit}"));
            assert!(!lines.join("\n").contains("winner"));
        }
    }

    #[test]
    fn comparison_refuses_a_different_metric_even_with_matching_source_and_arms() {
        let mut state = LabWorkflows::default();
        state.synchronize(HostSessionId::new(1), LabPanel::Experiments);
        state.selected_experiment = Some("actual-batch".into());
        state.accept_batch(batch(2));
        for metric in LabMetric::ALL {
            state.dispatch(LabAction::Metric(metric));
            let request = state.comparison_request().unwrap();
            let mut exact = comparison("arm-01");
            exact.metric = metric;
            deliver(
                &mut state,
                request.clone(),
                LabWorkflowResponse::Comparison(exact.clone()),
            );
            assert_eq!(state.comparison.as_ref(), Some(&exact));

            let different = if metric == LabMetric::AliveAgents {
                LabMetric::TotalEnergy
            } else {
                LabMetric::AliveAgents
            };
            let mut wrong_endpoint = exact.clone();
            wrong_endpoint.metric = different;
            deliver(
                &mut state,
                request.clone(),
                LabWorkflowResponse::Comparison(wrong_endpoint),
            );
            assert!(
                state.comparison.is_none(),
                "another endpoint cannot retain or replace a verified {metric:?} comparison"
            );
            assert!(state.notice.as_deref().unwrap().contains("did not match"));

            deliver(
                &mut state,
                request,
                LabWorkflowResponse::Comparison(exact),
            );
            assert!(state.comparison.is_some());
            state.dispatch(LabAction::Metric(different));
            assert!(
                state.comparison.is_none(),
                "changing endpoint clears the previously verified result"
            );
        }
    }

    #[test]
    fn changing_metric_during_comparison_keeps_one_worker_and_discards_its_late_result() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        let (started, observed_start) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        let released = Mutex::new(released);
        let mut state = LabWorkflows {
            submit: Some(Arc::new(move |request| {
                let LabWorkflowRequest::Compare { metric, .. } = request else {
                    panic!("expected an endpoint comparison");
                };
                observed_calls.fetch_add(1, AtomicOrdering::SeqCst);
                started.send(metric).unwrap();
                released
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap();
                let mut result = comparison("arm-01");
                result.metric = metric;
                Ok(LabWorkflowResponse::Comparison(result))
            })),
            ..LabWorkflows::default()
        };
        state.synchronize(HostSessionId::new(1), LabPanel::Experiments);
        state.selected_experiment = Some("actual-batch".into());
        state.accept_batch(batch(2));
        state.dispatch(LabAction::Compare);
        assert_eq!(
            observed_start.recv_timeout(Duration::from_secs(5)).unwrap(),
            LabMetric::AliveAgents
        );
        state.dispatch(LabAction::Metric(LabMetric::TotalEnergy));
        state.dispatch(LabAction::Compare);
        assert!(state.is_busy());
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
        // Returning to the old endpoint must not revive its discarded request.
        state.dispatch(LabAction::Metric(LabMetric::AliveAgents));
        release.send(()).unwrap();
        drain_worker(&mut state);
        assert!(state.comparison.is_none());
        assert!(state.notice.as_deref().unwrap().contains("discarded"));

        state.dispatch(LabAction::Metric(LabMetric::AverageEnergy));
        state.dispatch(LabAction::Compare);
        assert_eq!(
            observed_start.recv_timeout(Duration::from_secs(5)).unwrap(),
            LabMetric::AverageEnergy
        );
        let context = state.context;
        state.dispatch(LabAction::Metric(LabMetric::AverageEnergy));
        assert_eq!(
            state.context, context,
            "reselecting the same endpoint preserves its pending request"
        );
        release.send(()).unwrap();
        drain_worker(&mut state);
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
        assert_eq!(
            state.comparison.as_ref().unwrap().metric,
            LabMetric::AverageEnergy,
            "a new request carries the selected endpoint after the old worker drains"
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "every projection variant and each refusal case, checked against one fixture"
    )]
    fn lab_projection_keeps_every_variant_and_refuses_mismatched_or_unbounded_evidence() {
        let mut state = LabWorkflows::default();
        state.synchronize(HostSessionId::new(1), LabPanel::Experiments);
        state.selected_experiment = Some("actual-batch".into());
        let request = LabWorkflowRequest::GetExperiment {
            experiment_id: "actual-batch".into(),
        };
        deliver(
            &mut state,
            request.clone(),
            LabWorkflowResponse::Experiment(batch(64)),
        );
        assert_eq!(state.batch.as_ref().unwrap().variants.len(), 64);
        state.dispatch(LabAction::Challenger("arm-63".into()));
        state.dispatch(LabAction::VariantsPage(usize::MAX));
        assert_eq!(
            state.variant_offset, 56,
            "variant pages stay within retained arms"
        );
        let compare = state.comparison_request().unwrap();
        deliver(
            &mut state,
            compare.clone(),
            LabWorkflowResponse::Comparison(comparison("arm-63")),
        );
        let lines = comparison_lines(state.comparison.as_ref().unwrap()).join("\n");
        assert!(lines.contains("Descriptive only"));
        assert!(lines.contains("p: unavailable"));
        assert!(lines.contains("3.000000 agents"));
        let mut mismatched = comparison("arm-63");
        mismatched.plan_digest = "another-plan".into();
        deliver(
            &mut state,
            compare.clone(),
            LabWorkflowResponse::Comparison(mismatched),
        );
        assert!(
            state.comparison.is_none(),
            "inconsistent source evidence invalidates an earlier displayed comparison"
        );
        let mut unsupported_inference = comparison("arm-63");
        unsupported_inference.p_value = Some(0.01);
        deliver(
            &mut state,
            compare.clone(),
            LabWorkflowResponse::Comparison(unsupported_inference),
        );
        assert!(
            state.comparison.is_none(),
            "one seed cannot acquire an inferential p value"
        );
        let mut unsupported_interval = comparison("arm-63");
        unsupported_interval.ci_95 = Some((1.0, 4.0));
        deliver(
            &mut state,
            compare.clone(),
            LabWorkflowResponse::Comparison(unsupported_interval),
        );
        assert!(
            state.comparison.is_none(),
            "one seed cannot acquire an inferential confidence interval"
        );
        deliver(
            &mut state,
            compare,
            LabWorkflowResponse::Comparison(comparison("arm-63")),
        );
        assert!(state.comparison.is_some());
        let (send, result) = std::sync::mpsc::sync_channel(1);
        assert!(
            send.send(Err("retained artifact verification failed".into()))
                .is_ok()
        );
        state.generation += 1;
        state.job = Some(LabJob {
            request: request.clone(),
            generation: state.generation,
            context: state.context,
            session: state.session.unwrap(),
            result,
        });
        state.poll();
        assert!(
            state.comparison.is_none(),
            "a known artifact refusal must clear prior verified output"
        );
        let compare = state.comparison_request().unwrap();
        deliver(
            &mut state,
            compare,
            LabWorkflowResponse::Comparison(comparison("arm-63")),
        );
        assert!(state.comparison.is_some());
        let mut changed = batch(64);
        changed.generation += 1;
        deliver(
            &mut state,
            request,
            LabWorkflowResponse::Experiment(changed),
        );
        assert!(
            state.comparison.is_none(),
            "a new source generation invalidates the displayed comparison"
        );
        state.selected_experiment = None;
        let rows = vec![batch(2).summary; LAB_PAGE_SIZE + 1];
        deliver(
            &mut state,
            LabWorkflowRequest::ListExperiments { cursor: None },
            LabWorkflowResponse::Experiments(LabExperimentPage {
                total: rows.len(),
                items: rows,
                offset: 0,
                next_cursor: None,
                reserved_run_slots: 0,
                max_run_slots: 1,
            }),
        );
        assert!(
            state.experiments.is_none(),
            "an oversized page is rejected, not silently truncated"
        );
        assert!(
            !page_is_bounded(&[1], 0, 2, None),
            "remaining rows require a cursor"
        );
        assert!(
            !page_is_bounded(&[1], 0, 2, Some("0")),
            "pagination must move forward"
        );
    }

    #[test]
    fn checkpoint_review_validates_the_saved_cut_and_discards_late_results_after_return_live() {
        use crate::{HostCore, HostCoreOptions};
        use scriptbots_core::{ScriptBotsConfig, WorldState};
        let world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(42),
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            closed: true,
            population_minimum: 0,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        let owner =
            HostCore::new(HostSessionId::new(1), world, HostCoreOptions::default()).unwrap();
        let frame = owner.snapshot_hub().latest();
        let mut state = LabWorkflows::default();
        state.synchronize(frame.session_id, LabPanel::Checkpoints);
        deliver(
            &mut state,
            LabWorkflowRequest::ListCheckpoints { cursor: None },
            LabWorkflowResponse::Checkpoints(LabCheckpointPage {
                items: vec![LabCheckpoint {
                    checkpoint_id: "saved-cut".into(),
                    tick: frame.world.tick,
                    agent_count: frame.world.agents.len(),
                    byte_size: 1,
                    world_digest: "saved-science".into(),
                }],
                total: 1,
                offset: 0,
                next_cursor: None,
                retained_bytes: 1,
                max_bytes: 100,
            }),
        );
        state.dispatch(LabAction::SelectCheckpoint("saved-cut".into()));
        let request = LabWorkflowRequest::ReviewCheckpoint {
            checkpoint_id: "saved-cut".into(),
        };
        let response = || LabWorkflowResponse::CheckpointFrame {
            checkpoint_id: "saved-cut".into(),
            snapshot: Arc::clone(&frame),
        };
        assert!(deliver(&mut state, request.clone(), response()).is_some());
        state.checkpoints.as_mut().unwrap().items[0].tick += 1;
        assert!(
            deliver(&mut state, request.clone(), response()).is_none(),
            "a different cut cannot be presented as the selected checkpoint"
        );
        state.checkpoints.as_mut().unwrap().items[0].tick -= 1;
        state.checkpoints.as_mut().unwrap().items[0].agent_count += 1;
        assert!(
            deliver(&mut state, request.clone(), response()).is_none(),
            "a different saved population is not the catalog cut"
        );
        state.checkpoints.as_mut().unwrap().items[0].agent_count -= 1;
        let (send, result) = std::sync::mpsc::sync_channel(1);
        state.generation += 1;
        state.job = Some(LabJob {
            request,
            generation: state.generation,
            context: state.context,
            session: frame.session_id,
            result,
        });
        state.dispatch(LabAction::ReturnLive);
        assert!(state.is_busy());
        assert!(send.send(Ok(response())).is_ok());
        assert!(
            state.poll().is_none(),
            "return-to-live invalidates a late reconstructed frame"
        );
        assert!(
            Arc::ptr_eq(&owner.snapshot_hub().latest(), &frame),
            "review never advances the actual owner"
        );
        state.reset_observations();
        assert!(state.checkpoints.is_none() && state.selected_checkpoint.is_none());
        assert!(state.submit.is_none());
        assert_eq!(state.session, Some(frame.session_id));
    }

    // Synthetic protocol evidence isolates source, identity and completion guards.
    // Application tests separately reconstruct a real production detector event.
    fn event_protocol_fixture() -> (
        Arc<RenderSnapshot>,
        LabEventIdentity,
        scriptbots_core::narrative::EventRecord,
    ) {
        let world = scriptbots_core::WorldState::new(scriptbots_core::ScriptBotsConfig {
            rng_seed: Some(42),
            world_width: 40,
            world_height: 40,
            food_cell_size: 10,
            population_minimum: 0,
            persistence_interval: 0,
            ..scriptbots_core::ScriptBotsConfig::default()
        })
        .unwrap();
        let frame = crate::HostCore::new(
            HostSessionId::new(17),
            world,
            crate::HostCoreOptions::default(),
        )
        .unwrap()
        .latest_snapshot();
        let event = LabEventIdentity {
            session_id: frame.session_id,
            ordinal: 41,
            tick: frame.world.tick,
        };
        let record = scriptbots_core::narrative::EventRecord {
            schema_version: 1,
            tick: scriptbots_core::Tick(event.tick),
            kind: scriptbots_core::narrative::EventKind::Extinction,
            severity: 1.0,
            magnitude: 20.0,
            window: (0, event.tick),
            metric: "population".into(),
            before: 20.0,
            after: 0.0,
            score: 5.0,
            subject: None,
            human_text: "protocol fixture: population reached zero".into(),
        };
        (frame, event, record)
    }

    #[test]
    fn event_review_requires_the_selected_evidence_owner_target_and_source_checkpoint() {
        let (frame, event, record) = event_protocol_fixture();
        let mut state = LabWorkflows::default();
        state.synchronize(event.session_id, LabPanel::Narrative);
        state.select_event(Some((event, &record)));
        let context = state.context;
        state.select_event(Some((event, &record)));
        assert_eq!(
            state.context, context,
            "unchanged evidence stays reviewable"
        );
        let response = || LabEventReview {
            event,
            record: record.clone(),
            checkpoint_id: "actual-retained-cut".into(),
            base_tick: 0,
            snapshot: Arc::clone(&frame),
        };
        let request = LabWorkflowRequest::ReviewEvent { event };
        assert!(
            deliver(
                &mut state,
                request.clone(),
                LabWorkflowResponse::EventFrame(response())
            )
            .is_none()
        );
        let accepted = state.take_event_review().expect("exact selected event");
        assert_eq!(accepted.event, event);
        assert_eq!(accepted.record, record);
        assert!(Arc::ptr_eq(&accepted.snapshot, &frame));
        assert!(
            state.take_event_review().is_none(),
            "one result is consumed once"
        );

        for mismatch in 0..8 {
            let mut changed = response();
            match mismatch {
                0 => changed.event.ordinal += 1,
                1 => Arc::make_mut(&mut changed.snapshot).session_id = HostSessionId::new(99),
                2 => Arc::make_mut(&mut changed.snapshot).world.tick += 1,
                3 => changed.base_tick = event.tick + 1,
                4 => changed.record.before += 1.0,
                5 => changed.checkpoint_id.clear(),
                6 => changed.record.tick.0 += 1,
                7 => changed.event.tick += 1,
                _ => unreachable!(),
            }
            assert!(
                deliver(
                    &mut state,
                    request.clone(),
                    LabWorkflowResponse::EventFrame(changed)
                )
                .is_none()
            );
            assert!(
                state.take_event_review().is_none(),
                "mismatch {mismatch} must be refused"
            );
        }
    }

    #[test]
    fn event_review_discards_late_results_after_every_selection_or_live_view_change() {
        let (frame, event, record) = event_protocol_fixture();
        for change in 0..6 {
            let mut state = LabWorkflows::default();
            state.synchronize(event.session_id, LabPanel::Narrative);
            state.select_event(Some((event, &record)));
            let (send, result) = std::sync::mpsc::sync_channel(1);
            state.generation += 1;
            state.job = Some(LabJob {
                request: LabWorkflowRequest::ReviewEvent { event },
                generation: state.generation,
                context: state.context,
                session: event.session_id,
                result,
            });
            match change {
                0 => state.dispatch(LabAction::ReturnLive),
                1 => state.select_event(None),
                2 => {
                    let mut replacement = record.clone();
                    replacement.human_text = "different evidence at the same identity".into();
                    state.select_event(Some((event, &replacement)));
                }
                3 => state.synchronize(HostSessionId::new(99), LabPanel::Narrative),
                4 => state.synchronize(event.session_id, LabPanel::Checkpoints),
                5 => state.select_event(Some((
                    LabEventIdentity {
                        ordinal: event.ordinal + 1,
                        ..event
                    },
                    &record,
                ))),
                _ => unreachable!(),
            }
            assert!(
                state.is_busy(),
                "change {change} cannot release an occupied worker"
            );
            assert!(
                send.send(Ok(LabWorkflowResponse::EventFrame(LabEventReview {
                    event,
                    record: record.clone(),
                    checkpoint_id: "actual-retained-cut".into(),
                    base_tick: 0,
                    snapshot: Arc::clone(&frame),
                })))
                .is_ok()
            );
            assert!(state.poll().is_none());
            assert!(!state.is_busy());
            assert!(
                state.take_event_review().is_none(),
                "change {change} invalidates the old result"
            );
        }
    }

    #[test]
    fn event_review_refuses_stale_actions_before_starting_a_worker() {
        let (_, event, record) = event_protocol_fixture();
        let mut state = LabWorkflows {
            submit: Some(Arc::new(|_| Err("unexpected stale action".into()))),
            ..LabWorkflows::default()
        };
        state.synchronize(event.session_id, LabPanel::Narrative);
        state.select_event(Some((event, &record)));
        for stale in [
            LabEventIdentity {
                session_id: HostSessionId::new(99),
                ..event
            },
            LabEventIdentity {
                ordinal: event.ordinal + 1,
                ..event
            },
            LabEventIdentity {
                tick: event.tick + 1,
                ..event
            },
        ] {
            state.dispatch(LabAction::ReviewEvent(stale));
            assert!(!state.is_busy());
        }
        state.select_event(None);
        state.dispatch(LabAction::ReviewEvent(event));
        assert!(
            !state.is_busy(),
            "evicted evidence cannot be reconstructed by a stale button"
        );
    }

    #[test]
    fn narrative_eviction_preserves_a_pending_checkpoint_save_in_its_own_panel() {
        let (_, event, record) = event_protocol_fixture();
        let (started, observed_start) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        let released = Mutex::new(released);
        let mut state = LabWorkflows {
            submit: Some(Arc::new(move |request| {
                started.send(()).unwrap();
                released
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap();
                Ok(saved_response(&request, saved_checkpoint()))
            })),
            ..LabWorkflows::default()
        };
        state.synchronize(event.session_id, LabPanel::Narrative);
        state.select_event(Some((event, &record)));
        state.synchronize(event.session_id, LabPanel::Checkpoints);
        state.dispatch(LabAction::CreateCheckpoint);
        observed_start.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(state.is_busy());
        state.select_event(None);
        assert!(
            state.selected_event.is_none(),
            "eviction still updates cached evidence"
        );
        assert!(matches!(
            state.latest_checkpoint_save.as_ref().unwrap().state,
            LabCheckpointSaveState::Pending
        ));
        release.send(()).unwrap();
        drain_worker(&mut state);
        assert_eq!(
            state.latest_checkpoint_save.as_ref().unwrap().state,
            LabCheckpointSaveState::Saved(saved_checkpoint())
        );
        assert_eq!(state.selected_checkpoint.as_deref(), Some("retained-cut"));
    }

    fn saved_checkpoint() -> LabCheckpoint {
        LabCheckpoint {
            checkpoint_id: "retained-cut".into(),
            tick: 42,
            agent_count: 3,
            byte_size: 1024,
            world_digest: "a".repeat(std::mem::size_of::<u64>() * 2),
        }
    }

    #[test]
    fn checkpoint_receipt_accepts_the_actual_world_digest_format() {
        let world = scriptbots_core::WorldState::new(scriptbots_core::ScriptBotsConfig {
            rng_seed: Some(42),
            world_width: 40,
            world_height: 40,
            food_cell_size: 10,
            population_minimum: 0,
            persistence_interval: 0,
            ..scriptbots_core::ScriptBotsConfig::default()
        })
        .unwrap();
        let digest = world.world_digest_v1().unwrap();
        digest.validate_contract().unwrap();
        let mut checkpoint = saved_checkpoint();
        checkpoint.world_digest = digest.overall;
        assert!(checkpoint_receipt_is_bounded(&checkpoint));

        for invalid in [
            String::new(),
            "a".repeat(15),
            "a".repeat(17),
            "a".repeat(64),
            "A".repeat(16),
            "g".repeat(16),
        ] {
            checkpoint.world_digest = invalid;
            assert!(
                !checkpoint_receipt_is_bounded(&checkpoint),
                "a malformed digest or artifact checksum is not a scientific state digest"
            );
        }
    }

    fn saved_response(
        request: &LabWorkflowRequest,
        checkpoint: LabCheckpoint,
    ) -> LabWorkflowResponse {
        let LabWorkflowRequest::CreateCheckpoint {
            request_id,
            session_id,
        } = request
        else {
            panic!("expected checkpoint creation request");
        };
        LabWorkflowResponse::CheckpointCreated {
            request_id: request_id.clone(),
            session_id: *session_id,
            checkpoint,
        }
    }

    fn drain_worker(state: &mut LabWorkflows) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while state.is_busy() {
            assert!(Instant::now() < deadline, "Lab worker timed out");
            assert!(state.poll().is_none());
            std::thread::yield_now();
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one save lifecycle: retry identity and worker capacity held until verification"
    )]
    fn checkpoint_saves_retain_retry_identity_and_worker_capacity_until_verified() {
        let (started, observed_start) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        let released = Mutex::new(released);
        let mut state = LabWorkflows {
            submit: Some(Arc::new(move |request| {
                started.send(request).unwrap();
                released
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap()
            })),
            ..LabWorkflows::default()
        };
        state.dispatch(LabAction::CreateCheckpoint);
        assert!(
            !state.is_busy() && state.latest_checkpoint_save.is_none(),
            "an unidentified source cannot save"
        );
        let session = HostSessionId::new(7);
        state.synchronize(session, LabPanel::Experiments);
        state.dispatch(LabAction::CreateCheckpoint);
        assert!(
            !state.is_busy(),
            "a stale checkpoint button cannot submit from another panel"
        );
        state.synchronize(session, LabPanel::Checkpoints);
        state.dispatch(LabAction::CreateCheckpoint);
        let original = observed_start.recv_timeout(Duration::from_secs(5)).unwrap();
        let original_id = state
            .latest_checkpoint_save
            .as_ref()
            .unwrap()
            .request_id
            .clone();
        assert!(checkpoint_request_identity_is_valid(&original_id));
        assert!(!state.can_create_checkpoint());
        assert!(matches!(
            state.latest_checkpoint_save.as_ref().unwrap().state,
            LabCheckpointSaveState::Pending
        ));
        state.dispatch(LabAction::CancelRequest);
        state.dispatch(LabAction::CreateCheckpoint);
        assert!(
            observed_start.try_recv().is_err(),
            "cancellation must not free the sole worker"
        );
        state.synchronize(session, LabPanel::Inactive);
        state.synchronize(session, LabPanel::Checkpoints);
        assert!(
            release
                .send(Ok(saved_response(&original, saved_checkpoint())))
                .is_ok()
        );
        drain_worker(&mut state);
        assert!(state.will_retry_checkpoint_save());
        assert!(
            state.selected_checkpoint.is_none(),
            "a discarded receipt cannot select a saved cut"
        );
        state.synchronize(HostSessionId::new(8), LabPanel::Checkpoints);
        assert!(state.notice.as_deref().unwrap().contains("fresh identity"));
        state.synchronize(session, LabPanel::Checkpoints);
        assert!(state.will_retry_checkpoint_save());
        assert!(
            state.notice.is_none(),
            "reattaching the original source must not promise a fresh capture"
        );

        state.dispatch(LabAction::CreateCheckpoint);
        let retry = observed_start.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(
            retry, original,
            "discarding a completed result must retry the same logical save"
        );
        assert!(
            release
                .send(Err("unconfirmed receipt ".repeat(300)))
                .is_ok()
        );
        drain_worker(&mut state);
        let LabCheckpointSaveState::Retryable { reason } =
            &state.latest_checkpoint_save.as_ref().unwrap().state
        else {
            panic!("a service error must retain a retryable save");
        };
        assert!(reason.chars().count() <= 2048);
        state.dispatch(LabAction::CreateCheckpoint);
        let retry = observed_start.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(
            retry, original,
            "a service error must not allocate another capture identity"
        );
        assert!(
            release
                .send(Ok(saved_response(&retry, saved_checkpoint())))
                .is_ok()
        );
        drain_worker(&mut state);
        assert!(!state.will_retry_checkpoint_save());
        assert_eq!(
            state.selected_checkpoint_metadata(),
            Some(&saved_checkpoint())
        );

        state.dispatch(LabAction::CreateCheckpoint);
        let later = observed_start.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_ne!(
            later, original,
            "a verified save permits a distinct requested cut"
        );
        state.synchronize(HostSessionId::new(8), LabPanel::Checkpoints);
        assert!(
            !state.can_create_checkpoint(),
            "changing source retains the occupied worker"
        );
        assert!(
            release
                .send(Ok(saved_response(&later, saved_checkpoint())))
                .is_ok()
        );
        drain_worker(&mut state);
        assert!(
            state.can_create_checkpoint(),
            "an old session's ambiguous result cannot lock out a newly attached owner"
        );
        assert!(!state.will_retry_checkpoint_save());
        state.dispatch(LabAction::CreateCheckpoint);
        let new_source = observed_start.recv_timeout(Duration::from_secs(5)).unwrap();
        let LabWorkflowRequest::CreateCheckpoint {
            request_id,
            session_id,
        } = &new_source
        else {
            panic!("wrong request kind");
        };
        assert_eq!(*session_id, HostSessionId::new(8));
        assert_ne!(request_id, &original_id);
        assert!(
            release
                .send(Ok(saved_response(&new_source, saved_checkpoint())))
                .is_ok()
        );
        drain_worker(&mut state);
        assert!(matches!(
            state.latest_checkpoint_save.as_ref().unwrap().state,
            LabCheckpointSaveState::Saved(_)
        ));
        assert_ne!(
            next_checkpoint_request(session).unwrap(),
            original_id,
            "another model in this process must not reuse a frontend's save identity"
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one receipt lifecycle: identity, source and cut checks before review opens"
    )]
    fn checkpoint_receipts_validate_identity_source_and_cut_before_enabling_review() {
        let session = HostSessionId::new(1);
        let request = LabWorkflowRequest::CreateCheckpoint {
            request_id: "bounded-save".into(),
            session_id: session,
        };
        let mut state = LabWorkflows::default();
        state.synchronize(session, LabPanel::Checkpoints);
        state.latest_checkpoint_save = Some(LabCheckpointSave {
            request_id: "bounded-save".into(),
            session_id: session,
            state: LabCheckpointSaveState::Pending,
        });
        for (request_id, session_id) in [
            ("wrong-save", session),
            ("bounded-save", HostSessionId::new(2)),
        ] {
            deliver(
                &mut state,
                request.clone(),
                LabWorkflowResponse::CheckpointCreated {
                    request_id: request_id.into(),
                    session_id,
                    checkpoint: saved_checkpoint(),
                },
            );
            assert!(
                state.selected_checkpoint_metadata().is_none(),
                "a mismatched receipt must not authorize review"
            );
            assert!(state.will_retry_checkpoint_save());
        }
        let mut invalid_cuts = Vec::new();
        let mut checkpoint = saved_checkpoint();
        checkpoint.checkpoint_id.clear();
        invalid_cuts.push(checkpoint);
        let mut checkpoint = saved_checkpoint();
        checkpoint.byte_size = 0;
        invalid_cuts.push(checkpoint);
        let mut checkpoint = saved_checkpoint();
        checkpoint.byte_size = scriptbots_core::MAX_WORLD_CHECKPOINT_BYTES as u64 + 1;
        invalid_cuts.push(checkpoint);
        let mut checkpoint = saved_checkpoint();
        checkpoint.agent_count = usize::MAX;
        invalid_cuts.push(checkpoint);
        let mut checkpoint = saved_checkpoint();
        checkpoint.world_digest = "A".repeat(16);
        invalid_cuts.push(checkpoint);
        let mut checkpoint = saved_checkpoint();
        checkpoint.world_digest = "a".repeat(64);
        invalid_cuts.push(checkpoint);
        for checkpoint in invalid_cuts {
            deliver(
                &mut state,
                request.clone(),
                saved_response(&request, checkpoint),
            );
            assert!(
                state.selected_checkpoint_metadata().is_none(),
                "an invalid saved cut must not become a review source"
            );
            assert!(state.will_retry_checkpoint_save());
        }
        deliver(
            &mut state,
            request.clone(),
            saved_response(&request, saved_checkpoint()),
        );
        assert_eq!(
            state.selected_checkpoint_metadata(),
            Some(&saved_checkpoint())
        );
        deliver(
            &mut state,
            LabWorkflowRequest::ListCheckpoints { cursor: None },
            LabWorkflowResponse::Checkpoints(LabCheckpointPage {
                items: Vec::new(),
                total: 0,
                offset: 0,
                next_cursor: None,
                retained_bytes: 1024,
                max_bytes: 4096,
            }),
        );
        assert_eq!(
            state.selected_checkpoint_metadata(),
            Some(&saved_checkpoint()),
            "catalog paging cannot hide a verified just-saved cut from immediate review"
        );
        state.synchronize(HostSessionId::new(2), LabPanel::Checkpoints);
        state.dispatch(LabAction::SelectCheckpoint("retained-cut".into()));
        assert!(
            state.selected_checkpoint_metadata().is_none(),
            "a prior-session receipt is informational and cannot authorize review on another owner"
        );
        assert!(state.selected_checkpoint.is_none());
        state.synchronize(session, LabPanel::Checkpoints);
        state.dispatch(LabAction::SelectCheckpoint("retained-cut".into()));
        assert_eq!(
            state.selected_checkpoint_metadata(),
            Some(&saved_checkpoint()),
            "reattaching the original owner can recover its verified receipt"
        );
        for invalid in [String::new(), "a".repeat(257), "line\nbreak".into()] {
            assert!(!checkpoint_request_identity_is_valid(&invalid));
        }
    }

    #[test]
    fn checkpoint_sequence_reservations_are_unique_and_refuse_concurrent_exhaustion() {
        let initial = u64::MAX - 3;
        let counter = AtomicU64::new(initial);
        let contenders = 8;
        let start = std::sync::Barrier::new(contenders);
        let mut reserved = std::thread::scope(|scope| {
            let workers = (0..contenders)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        reserve_checkpoint_sequence(&counter)
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .filter_map(|worker| worker.join().expect("reservation worker"))
                .collect::<Vec<_>>()
        });
        reserved.sort_unstable();
        assert_eq!(
            reserved,
            (initial..u64::MAX).collect::<Vec<_>>(),
            "each remaining identity must be reserved exactly once under contention"
        );
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        assert!(reserve_checkpoint_sequence(&counter).is_none());
        assert!(reserve_checkpoint_sequence(&counter).is_none());
        assert_eq!(
            counter.load(Ordering::Relaxed),
            u64::MAX,
            "exhaustion must never wrap or reopen the identity domain"
        );
    }
}
