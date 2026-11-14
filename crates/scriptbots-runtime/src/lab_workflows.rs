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
        if matches!(&job.request, LabWorkflowRequest::CreateCheckpoint { .. }) {
            self.finish_checkpoint_save(&job.request, result);
            return None;
        }
        let response = match result {
            Ok(response) => response,
            Err(error) => {
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

    /// Apply one native UI intent, preserving request identity and bounded state.
    pub fn dispatch(&mut self, action: LabAction) {
        match action {
            LabAction::ReturnLive => self.invalidate(),
            LabAction::CancelRequest => {
                self.invalidate();
                self.set_notice("Result cancelled. The worker remains occupied until it finishes; submitted actions may still complete. Unconfirmed checkpoint saves retain their retry identity.");
            }
            LabAction::SelectExperiment(id) => self.select_experiment(id),
            LabAction::AllExperiments => {
                self.invalidate();
                self.selected_experiment = None;
                self.batch = None;
                self.comparison = None;
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
