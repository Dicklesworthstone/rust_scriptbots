//! Renderer-independent consumers of application-owned Lab services.
//!
//! A single bounded worker performs application reads and requests. Frontends
//! render these observations and apply returned checkpoint frames only to their
//! presentation; this module never owns a world, database or experiment runner.

use crate::{HostSessionId, RenderSnapshot};
use std::{
    sync::{
        Arc,
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
    },
    /// Read one checkpoint catalog page.
    ListCheckpoints {
        /// Application page cursor.
        cursor: Option<String>,
    },
    /// Reconstruct a saved cut without changing the live owner.
    ReviewCheckpoint {
        /// Stable retained checkpoint identity.
        checkpoint_id: String,
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

/// Source-bound paired population endpoint comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct LabComparison {
    /// Stable experiment identity.
    pub experiment_id: String,
    /// Reference arm.
    pub baseline_variant: String,
    /// Arm whose observations are compared with the reference.
    pub challenger_variant: String,
    /// Common observed endpoint.
    pub endpoint_tick: u64,
    /// Ordered unique independent seed pairs.
    pub matched_seeds: Vec<u64>,
    /// Challenger minus baseline mean, in agents.
    pub mean_difference: f64,
    /// Optional confidence interval.
    pub ci_95: Option<(f64, f64)>,
    /// Optional paired Cohen's d_z.
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
    /// Reconstructed immutable frame, never a live-owner replacement.
    CheckpointFrame {
        /// Requested checkpoint identity.
        checkpoint_id: String,
        /// Reconstructed scientific presentation.
        snapshot: Arc<RenderSnapshot>,
    },
}

/// Application adapter invoked on the model's bounded background worker.
pub type LabWorkflowSubmit = Arc<
    dyn Fn(LabWorkflowRequest) -> Result<LabWorkflowResponse, String> + Send + Sync,
>;

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
    /// Request a presentation-only reconstruction.
    ReviewCheckpoint,
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
    /// First visible variant row.
    pub variant_offset: usize,
    /// Last accepted comparison for the current plan and status generation.
    pub comparison: Option<LabComparison>,
    /// Last accepted checkpoint page.
    pub checkpoints: Option<LabCheckpointPage>,
    checkpoint_cursor: Option<String>,
    /// Selected retained cut.
    pub selected_checkpoint: Option<String>,
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

fn page_is_bounded<T>(items: &[T], offset: usize, total: usize, next: &Option<String>) -> bool {
    items.len() <= LAB_PAGE_SIZE
        && offset <= total
        && items.len() <= total.saturating_sub(offset)
        && next.as_ref().map_or(offset + items.len() == total, |cursor| {
            cursor.len() <= 20
                && cursor.parse::<usize>().is_ok_and(|next| {
                    next == offset + items.len() && next > offset && next < total
                })
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
        && batch.variants.iter().map(|row| row.total_runs).sum::<usize>() == batch.summary.total_runs
        && batch.variants.iter().map(|row| row.completed_runs).sum::<usize>() == batch.summary.completed_runs
        && batch.variants.iter().map(|row| row.failed_runs).sum::<usize>() == batch.summary.failed_runs
        && batch.variants.iter().map(|row| &row.variant_id).collect::<std::collections::BTreeSet<_>>().len() == batch.variants.len()
}

impl LabWorkflows {
    /// Invalidate pending results without releasing an occupied worker slot.
    pub fn invalidate(&mut self) {
        // Saturation cannot resurrect an old generation: request admission also
        // refuses when this counter is exhausted.
        self.context = self.context.saturating_add(1);
        self.last_refresh = None;
    }

    /// Bind observations and future work to the actual owner and visible panel.
    pub fn synchronize(&mut self, session: HostSessionId, panel: LabPanel) {
        if self.session != Some(session) {
            self.reset_observations();
            self.session = Some(session);
        }
        if self.panel != panel {
            self.invalidate();
            self.panel = panel;
        }
    }

    fn start(&mut self, request: LabWorkflowRequest) {
        if self.job.is_some() {
            self.notice = Some("A Lab request is still running; its worker slot remains occupied.".into());
            return;
        }
        let Some(submit) = self.submit.clone() else {
            self.notice = Some("Application experiment and checkpoint services are unavailable.".into());
            self.last_refresh = Some(Instant::now());
            return;
        };
        let Some(session) = self.session else { return };
        let Some(generation) = self.generation.checked_add(1).filter(|_| self.context < u64::MAX) else {
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
            })
        {
            Ok(_) => {
                self.generation = generation;
                self.job = Some(LabJob { request, generation, context: self.context, session, result });
                self.notice = Some("Loading application evidence…".into());
                self.last_refresh = Some(Instant::now());
            }
            Err(error) => {
                self.notice = Some(format!("Lab worker unavailable: {error}"));
                self.last_refresh = Some(Instant::now());
            }
        }
    }

    /// Selected metadata from the current verified page, if still present.
    #[must_use]
    pub fn selected_checkpoint_metadata(&self) -> Option<&LabCheckpoint> {
        let selected = self.selected_checkpoint.as_deref()?;
        self.checkpoints.as_ref()?.items.iter().find(|row| row.checkpoint_id == selected)
    }

    /// Current request for two distinct admitted variants.
    #[must_use]
    pub fn comparison_request(&self) -> Option<LabWorkflowRequest> {
        let batch = self.batch.as_ref()?;
        let baseline = self.baseline.as_ref()?;
        let challenger = self.challenger.as_ref()?;
        if baseline == challenger
            || !batch.variants.iter().any(|row| &row.variant_id == baseline)
            || !batch.variants.iter().any(|row| &row.variant_id == challenger)
        {
            return None;
        }
        Some(LabWorkflowRequest::Compare {
            experiment_id: batch.summary.experiment_id.clone(),
            baseline_variant: baseline.clone(),
            challenger_variant: challenger.clone(),
        })
    }

    /// Consume one worker result without blocking the frontend.
    ///
    /// Only an accepted, still-selected checkpoint returns a frame. The frontend
    /// must honor any newer local live-view intent before displaying it.
    pub fn poll(&mut self) -> Option<(String, Arc<RenderSnapshot>)> {
        let job = self.job.as_ref()?;
        let result = match job.result.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => Err("Lab worker disconnected".to_owned()),
        };
        let job = self.job.take()?;
        if job.generation != self.generation || job.context != self.context || self.session != Some(job.session) {
            self.notice = Some("Lab request finished after its selection or session changed; result discarded.".into());
            return None;
        }
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                if matches!(&job.request,
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
        let accepted = match (&job.request, response) {
            (LabWorkflowRequest::ListExperiments { cursor }, LabWorkflowResponse::Experiments(page))
                if cursor == &self.experiment_cursor
                    && page.offset == cursor.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0)
                    && page_is_bounded(&page.items, page.offset, page.total, &page.next_cursor)
                    && page.items.iter().all(summary_is_bounded) => {
                self.experiments = Some(page);
                true
            }
            (LabWorkflowRequest::GetExperiment { experiment_id }
                | LabWorkflowRequest::CancelExperiment { experiment_id }
                | LabWorkflowRequest::ResumeExperiment { experiment_id }, LabWorkflowResponse::Experiment(batch))
                if self.selected_experiment.as_ref() == Some(experiment_id)
                    && batch.summary.experiment_id == *experiment_id
                    && batch_is_bounded(&batch) => {
                self.accept_batch(batch);
                true
            }
            (LabWorkflowRequest::Compare { experiment_id, baseline_variant, challenger_variant }, LabWorkflowResponse::Comparison(comparison))
                if self.comparison_request().as_ref() == Some(&job.request)
                    && comparison.experiment_id == *experiment_id
                    && comparison.baseline_variant == *baseline_variant
                    && comparison.challenger_variant == *challenger_variant
                    && self.batch.as_ref().is_some_and(|batch| batch.plan_digest == comparison.plan_digest && batch.generation == comparison.status_generation)
                    && comparison_is_bounded(&comparison) => {
                self.comparison = Some(comparison);
                true
            }
            (LabWorkflowRequest::ListCheckpoints { cursor }, LabWorkflowResponse::Checkpoints(page))
                if cursor == &self.checkpoint_cursor
                    && page.offset == cursor.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0)
                    && page_is_bounded(&page.items, page.offset, page.total, &page.next_cursor)
                    && page.items.iter().all(|row| identity_is_bounded(&row.checkpoint_id) && row.world_digest.len() <= 128) => {
                if self.selected_checkpoint.as_ref().is_some_and(|id| !page.items.iter().any(|row| &row.checkpoint_id == id)) {
                    self.selected_checkpoint = None;
                }
                self.checkpoints = Some(page);
                true
            }
            (LabWorkflowRequest::ReviewCheckpoint { checkpoint_id }, LabWorkflowResponse::CheckpointFrame { checkpoint_id: returned_id, snapshot })
                if self.panel == LabPanel::Checkpoints
                    && checkpoint_id == &returned_id
                    && self.selected_checkpoint_metadata().is_some_and(|row| row.checkpoint_id == returned_id && row.tick == snapshot.world.tick && row.agent_count == snapshot.world.agents.len()) => {
                self.notice = Some(format!("Verified checkpoint {returned_id} at tick {}.", snapshot.world.tick));
                return Some((returned_id, snapshot));
            }
            _ => false,
        };
        if !accepted && matches!(&job.request,
            LabWorkflowRequest::GetExperiment { .. }
                | LabWorkflowRequest::CancelExperiment { .. }
                | LabWorkflowRequest::ResumeExperiment { .. }
                | LabWorkflowRequest::Compare { .. }
        ) {
            self.comparison = None;
        }
        self.notice = Some(if accepted {
            "Application evidence refreshed.".into()
        } else {
            "Lab response changed, exceeded its bounds, or did not match the request; refresh before retrying.".into()
        });
        None
    }

    fn refresh_request(&self) -> Option<LabWorkflowRequest> {
        match self.panel {
            LabPanel::Inactive => None,
            LabPanel::Experiments => Some(match &self.selected_experiment {
                Some(id) => LabWorkflowRequest::GetExperiment { experiment_id: id.clone() },
                None => LabWorkflowRequest::ListExperiments { cursor: self.experiment_cursor.clone() },
            }),
            LabPanel::Checkpoints => Some(LabWorkflowRequest::ListCheckpoints { cursor: self.checkpoint_cursor.clone() }),
        }
    }

    /// Refresh visible changing evidence at most once every two seconds.
    pub fn maybe_refresh(&mut self) {
        if self.job.is_some() || self.submit.is_none() || self.last_refresh.is_some_and(|last| last.elapsed() < REFRESH_INTERVAL) {
            return;
        }
        let needs_refresh = match self.panel {
            LabPanel::Inactive => false,
            LabPanel::Checkpoints => self.checkpoints.is_none(),
            LabPanel::Experiments => self.selected_experiment.is_none()
                || self.batch.as_ref().is_none_or(|batch| matches!(batch.summary.status.as_str(), "running" | "pending" | "cancelling")),
        };
        if needs_refresh && let Some(request) = self.refresh_request() {
            self.start(request);
        }
    }

    fn accept_batch(&mut self, mut batch: LabExperiment) {
                batch.error_reason = batch.error_reason.as_deref().map(bounded_message);
                if self.comparison.as_ref().is_some_and(|comparison| {
                    comparison.plan_digest != batch.plan_digest || comparison.status_generation != batch.generation
                }) {
                    self.comparison = None;
                }
                if self.baseline.as_ref().is_none_or(|id| !batch.variants.iter().any(|row| &row.variant_id == id)) {
                    self.baseline = batch.variants.first().map(|row| row.variant_id.clone());
                }
                if self.challenger.as_ref().is_none_or(|id| !batch.variants.iter().any(|row| &row.variant_id == id)) {
                    self.challenger = batch.variants.iter().find(|row| Some(&row.variant_id) != self.baseline.as_ref()).map(|row| row.variant_id.clone());
                }
                self.variant_offset = self.variant_offset.min(batch.variants.len().saturating_sub(1) / VARIANTS_PER_PAGE * VARIANTS_PER_PAGE);
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

    /// Clear observations after an owner epoch reset, preserving any occupied
    /// worker slot, application adapter and actual session identity.
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
        self.notice = None;
        self.variant_offset = 0;
    }

    /// Apply one native UI intent, preserving request identity and bounded state.
    pub fn dispatch(&mut self, action: LabAction) {
        match action {
            LabAction::ReturnLive => self.invalidate(),
            LabAction::CancelRequest => {
                self.invalidate();
                self.set_notice("Result cancelled. The worker remains occupied until it finishes; already submitted cancel/resume actions still take effect.");
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
                if self.checkpoints.as_ref().is_some_and(|page| page.items.iter().any(|row| row.checkpoint_id == id)) {
                    self.invalidate();
                    self.selected_checkpoint = Some(id);
                }
            }
            LabAction::Baseline(ref id) | LabAction::Challenger(ref id) => {
                let baseline = matches!(&action, LabAction::Baseline(_));
                if self.batch.as_ref().is_some_and(|batch| batch.variants.iter().any(|row| &row.variant_id == id)) {
                    self.invalidate();
                    if baseline { self.baseline = Some(id.clone()); }
                    else { self.challenger = Some(id.clone()); }
                    self.comparison = None;
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
                    } else { LabWorkflowRequest::ResumeExperiment { experiment_id: id } };
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
            LabAction::Refresh => {
                if let Some(request) = self.refresh_request() { self.start(request); }
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
    !comparison.matched_seeds.is_empty() && comparison.matched_seeds.len() <= 128
                    && comparison.matched_seeds.windows(2).all(|pair| pair[0] < pair[1])
                    && comparison.mean_difference.is_finite()
                    && comparison.ci_95.is_none_or(|(low, high)| low.is_finite() && high.is_finite() && low <= high)
                    && comparison.standardized_effect.is_none_or(f64::is_finite)
                    && comparison.p_value.is_none_or(|p| p.is_finite() && (0.0..=1.0).contains(&p))
                    && comparison.adjusted_p_value.is_none_or(|p| p.is_finite() && (0.0..=1.0).contains(&p))
                    && (comparison.matched_seeds.len() > 1 || (comparison.descriptive_only && comparison.ci_95.is_none() && comparison.standardized_effect.is_none() && comparison.p_value.is_none() && comparison.adjusted_p_value.is_none()))
}

/// Human-readable statistical observations shared by native Lab frontends.
#[must_use]
pub fn comparison_lines(comparison: &LabComparison) -> Vec<String> {
    let optional = |value: Option<f64>| value.map_or_else(|| "unavailable".into(), |value| format!("{value:.6}"));
    let mut lines = vec![
        format!("Verified population (agents), endpoint tick {}, {} matched seed pairs", comparison.endpoint_tick, comparison.matched_seeds.len()),
        format!("Challenger − baseline: {:.6} agents", comparison.mean_difference),
        format!("95% CI: {}", comparison.ci_95.map_or_else(|| "unavailable".into(), |(low, high)| format!("[{low:.6}, {high:.6}]"))),
        format!("p: {} · adjusted p: {} · paired Cohen's d_z: {}", optional(comparison.p_value), optional(comparison.adjusted_p_value), optional(comparison.standardized_effect)),
        format!("Verified source generation {} · plan {}", comparison.status_generation, comparison.plan_digest),
    ];
    if comparison.descriptive_only { lines.push("Descriptive only: one independent seed pair cannot support inference.".into()); }
    if let Some(reason) = &comparison.underpowered_reason { lines.push(bounded_message(reason)); }
    if let Some(reason) = &comparison.undefined_effect_reason { lines.push(bounded_message(reason)); }
    lines.push("A population difference is an observed endpoint, not a fitness or winner claim.".into());
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
            summary: LabExperimentSummary { experiment_id: "actual-batch".into(), status: "completed".into(), total_runs: variants, completed_runs: variants, failed_runs: 0 },
            plan_digest: "recorded-plan".into(),
            generation: 9,
            variants: (0..variants).map(|index| LabVariantProgress {
                variant_id: format!("arm-{index:02}"), brain_family: "mlp".into(), total_runs: 1, completed_runs: 1, failed_runs: 0, running_runs: 0,
            }).collect(),
            error_reason: None,
        }
    }

    fn comparison(challenger: &str) -> LabComparison {
        LabComparison {
            experiment_id: "actual-batch".into(), baseline_variant: "arm-00".into(), challenger_variant: challenger.into(), endpoint_tick: 2,
            matched_seeds: vec![7], mean_difference: 3.0, ci_95: None, standardized_effect: None, p_value: None, adjusted_p_value: None,
            descriptive_only: true, underpowered_reason: Some("one seed pair".into()), undefined_effect_reason: Some("one observation".into()),
            plan_digest: "recorded-plan".into(), status_generation: 9,
        }
    }

    fn deliver(state: &mut LabWorkflows, request: LabWorkflowRequest, response: LabWorkflowResponse) -> Option<(String, Arc<RenderSnapshot>)> {
        let (send, result) = std::sync::mpsc::sync_channel(1);
        assert!(send.send(Ok(response)).is_ok());
        state.generation += 1;
        state.job = Some(LabJob { request, generation: state.generation, context: state.context, session: state.session.unwrap(), result });
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
                released.lock().unwrap().recv_timeout(Duration::from_secs(5)).unwrap();
                Ok(LabWorkflowResponse::Experiments(LabExperimentPage {
                    items: vec![], total: 0, offset: 0, next_cursor: None, reserved_run_slots: 0, max_run_slots: 1,
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
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1, "discarding a result must not create a second worker");
        state.synchronize(HostSessionId::new(2), LabPanel::Experiments);
        state.reset_observations();
        assert!(state.is_busy(), "epoch reset retains the occupied worker slot");
        release.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while state.job.is_some() {
            assert!(Instant::now() < deadline);
            assert!(state.poll().is_none());
            std::thread::yield_now();
        }
        assert!(state.experiments.is_none(), "an old-session catalog must not become the new session's catalog");
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
        assert!(state.experiments.is_some(), "the slot must reopen after the original worker exits");
    }

    #[test]
    fn lab_projection_keeps_every_variant_and_refuses_mismatched_or_unbounded_evidence() {
        let mut state = LabWorkflows::default();
        state.synchronize(HostSessionId::new(1), LabPanel::Experiments);
        state.selected_experiment = Some("actual-batch".into());
        let request = LabWorkflowRequest::GetExperiment { experiment_id: "actual-batch".into() };
        deliver(&mut state, request.clone(), LabWorkflowResponse::Experiment(batch(64)));
        assert_eq!(state.batch.as_ref().unwrap().variants.len(), 64);
        state.dispatch(LabAction::Challenger("arm-63".into()));
        state.dispatch(LabAction::VariantsPage(usize::MAX));
        assert_eq!(state.variant_offset, 56, "variant pages stay within retained arms");
        let compare = state.comparison_request().unwrap();
        deliver(&mut state, compare.clone(), LabWorkflowResponse::Comparison(comparison("arm-63")));
        let lines = comparison_lines(state.comparison.as_ref().unwrap()).join("\n");
        assert!(lines.contains("Descriptive only"));
        assert!(lines.contains("p: unavailable"));
        assert!(lines.contains("3.000000 agents"));
        let mut mismatched = comparison("arm-63");
        mismatched.plan_digest = "another-plan".into();
        deliver(&mut state, compare.clone(), LabWorkflowResponse::Comparison(mismatched));
        assert!(state.comparison.is_none(), "inconsistent source evidence invalidates an earlier displayed comparison");
        let mut unsupported_inference = comparison("arm-63");
        unsupported_inference.p_value = Some(0.01);
        deliver(&mut state, compare.clone(), LabWorkflowResponse::Comparison(unsupported_inference));
        assert!(state.comparison.is_none(), "one seed cannot acquire an inferential p value");
        let mut unsupported_interval = comparison("arm-63");
        unsupported_interval.ci_95 = Some((1.0, 4.0));
        deliver(&mut state, compare.clone(), LabWorkflowResponse::Comparison(unsupported_interval));
        assert!(state.comparison.is_none(), "one seed cannot acquire an inferential confidence interval");
        deliver(&mut state, compare, LabWorkflowResponse::Comparison(comparison("arm-63")));
        assert!(state.comparison.is_some());
        let (send, result) = std::sync::mpsc::sync_channel(1);
        assert!(send.send(Err("retained artifact verification failed".into())).is_ok());
        state.generation += 1;
        state.job = Some(LabJob { request: request.clone(), generation: state.generation,
            context: state.context, session: state.session.unwrap(), result });
        state.poll();
        assert!(state.comparison.is_none(), "a known artifact refusal must clear prior verified output");
        let compare = state.comparison_request().unwrap();
        deliver(&mut state, compare, LabWorkflowResponse::Comparison(comparison("arm-63")));
        assert!(state.comparison.is_some());
        let mut changed = batch(64);
        changed.generation += 1;
        deliver(&mut state, request, LabWorkflowResponse::Experiment(changed));
        assert!(state.comparison.is_none(), "a new source generation invalidates the displayed comparison");
        state.selected_experiment = None;
        let rows = vec![batch(2).summary; LAB_PAGE_SIZE + 1];
        deliver(&mut state, LabWorkflowRequest::ListExperiments { cursor: None }, LabWorkflowResponse::Experiments(LabExperimentPage {
            total: rows.len(), items: rows, offset: 0, next_cursor: None, reserved_run_slots: 0, max_run_slots: 1,
        }));
        assert!(state.experiments.is_none(), "an oversized page is rejected, not silently truncated");
        assert!(!page_is_bounded(&[1], 0, 2, &None), "remaining rows require a cursor");
        assert!(!page_is_bounded(&[1], 0, 2, &Some("0".into())), "pagination must move forward");
    }

    #[test]
    fn checkpoint_review_validates_the_saved_cut_and_discards_late_results_after_return_live() {
        use crate::{HostCore, HostCoreOptions};
        use scriptbots_core::{ScriptBotsConfig, WorldState};
        let world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(42), world_width: 120, world_height: 120, food_cell_size: 20,
            closed: true, population_minimum: 0, persistence_interval: 0,
            ..ScriptBotsConfig::default()
        }).unwrap();
        let owner = HostCore::new(
            HostSessionId::new(1), world, HostCoreOptions::default(),
        ).unwrap();
        let frame = owner.snapshot_hub().latest();
        let mut state = LabWorkflows::default();
        state.synchronize(frame.session_id, LabPanel::Checkpoints);
        deliver(&mut state, LabWorkflowRequest::ListCheckpoints { cursor: None },
            LabWorkflowResponse::Checkpoints(LabCheckpointPage {
                items: vec![LabCheckpoint {
                    checkpoint_id: "saved-cut".into(), tick: frame.world.tick,
                    agent_count: frame.world.agents.len(), byte_size: 1,
                    world_digest: "saved-science".into(),
                }],
                total: 1, offset: 0, next_cursor: None, retained_bytes: 1, max_bytes: 100,
            }));
        state.dispatch(LabAction::SelectCheckpoint("saved-cut".into()));
        let request = LabWorkflowRequest::ReviewCheckpoint { checkpoint_id: "saved-cut".into() };
        let response = || LabWorkflowResponse::CheckpointFrame {
            checkpoint_id: "saved-cut".into(), snapshot: Arc::clone(&frame),
        };
        assert!(deliver(&mut state, request.clone(), response()).is_some());
        state.checkpoints.as_mut().unwrap().items[0].tick += 1;
        assert!(deliver(&mut state, request.clone(), response()).is_none(),
            "a different cut cannot be presented as the selected checkpoint");
        state.checkpoints.as_mut().unwrap().items[0].tick -= 1;
        state.checkpoints.as_mut().unwrap().items[0].agent_count += 1;
        assert!(deliver(&mut state, request.clone(), response()).is_none(),
            "a different saved population is not the catalog cut");
        state.checkpoints.as_mut().unwrap().items[0].agent_count -= 1;
        let (send, result) = std::sync::mpsc::sync_channel(1);
        state.generation += 1;
        state.job = Some(LabJob {
            request, generation: state.generation, context: state.context,
            session: frame.session_id, result,
        });
        state.dispatch(LabAction::ReturnLive);
        assert!(state.is_busy());
        assert!(send.send(Ok(response())).is_ok());
        assert!(state.poll().is_none(),
            "return-to-live invalidates a late reconstructed frame");
        assert!(Arc::ptr_eq(&owner.snapshot_hub().latest(), &frame),
            "review never advances the actual owner");
        state.reset_observations();
        assert!(state.checkpoints.is_none() && state.selected_checkpoint.is_none());
        assert!(state.submit.is_none());
        assert_eq!(state.session, Some(frame.session_id));
    }
}
