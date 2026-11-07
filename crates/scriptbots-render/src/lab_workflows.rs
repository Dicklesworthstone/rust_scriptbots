//! Native Lab consumers of application-owned experiment and checkpoint services.
//!
//! These projections contain observations only. The application verifies retained
//! artifacts and performs analysis; this module never opens a database or owns a world.

use super::*;
use std::sync::mpsc::{Receiver, TryRecvError};

pub const LAB_PAGE_SIZE: usize = 12;
pub const LAB_MAX_VARIANTS: usize = 64;
const VARIANTS_PER_PAGE: usize = 8;
const REFRESH_INTERVAL: Duration = Duration::from_secs(2);
// Retained experiment verification can read FrankenSQLite on this worker. Match
// the application's control worker stack rather than the unsafe 2 MiB default.
const LAB_WORKER_STACK_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LabWorkflowRequest {
    ListExperiments { cursor: Option<String> },
    GetExperiment { experiment_id: String },
    CancelExperiment { experiment_id: String },
    ResumeExperiment { experiment_id: String },
    Compare {
        experiment_id: String,
        baseline_variant: String,
        challenger_variant: String,
    },
    ListCheckpoints { cursor: Option<String> },
    ReviewCheckpoint { checkpoint_id: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabExperimentSummary {
    pub experiment_id: String,
    pub status: String,
    pub total_runs: usize,
    pub completed_runs: usize,
    pub failed_runs: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabExperimentPage {
    pub items: Vec<LabExperimentSummary>,
    pub total: usize,
    pub offset: usize,
    pub next_cursor: Option<String>,
    pub reserved_run_slots: usize,
    pub max_run_slots: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabVariantProgress {
    pub variant_id: String,
    pub brain_family: String,
    pub total_runs: usize,
    pub completed_runs: usize,
    pub failed_runs: usize,
    pub running_runs: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabExperiment {
    pub summary: LabExperimentSummary,
    pub plan_digest: String,
    pub generation: u64,
    /// All variants are retained, independently of the number of seed/run rows.
    pub variants: Vec<LabVariantProgress>,
    pub error_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LabComparison {
    pub experiment_id: String,
    pub baseline_variant: String,
    pub challenger_variant: String,
    pub endpoint_tick: u64,
    pub matched_seeds: Vec<u64>,
    pub mean_difference: f64,
    pub ci_95: Option<(f64, f64)>,
    pub standardized_effect: Option<f64>,
    pub p_value: Option<f64>,
    pub adjusted_p_value: Option<f64>,
    pub descriptive_only: bool,
    pub underpowered_reason: Option<String>,
    pub undefined_effect_reason: Option<String>,
    pub plan_digest: String,
    pub status_generation: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabCheckpoint {
    pub checkpoint_id: String,
    pub tick: u64,
    pub agent_count: usize,
    pub byte_size: u64,
    pub world_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabCheckpointPage {
    pub items: Vec<LabCheckpoint>,
    pub total: usize,
    pub offset: usize,
    pub next_cursor: Option<String>,
    pub retained_bytes: u64,
    pub max_bytes: u64,
}

pub enum LabWorkflowResponse {
    Experiments(LabExperimentPage),
    Experiment(LabExperiment),
    Comparison(LabComparison),
    Checkpoints(LabCheckpointPage),
    CheckpointFrame {
        checkpoint_id: String,
        snapshot: Arc<RenderSnapshot>,
    },
}

pub type LabWorkflowSubmit = Arc<
    dyn Fn(LabWorkflowRequest) -> std::result::Result<LabWorkflowResponse, String>
        + Send
        + Sync,
>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum LabPanel {
    #[default]
    Inactive,
    Experiments,
    Checkpoints,
}

struct LabJob {
    request: LabWorkflowRequest,
    generation: u64,
    context: u64,
    session: scriptbots_runtime::HostSessionId,
    result: Receiver<std::result::Result<LabWorkflowResponse, String>>,
}

#[derive(Default)]
pub(super) struct LabWorkflows {
    pub(super) submit: Option<LabWorkflowSubmit>,
    job: Option<LabJob>,
    generation: u64,
    context: u64,
    session: Option<scriptbots_runtime::HostSessionId>,
    panel: LabPanel,
    experiments: Option<LabExperimentPage>,
    experiment_cursor: Option<String>,
    selected_experiment: Option<String>,
    batch: Option<LabExperiment>,
    baseline: Option<String>,
    challenger: Option<String>,
    variant_offset: usize,
    comparison: Option<LabComparison>,
    checkpoints: Option<LabCheckpointPage>,
    checkpoint_cursor: Option<String>,
    selected_checkpoint: Option<String>,
    notice: Option<String>,
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
    pub(super) fn invalidate(&mut self) {
        // Saturation cannot resurrect an old generation: request admission also
        // refuses when this counter is exhausted.
        self.context = self.context.saturating_add(1);
        self.last_refresh = None;
    }

    fn synchronize(&mut self, session: scriptbots_runtime::HostSessionId, panel: LabPanel) {
        if self.session != Some(session) {
            self.invalidate();
            self.session = Some(session);
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
            .name("scriptbots-gui-lab".into())
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

    fn selected_checkpoint_metadata(&self) -> Option<&LabCheckpoint> {
        let selected = self.selected_checkpoint.as_deref()?;
        self.checkpoints.as_ref()?.items.iter().find(|row| row.checkpoint_id == selected)
    }

    fn comparison_request(&self) -> Option<LabWorkflowRequest> {
        let batch = self.batch.as_ref()?;
        let baseline = self.baseline.as_ref()?;
        let challenger = self.challenger.as_ref()?;
        if baseline == challenger { return None; }
        Some(LabWorkflowRequest::Compare {
            experiment_id: batch.summary.experiment_id.clone(),
            baseline_variant: baseline.clone(),
            challenger_variant: challenger.clone(),
        })
    }

    fn poll(&mut self) -> Option<(String, Arc<RenderSnapshot>)> {
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
                | LabWorkflowRequest::ResumeExperiment { experiment_id }, LabWorkflowResponse::Experiment(mut batch))
                if self.selected_experiment.as_ref() == Some(experiment_id)
                    && batch.summary.experiment_id == *experiment_id
                    && batch_is_bounded(&batch) => {
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
                true
            }
            (LabWorkflowRequest::Compare { experiment_id, baseline_variant, challenger_variant }, LabWorkflowResponse::Comparison(comparison))
                if self.comparison_request().as_ref() == Some(&job.request)
                    && comparison.experiment_id == *experiment_id
                    && comparison.baseline_variant == *baseline_variant
                    && comparison.challenger_variant == *challenger_variant
                    && self.batch.as_ref().is_some_and(|batch| batch.plan_digest == comparison.plan_digest && batch.generation == comparison.status_generation)
                    && !comparison.matched_seeds.is_empty() && comparison.matched_seeds.len() <= 128
                    && comparison.matched_seeds.windows(2).all(|pair| pair[0] < pair[1])
                    && comparison.mean_difference.is_finite()
                    && comparison.ci_95.is_none_or(|(low, high)| low.is_finite() && high.is_finite() && low <= high)
                    && comparison.standardized_effect.is_none_or(f64::is_finite)
                    && comparison.p_value.is_none_or(|p| p.is_finite() && (0.0..=1.0).contains(&p))
                    && comparison.adjusted_p_value.is_none_or(|p| p.is_finite() && (0.0..=1.0).contains(&p))
                    && (comparison.matched_seeds.len() > 1 || (comparison.descriptive_only && comparison.ci_95.is_none() && comparison.standardized_effect.is_none() && comparison.p_value.is_none() && comparison.adjusted_p_value.is_none())) => {
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
                    && self.selected_checkpoint_metadata().is_some_and(|row| row.checkpoint_id == returned_id && row.tick == snapshot.world.tick) => {
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

    fn maybe_refresh(&mut self) {
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
}

#[derive(Clone)]
enum LabAction {
    Refresh,
    ExperimentsPage(Option<String>),
    SelectExperiment(String),
    AllExperiments,
    Baseline(String),
    Challenger(String),
    VariantsPage(usize),
    Compare,
    CancelExperiment,
    ResumeExperiment,
    CheckpointsPage(Option<String>),
    SelectCheckpoint(String),
    ReviewCheckpoint,
    CancelRequest,
    ReturnLive,
}

impl SimulationView {
    pub(super) fn poll_lab_workflows(&mut self, cx: &mut Context<Self>) {
        if self.minimal_canvas_mode { return; }
        let panel = match self.dashboard_tab {
            DashboardTab::Experiments => LabPanel::Experiments,
            DashboardTab::Checkpoints => LabPanel::Checkpoints,
            _ => LabPanel::Inactive,
        };
        self.lab_workflows.synchronize(self.host.snapshot_hub().latest().session_id, panel);
        if let Some((id, snapshot)) = self.lab_workflows.poll() {
            // World toolbar actions share this token with Timeline. A later
            // live/seek/step intent must supersede a pending Lab reconstruction,
            // even when the Lab's own tab and checkpoint selection did not move.
            if self.timeline_cancelled.load(AtomicOrdering::Acquire) {
                self.lab_workflows.notice = Some(
                    "Checkpoint review cancelled after a newer World presentation action.".into(),
                );
                return;
            }
            let tick = snapshot.world.tick;
            let applied = self.world_view.as_ref().is_some_and(|world| {
                world.update(cx, |view, cx| {
                    let palette = view.accessibility_snapshot().palette;
                    let Some(frame) = HudSnapshot::from_checkpoint_canvas(&snapshot, id.clone(), tick, palette) else { return false; };
                    view.playback.reconstructed = Some(frame);
                    view.playback.mode = PlaybackMode::Paused;
                    view.playback.displaying_history = true;
                    cx.notify();
                    true
                }).unwrap_or(false)
            });
            self.lab_workflows.notice = Some(if applied {
                format!("Showing checkpoint {id} at tick {tick} in World. Live controls remain current.")
            } else { "World window unavailable; checkpoint view unchanged.".into() });
        }
        self.lab_workflows.maybe_refresh();
    }

    fn lab_action(&mut self, action: LabAction, cx: &mut Context<Self>) {
        match action {
            LabAction::ReturnLive => {
                self.lab_workflows.invalidate();
                self.return_world_to_live(cx);
                self.lab_workflows.notice = self.world_review_feedback.clone();
            }
            LabAction::CancelRequest => {
                self.lab_workflows.invalidate();
                self.lab_workflows.notice = Some("Result cancelled. The worker remains occupied until it finishes; already submitted cancel/resume actions still take effect.".into());
            }
            LabAction::SelectExperiment(id) => {
                self.lab_workflows.invalidate();
                self.lab_workflows.selected_experiment = Some(id.clone());
                self.lab_workflows.batch = None;
                self.lab_workflows.baseline = None;
                self.lab_workflows.challenger = None;
                self.lab_workflows.comparison = None;
                self.lab_workflows.variant_offset = 0;
                if self.lab_workflows.job.is_none() {
                    self.lab_workflows.start(LabWorkflowRequest::GetExperiment { experiment_id: id });
                }
            }
            LabAction::AllExperiments => {
                self.lab_workflows.invalidate();
                self.lab_workflows.selected_experiment = None;
                self.lab_workflows.batch = None;
                self.lab_workflows.comparison = None;
            }
            LabAction::ExperimentsPage(cursor) => {
                self.lab_workflows.invalidate();
                self.lab_workflows.experiment_cursor = cursor;
                self.lab_workflows.experiments = None;
            }
            LabAction::CheckpointsPage(cursor) => {
                self.lab_workflows.invalidate();
                self.lab_workflows.checkpoint_cursor = cursor;
                self.lab_workflows.checkpoints = None;
                self.lab_workflows.selected_checkpoint = None;
            }
            LabAction::SelectCheckpoint(id) => {
                self.lab_workflows.invalidate();
                self.lab_workflows.selected_checkpoint = Some(id);
            }
            LabAction::Baseline(ref id) | LabAction::Challenger(ref id) => {
                let baseline = matches!(&action, LabAction::Baseline(_));
                if self.lab_workflows.batch.as_ref().is_some_and(|batch| batch.variants.iter().any(|row| &row.variant_id == id)) {
                    self.lab_workflows.invalidate();
                    if baseline { self.lab_workflows.baseline = Some(id.clone()); }
                    else { self.lab_workflows.challenger = Some(id.clone()); }
                    self.lab_workflows.comparison = None;
                }
            }
            LabAction::VariantsPage(offset) => self.lab_workflows.variant_offset = offset,
            LabAction::Compare => {
                if let Some(request) = self.lab_workflows.comparison_request() {
                    self.lab_workflows.comparison = None;
                    self.lab_workflows.start(request);
                }
            }
            LabAction::CancelExperiment | LabAction::ResumeExperiment => {
                if let Some(id) = self.lab_workflows.selected_experiment.clone() {
                    let request = if matches!(action, LabAction::CancelExperiment) {
                        LabWorkflowRequest::CancelExperiment { experiment_id: id }
                    } else { LabWorkflowRequest::ResumeExperiment { experiment_id: id } };
                    self.lab_workflows.start(request);
                }
            }
            LabAction::ReviewCheckpoint => {
                if let Some(id) = self.lab_workflows.selected_checkpoint.clone() {
                    if self.lab_workflows.job.is_none() {
                        // Old Timeline work also checks its active tab/selection;
                        // this new Checkpoints intent can now observe subsequent
                        // cross-window cancellation without reviving that work.
                        self.timeline_cancelled.store(false, AtomicOrdering::Release);
                    }
                    self.lab_workflows.start(LabWorkflowRequest::ReviewCheckpoint { checkpoint_id: id });
                }
            }
            LabAction::Refresh => {
                if let Some(request) = self.lab_workflows.refresh_request() { self.lab_workflows.start(request); }
            }
        }
        cx.notify();
    }

    fn lab_button(&self, id: String, label: String, action: LabAction, enabled: bool, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let mouse_action = action.clone();
        div().id(SharedString::from(id.clone())).debug_selector(move || id.clone())
            .focusable().tab_stop(true).role(Role::Button).aria_label(label.clone())
            .rounded_md().border_1().border_color(chrome::border()).px_2().py_1().text_xs()
            .text_color(if enabled { chrome::text_primary() } else { chrome::text_muted() })
            .child(label)
            .on_mouse_down(MouseButton::Left, cx.listener(move |this, _, _, cx| {
                if enabled { this.lab_action(mouse_action.clone(), cx); }
            }))
            .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
                if matches!(event.keystroke.key.as_str(), "enter" | "space" | " ") {
                    if enabled { this.lab_action(action.clone(), cx); }
                    cx.stop_propagation();
                }
            }))
    }

    fn lab_panel_header(&self, title: &str, cx: &mut Context<Self>) -> Div {
        let idle = self.lab_workflows.job.is_none();
        let mut header = div().flex().flex_col().gap_2()
            .child(div().text_lg().child(title.to_owned()))
            .child(div().flex().flex_wrap().gap_2()
                .child(self.lab_button("lab-refresh".into(), "Refresh".into(), LabAction::Refresh, idle, cx))
                .child(self.lab_button("lab-cancel-request".into(), "Discard pending result".into(), LabAction::CancelRequest, !idle, cx)));
        if self.lab_workflows.submit.is_none() {
            header = header.child(div().text_xs().child("Application data services are unavailable in this view."));
        }
        if let Some(notice) = &self.lab_workflows.notice {
            header = header.child(div().text_xs().text_color(chrome::text_secondary()).child(notice.clone()));
        }
        header
    }

    pub(super) fn render_lab_experiments(&self, cx: &mut Context<Self>) -> Div {
        let state = &self.lab_workflows;
        let idle = state.job.is_none();
        let mut panel = div().flex().flex_col().gap_3().child(self.lab_panel_header("Experiments", cx));
        if state.selected_experiment.is_some() {
            panel = panel.child(self.lab_button("lab-all-experiments".into(), "All experiments".into(), LabAction::AllExperiments, true, cx));
            let Some(batch) = &state.batch else { return panel.child("Loading selected experiment…"); };
            let summary = &batch.summary;
            panel = panel.child(div().text_sm().child(format!("{} · {} · {}/{} completed · {} failed", summary.experiment_id, summary.status, summary.completed_runs, summary.total_runs, summary.failed_runs)))
                .child(div().text_xs().child(format!("Source generation {} · plan {}", batch.generation, batch.plan_digest)))
                .child(div().flex().gap_2()
                    .child(self.lab_button("lab-stop-experiment".into(), "Stop at durable boundary".into(), LabAction::CancelExperiment, idle && matches!(summary.status.as_str(), "running" | "pending"), cx))
                    .child(self.lab_button("lab-resume-experiment".into(), "Resume unfinished runs".into(), LabAction::ResumeExperiment, idle && !matches!(summary.status.as_str(), "running" | "pending" | "cancelling") && summary.completed_runs + summary.failed_runs < summary.total_runs, cx)));
            if let Some(error) = &batch.error_reason { panel = panel.child(div().text_xs().child(error.clone())); }
            for (index, variant) in batch.variants.iter().enumerate().skip(state.variant_offset).take(VARIANTS_PER_PAGE) {
                panel = panel.child(div().flex().flex_col().gap_1()
                    .child(div().text_xs().child(format!("{} ({}) · {}/{} completed · {} running · {} failed", variant.variant_id, variant.brain_family, variant.completed_runs, variant.total_runs, variant.running_runs, variant.failed_runs)))
                    .child(div().flex().gap_2()
                        .child(self.lab_button(format!("lab-baseline-{index}"), if state.baseline.as_ref() == Some(&variant.variant_id) { "Baseline selected" } else { "Use as baseline" }.into(), LabAction::Baseline(variant.variant_id.clone()), true, cx))
                        .child(self.lab_button(format!("lab-challenger-{index}"), if state.challenger.as_ref() == Some(&variant.variant_id) { "Challenger selected" } else { "Use as challenger" }.into(), LabAction::Challenger(variant.variant_id.clone()), true, cx))));
            }
            panel = panel.child(div().flex().gap_2()
                .child(self.lab_button("lab-variants-prev".into(), "Previous variants".into(), LabAction::VariantsPage(state.variant_offset.saturating_sub(VARIANTS_PER_PAGE)), state.variant_offset > 0, cx))
                .child(self.lab_button("lab-variants-next".into(), "Next variants".into(), LabAction::VariantsPage(state.variant_offset + VARIANTS_PER_PAGE), state.variant_offset + VARIANTS_PER_PAGE < batch.variants.len(), cx)))
                .child(div().text_xs().child(format!("Population comparison: {} − {}. Complete matched seed cohorts are required.", state.challenger.as_deref().unwrap_or("choose challenger"), state.baseline.as_deref().unwrap_or("choose baseline"))))
                .child(self.lab_button("lab-compare".into(), "Compare verified population endpoints".into(), LabAction::Compare, idle && state.comparison_request().is_some(), cx));
            if let Some(comparison) = &state.comparison {
                panel = panel.child(div().flex().flex_col().gap_1().children(comparison_lines(comparison).into_iter().map(|line| div().text_xs().child(line))));
            }
            return panel;
        }
        let Some(page) = &state.experiments else { return panel.child("Loading experiment catalog…"); };
        panel = panel.child(div().text_xs().child(format!("{} experiments · reserved run slots {}/{}", page.total, page.reserved_run_slots, page.max_run_slots)));
        for (index, row) in page.items.iter().enumerate() {
            panel = panel.child(self.lab_button(format!("lab-experiment-{index}"), format!("{} · {} · {}/{} completed · {} failed", row.experiment_id, row.status, row.completed_runs, row.total_runs, row.failed_runs), LabAction::SelectExperiment(row.experiment_id.clone()), true, cx));
        }
        if page.items.is_empty() {
            panel = panel.child(div().text_sm().child("No retained experiment batches. Create a matched-seed batch with the experiment API or CLI, then refresh."));
        }
        panel.child(div().flex().gap_2()
            .child(self.lab_button("lab-experiments-prev".into(), "Previous experiments".into(), LabAction::ExperimentsPage((page.offset > LAB_PAGE_SIZE).then(|| (page.offset - LAB_PAGE_SIZE).to_string())), idle && page.offset > 0, cx))
            .child(self.lab_button("lab-experiments-next".into(), "Next experiments".into(), LabAction::ExperimentsPage(page.next_cursor.clone()), idle && page.next_cursor.is_some(), cx)))
    }

    pub(super) fn render_lab_checkpoints(&self, cx: &mut Context<Self>) -> Div {
        let state = &self.lab_workflows;
        let idle = state.job.is_none();
        let mut panel = div().flex().flex_col().gap_3().child(self.lab_panel_header("Saved checkpoints", cx))
            .child(div().text_xs().child("Select a saved scientific cut to reconstruct it in World. The live owner keeps its current state and controls."))
            .child(self.lab_button("lab-world-live".into(), "Return World to live".into(), LabAction::ReturnLive, true, cx));
        let Some(page) = &state.checkpoints else { return panel.child("Loading checkpoint catalog…"); };
        panel = panel.child(div().text_xs().child(format!("{} checkpoints · retained bytes {}/{}", page.total, page.retained_bytes, page.max_bytes)));
        for (index, row) in page.items.iter().enumerate() {
            let selected = state.selected_checkpoint.as_ref() == Some(&row.checkpoint_id);
            panel = panel.child(self.lab_button(format!("lab-checkpoint-{index}"), format!("{}{} · tick {} · {} agents · {} bytes", if selected { "Selected: " } else { "" }, row.checkpoint_id, row.tick, row.agent_count, row.byte_size), LabAction::SelectCheckpoint(row.checkpoint_id.clone()), true, cx));
        }
        if let Some(selected) = state.selected_checkpoint_metadata() {
            panel = panel.child(div().text_xs().child(format!("Saved world digest {}", selected.world_digest)));
        }
        if page.items.is_empty() { panel = panel.child("No saved checkpoints. Use Timeline → Save replay checkpoint, then refresh."); }
        panel.child(self.lab_button("lab-review-checkpoint".into(), "Review selected checkpoint in World".into(), LabAction::ReviewCheckpoint, idle && state.selected_checkpoint_metadata().is_some(), cx))
            .child(div().flex().gap_2()
                .child(self.lab_button("lab-checkpoints-prev".into(), "Previous checkpoints".into(), LabAction::CheckpointsPage((page.offset > LAB_PAGE_SIZE).then(|| (page.offset - LAB_PAGE_SIZE).to_string())), idle && page.offset > 0, cx))
                .child(self.lab_button("lab-checkpoints-next".into(), "Next checkpoints".into(), LabAction::CheckpointsPage(page.next_cursor.clone()), idle && page.next_cursor.is_some(), cx)))
    }
}

fn comparison_lines(comparison: &LabComparison) -> Vec<String> {
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
    use std::sync::atomic::AtomicUsize;

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
        state.synchronize(scriptbots_runtime::HostSessionId::new(1), LabPanel::Experiments);
        let request = LabWorkflowRequest::ListExperiments { cursor: None };
        state.start(request.clone());
        observed_start.recv_timeout(Duration::from_secs(5)).unwrap();
        state.invalidate();
        state.start(request.clone());
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1, "discarding a result must not create a second worker");
        state.synchronize(scriptbots_runtime::HostSessionId::new(2), LabPanel::Experiments);
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
        state.synchronize(scriptbots_runtime::HostSessionId::new(1), LabPanel::Experiments);
        state.selected_experiment = Some("actual-batch".into());
        let request = LabWorkflowRequest::GetExperiment { experiment_id: "actual-batch".into() };
        deliver(&mut state, request.clone(), LabWorkflowResponse::Experiment(batch(64)));
        assert_eq!(state.batch.as_ref().unwrap().variants.len(), 64);
        state.challenger = Some("arm-63".into());
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
    fn gpui_lab_buttons_compare_and_review_checkpoint_without_advancing_live_owner() {
        let host = TestHost::new(WorldState::new(ScriptBotsConfig {
            rng_seed: Some(42), world_width: 120, world_height: 120, food_cell_size: 20,
            closed: true, population_minimum: 0, persistence_interval: 0, ..ScriptBotsConfig::default()
        }).unwrap());
        let saved = host.snapshot();
        host.apply(scriptbots_runtime::HostCommand::UpdateSimulation(SimulationCommand {
            paused: Some(true), speed_multiplier: None, step_once: true,
        }));
        let before = host.port.scientific_digest_v1().unwrap();
        let live_tick = host.snapshot().world.tick;
        assert!(live_tick > saved.world.tick);
        let mut session = GuiSession::new(host.port.clone(), AnalyticsSnapshotProvider::empty(), Arc::new(|_| panic!("Lab inspection must not submit a live scientific command")));
        let review_gate = Arc::new((Mutex::new(true), std::sync::Condvar::new()));
        let worker_gate = Arc::clone(&review_gate);
        let (review_started, observed_review) = std::sync::mpsc::channel();
        session.lab_submit = Some(Arc::new(move |request| Ok(match request {
            LabWorkflowRequest::ListExperiments { .. } => LabWorkflowResponse::Experiments(LabExperimentPage {
                items: vec![batch(2).summary], total: 1, offset: 0, next_cursor: None, reserved_run_slots: 0, max_run_slots: 1,
            }),
            LabWorkflowRequest::GetExperiment { .. } => LabWorkflowResponse::Experiment(batch(2)),
            LabWorkflowRequest::Compare { baseline_variant, challenger_variant, .. } => {
                assert_eq!(baseline_variant, "arm-00");
                assert_eq!(challenger_variant, "arm-01");
                LabWorkflowResponse::Comparison(comparison("arm-01"))
            }
            LabWorkflowRequest::ListCheckpoints { .. } => LabWorkflowResponse::Checkpoints(LabCheckpointPage {
                items: vec![LabCheckpoint { checkpoint_id: "saved-cut".into(), tick: saved.world.tick, agent_count: saved.world.agents.len(), byte_size: 1, world_digest: "component-world-cut".into() }],
                total: 1, offset: 0, next_cursor: None, retained_bytes: 1, max_bytes: 100,
            }),
            LabWorkflowRequest::ReviewCheckpoint { checkpoint_id } => {
                review_started.send(()).unwrap();
                let (lock, ready) = &*worker_gate;
                let (allowed, _) = ready.wait_timeout_while(
                    lock.lock().unwrap(), Duration::from_secs(5), |allowed| !*allowed,
                ).unwrap();
                assert!(*allowed, "native checkpoint gate was not released");
                LabWorkflowResponse::CheckpointFrame {
                    checkpoint_id,
                    snapshot: Arc::clone(&saved),
                }
            }
            _ => panic!("unexpected Lab request"),
        })));
        let context = gpui::TestAppContext::single();
        let windows = context.update(|app| Arc::new(session).install(app)).unwrap();
        let hud = context.update(|app| windows.hud.root(app)).unwrap();
        let world = context.update(|app| windows.canvas.root(app)).unwrap();
        let mut lab = gpui::VisualTestContext::from_window(windows.hud.into(), &context);
        lab.simulate_resize(size(px(1000.0), px(1800.0)));
        let draw = |lab: &mut gpui::VisualTestContext| lab.update(|window, app| { window.refresh(); window.draw(app).clear(); });
        let click = |lab: &mut gpui::VisualTestContext, selector: &'static str| {
            draw(lab);
            let bounds = lab.debug_bounds(selector).unwrap_or_else(|| panic!("missing production button {selector}"));
            lab.simulate_click(bounds.center(), gpui::Modifiers::default());
        };
        let wait = |lab: &mut gpui::VisualTestContext, ready: &dyn Fn(&LabWorkflows) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                draw(lab);
                if context.update(|app| ready(&hud.read(app).lab_workflows)) { break; }
                assert!(Instant::now() < deadline, "native Lab result did not arrive");
                std::thread::yield_now();
            }
        };
        click(&mut lab, "dashboard-tab-experiments");
        wait(&mut lab, &|state| state.experiments.is_some() && state.job.is_none());
        click(&mut lab, "lab-experiment-0");
        wait(&mut lab, &|state| state.batch.is_some() && state.job.is_none());
        click(&mut lab, "lab-compare");
        wait(&mut lab, &|state| state.comparison.is_some() && state.job.is_none());
        click(&mut lab, "dashboard-tab-checkpoints");
        wait(&mut lab, &|state| state.checkpoints.is_some() && state.job.is_none());
        click(&mut lab, "lab-checkpoint-0");
        click(&mut lab, "lab-review-checkpoint");
        wait(&mut lab, &|state| state.job.is_none() && state.notice.as_deref().is_some_and(|message| message.starts_with("Showing checkpoint")));
        context.update(|app| {
            let view = world.read(app);
            let reconstructed = view.playback.reconstructed.as_ref().unwrap();
            assert_eq!(reconstructed.tick, 0);
            assert_eq!(reconstructed.checkpoint_review.as_ref().unwrap().0, "saved-cut");
        });
        click(&mut lab, "lab-world-live");
        context.update(|app| assert!(world.read(app).playback.reconstructed.is_none()));
        observed_review.recv_timeout(Duration::from_secs(5)).unwrap();

        // The Lab tab/selection remains fixed while the separate World window
        // expresses a newer live intent. Releasing the retained worker must not
        // overwrite that action with the older checkpoint frame.
        *review_gate.0.lock().unwrap() = false;
        click(&mut lab, "lab-review-checkpoint");
        observed_review.recv_timeout(Duration::from_secs(5)).unwrap();
        context.update(|app| world.update(app, |view, cx| view.playback_go_live(cx)));
        *review_gate.0.lock().unwrap() = true;
        review_gate.1.notify_one();
        wait(&mut lab, &|state| state.job.is_none()
            && state.notice.as_deref().is_some_and(|message| message.starts_with("Checkpoint review cancelled")));
        context.update(|app| assert!(world.read(app).playback.reconstructed.is_none(),
            "late Lab review must not replace a newer World-window live action"));
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
        assert_eq!(host.snapshot().world.tick, live_tick);
    }
}
