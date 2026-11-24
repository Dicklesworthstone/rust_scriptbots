//! GPUI widgets for application-owned Lab services.
//!
//! The renderer-neutral model owns bounded background requests and validation;
//! this module only renders its observations and applies presentation intents.

use super::*;
use scriptbots_runtime::lab_workflows::{
    LAB_MAX_ARTIFACT_BYTES, LAB_MAX_TREATMENT_PERCENT, LAB_PAGE_SIZE, LabAction,
    LabArtifactExportState, LabBranchCreationState, LabBranchTreatment, LabCheckpointSaveState,
    LabCheckpointView, LabExperimentCapabilities, LabExperimentCreationState, LabExperimentDesign,
    LabExperimentTreatment, LabMetric, LabPanel, VARIANTS_PER_PAGE, artifact_export_lines,
    branch_report_lines, comparison_lines, replay_summary_lines,
};
#[cfg(test)]
use scriptbots_runtime::lab_workflows::{
    LabBranchCapabilities, LabBranchDivergence, LabBranchDraft, LabBranchOutcome, LabBranchReport,
    LabCheckpoint, LabCheckpointBranch, LabCheckpointBranchPage, LabCheckpointPage, LabComparison,
    LabExperiment, LabExperimentDraft, LabExperimentPage, LabExperimentParameter,
    LabExperimentParameterCapability, LabExperimentSummary, LabVariantProgress, LabWorkflowRequest,
    LabWorkflowResponse,
};

#[derive(Clone, Copy)]
enum LabGuideAction {
    Open,
    Hide,
    Restart,
    Next(LabGuideBoundary),
    Perform(LabGuideBoundary),
}

#[derive(Clone, Copy)]
struct LabGuideBoundary {
    generation: u64,
    step: LabGuideStep,
    session_id: scriptbots_runtime::HostSessionId,
    config_revision: scriptbots_runtime::ConfigRevision,
    growth_proposal: Option<f32>,
}

/// A disclosed one-coefficient proposal; no scientific state is changed here.
fn guide_growth_proposal(snapshot: &RenderSnapshot) -> Option<f32> {
    let current = snapshot.config.food_growth_rate;
    let proposed = if current == 0.0 {
        0.0005
    } else {
        current * 0.75
    };
    if !proposed.is_finite() || proposed.to_bits() == current.to_bits() {
        return None;
    }
    let mut config = snapshot.config.as_ref().clone();
    config.food_growth_rate = proposed;
    config.validate().ok().map(|()| proposed)
}

impl SimulationView {
    /// Both windows observe the same owner and guide; neither advances science.
    fn observe_lab_guide(&self) {
        let snapshot = self.host.snapshot_hub().latest();
        let focused = self.focused_agent();
        let selected = focused.and_then(|id| {
            snapshot
                .world
                .agents
                .iter()
                .find(|agent| agent.id == id.raw())
                .map(|agent| agent.uid)
        });
        let pending = if let Ok(mut guide) = self.lab_guide.lock() {
            guide.observe_snapshot(&snapshot, selected);
            // Pending is observed only at actual save dispatch below. Repainting
            // a request started before this learning step must not adopt it.
            if let Some(save) = &self.lab_workflows.latest_checkpoint_save
                && !matches!(&save.state, LabCheckpointSaveState::Pending)
            {
                guide.observe_checkpoint_save(save);
            }
            if let (Some(branch), Some(report)) = (
                &self.lab_workflows.branch,
                &self.lab_workflows.branch_report,
            ) {
                guide.observe_branch_report(snapshot.session_id, branch, report);
            }
            guide.pending_command_id()
        } else {
            None
        };
        if let Some(id) = pending
            && let Ok(Some(status)) = self.host.try_published_command_status(id)
            && let Ok(mut guide) = self.lab_guide.lock()
        {
            guide.observe_status(snapshot.session_id, &status);
        }
    }

    fn guide_navigate(&mut self, tab: DashboardTab) {
        if self.dashboard_tab != tab {
            self.timeline_cancelled.store(true, AtomicOrdering::Release);
            self.lab_workflows.invalidate();
            self.dashboard_tab = tab;
        }
        self.synchronize_lab_source();
    }

    fn guide_action(&mut self, action: LabGuideAction, cx: &mut Context<Self>) {
        match action {
            LabGuideAction::Open | LabGuideAction::Hide | LabGuideAction::Restart => {
                if let Ok(mut guide) = self.lab_guide.lock() {
                    match action {
                        LabGuideAction::Open => guide.open(),
                        LabGuideAction::Hide => guide.visible = false,
                        LabGuideAction::Restart => guide.restart(),
                        _ => unreachable!("guide visibility action"),
                    }
                }
                self.observe_lab_guide();
            }
            LabGuideAction::Next(expected) => {
                self.observe_lab_guide();
                if self.host.snapshot_hub().latest().session_id == expected.session_id
                    && let Ok(mut guide) = self.lab_guide.lock()
                    && guide.generation() == expected.generation
                    && guide.step() == expected.step
                {
                    guide.advance();
                }
                self.observe_lab_guide();
            }
            LabGuideAction::Perform(expected) => {
                self.observe_lab_guide();
                let snapshot = self.host.snapshot_hub().latest();
                if snapshot.session_id != expected.session_id {
                    return;
                }
                let current = self.lab_guide.lock().ok().map(|guide| {
                    (
                        guide.generation(),
                        guide.step(),
                        guide.finished(),
                        guide.ready(),
                        guide.pending_command_id().is_some(),
                        guide
                            .saved_checkpoint()
                            .map(|saved| saved.checkpoint_id.clone()),
                    )
                });
                let Some((generation, step, finished, ready, pending, saved)) = current else {
                    return;
                };
                let navigation = matches!(
                    step,
                    LabGuideStep::Identify | LabGuideStep::Inspect | LabGuideStep::Outcome
                );
                if generation != expected.generation
                    || step != expected.step
                    || finished
                    || pending
                    || (ready && !navigation)
                {
                    return;
                }
                if matches!(
                    step,
                    LabGuideStep::Pause
                        | LabGuideStep::Step
                        | LabGuideStep::Configure
                        | LabGuideStep::Checkpoint
                ) && !self.lab_world_is_live(cx)
                {
                    self.lab_workflows.set_notice(
                        "Return the World window to live before this guide action. Historical views do not control their saved past.",
                    );
                    cx.notify();
                    return;
                }
                if matches!(
                    step,
                    LabGuideStep::Step | LabGuideStep::Configure | LabGuideStep::Checkpoint
                ) && !snapshot.playback.paused
                {
                    self.lab_workflows
                        .set_notice("Pause the live world before continuing this guide step.");
                    cx.notify();
                    return;
                }
                match step {
                    LabGuideStep::Identify => self.guide_navigate(DashboardTab::Overview),
                    LabGuideStep::Inspect => self.guide_navigate(DashboardTab::Inspect),
                    LabGuideStep::Pause => {
                        self.submit_control_command(ControlCommand::Pause);
                    }
                    LabGuideStep::Step => {
                        self.submit_control_command(ControlCommand::Step);
                    }
                    LabGuideStep::Configure => {
                        if snapshot.revisions.config != expected.config_revision {
                            self.lab_workflows.set_notice(
                                "Configuration changed after this proposal was displayed. Review the updated value before retrying.",
                            );
                        } else if let Some(value) = expected.growth_proposal {
                            let mut config = snapshot.config.as_ref().clone();
                            config.food_growth_rate = value;
                            if let Some(submit) = self.config_update_submit.clone() {
                                let command =
                                    ControlCommand::UpdateConfig(Box::new(config.clone()));
                                match submit(Box::new(config), snapshot.revisions) {
                                    Ok(status) => {
                                        let id = status.command_id();
                                        if let Ok(mut guide) = self.lab_guide.lock() {
                                            guide.observe_snapshot(&snapshot, None);
                                            guide.record_command(&command, id, &snapshot);
                                            guide.observe_status(snapshot.session_id, &status);
                                        }
                                        self.record_gui_command(
                                            "Configuration",
                                            GuiCommandSubmission::Enqueued(id),
                                        );
                                    }
                                    Err(error) => self.lab_workflows.set_notice(format!(
                                        "Configuration submission could not be confirmed: {error}"
                                    )),
                                }
                            } else {
                                self.lab_workflows.set_notice(
                                    "Guarded configuration updates are unavailable in this view.",
                                );
                            }
                        } else {
                            self.lab_workflows.set_notice(
                                "No distinct valid food-regrowth proposal is available for this configuration.",
                            );
                        }
                    }
                    LabGuideStep::Checkpoint => {
                        self.guide_navigate(DashboardTab::Checkpoints);
                        self.lab_action(LabAction::CreateCheckpoint, cx);
                    }
                    LabGuideStep::Branch => {
                        self.guide_navigate(DashboardTab::Checkpoints);
                        if let Some(id) = saved {
                            if self.lab_workflows.is_busy() {
                                self.lab_workflows.set_notice(
                                    "Wait for the current Lab request before choosing this saved source.",
                                );
                            } else if self
                                .lab_workflows
                                .branch
                                .as_ref()
                                .is_some_and(|branch| branch.checkpoint_id == id)
                            {
                                let action = if self.lab_workflows.can_read_branch_report() {
                                    LabAction::ReadCheckpointBranchReport
                                } else {
                                    LabAction::Refresh
                                };
                                self.lab_action(action, cx);
                            } else {
                                self.lab_workflows.dispatch(LabAction::BackToCheckpoints);
                                self.lab_workflows
                                    .dispatch(LabAction::SelectCheckpoint(id.clone()));
                                if self
                                    .lab_workflows
                                    .selected_checkpoint_metadata()
                                    .is_some_and(|checkpoint| checkpoint.checkpoint_id == id)
                                {
                                    self.lab_action(LabAction::NewCheckpointBranch, cx);
                                } else {
                                    self.lab_workflows.set_notice(
                                        "The walkthrough's saved cut is unavailable in this catalog. Refresh or return to its retained save receipt.",
                                    );
                                }
                            }
                        }
                    }
                    LabGuideStep::Outcome => {
                        self.guide_navigate(DashboardTab::Checkpoints);
                        if self.lab_workflows.branch_report.is_none()
                            && self.lab_workflows.can_read_branch_report()
                        {
                            self.lab_action(LabAction::ReadCheckpointBranchReport, cx);
                        }
                    }
                }
            }
        }
        cx.notify();
    }

    fn guide_button(
        &self,
        id: &'static str,
        label: String,
        action: LabGuideAction,
        enabled: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        div()
            .id(id)
            .debug_selector(move || id.to_owned())
            .focusable()
            .tab_stop(true)
            .role(Role::Button)
            .aria_label(label.clone())
            .rounded_md()
            .border_1()
            .border_color(chrome::border())
            .px_2()
            .py_1()
            .text_xs()
            .text_color(if enabled {
                chrome::text_primary()
            } else {
                chrome::text_muted()
            })
            .child(label)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    if enabled {
                        this.guide_action(action, cx);
                    }
                }),
            )
            .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
                if matches!(event.keystroke.key.as_str(), "enter" | "space" | " ") {
                    if enabled && !event.is_held {
                        this.guide_action(action, cx);
                    }
                    cx.stop_propagation();
                }
            }))
    }

    pub(super) fn render_lab_guide_launcher(
        &self,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        self.guide_button(
            "lab-guide-open",
            "First-run guide".into(),
            LabGuideAction::Open,
            true,
            cx,
        )
    }

    pub(super) fn render_lab_guide(&self, cx: &mut Context<Self>) -> Div {
        let state = self.lab_guide.lock().ok().and_then(|guide| {
            guide.visible.then(|| {
                (
                    guide.generation(),
                    guide.step(),
                    guide.ready(),
                    guide.finished(),
                    guide.pending_command_id().is_some(),
                    guide.status().to_owned(),
                    guide.evidence().to_owned(),
                )
            })
        });
        let Some((generation, step, ready, finished, pending, status, evidence)) = state else {
            return div();
        };
        let snapshot = self.host.snapshot_hub().latest();
        let growth_proposal = (step == LabGuideStep::Configure)
            .then(|| guide_growth_proposal(&snapshot))
            .flatten();
        let boundary = LabGuideBoundary {
            generation,
            step,
            session_id: snapshot.session_id,
            config_revision: snapshot.revisions.config,
            growth_proposal,
        };
        let label = match step {
            LabGuideStep::Identify => "Open run overview".to_owned(),
            LabGuideStep::Inspect => "Open organism inspector".to_owned(),
            LabGuideStep::Pause => "Pause this run".to_owned(),
            LabGuideStep::Step => "Advance exactly one tick".to_owned(),
            LabGuideStep::Configure => growth_proposal.map_or_else(
                || "Food regrowth change unavailable".into(),
                |value| {
                    format!(
                        "Set food regrowth: {} → {value}",
                        snapshot.config.food_growth_rate,
                    )
                },
            ),
            LabGuideStep::Checkpoint => "Save live checkpoint".to_owned(),
            LabGuideStep::Branch => "Configure or inspect the saved-source study".to_owned(),
            LabGuideStep::Outcome => "Show verified study outcome".to_owned(),
        };
        let navigation = matches!(
            step,
            LabGuideStep::Identify | LabGuideStep::Inspect | LabGuideStep::Outcome
        );
        let available = !pending
            && (navigation || !ready)
            && !finished
            && (step != LabGuideStep::Configure
                || (growth_proposal.is_some() && self.config_update_submit.is_some()))
            && (!matches!(
                step,
                LabGuideStep::Step | LabGuideStep::Configure | LabGuideStep::Checkpoint
            ) || snapshot.playback.paused)
            && (!matches!(
                step,
                LabGuideStep::Checkpoint | LabGuideStep::Branch | LabGuideStep::Outcome
            ) || self.lab_workflows.submit.is_some());
        let mut panel = div()
            .id("lab-guide-panel")
            .debug_selector(|| "lab-guide-panel".to_owned())
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .mb_3()
            .rounded_lg()
            .border_1()
            .border_color(chrome::accent())
            .bg(chrome::surface())
            .child(div().text_sm().child(if finished {
                "Walkthrough observations complete".to_owned()
            } else {
                format!(
                    "{} / {} · {}",
                    step.index() + 1,
                    LabGuideStep::ALL.len(),
                    step.title(),
                )
            }))
            .child(div().text_xs().child(format!(
                "Live owner {} · seed {} · tick {} · {} agents",
                snapshot.session_id.get(),
                snapshot.visual_seed,
                snapshot.world.tick,
                snapshot.world.agents.len(),
            )))
            .child(div().text_xs().child(step.instruction()))
            .child(div().text_xs().child(status));
        if !evidence.is_empty() {
            panel = panel.child(div().text_xs().child(evidence));
        }
        if matches!(
            step,
            LabGuideStep::Step | LabGuideStep::Configure | LabGuideStep::Checkpoint
        ) && !snapshot.playback.paused
        {
            panel = panel.child(
                div()
                    .text_xs()
                    .child("Pause the live world before continuing this guide step."),
            );
        }
        if step == LabGuideStep::Configure && self.config_update_submit.is_none() {
            panel = panel.child(
                div()
                    .text_xs()
                    .child("Guarded configuration updates are unavailable in this view."),
            );
        }
        if self.lab_workflows.submit.is_none() {
            panel = panel.child(
                div()
                    .text_xs()
                    .child("Checkpoint and study services are unavailable in this view."),
            );
        }
        if let Some(notice) = &self.lab_workflows.notice {
            panel = panel.child(div().text_xs().child(notice.clone()));
        }
        div().child(
            panel.child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_2()
                    .child(self.guide_button(
                        "lab-guide-perform",
                        label,
                        LabGuideAction::Perform(boundary),
                        available,
                        cx,
                    ))
                    .child(self.guide_button(
                        "lab-guide-next",
                        if step == LabGuideStep::Outcome {
                            "Finish".into()
                        } else {
                            "Next".into()
                        },
                        LabGuideAction::Next(boundary),
                        ready && !finished,
                        cx,
                    ))
                    .child(self.guide_button(
                        "lab-guide-hide",
                        "Hide; keep progress".into(),
                        LabGuideAction::Hide,
                        true,
                        cx,
                    ))
                    .child(self.guide_button(
                        "lab-guide-restart",
                        "Restart guide".into(),
                        LabGuideAction::Restart,
                        true,
                        cx,
                    )),
            ),
        )
    }

    fn synchronize_lab_source(&mut self) {
        let panel = match self.dashboard_tab {
            DashboardTab::Experiments => LabPanel::Experiments,
            DashboardTab::Checkpoints => LabPanel::Checkpoints,
            _ => LabPanel::Inactive,
        };
        let source = self.host.snapshot_hub().latest();
        self.lab_workflows.synchronize(source.session_id, panel);
    }

    fn lab_world_is_live(&self, cx: &Context<Self>) -> bool {
        let session = self.host.snapshot_hub().latest().session_id;
        self.world_view.as_ref().is_some_and(|world| {
            world
                .read_with(cx, |view, _| {
                    view.host.snapshot_hub().latest().session_id == session
                        && view.playback.mode() == PlaybackMode::Live
                        && !view.playback.displaying_history
                        && view.playback.reconstructed.is_none()
                })
                .unwrap_or(false)
        })
    }

    pub(super) fn poll_lab_workflows(&mut self, cx: &mut Context<Self>) {
        if self.minimal_canvas_mode {
            self.observe_lab_guide();
            return;
        }
        self.synchronize_lab_source();
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
                world
                    .update(cx, |view, cx| {
                        let palette = view.accessibility_snapshot().palette;
                        let Some(frame) = HudSnapshot::from_checkpoint_canvas(
                            &snapshot,
                            id.clone(),
                            tick,
                            palette,
                        ) else {
                            return false;
                        };
                        view.playback.reconstructed = Some(frame);
                        view.playback.mode = PlaybackMode::Paused;
                        view.playback.displaying_history = true;
                        cx.notify();
                        true
                    })
                    .unwrap_or(false)
            });
            self.lab_workflows.notice = Some(if applied {
                format!(
                    "Showing checkpoint {id} at tick {tick} in World. Live controls remain current."
                )
            } else {
                "World window unavailable; checkpoint view unchanged.".into()
            });
        }
        if let Some(review) = self.lab_workflows.take_replay_review() {
            if self.timeline_cancelled.load(AtomicOrdering::Acquire) {
                self.lab_workflows.set_notice(
                    "Replay result discarded after a newer World presentation action; the displayed world is unchanged.",
                );
            } else {
                let summary = &review.summary;
                let applied = self.world_view.as_ref().is_some_and(|world| {
                    world
                        .update(cx, |view, cx| {
                            if view.host.snapshot_hub().latest().session_id != summary.session_id
                                || review.snapshot.session_id != summary.session_id
                                || review.snapshot.world.tick != summary.target_tick
                            {
                                return false;
                            }
                            let palette = view.accessibility_snapshot().palette;
                            let Some(frame) = HudSnapshot::from_checkpoint_canvas(
                                &review.snapshot,
                                summary.checkpoint.checkpoint_id.clone(),
                                summary.checkpoint.tick,
                                palette,
                            ) else {
                                return false;
                            };
                            view.playback.reconstructed = Some(frame);
                            view.playback.mode = PlaybackMode::Paused;
                            view.playback.displaying_history = true;
                            cx.notify();
                            true
                        })
                        .unwrap_or(false)
                });
                self.lab_workflows.set_notice(if applied {
                    format!(
                        "Showing replay tick {} from checkpoint {} at tick {}. The live owner is unchanged; edit the target to seek again.",
                        summary.target_tick,
                        summary.checkpoint.checkpoint_id,
                        summary.checkpoint.tick,
                    )
                } else {
                    "Replay visual layers or World window are unavailable; the displayed world is unchanged."
                        .to_owned()
                });
            }
        }
        self.lab_workflows.maybe_refresh();
        self.observe_lab_guide();
    }

    fn lab_action(&mut self, action: LabAction, cx: &mut Context<Self>) {
        self.synchronize_lab_source();
        // The World window can enter history after this button was painted.
        // Check its actual presentation, not the companion Lab's live cache.
        if action == LabAction::CreateCheckpoint && !self.lab_world_is_live(cx) {
            self.lab_workflows
                .set_notice("Show the live World window before saving or retrying a checkpoint.");
            cx.notify();
            return;
        }
        let return_live = action == LabAction::ReturnLive;
        if (action == LabAction::ReviewCheckpoint
            && !self.lab_workflows.is_busy()
            && self.lab_workflows.selected_checkpoint_metadata().is_some())
            || (matches!(&action, LabAction::SeekReplay { .. })
                && self.lab_workflows.can_seek_replay()
                && self.lab_workflows.replay_seek_action().as_ref() == Some(&action))
        {
            // A later World-window intent must supersede this reconstruction.
            self.timeline_cancelled
                .store(false, AtomicOrdering::Release);
        }
        let creating_checkpoint = action == LabAction::CreateCheckpoint;
        let previous_pending = self
            .lab_workflows
            .latest_checkpoint_save
            .as_ref()
            .filter(|save| matches!(&save.state, LabCheckpointSaveState::Pending))
            .map(|save| (save.session_id, save.request_id.clone()));
        self.lab_workflows.dispatch(action);
        // Observe only an actual transition into Pending for this save action,
        // including a retry. A cached old Pending or Saved cannot join a guide
        // that reached Checkpoint after that request began.
        if creating_checkpoint
            && let Some(save) = &self.lab_workflows.latest_checkpoint_save
            && matches!(&save.state, LabCheckpointSaveState::Pending)
            && previous_pending.as_ref().is_none_or(|(session, request)| {
                *session != save.session_id || request != &save.request_id
            })
            && let Ok(mut guide) = self.lab_guide.lock()
        {
            guide.observe_checkpoint_save(save);
        }
        if return_live {
            self.return_world_to_live(cx);
            self.lab_workflows.notice = self.world_review_feedback.clone();
        }
        cx.notify();
    }

    fn lab_button(
        &self,
        id: String,
        label: String,
        action: LabAction,
        enabled: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        let mouse_action = action.clone();
        div()
            .id(SharedString::from(id.clone()))
            .debug_selector(move || id.clone())
            .focusable()
            .tab_stop(true)
            .role(Role::Button)
            .aria_label(label.clone())
            .rounded_md()
            .border_1()
            .border_color(chrome::border())
            .px_2()
            .py_1()
            .text_xs()
            .text_color(if enabled {
                chrome::text_primary()
            } else {
                chrome::text_muted()
            })
            .child(label)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    if enabled {
                        this.lab_action(mouse_action.clone(), cx);
                    }
                }),
            )
            .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
                if matches!(event.keystroke.key.as_str(), "enter" | "space" | " ") {
                    if enabled && !event.is_held {
                        this.lab_action(action.clone(), cx);
                    }
                    cx.stop_propagation();
                }
            }))
    }

    fn lab_panel_header(&self, title: &str, cx: &mut Context<Self>) -> Div {
        let idle = !self.lab_workflows.is_busy();
        let mut header = div()
            .flex()
            .flex_col()
            .gap_2()
            .child(div().text_lg().child(title.to_owned()))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_2()
                    .child(self.lab_button(
                        "lab-refresh".into(),
                        "Refresh".into(),
                        LabAction::Refresh,
                        idle,
                        cx,
                    ))
                    .child(
                        self.lab_button(
                            "lab-cancel-request".into(),
                            if self
                                .lab_workflows
                                .latest_artifact_export
                                .as_ref()
                                .is_some_and(|export| {
                                    matches!(&export.state, LabArtifactExportState::Pending)
                                })
                            {
                                "Leave pending view; export may still complete".into()
                            } else {
                                "Discard pending result".into()
                            },
                            LabAction::CancelRequest,
                            !idle,
                            cx,
                        ),
                    ),
            );
        if !self.lab_workflows.viewing_artifacts {
            header = header.child(self.lab_button(
                "lab-artifacts-open".into(),
                "Artifacts / export".into(),
                LabAction::ArtifactsPage(None),
                true,
                cx,
            ));
            if self.lab_workflows.latest_artifact_export.is_some() {
                header = header.child(div().text_xs().child(
                    "A previous export request is retained. Open Artifacts / export to inspect its source, destination and outcome.",
                ));
            }
        }
        if self.lab_workflows.submit.is_none() {
            header = header.child(
                div()
                    .text_xs()
                    .child("Application data services are unavailable in this view."),
            );
        }
        if let Some(notice) = &self.lab_workflows.notice {
            header = header.child(
                div()
                    .text_xs()
                    .text_color(chrome::text_secondary())
                    .child(notice.clone()),
            );
        }
        header
    }

    fn render_lab_experiment_treatment(
        &self,
        treatment: &LabExperimentTreatment,
        capabilities: &LabExperimentCapabilities,
        horizon: u64,
        editing: bool,
        cx: &mut Context<Self>,
    ) -> Div {
        let mut parameters = div().flex().flex_wrap().gap_2();
        for offered in &capabilities.treatment_parameters {
            parameters = parameters.child(self.lab_button(
                format!("lab-experiment-parameter-{}", offered.parameter.key()),
                format!(
                    "{}{}",
                    if treatment.parameter == offered.parameter {
                        "Selected: "
                    } else {
                        ""
                    },
                    offered.parameter.label(),
                ),
                LabAction::TreatmentParameter(offered.parameter),
                editing,
                cx,
            ));
        }
        let mut scales = div().flex().flex_wrap().gap_2().child(
            self.lab_button(
                "lab-experiment-percent-prev".into(),
                "10 percentage points less".into(),
                LabAction::TreatmentPercent(
                    treatment
                        .scale_percent
                        .checked_sub(10)
                        .unwrap_or(treatment.scale_percent),
                ),
                editing && treatment.scale_percent >= 10,
                cx,
            ),
        );
        for percent in [0, 50, 100, 200, LAB_MAX_TREATMENT_PERCENT] {
            scales = scales.child(self.lab_button(
                format!("lab-experiment-percent-{percent}"),
                format!(
                    "{}{percent}% of baseline{}",
                    if treatment.scale_percent == percent {
                        "Selected: "
                    } else {
                        ""
                    },
                    if percent == 100 {
                        " (unchanged-value control)"
                    } else {
                        ""
                    },
                ),
                LabAction::TreatmentPercent(percent),
                editing,
                cx,
            ));
        }
        scales = scales.child(
            self.lab_button(
                "lab-experiment-percent-next".into(),
                "10 percentage points more".into(),
                LabAction::TreatmentPercent(
                    treatment
                        .scale_percent
                        .checked_add(10)
                        .unwrap_or(treatment.scale_percent),
                ),
                editing
                    && treatment
                        .scale_percent
                        .checked_add(10)
                        .is_some_and(|percent| percent <= LAB_MAX_TREATMENT_PERCENT),
                cx,
            ),
        );
        let mut boundaries = div().flex().flex_wrap().gap_2().child(
            self.lab_button(
                "lab-experiment-treatment-tick-prev".into(),
                "One boundary earlier".into(),
                LabAction::TreatmentTick(
                    treatment
                        .at_tick
                        .checked_sub(1)
                        .unwrap_or(treatment.at_tick),
                ),
                editing && treatment.at_tick > 0,
                cx,
            ),
        );
        for tick in [0, 10, 100] {
            if tick < horizon {
                boundaries = boundaries.child(self.lab_button(
                    format!("lab-experiment-treatment-tick-{tick}"),
                    format!(
                        "{}At completed tick {tick}",
                        if treatment.at_tick == tick {
                            "Selected: "
                        } else {
                            ""
                        },
                    ),
                    LabAction::TreatmentTick(tick),
                    editing,
                    cx,
                ));
            }
        }
        boundaries = boundaries.child(
            self.lab_button(
                "lab-experiment-treatment-tick-next".into(),
                "One boundary later".into(),
                LabAction::TreatmentTick(
                    treatment
                        .at_tick
                        .checked_add(1)
                        .unwrap_or(treatment.at_tick),
                ),
                editing
                    && treatment
                        .at_tick
                        .checked_add(1)
                        .is_some_and(|tick| tick < horizon),
                cx,
            ),
        );
        div()
            .flex()
            .flex_col()
            .gap_2()
            .child(parameters)
            .child(div().text_sm().child(format!(
                "Challenger value: {}% of the disclosed baseline (0–{LAB_MAX_TREATMENT_PERCENT}%).",
                treatment.scale_percent,
            )))
            .child(scales)
            .child(div().text_sm().child(format!(
                "Apply after completed tick {} and before the next transition; boundary must be earlier than horizon {horizon}.",
                treatment.at_tick,
            )))
            .child(boundaries)
    }

    fn render_lab_experiment_creation(&self, cx: &mut Context<Self>) -> Div {
        let state = &self.lab_workflows;
        let editing = state.can_edit_experiment_draft();
        let draft = state
            .latest_experiment_creation
            .as_ref()
            .filter(|creation| {
                !matches!(&creation.state, LabExperimentCreationState::Created { .. })
            })
            .map(|creation| &creation.draft)
            .unwrap_or(&state.experiment_draft);
        let mut section = div()
            .flex()
            .flex_col()
            .gap_3()
            .child(self.lab_button(
                "lab-experiment-back".into(),
                "Back to experiments".into(),
                LabAction::BackToExperiments,
                true,
                cx,
            ))
            .child(div().text_lg().child("Experiment in fresh worlds"))
            .child(div().text_xs().child(format!(
                "Application source session {}.",
                self.host.snapshot_hub().latest().session_id.get(),
            )));
        if let Some(creation) = &state.latest_experiment_creation {
            let current_source =
                self.host.snapshot_hub().latest().session_id == creation.session_id;
            section = section.child(div().text_xs().child(format!(
                "{}creation request {} · source session {}",
                if current_source {
                    ""
                } else {
                    "Previous-source "
                },
                creation.request_id,
                creation.session_id.get(),
            )));
            match &creation.state {
                LabExperimentCreationState::Pending => {
                    section = section.child(
                        div()
                            .text_xs()
                            .child("Awaiting the application's experiment receipt."),
                    );
                }
                LabExperimentCreationState::Retryable { reason } => {
                    section = section.child(
                        div()
                            .text_xs()
                            .child(format!("Creation is unconfirmed: {reason}")),
                    );
                }
                LabExperimentCreationState::Created { experiment_id } => {
                    section = section.child(div().text_xs().child(format!(
                        "Experiment receipt: {experiment_id}. Inspect its observed run status in the catalog.",
                    )));
                }
            }
            if !matches!(&creation.state, LabExperimentCreationState::Created { .. }) {
                section = section
                    .child(div().text_xs().child(format!(
                        "Retained plan: {} versus {} · first seed {} · {} seeds · {} ticks per run.",
                        creation.draft.baseline_family,
                        creation.draft.challenger_family,
                        creation.draft.first_seed,
                        creation.draft.seed_count,
                        creation.draft.ticks_per_run,
                    )))
                    .child(div().text_xs().child(if current_source {
                        "Retry retains the exact request and plan. Abandon clears the local retry; an admitted experiment may still exist in the catalog."
                    } else {
                        "This previous-source request cannot be retried here. Abandon clears the local retry only; an admitted experiment may still exist in the catalog."
                    }))
                    .child(self.lab_button(
                        "lab-experiment-abandon".into(),
                        "Abandon unconfirmed request".into(),
                        LabAction::AbandonExperimentRequest,
                        state.can_abandon_experiment_request(),
                        cx,
                    ));
            }
        }
        let mut question = div()
            .id("lab-experiment-question")
            .debug_selector(|| "lab-experiment-question".to_owned())
            .flex()
            .flex_col()
            .gap_1()
            .text_xs();
        for line in draft.question_lines() {
            question = question.child(div().child(line));
        }
        section = section.child(question);
        let Some(capabilities) = &state.experiment_capabilities else {
            return section.child(
                div()
                    .text_sm()
                    .child("Loading available brain families and experiment limits…"),
            );
        };
        section = section
            .child(
                div()
                    .text_sm()
                    .child(capabilities.launch_description.clone()),
            )
            .child(div().text_xs().child(format!(
                "Limits: {} seeds · {} ticks per run · {} total run ticks.",
                capabilities.max_seed_count,
                capabilities.max_ticks_per_run,
                capabilities.max_total_ticks,
            )));
        let design = draft.design();
        section = section.child(
            div()
                .flex()
                .flex_wrap()
                .gap_2()
                .child(self.lab_button(
                    "lab-experiment-design-brain-families".into(),
                    format!(
                        "{}Compare brain families{}",
                        if design == LabExperimentDesign::BrainFamilies {
                            "Selected: "
                        } else {
                            ""
                        },
                        if capabilities.families.len() < 2 {
                            " (needs two families)"
                        } else {
                            ""
                        }
                    ),
                    LabAction::ExperimentDesign(LabExperimentDesign::BrainFamilies),
                    editing && capabilities.families.len() >= 2,
                    cx,
                ))
                .child(self.lab_button(
                    "lab-experiment-design-ecological-treatment".into(),
                    format!(
                        "{}Compare ecological treatment{}",
                        if design == LabExperimentDesign::EcologicalTreatment {
                            "Selected: "
                        } else {
                            ""
                        },
                        if capabilities.treatment_parameters.is_empty() {
                            " (unavailable)"
                        } else {
                            ""
                        }
                    ),
                    LabAction::ExperimentDesign(LabExperimentDesign::EcologicalTreatment),
                    editing && !capabilities.treatment_parameters.is_empty(),
                    cx,
                )),
        );
        let families: &[(bool, &str)] = if design == LabExperimentDesign::EcologicalTreatment {
            &[(true, "Shared brain family in both arms")]
        } else {
            &[
                (true, "Baseline brain family"),
                (false, "Challenger brain family"),
            ]
        };
        for &(baseline, title) in families {
            let selected = if baseline {
                &draft.baseline_family
            } else {
                &draft.challenger_family
            };
            let mut choices = div().flex().flex_wrap().gap_2();
            for family in &capabilities.families {
                choices = choices.child(self.lab_button(
                    format!(
                        "lab-experiment-{}-{}",
                        if baseline { "baseline" } else { "challenger" },
                        family,
                    ),
                    format!(
                        "{}{}",
                        if selected == family { "Selected: " } else { "" },
                        family,
                    ),
                    if baseline {
                        LabAction::DraftBaseline(family.clone())
                    } else {
                        LabAction::DraftChallenger(family.clone())
                    },
                    editing,
                    cx,
                ));
            }
            section = section.child(div().text_sm().child(title)).child(choices);
        }
        if let Some(treatment) = &draft.treatment {
            section = section.child(self.render_lab_experiment_treatment(
                treatment,
                capabilities,
                draft.ticks_per_run,
                editing,
                cx,
            ));
        }
        let mut first_seed = div().flex().flex_wrap().gap_2();
        first_seed = first_seed.child(self.lab_button(
            "lab-experiment-seed-prev".into(),
            "Previous first seed".into(),
            LabAction::FirstSeed(draft.first_seed.checked_sub(1).unwrap_or(draft.first_seed)),
            editing && draft.first_seed > 0,
            cx,
        ));
        for seed in [11, 42, 137] {
            first_seed = first_seed.child(self.lab_button(
                format!("lab-experiment-seed-{seed}"),
                format!(
                    "{}First seed {seed}",
                    if draft.first_seed == seed {
                        "Selected: "
                    } else {
                        ""
                    },
                ),
                LabAction::FirstSeed(seed),
                editing,
                cx,
            ));
        }
        first_seed = first_seed.child(self.lab_button(
            "lab-experiment-seed-next".into(),
            "Next first seed".into(),
            LabAction::FirstSeed(draft.first_seed.checked_add(1).unwrap_or(draft.first_seed)),
            editing && draft.first_seed < u64::MAX,
            cx,
        ));
        let mut counts = div().flex().flex_wrap().gap_2().child(self.lab_button(
            "lab-experiment-seed-count-prev".into(),
            "One fewer seed".into(),
            LabAction::SeedCount(draft.seed_count.checked_sub(1).unwrap_or(draft.seed_count)),
            editing && draft.seed_count > 1,
            cx,
        ));
        for count in [1, 3, 8] {
            if count <= capabilities.max_seed_count {
                counts = counts.child(self.lab_button(
                    format!("lab-experiment-seed-count-{count}"),
                    format!(
                        "{}{count} seed{}",
                        if draft.seed_count == count {
                            "Selected: "
                        } else {
                            ""
                        },
                        if count == 1 { "" } else { "s" },
                    ),
                    LabAction::SeedCount(count),
                    editing,
                    cx,
                ));
            }
        }
        counts = counts.child(
            self.lab_button(
                "lab-experiment-seed-count-next".into(),
                "One more seed".into(),
                LabAction::SeedCount(draft.seed_count.checked_add(1).unwrap_or(draft.seed_count)),
                editing
                    && draft
                        .seed_count
                        .checked_add(1)
                        .is_some_and(|count| count <= capabilities.max_seed_count),
                cx,
            ),
        );
        let mut horizons = div().flex().flex_wrap().gap_2().child(
            self.lab_button(
                "lab-experiment-ticks-prev".into(),
                "One fewer tick".into(),
                LabAction::TicksPerRun(
                    draft
                        .ticks_per_run
                        .checked_sub(1)
                        .unwrap_or(draft.ticks_per_run),
                ),
                editing && draft.ticks_per_run > 1,
                cx,
            ),
        );
        for ticks in [1, 10, 100, 500, 1_000, 10_000] {
            if ticks <= capabilities.max_ticks_per_run {
                horizons = horizons.child(self.lab_button(
                    format!("lab-experiment-ticks-{ticks}"),
                    format!(
                        "{}{ticks} ticks per run",
                        if draft.ticks_per_run == ticks {
                            "Selected: "
                        } else {
                            ""
                        },
                    ),
                    LabAction::TicksPerRun(ticks),
                    editing,
                    cx,
                ));
            }
        }
        horizons = horizons.child(
            self.lab_button(
                "lab-experiment-ticks-next".into(),
                "One more tick".into(),
                LabAction::TicksPerRun(
                    draft
                        .ticks_per_run
                        .checked_add(1)
                        .unwrap_or(draft.ticks_per_run),
                ),
                editing
                    && draft
                        .ticks_per_run
                        .checked_add(1)
                        .is_some_and(|ticks| ticks <= capabilities.max_ticks_per_run),
                cx,
            ),
        );
        section = section
            .child(
                div()
                    .text_sm()
                    .child(format!("First seed: {}", draft.first_seed)),
            )
            .child(first_seed)
            .child(
                div()
                    .text_sm()
                    .child(format!("Matched seed cohort: {}", draft.seed_count)),
            )
            .child(counts)
            .child(
                div()
                    .text_sm()
                    .child(format!("Scientific horizon: {} ticks", draft.ticks_per_run,)),
            )
            .child(horizons);
        match draft.seeds() {
            Ok(seeds) => {
                section = section.child(
                    div()
                        .text_xs()
                        .child(format!("Exact seeds in each arm: {seeds:?}",)),
                );
            }
            Err(error) => {
                section = section.child(
                    div()
                        .text_xs()
                        .child(format!("Seed schedule refused: {error}")),
                );
            }
        }
        let total_runs = draft.seed_count.checked_mul(2);
        let total_ticks = total_runs
            .and_then(|runs| u64::try_from(runs).ok())
            .and_then(|runs| runs.checked_mul(draft.ticks_per_run));
        section = section.child(
            div()
                .id("lab-experiment-preview")
                .debug_selector(|| "lab-experiment-preview".to_owned())
                .text_xs()
                .child(format!(
                    "Review: baseline {} · challenger {} · {} seeds per arm · {} ticks per run · {} total runs · {} total scientific ticks. Runs execute sequentially, one owner at a time.",
                    draft.baseline_family,
                    draft.challenger_family,
                    draft.seed_count,
                    draft.ticks_per_run,
                    total_runs
                        .map(|count| count.to_string())
                        .unwrap_or_else(|| "overflow".into()),
                    total_ticks
                        .map(|ticks| ticks.to_string())
                        .unwrap_or_else(|| "overflow".into()),
                )),
        );
        if let Err(reason) = draft.validate(capabilities) {
            section = section.child(
                div()
                    .text_xs()
                    .child(format!("Plan needs attention: {reason}")),
            );
        }
        section.child(
            self.lab_button(
                "lab-experiment-create".into(),
                if state.is_busy() {
                    "Awaiting experiment service"
                } else if state.will_retry_experiment_creation() {
                    "Retry this exact experiment"
                } else {
                    "Run this experiment"
                }
                .into(),
                LabAction::CreateExperiment,
                state.can_create_experiment(),
                cx,
            ),
        )
    }

    fn render_lab_artifacts(&self, cx: &mut Context<Self>) -> Div {
        let state = &self.lab_workflows;
        let session = self.host.snapshot_hub().latest().session_id;
        let idle = !state.is_busy();
        let mut panel = div()
            .flex()
            .flex_col()
            .gap_3()
            .child(self.lab_panel_header("Artifacts / export", cx))
            .child(self.lab_button(
                "lab-artifacts-back".into(),
                "Back to previous Lab view".into(),
                LabAction::BackFromArtifacts,
                true,
                cx,
            ))
            .child(div().text_xs().child(
                "All retained artifacts from this application source. Export copies one verified artifact and records its source and checksums without changing the live world.",
            ));
        if let Some(export) = &state.latest_artifact_export {
            let mut evidence = div()
                .id("lab-artifact-export-status")
                .debug_selector(|| "lab-artifact-export-status".to_owned())
                .flex()
                .flex_col()
                .gap_1();
            if export.session_id != session {
                evidence = evidence.child(div().text_xs().child(
                    "Historical export: this request belongs to a previous application source.",
                ));
            }
            for line in artifact_export_lines(export) {
                evidence = evidence.child(div().text_xs().child(line));
            }
            panel = panel.child(evidence);
        }
        let Some(page) = &state.artifacts else {
            return panel.child(
                "No artifact page has been observed. Refresh when application services are available; retained export outcomes remain above.",
            );
        };
        panel = panel
            .child(div().text_xs().child(format!(
                "{} retained artifacts · source session {}",
                page.total,
                session.get(),
            )))
            .child(div().text_xs().child(format!(
                "Export directory on the application host: {}",
                page.export_directory,
            )));
        for (index, artifact) in page.items.iter().take(LAB_PAGE_SIZE).enumerate() {
            let selected = state.selected_artifact.as_ref() == Some(&artifact.artifact_id);
            panel = panel.child(self.lab_button(
                format!("lab-artifact-{index}"),
                format!(
                    "{}{} · {} · {} bytes",
                    if selected { "Selected: " } else { "" },
                    artifact.filename,
                    artifact.artifact_id,
                    artifact.byte_size,
                ),
                LabAction::SelectArtifact(artifact.artifact_id.clone()),
                true,
                cx,
            ));
        }
        if page.items.is_empty() {
            panel = panel.child("No retained artifacts are available from this source.");
        }
        if let Some(artifact) = state.selected_artifact_metadata() {
            let mut metadata = div()
                .id("lab-artifact-metadata")
                .debug_selector(|| "lab-artifact-metadata".to_owned())
                .flex()
                .flex_col()
                .gap_1();
            for line in [
                format!("Artifact: {} · {}", artifact.artifact_id, artifact.content_type),
                format!("Source path: {}", artifact.relative_path),
                format!("Created: {} · {} bytes", artifact.created_at_utc, artifact.byte_size),
                format!("BLAKE3: {}", artifact.checksum_blake3),
                format!("SHA-256: {}", artifact.checksum_sha256),
                format!(
                    "Destination: {}",
                    std::path::Path::new(&page.export_directory)
                        .join("<unique request ID>")
                        .join(&artifact.filename)
                        .display(),
                ),
                "A new request folder is assigned only when Export selected is pressed. It also contains export-manifest.json; existing paths are refused.".to_owned(),
            ] {
                metadata = metadata.child(div().text_xs().child(line));
            }
            panel = panel.child(metadata);
            if let Some(action) = state.export_artifact_action() {
                panel = panel.child(self.lab_button(
                    "lab-artifact-export".into(),
                    "Export selected".into(),
                    action,
                    state.can_export_artifact(),
                    cx,
                ));
            } else if artifact.byte_size > LAB_MAX_ARTIFACT_BYTES as u64 {
                panel = panel.child(div().text_xs().child(format!(
                    "Export unavailable: {} bytes exceeds the {LAB_MAX_ARTIFACT_BYTES}-byte native export limit. Metadata remains inspectable.",
                    artifact.byte_size,
                )));
            } else {
                panel = panel.child(
                    "Export unavailable until the current application source and its data services are available.",
                );
            }
        } else {
            panel =
                panel.child("Select a retained artifact to review its metadata and destination.");
        }
        panel.child(
            div()
                .flex()
                .flex_wrap()
                .gap_2()
                .child(
                    self.lab_button(
                        "lab-artifacts-prev".into(),
                        "Previous artifacts".into(),
                        LabAction::ArtifactsPage(
                            (page.offset > LAB_PAGE_SIZE)
                                .then(|| (page.offset - LAB_PAGE_SIZE).to_string()),
                        ),
                        idle && page.offset > 0,
                        cx,
                    ),
                )
                .child(self.lab_button(
                    "lab-artifacts-next".into(),
                    "Next artifacts".into(),
                    LabAction::ArtifactsPage(page.next_cursor.clone()),
                    idle && page.next_cursor.is_some(),
                    cx,
                )),
        )
    }

    pub(super) fn render_lab_experiments(&self, cx: &mut Context<Self>) -> Div {
        let state = &self.lab_workflows;
        if state.viewing_artifacts {
            return self.render_lab_artifacts(cx);
        }
        let idle = !state.is_busy();
        let mut panel = div()
            .flex()
            .flex_col()
            .gap_3()
            .child(self.lab_panel_header("Experiments", cx));
        if state.composing_experiment {
            return panel.child(self.render_lab_experiment_creation(cx));
        }
        if state.selected_experiment.is_some() {
            panel = panel.child(self.lab_button(
                "lab-all-experiments".into(),
                "All experiments".into(),
                LabAction::AllExperiments,
                true,
                cx,
            ));
            let Some(batch) = &state.batch else {
                return panel.child("Loading selected experiment…");
            };
            let summary = &batch.summary;
            panel = panel
                .child(div().text_sm().child(format!(
                    "{} · {} · {}/{} completed · {} failed",
                    summary.experiment_id,
                    summary.status,
                    summary.completed_runs,
                    summary.total_runs,
                    summary.failed_runs
                )))
                .child(div().text_xs().child(format!(
                    "Source generation {} · plan {}",
                    batch.generation, batch.plan_digest
                )))
                .child(
                    div()
                        .flex()
                        .gap_2()
                        .child(self.lab_button(
                            "lab-stop-experiment".into(),
                            "Stop at durable boundary".into(),
                            LabAction::CancelExperiment,
                            idle && matches!(summary.status.as_str(), "running" | "pending"),
                            cx,
                        ))
                        .child(self.lab_button(
                            "lab-resume-experiment".into(),
                            "Resume unfinished runs".into(),
                            LabAction::ResumeExperiment,
                            idle && !matches!(
                                summary.status.as_str(),
                                "running" | "pending" | "cancelling"
                            ) && summary.completed_runs + summary.failed_runs < summary.total_runs,
                            cx,
                        )),
                );
            if let Some(action) = state.failed_experiment_retry_action() {
                panel = panel
                    .child(self.lab_button(
                        "lab-retry-failed-experiment".into(),
                        "Retry failed runs".into(),
                        action,
                        true,
                        cx,
                    ))
                    .child(div().text_xs().child(
                        "Retry failed runs from their saved continuations and resume unfinished runs. Completed results and prior failure evidence are retained.",
                    ));
            }
            if let Some(error) = &batch.error_reason {
                panel = panel.child(div().text_xs().child(error.clone()));
            }
            if let Some(creation) = state.latest_experiment_creation.as_ref().filter(|creation| {
                creation.session_id == self.host.snapshot_hub().latest().session_id
                    && matches!(&creation.state, LabExperimentCreationState::Created { experiment_id }
                        if experiment_id == &summary.experiment_id)
            }) {
                let mut question = div()
                    .id("lab-experiment-retained-question")
                    .debug_selector(|| "lab-experiment-retained-question".to_owned())
                    .flex()
                    .flex_col()
                    .gap_1()
                    .text_xs()
                    .child("Retained launch question:");
                for line in creation.draft.question_lines() {
                    question = question.child(div().child(line));
                }
                panel = panel.child(question);
            }
            for (index, variant) in batch
                .variants
                .iter()
                .enumerate()
                .skip(state.variant_offset)
                .take(VARIANTS_PER_PAGE)
            {
                panel = panel.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(div().text_xs().child(format!(
                            "{} ({}) · {}/{} completed · {} running · {} failed",
                            variant.variant_id,
                            variant.brain_family,
                            variant.completed_runs,
                            variant.total_runs,
                            variant.running_runs,
                            variant.failed_runs
                        )))
                        .child(
                            div()
                                .flex()
                                .gap_2()
                                .child(
                                    self.lab_button(
                                        format!("lab-baseline-{index}"),
                                        if state.baseline.as_ref() == Some(&variant.variant_id) {
                                            "Baseline selected"
                                        } else {
                                            "Use as baseline"
                                        }
                                        .into(),
                                        LabAction::Baseline(variant.variant_id.clone()),
                                        true,
                                        cx,
                                    ),
                                )
                                .child(
                                    self.lab_button(
                                        format!("lab-challenger-{index}"),
                                        if state.challenger.as_ref() == Some(&variant.variant_id) {
                                            "Challenger selected"
                                        } else {
                                            "Use as challenger"
                                        }
                                        .into(),
                                        LabAction::Challenger(variant.variant_id.clone()),
                                        true,
                                        cx,
                                    ),
                                ),
                        ),
                );
            }
            let mut metrics = div().flex().flex_wrap().gap_2();
            for metric in LabMetric::ALL {
                metrics = metrics.child(self.lab_button(
                    format!("lab-metric-{}", metric.key()),
                    format!(
                        "{}{} ({})",
                        if state.metric == metric {
                            "Selected: "
                        } else {
                            ""
                        },
                        metric.label(),
                        metric.unit(),
                    ),
                    LabAction::Metric(metric),
                    true,
                    cx,
                ));
            }
            panel = panel
                .child(
                    div()
                        .flex()
                        .gap_2()
                        .child(self.lab_button(
                            "lab-variants-prev".into(),
                            "Previous variants".into(),
                            LabAction::VariantsPage(
                                state.variant_offset.saturating_sub(VARIANTS_PER_PAGE),
                            ),
                            state.variant_offset > 0,
                            cx,
                        ))
                        .child(self.lab_button(
                            "lab-variants-next".into(),
                            "Next variants".into(),
                            LabAction::VariantsPage(state.variant_offset + VARIANTS_PER_PAGE),
                            state.variant_offset + VARIANTS_PER_PAGE < batch.variants.len(),
                            cx,
                        )),
                )
                .child(div().text_sm().child("Compare endpoint"))
                .child(metrics)
                .child(div().text_xs().child(format!(
                    "{} comparison ({}): {} − {}. Complete matched seed cohorts are required.",
                    state.metric.label(),
                    state.metric.unit(),
                    state.challenger.as_deref().unwrap_or("choose challenger"),
                    state.baseline.as_deref().unwrap_or("choose baseline")
                )))
                .child(self.lab_button(
                    "lab-compare".into(),
                    format!("Compare verified {} endpoints", state.metric.label()),
                    LabAction::Compare,
                    idle && state.comparison_request().is_some(),
                    cx,
                ));
            if let Some(comparison) = &state.comparison {
                panel = panel.child(
                    div().flex().flex_col().gap_1().children(
                        comparison_lines(comparison)
                            .into_iter()
                            .map(|line| div().text_xs().child(line)),
                    ),
                );
            }
            return panel;
        }
        panel = panel.child(self.lab_button(
            "lab-new-experiment".into(),
            "New matched-seed experiment".into(),
            LabAction::NewExperiment,
            state.submit.is_some(),
            cx,
        ));
        let Some(page) = &state.experiments else {
            return panel.child("Loading experiment catalog…");
        };
        panel = panel.child(div().text_xs().child(format!(
            "{} experiments · reserved run slots {}/{}",
            page.total, page.reserved_run_slots, page.max_run_slots
        )));
        for (index, row) in page.items.iter().enumerate() {
            panel = panel.child(self.lab_button(
                format!("lab-experiment-{index}"),
                format!(
                    "{} · {} · {}/{} completed · {} failed",
                    row.experiment_id,
                    row.status,
                    row.completed_runs,
                    row.total_runs,
                    row.failed_runs
                ),
                LabAction::SelectExperiment(row.experiment_id.clone()),
                true,
                cx,
            ));
        }
        if page.items.is_empty() {
            panel =
                panel.child(div().text_sm().child(
                    "No retained experiment batches. Start a matched-seed experiment above.",
                ));
        }
        panel.child(
            div()
                .flex()
                .gap_2()
                .child(
                    self.lab_button(
                        "lab-experiments-prev".into(),
                        "Previous experiments".into(),
                        LabAction::ExperimentsPage(
                            (page.offset > LAB_PAGE_SIZE)
                                .then(|| (page.offset - LAB_PAGE_SIZE).to_string()),
                        ),
                        idle && page.offset > 0,
                        cx,
                    ),
                )
                .child(self.lab_button(
                    "lab-experiments-next".into(),
                    "Next experiments".into(),
                    LabAction::ExperimentsPage(page.next_cursor.clone()),
                    idle && page.next_cursor.is_some(),
                    cx,
                )),
        )
    }

    fn render_lab_checkpoint_save(&self, cx: &mut Context<Self>) -> Div {
        let state = &self.lab_workflows;
        let live = self.lab_world_is_live(cx);
        let source = self.host.snapshot_hub().latest();
        let saving = state.latest_checkpoint_save.as_ref().is_some_and(|save| {
            save.session_id == source.session_id
                && matches!(&save.state, LabCheckpointSaveState::Pending)
                && state.is_busy()
        });
        let label = if saving {
            "Saving checkpoint..."
        } else if state.will_retry_checkpoint_save() {
            "Retry previous checkpoint save"
        } else {
            "Save live checkpoint"
        };
        let mut section = div().flex().flex_col().gap_2()
            .child(self.lab_button(
                "lab-save-checkpoint".into(), label.into(), LabAction::CreateCheckpoint,
                live && state.can_create_checkpoint(), cx,
            ))
            .child(div().text_xs().child(
                "A new save records the live world. Its durable receipt identifies the exact saved tick.",
            ));
        if !live {
            section = section.child(
                div()
                    .text_xs()
                    .child("Return World to live before saving or retrying a checkpoint."),
            );
        }
        if let Some(save) = &state.latest_checkpoint_save {
            let current_source = source.session_id == save.session_id;
            if !current_source {
                section = section.child(div().text_xs().child(
                    "Previous source save: this receipt or unconfirmed request belongs to an earlier world source.",
                ));
            }
            match &save.state {
                LabCheckpointSaveState::Pending => {
                    section = section.child(
                        div()
                            .text_xs()
                            .child("Saving checkpoint - awaiting a durable receipt."),
                    );
                }
                LabCheckpointSaveState::Retryable { reason } => {
                    section = section.child(div().text_xs().child(format!("Save not confirmed: {reason}")))
                        .child(div().text_xs().child(if current_source {
                            "Retry previous checkpoint save uses the same request identity."
                        } else {
                            "Saving this live source starts a new request after the previous worker finishes."
                        }));
                }
                LabCheckpointSaveState::Saved(checkpoint) => {
                    section = section
                        .child(div().text_xs().child(format!(
                            "Saved checkpoint {} · tick {} · {} agents · {} bytes",
                            checkpoint.checkpoint_id,
                            checkpoint.tick,
                            checkpoint.agent_count,
                            checkpoint.byte_size,
                        )))
                        .child(
                            div()
                                .text_xs()
                                .child(format!("Saved world digest {}", checkpoint.world_digest,)),
                        );
                }
            }
            section = section.child(
                div()
                    .text_xs()
                    .child(format!("Save request: {}", save.request_id)),
            );
        }
        section
    }

    fn render_lab_branch_composer(&self, cx: &mut Context<Self>) -> Div {
        let state = &self.lab_workflows;
        let editing = state.can_edit_branch_draft();
        let retained = state
            .latest_branch_creation
            .as_ref()
            .filter(|creation| !matches!(&creation.state, LabBranchCreationState::Created { .. }));
        let draft = retained
            .map(|creation| &creation.draft)
            .or(state.branch_draft.as_ref());
        let mut panel = div()
            .flex()
            .flex_col()
            .gap_3()
            .child(div().text_lg().child("A future from a saved checkpoint"))
            .child(div().text_xs().child(format!(
                "Application source session {}.",
                self.host.snapshot_hub().latest().session_id.get(),
            )));
        if let Some(creation) = retained {
            let current_source =
                self.host.snapshot_hub().latest().session_id == creation.session_id;
            panel = panel.child(div().text_xs().child(format!(
                "{}study request {} · source session {}",
                if current_source {
                    "Retained "
                } else {
                    "Previous-source "
                },
                creation.request_id,
                creation.session_id.get(),
            )));
            panel = panel.child(div().text_xs().child(match &creation.state {
                LabBranchCreationState::Pending => {
                    "Awaiting the application's study admission receipt.".to_owned()
                }
                LabBranchCreationState::Retryable { reason } => {
                    format!("Study admission is unconfirmed: {reason}")
                }
                LabBranchCreationState::Created { .. } => String::new(),
            }));
            panel = panel
                .child(div().text_xs().child(if current_source {
                    "Retry preserves this exact saved source, treatment, horizon and request. Abandon clears the local retry only; an admitted study may still exist in the catalog."
                } else {
                    "This previous-source request cannot be retried here. Abandon clears the local retry only; an admitted study may still exist in the catalog."
                }))
                .child(self.lab_button(
                    "lab-branch-abandon".into(),
                    "Abandon unconfirmed study request".into(),
                    LabAction::AbandonCheckpointBranchRequest,
                    state.can_abandon_branch_request(),
                    cx,
                ));
        }
        let Some(draft) = draft else {
            return panel.child("Select a retained checkpoint to configure its future.");
        };
        panel = panel
            .child(div().text_sm().child(format!(
                "Saved source: {} · completed tick {} · {} agents",
                draft.checkpoint.checkpoint_id,
                draft.checkpoint.tick,
                draft.checkpoint.agent_count,
            )))
            .child(div().text_xs().child(format!(
                "Saved world digest: {}",
                draft.checkpoint.world_digest,
            )))
            .child(div().text_xs().child(
                "Each future starts from this exact saved scientific state. The live owner continues separately; its session, scheduled commands and history are not transferred.",
            ));
        let Some(capabilities) = &state.branch_capabilities else {
            return panel.child("Loading checkpoint-study limits…");
        };
        panel = panel.child(div().text_xs().child(format!(
            "Limits: {} future ticks per branch · {} source agents.",
            capabilities.max_ticks, capabilities.max_population,
        )));
        let mut treatments = div().flex().flex_wrap().gap_2();
        for treatment in LabBranchTreatment::ALL {
            treatments = treatments.child(self.lab_button(
                format!("lab-branch-treatment-{}", treatment.key()),
                format!(
                    "{}{}",
                    if draft.treatment == treatment {
                        "Selected: "
                    } else {
                        ""
                    },
                    treatment.label(),
                ),
                LabAction::BranchTreatment(treatment),
                editing,
                cx,
            ));
        }
        panel = panel
            .child(div().text_sm().child("Treatment"))
            .child(treatments)
            .child(div().text_xs().child(draft.treatment.description()));
        let mut horizons = div().flex().flex_wrap().gap_2().child(self.lab_button(
            "lab-branch-ticks-prev".into(),
            "One fewer future tick".into(),
            LabAction::BranchTicks(draft.ticks.checked_sub(1).unwrap_or(draft.ticks)),
            editing && draft.ticks > 1,
            cx,
        ));
        for ticks in [1, 10, 100, 240, 1_000, 10_000] {
            if ticks <= capabilities.max_ticks {
                horizons = horizons.child(self.lab_button(
                    format!("lab-branch-ticks-{ticks}"),
                    format!(
                        "{}{ticks} future ticks",
                        if draft.ticks == ticks {
                            "Selected: "
                        } else {
                            ""
                        },
                    ),
                    LabAction::BranchTicks(ticks),
                    editing,
                    cx,
                ));
            }
        }
        horizons = horizons.child(
            self.lab_button(
                "lab-branch-ticks-next".into(),
                "One more future tick".into(),
                LabAction::BranchTicks(draft.ticks.checked_add(1).unwrap_or(draft.ticks)),
                editing
                    && draft
                        .ticks
                        .checked_add(1)
                        .is_some_and(|ticks| ticks <= capabilities.max_ticks),
                cx,
            ),
        );
        let target = draft.checkpoint.tick.checked_add(draft.ticks);
        let total_ticks = draft.treatment.branch_count().checked_mul(draft.ticks);
        panel = panel
            .child(div().text_sm().child(format!("Future horizon: {} ticks", draft.ticks)))
            .child(horizons)
            .child(
                div()
                    .id("lab-branch-preview")
                    .debug_selector(|| "lab-branch-preview".to_owned())
                    .text_xs()
                    .child(format!(
                        "Review: {} · {} future{} · source tick {} → target tick {} · {} total scientific transitions.",
                        draft.treatment.label(),
                        draft.treatment.branch_count(),
                        if draft.treatment.branch_count() == 1 { "" } else { "s" },
                        draft.checkpoint.tick,
                        target.map(|tick| tick.to_string()).unwrap_or_else(|| "overflow".into()),
                        total_ticks.map(|ticks| ticks.to_string()).unwrap_or_else(|| "overflow".into()),
                    )),
            )
            .child(div().text_xs().child(
                "Outcomes describe one saved starting state. Repeated ticks are not independent seed replicates. Any selected intervention applies at the saved boundary before its next tick.",
            ));
        if let Err(reason) = draft.validate(capabilities) {
            panel = panel.child(
                div()
                    .text_xs()
                    .child(format!("Plan needs attention: {reason}")),
            );
        }
        panel.child(
            self.lab_button(
                "lab-branch-create".into(),
                if state.is_busy() {
                    "Awaiting study service"
                } else if state.will_retry_branch_creation() {
                    "Retry this exact checkpoint study"
                } else {
                    "Run this checkpoint study"
                }
                .into(),
                LabAction::CreateCheckpointBranch,
                state.can_create_checkpoint_branch(),
                cx,
            ),
        )
    }

    fn render_lab_checkpoint_studies(&self, cx: &mut Context<Self>) -> Div {
        let state = &self.lab_workflows;
        let idle = !state.is_busy();
        let mut panel = div()
            .flex()
            .flex_col()
            .gap_3()
            .child(self.lab_panel_header("Checkpoint studies", cx))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_2()
                    .child(self.lab_button(
                        "lab-branch-back".into(),
                        "Back to saved checkpoints".into(),
                        LabAction::BackToCheckpoints,
                        true,
                        cx,
                    ))
                    .child(self.lab_button(
                        "lab-branches-catalog".into(),
                        "Retained checkpoint studies".into(),
                        LabAction::CheckpointBranchesPage(None),
                        true,
                        cx,
                    )),
            );
        match state.checkpoint_view {
            LabCheckpointView::BranchComposer => panel.child(self.render_lab_branch_composer(cx)),
            LabCheckpointView::BranchCatalog => {
                let Some(page) = &state.checkpoint_branches else {
                    return panel.child("Loading retained checkpoint studies…");
                };
                panel = panel.child(div().text_xs().child(format!(
                    "{} retained studies. Reopen an observation without starting another future.",
                    page.total,
                )));
                if page.items.is_empty() {
                    panel = panel.child("No checkpoint studies are retained yet.");
                }
                for (index, branch) in page.items.iter().enumerate() {
                    panel = panel.child(self.lab_button(
                        format!("lab-branch-{index}"),
                        format!(
                            "{} · {} · source {} at tick {} · target tick {}",
                            branch.branch_name,
                            branch.status,
                            branch.checkpoint_id,
                            branch.source_tick,
                            branch.target_tick,
                        ),
                        LabAction::SelectCheckpointBranch(branch.branch_id.clone()),
                        true,
                        cx,
                    ));
                }
                panel.child(
                    div()
                        .flex()
                        .gap_2()
                        .child(
                            self.lab_button(
                                "lab-branches-prev".into(),
                                "Previous studies".into(),
                                LabAction::CheckpointBranchesPage(
                                    (page.offset > LAB_PAGE_SIZE)
                                        .then(|| (page.offset - LAB_PAGE_SIZE).to_string()),
                                ),
                                idle && page.offset > 0,
                                cx,
                            ),
                        )
                        .child(self.lab_button(
                            "lab-branches-next".into(),
                            "Next studies".into(),
                            LabAction::CheckpointBranchesPage(page.next_cursor.clone()),
                            idle && page.next_cursor.is_some(),
                            cx,
                        )),
                )
            }
            LabCheckpointView::BranchDetail => {
                let Some(branch) = &state.branch else {
                    return panel
                        .child("No checkpoint study observation has arrived. Refresh to retry.");
                };
                panel = panel
                    .child(div().text_lg().child(branch.branch_name.clone()))
                    .child(div().text_sm().child(format!("Observed status: {}", branch.status)))
                    .child(div().text_xs().child(format!("Study: {}", branch.branch_id)))
                    .child(div().text_xs().child(format!(
                        "Saved source: {} · tick {} · world digest {}",
                        branch.checkpoint_id, branch.source_tick, branch.world_digest,
                    )))
                    .child(div().text_xs().child(format!(
                        "Requested target tick: {}. Source and target describe the plan; they are not a live progress counter.",
                        branch.target_tick,
                    )))
                    .child(self.lab_button(
                        "lab-branch-cancel".into(),
                        "Request study cancellation".into(),
                        LabAction::CancelCheckpointBranch,
                        state.can_cancel_checkpoint_branch(),
                        cx,
                    ));
                if let Some(reason) = &branch.error_reason {
                    panel = panel.child(div().text_xs().child(format!("Study outcome: {reason}")));
                }
                if let Some(artifact_id) = &branch.artifact_id {
                    panel = panel.child(
                        div()
                            .text_xs()
                            .child(format!("Report artifact: {artifact_id}",)),
                    );
                }
                panel = panel.child(self.lab_button(
                    "lab-branch-read-report".into(),
                    "Read verified outcome report".into(),
                    LabAction::ReadCheckpointBranchReport,
                    state.can_read_branch_report(),
                    cx,
                ));
                if let Some(report) = &state.branch_report {
                    let mut evidence = div()
                        .id("lab-branch-report")
                        .debug_selector(|| "lab-branch-report".to_owned())
                        .flex()
                        .flex_col()
                        .gap_2();
                    for line in branch_report_lines(report) {
                        evidence = evidence.child(div().text_xs().child(line));
                    }
                    panel = panel.child(evidence);
                }
                panel
            }
            LabCheckpointView::SavedCuts => panel,
        }
    }

    fn render_lab_replay(&self, cx: &mut Context<Self>) -> Div {
        let state = &self.lab_workflows;
        let mut panel = div()
            .flex()
            .flex_col()
            .gap_3()
            .child(self.lab_panel_header("Replay / seek", cx))
            .child(self.lab_button(
                "lab-replay-back".into(),
                "Back to saved checkpoints".into(),
                LabAction::BackFromReplay,
                true,
                cx,
            ))
            .child(self.lab_button(
                "lab-replay-live".into(),
                "Return World to live".into(),
                LabAction::ReturnLive,
                true,
                cx,
            ))
            .child(div().text_xs().child(
                "Review an exact tick from the selected saved cut. Seeking reconstructs an independent historical view; it never steps or replaces the live owner.",
            ));
        let Some(checkpoint) = &state.replay_checkpoint else {
            return panel.child("Select a saved checkpoint before opening replay.");
        };
        panel = panel
            .child(div().text_sm().child(format!(
                "Source checkpoint {} · saved tick {} · {} agents · {} bytes",
                checkpoint.checkpoint_id,
                checkpoint.tick,
                checkpoint.agent_count,
                checkpoint.byte_size,
            )))
            .child(
                div()
                    .text_xs()
                    .child(format!("Saved world digest: {}", checkpoint.world_digest,)),
            );
        if let Some(capabilities) = &state.replay_capabilities {
            panel = panel.child(div().text_xs().child(format!(
                "Application session {} · observed live tick {} · maximum saved-cut-to-current verification interval {} ticks",
                capabilities.session_id.get(),
                capabilities.observed_tick,
                capabilities.max_verification_ticks,
            )));
        } else {
            panel = panel.child(
                "Replay bounds have not arrived. Refresh to observe this application's current tick and verification limit.",
            );
        }
        let target = div()
            .id("lab-replay-target")
            .debug_selector(|| "lab-replay-target".to_owned())
            .focusable()
            .tab_stop(true)
            .role(Role::SpinButton)
            .aria_label(format!("Replay target tick: {}", state.replay_tick_text))
            .rounded_md()
            .border_1()
            .border_color(chrome::border())
            .px_3()
            .py_2()
            .text_sm()
            .child(format!(
                "Target tick: {}",
                if state.replay_tick_text.is_empty() {
                    "(type decimal digits)"
                } else {
                    &state.replay_tick_text
                },
            ))
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, window, cx| {
                if !this.lab_workflows.viewing_replay {
                    return;
                }
                let key = event.keystroke.key.as_str();
                let modifiers = event.keystroke.modifiers;
                if (modifiers.control || modifiers.platform) && key.eq_ignore_ascii_case("a") {
                    this.lab_action(LabAction::ReplayTickText(String::new()), cx);
                } else if !modifiers.control && !modifiers.alt && !modifiers.platform {
                    match key {
                        "backspace" => {
                            let mut text = this.lab_workflows.replay_tick_text.clone();
                            text.pop();
                            this.lab_action(LabAction::ReplayTickText(text), cx);
                        }
                        "delete" => {
                            this.lab_action(LabAction::ReplayTickText(String::new()), cx);
                        }
                        "enter" if !event.is_held => {
                            if this.lab_workflows.can_seek_replay()
                                && let Some(action) = this.lab_workflows.replay_seek_action()
                            {
                                this.lab_action(action, cx);
                            }
                        }
                        "escape" => window.blur(),
                        _ => {
                            let typed = event.keystroke.key_char.as_deref().unwrap_or(key);
                            if !typed.is_empty() && typed.bytes().all(|byte| byte.is_ascii_digit())
                            {
                                // Read the current draft for each ordered key event; a redraw
                                // between digits is neither necessary nor guaranteed.
                                let mut text = this.lab_workflows.replay_tick_text.clone();
                                text.extend(typed.chars().take(21));
                                this.lab_action(LabAction::ReplayTickText(text), cx);
                            }
                        }
                    }
                }
                if key != "tab" {
                    cx.stop_propagation();
                }
            }));
        panel = panel
            .child(target)
            .child(div().text_xs().child(
                "Click the target or Tab to it, type the exact tick, then press Enter to seek. Backspace edits; Delete, Ctrl+A or Clear empties the field; Escape leaves the field.",
            ));
        let mut controls = div().flex().flex_wrap().gap_2();
        for (id, label, delta) in [
            ("lab-replay-minus-ten", "−10", -10_i64),
            ("lab-replay-minus-one", "−1", -1),
            ("lab-replay-plus-one", "+1", 1),
            ("lab-replay-plus-ten", "+10", 10),
        ] {
            let enabled = state.parsed_replay_tick().is_some_and(|tick| {
                tick.checked_add_signed(delta)
                    .is_some_and(|next| next >= checkpoint.tick)
            });
            controls = controls.child(self.lab_button(
                id.into(),
                label.into(),
                LabAction::ReplayStep(delta),
                enabled,
                cx,
            ));
        }
        controls = controls.child(self.lab_button(
            "lab-replay-clear".into(),
            "Clear target".into(),
            LabAction::ReplayTickText(String::new()),
            true,
            cx,
        ));
        if let Some(action) = state.replay_seek_action() {
            controls = controls.child(self.lab_button(
                "lab-replay-seek".into(),
                "Seek exact tick".into(),
                action,
                state.can_seek_replay(),
                cx,
            ));
        }
        panel = panel.child(controls);
        if state.parsed_replay_tick().is_none() {
            panel = panel.child(
                "Invalid target: enter one unsigned 64-bit decimal tick (0 through 18446744073709551615).",
            );
        } else if state.replay_seek_action().is_none() {
            panel = panel.child(
                "Seek unavailable: the target must be at or after the saved cut. A later tick also needs current replay bounds, a target no later than the observed live tick, and the entire saved-cut-to-current interval within the displayed limit.",
            );
        }
        panel = panel.child(div().text_xs().child(
            "The saved tick verifies the saved cut alone. A later target is published only after replay matches the captured-current scientific digest. This does not locate a first divergent tick or provide a per-tick oracle.",
        ));
        if let Some(summary) = &state.replay_summary {
            let mut evidence = div()
                .id("lab-replay-summary")
                .debug_selector(|| "lab-replay-summary".to_owned())
                .flex()
                .flex_col()
                .gap_1()
                .child(div().text_sm().child("Last verified review"));
            for line in replay_summary_lines(summary) {
                evidence = evidence.child(div().text_xs().child(line));
            }
            panel = panel.child(evidence);
        }
        panel
    }

    pub(super) fn render_lab_checkpoints(&self, cx: &mut Context<Self>) -> Div {
        let state = &self.lab_workflows;
        if state.viewing_artifacts {
            return self.render_lab_artifacts(cx);
        }
        if state.viewing_replay {
            return self.render_lab_replay(cx);
        }
        if state.checkpoint_view != LabCheckpointView::SavedCuts {
            return self.render_lab_checkpoint_studies(cx);
        }
        let idle = !state.is_busy();
        let mut panel = div().flex().flex_col().gap_3().child(self.lab_panel_header("Saved checkpoints", cx))
            .child(self.render_lab_checkpoint_save(cx))
            .child(div().text_xs().child("Select a saved scientific cut to reconstruct it in World. The live owner keeps its current state and controls."))
            .child(self.lab_button("lab-world-live".into(), "Return World to live".into(), LabAction::ReturnLive, true, cx))
            .child(self.lab_button("lab-review-checkpoint".into(), "Review selected checkpoint in World".into(), LabAction::ReviewCheckpoint, idle && state.selected_checkpoint_metadata().is_some(), cx));
        panel = panel
            .child(self.lab_button(
                "lab-replay-open".into(),
                "Replay / seek from selected checkpoint".into(),
                LabAction::OpenReplay,
                state.selected_checkpoint_metadata().is_some(),
                cx,
            ))
            .child(
                self.lab_button(
                    "lab-new-checkpoint-branch".into(),
                    "Configure a checkpoint study".into(),
                    LabAction::NewCheckpointBranch,
                    state.selected_checkpoint_metadata().is_some()
                        || state
                            .latest_branch_creation
                            .as_ref()
                            .is_some_and(|creation| {
                                !matches!(&creation.state, LabBranchCreationState::Created { .. })
                            }),
                    cx,
                ),
            )
            .child(self.lab_button(
                "lab-branches-catalog".into(),
                "Retained checkpoint studies".into(),
                LabAction::CheckpointBranchesPage(None),
                state.submit.is_some(),
                cx,
            ));
        let Some(page) = &state.checkpoints else {
            return panel.child("Loading checkpoint catalog…");
        };
        panel = panel.child(div().text_xs().child(format!(
            "{} checkpoints · retained bytes {}/{}",
            page.total, page.retained_bytes, page.max_bytes
        )));
        for (index, row) in page.items.iter().enumerate() {
            let selected = state.selected_checkpoint.as_ref() == Some(&row.checkpoint_id);
            panel = panel.child(self.lab_button(
                format!("lab-checkpoint-{index}"),
                format!(
                    "{}{} · tick {} · {} agents · {} bytes",
                    if selected { "Selected: " } else { "" },
                    row.checkpoint_id,
                    row.tick,
                    row.agent_count,
                    row.byte_size
                ),
                LabAction::SelectCheckpoint(row.checkpoint_id.clone()),
                true,
                cx,
            ));
        }
        if let Some(selected) = state.selected_checkpoint_metadata() {
            panel = panel.child(
                div()
                    .text_xs()
                    .child(format!("Saved world digest {}", selected.world_digest)),
            );
        }
        if page.items.is_empty() {
            panel = panel
                .child("No saved checkpoints. Return World to live and save a checkpoint above.");
        }
        panel.child(
            div()
                .flex()
                .gap_2()
                .child(
                    self.lab_button(
                        "lab-checkpoints-prev".into(),
                        "Previous checkpoints".into(),
                        LabAction::CheckpointsPage(
                            (page.offset > LAB_PAGE_SIZE)
                                .then(|| (page.offset - LAB_PAGE_SIZE).to_string()),
                        ),
                        idle && page.offset > 0,
                        cx,
                    ),
                )
                .child(self.lab_button(
                    "lab-checkpoints-next".into(),
                    "Next checkpoints".into(),
                    LabAction::CheckpointsPage(page.next_cursor.clone()),
                    idle && page.next_cursor.is_some(),
                    cx,
                )),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpui_replay_target_input_seeks_verified_frames_and_keeps_live_intent_authoritative() {
        use scriptbots_runtime::{
            HostCore, HostCoreOptions, HostSessionId,
            lab_workflows::{
                LabReplayCapabilities, LabReplayReview, LabReplaySummary, LabReplayVerification,
            },
        };

        let world_at = |tick: u64| {
            let mut world = WorldState::new(ScriptBotsConfig {
                rng_seed: Some(17),
                world_width: 80,
                world_height: 80,
                food_cell_size: 40,
                population_minimum: 0,
                population_spawn_interval: 0,
                persistence_interval: 0,
                closed: true,
                ..ScriptBotsConfig::default()
            })
            .unwrap();
            for _ in 0..tick {
                world.step().unwrap();
            }
            world
        };
        let saved_world = world_at(0);
        let saved_digest = saved_world.world_digest_v1().unwrap().overall;
        let projection = |world| {
            HostCore::new(
                HostSessionId::new(1),
                world,
                HostCoreOptions {
                    capture_agent_visuals: true,
                    ..HostCoreOptions::default()
                },
            )
            .unwrap()
            .latest_snapshot()
        };
        let saved = projection(saved_world);
        let target = projection(world_at(12));
        let failed_target = projection(world_at(13));
        let expected_target = Arc::clone(&target);
        let host = TestHost::new(world_at(20));
        let source = host.snapshot().session_id;
        let before = host.port.scientific_digest_v1().unwrap();
        let captured_digest = before.overall.clone();
        let checkpoint = LabCheckpoint {
            checkpoint_id: "gpui-replay-source".into(),
            tick: 0,
            agent_count: saved.world.agents.len(),
            byte_size: 64,
            world_digest: saved_digest,
        };
        let selected = checkpoint.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let (started, observed) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        let released = Mutex::new(released);
        let mut session = GuiSession::new(
            host.port.clone(),
            AnalyticsSnapshotProvider::empty(),
            Arc::new(|_| panic!("historical replay must not command the live owner")),
        );
        // Explicit application-service double with actual host publications:
        // this proves native controls/projections, not application replay or GPU pixels.
        session.lab_submit = Some(Arc::new(move |request| {
            Ok(match request {
                LabWorkflowRequest::ListCheckpoints { .. } => {
                    LabWorkflowResponse::Checkpoints(LabCheckpointPage {
                        items: vec![selected.clone()],
                        total: 1,
                        offset: 0,
                        next_cursor: None,
                        retained_bytes: 64,
                        max_bytes: 1_024,
                    })
                }
                LabWorkflowRequest::ReplayCapabilities => {
                    LabWorkflowResponse::ReplayCapabilities(LabReplayCapabilities {
                        session_id: source,
                        observed_tick: 20,
                        max_verification_ticks: 32,
                    })
                }
                request @ LabWorkflowRequest::ReviewReplay { .. } => {
                    let count = {
                        let mut requests = recorded.lock().unwrap();
                        requests.push(request.clone());
                        requests.len()
                    };
                    if count == 3 {
                        return Err(
                            "Captured-current scientific digest differs; earliest divergent tick is unknown."
                                .into(),
                        );
                    }
                    if count >= 4 {
                        started.send(()).unwrap();
                        released
                            .lock()
                            .unwrap()
                            .recv_timeout(Duration::from_secs(5))
                            .unwrap();
                    }
                    let LabWorkflowRequest::ReviewReplay {
                        session_id,
                        checkpoint,
                        target_tick,
                    } = request
                    else {
                        unreachable!("matched replay request")
                    };
                    let snapshot = match target_tick {
                        0 => Arc::clone(&saved),
                        12 => Arc::clone(&target),
                        13 => Arc::clone(&failed_target),
                        other => panic!("unexpected replay target {other}"),
                    };
                    LabWorkflowResponse::ReplayFrame(LabReplayReview {
                        summary: LabReplaySummary {
                            session_id,
                            checkpoint,
                            target_tick,
                            verification: if target_tick == 0 {
                                LabReplayVerification::SavedCut
                            } else {
                                LabReplayVerification::CapturedInterval {
                                    verified_through_tick: 20,
                                    captured_current_digest: captured_digest.clone(),
                                }
                            },
                        },
                        snapshot,
                    })
                }
                other => panic!("unexpected GPUI replay request: {other:?}"),
            })
        }));
        let context = gpui::TestAppContext::single();
        let windows = context
            .update(|app| Arc::new(session).install(app))
            .unwrap();
        let world = context.update(|app| windows.canvas.root(app)).unwrap();
        let hud = context.update(|app| windows.hud.root(app)).unwrap();
        let mut lab = gpui::VisualTestContext::from_window(windows.hud.into(), &context);
        lab.simulate_resize(size(px(1_100.0), px(1_900.0)));
        let draw = |lab: &mut gpui::VisualTestContext| {
            lab.update(|window, app| {
                window.refresh();
                window.draw(app).clear();
            })
        };
        let click = |lab: &mut gpui::VisualTestContext, selector: &'static str| {
            draw(lab);
            let bounds = lab.debug_bounds(selector).expect("visible replay control");
            lab.simulate_click(bounds.center(), gpui::Modifiers::default());
        };
        let wait = |lab: &mut gpui::VisualTestContext, ready: &dyn Fn(&LabWorkflows) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                draw(lab);
                if context.update(|app| ready(&hud.read(app).lab_workflows)) {
                    break;
                }
                assert!(Instant::now() < deadline, "GPUI replay response deadline");
                std::thread::yield_now();
            }
        };
        click(&mut lab, "dashboard-tab-checkpoints");
        wait(&mut lab, &|model| {
            model.checkpoints.is_some() && !model.is_busy()
        });
        click(&mut lab, "lab-checkpoint-0");
        click(&mut lab, "lab-replay-open");
        wait(&mut lab, &|model| {
            model.replay_capabilities.is_some() && !model.is_busy()
        });
        click(&mut lab, "lab-replay-seek");
        wait(&mut lab, &|model| {
            !model.is_busy()
                && model.replay_summary.as_ref().is_some_and(|summary| {
                    matches!(summary.verification, LabReplayVerification::SavedCut)
                })
        });
        context.update(|app| {
            assert!(hud.read(app).lab_workflows.viewing_replay);
            let shown = world.read(app).playback.reconstructed.as_ref().unwrap();
            assert_eq!(shown.tick, 0);
            assert_eq!(
                shown.checkpoint_review,
                Some((checkpoint.checkpoint_id.clone(), 0))
            );
        });

        click(&mut lab, "lab-replay-target");
        let full_width_keys = u64::MAX
            .to_string()
            .chars()
            .map(|digit| digit.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        lab.simulate_keystrokes(&format!("ctrl-a {full_width_keys}"));
        context.update(|app| {
            let model = &hud.read(app).lab_workflows;
            assert_eq!(model.parsed_replay_tick(), Some(u64::MAX));
            assert!(
                !model.can_seek_replay(),
                "future targets remain unavailable"
            );
        });
        lab.simulate_keystrokes("0");
        context.update(|app| {
            let model = &hud.read(app).lab_workflows;
            assert!(model.parsed_replay_tick().is_none());
            assert!(
                !model.can_seek_replay(),
                "overflow cannot become a truncated valid tick"
            );
        });
        lab.simulate_keystrokes("ctrl-a 1 2");
        context.update(|app| {
            assert_eq!(hud.read(app).lab_workflows.replay_tick_text, "12");
            assert_eq!(
                requests.lock().unwrap().len(),
                1,
                "editing never auto-seeks"
            );
        });
        lab.simulate_keystrokes("enter");
        wait(&mut lab, &|model| {
            !model.is_busy()
                && model
                    .replay_summary
                    .as_ref()
                    .is_some_and(|summary| summary.target_tick == 12)
        });
        draw(&mut lab);
        assert!(lab.debug_bounds("lab-replay-summary").is_some());
        assert!(lab.debug_bounds("lab-replay-target").is_some());
        let palette = context.update(|app| world.read(app).accessibility_snapshot().palette);
        let expected = RenderFrame::from_snapshot(&expected_target, palette).unwrap();
        context.update(|app| {
            let model = &hud.read(app).lab_workflows;
            assert!(model.viewing_replay);
            assert!(
                replay_summary_lines(model.replay_summary.as_ref().unwrap())
                    .iter()
                    .any(|line| line.contains("captured current tick 20"))
            );
            world.update(app, |view, _| {
                let live = view.snapshot();
                let shown = view.playback.snapshot_for_render(live);
                assert_eq!(shown.tick, 12);
                assert_eq!(shown.render_frame, Some(expected.clone()));
                assert_eq!(
                    shown.checkpoint_review,
                    Some((checkpoint.checkpoint_id.clone(), 0))
                );
                assert!(view.playback.displaying_history);
            });
        });

        click(&mut lab, "lab-replay-plus-one");
        assert_eq!(requests.lock().unwrap().len(), 2);
        click(&mut lab, "lab-replay-seek");
        wait(&mut lab, &|model| {
            !model.is_busy()
                && model
                    .notice
                    .as_ref()
                    .is_some_and(|notice| notice.contains("earliest divergent tick is unknown"))
        });
        context.update(|app| {
            assert_eq!(
                world
                    .read(app)
                    .playback
                    .reconstructed
                    .as_ref()
                    .unwrap()
                    .tick,
                12
            );
            assert_eq!(
                hud.read(app)
                    .lab_workflows
                    .replay_summary
                    .as_ref()
                    .unwrap()
                    .target_tick,
                12
            );
        });

        click(&mut lab, "lab-replay-seek");
        observed.recv_timeout(Duration::from_secs(5)).unwrap();
        click(&mut lab, "lab-replay-live");
        context.update(|app| {
            assert_eq!(world.read(app).playback.mode(), PlaybackMode::Live);
            assert!(hud.read(app).lab_workflows.is_busy());
            assert!(hud.read(app).lab_workflows.viewing_replay);
        });
        release.send(()).unwrap();
        wait(&mut lab, &|model| !model.is_busy());
        context.update(|app| {
            assert_eq!(world.read(app).playback.mode(), PlaybackMode::Live);
            assert!(world.read(app).playback.reconstructed.is_none());
        });

        click(&mut lab, "lab-replay-seek");
        observed.recv_timeout(Duration::from_secs(5)).unwrap();
        click(&mut lab, "lab-replay-plus-one");
        release.send(()).unwrap();
        wait(&mut lab, &|model| !model.is_busy());
        context.update(|app| {
            assert_eq!(hud.read(app).lab_workflows.replay_tick_text, "14");
            assert_eq!(world.read(app).playback.mode(), PlaybackMode::Live);
            assert!(world.read(app).playback.reconstructed.is_none());
        });
        assert_eq!(requests.lock().unwrap().len(), 5);
        assert!(matches!(
            &requests.lock().unwrap()[1],
            LabWorkflowRequest::ReviewReplay { session_id, checkpoint: observed, target_tick: 12 }
                if *session_id == source && observed == &checkpoint
        ));
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
        assert_eq!(host.snapshot().world.tick, 20);
    }

    #[test]
    fn gpui_artifact_controls_review_frozen_metadata_and_report_export_outcomes() {
        use scriptbots_runtime::lab_workflows::{
            LabArtifact, LabArtifactExportReceipt, LabArtifactPage,
        };

        let host = TestHost::new(
            WorldState::new(ScriptBotsConfig {
                rng_seed: Some(42),
                world_width: 120,
                world_height: 120,
                food_cell_size: 20,
                population_minimum: 0,
                persistence_interval: 0,
                closed: true,
                ..ScriptBotsConfig::default()
            })
            .unwrap(),
        );
        let source = host.snapshot().session_id;
        let before = host.port.scientific_digest_v1().unwrap();
        let artifact = LabArtifact {
            artifact_id: "saved-report".into(),
            filename: "report.json".into(),
            content_type: "application/json".into(),
            byte_size: 3,
            checksum_blake3: "a".repeat(64),
            checksum_sha256: "b".repeat(64),
            created_at_utc: "2026-11-23T00:00:00Z".into(),
            relative_path: "run-42/report.json".into(),
        };
        let directory = std::env::temp_dir()
            .join("scriptbots-gpui-artifact-ui-test")
            .to_string_lossy()
            .into_owned();
        let page = LabArtifactPage {
            items: vec![artifact.clone()],
            total: 1,
            offset: 0,
            next_cursor: None,
            export_directory: directory.clone(),
        };
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let mut session = GuiSession::new(
            host.port.clone(),
            AnalyticsSnapshotProvider::empty(),
            Arc::new(|_| panic!("artifact export must not command the live owner")),
        );
        // Explicit application-service double: these are real GPUI controls and
        // the shared worker, not a filesystem, native-window or GPU proof.
        session.lab_submit = Some(Arc::new(move |request| {
            Ok(match request {
                LabWorkflowRequest::ListExperiments { .. } => {
                    LabWorkflowResponse::Experiments(LabExperimentPage {
                        items: Vec::new(),
                        total: 0,
                        offset: 0,
                        next_cursor: None,
                        reserved_run_slots: 0,
                        max_run_slots: 1,
                    })
                }
                LabWorkflowRequest::ListCheckpoints { .. } => {
                    LabWorkflowResponse::Checkpoints(LabCheckpointPage {
                        items: Vec::new(),
                        total: 0,
                        offset: 0,
                        next_cursor: None,
                        retained_bytes: 0,
                        max_bytes: 10_000,
                    })
                }
                LabWorkflowRequest::ListArtifacts { cursor } => {
                    assert!(cursor.is_none());
                    LabWorkflowResponse::Artifacts(page.clone())
                }
                request @ LabWorkflowRequest::ExportArtifact { .. } => {
                    let count = {
                        let mut requests = recorded.lock().unwrap();
                        requests.push(request.clone());
                        requests.len()
                    };
                    if count == 2 {
                        return Err("destination readback verification failed".into());
                    }
                    let LabWorkflowRequest::ExportArtifact {
                        request_id,
                        session_id,
                        artifact,
                        export_directory,
                    } = request
                    else {
                        unreachable!("matched export request")
                    };
                    assert_eq!(session_id, source);
                    let capsule = std::path::Path::new(&export_directory).join(&request_id);
                    LabWorkflowResponse::ArtifactExported(LabArtifactExportReceipt {
                        output_path: capsule
                            .join(&artifact.filename)
                            .to_string_lossy()
                            .into_owned(),
                        manifest_path: capsule
                            .join("export-manifest.json")
                            .to_string_lossy()
                            .into_owned(),
                        request_id,
                        session_id,
                        artifact,
                        export_directory,
                    })
                }
                other => panic!("unexpected artifact UI request: {other:?}"),
            })
        }));
        let context = gpui::TestAppContext::single();
        let windows = context
            .update(|app| Arc::new(session).install(app))
            .unwrap();
        let hud = context.update(|app| windows.hud.root(app)).unwrap();
        let mut lab = gpui::VisualTestContext::from_window(windows.hud.into(), &context);
        lab.simulate_resize(size(px(1_100.0), px(1_800.0)));
        let draw = |lab: &mut gpui::VisualTestContext| {
            lab.update(|window, app| {
                window.refresh();
                window.draw(app).clear();
            })
        };
        let click = |lab: &mut gpui::VisualTestContext, selector: &'static str| {
            draw(lab);
            let bounds = lab
                .debug_bounds(selector)
                .expect("visible artifact control");
            lab.simulate_click(bounds.center(), gpui::Modifiers::default());
        };
        let wait = |lab: &mut gpui::VisualTestContext, ready: &dyn Fn(&LabWorkflows) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                draw(lab);
                if context.update(|app| ready(&hud.read(app).lab_workflows)) {
                    break;
                }
                assert!(Instant::now() < deadline, "artifact UI response deadline");
                std::thread::yield_now();
            }
        };
        click(&mut lab, "dashboard-tab-experiments");
        wait(&mut lab, &|model| {
            model.experiments.is_some() && !model.is_busy()
        });
        click(&mut lab, "lab-artifacts-open");
        wait(&mut lab, &|model| {
            model.artifacts.is_some() && !model.is_busy()
        });
        draw(&mut lab);
        assert!(lab.debug_bounds("lab-artifact-export").is_none());
        click(&mut lab, "lab-artifact-0");
        draw(&mut lab);
        assert!(lab.debug_bounds("lab-artifact-metadata").is_some());
        assert!(lab.debug_bounds("lab-artifact-export").is_some());
        let stale_action = context.update(|app| {
            hud.read(app)
                .lab_workflows
                .export_artifact_action()
                .unwrap()
        });
        assert_eq!(
            stale_action,
            LabAction::ExportArtifact {
                session_id: source,
                artifact: artifact.clone(),
                export_directory: directory.clone(),
            },
        );
        // A control captured before the displayed descriptor changes must not
        // write either the old artifact or its replacement.
        context.update(|app| {
            hud.update(app, |view, cx| {
                view.lab_workflows.artifacts.as_mut().unwrap().items[0].checksum_sha256 =
                    "c".repeat(64);
                view.lab_action(stale_action.clone(), cx);
                assert!(!view.lab_workflows.is_busy());
                assert!(view.lab_workflows.latest_artifact_export.is_none());
            })
        });
        assert!(requests.lock().unwrap().is_empty());
        click(&mut lab, "lab-refresh");
        wait(&mut lab, &|model| {
            !model.is_busy()
                && model
                    .artifacts
                    .as_ref()
                    .is_some_and(|page| page.items[0].checksum_sha256 == artifact.checksum_sha256)
        });
        click(&mut lab, "lab-artifact-0");
        click(&mut lab, "lab-artifact-export");
        wait(&mut lab, &|model| {
            !model.is_busy()
                && model.latest_artifact_export.as_ref().is_some_and(|export| {
                    matches!(&export.state, LabArtifactExportState::Exported(_))
                })
        });
        draw(&mut lab);
        assert!(lab.debug_bounds("lab-artifact-export-status").is_some());
        context.update(|app| {
            let model = &hud.read(app).lab_workflows;
            let LabArtifactExportState::Exported(receipt) =
                &model.latest_artifact_export.as_ref().unwrap().state
            else {
                panic!("matching service receipt is required");
            };
            assert_eq!(receipt.artifact, artifact);
            assert_eq!(receipt.export_directory, directory);
            let lines = artifact_export_lines(model.latest_artifact_export.as_ref().unwrap());
            assert!(lines.iter().any(|line| line.contains(&receipt.output_path)));
            assert!(
                lines
                    .iter()
                    .any(|line| line.contains(&receipt.manifest_path))
            );
        });
        click(&mut lab, "lab-artifacts-back");
        draw(&mut lab);
        assert!(lab.debug_bounds("lab-artifacts-open").is_some());
        click(&mut lab, "dashboard-tab-checkpoints");
        wait(&mut lab, &|model| {
            model.checkpoints.is_some() && !model.is_busy()
        });
        click(&mut lab, "lab-artifacts-open");
        wait(&mut lab, &|model| {
            model.artifacts.is_some() && !model.is_busy()
        });
        click(&mut lab, "lab-artifact-0");
        click(&mut lab, "lab-artifact-export");
        wait(&mut lab, &|model| {
            !model.is_busy()
                && model.latest_artifact_export.as_ref().is_some_and(|export| {
                    matches!(&export.state, LabArtifactExportState::Failed { .. })
                })
        });
        draw(&mut lab);
        assert!(lab.debug_bounds("lab-artifact-export-status").is_some());
        context.update(|app| {
            let model = &hud.read(app).lab_workflows;
            let export = model.latest_artifact_export.as_ref().unwrap();
            assert!(
                artifact_export_lines(export)
                    .iter()
                    .any(|line| { line.contains("destination readback verification failed") })
            );
        });
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let first_id = match &requests[0] {
            LabWorkflowRequest::ExportArtifact { request_id, .. } => request_id,
            other => panic!("unexpected recorded request: {other:?}"),
        };
        let second_id = match &requests[1] {
            LabWorkflowRequest::ExportArtifact { request_id, .. } => request_id,
            other => panic!("unexpected recorded request: {other:?}"),
        };
        assert_ne!(
            first_id, second_id,
            "new explicit exports use distinct destinations"
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
    }

    #[test]
    fn gpui_guide_observes_shared_owner_receipts_and_refuses_stale_actions() {
        use scriptbots_runtime::{HostCommand, lab_guide::LabGuideStep};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut world = WorldState::new(ScriptBotsConfig {
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
        let selected = world
            .try_spawn_agent(scriptbots_core::AgentData::default())
            .unwrap();
        let host = Arc::new(TestHost::new(world));
        let command_count = Arc::new(AtomicUsize::new(0));
        let submitted = Arc::clone(&command_count);
        let config_count = Arc::clone(&command_count);
        let config_host = Arc::clone(&host);
        let command_host = Arc::clone(&host);
        let mut session = GuiSession::new(
            host.port.clone(),
            AnalyticsSnapshotProvider::empty(),
            Arc::new(move |command| {
                submitted.fetch_add(1, Ordering::SeqCst);
                Some(
                    command_host
                        .apply(HostCommand::try_from(command).unwrap())
                        .command_id()
                        .to_string(),
                )
            }),
        );
        session.config_update_submit = Some(Arc::new(move |config, expected| {
            config_count.fetch_add(1, Ordering::SeqCst);
            let id = scriptbots_runtime::CommandId::from_client_sequence(
                1,
                config_host
                    .next_command
                    .fetch_add(1, AtomicOrdering::Relaxed),
            );
            config_host
                .port
                .clone()
                .submit(
                    scriptbots_runtime::CommandEnvelope::new(id, HostCommand::UpdateConfig(config))
                        .expecting_control_revision(expected.control)
                        .expecting_scientific_revision(expected.scientific)
                        .expecting_config_revision(expected.config),
                )
                .map_err(|error| error.to_string())
        }));
        let context = gpui::TestAppContext::single();
        let windows = context
            .update(|app| Arc::new(session).install(app))
            .unwrap();
        let hud = context.update(|app| windows.hud.root(app)).unwrap();
        let canvas = context.update(|app| windows.canvas.root(app)).unwrap();
        let mut lab = gpui::VisualTestContext::from_window(windows.hud.into(), &context);
        lab.simulate_resize(size(px(1_000.0), px(1_800.0)));
        let draw = |lab: &mut gpui::VisualTestContext| {
            lab.update(|window, app| {
                window.refresh();
                window.draw(app).clear();
            })
        };
        let click = |lab: &mut gpui::VisualTestContext, selector: &'static str| {
            draw(lab);
            let bounds = lab.debug_bounds(selector).expect("visible guide control");
            lab.simulate_click(bounds.center(), gpui::Modifiers::default());
        };
        let wait_ready = |lab: &mut gpui::VisualTestContext| {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                draw(lab);
                if context.update(|app| hud.read(app).lab_guide.lock().unwrap().ready()) {
                    break;
                }
                assert!(Instant::now() < deadline, "guide owner receipt deadline");
                std::thread::yield_now();
            }
        };
        let before = host.port.scientific_digest_v1().unwrap();
        click(&mut lab, "lab-guide-open");
        wait_ready(&mut lab);
        context.update(|app| {
            assert!(Arc::ptr_eq(
                &hud.read(app).lab_guide,
                &canvas.read(app).lab_guide,
            ));
        });
        click(&mut lab, "lab-guide-next");
        draw(&mut lab);
        context.update(|app| {
            let guide = hud.read(app).lab_guide.lock().unwrap();
            assert_eq!(guide.step(), LabGuideStep::Inspect);
            assert!(!guide.ready(), "an unselected living agent is insufficient");
        });
        click(&mut lab, "lab-guide-next");
        click(&mut lab, "lab-guide-hide");
        click(&mut lab, "lab-guide-open");
        context.update(|app| {
            let guide = hud.read(app).lab_guide.lock().unwrap();
            assert_eq!(guide.step(), LabGuideStep::Inspect);
            assert!(guide.visible);
        });
        assert_eq!(command_count.load(Ordering::SeqCst), 0);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);

        // Focus is frontend presentation state; commands below still pass
        // through the actual shared owner and its published journal receipts.
        context.update(|app| {
            canvas.update(app, |view, _| {
                view.inspector.lock().unwrap().focused_agent = Some(selected);
            });
        });
        wait_ready(&mut lab);
        click(&mut lab, "lab-guide-next");
        click(&mut lab, "lab-guide-perform");
        wait_ready(&mut lab);
        assert!(host.snapshot().playback.paused);
        assert_eq!(command_count.load(Ordering::SeqCst), 1);
        click(&mut lab, "lab-guide-next");
        let step_source = host.snapshot();
        let step_tick = step_source.world.tick;
        let step_action = LabGuideBoundary {
            generation: context.update(|app| hud.read(app).lab_guide.lock().unwrap().generation()),
            step: LabGuideStep::Step,
            session_id: step_source.session_id,
            config_revision: step_source.revisions.config,
            growth_proposal: None,
        };
        click(&mut lab, "lab-guide-perform");
        wait_ready(&mut lab);
        context.update(|app| {
            hud.update(app, |view, cx| {
                view.guide_action(LabGuideAction::Perform(step_action), cx);
            });
        });
        assert_eq!(host.snapshot().world.tick, step_tick + 1);
        assert!(host.snapshot().playback.paused);
        assert_eq!(command_count.load(Ordering::SeqCst), 2);
        click(&mut lab, "lab-guide-next");
        draw(&mut lab);

        let offered = host.snapshot();
        let stale = LabGuideBoundary {
            generation: context.update(|app| hud.read(app).lab_guide.lock().unwrap().generation()),
            step: LabGuideStep::Configure,
            session_id: offered.session_id,
            config_revision: offered.revisions.config,
            growth_proposal: guide_growth_proposal(&offered),
        };
        let mut updated = offered.config.as_ref().clone();
        updated.food_growth_rate *= 0.5;
        host.apply(HostCommand::UpdateConfig(Box::new(updated)));
        context.update(|app| {
            hud.update(app, |view, cx| {
                view.guide_action(LabGuideAction::Perform(stale), cx);
            });
        });
        assert_eq!(command_count.load(Ordering::SeqCst), 2);
        context.update(|app| {
            assert!(
                hud.read(app)
                    .lab_workflows
                    .notice
                    .as_deref()
                    .unwrap()
                    .contains("Configuration changed")
            );
        });
        let current = host.snapshot();
        let expected = guide_growth_proposal(&current).unwrap();
        click(&mut lab, "lab-guide-perform");
        wait_ready(&mut lab);
        assert_eq!(
            host.snapshot().config.food_growth_rate.to_bits(),
            expected.to_bits(),
        );
        assert_eq!(host.snapshot().world.tick, step_tick + 1);
        assert_eq!(command_count.load(Ordering::SeqCst), 3);

        // A callback from the previous guide step cannot perform another
        // configuration change or advance the newly entered checkpoint step.
        click(&mut lab, "lab-guide-next");
        context.update(|app| {
            hud.update(app, |view, cx| {
                view.guide_action(LabGuideAction::Perform(stale), cx);
                view.guide_action(LabGuideAction::Next(stale), cx);
            });
            let guide = hud.read(app).lab_guide.lock().unwrap();
            assert_eq!(guide.step(), LabGuideStep::Checkpoint);
            assert!(!guide.ready());
        });
        assert_eq!(command_count.load(Ordering::SeqCst), 3);
        click(&mut lab, "lab-guide-perform");
        context.update(|app| {
            let guide = hud.read(app).lab_guide.lock().unwrap();
            assert!(
                !guide.ready(),
                "missing checkpoint services cannot report a save"
            );
        });

        // A cached request from before Checkpoint cannot become fresh merely
        // because the guide paints its Pending and then its Saved DTO.
        let frame = host.snapshot();
        let mut cached = scriptbots_runtime::lab_workflows::LabCheckpointSave {
            request_id: "earlier-save".into(),
            session_id: frame.session_id,
            state: LabCheckpointSaveState::Pending,
        };
        context.update(|app| {
            hud.update(app, |view, _| {
                view.lab_workflows.latest_checkpoint_save = Some(cached.clone());
                view.observe_lab_guide();
            });
        });
        cached.state = LabCheckpointSaveState::Saved(LabCheckpoint {
            checkpoint_id: "earlier-cut".into(),
            tick: frame.world.tick,
            agent_count: frame.world.agents.len(),
            byte_size: 4096,
            world_digest: host.port.scientific_digest_v1().unwrap().overall,
        });
        context.update(|app| {
            hud.update(app, |view, _| {
                view.lab_workflows.latest_checkpoint_save = Some(cached);
                view.observe_lab_guide();
            });
            assert!(!hud.read(app).lab_guide.lock().unwrap().ready());
        });

        // Re-entering the same step after restart must not revive an old
        // callback, even though the owner and configuration still match.
        let frame = host.snapshot();
        let old_identify = LabGuideBoundary {
            generation: stale.generation,
            step: LabGuideStep::Identify,
            session_id: frame.session_id,
            config_revision: frame.revisions.config,
            growth_proposal: None,
        };
        click(&mut lab, "lab-guide-restart");
        wait_ready(&mut lab);
        context.update(|app| {
            hud.update(app, |view, cx| {
                view.guide_action(LabGuideAction::Next(old_identify), cx);
            });
            let guide = hud.read(app).lab_guide.lock().unwrap();
            assert_eq!(guide.step(), LabGuideStep::Identify);
            assert!(guide.ready());
            assert!(guide.generation() > old_identify.generation);
        });
        assert_eq!(command_count.load(Ordering::SeqCst), 3);
    }

    fn experiment_capabilities() -> LabExperimentCapabilities {
        LabExperimentCapabilities {
            families: vec!["mlp".into(), "dwraon".into(), "assembly".into()],
            treatment_parameters: LabExperimentParameter::ALL
                .into_iter()
                .map(|parameter| LabExperimentParameterCapability {
                    parameter,
                    baseline_value: 0.25,
                    min_value: 0.0,
                    max_value: 1.0,
                })
                .collect(),
            max_seed_count: 8,
            max_ticks_per_run: 1_000,
            max_total_ticks: 16_000,
            launch_description: "Fresh worlds with the fixture's declared founder recipe.".into(),
        }
    }

    fn completed_branch_fixture(draft: &LabBranchDraft) -> (LabCheckpointBranch, LabBranchReport) {
        let checkpoint = &draft.checkpoint;
        let target = checkpoint.tick.checked_add(draft.ticks).unwrap();
        let branch = LabCheckpointBranch {
            branch_id: "native-study-1".into(),
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            branch_name: draft.treatment.key().into(),
            status: "completed".into(),
            source_tick: checkpoint.tick,
            target_tick: target,
            world_digest: checkpoint.world_digest.clone(),
            artifact_id: Some("native-study-report-1".into()),
            error_reason: None,
        };
        let control = LabBranchOutcome {
            tick: target,
            population: checkpoint.agent_count,
            total_food: 4.0,
            total_energy: if checkpoint.agent_count == 0 {
                0.0
            } else {
                2.0
            },
            average_health: (checkpoint.agent_count > 0).then_some(1.0),
            final_digest: "1111111111111111".into(),
            command_applied_tick: None,
            minimum_population: checkpoint.agent_count,
            extinction_tick: (checkpoint.agent_count == 0).then_some(checkpoint.tick),
            baseline_population: checkpoint.agent_count,
            recovery_ratio: (checkpoint.agent_count > 0).then_some(1.0),
            first_divergence: None,
        };
        let report = LabBranchReport {
            branch_id: branch.branch_id.clone(),
            artifact_id: branch.artifact_id.clone().unwrap(),
            base_seed: 42,
            source_tick: checkpoint.tick,
            target_tick: target,
            source_world_digest: checkpoint.world_digest.clone(),
            source_byte_size: checkpoint.byte_size,
            source_checksum_blake3: "a".repeat(64),
            report_byte_size: 2_048,
            report_checksum_blake3: "b".repeat(64),
            report_checksum_sha256: "c".repeat(64),
            treatment: (draft.treatment != LabBranchTreatment::Unchanged).then(|| {
                LabBranchOutcome {
                    total_food: 6.0,
                    final_digest: "2222222222222222".into(),
                    command_applied_tick: Some(checkpoint.tick),
                    first_divergence: Some(LabBranchDivergence {
                        tick: checkpoint.tick + 1,
                        stage: "food".into(),
                        control_digest: "3333333333333333".into(),
                        branch_digest: "4444444444444444".into(),
                        command_applied_tick: checkpoint.tick,
                        details: "Fixture's first differing food lane.".into(),
                    }),
                    ..control.clone()
                }
            }),
            control,
        };
        (branch, report)
    }

    #[test]
    fn gpui_checkpoint_study_buttons_preserve_saved_source_retry_and_report() {
        let host = TestHost::new(
            WorldState::new(ScriptBotsConfig {
                rng_seed: Some(42),
                world_width: 120,
                world_height: 120,
                food_cell_size: 20,
                closed: true,
                population_minimum: 0,
                persistence_interval: 0,
                ..ScriptBotsConfig::default()
            })
            .unwrap(),
        );
        let saved = host.snapshot();
        let checkpoint = LabCheckpoint {
            checkpoint_id: "saved-study-cut".into(),
            tick: saved.world.tick,
            agent_count: saved.world.agents.len(),
            byte_size: 1_024,
            world_digest: host.port.scientific_digest_v1().unwrap().overall,
        };
        host.apply(scriptbots_runtime::HostCommand::UpdateSimulation(
            SimulationCommand {
                paused: Some(true),
                speed_multiplier: None,
                step_once: true,
            },
        ));
        let before = host.port.scientific_digest_v1().unwrap();
        let live_tick = host.snapshot().world.tick;
        assert!(live_tick > checkpoint.tick);
        let expected_checkpoint = checkpoint.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let retained_study = Mutex::new(None);
        let mut session = GuiSession::new(
            host.port.clone(),
            AnalyticsSnapshotProvider::empty(),
            Arc::new(|_| panic!("a saved-source study must not command the live owner")),
        );
        session.lab_submit = Some(Arc::new(move |request| {
            Ok(match request {
                LabWorkflowRequest::ListCheckpoints { .. } => {
                    LabWorkflowResponse::Checkpoints(LabCheckpointPage {
                        items: vec![checkpoint.clone()],
                        total: 1,
                        offset: 0,
                        next_cursor: None,
                        retained_bytes: checkpoint.byte_size,
                        max_bytes: 1_000_000,
                    })
                }
                LabWorkflowRequest::BranchCapabilities => {
                    LabWorkflowResponse::BranchCapabilities(LabBranchCapabilities {
                        max_ticks: 10_000,
                        max_population: 10_000,
                    })
                }
                LabWorkflowRequest::CreateCheckpointBranch {
                    request_id,
                    session_id,
                    draft,
                } => {
                    assert_eq!(session_id, saved.session_id);
                    assert_eq!(draft.checkpoint, checkpoint);
                    let first = {
                        let mut recorded = recorded.lock().unwrap();
                        recorded.push((request_id.clone(), session_id, draft.clone()));
                        recorded.len() == 1
                    };
                    let mut retained = retained_study.lock().unwrap();
                    let (branch, _) =
                        retained.get_or_insert_with(|| completed_branch_fixture(&draft));
                    if first {
                        return Err("study admitted but its first response was lost".into());
                    }
                    LabWorkflowResponse::CheckpointBranchCreated {
                        request_id,
                        session_id,
                        branch: branch.clone(),
                    }
                }
                LabWorkflowRequest::ListCheckpointBranches { .. } => {
                    let retained = retained_study.lock().unwrap();
                    let (branch, _) = retained.as_ref().unwrap();
                    LabWorkflowResponse::CheckpointBranches(LabCheckpointBranchPage {
                        items: vec![branch.clone()],
                        total: 1,
                        offset: 0,
                        next_cursor: None,
                    })
                }
                LabWorkflowRequest::GetCheckpointBranch { branch_id } => {
                    let retained = retained_study.lock().unwrap();
                    let (branch, _) = retained.as_ref().unwrap();
                    assert_eq!(branch_id, branch.branch_id);
                    LabWorkflowResponse::CheckpointBranch(branch.clone())
                }
                LabWorkflowRequest::GetCheckpointBranchReport { branch_id } => {
                    let retained = retained_study.lock().unwrap();
                    let (_, report) = retained.as_ref().unwrap();
                    assert_eq!(branch_id, report.branch_id);
                    LabWorkflowResponse::CheckpointBranchReport(Box::new(report.clone()))
                }
                other => panic!("unexpected checkpoint-study request: {other:?}"),
            })
        }));
        let context = gpui::TestAppContext::single();
        let windows = context
            .update(|app| Arc::new(session).install(app))
            .unwrap();
        let hud = context.update(|app| windows.hud.root(app)).unwrap();
        let mut lab = gpui::VisualTestContext::from_window(windows.hud.into(), &context);
        lab.simulate_resize(size(px(1_000.0), px(1_800.0)));
        let draw = |lab: &mut gpui::VisualTestContext| {
            lab.update(|window, app| {
                window.refresh();
                window.draw(app).clear();
            })
        };
        let click = |lab: &mut gpui::VisualTestContext, selector: &'static str| {
            draw(lab);
            let bounds = lab
                .debug_bounds(selector)
                .expect("visible production study control");
            lab.simulate_click(bounds.center(), gpui::Modifiers::default());
        };
        let wait = |lab: &mut gpui::VisualTestContext, ready: &dyn Fn(&LabWorkflows) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                draw(lab);
                if context.update(|app| ready(&hud.read(app).lab_workflows)) {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "native checkpoint-study result did not arrive"
                );
                std::thread::yield_now();
            }
        };
        click(&mut lab, "dashboard-tab-checkpoints");
        wait(&mut lab, &|state| {
            state.checkpoints.is_some() && !state.is_busy()
        });
        click(&mut lab, "lab-checkpoint-0");
        click(&mut lab, "lab-new-checkpoint-branch");
        wait(&mut lab, &|state| {
            state.checkpoint_view == LabCheckpointView::BranchComposer
                && state.branch_capabilities.is_some()
                && !state.is_busy()
        });
        click(&mut lab, "lab-branch-treatment-food-bloom");
        click(&mut lab, "lab-branch-ticks-10");
        click(&mut lab, "lab-branch-ticks-next");
        draw(&mut lab);
        assert!(lab.debug_bounds("lab-branch-preview").is_some());
        context.update(|app| {
            let draft = hud.read(app).lab_workflows.branch_draft.as_ref().unwrap();
            assert_eq!(draft.checkpoint, expected_checkpoint);
            assert_eq!(draft.ticks, 11);
            assert_eq!(draft.treatment, LabBranchTreatment::FoodBloom);
        });
        click(&mut lab, "lab-branch-create");
        wait(&mut lab, &|state| {
            !state.is_busy()
                && state
                    .latest_branch_creation
                    .as_ref()
                    .is_some_and(|creation| {
                        matches!(&creation.state, LabBranchCreationState::Retryable { .. })
                    })
        });
        click(&mut lab, "lab-branch-treatment-drought");
        click(&mut lab, "lab-branch-ticks-1000");
        context.update(|app| {
            let state = &hud.read(app).lab_workflows;
            let draft = state.branch_draft.as_ref().unwrap();
            assert_eq!(draft.ticks, 11);
            assert_eq!(draft.treatment, LabBranchTreatment::FoodBloom);
        });
        click(&mut lab, "lab-branch-create");
        wait(&mut lab, &|state| {
            !state.is_busy()
                && state.checkpoint_view == LabCheckpointView::BranchDetail
                && state
                    .branch
                    .as_ref()
                    .is_some_and(|branch| branch.status == "completed")
        });
        click(&mut lab, "lab-branch-read-report");
        wait(&mut lab, &|state| {
            state.branch_report.is_some() && !state.is_busy()
        });
        draw(&mut lab);
        assert!(lab.debug_bounds("lab-branch-report").is_some());
        context.update(|app| {
            let report = hud.read(app).lab_workflows.branch_report.as_ref().unwrap();
            assert_eq!(report.source_world_digest, expected_checkpoint.world_digest);
            assert_eq!(report.source_tick, expected_checkpoint.tick);
            assert_eq!(report.target_tick, expected_checkpoint.tick + 11);
            let treatment = report.treatment.as_ref().unwrap();
            assert_eq!(treatment.total_food - report.control.total_food, 2.0);
            assert_eq!(treatment.first_divergence.as_ref().unwrap().stage, "food");
        });
        click(&mut lab, "lab-branches-catalog");
        wait(&mut lab, &|state| {
            state.checkpoint_branches.is_some() && !state.is_busy()
        });
        click(&mut lab, "lab-branch-0");
        wait(&mut lab, &|state| {
            state.branch.is_some() && !state.is_busy()
        });
        click(&mut lab, "lab-branch-read-report");
        wait(&mut lab, &|state| {
            state.branch_report.is_some() && !state.is_busy()
        });
        let recorded = requests.lock().unwrap();
        assert_eq!(
            recorded.len(),
            2,
            "catalog reopen must not admit another study"
        );
        assert_eq!(
            recorded[0], recorded[1],
            "retry must preserve every request input"
        );
        assert_eq!(recorded[0].2.checkpoint, expected_checkpoint);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
        assert_eq!(host.snapshot().world.tick, live_tick);
    }

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

    fn created_batch(experiment_id: &str, draft: &LabExperimentDraft) -> LabExperiment {
        LabExperiment {
            summary: LabExperimentSummary {
                experiment_id: experiment_id.into(),
                status: "pending".into(),
                total_runs: draft.seed_count * 2,
                completed_runs: 0,
                failed_runs: 0,
            },
            plan_digest: "created-plan".into(),
            generation: 0,
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
        }
    }

    fn comparison(challenger: &str, metric: LabMetric) -> LabComparison {
        LabComparison {
            experiment_id: "actual-batch".into(),
            baseline_variant: "arm-00".into(),
            challenger_variant: challenger.into(),
            metric,
            endpoint_tick: 2,
            matched_seeds: vec![7],
            mean_difference: if metric == LabMetric::AverageEnergy {
                0.25
            } else {
                3.0
            },
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

    #[test]
    fn gpui_failed_run_retry_submits_the_displayed_generation_and_keeps_completed_peers() {
        let host = TestHost::new(
            WorldState::new(ScriptBotsConfig {
                rng_seed: Some(42),
                world_width: 120,
                world_height: 120,
                food_cell_size: 20,
                population_minimum: 0,
                persistence_interval: 0,
                closed: true,
                ..ScriptBotsConfig::default()
            })
            .unwrap(),
        );
        let source = host.snapshot().session_id;
        let before = host.port.scientific_digest_v1().unwrap();
        let mut failed = batch(2);
        failed.summary.status = "failed".into();
        failed.summary.completed_runs = 1;
        failed.summary.failed_runs = 1;
        failed.variants[1].completed_runs = 0;
        failed.variants[1].failed_runs = 1;
        let completed_peer = failed.variants[0].clone();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&calls);
        let retained = Mutex::new(failed);
        let mut session = GuiSession::new(
            host.port.clone(),
            AnalyticsSnapshotProvider::empty(),
            Arc::new(|_| panic!("failed-run retry cannot command the live owner")),
        );
        session.lab_submit = Some(Arc::new(move |request| {
            let mut observed = retained.lock().unwrap();
            Ok(match request {
                LabWorkflowRequest::ListExperiments { .. } => {
                    LabWorkflowResponse::Experiments(LabExperimentPage {
                        items: vec![observed.summary.clone()],
                        total: 1,
                        offset: 0,
                        next_cursor: None,
                        reserved_run_slots: 0,
                        max_run_slots: 1,
                    })
                }
                LabWorkflowRequest::GetExperiment { .. } => {
                    LabWorkflowResponse::Experiment(observed.clone())
                }
                request @ LabWorkflowRequest::RetryFailedExperiment {
                    session_id,
                    expected_generation,
                    ..
                } => {
                    assert_eq!(session_id, source);
                    assert_eq!(expected_generation, observed.generation);
                    recorded.lock().unwrap().push(request);
                    observed.generation += 1;
                    observed.summary.status = "running".into();
                    observed.summary.failed_runs = 0;
                    observed.variants[1].failed_runs = 0;
                    observed.variants[1].running_runs = 1;
                    LabWorkflowResponse::Experiment(observed.clone())
                }
                other => panic!("unexpected retry fixture request: {other:?}"),
            })
        }));
        let context = gpui::TestAppContext::single();
        let windows = context
            .update(|app| Arc::new(session).install(app))
            .unwrap();
        let hud = context.update(|app| windows.hud.root(app)).unwrap();
        let mut lab = gpui::VisualTestContext::from_window(windows.hud.into(), &context);
        lab.simulate_resize(size(px(1_100.0), px(1_800.0)));
        let draw = |lab: &mut gpui::VisualTestContext| {
            lab.update(|window, app| {
                window.refresh();
                window.draw(app).clear();
            })
        };
        let click = |lab: &mut gpui::VisualTestContext, selector: &'static str| {
            draw(lab);
            let bounds = lab
                .debug_bounds(selector)
                .expect("visible retry workflow control");
            lab.simulate_click(bounds.center(), gpui::Modifiers::default());
        };
        let wait = |lab: &mut gpui::VisualTestContext, ready: &dyn Fn(&LabWorkflows) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                draw(lab);
                if context.update(|app| ready(&hud.read(app).lab_workflows)) {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "failed-run retry response deadline"
                );
                std::thread::yield_now();
            }
        };
        click(&mut lab, "dashboard-tab-experiments");
        wait(&mut lab, &|state| {
            state.experiments.is_some() && !state.is_busy()
        });
        click(&mut lab, "lab-experiment-0");
        wait(&mut lab, &|state| {
            state.failed_experiment_retry_action().is_some()
        });
        click(&mut lab, "lab-retry-failed-experiment");
        wait(&mut lab, &|state| {
            state
                .batch
                .as_ref()
                .is_some_and(|batch| batch.generation == 10)
                && !state.is_busy()
        });
        assert_eq!(calls.lock().unwrap().len(), 1);
        context.update(|app| {
            let state = &hud.read(app).lab_workflows;
            assert_eq!(state.batch.as_ref().unwrap().variants[0], completed_peer);
            assert!(state.failed_experiment_retry_action().is_none());
        });
        draw(&mut lab);
        assert!(lab.debug_bounds("lab-retry-failed-experiment").is_none());
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
    }

    #[test]
    fn gpui_ecological_composer_preserves_question_through_retry_and_comparison() {
        let host = TestHost::new(
            WorldState::new(ScriptBotsConfig {
                rng_seed: Some(42),
                world_width: 120,
                world_height: 120,
                food_cell_size: 20,
                closed: true,
                population_minimum: 0,
                persistence_interval: 0,
                ..ScriptBotsConfig::default()
            })
            .unwrap(),
        );
        let source = host.snapshot().session_id;
        let before = host.port.scientific_digest_v1().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let retained = Mutex::new(None::<(LabExperiment, LabExperimentDraft)>);
        let mut session = GuiSession::new(
            host.port.clone(),
            AnalyticsSnapshotProvider::empty(),
            Arc::new(|_| panic!("independent treatment cannot command the live owner")),
        );
        session.lab_submit = Some(Arc::new(move |request| {
            Ok(match request {
                LabWorkflowRequest::ListExperiments { .. } => {
                    LabWorkflowResponse::Experiments(LabExperimentPage {
                        items: vec![batch(2).summary],
                        total: 1,
                        offset: 0,
                        next_cursor: None,
                        reserved_run_slots: 0,
                        max_run_slots: 1,
                    })
                }
                LabWorkflowRequest::ExperimentCapabilities => {
                    LabWorkflowResponse::ExperimentCapabilities(experiment_capabilities())
                }
                LabWorkflowRequest::CreateExperiment {
                    request_id,
                    session_id,
                    draft,
                } => {
                    assert_eq!(session_id, source);
                    draft.validate(&experiment_capabilities()).unwrap();
                    let first = {
                        let mut recorded = recorded.lock().unwrap();
                        recorded.push((request_id.clone(), session_id, draft.clone()));
                        recorded.len() == 1
                    };
                    let mut retained = retained.lock().unwrap();
                    let (experiment, _) = retained.get_or_insert_with(|| {
                        let mut experiment = created_batch(&request_id, &draft);
                        experiment.summary.status = "completed".into();
                        experiment.summary.completed_runs = experiment.summary.total_runs;
                        experiment.generation = 1;
                        for variant in &mut experiment.variants {
                            variant.completed_runs = variant.total_runs;
                        }
                        (experiment, draft)
                    });
                    if first {
                        return Err("ecological experiment admitted; response was lost".into());
                    }
                    LabWorkflowResponse::ExperimentCreated {
                        request_id,
                        session_id,
                        experiment: experiment.clone(),
                    }
                }
                LabWorkflowRequest::GetExperiment { experiment_id } => {
                    LabWorkflowResponse::Experiment(if experiment_id == "actual-batch" {
                        batch(2)
                    } else {
                        retained.lock().unwrap().as_ref().unwrap().0.clone()
                    })
                }
                LabWorkflowRequest::Compare {
                    experiment_id,
                    baseline_variant,
                    challenger_variant,
                    metric,
                } => {
                    let retained = retained.lock().unwrap();
                    let (experiment, draft) = retained.as_ref().unwrap();
                    assert_eq!(experiment_id, experiment.summary.experiment_id);
                    assert_eq!(baseline_variant, "baseline");
                    assert_eq!(challenger_variant, "challenger");
                    assert_eq!(metric, LabMetric::TotalFood);
                    LabWorkflowResponse::Comparison(LabComparison {
                        experiment_id,
                        baseline_variant,
                        challenger_variant,
                        metric,
                        endpoint_tick: draft.ticks_per_run,
                        matched_seeds: draft.seeds().unwrap(),
                        mean_difference: 0.5,
                        ci_95: Some((0.5, 0.5)),
                        standardized_effect: None,
                        p_value: None,
                        adjusted_p_value: None,
                        descriptive_only: false,
                        underpowered_reason: Some("Small fixture cohort.".into()),
                        undefined_effect_reason: Some("Constant paired difference.".into()),
                        plan_digest: experiment.plan_digest.clone(),
                        status_generation: experiment.generation,
                    })
                }
                other => panic!("unexpected ecological request: {other:?}"),
            })
        }));
        let context = gpui::TestAppContext::single();
        let windows = context
            .update(|app| Arc::new(session).install(app))
            .unwrap();
        let hud = context.update(|app| windows.hud.root(app)).unwrap();
        let mut lab = gpui::VisualTestContext::from_window(windows.hud.into(), &context);
        lab.simulate_resize(size(px(1_100.0), px(2_400.0)));
        let draw = |lab: &mut gpui::VisualTestContext| {
            lab.update(|window, app| {
                window.refresh();
                window.draw(app).clear();
            })
        };
        let click = |lab: &mut gpui::VisualTestContext, selector: &'static str| {
            draw(lab);
            let bounds = lab
                .debug_bounds(selector)
                .expect("visible ecological control");
            lab.simulate_click(bounds.center(), gpui::Modifiers::default());
        };
        let wait = |lab: &mut gpui::VisualTestContext, ready: &dyn Fn(&LabWorkflows) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                draw(lab);
                if context.update(|app| ready(&hud.read(app).lab_workflows)) {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "ecological experiment response did not arrive"
                );
                std::thread::yield_now();
            }
        };
        click(&mut lab, "dashboard-tab-experiments");
        wait(&mut lab, &|state| {
            state.experiments.is_some() && !state.is_busy()
        });
        click(&mut lab, "lab-new-experiment");
        wait(&mut lab, &|state| {
            state.experiment_capabilities.is_some() && !state.is_busy()
        });
        click(&mut lab, "lab-experiment-design-ecological-treatment");
        click(&mut lab, "lab-experiment-baseline-assembly");
        click(&mut lab, "lab-experiment-parameter-movement_drain");
        click(&mut lab, "lab-experiment-parameter-food_growth_rate");
        click(&mut lab, "lab-experiment-percent-100");
        context.update(|app| {
            let draft = &hud.read(app).lab_workflows.experiment_draft;
            assert_eq!(draft.baseline_family, "assembly");
            assert_eq!(draft.baseline_family, draft.challenger_family);
            assert!(
                draft
                    .question_lines()
                    .iter()
                    .any(|line| line.contains("Unchanged-value control"))
            );
        });
        draw(&mut lab);
        assert!(
            lab.debug_bounds("lab-experiment-challenger-assembly")
                .is_none()
        );
        click(&mut lab, "lab-experiment-percent-200");
        click(&mut lab, "lab-experiment-percent-prev");
        click(&mut lab, "lab-experiment-percent-next");
        click(&mut lab, "lab-experiment-treatment-tick-10");
        click(&mut lab, "lab-experiment-treatment-tick-next");
        click(&mut lab, "lab-experiment-ticks-10");
        click(&mut lab, "lab-experiment-create");
        assert!(
            requests.lock().unwrap().is_empty(),
            "invalid boundary cannot admit work"
        );
        context.update(|app| {
            let state = &hud.read(app).lab_workflows;
            assert_eq!(state.experiment_draft.treatment.unwrap().at_tick, 11);
            assert_eq!(state.experiment_draft.ticks_per_run, 10);
            assert!(!state.can_create_experiment());
        });
        click(&mut lab, "lab-experiment-treatment-tick-0");
        click(&mut lab, "lab-experiment-treatment-tick-next");
        click(&mut lab, "lab-experiment-ticks-next");
        click(&mut lab, "lab-experiment-seed-count-prev");
        draw(&mut lab);
        assert!(lab.debug_bounds("lab-experiment-question").is_some());
        let expected = context.update(|app| hud.read(app).lab_workflows.experiment_draft.clone());
        assert_eq!(expected.seeds().unwrap(), [42, 43]);
        assert_eq!(
            expected.treatment.unwrap(),
            LabExperimentTreatment {
                parameter: LabExperimentParameter::FoodGrowth,
                baseline_value: 0.25,
                scale_percent: 200,
                at_tick: 1,
            }
        );
        assert_eq!(expected.treatment.unwrap().value().unwrap(), 0.5);
        click(&mut lab, "lab-experiment-create");
        wait(&mut lab, &|state| {
            !state.is_busy()
                && state
                    .latest_experiment_creation
                    .as_ref()
                    .is_some_and(|creation| {
                        matches!(
                            &creation.state,
                            LabExperimentCreationState::Retryable { .. }
                        )
                    })
        });
        for selector in [
            "lab-experiment-design-brain-families",
            "lab-experiment-parameter-metabolism_drain",
            "lab-experiment-percent-0",
            "lab-experiment-treatment-tick-0",
            "lab-experiment-baseline-dwraon",
        ] {
            click(&mut lab, selector);
        }
        context.update(|app| assert_eq!(hud.read(app).lab_workflows.experiment_draft, expected));
        click(&mut lab, "lab-experiment-create");
        wait(&mut lab, &|state| {
            !state.is_busy() && !state.composing_experiment && state.batch.is_some()
        });
        draw(&mut lab);
        assert!(
            lab.debug_bounds("lab-experiment-retained-question")
                .is_some()
        );
        click(&mut lab, "lab-metric-total_food");
        click(&mut lab, "lab-compare");
        wait(&mut lab, &|state| {
            !state.is_busy() && state.comparison.is_some()
        });
        context.update(|app| {
            let result = hud.read(app).lab_workflows.comparison.as_ref().unwrap();
            assert_eq!(result.metric, LabMetric::TotalFood);
            assert_eq!(result.matched_seeds, [42, 43]);
            assert_eq!(result.endpoint_tick, 11);
            assert!(
                comparison_lines(result)
                    .iter()
                    .any(|line| line.contains(LabMetric::TotalFood.unit()))
            );
        });
        click(&mut lab, "lab-all-experiments");
        wait(&mut lab, &|state| {
            !state.is_busy() && state.experiments.is_some()
        });
        click(&mut lab, "lab-experiment-0");
        wait(&mut lab, &|state| !state.is_busy() && state.batch.is_some());
        draw(&mut lab);
        assert!(
            lab.debug_bounds("lab-experiment-retained-question")
                .is_none(),
            "historical unknown designs must not inherit the latest question"
        );
        let recorded = requests.lock().unwrap();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0], recorded[1]);
        assert_eq!(recorded[0].2, expected);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
    }

    #[test]
    fn gpui_lab_buttons_compare_and_review_checkpoint_without_advancing_live_owner() {
        let host = TestHost::new(
            WorldState::new(ScriptBotsConfig {
                rng_seed: Some(42),
                world_width: 120,
                world_height: 120,
                food_cell_size: 20,
                closed: true,
                population_minimum: 0,
                persistence_interval: 0,
                ..ScriptBotsConfig::default()
            })
            .unwrap(),
        );
        let saved = host.snapshot();
        let saved_world_digest = host.port.scientific_digest_v1().unwrap().overall;
        host.apply(scriptbots_runtime::HostCommand::UpdateSimulation(
            SimulationCommand {
                paused: Some(true),
                speed_multiplier: None,
                step_once: true,
            },
        ));
        let before = host.port.scientific_digest_v1().unwrap();
        let created_world_digest = before.overall.clone();
        let current = host.snapshot();
        let live_tick = current.world.tick;
        assert!(live_tick > saved.world.tick);
        let mut session = GuiSession::new(
            host.port.clone(),
            AnalyticsSnapshotProvider::empty(),
            Arc::new(|_| panic!("Lab inspection must not submit a live scientific command")),
        );
        let review_gate = Arc::new((Mutex::new(true), std::sync::Condvar::new()));
        let worker_gate = Arc::clone(&review_gate);
        let (review_started, observed_review) = std::sync::mpsc::channel();
        let save_requests = Arc::new(Mutex::new(Vec::new()));
        let recorded_saves = Arc::clone(&save_requests);
        let comparison_metrics = Arc::new(Mutex::new(Vec::new()));
        let recorded_metrics = Arc::clone(&comparison_metrics);
        let creation_requests = Arc::new(Mutex::new(Vec::new()));
        let recorded_creations = Arc::clone(&creation_requests);
        let created_experiments = Mutex::new(std::collections::BTreeMap::new());
        session.lab_submit = Some(Arc::new(move |request| {
            Ok(match request {
                LabWorkflowRequest::ListExperiments { .. } => {
                    LabWorkflowResponse::Experiments(LabExperimentPage {
                        items: vec![batch(2).summary],
                        total: 1,
                        offset: 0,
                        next_cursor: None,
                        reserved_run_slots: 0,
                        max_run_slots: 1,
                    })
                }
                LabWorkflowRequest::ExperimentCapabilities => {
                    LabWorkflowResponse::ExperimentCapabilities(experiment_capabilities())
                }
                LabWorkflowRequest::CreateExperiment {
                    request_id,
                    session_id,
                    draft,
                } => {
                    assert_eq!(session_id, saved.session_id);
                    let first_attempt = {
                        let mut requests = recorded_creations.lock().unwrap();
                        requests.push((request_id.clone(), session_id, draft.clone()));
                        requests.len() == 1
                    };
                    if first_attempt {
                        return Err("experiment receipt is unconfirmed".into());
                    }
                    let experiment = created_batch(&request_id, &draft);
                    created_experiments
                        .lock()
                        .unwrap()
                        .insert(request_id.clone(), experiment.clone());
                    LabWorkflowResponse::ExperimentCreated {
                        request_id,
                        session_id,
                        experiment,
                    }
                }
                LabWorkflowRequest::GetExperiment { experiment_id } => {
                    LabWorkflowResponse::Experiment(if experiment_id == "actual-batch" {
                        batch(2)
                    } else {
                        created_experiments
                            .lock()
                            .unwrap()
                            .get(&experiment_id)
                            .expect("created experiment is retained")
                            .clone()
                    })
                }
                LabWorkflowRequest::Compare {
                    baseline_variant,
                    challenger_variant,
                    metric,
                    ..
                } => {
                    assert_eq!(baseline_variant, "arm-00");
                    assert_eq!(challenger_variant, "arm-01");
                    recorded_metrics.lock().unwrap().push(metric);
                    LabWorkflowResponse::Comparison(comparison("arm-01", metric))
                }
                LabWorkflowRequest::ListCheckpoints { .. } => {
                    LabWorkflowResponse::Checkpoints(LabCheckpointPage {
                        items: vec![LabCheckpoint {
                            checkpoint_id: "saved-cut".into(),
                            tick: saved.world.tick,
                            agent_count: saved.world.agents.len(),
                            byte_size: 1,
                            world_digest: saved_world_digest.clone(),
                        }],
                        total: 1,
                        offset: 0,
                        next_cursor: None,
                        retained_bytes: 1,
                        max_bytes: 100,
                    })
                }
                LabWorkflowRequest::CreateCheckpoint {
                    request_id,
                    session_id,
                } => {
                    assert_eq!(session_id, saved.session_id);
                    let first_attempt = {
                        let mut requests = recorded_saves.lock().unwrap();
                        requests.push(request_id.clone());
                        requests.len() == 1
                    };
                    if first_attempt {
                        return Err("durable receipt not yet confirmed".into());
                    }
                    LabWorkflowResponse::CheckpointCreated {
                        request_id,
                        session_id,
                        checkpoint: LabCheckpoint {
                            checkpoint_id: "created-live-cut".into(),
                            tick: live_tick,
                            agent_count: current.world.agents.len(),
                            byte_size: u64::try_from(current.world.agents.len())
                                .unwrap()
                                .max(1_024),
                            world_digest: created_world_digest.clone(),
                        },
                    }
                }
                LabWorkflowRequest::ReviewCheckpoint { checkpoint_id } => {
                    review_started.send(()).unwrap();
                    let (lock, ready) = &*worker_gate;
                    let (allowed, _) = ready
                        .wait_timeout_while(
                            lock.lock().unwrap(),
                            Duration::from_secs(5),
                            |allowed| !*allowed,
                        )
                        .unwrap();
                    assert!(*allowed, "native checkpoint gate was not released");
                    let snapshot = if checkpoint_id == "created-live-cut" {
                        Arc::clone(&current)
                    } else {
                        Arc::clone(&saved)
                    };
                    LabWorkflowResponse::CheckpointFrame {
                        checkpoint_id,
                        snapshot,
                    }
                }
                _ => panic!("unexpected Lab request"),
            })
        }));
        let context = gpui::TestAppContext::single();
        let windows = context
            .update(|app| Arc::new(session).install(app))
            .unwrap();
        let hud = context.update(|app| windows.hud.root(app)).unwrap();
        let world = context.update(|app| windows.canvas.root(app)).unwrap();
        let mut lab = gpui::VisualTestContext::from_window(windows.hud.into(), &context);
        lab.simulate_resize(size(px(1000.0), px(1800.0)));
        let draw = |lab: &mut gpui::VisualTestContext| {
            lab.update(|window, app| {
                window.refresh();
                window.draw(app).clear();
            })
        };
        let click = |lab: &mut gpui::VisualTestContext, selector: &'static str| {
            draw(lab);
            let bounds = lab
                .debug_bounds(selector)
                .unwrap_or_else(|| panic!("missing production button {selector}"));
            lab.simulate_click(bounds.center(), gpui::Modifiers::default());
        };
        let wait = |lab: &mut gpui::VisualTestContext, ready: &dyn Fn(&LabWorkflows) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                draw(lab);
                if context.update(|app| ready(&hud.read(app).lab_workflows)) {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "native Lab result did not arrive"
                );
                std::thread::yield_now();
            }
        };
        click(&mut lab, "dashboard-tab-experiments");
        wait(&mut lab, &|state| {
            state.experiments.is_some() && !state.is_busy()
        });
        click(&mut lab, "lab-new-experiment");
        wait(&mut lab, &|state| {
            state.composing_experiment
                && state.experiment_capabilities.is_some()
                && !state.is_busy()
        });
        click(&mut lab, "lab-experiment-baseline-dwraon");
        click(&mut lab, "lab-experiment-challenger-assembly");
        click(&mut lab, "lab-experiment-seed-137");
        click(&mut lab, "lab-experiment-seed-count-3");
        click(&mut lab, "lab-experiment-seed-count-prev");
        click(&mut lab, "lab-experiment-ticks-10");
        click(&mut lab, "lab-experiment-ticks-next");
        draw(&mut lab);
        assert!(lab.debug_bounds("lab-experiment-preview").is_some());
        context.update(|app| {
            let draft = &hud.read(app).lab_workflows.experiment_draft;
            assert_eq!(draft.baseline_family, "dwraon");
            assert_eq!(draft.challenger_family, "assembly");
            assert_eq!(draft.seeds().unwrap(), [137, 138]);
            assert_eq!(draft.ticks_per_run, 11);
        });
        click(&mut lab, "lab-experiment-create");
        wait(&mut lab, &|state| {
            !state.is_busy()
                && state
                    .latest_experiment_creation
                    .as_ref()
                    .is_some_and(|creation| {
                        matches!(
                            &creation.state,
                            LabExperimentCreationState::Retryable { .. }
                        )
                    })
        });
        click(&mut lab, "lab-experiment-ticks-1000");
        context.update(|app| {
            assert_eq!(
                hud.read(app).lab_workflows.experiment_draft.ticks_per_run,
                11
            );
        });
        click(&mut lab, "lab-experiment-create");
        wait(&mut lab, &|state| {
            !state.is_busy()
                && !state.composing_experiment
                && state.latest_experiment_creation.as_ref().is_some_and(|creation| {
                    matches!(&creation.state, LabExperimentCreationState::Created { experiment_id }
                        if state.selected_experiment.as_ref() == Some(experiment_id))
                })
        });
        {
            let requests = creation_requests.lock().unwrap();
            assert_eq!(requests.len(), 2);
            assert_eq!(
                requests[0], requests[1],
                "retry must retain the exact source, identity and draft",
            );
            assert_eq!(requests[0].2.seeds().unwrap(), [137, 138]);
            assert_eq!(requests[0].2.ticks_per_run, 11);
        }
        click(&mut lab, "lab-all-experiments");
        wait(&mut lab, &|state| {
            state.experiments.is_some() && !state.is_busy()
        });
        click(&mut lab, "lab-experiment-0");
        wait(&mut lab, &|state| state.batch.is_some() && !state.is_busy());
        click(&mut lab, "lab-compare");
        wait(&mut lab, &|state| {
            state.comparison.is_some() && !state.is_busy()
        });
        click(&mut lab, "lab-metric-average_energy");
        context.update(|app| {
            let state = &hud.read(app).lab_workflows;
            assert_eq!(state.metric, LabMetric::AverageEnergy);
            assert!(
                state.comparison.is_none(),
                "changing endpoint must clear the population result"
            );
        });
        click(&mut lab, "lab-compare");
        wait(&mut lab, &|state| {
            !state.is_busy()
                && state.comparison.as_ref().is_some_and(|result| {
                    result.metric == LabMetric::AverageEnergy && result.mean_difference == 0.25
                })
        });
        context.update(|app| {
            let state = &hud.read(app).lab_workflows;
            let lines = comparison_lines(state.comparison.as_ref().unwrap());
            assert!(
                lines
                    .iter()
                    .any(|line| line.contains(LabMetric::AverageEnergy.unit()))
            );
        });
        assert_eq!(
            *comparison_metrics.lock().unwrap(),
            [LabMetric::AliveAgents, LabMetric::AverageEnergy]
        );
        click(&mut lab, "dashboard-tab-checkpoints");
        wait(&mut lab, &|state| {
            state.checkpoints.is_some() && !state.is_busy()
        });
        click(&mut lab, "lab-save-checkpoint");
        wait(&mut lab, &|state| {
            !state.is_busy()
                && state.latest_checkpoint_save.as_ref().is_some_and(|save| {
                    matches!(&save.state, LabCheckpointSaveState::Retryable { .. },)
                })
        });
        click(&mut lab, "lab-save-checkpoint");
        wait(&mut lab, &|state| {
            !state.is_busy()
            && state.latest_checkpoint_save.as_ref().is_some_and(|save| matches!(
                &save.state, LabCheckpointSaveState::Saved(checkpoint)
                    if checkpoint.checkpoint_id == "created-live-cut" && checkpoint.tick == live_tick,
            ))
        });
        {
            let requests = save_requests.lock().unwrap();
            assert_eq!(requests.len(), 2);
            assert_eq!(
                requests[0], requests[1],
                "retry must use the same save identity"
            );
        }
        wait(&mut lab, &|state| {
            state.checkpoints.is_some() && !state.is_busy()
        });
        context.update(|app| {
            let state = &hud.read(app).lab_workflows;
            assert!(
                state
                    .checkpoints
                    .as_ref()
                    .unwrap()
                    .items
                    .iter()
                    .all(|checkpoint| checkpoint.checkpoint_id != "created-live-cut")
            );
            assert_eq!(
                state.selected_checkpoint_metadata().unwrap().checkpoint_id,
                "created-live-cut"
            );
        });
        click(&mut lab, "lab-review-checkpoint");
        wait(&mut lab, &|state| {
            !state.is_busy()
                && state.notice.as_deref().is_some_and(|message| {
                    message.starts_with("Showing checkpoint created-live-cut")
                })
        });
        context.update(|app| {
            assert_eq!(
                world
                    .read(app)
                    .playback
                    .reconstructed
                    .as_ref()
                    .unwrap()
                    .tick,
                live_tick
            );
        });
        observed_review
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        click(&mut lab, "lab-world-live");
        click(&mut lab, "lab-checkpoint-0");
        click(&mut lab, "lab-review-checkpoint");
        wait(&mut lab, &|state| {
            !state.is_busy()
                && state
                    .notice
                    .as_deref()
                    .is_some_and(|message| message.starts_with("Showing checkpoint"))
        });
        context.update(|app| {
            let view = world.read(app);
            let reconstructed = view.playback.reconstructed.as_ref().unwrap();
            assert_eq!(reconstructed.tick, 0);
            assert_eq!(
                reconstructed.checkpoint_review.as_ref().unwrap().0,
                "saved-cut"
            );
        });
        // The production button is disabled while World shows history.
        click(&mut lab, "lab-save-checkpoint");
        assert_eq!(save_requests.lock().unwrap().len(), 2);
        // Also exercise a stale enabled handler from the earlier live paint.
        context.update(|app| {
            hud.update(app, |view, cx| {
                view.lab_action(LabAction::CreateCheckpoint, cx);
                assert!(
                    view.lab_workflows
                        .notice
                        .as_deref()
                        .is_some_and(|message| message.contains("live World"))
                );
            })
        });
        assert_eq!(save_requests.lock().unwrap().len(), 2);
        click(&mut lab, "lab-world-live");
        context.update(|app| {
            world.update(app, |view, _| {
                assert!(view.playback.reconstructed.is_none());
                assert_eq!(view.playback.mode(), PlaybackMode::Live);
                // Exercise the real state between a live intent and its next
                // presented frame, independent of background window drawing.
                view.playback.displaying_history = true;
            });
            hud.update(app, |view, cx| {
                view.lab_action(LabAction::CreateCheckpoint, cx)
            });
        });
        assert_eq!(
            save_requests.lock().unwrap().len(),
            2,
            "return-live intent cannot save before a live World frame is presented"
        );
        observed_review
            .recv_timeout(Duration::from_secs(5))
            .unwrap();

        // The Lab tab/selection remains fixed while the separate World window
        // expresses a newer live intent. Releasing the retained worker must not
        // overwrite that action with the older checkpoint frame.
        *review_gate.0.lock().unwrap() = false;
        click(&mut lab, "lab-review-checkpoint");
        observed_review
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        context.update(|app| world.update(app, |view, cx| view.playback_go_live(cx)));
        *review_gate.0.lock().unwrap() = true;
        review_gate.1.notify_one();
        wait(&mut lab, &|state| {
            !state.is_busy()
                && state
                    .notice
                    .as_deref()
                    .is_some_and(|message| message.starts_with("Checkpoint review cancelled"))
        });
        context.update(|app| {
            assert!(
                world.read(app).playback.reconstructed.is_none(),
                "late Lab review must not replace a newer World-window live action"
            )
        });
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
        assert_eq!(host.snapshot().world.tick, live_tick);
    }
}
