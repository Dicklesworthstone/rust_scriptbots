//! GPUI widgets for application-owned Lab services.
//!
//! The renderer-neutral model owns bounded background requests and validation;
//! this module only renders its observations and applies presentation intents.

use super::*;
use scriptbots_runtime::lab_workflows::{
    LAB_PAGE_SIZE, LabAction, LabCheckpointSaveState, LabExperimentCreationState, LabMetric, LabPanel,
    VARIANTS_PER_PAGE, comparison_lines,
};
#[cfg(test)]
use scriptbots_runtime::lab_workflows::{
    LabCheckpoint, LabCheckpointPage, LabComparison, LabExperiment, LabExperimentCapabilities,
    LabExperimentDraft, LabExperimentPage, LabExperimentSummary, LabVariantProgress,
    LabWorkflowRequest, LabWorkflowResponse,
};

impl SimulationView {
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
        self.lab_workflows.maybe_refresh();
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
        if action == LabAction::ReviewCheckpoint
            && !self.lab_workflows.is_busy()
            && self.lab_workflows.selected_checkpoint_metadata().is_some()
        {
            // A later World-window intent must supersede this reconstruction.
            self.timeline_cancelled
                .store(false, AtomicOrdering::Release);
        }
        self.lab_workflows.dispatch(action);
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
                    .child(self.lab_button(
                        "lab-cancel-request".into(),
                        "Discard pending result".into(),
                        LabAction::CancelRequest,
                        !idle,
                        cx,
                    )),
            );
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
            let current_source = self.host.snapshot_hub().latest().session_id == creation.session_id;
            section = section.child(div().text_xs().child(format!(
                "{}creation request {} · source session {}",
                if current_source { "" } else { "Previous-source " },
                creation.request_id,
                creation.session_id.get(),
            )));
            match &creation.state {
                LabExperimentCreationState::Pending => {
                    section = section.child(
                        div().text_xs().child("Awaiting the application's experiment receipt."),
                    );
                }
                LabExperimentCreationState::Retryable { reason } => {
                    section = section.child(
                        div().text_xs().child(format!("Creation is unconfirmed: {reason}")),
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
        let Some(capabilities) = &state.experiment_capabilities else {
            return section.child(
                div().text_sm().child("Loading available brain families and experiment limits…"),
            );
        };
        section = section
            .child(div().text_sm().child(capabilities.launch_description.clone()))
            .child(div().text_xs().child(format!(
                "Limits: {} seeds · {} ticks per run · {} total run ticks.",
                capabilities.max_seed_count,
                capabilities.max_ticks_per_run,
                capabilities.max_total_ticks,
            )));
        for (baseline, title) in [
            (true, "Baseline brain family"),
            (false, "Challenger brain family"),
        ] {
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
                    if draft.first_seed == seed { "Selected: " } else { "" },
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
                        if draft.seed_count == count { "Selected: " } else { "" },
                        if count == 1 { "" } else { "s" },
                    ),
                    LabAction::SeedCount(count),
                    editing,
                    cx,
                ));
            }
        }
        counts = counts.child(self.lab_button(
            "lab-experiment-seed-count-next".into(),
            "One more seed".into(),
            LabAction::SeedCount(draft.seed_count.checked_add(1).unwrap_or(draft.seed_count)),
            editing
                && draft
                    .seed_count
                    .checked_add(1)
                    .is_some_and(|count| count <= capabilities.max_seed_count),
            cx,
        ));
        let mut horizons = div().flex().flex_wrap().gap_2().child(self.lab_button(
            "lab-experiment-ticks-prev".into(),
            "One fewer tick".into(),
            LabAction::TicksPerRun(
                draft.ticks_per_run.checked_sub(1).unwrap_or(draft.ticks_per_run),
            ),
            editing && draft.ticks_per_run > 1,
            cx,
        ));
        for ticks in [1, 10, 100, 500, 1_000, 10_000] {
            if ticks <= capabilities.max_ticks_per_run {
                horizons = horizons.child(self.lab_button(
                    format!("lab-experiment-ticks-{ticks}"),
                    format!(
                        "{}{ticks} ticks per run",
                        if draft.ticks_per_run == ticks { "Selected: " } else { "" },
                    ),
                    LabAction::TicksPerRun(ticks),
                    editing,
                    cx,
                ));
            }
        }
        horizons = horizons.child(self.lab_button(
            "lab-experiment-ticks-next".into(),
            "One more tick".into(),
            LabAction::TicksPerRun(
                draft.ticks_per_run.checked_add(1).unwrap_or(draft.ticks_per_run),
            ),
            editing
                && draft
                    .ticks_per_run
                    .checked_add(1)
                    .is_some_and(|ticks| ticks <= capabilities.max_ticks_per_run),
            cx,
        ));
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
            .child(div().text_sm().child(format!(
                "Scientific horizon: {} ticks",
                draft.ticks_per_run,
            )))
            .child(horizons);
        match draft.seeds() {
            Ok(seeds) => {
                section = section.child(div().text_xs().child(format!(
                    "Exact seeds in each arm: {seeds:?}",
                )));
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
            section = section.child(div().text_xs().child(format!("Plan needs attention: {reason}")));
        }
        section.child(self.lab_button(
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
        ))
    }

    pub(super) fn render_lab_experiments(&self, cx: &mut Context<Self>) -> Div {
        let state = &self.lab_workflows;
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
            if let Some(error) = &batch.error_reason {
                panel = panel.child(div().text_xs().child(error.clone()));
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
            panel = panel.child(
                div().text_sm().child("No retained experiment batches. Start a matched-seed experiment above."),
            );
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

    pub(super) fn render_lab_checkpoints(&self, cx: &mut Context<Self>) -> Div {
        let state = &self.lab_workflows;
        let idle = !state.is_busy();
        let mut panel = div().flex().flex_col().gap_3().child(self.lab_panel_header("Saved checkpoints", cx))
            .child(self.render_lab_checkpoint_save(cx))
            .child(div().text_xs().child("Select a saved scientific cut to reconstruct it in World. The live owner keeps its current state and controls."))
            .child(self.lab_button("lab-world-live".into(), "Return World to live".into(), LabAction::ReturnLive, true, cx))
            .child(self.lab_button("lab-review-checkpoint".into(), "Review selected checkpoint in World".into(), LabAction::ReviewCheckpoint, idle && state.selected_checkpoint_metadata().is_some(), cx));
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
                    LabWorkflowResponse::ExperimentCapabilities(LabExperimentCapabilities {
                        families: vec!["mlp".into(), "dwraon".into(), "assembly".into()],
                        max_seed_count: 8,
                        max_ticks_per_run: 1_000,
                        max_total_ticks: 16_000,
                        launch_description: "Fresh worlds with the fixture's declared founder recipe."
                            .into(),
                    })
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
            state.composing_experiment && state.experiment_capabilities.is_some() && !state.is_busy()
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
                && state.latest_experiment_creation.as_ref().is_some_and(|creation| {
                    matches!(&creation.state, LabExperimentCreationState::Retryable { .. })
                })
        });
        click(&mut lab, "lab-experiment-ticks-1000");
        context.update(|app| {
            assert_eq!(hud.read(app).lab_workflows.experiment_draft.ticks_per_run, 11);
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
