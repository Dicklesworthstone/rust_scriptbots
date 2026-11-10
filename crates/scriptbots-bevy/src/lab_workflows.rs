//! Bevy presentation of the shared, application-owned Lab workflows.
//!
//! The reducer owns one bounded worker request. This module owns only widgets
//! and an optional saved-frame presentation; scientific work continues in the
//! existing host while a checkpoint is being viewed.

use super::*;
use scriptbots_runtime::{
    HostSessionId,
    lab_workflows::{
        LAB_PAGE_SIZE, LabAction, LabPanel, LabWorkflowSubmit, LabWorkflows,
        VARIANTS_PER_PAGE, comparison_lines,
    },
};

// Eight variant summaries and selectors, comparison evidence and navigation
// fit in this fixed pool. Retention is independent of experiment/run history.
const MAX_PANEL_ROWS: usize = 80;
const ROW_GAP: f32 = 5.0;

#[derive(Component)]
pub(super) struct OpenLabButton;

#[derive(Component)]
pub(super) struct LabWorldHeading;

#[derive(Clone, Debug, PartialEq, Eq)]
enum PanelAction {
    Panel(LabPanel),
    Close,
    Workflow(LabAction),
}

#[derive(Component, Default)]
pub(super) struct LabRow {
    index: usize,
    action: Option<PanelAction>,
    enabled: bool,
}

#[derive(Component)]
pub(super) struct LabPanelRoot;

#[derive(Component)]
pub(super) struct LabPanelBody;

#[derive(Component)]
pub(super) struct LabReviewBanner;

#[derive(Resource)]
pub(super) struct LabUi {
    rows: Vec<(Entity, Entity)>,
    shown: Vec<PanelRow>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PanelRow {
    text: String,
    action: Option<PanelAction>,
    enabled: bool,
}

impl PanelRow {
    fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            action: None,
            enabled: false,
        }
    }

    fn button(text: impl Into<String>, action: PanelAction, enabled: bool) -> Self {
        Self {
            text: text.into(),
            action: Some(action),
            enabled,
        }
    }

    fn workflow(text: impl Into<String>, action: LabAction, enabled: bool) -> Self {
        Self::button(text, PanelAction::Workflow(action), enabled)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReviewFrame {
    checkpoint_id: String,
    tick: u64,
}

/// At most one saved projection and the newest live projection are retained.
/// A source change exits review rather than mixing observations from owners.
#[derive(Resource, Default)]
pub(super) struct LabPresentation {
    panel: LabPanel,
    source: Option<(HostSessionId, u64)>,
    source_changed: bool,
    live: Option<WorldSnapshot>,
    review: Option<ReviewFrame>,
    keyboard_captured: bool,
    focus: Option<usize>,
    focus_changed: bool,
}

impl LabPresentation {
    pub(super) fn receive_live(
        &mut self,
        snapshot: WorldSnapshot,
        state: &mut SnapshotState,
        motion: &mut AgentMotionSettings,
    ) {
        let source = snapshot
            .owner_source
            .map(|(source, epoch)| (source.session_id, epoch));
        if self.source != source {
            self.source_changed = true;
            self.source = source;
            self.review = None;
            self.live = None;
            // Revision integers are meaningful only within their source.
            state.last_applied_palette = None;
        }
        if self.review.is_some() {
            self.live = Some(snapshot);
        } else {
            motion.enabled = !snapshot.reduced_motion;
            state.install_snapshot(snapshot);
        }
    }

    fn show_checkpoint(
        &mut self,
        checkpoint_id: String,
        snapshot: &RenderSnapshot,
        state: &mut SnapshotState,
        motion: &mut AgentMotionSettings,
    ) -> bool {
        let Some(mut saved) = WorldSnapshot::from_snapshot(snapshot) else {
            return false;
        };
        // A historical UID must never become a current-owner inspection or
        // selection request. Static vitals remain available in the projection.
        saved.owner_source = None;
        saved.motion = None;
        saved.motion_alpha = 1.0;
        saved.reduced_motion = true;
        saved.paused = true;
        if self.review.is_none() {
            self.live = state.latest.take();
        }
        self.review = Some(ReviewFrame {
            checkpoint_id,
            tick: saved.tick,
        });
        motion.enabled = false;
        state.last_applied_palette = None;
        state.hud_prev_tick = 0;
        state.sim_rate = 0.0;
        state.install_snapshot(saved);
        true
    }

    fn return_live(&mut self, state: &mut SnapshotState, motion: &mut AgentMotionSettings) {
        self.review = None;
        if let Some(live) = self.live.take() {
            motion.enabled = !live.reduced_motion;
            state.last_applied_palette = None;
            state.hud_prev_tick = 0;
            state.sim_rate = 0.0;
            state.install_snapshot(live);
        }
    }

    fn change_panel(&mut self, panel: LabPanel) {
        if self.panel != panel {
            self.panel = panel;
            self.focus = None;
            self.focus_changed = true;
        }
    }

    pub(super) fn reviewing(&self) -> bool {
        self.review.is_some()
    }

    fn world_controls_allowed(&self) -> bool {
        self.panel == LabPanel::Inactive && self.review.is_none() && !self.keyboard_captured
    }

    fn camera_controls_allowed(&self) -> bool {
        self.panel == LabPanel::Inactive && !self.keyboard_captured
    }
}

pub(super) fn consumer(submit: Option<LabWorkflowSubmit>) -> LabWorkflows {
    let mut model = LabWorkflows::default();
    model.submit = submit;
    model
}

pub(super) fn world_controls_allowed(presentation: Option<Res<LabPresentation>>) -> bool {
    presentation.is_none_or(|presentation| presentation.world_controls_allowed())
}

pub(super) fn camera_controls_allowed(presentation: Option<Res<LabPresentation>>) -> bool {
    presentation.is_none_or(|presentation| presentation.camera_controls_allowed())
}

pub(super) fn setup_panel(commands: &mut Commands, font: Handle<Font>) {
    // Keep the historical source visible with the modal closed. This banner is
    // a return button, so a mouse-only user cannot get stuck in saved review.
    commands.spawn((
        Button,
        Node {
            display: Display::None,
            position_type: PositionType::Absolute,
            left: Val::Percent(3.0),
            right: Val::Percent(3.0),
            bottom: Val::Px(8.0),
            padding: UiRect::all(Val::Px(8.0)),
            ..default()
        },
        GlobalZIndex(80),
        BackgroundColor(Color::srgba(0.22, 0.16, 0.04, 0.96)),
        Text::new("Saved checkpoint - return to live"),
        TextFont {
            font: font.clone(),
            font_size: 13.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 0.88, 0.57)),
        LabReviewBanner,
    ));
    let root = commands
        .spawn((
            Node {
                display: Display::None,
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            },
            GlobalZIndex(100),
            BackgroundColor(Color::srgba(0.01, 0.03, 0.025, 0.80)),
            Interaction::default(),
            FocusPolicy::Block,
            LabPanelRoot,
        ))
        .id();
    let body = commands
        .spawn((
            Node {
                width: Val::Px(760.0),
                max_width: Val::Percent(100.0),
                height: Val::Percent(96.0),
                max_height: Val::Px(880.0),
                min_height: Val::Px(0.0),
                padding: UiRect::all(Val::Px(14.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(ROW_GAP),
                overflow: Overflow::scroll_y(),
                ..default()
            },
            BackgroundColor(Color::srgba(0.035, 0.075, 0.062, 0.99)),
            BorderRadius::all(Val::Px(8.0)),
            RelativeCursorPosition::default(),
            ScrollPosition::default(),
            Interaction::default(),
            FocusPolicy::Block,
            LabPanelBody,
        ))
        .id();
    commands.entity(root).add_child(body);
    let mut rows = Vec::with_capacity(MAX_PANEL_ROWS);
    for index in 0..MAX_PANEL_ROWS {
        let text = commands
            .spawn((
                Text::default(),
                TextFont {
                    font: font.clone(),
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.92, 0.88)),
                TextLayout::new_with_linebreak(bevy::text::LineBreak::WordOrCharacter),
            ))
            .id();
        let row = commands
            .spawn((
                Node {
                    display: Display::None,
                    flex_shrink: 0.0,
                    width: Val::Percent(100.0),
                    padding: UiRect::axes(Val::Px(8.0), Val::Px(7.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(Color::NONE),
                BorderColor::all(Color::NONE),
                BorderRadius::all(Val::Px(4.0)),
                Interaction::default(),
                FocusPolicy::Block,
                LabRow {
                    index,
                    ..default()
                },
            ))
            .id();
        commands.entity(row).add_child(text);
        commands.entity(body).add_child(row);
        rows.push((row, text));
    }
    commands.insert_resource(LabUi {
        rows,
        shown: Vec::new(),
    });
}

/// Handle intent before ordinary world controls and consume results without
/// waiting. Escape stays captured for the whole frame when closing the modal.
pub(super) fn update(
    mut model: NonSendMut<LabWorkflows>,
    mut presentation: ResMut<LabPresentation>,
    mut state: ResMut<SnapshotState>,
    mut motion: ResMut<AgentMotionSettings>,
    keys: Res<ButtonInput<KeyCode>>,
    open: Query<&Interaction, (Changed<Interaction>, With<OpenLabButton>)>,
    rows: Query<(&LabRow, &Interaction), Changed<Interaction>>,
    banners: Query<&Interaction, (Changed<Interaction>, With<LabReviewBanner>)>,
    ui: Option<Res<LabUi>>,
    mut rig: ResMut<CameraRig>,
) {
    presentation.keyboard_captured = false;
    if presentation.source_changed {
        model.reset_observations();
        presentation.source_changed = false;
    }
    if let Some((session, _)) = presentation.source {
        model.synchronize(session, presentation.panel);
    }
    let mut action = rows.iter().find_map(|(row, interaction)| {
        if row.enabled && *interaction == Interaction::Pressed {
            row.action.clone()
        } else {
            None
        }
    });
    if open.iter().any(|interaction| *interaction == Interaction::Pressed)
        || keys.just_pressed(KeyCode::F6)
    {
        action = Some(if presentation.panel == LabPanel::Inactive {
            PanelAction::Panel(LabPanel::Experiments)
        } else {
            PanelAction::Close
        });
        presentation.keyboard_captured = true;
    }
    if keys.just_pressed(KeyCode::F7) {
        action = Some(PanelAction::Panel(LabPanel::Checkpoints));
        presentation.keyboard_captured = true;
    }
    if banners.iter().any(|interaction| *interaction == Interaction::Pressed)
        || (keys.just_pressed(KeyCode::Escape)
            && presentation.review.is_some()
            && presentation.panel == LabPanel::Inactive)
    {
        action = Some(PanelAction::Workflow(LabAction::ReturnLive));
        presentation.keyboard_captured = true;
    }
    if keys.just_pressed(KeyCode::Escape) && presentation.panel != LabPanel::Inactive {
        action = Some(PanelAction::Close);
        presentation.keyboard_captured = true;
    }
    if presentation.panel != LabPanel::Inactive && let Some(ui) = ui {
        let choices = ui
            .shown
            .iter()
            .enumerate()
            .filter(|(_, row)| row.enabled && row.action.is_some())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let previous = keys.just_pressed(KeyCode::ArrowUp)
            || (keys.just_pressed(KeyCode::Tab)
                && (keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight)));
        let next = keys.just_pressed(KeyCode::ArrowDown)
            || (keys.just_pressed(KeyCode::Tab) && !previous);
        if (previous || next) && !choices.is_empty() {
            let current = presentation
                .focus
                .and_then(|focus| choices.iter().position(|row| *row == focus));
            let index = match (current, previous) {
                (Some(0) | None, true) => choices.len() - 1,
                (Some(index), true) => index - 1,
                (Some(index), false) => (index + 1) % choices.len(),
                (None, false) => 0,
            };
            presentation.focus = Some(choices[index]);
            presentation.focus_changed = true;
        }
        if keys.just_pressed(KeyCode::Enter) {
            action = presentation
                .focus
                .and_then(|focus| ui.shown.get(focus))
                .filter(|row| row.enabled)
                .and_then(|row| row.action.clone())
                .or(action);
        }
    }
    if let Some(action) = action {
        rig.pending_clicked_follow = None;
        rig.clicked_follow_notice = None;
        match action {
            PanelAction::Panel(panel) => presentation.change_panel(panel),
            PanelAction::Close => {
                presentation.keyboard_captured = true;
                presentation.change_panel(LabPanel::Inactive);
            }
            PanelAction::Workflow(action) => {
                if action == LabAction::ReturnLive {
                    presentation.keyboard_captured = true;
                    presentation.return_live(&mut state, &mut motion);
                    rig.queue_fit(FitCommand::World);
                }
                model.dispatch(action);
            }
        }
    }
    if let Some((session, _)) = presentation.source {
        model.synchronize(session, presentation.panel);
    }
    if let Some((id, snapshot)) = model.poll() {
        if presentation.show_checkpoint(id, &snapshot, &mut state, &mut motion) {
            presentation.change_panel(LabPanel::Inactive);
            presentation.keyboard_captured = true;
            if let Some((session, _)) = presentation.source {
                model.synchronize(session, LabPanel::Inactive);
            }
            model.set_notice("Viewing a saved checkpoint. The live owner continues independently.");
            rig.pending_clicked_follow = None;
            rig.clicked_follow_notice = None;
            rig.queue_fit(FitCommand::World);
        } else {
            model.set_notice(
                "Checkpoint review failed: visual layers are incomplete; the displayed world is unchanged.",
            );
        }
    }
    model.maybe_refresh();
}

fn panel_rows(model: &LabWorkflows, presentation: &LabPresentation) -> Vec<PanelRow> {
    let mut rows = vec![
        PanelRow::text("EVOLUTION LAB"),
        PanelRow::text(
            "F6: Lab | F7: checkpoints | Tab / arrows / Enter: select | PageUp / PageDown: scroll | Escape: close",
        ),
        PanelRow::button("Experiments", PanelAction::Panel(LabPanel::Experiments), true),
        PanelRow::button("Saved checkpoints", PanelAction::Panel(LabPanel::Checkpoints), true),
        PanelRow::button("Close panel", PanelAction::Close, true),
    ];
    if let Some(review) = &presentation.review {
        rows.push(PanelRow::text(format!(
            "DISPLAYING SAVED CHECKPOINT {} AT TICK {}. Return to live to enable world controls.",
            review.checkpoint_id, review.tick,
        )));
        rows.push(PanelRow::workflow("Return to live world", LabAction::ReturnLive, true));
    }
    if model.submit.is_none() {
        rows.push(PanelRow::text("Lab services are unavailable in this launch."));
        return rows;
    }
    if presentation.source.is_none() {
        rows.push(PanelRow::text("Waiting for an identified host publication."));
        return rows;
    }
    let idle = !model.is_busy();
    rows.push(PanelRow::workflow(
        if idle { "Refresh" } else { "Request running..." },
        LabAction::Refresh,
        idle,
    ));
    if !idle {
        rows.push(PanelRow::workflow("Discard pending result", LabAction::CancelRequest, true));
    }
    if let Some(notice) = &model.notice {
        rows.push(PanelRow::text(notice.clone()));
    }
    match presentation.panel {
        LabPanel::Experiments => experiment_rows(model, idle, &mut rows),
        LabPanel::Checkpoints => checkpoint_rows(model, idle, &mut rows),
        LabPanel::Inactive => {}
    }
    // Fail visibly if future extensions outgrow the widget pool; never silently
    // omit the end of a scientific comparison or its provenance.
    if rows.len() > MAX_PANEL_ROWS {
        rows.clear();
        rows.push(PanelRow::text("Lab view exceeds its display capacity. Close the panel and use the control CLI."));
        rows.push(PanelRow::button("Close panel", PanelAction::Close, true));
    }
    rows
}

fn experiment_rows(model: &LabWorkflows, idle: bool, rows: &mut Vec<PanelRow>) {
    if model.selected_experiment.is_some() {
        rows.push(PanelRow::workflow("All experiments", LabAction::AllExperiments, true));
        let Some(batch) = &model.batch else {
            rows.push(PanelRow::text("Experiment details have not been observed. Refresh to retry."));
            return;
        };
        let summary = &batch.summary;
        rows.push(PanelRow::text(format!(
            "{} | {}\n{} / {} completed; {} failed\nPlan {} | observation generation {}",
            summary.experiment_id, summary.status, summary.completed_runs, summary.total_runs,
            summary.failed_runs, batch.plan_digest, batch.generation,
        )));
        if let Some(error) = &batch.error_reason {
            rows.push(PanelRow::text(format!("Run failure: {error}")));
        }
        rows.push(PanelRow::workflow(
            "Stop at durable boundary",
            LabAction::CancelExperiment,
            idle && matches!(summary.status.as_str(), "running" | "pending"),
        ));
        rows.push(PanelRow::workflow(
            "Resume unfinished runs",
            LabAction::ResumeExperiment,
            idle && !matches!(summary.status.as_str(), "running" | "pending" | "cancelling")
                && summary.completed_runs + summary.failed_runs < summary.total_runs,
        ));
        for variant in batch.variants.iter().skip(model.variant_offset).take(VARIANTS_PER_PAGE) {
            rows.push(PanelRow::text(format!(
                "{} | {} | {} / {} completed, {} running, {} failed",
                variant.variant_id, variant.brain_family, variant.completed_runs,
                variant.total_runs, variant.running_runs, variant.failed_runs,
            )));
            rows.push(PanelRow::workflow(
                if model.baseline.as_ref() == Some(&variant.variant_id) {
                    "Baseline selected"
                } else {
                    "Use as baseline"
                },
                LabAction::Baseline(variant.variant_id.clone()),
                true,
            ));
            rows.push(PanelRow::workflow(
                if model.challenger.as_ref() == Some(&variant.variant_id) {
                    "Challenger selected"
                } else {
                    "Use as challenger"
                },
                LabAction::Challenger(variant.variant_id.clone()),
                true,
            ));
        }
        rows.push(PanelRow::workflow(
            "Previous variants",
            LabAction::VariantsPage(model.variant_offset.saturating_sub(VARIANTS_PER_PAGE)),
            model.variant_offset > 0,
        ));
        rows.push(PanelRow::workflow(
            "Next variants",
            LabAction::VariantsPage(model.variant_offset.saturating_add(VARIANTS_PER_PAGE)),
            model.variant_offset.saturating_add(VARIANTS_PER_PAGE) < batch.variants.len(),
        ));
        rows.push(PanelRow::workflow(
            "Compare verified population endpoints",
            LabAction::Compare,
            idle && model.comparison_request().is_some(),
        ));
        if let Some(comparison) = &model.comparison {
            rows.extend(comparison_lines(comparison).into_iter().map(PanelRow::text));
        }
    } else if let Some(page) = &model.experiments {
        rows.push(PanelRow::text(format!(
            "{} experiments | {} / {} run slots reserved",
            page.total, page.reserved_run_slots, page.max_run_slots,
        )));
        if page.items.is_empty() {
            rows.push(PanelRow::text(
                "No experiments have been submitted. Create a matched-seed batch through the control CLI or API, then refresh.",
            ));
        }
        for item in &page.items {
            rows.push(PanelRow::workflow(
                format!(
                    "{} | {} | {} / {} completed | {} failed",
                    item.experiment_id, item.status, item.completed_runs,
                    item.total_runs, item.failed_runs,
                ),
                LabAction::SelectExperiment(item.experiment_id.clone()),
                true,
            ));
        }
        rows.push(PanelRow::workflow(
            "Previous experiments",
            LabAction::ExperimentsPage(previous_cursor(page.offset)),
            idle && page.offset > 0,
        ));
        rows.push(PanelRow::workflow(
            "Next experiments",
            LabAction::ExperimentsPage(page.next_cursor.clone()),
            idle && page.next_cursor.is_some(),
        ));
    } else {
        rows.push(PanelRow::text("No experiment page has been observed yet."));
    }
}

fn previous_cursor(offset: usize) -> Option<String> {
    (offset > LAB_PAGE_SIZE).then(|| (offset - LAB_PAGE_SIZE).to_string())
}

fn checkpoint_rows(model: &LabWorkflows, idle: bool, rows: &mut Vec<PanelRow>) {
    rows.push(PanelRow::text(
        "Review reconstructs the selected saved world. The live owner continues separately.",
    ));
    let Some(page) = &model.checkpoints else {
        rows.push(PanelRow::text("No checkpoint page has been observed yet."));
        return;
    };
    rows.push(PanelRow::text(format!(
        "{} checkpoints | {} / {} retained bytes",
        page.total, page.retained_bytes, page.max_bytes,
    )));
    if page.items.is_empty() {
        rows.push(PanelRow::text("No checkpoints are retained for this run."));
    }
    for checkpoint in &page.items {
        let selected = model.selected_checkpoint.as_ref() == Some(&checkpoint.checkpoint_id);
        rows.push(PanelRow::workflow(
            format!(
                "{}{} | tick {} | {} agents | {} bytes",
                if selected { "Selected: " } else { "" },
                checkpoint.checkpoint_id, checkpoint.tick,
                checkpoint.agent_count, checkpoint.byte_size,
            ),
            LabAction::SelectCheckpoint(checkpoint.checkpoint_id.clone()),
            true,
        ));
    }
    if let Some(checkpoint) = model.selected_checkpoint_metadata() {
        rows.push(PanelRow::text(format!("Saved world digest: {}", checkpoint.world_digest)));
    }
    rows.push(PanelRow::workflow(
        "Review selected checkpoint in world",
        LabAction::ReviewCheckpoint,
        idle && model.selected_checkpoint_metadata().is_some(),
    ));
    rows.push(PanelRow::workflow(
        "Previous checkpoints",
        LabAction::CheckpointsPage(previous_cursor(page.offset)),
        idle && page.offset > 0,
    ));
    rows.push(PanelRow::workflow(
        "Next checkpoints",
        LabAction::CheckpointsPage(page.next_cursor.clone()),
        idle && page.next_cursor.is_some(),
    ));
}

fn retained_focus(
    previous: &[PanelRow],
    next: &[PanelRow],
    focus: Option<usize>,
) -> Option<usize> {
    let action = previous.get(focus?)?.action.as_ref()?;
    next.iter().position(|row| row.enabled && row.action.as_ref() == Some(action))
}

/// Update the fixed widget pool. Stable observations do not mark text changed.
pub(super) fn refresh_panel(
    model: NonSend<LabWorkflows>,
    mut presentation: ResMut<LabPresentation>,
    ui: Option<ResMut<LabUi>>,
    mut nodes: Query<&mut Node>,
    roots: Query<Entity, With<LabPanelRoot>>,
    mut rows: Query<(&mut LabRow, &Interaction, &mut BackgroundColor, &mut BorderColor)>,
    mut texts: Query<&mut Text>,
    banners: Query<Entity, With<LabReviewBanner>>,
    headings: Query<Entity, With<LabWorldHeading>>,
    mut wheels: MessageReader<MouseWheel>,
    keys: Res<ButtonInput<KeyCode>>,
    mut bodies: Query<(&ComputedNode, &mut ScrollPosition), With<LabPanelBody>>,
    sizes: Query<&ComputedNode, With<LabRow>>,
) {
    let Some(mut ui) = ui else { return; };
    let open = presentation.panel != LabPanel::Inactive;
    for entity in &roots {
        if let Ok(mut node) = nodes.get_mut(entity) {
            let display = if open { Display::Flex } else { Display::None };
            if node.display != display {
                node.display = display;
            }
        }
    }
    for entity in &headings {
        if let Ok(mut text) = texts.get_mut(entity) {
            set_hud_text_if_changed(
                &mut text,
                if presentation.review.is_some() {
                    "SCRIPTBOTS  /  SAVED CHECKPOINT".into()
                } else {
                    "SCRIPTBOTS  /  LIVE WORLD".into()
                },
            );
        }
    }
    for entity in &banners {
        if let Ok(mut node) = nodes.get_mut(entity) {
            let display = if presentation.review.is_some() { Display::Flex } else { Display::None };
            if node.display != display {
                node.display = display;
            }
        }
        if let (Some(review), Ok(mut text)) = (&presentation.review, texts.get_mut(entity)) {
            set_hud_text_if_changed(
                &mut text,
                format!(
                    "SAVED CHECKPOINT {} | TICK {} | Live world continues separately. Click here or press Escape to return live.",
                    review.checkpoint_id, review.tick,
                ),
            );
        }
    }
    let desired = if open { panel_rows(&model, &presentation) } else { Vec::new() };
    if ui.shown != desired {
        for (index, (entity, label)) in ui.rows.iter().copied().enumerate() {
            let value = desired.get(index);
            if ui.shown.get(index) == value {
                continue;
            }
            if let Ok(mut node) = nodes.get_mut(entity) {
                node.display = if value.is_some() { Display::Flex } else { Display::None };
            }
            if let Ok((mut row, _, _, _)) = rows.get_mut(entity) {
                row.action = value.and_then(|value| value.action.clone());
                row.enabled = value.is_some_and(|value| value.enabled);
            }
            if let Ok(mut text) = texts.get_mut(label) {
                set_hud_text_if_changed(
                    &mut text,
                    value.map_or_else(String::new, |value| value.text.clone()),
                );
            }
        }
        // Async notices and detail rows can change offsets. Focus follows the
        // same enabled action, never whichever button inherits its row number.
        let focus = retained_focus(&ui.shown, &desired, presentation.focus);
        if presentation.focus != focus {
            presentation.focus = focus;
            presentation.focus_changed = true;
        }
        ui.shown = desired;
    }
    for (row, interaction, mut background, mut border) in &mut rows {
        let focused = presentation.focus == Some(row.index) && row.enabled;
        let color = if row.action.is_none() {
            Color::NONE
        } else if !row.enabled {
            Color::srgba(0.10, 0.13, 0.12, 0.65)
        } else if focused || *interaction != Interaction::None {
            follow_hover_color()
        } else {
            follow_idle_color()
        };
        if background.0 != color {
            background.0 = color;
        }
        let color = BorderColor::all(if focused { Color::srgb(0.75, 0.91, 0.65) } else { Color::NONE });
        if *border != color {
            *border = color;
        }
    }
    let wheel_pixels: f32 = wheels
        .read()
        .filter(|wheel| wheel.y.is_finite())
        .map(|wheel| match wheel.unit {
            MouseScrollUnit::Line => wheel.y.clamp(-100.0, 100.0) * 40.0,
            MouseScrollUnit::Pixel => wheel.y.clamp(-4000.0, 4000.0),
        })
        .sum();
    if !open {
        return;
    }
    let mut heights = [0.0_f32; MAX_PANEL_ROWS];
    for (index, (entity, _)) in ui.rows.iter().enumerate() {
        if let Ok(size) = sizes.get(*entity) {
            heights[index] = size.size.y * size.inverse_scale_factor;
        }
    }
    for (computed, mut scroll) in &mut bodies {
        let viewport = computed.size.y * computed.inverse_scale_factor;
        let limit = ((computed.content_size.y - computed.size.y).max(0.0)
            * computed.inverse_scale_factor)
            .max(0.0);
        let page = if keys.just_pressed(KeyCode::PageDown) {
            viewport * 0.8
        } else if keys.just_pressed(KeyCode::PageUp) {
            -viewport * 0.8
        } else {
            0.0
        };
        scroll.0.y = (scroll.0.y - wheel_pixels + page).clamp(0.0, limit);
        if presentation.focus_changed
            && let Some(index) = presentation.focus
            && heights.get(index).is_some_and(|height| *height > 0.0)
        {
            let top: f32 = heights[..index].iter().map(|height| height + ROW_GAP).sum();
            let bottom = top + heights[index];
            if top < scroll.0.y {
                scroll.0.y = top;
            }
            if bottom > scroll.0.y + viewport {
                scroll.0.y = (bottom - viewport).max(0.0);
            }
            scroll.0.y = scroll.0.y.clamp(0.0, limit);
            presentation.focus_changed = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain_tests::{motion_test_source, sample_world_snapshot};
    use scriptbots_runtime::lab_workflows::{
        LAB_MAX_VARIANTS, LabExperiment, LabExperimentSummary, LabVariantProgress,
    };
    use scriptbots_runtime::{HostCore, HostCoreOptions};

    fn live_frame(tick: u64) -> WorldSnapshot {
        let mut frame = sample_world_snapshot();
        frame.tick = tick;
        frame.revision = tick;
        motion_test_source(&mut frame);
        frame
    }

    fn saved_snapshot() -> Arc<RenderSnapshot> {
        let world = WorldState::new(scriptbots_core::ScriptBotsConfig {
            world_width: 80,
            world_height: 80,
            food_cell_size: 40,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            rng_seed: Some(17),
            ..default()
        })
        .expect("small world");
        HostCore::new(
            HostSessionId::new(717),
            world,
            HostCoreOptions {
                capture_agent_visuals: true,
                ..default()
            },
        )
        .expect("drawing host")
        .latest_snapshot()
    }

    #[test]
    fn checkpoint_review_keeps_latest_live_and_invalidates_equal_revision_caches() {
        let mut presentation = LabPresentation::default();
        let mut state = SnapshotState::default();
        let mut motion = AgentMotionSettings { enabled: true };
        presentation.receive_live(live_frame(12), &mut state, &mut motion);
        state.last_applied_revision = 1;
        state.last_applied_palette = Some(ColorPaletteMode::Natural);
        assert!(presentation.show_checkpoint("saved-1".into(), &saved_snapshot(), &mut state, &mut motion));
        let saved = state.latest.as_ref().unwrap();
        assert_eq!(saved.tick, 0);
        assert!(saved.owner_source.is_none());
        assert!(saved.motion.is_none());
        assert!(saved.paused && saved.reduced_motion);
        assert!(!motion.enabled);
        assert!(state.last_applied_palette.is_none());
        assert!(!presentation.world_controls_allowed());

        presentation.receive_live(live_frame(13), &mut state, &mut motion);
        presentation.receive_live(live_frame(29), &mut state, &mut motion);
        assert_eq!(state.latest.as_ref().unwrap().tick, 0);
        assert_eq!(presentation.live.as_ref().unwrap().tick, 29);
        state.last_applied_palette = Some(ColorPaletteMode::Natural);
        presentation.return_live(&mut state, &mut motion);
        assert_eq!(state.latest.as_ref().unwrap().tick, 29);
        assert!(state.latest.as_ref().unwrap().owner_source.is_some());
        assert!(presentation.live.is_none());
        assert!(presentation.review.is_none());
        assert!(state.last_applied_palette.is_none());
        assert!(motion.enabled);
        assert!(presentation.world_controls_allowed());
    }

    #[test]
    fn new_owner_epoch_leaves_saved_review_and_clears_old_retention() {
        let mut presentation = LabPresentation::default();
        let mut state = SnapshotState::default();
        let mut motion = AgentMotionSettings { enabled: true };
        presentation.receive_live(live_frame(12), &mut state, &mut motion);
        assert!(presentation.show_checkpoint("saved-1".into(), &saved_snapshot(), &mut state, &mut motion));
        presentation.source_changed = false;
        let mut reset = live_frame(1);
        reset.owner_source.as_mut().unwrap().1 = 9;
        presentation.receive_live(reset, &mut state, &mut motion);
        assert_eq!(state.latest.as_ref().unwrap().tick, 1);
        assert!(presentation.source_changed);
        assert!(presentation.review.is_none());
        assert!(presentation.live.is_none());
    }

    #[test]
    fn incomplete_checkpoint_preserves_live_projection() {
        let mut presentation = LabPresentation::default();
        let mut state = SnapshotState::default();
        let mut motion = AgentMotionSettings { enabled: true };
        presentation.receive_live(live_frame(12), &mut state, &mut motion);
        let mut saved = (*saved_snapshot()).clone();
        Arc::make_mut(&mut saved.layers.food).cells.clear();
        assert!(!presentation.show_checkpoint("broken".into(), &saved, &mut state, &mut motion));
        assert_eq!(state.latest.as_ref().unwrap().tick, 12);
        assert!(presentation.review.is_none());
        assert!(presentation.live.is_none());
        assert!(motion.enabled);
    }

    #[test]
    fn variant_page_reaches_final_variant_without_growing_the_widget_pool() {
        let mut model = consumer(Some(Arc::new(|_| Err("not used".into()))));
        model.selected_experiment = Some("cohort".into());
        model.batch = Some(LabExperiment {
            summary: LabExperimentSummary {
                experiment_id: "cohort".into(),
                status: "completed".into(),
                total_runs: LAB_MAX_VARIANTS,
                completed_runs: LAB_MAX_VARIANTS,
                failed_runs: 0,
            },
            plan_digest: "verified-plan".into(),
            generation: 3,
            variants: (0..LAB_MAX_VARIANTS).map(|index| LabVariantProgress {
                variant_id: format!("variant-{index}"),
                brain_family: "mlp".into(),
                total_runs: 1,
                completed_runs: 1,
                failed_runs: 0,
                running_runs: 0,
            }).collect(),
            error_reason: None,
        });
        let presentation = LabPresentation {
            panel: LabPanel::Experiments,
            source: Some((HostSessionId::new(1), 0)),
            ..default()
        };
        let last_variant = format!("variant-{} |", LAB_MAX_VARIANTS - 1);
        let first = panel_rows(&model, &presentation);
        assert!(first.iter().any(|row| row.text.starts_with("variant-0 |")));
        assert!(!first.iter().any(|row| row.text.starts_with(&last_variant)));
        model.dispatch(LabAction::VariantsPage(LAB_MAX_VARIANTS - VARIANTS_PER_PAGE));
        let last = panel_rows(&model, &presentation);
        assert!(last.iter().any(|row| row.text.starts_with(&last_variant)));
        assert!(!last.iter().any(|row| row.text.starts_with("variant-0 |")));
        assert!(first.len() <= MAX_PANEL_ROWS && last.len() <= MAX_PANEL_ROWS);
    }

    #[test]
    fn async_row_changes_keep_keyboard_focus_on_the_same_action() {
        let previous = vec![
            PanelRow::text("status"),
            PanelRow::workflow("refresh", LabAction::Refresh, true),
            PanelRow::workflow("stop", LabAction::CancelExperiment, true),
        ];
        let shifted = vec![
            PanelRow::text("new notice"),
            previous[0].clone(),
            previous[1].clone(),
            previous[2].clone(),
        ];
        assert_eq!(retained_focus(&previous, &shifted, Some(1)), Some(2));
        let replaced = vec![
            previous[0].clone(),
            previous[2].clone(),
        ];
        assert_eq!(retained_focus(&previous, &replaced, Some(1)), None);
        let disabled = vec![
            previous[0].clone(),
            PanelRow::workflow("refresh", LabAction::Refresh, false),
        ];
        assert_eq!(retained_focus(&previous, &disabled, Some(1)), None);
    }

    #[test]
    fn modal_escape_is_consumed_and_optional_capture_context_stays_live() {
        let mut app = App::new();
        let mut presentation = LabPresentation {
            panel: LabPanel::Experiments,
            ..default()
        };
        presentation.source = Some((HostSessionId::new(1), 0));
        let mut keys = ButtonInput::<KeyCode>::default();
        keys.press(KeyCode::Escape);
        app.insert_resource(keys)
            .insert_resource(presentation)
            .insert_resource(AgentMotionSettings { enabled: true })
            .init_resource::<SnapshotState>()
            .init_resource::<CameraRig>()
            .insert_non_send_resource(consumer(None))
            .add_systems(Update, update);
        app.update();
        let presentation = app.world().resource::<LabPresentation>();
        assert_eq!(presentation.panel, LabPanel::Inactive);
        assert!(presentation.keyboard_captured);
        assert!(!presentation.world_controls_allowed());
        assert!(world_controls_allowed(None));
        assert!(camera_controls_allowed(None));
    }

    #[test]
    fn panel_systems_initialize_without_query_conflicts_and_keep_a_fixed_pool() {
        let mut app = App::new();
        app.insert_non_send_resource(consumer(None))
            .init_resource::<LabPresentation>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_message::<MouseWheel>()
            .add_systems(Startup, |mut commands: Commands| {
                setup_panel(&mut commands, Handle::default());
            })
            .add_systems(Update, refresh_panel);
        app.world_mut().resource_mut::<LabPresentation>().panel = LabPanel::Experiments;
        app.update();
        let ui = app.world().resource::<LabUi>();
        assert_eq!(ui.rows.len(), MAX_PANEL_ROWS);
        assert!(ui.shown.iter().any(|row| row.text.contains("unavailable")));
        app.update();
        assert_eq!(app.world_mut().query::<&LabRow>().iter(app.world()).count(), MAX_PANEL_ROWS);
    }
}
