//! Bevy presentation of the shared, application-owned Lab workflows.
//!
//! The reducer owns one bounded worker request. This module owns only widgets
//! and an optional saved-frame presentation; scientific work continues in the
//! existing host while a checkpoint is being viewed.

use super::*;
use scriptbots_runtime::{
    HostSessionId,
    lab_workflows::{
        LAB_PAGE_SIZE, LabAction, LabCheckpointSaveState, LabEventIdentity, LabEventReview,
        LabMetric, LabPanel, LabWorkflowSubmit, LabWorkflows, VARIANTS_PER_PAGE, comparison_lines,
    },
};

// Eight variant summaries and selectors, comparison evidence and navigation
// fit in this fixed pool. Retention is independent of experiment/run history.
const MAX_PANEL_ROWS: usize = 80;
const ROW_GAP: f32 = 5.0;
const NARRATIVE_PAGE_SIZE: usize = 12;

#[derive(Component)]
pub(super) struct OpenLabButton;

#[derive(Component)]
pub(super) struct LabWorldHeading;

#[derive(Clone, Debug, PartialEq, Eq)]
enum PanelAction {
    Panel(LabPanel),
    Close,
    Workflow(LabAction),
    NarrativeOlder,
    NarrativeNewer,
    NarrativeLatest,
    SelectNarrative {
        session_id: HostSessionId,
        ordinal: u64,
    },
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
enum ReviewSource {
    Checkpoint(String),
    NarrativeEvent {
        event: LabEventIdentity,
        checkpoint_id: String,
        base_tick: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReviewFrame {
    source: ReviewSource,
    tick: u64,
}

impl ReviewFrame {
    fn heading(&self) -> &'static str {
        match &self.source {
            ReviewSource::Checkpoint(_) => "SCRIPTBOTS  /  SAVED CHECKPOINT",
            ReviewSource::NarrativeEvent { .. } => "SCRIPTBOTS  /  EVENT REVIEW",
        }
    }

    fn description(&self) -> String {
        match &self.source {
            ReviewSource::Checkpoint(id) => format!("SAVED CHECKPOINT {id} | TICK {}", self.tick),
            ReviewSource::NarrativeEvent {
                event,
                checkpoint_id,
                base_tick,
            } => format!(
                "EVENT #{} | TICK {} | SOURCE CHECKPOINT {checkpoint_id} AT TICK {base_tick} | SESSION {}",
                event.ordinal,
                self.tick,
                event.session_id.get(),
            ),
        }
    }
}

/// A page into the owner's immutable ring, addressed by absolute event ordinal.
/// New events cannot move a pinned page or silently select another event.
#[derive(Default)]
struct NarrativeTimeline {
    observed_tick: Option<u64>,
    records: Arc<Vec<scriptbots_core::narrative::EventRecord>>,
    dropped: u64,
    newest_on_page: Option<u64>,
    selected: Option<u64>,
    notice: Option<String>,
}

impl NarrativeTimeline {
    fn latest_ordinal(&self) -> Option<u64> {
        self.dropped
            .checked_add(u64::try_from(self.records.len()).ok()?)
    }

    fn record(&self, ordinal: u64) -> Option<&scriptbots_core::narrative::EventRecord> {
        let index = usize::try_from(ordinal.checked_sub(self.dropped)?.checked_sub(1)?).ok()?;
        self.records.get(index)
    }

    fn observe(&mut self, snapshot: &WorldSnapshot) {
        self.observed_tick = Some(snapshot.tick);
        if Arc::ptr_eq(&self.records, &snapshot.narrative_events)
            && self.dropped == snapshot.narrative_dropped_events
        {
            return;
        }
        let previous_latest = self.latest_ordinal();
        let previous_records = Arc::clone(&self.records);
        let previous_dropped = self.dropped;
        self.records = Arc::clone(&snapshot.narrative_events);
        self.dropped = snapshot.narrative_dropped_events;
        let Some(latest) = self.latest_ordinal() else {
            self.selected = None;
            self.newest_on_page = None;
            self.notice = Some("Narrative event ordinals exceed the supported range.".into());
            return;
        };
        if self.dropped < previous_dropped || previous_latest.is_some_and(|value| latest < value) {
            self.selected = None;
            self.newest_on_page = None;
            self.notice =
                Some("Narrative history restarted; the previous selection was cleared.".into());
        } else if let Some(ordinal) = self.selected {
            let previous = ordinal
                .checked_sub(previous_dropped)
                .and_then(|offset| offset.checked_sub(1))
                .and_then(|index| usize::try_from(index).ok())
                .and_then(|index| previous_records.get(index));
            if self.record(ordinal) != previous {
                self.selected = None;
                self.notice = Some(format!(
                    "Event #{ordinal} is no longer retained unchanged; the previous selection was cleared."
                ));
            }
        }
        if let Some(end) = self.newest_on_page {
            if end <= self.dropped {
                self.newest_on_page = self
                    .dropped
                    .checked_add(NARRATIVE_PAGE_SIZE as u64)
                    .map(|value| value.min(latest));
                self.notice = Some(
                    "The viewed page left retained history; showing the oldest available page."
                        .into(),
                );
            } else {
                self.newest_on_page = Some(end.min(latest));
            }
        }
    }

    fn page(&self) -> Vec<(u64, &scriptbots_core::narrative::EventRecord)> {
        let Some(latest) = self.latest_ordinal() else {
            return Vec::new();
        };
        let end = self.newest_on_page.unwrap_or(latest).min(latest);
        let Some(count) = end
            .checked_sub(self.dropped)
            .and_then(|value| usize::try_from(value).ok())
        else {
            return Vec::new();
        };
        self.records
            .iter()
            .take(count)
            .enumerate()
            .rev()
            .take(NARRATIVE_PAGE_SIZE)
            .map(|(index, record)| (self.dropped + index as u64 + 1, record))
            .collect()
    }

    fn can_show_older(&self) -> bool {
        self.page()
            .last()
            .is_some_and(|(ordinal, _)| *ordinal > self.dropped.saturating_add(1))
    }

    fn can_show_newer(&self) -> bool {
        self.newest_on_page
            .zip(self.latest_ordinal())
            .is_some_and(|(end, latest)| end < latest)
    }

    fn older(&mut self) {
        if self.can_show_older() {
            self.newest_on_page = self.page().last().map(|(ordinal, _)| ordinal - 1);
        }
    }

    fn newer(&mut self) {
        if let Some((end, latest)) = self.newest_on_page.zip(self.latest_ordinal()) {
            let next = end.saturating_add(NARRATIVE_PAGE_SIZE as u64).min(latest);
            self.newest_on_page = (next < latest).then_some(next);
        }
    }

    fn select(&mut self, ordinal: u64) {
        if self.record(ordinal).is_some() {
            self.selected = Some(ordinal);
            self.notice = None;
        }
    }
}

/// At most one saved projection and the newest live projection are retained.
/// A source change exits review rather than mixing observations from owners.
#[derive(Resource, Default)]
pub(super) struct LabPresentation {
    panel: LabPanel,
    // Scientific epoch advances during an ordinary run; it is not an owner identity.
    source: Option<HostSessionId>,
    source_changed: bool,
    live: Option<WorldSnapshot>,
    review: Option<ReviewFrame>,
    narrative: NarrativeTimeline,
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
        let source = snapshot.owner_source.map(|(source, _)| source.session_id);
        if self.source != source {
            self.source_changed = true;
            self.source = source;
            self.review = None;
            self.live = None;
            self.narrative = NarrativeTimeline::default();
            // Revision integers are meaningful only within their source.
            state.last_applied_palette = None;
        }
        if source.is_some() {
            self.narrative.observe(&snapshot);
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
        self.show_history(
            ReviewSource::Checkpoint(checkpoint_id),
            snapshot,
            state,
            motion,
        )
    }

    fn show_event_review(
        &mut self,
        review: LabEventReview,
        state: &mut SnapshotState,
        motion: &mut AgentMotionSettings,
    ) -> bool {
        if self
            .selected_event()
            .is_none_or(|(event, record)| event != review.event || record != &review.record)
            || review.snapshot.session_id != review.event.session_id
            || review.snapshot.world.tick != review.event.tick
            || review.base_tick > review.event.tick
        {
            return false;
        }
        self.show_history(
            ReviewSource::NarrativeEvent {
                event: review.event,
                checkpoint_id: review.checkpoint_id,
                base_tick: review.base_tick,
            },
            &review.snapshot,
            state,
            motion,
        )
    }

    fn show_history(
        &mut self,
        source: ReviewSource,
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
            source,
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

    fn select_narrative(&mut self, session_id: HostSessionId, ordinal: u64) {
        if self.source == Some(session_id) {
            self.narrative.select(ordinal);
        }
    }

    fn selected_event(
        &self,
    ) -> Option<(LabEventIdentity, &scriptbots_core::narrative::EventRecord)> {
        let session_id = self.source?;
        let ordinal = self.narrative.selected?;
        let record = self.narrative.record(ordinal)?;
        Some((
            LabEventIdentity {
                session_id,
                ordinal,
                tick: record.tick.0,
            },
            record,
        ))
    }

    pub(super) fn reviewing(&self) -> bool {
        self.review.is_some()
    }

    fn checkpoint_source_is_live(&self) -> bool {
        self.source.is_some() && self.review.is_none()
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
                LabRow { index, ..default() },
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
    if let Some(session) = presentation.source {
        model.synchronize(session, presentation.panel);
    }
    model.select_event(presentation.selected_event());
    let mut action = rows.iter().find_map(|(row, interaction)| {
        if row.enabled && *interaction == Interaction::Pressed {
            row.action.clone()
        } else {
            None
        }
    });
    if open
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
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
    if keys.just_pressed(KeyCode::F8) {
        action = Some(PanelAction::Panel(LabPanel::Narrative));
        presentation.keyboard_captured = true;
    }
    if banners
        .iter()
        .any(|interaction| *interaction == Interaction::Pressed)
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
    if presentation.panel != LabPanel::Inactive
        && let Some(ui) = ui
    {
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
        let next =
            keys.just_pressed(KeyCode::ArrowDown) || (keys.just_pressed(KeyCode::Tab) && !previous);
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
            PanelAction::NarrativeOlder => presentation.narrative.older(),
            PanelAction::NarrativeNewer => presentation.narrative.newer(),
            PanelAction::NarrativeLatest => presentation.narrative.newest_on_page = None,
            PanelAction::SelectNarrative {
                session_id,
                ordinal,
            } => {
                presentation.select_narrative(session_id, ordinal);
            }
            PanelAction::Workflow(action) => {
                // A row may have been painted before review/source changed.
                // Never let that stale input capture the hidden live owner.
                if action == LabAction::CreateCheckpoint
                    && !presentation.checkpoint_source_is_live()
                {
                    model.set_notice(
                        "Show the identified live world before saving or retrying a checkpoint.",
                    );
                } else {
                    if action == LabAction::ReturnLive {
                        presentation.keyboard_captured = true;
                        presentation.return_live(&mut state, &mut motion);
                        rig.queue_fit(FitCommand::World);
                    }
                    model.dispatch(action);
                }
            }
        }
    }
    if let Some(session) = presentation.source {
        model.synchronize(session, presentation.panel);
    }
    model.select_event(presentation.selected_event());
    if let Some((id, snapshot)) = model.poll() {
        if presentation.show_checkpoint(id, &snapshot, &mut state, &mut motion) {
            presentation.change_panel(LabPanel::Inactive);
            presentation.keyboard_captured = true;
            if let Some(session) = presentation.source {
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
    if let Some(review) = model.take_event_review() {
        if presentation.show_event_review(review, &mut state, &mut motion) {
            presentation.change_panel(LabPanel::Inactive);
            presentation.keyboard_captured = true;
            if let Some(session) = presentation.source {
                model.synchronize(session, LabPanel::Inactive);
            }
            model.set_notice("Viewing the selected event's reconstructed world. Return live to resume live-world controls.");
            rig.pending_clicked_follow = None;
            rig.clicked_follow_notice = None;
            rig.queue_fit(FitCommand::World);
        } else {
            model.set_notice("Event review could not match the selected evidence or complete visual layers; the displayed world is unchanged.");
        }
    }
    model.maybe_refresh();
}

fn panel_rows(model: &LabWorkflows, presentation: &LabPresentation) -> Vec<PanelRow> {
    let mut rows = vec![
        PanelRow::text("EVOLUTION LAB"),
        PanelRow::text(
            "F6: Lab | F7: checkpoints | F8: timeline | Tab / arrows / Enter: select | PageUp / PageDown: scroll | Escape: close",
        ),
        PanelRow::button(
            "Experiments",
            PanelAction::Panel(LabPanel::Experiments),
            true,
        ),
        PanelRow::button(
            "Saved checkpoints",
            PanelAction::Panel(LabPanel::Checkpoints),
            true,
        ),
        PanelRow::button(
            "Narrated timeline",
            PanelAction::Panel(LabPanel::Narrative),
            true,
        ),
        PanelRow::button("Close panel", PanelAction::Close, true),
    ];
    if let Some(review) = &presentation.review {
        rows.push(PanelRow::text(format!(
            "DISPLAYING {}. Return to live to enable world controls.",
            review.description(),
        )));
        rows.push(PanelRow::workflow(
            "Return to live world",
            LabAction::ReturnLive,
            true,
        ));
    }
    if presentation.panel == LabPanel::Narrative {
        narrative_rows(model, presentation, &mut rows);
        return rows;
    }
    if model.submit.is_none() {
        rows.push(PanelRow::text(
            "Lab services are unavailable in this launch.",
        ));
        return rows;
    }
    if presentation.source.is_none() {
        rows.push(PanelRow::text(
            "Waiting for an identified host publication.",
        ));
        return rows;
    }
    let idle = !model.is_busy();
    rows.push(PanelRow::workflow(
        if idle {
            "Refresh"
        } else {
            "Request running..."
        },
        LabAction::Refresh,
        idle,
    ));
    if !idle {
        rows.push(PanelRow::workflow(
            "Discard pending result",
            LabAction::CancelRequest,
            true,
        ));
    }
    if let Some(notice) = &model.notice {
        rows.push(PanelRow::text(notice.clone()));
    }
    match presentation.panel {
        LabPanel::Experiments => experiment_rows(model, idle, &mut rows),
        LabPanel::Checkpoints => checkpoint_rows(model, presentation, idle, &mut rows),
        LabPanel::Inactive | LabPanel::Narrative => {}
    }
    // Fail visibly if future extensions outgrow the widget pool; never silently
    // omit the end of a scientific comparison or its provenance.
    if rows.len() > MAX_PANEL_ROWS {
        rows.clear();
        rows.push(PanelRow::text(
            "Lab view exceeds its display capacity. Close the panel and use the control CLI.",
        ));
        rows.push(PanelRow::button("Close panel", PanelAction::Close, true));
    }
    rows
}

fn narrative_rows(model: &LabWorkflows, presentation: &LabPresentation, rows: &mut Vec<PanelRow>) {
    use scriptbots_core::narrative_text::fixed;

    let timeline = &presentation.narrative;
    let Some((session, tick)) = presentation.source.zip(timeline.observed_tick) else {
        rows.push(PanelRow::text("Waiting for identified scientific history."));
        return;
    };
    rows.push(PanelRow::text(format!(
        "LIVE NARRATIVE | session {} | observed tick {tick}",
        session.get(),
    )));
    rows.push(PanelRow::text(
        "These are the owner's canonical scientific events. Select an event to inspect its recorded evidence.",
    ));
    if presentation.review.is_some() {
        rows.push(PanelRow::text(
            "The timeline follows the live owner while the world shows a saved observation.",
        ));
    }
    if let Some(notice) = &timeline.notice {
        rows.push(PanelRow::text(notice.clone()));
    }
    if let Some(notice) = &model.notice {
        rows.push(PanelRow::text(notice.clone()));
    }
    if model.is_busy() {
        rows.push(PanelRow::workflow(
            "Discard pending review result",
            LabAction::CancelRequest,
            true,
        ));
    }
    if timeline.dropped > 0 {
        rows.push(PanelRow::text(format!(
            "Retention gap: {} earlier events are outside the owner's in-memory history.",
            timeline.dropped,
        )));
    }
    rows.push(PanelRow::text(
        "This observation does not confirm durable event storage.",
    ));
    let page = timeline.page();
    if page.is_empty() {
        rows.push(PanelRow::text(
            "No narrative events are retained in this observation.",
        ));
        return;
    }
    rows.push(PanelRow::button(
        "Older events",
        PanelAction::NarrativeOlder,
        timeline.can_show_older(),
    ));
    rows.push(PanelRow::button(
        "Newer events",
        PanelAction::NarrativeNewer,
        timeline.can_show_newer(),
    ));
    rows.push(PanelRow::button(
        if timeline.newest_on_page.is_none() {
            "Following newest events"
        } else {
            "Follow newest events"
        },
        PanelAction::NarrativeLatest,
        timeline.newest_on_page.is_some(),
    ));
    for (ordinal, record) in page {
        rows.push(PanelRow::button(
            format!(
                "{}#{ordinal} | tick {} | {}",
                if timeline.selected == Some(ordinal) {
                    "Selected: "
                } else {
                    ""
                },
                record.tick.0,
                record.human_text,
            ),
            PanelAction::SelectNarrative {
                session_id: session,
                ordinal,
            },
            true,
        ));
    }
    if let Some((ordinal, record)) = timeline
        .selected
        .and_then(|ordinal| timeline.record(ordinal).map(|record| (ordinal, record)))
    {
        rows.push(PanelRow::text(format!(
            "EVENT #{ordinal} | {} | {} | tick {}\n{}",
            record.kind.as_str(),
            record.metric,
            record.tick.0,
            record.human_text,
        )));
        rows.push(PanelRow::text(format!(
            "Recorded window: ticks {}-{} | before {} | after {}\nMagnitude {} | severity {} | subject {}",
            record.window.0,
            record.window.1,
            fixed(record.before, 3),
            fixed(record.after, 3),
            fixed(record.magnitude, 3),
            fixed(f64::from(record.severity), 3),
            record.subject.map_or_else(|| "unspecified".into(), scriptbots_core::narrative::SubjectRef::to_db_string),
        )));
        rows.push(PanelRow::workflow(
            "Review selected event tick in the world",
            LabAction::ReviewEvent(LabEventIdentity {
                session_id: session,
                ordinal,
                tick: record.tick.0,
            }),
            model.submit.is_some() && !model.is_busy(),
        ));
        rows.push(PanelRow::text(if model.submit.is_some() {
            "Review needs a retained checkpoint at or before this event and complete recorded inputs. Save checkpoints from the live world before events you may want to revisit."
        } else {
            "Historical reconstruction is unavailable in this launch; the canonical event evidence remains readable."
        }));
    }
}

fn experiment_rows(model: &LabWorkflows, idle: bool, rows: &mut Vec<PanelRow>) {
    if model.selected_experiment.is_some() {
        rows.push(PanelRow::workflow(
            "All experiments",
            LabAction::AllExperiments,
            true,
        ));
        let Some(batch) = &model.batch else {
            rows.push(PanelRow::text(
                "Experiment details have not been observed. Refresh to retry.",
            ));
            return;
        };
        let summary = &batch.summary;
        rows.push(PanelRow::text(format!(
            "{} | {}\n{} / {} completed; {} failed\nPlan {} | observation generation {}",
            summary.experiment_id,
            summary.status,
            summary.completed_runs,
            summary.total_runs,
            summary.failed_runs,
            batch.plan_digest,
            batch.generation,
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
            idle && !matches!(
                summary.status.as_str(),
                "running" | "pending" | "cancelling"
            ) && summary.completed_runs + summary.failed_runs < summary.total_runs,
        ));
        for variant in batch
            .variants
            .iter()
            .skip(model.variant_offset)
            .take(VARIANTS_PER_PAGE)
        {
            rows.push(PanelRow::text(format!(
                "{} | {} | {} / {} completed, {} running, {} failed",
                variant.variant_id,
                variant.brain_family,
                variant.completed_runs,
                variant.total_runs,
                variant.running_runs,
                variant.failed_runs,
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
        rows.push(PanelRow::text("Compare endpoint"));
        for metric in LabMetric::ALL {
            rows.push(PanelRow::workflow(
                format!(
                    "{}{} ({})",
                    if model.metric == metric {
                        "Selected: "
                    } else {
                        ""
                    },
                    metric.label(),
                    metric.unit(),
                ),
                LabAction::Metric(metric),
                true,
            ));
        }
        rows.push(PanelRow::text(format!(
            "{} comparison ({}): {} minus {}. Complete matched seed cohorts are required.",
            model.metric.label(),
            model.metric.unit(),
            model.challenger.as_deref().unwrap_or("choose challenger"),
            model.baseline.as_deref().unwrap_or("choose baseline"),
        )));
        rows.push(PanelRow::workflow(
            format!("Compare verified {} endpoints", model.metric.label()),
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
                    item.experiment_id,
                    item.status,
                    item.completed_runs,
                    item.total_runs,
                    item.failed_runs,
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

fn checkpoint_rows(
    model: &LabWorkflows,
    presentation: &LabPresentation,
    idle: bool,
    rows: &mut Vec<PanelRow>,
) {
    let saving = model.latest_checkpoint_save.as_ref().is_some_and(|save| {
        presentation.source == Some(save.session_id)
            && matches!(&save.state, LabCheckpointSaveState::Pending)
            && model.is_busy()
    });
    let label = if saving {
        "Saving checkpoint..."
    } else if model.will_retry_checkpoint_save() {
        "Retry previous checkpoint save"
    } else {
        "Save live checkpoint"
    };
    rows.push(PanelRow::workflow(
        label,
        LabAction::CreateCheckpoint,
        presentation.checkpoint_source_is_live() && model.can_create_checkpoint(),
    ));
    rows.push(PanelRow::text(
        "A new save records the live world. Its durable receipt identifies the exact saved tick.",
    ));
    if !presentation.checkpoint_source_is_live() {
        rows.push(PanelRow::text(
            "Return to live before saving or retrying a checkpoint.",
        ));
    }
    if let Some(save) = &model.latest_checkpoint_save {
        let current_source = presentation.source == Some(save.session_id);
        if !current_source {
            rows.push(PanelRow::text(
                "Previous source save: this receipt or unconfirmed request belongs to an earlier world source.",
            ));
        }
        match &save.state {
            LabCheckpointSaveState::Pending => rows.push(PanelRow::text(
                "Saving checkpoint - awaiting a durable receipt.",
            )),
            LabCheckpointSaveState::Retryable { reason } => {
                rows.push(PanelRow::text(format!("Save not confirmed: {reason}")));
                rows.push(PanelRow::text(if current_source {
                    "Retry previous checkpoint save uses the same request identity."
                } else {
                    "Saving this live source starts a new request after the previous worker finishes."
                }));
            }
            LabCheckpointSaveState::Saved(checkpoint) => {
                rows.push(PanelRow::text(format!(
                    "Saved checkpoint {} | tick {} | {} agents | {} bytes",
                    checkpoint.checkpoint_id,
                    checkpoint.tick,
                    checkpoint.agent_count,
                    checkpoint.byte_size,
                )));
                rows.push(PanelRow::text(format!(
                    "Saved world digest: {}",
                    checkpoint.world_digest,
                )));
            }
        }
        rows.push(PanelRow::text(format!("Save request: {}", save.request_id)));
    }
    rows.push(PanelRow::text(
        "Review reconstructs the selected saved world. The live owner continues separately.",
    ));
    rows.push(PanelRow::workflow(
        "Review selected checkpoint in world",
        LabAction::ReviewCheckpoint,
        idle && model.selected_checkpoint_metadata().is_some(),
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
                checkpoint.checkpoint_id,
                checkpoint.tick,
                checkpoint.agent_count,
                checkpoint.byte_size,
            ),
            LabAction::SelectCheckpoint(checkpoint.checkpoint_id.clone()),
            true,
        ));
    }
    if let Some(checkpoint) = model.selected_checkpoint_metadata() {
        rows.push(PanelRow::text(format!(
            "Saved world digest: {}",
            checkpoint.world_digest
        )));
    }
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

fn retained_focus(previous: &[PanelRow], next: &[PanelRow], focus: Option<usize>) -> Option<usize> {
    let action = previous.get(focus?)?.action.as_ref()?;
    next.iter()
        .position(|row| row.enabled && row.action.as_ref() == Some(action))
}

/// Update the fixed widget pool. Stable observations do not mark text changed.
pub(super) fn refresh_panel(
    model: NonSend<LabWorkflows>,
    mut presentation: ResMut<LabPresentation>,
    ui: Option<ResMut<LabUi>>,
    mut nodes: Query<&mut Node>,
    roots: Query<Entity, With<LabPanelRoot>>,
    mut rows: Query<(
        &mut LabRow,
        &Interaction,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
    mut texts: Query<&mut Text>,
    banners: Query<Entity, With<LabReviewBanner>>,
    headings: Query<Entity, With<LabWorldHeading>>,
    mut wheels: MessageReader<MouseWheel>,
    keys: Res<ButtonInput<KeyCode>>,
    mut bodies: Query<(&ComputedNode, &mut ScrollPosition), With<LabPanelBody>>,
    sizes: Query<&ComputedNode, With<LabRow>>,
) {
    let Some(mut ui) = ui else {
        return;
    };
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
                presentation
                    .review
                    .as_ref()
                    .map_or("SCRIPTBOTS  /  LIVE WORLD", ReviewFrame::heading)
                    .into(),
            );
        }
    }
    for entity in &banners {
        if let Ok(mut node) = nodes.get_mut(entity) {
            let display = if presentation.review.is_some() {
                Display::Flex
            } else {
                Display::None
            };
            if node.display != display {
                node.display = display;
            }
        }
        if let (Some(review), Ok(mut text)) = (&presentation.review, texts.get_mut(entity)) {
            set_hud_text_if_changed(
                &mut text,
                format!(
                    "{} | Live world continues separately. Click here or press Escape to return live.",
                    review.description(),
                ),
            );
        }
    }
    let desired = if open {
        panel_rows(&model, &presentation)
    } else {
        Vec::new()
    };
    if ui.shown != desired {
        for (index, (entity, label)) in ui.rows.iter().copied().enumerate() {
            let value = desired.get(index);
            if ui.shown.get(index) == value {
                continue;
            }
            if let Ok(mut node) = nodes.get_mut(entity) {
                node.display = if value.is_some() {
                    Display::Flex
                } else {
                    Display::None
                };
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
        let color = BorderColor::all(if focused {
            Color::srgb(0.75, 0.91, 0.65)
        } else {
            Color::NONE
        });
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
        LAB_MAX_VARIANTS, LabCheckpoint, LabCheckpointPage, LabCheckpointSave, LabComparison,
        LabExperiment, LabExperimentSummary, LabVariantProgress, LabWorkflowRequest,
        LabWorkflowResponse,
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

    fn narrative_record(tick: u64) -> scriptbots_core::narrative::EventRecord {
        scriptbots_core::narrative::EventRecord {
            schema_version: 1,
            tick: scriptbots_core::Tick(tick),
            kind: scriptbots_core::narrative::EventKind::PopulationCrash,
            severity: 0.7,
            magnitude: 70.0,
            window: (tick.saturating_sub(10), tick),
            metric: "population".into(),
            before: 100.0,
            after: 30.0,
            score: 9.5,
            subject: None,
            human_text: format!("t={tick} population fell 70% (100 -> 30)"),
        }
    }

    #[test]
    fn narrative_projection_shares_the_canonical_ring_and_observes_same_tick_changes() {
        let original = saved_snapshot();
        let mut snapshot = (*original).clone();
        let mut other_event = narrative_record(0);
        other_event.kind = scriptbots_core::narrative::EventKind::EnergyRecovery;
        other_event.metric = "mean_energy".into();
        other_event.human_text = "the exact second canonical event at this same tick".into();
        snapshot.narrative_events = Arc::new(vec![narrative_record(0), other_event]);
        snapshot.narrative_dropped_events = 40;
        let projected = WorldSnapshot::from_snapshot(&snapshot).unwrap();
        assert!(Arc::ptr_eq(
            &projected.narrative_events,
            &snapshot.narrative_events
        ));
        assert_eq!(projected.narrative_dropped_events, 40);
        assert!(!projected.same_render_content(&WorldSnapshot::from_snapshot(&original).unwrap()));
        let mut timeline = NarrativeTimeline::default();
        timeline.observe(&projected);
        assert_eq!(
            timeline
                .page()
                .iter()
                .map(|(ordinal, _)| *ordinal)
                .collect::<Vec<_>>(),
            [42, 41]
        );
        assert_eq!(timeline.record(42), snapshot.narrative_events.get(1));
        assert!(
            projected.events.is_empty(),
            "tick-summary activity is a separate feed"
        );
    }

    #[test]
    fn narrative_pages_and_selection_survive_new_events_and_refuse_evicted_identity() {
        let mut frame = live_frame(100);
        frame.narrative_events = Arc::new((1..=30).map(narrative_record).collect());
        let mut timeline = NarrativeTimeline::default();
        timeline.observe(&frame);
        assert_eq!(timeline.page().len(), NARRATIVE_PAGE_SIZE);
        assert_eq!(timeline.page().first().unwrap().0, 30);
        timeline.older();
        timeline.select(12);
        assert_eq!(timeline.page().first().unwrap().0, 18);
        assert_eq!(timeline.page().last().unwrap().0, 7);

        frame.narrative_events = Arc::new((1..=35).map(narrative_record).collect());
        timeline.observe(&frame);
        assert_eq!(
            timeline.page().first().unwrap().0,
            18,
            "new events must not move the page"
        );
        assert_eq!(timeline.selected, Some(12));
        timeline.older();
        assert_eq!(timeline.page().last().unwrap().0, 1);
        assert!(!timeline.can_show_older());
        timeline.newer();
        assert_eq!(timeline.page().first().unwrap().0, 18);
        timeline.newer();
        timeline.newer();
        assert_eq!(timeline.page().first().unwrap().0, 35);
        assert!(timeline.newest_on_page.is_none());

        timeline.newest_on_page = Some(18);
        frame.narrative_events = Arc::new((21..=35).map(narrative_record).collect());
        frame.narrative_dropped_events = 20;
        timeline.observe(&frame);
        assert_eq!(
            timeline.selected, None,
            "an evicted ordinal must not select a replacement"
        );
        assert_eq!(timeline.page().last().unwrap().0, 21);
        assert!(
            timeline
                .notice
                .as_ref()
                .unwrap()
                .contains("oldest available")
        );
        timeline.select(12);
        assert_eq!(timeline.selected, None);
        assert!(Arc::ptr_eq(&timeline.records, &frame.narrative_events));
    }

    #[test]
    fn narrative_owner_change_clears_selection_even_when_ticks_and_ordinals_repeat() {
        let mut presentation = LabPresentation::default();
        let mut state = SnapshotState::default();
        let mut motion = AgentMotionSettings { enabled: true };
        let mut frame = live_frame(100);
        frame.narrative_events = Arc::new(vec![narrative_record(50)]);
        presentation.receive_live(frame.clone(), &mut state, &mut motion);
        presentation.narrative.select(1);
        presentation.narrative.newest_on_page = Some(1);
        frame.owner_source.as_mut().unwrap().0.session_id = HostSessionId::new(99);
        frame.narrative_events = Arc::new(vec![narrative_record(60)]);
        presentation.receive_live(frame, &mut state, &mut motion);
        assert_eq!(presentation.narrative.selected, None);
        assert!(presentation.narrative.newest_on_page.is_none());
        assert_eq!(presentation.narrative.record(1).unwrap().tick.0, 60);
        presentation.select_narrative(HostSessionId::new(71), 1);
        assert_eq!(
            presentation.narrative.selected, None,
            "a previous owner's button must be inert"
        );
        presentation.select_narrative(HostSessionId::new(99), 1);
        assert_eq!(presentation.narrative.selected, Some(1));
    }

    #[test]
    fn narrative_panel_is_bounded_and_readable_without_application_services() {
        let mut presentation = LabPresentation {
            panel: LabPanel::Narrative,
            ..default()
        };
        let mut state = SnapshotState::default();
        let mut motion = AgentMotionSettings { enabled: true };
        let mut frame = live_frame(100);
        frame.narrative_events = Arc::new((21..=100).map(narrative_record).collect());
        frame.narrative_dropped_events = 20;
        presentation.receive_live(frame, &mut state, &mut motion);
        presentation.narrative.select(100);
        let rows = panel_rows(&consumer(None), &presentation);
        assert!(rows.len() <= MAX_PANEL_ROWS);
        assert_eq!(
            rows.iter()
                .filter(|row| matches!(&row.action, Some(PanelAction::SelectNarrative { .. })))
                .count(),
            NARRATIVE_PAGE_SIZE
        );
        assert!(
            rows.iter()
                .any(|row| row.text.contains("t=100 population fell 70% (100 -> 30)"))
        );
        assert!(
            rows.iter()
                .any(|row| row.text.contains("Retention gap: 20"))
        );
        assert!(
            rows.iter()
                .any(|row| row.text.contains("Recorded window: ticks 90-100"))
        );
        assert!(
            !rows
                .iter()
                .any(|row| row.text.contains("services are unavailable"))
        );
        let expected = LabAction::ReviewEvent(LabEventIdentity {
            session_id: HostSessionId::new(71),
            ordinal: 100,
            tick: 100,
        });
        let review = rows
            .iter()
            .find(|row| row.action == Some(PanelAction::Workflow(expected.clone())))
            .expect("selected evidence has a source-bound review action");
        assert!(!review.enabled, "reconstruction needs application services");
        let services = consumer(Some(Arc::new(
            |_| Err("not submitted while drawing".into()),
        )));
        assert!(panel_rows(&services, &presentation).iter().any(|row| {
            row.action == Some(PanelAction::Workflow(expected.clone())) && row.enabled
        }));

        presentation.narrative.records = Arc::new(Vec::new());
        let rows = panel_rows(&consumer(None), &presentation);
        assert!(
            !rows
                .iter()
                .any(|row| matches!(&row.action, Some(PanelAction::SelectNarrative { .. })))
        );
        assert!(
            rows.iter()
                .any(|row| row.text.contains("No narrative events are retained"))
        );
    }

    #[test]
    fn narrative_shortcut_opens_the_existing_modal_and_captures_world_controls() {
        let mut app = App::new();
        let mut keys = ButtonInput::<KeyCode>::default();
        keys.press(KeyCode::F8);
        app.insert_resource(keys)
            .init_resource::<LabPresentation>()
            .insert_resource(AgentMotionSettings { enabled: true })
            .init_resource::<SnapshotState>()
            .init_resource::<CameraRig>()
            .insert_non_send_resource(consumer(None))
            .add_systems(Update, update);
        app.update();
        let presentation = app.world().resource::<LabPresentation>();
        assert_eq!(presentation.panel, LabPanel::Narrative);
        assert!(presentation.keyboard_captured);
        assert!(!presentation.world_controls_allowed());
        assert!(!presentation.camera_controls_allowed());
    }

    #[test]
    fn checkpoint_review_keeps_latest_live_and_invalidates_equal_revision_caches() {
        let mut presentation = LabPresentation::default();
        let mut state = SnapshotState::default();
        let mut motion = AgentMotionSettings { enabled: true };
        presentation.receive_live(live_frame(12), &mut state, &mut motion);
        state.last_applied_revision = 1;
        state.last_applied_palette = Some(ColorPaletteMode::Natural);
        assert!(presentation.show_checkpoint(
            "saved-1".into(),
            &saved_snapshot(),
            &mut state,
            &mut motion
        ));
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

    fn selected_event_review() -> LabEventReview {
        let mut snapshot = (*saved_snapshot()).clone();
        snapshot.session_id = HostSessionId::new(71);
        LabEventReview {
            event: LabEventIdentity {
                session_id: snapshot.session_id,
                ordinal: 1,
                tick: snapshot.world.tick,
            },
            record: narrative_record(snapshot.world.tick),
            checkpoint_id: "event-source-checkpoint".into(),
            base_tick: 0,
            snapshot: Arc::new(snapshot),
        }
    }

    #[test]
    fn event_review_preserves_source_provenance_and_returns_to_the_newest_live_world() {
        let mut presentation = LabPresentation::default();
        let mut state = SnapshotState::default();
        let mut motion = AgentMotionSettings { enabled: true };
        let mut live = live_frame(12);
        live.narrative_events = Arc::new(vec![narrative_record(0)]);
        presentation.receive_live(live.clone(), &mut state, &mut motion);
        presentation.narrative.select(1);
        assert!(presentation.show_event_review(selected_event_review(), &mut state, &mut motion));
        let review = presentation.review.as_ref().unwrap();
        assert_eq!(review.heading(), "SCRIPTBOTS  /  EVENT REVIEW");
        assert!(review.description().contains("EVENT #1 | TICK 0"));
        assert!(
            review
                .description()
                .contains("SOURCE CHECKPOINT event-source-checkpoint AT TICK 0")
        );
        assert!(review.description().contains("SESSION 71"));
        let historical = state.latest.as_ref().unwrap();
        assert_eq!(historical.tick, 0);
        assert!(historical.owner_source.is_none() && historical.motion.is_none());
        assert!(historical.paused && historical.reduced_motion);
        assert!(!motion.enabled && !presentation.world_controls_allowed());

        live.tick = 29;
        live.revision = 29;
        presentation.receive_live(live, &mut state, &mut motion);
        assert_eq!(state.latest.as_ref().unwrap().tick, 0);
        presentation.return_live(&mut state, &mut motion);
        assert_eq!(state.latest.as_ref().unwrap().tick, 29);
        assert!(state.latest.as_ref().unwrap().owner_source.is_some());
        assert!(presentation.review.is_none() && presentation.live.is_none());
        assert!(motion.enabled && presentation.world_controls_allowed());
    }

    #[test]
    fn event_review_refuses_changed_evidence_or_incomplete_visuals_without_replacing_live() {
        let mut presentation = LabPresentation::default();
        let mut state = SnapshotState::default();
        let mut motion = AgentMotionSettings { enabled: true };
        let mut live = live_frame(12);
        live.narrative_events = Arc::new(vec![narrative_record(0)]);
        presentation.receive_live(live, &mut state, &mut motion);
        presentation.narrative.select(1);
        for mismatch in 0..6 {
            let mut review = selected_event_review();
            match mismatch {
                0 => review.event.ordinal += 1,
                1 => review.record.human_text = "different canonical evidence".into(),
                2 => Arc::make_mut(&mut review.snapshot).session_id = HostSessionId::new(99),
                3 => Arc::make_mut(&mut review.snapshot).world.tick += 1,
                4 => review.base_tick = review.event.tick + 1,
                5 => Arc::make_mut(&mut Arc::make_mut(&mut review.snapshot).layers.food)
                    .cells
                    .clear(),
                _ => unreachable!(),
            }
            assert!(
                !presentation.show_event_review(review, &mut state, &mut motion),
                "mismatch {mismatch} must be refused"
            );
            assert_eq!(state.latest.as_ref().unwrap().tick, 12);
            assert!(presentation.review.is_none() && presentation.live.is_none());
            assert!(motion.enabled);
        }
    }

    #[test]
    fn new_owner_session_leaves_saved_review_and_clears_old_retention() {
        let mut presentation = LabPresentation::default();
        let mut state = SnapshotState::default();
        let mut motion = AgentMotionSettings { enabled: true };
        presentation.receive_live(live_frame(12), &mut state, &mut motion);
        assert!(presentation.show_checkpoint(
            "saved-1".into(),
            &saved_snapshot(),
            &mut state,
            &mut motion
        ));
        presentation.source_changed = false;
        let mut reset = live_frame(1);
        reset.owner_source.as_mut().unwrap().0.session_id = HostSessionId::new(72);
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
            variants: (0..LAB_MAX_VARIANTS)
                .map(|index| LabVariantProgress {
                    variant_id: format!("variant-{index}"),
                    brain_family: "mlp".into(),
                    total_runs: 1,
                    completed_runs: 1,
                    failed_runs: 0,
                    running_runs: 0,
                })
                .collect(),
            error_reason: None,
        });
        let presentation = LabPresentation {
            panel: LabPanel::Experiments,
            source: Some(HostSessionId::new(1)),
            ..default()
        };
        let last_variant = format!("variant-{} |", LAB_MAX_VARIANTS - 1);
        let first = panel_rows(&model, &presentation);
        assert!(first.iter().any(|row| row.text.starts_with("variant-0 |")));
        assert!(!first.iter().any(|row| row.text.starts_with(&last_variant)));
        model.dispatch(LabAction::VariantsPage(
            LAB_MAX_VARIANTS - VARIANTS_PER_PAGE,
        ));
        let last = panel_rows(&model, &presentation);
        assert!(last.iter().any(|row| row.text.starts_with(&last_variant)));
        assert!(!last.iter().any(|row| row.text.starts_with("variant-0 |")));
        assert!(first.len() <= MAX_PANEL_ROWS && last.len() <= MAX_PANEL_ROWS);
    }

    #[test]
    fn metric_buttons_and_keyboard_compare_use_the_selected_scientific_endpoint() {
        let mut presentation = LabPresentation {
            panel: LabPanel::Experiments,
            ..default()
        };
        let mut state = SnapshotState::default();
        let mut motion = AgentMotionSettings { enabled: true };
        presentation.receive_live(live_frame(12), &mut state, &mut motion);
        presentation.source_changed = false;
        let session = presentation.source.unwrap();
        let (submitted, requests) = std::sync::mpsc::channel();
        let mut model = consumer(Some(Arc::new(move |request| {
            let LabWorkflowRequest::Compare {
                experiment_id,
                baseline_variant,
                challenger_variant,
                metric,
            } = request
            else {
                panic!("the complete fixture needs only a comparison request");
            };
            submitted.send(metric).unwrap();
            Ok(LabWorkflowResponse::Comparison(LabComparison {
                experiment_id,
                baseline_variant,
                challenger_variant,
                metric,
                endpoint_tick: 12,
                matched_seeds: vec![17],
                mean_difference: 0.125,
                ci_95: None,
                standardized_effect: None,
                p_value: None,
                adjusted_p_value: None,
                descriptive_only: true,
                underpowered_reason: Some("one independent seed pair".into()),
                undefined_effect_reason: Some("one observation".into()),
                plan_digest: "metric-plan".into(),
                status_generation: 1,
            }))
        })));
        model.synchronize(session, LabPanel::Experiments);
        model.selected_experiment = Some("metric-cohort".into());
        model.baseline = Some("baseline".into());
        model.challenger = Some("challenger".into());
        model.batch = Some(LabExperiment {
            summary: LabExperimentSummary {
                experiment_id: "metric-cohort".into(),
                status: "completed".into(),
                total_runs: 2,
                completed_runs: 2,
                failed_runs: 0,
            },
            plan_digest: "metric-plan".into(),
            generation: 1,
            variants: ["baseline", "challenger"]
                .into_iter()
                .map(|id| LabVariantProgress {
                    variant_id: id.into(),
                    brain_family: "mlp".into(),
                    total_runs: 1,
                    completed_runs: 1,
                    failed_runs: 0,
                    running_runs: 0,
                })
                .collect(),
            error_reason: None,
        });
        let rows = panel_rows(&model, &presentation);
        assert_eq!(
            rows.iter()
                .filter(|row| {
                    matches!(
                        &row.action,
                        Some(PanelAction::Workflow(LabAction::Metric(_)))
                    )
                })
                .count(),
            LabMetric::ALL.len()
        );
        let energy_action = PanelAction::Workflow(LabAction::Metric(LabMetric::AverageEnergy));
        let energy = rows
            .iter()
            .find(|row| row.action.as_ref() == Some(&energy_action))
            .unwrap();
        assert!(energy.enabled);
        assert!(energy.text.contains(LabMetric::AverageEnergy.unit()));
        let mut app = App::new();
        app.insert_non_send_resource(model)
            .insert_resource(presentation)
            .insert_resource(state)
            .insert_resource(motion)
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<CameraRig>()
            .add_systems(Update, update);
        app.world_mut().spawn((
            LabRow {
                index: 0,
                action: energy.action.clone(),
                enabled: energy.enabled,
            },
            Interaction::Pressed,
        ));
        app.update();
        let model = app.world().non_send_resource::<LabWorkflows>();
        assert_eq!(model.metric, LabMetric::AverageEnergy);
        let rows = panel_rows(model, app.world().resource::<LabPresentation>());
        assert!(rows.iter().any(|row| {
            row.text.starts_with("Selected: ") && row.action.as_ref() == Some(&energy_action)
        }));
        let compare = rows
            .iter()
            .position(|row| {
                row.enabled && row.action == Some(PanelAction::Workflow(LabAction::Compare))
            })
            .expect("selected endpoint has an enabled compare control");
        app.insert_resource(LabUi {
            rows: Vec::new(),
            shown: rows,
        });
        app.world_mut().resource_mut::<LabPresentation>().focus = Some(compare);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Enter);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        assert_eq!(
            requests
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap(),
            LabMetric::AverageEnergy
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            app.update();
            let model = app.world().non_send_resource::<LabWorkflows>();
            if !model.is_busy() {
                let comparison = model.comparison.as_ref().expect("matching endpoint result");
                assert_eq!(comparison.metric, LabMetric::AverageEnergy);
                assert_eq!(comparison.mean_difference, 0.125);
                assert!(
                    panel_rows(model, app.world().resource::<LabPresentation>())
                        .iter()
                        .any(|row| row.text.contains(LabMetric::AverageEnergy.unit()))
                );
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "comparison worker did not finish"
            );
            std::thread::yield_now();
        }
        assert_eq!(
            app.world()
                .resource::<SnapshotState>()
                .latest
                .as_ref()
                .unwrap()
                .tick,
            12,
            "comparison input must not advance the live presentation"
        );
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
        let replaced = vec![previous[0].clone(), previous[2].clone()];
        assert_eq!(retained_focus(&previous, &replaced, Some(1)), None);
        let disabled = vec![
            previous[0].clone(),
            PanelRow::workflow("refresh", LabAction::Refresh, false),
        ];
        assert_eq!(retained_focus(&previous, &disabled, Some(1)), None);
    }

    fn empty_checkpoint_page() -> LabCheckpointPage {
        LabCheckpointPage {
            items: Vec::new(),
            total: 0,
            offset: 0,
            next_cursor: None,
            retained_bytes: 0,
            max_bytes: 1_000,
        }
    }

    #[test]
    fn stale_save_button_cannot_capture_a_reviewed_or_unidentified_world() {
        let mut presentation = LabPresentation {
            panel: LabPanel::Checkpoints,
            ..default()
        };
        let mut state = SnapshotState::default();
        let mut motion = AgentMotionSettings { enabled: true };
        presentation.receive_live(live_frame(12), &mut state, &mut motion);
        let session = presentation.source.unwrap();
        assert!(presentation.show_checkpoint(
            "saved-1".into(),
            &saved_snapshot(),
            &mut state,
            &mut motion
        ));
        presentation.source_changed = false;

        let (submitted, requests) = std::sync::mpsc::channel();
        let mut model = consumer(Some(Arc::new(move |request| {
            let LabWorkflowRequest::CreateCheckpoint {
                request_id,
                session_id,
            } = request
            else {
                panic!("this fixture already has its checkpoint catalog");
            };
            submitted.send((request_id, session_id)).unwrap();
            Err("test receipt is unconfirmed".into())
        })));
        model.synchronize(session, LabPanel::Checkpoints);
        model.checkpoints = Some(empty_checkpoint_page());
        let save_row = panel_rows(&model, &presentation)
            .into_iter()
            .find(|row| row.action == Some(PanelAction::Workflow(LabAction::CreateCheckpoint)))
            .expect("visible save control");
        assert!(
            !save_row.enabled,
            "historical display must disable creation"
        );

        let mut app = App::new();
        app.insert_non_send_resource(model)
            .insert_resource(presentation)
            .insert_resource(state)
            .insert_resource(motion)
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<CameraRig>()
            .add_systems(Update, update);
        // Deliberately preserve an enabled row from the earlier live paint.
        let button = app
            .world_mut()
            .spawn((
                LabRow {
                    index: 0,
                    action: save_row.action,
                    enabled: true,
                },
                Interaction::Pressed,
            ))
            .id();
        app.update();
        assert!(
            app.world()
                .non_send_resource::<LabWorkflows>()
                .latest_checkpoint_save
                .is_none()
        );
        assert!(requests.try_recv().is_err());

        let mut presentation = app
            .world_mut()
            .remove_resource::<LabPresentation>()
            .unwrap();
        let mut state = app.world_mut().remove_resource::<SnapshotState>().unwrap();
        let mut motion = app
            .world_mut()
            .remove_resource::<AgentMotionSettings>()
            .unwrap();
        presentation.return_live(&mut state, &mut motion);
        // The model still remembers the previous owner. A missing current
        // presentation identity must nevertheless refuse a stale enabled row.
        presentation.source = None;
        app.insert_resource(presentation)
            .insert_resource(state)
            .insert_resource(motion);
        app.world_mut()
            .entity_mut(button)
            .insert(Interaction::Pressed);
        app.update();
        assert!(
            app.world()
                .non_send_resource::<LabWorkflows>()
                .latest_checkpoint_save
                .is_none()
        );
        assert!(requests.try_recv().is_err());

        app.world_mut().resource_mut::<LabPresentation>().source = Some(session);
        app.world_mut()
            .entity_mut(button)
            .insert(Interaction::Pressed);
        app.update();
        let (_, requested_session) = requests
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("live save submitted");
        assert_eq!(requested_session, session);
        assert!(
            app.world()
                .non_send_resource::<LabWorkflows>()
                .latest_checkpoint_save
                .is_some()
        );
        assert_eq!(
            app.world()
                .resource::<SnapshotState>()
                .latest
                .as_ref()
                .unwrap()
                .tick,
            12
        );
    }

    #[test]
    fn scientific_epoch_progress_preserves_review_and_unconfirmed_save_identity() {
        let mut presentation = LabPresentation {
            panel: LabPanel::Checkpoints,
            ..default()
        };
        let mut state = SnapshotState::default();
        let mut motion = AgentMotionSettings { enabled: true };
        presentation.receive_live(live_frame(9_999), &mut state, &mut motion);
        let session = presentation.source.unwrap();
        presentation.source_changed = false;
        assert!(presentation.show_checkpoint(
            "saved-1".into(),
            &saved_snapshot(),
            &mut state,
            &mut motion
        ));
        let mut next_epoch = live_frame(10_000);
        next_epoch.owner_source.as_mut().unwrap().1 = 1;
        presentation.receive_live(next_epoch, &mut state, &mut motion);
        assert!(
            presentation.reviewing(),
            "normal epoch progress must not leave saved review"
        );
        assert!(!presentation.source_changed);
        assert_eq!(state.latest.as_ref().unwrap().tick, 0);
        assert_eq!(presentation.live.as_ref().unwrap().tick, 10_000);

        let (submitted, requests) = std::sync::mpsc::channel();
        let mut model = consumer(Some(Arc::new(move |request| {
            let LabWorkflowRequest::CreateCheckpoint {
                request_id,
                session_id,
            } = request
            else {
                panic!("retry should be the only request");
            };
            submitted.send((request_id, session_id)).unwrap();
            Err("test receipt is unconfirmed".into())
        })));
        model.synchronize(session, LabPanel::Checkpoints);
        model.checkpoints = Some(empty_checkpoint_page());
        model.latest_checkpoint_save = Some(LabCheckpointSave {
            request_id: "save-before-epoch".into(),
            session_id: session,
            state: LabCheckpointSaveState::Retryable {
                reason: "receipt timed out".into(),
            },
        });
        presentation.return_live(&mut state, &mut motion);
        let mut app = App::new();
        app.insert_non_send_resource(model)
            .insert_resource(presentation)
            .insert_resource(state)
            .insert_resource(motion)
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<CameraRig>()
            .add_systems(Update, update);
        app.world_mut().spawn((
            LabRow {
                index: 0,
                action: Some(PanelAction::Workflow(LabAction::CreateCheckpoint)),
                enabled: true,
            },
            Interaction::Pressed,
        ));
        app.update();
        let (request_id, requested_session) = requests
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("same-source retry");
        assert_eq!(request_id, "save-before-epoch");
        assert_eq!(requested_session, session);
        assert_eq!(
            app.world()
                .resource::<SnapshotState>()
                .latest
                .as_ref()
                .unwrap()
                .tick,
            10_000
        );
    }

    #[test]
    fn checkpoint_save_rows_distinguish_retry_receipt_and_new_source() {
        let session = HostSessionId::new(71);
        let mut model = consumer(Some(Arc::new(|_| Err("unused".into()))));
        model.synchronize(session, LabPanel::Checkpoints);
        model.checkpoints = Some(empty_checkpoint_page());
        model.latest_checkpoint_save = Some(LabCheckpointSave {
            request_id: "request-1".into(),
            session_id: session,
            state: LabCheckpointSaveState::Retryable {
                reason: "receipt timed out".into(),
            },
        });
        let mut presentation = LabPresentation {
            panel: LabPanel::Checkpoints,
            source: Some(session),
            ..default()
        };
        let rows = panel_rows(&model, &presentation);
        assert!(
            rows.iter()
                .any(|row| row.enabled && row.text == "Retry previous checkpoint save")
        );
        assert!(
            rows.iter()
                .any(|row| row.text.contains("Save not confirmed"))
        );
        assert!(
            rows.iter()
                .any(|row| row.text.contains("same request identity"))
        );

        let next_session = HostSessionId::new(72);
        presentation.source = Some(next_session);
        model.synchronize(next_session, LabPanel::Checkpoints);
        let rows = panel_rows(&model, &presentation);
        assert!(
            rows.iter()
                .any(|row| row.enabled && row.text == "Save live checkpoint")
        );
        assert!(
            rows.iter()
                .any(|row| row.text.starts_with("Previous source save:"))
        );
        assert!(
            !rows
                .iter()
                .any(|row| row.text == "Retry previous checkpoint save")
        );

        model.latest_checkpoint_save.as_mut().unwrap().state =
            LabCheckpointSaveState::Saved(LabCheckpoint {
                checkpoint_id: "saved-cut".into(),
                tick: 37,
                agent_count: 9,
                byte_size: 640,
                world_digest: "a".repeat(16),
            });
        let rows = panel_rows(&model, &presentation);
        assert!(
            rows.iter().any(
                |row| row.text == "Saved checkpoint saved-cut | tick 37 | 9 agents | 640 bytes"
            )
        );
        assert!(
            rows.iter()
                .any(|row| row.text == format!("Saved world digest: {}", "a".repeat(16)))
        );
        assert!(rows.len() <= MAX_PANEL_ROWS);

        presentation.source = Some(session);
        model.synchronize(session, LabPanel::Checkpoints);
        model.selected_checkpoint = Some("saved-cut".into());
        model.checkpoints = None;
        let rows = panel_rows(&model, &presentation);
        assert!(
            rows.iter().any(|row| row.enabled
                && row.action == Some(PanelAction::Workflow(LabAction::ReviewCheckpoint))),
            "a verified saved cut must remain reviewable without a catalog page"
        );
    }

    #[test]
    fn modal_escape_is_consumed_and_optional_capture_context_stays_live() {
        let mut app = App::new();
        let mut presentation = LabPresentation {
            panel: LabPanel::Experiments,
            ..default()
        };
        presentation.source = Some(HostSessionId::new(1));
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
        assert_eq!(
            app.world_mut().query::<&LabRow>().iter(app.world()).count(),
            MAX_PANEL_ROWS
        );
    }
}
