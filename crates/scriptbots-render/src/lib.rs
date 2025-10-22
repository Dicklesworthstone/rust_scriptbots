//! GPUI rendering layer for ScriptBots.

use gpui::{
    App, Application, Background, Bounds, Context, Div, KeyDownEvent, Keystroke, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, PathBuilder, Pixels, Point, Rgba, ScrollDelta,
    ScrollWheelEvent, SharedString, StyleRefinement, Window, WindowBounds, WindowOptions, canvas,
    div, fill, point, prelude::*, px, rgb, size,
};
use rand::Rng;
use scriptbots_core::{
    AgentColumns, AgentData, AgentId, AgentRuntime, ControlCommand, Generation, INPUT_SIZE,
    IndicatorState, MutationRates, OUTPUT_SIZE, Position, ScriptBotsConfig, SelectionState,
    TerrainKind, TerrainLayer, TickSummary, TraitModifiers, Velocity, WorldState,
};
use scriptbots_storage::{MetricReading, Storage};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, VecDeque},
    f32::consts::{FRAC_PI_2, FRAC_PI_4, PI},
    sync::{Arc, Mutex},
    time::Instant,
};

#[cfg(feature = "audio")]
use kira::{
    DefaultBackend,
    frame::Frame,
    manager::{AudioManager, AudioManagerSettings},
    sound::static_sound::{StaticSoundData, StaticSoundSettings},
};

use tracing::{error, info, warn};

fn toroidal_delta(origin: f32, target: f32, extent: f32) -> f32 {
    let mut delta = target - origin;
    let half = extent * 0.5;
    if delta > half {
        delta -= extent;
    } else if delta < -half {
        delta += extent;
    }
    delta
}

/// Launch the ScriptBots GPUI shell with an interactive HUD.
pub fn run_demo(
    world: Arc<Mutex<WorldState>>,
    storage: Option<Arc<Mutex<Storage>>>,
    command_drain: Arc<dyn Fn(&mut WorldState) + Send + Sync + 'static>,
    command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync + 'static>,
) {
    if let Ok(world) = world.lock()
        && let Some(summary) = world.history().last()
    {
        info!(
            tick = summary.tick.0,
            agents = summary.agent_count,
            births = summary.births,
            deaths = summary.deaths,
            avg_energy = summary.average_energy,
            "Launching GPUI shell with latest world snapshot",
        );
    }

    let window_title: SharedString = "ScriptBots HUD".into();
    let title_for_options = window_title.clone();
    let title_for_view = window_title.clone();
    let world_for_view = Arc::clone(&world);
    let drain_for_view = Arc::clone(&command_drain);
    let submit_for_view = Arc::clone(&command_submit);
    let storage_for_view = storage.clone();

    Application::new().run(move |app: &mut App| {
        let bounds = Bounds::centered(None, size(px(1280.0), px(720.0)), app);
        let mut options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            ..Default::default()
        };

        if let Some(titlebar) = options.titlebar.as_mut() {
            titlebar.title = Some(title_for_options.clone());
        }

        let world_handle = Arc::clone(&world_for_view);
        let view_title = title_for_view.clone();

        if let Err(err) = app.open_window(options, move |_window, cx| {
            cx.new(|_| {
                SimulationView::new(
                    Arc::clone(&world_handle),
                    storage_for_view.clone(),
                    view_title.clone(),
                    Arc::clone(&drain_for_view),
                    Arc::clone(&submit_for_view),
                )
            })
        }) {
            error!(error = ?err, "failed to open ScriptBots window");
            return;
        }

        app.activate(true);
    });
}

const MAX_SELECTION_EVENTS: usize = 64;
const SIM_TICK_INTERVAL: f32 = 1.0 / 60.0;
const MAX_SIM_STEPS_PER_FRAME: usize = 240;

struct SimulationView {
    world: Arc<Mutex<WorldState>>,
    storage: Option<Arc<Mutex<Storage>>>,
    title: SharedString,
    command_drain: Arc<dyn Fn(&mut WorldState) + Send + Sync + 'static>,
    command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync + 'static>,
    camera: Arc<Mutex<CameraState>>,
    inspector: Arc<Mutex<InspectorState>>,
    playback: PlaybackState,
    perf: PerfStats,
    last_perf: PerfSnapshot,
    accessibility: AccessibilitySettings,
    debug: DebugOverlayState,
    selection_events: VecDeque<SelectionEvent>,
    controls: SimulationControls,
    sim_accumulator: f32,
    last_sim_instant: Option<Instant>,
    shift_inspect: bool,
    bindings: InputBindings,
    key_capture: Option<CommandAction>,
    settings_panel: SettingsPanelState,
    analytics_cache: Option<HudAnalytics>,
    analytics_tick: Option<u64>,
    #[cfg(feature = "audio")]
    audio: Option<AudioState>,
}

impl SimulationView {
    fn new(
        world: Arc<Mutex<WorldState>>,
        storage: Option<Arc<Mutex<Storage>>>,
        title: SharedString,
        command_drain: Arc<dyn Fn(&mut WorldState) + Send + Sync + 'static>,
        command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync + 'static>,
    ) -> Self {
        let mut inspector_state = InspectorState::default();
        if let Ok(world_guard) = world.lock() {
            let interval = world_guard.config().persistence_interval;
            if interval > 0 {
                inspector_state.persistence_last_enabled = interval;
            }
        }

        Self {
            world,
            storage,
            title,
            command_drain,
            command_submit,
            camera: Arc::new(Mutex::new(CameraState::default())),
            inspector: Arc::new(Mutex::new(inspector_state)),
            playback: PlaybackState::new(240),
            perf: PerfStats::new(240),
            last_perf: PerfSnapshot::default(),
            accessibility: AccessibilitySettings::default(),
            debug: DebugOverlayState::default(),
            selection_events: VecDeque::with_capacity(MAX_SELECTION_EVENTS),
            controls: SimulationControls::default(),
            sim_accumulator: 0.0,
            last_sim_instant: Some(Instant::now()),
            shift_inspect: false,
            bindings: InputBindings::default(),
            settings_panel: SettingsPanelState::default(),
            key_capture: None,
            analytics_cache: None,
            analytics_tick: None,
            #[cfg(feature = "audio")]
            audio: AudioState::new()
                .map_err(|err| {
                    error!(?err, "failed to initialize audio manager");
                    err
                })
                .ok(),
        }
    }

    fn camera_snapshot(&self) -> CameraState {
        self.camera
            .lock()
            .map(|camera| camera.clone())
            .unwrap_or_default()
    }

    fn pump_simulation(&mut self) {
        let now = Instant::now();
        let last = self.last_sim_instant.unwrap_or(now);
        self.last_sim_instant = Some(now);

        if self.controls.paused || self.controls.speed_multiplier <= 0.0 {
            self.sim_accumulator = 0.0;
            return;
        }

        let delta = (now - last).as_secs_f32();
        self.sim_accumulator += delta * self.controls.speed_multiplier;

        let step_interval = SIM_TICK_INTERVAL;
        if self.sim_accumulator < step_interval {
            return;
        }

        let max_accumulator = step_interval * MAX_SIM_STEPS_PER_FRAME as f32;
        if self.sim_accumulator > max_accumulator {
            self.sim_accumulator = max_accumulator;
        }

        let mut steps = (self.sim_accumulator / step_interval).floor() as usize;
        if steps == 0 {
            return;
        }
        if steps > MAX_SIM_STEPS_PER_FRAME {
            steps = MAX_SIM_STEPS_PER_FRAME;
        }

        self.sim_accumulator -= step_interval * steps as f32;

        if let Ok(mut world) = self.world.lock() {
            (self.command_drain.as_ref())(&mut world);
            for _ in 0..steps {
                world.step();
            }
        }
    }

    fn submit_config_update<F>(&self, update: F)
    where
        F: FnOnce(&mut ScriptBotsConfig),
    {
        if let Ok(world) = self.world.lock() {
            let mut new_config = world.config().clone();
            drop(world);
            update(&mut new_config);
            if !(self.command_submit.as_ref())(ControlCommand::UpdateConfig(new_config)) {
                warn!("failed to enqueue config update from renderer");
            }
        } else {
            warn!("failed to acquire world lock for config update");
        }
    }

    fn canvas_to_world(&self, position: Point<Pixels>) -> Option<(f32, f32)> {
        self.camera
            .lock()
            .ok()
            .and_then(|camera| camera.screen_to_world(position))
    }

    fn selection_pick_radius(&self, world: &WorldState) -> f32 {
        (world.config().bot_radius * 3.0).max(24.0)
    }

    fn pick_agent_near(
        &self,
        world: &WorldState,
        point: (f32, f32),
        radius: f32,
    ) -> Option<AgentId> {
        let arena = world.agents();
        let columns = arena.columns();
        let positions = columns.positions();
        let radius_sq = radius * radius;
        let extent_x = world.config().world_width as f32;
        let extent_y = world.config().world_height as f32;
        let mut best: Option<(AgentId, f32)> = None;

        for (idx, agent_id) in arena.iter_handles().enumerate() {
            let pos = positions[idx];
            let dx = toroidal_delta(point.0, pos.x, extent_x);
            let dy = toroidal_delta(point.1, pos.y, extent_y);
            let dist_sq = dx.mul_add(dx, dy * dy);
            if dist_sq <= radius_sq {
                if best.map_or(true, |(_, best_dist)| dist_sq < best_dist) {
                    best = Some((agent_id, dist_sq));
                }
            }
        }

        best.map(|(id, _)| id)
    }

    fn clear_all_selections(&mut self) -> bool {
        let prev_hover = self
            .inspector
            .lock()
            .map(|state| state.hovered_agent)
            .unwrap_or(None);

        let mut changed = false;
        if let Ok(mut world) = self.world.lock() {
            let runtime = world.runtime_mut();
            for entry in runtime.values_mut() {
                if !matches!(entry.selection, SelectionState::None) {
                    entry.selection = SelectionState::None;
                    changed = true;
                }
            }

            if let Some(prev) = prev_hover {
                if let Some(entry) = runtime.get_mut(prev) {
                    if matches!(entry.selection, SelectionState::Hovered) {
                        entry.selection = SelectionState::None;
                        changed = true;
                    }
                }
            }
        }

        if let Ok(mut inspector) = self.inspector.lock() {
            if inspector.focused_agent.is_some() {
                inspector.focused_agent = None;
                changed = true;
            }
            if inspector.hovered_agent.is_some() {
                inspector.hovered_agent = None;
                changed = true;
            }
        }

        changed
    }

    fn update_selection_from_point(&mut self, position: Point<Pixels>, extend: bool) -> bool {
        let Some(world_point) = self.canvas_to_world(position) else {
            if extend {
                return false;
            }
            let cleared = self.clear_all_selections();
            if cleared {
                self.record_selection_event(SelectionEventKind::Clear);
            }
            return cleared;
        };

        let (prior_focus, prev_hover) = self
            .inspector
            .lock()
            .map(|state| (state.focused_agent, state.hovered_agent))
            .unwrap_or((None, None));

        let mut selection_changed = false;
        let mut candidate_id = None;
        let mut selected_after: Vec<AgentId> = Vec::new();
        let mut world_applied = false;

        if let Ok(mut world) = self.world.lock() {
            let pick_radius = self.selection_pick_radius(&world);
            let candidate = self.pick_agent_near(&world, world_point, pick_radius);

            {
                let runtime = world.runtime_mut();

                if !extend {
                    for entry in runtime.values_mut() {
                        if !matches!(entry.selection, SelectionState::None) {
                            entry.selection = SelectionState::None;
                            selection_changed = true;
                        }
                    }
                }

                if let Some(id) = candidate {
                    if let Some(entry) = runtime.get_mut(id) {
                        let was_selected = matches!(entry.selection, SelectionState::Selected);
                        if extend && was_selected {
                            entry.selection = SelectionState::None;
                            selection_changed = true;
                        } else {
                            if !was_selected {
                                entry.selection = SelectionState::Selected;
                                selection_changed = true;
                            }
                            candidate_id = Some(id);
                        }
                    }
                } else if !extend {
                    candidate_id = None;
                }

                if let Some(prev) = prev_hover {
                    if let Some(entry) = runtime.get_mut(prev) {
                        if matches!(entry.selection, SelectionState::Hovered) {
                            entry.selection = SelectionState::None;
                            selection_changed = true;
                        }
                    }
                }

                selected_after = runtime
                    .iter()
                    .filter_map(|(id, entry)| {
                        if matches!(entry.selection, SelectionState::Selected) {
                            Some(id)
                        } else {
                            None
                        }
                    })
                    .collect();
            }

            world_applied = true;
        }

        if !world_applied {
            return false;
        }

        let mut focus_after = if let Some(id) = candidate_id {
            if selected_after.contains(&id) {
                Some(id)
            } else if extend {
                None
            } else {
                None
            }
        } else {
            None
        };

        if extend {
            if focus_after.is_none() {
                if let Some(prev) = prior_focus {
                    if selected_after.contains(&prev) {
                        focus_after = Some(prev);
                    }
                }
            }
            if focus_after.is_none() {
                focus_after = selected_after.first().copied();
            }
        }

        let focus_changed = focus_after != prior_focus;

        if let Ok(mut inspector) = self.inspector.lock() {
            inspector.focused_agent = focus_after;
            inspector.hovered_agent = None;
        }

        if selection_changed {
            self.record_selection_event(SelectionEventKind::Click);
        } else if focus_changed {
            self.record_selection_event(SelectionEventKind::Focus);
        }

        selection_changed || focus_changed
    }

    fn handle_canvas_hover(&mut self, event: &MouseMoveEvent) -> bool {
        let mut changed = self.set_shift_inspect(event.modifiers.shift);
        if self.update_hover_from_point(event.position) {
            changed = true;
        }
        changed
    }

    fn set_shift_inspect(&mut self, active: bool) -> bool {
        if self.shift_inspect != active {
            self.shift_inspect = active;
            true
        } else {
            false
        }
    }

    fn update_hover_from_point(&mut self, position: Point<Pixels>) -> bool {
        let hovered = if let Some(world_point) = self.canvas_to_world(position) {
            if let Ok(world) = self.world.lock() {
                let radius = self.selection_pick_radius(&world);
                self.pick_agent_near(&world, world_point, radius)
            } else {
                None
            }
        } else {
            None
        };

        self.apply_hover_change(hovered)
    }

    fn apply_hover_change(&mut self, hovered: Option<AgentId>) -> bool {
        let prev_hover = self
            .inspector
            .lock()
            .map(|state| state.hovered_agent)
            .unwrap_or(None);

        if prev_hover == hovered {
            return false;
        }

        let mut desired = hovered;
        let mut selection_changed = false;

        if let Ok(mut world) = self.world.lock() {
            let runtime = world.runtime_mut();

            if let Some(prev) = prev_hover {
                if let Some(entry) = runtime.get_mut(prev) {
                    if matches!(entry.selection, SelectionState::Hovered) {
                        entry.selection = SelectionState::None;
                        selection_changed = true;
                    }
                }
            }

            if let Some(curr) = hovered {
                if runtime
                    .get(curr)
                    .map(|entry| matches!(entry.selection, SelectionState::Selected))
                    .unwrap_or(false)
                {
                    desired = None;
                } else if let Some(entry) = runtime.get_mut(curr) {
                    if !matches!(entry.selection, SelectionState::Hovered) {
                        entry.selection = SelectionState::Hovered;
                        selection_changed = true;
                    }
                }
            }
        } else {
            return false;
        }

        let mut inspector_changed = false;
        if let Ok(mut inspector) = self.inspector.lock() {
            inspector_changed = inspector.hovered_agent != desired;
            inspector.hovered_agent = desired;
        }

        selection_changed || inspector_changed
    }
    fn snapshot(&mut self) -> HudSnapshot {
        self.pump_simulation();
        let mut snapshot = HudSnapshot::default();
        let inspector_state = self
            .inspector
            .lock()
            .map(|state| state.clone())
            .unwrap_or_default();

        let analytics_trigger = {
            let mut trigger: Option<(u64, usize)> = None;
            if let Ok(world) = self.world.lock() {
                snapshot.tick = world.tick().0;
                snapshot.epoch = world.epoch();
                snapshot.is_closed = world.is_closed();
                snapshot.agent_count = world.agent_count();

                let config = world.config();
                snapshot.world_size = (config.world_width, config.world_height);
                snapshot.history_capacity = config.history_capacity;
                snapshot.render_frame = RenderFrame::from_world(&world, self.accessibility.palette);

                let mut ring: VecDeque<TickSummary> = VecDeque::with_capacity(12);
                for summary in world.history() {
                    if ring.len() == 12 {
                        ring.pop_front();
                    }
                    ring.push_back(summary.clone());
                }
                if let Some(latest) = ring.back() {
                    snapshot.summary = Some(HudMetrics::from(latest));
                }
                snapshot.recent_history = ring.into_iter().map(HudHistoryEntry::from).collect();
                snapshot.inspector = InspectorSnapshot::from_world(&world, &inspector_state);

                if let Some(metrics) = snapshot.summary.as_ref() {
                    trigger = Some((metrics.tick, metrics.agent_count));
                }
            }
            trigger
        };

        if let Some((tick, count)) = analytics_trigger {
            self.maybe_refresh_analytics(tick, count);
        }

        if self.storage.is_some() {
            snapshot.analytics = self.analytics_cache.clone();
        }

        snapshot.perf = self.last_perf;
        snapshot.controls = self.controls.snapshot();

        self.playback.record(snapshot.clone());

        snapshot
    }

    fn maybe_refresh_analytics(&mut self, tick: u64, agent_count: usize) {
        let Some(storage) = &self.storage else {
            return;
        };
        if self.analytics_tick == Some(tick) {
            return;
        }
        match storage.try_lock() {
            Ok(mut guard) => match guard.latest_metrics(256) {
                Ok(readings) => {
                    if let Some(analytics) = parse_analytics(tick, agent_count, &readings) {
                        self.analytics_tick = Some(tick);
                        self.analytics_cache = Some(analytics);
                    }
                }
                Err(err) => {
                    error!(?err, "failed to fetch latest metrics for analytics");
                }
            },
            Err(_) => {
                // Avoid blocking the UI when storage is busy; we'll try again next frame.
            }
        }
    }

    fn render_header(&self, snapshot: &HudSnapshot) -> Div {
        let status_text = if snapshot.is_closed {
            "Closed Ecosystem"
        } else {
            "Open Ecosystem"
        };
        let status_color = if snapshot.is_closed {
            rgb(0xf97316)
        } else {
            rgb(0x22c55e)
        };

        let subline = format!(
            "Tick #{}, epoch {}, {} active agents",
            snapshot.tick, snapshot.epoch, snapshot.agent_count
        );

        let badge_canvas = {
            let state = HeaderBadgeState {
                phase: snapshot.tick as f32 * 0.02,
                palette: self.accessibility.palette,
            };
            canvas(
                move |_, _, _| state,
                move |bounds, state, window, _| paint_header_badge(bounds, state, window),
            )
            .w(px(56.0))
            .h(px(56.0))
            .flex_none()
        };

        div()
            .flex()
            .justify_between()
            .items_center()
            .gap_4()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(badge_canvas)
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(div().text_3xl().child(self.title.clone()))
                            .child(div().text_sm().text_color(rgb(0x94a3b8)).child(subline)),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .px_3()
                            .py_1()
                            .rounded_full()
                            .bg(status_color)
                            .text_sm()
                            .text_color(rgb(0x0f172a))
                            .child(status_text),
                    )
                    .child(div().text_sm().text_color(rgb(0x94a3b8)).child(format!(
                        "World {}×{} · History cap {}",
                        snapshot.world_size.0, snapshot.world_size.1, snapshot.history_capacity
                    ))),
            )
    }

    fn render_summary(&self, snapshot: &HudSnapshot) -> Div {
        let mut cards: Vec<Div> = Vec::new();

        if let Some(metrics) = snapshot.summary.as_ref() {
            let growth = metrics.net_growth();
            let growth_accent = if growth >= 0 { 0x22c55e } else { 0xef4444 };
            let growth_label = if growth >= 0 {
                format!("Net +{}", growth)
            } else {
                format!("Net {}", growth)
            };

            let history = &snapshot.recent_history;
            let agents_series = sparkline_from_history(history, |entry| entry.agent_count as f32);
            let growth_series = sparkline_from_history(history, |entry| entry.net_growth() as f32);
            let energy_series = sparkline_from_history(history, |entry| entry.average_energy);
            let health_series = sparkline_from_history(history, |entry| entry.average_health);

            cards.push(self.metric_card(
                "Tick",
                metrics.tick.to_string(),
                0x38bdf8,
                Some(format!("Epoch {}", snapshot.epoch)),
                None,
            ));
            cards.push(self.metric_card(
                "Agents",
                metrics.agent_count.to_string(),
                0x22c55e,
                Some(format!("{} active", snapshot.agent_count)),
                agents_series.clone(),
            ));
            cards.push(self.metric_card(
                "Births / Deaths",
                format!("{} / {}", metrics.births, metrics.deaths),
                growth_accent,
                Some(growth_label),
                growth_series.clone(),
            ));
            cards.push(self.metric_card(
                "Avg Energy",
                self.format_float(metrics.average_energy, 2),
                0xf59e0b,
                Some(format!(
                    "Total {}",
                    self.format_float(metrics.total_energy, 1)
                )),
                energy_series.clone(),
            ));
            cards.push(self.metric_card(
                "Avg Health",
                self.format_float(metrics.average_health, 2),
                0x8b5cf6,
                None,
                health_series,
            ));
        } else {
            cards.push(
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .rounded_lg()
                    .border_1()
                    .border_color(rgb(0x1d4ed8))
                    .bg(rgb(0x111827))
                    .p_5()
                    .child(
                        div()
                            .text_lg()
                            .text_color(rgb(0x93c5fd))
                            .child("No metrics yet"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(rgb(0x64748b))
                            .child("Run the simulation to generate tick summaries."),
                    ),
            );
        }

        let perf = snapshot.perf;
        let frame_value = if perf.sample_count == 0 {
            "—".to_string()
        } else {
            format!("{} ms", self.format_float(perf.latest_ms, 2))
        };
        let frame_detail = if perf.sample_count == 0 {
            "Collecting samples…".to_string()
        } else {
            format!(
                "avg {} · min {} · max {}",
                self.format_float(perf.average_ms, 2),
                self.format_float(perf.min_ms, 2),
                self.format_float(perf.max_ms, 2)
            )
        };
        cards.push(self.metric_card(
            "Frame Time",
            frame_value,
            0x14b8a6,
            Some(frame_detail),
            None,
        ));

        let fps_value = if perf.sample_count == 0 {
            "—".to_string()
        } else {
            self.format_float(perf.fps, 1)
        };
        let fps_detail = if perf.sample_count == 0 {
            "Awaiting samples".to_string()
        } else {
            format!("Samples {}", perf.sample_count)
        };
        cards.push(self.metric_card("FPS", fps_value, 0xf97316, Some(fps_detail), None));

        let controls = snapshot.controls;
        let speed_value = if controls.paused {
            "Paused".to_string()
        } else {
            format!("{}×", self.format_float(controls.speed_multiplier, 2))
        };
        let bool_label = |value: bool| if value { "On" } else { "Off" };
        let speed_detail = format!(
            "Agents {} · Food {} · Outline {} · {}",
            bool_label(controls.draw_agents),
            bool_label(controls.draw_food),
            bool_label(controls.agent_outline),
            controls.follow_mode.label()
        );
        cards.push(self.metric_card(
            "Sim Controls",
            speed_value,
            0x60a5fa,
            Some(speed_detail),
            None,
        ));

        let column_count = cards.len().clamp(1, 4) as u16;

        div().grid().grid_cols(column_count).gap_4().children(cards)
    }

    fn render_analytics_panel(&self, snapshot: &HudSnapshot) -> Div {
        let Some(analytics) = snapshot.analytics.as_ref() else {
            return div();
        };

        let total_agents = snapshot
            .summary
            .as_ref()
            .map(|metrics| metrics.agent_count)
            .unwrap_or(snapshot.agent_count)
            .max(1);

        let share_detail = |count: usize, avg_energy: f64| -> String {
            let share = (count as f64 / total_agents as f64 * 100.0).clamp(0.0, 100.0);
            format!("{share:.1}% share · avg ⚡ {avg_energy:.2}")
        };

        let trophic_cards = vec![
            self.metric_card(
                "Carnivores",
                analytics.carnivores.to_string(),
                0xcb2a3b,
                Some(share_detail(
                    analytics.carnivores,
                    analytics.carnivore_avg_energy,
                )),
                None,
            ),
            self.metric_card(
                "Herbivores",
                analytics.herbivores.to_string(),
                0x22c55e,
                Some(share_detail(
                    analytics.herbivores,
                    analytics.herbivore_avg_energy,
                )),
                None,
            ),
            self.metric_card(
                "Hybrids",
                analytics.hybrids.to_string(),
                0x8b5cf6,
                Some(share_detail(analytics.hybrids, analytics.hybrid_avg_energy)),
                None,
            ),
        ];

        let trophic_row = div()
            .grid()
            .grid_cols(trophic_cards.len() as u16)
            .gap_4()
            .children(trophic_cards);

        let meta_bar = div()
            .flex()
            .justify_between()
            .gap_4()
            .text_xs()
            .text_color(rgb(0x94a3b8))
            .child(div().child(format!("Tick {}", analytics.tick)))
            .child(div().child(format!("Boosts {}", analytics.boost_count)))
            .child(div().child(format!("Births {}", analytics.births_total)));

        let resource_panel = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_lg()
            .bg(rgb(0x0e172a))
            .border_1()
            .border_color(rgb(0x1e293b))
            .child(div().text_sm().text_color(rgb(0x7dd3fc)).child("Resources"))
            .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                "Total {:.1} · Mean {:.3} · σ {:.3}",
                analytics.food_total, analytics.food_mean, analytics.food_stddev
            )))
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
                "Δ mean {:.4} · |Δ| {:.4}",
                analytics.food_delta_mean, analytics.food_delta_mean_abs
            )));

        let mutation_panel = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_lg()
            .bg(rgb(0x101a2e))
            .border_1()
            .border_color(rgb(0x1f2a3d))
            .child(div().text_sm().text_color(rgb(0xfbbf24)).child("Mutation"))
            .child(div().text_xs().text_color(rgb(0xfef3c7)).child(format!(
                "Primary {:.4} ± {:.4}",
                analytics.mutation_primary_mean, analytics.mutation_primary_stddev
            )))
            .child(div().text_xs().text_color(rgb(0xfef3c7)).child(format!(
                "Secondary {:.4} ± {:.4}",
                analytics.mutation_secondary_mean, analytics.mutation_secondary_stddev
            )));

        let behavior_panel = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_lg()
            .bg(rgb(0x111d31))
            .border_1()
            .border_color(rgb(0x1e293b))
            .child(div().text_sm().text_color(rgb(0x93c5fd)).child("Behavior"))
            .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                "Sensors μ {:.3} · H {:.3}",
                analytics.behavior_sensor_mean, analytics.behavior_sensor_entropy
            )))
            .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                "Outputs μ {:.3} · H {:.3}",
                analytics.behavior_output_mean, analytics.behavior_output_entropy
            )));
        let age_panel = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_lg()
            .bg(rgb(0x101624))
            .border_1()
            .border_color(rgb(0x1d2738))
            .child(div().text_sm().text_color(rgb(0xf59e0b)).child("Age"))
            .child(div().text_xs().text_color(rgb(0xfef3c7)).child(format!(
                "Mean {:.2} · Max {:.0} · Gen μ {:.1}",
                analytics.age_mean, analytics.age_max, analytics.generation_mean
            )))
            .child(div().text_xs().text_color(rgb(0xfdba74)).child(format!(
                "Gen max {:.0} · Hybrid births {} ({:.1}%)",
                analytics.generation_max,
                analytics.births_hybrid,
                analytics.births_hybrid_ratio * 100.0
            )))
            .child(div().text_xs().text_color(rgb(0xfdba74)).child(format!(
                "Repro μ {:.2} · Boost {:.1}%",
                analytics.reproduction_counter_mean,
                analytics.boost_ratio * 100.0
            )));

        let temperature_panel = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_lg()
            .bg(rgb(0x121f33))
            .border_1()
            .border_color(rgb(0x1f2f46))
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x38bdf8))
                    .child("Temperature"),
            )
            .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                "Preference μ {:.3} · σ {:.3}",
                analytics.temperature_preference_mean, analytics.temperature_preference_stddev
            )))
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
                "Discomfort μ {:.3} · σ {:.3}",
                analytics.temperature_discomfort_mean, analytics.temperature_discomfort_stddev
            )));

        let mortality_panel = {
            let total = analytics.deaths_total.max(1);
            let make_row = |label: &str, count: usize| {
                let ratio = (count as f64 / total as f64) * 100.0;
                let label_text: SharedString = label.to_string().into();
                div()
                    .flex()
                    .justify_between()
                    .text_xs()
                    .text_color(rgb(0xe2e8f0))
                    .child(div().child(label_text))
                    .child(div().child(format!("{count} ({ratio:.1}%)")))
            };

            div()
                .flex()
                .flex_col()
                .gap_1()
                .p_3()
                .rounded_lg()
                .bg(rgb(0x160f24))
                .border_1()
                .border_color(rgb(0x27193a))
                .child(div().text_sm().text_color(rgb(0xf472b6)).child("Mortality"))
                .child(make_row("Carnivore", analytics.deaths_combat_carnivore))
                .child(make_row("Herbivore", analytics.deaths_combat_herbivore))
                .child(make_row("Starvation", analytics.deaths_starvation))
                .child(make_row("Aging", analytics.deaths_aging))
                .child(make_row("Other", analytics.deaths_unknown))
                .child(
                    div()
                        .flex()
                        .justify_between()
                        .text_xs()
                        .text_color(rgb(0x94a3b8))
                        .child(div().child("Total"))
                        .child(div().child(analytics.deaths_total.to_string())),
                )
        };

        let insights_row = div().grid().grid_cols(3).gap_4().children(vec![
            resource_panel,
            mutation_panel,
            behavior_panel,
            age_panel,
            temperature_panel,
            mortality_panel,
        ]);

        let mut brain_rows: Vec<Div> = Vec::new();
        brain_rows.push(
            div()
                .flex()
                .text_xs()
                .text_color(rgb(0x64748b))
                .gap_4()
                .child(div().w(px(140.0)).child("BRAIN"))
                .child(div().w(px(80.0)).child("COUNT"))
                .child(div().w(px(80.0)).child("SHARE"))
                .child(div().w(px(100.0)).child("AVG ENERGY")),
        );

        if analytics.brain_shares.is_empty() {
            brain_rows.push(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("No brain metrics yet"),
            );
        } else {
            for entry in analytics.brain_shares.iter().take(6) {
                let share = (entry.count as f64 / total_agents as f64 * 100.0).clamp(0.0, 100.0);
                brain_rows.push(
                    div()
                        .flex()
                        .gap_4()
                        .items_center()
                        .text_xs()
                        .text_color(rgb(0xe2e8f0))
                        .child(div().w(px(140.0)).child(entry.label.clone()))
                        .child(div().w(px(80.0)).child(entry.count.to_string()))
                        .child(div().w(px(80.0)).child(format!("{share:.1}%")))
                        .child(div().w(px(100.0)).child(format!("{:.3}", entry.avg_energy))),
                );
            }
        }

        let brain_panel = div()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .rounded_lg()
            .bg(rgb(0x0d1626))
            .border_1()
            .border_color(rgb(0x1a2337))
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x38bdf8))
                    .child("Brain Share"),
            )
            .children(brain_rows);

        div()
            .flex()
            .flex_col()
            .gap_4()
            .child(meta_bar)
            .child(trophic_row)
            .child(insights_row)
            .child(brain_panel)
    }

    fn render_history(&self, snapshot: &HudSnapshot) -> Div {
        let header = div()
            .flex()
            .justify_between()
            .items_center()
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x93c5fd))
                    .child("Recent Tick History"),
            )
            .child(div().text_xs().text_color(rgb(0x64748b)).child(format!(
                "Showing {} of {} entries",
                snapshot.recent_history.len(),
                snapshot.history_capacity
            )));

        let rows: Vec<Div> = if snapshot.recent_history.is_empty() {
            vec![
                div()
                    .rounded_lg()
                    .bg(rgb(0x0f172a))
                    .border_1()
                    .border_color(rgb(0x1d4ed8))
                    .p_4()
                    .child(
                        div()
                            .text_sm()
                            .text_color(rgb(0x64748b))
                            .child("No persisted tick history yet."),
                    ),
            ]
        } else {
            snapshot
                .recent_history
                .iter()
                .enumerate()
                .map(|(idx, entry)| {
                    let row_bg = if idx % 2 == 0 {
                        rgb(0x111b2b)
                    } else {
                        rgb(0x0f172a)
                    };
                    let growth = entry.net_growth();
                    let growth_color = if growth >= 0 {
                        rgb(0x22c55e)
                    } else {
                        rgb(0xef4444)
                    };
                    let growth_label = if growth >= 0 {
                        format!("+{}", growth)
                    } else {
                        growth.to_string()
                    };

                    div()
                        .flex()
                        .justify_between()
                        .items_center()
                        .rounded_lg()
                        .bg(row_bg)
                        .p_3()
                        .child(
                            div()
                                .text_sm()
                                .text_color(rgb(0x94a3b8))
                                .child(format!("Tick {}", entry.tick)),
                        )
                        .child(
                            div()
                                .flex()
                                .gap_4()
                                .items_center()
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(rgb(0xf8fafc))
                                        .child(format!("Agents {}", entry.agent_count)),
                                )
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(rgb(0xf97316))
                                        .child(format!("Births {}", entry.births)),
                                )
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(rgb(0x38bdf8))
                                        .child(format!("Deaths {}", entry.deaths)),
                                )
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(growth_color)
                                        .child(format!("Δ {}", growth_label)),
                                )
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(rgb(0xfacc15))
                                        .child(format!("⌀ energy {:.2}", entry.average_energy)),
                                ),
                        )
                })
                .collect()
        };

        div()
            .flex()
            .flex_col()
            .w(px(280.0))
            .flex_none()
            .bg(rgb(0x111827))
            .border_1()
            .border_color(rgb(0x1d4ed8))
            .rounded_xl()
            .shadow_lg()
            .p_4()
            .gap_3()
            .child(header)
            .children(rows)
    }

    fn render_canvas(&self, snapshot: &HudSnapshot, cx: &mut Context<Self>) -> Div {
        if let Some(frame) = snapshot.render_frame.clone() {
            self.render_canvas_world(snapshot, frame, cx)
        } else {
            self.render_canvas_placeholder(snapshot)
        }
    }

    fn render_canvas_world(
        &self,
        snapshot: &HudSnapshot,
        frame: RenderFrame,
        cx: &mut Context<Self>,
    ) -> Div {
        let follow_target = self.compute_follow_target(&frame, &snapshot.inspector);
        let canvas_state = CanvasState {
            frame: frame.clone(),
            camera: Arc::clone(&self.camera),
            focus_agent: snapshot.inspector.focus_id,
            controls: snapshot.controls,
            debug: self.debug,
            follow_target,
        };

        let canvas_element = canvas(
            move |_, _, _| canvas_state.clone(),
            move |bounds, state, window, _| {
                paint_frame(
                    &state.frame,
                    &state.camera,
                    state.focus_agent,
                    state.controls,
                    state.debug,
                    state.follow_target,
                    bounds,
                    window,
                )
            },
        )
        .flex_1();

        let canvas_stack = div()
            .relative()
            .flex_1()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &MouseDownEvent, _, cx| {
                    let extend = event.modifiers.shift;
                    let mut changed = this.update_selection_from_point(event.position, extend);
                    if this.set_shift_inspect(event.modifiers.shift) {
                        changed = true;
                    }
                    if changed {
                        cx.notify();
                    }
                }),
            )
            .on_mouse_down(
                MouseButton::Middle,
                cx.listener(|this, event: &MouseDownEvent, _, cx| {
                    if let Ok(mut camera) = this.camera.lock() {
                        camera.start_pan(event.position);
                        cx.notify();
                    }
                }),
            )
            .on_mouse_up(
                MouseButton::Middle,
                cx.listener(|this, _event: &MouseUpEvent, _, _| {
                    if let Ok(mut camera) = this.camera.lock() {
                        camera.end_pan();
                    }
                }),
            )
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                let mut changed = false;
                let mut panning = false;
                if let Ok(mut camera) = this.camera.lock() {
                    if camera.update_pan(event.position) {
                        changed = true;
                    }
                    panning = camera.panning;
                }

                if panning {
                    if this.apply_hover_change(None) {
                        changed = true;
                    }
                    if this.set_shift_inspect(false) {
                        changed = true;
                    }
                } else if this.handle_canvas_hover(event) {
                    changed = true;
                }

                if changed {
                    cx.notify();
                }
            }))
            .on_scroll_wheel(cx.listener(|this, event: &ScrollWheelEvent, _, cx| {
                if let Ok(mut camera) = this.camera.lock()
                    && camera.apply_scroll(event)
                {
                    cx.notify();
                }
            }))
            .child(canvas_element)
            .child(self.render_overlay(snapshot))
            .child(self.render_history_chart(snapshot));

        let camera_snapshot = self.camera_snapshot();
        let footer = div()
            .text_xs()
            .text_color(rgb(0x475569))
            .flex()
            .justify_between()
            .child(format!(
                "World {:.0}×{:.0} units • Zoom {:.2}×",
                frame.world_size.0, frame.world_size.1, camera_snapshot.zoom,
            ))
            .child(format!(
                "Pan X {:.1}, Y {:.1}",
                camera_snapshot.offset_px.0, camera_snapshot.offset_px.1
            ));

        div()
            .flex()
            .flex_col()
            .flex_1()
            .rounded_xl()
            .border_1()
            .border_color(rgb(0x0ea5e9))
            .bg(rgb(0x0b1120))
            .shadow_lg()
            .p_4()
            .gap_3()
            .child(canvas_stack)
            .child(footer)
    }

    fn compute_follow_target(
        &self,
        frame: &RenderFrame,
        inspector: &InspectorSnapshot,
    ) -> Option<Position> {
        match self.controls.follow_mode {
            FollowMode::Off => None,
            FollowMode::Selected => inspector.focus_id.and_then(|id| {
                frame
                    .agents
                    .iter()
                    .find(|agent| agent.agent_id == id)
                    .map(|agent| agent.position)
            }),
            FollowMode::Oldest => frame
                .agents
                .iter()
                .max_by_key(|agent| agent.age)
                .map(|agent| agent.position),
        }
    }

    fn focus_agent(&mut self, agent_id: AgentId, cx: &mut Context<Self>) {
        if let Ok(mut inspector) = self.inspector.lock() {
            inspector.focused_agent = Some(agent_id);
        }

        if let Ok(mut world) = self.world.lock()
            && let Some(runtime) = world.runtime_mut().get_mut(agent_id)
        {
            runtime.selection = SelectionState::Selected;
        }

        cx.notify();
    }

    fn set_brush_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if let Ok(mut inspector) = self.inspector.lock() {
            inspector.brush_enabled = enabled;
        }
        cx.notify();
    }

    fn adjust_brush_radius(&mut self, delta: f32, cx: &mut Context<Self>) {
        if let Ok(mut inspector) = self.inspector.lock() {
            let mut radius = inspector.brush_radius + delta;
            radius = radius.clamp(8.0, 256.0);
            inspector.brush_radius = radius;
        }
        cx.notify();
    }

    fn set_probe_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if let Ok(mut inspector) = self.inspector.lock() {
            inspector.probe_enabled = enabled;
        }
        cx.notify();
    }

    fn set_debug_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.debug.enabled = enabled;
        cx.notify();
    }

    fn set_debug_show_velocity(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.debug.show_velocity = enabled;
        cx.notify();
    }

    fn set_debug_show_sense_radius(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.debug.show_sense_radius = enabled;
        cx.notify();
    }

    fn record_selection_event(&mut self, kind: SelectionEventKind) {
        let (tick, selected) = if let Ok(world) = self.world.lock() {
            let mut ids = Vec::new();
            for (id, entry) in world.runtime().iter() {
                if matches!(entry.selection, SelectionState::Selected) {
                    ids.push(id);
                }
            }
            (world.tick().0, ids)
        } else {
            (0, Vec::<AgentId>::new())
        };

        let total = selected.len();
        let sample = selected.iter().copied().take(5).collect::<Vec<AgentId>>();
        let event = SelectionEvent {
            tick,
            kind,
            total_selected: total,
            sample_ids: sample,
        };
        if self.selection_events.len() >= MAX_SELECTION_EVENTS {
            self.selection_events.pop_front();
        }
        self.selection_events.push_back(event);
    }

    fn clear_selection(&mut self, cx: &mut Context<Self>) {
        if self.clear_all_selections() {
            self.record_selection_event(SelectionEventKind::Clear);
            cx.notify();
        }
    }

    fn select_all_agents(&mut self, cx: &mut Context<Self>) {
        let mut changed = false;
        let mut first_selected: Option<AgentId> = None;
        {
            if let Ok(mut world) = self.world.lock() {
                let runtime = world.runtime_mut();
                for (id, entry) in runtime.iter_mut() {
                    if entry.selection != SelectionState::Selected {
                        entry.selection = SelectionState::Selected;
                        changed = true;
                    }
                    if first_selected.is_none() {
                        first_selected = Some(id);
                    }
                }
            }
        }
        if changed {
            self.record_selection_event(SelectionEventKind::SelectAll);
            if let Some(id) = first_selected {
                self.focus_agent(id, cx);
            } else {
                cx.notify();
            }
        }
    }

    fn focus_first_selected(&mut self, cx: &mut Context<Self>) {
        let selected_id = {
            if let Ok(world) = self.world.lock() {
                world.runtime().iter().find_map(|(id, entry)| {
                    if matches!(entry.selection, SelectionState::Selected) {
                        Some(id)
                    } else {
                        None
                    }
                })
            } else {
                None
            }
        };

        if let Some(id) = selected_id {
            self.focus_agent(id, cx);
            self.record_selection_event(SelectionEventKind::Focus);
        }
    }

    fn set_persistence_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if enabled {
            let interval = self
                .inspector
                .lock()
                .map(|mut inspector| {
                    inspector.persistence_last_enabled = inspector.persistence_last_enabled.max(1);
                    inspector.persistence_last_enabled
                })
                .unwrap_or(60);

            self.submit_config_update(|config| {
                config.persistence_interval = interval;
            });
        } else {
            let current_interval = self
                .world
                .lock()
                .map(|world| world.config().persistence_interval)
                .unwrap_or(0);

            if current_interval > 0 {
                if let Ok(mut inspector) = self.inspector.lock() {
                    inspector.persistence_last_enabled = current_interval;
                }
            }

            self.submit_config_update(|config| {
                config.persistence_interval = 0;
            });
        }

        cx.notify();
    }

    fn adjust_persistence_interval(&mut self, delta: i32, cx: &mut Context<Self>) {
        let (current_interval, was_enabled) = {
            if let Ok(world) = self.world.lock() {
                let interval = world.config().persistence_interval;
                (interval, interval > 0)
            } else {
                (0, false)
            }
        };

        let cached_interval = self
            .inspector
            .lock()
            .map(|inspector| inspector.persistence_last_enabled)
            .unwrap_or(60);

        let base_interval = if was_enabled {
            current_interval
        } else {
            cached_interval
        };

        let new_interval = ((base_interval as i32) + delta).clamp(1, 10_000) as u32;

        if let Ok(mut inspector) = self.inspector.lock() {
            inspector.persistence_last_enabled = new_interval;
        }

        if was_enabled {
            self.submit_config_update(|config| {
                config.persistence_interval = new_interval;
            });
        }

        cx.notify();
    }

    fn adjust_agent_mutation_rates(
        &mut self,
        agent_id: AgentId,
        delta_primary: f32,
        delta_secondary: f32,
        cx: &mut Context<Self>,
    ) {
        if let Ok(mut world) = self.world.lock()
            && let Some(runtime) = world.runtime_mut().get_mut(agent_id)
        {
            runtime.mutation_rates.primary =
                (runtime.mutation_rates.primary + delta_primary).max(0.0001);
            runtime.mutation_rates.secondary =
                (runtime.mutation_rates.secondary + delta_secondary).max(0.0);
        }

        cx.notify();
    }

    fn handle_key_down(&mut self, event: &KeyDownEvent, cx: &mut Context<Self>) {
        if let Some(target) = self.key_capture {
            if event.keystroke.key.eq_ignore_ascii_case("escape") {
                self.key_capture = None;
                cx.notify();
                return;
            }
            self.bindings.assign(target, event.keystroke.clone());
            self.key_capture = None;
            info!(
                "Rebound {} to {}",
                target.label(),
                format_keystroke(&event.keystroke)
            );
            cx.notify();
            return;
        }

        if let Some(action) = self.bindings.action_for(&event.keystroke) {
            self.invoke_action(action, cx);
        }
    }

    fn invoke_action(&mut self, action: CommandAction, cx: &mut Context<Self>) {
        match action {
            CommandAction::TogglePlayback => self.playback_toggle(cx),
            CommandAction::GoLive => self.playback_go_live(cx),
            CommandAction::ToggleBrush => self.toggle_brush_state(cx),
            CommandAction::ToggleNarration => self.toggle_narration(cx),
            CommandAction::CyclePalette => self.cycle_palette(cx),
            CommandAction::ToggleSimulationPause => self.toggle_simulation_pause(cx),
            CommandAction::ToggleAgentDraw => self.toggle_agent_draw(cx),
            CommandAction::ToggleFoodOverlay => self.toggle_food_overlay(cx),
            CommandAction::ToggleAgentOutline => self.toggle_agent_outline(cx),
            CommandAction::IncreaseSimulationSpeed => self.adjust_simulation_speed(0.25, cx),
            CommandAction::DecreaseSimulationSpeed => self.adjust_simulation_speed(-0.25, cx),
            CommandAction::AddCrossoverAgents => self.spawn_crossover_agent(cx),
            CommandAction::SpawnCarnivore => self.spawn_agent_with_tendency(0.0, cx),
            CommandAction::SpawnHerbivore => self.spawn_agent_with_tendency(1.0, cx),
            CommandAction::ToggleClosedEnvironment => self.toggle_closed_environment(cx),
            CommandAction::FollowSelected => self.toggle_follow_selected(cx),
            CommandAction::FollowOldest => self.toggle_follow_oldest(cx),
            CommandAction::ToggleDebugOverlay => {
                let enabled = !self.debug.enabled;
                self.set_debug_enabled(enabled, cx);
            }
            CommandAction::ClearSelection => self.clear_selection(cx),
            CommandAction::SelectAll => self.select_all_agents(cx),
            CommandAction::FocusFirstSelected => self.focus_first_selected(cx),
            CommandAction::ToggleSettings => self.toggle_settings(cx),
        }
    }

    fn toggle_settings(&mut self, cx: &mut Context<Self>) {
        self.settings_panel.open = !self.settings_panel.open;

        // Reset scroll position and recalculate content height when opening panel
        if self.settings_panel.open {
            self.settings_panel.scroll_offset = 0.0;
            let total_categories = ConfigCategory::all().len();
            self.settings_panel.content_height = self
                .settings_panel
                .estimate_content_height(total_categories);
            // Note: viewport_height uses conservative default (400px) from state
            // This ensures content is never blocked, at cost of allowing blank space on large windows
        }

        info!(open = self.settings_panel.open, "Settings panel toggled");
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn clear_search(&mut self, cx: &mut Context<Self>) {
        self.settings_panel.search_query.clear();
        info!("Search cleared");
        cx.notify();
    }

    fn update_search(&mut self, query: String, cx: &mut Context<Self>) {
        self.settings_panel.search_query = query;
        info!(query = %self.settings_panel.search_query, "Search query updated");
        cx.notify();
    }

    /// Check if text matches the current search query (case-insensitive substring match)
    fn matches_search(&self, text: &str) -> bool {
        if self.settings_panel.search_query.is_empty() {
            return true; // No search filter - show everything
        }
        // Case-insensitive substring search
        text.to_lowercase()
            .contains(&self.settings_panel.search_query.to_lowercase())
    }

    /// Render a list of parameters with search filtering - ONE central filter point for ALL 60+ params!
    fn render_filtered_params(&self, params: Vec<(&str, String, &str)>) -> Div {
        let mut container = div()
            .flex()
            .flex_col()
            .gap_3()
            .px_4()
            .py_4()
            .rounded_lg()
            .bg(rgb(0x0f172a))
            .border_1()
            .border_color(rgb(0x1e293b));

        // ✨ SINGLE CENTRALIZED FILTERING LOOP - this replaces 60+ individual checks!
        for (label, value, desc) in params {
            if self.matches_search(label)
                || self.matches_search(&value)
                || self.matches_search(desc)
            {
                container = container.child(self.render_param_readonly(label, &value, desc));
            }
        }

        container
    }

    fn toggle_category_collapse(&mut self, category: ConfigCategory, cx: &mut Context<Self>) {
        if let Some(pos) = self
            .settings_panel
            .collapsed_categories
            .iter()
            .position(|c| *c == category)
        {
            // Category is collapsed, expand it
            self.settings_panel.collapsed_categories.remove(pos);
        } else {
            // Category is expanded, collapse it
            self.settings_panel.collapsed_categories.push(category);
        }

        // Update content height and clamp scroll
        let total_categories = ConfigCategory::all().len();
        self.settings_panel.content_height = self
            .settings_panel
            .estimate_content_height(total_categories);
        // Clamp scroll with updated content height (viewport_height from state)
        self.settings_panel.clamp_scroll();

        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn toggle_brush_state(&mut self, cx: &mut Context<Self>) {
        if let Ok(mut inspector) = self.inspector.lock() {
            let new_state = !inspector.brush_enabled;
            inspector.brush_enabled = new_state;
        }
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn toggle_narration(&mut self, cx: &mut Context<Self>) {
        self.accessibility.narration_enabled = !self.accessibility.narration_enabled;
        if self.accessibility.narration_enabled {
            info!("Narration enabled");
        } else {
            info!("Narration disabled");
        }
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn set_simulation_paused(&mut self, paused: bool, cx: &mut Context<Self>) {
        if self.controls.paused == paused {
            return;
        }
        self.controls.paused = paused;
        self.sim_accumulator = 0.0;
        self.last_sim_instant = Some(Instant::now());
        info!(paused, "Simulation pause state updated");
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn set_draw_agents(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if self.controls.draw_agents == enabled {
            return;
        }
        self.controls.draw_agents = enabled;
        info!(draw_agents = enabled, "Agent rendering toggled");
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn set_draw_food(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if self.controls.draw_food == enabled {
            return;
        }
        self.controls.draw_food = enabled;
        info!(draw_food = enabled, "Food overlay toggled");
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn set_agent_outline(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if self.controls.agent_outline == enabled {
            return;
        }
        self.controls.agent_outline = enabled;
        info!(agent_outline = enabled, "Agent outline toggled");
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn set_follow_mode(&mut self, mode: FollowMode, cx: &mut Context<Self>) {
        if self.controls.follow_mode == mode {
            return;
        }
        self.controls.follow_mode = mode;
        info!(mode = ?mode, "Follow mode updated");
        cx.notify();
    }

    fn set_world_closed(&mut self, closed: bool, cx: &mut Context<Self>) {
        let mut updated = false;
        if let Ok(mut world) = self.world.lock() {
            if world.is_closed() != closed {
                world.set_closed(closed);
                updated = true;
            }
        }
        if updated {
            info!(closed, "Updated closed environment toggle");
            #[cfg(feature = "audio")]
            if let Some(audio) = self.audio.as_mut() {
                audio.play(&audio.toggle_sound);
            }
            cx.notify();
        }
    }

    fn toggle_simulation_pause(&mut self, cx: &mut Context<Self>) {
        let next = !self.controls.paused;
        self.set_simulation_paused(next, cx);
    }

    fn toggle_agent_draw(&mut self, cx: &mut Context<Self>) {
        let next = !self.controls.draw_agents;
        self.set_draw_agents(next, cx);
    }

    fn toggle_food_overlay(&mut self, cx: &mut Context<Self>) {
        let next = !self.controls.draw_food;
        self.set_draw_food(next, cx);
    }

    fn toggle_agent_outline(&mut self, cx: &mut Context<Self>) {
        let next = !self.controls.agent_outline;
        self.set_agent_outline(next, cx);
    }

    fn adjust_simulation_speed(&mut self, delta: f32, cx: &mut Context<Self>) {
        let mut speed = self.controls.speed_multiplier + delta;
        speed = speed.clamp(0.25, 4.0);
        speed = (speed * 100.0).round() / 100.0;
        if (speed - self.controls.speed_multiplier).abs() > f32::EPSILON {
            self.controls.speed_multiplier = speed;
            info!(speed = speed, "Adjusted simulation speed");
            self.last_sim_instant = Some(Instant::now());
            #[cfg(feature = "audio")]
            if let Some(audio) = self.audio.as_mut() {
                audio.play(&audio.toggle_sound);
            }
            cx.notify();
        }
    }

    fn spawn_agent_with_tendency(&mut self, herbivore_bias: f32, cx: &mut Context<Self>) {
        let mut spawned = false;
        if let Ok(mut world) = self.world.lock() {
            spawned = self.spawn_agent_with_bias_internal(&mut world, herbivore_bias);
        }
        if spawned {
            cx.notify();
        }
    }

    fn spawn_agent_with_bias_internal(&self, world: &mut WorldState, herbivore_bias: f32) -> bool {
        let width = world.config().world_width as f32;
        let height = world.config().world_height as f32;
        if width <= 0.0 || height <= 0.0 {
            return false;
        }

        let (pos_x, pos_y, color) = {
            let rng = world.rng();
            let x = rng.random_range(0.0..width);
            let y = rng.random_range(0.0..height);
            let color = [
                rng.random_range(0.15..0.95),
                rng.random_range(0.15..0.95),
                rng.random_range(0.15..0.95),
            ];
            (x, y, color)
        };

        let mut agent = AgentData::default();
        agent.position = Position::new(pos_x, pos_y);
        agent.velocity = Velocity::new(0.0, 0.0);
        agent.color = color;
        let agent_id = world.spawn_agent(agent);
        if let Some(runtime) = world.runtime_mut().get_mut(agent_id) {
            runtime.herbivore_tendency = herbivore_bias.clamp(0.0, 1.0);
            runtime.energy = runtime.energy.max(1.0);
        }
        info!(agent = ?agent_id, bias = herbivore_bias, "Spawned agent");
        true
    }

    fn spawn_crossover_agent(&mut self, cx: &mut Context<Self>) {
        let mut spawned = false;
        if let Ok(mut world) = self.world.lock() {
            let selected: Vec<AgentId> = {
                let runtime = world.runtime();
                runtime
                    .iter()
                    .filter_map(|(id, entry)| {
                        matches!(entry.selection, SelectionState::Selected).then_some(id)
                    })
                    .collect()
            };

            if selected.len() >= 2 {
                if let (Some(parent_a), Some(parent_b)) = (
                    world.snapshot_agent(selected[0]),
                    world.snapshot_agent(selected[1]),
                ) {
                    let mut child = AgentData::default();
                    child.position = Position::new(
                        (parent_a.data.position.x + parent_b.data.position.x) * 0.5,
                        (parent_a.data.position.y + parent_b.data.position.y) * 0.5,
                    );
                    child.velocity = Velocity::new(0.0, 0.0);
                    child.heading = (parent_a.data.heading + parent_b.data.heading) * 0.5;
                    child.health =
                        ((parent_a.data.health + parent_b.data.health) * 0.5).clamp(0.5, 2.0);
                    child.color = [
                        (parent_a.data.color[0] + parent_b.data.color[0]) * 0.5,
                        (parent_a.data.color[1] + parent_b.data.color[1]) * 0.5,
                        (parent_a.data.color[2] + parent_b.data.color[2]) * 0.5,
                    ];
                    child.spike_length =
                        (parent_a.data.spike_length + parent_b.data.spike_length) * 0.5;

                    let child_id = world.spawn_agent(child);
                    if let Some(runtime) = world.runtime_mut().get_mut(child_id) {
                        runtime.herbivore_tendency = (parent_a.runtime.herbivore_tendency
                            + parent_b.runtime.herbivore_tendency)
                            * 0.5;
                        runtime.mutation_rates.primary = (parent_a.runtime.mutation_rates.primary
                            + parent_b.runtime.mutation_rates.primary)
                            * 0.5;
                        runtime.mutation_rates.secondary =
                            (parent_a.runtime.mutation_rates.secondary
                                + parent_b.runtime.mutation_rates.secondary)
                                * 0.5;
                        runtime.indicator.intensity = 0.6;
                        runtime.indicator.color = [0.2, 0.8, 0.9];
                    }
                    info!(child = ?child_id, "Spawned crossover agent");
                    spawned = true;
                }
            }

            if !spawned {
                spawned = self.spawn_agent_with_bias_internal(&mut world, 0.5);
            }
        }

        if spawned {
            cx.notify();
        }
    }

    fn toggle_closed_environment(&mut self, cx: &mut Context<Self>) {
        let next = {
            if let Ok(world) = self.world.lock() {
                !world.is_closed()
            } else {
                return;
            }
        };
        self.set_world_closed(next, cx);
    }

    fn toggle_follow_selected(&mut self, cx: &mut Context<Self>) {
        let next = match self.controls.follow_mode {
            FollowMode::Selected => FollowMode::Off,
            _ => FollowMode::Selected,
        };
        self.set_follow_mode(next, cx);
    }

    fn toggle_follow_oldest(&mut self, cx: &mut Context<Self>) {
        let next = match self.controls.follow_mode {
            FollowMode::Oldest => FollowMode::Off,
            _ => FollowMode::Oldest,
        };
        self.set_follow_mode(next, cx);
    }

    fn cycle_palette(&mut self, cx: &mut Context<Self>) {
        let next = self.accessibility.palette.next();
        self.accessibility.palette = next;
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn set_palette(&mut self, palette: ColorPaletteMode, cx: &mut Context<Self>) {
        if self.accessibility.palette != palette {
            self.accessibility.palette = palette;
            cx.notify();
        }
    }

    fn playback_restart(&mut self, cx: &mut Context<Self>) {
        self.playback.restart();
        cx.notify();
    }

    fn playback_step_back(&mut self, cx: &mut Context<Self>) {
        self.playback.step_back();
        cx.notify();
    }

    fn playback_toggle(&mut self, cx: &mut Context<Self>) {
        self.playback.toggle_play();
        cx.notify();
    }

    fn playback_step_forward(&mut self, cx: &mut Context<Self>) {
        self.playback.step_forward();
        cx.notify();
    }

    fn playback_go_live(&mut self, cx: &mut Context<Self>) {
        self.playback.go_live();
        cx.notify();
    }

    #[cfg(feature = "audio")]
    fn update_audio(&mut self, snapshot: &HudSnapshot) {
        let audio = match self.audio.as_mut() {
            Some(audio) => audio,
            None => return,
        };

        if self.playback.mode() != PlaybackMode::Live {
            return;
        }

        if let Some(summary) = snapshot.summary.as_ref() {
            if summary.tick != audio.last_tick {
                if summary.births > audio.last_births {
                    audio.play(&audio.birth_sound);
                }
                if summary.deaths > audio.last_deaths {
                    audio.play(&audio.death_sound);
                }
                audio.last_births = summary.births;
                audio.last_deaths = summary.deaths;
                audio.last_tick = summary.tick;
            }
        }

        if let Some(frame) = snapshot.render_frame.as_ref() {
            let spiked = frame.agents.iter().filter(|agent| agent.spiked).count();
            if spiked > audio.last_spike_count {
                audio.play(&audio.spike_sound);
            }
            audio.last_spike_count = spiked;
        }
    }

    fn render_inspector(&self, snapshot: &HudSnapshot, cx: &mut Context<Self>) -> Div {
        let inspector = &snapshot.inspector;

        let header = div()
            .flex()
            .justify_between()
            .items_center()
            .child(div().text_lg().child("Inspector"))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child(format!("{} agents live", inspector.total_agents)),
            );

        let hovered_block = if let Some(entry) = inspector.hovered.as_ref() {
            div()
                .flex()
                .flex_col()
                .gap_1()
                .rounded_md()
                .bg(rgb(0x111b2b))
                .border_1()
                .border_color(rgb(0x1e3a8a))
                .px_3()
                .py_2()
                .child(div().text_xs().text_color(rgb(0x38bdf8)).child("Hovered"))
                .child(
                    div()
                        .flex()
                        .gap_2()
                        .items_center()
                        .child(color_swatch(entry.color))
                        .child(div().text_sm().child(entry.label.clone())),
                )
                .child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
                    "E {:.2} · H {:.2} · Age {} · Gen {}",
                    entry.energy, entry.health, entry.age, entry.generation.0
                )))
        } else {
            div()
                .rounded_md()
                .bg(rgb(0x111b2b))
                .border_1()
                .border_color(rgb(0x1e293b))
                .px_3()
                .py_2()
                .text_xs()
                .text_color(rgb(0x475569))
                .child("Hover an agent to preview")
        };

        let mut list_children: Vec<Div> = inspector
            .selected
            .iter()
            .map(|entry| self.render_inspector_entry(entry, cx))
            .collect();

        if list_children.is_empty() {
            list_children.push(
                div()
                    .text_xs()
                    .text_color(rgb(0x475569))
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0x1e293b))
                    .bg(rgb(0x0f172a))
                    .px_3()
                    .py_2()
                    .child("No agents selected"),
            );
        }

        let selection_list = div().flex().flex_col().gap_2().children(list_children);

        let brush_tools = self.render_inspector_brush_tools(inspector, cx);
        let debug_tools = self.render_debug_controls(cx);
        let persistence_controls = self.render_persistence_controls(inspector, cx);
        let playback_controls = self.render_inspector_playback_controls(cx);

        let detail = inspector
            .focused
            .as_ref()
            .map(|detail| self.render_inspector_detail(detail, cx))
            .unwrap_or_else(|| {
                div()
                    .text_xs()
                    .text_color(rgb(0x475569))
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0x1e293b))
                    .bg(rgb(0x0f172a))
                    .px_3()
                    .py_3()
                    .child("Select an agent to inspect stats")
            });

        div()
            .flex()
            .flex_col()
            .gap_3()
            .w(px(320.0))
            .flex_none()
            .bg(rgb(0x0b1223))
            .border_1()
            .border_color(rgb(0x1d4ed8))
            .rounded_xl()
            .shadow_lg()
            .p_4()
            .child(header)
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
                        "Focused: {}",
                        inspector
                            .focus_id
                            .map(|id| format!("{id:?}"))
                            .unwrap_or_else(|| "—".to_string())
                    )))
            .child(hovered_block)
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x38bdf8))
                    .child(format!("Selected agents: {}", inspector.selected.len())),
            )
            .child(self.render_selection_controls(inspector, cx))
            .child(self.render_selection_log())
            .child(self.render_simulation_controls(snapshot, cx))
            .child(selection_list)
            .child(brush_tools)
            .child(debug_tools)
            .child(persistence_controls)
            .child(self.render_accessibility_panel(cx))
            .child(playback_controls)
            .child(detail)
    }

    fn render_inspector_entry(&self, entry: &AgentListEntry, cx: &mut Context<Self>) -> Div {
        let agent_id = entry.agent_id;
        let highlight_bg = if entry.is_focused {
            rgb(0x1d4ed8)
        } else {
            rgb(0x111b2b)
        };
        let border_color = if entry.is_focused {
            rgb(0x38bdf8)
        } else {
            rgb(0x1e293b)
        };

        let focus_listener = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
            this.focus_agent(agent_id, cx);
        });

        div()
            .flex()
            .flex_col()
            .gap_1()
            .rounded_md()
            .border_1()
            .border_color(border_color)
            .bg(highlight_bg)
            .px_3()
            .py_2()
            .on_mouse_down(MouseButton::Left, focus_listener)
            .child(
                div()
                    .flex()
                    .justify_between()
                    .items_center()
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .items_center()
                            .child(color_swatch(entry.color))
                            .child(div().text_sm().child(entry.label.clone())),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xf8fafc))
                            .child(format!("E {:.2}", entry.energy)),
                    ),
            )
            .child(div().text_xs().text_color(rgb(0xe0e7ff)).child(format!(
                "H {:.2} · Age {} · Gen {}",
                entry.health, entry.age, entry.generation.0
            )))
    }

    fn render_inspector_brush_tools(
        &self,
        inspector: &InspectorSnapshot,
        cx: &mut Context<Self>,
    ) -> Div {
        let brush_on = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_brush_enabled(true, cx);
        });
        let brush_off = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_brush_enabled(false, cx);
        });
        let radius_inc = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_brush_radius(8.0, cx);
        });
        let radius_dec = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_brush_radius(-8.0, cx);
        });
        let probe_on = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_probe_enabled(true, cx);
        });
        let probe_off = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_probe_enabled(false, cx);
        });

        let brush_on_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("On")
                .on_mouse_down(MouseButton::Left, brush_on);
            if inspector.brush_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let brush_off_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Off")
                .on_mouse_down(MouseButton::Left, brush_off);
            if !inspector.brush_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let brush_minus_button = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .px_2()
            .py_1()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .child("-")
            .on_mouse_down(MouseButton::Left, radius_dec);

        let brush_plus_button = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .px_2()
            .py_1()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .child("+")
            .on_mouse_down(MouseButton::Left, radius_inc);

        let probe_on_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("On")
                .on_mouse_down(MouseButton::Left, probe_on);
            if inspector.probe_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let probe_off_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Off")
                .on_mouse_down(MouseButton::Left, probe_off);
            if !inspector.probe_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Brush Tools"),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .children(vec![brush_on_button, brush_off_button]),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child(format!("Radius {:.0}", inspector.brush_radius)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .children(vec![brush_minus_button, brush_plus_button]),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x94a3b8))
                            .child("Debug probe"),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .children(vec![probe_on_button, probe_off_button]),
                    ),
            )
    }

    fn render_persistence_controls(
        &self,
        inspector: &InspectorSnapshot,
        cx: &mut Context<Self>,
    ) -> Div {
        let enable = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_persistence_enabled(true, cx);
        });
        let disable = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_persistence_enabled(false, cx);
        });
        let inc_small = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_persistence_interval(5, cx);
        });
        let dec_small = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_persistence_interval(-5, cx);
        });
        let inc_large = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_persistence_interval(25, cx);
        });
        let dec_large = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_persistence_interval(-25, cx);
        });

        let display_interval = if inspector.persistence_enabled {
            inspector.persistence_interval.max(1)
        } else {
            inspector.persistence_cached_interval.max(1)
        };

        let on_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("On")
                .on_mouse_down(MouseButton::Left, enable);
            if inspector.persistence_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let off_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Off")
                .on_mouse_down(MouseButton::Left, disable);
            if !inspector.persistence_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        fn build_interval_button<L>(label: &str, listener: L) -> Div
        where
            L: Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
        {
            div()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x111b2b))
                .px_2()
                .py_1()
                .text_xs()
                .text_color(rgb(0xcbd5f5))
                .child(label.to_string())
                .on_mouse_down(MouseButton::Left, listener)
        }

        let minus_large = build_interval_button("-25", dec_large);
        let minus_small = build_interval_button("-5", dec_small);
        let plus_small = build_interval_button("+5", inc_small);
        let plus_large = build_interval_button("+25", inc_large);

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Storage / Persistence"),
            )
            .child(div().flex().gap_2().children(vec![on_button, off_button]))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0xcbd5f5))
                    .child(format!("Interval {} ticks", display_interval)),
            )
            .child(div().flex().gap_1().children(vec![
                minus_large,
                minus_small,
                plus_small,
                plus_large,
            ]))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x64748b))
                    .child("Disabling stores the last interval for quick re-enable."),
            )
    }

    fn render_selection_controls(
        &self,
        inspector: &InspectorSnapshot,
        cx: &mut Context<Self>,
    ) -> Div {
        let clear = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.clear_selection(cx);
        });
        let select_all = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.select_all_agents(cx);
        });
        let focus_first = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.focus_first_selected(cx);
        });

        let clear_binding = self
            .bindings
            .map
            .get(&CommandAction::ClearSelection)
            .cloned()
            .unwrap_or_default();
        let select_all_binding = self
            .bindings
            .map
            .get(&CommandAction::SelectAll)
            .cloned()
            .unwrap_or_default();
        let focus_binding = self
            .bindings
            .map
            .get(&CommandAction::FocusFirstSelected)
            .cloned()
            .unwrap_or_default();

        let style_action_button = |button: Div| {
            button
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x111b2b))
                .text_color(rgb(0xcbd5f5))
        };

        let clear_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Clear")
                .on_mouse_down(MouseButton::Left, clear);
            style_action_button(base)
        };
        let select_all_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Select all")
                .on_mouse_down(MouseButton::Left, select_all);
            style_action_button(base)
        };
        let focus_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Focus first")
                .on_mouse_down(MouseButton::Left, focus_first);
            style_action_button(base)
        };

        let hover_label = inspector
            .hovered
            .as_ref()
            .map(|entry| entry.label.clone())
            .unwrap_or_else(|| "—".to_string());
        let focus_label = inspector
            .focus_id
            .map(|id| format!("{id:?}"))
            .unwrap_or_else(|| "—".to_string());

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Selection tools"),
            )
            .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                "Selected {} · Hover {} · Focus {}",
                inspector.selected.len(),
                hover_label,
                focus_label
            )))
            .child(div().flex().gap_1().children(vec![
                clear_button,
                select_all_button,
                focus_button,
            ]))
            .child(div().text_xs().text_color(rgb(0x64748b)).child(format!(
                "Shortcuts: Clear {} · Select all {} · Focus {}",
                format_keystroke(&clear_binding),
                format_keystroke(&select_all_binding),
                format_keystroke(&focus_binding)
            )))
    }

    fn render_selection_log(&self) -> Div {
        let mut items: Vec<Div> = Vec::new();
        if self.selection_events.is_empty() {
            items.push(
                div()
                    .text_xs()
                    .text_color(rgb(0x475569))
                    .bg(rgb(0x0f172a))
                    .border_1()
                    .border_color(rgb(0x1e293b))
                    .rounded_md()
                    .px_2()
                    .py_2()
                    .child("No recent selection changes"),
            );
        } else {
            for event in self.selection_events.iter().rev().take(8) {
                let sample = if event.sample_ids.is_empty() {
                    "—".to_string()
                } else {
                    event
                        .sample_ids
                        .iter()
                        .map(|id| format!("{:?}", id))
                        .collect::<Vec<_>>()
                        .join(", ")
                };

                items.push(
                    div()
                        .bg(rgb(0x0f172a))
                        .border_1()
                        .border_color(rgb(0x1e293b))
                        .rounded_md()
                        .px_2()
                        .py_2()
                        .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                            "Tick {} · {} · selected {}",
                            event.tick,
                            event.kind.label(),
                            event.total_selected
                        )))
                        .child(
                            div()
                                .text_xs()
                                .text_color(rgb(0x64748b))
                                .child(format!("Sample [{}]", sample)),
                        ),
                );
            }
        }

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Selection history"),
            )
            .children(items)
    }

    fn render_debug_controls(&self, cx: &mut Context<Self>) -> Div {
        let debug_state = self.debug;
        let overlay_binding = self
            .bindings
            .map
            .get(&CommandAction::ToggleDebugOverlay)
            .cloned()
            .unwrap_or_default();

        let enable_debug = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_debug_enabled(true, cx);
        });
        let disable_debug = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_debug_enabled(false, cx);
        });
        let show_velocity = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_debug_show_velocity(true, cx);
        });
        let hide_velocity = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_debug_show_velocity(false, cx);
        });
        let show_sense = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_debug_show_sense_radius(true, cx);
        });
        let hide_sense = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_debug_show_sense_radius(false, cx);
        });

        let style_toggle = |button: Div, active: bool| {
            if active {
                button
                    .border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                button
                    .border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let overlay_on = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("On")
                .on_mouse_down(MouseButton::Left, enable_debug);
            style_toggle(base, debug_state.enabled)
        };
        let overlay_off = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Off")
                .on_mouse_down(MouseButton::Left, disable_debug);
            style_toggle(base, !debug_state.enabled)
        };
        let velocity_on = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("On")
                .on_mouse_down(MouseButton::Left, show_velocity);
            style_toggle(base, debug_state.show_velocity)
        };
        let velocity_off = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Off")
                .on_mouse_down(MouseButton::Left, hide_velocity);
            style_toggle(base, !debug_state.show_velocity)
        };
        let sense_on = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("On")
                .on_mouse_down(MouseButton::Left, show_sense);
            style_toggle(base, debug_state.show_sense_radius)
        };
        let sense_off = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Off")
                .on_mouse_down(MouseButton::Left, hide_sense);
            style_toggle(base, !debug_state.show_sense_radius)
        };

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Debug overlays"),
            )
            .child(div().flex().gap_2().children(vec![overlay_on, overlay_off]))
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child("Velocity arrows"),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .children(vec![velocity_on, velocity_off]),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child("Sense radius"),
                    )
                    .child(div().flex().gap_2().children(vec![sense_on, sense_off])),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x64748b))
                    .child(format!("Shortcut {}", format_keystroke(&overlay_binding))),
            )
    }

    fn render_simulation_controls(&self, snapshot: &HudSnapshot, cx: &mut Context<Self>) -> Div {
        let controls = snapshot.controls;

        let run_listener = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_simulation_paused(false, cx);
        });
        let pause_listener = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_simulation_paused(true, cx);
        });
        let slower_listener = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_simulation_speed(-0.25, cx);
        });
        let faster_listener = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_simulation_speed(0.25, cx);
        });
        let agents_on = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_draw_agents(true, cx);
        });
        let agents_off = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_draw_agents(false, cx);
        });
        let food_on = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_draw_food(true, cx);
        });
        let food_off = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_draw_food(false, cx);
        });
        let outline_on = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_agent_outline(true, cx);
        });
        let outline_off = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_agent_outline(false, cx);
        });
        let follow_off = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_follow_mode(FollowMode::Off, cx);
        });
        let follow_selected = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_follow_mode(FollowMode::Selected, cx);
        });
        let follow_oldest = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_follow_mode(FollowMode::Oldest, cx);
        });
        let spawn_crossover = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.spawn_crossover_agent(cx);
        });
        let spawn_carnivore = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.spawn_agent_with_tendency(0.0, cx);
        });
        let spawn_herbivore = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.spawn_agent_with_tendency(1.0, cx);
        });
        let close_world = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_world_closed(true, cx);
        });
        let open_world = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_world_closed(false, cx);
        });

        let style_toggle = |button: Div, active: bool| {
            if active {
                button
                    .border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a5f))
                    .text_color(rgb(0xe0f2fe))
            } else {
                button
                    .border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let pause_binding = self
            .bindings
            .map
            .get(&CommandAction::ToggleSimulationPause)
            .cloned()
            .unwrap_or_default();
        let slower_binding = self
            .bindings
            .map
            .get(&CommandAction::DecreaseSimulationSpeed)
            .cloned()
            .unwrap_or_default();
        let faster_binding = self
            .bindings
            .map
            .get(&CommandAction::IncreaseSimulationSpeed)
            .cloned()
            .unwrap_or_default();
        let draw_binding = self
            .bindings
            .map
            .get(&CommandAction::ToggleAgentDraw)
            .cloned()
            .unwrap_or_default();
        let food_binding = self
            .bindings
            .map
            .get(&CommandAction::ToggleFoodOverlay)
            .cloned()
            .unwrap_or_default();
        let outline_binding = self
            .bindings
            .map
            .get(&CommandAction::ToggleAgentOutline)
            .cloned()
            .unwrap_or_default();
        let crossover_binding = self
            .bindings
            .map
            .get(&CommandAction::AddCrossoverAgents)
            .cloned()
            .unwrap_or_default();
        let carnivore_binding = self
            .bindings
            .map
            .get(&CommandAction::SpawnCarnivore)
            .cloned()
            .unwrap_or_default();
        let herbivore_binding = self
            .bindings
            .map
            .get(&CommandAction::SpawnHerbivore)
            .cloned()
            .unwrap_or_default();
        let closed_binding = self
            .bindings
            .map
            .get(&CommandAction::ToggleClosedEnvironment)
            .cloned()
            .unwrap_or_default();

        let run_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Run")
                .on_mouse_down(MouseButton::Left, run_listener),
            !controls.paused,
        );
        let pause_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(format!("Pause ({})", format_keystroke(&pause_binding)))
                .on_mouse_down(MouseButton::Left, pause_listener),
            controls.paused,
        );

        let slower_button = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .text_color(rgb(0xcbd5f5))
            .px_2()
            .py_1()
            .text_xs()
            .child(format!("– ({})", format_keystroke(&slower_binding)))
            .on_mouse_down(MouseButton::Left, slower_listener);
        let faster_button = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .text_color(rgb(0xcbd5f5))
            .px_2()
            .py_1()
            .text_xs()
            .child(format!("+ ({})", format_keystroke(&faster_binding)))
            .on_mouse_down(MouseButton::Left, faster_listener);

        let agents_on_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Agents ON")
                .on_mouse_down(MouseButton::Left, agents_on),
            controls.draw_agents,
        );
        let agents_off_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(format!("Agents OFF ({})", format_keystroke(&draw_binding)))
                .on_mouse_down(MouseButton::Left, agents_off),
            !controls.draw_agents,
        );

        let food_on_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Food ON")
                .on_mouse_down(MouseButton::Left, food_on),
            controls.draw_food,
        );
        let food_off_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(format!("Food OFF ({})", format_keystroke(&food_binding)))
                .on_mouse_down(MouseButton::Left, food_off),
            !controls.draw_food,
        );

        let outline_on_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Outline ON")
                .on_mouse_down(MouseButton::Left, outline_on),
            controls.agent_outline,
        );
        let outline_off_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(format!(
                    "Outline OFF ({})",
                    format_keystroke(&outline_binding)
                ))
                .on_mouse_down(MouseButton::Left, outline_off),
            !controls.agent_outline,
        );

        let follow_off_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Follow OFF")
                .on_mouse_down(MouseButton::Left, follow_off),
            matches!(controls.follow_mode, FollowMode::Off),
        );
        let follow_selected_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Follow selected")
                .on_mouse_down(MouseButton::Left, follow_selected),
            matches!(controls.follow_mode, FollowMode::Selected),
        );
        let follow_oldest_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Follow oldest")
                .on_mouse_down(MouseButton::Left, follow_oldest),
            matches!(controls.follow_mode, FollowMode::Oldest),
        );

        let spawn_row = div().flex().gap_2().children(vec![
            div()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x111b2b))
                .text_color(rgb(0xcbd5f5))
                .px_2()
                .py_1()
                .text_xs()
                .child(format!(
                    "Crossover ({})",
                    format_keystroke(&crossover_binding)
                ))
                .on_mouse_down(MouseButton::Left, spawn_crossover),
            div()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x111b2b))
                .text_color(rgb(0xcbd5f5))
                .px_2()
                .py_1()
                .text_xs()
                .child(format!(
                    "Carnivore ({})",
                    format_keystroke(&carnivore_binding)
                ))
                .on_mouse_down(MouseButton::Left, spawn_carnivore),
            div()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x111b2b))
                .text_color(rgb(0xcbd5f5))
                .px_2()
                .py_1()
                .text_xs()
                .child(format!(
                    "Herbivore ({})",
                    format_keystroke(&herbivore_binding)
                ))
                .on_mouse_down(MouseButton::Left, spawn_herbivore),
        ]);

        let closed_on_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Closed ON")
                .on_mouse_down(MouseButton::Left, close_world),
            snapshot.is_closed,
        );
        let closed_off_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(format!(
                    "Closed OFF ({})",
                    format_keystroke(&closed_binding)
                ))
                .on_mouse_down(MouseButton::Left, open_world),
            !snapshot.is_closed,
        );

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Simulation controls"),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .children(vec![run_button, pause_button]),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child(format!("Speed {:.2}×", controls.speed_multiplier)),
                    )
                    .child(slower_button)
                    .child(faster_button),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .children(vec![agents_on_button, agents_off_button]),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .children(vec![food_on_button, food_off_button]),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .children(vec![outline_on_button, outline_off_button]),
            )
            .child(div().flex().gap_2().children(vec![
                follow_off_button,
                follow_selected_button,
                follow_oldest_button,
            ]))
            .child(spawn_row)
            .child(
                div()
                    .flex()
                    .gap_2()
                    .children(vec![closed_off_button, closed_on_button]),
            )
    }

    fn render_inspector_playback_controls(&self, cx: &mut Context<Self>) -> Div {
        let status = self.playback.status();

        let restart = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.playback_restart(cx);
        });
        let prev = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.playback_step_back(cx);
        });
        let toggle = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.playback_toggle(cx);
        });
        let next = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.playback_step_forward(cx);
        });
        let live = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.playback_go_live(cx);
        });

        let play_label = if status.mode == PlaybackMode::Playing {
            "⏸"
        } else {
            "▶"
        };

        let style_button = |button: Div, active: bool| {
            if active {
                button
                    .border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                button
                    .border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let restart_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("⏮")
                .on_mouse_down(MouseButton::Left, restart);
            style_button(base, false)
        };

        let prev_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("⏪")
                .on_mouse_down(MouseButton::Left, prev);
            style_button(base, false)
        };

        let play_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(play_label)
                .on_mouse_down(MouseButton::Left, toggle);
            style_button(base, status.mode == PlaybackMode::Playing)
        };

        let next_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("⏩")
                .on_mouse_down(MouseButton::Left, next);
            style_button(base, false)
        };

        let live_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Live")
                .on_mouse_down(MouseButton::Left, live);
            style_button(base, status.mode == PlaybackMode::Live)
        };

        let frame_summary = if status.total == 0 {
            "No frames captured yet".to_string()
        } else {
            let frame_num = status.index.saturating_add(1);
            let tick = status.current_tick.unwrap_or(0);
            format!("Frame {frame_num}/{} · Tick {tick}", status.total)
        };

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Deterministic playback"),
            )
            .child(div().flex().gap_2().children(vec![
                restart_button,
                prev_button,
                play_button,
                next_button,
                live_button,
            ]))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0xcbd5f5))
                    .child(frame_summary),
            )
    }

    fn render_accessibility_panel(&self, cx: &mut Context<Self>) -> Div {
        let palette_buttons: Vec<Div> = ColorPaletteMode::ALL
            .iter()
            .map(|mode| {
                let mode = *mode;
                let active = self.accessibility.palette == mode;
                let listener = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
                    this.set_palette(mode, cx);
                });
                let preview_color = apply_palette(rgba_from_hex(0x58ff94, 1.0), mode);
                let preview = div()
                    .w(px(16.0))
                    .h(px(8.0))
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0x1e293b))
                    .bg(preview_color);
                let base = div()
                    .rounded_md()
                    .border_1()
                    .px_2()
                    .py_1()
                    .text_xs()
                    .flex()
                    .gap_1()
                    .items_center()
                    .child(preview)
                    .child(mode.label())
                    .on_mouse_down(MouseButton::Left, listener);
                if active {
                    base.border_color(rgb(0x38bdf8))
                        .bg(rgb(0x1e3a8a))
                        .text_color(rgb(0xe0f2fe))
                } else {
                    base.border_color(rgb(0x1e293b))
                        .bg(rgb(0x111b2b))
                        .text_color(rgb(0xcbd5f5))
                }
            })
            .collect();

        let narration_button = {
            let listener = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
                this.toggle_narration(cx);
            });
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(if self.accessibility.narration_enabled {
                    "Narration: On"
                } else {
                    "Narration: Off"
                })
                .on_mouse_down(MouseButton::Left, listener);
            if self.accessibility.narration_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let mut bindings_rows: Vec<Div> = Vec::new();
        for (action, stroke) in self.bindings.iter() {
            let capturing = self.key_capture == Some(action);
            let label = action.label();
            let binding_text = if capturing {
                "Press new key...".to_string()
            } else {
                format_keystroke(&stroke)
            };
            let listener = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
                if this.key_capture == Some(action) {
                    this.key_capture = None;
                } else {
                    this.key_capture = Some(action);
                }
                cx.notify();
            });
            let button = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(if capturing { "Cancel" } else { "Rebind" })
                .on_mouse_down(MouseButton::Left, listener);
            bindings_rows.push(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(label))
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x94a3b8))
                            .child(binding_text),
                    )
                    .child(button),
            );
        }

        if self.key_capture.is_some() {
            bindings_rows.push(
                div()
                    .text_xs()
                    .text_color(rgb(0xf97316))
                    .child("Press a key to assign, or Esc to cancel."),
            );
        }

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Accessibility"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0xcbd5f5))
                    .child("Color palette"),
            )
            .child(div().flex().gap_2().children(palette_buttons))
            .child(div().text_xs().text_color(rgb(0xcbd5f5)).child("Narration"))
            .child(narration_button)
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0xcbd5f5))
                    .child("Key bindings"),
            )
            .children(bindings_rows)
    }

    fn render_inspector_detail(
        &self,
        detail: &AgentInspectorDetails,
        cx: &mut Context<Self>,
    ) -> Div {
        let sensors_preview: Vec<String> = detail
            .sensors
            .iter()
            .take(6)
            .enumerate()
            .map(|(idx, value)| format!("s{idx}:{value:.2}"))
            .collect();
        let outputs_preview: Vec<String> = detail
            .outputs
            .iter()
            .take(4)
            .enumerate()
            .map(|(idx, value)| format!("o{idx}:{value:.2}"))
            .collect();

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e3a8a))
            .bg(rgb(0x111b2b))
            .px_3()
            .py_3()
            .child(
                div()
                    .flex()
                    .justify_between()
                    .items_center()
                    .child(div().text_sm().child(detail.label.clone()))
                    .child(color_swatch(detail.color)),
            )
            .child(div().text_xs().text_color(rgb(0x64748b)).child(format!(
                "Agent {:?} · Gen {} · Age {}",
                detail.agent_id, detail.generation.0, detail.age
            )))
            .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                "Energy {:.2} · Health {:.2} · Spike {:.1}",
                detail.energy, detail.health, detail.spike_length
            )))
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
                "Pos ({:.1}, {:.1}) · Brain {}",
                detail.position.x, detail.position.y, detail.brain_descriptor
            )))
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
                "Mutation rates p{:.3} s{:.3}",
                detail.mutation_rates.primary, detail.mutation_rates.secondary
            )))
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
                "Traits smell {:.2} · sound {:.2} · hearing {:.2} · eye {:.2} · blood {:.2}",
                detail.trait_modifiers.smell,
                detail.trait_modifiers.sound,
                detail.trait_modifiers.hearing,
                detail.trait_modifiers.eye,
                detail.trait_modifiers.blood
            )))
            .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                "Sensors ({}/{}) {}",
                detail.sensors.len(),
                INPUT_SIZE,
                sensors_preview.join(", ")
            )))
            .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                "Outputs ({}/{}) {}",
                detail.outputs.len(),
                OUTPUT_SIZE,
                outputs_preview.join(", ")
            )))
            .child(self.render_mutation_controls(detail, cx))
    }

    fn render_mutation_controls(
        &self,
        detail: &AgentInspectorDetails,
        cx: &mut Context<Self>,
    ) -> Div {
        let agent_id = detail.agent_id;
        let primary_step = 0.0005_f32;
        let secondary_step = 0.01_f32;

        let inc_primary = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
            this.adjust_agent_mutation_rates(agent_id, primary_step, 0.0, cx);
        });
        let dec_primary = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
            this.adjust_agent_mutation_rates(agent_id, -primary_step, 0.0, cx);
        });

        let agent_id_secondary = detail.agent_id;
        let inc_secondary = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
            this.adjust_agent_mutation_rates(agent_id_secondary, 0.0, secondary_step, cx);
        });
        let dec_secondary = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
            this.adjust_agent_mutation_rates(agent_id_secondary, 0.0, -secondary_step, cx);
        });

        let primary_minus = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .px_2()
            .py_1()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .child("-")
            .on_mouse_down(MouseButton::Left, dec_primary);

        let primary_plus = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .px_2()
            .py_1()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .child("+")
            .on_mouse_down(MouseButton::Left, inc_primary);

        let secondary_minus = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .px_2()
            .py_1()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .child("-")
            .on_mouse_down(MouseButton::Left, dec_secondary);

        let secondary_plus = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .px_2()
            .py_1()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .child("+")
            .on_mouse_down(MouseButton::Left, inc_secondary);

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Mutation controls"),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child(format!("Primary {:.4}", detail.mutation_rates.primary)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .children(vec![primary_minus, primary_plus]),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child(format!("Secondary {:.3}", detail.mutation_rates.secondary)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .children(vec![secondary_minus, secondary_plus]),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x64748b))
                    .child("Adjusts focused agent mutation rates in ± steps."),
            )
    }

    fn render_canvas_placeholder(&self, snapshot: &HudSnapshot) -> Div {
        div()
            .flex()
            .flex_col()
            .flex_1()
            .rounded_xl()
            .border_1()
            .border_color(rgb(0x0ea5e9))
            .bg(rgb(0x0b1120))
            .shadow_lg()
            .p_4()
            .justify_center()
            .items_center()
            .gap_2()
            .child(
                div()
                    .text_lg()
                    .text_color(rgb(0x38bdf8))
                    .child("Canvas viewport"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x64748b))
                    .child("Rendering pipeline will paint agents and terrain here."),
            )
            .child(div().text_xs().text_color(rgb(0x38bdf8)).child(format!(
                "Latest tick #{}, {} agents",
                snapshot.tick, snapshot.agent_count
            )))
    }

    fn render_overlay(&self, snapshot: &HudSnapshot) -> Div {
        let mut lines: Vec<String> = if let Some(summary) = snapshot.summary.as_ref() {
            vec![
                format!("Tick {} (epoch {})", summary.tick, snapshot.epoch),
                format!(
                    "Agents {} • Births {} • Deaths {}",
                    summary.agent_count, summary.births, summary.deaths
                ),
                format!(
                    "Avg energy {:.2} • Avg health {:.2}",
                    summary.average_energy, summary.average_health
                ),
            ]
        } else {
            vec![format!("Tick {} • epoch {}", snapshot.tick, snapshot.epoch)]
        };
        let camera = self.camera_snapshot();
        lines.push(format!(
            "Zoom {:.2}× • Pan ({:.0}, {:.0})",
            camera.zoom, camera.offset_px.0, camera.offset_px.1
        ));
        lines.push(format!(
            "Simulation {} · speed {:.2}×",
            if snapshot.controls.paused {
                "Paused"
            } else {
                "Running"
            },
            snapshot.controls.speed_multiplier
        ));
        lines.push(format!(
            "Draw agents {} · food {}",
            if snapshot.controls.draw_agents {
                "ON"
            } else {
                "OFF"
            },
            if snapshot.controls.draw_food {
                "ON"
            } else {
                "OFF"
            }
        ));
        lines.push(format!(
            "Outline {}",
            if snapshot.controls.agent_outline {
                "ON"
            } else {
                "OFF"
            }
        ));
        lines.push(snapshot.controls.follow_mode.label().to_string());

        let inspector = &snapshot.inspector;
        if let Some(focus_id) = inspector.focus_id {
            lines.push(format!("Focus {:?}", focus_id));
        }
        if let Some(hover) = inspector.hovered.as_ref() {
            lines.push(format!("Hover {}", hover.label));
        }
        lines.push(format!(
            "Brush {} · radius {:.0} · Probe {}",
            if inspector.brush_enabled { "ON" } else { "OFF" },
            inspector.brush_radius,
            if inspector.probe_enabled { "ON" } else { "OFF" }
        ));
        lines.push(format!(
            "Palette {} · Narration {}",
            self.accessibility.palette.label(),
            if self.accessibility.narration_enabled {
                "ON"
            } else {
                "OFF"
            }
        ));

        if self.debug.enabled {
            lines.push(format!(
                "Debug overlay ON · velocity {} · sense {}",
                if self.debug.show_velocity {
                    "ON"
                } else {
                    "OFF"
                },
                if self.debug.show_sense_radius {
                    "ON"
                } else {
                    "OFF"
                }
            ));
        } else {
            lines.push("Debug overlay OFF".to_string());
        }
        lines.push(format!(
            "Persistence {} · interval {}",
            if inspector.persistence_enabled {
                "ON"
            } else {
                "OFF"
            },
            if inspector.persistence_enabled {
                inspector.persistence_interval.max(1)
            } else {
                inspector.persistence_cached_interval.max(1)
            }
        ));
        if let Some(action) = self.key_capture {
            lines.push(format!("Rebinding {}...", action.label()));
        }

        if self.shift_inspect {
            if let Some(hover) = inspector.hovered.as_ref() {
                lines.push(format!(
                    "Inspect {} · E {:.2} · H {:.2} · Age {} · Gen {}",
                    hover.label, hover.energy, hover.health, hover.age, hover.generation.0
                ));
            } else if let Some(detail) = inspector.focused.as_ref() {
                lines.push(format!(
                    "Inspect {:?} · E {:.2} · H {:.2} · Age {} · Gen {}",
                    detail.agent_id, detail.energy, detail.health, detail.age, detail.generation.0
                ));
                if let Some((best_idx, best_value)) = detail
                    .outputs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                {
                    lines.push(format!(
                        "Outputs max [{}]={:.2} · Mutation {:.3}/{:.3}",
                        best_idx,
                        best_value,
                        detail.mutation_rates.primary,
                        detail.mutation_rates.secondary
                    ));
                }
            } else {
                lines.push("Inspect overlay active (no agent)".to_string());
            }
        }

        let playback = self.playback.status();
        if playback.total > 0 {
            let mode_label = match playback.mode {
                PlaybackMode::Live => "LIVE",
                PlaybackMode::Paused => "PAUSED",
                PlaybackMode::Playing => "PLAY",
            };
            let current_tick = playback.current_tick.unwrap_or(snapshot.tick);
            let total_frames = playback.total;
            lines.push(format!(
                "Playback {mode_label} · frame {}/{} · tick {}",
                playback.index.saturating_add(1),
                total_frames,
                current_tick
            ));
        } else {
            lines.push("Playback LIVE · no frames".to_string());
        }

        let mut container = div().flex().gap_3().items_start();

        if let Some(vector_state) = VectorHudState::from_snapshot(snapshot) {
            let canvas_state = vector_state.clone();
            let heading_deg = vector_state.heading_rad.to_degrees();
            let cohesion = if vector_state.max_speed > f32::EPSILON {
                (vector_state.vector_magnitude / vector_state.max_speed * 100.0).clamp(0.0, 100.0)
            } else {
                0.0
            };
            let vector_canvas = canvas(
                move |_, _, _| canvas_state.clone(),
                move |bounds, state, window, _| paint_vector_hud(bounds, &state, window),
            )
            .w(px(148.0))
            .h(px(108.0));

            let vector_card = div()
                .flex()
                .flex_col()
                .gap_2()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x07111f))
                .px_3()
                .py_3()
                .child(vector_canvas)
                .child(
                    div()
                        .text_xs()
                        .text_color(rgb(0x64748b))
                        .child("Vector HUD gauges"),
                )
                .child(div().text_xs().text_color(rgb(0xa5b4fc)).child(format!(
                    "Avg speed {:.2} · heading {:+.0}° · cohesion {:>3.0}%",
                    vector_state.mean_speed, heading_deg, cohesion
                )));

            container = container.child(vector_card);
        }

        let text_column = div().flex().flex_col().gap_1().children(
            lines
                .into_iter()
                .map(|line| div().text_sm().text_color(rgb(0xe2e8f0)).child(line)),
        );

        container = container.child(text_column);

        div()
            .absolute()
            .top(px(12.0))
            .left(px(12.0))
            .bg(rgb(0x0b1120))
            .rounded_md()
            .shadow_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .px_3()
            .py_3()
            .child(container)
    }

    fn render_perf_overlay(&self, stats: PerfSnapshot) -> Div {
        let mut lines = Vec::new();

        if stats.sample_count == 0 {
            lines.push("Performance stats: collecting...".to_string());
        } else {
            lines.push(format!(
                "Frame {:.2} ms ({:.1} fps)",
                stats.latest_ms, stats.fps
            ));
            lines.push(format!(
                "Avg {:.2} ms · Min {:.2} · Max {:.2}",
                stats.average_ms, stats.min_ms, stats.max_ms
            ));
            lines.push(format!("Samples {}", stats.sample_count));
        }

        div()
            .absolute()
            .top(px(12.0))
            .right(px(12.0))
            .bg(rgb(0x111b2b))
            .border_1()
            .border_color(rgb(0x1e293b))
            .rounded_md()
            .shadow_md()
            .px_3()
            .py_2()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .flex()
            .flex_col()
            .gap_1()
            .children(lines.into_iter().map(|line| div().child(line)))
    }

    fn render_history_chart(&self, snapshot: &HudSnapshot) -> Div {
        const WIDTH: f32 = 220.0;
        const HEIGHT: f32 = 120.0;

        match HistoryChartData::from_entries(&snapshot.recent_history, WIDTH, HEIGHT) {
            Some(data) => {
                let chart_canvas = canvas(
                    move |_, _, _| data.clone(),
                    move |bounds, data, window, _| paint_history_chart(bounds, &data, window),
                )
                .w(px(WIDTH))
                .h(px(HEIGHT - 28.0))
                .flex_none();

                let legend = div()
                    .flex()
                    .gap_2()
                    .text_xs()
                    .text_color(rgb(0xcbd5f5))
                    .child(legend_item(rgb(0x38bdf8), "Agents"))
                    .child(legend_item(rgb(0x22c55e), "Births"))
                    .child(legend_item(rgb(0xef4444), "Deaths"));

                div()
                    .absolute()
                    .bottom(px(12.0))
                    .right(px(12.0))
                    .w(px(WIDTH))
                    .bg(rgb(0x0a1629))
                    .border_1()
                    .border_color(rgb(0x13304e))
                    .rounded_md()
                    .shadow_md()
                    .px_3()
                    .py_2()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(chart_canvas)
                    .child(legend)
            }
            None => div()
                .absolute()
                .bottom(px(12.0))
                .right(px(12.0))
                .bg(rgb(0x0a1629))
                .border_1()
                .border_color(rgb(0x13304e))
                .rounded_md()
                .shadow_md()
                .px_3()
                .py_2()
                .text_xs()
                .text_color(rgb(0x94a3b8))
                .child("History chart pending data"),
        }
    }

    fn render_footer(&self, snapshot: &HudSnapshot) -> Div {
        div()
            .flex()
            .justify_between()
            .items_center()
            .text_xs()
            .text_color(rgb(0x475569))
            .child(format!(
                "World {}×{} · History capacity {}",
                snapshot.world_size.0, snapshot.world_size.1, snapshot.history_capacity
            ))
            .child(format!(
                "Showing {} recent ticks",
                snapshot.recent_history.len()
            ))
    }

    fn metric_card(
        &self,
        label: &str,
        value: String,
        accent_hex: u32,
        detail: Option<String>,
        sparkline: Option<SparklineSeries>,
    ) -> Div {
        let accent = rgb(accent_hex);
        let accent_rgba = rgba_from_hex(accent_hex, 1.0);
        let badge_state = MetricBadgeState {
            accent: accent_rgba,
        };
        let badge = canvas(
            move |_, _, _| badge_state.clone(),
            move |bounds, state: MetricBadgeState, window, _| {
                paint_metric_badge(bounds, state, window);
            },
        )
        .w(px(28.0))
        .h(px(28.0));

        let mut card = div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_lg()
            .border_1()
            .border_color(accent)
            .bg(rgb(0x111827))
            .shadow_md()
            .p_4()
            .child(
                div().flex().justify_between().items_center().child(
                    div().flex().items_center().gap_2().child(badge).child(
                        div()
                            .text_xs()
                            .text_color(accent)
                            .child(label.to_uppercase()),
                    ),
                ),
            )
            .child(div().text_3xl().child(value));

        if let Some(detail_text) = detail {
            card = card.child(div().text_sm().text_color(rgb(0x94a3b8)).child(detail_text));
        }

        if let Some(series) = sparkline {
            let spark_state = SparklineState {
                values: series.normalized.clone(),
                accent: accent_rgba,
                trend: series.trend,
            };
            let spark_canvas = canvas(
                move |_, _, _| spark_state.clone(),
                move |bounds, state: SparklineState, window, _| {
                    paint_sparkline(bounds, state, window);
                },
            )
            .h(px(28.0))
            .w_full();

            card = card.child(
                div()
                    .mt(px(6.0))
                    .rounded_md()
                    .bg(rgb(0x0a1628))
                    .border_1()
                    .border_color(rgb(0x1f2a3d))
                    .px_3()
                    .py_2()
                    .child(spark_canvas),
            );
        }

        card
    }

    fn render_settings_panel(&self, cx: &mut Context<Self>) -> Div {
        // Modern, world-class settings panel with beautiful design
        let backdrop = div()
            .absolute()
            .inset_0()
            .bg(rgb(0x0f172a))
            .opacity(0.5)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    // Click backdrop to close panel (standard modal UX)
                    this.toggle_settings(cx);
                }),
            );

        let panel = div()
            .absolute()
            .top(px(0.0))
            .left(px(0.0))
            .bottom(px(0.0))
            .w(px(540.0))
            .bg(rgb(0x0f172a))
            .border_r_1()
            .border_color(rgb(0x334155))
            .shadow_xl()
            .flex()
            .flex_col()
            .overflow_hidden()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|_, _, _, cx| {
                    // Prevent clicks on panel from propagating to backdrop
                    cx.stop_propagation();
                }),
            )
            .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                // Handle keyboard input for search when settings panel is open
                let key = &event.keystroke.key;

                if event.keystroke.key == "backspace" {
                    // Remove last character from search
                    let mut query = this.settings_panel.search_query.clone();
                    query.pop();
                    this.update_search(query, cx);
                } else if event.keystroke.key == "escape" {
                    // Clear search or close panel
                    if !this.settings_panel.search_query.is_empty() {
                        this.clear_search(cx);
                    } else {
                        this.toggle_settings(cx);
                    }
                } else if key.len() == 1
                    && key
                        .chars()
                        .all(|c| c.is_alphanumeric() || c.is_whitespace() || "._-".contains(c))
                {
                    // Add alphanumeric characters, spaces, and common punctuation to search
                    let mut query = this.settings_panel.search_query.clone();
                    query.push_str(key);
                    this.update_search(query, cx);
                }
            }));

        let header = div()
            .flex()
            .items_center()
            .justify_between()
            .px_6()
            .py_4()
            .border_b_1()
            .border_color(rgb(0x334155))
            .bg(rgb(0x1e293b))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(div().text_2xl().child("⚙️"))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_xl()
                                    .text_color(rgb(0xf1f5f9))
                                    .child("Configuration"),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        div()
                                            .text_sm()
                                            .text_color(rgb(0x94a3b8))
                                            .child("Simulation parameters & settings"),
                                    )
                                    .child(
                                        div()
                                            .px_2()
                                            .py_1()
                                            .rounded_md()
                                            .bg(rgb(0x334155))
                                            .text_xs()
                                            .text_color(rgb(0x94a3b8))
                                            .child("Press , to toggle"),
                                    ),
                            ),
                    ),
            )
            .child(
                div()
                    .px_4()
                    .py_2()
                    .rounded_lg()
                    .bg(rgb(0x475569))
                    .text_base()
                    .text_color(rgb(0xf1f5f9))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(0x64748b)))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _, _, cx| {
                            this.toggle_settings(cx);
                        }),
                    )
                    .child("✕ Close"),
            );

        // REAL functional search bar - displays current search query and allows filtering
        let search_query = self.settings_panel.search_query.clone();
        let has_search = !search_query.is_empty();

        let search_bar = div()
            .px_6()
            .py_4()
            .border_b_1()
            .border_color(rgb(0x334155))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .px_4()
                    .py_3()
                    .rounded_lg()
                    .bg(rgb(0x1e293b))
                    .border_1()
                    .border_color(if has_search {
                        rgb(0x60a5fa)
                    } else {
                        rgb(0x475569)
                    })
                    .child(div().text_color(rgb(0x94a3b8)).child("🔍"))
                    .child(
                        div()
                            .flex_1()
                            .text_sm()
                            .text_color(if has_search {
                                rgb(0xf1f5f9)
                            } else {
                                rgb(0x94a3b8)
                            })
                            .child(if has_search {
                                search_query.clone()
                            } else {
                                "Type to search parameters... (start typing when panel is open)"
                                    .to_string()
                            }),
                    )
                    .when(has_search, |container| {
                        container.child(
                            div()
                                .px_2()
                                .py_1()
                                .rounded_md()
                                .bg(rgb(0x334155))
                                .text_xs()
                                .text_color(rgb(0x94a3b8))
                                .cursor_pointer()
                                .hover(|s: StyleRefinement| {
                                    s.bg(rgb(0x475569)).text_color(rgb(0xf1f5f9))
                                })
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(|this, _, _, cx| {
                                        this.clear_search(cx);
                                    }),
                                )
                                .child("✕ Clear"),
                        )
                    }),
            );

        // Scrollable container for categories with mouse wheel handling
        // Use cached dimensions (updated when panel opens or categories collapse/expand)
        let content_height = self.settings_panel.content_height;
        let viewport_height = self.settings_panel.viewport_height;
        let scroll_offset = self.settings_panel.scroll_offset;

        // Calculate scroll bounds (must match clamp_scroll logic for consistency)
        let max_scroll = (content_height - viewport_height).max(0.0);
        let has_scrollable_content = max_scroll > 1.0;

        let categories_content = div()
            .flex_1()
            .overflow_hidden()
            .relative()
            .px_6()
            .py_4()
            .on_scroll_wheel(cx.listener(move |this, event: &ScrollWheelEvent, _, cx| {
                // Handle scroll wheel to update offset
                let scroll_delta = match event.delta {
                    ScrollDelta::Pixels(delta) => f32::from(delta.y),
                    ScrollDelta::Lines(lines) => lines.y * 20.0, // ~20px per line
                };

                // Update scroll offset with proper bounds
                // Positive delta = scroll down = increase offset to show lower content
                this.settings_panel.scroll_offset += scroll_delta;
                this.settings_panel.clamp_scroll();
                cx.notify();
            }))
            .child(
                div()
                    .absolute()
                    .top(px(-scroll_offset))
                    .left(px(0.0))
                    .right(px(0.0))
                    .child(self.render_all_config_categories(cx)),
            )
            .when(has_scrollable_content, |node| {
                node.child(
                    // Visual scroll indicator at bottom
                    div()
                        .absolute()
                        .bottom(px(8.0))
                        .right(px(16.0))
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .bg(rgb(0x1e293b))
                        .border_1()
                        .border_color(rgb(0x475569))
                        .text_xs()
                        .text_color(rgb(0x94a3b8))
                        .child(format!(
                            "{:.0}%",
                            if max_scroll > 0.0 {
                                (scroll_offset / max_scroll * 100.0).min(100.0)
                            } else {
                                0.0
                            }
                        )),
                )
            });

        let panel_content = panel
            .child(header)
            .child(search_bar)
            .child(categories_content);

        div()
            .absolute()
            .inset_0()
            .child(backdrop)
            .child(panel_content)
    }

    fn render_all_config_categories(&self, cx: &mut Context<Self>) -> Div {
        let mut container = div().flex().flex_col().gap_4();

        for category in ConfigCategory::all() {
            container = container.child(self.render_config_category(category, cx));
        }

        container
    }

    fn render_config_category(&self, category: ConfigCategory, cx: &mut Context<Self>) -> Div {
        let is_collapsed = self.settings_panel.collapsed_categories.contains(&category);

        let header = div()
            .flex()
            .items_center()
            .justify_between()
            .cursor_pointer()
            .px_4()
            .py_3()
            .rounded_lg()
            .bg(rgb(0x1e293b))
            .border_1()
            .border_color(rgb(0x334155))
            .hover(|s| s.bg(rgb(0x334155)))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    this.toggle_category_collapse(category, cx);
                }),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(div().text_xl().child(category.icon()))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_base()
                                    .text_color(rgb(0xf1f5f9))
                                    .child(category.label()),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(0x94a3b8))
                                    .child(category.description()),
                            ),
                    ),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x94a3b8))
                    .child(if is_collapsed { "▶" } else { "▼" }),
            );

        let mut category_div = div().flex().flex_col().gap_2().child(header);

        if !is_collapsed {
            category_div = category_div.child(self.render_category_parameters(category, cx));
        }

        category_div
    }

    fn render_category_parameters(&self, category: ConfigCategory, cx: &mut Context<Self>) -> Div {
        // Read current config from world
        let config = if let Ok(world) = self.world.lock() {
            world.config().clone()
        } else {
            scriptbots_core::ScriptBotsConfig::default()
        };

        // Match on category and return filtered params directly - ULTRA CLEAN data-driven approach!
        // ONE central filter loop in render_filtered_params handles ALL 60+ parameters!
        match category {
            ConfigCategory::World => {
                let params = vec![
                    ("World Width", format!("{} units", config.world_width), "Horizontal extent of the simulation world"),
                    ("World Height", format!("{} units", config.world_height), "Vertical extent of the simulation world"),
                    ("Food Cell Size", format!("{} units", config.food_cell_size), "Size of each food grid cell"),
                    ("Initial Food", self.format_float(config.initial_food, 3), "Starting food in each cell"),
                    ("RNG Seed", config.rng_seed.map(|s| s.to_string()).unwrap_or_else(|| "Random".to_string()), "Random number generator seed"),
                    ("Chart Flush Interval", format!("{} ticks", config.chart_flush_interval), "History chart update frequency"),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Food => {
                let params = vec![
                    ("Respawn Interval", format!("{} ticks", config.food_respawn_interval), "Ticks between food respawn events"),
                    ("Respawn Amount", self.format_float(config.food_respawn_amount, 3), "Food added per respawn"),
                    ("Maximum Food", self.format_float(config.food_max, 3), "Maximum food per cell"),
                    ("Growth Rate", self.format_float(config.food_growth_rate, 4), "Logistic regrowth rate"),
                    ("Decay Rate", self.format_float(config.food_decay_rate, 4), "Proportional decay rate"),
                    ("Diffusion Rate", self.format_float(config.food_diffusion_rate, 3), "Neighbor exchange rate"),
                    ("Intake Rate", self.format_float(config.food_intake_rate, 3), "Agent food consumption rate"),
                    ("Sharing Radius", self.format_float(config.food_sharing_radius, 1), "Friendly neighbor sharing distance"),
                    ("Sharing Rate", self.format_float(config.food_sharing_rate, 3), "Energy fraction shared per neighbor"),
                    ("Transfer Rate", self.format_float(config.food_transfer_rate, 4), "Altruistic sharing amount"),
                    ("Sharing Distance", self.format_float(config.food_sharing_distance, 1), "Altruistic sharing threshold"),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Agent => {
                let params = vec![
                    ("Bot Speed", self.format_float(config.bot_speed, 2), "Base wheel speed multiplier"),
                    ("Bot Radius", self.format_float(config.bot_radius, 1), "Agent radius for collisions"),
                    ("Boost Multiplier", format!("{}×", self.format_float(config.boost_multiplier, 2)), "Speed boost when activated"),
                    ("Sense Radius", self.format_float(config.sense_radius, 1), "Perception range"),
                    ("Max Neighbors", self.format_float(config.sense_max_neighbors, 0), "Normalization factor"),
                    ("Carnivore Threshold", self.format_float(config.carnivore_threshold, 2), "Herbivore tendency cutoff for carnivores"),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Metabolism => {
                let params = vec![
                    ("Base Drain", self.format_float(config.metabolism_drain, 4), "Baseline energy cost"),
                    ("Movement Drain", self.format_float(config.movement_drain, 4), "Cost per velocity"),
                    ("Ramp Floor", self.format_float(config.metabolism_ramp_floor, 2), "Energy level for ramping"),
                    ("Ramp Rate", self.format_float(config.metabolism_ramp_rate, 4), "Additional drain above floor"),
                    ("Boost Penalty", self.format_float(config.metabolism_boost_penalty, 4), "Fixed boost cost"),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Temperature => {
                let params = vec![
                    ("Discomfort Rate", self.format_float(config.temperature_discomfort_rate, 4), "Health drain multiplier"),
                    ("Comfort Band", format!("±{}", self.format_float(config.temperature_comfort_band, 3)), "Tolerance threshold"),
                    ("Gradient Exponent", self.format_float(config.temperature_gradient_exponent, 2), "Pole-to-equator shaping"),
                    ("Discomfort Exp", self.format_float(config.temperature_discomfort_exponent, 2), "Discomfort scaling power"),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Reproduction => {
                let params = vec![
                    ("Energy Threshold", self.format_float(config.reproduction_energy_threshold, 2), "Required energy to reproduce"),
                    ("Energy Cost", self.format_float(config.reproduction_energy_cost, 2), "Parent's energy deduction"),
                    ("Cooldown", format!("{} ticks", config.reproduction_cooldown), "Ticks between reproductions"),
                    ("Herbivore Rate", format!("{}×", self.format_float(config.reproduction_rate_herbivore, 3)), "Herbivore multiplier"),
                    ("Carnivore Rate", format!("{}×", self.format_float(config.reproduction_rate_carnivore, 3)), "Carnivore multiplier"),
                    ("Child Energy", self.format_float(config.reproduction_child_energy, 2), "Starting energy for child"),
                    ("Spawn Jitter", format!("±{}", self.format_float(config.reproduction_spawn_jitter, 1)), "Position randomization"),
                    ("Spawn Back Distance", self.format_float(config.reproduction_spawn_back_distance, 1), "Child spawn distance behind parent"),
                    ("Color Jitter", format!("±{}", self.format_float(config.reproduction_color_jitter, 3)), "RGB mutation range"),
                    ("Mutation Scale", self.format_float(config.reproduction_mutation_scale, 4), "Trait mutation magnitude"),
                    ("Partner Chance", format!("{}%", self.format_float(config.reproduction_partner_chance * 100.0, 1)), "Crossover probability"),
                    ("Gene Log Capacity", format!("{}", config.reproduction_gene_log_capacity), "Max gene history entries"),
                    ("Meta-Mutation Chance", format!("{}%", self.format_float(config.reproduction_meta_mutation_chance * 100.0, 1)), "Mutation rate mutation chance"),
                    ("Meta-Mutation Scale", self.format_float(config.reproduction_meta_mutation_scale, 4), "Mutation rate change magnitude"),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Aging => {
                let params = vec![
                    ("Decay Start Age", format!("{} ticks", config.aging_health_decay_start), "Age when decay begins"),
                    ("Decay Rate", self.format_float(config.aging_health_decay_rate, 5), "Health loss per tick"),
                    ("Decay Max", self.format_float(config.aging_health_decay_max, 4), "Maximum decay per tick"),
                    ("Energy Penalty", format!("{}×", self.format_float(config.aging_energy_penalty_rate, 3)), "Health-to-energy conversion"),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Combat => {
                let params = vec![
                    ("Spike Radius", self.format_float(config.spike_radius, 1), "Base spike collision radius"),
                    ("Spike Damage", self.format_float(config.spike_damage, 2), "Damage at full power"),
                    ("Spike Energy Cost", self.format_float(config.spike_energy_cost, 4), "Energy cost to deploy"),
                    ("Min Length", self.format_float(config.spike_min_length, 2), "Minimum for damage"),
                    ("Alignment Cosine", self.format_float(config.spike_alignment_cosine, 2), "Directional threshold"),
                    ("Speed Bonus", format!("{}×", self.format_float(config.spike_speed_damage_bonus, 3)), "Velocity scaling"),
                    ("Length Bonus", format!("{}×", self.format_float(config.spike_length_damage_bonus, 3)), "Length scaling"),
                    ("Growth Rate", self.format_float(config.spike_growth_rate, 4), "Spike extension rate"),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Carcass => {
                let params = vec![
                    ("Distribution Radius", self.format_float(config.carcass_distribution_radius, 1), "Reward share distance"),
                    ("Health Reward", self.format_float(config.carcass_health_reward, 2), "Base health given"),
                    ("Reproduction Reward", self.format_float(config.carcass_reproduction_reward, 1), "Cooldown reduction"),
                    ("Neighbor Exponent", self.format_float(config.carcass_neighbor_exponent, 2), "Sharing normalization"),
                    ("Maturity Age", format!("{} ticks", config.carcass_maturity_age), "Full reward age"),
                    ("Energy Share", format!("{}%", self.format_float(config.carcass_energy_share_rate * 100.0, 1)), "Health-to-energy conversion"),
                    ("Indicator Scale", self.format_float(config.carcass_indicator_scale, 2), "Visual pulse intensity"),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Topography => {
                // Topography has a toggle - hybrid approach with toggle first, then readonly params
                let mut container = div()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .px_4()
                    .py_4()
                    .rounded_lg()
                    .bg(rgb(0x0f172a))
                    .border_1()
                    .border_color(rgb(0x1e293b));

                // Add toggle (not filterable - always shown)
                container = container.child(self.render_param_toggle(
                    "Enabled",
                    config.topography_enabled,
                    "Enable terrain elevation effects",
                    cx,
                ));

                // Add filterable readonly params
                let params = vec![
                    ("Speed Gain", self.format_float(config.topography_speed_gain, 3), "Downhill boost per unit slope"),
                    ("Energy Penalty", self.format_float(config.topography_energy_penalty, 4), "Uphill cost per unit slope"),
                ];

                for (label, value, desc) in params {
                    if self.matches_search(label) || self.matches_search(&value) || self.matches_search(desc) {
                        container = container.child(self.render_param_readonly(label, &value, desc));
                    }
                }

                container
            }

            ConfigCategory::Population => {
                let params = vec![
                    ("Minimum Population", format!("{}", config.population_minimum), "Auto-seed threshold"),
                    ("Spawn Interval", format!("{} ticks", config.population_spawn_interval), "Ticks between spawns"),
                    ("Spawn Count", format!("{}", config.population_spawn_count), "Agents per interval"),
                    ("Crossover Chance", format!("{}%", self.format_float(config.population_crossover_chance * 100.0, 1)), "Breed vs. random spawn"),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Persistence => {
                let params = vec![
                    ("Interval", format!("{} ticks", config.persistence_interval), "Database flush frequency"),
                    ("History Capacity", format!("{}", config.history_capacity), "In-memory tick summaries"),
                ];
                self.render_filtered_params(params)
            }
        }
    }

    /// Helper to safely format floats with NaN/Inf guards
    fn format_float(&self, value: f32, precision: usize) -> String {
        if !value.is_finite() {
            if value.is_nan() {
                "NaN".to_string()
            } else if value.is_infinite() {
                if value.is_sign_positive() {
                    "∞".to_string()
                } else {
                    "-∞".to_string()
                }
            } else {
                "ERR".to_string()
            }
        } else {
            format!("{:.prec$}", value, prec = precision)
        }
    }

    fn render_param_readonly(&self, label: &str, value: &str, description: &str) -> Div {
        let label_owned = label.to_string();
        let value_owned = value.to_string();
        let description_owned = description.to_string();

        div()
            .flex()
            .flex_col()
            .gap_2()
            .py_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(div().text_sm().text_color(rgb(0xf1f5f9)).child(label_owned))
                    .child(
                        div()
                            .text_sm()
                            .text_color(rgb(0x60a5fa))
                            .child(value_owned),
                    ),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x94a3b8))
                    .child(description_owned),
            )
    }

    fn render_param_toggle(
        &self,
        label: &str,
        enabled: bool,
        description: &str,
        _cx: &mut Context<Self>,
    ) -> Div {
        let label_owned = label.to_string();
        let description_owned = description.to_string();

        div()
            .flex()
            .flex_col()
            .gap_2()
            .py_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(div().text_sm().text_color(rgb(0xf1f5f9)).child(label_owned))
                    .child(
                        div()
                            .px_3()
                            .py_1()
                            .rounded_md()
                            .bg(if enabled {
                                rgb(0x166534)
                            } else {
                                rgb(0x7f1d1d)
                            })
                            .border_1()
                            .border_color(if enabled {
                                rgb(0x22c55e)
                            } else {
                                rgb(0xef4444)
                            })
                            .text_sm()
                            .text_color(if enabled {
                                rgb(0x86efac)
                            } else {
                                rgb(0xfca5a5)
                            })
                            .child(if enabled { "ON" } else { "OFF" }),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x64748b))
                    .child(description_owned),
            )
    }
}

impl Render for SimulationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.perf.begin_frame();

        let live_snapshot = self.snapshot();
        let snapshot = self.playback.snapshot_for_render(live_snapshot);

        let mut content = div()
            .size_full()
            .relative()
            .flex()
            .flex_col()
            .bg(rgb(0x0f172a))
            .text_color(rgb(0xf8fafc))
            .p_6()
            .gap_4()
            .child(self.render_header(&snapshot))
            .child(self.render_summary(&snapshot))
            .child(self.render_analytics_panel(&snapshot))
            .child(
                div()
                    .flex()
                    .gap_4()
                    .flex_1()
                    .child(self.render_history(&snapshot))
                    .child(self.render_canvas(&snapshot, cx))
                    .child(self.render_inspector(&snapshot, cx)),
            )
            .child(self.render_footer(&snapshot))
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
                this.handle_key_down(event, cx);
            }));

        let perf_snapshot = self.perf.end_frame();
        self.last_perf = perf_snapshot;

        #[cfg(feature = "audio")]
        self.update_audio(&snapshot);

        content = content.child(self.render_perf_overlay(perf_snapshot));

        if self.settings_panel.open {
            content = content.child(self.render_settings_panel(cx));
        }

        content
    }
}

#[derive(Default, Clone)]
struct HudSnapshot {
    tick: u64,
    epoch: u64,
    is_closed: bool,
    world_size: (u32, u32),
    history_capacity: usize,
    agent_count: usize,
    summary: Option<HudMetrics>,
    analytics: Option<HudAnalytics>,
    recent_history: Vec<HudHistoryEntry>,
    render_frame: Option<RenderFrame>,
    inspector: InspectorSnapshot,
    controls: ControlsSnapshot,
    perf: PerfSnapshot,
}

#[derive(Clone)]
struct HudAnalytics {
    tick: u64,
    carnivores: usize,
    herbivores: usize,
    hybrids: usize,
    carnivore_avg_energy: f64,
    herbivore_avg_energy: f64,
    hybrid_avg_energy: f64,
    age_mean: f64,
    age_max: f64,
    boost_count: usize,
    boost_ratio: f64,
    reproduction_counter_mean: f64,
    temperature_preference_mean: f64,
    temperature_preference_stddev: f64,
    temperature_discomfort_mean: f64,
    temperature_discomfort_stddev: f64,
    food_total: f64,
    food_mean: f64,
    food_stddev: f64,
    food_delta_mean: f64,
    food_delta_mean_abs: f64,
    mutation_primary_mean: f64,
    mutation_primary_stddev: f64,
    mutation_secondary_mean: f64,
    mutation_secondary_stddev: f64,
    behavior_sensor_mean: f64,
    behavior_sensor_entropy: f64,
    behavior_output_mean: f64,
    behavior_output_entropy: f64,
    generation_mean: f64,
    generation_max: f64,
    deaths_combat_carnivore: usize,
    deaths_combat_herbivore: usize,
    deaths_starvation: usize,
    deaths_aging: usize,
    deaths_unknown: usize,
    deaths_total: usize,
    births_total: usize,
    births_hybrid: usize,
    births_hybrid_ratio: f64,
    brain_shares: Vec<BrainShareEntry>,
}

#[derive(Clone)]
struct BrainShareEntry {
    label: String,
    count: usize,
    avg_energy: f64,
}

fn parse_analytics(
    tick: u64,
    agent_count: usize,
    readings: &[MetricReading],
) -> Option<HudAnalytics> {
    if readings.is_empty() {
        return None;
    }

    let mut metrics = HashMap::with_capacity(readings.len());
    for reading in readings {
        metrics.insert(reading.name.clone(), reading.value);
    }

    let value = |key: &str| metrics.get(key).copied();
    let as_count = |key: &str| value(key).unwrap_or(0.0).max(0.0).round() as usize;
    let carnivores = as_count("population.carnivore.count");
    let herbivores = as_count("population.herbivore.count");
    let hybrids = as_count("population.hybrid.count");

    let carnivore_avg_energy = value("population.carnivore.avg_energy").unwrap_or(0.0);
    let herbivore_avg_energy = value("population.herbivore.avg_energy").unwrap_or(0.0);
    let hybrid_avg_energy = value("population.hybrid.avg_energy").unwrap_or(0.0);
    let age_mean = value("population.age.mean").unwrap_or(0.0);
    let age_max = value("population.age.max").unwrap_or(0.0);
    let boost_count = as_count("behavior.boost.count");
    let boost_ratio = value("behavior.boost.ratio").unwrap_or_else(|| {
        if agent_count > 0 {
            boost_count as f64 / agent_count as f64
        } else {
            0.0
        }
    });
    let reproduction_counter_mean = value("reproduction.counter.mean").unwrap_or(0.0);
    let temperature_preference_mean = value("temperature.preference.mean").unwrap_or(0.0);
    let temperature_preference_stddev = value("temperature.preference.stddev").unwrap_or(0.0);

    let food_total = value("food.total").unwrap_or(0.0);
    let food_mean = value("food.mean").unwrap_or(0.0);
    let food_stddev = value("food.stddev").unwrap_or(0.0);
    let food_delta_mean = value("food_delta.mean").unwrap_or(0.0);
    let food_delta_mean_abs = value("food_delta.mean_abs").unwrap_or(0.0);
    let temperature_discomfort_mean = value("temperature.discomfort.mean").unwrap_or(0.0);
    let temperature_discomfort_stddev = value("temperature.discomfort.stddev").unwrap_or(0.0);
    let generation_mean = value("population.generation.mean").unwrap_or(0.0);
    let generation_max = value("population.generation.max").unwrap_or(0.0);

    let mutation_primary_mean = value("mutation.primary.mean").unwrap_or(0.0);
    let mutation_primary_stddev = value("mutation.primary.stddev").unwrap_or(0.0);
    let mutation_secondary_mean = value("mutation.secondary.mean").unwrap_or(0.0);
    let mutation_secondary_stddev = value("mutation.secondary.stddev").unwrap_or(0.0);

    let behavior_sensor_mean = value("behavior.sensors.mean").unwrap_or(0.0);
    let behavior_sensor_entropy = value("behavior.sensors.entropy").unwrap_or(0.0);
    let behavior_output_mean = value("behavior.outputs.mean").unwrap_or(0.0);
    let behavior_output_entropy = value("behavior.outputs.entropy").unwrap_or(0.0);

    let mut brain_map: HashMap<String, BrainShareEntry> = HashMap::new();
    for (name, &metric_value) in &metrics {
        if let Some(rest) = name.strip_prefix("brain.population.") {
            if let Some(label) = rest.strip_suffix(".count") {
                let entry = brain_map
                    .entry(label.to_string())
                    .or_insert(BrainShareEntry {
                        label: label.to_string(),
                        count: 0,
                        avg_energy: 0.0,
                    });
                entry.count = metric_value.max(0.0).round() as usize;
                continue;
            }
            if let Some(label) = rest.strip_suffix(".avg_energy") {
                let entry = brain_map
                    .entry(label.to_string())
                    .or_insert(BrainShareEntry {
                        label: label.to_string(),
                        count: 0,
                        avg_energy: 0.0,
                    });
                entry.avg_energy = metric_value;
            }
        }
    }

    let mut brain_shares: Vec<BrainShareEntry> = brain_map.into_values().collect();
    brain_shares.sort_by(|a, b| b.count.cmp(&a.count));

    let deaths_combat_carnivore = as_count("mortality.combat_carnivore.count");
    let deaths_combat_herbivore = as_count("mortality.combat_herbivore.count");
    let deaths_starvation = as_count("mortality.starvation.count");
    let deaths_aging = as_count("mortality.aging.count");
    let deaths_unknown = as_count("mortality.unknown.count");
    let deaths_total = value("mortality.total.count")
        .map(|v| v.max(0.0).round() as usize)
        .unwrap_or(
            deaths_combat_carnivore
                + deaths_combat_herbivore
                + deaths_starvation
                + deaths_aging
                + deaths_unknown,
        );
    let births_total = as_count("births.total.count");
    let births_hybrid = as_count("births.hybrid.count");
    let births_hybrid_ratio = value("births.hybrid.ratio").unwrap_or_else(|| {
        if births_total > 0 {
            births_hybrid as f64 / births_total as f64
        } else {
            0.0
        }
    });

    Some(HudAnalytics {
        tick,
        carnivores,
        herbivores,
        hybrids,
        carnivore_avg_energy,
        herbivore_avg_energy,
        hybrid_avg_energy,
        age_mean,
        age_max,
        boost_count,
        boost_ratio,
        reproduction_counter_mean,
        temperature_preference_mean,
        temperature_preference_stddev,
        temperature_discomfort_mean,
        temperature_discomfort_stddev,
        food_total,
        food_mean,
        food_stddev,
        food_delta_mean,
        food_delta_mean_abs,
        mutation_primary_mean,
        mutation_primary_stddev,
        mutation_secondary_mean,
        mutation_secondary_stddev,
        behavior_sensor_mean,
        behavior_sensor_entropy,
        behavior_output_mean,
        behavior_output_entropy,
        generation_mean,
        generation_max,
        deaths_combat_carnivore,
        deaths_combat_herbivore,
        deaths_starvation,
        deaths_aging,
        deaths_unknown,
        deaths_total,
        births_total,
        births_hybrid,
        births_hybrid_ratio,
        brain_shares,
    })
}

#[derive(Clone)]
struct VectorHudState {
    population_ratio: f32,
    energy_ratio: f32,
    births_ratio: f32,
    deaths_ratio: f32,
    tick_phase: f32,
    mean_speed: f32,
    vector_magnitude: f32,
    max_speed: f32,
    heading_rad: f32,
}

impl VectorHudState {
    fn from_snapshot(snapshot: &HudSnapshot) -> Option<Self> {
        let metrics = snapshot.summary.as_ref()?;

        let max_agents = snapshot
            .recent_history
            .iter()
            .map(|entry| entry.agent_count)
            .chain(std::iter::once(metrics.agent_count))
            .max()
            .unwrap_or(metrics.agent_count)
            .max(1);

        let max_births = snapshot
            .recent_history
            .iter()
            .map(|entry| entry.births)
            .chain(std::iter::once(metrics.births))
            .max()
            .unwrap_or(metrics.births)
            .max(1);

        let max_deaths = snapshot
            .recent_history
            .iter()
            .map(|entry| entry.deaths)
            .chain(std::iter::once(metrics.deaths))
            .max()
            .unwrap_or(metrics.deaths)
            .max(1);

        let mut energy_max = metrics.average_energy.max(0.0);
        for entry in &snapshot.recent_history {
            energy_max = energy_max.max(entry.average_energy);
        }
        if energy_max <= f32::EPSILON {
            energy_max = 1.0;
        }

        let (mean_speed, vector_magnitude, max_speed, heading_rad) = snapshot
            .render_frame
            .as_ref()
            .map(|frame| {
                if frame.agents.is_empty() {
                    return (0.0, 0.0, 1.0, 0.0);
                }

                let mut sum_vx: f32 = 0.0;
                let mut sum_vy: f32 = 0.0;
                let mut sum_speed: f32 = 0.0;
                let mut max_speed: f32 = 0.0;

                for agent in &frame.agents {
                    let vel = agent.velocity;
                    let speed = (vel.vx * vel.vx + vel.vy * vel.vy).sqrt();
                    sum_vx += vel.vx;
                    sum_vy += vel.vy;
                    sum_speed += speed;
                    max_speed = max_speed.max(speed);
                }

                let count = frame.agents.len() as f32;
                let safe_count = if count <= f32::EPSILON { 1.0 } else { count };
                let avg_vx = sum_vx / safe_count;
                let avg_vy = sum_vy / safe_count;
                let mean_speed = sum_speed / safe_count;
                let vector_magnitude = (avg_vx * avg_vx + avg_vy * avg_vy).sqrt();
                let heading_rad = if vector_magnitude > f32::EPSILON {
                    avg_vy.atan2(avg_vx)
                } else {
                    0.0
                };
                let max_speed_final = max_speed.max(mean_speed).max(1e-3);

                (mean_speed, vector_magnitude, max_speed_final, heading_rad)
            })
            .unwrap_or((0.0, 0.0, 1.0, 0.0));

        Some(Self {
            population_ratio: metrics.agent_count as f32 / max_agents as f32,
            energy_ratio: (metrics.average_energy / energy_max).clamp(0.0, 1.0),
            births_ratio: metrics.births as f32 / max_births as f32,
            deaths_ratio: metrics.deaths as f32 / max_deaths as f32,
            tick_phase: (snapshot.tick % 960) as f32 / 960.0,
            mean_speed,
            vector_magnitude,
            max_speed,
            heading_rad,
        })
    }
}

#[derive(Clone)]
struct HudMetrics {
    tick: u64,
    agent_count: usize,
    births: usize,
    deaths: usize,
    total_energy: f32,
    average_energy: f32,
    average_health: f32,
}

impl HudMetrics {
    fn net_growth(&self) -> isize {
        self.births as isize - self.deaths as isize
    }
}

impl From<&TickSummary> for HudMetrics {
    fn from(summary: &TickSummary) -> Self {
        Self {
            tick: summary.tick.0,
            agent_count: summary.agent_count,
            births: summary.births,
            deaths: summary.deaths,
            total_energy: summary.total_energy,
            average_energy: summary.average_energy,
            average_health: summary.average_health,
        }
    }
}

#[derive(Clone)]
struct SparklineSeries {
    normalized: Vec<f32>,
    trend: f32,
}

#[derive(Clone)]
struct SparklineState {
    values: Vec<f32>,
    accent: Rgba,
    trend: f32,
}

#[derive(Clone)]
struct MetricBadgeState {
    accent: Rgba,
}

#[derive(Clone, Copy)]
struct HeaderBadgeState {
    phase: f32,
    palette: ColorPaletteMode,
}

#[derive(Clone, Copy)]
struct DebugOverlayState {
    enabled: bool,
    show_velocity: bool,
    show_sense_radius: bool,
}

impl Default for DebugOverlayState {
    fn default() -> Self {
        Self {
            enabled: false,
            show_velocity: true,
            show_sense_radius: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum FollowMode {
    #[default]
    Off,
    Selected,
    Oldest,
}

impl FollowMode {
    fn label(self) -> &'static str {
        match self {
            FollowMode::Off => "Follow off",
            FollowMode::Selected => "Follow selected",
            FollowMode::Oldest => "Follow oldest",
        }
    }
}

#[derive(Clone)]
struct SimulationControls {
    paused: bool,
    draw_agents: bool,
    draw_food: bool,
    speed_multiplier: f32,
    follow_mode: FollowMode,
    agent_outline: bool,
}

impl Default for SimulationControls {
    fn default() -> Self {
        Self {
            paused: false,
            draw_agents: true,
            draw_food: true,
            speed_multiplier: 1.0,
            follow_mode: FollowMode::Off,
            agent_outline: false,
        }
    }
}

impl SimulationControls {
    fn snapshot(&self) -> ControlsSnapshot {
        ControlsSnapshot {
            paused: self.paused,
            draw_agents: self.draw_agents,
            draw_food: self.draw_food,
            speed_multiplier: self.speed_multiplier,
            follow_mode: self.follow_mode,
            agent_outline: self.agent_outline,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ControlsSnapshot {
    paused: bool,
    draw_agents: bool,
    draw_food: bool,
    speed_multiplier: f32,
    follow_mode: FollowMode,
    agent_outline: bool,
}

fn sparkline_from_history<F>(history: &[HudHistoryEntry], map: F) -> Option<SparklineSeries>
where
    F: Fn(&HudHistoryEntry) -> f32,
{
    if history.len() < 2 {
        return None;
    }
    let raw: Vec<f32> = history.iter().map(map).collect();
    if raw.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let first = raw.first().copied()?;
    let last = raw.last().copied()?;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in &raw {
        min = min.min(*value);
        max = max.max(*value);
    }
    let span = (max - min).abs().max(1e-5);
    let normalized: Vec<f32> = if span <= 1e-5 {
        vec![0.5; raw.len()]
    } else {
        raw.iter()
            .map(|v| ((v - min) / span).clamp(0.0, 1.0))
            .collect()
    };

    Some(SparklineSeries {
        normalized,
        trend: last - first,
    })
}

#[derive(Clone)]
struct HudHistoryEntry {
    tick: u64,
    agent_count: usize,
    births: usize,
    deaths: usize,
    average_energy: f32,
    average_health: f32,
}

impl HudHistoryEntry {
    fn net_growth(&self) -> isize {
        self.births as isize - self.deaths as isize
    }
}

impl From<TickSummary> for HudHistoryEntry {
    fn from(summary: TickSummary) -> Self {
        Self {
            tick: summary.tick.0,
            agent_count: summary.agent_count,
            births: summary.births,
            deaths: summary.deaths,
            average_energy: summary.average_energy,
            average_health: summary.average_health,
        }
    }
}

#[derive(Clone)]
struct InspectorState {
    focused_agent: Option<AgentId>,
    hovered_agent: Option<AgentId>,
    brush_enabled: bool,
    brush_radius: f32,
    probe_enabled: bool,
    persistence_last_enabled: u32,
}

impl Default for InspectorState {
    fn default() -> Self {
        Self {
            focused_agent: None,
            hovered_agent: None,
            brush_enabled: false,
            brush_radius: 48.0,
            probe_enabled: false,
            persistence_last_enabled: 60,
        }
    }
}

#[derive(Default, Clone)]
struct InspectorSnapshot {
    focused: Option<AgentInspectorDetails>,
    selected: Vec<AgentListEntry>,
    hovered: Option<AgentListEntry>,
    focus_id: Option<AgentId>,
    total_agents: usize,
    brush_enabled: bool,
    brush_radius: f32,
    probe_enabled: bool,
    persistence_enabled: bool,
    persistence_interval: u32,
    persistence_cached_interval: u32,
}

/// Settings panel state for configuration management
#[derive(Clone)]
struct SettingsPanelState {
    open: bool,
    /// Search query for filtering parameters (future feature)
    #[allow(dead_code)]
    search_query: String,
    /// Currently active/focused category (future feature for single-category view)
    #[allow(dead_code)]
    active_category: Option<ConfigCategory>,
    /// List of collapsed categories (hidden parameters)
    collapsed_categories: Vec<ConfigCategory>,
    /// List of parameter names that have been modified (future feature for change tracking)
    #[allow(dead_code)]
    modified_params: Vec<String>,
    /// Name of current preset configuration (future feature for save/load)
    #[allow(dead_code)]
    preset_name: String,
    /// Vertical scroll offset for categories content (in pixels)
    scroll_offset: f32,
    /// Cached total content height for scroll bounds calculation
    content_height: f32,
    /// Cached viewport height for scroll bounds calculation
    viewport_height: f32,
}

impl Default for SettingsPanelState {
    fn default() -> Self {
        Self {
            open: false,
            search_query: String::new(),
            active_category: None,
            collapsed_categories: Vec::new(),
            modified_params: Vec::new(),
            preset_name: "Default".to_string(),
            scroll_offset: 0.0,
            content_height: 0.0,
            // CRITICAL: Must use conservative (small) value to prevent blocking content access
            // Default window: 720px - chrome (132px) = 588px actual viewport
            // Use 400px to ensure scrolling works even on small windows (600px)
            // Trade-off: Shows blank space on large windows vs. blocking content (blank space is acceptable)
            viewport_height: 400.0,
        }
    }
}

impl SettingsPanelState {
    /// Calculate maximum scroll offset based on content and viewport heights
    fn max_scroll_offset(&self) -> f32 {
        (self.content_height - self.viewport_height).max(0.0)
    }

    /// Clamp scroll offset to valid bounds
    fn clamp_scroll(&mut self) {
        self.scroll_offset = self.scroll_offset.clamp(0.0, self.max_scroll_offset());
    }

    /// Calculate accurate content height based on actual parameter counts
    /// Uses precise measurements from rendered categories
    fn estimate_content_height(&self, _total_categories: usize) -> f32 {
        let mut total_height = 0.0;

        for category in ConfigCategory::all() {
            let is_collapsed = self.collapsed_categories.contains(&category);

            if is_collapsed {
                // Collapsed: header (70px) + gap (16px) = 86px
                total_height += 86.0;
            } else {
                // Expanded: header (70px) + params container + gap (16px)
                // Params container: padding (32px) + params + gaps between params
                // Each param: ~44px + 12px gap = 56px per param
                let param_count = category.parameter_count();
                let params_height = 32.0 + (param_count as f32 * 56.0);
                total_height += 70.0 + params_height + 16.0;
            }
        }

        // Add top/bottom padding for categories container (py_4 = 16px each side = 32px total)
        total_height + 32.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ConfigCategory {
    World,
    Food,
    Agent,
    Metabolism,
    Temperature,
    Reproduction,
    Aging,
    Combat,
    Carcass,
    Topography,
    Population,
    Persistence,
}

impl ConfigCategory {
    fn label(self) -> &'static str {
        match self {
            ConfigCategory::World => "World",
            ConfigCategory::Food => "Food Dynamics",
            ConfigCategory::Agent => "Agent Behavior",
            ConfigCategory::Metabolism => "Metabolism & Energy",
            ConfigCategory::Temperature => "Temperature",
            ConfigCategory::Reproduction => "Reproduction",
            ConfigCategory::Aging => "Aging",
            ConfigCategory::Combat => "Combat & Spikes",
            ConfigCategory::Carcass => "Carcass Sharing",
            ConfigCategory::Topography => "Topography",
            ConfigCategory::Population => "Population Control",
            ConfigCategory::Persistence => "Data Persistence",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            ConfigCategory::World => "🌍",
            ConfigCategory::Food => "🌾",
            ConfigCategory::Agent => "🤖",
            ConfigCategory::Metabolism => "⚡",
            ConfigCategory::Temperature => "🌡️",
            ConfigCategory::Reproduction => "🧬",
            ConfigCategory::Aging => "⏳",
            ConfigCategory::Combat => "⚔️",
            ConfigCategory::Carcass => "🦴",
            ConfigCategory::Topography => "⛰️",
            ConfigCategory::Population => "👥",
            ConfigCategory::Persistence => "💾",
        }
    }

    fn description(self) -> &'static str {
        match self {
            ConfigCategory::World => "World dimensions and grid configuration",
            ConfigCategory::Food => "Food spawning, growth, decay, and diffusion",
            ConfigCategory::Agent => "Agent movement, sensing, and base behavior",
            ConfigCategory::Metabolism => "Energy consumption and metabolism mechanics",
            ConfigCategory::Temperature => "Environmental temperature and agent comfort",
            ConfigCategory::Reproduction => "Reproduction mechanics, mutation, and crossover",
            ConfigCategory::Aging => "Age-related health decay and penalties",
            ConfigCategory::Combat => "Spike damage, energy costs, and combat mechanics",
            ConfigCategory::Carcass => "Death rewards and resource distribution",
            ConfigCategory::Topography => "Terrain elevation effects on movement",
            ConfigCategory::Population => "Population maintenance and seeding",
            ConfigCategory::Persistence => "Database storage and analytics",
        }
    }

    fn all() -> Vec<Self> {
        vec![
            ConfigCategory::World,
            ConfigCategory::Food,
            ConfigCategory::Agent,
            ConfigCategory::Metabolism,
            ConfigCategory::Temperature,
            ConfigCategory::Reproduction,
            ConfigCategory::Aging,
            ConfigCategory::Combat,
            ConfigCategory::Carcass,
            ConfigCategory::Topography,
            ConfigCategory::Population,
            ConfigCategory::Persistence,
        ]
    }

    /// Returns the exact number of parameters displayed in this category
    fn parameter_count(self) -> usize {
        match self {
            ConfigCategory::World => 5, // ACTUAL: width, height, food_cell_size, initial_food, rng_seed
            ConfigCategory::Food => 11, // ACTUAL: respawn_interval, respawn_amount, max, growth_rate, decay_rate, diffusion_rate, intake_rate, sharing_radius, sharing_rate, transfer_rate, sharing_distance
            ConfigCategory::Agent => 6, // ACTUAL: bot_speed, bot_radius, boost_multiplier, sense_radius, sense_max_neighbors, carnivore_threshold
            ConfigCategory::Metabolism => 5, // ACTUAL: drain, movement_drain, ramp_floor, ramp_rate, boost_penalty
            ConfigCategory::Temperature => 4, // ACTUAL: discomfort_rate, comfort_band, gradient_exponent, discomfort_exponent
            ConfigCategory::Reproduction => 14, // ACTUAL: energy_threshold, energy_cost, cooldown, herbivore_rate, carnivore_rate, child_energy, spawn_jitter, spawn_back_distance, color_jitter, mutation_scale, partner_chance, gene_log_capacity, meta_mutation_chance, meta_mutation_scale
            ConfigCategory::Aging => 4, // ACTUAL: decay_start, decay_rate, decay_max, energy_penalty
            ConfigCategory::Combat => 8, // ACTUAL: spike_radius, spike_damage, spike_energy_cost, min_length, alignment_cosine, speed_bonus, length_bonus, growth_rate
            ConfigCategory::Carcass => 7, // ACTUAL: distribution_radius, health_reward, reproduction_reward, neighbor_exponent, maturity_age, energy_share, indicator_scale
            ConfigCategory::Topography => 3, // ACTUAL: enabled, speed_gain, energy_penalty
            ConfigCategory::Population => 4, // ACTUAL: minimum, spawn_interval, spawn_count, crossover_chance
            ConfigCategory::Persistence => 2, // ACTUAL: interval, enabled (2 params visible in render)
        }
    }
}

#[derive(Clone)]
struct SelectionEvent {
    tick: u64,
    kind: SelectionEventKind,
    total_selected: usize,
    sample_ids: Vec<AgentId>,
}

#[derive(Clone, Copy)]
enum SelectionEventKind {
    Clear,
    SelectAll,
    Click,
    Focus,
}

impl SelectionEventKind {
    fn label(&self) -> &'static str {
        match self {
            SelectionEventKind::Clear => "Cleared",
            SelectionEventKind::SelectAll => "Selected all",
            SelectionEventKind::Click => "Selection changed",
            SelectionEventKind::Focus => "Focus updated",
        }
    }
}

impl InspectorSnapshot {
    fn from_world(world: &WorldState, inspector: &InspectorState) -> Self {
        let mut snapshot = InspectorSnapshot {
            total_agents: world.agent_count(),
            brush_enabled: inspector.brush_enabled,
            brush_radius: inspector.brush_radius,
            probe_enabled: inspector.probe_enabled,
            persistence_cached_interval: inspector.persistence_last_enabled,
            ..InspectorSnapshot::default()
        };

        let config = world.config();
        snapshot.persistence_interval = config.persistence_interval;
        snapshot.persistence_enabled = config.persistence_interval > 0;
        if !snapshot.persistence_enabled && snapshot.persistence_interval > 0 {
            snapshot.persistence_cached_interval = snapshot.persistence_interval.max(1);
        }

        let arena = world.agents();
        let runtime = world.runtime();
        let columns = arena.columns();

        let mut selected = Vec::new();
        let mut hovered: Option<AgentListEntry> = None;

        for (row, agent_id) in arena.iter_handles().enumerate() {
            if let Some(agent_runtime) = runtime.get(agent_id) {
                let entry = AgentListEntry::from_world(row, agent_id, agent_runtime, columns);
                match agent_runtime.selection {
                    SelectionState::Selected => selected.push(entry),
                    SelectionState::Hovered => hovered = Some(entry),
                    SelectionState::None => {}
                }
            }
        }

        let mut focus_candidate = inspector.focused_agent.filter(|id| arena.contains(*id));

        if focus_candidate.is_none() {
            focus_candidate = selected.first().map(|entry| entry.agent_id);
        }
        if focus_candidate.is_none() {
            focus_candidate = hovered.as_ref().map(|entry| entry.agent_id);
        }

        let focus_id = focus_candidate;

        let focused =
            focus_id.and_then(|agent_id| AgentInspectorDetails::from_world(world, agent_id));

        for entry in &mut selected {
            entry.is_focused = Some(entry.agent_id) == focus_id;
        }
        if let Some(entry) = hovered.as_mut() {
            entry.is_focused = Some(entry.agent_id) == focus_id;
        }

        snapshot.focused = focused;
        snapshot.selected = selected;
        snapshot.hovered = hovered;
        snapshot.focus_id = focus_id;
        snapshot
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum ColorPaletteMode {
    #[default]
    Natural,
    Deuteranopia,
    Protanopia,
    Tritanopia,
    HighContrast,
}

impl ColorPaletteMode {
    const ALL: [ColorPaletteMode; 5] = [
        ColorPaletteMode::Natural,
        ColorPaletteMode::Deuteranopia,
        ColorPaletteMode::Protanopia,
        ColorPaletteMode::Tritanopia,
        ColorPaletteMode::HighContrast,
    ];

    fn next(self) -> Self {
        match self {
            ColorPaletteMode::Natural => ColorPaletteMode::Deuteranopia,
            ColorPaletteMode::Deuteranopia => ColorPaletteMode::Protanopia,
            ColorPaletteMode::Protanopia => ColorPaletteMode::Tritanopia,
            ColorPaletteMode::Tritanopia => ColorPaletteMode::HighContrast,
            ColorPaletteMode::HighContrast => ColorPaletteMode::Natural,
        }
    }

    fn label(self) -> &'static str {
        match self {
            ColorPaletteMode::Natural => "Natural",
            ColorPaletteMode::Deuteranopia => "Deuteranopia",
            ColorPaletteMode::Protanopia => "Protanopia",
            ColorPaletteMode::Tritanopia => "Tritanopia",
            ColorPaletteMode::HighContrast => "High Contrast",
        }
    }
}

#[derive(Default, Clone)]
struct AccessibilitySettings {
    palette: ColorPaletteMode,
    narration_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum CommandAction {
    TogglePlayback,
    GoLive,
    ToggleBrush,
    ToggleNarration,
    CyclePalette,
    ToggleSimulationPause,
    ToggleAgentDraw,
    ToggleFoodOverlay,
    ToggleAgentOutline,
    IncreaseSimulationSpeed,
    DecreaseSimulationSpeed,
    AddCrossoverAgents,
    SpawnCarnivore,
    SpawnHerbivore,
    ToggleClosedEnvironment,
    FollowSelected,
    FollowOldest,
    ToggleDebugOverlay,
    ClearSelection,
    SelectAll,
    FocusFirstSelected,
    ToggleSettings,
}

impl CommandAction {
    fn label(self) -> &'static str {
        match self {
            CommandAction::TogglePlayback => "Toggle playback",
            CommandAction::GoLive => "Jump to live",
            CommandAction::ToggleBrush => "Toggle brush",
            CommandAction::ToggleNarration => "Toggle narration",
            CommandAction::CyclePalette => "Cycle palette",
            CommandAction::ToggleSimulationPause => "Toggle simulation pause",
            CommandAction::ToggleAgentDraw => "Toggle agent drawing",
            CommandAction::ToggleFoodOverlay => "Toggle food overlay",
            CommandAction::ToggleAgentOutline => "Toggle agent outline",
            CommandAction::IncreaseSimulationSpeed => "Increase simulation speed",
            CommandAction::DecreaseSimulationSpeed => "Decrease simulation speed",
            CommandAction::AddCrossoverAgents => "Spawn crossover agent",
            CommandAction::SpawnCarnivore => "Spawn carnivore",
            CommandAction::SpawnHerbivore => "Spawn herbivore",
            CommandAction::ToggleClosedEnvironment => "Toggle closed environment",
            CommandAction::FollowSelected => "Follow selected agent",
            CommandAction::FollowOldest => "Follow oldest agent",
            CommandAction::ToggleDebugOverlay => "Toggle debug overlay",
            CommandAction::ClearSelection => "Clear selection",
            CommandAction::SelectAll => "Select all agents",
            CommandAction::FocusFirstSelected => "Focus first selected agent",
            CommandAction::ToggleSettings => "Toggle settings panel",
        }
    }
}

#[derive(Clone)]
struct InputBindings {
    map: BTreeMap<CommandAction, Keystroke>,
}

impl Default for InputBindings {
    fn default() -> Self {
        let mut map = BTreeMap::new();
        map.insert(
            CommandAction::TogglePlayback,
            Keystroke::parse("space").unwrap_or_default(),
        );
        map.insert(
            CommandAction::GoLive,
            Keystroke::parse("g").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleBrush,
            Keystroke::parse("b").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleNarration,
            Keystroke::parse("n").unwrap_or_default(),
        );
        map.insert(
            CommandAction::CyclePalette,
            Keystroke::parse("ctrl+p").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleSimulationPause,
            Keystroke::parse("p").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleAgentDraw,
            Keystroke::parse("d").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleFoodOverlay,
            Keystroke::parse("f").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleAgentOutline,
            Keystroke::parse("ctrl+shift+o").unwrap_or_default(),
        );
        map.insert(
            CommandAction::IncreaseSimulationSpeed,
            Keystroke::parse("shift+=").unwrap_or_default(),
        );
        map.insert(
            CommandAction::DecreaseSimulationSpeed,
            Keystroke::parse("-").unwrap_or_default(),
        );
        map.insert(
            CommandAction::AddCrossoverAgents,
            Keystroke::parse("a").unwrap_or_default(),
        );
        map.insert(
            CommandAction::SpawnCarnivore,
            Keystroke::parse("q").unwrap_or_default(),
        );
        map.insert(
            CommandAction::SpawnHerbivore,
            Keystroke::parse("h").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleClosedEnvironment,
            Keystroke::parse("c").unwrap_or_default(),
        );
        map.insert(
            CommandAction::FollowSelected,
            Keystroke::parse("s").unwrap_or_default(),
        );
        map.insert(
            CommandAction::FollowOldest,
            Keystroke::parse("o").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleDebugOverlay,
            Keystroke::parse("shift+f").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ClearSelection,
            Keystroke::parse("escape").unwrap_or_default(),
        );
        map.insert(
            CommandAction::SelectAll,
            Keystroke::parse("ctrl+a").unwrap_or_default(),
        );
        map.insert(
            CommandAction::FocusFirstSelected,
            Keystroke::parse("ctrl+f").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleSettings,
            Keystroke::parse(",").unwrap_or_default(),
        );
        Self { map }
    }
}

impl InputBindings {
    fn iter(&self) -> Vec<(CommandAction, Keystroke)> {
        self.map
            .iter()
            .map(|(action, ks)| (*action, ks.clone()))
            .collect()
    }

    fn assign(&mut self, action: CommandAction, stroke: Keystroke) {
        if stroke.key.is_empty() {
            self.map.insert(action, Keystroke::default());
            return;
        }

        let conflict = self.map.iter().find_map(|(other, ks)| {
            if *other != action && keystrokes_equal(ks, &stroke) {
                Some(*other)
            } else {
                None
            }
        });

        if let Some(conflict_action) = conflict {
            self.map.insert(conflict_action, Keystroke::default());
        }

        self.map.insert(action, stroke);
    }

    fn action_for(&self, stroke: &Keystroke) -> Option<CommandAction> {
        self.map
            .iter()
            .find(|(_, binding)| keystrokes_equal(binding, stroke))
            .map(|(action, _)| *action)
    }
}

#[cfg(feature = "audio")]
struct AudioState {
    manager: AudioManager<DefaultBackend>,
    birth_sound: StaticSoundData,
    death_sound: StaticSoundData,
    spike_sound: StaticSoundData,
    toggle_sound: StaticSoundData,
    last_births: usize,
    last_deaths: usize,
    last_spike_count: usize,
    last_tick: u64,
}

#[cfg(feature = "audio")]
impl AudioState {
    fn new() -> Result<Self, String> {
        let manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())
            .map_err(|err| format!("{err:?}"))?;
        let birth_sound = generate_tone(523.25, 0.18, 0.4);
        let death_sound = generate_tone(196.0, 0.22, 0.45);
        let spike_sound = generate_tone(880.0, 0.12, 0.35);
        let toggle_sound = generate_tone(660.0, 0.10, 0.3);
        Ok(Self {
            manager,
            birth_sound,
            death_sound,
            spike_sound,
            toggle_sound,
            last_births: 0,
            last_deaths: 0,
            last_spike_count: 0,
            last_tick: u64::MAX,
        })
    }

    fn play(&mut self, sound: &StaticSoundData) {
        if let Err(err) = self.manager.play(sound.clone()) {
            error!(?err, "failed to play audio cue");
        }
    }
}

fn keystrokes_equal(a: &Keystroke, b: &Keystroke) -> bool {
    if a.key.is_empty() || b.key.is_empty() {
        return false;
    }

    if a.modifiers != b.modifiers {
        return false;
    }

    let key_matches = a.key.eq_ignore_ascii_case(&b.key)
        || (a.key.eq_ignore_ascii_case("space") && b.key.trim().is_empty())
        || (b.key.eq_ignore_ascii_case("space") && a.key.trim().is_empty());

    if key_matches {
        return true;
    }

    if let Some(ref key_char) = a.key_char {
        if key_char.eq_ignore_ascii_case(&b.key) {
            return true;
        }
        if a.key.eq_ignore_ascii_case("space") && key_char.trim().is_empty() {
            return true;
        }
    }

    if let Some(ref key_char) = b.key_char {
        if key_char.eq_ignore_ascii_case(&a.key) {
            return true;
        }
        if b.key.eq_ignore_ascii_case("space") && key_char.trim().is_empty() {
            return true;
        }
    }

    false
}

fn format_keystroke(keystroke: &Keystroke) -> String {
    if keystroke.key.is_empty() {
        return "Unbound".to_string();
    }

    let mut parts = Vec::new();
    if keystroke.modifiers.control {
        parts.push("Ctrl".to_string());
    }
    if keystroke.modifiers.alt {
        parts.push("Alt".to_string());
    }
    if keystroke.modifiers.shift {
        parts.push("Shift".to_string());
    }
    if keystroke.modifiers.platform {
        parts.push(if cfg!(target_os = "macos") {
            "Cmd".to_string()
        } else {
            "Super".to_string()
        });
    }
    if keystroke.modifiers.function {
        parts.push("Fn".to_string());
    }

    let key = if keystroke.key.len() == 1 {
        keystroke.key.to_uppercase()
    } else {
        keystroke.key.clone()
    };
    parts.push(key);
    parts.join(" + ")
}

#[derive(Clone)]
struct AgentListEntry {
    agent_id: AgentId,
    label: String,
    color: [f32; 3],
    energy: f32,
    health: f32,
    generation: Generation,
    age: u32,
    is_focused: bool,
}

impl AgentListEntry {
    fn from_world(
        row: usize,
        agent_id: AgentId,
        runtime: &AgentRuntime,
        columns: &AgentColumns,
    ) -> Self {
        let generation = columns.generations()[row];
        let age = columns.ages()[row];
        let health = columns.health()[row];
        let color = columns.colors()[row];

        let label = format!("#{row} · {:?} · Gen {}", agent_id, generation.0);

        Self {
            agent_id,
            label,
            color,
            energy: runtime.energy,
            health,
            generation,
            age,
            is_focused: false,
        }
    }
}

#[derive(Clone)]
struct AgentInspectorDetails {
    agent_id: AgentId,
    label: String,
    color: [f32; 3],
    position: Position,
    energy: f32,
    health: f32,
    age: u32,
    generation: Generation,
    brain_descriptor: String,
    mutation_rates: MutationRates,
    trait_modifiers: TraitModifiers,
    spike_length: f32,
    sensors: Vec<f32>,
    outputs: Vec<f32>,
}

impl AgentInspectorDetails {
    fn from_world(world: &WorldState, agent_id: AgentId) -> Option<Self> {
        let arena = world.agents();
        let columns = arena.columns();
        let runtime = world.runtime();

        let index = arena.index_of(agent_id)?;
        let agent_runtime = runtime.get(agent_id)?;

        let position = columns.positions()[index];
        let color = columns.colors()[index];
        let health = columns.health()[index];
        let age = columns.ages()[index];
        let generation = columns.generations()[index];
        let spike_length = columns.spike_lengths()[index];

        let sensors = agent_runtime.sensors.to_vec();
        let outputs = agent_runtime.outputs.to_vec();

        let label = format!("Agent {:?} · Gen {} · Age {}", agent_id, generation.0, age);

        let brain_descriptor = agent_runtime.brain.describe().to_string();

        Some(Self {
            agent_id,
            label,
            color,
            position,
            energy: agent_runtime.energy,
            health,
            age,
            generation,
            brain_descriptor,
            mutation_rates: agent_runtime.mutation_rates,
            trait_modifiers: agent_runtime.trait_modifiers,
            spike_length,
            sensors,
            outputs,
        })
    }
}

fn rgb_from_triplet(color: [f32; 3]) -> Rgba {
    Rgba {
        r: color[0].clamp(0.0, 1.0),
        g: color[1].clamp(0.0, 1.0),
        b: color[2].clamp(0.0, 1.0),
        a: 1.0,
    }
}

fn rgba_from_triplet_with_alpha(color: [f32; 3], alpha: f32) -> Rgba {
    Rgba {
        r: color[0].clamp(0.0, 1.0),
        g: color[1].clamp(0.0, 1.0),
        b: color[2].clamp(0.0, 1.0),
        a: alpha.clamp(0.0, 1.0),
    }
}

fn rgba_from_hex(hex: u32, alpha: f32) -> Rgba {
    let r = ((hex >> 16) & 0xff) as f32 / 255.0;
    let g = ((hex >> 8) & 0xff) as f32 / 255.0;
    let b = (hex & 0xff) as f32 / 255.0;
    Rgba {
        r,
        g,
        b,
        a: alpha.clamp(0.0, 1.0),
    }
}

fn lerp_rgba(a: Rgba, b: Rgba, t: f32) -> Rgba {
    let clamped = t.clamp(0.0, 1.0);
    Rgba {
        r: a.r + (b.r - a.r) * clamped,
        g: a.g + (b.g - a.g) * clamped,
        b: a.b + (b.b - a.b) * clamped,
        a: a.a + (b.a - a.a) * clamped,
    }
}

fn scale_rgb(color: Rgba, factor: f32) -> Rgba {
    let clamp = factor.clamp(0.0, 2.0);
    Rgba {
        r: (color.r * clamp).clamp(0.0, 1.0),
        g: (color.g * clamp).clamp(0.0, 1.0),
        b: (color.b * clamp).clamp(0.0, 1.0),
        a: color.a,
    }
}

#[cfg(feature = "audio")]
const AUDIO_SAMPLE_RATE: u32 = 44_100;

#[cfg(feature = "audio")]
fn generate_tone(frequency: f32, duration: f32, amplitude: f32) -> StaticSoundData {
    let total_samples = (duration * AUDIO_SAMPLE_RATE as f32).max(1.0) as usize;
    let mut frames = Vec::with_capacity(total_samples);
    for i in 0..total_samples {
        let t = i as f32 / AUDIO_SAMPLE_RATE as f32;
        let envelope = (1.0 - t / duration).clamp(0.0, 1.0);
        let sample = (2.0 * PI * frequency * t).sin() * amplitude * envelope;
        frames.push(Frame::from_mono(sample));
    }
    StaticSoundData {
        sample_rate: AUDIO_SAMPLE_RATE,
        frames: frames.into(),
        settings: StaticSoundSettings::default(),
        slice: None,
    }
}

#[derive(Clone, Copy)]
struct PerfSnapshot {
    latest_ms: f32,
    average_ms: f32,
    min_ms: f32,
    max_ms: f32,
    sample_count: usize,
    fps: f32,
}

impl Default for PerfSnapshot {
    fn default() -> Self {
        Self {
            latest_ms: 0.0,
            average_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            sample_count: 0,
            fps: 0.0,
        }
    }
}

struct PerfStats {
    start: Option<Instant>,
    samples: VecDeque<f32>,
    capacity: usize,
}

impl PerfStats {
    fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            start: None,
            samples: VecDeque::with_capacity(cap),
            capacity: cap,
        }
    }

    fn begin_frame(&mut self) {
        self.start = Some(Instant::now());
    }

    fn end_frame(&mut self) -> PerfSnapshot {
        let elapsed_ms = self
            .start
            .take()
            .map(|start| start.elapsed().as_secs_f32() * 1000.0)
            .unwrap_or(0.0);
        self.samples.push_back(elapsed_ms);
        if self.samples.len() > self.capacity {
            self.samples.pop_front();
        }
        self.snapshot(elapsed_ms)
    }

    fn snapshot(&self, latest: f32) -> PerfSnapshot {
        if self.samples.is_empty() {
            return PerfSnapshot {
                latest_ms: latest,
                ..PerfSnapshot::default()
            };
        }

        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum = 0.0;
        for &sample in &self.samples {
            min = min.min(sample);
            max = max.max(sample);
            sum += sample;
        }
        let count = self.samples.len();
        let avg = if count > 0 { sum / count as f32 } else { 0.0 };
        let fps = if latest > f32::EPSILON {
            1000.0 / latest
        } else {
            0.0
        };

        PerfSnapshot {
            latest_ms: latest,
            average_ms: avg,
            min_ms: if min.is_finite() { min } else { 0.0 },
            max_ms: if max.is_finite() { max } else { 0.0 },
            sample_count: count,
            fps,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlaybackMode {
    Live,
    Paused,
    Playing,
}

#[derive(Clone, Copy)]
struct PlaybackStatus {
    mode: PlaybackMode,
    index: usize,
    total: usize,
    current_tick: Option<u64>,
}

struct PlaybackState {
    timeline: VecDeque<HudSnapshot>,
    max_frames: usize,
    mode: PlaybackMode,
    pointer: usize,
}

impl PlaybackState {
    fn new(max_frames: usize) -> Self {
        Self {
            timeline: VecDeque::with_capacity(max_frames),
            max_frames: max_frames.max(1),
            mode: PlaybackMode::Live,
            pointer: 0,
        }
    }

    #[allow(dead_code)]
    fn mode(&self) -> PlaybackMode {
        self.mode
    }

    fn record(&mut self, snapshot: HudSnapshot) {
        if self.timeline.len() == self.max_frames {
            self.timeline.pop_front();
            if self.pointer > 0 {
                self.pointer -= 1;
            }
        }
        self.timeline.push_back(snapshot);
        let last_index = self.timeline.len().saturating_sub(1);
        if matches!(self.mode, PlaybackMode::Live) {
            self.pointer = last_index;
        } else {
            self.pointer = self.pointer.min(last_index);
        }
    }

    fn snapshot_for_render(&mut self, live: HudSnapshot) -> HudSnapshot {
        match self.mode {
            PlaybackMode::Live => live,
            PlaybackMode::Paused => self.timeline.get(self.pointer).cloned().unwrap_or(live),
            PlaybackMode::Playing => {
                if self.timeline.is_empty() {
                    self.mode = PlaybackMode::Live;
                    return live;
                }
                let snapshot = self
                    .timeline
                    .get(self.pointer)
                    .cloned()
                    .unwrap_or_else(|| live.clone());
                if self.pointer + 1 < self.timeline.len() {
                    self.pointer += 1;
                } else {
                    self.mode = PlaybackMode::Live;
                    self.pointer = self.timeline.len().saturating_sub(1);
                }
                snapshot
            }
        }
    }

    fn restart(&mut self) {
        if self.timeline.is_empty() {
            return;
        }
        self.mode = PlaybackMode::Paused;
        self.pointer = 0;
    }

    fn step_back(&mut self) {
        if self.timeline.is_empty() {
            return;
        }
        self.mode = PlaybackMode::Paused;
        if self.pointer > 0 {
            self.pointer -= 1;
        }
    }

    fn step_forward(&mut self) {
        if self.timeline.is_empty() {
            return;
        }
        if self.pointer + 1 < self.timeline.len() {
            self.pointer += 1;
            self.mode = PlaybackMode::Paused;
        } else {
            self.go_live();
        }
    }

    fn toggle_play(&mut self) {
        match self.mode {
            PlaybackMode::Live => {
                if !self.timeline.is_empty() {
                    self.mode = PlaybackMode::Playing;
                    self.pointer = 0;
                }
            }
            PlaybackMode::Paused => {
                if !self.timeline.is_empty() {
                    self.mode = PlaybackMode::Playing;
                }
            }
            PlaybackMode::Playing => {
                self.mode = PlaybackMode::Paused;
            }
        }
    }

    fn go_live(&mut self) {
        self.mode = PlaybackMode::Live;
        if !self.timeline.is_empty() {
            self.pointer = self.timeline.len().saturating_sub(1);
        }
    }

    fn status(&self) -> PlaybackStatus {
        let current_tick = self.timeline.get(self.pointer).map(|snap| snap.tick);
        PlaybackStatus {
            mode: self.mode,
            index: if self.timeline.is_empty() {
                0
            } else {
                self.pointer.min(self.timeline.len() - 1)
            },
            total: self.timeline.len(),
            current_tick,
        }
    }
}

fn color_swatch(color: [f32; 3]) -> Div {
    div()
        .w(px(10.0))
        .h(px(10.0))
        .rounded_full()
        .border_1()
        .border_color(rgb(0x1e293b))
        .bg(rgb_from_triplet(color))
}

#[derive(Clone)]
struct RenderFrame {
    tick: u64,
    world_size: (f32, f32),
    terrain: TerrainFrame,
    food_dimensions: (u32, u32),
    food_cell_size: u32,
    food_cells: Vec<f32>,
    food_max: f32,
    agents: Vec<AgentRenderData>,
    agent_base_radius: f32,
    sense_radius: f32,
    post_stack: PostProcessStack,
    palette: ColorPaletteMode,
}

#[derive(Clone)]
struct TerrainFrame {
    dimensions: (u32, u32),
    cell_size: u32,
    tiles: Vec<TerrainTileVisual>,
}

#[derive(Clone, Copy)]
struct TerrainTileVisual {
    kind: TerrainKind,
    elevation: f32,
    moisture: f32,
    accent: f32,
    slope: f32,
}

#[derive(Clone)]
struct PostProcessStack {
    passes: Vec<PostProcessPass>,
}

#[derive(Clone, Copy)]
enum PostProcessPass {
    Vignette {
        strength: f32,
    },
    Bloom {
        strength: f32,
    },
    Scanlines {
        intensity: f32,
        spacing: f32,
    },
    FilmGrain {
        strength: f32,
        seed: u64,
    },
    ColorGrade {
        lift: f32,
        gain: f32,
        temperature: f32,
    },
}

fn build_post_process_stack(world: &WorldState, palette: ColorPaletteMode) -> PostProcessStack {
    let tick = world.tick().0;
    let day_phase = (tick as f32 * 0.00025).sin() * 0.5 + 0.5;
    let closed_bonus = if world.is_closed() { 0.18 } else { 0.0 };
    let agent_count = world.agent_count().max(1) as f32;
    let latest = world.history().last();
    let (births_ratio, deaths_ratio) = latest
        .map(|summary| {
            (
                summary.births as f32 / agent_count,
                summary.deaths as f32 / agent_count,
            )
        })
        .unwrap_or((0.0, 0.0));
    let life_delta = (births_ratio - deaths_ratio).clamp(-1.0, 1.0);
    let tension = life_delta.abs();

    let vignette_strength =
        (0.36 + day_phase * 0.24 + closed_bonus + (-life_delta).max(0.0) * 0.28).clamp(0.2, 0.82);
    let bloom_strength = (0.24 + day_phase * 0.32 + life_delta.max(0.0) * 0.28).clamp(0.12, 0.78);
    let scanline_intensity =
        (0.18 + (1.0 - day_phase) * 0.22 + closed_bonus * 0.35 + (-life_delta).max(0.0) * 0.18)
            .clamp(0.08, 0.65);
    let grain_strength =
        (0.11 + (tick % 4096) as f32 / 4096.0 * 0.08 + tension * 0.06).clamp(0.08, 0.26);

    let temperature = match palette {
        ColorPaletteMode::Natural => 0.08 - life_delta * 0.05,
        ColorPaletteMode::Deuteranopia => 0.05,
        ColorPaletteMode::Protanopia => -0.04,
        ColorPaletteMode::Tritanopia => 0.12,
        ColorPaletteMode::HighContrast => 0.20,
    };

    let color_grade = PostProcessPass::ColorGrade {
        lift: (0.05 + closed_bonus * 0.12 - life_delta * 0.06).clamp(0.0, 0.12),
        gain: (1.02 + day_phase * 0.08 + life_delta.max(0.0) * 0.12).clamp(1.0, 1.25),
        temperature,
    };

    let passes = vec![
        color_grade,
        PostProcessPass::Vignette {
            strength: vignette_strength,
        },
        PostProcessPass::Bloom {
            strength: bloom_strength,
        },
        PostProcessPass::Scanlines {
            intensity: scanline_intensity,
            spacing: 4.5,
        },
        PostProcessPass::FilmGrain {
            strength: grain_strength,
            seed: tick,
        },
    ];

    PostProcessStack { passes }
}

#[derive(Clone)]
struct AgentRenderData {
    agent_id: AgentId,
    position: Position,
    color: [f32; 3],
    spike_length: f32,
    velocity: Velocity,
    health: f32,
    age: u32,
    selection: SelectionState,
    indicator: IndicatorState,
    spiked: bool,
    reproduction_intent: f32,
}

#[derive(Clone)]
struct CanvasState {
    frame: RenderFrame,
    camera: Arc<Mutex<CameraState>>,
    focus_agent: Option<AgentId>,
    controls: ControlsSnapshot,
    debug: DebugOverlayState,
    follow_target: Option<Position>,
}

impl RenderFrame {
    fn from_world(world: &WorldState, palette: ColorPaletteMode) -> Option<Self> {
        let food = world.food();
        let width = food.width();
        let height = food.height();
        if width == 0 || height == 0 {
            return None;
        }

        let config = world.config();
        let arena = world.agents();
        let columns = arena.columns();
        let runtime = world.runtime();

        let positions = columns.positions();
        let colors = columns.colors();
        let spikes = columns.spike_lengths();
        let healths = columns.health();
        let velocities = columns.velocities();
        let ages = columns.ages();

        let agents = arena
            .iter_handles()
            .enumerate()
            .map(|(idx, agent_id)| {
                let runtime_entry = runtime.get(agent_id);
                let selection = runtime_entry.map(|rt| rt.selection).unwrap_or_default();
                let indicator = runtime_entry.map(|rt| rt.indicator).unwrap_or_default();
                let spiked = runtime_entry.map(|rt| rt.spiked).unwrap_or(false);
                let reproduction_intent = runtime_entry.map(|rt| rt.give_intent).unwrap_or(0.0);

                AgentRenderData {
                    agent_id,
                    position: positions[idx],
                    color: colors[idx],
                    spike_length: spikes[idx],
                    velocity: velocities[idx],
                    health: healths[idx],
                    age: ages[idx],
                    selection,
                    indicator,
                    spiked,
                    reproduction_intent,
                }
            })
            .collect();

        let food_cells = food.cells().to_vec();
        let terrain = build_terrain_frame(world.terrain());

        Some(Self {
            tick: world.tick().0,
            world_size: (config.world_width as f32, config.world_height as f32),
            terrain,
            food_dimensions: (width, height),
            food_cell_size: config.food_cell_size,
            food_cells,
            food_max: config.food_max,
            agents,
            agent_base_radius: (config.spike_radius * 0.5).max(12.0),
            sense_radius: config.sense_radius,
            post_stack: build_post_process_stack(world, palette),
            palette,
        })
    }
}

fn build_terrain_frame(layer: &TerrainLayer) -> TerrainFrame {
    let width = layer.width();
    let height = layer.height();
    let mut tiles = Vec::with_capacity((width as usize) * (height as usize));
    let source = layer.tiles();
    if width == 0 || height == 0 {
        return TerrainFrame {
            dimensions: (width, height),
            cell_size: layer.cell_size(),
            tiles,
        };
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    for y in 0..height_usize {
        for x in 0..width_usize {
            let idx = y * width_usize + x;
            let tile = source[idx];

            let left = if x > 0 { source[idx - 1] } else { tile };
            let right = if x + 1 < width_usize {
                source[idx + 1]
            } else {
                tile
            };
            let up = if y > 0 {
                source[idx - width_usize]
            } else {
                tile
            };
            let down = if y + 1 < height_usize {
                source[idx + width_usize]
            } else {
                tile
            };

            let slope = ((tile.elevation - left.elevation).abs()
                + (tile.elevation - right.elevation).abs()
                + (tile.elevation - up.elevation).abs()
                + (tile.elevation - down.elevation).abs())
                * 0.25;
            let slope = slope.min(1.0);

            tiles.push(TerrainTileVisual {
                kind: tile.kind,
                elevation: tile.elevation,
                moisture: tile.moisture,
                accent: tile.accent,
                slope,
            });
        }
    }

    TerrainFrame {
        dimensions: (width, height),
        cell_size: layer.cell_size(),
        tiles,
    }
}

#[derive(Clone)]
struct HistoryChartData {
    width: f32,
    height: f32,
    agents: Vec<(f32, f32)>,
    births: Vec<(f32, f32)>,
    deaths: Vec<(f32, f32)>,
}

impl HistoryChartData {
    fn from_entries(entries: &[HudHistoryEntry], width: f32, height: f32) -> Option<Self> {
        if entries.len() < 2 {
            return None;
        }

        let max_agents = entries.iter().map(|e| e.agent_count).max().unwrap_or(1);
        let max_births = entries.iter().map(|e| e.births).max().unwrap_or(0);
        let max_deaths = entries.iter().map(|e| e.deaths).max().unwrap_or(0);
        let scale_agents = max_agents.max(1) as f32;
        let scale_births = max_births.max(1) as f32;
        let scale_deaths = max_deaths.max(1) as f32;

        let samples = entries.len();
        let step = if samples > 1 {
            width / (samples as f32 - 1.0)
        } else {
            width
        };

        let y_clamp = height - 1.0;

        let to_points = |values: Vec<f32>| -> Vec<(f32, f32)> {
            values
                .into_iter()
                .enumerate()
                .map(|(idx, v)| {
                    let x = step * idx as f32;
                    let y = height - (v * y_clamp).min(y_clamp);
                    (x, y)
                })
                .collect()
        };

        let agents = to_points(
            entries
                .iter()
                .map(|e| e.agent_count as f32 / scale_agents)
                .collect(),
        );
        let births = to_points(
            entries
                .iter()
                .map(|e| e.births as f32 / scale_births)
                .collect(),
        );
        let deaths = to_points(
            entries
                .iter()
                .map(|e| e.deaths as f32 / scale_deaths)
                .collect(),
        );

        Some(Self {
            width,
            height,
            agents,
            births,
            deaths,
        })
    }
}

fn legend_item(color: Rgba, label: &str) -> Div {
    div()
        .flex()
        .items_center()
        .gap_1()
        .child(div().w(px(8.0)).h(px(8.0)).rounded_full().bg(color))
        .child(label.to_string())
}

fn paint_history_chart(bounds: Bounds<Pixels>, data: &HistoryChartData, window: &mut Window) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let chart_width = f32::from(bounds_size.width).max(1.0);
    let chart_height = f32::from(bounds_size.height).max(1.0);
    let scale_x = chart_width / data.width.max(1.0);
    let scale_y = chart_height / data.height.max(1.0);

    let to_point = |(x, y): (f32, f32)| {
        point(
            px(f32::from(origin.x) + x * scale_x),
            px(f32::from(origin.y) + y * scale_y),
        )
    };

    let mut draw_polyline = |points: &[(f32, f32)], color: Rgba| {
        if points.len() < 2 {
            return;
        }
        let mut builder = PathBuilder::stroke(px(1.6));
        builder.move_to(to_point(points[0]));
        for &pt in &points[1..] {
            builder.line_to(to_point(pt));
        }
        if let Ok(path) = builder.build() {
            window.paint_path(path, color);
        }
    };

    draw_polyline(&data.agents, rgb(0x38bdf8));
    draw_polyline(&data.births, rgb(0x22c55e));
    draw_polyline(&data.deaths, rgb(0xef4444));
}

fn append_arc_polyline(
    builder: &mut PathBuilder,
    cx: f32,
    cy: f32,
    radius: f32,
    start_deg: f32,
    sweep_deg: f32,
) {
    let segments = (sweep_deg.abs() / 6.0).ceil().max(1.0) as usize;
    let start = start_deg.to_radians();
    let sweep = sweep_deg.to_radians();
    let step = sweep / segments as f32;
    for i in 0..=segments {
        let angle = start + step * i as f32;
        let x = cx + radius * angle.cos();
        let y = cy + radius * angle.sin();
        if i == 0 {
            builder.move_to(point(px(x), px(y)));
        } else {
            builder.line_to(point(px(x), px(y)));
        }
    }
}

fn append_circle_polygon(builder: &mut PathBuilder, cx: f32, cy: f32, radius: f32) {
    append_arc_polyline(builder, cx, cy, radius, 0.0, 360.0);
}

fn paint_vector_hud(bounds: Bounds<Pixels>, state: &VectorHudState, window: &mut Window) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let width = f32::from(bounds_size.width).max(1.0);
    let height = f32::from(bounds_size.height).max(1.0);

    let backdrop = rgba_from_hex(0x091220, 0.88);
    window.paint_quad(fill(bounds, Background::from(backdrop)));

    let cx = f32::from(origin.x) + width * 0.5;
    let cy = f32::from(origin.y) + height * 0.52;
    let center = point(px(cx), px(cy));
    let radius = (width.min(height) * 0.36).max(18.0);

    let mut base_arc = PathBuilder::stroke(px(3.2));
    append_arc_polyline(&mut base_arc, cx, cy, radius, -140.0, 280.0);
    if let Ok(path) = base_arc.build() {
        window.paint_path(path, rgba_from_hex(0x142033, 0.95));
    }

    let progress_deg = 280.0 * state.population_ratio.clamp(0.0, 1.0);
    if progress_deg > 0.5 {
        let mut progress_arc = PathBuilder::stroke(px(4.2));
        append_arc_polyline(&mut progress_arc, cx, cy, radius, -140.0, progress_deg);
        if let Ok(path) = progress_arc.build() {
            let progress_color = lerp_rgba(
                rgba_from_hex(0x38bdf8, 0.95),
                rgba_from_hex(0x22c55e, 0.95),
                state.energy_ratio.clamp(0.0, 1.0),
            );
            window.paint_path(path, progress_color);
        }
    }

    let mut halo_arc = PathBuilder::stroke(px(1.6));
    append_arc_polyline(&mut halo_arc, cx, cy, radius * 1.08, -140.0, 280.0);
    if let Ok(path) = halo_arc.build() {
        window.paint_path(path, rgba_from_hex(0x1d3559, 0.25));
    }

    let energy_scale = state.energy_ratio.clamp(0.0, 1.0);
    let inner_radius = radius * (0.46 + energy_scale * 0.18);
    let mut inner_fill = PathBuilder::fill();
    append_circle_polygon(&mut inner_fill, cx, cy, inner_radius);
    if let Ok(path) = inner_fill.build() {
        let inner_color = lerp_rgba(
            rgba_from_hex(0x122033, 0.92),
            rgba_from_hex(0x3b82f6, 0.88),
            energy_scale,
        );
        window.paint_path(path, inner_color);
    }

    let pointer_deg = -140.0 + 280.0 * state.tick_phase;
    let pointer_rad = pointer_deg.to_radians();
    let pointer_radius = radius * 1.02;
    let px_pointer = cx + pointer_radius * pointer_rad.cos();
    let py_pointer = cy + pointer_radius * pointer_rad.sin();
    let mut pointer = PathBuilder::stroke(px(1.8));
    pointer.move_to(center);
    pointer.line_to(point(px(px_pointer), px(py_pointer)));
    if let Ok(path) = pointer.build() {
        window.paint_path(path, rgba_from_hex(0xfacc15, 0.75));
    }

    let velocity_ratio = if state.max_speed > f32::EPSILON {
        (state.vector_magnitude / state.max_speed).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if velocity_ratio > 0.01 {
        let heading = state.heading_rad;
        let arrow_length = radius * 0.85 * velocity_ratio;
        let tip_x = cx + arrow_length * heading.cos();
        let tip_y = cy + arrow_length * heading.sin();

        let mut arrow = PathBuilder::stroke(px(2.1));
        arrow.move_to(center);
        arrow.line_to(point(px(tip_x), px(tip_y)));
        if let Ok(path) = arrow.build() {
            window.paint_path(path, rgba_from_hex(0x38bdf8, 0.88));
        }

        let head_size = (8.0 + velocity_ratio * 18.0).clamp(6.0, 18.0);
        let left_angle = heading + PI - 0.4;
        let right_angle = heading + PI + 0.4;

        let left_point = point(
            px(tip_x + head_size * left_angle.cos()),
            px(tip_y + head_size * left_angle.sin()),
        );
        let right_point = point(
            px(tip_x + head_size * right_angle.cos()),
            px(tip_y + head_size * right_angle.sin()),
        );

        let mut left_head = PathBuilder::stroke(px(1.4));
        left_head.move_to(point(px(tip_x), px(tip_y)));
        left_head.line_to(left_point);
        if let Ok(path) = left_head.build() {
            window.paint_path(path, rgba_from_hex(0xe0f2fe, 0.82));
        }

        let mut right_head = PathBuilder::stroke(px(1.4));
        right_head.move_to(point(px(tip_x), px(tip_y)));
        right_head.line_to(right_point);
        if let Ok(path) = right_head.build() {
            window.paint_path(path, rgba_from_hex(0xe0f2fe, 0.82));
        }

        let coherence = (velocity_ratio * 100.0).clamp(0.0, 100.0);
        if coherence > 1.0 {
            let ring_inner = radius * 0.55;
            let mut ring = PathBuilder::stroke(px(1.2 + coherence * 0.02));
            append_arc_polyline(
                &mut ring,
                cx,
                cy,
                ring_inner,
                -140.0,
                280.0 * velocity_ratio,
            );
            if let Ok(path) = ring.build() {
                window.paint_path(path, rgba_from_hex(0x60a5fa, 0.65));
            }
        }
    }

    let bar_origin_x = f32::from(origin.x) + 12.0;
    let bar_origin_y = f32::from(origin.y) + height - 18.0;
    let bar_width = (width - 24.0).max(12.0);
    let bar_height = 6.0;
    let bar_bounds = Bounds::new(
        point(px(bar_origin_x), px(bar_origin_y)),
        size(px(bar_width), px(bar_height)),
    );
    window.paint_quad(fill(
        bar_bounds,
        Background::from(rgba_from_hex(0x132036, 0.95)),
    ));

    let births_width = bar_width * state.births_ratio.clamp(0.0, 1.0);
    if births_width > 0.5 {
        let births_bounds = Bounds::new(
            point(px(bar_origin_x), px(bar_origin_y)),
            size(px(births_width), px(bar_height * 0.55)),
        );
        window.paint_quad(fill(
            births_bounds,
            Background::from(rgba_from_hex(0x22c55e, 0.92)),
        ));
    }

    let deaths_width = bar_width * state.deaths_ratio.clamp(0.0, 1.0);
    if deaths_width > 0.5 {
        let deaths_bounds = Bounds::new(
            point(px(bar_origin_x), px(bar_origin_y + bar_height * 0.45)),
            size(px(deaths_width), px(bar_height * 0.55)),
        );
        window.paint_quad(fill(
            deaths_bounds,
            Background::from(rgba_from_hex(0xef4444, 0.92)),
        ));
    }

    let marker_x = bar_origin_x + bar_width * state.population_ratio.clamp(0.0, 1.0);
    let marker_bounds = Bounds::new(
        point(px(marker_x - 1.0), px(bar_origin_y - 4.0)),
        size(px(2.0), px(bar_height + 8.0)),
    );
    window.paint_quad(fill(
        marker_bounds,
        Background::from(rgba_from_hex(0x93c5fd, 0.65)),
    ));
}

#[derive(Clone)]
struct CameraState {
    offset_px: (f32, f32),
    zoom: f32,
    panning: bool,
    pan_anchor: Option<Point<Pixels>>,
    last_canvas_origin: (f32, f32),
    last_canvas_size: (f32, f32),
    last_world_size: (f32, f32),
    last_scale: f32,
    last_base_scale: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            offset_px: (0.0, 0.0),
            zoom: Self::default_zoom(),
            panning: false,
            pan_anchor: None,
            last_canvas_origin: (0.0, 0.0),
            last_canvas_size: (1.0, 1.0),
            last_world_size: (1.0, 1.0),
            last_scale: 1.0,
            last_base_scale: 1.0,
        }
    }
}

impl CameraState {
    const MIN_ZOOM: f32 = 0.4;
    const MAX_ZOOM: f32 = 2.5;
    fn default_zoom() -> f32 {
        1.0
    }

    fn start_pan(&mut self, cursor: Point<Pixels>) {
        self.panning = true;
        self.pan_anchor = Some(cursor);
    }

    fn update_pan(&mut self, cursor: Point<Pixels>) -> bool {
        if !self.panning {
            return false;
        }
        if let Some(anchor) = self.pan_anchor {
            let dx = f32::from(cursor.x) - f32::from(anchor.x);
            let dy = f32::from(cursor.y) - f32::from(anchor.y);
            if dx.abs() > f32::EPSILON || dy.abs() > f32::EPSILON {
                self.offset_px.0 += dx;
                self.offset_px.1 += dy;
                self.pan_anchor = Some(cursor);
                return true;
            }
        }
        false
    }

    fn end_pan(&mut self) {
        self.panning = false;
        self.pan_anchor = None;
    }

    fn apply_scroll(&mut self, event: &ScrollWheelEvent) -> bool {
        let scroll_y = match event.delta {
            ScrollDelta::Pixels(delta) => -f32::from(delta.y) / 120.0,
            ScrollDelta::Lines(lines) => -lines.y,
        };
        if scroll_y.abs() < 0.01 {
            return false;
        }

        let old_zoom = self.zoom;
        let new_zoom = (self.zoom * (1.0 + scroll_y * 0.1)).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        if (new_zoom - old_zoom).abs() < f32::EPSILON {
            return false;
        }

        let cursor_x = f32::from(event.position.x);
        let cursor_y = f32::from(event.position.y);

        let old_scale = self.last_scale.max(1e-6);
        let base_scale = self.last_base_scale.max(1e-6);
        let new_scale = base_scale * new_zoom;

        let pad_x_old = (self.last_canvas_size.0 - self.last_world_size.0 * old_scale) * 0.5;
        let pad_y_old = (self.last_canvas_size.1 - self.last_world_size.1 * old_scale) * 0.5;
        let pad_x_new = (self.last_canvas_size.0 - self.last_world_size.0 * new_scale) * 0.5;
        let pad_y_new = (self.last_canvas_size.1 - self.last_world_size.1 * new_scale) * 0.5;

        let origin_x = self.last_canvas_origin.0;
        let origin_y = self.last_canvas_origin.1;

        let world_x = (cursor_x - origin_x - pad_x_old - self.offset_px.0) / old_scale;
        let world_y = (cursor_y - origin_y - pad_y_old - self.offset_px.1) / old_scale;

        self.zoom = new_zoom;
        self.offset_px.0 = cursor_x - origin_x - pad_x_new - world_x * new_scale;
        self.offset_px.1 = cursor_y - origin_y - pad_y_new - world_y * new_scale;
        true
    }

    fn record_render_metrics(
        &mut self,
        canvas_origin: (f32, f32),
        canvas_size: (f32, f32),
        world_size: (f32, f32),
        base_scale: f32,
    ) {
        self.last_canvas_origin = canvas_origin;
        self.last_canvas_size = canvas_size;
        self.last_world_size = world_size;
        self.last_base_scale = base_scale;
        self.last_scale = base_scale * self.zoom;
    }

    fn center_on(&mut self, position: Position) {
        let scale = self.last_base_scale * self.zoom;
        if !scale.is_finite() || scale <= f32::EPSILON {
            return;
        }

        let center_x = self.last_canvas_origin.0 + self.last_canvas_size.0 * 0.5;
        let center_y = self.last_canvas_origin.1 + self.last_canvas_size.1 * 0.5;
        let pad_x = (self.last_canvas_size.0 - self.last_world_size.0 * scale) * 0.5;
        let pad_y = (self.last_canvas_size.1 - self.last_world_size.1 * scale) * 0.5;

        let world_screen_x =
            self.last_canvas_origin.0 + pad_x + self.offset_px.0 + position.x * scale;
        let world_screen_y =
            self.last_canvas_origin.1 + pad_y + self.offset_px.1 + position.y * scale;

        self.offset_px.0 += center_x - world_screen_x;
        self.offset_px.1 += center_y - world_screen_y;
    }

    fn screen_to_world(&self, point: Point<Pixels>) -> Option<(f32, f32)> {
        let scale = self.last_scale;
        if scale <= f32::EPSILON {
            return None;
        }
        let canvas_x = f32::from(point.x);
        let canvas_y = f32::from(point.y);
        let origin_x = self.last_canvas_origin.0;
        let origin_y = self.last_canvas_origin.1;
        let canvas_width = self.last_canvas_size.0;
        let canvas_height = self.last_canvas_size.1;
        let world_w = self.last_world_size.0;
        let world_h = self.last_world_size.1;

        let render_w = world_w * scale;
        let render_h = world_h * scale;
        let pad_x = (canvas_width - render_w) * 0.5;
        let pad_y = (canvas_height - render_h) * 0.5;

        let world_x = (canvas_x - origin_x - pad_x - self.offset_px.0) / scale;
        let world_y = (canvas_y - origin_y - pad_y - self.offset_px.1) / scale;

        if !world_x.is_finite() || !world_y.is_finite() {
            return None;
        }

        if world_x < 0.0 || world_y < 0.0 || world_x > world_w || world_y > world_h {
            return None;
        }

        Some((world_x, world_y))
    }
}

fn paint_terrain_layer(
    terrain: &TerrainFrame,
    offset_x: f32,
    offset_y: f32,
    scale: f32,
    daylight: f32,
    palette: ColorPaletteMode,
    window: &mut Window,
) {
    let width = terrain.dimensions.0 as usize;
    let height = terrain.dimensions.1 as usize;
    if width == 0 || height == 0 {
        return;
    }

    let cell_world = terrain.cell_size as f32;
    let cell_px = (cell_world * scale).max(1.0);
    let highlight_shift = (daylight * 0.45 + 0.35).clamp(0.2, 0.9);

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let Some(tile) = terrain.tiles.get(idx).copied() else {
                continue;
            };

            let px_x = offset_x + (x as f32 * cell_world * scale);
            let px_y = offset_y + (y as f32 * cell_world * scale);

            let surface = terrain_surface_color(tile, daylight, palette);
            let cell_bounds =
                Bounds::new(point(px(px_x), px(px_y)), size(px(cell_px), px(cell_px)));
            window.paint_quad(fill(cell_bounds, Background::from(surface)));

            if tile.slope > 0.12 {
                let stroke_width = (0.55 + tile.slope * 1.1) * scale.clamp(0.6, 3.0);
                let accent = terrain_slope_accent_color(tile, highlight_shift, palette);
                let mut builder = PathBuilder::stroke(px(stroke_width.min(cell_px * 0.85)));
                let diag_bias = if tile.accent > 0.5 {
                    (0.35, 0.9)
                } else {
                    (0.1, 0.65)
                };
                builder.move_to(point(px(px_x + cell_px * diag_bias.0), px(px_y)));
                builder.line_to(point(px(px_x), px(px_y + cell_px * diag_bias.1)));
                if let Ok(path) = builder.build() {
                    window.paint_path(path, accent);
                }
            }

            if matches!(tile.kind, TerrainKind::Bloom) && tile.accent > 0.66 {
                let blossom = terrain_bloom_color(tile, palette);
                let bloom_size = (cell_px * (0.25 + tile.accent * 0.18)).min(cell_px * 0.6);
                let half = bloom_size * 0.5;
                let bloom_bounds = Bounds::new(
                    point(
                        px(px_x + cell_px * 0.5 - half),
                        px(px_y + cell_px * 0.5 - half),
                    ),
                    size(px(bloom_size), px(bloom_size)),
                );
                window.paint_quad(fill(bloom_bounds, Background::from(blossom)));
            }

            if matches!(
                tile.kind,
                TerrainKind::ShallowWater | TerrainKind::DeepWater
            ) {
                let caustic = terrain_water_caustic_color(tile, daylight, palette);
                let wave_height =
                    (cell_px * 0.12 + tile.accent * cell_px * 0.18).min(cell_px * 0.35);
                let mut builder =
                    PathBuilder::stroke(px((cell_px * 0.08 + tile.accent * 0.18).clamp(0.5, 2.6)));
                builder.move_to(point(px(px_x), px(px_y + wave_height)));
                builder.line_to(point(
                    px(px_x + cell_px),
                    px(px_y + wave_height * 0.45 + tile.accent * 0.22 * cell_px),
                ));
                if let Ok(path) = builder.build() {
                    window.paint_path(path, caustic);
                }
            }
        }
    }
}

fn terrain_surface_color(
    tile: TerrainTileVisual,
    daylight: f32,
    palette: ColorPaletteMode,
) -> Rgba {
    let base = match tile.kind {
        TerrainKind::DeepWater => [0.05, 0.11, 0.23],
        TerrainKind::ShallowWater => [0.12, 0.34, 0.46],
        TerrainKind::Sand => [0.40, 0.33, 0.20],
        TerrainKind::Grass => [0.20, 0.34, 0.18],
        TerrainKind::Bloom => [0.18, 0.42, 0.22],
        TerrainKind::Rock => [0.34, 0.33, 0.41],
    };

    let mut color = rgba_from_triplet_with_alpha(base, 1.0);
    let brightness = match tile.kind {
        TerrainKind::DeepWater => (0.42 + daylight * 0.25 + tile.moisture * 0.2).clamp(0.25, 1.05),
        TerrainKind::ShallowWater => {
            (0.55 + daylight * 0.35 + tile.moisture * 0.3).clamp(0.4, 1.25)
        }
        TerrainKind::Sand => (0.72 + daylight * 0.18 + tile.elevation * 0.35).clamp(0.45, 1.35),
        TerrainKind::Grass => (0.62 + daylight * 0.28 + tile.moisture * 0.4).clamp(0.4, 1.35),
        TerrainKind::Bloom => (0.68 + daylight * 0.35 + tile.moisture * 0.5).clamp(0.45, 1.45),
        TerrainKind::Rock => (0.60 + daylight * 0.22 + tile.slope * 0.45).clamp(0.35, 1.25),
    };
    color = scale_rgb(color, brightness);

    if matches!(tile.kind, TerrainKind::Bloom | TerrainKind::Grass) {
        color = scale_rgb(color, 0.9 + tile.moisture * 0.3 + tile.accent * 0.05);
    } else if matches!(tile.kind, TerrainKind::Sand) {
        color = scale_rgb(color, 0.9 + tile.accent * 0.08);
    } else if matches!(tile.kind, TerrainKind::Rock) {
        color = scale_rgb(color, 0.85 + tile.slope * 0.3);
    }

    apply_palette(color, palette)
}

fn terrain_slope_accent_color(
    tile: TerrainTileVisual,
    highlight_shift: f32,
    palette: ColorPaletteMode,
) -> Rgba {
    let accent = match tile.kind {
        TerrainKind::DeepWater => [0.23, 0.58, 0.86],
        TerrainKind::ShallowWater => [0.45, 0.82, 0.98],
        TerrainKind::Sand => [0.75, 0.57, 0.29],
        TerrainKind::Grass => [0.34, 0.66, 0.28],
        TerrainKind::Bloom => [0.52, 0.86, 0.44],
        TerrainKind::Rock => [0.78, 0.78, 0.86],
    };
    let alpha = (0.09 + tile.slope * highlight_shift).clamp(0.04, 0.42);
    let mut color = rgba_from_triplet_with_alpha(accent, alpha);
    color = scale_rgb(color, 0.85 + tile.accent * 0.4);
    apply_palette(color, palette)
}

fn terrain_bloom_color(tile: TerrainTileVisual, palette: ColorPaletteMode) -> Rgba {
    let strength = ((tile.accent - 0.66) * 1.6).clamp(0.0, 1.0);
    let alpha = (0.12 + strength * 0.28).clamp(0.08, 0.35);
    let mut color = rgba_from_triplet_with_alpha([0.96, 0.62, 0.84], alpha);
    color = scale_rgb(color, 0.85 + tile.moisture * 0.35);
    apply_palette(color, palette)
}

fn terrain_water_caustic_color(
    tile: TerrainTileVisual,
    daylight: f32,
    palette: ColorPaletteMode,
) -> Rgba {
    let base = if matches!(tile.kind, TerrainKind::DeepWater) {
        [0.36, 0.74, 0.96]
    } else {
        [0.54, 0.90, 1.0]
    };
    let alpha = (0.10 + daylight * 0.12 + tile.accent * 0.18).clamp(0.05, 0.32);
    let mut color = rgba_from_triplet_with_alpha(base, alpha);
    color = scale_rgb(color, 0.9 + tile.moisture * 0.2);
    apply_palette(color, palette)
}

fn paint_sparkline(bounds: Bounds<Pixels>, state: SparklineState, window: &mut Window) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let width = f32::from(bounds_size.width).max(1.0);
    let height = f32::from(bounds_size.height).max(1.0);
    let samples = state.values.len();
    if samples < 2 {
        return;
    }
    let step = width / (samples.saturating_sub(1) as f32);

    let mut fill_builder = PathBuilder::fill();
    fill_builder.move_to(point(
        px(f32::from(origin.x)),
        px(f32::from(origin.y) + height),
    ));
    for (idx, value) in state.values.iter().enumerate() {
        let x = f32::from(origin.x) + step * idx as f32;
        let y = f32::from(origin.y) + height - value.clamp(0.0, 1.0) * height;
        fill_builder.line_to(point(px(x), px(y)));
    }
    fill_builder.line_to(point(
        px(f32::from(origin.x) + width),
        px(f32::from(origin.y) + height),
    ));
    fill_builder.close();
    if let Ok(path) = fill_builder.build() {
        let mut fill_color = state.accent;
        fill_color.a = if state.trend >= 0.0 { 0.18 } else { 0.12 };
        window.paint_path(path, fill_color);
    }

    let mut stroke_builder = PathBuilder::stroke(px(1.6));
    for (idx, value) in state.values.iter().enumerate() {
        let x = f32::from(origin.x) + step * idx as f32;
        let y = f32::from(origin.y) + height - value.clamp(0.0, 1.0) * height;
        if idx == 0 {
            stroke_builder.move_to(point(px(x), px(y)));
        } else {
            stroke_builder.line_to(point(px(x), px(y)));
        }
    }
    if let Ok(path) = stroke_builder.build() {
        let mut stroke_color = state.accent;
        stroke_color.a = 0.85;
        window.paint_path(path, stroke_color);
    }

    let marker_value = state.values.last().copied().unwrap_or(0.5).clamp(0.0, 1.0);
    let marker_size = 4.0;
    let marker_x = f32::from(origin.x) + width;
    let marker_y = f32::from(origin.y) + height - marker_value * height;
    let marker_bounds = Bounds::new(
        point(
            px(marker_x - marker_size * 0.5),
            px(marker_y - marker_size * 0.5),
        ),
        size(px(marker_size), px(marker_size)),
    );
    let mut marker_color = state.accent;
    marker_color.a = 1.0;
    window.paint_quad(fill(marker_bounds, Background::from(marker_color)));
}

fn paint_metric_badge(bounds: Bounds<Pixels>, state: MetricBadgeState, window: &mut Window) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let width = f32::from(bounds_size.width).max(1.0);
    let height = f32::from(bounds_size.height).max(1.0);
    let center_x = f32::from(origin.x) + width * 0.5;
    let center_y = f32::from(origin.y) + height * 0.5;
    let radius = width.min(height) * 0.5;

    let mut hex_builder = PathBuilder::fill();
    for step_idx in 0..6 {
        let angle = (PI / 3.0) * step_idx as f32 - FRAC_PI_2;
        let x = center_x + angle.cos() * radius;
        let y = center_y + angle.sin() * radius;
        if step_idx == 0 {
            hex_builder.move_to(point(px(x), px(y)));
        } else {
            hex_builder.line_to(point(px(x), px(y)));
        }
    }
    hex_builder.close();
    if let Ok(path) = hex_builder.build() {
        let mut outer = state.accent;
        outer.a = 0.45;
        window.paint_path(path, outer);
    }

    let inner_radius = radius * 0.58;
    let mut diamond = PathBuilder::fill();
    for idx in 0..4 {
        let angle = FRAC_PI_2 * idx as f32 + FRAC_PI_4;
        let x = center_x + angle.cos() * inner_radius;
        let y = center_y + angle.sin() * inner_radius;
        if idx == 0 {
            diamond.move_to(point(px(x), px(y)));
        } else {
            diamond.line_to(point(px(x), px(y)));
        }
    }
    diamond.close();
    if let Ok(path) = diamond.build() {
        let mut inner = state.accent;
        inner.a = 0.9;
        window.paint_path(path, inner);
    }

    let bar_width = inner_radius * 0.36;
    let bar_bounds = Bounds::new(
        point(px(center_x - bar_width * 0.5), px(center_y - inner_radius)),
        size(px(bar_width), px(inner_radius * 2.0)),
    );
    let mut bar_color = state.accent;
    bar_color.a = 0.65;
    window.paint_quad(fill(bar_bounds, Background::from(bar_color)));
}

fn paint_header_badge(bounds: Bounds<Pixels>, state: HeaderBadgeState, window: &mut Window) {
    let origin = bounds.origin;
    let size = bounds.size;
    let cx = f32::from(origin.x) + f32::from(size.width) * 0.5;
    let cy = f32::from(origin.y) + f32::from(size.height) * 0.5;
    let radius = f32::from(size.width.min(size.height)) * 0.5 - 1.5;

    let base = apply_palette(rgba_from_hex(0x0f172a, 1.0), state.palette);
    window.paint_quad(fill(bounds, Background::from(base)));

    let phase = state.phase;
    let glow_radius = radius * 0.9;
    for i in 0..5 {
        let t = i as f32 / 5.0;
        let angle = phase + t * std::f32::consts::TAU;
        let px = cx + angle.cos() * glow_radius;
        let py = cy + angle.sin() * glow_radius;
        let orb_radius = 6.0 + (angle.sin() * 2.0);
        let mut orb = PathBuilder::fill();
        append_circle_polygon(&mut orb, px, py, orb_radius);
        if let Ok(path) = orb.build() {
            let color = apply_palette(rgba_from_hex(0x60a5fa, 0.18 + t * 0.2), state.palette);
            window.paint_path(path, color);
        }
    }

    let mut ring = PathBuilder::stroke(px(3.0));
    append_arc_polyline(&mut ring, cx, cy, radius, 0.0, 360.0);
    if let Ok(path) = ring.build() {
        let ring_color = apply_palette(rgba_from_hex(0x38bdf8, 0.85), state.palette);
        window.paint_path(path, ring_color);
    }

    let mut inner = PathBuilder::fill();
    append_circle_polygon(&mut inner, cx, cy, radius * 0.55);
    if let Ok(path) = inner.build() {
        let inner_color = apply_palette(rgba_from_hex(0x1d4ed8, 0.5), state.palette);
        window.paint_path(path, inner_color);
    }

    let mut pulse = PathBuilder::stroke(px(1.6));
    let sweep = 140.0 + (phase.sin() + 1.0) * 100.0;
    append_arc_polyline(&mut pulse, cx, cy, radius * 0.68, -sweep * 0.5, sweep);
    if let Ok(path) = pulse.build() {
        let pulse_color = apply_palette(rgba_from_hex(0xfacc15, 0.75), state.palette);
        window.paint_path(path, pulse_color);
    }
}

fn apply_post_processing(
    stack: &PostProcessStack,
    palette: ColorPaletteMode,
    bounds: Bounds<Pixels>,
    window: &mut Window,
    daylight: f32,
    scale: f32,
) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let width_px = f32::from(bounds_size.width).max(1.0);
    let height_px = f32::from(bounds_size.height).max(1.0);
    let origin_x = f32::from(origin.x);
    let origin_y = f32::from(origin.y);

    for pass in &stack.passes {
        match *pass {
            PostProcessPass::ColorGrade {
                lift,
                gain,
                temperature,
            } => {
                if lift > 0.001 {
                    let lift_color =
                        apply_palette(rgba_from_hex(0x02060f, lift.clamp(0.0, 0.2)), palette);
                    window.paint_quad(fill(bounds, Background::from(lift_color)));
                }
                if temperature.abs() > 0.001 {
                    let temp_hex = if temperature >= 0.0 {
                        0xffa94d
                    } else {
                        0x3b82f6
                    };
                    let temp_alpha = temperature.abs().clamp(0.0, 0.25) * 0.6;
                    if temp_alpha > 0.0 {
                        let temp_color =
                            apply_palette(rgba_from_hex(temp_hex, temp_alpha), palette);
                        window.paint_quad(fill(bounds, Background::from(temp_color)));
                    }
                }
                if gain > 1.0 {
                    let gain_alpha = (gain - 1.0).clamp(0.0, 0.4);
                    if gain_alpha > 0.0 {
                        let gain_bounds = Bounds::new(
                            point(
                                px(origin_x + width_px * 0.12),
                                px(origin_y + height_px * 0.12),
                            ),
                            size(px(width_px * 0.76), px(height_px * 0.76)),
                        );
                        let gain_color =
                            apply_palette(rgba_from_hex(0xf1f5f9, gain_alpha * 0.35), palette);
                        window.paint_quad(fill(gain_bounds, Background::from(gain_color)));
                    }
                }
            }
            PostProcessPass::Vignette { strength } => {
                if strength > 0.01 {
                    let alpha = (0.22 + (1.0 - daylight) * 0.14) * strength;
                    let edge_color =
                        apply_palette(rgba_from_hex(0x01040c, alpha.clamp(0.05, 0.55)), palette);
                    let top_bounds = Bounds::new(
                        point(px(origin_x), px(origin_y)),
                        size(px(width_px), px(height_px * 0.18)),
                    );
                    let bottom_bounds = Bounds::new(
                        point(px(origin_x), px(origin_y + height_px * 0.82)),
                        size(px(width_px), px(height_px * 0.18)),
                    );
                    window.paint_quad(fill(top_bounds, Background::from(edge_color)));
                    window.paint_quad(fill(bottom_bounds, Background::from(edge_color)));

                    let side_alpha = (alpha * 0.8).clamp(0.04, 0.45);
                    let side_color = apply_palette(rgba_from_hex(0x020816, side_alpha), palette);
                    let side_width = width_px * 0.08;
                    let left_bounds = Bounds::new(
                        point(px(origin_x), px(origin_y)),
                        size(px(side_width), px(height_px)),
                    );
                    let right_bounds = Bounds::new(
                        point(px(origin_x + width_px - side_width), px(origin_y)),
                        size(px(side_width), px(height_px)),
                    );
                    window.paint_quad(fill(left_bounds, Background::from(side_color)));
                    window.paint_quad(fill(right_bounds, Background::from(side_color)));
                }
            }
            PostProcessPass::Bloom { strength } => {
                if strength > 0.01 {
                    let bloom_width = width_px * (0.48 + strength * 0.36);
                    let bloom_height = height_px * (0.48 + strength * 0.36);
                    let bloom_bounds = Bounds::new(
                        point(
                            px(origin_x + (width_px - bloom_width) * 0.5),
                            px(origin_y + (height_px - bloom_height) * 0.5),
                        ),
                        size(px(bloom_width), px(bloom_height)),
                    );
                    let bloom_color = apply_palette(
                        lerp_rgba(
                            rgba_from_hex(0x3b82f6, 0.14 * strength),
                            rgba_from_hex(0x22c55e, 0.10 * strength),
                            daylight.clamp(0.0, 1.0),
                        ),
                        palette,
                    );
                    window.paint_quad(fill(bloom_bounds, Background::from(bloom_color)));
                }
            }
            PostProcessPass::Scanlines { intensity, spacing } => {
                if intensity > 0.01 {
                    let spacing_px = (spacing / scale).clamp(2.0, 9.0);
                    let alpha = (0.07 * intensity).clamp(0.01, 0.2);
                    let mut y = origin_y;
                    while y < origin_y + height_px {
                        let scanline_bounds =
                            Bounds::new(point(px(origin_x), px(y)), size(px(width_px), px(1.0)));
                        window.paint_quad(fill(
                            scanline_bounds,
                            Background::from(apply_palette(
                                rgba_from_hex(0x020816, alpha),
                                palette,
                            )),
                        ));
                        y += spacing_px;
                    }
                }
            }
            PostProcessPass::FilmGrain { strength, seed } => {
                if strength > 0.01 {
                    paint_film_grain(bounds, seed, strength, palette, window);
                }
            }
        }
    }
}

fn paint_film_grain(
    bounds: Bounds<Pixels>,
    seed: u64,
    strength: f32,
    palette: ColorPaletteMode,
    window: &mut Window,
) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let width_px = f32::from(bounds_size.width).max(1.0);
    let height_px = f32::from(bounds_size.height).max(1.0);
    let cols = (width_px / 24.0).clamp(16.0, 64.0) as u32;
    let rows = (height_px / 24.0).clamp(12.0, 48.0) as u32;
    let cell_w = width_px / cols as f32;
    let cell_h = height_px / rows as f32;

    for row in 0..rows {
        for col in 0..cols {
            let noise = hashed_noise(seed, row as u64, col as u64);
            if noise < 0.2 {
                continue;
            }
            let alpha = (0.02 + noise * 0.06) * strength;
            if alpha < 0.01 {
                continue;
            }
            let base_hex = if noise > 0.6 { 0xffffff } else { 0x0f172a };
            let color = apply_palette(rgba_from_hex(base_hex, alpha.clamp(0.01, 0.12)), palette);
            let x = f32::from(origin.x) + col as f32 * cell_w;
            let y = f32::from(origin.y) + row as f32 * cell_h;
            let cell_bounds = Bounds::new(point(px(x), px(y)), size(px(cell_w), px(cell_h)));
            window.paint_quad(fill(cell_bounds, Background::from(color)));
        }
    }
}

fn hashed_noise(seed: u64, row: u64, col: u64) -> f32 {
    let mut value = seed.wrapping_add(row.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        ^ col.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 29;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    value ^= value >> 32;
    (value as f64 / u64::MAX as f64) as f32
}

fn paint_debug_overlays(
    frame: &RenderFrame,
    focus_agent: Option<AgentId>,
    debug: DebugOverlayState,
    offset_x: f32,
    offset_y: f32,
    scale: f32,
    window: &mut Window,
) {
    if !debug.enabled {
        return;
    }

    for agent in &frame.agents {
        let px_x = offset_x + agent.position.x * scale;
        let px_y = offset_y + agent.position.y * scale;

        if debug.show_sense_radius
            && frame.sense_radius > 0.0
            && matches!(
                agent.selection,
                SelectionState::Selected | SelectionState::Hovered
            )
        {
            let radius_px = (frame.sense_radius * scale).max(4.0);
            let mut circle = PathBuilder::stroke(px(1.4));
            append_arc_polyline(&mut circle, px_x, px_y, radius_px, 0.0, 360.0);
            if let Ok(path) = circle.build() {
                window.paint_path(path, rgba_from_hex(0x38bdf8, 0.35));
            }
        }

        if debug.show_velocity {
            let vx = agent.velocity.vx;
            let vy = agent.velocity.vy;
            let speed = (vx * vx + vy * vy).sqrt();
            if speed > 1e-3 {
                let norm_x = vx / speed;
                let norm_y = vy / speed;
                let min_length = (frame.agent_base_radius * 1.5).max(8.0) * scale;
                let dynamic_length = (speed * frame.sense_radius)
                    .clamp(frame.agent_base_radius, frame.sense_radius * 1.5)
                    * scale;
                let arrow_length = dynamic_length.max(min_length);
                let tip_x = px_x + norm_x * arrow_length;
                let tip_y = px_y + norm_y * arrow_length;

                let mut shaft = PathBuilder::stroke(px(1.6));
                shaft.move_to(point(px(px_x), px(px_y)));
                shaft.line_to(point(px(tip_x), px(tip_y)));
                if let Ok(path) = shaft.build() {
                    window.paint_path(path, rgba_from_hex(0x22d3ee, 0.85));
                }

                let head_size = (arrow_length * 0.18).clamp(4.0, 14.0);
                let angle = vy.atan2(vx);
                let left_angle = angle + PI - 0.5;
                let right_angle = angle + PI + 0.5;

                let left_point = point(
                    px(tip_x + head_size * left_angle.cos()),
                    px(tip_y + head_size * left_angle.sin()),
                );
                let right_point = point(
                    px(tip_x + head_size * right_angle.cos()),
                    px(tip_y + head_size * right_angle.sin()),
                );

                let mut head = PathBuilder::stroke(px(1.2));
                head.move_to(point(px(tip_x), px(tip_y)));
                head.line_to(left_point);
                head.move_to(point(px(tip_x), px(tip_y)));
                head.line_to(right_point);
                if let Ok(path) = head.build() {
                    window.paint_path(path, rgba_from_hex(0xe0f2fe, 0.78));
                }
            }
        }

        if Some(agent.agent_id) == focus_agent {
            let cross = (frame.agent_base_radius * scale).max(6.0);
            let mut crosshair = PathBuilder::stroke(px(1.6));
            crosshair.move_to(point(px(px_x - cross), px(px_y)));
            crosshair.line_to(point(px(px_x + cross), px(px_y)));
            crosshair.move_to(point(px(px_x), px(px_y - cross)));
            crosshair.line_to(point(px(px_x), px(px_y + cross)));
            if let Ok(path) = crosshair.build() {
                window.paint_path(path, rgba_from_hex(0xfacc15, 0.9));
            }
        }
    }
}

fn paint_frame(
    frame: &RenderFrame,
    camera: &Arc<Mutex<CameraState>>,
    focus_agent: Option<AgentId>,
    controls: ControlsSnapshot,
    debug: DebugOverlayState,
    follow_target: Option<Position>,
    bounds: Bounds<Pixels>,
    window: &mut Window,
) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;

    let mut camera_guard = camera.lock().expect("camera lock poisoned");

    let world_w = frame.world_size.0.max(1.0);
    let world_h = frame.world_size.1.max(1.0);

    let width_px = f32::from(bounds_size.width).max(1.0);
    let height_px = f32::from(bounds_size.height).max(1.0);

    let base_scale = (width_px / world_w).min(height_px / world_h).max(0.000_1);
    let scale = base_scale * camera_guard.zoom;
    let render_w = world_w * scale;
    let render_h = world_h * scale;
    let pad_x = (width_px - render_w) * 0.5;
    let pad_y = (height_px - render_h) * 0.5;
    let offset_x = f32::from(origin.x) + pad_x + camera_guard.offset_px.0;
    let offset_y = f32::from(origin.y) + pad_y + camera_guard.offset_px.1;

    camera_guard.record_render_metrics(
        (f32::from(origin.x), f32::from(origin.y)),
        (width_px, height_px),
        frame.world_size,
        base_scale,
    );
    if controls.follow_mode != FollowMode::Off {
        if let Some(target) = follow_target {
            camera_guard.center_on(target);
        }
    }
    drop(camera_guard);

    let day_phase = frame.tick as f32 * 0.00025;
    let daylight = day_phase.sin() * 0.5 + 0.5;

    let sky_base = lerp_rgba(
        rgba_from_hex(0x050b16, 1.0),
        rgba_from_hex(0x173f6a, 1.0),
        daylight,
    );
    window.paint_quad(fill(
        bounds,
        Background::from(apply_palette(sky_base, frame.palette)),
    ));

    let horizon_height = height_px * 0.35;
    if horizon_height > 1.0 {
        let horizon_bounds = Bounds::new(
            point(
                px(f32::from(origin.x)),
                px(f32::from(origin.y) + height_px - horizon_height),
            ),
            size(px(width_px), px(horizon_height)),
        );
        let horizon_color = apply_palette(
            rgba_from_hex(0xffa94d, (0.12 + 0.25 * daylight).clamp(0.0, 0.3)),
            frame.palette,
        );
        window.paint_quad(fill(horizon_bounds, Background::from(horizon_color)));
    }

    let aurora_strength = (1.0 - daylight).clamp(0.0, 1.0);
    if aurora_strength > 0.05 {
        let aurora_bounds = Bounds::new(
            point(px(f32::from(origin.x)), px(f32::from(origin.y))),
            size(px(width_px), px(height_px * 0.25)),
        );
        let aurora_color = apply_palette(
            rgba_from_hex(0x2fd3ff, 0.18 * aurora_strength),
            frame.palette,
        );
        window.paint_quad(fill(aurora_bounds, Background::from(aurora_color)));
    }

    let food_w = frame.food_dimensions.0 as usize;
    let food_h = frame.food_dimensions.1 as usize;
    paint_terrain_layer(
        &frame.terrain,
        offset_x,
        offset_y,
        scale,
        daylight,
        frame.palette,
        window,
    );
    let cell_world = frame.food_cell_size as f32;
    let cell_px = (cell_world * scale).max(1.0);
    let max_food = frame.food_max.max(f32::EPSILON);

    if controls.draw_food {
        for y in 0..food_h {
            for x in 0..food_w {
                let idx = y * food_w + x;
                let value = frame.food_cells.get(idx).copied().unwrap_or_default();
                if value <= 0.001 {
                    continue;
                }
                let intensity = (value / max_food).clamp(0.0, 1.0);
                let mut color = food_color(intensity);
                let shade_wave =
                    ((x as f32 * 0.35 + y as f32 * 0.27) + day_phase).sin() * 0.5 + 0.5;
                let shade = (0.75 + 0.35 * shade_wave).clamp(0.0, 1.3);
                color = scale_rgb(color, shade);
                color = apply_palette(color, frame.palette);
                let px_x = offset_x + (x as f32 * cell_world * scale);
                let px_y = offset_y + (y as f32 * cell_world * scale);
                let cell_bounds =
                    Bounds::new(point(px(px_x), px(px_y)), size(px(cell_px), px(cell_px)));
                window.paint_quad(fill(cell_bounds, Background::from(color)));
            }
        }
    }

    if controls.draw_agents {
        for agent in &frame.agents {
            let px_x = offset_x + agent.position.x * scale;
            let px_y = offset_y + agent.position.y * scale;
            let dynamic_radius = (frame.agent_base_radius + agent.spike_length * 0.25).max(6.0);
            let size_px = (dynamic_radius * scale).max(2.0);
            let half = size_px * 0.5;

            let mut highlight_layers: Vec<(f32, Rgba)> = Vec::new();
            match agent.selection {
                SelectionState::Selected => highlight_layers.push((
                    1.8,
                    apply_palette(
                        Rgba {
                            r: 0.20,
                            g: 0.65,
                            b: 0.96,
                            a: 0.35,
                        },
                        frame.palette,
                    ),
                )),
                SelectionState::Hovered => highlight_layers.push((
                    1.4,
                    apply_palette(
                        Rgba {
                            r: 0.92,
                            g: 0.58,
                            b: 0.20,
                            a: 0.30,
                        },
                        frame.palette,
                    ),
                )),
                SelectionState::None => {}
            }

            if focus_agent == Some(agent.agent_id) {
                highlight_layers.push((
                    2.05,
                    apply_palette(
                        Rgba {
                            r: 0.45,
                            g: 0.88,
                            b: 0.97,
                            a: 0.32,
                        },
                        frame.palette,
                    ),
                ));
            }

            for (factor, highlight) in highlight_layers {
                let highlight_size = size_px * factor;
                let highlight_half = highlight_size * 0.5;
                let highlight_bounds = Bounds::new(
                    point(px(px_x - highlight_half), px(px_y - highlight_half)),
                    size(px(highlight_size), px(highlight_size)),
                );
                window.paint_quad(fill(highlight_bounds, Background::from(highlight)));
            }

            if controls.agent_outline {
                let outline_radius = (size_px * 0.55).max(3.0);
                let mut outline = PathBuilder::stroke(px(1.8));
                append_arc_polyline(&mut outline, px_x, px_y, outline_radius, 0.0, 360.0);
                if let Ok(path) = outline.build() {
                    window.paint_path(path, rgba_from_hex(0xffffff, 0.35));
                }
            }

            if agent.indicator.intensity > 0.05 {
                let effect = agent.indicator.intensity.clamp(0.0, 1.0);
                let indicator_size = size_px * (1.2 + effect * 1.4);
                let indicator_half = indicator_size * 0.5;
                let indicator_bounds = Bounds::new(
                    point(px(px_x - indicator_half), px(px_y - indicator_half)),
                    size(px(indicator_size), px(indicator_size)),
                );
                let indicator_color = apply_palette(
                    rgba_from_triplet_with_alpha(agent.indicator.color, 0.15 + 0.35 * effect),
                    frame.palette,
                );
                window.paint_quad(fill(indicator_bounds, Background::from(indicator_color)));
            }

            if agent.reproduction_intent > 0.2 {
                let pulse = agent.reproduction_intent.clamp(0.0, 1.0);
                let pulse_size = size_px * (1.8 + pulse * 1.6);
                let pulse_half = pulse_size * 0.5;
                let pulse_bounds = Bounds::new(
                    point(px(px_x - pulse_half), px(px_y - pulse_half)),
                    size(px(pulse_size), px(pulse_size)),
                );
                let pulse_color = apply_palette(
                    Rgba {
                        r: 0.88,
                        g: 0.36,
                        b: 0.86,
                        a: 0.18 + 0.28 * pulse,
                    },
                    frame.palette,
                );
                window.paint_quad(fill(pulse_bounds, Background::from(pulse_color)));
            }

            if agent.spiked {
                let spike_size = size_px * 2.2;
                let spike_half = spike_size * 0.5;
                let spike_bounds = Bounds::new(
                    point(px(px_x - spike_half), px(px_y - spike_half)),
                    size(px(spike_size), px(spike_size)),
                );
                let spike_color = apply_palette(
                    Rgba {
                        r: 0.95,
                        g: 0.25,
                        b: 0.35,
                        a: 0.28,
                    },
                    frame.palette,
                );
                window.paint_quad(fill(spike_bounds, Background::from(spike_color)));
            }

            let agent_bounds = Bounds::new(
                point(px(px_x - half), px(px_y - half)),
                size(px(size_px), px(size_px)),
            );
            let shade_wave = ((agent.position.x + agent.position.y) * 0.04 + day_phase).cos();
            let agent_shade = (0.85 + 0.15 * shade_wave).clamp(0.65, 1.1);
            let mut color = agent_color(agent, agent_shade);
            color = apply_palette(color, frame.palette);
            window.paint_quad(fill(agent_bounds, Background::from(color)));
        }
    }

    if controls.draw_agents {
        paint_debug_overlays(frame, focus_agent, debug, offset_x, offset_y, scale, window);
    }

    apply_post_processing(
        &frame.post_stack,
        frame.palette,
        bounds,
        window,
        daylight,
        scale,
    );
}

fn food_color(intensity: f32) -> Rgba {
    let clamped = intensity.clamp(0.0, 1.0);
    Rgba {
        r: 0.06 + 0.25 * clamped,
        g: 0.22 + 0.55 * clamped,
        b: 0.12 + 0.25 * clamped,
        a: 0.2 + 0.45 * clamped,
    }
}

fn agent_color(agent: &AgentRenderData, shade: f32) -> Rgba {
    let base_r = agent.color[0].clamp(0.0, 1.0);
    let base_g = agent.color[1].clamp(0.0, 1.0);
    let base_b = agent.color[2].clamp(0.0, 1.0);
    let health_factor = (agent.health / 2.0).clamp(0.35, 1.0);

    Rgba {
        r: (base_r * health_factor * shade).clamp(0.0, 1.0),
        g: (base_g * health_factor * shade).clamp(0.0, 1.0),
        b: (base_b * health_factor * shade).clamp(0.0, 1.0),
        a: 0.9,
    }
}

fn apply_palette(color: Rgba, palette: ColorPaletteMode) -> Rgba {
    match palette {
        ColorPaletteMode::Natural => color,
        ColorPaletteMode::Deuteranopia => transform_color(
            color,
            [[0.43, 0.72, -0.15], [0.34, 0.57, 0.09], [-0.02, 0.03, 0.97]],
        ),
        ColorPaletteMode::Protanopia => transform_color(
            color,
            [[0.20, 0.99, -0.19], [0.16, 0.79, 0.04], [0.01, -0.01, 1.00]],
        ),
        ColorPaletteMode::Tritanopia => transform_color(
            color,
            [[0.95, 0.05, 0.00], [0.00, 0.43, 0.56], [0.00, 0.47, 0.53]],
        ),
        ColorPaletteMode::HighContrast => {
            let luminance = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
            if luminance > 0.5 {
                Rgba {
                    r: (color.r + 0.15).min(1.0),
                    g: (color.g + 0.15).min(1.0),
                    b: (color.b + 0.15).min(1.0),
                    a: color.a,
                }
            } else {
                Rgba {
                    r: (color.r * 0.6).clamp(0.0, 1.0),
                    g: (color.g * 0.6).clamp(0.0, 1.0),
                    b: (color.b * 0.6).clamp(0.0, 1.0),
                    a: color.a,
                }
            }
        }
    }
}

fn transform_color(color: Rgba, matrix: [[f32; 3]; 3]) -> Rgba {
    let r =
        (color.r * matrix[0][0] + color.g * matrix[0][1] + color.b * matrix[0][2]).clamp(0.0, 1.0);
    let g =
        (color.r * matrix[1][0] + color.g * matrix[1][1] + color.b * matrix[1][2]).clamp(0.0, 1.0);
    let b =
        (color.r * matrix[2][0] + color.g * matrix[2][1] + color.b * matrix[2][2]).clamp(0.0, 1.0);
    Rgba {
        r,
        g,
        b,
        a: color.a,
    }
}
