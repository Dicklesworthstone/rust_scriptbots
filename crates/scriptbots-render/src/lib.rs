//! GPUI rendering layer for ScriptBots.

use gpui::{
    App, Application, Background, Bounds, Context, Div, KeyDownEvent, Keystroke, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, PathBuilder, Pixels, Point, Rgba, ScrollDelta,
    ScrollWheelEvent, SharedString, Window, WindowBounds, WindowOptions, canvas, div, fill, point,
    prelude::*, px, rgb, size,
};
use scriptbots_core::{
    AgentColumns, AgentId, AgentRuntime, Generation, INPUT_SIZE, IndicatorState, MutationRates,
    OUTPUT_SIZE, Position, SelectionState, TerrainKind, TerrainLayer, TickSummary, TraitModifiers,
    Velocity, WorldState,
};
use std::{
    collections::{BTreeMap, VecDeque},
    f32::consts::PI,
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

use tracing::{error, info};

/// Launch the ScriptBots GPUI shell with an interactive HUD.
pub fn run_demo(world: Arc<Mutex<WorldState>>) {
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
            cx.new(|_| SimulationView::new(Arc::clone(&world_handle), view_title.clone()))
        }) {
            error!(error = ?err, "failed to open ScriptBots window");
            return;
        }

        app.activate(true);
    });
}

struct SimulationView {
    world: Arc<Mutex<WorldState>>,
    title: SharedString,
    camera: Arc<Mutex<CameraState>>,
    inspector: Arc<Mutex<InspectorState>>,
    playback: PlaybackState,
    perf: PerfStats,
    accessibility: AccessibilitySettings,
    bindings: InputBindings,
    key_capture: Option<CommandAction>,
    #[cfg(feature = "audio")]
    audio: Option<AudioState>,
}

impl SimulationView {
    fn new(world: Arc<Mutex<WorldState>>, title: SharedString) -> Self {
        Self {
            world,
            title,
            camera: Arc::new(Mutex::new(CameraState::default())),
            inspector: Arc::new(Mutex::new(InspectorState::default())),
            playback: PlaybackState::new(240),
            perf: PerfStats::new(240),
            accessibility: AccessibilitySettings::default(),
            bindings: InputBindings::default(),
            key_capture: None,
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

    fn snapshot(&mut self) -> HudSnapshot {
        let mut snapshot = HudSnapshot::default();
        let inspector_state = self
            .inspector
            .lock()
            .map(|state| state.clone())
            .unwrap_or_default();

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
        }

        self.playback.record(snapshot.clone());

        snapshot
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

        div()
            .flex()
            .justify_between()
            .items_center()
            .gap_4()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(div().text_3xl().child(self.title.clone()))
                    .child(div().text_sm().text_color(rgb(0x94a3b8)).child(subline)),
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
                format!("{:.2}", metrics.average_energy),
                0xf59e0b,
                Some(format!("Total {:.1}", metrics.total_energy)),
                energy_series.clone(),
            ));
            cards.push(self.metric_card(
                "Avg Health",
                format!("{:.2}", metrics.average_health),
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

        let column_count = cards.len().clamp(1, 4) as u16;

        div().grid().grid_cols(column_count).gap_4().children(cards)
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
        let canvas_state = CanvasState {
            frame: frame.clone(),
            camera: Arc::clone(&self.camera),
            focus_agent: snapshot.inspector.focus_id,
        };

        let canvas_element = canvas(
            move |_, _, _| canvas_state.clone(),
            move |bounds, state, window, _| {
                paint_frame(
                    &state.frame,
                    &state.camera,
                    state.focus_agent,
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
                if let Ok(mut camera) = this.camera.lock()
                    && camera.update_pan(event.position)
                {
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
        }
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
        let playback_controls = self.render_inspector_playback_controls(cx);

        let detail = inspector
            .focused
            .as_ref()
            .map(|detail| self.render_inspector_detail(detail))
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
            .child(selection_list)
            .child(brush_tools)
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

    fn render_inspector_detail(&self, detail: &AgentInspectorDetails) -> Div {
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
        if let Some(action) = self.key_capture {
            lines.push(format!("Rebinding {}...", action.label()));
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

        #[cfg(feature = "audio")]
        self.update_audio(&snapshot);

        content = content.child(self.render_perf_overlay(perf_snapshot));
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
    recent_history: Vec<HudHistoryEntry>,
    render_frame: Option<RenderFrame>,
    inspector: InspectorSnapshot,
}

#[derive(Clone)]
struct VectorHudState {
    population_ratio: f32,
    energy_ratio: f32,
    births_ratio: f32,
    deaths_ratio: f32,
    tick_phase: f32,
    avg_velocity: (f32, f32),
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

        let (avg_velocity, mean_speed, vector_magnitude, max_speed, heading_rad) = snapshot
            .render_frame
            .as_ref()
            .map(|frame| {
                if frame.agents.is_empty() {
                    return ((0.0, 0.0), 0.0, 0.0, 1.0, 0.0);
                }

                let mut sum_vx = 0.0;
                let mut sum_vy = 0.0;
                let mut sum_speed = 0.0;
                let mut max_speed = 0.0;

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
                let max_speed = max_speed.max(mean_speed).max(1e-3);

                (
                    (avg_vx, avg_vy),
                    mean_speed,
                    vector_magnitude,
                    max_speed,
                    heading_rad,
                )
            })
            .unwrap_or(((0.0, 0.0), 0.0, 0.0, 1.0, 0.0));

        Some(Self {
            population_ratio: metrics.agent_count as f32 / max_agents as f32,
            energy_ratio: (metrics.average_energy / energy_max).clamp(0.0, 1.0),
            births_ratio: metrics.births as f32 / max_births as f32,
            deaths_ratio: metrics.deaths as f32 / max_deaths as f32,
            tick_phase: (snapshot.tick % 960) as f32 / 960.0,
            avg_velocity,
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

fn sparkline_from_history<F>(history: &[HudHistoryEntry], map: F) -> Option<SparklineSeries>
where
    F: Fn(&HudHistoryEntry) -> f32,
{
    if history.len() < 2 {
        return None;
    }
    let mut raw: Vec<f32> = history.iter().map(map).collect();
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
    brush_enabled: bool,
    brush_radius: f32,
    probe_enabled: bool,
}

impl Default for InspectorState {
    fn default() -> Self {
        Self {
            focused_agent: None,
            brush_enabled: false,
            brush_radius: 48.0,
            probe_enabled: false,
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
}

impl InspectorSnapshot {
    fn from_world(world: &WorldState, inspector: &InspectorState) -> Self {
        let mut snapshot = InspectorSnapshot {
            total_agents: world.agent_count(),
            brush_enabled: inspector.brush_enabled,
            brush_radius: inspector.brush_radius,
            probe_enabled: inspector.probe_enabled,
            ..InspectorSnapshot::default()
        };

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
}

impl CommandAction {
    fn label(self) -> &'static str {
        match self {
            CommandAction::TogglePlayback => "Toggle playback",
            CommandAction::GoLive => "Jump to live",
            CommandAction::ToggleBrush => "Toggle brush",
            CommandAction::ToggleNarration => "Toggle narration",
            CommandAction::CyclePalette => "Cycle palette",
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
            Keystroke::parse("p").unwrap_or_default(),
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

#[derive(Clone, Copy, Default)]
struct PerfSnapshot {
    latest_ms: f32,
    average_ms: f32,
    min_ms: f32,
    max_ms: f32,
    sample_count: usize,
    fps: f32,
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
    Vignette { strength: f32 },
    Bloom { strength: f32 },
    Scanlines { intensity: f32, spacing: f32 },
    FilmGrain { strength: f32, seed: u64 },
    ColorGrade { lift: f32, gain: f32, temperature: f32 },
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

    let vignette_strength = (0.36
        + day_phase * 0.24
        + closed_bonus
        + (-life_delta).max(0.0) * 0.28)
        .clamp(0.2, 0.82);
    let bloom_strength = (0.24 + day_phase * 0.32 + life_delta.max(0.0) * 0.28)
        .clamp(0.12, 0.78);
    let scanline_intensity = (0.18
        + (1.0 - day_phase) * 0.22
        + closed_bonus * 0.35
        + (-life_delta).max(0.0) * 0.18)
        .clamp(0.08, 0.65);
    let grain_strength = (0.11 + (tick % 4096) as f32 / 4096.0 * 0.08 + tension * 0.06)
        .clamp(0.08, 0.26);

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
        let left_angle = heading + std::f32::consts::PI - 0.4;
        let right_angle = heading + std::f32::consts::PI + 0.4;

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
            append_arc_polyline(&mut ring, cx, cy, ring_inner, -140.0, 280.0 * velocity_ratio);
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

fn apply_post_processing(
    frame: &RenderFrame,
    bounds: Bounds<Pixels>,
    window: &mut Window,
    daylight: f32,
    scale: f32,
) {
    let fx = &frame.post_fx;
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let width_px = f32::from(bounds_size.width).max(1.0);
    let height_px = f32::from(bounds_size.height).max(1.0);
    let origin_x = f32::from(origin.x);
    let origin_y = f32::from(origin.y);
    let palette = frame.palette;

    if fx.vignette_strength > 0.01 {
        let alpha = (0.22 + (1.0 - daylight) * 0.14) * fx.vignette_strength;
        let edge_color = apply_palette(rgba_from_hex(0x01040c, alpha.clamp(0.05, 0.55)), palette);
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

    if fx.bloom_strength > 0.01 {
        let bloom_width = width_px * (0.48 + fx.bloom_strength * 0.36);
        let bloom_height = height_px * (0.48 + fx.bloom_strength * 0.36);
        let bloom_bounds = Bounds::new(
            point(
                px(origin_x + (width_px - bloom_width) * 0.5),
                px(origin_y + (height_px - bloom_height) * 0.5),
            ),
            size(px(bloom_width), px(bloom_height)),
        );
        let bloom_color = apply_palette(
            lerp_rgba(
                rgba_from_hex(0x3b82f6, 0.14 * fx.bloom_strength),
                rgba_from_hex(0x22c55e, 0.10 * fx.bloom_strength),
                daylight.clamp(0.0, 1.0),
            ),
            palette,
        );
        window.paint_quad(fill(bloom_bounds, Background::from(bloom_color)));
    }

    if fx.scanline_intensity > 0.01 {
        let spacing = (4.5 / scale).clamp(2.0, 7.0);
        let alpha = (0.07 * fx.scanline_intensity).clamp(0.01, 0.2);
        let mut y = origin_y;
        while y < origin_y + height_px {
            let scanline_bounds =
                Bounds::new(point(px(origin_x), px(y)), size(px(width_px), px(1.0)));
            window.paint_quad(fill(
                scanline_bounds,
                Background::from(apply_palette(rgba_from_hex(0x020816, alpha), palette)),
            ));
            y += spacing;
        }
    }

    if fx.grain_strength > 0.01 {
        let seed = frame.tick.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let grain = fx_noise(seed);
        let alpha = (0.02 + grain * 0.06) * fx.grain_strength;
        let grain_color = apply_palette(rgba_from_hex(0xffffff, alpha.clamp(0.01, 0.12)), palette);
        window.paint_quad(fill(bounds, Background::from(grain_color)));
    }
}

fn fx_noise(seed: u64) -> f32 {
    let mut value = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    value ^= value >> 33;
    (value as f64 / u64::MAX as f64) as f32
}

fn paint_frame(
    frame: &RenderFrame,
    camera: &Arc<Mutex<CameraState>>,
    focus_agent: Option<AgentId>,
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

    for y in 0..food_h {
        for x in 0..food_w {
            let idx = y * food_w + x;
            let value = frame.food_cells.get(idx).copied().unwrap_or_default();
            if value <= 0.001 {
                continue;
            }
            let intensity = (value / max_food).clamp(0.0, 1.0);
            let mut color = food_color(intensity);
            let shade_wave = ((x as f32 * 0.35 + y as f32 * 0.27) + day_phase).sin() * 0.5 + 0.5;
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

    apply_post_processing(frame, bounds, window, daylight, scale);
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
