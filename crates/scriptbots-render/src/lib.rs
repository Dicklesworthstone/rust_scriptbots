//! GPUI rendering layer for ScriptBots.

use gpui::{
    App, Application, Background, Bounds, Context, Div, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, Pixels, Point, Rgba, ScrollDelta, ScrollWheelEvent, SharedString,
    Window, WindowBounds, WindowOptions, canvas, div, fill, point, prelude::*, px, rgb, size,
};
use scriptbots_core::{Position, TickSummary, WorldState};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
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
    camera: CameraState,
}

impl SimulationView {
    fn new(world: Arc<Mutex<WorldState>>, title: SharedString) -> Self {
        Self {
            world,
            title,
            camera: CameraState::default(),
        }
    }

    fn snapshot(&self) -> HudSnapshot {
        let mut snapshot = HudSnapshot::default();

        if let Ok(world) = self.world.lock() {
            snapshot.tick = world.tick().0;
            snapshot.epoch = world.epoch();
            snapshot.is_closed = world.is_closed();
            snapshot.agent_count = world.agent_count();

            let config = world.config();
            snapshot.world_size = (config.world_width, config.world_height);
            snapshot.history_capacity = config.history_capacity;
            snapshot.render_frame = RenderFrame::from_world(&world);

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
        }

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

            cards.push(self.metric_card(
                "Tick",
                metrics.tick.to_string(),
                0x38bdf8,
                Some(format!("Epoch {}", snapshot.epoch)),
            ));
            cards.push(self.metric_card(
                "Agents",
                metrics.agent_count.to_string(),
                0x22c55e,
                Some(format!("{} active", snapshot.agent_count)),
            ));
            cards.push(self.metric_card(
                "Births / Deaths",
                format!("{} / {}", metrics.births, metrics.deaths),
                growth_accent,
                Some(growth_label),
            ));
            cards.push(self.metric_card(
                "Avg Energy",
                format!("{:.2}", metrics.average_energy),
                0xf59e0b,
                Some(format!("Total {:.1}", metrics.total_energy)),
            ));
            cards.push(self.metric_card(
                "Avg Health",
                format!("{:.2}", metrics.average_health),
                0x8b5cf6,
                None,
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
            .flex_1()
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
            camera: self.camera.clone(),
        };

        let canvas_element = canvas(
            move |_, _, _| canvas_state.clone(),
            move |bounds, state, window, _| {
                paint_frame(&state.frame, &state.camera, bounds, window)
            },
        )
        .flex_1();

        let canvas_stack = div()
            .relative()
            .flex_1()
            .on_mouse_down(MouseButton::Middle, cx.listener(|this, event: &MouseDownEvent, _, cx| {
                if event.button == MouseButton::Middle {
                    this.camera.start_pan(event.position);
                    cx.notify();
                }
            }))
            .on_mouse_up(MouseButton::Middle, cx.listener(|this, event: &MouseUpEvent, _, _| {
                if event.button == MouseButton::Middle {
                    this.camera.end_pan();
                }
            }))
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                if this.camera.update_pan(event.position) {
                    cx.notify();
                }
            }))
            .on_scroll_wheel(cx.listener(|this, event: &ScrollWheelEvent, _, cx| {
                if this.camera.apply_scroll(event) {
                    cx.notify();
                }
            }))
            .child(canvas_element)
            .child(self.render_overlay(snapshot));

        let footer = div()
            .text_xs()
            .text_color(rgb(0x475569))
            .flex()
            .justify_between()
            .child(format!(
                "World {:.0}×{:.0} units • Zoom {:.2}×",
                frame.world_size.0, frame.world_size.1, self.camera.zoom,
            ))
            .child(format!(
                "Pan X {:.1}, Y {:.1}",
                self.camera.offset_px.0, self.camera.offset_px.1
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
        lines.push(format!(
            "Zoom {:.2}× • Pan ({:.0}, {:.0})",
            self.camera.zoom, self.camera.offset_px.0, self.camera.offset_px.1
        ));

        div()
            .absolute()
            .top(px(12.0))
            .left(px(12.0))
            .bg(rgb(0x111b2b))
            .rounded_md()
            .shadow_md()
            .px_3()
            .py_2()
            .flex()
            .flex_col()
            .gap_1()
            .text_sm()
            .text_color(rgb(0xe2e8f0))
            .children(lines.into_iter().map(|line| div().child(line)))
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
    ) -> Div {
        let accent = rgb(accent_hex);
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
                div()
                    .text_xs()
                    .text_color(accent)
                    .child(label.to_uppercase()),
            )
            .child(div().text_3xl().child(value));

        if let Some(detail_text) = detail {
            card = card.child(div().text_sm().text_color(rgb(0x94a3b8)).child(detail_text));
        }

        card
    }
}

impl Render for SimulationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let snapshot = self.snapshot();

        div()
            .size_full()
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
                    .child(self.render_canvas(&snapshot, cx)),
            )
            .child(self.render_footer(&snapshot))
    }
}

#[derive(Default)]
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
}

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

struct HudHistoryEntry {
    tick: u64,
    agent_count: usize,
    births: usize,
    deaths: usize,
    average_energy: f32,
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
        }
    }
}

#[derive(Clone)]
struct RenderFrame {
    world_size: (f32, f32),
    food_dimensions: (u32, u32),
    food_cell_size: u32,
    food_cells: Vec<f32>,
    food_max: f32,
    agents: Vec<AgentRenderData>,
    agent_base_radius: f32,
}

#[derive(Clone)]
struct AgentRenderData {
    position: Position,
    color: [f32; 3],
    spike_length: f32,
    health: f32,
}

#[derive(Clone)]
struct CanvasState {
    frame: RenderFrame,
    camera: CameraState,
}

impl RenderFrame {
    fn from_world(world: &WorldState) -> Option<Self> {
        let food = world.food();
        let width = food.width();
        let height = food.height();
        if width == 0 || height == 0 {
            return None;
        }

        let config = world.config();
        let columns = world.agents().columns();

        let positions = columns.positions();
        let colors = columns.colors();
        let spikes = columns.spike_lengths();
        let healths = columns.health();

        let agents = positions
            .iter()
            .enumerate()
            .map(|(idx, position)| AgentRenderData {
                position: *position,
                color: colors[idx],
                spike_length: spikes[idx],
                health: healths[idx],
            })
            .collect();

        Some(Self {
            world_size: (config.world_width as f32, config.world_height as f32),
            food_dimensions: (width, height),
            food_cell_size: config.food_cell_size,
            food_cells: food.cells().to_vec(),
            food_max: config.food_max,
            agents,
            agent_base_radius: (config.spike_radius * 0.5).max(12.0),
        })
    }
}

#[derive(Clone)]
struct CameraState {
    offset_px: (f32, f32),
    zoom: f32,
    panning: bool,
    pan_anchor: Option<Point<Pixels>>,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            offset_px: (0.0, 0.0),
            zoom: Self::default_zoom(),
            panning: false,
            pan_anchor: None,
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
        self.zoom = (self.zoom * (1.0 + scroll_y * 0.1)).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        if (self.zoom - old_zoom).abs() < f32::EPSILON {
            return false;
        }
        let ratio = self.zoom / old_zoom;
        self.offset_px.0 *= ratio;
        self.offset_px.1 *= ratio;
        true
    }
}

fn paint_frame(
    frame: &RenderFrame,
    camera: &CameraState,
    bounds: Bounds<Pixels>,
    window: &mut Window,
) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;

    let world_w = frame.world_size.0.max(1.0);
    let world_h = frame.world_size.1.max(1.0);

    let width_px = f32::from(bounds_size.width).max(1.0);
    let height_px = f32::from(bounds_size.height).max(1.0);

    let base_scale = (width_px / world_w).min(height_px / world_h).max(0.000_1);
    let scale = base_scale * camera.zoom;
    let render_w = world_w * scale;
    let render_h = world_h * scale;
    let pad_x = (width_px - render_w) * 0.5;
    let pad_y = (height_px - render_h) * 0.5;
    let offset_x = f32::from(origin.x) + pad_x + camera.offset_px.0;
    let offset_y = f32::from(origin.y) + pad_y + camera.offset_px.1;

    window.paint_quad(fill(
        bounds,
        Background::from(Rgba {
            r: 0.03,
            g: 0.05,
            b: 0.08,
            a: 1.0,
        }),
    ));

    let food_w = frame.food_dimensions.0 as usize;
    let food_h = frame.food_dimensions.1 as usize;
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
            let color = food_color(intensity);
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
        let agent_bounds = Bounds::new(
            point(px(px_x - half), px(px_y - half)),
            size(px(size_px), px(size_px)),
        );
        let color = agent_color(agent);
        window.paint_quad(fill(agent_bounds, Background::from(color)));
    }
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

fn agent_color(agent: &AgentRenderData) -> Rgba {
    let base_r = agent.color[0].clamp(0.0, 1.0);
    let base_g = agent.color[1].clamp(0.0, 1.0);
    let base_b = agent.color[2].clamp(0.0, 1.0);
    let health_factor = (agent.health / 2.0).clamp(0.35, 1.0);

    Rgba {
        r: (base_r * health_factor).clamp(0.0, 1.0),
        g: (base_g * health_factor).clamp(0.0, 1.0),
        b: (base_b * health_factor).clamp(0.0, 1.0),
        a: 0.9,
    }
}
