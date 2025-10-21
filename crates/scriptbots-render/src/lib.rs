//! GPUI rendering layer for ScriptBots.

use gpui::{
    App, Application, Bounds, Context, Div, SharedString, Window, WindowBounds, WindowOptions, div,
    prelude::*, px, rgb, size,
};
use scriptbots_core::{TickSummary, WorldState};
use scriptbots_storage::{MetricReading, PredatorStats, Storage};
use std::sync::{Arc, Mutex};
use tracing::{error, info, warn};

/// Launch the ScriptBots GPUI shell with an interactive HUD.
pub fn run_demo(world: Arc<Mutex<WorldState>>, storage: Arc<Mutex<Storage>>) {
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
    let storage_for_view = Arc::clone(&storage);

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
        let storage_handle = Arc::clone(&storage_for_view);
        let view_title = title_for_view.clone();

        if let Err(err) = app.open_window(options, move |_window, cx| {
            cx.new(|_| {
                SimulationView::new(
                    Arc::clone(&world_handle),
                    Arc::clone(&storage_handle),
                    view_title.clone(),
                )
            })
        }) {
            error!(error = ?err, "failed to open ScriptBots window");
            return;
        }

        app.activate(true);
    });
}

struct SimulationView {
    world: Arc<Mutex<WorldState>>,
    storage: Arc<Mutex<Storage>>,
    title: SharedString,
}

impl SimulationView {
    fn new(
        world: Arc<Mutex<WorldState>>,
        storage: Arc<Mutex<Storage>>,
        title: SharedString,
    ) -> Self {
        Self {
            world,
            storage,
            title,
        }
    }

    fn snapshot(&self) -> HudSnapshot {
        let mut snapshot = HudSnapshot::default();
        let history: Vec<TickSummary> = if let Ok(world) = self.world.lock() {
            snapshot.tick = world.tick().0;
            snapshot.epoch = world.epoch();
            snapshot.is_closed = world.is_closed();
            snapshot.agent_count = world.agent_count();

            let config = world.config();
            snapshot.world_size = (config.world_width, config.world_height);
            snapshot.history_capacity = config.history_capacity;

            let collected: Vec<TickSummary> = world.history().cloned().collect();
            if let Some(latest) = collected.last() {
                snapshot.summary = Some(HudMetrics::from(latest));
            }
            collected
        } else {
            Vec::new()
        };

        snapshot.recent_history = history
            .iter()
            .rev()
            .take(12)
            .map(HudHistoryEntry::from)
            .collect();

        if let Ok(mut storage) = self.storage.lock() {
            match storage.latest_metrics(6) {
                Ok(metrics) => snapshot
                    .storage_metrics
                    .extend(metrics.into_iter().map(HudStorageMetric::from)),
                Err(err) => warn!(?err, "failed to load latest metrics"),
            }

            match storage.top_predators(5) {
                Ok(predators) => snapshot
                    .top_predators
                    .extend(predators.into_iter().map(HudPredator::from)),
                Err(err) => warn!(?err, "failed to load top predators"),
            }
        } else {
            warn!("storage mutex poisoned while collecting snapshot");
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
                format!("{}", metrics.tick),
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

    fn render_analytics(&self, snapshot: &HudSnapshot) -> Div {
        div()
            .flex()
            .gap_4()
            .flex_wrap()
            .child(self.render_storage_metrics(&snapshot.storage_metrics))
            .child(self.render_top_predators(&snapshot.top_predators))
    }

    fn render_storage_metrics(&self, metrics: &[HudStorageMetric]) -> Div {
        let mut card = div()
            .flex()
            .flex_col()
            .gap_3()
            .rounded_lg()
            .bg(rgb(0x0b1220))
            .p_4()
            .min_w(px(360.0))
            .flex_grow()
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x94a3b8))
                    .child("DuckDB Metrics"),
            );

        if metrics.is_empty() {
            return card.child(
                div()
                    .text_sm()
                    .text_color(rgb(0x64748b))
                    .child("Awaiting persisted metrics..."),
            );
        }

        let accents = [0x38bdf8, 0xa855f7, 0x22c55e, 0xfbbf24];
        let mut grid = div().flex().flex_wrap().gap_3();
        for (idx, metric) in metrics.iter().enumerate() {
            let accent = accents[idx % accents.len()];
            grid = grid.child(self.metric_card(
                &metric.name,
                format!("{:.3}", metric.value),
                accent,
                Some(format!("tick {}", metric.tick)),
            ));
        }
        card.child(grid)
    }

    fn render_top_predators(&self, predators: &[HudPredator]) -> Div {
        let mut card = div()
            .flex()
            .flex_col()
            .gap_3()
            .rounded_lg()
            .bg(rgb(0x0b1220))
            .p_4()
            .min_w(px(360.0))
            .flex_grow()
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x94a3b8))
                    .child("Top Predators"),
            );

        if predators.is_empty() {
            return card.child(
                div()
                    .text_sm()
                    .text_color(rgb(0x64748b))
                    .child("No predator data persisted yet."),
            );
        }

        let mut list = div().flex().flex_col().gap_2();
        for (rank, predator) in predators.iter().enumerate() {
            let badge = format!("#{:02}", rank + 1);
            let content = format!(
                "Agent {} • avg energy {:.2} • spike {:.1} • last tick {}",
                predator.agent_id,
                predator.avg_energy,
                predator.max_spike_length,
                predator.last_tick
            );
            list = list.child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .bg(rgb(0x111827))
                    .rounded_md()
                    .px_3()
                    .py_2()
                    .child(div().text_sm().text_color(rgb(0xfacc15)).child(badge))
                    .child(div().text_sm().text_color(rgb(0xe2e8f0)).child(content)),
            );
        }

        card.child(list)
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
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
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
            .child(self.render_analytics(&snapshot))
            .child(
                div()
                    .flex()
                    .gap_4()
                    .flex_1()
                    .child(self.render_history(&snapshot))
                    .child(self.render_canvas_placeholder(&snapshot)),
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
    storage_metrics: Vec<HudStorageMetric>,
    top_predators: Vec<HudPredator>,
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

impl From<&TickSummary> for HudHistoryEntry {
    fn from(summary: &TickSummary) -> Self {
        Self {
            tick: summary.tick.0,
            agent_count: summary.agent_count,
            births: summary.births,
            deaths: summary.deaths,
            average_energy: summary.average_energy,
        }
    }
}

struct HudStorageMetric {
    name: String,
    value: f64,
    tick: i64,
}

impl From<MetricReading> for HudStorageMetric {
    fn from(metric: MetricReading) -> Self {
        Self {
            name: metric.name,
            value: metric.value,
            tick: metric.tick,
        }
    }
}

struct HudPredator {
    agent_id: u64,
    avg_energy: f64,
    max_spike_length: f64,
    last_tick: i64,
}

impl From<PredatorStats> for HudPredator {
    fn from(stats: PredatorStats) -> Self {
        Self {
            agent_id: stats.agent_id,
            avg_energy: stats.avg_energy,
            max_spike_length: stats.max_spike_length,
            last_tick: stats.last_tick,
        }
    }
}
