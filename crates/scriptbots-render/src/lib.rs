//! GPUI rendering layer for ScriptBots.

use gpui::{App, Application, Canvas, Entity, EntityContext, View, ViewContext, ViewRef};
use scriptbots_core::WorldState;
use std::sync::{Arc, Mutex};
use tracing::info;

/// Root view holding the shared world state reference.
pub struct WorldView {
    world: Arc<Mutex<WorldState>>,
}

impl WorldView {
    fn new(world: Arc<Mutex<WorldState>>) -> Self {
        Self { world }
    }

    fn render_hud(&self, cx: &mut ViewContext<Self>, world: &WorldState) {
        if let Some(summary) = world.history().last() {
            cx.set_title(&format!(
                "ScriptBots — Tick {} — Agents {} (Δ+{} Δ-{}) AvgEnergy {:.2}",
                summary.tick.0,
                summary.agent_count,
                summary.births,
                summary.deaths,
                summary.average_energy,
            ));
        } else {
            cx.set_title("ScriptBots — (no summaries yet)");
        }
    }
}

impl View for WorldView {
    fn render(&mut self, cx: &mut ViewContext<Self>) {
        if let Ok(world) = self.world.lock() {
            self.render_hud(cx, &world);
            let mut canvas = Canvas::new();
            let width = world.config().world_width as f32;
            let height = world.config().world_height as f32;
            canvas.clear(gpui::Color::BLACK);
            let agents = world.agents().columns().positions();
            let health = world.agents().columns().health();
            for (idx, position) in agents.iter().enumerate() {
                let energy = health.get(idx).copied().unwrap_or(0.0).clamp(0.0, 2.0) / 2.0;
                let color = gpui::Color::rgba(energy, 1.0 - energy, 0.2, 1.0);
                canvas.fill_circle(position.x / width, position.y / height, 4.0, color);
            }
            cx.draw(&canvas);
        }
    }
}

/// Bootstraps a minimal GPUI application.
pub fn run_demo(world: Arc<Mutex<WorldState>>) {
    if let Ok(world) = world.lock() {
        if let Some(summary) = world.history().last() {
            info!(
                tick = summary.tick.0,
                agents = summary.agent_count,
                births = summary.births,
                deaths = summary.deaths,
                avg_energy = summary.average_energy,
                "Launching GPUI shell with latest world snapshot",
            );
        }
    }

    Application::new().run(move |cx: &mut App| {
        cx.open_window(|cx| cx.new_view(|cx| WorldView::new(world.clone())));
    });
}
