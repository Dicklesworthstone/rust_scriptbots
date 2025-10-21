//! GPUI rendering layer for ScriptBots.

use gpui::{
    App, Application, Context, IntoElement, Render, Window, WindowOptions, div, prelude::*,
};
use scriptbots_core::WorldState;
use std::sync::{Arc, Mutex};
use tracing::info;

/// Root view displaying high level simulation stats.
struct WorldView {
    world: Arc<Mutex<WorldState>>,
}

impl Render for WorldView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let summary_text = self
            .world
            .lock()
            .ok()
            .and_then(|world| world.history().last().cloned())
            .map(|s| {
                format!(
                    "Tick {} | Agents {} | Births {} | Deaths {} | Avg Energy {:.2} | Avg Health {:.2}",
                    s.tick.0,
                    s.agent_count,
                    s.births,
                    s.deaths,
                    s.average_energy,
                    s.average_health,
                )
            })
            .unwrap_or_else(|| "No simulation data yet.".to_string());

        div().child(div().child(summary_text)).child(
            div().child("Rendering milestones will add live agent visuals and controls here."),
        )
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

    Application::new().run(move |app: &mut App| {
        app.open_window(WindowOptions::default(), move |_, cx| {
            cx.new(|_| WorldView {
                world: world.clone(),
            })
        })
        .expect("failed to open GPUI window");
    });
}
