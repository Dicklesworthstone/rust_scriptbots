//! GPUI rendering layer for ScriptBots.

use gpui::{App, Application};
use scriptbots_core::WorldState;
use std::sync::{Arc, Mutex};
use tracing::info;

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

    Application::new().run(|_cx: &mut App| {
        // Rendering logic will be added in later milestones.
    });
}
