//! GPUI rendering layer for ScriptBots.

use gpui::{App, Application};

/// Bootstraps a minimal GPUI application.
pub fn run_demo() {
    Application::new().run(|_cx: &mut App| {
        // Rendering logic will be added in later milestones.
    });
}
