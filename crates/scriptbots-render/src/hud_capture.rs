//! Headless GPUI HUD capture (bd-abu3).
//!
//! The scene harness in `scriptbots-app` can capture the Bevy world offscreen, and
//! `--dump-png` can rasterize the world through `render_png_offscreen`. Neither draws
//! HUD chrome: `render_png_offscreen` takes `&WorldState` and builds its own camera, so
//! it never constructs a [`SimulationView`] and can never contain a panel. That left
//! every claim about HUD layout (bd-v9cz, and everything bd-f4x0 will do) resting on
//! code inspection.
//!
//! This module closes that gap by rendering the REAL element tree — the same
//! [`SimulationView`] the production windows use, via the same [`GuiSession`] — through
//! GPUI's headless Metal renderer, and reading the frame back as RGBA. No window server
//! and no display are involved, so it runs in CI and under an agent session.
//!
//! It lives inside the crate rather than in `tests/` because [`GuiSession`] and
//! [`SimulationView`] are private to the crate root; a child module can reach them, an
//! integration test cannot. It is `#[cfg(test)]` because GPUI's `HeadlessAppContext` is
//! gated behind that crate's `test-support` feature, which is a dev-dependency here and
//! must not leak into production builds.

use std::sync::{Arc, Mutex};

use gpui::{AppContext as _, HeadlessAppContext, px, size};
use image::RgbaImage;
use scriptbots_core::{ScriptBotsConfig, WorldState};

use crate::{AnalyticsSnapshotProvider, ControlCommand, GuiSession, GuiViewRole, WorldStepDriver};

/// Render one GPUI view offscreen at an exact logical size and read the frame back.
pub(crate) fn capture_view(
    world: Arc<Mutex<WorldState>>,
    role: GuiViewRole,
    width: f32,
    height: f32,
) -> Result<RgbaImage, String> {
    // `headless = true`: no window server is contacted. The text system still comes from
    // the real platform, so glyph rasterization matches what a user sees.
    let platform = gpui_platform::current_platform(true);
    let text_system = platform.text_system();

    let mut cx = HeadlessAppContext::with_platform(
        text_system,
        Arc::new(()),
        gpui_platform::current_headless_renderer,
    );

    let step_world = Arc::clone(&world);
    let simulation_step: WorldStepDriver = Arc::new(move || {
        step_world
            .lock()
            .expect("world mutex poisoned during offscreen capture")
            .step()
    });
    let command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> = Arc::new(Vec::new);
    let command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync> = Arc::new(|_| true);

    let session = Arc::new(GuiSession::new(
        world,
        simulation_step,
        AnalyticsSnapshotProvider::empty(),
        command_drain,
        command_submit,
    ));

    let window = cx
        .open_window(size(px(width), px(height)), move |window, app| {
            app.new(|cx| {
                let focus_handle = cx.focus_handle();
                focus_handle.focus(window, cx);
                session.new_view(role, focus_handle)
            })
        })
        .map_err(|error| format!("open headless GPUI window: {error:?}"))?;

    // Let the view settle so the first frame is a real composed frame rather than a
    // partially initialized one.
    cx.run_until_parked();

    cx.capture_screenshot(window.into())
        .map_err(|error| format!("read back headless GPUI frame: {error:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Panel background of the history card (`render_history_chart`). Deliberately NOT
    /// the stats card's `0x0b1120`, which is also the root container background and so
    /// cannot distinguish chrome from backdrop.
    const HISTORY_PANEL_BG: [u8; 3] = [0x0a, 0x16, 0x29];

    /// The two viewports the production windows actually open at
    /// (`GuiViewRole::window_options`).
    const VIEWPORTS: [(f32, f32); 2] = [(1280.0, 720.0), (1600.0, 900.0)];

    fn capture_world() -> Arc<Mutex<WorldState>> {
        let config = ScriptBotsConfig {
            world_width: 600,
            world_height: 600,
            food_cell_size: 50,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        };
        Arc::new(Mutex::new(
            WorldState::new(config).expect("offscreen capture world"),
        ))
    }

    fn probe_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/rendering_reference/live_probes/bd-v9cz")
    }

    fn count_color(image: &RgbaImage, rgb: [u8; 3], x0: u32, x1: u32) -> u32 {
        let mut hits = 0;
        for y in 0..image.height() {
            for x in x0..x1.min(image.width()) {
                let p = image.get_pixel(x, y).0;
                if p[0] == rgb[0] && p[1] == rgb[1] && p[2] == rgb[2] {
                    hits += 1;
                }
            }
        }
        hits
    }

    /// Captures the docked HUD at both production viewports and asserts the owner's
    /// non-negotiable from bd-v9cz on actual pixels: chrome lives in the rail on the
    /// right, and NOTHING sits over the world centre by default.
    ///
    /// The pair of assertions is what makes this non-vacuous. Requiring the panel colour
    /// to be absent from the centre would pass trivially if the panel failed to render at
    /// all, so the same colour must also be PRESENT in the right-hand band.
    #[test]
    fn docked_hud_never_covers_the_world_centre_at_either_viewport() {
        std::fs::create_dir_all(probe_dir()).expect("probe output directory");

        for (width, height) in VIEWPORTS {
            let image = capture_view(capture_world(), GuiViewRole::Hud, width, height)
                .unwrap_or_else(|error| {
                    panic!("headless HUD capture failed at {width}x{height}: {error:#}")
                });

            let w = image.width();
            let h = image.height();
            assert_eq!(
                (w, h),
                (width as u32, height as u32),
                "capture must match the requested viewport"
            );

            // Centre band: the middle third horizontally. This is "the world centre" in
            // the owner's complaint, and the region that must stay clear of chrome.
            let centre_lo = w / 3;
            let centre_hi = w - w / 3;
            let centre_hits = count_color(&image, HISTORY_PANEL_BG, centre_lo, centre_hi);

            // Right band: where the docked rail lives.
            let right_hits = count_color(&image, HISTORY_PANEL_BG, centre_hi, w);

            let path = probe_dir().join(format!("hud_docked_{}x{}.png", w, h));
            image.save(&path).expect("write HUD probe png");

            assert!(
                right_hits > 0,
                "no history-panel pixels found in the right band at {w}x{h}; the rail did \
                 not render, so the centre assertion below would be vacuous. Probe: {}",
                path.display()
            );
            assert_eq!(
                centre_hits,
                0,
                "HUD chrome is covering the world centre at {w}x{h}: {centre_hits} \
                 history-panel pixels between x={centre_lo} and x={centre_hi}. bd-v9cz's \
                 one non-negotiable is that nothing sits over the world centre by \
                 default. Probe: {}",
                path.display()
            );
        }
    }
}
