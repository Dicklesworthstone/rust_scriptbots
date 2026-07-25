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

use crate::{
    AnalyticsSnapshotProvider, ControlCommand, GuiSession, GuiViewRole, HUD_RAIL_WIDTH,
    WorldStepDriver,
};

// GPUI's test window reports a fixed 2× device scale. `HeadlessAppContext::open_window`
// accepts logical pixels, while `capture_screenshot` returns device pixels, so divide
// here to keep the capture API and evidence filenames expressed in physical pixels.
const HEADLESS_DEVICE_SCALE: f32 = 2.0;

/// Render one GPUI view offscreen at exact output dimensions in device pixels.
pub(crate) fn capture_view(
    world: Arc<Mutex<WorldState>>,
    role: GuiViewRole,
    width: f32,
    height: f32,
) -> Result<RgbaImage, String> {
    capture_view_with_world_painter(world, role, width, height, false)
}

fn capture_view_with_world_painter(
    world: Arc<Mutex<WorldState>>,
    role: GuiViewRole,
    width: f32,
    height: f32,
    force_legacy_world_painter: bool,
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
        .open_window(
            size(
                px(width / HEADLESS_DEVICE_SCALE),
                px(height / HEADLESS_DEVICE_SCALE),
            ),
            move |window, app| {
                app.new(|cx| {
                    let focus_handle = cx.focus_handle();
                    focus_handle.focus(window, cx);
                    let mut view = session.new_view(role, focus_handle);
                    view.force_legacy_world_painter = force_legacy_world_painter;
                    view
                })
            },
        )
        .map_err(|error| format!("open headless GPUI window: {error:?}"))?;

    // Let the view settle, then FORCE A REDRAW before reading back.
    //
    // `capture_screenshot` returns `self.rendered_frame.scene` — the last frame the
    // window actually composed. `run_until_parked` alone drains the task queue but does
    // not guarantee a full paint, and with `show: false` nothing else drives one. The
    // first working run of this harness proved it: the readback was a real, correctly
    // sized PNG that was almost entirely flat background — no rail, no world canvas,
    // only a thin strip of footer text. An empty scene reads exactly like a broken
    // renderer, so this must be explicit rather than incidental.
    cx.run_until_parked();
    let _ = cx.update_window(window.into(), |_, window, _| window.refresh());
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

    /// Border drawn by every docked panel (`render_overlay`, `render_perf_overlay`).
    /// Unlike the panel fills, this is not shared with the root container background.
    const PANEL_BORDER: [u8; 3] = [0x1e, 0x29, 0x3b];

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

    #[cfg(target_os = "macos")]
    fn capture_visual_world() -> Arc<Mutex<WorldState>> {
        const POPULATION: usize = 96;

        // Production initial-population framing shows 120 agent diameters across.
        // With the default 10-unit radius, a 2,400-unit-wide world therefore fills
        // the useful canvas instead of occupying the 288×288-pixel postage stamp
        // produced by the old 600×600 fixture. The 2:1 aspect also matches the
        // production world's shape.
        let config = ScriptBotsConfig {
            world_width: 2_400,
            world_height: 1_200,
            food_cell_size: 50,
            population_minimum: POPULATION,
            population_spawn_interval: 0,
            persistence_interval: 0,
            rng_seed: Some(0xBD11_5EED),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("offscreen visual capture world");
        world
            .step()
            .expect("seed offscreen capture world with visible agents");
        assert_eq!(
            world.agent_count(),
            POPULATION,
            "visual proof fixture must contain the advertised population"
        );

        // `initial_food` defaults to zero and tick 1 precedes the first scheduled
        // respawn, so relying on defaults makes a food-visibility proof vacuous.
        // Install sparse, deterministic hotspots before taking the read-only digest.
        let food_width = world.food().width() as usize;
        world
            .try_update_food(|cells| {
                for (index, value) in cells.iter_mut().enumerate() {
                    let x = index % food_width;
                    let y = index / food_width;
                    let signature = x.wrapping_mul(17).wrapping_add(y.wrapping_mul(31));
                    *value = if signature.is_multiple_of(29) {
                        0.5
                    } else if signature.is_multiple_of(13) {
                        0.25
                    } else {
                        0.0
                    };
                }
            })
            .expect("seed deterministic food motes for visual proof");
        assert!(
            world
                .food()
                .cells()
                .iter()
                .filter(|value| **value > 0.0)
                .count()
                >= 32,
            "visual proof fixture must contain visible food hotspots"
        );

        Arc::new(Mutex::new(world))
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

        for (logical_w, logical_h) in VIEWPORTS {
            // VIEWPORTS are the LOGICAL sizes the production windows open at, and the
            // layout must be exercised at those. `capture_view` takes DEVICE pixels and
            // divides by HEADLESS_DEVICE_SCALE, so ask for the scaled size or the app
            // sees a half-size window.
            //
            // This is not a detail. Requesting 1280 device px opens a 640-logical window,
            // and HUD_RAIL_COLLAPSE_WIDTH is 960 (WORLD_MIN_WIDTH 640 + HUD_RAIL_WIDTH
            // 320), so the resize rule correctly force-collapses the rail and the capture
            // shows a HUD with no chrome at all. The first real run of this test failed
            // exactly that way.
            let width = logical_w * HEADLESS_DEVICE_SCALE;
            let height = logical_h * HEADLESS_DEVICE_SCALE;
            let image = capture_view(capture_world(), GuiViewRole::Hud, width, height)
                .unwrap_or_else(|error| {
                    panic!(
                        "headless HUD capture failed at {logical_w}x{logical_h} logical \
                         ({width}x{height} device): {error:#}"
                    )
                });

            let w = image.width();
            let h = image.height();
            assert_eq!(
                (w, h),
                (width as u32, height as u32),
                "capture must match the requested viewport"
            );

            // Marker is the PANEL BORDER, not the history-panel background.
            //
            // The history background was the original marker and it was the wrong
            // choice: it encoded an assumption — that stats AND history are both
            // visible by default — which the first real captures disproved. At 720
            // logical the HUD's canvas row is only ~207 logical px tall, so the rail
            // fits one panel and the history chart legitimately does not appear. A
            // marker that can never be satisfied does not test the policy, it just
            // fails. The border is what every docked panel actually draws.
            let centre_lo = w / 3;
            let centre_hi = w - w / 3;
            let centre_hits = count_color(&image, PANEL_BORDER, centre_lo, centre_hi);

            // Right band: the rightmost rail-width strip, where the docked rail lives.
            let rail_band = (HUD_RAIL_WIDTH * HEADLESS_DEVICE_SCALE) as u32;
            let right_hits = count_color(&image, PANEL_BORDER, w.saturating_sub(rail_band), w);

            // Named by LOGICAL viewport so the evidence matches the window size a user
            // actually has; the file itself is a 2x HiDPI buffer, as a Retina screenshot
            // of that window would be.
            let path = probe_dir().join(format!(
                "hud_docked_{}x{}.png",
                logical_w as u32, logical_h as u32
            ));
            image.save(&path).expect("write HUD probe png");

            assert!(
                right_hits > 0,
                "no docked-panel border found in the rightmost rail band at {w}x{h}; the rail \
                 did not render, so the centre assertion below would be vacuous. Probe: {}",
                path.display()
            );
            assert_eq!(
                centre_hits,
                0,
                "HUD chrome is covering the world centre at {w}x{h}: {centre_hits} \
                 docked-panel border pixels between x={centre_lo} and x={centre_hi}. bd-v9cz's \
                 one non-negotiable is that nothing sits over the world centre by \
                 default. Probe: {}",
                path.display()
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn continuous_world_painter_changes_real_gpui_pixels_without_changing_science() {
        let probe_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/rendering_reference/live_probes/bd-1lls");
        std::fs::create_dir_all(&probe_dir).expect("bd-1lls probe output directory");

        let world = capture_visual_world();
        let digest_before = world
            .lock()
            .expect("capture world lock")
            .world_digest_v1()
            .expect("pre-capture world digest");
        let legacy = capture_view_with_world_painter(
            Arc::clone(&world),
            GuiViewRole::WorldCanvas,
            1280.0,
            720.0,
            true,
        )
        .unwrap_or_else(|error| panic!("headless legacy world capture failed: {error:#}"));
        let digest_after_legacy = world
            .lock()
            .expect("capture world lock")
            .world_digest_v1()
            .expect("post-legacy-capture world digest");
        let continuous = capture_view_with_world_painter(
            Arc::clone(&world),
            GuiViewRole::WorldCanvas,
            1280.0,
            720.0,
            false,
        )
        .unwrap_or_else(|error| panic!("headless continuous world capture failed: {error:#}"));
        let digest_after_continuous = world
            .lock()
            .expect("capture world lock")
            .world_digest_v1()
            .expect("post-continuous-capture world digest");

        assert_eq!(legacy.dimensions(), (1280, 720));
        assert_eq!(continuous.dimensions(), (1280, 720));
        assert_eq!(
            digest_before, digest_after_legacy,
            "legacy repaint mutated scientific state"
        );
        assert_eq!(
            digest_before, digest_after_continuous,
            "continuous repaint mutated scientific state"
        );

        let mut differing_pixels = 0usize;
        let mut min_changed_x = legacy.width();
        let mut min_changed_y = legacy.height();
        let mut max_changed_x = 0u32;
        let mut max_changed_y = 0u32;
        for (index, (left, right)) in legacy.pixels().zip(continuous.pixels()).enumerate() {
            if left == right {
                continue;
            }
            differing_pixels += 1;
            let x = index as u32 % legacy.width();
            let y = index as u32 / legacy.width();
            min_changed_x = min_changed_x.min(x);
            min_changed_y = min_changed_y.min(y);
            max_changed_x = max_changed_x.max(x);
            max_changed_y = max_changed_y.max(y);
        }
        let minimum_changed = (legacy.width() as usize * legacy.height() as usize) / 100;
        assert!(
            differing_pixels > minimum_changed,
            "production GPUI capture did not exercise the continuous world painter: only \
             {differing_pixels} pixels changed (minimum {minimum_changed})"
        );
        let changed_width = max_changed_x.saturating_sub(min_changed_x) + 1;
        let changed_height = max_changed_y.saturating_sub(min_changed_y) + 1;
        assert!(
            changed_width >= legacy.width() * 3 / 4 && changed_height >= legacy.height() * 2 / 3,
            "world proof is a postage stamp rather than a human-legible frame: changed \
             bounds are {changed_width}x{changed_height} in {}x{}",
            legacy.width(),
            legacy.height()
        );

        legacy
            .save(probe_dir.join("before_per_cell_quads_1280x720.png"))
            .expect("write bd-1lls baseline probe");
        continuous
            .save(probe_dir.join("after_continuous_fields_1280x720.png"))
            .expect("write bd-1lls continuous-field probe");
    }
}
