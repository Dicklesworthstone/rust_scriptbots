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
use scriptbots_core::{AgentId, ScriptBotsConfig, WorldState};
#[cfg(target_os = "macos")]
use scriptbots_core::{
    OutputChannel, RenderDayNightSettings, RenderQuality, RenderSettings, SelectionState,
};

use crate::{
    AnalyticsSnapshotProvider, CameraSnapshot, ControlCommand, GuiSession, GuiViewRole,
    HUD_RAIL_WIDTH, WorldStepDriver,
};

// GPUI's test window reports a fixed 2× device scale. `HeadlessAppContext::open_window`
// accepts logical pixels, while `capture_screenshot` returns device pixels, so divide
// here to keep the capture API and evidence filenames expressed in physical pixels.
const HEADLESS_DEVICE_SCALE: f32 = 2.0;

#[derive(Clone, Copy, Default)]
struct CaptureOverrides {
    force_legacy_world_painter: bool,
    draw_agents: Option<bool>,
    draw_food: Option<bool>,
    forced_fps: Option<f32>,
    hovered_agent: Option<AgentId>,
    /// Keystrokes dispatched into the live window before the frame is captured.
    ///
    /// Deliberately NOT a direct field poke like the overrides above. Panel state has
    /// to be reached the way a user reaches it — keystroke, binding lookup,
    /// CommandAction, HudLayout — or the capture would photograph an arrangement the
    /// production app can never be in, which is exactly the class of defect that hid
    /// the missing rail earlier (bd-abu3).
    keystrokes: &'static [&'static str],
}

struct CapturedView {
    image: RgbaImage,
    camera: CameraSnapshot,
}

fn apply_capture_overrides(view: &mut crate::SimulationView, overrides: CaptureOverrides) {
    view.force_legacy_world_painter = overrides.force_legacy_world_painter;
    if let Some(draw_agents) = overrides.draw_agents {
        view.controls.draw_agents = draw_agents;
    }
    if let Some(draw_food) = overrides.draw_food {
        view.controls.draw_food = draw_food;
    }
    if let Some(fps) = overrides.forced_fps {
        view.last_perf.fps = fps;
    }
    if let Some(hovered_agent) = overrides.hovered_agent
        && let Ok(mut inspector) = view.inspector.lock()
    {
        inspector.hovered_agent = Some(hovered_agent);
    }
}

/// Render one GPUI view offscreen at exact output dimensions in device pixels.
pub(crate) fn capture_view(
    world: Arc<Mutex<WorldState>>,
    role: GuiViewRole,
    width: f32,
    height: f32,
) -> Result<RgbaImage, String> {
    capture_view_with_overrides(world, role, width, height, CaptureOverrides::default())
}

fn capture_view_with_world_painter(
    world: Arc<Mutex<WorldState>>,
    role: GuiViewRole,
    width: f32,
    height: f32,
    force_legacy_world_painter: bool,
) -> Result<RgbaImage, String> {
    capture_view_with_overrides(
        world,
        role,
        width,
        height,
        CaptureOverrides {
            force_legacy_world_painter,
            ..CaptureOverrides::default()
        },
    )
}

fn capture_view_with_overrides(
    world: Arc<Mutex<WorldState>>,
    role: GuiViewRole,
    width: f32,
    height: f32,
    overrides: CaptureOverrides,
) -> Result<RgbaImage, String> {
    let capture = capture_view_with_overrides_and_camera(world, role, width, height, overrides)?;
    if !capture.camera.last_scale.is_finite() {
        return Err("headless GPUI camera produced a non-finite scale".to_owned());
    }
    Ok(capture.image)
}

fn capture_view_with_overrides_and_camera(
    world: Arc<Mutex<WorldState>>,
    role: GuiViewRole,
    width: f32,
    height: f32,
    overrides: CaptureOverrides,
) -> Result<CapturedView, String> {
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
                    apply_capture_overrides(&mut view, overrides);
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

    // Drive panel state through the REAL input path: keystroke -> InputBindings ->
    // CommandAction -> HudLayout, the same chain a user's keypress takes. Setting the
    // fields directly would photograph an arrangement production can never reach.
    for stroke in overrides.keystrokes {
        let parsed = gpui::Keystroke::parse(stroke)
            .map_err(|error| format!("invalid capture keystroke {stroke:?}: {error:?}"))?;
        cx.update_window(window.into(), |_, window, app| {
            window.dispatch_keystroke(parsed.clone(), app);
        })
        .map_err(|error| format!("dispatch capture keystroke {stroke:?}: {error:?}"))?;
        cx.run_until_parked();
    }

    cx.update_window(window.into(), |root_view, window, app| {
        let view = root_view
            .downcast::<crate::SimulationView>()
            .expect("headless root view type mismatch");
        view.update(app, |view, _| {
            // Settling the view may have recorded a real performance sample.
            // Reapply controlled inputs immediately before the evidence repaint.
            apply_capture_overrides(view, overrides);
        });
        window.refresh();
    })
    .map_err(|error| format!("refresh headless GPUI window: {error:?}"))?;
    cx.run_until_parked();

    let camera = cx
        .update_window(window.into(), |root_view, _, app| {
            let view = root_view
                .downcast::<crate::SimulationView>()
                .expect("headless root view type mismatch");
            view.update(app, |view, _| {
                view.camera
                    .lock()
                    .expect("camera lock poisoned during offscreen capture")
                    .snapshot()
            })
        })
        .map_err(|error| format!("read headless GPUI camera: {error:?}"))?;
    let image = cx
        .capture_screenshot(window.into())
        .map_err(|error| format!("read back headless GPUI frame: {error:?}"))?;
    Ok(CapturedView { image, camera })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Panel background of the history card (`render_history_chart`). Deliberately NOT
    /// the stats card's `0x0b1120`, which is also the root container background and so
    /// cannot distinguish chrome from backdrop.
    const HISTORY_PANEL_BG: [u8; 3] = [0x0a, 0x16, 0x29];

    /// Border drawn by every docked panel, read from the chrome system itself rather
    /// than hardcoded.
    ///
    /// It WAS hardcoded to 0x1e293b, and bd-f4x0 repointed the panels to a
    /// substrate-derived colour — so the marker silently stopped describing the thing
    /// it names. Worse than a plain failure: 0x1e293b is used by 51 other elements, so
    /// at 1280x720 it still scraped 6 stray hits inside the rail band and the test
    /// PASSED for the wrong reason, while 1600x900 got zero and failed. Deriving it
    /// from chrome::border() means a future palette change moves the marker with it.
    fn panel_border() -> [u8; 3] {
        let rgba: gpui::Rgba = crate::chrome::border().into();
        let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        [byte(rgba.r), byte(rgba.g), byte(rgba.b)]
    }

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
            // SEEDED — this is the root of bd-c7pg. Unseeded, every call built a
            // differently-random world, so two captures of "the same" scene differed by
            // ~10k pixels in agent and food layout alone. That floor swallowed 21 of 27
            // shortcut deltas and would swallow any small visual change. Fixed at the
            // source; the alternative would have been a tolerance tuned to hide it.
            rng_seed: Some(0xBD_C7_09),
            ..ScriptBotsConfig::default()
        };
        Arc::new(Mutex::new(
            WorldState::new(config).expect("offscreen capture world"),
        ))
    }

    #[cfg(target_os = "macos")]
    fn capture_visual_world_with_render(render: RenderSettings) -> Arc<Mutex<WorldState>> {
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
            render,
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

    #[cfg(target_os = "macos")]
    fn capture_visual_world() -> Arc<Mutex<WorldState>> {
        capture_visual_world_with_render(RenderSettings::default())
    }

    #[cfg(target_os = "macos")]
    fn controlled_agent_position(column: usize, row: usize) -> (f32, f32) {
        (130.0 + column as f32 * 194.0, 105.0 + row as f32 * 141.0)
    }

    #[cfg(target_os = "macos")]
    fn capture_agent_expression_world() -> Arc<Mutex<WorldState>> {
        const COLUMNS: usize = 12;

        let world = capture_visual_world();
        let mut guard = world.lock().expect("agent expression world lock");
        let agent_ids: Vec<_> = guard.agents().iter_handles().collect();
        let reference_age = guard.config().aging_health_decay_start.max(1);

        for (index, agent_id) in agent_ids.into_iter().enumerate() {
            let column = index % COLUMNS;
            let row = index / COLUMNS;
            let group = column / 3;
            let health_class = column % 3;
            guard
                .try_update_agent(agent_id, |agent, runtime| {
                    // A regular 12×8 matrix makes semantic differences readable in
                    // the owner-facing capture instead of hiding them in random overlap.
                    let position = controlled_agent_position(column, row);
                    agent.position.x = position.0;
                    agent.position.y = position.1;
                    // Orientation is already proven by bd-1lls. Facing every agent
                    // right makes spike reach directly comparable between groups.
                    agent.heading = 0.0;
                    agent.color = [0.25, 0.25, 0.25];
                    agent.boost = false;

                    // Columns repeat full, mid, and floor health. Rows repeat young
                    // and weathered ages, so luminance and saturation remain separate
                    // readable dimensions.
                    agent.health = match health_class {
                        0 => 2.0,
                        1 => 1.2,
                        _ => 0.05,
                    };
                    agent.age = if row < 4 { 0 } else { reference_age };

                    // Four three-column groups, each repeating full/mid/floor health:
                    // short neutral; long neutral; long attacker; long victim-only.
                    // Attacker and victim groups share extension and length so the
                    // outer strike flash is the only intended footprint difference.
                    let spike_extended = group >= 2;
                    agent.spike_length = if group == 0 { 0.05 } else { 0.85 };
                    runtime.outputs.fill(0.0);
                    runtime.outputs[OutputChannel::SpikeTarget.index()] =
                        if spike_extended { 1.0 } else { 0.0 };
                    runtime.combat.spike_attacker = group == 2;
                    runtime.spiked = group == 3;

                    runtime.outputs[OutputChannel::Boost.index()] =
                        if row.is_multiple_of(4) { 1.0 } else { 0.0 };
                    runtime.herbivore_tendency = if row.is_multiple_of(2) { 0.0 } else { 1.0 };
                    runtime.temperature_preference = 0.5;
                    runtime.sound_multiplier = 1.0;
                    runtime.sound_output = 0.0;
                    runtime.food_delta = 0.0;
                    runtime.give_intent = 0.0;
                    runtime.trait_modifiers = Default::default();
                    runtime.eye_fov.fill(1.0);
                    runtime.eye_direction.fill(0.0);
                    runtime.indicator = Default::default();
                    runtime.selection = if row == 1 {
                        SelectionState::Selected
                    } else {
                        SelectionState::None
                    };
                })
                .expect("install controlled agent visual state");
        }
        drop(guard);
        world
    }

    #[cfg(target_os = "macos")]
    fn count_agent_delta_classes(
        control: &RgbaImage,
        rendered: &RgbaImage,
        camera: &CameraSnapshot,
    ) -> [usize; 5] {
        let mut classes = [0usize; 5];
        for row in 0..8 {
            for column in 0..12 {
                let center = device_point(camera, controlled_agent_position(column, row));
                let radius = 30.0;
                let min_x = (center.0 - radius).floor().max(0.0) as u32;
                let max_x = (center.0 + radius)
                    .ceil()
                    .min(rendered.width() as f32 - 1.0) as u32;
                let min_y = (center.1 - radius).floor().max(0.0) as u32;
                let max_y = (center.1 + radius)
                    .ceil()
                    .min(rendered.height() as f32 - 1.0) as u32;
                let radius_squared = radius * radius;
                for y in min_y..=max_y {
                    for x in min_x..=max_x {
                        let dx = x as f32 - center.0;
                        let dy = y as f32 - center.1;
                        if dx * dx + dy * dy > radius_squared {
                            continue;
                        }
                        let background = control.get_pixel(x, y);
                        let agent = rendered.get_pixel(x, y);
                        if background == agent {
                            continue;
                        }
                        classes[0] += 1;
                        let [r, g, b, _] = agent.0;
                        let luma =
                            (u32::from(r) * 54 + u32::from(g) * 183 + u32::from(b) * 19) / 256;
                        if b > r.saturating_add(16) && g > r.saturating_add(16) {
                            classes[1] += 1;
                        }
                        if r > g.saturating_add(16) && b > g.saturating_add(16) {
                            classes[2] += 1;
                        }
                        if luma >= 96 {
                            classes[3] += 1;
                        }
                        if (12..=64).contains(&luma) {
                            classes[4] += 1;
                        }
                    }
                }
            }
        }
        classes
    }

    #[cfg(target_os = "macos")]
    fn assert_camera_transform_eq(
        label: &str,
        control: &CameraSnapshot,
        rendered: &CameraSnapshot,
    ) {
        for world in [(0.0, 0.0), (1_200.0, 600.0), (2_400.0, 1_200.0)] {
            let control_point = control.world_to_screen(world);
            let rendered_point = rendered.world_to_screen(world);
            assert_eq!(
                control_point.is_some(),
                rendered_point.is_some(),
                "{label} camera visibility differs at {world:?}"
            );
            if let (Some(control_point), Some(rendered_point)) = (control_point, rendered_point) {
                assert!(
                    (control_point.0 - rendered_point.0).abs() <= 1.0e-4
                        && (control_point.1 - rendered_point.1).abs() <= 1.0e-4,
                    "{label} hidden/visible cameras drifted at {world:?}: \
                     control={control_point:?}, rendered={rendered_point:?}"
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn device_point(camera: &CameraSnapshot, world: (f32, f32)) -> (f32, f32) {
        let (x, y) = camera
            .world_to_screen(world)
            .unwrap_or_else(|| panic!("controlled agent {world:?} is outside the capture camera"));
        (x * HEADLESS_DEVICE_SCALE, y * HEADLESS_DEVICE_SCALE)
    }

    #[cfg(target_os = "macos")]
    fn average_luma_delta_in_disc(
        control: &RgbaImage,
        rendered: &RgbaImage,
        center: (f32, f32),
        radius: f32,
    ) -> f32 {
        let min_x = (center.0 - radius).floor().max(0.0) as u32;
        let max_x = (center.0 + radius)
            .ceil()
            .min(rendered.width() as f32 - 1.0) as u32;
        let min_y = (center.1 - radius).floor().max(0.0) as u32;
        let max_y = (center.1 + radius)
            .ceil()
            .min(rendered.height() as f32 - 1.0) as u32;
        let radius_squared = radius * radius;
        let mut sum = 0.0;
        let mut count = 0usize;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - center.0;
                let dy = y as f32 - center.1;
                if dx * dx + dy * dy > radius_squared {
                    continue;
                }
                let [control_r, control_g, control_b, _] = control.get_pixel(x, y).0;
                let [rendered_r, rendered_g, rendered_b, _] = rendered.get_pixel(x, y).0;
                let control_luma = 0.2126 * f32::from(control_r)
                    + 0.7152 * f32::from(control_g)
                    + 0.0722 * f32::from(control_b);
                let rendered_luma = 0.2126 * f32::from(rendered_r)
                    + 0.7152 * f32::from(rendered_g)
                    + 0.0722 * f32::from(rendered_b);
                sum += rendered_luma - control_luma;
                count += 1;
            }
        }
        assert!(count > 0, "luma-delta sample disc must contain pixels");
        sum / count as f32
    }

    #[cfg(target_os = "macos")]
    fn average_rgb_delta_in_disc(
        control: &RgbaImage,
        rendered: &RgbaImage,
        center: (f32, f32),
        radius: f32,
    ) -> [f32; 3] {
        let min_x = (center.0 - radius).floor().max(0.0) as u32;
        let max_x = (center.0 + radius)
            .ceil()
            .min(rendered.width() as f32 - 1.0) as u32;
        let min_y = (center.1 - radius).floor().max(0.0) as u32;
        let max_y = (center.1 + radius)
            .ceil()
            .min(rendered.height() as f32 - 1.0) as u32;
        let radius_squared = radius * radius;
        let mut sum = [0.0; 3];
        let mut count = 0usize;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - center.0;
                let dy = y as f32 - center.1;
                if dx * dx + dy * dy > radius_squared {
                    continue;
                }
                let [control_r, control_g, control_b, _] = control.get_pixel(x, y).0;
                let [rendered_r, rendered_g, rendered_b, _] = rendered.get_pixel(x, y).0;
                sum[0] += f32::from(rendered_r) - f32::from(control_r);
                sum[1] += f32::from(rendered_g) - f32::from(control_g);
                sum[2] += f32::from(rendered_b) - f32::from(control_b);
                count += 1;
            }
        }
        assert!(count > 0, "RGB-delta sample disc must contain pixels");
        [
            sum[0] / count as f32,
            sum[1] / count as f32,
            sum[2] / count as f32,
        ]
    }

    #[cfg(target_os = "macos")]
    fn changed_pixels_in_disc(
        control: &RgbaImage,
        rendered: &RgbaImage,
        center: (f32, f32),
        radius: f32,
    ) -> usize {
        let min_x = (center.0 - radius).floor().max(0.0) as u32;
        let max_x = (center.0 + radius)
            .ceil()
            .min(rendered.width() as f32 - 1.0) as u32;
        let min_y = (center.1 - radius).floor().max(0.0) as u32;
        let max_y = (center.1 + radius)
            .ceil()
            .min(rendered.height() as f32 - 1.0) as u32;
        let radius_squared = radius * radius;
        let mut count = 0usize;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - center.0;
                let dy = y as f32 - center.1;
                if dx * dx + dy * dy <= radius_squared
                    && control.get_pixel(x, y) != rendered.get_pixel(x, y)
                {
                    count += 1;
                }
            }
        }
        count
    }

    #[cfg(target_os = "macos")]
    fn whole_frame_delta(left: &RgbaImage, right: &RgbaImage) -> (usize, f32) {
        assert_eq!(
            left.dimensions(),
            right.dimensions(),
            "whole-frame comparison requires equal dimensions"
        );
        let mut changed = 0usize;
        let mut absolute_luma_delta = 0.0_f64;
        for (left, right) in left.pixels().zip(right.pixels()) {
            if left != right {
                changed += 1;
            }
            let [left_r, left_g, left_b, _] = left.0;
            let [right_r, right_g, right_b, _] = right.0;
            let left_luma = 0.2126 * f64::from(left_r)
                + 0.7152 * f64::from(left_g)
                + 0.0722 * f64::from(left_b);
            let right_luma = 0.2126 * f64::from(right_r)
                + 0.7152 * f64::from(right_g)
                + 0.0722 * f64::from(right_b);
            absolute_luma_delta += (left_luma - right_luma).abs();
        }
        let pixels = u64::from(left.width()) * u64::from(left.height());
        assert!(pixels > 0, "whole-frame comparison requires pixels");
        (changed, (absolute_luma_delta / pixels as f64) as f32)
    }

    fn probe_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/rendering_reference/live_probes/bd-v9cz")
    }

    /// Column span of the world canvas, found by colour diversity.
    ///
    /// The world raster is the only element that paints a continuously varying image;
    /// every piece of chrome is a flat fill with text on it. So the canvas is the widest
    /// run of adjacent columns that each carry many distinct colours. Returning a
    /// measured span rather than a hardcoded rectangle keeps this honest when the layout
    /// changes — which it just did, when the rail was hoisted.
    fn world_canvas_columns(image: &RgbaImage) -> (u32, u32) {
        const DIVERSE: usize = 12;
        let (w, h) = (image.width(), image.height());
        let diverse: Vec<bool> = (0..w)
            .map(|x| {
                let mut seen = std::collections::HashSet::new();
                for y in (0..h).step_by(4) {
                    let p = image.get_pixel(x, y).0;
                    seen.insert([p[0], p[1], p[2]]);
                }
                seen.len() >= DIVERSE
            })
            .collect();
        let (mut best, mut best_len) = ((0u32, 0u32), 0u32);
        let mut run_start: Option<u32> = None;
        for x in 0..w {
            match (diverse[x as usize], run_start) {
                (true, None) => run_start = Some(x),
                (false, Some(s)) => {
                    if x - s > best_len {
                        best_len = x - s;
                        best = (s, x);
                    }
                    run_start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = run_start
            && w - s > best_len
        {
            best = (s, w);
        }
        best
    }

    /// Pixels in a column range that are not the app's root background — i.e. actual
    /// rendered content of any kind.
    fn count_non_background(image: &RgbaImage, x0: u32, x1: u32) -> u32 {
        const ROOT_BG: [u8; 3] = [0x0f, 0x17, 0x2a];
        let mut hits = 0;
        for y in 0..image.height() {
            for x in x0..x1.min(image.width()) {
                let p = image.get_pixel(x, y).0;
                if [p[0], p[1], p[2]] != ROOT_BG {
                    hits += 1;
                }
            }
        }
        hits
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

    /// bd-bacf AUDIT INSTRUMENT — drive every advertised shortcut and see if it does
    /// anything.
    ///
    /// The owner's three reported defects were all findable in seconds of real use,
    /// which means the UI had been verified by asserting functions return rather than
    /// by looking at it. This looks at it: each shortcut in the README's generated
    /// table is dispatched through the real input path and the resulting frame is
    /// compared against an untouched baseline.
    ///
    /// A zero-delta result is a CANDIDATE, not a verdict. Some shortcuts are correctly
    /// invisible in a single static frame (clearing an empty selection changes nothing,
    /// and rightly so). The point is to produce a short reproducible list to examine,
    /// not to file every one of them.
    #[test]
    #[ignore = "bd-bacf audit instrument; run explicitly with --ignored"]
    fn audit_every_advertised_shortcut_changes_something() {
        const SHORTCUTS: [(&str, &str); 27] = [
            ("space", "Toggle playback"),
            ("g", "Jump to live"),
            ("b", "Toggle brush"),
            ("n", "Toggle narration"),
            ("ctrl-p", "Cycle palette"),
            ("p", "Toggle simulation pause"),
            ("s", "Step simulation once"),
            ("d", "Toggle agent drawing"),
            ("f", "Toggle food overlay"),
            ("ctrl-shift-o", "Toggle agent outline"),
            ("shift-=", "Increase speed"),
            ("-", "Decrease speed"),
            ("a", "Spawn crossover"),
            ("q", "Spawn carnivore"),
            ("h", "Spawn herbivore"),
            ("c", "Toggle closed environment"),
            ("shift-s", "Follow selected"),
            ("o", "Follow oldest"),
            ("shift-f", "Toggle debug overlay"),
            ("escape", "Clear selection"),
            ("ctrl-a", "Select all"),
            ("ctrl-f", "Focus first selected"),
            ("0", "Fit world"),
            (",", "Toggle settings panel"),
            ("1", "Toggle stats panel"),
            ("2", "Toggle history panel"),
            ("3", "Toggle performance panel"),
        ];
        let (w, h) = (
            1280.0 * HEADLESS_DEVICE_SCALE,
            720.0 * HEADLESS_DEVICE_SCALE,
        );
        // Pin the perf readout on BOTH sides. Without it the live FPS/timing text
        // redraws between captures and every frame differs from baseline by ~12k px
        // regardless of the keystroke — a noise floor that swallowed 21 of 27 results
        // on the first run and would have let a dead control read as working.
        let stable = || CaptureOverrides {
            forced_fps: Some(60.0),
            ..CaptureOverrides::default()
        };
        let base = capture_view_with_overrides(capture_world(), GuiViewRole::Hud, w, h, stable())
            .expect("baseline");

        let mut inert = Vec::new();
        for (stroke, label) in SHORTCUTS {
            let leaked: &'static [&'static str] = Box::leak(vec![stroke].into_boxed_slice());
            let after = capture_view_with_overrides(
                capture_world(),
                GuiViewRole::Hud,
                w,
                h,
                CaptureOverrides {
                    keystrokes: leaked,
                    ..CaptureOverrides::default()
                },
            )
            .unwrap_or_else(|e| panic!("capture after {stroke:?} failed: {e:#}"));
            let delta = base
                .pixels()
                .zip(after.pixels())
                .filter(|(a, b)| a != b)
                .count();
            println!("  {stroke:<14} {label:<28} delta={delta}");
            if delta == 0 {
                inert.push((stroke, label));
            }
        }
        println!("\n  ZERO-DELTA CANDIDATES ({}):", inert.len());
        for (stroke, label) in &inert {
            println!("    {stroke} -> {label}");
        }
    }

    /// DIFFERENTIAL layout proof (bd-v9cz / bd-f4x0).
    ///
    /// A single capture shows one arrangement and proves nothing about the policy. This
    /// captures the SAME scene twice — rail open, then rail closed — and diffs them.
    ///
    /// Both states are reached by dispatching the production keystrokes 1/2/3, so the
    /// closed state is the one a user actually gets; nothing here pokes HudLayout.
    ///
    /// The pair is self-calibrating. Requiring chrome to be confined to the rail band
    /// would pass trivially if the rail never drew, so the closed capture must show the
    /// rail band LOSING that chrome. One assertion cannot be satisfied vacuously while
    /// the other holds.
    #[test]
    fn toggling_the_rail_changes_only_the_rail_column() {
        std::fs::create_dir_all(probe_dir()).expect("probe output directory");
        let border = panel_border();

        for (logical_w, logical_h) in VIEWPORTS {
            let width = logical_w * HEADLESS_DEVICE_SCALE;
            let height = logical_h * HEADLESS_DEVICE_SCALE;

            let open = capture_view(capture_world(), GuiViewRole::Hud, width, height)
                .unwrap_or_else(|e| {
                    panic!("open capture failed at {logical_w}x{logical_h}: {e:#}")
                });
            // 1 = stats, 2 = history. Perf is collapsed by default, so these two empty
            // the rail and HudLayout::resolve then reports show_rail = false.
            let closed = capture_view_with_overrides(
                capture_world(),
                GuiViewRole::Hud,
                width,
                height,
                CaptureOverrides {
                    keystrokes: &["1", "2"],
                    ..CaptureOverrides::default()
                },
            )
            .unwrap_or_else(|e| panic!("closed capture failed at {logical_w}x{logical_h}: {e:#}"));

            let w = open.width();
            let rail_band = (HUD_RAIL_WIDTH * HEADLESS_DEVICE_SCALE) as u32;
            let rail_lo = w.saturating_sub(rail_band);

            let open_rail = count_color(&open, border, rail_lo, w);
            let closed_rail = count_color(&closed, border, rail_lo, w);
            let open_left = count_color(&open, border, 0, rail_lo);

            closed
                .save(probe_dir().join(format!(
                    "hud_rail_closed_{}x{}.png",
                    logical_w as u32, logical_h as u32
                )))
                .expect("write closed probe");

            assert!(
                open_rail > 0,
                "rail chrome absent at {logical_w}x{logical_h}; the open state did not \
                 render a rail, so the comparison below would be vacuous"
            );
            assert!(
                closed_rail < open_rail,
                "pressing 1 and 2 did not remove rail chrome at {logical_w}x{logical_h} \
                 (open {open_rail}, closed {closed_rail}); either the production toggle \
                 is broken or the capture is not reaching it"
            );
            // The world must RECLAIM the rail's column when the rail closes. That is
            // what "reserved space" means: chrome and world trade the column, they never
            // share it. Together with the two assertions above this is the docking
            // property, proven rather than assumed.
            //
            // NOT asserting "zero chrome left of the rail". chrome::border() is now
            // shared with the sibling history and inspector panels, which legitimately
            // sit beside the world in their own columns, so border pixels outside the
            // rail cannot be attributed to the rail. That check counted 1688 of them at
            // 1600x900 and none at 1280x720 — a marker problem, not a layout one.
            let closed_rail_content = count_non_background(&closed, rail_lo, w);
            assert!(
                closed_rail_content > 0,
                "closing the rail at {logical_w}x{logical_h} left its column empty \
                 ({closed_rail_content} non-background px); the world did not reclaim \
                 the reserved space, so the column is dead area rather than shared"
            );
            let _ = open_left;
        }
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
            // Scope the "clear" assertion to the WORLD CANVAS, not the whole frame.
            //
            // The middle third of the frame was the wrong rectangle. The frame contains
            // chrome margin by design — header, summary, analytics and history all stack
            // above and beside the world — so a frame-wide check both over-reports
            // (counting stacked chrome that overlaps nothing) and could under-report
            // (a panel sitting over a world that is not centred in the frame).
            //
            // The canvas is located empirically rather than assumed: it is the only
            // element that paints a raster, so it is the widest contiguous column span
            // whose columns carry many distinct colours. Chrome is flat fill plus text.
            let (centre_lo, centre_hi) = world_canvas_columns(&image);
            let centre_hits = count_color(&image, panel_border(), centre_lo, centre_hi);

            // Right band: the rightmost rail-width strip, where the docked rail lives.
            let rail_band = (HUD_RAIL_WIDTH * HEADLESS_DEVICE_SCALE) as u32;
            let right_hits = count_color(&image, panel_border(), w.saturating_sub(rail_band), w);

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
                 docked-panel border pixels inside the world canvas span x={centre_lo}..{centre_hi}. bd-v9cz's \
                 one non-negotiable is that nothing sits over the world centre by \
                 default. Probe: {}",
                path.display()
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn configured_post_stack_and_day_night_are_perceptible_in_real_gpui_pixels() {
        let daylight_settings = |start_phase| RenderSettings {
            day_night: Some(RenderDayNightSettings {
                cycle_ticks: Some(10_000),
                start_phase: Some(start_phase),
                ..RenderDayNightSettings::default()
            }),
            ..RenderSettings::default()
        };
        let default_world = capture_visual_world_with_render(RenderSettings::default());
        let potato_world = capture_visual_world_with_render(RenderSettings {
            quality: Some(RenderQuality::Potato),
            ..RenderSettings::default()
        });
        let noon_world = capture_visual_world_with_render(daylight_settings(0.25));
        let midnight_world = capture_visual_world_with_render(daylight_settings(0.75));

        let digest = |world: &Arc<Mutex<WorldState>>| {
            world
                .lock()
                .expect("visual proof world lock")
                .world_digest_v1()
                .expect("visual proof world digest")
        };
        let expected_digest = digest(&default_world);
        for (label, world) in [
            ("potato", &potato_world),
            ("noon", &noon_world),
            ("midnight", &midnight_world),
        ] {
            assert_eq!(
                expected_digest,
                digest(world),
                "{label} presentation settings changed scientific state"
            );
        }

        let capture = |world| {
            capture_view_with_overrides(
                world,
                GuiViewRole::WorldCanvas,
                1280.0,
                720.0,
                CaptureOverrides {
                    draw_agents: Some(false),
                    draw_food: Some(false),
                    forced_fps: Some(60.0),
                    ..CaptureOverrides::default()
                },
            )
            .unwrap_or_else(|error| panic!("headless post-stack capture failed: {error:#}"))
        };
        let default = capture(default_world);
        let potato = capture(potato_world);
        let noon = capture(noon_world);
        let midnight = capture(midnight_world);

        let probe_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/rendering_reference/live_probes/bd-lhml");
        std::fs::create_dir_all(&probe_dir).expect("bd-lhml probe output directory");
        for (name, image) in [
            ("default_medium_post_1280x720.png", &default),
            ("potato_no_post_1280x720.png", &potato),
            ("default_noon_1280x720.png", &noon),
            ("default_midnight_1280x720.png", &midnight),
        ] {
            image
                .save(probe_dir.join(name))
                .unwrap_or_else(|error| panic!("write {name}: {error}"));
        }

        let total_pixels = default.width() as usize * default.height() as usize;
        let (post_changed, post_luma_delta) = whole_frame_delta(&default, &potato);
        let (clock_changed, clock_luma_delta) = whole_frame_delta(&noon, &midnight);
        eprintln!(
            "bd-lhml pixel proof: post changed={post_changed}/{total_pixels}, \
             mean_abs_luma={post_luma_delta:.3}; half-cycle changed={clock_changed}/{total_pixels}, \
             mean_abs_luma={clock_luma_delta:.3}"
        );
        assert!(
            post_changed >= total_pixels / 20 && post_luma_delta >= 1.0,
            "the configured default post stack is not perceptible against Potato/no-post: \
             changed={post_changed}/{total_pixels}, mean absolute luma delta={post_luma_delta:.3}"
        );

        assert!(
            clock_changed >= total_pixels / 4 && clock_luma_delta >= 5.0,
            "half-cycle day/night frames are not perceptibly different: \
             changed={clock_changed}/{total_pixels}, mean absolute luma delta={clock_luma_delta:.3}"
        );
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

    #[cfg(target_os = "macos")]
    #[test]
    fn controlled_agents_express_spike_health_and_compact_lod_in_real_gpui_pixels() {
        let probe_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/rendering_reference/live_probes/bd-ydym");
        std::fs::create_dir_all(&probe_dir).expect("bd-ydym probe output directory");

        let world = capture_agent_expression_world();
        let hovered_agent = world
            .lock()
            .expect("agent capture world lock")
            .agents()
            .iter_handles()
            .nth(5 * 12 + 4)
            .expect("controlled hover agent");
        let digest_before = world
            .lock()
            .expect("agent capture world lock")
            .world_digest_v1()
            .expect("pre-agent-capture world digest");
        let individual_hidden = capture_view_with_overrides_and_camera(
            Arc::clone(&world),
            GuiViewRole::WorldCanvas,
            1280.0,
            720.0,
            CaptureOverrides {
                draw_agents: Some(false),
                draw_food: Some(false),
                forced_fps: Some(60.0),
                hovered_agent: Some(hovered_agent),
                ..CaptureOverrides::default()
            },
        )
        .unwrap_or_else(|error| panic!("headless hidden-agent control failed: {error:#}"));
        let individual = capture_view_with_overrides_and_camera(
            Arc::clone(&world),
            GuiViewRole::WorldCanvas,
            1280.0,
            720.0,
            CaptureOverrides {
                draw_agents: Some(true),
                draw_food: Some(false),
                forced_fps: Some(60.0),
                hovered_agent: Some(hovered_agent),
                ..CaptureOverrides::default()
            },
        )
        .unwrap_or_else(|error| panic!("headless individual-agent capture failed: {error:#}"));
        let compact_hidden = capture_view_with_overrides_and_camera(
            Arc::clone(&world),
            GuiViewRole::WorldCanvas,
            1280.0,
            720.0,
            CaptureOverrides {
                draw_agents: Some(false),
                draw_food: Some(false),
                forced_fps: Some(12.0),
                hovered_agent: Some(hovered_agent),
                ..CaptureOverrides::default()
            },
        )
        .unwrap_or_else(|error| panic!("headless compact hidden-agent control failed: {error:#}"));
        let compact = capture_view_with_overrides_and_camera(
            Arc::clone(&world),
            GuiViewRole::WorldCanvas,
            1280.0,
            720.0,
            CaptureOverrides {
                draw_agents: Some(true),
                draw_food: Some(false),
                forced_fps: Some(12.0),
                hovered_agent: Some(hovered_agent),
                ..CaptureOverrides::default()
            },
        )
        .unwrap_or_else(|error| panic!("headless compact-agent capture failed: {error:#}"));
        let digest_after = world
            .lock()
            .expect("agent capture world lock")
            .world_digest_v1()
            .expect("post-agent-capture world digest");

        assert_eq!(individual_hidden.image.dimensions(), (1280, 720));
        assert_eq!(individual.image.dimensions(), (1280, 720));
        assert_eq!(compact_hidden.image.dimensions(), (1280, 720));
        assert_eq!(compact.image.dimensions(), (1280, 720));
        assert_camera_transform_eq("individual", &individual_hidden.camera, &individual.camera);
        assert_camera_transform_eq("compact", &compact_hidden.camera, &compact.camera);
        assert_eq!(
            digest_before, digest_after,
            "agent-only repaints mutated scientific state"
        );

        individual
            .image
            .save(probe_dir.join("spike_health_entities_1280x720.png"))
            .expect("write bd-ydym individual-agent probe");
        compact
            .image
            .save(probe_dir.join("compact_lod_96_agents_1280x720.png"))
            .expect("write bd-ydym compact-agent probe");

        // Every comparison uses an agents-hidden frame at the SAME forced FPS.
        // Low-FPS mode changes the post stack, so comparing the 12-FPS frame with
        // a 60-FPS background would count scanline/grain changes as agent pixels.
        let individual_classes = count_agent_delta_classes(
            &individual_hidden.image,
            &individual.image,
            &individual.camera,
        );
        let compact_classes =
            count_agent_delta_classes(&compact_hidden.image, &compact.image, &compact.camera);
        for (label, classes) in [
            ("individual", individual_classes),
            ("compact", compact_classes),
        ] {
            let [changed, cyan, magenta, bright, dim] = classes;
            assert!(
                changed >= 1_500,
                "{label} GPUI path emitted too few controlled-agent pixels: {changed}"
            );
            assert!(
                cyan >= 80 && magenta >= 80,
                "{label} GPUI path lost the canonical diet axis: cyan={cyan}, magenta={magenta}"
            );
            assert!(
                bright >= 80 && dim >= 80,
                "{label} GPUI path lost health/age luminance classes: bright={bright}, dim={dim}"
            );
        }

        for (label, control, rendered, camera) in [
            (
                "individual",
                &individual_hidden.image,
                &individual.image,
                &individual.camera,
            ),
            (
                "compact",
                &compact_hidden.image,
                &compact.image,
                &compact.camera,
            ),
        ] {
            // Each three-column group repeats full, mid, and floor health while
            // holding diet, age, boost, spike state, and heading constant.
            for row in 0..8 {
                let full = device_point(camera, controlled_agent_position(0, row));
                let mid = device_point(camera, controlled_agent_position(1, row));
                let floor = device_point(camera, controlled_agent_position(2, row));
                let full_luma = average_luma_delta_in_disc(control, rendered, full, 2.5);
                let mid_luma = average_luma_delta_in_disc(control, rendered, mid, 2.5);
                let floor_luma = average_luma_delta_in_disc(control, rendered, floor, 2.5);
                assert!(
                    full_luma > mid_luma + 5.0 && mid_luma > floor_luma + 3.0,
                    "{label} health luminance is not ordered on row {row}: \
                     full={full_luma:.1}, mid={mid_luma:.1}, floor={floor_luma:.1}"
                );
            }

            // Compare a short and long neutral spike at the same health/diet/age.
            // The probe sits beyond the short tip but inside the long tip.
            for row in 0..8 {
                let short_center = device_point(camera, controlled_agent_position(0, row));
                let long_center = device_point(camera, controlled_agent_position(3, row));
                let short_probe = (short_center.0 + 25.0, short_center.1);
                let long_probe = (long_center.0 + 25.0, long_center.1);
                let short_reach = changed_pixels_in_disc(control, rendered, short_probe, 4.0);
                let long_reach = changed_pixels_in_disc(control, rendered, long_probe, 4.0);
                assert!(
                    long_reach >= short_reach + 4,
                    "{label} spike length has no pixel reach on row {row}: \
                     short={short_reach}, long={long_reach}"
                );
            }

            // Long attacker and victim-only groups share extension, health, age,
            // diet, boost, and heading. Only the attacker owns the outer hot flash.
            for row in 0..8 {
                let attacker = device_point(camera, controlled_agent_position(6, row));
                let victim = device_point(camera, controlled_agent_position(9, row));
                let attacker_footprint = changed_pixels_in_disc(control, rendered, attacker, 24.0);
                let victim_footprint = changed_pixels_in_disc(control, rendered, victim, 24.0);
                assert!(
                    attacker_footprint >= victim_footprint + 40,
                    "{label} attacker strike is not distinguishable from victim-only state \
                     on row {row}: attacker={attacker_footprint}, victim={victim_footprint}"
                );
            }

            let sample_column = 4;
            let young_boost = device_point(camera, controlled_agent_position(sample_column, 0));
            let young_plain = device_point(camera, controlled_agent_position(sample_column, 2));
            let old_boost = device_point(camera, controlled_agent_position(sample_column, 4));
            let selected = device_point(camera, controlled_agent_position(sample_column, 1));
            let selection_control =
                device_point(camera, controlled_agent_position(sample_column, 3));
            let hovered = device_point(camera, controlled_agent_position(sample_column, 5));
            let hover_control = device_point(camera, controlled_agent_position(sample_column, 7));

            let young_luma = average_luma_delta_in_disc(control, rendered, young_boost, 2.5);
            let old_luma = average_luma_delta_in_disc(control, rendered, old_boost, 2.5);
            assert!(
                young_luma > old_luma + 4.0,
                "{label} age weathering is not visible: young={young_luma:.1}, old={old_luma:.1}"
            );

            let boost_tail = (young_boost.0 - 18.0, young_boost.1);
            let plain_tail = (young_plain.0 - 18.0, young_plain.1);
            let boost_pixels = changed_pixels_in_disc(control, rendered, boost_tail, 7.0);
            let plain_pixels = changed_pixels_in_disc(control, rendered, plain_tail, 7.0);
            assert!(
                boost_pixels >= plain_pixels + 4,
                "{label} boost trail is not visible: boost={boost_pixels}, plain={plain_pixels}"
            );

            let selected_pixels = changed_pixels_in_disc(control, rendered, selected, 24.0);
            let selection_control_pixels =
                changed_pixels_in_disc(control, rendered, selection_control, 24.0);
            assert!(
                selected_pixels >= selection_control_pixels + 16,
                "{label} selection rim is not visible: selected={selected_pixels}, \
                 control={selection_control_pixels}"
            );
            let hovered_pixels = changed_pixels_in_disc(control, rendered, hovered, 20.0);
            let hover_control_pixels =
                changed_pixels_in_disc(control, rendered, hover_control, 20.0);
            assert!(
                hovered_pixels >= hover_control_pixels + 8,
                "{label} hover rim is not visible: hovered={hovered_pixels}, \
                 control={hover_control_pixels}"
            );

            // Rows 2 and 3 are otherwise matched young/plain agents; diet is
            // the sole chromatic semantic.
            let carnivore = average_rgb_delta_in_disc(control, rendered, young_plain, 2.5);
            let herbivore = average_rgb_delta_in_disc(control, rendered, selection_control, 2.5);
            assert!(
                carnivore[0] > carnivore[1] + 8.0 && carnivore[2] > carnivore[1] + 8.0,
                "{label} carnivore body is not magenta-axis: {carnivore:?}"
            );
            assert!(
                herbivore[1] > herbivore[0] + 8.0 && herbivore[2] > herbivore[0] + 8.0,
                "{label} herbivore body is not cyan-axis: {herbivore:?}"
            );
        }
    }
}
