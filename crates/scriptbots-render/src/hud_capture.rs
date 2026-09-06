//! Headless GPUI HUD capture (bd-abu3).
//!
//! The scene harness in `scriptbots-app` can capture the Bevy world offscreen, and
//! `--dump-png` can rasterize the world through `render_png_offscreen`. Neither draws
//! HUD chrome: `render_png_offscreen` takes a host snapshot and builds its own camera, so
//! it never constructs a [`SimulationView`] and can never contain a panel. That left
//! every claim about HUD layout (bd-v9cz, and everything bd-f4x0 will do) resting on
//! code inspection.
//!
//! This module closes that gap by rendering the REAL element tree — the same
//! [`SimulationView`] the production windows use, via the same [`GuiSession`] — through
//! GPUI's headless Metal renderer, and reading the frame back as RGBA. No window server
//! and no display are involved, so it runs in CI and under an agent session.
//! Real pixel capture requires macOS: the pinned `gpui_platform::current_headless_renderer`
//! returns a Metal renderer there and `None` on other platforms (bd-h9ca).
//!
//! It lives inside the crate rather than in `tests/` because [`GuiSession`] and
//! [`SimulationView`] are private to the crate root; a child module can reach them, an
//! integration test cannot. It is `#[cfg(test)]` because GPUI's `HeadlessAppContext` is
//! gated behind that crate's `test-support` feature, which is a dev-dependency here and
//! must not leak into production builds.

use std::sync::Arc;

use gpui::{AppContext as _, HeadlessAppContext, px, size};
use image::RgbaImage;
use scriptbots_core::{AgentId, ScriptBotsConfig, WorldState};
#[cfg(target_os = "macos")]
use scriptbots_core::{
    OutputChannel, RenderDayNightSettings, RenderQuality, RenderSettings, SelectionState,
};

use crate::{
    AnalyticsSnapshotProvider, CameraSnapshot, ControlCommand, GuiSession, GuiViewRole, TestHost,
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
    /// Pins the ENTIRE PerfSnapshot, not just fps.
    ///
    /// bd-c7pg: perf reaches the frame via `snapshot.perf = self.last_perf` regardless
    /// of whether the collapsed perf panel renders, and five of its six fields are
    /// floating-point timings that differ every run. Pinning only `fps` left
    /// latest_ms/average_ms/min_ms/max_ms/sample_count varying — rendered text of five
    /// changing numbers, which is a plausible ~10k pixels of glyph difference.
    forced_perf: Option<crate::PerfSnapshot>,
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
    if let Some(perf) = overrides.forced_perf {
        view.last_perf = perf;
        // Sticky, not initial: without this render() overwrites it on frame one.
        view.forced_perf = Some(perf);
    }
    if let Some(hovered_agent) = overrides.hovered_agent
        && let Ok(mut inspector) = view.inspector.lock()
    {
        inspector.hovered_agent = Some(hovered_agent);
    }
}

/// Render one GPUI view offscreen at exact output dimensions in device pixels.
pub(crate) fn capture_view(
    world: Arc<TestHost>,
    role: GuiViewRole,
    width: f32,
    height: f32,
) -> Result<RgbaImage, String> {
    capture_view_with_overrides(world, role, width, height, CaptureOverrides::default())
}

#[cfg(target_os = "macos")]
fn capture_view_with_world_painter(
    world: Arc<TestHost>,
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
    world: Arc<TestHost>,
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
    world: Arc<TestHost>,
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

    let command_host = Arc::clone(&world);
    let command_submit: Arc<dyn Fn(ControlCommand) -> Option<String> + Send + Sync> =
        Arc::new(move |command| {
            let command = scriptbots_runtime::HostCommand::try_from(command).ok()?;
            let status = command_host.submit(command);
            Some(status.command_id().to_string())
        });

    let session = Arc::new(GuiSession::new(
        world.port.clone(),
        AnalyticsSnapshotProvider::empty(),
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
    //! Required real-pixel tests and their exclusive helpers use `cfg(target_os = "macos")`
    //! to match GPUI's actual headless-renderer availability (bd-h9ca). This is a compile-time
    //! platform boundary: no test reports successful pixel capture on an unsupported platform.
    //! The existing ignored diagnostics keep their explicit-run behavior and also need macOS
    //! to capture pixels. Platform-independent checks remain available on other targets.

    use super::*;
    #[cfg(target_os = "macos")]
    use std::path::PathBuf;

    /// Representative logical sizes from the production horizontal split. The World
    /// owns the flexible remainder; the Lab stays within its 380–480 px clamp.
    #[cfg(target_os = "macos")]
    const WORLD_VIEWPORT: (f32, f32) = (1400.0, 768.0);
    #[cfg(target_os = "macos")]
    const LAB_VIEWPORTS: [(f32, f32); 2] = [(460.0, 768.0), (380.0, 600.0)];

    fn capture_world() -> Arc<TestHost> {
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
        Arc::new(TestHost::new(
            WorldState::new(config).expect("offscreen capture world"),
        ))
    }

    #[cfg(target_os = "macos")]
    fn build_visual_world(render: RenderSettings) -> WorldState {
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

        world
    }

    #[cfg(target_os = "macos")]
    fn capture_visual_world_with_render(render: RenderSettings) -> Arc<TestHost> {
        Arc::new(TestHost::new(build_visual_world(render)))
    }

    #[cfg(target_os = "macos")]
    fn capture_visual_world() -> Arc<TestHost> {
        capture_visual_world_with_render(RenderSettings::default())
    }

    #[cfg(target_os = "macos")]
    fn controlled_agent_position(column: usize, row: usize) -> (f32, f32) {
        (130.0 + column as f32 * 194.0, 105.0 + row as f32 * 141.0)
    }

    #[cfg(target_os = "macos")]
    fn capture_agent_expression_world() -> Arc<TestHost> {
        const COLUMNS: usize = 12;

        let mut guard = build_visual_world(RenderSettings::default());
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
        Arc::new(TestHost::new(guard))
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

    /// Pixel tests are side-effect free by default. An explicit external directory
    /// turns on PNG retention for a human visual pass without rewriting checked-in
    /// historical probes during ordinary `cargo test`.
    #[cfg(target_os = "macos")]
    fn probe_dir() -> Option<PathBuf> {
        std::env::var_os("SCRIPTBOTS_HUD_PROBE_DIR").map(PathBuf::from)
    }

    #[cfg(target_os = "macos")]
    fn save_probe(image: &RgbaImage, name: &str) -> Option<PathBuf> {
        let directory = probe_dir()?;
        std::fs::create_dir_all(&directory).expect("probe output directory");
        let path = directory.join(name);
        image.save(&path).expect("write GUI probe png");
        Some(path)
    }

    /// Column span of the world canvas, found by colour diversity.
    ///
    /// The world raster is the only element that paints a continuously varying image;
    /// every piece of chrome is a flat fill with text on it. So the canvas is the widest
    /// run of adjacent columns that each carry many distinct colours. Returning a
    /// measured span rather than a hardcoded rectangle keeps this honest when the layout
    /// changes — which it just did, when the rail was hoisted.
    #[cfg(target_os = "macos")]
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

    /// Every perf field frozen. Any unpinned float here reaches rendered text and
    /// varies per run, which is the bd-c7pg floor.
    fn pinned_perf() -> crate::PerfSnapshot {
        pinned_perf_at_fps(60.0)
    }

    fn pinned_perf_at_fps(fps: f32) -> crate::PerfSnapshot {
        crate::PerfSnapshot {
            latest_ms: 16.0,
            average_ms: 16.0,
            min_ms: 16.0,
            max_ms: 16.0,
            sample_count: 128,
            fps,
        }
    }

    /// bd-c7pg: with perf pinned, the harness must be DETERMINISTIC — two captures of
    /// the same world at the same tick must be byte-identical, not merely close.
    ///
    /// Exact equality is the only honest assertion. A tolerance here would be a
    /// threshold tuned to hide the very noise the bead is about, which is the same
    /// defect wearing a different hat.
    ///
    /// macOS-gated for the headless-renderer limitation documented by this module (bd-h9ca).
    #[cfg(target_os = "macos")]
    #[test]
    fn captures_of_one_world_are_byte_identical_when_perf_is_pinned() {
        let (w, h) = (
            1280.0 * HEADLESS_DEVICE_SCALE,
            720.0 * HEADLESS_DEVICE_SCALE,
        );
        let world = capture_world();
        let overrides = CaptureOverrides {
            forced_perf: Some(pinned_perf()),
            ..CaptureOverrides::default()
        };
        let first =
            capture_view_with_overrides(Arc::clone(&world), GuiViewRole::Hud, w, h, overrides)
                .expect("first capture");
        let second =
            capture_view_with_overrides(Arc::clone(&world), GuiViewRole::Hud, w, h, overrides)
                .expect("second capture");

        let differing = first
            .pixels()
            .zip(second.pixels())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            differing, 0,
            "two captures of one world at one tick differ by {differing} px with perf \
             pinned; a residual means the bd-c7pg floor is only partly explained and the \
             remaining source must be named, not tolerated"
        );
    }

    /// bd-c7pg: does DISPATCHING AT ALL perturb the frame?
    ///
    /// Six hypotheses eliminated so far all assumed the floor came from something
    /// differing between the two captures. The no-op-key control disproved that: a key
    /// that changes nothing still moved ~14.8k px. The one remaining difference between
    /// the byte-identical test (0 px) and the audit (~14.8k) is that the audit
    /// dispatches a keystroke at all.
    ///
    /// A: two captures, neither dispatching.   B: one dispatching an unbound key.
    /// If A is 0 and B is nonzero, the dispatch machinery itself is the source.
    #[test]
    #[ignore = "bd-c7pg diagnostic; run explicitly with --ignored"]
    fn diagnose_whether_dispatch_alone_perturbs_the_frame() {
        let (w, h) = (
            1280.0 * HEADLESS_DEVICE_SCALE,
            720.0 * HEADLESS_DEVICE_SCALE,
        );
        let world = capture_world();
        let none = CaptureOverrides {
            forced_perf: Some(pinned_perf()),
            ..CaptureOverrides::default()
        };
        let zed = CaptureOverrides {
            forced_perf: Some(pinned_perf()),
            keystrokes: &["z"],
            ..CaptureOverrides::default()
        };
        let cap = |o| {
            capture_view_with_overrides(Arc::clone(&world), GuiViewRole::Hud, w, h, o)
                .expect("capture")
        };
        let a1 = cap(none);
        let a2 = cap(none);
        let b1 = cap(zed);
        let b2 = cap(zed);
        let diff = |x: &RgbaImage, y: &RgbaImage| {
            x.pixels().zip(y.pixels()).filter(|(p, q)| p != q).count()
        };
        println!("  A none-vs-none : {}", diff(&a1, &a2));
        println!("  B z-vs-z       : {}", diff(&b1, &b2));
        println!("  C none-vs-z    : {}", diff(&a1, &b1));
    }

    /// bd-c7pg CAUSATION PROBE — is the ~10k floor measurement noise, or two different
    /// worlds?
    ///
    /// Captures twice and reports the world tick after each. If the tick advances, the
    /// differing pixels are not noise at all: they are real simulation change being
    /// misread as instrument error, and the bead is about a harness photographing two
    /// different worlds rather than a flaky renderer.
    #[test]
    #[ignore = "bd-c7pg diagnostic; run explicitly with --ignored"]
    fn diagnose_whether_the_world_advances_between_captures() {
        let (w, h) = (
            1280.0 * HEADLESS_DEVICE_SCALE,
            720.0 * HEADLESS_DEVICE_SCALE,
        );

        // Case A: two captures sharing ONE world. Any tick advance is caused by capture.
        let shared = capture_world();
        let t0 = shared.snapshot().world.tick;
        let a1 = capture_view(Arc::clone(&shared), GuiViewRole::Hud, w, h).expect("a1");
        let t1 = shared.snapshot().world.tick;
        let a2 = capture_view(Arc::clone(&shared), GuiViewRole::Hud, w, h).expect("a2");
        let t2 = shared.snapshot().world.tick;
        let shared_delta = a1.pixels().zip(a2.pixels()).filter(|(x, y)| x != y).count();

        // Case B: two captures from two FRESH worlds, as every existing test does.
        let b1 = capture_view(capture_world(), GuiViewRole::Hud, w, h).expect("b1");
        let b2 = capture_view(capture_world(), GuiViewRole::Hud, w, h).expect("b2");
        let fresh_delta = b1.pixels().zip(b2.pixels()).filter(|(x, y)| x != y).count();

        println!(
            "  shared world ticks: {t0} -> {t1} -> {t2}  (advance per capture: {})",
            t1 - t0
        );
        println!("  shared-world pixel delta: {shared_delta}");
        println!("  fresh-world  pixel delta: {fresh_delta}");
        println!(
            "  VERDICT: {}",
            if t1 > t0 {
                "world ADVANCES during capture — the floor is real simulation change"
            } else {
                "world does not advance — the floor has another source"
            }
        );
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
        const SHORTCUTS: [(&str, &str, GuiViewRole); 26] = [
            ("space", "Toggle playback", GuiViewRole::Hud),
            ("g", "Jump to live", GuiViewRole::Hud),
            ("n", "Toggle narration", GuiViewRole::WorldCanvas),
            ("ctrl-p", "Cycle palette", GuiViewRole::WorldCanvas),
            ("p", "Toggle simulation pause", GuiViewRole::Hud),
            ("s", "Step simulation once", GuiViewRole::Hud),
            ("d", "Toggle agent drawing", GuiViewRole::WorldCanvas),
            ("f", "Toggle food overlay", GuiViewRole::WorldCanvas),
            (
                "ctrl-shift-o",
                "Toggle agent outline",
                GuiViewRole::WorldCanvas,
            ),
            ("shift-=", "Increase speed", GuiViewRole::Hud),
            ("-", "Decrease speed", GuiViewRole::Hud),
            ("a", "Spawn crossover", GuiViewRole::Hud),
            ("q", "Spawn carnivore", GuiViewRole::Hud),
            ("h", "Spawn herbivore", GuiViewRole::Hud),
            ("c", "Toggle closed environment", GuiViewRole::Hud),
            ("shift-s", "Follow selected", GuiViewRole::WorldCanvas),
            ("o", "Follow oldest", GuiViewRole::WorldCanvas),
            ("shift-f", "Toggle debug overlay", GuiViewRole::WorldCanvas),
            ("escape", "Clear selection", GuiViewRole::Hud),
            ("ctrl-a", "Select all", GuiViewRole::Hud),
            ("ctrl-f", "Focus first selected", GuiViewRole::Hud),
            ("0", "Fit world", GuiViewRole::WorldCanvas),
            (",", "Toggle settings panel", GuiViewRole::WorldCanvas),
            ("1", "Toggle stats panel", GuiViewRole::Hud),
            ("2", "Toggle history panel", GuiViewRole::Hud),
            ("3", "Toggle performance panel", GuiViewRole::Hud),
        ];
        let (w, h) = (
            1280.0 * HEADLESS_DEVICE_SCALE,
            720.0 * HEADLESS_DEVICE_SCALE,
        );
        // Pin the perf readout on BOTH sides. Without it the live FPS/timing text
        // redraws between captures and every frame differs from baseline by ~12k px
        // regardless of the keystroke — a noise floor that swallowed 21 of 27 results
        // on the first run and would have let a dead control read as working.
        // Baseline dispatches an UNBOUND key ("z" has no binding). Previously it
        // dispatched nothing while each shortcut capture dispatched one, so the sides
        // differed in input history AND frame count - confounded. Now both take an
        // identical dispatch path, so any delta is the shortcut's own effect.
        // Shortcut captures override `keystrokes` explicitly, so only the baseline
        // picks this up.
        let stable = || CaptureOverrides {
            forced_perf: Some(pinned_perf()),
            keystrokes: &["z"],
            ..CaptureOverrides::default()
        };
        // ONE world cloned into every capture (bd-c7pg isolation). Building a fresh
        // world per capture left world construction as an uncontrolled variable, so a
        // residual could not be attributed to the render path.
        let world = capture_world();
        let base_lab =
            capture_view_with_overrides(Arc::clone(&world), GuiViewRole::Hud, w, h, stable())
                .expect("Lab baseline");
        let base_world = capture_view_with_overrides(
            Arc::clone(&world),
            GuiViewRole::WorldCanvas,
            w,
            h,
            stable(),
        )
        .expect("World baseline");

        let mut inert = Vec::new();
        for (stroke, label, role) in SHORTCUTS {
            let leaked: &'static [&'static str] = Box::leak(vec![stroke].into_boxed_slice());
            let after = capture_view_with_overrides(
                Arc::clone(&world),
                role,
                w,
                h,
                CaptureOverrides {
                    forced_perf: Some(pinned_perf()),
                    keystrokes: leaked,
                    ..CaptureOverrides::default()
                },
            )
            .unwrap_or_else(|e| panic!("capture after {stroke:?} failed: {e:#}"));
            let base = match role {
                GuiViewRole::Hud => &base_lab,
                GuiViewRole::WorldCanvas => &base_world,
            };
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

    /// The production pair is complementary: the wide World contains the simulation
    /// raster, while the narrow Lab contains controls and telemetry instead of a second
    /// copy of that raster.
    #[cfg(target_os = "macos")]
    #[test]
    fn world_and_lab_are_complementary_surfaces_not_duplicate_canvases() {
        let (world_logical_w, world_logical_h) = WORLD_VIEWPORT;
        let (lab_logical_w, lab_logical_h) = LAB_VIEWPORTS[0];
        let overrides = CaptureOverrides {
            forced_perf: Some(pinned_perf()),
            ..CaptureOverrides::default()
        };
        let world = capture_world();
        let world_image = capture_view_with_overrides(
            Arc::clone(&world),
            GuiViewRole::WorldCanvas,
            world_logical_w * HEADLESS_DEVICE_SCALE,
            world_logical_h * HEADLESS_DEVICE_SCALE,
            overrides,
        )
        .expect("World capture");
        let lab_image = capture_view_with_overrides(
            world,
            GuiViewRole::Hud,
            lab_logical_w * HEADLESS_DEVICE_SCALE,
            lab_logical_h * HEADLESS_DEVICE_SCALE,
            overrides,
        )
        .expect("Lab capture");

        let (world_lo, world_hi) = world_canvas_columns(&world_image);
        let (lab_lo, lab_hi) = world_canvas_columns(&lab_image);
        let world_span = world_hi.saturating_sub(world_lo);
        let lab_span = lab_hi.saturating_sub(lab_lo);
        assert!(
            world_span >= world_image.width() * 3 / 5,
            "World raster occupies only {world_span}/{} device px; the visual subject \
             must own most of its dedicated window",
            world_image.width()
        );
        assert!(
            lab_span <= lab_image.width() / 3,
            "Lab contains a raster-like span of {lab_span}/{} device px; the companion \
             utility window must not duplicate the World canvas",
            lab_image.width()
        );

        let _ = save_probe(
            &world_image,
            &format!(
                "world_{}x{}.png",
                world_logical_w as u32, world_logical_h as u32
            ),
        );
        let _ = save_probe(
            &lab_image,
            &format!(
                "lab_overview_{}x{}.png",
                lab_logical_w as u32, lab_logical_h as u32
            ),
        );
    }

    /// Diagnostic disclosures begin closed, but the production 1/2/3 input path still
    /// mounts them in the scrollable Overview without forcing the Lab back to a desktop
    /// dashboard width.
    #[cfg(target_os = "macos")]
    #[test]
    fn narrow_lab_disclosures_are_reachable_without_reintroducing_a_canvas() {
        for (logical_w, logical_h) in LAB_VIEWPORTS {
            let width = logical_w * HEADLESS_DEVICE_SCALE;
            let height = logical_h * HEADLESS_DEVICE_SCALE;
            let world = capture_world();
            let neutral = capture_view_with_overrides(
                Arc::clone(&world),
                GuiViewRole::Hud,
                width,
                height,
                CaptureOverrides {
                    forced_perf: Some(pinned_perf()),
                    keystrokes: &["z"],
                    ..CaptureOverrides::default()
                },
            )
            .expect("neutral-dispatch Lab capture");
            let neutral_repeat = capture_view_with_overrides(
                Arc::clone(&world),
                GuiViewRole::Hud,
                width,
                height,
                CaptureOverrides {
                    forced_perf: Some(pinned_perf()),
                    keystrokes: &["z"],
                    ..CaptureOverrides::default()
                },
            )
            .expect("repeated neutral-dispatch Lab capture");
            let (negative_changed, negative_luma_delta) =
                whole_frame_delta(&neutral, &neutral_repeat);

            assert_eq!(
                neutral.dimensions(),
                (width as u32, height as u32),
                "Lab capture must preserve its requested companion-window size"
            );

            for (shortcut, label) in [("1", "stats"), ("2", "history"), ("3", "performance")] {
                let keystrokes: &'static [&'static str] = match shortcut {
                    "1" => &["1"],
                    "2" => &["2"],
                    "3" => &["3"],
                    _ => unreachable!("fixed diagnostic shortcut table"),
                };
                let opened = capture_view_with_overrides(
                    Arc::clone(&world),
                    GuiViewRole::Hud,
                    width,
                    height,
                    CaptureOverrides {
                        forced_perf: Some(pinned_perf()),
                        keystrokes,
                        ..CaptureOverrides::default()
                    },
                )
                .unwrap_or_else(|error| panic!("open-{label} Lab capture failed: {error:#}"));
                let (changed, mean_luma_delta) = whole_frame_delta(&neutral, &opened);
                assert!(
                    changed > negative_changed.saturating_add(500)
                        && mean_luma_delta > negative_luma_delta + 0.01,
                    "pressing {shortcut} at {logical_w}x{logical_h} changed {changed} pixels \
                     (mean luma delta {mean_luma_delta:.4}), while the equal-dispatch no-op \
                     control changed {negative_changed} pixels (mean luma delta \
                     {negative_luma_delta:.4}); the {label} disclosure is not causally \
                     visible through the production input path"
                );
                let (lab_lo, lab_hi) = world_canvas_columns(&opened);
                assert!(
                    lab_hi.saturating_sub(lab_lo) <= opened.width() / 3,
                    "opening {label} at {logical_w}x{logical_h} reintroduced a \
                     world-raster-like span into the Lab"
                );

                let _ = save_probe(
                    &opened,
                    &format!("lab_{label}_{}x{}.png", logical_w as u32, logical_h as u32),
                );
            }
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

        let digest = |world: &Arc<TestHost>| {
            world
                .port
                .scientific_digest_v1()
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
                    forced_perf: Some(pinned_perf()),
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
            .port
            .scientific_digest_v1()
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
            .port
            .scientific_digest_v1()
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
            .port
            .scientific_digest_v1()
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
        let hovered_agent = AgentId::from(slotmap::KeyData::from_ffi(
            world
                .snapshot()
                .world
                .agents
                .get(5 * 12 + 4)
                .expect("controlled hover agent")
                .id,
        ));
        let digest_before = world
            .port
            .scientific_digest_v1()
            .expect("pre-agent-capture world digest");
        let individual_hidden = capture_view_with_overrides_and_camera(
            Arc::clone(&world),
            GuiViewRole::WorldCanvas,
            1280.0,
            720.0,
            CaptureOverrides {
                draw_agents: Some(false),
                draw_food: Some(false),
                forced_perf: Some(pinned_perf()),
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
                forced_perf: Some(pinned_perf()),
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
                forced_perf: Some(pinned_perf_at_fps(12.0)),
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
                forced_perf: Some(pinned_perf_at_fps(12.0)),
                hovered_agent: Some(hovered_agent),
                ..CaptureOverrides::default()
            },
        )
        .unwrap_or_else(|error| panic!("headless compact-agent capture failed: {error:#}"));
        let digest_after = world
            .port
            .scientific_digest_v1()
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
