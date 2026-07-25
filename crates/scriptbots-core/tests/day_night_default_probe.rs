//! Whether the day/night cycle the code implements is ever actually seen (bd-lhml).
//!
//! bd-lhml asks a question that cannot be answered by grepping: the post stack and day/night
//! are extensively implemented, yet the owner reports seeing neither. "Implemented but not
//! running" and "running but imperceptible" look identical from outside and have completely
//! different fixes.
//!
//! For day/night specifically, the answer is neither of those two: it RUNS, it is correctly
//! threaded through the live GPUI paint path, and it is nonetheless CONSTANT, because the
//! feature is opt-in and nothing turns it on.
//!
//! THE TRACE, end to end:
//!
//! ```text
//!   ScriptBotsConfig::default().render.day_night        == None
//!     -> scriptbots-render/src/lib.rs:12894, `.map_or((0, 0.25), ...)`
//!        so RenderFrame::day_night_cycle_ticks           == 0
//!     -> scriptbots-render/src/lib.rs:15155, daylight_factor(tick, 0, 0.25)
//!     -> visual::daylight_factor returns DAYLIGHT_STATIC on `cycle_ticks == 0`
//! ```
//!
//! `daylight` then feeds the sky gradient, the horizon band, the aurora, and terrain shading
//! on the interactive canvas -- all of it real, all of it pinned to one value forever.
//!
//! This test deliberately asserts on the DEFAULT config. Every place day/night is currently
//! demonstrated sets `cycle_ticks` explicitly, which is precisely why a permanently static
//! default survived: the curve is well tested, so nobody checked whether anything selects it.
//! Compare `sense_radius` (bd-rs9f), where every sensing test overrode the default and a
//! divergence in the default survived for exactly the same reason.

use scriptbots_core::{ScriptBotsConfig, visual};

/// How the renderer resolves the config into the curve's `cycle_ticks` argument.
///
/// Transcribed from `crates/scriptbots-render/src/lib.rs:12894`. Core cannot depend on the
/// render crate, so this mirrors the mapping rather than calling it; if the renderer changes
/// its default, this transcription is what must be updated alongside.
fn renderer_cycle_ticks(config: &ScriptBotsConfig) -> u32 {
    config
        .render
        .day_night
        .as_ref()
        .map_or(0, |settings| settings.cycle_ticks.unwrap_or(0))
}

/// bd-lhml: at the default configuration the day/night curve never moves.
///
/// IGNORED PENDING bd-lhml: this passes once the default configuration selects a day/night
/// cycle. It currently fails reporting a single distinct daylight value across a full
/// simulated day, which is the defect itself, not a broken test.
///
/// Ignored rather than left red for the reason established on bd-rs9f: a permanently failing
/// suite trains everyone to read red as normal, and then the suite stops being a signal.
/// Removing this one attribute is the proof when the default lands.
#[test]
#[ignore = "bd-lhml: passes when the default configuration selects a day/night cycle"]
fn the_default_configuration_actually_varies_daylight_over_a_day() {
    let config = ScriptBotsConfig::default();
    let cycle_ticks = renderer_cycle_ticks(&config);
    let start_phase = config
        .render
        .day_night
        .as_ref()
        .and_then(|settings| settings.start_phase)
        .unwrap_or(0.25);

    // Sample across whatever cycle the default selects; fall back to a day's worth of ticks so
    // the zero case still samples widely instead of trivially reading one tick.
    let span = if cycle_ticks == 0 {
        10_000
    } else {
        cycle_ticks
    };
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for step in 0..64u64 {
        let tick = step * u64::from(span) / 64;
        let daylight = visual::daylight_factor(tick, cycle_ticks, start_phase);
        minimum = minimum.min(daylight);
        maximum = maximum.max(daylight);
    }

    assert!(
        maximum - minimum > 0.1,
        "the default configuration produces a CONSTANT daylight of {minimum} across {span} \
         ticks (observed range {minimum}..={maximum}). `render.day_night` defaults to None, the \
         renderer maps that to cycle_ticks=0, and `daylight_factor` short-circuits to \
         DAYLIGHT_STATIC ({static_value}). The curve, the sky gradient, the horizon band, the \
         aurora and the terrain daylight term are all live on the interactive canvas and all \
         pinned to one value, so the day/night cycle can never be observed at the default \
         quality tier.",
        static_value = visual::DAYLIGHT_STATIC,
    );
}

/// The companion assertion that keeps the test above non-vacuous.
///
/// If `daylight_factor` were broken outright, the test above would fail for a reason that has
/// nothing to do with defaults and the diagnosis would be wrong. This proves the curve itself
/// is healthy, which is what isolates the defect to the DEFAULT rather than the implementation.
#[test]
fn the_daylight_curve_itself_varies_once_a_cycle_is_selected() {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for tick in 0..1_000u64 {
        let daylight = visual::daylight_factor(tick, 1_000, 0.25);
        minimum = minimum.min(daylight);
        maximum = maximum.max(daylight);
    }

    assert!(
        (minimum - visual::DAYLIGHT_NIGHT_FLOOR).abs() < 1.0e-3,
        "midnight must reach the night floor {}, got {minimum}",
        visual::DAYLIGHT_NIGHT_FLOOR
    );
    assert!(
        (maximum - 1.0).abs() < 1.0e-3,
        "noon must reach full daylight 1.0, got {maximum}"
    );
}
