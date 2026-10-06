//! Core diagnostics that read a wall clock must not panic in the browser (bd-3dzw).
//!
//! `std::time::Instant::now()` and `SystemTime::now()` panic on wasm32-unknown-unknown, which has
//! no clock. Core times a profiled step and stamps generated map artifacts with the wall clock;
//! neither is reachable from `Simulation` today, so the browser build compiled cleanly while a
//! panic waited for the first caller. These run the real paths in a wasm runtime.
//!
//! This is its own test binary because the library's wasm tests are configured
//! `run_in_browser`; these need only a JS host with `performance` and `Date`, so they also run
//! under Node:
//!
//! ```text
//! CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
//!   cargo test -p scriptbots-web --target wasm32-unknown-unknown --test wasm_clock
//! ```
#![cfg(target_arch = "wasm32")]

use scriptbots_core::{
    RuleBasedMapGenerator, ScriptBotsConfig, WorldState, WorldStepProfiler, default_tileset_spec,
};
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn a_profiled_step_times_its_stages_in_the_browser() {
    let mut world = WorldState::new(ScriptBotsConfig::default()).expect("default world");
    let mut profiler = WorldStepProfiler::default();
    world
        .step_profiled(&mut profiler)
        .expect("profiled step completes");
    assert!(
        profiler.latest().is_some(),
        "the profiled step must leave a timing profile"
    );
}

#[wasm_bindgen_test]
fn a_generated_map_is_stamped_with_the_wall_clock_in_the_browser() {
    let generator =
        RuleBasedMapGenerator::new(default_tileset_spec()).expect("default tileset compiles");
    let artifact = generator.generate(12, 12, 8, 7).expect("map generates");
    assert!(
        artifact.metadata().generated_at_epoch_ms > 0,
        "the artifact must carry a real epoch timestamp"
    );
}
