//! C++ parity for the two clock sensor channels (bd-drhs).
//!
//! Reference: `original_scriptbots_code_for_reference/World.cpp`.
//!
//! ```text
//! void World::update()
//! {
//!     modcounter++;                                   // line 36 — BEFORE setInputs
//!     ...
//!     if (modcounter>=10000) {                        // lines 54-56
//!         modcounter=0;
//!         current_epoch++;
//!     }
//!     ...
//!     a->in[16]= abs(sin(modcounter/a->clockf1));     // lines 298-299
//!     a->in[17]= abs(sin(modcounter/a->clockf2));
//! }
//! ```
//!
//! Two consequences, and Rust honours neither:
//!
//! 1. The counter is **incremented before** the inputs are read, so the first transition
//!    presents `1`, not `0`. Rust's `stage_sense` reads `self.tick`, which is still the
//!    *completed* tick at that point, so every clock reading is one transition stale.
//!
//! 2. The counter **resets to 0 at 10000**, so the argument is bounded and the channel is
//!    periodic across epochs. Rust has no reset at all, so the argument grows without bound
//!    and the two implementations diverge permanently after the first epoch.
//!
//! This is a sensor-encoding divergence, which is the area the project's parity oracle cares
//! about most: it feeds every brain on every tick, so it changes actions, which changes the
//! whole trajectory.

use scriptbots_core::{AgentData, ScriptBotsConfig, WorldState};

/// The exact legacy expression, kept as a separate literal transcription of World.cpp:298
/// so the test is checking C++ semantics rather than re-deriving them from the Rust code.
#[expect(
    clippy::cast_precision_loss,
    reason = "Preserve the oracle's integer-to-f32 division from World.cpp:298; the tested counters 1..=4 are exactly representable"
)]
fn legacy_clock_input(modcounter: u64, clock_frequency: f32) -> f32 {
    (modcounter as f32 / clock_frequency.max(1.0)).sin().abs()
}

fn single_agent_world(seed: u64) -> (WorldState, scriptbots_core::AgentId) {
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        // Keep the population fixed so the agent under observation survives the window.
        closed: true,
        population_minimum: 0,
        population_spawn_interval: 0,
        reproduction_attempt_interval: 0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        rng_seed: Some(seed),
        ..ScriptBotsConfig::default()
    })
    .expect("world");

    let id = world
        .try_spawn_agent(AgentData {
            position: scriptbots_core::Position::new(100.0, 100.0),
            health: 2.0,
            ..AgentData::default()
        })
        .expect("seed agent");

    // Pin the clock frequencies so the expected value is exactly computable.
    world
        .try_update_agent_runtime(id, |runtime| {
            runtime.clocks = [7.0, 13.0];
        })
        .expect("pin clock frequencies");

    (world, id)
}

/// bd-drhs part 1: the clock argument must be the transition being computed, not the one
/// already completed.
///
/// After the first `step()` the world has completed tick 1, and C++ would have presented
/// `modcounter == 1` to the brain during that transition. Rust presents `0`.
#[test]
fn clock_channels_use_the_incremented_counter_like_world_cpp() {
    let (mut world, id) = single_agent_world(0xC10C_C0DE);

    for expected_modcounter in 1..=4u64 {
        world.step().expect("science tick");

        let runtime = world.agent_runtime(id).expect("agent survived the window");
        let clocks = runtime.clocks;

        assert_eq!(
            runtime.sensors[16],
            legacy_clock_input(expected_modcounter, clocks[0]),
            "clock1 must be sin(|modcounter/clockf1|) for the transition just computed \
             (modcounter={expected_modcounter}); reading the completed tick instead leaves \
             the channel one transition stale"
        );
        assert_eq!(
            runtime.sensors[17],
            legacy_clock_input(expected_modcounter, clocks[1]),
            "clock2 must track the same counter as clock1 (modcounter={expected_modcounter})"
        );
    }
}

/// bd-drhs part 1, stated as the sharpest single case: the very first transition.
///
/// C++ presents 1. The pre-fix Rust presents 0, and `sin(0).abs()` is exactly 0.0, so this
/// assertion fails against a literal zero — an unambiguous red rather than a rounding
/// argument.
#[test]
fn the_first_transition_does_not_present_a_zero_clock() {
    let (mut world, id) = single_agent_world(0xF185_7C1C);

    world.step().expect("first science tick");

    let runtime = world.agent_runtime(id).expect("agent survived");
    assert_ne!(
        runtime.sensors[16], 0.0,
        "the first transition must present modcounter=1, not the completed tick 0; \
         a literal 0.0 here is the off-by-one"
    );
    assert_eq!(
        runtime.sensors[16],
        legacy_clock_input(1, runtime.clocks[0]),
        "the first transition must match World.cpp with modcounter=1"
    );
}
