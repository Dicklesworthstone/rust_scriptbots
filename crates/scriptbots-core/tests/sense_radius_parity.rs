//! Differential parity for the DEFAULT sensing radius (bd-rs9f).
//!
//! Reference: `original_scriptbots_code_for_reference/settings.h:29`
//!
//! ```text
//! const float DIST = 150;   // how far can the eyes see on each bot?
//! ```
//!
//! `conf::DIST` is used by `World.cpp` as BOTH the cutoff and the attenuation normalizer:
//!
//! ```text
//! if (d<conf::DIST) {
//!     smaccum   += (conf::DIST-d)/conf::DIST;
//!     soaccum   += (conf::DIST-d)/conf::DIST*(max(fabs(a2->w1),fabs(a2->w2)));
//!     hearaccum += a2->soundmul*(conf::DIST-d)/conf::DIST;
//! }
//! ```
//!
//! Rust mirrors that structure -- `distance_factor = (radius - distance) / radius` -- so the
//! radius does not merely decide WHO is visible, it rescales HOW STRONGLY every in-range
//! neighbour is sensed across smell, sound, hearing and blood.
//!
//! WHY THESE TESTS EXIST AT ALL: every pre-existing sensing test overrides `sense_radius`
//! explicitly (100.0, 5.0, a local constant), so none of them can observe the DEFAULT. That is
//! exactly how a divergence in the default survived. These tests deliberately do NOT set
//! `sense_radius`; they inherit it, which is the whole point.

use scriptbots_core::{AgentData, Position, ScriptBotsConfig, WorldState};

/// The legacy sensing distance. Both assertions below are stated against this constant rather
/// than against whatever the Rust default currently happens to be.
const LEGACY_DIST: f32 = 150.0;

/// Smell channel index in the fixed sensor layout
/// (`P1 R1 G1 B1 FOOD P2 R2 G2 B2 SOUND SMELL HEALTH P3 R3 G3 B3 CLOCK1 CLOCK2`).
const SMELL: usize = 10;

/// A world that inherits the DEFAULT sensing radius on purpose.
fn default_radius_world(seed: u64) -> ScriptBotsConfig {
    ScriptBotsConfig {
        world_width: 600,
        world_height: 600,
        closed: true,
        population_minimum: 0,
        population_spawn_interval: 0,
        reproduction_attempt_interval: 0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        rng_seed: Some(seed),
        // NOTE: sense_radius is deliberately NOT set. Overriding it here would reproduce the
        // blind spot that let this divergence survive.
        ..ScriptBotsConfig::default()
    }
}

fn smell_with_neighbor_at(separation: f32, seed: u64) -> f32 {
    let mut world = WorldState::new(default_radius_world(seed)).expect("world");
    let observer = world
        .try_spawn_agent(AgentData {
            position: Position::new(300.0, 300.0),
            health: 2.0,
            ..AgentData::default()
        })
        .expect("observer");
    world
        .try_spawn_agent(AgentData {
            position: Position::new(300.0 + separation, 300.0),
            health: 2.0,
            ..AgentData::default()
        })
        .expect("neighbour");

    // Pin the observer's smell sensitivity. `TraitModifiers::smell` is randomized per agent
    // (`rng.random_range(0.1..0.5)`), and the final channel is
    // `from_fixed(accumulator.smell) * traits.smell`. Without pinning, two worlds built from
    // different seeds carry different multipliers and a ratio between them does NOT cancel the
    // trait -- which silently turns a curve measurement into a measurement of two unrelated
    // random numbers.
    world
        .try_update_agent_runtime(observer, |runtime| {
            runtime.trait_modifiers.smell = 1.0;
        })
        .expect("pin observer smell sensitivity");

    world.step().expect("sense tick");
    world
        .agent_runtime(observer)
        .expect("observer survived")
        .sensors[SMELL]
}

/// bd-rs9f part 1: a neighbour inside the legacy range must be perceived at all.
///
/// 135 units is inside `conf::DIST` (150) and outside a 120-unit default, so the reference
/// emits a nonzero smell contribution while a short default rejects the neighbour entirely and
/// reports exactly 0.0 for every neighbour-derived channel.
/// IGNORED PENDING bd-rs9f: this passes once `ScriptBotsConfig::default` aligns
/// `sense_radius` to World.cpp's `conf::DIST` of 150. It currently fails with `got 0`, which
/// is the divergence itself, not a broken test.
///
/// Ignored rather than left red because a permanently failing suite trains everyone to read
/// red as normal, and then the suite stops being a signal. Removing this one attribute is the
/// proof when the constant lands.
#[test]
#[ignore = "bd-rs9f: passes when the default sensing radius aligns to World.cpp's 150"]
fn a_neighbor_inside_the_legacy_range_is_perceived() {
    let smell = smell_with_neighbor_at(135.0, 0x5EED_0135);
    assert!(
        smell > 0.0,
        "a neighbour 135 units away is within the legacy sensing distance of {LEGACY_DIST}, \
         so the smell channel must be nonzero; got {smell}. A literal 0.0 means the default \
         sensing radius rejected a neighbour the reference model can perceive."
    );
}

/// bd-rs9f part 2: the attenuation CURVE, not just the cutoff.
///
/// This is the assertion that makes the bead's impact clear. Because the radius is also the
/// normalizer, its value rescales every in-range neighbour. Comparing two separations cancels
/// any per-agent trait scaling, so the ratio isolates the radius itself:
///
/// ```text
/// smell(d) is proportional to (R - d) / R
/// smell(30) / smell(90) = (R - 30) / (R - 90)
///     R = 150  ->  120 / 60 = 2.0      (legacy)
///     R = 120  ->   90 / 30 = 3.0      (current default)
/// ```
///
/// So this fails loudly on the wrong radius even though BOTH neighbours are in range in either
/// case -- proving the divergence is not merely "distant neighbours are invisible".
/// IGNORED PENDING bd-rs9f: same reason as above. Currently reads 3.000 against an expected
/// 2.000, which is precisely the predicted signature of a 120-unit radius -- both neighbours
/// remain in range, so this measures the attenuation CURVE rather than the cutoff.
#[test]
#[ignore = "bd-rs9f: passes when the default sensing radius aligns to World.cpp's 150"]
fn the_attenuation_curve_matches_the_legacy_normalizer() {
    // Same seed for both: the ONLY difference between the two worlds must be the
    // separation under test.
    const SEED: u64 = 0x5EED_C0DE;
    let near = smell_with_neighbor_at(30.0, SEED);
    let far = smell_with_neighbor_at(90.0, SEED);

    assert!(
        near > 0.0 && far > 0.0,
        "both neighbours must be in range: near={near}, far={far}"
    );

    let observed = near / far;
    let expected = (LEGACY_DIST - 30.0) / (LEGACY_DIST - 90.0);
    assert!(
        (observed - expected).abs() < 0.05,
        "smell(30)/smell(90) must be {expected:.3} under the legacy normalizer of {LEGACY_DIST}, \
         got {observed:.3}. A ratio near 3.0 indicates a 120-unit radius: both neighbours are \
         still in range, but every in-range neighbour is being sensed at the wrong intensity."
    );
}
