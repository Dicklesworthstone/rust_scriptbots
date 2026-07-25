//! Keeps `explain_sensors` honest about its completed-boundary contract (bd-uucv).
//!
//! `WorldState::step` runs, in order:
//!
//! ```text
//!   Interventions -> Aging (cadence) -> FoodDynamics -> Sense -> Brains -> Actuation -> ...
//! ```
//!
//! `explain_sensors` is an instantaneous counterfactual over the completed boundary. It does not
//! project the three stages that run before the next Sense, each of which can change an input:
//!
//!   - FoodDynamics grows, decays and diffuses the observer's food cell (channel 4)
//!   - Aging damages agents on its cadence, moving health- and blood-derived channels
//!   - Interventions can spawn, remove or damage agents outright
//!
//! These tests prove that distinction at all three seams. If public documentation ever starts
//! calling the projection a next-input guarantee again, these fixtures name the exact conditions
//! under which that claim is false.

use scriptbots_core::{
    AgentData, Intervention, Position, Region, SENSOR_LAYOUT, ScriptBotsConfig, WorldState,
};

/// Food channel index in the fixed sensor layout
/// (`P1 R1 G1 B1 FOOD P2 R2 G2 B2 SOUND SMELL HEALTH P3 R3 G3 B3 CLOCK1 CLOCK2`).
const FOOD: usize = 4;
/// Health channel index in the fixed sensor layout.
const HEALTH: usize = 11;

fn sensor_differences(predicted: &[f32], actual: &[f32]) -> Vec<(usize, &'static str, f32, f32)> {
    SENSOR_LAYOUT
        .iter()
        .enumerate()
        .filter_map(|(index, channel)| {
            let predicted = predicted[index];
            let actual = actual[index];
            ((predicted - actual).abs() >= 1e-5).then_some((index, channel.name, predicted, actual))
        })
        .collect()
}

/// Live food dynamics demonstrate why the completed-boundary projection is not
/// a next-step prediction.
///
/// Everything except the food economy is held still, so any divergence isolates to the stages
/// that run between the explanation and `stage_sense` rather than to agent movement.
#[test]
fn current_boundary_projection_does_not_forecast_live_food_dynamics() {
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        // Start below every terrain-derived cell capacity so positive regrowth
        // must change the observer's cell before Sense. The earlier 0.4
        // fixture could already sit at its terrain cap and therefore proved
        // nothing despite enabling growth.
        initial_food: 0.0,
        food_max: 1.0,
        // LEFT ON. This is the only active food-dynamics path.
        food_growth_rate: 0.25,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        food_respawn_interval: 0,
        // Hold the population still so nothing moves for unrelated reasons.
        closed: true,
        population_minimum: 0,
        population_spawn_interval: 0,
        reproduction_attempt_interval: 0,
        reproduction_attempt_chance: 0.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        rng_seed: Some(77),
        ..ScriptBotsConfig::default()
    })
    .expect("world");

    let observer = world
        .try_spawn_agent(AgentData {
            position: Position::new(100.0, 100.0),
            heading: 0.0,
            health: 1.0,
            ..AgentData::default()
        })
        .expect("observer");
    for (dx, dy, health) in [
        (30.0f32, 0.0f32, 0.4f32),
        (20.0, 20.0, 1.5),
        (-25.0, 10.0, 1.0),
    ] {
        world
            .try_spawn_agent(AgentData {
                position: Position::new(100.0 + dx, 100.0 + dy),
                heading: 1.0,
                health,
                color: [0.9, 0.2, 0.5],
                ..AgentData::default()
            })
            .expect("neighbour");
    }

    let attribution = world
        .explain_sensors(observer, 16)
        .expect("observer exists");

    world.step().expect("step");
    let sensed = world.agent_runtime(observer).expect("runtime").sensors;

    // Compare index by index so this guard names exactly which pre-Sense stage
    // separates the current projection from the next runtime vector.
    let diverged = sensor_differences(&attribution.clamped, &sensed);

    assert_eq!(
        diverged.len(),
        1,
        "only FoodDynamics should separate this completed-boundary projection from the next \
         runtime vector: {diverged:?}"
    );
    assert_eq!(
        diverged[0].0, FOOD,
        "FoodDynamics must separate only the food channel: {diverged:?}"
    );
    assert_eq!(attribution.clamped[FOOD].to_bits(), 0.0_f32.to_bits());
    assert!(sensed[FOOD] > 0.0, "positive regrowth must reach Sense");
}

/// A queued intervention is already pending at the completed boundary, but the
/// instantaneous projection deliberately does not execute it.
#[test]
fn current_boundary_projection_does_not_execute_a_queued_intervention() {
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        initial_food: 0.0,
        food_max: 1.0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        food_respawn_interval: 0,
        closed: true,
        population_minimum: 0,
        population_spawn_interval: 0,
        reproduction_attempt_interval: 0,
        reproduction_attempt_chance: 0.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        rng_seed: Some(88),
        ..ScriptBotsConfig::default()
    })
    .expect("world");
    let observer = world
        .try_spawn_agent(AgentData {
            position: Position::new(100.0, 100.0),
            health: 1.0,
            ..AgentData::default()
        })
        .expect("observer");
    world
        .enqueue_intervention(Intervention::Bloom {
            region: Region::All,
            amount: 0.25,
        })
        .expect("queue bloom");

    let attribution = world.explain_sensors(observer, 0).expect("observer exists");
    world.step().expect("step");
    let sensed = world.agent_runtime(observer).expect("runtime").sensors;
    let diverged = sensor_differences(&attribution.clamped, &sensed);

    assert_eq!(
        diverged.len(),
        1,
        "only the queued Bloom should separate this completed-boundary projection from the next \
         runtime vector: {diverged:?}"
    );
    assert_eq!(
        diverged[0].0, FOOD,
        "Bloom must separate only the food channel: {diverged:?}"
    );
    assert_eq!(attribution.clamped[FOOD].to_bits(), 0.0_f32.to_bits());
    assert_eq!(sensed[FOOD].to_bits(), 0.25_f32.to_bits());
}

/// Cadence aging also precedes Sense and is intentionally outside the
/// instantaneous completed-boundary projection.
#[test]
fn current_boundary_projection_does_not_forecast_cadence_aging() {
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        initial_food: 0.0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        food_respawn_interval: 0,
        aging_tick_interval: 1,
        aging_health_decay_start: 0,
        aging_health_decay_rate: 0.2,
        aging_health_decay_max: 0.2,
        aging_energy_penalty_rate: 0.0,
        closed: true,
        population_minimum: 0,
        population_spawn_interval: 0,
        reproduction_attempt_interval: 0,
        reproduction_attempt_chance: 0.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        rng_seed: Some(99),
        ..ScriptBotsConfig::default()
    })
    .expect("world");
    let observer = world
        .try_spawn_agent(AgentData {
            position: Position::new(100.0, 100.0),
            health: 1.0,
            ..AgentData::default()
        })
        .expect("observer");

    let attribution = world.explain_sensors(observer, 0).expect("observer exists");
    world.step().expect("step");
    let sensed = world.agent_runtime(observer).expect("runtime").sensors;
    let diverged = sensor_differences(&attribution.clamped, &sensed);

    assert_eq!(
        diverged.len(),
        1,
        "only Aging should separate this completed-boundary projection from the next runtime \
         vector: {diverged:?}"
    );
    assert_eq!(
        diverged[0].0, HEALTH,
        "Aging must separate only the health channel: {diverged:?}"
    );
    assert_eq!(attribution.clamped[HEALTH].to_bits(), 0.5_f32.to_bits());
    assert_eq!(sensed[HEALTH].to_bits(), 0.4_f32.to_bits());
}
