// bd-tqpj: deterministic-simulation policy — pinned floating-point evaluation
// order and fixed-width casts are part of the science contract; fma fusion,
// reassociation, or width changes alter world digests. Function lengths mirror
// the legacy C++ parity layout and are reviewed as units.
#![allow(clippy::suboptimal_flops, clippy::imprecise_flops)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
#![allow(clippy::float_cmp, clippy::while_float)]
#![allow(clippy::too_many_lines)]

use scriptbots_core::{
    AgentData, AgentId, BrainRunner, FoodCellProfileSnapshot, INPUT_SIZE, LocomotionModel,
    NUM_EYES, NullPersistence, OUTPUT_SIZE, OutputChannel, Position, SENSOR_LAYOUT,
    ScriptBotsConfig, SensorKind, Tick, TickSummary, TraitModifiers, WorldState,
};

#[derive(Debug, Clone, Copy)]
enum OracleContract {
    LegacyParity,
    DeliberatePolicy(&'static str),
}

impl OracleContract {
    const fn label(self) -> &'static str {
        match self {
            Self::LegacyParity => "legacy-parity",
            Self::DeliberatePolicy(_) => "deliberate-rust-policy",
        }
    }

    const fn rationale(self) -> &'static str {
        match self {
            Self::LegacyParity => "the Rust port claims to preserve this legacy mechanic",
            Self::DeliberatePolicy(rationale) => rationale,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NumericExpectation {
    quantity: &'static str,
    expected: f32,
    absolute_tolerance: f32,
}

#[derive(Debug)]
struct LegacyOracleCase<'a> {
    name: &'static str,
    seed: u64,
    deterministic_setup: &'static str,
    original_file: &'static str,
    original_lines: (u32, u32),
    contract: OracleContract,
    expectations: &'a [NumericExpectation],
}

impl LegacyOracleCase<'_> {
    fn assert_close(&self, quantity: &str, actual: f32) {
        assert!(!self.name.is_empty(), "oracle case name must not be empty");
        assert!(
            !self.deterministic_setup.is_empty(),
            "oracle '{}' must describe its deterministic setup",
            self.name
        );
        assert!(
            !self.original_file.is_empty() && self.original_lines.0 <= self.original_lines.1,
            "oracle '{}' must cite a valid legacy source range",
            self.name
        );

        let expectation = self
            .expectations
            .iter()
            .find(|expectation| expectation.quantity == quantity);
        assert!(
            expectation.is_some(),
            "oracle case '{}' has no numeric expectation for '{quantity}'",
            self.name
        );
        let expectation = expectation.expect("oracle expectation existence asserted above");
        assert!(
            expectation.expected.is_finite()
                && expectation.absolute_tolerance.is_finite()
                && expectation.absolute_tolerance >= 0.0,
            "oracle '{}' has an invalid numeric expectation for '{}'",
            self.name,
            expectation.quantity
        );

        let absolute_error = (actual - expectation.expected).abs();
        assert!(
            actual.is_finite() && absolute_error <= expectation.absolute_tolerance,
            "oracle '{}' failed: quantity='{}', expected={} +/- {}, actual={}, \
             absolute_error={}, contract={}, rationale='{}', seed={}, setup='{}', source={}:{}-{}",
            self.name,
            expectation.quantity,
            expectation.expected,
            expectation.absolute_tolerance,
            actual,
            absolute_error,
            self.contract.label(),
            self.contract.rationale(),
            self.seed,
            self.deterministic_setup,
            self.original_file,
            self.original_lines.0,
            self.original_lines.1,
        );
    }
}

struct ZeroBrain;

impl BrainRunner for ZeroBrain {
    fn kind(&self) -> &'static str {
        "test.oracle.zero"
    }

    fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        [0.0; OUTPUT_SIZE]
    }
}

#[derive(Clone, Copy)]
struct FixedOutputsBrain {
    outputs: [f32; OUTPUT_SIZE],
}

impl BrainRunner for FixedOutputsBrain {
    fn kind(&self) -> &'static str {
        "test.oracle.fixed-outputs"
    }

    fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        self.outputs
    }
}

fn bind_zero_brain(world: &mut WorldState, agents: &[AgentId]) {
    let key = world
        .brain_registry_mut()
        .expect("zero-brain registry mutation")
        .register("test.oracle.zero", |_rng| Ok(Box::new(ZeroBrain)));

    for &agent in agents {
        assert!(
            world
                .bind_agent_brain(agent, key)
                .expect("zero-brain factory"),
            "oracle agent should accept the deterministic zero brain"
        );
    }
}

fn bind_fixed_outputs_brain(
    world: &mut WorldState,
    agent: AgentId,
    kind: &'static str,
    outputs: [f32; OUTPUT_SIZE],
) {
    let key = world
        .brain_registry_mut()
        .expect("fixed-output registry mutation")
        .register(kind, move |_rng| {
            Ok(Box::new(FixedOutputsBrain { outputs }))
        });
    assert!(
        world
            .bind_agent_brain(agent, key)
            .expect("fixed-output brain factory"),
        "oracle agent should accept its deterministic fixed-output brain"
    );
}

fn wheel_outputs(left: f32, right: f32) -> [f32; OUTPUT_SIZE] {
    let mut outputs = [0.0; OUTPUT_SIZE];
    outputs[OutputChannel::WheelLeft.index()] = left;
    outputs[OutputChannel::WheelRight.index()] = right;
    outputs
}

fn quiet_locomotion_config(model: LocomotionModel, seed: u64) -> ScriptBotsConfig {
    ScriptBotsConfig {
        world_width: 400,
        world_height: 400,
        food_cell_size: 20,
        initial_food: 0.0,
        food_respawn_interval: 0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        sense_radius: 100.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        metabolism_ramp_rate: 0.0,
        metabolism_boost_penalty: 0.0,
        temperature_discomfort_rate: 0.0,
        food_intake_rate: 0.0,
        food_waste_rate: 0.0,
        reproduction_energy_threshold: 10.0,
        population_minimum: 0,
        population_spawn_interval: 0,
        persistence_interval: 0,
        topography_enabled: false,
        locomotion_model: model,
        rng_seed: Some(seed),
        ..ScriptBotsConfig::default()
    }
}

fn sound_sensor_index() -> usize {
    SENSOR_LAYOUT
        .iter()
        .find(|channel| channel.kind == SensorKind::Sound)
        .expect("canonical movement-noise sensor")
        .index
}

fn velocity_magnitude(world: &WorldState, agent: AgentId) -> f32 {
    let velocity = world
        .snapshot_agent(agent)
        .expect("oracle agent should remain alive")
        .data
        .velocity;
    (velocity.vx * velocity.vx + velocity.vy * velocity.vy).sqrt()
}

fn reposition_agent(world: &mut WorldState, agent: AgentId, position: Position) {
    assert!(
        world
            .try_update_agent(agent, |data, _runtime| {
                data.position = position;
                data.heading = 0.0;
            })
            .expect("finite oracle position update"),
        "oracle agent should still exist"
    );
}

fn default_profile(config: &ScriptBotsConfig) -> FoodCellProfileSnapshot {
    FoodCellProfileSnapshot {
        capacity: config.food_max,
        growth_multiplier: 1.0,
        decay_multiplier: 1.0,
        fertility: 0.0,
        nutrient_density: 0.3,
    }
}

fn expected_food_value(world: &WorldState, before: &[f32], x: u32, y: u32) -> f32 {
    let width = world.food().width() as usize;
    let height = world.food().height() as usize;
    let config = world.config();
    let profile = world
        .food_profile(x, y)
        .unwrap_or_else(|| default_profile(config));

    let diffusion = config.food_diffusion_rate;
    let decay = config.food_decay_rate;
    let growth = config.food_growth_rate;

    let idx = y as usize * width + x as usize;
    let previous = before[idx];
    let mut value = previous;

    if diffusion > 0.0 {
        let x_usize = x as usize;
        let y_usize = y as usize;
        let left = if x_usize == 0 { width - 1 } else { x_usize - 1 };
        let right = if x_usize + 1 == width { 0 } else { x_usize + 1 };
        let up = if y_usize == 0 {
            height - 1
        } else {
            y_usize - 1
        };
        let down = if y_usize + 1 == height {
            0
        } else {
            y_usize + 1
        };
        let neighbor_avg = (before[y_usize * width + left]
            + before[y_usize * width + right]
            + before[up * width + x_usize]
            + before[down * width + x_usize])
            * 0.25;
        value += diffusion * (neighbor_avg - previous);
    }

    if decay > 0.0 {
        value -= decay * profile.decay_multiplier * value;
    }

    if growth > 0.0 && config.food_max > 0.0 {
        let normalized = value / config.food_max;
        let growth_delta = growth * profile.growth_multiplier * (1.0 - normalized);
        value += growth_delta * config.food_max;
    }

    let mut cap = profile.capacity.max(previous);
    let global_cap = config.food_max.max(previous);
    if cap > global_cap {
        cap = global_cap;
    }

    value.clamp(0.0, cap.max(0.0))
}

#[test]
fn seeded_world_advances_deterministically() {
    let config = ScriptBotsConfig {
        world_width: 256,
        world_height: 256,
        food_cell_size: 16,
        initial_food: 0.25,
        food_max: 1.0,
        rng_seed: Some(0xDEAD_BEEF),
        ..ScriptBotsConfig::default()
    };

    let mut world_a = WorldState::new(config.clone()).expect("world_a");
    let mut world_b = WorldState::new(config).expect("world_b");

    let agent = AgentData::default();
    let id_a = world_a
        .try_spawn_agent(agent)
        .expect("default agent is finite");
    let agent = AgentData::default();
    let id_b = world_b
        .try_spawn_agent(agent)
        .expect("default agent is finite");

    for _ in 0..8 {
        world_a
            .step()
            .expect("first deterministic world should accept each simulation step");
        world_b
            .step()
            .expect("second deterministic world should accept each simulation step");
    }

    assert_eq!(world_a.tick(), Tick(8));
    assert_eq!(world_b.tick(), Tick(8));
    assert_eq!(world_a.agent_count(), 1);
    assert_eq!(world_b.agent_count(), 1);

    let runtime_a = world_a.agent_runtime(id_a).expect("runtime_a");
    let runtime_b = world_b.agent_runtime(id_b).expect("runtime_b");
    assert!(runtime_a.energy.is_finite());
    assert!(runtime_b.energy.is_finite());
    assert_eq!(runtime_a.outputs, runtime_b.outputs);
}

#[test]
fn registry_executes_custom_brain() {
    #[derive(Clone)]
    struct ConstantBrain {
        value: f32,
    }

    impl BrainRunner for ConstantBrain {
        fn kind(&self) -> &'static str {
            "test.constant"
        }

        fn tick(
            &mut self,
            _inputs: &[f32; scriptbots_core::INPUT_SIZE],
        ) -> [f32; scriptbots_core::OUTPUT_SIZE] {
            [self.value; scriptbots_core::OUTPUT_SIZE]
        }
    }

    let config = ScriptBotsConfig {
        world_width: 128,
        world_height: 128,
        food_cell_size: 16,
        initial_food: 0.25,
        food_max: 1.0,
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("world");

    let key = world
        .brain_registry_mut()
        .expect("constant-brain registry mutation")
        .register("test.constant", |_rng| {
            Ok(Box::new(ConstantBrain { value: 0.75 }))
        });

    let agent_id = world
        .try_spawn_agent(AgentData::default())
        .expect("default agent is finite");
    assert!(
        world
            .bind_agent_brain(agent_id, key)
            .expect("constant-brain factory")
    );

    world
        .step()
        .expect("custom-brain world should accept its simulation step");
    let runtime = world.agent_runtime(agent_id).expect("runtime");
    assert!(
        runtime
            .outputs
            .iter()
            .all(|v| (*v - 0.75).abs() < f32::EPSILON)
    );
}

#[test]
fn combat_records_carnivore_event_flags() {
    #[derive(Clone)]
    struct SpikeBrain;

    impl BrainRunner for SpikeBrain {
        fn kind(&self) -> &'static str {
            "test.spike"
        }

        fn tick(
            &mut self,
            _inputs: &[f32; scriptbots_core::INPUT_SIZE],
        ) -> [f32; scriptbots_core::OUTPUT_SIZE] {
            let mut outputs = [0.0; scriptbots_core::OUTPUT_SIZE];
            outputs[0] = 1.0;
            outputs[5] = 1.0;
            outputs
        }
    }

    let config = ScriptBotsConfig {
        world_width: 240,
        world_height: 240,
        food_cell_size: 40,
        initial_food: 0.2,
        food_max: 1.0,
        spike_damage: 0.5,
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("world");

    let attacker = AgentData {
        position: Position::new(80.0, 80.0),
        heading: 0.0,
        spike_length: 4.0,
        ..AgentData::default()
    };

    let victim = AgentData {
        position: Position::new(95.0, 80.0),
        heading: std::f32::consts::PI,
        ..AgentData::default()
    };

    let attacker_id = world.try_spawn_agent(attacker).expect("attacker is finite");
    let victim_id = world.try_spawn_agent(victim).expect("victim is finite");

    let spike_key = world
        .brain_registry_mut()
        .expect("spike-brain registry mutation")
        .register("test.spike", |_rng| Ok(Box::new(SpikeBrain)));
    assert!(
        world
            .bind_agent_brain(attacker_id, spike_key)
            .expect("spike-brain factory")
    );
    world
        .try_update_agent_runtime(attacker_id, |runtime| runtime.herbivore_tendency = 0.1)
        .expect("finite attacker update");
    world
        .try_update_agent_runtime(victim_id, |runtime| runtime.herbivore_tendency = 0.9)
        .expect("finite victim update");

    world
        .step()
        .expect("combat world should accept its simulation step");

    let attacker_runtime = world.agent_runtime(attacker_id).expect("attacker runtime");
    assert!(attacker_runtime.combat.spike_attacker);
    assert!(attacker_runtime.combat.hit_herbivore);
    assert!(!attacker_runtime.combat.hit_carnivore);

    if let Some(victim_snapshot) = world.snapshot_agent(victim_id) {
        assert!(
            victim_snapshot.data.health < 1.0,
            "victim health should drop after spike"
        );
        let victim_runtime = world.agent_runtime(victim_id).expect("victim runtime");
        assert!(victim_runtime.combat.spike_victim);
        assert!(victim_runtime.combat.was_spiked_by_carnivore);
        assert!(!victim_runtime.combat.was_spiked_by_herbivore);
    }
}

#[test]
fn movement_noise_distinguishes_wheel_effort_when_legacy_displacements_match() {
    const SEED: u64 = 0x501D_EFF0;
    const DISTANCE_FACTOR: f32 = 0.5;

    let mut world = WorldState::new(quiet_locomotion_config(LocomotionModel::Legacy, SEED))
        .expect("legacy wheel-effort world");
    world.set_closed(true).expect("close wheel-effort world");

    let observer = world
        .try_spawn_agent(AgentData {
            position: Position::new(200.0, 200.0),
            health: 2.0,
            ..AgentData::default()
        })
        .expect("observer is finite");
    let high_effort = world
        .try_spawn_agent(AgentData {
            position: Position::new(80.0, 80.0),
            health: 2.0,
            ..AgentData::default()
        })
        .expect("high-effort emitter is finite");
    let low_effort = world
        .try_spawn_agent(AgentData {
            position: Position::new(320.0, 320.0),
            health: 2.0,
            ..AgentData::default()
        })
        .expect("low-effort emitter is finite");

    bind_zero_brain(&mut world, &[observer]);
    bind_fixed_outputs_brain(
        &mut world,
        high_effort,
        "test.oracle.wheel-effort.high",
        wheel_outputs(0.0, 0.6),
    );
    bind_fixed_outputs_brain(
        &mut world,
        low_effort,
        "test.oracle.wheel-effort.low",
        wheel_outputs(0.3, 0.3),
    );
    world
        .try_update_agent_runtime(observer, |runtime| {
            runtime.trait_modifiers = TraitModifiers {
                smell: 0.0,
                sound: 1.0,
                hearing: 0.0,
                eye: 0.0,
                blood: 0.0,
            };
        })
        .expect("finite observer runtime update");

    world
        .step()
        .expect("wheel-effort world should produce one actuation");

    let high_displacement = velocity_magnitude(&world, high_effort);
    let low_displacement = velocity_magnitude(&world, low_effort);
    let obsolete_velocity_normalizer = world.config().bot_speed * world.config().boost_multiplier;
    assert!(
        (high_displacement - low_displacement).abs() < 0.002,
        "the fixture requires nearly equal physical displacement, got high={high_displacement}, low={low_displacement}"
    );
    assert!(
        high_displacement > 0.8 && low_displacement > 0.8,
        "the fixture must exercise substantial legacy displacement"
    );
    assert!(
        low_displacement > obsolete_velocity_normalizer,
        "the low-effort emitter must exceed the obsolete velocity normalization ceiling"
    );

    reposition_agent(&mut world, observer, Position::new(200.0, 200.0));
    reposition_agent(&mut world, high_effort, Position::new(250.0, 200.0));
    reposition_agent(&mut world, low_effort, Position::new(150.0, 200.0));

    let attribution = world
        .explain_sensors(observer, 8)
        .expect("movement-noise attribution");
    let high_contribution = attribution
        .contributions
        .iter()
        .find(|contribution| contribution.source == high_effort)
        .expect("high-effort contribution")
        .sound;
    let low_contribution = attribution
        .contributions
        .iter()
        .find(|contribution| contribution.source == low_effort)
        .expect("low-effort contribution")
        .sound;
    let expected_high = DISTANCE_FACTOR * 0.6;
    let expected_low = DISTANCE_FACTOR * 0.3;
    assert!(
        (high_contribution - expected_high).abs() <= 2.0e-6,
        "movement noise must follow peak wheel output, expected {expected_high}, got {high_contribution}"
    );
    assert!(
        (low_contribution - expected_low).abs() <= 2.0e-6,
        "movement noise must follow peak wheel output, expected {expected_low}, got {low_contribution}"
    );
    assert!(
        (high_contribution - low_contribution * 2.0).abs() <= 3.0e-6,
        "nearly equal displacement must not conflate 0.6 and 0.3 wheel effort"
    );
    assert!(
        low_contribution < 0.3,
        "large rotation-derived displacement must not saturate low wheel effort"
    );

    world
        .step()
        .expect("wheel-effort world should complete its sensing pass");
    let actual_total = world
        .agent_runtime(observer)
        .expect("observer survives")
        .sensors[sound_sensor_index()];
    assert!(
        (actual_total - (expected_high + expected_low)).abs() <= 3.0e-6,
        "production sensing must match the attributed wheel-effort total"
    );
}

fn movement_noise_for_model(model: LocomotionModel, seed: u64) -> (f32, f32) {
    let mut world =
        WorldState::new(quiet_locomotion_config(model, seed)).expect("model sound world");
    world.set_closed(true).expect("close model sound world");

    let observer = world
        .try_spawn_agent(AgentData {
            position: Position::new(200.0, 200.0),
            health: 2.0,
            ..AgentData::default()
        })
        .expect("observer is finite");
    let emitter = world
        .try_spawn_agent(AgentData {
            position: Position::new(80.0, 80.0),
            health: 2.0,
            ..AgentData::default()
        })
        .expect("emitter is finite");
    bind_zero_brain(&mut world, &[observer]);
    let kind = match model {
        LocomotionModel::Legacy => "test.oracle.model-sound.legacy",
        LocomotionModel::Differential => "test.oracle.model-sound.differential",
    };
    bind_fixed_outputs_brain(&mut world, emitter, kind, wheel_outputs(0.4, 0.4));
    world
        .try_update_agent_runtime(observer, |runtime| {
            runtime.trait_modifiers = TraitModifiers {
                smell: 0.0,
                sound: 1.0,
                hearing: 0.0,
                eye: 0.0,
                blood: 0.0,
            };
        })
        .expect("finite observer runtime update");

    world
        .step()
        .expect("model sound world should produce one actuation");
    let displacement = velocity_magnitude(&world, emitter);
    reposition_agent(&mut world, observer, Position::new(200.0, 200.0));
    reposition_agent(&mut world, emitter, Position::new(240.0, 200.0));
    world
        .step()
        .expect("model sound world should complete its sensing pass");
    let sound = world
        .agent_runtime(observer)
        .expect("observer survives")
        .sensors[sound_sensor_index()];
    (sound, displacement)
}

fn combat_damage_for_model(model: LocomotionModel, seed: u64) -> (f32, f32) {
    let mut config = quiet_locomotion_config(model, seed);
    config.spike_growth_rate = 0.0;
    config.spike_radius = 20.0;
    config.spike_damage = 0.25;
    config.spike_energy_cost = 0.0;
    config.spike_min_length = 0.1;
    config.spike_alignment_cosine = 0.99;
    config.spike_speed_damage_bonus = 0.6;
    config.spike_length_damage_bonus = 0.75;

    let mut world = WorldState::new(config).expect("model combat world");
    world.set_closed(true).expect("close model combat world");
    let attacker = world
        .try_spawn_agent(AgentData {
            position: Position::new(50.0, 50.0),
            heading: 0.0,
            health: 2.0,
            spike_length: 1.0,
            ..AgentData::default()
        })
        .expect("attacker is finite");
    let victim = world
        .try_spawn_agent(AgentData {
            position: Position::new(300.0, 300.0),
            heading: 0.0,
            health: 2.0,
            ..AgentData::default()
        })
        .expect("victim is finite");

    let mut attacker_outputs = wheel_outputs(0.4, 0.4);
    attacker_outputs[OutputChannel::SpikeTarget.index()] = 1.0;
    attacker_outputs[OutputChannel::Boost.index()] = 1.0;
    let kind = match model {
        LocomotionModel::Legacy => "test.oracle.model-combat.legacy",
        LocomotionModel::Differential => "test.oracle.model-combat.differential",
    };
    bind_fixed_outputs_brain(&mut world, attacker, kind, attacker_outputs);
    bind_zero_brain(&mut world, &[victim]);
    world
        .try_update_agent_runtime(attacker, |runtime| {
            runtime.herbivore_tendency = 0.0;
        })
        .expect("finite attacker runtime update");
    world
        .try_update_agent_runtime(victim, |runtime| {
            runtime.herbivore_tendency = 1.0;
        })
        .expect("finite victim runtime update");

    world
        .step()
        .expect("model combat world should produce one actuation");
    let displacement = world
        .snapshot_agent(attacker)
        .expect("attacker survives warmup")
        .data
        .velocity;
    let displacement_magnitude =
        (displacement.vx * displacement.vx + displacement.vy * displacement.vy).sqrt();

    let common_attacker_position = Position::new(200.0, 200.0);
    reposition_agent(
        &mut world,
        attacker,
        Position::new(
            common_attacker_position.x - displacement.vx,
            common_attacker_position.y - displacement.vy,
        ),
    );
    assert!(
        world
            .try_update_agent(victim, |data, _runtime| {
                data.position = Position::new(215.0, 200.0);
                data.heading = 0.0;
                data.health = 2.0;
            })
            .expect("finite victim reset"),
        "victim should still exist before the combat oracle"
    );

    world
        .step()
        .expect("model combat world should resolve the aligned hit");
    let victim_health = world
        .snapshot_agent(victim)
        .expect("the bounded oracle hit must not kill the victim")
        .data
        .health;
    (2.0 - victim_health, displacement_magnitude)
}

#[test]
fn identical_outputs_have_model_independent_sound_and_combat_scaling() {
    let (legacy_sound, legacy_sound_displacement) =
        movement_noise_for_model(LocomotionModel::Legacy, 0x501D_5A1E);
    let (differential_sound, differential_sound_displacement) =
        movement_noise_for_model(LocomotionModel::Differential, 0x501D_5A1E);
    assert!(
        legacy_sound_displacement > differential_sound_displacement * 5.0,
        "the sound oracle must compare materially different physical displacement"
    );
    assert!(
        (legacy_sound - differential_sound).abs() <= 2.0e-6,
        "identical wheel outputs must produce identical movement noise across locomotion models: legacy={legacy_sound}, differential={differential_sound}"
    );
    assert!(
        (legacy_sound - 0.24).abs() <= 2.0e-6,
        "0.4 wheel effort at distance factor 0.6 should produce 0.24 movement noise"
    );

    let (legacy_damage, legacy_combat_displacement) =
        combat_damage_for_model(LocomotionModel::Legacy, 0xC0AB_5A1E);
    let (differential_damage, differential_combat_displacement) =
        combat_damage_for_model(LocomotionModel::Differential, 0xC0AB_5A1E);
    assert!(
        legacy_combat_displacement > differential_combat_displacement * 5.0,
        "the combat oracle must compare materially different physical displacement"
    );
    assert!(
        legacy_damage > 0.0 && differential_damage > 0.0,
        "both locomotion models must resolve the aligned spike hit"
    );
    assert!(
        (legacy_damage - differential_damage).abs() <= 1.0e-4,
        "identical named wheel/spike/boost outputs must produce identical combat scaling without a second velocity bonus: legacy={legacy_damage}, differential={differential_damage}"
    );
    assert!(
        (legacy_damage - 0.98).abs() <= 1.0e-4,
        "0.25 base * 1.75 length * 2.24 wheel/boost factor should deal 0.98 damage"
    );
}

#[test]
fn legacy_eye_density_micro_oracle_single_neighbor() {
    const SEED: u64 = 0x51E5_EE01;
    let expectations = [NumericExpectation {
        quantity: "forward eye density",
        expected: 0.1875,
        absolute_tolerance: 1e-6,
    }];
    let oracle = LegacyOracleCase {
        name: "centered single-neighbor eye density",
        seed: SEED,
        deterministic_setup: "subject=(100,100), target=(125,100), heading=0, eye0_dir=0, eye_fov=pi/4, sense_radius=100, eye_modifier=1",
        original_file: "original_scriptbots_code_for_reference/World.cpp",
        original_lines: (241, 260),
        contract: OracleContract::LegacyParity,
        expectations: &expectations,
    };

    let config = ScriptBotsConfig {
        world_width: 400,
        world_height: 400,
        food_cell_size: 20,
        initial_food: 0.0,
        food_respawn_interval: 0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        sense_radius: 100.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        food_intake_rate: 0.0,
        food_waste_rate: 0.0,
        reproduction_energy_threshold: 0.0,
        rng_seed: Some(SEED),
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("legacy eye oracle world");
    world.set_closed(true).expect("close world");

    let subject = world
        .try_spawn_agent(AgentData {
            position: Position::new(100.0, 100.0),
            heading: 0.0,
            ..AgentData::default()
        })
        .expect("subject is finite");
    let target = world
        .try_spawn_agent(AgentData {
            position: Position::new(125.0, 100.0),
            heading: 0.0,
            health: 2.0,
            ..AgentData::default()
        })
        .expect("target is finite");
    bind_zero_brain(&mut world, &[subject, target]);

    world
        .try_update_agent_runtime(subject, |runtime| {
            runtime.trait_modifiers = TraitModifiers {
                smell: 0.0,
                sound: 0.0,
                hearing: 0.0,
                eye: 1.0,
                blood: 0.0,
            };
            runtime.eye_fov = [std::f32::consts::FRAC_PI_4; NUM_EYES];
            runtime.eye_direction = [
                0.0,
                std::f32::consts::FRAC_PI_2,
                std::f32::consts::PI,
                -std::f32::consts::FRAC_PI_2,
            ];
        })
        .expect("finite subject runtime update");

    world
        .step()
        .expect("legacy eye oracle should complete one simulation step");
    let density = world
        .agent_runtime(subject)
        .expect("subject runtime should survive the oracle step")
        .sensors[0];
    oracle.assert_close("forward eye density", density);
}

#[test]
fn legacy_eye_chunk_boundary_oracle_is_feature_invariant() {
    const SEED: u64 = 0x51E5_C84F;
    const SENSE_RADIUS: f32 = 100.0;
    const GRID_CELL_SIZE: u32 = 200;
    const GRID_CELL_SIZE_F32: f32 = 200.0;
    const SUBJECT_HEADING: f32 = 0.35;
    const EYE_DIRECTION: f32 = 0.25;
    const EYE_FOV: f32 = 0.75;
    const EYE_SENSITIVITY: f32 = 0.125;
    const SUBJECT_POSITION: Position = Position::new(100.0, 100.0);

    #[derive(Clone, Copy)]
    struct VisibleTarget {
        distance: f32,
        angle_offset: f32,
        color: [f32; 3],
    }

    // The subject is inserted before each tested prefix of these targets. Every
    // agent occupies one grid bucket, so 7/8/9 logical neighbors exercise the
    // 4n-1, 4n, and 4n+1 boundaries around two four-lane SIMD chunks. The
    // eight-neighbor case has one remainder target after the subject-inclusive
    // chunks, so the old inverted SIMD falloff cannot pass from that remainder.
    const TARGETS: [VisibleTarget; 9] = [
        VisibleTarget {
            distance: 18.0,
            angle_offset: -0.30,
            color: [1.0, 0.0, 0.0],
        },
        VisibleTarget {
            distance: 23.0,
            angle_offset: -0.20,
            color: [0.0, 1.0, 0.0],
        },
        VisibleTarget {
            distance: 29.0,
            angle_offset: -0.10,
            color: [0.0, 0.0, 1.0],
        },
        VisibleTarget {
            distance: 34.0,
            angle_offset: -0.03,
            color: [1.0, 1.0, 0.0],
        },
        VisibleTarget {
            distance: 39.0,
            angle_offset: 0.04,
            color: [0.0, 1.0, 1.0],
        },
        VisibleTarget {
            distance: 44.0,
            angle_offset: 0.12,
            color: [1.0, 0.0, 1.0],
        },
        VisibleTarget {
            distance: 49.0,
            angle_offset: 0.22,
            color: [0.25, 0.5, 0.75],
        },
        VisibleTarget {
            distance: 54.0,
            angle_offset: 0.32,
            color: [0.8, 0.2, 0.4],
        },
        VisibleTarget {
            distance: 58.0,
            angle_offset: -0.27,
            color: [0.4, 0.9, 0.1],
        },
    ];

    for visible_count in [7, 8, 9] {
        let targets = &TARGETS[..visible_count];
        let config = ScriptBotsConfig {
            world_width: 400,
            world_height: 400,
            food_cell_size: GRID_CELL_SIZE,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_growth_rate: 0.0,
            food_decay_rate: 0.0,
            food_diffusion_rate: 0.0,
            sense_radius: SENSE_RADIUS,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            food_intake_rate: 0.0,
            food_waste_rate: 0.0,
            reproduction_energy_threshold: 10.0,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            rng_seed: Some(SEED),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("legacy eye chunk-boundary oracle world");
        world.set_closed(true).expect("close world");

        let subject = world
            .try_spawn_agent(AgentData {
                position: SUBJECT_POSITION,
                heading: SUBJECT_HEADING,
                health: 2.0,
                ..AgentData::default()
            })
            .expect("eye-oracle subject is finite");
        let view_angle = SUBJECT_HEADING + EYE_DIRECTION;
        let mut target_ids = Vec::with_capacity(targets.len());
        for &target in targets {
            let angle = view_angle + target.angle_offset;
            let position = Position::new(
                SUBJECT_POSITION.x + target.distance * angle.cos(),
                SUBJECT_POSITION.y + target.distance * angle.sin(),
            );
            assert_eq!(
                (
                    (position.x / GRID_CELL_SIZE_F32).floor(),
                    (position.y / GRID_CELL_SIZE_F32).floor(),
                ),
                (
                    (SUBJECT_POSITION.x / GRID_CELL_SIZE_F32).floor(),
                    (SUBJECT_POSITION.y / GRID_CELL_SIZE_F32).floor(),
                ),
                "the fixture must keep every target in the subject's bucket",
            );
            target_ids.push(
                world
                    .try_spawn_agent(AgentData {
                        position,
                        heading: 0.0,
                        color: target.color,
                        health: 2.0,
                        ..AgentData::default()
                    })
                    .expect("eye-oracle target is finite"),
            );
        }
        let mut all_ids = Vec::with_capacity(targets.len() + 1);
        all_ids.push(subject);
        all_ids.extend_from_slice(&target_ids);
        bind_zero_brain(&mut world, &all_ids);

        world
            .try_update_agent_runtime(subject, |runtime| {
                runtime.trait_modifiers = TraitModifiers {
                    smell: 0.0,
                    sound: 0.0,
                    hearing: 0.0,
                    eye: EYE_SENSITIVITY,
                    blood: 0.0,
                };
                runtime.eye_fov = [EYE_FOV; NUM_EYES];
                runtime.eye_direction = [
                    EYE_DIRECTION,
                    EYE_DIRECTION + std::f32::consts::FRAC_PI_2,
                    EYE_DIRECTION + std::f32::consts::PI,
                    EYE_DIRECTION - std::f32::consts::FRAC_PI_2,
                ];
            })
            .expect("finite eye-oracle runtime update");

        // This is the independent legacy World.cpp:241-259 oracle: angular and
        // distance falloff contribute to every eye channel, while density alone
        // carries the additional distance/radius factor.
        let mut expected = [0.0_f32; 4];
        for &target in targets {
            let angular_factor = (EYE_FOV - target.angle_offset.abs()) / EYE_FOV;
            let distance_factor = (SENSE_RADIUS - target.distance) / SENSE_RADIUS;
            let intensity = EYE_SENSITIVITY * angular_factor * distance_factor;
            expected[0] += intensity * (target.distance / SENSE_RADIUS);
            expected[1] += intensity * target.color[0];
            expected[2] += intensity * target.color[1];
            expected[3] += intensity * target.color[2];
        }
        assert!(
            expected.iter().all(|value| *value < 1.0),
            "oracle values must remain below sensor clamping"
        );

        world
            .step()
            .expect("legacy eye chunk-boundary oracle should complete one step");
        let sensors = world
            .agent_runtime(subject)
            .expect("eye-oracle subject should survive")
            .sensors;
        let expected_kinds = [
            (SensorKind::EyeDensity, expected[0]),
            (SensorKind::EyeRed, expected[1]),
            (SensorKind::EyeGreen, expected[2]),
            (SensorKind::EyeBlue, expected[3]),
        ];
        for (kind, expected_value) in expected_kinds {
            let channel = SENSOR_LAYOUT
                .iter()
                .find(|channel| channel.eye == Some(0) && channel.kind == kind)
                .expect("canonical eye channel");
            let actual = sensors[channel.index];
            assert!(
                (actual - expected_value).abs() <= 2.0e-6,
                "{} differs from the {visible_count}-neighbor legacy analytic oracle: expected {expected_value}, got {actual}",
                channel.name,
            );
        }
        for channel in SENSOR_LAYOUT
            .iter()
            .filter(|channel| channel.eye.is_some_and(|eye| eye != 0))
        {
            assert_eq!(
                sensors[channel.index], 0.0,
                "{} must not see the forward-eye target cohort",
                channel.name,
            );
        }
    }
}

fn fixed_seed_blood_sensor_reading(seed: u64, target_angle: f32, target_health: f32) -> f32 {
    let config = ScriptBotsConfig {
        world_width: 400,
        world_height: 400,
        food_cell_size: 20,
        initial_food: 0.0,
        food_respawn_interval: 0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        sense_radius: 100.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        food_intake_rate: 0.0,
        food_waste_rate: 0.0,
        reproduction_energy_threshold: 10.0,
        population_minimum: 0,
        population_spawn_interval: 0,
        persistence_interval: 0,
        rng_seed: Some(seed),
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("legacy blood oracle world");
    world.set_closed(true).expect("close world");

    let subject_position = Position::new(200.0, 200.0);
    let target_distance = 40.0;
    let subject = world
        .try_spawn_agent(AgentData {
            position: subject_position,
            heading: 0.0,
            health: 2.0,
            ..AgentData::default()
        })
        .expect("subject is finite");
    let target = world
        .try_spawn_agent(AgentData {
            position: Position::new(
                subject_position.x + target_distance * target_angle.cos(),
                subject_position.y + target_distance * target_angle.sin(),
            ),
            heading: 0.0,
            health: target_health,
            ..AgentData::default()
        })
        .expect("target is finite");
    bind_zero_brain(&mut world, &[subject, target]);

    world
        .try_update_agent_runtime(subject, |runtime| {
            runtime.trait_modifiers = TraitModifiers {
                smell: 0.0,
                sound: 0.0,
                hearing: 0.0,
                eye: 0.0,
                blood: 1.0,
            };
        })
        .expect("finite blood-oracle runtime update");

    world
        .step()
        .expect("legacy blood oracle should complete one simulation step");
    world
        .agent_runtime(subject)
        .expect("blood-oracle subject should survive")
        .sensors[19]
}

#[test]
fn legacy_blood_sensor_fov_boundaries_and_wound_scaling_are_deterministic() {
    const SEED: u64 = 0xB100_DF0B;
    const LEGACY_HALF_FOV: f32 = std::f32::consts::PI * 3.0 / 16.0;
    const ANGLE_DELTA: f32 = 1.0e-3;
    const DISTANCE_FACTOR: f32 = 0.6;
    const HALF_WOUND_FACTOR: f32 = 0.5;

    let first = [
        fixed_seed_blood_sensor_reading(SEED, 0.0, 1.0),
        fixed_seed_blood_sensor_reading(SEED, LEGACY_HALF_FOV - ANGLE_DELTA, 1.0),
        fixed_seed_blood_sensor_reading(SEED, LEGACY_HALF_FOV, 1.0),
        fixed_seed_blood_sensor_reading(SEED, LEGACY_HALF_FOV + ANGLE_DELTA, 1.0),
        fixed_seed_blood_sensor_reading(SEED, 0.0, 2.0),
    ];
    let second = [
        fixed_seed_blood_sensor_reading(SEED, 0.0, 1.0),
        fixed_seed_blood_sensor_reading(SEED, LEGACY_HALF_FOV - ANGLE_DELTA, 1.0),
        fixed_seed_blood_sensor_reading(SEED, LEGACY_HALF_FOV, 1.0),
        fixed_seed_blood_sensor_reading(SEED, LEGACY_HALF_FOV + ANGLE_DELTA, 1.0),
        fixed_seed_blood_sensor_reading(SEED, 0.0, 2.0),
    ];
    assert_eq!(first, second, "fixed-seed blood sensing must be repeatable");

    let expected_just_inside =
        (ANGLE_DELTA / LEGACY_HALF_FOV) * DISTANCE_FACTOR * HALF_WOUND_FACTOR;
    let expectations = [
        NumericExpectation {
            quantity: "centered half-wounded target",
            expected: DISTANCE_FACTOR * HALF_WOUND_FACTOR,
            absolute_tolerance: 1.0e-6,
        },
        NumericExpectation {
            quantity: "target just inside 3pi/16",
            expected: expected_just_inside,
            absolute_tolerance: 1.0e-6,
        },
        NumericExpectation {
            quantity: "target on 3pi/16 boundary",
            expected: 0.0,
            absolute_tolerance: 1.0e-6,
        },
        NumericExpectation {
            quantity: "target just outside 3pi/16",
            expected: 0.0,
            absolute_tolerance: 1.0e-6,
        },
        NumericExpectation {
            quantity: "centered healthy target",
            expected: 0.0,
            absolute_tolerance: 1.0e-6,
        },
    ];
    let oracle = LegacyOracleCase {
        name: "blood sensor 3pi/16 half-FOV and wound scaling",
        seed: SEED,
        deterministic_setup: "heading-zero subject at (200,200), target distance 40, radius 100, blood modifier 1",
        original_file: "original_scriptbots_code_for_reference/World.cpp",
        original_lines: (193, 271),
        contract: OracleContract::LegacyParity,
        expectations: &expectations,
    };

    for (quantity, actual) in [
        ("centered half-wounded target", first[0]),
        ("target just inside 3pi/16", first[1]),
        ("target on 3pi/16 boundary", first[2]),
        ("target just outside 3pi/16", first[3]),
        ("centered healthy target", first[4]),
    ] {
        oracle.assert_close(quantity, actual);
    }
}

#[test]
fn sensory_pipeline_populates_expected_channels() {
    let config = ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        initial_food: 0.0,
        food_respawn_interval: 0,
        rng_seed: Some(42),
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        ..ScriptBotsConfig::default()
    };

    let mut world = WorldState::new(config).expect("world");
    let subject = world
        .try_spawn_agent(AgentData::default())
        .expect("subject is finite");
    let neighbor = world
        .try_spawn_agent(AgentData::default())
        .expect("neighbor is finite");

    world
        .try_update_agent(subject, |data, _runtime| {
            data.position = Position::new(80.0, 100.0);
            data.heading = 0.0;
        })
        .expect("finite subject state update");
    world
        .try_update_agent(neighbor, |data, _runtime| {
            data.position = Position::new(120.0, 100.0);
            data.heading = 0.0;
            data.color = [1.0, 0.2, 0.2];
            data.health = 0.4;
        })
        .expect("finite neighbor state update");

    let food_max = world.config().food_max;
    let food_index = 5 * world.food().width() as usize + 4;
    world
        .try_update_food(|cells| cells[food_index] = food_max * 0.8)
        .expect("finite food update");

    world
        .try_update_agent_runtime(subject, |runtime| {
            runtime.trait_modifiers = TraitModifiers {
                smell: 1.0,
                sound: 1.0,
                hearing: 1.0,
                eye: 1.0,
                blood: 1.0,
            };
            runtime.eye_fov = [1.2; NUM_EYES];
            runtime.eye_direction = [
                0.0,
                std::f32::consts::FRAC_PI_2,
                std::f32::consts::PI,
                -std::f32::consts::FRAC_PI_2,
            ];
            runtime.temperature_preference = 0.2;
        })
        .expect("finite subject runtime update");

    world
        .try_update_agent_runtime(neighbor, |runtime| runtime.sound_multiplier = 0.9)
        .expect("finite neighbor runtime update");

    world
        .step()
        .expect("sensory world should accept its simulation step");

    let sensors = world
        .agent_runtime(subject)
        .expect("subject runtime")
        .sensors;

    assert!(sensors[0] > 0.0, "forward eye intensity should register");
    assert!(
        sensors[1] > 0.0 && sensors[1] <= 1.0,
        "forward eye red channel populated"
    );
    let food_sensor = sensors[4];
    assert!(
        (food_sensor - 0.8).abs() < 1e-3,
        "local food sensor reflects configured cell (value={food_sensor})"
    );
    assert!(sensors[10] > 0.6, "smell sensor should react to neighbor");
    assert!(
        sensors[18] > 0.0,
        "hearing sensor should pick up neighbor sound"
    );
    assert!(
        sensors[19] > 0.0,
        "blood sensor should detect wounded neighbor"
    );
    assert!(
        (0.0..=1.0).contains(&sensors[16]),
        "clock sensor within bounds"
    );
    assert!(
        (0.0..=1.0).contains(&sensors[20]),
        "temperature discomfort normalized"
    );
    assert!(
        sensors[20] <= 0.1,
        "temperature discomfort low when preference matches"
    );
}

#[test]
fn ground_food_micro_oracle_documents_energy_policy() {
    const SEED: u64 = 0xF00D_0001;
    let expectations = [
        NumericExpectation {
            quantity: "cell fertility",
            expected: 0.0,
            absolute_tolerance: 1e-6,
        },
        NumericExpectation {
            quantity: "nutrient density",
            expected: 0.3,
            absolute_tolerance: 1e-6,
        },
        NumericExpectation {
            quantity: "agent energy",
            expected: 0.5013,
            absolute_tolerance: 1e-6,
        },
        NumericExpectation {
            quantity: "food balance total",
            expected: 0.0013,
            absolute_tolerance: 1e-6,
        },
        NumericExpectation {
            quantity: "reproduction progress",
            expected: 0.006,
            absolute_tolerance: 1e-6,
        },
        NumericExpectation {
            quantity: "cell food",
            expected: 0.199,
            absolute_tolerance: 1e-6,
        },
        NumericExpectation {
            quantity: "agent health",
            expected: 1.0,
            absolute_tolerance: 1e-6,
        },
    ];
    let oracle = LegacyOracleCase {
        name: "stationary herbivore ground-food policy",
        seed: SEED,
        deterministic_setup: "agent=(5,5), cell=(0,0), food=0.2, energy=0.5, herbivore=1, wheels=0, intake=0.002, waste=0.001, nutrient=0.3",
        original_file: "original_scriptbots_code_for_reference/World.cpp",
        original_lines: (381, 395),
        contract: OracleContract::DeliberatePolicy(
            "Rust converts nutrient-weighted intake into energy and positive reproduction progress; legacy C++ changed health and decremented a countdown",
        ),
        expectations: &expectations,
    };

    let config = ScriptBotsConfig {
        world_width: 40,
        world_height: 40,
        food_cell_size: 10,
        initial_food: 0.0,
        food_respawn_interval: 0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        sense_radius: 5.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        food_intake_rate: 0.002,
        food_waste_rate: 0.001,
        food_fertility_base: 0.0,
        food_moisture_weight: 0.0,
        food_elevation_weight: 0.0,
        food_slope_weight: 0.0,
        reproduction_energy_threshold: 0.0,
        reproduction_food_bonus: 3.0,
        reproduction_fertility_bonus: 0.5,
        rng_seed: Some(SEED),
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("ground-food oracle world");
    world.set_closed(true).expect("close world");

    let agent = world
        .try_spawn_agent(AgentData {
            position: Position::new(5.0, 5.0),
            health: 1.0,
            ..AgentData::default()
        })
        .expect("agent is finite");
    bind_zero_brain(&mut world, &[agent]);
    world
        .try_update_agent_runtime(agent, |runtime| {
            runtime.energy = 0.5;
            runtime.herbivore_tendency = 1.0;
            runtime.reproduction_counter = 0.0;
            runtime.food_balance_total = 0.0;
        })
        .expect("finite herbivore runtime update");

    let profile = world
        .food_profile(0, 0)
        .expect("oracle food cell should have a generated profile");
    oracle.assert_close("cell fertility", profile.fertility);
    oracle.assert_close("nutrient density", profile.nutrient_density);
    world
        .try_update_food(|cells| cells[0] = 0.2)
        .expect("finite food update");

    world
        .step()
        .expect("ground-food oracle should complete one simulation step");
    let runtime = world
        .agent_runtime(agent)
        .expect("oracle herbivore should survive the step");
    oracle.assert_close("agent energy", runtime.energy);
    oracle.assert_close("food balance total", runtime.food_balance_total);
    oracle.assert_close("reproduction progress", runtime.reproduction_counter);
    oracle.assert_close(
        "cell food",
        world
            .food()
            .get(0, 0)
            .expect("oracle food cell should remain addressable"),
    );
    oracle.assert_close(
        "agent health",
        world
            .snapshot_agent(agent)
            .expect("oracle herbivore snapshot should exist")
            .data
            .health,
    );
}

#[test]
fn food_growth_moves_toward_capacity() {
    let config = ScriptBotsConfig {
        world_width: 64,
        world_height: 64,
        food_cell_size: 32,
        initial_food: 0.0,
        food_respawn_interval: 0,
        food_growth_rate: 0.1,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        rng_seed: Some(1234),
        ..ScriptBotsConfig::default()
    };

    let mut world = WorldState::new(config).expect("world");
    let before = world.food().cells().to_vec();
    world
        .step()
        .expect("food-growth world should accept its simulation step");

    let width = world.food().width() as usize;
    let height = world.food().height() as usize;
    let cells = world.food().cells();
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let value = cells[idx];
            let expected = expected_food_value(&world, &before, x as u32, y as u32);
            assert!(
                (value - expected).abs() < 1e-6,
                "cell=({x},{y}) value={value} expected={expected}"
            );
        }
    }
}

#[test]
fn food_diffusion_spreads_across_neighbors() {
    let config = ScriptBotsConfig {
        world_width: 40,
        world_height: 40,
        food_cell_size: 10,
        initial_food: 0.0,
        food_respawn_interval: 0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.2,
        rng_seed: Some(99),
        ..ScriptBotsConfig::default()
    };

    let mut world = WorldState::new(config).expect("world");
    let max_food = world.config().food_max;
    world
        .try_update_food(|cells| cells[0] = max_food)
        .expect("finite food update");

    let before = world.food().cells().to_vec();
    world
        .step()
        .expect("food-diffusion world should accept its simulation step");

    let width = world.food().width() as usize;
    let cells = world.food().cells();
    let center_expected = expected_food_value(&world, &before, 0, 0);
    assert!(
        (cells[0] - center_expected).abs() < 1e-6,
        "center value={} expected={center_expected}",
        cells[0]
    );

    let last_x = world.food().width() - 1;
    let last_y = world.food().height() - 1;
    let neighbors = [(1_u32, 0_u32), (last_x, 0), (0, 1), (0, last_y)];
    for &(x, y) in &neighbors {
        let idx = y as usize * width + x as usize;
        let expected = expected_food_value(&world, &before, x, y);
        let value = cells[idx];
        assert!(
            (value - expected).abs() < 1e-6,
            "cell=({x},{y}) value={value} expected={expected}"
        );
    }
}

#[test]
fn food_decay_reduces_cell_values() {
    let config = ScriptBotsConfig {
        world_width: 40,
        world_height: 40,
        food_cell_size: 10,
        initial_food: 0.4,
        food_respawn_interval: 0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.1,
        food_diffusion_rate: 0.0,
        rng_seed: Some(7),
        ..ScriptBotsConfig::default()
    };

    let mut world = WorldState::new(config).expect("world");
    let before = world.food().cells().to_vec();
    world
        .step()
        .expect("food-decay world should accept its simulation step");

    let width = world.food().width() as usize;
    let height = world.food().height() as usize;
    let cells = world.food().cells();
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let value = cells[idx];
            let expected = expected_food_value(&world, &before, x as u32, y as u32);
            assert!(
                (value - expected).abs() < 1e-6,
                "cell=({x},{y}) value={value} expected={expected}"
            );
        }
    }
}

fn run_world_summary(seed: u64, ticks: u32) -> TickSummary {
    let config = ScriptBotsConfig {
        world_width: 600,
        world_height: 600,
        food_cell_size: 20,
        food_max: 1.2,
        rng_seed: Some(seed),
        initial_food: 0.4,
        persistence_interval: 1,
        ..ScriptBotsConfig::default()
    };

    let (mut world, mut persistence) =
        WorldState::with_persistence(config, Box::new(NullPersistence)).expect("world");
    world
        .try_spawn_agent(AgentData::default())
        .expect("default agent is finite");

    for _ in 0..ticks {
        persistence
            .step(&mut world)
            .expect("regression world should accept each simulation step");
    }

    let summaries: Vec<_> = world.history().cloned().collect();
    assert!(!summaries.is_empty(), "expected tick summaries");
    summaries.last().cloned().expect("latest summary")
}

#[test]
fn regression_seed_42_matches_baseline() {
    let summary = run_world_summary(42, 40);
    assert_eq!(summary.tick.0, 40);
    assert_eq!(summary.agent_count, 1);
    assert_eq!(summary.births, 0);
    assert_eq!(summary.deaths, 0);
    assert!(
        summary.total_energy.is_finite() && summary.average_energy.is_finite(),
        "energy metrics should be finite numbers"
    );
    assert!(
        (summary.total_energy - summary.average_energy).abs() < 1e-6,
        "with one agent total and average energy should match (total={}, average={})",
        summary.total_energy,
        summary.average_energy
    );
    assert!(
        (0.0..=2.0 + 1e-6).contains(&summary.total_energy),
        "post-graze energy should remain within [0, 2], got {}",
        summary.total_energy
    );
    assert!(
        summary.average_health.is_finite() && (0.0..=2.0 + 1e-6).contains(&summary.average_health),
        "average health should stay in [0,2], got {}",
        summary.average_health
    );
}

/// The false-positive budget for the narrated timeline (`bd-16g.2.3`), measured
/// against REAL runs rather than synthetic fixtures.
///
/// Unit tests on hand-built series prove the maths; only a real run proves the
/// *calibration*, and the first version of this gate proved the calibration was
/// wrong. A purely statistical stream produced **853 events per 10,000 ticks**
/// on a 3,000-tick run — "population fell 3% (23 -> 22)", "mean energy collapsed
/// (0.99 -> 0.98)". Every one of those was statistically impeccable (a nearly
/// flat baseline makes a one-agent change significant) and every one was
/// worthless. That is static, not story, and a timeline of static is one users
/// learn to ignore.
///
/// The fix was a materiality floor plus a cooldown (`NarrativePolicy`), which
/// took the same run to **26.7 events per 10k ticks** — all of them real: a
/// combat surge, genuine energy collapses and recoveries, and persistent regime
/// shifts. Measurement also corrected a second mistaken assumption: there is no
/// "quiet" seed to compare against, because this simulation's dynamics ARE
/// eventful (populations boom, crash, and oscillate by design).
///
/// So this gate asserts what actually protects the reader:
///   1. every emitted event is MATERIAL (the policy is really enforced),
///   2. the rate stays under a measured ceiling,
///   3. a world where nothing happens narrates nothing,
///   4. a world that is actually wiped out says so, and
///   5. the same seed tells the same story.
#[test]
fn narrated_timeline_respects_its_false_positive_budget_on_real_runs() {
    fn run(config: ScriptBotsConfig, agents: usize, ticks: usize) -> Vec<(u64, String, f64, f64)> {
        let mut world = WorldState::new(config).expect("world");
        for seed in 0..agents {
            world
                .try_spawn_agent(AgentData {
                    position: Position::new((seed * 37 % 190) as f32, (seed * 53 % 190) as f32),
                    health: 1.0,
                    ..AgentData::default()
                })
                .expect("generated agent is finite");
        }
        for _ in 0..ticks {
            world.step().expect("step");
        }
        world
            .narrative_events()
            .iter()
            .map(|event| {
                (
                    event.tick.0,
                    event.human_text.clone(),
                    event.before,
                    event.after,
                )
            })
            .collect()
    }

    let base = ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        rng_seed: Some(0xCA1F),
        ..ScriptBotsConfig::default()
    };

    // A normal, eventful run.
    let events = run(base.clone(), 24, 3_000);
    let per_10k = (events.len() as f64) * 10_000.0 / 3_000.0;
    assert!(
        per_10k <= 40.0,
        "narrated {per_10k:.1} events per 10k ticks (ceiling 40): {events:?}"
    );

    // Every single event must clear the materiality floor. This is the real
    // guard: it is what stops "population fell 3% (23 -> 22)" from coming back.
    for (tick, text, before, after) in &events {
        let delta = (after - before).abs();
        if text.starts_with("population fell") || text.starts_with("population rose") {
            assert!(
                delta >= 5.0 && delta / before.abs().max(1.0) >= 0.20,
                "trivial population event at t={tick}: {text} (delta {delta})"
            );
        }
        if text.starts_with("mean energy") {
            assert!(
                delta >= 0.15,
                "trivial energy event at t={tick}: {text} (delta {delta})"
            );
        }
    }

    // A world where nothing happens narrates nothing.
    let empty = run(
        ScriptBotsConfig {
            population_minimum: 0,
            population_spawn_interval: 0,
            ..base.clone()
        },
        0,
        1_000,
    );
    assert!(empty.is_empty(), "an empty world has no story: {empty:?}");

    // A world that starves to death must SAY SO. A detector that never fires is
    // as useless as one that always does, so the gate binds in both directions.
    let doomed = run(
        ScriptBotsConfig {
            metabolism_drain: 0.05,
            food_intake_rate: 0.0,
            population_minimum: 0,
            population_spawn_interval: 0,
            ..base.clone()
        },
        40,
        3_000,
    );
    assert!(
        doomed
            .iter()
            .any(|(_, text, _, _)| text.contains("fell") || text.contains("zero")),
        "a total population collapse must be narrated: {doomed:?}"
    );

    // And the story must be reproducible.
    assert_eq!(
        events,
        run(base, 24, 3_000),
        "same seed must tell the same story"
    );
}
