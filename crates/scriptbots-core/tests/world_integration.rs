use scriptbots_core::{
    AgentData, BrainRunner, FoodCellProfileSnapshot, NUM_EYES, Position, ScriptBotsConfig, Tick,
    TickSummary, TraitModifiers, WorldState,
};

fn default_profile(config: &ScriptBotsConfig) -> FoodCellProfileSnapshot {
    FoodCellProfileSnapshot {
        capacity: config.food_max,
        growth_multiplier: 1.0,
        decay_multiplier: 1.0,
        fertility: 0.0,
        nutrient_density: 0.3,
    }
}

fn expected_food_value(
    profiles: &[FoodCellProfileSnapshot],
    config: &ScriptBotsConfig,
    before: &[f32],
    width: usize,
    height: usize,
    x: u32,
    y: u32,
) -> f32 {
    let idx = y as usize * width + x as usize;
    let profile = profiles
        .get(idx)
        .copied()
        .unwrap_or_else(|| default_profile(config));

    let diffusion = config.food_diffusion_rate;
    let decay = config.food_decay_rate;
    let growth = config.food_growth_rate;

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

    if growth > 0.0 {
        value += growth * profile.growth_multiplier * (profile.capacity - value);
    }

    value.clamp(0.0, profile.capacity.max(0.0))
}

#[test]
fn seeded_world_advances_deterministically() {
    let config = ScriptBotsConfig {
        world_width: 256,
        world_height: 256,
        food_cell_size: 16,
        initial_food: 0.25,
        food_max: 1.0,
        rng_seed: Some(0xDEADBEEF),
        ..ScriptBotsConfig::default()
    };

    let mut world_a = WorldState::new(config.clone()).expect("world_a");
    let mut world_b = WorldState::new(config).expect("world_b");

    let agent = AgentData::default();
    let id_a = world_a.spawn_agent(agent);
    let agent = AgentData::default();
    let id_b = world_b.spawn_agent(agent);

    for _ in 0..8 {
        world_a.step();
        world_b.step();
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
        .register("test.constant", |_rng| {
            Box::new(ConstantBrain { value: 0.75 })
        });

    let agent_id = world.spawn_agent(AgentData::default());
    assert!(world.bind_agent_brain(agent_id, key));

    world.step();
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

    let attacker_id = world.spawn_agent(attacker);
    let victim_id = world.spawn_agent(victim);

    let spike_key = world
        .brain_registry_mut()
        .register("test.spike", |_rng| Box::new(SpikeBrain));
    assert!(world.bind_agent_brain(attacker_id, spike_key));
    if let Some(runtime) = world.agent_runtime_mut(attacker_id) {
        runtime.herbivore_tendency = 0.1;
    }
    if let Some(runtime) = world.agent_runtime_mut(victim_id) {
        runtime.herbivore_tendency = 0.9;
    }

    world.step();

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
    let subject = world.spawn_agent(AgentData::default());
    let neighbor = world.spawn_agent(AgentData::default());

    {
        let arena = world.agents_mut();
        let idx_subject = arena.index_of(subject).expect("subject index");
        let idx_neighbor = arena.index_of(neighbor).expect("neighbor index");
        let columns = arena.columns_mut();
        columns.positions_mut()[idx_subject] = Position::new(80.0, 100.0);
        columns.positions_mut()[idx_neighbor] = Position::new(120.0, 100.0);
        columns.headings_mut()[idx_subject] = 0.0;
        columns.headings_mut()[idx_neighbor] = 0.0;
        columns.colors_mut()[idx_neighbor] = [1.0, 0.2, 0.2];
        columns.health_mut()[idx_neighbor] = 0.4;
    }

    let food_max = world.config().food_max;
    if let Some(cell) = world.food_mut().get_mut(4, 5) {
        *cell = food_max * 0.8;
    }

    if let Some(runtime) = world.runtime_mut().get_mut(subject) {
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
    }

    if let Some(runtime) = world.runtime_mut().get_mut(neighbor) {
        runtime.sound_multiplier = 0.9;
    }

    world.step();

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
        sensors[16] >= 0.0 && sensors[16] <= 1.0,
        "clock sensor within bounds"
    );
    assert!(
        sensors[20] >= 0.0 && sensors[20] <= 1.0,
        "temperature discomfort normalized"
    );
    assert!(
        sensors[20] <= 0.1,
        "temperature discomfort low when preference matches"
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
    let width = world.food().width() as usize;
    let height = world.food().height() as usize;
    let config_snapshot = world.config().clone();
    let mut profiles = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            profiles.push(
                world
                    .food_profile(x as u32, y as u32)
                    .unwrap_or_else(|| default_profile(&config_snapshot)),
            );
        }
    }

    let before = world.food().cells().to_vec();
    world.step();

    let cells = world.food().cells();
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let value = cells[idx];
            let expected = expected_food_value(
                &profiles,
                &config_snapshot,
                &before,
                width,
                height,
                x as u32,
                y as u32,
            );
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
    let width = world.food().width() as usize;
    let height = world.food().height() as usize;
    let config_snapshot = world.config().clone();
    let mut profiles = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            profiles.push(
                world
                    .food_profile(x as u32, y as u32)
                    .unwrap_or_else(|| default_profile(&config_snapshot)),
            );
        }
    }

    let max_food = world.config().food_max;
    if let Some(cell) = world.food_mut().get_mut(0, 0) {
        *cell = max_food;
    }

    let before = world.food().cells().to_vec();
    world.step();

    let cells = world.food().cells();
    let center_expected =
        expected_food_value(&profiles, &config_snapshot, &before, width, height, 0, 0);
    assert!(
        (cells[0] - center_expected).abs() < 1e-6,
        "center value={} expected={center_expected}",
        cells[0]
    );

    let neighbors = [
        (1_u32, 0_u32),
        (world.food().width() - 1, 0),
        (0, 1),
        (0, world.food().height() - 1),
    ];
    for &(x, y) in &neighbors {
        let idx = y as usize * width + x as usize;
        let expected =
            expected_food_value(&profiles, &config_snapshot, &before, width, height, x, y);
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
    let width = world.food().width() as usize;
    let height = world.food().height() as usize;
    let config_snapshot = world.config().clone();
    let mut profiles = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            profiles.push(
                world
                    .food_profile(x as u32, y as u32)
                    .unwrap_or_else(|| default_profile(&config_snapshot)),
            );
        }
    }

    let before = world.food().cells().to_vec();
    world.step();

    let cells = world.food().cells();
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let value = cells[idx];
            let expected = expected_food_value(
                &profiles,
                &config_snapshot,
                &before,
                width,
                height,
                x as u32,
                y as u32,
            );
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

    let mut world = WorldState::new(config).expect("world");
    world.spawn_agent(AgentData::default());

    for _ in 0..ticks {
        world.step();
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
