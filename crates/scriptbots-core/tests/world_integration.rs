use scriptbots_core::{
    AgentData, BrainBinding, BrainRunner, ScriptBotsConfig, Tick, TickSummary, WorldState,
};

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
        .register(Box::new(ConstantBrain { value: 0.75 }));

    let agent_id = world.spawn_agent(AgentData::default());
    if let Some(runtime) = world.agent_runtime_mut(agent_id) {
        runtime.brain = BrainBinding::Registry { key };
    }

    world.step();
    let runtime = world.agent_runtime(agent_id).expect("runtime");
    assert!(runtime
        .outputs
        .iter()
        .all(|v| (*v - 0.75).abs() < f32::EPSILON));
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
    assert!((summary.total_energy - 1.026_941_3).abs() < 1e-6);
    assert!((summary.average_energy - 1.026_941_3).abs() < 1e-6);
    assert!((summary.average_health - 0.826_941_55).abs() < 1e-6);
}
