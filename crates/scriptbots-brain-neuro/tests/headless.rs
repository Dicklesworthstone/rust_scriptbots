use scriptbots_brain_neuro::{NeuroflowBrain, NeuroflowBrainConfig};
use scriptbots_core::{AgentData, ScriptBotsConfig, WorldState};

#[test]
fn neuroflow_only_world_is_deterministic() {
    let mut config = ScriptBotsConfig::default();
    config.neuroflow.enabled = true;
    config.neuroflow.hidden_layers = vec![32, 16];
    config.neuroflow.activation = scriptbots_core::NeuroflowActivationKind::Sigmoid;
    config.world_width = 200;
    config.world_height = 200;
    config.food_cell_size = 20;
    config.initial_food = 0.25;
    config.rng_seed = Some(0xA1B2C3);

    let mut world_a = WorldState::new(config.clone()).expect("world_a");
    let mut world_b = WorldState::new(config).expect("world_b");

    let brain_config = NeuroflowBrainConfig::from_settings(&world_a.config().neuroflow);
    let key_a = NeuroflowBrain::register(&mut world_a, brain_config.clone());
    let key_b = NeuroflowBrain::register(&mut world_b, brain_config);

    let agent_a = world_a.spawn_agent(AgentData::default());
    let agent_b = world_b.spawn_agent(AgentData::default());

    assert!(world_a.bind_agent_brain(agent_a, key_a));
    assert!(world_b.bind_agent_brain(agent_b, key_b));

    for _ in 0..6 {
        world_a.step();
        world_b.step();
    }

    let outputs_a = world_a.agent_runtime(agent_a).expect("runtime_a").outputs;
    let outputs_b = world_b.agent_runtime(agent_b).expect("runtime_b").outputs;
    assert_eq!(
        outputs_a, outputs_b,
        "NeuroFlow-only brains should produce deterministic outputs with identical seeds"
    );
}
