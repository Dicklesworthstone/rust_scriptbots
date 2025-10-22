use anyhow::Result;
use rand::{SeedableRng, rngs::SmallRng};
use scriptbots_brain::MlpBrain;
use scriptbots_core::{
    AgentData, BrainBinding, NeuroflowActivationKind, ScriptBotsConfig, WorldState,
};
use scriptbots_render::run_demo;
use scriptbots_storage::{Storage, StoragePipeline};
use std::{
    env,
    sync::{Arc, Mutex},
};
use tracing::{info, warn};

type SharedWorld = Arc<Mutex<WorldState>>;
type SharedStorageArc = Arc<Mutex<Storage>>;

fn main() -> Result<()> {
    init_tracing();
    let (world, _storage) = bootstrap_world()?;
    info!("Starting ScriptBots simulation shell");
    run_demo(world);
    Ok(())
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}

fn bootstrap_world() -> Result<(SharedWorld, SharedStorageArc)> {
    let mut config = ScriptBotsConfig {
        persistence_interval: 60,
        history_capacity: 600,
        ..ScriptBotsConfig::default()
    };
    apply_env_overrides(&mut config);

    let pipeline = StoragePipeline::new("scriptbots.db")?;
    let storage: SharedStorageArc = pipeline.storage();
    let mut world = WorldState::with_persistence(config, Box::new(pipeline))?;
    let mut rng =
        SmallRng::seed_from_u64(world.config().rng_seed.unwrap_or(0xFACA_DEAF_0123_4567_u64));
    let brain_keys = install_brains(&mut world, &mut rng);

    seed_agents(&mut world, &brain_keys);

    for _ in 0..120 {
        world.step();
    }

    if let Some(summary) = world.history().last() {
        info!(
            tick = summary.tick.0,
            agents = summary.agent_count,
            births = summary.births,
            deaths = summary.deaths,
            avg_energy = summary.average_energy,
            "Primed world and persisted initial summary",
        );
    } else {
        warn!("World bootstrap completed without persistence summaries");
    }

    Ok((Arc::new(Mutex::new(world)), storage))
}

fn install_brains(world: &mut WorldState, rng: &mut SmallRng) -> Vec<u64> {
    let mut keys = Vec::new();

    let mlp_key = world.brain_registry_mut().register(MlpBrain::runner(rng));
    keys.push(mlp_key);

    #[cfg(feature = "ml")]
    {
        let key = world
            .brain_registry_mut()
            .register(scriptbots_brain_ml::runner());
        keys.push(key);
    }

    #[cfg(feature = "neuro")]
    {
        use scriptbots_brain_neuro::{NeuroflowBrain, NeuroflowBrainConfig};
        let settings = world.config().neuroflow.clone();
        if settings.enabled {
            let config = NeuroflowBrainConfig::from_settings(&settings);
            let key = NeuroflowBrain::register(world, config, rng);
            keys.push(key);
        }
    }

    keys
}

fn apply_env_overrides(config: &mut ScriptBotsConfig) {
    if let Ok(value) = env::var("SCRIPTBOTS_NEUROFLOW_ENABLED") {
        match parse_bool(&value) {
            Some(flag) => config.neuroflow.enabled = flag,
            None => {
                warn!(value = %value, "Invalid SCRIPTBOTS_NEUROFLOW_ENABLED value; expected true/false")
            }
        }
    }

    if let Ok(value) = env::var("SCRIPTBOTS_NEUROFLOW_HIDDEN") {
        match parse_layers(&value) {
            Some(layers) => config.neuroflow.hidden_layers = layers,
            None => {
                warn!(value = %value, "Invalid SCRIPTBOTS_NEUROFLOW_HIDDEN value; expected comma-separated integers")
            }
        }
    }

    if let Ok(value) = env::var("SCRIPTBOTS_NEUROFLOW_ACTIVATION") {
        match parse_activation(&value) {
            Some(activation) => config.neuroflow.activation = activation,
            None => {
                warn!(value = %value, "Invalid SCRIPTBOTS_NEUROFLOW_ACTIVATION value; expected tanh|sigmoid|relu")
            }
        }
    }
}

fn parse_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_layers(raw: &str) -> Option<Vec<usize>> {
    let mut layers = Vec::new();
    for token in raw.split(',') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed.parse::<usize>() {
            Ok(value) if value > 0 => layers.push(value),
            _ => return None,
        }
    }
    Some(layers)
}

fn parse_activation(raw: &str) -> Option<NeuroflowActivationKind> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "tanh" => Some(NeuroflowActivationKind::Tanh),
        "sigmoid" => Some(NeuroflowActivationKind::Sigmoid),
        "relu" => Some(NeuroflowActivationKind::Relu),
        _ => None,
    }
}

fn seed_agents(world: &mut WorldState, brain_keys: &[u64]) {
    let mut agent = AgentData::default();
    let spacing = 120.0;
    for row in 0..4 {
        for col in 0..4 {
            agent.position.x = col as f32 * spacing + spacing * 0.5;
            agent.position.y = row as f32 * spacing + spacing * 0.5;
            agent.heading = 0.0;
            agent.spike_length = 10.0;
            let id = world.spawn_agent(agent);
            if let Some(runtime) = world.agent_runtime_mut(id)
                && let Some(&key) = brain_keys.get((row * 4 + col) % brain_keys.len())
            {
                runtime.brain = BrainBinding::Registry { key };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "neuro")]
    use rand::{SeedableRng, rngs::SmallRng};
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn with_env_lock<F: FnOnce()>(f: F) {
        let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock.lock().expect("env mutex poisoned");
        f();
    }

    fn restore_env(var: &str, previous: Option<String>) {
        unsafe {
            if let Some(value) = previous {
                std::env::set_var(var, value);
            } else {
                std::env::remove_var(var);
            }
        }
    }

    #[test]
    fn env_overrides_apply_expected_settings() {
        with_env_lock(|| {
            let prev_enabled = std::env::var("SCRIPTBOTS_NEUROFLOW_ENABLED").ok();
            let prev_hidden = std::env::var("SCRIPTBOTS_NEUROFLOW_HIDDEN").ok();
            let prev_activation = std::env::var("SCRIPTBOTS_NEUROFLOW_ACTIVATION").ok();

            unsafe {
                std::env::set_var("SCRIPTBOTS_NEUROFLOW_ENABLED", "true");
                std::env::set_var("SCRIPTBOTS_NEUROFLOW_HIDDEN", "64, 32 ,16");
                std::env::set_var("SCRIPTBOTS_NEUROFLOW_ACTIVATION", "relu");
            }

            let mut config = ScriptBotsConfig::default();
            apply_env_overrides(&mut config);

            assert!(config.neuroflow.enabled);
            assert_eq!(config.neuroflow.hidden_layers, vec![64, 32, 16]);
            assert_eq!(config.neuroflow.activation, NeuroflowActivationKind::Relu);

            restore_env("SCRIPTBOTS_NEUROFLOW_ENABLED", prev_enabled);
            restore_env("SCRIPTBOTS_NEUROFLOW_HIDDEN", prev_hidden);
            restore_env("SCRIPTBOTS_NEUROFLOW_ACTIVATION", prev_activation);
        });
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn neuroflow_installation_respects_toggle() {
        let mut config = ScriptBotsConfig::default();
        config.neuroflow.enabled = false;
        let mut world = WorldState::new(config).expect("world");
        let mut rng = SmallRng::seed_from_u64(7);
        let keys = install_brains(&mut world, &mut rng);
        assert_eq!(
            keys.len(),
            1,
            "NeuroFlow brain should not register when disabled"
        );

        let mut config_enabled = ScriptBotsConfig::default();
        config_enabled.neuroflow.enabled = true;
        config_enabled.neuroflow.hidden_layers = vec![12, 6];
        config_enabled.neuroflow.activation = NeuroflowActivationKind::Sigmoid;
        config_enabled.rng_seed = Some(99);
        let mut world_enabled = WorldState::new(config_enabled.clone()).expect("world");
        let mut rng_enabled = SmallRng::seed_from_u64(99);
        let keys_enabled = install_brains(&mut world_enabled, &mut rng_enabled);
        assert_eq!(
            keys_enabled.len(),
            2,
            "Expected both MLP and NeuroFlow brains"
        );

        let neuro_key = *keys_enabled.last().expect("neuro key");
        let agent_id = world_enabled.spawn_agent(AgentData::default());
        if let Some(runtime) = world_enabled.agent_runtime_mut(agent_id) {
            runtime.brain = BrainBinding::Registry { key: neuro_key };
        }
        world_enabled.step();
        let outputs_one = world_enabled.agent_runtime(agent_id).unwrap().outputs;

        let mut world_repeat = WorldState::new(config_enabled).expect("world");
        let mut rng_repeat = SmallRng::seed_from_u64(99);
        let keys_repeat = install_brains(&mut world_repeat, &mut rng_repeat);
        assert_eq!(keys_repeat.len(), 2);
        let neuro_repeat = *keys_repeat.last().unwrap();
        let agent_repeat = world_repeat.spawn_agent(AgentData::default());
        if let Some(runtime) = world_repeat.agent_runtime_mut(agent_repeat) {
            runtime.brain = BrainBinding::Registry { key: neuro_repeat };
        }
        world_repeat.step();
        let outputs_two = world_repeat.agent_runtime(agent_repeat).unwrap().outputs;

        assert_eq!(
            outputs_one, outputs_two,
            "NeuroFlow outputs should be deterministic for same seed"
        );
    }
}
