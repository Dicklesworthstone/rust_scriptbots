use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
use scriptbots_brain::MlpBrain;
use scriptbots_core::{
    AgentData, BrainBinding, NeuroflowActivationKind, ScriptBotsConfig, WorldState,
};
use scriptbots_render::run_demo;
use scriptbots_storage::{SharedStorage, Storage};
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

    let storage: SharedStorageArc = Arc::new(Mutex::new(Storage::open("scriptbots.db")?));
    let persistence = SharedStorage::new(Arc::clone(&storage));
    let mut world = WorldState::with_persistence(config, Box::new(persistence))?;
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
            if let Some(runtime) = world.agent_runtime_mut(id) {
                if let Some(&key) = brain_keys.get((row * 4 + col) % brain_keys.len()) {
                    runtime.brain = BrainBinding::Registry { key };
                }
            }
        }
    }
}
