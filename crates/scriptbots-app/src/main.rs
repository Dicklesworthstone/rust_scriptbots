use anyhow::Result;
use rand::{SeedableRng, rngs::SmallRng};
use scriptbots_brain::MlpBrain;
use scriptbots_core::{AgentData, BrainBinding, ScriptBotsConfig, WorldState};
use scriptbots_render::run_demo;
use scriptbots_storage::{SharedStorage, Storage};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

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

fn bootstrap_world() -> Result<(Arc<Mutex<WorldState>>, Arc<Mutex<Storage>>)> {
    let config = ScriptBotsConfig {
        persistence_interval: 60,
        history_capacity: 600,
        ..ScriptBotsConfig::default()
    };

    let storage = Arc::new(Mutex::new(Storage::open("scriptbots.db")?));
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
        let config = NeuroflowBrainConfig::default();
        let key = NeuroflowBrain::register(world, config, rng);
        keys.push(key);
    }

    keys
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
