use anyhow::Result;
use scriptbots_core::{AgentData, ScriptBotsConfig, WorldState};
use scriptbots_render::run_demo;
use scriptbots_storage::Storage;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

fn main() -> Result<()> {
    init_tracing();
    let world = bootstrap_world()?;
    info!("Starting ScriptBots simulation shell");
    run_demo(world);
    Ok(())
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}

fn bootstrap_world() -> Result<Arc<Mutex<WorldState>>> {
    let mut config = ScriptBotsConfig::default();
    config.persistence_interval = 60;
    config.history_capacity = 600;

    let storage = Storage::open("scriptbots.db")?;
    let mut world = WorldState::with_persistence(config, Box::new(storage))?;

    seed_agents(&mut world);

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

    Ok(Arc::new(Mutex::new(world)))
}

fn seed_agents(world: &mut WorldState) {
    let mut agent = AgentData::default();
    let spacing = 120.0;
    for row in 0..4 {
        for col in 0..4 {
            agent.position.x = col as f32 * spacing + spacing * 0.5;
            agent.position.y = row as f32 * spacing + spacing * 0.5;
            agent.heading = 0.0;
            agent.spike_length = 10.0;
            world.spawn_agent(agent);
        }
    }
}
