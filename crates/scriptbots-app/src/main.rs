use anyhow::Result;
use clap::{Parser, ValueEnum};
use scriptbots_app::{
    ControlRuntime, ControlServerConfig, SharedStorage, SharedWorld,
    renderer::{Renderer, RendererContext},
    terminal::TerminalRenderer,
};
use scriptbots_brain::MlpBrain;
use scriptbots_core::{AgentData, NeuroflowActivationKind, ScriptBotsConfig, WorldState};
use scriptbots_render::run_demo;
use scriptbots_storage::StoragePipeline;
use std::{
    env, fmt,
    sync::{Arc, Mutex},
};
use tracing::{debug, info, warn};

fn main() -> Result<()> {
    let cli = AppCli::parse();
    init_tracing();
    let (world, storage) = bootstrap_world()?;
    let control_config = ControlServerConfig::from_env();
    let (control_runtime, command_drain, command_submit) =
        ControlRuntime::launch(world.clone(), control_config)?;
    let (active_mode, renderer) = resolve_renderer(cli.mode)?;
    info!(
        requested_mode = cli.mode.as_str(),
        active_mode = active_mode.as_str(),
        renderer = renderer.name(),
        "Starting ScriptBots simulation shell"
    );
    let context = RendererContext {
        world: Arc::clone(&world),
        storage: Arc::clone(&storage),
        control_runtime: &control_runtime,
        command_drain,
        command_submit,
    };
    renderer.run(context)?;
    control_runtime.shutdown()?;
    Ok(())
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}

fn bootstrap_world() -> Result<(SharedWorld, SharedStorage)> {
    let mut config = ScriptBotsConfig {
        persistence_interval: 60,
        history_capacity: 600,
        ..ScriptBotsConfig::default()
    };
    apply_env_overrides(&mut config);

    let pipeline = StoragePipeline::new("scriptbots.db")?;
    let storage: SharedStorage = pipeline.storage();
    let mut world = WorldState::with_persistence(config, Box::new(pipeline))?;
    let brain_keys = install_brains(&mut world);

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

#[derive(Parser, Debug)]
#[command(
    name = "scriptbots-app",
    version,
    about = "ScriptBots simulation shell"
)]
struct AppCli {
    /// Rendering mode for the simulation shell (auto detects GPUI fallback).
    #[arg(
        long,
        value_enum,
        env = "SCRIPTBOTS_MODE",
        default_value_t = RendererMode::Auto
    )]
    mode: RendererMode,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum RendererMode {
    Auto,
    Gui,
    Terminal,
}

impl RendererMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gui => "gui",
            Self::Terminal => "terminal",
        }
    }
}

impl fmt::Display for RendererMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

fn resolve_renderer(mode: RendererMode) -> Result<(RendererMode, Box<dyn Renderer>)> {
    match mode {
        RendererMode::Gui => Ok((RendererMode::Gui, Box::new(GuiRenderer))),
        RendererMode::Terminal => Ok((
            RendererMode::Terminal,
            Box::new(TerminalRenderer::default()),
        )),
        RendererMode::Auto => {
            if should_use_terminal_mode() {
                debug!("Auto-selected terminal renderer due to headless environment");
                Ok((
                    RendererMode::Terminal,
                    Box::new(TerminalRenderer::default()),
                ))
            } else {
                Ok((RendererMode::Gui, Box::new(GuiRenderer)))
            }
        }
    }
}

#[derive(Default)]
struct GuiRenderer;

impl Renderer for GuiRenderer {
    fn name(&self) -> &'static str {
        "gpui"
    }

    fn run(&self, ctx: RendererContext<'_>) -> Result<()> {
        run_demo(
            Arc::clone(&ctx.world),
            Some(Arc::clone(&ctx.storage)),
            Arc::clone(&ctx.command_drain),
            Arc::clone(&ctx.command_submit),
        );
        Ok(())
    }
}

fn should_use_terminal_mode() -> bool {
    if let Ok(value) = env::var("SCRIPTBOTS_FORCE_TERMINAL")
        && let Some(flag) = parse_bool(&value)
    {
        return flag;
    }

    if let Ok(value) = env::var("SCRIPTBOTS_FORCE_GUI")
        && matches!(parse_bool(&value), Some(true))
    {
        return false;
    }

    #[cfg(target_family = "unix")]
    {
        let has_display =
            env::var_os("DISPLAY").is_some() || env::var_os("WAYLAND_DISPLAY").is_some();
        if !has_display {
            return true;
        }
    }

    false
}

fn install_brains(world: &mut WorldState) -> Vec<u64> {
    let mut keys = Vec::new();

    let mlp_key = world
        .brain_registry_mut()
        .register(MlpBrain::KIND.as_str(), |seed_rng| {
            MlpBrain::runner(seed_rng)
        });
    keys.push(mlp_key);

    #[cfg(feature = "ml")]
    {
        let label = {
            let prototype = scriptbots_brain_ml::runner();
            prototype.kind().to_string()
        };
        let key = world
            .brain_registry_mut()
            .register(label, |_seed_rng| scriptbots_brain_ml::runner());
        keys.push(key);
    }

    #[cfg(feature = "neuro")]
    {
        use scriptbots_brain_neuro::{NeuroflowBrain, NeuroflowBrainConfig};
        let settings = world.config().neuroflow.clone();
        if settings.enabled {
            let config = NeuroflowBrainConfig::from_settings(&settings);
            let key = NeuroflowBrain::register(world, config);
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
            if let Some(&key) = brain_keys.get((row * 4 + col) % brain_keys.len())
                && !world.bind_agent_brain(id, key)
            {
                warn!(agent = ?id, key, "Failed to bind brain to seeded agent");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let keys = install_brains(&mut world);
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
        let keys_enabled = install_brains(&mut world_enabled);
        assert_eq!(
            keys_enabled.len(),
            2,
            "Expected both MLP and NeuroFlow brains"
        );

        let neuro_key = *keys_enabled.last().expect("neuro key");
        let agent_id = world_enabled.spawn_agent(AgentData::default());
        assert!(world_enabled.bind_agent_brain(agent_id, neuro_key));
        world_enabled.step();
        let outputs_one = world_enabled.agent_runtime(agent_id).unwrap().outputs;

        let mut world_repeat = WorldState::new(config_enabled).expect("world");
        let keys_repeat = install_brains(&mut world_repeat);
        assert_eq!(keys_repeat.len(), 2);
        let neuro_repeat = *keys_repeat.last().unwrap();
        let agent_repeat = world_repeat.spawn_agent(AgentData::default());
        assert!(world_repeat.bind_agent_brain(agent_repeat, neuro_repeat));
        world_repeat.step();
        let outputs_two = world_repeat.agent_runtime(agent_repeat).unwrap().outputs;

        assert_eq!(
            outputs_one, outputs_two,
            "NeuroFlow outputs should be deterministic for same seed"
        );
    }
}
