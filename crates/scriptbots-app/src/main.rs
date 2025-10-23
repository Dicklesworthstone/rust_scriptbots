use anyhow::{Context, Result, bail};
use clap::{ArgAction, Parser, ValueEnum};
use owo_colors::OwoColorize;
use scriptbots_app::{
    ControlRuntime, ControlServerConfig, SharedStorage, SharedWorld,
    renderer::{Renderer, RendererContext},
    terminal::TerminalRenderer,
};
use scriptbots_brain::MlpBrain;
use scriptbots_core::{
    AgentData, NeuroflowActivationKind, ReplayEventKind, ScriptBotsConfig, TickSummary,
    WorldPersistence, WorldState,
};
use scriptbots_render::run_demo;
use scriptbots_storage::{PersistedReplayEvent, Storage, StoragePipeline};
use serde_json::{self, Value as JsonValue};
use std::{
    collections::HashMap,
    env, fmt, fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tracing::{debug, info, warn};

fn main() -> Result<()> {
    let cli = AppCli::parse();
    init_tracing();
    if cli.replay_db.is_some() {
        run_replay_cli(&cli)?;
        return Ok(());
    }

    let (world, storage) = bootstrap_world(&cli)?;
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

fn bootstrap_world(cli: &AppCli) -> Result<(SharedWorld, SharedStorage)> {
    let config = compose_config(cli)?;

    let storage_path =
        env::var("SCRIPTBOTS_STORAGE_PATH").unwrap_or_else(|_| "scriptbots.db".to_string());
    if let Some(parent) = Path::new(&storage_path)
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    let pipeline = StoragePipeline::new(&storage_path)?;
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

fn compose_config(cli: &AppCli) -> Result<ScriptBotsConfig> {
    let mut config = ScriptBotsConfig {
        persistence_interval: 60,
        history_capacity: 600,
        ..ScriptBotsConfig::default()
    };
    config = apply_config_layers(config, &cli.config_layers)?;
    apply_env_overrides(&mut config);
    Ok(config)
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
    /// Layered configuration files (TOML or RON) applied in order.
    #[arg(
        long = "config",
        value_name = "FILE",
        action = ArgAction::Append,
        env = "SCRIPTBOTS_CONFIG",
        value_delimiter = ';'
    )]
    config_layers: Vec<PathBuf>,
    /// Path to a DuckDB run to verify via headless deterministic replay.
    #[arg(long = "replay-db", value_name = "FILE", env = "SCRIPTBOTS_REPLAY_DB")]
    replay_db: Option<PathBuf>,
    /// Optional comparison database for divergence analysis.
    #[arg(long = "compare-db", value_name = "FILE", requires = "replay_db")]
    compare_db: Option<PathBuf>,
    /// Limit the number of ticks simulated during replay verification.
    #[arg(long = "tick-limit", value_name = "TICKS", requires = "replay_db")]
    tick_limit: Option<u64>,
}

fn apply_config_layers(base: ScriptBotsConfig, layers: &[PathBuf]) -> Result<ScriptBotsConfig> {
    if layers.is_empty() {
        return Ok(base);
    }

    let mut merged = serde_json::to_value(&base).expect("serialize base config");
    for path in layers {
        let layer_value = load_config_layer(path)?;
        info!(
            layer = %path.display(),
            "Applying configuration layer"
        );
        merge_layer(&mut merged, layer_value);
    }

    serde_json::from_value(merged)
        .map_err(|err| anyhow::anyhow!("failed to deserialize merged configuration: {err}"))
}

fn load_config_layer(path: &Path) -> Result<JsonValue> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration layer {}", path.display()))?;

    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("ron") => ron::from_str(&contents)
            .with_context(|| format!("failed to parse RON config layer {}", path.display())),
        _ => toml::from_str(&contents)
            .with_context(|| format!("failed to parse TOML config layer {}", path.display())),
    }
}

fn merge_layer(base: &mut JsonValue, layer: JsonValue) {
    match (base, layer) {
        (JsonValue::Object(base_map), JsonValue::Object(layer_map)) => {
            for (key, value) in layer_map {
                if let Some(existing) = base_map.get_mut(&key) {
                    merge_layer(existing, value);
                } else {
                    base_map.insert(key, value);
                }
            }
        }
        (target, value) => {
            *target = value;
        }
    }
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

fn run_replay_cli(cli: &AppCli) -> Result<()> {
    let db_path = cli
        .replay_db
        .as_ref()
        .expect("replay_db required to enter replay mode");
    let db_display = db_path.display().to_string();

    let mut storage = Storage::open(&db_display)
        .with_context(|| format!("failed to open replay database {db_display}"))?;
    let recorded_max_tick = storage.max_tick()?.unwrap_or(0);
    let persisted_events = storage.load_replay_events()?;
    let recorded_counts = storage.replay_event_counts()?;
    drop(storage);

    let events_max_tick = persisted_events.iter().map(|e| e.tick).max().unwrap_or(0);
    let tick_limit = cli
        .tick_limit
        .unwrap_or(recorded_max_tick.max(events_max_tick));

    let config = compose_config(cli)?;
    if config.rng_seed.is_none() {
        warn!(
            "config_rng_seed" = false,
            "Replay config has no rng_seed; deterministic verification may fail"
        );
    }

    let replay_run = run_headless_simulation(&config, tick_limit)?;
    let simulated_tick_count = replay_run.summaries.len() as u64;
    debug!(
        simulated_ticks = simulated_tick_count,
        simulated_events = replay_run.events.len(),
        "Headless replay complete"
    );
    if simulated_tick_count != tick_limit {
        warn!(
            requested_ticks = tick_limit,
            simulated_ticks = simulated_tick_count,
            "Simulated tick count differs from requested limit"
        );
    }
    let diff = diff_event_stream(&persisted_events, &replay_run.events);

    let recorded_map = recorded_counts
        .into_iter()
        .map(|entry| (entry.event_type, entry.count))
        .collect::<HashMap<_, _>>();
    let simulated_counts = count_event_kinds(&replay_run.events)
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect::<HashMap<_, _>>();

    println!(
        "{} Replaying {} ticks ({} recorded events) against {}",
        "▶".bright_blue().bold(),
        tick_limit,
        persisted_events.len(),
        db_display.cyan()
    );
    print_event_counts("recorded", &recorded_map, None);
    print_event_counts("simulated", &simulated_counts, Some(&recorded_map));

    if let Some(divergence) = diff {
        report_divergence("recorded", "simulated", divergence)?;
    } else {
        println!(
            "{} Replay matched {} events across {} ticks",
            "✔".green().bold(),
            replay_run.events.len().green(),
            simulated_tick_count.green()
        );
    }

    if let Some(compare_path) = cli.compare_db.as_ref() {
        let compare_display = compare_path.display().to_string();
        let mut other = Storage::open(&compare_display)
            .with_context(|| format!("failed to open comparison database {compare_display}"))?;
        let other_events = other.load_replay_events()?;
        let other_counts = other.replay_event_counts()?;
        drop(other);

        println!(
            "{} Comparing {} against {}",
            "▶".bright_blue().bold(),
            db_display.cyan(),
            compare_display.cyan()
        );
        let compare_diff = diff_event_stream(&persisted_events, &other_events);
        let other_map = other_counts
            .into_iter()
            .map(|entry| (entry.event_type, entry.count))
            .collect::<HashMap<_, _>>();
        print_event_counts("baseline", &recorded_map, None);
        print_event_counts("candidate", &other_map, Some(&recorded_map));

        if let Some(divergence) = compare_diff {
            report_divergence("baseline", "candidate", divergence)?;
        } else {
            println!(
                "{} Event streams are identical ({} events)",
                "✔".green().bold(),
                other_events.len().green()
            );
        }
    }

    Ok(())
}

struct ReplayCollector {
    ticks: Arc<Mutex<Vec<ReplayTickRecord>>>,
}

impl ReplayCollector {
    fn new() -> (Self, Arc<Mutex<Vec<ReplayTickRecord>>>) {
        let ticks = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                ticks: Arc::clone(&ticks),
            },
            ticks,
        )
    }
}

impl WorldPersistence for ReplayCollector {
    fn on_tick(&mut self, payload: &scriptbots_core::PersistenceBatch) {
        if let Ok(mut guard) = self.ticks.lock() {
            guard.push(ReplayTickRecord {
                tick: payload.summary.tick.0,
                events: payload.replay_events.clone(),
                summary: payload.summary.clone(),
            });
        }
    }
}

struct ReplayTickRecord {
    tick: u64,
    events: Vec<scriptbots_core::ReplayEvent>,
    summary: TickSummary,
}

struct ReplayRun {
    events: Vec<PersistedReplayEvent>,
    summaries: Vec<TickSummary>,
}

fn run_headless_simulation(config: &ScriptBotsConfig, tick_limit: u64) -> Result<ReplayRun> {
    let (collector, handle) = ReplayCollector::new();
    let mut world = WorldState::with_persistence(config.clone(), Box::new(collector))?;
    let brain_keys = install_brains(&mut world);
    seed_agents(&mut world, &brain_keys);

    for _ in 0..tick_limit {
        world.step();
    }

    drop(world);

    let records = Arc::try_unwrap(handle)
        .map_err(|_| anyhow::anyhow!("replay collector still in use"))?
        .into_inner()
        .map_err(|err| anyhow::anyhow!("replay collector poisoned: {err}"))?;

    let mut events = Vec::new();
    let mut summaries = Vec::with_capacity(records.len());
    for record in records {
        summaries.push(record.summary);
        for (seq, event) in record.events.into_iter().enumerate() {
            events.push(PersistedReplayEvent {
                tick: record.tick,
                seq: seq as u64,
                event,
            });
        }
    }

    Ok(ReplayRun { events, summaries })
}

#[derive(Debug)]
struct Divergence {
    kind: DivergenceKind,
    expected: Option<PersistedReplayEvent>,
    actual: Option<PersistedReplayEvent>,
}

#[derive(Debug)]
enum DivergenceKind {
    TickMismatch,
    SequenceMismatch,
    EventMismatch,
    MissingActual,
    ExtraActual,
}

fn diff_event_stream(
    expected: &[PersistedReplayEvent],
    actual: &[PersistedReplayEvent],
) -> Option<Divergence> {
    let mut idx = 0;
    loop {
        match (expected.get(idx), actual.get(idx)) {
            (Some(left), Some(right)) => {
                if left.tick != right.tick {
                    return Some(Divergence {
                        kind: DivergenceKind::TickMismatch,
                        expected: Some(left.clone()),
                        actual: Some(right.clone()),
                    });
                }
                if left.seq != right.seq {
                    return Some(Divergence {
                        kind: DivergenceKind::SequenceMismatch,
                        expected: Some(left.clone()),
                        actual: Some(right.clone()),
                    });
                }
                if left.event != right.event {
                    return Some(Divergence {
                        kind: DivergenceKind::EventMismatch,
                        expected: Some(left.clone()),
                        actual: Some(right.clone()),
                    });
                }
            }
            (Some(left), None) => {
                return Some(Divergence {
                    kind: DivergenceKind::MissingActual,
                    expected: Some(left.clone()),
                    actual: None,
                });
            }
            (None, Some(right)) => {
                return Some(Divergence {
                    kind: DivergenceKind::ExtraActual,
                    expected: None,
                    actual: Some(right.clone()),
                });
            }
            (None, None) => return None,
        }
        idx += 1;
    }
}

fn count_event_kinds(events: &[PersistedReplayEvent]) -> HashMap<&'static str, u64> {
    let mut counts = HashMap::new();
    for entry in events {
        let key = match entry.event.kind {
            ReplayEventKind::BrainOutputs { .. } => "brain_outputs",
            ReplayEventKind::Action { .. } => "action",
            ReplayEventKind::RngSample { .. } => "rng_sample",
        };
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn report_divergence(left_label: &str, right_label: &str, divergence: Divergence) -> Result<()> {
    let marker = format!("{}", "✖".red().bold());
    match divergence.kind {
        DivergenceKind::TickMismatch => {
            if let (Some(exp), Some(act)) = (&divergence.expected, &divergence.actual) {
                println!(
                    "{marker} Tick mismatch: {left_label} tick {} vs {right_label} tick {}",
                    exp.tick.red(),
                    act.tick.red()
                );
            }
        }
        DivergenceKind::SequenceMismatch => {
            if let (Some(exp), Some(act)) = (&divergence.expected, &divergence.actual) {
                println!(
                    "{marker} Sequence mismatch at tick {}: {left_label} seq {} vs {right_label} seq {}",
                    exp.tick.red(),
                    exp.seq.red(),
                    act.seq.red()
                );
            }
        }
        DivergenceKind::EventMismatch => {
            if let (Some(exp), Some(act)) = (&divergence.expected, &divergence.actual) {
                println!(
                    "{marker} Event mismatch at tick {} seq {}",
                    exp.tick.red(),
                    exp.seq.red()
                );
                println!("    expected: {}", format_replay_event(&exp.event).yellow());
                println!("    actual:   {}", format_replay_event(&act.event).yellow());
            }
        }
        DivergenceKind::MissingActual => {
            if let Some(exp) = divergence.expected {
                println!(
                    "{marker} {right_label} stream ended before event tick {} seq {}",
                    exp.tick.red(),
                    exp.seq.red()
                );
            }
        }
        DivergenceKind::ExtraActual => {
            if let Some(act) = divergence.actual {
                println!(
                    "{marker} {right_label} has extra event at tick {} seq {}",
                    act.tick.red(),
                    act.seq.red()
                );
            }
        }
    }

    bail!("replay divergence detected")
}

fn format_replay_event(event: &scriptbots_core::ReplayEvent) -> String {
    match &event.kind {
        ReplayEventKind::BrainOutputs { outputs } => format!(
            "BrainOutputs(agent={:?}, len={})",
            event.agent_id,
            outputs.len()
        ),
        ReplayEventKind::Action {
            left_wheel,
            right_wheel,
            boost,
            spike_target,
            sound_level,
            give_intent,
        } => format!(
            "Action(agent={:?}, lw={:.3}, rw={:.3}, boost={}, spike={:?}, sound={:.3}, give={:.3})",
            event.agent_id, left_wheel, right_wheel, boost, spike_target, sound_level, give_intent
        ),
        ReplayEventKind::RngSample {
            scope,
            range_min,
            range_max,
            value,
        } => format!(
            "RngSample(scope={:?}, min={:.3}, max={:.3}, value={:.3})",
            scope, range_min, range_max, value
        ),
    }
}

fn print_event_counts(
    label: &str,
    counts: &HashMap<String, u64>,
    reference: Option<&HashMap<String, u64>>,
) {
    let keys = ["brain_outputs", "action", "rng_sample"];
    println!("  {}", label.cyan().bold());
    for key in keys {
        let value = counts.get(key).copied().unwrap_or(0);
        if let Some(baseline) = reference {
            let baseline_value = baseline.get(key).copied().unwrap_or(0);
            let delta = value as i64 - baseline_value as i64;
            let delta_fmt = format!("Δ {delta:+}");
            let delta_colored = if delta == 0 {
                format!("{}", delta_fmt.yellow())
            } else if delta > 0 {
                format!("{}", delta_fmt.green())
            } else {
                format!("{}", delta_fmt.red())
            };
            println!("    {:<14} {:>8} ({delta_colored})", key, value);
        } else {
            println!("    {:<14} {:>8}", key, value);
        }
    }
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
    use scriptbots_storage::{Storage, StoragePipeline};
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn with_env_lock<F: FnOnce()>(f: F) {
        let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock.lock().expect("env mutex poisoned");
        f();
    }

    #[test]
    fn layered_configs_apply_in_order() {
        let dir = tempdir().expect("tempdir");
        let base_path = dir.path().join("base.toml");
        fs::write(
            &base_path,
            r#"
persistence_interval = 120
rng_seed = 1337

[neuroflow]
enabled = true
hidden_layers = [64, 32]
activation = "Tanh"
"#,
        )
        .expect("write base layer");

        let overlay_path = dir.path().join("overlay.ron");
        fs::write(
            &overlay_path,
            "(history_capacity: 1024, neuroflow: (hidden_layers: [8, 4], activation: \"Sigmoid\"), world_width: 2048)",
        )
        .expect("write overlay layer");

        let base_config = ScriptBotsConfig {
            persistence_interval: 60,
            history_capacity: 600,
            ..ScriptBotsConfig::default()
        };

        let layered = apply_config_layers(base_config, &[base_path, overlay_path])
            .expect("apply config layers");

        assert_eq!(layered.persistence_interval, 120);
        assert_eq!(layered.history_capacity, 1024);
        assert_eq!(layered.world_width, 2048);
        assert_eq!(layered.rng_seed, Some(1337));
        assert!(layered.neuroflow.enabled);
        assert_eq!(layered.neuroflow.hidden_layers, vec![8, 4]);
        assert_eq!(
            layered.neuroflow.activation,
            NeuroflowActivationKind::Sigmoid
        );
    }

    #[test]
    fn headless_replay_matches_storage() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("replay.duckdb");
        let db_str = db_path.to_string_lossy().to_string();

        let config = ScriptBotsConfig {
            world_width: 600,
            world_height: 600,
            food_cell_size: 60,
            persistence_interval: 1,
            history_capacity: 128,
            rng_seed: Some(0xA1B2C3D4),
            ..ScriptBotsConfig::default()
        };

        {
            let pipeline = StoragePipeline::with_thresholds(&db_str, 1, 1, 1, 1).expect("pipeline");
            let mut world =
                WorldState::with_persistence(config.clone(), Box::new(pipeline)).expect("world");
            let keys = install_brains(&mut world);
            seed_agents(&mut world, &keys);
            for _ in 0..16 {
                world.step();
            }
        }

        let mut storage = Storage::open(&db_str).expect("open storage");
        let recorded_events = storage.load_replay_events().expect("load events");
        let max_tick = storage.max_tick().expect("max tick").unwrap_or(0);
        drop(storage);

        let replay = run_headless_simulation(&config, max_tick).expect("replay run");
        let diff = diff_event_stream(&recorded_events, &replay.events);
        assert!(diff.is_none(), "expected replay to match persisted events");
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
