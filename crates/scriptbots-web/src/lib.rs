#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{Context, Result, ensure};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use scriptbots_core::{
    AgentData, AgentId, BrainBinding, BrainRunner, Generation, INPUT_SIZE, OUTPUT_SIZE, Position,
    ScriptBotsConfig, Velocity, WorldState,
};
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use slotmap::Key;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SimHandle {
    inner: Rc<RefCell<Simulation>>,
}

struct Simulation {
    world: WorldState,
    spec: SimSpec,
}

#[derive(Clone)]
struct SimSpec {
    base_config: ScriptBotsConfig,
    initial_population: usize,
    seed: Option<u64>,
}

impl SimSpec {
    fn new(base_config: ScriptBotsConfig, initial_population: usize, seed: Option<u64>) -> Self {
        Self {
            base_config,
            initial_population,
            seed,
        }
    }

    fn with_seed(&self, seed: Option<u64>) -> Self {
        Self {
            seed,
            ..self.clone()
        }
    }

    fn config(&self) -> ScriptBotsConfig {
        let mut config = self.base_config.clone();
        config.rng_seed = self.seed;
        config.population_minimum = 0;
        config.population_spawn_interval = 0;
        config
    }
}

impl Simulation {
    fn new(spec: SimSpec) -> Result<Self> {
        let mut world = WorldState::new(spec.config())
            .context("failed to initialize ScriptBots world state")?;
        seed_agents(&mut world, spec.initial_population)?;
        Ok(Self { world, spec })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<()> {
        let spec = self.spec.with_seed(seed);
        let mut world = WorldState::new(spec.config())
            .context("failed to rebuild ScriptBots world state during reset")?;
        seed_agents(&mut world, spec.initial_population)?;
        self.world = world;
        self.spec = spec;
        Ok(())
    }

    fn tick(&mut self, steps: u32) -> SimulationSnapshot {
        for _ in 0..steps {
            self.world.step();
        }
        SimulationSnapshot::from_world(&self.world)
    }

    fn snapshot(&self) -> SimulationSnapshot {
        SimulationSnapshot::from_world(&self.world)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
struct InitOptions {
    seed: Option<u64>,
    population: usize,
    world_width: Option<u32>,
    world_height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<ScriptBotsConfig>,
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            seed: None,
            population: 64,
            world_width: None,
            world_height: None,
            config: None,
        }
    }
}

impl InitOptions {
    fn into_spec(self) -> SimSpec {
        let mut config = self.config.unwrap_or_else(ScriptBotsConfig::default);
        if let Some(width) = self.world_width {
            config.world_width = width;
        }
        if let Some(height) = self.world_height {
            config.world_height = height;
        }
        config.population_minimum = 0;
        config.population_spawn_interval = 0;

        SimSpec::new(config, self.population, self.seed)
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SimulationSnapshot {
    tick: u64,
    epoch: u64,
    world: SnapshotWorld,
    summary: SnapshotSummary,
    agents: Vec<AgentSnapshot>,
}

impl SimulationSnapshot {
    fn from_world(world: &WorldState) -> Self {
        let handles: Vec<AgentId> = world.agents().iter_handles().collect();
        let columns = world.agents().columns();
        let mut agents = Vec::with_capacity(handles.len());
        let mut total_energy = 0.0_f32;
        let mut total_health = 0.0_f32;

        for (dense_index, id) in handles.iter().enumerate() {
            let data = columns.snapshot(dense_index);
            let energy = world
                .agent_runtime(*id)
                .map(|runtime| runtime.energy)
                .unwrap_or_default();
            total_energy += energy;
            total_health += data.health;

            agents.push(AgentSnapshot {
                id: id.data().as_ffi(),
                position: [data.position.x, data.position.y],
                velocity: [data.velocity.vx, data.velocity.vy],
                heading: data.heading,
                health: data.health,
                energy,
                color: data.color,
                spike_length: data.spike_length,
                boost: data.boost,
            });
        }

        let summary = world
            .history()
            .last()
            .cloned()
            .map(|entry| SnapshotSummary {
                agent_count: entry.agent_count,
                births: entry.births,
                deaths: entry.deaths,
                total_energy: entry.total_energy,
                average_energy: entry.average_energy,
                average_health: entry.average_health,
            })
            .unwrap_or_else(|| {
                let agent_count = agents.len();
                let average_energy = if agent_count > 0 {
                    total_energy / agent_count as f32
                } else {
                    0.0
                };
                let average_health = if agent_count > 0 {
                    total_health / agent_count as f32
                } else {
                    0.0
                };
                SnapshotSummary {
                    agent_count,
                    births: 0,
                    deaths: 0,
                    total_energy,
                    average_energy,
                    average_health,
                }
            });

        let config = world.config();
        let world_info = SnapshotWorld {
            width: config.world_width,
            height: config.world_height,
            closed: world.is_closed(),
        };

        Self {
            tick: world.tick().0,
            epoch: world.epoch(),
            world: world_info,
            summary,
            agents,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SnapshotSummary {
    agent_count: usize,
    births: usize,
    deaths: usize,
    total_energy: f32,
    average_energy: f32,
    average_health: f32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentSnapshot {
    id: u64,
    position: [f32; 2],
    velocity: [f32; 2],
    heading: f32,
    health: f32,
    energy: f32,
    color: [f32; 3],
    spike_length: f32,
    boost: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SnapshotWorld {
    width: u32,
    height: u32,
    closed: bool,
}

#[wasm_bindgen]
impl SimHandle {
    #[wasm_bindgen(js_name = tick)]
    pub fn tick_js(&self, steps: u32) -> Result<JsValue, JsValue> {
        let mut simulation = self.inner.borrow_mut();
        let snapshot = simulation.tick(steps);
        to_value(&snapshot).map_err(js_error)
    }

    #[wasm_bindgen(js_name = snapshot)]
    pub fn snapshot_js(&self) -> Result<JsValue, JsValue> {
        let simulation = self.inner.borrow();
        to_value(&simulation.snapshot()).map_err(js_error)
    }

    #[wasm_bindgen(js_name = reset)]
    pub fn reset_js(&self, seed: Option<f64>) -> Result<(), JsValue> {
        let seed = normalize_seed(seed).map_err(js_error)?;
        let mut simulation = self.inner.borrow_mut();
        simulation.reset(seed).map_err(js_error)
    }
}

#[wasm_bindgen]
pub fn init_sim(config: JsValue) -> Result<SimHandle, JsValue> {
    let options = if config.is_null() || config.is_undefined() {
        InitOptions::default()
    } else {
        from_value::<InitOptions>(config).map_err(js_error)?
    };

    if options.population > 50_000 {
        return Err(js_error(
            "population must be 50,000 agents or fewer for browser builds",
        ));
    }

    let spec = options.into_spec();
    let simulation = Simulation::new(spec).map_err(js_error)?;
    Ok(SimHandle {
        inner: Rc::new(RefCell::new(simulation)),
    })
}

fn seed_agents(world: &mut WorldState, count: usize) -> Result<()> {
    if count == 0 {
        return Ok(());
    }

    let world_width = world.config().world_width as f32;
    let world_height = world.config().world_height as f32;

    for _ in 0..count {
        let (agent, brain_seed) = {
            let rng = world.rng();
            let x = rng.random_range(0.0..world_width);
            let y = rng.random_range(0.0..world_height);
            let heading = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
            let color = [
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..1.0),
            ];
            let seed = rng.random::<u64>();
            (
                AgentData::new(
                    Position::new(x, y),
                    Velocity::default(),
                    heading,
                    1.0,
                    color,
                    0.0,
                    false,
                    0,
                    Generation::default(),
                ),
                seed,
            )
        };

        let id = world.spawn_agent(agent);
        bind_brain(world, id, brain_seed)?;
    }

    Ok(())
}

fn normalize_seed(seed: Option<f64>) -> Result<Option<u64>> {
    let Some(value) = seed else {
        return Ok(None);
    };
    ensure!(value.is_finite(), "seed must be a finite number");
    ensure!(value >= 0.0, "seed must be non-negative");
    let truncated = value.floor();
    ensure!(
        truncated <= u64::MAX as f64,
        "seed must be representable as u64"
    );
    Ok(Some(truncated as u64))
}

fn js_error(err: impl std::fmt::Display) -> JsValue {
    JsError::new(&err.to_string()).into()
}

#[wasm_bindgen]
pub fn version() -> String {
    format!("scriptbots-web {}", env!("CARGO_PKG_VERSION"))
}

#[wasm_bindgen]
pub fn default_init_options() -> Result<JsValue, JsValue> {
    to_value(&InitOptions::default()).map_err(js_error)
}

fn bind_brain(world: &mut WorldState, agent: AgentId, seed: u64) -> Result<()> {
    let runtime = world
        .agent_runtime_mut(agent)
        .with_context(|| "agent runtime missing while binding wander brain")?;
    runtime.brain = BrainBinding::with_runner(Box::new(WanderBrain::new(seed)));
    Ok(())
}

struct WanderBrain {
    rng: SmallRng,
}

impl WanderBrain {
    fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl BrainRunner for WanderBrain {
    fn kind(&self) -> &'static str {
        "wasm.wander"
    }

    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        let mut outputs = [0.0; OUTPUT_SIZE];
        let left_eye = sensor(inputs, 0);
        let right_eye = sensor(inputs, 1);
        let drift = self.rng.random_range(-0.35..0.35);
        let forward = self.rng.random_range(0.45..0.9);
        let turn = (right_eye - left_eye) * 0.5 + drift;

        outputs[0] = clamp01(forward + turn);
        outputs[1] = clamp01(forward - turn);
        outputs[2] = clamp01(0.6 * sensor(inputs, 2) + 0.4 * self.rng.random::<f32>());
        outputs[3] = clamp01(0.6 * sensor(inputs, 3) + 0.4 * self.rng.random::<f32>());
        outputs[4] = clamp01(0.6 * sensor(inputs, 4) + 0.4 * self.rng.random::<f32>());
        outputs[5] = clamp01(sensor(inputs, 5) * 0.4 + self.rng.random_range(0.0..0.15));
        outputs[6] = if self.rng.random::<f32>() > 0.98 {
            1.0
        } else {
            0.0
        };
        outputs[7] = clamp01(sensor(inputs, 6) * 0.3);
        outputs[8] = clamp01(sensor(inputs, 7) * 0.3);
        outputs
    }
}

fn sensor(inputs: &[f32; INPUT_SIZE], idx: usize) -> f32 {
    inputs.get(idx).copied().unwrap_or_default()
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}
