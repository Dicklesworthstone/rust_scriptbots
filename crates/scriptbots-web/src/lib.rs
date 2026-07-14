#![cfg(any(target_arch = "wasm32", test))]

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{Context, Result, ensure};
use js_sys::Uint8Array;
use postcard::{from_bytes, to_allocvec};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use scriptbots_brain::MlpBrain;
use scriptbots_core::{
    AgentData, AgentId, BrainBinding, BrainRunner, DynamicWorldSnapshot as SimulationSnapshot,
    Generation, INPUT_SIZE, OUTPUT_SIZE, Position, ScriptBotsConfig, Velocity, WorldState,
};
#[cfg(test)]
use scriptbots_core::{
    DynamicAgentSnapshot as AgentSnapshot, DynamicSnapshotSummary as SnapshotSummary,
    DynamicSnapshotWorld as SnapshotWorld,
};
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SimHandle {
    inner: Rc<RefCell<Simulation>>,
}

struct Simulation {
    world: WorldState,
    spec: SimSpec,
    mlp_key: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum SnapshotFormat {
    #[default]
    Json,
    Binary,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum SeedStrategy {
    #[default]
    Wander,
    None,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum BrainPreset {
    Mlp,
}

#[derive(Clone)]
struct SimSpec {
    base_config: ScriptBotsConfig,
    initial_population: usize,
    seed: Option<u64>,
    snapshot_format: SnapshotFormat,
    seed_strategy: SeedStrategy,
    default_brain: Option<BrainPreset>,
}

impl SimSpec {
    fn new(
        base_config: ScriptBotsConfig,
        initial_population: usize,
        seed: Option<u64>,
        snapshot_format: SnapshotFormat,
        seed_strategy: SeedStrategy,
        default_brain: Option<BrainPreset>,
    ) -> Self {
        Self {
            base_config,
            initial_population,
            seed,
            snapshot_format,
            seed_strategy,
            default_brain,
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
        let world = WorldState::new(spec.config())
            .context("failed to initialize ScriptBots world state")?;
        let mut sim = Self {
            world,
            spec,
            mlp_key: None,
        };
        sim.seed_population()?;
        Ok(sim)
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<()> {
        let spec = self.spec.with_seed(seed);
        self.world = WorldState::new(spec.config())
            .context("failed to rebuild ScriptBots world state during reset")?;
        self.mlp_key = None;
        self.spec = spec;
        self.seed_population()?;
        Ok(())
    }

    fn tick(&mut self, steps: u32) -> Result<SimulationSnapshot> {
        for step_index in 0..steps {
            self.world.step().with_context(|| {
                format!(
                    "simulation failed during WASM step {} of {steps}",
                    step_index + 1
                )
            })?;
        }
        Ok(SimulationSnapshot::from_world(&self.world))
    }

    fn snapshot(&self) -> SimulationSnapshot {
        SimulationSnapshot::from_world(&self.world)
    }

    fn seed_population(&mut self) -> Result<()> {
        seed_agents(
            &mut self.world,
            self.spec.initial_population,
            self.spec.seed_strategy,
            self.spec.default_brain,
            &mut self.mlp_key,
        )
    }

    fn ensure_mlp_key(&mut self) -> Result<u64> {
        if let Some(key) = self.mlp_key {
            return Ok(key);
        }
        let key = self
            .world
            .brain_registry_mut()?
            .register(MlpBrain::KIND.as_str(), |rng| Ok(MlpBrain::runner(rng)));
        self.mlp_key = Some(key);
        Ok(key)
    }

    fn bind_brain_to_all(&mut self, key: u64) -> Result<()> {
        let handles: Vec<_> = self.world.agents().iter_handles().collect();
        for id in handles {
            ensure!(
                self.world
                    .bind_agent_brain(id, key)
                    .with_context(|| format!("failed to construct brain {key} for agent {id:?}"))?,
                "registered brain key {key} disappeared before binding agent {id:?}"
            );
        }
        Ok(())
    }

    fn apply_wander_to_all(&mut self) -> Result<()> {
        let handles: Vec<_> = self.world.agents().iter_handles().collect();
        for id in handles {
            let seed = {
                let rng = self.world.rng()?;
                rng.random::<u64>()
            };
            bind_wander_brain(&mut self.world, id, seed)?;
        }
        Ok(())
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    snapshot_format: Option<SnapshotFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed_strategy: Option<SeedStrategy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    default_brain: Option<BrainPreset>,
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            seed: None,
            population: 64,
            world_width: None,
            world_height: None,
            config: None,
            snapshot_format: None,
            seed_strategy: None,
            default_brain: None,
        }
    }
}

impl InitOptions {
    fn into_spec(self) -> SimSpec {
        let mut config = self.config.unwrap_or_default();
        if let Some(width) = self.world_width {
            config.world_width = width;
        }
        if let Some(height) = self.world_height {
            config.world_height = height;
        }
        config.population_minimum = 0;
        config.population_spawn_interval = 0;

        SimSpec::new(
            config,
            self.population,
            self.seed,
            self.snapshot_format.unwrap_or_default(),
            self.seed_strategy.unwrap_or_default(),
            self.default_brain,
        )
    }
}

#[wasm_bindgen]
impl SimHandle {
    #[wasm_bindgen(js_name = tick)]
    pub fn tick_js(&self, steps: u32) -> Result<JsValue, JsValue> {
        let mut simulation = self.inner.borrow_mut();
        let snapshot = simulation.tick(steps).map_err(js_error)?;
        encode_snapshot(&snapshot, simulation.spec.snapshot_format)
    }

    #[wasm_bindgen(js_name = snapshot)]
    pub fn snapshot_js(&self) -> Result<JsValue, JsValue> {
        let simulation = self.inner.borrow();
        let snapshot = simulation.snapshot();
        encode_snapshot(&snapshot, simulation.spec.snapshot_format)
    }

    #[wasm_bindgen(js_name = reset)]
    pub fn reset_js(&self, seed: Option<f64>) -> Result<(), JsValue> {
        let seed = normalize_seed(seed).map_err(js_error)?;
        let mut simulation = self.inner.borrow_mut();
        simulation.reset(seed).map_err(js_error)
    }

    #[wasm_bindgen(js_name = registerBrain)]
    pub fn register_brain_js(&self, kind: String) -> Result<(), JsValue> {
        let mut simulation = self.inner.borrow_mut();
        match kind.as_str() {
            "mlp" => {
                let key = simulation.ensure_mlp_key().map_err(js_error)?;
                simulation.spec.default_brain = Some(BrainPreset::Mlp);
                simulation.spec.seed_strategy = SeedStrategy::None;
                simulation.bind_brain_to_all(key).map_err(js_error)?;
                Ok(())
            }
            "wander" => {
                simulation.spec.default_brain = None;
                simulation.spec.seed_strategy = SeedStrategy::Wander;
                simulation.apply_wander_to_all().map_err(js_error)?;
                Ok(())
            }
            "none" => {
                simulation.spec.default_brain = None;
                simulation.spec.seed_strategy = SeedStrategy::None;
                Ok(())
            }
            other => Err(js_error(format!("unknown brain preset: {other}"))),
        }
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

fn seed_agents(
    world: &mut WorldState,
    count: usize,
    strategy: SeedStrategy,
    default_brain: Option<BrainPreset>,
    mlp_key_cache: &mut Option<u64>,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }

    let world_width = world.config().world_width as f32;
    let world_height = world.config().world_height as f32;

    let mlp_key = if matches!(default_brain, Some(BrainPreset::Mlp)) {
        Some(match mlp_key_cache {
            Some(key) => *key,
            None => {
                let key = world
                    .brain_registry_mut()?
                    .register(MlpBrain::KIND.as_str(), |rng| Ok(MlpBrain::runner(rng)));
                *mlp_key_cache = Some(key);
                key
            }
        })
    } else {
        None
    };

    for _ in 0..count {
        let (agent, wander_seed) = {
            let rng = world.rng()?;
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

        let id = world
            .try_spawn_agent(agent)
            .context("generated web agent must be finite")?;
        if let Some(key) = mlp_key {
            ensure!(
                world
                    .bind_agent_brain(id, key)
                    .with_context(|| format!("failed to construct MLP brain for agent {id:?}"))?,
                "registered MLP brain disappeared before binding agent {id:?}"
            );
        } else if matches!(strategy, SeedStrategy::Wander) {
            bind_wander_brain(world, id, wander_seed)?;
        }
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

fn encode_snapshot(
    snapshot: &SimulationSnapshot,
    format: SnapshotFormat,
) -> Result<JsValue, JsValue> {
    match format {
        SnapshotFormat::Json => to_value(snapshot).map_err(js_error),
        SnapshotFormat::Binary => {
            let bytes = to_allocvec(snapshot).map_err(|err| js_error(err.to_string()))?;
            Ok(Uint8Array::from(bytes.as_slice()).into())
        }
    }
}

#[wasm_bindgen]
pub fn version() -> String {
    format!("scriptbots-web {}", env!("CARGO_PKG_VERSION"))
}

#[wasm_bindgen]
pub fn default_init_options() -> Result<JsValue, JsValue> {
    to_value(&InitOptions::default()).map_err(js_error)
}

#[wasm_bindgen]
pub fn decode_snapshot_binary(bytes: &[u8]) -> Result<JsValue, JsValue> {
    let snapshot: SimulationSnapshot =
        from_bytes(bytes).map_err(|err| js_error(format!("postcard decode failed: {err}")))?;
    to_value(&snapshot).map_err(js_error)
}

fn bind_wander_brain(world: &mut WorldState, agent: AgentId, seed: u64) -> Result<()> {
    match world.try_update_agent_runtime(agent, |runtime| {
        runtime.brain = BrainBinding::with_runner(Box::new(WanderBrain::new(seed)));
    })? {
        true => Ok(()),
        false => Err(anyhow::anyhow!(
            "agent runtime missing while binding wander brain"
        )),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{BirthOrigin, ScriptBotsConfig};
    use std::sync::{Arc, Mutex};
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn postcard_snapshot_wire_golden_round_trips() {
        let snapshot = SimulationSnapshot {
            tick: 300,
            epoch: 2,
            world: SnapshotWorld {
                width: 640,
                height: 480,
                closed: true,
            },
            summary: SnapshotSummary {
                agent_count: 1,
                births: 2,
                deaths: 3,
                total_energy: 4.5,
                average_energy: 4.5,
                average_health: 0.75,
            },
            agents: vec![AgentSnapshot {
                id: 7,
                position: [1.25, -2.5],
                velocity: [0.5, -0.25],
                heading: 3.0,
                health: 1.5,
                energy: 4.5,
                color: [0.1, 0.2, 0.3],
                spike_length: 0.75,
                boost: true,
            }],
        };

        let encoded = to_allocvec(&snapshot).expect("encode postcard snapshot");
        assert_eq!(
            encoded,
            [
                172, 2, 2, 128, 5, 224, 3, 1, 1, 2, 3, 0, 0, 144, 64, 0, 0, 144, 64, 0, 0, 64, 63,
                1, 7, 0, 0, 160, 63, 0, 0, 32, 192, 0, 0, 0, 63, 0, 0, 128, 190, 0, 0, 64, 64, 0,
                0, 192, 63, 0, 0, 144, 64, 205, 204, 204, 61, 205, 204, 76, 62, 154, 153, 153, 62,
                0, 0, 64, 63, 1,
            ],
            "Postcard is positional: changing these bytes requires a versioned snapshot schema"
        );
        let decoded: SimulationSnapshot = from_bytes(&encoded).expect("decode postcard snapshot");
        assert_eq!(decoded.tick, snapshot.tick);
        assert_eq!(decoded.epoch, snapshot.epoch);
        assert_eq!(decoded.world.width, snapshot.world.width);
        assert_eq!(decoded.world.height, snapshot.world.height);
        assert_eq!(decoded.world.closed, snapshot.world.closed);
        assert_eq!(decoded.summary.agent_count, snapshot.summary.agent_count);
        assert_eq!(decoded.summary.births, snapshot.summary.births);
        assert_eq!(decoded.summary.deaths, snapshot.summary.deaths);
        assert_eq!(decoded.summary.total_energy, snapshot.summary.total_energy);
        assert_eq!(
            decoded.summary.average_energy,
            snapshot.summary.average_energy
        );
        assert_eq!(
            decoded.summary.average_health,
            snapshot.summary.average_health
        );
        assert_eq!(decoded.agents.len(), 1);
        assert_eq!(decoded.agents[0].id, snapshot.agents[0].id);
        assert_eq!(decoded.agents[0].position, snapshot.agents[0].position);
        assert_eq!(decoded.agents[0].velocity, snapshot.agents[0].velocity);
        assert_eq!(decoded.agents[0].heading, snapshot.agents[0].heading);
        assert_eq!(decoded.agents[0].health, snapshot.agents[0].health);
        assert_eq!(decoded.agents[0].energy, snapshot.agents[0].energy);
        assert_eq!(decoded.agents[0].color, snapshot.agents[0].color);
        assert_eq!(
            decoded.agents[0].spike_length,
            snapshot.agents[0].spike_length
        );
        assert_eq!(decoded.agents[0].boost, snapshot.agents[0].boost);
        assert_eq!(
            to_allocvec(&decoded).expect("re-encode postcard snapshot"),
            encoded
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn seeded_birth_records_capture_registry_and_runtime_brains_after_binding() {
        struct BirthCapture {
            batches: Arc<Mutex<Vec<scriptbots_core::PersistenceBatch>>>,
        }

        impl scriptbots_core::WorldPersistence for BirthCapture {
            fn on_tick(
                &mut self,
                payload: &scriptbots_core::PersistenceBatch,
            ) -> std::result::Result<(), scriptbots_core::PersistenceAdmissionError> {
                self.batches
                    .lock()
                    .expect("birth capture lock")
                    .push(payload.clone());
                Ok(())
            }
        }

        let capture = |strategy, default_brain| {
            let batches = Arc::new(Mutex::new(Vec::new()));
            let mut world = WorldState::with_persistence(
                ScriptBotsConfig {
                    world_width: 200,
                    world_height: 200,
                    food_cell_size: 20,
                    persistence_interval: 1,
                    population_minimum: 0,
                    population_spawn_interval: 0,
                    reproduction_attempt_chance: 0.0,
                    rng_seed: Some(0xB17A_0A5A),
                    ..ScriptBotsConfig::default()
                },
                Box::new(BirthCapture {
                    batches: Arc::clone(&batches),
                }),
            )
            .expect("world");
            let mut mlp_key = None;
            seed_agents(&mut world, 3, strategy, default_brain, &mut mlp_key)
                .expect("seed web agents");
            let random_stream = world.random_stream_state();
            world.step().expect("persist seeded lifecycle records");
            let births = batches
                .lock()
                .expect("birth capture lock")
                .last()
                .expect("first persistence cadence batch")
                .births
                .clone();
            (births, mlp_key, random_stream)
        };

        let (mlp_births, mlp_key, _) = capture(SeedStrategy::None, Some(BrainPreset::Mlp));
        let mlp_key = mlp_key.expect("MLP seeding registers its brain family");
        assert_eq!(mlp_births.len(), 3);
        for birth in mlp_births {
            assert_eq!(birth.origin, BirthOrigin::Seeded);
            assert_eq!(birth.brain_kind.as_deref(), Some(MlpBrain::KIND.as_str()));
            assert_eq!(birth.brain_key, Some(mlp_key));
        }

        let (wander_births, wander_key, wander_random_stream) = capture(SeedStrategy::Wander, None);
        assert!(wander_key.is_none());
        assert_eq!(wander_births.len(), 3);
        for birth in wander_births {
            assert_eq!(birth.origin, BirthOrigin::Seeded);
            assert_eq!(birth.brain_kind.as_deref(), Some("wasm.wander"));
            assert_eq!(birth.brain_key, None);
        }

        let (_, unbound_key, unbound_random_stream) = capture(SeedStrategy::None, None);
        assert!(unbound_key.is_none());
        assert_eq!(
            wander_random_stream, unbound_random_stream,
            "installing per-agent wander runners and refreshing their seeded origin records must not consume the world RNG"
        );
    }

    #[wasm_bindgen_test]
    fn wasm_harness_matches_native_world() {
        let cases = [
            (
                480_u32,
                360_u32,
                32_usize,
                24_u32,
                8102_u64,
                SeedStrategy::Wander,
                None,
            ),
            (
                640,
                480,
                48,
                40,
                1337,
                SeedStrategy::None,
                Some(BrainPreset::Mlp),
            ),
            (320, 320, 16, 64, 202501u64, SeedStrategy::Wander, None),
        ];

        for (width, height, population, ticks, seed, strategy, default_brain) in cases {
            let config = ScriptBotsConfig {
                world_width: width,
                world_height: height,
                // Ensure food_cell_size divides world dimensions evenly.
                food_cell_size: 20,
                rng_seed: Some(seed),
                ..ScriptBotsConfig::default()
            };

            // Native world execution
            let mut native_world = WorldState::new(config.clone()).expect("native world");
            let mut mlp_cache = None;
            seed_agents(
                &mut native_world,
                population,
                strategy,
                default_brain,
                &mut mlp_cache,
            )
            .expect("seed native world");
            for _ in 0..ticks {
                native_world
                    .step()
                    .expect("native conformance world should accept each simulation step");
            }
            let native_snapshot = SimulationSnapshot::from_world(&native_world);

            // WASM harness execution (same config + seed)
            let mut sim = Simulation::new(SimSpec::new(
                config,
                population,
                Some(seed),
                SnapshotFormat::Json,
                strategy,
                default_brain,
            ))
            .expect("sim");
            let wasm_snapshot = sim
                .tick(ticks)
                .expect("WASM conformance simulation should accept each step");

            assert_eq!(native_snapshot.tick, wasm_snapshot.tick);
            assert_eq!(
                native_snapshot.summary.agent_count,
                wasm_snapshot.summary.agent_count
            );
            assert_eq!(native_snapshot.agents.len(), wasm_snapshot.agents.len());

            assert_float_eq(
                native_snapshot.summary.total_energy,
                wasm_snapshot.summary.total_energy,
            );
            assert_float_eq(
                native_snapshot.summary.average_energy,
                wasm_snapshot.summary.average_energy,
            );
            assert_float_eq(
                native_snapshot.summary.average_health,
                wasm_snapshot.summary.average_health,
            );

            for (expected, actual) in native_snapshot
                .agents
                .iter()
                .zip(wasm_snapshot.agents.iter())
            {
                assert_float_eq(expected.position[0], actual.position[0]);
                assert_float_eq(expected.position[1], actual.position[1]);
                assert_float_eq(expected.velocity[0], actual.velocity[0]);
                assert_float_eq(expected.velocity[1], actual.velocity[1]);
                assert_float_eq(expected.heading, actual.heading);
                assert_float_eq(expected.health, actual.health);
                assert_float_eq(expected.energy, actual.energy);
                assert_float_eq(expected.spike_length, actual.spike_length);
                assert_eq!(expected.boost, actual.boost);
            }
        }
    }

    fn assert_float_eq(left: f32, right: f32) {
        let delta = (left - right).abs();
        assert!(
            delta <= 1e-5,
            "expected {:.6} ~= {:.6} (Δ={:.6})",
            left,
            right,
            delta
        );
    }
}
