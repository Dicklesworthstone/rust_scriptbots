#![cfg(any(target_arch = "wasm32", test))]

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{Context, Result, ensure};
use js_sys::Uint8Array;
use postcard::{from_bytes, to_allocvec};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use scriptbots_brain::{MlpBrain, mlp::MlpBrainFamily};
use scriptbots_core::rng_domains::RngDomain;
use scriptbots_core::{
    AgentData, AgentId, BrainBinding, BrainRunner, DynamicWorldSnapshot as SimulationSnapshot,
    Generation, INPUT_SIZE, OUTPUT_SIZE, Position, ScriptBotsConfig, Velocity, WorldState,
};
#[cfg(test)]
use scriptbots_core::{
    AgentUid, CoreBuildIdentityV0, DynamicAgentSnapshot as AgentSnapshot,
    DynamicSnapshotSummary as SnapshotSummary, DynamicSnapshotWorld as SnapshotWorld,
};
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use wasm_bindgen::prelude::*;

use scriptbots_runtime::{
    HostCore, HostCoreOptions, HostSessionId, ManualHostDriver, ManualInstant, PlaybackSnapshot,
};

#[wasm_bindgen]
pub struct SimHandle {
    inner: Rc<RefCell<Simulation>>,
}

struct Simulation {
    core: HostCore,
    spec: SimSpec,
    mlp_key: Option<u64>,
    now_nanos: u64,
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

    fn effective_seed(&self) -> Option<u64> {
        self.seed.or(self.base_config.rng_seed)
    }

    fn config(&self) -> ScriptBotsConfig {
        let mut config = self.base_config.clone();
        config.rng_seed = self.effective_seed();
        config.population_minimum = 0;
        config.population_spawn_interval = 0;
        config
    }
}

impl Simulation {
    fn new(spec: SimSpec) -> Result<Self> {
        let mut world = WorldState::new(spec.config())
            .context("failed to initialize ScriptBots world state")?;
        let mut mlp_key = None;
        seed_agents(
            &mut world,
            spec.initial_population,
            spec.seed_strategy,
            spec.default_brain,
            &mut mlp_key,
        )?;
        let session_id = HostSessionId::new(spec.effective_seed().unwrap_or(0));
        let options = HostCoreOptions {
            initial_playback: PlaybackSnapshot {
                paused: false,
                speed_multiplier: 1.0,
            },
            ..Default::default()
        };
        let mut core = HostCore::new(session_id, world, options)
            .map_err(|e| anyhow::anyhow!("failed to build HostCore for web: {e}"))?;
        core.drive(ManualInstant::from_nanos(0))
            .context("failed to establish the initial ScriptBots web time boundary")?;
        Ok(Self {
            core,
            spec,
            mlp_key,
            now_nanos: 0,
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<()> {
        let spec = self.spec.with_seed(seed);
        let mut world = WorldState::new(spec.config())
            .context("failed to rebuild ScriptBots world state during reset")?;
        self.mlp_key = None;
        seed_agents(
            &mut world,
            spec.initial_population,
            spec.seed_strategy,
            spec.default_brain,
            &mut self.mlp_key,
        )?;
        let session_id = HostSessionId::new(spec.effective_seed().unwrap_or(0));
        let options = HostCoreOptions {
            initial_playback: PlaybackSnapshot {
                paused: false,
                speed_multiplier: 1.0,
            },
            ..Default::default()
        };
        let mut core = HostCore::new(session_id, world, options)
            .map_err(|e| anyhow::anyhow!("failed to rebuild HostCore for web: {e}"))?;
        core.drive(ManualInstant::from_nanos(0))
            .context("failed to establish the reset ScriptBots web time boundary")?;
        self.core = core;
        self.spec = spec;
        self.now_nanos = 0;
        Ok(())
    }

    fn tick(&mut self, steps: u32) -> Result<SimulationSnapshot> {
        let period = self.core.tick_period_nanos();
        for step_index in 0..steps {
            self.now_nanos = self.now_nanos.saturating_add(period);
            let receipt = self
                .core
                .drive(ManualInstant::from_nanos(self.now_nanos))
                .with_context(|| {
                    format!(
                        "HostCore failed during WASM step {} of {steps}",
                        step_index + 1
                    )
                })?;
            ensure!(
                receipt.scientific_steps == 1,
                "HostCore completed {} scientific steps during requested WASM step {} of {steps}",
                receipt.scientific_steps,
                step_index + 1
            );
        }
        Ok(self.core.latest_snapshot().world.clone())
    }

    fn snapshot(&self) -> SimulationSnapshot {
        self.core.latest_snapshot().world.clone()
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
                simulation.spec.default_brain = Some(BrainPreset::Mlp);
                simulation.spec.seed_strategy = SeedStrategy::None;
                let seed = simulation.spec.seed;
                simulation.reset(seed).map_err(js_error)?;
                Ok(())
            }
            "wander" => {
                simulation.spec.default_brain = None;
                simulation.spec.seed_strategy = SeedStrategy::Wander;
                let seed = simulation.spec.seed;
                simulation.reset(seed).map_err(js_error)?;
                Ok(())
            }
            "none" => {
                simulation.spec.default_brain = None;
                simulation.spec.seed_strategy = SeedStrategy::None;
                let seed = simulation.spec.seed;
                simulation.reset(seed).map_err(js_error)?;
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
                    .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
                    .context("failed to register the versioned MLP brain family")?;
                *mlp_key_cache = Some(key);
                key
            }
        })
    } else {
        None
    };

    for _ in 0..count {
        let (agent, wander_seed) = {
            let rng = world.rng(RngDomain::Population)?;
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
    use scriptbots_core::{BirthOrigin, DYNAMIC_WORLD_SNAPSHOT_SCHEMA, ScriptBotsConfig};
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
                uid: AgentUid(11),
                position: [1.25, -2.5],
                velocity: [0.5, -0.25],
                heading: 3.0,
                health: 1.5,
                energy: 4.5,
                color: [0.1, 0.2, 0.3],
                spike_length: 0.75,
                boost: true,
                age: 12,
                generation: Generation(13),
                herbivore_tendency: 0.625,
                brain_key: Some(17),
            }],
        };

        let encoded = to_allocvec(&snapshot).expect("encode postcard snapshot");
        assert_eq!(
            encoded,
            [
                172, 2, 2, 128, 5, 224, 3, 1, 1, 2, 3, 0, 0, 144, 64, 0, 0, 144, 64, 0, 0, 64, 63,
                1, 7, 11, 0, 0, 160, 63, 0, 0, 32, 192, 0, 0, 0, 63, 0, 0, 128, 190, 0, 0, 64, 64,
                0, 0, 192, 63, 0, 0, 144, 64, 205, 204, 204, 61, 205, 204, 76, 62, 154, 153, 153,
                62, 0, 0, 64, 63, 1, 12, 13, 0, 0, 32, 63, 1, 17,
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
        assert_eq!(decoded.agents[0].uid, snapshot.agents[0].uid);
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
        assert_eq!(decoded.agents[0].age, snapshot.agents[0].age);
        assert_eq!(decoded.agents[0].generation, snapshot.agents[0].generation);
        assert_eq!(
            decoded.agents[0].herbivore_tendency,
            snapshot.agents[0].herbivore_tendency
        );
        assert_eq!(decoded.agents[0].brain_key, snapshot.agents[0].brain_key);
        assert_eq!(
            to_allocvec(&decoded).expect("re-encode postcard snapshot"),
            encoded
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn browser_demo_consumes_the_camel_case_snapshot_contract() {
        let source = include_str!("../web/main.js");
        for access in [
            "snapshot.summary.agentCount",
            "snapshot.summary.averageEnergy",
            "snapshot.summary.averageHealth",
        ] {
            assert!(
                source.contains(access),
                "browser demo must read the serialized camelCase access `{access}`"
            );
        }
        for stale_access in [
            "snapshot.summary.agent_count",
            "snapshot.summary.average_energy",
            "snapshot.summary.average_health",
        ] {
            assert!(
                !source.contains(stale_access),
                "browser demo still reads nonexistent snake_case access `{stale_access}`"
            );
        }
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
            let (mut world, mut persistence) = WorldState::with_persistence(
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
            if let Some(key) = mlp_key {
                let registry = world.brain_registry();
                assert!(
                    registry.is_protocol_family(key),
                    "WASM MLP seeding must use the versioned family adapter"
                );
                assert_eq!(
                    registry
                        .family(key)
                        .expect("protocol MLP key must expose its family adapter")
                        .family_id()
                        .as_str(),
                    "mlp-baseline"
                );
            }
            let random_streams = world.random_streams_checkpoint();
            persistence
                .step(&mut world)
                .expect("persist seeded lifecycle records");
            let births = batches
                .lock()
                .expect("birth capture lock")
                .last()
                .expect("first persistence cadence batch")
                .births
                .clone();
            (births, mlp_key, random_streams)
        };

        let (mlp_births, mlp_key, _) = capture(SeedStrategy::None, Some(BrainPreset::Mlp));
        let mlp_key = mlp_key.expect("MLP seeding registers its brain family");
        assert_eq!(mlp_births.len(), 3);
        for birth in mlp_births {
            assert_eq!(birth.origin, BirthOrigin::Seeded);
            assert_eq!(birth.brain_kind.as_deref(), Some(MlpBrain::KIND.as_str()));
            assert_eq!(birth.brain_key, Some(mlp_key));
        }

        let (wander_births, wander_key, wander_random_streams) =
            capture(SeedStrategy::Wander, None);
        assert!(wander_key.is_none());
        assert_eq!(wander_births.len(), 3);
        for birth in wander_births {
            assert_eq!(birth.origin, BirthOrigin::Seeded);
            assert_eq!(birth.brain_kind.as_deref(), Some("wasm.wander"));
            assert_eq!(birth.brain_key, None);
        }

        let (_, unbound_key, unbound_random_streams) = capture(SeedStrategy::None, None);
        assert!(unbound_key.is_none());
        assert_eq!(
            wander_random_streams, unbound_random_streams,
            "installing per-agent wander runners and refreshing their seeded origin records must not consume any world RNG domain"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn nested_config_seed_controls_world_and_host_session_when_top_level_is_omitted() {
        let simulation = Simulation::new(
            InitOptions {
                population: 0,
                config: Some(ScriptBotsConfig {
                    rng_seed: Some(11),
                    ..ScriptBotsConfig::default()
                }),
                ..InitOptions::default()
            }
            .into_spec(),
        )
        .expect("nested-seeded web simulation");

        assert_eq!(simulation.core.world().config().rng_seed, Some(11));
        assert_eq!(
            simulation.core.latest_snapshot().session_id,
            HostSessionId::new(11)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn top_level_seed_overrides_nested_config_for_world_and_host_session() {
        let simulation = Simulation::new(
            InitOptions {
                seed: Some(22),
                population: 0,
                config: Some(ScriptBotsConfig {
                    rng_seed: Some(11),
                    ..ScriptBotsConfig::default()
                }),
                ..InitOptions::default()
            }
            .into_spec(),
        )
        .expect("top-level-seeded web simulation");

        assert_eq!(simulation.core.world().config().rng_seed, Some(22));
        assert_eq!(
            simulation.core.latest_snapshot().session_id,
            HostSessionId::new(22)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn omitted_seeds_keep_entropy_config_with_zero_host_session_sentinel() {
        let simulation = Simulation::new(
            InitOptions {
                population: 0,
                ..InitOptions::default()
            }
            .into_spec(),
        )
        .expect("entropy-seeded web simulation");

        assert_eq!(simulation.core.world().config().rng_seed, None);
        assert_eq!(
            simulation.core.latest_snapshot().session_id,
            HostSessionId::new(0)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn reset_uses_explicit_seed_then_falls_back_to_nested_config_seed() {
        let mut simulation = Simulation::new(
            InitOptions {
                population: 3,
                config: Some(ScriptBotsConfig {
                    rng_seed: Some(11),
                    ..ScriptBotsConfig::default()
                }),
                seed_strategy: Some(SeedStrategy::None),
                ..InitOptions::default()
            }
            .into_spec(),
        )
        .expect("nested-seeded web simulation");
        let nested_seed_digest = simulation
            .core
            .world()
            .world_digest_v1()
            .expect("nested-seed digest");

        simulation.reset(Some(22)).expect("explicit-seed web reset");
        assert_eq!(simulation.core.world().config().rng_seed, Some(22));
        assert_eq!(
            simulation.core.latest_snapshot().session_id,
            HostSessionId::new(22)
        );

        simulation.reset(None).expect("nested-seed web reset");
        assert_eq!(simulation.core.world().config().rng_seed, Some(11));
        assert_eq!(
            simulation.core.latest_snapshot().session_id,
            HostSessionId::new(11)
        );
        assert_eq!(
            simulation
                .core
                .world()
                .world_digest_v1()
                .expect("repeated nested-seed digest"),
            nested_seed_digest,
            "reset without an explicit seed must reproduce the nested seeded world"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn parity_regression_tick_advances_every_requested_science_step() {
        let mut simulation = Simulation::new(
            InitOptions {
                population: 0,
                seed: Some(91),
                ..InitOptions::default()
            }
            .into_spec(),
        )
        .expect("seeded web simulation");

        assert_eq!(simulation.tick(0).expect("zero-step snapshot").tick, 0);
        assert_eq!(simulation.tick(1).expect("first requested step").tick, 1);
        assert_eq!(
            simulation
                .tick(2)
                .expect("two additional requested steps")
                .tick,
            3
        );

        simulation.reset(None).expect("reset web simulation");
        assert_eq!(simulation.tick(1).expect("first step after reset").tick, 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn reset_replaces_the_matching_persistence_session_and_remains_step_capable() {
        let mut simulation = Simulation::new(SimSpec::new(
            ScriptBotsConfig {
                world_width: 200,
                world_height: 200,
                food_cell_size: 20,
                persistence_interval: 1,
                population_minimum: 0,
                population_spawn_interval: 0,
                rng_seed: Some(11),
                ..ScriptBotsConfig::default()
            },
            3,
            Some(11),
            SnapshotFormat::Json,
            SeedStrategy::None,
            None,
        ))
        .expect("persistence-enabled web simulation");

        let before_reset = simulation.tick(1).expect("pre-reset session step");
        assert_eq!(before_reset.tick, 1);
        assert_eq!(
            simulation.core.persistence().last_admitted_tick(),
            Some(scriptbots_core::Tick(1))
        );

        simulation.reset(Some(22)).expect("reset web simulation");
        assert_eq!(simulation.core.world().tick().0, 0);
        assert_eq!(simulation.core.world().config().rng_seed, Some(22));
        assert_eq!(simulation.core.world().agent_count(), 3);
        assert_eq!(simulation.core.persistence().last_admitted_tick(), None);

        let after_reset = simulation
            .tick(2)
            .expect("replacement session must remain bound to the replacement world");
        assert_eq!(after_reset.tick, 2);
        assert_eq!(
            simulation.core.persistence().last_admitted_tick(),
            Some(scriptbots_core::Tick(2))
        );
        assert!(!simulation.core.persistence().has_pending_batch());
        assert!(simulation.core.persistence().fault().is_none());
    }

    /// One scenario of the parity matrix, used identically by the same-runtime
    /// determinism test, the native fixture generator, and the cross-architecture
    /// comparison test — one shared table so no side can silently drift onto a
    /// different scenario, which is exactly how the old test went wrong.
    #[derive(Clone, Copy)]
    struct ParityCase {
        width: u32,
        height: u32,
        population: usize,
        same_runtime_ticks: u32,
        seed: u64,
        strategy: SeedStrategy,
        default_brain: Option<BrainPreset>,
    }

    /// Cumulative checkpoint ticks captured in the committed fixture.
    const PARITY_CHECKPOINT_TICKS: [u32; 4] = [1, 8, 64, 150];
    /// Per-field float tolerance for parity comparison. A genuine libm divergence
    /// (native vs wasm sin/cos/atan2/powf) becomes a DOCUMENTED per-field policy in
    /// the fixture header, never a silent loosening of this constant.
    const PARITY_TOLERANCE: f32 = 1e-5;
    #[cfg(any(not(target_arch = "wasm32"), feature = "native-parity-fixture"))]
    const PARITY_FIXTURE_SCHEMA: &str = "scriptbots-web.native-parity.v2";

    fn parity_cases() -> [ParityCase; 3] {
        [
            ParityCase {
                width: 480,
                height: 360,
                population: 32,
                same_runtime_ticks: 24,
                seed: 8102,
                strategy: SeedStrategy::Wander,
                default_brain: None,
            },
            ParityCase {
                width: 640,
                height: 480,
                population: 48,
                same_runtime_ticks: 40,
                seed: 1337,
                strategy: SeedStrategy::None,
                default_brain: Some(BrainPreset::Mlp),
            },
            ParityCase {
                // 160 ticks is past the default population_spawn_interval (100): if
                // either side ever runs a raw config again instead of the shared
                // spec.config(), scheduled spawning fires on one side only and this
                // case goes red instead of hiding the mismatch below tick 100.
                // A 17x13 food grid and 17 agents exercise both four-lane SIMD
                // chunks and their scalar remainders.
                width: 340,
                height: 260,
                population: 17,
                same_runtime_ticks: 160,
                seed: 202_501,
                strategy: SeedStrategy::Wander,
                default_brain: None,
            },
        ]
    }

    fn parity_spec(case: ParityCase) -> SimSpec {
        SimSpec::new(
            ScriptBotsConfig {
                world_width: case.width,
                world_height: case.height,
                // Ensure food_cell_size divides world dimensions evenly.
                food_cell_size: 20,
                rng_seed: Some(case.seed),
                // Force the temperature SIMD/scalar fork to execute. The zero
                // comfort band makes every non-identical preference observable.
                temperature_discomfort_rate: 0.001,
                temperature_comfort_band: 0.0,
                ..ScriptBotsConfig::default()
            },
            case.population,
            Some(case.seed),
            SnapshotFormat::Json,
            case.strategy,
            case.default_brain,
        )
    }

    /// FNV-1a64 over the postcard encoding of the effective config, so a divergence
    /// report identifies WHICH scenario diverged from CI output alone.
    fn parity_config_hash(config: &ScriptBotsConfig) -> u64 {
        const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let bytes = to_allocvec(config).expect("config postcard-encodes");
        let mut hash = OFFSET_BASIS;
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PRIME);
        }
        hash
    }

    /// Compare two snapshots field by field; `Err` carries the FIRST divergence with
    /// tick, agent index, field name, both values, absolute delta, and the config
    /// hash — enough to diagnose from CI output alone.
    fn assert_snapshot_projection_is_exhaustively_named(snapshot: &SimulationSnapshot) {
        let SimulationSnapshot {
            tick: _,
            epoch: _,
            world,
            summary,
            agents,
        } = snapshot;
        let SnapshotWorld {
            width: _,
            height: _,
            closed: _,
        } = world;
        let SnapshotSummary {
            agent_count: _,
            births: _,
            deaths: _,
            total_energy: _,
            average_energy: _,
            average_health: _,
        } = summary;
        for agent in agents {
            let AgentSnapshot {
                id: _,
                uid: _,
                position: _,
                velocity: _,
                heading: _,
                health: _,
                energy: _,
                color: _,
                spike_length: _,
                boost: _,
                age: _,
                generation: _,
                herbivore_tendency: _,
                brain_key: _,
            } = agent;
        }
    }

    #[allow(clippy::too_many_lines)]
    fn compare_snapshots(
        config_hash: u64,
        tick: u32,
        expected: &SimulationSnapshot,
        actual: &SimulationSnapshot,
    ) -> std::result::Result<(), String> {
        assert_snapshot_projection_is_exhaustively_named(expected);
        assert_snapshot_projection_is_exhaustively_named(actual);
        let report = |subject: &str, field: &str, expected: f64, actual: f64| {
            format!(
                "first divergence at tick {tick}, {subject}, field `{field}`: \
                 expected {expected:.9}, actual {actual:.9}, |delta|={:.9}, \
                 config_hash={config_hash:#018x}",
                (expected - actual).abs()
            )
        };
        let float_field = |subject: &str,
                           field: &str,
                           expected: f32,
                           actual: f32|
         -> std::result::Result<(), String> {
            if (expected - actual).abs() <= PARITY_TOLERANCE {
                Ok(())
            } else {
                Err(report(
                    subject,
                    field,
                    f64::from(expected),
                    f64::from(actual),
                ))
            }
        };
        let exact = |subject: &str, field: &str, matches: bool, detail: String| {
            if matches {
                Ok(())
            } else {
                Err(format!(
                    "first divergence at tick {tick}, {subject}, field `{field}`: {detail}, \
                     config_hash={config_hash:#018x}"
                ))
            }
        };

        exact(
            "summary",
            "tick",
            expected.tick == u64::from(tick) && actual.tick == u64::from(tick),
            format!(
                "checkpoint {tick}, expected snapshot {}, actual snapshot {}",
                expected.tick, actual.tick
            ),
        )?;
        exact(
            "summary",
            "epoch",
            expected.epoch == actual.epoch,
            format!("expected {}, actual {}", expected.epoch, actual.epoch),
        )?;
        exact(
            "summary",
            "world.width",
            expected.world.width == actual.world.width,
            format!(
                "expected {}, actual {}",
                expected.world.width, actual.world.width
            ),
        )?;
        exact(
            "summary",
            "world.height",
            expected.world.height == actual.world.height,
            format!(
                "expected {}, actual {}",
                expected.world.height, actual.world.height
            ),
        )?;
        exact(
            "summary",
            "world.closed",
            expected.world.closed == actual.world.closed,
            format!(
                "expected {}, actual {}",
                expected.world.closed, actual.world.closed
            ),
        )?;
        exact(
            "summary",
            "agent_count",
            expected.summary.agent_count == actual.summary.agent_count,
            format!(
                "expected {}, actual {}",
                expected.summary.agent_count, actual.summary.agent_count
            ),
        )?;
        exact(
            "summary",
            "agents.len",
            expected.agents.len() == actual.agents.len(),
            format!(
                "expected {}, actual {}",
                expected.agents.len(),
                actual.agents.len()
            ),
        )?;
        exact(
            "summary",
            "births",
            expected.summary.births == actual.summary.births,
            format!(
                "expected {}, actual {}",
                expected.summary.births, actual.summary.births
            ),
        )?;
        exact(
            "summary",
            "deaths",
            expected.summary.deaths == actual.summary.deaths,
            format!(
                "expected {}, actual {}",
                expected.summary.deaths, actual.summary.deaths
            ),
        )?;
        float_field(
            "summary",
            "total_energy",
            expected.summary.total_energy,
            actual.summary.total_energy,
        )?;
        float_field(
            "summary",
            "average_energy",
            expected.summary.average_energy,
            actual.summary.average_energy,
        )?;
        float_field(
            "summary",
            "average_health",
            expected.summary.average_health,
            actual.summary.average_health,
        )?;

        for (index, (want, got)) in expected.agents.iter().zip(actual.agents.iter()).enumerate() {
            let subject = format!("agent[{index}] (uid {:?})", want.uid);
            exact(
                &subject,
                "id",
                want.id == got.id,
                format!("expected {:?}, actual {:?}", want.id, got.id),
            )?;
            exact(
                &subject,
                "uid",
                want.uid == got.uid,
                format!("expected {:?}, actual {:?}", want.uid, got.uid),
            )?;
            float_field(&subject, "position.x", want.position[0], got.position[0])?;
            float_field(&subject, "position.y", want.position[1], got.position[1])?;
            float_field(&subject, "velocity.x", want.velocity[0], got.velocity[0])?;
            float_field(&subject, "velocity.y", want.velocity[1], got.velocity[1])?;
            float_field(&subject, "heading", want.heading, got.heading)?;
            float_field(&subject, "health", want.health, got.health)?;
            float_field(&subject, "energy", want.energy, got.energy)?;
            float_field(
                &subject,
                "spike_length",
                want.spike_length,
                got.spike_length,
            )?;
            float_field(
                &subject,
                "herbivore_tendency",
                want.herbivore_tendency,
                got.herbivore_tendency,
            )?;
            exact(
                &subject,
                "color",
                want.color == got.color,
                format!("expected {:?}, actual {:?}", want.color, got.color),
            )?;
            exact(
                &subject,
                "boost",
                want.boost == got.boost,
                format!("expected {}, actual {}", want.boost, got.boost),
            )?;
            exact(
                &subject,
                "age",
                want.age == got.age,
                format!("expected {}, actual {}", want.age, got.age),
            )?;
            exact(
                &subject,
                "generation",
                want.generation == got.generation,
                format!(
                    "expected {:?}, actual {:?}",
                    want.generation, got.generation
                ),
            )?;
            exact(
                &subject,
                "brain_key",
                want.brain_key == got.brain_key,
                format!("expected {:?}, actual {:?}", want.brain_key, got.brain_key),
            )?;
        }
        Ok(())
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn parity_regression_comparator_covers_every_exported_non_agent_field() {
        let spec = parity_spec(parity_cases()[0]);
        let expected =
            SimulationSnapshot::from_world(&WorldState::new(spec.config()).expect("world"))
                .expect("fresh parity world has coherent agent companions");

        let mut actual = expected.clone();
        actual.epoch = actual.epoch.saturating_add(1);
        assert!(
            compare_snapshots(0xABCD, 0, &expected, &actual)
                .expect_err("epoch divergence must be rejected")
                .contains("epoch")
        );

        let mut actual = expected.clone();
        actual.world.width = actual.world.width.saturating_add(1);
        assert!(
            compare_snapshots(0xABCD, 0, &expected, &actual)
                .expect_err("world width divergence must be rejected")
                .contains("world.width")
        );

        let mut actual = expected.clone();
        actual.world.height = actual.world.height.saturating_add(1);
        assert!(
            compare_snapshots(0xABCD, 0, &expected, &actual)
                .expect_err("world height divergence must be rejected")
                .contains("world.height")
        );

        let mut actual = expected.clone();
        actual.world.closed = !actual.world.closed;
        assert!(
            compare_snapshots(0xABCD, 0, &expected, &actual)
                .expect_err("closed-world divergence must be rejected")
                .contains("world.closed")
        );

        let mut actual = expected.clone();
        actual.summary.births = actual.summary.births.saturating_add(1);
        assert!(
            compare_snapshots(0xABCD, 0, &expected, &actual)
                .expect_err("birth-count divergence must be rejected")
                .contains("births")
        );

        let mut actual = expected.clone();
        actual.summary.deaths = actual.summary.deaths.saturating_add(1);
        assert!(
            compare_snapshots(0xABCD, 0, &expected, &actual)
                .expect_err("death-count divergence must be rejected")
                .contains("deaths")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn parity_regression_divergence_report_identifies_the_first_agent_field() {
        let case = parity_cases()[0];
        let spec = parity_spec(case);
        let config_hash = parity_config_hash(&spec.config());
        let simulation = Simulation::new(spec).expect("seeded parity simulation");
        let expected = simulation.snapshot();
        let mut actual = expected.clone();
        actual.agents[0].position[0] += PARITY_TOLERANCE * 2.0;

        let error = compare_snapshots(config_hash, 0, &expected, &actual)
            .expect_err("injected position divergence must fail");
        let hash_fragment = format!("config_hash={config_hash:#018x}");
        for fragment in [
            "first divergence at tick 0",
            "agent[0]",
            "field `position.x`",
            "expected ",
            "actual ",
            "|delta|=",
            hash_fragment.as_str(),
        ] {
            assert!(
                error.contains(fragment),
                "diagnostic must include {fragment:?}: {error}"
            );
        }
    }

    /// Same-runtime determinism of the harness wrapper: a reference world built from
    /// the IDENTICAL effective config must match the wrapped simulation step for
    /// step. This is NOT cross-architecture parity — both sides execute in whatever
    /// runtime runs this test; true native-vs-wasm parity is the committed-fixture
    /// test below. (Renamed from `wasm_harness_matches_native_world`, which claimed
    /// parity this structure cannot prove while its reference side ALSO ran a
    /// different scenario: a raw config whose default population_spawn_interval=100
    /// never fired under the old <=64-tick cases, while the harness zeroed it.)
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn wasm_harness_matches_reference_world_in_the_same_runtime() {
        for case in parity_cases() {
            let spec = parity_spec(case);
            let effective_config = spec.config();
            let config_hash = parity_config_hash(&effective_config);

            let mut reference_world = WorldState::new(effective_config).expect("reference world");
            let mut mlp_cache = None;
            seed_agents(
                &mut reference_world,
                case.population,
                case.strategy,
                case.default_brain,
                &mut mlp_cache,
            )
            .expect("seed reference world");
            for _ in 0..case.same_runtime_ticks {
                reference_world
                    .step()
                    .expect("reference world should accept each simulation step");
            }
            let reference_snapshot = SimulationSnapshot::from_world(&reference_world)
                .expect("seeded parity world has coherent agent companions");

            let mut sim = Simulation::new(spec).expect("sim");
            let harness_snapshot = sim
                .tick(case.same_runtime_ticks)
                .expect("harness simulation should accept each step");

            let parity = compare_snapshots(
                config_hash,
                case.same_runtime_ticks,
                &reference_snapshot,
                &harness_snapshot,
            );
            assert!(
                parity.is_ok(),
                "same-runtime harness divergence (seed {}): {parity:?}",
                case.seed
            );
        }
    }

    /// The committed cross-architecture fixture: native-computed snapshots for the
    /// shared parity cases at the checkpoint ticks. Version the SCHEMA, not the
    /// bytes: if the snapshot projection changes (coordinate with bd-2z0.12.1),
    /// regenerate a new fixture version rather than mutating an older one in place.
    #[cfg(any(not(target_arch = "wasm32"), feature = "native-parity-fixture"))]
    #[derive(Clone, Serialize, Deserialize)]
    struct ParityFixtureV2 {
        schema: String,
        snapshot_schema: String,
        tolerance: f32,
        native_core_features: ParityCoreFeatureLaneV1,
        cases: Vec<ParityFixtureCase>,
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "native-parity-fixture"))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct ParityCoreFeatureLaneV1 {
        parallel: bool,
        simd_wide: bool,
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "native-parity-fixture"))]
    #[derive(Clone, Serialize, Deserialize)]
    struct ParityFixtureCase {
        width: u32,
        height: u32,
        population: u32,
        seed: u64,
        strategy: SeedStrategy,
        default_brain: Option<BrainPreset>,
        config_hash: u64,
        checkpoints: Vec<ParityCheckpoint>,
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "native-parity-fixture"))]
    #[derive(Clone, Serialize, Deserialize)]
    struct ParityCheckpoint {
        tick: u32,
        snapshot: SimulationSnapshot,
        food_digest: String,
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "native-parity-fixture"))]
    fn current_core_feature_lane() -> ParityCoreFeatureLaneV1 {
        let build = CoreBuildIdentityV0::current();
        ParityCoreFeatureLaneV1 {
            parallel: build.parallel,
            simd_wide: build.simd_wide,
        }
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "native-parity-fixture"))]
    fn expected_native_core_feature_lane() -> ParityCoreFeatureLaneV1 {
        ParityCoreFeatureLaneV1 {
            parallel: true,
            simd_wide: true,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_parity_oracle_uses_production_core_lane() {
        let build = CoreBuildIdentityV0::current();
        assert_ne!(
            build.target_arch, "wasm32",
            "the committed native parity oracle cannot be generated by wasm"
        );
        assert!(
            build.parallel,
            "the committed native parity oracle must exercise core's production `parallel` path: \
             {build:?}"
        );
        assert!(
            build.simd_wide,
            "the committed native parity oracle must exercise core's production `simd_wide` path: \
            {build:?}"
        );
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "native-parity-fixture"))]
    fn validate_parity_fixture(fixture: &ParityFixtureV2) -> std::result::Result<(), String> {
        if fixture.schema != PARITY_FIXTURE_SCHEMA {
            return Err(format!(
                "fixture schema mismatch: expected {PARITY_FIXTURE_SCHEMA}, actual {}",
                fixture.schema
            ));
        }
        if fixture.snapshot_schema != DYNAMIC_WORLD_SNAPSHOT_SCHEMA {
            return Err(format!(
                "snapshot schema mismatch: expected {DYNAMIC_WORLD_SNAPSHOT_SCHEMA}, actual {}",
                fixture.snapshot_schema
            ));
        }
        if fixture.tolerance.to_bits() != PARITY_TOLERANCE.to_bits() {
            return Err(format!(
                "fixture tolerance mismatch: expected {PARITY_TOLERANCE}, actual {}",
                fixture.tolerance
            ));
        }
        let expected_native_lane = expected_native_core_feature_lane();
        if fixture.native_core_features != expected_native_lane {
            return Err(format!(
                "fixture native core feature lane mismatch: expected \
                 {expected_native_lane:?}, actual {:?}",
                fixture.native_core_features
            ));
        }

        let cases = parity_cases();
        if fixture.cases.len() != cases.len() {
            return Err(format!(
                "fixture case count mismatch: expected {}, actual {}",
                cases.len(),
                fixture.cases.len()
            ));
        }

        for (case_index, (case, fixed)) in cases.into_iter().zip(fixture.cases.iter()).enumerate() {
            let expected_population =
                u32::try_from(case.population).expect("parity population fits u32");
            let expected_metadata = (
                case.width,
                case.height,
                expected_population,
                case.seed,
                case.strategy,
                case.default_brain,
            );
            let actual_metadata = (
                fixed.width,
                fixed.height,
                fixed.population,
                fixed.seed,
                fixed.strategy,
                fixed.default_brain,
            );
            if actual_metadata != expected_metadata {
                return Err(format!(
                    "fixture case {case_index} scenario mismatch: expected \
                     {expected_metadata:?}, actual {actual_metadata:?}"
                ));
            }

            let config_hash = parity_config_hash(&parity_spec(case).config());
            if fixed.config_hash != config_hash {
                return Err(format!(
                    "fixture case {case_index} config hash mismatch: expected \
                     {config_hash:#018x}, actual {:#018x}",
                    fixed.config_hash
                ));
            }

            let checkpoint_ticks = fixed
                .checkpoints
                .iter()
                .map(|checkpoint| checkpoint.tick)
                .collect::<Vec<_>>();
            if checkpoint_ticks.as_slice() != PARITY_CHECKPOINT_TICKS.as_slice() {
                return Err(format!(
                    "fixture case {case_index} checkpoint schedule mismatch: expected \
                     {PARITY_CHECKPOINT_TICKS:?}, actual {checkpoint_ticks:?}"
                ));
            }
            for (checkpoint_index, checkpoint) in fixed.checkpoints.iter().enumerate() {
                if checkpoint.snapshot.tick != u64::from(checkpoint.tick) {
                    return Err(format!(
                        "fixture case {case_index} checkpoint {checkpoint_index} is mislabeled: \
                         declared tick {}, embedded snapshot tick {}",
                        checkpoint.tick, checkpoint.snapshot.tick
                    ));
                }
                if checkpoint.food_digest.is_empty() {
                    return Err(format!(
                        "fixture case {case_index} checkpoint {checkpoint_index} has an empty \
                         food digest"
                    ));
                }
            }
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn build_parity_fixture() -> ParityFixtureV2 {
        let native_core_features = current_core_feature_lane();
        assert_eq!(
            native_core_features,
            expected_native_core_feature_lane(),
            "refusing to build a native parity fixture outside the production core feature lane"
        );
        let cases = parity_cases()
            .into_iter()
            .map(|case| {
                let spec = parity_spec(case);
                let config_hash = parity_config_hash(&spec.config());
                let mut sim = Simulation::new(spec).expect("fixture simulation");
                let mut checkpoints = Vec::with_capacity(PARITY_CHECKPOINT_TICKS.len());
                let mut current_tick = 0_u32;
                for target in PARITY_CHECKPOINT_TICKS {
                    let snapshot = sim
                        .tick(target - current_tick)
                        .expect("fixture simulation should accept each step");
                    assert_eq!(
                        snapshot.tick,
                        u64::from(target),
                        "fixture checkpoint must name the completed scientific tick"
                    );
                    let food_digest = sim
                        .core
                        .world()
                        .world_digest_v1()
                        .expect("fixture world must expose a valid scientific digest")
                        .food;
                    current_tick = target;
                    checkpoints.push(ParityCheckpoint {
                        tick: target,
                        snapshot,
                        food_digest,
                    });
                }
                ParityFixtureCase {
                    width: case.width,
                    height: case.height,
                    population: u32::try_from(case.population).expect("population fits u32"),
                    seed: case.seed,
                    strategy: case.strategy,
                    default_brain: case.default_brain,
                    config_hash,
                    checkpoints,
                }
            })
            .collect();
        ParityFixtureV2 {
            schema: PARITY_FIXTURE_SCHEMA.to_owned(),
            snapshot_schema: DYNAMIC_WORLD_SNAPSHOT_SCHEMA.to_owned(),
            tolerance: PARITY_TOLERANCE,
            native_core_features,
            cases,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parity_fixture_validation_rejects_ceremonial_or_stale_payloads() {
        let valid = build_parity_fixture();
        validate_parity_fixture(&valid).expect("fresh native fixture must validate");

        let mut empty = valid.clone();
        empty.cases[0].checkpoints.clear();
        assert!(
            validate_parity_fixture(&empty)
                .expect_err("empty checkpoint schedule must fail closed")
                .contains("checkpoint schedule")
        );

        let mut mislabeled = valid.clone();
        mislabeled.cases[0].checkpoints[0].snapshot.tick = 0;
        assert!(
            validate_parity_fixture(&mislabeled)
                .expect_err("mislabeled checkpoint must fail closed")
                .contains("mislabeled")
        );

        let mut scenario_drift = valid.clone();
        scenario_drift.cases[0].population = scenario_drift.cases[0].population.saturating_add(1);
        assert!(
            validate_parity_fixture(&scenario_drift)
                .expect_err("scenario metadata drift must fail closed")
                .contains("scenario mismatch")
        );

        let mut schema_drift = valid;
        schema_drift.snapshot_schema = "scriptbots.dynamic-world-snapshot.stale".to_owned();
        assert!(
            validate_parity_fixture(&schema_drift)
                .expect_err("snapshot schema drift must fail closed")
                .contains("snapshot schema mismatch")
        );

        let mut scalar_oracle = build_parity_fixture();
        scalar_oracle.native_core_features.simd_wide = false;
        assert!(
            validate_parity_fixture(&scalar_oracle)
                .expect_err("a scalar native oracle must fail closed")
                .contains("native core feature lane mismatch")
        );

        let mut serial_oracle = build_parity_fixture();
        serial_oracle.native_core_features.parallel = false;
        assert!(
            validate_parity_fixture(&serial_oracle)
                .expect_err("a non-production native oracle must fail closed")
                .contains("native core feature lane mismatch")
        );

        let mut missing_food = build_parity_fixture();
        missing_food.cases[0].checkpoints[0].food_digest.clear();
        assert!(
            validate_parity_fixture(&missing_food)
                .expect_err("missing internal food-state evidence must fail closed")
                .contains("empty food digest")
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn committed_native_parity_fixture_matches_fresh_generation() {
        let committed: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/native_parity_v2.postcard"
        ));
        let decoded: ParityFixtureV2 =
            postcard::from_bytes(committed).expect("decode committed native fixture");
        validate_parity_fixture(&decoded)
            .expect("committed native fixture must satisfy the full fixture contract");

        let regenerated =
            to_allocvec(&build_parity_fixture()).expect("encode freshly generated fixture");
        assert!(
            committed == regenerated.as_slice(),
            "committed native parity fixture is stale (committed {} bytes, regenerated {} bytes); \
             regenerate it deliberately with the guarded native_parity_fixture_generator",
            committed.len(),
            regenerated.len()
        );
    }

    /// Regenerate the committed cross-architecture fixture. NATIVE ONLY. Legitimate
    /// only after an INTENTIONAL semantic change, regenerated in the same commit,
    /// with the digest-change rationale in the commit message (same discipline as a
    /// characterization re-pin). Through the repo's DSR lane, the underlying
    /// command is:
    ///
    /// ```text
    /// SCRIPTBOTS_WEB_WRITE_PARITY_FIXTURE=1 \
    ///   cargo test -p scriptbots-web --lib \
    ///   tests::native_parity_fixture_generator -- --exact --ignored
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "fixture writer; run explicitly with SCRIPTBOTS_WEB_WRITE_PARITY_FIXTURE=1"]
    fn native_parity_fixture_generator() {
        assert_eq!(
            std::env::var("SCRIPTBOTS_WEB_WRITE_PARITY_FIXTURE").as_deref(),
            Ok("1"),
            "fixture generation requires SCRIPTBOTS_WEB_WRITE_PARITY_FIXTURE=1"
        );
        // Determinism proof before anything is written: two independent builds of
        // the fixture must be byte-identical, or the fixture would encode luck.
        let first_fixture = build_parity_fixture();
        validate_parity_fixture(&first_fixture).expect("generated fixture must validate");
        let first = to_allocvec(&first_fixture).expect("encode fixture");
        let second_fixture = build_parity_fixture();
        validate_parity_fixture(&second_fixture)
            .expect("independently generated fixture must validate");
        let second = to_allocvec(&second_fixture).expect("encode fixture again");
        assert_eq!(
            first, second,
            "fixture generation is not deterministic; refusing to write"
        );
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/native_parity_v2.postcard");
        std::fs::create_dir_all(path.parent().expect("fixture parent"))
            .expect("create fixture directory");
        std::fs::write(&path, &first).expect("write fixture");
        eprintln!(
            "wrote {} ({} bytes, schema {PARITY_FIXTURE_SCHEMA})",
            path.display(),
            first.len()
        );
    }

    /// TRUE cross-architecture parity: wasm-computed snapshots against the committed
    /// NATIVE-generated fixture. The authoritative browser lane enables the
    /// `native-parity-fixture` feature explicitly; ordinary library builds do not
    /// carry test-only fixture bytes. If a genuine libm divergence appears, the
    /// deliverable is a documented per-field tolerance policy recorded in the
    /// fixture header — not a silent loosening.
    #[cfg(all(target_arch = "wasm32", feature = "native-parity-fixture"))]
    #[wasm_bindgen_test]
    fn wasm_matches_committed_native_fixture() {
        let bytes: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/native_parity_v2.postcard"
        ));
        let fixture: ParityFixtureV2 =
            postcard::from_bytes(bytes).expect("decode committed native fixture");
        validate_parity_fixture(&fixture)
            .expect("committed native parity fixture must satisfy the full contract");
        let wasm_build = CoreBuildIdentityV0::current();
        assert_eq!(
            wasm_build.target_arch, "wasm32",
            "the fixture consumer must execute in a real wasm32 runtime: {wasm_build:?}"
        );
        assert!(
            !wasm_build.parallel && !wasm_build.simd_wide,
            "the wasm fixture consumer must exercise core's scalar, single-thread path: \
             {wasm_build:?}"
        );
        let cases = parity_cases();

        for (case, fixed) in cases.into_iter().zip(fixture.cases.iter()) {
            let spec = parity_spec(case);
            let config_hash = parity_config_hash(&spec.config());
            let mut sim = Simulation::new(spec).expect("wasm parity simulation");
            let mut current_tick = 0_u32;
            for checkpoint in &fixed.checkpoints {
                let steps = checkpoint
                    .tick
                    .checked_sub(current_tick)
                    .expect("validated parity checkpoints are strictly increasing");
                let snapshot = sim
                    .tick(steps)
                    .expect("wasm parity simulation should accept each step");
                #[cfg(feature = "native-parity-fault-injection")]
                let snapshot = {
                    let mut perturbed = snapshot;
                    if checkpoint.tick == PARITY_CHECKPOINT_TICKS[0] {
                        perturbed.agents[0].position[0] += PARITY_TOLERANCE * 2.0;
                    }
                    perturbed
                };
                current_tick = checkpoint.tick;
                let parity = compare_snapshots(
                    config_hash,
                    checkpoint.tick,
                    &checkpoint.snapshot,
                    &snapshot,
                );
                assert!(
                    parity.is_ok(),
                    "NATIVE-vs-WASM divergence (seed {}): {parity:?}",
                    case.seed
                );
                let wasm_food_digest = sim
                    .core
                    .world()
                    .world_digest_v1()
                    .expect("wasm parity world must expose a valid scientific digest")
                    .food;
                assert_eq!(
                    checkpoint.food_digest, wasm_food_digest,
                    "NATIVE-vs-WASM divergence at tick {} field `food_digest` \
                     expected {}, actual {}, config_hash={config_hash:#018x}",
                    checkpoint.tick, checkpoint.food_digest, wasm_food_digest
                );
            }
        }
    }
}
