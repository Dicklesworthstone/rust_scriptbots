//! Core types shared across the ScriptBots workspace.

use ordered_float::OrderedFloat;
use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scriptbots_index::{NeighborhoodIndex, UniformGridIndex};
use serde::{Deserialize, Serialize};
use slotmap::{SecondaryMap, SlotMap, new_key_type};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use thiserror::Error;

#[cfg(feature = "parallel")]
macro_rules! collect_handles {
    ($handles:expr, |$idx:ident, $handle:pat_param| $body:expr) => {{
        ($handles)
            .par_iter()
            .enumerate()
            .map(|($idx, $handle)| $body)
            .collect::<Vec<_>>()
    }};
}

#[cfg(not(feature = "parallel"))]
macro_rules! collect_handles {
    ($handles:expr, |$idx:ident, $handle:pat_param| $body:expr) => {{
        ($handles)
            .iter()
            .enumerate()
            .map(|($idx, $handle)| $body)
            .collect::<Vec<_>>()
    }};
}

new_key_type! {
    /// Stable handle for agents backed by a generational slot map.
    pub struct AgentId;
}

/// Convenience alias for associating side data with agents.
pub type AgentMap<T> = SecondaryMap<AgentId, T>;

/// Number of sensor inputs wired into each agent brain.
pub const INPUT_SIZE: usize = 25;
/// Number of control outputs produced by each agent brain.
pub const OUTPUT_SIZE: usize = 9;
/// Number of directional eyes each agent possesses.
pub const NUM_EYES: usize = 4;

const FULL_TURN: f32 = std::f32::consts::TAU;
const HALF_TURN: f32 = std::f32::consts::PI;
const BLOOD_HALF_FOV: f32 = std::f32::consts::PI * 0.375; // 3π/8

fn wrap_signed_angle(mut angle: f32) -> f32 {
    if angle.is_nan() {
        return 0.0;
    }
    while angle <= -HALF_TURN {
        angle += FULL_TURN;
    }
    while angle > HALF_TURN {
        angle -= FULL_TURN;
    }
    angle
}

fn wrap_unsigned_angle(mut angle: f32) -> f32 {
    if angle.is_nan() {
        return 0.0;
    }
    while angle < 0.0 {
        angle += FULL_TURN;
    }
    while angle >= FULL_TURN {
        angle -= FULL_TURN;
    }
    angle
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn toroidal_delta(a: f32, b: f32, extent: f32) -> f32 {
    let mut delta = a - b;
    let half = extent * 0.5;
    if delta > half {
        delta -= extent;
    } else if delta < -half {
        delta += extent;
    }
    delta
}

fn angle_to(dx: f32, dy: f32) -> f32 {
    dy.atan2(dx)
}

fn angle_difference(a: f32, b: f32) -> f32 {
    let diff = wrap_signed_angle(a - b);
    diff.abs()
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn sample_temperature(config: &ScriptBotsConfig, x: f32) -> f32 {
    if config.world_width == 0 {
        return 0.5;
    }
    let width = config.world_width as f32;
    if width <= f32::EPSILON {
        return 0.5;
    }
    let normalized = (x / width).rem_euclid(1.0);
    let distance = ((normalized - 0.5).abs() * 2.0).clamp(0.0, 1.0);
    let exponent = config.temperature_gradient_exponent.max(f32::EPSILON);
    distance.powf(exponent).clamp(0.0, 1.0)
}

fn temperature_discomfort(env_temperature: f32, preference: f32) -> f32 {
    (env_temperature - clamp01(preference)).abs()
}

/// Per-agent mutation rate configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct MutationRates {
    pub primary: f32,
    pub secondary: f32,
}

impl Default for MutationRates {
    fn default() -> Self {
        Self {
            primary: 0.003,
            secondary: 0.05,
        }
    }
}

/// Trait modifiers affecting sense organs and physiology.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TraitModifiers {
    pub smell: f32,
    pub sound: f32,
    pub hearing: f32,
    pub eye: f32,
    pub blood: f32,
}

impl Default for TraitModifiers {
    fn default() -> Self {
        Self {
            smell: 0.3,
            sound: 0.4,
            hearing: 1.0,
            eye: 1.5,
            blood: 1.5,
        }
    }
}

/// Highlight shown around an agent in the UI.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct IndicatorState {
    pub intensity: f32,
    pub color: [f32; 3],
}

impl Default for IndicatorState {
    fn default() -> Self {
        Self {
            intensity: 0.0,
            color: [0.0, 0.0, 0.0],
        }
    }
}

/// Selection state applied by user interaction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum SelectionState {
    #[default]
    None,
    Hovered,
    Selected,
}

/// Per-tick combat markers used by UI, analytics, and audio layers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct CombatEventFlags {
    pub spike_attacker: bool,
    pub spike_victim: bool,
    pub hit_carnivore: bool,
    pub hit_herbivore: bool,
    pub was_spiked_by_carnivore: bool,
    pub was_spiked_by_herbivore: bool,
}

/// Runtime brain attachment tracking.
#[derive(Serialize, Deserialize)]
pub struct BrainBinding {
    #[serde(skip)]
    runner: Option<Box<dyn BrainRunner>>,
    registry_key: Option<u64>,
    kind: Option<String>,
}

impl Default for BrainBinding {
    fn default() -> Self {
        Self::unbound()
    }
}

impl Clone for BrainBinding {
    fn clone(&self) -> Self {
        Self {
            runner: None,
            registry_key: self.registry_key,
            kind: self.kind.clone(),
        }
    }
}

impl fmt::Debug for BrainBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BrainBinding")
            .field("registry_key", &self.registry_key)
            .field("kind", &self.kind)
            .finish()
    }
}

impl BrainBinding {
    /// Construct an unbound brain attachment.
    #[must_use]
    pub fn unbound() -> Self {
        Self {
            runner: None,
            registry_key: None,
            kind: None,
        }
    }

    /// Attach a brain runner produced outside the registry.
    #[must_use]
    pub fn with_runner(runner: Box<dyn BrainRunner>) -> Self {
        let kind = Some(runner.kind().to_string());
        Self {
            runner: Some(runner),
            registry_key: None,
            kind,
        }
    }

    /// Instantiate a brain from the registry and bind it to the agent.
    #[must_use]
    pub fn from_registry(
        registry: &BrainRegistry,
        rng: &mut dyn RngCore,
        key: u64,
    ) -> Option<Self> {
        let runner = registry.spawn(rng, key)?;
        let kind = registry.kind(key).map(str::to_string);
        Some(Self {
            runner: Some(runner),
            registry_key: Some(key),
            kind,
        })
    }

    /// Return the registry key, if any, associated with this binding.
    #[must_use]
    pub const fn registry_key(&self) -> Option<u64> {
        self.registry_key
    }

    /// Return the brain identifier when available.
    #[must_use]
    pub fn kind(&self) -> Option<&str> {
        self.kind.as_deref()
    }

    /// Whether a brain runner is currently attached.
    #[must_use]
    pub const fn is_bound(&self) -> bool {
        self.runner.is_some()
    }

    /// Produce a short descriptor suitable for persistence logs.
    #[must_use]
    pub fn describe(&self) -> Cow<'_, str> {
        if let Some(key) = self.registry_key {
            Cow::Owned(format!("registry:{key}"))
        } else if let Some(kind) = &self.kind {
            Cow::Borrowed(kind.as_str())
        } else {
            Cow::Borrowed("unbound")
        }
    }

    /// Evaluate the brain if one is bound, returning the outputs.
    #[must_use]
    pub fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> Option<[f32; OUTPUT_SIZE]> {
        self.runner.as_mut().map(|brain| brain.tick(inputs))
    }
}

/// Thin trait object used to drive brain evaluations without coupling to concrete brain crates.
pub trait BrainRunner: Send + Sync {
    /// Static identifier of the brain implementation.
    fn kind(&self) -> &'static str;

    /// Evaluate outputs for the provided sensors.
    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE];
}

type BrainSpawner = Box<dyn Fn(&mut dyn RngCore) -> Box<dyn BrainRunner> + Send + Sync + 'static>;

struct BrainEntry {
    kind: Cow<'static, str>,
    spawner: BrainSpawner,
}

/// Registry owning brain runners keyed by opaque handles.
#[derive(Default)]
pub struct BrainRegistry {
    next_key: u64,
    entries: HashMap<u64, BrainEntry>,
}

impl std::fmt::Debug for BrainRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrainRegistry")
            .field("next_key", &self.next_key)
            .field("entry_count", &self.entries.len())
            .finish()
    }
}

impl BrainRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new brain factory, returning its registry key.
    pub fn register<F>(&mut self, kind: impl Into<Cow<'static, str>>, factory: F) -> u64
    where
        F: Fn(&mut dyn RngCore) -> Box<dyn BrainRunner> + Send + Sync + 'static,
    {
        let key = self.next_key;
        self.next_key += 1;
        self.entries.insert(
            key,
            BrainEntry {
                kind: kind.into(),
                spawner: Box::new(factory),
            },
        );
        key
    }

    /// Removes a brain factory from the registry.
    pub fn unregister(&mut self, key: u64) -> bool {
        self.entries.remove(&key).is_some()
    }

    /// Instantiate a new brain runner using the factory referenced by `key`.
    pub fn spawn(&self, rng: &mut dyn RngCore, key: u64) -> Option<Box<dyn BrainRunner>> {
        self.entries.get(&key).map(|entry| (entry.spawner)(rng))
    }

    /// Retrieve the descriptive identifier associated with a registry entry.
    #[must_use]
    pub fn kind(&self, key: u64) -> Option<&str> {
        self.entries.get(&key).map(|entry| entry.kind.as_ref())
    }

    /// Returns whether a key is registered.
    #[must_use]
    pub fn contains(&self, key: u64) -> bool {
        self.entries.contains_key(&key)
    }
}

/// Runtime data associated with an agent beyond the dense SoA columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRuntime {
    pub energy: f32,
    pub reproduction_counter: f32,
    pub herbivore_tendency: f32,
    pub mutation_rates: MutationRates,
    pub trait_modifiers: TraitModifiers,
    pub clocks: [f32; 2],
    pub eye_fov: [f32; NUM_EYES],
    pub eye_direction: [f32; NUM_EYES],
    pub sound_multiplier: f32,
    pub give_intent: f32,
    pub sensors: [f32; INPUT_SIZE],
    pub outputs: [f32; OUTPUT_SIZE],
    pub indicator: IndicatorState,
    pub selection: SelectionState,
    pub combat: CombatEventFlags,
    pub food_delta: f32,
    pub spiked: bool,
    pub hybrid: bool,
    pub sound_output: f32,
    pub temperature_preference: f32,
    pub brain: BrainBinding,
    pub lineage: [Option<AgentId>; 2],
    pub mutation_log: Vec<String>,
}

impl Default for AgentRuntime {
    fn default() -> Self {
        Self {
            energy: 1.0,
            reproduction_counter: 0.0,
            herbivore_tendency: 0.5,
            mutation_rates: MutationRates::default(),
            trait_modifiers: TraitModifiers::default(),
            clocks: [50.0, 50.0],
            eye_fov: [1.0; NUM_EYES],
            eye_direction: [0.0; NUM_EYES],
            sound_multiplier: 1.0,
            give_intent: 0.0,
            sensors: [0.0; INPUT_SIZE],
            outputs: [0.0; OUTPUT_SIZE],
            indicator: IndicatorState::default(),
            selection: SelectionState::None,
            combat: CombatEventFlags::default(),
            food_delta: 0.0,
            spiked: false,
            hybrid: false,
            sound_output: 0.0,
            temperature_preference: 0.5,
            brain: BrainBinding::default(),
            lineage: [None, None],
            mutation_log: Vec::new(),
        }
    }
}

impl AgentRuntime {
    /// Sample randomized sensory parameters matching the legacy ScriptBots defaults.
    pub fn new_random(rng: &mut dyn RngCore) -> Self {
        let mut runtime = Self::default();
        runtime.randomize_spawn(rng);
        runtime
    }

    /// Randomize spawn-time traits and sensory configuration.
    pub fn randomize_spawn(&mut self, rng: &mut dyn RngCore) {
        self.herbivore_tendency = rng.random_range(0.0..1.0);
        self.mutation_rates.primary = rng.random_range(0.001..0.005);
        self.mutation_rates.secondary = rng.random_range(0.03..0.07);
        self.trait_modifiers.smell = rng.random_range(0.1..0.5);
        self.trait_modifiers.sound = rng.random_range(0.2..0.6);
        self.trait_modifiers.hearing = rng.random_range(0.7..1.3);
        self.trait_modifiers.eye = rng.random_range(1.0..3.0);
        self.trait_modifiers.blood = rng.random_range(1.0..3.0);
        self.clocks[0] = rng.random_range(5.0..100.0);
        self.clocks[1] = rng.random_range(5.0..100.0);
        for fov in &mut self.eye_fov {
            *fov = rng.random_range(0.5..2.0);
        }
        for dir in &mut self.eye_direction {
            *dir = rng.random_range(0.0..FULL_TURN);
        }
        self.temperature_preference = rng.random_range(0.0..1.0);
        self.lineage = [None, None];
    }

    fn push_gene_log(&mut self, capacity: usize, message: impl Into<String>) {
        if capacity == 0 {
            return;
        }
        let entry = message.into();
        if entry.is_empty() {
            return;
        }
        if self.mutation_log.len() + 1 > capacity {
            let remove = self.mutation_log.len() + 1 - capacity;
            self.mutation_log.drain(0..remove);
        }
        self.mutation_log.push(entry);
    }

    fn log_change(&mut self, capacity: usize, label: &str, before: f32, after: f32) {
        if (after - before).abs() > 1e-4 {
            self.push_gene_log(capacity, format!("{label}: {:.3}->{:.3}", before, after));
        }
    }
}

/// Combined snapshot of dense columns and runtime metadata for a single agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub id: AgentId,
    pub data: AgentData,
    pub runtime: AgentRuntime,
}

#[derive(Debug, Clone)]
struct ActuationDelta {
    heading: f32,
    velocity: Velocity,
    position: Position,
    health_delta: f32,
}

#[derive(Debug, Clone, Default)]
struct ActuationResult {
    delta: Option<ActuationDelta>,
    energy: f32,
    color: [f32; 3],
    spike_length: f32,
    sound_level: f32,
    give_intent: f32,
    spiked: bool,
}

#[derive(Debug, Default)]
struct CombatResult {
    energy: f32,
    attacker_carnivore: bool,
    hit_carnivore: bool,
    hit_herbivore: bool,
    total_damage: f32,
    hits: Vec<CombatHit>,
}

#[derive(Debug, Clone, Copy, Default)]
struct CombatHit {
    target_idx: usize,
    damage: f32,
    attacker_carnivore: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct DamageBucket {
    total: f32,
    carnivore: f32,
    herbivore: f32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SpawnOrder {
    parent_index: usize,
    data: AgentData,
    runtime: AgentRuntime,
}

/// Events emitted after processing a world tick.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct TickEvents {
    pub tick: Tick,
    pub charts_flushed: bool,
    pub epoch_rolled: bool,
    pub food_respawned: Option<(u32, u32)>,
}

/// Summary emitted to persistence hooks each tick.
#[derive(Debug, Clone, PartialEq)]
pub struct TickSummary {
    pub tick: Tick,
    pub agent_count: usize,
    pub births: usize,
    pub deaths: usize,
    pub total_energy: f32,
    pub average_energy: f32,
    pub average_health: f32,
}

/// Scalar metric sampled during persistence.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricSample {
    pub name: Cow<'static, str>,
    pub value: f64,
}

impl MetricSample {
    /// Creates a new metric sample.
    #[must_use]
    pub fn new(name: impl Into<Cow<'static, str>>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }

    /// Helper for `f32` values.
    #[must_use]
    pub fn from_f32(name: &'static str, value: f32) -> Self {
        Self::new(name, f64::from(value))
    }
}

/// Event type recorded for persistence.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PersistenceEventKind {
    Births,
    Deaths,
    Custom(Cow<'static, str>),
}

/// Structured persistence event entry.
#[derive(Debug, Clone, PartialEq)]
pub struct PersistenceEvent {
    pub kind: PersistenceEventKind,
    pub count: usize,
}

impl PersistenceEvent {
    /// Construct a new event entry.
    #[must_use]
    pub fn new(kind: PersistenceEventKind, count: usize) -> Self {
        Self { kind, count }
    }
}

/// Aggregate payload forwarded to persistence sinks.
#[derive(Debug, Clone)]
pub struct PersistenceBatch {
    pub summary: TickSummary,
    pub epoch: u64,
    pub closed: bool,
    pub metrics: Vec<MetricSample>,
    pub events: Vec<PersistenceEvent>,
    pub agents: Vec<AgentState>,
}

/// Persistence sink invoked after each tick.
pub trait WorldPersistence: Send {
    fn on_tick(&mut self, payload: &PersistenceBatch);
}

/// No-op persistence sink.
#[derive(Debug, Default)]
pub struct NullPersistence;

impl WorldPersistence for NullPersistence {
    fn on_tick(&mut self, _payload: &PersistenceBatch) {}
}

/// Current on-disk schema version for serialized brain genomes.
pub const GENOME_FORMAT_VERSION: u16 = 1;

/// Supported brain family discriminants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum BrainFamily {
    #[default]
    Mlp,
    Dwraon,
    Assembly,
    External(String),
}

/// Supported activation functions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum ActivationKind {
    #[default]
    Identity,
    Relu,
    Sigmoid,
    Tanh,
    Softplus,
    LeakyRelu {
        slope: f32,
    },
    Custom(String),
}

/// Layer specification used by fully-connected style brains.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerSpec {
    pub inputs: usize,
    pub outputs: usize,
    pub activation: ActivationKind,
    pub bias: bool,
    pub dropout: f32,
}

impl LayerSpec {
    /// Convenience helper to build a dense layer.
    #[must_use]
    pub fn dense(inputs: usize, outputs: usize, activation: ActivationKind) -> Self {
        Self {
            inputs,
            outputs,
            activation,
            bias: true,
            dropout: 0.0,
        }
    }
}

impl Default for LayerSpec {
    fn default() -> Self {
        Self::dense(1, 1, ActivationKind::Identity)
    }
}

/// Hyperparameter bundle stored alongside genomes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenomeHyperParams {
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub temperature: f32,
}

impl Default for GenomeHyperParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
            temperature: 1.0,
        }
    }
}

/// Provenance metadata for lineage tracking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenomeProvenance {
    pub parents: [Option<AgentId>; 2],
    pub created_at: Tick,
    pub comment: Option<String>,
}

impl Default for GenomeProvenance {
    fn default() -> Self {
        Self {
            parents: [None, None],
            created_at: Tick::zero(),
            comment: None,
        }
    }
}

/// Errors raised when validating genome structures.
#[derive(Debug, Error, PartialEq)]
pub enum GenomeError {
    #[error("layer stack must contain at least one layer")]
    EmptyLayers,
    #[error("layer {index} has zero-sized dimensions")]
    ZeroSizedLayer { index: usize },
    #[error("layer {index} dropout {dropout} must be between 0.0 and 1.0")]
    InvalidDropout { index: usize, dropout: f32 },
    #[error("layer {index} input {actual} does not match previous output {expected}")]
    MismatchedTopology {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("final layer outputs {actual} do not match genome output_size {expected}")]
    OutputMismatch { expected: usize, actual: usize },
    #[error("input_size must be non-zero")]
    ZeroInput,
    #[error("output_size must be non-zero")]
    ZeroOutput,
}

/// Versioned, serializable genome description.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BrainGenome {
    pub version: u16,
    pub family: BrainFamily,
    pub input_size: usize,
    pub output_size: usize,
    pub layers: Vec<LayerSpec>,
    pub mutation: MutationRates,
    pub hyper_params: GenomeHyperParams,
    pub provenance: GenomeProvenance,
}

impl BrainGenome {
    /// Construct and validate a new genome.
    pub fn new(
        family: BrainFamily,
        input_size: usize,
        output_size: usize,
        layers: Vec<LayerSpec>,
        mutation: MutationRates,
        hyper_params: GenomeHyperParams,
        provenance: GenomeProvenance,
    ) -> Result<Self, GenomeError> {
        let genome = Self {
            version: GENOME_FORMAT_VERSION,
            family,
            input_size,
            output_size,
            layers,
            mutation,
            hyper_params,
            provenance,
        };
        genome.validate()?;
        Ok(genome)
    }

    /// Ensure layer topology matches declared IO sizes.
    pub fn validate(&self) -> Result<(), GenomeError> {
        if self.input_size == 0 {
            return Err(GenomeError::ZeroInput);
        }
        if self.output_size == 0 {
            return Err(GenomeError::ZeroOutput);
        }
        if self.layers.is_empty() {
            return Err(GenomeError::EmptyLayers);
        }
        let mut expected_inputs = self.input_size;
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.inputs == 0 || layer.outputs == 0 {
                return Err(GenomeError::ZeroSizedLayer { index });
            }
            if layer.inputs != expected_inputs {
                return Err(GenomeError::MismatchedTopology {
                    index,
                    expected: expected_inputs,
                    actual: layer.inputs,
                });
            }
            if !(0.0..=1.0).contains(&layer.dropout) {
                return Err(GenomeError::InvalidDropout {
                    index,
                    dropout: layer.dropout,
                });
            }
            expected_inputs = layer.outputs;
        }
        if expected_inputs != self.output_size {
            return Err(GenomeError::OutputMismatch {
                expected: self.output_size,
                actual: expected_inputs,
            });
        }
        Ok(())
    }

    /// Returns true if the genome references at least one parent.
    #[must_use]
    pub fn is_descendant(&self) -> bool {
        self.provenance.parents.iter().any(Option::is_some)
    }
}

/// High level simulation clock (ticks processed since boot).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Tick(pub u64);

impl Tick {
    /// Returns the next sequential tick.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// Resets the tick counter back to zero.
    #[must_use]
    pub const fn zero() -> Self {
        Self(0)
    }
}

/// Axis-aligned 2D position (SoA column representation).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

impl Position {
    /// Construct a new position.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Velocity (wheel outputs translated to world-space delta).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Velocity {
    pub vx: f32,
    pub vy: f32,
}

impl Velocity {
    /// Construct a new velocity vector.
    #[must_use]
    pub const fn new(vx: f32, vy: f32) -> Self {
        Self { vx, vy }
    }
}

/// Lineage counter (agents produced by reproduction increment this).
#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord,
)]
pub struct Generation(pub u32);

impl Generation {
    /// Advances to the next lineage generation.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

/// Scalar fields for a single agent used when inserting or snapshotting from the SoA store.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AgentData {
    pub position: Position,
    pub velocity: Velocity,
    pub heading: f32,
    pub health: f32,
    pub color: [f32; 3],
    pub spike_length: f32,
    pub boost: bool,
    pub age: u32,
    pub generation: Generation,
}

impl AgentData {
    /// Creates a new agent payload with the provided scalar fields.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        position: Position,
        velocity: Velocity,
        heading: f32,
        health: f32,
        color: [f32; 3],
        spike_length: f32,
        boost: bool,
        age: u32,
        generation: Generation,
    ) -> Self {
        Self {
            position,
            velocity,
            heading,
            health,
            color,
            spike_length,
            boost,
            age,
            generation,
        }
    }
}

impl Default for AgentData {
    fn default() -> Self {
        Self {
            position: Position::default(),
            velocity: Velocity::default(),
            heading: 0.0,
            health: 1.0,
            color: [0.0; 3],
            spike_length: 0.0,
            boost: false,
            age: 0,
            generation: Generation::default(),
        }
    }
}

/// Collection of per-agent columns for hot-path iteration.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AgentColumns {
    positions: Vec<Position>,
    velocities: Vec<Velocity>,
    headings: Vec<f32>,
    health: Vec<f32>,
    colors: Vec<[f32; 3]>,
    spike_lengths: Vec<f32>,
    boosts: Vec<bool>,
    ages: Vec<u32>,
    generations: Vec<Generation>,
}

impl AgentColumns {
    /// Create an empty collection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a collection with reserved capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            positions: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity),
            headings: Vec::with_capacity(capacity),
            health: Vec::with_capacity(capacity),
            colors: Vec::with_capacity(capacity),
            spike_lengths: Vec::with_capacity(capacity),
            boosts: Vec::with_capacity(capacity),
            ages: Vec::with_capacity(capacity),
            generations: Vec::with_capacity(capacity),
        }
    }

    /// Number of active rows in the columns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns true if there are no active rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reserve additional capacity in each backing vector.
    pub fn reserve(&mut self, additional: usize) {
        self.positions.reserve(additional);
        self.velocities.reserve(additional);
        self.headings.reserve(additional);
        self.health.reserve(additional);
        self.colors.reserve(additional);
        self.spike_lengths.reserve(additional);
        self.boosts.reserve(additional);
        self.ages.reserve(additional);
        self.generations.reserve(additional);
    }

    /// Remove all rows while retaining capacity.
    pub fn clear(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.headings.clear();
        self.health.clear();
        self.colors.clear();
        self.spike_lengths.clear();
        self.boosts.clear();
        self.ages.clear();
        self.generations.clear();
    }

    /// Push a new row onto each column.
    pub fn push(&mut self, agent: AgentData) {
        self.positions.push(agent.position);
        self.velocities.push(agent.velocity);
        self.headings.push(agent.heading);
        self.health.push(agent.health);
        self.colors.push(agent.color);
        self.spike_lengths.push(agent.spike_length);
        self.boosts.push(agent.boost);
        self.ages.push(agent.age);
        self.generations.push(agent.generation);
        self.debug_assert_coherent();
    }

    /// Swap-remove the row at `index` and return its scalar fields.
    pub fn swap_remove(&mut self, index: usize) -> AgentData {
        let removed = AgentData {
            position: self.positions.swap_remove(index),
            velocity: self.velocities.swap_remove(index),
            heading: self.headings.swap_remove(index),
            health: self.health.swap_remove(index),
            color: self.colors.swap_remove(index),
            spike_length: self.spike_lengths.swap_remove(index),
            boost: self.boosts.swap_remove(index),
            age: self.ages.swap_remove(index),
            generation: self.generations.swap_remove(index),
        };
        self.debug_assert_coherent();
        removed
    }

    /// Copy the row at `from` into position `to` without altering length.
    pub fn move_row(&mut self, from: usize, to: usize) {
        debug_assert!(from < self.len(), "move_row from out of bounds");
        debug_assert!(to < self.len(), "move_row to out of bounds");
        if from == to {
            return;
        }
        self.positions[to] = self.positions[from];
        self.velocities[to] = self.velocities[from];
        self.headings[to] = self.headings[from];
        self.health[to] = self.health[from];
        self.colors[to] = self.colors[from];
        self.spike_lengths[to] = self.spike_lengths[from];
        self.boosts[to] = self.boosts[from];
        self.ages[to] = self.ages[from];
        self.generations[to] = self.generations[from];
    }

    /// Truncate all columns to the provided length.
    pub fn truncate(&mut self, len: usize) {
        self.positions.truncate(len);
        self.velocities.truncate(len);
        self.headings.truncate(len);
        self.health.truncate(len);
        self.colors.truncate(len);
        self.spike_lengths.truncate(len);
        self.boosts.truncate(len);
        self.ages.truncate(len);
        self.generations.truncate(len);
        self.debug_assert_coherent();
    }

    /// Return a copy of the scalar fields at `index`.
    #[must_use]
    pub fn snapshot(&self, index: usize) -> AgentData {
        AgentData {
            position: self.positions[index],
            velocity: self.velocities[index],
            heading: self.headings[index],
            health: self.health[index],
            color: self.colors[index],
            spike_length: self.spike_lengths[index],
            boost: self.boosts[index],
            age: self.ages[index],
            generation: self.generations[index],
        }
    }

    /// Immutable access to the positions slice.
    #[must_use]
    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    /// Mutable access to the positions slice.
    #[must_use]
    pub fn positions_mut(&mut self) -> &mut [Position] {
        &mut self.positions
    }

    /// Immutable access to the velocities slice.
    #[must_use]
    pub fn velocities(&self) -> &[Velocity] {
        &self.velocities
    }

    /// Mutable access to the velocities slice.
    #[must_use]
    pub fn velocities_mut(&mut self) -> &mut [Velocity] {
        &mut self.velocities
    }

    /// Immutable access to headings.
    #[must_use]
    pub fn headings(&self) -> &[f32] {
        &self.headings
    }

    /// Mutable access to headings.
    #[must_use]
    pub fn headings_mut(&mut self) -> &mut [f32] {
        &mut self.headings
    }

    /// Immutable access to health values.
    #[must_use]
    pub fn health(&self) -> &[f32] {
        &self.health
    }

    /// Mutable access to health values.
    #[must_use]
    pub fn health_mut(&mut self) -> &mut [f32] {
        &mut self.health
    }

    /// Immutable access to color triples.
    #[must_use]
    pub fn colors(&self) -> &[[f32; 3]] {
        &self.colors
    }

    /// Mutable access to color triples.
    #[must_use]
    pub fn colors_mut(&mut self) -> &mut [[f32; 3]] {
        &mut self.colors
    }

    /// Immutable access to spike lengths.
    #[must_use]
    pub fn spike_lengths(&self) -> &[f32] {
        &self.spike_lengths
    }

    /// Mutable access to spike lengths.
    #[must_use]
    pub fn spike_lengths_mut(&mut self) -> &mut [f32] {
        &mut self.spike_lengths
    }

    /// Immutable access to boost flags.
    #[must_use]
    pub fn boosts(&self) -> &[bool] {
        &self.boosts
    }

    /// Mutable access to boost flags.
    #[must_use]
    pub fn boosts_mut(&mut self) -> &mut [bool] {
        &mut self.boosts
    }

    /// Immutable access to age counters.
    #[must_use]
    pub fn ages(&self) -> &[u32] {
        &self.ages
    }

    /// Mutable access to age counters.
    #[must_use]
    pub fn ages_mut(&mut self) -> &mut [u32] {
        &mut self.ages
    }

    /// Immutable access to agent generations.
    #[must_use]
    pub fn generations(&self) -> &[Generation] {
        &self.generations
    }

    /// Mutable access to agent generations.
    #[must_use]
    pub fn generations_mut(&mut self) -> &mut [Generation] {
        &mut self.generations
    }

    #[inline]
    fn debug_assert_coherent(&self) {
        debug_assert_eq!(self.positions.len(), self.velocities.len());
        debug_assert_eq!(self.positions.len(), self.headings.len());
        debug_assert_eq!(self.positions.len(), self.health.len());
        debug_assert_eq!(self.positions.len(), self.colors.len());
        debug_assert_eq!(self.positions.len(), self.spike_lengths.len());
        debug_assert_eq!(self.positions.len(), self.boosts.len());
        debug_assert_eq!(self.positions.len(), self.ages.len());
        debug_assert_eq!(self.positions.len(), self.generations.len());
    }
}

/// Dense SoA storage with generational handles for agent access.
#[derive(Debug)]
pub struct AgentArena {
    slots: SlotMap<AgentId, usize>,
    handles: Vec<AgentId>,
    columns: AgentColumns,
}

impl Default for AgentArena {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentArena {
    /// Create an empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: SlotMap::with_key(),
            handles: Vec::new(),
            columns: AgentColumns::new(),
        }
    }

    /// Create an arena with reserved capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: SlotMap::with_capacity_and_key(capacity),
            handles: Vec::with_capacity(capacity),
            columns: AgentColumns::with_capacity(capacity),
        }
    }

    /// Number of active agents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Returns true when no agents are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Reserve space for additional agents.
    pub fn reserve(&mut self, additional: usize) {
        self.slots.reserve(additional);
        self.handles.reserve(additional);
        self.columns.reserve(additional);
    }

    /// Iterate over active agent handles in dense iteration order.
    pub fn iter_handles(&self) -> impl Iterator<Item = AgentId> + '_ {
        self.handles.iter().copied()
    }

    /// Borrow the underlying column storage.
    #[must_use]
    pub fn columns(&self) -> &AgentColumns {
        &self.columns
    }

    /// Mutably borrow the underlying column storage.
    #[must_use]
    pub fn columns_mut(&mut self) -> &mut AgentColumns {
        &mut self.columns
    }

    /// Returns the dense index for `id`, if present.
    #[must_use]
    pub fn index_of(&self, id: AgentId) -> Option<usize> {
        self.slots.get(id).copied()
    }

    /// Returns true if `id` refers to a live agent.
    #[must_use]
    pub fn contains(&self, id: AgentId) -> bool {
        self.slots.contains_key(id)
    }

    /// Insert a new agent and return its handle.
    pub fn insert(&mut self, agent: AgentData) -> AgentId {
        let index = self.columns.len();
        self.columns.push(agent);
        let id = self.slots.insert(index);
        self.handles.push(id);
        id
    }

    /// Remove `id` returning its scalar data if it was present.
    pub fn remove(&mut self, id: AgentId) -> Option<AgentData> {
        let index = self.slots.remove(id)?;
        let removed = self.columns.swap_remove(index);
        let removed_handle = self.handles.swap_remove(index);
        debug_assert_eq!(removed_handle, id);
        if index < self.handles.len() {
            let moved = self.handles[index];
            if let Some(slot) = self.slots.get_mut(moved) {
                *slot = index;
            }
        }
        Some(removed)
    }

    /// Remove all agents whose ids are contained in `dead`, preserving iteration order.
    pub fn remove_many(&mut self, dead: &HashSet<AgentId>) -> usize {
        if dead.is_empty() {
            return 0;
        }
        let mut write = 0;
        for read in 0..self.handles.len() {
            let id = self.handles[read];
            if dead.contains(&id) {
                self.slots.remove(id);
                continue;
            }
            if write != read {
                self.handles[write] = id;
                self.columns.move_row(read, write);
            }
            if let Some(slot) = self.slots.get_mut(id) {
                *slot = write;
            }
            write += 1;
        }
        let removed = self.handles.len().saturating_sub(write);
        self.handles.truncate(write);
        self.columns.truncate(write);
        removed
    }

    /// Produce a copy of the scalar data for `id`.
    #[must_use]
    pub fn snapshot(&self, id: AgentId) -> Option<AgentData> {
        let index = self.index_of(id)?;
        Some(self.columns.snapshot(index))
    }

    /// Clear all stored agents.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.handles.clear();
        self.columns.clear();
    }
}

/// Errors that can occur when constructing world state.
#[derive(Debug, Error)]
pub enum WorldStateError {
    /// Indicates an invalid configuration value.
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
}

/// Static configuration for a ScriptBots world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptBotsConfig {
    /// Width of the world in world units.
    pub world_width: u32,
    /// Height of the world in world units.
    pub world_height: u32,
    /// Size of one food cell in world units (must evenly divide width/height).
    pub food_cell_size: u32,
    /// Initial food value seeded into each grid cell.
    pub initial_food: f32,
    /// Optional RNG seed for reproducible worlds.
    pub rng_seed: Option<u64>,
    /// How frequently (in ticks) to flush chart history; 0 disables flushes.
    pub chart_flush_interval: u32,
    /// Number of ticks between food respawn events; 0 disables respawns.
    pub food_respawn_interval: u32,
    /// Amount of food to add on each respawn.
    pub food_respawn_amount: f32,
    /// Maximum food allowed per cell.
    pub food_max: f32,
    /// Logistic regrowth rate applied to each food cell every tick.
    pub food_growth_rate: f32,
    /// Proportional decay applied to each food cell every tick.
    pub food_decay_rate: f32,
    /// Diffusion factor exchanging food between neighboring cells each tick.
    pub food_diffusion_rate: f32,
    /// Radius used for neighborhood sensing.
    pub sense_radius: f32,
    /// Normalization factor for counting neighbors.
    pub sense_max_neighbors: f32,
    /// Base wheel speed produced when outputs saturate.
    pub bot_speed: f32,
    /// Half the distance between differential wheels (also used for wrapping vision bias).
    pub bot_radius: f32,
    /// Multiplier applied when boost output is triggered.
    pub boost_multiplier: f32,
    /// Increment applied to spike length toward its target each tick.
    pub spike_growth_rate: f32,
    /// Baseline metabolism drain applied each tick.
    pub metabolism_drain: f32,
    /// Fraction of velocity converted to additional energy cost.
    pub movement_drain: f32,
    /// Minimum energy level before metabolism ramping activates.
    pub metabolism_ramp_floor: f32,
    /// Additional drain applied per unit energy above the ramp floor.
    pub metabolism_ramp_rate: f32,
    /// Fixed drain added when boost output is engaged.
    pub metabolism_boost_penalty: f32,
    /// Health drain multiplier applied when agents experience temperature discomfort.
    pub temperature_discomfort_rate: f32,
    /// Difference threshold below which temperature discomfort is ignored.
    pub temperature_comfort_band: f32,
    /// Exponent shaping the environmental temperature gradient from equator to poles.
    pub temperature_gradient_exponent: f32,
    /// Exponent applied to discomfort beyond the comfort band before scaling by the drain rate.
    pub temperature_discomfort_exponent: f32,
    /// Base rate at which agents siphon food from cells.
    pub food_intake_rate: f32,
    /// Amount of food removed from a cell whenever an agent grazes.
    pub food_waste_rate: f32,
    /// Radius used for food sharing with friendly neighbors.
    pub food_sharing_radius: f32,
    /// Fraction of energy shared per neighbor when donating.
    pub food_sharing_rate: f32,
    /// Constant amount of energy transferred during altruistic sharing.
    pub food_transfer_rate: f32,
    /// Distance threshold for altruistic sharing interactions.
    pub food_sharing_distance: f32,
    /// Energy threshold required before reproduction can trigger.
    pub reproduction_energy_threshold: f32,
    /// Energy deducted from a parent upon reproduction.
    pub reproduction_energy_cost: f32,
    /// Cooldown in ticks between reproductions.
    pub reproduction_cooldown: u32,
    /// Herbivore reproduction rate multiplier applied per tick.
    pub reproduction_rate_herbivore: f32,
    /// Carnivore reproduction rate multiplier applied per tick.
    pub reproduction_rate_carnivore: f32,
    /// Bonus applied to the reproduction counter per unit ground intake.
    pub reproduction_food_bonus: f32,
    /// Starting energy assigned to a child agent.
    pub reproduction_child_energy: f32,
    /// Spatial jitter applied to child spawn positions.
    pub reproduction_spawn_jitter: f32,
    /// Color mutation range applied per channel.
    pub reproduction_color_jitter: f32,
    /// Scale factor applied to trait mutations.
    pub reproduction_mutation_scale: f32,
    /// Probability of selecting a second parent for crossover.
    pub reproduction_partner_chance: f32,
    /// Distance behind the parent where children spawn before jitter.
    pub reproduction_spawn_back_distance: f32,
    /// Maximum number of gene log entries retained per agent.
    pub reproduction_gene_log_capacity: usize,
    /// Chance to perturb mutation rates during reproduction.
    pub reproduction_meta_mutation_chance: f32,
    /// Magnitude of meta-mutation applied to mutation rates.
    pub reproduction_meta_mutation_scale: f32,
    /// Age (in ticks) after which health decay begins to scale.
    pub aging_health_decay_start: u32,
    /// Incremental health decay applied per tick beyond the start age.
    pub aging_health_decay_rate: f32,
    /// Cap applied to the age-based health decay each tick.
    pub aging_health_decay_max: f32,
    /// Multiplier converting health decay into additional energy drain.
    pub aging_energy_penalty_rate: f32,
    /// Radius within which carcass rewards are distributed.
    pub carcass_distribution_radius: f32,
    /// Base health reward shared from a carcass before scaling.
    pub carcass_health_reward: f32,
    /// Base reproduction counter reduction granted from a carcass.
    pub carcass_reproduction_reward: f32,
    /// Exponent applied to neighbor count when normalizing carcass rewards.
    pub carcass_neighbor_exponent: f32,
    /// Age at which carcass rewards reach full strength.
    pub carcass_maturity_age: u32,
    /// Fraction of health reward converted into energy.
    pub carcass_energy_share_rate: f32,
    /// Intensity scale applied to indicator pulses when feasting on carcasses.
    pub carcass_indicator_scale: f32,
    /// Whether terrain elevation influences agent locomotion and energy.
    pub topography_enabled: bool,
    /// Speed gain applied per unit downhill slope (subtracted when moving uphill).
    pub topography_speed_gain: f32,
    /// Additional metabolism drain incurred per unit uphill slope.
    pub topography_energy_penalty: f32,
    /// Minimum population size maintained via automatic seeding.
    pub population_minimum: usize,
    /// Interval (in ticks) for injecting new agents when the world is open.
    pub population_spawn_interval: u32,
    /// Number of agents added per spawn interval.
    pub population_spawn_count: u32,
    /// Probability that a spawn interval produces a crossover child instead of a random newcomer.
    pub population_crossover_chance: f32,
    /// Base radius used when checking spike impacts.
    pub spike_radius: f32,
    /// Damage applied by a spike at full power.
    pub spike_damage: f32,
    /// Energy cost of deploying a spike.
    pub spike_energy_cost: f32,
    /// Minimum spike extension required before damage can be applied.
    pub spike_min_length: f32,
    /// Cosine threshold for considering a spike aligned with its target.
    pub spike_alignment_cosine: f32,
    /// Scalar applied to velocity when scaling spike damage.
    pub spike_speed_damage_bonus: f32,
    /// Scalar applied to spike length when scaling damage.
    pub spike_length_damage_bonus: f32,
    /// Herbivore tendency threshold separating carnivores from herbivores.
    pub carnivore_threshold: f32,
    /// Maximum number of recent tick summaries retained in-memory.
    pub history_capacity: usize,
    /// Interval (ticks) between persistence flushes. 0 disables persistence.
    pub persistence_interval: u32,
    /// NeuroFlow runtime configuration.
    pub neuroflow: NeuroflowSettings,
}

impl Default for ScriptBotsConfig {
    fn default() -> Self {
        Self {
            world_width: 6_000,
            world_height: 6_000,
            food_cell_size: 60,
            initial_food: 0.0,
            rng_seed: None,
            chart_flush_interval: 1_000,
            food_respawn_interval: 15,
            food_respawn_amount: 0.5,
            food_max: 0.5,
            food_growth_rate: 0.05,
            food_decay_rate: 0.002,
            food_diffusion_rate: 0.15,
            sense_radius: 120.0,
            sense_max_neighbors: 12.0,
            bot_speed: 0.3,
            bot_radius: 10.0,
            boost_multiplier: 2.0,
            spike_growth_rate: 0.005,
            metabolism_drain: 0.002,
            movement_drain: 0.005,
            metabolism_ramp_floor: 1.0,
            metabolism_ramp_rate: 0.0,
            metabolism_boost_penalty: 0.0,
            temperature_discomfort_rate: 0.0,
            temperature_comfort_band: 0.08,
            temperature_gradient_exponent: 1.0,
            temperature_discomfort_exponent: 2.0,
            food_intake_rate: 0.002,
            food_waste_rate: 0.001,
            food_sharing_radius: 50.0,
            food_sharing_rate: 0.1,
            food_transfer_rate: 0.001,
            food_sharing_distance: 50.0,
            reproduction_energy_threshold: 0.65,
            reproduction_energy_cost: 0.0,
            reproduction_cooldown: 300,
            reproduction_rate_herbivore: 1.0,
            reproduction_rate_carnivore: 1.0,
            reproduction_food_bonus: 3.0,
            reproduction_child_energy: 1.0,
            reproduction_spawn_jitter: 20.0,
            reproduction_color_jitter: 0.05,
            reproduction_mutation_scale: 0.02,
            reproduction_partner_chance: 0.15,
            reproduction_spawn_back_distance: 12.0,
            reproduction_gene_log_capacity: 12,
            reproduction_meta_mutation_chance: 0.2,
            reproduction_meta_mutation_scale: 0.5,
            aging_health_decay_start: 12_000,
            aging_health_decay_rate: 0.0,
            aging_health_decay_max: 0.0,
            aging_energy_penalty_rate: 0.0,
            carcass_distribution_radius: 100.0,
            carcass_health_reward: 5.0,
            carcass_reproduction_reward: 5.0,
            carcass_neighbor_exponent: 1.25,
            carcass_maturity_age: 5,
            carcass_energy_share_rate: 0.5,
            carcass_indicator_scale: 20.0,
            topography_enabled: false,
            topography_speed_gain: 0.35,
            topography_energy_penalty: 0.002,
            population_minimum: 0,
            population_spawn_interval: 100,
            population_spawn_count: 1,
            population_crossover_chance: 0.5,
            spike_radius: 40.0,
            spike_damage: 0.25,
            spike_energy_cost: 0.02,
            spike_min_length: 0.2,
            spike_alignment_cosine: (std::f32::consts::FRAC_PI_8).cos(),
            spike_speed_damage_bonus: 0.6,
            spike_length_damage_bonus: 0.75,
            carnivore_threshold: 0.5,
            history_capacity: 256,
            persistence_interval: 0,
            neuroflow: NeuroflowSettings::default(),
        }
    }
}

/// Runtime configuration options for NeuroFlow-backed brains.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeuroflowSettings {
    /// Whether NeuroFlow brains are registered at runtime.
    pub enabled: bool,
    /// Hidden layer sizes supplied to the NeuroFlow network.
    pub hidden_layers: Vec<usize>,
    /// Activation function applied to the hidden/output layers.
    pub activation: NeuroflowActivationKind,
}

impl Default for NeuroflowSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            hidden_layers: vec![48, 32, 24],
            activation: NeuroflowActivationKind::Tanh,
        }
    }
}

/// Supported activation functions for NeuroFlow networks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum NeuroflowActivationKind {
    #[default]
    Tanh,
    Sigmoid,
    Relu,
}

impl ScriptBotsConfig {
    /// Validates the configuration, returning derived grid dimensions.
    fn food_dimensions(&self) -> Result<(u32, u32), WorldStateError> {
        if self.world_width == 0 || self.world_height == 0 {
            return Err(WorldStateError::InvalidConfig(
                "world dimensions must be non-zero",
            ));
        }
        if self.food_cell_size == 0 {
            return Err(WorldStateError::InvalidConfig(
                "food_cell_size must be non-zero",
            ));
        }
        if !self.world_width.is_multiple_of(self.food_cell_size)
            || !self.world_height.is_multiple_of(self.food_cell_size)
        {
            return Err(WorldStateError::InvalidConfig(
                "world dimensions must be divisible by food_cell_size",
            ));
        }
        let dims = (
            self.world_width / self.food_cell_size,
            self.world_height / self.food_cell_size,
        );
        if self.initial_food < 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "initial_food must be non-negative",
            ));
        }
        if self.food_max <= 0.0 {
            return Err(WorldStateError::InvalidConfig("food_max must be positive"));
        }
        if self.food_respawn_amount < 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "food_respawn_amount must be non-negative",
            ));
        }
        if self.initial_food > self.food_max {
            return Err(WorldStateError::InvalidConfig(
                "initial_food cannot exceed food_max",
            ));
        }
        if self.food_respawn_amount > self.food_max {
            return Err(WorldStateError::InvalidConfig(
                "food_respawn_amount cannot exceed food_max",
            ));
        }
        if self.food_waste_rate < 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "food_waste_rate must be non-negative",
            ));
        }
        if self.food_waste_rate > self.food_max {
            return Err(WorldStateError::InvalidConfig(
                "food_waste_rate cannot exceed food_max",
            ));
        }
        if self.food_growth_rate < 0.0
            || self.food_decay_rate < 0.0
            || self.food_diffusion_rate < 0.0
            || self.food_diffusion_rate > 0.25
        {
            return Err(WorldStateError::InvalidConfig(
                "food growth/decay must be non-negative and diffusion in [0, 0.25]",
            ));
        }
        if !(0.0..=1.0).contains(&self.reproduction_partner_chance) {
            return Err(WorldStateError::InvalidConfig(
                "reproduction_partner_chance must be within [0, 1]",
            ));
        }
        if self.reproduction_spawn_back_distance < 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "reproduction_spawn_back_distance must be non-negative",
            ));
        }
        if !(0.0..=1.0).contains(&self.reproduction_meta_mutation_chance) {
            return Err(WorldStateError::InvalidConfig(
                "reproduction_meta_mutation_chance must be within [0, 1]",
            ));
        }
        if self.reproduction_meta_mutation_scale < 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "reproduction_meta_mutation_scale must be non-negative",
            ));
        }
        if self.metabolism_drain < 0.0
            || self.movement_drain < 0.0
            || self.metabolism_ramp_floor < 0.0
            || self.metabolism_ramp_rate < 0.0
            || self.metabolism_boost_penalty < 0.0
            || self.food_intake_rate < 0.0
            || self.food_waste_rate < 0.0
            || self.reproduction_food_bonus < 0.0
            || self.food_sharing_radius <= 0.0
            || self.food_sharing_rate < 0.0
            || self.food_transfer_rate < 0.0
            || self.food_sharing_distance <= 0.0
            || self.reproduction_energy_threshold < 0.0
            || self.reproduction_energy_cost < 0.0
            || self.reproduction_child_energy < 0.0
            || self.reproduction_spawn_jitter < 0.0
            || self.reproduction_color_jitter < 0.0
            || self.reproduction_mutation_scale < 0.0
            || self.reproduction_rate_herbivore <= 0.0
            || self.reproduction_rate_carnivore <= 0.0
            || self.spike_radius <= 0.0
            || self.spike_damage < 0.0
            || self.spike_energy_cost < 0.0
            || self.spike_min_length < 0.0
            || self.spike_alignment_cosine <= 0.0
            || self.spike_alignment_cosine > 1.0
            || self.spike_speed_damage_bonus < 0.0
            || self.spike_length_damage_bonus < 0.0
            || self.carnivore_threshold <= 0.0
            || self.carnivore_threshold >= 1.0
            || self.history_capacity == 0
            || self.temperature_discomfort_rate < 0.0
            || self.aging_health_decay_rate < 0.0
            || self.aging_health_decay_max < 0.0
            || self.aging_energy_penalty_rate < 0.0
            || self.carcass_distribution_radius < 0.0
            || self.carcass_health_reward < 0.0
            || self.carcass_reproduction_reward < 0.0
            || self.carcass_energy_share_rate < 0.0
            || self.carcass_indicator_scale < 0.0
            || self.topography_speed_gain < 0.0
            || self.topography_energy_penalty < 0.0
        {
            return Err(WorldStateError::InvalidConfig(
                "metabolism, reproduction, sharing, and history parameters must be non-negative; spike and diet thresholds must be within valid ranges",
            ));
        }
        if !(0.0..=1.0).contains(&self.temperature_comfort_band) {
            return Err(WorldStateError::InvalidConfig(
                "temperature_comfort_band must be within [0, 1]",
            ));
        }
        if self.temperature_gradient_exponent <= 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "temperature_gradient_exponent must be positive",
            ));
        }
        if self.temperature_discomfort_exponent <= 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "temperature_discomfort_exponent must be positive",
            ));
        }
        if self.aging_health_decay_rate > 0.0
            && self.aging_health_decay_max < self.aging_health_decay_rate
        {
            return Err(WorldStateError::InvalidConfig(
                "aging_health_decay_max must be >= aging_health_decay_rate when decay is enabled",
            ));
        }
        if self.carcass_neighbor_exponent <= 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "carcass_neighbor_exponent must be positive",
            ));
        }
        if self.carcass_maturity_age == 0 {
            return Err(WorldStateError::InvalidConfig(
                "carcass_maturity_age must be at least 1",
            ));
        }
        if !(0.0..=1.0).contains(&self.population_crossover_chance) {
            return Err(WorldStateError::InvalidConfig(
                "population_crossover_chance must be within [0, 1]",
            ));
        }
        if self.population_spawn_count == 0 {
            return Err(WorldStateError::InvalidConfig(
                "population_spawn_count must be at least 1",
            ));
        }
        if self.reproduction_energy_cost > self.reproduction_energy_threshold {
            return Err(WorldStateError::InvalidConfig(
                "reproduction_energy_cost cannot exceed reproduction_energy_threshold",
            ));
        }
        if self.sense_radius <= 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "sense_radius must be positive",
            ));
        }
        if self.sense_max_neighbors <= 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "sense_max_neighbors must be positive",
            ));
        }
        if self.bot_radius <= 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "bot_radius must be positive",
            ));
        }
        if self.bot_speed < 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "bot_speed must be non-negative",
            ));
        }
        if self.boost_multiplier < 1.0 {
            return Err(WorldStateError::InvalidConfig(
                "boost_multiplier must be at least 1.0",
            ));
        }
        if self.spike_growth_rate < 0.0 {
            return Err(WorldStateError::InvalidConfig(
                "spike_growth_rate must be non-negative",
            ));
        }
        Ok(dims)
    }

    /// Returns the configured RNG seed, generating one from entropy if absent.
    fn seeded_rng(&self) -> SmallRng {
        match self.rng_seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => {
                let seed: u64 = rand::random();
                SmallRng::seed_from_u64(seed)
            }
        }
    }
}

/// 2D food grid storing scalar energy values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoodGrid {
    width: u32,
    height: u32,
    cells: Vec<f32>,
}

impl FoodGrid {
    /// Construct a grid with `width * height` cells initialised to `initial`.
    pub fn new(width: u32, height: u32, initial: f32) -> Result<Self, WorldStateError> {
        if width == 0 || height == 0 {
            return Err(WorldStateError::InvalidConfig(
                "food grid dimensions must be non-zero",
            ));
        }
        Ok(Self {
            width,
            height,
            cells: vec![initial; (width as usize) * (height as usize)],
        })
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub fn cells(&self) -> &[f32] {
        &self.cells
    }

    #[must_use]
    pub fn cells_mut(&mut self) -> &mut [f32] {
        &mut self.cells
    }

    /// Returns the flat index for `(x, y)` without bounds checks.
    #[inline]
    fn offset(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.width as usize) + (x as usize)
    }

    /// Immutable access to a specific cell.
    pub fn get(&self, x: u32, y: u32) -> Option<f32> {
        if x < self.width && y < self.height {
            Some(self.cells[self.offset(x, y)])
        } else {
            None
        }
    }

    /// Mutable access to a specific cell.
    pub fn get_mut(&mut self, x: u32, y: u32) -> Option<&mut f32> {
        if x < self.width && y < self.height {
            let idx = self.offset(x, y);
            Some(&mut self.cells[idx])
        } else {
            None
        }
    }

    /// Fills the grid with the provided scalar value.
    pub fn fill(&mut self, value: f32) {
        self.cells.fill(value);
    }
}

/// Tile-based terrain layer used for rendering biomes and overlays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainLayer {
    width: u32,
    height: u32,
    cell_size: u32,
    tiles: Vec<TerrainTile>,
}

impl TerrainLayer {
    /// Generate a deterministic terrain layer using the supplied RNG.
    pub fn generate(
        width: u32,
        height: u32,
        cell_size: u32,
        rng: &mut SmallRng,
    ) -> Result<Self, WorldStateError> {
        if width == 0 || height == 0 {
            return Err(WorldStateError::InvalidConfig(
                "terrain dimensions must be non-zero",
            ));
        }

        let mut tiles = Vec::with_capacity((width as usize) * (height as usize));
        let width_f = width as f32;
        let height_f = height as f32;

        for y in 0..height {
            for x in 0..width {
                let fx = x as f32 / width_f;
                let fy = y as f32 / height_f;
                let distance = ((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt();
                let ridge = ((fx - fy).abs() * 0.75).clamp(0.0, 1.0);
                let base_noise = rng.random_range(0.0..1.0);
                let accent_noise = rng.random_range(0.0..1.0);
                let elevation =
                    (1.0 - distance * 1.5 + base_noise * 0.35 - ridge * 0.2).clamp(0.0, 1.0);
                let moisture = ((0.5 - (fy - 0.5).abs()) * 1.4
                    + rng.random_range(0.0..1.0) * 0.4
                    + ridge * 0.15)
                    .clamp(0.0, 1.0);

                let kind = if elevation < 0.22 {
                    TerrainKind::DeepWater
                } else if elevation < 0.32 {
                    TerrainKind::ShallowWater
                } else if elevation < 0.36 {
                    TerrainKind::Sand
                } else if elevation > 0.78 {
                    TerrainKind::Rock
                } else if moisture > 0.68 {
                    TerrainKind::Bloom
                } else {
                    TerrainKind::Grass
                };

                tiles.push(TerrainTile {
                    kind,
                    elevation,
                    moisture,
                    accent: accent_noise,
                });
            }
        }

        Ok(Self {
            width,
            height,
            cell_size,
            tiles,
        })
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub const fn cell_size(&self) -> u32 {
        self.cell_size
    }

    #[must_use]
    pub fn tiles(&self) -> &[TerrainTile] {
        &self.tiles
    }

    #[must_use]
    pub fn tile(&self, x: u32, y: u32) -> Option<&TerrainTile> {
        if x < self.width && y < self.height {
            let idx = (y as usize) * (self.width as usize) + (x as usize);
            Some(&self.tiles[idx])
        } else {
            None
        }
    }

    fn tile_wrapped(&self, x: i32, y: i32) -> &TerrainTile {
        let w = self.width as i32;
        let h = self.height as i32;
        let ix = ((x % w) + w) % w;
        let iy = ((y % h) + h) % h;
        let idx = (iy as usize) * (self.width as usize) + ix as usize;
        &self.tiles[idx]
    }

    fn sample_elevation(&self, fx: f32, fy: f32) -> f32 {
        let width = self.width as f32;
        let height = self.height as f32;
        let mut x = fx;
        let mut y = fy;
        if width > 0.0 {
            x = x.rem_euclid(width);
        }
        if height > 0.0 {
            y = y.rem_euclid(height);
        }
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let e00 = self.tile_wrapped(x0, y0).elevation;
        let e10 = self.tile_wrapped(x1, y0).elevation;
        let e01 = self.tile_wrapped(x0, y1).elevation;
        let e11 = self.tile_wrapped(x1, y1).elevation;

        let ex0 = e00 + (e10 - e00) * tx;
        let ex1 = e01 + (e11 - e01) * tx;
        ex0 + (ex1 - ex0) * ty
    }

    /// Returns the elevation gradient (∂e/∂x, ∂e/∂y) in world units.
    pub fn gradient_world(&self, x: f32, y: f32, cell_size: f32) -> (f32, f32) {
        if cell_size <= 0.0 {
            return (0.0, 0.0);
        }
        let fx = x / cell_size;
        let fy = y / cell_size;
        let e_px = self.sample_elevation(fx + 1.0, fy);
        let e_mx = self.sample_elevation(fx - 1.0, fy);
        let e_py = self.sample_elevation(fx, fy + 1.0);
        let e_my = self.sample_elevation(fx, fy - 1.0);

        let grad_x = (e_px - e_mx) * 0.5 / cell_size;
        let grad_y = (e_py - e_my) * 0.5 / cell_size;
        (grad_x, grad_y)
    }
}

/// Terrain classification for each tile.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TerrainKind {
    DeepWater,
    ShallowWater,
    Sand,
    Grass,
    Bloom,
    Rock,
}

/// Metadata captured for every terrain tile.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TerrainTile {
    pub kind: TerrainKind,
    pub elevation: f32,
    pub moisture: f32,
    pub accent: f32,
}

/// Aggregate world state shared by the simulation and rendering layers.
pub struct WorldState {
    config: ScriptBotsConfig,
    tick: Tick,
    epoch: u64,
    closed: bool,
    rng: SmallRng,
    agents: AgentArena,
    food: FoodGrid,
    terrain: TerrainLayer,
    runtime: AgentMap<AgentRuntime>,
    index: UniformGridIndex,
    brain_registry: BrainRegistry,
    food_scratch: Vec<f32>,
    pending_deaths: Vec<AgentId>,
    #[allow(dead_code)]
    pending_spawns: Vec<SpawnOrder>,
    persistence: Box<dyn WorldPersistence>,
    last_births: usize,
    last_deaths: usize,
    history: VecDeque<TickSummary>,
    #[allow(dead_code)]
    carcass_health_distributed: f32,
    #[allow(dead_code)]
    carcass_reproduction_bonus: f32,
}

impl fmt::Debug for WorldState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorldState")
            .field("config", &self.config)
            .field("tick", &self.tick)
            .field("epoch", &self.epoch)
            .field("closed", &self.closed)
            .field("agent_count", &self.agents.len())
            .finish()
    }
}

impl WorldState {
    /// Instantiate a new world using the supplied configuration.
    pub fn new(config: ScriptBotsConfig) -> Result<Self, WorldStateError> {
        Self::with_persistence(config, Box::new(NullPersistence))
    }

    /// Instantiate a new world using the supplied configuration and persistence sink.
    pub fn with_persistence(
        config: ScriptBotsConfig,
        persistence: Box<dyn WorldPersistence>,
    ) -> Result<Self, WorldStateError> {
        let (food_w, food_h) = config.food_dimensions()?;
        let rng = config.seeded_rng();
        let mut terrain_rng = rng.clone();
        let food = FoodGrid::new(food_w, food_h, config.initial_food)?;
        let terrain =
            TerrainLayer::generate(food_w, food_h, config.food_cell_size, &mut terrain_rng)?;
        let index = UniformGridIndex::new(
            config.food_cell_size as f32,
            config.world_width as f32,
            config.world_height as f32,
        );
        let history_capacity = config.history_capacity;
        Ok(Self {
            food,
            terrain,
            config,
            tick: Tick::zero(),
            epoch: 0,
            closed: false,
            rng,
            agents: AgentArena::new(),
            runtime: AgentMap::new(),
            index,
            brain_registry: BrainRegistry::new(),
            food_scratch: vec![0.0; (food_w as usize) * (food_h as usize)],
            pending_deaths: Vec::new(),
            pending_spawns: Vec::new(),
            persistence,
            last_births: 0,
            last_deaths: 0,
            history: VecDeque::with_capacity(history_capacity),
            carcass_health_distributed: 0.0,
            carcass_reproduction_bonus: 0.0,
        })
    }

    fn stage_aging(&mut self) {
        {
            let columns = self.agents.columns_mut();
            for age in columns.ages_mut() {
                *age = age.saturating_add(1);
            }
        }

        let rate = self.config.aging_health_decay_rate;
        if rate <= 0.0 {
            return;
        }

        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        if handles.is_empty() {
            return;
        }

        let ages_snapshot = self.agents.columns().ages().to_vec();
        let mut penalties = vec![0.0f32; handles.len()];
        let start = self.config.aging_health_decay_start;
        let max_penalty = self.config.aging_health_decay_max;

        for (idx, age) in ages_snapshot.iter().enumerate() {
            if *age > start {
                let over = (*age - start) as f32;
                let penalty = (over * rate).min(max_penalty);
                if penalty > 0.0 {
                    penalties[idx] = penalty;
                }
            }
        }

        if penalties.iter().all(|penalty| *penalty <= 0.0) {
            return;
        }

        {
            let columns = self.agents.columns_mut();
            let healths = columns.health_mut();
            for (idx, penalty) in penalties.iter().enumerate() {
                if *penalty > 0.0 {
                    healths[idx] = (healths[idx] - *penalty).max(0.0);
                }
            }
        }

        let energy_scale = self.config.aging_energy_penalty_rate.max(0.0);
        let health_snapshot = self.agents.columns().health().to_vec();

        for (idx, agent_id) in handles.iter().enumerate() {
            let penalty = penalties[idx];
            if penalty <= 0.0 {
                continue;
            }
            if energy_scale > 0.0 {
                if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                    let energy_penalty = penalty * energy_scale;
                    runtime.energy = (runtime.energy - energy_penalty).max(0.0);
                    runtime.food_delta -= energy_penalty;
                }
            }
            if health_snapshot.get(idx).copied().unwrap_or(0.0) <= 0.0 {
                self.pending_deaths.push(*agent_id);
            }
        }
    }

    fn stage_food_respawn(&mut self, next_tick: Tick) -> Option<(u32, u32)> {
        let interval = self.config.food_respawn_interval;
        if interval == 0 {
            return None;
        }
        if !next_tick.0.is_multiple_of(interval as u64) {
            return None;
        }
        let width = self.food.width();
        let height = self.food.height();
        if width == 0 || height == 0 {
            return None;
        }
        let x = self.rng.random_range(0..width);
        let y = self.rng.random_range(0..height);
        if let Some(cell) = self.food.get_mut(x, y) {
            *cell = (*cell + self.config.food_respawn_amount).min(self.config.food_max);
            Some((x, y))
        } else {
            None
        }
    }

    fn stage_food_dynamics(&mut self, next_tick: Tick) -> Option<(u32, u32)> {
        let respawned = self.stage_food_respawn(next_tick);
        self.apply_food_regrowth();
        respawned
    }

    fn apply_food_regrowth(&mut self) {
        let growth = self.config.food_growth_rate;
        let decay = self.config.food_decay_rate;
        let diffusion = self.config.food_diffusion_rate;
        if growth <= 0.0 && decay <= 0.0 && diffusion <= 0.0 {
            return;
        }

        let width = self.food.width() as usize;
        let height = self.food.height() as usize;
        let len = width * height;
        if self.food_scratch.len() != len {
            self.food_scratch.resize(len, 0.0);
        }

        {
            let cells = self.food.cells();
            self.food_scratch[..len].copy_from_slice(cells);
        }

        let max_value = self.config.food_max;
        let previous = &self.food_scratch;
        let cells_mut = self.food.cells_mut();

        for y in 0..height {
            let up_row = if y == 0 { height - 1 } else { y - 1 };
            let down_row = if y + 1 == height { 0 } else { y + 1 };
            for x in 0..width {
                let left_col = if x == 0 { width - 1 } else { x - 1 };
                let right_col = if x + 1 == width { 0 } else { x + 1 };
                let idx = y * width + x;
                let mut value = previous[idx];

                if diffusion > 0.0 {
                    let left = previous[y * width + left_col];
                    let right = previous[y * width + right_col];
                    let up = previous[up_row * width + x];
                    let down = previous[down_row * width + x];
                    let neighbor_avg = (left + right + up + down) * 0.25;
                    value += diffusion * (neighbor_avg - previous[idx]);
                }

                if decay > 0.0 {
                    value -= decay * value;
                }

                if growth > 0.0 {
                    value += growth * (max_value - value);
                }

                cells_mut[idx] = value.clamp(0.0, max_value);
            }
        }
    }

    fn stage_sense(&mut self) {
        let agent_count = self.agents.len();
        if agent_count == 0 {
            return;
        }

        let columns = self.agents.columns();
        let positions = columns.positions();
        let headings = columns.headings();
        let velocities = columns.velocities();
        let colors = columns.colors();
        let healths = columns.health();

        let position_pairs: Vec<(f32, f32)> = positions.iter().map(|p| (p.x, p.y)).collect();
        if self.index.rebuild(&position_pairs).is_err() {
            return;
        }

        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        let runtime = &self.runtime;

        let trait_modifiers: Vec<TraitModifiers> = handles
            .iter()
            .map(|id| {
                runtime
                    .get(*id)
                    .map_or(TraitModifiers::default(), |rt| rt.trait_modifiers)
            })
            .collect();
        let eye_directions: Vec<[f32; NUM_EYES]> = handles
            .iter()
            .map(|id| {
                runtime
                    .get(*id)
                    .map_or([0.0; NUM_EYES], |rt| rt.eye_direction)
            })
            .collect();
        let eye_fov: Vec<[f32; NUM_EYES]> = handles
            .iter()
            .map(|id| runtime.get(*id).map_or([1.0; NUM_EYES], |rt| rt.eye_fov))
            .collect();
        let clocks: Vec<[f32; 2]> = handles
            .iter()
            .map(|id| runtime.get(*id).map_or([50.0, 50.0], |rt| rt.clocks))
            .collect();
        let temperature_preferences: Vec<f32> = handles
            .iter()
            .map(|id| runtime.get(*id).map_or(0.5, |rt| rt.temperature_preference))
            .collect();
        let sound_emitters: Vec<f32> = handles
            .iter()
            .map(|id| runtime.get(*id).map_or(0.0, |rt| rt.sound_multiplier))
            .collect();

        let world_width = self.config.world_width as f32;
        let world_height = self.config.world_height as f32;
        let radius = self.config.sense_radius;
        let radius_sq = radius * radius;
        let cell_size = self.config.food_cell_size as f32;
        let food_width = self.food.width();
        let food_height = self.food.height();
        let food_cells = self.food.cells();
        let food_max = self.config.food_max;
        let max_speed = (self.config.bot_speed * self.config.boost_multiplier).max(1e-3);
        let tick_value = self.tick.0 as f32;
        let index = &self.index;

        let sensor_results: Vec<[f32; INPUT_SIZE]> = collect_handles!(handles, |idx, _handle| {
            let mut sensors = [0.0f32; INPUT_SIZE];
            let mut density = [0.0f32; NUM_EYES];
            let mut eye_r = [0.0f32; NUM_EYES];
            let mut eye_g = [0.0f32; NUM_EYES];
            let mut eye_b = [0.0f32; NUM_EYES];
            let mut smell = 0.0f32;
            let mut sound = 0.0f32;
            let mut hearing = 0.0f32;
            let mut blood = 0.0f32;

            let position = positions[idx];
            let heading = headings[idx];
            let traits = trait_modifiers[idx];
            let eyes_dir = eye_directions[idx];
            let eyes_fov = eye_fov[idx];

            index.neighbors_within(
                idx,
                radius_sq,
                &mut |other_idx, dist_sq: OrderedFloat<f32>| {
                    if other_idx == idx {
                        return;
                    }
                    let dist_sq_val = dist_sq.into_inner();
                    if dist_sq_val <= f32::EPSILON {
                        return;
                    }
                    let dist = dist_sq_val.sqrt();
                    if dist > radius {
                        return;
                    }

                    let dx = toroidal_delta(positions[other_idx].x, position.x, world_width);
                    let dy = toroidal_delta(positions[other_idx].y, position.y, world_height);
                    let ang = angle_to(dx, dy);
                    let dist_factor = (radius - dist) / radius;
                    if dist_factor <= 0.0 {
                        return;
                    }

                    for eye in 0..NUM_EYES {
                        let view_dir = wrap_signed_angle(heading + eyes_dir[eye]);
                        let diff = angle_difference(view_dir, ang);
                        let fov = eyes_fov[eye].max(0.01);
                        if diff < fov {
                            let fov_factor = ((fov - diff) / fov).max(0.0);
                            let intensity = traits.eye * fov_factor * dist_factor * (dist / radius);
                            density[eye] += intensity;
                            let color = colors[other_idx];
                            eye_r[eye] += intensity * color[0];
                            eye_g[eye] += intensity * color[1];
                            eye_b[eye] += intensity * color[2];
                        }
                    }

                    smell += dist_factor;

                    let velocity = velocities[other_idx];
                    let speed = (velocity.vx * velocity.vx + velocity.vy * velocity.vy).sqrt();
                    sound += dist_factor * (speed / max_speed).clamp(0.0, 1.0);
                    hearing += dist_factor * sound_emitters[other_idx];

                    let forward_diff = angle_difference(heading, ang);
                    if forward_diff < BLOOD_HALF_FOV {
                        let bleed = (BLOOD_HALF_FOV - forward_diff) / BLOOD_HALF_FOV;
                        let health = healths[other_idx];
                        let wound = (1.0 - (health * 0.5).clamp(0.0, 1.0)).max(0.0);
                        blood += bleed * dist_factor * wound;
                    }
                },
            );

            smell *= traits.smell;
            sound *= traits.sound;
            hearing *= traits.hearing;
            blood *= traits.blood;

            let cell_x =
                ((position.x / cell_size).floor() as i32).rem_euclid(food_width as i32) as u32;
            let cell_y =
                ((position.y / cell_size).floor() as i32).rem_euclid(food_height as i32) as u32;
            let food_idx = (cell_y as usize) * (food_width as usize) + cell_x as usize;
            let food_value = food_cells.get(food_idx).copied().unwrap_or(0.0) / food_max;

            sensors[0] = clamp01(density[0]);
            sensors[1] = clamp01(eye_r[0]);
            sensors[2] = clamp01(eye_g[0]);
            sensors[3] = clamp01(eye_b[0]);
            sensors[4] = clamp01(food_value);
            sensors[5] = clamp01(density[1]);
            sensors[6] = clamp01(eye_r[1]);
            sensors[7] = clamp01(eye_g[1]);
            sensors[8] = clamp01(eye_b[1]);
            sensors[9] = clamp01(sound);
            sensors[10] = clamp01(smell);
            sensors[11] = clamp01(healths[idx] * 0.5);
            sensors[12] = clamp01(density[2]);
            sensors[13] = clamp01(eye_r[2]);
            sensors[14] = clamp01(eye_g[2]);
            sensors[15] = clamp01(eye_b[2]);
            sensors[16] = (tick_value / clocks[idx][0].max(1.0)).sin().abs();
            sensors[17] = (tick_value / clocks[idx][1].max(1.0)).sin().abs();
            sensors[18] = clamp01(hearing);
            sensors[19] = clamp01(blood);
            let env_temperature = sample_temperature(&self.config, position.x);
            let discomfort = temperature_discomfort(env_temperature, temperature_preferences[idx]);
            sensors[20] = clamp01(discomfort);
            sensors[21] = clamp01(density[3]);
            sensors[22] = clamp01(eye_r[3]);
            sensors[23] = clamp01(eye_g[3]);
            sensors[24] = clamp01(eye_b[3]);
            sensors
        });

        for (idx, agent_id) in handles.iter().enumerate() {
            if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                runtime.sensors.copy_from_slice(&sensor_results[idx]);
            }
        }
    }

    fn default_outputs(inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        let mut outputs = [0.0; OUTPUT_SIZE];
        let limit = OUTPUT_SIZE.min(INPUT_SIZE);
        outputs[..limit].copy_from_slice(&inputs[..limit]);
        outputs
    }

    fn stage_brains(&mut self) {
        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        for agent_id in handles {
            if let Some(runtime) = self.runtime.get_mut(agent_id) {
                let outputs = runtime
                    .brain
                    .tick(&runtime.sensors)
                    .unwrap_or_else(|| Self::default_outputs(&runtime.sensors));
                runtime.outputs = outputs;
            }
        }
    }

    fn wrap_position(value: f32, extent: f32) -> f32 {
        if extent <= 0.0 {
            return 0.0;
        }
        let mut v = value % extent;
        if v < 0.0 {
            v += extent;
        }
        v
    }

    fn wrap_delta(origin: f32, target: f32, extent: f32) -> f32 {
        if extent <= 0.0 {
            return target - origin;
        }
        let mut delta = target - origin;
        let half = extent * 0.5;
        if delta > half {
            delta -= extent;
        } else if delta < -half {
            delta += extent;
        }
        delta
    }

    fn stage_actuation(&mut self) {
        let width = self.config.world_width as f32;
        let height = self.config.world_height as f32;
        let bot_speed = self.config.bot_speed.max(0.0);
        let bot_radius = self.config.bot_radius.max(1.0);
        let wheel_base = (bot_radius * 2.0).max(1.0);
        let boost_multiplier = self.config.boost_multiplier.max(1.0);
        let spike_growth = self.config.spike_growth_rate.max(0.0);
        let movement_drain = self.config.movement_drain;
        let metabolism_drain = self.config.metabolism_drain;
        let ramp_floor = self.config.metabolism_ramp_floor;
        let ramp_rate = self.config.metabolism_ramp_rate;
        let boost_penalty = self.config.metabolism_boost_penalty.max(0.0);

        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        if handles.is_empty() {
            return;
        }

        let columns = self.agents.columns();
        let positions_snapshot: Vec<Position> = columns.positions().to_vec();
        let headings_snapshot: Vec<f32> = columns.headings().to_vec();
        let spike_lengths_snapshot: Vec<f32> = columns.spike_lengths().to_vec();

        let runtime = &self.runtime;
        let terrain = &self.terrain;
        let cell_size = self.config.food_cell_size as f32;
        let topo_enabled = self.config.topography_enabled;
        let topo_gain = self.config.topography_speed_gain.max(0.0);
        let topo_penalty = self.config.topography_energy_penalty.max(0.0);
        let results: Vec<ActuationResult> = collect_handles!(handles, |idx, agent_id| {
            if let Some(runtime) = runtime.get(*agent_id) {
                let outputs = runtime.outputs;
                let left = outputs.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let right = outputs.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let color = [
                    clamp01(outputs.get(2).copied().unwrap_or(0.0)),
                    clamp01(outputs.get(3).copied().unwrap_or(0.0)),
                    clamp01(outputs.get(4).copied().unwrap_or(0.0)),
                ];
                let spike_target = outputs.get(5).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let boost = outputs.get(6).copied().unwrap_or(0.0) > 0.5;
                let sound_level = outputs.get(7).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let give_intent = outputs.get(8).copied().unwrap_or(0.0).clamp(0.0, 1.0);

                let mut left_speed = left * bot_speed;
                let mut right_speed = right * bot_speed;
                if boost {
                    left_speed *= boost_multiplier;
                    right_speed *= boost_multiplier;
                }

                let mut heading = headings_snapshot[idx];
                let angular = (right_speed - left_speed) / wheel_base;
                heading = wrap_signed_angle(heading + angular);
                let mut slope_along: f32 = 0.0;
                if topo_enabled && cell_size > 0.0 {
                    let (grad_x, grad_y) = terrain.gradient_world(
                        positions_snapshot[idx].x,
                        positions_snapshot[idx].y,
                        cell_size,
                    );
                    let dir_x = heading.cos();
                    let dir_y = heading.sin();
                    slope_along = grad_x * dir_x + grad_y * dir_y;
                    if topo_gain > 0.0 {
                        let downhill = (-slope_along).max(0.0);
                        let uphill = slope_along.max(0.0);
                        let mut speed_factor: f32 = 1.0;
                        if downhill > 0.0 {
                            speed_factor *= 1.0 + downhill * topo_gain;
                        }
                        if uphill > 0.0 {
                            speed_factor /= 1.0 + uphill * topo_gain;
                        }
                        speed_factor = speed_factor.clamp(0.4, 1.8);
                        left_speed *= speed_factor;
                        right_speed *= speed_factor;
                    }
                }

                let linear = (left_speed + right_speed) * 0.5;
                let vx = heading.cos() * linear;
                let vy = heading.sin() * linear;

                let mut next_pos = positions_snapshot[idx];
                next_pos.x = Self::wrap_position(next_pos.x + vx, width);
                next_pos.y = Self::wrap_position(next_pos.y + vy, height);

                let movement_penalty =
                    movement_drain * (left_speed.abs() + right_speed.abs()) * 0.5;
                let mut drain = metabolism_drain + movement_penalty;
                if ramp_rate > 0.0 {
                    let active_energy = (runtime.energy - ramp_floor).max(0.0);
                    drain += active_energy * ramp_rate;
                }
                if boost && boost_penalty > 0.0 {
                    drain += boost_penalty;
                }
                if topo_enabled && topo_penalty > 0.0 {
                    if slope_along > 0.0 {
                        drain += slope_along * topo_penalty;
                    } else if slope_along < 0.0 {
                        drain = (drain + slope_along * topo_penalty * 0.5).max(0.0);
                    }
                }
                let health_delta = -drain;
                let energy = (runtime.energy - drain).max(0.0);

                let mut spike_length = spike_lengths_snapshot[idx];
                if spike_length < spike_target {
                    spike_length = (spike_length + spike_growth).min(spike_target);
                } else if spike_length > spike_target {
                    spike_length = (spike_length - spike_growth).max(spike_target);
                }
                let spiked = spike_length > 0.5;

                ActuationResult {
                    delta: Some(ActuationDelta {
                        heading,
                        velocity: Velocity::new(vx, vy),
                        position: next_pos,
                        health_delta,
                    }),
                    energy,
                    color,
                    spike_length,
                    sound_level,
                    give_intent,
                    spiked,
                }
            } else {
                ActuationResult::default()
            }
        });

        let columns = self.agents.columns_mut();
        {
            let headings = columns.headings_mut();
            for (idx, result) in results.iter().enumerate() {
                if let Some(delta) = &result.delta {
                    headings[idx] = delta.heading;
                }
            }
        }
        {
            let velocities = columns.velocities_mut();
            for (idx, result) in results.iter().enumerate() {
                if let Some(delta) = &result.delta {
                    velocities[idx] = delta.velocity;
                }
            }
        }
        {
            let healths = columns.health_mut();
            for (idx, result) in results.iter().enumerate() {
                if let Some(delta) = &result.delta {
                    healths[idx] = (healths[idx] + delta.health_delta).clamp(0.0, 2.0);
                }
            }
        }
        {
            let positions = columns.positions_mut();
            for (idx, result) in results.iter().enumerate() {
                if let Some(delta) = &result.delta {
                    positions[idx] = delta.position;
                }
            }
        }
        {
            let colors = columns.colors_mut();
            for (idx, result) in results.iter().enumerate() {
                colors[idx] = result.color;
            }
        }
        {
            let spikes = columns.spike_lengths_mut();
            for (idx, result) in results.iter().enumerate() {
                spikes[idx] = result.spike_length;
            }
        }

        for (idx, agent_id) in handles.iter().enumerate() {
            if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                runtime.energy = results[idx].energy;
                runtime.spiked = results[idx].spiked;
                runtime.sound_output = results[idx].sound_level;
                runtime.sound_multiplier = results[idx].sound_level;
                runtime.give_intent = results[idx].give_intent;
            }
        }
    }

    fn stage_temperature_discomfort(&mut self) {
        let rate = self.config.temperature_discomfort_rate;
        if rate <= 0.0 || self.config.world_width == 0 {
            return;
        }

        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        if handles.is_empty() {
            return;
        }

        let comfort_band = self.config.temperature_comfort_band.clamp(0.0, 1.0);
        let exponent = self
            .config
            .temperature_discomfort_exponent
            .max(f32::EPSILON);

        let positions_snapshot: Vec<Position> = self.agents.columns().positions().to_vec();
        let mut penalties = vec![0.0f32; handles.len()];

        for (idx, agent_id) in handles.iter().enumerate() {
            let env_temperature = sample_temperature(&self.config, positions_snapshot[idx].x);
            let Some(runtime) = self.runtime.get(*agent_id) else {
                continue;
            };
            let mut discomfort =
                temperature_discomfort(env_temperature, runtime.temperature_preference);
            if discomfort <= comfort_band {
                continue;
            }
            discomfort = (discomfort - comfort_band).max(0.0);
            let penalty = rate * discomfort.powf(exponent);
            if penalty > 0.0 {
                penalties[idx] = penalty;
            }
        }

        if penalties.iter().all(|penalty| penalty <= &0.0) {
            return;
        }

        let columns = self.agents.columns_mut();
        let healths = columns.health_mut();

        for (idx, agent_id) in handles.iter().enumerate() {
            let penalty = penalties[idx];
            if penalty <= 0.0 {
                continue;
            }
            if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                let health = &mut healths[idx];
                *health = (*health - penalty).max(0.0);
                runtime.energy = (runtime.energy - penalty).max(0.0);
                runtime.food_delta -= penalty;
                if *health <= 0.0 {
                    self.pending_deaths.push(*agent_id);
                }
            }
        }
    }

    fn stage_reset_events(&mut self) {
        for runtime in self.runtime.values_mut() {
            runtime.spiked = false;
            runtime.food_delta = 0.0;
            runtime.sound_output = runtime.sound_multiplier;
            runtime.give_intent *= 0.9;
            if runtime.indicator.intensity > 0.0 {
                runtime.indicator.intensity = (runtime.indicator.intensity - 1.0).max(0.0);
                if runtime.indicator.intensity <= 0.0 {
                    runtime.indicator = IndicatorState::default();
                }
            }
        }
    }

    fn stage_food(&mut self) {
        let cell_size = self.config.food_cell_size as f32;
        let positions = self.agents.columns().positions().to_vec();
        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        let mut sharers: Vec<usize> = Vec::new();

        let intake_rate = self.config.food_intake_rate.max(0.0);
        let waste_rate = self.config.food_waste_rate.max(0.0);
        let reproduction_bonus = self.config.reproduction_food_bonus.max(0.0);
        for (idx, agent_id) in handles.iter().enumerate() {
            if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                if intake_rate > 0.0 || waste_rate > 0.0 {
                    let pos = positions[idx];
                    let cell_x = (pos.x / cell_size).floor() as u32 % self.food.width();
                    let cell_y = (pos.y / cell_size).floor() as u32 % self.food.height();
                    if let Some(cell) = self.food.get_mut(cell_x, cell_y) {
                        let available = *cell;
                        if available > 0.0 {
                            let base_intake = available.min(intake_rate);
                            let waste = available.min(waste_rate);
                            let herbivore = clamp01(runtime.herbivore_tendency);
                            let mut intake = 0.0;
                            if herbivore > 0.0 && base_intake > 0.0 {
                                let left = runtime
                                    .outputs
                                    .first()
                                    .copied()
                                    .unwrap_or(0.0)
                                    .clamp(0.0, 1.0);
                                let right = runtime
                                    .outputs
                                    .get(1)
                                    .copied()
                                    .unwrap_or(0.0)
                                    .clamp(0.0, 1.0);
                                let average_speed = (left.abs() + right.abs()) * 0.5;
                                let speed_scale = (1.0 - average_speed).clamp(0.0, 1.0) * 0.7 + 0.3;
                                intake = base_intake * herbivore * speed_scale;
                            }
                            if waste > 0.0 {
                                *cell = (available - waste).max(0.0);
                            }
                            if intake > 0.0 {
                                runtime.energy = (runtime.energy + intake).min(2.0);
                                runtime.food_delta += intake;
                                if reproduction_bonus > 0.0 {
                                    runtime.reproduction_counter += intake * reproduction_bonus;
                                }
                            }
                        }
                    }
                }
                if runtime.give_intent > 0.5 {
                    sharers.push(idx);
                }
            }
        }

        if sharers.is_empty() {
            return;
        }

        let transfer_rate = self.config.food_transfer_rate;
        if transfer_rate <= 0.0 {
            return;
        }
        let distance = if self.config.food_sharing_distance > 0.0 {
            self.config.food_sharing_distance
        } else {
            self.config.food_sharing_radius
        };
        let distance_sq = distance * distance;
        let world_width = self.config.world_width as f32;
        let world_height = self.config.world_height as f32;

        for &giver_idx in &sharers {
            let giver_id = handles[giver_idx];
            for (recipient_idx, recipient_id) in handles.iter().enumerate() {
                if recipient_idx == giver_idx {
                    continue;
                }
                let dx = toroidal_delta(
                    positions[recipient_idx].x,
                    positions[giver_idx].x,
                    world_width,
                );
                let dy = toroidal_delta(
                    positions[recipient_idx].y,
                    positions[giver_idx].y,
                    world_height,
                );
                if dx * dx + dy * dy > distance_sq {
                    continue;
                }
                let recipient_energy = match self.runtime.get(*recipient_id) {
                    Some(runtime) => runtime.energy,
                    None => continue,
                };
                if recipient_energy >= 2.0 - f32::EPSILON {
                    continue;
                }
                let giver_energy = match self.runtime.get(giver_id) {
                    Some(runtime) => runtime.energy,
                    None => break,
                };
                if giver_energy <= f32::EPSILON {
                    break;
                }
                let capacity = (2.0 - recipient_energy).max(0.0);
                if capacity <= 0.0 {
                    continue;
                }
                let actual_transfer = transfer_rate.min(giver_energy).min(capacity);
                if actual_transfer <= 0.0 {
                    continue;
                }
                {
                    if let Some(giver_runtime) = self.runtime.get_mut(giver_id) {
                        giver_runtime.energy = (giver_runtime.energy - actual_transfer).max(0.0);
                        giver_runtime.food_delta -= actual_transfer;
                    } else {
                        break;
                    }
                }
                if let Some(recipient_runtime) = self.runtime.get_mut(*recipient_id) {
                    recipient_runtime.energy =
                        (recipient_runtime.energy + actual_transfer).min(2.0);
                    recipient_runtime.food_delta += actual_transfer;
                }
                self.pulse_indicator(giver_id, 10.0, [1.0, 1.0, 1.0]);
                self.pulse_indicator(*recipient_id, 10.0, [1.0, 1.0, 1.0]);
            }
        }
    }
    fn spawn_crossover_agent(&mut self) -> bool {
        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        let count = handles.len();
        if count < 2 {
            return false;
        }

        let (idx1, idx2) = {
            let columns = self.agents.columns();
            let ages = columns.ages();
            let mut first = self.rng.random_range(0..count);
            let mut second = if count > 1 {
                self.rng.random_range(0..count)
            } else {
                first
            };
            if count > 1 {
                while second == first {
                    second = self.rng.random_range(0..count);
                }
            }
            for (idx, &age) in ages.iter().enumerate() {
                if age > ages[first] && self.rng.random_range(0.0..1.0) < 0.1 {
                    first = idx;
                }
                if idx != first && age > ages[second] && self.rng.random_range(0.0..1.0) < 0.1 {
                    second = idx;
                }
            }
            if first == second {
                second = (second + 1) % count;
                if second == first {
                    return false;
                }
            }
            (first, second)
        };

        let parent_id = handles[idx1];
        let partner_id = handles[idx2];

        let parent_data = {
            let columns = self.agents.columns();
            columns.snapshot(idx1)
        };
        let partner_data = {
            let columns = self.agents.columns();
            columns.snapshot(idx2)
        };
        let parent_runtime = match self.runtime.get(parent_id).cloned() {
            Some(rt) => rt,
            None => return false,
        };
        let partner_runtime = self.runtime.get(partner_id).cloned();

        let width = self.config.world_width as f32;
        let height = self.config.world_height as f32;
        let child_data = self.build_child_data(
            &parent_data,
            Some(&partner_data),
            self.config.reproduction_spawn_jitter,
            self.config.reproduction_spawn_back_distance,
            self.config.reproduction_color_jitter,
            width,
            height,
        );
        let child_runtime = self.build_child_runtime(
            &parent_runtime,
            partner_runtime.as_ref(),
            self.config.reproduction_gene_log_capacity,
            parent_id,
            Some(partner_id),
        );

        let child_id = self.spawn_agent(child_data);
        if let Some(runtime) = self.runtime.get_mut(child_id) {
            *runtime = child_runtime;
            true
        } else {
            false
        }
    }

    fn stage_population(&mut self, next_tick: Tick) {
        if self.closed {
            return;
        }

        let minimum = self.config.population_minimum;
        if minimum > 0 {
            while self.agents.len() < minimum {
                self.spawn_random_agent();
            }
        }

        let interval = self.config.population_spawn_interval;
        if interval == 0 {
            return;
        }
        if !next_tick.0.is_multiple_of(interval as u64) {
            return;
        }

        let spawn_count = self.config.population_spawn_count.max(1);
        let crossover_chance = self.config.population_crossover_chance.clamp(0.0, 1.0);
        for _ in 0..spawn_count {
            let use_crossover = self.agents.len() >= 2
                && crossover_chance > 0.0
                && self.rng.random_range(0.0..1.0) < crossover_chance;
            let spawned = if use_crossover {
                self.spawn_crossover_agent()
            } else {
                false
            };
            if !spawned {
                self.spawn_random_agent();
            }
        }
    }

    fn spawn_random_agent(&mut self) {
        let width = self.config.world_width as f32;
        let height = self.config.world_height as f32;
        let position = Position::new(
            self.rng.random_range(0.0..width),
            self.rng.random_range(0.0..height),
        );
        let heading = self
            .rng
            .random_range(-std::f32::consts::PI..std::f32::consts::PI);
        let color = [
            self.rng.random_range(0.0..1.0),
            self.rng.random_range(0.0..1.0),
            self.rng.random_range(0.0..1.0),
        ];
        let data = AgentData::new(
            position,
            Velocity::default(),
            heading,
            1.0,
            color,
            0.0,
            false,
            0,
            Generation::default(),
        );
        let _ = self.spawn_agent(data);
    }
    fn pulse_indicator(&mut self, id: AgentId, intensity: f32, color: [f32; 3]) {
        if let Some(runtime) = self.runtime.get_mut(id) {
            runtime.indicator.intensity = (runtime.indicator.intensity + intensity).min(100.0);
            runtime.indicator.color = color;
        }
    }

    fn stage_combat(&mut self) {
        let spike_radius = self.config.spike_radius;
        if spike_radius <= 0.0 {
            return;
        }

        for runtime in self.runtime.values_mut() {
            runtime.combat = CombatEventFlags::default();
        }

        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        if handles.is_empty() {
            return;
        }

        let world_w = self.config.world_width as f32;
        let world_h = self.config.world_height as f32;
        let min_length = self.config.spike_min_length;
        let alignment_threshold = self.config.spike_alignment_cosine.clamp(0.0, 1.0);
        let speed_bonus = self.config.spike_speed_damage_bonus;
        let length_bonus = self.config.spike_length_damage_bonus;
        let carnivore_threshold = self.config.carnivore_threshold;

        let positions = self.agents.columns().positions();
        let headings = self.agents.columns().headings();
        let velocities: Vec<Velocity> = self.agents.columns().velocities().to_vec();
        let spike_lengths = self.agents.columns().spike_lengths();
        let positions_pairs: Vec<(f32, f32)> = positions.iter().map(|p| (p.x, p.y)).collect();
        let _ = self.index.rebuild(&positions_pairs);

        let spike_damage = self.config.spike_damage;
        let spike_energy_cost = self.config.spike_energy_cost;
        let index = &self.index;
        let runtime_snapshot: Vec<AgentRuntime> = handles
            .iter()
            .map(|id| self.runtime.get(*id).cloned().unwrap_or_default())
            .collect();

        let results: Vec<CombatResult> = collect_handles!(handles, |idx, _handle| {
            let mut result = CombatResult::default();
            let attacker_runtime = &runtime_snapshot[idx];

            let is_carnivore = attacker_runtime.herbivore_tendency < carnivore_threshold;
            result.attacker_carnivore = is_carnivore;
            let energy_before = attacker_runtime.energy;

            let spike_power = attacker_runtime
                .outputs
                .get(5)
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);

            if !attacker_runtime.spiked {
                result.energy = energy_before;
                return result;
            }

            if !is_carnivore {
                result.energy = (energy_before - spike_energy_cost * spike_power).max(0.0);
                return result;
            }
            if spike_power <= f32::EPSILON {
                result.energy = energy_before;
                return result;
            }

            let spike_length = spike_lengths[idx];
            if spike_length < min_length {
                result.energy = (energy_before - spike_energy_cost * spike_power).max(0.0);
                return result;
            }

            let reach = (spike_radius + spike_length).max(1.0);
            let reach_sq = reach * reach;
            let heading = headings[idx];
            let facing = (heading.cos(), heading.sin());
            let wheel_left = attacker_runtime
                .outputs
                .first()
                .copied()
                .unwrap_or(0.0)
                .abs();
            let wheel_right = attacker_runtime
                .outputs
                .get(1)
                .copied()
                .unwrap_or(0.0)
                .abs();
            let velocity = velocities[idx];
            let speed_mag = (velocity.vx * velocity.vx + velocity.vy * velocity.vy).sqrt();
            let boost = attacker_runtime
                .outputs
                .get(3)
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);

            let base_power = spike_damage * spike_power;
            let length_factor = 1.0 + spike_length * length_bonus;
            let speed_factor =
                1.0 + (wheel_left.max(wheel_right) + speed_mag) * speed_bonus + boost;
            let base_damage = base_power * length_factor * speed_factor;

            let origin = positions[idx];
            let mut hits = Vec::new();
            index.neighbors_within(
                idx,
                reach_sq,
                &mut |other_idx, _dist_sq: OrderedFloat<f32>| {
                    if other_idx == idx {
                        return;
                    }
                    let target_runtime = &runtime_snapshot[other_idx];
                    let dx = Self::wrap_delta(origin.x, positions[other_idx].x, world_w);
                    let dy = Self::wrap_delta(origin.y, positions[other_idx].y, world_h);
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq <= f32::EPSILON || dist_sq > reach_sq {
                        return;
                    }
                    let dist = dist_sq.sqrt();
                    let dir_x = dx / dist;
                    let dir_y = dy / dist;
                    let alignment = facing.0 * dir_x + facing.1 * dir_y;
                    if alignment < alignment_threshold {
                        return;
                    }

                    let damage = base_damage * alignment.max(0.0);
                    if damage <= 0.0 {
                        return;
                    }
                    let victim_carnivore = target_runtime.herbivore_tendency < carnivore_threshold;
                    if victim_carnivore {
                        result.hit_carnivore = true;
                    } else {
                        result.hit_herbivore = true;
                    }
                    hits.push(CombatHit {
                        target_idx: other_idx,
                        damage,
                        attacker_carnivore: is_carnivore,
                    });
                },
            );

            result.total_damage = hits.iter().map(|hit| hit.damage).sum();
            result.hits = hits;
            result.energy = (energy_before - spike_energy_cost * spike_power).max(0.0);
            result
        });

        let mut buckets = vec![DamageBucket::default(); handles.len()];
        let columns = self.agents.columns_mut();
        let healths = columns.health_mut();

        for (idx, agent_id) in handles.iter().enumerate() {
            if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                runtime.energy = results[idx].energy;
                if results[idx].total_damage > 0.0 {
                    runtime.combat.spike_attacker = true;
                    if results[idx].hit_carnivore {
                        runtime.combat.hit_carnivore = true;
                    }
                    if results[idx].hit_herbivore {
                        runtime.combat.hit_herbivore = true;
                    }
                    let attacker_color = if results[idx].attacker_carnivore {
                        [1.0, 0.5, 0.2]
                    } else {
                        [0.4, 0.9, 0.4]
                    };
                    runtime.indicator = IndicatorState {
                        intensity: (runtime.indicator.intensity + results[idx].total_damage * 25.0)
                            .min(100.0),
                        color: attacker_color,
                    };
                }
            }
            for hit in &results[idx].hits {
                if let Some(bucket) = buckets.get_mut(hit.target_idx) {
                    bucket.total += hit.damage;
                    if hit.attacker_carnivore {
                        bucket.carnivore += hit.damage;
                    } else {
                        bucket.herbivore += hit.damage;
                    }
                }
            }
        }

        for (idx, bucket) in buckets.into_iter().enumerate() {
            if bucket.total <= 0.0 {
                continue;
            }
            healths[idx] = (healths[idx] - bucket.total).max(0.0);
            let victim_id = handles[idx];
            if let Some(runtime) = self.runtime.get_mut(victim_id) {
                runtime.food_delta -= bucket.total;
                runtime.spiked = true;
                runtime.combat.spike_victim = true;
                if bucket.carnivore > 0.0 {
                    runtime.combat.was_spiked_by_carnivore = true;
                }
                if bucket.herbivore > 0.0 {
                    runtime.combat.was_spiked_by_herbivore = true;
                }
                let victim_color = if bucket.carnivore >= bucket.herbivore {
                    [1.0, 0.2, 0.2]
                } else {
                    [1.0, 0.8, 0.2]
                };
                runtime.indicator = IndicatorState {
                    intensity: (runtime.indicator.intensity + bucket.total * 30.0).min(100.0),
                    color: victim_color,
                };
            }
            if healths[idx] <= 0.0 {
                self.pending_deaths.push(victim_id);
            }
        }

        let spike_columns = columns.spike_lengths_mut();
        for (idx, result) in results.iter().enumerate() {
            if result.total_damage <= 0.0 {
                continue;
            }
            if let Some(spike_len) = spike_columns.get_mut(idx) {
                *spike_len = (*spike_len * 0.25).max(0.0_f32);
            }
        }
    }

    fn distribute_carcass_rewards(&mut self, dead: &[(usize, AgentId)]) {
        if dead.is_empty() {
            return;
        }
        let radius = self.config.carcass_distribution_radius;
        let health_base = self.config.carcass_health_reward;
        let reproduction_base = self.config.carcass_reproduction_reward;
        if radius <= 0.0 || (health_base <= 0.0 && reproduction_base <= 0.0) {
            return;
        }

        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        if handles.is_empty() {
            return;
        }

        let positions: Vec<Position> = self.agents.columns().positions().to_vec();
        let ages: Vec<u32> = self.agents.columns().ages().to_vec();
        let healths: Vec<f32> = self.agents.columns().health().to_vec();

        let agent_count = handles.len();
        let mut health_add = vec![0.0f32; agent_count];
        let mut energy_add = vec![0.0f32; agent_count];
        let mut reproduction_bonus = vec![0.0f32; agent_count];
        let mut indicator_add = vec![0.0f32; agent_count];

        let radius_sq = radius * radius;
        let exponent = self.config.carcass_neighbor_exponent.max(1.0);
        let maturity_age = self.config.carcass_maturity_age.max(1);
        let energy_rate = self.config.carcass_energy_share_rate.max(0.0);
        let indicator_scale = self.config.carcass_indicator_scale.max(0.0);
        let width = self.config.world_width as f32;
        let height = self.config.world_height as f32;

        for (dense_idx, agent_id) in dead {
            let Some(victim_runtime) = self.runtime.get(*agent_id) else {
                continue;
            };
            if !victim_runtime.spiked {
                continue;
            }
            let victim_index = *dense_idx;
            if victim_index >= agent_count {
                continue;
            }
            if healths.get(victim_index).copied().unwrap_or(1.0) > 0.0 {
                continue;
            }
            let victim_pos = positions.get(victim_index).copied().unwrap_or_default();
            let age = ages.get(victim_index).copied().unwrap_or(0);
            let age_multiplier = if age < maturity_age {
                (age as f32) / (maturity_age as f32)
            } else {
                1.0
            };
            if age_multiplier <= 0.0 {
                continue;
            }

            let mut neighbor_indices = Vec::new();
            for (idx, neighbor_id) in handles.iter().enumerate() {
                if *neighbor_id == *agent_id {
                    continue;
                }
                if healths.get(idx).copied().unwrap_or(0.0) <= 0.0 {
                    continue;
                }
                let dx = toroidal_delta(positions[idx].x, victim_pos.x, width);
                let dy = toroidal_delta(positions[idx].y, victim_pos.y, height);
                if dx * dx + dy * dy <= radius_sq {
                    neighbor_indices.push(idx);
                }
            }
            if neighbor_indices.is_empty() {
                continue;
            }
            let count = neighbor_indices.len() as f32;
            let norm = count.powf(exponent);

            for idx in neighbor_indices {
                if let Some(runtime_neighbor) = self.runtime.get(handles[idx]) {
                    let herb = clamp01(runtime_neighbor.herbivore_tendency);
                    let carnivore_factor = (1.0 - herb) * (1.0 - herb);
                    if carnivore_factor <= f32::EPSILON {
                        continue;
                    }
                    if health_base > 0.0 {
                        let share = health_base * carnivore_factor * age_multiplier / norm;
                        if share > 0.0 {
                            health_add[idx] += share;
                            if energy_rate > 0.0 {
                                energy_add[idx] += share * energy_rate;
                            }
                            if indicator_scale > 0.0 {
                                indicator_add[idx] += share * indicator_scale;
                            }
                            self.carcass_health_distributed += share;
                        }
                    }
                    if reproduction_base > 0.0 {
                        let bonus = reproduction_base * carnivore_factor * age_multiplier / norm;
                        if bonus > 0.0 {
                            reproduction_bonus[idx] += bonus;
                            self.carcass_reproduction_bonus += bonus;
                            if indicator_scale > 0.0 && health_base <= 0.0 {
                                indicator_add[idx] += indicator_scale;
                            }
                        }
                    }
                }
            }
        }

        if health_add.iter().any(|v| *v > 0.0) {
            let columns = self.agents.columns_mut();
            let healths_mut = columns.health_mut();
            for (idx, add) in health_add.iter().enumerate() {
                if *add > 0.0 {
                    healths_mut[idx] = (healths_mut[idx] + *add).min(2.0);
                }
            }
        }

        if energy_add.iter().any(|v| *v > 0.0)
            || reproduction_bonus.iter().any(|v| *v > 0.0)
            || indicator_add.iter().any(|v| *v > 0.0)
        {
            for (idx, agent_id) in handles.iter().enumerate() {
                if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                    let energy = energy_add[idx];
                    if energy > 0.0 {
                        runtime.energy = (runtime.energy + energy).min(2.0);
                        runtime.food_delta += energy;
                    }
                    let repro = reproduction_bonus[idx];
                    if repro > 0.0 {
                        runtime.reproduction_counter += repro;
                    }
                    let indicator_bonus = indicator_add[idx];
                    if indicator_bonus > 0.0 {
                        runtime.indicator.intensity =
                            (runtime.indicator.intensity + indicator_bonus).min(100.0);
                        runtime.indicator.color = [1.0, 1.0, 1.0];
                    }
                }
            }
        }
    }

    fn stage_death_cleanup(&mut self) {
        if self.pending_deaths.is_empty() {
            return;
        }
        let mut seen = HashSet::new();
        let mut dead = Vec::new();
        for agent_id in self.pending_deaths.drain(..) {
            if seen.insert(agent_id)
                && self.agents.contains(agent_id)
                && let Some(idx) = self.agents.index_of(agent_id)
            {
                dead.push((idx, agent_id));
            }
        }
        if dead.is_empty() {
            self.last_deaths = 0;
            return;
        }
        dead.sort_by_key(|(idx, _)| *idx);
        self.distribute_carcass_rewards(&dead);
        let mut removed = 0usize;
        for (_, agent_id) in dead.into_iter().rev() {
            if self.remove_agent(agent_id).is_some() {
                removed += 1;
            }
        }
        self.last_deaths = removed;
    }

    fn stage_reproduction(&mut self) {
        if self.config.reproduction_energy_threshold <= 0.0 {
            return;
        }

        let width = self.config.world_width as f32;
        let height = self.config.world_height as f32;
        let jitter = self.config.reproduction_spawn_jitter;
        let back_offset = self.config.reproduction_spawn_back_distance;
        let color_jitter = self.config.reproduction_color_jitter;
        let partner_chance = self.config.reproduction_partner_chance;
        let gene_log_capacity = self.config.reproduction_gene_log_capacity;
        let cooldown = self.config.reproduction_cooldown.max(1) as f32;
        let rate_carnivore = self.config.reproduction_rate_carnivore;
        let rate_herbivore = self.config.reproduction_rate_herbivore;

        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        if handles.is_empty() {
            return;
        }

        let columns = self.agents.columns();
        let parent_snapshots: Vec<AgentData> = (0..columns.len())
            .map(|idx| columns.snapshot(idx))
            .collect();
        let ages: Vec<u32> = columns.ages().to_vec();
        let runtime_snapshots: Vec<AgentRuntime> = handles
            .iter()
            .map(|id| self.runtime.get(*id).cloned().unwrap_or_default())
            .collect();

        for (idx, agent_id) in handles.iter().enumerate() {
            let mut parent_runtime_snapshot = None;
            {
                let runtime = match self.runtime.get_mut(*agent_id) {
                    Some(rt) => rt,
                    None => continue,
                };
                let herb = runtime.herbivore_tendency.clamp(0.0, 1.0);
                let reproduction_rate = rate_carnivore + (rate_herbivore - rate_carnivore) * herb;
                runtime.reproduction_counter += reproduction_rate;
                if runtime.energy < self.config.reproduction_energy_threshold {
                    continue;
                }
                if runtime.reproduction_counter < cooldown {
                    continue;
                }
                if runtime.energy < self.config.reproduction_energy_cost {
                    continue;
                }
                runtime.energy -= self.config.reproduction_energy_cost;
                runtime.reproduction_counter = 0.0;
                parent_runtime_snapshot = Some(runtime.clone());
            }

            let Some(parent_runtime_snapshot) = parent_runtime_snapshot else {
                continue;
            };

            let partner_index =
                self.select_partner_index(idx, &ages, partner_chance, handles.len());
            let partner_data = partner_index.map(|j| parent_snapshots[j]);
            let partner_runtime = partner_index.map(|j| runtime_snapshots[j].clone());

            let child_data = self.build_child_data(
                &parent_snapshots[idx],
                partner_data.as_ref(),
                jitter,
                back_offset,
                color_jitter,
                width,
                height,
            );
            let child_runtime = self.build_child_runtime(
                &parent_runtime_snapshot,
                partner_runtime.as_ref(),
                gene_log_capacity,
                *agent_id,
                partner_index.map(|j| handles[j]),
            );
            self.pending_spawns.push(SpawnOrder {
                parent_index: idx,
                data: child_data,
                runtime: child_runtime,
            });
        }
    }

    fn select_partner_index(
        &mut self,
        parent_idx: usize,
        ages: &[u32],
        partner_chance: f32,
        population: usize,
    ) -> Option<usize> {
        if population < 2 || partner_chance <= 0.0 {
            return None;
        }
        if self.rng.random_range(0.0..1.0) >= partner_chance {
            return None;
        }
        let mut best: Option<(usize, u32)> = None;
        for (idx, age) in ages.iter().enumerate() {
            if idx == parent_idx {
                continue;
            }
            match best {
                Some((best_idx, best_age)) => {
                    if *age > best_age || (*age == best_age && idx < best_idx) {
                        best = Some((idx, *age));
                    }
                }
                None => best = Some((idx, *age)),
            }
        }
        best.map(|(idx, _)| idx)
    }

    fn stage_spawn_commit(&mut self) {
        if self.pending_spawns.is_empty() {
            return;
        }
        let mut orders = std::mem::take(&mut self.pending_spawns);
        orders.sort_by_key(|order| order.parent_index);
        self.last_births = orders.len();
        for order in orders {
            let child_id = self.spawn_agent(order.data);
            if let Some(runtime) = self.runtime.get_mut(child_id) {
                *runtime = order.runtime;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_child_data(
        &mut self,
        parent: &AgentData,
        partner: Option<&AgentData>,
        jitter: f32,
        back_offset: f32,
        color_jitter: f32,
        width: f32,
        height: f32,
    ) -> AgentData {
        let mut child = *parent;
        let heading = parent.heading;
        let base_dx = -heading.cos() * back_offset;
        let base_dy = -heading.sin() * back_offset;
        let jitter_dx = if jitter > 0.0 {
            self.rng.random_range(-jitter..jitter)
        } else {
            0.0
        };
        let jitter_dy = if jitter > 0.0 {
            self.rng.random_range(-jitter..jitter)
        } else {
            0.0
        };
        child.position.x = Self::wrap_position(parent.position.x + base_dx + jitter_dx, width);
        child.position.y = Self::wrap_position(parent.position.y + base_dy + jitter_dy, height);
        child.velocity = Velocity::default();
        child.heading = wrap_signed_angle(parent.heading + self.rng.random_range(-0.2..0.2));
        child.health = 1.0;
        child.boost = false;
        child.age = 0;
        child.spike_length = 0.0;
        child.generation = parent.generation.next();

        if let Some(partner) = partner {
            for (channel, partner_channel) in child.color.iter_mut().zip(partner.color.iter()) {
                *channel = ((*channel + partner_channel) * 0.5).clamp(0.0, 1.0);
            }
        }

        if color_jitter > 0.0 {
            for channel in &mut child.color {
                *channel =
                    (*channel + self.rng.random_range(-color_jitter..color_jitter)).clamp(0.0, 1.0);
            }
        }
        child
    }

    fn build_child_runtime(
        &mut self,
        parent: &AgentRuntime,
        partner: Option<&AgentRuntime>,
        gene_log_capacity: usize,
        parent_id: AgentId,
        partner_id: Option<AgentId>,
    ) -> AgentRuntime {
        let mut runtime = parent.clone();
        runtime.energy = self.config.reproduction_child_energy.clamp(0.0, 2.0);
        runtime.reproduction_counter = 0.0;
        runtime.sensors = [0.0; INPUT_SIZE];
        runtime.outputs = [0.0; OUTPUT_SIZE];
        runtime.food_delta = 0.0;
        runtime.spiked = false;
        runtime.sound_output = 0.0;
        runtime.give_intent = 0.0;
        runtime.combat = CombatEventFlags::default();
        runtime.indicator = IndicatorState::default();
        runtime.selection = SelectionState::None;
        runtime.mutation_log.clear();
        runtime.brain = BrainBinding::default();
        runtime.lineage = [Some(parent_id), partner_id];

        if let Some(partner_runtime) = partner {
            runtime.hybrid = true;
            let blend = self.rng.random_range(0.35..0.65);
            let mix = |a: f32, b: f32| lerp(a, b, blend);

            let before = runtime.herbivore_tendency;
            runtime.herbivore_tendency = mix(
                parent.herbivore_tendency,
                partner_runtime.herbivore_tendency,
            )
            .clamp(0.0, 1.0);
            runtime.log_change(
                gene_log_capacity,
                "herbivore",
                before,
                runtime.herbivore_tendency,
            );

            let before_smell = runtime.trait_modifiers.smell;
            runtime.trait_modifiers.smell = mix(
                parent.trait_modifiers.smell,
                partner_runtime.trait_modifiers.smell,
            );
            runtime.log_change(
                gene_log_capacity,
                "smell",
                before_smell,
                runtime.trait_modifiers.smell,
            );

            let before_sound = runtime.trait_modifiers.sound;
            runtime.trait_modifiers.sound = mix(
                parent.trait_modifiers.sound,
                partner_runtime.trait_modifiers.sound,
            );
            runtime.log_change(
                gene_log_capacity,
                "sound",
                before_sound,
                runtime.trait_modifiers.sound,
            );

            let before_hearing = runtime.trait_modifiers.hearing;
            runtime.trait_modifiers.hearing = mix(
                parent.trait_modifiers.hearing,
                partner_runtime.trait_modifiers.hearing,
            );
            runtime.log_change(
                gene_log_capacity,
                "hearing",
                before_hearing,
                runtime.trait_modifiers.hearing,
            );

            let before_eye = runtime.trait_modifiers.eye;
            runtime.trait_modifiers.eye = mix(
                parent.trait_modifiers.eye,
                partner_runtime.trait_modifiers.eye,
            );
            runtime.log_change(
                gene_log_capacity,
                "eye",
                before_eye,
                runtime.trait_modifiers.eye,
            );

            let before_blood = runtime.trait_modifiers.blood;
            runtime.trait_modifiers.blood = mix(
                parent.trait_modifiers.blood,
                partner_runtime.trait_modifiers.blood,
            );
            runtime.log_change(
                gene_log_capacity,
                "blood",
                before_blood,
                runtime.trait_modifiers.blood,
            );

            let before_primary = runtime.mutation_rates.primary;
            runtime.mutation_rates.primary = mix(
                parent.mutation_rates.primary,
                partner_runtime.mutation_rates.primary,
            )
            .max(0.0001);
            runtime.log_change(
                gene_log_capacity,
                "mut_rate_primary",
                before_primary,
                runtime.mutation_rates.primary,
            );

            let before_secondary = runtime.mutation_rates.secondary;
            runtime.mutation_rates.secondary = mix(
                parent.mutation_rates.secondary,
                partner_runtime.mutation_rates.secondary,
            )
            .max(0.001);
            runtime.log_change(
                gene_log_capacity,
                "mut_rate_secondary",
                before_secondary,
                runtime.mutation_rates.secondary,
            );

            runtime.clocks[0] = if self.rng.random_range(0.0..1.0) < 0.5 {
                parent.clocks[0]
            } else {
                partner_runtime.clocks[0]
            };
            runtime.clocks[1] = if self.rng.random_range(0.0..1.0) < 0.5 {
                parent.clocks[1]
            } else {
                partner_runtime.clocks[1]
            };

            let before_temp = runtime.temperature_preference;
            runtime.temperature_preference = mix(
                parent.temperature_preference,
                partner_runtime.temperature_preference,
            )
            .clamp(0.0, 1.0);
            runtime.log_change(
                gene_log_capacity,
                "temp_pref",
                before_temp,
                runtime.temperature_preference,
            );

            runtime.push_gene_log(
                gene_log_capacity,
                format!("hybrid crossover ({:.2})", blend),
            );
        } else {
            runtime.hybrid = false;
            runtime.lineage[1] = None;
        }

        let meta_chance = self.config.reproduction_meta_mutation_chance;
        let meta_scale = self.config.reproduction_meta_mutation_scale;
        if meta_chance > 0.0 && meta_scale > 0.0 && self.rng.random_range(0.0..1.0) < meta_chance {
            let delta_primary = self.rng.random_range(-meta_scale..meta_scale);
            let before = runtime.mutation_rates.primary;
            runtime.mutation_rates.primary =
                (runtime.mutation_rates.primary + delta_primary).max(0.0001);
            runtime.log_change(
                gene_log_capacity,
                "meta_mut_primary",
                before,
                runtime.mutation_rates.primary,
            );

            let delta_secondary = self.rng.random_range(-meta_scale..meta_scale);
            let before = runtime.mutation_rates.secondary;
            runtime.mutation_rates.secondary =
                (runtime.mutation_rates.secondary + delta_secondary).max(0.001);
            runtime.log_change(
                gene_log_capacity,
                "meta_mut_secondary",
                before,
                runtime.mutation_rates.secondary,
            );
        }

        let mutation_scale =
            runtime.mutation_rates.secondary * self.config.reproduction_mutation_scale;
        let primary_rate = runtime.mutation_rates.primary;
        if mutation_scale > 0.0 {
            let before = runtime.herbivore_tendency;
            runtime.herbivore_tendency =
                self.mutate_value(runtime.herbivore_tendency, mutation_scale, 0.0, 1.0);
            runtime.log_change(
                gene_log_capacity,
                "mut_herbivore",
                before,
                runtime.herbivore_tendency,
            );

            let (before_smell, after_smell) = {
                let before = runtime.trait_modifiers.smell;
                let after = self.mutate_value(before, mutation_scale, 0.05, 3.0);
                runtime.trait_modifiers.smell = after;
                (before, after)
            };
            runtime.log_change(gene_log_capacity, "mut_smell", before_smell, after_smell);

            let (before_sound, after_sound) = {
                let before = runtime.trait_modifiers.sound;
                let after = self.mutate_value(before, mutation_scale, 0.05, 3.0);
                runtime.trait_modifiers.sound = after;
                (before, after)
            };
            runtime.log_change(gene_log_capacity, "mut_sound", before_sound, after_sound);

            let (before_hearing, after_hearing) = {
                let before = runtime.trait_modifiers.hearing;
                let after = self.mutate_value(before, mutation_scale, 0.1, 4.0);
                runtime.trait_modifiers.hearing = after;
                (before, after)
            };
            runtime.log_change(
                gene_log_capacity,
                "mut_hearing",
                before_hearing,
                after_hearing,
            );

            let (before_eye, after_eye) = {
                let before = runtime.trait_modifiers.eye;
                let after = self.mutate_value(before, mutation_scale, 0.5, 4.0);
                runtime.trait_modifiers.eye = after;
                (before, after)
            };
            runtime.log_change(gene_log_capacity, "mut_eye", before_eye, after_eye);

            let (before_blood, after_blood) = {
                let before = runtime.trait_modifiers.blood;
                let after = self.mutate_value(before, mutation_scale, 0.5, 4.0);
                runtime.trait_modifiers.blood = after;
                (before, after)
            };
            runtime.log_change(gene_log_capacity, "mut_blood", before_blood, after_blood);

            for i in 0..runtime.clocks.len() {
                let before = runtime.clocks[i];
                let after = self.mutate_value_with_probability(
                    runtime.clocks[i],
                    primary_rate,
                    mutation_scale,
                    2.0,
                    200.0,
                );
                runtime.clocks[i] = after;
                runtime.log_change(
                    gene_log_capacity,
                    if i == 0 { "clock1" } else { "clock2" },
                    before,
                    after,
                );
            }

            let before_temp = runtime.temperature_preference;
            runtime.temperature_preference = self.mutate_value_with_probability(
                runtime.temperature_preference,
                primary_rate,
                mutation_scale,
                0.0,
                1.0,
            );
            runtime.log_change(
                gene_log_capacity,
                "mut_temp_pref",
                before_temp,
                runtime.temperature_preference,
            );

            for i in 0..runtime.eye_fov.len() {
                let before = runtime.eye_fov[i];
                let after = self.mutate_value_with_probability(
                    runtime.eye_fov[i],
                    primary_rate,
                    mutation_scale,
                    0.2,
                    4.5,
                );
                runtime.eye_fov[i] = after;
                runtime.log_change(gene_log_capacity, &format!("eye_fov{}", i), before, after);
            }
            for i in 0..runtime.eye_direction.len() {
                let before = runtime.eye_direction[i];
                let after =
                    if primary_rate > 0.0 && self.rng.random_range(0.0..1.0) < primary_rate * 5.0 {
                        let delta = self.rng.random_range(-mutation_scale..mutation_scale);
                        wrap_unsigned_angle(runtime.eye_direction[i] + delta)
                    } else {
                        wrap_unsigned_angle(runtime.eye_direction[i])
                    };
                runtime.eye_direction[i] = after;
                if (after - before).abs() > 1e-4 {
                    runtime.push_gene_log(
                        gene_log_capacity,
                        format!("eye_dir{}: {:.3}->{:.3}", i, before, after),
                    );
                }
            }
        }

        runtime
    }

    fn mutate_value(&mut self, value: f32, scale: f32, min: f32, max: f32) -> f32 {
        if scale <= 0.0 {
            return value.clamp(min, max);
        }
        let delta = self.rng.random_range(-scale..scale);
        (value + delta).clamp(min, max)
    }

    fn mutate_value_with_probability(
        &mut self,
        value: f32,
        rate: f32,
        scale: f32,
        min: f32,
        max: f32,
    ) -> f32 {
        if scale <= 0.0 || rate <= 0.0 {
            return value.clamp(min, max);
        }
        if self.rng.random_range(0.0..1.0) < rate * 5.0 {
            self.mutate_value(value, scale, min, max)
        } else {
            value.clamp(min, max)
        }
    }

    fn stage_persistence(&mut self, next_tick: Tick) {
        if self.config.persistence_interval == 0
            || !next_tick
                .0
                .is_multiple_of(self.config.persistence_interval as u64)
        {
            self.last_births = 0;
            self.last_deaths = 0;
            return;
        }

        let agent_count = self.agents.len();
        let mut total_energy = 0.0;
        for id in self.agents.iter_handles() {
            if let Some(runtime) = self.runtime.get(id) {
                total_energy += runtime.energy;
            }
        }
        let average_energy = if agent_count > 0 {
            total_energy / agent_count as f32
        } else {
            0.0
        };
        let healths = self.agents.columns().health();
        let total_health: f32 = healths.iter().sum();
        let average_health = if agent_count > 0 {
            total_health / agent_count as f32
        } else {
            0.0
        };

        let summary = TickSummary {
            tick: next_tick,
            agent_count,
            births: self.last_births,
            deaths: self.last_deaths,
            total_energy,
            average_energy,
            average_health,
        };
        let mut metrics = vec![
            MetricSample::from_f32("total_energy", summary.total_energy),
            MetricSample::from_f32("average_energy", summary.average_energy),
            MetricSample::from_f32("average_health", summary.average_health),
        ];
        if self.carcass_health_distributed > 0.0 {
            metrics.push(MetricSample::from_f32(
                "carcass_health_distributed",
                self.carcass_health_distributed,
            ));
        }
        if self.carcass_reproduction_bonus > 0.0 {
            metrics.push(MetricSample::from_f32(
                "carcass_reproduction_bonus",
                self.carcass_reproduction_bonus,
            ));
        }

        let mut events = Vec::with_capacity(2);
        if self.last_births > 0 {
            events.push(PersistenceEvent::new(
                PersistenceEventKind::Births,
                self.last_births,
            ));
        }
        if self.last_deaths > 0 {
            events.push(PersistenceEvent::new(
                PersistenceEventKind::Deaths,
                self.last_deaths,
            ));
        }

        let mut agents = Vec::with_capacity(agent_count);
        for id in self.agents.iter_handles() {
            if let Some(snapshot) = self.snapshot_agent(id) {
                agents.push(snapshot);
            }
        }

        let batch = PersistenceBatch {
            summary: summary.clone(),
            epoch: self.epoch,
            closed: self.closed,
            metrics,
            events,
            agents,
        };
        self.persistence.on_tick(&batch);
        if self.history.len() >= self.config.history_capacity {
            self.history.pop_front();
        }
        self.history.push_back(summary);
        self.last_births = 0;
        self.last_deaths = 0;
        self.carcass_health_distributed = 0.0;
        self.carcass_reproduction_bonus = 0.0;
    }

    /// Execute one simulation tick pipeline returning emitted events.
    pub fn step(&mut self) -> TickEvents {
        let next_tick = self.tick.next();
        let previous_epoch = self.epoch;

        self.stage_aging();
        let food_respawned = self.stage_food_dynamics(next_tick);
        self.stage_sense();
        self.stage_brains();
        self.stage_actuation();
        self.stage_temperature_discomfort();
        self.stage_food();
        self.stage_combat();
        self.stage_death_cleanup();
        self.stage_reproduction();
        self.stage_population(next_tick);
        self.stage_spawn_commit();
        self.stage_persistence(next_tick);

        let mut events = TickEvents {
            tick: next_tick,
            charts_flushed: self.config.chart_flush_interval > 0
                && next_tick
                    .0
                    .is_multiple_of(self.config.chart_flush_interval as u64),
            epoch_rolled: false,
            food_respawned,
        };

        self.stage_reset_events();
        self.advance_tick();
        events.tick = self.tick;
        events.epoch_rolled = self.epoch != previous_epoch;
        events
    }

    /// Returns an immutable reference to configuration.
    #[must_use]
    pub fn config(&self) -> &ScriptBotsConfig {
        &self.config
    }

    /// Mutable access to the configuration (for hot edits).
    #[must_use]
    pub fn config_mut(&mut self) -> &mut ScriptBotsConfig {
        &mut self.config
    }

    /// Replace the persistence sink.
    pub fn set_persistence(&mut self, persistence: Box<dyn WorldPersistence>) {
        self.persistence = persistence;
    }

    /// Current simulation tick.
    #[must_use]
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Current epoch counter.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Returns whether the environment is closed to random spawning.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    /// Toggle the closed-environment flag.
    pub fn set_closed(&mut self, closed: bool) {
        self.closed = closed;
    }

    /// Iterate over retained tick summaries.
    pub fn history(&self) -> impl Iterator<Item = &TickSummary> {
        self.history.iter()
    }

    /// Advances the world tick counter, rolling epochs when needed.
    pub fn advance_tick(&mut self) {
        self.tick = self.tick.next();
        if self.tick.0.is_multiple_of(10_000) {
            self.epoch += 1;
        }
    }

    /// Resets ticks and epochs (useful for restarts).
    pub fn reset_time(&mut self) {
        self.tick = Tick::zero();
        self.epoch = 0;
    }

    /// Borrow the world RNG mutably for deterministic sampling.
    #[must_use]
    pub fn rng(&mut self) -> &mut SmallRng {
        &mut self.rng
    }

    /// Read-only access to the agent arena.
    #[must_use]
    pub fn agents(&self) -> &AgentArena {
        &self.agents
    }

    /// Mutable access to the agent arena.
    #[must_use]
    pub fn agents_mut(&mut self) -> &mut AgentArena {
        &mut self.agents
    }

    /// Number of live agents.
    #[must_use]
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Spawn a new agent, returning its handle.
    pub fn spawn_agent(&mut self, agent: AgentData) -> AgentId {
        let id = self.agents.insert(agent);
        let runtime = AgentRuntime::new_random(&mut self.rng);
        self.runtime.insert(id, runtime);
        id
    }

    /// Remove an agent by handle, returning its last known data.
    pub fn remove_agent(&mut self, id: AgentId) -> Option<AgentData> {
        self.runtime.remove(id);
        self.agents.remove(id)
    }

    /// Immutable access to the food grid.
    #[must_use]
    pub fn food(&self) -> &FoodGrid {
        &self.food
    }

    /// Mutable access to the food grid.
    #[must_use]
    pub fn food_mut(&mut self) -> &mut FoodGrid {
        &mut self.food
    }

    /// Immutable access to the terrain tile layer.
    #[must_use]
    pub fn terrain(&self) -> &TerrainLayer {
        &self.terrain
    }

    /// Mutable access to the terrain layer.
    #[must_use]
    pub fn terrain_mut(&mut self) -> &mut TerrainLayer {
        &mut self.terrain
    }

    /// Immutable access to the brain registry.
    #[must_use]
    pub fn brain_registry(&self) -> &BrainRegistry {
        &self.brain_registry
    }

    /// Mutable access to the brain registry.
    #[must_use]
    pub fn brain_registry_mut(&mut self) -> &mut BrainRegistry {
        &mut self.brain_registry
    }

    /// Bind a brain from the registry to the specified agent. Returns `true` on success.
    pub fn bind_agent_brain(&mut self, id: AgentId, key: u64) -> bool {
        if !self.agents.contains(id) {
            return false;
        }
        if let Some(runtime) = self.runtime.get_mut(id)
            && let Some(binding) =
                BrainBinding::from_registry(&self.brain_registry, &mut self.rng, key)
        {
            runtime.brain = binding;
            return true;
        }
        false
    }

    /// Immutable access to per-agent runtime metadata.
    #[must_use]
    pub fn runtime(&self) -> &AgentMap<AgentRuntime> {
        &self.runtime
    }

    /// Mutable access to per-agent runtime metadata.
    #[must_use]
    pub fn runtime_mut(&mut self) -> &mut AgentMap<AgentRuntime> {
        &mut self.runtime
    }

    /// Borrow runtime data for a specific agent.
    #[must_use]
    pub fn agent_runtime(&self, id: AgentId) -> Option<&AgentRuntime> {
        self.runtime.get(id)
    }

    /// Mutably borrow runtime data for a specific agent.
    #[must_use]
    pub fn agent_runtime_mut(&mut self, id: AgentId) -> Option<&mut AgentRuntime> {
        self.runtime.get_mut(id)
    }

    /// Produce a combined snapshot of an agent's scalar columns and runtime data.
    #[must_use]
    pub fn snapshot_agent(&self, id: AgentId) -> Option<AgentState> {
        let data = self.agents.snapshot(id)?;
        let runtime = self.runtime.get(id)?.clone();
        Some(AgentState { id, data, runtime })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn sample_agent(seed: u32) -> AgentData {
        AgentData {
            position: Position::new(seed as f32, seed as f32 + 1.0),
            velocity: Velocity::new(seed as f32 * 0.1, -(seed as f32) * 0.1),
            heading: seed as f32 * 0.5,
            health: 1.0 + seed as f32,
            color: [seed as f32, seed as f32 + 0.5, seed as f32 + 1.0],
            spike_length: seed as f32 * 2.0,
            boost: seed.is_multiple_of(2),
            age: seed,
            generation: Generation(seed),
        }
    }

    #[test]
    fn insert_allocates_unique_handles() {
        let mut arena = AgentArena::new();
        let a = arena.insert(sample_agent(0));
        let b = arena.insert(sample_agent(1));
        assert_ne!(a, b);
        assert_eq!(arena.len(), 2);
        assert!(arena.contains(a));
        assert!(arena.contains(b));
    }

    #[test]
    fn remove_keeps_dense_storage_coherent() {
        let mut arena = AgentArena::new();
        let a = arena.insert(sample_agent(0));
        let b = arena.insert(sample_agent(1));
        let c = arena.insert(sample_agent(2));
        assert_eq!(arena.len(), 3);

        let removed = arena.remove(b).expect("agent removed");
        assert_eq!(removed.generation, Generation(1));
        assert_eq!(arena.len(), 2);
        assert!(arena.contains(a));
        assert!(arena.contains(c));
        assert!(!arena.contains(b));

        let snapshot_c = arena.snapshot(c).expect("snapshot");
        assert_eq!(snapshot_c.position, Position::new(2.0, 3.0));
        assert_eq!(arena.index_of(c), Some(1));

        let d = arena.insert(sample_agent(3));
        assert_ne!(
            b, d,
            "generational handles should not be reused immediately"
        );
    }

    #[test]
    fn food_grid_accessors() {
        let mut grid = FoodGrid::new(4, 2, 0.5).expect("grid");
        assert_eq!(grid.width(), 4);
        assert_eq!(grid.height(), 2);
        assert_eq!(grid.get(1, 1), Some(0.5));
        *grid.get_mut(2, 0).expect("cell") = 3.0;
        assert_eq!(grid.get(2, 0), Some(3.0));
        assert!(grid.get(5, 0).is_none());
        grid.fill(2.0);
        assert!(
            grid.cells()
                .iter()
                .all(|&cell| (cell - 2.0).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn default_config_constructs_world() {
        let config = ScriptBotsConfig::default();
        WorldState::new(config).expect("default config should be valid");
    }

    #[test]
    fn default_config_matches_legacy_food_settings() {
        let config = ScriptBotsConfig::default();
        assert!(
            (config.food_intake_rate - 0.002).abs() < f32::EPSILON,
            "expected default food_intake_rate to mirror legacy FOODINTAKE (0.002)"
        );
        assert!(
            (config.food_waste_rate - 0.001).abs() < f32::EPSILON,
            "expected default food_waste_rate to mirror legacy FOODWASTE (0.001)"
        );
        assert!(
            (config.food_transfer_rate - 0.001).abs() < f32::EPSILON,
            "expected default food_transfer_rate to mirror legacy FOODTRANSFER (0.001)"
        );
        assert!(
            (config.food_sharing_distance - 50.0).abs() < f32::EPSILON,
            "expected default food_sharing_distance to mirror legacy FOOD_SHARING_DISTANCE (50)"
        );
        assert!(
            (config.reproduction_energy_threshold - 0.65).abs() < f32::EPSILON,
            "expected default reproduction_energy_threshold to mirror legacy health gate (0.65)"
        );
        assert!(
            config.reproduction_energy_cost <= config.reproduction_energy_threshold,
            "reproduction_energy_cost should never exceed reproduction_energy_threshold"
        );
    }

    #[test]
    fn config_validation_rejects_excessive_food_waste() {
        let mut config = ScriptBotsConfig::default();
        config.food_waste_rate = config.food_max + 0.1;
        let error = WorldState::new(config).unwrap_err();
        if let WorldStateError::InvalidConfig(message) = error {
            assert!(
                message.contains("food_waste_rate"),
                "expected food_waste_rate validation error, got {message}"
            );
        } else {
            panic!("expected invalid config error, got {error:?}");
        }
    }

    #[test]
    fn world_state_initialises_from_config() {
        let config = ScriptBotsConfig {
            initial_food: 0.25,
            rng_seed: Some(42),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config.clone()).expect("world");
        assert_eq!(world.agent_count(), 0);
        assert_eq!(world.food().width(), 100);
        assert_eq!(world.food().height(), 100);
        assert_eq!(world.food().get(0, 0), Some(0.25));
        assert_eq!(world.config().world_width, config.world_width);

        let id = world.spawn_agent(sample_agent(5));
        assert_eq!(world.agent_count(), 1);
        assert!(world.agents().contains(id));
        let runtime = world.agent_runtime(id).expect("runtime");
        assert!(runtime.mutation_log.is_empty());
        assert_eq!(runtime.sensors, [0.0; INPUT_SIZE]);
        let snapshot = world.snapshot_agent(id).expect("snapshot");
        assert_eq!(snapshot.runtime.indicator.intensity, 0.0);

        world.advance_tick();
        world.advance_tick();
        assert_eq!(world.tick(), Tick(2));

        let removed = world.remove_agent(id).expect("removed agent");
        assert_eq!(removed.generation, Generation(5));
        assert_eq!(world.agent_count(), 0);
        assert!(world.agent_runtime(id).is_none());
    }

    #[test]
    fn step_executes_pipeline() {
        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 10,
            initial_food: 0.1,
            food_respawn_interval: 1,
            food_respawn_amount: 0.4,
            food_max: 0.5,
            chart_flush_interval: 2,
            food_intake_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            food_sharing_radius: 20.0,
            food_sharing_rate: 0.0,
            reproduction_energy_threshold: 10.0,
            reproduction_energy_cost: 0.0,
            reproduction_cooldown: 10,
            reproduction_spawn_jitter: 0.0,
            reproduction_color_jitter: 0.0,
            reproduction_mutation_scale: 0.0,
            reproduction_partner_chance: 0.0,
            spike_radius: 1.0,
            spike_damage: 0.0,
            spike_energy_cost: 0.0,
            rng_seed: Some(7),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let id = world.spawn_agent(sample_agent(0));
        {
            let runtime = world.agent_runtime_mut(id).expect("runtime");
            runtime.spiked = true;
            runtime.food_delta = 1.0;
            runtime.sound_output = 0.5;
            runtime.give_intent = 0.2;
        }

        let events = world.step();
        assert_eq!(world.tick(), Tick(1));
        assert_eq!(events.tick, Tick(1));
        assert!(events.food_respawned.is_some());
        assert!(!events.charts_flushed);
        let ages = world.agents().columns().ages();
        assert_eq!(ages[0], 1);
        let runtime = world.agent_runtime(id).expect("runtime");
        assert!(!runtime.spiked);
        assert_eq!(runtime.food_delta, 0.0);
        assert_eq!(runtime.sound_output, 0.0);
        assert_eq!(runtime.give_intent, 0.0);
        assert!(runtime.sensors.iter().all(|value| value.is_finite()));

        let events_second = world.step();
        assert_eq!(world.tick(), Tick(2));
        assert!(events_second.charts_flushed);
        assert_eq!(events_second.tick, Tick(2));
        assert!(!events_second.epoch_rolled);
    }

    struct StubBrain;

    impl BrainRunner for StubBrain {
        fn kind(&self) -> &'static str {
            "stub"
        }

        fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
            let mut outputs = [0.0; OUTPUT_SIZE];
            outputs[0] = 1.0;
            outputs[3] = 0.5;
            outputs[4] = 1.0;
            if !inputs.is_empty() {
                outputs[6] = inputs[0];
            }
            outputs
        }
    }

    #[test]
    fn brain_registry_executes_registered_brain() {
        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 10,
            initial_food: 0.1,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            metabolism_drain: 0.05,
            movement_drain: 0.01,
            food_sharing_rate: 0.0,
            food_sharing_radius: 20.0,
            reproduction_energy_threshold: 10.0,
            reproduction_energy_cost: 0.0,
            reproduction_cooldown: 1_000,
            reproduction_child_energy: 0.0,
            reproduction_spawn_jitter: 0.0,
            reproduction_color_jitter: 0.0,
            reproduction_mutation_scale: 0.0,
            spike_radius: 1.0,
            spike_damage: 0.0,
            spike_energy_cost: 0.0,
            rng_seed: Some(9),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let id = world.spawn_agent(sample_agent(0));
        let key = world
            .brain_registry_mut()
            .register("stub", |_rng| Box::new(StubBrain));
        assert!(world.bind_agent_brain(id, key));

        let events = world.step();
        assert_eq!(events.tick, Tick(1));
        let runtime = world.agent_runtime(id).expect("runtime");
        assert!((runtime.outputs[0] - 1.0).abs() < f32::EPSILON);
        let position = world.agents().columns().positions()[0];
        assert!(position.x != 0.0 || position.y != 0.0);
        assert!(runtime.energy < 1.0);
    }

    fn run_seeded_history(
        mut config: ScriptBotsConfig,
        steps: usize,
    ) -> (Vec<TickSummary>, Vec<f32>) {
        assert!(steps > 0, "steps must be greater than zero");
        config.history_capacity = steps;
        config.persistence_interval = 1;
        let mut world = WorldState::new(config).expect("world");
        for seed in 0..6 {
            world.spawn_agent(sample_agent(seed));
        }
        for _ in 0..steps {
            world.step();
        }
        let history: Vec<_> = world.history().cloned().collect();
        let food: Vec<f32> = world.food().cells().to_vec();
        (history, food)
    }

    #[test]
    fn seeded_runs_are_deterministic() {
        const STEPS: usize = 48;
        let base_config = ScriptBotsConfig {
            world_width: 160,
            world_height: 160,
            food_cell_size: 20,
            initial_food: 0.25,
            food_respawn_interval: 2,
            food_respawn_amount: 0.3,
            food_max: 1.5,
            chart_flush_interval: 12,
            rng_seed: Some(0xDEADBEEF),
            ..ScriptBotsConfig::default()
        };

        let (history_a, food_a) = run_seeded_history(base_config.clone(), STEPS);
        let (history_b, food_b) = run_seeded_history(base_config.clone(), STEPS);
        assert_eq!(
            history_a, history_b,
            "identical seeds should produce identical histories"
        );
        assert_eq!(
            food_a, food_b,
            "identical seeds should produce identical food distributions"
        );

        let mut different_seed = base_config;
        different_seed.rng_seed = Some(0xF00DF00D);
        let (history_c, food_c) = run_seeded_history(different_seed, STEPS);
        assert!(
            history_a != history_c || food_a != food_c,
            "different seeds should produce different histories or food distributions"
        );
    }

    #[test]
    fn combat_skips_herbivores() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            initial_food: 0.2,
            food_max: 1.0,
            spike_radius: 40.0,
            spike_damage: 0.4,
            spike_energy_cost: 0.0,
            food_intake_rate: 0.0,
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("world");
        let attacker = world.spawn_agent(sample_agent(0));
        let victim = world.spawn_agent(sample_agent(1));
        let attacker_idx = world.agents().index_of(attacker).unwrap();
        let victim_idx = world.agents().index_of(victim).unwrap();
        {
            let columns = world.agents_mut().columns_mut();
            columns.positions_mut()[attacker_idx] = Position::new(10.0, 10.0);
            columns.positions_mut()[victim_idx] = Position::new(12.0, 10.0);
            columns.spike_lengths_mut()[attacker_idx] = 1.0;
            columns.health_mut()[victim_idx] = 1.2;
        }
        if let Some(runtime) = world.agent_runtime_mut(attacker) {
            runtime.herbivore_tendency = 0.9;
            runtime.spiked = true;
            runtime.outputs = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        }

        world.stage_combat();

        let columns = world.agents().columns();
        let victim_health = columns.health()[victim_idx];
        assert!((victim_health - 1.2).abs() < 1e-6);
        let victim_runtime = world.agent_runtime(victim).unwrap();
        assert!(!victim_runtime.combat.spike_victim);
        let attacker_runtime = world.agent_runtime(attacker).unwrap();
        assert!(!attacker_runtime.combat.spike_attacker);
    }

    #[test]
    fn combat_applies_damage_and_marks_events() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            initial_food: 0.2,
            food_max: 1.0,
            spike_radius: 50.0,
            spike_damage: 0.6,
            spike_energy_cost: 0.0,
            food_intake_rate: 0.0,
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("world");
        let attacker = world.spawn_agent(sample_agent(0));
        let victim = world.spawn_agent(sample_agent(1));
        let attacker_idx = world.agents().index_of(attacker).unwrap();
        let victim_idx = world.agents().index_of(victim).unwrap();
        {
            let columns = world.agents_mut().columns_mut();
            columns.positions_mut()[attacker_idx] = Position::new(10.0, 10.0);
            columns.positions_mut()[victim_idx] = Position::new(12.0, 10.0);
            columns.spike_lengths_mut()[attacker_idx] = 1.5;
            columns.velocities_mut()[attacker_idx] = Velocity::new(0.4, 0.0);
            columns.health_mut()[victim_idx] = 1.6;
        }
        if let Some(runtime) = world.agent_runtime_mut(attacker) {
            runtime.herbivore_tendency = 0.1;
            runtime.spiked = true;
            runtime.outputs = [1.0, 0.8, 0.0, 1.0, 0.0, 1.0, 0.0, 0.2, 0.0];
        }
        if let Some(runtime) = world.agent_runtime_mut(victim) {
            runtime.herbivore_tendency = 0.2;
        }

        world.stage_combat();

        let columns = world.agents().columns();
        let victim_health = columns.health()[victim_idx];
        assert!(victim_health < 1.6);
        let victim_runtime = world.agent_runtime(victim).unwrap();
        assert!(victim_runtime.spiked);
        assert!(victim_runtime.indicator.intensity > 0.0);
        assert!(victim_runtime.combat.was_spiked_by_carnivore);
        assert!(!victim_runtime.combat.was_spiked_by_herbivore);
        let attacker_runtime = world.agent_runtime(attacker).unwrap();
        assert!(attacker_runtime.indicator.intensity > 0.0);
        assert!(attacker_runtime.combat.spike_attacker);
        assert!(attacker_runtime.combat.hit_carnivore);
        assert!(!attacker_runtime.combat.hit_herbivore);
    }

    #[test]
    fn brain_genome_validation_passes() {
        let layers = vec![
            LayerSpec::dense(INPUT_SIZE, 32, ActivationKind::Relu),
            LayerSpec::dense(32, OUTPUT_SIZE, ActivationKind::Sigmoid),
        ];
        let genome = BrainGenome::new(
            BrainFamily::Mlp,
            INPUT_SIZE,
            OUTPUT_SIZE,
            layers,
            MutationRates::default(),
            GenomeHyperParams::default(),
            GenomeProvenance::default(),
        )
        .expect("genome valid");
        assert_eq!(genome.version, GENOME_FORMAT_VERSION);
        assert!(genome.validate().is_ok());
        assert!(!genome.is_descendant());
    }

    #[test]
    fn brain_genome_validation_detects_errors() {
        let layers = vec![
            LayerSpec::dense(INPUT_SIZE, 16, ActivationKind::Relu),
            LayerSpec::dense(16, OUTPUT_SIZE, ActivationKind::Sigmoid),
        ];
        let mut genome = BrainGenome::new(
            BrainFamily::Mlp,
            INPUT_SIZE,
            OUTPUT_SIZE,
            layers.clone(),
            MutationRates::default(),
            GenomeHyperParams::default(),
            GenomeProvenance::default(),
        )
        .expect("base genome valid");

        genome.layers[0].dropout = 1.2;
        assert_eq!(
            genome.validate(),
            Err(GenomeError::InvalidDropout {
                index: 0,
                dropout: 1.2
            })
        );

        genome.layers[0].dropout = 0.0;
        genome.layers[1].inputs = OUTPUT_SIZE + 1;
        assert_eq!(
            genome.validate(),
            Err(GenomeError::MismatchedTopology {
                index: 1,
                expected: 16,
                actual: OUTPUT_SIZE + 1
            })
        );

        genome.layers[1].inputs = 16;
        genome.layers[1].outputs = OUTPUT_SIZE + 2;
        assert_eq!(
            genome.validate(),
            Err(GenomeError::OutputMismatch {
                expected: OUTPUT_SIZE,
                actual: OUTPUT_SIZE + 2
            })
        );
        genome.layers = layers;
        assert!(genome.validate().is_ok());
    }

    #[derive(Clone, Default)]
    struct SpyPersistence {
        logs: Arc<Mutex<Vec<PersistenceBatch>>>,
    }

    impl WorldPersistence for SpyPersistence {
        fn on_tick(&mut self, payload: &PersistenceBatch) {
            self.logs.lock().unwrap().push(payload.clone());
        }
    }

    #[test]
    fn persistence_receives_tick_batch() {
        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            food_sharing_rate: 0.0,
            food_sharing_radius: 20.0,
            reproduction_energy_threshold: 10.0,
            reproduction_energy_cost: 0.0,
            reproduction_cooldown: 10,
            reproduction_child_energy: 0.0,
            reproduction_spawn_jitter: 0.0,
            reproduction_color_jitter: 0.0,
            reproduction_mutation_scale: 0.0,
            spike_radius: 1.0,
            spike_damage: 0.0,
            spike_energy_cost: 0.0,
            persistence_interval: 1,
            history_capacity: 4,
            rng_seed: Some(123),
            ..ScriptBotsConfig::default()
        };

        let spy = SpyPersistence::default();
        let logs = spy.logs.clone();
        let mut world = WorldState::with_persistence(config, Box::new(spy)).expect("world");
        let id = world.spawn_agent(sample_agent(0));
        world.agent_runtime_mut(id).unwrap().energy = 1.0;

        world.step();

        let entries = logs.lock().unwrap();
        assert_eq!(entries.len(), 1);
        let batch = &entries[0];
        let summary = &batch.summary;
        assert_eq!(summary.tick, Tick(1));
        assert_eq!(summary.agent_count, 1);
        assert_eq!(summary.births, 0);
        assert_eq!(summary.deaths, 0);
        assert!((summary.average_energy - 1.0).abs() < 1e-6);

        let history: Vec<_> = world.history().cloned().collect();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].tick, Tick(1));
    }

    #[test]
    fn reproduction_spawns_child() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            food_sharing_rate: 0.0,
            food_sharing_radius: 20.0,
            reproduction_energy_threshold: 0.4,
            reproduction_energy_cost: 0.1,
            reproduction_cooldown: 1,
            reproduction_child_energy: 0.6,
            reproduction_spawn_jitter: 0.0,
            reproduction_color_jitter: 0.0,
            reproduction_mutation_scale: 0.0,
            reproduction_partner_chance: 0.0,
            reproduction_spawn_back_distance: 12.0,
            reproduction_meta_mutation_chance: 0.0,
            reproduction_meta_mutation_scale: 0.0,
            spike_radius: 1.0,
            spike_damage: 0.0,
            spike_energy_cost: 0.0,
            persistence_interval: 1,
            history_capacity: 8,
            chart_flush_interval: 0,
            rng_seed: Some(11),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let parent_id = world.spawn_agent(sample_agent(0));
        {
            let runtime = world.agent_runtime_mut(parent_id).expect("runtime");
            runtime.energy = 1.0;
            runtime.reproduction_counter = 1.0;
        }

        assert_eq!(world.agent_count(), 1);
        world.step();
        assert_eq!(world.agent_count(), 2);

        let handles: Vec<_> = world.agents().iter_handles().collect();
        let child_id = handles
            .into_iter()
            .find(|id| *id != parent_id)
            .expect("child");
        let child_state = world.snapshot_agent(child_id).expect("child state");
        assert_eq!(child_state.data.generation, Generation(1));
        assert!((child_state.runtime.energy - 0.6).abs() < 1e-6);
        assert!(
            world
                .agent_runtime(parent_id)
                .expect("parent runtime")
                .energy
                < 1.0
        );
    }

    #[test]
    fn hybrid_reproduction_blends_traits() {
        let config = ScriptBotsConfig {
            world_width: 320,
            world_height: 320,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            reproduction_energy_threshold: 0.3,
            reproduction_energy_cost: 0.1,
            reproduction_cooldown: 1,
            reproduction_child_energy: 0.5,
            reproduction_spawn_jitter: 4.0,
            reproduction_partner_chance: 1.0,
            reproduction_meta_mutation_chance: 0.0,
            reproduction_meta_mutation_scale: 0.0,
            reproduction_gene_log_capacity: 6,
            rng_seed: Some(2025),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let parent = world.spawn_agent(sample_agent(0));
        let partner = world.spawn_agent(sample_agent(1));

        {
            let arena = world.agents_mut();
            let idx_parent = arena.index_of(parent).unwrap();
            let idx_partner = arena.index_of(partner).unwrap();
            let columns = arena.columns_mut();
            columns.ages_mut()[idx_parent] = 3;
            columns.ages_mut()[idx_partner] = 40;
        }

        world.agent_runtime_mut(parent).unwrap().energy = 1.0;
        world.agent_runtime_mut(partner).unwrap().energy = 0.2;

        world.step();

        let child_id = world
            .agents()
            .iter_handles()
            .find(|id| *id != parent && *id != partner)
            .expect("child spawned");
        let child_runtime = world.agent_runtime(child_id).expect("child runtime");
        assert!(child_runtime.hybrid, "child should be marked hybrid");
        assert_eq!(child_runtime.lineage[0], Some(parent));
        assert_eq!(child_runtime.lineage[1], Some(partner));
        assert!(
            !child_runtime.mutation_log.is_empty(),
            "expected gene log entries for hybrid child"
        );
    }

    #[test]
    fn child_spawns_behind_parent() {
        #[derive(Clone)]
        struct IdleBrain;

        impl BrainRunner for IdleBrain {
            fn kind(&self) -> &'static str {
                "test.idle"
            }

            fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
                [0.0; OUTPUT_SIZE]
            }
        }

        let config = ScriptBotsConfig {
            world_width: 240,
            world_height: 240,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            reproduction_energy_threshold: 0.3,
            reproduction_energy_cost: 0.1,
            reproduction_cooldown: 1,
            reproduction_child_energy: 0.5,
            reproduction_spawn_jitter: 0.0,
            reproduction_spawn_back_distance: 18.0,
            reproduction_partner_chance: 0.0,
            reproduction_meta_mutation_chance: 0.0,
            reproduction_meta_mutation_scale: 0.0,
            rng_seed: Some(77),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let parent = world.spawn_agent(sample_agent(0));
        world.agent_runtime_mut(parent).unwrap().energy = 1.0;

        {
            let arena = world.agents_mut();
            let idx_parent = arena.index_of(parent).unwrap();
            let columns = arena.columns_mut();
            columns.positions_mut()[idx_parent] = Position::new(80.0, 120.0);
            columns.headings_mut()[idx_parent] = 0.0;
        }

        let idle_key = world
            .brain_registry_mut()
            .register("test.idle", |_rng| Box::new(IdleBrain));
        assert!(world.bind_agent_brain(parent, idle_key));

        world.step();

        let child_id = world
            .agents()
            .iter_handles()
            .find(|id| *id != parent)
            .expect("child spawned");
        let parent_state = world.snapshot_agent(parent).expect("parent state");
        let child_state = world.snapshot_agent(child_id).expect("child state");

        let dx = toroidal_delta(
            child_state.data.position.x,
            parent_state.data.position.x,
            world.config().world_width as f32,
        );
        let dy = toroidal_delta(
            child_state.data.position.y,
            parent_state.data.position.y,
            world.config().world_height as f32,
        );
        assert!(dx < -12.0, "child should spawn behind the parent along x");
        assert!(dy.abs() < 6.0, "child jitter keeps y near parent");
        let child_runtime = world.agent_runtime(child_id).expect("child runtime");
        assert!(
            !child_runtime.hybrid,
            "child should not be hybrid without partner"
        );
    }

    #[test]
    fn temperature_discomfort_drains_health_and_energy() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.5,
            temperature_comfort_band: 0.0,
            temperature_gradient_exponent: 1.0,
            temperature_discomfort_exponent: 2.0,
            rng_seed: Some(99),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let agent = world.spawn_agent(sample_agent(0));

        {
            let arena = world.agents_mut();
            let idx = arena.index_of(agent).unwrap();
            let columns = arena.columns_mut();
            columns.positions_mut()[idx] = Position::new(0.0, 10.0);
            columns.health_mut()[idx] = 1.0;
        }
        {
            let runtime = world.agent_runtime_mut(agent).unwrap();
            runtime.temperature_preference = 0.0;
            runtime.energy = 1.0;
            runtime.food_delta = 0.0;
        }

        world.stage_temperature_discomfort();

        {
            let arena = world.agents();
            let idx = arena.index_of(agent).unwrap();
            let health = arena.columns().health()[idx];
            assert!(
                (health - 0.5).abs() < 1e-6,
                "expected health to drop by 0.5, got {health}"
            );
        }
        let runtime = world.agent_runtime(agent).unwrap();
        assert!(
            (runtime.energy - 0.5).abs() < 1e-6,
            "expected energy to mirror temperature drain"
        );

        {
            let arena = world.agents_mut();
            let idx = arena.index_of(agent).unwrap();
            let columns = arena.columns_mut();
            columns.positions_mut()[idx] = Position::new(100.0, 10.0);
            columns.health_mut()[idx] = 1.0;
        }
        {
            let runtime = world.agent_runtime_mut(agent).unwrap();
            runtime.temperature_preference = 0.0;
            runtime.energy = 1.0;
            runtime.food_delta = 0.0;
        }

        world.stage_temperature_discomfort();

        {
            let arena = world.agents();
            let idx = arena.index_of(agent).unwrap();
            let health = arena.columns().health()[idx];
            assert!(
                (health - 1.0).abs() < 1e-6,
                "expected no health drain at preferred equator temperature"
            );
        }
        let runtime = world.agent_runtime(agent).unwrap();
        assert!(
            (runtime.energy - 1.0).abs() < 1e-6,
            "expected energy to remain unchanged when discomfort is zero"
        );
    }

    #[test]
    fn temperature_gradient_shapes_sensor_values() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            temperature_gradient_exponent: 2.0,
            rng_seed: Some(7),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let agent = world.spawn_agent(sample_agent(0));

        {
            let arena = world.agents_mut();
            let idx = arena.index_of(agent).unwrap();
            let columns = arena.columns_mut();
            columns.positions_mut()[idx] = Position::new(50.0, 20.0);
        }
        world
            .agent_runtime_mut(agent)
            .unwrap()
            .temperature_preference = 0.0;

        world.stage_sense();
        let runtime = world.agent_runtime(agent).unwrap();
        assert!(
            (runtime.sensors[20] - 0.25).abs() < 1e-6,
            "expected gradient-shaped discomfort of 0.25, got {}",
            runtime.sensors[20]
        );

        {
            let arena = world.agents_mut();
            let idx = arena.index_of(agent).unwrap();
            let columns = arena.columns_mut();
            columns.positions_mut()[idx] = Position::new(100.0, 20.0);
        }

        world.stage_sense();
        let runtime = world.agent_runtime(agent).unwrap();
        assert!(
            runtime.sensors[20] < 1e-6,
            "expected zero discomfort at equator, got {}",
            runtime.sensors[20]
        );
    }

    #[test]
    fn carcass_distribution_rewards_neighbors() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            carcass_distribution_radius: 50.0,
            carcass_health_reward: 4.0,
            carcass_reproduction_reward: 2.0,
            carcass_neighbor_exponent: 1.0,
            carcass_maturity_age: 5,
            carcass_energy_share_rate: 1.0,
            carcass_indicator_scale: 10.0,
            rng_seed: Some(314),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let victim = world.spawn_agent(sample_agent(0));
        let neighbor = world.spawn_agent(sample_agent(1));

        {
            let arena = world.agents_mut();
            let idx_victim = arena.index_of(victim).unwrap();
            let idx_neighbor = arena.index_of(neighbor).unwrap();
            let columns = arena.columns_mut();
            columns.positions_mut()[idx_victim] = Position::new(10.0, 10.0);
            columns.positions_mut()[idx_neighbor] = Position::new(12.0, 10.0);
            columns.ages_mut()[idx_victim] = 10;
            columns.health_mut()[idx_victim] = 0.0;
            columns.health_mut()[idx_neighbor] = 0.5;
        }
        {
            let runtime_victim = world.agent_runtime_mut(victim).unwrap();
            runtime_victim.spiked = true;
            runtime_victim.energy = 0.0;
        }
        {
            let runtime_neighbor = world.agent_runtime_mut(neighbor).unwrap();
            runtime_neighbor.herbivore_tendency = 0.0;
            runtime_neighbor.energy = 0.5;
            runtime_neighbor.reproduction_counter = 3.0;
            runtime_neighbor.indicator = IndicatorState::default();
        }

        world.pending_deaths.push(victim);
        world.stage_death_cleanup();

        assert!(
            !world.agents().contains(victim),
            "victim should be removed after cleanup"
        );
        let idx_neighbor = world.agents().index_of(neighbor).unwrap();
        let columns = world.agents().columns();
        assert!(
            (columns.health()[idx_neighbor] - 2.0).abs() < 1e-6,
            "neighbor health should clamp to 2 after reward"
        );
        let runtime_neighbor = world.agent_runtime(neighbor).unwrap();
        assert!(
            (runtime_neighbor.energy - 2.0).abs() < 1e-6,
            "neighbor energy should increase and clamp to 2"
        );
        assert!(
            (runtime_neighbor.reproduction_counter - 5.0).abs() < 1e-6,
            "reproduction counter should increase by reward"
        );
        assert!(
            runtime_neighbor.indicator.intensity > 0.0,
            "indicator should pulse after feasting"
        );
        assert!(
            (world.carcass_health_distributed - 4.0).abs() < 1e-6,
            "carcass health totals should track distributed amount"
        );
        assert!(
            (world.carcass_reproduction_bonus - 2.0).abs() < 1e-6,
            "carcass reproduction totals should track distributed amount"
        );
    }

    #[test]
    fn carcass_rewards_emit_metrics() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            carcass_distribution_radius: 40.0,
            carcass_health_reward: 3.0,
            carcass_reproduction_reward: 1.5,
            carcass_neighbor_exponent: 1.0,
            carcass_maturity_age: 5,
            carcass_energy_share_rate: 0.5,
            carcass_indicator_scale: 5.0,
            persistence_interval: 1,
            rng_seed: Some(99),
            ..ScriptBotsConfig::default()
        };

        let spy = SpyPersistence::default();
        let logs = spy.logs.clone();
        let mut world = WorldState::with_persistence(config, Box::new(spy)).expect("world");
        let victim = world.spawn_agent(sample_agent(0));
        let neighbor = world.spawn_agent(sample_agent(1));

        {
            let arena = world.agents_mut();
            let idx_victim = arena.index_of(victim).unwrap();
            let idx_neighbor = arena.index_of(neighbor).unwrap();
            let columns = arena.columns_mut();
            columns.positions_mut()[idx_victim] = Position::new(20.0, 20.0);
            columns.positions_mut()[idx_neighbor] = Position::new(25.0, 20.0);
            columns.ages_mut()[idx_victim] = 8;
            columns.health_mut()[idx_victim] = 0.0;
            columns.health_mut()[idx_neighbor] = 1.0;
        }
        {
            let runtime_victim = world.agent_runtime_mut(victim).unwrap();
            runtime_victim.spiked = true;
            runtime_victim.energy = 0.0;
        }
        {
            let runtime_neighbor = world.agent_runtime_mut(neighbor).unwrap();
            runtime_neighbor.herbivore_tendency = 0.0;
            runtime_neighbor.energy = 1.0;
            runtime_neighbor.reproduction_counter = 2.0;
        }

        world.pending_deaths.push(victim);
        world.stage_death_cleanup();
        world.stage_persistence(Tick(1));

        let entries = logs.lock().unwrap();
        assert_eq!(entries.len(), 1);
        let metrics = &entries[0].metrics;
        let mut found_health = false;
        let mut found_repro = false;
        for metric in metrics {
            match metric.name.as_ref() {
                "carcass_health_distributed" => {
                    found_health = true;
                    assert!(metric.value > 0.0);
                }
                "carcass_reproduction_bonus" => {
                    found_repro = true;
                    assert!(metric.value > 0.0);
                }
                _ => {}
            }
        }
        assert!(found_health, "expected carcass health metric");
        assert!(found_repro, "expected carcass reproduction metric");
        assert!(
            world.carcass_health_distributed.abs() < 1e-6,
            "carcass totals should reset after persistence"
        );
        assert!(
            world.carcass_reproduction_bonus.abs() < 1e-6,
            "carcass reproduction totals should reset after persistence"
        );
    }

    #[test]
    fn herbivores_gain_energy_from_ground_food() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            rng_seed: Some(11),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let agent = world.spawn_agent(sample_agent(0));

        {
            let arena = world.agents_mut();
            let idx = arena.index_of(agent).unwrap();
            let columns = arena.columns_mut();
            columns.positions_mut()[idx] = Position::new(5.0, 5.0);
        }
        {
            let runtime = world.agent_runtime_mut(agent).unwrap();
            runtime.energy = 0.5;
            runtime.reproduction_counter = 0.0;
            runtime.herbivore_tendency = 1.0;
            runtime.outputs[0] = 0.0;
            runtime.outputs[1] = 0.0;
        }
        if let Some(cell) = world.food_mut().get_mut(0, 0) {
            *cell = 0.2;
        }

        world.stage_food();

        let runtime = world.agent_runtime(agent).unwrap();
        let config = world.config();
        let expected_speed_scale = ((1.0_f32 - 0.0_f32).clamp(0.0, 1.0) * 0.7) + 0.3;
        let expected_intake = config.food_intake_rate * expected_speed_scale;
        assert!(
            (runtime.energy - (0.5 + expected_intake)).abs() < 1e-6,
            "expected herbivore energy gain of {expected_intake:.6}, got {}",
            runtime.energy - 0.5
        );
        assert!(
            (runtime.food_delta - expected_intake).abs() < 1e-6,
            "expected food_delta to match intake ({expected_intake:.6}), got {}",
            runtime.food_delta
        );
        assert!(
            (runtime.reproduction_counter - expected_intake * config.reproduction_food_bonus).abs()
                < 1e-6,
            "expected reproduction counter bonus of {:.6}, got {}",
            expected_intake * config.reproduction_food_bonus,
            runtime.reproduction_counter
        );
        let cell_value = world.food().get(0, 0).unwrap();
        let expected_cell = (0.2 - config.food_waste_rate).max(0.0);
        assert!(
            (cell_value - expected_cell).abs() < 1e-6,
            "expected cell value {:.6}, got {:.6}",
            expected_cell,
            cell_value
        );
    }

    #[test]
    fn carnivores_only_waste_ground_food() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            rng_seed: Some(42),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let agent = world.spawn_agent(sample_agent(1));

        {
            let arena = world.agents_mut();
            let idx = arena.index_of(agent).unwrap();
            let columns = arena.columns_mut();
            columns.positions_mut()[idx] = Position::new(15.0, 5.0);
        }
        {
            let runtime = world.agent_runtime_mut(agent).unwrap();
            runtime.energy = 0.5;
            runtime.reproduction_counter = 1.0;
            runtime.herbivore_tendency = 0.0;
            runtime.outputs[0] = 0.0;
            runtime.outputs[1] = 0.0;
        }
        if let Some(cell) = world.food_mut().get_mut(1, 0) {
            *cell = 0.15;
        }

        world.stage_food();

        let runtime = world.agent_runtime(agent).unwrap();
        assert!(
            (runtime.energy - 0.5).abs() < 1e-6,
            "carnivore energy should remain unchanged when grazing ground food"
        );
        assert!(
            (runtime.food_delta).abs() < 1e-6,
            "carnivore food_delta should remain zero when not gaining intake"
        );
        assert!(
            (runtime.reproduction_counter - 1.0).abs() < 1e-6,
            "carnivore reproduction counter should remain unchanged by ground food waste"
        );
        let cell_value = world.food().get(1, 0).unwrap();
        let expected_cell = (0.15 - world.config().food_waste_rate).max(0.0);
        assert!(
            (cell_value - expected_cell).abs() < 1e-6,
            "expected ground food to waste down to {:.6}, got {:.6}",
            expected_cell,
            cell_value
        );
    }

    #[test]
    fn food_sharing_uses_constant_transfer_rate() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            food_transfer_rate: 0.01,
            food_sharing_distance: 25.0,
            rng_seed: Some(202),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let giver = world.spawn_agent(sample_agent(0));
        let receiver = world.spawn_agent(sample_agent(1));

        {
            let arena = world.agents_mut();
            let idx_giver = arena.index_of(giver).unwrap();
            let idx_receiver = arena.index_of(receiver).unwrap();
            let columns = arena.columns_mut();
            columns.positions_mut()[idx_giver] = Position::new(10.0, 10.0);
            columns.positions_mut()[idx_receiver] = Position::new(12.0, 10.0);
        }
        {
            let runtime_giver = world.agent_runtime_mut(giver).unwrap();
            runtime_giver.energy = 1.0;
            runtime_giver.food_delta = 0.0;
            runtime_giver.give_intent = 1.0;
        }
        {
            let runtime_receiver = world.agent_runtime_mut(receiver).unwrap();
            runtime_receiver.energy = 0.5;
            runtime_receiver.food_delta = 0.0;
            runtime_receiver.give_intent = 0.0;
        }

        world.stage_food();

        let giver_runtime = world.agent_runtime(giver).unwrap();
        let receiver_runtime = world.agent_runtime(receiver).unwrap();
        assert!(
            (giver_runtime.energy - 0.99).abs() < 1e-6,
            "giver energy should decrease by transfer rate"
        );
        assert!(
            (receiver_runtime.energy - 0.51).abs() < 1e-6,
            "receiver energy should increase by transfer rate"
        );
        assert!(
            (giver_runtime.food_delta + 0.01).abs() < 1e-6,
            "giver food delta should reflect donation"
        );
        assert!(
            (receiver_runtime.food_delta - 0.01).abs() < 1e-6,
            "receiver food delta should reflect intake"
        );
        assert!(
            giver_runtime.indicator.intensity > 0.0,
            "giver indicator should pulse when sharing"
        );
        assert!(
            receiver_runtime.indicator.intensity > 0.0,
            "receiver indicator should pulse when sharing"
        );
        assert!(
            giver_runtime.give_intent > 0.5,
            "give intent should persist for downstream consumers"
        );
    }

    #[test]
    fn population_seeding_fills_minimum_when_open() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            population_minimum: 3,
            population_spawn_interval: 0,
            rng_seed: Some(111),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        world.step();
        assert!(
            world.agent_count() >= 3,
            "expected minimum population seeding"
        );
    }

    #[test]
    fn population_seeding_respects_closed_flag() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            population_minimum: 3,
            population_spawn_interval: 10,
            rng_seed: Some(222),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        world.set_closed(true);
        world.step();
        assert_eq!(
            world.agent_count(),
            0,
            "closed world should not seed agents"
        );
    }

    #[test]
    fn population_interval_spawns_agents() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 2,
            population_spawn_count: 1,
            population_crossover_chance: 0.0,
            rng_seed: Some(333),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        world.step();
        assert_eq!(world.agent_count(), 0, "no spawn on first step");
        world.step();
        assert_eq!(world.agent_count(), 1, "expected spawn on interval");
    }

    #[test]
    fn metabolism_ramp_increases_drain() {
        let config = ScriptBotsConfig {
            world_width: 120,
            world_height: 120,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            metabolism_ramp_floor: 0.25,
            metabolism_ramp_rate: 0.5,
            metabolism_boost_penalty: 0.1,
            rng_seed: Some(11),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let agent = world.spawn_agent(sample_agent(0));
        world.agent_runtime_mut(agent).unwrap().energy = 1.0;
        {
            let arena = world.agents_mut();
            let idx = arena.index_of(agent).unwrap();
            let columns = arena.columns_mut();
            columns.health_mut()[idx] = 1.0;
        }
        {
            let runtime = world.agent_runtime_mut(agent).unwrap();
            runtime.outputs = [0.0; OUTPUT_SIZE];
            runtime.outputs[6] = 1.0; // enable boost
        }

        world.stage_actuation();

        let runtime = world.agent_runtime(agent).expect("runtime");
        let expected_drain = (1.0 - 0.25) * 0.5 + 0.1;
        assert!(
            (runtime.energy - (1.0 - expected_drain)).abs() < 1e-6,
            "expected energy {:.6}, got {:.6}",
            1.0 - expected_drain,
            runtime.energy
        );
        let arena = world.agents();
        let idx = arena.index_of(agent).unwrap();
        let health = arena.columns().health()[idx];
        assert!(
            (health - (1.0 - expected_drain)).abs() < 1e-6,
            "expected health {:.6}, got {:.6}",
            1.0 - expected_drain,
            health
        );
    }

    #[test]
    fn aging_decay_applies_after_threshold() {
        let config = ScriptBotsConfig {
            world_width: 120,
            world_height: 120,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            aging_health_decay_start: 5,
            aging_health_decay_rate: 0.02,
            aging_health_decay_max: 0.05,
            aging_energy_penalty_rate: 1.5,
            rng_seed: Some(23),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let agent = world.spawn_agent(sample_agent(0));

        {
            let arena = world.agents_mut();
            let idx = arena.index_of(agent).unwrap();
            let columns = arena.columns_mut();
            columns.ages_mut()[idx] = 5;
            columns.health_mut()[idx] = 1.0;
        }
        {
            let runtime = world.agent_runtime_mut(agent).unwrap();
            runtime.energy = 1.0;
        }

        world.stage_aging();

        let expected_penalty = 0.02; // age increments to 6 => over=1
        let expected_energy_penalty = expected_penalty * 1.5;

        {
            let arena = world.agents();
            let idx = arena.index_of(agent).unwrap();
            let ages = arena.columns().ages();
            assert_eq!(ages[idx], 6);
            let health = arena.columns().health()[idx];
            assert!(
                (health - (1.0 - expected_penalty)).abs() < 1e-6,
                "expected health {:.6}, got {:.6}",
                1.0 - expected_penalty,
                health
            );
        }
        let runtime = world.agent_runtime(agent).unwrap();
        assert!(
            (runtime.energy - (1.0 - expected_energy_penalty)).abs() < 1e-6,
            "expected energy {:.6}, got {:.6}",
            1.0 - expected_energy_penalty,
            runtime.energy
        );
    }

    #[test]
    fn death_cleanup_is_stable_and_deduplicated() {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            reproduction_energy_threshold: 0.0,
            reproduction_energy_cost: 0.0,
            reproduction_cooldown: 0,
            reproduction_child_energy: 0.0,
            rng_seed: Some(1234),
            ..ScriptBotsConfig::default()
        })
        .expect("world");

        let ids: Vec<_> = (0..4)
            .map(|seed| world.spawn_agent(sample_agent(seed)))
            .collect();

        world.pending_deaths.push(ids[1]);
        world.pending_deaths.push(ids[3]);
        world.pending_deaths.push(ids[1]);

        world.stage_death_cleanup();

        let survivors: Vec<_> = world.agents().iter_handles().collect();
        assert_eq!(survivors, vec![ids[0], ids[2]]);
        assert!(world.agent_runtime(ids[1]).is_none());
        assert!(world.agent_runtime(ids[3]).is_none());
        assert_eq!(world.agent_count(), 2);
        assert!(world.pending_deaths.is_empty());
        assert_eq!(world.last_deaths, 2);
    }
}
