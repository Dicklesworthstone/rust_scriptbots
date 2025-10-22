//! Core types shared across the ScriptBots workspace.

use ordered_float::OrderedFloat;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use scriptbots_index::{NeighborhoodIndex, UniformGridIndex};
use serde::{Deserialize, Serialize};
use slotmap::{SecondaryMap, SlotMap, new_key_type};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use thiserror::Error;

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

/// Runtime brain attachment tracking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum BrainBinding {
    /// Brain not yet attached.
    #[default]
    Unbound,
    /// Brain keyed in the world brain registry.
    Registry { key: u64 },
}

/// Thin trait object used to drive brain evaluations without coupling to concrete brain crates.
pub trait BrainRunner: Send {
    /// Static identifier of the brain implementation.
    fn kind(&self) -> &'static str;

    /// Evaluate outputs for the provided sensors.
    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE];
}

/// Registry owning brain runners keyed by opaque handles.
#[derive(Default)]
pub struct BrainRegistry {
    next_key: u64,
    runners: HashMap<u64, Box<dyn BrainRunner>>, // stored in insertion order for determinism via key
}

impl std::fmt::Debug for BrainRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrainRegistry")
            .field("next_key", &self.next_key)
            .field("runner_count", &self.runners.len())
            .finish()
    }
}

impl BrainRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new brain runner, returning its registry key.
    pub fn register(&mut self, brain: Box<dyn BrainRunner>) -> u64 {
        let key = self.next_key;
        self.next_key += 1;
        self.runners.insert(key, brain);
        key
    }

    /// Removes a brain runner from the registry.
    pub fn unregister(&mut self, key: u64) -> Option<Box<dyn BrainRunner>> {
        self.runners.remove(&key)
    }

    /// Executes the brain assigned to `key`, returning the outputs if the key exists.
    pub fn tick(&mut self, key: u64, inputs: &[f32; INPUT_SIZE]) -> Option<[f32; OUTPUT_SIZE]> {
        self.runners.get_mut(&key).map(|brain| brain.tick(inputs))
    }

    /// Returns whether a key is registered.
    #[must_use]
    pub fn contains(&self, key: u64) -> bool {
        self.runners.contains_key(&key)
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
    pub sound_multiplier: f32,
    pub give_intent: f32,
    pub sensors: [f32; INPUT_SIZE],
    pub outputs: [f32; OUTPUT_SIZE],
    pub indicator: IndicatorState,
    pub selection: SelectionState,
    pub food_delta: f32,
    pub spiked: bool,
    pub hybrid: bool,
    pub sound_output: f32,
    pub brain: BrainBinding,
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
            sound_multiplier: 1.0,
            give_intent: 0.0,
            sensors: [0.0; INPUT_SIZE],
            outputs: [0.0; OUTPUT_SIZE],
            indicator: IndicatorState::default(),
            selection: SelectionState::None,
            food_delta: 0.0,
            spiked: false,
            hybrid: false,
            sound_output: 0.0,
            brain: BrainBinding::default(),
            mutation_log: Vec::new(),
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
    spiked: bool,
}

#[derive(Debug, Default)]
struct CombatResult {
    energy: f32,
    contributions: Vec<(usize, f32)>,
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
    /// Radius used for neighborhood sensing.
    pub sense_radius: f32,
    /// Normalization factor for counting neighbors.
    pub sense_max_neighbors: f32,
    /// Baseline metabolism drain applied each tick.
    pub metabolism_drain: f32,
    /// Fraction of velocity converted to additional energy cost.
    pub movement_drain: f32,
    /// Base rate at which agents siphon food from cells.
    pub food_intake_rate: f32,
    /// Radius used for food sharing with friendly neighbors.
    pub food_sharing_radius: f32,
    /// Fraction of energy shared per neighbor when donating.
    pub food_sharing_rate: f32,
    /// Energy threshold required before reproduction can trigger.
    pub reproduction_energy_threshold: f32,
    /// Energy deducted from a parent upon reproduction.
    pub reproduction_energy_cost: f32,
    /// Cooldown in ticks between reproductions.
    pub reproduction_cooldown: u32,
    /// Starting energy assigned to a child agent.
    pub reproduction_child_energy: f32,
    /// Spatial jitter applied to child spawn positions.
    pub reproduction_spawn_jitter: f32,
    /// Color mutation range applied per channel.
    pub reproduction_color_jitter: f32,
    /// Scale factor applied to trait mutations.
    pub reproduction_mutation_scale: f32,
    /// Base radius used when checking spike impacts.
    pub spike_radius: f32,
    /// Damage applied by a spike at full power.
    pub spike_damage: f32,
    /// Energy cost of deploying a spike.
    pub spike_energy_cost: f32,
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
            initial_food: 1.0,
            rng_seed: None,
            chart_flush_interval: 1_000,
            food_respawn_interval: 15,
            food_respawn_amount: 0.5,
            food_max: 0.5,
            sense_radius: 120.0,
            sense_max_neighbors: 12.0,
            metabolism_drain: 0.002,
            movement_drain: 0.005,
            food_intake_rate: 0.05,
            food_sharing_radius: 80.0,
            food_sharing_rate: 0.1,
            reproduction_energy_threshold: 1.5,
            reproduction_energy_cost: 0.75,
            reproduction_cooldown: 300,
            reproduction_child_energy: 1.0,
            reproduction_spawn_jitter: 20.0,
            reproduction_color_jitter: 0.05,
            reproduction_mutation_scale: 0.02,
            spike_radius: 40.0,
            spike_damage: 0.25,
            spike_energy_cost: 0.02,
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
        if self.metabolism_drain < 0.0
            || self.movement_drain < 0.0
            || self.food_intake_rate < 0.0
            || self.food_sharing_radius <= 0.0
            || self.food_sharing_rate < 0.0
            || self.reproduction_energy_threshold < 0.0
            || self.reproduction_energy_cost < 0.0
            || self.reproduction_child_energy < 0.0
            || self.reproduction_spawn_jitter < 0.0
            || self.reproduction_color_jitter < 0.0
            || self.reproduction_mutation_scale < 0.0
            || self.spike_radius <= 0.0
            || self.spike_damage < 0.0
            || self.spike_energy_cost < 0.0
            || self.history_capacity == 0
        {
            return Err(WorldStateError::InvalidConfig(
                "metabolism, reproduction, sharing, and history parameters must be non-negative, radius positive",
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

/// Aggregate world state shared by the simulation and rendering layers.
pub struct WorldState {
    config: ScriptBotsConfig,
    tick: Tick,
    epoch: u64,
    closed: bool,
    rng: SmallRng,
    agents: AgentArena,
    food: FoodGrid,
    runtime: AgentMap<AgentRuntime>,
    index: UniformGridIndex,
    brain_registry: BrainRegistry,
    pending_deaths: Vec<AgentId>,
    #[allow(dead_code)]
    pending_spawns: Vec<SpawnOrder>,
    persistence: Box<dyn WorldPersistence>,
    last_births: usize,
    last_deaths: usize,
    history: VecDeque<TickSummary>,
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
        let index = UniformGridIndex::new(
            config.food_cell_size as f32,
            config.world_width as f32,
            config.world_height as f32,
        );
        let history_capacity = config.history_capacity;
        Ok(Self {
            food: FoodGrid::new(food_w, food_h, config.initial_food)?,
            config,
            tick: Tick::zero(),
            epoch: 0,
            closed: false,
            rng,
            agents: AgentArena::new(),
            runtime: AgentMap::new(),
            index,
            brain_registry: BrainRegistry::new(),
            pending_deaths: Vec::new(),
            pending_spawns: Vec::new(),
            persistence,
            last_births: 0,
            last_deaths: 0,
            history: VecDeque::with_capacity(history_capacity),
        })
    }

    fn stage_aging(&mut self) {
        for age in self.agents.columns_mut().ages_mut() {
            *age = age.saturating_add(1);
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

    fn stage_sense(&mut self) {
        let agent_count = self.agents.len();
        if agent_count == 0 {
            return;
        }

        let columns = self.agents.columns();
        let positions_slice = columns.positions();
        let health_slice = columns.health();
        let ages_slice = columns.ages();
        let position_pairs: Vec<(f32, f32)> = positions_slice.iter().map(|p| (p.x, p.y)).collect();

        if self.index.rebuild(&position_pairs).is_err() {
            return;
        }

        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        let energies: Vec<f32> = handles
            .iter()
            .map(|id| self.runtime.get(*id).map_or(0.0, |rt| rt.energy))
            .collect();

        let radius = self.config.sense_radius;
        let radius_sq = radius * radius;
        let neighbor_normalizer = self.config.sense_max_neighbors;
        let index = &self.index;

        let sensor_results: Vec<[f32; INPUT_SIZE]> = handles
            .par_iter()
            .enumerate()
            .map(|(idx, _agent_id)| {
                let mut sensors = [0.0f32; INPUT_SIZE];
                let mut nearest_sq = f32::INFINITY;
                let mut neighbor_count = 0usize;
                let mut neighbor_energy_sum = 0.0_f32;
                let mut neighbor_health_sum = 0.0_f32;

                index.neighbors_within(
                    idx,
                    radius_sq,
                    &mut |other_idx, dist_sq: OrderedFloat<f32>| {
                        neighbor_count += 1;
                        let dist_sq_val = dist_sq.into_inner();
                        if dist_sq_val < nearest_sq {
                            nearest_sq = dist_sq_val;
                        }
                        neighbor_energy_sum += energies[other_idx];
                        neighbor_health_sum += health_slice[other_idx];
                    },
                );

                let nearest_dist = if neighbor_count > 0 {
                    nearest_sq.sqrt()
                } else {
                    radius
                };

                sensors.fill(0.0);
                sensors[0] = (1.0 / (1.0 + nearest_dist)).clamp(0.0, 1.0);
                sensors[1] = (neighbor_count as f32 / neighbor_normalizer).clamp(0.0, 1.0);
                sensors[2] = (health_slice[idx] / 2.0).clamp(0.0, 1.0);
                let self_energy = energies[idx];
                sensors[3] = (self_energy / 2.0).clamp(0.0, 1.0);
                sensors[4] = (ages_slice[idx] as f32 / 1_000.0).clamp(0.0, 1.0);
                if neighbor_count > 0 {
                    sensors[5] =
                        (neighbor_energy_sum / neighbor_count as f32 / 2.0).clamp(0.0, 1.0);
                    sensors[6] =
                        (neighbor_health_sum / neighbor_count as f32 / 2.0).clamp(0.0, 1.0);
                }
                sensors
            })
            .collect();

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
                let outputs = match runtime.brain {
                    BrainBinding::Registry { key } => self
                        .brain_registry
                        .tick(key, &runtime.sensors)
                        .unwrap_or_else(|| Self::default_outputs(&runtime.sensors)),
                    BrainBinding::Unbound => Self::default_outputs(&runtime.sensors),
                };
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

    fn stage_actuation(&mut self) {
        let width = self.config.world_width as f32;
        let height = self.config.world_height as f32;
        let speed_base = self.config.sense_radius * 0.1;
        let movement_drain = self.config.movement_drain;
        let metabolism_drain = self.config.metabolism_drain;
        let handles: Vec<AgentId> = self.agents.iter_handles().collect();

        let positions_snapshot: Vec<Position> = self.agents.columns().positions().to_vec();
        let headings_snapshot: Vec<f32> = self.agents.columns().headings().to_vec();

        let runtime = &self.runtime;
        let results: Vec<ActuationResult> = handles
            .par_iter()
            .enumerate()
            .map(|(idx, agent_id)| {
                if let Some(runtime) = runtime.get(*agent_id) {
                    let outputs = runtime.outputs;
                    let forward = outputs.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0);
                    let strafe = outputs.get(1).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
                    let turn = outputs.get(2).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
                    let boost = outputs.get(3).copied().unwrap_or(0.0).clamp(0.0, 1.0);

                    let heading: f32 = headings_snapshot[idx] + turn * 0.1;
                    let cos_h = heading.cos();
                    let sin_h = heading.sin();
                    let forward_dx = cos_h * forward;
                    let forward_dy = sin_h * forward;
                    let strafe_dx = -sin_h * strafe;
                    let strafe_dy = cos_h * strafe;

                    let speed_scale = speed_base * (1.0 + boost);
                    let vx = (forward_dx + strafe_dx) * speed_scale;
                    let vy = (forward_dy + strafe_dy) * speed_scale;

                    let mut next_pos = positions_snapshot[idx];
                    next_pos.x = Self::wrap_position(next_pos.x + vx, width);
                    next_pos.y = Self::wrap_position(next_pos.y + vy, height);

                    let movement_penalty = movement_drain * (vx.abs() + vy.abs());
                    let boost_penalty = boost * movement_drain * 0.5;
                    let metabolism_penalty = metabolism_drain;
                    let drain = metabolism_penalty + movement_penalty + boost_penalty;
                    let health_delta = -drain;
                    let energy = (runtime.energy - drain).max(0.0);

                    let spike_power = outputs.get(5).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                    let spiked = spike_power > 0.5;

                    ActuationResult {
                        delta: Some(ActuationDelta {
                            heading,
                            velocity: Velocity::new(vx, vy),
                            position: next_pos,
                            health_delta,
                        }),
                        energy,
                        spiked,
                    }
                } else {
                    ActuationResult::default()
                }
            })
            .collect();

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

        for (idx, agent_id) in handles.iter().enumerate() {
            if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                runtime.energy = results[idx].energy;
                runtime.spiked = results[idx].spiked;
            }
        }
    }

    fn stage_reset_events(&mut self) {
        for runtime in self.runtime.values_mut() {
            runtime.spiked = false;
            runtime.food_delta = 0.0;
            runtime.sound_output = 0.0;
            runtime.give_intent = 0.0;
        }
    }

    fn stage_food(&mut self) {
        let intake_rate = self.config.food_intake_rate;
        if intake_rate <= 0.0 {
            return;
        }

        let cell_size = self.config.food_cell_size as f32;
        let positions = self.agents.columns().positions();
        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        let mut sharers: Vec<usize> = Vec::new();

        for (idx, agent_id) in handles.iter().enumerate() {
            if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                let pos = positions[idx];
                let cell_x = (pos.x / cell_size).floor() as u32 % self.food.width();
                let cell_y = (pos.y / cell_size).floor() as u32 % self.food.height();
                if let Some(cell) = self.food.get_mut(cell_x, cell_y) {
                    let intake = cell.min(intake_rate);
                    *cell -= intake;
                    runtime.energy = (runtime.energy + intake * 0.5).min(2.0);
                    runtime.food_delta += intake;
                }
                if runtime.outputs.get(4).copied().unwrap_or(0.0) > 0.5 {
                    sharers.push(idx);
                }
            }
        }

        if sharers.len() < 2 {
            return;
        }

        let radius_sq = self.config.food_sharing_radius * self.config.food_sharing_radius;
        let share_rate = self.config.food_sharing_rate;

        for (i, idx_a) in sharers.iter().enumerate() {
            for idx_b in sharers.iter().skip(i + 1) {
                let idx_a = *idx_a;
                let idx_b = *idx_b;
                let id_a = handles[idx_a];
                let id_b = handles[idx_b];
                let pos_a = positions[idx_a];
                let pos_b = positions[idx_b];
                let dx = pos_a.x - pos_b.x;
                let dy = pos_a.y - pos_b.y;
                if dx * dx + dy * dy <= radius_sq {
                    let energy_a = self.runtime.get(id_a).map_or(0.0, |r| r.energy);
                    let energy_b = self.runtime.get(id_b).map_or(0.0, |r| r.energy);
                    let diff = energy_a - energy_b;
                    if diff.abs() <= f32::EPSILON {
                        continue;
                    }
                    let transfer = (diff.abs() * 0.5).min(share_rate);
                    if transfer <= 0.0 {
                        continue;
                    }
                    if diff > 0.0 {
                        if let Some(runtime_a) = self.runtime.get_mut(id_a) {
                            runtime_a.energy = (runtime_a.energy - transfer).max(0.0);
                            runtime_a.food_delta -= transfer;
                        }
                        if let Some(runtime_b) = self.runtime.get_mut(id_b) {
                            runtime_b.energy = (runtime_b.energy + transfer).min(2.0);
                            runtime_b.food_delta += transfer;
                        }
                    } else {
                        if let Some(runtime_b) = self.runtime.get_mut(id_b) {
                            runtime_b.energy = (runtime_b.energy - transfer).max(0.0);
                            runtime_b.food_delta -= transfer;
                        }
                        if let Some(runtime_a) = self.runtime.get_mut(id_a) {
                            runtime_a.energy = (runtime_a.energy + transfer).min(2.0);
                            runtime_a.food_delta += transfer;
                        }
                    }
                }
            }
        }
    }

    fn stage_combat(&mut self) {
        let spike_radius = self.config.spike_radius;
        if spike_radius <= 0.0 {
            return;
        }

        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        if handles.is_empty() {
            return;
        }

        let positions = self.agents.columns().positions();
        let spike_lengths = self.agents.columns().spike_lengths();
        let positions_pairs: Vec<(f32, f32)> = positions.iter().map(|p| (p.x, p.y)).collect();
        let _ = self.index.rebuild(&positions_pairs);

        let spike_damage = self.config.spike_damage;
        let spike_energy_cost = self.config.spike_energy_cost;
        let index = &self.index;
        let runtime = &self.runtime;

        let results: Vec<CombatResult> = handles
            .par_iter()
            .enumerate()
            .map(|(idx, agent_id)| {
                if let Some(runtime) = runtime.get(*agent_id) {
                    let energy_before = runtime.energy;
                    if !runtime.spiked {
                        return CombatResult {
                            energy: energy_before,
                            contributions: Vec::new(),
                        };
                    }
                    let spike_power = runtime
                        .outputs
                        .get(5)
                        .copied()
                        .unwrap_or(0.0)
                        .clamp(0.0, 1.0);
                    if spike_power <= f32::EPSILON {
                        return CombatResult {
                            energy: energy_before,
                            contributions: Vec::new(),
                        };
                    }

                    let reach: f32 = (spike_radius + spike_lengths[idx]).max(1.0);
                    let reach_sq = reach * reach;
                    let mut contributions = Vec::new();
                    index.neighbors_within(
                        idx,
                        reach_sq,
                        &mut |other_idx, _dist_sq: OrderedFloat<f32>| {
                            contributions.push((other_idx, spike_damage * spike_power));
                        },
                    );

                    CombatResult {
                        energy: (energy_before - spike_energy_cost * spike_power).max(0.0),
                        contributions,
                    }
                } else {
                    CombatResult::default()
                }
            })
            .collect();

        let mut damage = vec![0.0f32; handles.len()];
        for (idx, agent_id) in handles.iter().enumerate() {
            if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                runtime.energy = results[idx].energy;
            }
            for &(other_idx, dmg) in &results[idx].contributions {
                if let Some(target) = damage.get_mut(other_idx) {
                    *target += dmg;
                }
            }
        }

        let columns = self.agents.columns_mut();
        let healths = columns.health_mut();
        for (idx, dmg) in damage.into_iter().enumerate() {
            if dmg <= 0.0 {
                continue;
            }
            healths[idx] = (healths[idx] - dmg).max(0.0);
            if let Some(runtime) = self.runtime.get_mut(handles[idx]) {
                runtime.food_delta -= dmg;
            }
            if healths[idx] <= 0.0 {
                self.pending_deaths.push(handles[idx]);
            }
        }
    }

    fn stage_death_cleanup(&mut self) {
        if self.pending_deaths.is_empty() {
            return;
        }
        let mut unique = HashSet::new();
        let removals: Vec<AgentId> = self.pending_deaths.drain(..).collect();
        for agent_id in removals {
            if unique.insert(agent_id) {
                self.remove_agent(agent_id);
            }
        }
        self.last_deaths = unique.len();
    }

    fn stage_reproduction(&mut self) {
        if self.config.reproduction_energy_threshold <= 0.0 {
            return;
        }

        let cooldown = self.config.reproduction_cooldown.max(1) as f32;
        let width = self.config.world_width as f32;
        let height = self.config.world_height as f32;
        let jitter = self.config.reproduction_spawn_jitter;
        let color_jitter = self.config.reproduction_color_jitter;

        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        if handles.is_empty() {
            return;
        }

        let parent_snapshots: Vec<AgentData> = {
            let columns = self.agents.columns();
            (0..columns.len())
                .map(|idx| columns.snapshot(idx))
                .collect()
        };

        for (idx, agent_id) in handles.iter().enumerate() {
            let mut parent_runtime = None;
            {
                let runtime = match self.runtime.get_mut(*agent_id) {
                    Some(rt) => rt,
                    None => continue,
                };
                runtime.reproduction_counter += 1.0;
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
                parent_runtime = Some(runtime.clone());
            }

            if let Some(parent_runtime) = parent_runtime {
                let parent_data = parent_snapshots[idx];
                let child_data =
                    self.build_child_data(&parent_data, jitter, color_jitter, width, height);
                let child_runtime = self.build_child_runtime(&parent_runtime);
                self.pending_spawns.push(SpawnOrder {
                    parent_index: idx,
                    data: child_data,
                    runtime: child_runtime,
                });
            }
        }
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

    fn build_child_data(
        &mut self,
        parent: &AgentData,
        jitter: f32,
        color_jitter: f32,
        width: f32,
        height: f32,
    ) -> AgentData {
        let mut child = *parent;
        let jitter_x = if jitter > 0.0 {
            self.rng.random_range(-jitter..jitter)
        } else {
            0.0
        };
        let jitter_y = if jitter > 0.0 {
            self.rng.random_range(-jitter..jitter)
        } else {
            0.0
        };
        child.position.x = Self::wrap_position(parent.position.x + jitter_x, width);
        child.position.y = Self::wrap_position(parent.position.y + jitter_y, height);
        child.velocity = Velocity::default();
        child.heading = self
            .rng
            .random_range(-std::f32::consts::PI..std::f32::consts::PI);
        child.health = 1.0;
        child.boost = false;
        child.age = 0;
        child.generation = parent.generation.next();
        let spike_variance = self.config.reproduction_mutation_scale;
        if spike_variance > 0.0 {
            child.spike_length = (child.spike_length
                + self.rng.random_range(-spike_variance..spike_variance))
            .clamp(0.0, (parent.spike_length + spike_variance).max(0.1));
        }
        if color_jitter > 0.0 {
            for channel in &mut child.color {
                *channel =
                    (*channel + self.rng.random_range(-color_jitter..color_jitter)).clamp(0.0, 1.0);
            }
        }
        child
    }

    fn build_child_runtime(&mut self, parent: &AgentRuntime) -> AgentRuntime {
        let mut runtime = parent.clone();
        runtime.energy = self.config.reproduction_child_energy.clamp(0.0, 2.0);
        runtime.reproduction_counter = 0.0;
        runtime.sensors = [0.0; INPUT_SIZE];
        runtime.outputs = [0.0; OUTPUT_SIZE];
        runtime.food_delta = 0.0;
        runtime.spiked = false;
        runtime.sound_output = 0.0;
        runtime.give_intent = 0.0;
        runtime.indicator = IndicatorState::default();
        runtime.selection = SelectionState::None;
        runtime.mutation_log.clear();
        runtime.brain = BrainBinding::default();

        let mutation_scale =
            runtime.mutation_rates.secondary * self.config.reproduction_mutation_scale;
        if mutation_scale > 0.0 {
            runtime.herbivore_tendency =
                self.mutate_value(runtime.herbivore_tendency, mutation_scale, 0.0, 1.0);
            runtime.trait_modifiers.smell =
                self.mutate_value(runtime.trait_modifiers.smell, mutation_scale, 0.05, 3.0);
            runtime.trait_modifiers.sound =
                self.mutate_value(runtime.trait_modifiers.sound, mutation_scale, 0.05, 3.0);
            runtime.trait_modifiers.hearing =
                self.mutate_value(runtime.trait_modifiers.hearing, mutation_scale, 0.1, 4.0);
            runtime.trait_modifiers.eye =
                self.mutate_value(runtime.trait_modifiers.eye, mutation_scale, 0.5, 4.0);
            runtime.trait_modifiers.blood =
                self.mutate_value(runtime.trait_modifiers.blood, mutation_scale, 0.5, 4.0);
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
        let metrics = vec![
            MetricSample::from_f32("total_energy", summary.total_energy),
            MetricSample::from_f32("average_energy", summary.average_energy),
            MetricSample::from_f32("average_health", summary.average_health),
        ];

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
    }

    /// Execute one simulation tick pipeline returning emitted events.
    pub fn step(&mut self) -> TickEvents {
        let next_tick = self.tick.next();
        let previous_epoch = self.epoch;

        self.stage_aging();
        self.stage_sense();
        self.stage_brains();
        self.stage_actuation();
        self.stage_food();
        self.stage_combat();
        self.stage_death_cleanup();
        self.stage_reproduction();
        self.stage_spawn_commit();
        self.stage_persistence(next_tick);

        let mut events = TickEvents {
            tick: next_tick,
            charts_flushed: self.config.chart_flush_interval > 0
                && next_tick
                    .0
                    .is_multiple_of(self.config.chart_flush_interval as u64),
            epoch_rolled: false,
            food_respawned: self.stage_food_respawn(next_tick),
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
        self.runtime.insert(id, AgentRuntime::default());
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
        assert!(runtime.sensors[0] > 0.0);

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
        let key = world.brain_registry_mut().register(Box::new(StubBrain));
        if let Some(runtime) = world.agent_runtime_mut(id) {
            runtime.brain = BrainBinding::Registry { key };
        }

        let events = world.step();
        assert_eq!(events.tick, Tick(1));
        let runtime = world.agent_runtime(id).expect("runtime");
        assert!((runtime.outputs[0] - 1.0).abs() < f32::EPSILON);
        let position = world.agents().columns().positions()[0];
        assert!(position.x != 0.0 || position.y != 0.0);
        assert!(runtime.energy < 1.0);
    }

    fn run_seeded_history(mut config: ScriptBotsConfig, steps: usize) -> Vec<TickSummary> {
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
        world.history().cloned().collect()
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

        let history_a = run_seeded_history(base_config.clone(), STEPS);
        let history_b = run_seeded_history(base_config.clone(), STEPS);
        assert_eq!(
            history_a, history_b,
            "identical seeds should produce identical histories"
        );

        let mut different_seed = base_config;
        different_seed.rng_seed = Some(0xF00DF00D);
        let history_c = run_seeded_history(different_seed, STEPS);
        assert_ne!(
            history_a, history_c,
            "different seeds should produce different histories"
        );
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
            spike_radius: 1.0,
            spike_damage: 0.0,
            spike_energy_cost: 0.0,
            persistence_interval: 0,
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
}
