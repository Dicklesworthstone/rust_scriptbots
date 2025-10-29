//! Core types shared across the ScriptBots workspace.

#[allow(unused_imports)]
use ordered_float::OrderedFloat;
use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scriptbots_index::{NeighborhoodIndex, UniformGridIndex};
use serde::{Deserialize, Serialize};
use slotmap::{Key, KeyData, SecondaryMap, SlotMap, new_key_type};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
#[cfg(feature = "parallel")]
use std::sync::OnceLock;
use thiserror::Error;
#[cfg(feature = "simd_wide")]
use wide::f32x4;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainActivations {
    pub layers: Vec<ActivationLayer>,
    #[serde(default)]
    pub connections: Vec<ActivationEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationLayer {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationEdge {
    pub from: usize,
    pub to: usize,
    pub weight: f32,
}

#[cfg(feature = "parallel")]
static RAYON_LIMIT_GUARD: OnceLock<()> = OnceLock::new();

#[cfg(feature = "parallel")]
fn par_min_split() -> usize {
    use std::sync::OnceLock;
    static PAR_MIN_SPLIT: OnceLock<usize> = OnceLock::new();
    *PAR_MIN_SPLIT.get_or_init(|| {
        std::env::var("SCRIPTBOTS_PAR_MIN_SPLIT")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1024)
    })
}

#[cfg(feature = "parallel")]
macro_rules! collect_handles {
    ($handles:expr, |$idx:ident, $handle:pat_param| $body:expr) => {{
        ($handles)
            .par_iter()
            .with_min_len(par_min_split())
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

#[cfg(feature = "parallel")]
fn configure_parallelism() {
    use std::cmp::max;

    RAYON_LIMIT_GUARD.get_or_init(|| {
        if std::env::var("RAYON_NUM_THREADS").is_ok() {
            return;
        }

        let cpu_count = max(1, num_cpus::get_physical());
        let env_limit = std::env::var("SCRIPTBOTS_MAX_THREADS")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0);

        let mut limit = env_limit.unwrap_or_else(|| default_thread_budget(cpu_count));
        if limit > cpu_count {
            limit = cpu_count;
        }
        if limit == 0 {
            limit = 1;
        }

        // SAFETY: `limit` is a finite positive integer converted to string; the standard library
        // marks `set_var` as unsafe on nightly, but providing well-formed Unicode strings is safe.
        unsafe {
            std::env::set_var("RAYON_NUM_THREADS", limit.to_string());
        }
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(limit)
            .build_global();
    });
}

#[cfg(feature = "parallel")]
fn default_thread_budget(cpu_count: usize) -> usize {
    match cpu_count {
        0..=2 => 1,
        3..=4 => 2,
        5..=7 => 4,
        _ => 8,
    }
}

#[cfg(not(feature = "parallel"))]
fn configure_parallelism() {}

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

#[inline]
fn dot2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ax.mul_add(bx, ay * by)
}

/// Commands that can be applied to the world from external control surfaces.
#[derive(Debug, Clone)]
pub enum ControlCommand {
    UpdateConfig(Box<ScriptBotsConfig>),
    UpdateSelection(SelectionUpdate),
}

/// Apply a control command to the world state.
pub fn apply_control_command(
    world: &mut WorldState,
    command: ControlCommand,
) -> Result<(), WorldStateError> {
    match command {
        ControlCommand::UpdateConfig(config) => world.apply_config_update(*config),
        ControlCommand::UpdateSelection(update) => {
            world.apply_selection_update(update);
            Ok(())
        }
    }
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

#[derive(Default, Clone)]
struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        if self.count > 1 {
            self.m2 / (self.count - 1) as f64
        } else {
            0.0
        }
    }

    fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }
}

fn summarize_signal(values: &[f32]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let len = values.len() as f64;
    let mut sum = 0.0f64;
    let mut max = 0.0f32;
    let mut positive_sum = 0.0f64;
    for &value in values {
        let v = f64::from(value);
        sum += v;
        let magnitude = value.abs();
        if magnitude > max {
            max = magnitude;
        }
        if value > 0.0 {
            positive_sum += f64::from(value);
        } else if value < 0.0 {
            positive_sum += f64::from(-value);
        }
    }
    let mean = sum / len;
    let peak = max as f64;

    if positive_sum <= f64::EPSILON {
        return (mean, peak, 0.0);
    }

    let mut entropy = 0.0f64;
    for &value in values {
        let weight = value.abs() as f64 / positive_sum;
        if weight > 0.0 {
            entropy -= weight * weight.ln();
        }
    }
    (mean, peak, entropy)
}

fn sanitize_metric_key(label: &str) -> String {
    let mut result = String::with_capacity(label.len());
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            result.push(ch.to_ascii_lowercase());
        } else {
            result.push('_');
        }
    }
    result
}

fn summarize_food_grid(cells: &[f32]) -> Option<(f64, f64, f64, f32)> {
    if cells.is_empty() {
        return None;
    }
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut max = f32::MIN;
    for &value in cells {
        let v = f64::from(value);
        sum += v;
        sum_sq += v * v;
        if value > max {
            max = value;
        }
    }
    let count = cells.len() as f64;
    let mean = sum / count;
    let variance = if count > 1.0 {
        (sum_sq - sum * mean) / (count - 1.0)
    } else {
        0.0
    };
    Some((sum, mean, variance.max(0.0), max))
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

/// Coarse dietary classification used for debug surfacing.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum DietClass {
    #[default]
    Herbivore,
    Omnivore,
    Carnivore,
}

impl DietClass {
    #[must_use]
    pub fn from_tendency(tendency: f32) -> Self {
        if tendency <= 0.33 {
            Self::Herbivore
        } else if tendency >= 0.66 {
            Self::Carnivore
        } else {
            Self::Omnivore
        }
    }
}

/// Strategies for applying selection updates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SelectionMode {
    Replace,
    Add,
    Clear,
}

/// External selection update request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionUpdate {
    pub mode: SelectionMode,
    #[serde(default)]
    pub agent_ids: Vec<u64>,
    #[serde(default = "SelectionUpdate::default_state")]
    pub state: SelectionState,
}

impl SelectionUpdate {
    const fn default_state() -> SelectionState {
        SelectionState::Selected
    }
}

/// Resulting counts from applying a selection update.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct SelectionResult {
    pub applied: usize,
    pub cleared: usize,
    pub remaining_selected: usize,
}

/// Sort options for agent debug listings.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum AgentDebugSort {
    #[default]
    EnergyDesc,
    AgeDesc,
}

/// Query parameters for a debug view of agents.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentDebugQuery {
    #[serde(default)]
    pub ids: Option<Vec<u64>>,
    #[serde(default)]
    pub diet: Option<DietClass>,
    #[serde(default)]
    pub selection: Option<SelectionState>,
    #[serde(default)]
    pub brain_kind: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub sort: AgentDebugSort,
}

/// Debug projection of an agent suitable for external tooling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDebugInfo {
    pub agent_id: u64,
    pub selection: SelectionState,
    pub position: Position,
    pub energy: f32,
    pub health: f32,
    pub age: u32,
    pub generation: u32,
    pub herbivore_tendency: f32,
    pub diet: DietClass,
    pub brain_kind: Option<String>,
    pub brain_key: Option<u64>,
    pub mutation_primary: f32,
    pub mutation_secondary: f32,
    pub indicator: IndicatorState,
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

    /// Fetch a snapshot of internal brain activations if supported by the runner.
    #[must_use]
    pub fn snapshot_activations(&self) -> Option<BrainActivations> {
        self.runner.as_ref().and_then(|r| r.snapshot_activations())
    }
}

/// Thin trait object used to drive brain evaluations without coupling to concrete brain crates.
pub trait BrainRunner: Send + Sync {
    /// Static identifier of the brain implementation.
    fn kind(&self) -> &'static str;

    /// Evaluate outputs for the provided sensors.
    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE];

    /// Optional snapshot of internal activation state for visualization.
    /// Defaults to `None` when the runner does not support introspection.
    fn snapshot_activations(&self) -> Option<BrainActivations> {
        None
    }
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

    /// Pick a random registered brain key, if any.
    pub fn random_key(&self, rng: &mut dyn RngCore) -> Option<u64> {
        if self.entries.is_empty() {
            return None;
        }
        // Select from a sorted key list for stable ordering across hashseed/platforms
        let mut keys: Vec<u64> = self.entries.keys().copied().collect();
        keys.sort_unstable();
        let idx = rng.random_range(0..keys.len());
        keys.get(idx).copied()
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
    pub food_balance_total: f32,
    #[serde(skip)]
    pub brain_activations: Option<BrainActivations>,
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
            food_balance_total: 0.0,
            brain_activations: None,
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
    parent_id: AgentId,
    partner_id: Option<AgentId>,
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TickSummary {
    pub tick: Tick,
    pub agent_count: usize,
    pub births: usize,
    pub deaths: usize,
    pub total_energy: f32,
    pub average_energy: f32,
    pub average_health: f32,
    #[serde(default)]
    pub max_age: u32,
    #[serde(default)]
    pub spike_hits: u32,
}

/// Serializable representation of [`TickSummary`] for API surfaces.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TickSummaryDto {
    pub tick: u64,
    pub agent_count: usize,
    pub births: usize,
    pub deaths: usize,
    pub total_energy: f32,
    pub average_energy: f32,
    pub average_health: f32,
    pub max_age: u32,
    pub spike_hits: u32,
}

impl From<TickSummary> for TickSummaryDto {
    fn from(summary: TickSummary) -> Self {
        Self {
            tick: summary.tick.0,
            agent_count: summary.agent_count,
            births: summary.births,
            deaths: summary.deaths,
            total_energy: summary.total_energy,
            average_energy: summary.average_energy,
            average_health: summary.average_health,
            max_age: summary.max_age,
            spike_hits: summary.spike_hits,
        }
    }
}

impl From<TickSummaryDto> for TickSummary {
    fn from(dto: TickSummaryDto) -> Self {
        Self {
            tick: Tick(dto.tick),
            agent_count: dto.agent_count,
            births: dto.births,
            deaths: dto.deaths,
            total_energy: dto.total_energy,
            average_energy: dto.average_energy,
            average_health: dto.average_health,
            max_age: dto.max_age,
            spike_hits: dto.spike_hits,
        }
    }
}

// --- Centralized preset definitions and helpers ---
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PresetKind {
    Arctic,
    BoomBust,
    ClosedWorld,
}

impl PresetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            PresetKind::Arctic => "arctic",
            PresetKind::BoomBust => "boom_bust",
            PresetKind::ClosedWorld => "closed_world",
        }
    }

    pub fn all() -> &'static [PresetKind] {
        const ALL: &[PresetKind] = &[
            PresetKind::Arctic,
            PresetKind::BoomBust,
            PresetKind::ClosedWorld,
        ];
        ALL
    }

    pub fn from_name(name: &str) -> Option<PresetKind> {
        match name.trim().to_ascii_lowercase().as_str() {
            "arctic" => Some(PresetKind::Arctic),
            "boom_bust" | "boombust" | "boom-bust" => Some(PresetKind::BoomBust),
            "closed_world" | "closedworld" | "closed-world" => Some(PresetKind::ClosedWorld),
            _ => None,
        }
    }

    pub fn apply_to_config(self, config: &mut ScriptBotsConfig) {
        match self {
            PresetKind::Arctic => {
                config.temperature_gradient_exponent = 1.6;
                config.food_max = 0.35;
                config.food_growth_rate = 0.03;
            }
            PresetKind::BoomBust => {
                config.food_growth_rate = 0.12;
                config.food_decay_rate = 0.01;
                config.population_spawn_interval = 60;
            }
            PresetKind::ClosedWorld => {
                config.population_minimum = 0;
                config.population_spawn_interval = 0;
            }
        }
    }

    pub fn patch(self) -> serde_json::Value {
        match self {
            PresetKind::Arctic => serde_json::json!({
                "temperature_gradient_exponent": 1.6,
                "food_max": 0.35,
                "food_growth_rate": 0.03
            }),
            PresetKind::BoomBust => serde_json::json!({
                "food_growth_rate": 0.12,
                "food_decay_rate": 0.01,
                "population_spawn_interval": 60
            }),
            PresetKind::ClosedWorld => serde_json::json!({
                "population_minimum": 0,
                "population_spawn_interval": 0
            }),
        }
    }
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

/// Reason recorded for an agent death.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DeathCause {
    CombatCarnivore,
    CombatHerbivore,
    Starvation,
    Aging,
    Unknown,
}

/// Metadata captured when an agent is spawned.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BirthRecord {
    pub tick: Tick,
    pub agent_id: AgentId,
    pub parent_a: Option<AgentId>,
    pub parent_b: Option<AgentId>,
    pub brain_kind: Option<String>,
    pub brain_key: Option<u64>,
    pub herbivore_tendency: f32,
    pub generation: Generation,
    pub position: Position,
    pub is_hybrid: bool,
}

/// Lifecycle summary recorded when an agent is removed from the world.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeathRecord {
    pub tick: Tick,
    pub agent_id: AgentId,
    pub age: u32,
    pub generation: Generation,
    pub herbivore_tendency: f32,
    pub brain_kind: Option<String>,
    pub brain_key: Option<u64>,
    pub energy: f32,
    pub food_balance_total: f32,
    pub cause: DeathCause,
    pub was_hybrid: bool,
    pub combat_flags: CombatEventFlags,
}

/// Agent pipeline stages used to categorize replay RNG scopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayAgentPhase {
    Movement,
    Reproduction,
    Mutation,
    Spawn,
    Selection,
    Misc,
}

/// Identifies where in the simulation a random sample originated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayRngScope {
    World,
    Agent {
        agent_id: AgentId,
        phase: ReplayAgentPhase,
    },
}

/// Detailed event recordings emitted for deterministic replays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReplayEventKind {
    BrainOutputs {
        outputs: Vec<f32>,
    },
    Action {
        left_wheel: f32,
        right_wheel: f32,
        boost: bool,
        spike_target: Option<AgentId>,
        sound_level: f32,
        give_intent: f32,
    },
    RngSample {
        scope: ReplayRngScope,
        range_min: f32,
        range_max: f32,
        value: f32,
    },
}

/// Lightweight wrapper pairing an agent context with a replay event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayEvent {
    pub agent_id: Option<AgentId>,
    pub kind: ReplayEventKind,
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
    pub births: Vec<BirthRecord>,
    pub deaths: Vec<DeathRecord>,
    pub replay_events: Vec<ReplayEvent>,
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

/// Controls analytics sampling cadence for various metric families.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnalyticsStride {
    /// Additional macro-level summaries (population mix, resources).
    pub macro_metrics: u32,
    /// Behavior fingerprints and sensor/output aggregates.
    pub behavior_metrics: u32,
    /// Birth/death lifecycle event persistence.
    pub lifecycle_events: u32,
}

impl Default for AnalyticsStride {
    fn default() -> Self {
        Self {
            macro_metrics: 1,
            behavior_metrics: 120,
            lifecycle_events: 1,
        }
    }
}

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

/// Control-related runtime behavior toggles.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ControlSettings {
    /// Auto-pause the simulation when population is at or below this threshold. None disables.
    pub auto_pause_population_below: Option<u32>,
    /// Auto-pause when any agent reaches at least this age. None disables.
    pub auto_pause_age_above: Option<u32>,
    /// Auto-pause after a spike hit is recorded.
    pub auto_pause_on_spike_hit: bool,
}

/// Configuration change audit entry captured in-process.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigAuditEntry {
    pub tick: u64,
    pub patch: serde_json::Value,
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
    /// Baseline fertility offset applied to every terrain tile before other weights.
    pub food_fertility_base: f32,
    /// Weight applied to terrain moisture when computing fertility.
    pub food_moisture_weight: f32,
    /// Weight applied to terrain elevation when computing fertility.
    pub food_elevation_weight: f32,
    /// Weight applied to local slope magnitude when computing fertility.
    pub food_slope_weight: f32,
    /// Minimum fraction of `food_max` available as capacity regardless of fertility.
    pub food_capacity_base: f32,
    /// Additional capacity fraction unlocked by perfect fertility.
    pub food_capacity_fertility: f32,
    /// Multiplier controlling how strongly fertility accelerates regrowth.
    pub food_growth_fertility: f32,
    /// Multiplier controlling how strongly infertility increases decay.
    pub food_decay_infertility: f32,
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
    /// Interval (in ticks) controlling when reproduction attempts are evaluated. `0` allows attempts every tick.
    pub reproduction_attempt_interval: u32,
    /// Probability that a ready agent reproduces when the attempt cadence fires.
    pub reproduction_attempt_chance: f32,
    /// Herbivore reproduction rate multiplier applied per tick.
    pub reproduction_rate_herbivore: f32,
    /// Carnivore reproduction rate multiplier applied per tick.
    pub reproduction_rate_carnivore: f32,
    /// Bonus applied to the reproduction counter per unit ground intake.
    pub reproduction_food_bonus: f32,
    /// Fertility-based multiplier applied to reproduction bonuses.
    pub reproduction_fertility_bonus: f32,
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
    /// Number of ticks between age increments and associated aging checks.
    pub aging_tick_interval: u32,
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
    /// Sampling cadence for analytics families.
    pub analytics_stride: AnalyticsStride,
    /// NeuroFlow runtime configuration.
    pub neuroflow: NeuroflowSettings,
    /// Control-related runtime behavior toggles.
    pub control: ControlSettings,
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
            food_fertility_base: 0.2,
            food_moisture_weight: 0.6,
            food_elevation_weight: 0.4,
            food_slope_weight: 6.0,
            food_capacity_base: 0.3,
            food_capacity_fertility: 0.6,
            food_growth_fertility: 0.7,
            food_decay_infertility: 0.5,
            food_sharing_radius: 50.0,
            food_sharing_rate: 0.1,
            food_transfer_rate: 0.001,
            food_sharing_distance: 50.0,
            reproduction_energy_threshold: 0.65,
            reproduction_energy_cost: 0.0,
            reproduction_cooldown: 300,
            reproduction_attempt_interval: 15,
            reproduction_attempt_chance: 0.1,
            reproduction_rate_herbivore: 1.0,
            reproduction_rate_carnivore: 1.0,
            reproduction_food_bonus: 3.0,
            reproduction_fertility_bonus: 0.5,
            reproduction_child_energy: 1.0,
            reproduction_spawn_jitter: 20.0,
            reproduction_color_jitter: 0.05,
            reproduction_mutation_scale: 0.02,
            reproduction_partner_chance: 0.15,
            reproduction_spawn_back_distance: 12.0,
            reproduction_gene_log_capacity: 12,
            reproduction_meta_mutation_chance: 0.2,
            reproduction_meta_mutation_scale: 0.5,
            aging_tick_interval: 100,
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
            analytics_stride: AnalyticsStride::default(),
            neuroflow: NeuroflowSettings {
                enabled: true,
                ..NeuroflowSettings::default()
            },
            control: ControlSettings::default(),
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
    pub fn food_dimensions(&self) -> Result<(u32, u32), WorldStateError> {
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
            || self.food_fertility_base < 0.0
            || self.food_fertility_base > 1.0
            || self.food_moisture_weight < 0.0
            || self.food_elevation_weight < 0.0
            || self.food_slope_weight < 0.0
            || self.food_capacity_base < 0.0
            || self.food_capacity_base > 1.0
            || self.food_capacity_fertility < 0.0
            || self.food_growth_fertility < 0.0
            || self.food_decay_infertility < 0.0
            || self.reproduction_food_bonus < 0.0
            || self.reproduction_fertility_bonus < 0.0
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
            || !(0.0..=1.0).contains(&self.reproduction_attempt_chance)
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
            || self.aging_tick_interval == 0
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
        if self.food_capacity_base + self.food_capacity_fertility > 1.0 {
            return Err(WorldStateError::InvalidConfig(
                "food_capacity_base + food_capacity_fertility must be <= 1.0",
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

                let fertility_bias = default_tile_fertility_bias(kind, elevation, moisture);
                let temperature_bias = default_tile_temperature_bias(fy);
                let palette_index = default_tile_palette_index(kind);

                tiles.push(TerrainTile {
                    kind,
                    elevation,
                    moisture,
                    accent: accent_noise,
                    fertility_bias,
                    temperature_bias,
                    palette_index,
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

    pub fn from_tiles(
        width: u32,
        height: u32,
        cell_size: u32,
        tiles: Vec<TerrainTile>,
    ) -> Result<Self, WorldStateError> {
        if width == 0 || height == 0 {
            return Err(WorldStateError::InvalidConfig(
                "terrain dimensions must be non-zero",
            ));
        }
        let expected = (width as usize) * (height as usize);
        if tiles.len() != expected {
            return Err(WorldStateError::InvalidConfig(
                "terrain tile count does not match dimensions",
            ));
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

fn default_tile_fertility_bias(kind: TerrainKind, elevation: f32, moisture: f32) -> f32 {
    let kind_bonus = terrain_kind_fertility_bonus(kind);
    let moisture_term = (moisture - 0.5) * 0.35;
    let elevation_term = (elevation - 0.5) * 0.4;
    (kind_bonus + 0.5 + moisture_term - elevation_term).clamp(0.0, 1.0)
}

fn default_tile_temperature_bias(normalized_y: f32) -> f32 {
    (1.0 - normalized_y).clamp(0.0, 1.0)
}

fn default_tile_palette_index(kind: TerrainKind) -> u16 {
    match kind {
        TerrainKind::DeepWater => 0,
        TerrainKind::ShallowWater => 1,
        TerrainKind::Sand => 2,
        TerrainKind::Grass => 3,
        TerrainKind::Bloom => 4,
        TerrainKind::Rock => 5,
    }
}

/// Terrain classification for each tile.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
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
    #[serde(default)]
    pub fertility_bias: f32,
    #[serde(default)]
    pub temperature_bias: f32,
    #[serde(default)]
    pub palette_index: u16,
}
mod map_sandbox {
    use super::{
        TerrainKind, TerrainLayer, TerrainTile, default_tile_fertility_bias,
        default_tile_palette_index,
    };
    use direction::{CardinalDirection, CardinalDirectionTable};
    use rand08::{SeedableRng, rngs::StdRng};
    use serde::{Deserialize, Serialize};
    use std::collections::{HashMap, HashSet};
    use std::hash::{DefaultHasher, Hasher};
    use std::num::NonZeroU32;
    use std::time::{SystemTime, UNIX_EPOCH};
    use wfc::{
        Coord, GlobalStats, PatternDescription, PatternTable, RunOwnAll, Size, Wave,
        retry::{self, RetryOwnAll},
    };

    const DEFAULT_RETRY_BUDGET: usize = 32;

    #[derive(Debug, thiserror::Error)]
    pub enum MapGenerationError {
        #[error("tileset contains no tiles")]
        EmptyTileset,
        #[error("duplicate tile id `{0}` in tileset")]
        DuplicateTileId(String),
        #[error("adjacency references unknown tile `{0}`")]
        UnknownTile(String),
        #[error("adjacency uses invalid direction `{0}`")]
        InvalidDirection(String),
        #[error("tile `{0}` weight must be greater than zero")]
        InvalidTileWeight(String),
        #[error("no compatible neighbors remain for tile `{tile}` toward `{direction:?}`")]
        EmptyAdjacency {
            tile: String,
            direction: CardinalDirection,
        },
        #[error("generation failed after {attempts} attempts due to contradictions")]
        Contradiction { attempts: usize },
        #[error("terrain dimensions must be non-zero")]
        InvalidDimensions,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TilesetSpec {
        pub id: String,
        #[serde(default)]
        pub label: Option<String>,
        #[serde(default)]
        pub description: Option<String>,
        #[serde(default)]
        pub tiles: Vec<TileSpec>,
        #[serde(default)]
        pub adjacency: Vec<AdjacencySpec>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TileSpec {
        pub id: String,
        #[serde(default)]
        pub label: Option<String>,
        #[serde(default = "TileSpec::default_weight")]
        pub weight: u32,
        pub terrain_kind: TerrainKind,
        #[serde(default)]
        pub fertility_bias: Option<f32>,
        #[serde(default)]
        pub temperature_bias: Option<f32>,
        #[serde(default)]
        pub elevation: Option<f32>,
        #[serde(default)]
        pub moisture: Option<f32>,
        #[serde(default)]
        pub accent: Option<f32>,
        #[serde(default)]
        pub palette_index: Option<u16>,
        #[serde(default)]
        pub permeability: Option<f32>,
        #[serde(default)]
        pub runoff_bias: Option<f32>,
        #[serde(default)]
        pub basin_rank: Option<f32>,
        #[serde(default)]
        pub channel_priority: Option<f32>,
        #[serde(default)]
        pub swim_cost: Option<f32>,
    }

    impl TileSpec {
        const fn default_weight() -> u32 {
            1
        }

        fn weight(&self) -> Result<NonZeroU32, MapGenerationError> {
            NonZeroU32::new(self.weight)
                .ok_or_else(|| MapGenerationError::InvalidTileWeight(self.id.clone()))
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AdjacencySpec {
        pub tile_a: String,
        pub side_a: String,
        pub tile_b: String,
        pub side_b: String,
        #[serde(default = "AdjacencySpec::default_allowed")]
        pub allowed: bool,
    }

    impl AdjacencySpec {
        const fn default_allowed() -> bool {
            true
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
    pub enum MapGeneratorKind {
        RuleBased,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MapArtifactMetadata {
        pub generator: MapGeneratorKind,
        pub tileset_id: String,
        pub tileset_hash: u64,
        pub seed: u64,
        pub width: u32,
        pub height: u32,
        pub attempt_count: usize,
        pub succeeded_on: usize,
        pub generated_at_epoch_ms: u128,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ScalarField {
        width: u32,
        height: u32,
        values: Vec<f32>,
    }

    impl ScalarField {
        pub fn new(width: u32, height: u32, values: Vec<f32>) -> Self {
            debug_assert_eq!(values.len(), (width as usize) * (height as usize));
            Self {
                width,
                height,
                values,
            }
        }

        pub fn width(&self) -> u32 {
            self.width
        }

        pub fn height(&self) -> u32 {
            self.height
        }

        pub fn values(&self) -> &[f32] {
            &self.values
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HydrologyTile {
        pub permeability: f32,
        pub runoff_bias: f32,
        pub basin_rank: f32,
        pub channel_priority: f32,
        pub swim_cost: f32,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HydrologyTileLayer {
        width: u32,
        height: u32,
        tiles: Vec<HydrologyTile>,
    }

    impl HydrologyTileLayer {
        pub fn new(width: u32, height: u32, tiles: Vec<HydrologyTile>) -> Self {
            debug_assert_eq!(tiles.len(), (width as usize) * (height as usize));
            Self {
                width,
                height,
                tiles,
            }
        }

        pub fn width(&self) -> u32 {
            self.width
        }

        pub fn height(&self) -> u32 {
            self.height
        }

        pub fn tiles(&self) -> &[HydrologyTile] {
            &self.tiles
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
    pub enum HydrologyFlowDirection {
        None,
        North,
        South,
        East,
        West,
    }

    impl HydrologyFlowDirection {
        fn from_cardinal(direction: Option<CardinalDirection>) -> Self {
            match direction {
                Some(CardinalDirection::North) => Self::North,
                Some(CardinalDirection::South) => Self::South,
                Some(CardinalDirection::East) => Self::East,
                Some(CardinalDirection::West) => Self::West,
                None => Self::None,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HydrologyField {
        width: u32,
        height: u32,
        flow_directions: Vec<HydrologyFlowDirection>,
        accumulation: Vec<f32>,
        spill_elevation: Vec<f32>,
        basin_ids: Vec<u32>,
        initial_water_depth: Vec<f32>,
    }

    impl HydrologyField {
        pub fn new(
            width: u32,
            height: u32,
            flow_directions: Vec<HydrologyFlowDirection>,
            accumulation: Vec<f32>,
            spill_elevation: Vec<f32>,
            basin_ids: Vec<u32>,
            initial_water_depth: Vec<f32>,
        ) -> Self {
            debug_assert_eq!(flow_directions.len(), (width as usize) * (height as usize));
            debug_assert_eq!(accumulation.len(), flow_directions.len());
            debug_assert_eq!(spill_elevation.len(), flow_directions.len());
            debug_assert_eq!(basin_ids.len(), flow_directions.len());
            debug_assert_eq!(initial_water_depth.len(), flow_directions.len());
            Self {
                width,
                height,
                flow_directions,
                accumulation,
                spill_elevation,
                basin_ids,
                initial_water_depth,
            }
        }

        pub fn width(&self) -> u32 {
            self.width
        }

        pub fn height(&self) -> u32 {
            self.height
        }

        pub fn flow_directions(&self) -> &[HydrologyFlowDirection] {
            &self.flow_directions
        }

        pub fn accumulation(&self) -> &[f32] {
            &self.accumulation
        }

        pub fn spill_elevation(&self) -> &[f32] {
            &self.spill_elevation
        }

        pub fn basin_ids(&self) -> &[u32] {
            &self.basin_ids
        }

        pub fn initial_water_depth(&self) -> &[f32] {
            &self.initial_water_depth
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MapArtifact {
        terrain: TerrainLayer,
        fertility: Option<ScalarField>,
        temperature: Option<ScalarField>,
        hydrology_tiles: Option<HydrologyTileLayer>,
        hydrology_field: Option<HydrologyField>,
        metadata: MapArtifactMetadata,
    }

    impl MapArtifact {
        pub fn terrain(&self) -> &TerrainLayer {
            &self.terrain
        }

        pub fn fertility(&self) -> Option<&ScalarField> {
            self.fertility.as_ref()
        }

        pub fn temperature(&self) -> Option<&ScalarField> {
            self.temperature.as_ref()
        }

        pub fn hydrology_tiles(&self) -> Option<&HydrologyTileLayer> {
            self.hydrology_tiles.as_ref()
        }

        pub fn hydrology_field(&self) -> Option<&HydrologyField> {
            self.hydrology_field.as_ref()
        }

        pub fn metadata(&self) -> &MapArtifactMetadata {
            &self.metadata
        }
    }

    #[derive(Clone)]
    struct CompiledTile {
        id: String,
        _label: Option<String>,
        terrain_kind: TerrainKind,
        weight: NonZeroU32,
        fertility_bias: f32,
        temperature_bias: f32,
        elevation: f32,
        moisture: f32,
        accent: f32,
        palette_index: u16,
        permeability: f32,
        runoff_bias: f32,
        basin_rank: f32,
        channel_priority: f32,
        swim_cost: f32,
    }

    #[derive(Default, Clone)]
    struct DirectionRule {
        explicit_allows: bool,
        allowed: HashSet<usize>,
        forbidden: HashSet<usize>,
    }

    impl DirectionRule {
        fn allow(&mut self, idx: usize) {
            if !self.explicit_allows {
                self.explicit_allows = true;
                self.allowed.clear();
            }
            self.allowed.insert(idx);
            self.forbidden.remove(&idx);
        }

        fn forbid(&mut self, idx: usize) {
            if self.explicit_allows {
                self.allowed.remove(&idx);
            } else {
                self.forbidden.insert(idx);
            }
        }

        fn resolve(&self, population: usize) -> Vec<usize> {
            let mut entries = if self.explicit_allows {
                self.allowed.iter().copied().collect::<Vec<_>>()
            } else {
                (0..population)
                    .filter(|candidate| !self.forbidden.contains(candidate))
                    .collect::<Vec<_>>()
            };
            entries.sort_unstable();
            entries.dedup();
            entries
        }
    }

    pub struct RuleBasedMapGenerator {
        spec: TilesetSpec,
        compiled_tiles: Vec<CompiledTile>,
        global_stats: GlobalStats,
        tileset_hash: u64,
        retry_budget: usize,
    }

    impl RuleBasedMapGenerator {
        pub fn new(spec: TilesetSpec) -> Result<Self, MapGenerationError> {
            let tileset_hash = compute_tileset_hash(&spec);
            let (compiled_tiles, global_stats) = compile_tileset(&spec)?;
            Ok(Self {
                spec,
                compiled_tiles,
                global_stats,
                tileset_hash,
                retry_budget: DEFAULT_RETRY_BUDGET,
            })
        }

        pub fn with_retry_budget(mut self, retries: usize) -> Self {
            self.retry_budget = retries;
            self
        }

        pub fn spec(&self) -> &TilesetSpec {
            &self.spec
        }

        pub fn generate(
            &self,
            width: u32,
            height: u32,
            cell_size: u32,
            seed: u64,
        ) -> Result<MapArtifact, MapGenerationError> {
            if width == 0 || height == 0 {
                return Err(MapGenerationError::InvalidDimensions);
            }

            let mut rng = StdRng::seed_from_u64(seed);
            let runner = RunOwnAll::new(
                Size::new(width, height),
                self.global_stats.clone(),
                &mut rng,
            );
            let budget = self.retry_budget;
            let mut retry = retry::NumTimes(budget);
            let wave: Wave = match retry.retry(runner, &mut rng) {
                Ok(wave) => wave,
                Err(_) => {
                    return Err(MapGenerationError::Contradiction {
                        attempts: budget + 1,
                    });
                }
            };

            let attempts_spent = budget - retry.0;
            let success_attempt = attempts_spent + 1;

            let tile_capacity = (width as usize) * (height as usize);
            let mut tiles = Vec::with_capacity(tile_capacity);
            let mut fertility = Vec::with_capacity(tile_capacity);
            let mut temperature = Vec::with_capacity(tile_capacity);
            let mut hydrology_tiles = Vec::with_capacity(tile_capacity);

            for (coord, cell) in wave.grid().enumerate() {
                let pattern_id =
                    cell.chosen_pattern_id()
                        .map_err(|_| MapGenerationError::Contradiction {
                            attempts: success_attempt,
                        })?;
                let idx = pattern_id as usize;
                let tile =
                    self.compiled_tiles
                        .get(idx)
                        .ok_or(MapGenerationError::Contradiction {
                            attempts: success_attempt,
                        })?;
                let accent_noise = coordinate_noise(seed, coord);
                let accent = (tile.accent + accent_noise * 0.35).clamp(0.0, 1.0);
                tiles.push(TerrainTile {
                    kind: tile.terrain_kind,
                    elevation: tile.elevation,
                    moisture: tile.moisture,
                    accent,
                    fertility_bias: tile.fertility_bias,
                    temperature_bias: tile.temperature_bias,
                    palette_index: tile.palette_index,
                });
                fertility.push(tile.fertility_bias);
                temperature.push(tile.temperature_bias);
                hydrology_tiles.push(HydrologyTile {
                    permeability: tile.permeability,
                    runoff_bias: tile.runoff_bias,
                    basin_rank: tile.basin_rank,
                    channel_priority: tile.channel_priority,
                    swim_cost: tile.swim_cost,
                });
            }

            let terrain = TerrainLayer::from_tiles(width, height, cell_size, tiles)
                .map_err(|_| MapGenerationError::InvalidDimensions)?;
            let fertility_field = ScalarField::new(width, height, fertility);
            let temperature_field = ScalarField::new(width, height, temperature);
            let hydrology_layer = HydrologyTileLayer::new(width, height, hydrology_tiles);
            let hydrology_field =
                compute_hydrology_field(width, height, &terrain, &hydrology_layer);
            let metadata = MapArtifactMetadata {
                generator: MapGeneratorKind::RuleBased,
                tileset_id: self.spec.id.clone(),
                tileset_hash: self.tileset_hash,
                seed,
                width,
                height,
                attempt_count: budget + 1,
                succeeded_on: success_attempt,
                generated_at_epoch_ms: current_epoch_ms(),
            };

            Ok(MapArtifact {
                terrain,
                fertility: Some(fertility_field),
                temperature: Some(temperature_field),
                hydrology_tiles: Some(hydrology_layer),
                hydrology_field: Some(hydrology_field),
                metadata,
            })
        }
    }

    fn compile_tileset(
        spec: &TilesetSpec,
    ) -> Result<(Vec<CompiledTile>, GlobalStats), MapGenerationError> {
        if spec.tiles.is_empty() {
            return Err(MapGenerationError::EmptyTileset);
        }

        let mut index_by_id = HashMap::new();
        for (idx, tile) in spec.tiles.iter().enumerate() {
            if index_by_id.insert(tile.id.clone(), idx).is_some() {
                return Err(MapGenerationError::DuplicateTileId(tile.id.clone()));
            }
        }

        let compiled_tiles = spec
            .tiles
            .iter()
            .map(compile_tile)
            .collect::<Result<Vec<_>, _>>()?;

        let mut rules = Vec::with_capacity(compiled_tiles.len());
        for _ in &compiled_tiles {
            rules.push(CardinalDirectionTable::default());
        }

        for adjacency in &spec.adjacency {
            let Some(&a_idx) = index_by_id.get(&adjacency.tile_a) else {
                return Err(MapGenerationError::UnknownTile(adjacency.tile_a.clone()));
            };
            let Some(&b_idx) = index_by_id.get(&adjacency.tile_b) else {
                return Err(MapGenerationError::UnknownTile(adjacency.tile_b.clone()));
            };
            let dir_a = parse_direction(&adjacency.side_a)
                .ok_or_else(|| MapGenerationError::InvalidDirection(adjacency.side_a.clone()))?;
            let dir_b = parse_direction(&adjacency.side_b)
                .ok_or_else(|| MapGenerationError::InvalidDirection(adjacency.side_b.clone()))?;

            update_direction_rule(&mut rules[a_idx][dir_a], b_idx, adjacency.allowed);
            update_direction_rule(&mut rules[b_idx][dir_b], a_idx, adjacency.allowed);
        }

        let pattern_descriptions = compiled_tiles
            .iter()
            .enumerate()
            .map(|(idx, tile)| {
                let mut neighbors = CardinalDirectionTable::default();
                for direction in direction::CardinalDirections {
                    let resolved = rules[idx][direction].resolve(compiled_tiles.len());
                    if resolved.is_empty() {
                        return Err(MapGenerationError::EmptyAdjacency {
                            tile: tile.id.clone(),
                            direction,
                        });
                    }
                    neighbors[direction] = resolved.iter().map(|&value| value as u32).collect();
                }
                Ok(PatternDescription::new(Some(tile.weight), neighbors))
            })
            .collect::<Result<Vec<_>, MapGenerationError>>()?;

        let global_stats = GlobalStats::new(PatternTable::from_vec(pattern_descriptions));
        Ok((compiled_tiles, global_stats))
    }

    fn compile_tile(tile: &TileSpec) -> Result<CompiledTile, MapGenerationError> {
        let elevation = tile
            .elevation
            .unwrap_or(default_elevation_for_kind(tile.terrain_kind));
        let moisture = tile
            .moisture
            .unwrap_or(default_moisture_for_kind(tile.terrain_kind));
        let fertility_bias = tile.fertility_bias.unwrap_or(default_tile_fertility_bias(
            tile.terrain_kind,
            elevation,
            moisture,
        ));
        let temperature_bias = tile.temperature_bias.unwrap_or(0.5);
        let accent = tile.accent.unwrap_or(0.5);
        let palette_index = tile
            .palette_index
            .unwrap_or(default_tile_palette_index(tile.terrain_kind));
        let permeability = tile
            .permeability
            .unwrap_or(default_permeability_for_kind(tile.terrain_kind));
        let runoff_bias = tile
            .runoff_bias
            .unwrap_or(default_runoff_bias_for_kind(tile.terrain_kind));
        let basin_rank = tile
            .basin_rank
            .unwrap_or(default_basin_rank_for_kind(tile.terrain_kind));
        let channel_priority = tile
            .channel_priority
            .unwrap_or(default_channel_priority_for_kind(tile.terrain_kind));
        let swim_cost = tile
            .swim_cost
            .unwrap_or(default_swim_cost_for_kind(tile.terrain_kind));

        Ok(CompiledTile {
            id: tile.id.clone(),
            _label: tile.label.clone(),
            terrain_kind: tile.terrain_kind,
            weight: tile.weight()?,
            fertility_bias: fertility_bias.clamp(0.0, 1.0),
            temperature_bias: temperature_bias.clamp(0.0, 1.0),
            elevation: elevation.clamp(0.0, 1.0),
            moisture: moisture.clamp(0.0, 1.0),
            accent: accent.clamp(0.0, 1.0),
            palette_index,
            permeability: permeability.clamp(0.0, 1.0),
            runoff_bias: runoff_bias.clamp(-1.0, 1.0),
            basin_rank: basin_rank.clamp(0.0, 1.0),
            channel_priority: channel_priority.clamp(0.0, 1.0),
            swim_cost: swim_cost.max(0.0),
        })
    }

    fn update_direction_rule(rule: &mut DirectionRule, neighbor: usize, allowed: bool) {
        if allowed {
            rule.allow(neighbor);
        } else {
            rule.forbid(neighbor);
        }
    }

    fn parse_direction(raw: &str) -> Option<CardinalDirection> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "n" | "north" | "up" => Some(CardinalDirection::North),
            "s" | "south" | "down" => Some(CardinalDirection::South),
            "e" | "east" | "right" => Some(CardinalDirection::East),
            "w" | "west" | "left" => Some(CardinalDirection::West),
            _ => None,
        }
    }

    fn default_elevation_for_kind(kind: TerrainKind) -> f32 {
        match kind {
            TerrainKind::DeepWater => 0.1,
            TerrainKind::ShallowWater => 0.23,
            TerrainKind::Sand => 0.34,
            TerrainKind::Grass => 0.5,
            TerrainKind::Bloom => 0.58,
            TerrainKind::Rock => 0.85,
        }
    }

    fn default_moisture_for_kind(kind: TerrainKind) -> f32 {
        match kind {
            TerrainKind::DeepWater => 0.95,
            TerrainKind::ShallowWater => 0.85,
            TerrainKind::Sand => 0.2,
            TerrainKind::Grass => 0.5,
            TerrainKind::Bloom => 0.8,
            TerrainKind::Rock => 0.25,
        }
    }

    fn default_permeability_for_kind(kind: TerrainKind) -> f32 {
        match kind {
            TerrainKind::DeepWater => 0.05,
            TerrainKind::ShallowWater => 0.15,
            TerrainKind::Sand => 0.8,
            TerrainKind::Grass => 0.6,
            TerrainKind::Bloom => 0.5,
            TerrainKind::Rock => 0.1,
        }
    }

    fn default_runoff_bias_for_kind(kind: TerrainKind) -> f32 {
        match kind {
            TerrainKind::DeepWater => 0.9,
            TerrainKind::ShallowWater => 0.6,
            TerrainKind::Sand => -0.2,
            TerrainKind::Grass => 0.1,
            TerrainKind::Bloom => -0.1,
            TerrainKind::Rock => 0.5,
        }
    }

    fn default_basin_rank_for_kind(kind: TerrainKind) -> f32 {
        match kind {
            TerrainKind::DeepWater => 1.0,
            TerrainKind::ShallowWater => 0.8,
            TerrainKind::Sand => 0.35,
            TerrainKind::Grass => 0.4,
            TerrainKind::Bloom => 0.55,
            TerrainKind::Rock => 0.2,
        }
    }

    fn default_channel_priority_for_kind(kind: TerrainKind) -> f32 {
        match kind {
            TerrainKind::DeepWater => 0.2,
            TerrainKind::ShallowWater => 0.6,
            TerrainKind::Sand => 0.4,
            TerrainKind::Grass => 0.5,
            TerrainKind::Bloom => 0.35,
            TerrainKind::Rock => 0.7,
        }
    }

    fn default_swim_cost_for_kind(kind: TerrainKind) -> f32 {
        match kind {
            TerrainKind::DeepWater => 0.0,
            TerrainKind::ShallowWater => 0.3,
            TerrainKind::Sand => 2.0,
            TerrainKind::Grass => 1.5,
            TerrainKind::Bloom => 1.2,
            TerrainKind::Rock => 2.5,
        }
    }

    fn coordinate_noise(seed: u64, coord: Coord) -> f32 {
        let mut value = seed
            .wrapping_mul(0x9e3779b185ebca87)
            .wrapping_add((coord.x as u64).wrapping_mul(0xc2b2ae3d27d4eb4f))
            .wrapping_add((coord.y as u64).wrapping_mul(0x165667b19e3779f9));
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58476d1ce4e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d049bb133111eb);
        value ^= value >> 31;
        ((value >> 11) as f64 / (1u64 << 53) as f64) as f32
    }

    fn compute_tileset_hash(spec: &TilesetSpec) -> u64 {
        let mut hasher = DefaultHasher::new();
        match serde_json::to_vec(spec) {
            Ok(bytes) => hasher.write(&bytes),
            Err(_) => hasher.write_u64(spec.tiles.len() as u64),
        }
        hasher.finish()
    }
    fn compute_hydrology_field(
        width: u32,
        height: u32,
        terrain: &TerrainLayer,
        hydrology: &HydrologyTileLayer,
    ) -> HydrologyField {
        let len = (width as usize) * (height as usize);
        let terrain_tiles = terrain.tiles();
        let hydrology_tiles = hydrology.tiles();
        let mut flow_directions = vec![HydrologyFlowDirection::None; len];
        let mut flow_targets: Vec<Option<usize>> = vec![None; len];
        let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); len];
        let mut spill_elevation = vec![0.0f32; len];
        let mut effective_elevation = vec![0.0f32; len];

        let width_i32 = width as i32;
        let height_i32 = height as i32;

        let neighbors = [
            (CardinalDirection::North, (0, -1)),
            (CardinalDirection::South, (0, 1)),
            (CardinalDirection::East, (1, 0)),
            (CardinalDirection::West, (-1, 0)),
        ];

        for y in 0..height_i32 {
            for x in 0..width_i32 {
                let idx = (y as usize) * (width as usize) + (x as usize);
                let tile = &terrain_tiles[idx];
                let hyd = &hydrology_tiles[idx];
                let permeability_penalty = (1.0 - hyd.permeability) * 0.04;
                let runoff_bonus = hyd.runoff_bias.max(0.0) * 0.03;
                let channel_bonus = (1.0 - hyd.channel_priority) * 0.02;
                effective_elevation[idx] =
                    tile.elevation + permeability_penalty + channel_bonus + runoff_bonus;

                let mut best_direction = HydrologyFlowDirection::None;
                let mut best_score = effective_elevation[idx] - 1e-6;
                let mut best_target = None;
                let mut min_neighbor_elevation = tile.elevation;

                for (direction, (dx, dy)) in neighbors.iter() {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || nx >= width_i32 || ny < 0 || ny >= height_i32 {
                        continue;
                    }
                    let nidx = (ny as usize) * (width as usize) + (nx as usize);
                    let neighbor_tile = &terrain_tiles[nidx];
                    let neighbor_hyd = &hydrology_tiles[nidx];
                    let slope_bonus = (tile.elevation - neighbor_tile.elevation).max(0.0) * 0.5;
                    let channel_synergy =
                        (hyd.channel_priority + neighbor_hyd.channel_priority) * 0.03;
                    let neighbor_permeability_penalty = (1.0 - neighbor_hyd.permeability) * 0.02;
                    let neighbor_score = effective_elevation[nidx] - slope_bonus - channel_synergy
                        + neighbor_permeability_penalty;

                    if neighbor_score < best_score {
                        best_score = neighbor_score;
                        best_direction = HydrologyFlowDirection::from_cardinal(Some(*direction));
                        best_target = Some(nidx);
                    }

                    if neighbor_tile.elevation < min_neighbor_elevation {
                        min_neighbor_elevation = neighbor_tile.elevation;
                    }
                }

                flow_directions[idx] = best_direction;
                flow_targets[idx] = best_target;
                if let Some(target) = best_target {
                    incoming[target].push(idx);
                }

                spill_elevation[idx] = min_neighbor_elevation;
            }
        }

        let mut accumulation = vec![0.0f32; len];
        let mut visited = vec![false; len];
        for idx in 0..len {
            accumulate_flow(idx, &incoming, &mut accumulation, &mut visited);
        }

        let mut basin_ids = vec![u32::MAX; len];
        let mut next_basin_id: u32 = 0;
        for idx in 0..len {
            if basin_ids[idx] != u32::MAX {
                continue;
            }
            let mut trail = Vec::new();
            let mut current = idx;
            let basin_id = loop {
                if basin_ids[current] != u32::MAX {
                    break basin_ids[current];
                }
                if let Some(pos) = trail.iter().position(|&value| value == current) {
                    let basin = next_basin_id;
                    next_basin_id += 1;
                    for node in &trail[pos..] {
                        basin_ids[*node] = basin;
                    }
                    break basin;
                }
                trail.push(current);
                match flow_targets[current] {
                    Some(next) if next != current => {
                        current = next;
                    }
                    _ => {
                        let basin = next_basin_id;
                        next_basin_id += 1;
                        basin_ids[current] = basin;
                        break basin;
                    }
                }
            };
            for node in trail {
                basin_ids[node] = basin_id;
            }
        }

        let mut initial_water_depth = Vec::with_capacity(len);
        for hyd in hydrology_tiles.iter().take(len) {
            let base_depth = hyd.basin_rank * 0.25 + hyd.runoff_bias.max(0.0) * 0.05;
            let permeability_discount = hyd.permeability * 0.1;
            let depth = (base_depth - permeability_discount).clamp(0.0, 0.6);
            initial_water_depth.push(depth);
        }

        HydrologyField::new(
            width,
            height,
            flow_directions,
            accumulation,
            spill_elevation,
            basin_ids,
            initial_water_depth,
        )
    }

    fn accumulate_flow(
        idx: usize,
        incoming: &[Vec<usize>],
        accumulation: &mut [f32],
        visited: &mut [bool],
    ) -> f32 {
        if visited[idx] {
            return accumulation[idx];
        }
        visited[idx] = true;
        let mut total = 1.0f32;
        accumulation[idx] = total;
        for &child in &incoming[idx] {
            total += accumulate_flow(child, incoming, accumulation, visited);
        }
        accumulation[idx] = total;
        total
    }

    fn current_epoch_ms() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|dur| dur.as_millis())
            .unwrap_or(0)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn rule_based_generator_produces_map() {
            let tileset = TilesetSpec {
                id: "unit".into(),
                label: None,
                description: None,
                tiles: vec![
                    TileSpec {
                        id: "grass".into(),
                        label: None,
                        weight: 1,
                        terrain_kind: TerrainKind::Grass,
                        fertility_bias: Some(0.7),
                        temperature_bias: Some(0.5),
                        elevation: Some(0.48),
                        moisture: Some(0.6),
                        accent: Some(0.3),
                        palette_index: Some(3),
                        permeability: Some(0.35),
                        runoff_bias: Some(0.2),
                        basin_rank: Some(0.55),
                        channel_priority: Some(0.4),
                        swim_cost: Some(1.2),
                    },
                    TileSpec {
                        id: "water".into(),
                        label: None,
                        weight: 1,
                        terrain_kind: TerrainKind::DeepWater,
                        fertility_bias: Some(0.05),
                        temperature_bias: Some(0.9),
                        elevation: Some(0.12),
                        moisture: Some(0.95),
                        accent: Some(0.4),
                        palette_index: Some(0),
                        permeability: Some(0.9),
                        runoff_bias: Some(0.8),
                        basin_rank: Some(0.1),
                        channel_priority: Some(0.9),
                        swim_cost: Some(0.2),
                    },
                ],
                adjacency: Vec::new(),
            };

            let generator = RuleBasedMapGenerator::new(tileset).expect("compile tileset");
            let artifact = generator.generate(8, 8, 16, 42).expect("generate artifact");

            assert_eq!(artifact.terrain().width(), 8);
            assert_eq!(artifact.terrain().height(), 8);
            assert_eq!(artifact.metadata().tileset_id, "unit");
            assert!(artifact.fertility().is_some());
            let hydrology = artifact.hydrology_tiles().expect("hydrology tiles present");
            assert_eq!(hydrology.width(), 8);
            assert_eq!(hydrology.height(), 8);
            assert!(hydrology.tiles().iter().any(|tile| tile.permeability > 0.0));
            let hydrology_field = artifact.hydrology_field().expect("hydrology field present");
            assert_eq!(hydrology_field.width(), 8);
            assert_eq!(hydrology_field.height(), 8);
            assert!(
                hydrology_field
                    .accumulation()
                    .iter()
                    .all(|value| *value >= 1.0)
            );
        }
    }
}

pub use map_sandbox::{
    AdjacencySpec, HydrologyField, HydrologyFlowDirection, HydrologyTile, HydrologyTileLayer,
    MapArtifact, MapArtifactMetadata, MapGenerationError, MapGeneratorKind, RuleBasedMapGenerator,
    ScalarField, TileSpec, TilesetSpec,
};

/// Runtime hydrology state tracked by the world.
#[derive(Debug, Clone)]
pub struct HydrologyState {
    tiles: HydrologyTileLayer,
    field: HydrologyField,
    water_depth: Vec<f32>,
}

impl HydrologyState {
    pub fn new(tiles: HydrologyTileLayer, field: HydrologyField) -> Self {
        let len = (tiles.width() as usize) * (tiles.height() as usize);
        let mut water_depth = Vec::with_capacity(len);
        water_depth.extend_from_slice(field.initial_water_depth());
        Self {
            tiles,
            field,
            water_depth,
        }
    }

    pub fn tiles(&self) -> &HydrologyTileLayer {
        &self.tiles
    }

    pub fn field(&self) -> &HydrologyField {
        &self.field
    }

    pub fn width(&self) -> u32 {
        self.tiles.width()
    }

    pub fn height(&self) -> u32 {
        self.tiles.height()
    }

    pub fn cell_count(&self) -> usize {
        self.water_depth.len()
    }

    pub fn water_depth(&self) -> &[f32] {
        &self.water_depth
    }

    pub fn water_depth_mut(&mut self) -> &mut [f32] {
        &mut self.water_depth
    }

    pub fn total_water_depth(&self) -> f32 {
        self.water_depth.iter().sum()
    }

    pub fn flooded_cell_counts(
        &self,
        shallow_threshold: f32,
        deep_threshold: f32,
    ) -> (usize, usize) {
        let mut shallow = 0usize;
        let mut deep = 0usize;
        for depth in &self.water_depth {
            if *depth >= shallow_threshold {
                shallow += 1;
            }
            if *depth >= deep_threshold {
                deep += 1;
            }
        }
        (shallow, deep)
    }
}

#[derive(Debug, Clone, Copy)]
struct FoodCellProfile {
    capacity: f32,
    growth_multiplier: f32,
    decay_multiplier: f32,
    fertility: f32,
    nutrient_density: f32,
}

/// Public snapshot of derived food cell parameters.
#[derive(Debug, Clone, Copy)]
pub struct FoodCellProfileSnapshot {
    pub capacity: f32,
    pub growth_multiplier: f32,
    pub decay_multiplier: f32,
    pub fertility: f32,
    pub nutrient_density: f32,
}

impl From<&FoodCellProfile> for FoodCellProfileSnapshot {
    fn from(profile: &FoodCellProfile) -> Self {
        Self {
            capacity: profile.capacity,
            growth_multiplier: profile.growth_multiplier,
            decay_multiplier: profile.decay_multiplier,
            fertility: profile.fertility,
            nutrient_density: profile.nutrient_density,
        }
    }
}

impl FoodCellProfile {
    fn compute(config: &ScriptBotsConfig, terrain: &TerrainLayer) -> Vec<FoodCellProfile> {
        let width = terrain.width() as usize;
        let height = terrain.height() as usize;
        if width == 0 || height == 0 {
            return Vec::new();
        }

        let mut profiles = Vec::with_capacity(width * height);
        let cell_size = config.food_cell_size as f32;
        let base = config.food_fertility_base;
        let moisture_weight = config.food_moisture_weight;
        let elevation_weight = config.food_elevation_weight;
        let slope_weight = config.food_slope_weight;
        let cap_base = config.food_capacity_base;
        let cap_fertility = config.food_capacity_fertility;
        let growth_scale = config.food_growth_fertility;
        let decay_scale = config.food_decay_infertility;

        for y in 0..height {
            for x in 0..width {
                let tile = terrain
                    .tile(x as u32, y as u32)
                    .expect("terrain tile should exist");
                let kind_bonus = terrain_kind_fertility_bonus(tile.kind);
                let moisture_term = tile.moisture * moisture_weight;
                let elevation_term = tile.elevation * elevation_weight;
                let world_x = (x as f32 + 0.5) * cell_size;
                let world_y = (y as f32 + 0.5) * cell_size;
                let (grad_x, grad_y) = terrain.gradient_world(world_x, world_y, cell_size);
                let slope = (grad_x * grad_x + grad_y * grad_y).sqrt();
                let slope_term = slope * slope_weight;
                let fertility_raw = base + kind_bonus + moisture_term - elevation_term - slope_term;
                let fertility = fertility_raw.clamp(0.0, 1.0);
                let capacity_factor = (cap_base + fertility * cap_fertility).clamp(0.05, 1.0);
                let growth_multiplier = (0.5 + fertility * growth_scale).clamp(0.1, 5.0);
                let decay_multiplier = (1.0 + (1.0 - fertility) * decay_scale).max(0.0);
                let nutrient_density = (0.3 + fertility * 0.7).clamp(0.0, 1.0);

                profiles.push(FoodCellProfile {
                    capacity: config.food_max * capacity_factor,
                    growth_multiplier,
                    decay_multiplier,
                    fertility,
                    nutrient_density,
                });
            }
        }

        profiles
    }
}

fn terrain_kind_fertility_bonus(kind: TerrainKind) -> f32 {
    match kind {
        TerrainKind::Bloom => 0.35,
        TerrainKind::Grass => 0.2,
        TerrainKind::Sand => -0.25,
        TerrainKind::ShallowWater => -0.2,
        TerrainKind::DeepWater => -0.8,
        TerrainKind::Rock => -0.45,
    }
}

#[derive(Debug, Clone)]
struct TickCadence {
    aging_interval: u32,
    chart_interval: u32,
    reproduction_interval: u32,
    reproduction_chance: f32,
}

impl TickCadence {
    fn from_config(config: &ScriptBotsConfig) -> Self {
        Self {
            aging_interval: config.aging_tick_interval.max(1),
            chart_interval: config.chart_flush_interval,
            reproduction_interval: config.reproduction_attempt_interval,
            reproduction_chance: config.reproduction_attempt_chance.clamp(0.0, 1.0),
        }
    }

    fn should_age(&self, tick: Tick) -> bool {
        self.aging_interval > 0 && tick.0.is_multiple_of(self.aging_interval as u64)
    }

    fn should_emit_chart_event(&self, tick: Tick) -> bool {
        self.chart_interval > 0 && tick.0.is_multiple_of(self.chart_interval as u64)
    }

    fn reproduction_window(&self, tick: Tick) -> bool {
        self.reproduction_interval == 0 || tick.0.is_multiple_of(self.reproduction_interval as u64)
    }

    fn reproduction_chance(&self) -> f32 {
        self.reproduction_chance
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
    food_profiles: Vec<FoodCellProfile>,
    terrain: TerrainLayer,
    map_metadata: Option<MapArtifactMetadata>,
    hydrology: Option<HydrologyState>,
    runtime: AgentMap<AgentRuntime>,
    index: UniformGridIndex,
    brain_registry: BrainRegistry,
    cadence: TickCadence,
    food_scratch: Vec<f32>,
    // Reusable per-tick working buffers to avoid allocations
    work_handles: Vec<AgentId>,
    work_position_pairs: Vec<(f32, f32)>,
    work_trait_modifiers: Vec<TraitModifiers>,
    work_eye_directions: Vec<[f32; NUM_EYES]>,
    work_eye_fov: Vec<[f32; NUM_EYES]>,
    work_eye_view_dirs: Vec<[f32; NUM_EYES]>,
    work_eye_fov_clamped: Vec<[f32; NUM_EYES]>,
    work_eye_fov_cos: Vec<[f32; NUM_EYES]>,
    work_clocks: Vec<[f32; 2]>,
    work_temperature_preferences: Vec<f32>,
    work_sound_emitters: Vec<f32>,
    work_positions: Vec<Position>,
    work_headings: Vec<f32>,
    work_heading_dir_x: Vec<f32>,
    work_heading_dir_y: Vec<f32>,
    work_spike_lengths: Vec<f32>,
    work_velocities: Vec<Velocity>,
    work_speed_norm: Vec<f32>,
    work_runtime_snapshot: Vec<AgentRuntime>,
    work_penalties: Vec<f32>,
    pending_deaths: Vec<AgentId>,
    #[allow(dead_code)]
    pending_spawns: Vec<SpawnOrder>,
    pending_birth_records: Vec<BirthRecord>,
    pending_death_records: Vec<DeathRecord>,
    #[allow(dead_code)]
    replay_tick: u64,
    replay_events: Vec<ReplayEvent>,
    persistence: Box<dyn WorldPersistence>,
    last_births: usize,
    last_deaths: usize,
    last_spike_hits: u32,
    last_max_age: u32,
    history: VecDeque<TickSummary>,
    #[allow(dead_code)]
    carcass_health_distributed: f32,
    #[allow(dead_code)]
    carcass_reproduction_bonus: f32,
    combat_spike_attempts: u32,
    combat_spike_hits: u32,
    config_audit: Vec<ConfigAuditEntry>,
}

impl fmt::Debug for WorldState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorldState")
            .field("config", &self.config)
            .field("tick", &self.tick)
            .field("epoch", &self.epoch)
            .field("closed", &self.closed)
            .field("agent_count", &self.agents.len())
            .field("food_profiles", &self.food_profiles.len())
            .field(
                "map_metadata",
                &self
                    .map_metadata
                    .as_ref()
                    .map(|meta| meta.tileset_id.as_str()),
            )
            .field(
                "hydrology",
                &self.hydrology.as_ref().map(|state| state.tiles().width()),
            )
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
        configure_parallelism();
        let (food_w, food_h) = config.food_dimensions()?;
        let rng = config.seeded_rng();
        let mut terrain_rng = rng.clone();
        let terrain =
            TerrainLayer::generate(food_w, food_h, config.food_cell_size, &mut terrain_rng)?;
        let food = FoodGrid::new(food_w, food_h, config.initial_food)?;
        let food_profiles = FoodCellProfile::compute(&config, &terrain);
        let index = UniformGridIndex::new(
            config.food_cell_size as f32,
            config.world_width as f32,
            config.world_height as f32,
        );
        let history_capacity = config.history_capacity;
        let cadence = TickCadence::from_config(&config);
        Ok(Self {
            food,
            terrain,
            map_metadata: None,
            hydrology: None,
            config,
            tick: Tick::zero(),
            epoch: 0,
            closed: false,
            rng,
            agents: AgentArena::new(),
            runtime: AgentMap::new(),
            index,
            brain_registry: BrainRegistry::new(),
            cadence,
            food_profiles,
            food_scratch: vec![0.0; (food_w as usize) * (food_h as usize)],
            work_handles: Vec::new(),
            work_position_pairs: Vec::new(),
            work_trait_modifiers: Vec::new(),
            work_eye_directions: Vec::new(),
            work_eye_fov: Vec::new(),
            work_eye_view_dirs: Vec::new(),
            work_eye_fov_clamped: Vec::new(),
            work_eye_fov_cos: Vec::new(),
            work_clocks: Vec::new(),
            work_temperature_preferences: Vec::new(),
            work_sound_emitters: Vec::new(),
            work_positions: Vec::new(),
            work_headings: Vec::new(),
            work_heading_dir_x: Vec::new(),
            work_heading_dir_y: Vec::new(),
            work_spike_lengths: Vec::new(),
            work_velocities: Vec::new(),
            work_speed_norm: Vec::new(),
            work_runtime_snapshot: Vec::new(),
            work_penalties: Vec::new(),
            pending_deaths: Vec::new(),
            pending_spawns: Vec::new(),
            pending_birth_records: Vec::new(),
            pending_death_records: Vec::new(),
            replay_tick: 0,
            replay_events: Vec::new(),
            persistence,
            last_births: 0,
            last_deaths: 0,
            last_spike_hits: 0,
            last_max_age: 0,
            history: VecDeque::with_capacity(history_capacity),
            carcass_health_distributed: 0.0,
            carcass_reproduction_bonus: 0.0,
            combat_spike_attempts: 0,
            combat_spike_hits: 0,
            config_audit: Vec::with_capacity(32),
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
            if energy_scale <= 0.0 {
                // no additional energy penalty
            } else if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                let energy_penalty = penalty * energy_scale;
                runtime.energy = (runtime.energy - energy_penalty).max(0.0);
                runtime.food_delta -= energy_penalty;
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
        let idx = (y as usize) * (width as usize) + x as usize;
        let capacity = self
            .food_profiles
            .get(idx)
            .map_or(self.config.food_max, |profile| profile.capacity);
        if let Some(cell) = self.food.get_mut(x, y) {
            *cell = (*cell + self.config.food_respawn_amount).min(capacity);
            Some((x, y))
        } else {
            None
        }
    }

    fn stage_food_dynamics(&mut self, next_tick: Tick) -> Option<(u32, u32)> {
        let respawned = self.stage_food_respawn(next_tick);
        self.apply_food_regrowth();
        if let Some((x, y)) = respawned {
            let width = self.food.width() as usize;
            let idx = (y as usize) * width + x as usize;
            let capacity = self
                .food_profiles
                .get(idx)
                .map_or(self.config.food_max, |profile| profile.capacity);
            if let Some(cell) = self.food.get_mut(x, y) {
                *cell = (*cell).min(capacity);
            }
        }
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

        let previous = &self.food_scratch;
        let profiles = &self.food_profiles;
        let food_max = self.config.food_max;
        let cells_mut = self.food.cells_mut();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            cells_mut
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(y, row)| {
                    let up_row = if y == 0 { height - 1 } else { y - 1 };
                    let down_row = if y + 1 == height { 0 } else { y + 1 };
                    #[cfg(feature = "simd_wide")]
                    {
                        use wide::f32x4;
                        let mut x = 0usize;
                        while x + 3 < width {
                            let xs = [x, x + 1, x + 2, x + 3];
                            let lefts = [
                                if xs[0] == 0 { width - 1 } else { xs[0] - 1 },
                                if xs[1] == 0 { width - 1 } else { xs[1] - 1 },
                                if xs[2] == 0 { width - 1 } else { xs[2] - 1 },
                                if xs[3] == 0 { width - 1 } else { xs[3] - 1 },
                            ];
                            let rights = [
                                if xs[0] + 1 == width { 0 } else { xs[0] + 1 },
                                if xs[1] + 1 == width { 0 } else { xs[1] + 1 },
                                if xs[2] + 1 == width { 0 } else { xs[2] + 1 },
                                if xs[3] + 1 == width { 0 } else { xs[3] + 1 },
                            ];
                            let idxs = [
                                y * width + xs[0],
                                y * width + xs[1],
                                y * width + xs[2],
                                y * width + xs[3],
                            ];
                            let prev_v = f32x4::new([
                                previous[idxs[0]],
                                previous[idxs[1]],
                                previous[idxs[2]],
                                previous[idxs[3]],
                            ]);
                            let left_v = f32x4::new([
                                previous[y * width + lefts[0]],
                                previous[y * width + lefts[1]],
                                previous[y * width + lefts[2]],
                                previous[y * width + lefts[3]],
                            ]);
                            let right_v = f32x4::new([
                                previous[y * width + rights[0]],
                                previous[y * width + rights[1]],
                                previous[y * width + rights[2]],
                                previous[y * width + rights[3]],
                            ]);
                            let up_v = f32x4::new([
                                previous[up_row * width + xs[0]],
                                previous[up_row * width + xs[1]],
                                previous[up_row * width + xs[2]],
                                previous[up_row * width + xs[3]],
                            ]);
                            let down_v = f32x4::new([
                                previous[down_row * width + xs[0]],
                                previous[down_row * width + xs[1]],
                                previous[down_row * width + xs[2]],
                                previous[down_row * width + xs[3]],
                            ]);
                            let mut val_v = prev_v;
                            if diffusion > 0.0 {
                                let neigh = (left_v + right_v + up_v + down_v) * f32x4::splat(0.25);
                                val_v = val_v + f32x4::splat(diffusion) * (neigh - prev_v);
                            }
                            let cap_arr = idxs.map(|i| {
                                profiles
                                    .get(i)
                                    .copied()
                                    .unwrap_or(FoodCellProfile {
                                        capacity: food_max,
                                        growth_multiplier: 1.0,
                                        decay_multiplier: 1.0,
                                        fertility: 0.0,
                                        nutrient_density: 0.3,
                                    })
                                    .capacity
                            });
                            let grow_arr = idxs.map(|i| {
                                profiles
                                    .get(i)
                                    .copied()
                                    .unwrap_or(FoodCellProfile {
                                        capacity: food_max,
                                        growth_multiplier: 1.0,
                                        decay_multiplier: 1.0,
                                        fertility: 0.0,
                                        nutrient_density: 0.3,
                                    })
                                    .growth_multiplier
                            });
                            let decay_arr = idxs.map(|i| {
                                profiles
                                    .get(i)
                                    .copied()
                                    .unwrap_or(FoodCellProfile {
                                        capacity: food_max,
                                        growth_multiplier: 1.0,
                                        decay_multiplier: 1.0,
                                        fertility: 0.0,
                                        nutrient_density: 0.3,
                                    })
                                    .decay_multiplier
                            });
                            let cap_v = f32x4::new(cap_arr);
                            let grow_v = f32x4::new(grow_arr);
                            let decay_v = f32x4::new(decay_arr);
                            if decay > 0.0 {
                                val_v = val_v - f32x4::splat(decay) * decay_v * val_v;
                            }
                            if growth > 0.0 && food_max > 0.0 {
                                let norm = val_v / f32x4::splat(food_max);
                                let delta =
                                    f32x4::splat(growth) * grow_v * (f32x4::splat(1.0) - norm);
                                val_v = val_v + delta * f32x4::splat(food_max);
                            }
                            // Clamp to capacity and global cap
                            let prev_cap_v = prev_v; // previous_value for max with capacity floor
                            let mut cap_eff_v = cap_v.max(prev_cap_v);
                            let global_cap_v = f32x4::splat(food_max).max(prev_cap_v);
                            // min(capacity, global_cap)
                            cap_eff_v = cap_eff_v.min(global_cap_v).max(f32x4::splat(0.0));
                            let out_v = val_v.max(f32x4::splat(0.0)).min(cap_eff_v);
                            let out_arr = out_v.to_array();
                            row[x + 0] = out_arr[0];
                            row[x + 1] = out_arr[1];
                            row[x + 2] = out_arr[2];
                            row[x + 3] = out_arr[3];
                            x += 4;
                        }
                        // Remainder scalar
                        for x in x..width {
                            let left_col = if x == 0 { width - 1 } else { x - 1 };
                            let right_col = if x + 1 == width { 0 } else { x + 1 };
                            let idx = y * width + x;
                            let previous_value = previous[idx];
                            let mut value = previous_value;
                            let profile = profiles.get(idx).copied().unwrap_or(FoodCellProfile {
                                capacity: food_max,
                                growth_multiplier: 1.0,
                                decay_multiplier: 1.0,
                                fertility: 0.0,
                                nutrient_density: 0.3,
                            });
                            if diffusion > 0.0 {
                                let left = previous[y * width + left_col];
                                let right = previous[y * width + right_col];
                                let up = previous[up_row * width + x];
                                let down = previous[down_row * width + x];
                                let neighbor_avg = (left + right + up + down) * 0.25;
                                value += diffusion * (neighbor_avg - previous_value);
                            }
                            if decay > 0.0 {
                                value -= decay * profile.decay_multiplier * value;
                            }
                            if growth > 0.0 && food_max > 0.0 {
                                let normalized = value / food_max;
                                let growth_delta =
                                    growth * profile.growth_multiplier * (1.0 - normalized);
                                value += growth_delta * food_max;
                            }
                            let mut capacity = profile.capacity.max(previous_value);
                            let global_cap = food_max.max(previous_value);
                            if capacity > global_cap {
                                capacity = global_cap;
                            }
                            capacity = capacity.max(0.0);
                            row[x] = value.clamp(0.0, capacity);
                        }
                    }
                    #[cfg(not(feature = "simd_wide"))]
                    for x in 0..width {
                        let left_col = if x == 0 { width - 1 } else { x - 1 };
                        let right_col = if x + 1 == width { 0 } else { x + 1 };
                        let idx = y * width + x;
                        let previous_value = previous[idx];
                        let mut value = previous_value;
                        let profile = profiles.get(idx).copied().unwrap_or(FoodCellProfile {
                            capacity: food_max,
                            growth_multiplier: 1.0,
                            decay_multiplier: 1.0,
                            fertility: 0.0,
                            nutrient_density: 0.3,
                        });
                        if diffusion > 0.0 {
                            let left = previous[y * width + left_col];
                            let right = previous[y * width + right_col];
                            let up = previous[up_row * width + x];
                            let down = previous[down_row * width + x];
                            let neighbor_avg = (left + right + up + down) * 0.25;
                            value += diffusion * (neighbor_avg - previous_value);
                        }

                        if decay > 0.0 {
                            value -= decay * profile.decay_multiplier * value;
                        }

                        if growth > 0.0 && food_max > 0.0 {
                            let normalized = value / food_max;
                            let growth_delta =
                                growth * profile.growth_multiplier * (1.0 - normalized);
                            value += growth_delta * food_max;
                        }

                        let mut capacity = profile.capacity.max(previous_value);
                        let global_cap = food_max.max(previous_value);
                        if capacity > global_cap {
                            capacity = global_cap;
                        }
                        capacity = capacity.max(0.0);
                        row[x] = value.clamp(0.0, capacity);
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for y in 0..height {
                let up_row = if y == 0 { height - 1 } else { y - 1 };
                let down_row = if y + 1 == height { 0 } else { y + 1 };
                for x in 0..width {
                    let left_col = if x == 0 { width - 1 } else { x - 1 };
                    let right_col = if x + 1 == width { 0 } else { x + 1 };
                    let idx = y * width + x;
                    let previous_value = previous[idx];
                    let mut value = previous_value;
                    let profile = profiles.get(idx).copied().unwrap_or(FoodCellProfile {
                        capacity: food_max,
                        growth_multiplier: 1.0,
                        decay_multiplier: 1.0,
                        fertility: 0.0,
                        nutrient_density: 0.3,
                    });
                    if diffusion > 0.0 {
                        let left = previous[y * width + left_col];
                        let right = previous[y * width + right_col];
                        let up = previous[up_row * width + x];
                        let down = previous[down_row * width + x];
                        let neighbor_avg = (left + right + up + down) * 0.25;
                        value += diffusion * (neighbor_avg - previous_value);
                    }

                    if decay > 0.0 {
                        value -= decay * profile.decay_multiplier * value;
                    }

                    if growth > 0.0 && food_max > 0.0 {
                        let normalized = value / food_max;
                        let growth_delta = growth * profile.growth_multiplier * (1.0 - normalized);
                        value += growth_delta * food_max;
                    }

                    let mut capacity = profile.capacity.max(previous_value);
                    let global_cap = food_max.max(previous_value);
                    if capacity > global_cap {
                        capacity = global_cap;
                    }
                    capacity = capacity.max(0.0);
                    cells_mut[idx] = value.clamp(0.0, capacity);
                }
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

        // Build and reuse position pairs buffer
        self.work_position_pairs.clear();
        self.work_position_pairs.reserve(positions.len());
        for p in positions.iter() {
            self.work_position_pairs.push((p.x, p.y));
        }
        if self.index.rebuild(&self.work_position_pairs).is_err() {
            return;
        }

        // Build and reuse handles buffer
        self.work_handles.clear();
        self.work_handles.reserve(agent_count);
        self.work_handles.extend(self.agents.iter_handles());
        let handles = &self.work_handles;
        let runtime = &self.runtime;

        // Populate reusable runtime-derived SoA buffers
        self.work_trait_modifiers
            .resize(agent_count, TraitModifiers::default());
        self.work_eye_directions
            .resize(agent_count, [0.0; NUM_EYES]);
        self.work_eye_fov.resize(agent_count, [1.0; NUM_EYES]);
        self.work_eye_view_dirs.resize(agent_count, [0.0; NUM_EYES]);
        self.work_eye_fov_clamped
            .resize(agent_count, [1.0; NUM_EYES]);
        self.work_eye_fov_cos.resize(agent_count, [0.0; NUM_EYES]);
        self.work_clocks.resize(agent_count, [50.0, 50.0]);
        self.work_temperature_preferences.resize(agent_count, 0.5);
        self.work_sound_emitters.resize(agent_count, 0.0);
        self.work_speed_norm.resize(agent_count, 0.0);
        for (idx, id) in handles.iter().enumerate() {
            if let Some(rt) = runtime.get(*id) {
                self.work_trait_modifiers[idx] = rt.trait_modifiers;
                self.work_eye_directions[idx] = rt.eye_direction;
                self.work_eye_fov[idx] = rt.eye_fov;
                // Precompute per-eye view directions and clamped FOV once per agent
                let mut views = [0.0; NUM_EYES];
                let mut fovc = [1.0; NUM_EYES];
                let mut fovcos = [0.0; NUM_EYES];
                let base_heading = headings[idx];
                for e in 0..NUM_EYES {
                    views[e] = wrap_signed_angle(base_heading + rt.eye_direction[e]);
                    fovc[e] = rt.eye_fov[e].max(0.01);
                    fovcos[e] = fovc[e].cos();
                }
                self.work_eye_view_dirs[idx] = views;
                self.work_eye_fov_clamped[idx] = fovc;
                self.work_eye_fov_cos[idx] = fovcos;
                self.work_clocks[idx] = rt.clocks;
                self.work_temperature_preferences[idx] = rt.temperature_preference;
                self.work_sound_emitters[idx] = rt.sound_multiplier;
            }
        }
        let trait_modifiers = &self.work_trait_modifiers;
        let eye_directions = &self.work_eye_directions;
        let eye_fov = &self.work_eye_fov;
        let clocks = &self.work_clocks;
        let temperature_preferences = &self.work_temperature_preferences;
        let sound_emitters = &self.work_sound_emitters;

        // Sanity checks (debug-only) to validate buffers are well-formed
        debug_assert!(eye_directions.len() == handles.len());
        debug_assert!(eye_fov.len() == handles.len());
        debug_assert!(clocks.len() == handles.len());
        debug_assert!(temperature_preferences.len() == handles.len());
        debug_assert!(sound_emitters.len() == handles.len());
        debug_assert!({
            // Ensure FOV and directions contain finite values
            let mut ok = true;
            for dir in eye_directions.iter() {
                for &d in dir.iter() {
                    if !d.is_finite() {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }
            }
            for fovs in eye_fov.iter() {
                for &f in fovs.iter() {
                    if !f.is_finite() {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }
            }
            ok
        });

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
        // Precompute normalized speed per agent for sound channel
        for (idx, vel) in velocities.iter().enumerate() {
            let sp = (vel.vx * vel.vx + vel.vy * vel.vy).sqrt();
            self.work_speed_norm[idx] = (sp / max_speed).clamp(0.0, 1.0);
        }
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
            let hx = heading.cos();
            let hy = heading.sin();
            let cos_bhf = (BLOOD_HALF_FOV).cos();
            let traits = trait_modifiers[idx];
            let eyes_dir = &self.work_eye_view_dirs[idx];
            let eyes_fov = &self.work_eye_fov_clamped[idx];
            let eyes_fov_cos = &self.work_eye_fov_cos[idx];

            index.visit_neighbor_buckets(idx, radius, &mut |indices| {
                #[cfg(feature = "simd_wide")]
                {
                    // SIMD-batch smell/sound/hearing; eyes/blood remain per-lane for correctness
                    for chunk in indices.chunks_exact(4) {
                        let ids = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        let dx_arr = [
                            toroidal_delta(positions[ids[0]].x, position.x, world_width),
                            toroidal_delta(positions[ids[1]].x, position.x, world_width),
                            toroidal_delta(positions[ids[2]].x, position.x, world_width),
                            toroidal_delta(positions[ids[3]].x, position.x, world_width),
                        ];
                        let dy_arr = [
                            toroidal_delta(positions[ids[0]].y, position.y, world_height),
                            toroidal_delta(positions[ids[1]].y, position.y, world_height),
                            toroidal_delta(positions[ids[2]].y, position.y, world_height),
                            toroidal_delta(positions[ids[3]].y, position.y, world_height),
                        ];
                        let dx_v = f32x4::new(dx_arr);
                        let dy_v = f32x4::new(dy_arr);
                        let dist_sq_v = dx_v * dx_v + dy_v * dy_v;
                        let dist_v = dist_sq_v.sqrt();
                        let mut df_v = (f32x4::splat(radius) - dist_v) / f32x4::splat(radius);
                        df_v = df_v.max(f32x4::splat(0.0));
                        // Zero out invalid lanes (self, <= eps, > radius^2)
                        let dsq = dist_sq_v.to_array();
                        let mut df = df_v.to_array();
                        for lane in 0..4 {
                            let oid = ids[lane];
                            if oid == idx || dsq[lane] <= f32::EPSILON || dsq[lane] > radius_sq {
                                df[lane] = 0.0;
                            }
                        }
                        let df_v = f32x4::new(df);
                        // Smell accumulation
                        smell += df.iter().copied().sum::<f32>();
                        // Sound accumulation
                        let sp = f32x4::new([
                            self.work_speed_norm[ids[0]],
                            self.work_speed_norm[ids[1]],
                            self.work_speed_norm[ids[2]],
                            self.work_speed_norm[ids[3]],
                        ]);
                        sound += (df_v * sp).to_array().iter().copied().sum::<f32>();
                        // Hearing accumulation
                        let em = f32x4::new([
                            sound_emitters[ids[0]],
                            sound_emitters[ids[1]],
                            sound_emitters[ids[2]],
                            sound_emitters[ids[3]],
                        ]);
                        hearing += (df_v * em).to_array().iter().copied().sum::<f32>();

                        // Eyes and blood per-lane for these four
                        let dist_arr = dist_v.to_array();
                        for lane in 0..4 {
                            let other_idx = ids[lane];
                            if df[lane] <= 0.0 {
                                continue;
                            }
                            let dx = dx_arr[lane];
                            let dy = dy_arr[lane];
                            let dist = dist_arr[lane];
                            let dist_factor = (radius - dist) / radius;
                            // Neighbor unit dir
                            let nx = dx / dist;
                            let ny = dy / dist;
                            #[cfg(feature = "simd_wide")]
                            {
                                // Dot against per-eye view directions; threshold by cos(FOV)
                                let eye_dirs_x = [
                                    eyes_dir[0].cos(),
                                    eyes_dir[1].cos(),
                                    eyes_dir[2].cos(),
                                    eyes_dir[3].cos(),
                                ];
                                let eye_dirs_y = [
                                    eyes_dir[0].sin(),
                                    eyes_dir[1].sin(),
                                    eyes_dir[2].sin(),
                                    eyes_dir[3].sin(),
                                ];
                                let dot_v = f32x4::new([
                                    eye_dirs_x[0] * nx + eye_dirs_y[0] * ny,
                                    eye_dirs_x[1] * nx + eye_dirs_y[1] * ny,
                                    eye_dirs_x[2] * nx + eye_dirs_y[2] * ny,
                                    eye_dirs_x[3] * nx + eye_dirs_y[3] * ny,
                                ]);
                                let cos_fov_v = f32x4::new([
                                    eyes_fov_cos[0],
                                    eyes_fov_cos[1],
                                    eyes_fov_cos[2],
                                    eyes_fov_cos[3],
                                ]);
                                let mask_v = f32x4::new([
                                    (dot_v.to_array()[0] >= cos_fov_v.to_array()[0]) as i32 as f32,
                                    (dot_v.to_array()[1] >= cos_fov_v.to_array()[1]) as i32 as f32,
                                    (dot_v.to_array()[2] >= cos_fov_v.to_array()[2]) as i32 as f32,
                                    (dot_v.to_array()[3] >= cos_fov_v.to_array()[3]) as i32 as f32,
                                ]);
                                // fov_factor ~ (cos_fov - dot)/cos_fov, clamped to [0,1]
                                let fov_factor =
                                    ((cos_fov_v - dot_v) / cos_fov_v).max(f32x4::splat(0.0));
                                let intensity_v = fov_factor
                                    * f32x4::splat(traits.eye * dist_factor * (dist / radius))
                                    * mask_v;
                                let color = colors[other_idx];
                                let mut dens =
                                    f32x4::new([density[0], density[1], density[2], density[3]]);
                                let mut r = f32x4::new([eye_r[0], eye_r[1], eye_r[2], eye_r[3]]);
                                let mut g = f32x4::new([eye_g[0], eye_g[1], eye_g[2], eye_g[3]]);
                                let mut b = f32x4::new([eye_b[0], eye_b[1], eye_b[2], eye_b[3]]);
                                dens = dens + intensity_v;
                                r = r + intensity_v * f32x4::splat(color[0]);
                                g = g + intensity_v * f32x4::splat(color[1]);
                                b = b + intensity_v * f32x4::splat(color[2]);
                                let out_d = dens.to_array();
                                let out_r = r.to_array();
                                let out_g = g.to_array();
                                let out_b = b.to_array();
                                density[0] = out_d[0];
                                density[1] = out_d[1];
                                density[2] = out_d[2];
                                density[3] = out_d[3];
                                eye_r[0] = out_r[0];
                                eye_r[1] = out_r[1];
                                eye_r[2] = out_r[2];
                                eye_r[3] = out_r[3];
                                eye_g[0] = out_g[0];
                                eye_g[1] = out_g[1];
                                eye_g[2] = out_g[2];
                                eye_g[3] = out_g[3];
                                eye_b[0] = out_b[0];
                                eye_b[1] = out_b[1];
                                eye_b[2] = out_b[2];
                                eye_b[3] = out_b[3];
                            }
                            #[cfg(not(feature = "simd_wide"))]
                            {
                                for eye in 0..NUM_EYES {
                                    // Dot mask vs. cos(FOV)
                                    let vx = eyes_dir[eye].cos();
                                    let vy = eyes_dir[eye].sin();
                                    let dot = vx * nx + vy * ny;
                                    if dot >= eyes_fov_cos[eye] {
                                        // approximate fov_factor via dot/cos_fov
                                        let fov = eyes_fov[eye];
                                        let diff =
                                            angle_difference(eyes_dir[eye], angle_to(dx, dy));
                                        let fov_factor = ((fov - diff) / fov).max(0.0);
                                        let intensity =
                                            traits.eye * fov_factor * dist_factor * (dist / radius);
                                        density[eye] += intensity;
                                        let color = colors[other_idx];
                                        eye_r[eye] += intensity * color[0];
                                        eye_g[eye] += intensity * color[1];
                                        eye_b[eye] += intensity * color[2];
                                    }
                                }
                            }
                            // Blood via dot threshold to prune; magnitude via angle diff
                            let align = hx * nx + hy * ny;
                            if align >= cos_bhf {
                                let ang = angle_to(dx, dy);
                                let forward_diff = angle_difference(heading, ang);
                                let bleed = (BLOOD_HALF_FOV - forward_diff) / BLOOD_HALF_FOV;
                                let health = healths[other_idx];
                                let wound = (1.0 - (health * 0.5).clamp(0.0, 1.0)).max(0.0);
                                blood += bleed * dist_factor * wound;
                            }
                        }
                    }
                    // Remainder (less than 4)
                    for &other_idx in indices.chunks_exact(4).remainder() {
                        if other_idx == idx {
                            continue;
                        }
                        let dx = toroidal_delta(positions[other_idx].x, position.x, world_width);
                        let dy = toroidal_delta(positions[other_idx].y, position.y, world_height);
                        let dist_sq_val = dx.mul_add(dx, dy * dy);
                        if dist_sq_val <= f32::EPSILON || dist_sq_val > radius_sq {
                            continue;
                        }
                        let dist = dist_sq_val.sqrt();
                        let ang = angle_to(dx, dy);
                        let dist_factor = (radius - dist) / radius;
                        if dist_factor <= 0.0 {
                            continue;
                        }
                        smell += dist_factor;
                        sound += dist_factor * self.work_speed_norm[other_idx];
                        hearing += dist_factor * sound_emitters[other_idx];
                        #[cfg(feature = "simd_wide")]
                        {
                            let base = [eyes_dir[0], eyes_dir[1], eyes_dir[2], eyes_dir[3]];
                            let fov = [eyes_fov[0], eyes_fov[1], eyes_fov[2], eyes_fov[3]];
                            let diff = [
                                angle_difference(base[0], ang),
                                angle_difference(base[1], ang),
                                angle_difference(base[2], ang),
                                angle_difference(base[3], ang),
                            ];
                            let diff_v = f32x4::new(diff);
                            let fov_v = f32x4::new(fov);
                            let mut fov_factor = (fov_v - diff_v) / fov_v;
                            fov_factor = fov_factor.max(f32x4::splat(0.0));
                            let scalar = traits.eye * dist_factor * (dist / radius);
                            let intensity_v = fov_factor * f32x4::splat(scalar);
                            let color = colors[other_idx];
                            let mut dens =
                                f32x4::new([density[0], density[1], density[2], density[3]]);
                            let mut r = f32x4::new([eye_r[0], eye_r[1], eye_r[2], eye_r[3]]);
                            let mut g = f32x4::new([eye_g[0], eye_g[1], eye_g[2], eye_g[3]]);
                            let mut b = f32x4::new([eye_b[0], eye_b[1], eye_b[2], eye_b[3]]);
                            dens = dens + intensity_v;
                            r = r + intensity_v * f32x4::splat(color[0]);
                            g = g + intensity_v * f32x4::splat(color[1]);
                            b = b + intensity_v * f32x4::splat(color[2]);
                            let out_d = dens.to_array();
                            let out_r = r.to_array();
                            let out_g = g.to_array();
                            let out_b = b.to_array();
                            density[0] = out_d[0];
                            density[1] = out_d[1];
                            density[2] = out_d[2];
                            density[3] = out_d[3];
                            eye_r[0] = out_r[0];
                            eye_r[1] = out_r[1];
                            eye_r[2] = out_r[2];
                            eye_r[3] = out_r[3];
                            eye_g[0] = out_g[0];
                            eye_g[1] = out_g[1];
                            eye_g[2] = out_g[2];
                            eye_g[3] = out_g[3];
                            eye_b[0] = out_b[0];
                            eye_b[1] = out_b[1];
                            eye_b[2] = out_b[2];
                            eye_b[3] = out_b[3];
                        }
                        #[cfg(not(feature = "simd_wide"))]
                        {
                            for eye in 0..NUM_EYES {
                                let diff = angle_difference(eyes_dir[eye], ang);
                                let fov = eyes_fov[eye];
                                if diff < fov {
                                    let fov_factor = ((fov - diff) / fov).max(0.0);
                                    let intensity =
                                        traits.eye * fov_factor * dist_factor * (dist / radius);
                                    density[eye] += intensity;
                                    let color = colors[other_idx];
                                    eye_r[eye] += intensity * color[0];
                                    eye_g[eye] += intensity * color[1];
                                    eye_b[eye] += intensity * color[2];
                                }
                            }
                        }
                        let forward_diff = angle_difference(heading, ang);
                        if forward_diff < BLOOD_HALF_FOV {
                            let bleed = (BLOOD_HALF_FOV - forward_diff) / BLOOD_HALF_FOV;
                            let health = healths[other_idx];
                            let wound = (1.0 - (health * 0.5).clamp(0.0, 1.0)).max(0.0);
                            blood += bleed * dist_factor * wound;
                        }
                    }
                    return;
                }
                #[cfg(not(feature = "simd_wide"))]
                for &other_idx in indices {
                    if other_idx == idx {
                        continue;
                    }
                    let dx = toroidal_delta(positions[other_idx].x, position.x, world_width);
                    let dy = toroidal_delta(positions[other_idx].y, position.y, world_height);
                    let dist_sq_val = dx.mul_add(dx, dy * dy);
                    if dist_sq_val <= f32::EPSILON {
                        continue;
                    }
                    if dist_sq_val > radius_sq {
                        continue;
                    }
                    let dist = dist_sq_val.sqrt();
                    let ang = angle_to(dx, dy);
                    let dist_factor = (radius - dist) / radius;
                    if dist_factor <= 0.0 {
                        continue;
                    }

                    #[cfg(feature = "simd_wide")]
                    {
                        // Compute neighbor unit direction
                        let nx = dx / dist;
                        let ny = dy / dist;
                        // Precompute eye unit vectors from view angles
                        let eye_dirs_x = [
                            eyes_dir[0].cos(),
                            eyes_dir[1].cos(),
                            eyes_dir[2].cos(),
                            eyes_dir[3].cos(),
                        ];
                        let eye_dirs_y = [
                            eyes_dir[0].sin(),
                            eyes_dir[1].sin(),
                            eyes_dir[2].sin(),
                            eyes_dir[3].sin(),
                        ];
                        let dot_v = f32x4::new([
                            eye_dirs_x[0] * nx + eye_dirs_y[0] * ny,
                            eye_dirs_x[1] * nx + eye_dirs_y[1] * ny,
                            eye_dirs_x[2] * nx + eye_dirs_y[2] * ny,
                            eye_dirs_x[3] * nx + eye_dirs_y[3] * ny,
                        ]);
                        let cos_fov_v = f32x4::new([
                            eyes_fov_cos[0],
                            eyes_fov_cos[1],
                            eyes_fov_cos[2],
                            eyes_fov_cos[3],
                        ]);
                        // mask = dot >= cos(fov)
                        let mask = [
                            (dot_v.to_array()[0] >= cos_fov_v.to_array()[0]) as i32 as f32,
                            (dot_v.to_array()[1] >= cos_fov_v.to_array()[1]) as i32 as f32,
                            (dot_v.to_array()[2] >= cos_fov_v.to_array()[2]) as i32 as f32,
                            (dot_v.to_array()[3] >= cos_fov_v.to_array()[3]) as i32 as f32,
                        ];
                        let mask_v = f32x4::new(mask);
                        // intensity = traits.eye * dist_factor * (dist/radius) * ((cos_fov - dot)/cos_fov) approximated by mask * (cos_fov - dot)/cos_fov
                        let fov_factor = (cos_fov_v - dot_v) / cos_fov_v;
                        let fov_factor = fov_factor.max(f32x4::splat(0.0));
                        let intensity_v = fov_factor
                            * f32x4::splat(traits.eye * dist_factor * (dist / radius))
                            * mask_v;
                        let color = colors[other_idx];
                        let mut dens = f32x4::new([density[0], density[1], density[2], density[3]]);
                        let mut r = f32x4::new([eye_r[0], eye_r[1], eye_r[2], eye_r[3]]);
                        let mut g = f32x4::new([eye_g[0], eye_g[1], eye_g[2], eye_g[3]]);
                        let mut b = f32x4::new([eye_b[0], eye_b[1], eye_b[2], eye_b[3]]);
                        dens = dens + intensity_v;
                        r = r + intensity_v * f32x4::splat(color[0]);
                        g = g + intensity_v * f32x4::splat(color[1]);
                        b = b + intensity_v * f32x4::splat(color[2]);
                        let out_d = dens.to_array();
                        let out_r = r.to_array();
                        let out_g = g.to_array();
                        let out_b = b.to_array();
                        density[0] = out_d[0];
                        density[1] = out_d[1];
                        density[2] = out_d[2];
                        density[3] = out_d[3];
                        eye_r[0] = out_r[0];
                        eye_r[1] = out_r[1];
                        eye_r[2] = out_r[2];
                        eye_r[3] = out_r[3];
                        eye_g[0] = out_g[0];
                        eye_g[1] = out_g[1];
                        eye_g[2] = out_g[2];
                        eye_g[3] = out_g[3];
                        eye_b[0] = out_b[0];
                        eye_b[1] = out_b[1];
                        eye_b[2] = out_b[2];
                        eye_b[3] = out_b[3];
                    }
                    #[cfg(not(feature = "simd_wide"))]
                    {
                        for eye in 0..NUM_EYES {
                            let view_dir = wrap_signed_angle(heading + eyes_dir[eye]);
                            let diff = angle_difference(view_dir, ang);
                            let fov = eyes_fov[eye].max(0.01);
                            if diff < fov {
                                let fov_factor = ((fov - diff) / fov).max(0.0);
                                let intensity =
                                    traits.eye * fov_factor * dist_factor * (dist / radius);
                                density[eye] += intensity;
                                let color = colors[other_idx];
                                eye_r[eye] += intensity * color[0];
                                eye_g[eye] += intensity * color[1];
                                eye_b[eye] += intensity * color[2];
                            }
                        }
                    }

                    smell += dist_factor;

                    let velocity = velocities[other_idx];
                    sound += dist_factor * self.work_speed_norm[other_idx];
                    hearing += dist_factor * sound_emitters[other_idx];

                    // Blood via dot(heading_dir, n) >= cos(BLOOD_HALF_FOV)
                    let hx = heading.cos();
                    let hy = heading.sin();
                    let align = hx * (dx / dist) + hy * (dy / dist);
                    let cos_bhf = (BLOOD_HALF_FOV).cos();
                    if align >= cos_bhf {
                        let forward_diff = angle_difference(heading, ang);
                        let bleed = (BLOOD_HALF_FOV - forward_diff) / BLOOD_HALF_FOV;
                        let health = healths[other_idx];
                        let wound = (1.0 - (health * 0.5).clamp(0.0, 1.0)).max(0.0);
                        blood += bleed * dist_factor * wound;
                    }
                }
            });

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
                // Capture activations if available
                runtime.brain_activations = runtime.brain.snapshot_activations();
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

        // Reuse handles buffer
        self.work_handles.clear();
        self.work_handles.extend(self.agents.iter_handles());
        let handles = &self.work_handles;
        if handles.is_empty() {
            return;
        }

        let columns = self.agents.columns();
        // Reuse working snapshots
        let _count = handles.len();
        self.work_positions.clear();
        self.work_headings.clear();
        self.work_spike_lengths.clear();
        self.work_positions.extend_from_slice(columns.positions());
        self.work_headings.extend_from_slice(columns.headings());
        self.work_heading_dir_x.clear();
        self.work_heading_dir_y.clear();
        self.work_heading_dir_x
            .resize(self.work_headings.len(), 0.0);
        self.work_heading_dir_y
            .resize(self.work_headings.len(), 0.0);
        for (i, &h) in self.work_headings.iter().enumerate() {
            self.work_heading_dir_x[i] = h.cos();
            self.work_heading_dir_y[i] = h.sin();
        }
        self.work_spike_lengths
            .extend_from_slice(columns.spike_lengths());
        let positions_snapshot = &self.work_positions;
        let headings_snapshot = &self.work_headings;
        let spike_lengths_snapshot = &self.work_spike_lengths;

        let runtime = &self.runtime;
        let terrain = &self.terrain;
        let cell_size = self.config.food_cell_size as f32;
        let topo_enabled = self.config.topography_enabled;
        let topo_gain = self.config.topography_speed_gain.max(0.0);
        let topo_penalty = self.config.topography_energy_penalty.max(0.0);
        #[cfg(feature = "simd_wide")]
        let mut results: Vec<ActuationResult> = vec![ActuationResult::default(); handles.len()];
        #[cfg(feature = "simd_wide")]
        {
            for (chunk_i, chunk) in handles.chunks_exact(4).enumerate() {
                let base = chunk_i * 4;
                for lane in 0..4 {
                    let idx = base + lane;
                    let agent_id = chunk[lane];
                    let Some(rt) = runtime.get(agent_id) else {
                        continue;
                    };
                    let outputs = rt.outputs;
                    let left = outputs.get(0).copied().unwrap_or(0.0).clamp(0.0, 1.0);
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
                        let active_energy = (rt.energy - ramp_floor).max(0.0);
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
                    let energy = (rt.energy - drain).max(0.0);

                    let mut spike_length = spike_lengths_snapshot[idx];
                    if spike_length < spike_target {
                        spike_length = (spike_length + spike_growth).min(spike_target);
                    } else if spike_length > spike_target {
                        spike_length = (spike_length - spike_growth).max(spike_target);
                    }
                    let spiked = spike_length > 0.5;

                    results[idx] = ActuationResult {
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
                    };
                }

                // Remainder handled below outside loop
            }

            let rem = handles.chunks_exact(4).remainder();
            let base = handles.len() - rem.len();
            for (o, agent_id) in rem.iter().enumerate() {
                let idx = base + o;
                let Some(runtime) = runtime.get(*agent_id) else {
                    continue;
                };
                let outputs = runtime.outputs;
                let left = outputs.get(0).copied().unwrap_or(0.0).clamp(0.0, 1.0);
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
                    let (gx, gy) = terrain.gradient_world(
                        positions_snapshot[idx].x,
                        positions_snapshot[idx].y,
                        cell_size,
                    );
                    let dir_x = heading.cos();
                    let dir_y = heading.sin();
                    slope_along = gx * dir_x + gy * dir_y;
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
                results[idx] = ActuationResult {
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
                };
            }
        }

        #[cfg(not(feature = "simd_wide"))]
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

        self.work_handles.clear();
        self.work_handles.extend(self.agents.iter_handles());
        let handles = &self.work_handles;
        if handles.is_empty() {
            return;
        }

        let comfort_band = self.config.temperature_comfort_band.clamp(0.0, 1.0);
        let exponent = self
            .config
            .temperature_discomfort_exponent
            .max(f32::EPSILON);

        self.work_positions.clear();
        self.work_positions
            .extend_from_slice(self.agents.columns().positions());
        let positions_snapshot = &self.work_positions;
        self.work_penalties.clear();
        self.work_penalties.resize(handles.len(), 0.0);
        let penalties = &mut self.work_penalties;

        #[cfg(feature = "simd_wide")]
        {
            use wide::f32x4;

            for (base, chunk) in handles.chunks_exact(4).enumerate() {
                let i0 = base * 4;
                let idxs = [chunk[0], chunk[1], chunk[2], chunk[3]];
                // Gather env temps and preferences per lane
                let t0 = sample_temperature(&self.config, positions_snapshot[i0 + 0].x);
                let t1 = sample_temperature(&self.config, positions_snapshot[i0 + 1].x);
                let t2 = sample_temperature(&self.config, positions_snapshot[i0 + 2].x);
                let t3 = sample_temperature(&self.config, positions_snapshot[i0 + 3].x);

                let p0 = self
                    .runtime
                    .get(idxs[0])
                    .map(|r| r.temperature_preference)
                    .unwrap_or(0.5);
                let p1 = self
                    .runtime
                    .get(idxs[1])
                    .map(|r| r.temperature_preference)
                    .unwrap_or(0.5);
                let p2 = self
                    .runtime
                    .get(idxs[2])
                    .map(|r| r.temperature_preference)
                    .unwrap_or(0.5);
                let p3 = self
                    .runtime
                    .get(idxs[3])
                    .map(|r| r.temperature_preference)
                    .unwrap_or(0.5);

                let t_v = f32x4::new([t0, t1, t2, t3]);
                let p_v = f32x4::new([p0, p1, p2, p3]);
                let diff_v = (t_v - p_v).abs();
                let band_v = f32x4::splat(comfort_band);
                let above_v = (diff_v - band_v).max(f32x4::splat(0.0));

                // Exponent may be non-integer; compute per-lane powf when needed
                let above = above_v.to_array();
                let mut pen = [0.0_f32; 4];
                for lane in 0..4 {
                    let a = above[lane];
                    if a > 0.0 {
                        // Fast path for exponent ~ 2
                        let val = if (exponent - 2.0).abs() < 1e-6 {
                            a * a
                        } else {
                            a.powf(exponent)
                        };
                        pen[lane] = rate * val;
                    }
                }
                // Store penalties back
                penalties[i0 + 0] = pen[0].max(0.0);
                penalties[i0 + 1] = pen[1].max(0.0);
                penalties[i0 + 2] = pen[2].max(0.0);
                penalties[i0 + 3] = pen[3].max(0.0);
            }

            // Remainder (less than 4)
            let rem = handles.chunks_exact(4).remainder();
            let base = handles.len() - rem.len();
            for (o, agent_id) in rem.iter().enumerate() {
                let idx = base + o;
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
        }

        #[cfg(not(feature = "simd_wide"))]
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
        // Reuse buffers: positions, handles, sharers
        self.work_positions.clear();
        self.work_positions
            .extend_from_slice(self.agents.columns().positions());
        let positions = &self.work_positions;

        self.work_handles.clear();
        self.work_handles.extend(self.agents.iter_handles());
        let handles = &self.work_handles;

        let mut sharers: Vec<usize> = Vec::new();
        let food_width = self.food.width() as usize;
        if food_width == 0 || self.food.height() == 0 {
            return;
        }

        let intake_rate = self.config.food_intake_rate.max(0.0);
        let waste_rate = self.config.food_waste_rate.max(0.0);
        let reproduction_bonus = self.config.reproduction_food_bonus.max(0.0);
        let fertility_bonus_scale = self.config.reproduction_fertility_bonus.max(0.0);
        for (idx, agent_id) in handles.iter().enumerate() {
            if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                if intake_rate > 0.0 || waste_rate > 0.0 {
                    let pos = positions[idx];
                    let cell_x = (pos.x / cell_size).floor() as u32 % self.food.width();
                    let cell_y = (pos.y / cell_size).floor() as u32 % self.food.height();
                    let profile_index = (cell_y as usize) * food_width + cell_x as usize;
                    let profile =
                        self.food_profiles
                            .get(profile_index)
                            .copied()
                            .unwrap_or(FoodCellProfile {
                                capacity: self.config.food_max,
                                growth_multiplier: 1.0,
                                decay_multiplier: 1.0,
                                fertility: 0.0,
                                nutrient_density: 0.3,
                            });
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
                                let nutrient = profile.nutrient_density;
                                let energy_gain = intake * (0.5 + nutrient * 0.5);
                                runtime.energy = (runtime.energy + energy_gain).min(2.0);
                                runtime.food_delta += energy_gain;
                                if reproduction_bonus > 0.0 {
                                    let fertility_multiplier =
                                        1.0 + profile.fertility * fertility_bonus_scale;
                                    runtime.reproduction_counter +=
                                        intake * reproduction_bonus * fertility_multiplier;
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

        // Defer indicator pulses to avoid borrowing conflicts
        let mut indicator_pulses: Vec<(AgentId, f32, [f32; 3])> = Vec::new();
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
                indicator_pulses.push((giver_id, 10.0, [1.0, 1.0, 1.0]));
                indicator_pulses.push((*recipient_id, 10.0, [1.0, 1.0, 1.0]));
            }
        }
        for (id, intensity, color) in indicator_pulses {
            self.pulse_indicator(id, intensity, color);
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

        // Species barrier: require matching brain kinds for sexual reproduction
        if let Some(ref partner_rt) = partner_runtime {
            let pk1 = parent_runtime.brain.registry_key();
            let pk2 = partner_rt.brain.registry_key();
            let kind_match = match (pk1, pk2) {
                (Some(k1), Some(k2)) => {
                    let a = self.brain_registry.kind(k1);
                    let b = self.brain_registry.kind(k2);
                    a.is_some() && a == b
                }
                _ => false,
            };
            if !kind_match {
                return false; // fall back to random spawn in caller
            }
        } else {
            return false;
        }

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
            // Inherit parent brain binding (same kind already enforced)
            if let Some(key) = parent_runtime.brain.registry_key() {
                let _ = self.bind_agent_brain(child_id, key);
            }
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
        let id = self.spawn_agent(data);
        if let Some(key) = self.brain_registry.random_key(&mut self.rng) {
            let _ = self.bind_agent_brain(id, key);
        }
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

        // Reuse handles buffer
        self.work_handles.clear();
        self.work_handles.extend(self.agents.iter_handles());
        let handles = &self.work_handles;
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
        // Reuse velocity buffer
        self.work_velocities.clear();
        self.work_velocities
            .extend_from_slice(self.agents.columns().velocities());
        let velocities = &self.work_velocities;
        let spike_lengths = self.agents.columns().spike_lengths();
        // Reuse position_pairs buffer for index rebuild
        self.work_position_pairs.clear();
        self.work_position_pairs.reserve(positions.len());
        for p in positions.iter() {
            self.work_position_pairs.push((p.x, p.y));
        }
        let _ = self.index.rebuild(&self.work_position_pairs);

        let spike_damage = self.config.spike_damage;
        let spike_energy_cost = self.config.spike_energy_cost;
        let index = &self.index;
        // Reuse runtime snapshot buffer
        self.work_runtime_snapshot.clear();
        self.work_runtime_snapshot.reserve(handles.len());
        for id in handles.iter() {
            self.work_runtime_snapshot
                .push(self.runtime.get(*id).cloned().unwrap_or_default());
        }
        let runtime_snapshot = &self.work_runtime_snapshot;

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
            index.visit_neighbor_buckets(idx, reach, &mut |indices| {
                #[cfg(feature = "simd_wide")]
                {
                    for chunk in indices.chunks_exact(4) {
                        let a0 = chunk[0];
                        let a1 = chunk[1];
                        let a2 = chunk[2];
                        let a3 = chunk[3];
                        let dx_arr = [
                            Self::wrap_delta(origin.x, positions[a0].x, world_w),
                            Self::wrap_delta(origin.x, positions[a1].x, world_w),
                            Self::wrap_delta(origin.x, positions[a2].x, world_w),
                            Self::wrap_delta(origin.x, positions[a3].x, world_w),
                        ];
                        let dy_arr = [
                            Self::wrap_delta(origin.y, positions[a0].y, world_h),
                            Self::wrap_delta(origin.y, positions[a1].y, world_h),
                            Self::wrap_delta(origin.y, positions[a2].y, world_h),
                            Self::wrap_delta(origin.y, positions[a3].y, world_h),
                        ];
                        let dx_v = f32x4::new(dx_arr);
                        let dy_v = f32x4::new(dy_arr);
                        let dist_sq_v = dx_v * dx_v + dy_v * dy_v;
                        let dist_v = dist_sq_v.sqrt();
                        let dir_x_v = dx_v / dist_v;
                        let dir_y_v = dy_v / dist_v;
                        let align_v =
                            dir_x_v * f32x4::splat(facing.0) + dir_y_v * f32x4::splat(facing.1);
                        // Build lane mask for (not self) && (dist within reach) && (alignment >= threshold)
                        let dist_sq_arr = dist_sq_v.to_array();
                        let align_arr = align_v.to_array();
                        let ids = [a0, a1, a2, a3];
                        let mut dmg_arr = [0.0_f32; 4];
                        for lane in 0..4 {
                            let oid = ids[lane];
                            if oid == idx {
                                continue;
                            }
                            let d2 = dist_sq_arr[lane];
                            if d2 <= f32::EPSILON || d2 > reach_sq {
                                continue;
                            }
                            let al = align_arr[lane];
                            if al < alignment_threshold {
                                continue;
                            }
                            let dmg = base_damage * al.max(0.0);
                            if dmg > 0.0 {
                                dmg_arr[lane] = dmg;
                            }
                        }
                        // Emit per-lane respecting order
                        for lane in 0..4 {
                            let damage = dmg_arr[lane];
                            if damage <= 0.0 {
                                continue;
                            }
                            let other_idx = ids[lane];
                            let target_runtime = &runtime_snapshot[other_idx];
                            let victim_carnivore =
                                target_runtime.herbivore_tendency < carnivore_threshold;
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
                        }
                    }
                    let rem = indices.chunks_exact(4).remainder();
                    for &other_idx in rem {
                        if other_idx == idx {
                            continue;
                        }
                        let target_runtime = &runtime_snapshot[other_idx];
                        let dx = Self::wrap_delta(origin.x, positions[other_idx].x, world_w);
                        let dy = Self::wrap_delta(origin.y, positions[other_idx].y, world_h);
                        let dist_sq = dx * dx + dy * dy;
                        if dist_sq <= f32::EPSILON || dist_sq > reach_sq {
                            continue;
                        }
                        let dist = dist_sq.sqrt();
                        let dir_x = dx / dist;
                        let dir_y = dy / dist;
                        let alignment = dot2(facing.0, facing.1, dir_x, dir_y);
                        if alignment < alignment_threshold {
                            continue;
                        }
                        let damage = base_damage * alignment.max(0.0);
                        if damage <= 0.0 {
                            continue;
                        }
                        let victim_carnivore =
                            target_runtime.herbivore_tendency < carnivore_threshold;
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
                    }
                }
                #[cfg(not(feature = "simd_wide"))]
                {
                    for &other_idx in indices {
                        if other_idx == idx {
                            continue;
                        }
                        let target_runtime = &runtime_snapshot[other_idx];
                        let dx = Self::wrap_delta(origin.x, positions[other_idx].x, world_w);
                        let dy = Self::wrap_delta(origin.y, positions[other_idx].y, world_h);
                        let dist_sq = dx * dx + dy * dy;
                        if dist_sq <= f32::EPSILON || dist_sq > reach_sq {
                            continue;
                        }
                        let dist = dist_sq.sqrt();
                        let dir_x = dx / dist;
                        let dir_y = dy / dist;
                        let alignment = dot2(facing.0, facing.1, dir_x, dir_y);
                        if alignment < alignment_threshold {
                            continue;
                        }

                        let damage = base_damage * alignment.max(0.0);
                        if damage <= 0.0 {
                            continue;
                        }
                        let victim_carnivore =
                            target_runtime.herbivore_tendency < carnivore_threshold;
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
                    }
                }
            });

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

        let attempts = results
            .iter()
            .filter(|result| !result.hits.is_empty())
            .count() as u32;
        let hits = results
            .iter()
            .map(|result| result.hits.len() as u32)
            .sum::<u32>();
        self.combat_spike_attempts = self.combat_spike_attempts.saturating_add(attempts);
        self.combat_spike_hits = self.combat_spike_hits.saturating_add(hits);
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
            #[cfg(feature = "simd_wide")]
            {
                use wide::f32x4;
                for (chunk_i, chunk) in handles.chunks_exact(4).enumerate() {
                    let base = chunk_i * 4;
                    let ids = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    let mut dx_arr = [0.0_f32; 4];
                    let mut dy_arr = [0.0_f32; 4];
                    for lane in 0..4 {
                        let idx = base + lane;
                        dx_arr[lane] = toroidal_delta(positions[idx].x, victim_pos.x, width);
                        dy_arr[lane] = toroidal_delta(positions[idx].y, victim_pos.y, height);
                    }
                    let dx_v = f32x4::new(dx_arr);
                    let dy_v = f32x4::new(dy_arr);
                    let dist2_v = dx_v * dx_v + dy_v * dy_v;
                    let dist2 = dist2_v.to_array();
                    for lane in 0..4 {
                        let idx = base + lane;
                        if ids[lane] == *agent_id {
                            continue;
                        }
                        if healths.get(idx).copied().unwrap_or(0.0) <= 0.0 {
                            continue;
                        }
                        if dist2[lane] <= radius_sq {
                            neighbor_indices.push(idx);
                        }
                    }
                }
                let rem = handles.chunks_exact(4).remainder();
                let base = handles.len() - rem.len();
                for (o, neighbor_id) in rem.iter().enumerate() {
                    let idx = base + o;
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
            }
            #[cfg(not(feature = "simd_wide"))]
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

    fn stage_death_cleanup(&mut self, tick: Tick) {
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

        let death_records: Vec<DeathRecord> = {
            let columns = self.agents.columns();
            dead.iter()
                .filter_map(|(idx, agent_id)| {
                    let data = columns.snapshot(*idx);
                    let runtime = self.runtime.get(*agent_id)?.clone();
                    let herbivore = clamp01(runtime.herbivore_tendency);
                    let brain_kind = runtime.brain.kind().map(str::to_string);
                    let brain_key = runtime.brain.registry_key();
                    let cause = if runtime.combat.was_spiked_by_carnivore {
                        DeathCause::CombatCarnivore
                    } else if runtime.combat.was_spiked_by_herbivore {
                        DeathCause::CombatHerbivore
                    } else if runtime.energy <= f32::EPSILON && runtime.food_delta < 0.0 {
                        DeathCause::Starvation
                    } else if data.age >= self.config.aging_health_decay_start {
                        DeathCause::Aging
                    } else {
                        DeathCause::Unknown
                    };

                    Some(DeathRecord {
                        tick,
                        agent_id: *agent_id,
                        age: data.age,
                        generation: data.generation,
                        herbivore_tendency: herbivore,
                        brain_kind,
                        brain_key,
                        energy: runtime.energy,
                        food_balance_total: runtime.food_balance_total + runtime.food_delta,
                        cause,
                        was_hybrid: runtime.hybrid,
                        combat_flags: runtime.combat,
                    })
                })
                .collect()
        };
        if !death_records.is_empty() {
            self.pending_death_records.extend(death_records);
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
        let reproduction_window = self.cadence.reproduction_window(self.tick.next());
        let reproduction_chance = self.cadence.reproduction_chance();

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
            }

            if !reproduction_window {
                continue;
            }
            if reproduction_chance <= 0.0 {
                continue;
            }
            if reproduction_chance < 1.0 && self.rng.random_range(0.0..1.0) >= reproduction_chance {
                continue;
            }

            {
                let runtime = match self.runtime.get_mut(*agent_id) {
                    Some(rt) => rt,
                    None => continue,
                };
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
                parent_id: *agent_id,
                partner_id: partner_index.map(|j| handles[j]),
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

    fn stage_spawn_commit(&mut self, tick: Tick) {
        if self.pending_spawns.is_empty() {
            return;
        }
        let mut orders = std::mem::take(&mut self.pending_spawns);
        orders.sort_by_key(|order| order.parent_index);
        self.last_births = orders.len();
        for order in orders {
            let SpawnOrder {
                parent_index: _,
                parent_id,
                partner_id,
                data,
                runtime,
            } = order;
            let child_id = self.spawn_agent(data);
            self.runtime.insert(child_id, runtime);

            if let (Some(child_runtime), Some(idx)) =
                (self.runtime.get(child_id), self.agents.index_of(child_id))
            {
                let snapshot = self.agents.columns().snapshot(idx);
                let brain_kind = child_runtime.brain.kind().map(str::to_string);
                let brain_key = child_runtime.brain.registry_key();
                let record = BirthRecord {
                    tick,
                    agent_id: child_id,
                    parent_a: Some(parent_id),
                    parent_b: partner_id,
                    brain_kind,
                    brain_key,
                    herbivore_tendency: clamp01(child_runtime.herbivore_tendency),
                    generation: snapshot.generation,
                    position: snapshot.position,
                    is_hybrid: child_runtime.hybrid,
                };
                self.pending_birth_records.push(record);
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
            self.pending_birth_records.clear();
            self.pending_death_records.clear();
            self.combat_spike_attempts = 0;
            self.combat_spike_hits = 0;
            return;
        }

        let analytics = self.config.analytics_stride;
        let macro_enabled = analytics.macro_metrics != 0
            && next_tick.0.is_multiple_of(analytics.macro_metrics as u64);
        let behavior_enabled = analytics.behavior_metrics != 0
            && next_tick
                .0
                .is_multiple_of(analytics.behavior_metrics as u64);
        let lifecycle_enabled = analytics.lifecycle_events != 0
            && next_tick
                .0
                .is_multiple_of(analytics.lifecycle_events as u64);

        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        let agent_count = handles.len();

        let mut total_energy = 0.0f32;
        let mut total_health = 0.0f32;

        let mut carnivores = 0usize;
        let mut herbivores = 0usize;
        let mut hybrids = 0usize;
        let mut carnivore_energy = 0.0f64;
        let mut herbivore_energy = 0.0f64;
        let mut hybrid_energy = 0.0f64;

        let mut mutation_primary = RunningStats::default();
        let mut mutation_secondary = RunningStats::default();
        let mut trait_smell = RunningStats::default();
        let mut trait_sound = RunningStats::default();
        let mut trait_hearing = RunningStats::default();
        let mut trait_eye = RunningStats::default();
        let mut trait_blood = RunningStats::default();
        let mut herbivore_tendency_stats = RunningStats::default();

        let mut sensor_mean = RunningStats::default();
        let mut sensor_max = RunningStats::default();
        let mut sensor_entropy = RunningStats::default();
        let mut output_mean = RunningStats::default();
        let mut output_max = RunningStats::default();
        let mut output_entropy = RunningStats::default();

        let mut reproduction_counter_stats = RunningStats::default();
        let mut temperature_pref_stats = RunningStats::default();
        let mut age_sum = 0.0f64;
        let mut age_max = 0u32;
        let mut boost_count = 0usize;

        let mut food_delta_sum = 0.0f64;
        let mut food_delta_abs_sum = 0.0f64;

        let carnivore_threshold = self.config.carnivore_threshold;
        let mut brain_map: HashMap<String, (usize, f64)> = HashMap::new();

        let columns = self.agents.columns();
        let healths = columns.health();
        let ages = columns.ages();
        let boosts = columns.boosts();
        let positions = columns.positions();
        let generations = columns.generations();

        let mut generation_sum = 0.0f64;
        let mut generation_max = 0u32;
        let mut temperature_discomfort_stats = RunningStats::default();

        for (idx, agent_id) in handles.iter().enumerate() {
            total_health += healths.get(idx).copied().unwrap_or(0.0);
            if let Some(age) = ages.get(idx).copied() {
                age_sum += age as f64;
                if age > age_max {
                    age_max = age;
                }
            }
            if boosts.get(idx).copied().unwrap_or(false) {
                boost_count += 1;
            }
            if let Some(runtime) = self.runtime.get(*agent_id) {
                total_energy += runtime.energy;

                reproduction_counter_stats.update(f64::from(runtime.reproduction_counter));
                temperature_pref_stats.update(f64::from(runtime.temperature_preference));
                if let Some(generation) = generations.get(idx) {
                    let value = generation.0;
                    generation_sum += value as f64;
                    if value > generation_max {
                        generation_max = value;
                    }
                }

                if let Some(position) = positions.get(idx).filter(|_| macro_enabled) {
                    let env_temperature = sample_temperature(&self.config, position.x);
                    let discomfort = f64::from(temperature_discomfort(
                        env_temperature,
                        runtime.temperature_preference,
                    ));
                    temperature_discomfort_stats.update(discomfort);
                }

                if macro_enabled {
                    let herb = clamp01(runtime.herbivore_tendency);
                    herbivore_tendency_stats.update(f64::from(herb));
                    if runtime.hybrid {
                        hybrids += 1;
                        hybrid_energy += f64::from(runtime.energy);
                    } else if herb >= carnivore_threshold {
                        herbivores += 1;
                        herbivore_energy += f64::from(runtime.energy);
                    } else {
                        carnivores += 1;
                        carnivore_energy += f64::from(runtime.energy);
                    }

                    mutation_primary.update(f64::from(runtime.mutation_rates.primary));
                    mutation_secondary.update(f64::from(runtime.mutation_rates.secondary));
                    trait_smell.update(f64::from(runtime.trait_modifiers.smell));
                    trait_sound.update(f64::from(runtime.trait_modifiers.sound));
                    trait_hearing.update(f64::from(runtime.trait_modifiers.hearing));
                    trait_eye.update(f64::from(runtime.trait_modifiers.eye));
                    trait_blood.update(f64::from(runtime.trait_modifiers.blood));

                    let label = runtime
                        .brain
                        .kind()
                        .map(str::to_string)
                        .unwrap_or_else(|| "unbound".to_string());
                    let entry = brain_map.entry(label).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += f64::from(runtime.energy);
                }

                if behavior_enabled {
                    let (sensor_avg, sensor_peak, sensor_ent) = summarize_signal(&runtime.sensors);
                    sensor_mean.update(sensor_avg);
                    sensor_max.update(sensor_peak);
                    sensor_entropy.update(sensor_ent);

                    let (output_avg, output_peak, output_ent) = summarize_signal(&runtime.outputs);
                    output_mean.update(output_avg);
                    output_max.update(output_peak);
                    output_entropy.update(output_ent);
                }

                if behavior_enabled || macro_enabled {
                    let delta = f64::from(runtime.food_delta);
                    food_delta_sum += delta;
                    food_delta_abs_sum += delta.abs();
                }
            }
        }

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

        let summary = TickSummary {
            tick: next_tick,
            agent_count,
            births: self.last_births,
            deaths: self.last_deaths,
            total_energy,
            average_energy,
            average_health,
            max_age: age_max,
            spike_hits: self.combat_spike_hits,
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

        if macro_enabled {
            let as_f64 = |value: usize| value as f64;
            metrics.push(MetricSample::new(
                "population.carnivore.count",
                as_f64(carnivores),
            ));
            metrics.push(MetricSample::new(
                "population.herbivore.count",
                as_f64(herbivores),
            ));
            metrics.push(MetricSample::new(
                "population.hybrid.count",
                as_f64(hybrids),
            ));

            if carnivores > 0 {
                metrics.push(MetricSample::new(
                    "population.carnivore.avg_energy",
                    carnivore_energy / as_f64(carnivores),
                ));
            }
            if herbivores > 0 {
                metrics.push(MetricSample::new(
                    "population.herbivore.avg_energy",
                    herbivore_energy / as_f64(herbivores),
                ));
            }
            if hybrids > 0 {
                metrics.push(MetricSample::new(
                    "population.hybrid.avg_energy",
                    hybrid_energy / as_f64(hybrids),
                ));
            }

            metrics.push(MetricSample::new(
                "mutation.primary.mean",
                mutation_primary.mean(),
            ));
            metrics.push(MetricSample::new(
                "mutation.primary.stddev",
                mutation_primary.stddev(),
            ));
            metrics.push(MetricSample::new(
                "mutation.secondary.mean",
                mutation_secondary.mean(),
            ));
            metrics.push(MetricSample::new(
                "mutation.secondary.stddev",
                mutation_secondary.stddev(),
            ));
            metrics.push(MetricSample::new("traits.smell.mean", trait_smell.mean()));
            metrics.push(MetricSample::new("traits.sound.mean", trait_sound.mean()));
            metrics.push(MetricSample::new(
                "traits.hearing.mean",
                trait_hearing.mean(),
            ));
            metrics.push(MetricSample::new("traits.eye.mean", trait_eye.mean()));
            metrics.push(MetricSample::new("traits.blood.mean", trait_blood.mean()));
            metrics.push(MetricSample::new(
                "herbivore_tendency.mean",
                herbivore_tendency_stats.mean(),
            ));
            metrics.push(MetricSample::new(
                "herbivore_tendency.stddev",
                herbivore_tendency_stats.stddev(),
            ));

            if agent_count > 0 {
                metrics.push(MetricSample::new(
                    "food_delta.mean",
                    food_delta_sum / agent_count as f64,
                ));
                metrics.push(MetricSample::new(
                    "food_delta.mean_abs",
                    food_delta_abs_sum / agent_count as f64,
                ));
                metrics.push(MetricSample::new(
                    "population.age.mean",
                    age_sum / agent_count as f64,
                ));
                metrics.push(MetricSample::new("population.age.max", age_max as f64));
                metrics.push(MetricSample::new(
                    "behavior.boost.count",
                    boost_count as f64,
                ));
                metrics.push(MetricSample::new(
                    "behavior.boost.ratio",
                    if agent_count > 0 {
                        boost_count as f64 / agent_count as f64
                    } else {
                        0.0
                    },
                ));
                metrics.push(MetricSample::new(
                    "reproduction.counter.mean",
                    reproduction_counter_stats.mean(),
                ));
                metrics.push(MetricSample::new(
                    "temperature.preference.mean",
                    temperature_pref_stats.mean(),
                ));
                metrics.push(MetricSample::new(
                    "temperature.preference.stddev",
                    temperature_pref_stats.stddev(),
                ));
                metrics.push(MetricSample::new(
                    "population.generation.mean",
                    generation_sum / agent_count as f64,
                ));
                metrics.push(MetricSample::new(
                    "population.generation.max",
                    generation_max as f64,
                ));
                metrics.push(MetricSample::new(
                    "temperature.discomfort.mean",
                    temperature_discomfort_stats.mean(),
                ));
                metrics.push(MetricSample::new(
                    "temperature.discomfort.stddev",
                    temperature_discomfort_stats.stddev(),
                ));
            }

            if let Some((total, mean, variance, max)) = summarize_food_grid(self.food.cells()) {
                metrics.push(MetricSample::new("food.total", total));
                metrics.push(MetricSample::new("food.mean", mean));
                metrics.push(MetricSample::new("food.stddev", variance.sqrt()));
                metrics.push(MetricSample::from_f32("food.max", max));
            }

            if let Some(hydrology) = self.hydrology.as_ref() {
                let total_water = hydrology.total_water_depth();
                let flooded = hydrology.flooded_cell_counts(0.05, 0.2);
                let cell_count = hydrology.cell_count().max(1) as f64;
                metrics.push(MetricSample::new(
                    "hydrology.water.total_depth",
                    f64::from(total_water),
                ));
                metrics.push(MetricSample::new(
                    "hydrology.water.mean_depth",
                    f64::from(total_water) / cell_count,
                ));
                metrics.push(MetricSample::new(
                    "hydrology.water.flooded.shallow.count",
                    flooded.0 as f64,
                ));
                metrics.push(MetricSample::new(
                    "hydrology.water.flooded.deep.count",
                    flooded.1 as f64,
                ));
                metrics.push(MetricSample::new(
                    "hydrology.water.flooded.shallow.ratio",
                    flooded.0 as f64 / cell_count,
                ));
                metrics.push(MetricSample::new(
                    "hydrology.water.flooded.deep.ratio",
                    flooded.1 as f64 / cell_count,
                ));
            }

            for (label, (count, energy_sum)) in brain_map {
                let key = sanitize_metric_key(&label);
                metrics.push(MetricSample::new(
                    format!("brain.population.{key}.count"),
                    count as f64,
                ));
                if count > 0 {
                    metrics.push(MetricSample::new(
                        format!("brain.population.{key}.avg_energy"),
                        energy_sum / count as f64,
                    ));
                }
            }
        }

        if behavior_enabled {
            metrics.push(MetricSample::new(
                "behavior.sensors.mean",
                sensor_mean.mean(),
            ));
            metrics.push(MetricSample::new(
                "behavior.sensors.stddev",
                sensor_mean.stddev(),
            ));
            metrics.push(MetricSample::new("behavior.sensors.max", sensor_max.mean()));
            metrics.push(MetricSample::new(
                "behavior.sensors.entropy",
                sensor_entropy.mean(),
            ));

            metrics.push(MetricSample::new(
                "behavior.outputs.mean",
                output_mean.mean(),
            ));
            metrics.push(MetricSample::new(
                "behavior.outputs.stddev",
                output_mean.stddev(),
            ));
            metrics.push(MetricSample::new("behavior.outputs.max", output_max.mean()));
            metrics.push(MetricSample::new(
                "behavior.outputs.entropy",
                output_entropy.mean(),
            ));
        }

        let mut events = Vec::with_capacity(4);
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
        if self.combat_spike_attempts > 0 {
            events.push(PersistenceEvent::new(
                PersistenceEventKind::Custom(Cow::Borrowed("spike_attempts")),
                self.combat_spike_attempts as usize,
            ));
        }
        if self.combat_spike_hits > 0 {
            events.push(PersistenceEvent::new(
                PersistenceEventKind::Custom(Cow::Borrowed("spike_hits")),
                self.combat_spike_hits as usize,
            ));
        }

        let mut agents = Vec::with_capacity(agent_count);
        for id in &handles {
            if let Some(snapshot) = self.snapshot_agent(*id) {
                agents.push(snapshot);
            }
        }

        for id in &handles {
            if let Some(runtime) = self.runtime.get_mut(*id) {
                runtime.food_balance_total += runtime.food_delta;
            }
        }

        if lifecycle_enabled && !self.pending_death_records.is_empty() {
            let mut combat_carnivore = 0usize;
            let mut combat_herbivore = 0usize;
            let mut starvation = 0usize;
            let mut aging = 0usize;
            let mut unknown = 0usize;
            for record in &self.pending_death_records {
                match record.cause {
                    DeathCause::CombatCarnivore => combat_carnivore += 1,
                    DeathCause::CombatHerbivore => combat_herbivore += 1,
                    DeathCause::Starvation => starvation += 1,
                    DeathCause::Aging => aging += 1,
                    DeathCause::Unknown => unknown += 1,
                }
            }
            let total = combat_carnivore + combat_herbivore + starvation + aging + unknown;
            if total > 0 {
                metrics.push(MetricSample::new(
                    "mortality.combat_carnivore.count",
                    combat_carnivore as f64,
                ));
                metrics.push(MetricSample::new(
                    "mortality.combat_herbivore.count",
                    combat_herbivore as f64,
                ));
                metrics.push(MetricSample::new(
                    "mortality.starvation.count",
                    starvation as f64,
                ));
                metrics.push(MetricSample::new("mortality.aging.count", aging as f64));
                metrics.push(MetricSample::new("mortality.unknown.count", unknown as f64));
                metrics.push(MetricSample::new("mortality.total.count", total as f64));
                metrics.push(MetricSample::new(
                    "mortality.combat_carnivore.ratio",
                    combat_carnivore as f64 / total as f64,
                ));
                metrics.push(MetricSample::new(
                    "mortality.combat_herbivore.ratio",
                    combat_herbivore as f64 / total as f64,
                ));
                metrics.push(MetricSample::new(
                    "mortality.starvation.ratio",
                    starvation as f64 / total as f64,
                ));
                metrics.push(MetricSample::new(
                    "mortality.aging.ratio",
                    aging as f64 / total as f64,
                ));
                metrics.push(MetricSample::new(
                    "mortality.unknown.ratio",
                    unknown as f64 / total as f64,
                ));
            }
        }

        if lifecycle_enabled && !self.pending_birth_records.is_empty() {
            let total = self.pending_birth_records.len();
            let hybrid = self
                .pending_birth_records
                .iter()
                .filter(|record| record.is_hybrid)
                .count();
            metrics.push(MetricSample::new("births.total.count", total as f64));
            metrics.push(MetricSample::new("births.hybrid.count", hybrid as f64));
            if total > 0 {
                metrics.push(MetricSample::new(
                    "births.hybrid.ratio",
                    hybrid as f64 / total as f64,
                ));
            }
        }

        let births = if lifecycle_enabled {
            std::mem::take(&mut self.pending_birth_records)
        } else if analytics.lifecycle_events == 0 {
            self.pending_birth_records.clear();
            Vec::new()
        } else {
            Vec::new()
        };

        let deaths = if lifecycle_enabled {
            std::mem::take(&mut self.pending_death_records)
        } else if analytics.lifecycle_events == 0 {
            self.pending_death_records.clear();
            Vec::new()
        } else {
            Vec::new()
        };

        let batch = PersistenceBatch {
            summary: summary.clone(),
            epoch: self.epoch,
            closed: self.closed,
            metrics,
            events,
            agents,
            births,
            deaths,
            replay_events: std::mem::take(&mut self.replay_events),
        };
        self.last_spike_hits = self.combat_spike_hits;
        self.last_max_age = age_max;
        self.persistence.on_tick(&batch);
        if self.history.len() >= self.config.history_capacity {
            self.history.pop_front();
        }
        self.history.push_back(summary);
        self.last_births = 0;
        self.last_deaths = 0;
        self.carcass_health_distributed = 0.0;
        self.carcass_reproduction_bonus = 0.0;
        self.combat_spike_attempts = 0;
        self.combat_spike_hits = 0;
    }

    /// Execute one simulation tick pipeline returning emitted events.
    pub fn step(&mut self) -> TickEvents {
        let next_tick = self.tick.next();
        let previous_epoch = self.epoch;

        if self.cadence.should_age(next_tick) {
            self.stage_aging();
        }
        let food_respawned = self.stage_food_dynamics(next_tick);
        self.stage_sense();
        self.stage_brains();
        self.stage_actuation();
        self.stage_temperature_discomfort();
        self.stage_food();
        self.stage_combat();
        self.stage_death_cleanup(next_tick);
        self.stage_reproduction();
        self.stage_population(next_tick);
        self.stage_spawn_commit(next_tick);
        self.stage_persistence(next_tick);

        let mut events = TickEvents {
            tick: next_tick,
            charts_flushed: self.cadence.should_emit_chart_event(next_tick),
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

    /// Apply a new configuration, refreshing derived caches while preserving runtime state.
    pub fn apply_config_update(
        &mut self,
        new_config: ScriptBotsConfig,
    ) -> Result<(), WorldStateError> {
        let (food_w, food_h) = new_config.food_dimensions()?;
        let current_dims = (self.food.width(), self.food.height());
        if (food_w, food_h) != current_dims {
            return Err(WorldStateError::InvalidConfig(
                "changing world dimensions at runtime is not supported; restart with the new configuration",
            ));
        }

        let food_profiles = FoodCellProfile::compute(&new_config, &self.terrain);
        let scratch_len = (food_w as usize) * (food_h as usize);
        if self.food_scratch.len() != scratch_len {
            self.food_scratch.resize(scratch_len, 0.0);
        }

        {
            let cells = self.food.cells_mut();
            if !food_profiles.is_empty() {
                for (idx, cell) in cells.iter_mut().enumerate() {
                    if let Some(profile) = food_profiles.get(idx) {
                        if *cell > profile.capacity {
                            *cell = profile.capacity;
                        }
                    } else if *cell > new_config.food_max {
                        *cell = new_config.food_max;
                    }
                }
            } else {
                for cell in cells.iter_mut() {
                    if *cell > new_config.food_max {
                        *cell = new_config.food_max;
                    }
                }
            }
        }

        let new_index = UniformGridIndex::new(
            new_config.food_cell_size as f32,
            new_config.world_width as f32,
            new_config.world_height as f32,
        );

        // Record audit entry
        let tick = self.tick.0;
        if let Ok(value) = serde_json::to_value(&new_config) {
            self.config_audit
                .push(ConfigAuditEntry { tick, patch: value });
            if self.config_audit.len() > 64 {
                let drop_count = self.config_audit.len() - 64;
                self.config_audit.drain(0..drop_count);
            }
        }

        self.config = new_config;
        self.food_profiles = food_profiles;
        self.index = new_index;
        self.cadence = TickCadence::from_config(&self.config);
        Ok(())
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

    pub fn config_audit(&self) -> &[ConfigAuditEntry] {
        &self.config_audit
    }

    /// Toggle the closed-environment flag.
    pub fn set_closed(&mut self, closed: bool) {
        self.closed = closed;
    }

    /// Iterate over retained tick summaries.
    pub fn history(&self) -> impl DoubleEndedIterator<Item = &TickSummary> {
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

    /// Spike hits recorded during the most recent tick.
    pub fn last_spike_hits(&self) -> u32 {
        self.last_spike_hits
    }

    /// Maximum agent age observed during the most recent tick.
    pub fn last_max_age(&self) -> u32 {
        self.last_max_age
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

    /// Return the derived profile for the specified food cell, when available.
    #[must_use]
    pub fn food_profile(&self, x: u32, y: u32) -> Option<FoodCellProfileSnapshot> {
        if x >= self.food.width() || y >= self.food.height() {
            return None;
        }
        let idx = (y as usize) * (self.food.width() as usize) + x as usize;
        self.food_profiles.get(idx).map(Into::into)
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

    /// Replace the current terrain and food fields using a pre-generated map artifact.
    pub fn apply_map_artifact(&mut self, artifact: &MapArtifact) -> Result<(), WorldStateError> {
        let terrain = artifact.terrain();
        if terrain.width() != self.food.width() || terrain.height() != self.food.height() {
            return Err(WorldStateError::InvalidConfig(
                "map artifact dimensions must match existing food grid",
            ));
        }
        if terrain.cell_size() != self.config.food_cell_size {
            return Err(WorldStateError::InvalidConfig(
                "map artifact cell size must match configuration",
            ));
        }

        self.terrain = terrain.clone();
        self.food_profiles = FoodCellProfile::compute(&self.config, &self.terrain);

        if let Some(field) = artifact.fertility() {
            if field.width() != self.food.width() || field.height() != self.food.height() {
                return Err(WorldStateError::InvalidConfig(
                    "fertility artifact dimensions must match existing food grid",
                ));
            }
            let max_food = self.config.food_max;
            for (cell, value) in self.food.cells_mut().iter_mut().zip(field.values().iter()) {
                *cell = value.clamp(0.0, 1.0) * max_food;
            }
        }

        self.hydrology = match (artifact.hydrology_tiles(), artifact.hydrology_field()) {
            (Some(tiles), Some(field)) => Some(HydrologyState::new(tiles.clone(), field.clone())),
            _ => None,
        };

        self.map_metadata = Some(artifact.metadata().clone());
        Ok(())
    }

    /// Metadata describing the last applied procedural map, when available.
    pub fn map_metadata(&self) -> Option<&MapArtifactMetadata> {
        self.map_metadata.as_ref()
    }

    /// Immutable access to hydrology state when available.
    pub fn hydrology(&self) -> Option<&HydrologyState> {
        self.hydrology.as_ref()
    }

    /// Mutable access to hydrology state when available.
    pub fn hydrology_mut(&mut self) -> Option<&mut HydrologyState> {
        self.hydrology.as_mut()
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

    /// Produce a filtered, sorted listing of agents for debug consumers.
    pub fn agent_debug_view(&self, query: AgentDebugQuery) -> Vec<AgentDebugInfo> {
        let AgentDebugQuery {
            ids,
            diet,
            selection,
            brain_kind,
            limit,
            sort,
        } = query;

        let id_filter: Option<HashSet<AgentId>> = ids.map(|list| {
            list.into_iter()
                .map(Self::decode_agent_id)
                .collect::<HashSet<_>>()
        });

        let brain_filter = brain_kind.as_ref().map(|value| value.to_lowercase());

        let mut entries: Vec<AgentDebugInfo> = Vec::new();
        for handle in self.agents.iter_handles() {
            if let Some(filter) = &id_filter
                && !filter.contains(&handle)
            {
                continue;
            }

            let Some(snapshot) = self.snapshot_agent(handle) else {
                continue;
            };
            let runtime = snapshot.runtime;

            if let Some(expected) = selection
                && runtime.selection != expected
            {
                continue;
            }

            let diet_class = DietClass::from_tendency(runtime.herbivore_tendency);
            if let Some(expected_diet) = diet
                && diet_class != expected_diet
            {
                continue;
            }

            if let Some(filter) = &brain_filter {
                let actual = runtime
                    .brain
                    .kind()
                    .map(|kind| kind.to_lowercase())
                    .unwrap_or_default();
                if !actual.contains(filter) {
                    continue;
                }
            }

            entries.push(AgentDebugInfo {
                agent_id: Self::encode_agent_id(handle),
                selection: runtime.selection,
                position: snapshot.data.position,
                energy: runtime.energy,
                health: snapshot.data.health,
                age: snapshot.data.age,
                generation: snapshot.data.generation.0,
                herbivore_tendency: runtime.herbivore_tendency,
                diet: diet_class,
                brain_kind: runtime.brain.kind().map(str::to_string),
                brain_key: runtime.brain.registry_key(),
                mutation_primary: runtime.mutation_rates.primary,
                mutation_secondary: runtime.mutation_rates.secondary,
                indicator: runtime.indicator,
            });
        }

        match sort {
            AgentDebugSort::EnergyDesc => entries.sort_by(|a, b| {
                b.energy
                    .partial_cmp(&a.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            AgentDebugSort::AgeDesc => entries.sort_by(|a, b| b.age.cmp(&a.age)),
        }

        if let Some(limit) = limit
            && entries.len() > limit
        {
            entries.truncate(limit);
        }

        entries
    }

    /// Apply a selection update to highlight agents.
    pub fn apply_selection_update(&mut self, update: SelectionUpdate) -> SelectionResult {
        let mut cleared = 0usize;
        let mut applied = 0usize;

        let SelectionUpdate {
            mode,
            agent_ids,
            state,
        } = update;

        let targets: HashSet<AgentId> = agent_ids
            .into_iter()
            .map(Self::decode_agent_id)
            .filter(|id| self.agents.contains(*id))
            .collect();

        match mode {
            SelectionMode::Replace => {
                for runtime in self.runtime.values_mut() {
                    if !matches!(runtime.selection, SelectionState::None) {
                        runtime.selection = SelectionState::None;
                        cleared += 1;
                    }
                }
                for id in &targets {
                    if let Some(runtime) = self.runtime.get_mut(*id) {
                        runtime.selection = state;
                        applied += 1;
                    }
                }
            }
            SelectionMode::Add => {
                for id in &targets {
                    if let Some(runtime) = self.runtime.get_mut(*id)
                        && runtime.selection != state
                    {
                        runtime.selection = state;
                        applied += 1;
                    }
                }
            }
            SelectionMode::Clear => {
                if targets.is_empty() {
                    for runtime in self.runtime.values_mut() {
                        if !matches!(runtime.selection, SelectionState::None) {
                            runtime.selection = SelectionState::None;
                            cleared += 1;
                        }
                    }
                } else {
                    for id in &targets {
                        if let Some(runtime) = self.runtime.get_mut(*id)
                            && !matches!(runtime.selection, SelectionState::None)
                        {
                            runtime.selection = SelectionState::None;
                            cleared += 1;
                        }
                    }
                }
            }
        }

        let remaining_selected = self
            .runtime
            .values()
            .filter(|runtime| matches!(runtime.selection, SelectionState::Selected))
            .count();

        SelectionResult {
            applied,
            cleared,
            remaining_selected,
        }
    }

    fn encode_agent_id(id: AgentId) -> u64 {
        id.data().as_ffi()
    }

    fn decode_agent_id(raw: u64) -> AgentId {
        AgentId::from(KeyData::from_ffi(raw))
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
            (config.food_fertility_base - 0.2).abs() < f32::EPSILON,
            "expected default food_fertility_base to match new terrain baseline (0.2)"
        );
        assert!(
            (config.food_moisture_weight - 0.6).abs() < f32::EPSILON,
            "expected default food_moisture_weight to match design weight (0.6)"
        );
        assert!(
            (config.food_elevation_weight - 0.4).abs() < f32::EPSILON,
            "expected default food_elevation_weight to match design weight (0.4)"
        );
        assert!(
            (config.food_slope_weight - 6.0).abs() < f32::EPSILON,
            "expected default food_slope_weight to match design weight (6.0)"
        );
        assert!(
            (config.food_capacity_base - 0.3).abs() < f32::EPSILON,
            "expected default food_capacity_base to match design baseline (0.3)"
        );
        assert!(
            (config.food_capacity_fertility - 0.6).abs() < f32::EPSILON,
            "expected default food_capacity_fertility to match design scale (0.6)"
        );
        assert!(
            (config.food_growth_fertility - 0.7).abs() < f32::EPSILON,
            "expected default food_growth_fertility to match design scale (0.7)"
        );
        assert!(
            (config.food_decay_infertility - 0.5).abs() < f32::EPSILON,
            "expected default food_decay_infertility to match design scale (0.5)"
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
            (config.reproduction_fertility_bonus - 0.5).abs() < f32::EPSILON,
            "expected default reproduction_fertility_bonus to match design scale (0.5)"
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
        let WorldStateError::InvalidConfig(message) = WorldState::new(config).unwrap_err();
        assert!(
            message.contains("food_waste_rate"),
            "expected food_waste_rate validation error, got {message}"
        );
    }

    #[test]
    fn world_state_initialises_from_config() {
        let config = ScriptBotsConfig {
            initial_food: 0.25,
            rng_seed: Some(42),
            ..ScriptBotsConfig::default()
        };
        let expected_width = config.world_width;
        let mut world = WorldState::new(config).expect("world");
        assert_eq!(world.agent_count(), 0);
        assert_eq!(world.food().width(), 100);
        assert_eq!(world.food().height(), 100);
        assert_eq!(world.food().get(0, 0), Some(0.25));
        assert_eq!(world.config().world_width, expected_width);

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
            aging_tick_interval: 1,
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

    #[test]
    fn aging_respects_tick_cadence() {
        let config = ScriptBotsConfig {
            world_width: 120,
            world_height: 120,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            aging_tick_interval: 5,
            chart_flush_interval: 0,
            rng_seed: Some(11),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        world.spawn_agent(sample_agent(0));

        let mut ages = Vec::new();
        for _ in 0..10 {
            world.step();
            ages.push(world.agents().columns().ages()[0]);
        }

        assert!(ages.iter().take(4).all(|age| *age == 0));
        assert_eq!(ages[4], 1);
        assert!(ages.iter().skip(5).take(4).all(|age| *age == 1));
        assert_eq!(ages[9], 2);
    }

    #[test]
    fn chart_history_uses_cadence() {
        let config = ScriptBotsConfig {
            world_width: 150,
            world_height: 150,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            chart_flush_interval: 3,
            history_capacity: 8,
            persistence_interval: 1,
            aging_tick_interval: 1,
            rng_seed: Some(13),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        world.spawn_agent(sample_agent(0));

        let mut flushed = Vec::new();
        for _ in 0..6 {
            let events = world.step();
            if events.charts_flushed {
                flushed.push(events.tick.0);
            }
        }

        assert_eq!(flushed, vec![3, 6]);
        let history: Vec<_> = world.history().cloned().collect();
        assert_eq!(history.len(), 6);
        assert_eq!(history.first().map(|s| s.tick), Some(Tick(1)));
        assert_eq!(history.last().map(|s| s.tick), Some(Tick(6)));
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
            chart_flush_interval: 1,
            aging_tick_interval: 1,
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
            reproduction_attempt_interval: 1,
            reproduction_attempt_chance: 1.0,
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
    fn reproduction_respects_tick_cadence() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            reproduction_energy_threshold: 0.2,
            reproduction_energy_cost: 0.0,
            reproduction_cooldown: 1,
            reproduction_attempt_interval: 3,
            reproduction_attempt_chance: 1.0,
            reproduction_child_energy: 0.0,
            reproduction_spawn_jitter: 0.0,
            reproduction_color_jitter: 0.0,
            reproduction_mutation_scale: 0.0,
            reproduction_partner_chance: 0.0,
            aging_tick_interval: 1,
            chart_flush_interval: 0,
            rng_seed: Some(21),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        world.spawn_agent(sample_agent(0));

        let mut counts = Vec::new();
        for _ in 0..6 {
            world.step();
            counts.push(world.agent_count());
        }

        assert_eq!(counts, vec![1, 1, 2, 2, 2, 3]);
    }

    fn reproduction_tick_sequence(mut config: ScriptBotsConfig, steps: usize) -> Vec<u64> {
        assert!(steps > 0, "steps must be positive");
        config.history_capacity = steps.max(config.history_capacity);
        let mut world = WorldState::new(config).expect("world");
        world.spawn_agent(sample_agent(0));
        let mut ticks = Vec::new();
        let mut last_count = world.agent_count();
        for _ in 0..steps {
            let events = world.step();
            let count = world.agent_count();
            if count > last_count {
                ticks.push(events.tick.0);
            }
            last_count = count;
        }
        ticks
    }
    #[test]
    fn reproduction_gate_is_seed_deterministic() {
        let base = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            reproduction_energy_threshold: 0.2,
            reproduction_energy_cost: 0.0,
            reproduction_cooldown: 1,
            reproduction_attempt_interval: 2,
            reproduction_attempt_chance: 0.65,
            reproduction_child_energy: 0.0,
            reproduction_spawn_jitter: 0.0,
            reproduction_color_jitter: 0.0,
            reproduction_mutation_scale: 0.0,
            reproduction_partner_chance: 0.0,
            aging_tick_interval: 1,
            chart_flush_interval: 0,
            rng_seed: Some(1312),
            ..ScriptBotsConfig::default()
        };

        let ticks_a = reproduction_tick_sequence(base.clone(), 24);
        let ticks_b = reproduction_tick_sequence(base.clone(), 24);
        assert_eq!(ticks_a, ticks_b);
        assert!(!ticks_a.is_empty());
    }

    #[test]
    fn selection_updates_replace_add_and_clear() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let id_a = world.spawn_agent(sample_agent(0));
        let id_b = world.spawn_agent(sample_agent(1));

        let raw_a = id_a.data().as_ffi();
        let raw_b = id_b.data().as_ffi();

        let result = world.apply_selection_update(SelectionUpdate {
            mode: SelectionMode::Replace,
            agent_ids: vec![raw_a],
            state: SelectionState::Selected,
        });
        assert_eq!(result.applied, 1);
        assert!(matches!(
            world.agent_runtime(id_a).unwrap().selection,
            SelectionState::Selected
        ));
        assert!(matches!(
            world.agent_runtime(id_b).unwrap().selection,
            SelectionState::None
        ));

        let result = world.apply_selection_update(SelectionUpdate {
            mode: SelectionMode::Add,
            agent_ids: vec![raw_b],
            state: SelectionState::Hovered,
        });
        assert_eq!(result.applied, 1);
        assert!(matches!(
            world.agent_runtime(id_b).unwrap().selection,
            SelectionState::Hovered
        ));

        let result = world.apply_selection_update(SelectionUpdate {
            mode: SelectionMode::Clear,
            agent_ids: Vec::new(),
            state: SelectionState::Selected,
        });
        assert!(result.cleared >= 2);
        assert!(matches!(
            world.agent_runtime(id_a).unwrap().selection,
            SelectionState::None
        ));
        assert!(matches!(
            world.agent_runtime(id_b).unwrap().selection,
            SelectionState::None
        ));

        // Clearing specific ids
        world.apply_selection_update(SelectionUpdate {
            mode: SelectionMode::Add,
            agent_ids: vec![raw_a, raw_b],
            state: SelectionState::Selected,
        });
        let result = world.apply_selection_update(SelectionUpdate {
            mode: SelectionMode::Clear,
            agent_ids: vec![raw_a],
            state: SelectionState::Selected,
        });
        assert!(result.cleared >= 1);
        assert!(matches!(
            world.agent_runtime(id_a).unwrap().selection,
            SelectionState::None
        ));
        assert!(matches!(
            world.agent_runtime(id_b).unwrap().selection,
            SelectionState::Selected
        ));

        // Ensure raw conversion round-trips to live id
        let round_trip = AgentId::from(KeyData::from_ffi(raw_b));
        assert!(world.agents().contains(round_trip));
        assert_eq!(round_trip.data().as_ffi(), raw_b);
    }

    #[test]
    fn agent_debug_view_filters_by_selection_and_diet() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let id_a = world.spawn_agent(sample_agent(0));
        let id_b = world.spawn_agent(sample_agent(1));

        world.agent_runtime_mut(id_a).unwrap().herbivore_tendency = 0.8;
        world.agent_runtime_mut(id_b).unwrap().herbivore_tendency = 0.1;
        world.agent_runtime_mut(id_b).unwrap().energy = 5.0;

        world.apply_selection_update(SelectionUpdate {
            mode: SelectionMode::Replace,
            agent_ids: vec![id_a.data().as_ffi()],
            state: SelectionState::Selected,
        });

        let selected = world.agent_debug_view(AgentDebugQuery {
            selection: Some(SelectionState::Selected),
            ..AgentDebugQuery::default()
        });
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].agent_id, id_a.data().as_ffi());

        let carnivores = world.agent_debug_view(AgentDebugQuery {
            diet: Some(DietClass::Carnivore),
            sort: AgentDebugSort::EnergyDesc,
            ..AgentDebugQuery::default()
        });
        assert_eq!(carnivores.len(), 1);
        assert_eq!(carnivores[0].agent_id, id_a.data().as_ffi());

        let specific = world.agent_debug_view(AgentDebugQuery {
            ids: Some(vec![id_b.data().as_ffi()]),
            limit: Some(1),
            ..AgentDebugQuery::default()
        });
        assert_eq!(specific.len(), 1);
        assert_eq!(specific[0].agent_id, id_b.data().as_ffi());

        let nonexistent = world.agent_debug_view(AgentDebugQuery {
            ids: Some(vec![u64::MAX]),
            ..AgentDebugQuery::default()
        });
        assert!(nonexistent.is_empty());
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
            reproduction_attempt_interval: 1,
            reproduction_attempt_chance: 1.0,
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
            reproduction_attempt_interval: 1,
            reproduction_attempt_chance: 1.0,
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
        world.stage_death_cleanup(Tick::zero());

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
        world.stage_death_cleanup(Tick::zero());
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
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            rng_seed: Some(11),
            ..ScriptBotsConfig::default()
        })
        .expect("world");
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

        let profile = world.food_profiles[0];
        let nutrient_density = profile.nutrient_density;
        world.stage_food();

        let runtime = world.agent_runtime(agent).unwrap();
        let config = world.config();
        let fertility_multiplier = 1.0 + profile.fertility * config.reproduction_fertility_bonus;
        let expected_speed_scale = ((1.0_f32 - 0.0_f32).clamp(0.0, 1.0) * 0.7) + 0.3;
        let expected_intake = config.food_intake_rate * expected_speed_scale;
        let expected_energy_gain = expected_intake * (0.5 + nutrient_density * 0.5);
        assert!(
            (runtime.energy - (0.5 + expected_energy_gain)).abs() < 1e-6,
            "expected herbivore energy gain of {expected_energy_gain:.6}, got {}",
            runtime.energy - 0.5
        );
        assert!(
            (runtime.food_delta - expected_energy_gain).abs() < 1e-6,
            "expected food_delta to match energy gain ({expected_energy_gain:.6}), got {}",
            runtime.food_delta
        );
        assert!(
            (runtime.reproduction_counter
                - expected_intake * config.reproduction_food_bonus * fertility_multiplier)
                .abs()
                < 1e-6,
            "expected reproduction counter bonus of {:.6}, got {}",
            expected_intake * config.reproduction_food_bonus * fertility_multiplier,
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
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            rng_seed: Some(42),
            ..ScriptBotsConfig::default()
        })
        .expect("world");
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
    fn fertile_terrain_accelerates_regrowth() {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            rng_seed: Some(123),
            ..ScriptBotsConfig::default()
        })
        .expect("world");
        let profiles = world.food_profiles.clone();
        assert!(
            !profiles.is_empty(),
            "expected food profiles to be populated"
        );
        let (fertile_idx, infertile_idx) =
            profiles
                .iter()
                .enumerate()
                .fold((0usize, 0usize), |acc, (idx, profile)| {
                    let (fertile, infertile) = acc;
                    let fertile = if profile.fertility > profiles[fertile].fertility {
                        idx
                    } else {
                        fertile
                    };
                    let infertile = if profile.fertility < profiles[infertile].fertility {
                        idx
                    } else {
                        infertile
                    };
                    (fertile, infertile)
                });
        assert!(
            profiles[fertile_idx].fertility > profiles[infertile_idx].fertility + 0.05,
            "expected noticeable fertility variation between sampled cells"
        );

        {
            let cells = world.food_mut().cells_mut();
            cells[fertile_idx] = 0.1;
            cells[infertile_idx] = 0.1;
        }

        world.apply_food_regrowth();

        let cells = world.food().cells();
        let fertile_value = cells[fertile_idx];
        let infertile_value = cells[infertile_idx];
        assert!(
            fertile_value > infertile_value + 1e-4,
            "fertile cell should regrow faster ({} <= {})",
            fertile_value,
            infertile_value
        );
        assert!(
            fertile_value <= profiles[fertile_idx].capacity + 1e-6,
            "fertile cell should respect capacity"
        );
    }
    #[test]
    fn respawn_respects_local_capacity() {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 1,
            food_respawn_amount: 1.0,
            food_max: 1.0,
            food_capacity_base: 0.1,
            food_capacity_fertility: 0.0,
            food_growth_fertility: 0.0,
            food_decay_infertility: 0.0,
            rng_seed: Some(789),
            ..ScriptBotsConfig::default()
        })
        .expect("world");

        let width = world.food().width() as usize;
        assert!(width > 0);
        let capacity = world.food_profiles[0].capacity;
        assert!(
            capacity < world.config().food_max,
            "capacity baseline should be below global cap"
        );

        world.food_mut().cells_mut()[0] = 0.0;
        let (rx, ry) = world
            .stage_food_dynamics(Tick(1))
            .expect("respawn event expected");
        let idx = (ry as usize) * width + rx as usize;
        let capacity = world.food_profiles[idx].capacity;
        let cell_value = world.food().cells()[idx];
        assert!(
            cell_value <= capacity + 1e-6,
            "respawned value {:.6} should not exceed local capacity {:.6}",
            cell_value,
            capacity
        );
    }

    #[test]
    fn fertile_cells_boost_reproduction_from_grazing() {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            rng_seed: Some(456),
            ..ScriptBotsConfig::default()
        })
        .expect("world");
        let food_width = world.food().width() as usize;
        assert!(food_width > 0);
        let profiles = world.food_profiles.clone();
        let (fertile_idx, infertile_idx) =
            profiles
                .iter()
                .enumerate()
                .fold((0usize, 0usize), |acc, (idx, profile)| {
                    let (fertile, infertile) = acc;
                    let fertile = if profile.fertility > profiles[fertile].fertility {
                        idx
                    } else {
                        fertile
                    };
                    let infertile = if profile.fertility < profiles[infertile].fertility {
                        idx
                    } else {
                        infertile
                    };
                    (fertile, infertile)
                });
        assert!(
            profiles[fertile_idx].fertility > profiles[infertile_idx].fertility + 0.05,
            "expected noticeable fertility variation between sampled cells"
        );

        let fertile_pos = {
            let x = (fertile_idx % food_width) as f32;
            let y = (fertile_idx / food_width) as f32;
            let cell = world.config().food_cell_size as f32;
            Position::new(x * cell + cell * 0.5, y * cell + cell * 0.5)
        };
        let infertile_pos = {
            let x = (infertile_idx % food_width) as f32;
            let y = (infertile_idx / food_width) as f32;
            let cell = world.config().food_cell_size as f32;
            Position::new(x * cell + cell * 0.5, y * cell + cell * 0.5)
        };

        let fertile_agent = world.spawn_agent(sample_agent(2));
        let infertile_agent = world.spawn_agent(sample_agent(3));
        {
            let arena = world.agents_mut();
            let fertile_slot = arena.index_of(fertile_agent).unwrap();
            let infertile_slot = arena.index_of(infertile_agent).unwrap();
            let columns = arena.columns_mut();
            columns.positions_mut()[fertile_slot] = fertile_pos;
            columns.positions_mut()[infertile_slot] = infertile_pos;
        }
        for agent in [fertile_agent, infertile_agent] {
            if let Some(runtime) = world.agent_runtime_mut(agent) {
                runtime.energy = 0.5;
                runtime.reproduction_counter = 0.0;
                runtime.herbivore_tendency = 1.0;
                runtime.outputs = [0.0; OUTPUT_SIZE];
            }
        }
        {
            let cells = world.food_mut().cells_mut();
            cells[fertile_idx] = 0.2;
            cells[infertile_idx] = 0.2;
        }

        world.stage_food();

        let fertile_runtime = world.agent_runtime(fertile_agent).unwrap();
        let infertile_runtime = world.agent_runtime(infertile_agent).unwrap();

        assert!(
            fertile_runtime.energy > infertile_runtime.energy + 1e-4,
            "fertile terrain should yield more grazing energy"
        );
        assert!(
            fertile_runtime.reproduction_counter > infertile_runtime.reproduction_counter + 1e-4,
            "fertile terrain should advance reproduction counter more quickly"
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

        world.stage_death_cleanup(Tick::zero());

        let survivors: Vec<_> = world.agents().iter_handles().collect();
        assert_eq!(survivors, vec![ids[0], ids[2]]);
        assert!(world.agent_runtime(ids[1]).is_none());
        assert!(world.agent_runtime(ids[3]).is_none());
        assert_eq!(world.agent_count(), 2);
        assert!(world.pending_deaths.is_empty());
        assert_eq!(world.last_deaths, 2);
    }
}
