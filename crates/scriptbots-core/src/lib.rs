//! Core types shared across the ScriptBots workspace.

pub mod ancestry;
pub mod channels;
pub mod detect;
pub mod sense_fixed;

pub use channels::{
    BOOST_THRESHOLD, OutputChannel, OutputsExt, SENSOR_LAYOUT, SensorChannel, SensorKind,
    SensorsExt,
};

use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scriptbots_index::{NeighborhoodIndex, UniformGridIndex};
use serde::{Deserialize, Serialize};
use slotmap::{Key, KeyData, SecondaryMap, SlotMap, new_key_type};
use std::any::Any;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
use std::sync::OnceLock;
use thiserror::Error;
#[cfg(feature = "simd_wide")]
use wide::f32x4;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainActivations {
    pub layers: Vec<ActivationLayer>,
    #[serde(default)]
    pub connections: Vec<ActivationEdge>,
    /// Set when the snapshot was clipped to fit [`ACTIVATION_VALUE_BUDGET`].
    ///
    /// An inspector showing a truncated view must say so. Silently dropping
    /// layers would let a user conclude a brain has no deep structure when in
    /// fact we simply refused to copy it.
    #[serde(default)]
    pub truncated: bool,
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

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
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

impl AgentId {
    /// Stable raw key representation used by external control DTOs.
    #[must_use]
    pub fn raw(self) -> u64 {
        self.data().as_ffi()
    }
}

/// Stable logical identity for an agent within one simulation run.
///
/// Unlike [`AgentId`], this value is never recycled when a SlotMap entry is removed. Scientific
/// snapshots, lineage, replay, and persistence use this identity; the generational handle remains
/// the efficient in-memory lookup key.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AgentUid(pub u64);

impl AgentUid {
    /// Return the integer representation used by persistence and protocol DTOs.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Stable creation-order metadata attached to one live agent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentIdentity {
    pub uid: AgentUid,
    /// Monotonic ordinal assigned to every successful insertion, including injected agents.
    pub spawn_ordinal: u64,
    /// Monotonic ordinal assigned only to offspring produced from parent agents.
    pub birth_ordinal: Option<u64>,
}

/// Version of the restorable adapter state carried by [`RandomStreamState`].
pub const RANDOM_STREAM_STATE_VERSION: u16 = 1;
/// Maximum opaque state payload accepted from any random-stream implementation.
pub const MAX_RANDOM_STREAM_STATE_BYTES: usize = 256;
const SMALL_RNG_STATE_CODEC_VERSION: u16 = 1;
const SMALL_RNG_STATE_BYTES: usize = 8 + 4 * 8;

#[cfg(target_pointer_width = "64")]
const RANDOM_STREAM_ALGORITHM: &str = "rand-0.9.5-smallrng-xoshiro256plusplus-64-seed-from-u64";
#[cfg(any(target_pointer_width = "16", target_pointer_width = "32"))]
const RANDOM_STREAM_ALGORITHM: &str = "rand-0.9.5-smallrng-xoshiro128plusplus-32-seed-from-u64";

/// Serializable continuation state for the current world random stream.
///
/// `SmallRng` deliberately does not promise a portable, serializable state. This first protocol
/// therefore carries a bounded opaque payload identified by an algorithm and codec version. The
/// current adapter's private codec records its seed and four state words as explicit little-endian
/// `u64`s. Restore is constant-time and validates versions and lengths before decoding, so
/// untrusted state cannot request unbounded work. This preserves the existing generator and
/// fixed-seed behavior without constraining future adapters to xoshiro's state shape, selecting a
/// replacement generator, or claiming domain separation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RandomStreamState {
    pub version: u16,
    pub algorithm: String,
    pub codec_version: u16,
    #[serde(deserialize_with = "deserialize_bounded_random_stream_state")]
    pub state: Vec<u8>,
}

fn deserialize_bounded_random_stream_state<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct BoundedStateVisitor;

    impl<'de> serde::de::Visitor<'de> for BoundedStateVisitor {
        type Value = Vec<u8>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                formatter,
                "at most {MAX_RANDOM_STREAM_STATE_BYTES} random-stream state bytes"
            )
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let hinted = sequence.size_hint().unwrap_or_default();
            if hinted > MAX_RANDOM_STREAM_STATE_BYTES {
                return Err(serde::de::Error::invalid_length(hinted, &self));
            }
            let mut state = Vec::with_capacity(hinted);
            while let Some(byte) = sequence.next_element()? {
                if state.len() == MAX_RANDOM_STREAM_STATE_BYTES {
                    return Err(serde::de::Error::invalid_length(state.len() + 1, &self));
                }
                state.push(byte);
            }
            Ok(state)
        }
    }

    deserializer.deserialize_seq(BoundedStateVisitor)
}

/// Failure to restore a random stream state produced by another protocol or algorithm lane.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum RandomStreamRestoreError {
    #[error("unsupported random-stream state version {found}; expected {expected}")]
    UnsupportedVersion { found: u16, expected: u16 },
    #[error("unsupported random-stream algorithm `{found}`; expected `{expected}`")]
    UnsupportedAlgorithm {
        found: String,
        expected: &'static str,
    },
    #[error("unsupported random-stream codec version {found}; expected {expected}")]
    UnsupportedCodecVersion { found: u16, expected: u16 },
    #[error("random-stream state payload is {found} bytes; maximum is {maximum}")]
    StateTooLarge { found: usize, maximum: usize },
    #[error("invalid random-stream state length {found}; expected exactly {expected} bytes")]
    InvalidStateLength { found: usize, expected: usize },
    #[error("invalid all-zero random-stream state")]
    AllZeroState,
    #[cfg(any(target_pointer_width = "16", target_pointer_width = "32"))]
    #[error("random-stream state word {index} exceeds the 32-bit algorithm width: {value}")]
    StateWordOutOfRange { index: usize, value: u64 },
}

/// Object-safe random-stream protocol consumed by core and brain families.
///
/// The current world owns a [`SmallRngStream`], while consumers depend only on this checkpointable
/// interface. Named domains and scheduler-independent child streams are deliberately not part of
/// this first protocol.
pub trait RandomStream: RngCore {
    /// Stable identity of the concrete algorithm/state encoding.
    fn algorithm_id(&self) -> &'static str;

    /// Capture a versioned, serializable continuation state.
    fn checkpoint(&self) -> RandomStreamState;
}

/// Restorable adapter around the world's existing [`SmallRng`] algorithm.
///
/// The adapter implements [`RngCore`], so existing core and brain-family consumers retain the
/// exact sampling calls they make today. It is one global stream for now; named domains and
/// scheduler-independent substreams remain a later protocol decision.
#[derive(Clone, Debug)]
pub struct SmallRngStream {
    seed: u64,
    state_words: [u64; 4],
}

impl SmallRngStream {
    /// Construct the current adapter with the same `SmallRng::seed_from_u64` behavior used before
    /// this protocol existed.
    #[must_use]
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut splitmix_state = seed;
        #[cfg(target_pointer_width = "64")]
        let state_words = std::array::from_fn(|_| splitmix64(&mut splitmix_state));
        #[cfg(any(target_pointer_width = "16", target_pointer_width = "32"))]
        let state_words = {
            let first = splitmix64(&mut splitmix_state);
            let second = splitmix64(&mut splitmix_state);
            [
                first & u64::from(u32::MAX),
                first >> 32,
                second & u64::from(u32::MAX),
                second >> 32,
            ]
        };
        debug_assert!(state_words.iter().any(|word| *word != 0));
        Self { seed, state_words }
    }

    /// Stable identity of the current adapted algorithm lane.
    #[must_use]
    pub const fn algorithm() -> &'static str {
        RANDOM_STREAM_ALGORITHM
    }

    /// Capture a serializable continuation state.
    #[must_use]
    fn state(&self) -> RandomStreamState {
        let mut state = Vec::with_capacity(SMALL_RNG_STATE_BYTES);
        state.extend_from_slice(&self.seed.to_le_bytes());
        for word in self.state_words {
            state.extend_from_slice(&word.to_le_bytes());
        }
        RandomStreamState {
            version: RANDOM_STREAM_STATE_VERSION,
            algorithm: RANDOM_STREAM_ALGORITHM.to_owned(),
            codec_version: SMALL_RNG_STATE_CODEC_VERSION,
            state,
        }
    }

    /// Restore the exact continuation represented by `state` in constant time.
    ///
    /// Version, algorithm, word width, and the xoshiro all-zero forbidden state are validated
    /// before construction. An error performs no sampling and has no externally visible state.
    pub fn from_state(state: &RandomStreamState) -> Result<Self, RandomStreamRestoreError> {
        if state.version != RANDOM_STREAM_STATE_VERSION {
            return Err(RandomStreamRestoreError::UnsupportedVersion {
                found: state.version,
                expected: RANDOM_STREAM_STATE_VERSION,
            });
        }
        if state.algorithm != RANDOM_STREAM_ALGORITHM {
            return Err(RandomStreamRestoreError::UnsupportedAlgorithm {
                found: state.algorithm.clone(),
                expected: RANDOM_STREAM_ALGORITHM,
            });
        }
        if state.codec_version != SMALL_RNG_STATE_CODEC_VERSION {
            return Err(RandomStreamRestoreError::UnsupportedCodecVersion {
                found: state.codec_version,
                expected: SMALL_RNG_STATE_CODEC_VERSION,
            });
        }
        if state.state.len() > MAX_RANDOM_STREAM_STATE_BYTES {
            return Err(RandomStreamRestoreError::StateTooLarge {
                found: state.state.len(),
                maximum: MAX_RANDOM_STREAM_STATE_BYTES,
            });
        }
        if state.state.len() != SMALL_RNG_STATE_BYTES {
            return Err(RandomStreamRestoreError::InvalidStateLength {
                found: state.state.len(),
                expected: SMALL_RNG_STATE_BYTES,
            });
        }

        let seed = decode_le_u64(&state.state[0..8]);
        let state_words = std::array::from_fn(|index| {
            let start = 8 + index * 8;
            decode_le_u64(&state.state[start..start + 8])
        });
        if state_words.iter().all(|word| *word == 0) {
            return Err(RandomStreamRestoreError::AllZeroState);
        }
        #[cfg(any(target_pointer_width = "16", target_pointer_width = "32"))]
        for (index, value) in state_words.iter().copied().enumerate() {
            if value > u64::from(u32::MAX) {
                return Err(RandomStreamRestoreError::StateWordOutOfRange { index, value });
            }
        }
        Ok(Self { seed, state_words })
    }
}

#[inline]
fn decode_le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    const PHI: u64 = 0x9e37_79b9_7f4a_7c15;
    *state = state.wrapping_add(PHI);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

impl RngCore for SmallRngStream {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        #[cfg(target_pointer_width = "64")]
        {
            (self.next_u64() >> 32) as u32
        }
        #[cfg(any(target_pointer_width = "16", target_pointer_width = "32"))]
        {
            let mut words = self.state_words.map(|word| word as u32);
            let result = words[0]
                .wrapping_add(words[3])
                .rotate_left(7)
                .wrapping_add(words[0]);
            let shifted = words[1] << 9;
            words[2] ^= words[0];
            words[3] ^= words[1];
            words[1] ^= words[2];
            words[0] ^= words[3];
            words[2] ^= shifted;
            words[3] = words[3].rotate_left(11);
            self.state_words = words.map(u64::from);
            result
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        #[cfg(target_pointer_width = "64")]
        {
            let result = self.state_words[0]
                .wrapping_add(self.state_words[3])
                .rotate_left(23)
                .wrapping_add(self.state_words[0]);
            let shifted = self.state_words[1] << 17;
            self.state_words[2] ^= self.state_words[0];
            self.state_words[3] ^= self.state_words[1];
            self.state_words[1] ^= self.state_words[2];
            self.state_words[0] ^= self.state_words[3];
            self.state_words[2] ^= shifted;
            self.state_words[3] = self.state_words[3].rotate_left(45);
            result
        }
        #[cfg(any(target_pointer_width = "16", target_pointer_width = "32"))]
        {
            let low = u64::from(self.next_u32());
            let high = u64::from(self.next_u32());
            (high << 32) | low
        }
    }

    #[inline]
    fn fill_bytes(&mut self, destination: &mut [u8]) {
        #[cfg(target_pointer_width = "64")]
        {
            let (chunks, remainder) = destination.as_chunks_mut::<8>();
            for chunk in chunks {
                *chunk = self.next_u64().to_le_bytes();
            }
            if remainder.len() > 4 {
                let bytes = self.next_u64().to_le_bytes();
                remainder.copy_from_slice(&bytes[..remainder.len()]);
            } else if !remainder.is_empty() {
                let bytes = self.next_u32().to_le_bytes();
                remainder.copy_from_slice(&bytes[..remainder.len()]);
            }
        }
        #[cfg(any(target_pointer_width = "16", target_pointer_width = "32"))]
        {
            let (chunks, remainder) = destination.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = self.next_u32().to_le_bytes();
            }
            if !remainder.is_empty() {
                let bytes = self.next_u32().to_le_bytes();
                remainder.copy_from_slice(&bytes[..remainder.len()]);
            }
        }
    }
}

impl RandomStream for SmallRngStream {
    fn algorithm_id(&self) -> &'static str {
        Self::algorithm()
    }

    fn checkpoint(&self) -> RandomStreamState {
        self.state()
    }
}

/// Convenience alias for associating side data with agents.
pub type AgentMap<T> = SecondaryMap<AgentId, T>;

/// Number of sensor inputs wired into each agent brain.
pub const INPUT_SIZE: usize = 25;
/// Number of control outputs produced by each agent brain.
pub const OUTPUT_SIZE: usize = 9;
/// Number of directional eyes each agent possesses.
pub const NUM_EYES: usize = 4;

/// Maximum number of *selected* agents whose brain activations are captured per
/// tick, on top of the always-captured activation probe.
///
/// Activation snapshots allocate per-agent layer buffers (a Neuroflow brain
/// serializes its whole network), so capture must be bounded, not merely
/// demand-driven: without this cap a single "select all" from a frontend would
/// reinstate population-wide capture every tick. Inspectors show one agent at a
/// time; a small budget covers every real use while keeping the cost
/// population-independent.
pub const ACTIVATION_CAPTURE_BUDGET: usize = 8;

/// Maximum number of activation values copied out of a single brain per tick.
///
/// Bounding *how many* agents are captured is not enough: one agent's snapshot
/// must also be bounded in size. Brain topology is configuration (a Neuroflow
/// net can be declared with arbitrarily wide hidden layers), so without a cap a
/// single inspected agent could copy megabytes out of the simulation every tick.
/// Snapshots past this budget are clipped and marked
/// [`BrainActivations::truncated`] — never silently shortened.
pub const ACTIVATION_VALUE_BUDGET: usize = 4_096;

/// Clip an activation snapshot to [`ACTIVATION_VALUE_BUDGET`] values.
///
/// Whole layers are kept or dropped rather than partially copied: half a layer
/// is a lie about the shape of the network.
fn clamp_activations(mut activations: BrainActivations) -> BrainActivations {
    let mut budget = ACTIVATION_VALUE_BUDGET;
    let mut kept = Vec::with_capacity(activations.layers.len());
    let mut truncated = false;
    for layer in activations.layers {
        if layer.values.len() <= budget {
            budget -= layer.values.len();
            kept.push(layer);
        } else {
            truncated = true;
        }
    }
    activations.layers = kept;
    activations.truncated = truncated;
    if truncated {
        // Edges index into layers we may have dropped; a dangling edge would
        // paint a connection to a node the viewer cannot see.
        activations.connections.clear();
    }
    activations
}

const FULL_TURN: f32 = std::f32::consts::TAU;
const HALF_TURN: f32 = std::f32::consts::PI;
// Legacy-parity policy: World.cpp defines PI8 = π/16 and PI38 = 3 * PI8, then admits blood
// targets only for `diff4 < PI38`. Commit e2d9aaa already corrected the former accidental 3π/8
// cone on this baseline; the shared contribution function below proves its strict boundary.
const BLOOD_HALF_FOV: f32 = std::f32::consts::PI * 0.1875;

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

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
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

#[cfg(all(feature = "parallel", target_arch = "wasm32"))]
fn configure_parallelism() {
    // No-op on WASM: cannot set environment variables or configure rayon
}

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
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

/// Legacy-parity blood-sensor contribution for one target.
///
/// The boundary is deliberately strict, matching `diff4 < PI38` in `World.cpp`. Within the
/// model's valid health interval `[0, 2]`, the wound term is exactly `1 - health / 2`; clamping is
/// retained only as a defensive Rust policy for invalid state outside that interval.
fn blood_sensor_contribution(
    forward_difference: f32,
    distance_factor: f32,
    target_health: f32,
) -> f32 {
    if !(0.0..BLOOD_HALF_FOV).contains(&forward_difference) || distance_factor <= 0.0 {
        return 0.0;
    }

    let angular_factor = (BLOOD_HALF_FOV - forward_difference) / BLOOD_HALF_FOV;
    let wound_factor = (1.0 - (target_health * 0.5).clamp(0.0, 1.0)).max(0.0);
    angular_factor * distance_factor * wound_factor
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
    UpdateSimulation(SimulationCommand),
}

#[derive(Debug, Clone, Default)]
pub struct SimulationCommand {
    pub paused: Option<bool>,
    pub speed_multiplier: Option<f32>,
    pub step_once: bool,
}

impl SimulationCommand {
    /// Validate values supplied by renderer and control front-ends before queue admission.
    pub fn validate(&self) -> Result<(), WorldStateError> {
        if let Some(speed) = self.speed_multiplier
            && !speed.is_finite()
        {
            return Err(WorldStateError::InvalidConfig(
                "speed_multiplier must be finite",
            ));
        }
        Ok(())
    }
}

impl ControlCommand {
    /// Validate state-changing input without mutating the world or admitting queue work.
    pub fn validate(&self) -> Result<(), WorldStateError> {
        match self {
            Self::UpdateConfig(config) => config.validate(),
            Self::UpdateSelection(_) => Ok(()),
            Self::UpdateSimulation(command) => command.validate(),
        }
    }
}

/// Apply a control command to the world state.
pub fn apply_control_command(
    world: &mut WorldState,
    command: ControlCommand,
) -> Result<(), WorldStateError> {
    command.validate()?;
    match command {
        ControlCommand::UpdateConfig(config) => world.apply_config_update(*config),
        ControlCommand::UpdateSelection(update) => {
            world.apply_selection_update(update);
            Ok(())
        }
        ControlCommand::UpdateSimulation(update) => world.enqueue_simulation_command(update),
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Legacy's dead-zone threshold, in the SQUARED discomfort domain.
///
/// `World.cpp:95-100` computes the health drain as:
///
/// ```text
///   dd         = 2 * |pos.x / WIDTH - 0.5|      // 0 at the equator, 1 at the edges
///   discomfort = |dd - temperature_preference|
///   discomfort = discomfort * discomfort         // SQUARED FIRST
///   if (discomfort < 0.08) discomfort = 0;       // ...and the gate is on the SQUARE
///   health -= TEMPERATURE_DISCOMFORT * discomfort
/// ```
///
/// The gate is applied to the SQUARE, so in the raw domain it opens at
/// `sqrt(0.08)` — see [`DEFAULT_TEMPERATURE_COMFORT_BAND`]. Porting the literal
/// `0.08` into a raw-domain comparison, as this codebase did, silently shrinks
/// the comfort zone by a factor of ~3.5 and drains health from agents legacy
/// considered perfectly comfortable. Same magic number; different meaning.
pub const LEGACY_COMFORT_BAND_SQUARED: f32 = 0.08;

/// The comfort band, in the RAW discomfort domain: `sqrt(0.08)`.
///
/// This is the width legacy actually intended (see [`LEGACY_COMFORT_BAND_SQUARED`]).
///
/// # Parity versus policy — the decision, stated once
///
/// Two things differ from legacy, and only one of them is deliberate:
///
/// - **Deliberate policy.** Beyond the band, the drain scales with the EXCESS
///   discomfort (`(d - band)^exponent`), not the full discomfort. Legacy jumps
///   discontinuously from zero to `0.08 * rate` the instant the gate opens. That
///   cliff is an artefact of writing the gate as an `if`, not a modelling
///   intent, and a step change in a health drain puts a sharp, arbitrary
///   selection boundary in the middle of the temperature axis. The continuous
///   ramp is kept.
/// - **A defect, now fixed.** The band CONSTANT was ported across domains
///   unchanged, which is the bug documented above.
///
/// # The contract
///
/// - **Units.** Temperature and preference are both dimensionless, in `[0, 1]`.
///   `0` is the equator (the middle of the world), `1` is the east/west edges.
/// - **Formula.** `T(x) = (2 * |x/W - 0.5|) ^ gradient_exponent`, clamped to
///   `[0, 1]`. The default exponent of `1.0` is exactly legacy's linear gradient.
/// - **Cadence.** Once per tick, in `stage_temperature_discomfort`.
/// - **RNG inputs: NONE.** Temperature is a pure function of position and config.
///   This is a property worth protecting: a climate that consumed entropy would
///   make every run's weather a function of how many other draws happened first.
/// - **Bounds.** Every intermediate and every result is finite; a degenerate
///   (zero-width) world reports a uniform `0.5` rather than dividing by zero.
/// - **Default drain rate is 0.0**, exactly as in legacy (`TEMPERATURE_DISCOMFORT
///   = 0`), so the whole mechanism is inert unless a scenario turns it on — which
///   is why correcting the band moves no default digest.
pub const DEFAULT_TEMPERATURE_COMFORT_BAND: f32 = 0.282_842_7;

/// Legacy's health drain, transcribed exactly from `World.cpp:95-100`.
///
/// The micro-oracle. It exists to be DIFFERENT from the code under test where we
/// have chosen to differ, and identical where we have not — a parity claim that
/// cannot be checked against the thing it claims parity with is just an assertion.
#[must_use]
pub fn legacy_temperature_health_drain(
    normalized_x: f32,
    preference: f32,
    discomfort_rate: f32,
) -> f32 {
    let dd = 2.0 * (normalized_x - 0.5).abs();
    let mut discomfort = (dd - preference).abs();
    discomfort *= discomfort;
    if discomfort < LEGACY_COMFORT_BAND_SQUARED {
        discomfort = 0.0;
    }
    discomfort_rate * discomfort
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
    /// Transient generational lookup handle encoded for live control requests.
    pub agent_id: u64,
    /// Stable logical identity used by scientific records and lineage.
    pub agent_uid: AgentUid,
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
    pub fn from_registry(
        registry: &BrainRegistry,
        rng: &mut dyn RandomStream,
        key: u64,
    ) -> Result<Option<Self>, BrainSpawnError> {
        let Some(runner) = registry.spawn(rng, key)? else {
            return Ok(None);
        };
        let kind = registry.kind(key).map(str::to_string);
        Ok(Some(Self {
            runner: Some(runner),
            registry_key: Some(key),
            kind,
        }))
    }

    /// Attach an inherited runner while preserving the family registry key,
    /// so later generations can still fall back to the registry factory.
    #[must_use]
    pub fn inherited(runner: Box<dyn BrainRunner>, registry_key: Option<u64>) -> Self {
        let kind = Some(runner.kind().to_string());
        Self {
            runner: Some(runner),
            registry_key,
            kind,
        }
    }

    /// Borrow the live runner, if any.
    #[must_use]
    pub fn runner(&self) -> Option<&dyn BrainRunner> {
        self.runner.as_deref()
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

    /// Duplicate this runner including all evolved parameters.
    ///
    /// `None` marks the family as non-heritable; reproduction then falls back
    /// to spawning a fresh runner from the registry. An exact-snapshot failure
    /// is returned as an error and must never be normalized into `None`.
    fn clone_runner(&self) -> Result<Option<Box<dyn BrainRunner>>, BrainSpawnError> {
        Ok(None)
    }

    /// Perturb parameters in place using the agent's mutation rates.
    fn mutate(
        &mut self,
        _rng: &mut dyn RandomStream,
        _rate: f32,
        _scale: f32,
    ) -> Result<(), BrainSpawnError> {
        Ok(())
    }

    /// Produce an offspring runner by recombining with a same-kind partner.
    fn crossover(
        &self,
        _partner: &dyn BrainRunner,
        _rng: &mut dyn RandomStream,
    ) -> Option<Box<dyn BrainRunner>> {
        None
    }

    /// Downcast hook so cross-crate `crossover` implementations can identify
    /// same-family partners behind the trait object.
    fn as_any(&self) -> Option<&(dyn Any + Send + Sync)> {
        None
    }
}

/// Typed failure returned by a registered brain factory.
#[derive(Debug, Clone)]
pub struct BrainSpawnError {
    kind: Cow<'static, str>,
    source: std::sync::Arc<dyn std::error::Error + Send + Sync>,
}

impl std::fmt::Display for BrainSpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "brain factory `{}` failed: {}", self.kind, self.source)
    }
}

impl std::error::Error for BrainSpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

impl BrainSpawnError {
    /// Attach a concrete adapter error to its registered brain-family label.
    pub fn new<E>(kind: impl Into<Cow<'static, str>>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            kind: kind.into(),
            source: std::sync::Arc::new(source),
        }
    }

    /// Brain-family label whose factory failed.
    #[must_use]
    pub fn kind(&self) -> &str {
        self.kind.as_ref()
    }
}

#[derive(Debug, Error)]
#[error("brain registry key {key} disappeared before offspring construction")]
struct MissingBrainFactory {
    key: u64,
}

#[derive(Debug, Error)]
#[error("bound parent has no exact heritable snapshot and no registry fallback")]
struct MissingHeritableBrain;

type BrainSpawner = Box<
    dyn Fn(&mut dyn RandomStream) -> Result<Box<dyn BrainRunner>, BrainSpawnError>
        + Send
        + Sync
        + 'static,
>;

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

    /// Registers a fallible brain factory, returning its registry key.
    pub fn register<F>(&mut self, kind: impl Into<Cow<'static, str>>, factory: F) -> u64
    where
        F: Fn(&mut dyn RandomStream) -> Result<Box<dyn BrainRunner>, BrainSpawnError>
            + Send
            + Sync
            + 'static,
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
    pub fn spawn(
        &self,
        rng: &mut dyn RandomStream,
        key: u64,
    ) -> Result<Option<Box<dyn BrainRunner>>, BrainSpawnError> {
        self.entries
            .get(&key)
            .map_or(Ok(None), |entry| (entry.spawner)(rng).map(Some))
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

    /// Return registered brain keys and kind labels in stable key order.
    #[must_use]
    pub fn descriptors(&self) -> Vec<(u64, String)> {
        let mut descriptors: Vec<_> = self
            .entries
            .iter()
            .map(|(key, entry)| (*key, entry.kind.to_string()))
            .collect();
        descriptors.sort_unstable_by_key(|(key, _)| *key);
        descriptors
    }

    /// Pick a random registered brain key, if any.
    pub fn random_key(&self, rng: &mut dyn RandomStream) -> Option<u64> {
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
    pub lineage: [Option<AgentUid>; 2],
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
    /// Validate every floating-point runtime field before committing an external update.
    pub fn validate(&self) -> Result<(), ScientificStateError> {
        self.validate_at("runtime")
    }

    fn validate_at(&self, path: &str) -> Result<(), ScientificStateError> {
        validate_finite(&format!("{path}.energy"), self.energy)?;
        validate_finite(
            &format!("{path}.reproduction_counter"),
            self.reproduction_counter,
        )?;
        validate_finite(
            &format!("{path}.herbivore_tendency"),
            self.herbivore_tendency,
        )?;
        validate_finite(
            &format!("{path}.mutation_rates.primary"),
            self.mutation_rates.primary,
        )?;
        validate_finite(
            &format!("{path}.mutation_rates.secondary"),
            self.mutation_rates.secondary,
        )?;
        for (name, value) in [
            ("smell", self.trait_modifiers.smell),
            ("sound", self.trait_modifiers.sound),
            ("hearing", self.trait_modifiers.hearing),
            ("eye", self.trait_modifiers.eye),
            ("blood", self.trait_modifiers.blood),
        ] {
            validate_finite(&format!("{path}.trait_modifiers.{name}"), value)?;
        }
        validate_finite_slice(&format!("{path}.clocks"), &self.clocks)?;
        validate_finite_slice(&format!("{path}.eye_fov"), &self.eye_fov)?;
        validate_finite_slice(&format!("{path}.eye_direction"), &self.eye_direction)?;
        validate_finite(&format!("{path}.sound_multiplier"), self.sound_multiplier)?;
        validate_finite(&format!("{path}.give_intent"), self.give_intent)?;
        validate_finite_slice(&format!("{path}.sensors"), &self.sensors)?;
        validate_finite_slice(&format!("{path}.outputs"), &self.outputs)?;
        validate_finite(
            &format!("{path}.indicator.intensity"),
            self.indicator.intensity,
        )?;
        validate_finite_slice(&format!("{path}.indicator.color"), &self.indicator.color)?;
        validate_finite(&format!("{path}.food_delta"), self.food_delta)?;
        validate_finite(&format!("{path}.sound_output"), self.sound_output)?;
        validate_finite(
            &format!("{path}.temperature_preference"),
            self.temperature_preference,
        )?;
        validate_finite(
            &format!("{path}.food_balance_total"),
            self.food_balance_total,
        )
    }

    /// Sample randomized sensory parameters matching the legacy ScriptBots defaults.
    pub fn new_random(rng: &mut dyn RandomStream) -> Self {
        let mut runtime = Self::default();
        runtime.randomize_spawn(rng);
        runtime
    }

    /// Randomize spawn-time traits and sensory configuration.
    pub fn randomize_spawn(&mut self, rng: &mut dyn RandomStream) {
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
    pub identity: AgentIdentity,
    pub data: AgentData,
    pub runtime: AgentRuntime,
}

/// Schema identifier for the temporary pre-redesign world characterization digest.
pub const CHARACTERIZATION_DIGEST_V0_SCHEMA: &str = "scriptbots.world.characterization.v0";

/// Stable, non-cryptographic fingerprint of the deterministic world fields available today.
///
/// Version zero is intentionally a characterization aid rather than a replay guarantee. It
/// captures boundary-visible world data and a non-mutating RNG probe. A restorable random-stream
/// protocol now exists, but V0 deliberately retains its historical probe-only field and raw-handle
/// ordering so its characterization fixtures do not masquerade as the later canonical digest.
/// Live brain-runner state, persistence sinks, and registered factory closures also remain outside
/// V0. [`WorldState::characterization_digest_v0`] documents the complete inclusion/exclusion
/// contract. A future full-state digest must use a new schema version.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CharacterizationDigestV0 {
    pub schema: String,
    pub algorithm: String,
    pub tick: Tick,
    pub overall: String,
    pub agents: String,
    pub food: String,
    pub terrain: String,
    pub hydrology: Option<String>,
    pub rng_probe: String,
    pub brain_registry: String,
}

/// Compile-lane identity needed to interpret a characterization digest honestly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CoreBuildIdentityV0 {
    pub parallel: bool,
    pub simd_wide: bool,
    pub rayon_threads: usize,
    pub target_arch: String,
    pub target_os: String,
    pub target_family: String,
    pub target_endian: String,
    pub pointer_width: u8,
}

impl CoreBuildIdentityV0 {
    #[must_use]
    pub fn current() -> Self {
        #[cfg(feature = "parallel")]
        let rayon_threads = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let rayon_threads = 1;

        Self {
            parallel: cfg!(feature = "parallel"),
            simd_wide: cfg!(feature = "simd_wide"),
            rayon_threads,
            target_arch: std::env::consts::ARCH.to_owned(),
            target_os: std::env::consts::OS.to_owned(),
            target_family: std::env::consts::FAMILY.to_owned(),
            target_endian: if cfg!(target_endian = "little") {
                "little".to_owned()
            } else {
                "big".to_owned()
            },
            pointer_width: u8::try_from(usize::BITS).unwrap_or(u8::MAX),
        }
    }
}

/// Failures encountered while characterizing a world boundary.
#[derive(Debug, Error)]
pub enum CharacterizationError {
    /// V0 is defined only between ticks, with no queued stage/control work.
    #[error(
        "world is not at a quiescent boundary (pending deaths: {pending_deaths}, pending spawns: {pending_spawns}, simulation commands: {simulation_commands})"
    )]
    NonQuiescent {
        pending_deaths: usize,
        pending_spawns: usize,
        simulation_commands: usize,
    },
    /// An arena handle did not resolve to its dense scalar state.
    #[error("agent {agent_id} is missing dense scalar state")]
    MissingAgentData { agent_id: u64 },
    /// An arena handle did not have matching runtime state.
    #[error("agent {agent_id} is missing runtime state")]
    MissingAgentRuntime { agent_id: u64 },
}

fn characterization_fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

struct CharacterizationEncoderV0 {
    hash: u64,
}

impl CharacterizationEncoderV0 {
    fn new(domain: &str) -> Self {
        let mut encoder = Self {
            hash: characterization_fnv1a64(&[]),
        };
        encoder.string(CHARACTERIZATION_DIGEST_V0_SCHEMA);
        encoder.string(domain);
        encoder
    }

    fn raw(&mut self, bytes: &[u8]) {
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        for &byte in bytes {
            self.hash ^= u64::from(byte);
            self.hash = self.hash.wrapping_mul(PRIME);
        }
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u8(&mut self, value: u8) {
        self.raw(&[value]);
    }

    fn u16(&mut self, value: u16) {
        self.raw(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(u64::try_from(value).unwrap_or(u64::MAX));
    }

    fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    fn string(&mut self, value: &str) {
        self.usize(value.len());
        self.raw(value.as_bytes());
    }

    fn option_u64(&mut self, value: Option<u64>) {
        self.bool(value.is_some());
        if let Some(value) = value {
            self.u64(value);
        }
    }

    fn option_agent_uid(&mut self, value: Option<AgentUid>) {
        self.option_u64(value.map(AgentUid::get));
    }

    fn option_string(&mut self, value: Option<&str>) {
        self.bool(value.is_some());
        if let Some(value) = value {
            self.string(value);
        }
    }

    fn finish(self) -> String {
        format!("{:016x}", self.hash)
    }
}

fn encode_agent_data_v0(encoder: &mut CharacterizationEncoderV0, data: AgentData) {
    encoder.f32(data.position.x);
    encoder.f32(data.position.y);
    encoder.f32(data.velocity.vx);
    encoder.f32(data.velocity.vy);
    encoder.f32(data.heading);
    encoder.f32(data.health);
    for value in data.color {
        encoder.f32(value);
    }
    encoder.f32(data.spike_length);
    encoder.bool(data.boost);
    encoder.u32(data.age);
    encoder.u32(data.generation.0);
}

fn encode_agent_runtime_v0(encoder: &mut CharacterizationEncoderV0, runtime: &AgentRuntime) {
    encoder.f32(runtime.energy);
    encoder.f32(runtime.reproduction_counter);
    encoder.f32(runtime.herbivore_tendency);
    encoder.f32(runtime.mutation_rates.primary);
    encoder.f32(runtime.mutation_rates.secondary);
    encoder.f32(runtime.trait_modifiers.smell);
    encoder.f32(runtime.trait_modifiers.sound);
    encoder.f32(runtime.trait_modifiers.hearing);
    encoder.f32(runtime.trait_modifiers.eye);
    encoder.f32(runtime.trait_modifiers.blood);
    for value in runtime.clocks {
        encoder.f32(value);
    }
    for value in runtime.eye_fov {
        encoder.f32(value);
    }
    for value in runtime.eye_direction {
        encoder.f32(value);
    }
    encoder.f32(runtime.sound_multiplier);
    encoder.f32(runtime.give_intent);
    for value in runtime.sensors {
        encoder.f32(value);
    }
    for value in runtime.outputs {
        encoder.f32(value);
    }
    encoder.f32(runtime.food_delta);
    encoder.bool(runtime.spiked);
    encoder.bool(runtime.hybrid);
    encoder.f32(runtime.sound_output);
    encoder.f32(runtime.temperature_preference);
    for parent in runtime.lineage {
        encoder.option_agent_uid(parent);
    }
    encoder.option_u64(runtime.brain.registry_key());
    encoder.option_string(runtime.brain.kind());
    encoder.bool(runtime.brain.is_bound());
    encoder.f32(runtime.food_balance_total);
}

const fn terrain_kind_tag_v0(kind: TerrainKind) -> u8 {
    match kind {
        TerrainKind::DeepWater => 0,
        TerrainKind::ShallowWater => 1,
        TerrainKind::Sand => 2,
        TerrainKind::Grass => 3,
        TerrainKind::Bloom => 4,
        TerrainKind::Rock => 5,
    }
}

const fn map_generator_tag_v0(generator: MapGeneratorKind) -> u8 {
    match generator {
        MapGeneratorKind::RuleBased => 0,
    }
}

const fn hydrology_flow_tag_v0(direction: HydrologyFlowDirection) -> u8 {
    match direction {
        HydrologyFlowDirection::None => 0,
        HydrologyFlowDirection::North => 1,
        HydrologyFlowDirection::South => 2,
        HydrologyFlowDirection::East => 3,
        HydrologyFlowDirection::West => 4,
    }
}

#[derive(Debug, Clone)]
struct ActuationDelta {
    heading: f32,
    velocity: Velocity,
    position: Position,
    health_delta: f32,
}

/// One neighbour's share of what an agent currently perceives.
///
/// Every field is the *delta this neighbour added*, in the same units the
/// sensor vector uses before clamping.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorContribution {
    /// Transient live handle for the neighbour responsible.
    pub source: AgentId,
    /// Stable logical identity used for scientific attribution and deterministic ties.
    pub source_uid: AgentUid,
    /// Bearing from the observer's heading, in radians, wrapped to [-pi, pi].
    pub bearing: f32,
    /// Toroidal distance to the neighbour.
    pub distance: f32,
    /// The neighbour's body colour, as the observer sees it.
    pub color: [f32; 3],
    /// Density added to each eye.
    pub eye_density: [f32; NUM_EYES],
    /// Red/green/blue added to each eye.
    pub eye_rgb: [[f32; 3]; NUM_EYES],
    /// Smell added (pre trait multiplier).
    pub smell: f32,
    /// Movement noise added (pre trait multiplier).
    pub sound: f32,
    /// Deliberate signal added (pre trait multiplier).
    pub hearing: f32,
    /// Blood added (pre trait multiplier).
    pub blood: f32,
    /// Ranking key: total sensory energy this neighbour contributed.
    pub total: f32,
}

/// What an agent perceives right now, and *why*.
///
/// # Which tick does this describe?
///
/// It describes the world **as it stands**, i.e. what the agent's next
/// sensing pass will see. It deliberately does NOT try to reproduce
/// `AgentRuntime::sensors`, because those were computed in `stage_sense`
/// from the positions agents held *before* actuation moved them — the world
/// that produced them no longer exists. Claiming to explain a vector while
/// silently using different positions would be the worst possible outcome:
/// an explanation that looks authoritative and is wrong.
///
/// The honest contract, which the tests enforce, is: step the world and the
/// next `runtime.sensors` will match [`SensorAttribution::clamped`].
#[derive(Debug, Clone, PartialEq)]
pub struct SensorAttribution {
    /// Observer.
    pub agent: AgentId,
    /// Tick the attribution was taken at.
    pub tick: Tick,
    /// Sensor values *before* clamping. Contributions sum to these.
    pub raw: [f32; INPUT_SIZE],
    /// Sensor values after clamping — what a brain actually receives.
    pub clamped: [f32; INPUT_SIZE],
    /// Channels whose raw value exceeded the clamp.
    ///
    /// This mask is not a nicety. Contributions routinely sum above 1.0, so
    /// a panel listing contributors totalling 2.4 beside a displayed value
    /// of 1.0 looks broken — and the "fix" someone reaches for is to
    /// normalise the contributors, which destroys the information. Saying
    /// "saturated" is the difference between a confusing panel and a
    /// truthful one.
    pub saturated: [bool; INPUT_SIZE],
    /// Contributing neighbours, strongest first. Bounded.
    pub contributions: Vec<SensorContribution>,
    /// How many contributors were dropped to honour the bound.
    pub truncated: usize,
}

/// The admissible range of one externally-settable configuration knob.
///
/// # Why ranges exist at all
///
/// `ScriptBotsConfig::validate` checks *admissibility* — finite, non-negative,
/// non-zero where required — but it has no upper bounds. `food_regrowth_rate =
/// 1e9` passes today, from the REST API, from MCP, and therefore from any agent
/// driving them. The safety story for an autonomous experimenter ("a confused
/// model can only request what a human could request") is not true until
/// somebody writes the ranges down. This is that list.
///
/// The bounds are deliberately GENEROUS: they exist to reject the absurd, not to
/// enforce taste. A researcher must still be able to build a hostile world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KnobRange {
    /// Dotted knob path, as produced by flattening the config.
    pub path: &'static str,
    /// Smallest accepted value.
    pub min: f64,
    /// Largest accepted value.
    pub max: f64,
    /// Whether changing this knob requires a fresh world.
    ///
    /// Dimensions cannot be changed on a live world (`apply_config_update`
    /// rejects them), so a sweep over them has to construct new worlds. An
    /// experiment planner that does not know this will generate specs that are
    /// rejected at apply time, one run at a time, and blame the simulation.
    pub fresh_world_only: bool,
}

impl KnobRange {
    const fn live(path: &'static str, min: f64, max: f64) -> Self {
        Self {
            path,
            min,
            max,
            fresh_world_only: false,
        }
    }

    const fn fresh(path: &'static str, min: f64, max: f64) -> Self {
        Self {
            path,
            min,
            max,
            fresh_world_only: true,
        }
    }
}

/// Admissible ranges for the numeric knobs external surfaces may set.
///
/// A knob absent from this table is not range-checked; it is still subject to
/// `ScriptBotsConfig::validate`. Absence is a gap, not a licence — add ranges as
/// knobs are exposed.
pub const KNOB_RANGES: &[KnobRange] = &[
    // World geometry: fresh worlds only.
    KnobRange::fresh("world_width", 32.0, 20_000.0),
    KnobRange::fresh("world_height", 32.0, 20_000.0),
    KnobRange::fresh("food_cell_size", 1.0, 500.0),
    // Food economy.
    KnobRange::live("food_max", 0.001, 100.0),
    KnobRange::live("initial_food", 0.0, 100.0),
    KnobRange::live("food_growth_rate", 0.0, 1.0),
    KnobRange::live("food_decay_rate", 0.0, 1.0),
    KnobRange::live("food_diffusion_rate", 0.0, 0.25),
    KnobRange::live("food_intake_rate", 0.0, 1.0),
    KnobRange::live("food_waste_rate", 0.0, 1.0),
    KnobRange::live("food_transfer_rate", 0.0, 1.0),
    // Metabolism and locomotion.
    KnobRange::live("metabolism_drain", 0.0, 1.0),
    KnobRange::live("movement_drain", 0.0, 1.0),
    KnobRange::live("bot_speed", 0.0, 100.0),
    KnobRange::live("boost_multiplier", 1.0, 50.0),
    KnobRange::live("sense_radius", 1.0, 5_000.0),
    // Combat.
    KnobRange::live("spike_damage", 0.0, 10.0),
    KnobRange::live("spike_radius", 0.0, 1_000.0),
    KnobRange::live("spike_energy_cost", 0.0, 10.0),
    // Climate.
    KnobRange::live("temperature_discomfort_rate", 0.0, 10.0),
    // Evolution.
    KnobRange::live("mutation.primary", 0.0, 1.0),
    KnobRange::live("mutation.secondary", 0.0, 10.0),
    KnobRange::live("reproduction_energy_threshold", 0.0, 2.0),
    KnobRange::live("reproduction_energy_cost", 0.0, 2.0),
    // Population.
    KnobRange::live("population_minimum", 0.0, 100_000.0),
    KnobRange::live("population_spawn_interval", 0.0, 1_000_000.0),
];

/// One rejected knob assignment.
#[derive(Debug, Clone, PartialEq)]
pub struct KnobViolation {
    /// Which knob.
    pub path: String,
    /// What was requested.
    pub value: f64,
    /// Smallest accepted value.
    pub min: f64,
    /// Largest accepted value.
    pub max: f64,
}

impl fmt::Display for KnobViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} = {} is outside the accepted range [{}, {}]",
            self.path, self.value, self.min, self.max
        )
    }
}

/// Check a flattened knob assignment against [`KNOB_RANGES`].
///
/// Reports EVERY violation rather than the first: a caller fixing a rejected
/// experiment one knob per round trip is a caller who gives up, and an
/// autonomous one burns its whole budget doing it.
///
/// Non-finite values are always rejected, even for unlisted knobs — a `NaN`
/// silently poisons every downstream reduction it touches.
#[must_use]
pub fn check_knob_ranges(assignments: &[(String, f64)]) -> Vec<KnobViolation> {
    let mut violations = Vec::new();
    for (path, value) in assignments {
        if !value.is_finite() {
            violations.push(KnobViolation {
                path: path.clone(),
                value: *value,
                min: f64::NEG_INFINITY,
                max: f64::INFINITY,
            });
            continue;
        }
        if let Some(range) = KNOB_RANGES.iter().find(|range| range.path == path)
            && (*value < range.min || *value > range.max)
        {
            violations.push(KnobViolation {
                path: path.clone(),
                value: *value,
                min: range.min,
                max: range.max,
            });
        }
    }
    violations
}

/// Look up a knob's declared range, if it has one.
#[must_use]
pub fn knob_range(path: &str) -> Option<&'static KnobRange> {
    KNOB_RANGES.iter().find(|range| range.path == path)
}

/// A region of the world, measured with the same toroidal metric the simulation
/// uses everywhere else.
///
/// The world is a torus. A region that measures distance naively selects the
/// wrong agents near the seam — an agent at x=5 and one at x=995 in a 1000-wide
/// world are 10 apart, not 990 — and the bug is invisible until someone drops a
/// meteor near an edge and watches the wrong things die.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum Region {
    /// The entire world.
    All,
    /// A disc, wrapped across the seam.
    Disc {
        /// Centre x.
        x: f32,
        /// Centre y.
        y: f32,
        /// Radius.
        radius: f32,
    },
}

impl Region {
    /// Whether a point lies inside this region on the torus.
    #[must_use]
    pub fn contains(&self, px: f32, py: f32, world_width: f32, world_height: f32) -> bool {
        match *self {
            Self::All => true,
            Self::Disc { x, y, radius } => {
                let dx = toroidal_delta(px, x, world_width);
                let dy = toroidal_delta(py, y, world_height);
                dx.mul_add(dx, dy * dy) <= radius * radius
            }
        }
    }

    fn validate(&self) -> Result<(), WorldStateError> {
        match *self {
            Self::All => Ok(()),
            Self::Disc { x, y, radius } => {
                if !x.is_finite() || !y.is_finite() || !radius.is_finite() || radius <= 0.0 {
                    return Err(WorldStateError::InvalidConfig(
                        "disc region needs finite coordinates and a positive radius",
                    ));
                }
                Ok(())
            }
        }
    }
}

/// A deliberate perturbation of the world.
///
/// Interventions are the difference between watching an ecosystem and doing
/// experiments on one. They are also the actuator the paired intervention
/// studies need: that bead cannot run a drought study if nothing can cause a
/// drought.
///
/// Every intervention is QUEUED and applied inside the tick loop, never straight
/// from a host thread. An intervention applied immediately lands on whichever
/// tick the world mutex happened to be free, which is nondeterminism walking in
/// through the front door — and it would make a "replayable session" a lie.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Intervention {
    /// Suppress food regrowth in a region for a while.
    Drought {
        /// Where.
        region: Region,
        /// How long, in ticks.
        ticks: u32,
        /// Multiplier applied to the growth rate (0 = total).
        growth_scale: f32,
    },
    /// Add food to every cell in a region, right now.
    Bloom {
        /// Where.
        region: Region,
        /// How much to add per cell.
        amount: f32,
    },
    /// Damage every agent in a region and scorch the food under them.
    Meteor {
        /// Where.
        region: Region,
        /// Health removed from each agent inside.
        lethality: f32,
        /// Fraction of each cell's food destroyed, in `[0, 1]`.
        scorch: f32,
    },
}

impl Intervention {
    /// Reject an intervention that cannot be honoured, rather than clamping it
    /// into a different experiment than the one that was asked for.
    ///
    /// # Errors
    ///
    /// Returns [`WorldStateError::InvalidConfig`] for a non-finite or negative
    /// magnitude, or an unusable region.
    pub fn validate(&self) -> Result<(), WorldStateError> {
        match *self {
            Self::Drought {
                region,
                growth_scale,
                ..
            } => {
                region.validate()?;
                if !(0.0..=1.0).contains(&growth_scale) {
                    return Err(WorldStateError::InvalidConfig(
                        "drought growth_scale must lie in [0, 1]",
                    ));
                }
                Ok(())
            }
            Self::Bloom { region, amount } => {
                region.validate()?;
                if !amount.is_finite() || amount < 0.0 {
                    return Err(WorldStateError::InvalidConfig(
                        "bloom amount must be finite and non-negative",
                    ));
                }
                Ok(())
            }
            Self::Meteor {
                region,
                lethality,
                scorch,
            } => {
                region.validate()?;
                if !lethality.is_finite() || lethality < 0.0 {
                    return Err(WorldStateError::InvalidConfig(
                        "meteor lethality must be finite and non-negative",
                    ));
                }
                if !(0.0..=1.0).contains(&scorch) {
                    return Err(WorldStateError::InvalidConfig(
                        "meteor scorch must lie in [0, 1]",
                    ));
                }
                Ok(())
            }
        }
    }
}

/// A timed intervention still in force.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActiveEffect {
    /// Where it applies.
    pub region: Region,
    /// Ticks left before it lapses.
    pub ticks_remaining: u32,
    /// Growth multiplier applied inside the region.
    pub growth_scale: f32,
}

#[derive(Debug, Clone, Default)]
struct ActuationResult {
    delta: Option<ActuationDelta>,
    energy: f32,
    drain: ActuationDrain,
    color: [f32; 3],
    spike_length: f32,
    sound_level: f32,
    give_intent: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct ActuationDrain {
    basal: f32,
    movement: f32,
    ramp: f32,
    boost: f32,
    topography: f32,
}

impl ActuationDrain {
    fn total(self) -> f32 {
        self.basal + self.movement + self.ramp + self.boost + self.topography
    }
}

/// Compact per-agent copy of the only runtime fields combat reads; cloning
/// whole `AgentRuntime`s (logs, sensor arrays) per tick dwarfed the stage.
#[derive(Debug, Clone, Copy, Default)]
struct CombatAgentView {
    herbivore_tendency: f32,
    energy: f32,
    outputs: [f32; OUTPUT_SIZE],
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

#[derive(Debug, Clone, Copy, Default)]
struct FoodResourceActivity {
    shared_energy: f64,
    sharing_delta_energy: f64,
    rejected_energy: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct DeathResourceActivity {
    carcass_delta: ResourceAmounts,
    removal_delta: ResourceAmounts,
    rejected: ResourceAmounts,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SpawnOrder {
    parent_index: usize,
    parent_id: AgentId,
    partner_id: Option<AgentId>,
    parent_energy_before_debit: f32,
    parent_reproduction_counter_before_reset: f32,
    data: AgentData,
    runtime: AgentRuntime,
}

struct PopulationSpawnReceipt {
    inserted: Vec<AgentId>,
    arena_checkpoint: (SlotMap<AgentId, usize>, usize),
    rng_before: SmallRngStream,
    next_agent_uid_before: u64,
    next_spawn_ordinal_before: u64,
    next_birth_ordinal_before: u64,
}

/// Absolute error floor used by resource-ledger reconciliation.
///
/// Resource state is stored as `f32`, while the ledger accumulates observations
/// as `f64`. The floor covers the rounding introduced when independently
/// summing many `f32` cells and agents at adjacent stage boundaries.
pub const RESOURCE_LEDGER_ABSOLUTE_TOLERANCE: f64 = 1.0e-5;

/// Scale-dependent part of the resource-ledger reconciliation tolerance.
pub const RESOURCE_LEDGER_RELATIVE_TOLERANCE: f64 = 1.0e-6;

/// The three conserved/accounted resource pools in the simulation kernel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceAmounts {
    /// Food stored in the ground grid.
    pub food: f64,
    /// Energy held by living agents.
    pub energy: f64,
    /// Health held by living agents.
    pub health: f64,
}

impl ResourceAmounts {
    fn delta_from(self, before: Self) -> Self {
        Self {
            food: self.food - before.food,
            energy: self.energy - before.energy,
            health: self.health - before.health,
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.food += other.food;
        self.energy += other.energy;
        self.health += other.health;
    }

    fn subtract(self, other: Self) -> Self {
        Self {
            food: self.food - other.food,
            energy: self.energy - other.energy,
            health: self.health - other.health,
        }
    }

    fn scale(self) -> f64 {
        self.food
            .abs()
            .max(self.energy.abs())
            .max(self.health.abs())
    }

    fn within(self, tolerance: f64) -> bool {
        self.food.abs() <= tolerance
            && self.energy.abs() <= tolerance
            && self.health.abs() <= tolerance
    }
}

/// Stable attribution categories for every food, energy, and health mutation.
///
/// Positive deltas add a resource to the measured world; negative deltas
/// remove it. Transfers additionally publish a positive `activity` magnitude
/// even when their world-wide delta is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceFlowKind {
    /// Queued bloom and meteor interventions.
    ScenarioIntervention,
    /// Scheduled food respawn, growth, decay, and diffusion as one exact field update.
    FoodDynamics,
    /// Age-dependent health and energy decay.
    Aging,
    /// Baseline per-tick metabolism.
    BasalMetabolism,
    /// Wheel-speed-dependent movement cost.
    Movement,
    /// Energy-level-dependent metabolism ramp.
    MetabolismRamp,
    /// Additional boost cost.
    Boost,
    /// Uphill cost or downhill relief.
    Topography,
    /// Temperature mismatch cost.
    TemperatureStress,
    /// Ground-food removal and nutrient-weighted energy conversion.
    GroundFoodConversion,
    /// Agent-to-agent energy giving; its net delta is zero.
    EnergySharing,
    /// Spike energy cost and victim health damage.
    Combat,
    /// Manufactured health/energy rewards from a carcass.
    CarcassReward,
    /// Resources removed with dead agents.
    DeathRemoval,
    /// Parent energy debit, rollback, and child resource allocation.
    ReproductionAllocation,
    /// Open-world population-floor and scheduled injection.
    PopulationInjection,
    /// Requested source/transfer magnitude rejected by a resource cap.
    CapacityRejection,
}

impl ResourceFlowKind {
    const fn index(self) -> usize {
        match self {
            Self::ScenarioIntervention => 0,
            Self::FoodDynamics => 1,
            Self::Aging => 2,
            Self::BasalMetabolism => 3,
            Self::Movement => 4,
            Self::MetabolismRamp => 5,
            Self::Boost => 6,
            Self::Topography => 7,
            Self::TemperatureStress => 8,
            Self::GroundFoodConversion => 9,
            Self::EnergySharing => 10,
            Self::Combat => 11,
            Self::CarcassReward => 12,
            Self::DeathRemoval => 13,
            Self::ReproductionAllocation => 14,
            Self::PopulationInjection => 15,
            Self::CapacityRejection => 16,
        }
    }
}

const RESOURCE_FLOW_KINDS: [ResourceFlowKind; 17] = [
    ResourceFlowKind::ScenarioIntervention,
    ResourceFlowKind::FoodDynamics,
    ResourceFlowKind::Aging,
    ResourceFlowKind::BasalMetabolism,
    ResourceFlowKind::Movement,
    ResourceFlowKind::MetabolismRamp,
    ResourceFlowKind::Boost,
    ResourceFlowKind::Topography,
    ResourceFlowKind::TemperatureStress,
    ResourceFlowKind::GroundFoodConversion,
    ResourceFlowKind::EnergySharing,
    ResourceFlowKind::Combat,
    ResourceFlowKind::CarcassReward,
    ResourceFlowKind::DeathRemoval,
    ResourceFlowKind::ReproductionAllocation,
    ResourceFlowKind::PopulationInjection,
    ResourceFlowKind::CapacityRejection,
];

/// One attributed resource flow.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceFlow {
    /// Stable category for this flow.
    pub kind: ResourceFlowKind,
    /// Signed world-wide resource change attributed to this category.
    pub delta: ResourceAmounts,
    /// Positive gross activity (for transfers and rejected capacity).
    pub activity: ResourceAmounts,
}

impl ResourceFlow {
    const fn empty(kind: ResourceFlowKind) -> Self {
        Self {
            kind,
            delta: ResourceAmounts {
                food: 0.0,
                energy: 0.0,
                health: 0.0,
            },
            activity: ResourceAmounts {
                food: 0.0,
                energy: 0.0,
                health: 0.0,
            },
        }
    }
}

/// Conservation proof attached to a completed ledger tick.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceReconciliation {
    /// Closing pools minus opening pools.
    pub observed_delta: ResourceAmounts,
    /// Sum of every attributed flow delta.
    pub attributed_delta: ResourceAmounts,
    /// `observed_delta - attributed_delta`; expected to be zero within tolerance.
    pub unexplained_delta: ResourceAmounts,
    /// Declared absolute-plus-relative tolerance for this tick.
    pub tolerance: f64,
    /// Whether every unexplained pool delta is within `tolerance`.
    pub reconciled: bool,
}

/// Immutable accounting report for one completed simulation tick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceLedgerTick {
    /// Tick whose completed boundary this report describes.
    pub tick: Tick,
    /// Resource pools at the start of the tick.
    pub opening: ResourceAmounts,
    /// Resource pools at the completed tick boundary.
    pub closing: ResourceAmounts,
    /// Stable enum order, including zero-valued categories.
    pub flows: Vec<ResourceFlow>,
    /// Conservation proof comparing observed and attributed deltas.
    pub reconciliation: ResourceReconciliation,
}

/// Immutable latest and cumulative resource accounting read model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceLedgerReport {
    /// Number of instrumented ticks retained in cumulative totals.
    pub completed_ticks: u64,
    /// Most recently completed instrumented tick.
    pub latest: Option<ResourceLedgerTick>,
    /// Stable enum order, including categories not yet observed.
    pub cumulative: Vec<ResourceFlow>,
}

impl Default for ResourceLedgerReport {
    fn default() -> Self {
        Self {
            completed_ticks: 0,
            latest: None,
            cumulative: RESOURCE_FLOW_KINDS
                .into_iter()
                .map(ResourceFlow::empty)
                .collect(),
        }
    }
}

#[derive(Debug)]
struct WorkingResourceLedger {
    tick: Tick,
    opening: ResourceAmounts,
    flows: Vec<ResourceFlow>,
}

#[derive(Debug, Default)]
struct ResourceLedgerState {
    enabled: bool,
    working: Option<WorkingResourceLedger>,
    report: ResourceLedgerReport,
}

impl ResourceLedgerState {
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.working = None;
        }
    }

    fn begin_tick(&mut self, tick: Tick, opening: ResourceAmounts) {
        if self.enabled {
            self.working = Some(WorkingResourceLedger {
                tick,
                opening,
                flows: RESOURCE_FLOW_KINDS
                    .into_iter()
                    .map(ResourceFlow::empty)
                    .collect(),
            });
        }
    }

    fn record(
        &mut self,
        kind: ResourceFlowKind,
        delta: ResourceAmounts,
        activity: ResourceAmounts,
    ) {
        let Some(working) = self.working.as_mut() else {
            return;
        };
        let index = kind.index();
        working.flows[index].delta.add_assign(delta);
        working.flows[index].activity.add_assign(activity);
    }

    fn record_change(
        &mut self,
        kind: ResourceFlowKind,
        before: ResourceAmounts,
        after: ResourceAmounts,
    ) {
        self.record(kind, after.delta_from(before), ResourceAmounts::default());
    }

    fn finish_tick(&mut self, closing: ResourceAmounts) {
        let Some(working) = self.working.take() else {
            return;
        };
        let mut attributed_delta = ResourceAmounts::default();
        for flow in &working.flows {
            attributed_delta.add_assign(flow.delta);
        }
        let observed_delta = closing.delta_from(working.opening);
        let unexplained_delta = observed_delta.subtract(attributed_delta);
        let tolerance = RESOURCE_LEDGER_ABSOLUTE_TOLERANCE
            + RESOURCE_LEDGER_RELATIVE_TOLERANCE * working.opening.scale().max(closing.scale());
        let reconciliation = ResourceReconciliation {
            observed_delta,
            attributed_delta,
            unexplained_delta,
            tolerance,
            reconciled: unexplained_delta.within(tolerance),
        };
        for (cumulative, flow) in self.report.cumulative.iter_mut().zip(&working.flows) {
            cumulative.delta.add_assign(flow.delta);
            cumulative.activity.add_assign(flow.activity);
        }
        self.report.completed_ticks = self.report.completed_ticks.saturating_add(1);
        self.report.latest = Some(ResourceLedgerTick {
            tick: working.tick,
            opening: working.opening,
            closing,
            flows: working.flows,
            reconciliation,
        });
    }
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
                config.closed = true;
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
                "closed": true
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
    pub agent_uid: AgentUid,
    pub spawn_ordinal: u64,
    pub birth_ordinal: u64,
    pub parent_a: Option<AgentUid>,
    pub parent_b: Option<AgentUid>,
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
    pub agent_uid: AgentUid,
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
        agent_uid: AgentUid,
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
        spike_target: Option<AgentUid>,
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
    pub agent_uid: Option<AgentUid>,
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
    /// Admit a completed tick without silently discarding an unacknowledged batch.
    ///
    /// An error means the caller has no admission proof and must retain the exact batch. A sink
    /// must either prove `NotAdmitted` or make an indeterminate acknowledgement failure safe to
    /// retry through stable identity and exact-payload deduplication. A changed payload is never
    /// a valid retry for the same completed tick.
    fn on_tick(&mut self, payload: &PersistenceBatch) -> Result<(), PersistenceAdmissionError>;
}

/// What the caller knows about a completed batch after admission failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistenceAdmissionState {
    /// The sink proved that the completed batch did not enter durable admission state.
    NotAdmitted,
    /// Admission may have committed, but its acknowledgement did not reach the caller.
    Indeterminate,
}

/// Typed rejection at the lossless persistence admission boundary.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("persistence did not acknowledge completed simulation tick {tick} ({state:?}): {detail}")]
pub struct PersistenceAdmissionError {
    tick: u64,
    state: PersistenceAdmissionState,
    detail: String,
}

impl PersistenceAdmissionError {
    /// Create a proven non-admission failure while preserving the backend's diagnostic text.
    #[must_use]
    pub fn new(tick: u64, detail: impl Into<String>) -> Self {
        Self {
            tick,
            state: PersistenceAdmissionState::NotAdmitted,
            detail: detail.into(),
        }
    }

    /// Create an acknowledgement failure whose admission outcome is indeterminate.
    #[must_use]
    pub fn indeterminate(tick: u64, detail: impl Into<String>) -> Self {
        Self {
            tick,
            state: PersistenceAdmissionState::Indeterminate,
            detail: detail.into(),
        }
    }

    /// Completed tick whose exact batch must remain available for retry.
    #[must_use]
    pub const fn tick(&self) -> u64 {
        self.tick
    }

    /// Proven non-admission versus a lost acknowledgement after possible admission.
    #[must_use]
    pub const fn state(&self) -> PersistenceAdmissionState {
        self.state
    }

    /// Backend diagnostic suitable for structured host reporting.
    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

/// Failure while executing a simulation tick.
#[derive(Debug, Clone, Error)]
pub enum WorldStepError {
    /// A registered brain factory could not construct a runner.
    #[error(transparent)]
    BrainSpawn(#[from] BrainSpawnError),
    /// The completed tick could not be admitted to persistence.
    #[error(transparent)]
    Persistence(#[from] PersistenceAdmissionError),
    /// Brain construction and persistence admission both failed at the same completed boundary.
    #[error(
        "brain construction failed while the completed tick was also rejected by persistence: {brain}; {persistence}"
    )]
    BrainAndPersistence {
        brain: BrainSpawnError,
        persistence: PersistenceAdmissionError,
    },
}

/// No-op persistence sink.
#[derive(Debug, Default)]
pub struct NullPersistence;

impl WorldPersistence for NullPersistence {
    fn on_tick(&mut self, _payload: &PersistenceBatch) -> Result<(), PersistenceAdmissionError> {
        Ok(())
    }
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

/// Legacy genome-provenance placeholder awaiting the versioned genome protocol.
///
/// Its transient `AgentId` parents are intentionally not half-migrated here: `bd-2z0.3.2`
/// replaces this entire placeholder with the stable-UID genome/evaluator-state envelope.
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

    /// Validate every floating-point field before this payload crosses a scientific-state
    /// boundary.
    pub fn validate(&self) -> Result<(), ScientificStateError> {
        self.validate_at("agent")
    }

    fn validate_at(&self, path: &str) -> Result<(), ScientificStateError> {
        validate_finite(&format!("{path}.position.x"), self.position.x)?;
        validate_finite(&format!("{path}.position.y"), self.position.y)?;
        validate_finite(&format!("{path}.velocity.vx"), self.velocity.vx)?;
        validate_finite(&format!("{path}.velocity.vy"), self.velocity.vy)?;
        validate_finite(&format!("{path}.heading"), self.heading)?;
        validate_finite(&format!("{path}.health"), self.health)?;
        for (index, value) in self.color.iter().copied().enumerate() {
            validate_finite(&format!("{path}.color[{index}]"), value)?;
        }
        validate_finite(&format!("{path}.spike_length"), self.spike_length)
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
#[derive(Debug, Default, Serialize)]
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

    /// Validate and append a new row atomically.
    pub fn try_push(&mut self, agent: AgentData) -> Result<(), ScientificStateError> {
        agent.validate_at(&format!("agents[{}]", self.len()))?;
        self.push_trusted(agent);
        Ok(())
    }

    /// Push a row that has already been validated at the owning boundary.
    fn push_trusted(&mut self, agent: AgentData) {
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

    /// Internal mutable access to trusted positions in the tick hot path.
    #[must_use]
    fn positions_mut(&mut self) -> &mut [Position] {
        &mut self.positions
    }

    /// Immutable access to the velocities slice.
    #[must_use]
    pub fn velocities(&self) -> &[Velocity] {
        &self.velocities
    }

    /// Internal mutable access to trusted velocities in the tick hot path.
    #[must_use]
    fn velocities_mut(&mut self) -> &mut [Velocity] {
        &mut self.velocities
    }

    /// Immutable access to headings.
    #[must_use]
    pub fn headings(&self) -> &[f32] {
        &self.headings
    }

    /// Internal mutable access to trusted headings in the tick hot path.
    #[must_use]
    fn headings_mut(&mut self) -> &mut [f32] {
        &mut self.headings
    }

    /// Immutable access to health values.
    #[must_use]
    pub fn health(&self) -> &[f32] {
        &self.health
    }

    /// Internal mutable access to trusted health in the tick hot path.
    #[must_use]
    fn health_mut(&mut self) -> &mut [f32] {
        &mut self.health
    }

    /// Immutable access to color triples.
    #[must_use]
    pub fn colors(&self) -> &[[f32; 3]] {
        &self.colors
    }

    /// Internal mutable access to trusted colors in the tick hot path.
    #[must_use]
    fn colors_mut(&mut self) -> &mut [[f32; 3]] {
        &mut self.colors
    }

    /// Immutable access to spike lengths.
    #[must_use]
    pub fn spike_lengths(&self) -> &[f32] {
        &self.spike_lengths
    }

    /// Internal mutable access to trusted spike lengths in the tick hot path.
    #[must_use]
    fn spike_lengths_mut(&mut self) -> &mut [f32] {
        &mut self.spike_lengths
    }

    /// Immutable access to boost flags.
    #[must_use]
    pub fn boosts(&self) -> &[bool] {
        &self.boosts
    }

    /// Immutable access to age counters.
    #[must_use]
    pub fn ages(&self) -> &[u32] {
        &self.ages
    }

    /// Internal mutable access to age counters in the tick hot path.
    #[must_use]
    fn ages_mut(&mut self) -> &mut [u32] {
        &mut self.ages
    }

    /// Immutable access to agent generations.
    #[must_use]
    pub fn generations(&self) -> &[Generation] {
        &self.generations
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

    /// Mutably borrow trusted column storage inside the simulation crate.
    #[must_use]
    fn columns_mut(&mut self) -> &mut AgentColumns {
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

    /// Validate and insert an agent without changing allocator state on rejection.
    pub fn try_insert(&mut self, agent: AgentData) -> Result<AgentId, ScientificStateError> {
        agent.validate_at(&format!("agents[{}]", self.len()))?;
        Ok(self.insert_trusted(agent))
    }

    /// Insert a payload already validated by the owning boundary.
    fn insert_trusted(&mut self, agent: AgentData) -> AgentId {
        let index = self.columns.len();
        self.columns.push_trusted(agent);
        let id = self.slots.insert(index);
        self.handles.push(id);
        id
    }

    /// Internal insertion path for values produced by validated simulation logic.
    fn insert(&mut self, agent: AgentData) -> AgentId {
        debug_assert!(agent.validate().is_ok());
        self.insert_trusted(agent)
    }

    /// Replace one dense row after validating the complete candidate.
    fn replace_trusted(&mut self, id: AgentId, agent: AgentData) -> bool {
        let Some(index) = self.index_of(id) else {
            return false;
        };
        self.columns.positions[index] = agent.position;
        self.columns.velocities[index] = agent.velocity;
        self.columns.headings[index] = agent.heading;
        self.columns.health[index] = agent.health;
        self.columns.colors[index] = agent.color;
        self.columns.spike_lengths[index] = agent.spike_length;
        self.columns.boosts[index] = agent.boost;
        self.columns.ages[index] = agent.age;
        self.columns.generations[index] = agent.generation;
        true
    }

    /// Capture the allocator and dense length before an append-only transaction.
    ///
    /// Restoring this checkpoint preserves both live rows and the otherwise
    /// invisible `SlotMap` generation/free-list state when fallible preparation
    /// forces recently appended agents to roll back.
    fn append_checkpoint(&self) -> (SlotMap<AgentId, usize>, usize) {
        (self.slots.clone(), self.columns.len())
    }

    /// Roll back agents appended after `checkpoint` without advancing handle
    /// generations or perturbing the pre-existing dense order.
    fn restore_append_checkpoint(&mut self, checkpoint: (SlotMap<AgentId, usize>, usize)) {
        let (slots, len) = checkpoint;
        debug_assert!(len <= self.columns.len());
        self.slots = slots;
        self.handles.truncate(len);
        self.columns.truncate(len);
        self.columns.debug_assert_coherent();
        debug_assert_eq!(self.handles.len(), len);
        debug_assert_eq!(self.slots.len(), len);
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

/// Typed rejection at a scientific-state construction or mutation boundary.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ScientificStateError {
    /// A float that can influence simulation or imported environment state was not finite.
    #[error("non-finite scientific state at `{path}`")]
    NonFinite { path: String },
    /// A dense imported field did not contain exactly one value per declared cell.
    #[error("scientific-state length mismatch at `{path}`: expected {expected}, got {actual}")]
    LengthMismatch {
        path: String,
        expected: usize,
        actual: usize,
    },
    /// Declared dimensions could not be represented safely as a flat allocation length.
    #[error("scientific-state dimensions overflow at `{path}`")]
    DimensionOverflow { path: String },
    /// Two coupled dense fields declared different shapes even when their flat lengths matched.
    #[error(
        "scientific-state dimensions mismatch at `{path}`: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}"
    )]
    DimensionsMismatch {
        path: String,
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    /// A coupled imported state supplied only one half of a required pair.
    #[error("incomplete coupled scientific state at `{path}`")]
    IncompletePair { path: String },
}

impl ScientificStateError {
    /// Exact field or collection path rejected by validation.
    #[must_use]
    pub fn path(&self) -> &str {
        match self {
            Self::NonFinite { path }
            | Self::LengthMismatch { path, .. }
            | Self::DimensionOverflow { path }
            | Self::DimensionsMismatch { path, .. }
            | Self::IncompletePair { path } => path,
        }
    }
}

fn validate_finite(path: &str, value: f32) -> Result<(), ScientificStateError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScientificStateError::NonFinite {
            path: path.to_owned(),
        })
    }
}

fn validated_cell_count_for<T>(
    path: &str,
    width: u32,
    height: u32,
) -> Result<usize, ScientificStateError> {
    let width = usize::try_from(width).map_err(|_| ScientificStateError::DimensionOverflow {
        path: path.to_owned(),
    })?;
    let height = usize::try_from(height).map_err(|_| ScientificStateError::DimensionOverflow {
        path: path.to_owned(),
    })?;
    let count =
        width
            .checked_mul(height)
            .ok_or_else(|| ScientificStateError::DimensionOverflow {
                path: path.to_owned(),
            })?;
    std::alloc::Layout::array::<T>(count)
        .map(|_| count)
        .map_err(|_| ScientificStateError::DimensionOverflow {
            path: path.to_owned(),
        })
}

fn validate_finite_slice(path: &str, values: &[f32]) -> Result<(), ScientificStateError> {
    for (index, value) in values.iter().copied().enumerate() {
        validate_finite(&format!("{path}[{index}]"), value)?;
    }
    Ok(())
}

/// Errors that can occur when constructing world state.
#[derive(Debug, Error)]
pub enum WorldStateError {
    /// Indicates an invalid configuration value.
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
    /// A direct-Rust or imported-map state payload violated the finite-state contract.
    #[error(transparent)]
    InvalidState(#[from] ScientificStateError),
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

/// Render-specific configuration shared across front-ends.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RenderSettings {
    /// Preferred tonemapping curve for HDR output. None falls back to renderer defaults.
    #[serde(default)]
    pub tonemap_mode: Option<RenderTonemapMode>,
    /// Exposure bias applied on top of the selected tonemap curve.
    #[serde(default)]
    pub tonemap_exposure_bias: Option<f32>,
    /// Auto-exposure parameters; omitted values defer to renderer defaults.
    #[serde(default)]
    pub auto_exposure: Option<RenderAutoExposureSettings>,
}

/// Supported tonemapping curves for renderer configuration.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RenderTonemapMode {
    #[default]
    Aces,
    Agx,
    Tony,
}

/// Auto-exposure configuration applied by renderers that support HDR adaption.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RenderAutoExposureSettings {
    /// Enable/disable the AutoExposure component.
    pub enabled: bool,
    /// Dark-to-bright adaptation speed; None keeps renderer default.
    #[serde(default)]
    pub speed_brighten: Option<f32>,
    /// Bright-to-dark adaptation speed; None keeps renderer default.
    #[serde(default)]
    pub speed_darken: Option<f32>,
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
    /// Whether automatic population-floor and scheduled injection are disabled.
    #[serde(default)]
    pub closed: bool,
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
    /// Ticks between narrative-detector passes over the tick history. 0 disables
    /// narration.
    ///
    /// The pass is a bounded scan of the history ring, so running it every tick
    /// would be pure waste at 60+ ticks per second; a cadence costs nothing in
    /// fidelity because the detectors are sequential and the ring is retained.
    #[serde(default = "default_narrative_interval")]
    pub narrative_interval: u32,
    /// Maximum number of narrative events retained in-memory.
    #[serde(default = "default_narrative_capacity")]
    pub narrative_capacity: usize,
    /// Interval (ticks) between persistence flushes. 0 disables persistence.
    pub persistence_interval: u32,
    /// Sampling cadence for analytics families.
    pub analytics_stride: AnalyticsStride,
    /// NeuroFlow runtime configuration.
    pub neuroflow: NeuroflowSettings,
    /// Control-related runtime behavior toggles.
    pub control: ControlSettings,
    /// Renderer configuration shared across front-ends.
    #[serde(default)]
    pub render: RenderSettings,
}

impl Default for ScriptBotsConfig {
    fn default() -> Self {
        Self {
            world_width: 6_000,
            world_height: 3_000,
            food_cell_size: 50,
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
            metabolism_drain: 0.0002,
            movement_drain: 0.005,
            metabolism_ramp_floor: 1.0,
            metabolism_ramp_rate: 0.0,
            metabolism_boost_penalty: 0.0,
            temperature_discomfort_rate: 0.0,
            temperature_comfort_band: DEFAULT_TEMPERATURE_COMFORT_BAND,
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
            closed: false,
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
            narrative_interval: default_narrative_interval(),
            narrative_capacity: default_narrative_capacity(),
            persistence_interval: 0,
            analytics_stride: AnalyticsStride::default(),
            neuroflow: NeuroflowSettings {
                enabled: true,
                ..NeuroflowSettings::default()
            },
            control: ControlSettings::default(),
            render: RenderSettings::default(),
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
    /// Validate every public configuration invariant without mutating runtime state.
    pub fn validate(&self) -> Result<(), WorldStateError> {
        macro_rules! reject_unless {
            ($condition:expr, $message:expr) => {
                if $condition {
                } else {
                    return Err(WorldStateError::InvalidConfig($message));
                }
            };
        }

        reject_unless!(self.world_width != 0, "world_width must be non-zero");
        reject_unless!(self.world_height != 0, "world_height must be non-zero");
        reject_unless!(self.food_cell_size != 0, "food_cell_size must be non-zero");
        reject_unless!(
            self.world_width.is_multiple_of(self.food_cell_size),
            "world_width must be divisible by food_cell_size"
        );
        reject_unless!(
            self.world_height.is_multiple_of(self.food_cell_size),
            "world_height must be divisible by food_cell_size"
        );

        let finite_fields: [(f32, &'static str); 70] = [
            (self.initial_food, "initial_food must be finite"),
            (
                self.food_respawn_amount,
                "food_respawn_amount must be finite",
            ),
            (self.food_max, "food_max must be finite"),
            (self.food_growth_rate, "food_growth_rate must be finite"),
            (self.food_decay_rate, "food_decay_rate must be finite"),
            (
                self.food_diffusion_rate,
                "food_diffusion_rate must be finite",
            ),
            (self.sense_radius, "sense_radius must be finite"),
            (
                self.sense_max_neighbors,
                "sense_max_neighbors must be finite",
            ),
            (self.bot_speed, "bot_speed must be finite"),
            (self.bot_radius, "bot_radius must be finite"),
            (self.boost_multiplier, "boost_multiplier must be finite"),
            (self.spike_growth_rate, "spike_growth_rate must be finite"),
            (self.metabolism_drain, "metabolism_drain must be finite"),
            (self.movement_drain, "movement_drain must be finite"),
            (
                self.metabolism_ramp_floor,
                "metabolism_ramp_floor must be finite",
            ),
            (
                self.metabolism_ramp_rate,
                "metabolism_ramp_rate must be finite",
            ),
            (
                self.metabolism_boost_penalty,
                "metabolism_boost_penalty must be finite",
            ),
            (
                self.temperature_discomfort_rate,
                "temperature_discomfort_rate must be finite",
            ),
            (
                self.temperature_comfort_band,
                "temperature_comfort_band must be finite",
            ),
            (
                self.temperature_gradient_exponent,
                "temperature_gradient_exponent must be finite",
            ),
            (
                self.temperature_discomfort_exponent,
                "temperature_discomfort_exponent must be finite",
            ),
            (self.food_intake_rate, "food_intake_rate must be finite"),
            (self.food_waste_rate, "food_waste_rate must be finite"),
            (
                self.food_fertility_base,
                "food_fertility_base must be finite",
            ),
            (
                self.food_moisture_weight,
                "food_moisture_weight must be finite",
            ),
            (
                self.food_elevation_weight,
                "food_elevation_weight must be finite",
            ),
            (self.food_slope_weight, "food_slope_weight must be finite"),
            (self.food_capacity_base, "food_capacity_base must be finite"),
            (
                self.food_capacity_fertility,
                "food_capacity_fertility must be finite",
            ),
            (
                self.food_growth_fertility,
                "food_growth_fertility must be finite",
            ),
            (
                self.food_decay_infertility,
                "food_decay_infertility must be finite",
            ),
            (
                self.food_sharing_radius,
                "food_sharing_radius must be finite",
            ),
            (self.food_sharing_rate, "food_sharing_rate must be finite"),
            (self.food_transfer_rate, "food_transfer_rate must be finite"),
            (
                self.food_sharing_distance,
                "food_sharing_distance must be finite",
            ),
            (
                self.reproduction_energy_threshold,
                "reproduction_energy_threshold must be finite",
            ),
            (
                self.reproduction_energy_cost,
                "reproduction_energy_cost must be finite",
            ),
            (
                self.reproduction_attempt_chance,
                "reproduction_attempt_chance must be finite",
            ),
            (
                self.reproduction_rate_herbivore,
                "reproduction_rate_herbivore must be finite",
            ),
            (
                self.reproduction_rate_carnivore,
                "reproduction_rate_carnivore must be finite",
            ),
            (
                self.reproduction_food_bonus,
                "reproduction_food_bonus must be finite",
            ),
            (
                self.reproduction_fertility_bonus,
                "reproduction_fertility_bonus must be finite",
            ),
            (
                self.reproduction_child_energy,
                "reproduction_child_energy must be finite",
            ),
            (
                self.reproduction_spawn_jitter,
                "reproduction_spawn_jitter must be finite",
            ),
            (
                self.reproduction_color_jitter,
                "reproduction_color_jitter must be finite",
            ),
            (
                self.reproduction_mutation_scale,
                "reproduction_mutation_scale must be finite",
            ),
            (
                self.reproduction_partner_chance,
                "reproduction_partner_chance must be finite",
            ),
            (
                self.reproduction_spawn_back_distance,
                "reproduction_spawn_back_distance must be finite",
            ),
            (
                self.reproduction_meta_mutation_chance,
                "reproduction_meta_mutation_chance must be finite",
            ),
            (
                self.reproduction_meta_mutation_scale,
                "reproduction_meta_mutation_scale must be finite",
            ),
            (
                self.aging_health_decay_rate,
                "aging_health_decay_rate must be finite",
            ),
            (
                self.aging_health_decay_max,
                "aging_health_decay_max must be finite",
            ),
            (
                self.aging_energy_penalty_rate,
                "aging_energy_penalty_rate must be finite",
            ),
            (
                self.carcass_distribution_radius,
                "carcass_distribution_radius must be finite",
            ),
            (
                self.carcass_health_reward,
                "carcass_health_reward must be finite",
            ),
            (
                self.carcass_reproduction_reward,
                "carcass_reproduction_reward must be finite",
            ),
            (
                self.carcass_neighbor_exponent,
                "carcass_neighbor_exponent must be finite",
            ),
            (
                self.carcass_energy_share_rate,
                "carcass_energy_share_rate must be finite",
            ),
            (
                self.carcass_indicator_scale,
                "carcass_indicator_scale must be finite",
            ),
            (
                self.topography_speed_gain,
                "topography_speed_gain must be finite",
            ),
            (
                self.topography_energy_penalty,
                "topography_energy_penalty must be finite",
            ),
            (
                self.population_crossover_chance,
                "population_crossover_chance must be finite",
            ),
            (self.spike_radius, "spike_radius must be finite"),
            (self.spike_damage, "spike_damage must be finite"),
            (self.spike_energy_cost, "spike_energy_cost must be finite"),
            (self.spike_min_length, "spike_min_length must be finite"),
            (
                self.spike_alignment_cosine,
                "spike_alignment_cosine must be finite",
            ),
            (
                self.spike_speed_damage_bonus,
                "spike_speed_damage_bonus must be finite",
            ),
            (
                self.spike_length_damage_bonus,
                "spike_length_damage_bonus must be finite",
            ),
            (
                self.carnivore_threshold,
                "carnivore_threshold must be finite",
            ),
        ];
        for (value, message) in finite_fields {
            reject_unless!(value.is_finite(), message);
        }

        if let Some(value) = self.render.tonemap_exposure_bias {
            reject_unless!(
                value.is_finite(),
                "render.tonemap_exposure_bias must be finite"
            );
        }
        if let Some(auto_exposure) = &self.render.auto_exposure {
            if let Some(value) = auto_exposure.speed_brighten {
                reject_unless!(
                    value.is_finite(),
                    "render.auto_exposure.speed_brighten must be finite"
                );
                reject_unless!(
                    value >= 0.0,
                    "render.auto_exposure.speed_brighten must be non-negative"
                );
            }
            if let Some(value) = auto_exposure.speed_darken {
                reject_unless!(
                    value.is_finite(),
                    "render.auto_exposure.speed_darken must be finite"
                );
                reject_unless!(
                    value >= 0.0,
                    "render.auto_exposure.speed_darken must be non-negative"
                );
            }
        }

        reject_unless!(
            self.initial_food >= 0.0,
            "initial_food must be non-negative"
        );
        reject_unless!(self.food_max > 0.0, "food_max must be positive");
        reject_unless!(
            self.food_respawn_amount >= 0.0,
            "food_respawn_amount must be non-negative"
        );
        reject_unless!(
            self.initial_food <= self.food_max,
            "initial_food cannot exceed food_max"
        );
        reject_unless!(
            self.food_respawn_amount <= self.food_max,
            "food_respawn_amount cannot exceed food_max"
        );
        reject_unless!(
            self.food_growth_rate >= 0.0,
            "food_growth_rate must be non-negative"
        );
        reject_unless!(
            self.food_decay_rate >= 0.0,
            "food_decay_rate must be non-negative"
        );
        reject_unless!(
            (0.0..=0.25).contains(&self.food_diffusion_rate),
            "food_diffusion_rate must be within [0, 0.25]"
        );
        reject_unless!(self.sense_radius > 0.0, "sense_radius must be positive");
        reject_unless!(
            self.sense_max_neighbors > 0.0,
            "sense_max_neighbors must be positive"
        );
        reject_unless!(self.bot_speed >= 0.0, "bot_speed must be non-negative");
        reject_unless!(self.bot_radius > 0.0, "bot_radius must be positive");
        reject_unless!(
            self.boost_multiplier >= 1.0,
            "boost_multiplier must be at least 1.0"
        );
        reject_unless!(
            self.spike_growth_rate >= 0.0,
            "spike_growth_rate must be non-negative"
        );
        reject_unless!(
            self.metabolism_drain >= 0.0,
            "metabolism_drain must be non-negative"
        );
        reject_unless!(
            self.movement_drain >= 0.0,
            "movement_drain must be non-negative"
        );
        reject_unless!(
            self.metabolism_ramp_floor >= 0.0,
            "metabolism_ramp_floor must be non-negative"
        );
        reject_unless!(
            self.metabolism_ramp_rate >= 0.0,
            "metabolism_ramp_rate must be non-negative"
        );
        reject_unless!(
            self.metabolism_boost_penalty >= 0.0,
            "metabolism_boost_penalty must be non-negative"
        );
        reject_unless!(
            self.temperature_discomfort_rate >= 0.0,
            "temperature_discomfort_rate must be non-negative"
        );
        reject_unless!(
            (0.0..=1.0).contains(&self.temperature_comfort_band),
            "temperature_comfort_band must be within [0, 1]"
        );
        reject_unless!(
            self.temperature_gradient_exponent > 0.0,
            "temperature_gradient_exponent must be positive"
        );
        reject_unless!(
            self.temperature_discomfort_exponent > 0.0,
            "temperature_discomfort_exponent must be positive"
        );
        reject_unless!(
            self.food_intake_rate >= 0.0,
            "food_intake_rate must be non-negative"
        );
        reject_unless!(
            self.food_waste_rate >= 0.0,
            "food_waste_rate must be non-negative"
        );
        reject_unless!(
            self.food_waste_rate <= self.food_max,
            "food_waste_rate cannot exceed food_max"
        );
        reject_unless!(
            (0.0..=1.0).contains(&self.food_fertility_base),
            "food_fertility_base must be within [0, 1]"
        );
        reject_unless!(
            self.food_moisture_weight >= 0.0,
            "food_moisture_weight must be non-negative"
        );
        reject_unless!(
            self.food_elevation_weight >= 0.0,
            "food_elevation_weight must be non-negative"
        );
        reject_unless!(
            self.food_slope_weight >= 0.0,
            "food_slope_weight must be non-negative"
        );
        reject_unless!(
            (0.0..=1.0).contains(&self.food_capacity_base),
            "food_capacity_base must be within [0, 1]"
        );
        reject_unless!(
            self.food_capacity_fertility >= 0.0,
            "food_capacity_fertility must be non-negative"
        );
        reject_unless!(
            self.food_growth_fertility >= 0.0,
            "food_growth_fertility must be non-negative"
        );
        reject_unless!(
            self.food_decay_infertility >= 0.0,
            "food_decay_infertility must be non-negative"
        );
        reject_unless!(
            self.food_capacity_base + self.food_capacity_fertility <= 1.0,
            "food_capacity_base + food_capacity_fertility must be <= 1.0"
        );
        reject_unless!(
            self.food_sharing_radius > 0.0,
            "food_sharing_radius must be positive"
        );
        reject_unless!(
            self.food_sharing_rate >= 0.0,
            "food_sharing_rate must be non-negative"
        );
        reject_unless!(
            self.food_transfer_rate >= 0.0,
            "food_transfer_rate must be non-negative"
        );
        reject_unless!(
            self.food_sharing_distance > 0.0,
            "food_sharing_distance must be positive"
        );
        reject_unless!(
            self.reproduction_energy_threshold >= 0.0,
            "reproduction_energy_threshold must be non-negative"
        );
        reject_unless!(
            self.reproduction_energy_cost >= 0.0,
            "reproduction_energy_cost must be non-negative"
        );
        reject_unless!(
            self.reproduction_energy_cost <= self.reproduction_energy_threshold,
            "reproduction_energy_cost cannot exceed reproduction_energy_threshold"
        );
        reject_unless!(
            (0.0..=1.0).contains(&self.reproduction_attempt_chance),
            "reproduction_attempt_chance must be within [0, 1]"
        );
        reject_unless!(
            self.reproduction_rate_herbivore > 0.0,
            "reproduction_rate_herbivore must be positive"
        );
        reject_unless!(
            self.reproduction_rate_carnivore > 0.0,
            "reproduction_rate_carnivore must be positive"
        );
        reject_unless!(
            self.reproduction_food_bonus >= 0.0,
            "reproduction_food_bonus must be non-negative"
        );
        reject_unless!(
            self.reproduction_fertility_bonus >= 0.0,
            "reproduction_fertility_bonus must be non-negative"
        );
        reject_unless!(
            self.reproduction_child_energy >= 0.0,
            "reproduction_child_energy must be non-negative"
        );
        reject_unless!(
            self.reproduction_spawn_jitter >= 0.0,
            "reproduction_spawn_jitter must be non-negative"
        );
        reject_unless!(
            self.reproduction_color_jitter >= 0.0,
            "reproduction_color_jitter must be non-negative"
        );
        reject_unless!(
            self.reproduction_mutation_scale >= 0.0,
            "reproduction_mutation_scale must be non-negative"
        );
        reject_unless!(
            (0.0..=1.0).contains(&self.reproduction_partner_chance),
            "reproduction_partner_chance must be within [0, 1]"
        );
        reject_unless!(
            self.reproduction_spawn_back_distance >= 0.0,
            "reproduction_spawn_back_distance must be non-negative"
        );
        reject_unless!(
            (0.0..=1.0).contains(&self.reproduction_meta_mutation_chance),
            "reproduction_meta_mutation_chance must be within [0, 1]"
        );
        reject_unless!(
            self.reproduction_meta_mutation_scale >= 0.0,
            "reproduction_meta_mutation_scale must be non-negative"
        );
        reject_unless!(
            self.aging_tick_interval != 0,
            "aging_tick_interval must be at least 1"
        );
        reject_unless!(
            self.aging_health_decay_rate >= 0.0,
            "aging_health_decay_rate must be non-negative"
        );
        reject_unless!(
            self.aging_health_decay_max >= 0.0,
            "aging_health_decay_max must be non-negative"
        );
        reject_unless!(
            self.aging_health_decay_rate == 0.0
                || self.aging_health_decay_max >= self.aging_health_decay_rate,
            "aging_health_decay_max must be >= aging_health_decay_rate when decay is enabled"
        );
        reject_unless!(
            self.aging_energy_penalty_rate >= 0.0,
            "aging_energy_penalty_rate must be non-negative"
        );
        reject_unless!(
            self.carcass_distribution_radius >= 0.0,
            "carcass_distribution_radius must be non-negative"
        );
        reject_unless!(
            self.carcass_health_reward >= 0.0,
            "carcass_health_reward must be non-negative"
        );
        reject_unless!(
            self.carcass_reproduction_reward >= 0.0,
            "carcass_reproduction_reward must be non-negative"
        );
        reject_unless!(
            self.carcass_neighbor_exponent > 0.0,
            "carcass_neighbor_exponent must be positive"
        );
        reject_unless!(
            self.carcass_maturity_age != 0,
            "carcass_maturity_age must be at least 1"
        );
        reject_unless!(
            self.carcass_energy_share_rate >= 0.0,
            "carcass_energy_share_rate must be non-negative"
        );
        reject_unless!(
            self.carcass_indicator_scale >= 0.0,
            "carcass_indicator_scale must be non-negative"
        );
        reject_unless!(
            self.topography_speed_gain >= 0.0,
            "topography_speed_gain must be non-negative"
        );
        reject_unless!(
            self.topography_energy_penalty >= 0.0,
            "topography_energy_penalty must be non-negative"
        );
        reject_unless!(
            self.population_spawn_count != 0,
            "population_spawn_count must be at least 1"
        );
        reject_unless!(
            (0.0..=1.0).contains(&self.population_crossover_chance),
            "population_crossover_chance must be within [0, 1]"
        );
        reject_unless!(self.spike_radius > 0.0, "spike_radius must be positive");
        reject_unless!(
            self.spike_damage >= 0.0,
            "spike_damage must be non-negative"
        );
        reject_unless!(
            self.spike_energy_cost >= 0.0,
            "spike_energy_cost must be non-negative"
        );
        reject_unless!(
            self.spike_min_length >= 0.0,
            "spike_min_length must be non-negative"
        );
        reject_unless!(
            (0.0..=1.0).contains(&self.spike_alignment_cosine) && self.spike_alignment_cosine > 0.0,
            "spike_alignment_cosine must be within (0, 1]"
        );
        reject_unless!(
            self.spike_speed_damage_bonus >= 0.0,
            "spike_speed_damage_bonus must be non-negative"
        );
        reject_unless!(
            self.spike_length_damage_bonus >= 0.0,
            "spike_length_damage_bonus must be non-negative"
        );
        reject_unless!(
            self.carnivore_threshold > 0.0 && self.carnivore_threshold < 1.0,
            "carnivore_threshold must be within (0, 1)"
        );
        reject_unless!(
            self.history_capacity != 0,
            "history_capacity must be at least 1"
        );
        Ok(())
    }

    /// Validate the configuration and return its derived food-grid dimensions.
    pub fn food_dimensions(&self) -> Result<(u32, u32), WorldStateError> {
        self.validate()?;
        Ok((
            self.world_width / self.food_cell_size,
            self.world_height / self.food_cell_size,
        ))
    }

    /// Returns the configured RNG seed, generating one from entropy if absent.
    fn seeded_rng(&self) -> SmallRngStream {
        SmallRngStream::seed_from_u64(self.rng_seed.unwrap_or_else(rand::random))
    }
}

/// 2D food grid storing scalar energy values.
#[derive(Debug, Clone, Serialize)]
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
        validate_finite("food.initial", initial)?;
        let len = validated_cell_count_for::<f32>("food", width, height)?;
        Ok(Self {
            width,
            height,
            cells: vec![initial; len],
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
    fn cells_mut(&mut self) -> &mut [f32] {
        &mut self.cells
    }

    /// Apply a bulk edit to a detached copy, validate every resulting cell, then commit once.
    pub fn try_update_cells(
        &mut self,
        update: impl FnOnce(&mut [f32]),
    ) -> Result<(), ScientificStateError> {
        let mut candidate = self.cells.clone();
        update(&mut candidate);
        validate_finite_slice("food.cells", &candidate)?;
        self.cells = candidate;
        Ok(())
    }

    /// Replace the complete dense field after validating its length and all values.
    pub fn try_replace_cells(&mut self, cells: Vec<f32>) -> Result<(), ScientificStateError> {
        if cells.len() != self.cells.len() {
            return Err(ScientificStateError::LengthMismatch {
                path: "food.cells".to_owned(),
                expected: self.cells.len(),
                actual: cells.len(),
            });
        }
        validate_finite_slice("food.cells", &cells)?;
        self.cells = cells;
        Ok(())
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
    fn get_mut(&mut self, x: u32, y: u32) -> Option<&mut f32> {
        if x < self.width && y < self.height {
            let idx = self.offset(x, y);
            Some(&mut self.cells[idx])
        } else {
            None
        }
    }

    /// Fills the grid with the provided scalar value.
    pub fn fill(&mut self, value: f32) -> Result<(), ScientificStateError> {
        validate_finite("food.fill", value)?;
        self.cells.fill(value);
        Ok(())
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
        if width == 0 || height == 0 || cell_size == 0 {
            return Err(WorldStateError::InvalidConfig(
                "terrain dimensions and cell size must be non-zero",
            ));
        }

        let mut tiles = Vec::with_capacity(validated_cell_count_for::<TerrainTile>(
            "terrain.tiles",
            width,
            height,
        )?);
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
                let temperature_bias = default_tile_temperature_bias(fx);
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
        if width == 0 || height == 0 || cell_size == 0 {
            return Err(WorldStateError::InvalidConfig(
                "terrain dimensions and cell size must be non-zero",
            ));
        }
        let expected = validated_cell_count_for::<TerrainTile>("terrain.tiles", width, height)?;
        if tiles.len() != expected {
            return Err(WorldStateError::InvalidConfig(
                "terrain tile count does not match dimensions",
            ));
        }
        let layer = Self {
            width,
            height,
            cell_size,
            tiles,
        };
        layer.validate()?;
        Ok(layer)
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

    /// Validate all imported floating tile fields without normalizing them.
    pub fn validate(&self) -> Result<(), ScientificStateError> {
        let expected =
            validated_cell_count_for::<TerrainTile>("terrain.tiles", self.width, self.height)?;
        if self.tiles.len() != expected {
            return Err(ScientificStateError::LengthMismatch {
                path: "terrain.tiles".to_owned(),
                expected,
                actual: self.tiles.len(),
            });
        }
        for (index, tile) in self.tiles.iter().enumerate() {
            tile.validate_at(&format!("terrain.tiles[{index}]"))?;
        }
        Ok(())
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

fn default_tile_temperature_bias(normalized_x: f32) -> f32 {
    // Must track sample_temperature's west→east gradient (distance from the
    // vertical equator), or temperature overlays contradict what agents feel.
    ((normalized_x - 0.5).abs() * 2.0).clamp(0.0, 1.0)
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

impl TerrainTile {
    /// Validate every floating value carried by an imported terrain tile.
    pub fn validate(&self) -> Result<(), ScientificStateError> {
        self.validate_at("terrain_tile")
    }

    fn validate_at(&self, path: &str) -> Result<(), ScientificStateError> {
        validate_finite(&format!("{path}.elevation"), self.elevation)?;
        validate_finite(&format!("{path}.moisture"), self.moisture)?;
        validate_finite(&format!("{path}.accent"), self.accent)?;
        validate_finite(&format!("{path}.fertility_bias"), self.fertility_bias)?;
        validate_finite(&format!("{path}.temperature_bias"), self.temperature_bias)
    }
}

const fn default_narrative_interval() -> u32 {
    30
}

const fn default_narrative_capacity() -> usize {
    256
}

/// Turns detector output into the run's *story*: a bounded, deterministic
/// stream of typed events with human-readable prose.
///
/// This layer reads [`TickSummary`] history and nothing else. It never observes
/// or mutates simulation state, so narrating a run cannot change it — the
/// storyteller must not be able to alter the story.
///
/// The prose is **templated and deterministic**, never LLM-authored: the stream
/// has to be diffable across runs and builds, and an LLM reading these events
/// must not be reading its own prose fed back to it.
pub mod narrative {
    use super::{Tick, TickSummary};
    use crate::detect::{
        ChangePoint, CrossDirection, CusumParams, Direction, Regime, RegimeParams, Sample,
        Threshold, change_points_cusum, regimes, threshold_crossings,
    };
    use serde::{Deserialize, Serialize};
    use std::collections::VecDeque;

    /// What kind of thing happened.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum EventKind {
        /// The population fell sharply.
        PopulationCrash,
        /// The population rose sharply.
        PopulationBoom,
        /// The population reached zero.
        Extinction,
        /// Mean energy fell sharply.
        EnergyCollapse,
        /// Mean energy recovered sharply.
        EnergyRecovery,
        /// Combat activity rose sharply.
        CombatSurge,
        /// The population dynamics changed character.
        RegimeChange,
    }

    impl EventKind {
        /// Stable machine-readable identifier.
        #[must_use]
        pub const fn as_str(self) -> &'static str {
            match self {
                Self::PopulationCrash => "population_crash",
                Self::PopulationBoom => "population_boom",
                Self::Extinction => "extinction",
                Self::EnergyCollapse => "energy_collapse",
                Self::EnergyRecovery => "energy_recovery",
                Self::CombatSurge => "combat_surge",
                Self::RegimeChange => "regime_change",
            }
        }
    }

    /// One thing that happened, with the evidence that says so.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EventRecord {
        /// Tick the detector fired at.
        pub tick: Tick,
        /// What happened.
        pub kind: EventKind,
        /// Rough importance in `[0, 1]`, for ranking and for highlight reels.
        pub severity: f32,
        /// Which series this was detected on.
        pub metric: &'static str,
        /// Representative value before the change.
        pub before: f64,
        /// Representative value after the change.
        pub after: f64,
        /// Detector statistic that fired.
        pub score: f64,
        /// Deterministic, templated prose. Never model-generated.
        pub human_text: String,
    }

    /// What counts as *worth telling a human about*.
    ///
    /// A detection can be statistically impeccable and narratively worthless.
    /// In a world of 23 agents, losing one is a 4% drop against a nearly flat
    /// baseline — a textbook significant change, and nobody cares. Measured on
    /// a real 3,000-tick run, a purely statistical stream produced **853 events
    /// per 10k ticks** ("population fell 3% (23 -> 22)", "mean energy collapsed
    /// (0.99 -> 0.98)"), which is not a story, it is static.
    ///
    /// So significance is necessary and not sufficient: an event must ALSO be
    /// material (a big change, in both relative and absolute terms) and it must
    /// not repeat. These floors are the difference between a timeline people
    /// read and one they learn to ignore.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct NarrativePolicy {
        /// Minimum fractional population change (e.g. `0.20` = 20%).
        pub min_population_fraction: f64,
        /// Minimum absolute population change, so tiny worlds do not narrate
        /// every individual birth and death.
        pub min_population_absolute: f64,
        /// Minimum absolute change in mean energy (energy lives in `[0, 2]`).
        pub min_energy_absolute: f64,
        /// Minimum spike-hit rate for a combat surge to be worth reporting.
        pub min_combat_absolute: f64,
        /// Minimum ticks between two events of the same kind.
        pub cooldown_ticks: u64,
    }

    impl Default for NarrativePolicy {
        fn default() -> Self {
            Self {
                min_population_fraction: 0.20,
                min_population_absolute: 5.0,
                min_energy_absolute: 0.15,
                min_combat_absolute: 3.0,
                cooldown_ticks: 200,
            }
        }
    }

    /// Bounded, deduplicated stream of a run's narrative events.
    #[derive(Debug, Default)]
    pub struct RunNarrative {
        events: VecDeque<EventRecord>,
        /// Last tick emitted per event kind. This serves two purposes: a
        /// sliding window re-detects the same change on every pass (dedupe),
        /// and a genuinely churning metric would otherwise emit a new event
        /// every pass (cooldown).
        last_emitted: Vec<(EventKind, u64)>,
        policy: NarrativePolicy,
    }

    impl RunNarrative {
        /// Recently detected events, oldest first.
        #[must_use]
        pub const fn events(&self) -> &VecDeque<EventRecord> {
            &self.events
        }

        /// Run the detectors over the tick history and append anything new.
        pub fn observe<'a, I>(&mut self, history: I, capacity: usize)
        where
            I: Iterator<Item = &'a TickSummary>,
        {
            let summaries: Vec<&TickSummary> = history.collect();
            if summaries.len() < 8 || capacity == 0 {
                return;
            }

            let population: Vec<Sample> = summaries
                .iter()
                .map(|s| Sample::new(s.tick.0, s.agent_count as f64))
                .collect();
            let energy: Vec<Sample> = summaries
                .iter()
                .map(|s| Sample::new(s.tick.0, f64::from(s.average_energy)))
                .collect();
            let combat: Vec<Sample> = summaries
                .iter()
                .map(|s| Sample::new(s.tick.0, f64::from(s.spike_hits)))
                .collect();

            // A warmup longer than the ring would silently detect nothing, so
            // scale it to the available history rather than trusting a default
            // that assumes a long series.
            let warmup = (summaries.len() / 4).clamp(4, 64);
            let params = CusumParams {
                warmup,
                ..CusumParams::default()
            };

            self.emit_changes(&population, &params, "population", capacity, |cp| {
                if cp.direction == Direction::Down {
                    EventKind::PopulationCrash
                } else {
                    EventKind::PopulationBoom
                }
            });
            self.emit_changes(&energy, &params, "average_energy", capacity, |cp| {
                if cp.direction == Direction::Down {
                    EventKind::EnergyCollapse
                } else {
                    EventKind::EnergyRecovery
                }
            });
            self.emit_changes(&combat, &params, "spike_hits", capacity, |cp| {
                if cp.direction == Direction::Up {
                    EventKind::CombatSurge
                } else {
                    // A lull in combat is not a story worth telling; only the
                    // surge is. Reuse the surge kind and let the dedupe drop it.
                    EventKind::CombatSurge
                }
            });

            let extinction = Threshold {
                name: "extinction",
                level: 0.5,
                direction: CrossDirection::Falling,
            };
            if let Ok(crossings) = threshold_crossings(&population, &[extinction]) {
                for crossing in crossings {
                    let record = EventRecord {
                        tick: Tick(crossing.tick),
                        kind: EventKind::Extinction,
                        severity: 1.0,
                        metric: "population",
                        before: crossing.from,
                        after: crossing.to,
                        score: f64::INFINITY.min(f64::MAX),
                        human_text: "population reached zero".to_owned(),
                    };
                    self.push(record, capacity);
                }
            }

            if let Ok(windows) = regimes(&population, RegimeParams::default()) {
                // A regime label that flips every window is noise, not news.
                // Only report a shift that (a) actually lands somewhere
                // dramatic, and (b) PERSISTS into the following window.
                for triple in windows.windows(3) {
                    let (previous, current, next) = (triple[0], triple[1], triple[2]);
                    if previous.regime == current.regime || current.regime != next.regime {
                        continue;
                    }
                    if !matches!(current.regime, Regime::Collapse | Regime::Growth) {
                        continue;
                    }
                    let record = EventRecord {
                        tick: Tick(current.start_tick),
                        kind: EventKind::RegimeChange,
                        severity: 0.4,
                        metric: "population",
                        before: previous.relative_slope,
                        after: current.relative_slope,
                        score: current.autocorrelation,
                        human_text: format!(
                            "population dynamics shifted from {} to {}",
                            regime_word(previous.regime),
                            regime_word(current.regime)
                        ),
                    };
                    self.push(record, capacity);
                }
            }
        }

        /// Is this change big enough that a human would want to hear about it?
        fn is_material(&self, kind: EventKind, before: f64, after: f64) -> bool {
            let delta = (after - before).abs();
            match kind {
                EventKind::PopulationCrash | EventKind::PopulationBoom => {
                    let fraction = if before.abs() > f64::EPSILON {
                        delta / before.abs()
                    } else {
                        1.0
                    };
                    fraction >= self.policy.min_population_fraction
                        && delta >= self.policy.min_population_absolute
                }
                EventKind::EnergyCollapse | EventKind::EnergyRecovery => {
                    delta >= self.policy.min_energy_absolute
                }
                EventKind::CombatSurge => {
                    after >= self.policy.min_combat_absolute && after > before
                }
                // Extinction and regime shifts carry their own gates.
                EventKind::Extinction | EventKind::RegimeChange => true,
            }
        }

        fn emit_changes<F>(
            &mut self,
            series: &[Sample],
            params: &CusumParams,
            metric: &'static str,
            capacity: usize,
            classify: F,
        ) where
            F: Fn(&ChangePoint) -> EventKind,
        {
            let Ok(changes) = change_points_cusum(series, *params) else {
                return;
            };
            for change in &changes {
                let kind = classify(change);
                let before = change.baseline_mean;
                let after = change.baseline_mean + change.magnitude;
                // Statistical significance is necessary, not sufficient.
                if !self.is_material(kind, before, after) {
                    continue;
                }
                let record = EventRecord {
                    tick: Tick(change.tick),
                    kind,
                    severity: severity_from(change.score),
                    metric,
                    before,
                    after,
                    score: change.score,
                    human_text: describe(kind, metric, before, after),
                };
                self.push(record, capacity);
            }
        }

        fn push(&mut self, record: EventRecord, capacity: usize) {
            // Dedupe + cooldown. The sliding window re-detects the same change
            // on every pass, and a churning metric would otherwise emit a fresh
            // event each pass; both produce a stutter that makes the timeline
            // unreadable. One event of a kind per cooldown window.
            if let Some(entry) = self
                .last_emitted
                .iter_mut()
                .find(|(kind, _)| *kind == record.kind)
            {
                if record.tick.0 <= entry.1.saturating_add(self.policy.cooldown_ticks) {
                    return;
                }
                entry.1 = record.tick.0;
            } else {
                self.last_emitted.push((record.kind, record.tick.0));
            }

            if self.events.len() >= capacity {
                self.events.pop_front();
            }
            self.events.push_back(record);
        }
    }

    const fn regime_word(regime: Regime) -> &'static str {
        match regime {
            Regime::Growth => "growth",
            Regime::Equilibrium => "equilibrium",
            Regime::Oscillation => "oscillation",
            Regime::Collapse => "collapse",
        }
    }

    fn severity_from(score: f64) -> f32 {
        // The CUSUM statistic is unbounded; squash it into [0,1] so severities
        // are comparable across metrics and rankable by a highlight reel.
        let normalized = (score / 32.0).clamp(0.0, 1.0);
        normalized as f32
    }

    /// Render deterministic prose. Fixed precision, no locale-dependent
    /// formatting, so the text is byte-stable across runs and platforms.
    fn describe(kind: EventKind, metric: &str, before: f64, after: f64) -> String {
        let delta = after - before;
        let percent = if before.abs() > f64::EPSILON {
            (delta / before) * 100.0
        } else {
            0.0
        };
        match kind {
            EventKind::PopulationCrash => format!(
                "population fell {:.0}% ({:.0} -> {:.0})",
                percent.abs(),
                before,
                after
            ),
            EventKind::PopulationBoom => format!(
                "population rose {:.0}% ({:.0} -> {:.0})",
                percent.abs(),
                before,
                after
            ),
            EventKind::EnergyCollapse => {
                format!("mean energy collapsed ({before:.2} -> {after:.2})")
            }
            EventKind::EnergyRecovery => {
                format!("mean energy recovered ({before:.2} -> {after:.2})")
            }
            EventKind::CombatSurge => format!(
                "combat surged ({:.0} -> {:.0} spike hits per tick)",
                before, after
            ),
            EventKind::Extinction => "population reached zero".to_owned(),
            EventKind::RegimeChange => format!("{metric} dynamics changed"),
        }
    }
}

mod map_sandbox {
    use super::{
        ScientificStateError, TerrainKind, TerrainLayer, TerrainTile, default_tile_fertility_bias,
        default_tile_palette_index, validate_finite, validate_finite_slice,
        validated_cell_count_for,
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
        #[error(transparent)]
        InvalidState(#[from] ScientificStateError),
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
        pub fn new(
            width: u32,
            height: u32,
            values: Vec<f32>,
        ) -> Result<Self, ScientificStateError> {
            let field = Self {
                width,
                height,
                values,
            };
            field.validate_at("scalar_field")?;
            Ok(field)
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

        pub fn validate(&self) -> Result<(), ScientificStateError> {
            self.validate_at("scalar_field")
        }

        fn validate_at(&self, path: &str) -> Result<(), ScientificStateError> {
            let expected = validated_cell_count_for::<f32>(path, self.width, self.height)?;
            if self.values.len() != expected {
                return Err(ScientificStateError::LengthMismatch {
                    path: format!("{path}.values"),
                    expected,
                    actual: self.values.len(),
                });
            }
            validate_finite_slice(&format!("{path}.values"), &self.values)
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

    impl HydrologyTile {
        pub fn validate(&self) -> Result<(), ScientificStateError> {
            self.validate_at("hydrology_tile")
        }

        fn validate_at(&self, path: &str) -> Result<(), ScientificStateError> {
            validate_finite(&format!("{path}.permeability"), self.permeability)?;
            validate_finite(&format!("{path}.runoff_bias"), self.runoff_bias)?;
            validate_finite(&format!("{path}.basin_rank"), self.basin_rank)?;
            validate_finite(&format!("{path}.channel_priority"), self.channel_priority)?;
            validate_finite(&format!("{path}.swim_cost"), self.swim_cost)
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HydrologyTileLayer {
        width: u32,
        height: u32,
        tiles: Vec<HydrologyTile>,
    }

    impl HydrologyTileLayer {
        pub fn new(
            width: u32,
            height: u32,
            tiles: Vec<HydrologyTile>,
        ) -> Result<Self, ScientificStateError> {
            let layer = Self {
                width,
                height,
                tiles,
            };
            layer.validate()?;
            Ok(layer)
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

        pub fn validate(&self) -> Result<(), ScientificStateError> {
            let expected = validated_cell_count_for::<HydrologyTile>(
                "hydrology.tiles",
                self.width,
                self.height,
            )?;
            if self.tiles.len() != expected {
                return Err(ScientificStateError::LengthMismatch {
                    path: "hydrology.tiles".to_owned(),
                    expected,
                    actual: self.tiles.len(),
                });
            }
            for (index, tile) in self.tiles.iter().enumerate() {
                tile.validate_at(&format!("hydrology.tiles[{index}]"))?;
            }
            Ok(())
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
        ) -> Result<Self, ScientificStateError> {
            let field = Self {
                width,
                height,
                flow_directions,
                accumulation,
                spill_elevation,
                basin_ids,
                initial_water_depth,
            };
            field.validate()?;
            Ok(field)
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

        pub fn validate(&self) -> Result<(), ScientificStateError> {
            let expected =
                validated_cell_count_for::<f32>("hydrology.field", self.width, self.height)?;
            for (path, actual) in [
                (
                    "hydrology.field.flow_directions",
                    self.flow_directions.len(),
                ),
                ("hydrology.field.accumulation", self.accumulation.len()),
                (
                    "hydrology.field.spill_elevation",
                    self.spill_elevation.len(),
                ),
                ("hydrology.field.basin_ids", self.basin_ids.len()),
                (
                    "hydrology.field.initial_water_depth",
                    self.initial_water_depth.len(),
                ),
            ] {
                if actual != expected {
                    return Err(ScientificStateError::LengthMismatch {
                        path: path.to_owned(),
                        expected,
                        actual,
                    });
                }
            }
            validate_finite_slice("hydrology.field.accumulation", &self.accumulation)?;
            validate_finite_slice("hydrology.field.spill_elevation", &self.spill_elevation)?;
            validate_finite_slice(
                "hydrology.field.initial_water_depth",
                &self.initial_water_depth,
            )
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
        pub fn new(
            terrain: TerrainLayer,
            fertility: Option<ScalarField>,
            temperature: Option<ScalarField>,
            hydrology_tiles: Option<HydrologyTileLayer>,
            hydrology_field: Option<HydrologyField>,
            metadata: MapArtifactMetadata,
        ) -> Result<Self, ScientificStateError> {
            let artifact = Self {
                terrain,
                fertility,
                temperature,
                hydrology_tiles,
                hydrology_field,
                metadata,
            };
            artifact.validate()?;
            Ok(artifact)
        }

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

        pub fn validate(&self) -> Result<(), ScientificStateError> {
            self.terrain.validate()?;
            if let Some(field) = &self.fertility {
                field.validate_at("map.fertility")?;
                validate_matching_dimensions(
                    "map.fertility.dimensions",
                    self.terrain.width(),
                    self.terrain.height(),
                    field.width(),
                    field.height(),
                )?;
            }
            if let Some(field) = &self.temperature {
                field.validate_at("map.temperature")?;
                validate_matching_dimensions(
                    "map.temperature.dimensions",
                    self.terrain.width(),
                    self.terrain.height(),
                    field.width(),
                    field.height(),
                )?;
            }
            match (&self.hydrology_tiles, &self.hydrology_field) {
                (Some(tiles), Some(field)) => {
                    tiles.validate()?;
                    field.validate()?;
                    validate_matching_dimensions(
                        "map.hydrology_tiles.dimensions",
                        self.terrain.width(),
                        self.terrain.height(),
                        tiles.width(),
                        tiles.height(),
                    )?;
                    validate_matching_dimensions(
                        "map.hydrology_field.dimensions",
                        self.terrain.width(),
                        self.terrain.height(),
                        field.width(),
                        field.height(),
                    )?;
                }
                (None, None) => {}
                _ => {
                    return Err(ScientificStateError::IncompletePair {
                        path: "map.hydrology".to_owned(),
                    });
                }
            }
            Ok(())
        }
    }

    fn validate_matching_dimensions(
        path: &str,
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    ) -> Result<(), ScientificStateError> {
        if (actual_width, actual_height) == (expected_width, expected_height) {
            Ok(())
        } else {
            Err(ScientificStateError::DimensionsMismatch {
                path: path.to_owned(),
                expected_width,
                expected_height,
                actual_width,
                actual_height,
            })
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
            let tile_capacity =
                validated_cell_count_for::<TerrainTile>("map.tiles", width, height)?;

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
            let fertility_field = ScalarField::new(width, height, fertility)?;
            let temperature_field = ScalarField::new(width, height, temperature)?;
            let hydrology_layer = HydrologyTileLayer::new(width, height, hydrology_tiles)?;
            let hydrology_field =
                compute_hydrology_field(width, height, &terrain, &hydrology_layer)?;
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

            Ok(MapArtifact::new(
                terrain,
                Some(fertility_field),
                Some(temperature_field),
                Some(hydrology_layer),
                Some(hydrology_field),
                metadata,
            )?)
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

        for (field, value) in [
            ("fertility_bias", fertility_bias),
            ("temperature_bias", temperature_bias),
            ("elevation", elevation),
            ("moisture", moisture),
            ("accent", accent),
            ("permeability", permeability),
            ("runoff_bias", runoff_bias),
            ("basin_rank", basin_rank),
            ("channel_priority", channel_priority),
            ("swim_cost", swim_cost),
        ] {
            validate_finite(&format!("tileset.tiles[{}].{field}", tile.id), value)?;
        }

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
    ) -> Result<HydrologyField, ScientificStateError> {
        let len = validated_cell_count_for::<Vec<usize>>("hydrology.field", width, height)?;
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
        // Iterative post-order: upstream chains can span the whole map, and
        // one recursion frame per cell overflows the stack on large maps.
        // Frames are (cell, next-child cursor, running total); a node's
        // accumulation stays at 1.0 until its subtree completes, matching the
        // recursive cycle-guard behavior for back-edges.
        let mut stack: Vec<(usize, usize, f32)> = Vec::new();
        visited[idx] = true;
        #[cfg(test)]
        record_accumulation_visit(idx);
        accumulation[idx] = 1.0;
        stack.push((idx, 0, 1.0));
        while let Some(&(node, child_pos, _)) = stack.last() {
            if let Some(&child) = incoming[node].get(child_pos) {
                if let Some(frame) = stack.last_mut() {
                    frame.1 += 1;
                    if visited[child] {
                        frame.2 += accumulation[child];
                    }
                }
                if !visited[child] {
                    visited[child] = true;
                    #[cfg(test)]
                    record_accumulation_visit(child);
                    accumulation[child] = 1.0;
                    stack.push((child, 0, 1.0));
                }
            } else if let Some((finished, _, total)) = stack.pop() {
                accumulation[finished] = total;
                if let Some(parent) = stack.last_mut() {
                    parent.2 += total;
                }
            }
        }
        accumulation[idx]
    }

    #[cfg(test)]
    thread_local! {
        static ACCUMULATION_VISIT_TRACE: std::cell::RefCell<Option<Vec<usize>>> =
            const { std::cell::RefCell::new(None) };
    }

    #[cfg(test)]
    fn record_accumulation_visit(idx: usize) {
        ACCUMULATION_VISIT_TRACE.with(|trace| {
            if let Some(visits) = trace.borrow_mut().as_mut() {
                visits.push(idx);
            }
        });
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

        fn incoming_from_targets(targets: &[Option<usize>]) -> Vec<Vec<usize>> {
            let mut incoming = vec![Vec::new(); targets.len()];
            for (source, target) in targets.iter().enumerate() {
                if let Some(target) = target {
                    assert!(*target < targets.len(), "flow target must be in bounds");
                    incoming[*target].push(source);
                }
            }
            incoming
        }

        /// Frozen copy of the recursive implementation replaced by `e2d9aaa`.
        ///
        /// Keep this deliberately direct: it is an executable characterization
        /// oracle, not an alternative production implementation.
        fn recursive_accumulation_oracle(
            idx: usize,
            incoming: &[Vec<usize>],
            accumulation: &mut [f32],
            visited: &mut [bool],
            visit_trace: &mut Vec<usize>,
        ) -> f32 {
            if visited[idx] {
                return accumulation[idx];
            }
            visited[idx] = true;
            visit_trace.push(idx);
            let mut total = 1.0f32;
            accumulation[idx] = total;
            for &child in &incoming[idx] {
                total += recursive_accumulation_oracle(
                    child,
                    incoming,
                    accumulation,
                    visited,
                    visit_trace,
                );
            }
            accumulation[idx] = total;
            total
        }

        fn run_recursive_oracle(incoming: &[Vec<usize>]) -> (Vec<f32>, Vec<usize>) {
            let mut accumulation = vec![0.0; incoming.len()];
            let mut visited = vec![false; incoming.len()];
            let mut visit_trace = Vec::with_capacity(incoming.len());
            for idx in 0..incoming.len() {
                recursive_accumulation_oracle(
                    idx,
                    incoming,
                    &mut accumulation,
                    &mut visited,
                    &mut visit_trace,
                );
            }
            (accumulation, visit_trace)
        }

        fn run_iterative_production(incoming: &[Vec<usize>]) -> (Vec<f32>, Vec<usize>) {
            ACCUMULATION_VISIT_TRACE.with(|trace| {
                *trace.borrow_mut() = Some(Vec::with_capacity(incoming.len()));
            });

            let mut accumulation = vec![0.0; incoming.len()];
            let mut visited = vec![false; incoming.len()];
            for idx in 0..incoming.len() {
                accumulate_flow(idx, incoming, &mut accumulation, &mut visited);
            }

            let visit_trace = ACCUMULATION_VISIT_TRACE.with(|trace| {
                trace
                    .borrow_mut()
                    .take()
                    .expect("visit tracing was enabled for this traversal")
            });
            (accumulation, visit_trace)
        }

        fn assert_oracle_equivalent(label: &str, incoming: &[Vec<usize>]) {
            let (expected_accumulation, expected_trace) = run_recursive_oracle(incoming);
            let expected_bits = expected_accumulation
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();

            for repetition in 0..4 {
                let (actual_accumulation, actual_trace) = run_iterative_production(incoming);
                let actual_bits = actual_accumulation
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual_bits, expected_bits,
                    "{label}: accumulation differs on repetition {repetition}"
                );
                assert_eq!(
                    actual_trace, expected_trace,
                    "{label}: first-visit order differs on repetition {repetition}"
                );
            }
        }

        fn seeded_targets(seed: u64, len: usize) -> Vec<Option<usize>> {
            let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
            (0..len)
                .map(|_| {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    let sample = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
                    if sample.is_multiple_of(7) {
                        None
                    } else {
                        Some((sample as usize) % len)
                    }
                })
                .collect()
        }

        fn meandering_channel(width: usize, height: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
            let mut path = Vec::with_capacity(width * height);
            for y in 0..height {
                if y.is_multiple_of(2) {
                    path.extend((0..width).map(|x| y * width + x));
                } else {
                    path.extend((0..width).rev().map(|x| y * width + x));
                }
            }

            let mut targets = vec![None; path.len()];
            for pair in path.windows(2) {
                targets[pair[1]] = Some(pair[0]);
            }
            (incoming_from_targets(&targets), path)
        }

        #[test]
        fn iterative_hydrology_matches_frozen_recursive_oracle_on_shapes_and_seeded_graphs() {
            let one_by_n = incoming_from_targets(
                &(0..31)
                    .map(|idx| (idx > 0).then(|| idx - 1))
                    .collect::<Vec<_>>(),
            );
            assert_oracle_equivalent("1xN descending channel", &one_by_n);

            let n_by_one = incoming_from_targets(
                &(0..31)
                    .map(|idx| (idx + 1 < 31).then_some(idx + 1))
                    .collect::<Vec<_>>(),
            );
            assert_oracle_equivalent("Nx1 ascending channel", &n_by_one);

            assert_oracle_equivalent("plateau", &incoming_from_targets(&[None; 36]));

            let basin_center = 12;
            let basin = incoming_from_targets(
                &(0..25)
                    .map(|idx| (idx != basin_center).then_some(basin_center))
                    .collect::<Vec<_>>(),
            );
            assert_oracle_equivalent("basin", &basin);

            let seam_cycle = incoming_from_targets(&[
                Some(4),
                Some(0),
                Some(1),
                Some(2),
                Some(0),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
                Some(5),
            ]);
            assert_oracle_equivalent("wrapped seam cycle", &seam_cycle);

            for seed in 0..96 {
                let len = 1 + ((seed as usize * 37) % 127);
                let incoming = incoming_from_targets(&seeded_targets(seed, len));
                assert_oracle_equivalent(&format!("seeded graph {seed}"), &incoming);
            }
        }

        fn grass_tile() -> TerrainTile {
            TerrainTile {
                kind: TerrainKind::Grass,
                elevation: 0.5,
                moisture: 0.5,
                accent: 0.25,
                fertility_bias: 0.6,
                temperature_bias: 0.4,
                palette_index: 3,
            }
        }

        #[test]
        fn maximal_length_meandering_large_grid_is_stack_safe_and_deterministic() {
            // 512^2 first visits would require 262,144 nested calls in the
            // frozen oracle. This bounded fixture is large enough to exceed a
            // normal test-thread stack while remaining suitable for every CI
            // lane, so only the production iterative traversal runs here.
            const SIDE: usize = 512;
            let (incoming, expected_trace) = meandering_channel(SIDE, SIDE);
            let expected_accumulation = (0..expected_trace.len())
                .map(|depth| (expected_trace.len() - depth) as f32)
                .collect::<Vec<_>>();

            let baseline = run_iterative_production(&incoming);
            assert_eq!(baseline.1.len(), expected_trace.len());
            assert!(
                baseline.1.iter().eq(&expected_trace),
                "large-grid traversal did not follow the meandering channel"
            );
            for (depth, &node) in expected_trace.iter().enumerate() {
                assert_eq!(
                    baseline.0[node].to_bits(),
                    expected_accumulation[depth].to_bits(),
                    "wrong accumulation at meander depth {depth}"
                );
            }

            for repetition in 0..2 {
                assert_eq!(
                    run_iterative_production(&incoming),
                    baseline,
                    "large-grid traversal changed on repetition {repetition}"
                );
            }

            #[cfg(feature = "parallel")]
            for thread_count in [1, 2, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(thread_count)
                    .build()
                    .expect("construct bounded Rayon test pool");
                let threaded = pool.install(|| run_iterative_production(&incoming));
                assert_eq!(
                    threaded, baseline,
                    "ambient Rayon thread count {thread_count} changed traversal"
                );
            }
        }

        #[test]
        fn imported_dense_fields_validate_empty_single_bulk_and_exact_non_finite_paths() {
            ScalarField::new(0, 0, Vec::new()).expect("empty scalar field");
            ScalarField::new(1, 1, vec![-0.0]).expect("single scalar field");
            ScalarField::new(2, 2, vec![0.0, 0.25, 0.5, f32::MIN_POSITIVE])
                .expect("bulk scalar field");

            let error = ScalarField::new(u32::MAX, u32::MAX, Vec::new())
                .expect_err("oversized scalar layout must reject before length comparison");
            assert_eq!(error.path(), "scalar_field");
            assert!(matches!(
                error,
                ScientificStateError::DimensionOverflow { .. }
            ));

            let error = TerrainLayer::from_tiles(u32::MAX, u32::MAX, 1, Vec::new())
                .expect_err("oversized terrain layout must reject before length comparison");
            let super::super::WorldStateError::InvalidState(error) = error else {
                panic!("expected terrain layout state error");
            };
            assert_eq!(error.path(), "terrain.tiles");
            assert!(matches!(
                error,
                ScientificStateError::DimensionOverflow { .. }
            ));

            let error = HydrologyTileLayer::new(u32::MAX, u32::MAX, Vec::new())
                .expect_err("oversized hydrology tile layout must reject");
            assert_eq!(error.path(), "hydrology.tiles");
            assert!(matches!(
                error,
                ScientificStateError::DimensionOverflow { .. }
            ));

            let error = HydrologyField::new(
                u32::MAX,
                u32::MAX,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
            .expect_err("oversized hydrology field layout must reject");
            assert_eq!(error.path(), "hydrology.field");
            assert!(matches!(
                error,
                ScientificStateError::DimensionOverflow { .. }
            ));

            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let error = ScalarField::new(2, 1, vec![0.0, value])
                    .expect_err("non-finite scalar field must fail");
                assert_eq!(error.path(), "scalar_field.values[1]");

                let mut tiles = vec![grass_tile(), grass_tile()];
                tiles[1].moisture = value;
                let error = TerrainLayer::from_tiles(2, 1, 16, tiles)
                    .expect_err("non-finite terrain tile must fail");
                let super::super::WorldStateError::InvalidState(error) = error else {
                    panic!("expected terrain state error");
                };
                assert_eq!(error.path(), "terrain.tiles[1].moisture");

                let hydrology = vec![
                    HydrologyTile {
                        permeability: 0.5,
                        runoff_bias: 0.0,
                        basin_rank: 0.25,
                        channel_priority: 0.5,
                        swim_cost: 1.0,
                    },
                    HydrologyTile {
                        permeability: 0.5,
                        runoff_bias: 0.0,
                        basin_rank: value,
                        channel_priority: 0.5,
                        swim_cost: 1.0,
                    },
                ];
                let error = HydrologyTileLayer::new(2, 1, hydrology)
                    .expect_err("non-finite hydrology tile must fail");
                assert_eq!(error.path(), "hydrology.tiles[1].basin_rank");

                let error = HydrologyField::new(
                    2,
                    1,
                    vec![HydrologyFlowDirection::None; 2],
                    vec![1.0, value],
                    vec![0.5; 2],
                    vec![0; 2],
                    vec![0.0; 2],
                )
                .expect_err("non-finite hydrology vector must fail");
                assert_eq!(error.path(), "hydrology.field.accumulation[1]");
            }
        }

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

        #[test]
        fn map_application_rejects_non_finite_bulk_input_atomically() {
            let tileset = TilesetSpec {
                id: "atomic-map".into(),
                label: None,
                description: None,
                tiles: vec![TileSpec {
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
                }],
                adjacency: Vec::new(),
            };
            let generator = RuleBasedMapGenerator::new(tileset).expect("compile tileset");
            let artifact = generator.generate(8, 8, 16, 7).expect("generate artifact");
            let mut world = super::super::WorldState::new(super::super::ScriptBotsConfig {
                world_width: 128,
                world_height: 128,
                food_cell_size: 16,
                rng_seed: Some(7),
                population_minimum: 0,
                population_spawn_interval: 0,
                ..super::super::ScriptBotsConfig::default()
            })
            .expect("world");
            world
                .apply_map_artifact(&artifact)
                .expect("valid artifact applies");
            let baseline = world.characterization_digest_v0().expect("baseline digest");
            let baseline_revision = world.config_revision();
            let baseline_audit = world.config_audit().to_vec();

            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut invalid = artifact.clone();
                invalid.fertility.as_mut().expect("fertility").values[17] = value;
                let encoded = postcard::to_stdvec(&invalid).expect("encode invalid fixture");
                let imported = postcard::from_bytes::<MapArtifact>(&encoded)
                    .expect("decode untrusted imported-map fixture");
                let error = world
                    .apply_map_artifact(&imported)
                    .expect_err("invalid imported map must fail");
                let super::super::WorldStateError::InvalidState(error) = error else {
                    panic!("expected imported-map state error");
                };
                assert_eq!(error.path(), "map.fertility.values[17]");
                assert_eq!(
                    world
                        .characterization_digest_v0()
                        .expect("unchanged digest"),
                    baseline
                );
                assert_eq!(world.config_revision(), baseline_revision);
                assert_eq!(world.config_audit(), baseline_audit);
            }
        }

        #[test]
        fn rule_based_generator_rejects_unrepresentable_layout_before_wfc_allocation() {
            let generator = RuleBasedMapGenerator::new(TilesetSpec {
                id: "layout-guard".into(),
                label: None,
                description: None,
                tiles: vec![TileSpec {
                    id: "grass".into(),
                    label: None,
                    weight: 1,
                    terrain_kind: TerrainKind::Grass,
                    fertility_bias: None,
                    temperature_bias: None,
                    elevation: None,
                    moisture: None,
                    accent: None,
                    palette_index: None,
                    permeability: None,
                    runoff_bias: None,
                    basin_rank: None,
                    channel_priority: None,
                    swim_cost: None,
                }],
                adjacency: Vec::new(),
            })
            .expect("compile one-tile generator");
            let error = generator
                .generate(u32::MAX, u32::MAX, 1, 7)
                .expect_err("oversized map must reject before WFC allocation");
            let MapGenerationError::InvalidState(error) = error else {
                panic!("expected typed map layout error");
            };
            assert_eq!(error.path(), "map.tiles");
            assert!(matches!(
                error,
                ScientificStateError::DimensionOverflow { .. }
            ));
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
    pub fn new(
        tiles: HydrologyTileLayer,
        field: HydrologyField,
    ) -> Result<Self, ScientificStateError> {
        tiles.validate()?;
        field.validate()?;
        let len = validated_cell_count_for::<f32>(
            "hydrology.water_depth",
            tiles.width(),
            tiles.height(),
        )?;
        if field.width() != tiles.width() || field.height() != tiles.height() {
            return Err(ScientificStateError::DimensionsMismatch {
                path: "hydrology.field.dimensions".to_owned(),
                expected_width: tiles.width(),
                expected_height: tiles.height(),
                actual_width: field.width(),
                actual_height: field.height(),
            });
        }
        let mut water_depth = Vec::with_capacity(len);
        water_depth.extend_from_slice(field.initial_water_depth());
        validate_finite_slice("hydrology.water_depth", &water_depth)?;
        Ok(Self {
            tiles,
            field,
            water_depth,
        })
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

    /// Apply a detached bulk water-depth edit and commit only when every value is finite.
    pub fn try_update_water_depth(
        &mut self,
        update: impl FnOnce(&mut [f32]),
    ) -> Result<(), ScientificStateError> {
        let mut candidate = self.water_depth.clone();
        update(&mut candidate);
        validate_finite_slice("hydrology.water_depth", &candidate)?;
        self.water_depth = candidate;
        Ok(())
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

#[derive(Clone, Copy, Debug)]
struct PersistenceRuntimeTail {
    food_delta: f32,
    spiked: bool,
    sound_output: f32,
    give_intent: f32,
    indicator: IndicatorState,
}

impl PersistenceRuntimeTail {
    fn capture(runtime: &AgentRuntime) -> Self {
        Self {
            food_delta: runtime.food_delta,
            spiked: runtime.spiked,
            sound_output: runtime.sound_output,
            give_intent: runtime.give_intent,
            indicator: runtime.indicator,
        }
    }

    fn restore_into(self, runtime: &mut AgentRuntime) {
        runtime.food_delta = self.food_delta;
        runtime.spiked = self.spiked;
        runtime.sound_output = self.sound_output;
        runtime.give_intent = self.give_intent;
        runtime.indicator = self.indicator;
    }
}

/// Aggregate world state shared by the simulation and rendering layers.
pub struct WorldState {
    config: ScriptBotsConfig,
    tick: Tick,
    epoch: u64,
    rng: SmallRngStream,
    agents: AgentArena,
    identities: AgentMap<AgentIdentity>,
    next_agent_uid: u64,
    next_spawn_ordinal: u64,
    next_birth_ordinal: u64,
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
    work_combat_views: Vec<CombatAgentView>,
    work_penalties: Vec<f32>,
    pending_deaths: Vec<AgentId>,
    #[allow(dead_code)]
    pending_spawns: Vec<SpawnOrder>,
    pending_birth_records: Vec<BirthRecord>,
    pending_death_records: Vec<DeathRecord>,
    pending_lifecycle_birth_metrics: Vec<BirthRecord>,
    pending_lifecycle_death_metrics: Vec<DeathRecord>,
    #[allow(dead_code)]
    replay_tick: u64,
    replay_events: Vec<ReplayEvent>,
    persistence: Box<dyn WorldPersistence>,
    pending_persistence_batch: Option<PersistenceBatch>,
    persistence_fault: Option<PersistenceAdmissionError>,
    brain_fault: Option<BrainSpawnError>,
    last_admitted_persistence_tick: Option<Tick>,
    pending_persistence_runtime_tail: AgentMap<PersistenceRuntimeTail>,
    pending_birth_events: usize,
    pending_death_events: usize,
    pending_spike_attempt_events: u32,
    pending_spike_hit_events: u32,
    last_births: usize,
    last_deaths: usize,
    last_spike_hits: u32,
    last_max_age: u32,
    history: VecDeque<TickSummary>,
    narrative: narrative::RunNarrative,
    pending_interventions: Vec<Intervention>,
    active_effects: Vec<ActiveEffect>,
    #[allow(dead_code)]
    carcass_health_distributed: f32,
    #[allow(dead_code)]
    carcass_reproduction_bonus: f32,
    combat_spike_attempts: u32,
    combat_spike_hits: u32,
    config_audit: Vec<ConfigAuditEntry>,
    config_revision: u64,
    resource_ledger: ResourceLedgerState,
    activation_probe: Option<AgentId>,
    simulation_commands: Vec<SimulationCommand>,
}

impl fmt::Debug for WorldState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorldState")
            .field("config", &self.config)
            .field("tick", &self.tick)
            .field("epoch", &self.epoch)
            .field("closed", &self.config.closed)
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
        let mut rng = config.seeded_rng();
        // Decorrelate terrain noise from the world RNG stream: a plain clone
        // replays identical draws for terrain and the first agent spawns.
        let mut terrain_rng = SmallRng::seed_from_u64(rng.next_u64());
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
            rng,
            agents: AgentArena::new(),
            identities: AgentMap::new(),
            next_agent_uid: 1,
            next_spawn_ordinal: 0,
            next_birth_ordinal: 0,
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
            work_combat_views: Vec::new(),
            work_penalties: Vec::new(),
            pending_deaths: Vec::new(),
            pending_spawns: Vec::new(),
            pending_birth_records: Vec::new(),
            pending_death_records: Vec::new(),
            pending_lifecycle_birth_metrics: Vec::new(),
            pending_lifecycle_death_metrics: Vec::new(),
            replay_tick: 0,
            replay_events: Vec::new(),
            persistence,
            pending_persistence_batch: None,
            persistence_fault: None,
            brain_fault: None,
            last_admitted_persistence_tick: None,
            pending_persistence_runtime_tail: AgentMap::new(),
            pending_birth_events: 0,
            pending_death_events: 0,
            pending_spike_attempt_events: 0,
            pending_spike_hit_events: 0,
            last_births: 0,
            last_deaths: 0,
            last_spike_hits: 0,
            last_max_age: 0,
            history: VecDeque::with_capacity(history_capacity),
            narrative: narrative::RunNarrative::default(),
            pending_interventions: Vec::new(),
            active_effects: Vec::new(),
            carcass_health_distributed: 0.0,
            carcass_reproduction_bonus: 0.0,
            combat_spike_attempts: 0,
            combat_spike_hits: 0,
            config_audit: Vec::with_capacity(32),
            config_revision: 0,
            resource_ledger: ResourceLedgerState::default(),
            activation_probe: None,
            simulation_commands: Vec::new(),
        })
    }

    fn resource_amounts(&self) -> ResourceAmounts {
        ResourceAmounts {
            food: self
                .food
                .cells()
                .iter()
                .map(|value| f64::from(*value))
                .sum(),
            energy: self
                .runtime
                .values()
                .map(|runtime| f64::from(runtime.energy))
                .sum(),
            health: self
                .agents
                .columns()
                .health()
                .iter()
                .map(|value| f64::from(*value))
                .sum(),
        }
    }

    fn capture_resource_amounts(&self) -> Option<ResourceAmounts> {
        self.resource_ledger
            .enabled
            .then(|| self.resource_amounts())
    }

    fn record_resource_change(&mut self, kind: ResourceFlowKind, before: Option<ResourceAmounts>) {
        if let Some(before) = before {
            let after = self.resource_amounts();
            self.resource_ledger.record_change(kind, before, after);
        }
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

    /// Growth multiplier for one food cell, given the droughts currently in force.
    ///
    /// A drought that did not actually suppress regrowth would be theatre: the
    /// event log would say "drought" and the ecosystem would carry on as if
    /// nothing had happened.
    fn drought_scale_for_cell(&self, cell_x: usize, cell_y: usize) -> f32 {
        if self.active_effects.is_empty() {
            return 1.0;
        }
        let cell_size = self.config.food_cell_size as f32;
        let px = (cell_x as f32 + 0.5) * cell_size;
        let py = (cell_y as f32 + 0.5) * cell_size;
        let world_width = self.config.world_width as f32;
        let world_height = self.config.world_height as f32;
        let mut scale = 1.0f32;
        for effect in &self.active_effects {
            if effect.region.contains(px, py, world_width, world_height) {
                scale *= effect.growth_scale;
            }
        }
        scale
    }

    fn apply_food_regrowth(&mut self) {
        let growth = self.config.food_growth_rate;
        let decay = self.config.food_decay_rate;
        let diffusion = self.config.food_diffusion_rate;
        if growth <= 0.0 && decay <= 0.0 && diffusion <= 0.0 {
            return;
        }

        // Droughts scale regrowth per cell. Built once per tick, and only when an
        // effect is actually in force, so the common case costs nothing.
        let drought: Option<Vec<f32>> = if self.active_effects.is_empty() {
            None
        } else {
            let (w, h) = (self.food.width() as usize, self.food.height() as usize);
            let mut scales = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    scales.push(self.drought_scale_for_cell(x, y));
                }
            }
            Some(scales)
        };

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
        // A drought is expressed by scaling the cell's growth multiplier, so all
        // three regrowth paths (SIMD chunk, SIMD remainder, scalar) honour it
        // without any of them having to know droughts exist.
        let droughted_profiles: Vec<FoodCellProfile>;
        let profiles: &[FoodCellProfile] = if let Some(scales) = drought.as_ref() {
            droughted_profiles = self
                .food_profiles
                .iter()
                .zip(scales.iter())
                .map(|(profile, scale)| FoodCellProfile {
                    growth_multiplier: profile.growth_multiplier * scale,
                    ..*profile
                })
                .collect();
            &droughted_profiles
        } else {
            &self.food_profiles
        };
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
                                val_v += f32x4::splat(diffusion) * (neigh - prev_v);
                            }
                            let lane_profiles = idxs.map(|i| {
                                profiles.get(i).copied().unwrap_or(FoodCellProfile {
                                    capacity: food_max,
                                    growth_multiplier: 1.0,
                                    decay_multiplier: 1.0,
                                    fertility: 0.0,
                                    nutrient_density: 0.3,
                                })
                            });
                            let cap_arr = lane_profiles.map(|p| p.capacity);
                            let grow_arr = lane_profiles.map(|p| p.growth_multiplier);
                            let decay_arr = lane_profiles.map(|p| p.decay_multiplier);
                            let cap_v = f32x4::new(cap_arr);
                            let grow_v = f32x4::new(grow_arr);
                            let decay_v = f32x4::new(decay_arr);
                            if decay > 0.0 {
                                val_v -= f32x4::splat(decay) * decay_v * val_v;
                            }
                            if growth > 0.0 && food_max > 0.0 {
                                let norm = val_v / f32x4::splat(food_max);
                                let delta =
                                    f32x4::splat(growth) * grow_v * (f32x4::splat(1.0) - norm);
                                val_v += delta * f32x4::splat(food_max);
                            }
                            // Clamp to capacity and global cap
                            let prev_cap_v = prev_v; // previous_value for max with capacity floor
                            let mut cap_eff_v = cap_v.max(prev_cap_v);
                            let global_cap_v = f32x4::splat(food_max).max(prev_cap_v);
                            // min(capacity, global_cap)
                            cap_eff_v = cap_eff_v.min(global_cap_v).max(f32x4::splat(0.0));
                            let out_v = val_v.max(f32x4::splat(0.0)).min(cap_eff_v);
                            let out_arr = out_v.to_array();
                            row[x] = out_arr[0];
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
                let base_heading = headings[idx];
                for e in 0..NUM_EYES {
                    views[e] = wrap_signed_angle(base_heading + rt.eye_direction[e]);
                    fovc[e] = rt.eye_fov[e].max(0.01);
                }
                self.work_eye_view_dirs[idx] = views;
                self.work_eye_fov_clamped[idx] = fovc;
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

            index.visit_neighbor_buckets(idx, radius, &mut |indices| {
                #[cfg(feature = "simd_wide")]
                {
                    // SIMD-batch smell/sound/hearing; eyes/blood remain per-lane for correctness
                    let (chunks, remainder) = indices.as_chunks::<4>();
                    for chunk in chunks {
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
                        for (lane, &oid) in ids.iter().enumerate() {
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
                        for (lane, &other_idx) in ids.iter().enumerate() {
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
                            {
                                // Same falloff as the scalar path and legacy C++:
                                // (fov - diff)/fov * (radius - dist)/radius.
                                let ang = angle_to(dx, dy);
                                let diff_v = f32x4::new([
                                    angle_difference(eyes_dir[0], ang),
                                    angle_difference(eyes_dir[1], ang),
                                    angle_difference(eyes_dir[2], ang),
                                    angle_difference(eyes_dir[3], ang),
                                ]);
                                let fov_v = f32x4::new([
                                    eyes_fov[0],
                                    eyes_fov[1],
                                    eyes_fov[2],
                                    eyes_fov[3],
                                ]);
                                let fov_factor = ((fov_v - diff_v) / fov_v).max(f32x4::splat(0.0));
                                let intensity_v =
                                    fov_factor * f32x4::splat(traits.eye * dist_factor);
                                let color = colors[other_idx];
                                let mut dens =
                                    f32x4::new([density[0], density[1], density[2], density[3]]);
                                let mut r = f32x4::new([eye_r[0], eye_r[1], eye_r[2], eye_r[3]]);
                                let mut g = f32x4::new([eye_g[0], eye_g[1], eye_g[2], eye_g[3]]);
                                let mut b = f32x4::new([eye_b[0], eye_b[1], eye_b[2], eye_b[3]]);
                                // legacy C++: proximity channel carries an extra d/DIST
                                dens += intensity_v * f32x4::splat(dist / radius);
                                r += intensity_v * f32x4::splat(color[0]);
                                g += intensity_v * f32x4::splat(color[1]);
                                b += intensity_v * f32x4::splat(color[2]);
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
                            // Blood via dot threshold to prune; magnitude via angle diff
                            let align = hx * nx + hy * ny;
                            if align >= cos_bhf {
                                let ang = angle_to(dx, dy);
                                let forward_diff = angle_difference(heading, ang);
                                blood += blood_sensor_contribution(
                                    forward_diff,
                                    dist_factor,
                                    healths[other_idx],
                                );
                            }
                        }
                    }
                    // Remainder (less than 4)
                    for &other_idx in remainder {
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
                            let scalar = traits.eye * dist_factor;
                            let intensity_v = fov_factor * f32x4::splat(scalar);
                            let color = colors[other_idx];
                            let mut dens =
                                f32x4::new([density[0], density[1], density[2], density[3]]);
                            let mut r = f32x4::new([eye_r[0], eye_r[1], eye_r[2], eye_r[3]]);
                            let mut g = f32x4::new([eye_g[0], eye_g[1], eye_g[2], eye_g[3]]);
                            let mut b = f32x4::new([eye_b[0], eye_b[1], eye_b[2], eye_b[3]]);
                            // legacy C++: proximity channel carries an extra d/DIST
                            dens += intensity_v * f32x4::splat(dist / radius);
                            r += intensity_v * f32x4::splat(color[0]);
                            g += intensity_v * f32x4::splat(color[1]);
                            b += intensity_v * f32x4::splat(color[2]);
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
                        let forward_diff = angle_difference(heading, ang);
                        blood += blood_sensor_contribution(
                            forward_diff,
                            dist_factor,
                            healths[other_idx],
                        );
                    }
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

                    for eye in 0..NUM_EYES {
                        // eyes_dir already includes the agent heading (see work_eye_view_dirs)
                        let diff = angle_difference(eyes_dir[eye], ang);
                        let fov = eyes_fov[eye];
                        if diff < fov {
                            let fov_factor = ((fov - diff) / fov).max(0.0);
                            let intensity = traits.eye * fov_factor * dist_factor;
                            // legacy C++: proximity channel carries an extra d/DIST
                            density[eye] += intensity * (dist / radius);
                            let color = colors[other_idx];
                            eye_r[eye] += intensity * color[0];
                            eye_g[eye] += intensity * color[1];
                            eye_b[eye] += intensity * color[2];
                        }
                    }

                    smell += dist_factor;

                    sound += dist_factor * self.work_speed_norm[other_idx];
                    hearing += dist_factor * sound_emitters[other_idx];

                    // Blood via dot(heading_dir, n) >= cos(BLOOD_HALF_FOV)
                    let align = hx * (dx / dist) + hy * (dy / dist);
                    if align >= cos_bhf {
                        let forward_diff = angle_difference(heading, ang);
                        blood += blood_sensor_contribution(
                            forward_diff,
                            dist_factor,
                            healths[other_idx],
                        );
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

    /// Explain what `agent` currently perceives, attributing the neighbour-derived
    /// channels to the neighbours responsible.
    ///
    /// Attribution is computed here, in core, and never re-derived in a UI: the
    /// falloff is subtle (a trait multiplier, a legacy proximity factor on the
    /// density channel only, and trait multipliers applied *after* the neighbour
    /// loop), and a frontend that re-implements it will get it slightly wrong and
    /// then display the wrong explanation with total confidence.
    ///
    /// Only the probed agent is ever explained — this is an on-demand,
    /// population-independent query (see `ACTIVATION_CAPTURE_BUDGET` for the same
    /// principle applied to brain activations).
    ///
    /// Returns `None` if the agent is gone.
    #[must_use]
    pub fn explain_sensors(
        &self,
        agent: AgentId,
        max_contributors: usize,
    ) -> Option<SensorAttribution> {
        let idx = self.agents.index_of(agent)?;
        let observer = self.runtime.get(agent)?;
        let columns = self.agents.columns();
        let positions = columns.positions();
        let colors = columns.colors();
        let healths = columns.health();
        let velocities = columns.velocities();
        let headings = columns.headings();

        let position = positions[idx];
        let heading = headings[idx];
        let traits = observer.trait_modifiers;
        let radius = self.config.sense_radius;
        let radius_sq = radius * radius;
        let world_width = self.config.world_width as f32;
        let world_height = self.config.world_height as f32;
        let max_speed = (self.config.bot_speed * self.config.boost_multiplier).max(1e-3);
        let (hx, hy) = (heading.cos(), heading.sin());
        let cos_bhf = BLOOD_HALF_FOV.cos();

        let mut eye_dirs = [0.0f32; NUM_EYES];
        let mut eye_fovs = [1.0f32; NUM_EYES];
        for eye in 0..NUM_EYES {
            eye_dirs[eye] = wrap_signed_angle(heading + observer.eye_direction[eye]);
            eye_fovs[eye] = observer.eye_fov[eye].max(0.01);
        }

        let mut density = [0.0f32; NUM_EYES];
        let mut eye_r = [0.0f32; NUM_EYES];
        let mut eye_g = [0.0f32; NUM_EYES];
        let mut eye_b = [0.0f32; NUM_EYES];
        let (mut smell, mut sound, mut hearing, mut blood) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut contributions: Vec<SensorContribution> = Vec::new();

        // A full scan rather than a spatial-index query: the index holds the
        // membership of whichever stage last rebuilt it, and an explanation
        // computed against stale buckets would omit real contributors. This runs
        // for one agent, on demand, so O(n) is the right trade.
        for (other_idx, other_id) in self.agents.iter_handles().enumerate() {
            if other_idx == idx {
                continue;
            }
            let dx = toroidal_delta(positions[other_idx].x, position.x, world_width);
            let dy = toroidal_delta(positions[other_idx].y, position.y, world_height);
            let dist_sq = dx.mul_add(dx, dy * dy);
            if dist_sq <= f32::EPSILON || dist_sq > radius_sq {
                continue;
            }
            let dist = dist_sq.sqrt();
            let dist_factor = (radius - dist) / radius;
            if dist_factor <= 0.0 {
                continue;
            }
            let ang = angle_to(dx, dy);
            let color = colors[other_idx];
            let source_uid = self.agent_uid(other_id)?;

            let mut share = SensorContribution {
                source: other_id,
                source_uid,
                bearing: wrap_signed_angle(ang - heading),
                distance: dist,
                color,
                eye_density: [0.0; NUM_EYES],
                eye_rgb: [[0.0; 3]; NUM_EYES],
                smell: 0.0,
                sound: 0.0,
                hearing: 0.0,
                blood: 0.0,
                total: 0.0,
            };

            for eye in 0..NUM_EYES {
                let diff = angle_difference(eye_dirs[eye], ang);
                let fov = eye_fovs[eye];
                if diff >= fov {
                    continue;
                }
                let fov_factor = ((fov - diff) / fov).max(0.0);
                let intensity = traits.eye * fov_factor * dist_factor;
                // The density channel alone carries the legacy proximity factor.
                let density_delta = intensity * (dist / radius);
                share.eye_density[eye] = density_delta;
                share.eye_rgb[eye] = [
                    intensity * color[0],
                    intensity * color[1],
                    intensity * color[2],
                ];
                density[eye] += density_delta;
                eye_r[eye] += intensity * color[0];
                eye_g[eye] += intensity * color[1];
                eye_b[eye] += intensity * color[2];
            }

            let velocity = velocities[other_idx];
            let speed_norm = ((velocity.vx * velocity.vx + velocity.vy * velocity.vy).sqrt()
                / max_speed)
                .clamp(0.0, 1.0);
            let emitter = self
                .runtime
                .get(other_id)
                .map_or(0.0, |rt| rt.sound_multiplier);

            share.smell = dist_factor;
            share.sound = dist_factor * speed_norm;
            share.hearing = dist_factor * emitter;
            smell += share.smell;
            sound += share.sound;
            hearing += share.hearing;

            let align = hx * (dx / dist) + hy * (dy / dist);
            if align >= cos_bhf {
                let forward_diff = angle_difference(heading, ang);
                share.blood =
                    blood_sensor_contribution(forward_diff, dist_factor, healths[other_idx]);
                blood += share.blood;
            }

            share.total = share.eye_density.iter().sum::<f32>()
                + share
                    .eye_rgb
                    .iter()
                    .map(|rgb| rgb[0] + rgb[1] + rgb[2])
                    .sum::<f32>()
                + share.smell
                + share.sound
                + share.hearing
                + share.blood;
            contributions.push(share);
        }

        // Trait multipliers apply to the ACCUMULATED totals, after the neighbour
        // loop — exactly as stage_sense does. Folding them in per-neighbour would
        // be plausible-looking and wrong.
        smell *= traits.smell;
        sound *= traits.sound;
        hearing *= traits.hearing;
        blood *= traits.blood;

        let cell_size = self.config.food_cell_size as f32;
        let food_width = self.food.width();
        let food_height = self.food.height();
        let food_max = self.config.food_max;
        let cell_x = ((position.x / cell_size).floor() as i32).rem_euclid(food_width as i32) as u32;
        let cell_y =
            ((position.y / cell_size).floor() as i32).rem_euclid(food_height as i32) as u32;
        let food_idx = (cell_y as usize) * (food_width as usize) + cell_x as usize;
        let food_value = self.food.cells().get(food_idx).copied().unwrap_or(0.0) / food_max;

        let tick_value = self.tick.0 as f32;
        let clocks = observer.clocks;
        let env_temperature = sample_temperature(&self.config, position.x);
        let discomfort = temperature_discomfort(env_temperature, observer.temperature_preference);

        let mut raw = [0.0f32; INPUT_SIZE];
        raw[0] = density[0];
        raw[1] = eye_r[0];
        raw[2] = eye_g[0];
        raw[3] = eye_b[0];
        raw[4] = food_value;
        raw[5] = density[1];
        raw[6] = eye_r[1];
        raw[7] = eye_g[1];
        raw[8] = eye_b[1];
        raw[9] = sound;
        raw[10] = smell;
        raw[11] = healths[idx] * 0.5;
        raw[12] = density[2];
        raw[13] = eye_r[2];
        raw[14] = eye_g[2];
        raw[15] = eye_b[2];
        raw[16] = (tick_value / clocks[0].max(1.0)).sin().abs();
        raw[17] = (tick_value / clocks[1].max(1.0)).sin().abs();
        raw[18] = hearing;
        raw[19] = blood;
        raw[20] = discomfort;
        raw[21] = density[3];
        raw[22] = eye_r[3];
        raw[23] = eye_g[3];
        raw[24] = eye_b[3];

        let mut clamped = [0.0f32; INPUT_SIZE];
        let mut saturated = [false; INPUT_SIZE];
        for i in 0..INPUT_SIZE {
            clamped[i] = clamp01(raw[i]);
            saturated[i] = raw[i] > 1.0;
        }

        // Deterministic, total order: strongest first, ties broken by a stable
        // agent identity so the same world always explains itself the same way.
        contributions.sort_by(|a, b| {
            b.total
                .total_cmp(&a.total)
                .then_with(|| a.source_uid.cmp(&b.source_uid))
        });
        let truncated = contributions.len().saturating_sub(max_contributors);
        contributions.truncate(max_contributors);

        Some(SensorAttribution {
            agent,
            tick: self.tick,
            raw,
            clamped,
            saturated,
            contributions,
            truncated,
        })
    }

    fn default_outputs(inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        let mut outputs = [0.0; OUTPUT_SIZE];
        let limit = OUTPUT_SIZE.min(INPUT_SIZE);
        outputs[..limit].copy_from_slice(&inputs[..limit]);
        outputs
    }

    fn stage_brains(&mut self) {
        struct BrainJob {
            agent_id: AgentId,
            runner: Option<Box<dyn BrainRunner>>,
            sensors: [f32; INPUT_SIZE],
            capture: bool,
            outputs: [f32; OUTPUT_SIZE],
            activations: Option<BrainActivations>,
        }

        let probe = self.activation_probe;
        // Activation capture is demand-driven AND bounded. Selection alone must
        // never authorize population-wide capture: a single "select all" would
        // otherwise reinstate the per-agent, per-tick layer allocations this
        // gate exists to remove. The probed agent is always captured; selected
        // agents are captured in stable handle order until the budget is spent.
        let mut capture_budget = ACTIVATION_CAPTURE_BUDGET;
        // Pull each runner out of its binding so evaluation can run
        // data-parallel (independent networks, no RNG); results are written
        // back serially in handle order, keeping the stage deterministic.
        let mut jobs: Vec<BrainJob> = Vec::with_capacity(self.agents.len());
        for agent_id in self.agents.iter_handles() {
            if let Some(runtime) = self.runtime.get_mut(agent_id) {
                let probed = probe == Some(agent_id);
                let selected = runtime.selection != SelectionState::None;
                let capture = if probed {
                    true
                } else if selected && capture_budget > 0 {
                    capture_budget -= 1;
                    true
                } else {
                    false
                };
                jobs.push(BrainJob {
                    agent_id,
                    runner: runtime.brain.runner.take(),
                    sensors: runtime.sensors,
                    capture,
                    outputs: [0.0; OUTPUT_SIZE],
                    activations: None,
                });
            }
        }

        let evaluate = |job: &mut BrainJob| {
            if let Some(runner) = job.runner.as_mut() {
                job.outputs = runner.tick(&job.sensors);
                // Activation snapshots feed inspector UIs only; capturing
                // them for every agent allocates layer buffers across the
                // whole population every tick.
                if job.capture {
                    job.activations = runner.snapshot_activations().map(clamp_activations);
                }
            } else {
                job.outputs = Self::default_outputs(&job.sensors);
            }
        };
        #[cfg(feature = "parallel")]
        jobs.par_iter_mut().for_each(evaluate);
        #[cfg(not(feature = "parallel"))]
        jobs.iter_mut().for_each(evaluate);

        for job in jobs {
            if let Some(runtime) = self.runtime.get_mut(job.agent_id) {
                runtime.brain.runner = job.runner;
                runtime.outputs = job.outputs;
                runtime.brain_activations = job.activations;
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

        #[derive(Copy, Clone)]
        struct DecodedOutputs {
            left: f32,
            right: f32,
            color: [f32; 3],
            spike_target: f32,
            boost: bool,
            sound_level: f32,
            give_intent: f32,
        }

        // The one place raw actuator slots become meaning. Every reader goes
        // through the named channels (channels.rs) — hand-indexing this vector
        // is how combat spent months reading the green colour slot as "boost".
        fn decode_outputs(outputs: [f32; OUTPUT_SIZE]) -> DecodedOutputs {
            DecodedOutputs {
                left: outputs.channel_clamped(OutputChannel::WheelLeft),
                right: outputs.channel_clamped(OutputChannel::WheelRight),
                color: [
                    outputs.channel_clamped(OutputChannel::ColorRed),
                    outputs.channel_clamped(OutputChannel::ColorGreen),
                    outputs.channel_clamped(OutputChannel::ColorBlue),
                ],
                spike_target: outputs.channel_clamped(OutputChannel::SpikeTarget),
                boost: outputs.boost_engaged(),
                sound_level: outputs.channel_clamped(OutputChannel::SoundLevel),
                give_intent: outputs.channel_clamped(OutputChannel::GiveIntent),
            }
        }

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
            let (handle_chunks, remainder) = handles.as_chunks::<4>();
            for (chunk_i, chunk) in handle_chunks.iter().enumerate() {
                let base = chunk_i * 4;
                for (lane, &agent_id) in chunk.iter().enumerate() {
                    let idx = base + lane;
                    let Some(rt) = runtime.get(agent_id) else {
                        continue;
                    };
                    let decoded = decode_outputs(rt.outputs);

                    let mut left_speed = decoded.left * bot_speed;
                    let mut right_speed = decoded.right * bot_speed;
                    if decoded.boost {
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
                    let mut ramp_penalty = 0.0;
                    if ramp_rate > 0.0 {
                        let active_energy = (rt.energy - ramp_floor).max(0.0);
                        ramp_penalty = active_energy * ramp_rate;
                        drain += ramp_penalty;
                    }
                    let boost_drain = if decoded.boost && boost_penalty > 0.0 {
                        boost_penalty
                    } else {
                        0.0
                    };
                    drain += boost_drain;
                    let before_topography = drain;
                    if topo_enabled && topo_penalty > 0.0 {
                        if slope_along > 0.0 {
                            drain += slope_along * topo_penalty;
                        } else if slope_along < 0.0 {
                            drain = (drain + slope_along * topo_penalty * 0.5).max(0.0);
                        }
                    }
                    let drain_breakdown = ActuationDrain {
                        basal: metabolism_drain,
                        movement: movement_penalty,
                        ramp: ramp_penalty,
                        boost: boost_drain,
                        topography: drain - before_topography,
                    };
                    let health_delta = -drain;
                    let energy = (rt.energy - drain).max(0.0);

                    let mut spike_length = spike_lengths_snapshot[idx];
                    if spike_length < decoded.spike_target {
                        spike_length = (spike_length + spike_growth).min(decoded.spike_target);
                    } else if spike_length > decoded.spike_target {
                        spike_length = (spike_length - spike_growth).max(decoded.spike_target);
                    }
                    results[idx] = ActuationResult {
                        delta: Some(ActuationDelta {
                            heading,
                            velocity: Velocity::new(vx, vy),
                            position: next_pos,
                            health_delta,
                        }),
                        energy,
                        drain: drain_breakdown,
                        color: decoded.color,
                        spike_length,
                        sound_level: decoded.sound_level,
                        give_intent: decoded.give_intent,
                    };
                }

                // Remainder handled below outside loop
            }

            let base = handles.len() - remainder.len();
            for (o, agent_id) in remainder.iter().enumerate() {
                let idx = base + o;
                let Some(runtime) = runtime.get(*agent_id) else {
                    continue;
                };
                let decoded = decode_outputs(runtime.outputs);

                let mut left_speed = decoded.left * bot_speed;
                let mut right_speed = decoded.right * bot_speed;
                if decoded.boost {
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
                let mut ramp_penalty = 0.0;
                if ramp_rate > 0.0 {
                    let active_energy = (runtime.energy - ramp_floor).max(0.0);
                    ramp_penalty = active_energy * ramp_rate;
                    drain += ramp_penalty;
                }
                let boost_drain = if decoded.boost && boost_penalty > 0.0 {
                    boost_penalty
                } else {
                    0.0
                };
                drain += boost_drain;
                let before_topography = drain;
                if topo_enabled && topo_penalty > 0.0 {
                    if slope_along > 0.0 {
                        drain += slope_along * topo_penalty;
                    } else if slope_along < 0.0 {
                        drain = (drain + slope_along * topo_penalty * 0.5).max(0.0);
                    }
                }
                let drain_breakdown = ActuationDrain {
                    basal: metabolism_drain,
                    movement: movement_penalty,
                    ramp: ramp_penalty,
                    boost: boost_drain,
                    topography: drain - before_topography,
                };
                let health_delta = -drain;
                let energy = (runtime.energy - drain).max(0.0);
                let mut spike_length = spike_lengths_snapshot[idx];
                if spike_length < decoded.spike_target {
                    spike_length = (spike_length + spike_growth).min(decoded.spike_target);
                } else if spike_length > decoded.spike_target {
                    spike_length = (spike_length - spike_growth).max(decoded.spike_target);
                }
                results[idx] = ActuationResult {
                    delta: Some(ActuationDelta {
                        heading,
                        velocity: Velocity::new(vx, vy),
                        position: next_pos,
                        health_delta,
                    }),
                    energy,
                    drain: drain_breakdown,
                    color: decoded.color,
                    spike_length,
                    sound_level: decoded.sound_level,
                    give_intent: decoded.give_intent,
                };
            }
        }

        #[cfg(not(feature = "simd_wide"))]
        let results: Vec<ActuationResult> = collect_handles!(handles, |idx, agent_id| {
            if let Some(runtime) = runtime.get(*agent_id) {
                let decoded = decode_outputs(runtime.outputs);

                let mut left_speed = decoded.left * bot_speed;
                let mut right_speed = decoded.right * bot_speed;
                if decoded.boost {
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
                let mut ramp_penalty = 0.0;
                if ramp_rate > 0.0 {
                    let active_energy = (runtime.energy - ramp_floor).max(0.0);
                    ramp_penalty = active_energy * ramp_rate;
                    drain += ramp_penalty;
                }
                let boost_drain = if decoded.boost && boost_penalty > 0.0 {
                    boost_penalty
                } else {
                    0.0
                };
                drain += boost_drain;
                let before_topography = drain;
                if topo_enabled && topo_penalty > 0.0 {
                    if slope_along > 0.0 {
                        drain += slope_along * topo_penalty;
                    } else if slope_along < 0.0 {
                        drain = (drain + slope_along * topo_penalty * 0.5).max(0.0);
                    }
                }
                let drain_breakdown = ActuationDrain {
                    basal: metabolism_drain,
                    movement: movement_penalty,
                    ramp: ramp_penalty,
                    boost: boost_drain,
                    topography: drain - before_topography,
                };
                let health_delta = -drain;
                let energy = (runtime.energy - drain).max(0.0);

                let mut spike_length = spike_lengths_snapshot[idx];
                if spike_length < decoded.spike_target {
                    spike_length = (spike_length + spike_growth).min(decoded.spike_target);
                } else if spike_length > decoded.spike_target {
                    spike_length = (spike_length - spike_growth).max(decoded.spike_target);
                }
                ActuationResult {
                    delta: Some(ActuationDelta {
                        heading,
                        velocity: Velocity::new(vx, vy),
                        position: next_pos,
                        health_delta,
                    }),
                    energy,
                    drain: drain_breakdown,
                    color: decoded.color,
                    spike_length,
                    sound_level: decoded.sound_level,
                    give_intent: decoded.give_intent,
                }
            } else {
                ActuationResult::default()
            }
        });

        if self.resource_ledger.enabled {
            let mut deltas = [ResourceAmounts::default(); 5];
            let healths = self.agents.columns().health();
            for (idx, agent_id) in handles.iter().enumerate() {
                let Some(runtime) = self.runtime.get(*agent_id) else {
                    continue;
                };
                let total = results[idx].drain.total();
                if total <= f32::EPSILON {
                    continue;
                }
                let energy_loss = (runtime.energy - results[idx].energy).max(0.0);
                let health_after = (healths[idx]
                    + results[idx].delta.as_ref().map_or(0.0, |d| d.health_delta))
                .clamp(0.0, 2.0);
                let health_loss = (healths[idx] - health_after).max(0.0);
                let components = [
                    results[idx].drain.basal,
                    results[idx].drain.movement,
                    results[idx].drain.ramp,
                    results[idx].drain.boost,
                    results[idx].drain.topography,
                ];
                for (flow, component) in deltas.iter_mut().zip(components) {
                    let share = f64::from(component / total);
                    flow.energy -= f64::from(energy_loss) * share;
                    flow.health -= f64::from(health_loss) * share;
                }
            }
            for (kind, delta) in [
                ResourceFlowKind::BasalMetabolism,
                ResourceFlowKind::Movement,
                ResourceFlowKind::MetabolismRamp,
                ResourceFlowKind::Boost,
                ResourceFlowKind::Topography,
            ]
            .into_iter()
            .zip(deltas)
            {
                self.resource_ledger
                    .record(kind, delta, ResourceAmounts::default());
            }
        }

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

            let (handle_chunks, remainder) = handles.as_chunks::<4>();
            for (base, chunk) in handle_chunks.iter().enumerate() {
                let i0 = base * 4;
                let idxs = [chunk[0], chunk[1], chunk[2], chunk[3]];
                // Gather env temps and preferences per lane
                let t0 = sample_temperature(&self.config, positions_snapshot[i0].x);
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
                penalties[i0] = pen[0].max(0.0);
                penalties[i0 + 1] = pen[1].max(0.0);
                penalties[i0 + 2] = pen[2].max(0.0);
                penalties[i0 + 3] = pen[3].max(0.0);
            }

            // Remainder (less than 4)
            let base = handles.len() - remainder.len();
            for (o, agent_id) in remainder.iter().enumerate() {
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

    fn stage_accumulate_food_balance(&mut self) {
        for runtime in self.runtime.values_mut() {
            runtime.food_balance_total += runtime.food_delta;
        }
    }

    fn stage_accumulate_tick_events(&mut self) {
        self.pending_birth_events = self.pending_birth_events.saturating_add(self.last_births);
        self.pending_death_events = self.pending_death_events.saturating_add(self.last_deaths);
        self.pending_spike_attempt_events = self
            .pending_spike_attempt_events
            .saturating_add(self.combat_spike_attempts);
        self.pending_spike_hit_events = self
            .pending_spike_hit_events
            .saturating_add(self.combat_spike_hits);
    }

    fn stage_reset_events(&mut self, preserve_persistence_tail: bool) {
        // Reuse the same secondary-map allocation and populate it during the reset pass that we
        // already owe. This preserves a non-cadence-aligned final tick without allocating and
        // copying a second full agent map on every simulation tick.
        self.pending_persistence_runtime_tail.clear();
        for (agent_id, runtime) in self.runtime.iter_mut() {
            if preserve_persistence_tail {
                self.pending_persistence_runtime_tail
                    .insert(agent_id, PersistenceRuntimeTail::capture(runtime));
            }
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
        self.last_births = 0;
        self.last_deaths = 0;
        self.combat_spike_attempts = 0;
        self.combat_spike_hits = 0;
    }

    fn stage_record_history(&mut self, next_tick: Tick) {
        let agent_count = self.agents.len();
        let total_energy: f32 = self.runtime.values().map(|runtime| runtime.energy).sum();
        let total_health: f32 = self.agents.columns().health().iter().copied().sum();
        let max_age = self
            .agents
            .columns()
            .ages()
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let divisor = agent_count as f32;
        let summary = TickSummary {
            tick: next_tick,
            agent_count,
            births: self.last_births,
            deaths: self.last_deaths,
            total_energy,
            average_energy: if agent_count == 0 {
                0.0
            } else {
                total_energy / divisor
            },
            average_health: if agent_count == 0 {
                0.0
            } else {
                total_health / divisor
            },
            max_age,
            spike_hits: self.combat_spike_hits,
        };
        if self.history.len() >= self.config.history_capacity {
            self.history.pop_front();
        }
        self.history.push_back(summary);
        self.last_spike_hits = self.combat_spike_hits;
        self.last_max_age = max_age;
    }

    /// Queue a deliberate perturbation of the world.
    ///
    /// The intervention is applied at the top of the next tick, inside the
    /// pipeline — never immediately from the caller's thread. Immediate
    /// application would land on whichever tick the world mutex happened to be
    /// free, and a session recorded that way could never be replayed.
    ///
    /// # Errors
    ///
    /// Returns [`WorldStateError::InvalidConfig`] when the intervention is not
    /// honourable as asked. It is rejected, never silently clamped into a
    /// different experiment.
    pub fn enqueue_intervention(
        &mut self,
        intervention: Intervention,
    ) -> Result<(), WorldStateError> {
        intervention.validate()?;
        self.pending_interventions.push(intervention);
        Ok(())
    }

    /// Timed interventions still in force.
    #[must_use]
    pub fn active_effects(&self) -> &[ActiveEffect] {
        &self.active_effects
    }

    /// Apply queued interventions and age the timed ones.
    ///
    /// Runs at the TOP of the tick, before food dynamics and sensing, so a
    /// drought scales this tick's regrowth and a meteor's craters are visible to
    /// every agent's senses on the tick it lands. No agent ever acts on a
    /// half-applied world.
    ///
    /// Applied in queue order, which is the order they were enqueued: the same
    /// command sequence against the same seed produces the same world.
    fn stage_interventions(&mut self) -> ResourceAmounts {
        let mut rejected = ResourceAmounts::default();
        let queued = std::mem::take(&mut self.pending_interventions);
        let world_width = self.config.world_width as f32;
        let world_height = self.config.world_height as f32;
        let cell_size = self.config.food_cell_size as f32;

        for intervention in queued {
            match intervention {
                Intervention::Drought {
                    region,
                    ticks,
                    growth_scale,
                } => {
                    if ticks > 0 {
                        self.active_effects.push(ActiveEffect {
                            region,
                            ticks_remaining: ticks,
                            growth_scale,
                        });
                    }
                }
                Intervention::Bloom { region, amount } => {
                    let cap = self.config.food_max;
                    let (width, height) = (self.food.width(), self.food.height());
                    for cy in 0..height {
                        for cx in 0..width {
                            let (px, py) =
                                ((cx as f32 + 0.5) * cell_size, (cy as f32 + 0.5) * cell_size);
                            if region.contains(px, py, world_width, world_height)
                                && let Some(cell) = self.food.get_mut(cx, cy)
                            {
                                let before = *cell;
                                *cell = (*cell + amount).min(cap);
                                rejected.food += f64::from((amount - (*cell - before)).max(0.0));
                            }
                        }
                    }
                }
                Intervention::Meteor {
                    region,
                    lethality,
                    scorch,
                } => {
                    let (width, height) = (self.food.width(), self.food.height());
                    for cy in 0..height {
                        for cx in 0..width {
                            let (px, py) =
                                ((cx as f32 + 0.5) * cell_size, (cy as f32 + 0.5) * cell_size);
                            if region.contains(px, py, world_width, world_height)
                                && let Some(cell) = self.food.get_mut(cx, cy)
                            {
                                *cell *= 1.0 - scorch;
                            }
                        }
                    }
                    if lethality > 0.0 {
                        let columns = self.agents.columns_mut();
                        let positions: Vec<Position> = columns.positions().to_vec();
                        let healths = columns.health_mut();
                        for (idx, position) in positions.iter().enumerate() {
                            if region.contains(position.x, position.y, world_width, world_height) {
                                let before = healths[idx];
                                healths[idx] = (healths[idx] - lethality).max(0.0);
                                rejected.health +=
                                    f64::from((lethality - (before - healths[idx])).max(0.0));
                            }
                        }
                    }
                }
            }
        }

        // Age timed effects AFTER applying this tick's queue, so an effect that
        // lands with N ticks to live is in force for exactly N ticks. Ageing
        // first would give a freshly-queued N-tick drought N+1 ticks of life —
        // an off-by-one that a caller could never see, and that would quietly
        // corrupt every duration in every study.
        for effect in &mut self.active_effects {
            effect.ticks_remaining = effect.ticks_remaining.saturating_sub(1);
        }
        self.active_effects
            .retain(|effect| effect.ticks_remaining > 0);
        rejected
    }

    /// Run the narrative detectors over the tick history and append any newly
    /// detected events to the bounded run narrative.
    ///
    /// This stage is a pure *reader* of [`TickSummary`] history: it never
    /// observes or mutates simulation state, so enabling it cannot perturb a
    /// run (proved by `narrative_layer_does_not_perturb_the_simulation`).
    fn stage_narrative(&mut self, next_tick: Tick) {
        let interval = self.config.narrative_interval;
        if interval == 0 || !next_tick.0.is_multiple_of(interval as u64) {
            return;
        }
        self.narrative
            .observe(self.history.iter(), self.config.narrative_capacity);
    }

    /// Recently detected narrative events, oldest first.
    #[must_use]
    pub fn narrative_events(&self) -> &VecDeque<narrative::EventRecord> {
        self.narrative.events()
    }

    fn stage_food(&mut self) -> FoodResourceActivity {
        let mut activity = FoodResourceActivity::default();
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
            return activity;
        }

        let intake_rate = self.config.food_intake_rate.max(0.0);
        let waste_rate = self.config.food_waste_rate.max(0.0);
        let reproduction_bonus = self.config.reproduction_food_bonus.max(0.0);
        let fertility_bonus_scale = self.config.reproduction_fertility_bonus.max(0.0);
        let healths = self.agents.columns().health();
        for (idx, agent_id) in handles.iter().enumerate() {
            if let Some(runtime) = self.runtime.get_mut(*agent_id) {
                // legacy C++ gate: a full agent neither eats nor wastes cell food
                if (intake_rate > 0.0 || waste_rate > 0.0) && healths[idx] < 2.0 {
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
                                let left =
                                    runtime.outputs.channel_clamped(OutputChannel::WheelLeft);
                                let right =
                                    runtime.outputs.channel_clamped(OutputChannel::WheelRight);
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
                                let energy_before = runtime.energy;
                                runtime.energy = (runtime.energy + energy_gain).min(2.0);
                                activity.rejected_energy += f64::from(
                                    (energy_gain - (runtime.energy - energy_before)).max(0.0),
                                );
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
            return activity;
        }

        let transfer_rate = self.config.food_transfer_rate;
        if transfer_rate <= 0.0 {
            return activity;
        }
        let distance = if self.config.food_sharing_distance > 0.0 {
            self.config.food_sharing_distance
        } else {
            self.config.food_sharing_radius
        };
        let distance_sq = distance * distance;
        let world_width = self.config.world_width as f32;
        let world_height = self.config.world_height as f32;

        // Sharing is a self-contained simulation stage: rebuild from the exact
        // positions this stage uses rather than relying on `stage_sense` to
        // have populated an index earlier in the tick. This also prevents an
        // actuation move or a direct stage invocation from querying stale or
        // empty buckets.
        self.work_position_pairs.clear();
        self.work_position_pairs.reserve(positions.len());
        for position in positions {
            self.work_position_pairs.push((position.x, position.y));
        }
        if self.index.rebuild(&self.work_position_pairs).is_err() {
            return activity;
        }

        // Defer indicator pulses to avoid borrowing conflicts
        let mut indicator_pulses: Vec<(AgentId, f32, [f32; 3])> = Vec::new();
        let mut recipient_candidates: Vec<usize> = Vec::new();
        for &giver_idx in &sharers {
            let giver_id = handles[giver_idx];
            recipient_candidates.clear();
            self.index
                .visit_neighbor_buckets(giver_idx, distance, &mut |indices| {
                    recipient_candidates.extend_from_slice(indices);
                });
            // Ascending order matches the previous full-population scan.
            recipient_candidates.sort_unstable();
            recipient_candidates.dedup();
            for &recipient_idx in &recipient_candidates {
                if recipient_idx == giver_idx || recipient_idx >= handles.len() {
                    continue;
                }
                let recipient_id = &handles[recipient_idx];
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
                let requested_transfer = transfer_rate.min(giver_energy);
                let actual_transfer = requested_transfer.min(capacity);
                if actual_transfer <= 0.0 {
                    continue;
                }
                activity.rejected_energy +=
                    f64::from((requested_transfer - actual_transfer).max(0.0));
                activity.shared_energy += f64::from(actual_transfer);
                let giver_after = (giver_energy - actual_transfer).max(0.0);
                let recipient_after = (recipient_energy + actual_transfer).min(2.0);
                activity.sharing_delta_energy +=
                    f64::from(giver_after + recipient_after - giver_energy - recipient_energy);
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
        activity
    }
    fn spawn_crossover_agent(&mut self) -> Result<Option<AgentId>, BrainSpawnError> {
        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        let count = handles.len();
        if count < 2 {
            return Ok(None);
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
                    return Ok(None);
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
            None => return Ok(None),
        };
        let partner_runtime = self.runtime.get(partner_id).cloned();

        // Species barrier: require matching brain kinds for sexual reproduction
        if let Some(ref partner_rt) = partner_runtime {
            let parent_kind = parent_runtime.brain.kind();
            let partner_kind = partner_rt.brain.kind();
            let kind_match = parent_kind.is_some() && parent_kind == partner_kind;
            if !kind_match {
                return Ok(None); // fall back to random spawn in caller
            }
        } else {
            return Ok(None);
        }

        let width = self.config.world_width as f32;
        let height = self.config.world_height as f32;
        let parent_uid = self
            .agent_uid(parent_id)
            .expect("live crossover parent must have stable identity");
        let partner_uid = self
            .agent_uid(partner_id)
            .expect("live crossover partner must have stable identity");
        let child_data = self.build_child_data(
            &parent_data,
            Some(&partner_data),
            self.config.reproduction_spawn_jitter,
            self.config.reproduction_spawn_back_distance,
            self.config.reproduction_color_jitter,
            width,
            height,
        );
        let mut child_runtime = self.build_child_runtime(
            &parent_runtime,
            partner_runtime.as_ref(),
            self.config.reproduction_gene_log_capacity,
            parent_uid,
            Some(partner_uid),
        );

        // Preserve the historical RNG position of `spawn_agent`, whose random runtime was
        // immediately replaced below, while constructing any fallible fallback brain before
        // insertion so a factory error cannot leave a partially inserted child.
        let _discarded_runtime = AgentRuntime::new_random(&mut self.rng);

        let child_rates = child_runtime.mutation_rates;
        let inherited_key = parent_runtime.brain.registry_key();
        let parent_was_bound = self
            .runtime
            .get(parent_id)
            .is_some_and(|runtime| runtime.brain.is_bound());
        let parent_kind = parent_runtime.brain.kind().unwrap_or("unknown").to_owned();
        let inherited_runner: Option<Box<dyn BrainRunner>> = {
            let parent_runner = self.runtime.get(parent_id).and_then(|rt| rt.brain.runner());
            let partner_runner = self
                .runtime
                .get(partner_id)
                .and_then(|rt| rt.brain.runner());
            match (parent_runner, partner_runner) {
                (Some(parent), Some(partner)) => {
                    if let Some(runner) = parent.crossover(partner, &mut self.rng) {
                        Some(runner)
                    } else {
                        parent.clone_runner()?
                    }
                }
                (Some(parent), None) => parent.clone_runner()?,
                _ => None,
            }
        };
        if let Some(mut runner) = inherited_runner {
            runner.mutate(&mut self.rng, child_rates.primary, child_rates.secondary)?;
            child_runtime.brain = BrainBinding::inherited(runner, inherited_key);
        } else if let Some(key) = inherited_key {
            let Some(binding) =
                BrainBinding::from_registry(&self.brain_registry, &mut self.rng, key)?
            else {
                return Err(BrainSpawnError::new(
                    parent_kind.clone(),
                    MissingBrainFactory { key },
                ));
            };
            child_runtime.brain = binding;
        } else if parent_was_bound {
            return Err(BrainSpawnError::new(parent_kind, MissingHeritableBrain));
        }

        let child_id = self.insert_agent(child_data, child_runtime, true);
        Ok(Some(child_id))
    }

    fn rollback_population_spawns(&mut self, receipt: PopulationSpawnReceipt) {
        for id in receipt.inserted {
            let removed = self.runtime.remove(id);
            debug_assert!(removed.is_some());
            let identity = self.identities.remove(id);
            debug_assert!(identity.is_some());
        }
        self.agents
            .restore_append_checkpoint(receipt.arena_checkpoint);
        self.rng = receipt.rng_before;
        self.next_agent_uid = receipt.next_agent_uid_before;
        self.next_spawn_ordinal = receipt.next_spawn_ordinal_before;
        self.next_birth_ordinal = receipt.next_birth_ordinal_before;
    }

    fn stage_population(
        &mut self,
        next_tick: Tick,
    ) -> Result<Option<PopulationSpawnReceipt>, BrainSpawnError> {
        if self.config.closed {
            return Ok(None);
        }

        let minimum = self.config.population_minimum;
        let interval = self.config.population_spawn_interval;
        let minimum_requires_spawn = minimum > 0 && self.agents.len() < minimum;
        let scheduled_spawn = interval != 0 && next_tick.0.is_multiple_of(interval as u64);
        if !minimum_requires_spawn && !scheduled_spawn {
            return Ok(None);
        }

        let mut receipt = PopulationSpawnReceipt {
            inserted: Vec::new(),
            arena_checkpoint: self.agents.append_checkpoint(),
            rng_before: self.rng.clone(),
            next_agent_uid_before: self.next_agent_uid,
            next_spawn_ordinal_before: self.next_spawn_ordinal,
            next_birth_ordinal_before: self.next_birth_ordinal,
        };
        let result: Result<(), BrainSpawnError> = (|| {
            while self.agents.len() < minimum {
                receipt.inserted.push(self.spawn_random_agent()?);
            }

            if scheduled_spawn {
                let spawn_count = self.config.population_spawn_count.max(1);
                let crossover_chance = self.config.population_crossover_chance.clamp(0.0, 1.0);
                for _ in 0..spawn_count {
                    let use_crossover = self.agents.len() >= 2
                        && crossover_chance > 0.0
                        && self.rng.random_range(0.0..1.0) < crossover_chance;
                    let spawned = if use_crossover {
                        self.spawn_crossover_agent()?
                    } else {
                        None
                    };
                    if let Some(id) = spawned {
                        receipt.inserted.push(id);
                    } else {
                        receipt.inserted.push(self.spawn_random_agent()?);
                    }
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(Some(receipt)),
            Err(error) => {
                self.rollback_population_spawns(receipt);
                Err(error)
            }
        }
    }

    fn spawn_random_agent(&mut self) -> Result<AgentId, BrainSpawnError> {
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
        // `spawn_agent` historically initialized runtime traits before choosing a registry key.
        // Build the same runtime first so fallible brain construction preserves seeded behavior
        // without leaving a partially inserted agent on error.
        let mut runtime = AgentRuntime::new_random(&mut self.rng);
        let binding = if let Some(key) = self.brain_registry.random_key(&mut self.rng) {
            BrainBinding::from_registry(&self.brain_registry, &mut self.rng, key)?
        } else {
            None
        };
        if let Some(binding) = binding {
            runtime.brain = binding;
        }
        let id = self.insert_agent(data, runtime, false);
        Ok(id)
    }

    fn allocate_identity(&mut self, is_birth: bool) -> AgentIdentity {
        assert!(
            self.next_agent_uid < u64::MAX,
            "AgentUid space exhausted for this run"
        );
        assert!(
            self.next_spawn_ordinal < u64::MAX,
            "agent spawn ordinal space exhausted for this run"
        );
        let birth_ordinal = if is_birth {
            assert!(
                self.next_birth_ordinal < u64::MAX,
                "agent birth ordinal space exhausted for this run"
            );
            let ordinal = self.next_birth_ordinal;
            self.next_birth_ordinal += 1;
            Some(ordinal)
        } else {
            None
        };
        let identity = AgentIdentity {
            uid: AgentUid(self.next_agent_uid),
            spawn_ordinal: self.next_spawn_ordinal,
            birth_ordinal,
        };
        self.next_agent_uid += 1;
        self.next_spawn_ordinal += 1;
        identity
    }

    fn insert_agent(&mut self, data: AgentData, runtime: AgentRuntime, is_birth: bool) -> AgentId {
        let identity = self.allocate_identity(is_birth);
        let id = self.agents.insert(data);
        self.identities.insert(id, identity);
        self.runtime.insert(id, runtime);
        id
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
        if self.index.rebuild(&self.work_position_pairs).is_err() {
            // Same policy as stage_sense: never run queries against an index
            // whose membership reflects a previous tick.
            return;
        }

        let spike_damage = self.config.spike_damage;
        let spike_energy_cost = self.config.spike_energy_cost;
        let index = &self.index;
        // Reuse the compact combat view buffer
        self.work_combat_views.clear();
        self.work_combat_views.reserve(handles.len());
        for id in handles.iter() {
            let view = self
                .runtime
                .get(*id)
                .map(|rt| CombatAgentView {
                    herbivore_tendency: rt.herbivore_tendency,
                    energy: rt.energy,
                    outputs: rt.outputs,
                })
                .unwrap_or_default();
            self.work_combat_views.push(view);
        }
        let runtime_snapshot = &self.work_combat_views;

        let results: Vec<CombatResult> = collect_handles!(handles, |idx, _handle| {
            let mut result = CombatResult::default();
            let attacker_runtime = &runtime_snapshot[idx];

            let is_carnivore = attacker_runtime.herbivore_tendency < carnivore_threshold;
            result.attacker_carnivore = is_carnivore;
            let energy_before = attacker_runtime.energy;

            let spike_power = attacker_runtime
                .outputs
                .channel_clamped(OutputChannel::SpikeTarget);

            // Attack eligibility is the physical spike extension, not the
            // "was stabbed this tick" flag that combat writes on victims.
            if spike_lengths[idx] <= 0.5 {
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
                .channel(OutputChannel::WheelLeft)
                .abs();
            let wheel_right = attacker_runtime
                .outputs
                .channel(OutputChannel::WheelRight)
                .abs();
            let velocity = velocities[idx];
            let speed_mag = (velocity.vx * velocity.vx + velocity.vy * velocity.vy).sqrt();
            // Legacy C++ gates the damage bonus on boost. This line once read
            // the green colour slot instead, rewarding carnivores for being
            // green; the named channel is what makes that unrepresentable.
            let boost_bonus = if attacker_runtime.outputs.boost_engaged() {
                1.0
            } else {
                0.0
            };

            let base_power = spike_damage * spike_power;
            let length_factor = 1.0 + spike_length * length_bonus;
            let speed_factor =
                1.0 + (wheel_left.max(wheel_right) + speed_mag) * speed_bonus + boost_bonus;
            let base_damage = base_power * length_factor * speed_factor;

            let origin = positions[idx];
            let mut hits = Vec::new();
            index.visit_neighbor_buckets(idx, reach, &mut |indices| {
                #[cfg(feature = "simd_wide")]
                {
                    let (chunks, remainder) = indices.as_chunks::<4>();
                    for chunk in chunks {
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
                    for &other_idx in remainder {
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

    fn distribute_carcass_rewards(&mut self, dead: &[(usize, AgentId)]) -> ResourceAmounts {
        let mut rejected = ResourceAmounts::default();
        if dead.is_empty() {
            return rejected;
        }
        let radius = self.config.carcass_distribution_radius;
        let health_base = self.config.carcass_health_reward;
        let reproduction_base = self.config.carcass_reproduction_reward;
        if radius <= 0.0 || (health_base <= 0.0 && reproduction_base <= 0.0) {
            return rejected;
        }

        let handles: Vec<AgentId> = self.agents.iter_handles().collect();
        if handles.is_empty() {
            return rejected;
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
                let (handle_chunks, remainder) = handles.as_chunks::<4>();
                for (chunk_i, chunk) in handle_chunks.iter().enumerate() {
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
                let base = handles.len() - remainder.len();
                for (o, neighbor_id) in remainder.iter().enumerate() {
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
                    let before = healths_mut[idx];
                    healths_mut[idx] = (healths_mut[idx] + *add).min(2.0);
                    rejected.health += f64::from((*add - (healths_mut[idx] - before)).max(0.0));
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
                        let before = runtime.energy;
                        runtime.energy = (runtime.energy + energy).min(2.0);
                        rejected.energy += f64::from((energy - (runtime.energy - before)).max(0.0));
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
        rejected
    }

    fn stage_death_cleanup(&mut self, tick: Tick) -> DeathResourceActivity {
        // Health exhaustion is fatal no matter which stage drained it; the
        // legacy sim erases every health<=0 agent each tick. Without this
        // sweep, agents drained by actuation/metabolism linger as zombies.
        {
            let healths = self.agents.columns().health();
            let exhausted: Vec<AgentId> = self
                .agents
                .iter_handles()
                .enumerate()
                .filter(|(idx, _)| healths[*idx] <= 0.0)
                .map(|(_, id)| id)
                .collect();
            self.pending_deaths.extend(exhausted);
        }
        if self.pending_deaths.is_empty() {
            return DeathResourceActivity::default();
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
            return DeathResourceActivity::default();
        }

        let death_records: Vec<DeathRecord> = {
            let columns = self.agents.columns();
            dead.iter()
                .map(|(idx, agent_id)| {
                    let data = columns.snapshot(*idx);
                    let identity = self
                        .identities
                        .get(*agent_id)
                        .expect("live dying agent must have stable identity");
                    let runtime = self
                        .runtime
                        .get(*agent_id)
                        .expect("live dying agent must have runtime state")
                        .clone();
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

                    DeathRecord {
                        tick,
                        agent_uid: identity.uid,
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
                    }
                })
                .collect()
        };
        if !death_records.is_empty() {
            self.pending_lifecycle_death_metrics
                .extend(death_records.iter().cloned());
            self.pending_death_records.extend(death_records);
        }

        dead.sort_by_key(|(idx, _)| *idx);
        let before_carcass = self.capture_resource_amounts();
        let rejected = self.distribute_carcass_rewards(&dead);
        let after_carcass = before_carcass.map(|_| self.resource_amounts());
        let before_removal = after_carcass;
        let mut removed = 0usize;
        for (_, agent_id) in dead.into_iter().rev() {
            if self.remove_agent(agent_id).is_some() {
                removed += 1;
            }
        }
        self.last_deaths = removed;
        let after_removal = before_removal.map(|_| self.resource_amounts());
        DeathResourceActivity {
            carcass_delta: match (before_carcass, after_carcass) {
                (Some(before), Some(after)) => after.delta_from(before),
                _ => ResourceAmounts::default(),
            },
            removal_delta: match (before_removal, after_removal) {
                (Some(before), Some(after)) => after.delta_from(before),
                _ => ResourceAmounts::default(),
            },
            rejected,
        }
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
        let reproduction_window = self.cadence.reproduction_window(self.tick.next());
        let reproduction_chance = self.cadence.reproduction_chance();

        for (idx, agent_id) in handles.iter().enumerate() {
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

            let (
                parent_runtime_snapshot,
                parent_energy_before_debit,
                parent_reproduction_counter_before_reset,
            ) = {
                let runtime = match self.runtime.get_mut(*agent_id) {
                    Some(rt) => rt,
                    None => continue,
                };
                let energy_before_debit = runtime.energy;
                let reproduction_counter_before_reset = runtime.reproduction_counter;
                runtime.energy -= self.config.reproduction_energy_cost;
                runtime.reproduction_counter = 0.0;
                (
                    runtime.clone(),
                    energy_before_debit,
                    reproduction_counter_before_reset,
                )
            };

            let partner_index =
                self.select_partner_index(idx, &ages, partner_chance, handles.len());
            let partner_data = partner_index.map(|j| parent_snapshots[j]);
            // Cloning an AgentRuntime is deep (logs, sensor arrays); do it only
            // for the one partner of an actual birth, not the whole population.
            let partner_runtime = partner_index.and_then(|j| self.runtime.get(handles[j]).cloned());

            let child_data = self.build_child_data(
                &parent_snapshots[idx],
                partner_data.as_ref(),
                jitter,
                back_offset,
                color_jitter,
                width,
                height,
            );
            let parent_uid = self
                .agent_uid(*agent_id)
                .expect("live birth parent must have stable identity");
            let partner_id = partner_index.map(|j| handles[j]);
            let partner_uid = partner_id.map(|id| {
                self.agent_uid(id)
                    .expect("live birth partner must have stable identity")
            });
            let child_runtime = self.build_child_runtime(
                &parent_runtime_snapshot,
                partner_runtime.as_ref(),
                gene_log_capacity,
                parent_uid,
                partner_uid,
            );
            self.pending_spawns.push(SpawnOrder {
                parent_index: idx,
                parent_id: *agent_id,
                partner_id,
                parent_energy_before_debit,
                parent_reproduction_counter_before_reset,
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

    fn refund_spawn_orders(&mut self, orders: &[SpawnOrder]) {
        for order in orders {
            debug_assert!(
                self.runtime.contains_key(order.parent_id),
                "queued birth parent disappeared before refund"
            );
            if let Some(parent) = self.runtime.get_mut(order.parent_id) {
                parent.energy = order.parent_energy_before_debit;
                parent.reproduction_counter = order.parent_reproduction_counter_before_reset;
            }
        }
    }

    fn abort_pending_spawns(&mut self) {
        let orders = std::mem::take(&mut self.pending_spawns);
        self.refund_spawn_orders(&orders);
        self.last_births = 0;
    }

    fn stage_spawn_commit(&mut self, tick: Tick) -> Result<(), BrainSpawnError> {
        if self.pending_spawns.is_empty() {
            return Ok(());
        }
        let mut orders = std::mem::take(&mut self.pending_spawns);
        orders.sort_by_key(|order| order.parent_index);

        // Construct every offspring brain before inserting any child. This keeps a fallible
        // registry factory from leaving a partially committed birth set. The discarded runtime
        // preserves the historical RNG position of `spawn_agent`, which used to create one and
        // immediately overwrite it for every queued birth.
        let rng_before = self.rng.clone();
        let preparation = (|| -> Result<(), BrainSpawnError> {
            for order in &mut orders {
                let _discarded_runtime = AgentRuntime::new_random(&mut self.rng);
                let inherited_key = order.runtime.brain.registry_key();
                let child_rates = order.runtime.mutation_rates;
                let parent_was_bound = self
                    .runtime
                    .get(order.parent_id)
                    .is_some_and(|runtime| runtime.brain.is_bound());
                let parent_kind = self
                    .runtime
                    .get(order.parent_id)
                    .and_then(|runtime| runtime.brain.kind())
                    .unwrap_or("unknown")
                    .to_owned();
                let inherited_runner: Option<Box<dyn BrainRunner>> = {
                    let parent_runner = self
                        .runtime
                        .get(order.parent_id)
                        .and_then(|runtime| runtime.brain.runner());
                    if let Some(parent_runner) = parent_runner {
                        let partner_runner = order
                            .partner_id
                            .and_then(|partner_id| self.runtime.get(partner_id))
                            .and_then(|runtime| runtime.brain.runner())
                            .filter(|partner| partner.kind() == parent_runner.kind());
                        if let Some(runner) = partner_runner
                            .and_then(|partner| parent_runner.crossover(partner, &mut self.rng))
                        {
                            Some(runner)
                        } else {
                            parent_runner.clone_runner()?
                        }
                    } else {
                        None
                    }
                };
                if let Some(mut runner) = inherited_runner {
                    runner.mutate(&mut self.rng, child_rates.primary, child_rates.secondary)?;
                    order.runtime.brain = BrainBinding::inherited(runner, inherited_key);
                } else if let Some(key) = inherited_key {
                    let kind = order.runtime.brain.kind().unwrap_or("unknown").to_owned();
                    let Some(binding) =
                        BrainBinding::from_registry(&self.brain_registry, &mut self.rng, key)?
                    else {
                        return Err(BrainSpawnError::new(kind, MissingBrainFactory { key }));
                    };
                    order.runtime.brain = binding;
                } else if parent_was_bound {
                    return Err(BrainSpawnError::new(parent_kind, MissingHeritableBrain));
                }
            }
            Ok(())
        })();
        if let Err(error) = preparation {
            self.rng = rng_before;
            self.refund_spawn_orders(&orders);
            self.last_births = 0;
            return Err(error);
        }

        self.last_births = orders.len();
        for order in orders {
            let SpawnOrder {
                parent_index: _,
                parent_id: _,
                partner_id: _,
                parent_energy_before_debit: _,
                parent_reproduction_counter_before_reset: _,
                data,
                runtime,
            } = order;
            let child_id = self.insert_agent(data, runtime, true);
            let child_runtime = self
                .runtime
                .get(child_id)
                .expect("newborn agent must have runtime state");
            let identity = self
                .identities
                .get(child_id)
                .copied()
                .expect("newborn agent must have stable identity");
            let birth_ordinal = identity
                .birth_ordinal
                .expect("newborn agent must have a birth ordinal");
            let idx = self
                .agents
                .index_of(child_id)
                .expect("newborn agent must have dense scalar state");
            let snapshot = self.agents.columns().snapshot(idx);
            let brain_kind = child_runtime.brain.kind().map(str::to_string);
            let brain_key = child_runtime.brain.registry_key();
            let record = BirthRecord {
                tick,
                agent_uid: identity.uid,
                spawn_ordinal: identity.spawn_ordinal,
                birth_ordinal,
                parent_a: child_runtime.lineage[0],
                parent_b: child_runtime.lineage[1],
                brain_kind,
                brain_key,
                herbivore_tendency: clamp01(child_runtime.herbivore_tendency),
                generation: snapshot.generation,
                position: snapshot.position,
                is_hybrid: child_runtime.hybrid,
            };
            self.pending_lifecycle_birth_metrics.push(record.clone());
            self.pending_birth_records.push(record);
        }
        Ok(())
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
        parent_uid: AgentUid,
        partner_uid: Option<AgentUid>,
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
        // parent.clone() keeps registry_key/kind with runner=None; stage_spawn_commit
        // rebinds a live runner from the registry using that key. Without a key no
        // runner can be rehydrated, so drop the stale kind instead of claiming one.
        if runtime.brain.registry_key().is_none() {
            runtime.brain = BrainBinding::default();
        }
        runtime.lineage = [Some(parent_uid), partner_uid];

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
    fn stage_persistence(
        &mut self,
        next_tick: Tick,
        force_partial_batch: bool,
    ) -> Result<(), PersistenceAdmissionError> {
        if self.config.persistence_interval == 0 {
            self.pending_birth_records.clear();
            self.pending_death_records.clear();
            self.pending_lifecycle_birth_metrics.clear();
            self.pending_lifecycle_death_metrics.clear();
            self.replay_events.clear();
            self.pending_birth_events = 0;
            self.pending_death_events = 0;
            self.pending_spike_attempt_events = 0;
            self.pending_spike_hit_events = 0;
            self.pending_persistence_runtime_tail.clear();
            return Ok(());
        }

        let analytics = self.config.analytics_stride;
        if !force_partial_batch
            && !next_tick
                .0
                .is_multiple_of(self.config.persistence_interval as u64)
        {
            if analytics.lifecycle_events == 0 {
                self.pending_lifecycle_birth_metrics.clear();
                self.pending_lifecycle_death_metrics.clear();
            }
            return Ok(());
        }

        let macro_enabled = analytics.macro_metrics != 0
            && (force_partial_batch || next_tick.0.is_multiple_of(analytics.macro_metrics as u64));
        let behavior_enabled = analytics.behavior_metrics != 0
            && (force_partial_batch
                || next_tick
                    .0
                    .is_multiple_of(analytics.behavior_metrics as u64));
        let lifecycle_enabled = analytics.lifecycle_events != 0
            && (force_partial_batch
                || next_tick
                    .0
                    .is_multiple_of(analytics.lifecycle_events as u64));

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
                    let food_delta = if force_partial_batch {
                        self.pending_persistence_runtime_tail
                            .get(*agent_id)
                            .map_or(runtime.food_delta, |tail| tail.food_delta)
                    } else {
                        runtime.food_delta
                    };
                    let delta = f64::from(food_delta);
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
            births: self.pending_birth_events,
            deaths: self.pending_death_events,
            total_energy,
            average_energy,
            average_health,
            max_age: age_max,
            spike_hits: self.pending_spike_hit_events,
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
        if summary.births > 0 {
            events.push(PersistenceEvent::new(
                PersistenceEventKind::Births,
                summary.births,
            ));
        }
        if summary.deaths > 0 {
            events.push(PersistenceEvent::new(
                PersistenceEventKind::Deaths,
                summary.deaths,
            ));
        }
        if self.pending_spike_attempt_events > 0 {
            events.push(PersistenceEvent::new(
                PersistenceEventKind::Custom(Cow::Borrowed("spike_attempts")),
                self.pending_spike_attempt_events as usize,
            ));
        }
        if self.pending_spike_hit_events > 0 {
            events.push(PersistenceEvent::new(
                PersistenceEventKind::Custom(Cow::Borrowed("spike_hits")),
                self.pending_spike_hit_events as usize,
            ));
        }

        let mut agents = Vec::with_capacity(agent_count);
        for id in &handles {
            if let (Some(data), Some(mut runtime)) =
                (self.agents.snapshot(*id), self.runtime.get(*id).cloned())
            {
                if force_partial_batch
                    && let Some(tail) = self.pending_persistence_runtime_tail.get(*id)
                {
                    tail.restore_into(&mut runtime);
                }
                agents.push(AgentState {
                    id: *id,
                    identity: *self
                        .identities
                        .get(*id)
                        .expect("live agent must have stable identity"),
                    data,
                    runtime,
                });
            }
        }

        if lifecycle_enabled && !self.pending_lifecycle_death_metrics.is_empty() {
            let mut combat_carnivore = 0usize;
            let mut combat_herbivore = 0usize;
            let mut starvation = 0usize;
            let mut aging = 0usize;
            let mut unknown = 0usize;
            for record in &self.pending_lifecycle_death_metrics {
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

        if lifecycle_enabled && !self.pending_lifecycle_birth_metrics.is_empty() {
            let total = self.pending_lifecycle_birth_metrics.len();
            let hybrid = self
                .pending_lifecycle_birth_metrics
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

        let births = std::mem::take(&mut self.pending_birth_records);
        let deaths = std::mem::take(&mut self.pending_death_records);
        if lifecycle_enabled || analytics.lifecycle_events == 0 {
            self.pending_lifecycle_birth_metrics.clear();
            self.pending_lifecycle_death_metrics.clear();
        }

        let batch = PersistenceBatch {
            summary: summary.clone(),
            epoch: self.epoch,
            closed: self.config.closed,
            metrics,
            events,
            agents,
            births,
            deaths,
            replay_events: std::mem::take(&mut self.replay_events),
        };
        self.pending_persistence_runtime_tail.clear();
        let persistence_result = match self.persistence.on_tick(&batch) {
            Ok(()) => {
                self.last_admitted_persistence_tick = Some(next_tick);
                Ok(())
            }
            Err(error) => {
                self.persistence_fault = Some(error.clone());
                self.pending_persistence_batch = Some(batch);
                Err(error)
            }
        };
        self.pending_birth_events = 0;
        self.pending_death_events = 0;
        self.pending_spike_attempt_events = 0;
        self.pending_spike_hit_events = 0;
        self.carcass_health_distributed = 0.0;
        self.carcass_reproduction_bonus = 0.0;
        persistence_result
    }

    /// Execute one simulation tick pipeline returning emitted events.
    ///
    /// A returned error can describe a tick that has already reached its completed boundary. A
    /// persistence rejection retains that exact completed batch for explicit retry and sets
    /// [`Self::persistence_fault`]. A brain-construction failure rolls back population inserts and
    /// refuses a partial queued-birth commit, completes the remaining tick bookkeeping and
    /// persistence boundary, advances `tick`, and sets [`Self::brain_fault`]. If both fail, both faults are retained and
    /// [`WorldStepError::BrainAndPersistence`] reports them together. In every case, a latched fault
    /// blocks later science ticks without mutation; callers must not retry `step` as though the
    /// failed return meant the current tick was unapplied.
    pub fn step(&mut self) -> Result<TickEvents, WorldStepError> {
        if let Some(error) = self.latched_step_error() {
            return Err(error);
        }

        let next_tick = self.tick.next();
        let previous_epoch = self.epoch;

        if self.resource_ledger.enabled {
            let opening = self.resource_amounts();
            self.resource_ledger.begin_tick(next_tick, opening);
        }

        let before = self.capture_resource_amounts();
        let intervention_rejection = self.stage_interventions();
        self.record_resource_change(ResourceFlowKind::ScenarioIntervention, before);
        self.resource_ledger.record(
            ResourceFlowKind::CapacityRejection,
            ResourceAmounts::default(),
            intervention_rejection,
        );
        if self.cadence.should_age(next_tick) {
            let before = self.capture_resource_amounts();
            self.stage_aging();
            self.record_resource_change(ResourceFlowKind::Aging, before);
        }
        let before = self.capture_resource_amounts();
        let food_respawned = self.stage_food_dynamics(next_tick);
        self.record_resource_change(ResourceFlowKind::FoodDynamics, before);
        self.stage_sense();
        self.stage_brains();
        self.stage_actuation();
        let before = self.capture_resource_amounts();
        self.stage_temperature_discomfort();
        self.record_resource_change(ResourceFlowKind::TemperatureStress, before);
        let before = self.capture_resource_amounts();
        let food_activity = self.stage_food();
        if let Some(before) = before {
            let mut ground_delta = self.resource_amounts().delta_from(before);
            ground_delta.energy -= food_activity.sharing_delta_energy;
            self.resource_ledger.record(
                ResourceFlowKind::GroundFoodConversion,
                ground_delta,
                ResourceAmounts::default(),
            );
        }
        self.resource_ledger.record(
            ResourceFlowKind::EnergySharing,
            ResourceAmounts {
                energy: food_activity.sharing_delta_energy,
                ..ResourceAmounts::default()
            },
            ResourceAmounts {
                energy: food_activity.shared_energy,
                ..ResourceAmounts::default()
            },
        );
        self.resource_ledger.record(
            ResourceFlowKind::CapacityRejection,
            ResourceAmounts::default(),
            ResourceAmounts {
                energy: food_activity.rejected_energy,
                ..ResourceAmounts::default()
            },
        );
        let before = self.capture_resource_amounts();
        self.stage_combat();
        self.record_resource_change(ResourceFlowKind::Combat, before);
        let death_activity = self.stage_death_cleanup(next_tick);
        self.resource_ledger.record(
            ResourceFlowKind::CarcassReward,
            death_activity.carcass_delta,
            ResourceAmounts::default(),
        );
        self.resource_ledger.record(
            ResourceFlowKind::DeathRemoval,
            death_activity.removal_delta,
            ResourceAmounts::default(),
        );
        self.resource_ledger.record(
            ResourceFlowKind::CapacityRejection,
            ResourceAmounts::default(),
            death_activity.rejected,
        );
        let reproduction_before = self.capture_resource_amounts();
        self.stage_reproduction();
        self.record_resource_change(
            ResourceFlowKind::ReproductionAllocation,
            reproduction_before,
        );
        let population_before = self.capture_resource_amounts();
        let population_result = self.stage_population(next_tick);
        self.record_resource_change(ResourceFlowKind::PopulationInjection, population_before);
        let brain_result = match population_result {
            Ok(population_receipt) => {
                let spawn_before = self.capture_resource_amounts();
                let spawn_result = self.stage_spawn_commit(next_tick);
                self.record_resource_change(ResourceFlowKind::ReproductionAllocation, spawn_before);
                match spawn_result {
                    Ok(()) => Ok(()),
                    Err(error) => {
                        if let Some(receipt) = population_receipt {
                            let rollback_before = self.capture_resource_amounts();
                            self.rollback_population_spawns(receipt);
                            self.record_resource_change(
                                ResourceFlowKind::PopulationInjection,
                                rollback_before,
                            );
                        }
                        Err(error)
                    }
                }
            }
            Err(error) => {
                let abort_before = self.capture_resource_amounts();
                self.abort_pending_spawns();
                self.record_resource_change(ResourceFlowKind::ReproductionAllocation, abort_before);
                Err(error)
            }
        };
        self.stage_accumulate_food_balance();
        self.stage_accumulate_tick_events();
        self.stage_record_history(next_tick);
        self.stage_narrative(next_tick);
        let preserve_persistence_tail = self.config.persistence_interval != 0
            && !next_tick
                .0
                .is_multiple_of(self.config.persistence_interval as u64);
        let persistence_result = self.stage_persistence(next_tick, false);

        let mut events = TickEvents {
            tick: next_tick,
            charts_flushed: self.cadence.should_emit_chart_event(next_tick),
            epoch_rolled: false,
            food_respawned,
        };

        self.stage_reset_events(preserve_persistence_tail);
        if self.resource_ledger.enabled {
            let closing = self.resource_amounts();
            self.resource_ledger.finish_tick(closing);
        }
        self.advance_tick();
        events.tick = self.tick;
        events.epoch_rolled = self.epoch != previous_epoch;
        match (brain_result, persistence_result) {
            (Ok(()), Ok(())) => Ok(events),
            (Ok(()), Err(persistence)) => Err(persistence.into()),
            (Err(brain), Ok(())) => {
                self.brain_fault = Some(brain.clone());
                Err(brain.into())
            }
            (Err(brain), Err(persistence)) => {
                self.brain_fault = Some(brain.clone());
                Err(WorldStepError::BrainAndPersistence { brain, persistence })
            }
        }
    }

    /// Returns an immutable reference to configuration.
    #[must_use]
    pub fn config(&self) -> &ScriptBotsConfig {
        &self.config
    }

    /// Enable or disable per-stage resource accounting for future ticks.
    ///
    /// The report accumulated before disabling is retained. Accounting state
    /// is diagnostic-only: it is excluded from characterization and world
    /// digests, persistence, replay, RNG use, and every simulation decision.
    pub fn set_resource_ledger_enabled(&mut self, enabled: bool) {
        self.resource_ledger.set_enabled(enabled);
    }

    /// Whether future ticks will produce resource-ledger reports.
    #[must_use]
    pub const fn resource_ledger_enabled(&self) -> bool {
        self.resource_ledger.enabled
    }

    /// Latest and cumulative immutable resource accounting.
    #[must_use]
    pub const fn resource_ledger(&self) -> &ResourceLedgerReport {
        &self.resource_ledger.report
    }

    /// Fingerprint deterministic science state at a quiescent tick boundary.
    ///
    /// Version zero hashes explicit little-endian integers and `f32::to_bits()` values with
    /// domain-separated FNV-1a 64. It includes time/closure state; agents sorted by raw `AgentId`;
    /// scalar agent data; science-relevant runtime fields; food; terrain; deterministic map
    /// metadata; hydrology; registered brain key/kind pairs; and four samples from a cloned RNG.
    ///
    /// UI-owned selection/indicator state, activation snapshots, mutation log strings, history,
    /// configuration/audit state, analytics and persistence buffers, derived indexes, scratch
    /// vectors, the map generation timestamp, factory closures, and opaque evaluator state are
    /// excluded. The restorable random-stream state and stable identity-allocation counters are
    /// intentionally not encoded into this historical V0 format. Therefore this is a stable V0
    /// regression oracle for one pinned build lane, not a checkpoint or replay guarantee.
    pub fn characterization_digest_v0(
        &self,
    ) -> Result<CharacterizationDigestV0, CharacterizationError> {
        if !self.pending_deaths.is_empty()
            || !self.pending_spawns.is_empty()
            || !self.simulation_commands.is_empty()
            // A queued intervention is undelivered science: digesting now would
            // fingerprint a world that is about to change for reasons the digest
            // cannot see.
            || !self.pending_interventions.is_empty()
        {
            return Err(CharacterizationError::NonQuiescent {
                pending_deaths: self.pending_deaths.len(),
                pending_spawns: self.pending_spawns.len(),
                simulation_commands: self.simulation_commands.len()
                    + self.pending_interventions.len(),
            });
        }

        let mut handles: Vec<_> = self.agents.iter_handles().collect();
        handles.sort_unstable_by_key(|id| id.data().as_ffi());

        let mut agents_encoder = CharacterizationEncoderV0::new("agents");
        agents_encoder.usize(handles.len());
        for id in handles {
            let raw_id = id.data().as_ffi();
            let data = self
                .agents
                .snapshot(id)
                .ok_or(CharacterizationError::MissingAgentData { agent_id: raw_id })?;
            let runtime = self
                .runtime
                .get(id)
                .ok_or(CharacterizationError::MissingAgentRuntime { agent_id: raw_id })?;
            agents_encoder.u64(raw_id);
            encode_agent_data_v0(&mut agents_encoder, data);
            encode_agent_runtime_v0(&mut agents_encoder, runtime);
        }
        let agents = agents_encoder.finish();

        let mut food_encoder = CharacterizationEncoderV0::new("food");
        food_encoder.u32(self.food.width());
        food_encoder.u32(self.food.height());
        food_encoder.usize(self.food.cells().len());
        for &value in self.food.cells() {
            food_encoder.f32(value);
        }
        let food = food_encoder.finish();

        let mut terrain_encoder = CharacterizationEncoderV0::new("terrain");
        terrain_encoder.u32(self.terrain.width());
        terrain_encoder.u32(self.terrain.height());
        terrain_encoder.u32(self.terrain.cell_size());
        terrain_encoder.usize(self.terrain.tiles().len());
        for tile in self.terrain.tiles() {
            terrain_encoder.u8(terrain_kind_tag_v0(tile.kind));
            terrain_encoder.f32(tile.elevation);
            terrain_encoder.f32(tile.moisture);
            terrain_encoder.f32(tile.accent);
            terrain_encoder.f32(tile.fertility_bias);
            terrain_encoder.f32(tile.temperature_bias);
            terrain_encoder.u16(tile.palette_index);
        }
        terrain_encoder.bool(self.map_metadata.is_some());
        if let Some(metadata) = &self.map_metadata {
            terrain_encoder.u8(map_generator_tag_v0(metadata.generator));
            terrain_encoder.string(&metadata.tileset_id);
            terrain_encoder.u64(metadata.tileset_hash);
            terrain_encoder.u64(metadata.seed);
            terrain_encoder.u32(metadata.width);
            terrain_encoder.u32(metadata.height);
            terrain_encoder.usize(metadata.attempt_count);
            terrain_encoder.usize(metadata.succeeded_on);
        }
        let terrain = terrain_encoder.finish();

        let hydrology = self.hydrology.as_ref().map(|state| {
            let mut encoder = CharacterizationEncoderV0::new("hydrology");
            let tiles = state.tiles();
            encoder.u32(tiles.width());
            encoder.u32(tiles.height());
            encoder.usize(tiles.tiles().len());
            for tile in tiles.tiles() {
                encoder.f32(tile.permeability);
                encoder.f32(tile.runoff_bias);
                encoder.f32(tile.basin_rank);
                encoder.f32(tile.channel_priority);
                encoder.f32(tile.swim_cost);
            }
            let field = state.field();
            encoder.u32(field.width());
            encoder.u32(field.height());
            encoder.usize(field.flow_directions().len());
            for &direction in field.flow_directions() {
                encoder.u8(hydrology_flow_tag_v0(direction));
            }
            encoder.usize(field.accumulation().len());
            for &value in field.accumulation() {
                encoder.f32(value);
            }
            encoder.usize(field.spill_elevation().len());
            for &value in field.spill_elevation() {
                encoder.f32(value);
            }
            encoder.usize(field.basin_ids().len());
            for &value in field.basin_ids() {
                encoder.u32(value);
            }
            encoder.usize(field.initial_water_depth().len());
            for &value in field.initial_water_depth() {
                encoder.f32(value);
            }
            encoder.usize(state.water_depth().len());
            for &value in state.water_depth() {
                encoder.f32(value);
            }
            encoder.finish()
        });

        let mut rng = self.rng.clone();
        let mut rng_encoder = CharacterizationEncoderV0::new("rng-probe");
        for _ in 0..4 {
            rng_encoder.u64(rng.next_u64());
        }
        let rng_probe = rng_encoder.finish();

        let mut registrations: Vec<_> = self.brain_registry.entries.iter().collect();
        registrations.sort_unstable_by_key(|(key, _)| **key);
        let mut brain_encoder = CharacterizationEncoderV0::new("brain-registry");
        brain_encoder.u64(self.brain_registry.next_key);
        brain_encoder.usize(registrations.len());
        for (key, entry) in registrations {
            brain_encoder.u64(*key);
            brain_encoder.string(entry.kind.as_ref());
        }
        let brain_registry = brain_encoder.finish();

        // Interventions in force are science state: a drought changes what the
        // world does next. A digest that ignored them would certify two runs as
        // identical while one of them was in the middle of a famine, and any
        // replay proof built on it would be hollow.
        let mut effects_encoder = CharacterizationEncoderV0::new("effects");
        effects_encoder.usize(self.active_effects.len());
        for effect in &self.active_effects {
            match effect.region {
                Region::All => effects_encoder.u8(0),
                Region::Disc { x, y, radius } => {
                    effects_encoder.u8(1);
                    effects_encoder.f32(x);
                    effects_encoder.f32(y);
                    effects_encoder.f32(radius);
                }
            }
            effects_encoder.u32(effect.ticks_remaining);
            effects_encoder.f32(effect.growth_scale);
        }
        let effects = effects_encoder.finish();

        let mut overall_encoder = CharacterizationEncoderV0::new("overall");
        overall_encoder.u64(self.tick.0);
        overall_encoder.u64(self.epoch);
        overall_encoder.bool(self.config.closed);
        overall_encoder.string(&agents);
        overall_encoder.string(&food);
        overall_encoder.string(&terrain);
        overall_encoder.option_string(hydrology.as_deref());
        overall_encoder.string(&rng_probe);
        overall_encoder.string(&brain_registry);
        overall_encoder.string(&effects);
        let overall = overall_encoder.finish();

        Ok(CharacterizationDigestV0 {
            schema: CHARACTERIZATION_DIGEST_V0_SCHEMA.to_owned(),
            algorithm: "fnv1a64-v0".to_owned(),
            tick: self.tick,
            overall,
            agents,
            food,
            terrain,
            hydrology,
            rng_probe,
            brain_registry,
        })
    }

    /// Queue a simulation control request for external renderers.
    pub fn enqueue_simulation_command(
        &mut self,
        mut command: SimulationCommand,
    ) -> Result<(), WorldStateError> {
        command.validate()?;
        if let Some(speed) = command.speed_multiplier.as_mut() {
            *speed = speed.clamp(0.0, 32.0);
        }
        self.simulation_commands.push(command);
        Ok(())
    }

    /// Drain pending simulation control requests (clearing the queue).
    #[must_use]
    pub fn drain_simulation_commands(&mut self) -> Vec<SimulationCommand> {
        if self.simulation_commands.is_empty() {
            Vec::new()
        } else {
            std::mem::take(&mut self.simulation_commands)
        }
    }

    fn record_config_audit(&mut self, patch: serde_json::Value) {
        self.config_audit.push(ConfigAuditEntry {
            tick: self.tick.0,
            patch,
        });
        if self.config_audit.len() > 64 {
            let drop_count = self.config_audit.len() - 64;
            self.config_audit.drain(0..drop_count);
        }
    }

    /// Apply a new configuration, refreshing derived caches while preserving runtime state.
    pub fn apply_config_update(
        &mut self,
        new_config: ScriptBotsConfig,
    ) -> Result<(), WorldStateError> {
        let (food_w, food_h) = new_config.food_dimensions()?;
        let current_dims = (self.food.width(), self.food.height());
        // Compare the raw geometry too: proportional width/cell_size changes
        // keep the derived food dims identical while rescaling the world.
        if (food_w, food_h) != current_dims
            || new_config.world_width != self.config.world_width
            || new_config.world_height != self.config.world_height
            || new_config.food_cell_size != self.config.food_cell_size
        {
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

        if let Ok(value) = serde_json::to_value(&new_config) {
            self.record_config_audit(value);
        }

        self.config = new_config;
        self.food_profiles = food_profiles;
        self.index = new_index;
        self.cadence = TickCadence::from_config(&self.config);
        self.config_revision = self.config_revision.saturating_add(1);
        Ok(())
    }

    /// Monotonic count of applied configuration updates. Unlike the capped
    /// audit log length, this never plateaus, so caches can key off it.
    #[must_use]
    pub const fn config_revision(&self) -> u64 {
        self.config_revision
    }

    /// Ask the next ticks to capture brain activations for `agent` (in
    /// addition to any hovered/selected agents). Frontends set this for the
    /// agent their inspector is focused on; `None` disables the extra probe.
    pub fn set_activation_probe(&mut self, agent: Option<AgentId>) {
        self.activation_probe = agent;
    }

    /// Replace the persistence sink.
    pub fn set_persistence(&mut self, persistence: Box<dyn WorldPersistence>) {
        self.persistence = persistence;
    }

    /// Retry the exact completed batch retained after an unacknowledged admission attempt.
    ///
    /// Returns `Ok(true)` after admitting a retained batch, `Ok(false)` when no retry was
    /// pending, and leaves the world latched on the same batch after another definite or
    /// indeterminate admission failure.
    pub fn retry_pending_persistence(&mut self) -> Result<bool, PersistenceAdmissionError> {
        let Some(batch) = self.pending_persistence_batch.take() else {
            debug_assert!(self.persistence_fault.is_none());
            return Ok(false);
        };
        match self.persistence.on_tick(&batch) {
            Ok(()) => {
                self.last_admitted_persistence_tick = Some(batch.summary.tick);
                self.persistence_fault = None;
                Ok(true)
            }
            Err(error) => {
                self.persistence_fault = Some(error.clone());
                self.pending_persistence_batch = Some(batch);
                Err(error)
            }
        }
    }

    /// Admit the final partial persistence-cadence batch, if one exists.
    ///
    /// This is idempotent at a completed tick boundary. It proves only the configured sink's
    /// synchronous admission guarantee. The FrankenSQLite file sink thereby proves its durable
    /// outbox commit; callers still need a flush or shutdown receipt before claiming scientific
    /// table application and terminal durable-watermark advancement.
    pub fn finalize_persistence(&mut self) -> Result<bool, PersistenceAdmissionError> {
        if let Some(error) = &self.persistence_fault {
            return Err(error.clone());
        }
        if self.config.persistence_interval == 0
            || self.tick == Tick::zero()
            || self.last_admitted_persistence_tick == Some(self.tick)
        {
            return Ok(false);
        }

        self.stage_persistence(self.tick, true)?;
        Ok(true)
    }

    /// Whether a completed tick is paused at the persistence admission boundary.
    #[must_use]
    pub const fn has_pending_persistence_batch(&self) -> bool {
        self.pending_persistence_batch.is_some()
    }

    /// Latched admission error that prevents any later science tick from starting.
    #[must_use]
    pub const fn persistence_fault(&self) -> Option<&PersistenceAdmissionError> {
        self.persistence_fault.as_ref()
    }

    /// Latched brain-construction error that prevents any later science tick from starting.
    #[must_use]
    pub const fn brain_fault(&self) -> Option<&BrainSpawnError> {
        self.brain_fault.as_ref()
    }

    /// Combined typed view of any terminal fault that prevents a later science tick.
    #[must_use]
    pub fn latched_step_error(&self) -> Option<WorldStepError> {
        match (&self.brain_fault, &self.persistence_fault) {
            (Some(brain), Some(persistence)) => Some(WorldStepError::BrainAndPersistence {
                brain: brain.clone(),
                persistence: persistence.clone(),
            }),
            (Some(brain), None) => Some(WorldStepError::BrainSpawn(brain.clone())),
            (None, Some(persistence)) => Some(WorldStepError::Persistence(persistence.clone())),
            (None, None) => None,
        }
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
        self.config.closed
    }

    pub fn config_audit(&self) -> &[ConfigAuditEntry] {
        &self.config_audit
    }

    /// Apply the closed-world population policy at the current completed tick boundary.
    ///
    /// Closing disables both floor enforcement and scheduled injection without altering their
    /// configured values. Scheduled opportunities while closed are skipped, not deferred. A later
    /// open transition makes the existing floor and cadence effective for subsequent ticks.
    pub fn set_closed(&mut self, closed: bool) {
        if self.config.closed == closed {
            return;
        }
        self.config.closed = closed;
        self.config_revision = self.config_revision.saturating_add(1);
        self.record_config_audit(serde_json::json!({ "closed": closed }));
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
    pub fn rng(&mut self) -> &mut dyn RandomStream {
        &mut self.rng
    }

    /// Read-only access to the agent arena.
    #[must_use]
    pub fn agents(&self) -> &AgentArena {
        &self.agents
    }

    /// Internal mutable access to trusted agent storage.
    #[cfg(test)]
    #[must_use]
    fn agents_mut(&mut self) -> &mut AgentArena {
        &mut self.agents
    }

    /// Number of live agents.
    #[must_use]
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Return the stable logical identity metadata for a live SlotMap handle.
    #[must_use]
    pub fn agent_identity(&self, id: AgentId) -> Option<AgentIdentity> {
        self.identities.get(id).copied()
    }

    /// Return the stable logical UID for a live SlotMap handle.
    #[must_use]
    pub fn agent_uid(&self, id: AgentId) -> Option<AgentUid> {
        self.agent_identity(id).map(|identity| identity.uid)
    }

    /// Capture the restorable state of the current, single world random stream.
    #[must_use]
    pub fn random_stream_state(&self) -> RandomStreamState {
        self.rng.checkpoint()
    }

    /// Next stable identity and creation ordinals, for manifest/checkpoint protocols.
    #[must_use]
    pub const fn identity_sequence_state(&self) -> (u64, u64, u64) {
        (
            self.next_agent_uid,
            self.next_spawn_ordinal,
            self.next_birth_ordinal,
        )
    }

    /// Spike hits recorded during the most recent tick.
    pub fn last_spike_hits(&self) -> u32 {
        self.last_spike_hits
    }

    /// Maximum agent age observed during the most recent tick.
    pub fn last_max_age(&self) -> u32 {
        self.last_max_age
    }

    /// Validate and spawn one direct-Rust agent without consuming RNG or allocator state on
    /// rejection.
    pub fn try_spawn_agent(&mut self, agent: AgentData) -> Result<AgentId, ScientificStateError> {
        self.try_spawn_agent_with(agent, |_| {})
    }

    /// Validate scalar and caller-customized runtime state as one atomic spawn transaction.
    pub fn try_spawn_agent_with(
        &mut self,
        agent: AgentData,
        update_runtime: impl FnOnce(&mut AgentRuntime),
    ) -> Result<AgentId, ScientificStateError> {
        agent.validate()?;
        let rng_before = self.rng.clone();
        let mut runtime = AgentRuntime::new_random(&mut self.rng);
        update_runtime(&mut runtime);
        if let Err(error) = runtime.validate_at("agent.runtime") {
            self.rng = rng_before;
            return Err(error);
        }
        let id = self.insert_agent(agent, runtime, false);
        Ok(id)
    }

    /// Internal spawn path for values constructed by trusted simulation logic.
    #[cfg(test)]
    fn spawn_agent(&mut self, agent: AgentData) -> AgentId {
        debug_assert!(agent.validate().is_ok());
        let runtime = AgentRuntime::new_random(&mut self.rng);
        debug_assert!(runtime.validate().is_ok());
        self.insert_agent(agent, runtime, false)
    }

    /// Remove an agent by handle, returning its last known data.
    pub fn remove_agent(&mut self, id: AgentId) -> Option<AgentData> {
        self.runtime.remove(id);
        self.identities.remove(id);
        self.agents.remove(id)
    }

    /// Immutable access to the food grid.
    #[must_use]
    pub fn food(&self) -> &FoodGrid {
        &self.food
    }

    /// Apply a transactional bulk edit to the food field.
    pub fn try_update_food(
        &mut self,
        update: impl FnOnce(&mut [f32]),
    ) -> Result<(), ScientificStateError> {
        self.food.try_update_cells(update)
    }

    /// Internal mutable access to the trusted food grid.
    #[cfg(test)]
    #[must_use]
    fn food_mut(&mut self) -> &mut FoodGrid {
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
    /// Replace the current terrain and food fields using a pre-generated map artifact.
    pub fn apply_map_artifact(&mut self, artifact: &MapArtifact) -> Result<(), WorldStateError> {
        artifact.validate()?;
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

        let candidate_terrain = terrain.clone();
        let candidate_food_profiles = FoodCellProfile::compute(&self.config, &candidate_terrain);
        let mut candidate_food = self.food.clone();

        if let Some(field) = artifact.fertility() {
            let max_food = self.config.food_max;
            for (cell, value) in candidate_food
                .cells_mut()
                .iter_mut()
                .zip(field.values().iter())
            {
                *cell = value.clamp(0.0, 1.0) * max_food;
            }
        }

        let candidate_hydrology = match (artifact.hydrology_tiles(), artifact.hydrology_field()) {
            (Some(tiles), Some(field)) => Some(HydrologyState::new(tiles.clone(), field.clone())?),
            _ => None,
        };

        self.terrain = candidate_terrain;
        self.food_profiles = candidate_food_profiles;
        self.food = candidate_food;
        self.hydrology = candidate_hydrology;
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
    pub fn bind_agent_brain(&mut self, id: AgentId, key: u64) -> Result<bool, BrainSpawnError> {
        if !self.agents.contains(id) {
            return Ok(false);
        }
        let rng_before = self.rng.clone();
        let binding = match BrainBinding::from_registry(&self.brain_registry, &mut self.rng, key) {
            Ok(Some(binding)) => binding,
            Ok(None) => {
                self.rng = rng_before;
                return Ok(false);
            }
            Err(error) => {
                self.rng = rng_before;
                return Err(error);
            }
        };
        let Some(runtime) = self.runtime.get_mut(id) else {
            self.rng = rng_before;
            return Ok(false);
        };
        runtime.brain = binding;
        Ok(true)
    }

    /// Immutable access to per-agent runtime metadata.
    #[must_use]
    pub fn runtime(&self) -> &AgentMap<AgentRuntime> {
        &self.runtime
    }

    /// Borrow runtime data for a specific agent.
    #[must_use]
    pub fn agent_runtime(&self, id: AgentId) -> Option<&AgentRuntime> {
        self.runtime.get(id)
    }

    /// Transactionally edit both scalar and runtime state for one agent.
    pub fn try_update_agent(
        &mut self,
        id: AgentId,
        update: impl FnOnce(&mut AgentData, &mut AgentRuntime),
    ) -> Result<bool, ScientificStateError> {
        let Some(mut data) = self.agents.snapshot(id) else {
            return Ok(false);
        };
        let Some(original_runtime) = self.runtime.get(id) else {
            return Ok(false);
        };
        let original_brain_key = original_runtime.brain.registry_key();
        let original_brain_kind = original_runtime.brain.kind().map(str::to_owned);
        let mut runtime = original_runtime.clone();
        update(&mut data, &mut runtime);
        let agent_path = format!("agents[{}]", id.data().as_ffi());
        data.validate_at(&agent_path)?;
        runtime.validate_at(&format!("{agent_path}.runtime"))?;
        let replaced = self.agents.replace_trusted(id, data);
        debug_assert!(replaced);
        if !runtime.brain.is_bound()
            && runtime.brain.registry_key() == original_brain_key
            && runtime.brain.kind() == original_brain_kind.as_deref()
            && let Some(original_runtime) = self.runtime.get_mut(id)
        {
            runtime.brain = std::mem::take(&mut original_runtime.brain);
        }
        self.runtime.insert(id, runtime);
        Ok(true)
    }

    /// Transactionally edit runtime metadata for one agent.
    pub fn try_update_agent_runtime(
        &mut self,
        id: AgentId,
        update: impl FnOnce(&mut AgentRuntime),
    ) -> Result<bool, ScientificStateError> {
        let Some(original_runtime) = self.runtime.get(id) else {
            return Ok(false);
        };
        let original_brain_key = original_runtime.brain.registry_key();
        let original_brain_kind = original_runtime.brain.kind().map(str::to_owned);
        let mut runtime = original_runtime.clone();
        update(&mut runtime);
        runtime.validate_at(&format!("agents[{}].runtime", id.data().as_ffi()))?;
        if !runtime.brain.is_bound()
            && runtime.brain.registry_key() == original_brain_key
            && runtime.brain.kind() == original_brain_kind.as_deref()
            && let Some(original_runtime) = self.runtime.get_mut(id)
        {
            runtime.brain = std::mem::take(&mut original_runtime.brain);
        }
        self.runtime.insert(id, runtime);
        Ok(true)
    }

    /// Internal mutable borrow for trusted tick logic and in-module oracle setup.
    #[cfg(test)]
    #[must_use]
    fn agent_runtime_mut(&mut self, id: AgentId) -> Option<&mut AgentRuntime> {
        self.runtime.get_mut(id)
    }

    /// Produce a combined snapshot of an agent's scalar columns and runtime data.
    #[must_use]
    pub fn snapshot_agent(&self, id: AgentId) -> Option<AgentState> {
        let data = self.agents.snapshot(id)?;
        let identity = self.identities.get(id).copied()?;
        let runtime = self.runtime.get(id)?.clone();
        Some(AgentState {
            id,
            identity,
            data,
            runtime,
        })
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
                agent_uid: snapshot.identity.uid,
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
            AgentDebugSort::AgeDesc => entries.sort_by_key(|e| std::cmp::Reverse(e.age)),
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

    fn assert_small_rng_stream_continuation(stream: &SmallRngStream) {
        let checkpoint = stream.checkpoint();
        let encoded = postcard::to_allocvec(&checkpoint).expect("encode random-stream state");
        let decoded: RandomStreamState =
            postcard::from_bytes(&encoded).expect("decode random-stream state");
        assert_eq!(decoded, checkpoint);

        let mut restored = SmallRngStream::from_state(&decoded).expect("restore random stream");
        let mut expected = stream.clone();
        assert_eq!(restored.next_u32(), expected.next_u32());
        assert_eq!(restored.next_u64(), expected.next_u64());
        let mut restored_bytes = [0_u8; 13];
        let mut expected_bytes = [0_u8; 13];
        restored.fill_bytes(&mut restored_bytes);
        expected.fill_bytes(&mut expected_bytes);
        assert_eq!(restored_bytes, expected_bytes);
        assert_eq!(restored.next_u64(), expected.next_u64());
    }

    #[test]
    fn small_rng_stream_matches_rand_095_and_restores_every_continuation() {
        for seed in [0, 1, 0x5eed_cafe_dead_beef, u64::MAX] {
            let mut expected = SmallRng::seed_from_u64(seed);
            let mut actual = SmallRngStream::seed_from_u64(seed);
            assert_eq!(actual.next_u32(), expected.next_u32());
            assert_small_rng_stream_continuation(&actual);
            assert_eq!(actual.next_u64(), expected.next_u64());
            assert_small_rng_stream_continuation(&actual);
            assert_eq!(actual.next_u32(), expected.next_u32());
            assert_small_rng_stream_continuation(&actual);
            assert_eq!(actual.next_u64(), expected.next_u64());
            assert_small_rng_stream_continuation(&actual);

            for length in 0..=20 {
                let mut expected = SmallRng::seed_from_u64(seed);
                let mut actual = SmallRngStream::seed_from_u64(seed);
                let mut expected_bytes = vec![0_u8; length];
                let mut actual_bytes = vec![0_u8; length];
                expected.fill_bytes(&mut expected_bytes);
                actual.fill_bytes(&mut actual_bytes);
                assert_eq!(actual_bytes, expected_bytes, "seed={seed}, length={length}");
                assert_eq!(actual.next_u32(), expected.next_u32());
                assert_eq!(actual.next_u64(), expected.next_u64());
                assert_small_rng_stream_continuation(&actual);
            }
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn small_rng_stream_codec_v1_freezes_seed_and_state_words_as_little_endian() {
        let state = SmallRngStream::seed_from_u64(0).checkpoint();
        assert_eq!(state.version, RANDOM_STREAM_STATE_VERSION);
        assert_eq!(state.algorithm, SmallRngStream::algorithm());
        assert_eq!(state.codec_version, SMALL_RNG_STATE_CODEC_VERSION);
        assert_eq!(
            state.state,
            [
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xaf, 0xcd, 0x1d, 0x7b, 0x39, 0xa8,
                0x20, 0xe2, 0xf4, 0x65, 0xb9, 0xa1, 0x6a, 0x9e, 0x78, 0x6e, 0x4f, 0x45, 0x09, 0x80,
                0x18, 0x5d, 0xc4, 0x06, 0xec, 0x81, 0x4c, 0x72, 0xa8, 0xb8, 0x8b, 0xf8,
            ]
        );
    }

    #[test]
    fn random_stream_restore_rejects_protocol_mismatch_and_zero_state_atomically() {
        let original = SmallRngStream::seed_from_u64(9).checkpoint();

        let mut wrong_version = original.clone();
        wrong_version.version += 1;
        assert!(matches!(
            SmallRngStream::from_state(&wrong_version),
            Err(RandomStreamRestoreError::UnsupportedVersion { .. })
        ));
        assert_eq!(original, SmallRngStream::seed_from_u64(9).checkpoint());

        let mut wrong_algorithm = original.clone();
        wrong_algorithm.algorithm = "not-the-current-algorithm".to_owned();
        assert!(matches!(
            SmallRngStream::from_state(&wrong_algorithm),
            Err(RandomStreamRestoreError::UnsupportedAlgorithm { .. })
        ));
        assert_eq!(original, SmallRngStream::seed_from_u64(9).checkpoint());

        let mut wrong_codec = original.clone();
        wrong_codec.codec_version += 1;
        assert!(matches!(
            SmallRngStream::from_state(&wrong_codec),
            Err(RandomStreamRestoreError::UnsupportedCodecVersion { .. })
        ));
        assert_eq!(original, SmallRngStream::seed_from_u64(9).checkpoint());

        let mut oversized = original.clone();
        oversized.state = vec![0; MAX_RANDOM_STREAM_STATE_BYTES + 1];
        assert!(matches!(
            SmallRngStream::from_state(&oversized),
            Err(RandomStreamRestoreError::StateTooLarge { .. })
        ));
        let oversized_json = serde_json::to_vec(&oversized).expect("encode oversized state");
        assert!(
            serde_json::from_slice::<RandomStreamState>(&oversized_json).is_err(),
            "the protocol ceiling must also apply while decoding an opaque envelope"
        );
        assert_eq!(original, SmallRngStream::seed_from_u64(9).checkpoint());

        let mut wrong_length = original.clone();
        wrong_length.state.pop();
        assert!(matches!(
            SmallRngStream::from_state(&wrong_length),
            Err(RandomStreamRestoreError::InvalidStateLength { .. })
        ));
        assert_eq!(original, SmallRngStream::seed_from_u64(9).checkpoint());

        let mut all_zero = original.clone();
        all_zero.state[8..].fill(0);
        assert_eq!(
            SmallRngStream::from_state(&all_zero).expect_err("zero state must fail"),
            RandomStreamRestoreError::AllZeroState
        );
        assert_eq!(original, SmallRngStream::seed_from_u64(9).checkpoint());
    }

    #[test]
    fn random_stream_protocol_is_object_safe_and_not_small_rng_specific() {
        #[derive(Clone)]
        struct MockStream(u64);

        impl RngCore for MockStream {
            fn next_u32(&mut self) -> u32 {
                self.next_u64() as u32
            }

            fn next_u64(&mut self) -> u64 {
                let value = self.0;
                self.0 += 1;
                value
            }

            fn fill_bytes(&mut self, destination: &mut [u8]) {
                for byte in destination {
                    *byte = self.next_u32() as u8;
                }
            }
        }

        impl RandomStream for MockStream {
            fn algorithm_id(&self) -> &'static str {
                "test.mock-counter"
            }

            fn checkpoint(&self) -> RandomStreamState {
                RandomStreamState {
                    version: RANDOM_STREAM_STATE_VERSION,
                    algorithm: self.algorithm_id().to_owned(),
                    codec_version: 1,
                    state: self.0.to_le_bytes().to_vec(),
                }
            }
        }

        fn consume_protocol(stream: &mut dyn RandomStream) -> (&'static str, u64, u64) {
            let algorithm = stream.algorithm_id();
            let before = decode_le_u64(&stream.checkpoint().state);
            let sample = stream.next_u64();
            (algorithm, before, sample)
        }

        let mut stream = MockStream(41);
        assert_eq!(consume_protocol(&mut stream), ("test.mock-counter", 41, 41));
        assert_eq!(decode_le_u64(&stream.checkpoint().state), 42);
    }

    fn invalid_config_message(error: WorldStateError) -> &'static str {
        match error {
            WorldStateError::InvalidConfig(message) => message,
            WorldStateError::InvalidState(error) => {
                panic!("expected configuration rejection, got state rejection: {error}")
            }
        }
    }

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
    fn agent_uid_survives_snapshot_round_trip_and_never_reuses_after_arena_churn() {
        let config = ScriptBotsConfig {
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(0x1d),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("world");
        let first_id = world.spawn_agent(sample_agent(1));
        let first = world.snapshot_agent(first_id).expect("first snapshot");
        let encoded = postcard::to_allocvec(&first).expect("encode agent snapshot");
        let decoded: AgentState = postcard::from_bytes(&encoded).expect("decode agent snapshot");
        assert_eq!(decoded.identity, first.identity);
        assert_eq!(first.identity.uid, AgentUid(1));
        assert_eq!(first.identity.spawn_ordinal, 0);
        assert_eq!(first.identity.birth_ordinal, None);

        world.remove_agent(first_id).expect("remove first agent");
        let second_id = world.spawn_agent(sample_agent(2));
        let second = world.snapshot_agent(second_id).expect("second snapshot");
        assert_ne!(
            first_id, second_id,
            "SlotMap generation must change on reuse"
        );
        assert_ne!(first.identity.uid, second.identity.uid);
        assert_eq!(second.identity.uid, AgentUid(2));
        assert_eq!(second.identity.spawn_ordinal, 1);
        assert_eq!(world.agent_uid(first_id), None);
        assert_eq!(world.agent_uid(second_id), Some(AgentUid(2)));
    }

    #[test]
    fn spawn_and_birth_ordinals_are_deterministic_and_independent_of_slotmap_handles() {
        let identity_sequence = || {
            let config = ScriptBotsConfig {
                population_minimum: 0,
                population_spawn_interval: 0,
                rng_seed: Some(0x51),
                ..ScriptBotsConfig::default()
            };
            let mut world = WorldState::new(config).expect("world");
            let removed = world.spawn_agent(sample_agent(0));
            let survivor = world.spawn_agent(sample_agent(1));
            world.remove_agent(removed).expect("remove first spawn");
            let replacement = world.spawn_agent(sample_agent(2));
            let first_birth = world.insert_agent(sample_agent(3), AgentRuntime::default(), true);
            let second_birth = world.insert_agent(sample_agent(4), AgentRuntime::default(), true);
            [survivor, replacement, first_birth, second_birth]
                .map(|id| (id.raw(), world.agent_identity(id).expect("stable identity")))
        };

        let first = identity_sequence();
        let second = identity_sequence();
        assert_eq!(
            first, second,
            "fixed construction order must reproduce exactly"
        );
        assert_eq!(first[0].1.uid, AgentUid(2));
        assert_eq!(first[1].1.uid, AgentUid(3));
        assert_eq!(first[2].1.uid, AgentUid(4));
        assert_eq!(first[3].1.uid, AgentUid(5));
        assert_eq!(first[0].1.spawn_ordinal, 1);
        assert_eq!(first[1].1.spawn_ordinal, 2);
        assert_eq!(first[2].1.birth_ordinal, Some(0));
        assert_eq!(first[3].1.birth_ordinal, Some(1));
        assert_ne!(
            first[0].0, first[1].0,
            "SlotMap handles remain generational"
        );
    }

    fn assert_non_finite_path(error: ScientificStateError, expected_path: &str) {
        assert_eq!(error.path(), expected_path);
        assert!(matches!(error, ScientificStateError::NonFinite { .. }));
    }

    fn assert_dimension_overflow(error: WorldStateError, expected_path: &str) {
        let WorldStateError::InvalidState(error) = error else {
            panic!("expected typed scientific-state rejection");
        };
        assert_eq!(error.path(), expected_path);
        assert!(matches!(
            error,
            ScientificStateError::DimensionOverflow { .. }
        ));
    }

    #[test]
    fn direct_agent_ingress_rejects_each_non_finite_class_without_state_or_rng_drift() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let config = ScriptBotsConfig {
                rng_seed: Some(0x5EA1_ED11),
                population_minimum: 0,
                population_spawn_interval: 0,
                ..ScriptBotsConfig::default()
            };
            let mut rejected = WorldState::new(config.clone()).expect("rejected world");
            let mut reference = WorldState::new(config).expect("reference world");
            let before = rejected
                .characterization_digest_v0()
                .expect("quiescent digest");
            let before_revision = rejected.config_revision();
            let before_audit = rejected.config_audit().to_vec();

            let error = rejected
                .try_spawn_agent(AgentData {
                    velocity: Velocity::new(0.25, value),
                    ..AgentData::default()
                })
                .expect_err("non-finite velocity must be rejected");
            assert_non_finite_path(error, "agent.velocity.vy");
            assert_eq!(
                rejected
                    .characterization_digest_v0()
                    .expect("unchanged digest"),
                before
            );
            assert_eq!(rejected.config_revision(), before_revision);
            assert_eq!(rejected.config_audit(), before_audit);

            let error = rejected
                .try_spawn_agent_with(AgentData::default(), |runtime| {
                    runtime.outputs[2] = value;
                })
                .expect_err("non-finite runtime must reject the whole spawn");
            assert_non_finite_path(error, "agent.runtime.outputs[2]");
            assert_eq!(
                rejected
                    .characterization_digest_v0()
                    .expect("unchanged runtime-rejection digest"),
                before
            );

            let rejected_id = rejected
                .try_spawn_agent(AgentData::default())
                .expect("finite follow-up spawn");
            let reference_id = reference
                .try_spawn_agent(AgentData::default())
                .expect("finite reference spawn");
            assert_eq!(rejected_id, reference_id, "allocator state must not drift");
            assert_eq!(
                rejected
                    .characterization_digest_v0()
                    .expect("follow-up digest"),
                reference
                    .characterization_digest_v0()
                    .expect("reference digest"),
                "rejection must not consume the fixed-seed random stream"
            );
        }
    }

    #[test]
    fn transactional_agent_and_food_updates_are_atomic_and_report_exact_indexes() {
        struct IngressTestBrain;
        impl BrainRunner for IngressTestBrain {
            fn kind(&self) -> &'static str {
                "ingress-test"
            }

            fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
                [0.0; OUTPUT_SIZE]
            }
        }

        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0xA70C_1C11),
            population_minimum: 0,
            population_spawn_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .expect("world");
        let agent = world
            .try_spawn_agent_with(AgentData::default(), |runtime| {
                runtime.energy = -0.0;
                runtime.sensors[INPUT_SIZE - 1] = f32::MIN_POSITIVE;
                runtime.brain = BrainBinding::with_runner(Box::new(IngressTestBrain));
            })
            .expect("representative finite boundaries are admitted");
        world
            .try_update_agent_runtime(agent, |runtime| runtime.energy = 0.75)
            .expect("finite runtime edit");
        assert!(
            world
                .agent_runtime(agent)
                .expect("runtime")
                .brain
                .is_bound(),
            "staging a finite runtime edit must preserve its non-cloneable live brain runner"
        );
        let baseline = world.characterization_digest_v0().expect("baseline digest");
        let baseline_food = world.food().cells().to_vec();
        let baseline_revision = world.config_revision();
        let baseline_audit = world.config_audit().to_vec();

        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = world
                .try_update_agent(agent, |data, runtime| {
                    data.heading = 0.75;
                    runtime.sensors[7] = value;
                })
                .expect_err("runtime non-finite value must reject the whole candidate");
            assert_non_finite_path(
                error,
                &format!("agents[{}].runtime.sensors[7]", agent.raw()),
            );
            assert_eq!(
                world
                    .characterization_digest_v0()
                    .expect("unchanged agent digest"),
                baseline,
                "the finite scalar edit must roll back with the invalid runtime edit"
            );

            let error = world
                .try_update_food(|cells| {
                    cells[0] = 0.125;
                    cells[3] = value;
                })
                .expect_err("bulk food update must reject non-finite cell");
            assert_non_finite_path(error, "food.cells[3]");
            assert_eq!(world.food().cells(), baseline_food);
            assert_eq!(world.config_revision(), baseline_revision);
            assert_eq!(world.config_audit(), baseline_audit);
        }
    }

    #[test]
    fn detached_food_and_dense_agent_boundaries_cover_empty_single_and_bulk_values() {
        let mut columns = AgentColumns::new();
        assert!(columns.is_empty());
        columns
            .try_push(AgentData::default())
            .expect("single finite row");
        columns
            .try_push(AgentData {
                position: Position::new(-0.0, f32::MIN_POSITIVE),
                ..AgentData::default()
            })
            .expect("bulk finite row");
        let before_len = columns.len();
        let error = columns
            .try_push(AgentData {
                color: [0.0, f32::INFINITY, 1.0],
                ..AgentData::default()
            })
            .expect_err("invalid append must reject before touching any column");
        assert_non_finite_path(error, "agents[2].color[1]");
        assert_eq!(columns.len(), before_len);

        let mut food = FoodGrid::new(2, 2, -0.0).expect("finite initial field");
        food.try_replace_cells(vec![0.0, 0.25, 0.5, 1.0])
            .expect("finite bulk replacement");
        let before = food.cells().to_vec();
        let error = food
            .try_replace_cells(vec![0.0])
            .expect_err("short replacement must reject atomically");
        assert_eq!(error.path(), "food.cells");
        assert_eq!(food.cells(), before);
        let error = FoodGrid::new(1, 1, f32::NAN).expect_err("invalid single cell");
        let WorldStateError::InvalidState(error) = error else {
            panic!("expected typed state error");
        };
        assert_non_finite_path(error, "food.initial");
    }

    #[test]
    fn public_dense_allocators_reject_unrepresentable_layouts_before_allocation_or_rng_use() {
        assert_dimension_overflow(
            FoodGrid::new(u32::MAX, u32::MAX, 0.0)
                .expect_err("oversized food layout must be rejected"),
            "food",
        );

        let mut rng = SmallRng::seed_from_u64(0xA110_CAFE);
        let mut untouched_rng = rng.clone();
        assert_dimension_overflow(
            TerrainLayer::generate(u32::MAX, u32::MAX, 1, &mut rng)
                .expect_err("oversized terrain layout must be rejected"),
            "terrain.tiles",
        );
        assert_eq!(
            rng.next_u64(),
            untouched_rng.next_u64(),
            "layout rejection must happen before terrain generation consumes RNG"
        );
    }

    #[test]
    fn blood_sensor_cone_uses_strict_legacy_half_fov_boundary() {
        const ANGLE_EPSILON: f32 = 1.0e-4;
        const DISTANCE_FACTOR: f32 = 0.75;
        const TARGET_HEALTH: f32 = 1.0;

        let just_inside = blood_sensor_contribution(
            BLOOD_HALF_FOV - ANGLE_EPSILON,
            DISTANCE_FACTOR,
            TARGET_HEALTH,
        );
        let expected_inside =
            (ANGLE_EPSILON / BLOOD_HALF_FOV) * DISTANCE_FACTOR * (1.0 - TARGET_HEALTH / 2.0);

        assert!(just_inside > 0.0);
        assert!((just_inside - expected_inside).abs() <= 1.0e-7);
        assert_eq!(
            blood_sensor_contribution(BLOOD_HALF_FOV, DISTANCE_FACTOR, TARGET_HEALTH),
            0.0,
            "the legacy `diff4 < PI38` boundary excludes an on-edge target"
        );
        assert_eq!(
            blood_sensor_contribution(
                BLOOD_HALF_FOV + ANGLE_EPSILON,
                DISTANCE_FACTOR,
                TARGET_HEALTH,
            ),
            0.0,
            "a target just outside the legacy cone must not contribute"
        );
    }

    fn set_render_tonemap_exposure_bias(config: &mut ScriptBotsConfig, value: f32) {
        config.render.tonemap_exposure_bias = Some(value);
    }

    fn set_render_auto_exposure_speed_brighten(config: &mut ScriptBotsConfig, value: f32) {
        let settings = config
            .render
            .auto_exposure
            .get_or_insert(RenderAutoExposureSettings {
                enabled: true,
                speed_brighten: None,
                speed_darken: None,
            });
        settings.speed_brighten = Some(value);
    }

    fn set_render_auto_exposure_speed_darken(config: &mut ScriptBotsConfig, value: f32) {
        let settings = config
            .render
            .auto_exposure
            .get_or_insert(RenderAutoExposureSettings {
                enabled: true,
                speed_brighten: None,
                speed_darken: None,
            });
        settings.speed_darken = Some(value);
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
        grid.fill(2.0).expect("finite fill");
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
    fn every_public_config_float_rejects_non_finite_values_with_its_field_path() {
        fn collect_float_paths(
            prefix: &str,
            value: &serde_json::Value,
            paths: &mut std::collections::BTreeSet<String>,
        ) {
            match value {
                serde_json::Value::Object(map) => {
                    for (key, child) in map {
                        let path = if prefix.is_empty() {
                            key.clone()
                        } else {
                            format!("{prefix}.{key}")
                        };
                        collect_float_paths(&path, child, paths);
                    }
                }
                serde_json::Value::Number(number) if number.is_f64() => {
                    paths.insert(prefix.to_owned());
                }
                _ => {}
            }
        }

        type Setter = fn(&mut ScriptBotsConfig, f32);
        let fields: [(&str, Setter); 73] = [
            ("initial_food", |config, value| config.initial_food = value),
            ("food_respawn_amount", |config, value| {
                config.food_respawn_amount = value;
            }),
            ("food_max", |config, value| config.food_max = value),
            ("food_growth_rate", |config, value| {
                config.food_growth_rate = value;
            }),
            ("food_decay_rate", |config, value| {
                config.food_decay_rate = value;
            }),
            ("food_diffusion_rate", |config, value| {
                config.food_diffusion_rate = value;
            }),
            ("sense_radius", |config, value| config.sense_radius = value),
            ("sense_max_neighbors", |config, value| {
                config.sense_max_neighbors = value;
            }),
            ("bot_speed", |config, value| config.bot_speed = value),
            ("bot_radius", |config, value| config.bot_radius = value),
            ("boost_multiplier", |config, value| {
                config.boost_multiplier = value;
            }),
            ("spike_growth_rate", |config, value| {
                config.spike_growth_rate = value;
            }),
            ("metabolism_drain", |config, value| {
                config.metabolism_drain = value;
            }),
            ("movement_drain", |config, value| {
                config.movement_drain = value;
            }),
            ("metabolism_ramp_floor", |config, value| {
                config.metabolism_ramp_floor = value;
            }),
            ("metabolism_ramp_rate", |config, value| {
                config.metabolism_ramp_rate = value;
            }),
            ("metabolism_boost_penalty", |config, value| {
                config.metabolism_boost_penalty = value;
            }),
            ("temperature_discomfort_rate", |config, value| {
                config.temperature_discomfort_rate = value;
            }),
            ("temperature_comfort_band", |config, value| {
                config.temperature_comfort_band = value;
            }),
            ("temperature_gradient_exponent", |config, value| {
                config.temperature_gradient_exponent = value;
            }),
            ("temperature_discomfort_exponent", |config, value| {
                config.temperature_discomfort_exponent = value;
            }),
            ("food_intake_rate", |config, value| {
                config.food_intake_rate = value;
            }),
            ("food_waste_rate", |config, value| {
                config.food_waste_rate = value;
            }),
            ("food_fertility_base", |config, value| {
                config.food_fertility_base = value;
            }),
            ("food_moisture_weight", |config, value| {
                config.food_moisture_weight = value;
            }),
            ("food_elevation_weight", |config, value| {
                config.food_elevation_weight = value;
            }),
            ("food_slope_weight", |config, value| {
                config.food_slope_weight = value;
            }),
            ("food_capacity_base", |config, value| {
                config.food_capacity_base = value;
            }),
            ("food_capacity_fertility", |config, value| {
                config.food_capacity_fertility = value;
            }),
            ("food_growth_fertility", |config, value| {
                config.food_growth_fertility = value;
            }),
            ("food_decay_infertility", |config, value| {
                config.food_decay_infertility = value;
            }),
            ("food_sharing_radius", |config, value| {
                config.food_sharing_radius = value;
            }),
            ("food_sharing_rate", |config, value| {
                config.food_sharing_rate = value;
            }),
            ("food_transfer_rate", |config, value| {
                config.food_transfer_rate = value;
            }),
            ("food_sharing_distance", |config, value| {
                config.food_sharing_distance = value;
            }),
            ("reproduction_energy_threshold", |config, value| {
                config.reproduction_energy_threshold = value;
            }),
            ("reproduction_energy_cost", |config, value| {
                config.reproduction_energy_cost = value;
            }),
            ("reproduction_attempt_chance", |config, value| {
                config.reproduction_attempt_chance = value;
            }),
            ("reproduction_rate_herbivore", |config, value| {
                config.reproduction_rate_herbivore = value;
            }),
            ("reproduction_rate_carnivore", |config, value| {
                config.reproduction_rate_carnivore = value;
            }),
            ("reproduction_food_bonus", |config, value| {
                config.reproduction_food_bonus = value;
            }),
            ("reproduction_fertility_bonus", |config, value| {
                config.reproduction_fertility_bonus = value;
            }),
            ("reproduction_child_energy", |config, value| {
                config.reproduction_child_energy = value;
            }),
            ("reproduction_spawn_jitter", |config, value| {
                config.reproduction_spawn_jitter = value;
            }),
            ("reproduction_color_jitter", |config, value| {
                config.reproduction_color_jitter = value;
            }),
            ("reproduction_mutation_scale", |config, value| {
                config.reproduction_mutation_scale = value;
            }),
            ("reproduction_partner_chance", |config, value| {
                config.reproduction_partner_chance = value;
            }),
            ("reproduction_spawn_back_distance", |config, value| {
                config.reproduction_spawn_back_distance = value;
            }),
            ("reproduction_meta_mutation_chance", |config, value| {
                config.reproduction_meta_mutation_chance = value;
            }),
            ("reproduction_meta_mutation_scale", |config, value| {
                config.reproduction_meta_mutation_scale = value;
            }),
            ("aging_health_decay_rate", |config, value| {
                config.aging_health_decay_rate = value;
            }),
            ("aging_health_decay_max", |config, value| {
                config.aging_health_decay_max = value;
            }),
            ("aging_energy_penalty_rate", |config, value| {
                config.aging_energy_penalty_rate = value;
            }),
            ("carcass_distribution_radius", |config, value| {
                config.carcass_distribution_radius = value;
            }),
            ("carcass_health_reward", |config, value| {
                config.carcass_health_reward = value;
            }),
            ("carcass_reproduction_reward", |config, value| {
                config.carcass_reproduction_reward = value;
            }),
            ("carcass_neighbor_exponent", |config, value| {
                config.carcass_neighbor_exponent = value;
            }),
            ("carcass_energy_share_rate", |config, value| {
                config.carcass_energy_share_rate = value;
            }),
            ("carcass_indicator_scale", |config, value| {
                config.carcass_indicator_scale = value;
            }),
            ("topography_speed_gain", |config, value| {
                config.topography_speed_gain = value;
            }),
            ("topography_energy_penalty", |config, value| {
                config.topography_energy_penalty = value;
            }),
            ("population_crossover_chance", |config, value| {
                config.population_crossover_chance = value;
            }),
            ("spike_radius", |config, value| config.spike_radius = value),
            ("spike_damage", |config, value| config.spike_damage = value),
            ("spike_energy_cost", |config, value| {
                config.spike_energy_cost = value;
            }),
            ("spike_min_length", |config, value| {
                config.spike_min_length = value;
            }),
            ("spike_alignment_cosine", |config, value| {
                config.spike_alignment_cosine = value;
            }),
            ("spike_speed_damage_bonus", |config, value| {
                config.spike_speed_damage_bonus = value;
            }),
            ("spike_length_damage_bonus", |config, value| {
                config.spike_length_damage_bonus = value;
            }),
            ("carnivore_threshold", |config, value| {
                config.carnivore_threshold = value;
            }),
            (
                "render.tonemap_exposure_bias",
                set_render_tonemap_exposure_bias,
            ),
            (
                "render.auto_exposure.speed_brighten",
                set_render_auto_exposure_speed_brighten,
            ),
            (
                "render.auto_exposure.speed_darken",
                set_render_auto_exposure_speed_darken,
            ),
        ];

        let expected_paths = fields
            .iter()
            .map(|(field, _)| (*field).to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            expected_paths.len(),
            fields.len(),
            "config float inventory contains a duplicate field path"
        );
        let mut schema_probe = ScriptBotsConfig::default();
        for (_, setter) in fields {
            setter(&mut schema_probe, 0.5);
        }
        let serialized = serde_json::to_value(schema_probe).expect("serialize schema probe");
        let mut serialized_paths = std::collections::BTreeSet::new();
        collect_float_paths("", &serialized, &mut serialized_paths);
        assert_eq!(
            serialized_paths, expected_paths,
            "table must mechanically cover every serialized public config float exactly once"
        );

        for (field, setter) in fields {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut config = ScriptBotsConfig::default();
                setter(&mut config, value);
                let error = config
                    .validate()
                    .expect_err("every non-finite public config float must be rejected");
                let WorldStateError::InvalidConfig(message) = error else {
                    panic!("non-finite config unexpectedly produced state error: {error}");
                };
                assert!(
                    message.starts_with(field),
                    "{field}={value:?} produced unrelated validation error: {message}"
                );
            }
        }
    }

    #[test]
    fn every_bounded_public_config_float_rejects_a_finite_out_of_range_value() {
        type Setter = fn(&mut ScriptBotsConfig, f32);
        let fields: [(&str, f32, Setter); 72] = [
            ("initial_food", -1.0, |config, value| {
                config.initial_food = value
            }),
            ("food_respawn_amount", -1.0, |config, value| {
                config.food_respawn_amount = value;
            }),
            ("food_max", 0.0, |config, value| config.food_max = value),
            ("food_growth_rate", -1.0, |config, value| {
                config.food_growth_rate = value;
            }),
            ("food_decay_rate", -1.0, |config, value| {
                config.food_decay_rate = value;
            }),
            ("food_diffusion_rate", 0.26, |config, value| {
                config.food_diffusion_rate = value;
            }),
            ("sense_radius", 0.0, |config, value| {
                config.sense_radius = value
            }),
            ("sense_max_neighbors", 0.0, |config, value| {
                config.sense_max_neighbors = value;
            }),
            ("bot_speed", -1.0, |config, value| config.bot_speed = value),
            ("bot_radius", 0.0, |config, value| config.bot_radius = value),
            ("boost_multiplier", 0.99, |config, value| {
                config.boost_multiplier = value;
            }),
            ("spike_growth_rate", -1.0, |config, value| {
                config.spike_growth_rate = value;
            }),
            ("metabolism_drain", -1.0, |config, value| {
                config.metabolism_drain = value;
            }),
            ("movement_drain", -1.0, |config, value| {
                config.movement_drain = value;
            }),
            ("metabolism_ramp_floor", -1.0, |config, value| {
                config.metabolism_ramp_floor = value;
            }),
            ("metabolism_ramp_rate", -1.0, |config, value| {
                config.metabolism_ramp_rate = value;
            }),
            ("metabolism_boost_penalty", -1.0, |config, value| {
                config.metabolism_boost_penalty = value;
            }),
            ("temperature_discomfort_rate", -1.0, |config, value| {
                config.temperature_discomfort_rate = value;
            }),
            ("temperature_comfort_band", -0.1, |config, value| {
                config.temperature_comfort_band = value;
            }),
            ("temperature_gradient_exponent", 0.0, |config, value| {
                config.temperature_gradient_exponent = value;
            }),
            ("temperature_discomfort_exponent", 0.0, |config, value| {
                config.temperature_discomfort_exponent = value;
            }),
            ("food_intake_rate", -1.0, |config, value| {
                config.food_intake_rate = value;
            }),
            ("food_waste_rate", -1.0, |config, value| {
                config.food_waste_rate = value;
            }),
            ("food_fertility_base", -0.1, |config, value| {
                config.food_fertility_base = value;
            }),
            ("food_moisture_weight", -1.0, |config, value| {
                config.food_moisture_weight = value;
            }),
            ("food_elevation_weight", -1.0, |config, value| {
                config.food_elevation_weight = value;
            }),
            ("food_slope_weight", -1.0, |config, value| {
                config.food_slope_weight = value;
            }),
            ("food_capacity_base", -0.1, |config, value| {
                config.food_capacity_base = value;
            }),
            ("food_capacity_fertility", -0.1, |config, value| {
                config.food_capacity_fertility = value;
            }),
            ("food_growth_fertility", -1.0, |config, value| {
                config.food_growth_fertility = value;
            }),
            ("food_decay_infertility", -1.0, |config, value| {
                config.food_decay_infertility = value;
            }),
            ("food_sharing_radius", 0.0, |config, value| {
                config.food_sharing_radius = value;
            }),
            ("food_sharing_rate", -1.0, |config, value| {
                config.food_sharing_rate = value;
            }),
            ("food_transfer_rate", -1.0, |config, value| {
                config.food_transfer_rate = value;
            }),
            ("food_sharing_distance", 0.0, |config, value| {
                config.food_sharing_distance = value;
            }),
            ("reproduction_energy_threshold", -1.0, |config, value| {
                config.reproduction_energy_threshold = value;
            }),
            ("reproduction_energy_cost", -1.0, |config, value| {
                config.reproduction_energy_cost = value;
            }),
            ("reproduction_attempt_chance", -0.1, |config, value| {
                config.reproduction_attempt_chance = value;
            }),
            ("reproduction_rate_herbivore", 0.0, |config, value| {
                config.reproduction_rate_herbivore = value;
            }),
            ("reproduction_rate_carnivore", 0.0, |config, value| {
                config.reproduction_rate_carnivore = value;
            }),
            ("reproduction_food_bonus", -1.0, |config, value| {
                config.reproduction_food_bonus = value;
            }),
            ("reproduction_fertility_bonus", -1.0, |config, value| {
                config.reproduction_fertility_bonus = value;
            }),
            ("reproduction_child_energy", -1.0, |config, value| {
                config.reproduction_child_energy = value;
            }),
            ("reproduction_spawn_jitter", -1.0, |config, value| {
                config.reproduction_spawn_jitter = value;
            }),
            ("reproduction_color_jitter", -1.0, |config, value| {
                config.reproduction_color_jitter = value;
            }),
            ("reproduction_mutation_scale", -1.0, |config, value| {
                config.reproduction_mutation_scale = value;
            }),
            ("reproduction_partner_chance", -0.1, |config, value| {
                config.reproduction_partner_chance = value;
            }),
            ("reproduction_spawn_back_distance", -1.0, |config, value| {
                config.reproduction_spawn_back_distance = value;
            }),
            (
                "reproduction_meta_mutation_chance",
                -0.1,
                |config, value| {
                    config.reproduction_meta_mutation_chance = value;
                },
            ),
            ("reproduction_meta_mutation_scale", -1.0, |config, value| {
                config.reproduction_meta_mutation_scale = value;
            }),
            ("aging_health_decay_rate", -1.0, |config, value| {
                config.aging_health_decay_rate = value;
            }),
            ("aging_health_decay_max", -1.0, |config, value| {
                config.aging_health_decay_max = value;
            }),
            ("aging_energy_penalty_rate", -1.0, |config, value| {
                config.aging_energy_penalty_rate = value;
            }),
            ("carcass_distribution_radius", -1.0, |config, value| {
                config.carcass_distribution_radius = value;
            }),
            ("carcass_health_reward", -1.0, |config, value| {
                config.carcass_health_reward = value;
            }),
            ("carcass_reproduction_reward", -1.0, |config, value| {
                config.carcass_reproduction_reward = value;
            }),
            ("carcass_neighbor_exponent", 0.0, |config, value| {
                config.carcass_neighbor_exponent = value;
            }),
            ("carcass_energy_share_rate", -1.0, |config, value| {
                config.carcass_energy_share_rate = value;
            }),
            ("carcass_indicator_scale", -1.0, |config, value| {
                config.carcass_indicator_scale = value;
            }),
            ("topography_speed_gain", -1.0, |config, value| {
                config.topography_speed_gain = value;
            }),
            ("topography_energy_penalty", -1.0, |config, value| {
                config.topography_energy_penalty = value;
            }),
            ("population_crossover_chance", -0.1, |config, value| {
                config.population_crossover_chance = value;
            }),
            ("spike_radius", 0.0, |config, value| {
                config.spike_radius = value
            }),
            ("spike_damage", -1.0, |config, value| {
                config.spike_damage = value
            }),
            ("spike_energy_cost", -1.0, |config, value| {
                config.spike_energy_cost = value;
            }),
            ("spike_min_length", -1.0, |config, value| {
                config.spike_min_length = value;
            }),
            ("spike_alignment_cosine", 0.0, |config, value| {
                config.spike_alignment_cosine = value;
            }),
            ("spike_speed_damage_bonus", -1.0, |config, value| {
                config.spike_speed_damage_bonus = value;
            }),
            ("spike_length_damage_bonus", -1.0, |config, value| {
                config.spike_length_damage_bonus = value;
            }),
            ("carnivore_threshold", 0.0, |config, value| {
                config.carnivore_threshold = value;
            }),
            (
                "render.auto_exposure.speed_brighten",
                -1.0,
                set_render_auto_exposure_speed_brighten,
            ),
            (
                "render.auto_exposure.speed_darken",
                -1.0,
                set_render_auto_exposure_speed_darken,
            ),
        ];

        for (field, value, setter) in fields {
            let mut config = ScriptBotsConfig::default();
            setter(&mut config, value);
            let result = config.validate();
            assert!(
                result.is_err(),
                "{field} accepted invalid finite value {value:?}"
            );
            let Err(WorldStateError::InvalidConfig(message)) = result else {
                continue;
            };
            assert!(
                message.starts_with(field),
                "{field}={value:?} produced unrelated validation error: {message}"
            );
        }

        let mut config = ScriptBotsConfig::default();
        config.render.tonemap_exposure_bias = Some(f32::MAX);
        config
            .validate()
            .expect("tonemap exposure bias accepts every finite f32");
    }

    #[test]
    fn finite_boundaries_and_coupled_constraints_preserve_existing_behavior() {
        let mut config = ScriptBotsConfig {
            initial_food: 0.0,
            food_respawn_amount: 0.0,
            food_growth_rate: 0.0,
            food_decay_rate: 0.0,
            food_diffusion_rate: 0.25,
            bot_speed: 0.0,
            boost_multiplier: 1.0,
            temperature_comfort_band: 1.0,
            food_capacity_base: 0.4,
            food_capacity_fertility: 0.6,
            reproduction_energy_threshold: 0.65,
            reproduction_energy_cost: 0.65,
            reproduction_attempt_chance: 1.0,
            reproduction_partner_chance: 0.0,
            reproduction_meta_mutation_chance: 1.0,
            aging_health_decay_rate: 0.01,
            aging_health_decay_max: 0.01,
            population_crossover_chance: 1.0,
            spike_alignment_cosine: 1.0,
            carnivore_threshold: f32::EPSILON,
            ..ScriptBotsConfig::default()
        };
        set_render_auto_exposure_speed_brighten(&mut config, 0.0);
        set_render_auto_exposure_speed_darken(&mut config, 0.0);
        config
            .validate()
            .expect("documented inclusive boundaries must remain valid");

        let capacity = ScriptBotsConfig {
            food_capacity_base: 0.6,
            food_capacity_fertility: 0.5,
            ..ScriptBotsConfig::default()
        };
        let message = invalid_config_message(
            capacity
                .validate()
                .expect_err("capacity fractions above one must be rejected"),
        );
        assert_eq!(
            message,
            "food_capacity_base + food_capacity_fertility must be <= 1.0"
        );

        let reproduction = ScriptBotsConfig {
            reproduction_energy_cost: 0.66,
            ..ScriptBotsConfig::default()
        };
        let message = invalid_config_message(
            reproduction
                .validate()
                .expect_err("reproduction cost above threshold must be rejected"),
        );
        assert_eq!(
            message,
            "reproduction_energy_cost cannot exceed reproduction_energy_threshold"
        );

        let aging = ScriptBotsConfig {
            aging_health_decay_rate: 0.02,
            aging_health_decay_max: 0.01,
            ..ScriptBotsConfig::default()
        };
        let message = invalid_config_message(
            aging
                .validate()
                .expect_err("enabled aging decay must fit within its cap"),
        );
        assert_eq!(
            message,
            "aging_health_decay_max must be >= aging_health_decay_rate when decay is enabled"
        );
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
        let message = invalid_config_message(WorldState::new(config).unwrap_err());
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
        let expected_food_dimensions = config.food_dimensions().expect("food dimensions");
        let mut world = WorldState::new(config).expect("world");
        assert_eq!(world.agent_count(), 0);
        assert_eq!(world.food().width(), expected_food_dimensions.0);
        assert_eq!(world.food().height(), expected_food_dimensions.1);
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

        let events = world.step().expect("first simulation step");
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

        let events_second = world.step().expect("second simulation step");
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
            world.step().expect("aging cadence step");
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
            let events = world.step().expect("chart cadence step");
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
            .register("stub", |_rng| Ok(Box::new(StubBrain)));
        assert!(world.bind_agent_brain(id, key).expect("stub brain factory"));

        let events = world.step().expect("registered brain step");
        assert_eq!(events.tick, Tick(1));
        let runtime = world.agent_runtime(id).expect("runtime");
        assert!((runtime.outputs[0] - 1.0).abs() < f32::EPSILON);
        let position = world.agents().columns().positions()[0];
        assert!(position.x != 0.0 || position.y != 0.0);
        assert!(runtime.energy < 1.0);
    }

    /// Build a synthetic history so the narrative layer can be exercised
    /// without running thousands of ticks.
    fn narrative_history(values: &[usize]) -> Vec<TickSummary> {
        values
            .iter()
            .enumerate()
            .map(|(i, count)| TickSummary {
                tick: Tick(i as u64 + 1),
                agent_count: *count,
                births: 0,
                deaths: 0,
                total_energy: *count as f32,
                average_energy: 1.0,
                average_health: 1.0,
                max_age: 0,
                spike_hits: 0,
            })
            .collect()
    }

    #[test]
    fn narrative_detects_a_crash_and_names_it_in_deterministic_prose() {
        let mut values = vec![1000usize; 120];
        values.extend(std::iter::repeat_n(300usize, 120));
        let history = narrative_history(&values);

        let mut narrative = narrative::RunNarrative::default();
        narrative.observe(history.iter(), 256);

        let crash = narrative
            .events()
            .iter()
            .find(|e| e.kind == narrative::EventKind::PopulationCrash)
            .expect("a 70% population drop is a crash");
        assert!(crash.tick.0 >= 120, "must not fire before the crash");
        assert_eq!(crash.metric, "population");
        assert_eq!(crash.human_text, "population fell 70% (1000 -> 300)");
        assert!(crash.severity > 0.0 && crash.severity <= 1.0);
    }

    #[test]
    fn narrative_deduplicates_a_sustained_change_across_passes() {
        // As the window slides, the same crash re-detects on every pass. If the
        // stream emitted it each time, the timeline would be a stutter of
        // duplicates and users would stop reading it.
        let mut values = vec![800usize; 100];
        values.extend(std::iter::repeat_n(200usize, 100));
        let history = narrative_history(&values);

        let mut narrative = narrative::RunNarrative::default();
        for _ in 0..10 {
            narrative.observe(history.iter(), 256);
        }
        let crashes = narrative
            .events()
            .iter()
            .filter(|e| e.kind == narrative::EventKind::PopulationCrash)
            .count();
        assert_eq!(crashes, 1, "ten passes over one crash is still one crash");
    }

    #[test]
    fn named_channels_decode_exactly_like_the_legacy_slot_order() {
        // The channel refactor (bd-2z0.2.4) must be a pure rename: same slots,
        // same clamping, same boost threshold. This pins the mapping against an
        // independently written positional decode, so a renumbering of the enum
        // cannot quietly change what a brain's output MEANS.
        let outputs: [f32; OUTPUT_SIZE] = [0.9, 0.2, 0.11, 0.22, 0.33, 0.7, 0.8, 0.44, 0.55];

        assert!((outputs.channel(OutputChannel::WheelLeft) - outputs[0]).abs() < f32::EPSILON);
        assert!((outputs.channel(OutputChannel::WheelRight) - outputs[1]).abs() < f32::EPSILON);
        assert!((outputs.channel(OutputChannel::ColorRed) - outputs[2]).abs() < f32::EPSILON);
        assert!((outputs.channel(OutputChannel::ColorGreen) - outputs[3]).abs() < f32::EPSILON);
        assert!((outputs.channel(OutputChannel::ColorBlue) - outputs[4]).abs() < f32::EPSILON);
        assert!((outputs.channel(OutputChannel::SpikeTarget) - outputs[5]).abs() < f32::EPSILON);
        assert!((outputs.channel(OutputChannel::Boost) - outputs[6]).abs() < f32::EPSILON);
        assert!((outputs.channel(OutputChannel::SoundLevel) - outputs[7]).abs() < f32::EPSILON);
        assert!((outputs.channel(OutputChannel::GiveIntent) - outputs[8]).abs() < f32::EPSILON);
        assert_eq!(outputs.boost_engaged(), outputs[6] > 0.5);
    }

    #[test]
    fn narrative_ignores_statistically_real_but_trivial_changes() {
        // The bug this pins: against a nearly flat baseline, losing ONE agent
        // out of 23 is a statistically impeccable change — and narrating it is
        // static, not story. A real 3k-tick run produced 853 events per 10k
        // ticks before the materiality floor existed ("population fell 3%
        // (23 -> 22)", "mean energy collapsed (0.99 -> 0.98)").
        let mut values = vec![23usize; 120];
        values.extend(std::iter::repeat_n(22usize, 120));
        let history = narrative_history(&values);

        let mut narrative = narrative::RunNarrative::default();
        narrative.observe(history.iter(), 256);
        assert!(
            narrative.events().is_empty(),
            "losing one agent out of 23 is not news: {:?}",
            narrative.events()
        );

        // ...but a real collapse of the same small population still is.
        let mut values = vec![23usize; 120];
        values.extend(std::iter::repeat_n(3usize, 120));
        let history = narrative_history(&values);
        let mut narrative = narrative::RunNarrative::default();
        narrative.observe(history.iter(), 256);
        assert!(
            narrative
                .events()
                .iter()
                .any(|e| e.kind == narrative::EventKind::PopulationCrash),
            "losing 87% of the population IS news"
        );
    }

    #[test]
    fn narrative_is_quiet_on_a_flat_run() {
        let history = narrative_history(&[500usize; 200]);
        let mut narrative = narrative::RunNarrative::default();
        narrative.observe(history.iter(), 256);
        assert!(
            narrative.events().is_empty(),
            "a flat run has no story: {:?}",
            narrative.events()
        );
    }

    #[test]
    fn narrative_ring_is_bounded() {
        let mut narrative = narrative::RunNarrative::default();
        // Feed many distinct, MATERIAL crashes far enough apart to clear the
        // cooldown; only the newest `capacity` may survive.
        for round in 1..=50u64 {
            let mut values = vec![1000usize; 40];
            values.extend(std::iter::repeat_n(200usize, 40));
            let history: Vec<TickSummary> = narrative_history(&values)
                .into_iter()
                .map(|mut s| {
                    s.tick = Tick(s.tick.0 + round * 1000);
                    s
                })
                .collect();
            narrative.observe(history.iter(), 4);
        }
        assert!(
            narrative.events().len() <= 4,
            "ring must stay bounded, got {}",
            narrative.events().len()
        );
    }

    #[test]
    fn narrative_layer_does_not_perturb_the_simulation() {
        // The storyteller must not be able to change the story: enabling
        // narration must leave the science digest bit-identical.
        let base = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            rng_seed: Some(0x5EED),
            ..ScriptBotsConfig::default()
        };

        let digest_for = |interval: u32| {
            let config = ScriptBotsConfig {
                narrative_interval: interval,
                ..base.clone()
            };
            let mut world = WorldState::new(config).expect("world");
            for seed in 0..8 {
                world.spawn_agent(sample_agent(seed));
            }
            for _ in 0..120 {
                world.step().expect("step");
            }
            world
                .characterization_digest_v0()
                .expect("quiescent digest")
                .overall
        };

        assert_eq!(
            digest_for(0),
            digest_for(30),
            "narration must not perturb the simulation"
        );
    }

    #[test]
    fn narrative_events_are_deterministic_for_a_seed() {
        let run = || {
            let mut world = WorldState::new(ScriptBotsConfig {
                world_width: 200,
                world_height: 200,
                food_cell_size: 20,
                rng_seed: Some(4242),
                ..ScriptBotsConfig::default()
            })
            .expect("world");
            for seed in 0..12 {
                world.spawn_agent(sample_agent(seed));
            }
            for _ in 0..300 {
                world.step().expect("step");
            }
            world
                .narrative_events()
                .iter()
                .map(|e| (e.tick.0, e.kind, e.human_text.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "same seed must yield the same story");
    }

    /// The attribution must reproduce core's own sensing, or the inspector is
    /// confidently lying to the user — the worst possible outcome for a panel
    /// whose entire job is to explain.
    ///
    /// The proof: explain what the agent perceives now, then step the world once
    /// (stage_sense runs before anything moves, so it senses exactly the world we
    /// just explained) and require core's sensors to match what we predicted.
    #[test]
    fn a_meteor_selects_agents_across_the_toroidal_seam() {
        // The world is a TORUS. A region that measured distance naively would
        // select the wrong agents near an edge — an agent at x=5 and one at
        // x=195 in a 200-wide world are 10 apart, not 190 — and nobody would
        // notice until they dropped a meteor near a seam and watched the wrong
        // things die.
        // Zero the metabolic drains: the meteor must be the ONLY thing that can
        // change an agent's health, or the test measures starvation and calls it
        // a crater.
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            rng_seed: Some(5),
            ..ScriptBotsConfig::default()
        })
        .expect("world");

        // Two agents straddling the x=0 seam, and one far away in the middle.
        let near_left = world.spawn_agent(AgentData {
            position: Position::new(5.0, 100.0),
            health: 2.0,
            ..AgentData::default()
        });
        let near_right = world.spawn_agent(AgentData {
            position: Position::new(195.0, 100.0),
            health: 2.0,
            ..AgentData::default()
        });
        let far = world.spawn_agent(AgentData {
            position: Position::new(100.0, 100.0),
            health: 2.0,
            ..AgentData::default()
        });

        world
            .enqueue_intervention(Intervention::Meteor {
                region: Region::Disc {
                    x: 0.0,
                    y: 100.0,
                    radius: 12.0,
                },
                lethality: 0.5,
                scorch: 1.0,
            })
            .expect("valid meteor");
        world.step().expect("step");

        let health_of = |id| {
            let idx = world.agents().index_of(id).expect("alive");
            world.agents().columns().health()[idx]
        };
        assert!(
            (health_of(near_left) - 1.5).abs() < 1e-5,
            "the agent just inside the seam must take exactly the meteor's damage, got {}",
            health_of(near_left)
        );
        assert!(
            (health_of(near_right) - 1.5).abs() < 1e-5,
            "the agent on the OTHER side of the seam is 5 units away, not 195, so it \
             must take the same damage; got {}",
            health_of(near_right)
        );
        assert!(
            (health_of(far) - 2.0).abs() < 1e-5,
            "an agent 100 units away must be untouched, got {}",
            health_of(far)
        );
    }

    #[test]
    fn a_drought_actually_suppresses_regrowth_and_then_lapses() {
        // A drought that did not suppress regrowth would be theatre: the event
        // log would say "drought" and the ecosystem would carry on regardless.
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            initial_food: 0.1,
            food_growth_rate: 0.05,
            food_decay_rate: 0.0,
            food_diffusion_rate: 0.0,
            food_respawn_interval: 0,
            rng_seed: Some(11),
            ..ScriptBotsConfig::default()
        };

        let total_food = |world: &WorldState| -> f32 { world.food().cells().iter().sum() };

        // Control: no drought.
        let mut control = WorldState::new(config.clone()).expect("world");
        for _ in 0..10 {
            control.step().expect("step");
        }

        // Same world, world-wide drought for 5 ticks.
        let mut droughted = WorldState::new(config).expect("world");
        droughted
            .enqueue_intervention(Intervention::Drought {
                region: Region::All,
                ticks: 5,
                growth_scale: 0.0,
            })
            .expect("valid drought");
        for _ in 0..5 {
            droughted.step().expect("step");
        }
        let during = total_food(&droughted);
        assert!(
            during < total_food(&control) * 0.9,
            "five ticks of total drought must visibly starve the world"
        );
        assert!(
            droughted.active_effects().is_empty(),
            "a 5-tick drought must lapse after 5 ticks"
        );

        // ...and once it lapses, the world recovers.
        for _ in 0..5 {
            droughted.step().expect("step");
        }
        assert!(
            total_food(&droughted) > during,
            "regrowth must resume once the drought lapses"
        );
    }

    #[test]
    fn interventions_are_rejected_not_silently_clamped() {
        // Clamping would hand the caller a DIFFERENT experiment than the one
        // they asked for, and they would never know.
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        assert!(
            world
                .enqueue_intervention(Intervention::Drought {
                    region: Region::All,
                    ticks: 10,
                    growth_scale: 5.0,
                })
                .is_err(),
            "a growth_scale above 1.0 is not a drought"
        );
        assert!(
            world
                .enqueue_intervention(Intervention::Meteor {
                    region: Region::Disc {
                        x: 0.0,
                        y: 0.0,
                        radius: -1.0
                    },
                    lethality: 1.0,
                    scorch: 0.5,
                })
                .is_err(),
            "a negative radius is not a region"
        );
        assert!(
            world
                .enqueue_intervention(Intervention::Bloom {
                    region: Region::All,
                    amount: f32::NAN,
                })
                .is_err(),
            "a NaN bloom would poison every downstream reduction"
        );
    }

    #[test]
    fn the_same_intervention_script_produces_the_same_world() {
        // This is what makes a hand-played session an experiment rather than
        // vandalism: replay the command sequence, get the same world.
        let script = |world: &mut WorldState| {
            world
                .enqueue_intervention(Intervention::Meteor {
                    region: Region::Disc {
                        x: 60.0,
                        y: 60.0,
                        radius: 30.0,
                    },
                    lethality: 0.4,
                    scorch: 0.8,
                })
                .expect("valid");
            world
                .enqueue_intervention(Intervention::Drought {
                    region: Region::All,
                    ticks: 20,
                    growth_scale: 0.25,
                })
                .expect("valid");
        };

        let run = || {
            let mut world = WorldState::new(ScriptBotsConfig {
                world_width: 200,
                world_height: 200,
                food_cell_size: 20,
                rng_seed: Some(0xBEEF),
                ..ScriptBotsConfig::default()
            })
            .expect("world");
            for seed in 0..10 {
                world.spawn_agent(sample_agent(seed));
            }
            for tick in 0..40 {
                if tick == 5 {
                    script(&mut world);
                }
                world.step().expect("step");
            }
            world
                .characterization_digest_v0()
                .expect("quiescent")
                .overall
        };
        assert_eq!(run(), run(), "same script, same seed, same world");
    }

    #[test]
    fn an_intervention_that_matches_nothing_changes_nothing() {
        // A meteor in an empty corner must not perturb the simulation at all —
        // no RNG draws, no agent state, no food. If it did, the intervention
        // system would be injecting hidden nondeterminism into every study.
        let build = || {
            let mut world = WorldState::new(ScriptBotsConfig {
                world_width: 200,
                world_height: 200,
                food_cell_size: 20,
                initial_food: 0.0,
                food_growth_rate: 0.0,
                food_respawn_interval: 0,
                rng_seed: Some(0xFEED),
                ..ScriptBotsConfig::default()
            })
            .expect("world");
            for seed in 0..6 {
                world.spawn_agent(sample_agent(seed));
            }
            world
        };

        let mut untouched = build();
        for _ in 0..20 {
            untouched.step().expect("step");
        }

        let mut poked = build();
        poked
            .enqueue_intervention(Intervention::Meteor {
                // Every agent from sample_agent() sits near the origin, so a
                // meteor way out here can hit nothing.
                region: Region::Disc {
                    x: 150.0,
                    y: 150.0,
                    radius: 5.0,
                },
                lethality: 1.0,
                scorch: 1.0,
            })
            .expect("valid");
        for _ in 0..20 {
            poked.step().expect("step");
        }

        assert_eq!(
            untouched
                .characterization_digest_v0()
                .expect("quiescent")
                .agents,
            poked
                .characterization_digest_v0()
                .expect("quiescent")
                .agents,
            "a meteor that hits nothing must change nothing"
        );
    }

    #[test]
    fn knob_ranges_reject_the_absurd_and_admit_the_hostile() {
        // The hole this closes: ScriptBotsConfig::validate() proves admissibility
        // (finite, non-negative) but declares NO upper bounds, so a value like
        // food_growth_rate = 1e9 sailed through from REST, from MCP, and
        // therefore from any agent driving them. The "a confused model can only
        // request what a human could" safety argument was simply not true.
        let absurd = vec![
            ("food_growth_rate".to_owned(), 1e9),
            ("metabolism_drain".to_owned(), 50.0),
            ("mutation.primary".to_owned(), 4.0),
        ];
        let violations = check_knob_ranges(&absurd);
        assert_eq!(
            violations.len(),
            3,
            "every violation must be reported at once: a caller fixing one knob \
             per round trip gives up, and an autonomous one burns its budget"
        );
        assert!(violations[0].to_string().contains("food_growth_rate"));

        // ...but a researcher must still be able to build a hostile world. These
        // are bounds against the absurd, not against taste.
        let harsh = vec![
            ("metabolism_drain".to_owned(), 0.9),
            ("spike_damage".to_owned(), 9.0),
            ("food_growth_rate".to_owned(), 0.0),
            ("temperature_discomfort_rate".to_owned(), 5.0),
        ];
        assert!(
            check_knob_ranges(&harsh).is_empty(),
            "a brutal-but-sane world must remain expressible: {:?}",
            check_knob_ranges(&harsh)
        );
    }

    #[test]
    fn knob_ranges_reject_non_finite_values_even_for_unlisted_knobs() {
        // A NaN silently poisons every reduction it reaches, so it is refused
        // whether or not the knob carries a declared range.
        let poison = vec![
            ("some_unlisted_knob".to_owned(), f64::NAN),
            ("another_unlisted".to_owned(), f64::INFINITY),
        ];
        assert_eq!(check_knob_ranges(&poison).len(), 2);

        // An unlisted, finite knob is not range-checked here; validate() still
        // governs it. Absence from the table is a gap, not a licence.
        let unlisted = vec![("some_unlisted_knob".to_owned(), 1234.0)];
        assert!(check_knob_ranges(&unlisted).is_empty());
    }

    #[test]
    fn dimension_knobs_are_marked_fresh_world_only() {
        // apply_config_update rejects live dimension changes, so an experiment
        // planner that does not know this generates specs that die at apply
        // time, one run at a time, and blames the simulation.
        for path in ["world_width", "world_height", "food_cell_size"] {
            let range = knob_range(path).expect("dimension knobs are declared");
            assert!(
                range.fresh_world_only,
                "{path} cannot be changed on a live world"
            );
        }
        assert!(
            !knob_range("food_max").expect("declared").fresh_world_only,
            "food_max is settable live"
        );
    }

    #[test]
    fn explain_sensors_reproduces_the_sensors_core_itself_computes() {
        // Freeze the food economy so sensor[4] cannot drift between the
        // explanation and the step; every other channel is position-derived.
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            initial_food: 0.4,
            food_respawn_interval: 0,
            food_growth_rate: 0.0,
            food_decay_rate: 0.0,
            food_diffusion_rate: 0.0,
            rng_seed: Some(77),
            ..ScriptBotsConfig::default()
        })
        .expect("world");

        let observer = world.spawn_agent(AgentData {
            position: Position::new(100.0, 100.0),
            heading: 0.0,
            health: 1.0,
            ..AgentData::default()
        });
        // Neighbours spread around the observer so several eyes and the blood
        // cone are all exercised.
        for (dx, dy, health) in [(30.0, 0.0, 0.4), (20.0, 20.0, 1.5), (-25.0, 10.0, 1.0)] {
            world.spawn_agent(AgentData {
                position: Position::new(100.0 + dx, 100.0 + dy),
                heading: 1.0,
                health,
                color: [0.9, 0.2, 0.5],
                ..AgentData::default()
            });
        }

        let attribution = world
            .explain_sensors(observer, 16)
            .expect("observer exists");
        assert!(
            !attribution.contributions.is_empty(),
            "three neighbours inside the sense radius must contribute"
        );

        world.step().expect("step");
        let sensed = world.agent_runtime(observer).expect("runtime").sensors;

        for (index, channel) in SENSOR_LAYOUT.iter().enumerate() {
            let predicted = attribution.clamped[index];
            let actual = sensed[index];
            assert!(
                (predicted - actual).abs() < 1e-5,
                "channel {} ({index}): explained {predicted}, core computed {actual}",
                channel.name
            );
        }
    }

    #[test]
    fn explain_sensors_is_bounded_deterministic_and_honest_about_saturation() {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            rng_seed: Some(9),
            ..ScriptBotsConfig::default()
        })
        .expect("world");
        let observer = world.spawn_agent(AgentData {
            position: Position::new(100.0, 100.0),
            heading: 0.0,
            health: 1.0,
            ..AgentData::default()
        });
        // A crowd: enough neighbours to saturate the channels and to exceed any
        // sane contributor bound.
        for i in 0..40 {
            let angle = i as f32 * 0.157;
            world.spawn_agent(AgentData {
                position: Position::new(100.0 + 12.0 * angle.cos(), 100.0 + 12.0 * angle.sin()),
                heading: 0.0,
                health: 0.2,
                color: [1.0, 1.0, 1.0],
                ..AgentData::default()
            });
        }

        let attribution = world.explain_sensors(observer, 5).expect("observer");
        assert_eq!(attribution.contributions.len(), 5, "bounded to top-k");
        assert!(
            attribution.truncated >= 30,
            "dropped count must be reported"
        );
        // Strongest first, and the order is total (never float-equality dependent).
        for pair in attribution.contributions.windows(2) {
            assert!(pair[0].total >= pair[1].total);
        }
        // Same world, same explanation.
        let again = world.explain_sensors(observer, 5).expect("observer");
        assert_eq!(attribution, again);

        // Saturation must be SAID, not hidden: contributions legitimately sum
        // past 1.0, and a panel that silently clamps them invites someone to
        // "fix" it by normalising, which destroys the information.
        let smell_index = 10;
        assert!(
            attribution.raw[smell_index] > 1.0,
            "forty close neighbours should oversaturate smell, got {}",
            attribution.raw[smell_index]
        );
        assert!(attribution.saturated[smell_index]);
        assert!((attribution.clamped[smell_index] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sensor_attribution_ties_follow_agent_uid_across_slot_reuse() {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            sense_radius: 80.0,
            rng_seed: Some(19),
            ..ScriptBotsConfig::default()
        })
        .expect("world");
        let observer = world.spawn_agent(AgentData {
            position: Position::new(100.0, 100.0),
            ..AgentData::default()
        });
        let removed = world.spawn_agent(AgentData::default());
        world
            .remove_agent(removed)
            .expect("remove sacrificial slot");

        let tied_agent = || AgentData {
            position: Position::new(120.0, 100.0),
            health: 0.5,
            color: [0.4, 0.6, 0.8],
            ..AgentData::default()
        };
        let reused_slot = world.spawn_agent(tied_agent());
        let appended_slot = world.spawn_agent(tied_agent());
        assert!(
            reused_slot.raw() > appended_slot.raw(),
            "the recycled SlotMap generation must oppose logical creation order in this fixture"
        );

        let attribution = world.explain_sensors(observer, 8).expect("attribution");
        assert_eq!(attribution.contributions.len(), 2);
        assert_eq!(
            attribution.contributions[0].total, attribution.contributions[1].total,
            "fixture must exercise the stable tie-break"
        );
        assert_eq!(
            attribution
                .contributions
                .iter()
                .map(|entry| entry.source)
                .collect::<Vec<_>>(),
            [reused_slot, appended_slot]
        );
        assert_eq!(
            attribution
                .contributions
                .iter()
                .map(|entry| entry.source_uid)
                .collect::<Vec<_>>(),
            [AgentUid(3), AgentUid(4)]
        );
    }

    #[test]
    fn explain_sensors_returns_none_for_a_dead_agent_and_nothing_for_a_lone_one() {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            rng_seed: Some(3),
            ..ScriptBotsConfig::default()
        })
        .expect("world");
        let lone = world.spawn_agent(AgentData {
            position: Position::new(50.0, 50.0),
            health: 1.0,
            ..AgentData::default()
        });

        let attribution = world.explain_sensors(lone, 8).expect("lone agent exists");
        assert!(attribution.contributions.is_empty());
        assert_eq!(attribution.truncated, 0);
        // Self-state channels still carry values; an empty contributor list must
        // never be read as "this agent senses nothing".
        assert!(attribution.clamped[11] > 0.0, "health is self-state");
        assert!(attribution.saturated.iter().all(|hit| !hit));

        world.remove_agent(lone);
        assert!(world.explain_sensors(lone, 8).is_none());
    }

    #[test]
    fn a_single_activation_snapshot_is_size_bounded_and_says_when_it_was_clipped() {
        // Bounding how MANY agents are captured is not enough: brain topology is
        // configuration (a Neuroflow net can declare arbitrarily wide hidden
        // layers), so one inspected agent could otherwise copy megabytes out of
        // the simulation every single tick.
        let huge = BrainActivations {
            layers: vec![
                ActivationLayer {
                    name: "small".to_owned(),
                    width: 4,
                    height: 4,
                    values: vec![0.1; 16],
                },
                ActivationLayer {
                    name: "enormous".to_owned(),
                    width: 1_000,
                    height: 1_000,
                    values: vec![0.2; 1_000_000],
                },
            ],
            connections: vec![ActivationEdge {
                from: 0,
                to: 1,
                weight: 1.0,
            }],
            truncated: false,
        };

        let clipped = clamp_activations(huge);
        let total: usize = clipped.layers.iter().map(|l| l.values.len()).sum();
        assert!(
            total <= ACTIVATION_VALUE_BUDGET,
            "snapshot kept {total} values, budget is {ACTIVATION_VALUE_BUDGET}"
        );
        assert!(
            clipped.truncated,
            "a clipped snapshot must SAY it was clipped; silently dropping layers \
             would let a user conclude the brain has no deep structure"
        );
        // Whole layers survive or are dropped — half a layer is a lie about the
        // shape of the network — and dangling edges are removed.
        assert_eq!(clipped.layers.len(), 1);
        assert_eq!(clipped.layers[0].name, "small");
        assert!(clipped.connections.is_empty());

        // A snapshot that fits is passed through untouched and unflagged.
        let small = BrainActivations {
            layers: vec![ActivationLayer {
                name: "fits".to_owned(),
                width: 2,
                height: 2,
                values: vec![0.5; 4],
            }],
            connections: Vec::new(),
            truncated: false,
        };
        let kept = clamp_activations(small);
        assert!(!kept.truncated);
        assert_eq!(kept.layers.len(), 1);
    }

    #[test]
    fn activation_capture_is_bounded_even_when_every_agent_is_selected() {
        // Regression guard: a frontend "select all" must not reinstate
        // population-wide activation capture. Capture is demand-driven AND
        // bounded; the probed agent is always captured, selected agents only
        // up to the budget.
        #[derive(Debug)]
        struct ChattyBrain;
        impl BrainRunner for ChattyBrain {
            fn kind(&self) -> &'static str {
                "chatty"
            }
            fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
                [0.0; OUTPUT_SIZE]
            }
            fn snapshot_activations(&self) -> Option<BrainActivations> {
                Some(BrainActivations {
                    layers: vec![ActivationLayer {
                        name: "layer".to_owned(),
                        width: 1,
                        height: 1,
                        values: vec![0.5],
                    }],
                    connections: Vec::new(),
                    truncated: false,
                })
            }
        }

        let population = ACTIVATION_CAPTURE_BUDGET * 4;
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(21),
            ..ScriptBotsConfig::default()
        })
        .expect("world");
        let key = world
            .brain_registry_mut()
            .register("chatty", |_rng| Ok(Box::new(ChattyBrain)));

        let mut ids = Vec::new();
        for seed in 0..population {
            let id = world.spawn_agent(sample_agent(seed as u32));
            world
                .bind_agent_brain(id, key)
                .expect("chatty brain factory");
            ids.push(id);
        }

        // Simulate a "select all" from a frontend.
        for id in &ids {
            if let Some(runtime) = world.agent_runtime_mut(*id) {
                runtime.selection = SelectionState::Selected;
            }
        }
        // The inspector is focused on an agent well past the budget cutoff.
        let probed = ids[population - 1];
        world.set_activation_probe(Some(probed));

        world.stage_brains();

        let captured = ids
            .iter()
            .filter(|id| {
                world
                    .agent_runtime(**id)
                    .is_some_and(|rt| rt.brain_activations.is_some())
            })
            .count();
        assert!(
            captured <= ACTIVATION_CAPTURE_BUDGET + 1,
            "select-all captured {captured} activations; budget is {ACTIVATION_CAPTURE_BUDGET} (+1 probe)"
        );
        assert!(
            world
                .agent_runtime(probed)
                .expect("probed runtime")
                .brain_activations
                .is_some(),
            "the probed agent must always be captured, even beyond the budget"
        );
    }

    #[test]
    fn brain_factory_errors_propagate_through_bind_and_world_step() {
        #[derive(Debug, Error)]
        #[error("deliberate adapter construction failure")]
        struct DeliberateFactoryError;

        let mut bind_world = WorldState::new(ScriptBotsConfig::default()).expect("bind world");
        let bind_key = bind_world
            .brain_registry_mut()
            .register("test.fallible", |rng| {
                let _ = rng.next_u64();
                Err(BrainSpawnError::new(
                    "test.fallible",
                    DeliberateFactoryError,
                ))
            });
        let agent = bind_world.spawn_agent(sample_agent(0));
        let before_failed_bind = bind_world
            .characterization_digest_v0()
            .expect("pre-bind digest");
        let bind_error = bind_world
            .bind_agent_brain(agent, bind_key)
            .expect_err("binding must preserve the factory error");
        assert_eq!(bind_error.kind(), "test.fallible");
        assert!(
            std::error::Error::source(&bind_error)
                .and_then(|source| source.downcast_ref::<DeliberateFactoryError>())
                .is_some()
        );
        assert_eq!(
            bind_world
                .characterization_digest_v0()
                .expect("post-bind digest"),
            before_failed_bind,
            "failed binding must restore RNG and leave agent state untouched"
        );

        let config = ScriptBotsConfig {
            population_minimum: 3,
            population_spawn_interval: 0,
            ..ScriptBotsConfig::default()
        };
        let mut identity_reference =
            WorldState::new(config.clone()).expect("identity reference world");
        let reference_existing = identity_reference.spawn_agent(sample_agent(1));
        let expected_next_agent = identity_reference.spawn_agent(sample_agent(2));

        let mut step_world = WorldState::new(config).expect("step world");
        let stable_key = step_world
            .brain_registry_mut()
            .register("test.stable", |_rng| Ok(Box::new(StubBrain)));
        let existing_agent = step_world.spawn_agent(sample_agent(1));
        assert_eq!(existing_agent, reference_existing);
        assert!(
            step_world
                .bind_agent_brain(existing_agent, stable_key)
                .expect("stable brain factory")
        );
        assert!(step_world.brain_registry_mut().unregister(stable_key));
        let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let factory_attempts = std::sync::Arc::clone(&attempts);
        step_world
            .brain_registry_mut()
            .register("test.fallible", move |_rng| {
                if factory_attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    Ok(Box::new(StubBrain))
                } else {
                    Err(BrainSpawnError::new(
                        "test.fallible",
                        DeliberateFactoryError,
                    ))
                }
            });
        let step_error = step_world
            .step()
            .expect_err("population spawn must preserve the factory error");
        assert!(matches!(&step_error, WorldStepError::BrainSpawn(_)));
        let WorldStepError::BrainSpawn(step_error) = step_error else {
            return;
        };
        assert_eq!(step_error.kind(), "test.fallible");
        assert_eq!(step_world.tick(), Tick(1));
        assert_eq!(step_world.agents().len(), 1);
        assert!(
            step_world
                .agent_runtime(existing_agent)
                .expect("existing runtime survives rollback")
                .brain
                .is_bound(),
            "rollback must not strip a pre-existing live brain runner"
        );
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(
            step_world.brain_fault().map(BrainSpawnError::kind),
            Some("test.fallible")
        );
        let actual_next_agent = step_world.spawn_agent(sample_agent(2));
        assert_eq!(
            actual_next_agent, expected_next_agent,
            "failed population construction must restore SlotMap generations and free-list state"
        );
        let completed_digest = step_world
            .characterization_digest_v0()
            .expect("brain-fault tick must finish at a coherent boundary");
        let repeated_error = step_world
            .step()
            .expect_err("latched brain failure must block the next tick");
        assert!(matches!(&repeated_error, WorldStepError::BrainSpawn(_)));
        let WorldStepError::BrainSpawn(repeated_error) = repeated_error else {
            return;
        };
        assert_eq!(repeated_error.kind(), "test.fallible");
        assert_eq!(step_world.tick(), Tick(1));
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(
            step_world
                .characterization_digest_v0()
                .expect("latched fault digest"),
            completed_digest
        );
    }

    #[test]
    fn population_factory_failure_aborts_and_refunds_queued_births() {
        #[derive(Debug, Error)]
        #[error("deliberate population construction failure")]
        struct DeliberatePopulationFailure;

        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_growth_rate: 0.0,
            food_decay_rate: 0.0,
            food_diffusion_rate: 0.0,
            food_intake_rate: 0.0,
            food_waste_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            metabolism_ramp_rate: 0.0,
            metabolism_boost_penalty: 0.0,
            temperature_discomfort_rate: 0.0,
            population_minimum: 2,
            population_spawn_interval: 0,
            reproduction_energy_threshold: 0.5,
            reproduction_energy_cost: 0.25,
            reproduction_cooldown: 1,
            reproduction_attempt_interval: 0,
            reproduction_attempt_chance: 1.0,
            reproduction_rate_herbivore: f32::MIN_POSITIVE,
            reproduction_rate_carnivore: f32::MIN_POSITIVE,
            reproduction_partner_chance: 0.0,
            rng_seed: Some(0xB17A_0B0A),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("world");
        let parent = world.spawn_agent(sample_agent(0));
        {
            let runtime = world.agent_runtime_mut(parent).expect("parent runtime");
            runtime.energy = 1.0;
            runtime.reproduction_counter = 1.0;
        }
        world
            .brain_registry_mut()
            .register("test.population-failure", |_rng| {
                Err(BrainSpawnError::new(
                    "test.population-failure",
                    DeliberatePopulationFailure,
                ))
            });

        let error = world
            .step()
            .expect_err("population construction must fail after queuing a natural birth");
        assert!(matches!(&error, WorldStepError::BrainSpawn(_)));
        let WorldStepError::BrainSpawn(error) = error else {
            return;
        };
        assert_eq!(error.kind(), "test.population-failure");
        assert_eq!(world.tick(), Tick(1));
        assert_eq!(world.agents.len(), 1);
        assert!(world.pending_spawns.is_empty());
        assert_eq!(world.last_births, 0);
        let parent_runtime = world
            .agent_runtime(parent)
            .expect("surviving parent runtime");
        assert!((parent_runtime.energy - 1.0).abs() < f32::EPSILON);
        assert!((parent_runtime.reproduction_counter - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn offspring_snapshot_and_mutation_failures_are_terminal_and_refunded() {
        #[derive(Clone, Copy)]
        enum FailureMode {
            Snapshot,
            Mutation,
        }

        #[derive(Debug, Error)]
        #[error("deliberate inherited-brain operation failure")]
        struct DeliberateHeritageFailure;

        struct FailingHeritageRunner {
            mode: FailureMode,
        }

        impl BrainRunner for FailingHeritageRunner {
            fn kind(&self) -> &'static str {
                match self.mode {
                    FailureMode::Snapshot => "test.snapshot-failure",
                    FailureMode::Mutation => "test.mutation-failure",
                }
            }

            fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
                [0.0; OUTPUT_SIZE]
            }

            fn clone_runner(&self) -> Result<Option<Box<dyn BrainRunner>>, BrainSpawnError> {
                match self.mode {
                    FailureMode::Snapshot => {
                        Err(BrainSpawnError::new(self.kind(), DeliberateHeritageFailure))
                    }
                    FailureMode::Mutation => Ok(Some(Box::new(Self { mode: self.mode }))),
                }
            }

            fn mutate(
                &mut self,
                _rng: &mut dyn RandomStream,
                _rate: f32,
                _scale: f32,
            ) -> Result<(), BrainSpawnError> {
                match self.mode {
                    FailureMode::Snapshot => Ok(()),
                    FailureMode::Mutation => {
                        Err(BrainSpawnError::new(self.kind(), DeliberateHeritageFailure))
                    }
                }
            }
        }

        for mode in [FailureMode::Snapshot, FailureMode::Mutation] {
            let config = ScriptBotsConfig {
                world_width: 100,
                world_height: 100,
                initial_food: 0.0,
                food_respawn_interval: 0,
                food_growth_rate: 0.0,
                food_decay_rate: 0.0,
                food_diffusion_rate: 0.0,
                food_intake_rate: 0.0,
                food_waste_rate: 0.0,
                metabolism_drain: 0.0,
                movement_drain: 0.0,
                metabolism_ramp_rate: 0.0,
                metabolism_boost_penalty: 0.0,
                temperature_discomfort_rate: 0.0,
                population_minimum: 2,
                population_spawn_interval: 0,
                reproduction_energy_threshold: 0.5,
                reproduction_energy_cost: 0.25,
                reproduction_cooldown: 1,
                reproduction_attempt_interval: 0,
                reproduction_attempt_chance: 1.0,
                reproduction_rate_herbivore: f32::MIN_POSITIVE,
                reproduction_rate_carnivore: f32::MIN_POSITIVE,
                reproduction_partner_chance: 0.0,
                persistence_interval: 0,
                rng_seed: Some(0xF411_1B1E),
                ..ScriptBotsConfig::default()
            };
            let mut identity_reference =
                WorldState::new(config.clone()).expect("identity reference world");
            identity_reference.spawn_agent(sample_agent(0));
            let expected_next_agent = identity_reference.spawn_agent(sample_agent(1));

            let mut world = WorldState::new(config).expect("world");
            let factory_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let calls = Arc::clone(&factory_calls);
            let kind = match mode {
                FailureMode::Snapshot => "test.snapshot-failure",
                FailureMode::Mutation => "test.mutation-failure",
            };
            let key = world.brain_registry_mut().register(kind, move |_rng| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(Box::new(FailingHeritageRunner { mode }))
            });
            let parent = world.spawn_agent(sample_agent(0));
            assert!(
                world
                    .bind_agent_brain(parent, key)
                    .expect("initial parent brain construction")
            );
            {
                let runtime = world.agent_runtime_mut(parent).expect("parent runtime");
                runtime.energy = 1.0;
                runtime.reproduction_counter = 1.0;
            }

            let error = world
                .step()
                .expect_err("failed inherited-brain operation must terminate the tick");
            assert!(matches!(&error, WorldStepError::BrainSpawn(_)));
            let WorldStepError::BrainSpawn(error) = error else {
                return;
            };
            assert_eq!(error.kind(), kind);
            assert_eq!(factory_calls.load(std::sync::atomic::Ordering::SeqCst), 2);
            assert_eq!(world.tick(), Tick(1));
            assert_eq!(world.agents.len(), 1);
            assert_eq!(world.runtime.len(), world.agents.len());
            assert!(world.pending_spawns.is_empty());
            assert_eq!(world.last_births, 0);
            let parent_runtime = world
                .agent_runtime(parent)
                .expect("surviving parent runtime");
            assert!((parent_runtime.energy - 1.0).abs() < f32::EPSILON);
            assert!((parent_runtime.reproduction_counter - 1.0).abs() < f32::EPSILON);
            let actual_next_agent = world.spawn_agent(sample_agent(1));
            assert_eq!(actual_next_agent, expected_next_agent);
        }
    }

    #[test]
    fn scheduled_crossover_refuses_bound_nonheritable_parents_without_registry_fallback() {
        struct NonHeritableRunner;

        impl BrainRunner for NonHeritableRunner {
            fn kind(&self) -> &'static str {
                "test.non-heritable"
            }

            fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
                [0.0; OUTPUT_SIZE]
            }
        }

        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            food_waste_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            metabolism_ramp_rate: 0.0,
            metabolism_boost_penalty: 0.0,
            temperature_discomfort_rate: 0.0,
            population_minimum: 0,
            population_spawn_interval: 1,
            population_spawn_count: 1,
            population_crossover_chance: 1.0,
            reproduction_energy_threshold: 0.0,
            persistence_interval: 0,
            rng_seed: Some(0xC205_50A3),
            ..ScriptBotsConfig::default()
        };
        for stale_registry_key in [false, true] {
            let mut world = WorldState::new(config.clone()).expect("world");
            let registry_key = stale_registry_key.then(|| {
                world
                    .brain_registry_mut()
                    .register("test.non-heritable", |_rng| {
                        Ok(Box::new(NonHeritableRunner))
                    })
            });
            for seed in 0..2 {
                let id = world.spawn_agent(sample_agent(seed));
                if let Some(key) = registry_key {
                    assert!(
                        world
                            .bind_agent_brain(id, key)
                            .expect("initial non-heritable runner")
                    );
                } else {
                    world.agent_runtime_mut(id).expect("runtime").brain =
                        BrainBinding::with_runner(Box::new(NonHeritableRunner));
                }
            }
            if let Some(key) = registry_key {
                assert!(world.brain_registry_mut().unregister(key));
            }

            let error = world
                .step()
                .expect_err("non-heritable bound parents must not normalize to random offspring");
            assert!(matches!(&error, WorldStepError::BrainSpawn(_)));
            let WorldStepError::BrainSpawn(error) = error else {
                return;
            };
            assert_eq!(error.kind(), "test.non-heritable");
            assert_eq!(world.tick(), Tick(1));
            assert_eq!(world.agents.len(), 2);
            assert!(world.pending_spawns.is_empty());
        }
    }

    #[test]
    fn fallible_random_spawn_preserves_the_seeded_rng_order() {
        fn register_rng_consuming_brains(world: &mut WorldState) {
            for kind in ["test.rng-a", "test.rng-b"] {
                world.brain_registry_mut().register(kind, move |rng| {
                    let _ = rng.next_u64();
                    Ok(Box::new(StubBrain))
                });
            }
        }

        let config = ScriptBotsConfig {
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(0x5EED),
            ..ScriptBotsConfig::default()
        };
        let mut fallible_world = WorldState::new(config.clone()).expect("fallible world");
        let mut reference_world = WorldState::new(config).expect("reference world");
        register_rng_consuming_brains(&mut fallible_world);
        register_rng_consuming_brains(&mut reference_world);

        fallible_world
            .spawn_random_agent()
            .expect("fallible random spawn");

        let width = reference_world.config.world_width as f32;
        let height = reference_world.config.world_height as f32;
        let data = AgentData::new(
            Position::new(
                reference_world.rng.random_range(0.0..width),
                reference_world.rng.random_range(0.0..height),
            ),
            Velocity::default(),
            reference_world
                .rng
                .random_range(-std::f32::consts::PI..std::f32::consts::PI),
            1.0,
            [
                reference_world.rng.random_range(0.0..1.0),
                reference_world.rng.random_range(0.0..1.0),
                reference_world.rng.random_range(0.0..1.0),
            ],
            0.0,
            false,
            0,
            Generation::default(),
        );
        let id = reference_world.spawn_agent(data);
        if let Some(key) = reference_world
            .brain_registry
            .random_key(&mut reference_world.rng)
        {
            assert!(
                reference_world
                    .bind_agent_brain(id, key)
                    .expect("reference brain factory")
            );
        }

        assert_eq!(
            fallible_world
                .characterization_digest_v0()
                .expect("fallible digest"),
            reference_world
                .characterization_digest_v0()
                .expect("reference digest"),
            "introducing fallible construction must not perturb seeded spawn behavior"
        );
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
            world.step().expect("seeded history step");
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
        fn on_tick(&mut self, payload: &PersistenceBatch) -> Result<(), PersistenceAdmissionError> {
            self.logs.lock().unwrap().push(payload.clone());
            Ok(())
        }
    }

    struct RejectOncePersistence {
        logs: Arc<Mutex<Vec<PersistenceBatch>>>,
        reject_next: bool,
    }

    impl WorldPersistence for RejectOncePersistence {
        fn on_tick(&mut self, payload: &PersistenceBatch) -> Result<(), PersistenceAdmissionError> {
            self.logs.lock().unwrap().push(payload.clone());
            if self.reject_next {
                self.reject_next = false;
                Err(PersistenceAdmissionError::new(
                    payload.summary.tick.0,
                    "injected definite non-admission",
                ))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn simultaneous_brain_and_persistence_failures_are_both_latched() {
        #[derive(Debug, Error)]
        #[error("deliberate population factory failure")]
        struct DeliberatePopulationFactoryError;

        let config = ScriptBotsConfig {
            population_minimum: 3,
            population_spawn_interval: 0,
            persistence_interval: 1,
            rng_seed: Some(0x0BAD_5EED),
            ..ScriptBotsConfig::default()
        };
        let persistence_logs = Arc::new(Mutex::new(Vec::new()));
        let persistence = RejectOncePersistence {
            logs: Arc::clone(&persistence_logs),
            reject_next: true,
        };
        let mut world =
            WorldState::with_persistence(config, Box::new(persistence)).expect("test world");
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let factory_attempts = Arc::clone(&attempts);
        world
            .brain_registry_mut()
            .register("test.double-fault", move |_rng| {
                if factory_attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    Ok(Box::new(StubBrain))
                } else {
                    Err(BrainSpawnError::new(
                        "test.double-fault",
                        DeliberatePopulationFactoryError,
                    ))
                }
            });

        let first_error = world
            .step()
            .expect_err("both terminal failures must be reported at the completed boundary");
        assert!(matches!(
            &first_error,
            WorldStepError::BrainAndPersistence { .. }
        ));
        let WorldStepError::BrainAndPersistence { brain, persistence } = first_error else {
            return;
        };
        assert_eq!(brain.kind(), "test.double-fault");
        assert_eq!(persistence.tick(), 1);
        assert_eq!(world.tick(), Tick(1));
        assert!(
            world.agents().is_empty(),
            "population inserts must roll back"
        );
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(persistence_logs.lock().unwrap().len(), 1);
        assert_eq!(
            world.brain_fault().map(BrainSpawnError::kind),
            Some("test.double-fault")
        );
        assert_eq!(world.persistence_fault(), Some(&persistence));

        let completed_digest = world
            .characterization_digest_v0()
            .expect("double-fault tick must finish at a coherent boundary");
        assert!(matches!(
            world.step(),
            Err(WorldStepError::BrainAndPersistence { .. })
        ));
        assert_eq!(world.tick(), Tick(1));
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(persistence_logs.lock().unwrap().len(), 1);
        assert_eq!(
            world
                .characterization_digest_v0()
                .expect("latched double fault digest"),
            completed_digest,
            "repeated step must not mutate after a combined terminal fault"
        );
    }

    fn lifecycle_birth(tick: u64) -> BirthRecord {
        BirthRecord {
            tick: Tick(tick),
            agent_uid: AgentUid(tick + 1),
            spawn_ordinal: tick,
            birth_ordinal: tick,
            parent_a: None,
            parent_b: None,
            brain_kind: Some("test".to_owned()),
            brain_key: Some(tick),
            herbivore_tendency: 0.5,
            generation: Generation(tick as u32),
            position: Position::new(tick as f32, 1.0),
            is_hybrid: false,
        }
    }

    fn lifecycle_death(tick: u64) -> DeathRecord {
        DeathRecord {
            tick: Tick(tick),
            agent_uid: AgentUid(tick + 1),
            age: tick as u32,
            generation: Generation(tick as u32),
            herbivore_tendency: 0.5,
            brain_kind: Some("test".to_owned()),
            brain_key: Some(tick),
            energy: 0.0,
            food_balance_total: -1.0,
            cause: DeathCause::Starvation,
            was_hybrid: false,
            combat_flags: CombatEventFlags::default(),
        }
    }

    fn replay_marker(value: f32) -> ReplayEvent {
        ReplayEvent {
            agent_uid: None,
            kind: ReplayEventKind::RngSample {
                scope: ReplayRngScope::World,
                range_min: 0.0,
                range_max: 1.0,
                value,
            },
        }
    }

    #[test]
    fn rejected_persistence_latches_world_until_explicit_exact_batch_retry() {
        let config = ScriptBotsConfig {
            persistence_interval: 1,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(77),
            ..ScriptBotsConfig::default()
        };
        let rejected_logs = Arc::new(Mutex::new(Vec::new()));
        let rejecting = RejectOncePersistence {
            logs: Arc::clone(&rejected_logs),
            reject_next: true,
        };
        let mut world =
            WorldState::with_persistence(config, Box::new(rejecting)).expect("test world");
        world.pending_birth_records.push(lifecycle_birth(1));
        world
            .pending_lifecycle_birth_metrics
            .push(lifecycle_birth(1));
        world.pending_death_records.push(lifecycle_death(1));
        world.last_births = 1;
        world.last_deaths = 1;
        world.replay_events.push(replay_marker(0.25));

        let first_error = world
            .step()
            .expect_err("definite persistence rejection must fail the completed tick");
        assert!(matches!(&first_error, WorldStepError::Persistence(_)));
        let WorldStepError::Persistence(first_error) = first_error else {
            return;
        };
        assert_eq!(first_error.tick(), 1);
        assert_eq!(first_error.state(), PersistenceAdmissionState::NotAdmitted);
        assert_eq!(world.tick(), Tick(1));
        assert!(world.has_pending_persistence_batch());
        assert_eq!(world.persistence_fault(), Some(&first_error));

        let digest_after_failure = world
            .characterization_digest_v0()
            .expect("failed tick must end at a quiescent science boundary");
        let rejected_call_count = rejected_logs.lock().unwrap().len();
        let repeated_error = world
            .step()
            .expect_err("a latched failure must prevent tick two from starting");
        assert!(matches!(&repeated_error, WorldStepError::Persistence(_)));
        let WorldStepError::Persistence(repeated_error) = repeated_error else {
            return;
        };
        assert_eq!(repeated_error, first_error);
        assert_eq!(world.tick(), Tick(1));
        assert_eq!(rejected_logs.lock().unwrap().len(), rejected_call_count);
        assert_eq!(
            world.characterization_digest_v0().unwrap(),
            digest_after_failure,
            "latched step must not mutate science state"
        );

        let accepted = SpyPersistence::default();
        let accepted_logs = Arc::clone(&accepted.logs);
        world.set_persistence(Box::new(accepted));
        assert!(
            world
                .retry_pending_persistence()
                .expect("replacement sink must accept retained batch")
        );
        assert!(!world.has_pending_persistence_batch());
        assert!(world.persistence_fault().is_none());

        let rejected = rejected_logs.lock().unwrap();
        let accepted = accepted_logs.lock().unwrap();
        assert_eq!(rejected.len(), 1);
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].summary, rejected[0].summary);
        assert_eq!(accepted[0].metrics, rejected[0].metrics);
        assert_eq!(accepted[0].events, rejected[0].events);
        assert_eq!(accepted[0].births, rejected[0].births);
        assert_eq!(accepted[0].deaths, rejected[0].deaths);
        assert_eq!(accepted[0].replay_events, rejected[0].replay_events);
        assert_eq!(accepted[0].agents.len(), rejected[0].agents.len());
        drop(accepted);
        drop(rejected);

        world
            .step()
            .expect("world may advance only after retained batch admission");
        assert_eq!(world.tick(), Tick(2));
        assert_eq!(accepted_logs.lock().unwrap().len(), 2);
    }

    #[test]
    fn persistence_stage_does_not_own_current_tick_counters() {
        let config = ScriptBotsConfig {
            persistence_interval: 3,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(82),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("test world");
        world.last_births = 2;
        world.last_deaths = 1;
        world.combat_spike_attempts = 3;
        world.combat_spike_hits = 2;

        world.stage_accumulate_tick_events();
        world.stage_record_history(Tick(1));

        world
            .stage_persistence(Tick(1), false)
            .expect("non-boundary persistence stage");

        assert_eq!(world.pending_birth_events, 2);
        assert_eq!(world.pending_death_events, 1);
        assert_eq!(world.pending_spike_attempt_events, 3);
        assert_eq!(world.pending_spike_hit_events, 2);
        assert_eq!(world.last_births, 2);
        assert_eq!(world.last_deaths, 1);
        assert_eq!(world.combat_spike_attempts, 3);
        assert_eq!(world.combat_spike_hits, 2);
        let summary = world.history().next_back().expect("current summary");
        assert_eq!(summary.tick, Tick(1));
        assert_eq!(summary.births, 2);
        assert_eq!(summary.deaths, 1);
        assert_eq!(summary.spike_hits, 2);

        world.stage_reset_events(false);
        assert_eq!(world.last_births, 0);
        assert_eq!(world.last_deaths, 0);
        assert_eq!(world.combat_spike_attempts, 0);
        assert_eq!(world.combat_spike_hits, 0);
        assert_eq!(world.pending_birth_events, 2);
        assert_eq!(world.pending_death_events, 1);
        assert_eq!(world.pending_spike_attempt_events, 3);
        assert_eq!(world.pending_spike_hit_events, 2);
    }

    #[test]
    fn disabled_persistence_keeps_current_summary_honest_without_retaining_batches() {
        let config = ScriptBotsConfig {
            persistence_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(84),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("test world");
        world.pending_birth_records.push(lifecycle_birth(1));
        world.pending_death_records.push(lifecycle_death(1));
        world.last_births = 1;
        world.last_deaths = 1;
        world.combat_spike_attempts = 2;
        world.combat_spike_hits = 3;
        world.replay_events.push(replay_marker(0.5));

        world.step().expect("persistence-disabled tick");

        let summary = world.history().next_back().expect("current summary");
        assert_eq!(summary.tick, Tick(1));
        assert_eq!(summary.births, 1);
        assert_eq!(summary.deaths, 1);
        assert_eq!(summary.spike_hits, 3);
        assert!(world.pending_birth_records.is_empty());
        assert!(world.pending_death_records.is_empty());
        assert!(world.replay_events.is_empty());
        assert_eq!(world.pending_birth_events, 0);
        assert_eq!(world.pending_death_events, 0);
        assert_eq!(world.pending_spike_attempt_events, 0);
        assert_eq!(world.pending_spike_hit_events, 0);
        assert_eq!(world.last_births, 0);
        assert_eq!(world.last_deaths, 0);
        assert_eq!(world.combat_spike_attempts, 0);
        assert_eq!(world.combat_spike_hits, 0);
    }

    #[test]
    fn skipped_persistence_ticks_retain_lifecycle_and_replay_rows() {
        let config = ScriptBotsConfig {
            persistence_interval: 3,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(78),
            ..ScriptBotsConfig::default()
        };
        let spy = SpyPersistence::default();
        let logs = Arc::clone(&spy.logs);
        let mut world = WorldState::with_persistence(config, Box::new(spy)).expect("test world");

        world.pending_birth_records.push(lifecycle_birth(1));
        world.last_births = 1;
        world.combat_spike_attempts = 1;
        world.combat_spike_hits = 2;
        world.replay_events.push(replay_marker(0.1));
        world.step().expect("tick one");
        assert!(logs.lock().unwrap().is_empty());
        let tick_one = world.history().next_back().expect("tick one summary");
        assert_eq!(tick_one.tick, Tick(1));
        assert_eq!(tick_one.births, 1);
        assert_eq!(tick_one.deaths, 0);
        assert_eq!(tick_one.spike_hits, 2);

        world.pending_birth_records.push(lifecycle_birth(2));
        world.pending_death_records.push(lifecycle_death(2));
        world.last_births = 1;
        world.last_deaths = 1;
        world.combat_spike_attempts = 2;
        world.combat_spike_hits = 1;
        world.replay_events.push(replay_marker(0.2));
        world.step().expect("tick two");
        assert!(logs.lock().unwrap().is_empty());
        let tick_two = world.history().next_back().expect("tick two summary");
        assert_eq!(tick_two.tick, Tick(2));
        assert_eq!(tick_two.births, 1);
        assert_eq!(tick_two.deaths, 1);
        assert_eq!(tick_two.spike_hits, 1);

        world.step().expect("tick three persistence boundary");
        let tick_three = world.history().next_back().expect("tick three summary");
        assert_eq!(tick_three.tick, Tick(3));
        assert_eq!(tick_three.births, 0);
        assert_eq!(tick_three.deaths, 0);
        assert_eq!(tick_three.spike_hits, 0);
        let entries = logs.lock().unwrap();
        assert_eq!(entries.len(), 1);
        let batch = &entries[0];
        assert_eq!(batch.summary.tick, Tick(3));
        assert_eq!(batch.summary.births, 2);
        assert_eq!(batch.summary.deaths, 1);
        assert_eq!(batch.summary.spike_hits, 3);
        assert_eq!(batch.births, vec![lifecycle_birth(1), lifecycle_birth(2)]);
        assert_eq!(batch.deaths, vec![lifecycle_death(2)]);
        assert!(batch.events.iter().any(|event| {
            matches!(
                &event.kind,
                PersistenceEventKind::Custom(name) if name == "spike_attempts"
            ) && event.count == 3
        }));
        assert!(batch.events.iter().any(|event| {
            matches!(
                &event.kind,
                PersistenceEventKind::Custom(name) if name == "spike_hits"
            ) && event.count == 3
        }));
        assert_eq!(
            batch.replay_events,
            vec![replay_marker(0.1), replay_marker(0.2)]
        );
    }

    #[test]
    fn persistence_and_analytics_cadence_preserve_science_and_event_totals() {
        let run = |persistence_interval: u32, lifecycle_events: u32| {
            let config = ScriptBotsConfig {
                persistence_interval,
                analytics_stride: AnalyticsStride {
                    lifecycle_events,
                    ..AnalyticsStride::default()
                },
                population_minimum: 0,
                population_spawn_interval: 0,
                rng_seed: Some(83),
                ..ScriptBotsConfig::default()
            };
            let spy = SpyPersistence::default();
            let logs = Arc::clone(&spy.logs);
            let mut world =
                WorldState::with_persistence(config, Box::new(spy)).expect("cadence world");

            for tick in 1..=6 {
                match tick {
                    1 | 4 => {
                        let birth = lifecycle_birth(tick);
                        world.pending_birth_records.push(birth.clone());
                        world.pending_lifecycle_birth_metrics.push(birth);
                        world.last_births = 1;
                    }
                    2 | 5 => {
                        let death = lifecycle_death(tick);
                        world.pending_death_records.push(death.clone());
                        world.pending_lifecycle_death_metrics.push(death);
                        world.last_deaths = 1;
                    }
                    _ => {}
                }
                if matches!(tick, 1 | 2 | 4 | 5) {
                    world.combat_spike_attempts = 1;
                    world.combat_spike_hits = 2;
                }
                world.step().expect("cadence comparison tick");
            }

            let entries = logs.lock().unwrap();
            let mut event_totals = (0usize, 0usize, 0usize, 0usize);
            for event in entries.iter().flat_map(|batch| &batch.events) {
                match &event.kind {
                    PersistenceEventKind::Births => event_totals.0 += event.count,
                    PersistenceEventKind::Deaths => event_totals.1 += event.count,
                    PersistenceEventKind::Custom(name) if name == "spike_attempts" => {
                        event_totals.2 += event.count;
                    }
                    PersistenceEventKind::Custom(name) if name == "spike_hits" => {
                        event_totals.3 += event.count;
                    }
                    PersistenceEventKind::Custom(_) => {}
                }
            }
            let summary_totals = entries.iter().fold((0usize, 0usize, 0u32), |total, batch| {
                (
                    total.0 + batch.summary.births,
                    total.1 + batch.summary.deaths,
                    total.2 + batch.summary.spike_hits,
                )
            });
            let birth_ticks = entries
                .iter()
                .flat_map(|batch| &batch.births)
                .map(|record| record.tick)
                .collect::<Vec<_>>();
            let death_ticks = entries
                .iter()
                .flat_map(|batch| &batch.deaths)
                .map(|record| record.tick)
                .collect::<Vec<_>>();
            (
                world.history().cloned().collect::<Vec<_>>(),
                world.food().cells().to_vec(),
                world.tick(),
                world.agent_count(),
                event_totals,
                summary_totals,
                birth_ticks,
                death_ticks,
            )
        };

        let every_two_ticks = run(2, 5);
        let every_three_ticks = run(3, 4);
        assert_eq!(every_two_ticks, every_three_ticks);
        assert_eq!(every_two_ticks.4, (2, 2, 4, 8));
        assert_eq!(every_two_ticks.5, (2, 2, 8));
        assert_eq!(every_two_ticks.6, vec![Tick(1), Tick(4)]);
        assert_eq!(every_two_ticks.7, vec![Tick(2), Tick(5)]);
    }

    #[test]
    fn persistence_and_lifecycle_cadences_do_not_double_count_events() {
        let config = ScriptBotsConfig {
            persistence_interval: 3,
            analytics_stride: AnalyticsStride {
                lifecycle_events: 6,
                ..AnalyticsStride::default()
            },
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(79),
            ..ScriptBotsConfig::default()
        };
        let spy = SpyPersistence::default();
        let logs = Arc::clone(&spy.logs);
        let mut world = WorldState::with_persistence(config, Box::new(spy)).expect("test world");

        world.pending_birth_records.push(lifecycle_birth(1));
        world
            .pending_lifecycle_birth_metrics
            .push(lifecycle_birth(1));
        world.last_births = 1;
        world.step().expect("tick one");
        world.step().expect("tick two");
        world.step().expect("tick three persistence boundary");

        world.pending_birth_records.push(lifecycle_birth(4));
        world
            .pending_lifecycle_birth_metrics
            .push(lifecycle_birth(4));
        world.last_births = 1;
        world.step().expect("tick four");
        world.step().expect("tick five");
        world
            .step()
            .expect("tick six persistence and lifecycle boundary");

        let entries = logs.lock().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].summary.births, 1);
        assert_eq!(entries[0].births, vec![lifecycle_birth(1)]);
        assert_eq!(entries[1].summary.births, 1);
        assert_eq!(entries[1].births, vec![lifecycle_birth(4)]);
        let birth_event_total: usize = entries
            .iter()
            .flat_map(|batch| &batch.events)
            .filter(|event| matches!(event.kind, PersistenceEventKind::Births))
            .map(|event| event.count)
            .sum();
        let raw_birth_total: usize = entries.iter().map(|batch| batch.births.len()).sum();
        assert_eq!(birth_event_total, raw_birth_total);
        let lifecycle_metric_total: f64 = entries
            .iter()
            .flat_map(|batch| &batch.metrics)
            .filter(|metric| metric.name == "births.total.count")
            .map(|metric| metric.value)
            .sum();
        assert_eq!(lifecycle_metric_total, raw_birth_total as f64);
        assert_eq!(
            entries
                .iter()
                .flat_map(|batch| &batch.births)
                .map(|record| record.tick)
                .collect::<Vec<_>>(),
            vec![Tick(1), Tick(4)]
        );
    }

    #[test]
    fn finalize_persistence_admits_partial_tail_exactly_once() {
        let config = ScriptBotsConfig {
            persistence_interval: 3,
            analytics_stride: AnalyticsStride {
                lifecycle_events: 6,
                ..AnalyticsStride::default()
            },
            population_minimum: 0,
            population_spawn_interval: 0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            food_intake_rate: 0.0,
            food_waste_rate: 0.0,
            food_sharing_rate: 0.0,
            reproduction_energy_threshold: 10.0,
            rng_seed: Some(81),
            ..ScriptBotsConfig::default()
        };
        let spy = SpyPersistence::default();
        let logs = Arc::clone(&spy.logs);
        let mut world = WorldState::with_persistence(config, Box::new(spy)).expect("test world");
        let agent_id = world.spawn_agent(sample_agent(0));

        world.pending_birth_records.push(lifecycle_birth(1));
        world.last_births = 1;
        world.step().expect("tick one");
        world.step().expect("tick two");
        world.step().expect("tick three persistence boundary");

        world.pending_birth_records.push(lifecycle_birth(4));
        world.pending_death_records.push(lifecycle_death(4));
        world.last_births = 1;
        world.last_deaths = 1;
        world.replay_events.push(replay_marker(0.4));
        world.agent_runtime_mut(agent_id).unwrap().food_delta = 0.4;
        world.step().expect("tick four partial cadence tail");
        assert_eq!(world.agent_runtime(agent_id).unwrap().food_delta, 0.0);

        assert!(world.finalize_persistence().expect("tail admission"));
        assert!(
            !world
                .finalize_persistence()
                .expect("idempotent tail finalization")
        );

        let entries = logs.lock().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].summary.tick, Tick(3));
        assert_eq!(entries[0].births, vec![lifecycle_birth(1)]);
        let tail = &entries[1];
        assert_eq!(tail.summary.tick, Tick(4));
        assert_eq!(tail.summary.births, 1);
        assert_eq!(tail.summary.deaths, 1);
        assert_eq!(tail.births, vec![lifecycle_birth(4)]);
        assert_eq!(tail.deaths, vec![lifecycle_death(4)]);
        assert_eq!(tail.replay_events, vec![replay_marker(0.4)]);
        assert!((tail.agents[0].runtime.food_delta - 0.4).abs() < 1e-6);
        assert!(tail.metrics.iter().any(|metric| {
            metric.name == "food_delta.mean" && (metric.value - 0.4).abs() < 1e-6
        }));
    }

    #[test]
    fn food_balance_accumulates_every_tick_between_persistence_boundaries() {
        let config = ScriptBotsConfig {
            persistence_interval: 3,
            population_minimum: 0,
            population_spawn_interval: 0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            food_intake_rate: 0.0,
            food_waste_rate: 0.0,
            food_sharing_rate: 0.0,
            reproduction_energy_threshold: 10.0,
            rng_seed: Some(80),
            ..ScriptBotsConfig::default()
        };
        let spy = SpyPersistence::default();
        let logs = Arc::clone(&spy.logs);
        let mut world = WorldState::with_persistence(config, Box::new(spy)).expect("test world");
        let agent_id = world.spawn_agent(sample_agent(0));

        for (index, delta) in [0.5, -0.25, 1.0].into_iter().enumerate() {
            world.agent_runtime_mut(agent_id).unwrap().food_delta = delta;
            world.step().expect("persistence cadence step");
            assert_eq!(world.tick(), Tick(index as u64 + 1));
        }

        let expected = 1.25;
        assert!(
            (world
                .agent_runtime(agent_id)
                .expect("live agent runtime")
                .food_balance_total
                - expected)
                .abs()
                < 1e-6
        );
        let entries = logs.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(
            (entries[0].agents[0].runtime.food_balance_total - expected).abs() < 1e-6,
            "persisted agent snapshot must include every completed tick"
        );
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

        world.step().expect("persistence fixture step");

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
        world.step().expect("reproduction step");
        assert_eq!(world.agent_count(), 2);

        let handles: Vec<_> = world.agents().iter_handles().collect();
        let child_id = handles
            .into_iter()
            .find(|id| *id != parent_id)
            .expect("child");
        let child_state = world.snapshot_agent(child_id).expect("child state");
        let parent_uid = world.agent_uid(parent_id).expect("parent uid");
        assert_eq!(child_state.data.generation, Generation(1));
        assert_eq!(child_state.identity.uid, AgentUid(2));
        assert_eq!(child_state.identity.spawn_ordinal, 1);
        assert_eq!(child_state.identity.birth_ordinal, Some(0));
        assert_eq!(child_state.runtime.lineage, [Some(parent_uid), None]);
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
            world.step().expect("reproduction cadence step");
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
            let events = world.step().expect("reproduction sequence step");
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

        world.step().expect("hybrid reproduction step");

        let child_id = world
            .agents()
            .iter_handles()
            .find(|id| *id != parent && *id != partner)
            .expect("child spawned");
        let child_runtime = world.agent_runtime(child_id).expect("child runtime");
        assert!(child_runtime.hybrid, "child should be marked hybrid");
        assert_eq!(child_runtime.lineage[0], world.agent_uid(parent));
        assert_eq!(child_runtime.lineage[1], world.agent_uid(partner));
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
            .register("test.idle", |_rng| Ok(Box::new(IdleBrain)));
        assert!(
            world
                .bind_agent_brain(parent, idle_key)
                .expect("idle brain factory")
        );

        world.step().expect("child placement step");

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
    fn the_climate_gradient_is_a_pure_function_with_no_entropy_and_no_infinities() {
        // Temperature must never consume RNG. A climate that drew entropy would
        // make every run's weather a function of how many other draws happened
        // first, and the whole determinism programme would be built on sand.
        let config = ScriptBotsConfig {
            world_width: 1000,
            world_height: 200,
            temperature_gradient_exponent: 1.0,
            ..ScriptBotsConfig::default()
        };

        // COLD at the equator (the middle), HOT at the edges — legacy's convention.
        assert!(
            sample_temperature(&config, 500.0) < 1e-6,
            "the equator is cold"
        );
        assert!(
            (sample_temperature(&config, 0.0) - 1.0).abs() < 1e-6,
            "the west edge is hot"
        );
        assert!(
            (sample_temperature(&config, 1000.0) - 1.0).abs() < 1e-6,
            "the east edge is hot"
        );

        // THE SEAM. x = 0 and x = W are the SAME point on a torus, so they must
        // report the same temperature. A climate with a discontinuity at the seam
        // would give agents a free thermal cliff to exploit by stepping across it.
        assert!(
            (sample_temperature(&config, 0.0) - sample_temperature(&config, 1000.0)).abs() < 1e-6,
            "the seam must not be a temperature discontinuity"
        );

        // BOUNDARY and beyond: out-of-range coordinates wrap rather than escaping
        // the [0, 1] range.
        for x in [-2500.0f32, -1.0, 0.0, 499.9, 500.1, 1000.0, 2500.0, 1e9] {
            let temperature = sample_temperature(&config, x);
            assert!(
                temperature.is_finite() && (0.0..=1.0).contains(&temperature),
                "temperature at x = {x} left [0, 1]: {temperature}"
            );
        }

        // DEGENERATE: a zero-width world must report a uniform temperature rather
        // than dividing by zero.
        let degenerate = ScriptBotsConfig {
            world_width: 0,
            ..ScriptBotsConfig::default()
        };
        let temperature = sample_temperature(&degenerate, 123.0);
        assert!(
            temperature.is_finite(),
            "a degenerate world must not produce NaN"
        );

        // Purity: same input, same answer, forever.
        assert_eq!(
            sample_temperature(&config, 321.0).to_bits(),
            sample_temperature(&config, 321.0).to_bits()
        );
    }

    #[test]
    fn the_climate_matches_the_legacy_oracle_except_where_we_chose_to_differ() {
        // The parity-versus-policy decision, checked against the actual legacy
        // formula (World.cpp:95-100) rather than against a memory of it.
        let rate = 0.005_f32; // legacy's own "decent value" comment
        let config = ScriptBotsConfig {
            world_width: 1000,
            temperature_gradient_exponent: 1.0, // legacy is linear
            temperature_comfort_band: DEFAULT_TEMPERATURE_COMFORT_BAND,
            temperature_discomfort_exponent: 2.0, // legacy squares
            temperature_discomfort_rate: rate,
            ..ScriptBotsConfig::default()
        };

        let ours = |x: f32, preference: f32| -> f32 {
            let temperature = sample_temperature(&config, x);
            let discomfort = temperature_discomfort(temperature, preference);
            let band = config.temperature_comfort_band;
            if discomfort <= band {
                return 0.0;
            }
            rate * (discomfort - band).powf(config.temperature_discomfort_exponent)
        };

        for step in 0..=100 {
            let x = step as f32 * 10.0;
            for preference in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
                let legacy = legacy_temperature_health_drain(x / 1000.0, preference, rate);
                let mine = ours(x, preference);

                assert!(mine.is_finite() && legacy.is_finite());
                assert!(mine >= 0.0, "a drain must never heal an agent");

                // AGREEMENT ON THE COMFORT ZONE. This is the part that must match,
                // and the part the old constant got wrong: inside the band that
                // legacy considers comfortable, we must not drain health either.
                if legacy == 0.0 {
                    assert_eq!(
                        mine, 0.0,
                        "x = {x}, preference = {preference}: legacy considers this \
                         agent comfortable, so we must not be draining its health. \
                         (This is exactly what the ported 0.08 constant got wrong: \
                         it gated the RAW discomfort on a threshold legacy applied \
                         to the SQUARE.)"
                    );
                } else {
                    // DELIBERATE DIVERGENCE, bounded. Beyond the band we ramp from
                    // zero instead of stepping off legacy's cliff, so our drain is
                    // always <= legacy's. It is never larger, so no agent is
                    // punished harder than legacy would punish it.
                    assert!(
                        mine <= legacy + 1e-6,
                        "x = {x}, preference = {preference}: our continuous ramp must \
                         never exceed legacy's step ({mine} > {legacy})"
                    );
                }
            }
        }
    }

    #[test]
    fn the_ported_constant_really_did_punish_agents_legacy_called_comfortable() {
        // Proof that the defect was real, not a story. Legacy gates on the SQUARE
        // of the discomfort, so its comfort zone extends to sqrt(0.08) ~= 0.283 in
        // the raw domain. The port compared the RAW discomfort against 0.08, which
        // is a comfort zone ~3.5x narrower.
        //
        // Pick a discomfort that sits in the gap: comfortable to legacy, punished
        // by the old port.
        let discomfort = 0.15_f32;
        assert!(
            discomfort > LEGACY_COMFORT_BAND_SQUARED,
            "must exceed the OLD (wrongly-ported) raw threshold"
        );
        assert!(
            discomfort < DEFAULT_TEMPERATURE_COMFORT_BAND,
            "but must lie INSIDE the comfort zone legacy actually intended"
        );

        // Legacy: comfortable. No drain.
        assert_eq!(
            legacy_temperature_health_drain(0.5 + discomfort / 2.0, 0.0, 0.005),
            0.0,
            "legacy considers this agent comfortable"
        );

        // The corrected band agrees with legacy; the old one did not.
        assert!(
            discomfort <= DEFAULT_TEMPERATURE_COMFORT_BAND,
            "the corrected band leaves this agent alone"
        );
        assert!(
            discomfort > LEGACY_COMFORT_BAND_SQUARED,
            "the old band drained its health every single tick, forever, for being \
             0.15 away from its preferred temperature — a distance legacy treats as \
             perfectly comfortable"
        );
    }

    #[test]
    fn the_climate_drain_is_identical_however_many_threads_run_it() {
        // The SIMD and scalar lanes must agree, and so must one thread and many:
        // a health drain that depended on the worker count would make a run's
        // outcome a property of the machine it ran on.
        let config = ScriptBotsConfig {
            world_width: 400,
            world_height: 400,
            food_cell_size: 20,
            initial_food: 0.0,
            food_respawn_interval: 0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.5,
            temperature_comfort_band: 0.05,
            temperature_gradient_exponent: 1.0,
            temperature_discomfort_exponent: 2.0,
            rng_seed: Some(7),
            ..ScriptBotsConfig::default()
        };

        let run = || {
            let mut world = WorldState::new(config.clone()).expect("world");
            for seed in 0..24 {
                world.spawn_agent(sample_agent(seed));
            }
            for _ in 0..25 {
                world.step().expect("step");
            }
            world
                .characterization_digest_v0()
                .expect("quiescent")
                .agents
        };
        assert_eq!(run(), run(), "the climate drain must be reproducible");
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
        world.stage_accumulate_tick_events();
        world
            .stage_persistence(Tick(1), false)
            .expect("carcass metrics should be admitted");

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
            columns.health_mut()[idx] = 1.0;
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
        let health = world
            .snapshot_agent(agent)
            .expect("herbivore should remain alive")
            .data
            .health;
        assert!(
            (health - 1.0).abs() < 1e-6,
            "ground-food policy should leave health unchanged, got {health:.6}"
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
            // below the health cap so the intake/waste gate stays open
            columns.health_mut()[idx] = 1.0;
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
            // below the health cap so the intake/waste gate stays open
            columns.health_mut()[fertile_slot] = 1.0;
            columns.health_mut()[infertile_slot] = 1.0;
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
    fn food_sharing_rebuilds_current_positions_across_toroidal_seam() {
        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            food_transfer_rate: 0.01,
            food_sharing_distance: 6.0,
            rng_seed: Some(203),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let giver = world.spawn_agent(sample_agent(0));
        let receiver = world.spawn_agent(sample_agent(1));

        {
            let arena = world.agents_mut();
            let giver_idx = arena.index_of(giver).unwrap();
            let receiver_idx = arena.index_of(receiver).unwrap();
            let positions = arena.columns_mut().positions_mut();
            positions[giver_idx] = Position::new(30.0, 50.0);
            positions[receiver_idx] = Position::new(70.0, 50.0);
        }
        world.stage_sense();
        {
            let arena = world.agents_mut();
            let giver_idx = arena.index_of(giver).unwrap();
            let receiver_idx = arena.index_of(receiver).unwrap();
            let positions = arena.columns_mut().positions_mut();
            positions[giver_idx] = Position::new(2.0, 50.0);
            positions[receiver_idx] = Position::new(98.0, 50.0);
        }
        {
            let runtime = world.agent_runtime_mut(giver).unwrap();
            runtime.energy = 1.0;
            runtime.give_intent = 1.0;
        }
        world.agent_runtime_mut(receiver).unwrap().energy = 0.5;

        world.stage_food();

        assert!(
            (world.agent_runtime(giver).unwrap().energy - 0.99).abs() < 1e-6,
            "giver should find the receiver using the stage's current position index"
        );
        assert!(
            (world.agent_runtime(receiver).unwrap().energy - 0.51).abs() < 1e-6,
            "minimum-image distance should share across the world seam"
        );
    }

    #[test]
    fn food_sharing_sorts_wrapped_bucket_candidates_by_dense_index() {
        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            food_transfer_rate: 0.01,
            food_sharing_distance: 5.0,
            rng_seed: Some(204),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        let giver = world.spawn_agent(sample_agent(0));
        let lower_index_recipient = world.spawn_agent(sample_agent(1));
        let higher_index_recipient = world.spawn_agent(sample_agent(2));

        {
            let arena = world.agents_mut();
            let giver_idx = arena.index_of(giver).unwrap();
            let lower_idx = arena.index_of(lower_index_recipient).unwrap();
            let higher_idx = arena.index_of(higher_index_recipient).unwrap();
            assert!(
                lower_idx < higher_idx,
                "fixture must encode dense-index order"
            );
            let positions = arena.columns_mut().positions_mut();
            positions[giver_idx] = Position::new(1.0, 50.0);
            positions[lower_idx] = Position::new(2.0, 50.0);
            // Wrapped bucket traversal sees this higher index before the
            // lower-index recipient in the giver's own bucket.
            positions[higher_idx] = Position::new(99.0, 50.0);
        }
        {
            let runtime = world.agent_runtime_mut(giver).unwrap();
            runtime.energy = 0.015;
            runtime.give_intent = 1.0;
        }
        world
            .agent_runtime_mut(lower_index_recipient)
            .unwrap()
            .energy = 0.0;
        world
            .agent_runtime_mut(higher_index_recipient)
            .unwrap()
            .energy = 0.0;

        world.stage_food();

        assert!(
            (world.agent_runtime(lower_index_recipient).unwrap().energy - 0.01).abs() < 1e-6,
            "lower dense index should receive the first full transfer"
        );
        assert!(
            (world.agent_runtime(higher_index_recipient).unwrap().energy - 0.005).abs() < 1e-6,
            "higher dense index should deterministically receive the remainder"
        );
    }

    #[test]
    fn direct_and_full_tick_food_sharing_agree() {
        struct SharingBrain {
            give: f32,
        }

        impl BrainRunner for SharingBrain {
            fn kind(&self) -> &'static str {
                "test.sharing"
            }

            fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
                let mut outputs = [0.0; OUTPUT_SIZE];
                outputs[8] = self.give;
                outputs
            }
        }

        let make_world = || {
            let config = ScriptBotsConfig {
                world_width: 100,
                world_height: 100,
                food_cell_size: 10,
                initial_food: 0.0,
                food_respawn_interval: 0,
                food_growth_rate: 0.0,
                food_decay_rate: 0.0,
                food_diffusion_rate: 0.0,
                food_intake_rate: 0.0,
                food_waste_rate: 0.0,
                food_transfer_rate: 0.01,
                food_sharing_distance: 5.0,
                bot_speed: 0.0,
                metabolism_drain: 0.0,
                movement_drain: 0.0,
                metabolism_ramp_rate: 0.0,
                metabolism_boost_penalty: 0.0,
                temperature_discomfort_rate: 0.0,
                reproduction_energy_threshold: 10.0,
                reproduction_attempt_chance: 0.0,
                spike_damage: 0.0,
                spike_energy_cost: 0.0,
                persistence_interval: 0,
                rng_seed: Some(205),
                ..ScriptBotsConfig::default()
            };
            let mut world = WorldState::new(config).expect("world");
            let giver = world.spawn_agent(sample_agent(0));
            let receiver = world.spawn_agent(sample_agent(1));
            {
                let arena = world.agents_mut();
                let giver_idx = arena.index_of(giver).unwrap();
                let receiver_idx = arena.index_of(receiver).unwrap();
                let positions = arena.columns_mut().positions_mut();
                positions[giver_idx] = Position::new(10.0, 10.0);
                positions[receiver_idx] = Position::new(12.0, 10.0);
            }
            {
                let runtime = world.agent_runtime_mut(giver).unwrap();
                runtime.energy = 1.0;
                runtime.give_intent = 1.0;
            }
            world.agent_runtime_mut(receiver).unwrap().energy = 0.5;

            let giver_key = world.brain_registry_mut().register("test.giver", |_rng| {
                Ok(Box::new(SharingBrain { give: 1.0 }))
            });
            let receiver_key = world
                .brain_registry_mut()
                .register("test.receiver", |_rng| {
                    Ok(Box::new(SharingBrain { give: 0.0 }))
                });
            assert!(
                world
                    .bind_agent_brain(giver, giver_key)
                    .expect("giver brain factory")
            );
            assert!(
                world
                    .bind_agent_brain(receiver, receiver_key)
                    .expect("receiver brain factory")
            );
            (world, giver, receiver)
        };

        let (mut direct, direct_giver, direct_receiver) = make_world();
        direct.stage_food();
        let direct_energies = (
            direct.agent_runtime(direct_giver).unwrap().energy,
            direct.agent_runtime(direct_receiver).unwrap().energy,
        );

        let (mut full_tick, full_giver, full_receiver) = make_world();
        let events = full_tick.step().expect("full tick");
        let full_tick_energies = (
            full_tick.agent_runtime(full_giver).unwrap().energy,
            full_tick.agent_runtime(full_receiver).unwrap().energy,
        );

        assert_eq!(events.tick, Tick(1));
        assert_eq!(full_tick_energies, direct_energies);
        assert_eq!(direct_energies, (0.99, 0.51));
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
        world.step().expect("open-world population seeding step");
        assert!(
            world.agent_count() >= 3,
            "expected minimum population seeding"
        );
    }

    #[test]
    fn closed_world_construction_preserves_floor_for_a_later_open_boundary() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            closed: true,
            population_minimum: 3,
            population_spawn_interval: 0,
            reproduction_energy_threshold: 10.0,
            rng_seed: Some(222),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        assert!(world.is_closed());
        assert!(world.config().closed);
        assert_eq!(world.config().population_minimum, 3);
        world.step().expect("closed-world population step");
        assert_eq!(
            world.agent_count(),
            0,
            "closed world should not seed agents"
        );

        world.set_closed(false);
        assert!(!world.is_closed());
        assert_eq!(world.config_revision(), 1);
        assert_eq!(
            world.config_audit(),
            [ConfigAuditEntry {
                tick: 1,
                patch: serde_json::json!({ "closed": false }),
            }]
        );
        world.step().expect("reopened population-floor step");
        assert_eq!(
            world.agent_count(),
            3,
            "reopening should restore the configured floor"
        );
    }

    #[test]
    fn closed_boundaries_skip_scheduled_injection_instead_of_queueing_it() {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            closed: false,
            population_minimum: 0,
            population_spawn_interval: 2,
            population_spawn_count: 1,
            population_crossover_chance: 0.0,
            reproduction_energy_threshold: 10.0,
            rng_seed: Some(223),
            ..ScriptBotsConfig::default()
        };

        let mut world = WorldState::new(config).expect("world");
        world.step().expect("open tick one");
        assert_eq!(world.agent_count(), 0);

        world.set_closed(true);
        world.set_closed(true);
        assert_eq!(
            world.config_revision(),
            1,
            "idempotent close must not create a transition"
        );
        world.step().expect("closed scheduled tick two");
        assert_eq!(
            world.agent_count(),
            0,
            "closed scheduled opportunity must be skipped"
        );

        world.set_closed(false);
        assert_eq!(world.config_revision(), 2);
        world.step().expect("open nonscheduled tick three");
        assert_eq!(
            world.agent_count(),
            0,
            "missed injection must not be queued for reopening"
        );
        world.step().expect("open scheduled tick four");
        assert_eq!(
            world.agent_count(),
            1,
            "next matching open cadence should inject once"
        );
        assert_eq!(
            world.config_audit(),
            [
                ConfigAuditEntry {
                    tick: 1,
                    patch: serde_json::json!({ "closed": true }),
                },
                ConfigAuditEntry {
                    tick: 2,
                    patch: serde_json::json!({ "closed": false }),
                },
            ]
        );
    }

    #[test]
    fn closed_world_preset_preserves_the_configured_open_world_policy() {
        let mut config = ScriptBotsConfig {
            closed: false,
            population_minimum: 7,
            population_spawn_interval: 13,
            population_spawn_count: 2,
            ..ScriptBotsConfig::default()
        };

        PresetKind::ClosedWorld.apply_to_config(&mut config);

        assert!(config.closed);
        assert_eq!(config.population_minimum, 7);
        assert_eq!(config.population_spawn_interval, 13);
        assert_eq!(config.population_spawn_count, 2);
        assert_eq!(
            PresetKind::ClosedWorld.patch(),
            serde_json::json!({ "closed": true })
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
        world.step().expect("first population interval step");
        assert_eq!(world.agent_count(), 0, "no spawn on first step");
        world.step().expect("second population interval step");
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

    #[test]
    #[should_panic(expected = "live dying agent must have stable identity")]
    fn death_cleanup_never_silently_drops_a_scientific_record() {
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(1234),
            ..ScriptBotsConfig::default()
        })
        .expect("world");
        let agent = world.spawn_agent(sample_agent(0));
        world.identities.remove(agent);
        world.pending_deaths.push(agent);

        world.stage_death_cleanup(Tick::zero());
    }

    fn characterization_world(seed: u64) -> (WorldState, Vec<AgentId>) {
        let config = ScriptBotsConfig {
            world_width: 40,
            world_height: 40,
            food_cell_size: 10,
            initial_food: 0.25,
            food_respawn_interval: 0,
            population_spawn_interval: 0,
            population_minimum: 0,
            rng_seed: Some(seed),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("characterization world");
        let ids = vec![
            world.spawn_agent(sample_agent(0)),
            world.spawn_agent(sample_agent(1)),
        ];
        (world, ids)
    }

    #[test]
    fn fnv1a64_v0_matches_reference_vectors_and_preserves_float_bits() {
        assert_eq!(characterization_fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(characterization_fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);

        let mut positive_zero = CharacterizationEncoderV0::new("float-reference");
        positive_zero.f32(0.0);
        let mut negative_zero = CharacterizationEncoderV0::new("float-reference");
        negative_zero.f32(-0.0);
        assert_ne!(positive_zero.finish(), negative_zero.finish());
    }

    #[test]
    fn characterization_v0_is_repeatable_and_does_not_advance_rng() {
        let (mut world_a, _) = characterization_world(0xC0FFEE);
        let (mut world_b, _) = characterization_world(0xC0FFEE);

        let first = world_a.characterization_digest_v0().expect("first digest");
        let second = world_a.characterization_digest_v0().expect("second digest");
        let peer = world_b.characterization_digest_v0().expect("peer digest");
        assert_eq!(first, second);
        assert_eq!(first, peer);
        assert_eq!(world_a.rng().next_u64(), world_b.rng().next_u64());
    }

    #[test]
    fn characterization_v0_components_detect_science_state_changes() {
        let (baseline_world, _) = characterization_world(42);
        let baseline = baseline_world
            .characterization_digest_v0()
            .expect("baseline digest");

        let (mut food_world, _) = characterization_world(42);
        food_world.food_mut().cells_mut()[0] = -0.0;
        let changed = food_world
            .characterization_digest_v0()
            .expect("food digest");
        assert_ne!(baseline.food, changed.food);
        assert_eq!(baseline.agents, changed.agents);
        assert_ne!(baseline.overall, changed.overall);

        let (mut terrain_world, _) = characterization_world(42);
        terrain_world.terrain.tiles[0].elevation = 0.123;
        let changed = terrain_world
            .characterization_digest_v0()
            .expect("terrain digest");
        assert_ne!(baseline.terrain, changed.terrain);
        assert_ne!(baseline.overall, changed.overall);

        let (mut agent_world, ids) = characterization_world(42);
        let index = agent_world.agents.index_of(ids[0]).expect("agent index");
        agent_world.agents.columns.health[index] = 0.75;
        let changed = agent_world
            .characterization_digest_v0()
            .expect("agent digest");
        assert_ne!(baseline.agents, changed.agents);
        assert_ne!(baseline.overall, changed.overall);

        let (mut runtime_world, ids) = characterization_world(42);
        runtime_world
            .agent_runtime_mut(ids[0])
            .expect("runtime")
            .energy = 0.125;
        let changed = runtime_world
            .characterization_digest_v0()
            .expect("runtime digest");
        assert_ne!(baseline.agents, changed.agents);
        assert_ne!(baseline.overall, changed.overall);

        let (mut closed_world, _) = characterization_world(42);
        closed_world.set_closed(true);
        let changed = closed_world
            .characterization_digest_v0()
            .expect("closed digest");
        assert_eq!(baseline.agents, changed.agents);
        assert_ne!(baseline.overall, changed.overall);

        let (mut tick_world, _) = characterization_world(42);
        tick_world.advance_tick();
        let changed = tick_world
            .characterization_digest_v0()
            .expect("tick digest");
        assert_ne!(baseline.overall, changed.overall);

        let (mut rng_world, _) = characterization_world(42);
        rng_world.rng().next_u64();
        let changed = rng_world.characterization_digest_v0().expect("rng digest");
        assert_ne!(baseline.rng_probe, changed.rng_probe);
        assert_ne!(baseline.overall, changed.overall);

        let (mut registry_world, _) = characterization_world(42);
        registry_world
            .brain_registry_mut()
            .register("stub", |_rng| Ok(Box::new(StubBrain)));
        let changed = registry_world
            .characterization_digest_v0()
            .expect("brain registry digest");
        assert_ne!(baseline.brain_registry, changed.brain_registry);
        assert_ne!(baseline.overall, changed.overall);
    }

    #[test]
    fn characterization_v0_excludes_selection_and_sorts_agent_handles() {
        let (mut world, ids) = characterization_world(9);
        let baseline = world.characterization_digest_v0().expect("baseline digest");
        world.agent_runtime_mut(ids[0]).expect("runtime").selection = SelectionState::Selected;
        let selected = world.characterization_digest_v0().expect("selected digest");
        assert_eq!(baseline, selected);

        world.agents.handles.swap(0, 1);
        let reordered = world
            .characterization_digest_v0()
            .expect("reordered digest");
        assert_eq!(selected, reordered);
    }

    #[test]
    fn characterization_v0_rejects_queued_control_work() {
        let (mut world, _) = characterization_world(7);
        world
            .enqueue_simulation_command(SimulationCommand {
                paused: Some(true),
                speed_multiplier: None,
                step_once: false,
            })
            .expect("valid simulation command");
        assert!(matches!(
            world.characterization_digest_v0(),
            Err(CharacterizationError::NonQuiescent {
                simulation_commands: 1,
                ..
            })
        ));
    }

    #[test]
    fn simulation_commands_queue_and_drain() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        assert!(world.drain_simulation_commands().is_empty());

        apply_control_command(
            &mut world,
            ControlCommand::UpdateSimulation(SimulationCommand {
                paused: Some(true),
                speed_multiplier: Some(0.0),
                step_once: false,
            }),
        )
        .expect("apply control command");

        let pending = world.drain_simulation_commands();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].paused, Some(true));
        assert_eq!(pending[0].speed_multiplier, Some(0.0));
        assert!(!pending[0].step_once);
        assert!(world.drain_simulation_commands().is_empty());
    }

    #[test]
    fn invalid_config_update_is_atomic() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let before_config = serde_json::to_value(world.config()).expect("serialize config");
        let before_food = world.food().cells().to_vec();
        let before_audit_len = world.config_audit().len();

        let mut invalid = world.config().clone();
        invalid.food_growth_rate = f32::NAN;
        let message = invalid_config_message(
            world
                .apply_config_update(invalid)
                .expect_err("non-finite runtime update must be rejected"),
        );
        assert_eq!(message, "food_growth_rate must be finite");
        assert_eq!(
            serde_json::to_value(world.config()).expect("serialize unchanged config"),
            before_config
        );
        assert_eq!(world.food().cells(), before_food);
        assert_eq!(world.config_audit().len(), before_audit_len);
    }

    #[test]
    fn simulation_command_rejects_non_finite_speed_without_queueing() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let command = SimulationCommand {
            paused: Some(false),
            speed_multiplier: Some(f32::NAN),
            step_once: false,
        };
        let message = invalid_config_message(
            world
                .enqueue_simulation_command(command)
                .expect_err("non-finite speed must be rejected"),
        );
        assert_eq!(message, "speed_multiplier must be finite");
        assert!(world.drain_simulation_commands().is_empty());
    }

    #[test]
    fn simulation_command_preserves_finite_clamp_semantics() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        world
            .enqueue_simulation_command(SimulationCommand {
                paused: Some(false),
                speed_multiplier: Some(128.0),
                step_once: false,
            })
            .expect("finite speed remains admissible");
        let pending = world.drain_simulation_commands();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].speed_multiplier, Some(32.0));
    }

    fn ledger_flow(world: &WorldState, kind: ResourceFlowKind) -> ResourceFlow {
        *world
            .resource_ledger()
            .latest
            .as_ref()
            .expect("resource ledger should contain a completed tick")
            .flows
            .iter()
            .find(|flow| flow.kind == kind)
            .expect("every stable resource category should be present")
    }

    fn assert_latest_ledger_reconciles(world: &WorldState) {
        let reconciliation = world
            .resource_ledger()
            .latest
            .as_ref()
            .expect("resource ledger should contain a completed tick")
            .reconciliation;
        assert!(
            reconciliation.reconciled,
            "unexplained resource delta {:?} exceeds tolerance {}",
            reconciliation.unexplained_delta, reconciliation.tolerance
        );
    }

    struct LedgerAggressorBrain;

    impl BrainRunner for LedgerAggressorBrain {
        fn kind(&self) -> &'static str {
            "test.resource-ledger-aggressor"
        }

        fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
            let mut outputs = [0.0; OUTPUT_SIZE];
            outputs[0] = 0.6;
            outputs[1] = 0.8;
            outputs[5] = 1.0;
            outputs[6] = 1.0;
            outputs[8] = 1.0;
            outputs
        }
    }

    struct LedgerIdleBrain;

    impl BrainRunner for LedgerIdleBrain {
        fn kind(&self) -> &'static str {
            "test.resource-ledger-idle"
        }

        fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
            [0.0; OUTPUT_SIZE]
        }
    }

    #[test]
    fn resource_ledger_attributes_ecology_combat_and_interventions() {
        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 10,
            initial_food: 0.95,
            food_max: 1.0,
            food_respawn_interval: 1,
            food_respawn_amount: 0.02,
            food_growth_rate: 0.01,
            food_decay_rate: 0.002,
            food_diffusion_rate: 0.05,
            food_intake_rate: 0.02,
            food_waste_rate: 0.02,
            food_transfer_rate: 0.04,
            food_sharing_distance: 20.0,
            bot_speed: 0.5,
            movement_drain: 0.01,
            metabolism_drain: 0.01,
            metabolism_ramp_floor: 0.2,
            metabolism_ramp_rate: 0.01,
            metabolism_boost_penalty: 0.02,
            temperature_discomfort_rate: 0.01,
            temperature_comfort_band: 0.0,
            temperature_discomfort_exponent: 1.0,
            aging_tick_interval: 1,
            aging_health_decay_start: 0,
            aging_health_decay_rate: 0.005,
            aging_health_decay_max: 0.02,
            aging_energy_penalty_rate: 0.5,
            reproduction_energy_threshold: 10.0,
            reproduction_attempt_chance: 0.0,
            spike_radius: 20.0,
            spike_damage: 0.5,
            spike_energy_cost: 0.02,
            spike_min_length: 0.1,
            spike_alignment_cosine: 0.1,
            spike_speed_damage_bonus: 0.0,
            spike_length_damage_bonus: 0.0,
            carcass_distribution_radius: 30.0,
            carcass_health_reward: 0.4,
            carcass_reproduction_reward: 0.0,
            carcass_neighbor_exponent: 1.0,
            carcass_maturity_age: 1,
            carcass_energy_share_rate: 0.5,
            population_minimum: 0,
            population_spawn_interval: 0,
            closed: true,
            persistence_interval: 0,
            rng_seed: Some(2_808),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("resource-ledger ecology world");
        let attacker = world.spawn_agent(sample_agent(0));
        let victim = world.spawn_agent(sample_agent(1));
        let attacker_brain = world
            .brain_registry_mut()
            .register("test.resource-ledger-aggressor", |_rng| {
                Ok(Box::new(LedgerAggressorBrain))
            });
        let idle_brain = world
            .brain_registry_mut()
            .register("test.resource-ledger-idle", |_rng| {
                Ok(Box::new(LedgerIdleBrain))
            });
        assert!(
            world
                .bind_agent_brain(attacker, attacker_brain)
                .expect("aggressor brain")
        );
        assert!(
            world
                .bind_agent_brain(victim, idle_brain)
                .expect("idle brain")
        );
        {
            let attacker_idx = world.agents.index_of(attacker).expect("attacker index");
            let victim_idx = world.agents.index_of(victim).expect("victim index");
            let columns = world.agents.columns_mut();
            columns.positions_mut()[attacker_idx] = Position::new(10.0, 10.0);
            columns.positions_mut()[victim_idx] = Position::new(15.0, 10.0);
            columns.headings_mut()[attacker_idx] = 0.0;
            columns.headings_mut()[victim_idx] = 0.0;
            columns.health_mut()[attacker_idx] = 1.0;
            columns.health_mut()[victim_idx] = 0.08;
            columns.ages_mut()[attacker_idx] = 5;
            columns.ages_mut()[victim_idx] = 5;
            columns.spike_lengths_mut()[attacker_idx] = 1.0;
        }
        {
            let runtime = world.agent_runtime_mut(attacker).expect("attacker runtime");
            runtime.energy = 1.0;
            runtime.herbivore_tendency = 0.0;
            runtime.temperature_preference = 0.0;
        }
        {
            let runtime = world.agent_runtime_mut(victim).expect("victim runtime");
            runtime.energy = 0.2;
            runtime.herbivore_tendency = 1.0;
            runtime.temperature_preference = 0.0;
        }
        world
            .enqueue_intervention(Intervention::Bloom {
                region: Region::All,
                amount: 0.2,
            })
            .expect("bounded bloom");
        world
            .enqueue_intervention(Intervention::Meteor {
                region: Region::All,
                lethality: 0.01,
                scorch: 0.1,
            })
            .expect("bounded meteor");
        world.set_resource_ledger_enabled(true);
        world.step().expect("resource-ledger ecology tick");

        assert_latest_ledger_reconciles(&world);
        for kind in [
            ResourceFlowKind::ScenarioIntervention,
            ResourceFlowKind::FoodDynamics,
            ResourceFlowKind::Aging,
            ResourceFlowKind::BasalMetabolism,
            ResourceFlowKind::Movement,
            ResourceFlowKind::MetabolismRamp,
            ResourceFlowKind::Boost,
            ResourceFlowKind::TemperatureStress,
            ResourceFlowKind::GroundFoodConversion,
            ResourceFlowKind::Combat,
            ResourceFlowKind::CarcassReward,
            ResourceFlowKind::DeathRemoval,
        ] {
            assert!(
                ledger_flow(&world, kind).delta.scale() > 0.0,
                "expected a non-zero {kind:?} attribution"
            );
        }
        assert!(
            ledger_flow(&world, ResourceFlowKind::EnergySharing)
                .activity
                .energy
                > 0.0,
            "giving must report its gross transfer despite zero net energy delta"
        );
        assert!(
            ledger_flow(&world, ResourceFlowKind::CapacityRejection)
                .activity
                .scale()
                > 0.0,
            "the capped bloom must report rejected source capacity"
        );
    }

    #[test]
    fn resource_ledger_reconciles_reproduction_and_population_injection() {
        let reproduction_config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_growth_rate: 0.0,
            food_decay_rate: 0.0,
            food_diffusion_rate: 0.0,
            food_intake_rate: 0.0,
            food_waste_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            metabolism_ramp_rate: 0.0,
            metabolism_boost_penalty: 0.0,
            temperature_discomfort_rate: 0.0,
            aging_health_decay_rate: 0.0,
            reproduction_energy_threshold: 0.5,
            reproduction_energy_cost: 0.2,
            reproduction_cooldown: 1,
            reproduction_attempt_interval: 1,
            reproduction_attempt_chance: 1.0,
            reproduction_rate_herbivore: 1.0,
            reproduction_rate_carnivore: 1.0,
            reproduction_child_energy: 0.4,
            reproduction_partner_chance: 0.0,
            population_minimum: 0,
            population_spawn_interval: 0,
            closed: true,
            persistence_interval: 0,
            rng_seed: Some(2_809),
            ..ScriptBotsConfig::default()
        };
        let mut reproduction_world =
            WorldState::new(reproduction_config).expect("reproduction ledger world");
        let parent = reproduction_world.spawn_agent(sample_agent(0));
        reproduction_world
            .agent_runtime_mut(parent)
            .expect("parent runtime")
            .energy = 1.0;
        reproduction_world.set_resource_ledger_enabled(true);
        reproduction_world
            .step()
            .expect("resource-ledger reproduction tick");
        assert_latest_ledger_reconciles(&reproduction_world);
        let reproduction = ledger_flow(
            &reproduction_world,
            ResourceFlowKind::ReproductionAllocation,
        );
        assert!(reproduction.delta.energy.abs() > 0.0);
        assert!(reproduction.delta.health > 0.0);

        let population_config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 10,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_growth_rate: 0.0,
            food_decay_rate: 0.0,
            food_diffusion_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            metabolism_ramp_rate: 0.0,
            temperature_discomfort_rate: 0.0,
            aging_health_decay_rate: 0.0,
            reproduction_energy_threshold: 10.0,
            reproduction_attempt_chance: 0.0,
            population_minimum: 2,
            population_spawn_interval: 0,
            closed: false,
            persistence_interval: 0,
            rng_seed: Some(2_810),
            ..ScriptBotsConfig::default()
        };
        let mut population_world =
            WorldState::new(population_config).expect("population ledger world");
        population_world.set_resource_ledger_enabled(true);
        population_world
            .step()
            .expect("resource-ledger population tick");
        assert_latest_ledger_reconciles(&population_world);
        let injection = ledger_flow(&population_world, ResourceFlowKind::PopulationInjection);
        assert_eq!(population_world.agent_count(), 2);
        assert!(injection.delta.health > 0.0);
        assert!(injection.delta.energy > 0.0);
    }

    #[test]
    fn enabling_resource_ledger_does_not_change_characterization_digest() {
        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 10,
            initial_food: 0.4,
            food_respawn_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 0,
            closed: true,
            persistence_interval: 0,
            rng_seed: Some(2_811),
            ..ScriptBotsConfig::default()
        };
        let mut uninstrumented = WorldState::new(config.clone()).expect("control world");
        let mut instrumented = WorldState::new(config).expect("instrumented world");
        instrumented.set_resource_ledger_enabled(true);
        let mut expected_cumulative: Vec<ResourceFlow> = RESOURCE_FLOW_KINDS
            .into_iter()
            .map(ResourceFlow::empty)
            .collect();

        for _ in 0..4 {
            uninstrumented.step().expect("control tick");
            instrumented.step().expect("instrumented tick");
            assert_latest_ledger_reconciles(&instrumented);
            for (expected, actual) in expected_cumulative.iter_mut().zip(
                &instrumented
                    .resource_ledger()
                    .latest
                    .as_ref()
                    .expect("instrumented tick report")
                    .flows,
            ) {
                expected.delta.add_assign(actual.delta);
                expected.activity.add_assign(actual.activity);
            }
            assert_eq!(
                uninstrumented
                    .characterization_digest_v0()
                    .expect("control digest"),
                instrumented
                    .characterization_digest_v0()
                    .expect("instrumented digest")
            );
        }
        assert_eq!(instrumented.resource_ledger().completed_ticks, 4);
        assert_eq!(
            instrumented.resource_ledger().cumulative,
            expected_cumulative
        );
        assert_eq!(uninstrumented.resource_ledger().completed_ticks, 0);
    }
}
