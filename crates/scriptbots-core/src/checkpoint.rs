//! Versioned science-state checkpoints.
//!
//! A checkpoint is deliberately narrower than a product "resume run" artifact. It restores every
//! checkpoint-owned value that can change the next core simulation transition, assuming
//! caller-supplied executable brain adapters implement the recorded family/schema/codec and
//! family-authored semantic-identity contracts. It does not claim to restore a persistence
//! session, retained analytics output,
//! configuration-audit revision, UI selection, mutation-log prose, chart history, narrative
//! history, or renderer state. Those host-owned surfaces belong to the later replay/runtime work.
//! Keeping that boundary explicit prevents a core round trip from being advertised as a storage
//! or application recovery feature.

// bd-tqpj: deterministic-simulation policy — pinned floating-point evaluation
// order and fixed-width casts are part of the science contract; fma fusion,
// reassociation, or width changes alter world digests. Function lengths mirror
// the legacy C++ parity layout and are reviewed as units.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
#![allow(clippy::float_cmp, clippy::while_float)]
#![allow(clippy::too_many_lines)]

use crate::rng_domains::{
    AgentRngCountersV1, AgentSubstreamProtocolError, AgentSubstreamProtocolV1,
    DomainStreamRestoreError, DomainStreams, DomainStreamsCheckpoint, RngDomain,
};
use crate::{
    ActiveEffect, ActiveEffectKind, AgentArena, AgentData, AgentId, AgentIdentity, AgentMap,
    AgentRngCounterStateV1, AgentRuntime, AgentUid, BirthOrigin, BirthRecord,
    BrainAdapterIdentityV1, BrainBinding, BrainEvaluatorStateEnvelope, BrainFamilyId,
    BrainGenomeEnvelope, BrainProtocolError, BrainRegistry, BrainRegistryDigestEntryV1,
    CharacterizationError, CombatEventFlags, FoodCellProfile, FoodGrid, Generation, HydrologyField,
    HydrologyFlowDirection, HydrologyState, HydrologyTile, HydrologyTileLayer, INPUT_SIZE,
    IndicatorState, MapArtifactMetadata, MapGeneratorKind, MutationRates, NUM_EYES, OUTPUT_SIZE,
    PersistenceBoundaryStatus, Position, Region, RngDomainDigestV1, RngDomainDigestsV1,
    ScientificStateError, ScriptBotsConfig, SelectionState, TerrainLayer, TerrainTile, Tick,
    TickCadence, TraitModifiers, WorldDigestV1, WorldDigestV1ContractError, WorldState,
    WorldStateError, brain_registry_digest_v1, clamp01, validate_finite, world_counters_digest_v1,
};
use scriptbots_index::UniformGridIndex;
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::marker::PhantomData;
use thiserror::Error;

/// Strict schema carried by the first world checkpoint envelope.
pub const WORLD_CHECKPOINT_V1_SCHEMA: &str = "scriptbots.world-checkpoint.v1.3";
/// Codec revision for [`WorldCheckpointV1`].
///
/// Any serialized field or variant layout, nested live DTO layout, or canonicalization change
/// requires a codec bump. A change to future-state coverage or field meaning requires a new
/// checkpoint schema. Never rebless the representative V1 wire golden without reviewing both
/// version identities.
pub const WORLD_CHECKPOINT_V1_CODEC_VERSION: u16 = 5;
const WORLD_CHECKPOINT_V1_CODEC: &str = "postcard+blake3-v5";

/// Maximum complete checkpoint wire accepted by the decoder.
///
/// Family-owned genome and evaluator payloads have smaller independent limits. This outer bound
/// prevents an imported checkpoint from requesting unbounded buffering before any semantic
/// validation can run.
pub const MAX_WORLD_CHECKPOINT_BYTES: usize = 128 * 1024 * 1024;
const MAX_WORLD_CHECKPOINT_PAYLOAD_BYTES: usize = MAX_WORLD_CHECKPOINT_BYTES - 1024;
const MAX_CHECKPOINT_AGENTS: usize = 1_000_000;
const MAX_CHECKPOINT_REGISTRY_ENTRIES: usize = 65_536;
const MAX_CHECKPOINT_GRID_CELLS: usize = 4_194_304;
const MAX_CHECKPOINT_ACTIVE_EFFECTS: usize = 65_536;
const MAX_CHECKPOINT_ORIGINS: usize = 1_000_000;
const MAX_CHECKPOINT_KIND_BYTES: usize = 128;
const MAX_CHECKPOINT_TILESET_ID_BYTES: usize = 1024;
const MAX_CHECKPOINT_DIGEST_STRING_BYTES: usize = 128;
const MAX_CHECKPOINT_UNCOVERED_FAMILIES: usize = MAX_CHECKPOINT_REGISTRY_ENTRIES;
const MAX_CHECKPOINT_EAGER_ALLOCATION_BYTES: usize = 1024 * 1024;

/// A validated, decoded V1 science checkpoint.
///
/// The inner state is private so callers cannot construct a partially validated checkpoint by
/// filling public fields. Use [`WorldState::checkpoint_v1`] to capture one or [`Self::decode`] to
/// import one. Restoring requires a caller-provided [`BrainRegistry`] because its adapters are
/// executable trait objects and must never be deserialized from data.
#[derive(Clone)]
pub struct WorldCheckpointV1 {
    state: WorldCheckpointStateV1,
}

/// Data-only registry roster a host must recreate before restoring a checkpoint.
///
/// This is an inspection aid, not executable code. The semantic identity is a family-authored
/// attestation rather than executable-byte authentication; the host still owns adapter
/// construction and the version discipline described by [`WorldState::restore_checkpoint_v1`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointBrainRegistryRequirementV1 {
    /// Stable registry key that must be recreated.
    pub key: u64,
    /// Human-facing family label registered at this key.
    pub kind: String,
    /// Declared legacy construction-state digest, when the source supplied one.
    pub factory_state_digest: Option<u64>,
    /// Family-authored executable semantic identity, when a protocol adapter is admitted.
    pub adapter_identity: Option<BrainAdapterIdentityV1>,
    /// Versioned protocol family attached at this key, when admitted.
    pub protocol_family: Option<BrainFamilyId>,
}

/// Complete declarative registry recipe required to restore a V1 checkpoint.
///
/// The allocation cursor is part of science state even when keys were retired: it determines the
/// key assigned to the next registered family. Therefore a host cannot reconstruct this recipe
/// from the surviving entries alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointBrainRegistryRequirementsV1 {
    /// Exact next key the prepared registry must allocate after restoration.
    pub next_key: u64,
    /// Surviving entries in stable key order, including any gaps left by retired keys.
    pub entries: Vec<CheckpointBrainRegistryRequirementV1>,
}

impl fmt::Debug for WorldCheckpointV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorldCheckpointV1")
            .field("tick", &self.state.tick)
            .field("epoch", &self.state.epoch)
            .field("agent_count", &self.state.agents.len())
            .field("source_digest", &self.state.source_digest.overall)
            .finish()
    }
}

/// Why a checkpoint could not be captured, decoded, validated, or restored.
#[derive(Debug, Error)]
pub enum WorldCheckpointError {
    /// The complete wire or declared payload exceeds its fixed resource ceiling.
    #[error("checkpoint wire is {found} bytes; maximum is {maximum}")]
    WireTooLarge {
        /// Size in bytes of the offending wire or declared payload.
        found: usize,
        /// Fixed resource ceiling in bytes that `found` exceeded.
        maximum: usize,
    },
    /// Postcard could not encode, decode, or canonicalize one layer.
    #[error("checkpoint {operation} failed: {detail}")]
    Codec {
        /// Which codec stage failed (e.g. encode, decode, canonicalize).
        operation: &'static str,
        /// Human-readable failure detail from Postcard.
        detail: String,
    },
    /// The envelope names a foreign schema.
    #[error("checkpoint schema `{found}` does not match `{expected}`")]
    Schema {
        /// Schema identifier found in the envelope.
        found: String,
        /// Schema identifier this build expects.
        expected: &'static str,
    },
    /// The envelope names a foreign codec revision.
    #[error("checkpoint codec version {found} does not match {expected}")]
    CodecVersion {
        /// Codec revision found in the envelope.
        found: u16,
        /// Codec revision this build expects.
        expected: u16,
    },
    /// The envelope names a foreign serialization/checksum combination.
    #[error("checkpoint codec `{found}` does not match `{expected}`")]
    CodecIdentity {
        /// Serialization/checksum combination found in the envelope.
        found: String,
        /// Serialization/checksum combination this build expects.
        expected: &'static str,
    },
    /// The payload does not match the unkeyed corruption checksum in its envelope.
    #[error("checkpoint payload BLAKE3 does not match the envelope")]
    PayloadHashMismatch,
    /// A decoded Postcard value left extra bytes unconsumed.
    #[error("checkpoint {layer} contains {count} trailing bytes")]
    TrailingBytes {
        /// Which checkpoint layer (envelope or payload) held the extra bytes.
        layer: &'static str,
        /// Number of unconsumed trailing bytes.
        count: usize,
    },
    /// A semantically decodable Postcard value used a noncanonical representation.
    #[error("checkpoint {layer} is not in canonical Postcard form")]
    NonCanonical {
        /// Which checkpoint layer (envelope or payload) was noncanonical.
        layer: &'static str,
    },
    /// A decoded value violates a checkpoint-local semantic invariant.
    #[error("checkpoint contract violation at `{path}`: {detail}")]
    Contract {
        /// Dotted path to the value that violated the invariant.
        path: String,
        /// Human-readable description of the violated invariant.
        detail: String,
    },
    /// Core checkpoint capture does not own or reconstruct persistence sessions.
    #[error(
        "checkpoint capture requires persistence_interval=0; found {persistence_interval} (product persistence/resume belongs to the later runtime replay layer)"
    )]
    PersistenceEnabled {
        /// The configured persistence interval that must be zero for core capture.
        persistence_interval: u32,
    },
    /// Capture was attempted outside an open completed persistence boundary.
    #[error(
        "checkpoint capture requires an open persistence boundary at tick {tick}; found {found:?}"
    )]
    PersistenceBoundary {
        /// Tick at which capture was attempted.
        tick: u64,
        /// Actual boundary status observed at that tick.
        found: PersistenceBoundaryStatus,
    },
    /// Host-owned deferred output remained at the requested capture boundary.
    #[error("checkpoint capture found deferred host/persistence output at `{field}`")]
    DeferredHostOutput {
        /// Name of the state field still holding deferred host/persistence output.
        field: &'static str,
    },
    /// A live legacy runner has no versioned genome/evaluator reconstruction protocol.
    #[error(
        "agent UID {agent_uid} has legacy brain `{kind}` at registry key {registry_key:?}; only versioned protocol brains are restorable"
    )]
    LegacyBrain {
        /// UID of the agent whose brain is not restorable.
        agent_uid: u64,
        /// Legacy brain kind string.
        kind: String,
        /// Registry key the legacy brain occupies, if any.
        registry_key: Option<u64>,
    },
    /// A versioned brain family rejected capture or reconstruction.
    #[error("agent UID {agent_uid} brain `{kind}` could not be checkpointed or restored: {detail}")]
    Brain {
        /// UID of the agent whose brain failed capture or reconstruction.
        agent_uid: u64,
        /// Versioned brain kind string.
        kind: String,
        /// Human-readable failure detail from the brain family.
        detail: String,
    },
    /// The trusted host prepared a registry roster different from the saved declaration.
    #[error("prepared brain registry does not match the checkpoint: {detail}")]
    RegistryMismatch {
        /// Human-readable description of how the prepared roster diverges.
        detail: String,
    },
    /// Reconstructed science state differs from the source boundary digest.
    #[error(
        "restored world diverged in `{lane}`: checkpoint recorded `{expected}`, reconstructed `{actual}`"
    )]
    DigestMismatch {
        /// Which digest lane (state domain) diverged.
        lane: &'static str,
        /// Digest recorded in the checkpoint.
        expected: String,
        /// Digest reconstructed from restored state.
        actual: String,
    },
    /// A nested legacy characterization contract failed validation.
    #[error(transparent)]
    Characterization(#[from] CharacterizationError),
    /// A world/configuration constructor rejected checkpoint state.
    #[error(transparent)]
    World(#[from] WorldStateError),
    /// A scientific-state invariant rejected checkpoint state.
    #[error(transparent)]
    ScientificState(#[from] ScientificStateError),
    /// A generic genome/evaluator envelope violated its protocol.
    #[error(transparent)]
    BrainProtocol(#[from] BrainProtocolError),
    /// The six-domain random-stream checkpoint could not be restored.
    #[error(transparent)]
    RandomStreams(#[from] DomainStreamRestoreError),
    /// The agent-keyed random-substream protocol is foreign or bound to another root.
    #[error(transparent)]
    AgentSubstreamProtocol(#[from] AgentSubstreamProtocolError),
    /// The embedded V1.6 source digest violated its own contract.
    #[error(transparent)]
    DigestContract(#[from] WorldDigestV1ContractError),
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldCheckpointWireV1 {
    schema: String,
    codec_version: u16,
    codec: String,
    #[serde(deserialize_with = "deserialize_checkpoint_payload")]
    payload: Vec<u8>,
    payload_blake3: [u8; 32],
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldCheckpointStateV1 {
    #[serde(deserialize_with = "deserialize_checkpoint_world_digest")]
    source_digest: WorldDigestV1,
    config: ScriptBotsConfig,
    tick: Tick,
    epoch: u64,
    random_streams: DomainStreamsCheckpoint,
    agent_substream_protocol: AgentSubstreamProtocolV1,
    next_agent_uid: u64,
    next_spawn_ordinal: u64,
    next_birth_ordinal: u64,
    registry: BrainRegistryCheckpointV1,
    #[serde(deserialize_with = "deserialize_checkpoint_agents")]
    agents: Vec<AgentCheckpointV1>,
    food: FoodCheckpointV1,
    terrain: TerrainCheckpointV1,
    map_metadata: Option<MapMetadataCheckpointV1>,
    hydrology: Option<HydrologyCheckpointV1>,
    #[serde(deserialize_with = "deserialize_checkpoint_active_effects")]
    active_effects: Vec<ActiveEffectCheckpointV1>,
    #[serde(deserialize_with = "deserialize_checkpoint_origins")]
    pending_birth_records: Vec<BirthRecordCheckpointV1>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldDigestCheckpointDecodeV1 {
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    schema: String,
    codec_version: u16,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    algorithm: String,
    tick: Tick,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    overall: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    agents: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    brains: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    food: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    terrain: String,
    #[serde(deserialize_with = "deserialize_optional_checkpoint_digest_string")]
    hydrology: Option<String>,
    rng: RngDomainDigestCheckpointDecodeV1,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    counters: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    brain_registry: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    config: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    effects: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    derived_transition: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    origins: String,
    evaluator_state_covered: bool,
    #[serde(deserialize_with = "deserialize_checkpoint_uncovered_families")]
    uncovered_families: Vec<CheckpointFamilyName>,
    factory_state_covered: bool,
    #[serde(deserialize_with = "deserialize_checkpoint_uncovered_families")]
    uncovered_factory_families: Vec<CheckpointFamilyName>,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    agent_identity: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RngDomainDigestCheckpointDecodeV1 {
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    overall: String,
    domains: RngDomainDigestsCheckpointDecodeV1,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RngDomainDigestsCheckpointDecodeV1 {
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    environment: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    food: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    population: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    lineage: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    mutation: String,
    #[serde(deserialize_with = "deserialize_checkpoint_digest_string")]
    crossover: String,
}

#[derive(Deserialize)]
struct CheckpointFamilyName(
    #[serde(deserialize_with = "deserialize_checkpoint_family_name")] String,
);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct BrainRegistryCheckpointV1 {
    next_key: u64,
    #[serde(deserialize_with = "deserialize_checkpoint_registry_entries")]
    entries: Vec<BrainRegistryEntryCheckpointV1>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct BrainRegistryEntryCheckpointV1 {
    key: u64,
    #[serde(deserialize_with = "deserialize_checkpoint_kind")]
    kind: String,
    factory_state_digest: Option<u64>,
    adapter_identity: Option<BrainAdapterIdentityV1>,
    protocol_family: Option<BrainFamilyId>,
}

impl BrainRegistryCheckpointV1 {
    fn digest_v1(&self) -> String {
        brain_registry_digest_v1(
            self.next_key,
            self.entries.iter().map(|entry| BrainRegistryDigestEntryV1 {
                key: entry.key,
                kind: &entry.kind,
                factory_state_digest: entry.factory_state_digest,
                protocol_family: entry.protocol_family.as_ref().map(BrainFamilyId::as_str),
                adapter_identity: entry.adapter_identity,
            }),
        )
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentCheckpointV1 {
    identity: AgentIdentity,
    rng_counters: AgentRngCountersV1,
    data: AgentData,
    runtime: AgentRuntimeCheckpointV1,
    brain: AgentBrainCheckpointV1,
}

#[derive(Clone, Serialize, Deserialize)]
// Boxing the protocol variant would add one allocation and indirection per agent; both
// family-owned payloads already have strict deserialize-time byte bounds.
#[allow(clippy::large_enum_variant)]
enum AgentBrainCheckpointV1 {
    Unbound,
    Protocol {
        registry_key: u64,
        #[serde(deserialize_with = "deserialize_checkpoint_kind")]
        kind: String,
        genome: BrainGenomeEnvelope,
        evaluator_state: BrainEvaluatorStateEnvelope,
    },
}

/// Science-transition fields from [`AgentRuntime`].
///
/// UI selection/indicator state, per-tick combat presentation flags, and mutation-log prose are
/// intentionally reset on restore. Every valid transition clears combat flags before they can
/// affect death classification. These fields do not participate in the future science state or
/// [`WorldDigestV1`], and pretending otherwise would turn this core envelope into an incomplete
/// application-session format.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentRuntimeCheckpointV1 {
    energy: f32,
    reproduction_counter: f32,
    herbivore_tendency: f32,
    mutation_rates: MutationRates,
    trait_modifiers: TraitModifiers,
    clocks: [f32; 2],
    eye_fov: [f32; NUM_EYES],
    eye_direction: [f32; NUM_EYES],
    sound_multiplier: f32,
    give_intent: f32,
    sensors: [f32; INPUT_SIZE],
    outputs: [f32; OUTPUT_SIZE],
    food_delta: f32,
    spiked: bool,
    hybrid: bool,
    sound_output: f32,
    temperature_preference: f32,
    lineage: [Option<AgentUid>; 2],
    food_balance_total: f32,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FoodCheckpointV1 {
    width: u32,
    height: u32,
    #[serde(deserialize_with = "deserialize_checkpoint_f32_cells")]
    cells: Vec<f32>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TerrainCheckpointV1 {
    width: u32,
    height: u32,
    cell_size: u32,
    #[serde(deserialize_with = "deserialize_checkpoint_terrain_tiles")]
    tiles: Vec<TerrainTile>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MapMetadataCheckpointV1 {
    generator: MapGeneratorKind,
    #[serde(deserialize_with = "deserialize_checkpoint_tileset_id")]
    tileset_id: String,
    tileset_hash: u64,
    seed: u64,
    width: u32,
    height: u32,
    attempt_count: usize,
    succeeded_on: usize,
    generated_at_epoch_ms: u128,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HydrologyCheckpointV1 {
    width: u32,
    height: u32,
    #[serde(deserialize_with = "deserialize_checkpoint_hydrology_tiles")]
    tiles: Vec<HydrologyTile>,
    #[serde(deserialize_with = "deserialize_checkpoint_flow_directions")]
    flow_directions: Vec<HydrologyFlowDirection>,
    #[serde(deserialize_with = "deserialize_checkpoint_f32_cells")]
    accumulation: Vec<f32>,
    #[serde(deserialize_with = "deserialize_checkpoint_f32_cells")]
    spill_elevation: Vec<f32>,
    #[serde(deserialize_with = "deserialize_checkpoint_u32_cells")]
    basin_ids: Vec<u32>,
    #[serde(deserialize_with = "deserialize_checkpoint_f32_cells")]
    initial_water_depth: Vec<f32>,
    #[serde(deserialize_with = "deserialize_checkpoint_f32_cells")]
    water_depth: Vec<f32>,
}

/// Checkpoint-local region wire.
///
/// The public [`Region`] representation is internally tagged for the JSON control API. Postcard
/// deliberately cannot deserialize internally tagged enums because they require
/// `deserialize_any`, so the checkpoint owns an externally tagged wire instead of leaking that
/// host-facing representation into the binary science contract.
#[derive(Clone, Copy, Serialize, Deserialize)]
enum RegionCheckpointV1 {
    All,
    Disc { x: f32, y: f32, radius: f32 },
    Rect { x: f32, y: f32, w: f32, h: f32 },
}

#[derive(Clone, Copy, Serialize, Deserialize)]
enum ActiveEffectKindCheckpointV1 {
    GrowthScale(f32),
    Embargo,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActiveEffectCheckpointV1 {
    region: RegionCheckpointV1,
    ticks_remaining: u32,
    kind: ActiveEffectKindCheckpointV1,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BirthRecordCheckpointV1 {
    tick: Tick,
    agent_uid: AgentUid,
    spawn_ordinal: u64,
    birth_ordinal: Option<u64>,
    origin: BirthOrigin,
    parent_a: Option<AgentUid>,
    parent_b: Option<AgentUid>,
    #[serde(deserialize_with = "deserialize_optional_checkpoint_kind")]
    brain_kind: Option<String>,
    brain_key: Option<u64>,
    herbivore_tendency: f32,
    generation: Generation,
    position: Position,
    is_hybrid: bool,
}

impl WorldCheckpointV1 {
    /// Serialize this checkpoint into its strict, integrity-bound V1 envelope.
    pub fn encode(&self) -> Result<Vec<u8>, WorldCheckpointError> {
        self.validate_contract()?;
        let payload =
            postcard::to_allocvec(&self.state).map_err(|error| WorldCheckpointError::Codec {
                operation: "payload encoding",
                detail: error.to_string(),
            })?;
        if payload.len() > MAX_WORLD_CHECKPOINT_PAYLOAD_BYTES {
            return Err(WorldCheckpointError::WireTooLarge {
                found: payload.len(),
                maximum: MAX_WORLD_CHECKPOINT_PAYLOAD_BYTES,
            });
        }
        let wire = WorldCheckpointWireV1 {
            schema: WORLD_CHECKPOINT_V1_SCHEMA.to_owned(),
            codec_version: WORLD_CHECKPOINT_V1_CODEC_VERSION,
            codec: WORLD_CHECKPOINT_V1_CODEC.to_owned(),
            payload_blake3: *blake3::hash(&payload).as_bytes(),
            payload,
        };
        let encoded =
            postcard::to_allocvec(&wire).map_err(|error| WorldCheckpointError::Codec {
                operation: "envelope encoding",
                detail: error.to_string(),
            })?;
        if encoded.len() > MAX_WORLD_CHECKPOINT_BYTES {
            return Err(WorldCheckpointError::WireTooLarge {
                found: encoded.len(),
                maximum: MAX_WORLD_CHECKPOINT_BYTES,
            });
        }
        Ok(encoded)
    }

    /// Decode, verify checksum integrity and canonical form, and validate the adapter-independent
    /// semantics of a V1 checkpoint envelope. Family-specific semantics are checked during
    /// [`WorldState::restore_checkpoint_v1`] after the caller supplies executable adapters.
    pub fn decode(encoded: &[u8]) -> Result<Self, WorldCheckpointError> {
        if encoded.len() > MAX_WORLD_CHECKPOINT_BYTES {
            return Err(WorldCheckpointError::WireTooLarge {
                found: encoded.len(),
                maximum: MAX_WORLD_CHECKPOINT_BYTES,
            });
        }
        let (wire, trailing): (WorldCheckpointWireV1, &[u8]) =
            postcard::take_from_bytes(encoded).map_err(|error| WorldCheckpointError::Codec {
                operation: "envelope decoding",
                detail: error.to_string(),
            })?;
        if !trailing.is_empty() {
            return Err(WorldCheckpointError::TrailingBytes {
                layer: "envelope",
                count: trailing.len(),
            });
        }
        if wire.schema != WORLD_CHECKPOINT_V1_SCHEMA {
            return Err(WorldCheckpointError::Schema {
                found: wire.schema,
                expected: WORLD_CHECKPOINT_V1_SCHEMA,
            });
        }
        if wire.codec_version != WORLD_CHECKPOINT_V1_CODEC_VERSION {
            return Err(WorldCheckpointError::CodecVersion {
                found: wire.codec_version,
                expected: WORLD_CHECKPOINT_V1_CODEC_VERSION,
            });
        }
        if wire.codec != WORLD_CHECKPOINT_V1_CODEC {
            return Err(WorldCheckpointError::CodecIdentity {
                found: wire.codec,
                expected: WORLD_CHECKPOINT_V1_CODEC,
            });
        }
        if *blake3::hash(&wire.payload).as_bytes() != wire.payload_blake3 {
            return Err(WorldCheckpointError::PayloadHashMismatch);
        }
        let canonical_wire =
            postcard::to_allocvec(&wire).map_err(|error| WorldCheckpointError::Codec {
                operation: "envelope canonicalization",
                detail: error.to_string(),
            })?;
        if canonical_wire != encoded {
            return Err(WorldCheckpointError::NonCanonical { layer: "envelope" });
        }
        drop(canonical_wire);
        let (state, trailing): (WorldCheckpointStateV1, &[u8]) =
            postcard::take_from_bytes(&wire.payload).map_err(|error| {
                WorldCheckpointError::Codec {
                    operation: "payload decoding",
                    detail: error.to_string(),
                }
            })?;
        if !trailing.is_empty() {
            return Err(WorldCheckpointError::TrailingBytes {
                layer: "payload",
                count: trailing.len(),
            });
        }
        let canonical_payload =
            postcard::to_allocvec(&state).map_err(|error| WorldCheckpointError::Codec {
                operation: "payload canonicalization",
                detail: error.to_string(),
            })?;
        if canonical_payload != wire.payload {
            return Err(WorldCheckpointError::NonCanonical { layer: "payload" });
        }
        drop(canonical_payload);
        drop(wire);
        let checkpoint = Self { state };
        checkpoint.validate_contract()?;
        Ok(checkpoint)
    }

    /// Digest captured at the save boundary and rechecked after restoration.
    #[must_use]
    pub const fn source_digest(&self) -> &WorldDigestV1 {
        &self.state.source_digest
    }

    /// Completed simulation tick carried by this checkpoint.
    #[must_use]
    pub const fn tick(&self) -> Tick {
        self.state.tick
    }

    /// Number of live stable agents carried by this checkpoint.
    #[must_use]
    pub const fn agent_count(&self) -> usize {
        self.state.agents.len()
    }

    /// Validated simulation configuration carried by this checkpoint.
    #[must_use]
    pub const fn config(&self) -> &ScriptBotsConfig {
        &self.state.config
    }

    /// Complete data-only registry recipe the host must reproduce before restoration.
    #[must_use]
    pub fn required_brain_registry(&self) -> CheckpointBrainRegistryRequirementsV1 {
        CheckpointBrainRegistryRequirementsV1 {
            next_key: self.state.registry.next_key,
            entries: self
                .state
                .registry
                .entries
                .iter()
                .map(|entry| CheckpointBrainRegistryRequirementV1 {
                    key: entry.key,
                    kind: entry.kind.clone(),
                    factory_state_digest: entry.factory_state_digest,
                    adapter_identity: entry.adapter_identity,
                    protocol_family: entry.protocol_family.clone(),
                })
                .collect(),
        }
    }

    fn validate_contract(&self) -> Result<(), WorldCheckpointError> {
        let state = &self.state;
        state.source_digest.validate_contract()?;
        if state.source_digest.tick != state.tick {
            return contract_error(
                "source_digest.tick",
                format!(
                    "digest tick {} does not match checkpoint tick {}",
                    state.source_digest.tick.0, state.tick.0
                ),
            );
        }
        state.config.validate()?;
        if state.config.persistence_interval != 0 {
            return Err(WorldCheckpointError::PersistenceEnabled {
                persistence_interval: state.config.persistence_interval,
            });
        }
        state
            .agent_substream_protocol
            .validate(state.random_streams.root_seed)?;
        ensure_count_bound("agents", state.agents.len(), MAX_CHECKPOINT_AGENTS)?;
        ensure_count_bound(
            "registry.entries",
            state.registry.entries.len(),
            MAX_CHECKPOINT_REGISTRY_ENTRIES,
        )?;
        ensure_count_bound(
            "active_effects",
            state.active_effects.len(),
            MAX_CHECKPOINT_ACTIVE_EFFECTS,
        )?;
        ensure_count_bound(
            "pending_birth_records",
            state.pending_birth_records.len(),
            MAX_CHECKPOINT_ORIGINS,
        )?;
        validate_registry_checkpoint(&state.registry)?;
        let registry_digest = state.registry.digest_v1();
        if registry_digest != state.source_digest.brain_registry {
            return contract_error(
                "source_digest.brain_registry",
                format!(
                    "digest records `{}`, but checkpoint registry recomputes to `{registry_digest}`",
                    state.source_digest.brain_registry
                ),
            );
        }
        let has_legacy_registry_entry = state
            .registry
            .entries
            .iter()
            .any(|entry| entry.protocol_family.is_none());
        let has_protocol_registry_entry = state
            .registry
            .entries
            .iter()
            .any(|entry| entry.protocol_family.is_some());
        if has_legacy_registry_entry
            && !has_protocol_registry_entry
            && (state.config.population_minimum != 0 || state.config.population_spawn_interval != 0)
        {
            return contract_error(
                "registry.entries",
                "a legacy-only registry is not checkpointable while automatic population construction is enabled",
            );
        }
        validate_environment_checkpoint(state)?;
        validate_agent_checkpoint(state)?;
        validate_agent_rng_checkpoint(state)?;
        validate_origin_checkpoint(state)?;
        for (index, effect) in state.active_effects.iter().enumerate() {
            effect.region.restore().validate_basic().map_err(|error| {
                WorldCheckpointError::Contract {
                    path: format!("active_effects[{index}].region"),
                    detail: error.to_string(),
                }
            })?;
            if effect.ticks_remaining == 0 {
                return contract_error(
                    format!("active_effects[{index}].ticks_remaining"),
                    "active effects must retain at least one tick",
                );
            }
            match effect.kind {
                ActiveEffectKindCheckpointV1::GrowthScale(growth_scale)
                    if !(0.0..=1.0).contains(&growth_scale) =>
                {
                    return contract_error(
                        format!("active_effects[{index}].kind.growth_scale"),
                        "growth scale must be finite and lie in [0, 1]",
                    );
                }
                ActiveEffectKindCheckpointV1::Embargo
                | ActiveEffectKindCheckpointV1::GrowthScale(_) => {}
            }
        }
        Ok(())
    }
}

impl WorldState {
    /// Capture a strict V1 checkpoint at a quiescent, persistence-disabled science boundary.
    ///
    /// Bound legacy runners are rejected because they have no versioned genome/evaluator-state
    /// reconstruction contract. Registered but unused legacy families may remain in a mixed
    /// roster because automatic selection becomes protocol-only once an admitted family exists.
    /// A legacy-only roster is rejected whenever automatic population construction is enabled.
    pub fn checkpoint_v1(&self) -> Result<WorldCheckpointV1, WorldCheckpointError> {
        if self.config.persistence_interval != 0 {
            return Err(WorldCheckpointError::PersistenceEnabled {
                persistence_interval: self.config.persistence_interval,
            });
        }
        let expected_boundary = PersistenceBoundaryStatus::Open { tick: self.tick };
        if self.persistence_boundary != expected_boundary {
            return Err(WorldCheckpointError::PersistenceBoundary {
                tick: self.tick.0,
                found: self.persistence_boundary,
            });
        }
        self.ensure_checkpoint_has_no_deferred_host_output()?;
        let source_digest = self.world_digest_v1()?;

        let mut ordered = self
            .agents
            .iter_handles()
            .map(|id| {
                let identity = self.identities.get(id).copied().ok_or_else(|| {
                    WorldCheckpointError::Contract {
                        path: format!("agents[{}].identity", id.raw()),
                        detail: "live arena handle has no stable identity".to_owned(),
                    }
                })?;
                Ok((identity.uid, id))
            })
            .collect::<Result<Vec<_>, WorldCheckpointError>>()?;
        ordered.sort_unstable_by_key(|(uid, _)| *uid);

        let agents = ordered
            .into_iter()
            .map(|(uid, id)| self.capture_checkpoint_agent(uid, id))
            .collect::<Result<Vec<_>, _>>()?;
        let state = WorldCheckpointStateV1 {
            source_digest,
            config: self.config.clone(),
            tick: self.tick,
            epoch: self.epoch,
            random_streams: self.rng.checkpoint(),
            agent_substream_protocol: self.agent_substream_protocol_v1(),
            next_agent_uid: self.next_agent_uid,
            next_spawn_ordinal: self.next_spawn_ordinal,
            next_birth_ordinal: self.next_birth_ordinal,
            registry: self.brain_registry.checkpoint_v1(),
            agents,
            food: FoodCheckpointV1::capture(&self.food),
            terrain: TerrainCheckpointV1::capture(&self.terrain),
            map_metadata: self
                .map_metadata
                .as_ref()
                .map(MapMetadataCheckpointV1::capture),
            hydrology: self.hydrology.as_ref().map(HydrologyCheckpointV1::capture),
            active_effects: self
                .active_effects
                .iter()
                .copied()
                .map(ActiveEffectCheckpointV1::capture)
                .collect(),
            pending_birth_records: self
                .pending_birth_records
                .iter()
                .map(BirthRecordCheckpointV1::capture)
                .collect(),
        };
        let checkpoint = WorldCheckpointV1 { state };
        checkpoint.validate_contract()?;
        Ok(checkpoint)
    }

    /// Reconstruct a live world from a V1 checkpoint and a freshly prepared exact registry.
    ///
    /// `brain_registry` must reproduce the complete recipe returned by
    /// [`WorldCheckpointV1::required_brain_registry`]: exact allocation cursor, retired-key gaps,
    /// surviving key order, kinds, protocol families, legacy/protocol classification, legacy
    /// factory state declarations, and protocol adapter semantic identities. The registry itself
    /// is trusted host code and is never accepted from checkpoint bytes. Adapter identity is a
    /// family-authored semantic attestation rather than executable-byte authentication: authors
    /// must change it whenever construction or evaluation behavior changes, and must additionally
    /// bump their family schema/codec whenever serialized payload interpretation changes.
    pub fn restore_checkpoint_v1(
        checkpoint: &WorldCheckpointV1,
        brain_registry: BrainRegistry,
    ) -> Result<Self, WorldCheckpointError> {
        checkpoint.validate_contract()?;
        let state = &checkpoint.state;
        let prepared_registry = brain_registry.checkpoint_v1();
        if prepared_registry != state.registry {
            return Err(WorldCheckpointError::RegistryMismatch {
                detail: registry_mismatch_detail(&state.registry, &prepared_registry),
            });
        }

        let random_streams = DomainStreams::restore(&state.random_streams)?;
        let food = state.food.restore()?;
        let terrain = state.terrain.restore()?;
        let hydrology = state
            .hydrology
            .as_ref()
            .map(HydrologyCheckpointV1::restore)
            .transpose()?;
        let map_metadata = state
            .map_metadata
            .as_ref()
            .map(MapMetadataCheckpointV1::restore);

        // Build only a one-cell scratch shell. The decoded environment has already been bounded
        // and constructed above; generating a second full terrain would double peak memory, and
        // preallocating an imported host-history capacity would let a tiny checkpoint request an
        // enormous allocation even though history is intentionally outside this format.
        let mut construction_config = state.config.clone();
        construction_config.world_width = construction_config.food_cell_size;
        construction_config.world_height = construction_config.food_cell_size;
        construction_config.history_capacity = 1;
        construction_config.rng_seed = Some(0);
        let mut restored = Self::build(construction_config)?;
        restored.config.clone_from(&state.config);
        restored.tick = state.tick;
        restored.epoch = state.epoch;
        restored.rng = random_streams;
        restored.next_agent_uid = state.next_agent_uid;
        restored.next_spawn_ordinal = state.next_spawn_ordinal;
        restored.next_birth_ordinal = state.next_birth_ordinal;
        restored.food = food;
        restored.terrain = terrain;
        restored.map_metadata = map_metadata;
        restored.hydrology = hydrology;
        restored.index = UniformGridIndex::new(
            restored.config.food_cell_size as f32,
            restored.config.world_width as f32,
            restored.config.world_height as f32,
        );
        restored.food_profiles = FoodCellProfile::compute(&restored.config, &restored.terrain);
        restored.food_scratch = vec![0.0; restored.food.cells().len()];
        restored.cadence = TickCadence::from_config(&restored.config);
        restored.brain_registry = brain_registry;
        restored.agents = AgentArena::with_capacity(state.agents.len());
        restored.identities = AgentMap::new();
        restored.agent_rng_counters = AgentMap::new();
        restored.runtime = AgentMap::new();

        for saved in &state.agents {
            let brain = match &saved.brain {
                AgentBrainCheckpointV1::Unbound => BrainBinding::Unbound,
                AgentBrainCheckpointV1::Protocol {
                    registry_key,
                    kind,
                    genome,
                    evaluator_state,
                } => {
                    let adapter = restored
                        .brain_registry
                        .family(*registry_key)
                        .ok_or_else(|| WorldCheckpointError::RegistryMismatch {
                            detail: format!(
                                "agent UID {} requires protocol adapter `{}` at key {}, but the prepared registry has none",
                                saved.identity.uid.0,
                                genome.family_id(),
                                registry_key
                            ),
                        })?;
                    Self::instantiate_protocol_binding(
                        adapter,
                        *registry_key,
                        kind.clone(),
                        genome.clone(),
                        &evaluator_state,
                    )
                    .map_err(|error| WorldCheckpointError::Brain {
                        agent_uid: saved.identity.uid.0,
                        kind: kind.clone(),
                        detail: error.to_string(),
                    })?
                }
            };
            let runtime = saved.runtime.restore(brain);
            runtime.validate_at(&format!("agents[uid={}].runtime", saved.identity.uid.0))?;
            let id = restored.agents.try_insert(saved.data)?;
            restored.identities.insert(id, saved.identity);
            restored.agent_rng_counters.insert(id, saved.rng_counters);
            restored.runtime.insert(id, runtime);
        }
        restored.agent_execution_order_canonical = true;
        restored.active_effects = state
            .active_effects
            .iter()
            .copied()
            .map(ActiveEffectCheckpointV1::restore)
            .collect();
        restored.pending_birth_records = state
            .pending_birth_records
            .iter()
            .map(BirthRecordCheckpointV1::restore)
            .collect();
        // Lifecycle metrics are host output, not science state. Capturable tick-zero pending rows
        // are Seeded/Injected roots and were never lifecycle metrics in the source. At later
        // disabled boundaries the marker prevents a caller from splicing persistence onto an
        // incomplete historical run.
        if restored.tick != Tick::zero() {
            restored.persistence_discarded_records_at = Some(restored.tick);
        }
        restored.persistence_boundary = PersistenceBoundaryStatus::Open {
            tick: restored.tick,
        };

        let actual_digest = restored.world_digest_v1()?;
        if actual_digest != state.source_digest {
            return Err(digest_mismatch(&state.source_digest, &actual_digest));
        }
        Ok(restored)
    }

    fn ensure_checkpoint_has_no_deferred_host_output(&self) -> Result<(), WorldCheckpointError> {
        // `last_*` and combat totals describe the completed tick whose owned outcome has already
        // been returned; they are reset or overwritten before they can influence science.
        // Carcass totals are persistence-only analytics accumulators; disabled persistence cannot
        // later be spliced back in after its discard marker. None of these fields is checkpoint
        // science state or deferred work. Only still-undelivered queues and persistence
        // accumulators block capture.
        let blockers = [
            ("pending_deaths", !self.pending_deaths.is_empty()),
            ("pending_spawns", !self.pending_spawns.is_empty()),
            (
                "pending_death_records",
                !self.pending_death_records.is_empty(),
            ),
            (
                "pending_lifecycle_birth_metrics",
                !self.pending_lifecycle_birth_metrics.is_empty(),
            ),
            (
                "pending_lifecycle_death_metrics",
                !self.pending_lifecycle_death_metrics.is_empty(),
            ),
            ("replay_events", !self.replay_events.is_empty()),
            (
                "pending_persistence_runtime_tail",
                !self.pending_persistence_runtime_tail.is_empty(),
            ),
            (
                "pending_interventions",
                !self.pending_interventions.is_empty(),
            ),
            ("pending_birth_events", self.pending_birth_events != 0),
            ("pending_death_events", self.pending_death_events != 0),
            (
                "pending_spike_attempt_events",
                self.pending_spike_attempt_events != 0,
            ),
            (
                "pending_spike_hit_events",
                self.pending_spike_hit_events != 0,
            ),
        ];
        if let Some((field, _)) = blockers.into_iter().find(|(_, blocked)| *blocked) {
            return Err(WorldCheckpointError::DeferredHostOutput { field });
        }
        Ok(())
    }

    fn capture_checkpoint_agent(
        &self,
        uid: AgentUid,
        id: AgentId,
    ) -> Result<AgentCheckpointV1, WorldCheckpointError> {
        let identity =
            self.identities
                .get(id)
                .copied()
                .ok_or_else(|| WorldCheckpointError::Contract {
                    path: format!("agents[uid={}].identity", uid.0),
                    detail: "missing stable identity".to_owned(),
                })?;
        let rng_counters =
            self.agent_rng_counters(id)
                .ok_or_else(|| WorldCheckpointError::Contract {
                    path: format!("agents[uid={}].rng_counters", uid.0),
                    detail: "missing agent random-substream continuation counters".to_owned(),
                })?;
        let data = self
            .agents
            .snapshot(id)
            .ok_or_else(|| WorldCheckpointError::Contract {
                path: format!("agents[uid={}].data", uid.0),
                detail: "missing dense scalar state".to_owned(),
            })?;
        let runtime = self
            .runtime
            .get(id)
            .ok_or_else(|| WorldCheckpointError::Contract {
                path: format!("agents[uid={}].runtime", uid.0),
                detail: "missing runtime state".to_owned(),
            })?;
        let brain = match &runtime.brain {
            BrainBinding::Unbound => AgentBrainCheckpointV1::Unbound,
            BrainBinding::Legacy {
                registry_key, kind, ..
            } => {
                return Err(WorldCheckpointError::LegacyBrain {
                    agent_uid: uid.0,
                    kind: kind.clone(),
                    registry_key: *registry_key,
                });
            }
            BrainBinding::Protocol {
                registry_key,
                kind,
                genome,
                ..
            } => {
                let adapter = self.brain_registry.family(*registry_key).ok_or_else(|| {
                    WorldCheckpointError::Brain {
                        agent_uid: uid.0,
                        kind: kind.clone(),
                        detail: format!(
                            "registry key {registry_key} has no protocol adapter for `{}`",
                            genome.family_id()
                        ),
                    }
                })?;
                let evaluator_state = runtime
                    .brain
                    .checkpoint_evaluator_state_with(adapter)
                    .map_err(|error| WorldCheckpointError::Brain {
                        agent_uid: uid.0,
                        kind: kind.clone(),
                        detail: error.to_string(),
                    })?
                    .ok_or_else(|| WorldCheckpointError::Brain {
                        agent_uid: uid.0,
                        kind: kind.clone(),
                        detail: "protocol binding did not expose evaluator state".to_owned(),
                    })?;
                AgentBrainCheckpointV1::Protocol {
                    registry_key: *registry_key,
                    kind: kind.clone(),
                    genome: genome.clone(),
                    evaluator_state,
                }
            }
        };
        Ok(AgentCheckpointV1 {
            identity,
            rng_counters,
            data,
            runtime: AgentRuntimeCheckpointV1::capture(runtime),
            brain,
        })
    }
}

impl BrainRegistry {
    fn checkpoint_v1(&self) -> BrainRegistryCheckpointV1 {
        let mut entries = self
            .entries
            .iter()
            .map(|(key, entry)| BrainRegistryEntryCheckpointV1 {
                key: *key,
                kind: entry.kind.to_string(),
                factory_state_digest: entry.factory_state_digest,
                adapter_identity: entry.adapter_identity,
                protocol_family: entry
                    .protocol_adapter
                    .as_ref()
                    .map(|adapter| adapter.family_id().clone()),
            })
            .collect::<Vec<_>>();
        entries.sort_unstable_by_key(|entry| entry.key);
        BrainRegistryCheckpointV1 {
            next_key: self.next_key,
            entries,
        }
    }
}

impl AgentRuntimeCheckpointV1 {
    const fn capture(runtime: &AgentRuntime) -> Self {
        Self {
            energy: runtime.energy,
            reproduction_counter: runtime.reproduction_counter,
            herbivore_tendency: runtime.herbivore_tendency,
            mutation_rates: runtime.mutation_rates,
            trait_modifiers: runtime.trait_modifiers,
            clocks: runtime.clocks,
            eye_fov: runtime.eye_fov,
            eye_direction: runtime.eye_direction,
            sound_multiplier: runtime.sound_multiplier,
            give_intent: runtime.give_intent,
            sensors: runtime.sensors,
            outputs: runtime.outputs,
            food_delta: runtime.food_delta,
            spiked: runtime.spiked,
            hybrid: runtime.hybrid,
            sound_output: runtime.sound_output,
            temperature_preference: runtime.temperature_preference,
            lineage: runtime.lineage,
            food_balance_total: runtime.food_balance_total,
        }
    }

    fn restore(&self, brain: BrainBinding) -> AgentRuntime {
        AgentRuntime {
            energy: self.energy,
            reproduction_counter: self.reproduction_counter,
            herbivore_tendency: self.herbivore_tendency,
            mutation_rates: self.mutation_rates,
            trait_modifiers: self.trait_modifiers,
            clocks: self.clocks,
            eye_fov: self.eye_fov,
            eye_direction: self.eye_direction,
            sound_multiplier: self.sound_multiplier,
            give_intent: self.give_intent,
            sensors: self.sensors,
            outputs: self.outputs,
            indicator: IndicatorState::default(),
            selection: SelectionState::None,
            combat: CombatEventFlags::default(),
            food_delta: self.food_delta,
            spiked: self.spiked,
            hybrid: self.hybrid,
            sound_output: self.sound_output,
            temperature_preference: self.temperature_preference,
            brain,
            lineage: self.lineage,
            mutation_log: Vec::new(),
            food_balance_total: self.food_balance_total,
        }
    }
}

impl FoodCheckpointV1 {
    fn capture(food: &FoodGrid) -> Self {
        Self {
            width: food.width(),
            height: food.height(),
            cells: food.cells().to_vec(),
        }
    }

    fn restore(&self) -> Result<FoodGrid, WorldCheckpointError> {
        let mut food = FoodGrid::new(self.width, self.height, 0.0)?;
        food.try_replace_cells(self.cells.clone())?;
        Ok(food)
    }
}

impl TerrainCheckpointV1 {
    fn capture(terrain: &TerrainLayer) -> Self {
        Self {
            width: terrain.width(),
            height: terrain.height(),
            cell_size: terrain.cell_size(),
            tiles: terrain.tiles().to_vec(),
        }
    }

    fn restore(&self) -> Result<TerrainLayer, WorldCheckpointError> {
        Ok(TerrainLayer::from_tiles(
            self.width,
            self.height,
            self.cell_size,
            self.tiles.clone(),
        )?)
    }
}

impl MapMetadataCheckpointV1 {
    fn capture(metadata: &MapArtifactMetadata) -> Self {
        Self {
            generator: metadata.generator,
            tileset_id: metadata.tileset_id.clone(),
            tileset_hash: metadata.tileset_hash,
            seed: metadata.seed,
            width: metadata.width,
            height: metadata.height,
            attempt_count: metadata.attempt_count,
            succeeded_on: metadata.succeeded_on,
            generated_at_epoch_ms: metadata.generated_at_epoch_ms,
        }
    }

    fn restore(&self) -> MapArtifactMetadata {
        MapArtifactMetadata {
            generator: self.generator,
            tileset_id: self.tileset_id.clone(),
            tileset_hash: self.tileset_hash,
            seed: self.seed,
            width: self.width,
            height: self.height,
            attempt_count: self.attempt_count,
            succeeded_on: self.succeeded_on,
            generated_at_epoch_ms: self.generated_at_epoch_ms,
        }
    }
}

impl HydrologyCheckpointV1 {
    fn capture(hydrology: &HydrologyState) -> Self {
        let tiles = hydrology.tiles();
        let field = hydrology.field();
        Self {
            width: tiles.width(),
            height: tiles.height(),
            tiles: tiles.tiles().to_vec(),
            flow_directions: field.flow_directions().to_vec(),
            accumulation: field.accumulation().to_vec(),
            spill_elevation: field.spill_elevation().to_vec(),
            basin_ids: field.basin_ids().to_vec(),
            initial_water_depth: field.initial_water_depth().to_vec(),
            water_depth: hydrology.water_depth().to_vec(),
        }
    }

    fn restore(&self) -> Result<HydrologyState, WorldCheckpointError> {
        let tiles = HydrologyTileLayer::new(self.width, self.height, self.tiles.clone())?;
        let field = HydrologyField::new(
            self.width,
            self.height,
            self.flow_directions.clone(),
            self.accumulation.clone(),
            self.spill_elevation.clone(),
            self.basin_ids.clone(),
            self.initial_water_depth.clone(),
        )?;
        let mut hydrology = HydrologyState::new(tiles, field)?;
        if self.water_depth.len() != hydrology.cell_count() {
            return contract_error(
                "hydrology.water_depth",
                format!(
                    "found {} cells, expected {}",
                    self.water_depth.len(),
                    hydrology.cell_count()
                ),
            );
        }
        hydrology.try_update_water_depth(|depth| depth.copy_from_slice(&self.water_depth))?;
        Ok(hydrology)
    }
}

impl RegionCheckpointV1 {
    const fn capture(region: Region) -> Self {
        match region {
            Region::All => Self::All,
            Region::Disc { x, y, radius } => Self::Disc { x, y, radius },
            Region::Rect { x, y, w, h } => Self::Rect { x, y, w, h },
        }
    }

    const fn restore(self) -> Region {
        match self {
            Self::All => Region::All,
            Self::Disc { x, y, radius } => Region::Disc { x, y, radius },
            Self::Rect { x, y, w, h } => Region::Rect { x, y, w, h },
        }
    }
}

impl ActiveEffectCheckpointV1 {
    const fn capture(effect: ActiveEffect) -> Self {
        let kind = match effect.kind {
            ActiveEffectKind::GrowthScale(growth_scale) => {
                ActiveEffectKindCheckpointV1::GrowthScale(growth_scale)
            }
            ActiveEffectKind::Embargo => ActiveEffectKindCheckpointV1::Embargo,
        };
        Self {
            region: RegionCheckpointV1::capture(effect.region),
            ticks_remaining: effect.ticks_remaining,
            kind,
        }
    }

    const fn restore(self) -> ActiveEffect {
        let kind = match self.kind {
            ActiveEffectKindCheckpointV1::GrowthScale(growth_scale) => {
                ActiveEffectKind::GrowthScale(growth_scale)
            }
            ActiveEffectKindCheckpointV1::Embargo => ActiveEffectKind::Embargo,
        };
        ActiveEffect {
            region: self.region.restore(),
            ticks_remaining: self.ticks_remaining,
            kind,
        }
    }
}

impl BirthRecordCheckpointV1 {
    fn capture(record: &BirthRecord) -> Self {
        Self {
            tick: record.tick,
            agent_uid: record.agent_uid,
            spawn_ordinal: record.spawn_ordinal,
            birth_ordinal: record.birth_ordinal,
            origin: record.origin,
            parent_a: record.parent_a,
            parent_b: record.parent_b,
            brain_kind: record.brain_kind.clone(),
            brain_key: record.brain_key,
            herbivore_tendency: record.herbivore_tendency,
            generation: record.generation,
            position: record.position,
            is_hybrid: record.is_hybrid,
        }
    }

    fn restore(&self) -> BirthRecord {
        BirthRecord {
            tick: self.tick,
            agent_uid: self.agent_uid,
            spawn_ordinal: self.spawn_ordinal,
            birth_ordinal: self.birth_ordinal,
            origin: self.origin,
            parent_a: self.parent_a,
            parent_b: self.parent_b,
            brain_kind: self.brain_kind.clone(),
            brain_key: self.brain_key,
            herbivore_tendency: self.herbivore_tendency,
            generation: self.generation,
            position: self.position,
            is_hybrid: self.is_hybrid,
        }
    }
}

fn validate_registry_checkpoint(
    registry: &BrainRegistryCheckpointV1,
) -> Result<(), WorldCheckpointError> {
    if registry.next_key == u64::MAX {
        return contract_error(
            "registry.next_key",
            "registry key allocation has no remaining headroom",
        );
    }
    let mut previous = None;
    for (index, entry) in registry.entries.iter().enumerate() {
        if entry.kind.is_empty() || entry.kind.len() > MAX_CHECKPOINT_KIND_BYTES {
            return contract_error(
                format!("registry.entries[{index}].kind"),
                format!("kind must contain 1..={MAX_CHECKPOINT_KIND_BYTES} UTF-8 bytes"),
            );
        }
        if previous.is_some_and(|key| key >= entry.key) {
            return contract_error(
                "registry.entries",
                "registry keys must be strictly increasing and unique",
            );
        }
        if entry.key >= registry.next_key {
            return contract_error(
                format!("registry.entries[{index}].key"),
                format!(
                    "key {} is not below next_key {}",
                    entry.key, registry.next_key
                ),
            );
        }
        match (
            entry.protocol_family.is_some(),
            entry.factory_state_digest.is_some(),
            entry.adapter_identity.is_some(),
        ) {
            (true, false, true) | (false, _, false) => {}
            (true, true, _) => {
                return contract_error(
                    format!("registry.entries[{index}].factory_state_digest"),
                    "protocol families must use adapter identity rather than a legacy factory-state digest",
                );
            }
            (true, false, false) => {
                return contract_error(
                    format!("registry.entries[{index}].adapter_identity"),
                    "protocol families must carry a family-authored adapter identity",
                );
            }
            (false, _, true) => {
                return contract_error(
                    format!("registry.entries[{index}].adapter_identity"),
                    "legacy factories cannot claim a protocol adapter identity",
                );
            }
        }
        previous = Some(entry.key);
    }
    // Gaps caused by unregistering are valid, including an empty registry whose keys were all
    // retired. Do not coerce `next_key` back to zero; future registration identity depends on it.
    Ok(())
}

fn validate_environment_checkpoint(
    state: &WorldCheckpointStateV1,
) -> Result<(), WorldCheckpointError> {
    let (expected_width, expected_height) = state.config.food_dimensions()?;
    let expected_cells = usize::try_from(expected_width)
        .ok()
        .and_then(|width| {
            usize::try_from(expected_height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| WorldCheckpointError::Contract {
            path: "environment.dimensions".to_owned(),
            detail: "config-derived grid dimensions overflow usize".to_owned(),
        })?;
    ensure_count_bound(
        "environment.dimensions",
        expected_cells,
        MAX_CHECKPOINT_GRID_CELLS,
    )?;
    for (path, count) in [
        ("food.cells", state.food.cells.len()),
        ("terrain.tiles", state.terrain.tiles.len()),
    ] {
        ensure_count_bound(path, count, MAX_CHECKPOINT_GRID_CELLS)?;
    }
    if (state.food.width, state.food.height) != (expected_width, expected_height) {
        return contract_error(
            "food.dimensions",
            format!(
                "found {}x{}, config requires {expected_width}x{expected_height}",
                state.food.width, state.food.height
            ),
        );
    }
    if state.food.cells.len() != expected_cells {
        return contract_error(
            "food.cells",
            format!(
                "found {} cells, expected {expected_cells}",
                state.food.cells.len()
            ),
        );
    }
    if state.terrain.tiles.len() != expected_cells {
        return contract_error(
            "terrain.tiles",
            format!(
                "found {} cells, expected {expected_cells}",
                state.terrain.tiles.len()
            ),
        );
    }
    let _food = state.food.restore()?;
    let terrain = state.terrain.restore()?;
    if terrain.width() != expected_width
        || terrain.height() != expected_height
        || terrain.cell_size() != state.config.food_cell_size
    {
        return contract_error(
            "terrain.dimensions",
            "terrain must exactly match the config-derived food grid and cell size",
        );
    }
    if let Some(metadata) = &state.map_metadata
        && metadata.tileset_id.len() > MAX_CHECKPOINT_TILESET_ID_BYTES
    {
        return contract_error(
            "map_metadata.tileset_id",
            format!(
                "tileset ID must contain at most {MAX_CHECKPOINT_TILESET_ID_BYTES} UTF-8 bytes"
            ),
        );
    }
    if let Some(hydrology) = &state.hydrology {
        for (path, count) in [
            ("hydrology.tiles", hydrology.tiles.len()),
            ("hydrology.flow_directions", hydrology.flow_directions.len()),
            ("hydrology.accumulation", hydrology.accumulation.len()),
            ("hydrology.spill_elevation", hydrology.spill_elevation.len()),
            ("hydrology.basin_ids", hydrology.basin_ids.len()),
            (
                "hydrology.initial_water_depth",
                hydrology.initial_water_depth.len(),
            ),
            ("hydrology.water_depth", hydrology.water_depth.len()),
        ] {
            ensure_count_bound(path, count, MAX_CHECKPOINT_GRID_CELLS)?;
        }
        if (hydrology.width, hydrology.height) != (expected_width, expected_height) {
            return contract_error(
                "hydrology.dimensions",
                "hydrology dimensions do not match terrain",
            );
        }
        for (path, count) in [
            ("hydrology.tiles", hydrology.tiles.len()),
            ("hydrology.flow_directions", hydrology.flow_directions.len()),
            ("hydrology.accumulation", hydrology.accumulation.len()),
            ("hydrology.spill_elevation", hydrology.spill_elevation.len()),
            ("hydrology.basin_ids", hydrology.basin_ids.len()),
            (
                "hydrology.initial_water_depth",
                hydrology.initial_water_depth.len(),
            ),
            ("hydrology.water_depth", hydrology.water_depth.len()),
        ] {
            if count != expected_cells {
                return contract_error(
                    path,
                    format!("found {count} cells, expected {expected_cells}"),
                );
            }
        }
        let _validated = hydrology.restore()?;
    }
    Ok(())
}

fn validate_agent_checkpoint(state: &WorldCheckpointStateV1) -> Result<(), WorldCheckpointError> {
    let expected_epoch = state.tick.0 / 10_000;
    if state.epoch != expected_epoch {
        return contract_error(
            "epoch",
            format!(
                "found epoch {}, but tick {} requires epoch {expected_epoch}",
                state.epoch, state.tick.0
            ),
        );
    }
    if state.next_agent_uid != state.next_spawn_ordinal.saturating_add(1) {
        return contract_error(
            "next_agent_uid",
            "next_agent_uid must equal next_spawn_ordinal + 1",
        );
    }
    let mut uids = BTreeSet::new();
    let mut spawn_ordinals = BTreeSet::new();
    let mut birth_ordinals = BTreeSet::new();
    let mut previous_uid = None;
    let registry_by_key = state
        .registry
        .entries
        .iter()
        .map(|entry| (entry.key, entry))
        .collect::<HashMap<_, _>>();

    for (index, saved) in state.agents.iter().enumerate() {
        let identity = saved.identity;
        let path = format!("agents[{index}]");
        if identity.uid.0 == 0 {
            return contract_error(format!("{path}.identity.uid"), "UID zero is reserved");
        }
        if identity.uid.0 != identity.spawn_ordinal.saturating_add(1) {
            return contract_error(
                format!("{path}.identity"),
                "stable UID must equal spawn_ordinal + 1",
            );
        }
        if previous_uid.is_some_and(|uid| uid >= identity.uid) {
            return contract_error(
                "agents",
                "agents must be strictly ordered by stable AgentUid",
            );
        }
        previous_uid = Some(identity.uid);
        if !uids.insert(identity.uid) {
            return contract_error(format!("{path}.identity.uid"), "duplicate stable UID");
        }
        if !spawn_ordinals.insert(identity.spawn_ordinal) {
            return contract_error(
                format!("{path}.identity.spawn_ordinal"),
                "duplicate live spawn ordinal",
            );
        }
        if let Some(ordinal) = identity.birth_ordinal
            && !birth_ordinals.insert(ordinal)
        {
            return contract_error(
                format!("{path}.identity.birth_ordinal"),
                "duplicate live birth ordinal",
            );
        }
        saved.data.validate_at(&path)?;
        let runtime = saved.runtime.restore(BrainBinding::Unbound);
        runtime.validate_at(&format!("{path}.runtime"))?;
        validate_lineage(
            &format!("{path}.runtime"),
            identity.uid,
            saved.runtime.lineage,
        )?;
        match &saved.brain {
            AgentBrainCheckpointV1::Unbound => {}
            AgentBrainCheckpointV1::Protocol {
                registry_key,
                kind,
                genome,
                evaluator_state,
            } => {
                let Some(entry) = registry_by_key.get(registry_key) else {
                    return contract_error(
                        format!("{path}.brain.registry_key"),
                        format!("registry key {registry_key} is absent"),
                    );
                };
                if &entry.kind != kind {
                    return contract_error(
                        format!("{path}.brain.kind"),
                        format!("found `{kind}`, registry declares `{}`", entry.kind),
                    );
                }
                if entry.protocol_family.as_ref() != Some(genome.family_id()) {
                    return contract_error(
                        format!("{path}.brain.genome.family_id"),
                        "genome family does not match the registered protocol adapter",
                    );
                }
                if evaluator_state.family_id() != genome.family_id() {
                    return contract_error(
                        format!("{path}.brain.evaluator_state.family_id"),
                        "evaluator and genome families differ",
                    );
                }
                genome.require_protocol(
                    genome.family_id(),
                    genome.schema_version(),
                    genome.codec_version(),
                )?;
                evaluator_state.require_protocol(
                    evaluator_state.family_id(),
                    evaluator_state.schema_version(),
                    evaluator_state.codec_version(),
                )?;
                if genome.provenance().created_at.0 > state.tick.0 {
                    return contract_error(
                        format!("{path}.brain.genome.provenance.created_at"),
                        "genome provenance cannot originate after the checkpoint tick",
                    );
                }
                if genome.provenance().parents != saved.runtime.lineage {
                    return contract_error(
                        format!("{path}.brain.genome.provenance.parents"),
                        "genome provenance parents must match runtime lineage",
                    );
                }
                validate_lineage(
                    &format!("{path}.brain.genome.provenance"),
                    identity.uid,
                    genome.provenance().parents,
                )?;
            }
        }
    }

    if let Some(max_uid) = uids.last()
        && state.next_agent_uid <= max_uid.0
    {
        return contract_error(
            "next_agent_uid",
            format!(
                "next UID {} must exceed live UID {}",
                state.next_agent_uid, max_uid.0
            ),
        );
    }
    if let Some(max_spawn) = spawn_ordinals.last()
        && state.next_spawn_ordinal <= *max_spawn
    {
        return contract_error(
            "next_spawn_ordinal",
            format!(
                "next spawn ordinal {} must exceed live ordinal {max_spawn}",
                state.next_spawn_ordinal
            ),
        );
    }
    if let Some(max_birth) = birth_ordinals.last()
        && state.next_birth_ordinal <= *max_birth
    {
        return contract_error(
            "next_birth_ordinal",
            format!(
                "next birth ordinal {} must exceed live ordinal {max_birth}",
                state.next_birth_ordinal
            ),
        );
    }
    if state.next_agent_uid == 0 {
        return contract_error("next_agent_uid", "UID zero is reserved");
    }
    for (path, value) in [
        ("next_agent_uid", state.next_agent_uid),
        ("next_spawn_ordinal", state.next_spawn_ordinal),
        ("next_birth_ordinal", state.next_birth_ordinal),
    ] {
        if value == u64::MAX {
            return contract_error(path, "allocation counter has no remaining headroom");
        }
    }
    if state.tick.0 == u64::MAX {
        return contract_error("tick", "checkpoint tick has no continuation headroom");
    }
    Ok(())
}

fn validate_agent_rng_checkpoint(
    state: &WorldCheckpointStateV1,
) -> Result<(), WorldCheckpointError> {
    let counters = state
        .agents
        .iter()
        .map(|saved| AgentRngCounterStateV1::new(saved.identity.uid, saved.rng_counters))
        .collect::<Vec<_>>();
    let actual = world_counters_digest_v1(
        &state.agent_substream_protocol,
        state.tick,
        state.epoch,
        state.next_agent_uid,
        state.next_spawn_ordinal,
        state.next_birth_ordinal,
        &counters,
    );
    if actual != state.source_digest.counters {
        return contract_error(
            "source_digest.counters",
            format!(
                "digest records `{}`, but checkpoint counters recompute to `{actual}`",
                state.source_digest.counters
            ),
        );
    }
    Ok(())
}

fn validate_lineage(
    path: &str,
    child: AgentUid,
    lineage: [Option<AgentUid>; 2],
) -> Result<(), WorldCheckpointError> {
    if lineage[0].is_none() && lineage[1].is_some() {
        return contract_error(
            format!("{path}.lineage"),
            "parent slot 1 cannot be populated while slot 0 is empty",
        );
    }
    for (index, parent) in lineage.into_iter().enumerate() {
        if let Some(parent) = parent {
            if parent.0 == 0 {
                return contract_error(
                    format!("{path}.lineage[{index}]"),
                    "parent UID zero is reserved",
                );
            }
            if parent == child {
                return contract_error(
                    format!("{path}.lineage[{index}]"),
                    "an agent cannot be its own parent",
                );
            }
            if parent >= child {
                return contract_error(
                    format!("{path}.lineage[{index}]"),
                    "parent UID must precede the child UID",
                );
            }
        }
    }
    if lineage[0].is_some() && lineage[0] == lineage[1] {
        return contract_error(
            format!("{path}.lineage"),
            "two-parent lineage must name distinct parents",
        );
    }
    Ok(())
}

fn validate_origin_checkpoint(state: &WorldCheckpointStateV1) -> Result<(), WorldCheckpointError> {
    let agents = state
        .agents
        .iter()
        .map(|saved| (saved.identity.uid, saved))
        .collect::<HashMap<_, _>>();
    let mut origin_uids = BTreeSet::new();
    let mut previous_spawn_ordinal = None;
    for (index, origin) in state.pending_birth_records.iter().enumerate() {
        let path = format!("pending_birth_records[{index}]");
        if origin.tick != state.tick {
            return contract_error(
                format!("{path}.tick"),
                "only current-boundary origin rows may remain pending",
            );
        }
        if !origin_uids.insert(origin.agent_uid) {
            return contract_error(format!("{path}.agent_uid"), "duplicate pending origin");
        }
        if previous_spawn_ordinal.is_some_and(|ordinal| ordinal >= origin.spawn_ordinal) {
            return contract_error(
                "pending_birth_records",
                "pending origins must retain strictly increasing spawn order",
            );
        }
        previous_spawn_ordinal = Some(origin.spawn_ordinal);
        let Some(saved) = agents.get(&origin.agent_uid) else {
            return contract_error(
                format!("{path}.agent_uid"),
                "pending origin does not identify a live agent",
            );
        };
        if saved.identity.spawn_ordinal != origin.spawn_ordinal
            || saved.identity.birth_ordinal != origin.birth_ordinal
            || saved.data.generation != origin.generation
            || saved.data.position != origin.position
            || saved.runtime.lineage != [origin.parent_a, origin.parent_b]
            || saved.runtime.hybrid != origin.is_hybrid
            || clamp01(saved.runtime.herbivore_tendency).to_bits()
                != origin.herbivore_tendency.to_bits()
        {
            return contract_error(
                path,
                "pending origin disagrees with the live agent identity, lineage, or scalar state",
            );
        }
        let (brain_key, brain_kind) = match &saved.brain {
            AgentBrainCheckpointV1::Unbound => (None, None),
            AgentBrainCheckpointV1::Protocol {
                registry_key, kind, ..
            } => (Some(*registry_key), Some(kind.as_str())),
        };
        if origin.brain_key != brain_key || origin.brain_kind.as_deref() != brain_kind {
            return contract_error(
                format!("{path}.brain"),
                "pending origin brain identity disagrees with the live agent",
            );
        }
        match origin.origin {
            BirthOrigin::Born if origin.birth_ordinal.is_none() => {
                return contract_error(
                    format!("{path}.birth_ordinal"),
                    "Born origin requires a demographic birth ordinal",
                );
            }
            BirthOrigin::Seeded | BirthOrigin::Injected if origin.birth_ordinal.is_some() => {
                return contract_error(
                    format!("{path}.birth_ordinal"),
                    "Seeded and Injected origins cannot carry a birth ordinal",
                );
            }
            BirthOrigin::Born if origin.parent_a.is_none() => {
                return contract_error(
                    format!("{path}.parent_a"),
                    "Born origin requires at least one parent",
                );
            }
            BirthOrigin::Seeded if state.tick != Tick::zero() => {
                return contract_error(
                    format!("{path}.origin"),
                    "Seeded origin is valid only at the tick-zero bootstrap boundary",
                );
            }
            _ => {}
        }
        if origin.origin == BirthOrigin::Born && origin.is_hybrid != origin.parent_b.is_some() {
            return contract_error(
                format!("{path}.is_hybrid"),
                "hybrid origin must carry exactly two parents",
            );
        }
        validate_finite(
            &format!("{path}.herbivore_tendency"),
            origin.herbivore_tendency,
        )?;
        validate_finite(&format!("{path}.position.x"), origin.position.x)?;
        validate_finite(&format!("{path}.position.y"), origin.position.y)?;
    }
    Ok(())
}

fn registry_mismatch_detail(
    expected: &BrainRegistryCheckpointV1,
    actual: &BrainRegistryCheckpointV1,
) -> String {
    if expected.next_key != actual.next_key {
        return format!(
            "next_key is {}, expected {}",
            actual.next_key, expected.next_key
        );
    }
    if expected.entries.len() != actual.entries.len() {
        return format!(
            "entry count is {}, expected {}",
            actual.entries.len(),
            expected.entries.len()
        );
    }
    for (expected, actual) in expected.entries.iter().zip(&actual.entries) {
        if expected != actual {
            return format!("entry mismatch: found {actual:?}, expected {expected:?}");
        }
    }
    "registry differs in an unclassified field".to_owned()
}

fn digest_mismatch(expected: &WorldDigestV1, actual: &WorldDigestV1) -> WorldCheckpointError {
    let string_lanes = [
        ("schema", expected.schema.as_str(), actual.schema.as_str()),
        (
            "algorithm",
            expected.algorithm.as_str(),
            actual.algorithm.as_str(),
        ),
        (
            "agent_identity",
            expected.agent_identity.as_str(),
            actual.agent_identity.as_str(),
        ),
        ("agents", expected.agents.as_str(), actual.agents.as_str()),
        ("brains", expected.brains.as_str(), actual.brains.as_str()),
        ("food", expected.food.as_str(), actual.food.as_str()),
        (
            "terrain",
            expected.terrain.as_str(),
            actual.terrain.as_str(),
        ),
        (
            "rng",
            expected.rng.overall.as_str(),
            actual.rng.overall.as_str(),
        ),
        (
            "counters",
            expected.counters.as_str(),
            actual.counters.as_str(),
        ),
        (
            "brain_registry",
            expected.brain_registry.as_str(),
            actual.brain_registry.as_str(),
        ),
        ("config", expected.config.as_str(), actual.config.as_str()),
        (
            "effects",
            expected.effects.as_str(),
            actual.effects.as_str(),
        ),
        (
            "derived_transition",
            expected.derived_transition.as_str(),
            actual.derived_transition.as_str(),
        ),
        (
            "origins",
            expected.origins.as_str(),
            actual.origins.as_str(),
        ),
    ];
    for (lane, expected, actual) in string_lanes {
        if expected != actual {
            return WorldCheckpointError::DigestMismatch {
                lane,
                expected: expected.to_owned(),
                actual: actual.to_owned(),
            };
        }
    }
    for domain in RngDomain::ALL {
        let expected_domain = expected.rng.domains.get(domain);
        let actual_domain = actual.rng.domains.get(domain);
        if expected_domain != actual_domain {
            return WorldCheckpointError::DigestMismatch {
                lane: "rng_domain",
                expected: format!("{}={expected_domain}", domain.tag()),
                actual: format!("{}={actual_domain}", domain.tag()),
            };
        }
    }
    let expected_diagnostic = format!(
        "tick={:?}, codec={}, hydrology={:?}, evaluator_coverage={:?}, factory_coverage={:?}, overall={}",
        expected.tick,
        expected.codec_version,
        expected.hydrology,
        (
            expected.evaluator_state_covered,
            &expected.uncovered_families
        ),
        (
            expected.factory_state_covered,
            &expected.uncovered_factory_families
        ),
        expected.overall
    );
    let actual_diagnostic = format!(
        "tick={:?}, codec={}, hydrology={:?}, evaluator_coverage={:?}, factory_coverage={:?}, overall={}",
        actual.tick,
        actual.codec_version,
        actual.hydrology,
        (actual.evaluator_state_covered, &actual.uncovered_families),
        (
            actual.factory_state_covered,
            &actual.uncovered_factory_families
        ),
        actual.overall
    );
    WorldCheckpointError::DigestMismatch {
        lane: "metadata_or_coverage",
        expected: expected_diagnostic,
        actual: actual_diagnostic,
    }
}

fn contract_error<T>(
    path: impl Into<String>,
    detail: impl Into<String>,
) -> Result<T, WorldCheckpointError> {
    Err(WorldCheckpointError::Contract {
        path: path.into(),
        detail: detail.into(),
    })
}

fn ensure_count_bound(
    path: impl Into<String>,
    found: usize,
    maximum: usize,
) -> Result<(), WorldCheckpointError> {
    if found > maximum {
        contract_error(
            path,
            format!("contains {found} entries; maximum is {maximum}"),
        )
    } else {
        Ok(())
    }
}

struct BoundedVecVisitor<T, const LIMIT: usize> {
    label: &'static str,
    marker: PhantomData<T>,
}

impl<'de, T, const LIMIT: usize> Visitor<'de> for BoundedVecVisitor<T, LIMIT>
where
    T: Deserialize<'de>,
{
    type Value = Vec<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} with at most {LIMIT} entries", self.label)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let hinted = sequence.size_hint().unwrap_or_default();
        if hinted > LIMIT {
            return Err(serde::de::Error::invalid_length(hinted, &self));
        }
        let element_bytes = std::mem::size_of::<T>().max(1);
        let eager_limit = MAX_CHECKPOINT_EAGER_ALLOCATION_BYTES / element_bytes;
        let mut values = Vec::with_capacity(hinted.min(LIMIT).min(eager_limit));
        while let Some(value) = sequence.next_element()? {
            if values.len() == LIMIT {
                return Err(serde::de::Error::invalid_length(values.len() + 1, &self));
            }
            values.push(value);
        }
        Ok(values)
    }
}

fn deserialize_bounded_vec<'de, D, T, const LIMIT: usize>(
    deserializer: D,
    label: &'static str,
) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    deserializer.deserialize_seq(BoundedVecVisitor::<T, LIMIT> {
        label,
        marker: PhantomData,
    })
}

fn deserialize_checkpoint_payload<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, u8, MAX_WORLD_CHECKPOINT_PAYLOAD_BYTES>(
        deserializer,
        "checkpoint payload",
    )
}

fn deserialize_checkpoint_world_digest<'de, D>(deserializer: D) -> Result<WorldDigestV1, D::Error>
where
    D: Deserializer<'de>,
{
    let decoded = WorldDigestCheckpointDecodeV1::deserialize(deserializer)?;
    let domains = decoded.rng.domains;
    Ok(WorldDigestV1 {
        schema: decoded.schema,
        codec_version: decoded.codec_version,
        algorithm: decoded.algorithm,
        tick: decoded.tick,
        overall: decoded.overall,
        agents: decoded.agents,
        brains: decoded.brains,
        food: decoded.food,
        terrain: decoded.terrain,
        hydrology: decoded.hydrology,
        rng: RngDomainDigestV1 {
            overall: decoded.rng.overall,
            domains: RngDomainDigestsV1 {
                environment: domains.environment,
                food: domains.food,
                population: domains.population,
                lineage: domains.lineage,
                mutation: domains.mutation,
                crossover: domains.crossover,
            },
        },
        counters: decoded.counters,
        brain_registry: decoded.brain_registry,
        config: decoded.config,
        effects: decoded.effects,
        derived_transition: decoded.derived_transition,
        origins: decoded.origins,
        evaluator_state_covered: decoded.evaluator_state_covered,
        uncovered_families: decoded
            .uncovered_families
            .into_iter()
            .map(|family| family.0)
            .collect(),
        factory_state_covered: decoded.factory_state_covered,
        uncovered_factory_families: decoded
            .uncovered_factory_families
            .into_iter()
            .map(|family| family.0)
            .collect(),
        agent_identity: decoded.agent_identity,
    })
}

fn deserialize_checkpoint_agents<'de, D>(
    deserializer: D,
) -> Result<Vec<AgentCheckpointV1>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, AgentCheckpointV1, MAX_CHECKPOINT_AGENTS>(
        deserializer,
        "checkpoint agents",
    )
}

fn deserialize_checkpoint_registry_entries<'de, D>(
    deserializer: D,
) -> Result<Vec<BrainRegistryEntryCheckpointV1>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, BrainRegistryEntryCheckpointV1, MAX_CHECKPOINT_REGISTRY_ENTRIES>(
        deserializer,
        "checkpoint registry entries",
    )
}

fn deserialize_checkpoint_f32_cells<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, f32, MAX_CHECKPOINT_GRID_CELLS>(
        deserializer,
        "checkpoint scalar cells",
    )
}

fn deserialize_checkpoint_u32_cells<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, u32, MAX_CHECKPOINT_GRID_CELLS>(
        deserializer,
        "checkpoint integer cells",
    )
}

fn deserialize_checkpoint_terrain_tiles<'de, D>(
    deserializer: D,
) -> Result<Vec<TerrainTile>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, TerrainTile, MAX_CHECKPOINT_GRID_CELLS>(
        deserializer,
        "checkpoint terrain tiles",
    )
}

fn deserialize_checkpoint_hydrology_tiles<'de, D>(
    deserializer: D,
) -> Result<Vec<HydrologyTile>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, HydrologyTile, MAX_CHECKPOINT_GRID_CELLS>(
        deserializer,
        "checkpoint hydrology tiles",
    )
}

fn deserialize_checkpoint_flow_directions<'de, D>(
    deserializer: D,
) -> Result<Vec<HydrologyFlowDirection>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, HydrologyFlowDirection, MAX_CHECKPOINT_GRID_CELLS>(
        deserializer,
        "checkpoint hydrology flow directions",
    )
}

fn deserialize_checkpoint_active_effects<'de, D>(
    deserializer: D,
) -> Result<Vec<ActiveEffectCheckpointV1>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, ActiveEffectCheckpointV1, MAX_CHECKPOINT_ACTIVE_EFFECTS>(
        deserializer,
        "checkpoint active effects",
    )
}

fn deserialize_checkpoint_origins<'de, D>(
    deserializer: D,
) -> Result<Vec<BirthRecordCheckpointV1>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, BirthRecordCheckpointV1, MAX_CHECKPOINT_ORIGINS>(
        deserializer,
        "checkpoint pending origins",
    )
}

struct BoundedStringVisitor<const LIMIT: usize> {
    label: &'static str,
}

impl<const LIMIT: usize> BoundedStringVisitor<LIMIT> {
    fn validate<E>(self, value: &str) -> Result<String, E>
    where
        E: serde::de::Error,
    {
        if value.len() > LIMIT {
            Err(E::invalid_length(value.len(), &self))
        } else {
            Ok(value.to_owned())
        }
    }
}

impl<'de, const LIMIT: usize> Visitor<'de> for BoundedStringVisitor<LIMIT> {
    type Value = String;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} with at most {LIMIT} UTF-8 bytes", self.label)
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.validate(value)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.validate(value)
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if value.len() > LIMIT {
            Err(E::invalid_length(value.len(), &self))
        } else {
            Ok(value)
        }
    }
}

fn deserialize_bounded_string<'de, D, const LIMIT: usize>(
    deserializer: D,
    label: &'static str,
) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_string(BoundedStringVisitor::<LIMIT> { label })
}

fn deserialize_checkpoint_kind<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_string::<D, MAX_CHECKPOINT_KIND_BYTES>(deserializer, "brain kind")
}

fn deserialize_optional_checkpoint_kind<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptionalKindVisitor;

    impl<'de> Visitor<'de> for OptionalKindVisitor {
        type Value = Option<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an optional bounded brain kind")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize_checkpoint_kind(deserializer).map(Some)
        }
    }

    deserializer.deserialize_option(OptionalKindVisitor)
}

fn deserialize_checkpoint_tileset_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_string::<D, MAX_CHECKPOINT_TILESET_ID_BYTES>(deserializer, "tileset ID")
}

fn deserialize_checkpoint_digest_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_string::<D, MAX_CHECKPOINT_DIGEST_STRING_BYTES>(
        deserializer,
        "world-digest field",
    )
}

fn deserialize_checkpoint_family_name<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_string::<D, MAX_CHECKPOINT_KIND_BYTES>(
        deserializer,
        "world-digest family name",
    )
}

fn deserialize_optional_checkpoint_digest_string<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptionalDigestVisitor;

    impl<'de> Visitor<'de> for OptionalDigestVisitor {
        type Value = Option<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an optional bounded world-digest field")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize_checkpoint_digest_string(deserializer).map(Some)
        }
    }

    deserializer.deserialize_option(OptionalDigestVisitor)
}

fn deserialize_checkpoint_uncovered_families<'de, D>(
    deserializer: D,
) -> Result<Vec<CheckpointFamilyName>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, CheckpointFamilyName, MAX_CHECKPOINT_UNCOVERED_FAMILIES>(
        deserializer,
        "world-digest uncovered families",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{
        boxed_fixture_brain_family, boxed_fixture_brain_family_with_behavior_probe,
    };
    use crate::{BrainGenomeDerivation, BrainRunner, LocomotionModel};
    use crate::{
        Intervention, MAX_NEUROFLOW_HIDDEN_LAYERS, MAX_NEUROFLOW_LAYER_NEURONS, RenderSettings,
    };
    use rand::RngCore;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    const CHECKPOINT_KIND: &str = "checkpoint-test-protocol";
    const CHECKPOINT_FAMILY_ID: &str = "checkpoint-test-protocol-family";

    fn checkpoint_config() -> ScriptBotsConfig {
        ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            initial_food: 0.25,
            food_respawn_interval: 0,
            food_growth_rate: 0.0,
            food_decay_rate: 0.0,
            food_diffusion_rate: 0.0,
            food_intake_rate: 0.0,
            food_sharing_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            bot_speed: 0.0,
            locomotion_model: LocomotionModel::Differential,
            spike_damage: 0.0,
            spike_energy_cost: 0.0,
            aging_health_decay_rate: 0.0,
            population_minimum: 0,
            population_spawn_interval: 0,
            reproduction_energy_threshold: 0.5,
            reproduction_energy_cost: 0.0,
            reproduction_cooldown: 1,
            reproduction_attempt_interval: 1,
            reproduction_attempt_chance: 1.0,
            reproduction_child_energy: 1.0,
            reproduction_spawn_jitter: 0.0,
            reproduction_color_jitter: 0.0,
            reproduction_spawn_back_distance: 0.0,
            reproduction_partner_chance: 0.0,
            reproduction_meta_mutation_chance: 0.0,
            reproduction_meta_mutation_scale: 0.0,
            persistence_interval: 0,
            chart_flush_interval: 0,
            history_capacity: 1,
            narrative_interval: 0,
            narrative_capacity: 0,
            rng_seed: Some(0xC4EC_9A71),
            ..ScriptBotsConfig::default()
        }
    }

    fn register_checkpoint_family(registry: &mut BrainRegistry) -> u64 {
        registry
            .register_family(
                CHECKPOINT_KIND,
                boxed_fixture_brain_family(CHECKPOINT_FAMILY_ID),
            )
            .expect("register checkpoint protocol fixture")
    }

    fn world_with_checkpoint_family() -> (WorldState, u64) {
        let mut world = WorldState::new(checkpoint_config()).expect("checkpoint test world");
        let key = world
            .register_brain_family(
                CHECKPOINT_KIND,
                boxed_fixture_brain_family(CHECKPOINT_FAMILY_ID),
            )
            .expect("register source protocol fixture");
        (world, key)
    }

    fn prepared_checkpoint_registry() -> BrainRegistry {
        let mut registry = BrainRegistry::new();
        assert_eq!(register_checkpoint_family(&mut registry), 0);
        registry
    }

    fn encode_unvalidated_state(state: &WorldCheckpointStateV1) -> Vec<u8> {
        let payload = postcard::to_allocvec(state).expect("raw checkpoint payload fixture");
        let payload_blake3 = *blake3::hash(&payload).as_bytes();
        postcard::to_allocvec(&WorldCheckpointWireV1 {
            schema: WORLD_CHECKPOINT_V1_SCHEMA.to_owned(),
            codec_version: WORLD_CHECKPOINT_V1_CODEC_VERSION,
            codec: WORLD_CHECKPOINT_V1_CODEC.to_owned(),
            payload,
            payload_blake3,
        })
        .expect("raw checkpoint envelope fixture")
    }

    fn protocol_with_json_field(
        protocol: &AgentSubstreamProtocolV1,
        field: &str,
        value: serde_json::Value,
    ) -> AgentSubstreamProtocolV1 {
        let mut encoded = serde_json::to_value(protocol).expect("protocol JSON fixture");
        encoded
            .as_object_mut()
            .expect("protocol must serialize as an object")
            .insert(field.to_owned(), value);
        serde_json::from_value(encoded).expect("structurally valid protocol tamper")
    }

    fn agent_id_for_uid(world: &WorldState, uid: AgentUid) -> AgentId {
        world
            .agents()
            .iter_handles()
            .find(|id| world.agent_uid(*id) == Some(uid))
            .expect("stable UID must resolve to a live handle")
    }

    fn uid_handle_signature(world: &WorldState) -> Vec<(AgentUid, u64)> {
        let mut signature = world
            .agents()
            .iter_handles()
            .map(|id| (world.agent_uid(id).expect("live handle identity"), id.raw()))
            .collect::<Vec<_>>();
        signature.sort_unstable_by_key(|(uid, _)| *uid);
        signature
    }

    fn assert_digest_lanes_equal(expected: &WorldDigestV1, actual: &WorldDigestV1) {
        assert_eq!(actual.schema, expected.schema, "schema");
        assert_eq!(actual.codec_version, expected.codec_version, "codec");
        assert_eq!(actual.algorithm, expected.algorithm, "algorithm");
        assert_eq!(actual.tick, expected.tick, "tick");
        assert_eq!(actual.agent_identity, expected.agent_identity, "identity");
        assert_eq!(actual.agents, expected.agents, "agents lane");
        assert_eq!(actual.brains, expected.brains, "brains lane");
        assert_eq!(actual.food, expected.food, "food lane");
        assert_eq!(actual.terrain, expected.terrain, "terrain lane");
        assert_eq!(actual.hydrology, expected.hydrology, "hydrology lane");
        for domain in RngDomain::ALL {
            assert_eq!(
                actual.rng.domains.get(domain),
                expected.rng.domains.get(domain),
                "{} RNG lane",
                domain.tag()
            );
        }
        assert_eq!(actual.rng.overall, expected.rng.overall, "RNG aggregate");
        assert_eq!(actual.counters, expected.counters, "counters lane");
        assert_eq!(
            actual.brain_registry, expected.brain_registry,
            "registry lane"
        );
        assert_eq!(actual.config, expected.config, "config lane");
        assert_eq!(actual.effects, expected.effects, "effects lane");
        assert_eq!(
            actual.derived_transition, expected.derived_transition,
            "derived lane"
        );
        assert_eq!(actual.origins, expected.origins, "origins lane");
        assert_eq!(
            actual.evaluator_state_covered, expected.evaluator_state_covered,
            "evaluator coverage"
        );
        assert_eq!(
            actual.uncovered_families, expected.uncovered_families,
            "uncovered evaluators"
        );
        assert_eq!(
            actual.factory_state_covered, expected.factory_state_covered,
            "factory coverage"
        );
        assert_eq!(
            actual.uncovered_factory_families, expected.uncovered_factory_families,
            "uncovered factories"
        );
        assert_eq!(actual.overall, expected.overall, "overall");
        assert_eq!(actual, expected, "complete digest DTO");
    }

    fn assert_protocol_state_equal(left: &WorldState, right: &WorldState) {
        assert_eq!(
            right.agent_substream_protocol_v1(),
            left.agent_substream_protocol_v1(),
            "agent random-substream protocol"
        );
        assert_eq!(
            right
                .ordered_agent_rng_counters_v1()
                .expect("right ordered random counters"),
            left.ordered_agent_rng_counters_v1()
                .expect("left ordered random counters"),
            "complete stable-UID random-counter lane"
        );
        let mut uids = left
            .agents()
            .iter_handles()
            .map(|id| left.agent_uid(id).expect("left stable UID"))
            .collect::<Vec<_>>();
        uids.sort_unstable();
        let mut right_uids = right
            .agents()
            .iter_handles()
            .map(|id| right.agent_uid(id).expect("right stable UID"))
            .collect::<Vec<_>>();
        right_uids.sort_unstable();
        assert_eq!(right_uids, uids, "stable UID roster");

        for uid in uids {
            let left_id = agent_id_for_uid(left, uid);
            let right_id = agent_id_for_uid(right, uid);
            let left_runtime = left.agent_runtime(left_id).expect("left runtime");
            let right_runtime = right.agent_runtime(right_id).expect("right runtime");
            assert_eq!(
                right.agent_rng_counters(right_id),
                left.agent_rng_counters(left_id),
                "agent random counters for UID {uid:?}"
            );
            assert_eq!(right_runtime.lineage, left_runtime.lineage, "UID {uid:?}");
            assert_eq!(
                right.agent_brain_genome(right_id),
                left.agent_brain_genome(left_id),
                "full genome envelope and provenance for UID {uid:?}"
            );
            assert_eq!(
                right
                    .agent_brain_evaluator_state(right_id)
                    .expect("right evaluator checkpoint"),
                left.agent_brain_evaluator_state(left_id)
                    .expect("left evaluator checkpoint"),
                "evaluator state for UID {uid:?}"
            );
        }
    }

    fn install_nontrivial_hydrology(world: &mut WorldState) {
        let width = world.terrain.width();
        let height = world.terrain.height();
        let count = (width as usize) * (height as usize);
        let tiles = HydrologyTileLayer::new(
            width,
            height,
            vec![
                HydrologyTile {
                    permeability: 0.4,
                    runoff_bias: 0.2,
                    basin_rank: 0.5,
                    channel_priority: 0.7,
                    swim_cost: 1.1,
                };
                count
            ],
        )
        .expect("hydrology tiles");
        let field = HydrologyField::new(
            width,
            height,
            vec![HydrologyFlowDirection::East; count],
            vec![2.0; count],
            vec![0.6; count],
            vec![3; count],
            vec![0.15; count],
        )
        .expect("hydrology field");
        let mut hydrology = HydrologyState::new(tiles, field).expect("hydrology state");
        hydrology
            .try_update_water_depth(|depth| depth[7] = 0.875)
            .expect("nontrivial live water depth");
        world.hydrology = Some(hydrology);
    }

    // The nested RNG algorithm identity deliberately names its pointer-width lane. Keep this
    // literal golden on the pinned 64-bit verification target; semantic round-trip tests remain
    // portable across the other supported lanes.
    #[cfg(target_pointer_width = "64")]
    #[test]
    #[ignore = "bd-16g.10.1: ActiveEffect gained a kind discriminant (GrowthScale/Embargo) and \
     Region a Rect variant; the V1.3/codec-5 wire re-pin must be recorded with fresh DSR \
     evidence before this byte golden runs again"]
    fn checkpoint_v1_representative_wire_golden() {
        let (mut world, brain_key) = world_with_checkpoint_family();
        install_nontrivial_hydrology(&mut world);
        world.active_effects.push(ActiveEffect {
            region: Region::Disc {
                x: 40.0,
                y: 60.0,
                radius: 25.0,
            },
            ticks_remaining: 4,
            kind: ActiveEffectKind::GrowthScale(0.35),
        });
        world
            .try_update_food(|cells| {
                cells[3] = 0.75;
                cells[41] = 0.125;
            })
            .expect("representative food field");
        let data = AgentData {
            position: Position::new(37.5, 82.25),
            health: 1.75,
            ..AgentData::default()
        };
        let agent = world
            .try_inject_agent(data)
            .expect("representative checkpoint agent");
        assert!(
            world
                .bind_agent_brain(agent, brain_key)
                .expect("bind representative protocol brain")
        );
        world
            .try_update_agent_runtime(agent, |runtime| {
                runtime.energy = 1.25;
                runtime.reproduction_counter = 0.375;
                runtime.herbivore_tendency = 0.625;
            })
            .expect("customize representative runtime");

        let wire = world
            .checkpoint_v1()
            .expect("representative checkpoint")
            .encode()
            .expect("representative checkpoint wire");
        let decoded = WorldCheckpointV1::decode(&wire).expect("decode representative wire");
        assert_eq!(
            decoded.encode().expect("re-encode representative wire"),
            wire,
            "the representative wire must remain canonical and idempotent"
        );
        let actual = blake3::hash(&wire).to_hex().to_string();
        assert_eq!(
            (wire.len(), actual.as_str()),
            (
                8_547,
                "2a1483ddfdb2300cdcb5cdaf09179b1a2b0d1f0bfbc22934b057fd9d57c7fecf",
            ),
            "the reviewed V1.3/codec-5 wire must remain byte-identical: re-pinned in \
             bd-2i1 after the locomotion model was bound, then re-pinned in \
             bd-2z0.14.3.1 after RenderSettings v2 added presentation-only fields \
             (quality tier, post stack, day/night, theme, palette) to the serialized \
             config tree (+7 bytes of Option discriminants), and re-pinned in \
             bd-2z0.8.9.8 after replay_event_tick_cap (a persistence-recording knob, \
             default 0) joined the serialized config tree (+1 byte of usize varint); \
             no science field changed"
        );
    }

    #[test]
    fn evolved_world_round_trips_by_uid_and_continues_identically() {
        let (mut original, brain_key) = world_with_checkpoint_family();
        install_nontrivial_hydrology(&mut original);
        original.active_effects.push(ActiveEffect {
            region: Region::Disc {
                x: 40.0,
                y: 60.0,
                radius: 25.0,
            },
            ticks_remaining: 4,
            kind: ActiveEffectKind::GrowthScale(0.35),
        });
        original
            .try_update_food(|cells| {
                cells[3] = 0.75;
                cells[41] = 0.125;
            })
            .expect("nontrivial food field");

        let founder_data = AgentData {
            position: Position::new(80.0, 90.0),
            ..AgentData::default()
        };
        let founder = original
            .try_spawn_agent(founder_data)
            .expect("protocol founder");
        assert!(
            original
                .bind_agent_brain(founder, brain_key)
                .expect("bind founder")
        );
        original
            .try_update_agent_runtime(founder, |runtime| {
                runtime.energy = 2.0;
                runtime.reproduction_counter = 1.0;
                runtime.mutation_rates = MutationRates {
                    primary: 1.0,
                    secondary: 0.5,
                };
            })
            .expect("make founder reproductive");
        let founder_uid = original.agent_uid(founder).expect("founder UID");
        let founder_genome = original
            .agent_brain_genome(founder)
            .expect("founder genome")
            .clone();
        let founder_evaluator_before = original
            .agent_brain_evaluator_state(founder)
            .expect("initial founder evaluator checkpoint");

        let victim_data = AgentData {
            position: Position::new(120.0, 90.0),
            health: 0.0,
            ..AgentData::default()
        };
        let victim = original
            .try_spawn_agent(victim_data)
            .expect("sacrificial agent");
        let victim_uid = original.agent_uid(victim).expect("victim UID");
        let victim_slot = victim.raw() & u64::from(u32::MAX);
        original
            .try_update_agent_runtime(victim, |runtime| runtime.energy = 0.0)
            .expect("make victim terminal");

        let completion = original.step_outcome().expect("evolutionary step");
        assert!(
            completion.fault.is_none(),
            "evolutionary fixture must reach a fault-free completed boundary"
        );
        let evolution = completion.outcome;
        assert_eq!(evolution.births.len(), 1, "one real demographic birth");
        assert_eq!(evolution.births[0].origin, BirthOrigin::Born);
        assert_eq!(evolution.births[0].parent_a, Some(founder_uid));
        assert_eq!(evolution.births[0].parent_b, None);
        assert_eq!(evolution.deaths.len(), 1, "one real lifecycle death");
        assert_eq!(evolution.deaths[0].agent_uid, victim_uid);
        assert!(
            original.agent_uid(victim).is_none(),
            "the evolved fixture must contain a real death"
        );
        assert_eq!(original.agent_count(), 2, "one death and one birth");
        let child_uid = evolution.births[0].agent_uid;
        assert_ne!(child_uid, victim_uid, "stable UIDs are never recycled");
        let child_id = agent_id_for_uid(&original, child_uid);
        assert_ne!(child_id, victim, "slot reuse must advance the generation");
        assert_eq!(
            child_id.raw() & u64::from(u32::MAX),
            victim_slot,
            "death-before-birth must recycle the physical slot in this fixture"
        );
        let child_runtime = original.agent_runtime(child_id).expect("child runtime");
        assert_eq!(child_runtime.lineage, [Some(founder_uid), None]);
        let child_genome = original.agent_brain_genome(child_id).expect("child genome");
        assert_ne!(
            child_genome.material_hash(),
            founder_genome.material_hash(),
            "primary mutation probability 1.0 must change the child genome"
        );
        assert_eq!(
            child_genome.provenance().parents,
            [Some(founder_uid), None],
            "real genome provenance must name the parent"
        );
        assert_eq!(
            child_genome.provenance().parent_genome_hashes,
            [Some(founder_genome.material_hash()), None]
        );
        assert_eq!(child_genome.provenance().created_at, Tick(1));
        assert_eq!(
            child_genome.provenance().derivation,
            BrainGenomeDerivation::MutationOnly
        );
        assert_ne!(
            original
                .agent_brain_evaluator_state(founder)
                .expect("evolved founder evaluator checkpoint"),
            founder_evaluator_before,
            "the fixture must exercise a non-default evaluator state"
        );
        assert!(
            original
                .ordered_agent_rng_counters_v1()
                .expect("evolved random counters")
                .iter()
                .any(|state| state.counters() != AgentRngCountersV1::default()),
            "the round-trip fixture must advance at least one persisted agent random counter"
        );

        let injected_data = AgentData {
            position: Position::new(15.0, 25.0),
            ..AgentData::default()
        };
        let injected = original
            .try_inject_agent(injected_data)
            .expect("same-boundary injected arrival");
        assert!(
            original
                .bind_agent_brain(injected, brain_key)
                .expect("bind injected protocol brain")
        );
        assert_eq!(
            original.pending_birth_records.len(),
            1,
            "the origins lane must be nonempty at save time"
        );

        let source_signature = uid_handle_signature(&original);
        let checkpoint = original.checkpoint_v1().expect("capture evolved world");
        let first_wire = checkpoint.encode().expect("encode checkpoint");
        let decoded = WorldCheckpointV1::decode(&first_wire).expect("decode checkpoint");
        let second_wire = decoded.encode().expect("re-encode checkpoint");
        assert_eq!(second_wire, first_wire, "checkpoint encoding is idempotent");

        let mut restored =
            WorldState::restore_checkpoint_v1(&decoded, prepared_checkpoint_registry())
                .expect("restore evolved checkpoint");
        assert_ne!(
            original.config().history_capacity,
            ScriptBotsConfig::default().history_capacity,
            "the fixture must exercise a digest-neutralized nondefault config field"
        );
        assert_eq!(
            restored.config(),
            original.config(),
            "the complete config must round-trip, including fields neutralized by WorldDigestV1"
        );
        assert_ne!(
            uid_handle_signature(&restored),
            source_signature,
            "restore must allocate fresh physical AgentIds"
        );
        assert_protocol_state_equal(&original, &restored);
        assert_digest_lanes_equal(
            &original.world_digest_v1().expect("source digest"),
            &restored.world_digest_v1().expect("restored digest"),
        );

        // Deliberately advance every domain once on both worlds. This proves exact restoration of
        // all six RNG checkpoints even though the continuation fixture below does not naturally
        // draw from every domain.
        for domain in RngDomain::ALL {
            assert_eq!(
                original.rng.stream(domain).next_u64(),
                restored.rng.stream(domain).next_u64(),
                "{} RNG continuation",
                domain.tag()
            );
        }

        // Force another real reproduction boundary after restore. Together with the explicit RNG
        // probes above, this exercises evaluator, genome, lineage, counter, and mutation
        // continuation rather than only save-time equality.
        let uids = original
            .agents()
            .iter_handles()
            .map(|id| original.agent_uid(id).expect("continuation UID"))
            .collect::<Vec<_>>();
        for uid in uids {
            for world in [&mut original, &mut restored] {
                let id = agent_id_for_uid(world, uid);
                world
                    .try_update_agent_runtime(id, |runtime| {
                        runtime.energy = 2.0;
                        runtime.reproduction_counter = 1.0;
                    })
                    .expect("force continuation reproduction");
            }
        }
        let source_completion = original.step_outcome().expect("source continuation step");
        let restored_completion = restored.step_outcome().expect("restored continuation step");
        assert!(
            source_completion.fault.is_none(),
            "source continuation fault"
        );
        assert!(
            restored_completion.fault.is_none(),
            "restored continuation fault"
        );
        let source_outcome = source_completion.outcome;
        let restored_outcome = restored_completion.outcome;
        assert_eq!(restored_outcome.events, source_outcome.events);
        assert_eq!(restored_outcome.summary, source_outcome.summary);
        assert_eq!(restored_outcome.births, source_outcome.births);
        assert_eq!(restored_outcome.deaths, source_outcome.deaths);
        assert_eq!(restored_outcome.combat, source_outcome.combat);
        assert_eq!(
            restored_outcome.config_revision,
            source_outcome.config_revision
        );
        assert_eq!(
            restored_outcome.persistence.status(),
            source_outcome.persistence.status()
        );
        assert_eq!(restored_outcome.resource_tick, source_outcome.resource_tick);
        assert_protocol_state_equal(&original, &restored);
        assert_digest_lanes_equal(
            &original
                .world_digest_v1()
                .expect("source continuation digest"),
            &restored
                .world_digest_v1()
                .expect("restored continuation digest"),
        );
    }

    #[test]
    fn wire_metadata_integrity_and_semantic_tampering_fail_closed() {
        let (world, _) = world_with_checkpoint_family();
        let checkpoint = world.checkpoint_v1().expect("empty-world checkpoint");
        let encoded = checkpoint.encode().expect("checkpoint wire");
        let mut envelope_with_trailing_data = encoded.clone();
        envelope_with_trailing_data.push(0xA5);
        assert!(matches!(
            WorldCheckpointV1::decode(&envelope_with_trailing_data),
            Err(WorldCheckpointError::TrailingBytes {
                layer: "envelope",
                count: 1
            })
        ));

        assert!(encoded[0] < 0x80, "schema length uses a one-byte varint");
        let mut noncanonical_envelope = Vec::with_capacity(encoded.len() + 1);
        noncanonical_envelope.push(encoded[0] | 0x80);
        noncanonical_envelope.push(0);
        noncanonical_envelope.extend_from_slice(&encoded[1..]);
        assert!(matches!(
            WorldCheckpointV1::decode(&noncanonical_envelope),
            Err(WorldCheckpointError::NonCanonical { layer: "envelope" })
        ));

        let mut declared_oversize_payload = Vec::new();
        declared_oversize_payload.extend(
            postcard::to_allocvec(&WORLD_CHECKPOINT_V1_SCHEMA.to_owned()).expect("schema prefix"),
        );
        declared_oversize_payload.extend(
            postcard::to_allocvec(&WORLD_CHECKPOINT_V1_CODEC_VERSION)
                .expect("codec-version prefix"),
        );
        declared_oversize_payload.extend(
            postcard::to_allocvec(&WORLD_CHECKPOINT_V1_CODEC.to_owned()).expect("codec prefix"),
        );
        declared_oversize_payload.extend(
            postcard::to_allocvec(&(MAX_WORLD_CHECKPOINT_PAYLOAD_BYTES + 1))
                .expect("oversize vector-length prefix"),
        );
        assert!(matches!(
            WorldCheckpointV1::decode(&declared_oversize_payload),
            Err(WorldCheckpointError::Codec {
                operation: "envelope decoding",
                ..
            })
        ));

        let mut wire: WorldCheckpointWireV1 =
            postcard::from_bytes(&encoded).expect("decode private wire fixture");

        wire.schema = "scriptbots.world-checkpoint.v1.2".to_owned();
        let foreign_schema = postcard::to_allocvec(&wire).expect("foreign schema wire");
        assert!(matches!(
            WorldCheckpointV1::decode(&foreign_schema),
            Err(WorldCheckpointError::Schema { .. })
        ));

        let mut wire: WorldCheckpointWireV1 =
            postcard::from_bytes(&encoded).expect("decode integrity fixture");
        wire.codec_version = WORLD_CHECKPOINT_V1_CODEC_VERSION - 1;
        let foreign_codec = postcard::to_allocvec(&wire).expect("foreign codec wire");
        assert!(matches!(
            WorldCheckpointV1::decode(&foreign_codec),
            Err(WorldCheckpointError::CodecVersion { .. })
        ));

        let mut wire: WorldCheckpointWireV1 =
            postcard::from_bytes(&encoded).expect("decode codec-identity fixture");
        wire.codec = "postcard+blake3-v3".to_owned();
        let foreign_codec_identity =
            postcard::to_allocvec(&wire).expect("foreign codec identity wire");
        assert!(matches!(
            WorldCheckpointV1::decode(&foreign_codec_identity),
            Err(WorldCheckpointError::CodecIdentity { .. })
        ));

        let mut wire: WorldCheckpointWireV1 =
            postcard::from_bytes(&encoded).expect("decode corruption fixture");
        let last = wire.payload.last_mut().expect("nonempty payload");
        *last ^= 0x80;
        let corrupt = postcard::to_allocvec(&wire).expect("corrupt wire");
        assert!(matches!(
            WorldCheckpointV1::decode(&corrupt),
            Err(WorldCheckpointError::PayloadHashMismatch)
        ));

        let mut wire: WorldCheckpointWireV1 =
            postcard::from_bytes(&encoded).expect("decode trailing-payload fixture");
        wire.payload.push(0x5A);
        wire.payload_blake3 = *blake3::hash(&wire.payload).as_bytes();
        let payload_with_trailing_data =
            postcard::to_allocvec(&wire).expect("checksum-bound trailing-payload wire");
        assert!(matches!(
            WorldCheckpointV1::decode(&payload_with_trailing_data),
            Err(WorldCheckpointError::TrailingBytes {
                layer: "payload",
                count: 1
            })
        ));

        let mut wire: WorldCheckpointWireV1 =
            postcard::from_bytes(&encoded).expect("decode noncanonical-payload fixture");
        assert!(
            wire.payload[0] < 0x80,
            "digest schema length uses a one-byte varint"
        );
        let mut noncanonical_payload = Vec::with_capacity(wire.payload.len() + 1);
        noncanonical_payload.push(wire.payload[0] | 0x80);
        noncanonical_payload.push(0);
        noncanonical_payload.extend_from_slice(&wire.payload[1..]);
        wire.payload = noncanonical_payload;
        wire.payload_blake3 = *blake3::hash(&wire.payload).as_bytes();
        let noncanonical_payload_wire =
            postcard::to_allocvec(&wire).expect("noncanonical payload wire");
        assert!(matches!(
            WorldCheckpointV1::decode(&noncanonical_payload_wire),
            Err(WorldCheckpointError::NonCanonical { layer: "payload" })
        ));

        let mut missing_protocol =
            serde_json::to_value(&checkpoint.state).expect("checkpoint state JSON fixture");
        missing_protocol
            .as_object_mut()
            .expect("checkpoint state is an object")
            .remove("agent_substream_protocol");
        assert!(
            serde_json::from_value::<WorldCheckpointStateV1>(missing_protocol).is_err(),
            "checkpoint state missing the agent-substream protocol decoded"
        );

        let mut unknown_protocol_peer =
            serde_json::to_value(&checkpoint.state).expect("checkpoint state JSON fixture");
        unknown_protocol_peer
            .as_object_mut()
            .expect("checkpoint state is an object")
            .insert(
                "agent_substream_dense_lane".to_owned(),
                serde_json::json!("forbidden"),
            );
        assert!(
            serde_json::from_value::<WorldCheckpointStateV1>(unknown_protocol_peer).is_err(),
            "checkpoint state accepted an unknown agent-substream field"
        );

        let mut wrong_protocol_version = checkpoint.clone();
        let unsupported_version = wrong_protocol_version
            .state
            .agent_substream_protocol
            .version()
            + 1;
        let tampered_protocol = protocol_with_json_field(
            &wrong_protocol_version.state.agent_substream_protocol,
            "version",
            serde_json::json!(unsupported_version),
        );
        wrong_protocol_version.state.agent_substream_protocol = tampered_protocol;
        assert!(matches!(
            wrong_protocol_version.encode(),
            Err(WorldCheckpointError::AgentSubstreamProtocol(
                AgentSubstreamProtocolError::Version { .. }
            ))
        ));

        let mut wrong_protocol_algorithm = checkpoint.clone();
        let tampered_protocol = protocol_with_json_field(
            &wrong_protocol_algorithm.state.agent_substream_protocol,
            "algorithm",
            serde_json::json!("dense-agent-rng-v0"),
        );
        wrong_protocol_algorithm.state.agent_substream_protocol = tampered_protocol;
        assert!(matches!(
            wrong_protocol_algorithm.encode(),
            Err(WorldCheckpointError::AgentSubstreamProtocol(
                AgentSubstreamProtocolError::Algorithm { .. }
            ))
        ));

        let mut wrong_protocol_root = checkpoint.clone();
        wrong_protocol_root.state.agent_substream_protocol =
            AgentSubstreamProtocolV1::from_root_seed(
                wrong_protocol_root.state.random_streams.root_seed ^ 1,
            );
        assert!(matches!(
            wrong_protocol_root.encode(),
            Err(WorldCheckpointError::AgentSubstreamProtocol(
                AgentSubstreamProtocolError::RootSeed { .. }
            ))
        ));

        for hidden_layers in [
            vec![4, 0, 2],
            vec![1; MAX_NEUROFLOW_HIDDEN_LAYERS + 1],
            vec![MAX_NEUROFLOW_LAYER_NEURONS + 1],
            vec![1_024, 1_024],
        ] {
            let mut oversized_config = checkpoint.clone();
            oversized_config.state.config.neuroflow.hidden_layers = hidden_layers;
            let oversized_config_wire = encode_unvalidated_state(&oversized_config.state);
            assert!(matches!(
                WorldCheckpointV1::decode(&oversized_config_wire),
                Err(WorldCheckpointError::Codec {
                    operation: "payload decoding",
                    ..
                })
            ));
        }

        let (mut counter_world, brain_key) = world_with_checkpoint_family();
        let counter_agent = counter_world
            .try_spawn_agent(AgentData::default())
            .expect("counter fixture agent");
        assert!(
            counter_world
                .bind_agent_brain(counter_agent, brain_key)
                .expect("bind counter fixture brain")
        );
        let counter_checkpoint = counter_world
            .checkpoint_v1()
            .expect("checkpoint with agent counters");
        let mut missing_counters = serde_json::to_value(&counter_checkpoint.state.agents[0])
            .expect("agent checkpoint JSON fixture");
        missing_counters
            .as_object_mut()
            .expect("agent checkpoint is an object")
            .remove("rng_counters");
        assert!(
            serde_json::from_value::<AgentCheckpointV1>(missing_counters).is_err(),
            "agent checkpoint missing its random continuation counters decoded"
        );
        let mut unknown_counter_peer = serde_json::to_value(&counter_checkpoint.state.agents[0])
            .expect("agent checkpoint JSON fixture");
        unknown_counter_peer
            .as_object_mut()
            .expect("agent checkpoint is an object")
            .insert("rng_dense_slot".to_owned(), serde_json::json!(3));
        assert!(
            serde_json::from_value::<AgentCheckpointV1>(unknown_counter_peer).is_err(),
            "agent checkpoint accepted a dense-slot random continuation"
        );
        let mut changed_counters = counter_checkpoint;
        let saved_counters = changed_counters.state.agents[0].rng_counters;
        changed_counters.state.agents[0].rng_counters = AgentRngCountersV1::from_ordinals(
            saved_counters.reproduction_attempt_ordinal() + 1,
            saved_counters.birth_ordinal(),
            saved_counters.brain_initialization_ordinal(),
        );
        assert!(matches!(
            changed_counters.encode(),
            Err(WorldCheckpointError::Contract { ref path, .. })
                if path == "source_digest.counters"
        ));

        let mut duplicate = checkpoint.clone();
        duplicate.state.agents = vec![
            AgentCheckpointV1 {
                identity: AgentIdentity {
                    uid: AgentUid(1),
                    spawn_ordinal: 0,
                    birth_ordinal: None,
                },
                rng_counters: AgentRngCountersV1::default(),
                data: AgentData::default(),
                runtime: AgentRuntimeCheckpointV1::capture(&AgentRuntime::default()),
                brain: AgentBrainCheckpointV1::Unbound,
            },
            AgentCheckpointV1 {
                identity: AgentIdentity {
                    uid: AgentUid(1),
                    spawn_ordinal: 0,
                    birth_ordinal: None,
                },
                rng_counters: AgentRngCountersV1::default(),
                data: AgentData::default(),
                runtime: AgentRuntimeCheckpointV1::capture(&AgentRuntime::default()),
                brain: AgentBrainCheckpointV1::Unbound,
            },
        ];
        duplicate.state.next_agent_uid = 3;
        duplicate.state.next_spawn_ordinal = 2;
        assert!(matches!(
            duplicate.encode(),
            Err(WorldCheckpointError::Contract { ref path, .. }) if path == "agents"
        ));

        let mut non_finite = checkpoint;
        non_finite.state.agents.push(AgentCheckpointV1 {
            identity: AgentIdentity {
                uid: AgentUid(1),
                spawn_ordinal: 0,
                birth_ordinal: None,
            },
            rng_counters: AgentRngCountersV1::default(),
            data: AgentData::default(),
            runtime: AgentRuntimeCheckpointV1 {
                energy: f32::NAN,
                ..AgentRuntimeCheckpointV1::capture(&AgentRuntime::default())
            },
            brain: AgentBrainCheckpointV1::Unbound,
        });
        non_finite.state.next_agent_uid = 2;
        non_finite.state.next_spawn_ordinal = 1;
        assert!(matches!(
            non_finite.encode(),
            Err(WorldCheckpointError::ScientificState(
                ScientificStateError::NonFinite { .. }
            ))
        ));
    }

    struct LegacyCheckpointBrain;

    impl BrainRunner for LegacyCheckpointBrain {
        fn kind(&self) -> &'static str {
            "legacy-checkpoint-brain"
        }

        fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
            [0.0; OUTPUT_SIZE]
        }
    }

    #[test]
    fn legacy_brains_and_wrong_prepared_registries_are_rejected() {
        let mut legacy_world =
            WorldState::new(checkpoint_config()).expect("legacy checkpoint world");
        let legacy_key = legacy_world
            .brain_registry_mut()
            .expect("mutable legacy registry")
            .register("legacy-checkpoint-brain", |_rng| {
                Ok(Box::new(LegacyCheckpointBrain))
            });
        let agent = legacy_world
            .try_spawn_agent(AgentData::default())
            .expect("legacy agent");
        assert!(
            legacy_world
                .bind_agent_brain(agent, legacy_key)
                .expect("bind registered legacy brain")
        );
        assert!(matches!(
            legacy_world.checkpoint_v1(),
            Err(WorldCheckpointError::LegacyBrain {
                agent_uid: 1,
                ref kind,
                registry_key: Some(key),
            }) if kind == "legacy-checkpoint-brain"
                && key == legacy_key
        ));

        let (source, _) = world_with_checkpoint_family();
        let checkpoint = source.checkpoint_v1().expect("protocol checkpoint");
        assert!(matches!(
            WorldState::restore_checkpoint_v1(&checkpoint, BrainRegistry::new()),
            Err(WorldCheckpointError::RegistryMismatch { .. })
        ));

        let mut wrong_kind_registry = BrainRegistry::new();
        wrong_kind_registry
            .register_family(
                "wrong-kind",
                boxed_fixture_brain_family(CHECKPOINT_FAMILY_ID),
            )
            .expect("same-cardinality wrong-kind registry");
        assert!(matches!(
            WorldState::restore_checkpoint_v1(&checkpoint, wrong_kind_registry),
            Err(WorldCheckpointError::RegistryMismatch { .. })
        ));

        let mut retired_source =
            WorldState::new(checkpoint_config()).expect("retired-key checkpoint world");
        let source_registry = retired_source
            .brain_registry_mut()
            .expect("mutable retired-key source registry");
        let leading_retired_key = source_registry.register("retired-checkpoint-brain", |_rng| {
            Ok(Box::new(LegacyCheckpointBrain))
        });
        assert_eq!(leading_retired_key, 0);
        assert!(source_registry.unregister(leading_retired_key));
        assert_eq!(register_checkpoint_family(source_registry), 1);
        let trailing_retired_key = source_registry.register("retired-checkpoint-brain", |_rng| {
            Ok(Box::new(LegacyCheckpointBrain))
        });
        assert_eq!(trailing_retired_key, 2);
        assert!(source_registry.unregister(trailing_retired_key));
        let retired_checkpoint = retired_source
            .checkpoint_v1()
            .expect("checkpoint with retired leading and trailing registry keys");
        let requirements = retired_checkpoint.required_brain_registry();
        assert_eq!(requirements.next_key, 3);
        assert_eq!(requirements.entries.len(), 1);
        assert_eq!(requirements.entries[0].key, 1);
        assert_eq!(requirements.entries[0].kind, CHECKPOINT_KIND);
        assert_eq!(requirements.entries[0].factory_state_digest, None);
        assert_eq!(
            requirements.entries[0].adapter_identity,
            retired_source
                .brain_registry
                .adapter_identity(requirements.entries[0].key)
        );
        assert_eq!(
            requirements.entries[0]
                .protocol_family
                .as_ref()
                .map(BrainFamilyId::as_str),
            Some(CHECKPOINT_FAMILY_ID)
        );

        let mut cursor_too_low = BrainRegistry::new();
        let retired_key = cursor_too_low.register("retired-checkpoint-brain", |_rng| {
            Ok(Box::new(LegacyCheckpointBrain))
        });
        assert!(cursor_too_low.unregister(retired_key));
        assert_eq!(register_checkpoint_family(&mut cursor_too_low), 1);
        assert!(matches!(
            WorldState::restore_checkpoint_v1(&retired_checkpoint, cursor_too_low),
            Err(WorldCheckpointError::RegistryMismatch { .. })
        ));

        let mut retired_prepared = BrainRegistry::new();
        let prepared_leading_key = retired_prepared.register("retired-checkpoint-brain", |_rng| {
            Ok(Box::new(LegacyCheckpointBrain))
        });
        assert_eq!(prepared_leading_key, leading_retired_key);
        assert!(retired_prepared.unregister(prepared_leading_key));
        assert_eq!(register_checkpoint_family(&mut retired_prepared), 1);
        let prepared_trailing_key = retired_prepared.register("retired-checkpoint-brain", |_rng| {
            Ok(Box::new(LegacyCheckpointBrain))
        });
        assert_eq!(prepared_trailing_key, trailing_retired_key);
        assert!(retired_prepared.unregister(prepared_trailing_key));
        WorldState::restore_checkpoint_v1(&retired_checkpoint, retired_prepared)
            .expect("exact retired-key allocation cursor restores");

        let mut automatic_config = checkpoint_config();
        automatic_config.population_spawn_interval = 1;
        let mut legacy_only =
            WorldState::new(automatic_config).expect("automatic legacy-only world");
        legacy_only
            .brain_registry_mut()
            .expect("automatic legacy registry")
            .register("legacy-checkpoint-brain", |_rng| {
                Ok(Box::new(LegacyCheckpointBrain))
            });
        assert!(matches!(
            legacy_only.checkpoint_v1(),
            Err(WorldCheckpointError::Contract { ref path, .. })
                if path == "registry.entries"
        ));
    }

    #[test]
    fn agent_random_checkpoint_tampering_rejects_before_evaluator_reconstruction() {
        let source_constructions = Arc::new(AtomicUsize::new(0));
        let mut source = WorldState::new(checkpoint_config()).expect("random source world");
        let source_key = source
            .register_brain_family(
                CHECKPOINT_KIND,
                boxed_fixture_brain_family_with_behavior_probe(
                    CHECKPOINT_FAMILY_ID,
                    0,
                    Arc::clone(&source_constructions),
                ),
            )
            .expect("register random source fixture");
        let agent = source
            .try_spawn_agent(AgentData::default())
            .expect("random source agent");
        assert!(
            source
                .bind_agent_brain(agent, source_key)
                .expect("bind random source brain")
        );
        assert!(
            source_constructions.load(Ordering::Relaxed) > 0,
            "the source fixture must construct an evaluator before checkpointing"
        );
        let checkpoint = source.checkpoint_v1().expect("random source checkpoint");

        let prepared_constructions = Arc::new(AtomicUsize::new(0));
        let prepared_registry = || {
            let mut registry = BrainRegistry::new();
            let key = registry
                .register_family(
                    CHECKPOINT_KIND,
                    boxed_fixture_brain_family_with_behavior_probe(
                        CHECKPOINT_FAMILY_ID,
                        0,
                        Arc::clone(&prepared_constructions),
                    ),
                )
                .expect("register exact prepared random fixture");
            assert_eq!(key, source_key);
            registry
        };

        let mut wrong_root = checkpoint.clone();
        wrong_root.state.agent_substream_protocol =
            AgentSubstreamProtocolV1::from_root_seed(wrong_root.state.random_streams.root_seed ^ 1);
        assert!(matches!(
            WorldState::restore_checkpoint_v1(&wrong_root, prepared_registry()),
            Err(WorldCheckpointError::AgentSubstreamProtocol(
                AgentSubstreamProtocolError::RootSeed { .. }
            ))
        ));
        assert_eq!(
            prepared_constructions.load(Ordering::Relaxed),
            0,
            "protocol/root mismatch must reject before reconstructing any evaluator or agent"
        );

        let mut wrong_counter = checkpoint.clone();
        let saved = wrong_counter.state.agents[0].rng_counters;
        wrong_counter.state.agents[0].rng_counters = AgentRngCountersV1::from_ordinals(
            saved.reproduction_attempt_ordinal(),
            saved.birth_ordinal() + 1,
            saved.brain_initialization_ordinal(),
        );
        assert!(matches!(
            WorldState::restore_checkpoint_v1(&wrong_counter, prepared_registry()),
            Err(WorldCheckpointError::Contract { ref path, .. })
                if path == "source_digest.counters"
        ));
        assert_eq!(
            prepared_constructions.load(Ordering::Relaxed),
            0,
            "counter-lane mismatch must reject before reconstructing any evaluator or agent"
        );

        WorldState::restore_checkpoint_v1(&checkpoint, prepared_registry())
            .expect("untampered checkpoint reconstructs evaluators");
        assert!(
            prepared_constructions.load(Ordering::Relaxed) > 0,
            "the valid-control restore must prove the evaluator-construction probe is live"
        );
    }

    #[test]
    fn changed_adapter_identity_is_rejected_before_evaluator_reconstruction() {
        let source_constructions = Arc::new(AtomicUsize::new(0));
        let mut source = WorldState::new(checkpoint_config()).expect("identity source world");
        let source_key = source
            .register_brain_family(
                CHECKPOINT_KIND,
                boxed_fixture_brain_family_with_behavior_probe(
                    CHECKPOINT_FAMILY_ID,
                    0,
                    Arc::clone(&source_constructions),
                ),
            )
            .expect("register source identity fixture");
        let agent = source
            .try_spawn_agent(AgentData::default())
            .expect("identity fixture agent");
        assert!(
            source
                .bind_agent_brain(agent, source_key)
                .expect("bind identity fixture brain")
        );
        assert!(
            source_constructions.load(Ordering::Relaxed) > 0,
            "the source fixture must construct at least one evaluator before checkpointing"
        );

        let checkpoint = source.checkpoint_v1().expect("identity checkpoint");
        let requirement = &checkpoint.required_brain_registry().entries[0];
        let saved_identity = requirement
            .adapter_identity
            .expect("protocol checkpoint carries adapter identity");
        let mut missing_identity = checkpoint.clone();
        missing_identity.state.registry.entries[0].adapter_identity = None;
        assert!(matches!(
            missing_identity.encode(),
            Err(WorldCheckpointError::Contract { ref path, .. })
                if path == "registry.entries[0].adapter_identity"
        ));

        let prepared_constructions = Arc::new(AtomicUsize::new(0));
        let mut changed_registry = BrainRegistry::new();
        let changed_key = changed_registry
            .register_family(
                CHECKPOINT_KIND,
                boxed_fixture_brain_family_with_behavior_probe(
                    CHECKPOINT_FAMILY_ID,
                    1,
                    Arc::clone(&prepared_constructions),
                ),
            )
            .expect("register changed-behavior identity fixture");
        assert_eq!(changed_key, source_key);
        let changed_identity = changed_registry
            .adapter_identity(changed_key)
            .expect("prepared adapter identity");
        assert_ne!(
            changed_identity, saved_identity,
            "the fixture's changed evaluator behavior must move its semantic identity"
        );

        assert!(matches!(
            WorldState::restore_checkpoint_v1(&checkpoint, changed_registry),
            Err(WorldCheckpointError::RegistryMismatch { .. })
        ));
        assert_eq!(
            prepared_constructions.load(Ordering::Relaxed),
            0,
            "registry identity mismatch must reject before reconstructing any evaluator or agent"
        );

        let mut tampered = checkpoint.clone();
        tampered.state.registry.entries[0].adapter_identity = Some(changed_identity);
        let tampered_wire = encode_unvalidated_state(&tampered.state);
        assert!(matches!(
            WorldCheckpointV1::decode(&tampered_wire),
            Err(WorldCheckpointError::Contract { ref path, .. })
                if path == "source_digest.brain_registry"
        ));

        let mut tampered_registry = BrainRegistry::new();
        assert_eq!(
            tampered_registry
                .register_family(
                    CHECKPOINT_KIND,
                    boxed_fixture_brain_family_with_behavior_probe(
                        CHECKPOINT_FAMILY_ID,
                        1,
                        Arc::clone(&prepared_constructions),
                    ),
                )
                .expect("register tampered-registry identity fixture"),
            source_key
        );
        assert!(matches!(
            WorldState::restore_checkpoint_v1(&tampered, tampered_registry),
            Err(WorldCheckpointError::Contract { ref path, .. })
                if path == "source_digest.brain_registry"
        ));
        assert_eq!(
            prepared_constructions.load(Ordering::Relaxed),
            0,
            "checkpoint self-binding mismatch must reject before reconstructing any evaluator or agent"
        );
    }

    #[test]
    fn customized_root_origin_round_trips_without_fabricating_host_metrics() {
        let mut original =
            WorldState::new(checkpoint_config()).expect("custom root checkpoint world");
        let root = original
            .try_inject_agent(AgentData::default())
            .expect("injected root");
        original
            .try_update_agent_runtime(root, |runtime| {
                runtime.herbivore_tendency = 2.0;
                runtime.hybrid = true;
            })
            .expect("customize pending root origin");
        assert_eq!(original.pending_birth_records.len(), 1);
        assert_eq!(original.pending_birth_records[0].herbivore_tendency, 1.0);
        assert!(original.pending_birth_records[0].is_hybrid);
        assert!(original.pending_birth_records[0].parent_a.is_none());
        assert!(original.pending_birth_records[0].parent_b.is_none());

        let checkpoint = original.checkpoint_v1().expect("capture custom root");
        let restored = WorldState::restore_checkpoint_v1(&checkpoint, BrainRegistry::new())
            .expect("restore custom root");
        assert!(restored.pending_lifecycle_birth_metrics.is_empty());
        assert_eq!(
            restored
                .checkpoint_v1()
                .expect("immediate re-checkpoint")
                .encode()
                .expect("re-encode custom root"),
            checkpoint.encode().expect("source custom-root wire")
        );
    }

    #[test]
    fn capture_rejects_product_persistence_and_deferred_boundary_work() {
        let persistent = WorldState::new(ScriptBotsConfig {
            persistence_interval: 1,
            ..checkpoint_config()
        })
        .expect("persistence-enabled world");
        assert!(matches!(
            persistent.checkpoint_v1(),
            Err(WorldCheckpointError::PersistenceEnabled {
                persistence_interval: 1
            })
        ));

        let (mut queued, _) = world_with_checkpoint_family();
        queued
            .enqueue_intervention(Intervention::Bloom {
                region: Region::All,
                amount: 0.1,
            })
            .expect("queue deterministic intervention");
        assert!(matches!(
            queued.checkpoint_v1(),
            Err(WorldCheckpointError::DeferredHostOutput {
                field: "pending_interventions"
            })
        ));
    }
}
