//! Renderer-neutral host protocol for `ScriptBots`.
//!
//! The public protocol remains renderer, server, storage-engine, and platform-
//! runtime neutral. [`HostCore`] implements that protocol as a deterministic
//! synchronous state machine: it owns one [`scriptbots_core::WorldState`] by
//! value, advances it only under injected time, and delegates native or browser
//! scheduling to platform adapters.

#![warn(missing_docs, unsafe_code)]

use arc_swap::ArcSwap;
use scriptbots_core::{
    AgentUid, BirthRecord, BrainInspectionLimits, BrainInspectionResponse, DeathRecord,
    DynamicAgentSnapshot, DynamicWorldSnapshot, Generation, HydrologyFlowDirection,
    PersistenceBatch, ResourceLedgerTick, ScriptBotsConfig, TerrainKind, Tick, TickCombatSummary,
    TickEvents, TickSummary,
};
use serde::{Deserialize, Serialize};
use std::{
    any::Any,
    cmp::Ordering,
    collections::{BinaryHeap, VecDeque},
    fmt,
    mem::size_of,
    sync::{Arc, Mutex},
};
use thiserror::Error;

mod serde_arc {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::Arc;

    pub fn serialize<T, S>(value: &Arc<T>, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Serialize,
        S: Serializer,
    {
        value.as_ref().serialize(serializer)
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<Arc<T>, D::Error>
    where
        T: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Arc::new)
    }
}

mod serde_optional_arc {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::Arc;

    #[allow(
        clippy::ref_option,
        reason = "serde with-modules require a serializer accepting a shared reference to the field type"
    )]
    pub fn serialize<T, S>(value: &Option<Arc<T>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Serialize,
        S: Serializer,
    {
        value.as_deref().serialize(serializer)
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<Option<Arc<T>>, D::Error>
    where
        T: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(|value| value.map(Arc::new))
    }
}

pub mod archipelago;
pub mod channel;
pub mod host_core;
pub mod migrator;
pub mod native;

const MAX_EVENT_PAGE_SIZE: usize = 4_096;
const DEFAULT_PROJECTION_CACHE_BYTES: usize = 64 * 1024 * 1024;

pub use archipelago::{
    Archipelago, ArchipelagoConfig, ArchipelagoError, BarrierReport, IslandBarrierReport, IslandId,
    IslandMeta, IslandSpec, MAX_ISLANDS, StepTopology, Topology,
};
pub use host_core::{
    HostCore, HostCoreBuildError, HostCoreOptions, LocalHostPort, VolatileJournal,
};
pub use native::{FixedDeadlineHost, NativeDriveReceipt, NativeDriveTrigger, NativeScheduleError};

#[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
pub use native::{
    NativeControl, NativeIngressError, NativeLifecycleMetrics, NativeRunError, NativeRunOutcome,
    NativeRunner, NativeRunnerOptions, NativeRunnerOptionsError, NativeWakeResult,
};

macro_rules! monotonic_newtype {
    ($(#[$metadata:meta])* $name:ident) => {
        $(#[$metadata])*
        #[derive(
            Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize,
            Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Construct a value from its protocol representation.
            #[must_use]
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            /// Return the protocol representation.
            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }

            /// Return the following value, or `None` at the end of the domain.
            #[must_use]
            pub const fn checked_next(self) -> Option<Self> {
                match self.0.checked_add(1) {
                    Some(value) => Some(Self(value)),
                    None => None,
                }
            }
        }
    };
}

fn parse_fixed_lower_hex_u128(encoded: &str) -> Option<u128> {
    if encoded.len() != 32
        || !encoded
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return None;
    }
    u128::from_str_radix(encoded, 16).ok()
}

/// Stable namespace for one durable simulation run.
///
/// Every run-scoped scientific and provenance record uses this identifier as
/// its outer database key. The canonical text and serialization form is
/// exactly 32 lowercase hexadecimal characters, including leading zeroes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunId(u128);

impl RunId {
    /// Construct an identifier from its protocol representation.
    #[must_use]
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// Construct an identifier from a stable namespace and its local sequence.
    ///
    /// The namespace occupies the high 64 bits and the sequence occupies the
    /// low 64 bits, so separate allocators can issue run identifiers without
    /// sharing a counter.
    #[must_use]
    pub fn from_namespace_sequence(namespace: u64, sequence: u64) -> Self {
        Self((u128::from(namespace) << 64) | u128::from(sequence))
    }

    /// Return the protocol representation.
    #[must_use]
    pub const fn get(self) -> u128 {
        self.0
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:032x}", self.0)
    }
}

/// Error returned when a run identifier is not in canonical wire form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("run id must be exactly 32 lowercase hexadecimal characters")]
pub struct RunIdParseError;

impl std::str::FromStr for RunId {
    type Err = RunIdParseError;

    fn from_str(encoded: &str) -> Result<Self, Self::Err> {
        parse_fixed_lower_hex_u128(encoded)
            .map(Self)
            .ok_or(RunIdParseError)
    }
}

impl Serialize for RunId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for RunId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// Stable idempotency key supplied by a client for one logical command.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommandId(u128);

impl CommandId {
    /// Construct an identifier from its protocol representation.
    #[must_use]
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// Construct a collision-resistant identifier from a stable client namespace and sequence.
    #[must_use]
    pub fn from_client_sequence(client_namespace: u64, sequence: u64) -> Self {
        Self((u128::from(client_namespace) << 64) | u128::from(sequence))
    }

    /// Return the protocol representation.
    #[must_use]
    pub const fn get(self) -> u128 {
        self.0
    }

    /// Stable client namespace carried in the high 64 bits of this identifier.
    ///
    /// This is the source identity used by command-lifecycle evidence until a
    /// transport supplies a richer authenticated client identity.
    #[must_use]
    pub const fn client_namespace(self) -> u64 {
        let bytes = self.0.to_be_bytes();
        u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }

    /// Client-local sequence carried in the low 64 bits of this identifier.
    #[must_use]
    pub const fn client_sequence(self) -> u64 {
        let bytes = self.0.to_be_bytes();
        u64::from_be_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ])
    }
}

impl fmt::Display for CommandId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:032x}", self.0)
    }
}

impl Serialize for CommandId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for CommandId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let encoded = String::deserialize(deserializer)?;
        parse_fixed_lower_hex_u128(&encoded)
            .map(Self)
            .ok_or_else(|| {
                serde::de::Error::custom(
                    "command id must be exactly 32 lowercase hexadecimal characters",
                )
            })
    }
}

monotonic_newtype!(
    /// Stable identity shared by one host's ingress port and manual driver.
    HostSessionId
);

/// Stable identity of one immutable host-journal batch.
///
/// The host-local sequence is paired with the host session so retries remain
/// unambiguous even when several hosts share one journal adapter.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct JournalBatchId {
    session_id: HostSessionId,
    sequence: u64,
}

impl JournalBatchId {
    /// Construct a host-scoped journal identity.
    #[must_use]
    pub const fn new(session_id: HostSessionId, sequence: u64) -> Self {
        Self {
            session_id,
            sequence,
        }
    }

    /// Host session that allocated this identity.
    #[must_use]
    pub const fn session_id(self) -> HostSessionId {
        self.session_id
    }

    /// Monotonic journal sequence within the host session.
    #[must_use]
    pub const fn sequence(self) -> u64 {
        self.sequence
    }
}

monotonic_newtype!(
    /// Total order assigned to successfully admitted commands.
    AdmissionSequence
);
monotonic_newtype!(
    /// Revision of externally visible control state.
    ControlRevision
);
monotonic_newtype!(
    /// Revision of scientific world state.
    ScientificRevision
);
monotonic_newtype!(
    /// Revision of the active simulation configuration.
    ConfigRevision
);
monotonic_newtype!(
    /// Revision of the immutable snapshot publication stream.
    SnapshotRevision
);
monotonic_newtype!(
    /// Revision of one immutable renderer layer's exact content.
    LayerRevision
);
monotonic_newtype!(
    /// Sequence number in the lossless scientific-event stream.
    EventSequence
);
monotonic_newtype!(
    /// Sequence number in the bounded ephemeral host-notification stream.
    ProtocolEventSequence
);
monotonic_newtype!(
    /// Stable frontend identity used only to isolate presentation projections.
    ProjectionClientId
);
monotonic_newtype!(
    /// Client-owned revision of one separately requested presentation detail payload.
    ProjectionRequestRevision
);

/// Monotonic time supplied by a deterministic or browser-owned driver.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ManualInstant(u64);

impl ManualInstant {
    /// Construct an instant measured in monotonically increasing nanoseconds.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Return the instant in nanoseconds.
    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }
}

/// The independent revision domains observed at one host boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostRevisions {
    /// Revision of playback, lifecycle, and command-visible control state.
    pub control: ControlRevision,
    /// Revision of the scientific simulation state.
    pub scientific: ScientificRevision,
    /// Revision of the active configuration.
    pub config: ConfigRevision,
}

/// Playback state included in every host snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlaybackSnapshot {
    /// Whether automatic ticking is paused.
    pub paused: bool,
    /// Requested tick-rate multiplier.
    pub speed_multiplier: f32,
}

impl Default for PlaybackSnapshot {
    fn default() -> Self {
        Self {
            paused: false,
            speed_multiplier: 1.0,
        }
    }
}

/// Lifecycle visible to clients without exposing the concrete host implementation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostLifecycle {
    /// The host accepts commands and may advance science.
    #[default]
    Running,
    /// Shutdown was admitted but finalization is not complete.
    Stopping,
    /// Finalization completed and no new command may be admitted.
    Stopped,
}

/// Independent exact-content revisions for renderer-facing world layers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotLayerRevisions {
    /// Revision of the terrain tile payload.
    pub terrain: LayerRevision,
    /// Revision of the dense food payload.
    pub food: LayerRevision,
    /// Revision of the optional hydrology payload.
    pub hydrology: LayerRevision,
}

/// Renderer-neutral copy of one terrain tile.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerrainTileSnapshot {
    /// Terrain classification.
    pub kind: TerrainKind,
    /// Normalized elevation.
    pub elevation: f32,
    /// Normalized moisture.
    pub moisture: f32,
    /// Renderer accent value.
    pub accent: f32,
    /// Fertility contribution applied by the scientific kernel.
    pub fertility_bias: f32,
    /// Temperature contribution applied by the scientific kernel.
    pub temperature_bias: f32,
    /// Stable palette lookup index.
    pub palette_index: u16,
}

/// Immutable terrain payload shared across render publications until its content changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerrainLayerSnapshot {
    /// Tile width.
    pub width: u32,
    /// Tile height.
    pub height: u32,
    /// Tile edge length in world units.
    pub cell_size: u32,
    /// Dense row-major terrain tiles.
    pub tiles: Vec<TerrainTileSnapshot>,
}

/// Immutable dense food payload shared until one exact cell bit-pattern changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FoodLayerSnapshot {
    /// Grid width.
    pub width: u32,
    /// Grid height.
    pub height: u32,
    /// Dense row-major food values.
    pub cells: Vec<f32>,
}

/// Renderer-neutral copy of one hydrology policy tile.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HydrologyTileSnapshot {
    /// Terrain permeability.
    pub permeability: f32,
    /// Runoff bias.
    pub runoff_bias: f32,
    /// Basin ordering rank.
    pub basin_rank: f32,
    /// Channel-selection priority.
    pub channel_priority: f32,
    /// Agent swim-cost multiplier.
    pub swim_cost: f32,
}

/// Immutable hydrology payload shared until one exact field bit-pattern changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HydrologyLayerSnapshot {
    /// Grid width.
    pub width: u32,
    /// Grid height.
    pub height: u32,
    /// Dense row-major hydrology policy tiles.
    pub tiles: Vec<HydrologyTileSnapshot>,
    /// Precomputed flow direction per cell.
    pub flow_directions: Vec<HydrologyFlowDirection>,
    /// Precomputed flow accumulation per cell.
    pub accumulation: Vec<f32>,
    /// Precomputed spill elevation per cell.
    pub spill_elevation: Vec<f32>,
    /// Stable basin identity per cell.
    pub basin_ids: Vec<u32>,
    /// Current water depth per cell.
    pub water_depth: Vec<f32>,
}

/// Arc-shared world layers captured coherently with one render publication.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotLayers {
    /// Exact-content revisions paired with the payload pointers.
    pub revisions: SnapshotLayerRevisions,
    /// Terrain payload; pointer-stable while `revisions.terrain` is unchanged.
    #[serde(with = "serde_arc")]
    pub terrain: Arc<TerrainLayerSnapshot>,
    /// Food payload; pointer-stable while `revisions.food` is unchanged.
    #[serde(with = "serde_arc")]
    pub food: Arc<FoodLayerSnapshot>,
    /// Hydrology payload; pointer-stable while `revisions.hydrology` is unchanged.
    #[serde(with = "serde_optional_arc")]
    pub hydrology: Option<Arc<HydrologyLayerSnapshot>>,
}

/// Deterministic bulk-payload accounting for one render-snapshot build.
///
/// Counts deliberately exclude allocator headers and small control structs. They cover the
/// large vectors whose cost scales with agent or layer cardinality, making allocation and byte
/// regressions comparable across native allocators.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotBuildStats {
    /// Dynamic agents copied into this publication.
    pub dynamic_agent_count: usize,
    /// Bounded tick summaries copied into the projection history.
    pub summary_history_count: usize,
    /// Bulk vector allocations created by this build.
    pub bulk_allocations: usize,
    /// Capacity bytes newly allocated for dynamic agents and changed layers.
    pub newly_allocated_capacity_bytes: usize,
    /// Capacity bytes reused through layer Arcs.
    pub reused_layer_capacity_bytes: usize,
    /// Total capacity bytes referenced by the dynamic and layer payloads.
    pub total_payload_capacity_bytes: usize,
}

/// Immutable renderer-neutral publication from the sole-owner host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderSnapshot {
    /// Stable identity of the publishing host session.
    pub session_id: HostSessionId,
    /// Monotonic publication revision within `session_id`.
    pub revision: SnapshotRevision,
    /// Revisions captured atomically with the payload.
    pub revisions: HostRevisions,
    /// Playback state captured at this boundary.
    pub playback: PlaybackSnapshot,
    /// Host lifecycle captured at this boundary.
    pub lifecycle: HostLifecycle,
    /// Queryable health captured at this boundary.
    pub health: HostHealth,
    /// Admitted envelopes still waiting behind the owner boundary.
    pub command_queue_depth: usize,
    /// Most recently applied command, independent of journal durability.
    pub last_applied_command: Option<CommandId>,
    /// Exact latest completed `StepOutcome` summary; absent before the first completed tick.
    pub completed_summary: Option<TickSummary>,
    /// Bounded scientific summary history used by pure per-client chart projections.
    #[serde(with = "serde_arc")]
    pub summary_history: Arc<Vec<TickSummary>>,
    /// Content-revisioned Arc-shared terrain, food, and hydrology payloads.
    pub layers: SnapshotLayers,
    /// Deterministic payload allocation and byte accounting.
    pub build: SnapshotBuildStats,
    /// Compact renderer-neutral dynamic world projection.
    pub world: DynamicWorldSnapshot,
}

/// Hard bounds applied before a renderer-neutral projection allocates output buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionLimits {
    /// Maximum logical canvas cells in one request.
    pub max_canvas_cells: u32,
    /// Maximum stable agent identities carried by one client selection.
    pub max_selected_agents: u16,
    /// Maximum points emitted by one chart window.
    pub max_chart_points: u16,
    /// Maximum agents retained by one top-K panel.
    pub max_top_k: u16,
    /// Maximum visible-agent records emitted by one viewport.
    pub max_visible_agents: u32,
}

impl Default for ProjectionLimits {
    fn default() -> Self {
        Self {
            max_canvas_cells: 1_048_576,
            max_selected_agents: 256,
            max_chart_points: 4_096,
            max_top_k: 1_024,
            max_visible_agents: 100_000,
        }
    }
}

/// Logical viewport dimensions independent of a concrete graphics toolkit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionViewport {
    /// Logical output width in cells or pixels.
    pub width: u32,
    /// Logical output height in cells or pixels.
    pub height: u32,
}

/// World-space camera used by the pure projection transform.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProjectionCamera {
    /// Camera center in world coordinates.
    pub center: [f32; 2],
    /// Positive finite magnification; `1.0` fits the whole world.
    pub zoom: f32,
}

/// Client-owned selection that never mutates scientific or global host state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionSelection {
    /// Primary agent shown by an inspector, when present in the source snapshot.
    pub focused: Option<AgentUid>,
    /// Client-local selected identities.
    pub selected: Vec<AgentUid>,
}

/// Amount of compact per-agent detail requested for visible and selected agents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionDetail {
    /// Position and display identity only.
    #[default]
    Minimal,
    /// Include health, energy, age, generation, diet tendency, and brain key.
    Vitals,
    /// Include vitals plus velocity, heading, spike extension, and boost state.
    Kinematics,
}

/// Scalar used by one bounded top-K projection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionRanking {
    /// Rank by current runtime energy.
    #[default]
    Energy,
    /// Rank by current health.
    Health,
    /// Rank by completed scientific age.
    Age,
    /// Rank by lineage generation.
    Generation,
}

/// Complete normalized presentation request for one client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectionRequest {
    /// Stable identity used for cache isolation and accounting only.
    pub client_id: ProjectionClientId,
    /// Logical output dimensions.
    pub viewport: ProjectionViewport,
    /// World-space camera.
    pub camera: ProjectionCamera,
    /// Client-local focus and selection.
    pub selection: ProjectionSelection,
    /// Requested compact detail level.
    pub detail: ProjectionDetail,
    /// Number of most-recent summaries eligible for the chart.
    pub chart_window: u32,
    /// Maximum downsampled points emitted for the chart.
    pub chart_points: u16,
    /// Maximum agents emitted by the ranking panel.
    pub top_k: u16,
    /// Ranking used by the top-K panel.
    pub ranking: ProjectionRanking,
}

/// Separately requested selected-brain detail; never embedded in every world snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrainProjectionRequest {
    /// Stable client identity used only for presentation isolation.
    pub client_id: ProjectionClientId,
    /// Client-owned request revision returned verbatim in the response.
    pub revision: ProjectionRequestRevision,
    /// Stable scientific identities requested in client order.
    #[serde(deserialize_with = "scriptbots_core::deserialize_brain_inspection_targets")]
    pub targets: Vec<AgentUid>,
    /// Producer-side structural, work, and payload limits.
    pub limits: BrainInspectionLimits,
}

impl BrainProjectionRequest {
    /// Construct one focused-agent request under the project hard limits.
    #[must_use]
    pub fn focused(
        client_id: ProjectionClientId,
        revision: ProjectionRequestRevision,
        target: AgentUid,
    ) -> Self {
        Self {
            client_id,
            revision,
            targets: vec![target],
            limits: BrainInspectionLimits::hard(),
        }
    }
}

/// Source identity included in every projection and cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionSourceKey {
    /// Host session that owns the source snapshot.
    pub session_id: HostSessionId,
    /// Immutable snapshot publication revision.
    pub snapshot: SnapshotRevision,
    /// Independent host revision domains.
    pub host: HostRevisions,
    /// Independent static-layer content revisions.
    pub layers: SnapshotLayerRevisions,
}

/// Exact source identity for a synchronous selected-brain detail response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrainProjectionSource {
    /// Host session that owns both world state and the published snapshot stream.
    pub session_id: HostSessionId,
    /// Latest immutable snapshot revision visible when inspection occurred.
    pub published_snapshot: SnapshotRevision,
    /// Host revisions carried by that latest immutable snapshot.
    pub published_host: HostRevisions,
    /// Current owner revisions inspected synchronously.
    pub inspected_host: HostRevisions,
    /// Exact completed world tick inspected synchronously.
    pub inspected_tick: Tick,
}

impl BrainProjectionSource {
    /// Whether this detail can be paired with the supplied immutable snapshot without staleness.
    #[must_use]
    pub fn matches_snapshot(self, snapshot: &RenderSnapshot) -> bool {
        let published_source = (self.published_snapshot, self.published_host);
        let snapshot_source = (snapshot.revision, snapshot.revisions);
        let inspected_source = (self.inspected_host, self.inspected_tick.0);
        let snapshot_state = (snapshot.revisions, snapshot.world.tick);

        self.session_id == snapshot.session_id
            && published_source == snapshot_source
            && inspected_source == snapshot_state
    }
}

/// Immutable, client-isolated, revisioned selected-brain projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainProjection {
    /// Exact current and latest-published source revisions.
    pub source: BrainProjectionSource,
    /// Canonical request returned for cache and response correlation.
    pub request: BrainProjectionRequest,
    /// Bounded current evaluator detail from core.
    pub inspection: BrainInspectionResponse,
}

/// Pure world-to-canvas transform returned for frontend reuse and picking.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProjectionTransform {
    /// Camera center used after finite-zero canonicalization.
    pub center: [f32; 2],
    /// World units visible across the logical canvas.
    pub visible_world: [f32; 2],
    /// Logical output dimensions.
    pub viewport: ProjectionViewport,
    /// Logical cells per world unit.
    pub scale: f32,
}

/// Optional compact vitals and kinematics for one projected agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectedAgentDetail {
    /// Current health.
    pub health: f32,
    /// Current runtime energy.
    pub energy: f32,
    /// Completed scientific age.
    pub age: u32,
    /// Heritable lineage generation.
    pub generation: Generation,
    /// Continuous diet tendency.
    pub herbivore_tendency: f32,
    /// Stable brain-registry key when present.
    pub brain_key: Option<u64>,
    /// Velocity when kinematics were requested.
    pub velocity: Option<[f32; 2]>,
    /// Heading when kinematics were requested.
    pub heading: Option<f32>,
    /// Spike extension when kinematics were requested.
    pub spike_length: Option<f32>,
    /// Movement boost state when kinematics were requested.
    pub boost: Option<bool>,
}

/// One compact visible agent projected into logical canvas coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectedAgent {
    /// Stable logical identity used by client-local selection.
    pub uid: AgentUid,
    /// Transient generational handle retained for live control adapters.
    pub handle: u64,
    /// World-space position from the immutable source.
    pub world_position: [f32; 2],
    /// Logical canvas position.
    pub canvas_position: [f64; 2],
    /// Toroidal image offset chosen relative to the camera.
    pub wrap_offset: [f64; 2],
    /// Renderer-neutral linear RGB color.
    pub color: [f32; 3],
    /// Whether this client selected the agent.
    pub selected: bool,
    /// Whether this is the client's primary focused agent.
    pub focused: bool,
    /// Optional requested detail.
    pub detail: Option<ProjectedAgentDetail>,
}

/// Aggregate for one logical canvas cell.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ProjectionCell {
    /// Agents projected into this cell.
    pub agent_count: u32,
    /// Sum of current energy in this cell.
    pub total_energy: f32,
    /// Sum of current health in this cell.
    pub total_health: f32,
}

/// One entry in a deterministic bounded ranking panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectedRankingEntry {
    /// Stable logical identity.
    pub uid: AgentUid,
    /// Ranked scalar value.
    pub value: f64,
}

#[derive(Debug)]
struct RankingHeapEntry(ProjectedRankingEntry);

impl PartialEq for RankingHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.uid == other.0.uid && self.0.value.total_cmp(&other.0.value).is_eq()
    }
}

impl Eq for RankingHeapEntry {}

impl PartialOrd for RankingHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankingHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .value
            .total_cmp(&other.0.value)
            .reverse()
            .then_with(|| self.0.uid.cmp(&other.0.uid))
    }
}

/// Structural allocation and scaling evidence for one pure projection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionBuildStats {
    /// Source agents examined exactly once.
    pub agents_examined: usize,
    /// Visible agents emitted.
    pub visible_agents: usize,
    /// Client-selected agents emitted, including offscreen selections.
    pub selected_agents: usize,
    /// Logical canvas cells allocated.
    pub canvas_cells: usize,
    /// Maximum elements held by the bounded top-K scratch/output.
    pub top_k_peak: usize,
    /// Source history samples examined for the requested window.
    pub chart_samples_examined: usize,
    /// Downsampled chart points emitted.
    pub chart_points_emitted: usize,
    /// Capacity bytes owned by bulk projection vectors.
    pub output_capacity_bytes: usize,
}

/// Complete deterministic renderer-neutral projection for one client request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClientProjection {
    /// Exact immutable source identity.
    pub source: ProjectionSourceKey,
    /// Canonical validated request used to build this result.
    pub request: ProjectionRequest,
    /// World-to-canvas transform.
    pub transform: ProjectionTransform,
    /// Visible agents in deterministic source order.
    pub visible_agents: Vec<ProjectedAgent>,
    /// Client-selected agents in deterministic source order, including offscreen selections.
    pub selected_agents: Vec<ProjectedAgent>,
    /// Dense row-major logical canvas aggregates.
    pub cells: Vec<ProjectionCell>,
    /// Requested focused-agent detail, or `None` when absent/unrequested.
    pub focused_agent: Option<ProjectedAgent>,
    /// Deterministic bounded ranking.
    pub top_agents: Vec<ProjectedRankingEntry>,
    /// Deterministically downsampled recent scientific summaries.
    pub chart: Vec<TickSummary>,
    /// Whether the requested chart window began before retained source history.
    pub chart_truncated: bool,
    /// Structural resource evidence.
    pub build: ProjectionBuildStats,
}

/// Validation or bounded-allocation failure for a projection request.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProjectionError {
    /// One camera field was non-finite or zoom was not positive.
    #[error("projection camera must have finite coordinates and positive finite zoom")]
    InvalidCamera,
    /// A zero-sized viewport was requested.
    #[error("projection viewport dimensions must both be nonzero")]
    EmptyViewport,
    /// The immutable source declared a zero-sized world.
    #[error("projection source world dimensions must both be nonzero")]
    InvalidSourceDimensions,
    /// Finite inputs produced an unrepresentable world-to-canvas scale.
    #[error("projection camera and viewport must produce a finite positive scale")]
    ScaleOutOfRange,
    /// Canvas area exceeded its declared bound or integer domain.
    #[error("projection canvas requires {requested} cells, exceeding limit {limit}")]
    CanvasTooLarge {
        /// Requested logical cells.
        requested: u64,
        /// Configured maximum logical cells.
        limit: u32,
    },
    /// Client selection exceeded its declared bound.
    #[error("projection selection contains {requested} identities, exceeding limit {limit}")]
    SelectionTooLarge {
        /// Requested identities before canonical deduplication.
        requested: usize,
        /// Configured maximum identities.
        limit: u16,
    },
    /// Chart output exceeded its declared bound.
    #[error("projection chart requests {requested} points, exceeding limit {limit}")]
    ChartTooLarge {
        /// Requested output points.
        requested: u16,
        /// Configured maximum output points.
        limit: u16,
    },
    /// Ranking output exceeded its declared bound.
    #[error("projection top-K requests {requested} agents, exceeding limit {limit}")]
    TopKTooLarge {
        /// Requested retained agents.
        requested: u16,
        /// Configured maximum retained agents.
        limit: u16,
    },
    /// Visible output exceeded its declared bound.
    #[error("projection emitted more than {limit} visible agents")]
    VisibleAgentsTooLarge {
        /// Configured maximum visible records.
        limit: u32,
    },
    /// A bounded keyed cache was constructed with zero capacity.
    #[error("projection cache capacity must be nonzero")]
    EmptyCache,
    /// A bounded keyed cache was constructed without any retained-payload budget.
    #[error("projection cache byte capacity must be nonzero")]
    EmptyCacheByteCapacity,
}

const fn canonical_f32(value: f32) -> f32 {
    if value == 0.0 { 0.0 } else { value }
}

fn normalize_projection_request(
    request: &ProjectionRequest,
    limits: ProjectionLimits,
) -> Result<ProjectionRequest, ProjectionError> {
    if request.viewport.width == 0 || request.viewport.height == 0 {
        return Err(ProjectionError::EmptyViewport);
    }
    let canvas_cells =
        u64::from(request.viewport.width).saturating_mul(u64::from(request.viewport.height));
    if canvas_cells > u64::from(limits.max_canvas_cells) {
        return Err(ProjectionError::CanvasTooLarge {
            requested: canvas_cells,
            limit: limits.max_canvas_cells,
        });
    }
    if !request.camera.center.iter().all(|value| value.is_finite())
        || !request.camera.zoom.is_finite()
        || request.camera.zoom <= 0.0
    {
        return Err(ProjectionError::InvalidCamera);
    }
    if request.chart_points > limits.max_chart_points {
        return Err(ProjectionError::ChartTooLarge {
            requested: request.chart_points,
            limit: limits.max_chart_points,
        });
    }
    if request.top_k > limits.max_top_k {
        return Err(ProjectionError::TopKTooLarge {
            requested: request.top_k,
            limit: limits.max_top_k,
        });
    }
    if request.selection.selected.len() > usize::from(limits.max_selected_agents) {
        return Err(ProjectionError::SelectionTooLarge {
            requested: request.selection.selected.len(),
            limit: limits.max_selected_agents,
        });
    }
    let mut normalized = request.clone();
    normalized.camera.center = normalized.camera.center.map(canonical_f32);
    normalized.camera.zoom = canonical_f32(normalized.camera.zoom);
    normalized.selection.selected.sort_unstable();
    normalized.selection.selected.dedup();
    Ok(normalized)
}

const fn projection_source_key(snapshot: &RenderSnapshot) -> ProjectionSourceKey {
    ProjectionSourceKey {
        session_id: snapshot.session_id,
        snapshot: snapshot.revision,
        host: snapshot.revisions,
        layers: snapshot.layers.revisions,
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the wrapped delta is bounded to one f32 world extent before conversion"
)]
fn nearest_wrapped_delta(position: f32, center: f32, extent: f32) -> (f32, f64) {
    let raw = f64::from(position) - f64::from(center);
    let extent = f64::from(extent);
    let half = extent * 0.5;
    let wrapped = (raw + half).rem_euclid(extent) - half;
    (wrapped as f32, canonical_f64(wrapped - raw))
}

const fn canonical_f64(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn projected_detail(
    agent: &DynamicAgentSnapshot,
    detail: ProjectionDetail,
) -> Option<ProjectedAgentDetail> {
    if detail == ProjectionDetail::Minimal {
        return None;
    }
    let kinematics = detail == ProjectionDetail::Kinematics;
    Some(ProjectedAgentDetail {
        health: agent.health,
        energy: agent.energy,
        age: agent.age,
        generation: agent.generation,
        herbivore_tendency: agent.herbivore_tendency,
        brain_key: agent.brain_key,
        velocity: kinematics.then_some(agent.velocity),
        heading: kinematics.then_some(agent.heading),
        spike_length: kinematics.then_some(agent.spike_length),
        boost: kinematics.then_some(agent.boost),
    })
}

fn ranking_value(agent: &DynamicAgentSnapshot, ranking: ProjectionRanking) -> f64 {
    match ranking {
        ProjectionRanking::Energy => f64::from(agent.energy),
        ProjectionRanking::Health => f64::from(agent.health),
        ProjectionRanking::Age => f64::from(agent.age),
        ProjectionRanking::Generation => f64::from(agent.generation.0),
    }
}

fn build_chart(
    snapshot: &RenderSnapshot,
    window: u32,
    points: u16,
) -> (Vec<TickSummary>, bool, usize) {
    if window == 0 || points == 0 || snapshot.summary_history.is_empty() {
        return (Vec::new(), false, 0);
    }
    let requested_window = usize::try_from(window).unwrap_or(usize::MAX);
    let retained = snapshot.summary_history.len();
    let examined = retained.min(requested_window);
    let start = retained.saturating_sub(examined);
    let source = &snapshot.summary_history[start..];
    let point_limit = usize::from(points).min(source.len());
    let mut chart = Vec::with_capacity(point_limit);
    match point_limit {
        0 => {}
        1 => chart.push(source[source.len() - 1].clone()),
        count if count == source.len() => chart.extend_from_slice(source),
        count => {
            for output_index in 0..count {
                let source_index = output_index * (source.len() - 1) / (count - 1);
                chart.push(source[source_index].clone());
            }
        }
    }
    (chart, requested_window > retained, examined)
}

/// Build one deterministic projection from an immutable source and client-owned request.
///
/// # Errors
///
/// Returns [`ProjectionError`] before oversized or invalid output is allocated.
pub fn project_snapshot(
    snapshot: &RenderSnapshot,
    request: &ProjectionRequest,
    limits: ProjectionLimits,
) -> Result<ClientProjection, ProjectionError> {
    let request = normalize_projection_request(request, limits)?;
    project_normalized_snapshot(snapshot, request, limits)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "renderer dimensions intentionally narrow into validated presentation coordinates while one pass keeps projection bounds auditable"
)]
fn project_normalized_snapshot(
    snapshot: &RenderSnapshot,
    request: ProjectionRequest,
    limits: ProjectionLimits,
) -> Result<ClientProjection, ProjectionError> {
    if snapshot.world.world.width == 0 || snapshot.world.world.height == 0 {
        return Err(ProjectionError::InvalidSourceDimensions);
    }
    let world_width = snapshot.world.world.width as f32;
    let world_height = snapshot.world.world.height as f32;
    let viewport_width = request.viewport.width as f32;
    let viewport_height = request.viewport.height as f32;
    let base_scale = (viewport_width / world_width).min(viewport_height / world_height);
    let scale = base_scale * request.camera.zoom;
    let visible_world = [viewport_width / scale, viewport_height / scale];
    if !scale.is_finite() || scale <= 0.0 || !visible_world.iter().all(|extent| extent.is_finite())
    {
        return Err(ProjectionError::ScaleOutOfRange);
    }
    let transform = ProjectionTransform {
        center: request.camera.center,
        visible_world,
        viewport: request.viewport,
        scale,
    };
    let canvas_len =
        usize::try_from(u64::from(request.viewport.width) * u64::from(request.viewport.height))
            .map_err(|_| ProjectionError::CanvasTooLarge {
                requested: u64::from(request.viewport.width) * u64::from(request.viewport.height),
                limit: limits.max_canvas_cells,
            })?;
    let mut cells = vec![ProjectionCell::default(); canvas_len];
    let selected = &request.selection.selected;
    let mut visible_agents = Vec::new();
    let mut selected_agents = Vec::with_capacity(selected.len());
    let mut focused_agent = None;
    let top_k = usize::from(request.top_k);
    let mut top_heap = BinaryHeap::with_capacity(top_k);

    for agent in &snapshot.world.agents {
        let candidate = RankingHeapEntry(ProjectedRankingEntry {
            uid: agent.uid,
            value: ranking_value(agent, request.ranking),
        });
        if top_k != 0 {
            if top_heap.len() < top_k {
                top_heap.push(candidate);
            } else if top_heap
                .peek()
                .is_some_and(|worst| candidate.cmp(worst).is_lt())
            {
                top_heap.pop();
                top_heap.push(candidate);
            }
        }

        let (delta_x, wrap_x) =
            nearest_wrapped_delta(agent.position[0], request.camera.center[0], world_width);
        let (delta_y, wrap_y) =
            nearest_wrapped_delta(agent.position[1], request.camera.center[1], world_height);
        let canvas_x =
            f64::from(delta_x).mul_add(f64::from(scale), f64::from(viewport_width) * 0.5);
        let canvas_y =
            f64::from(delta_y).mul_add(f64::from(scale), f64::from(viewport_height) * 0.5);
        let is_selected = selected.binary_search(&agent.uid).is_ok();
        let is_focused = request.selection.focused == Some(agent.uid);
        let projected = ProjectedAgent {
            uid: agent.uid,
            handle: agent.id,
            world_position: agent.position,
            canvas_position: [canvas_x, canvas_y],
            wrap_offset: [wrap_x, wrap_y],
            color: agent.color,
            selected: is_selected,
            focused: is_focused,
            detail: projected_detail(agent, request.detail),
        };
        if is_focused {
            focused_agent = Some(projected.clone());
        }
        if is_selected {
            selected_agents.push(projected.clone());
        }
        if !(0.0..f64::from(viewport_width)).contains(&canvas_x)
            || !(0.0..f64::from(viewport_height)).contains(&canvas_y)
        {
            continue;
        }
        if visible_agents.len() >= usize::try_from(limits.max_visible_agents).unwrap_or(usize::MAX)
        {
            return Err(ProjectionError::VisibleAgentsTooLarge {
                limit: limits.max_visible_agents,
            });
        }
        let cell_x = (canvas_x.floor() as u32).min(request.viewport.width - 1);
        let cell_y = (canvas_y.floor() as u32).min(request.viewport.height - 1);
        let cell_index_u64 =
            u64::from(cell_y) * u64::from(request.viewport.width) + u64::from(cell_x);
        let cell_index =
            usize::try_from(cell_index_u64).map_err(|_| ProjectionError::CanvasTooLarge {
                requested: cell_index_u64.saturating_add(1),
                limit: limits.max_canvas_cells,
            })?;
        let cell = cells
            .get_mut(cell_index)
            .ok_or_else(|| ProjectionError::CanvasTooLarge {
                requested: cell_index_u64.saturating_add(1),
                limit: limits.max_canvas_cells,
            })?;
        cell.agent_count = cell.agent_count.saturating_add(1);
        cell.total_energy += agent.energy;
        cell.total_health += agent.health;
        visible_agents.push(projected);
    }

    let mut top_heap = top_heap.into_vec();
    top_heap.sort_unstable();
    let top_agents = top_heap
        .into_iter()
        .map(|entry| entry.0)
        .collect::<Vec<_>>();
    let (chart, chart_truncated, chart_samples_examined) =
        build_chart(snapshot, request.chart_window, request.chart_points);
    let output_capacity_bytes = visible_agents
        .capacity()
        .saturating_mul(size_of::<ProjectedAgent>())
        .saturating_add(
            selected_agents
                .capacity()
                .saturating_mul(size_of::<ProjectedAgent>()),
        )
        .saturating_add(cells.capacity().saturating_mul(size_of::<ProjectionCell>()))
        .saturating_add(
            top_agents
                .capacity()
                .saturating_mul(size_of::<ProjectedRankingEntry>()),
        )
        .saturating_add(chart.capacity().saturating_mul(size_of::<TickSummary>()))
        .saturating_add(
            request
                .selection
                .selected
                .capacity()
                .saturating_mul(size_of::<AgentUid>()),
        );
    Ok(ClientProjection {
        source: projection_source_key(snapshot),
        request,
        transform,
        build: ProjectionBuildStats {
            agents_examined: snapshot.world.agents.len(),
            visible_agents: visible_agents.len(),
            selected_agents: selected_agents.len(),
            canvas_cells: cells.len(),
            top_k_peak: top_agents.len(),
            chart_samples_examined,
            chart_points_emitted: chart.len(),
            output_capacity_bytes,
        },
        visible_agents,
        selected_agents,
        cells,
        focused_agent,
        top_agents,
        chart,
        chart_truncated,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct ProjectionCacheIdentity {
    source: ProjectionSourceKey,
    request: ProjectionRequest,
    limits: ProjectionLimits,
}

#[derive(Debug)]
struct ProjectionCacheEntry {
    source: ProjectionSourceKey,
    limits: ProjectionLimits,
    projection: Arc<ClientProjection>,
}

/// Bounded keyed cache over the pure [`project_snapshot`] function.
pub struct ProjectionBroker {
    capacity: usize,
    byte_capacity: usize,
    retained_output_capacity_bytes: usize,
    entries: VecDeque<ProjectionCacheEntry>,
    hits: u64,
    misses: u64,
    evictions: u64,
    uncached_oversize: u64,
}

impl ProjectionBroker {
    /// Construct a cache retaining at most `capacity` complete projections.
    ///
    /// # Errors
    ///
    /// Returns [`ProjectionError::EmptyCache`] when `capacity` is zero.
    pub fn new(capacity: usize) -> Result<Self, ProjectionError> {
        Self::with_byte_capacity(capacity, DEFAULT_PROJECTION_CACHE_BYTES)
    }

    /// Construct an entry- and bulk-payload-byte-bounded projection cache.
    ///
    /// # Errors
    ///
    /// Returns a typed construction error when either bound is zero.
    pub fn with_byte_capacity(
        capacity: usize,
        byte_capacity: usize,
    ) -> Result<Self, ProjectionError> {
        if capacity == 0 {
            return Err(ProjectionError::EmptyCache);
        }
        if byte_capacity == 0 {
            return Err(ProjectionError::EmptyCacheByteCapacity);
        }
        Ok(Self {
            capacity,
            byte_capacity,
            retained_output_capacity_bytes: 0,
            entries: VecDeque::with_capacity(capacity),
            hits: 0,
            misses: 0,
            evictions: 0,
            uncached_oversize: 0,
        })
    }

    /// Return a pointer-reused exact hit or build and insert one pure projection.
    ///
    /// # Errors
    ///
    /// Propagates request validation and bounded-allocation errors.
    pub fn project(
        &mut self,
        snapshot: &RenderSnapshot,
        request: &ProjectionRequest,
        limits: ProjectionLimits,
    ) -> Result<Arc<ClientProjection>, ProjectionError> {
        let normalized = normalize_projection_request(request, limits)?;
        let identity = ProjectionCacheIdentity {
            source: projection_source_key(snapshot),
            request: normalized,
            limits,
        };
        if let Some(index) = self.entries.iter().position(|entry| {
            entry.source == identity.source
                && entry.limits == identity.limits
                && entry.projection.request == identity.request
        }) && let Some(entry) = self.entries.remove(index)
        {
            let projection = Arc::clone(&entry.projection);
            self.entries.push_back(entry);
            self.hits = self.hits.saturating_add(1);
            return Ok(projection);
        }
        let projection = Arc::new(project_normalized_snapshot(
            snapshot,
            identity.request.clone(),
            identity.limits,
        )?);
        let projection_bytes = projection.build.output_capacity_bytes;
        self.misses = self.misses.saturating_add(1);
        if projection_bytes > self.byte_capacity {
            self.uncached_oversize = self.uncached_oversize.saturating_add(1);
            return Ok(projection);
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= self.capacity
                || self
                    .retained_output_capacity_bytes
                    .saturating_add(projection_bytes)
                    > self.byte_capacity)
        {
            let Some(evicted) = self.entries.pop_front() else {
                break;
            };
            self.retained_output_capacity_bytes = self
                .retained_output_capacity_bytes
                .saturating_sub(evicted.projection.build.output_capacity_bytes);
            self.evictions = self.evictions.saturating_add(1);
        }
        self.retained_output_capacity_bytes = self
            .retained_output_capacity_bytes
            .saturating_add(projection_bytes);
        self.entries.push_back(ProjectionCacheEntry {
            source: identity.source,
            limits: identity.limits,
            projection: Arc::clone(&projection),
        });
        Ok(projection)
    }

    /// Current retained cache entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether this broker retains no projection.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Exact cache-hit count.
    #[must_use]
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// Exact cache-miss count.
    #[must_use]
    pub const fn misses(&self) -> u64 {
        self.misses
    }

    /// Exact bounded-entry eviction count.
    #[must_use]
    pub const fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Cache misses returned without retention because one result exceeded the byte budget.
    #[must_use]
    pub const fn uncached_oversize(&self) -> u64 {
        self.uncached_oversize
    }

    /// Configured upper bound for retained bulk projection payloads.
    #[must_use]
    pub const fn byte_capacity(&self) -> usize {
        self.byte_capacity
    }

    /// Sum of structural bulk-vector capacities retained by cached results.
    #[must_use]
    pub const fn retained_output_capacity_bytes(&self) -> usize {
        self.retained_output_capacity_bytes
    }
}

/// Cloneable, thread-safe latest-value read handle for render snapshots.
///
/// The hub retains exactly one current `Arc`. Every subscriber owns only scalar cursor state, so
/// a stalled subscriber may skip revisions but never creates a host-side backlog. Holding a
/// previously returned `Arc` intentionally keeps that one immutable value alive.
#[derive(Clone)]
pub struct SnapshotHub {
    session_id: HostSessionId,
    latest: Arc<ArcSwap<RenderSnapshot>>,
}

impl fmt::Debug for SnapshotHub {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SnapshotHub")
            .field("session_id", &self.session_id)
            .field("revision", &self.latest.load().revision)
            .finish_non_exhaustive()
    }
}

impl SnapshotHub {
    #[must_use]
    pub(crate) fn new(initial: Arc<RenderSnapshot>) -> Self {
        Self {
            session_id: initial.session_id,
            latest: Arc::new(ArcSwap::from(initial)),
        }
    }

    /// Stable host session published by this hub.
    #[must_use]
    pub const fn session_id(&self) -> HostSessionId {
        self.session_id
    }

    /// Load the newest complete immutable publication.
    #[must_use]
    pub fn latest(&self) -> Arc<RenderSnapshot> {
        self.latest.load_full()
    }

    /// Create an independent cursor that first observes the current publication.
    #[must_use]
    pub const fn subscribe(&self) -> SnapshotSubscription {
        SnapshotSubscription::current(self.session_id)
    }

    /// Reconnect after a publication already observed in this host session.
    #[must_use]
    pub const fn resume_after(&self, revision: SnapshotRevision) -> SnapshotSubscription {
        SnapshotSubscription::after(self.session_id, revision)
    }

    /// Load the newest publication and advance only the supplied cursor.
    pub fn poll_latest(
        &self,
        subscription: &mut SnapshotSubscription,
    ) -> Result<Option<Arc<RenderSnapshot>>, HostAccessError> {
        let snapshot = self.latest();
        if subscription.observe(self.session_id, snapshot.revision)? {
            Ok(Some(snapshot))
        } else {
            Ok(None)
        }
    }

    fn snapshot_after(&self, after: Option<SnapshotRevision>) -> Option<Arc<RenderSnapshot>> {
        let snapshot = self.latest();
        after
            .is_none_or(|revision| snapshot.revision > revision)
            .then_some(snapshot)
    }

    pub(crate) fn publish(&self, snapshot: Arc<RenderSnapshot>) -> Result<(), HostAccessError> {
        if snapshot.session_id != self.session_id {
            return Err(protocol_violation(
                "snapshot publisher changed its host session identity",
            ));
        }
        if snapshot.revision <= self.latest.load().revision {
            return Err(protocol_violation(
                "snapshot publisher did not advance its publication revision",
            ));
        }
        self.latest.store(snapshot);
        Ok(())
    }
}

/// A state-changing request understood by the runtime boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum HostCommand {
    /// Pause automatic simulation ticks.
    Pause,
    /// Resume automatic simulation ticks.
    Resume,
    /// Set the requested playback multiplier.
    SetSpeed(f32),
    /// Advance exactly one scientific tick while paused.
    Step,
    /// Atomically replace the active simulation configuration.
    UpdateConfig(Box<ScriptBotsConfig>),
    /// Begin orderly host shutdown.
    Shutdown,
}

impl HostCommand {
    /// Validate input that can be rejected before admission.
    pub fn validate(&self) -> Result<(), CommandValidationError> {
        match self {
            Self::SetSpeed(speed) if !speed.is_finite() || *speed < 0.0 => {
                Err(CommandValidationError::InvalidSpeed)
            }
            Self::UpdateConfig(config) => {
                config
                    .validate()
                    .map_err(|error| CommandValidationError::InvalidConfig {
                        message: error.to_string(),
                    })
            }
            _ => Ok(()),
        }
    }

    /// Whether this command's terminal lifecycle requires a journal acknowledgement.
    ///
    /// Lifecycle auditing is universal. Whether a boundary also carries
    /// scientific or persistence payloads is an independent concern.
    #[must_use]
    pub const fn requires_journal(&self) -> bool {
        match self {
            Self::Pause
            | Self::Resume
            | Self::SetSpeed(_)
            | Self::Step
            | Self::UpdateConfig(_)
            | Self::Shutdown => true,
        }
    }
}

/// A command plus its stable identity and optional control-revision compare-and-set guard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandEnvelope {
    /// Stable idempotency key. Retrying this id returns its existing status.
    pub command_id: CommandId,
    /// Reject unless this is the host's current control revision.
    pub expected_control_revision: Option<ControlRevision>,
    /// Reject unless this is the host's current scientific revision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_scientific_revision: Option<ScientificRevision>,
    /// Reject unless this is the host's current configuration revision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_config_revision: Option<ConfigRevision>,
    /// Requested operation.
    pub command: HostCommand,
}

impl CommandEnvelope {
    /// Construct an unguarded command envelope.
    #[must_use]
    pub const fn new(command_id: CommandId, command: HostCommand) -> Self {
        Self {
            command_id,
            expected_control_revision: None,
            expected_scientific_revision: None,
            expected_config_revision: None,
            command,
        }
    }

    /// Add an expected control revision for compare-and-set admission.
    #[must_use]
    pub const fn expecting_control_revision(mut self, revision: ControlRevision) -> Self {
        self.expected_control_revision = Some(revision);
        self
    }

    /// Add an expected scientific revision for ordered compare-and-set application.
    #[must_use]
    pub const fn expecting_scientific_revision(mut self, revision: ScientificRevision) -> Self {
        self.expected_scientific_revision = Some(revision);
        self
    }

    /// Add an expected configuration revision for ordered compare-and-set application.
    #[must_use]
    pub const fn expecting_config_revision(mut self, revision: ConfigRevision) -> Self {
        self.expected_config_revision = Some(revision);
        self
    }
}

/// Reason a command was rejected before admission or at its ordered application boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RejectionReason {
    /// The request failed protocol-level validation.
    Validation {
        /// Actionable validation detail.
        message: String,
    },
    /// The optimistic control-revision guard did not match.
    ControlRevisionConflict {
        /// Revision requested by the client.
        expected: ControlRevision,
        /// Current host revision.
        actual: ControlRevision,
    },
    /// The optimistic scientific-revision guard did not match.
    ScientificRevisionConflict {
        /// Revision requested by the client.
        expected: ScientificRevision,
        /// Current host revision.
        actual: ScientificRevision,
    },
    /// The optimistic configuration-revision guard did not match.
    ConfigRevisionConflict {
        /// Revision requested by the client.
        expected: ConfigRevision,
        /// Current host revision.
        actual: ConfigRevision,
    },
    /// The bounded host admission queue had no capacity for this command.
    Overloaded {
        /// Configured admission capacity at the rejected boundary.
        capacity: usize,
    },
    /// The host lifecycle no longer admits new work.
    HostStopping,
}

/// Failure encountered after a command had been admitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplicationFailure {
    /// Stable machine-readable failure category.
    pub code: String,
    /// Human-readable diagnostic detail.
    pub message: String,
}

/// Actual boundary at which an admitted command finished applying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedCommand {
    /// Scientific tick visible after application.
    pub tick: Tick,
    /// Typed revisions visible after application.
    pub revisions: HostRevisions,
}

/// Application axis of a command's status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum ApplicationState {
    /// The command has an admission order but has not finished applying.
    Admitted,
    /// The command applied exactly once.
    Applied(AppliedCommand),
    /// The command was rejected before admission or at its ordered application boundary.
    Rejected(RejectionReason),
    /// The command was admitted but application failed.
    Failed(ApplicationFailure),
}

/// Failure of the independent command-journal axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalFailure {
    /// Stable machine-readable failure category.
    pub code: String,
    /// Human-readable diagnostic detail.
    pub message: String,
}

/// Journal axis of a command's status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum JournalState {
    /// No journal record exists for this status.
    ///
    /// Runtime-emitted command statuses never use this state: every terminal
    /// command lifecycle, including control success and rejection, is audited.
    /// It remains in the wire enum for non-runtime and historical producers.
    NotRequired,
    /// The command's lifecycle record has not committed yet.
    Pending,
    /// The record committed to volatile storage.
    CommittedVolatile,
    /// The record is durable according to the configured storage contract.
    Durable,
    /// Journal persistence failed independently of application.
    Failed(JournalFailure),
}

/// Stable schema version of [`CommandLifecycleEvidence`].
pub const COMMAND_LIFECYCLE_SCHEMA_VERSION: u16 = 1;

/// One ordered application-axis transition observed at an exact host boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandLifecycleTransition {
    ordinal: u32,
    boundary: AppliedCommand,
    application: ApplicationState,
}

impl CommandLifecycleTransition {
    /// Construct one transition for later validation by
    /// [`CommandLifecycleEvidence::try_new`].
    #[must_use]
    pub const fn new(
        ordinal: u32,
        boundary: AppliedCommand,
        application: ApplicationState,
    ) -> Self {
        Self {
            ordinal,
            boundary,
            application,
        }
    }

    /// Zero-based position in this command's application lifecycle.
    #[must_use]
    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Tick and revisions visible when this transition was observed.
    #[must_use]
    pub const fn boundary(&self) -> AppliedCommand {
        self.boundary
    }

    /// Application state established by this transition.
    #[must_use]
    pub const fn application(&self) -> &ApplicationState {
        &self.application
    }
}

/// Immutable, serde-stable evidence for one command's application lifecycle.
///
/// This record deliberately excludes `CommittedVolatile` and `Durable`: those
/// are storage-ledger states, not application transitions. The source identity
/// is the stable high 64-bit client namespace already carried by [`CommandId`].
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommandLifecycleEvidence {
    schema_version: u16,
    source_client_namespace: u64,
    envelope: CommandEnvelope,
    admission_sequence: Option<AdmissionSequence>,
    transitions: Vec<CommandLifecycleTransition>,
}

impl CommandLifecycleEvidence {
    /// Construct and validate one complete terminal application lifecycle.
    pub fn try_new(
        envelope: CommandEnvelope,
        admission_sequence: Option<AdmissionSequence>,
        transitions: Vec<CommandLifecycleTransition>,
    ) -> Result<Self, CommandLifecycleEvidenceError> {
        let evidence = Self {
            schema_version: COMMAND_LIFECYCLE_SCHEMA_VERSION,
            source_client_namespace: envelope.command_id.client_namespace(),
            envelope,
            admission_sequence,
            transitions,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub(crate) fn from_terminal(
        envelope: CommandEnvelope,
        admission_sequence: Option<AdmissionSequence>,
        initial_boundary: AppliedCommand,
        terminal_boundary: AppliedCommand,
        terminal_application: ApplicationState,
    ) -> Result<Self, CommandLifecycleEvidenceError> {
        let transitions = if admission_sequence.is_some() {
            vec![
                CommandLifecycleTransition {
                    ordinal: 0,
                    boundary: initial_boundary,
                    application: ApplicationState::Admitted,
                },
                CommandLifecycleTransition {
                    ordinal: 1,
                    boundary: terminal_boundary,
                    application: terminal_application,
                },
            ]
        } else {
            vec![CommandLifecycleTransition {
                ordinal: 0,
                boundary: terminal_boundary,
                application: terminal_application,
            }]
        };
        Self::try_new(envelope, admission_sequence, transitions)
    }

    /// Validate schema, source, ordinal, admission, and terminal-state invariants.
    pub fn validate(&self) -> Result<(), CommandLifecycleEvidenceError> {
        if self.schema_version != COMMAND_LIFECYCLE_SCHEMA_VERSION {
            return Err(CommandLifecycleEvidenceError::UnsupportedSchema {
                found: self.schema_version,
            });
        }
        let expected_source = self.envelope.command_id.client_namespace();
        if self.source_client_namespace != expected_source {
            return Err(CommandLifecycleEvidenceError::SourceMismatch {
                expected: expected_source,
                actual: self.source_client_namespace,
            });
        }
        if self.transitions.is_empty() {
            return Err(CommandLifecycleEvidenceError::EmptyTransitions);
        }
        for (index, transition) in self.transitions.iter().enumerate() {
            let expected = u32::try_from(index)
                .map_err(|_| CommandLifecycleEvidenceError::TransitionCountOverflow)?;
            if transition.ordinal != expected {
                return Err(CommandLifecycleEvidenceError::NoncontiguousOrdinal {
                    expected,
                    actual: transition.ordinal,
                });
            }
            if let ApplicationState::Applied(applied) = &transition.application
                && *applied != transition.boundary
            {
                return Err(CommandLifecycleEvidenceError::AppliedBoundaryMismatch);
            }
        }

        if self.admission_sequence.is_none() {
            if self.transitions.len() != 1
                || !matches!(
                    &self.transitions[0].application,
                    ApplicationState::Rejected(
                        RejectionReason::Validation { .. }
                            | RejectionReason::Overloaded { .. }
                            | RejectionReason::HostStopping
                    )
                )
            {
                return Err(CommandLifecycleEvidenceError::InvalidPreAdmissionLifecycle);
            }
        } else {
            if self.transitions.len() != 2
                || self.transitions[0].application != ApplicationState::Admitted
                || !matches!(
                    &self.transitions[1].application,
                    ApplicationState::Applied(_)
                        | ApplicationState::Failed(_)
                        | ApplicationState::Rejected(
                            RejectionReason::ControlRevisionConflict { .. }
                                | RejectionReason::ScientificRevisionConflict { .. }
                                | RejectionReason::ConfigRevisionConflict { .. }
                        )
                )
            {
                return Err(CommandLifecycleEvidenceError::InvalidAdmittedLifecycle);
            }
            let admitted = self.transitions[0].boundary;
            let terminal = self.transitions[1].boundary;
            if terminal.tick < admitted.tick {
                return Err(CommandLifecycleEvidenceError::TerminalTickRegressed);
            }
            if terminal.revisions.control < admitted.revisions.control {
                return Err(CommandLifecycleEvidenceError::TerminalControlRevisionRegressed);
            }
            if terminal.revisions.scientific < admitted.revisions.scientific {
                return Err(CommandLifecycleEvidenceError::TerminalScientificRevisionRegressed);
            }
            if terminal.revisions.config < admitted.revisions.config {
                return Err(CommandLifecycleEvidenceError::TerminalConfigRevisionRegressed);
            }
        }
        Ok(())
    }

    /// Stable command-lifecycle schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Stable source identity derived from the command id's client namespace.
    #[must_use]
    pub const fn source_client_namespace(&self) -> u64 {
        self.source_client_namespace
    }

    /// Exact command envelope, including all expected-revision guards.
    #[must_use]
    pub const fn envelope(&self) -> &CommandEnvelope {
        &self.envelope
    }

    /// Host admission order, absent exactly for pre-admission rejection.
    #[must_use]
    pub const fn admission_sequence(&self) -> Option<AdmissionSequence> {
        self.admission_sequence
    }

    /// Complete ordered application transition sequence.
    #[must_use]
    pub fn transitions(&self) -> &[CommandLifecycleTransition] {
        &self.transitions
    }

    /// Terminal application transition.
    #[must_use]
    pub fn terminal(&self) -> Option<&CommandLifecycleTransition> {
        self.transitions.last()
    }

    /// Whether this record is the successfully applied ordered shutdown barrier.
    #[must_use]
    pub fn is_applied_shutdown(&self) -> bool {
        matches!(&self.envelope.command, HostCommand::Shutdown)
            && self.terminal().is_some_and(|transition| {
                matches!(&transition.application, ApplicationState::Applied(_))
            })
    }

    /// Whether this terminal lifecycle is tracked by runtime journal receipts.
    ///
    /// Every lifecycle evidence record offered by the runtime is audited,
    /// independently of whether the command also carries scientific persistence.
    #[must_use]
    pub const fn requires_runtime_journal(&self) -> bool {
        self.envelope.command.requires_journal() && !self.transitions.is_empty()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandLifecycleEvidenceWire {
    schema_version: u16,
    source_client_namespace: u64,
    envelope: CommandEnvelope,
    admission_sequence: Option<AdmissionSequence>,
    transitions: Vec<CommandLifecycleTransition>,
}

impl<'de> Deserialize<'de> for CommandLifecycleEvidence {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = CommandLifecycleEvidenceWire::deserialize(deserializer)?;
        let evidence = Self {
            schema_version: wire.schema_version,
            source_client_namespace: wire.source_client_namespace,
            envelope: wire.envelope,
            admission_sequence: wire.admission_sequence,
            transitions: wire.transitions,
        };
        evidence.validate().map_err(serde::de::Error::custom)?;
        Ok(evidence)
    }
}

/// Invalid immutable command-lifecycle evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CommandLifecycleEvidenceError {
    /// The payload declared an unsupported schema revision.
    #[error("unsupported command lifecycle schema version {found}")]
    UnsupportedSchema {
        /// The schema revision the payload declared.
        found: u16,
    },
    /// The persisted source did not match the command id namespace.
    #[error("command source namespace {actual} does not match command id namespace {expected}")]
    SourceMismatch {
        /// The namespace the command id claims.
        expected: u64,
        /// The namespace the persisted evidence actually carried.
        actual: u64,
    },
    /// A terminal lifecycle must contain evidence.
    #[error("command lifecycle has no transitions")]
    EmptyTransitions,
    /// The transition vector cannot be represented by its wire ordinal.
    #[error("command lifecycle transition count exceeds u32")]
    TransitionCountOverflow,
    /// Transition ordinals must be the exact zero-based vector order.
    #[error("command lifecycle transition ordinal {actual} does not follow {expected}")]
    NoncontiguousOrdinal {
        /// The ordinal the evidence was required to carry.
        expected: u32,
        /// The ordinal the evidence actually carried.
        actual: u32,
    },
    /// An applied state's embedded boundary disagreed with its observed boundary.
    #[error("applied command transition does not match its observed host boundary")]
    AppliedBoundaryMismatch,
    /// A non-admitted command must be one exact pre-admission rejection.
    #[error("command lifecycle without admission is not one pre-admission rejection")]
    InvalidPreAdmissionLifecycle,
    /// An admitted command must contain admitted then one truthful terminal transition.
    #[error("admitted command lifecycle is not admitted followed by applied, rejected, or failed")]
    InvalidAdmittedLifecycle,
    /// A terminal transition cannot report an earlier scientific tick than admission.
    #[error("command lifecycle terminal tick precedes its admission tick")]
    TerminalTickRegressed,
    /// A terminal transition cannot report an earlier control revision than admission.
    #[error("command lifecycle terminal control revision precedes its admission revision")]
    TerminalControlRevisionRegressed,
    /// A terminal transition cannot report an earlier scientific revision than admission.
    #[error("command lifecycle terminal scientific revision precedes its admission revision")]
    TerminalScientificRevisionRegressed,
    /// A terminal transition cannot report an earlier configuration revision than admission.
    #[error("command lifecycle terminal configuration revision precedes its admission revision")]
    TerminalConfigRevisionRegressed,
}

/// Exact immutable work offered to a nonblocking host-journal adapter.
///
/// A host constructs this value from the completed transition and command
/// boundary, wraps it in an [`Arc`], and retains that same allocation until
/// [`JournalPort::try_admit`] accepts it. In particular, retry code must never
/// reconstruct `persistence` by rereading mutable world state.
#[derive(Debug, Clone)]
pub struct JournalBatch {
    id: JournalBatchId,
    scientific_event_sequence: Option<EventSequence>,
    command_lifecycle: Option<CommandLifecycleEvidence>,
    applied: AppliedCommand,
    scientific: Option<Arc<ScientificBoundary>>,
    persistence: Option<Arc<PersistenceBatch>>,
    retained_bytes: usize,
}

/// Complete engine-neutral payload produced by one scientific transition.
///
/// This mirrors every non-persistence field of `StepOutcome`. Keeping it in the
/// runtime journal prevents disabled or deferred persistence cadence from
/// silently erasing births, deaths, combat, resource, summary, or tick-event
/// evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificBoundary {
    events: TickEvents,
    summary: TickSummary,
    births: Vec<BirthRecord>,
    deaths: Vec<DeathRecord>,
    combat: TickCombatSummary,
    config_revision: u64,
    resource_tick: Option<ResourceLedgerTick>,
    fault: Option<ScientificBoundaryFault>,
}

/// Durable runtime-neutral record of a fault discovered after science completed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificBoundaryFault {
    code: String,
    message: String,
}

impl ScientificBoundaryFault {
    /// Construct a stable fault record from a core-specific completed fault.
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    /// Stable machine-readable category.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Human-readable diagnostic detail.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl ScientificBoundary {
    /// Capture one completed scientific boundary without downstream I/O.
    #[must_use]
    pub const fn new(
        events: TickEvents,
        summary: TickSummary,
        births: Vec<BirthRecord>,
        deaths: Vec<DeathRecord>,
        combat: TickCombatSummary,
        config_revision: u64,
        resource_tick: Option<ResourceLedgerTick>,
    ) -> Self {
        Self {
            events,
            summary,
            births,
            deaths,
            combat,
            config_revision,
            resource_tick,
            fault: None,
        }
    }

    /// Attach a fault discovered after this scientific boundary completed.
    #[must_use]
    pub fn with_fault(mut self, fault: ScientificBoundaryFault) -> Self {
        self.fault = Some(fault);
        self
    }

    /// User-facing events from the completed tick.
    #[must_use]
    pub const fn events(&self) -> &TickEvents {
        &self.events
    }

    /// Exact current-tick summary.
    #[must_use]
    pub const fn summary(&self) -> &TickSummary {
        &self.summary
    }

    /// Complete birth stream independent of persistence cadence.
    #[must_use]
    pub fn births(&self) -> &[BirthRecord] {
        &self.births
    }

    /// Complete death stream independent of persistence cadence.
    #[must_use]
    pub fn deaths(&self) -> &[DeathRecord] {
        &self.deaths
    }

    /// Combat counters accumulated during this tick.
    #[must_use]
    pub const fn combat(&self) -> TickCombatSummary {
        self.combat
    }

    /// Configuration revision active at this boundary.
    #[must_use]
    pub const fn config_revision(&self) -> u64 {
        self.config_revision
    }

    /// Optional resource-conservation evidence for this tick.
    #[must_use]
    pub const fn resource_tick(&self) -> Option<&ResourceLedgerTick> {
        self.resource_tick.as_ref()
    }

    /// Fault discovered after completion, when the boundary advanced but future science stopped.
    #[must_use]
    pub const fn fault(&self) -> Option<&ScientificBoundaryFault> {
        self.fault.as_ref()
    }
}

const ARC_ALLOCATION_OVERHEAD_BYTES: usize = size_of::<usize>() * 2;

const fn retained_vec_bytes<T>(capacity: usize) -> usize {
    capacity.saturating_mul(size_of::<T>())
}

// The `Cow` variant and owned `String` capacity are the evidence being measured; `&str` would
// erase both, so this deliberately differs from an ordinary read-only string parameter.
#[allow(clippy::ptr_arg)]
const fn retained_cow_bytes(value: &std::borrow::Cow<'static, str>) -> usize {
    match value {
        std::borrow::Cow::Borrowed(_) => 0,
        std::borrow::Cow::Owned(value) => value.capacity(),
    }
}

fn retained_birth_bytes(record: &BirthRecord) -> usize {
    record
        .brain_kind
        .as_ref()
        .map_or(0, std::string::String::capacity)
}

fn retained_death_bytes(record: &DeathRecord) -> usize {
    record
        .brain_kind
        .as_ref()
        .map_or(0, std::string::String::capacity)
}

fn retained_command_bytes(command: Option<&CommandEnvelope>) -> usize {
    let Some(CommandEnvelope {
        command: HostCommand::UpdateConfig(config),
        ..
    }) = command
    else {
        return 0;
    };
    size_of::<ScriptBotsConfig>().saturating_add(retained_vec_bytes::<usize>(
        config.neuroflow.hidden_layers.capacity(),
    ))
}

const fn retained_application_state_bytes(application: &ApplicationState) -> usize {
    match application {
        ApplicationState::Rejected(RejectionReason::Validation { message }) => message.capacity(),
        ApplicationState::Failed(failure) => failure
            .code
            .capacity()
            .saturating_add(failure.message.capacity()),
        ApplicationState::Admitted
        | ApplicationState::Applied(_)
        | ApplicationState::Rejected(
            RejectionReason::ControlRevisionConflict { .. }
            | RejectionReason::ScientificRevisionConflict { .. }
            | RejectionReason::ConfigRevisionConflict { .. }
            | RejectionReason::Overloaded { .. }
            | RejectionReason::HostStopping,
        ) => 0,
    }
}

fn retained_command_lifecycle_bytes(lifecycle: Option<&CommandLifecycleEvidence>) -> usize {
    let Some(lifecycle) = lifecycle else {
        return 0;
    };
    let mut retained =
        retained_command_bytes(Some(&lifecycle.envelope)).saturating_add(retained_vec_bytes::<
            CommandLifecycleTransition,
        >(
            lifecycle.transitions.capacity(),
        ));
    for transition in &lifecycle.transitions {
        retained =
            retained.saturating_add(retained_application_state_bytes(&transition.application));
    }
    retained
}

fn retained_scientific_bytes(scientific: Option<&ScientificBoundary>) -> usize {
    let Some(scientific) = scientific else {
        return 0;
    };
    let mut retained = ARC_ALLOCATION_OVERHEAD_BYTES
        .saturating_add(size_of::<ScientificBoundary>())
        .saturating_add(retained_vec_bytes::<BirthRecord>(
            scientific.births.capacity(),
        ))
        .saturating_add(retained_vec_bytes::<DeathRecord>(
            scientific.deaths.capacity(),
        ));
    for birth in &scientific.births {
        retained = retained.saturating_add(retained_birth_bytes(birth));
    }
    for death in &scientific.deaths {
        retained = retained.saturating_add(retained_death_bytes(death));
    }
    if let Some(resource_tick) = &scientific.resource_tick {
        retained = retained.saturating_add(retained_vec_bytes::<scriptbots_core::ResourceFlow>(
            resource_tick.flows.capacity(),
        ));
    }
    if let Some(fault) = &scientific.fault {
        retained = retained
            .saturating_add(fault.code.capacity())
            .saturating_add(fault.message.capacity());
    }
    retained
}

fn retained_agent_bytes(agent: &scriptbots_core::AgentState) -> usize {
    let mut retained = retained_vec_bytes::<String>(agent.runtime.mutation_log.capacity());
    for entry in &agent.runtime.mutation_log {
        retained = retained.saturating_add(entry.capacity());
    }
    match &agent.runtime.brain {
        scriptbots_core::BrainBinding::Unbound => {}
        scriptbots_core::BrainBinding::Legacy { kind, .. } => {
            retained = retained.saturating_add(kind.capacity());
        }
        scriptbots_core::BrainBinding::Protocol { kind, genome, .. } => {
            retained = retained
                .saturating_add(kind.capacity())
                .saturating_add(genome.family_id().as_str().len())
                .saturating_add(genome.payload().len());
        }
    }
    retained
}

fn retained_persistence_bytes(persistence: Option<&PersistenceBatch>) -> usize {
    let Some(persistence) = persistence else {
        return 0;
    };
    let mut retained = ARC_ALLOCATION_OVERHEAD_BYTES
        .saturating_add(size_of::<PersistenceBatch>())
        .saturating_add(retained_vec_bytes::<scriptbots_core::MetricSample>(
            persistence.metrics.capacity(),
        ))
        .saturating_add(retained_vec_bytes::<scriptbots_core::PersistenceEvent>(
            persistence.events.capacity(),
        ))
        .saturating_add(retained_vec_bytes::<scriptbots_core::AgentState>(
            persistence.agents.capacity(),
        ))
        .saturating_add(retained_vec_bytes::<BirthRecord>(
            persistence.births.capacity(),
        ))
        .saturating_add(retained_vec_bytes::<DeathRecord>(
            persistence.deaths.capacity(),
        ))
        .saturating_add(retained_vec_bytes::<scriptbots_core::ReplayEvent>(
            persistence.replay_events.capacity(),
        ));
    for metric in &persistence.metrics {
        retained = retained.saturating_add(retained_cow_bytes(&metric.name));
    }
    for event in &persistence.events {
        if let scriptbots_core::PersistenceEventKind::Custom(kind) = &event.kind {
            retained = retained.saturating_add(retained_cow_bytes(kind));
        }
    }
    for agent in &persistence.agents {
        retained = retained.saturating_add(retained_agent_bytes(agent));
    }
    for birth in &persistence.births {
        retained = retained.saturating_add(retained_birth_bytes(birth));
    }
    for death in &persistence.deaths {
        retained = retained.saturating_add(retained_death_bytes(death));
    }
    for event in &persistence.replay_events {
        if let scriptbots_core::ReplayEventKind::BrainOutputs { outputs } = &event.kind {
            retained = retained.saturating_add(retained_vec_bytes::<f32>(outputs.capacity()));
        }
    }
    retained
}

fn journal_batch_retained_bytes(
    command_lifecycle: Option<&CommandLifecycleEvidence>,
    scientific: Option<&ScientificBoundary>,
    persistence: Option<&PersistenceBatch>,
) -> usize {
    size_of::<JournalBatch>()
        .saturating_add(retained_command_lifecycle_bytes(command_lifecycle))
        .saturating_add(retained_scientific_bytes(scientific))
        .saturating_add(retained_persistence_bytes(persistence))
}

impl JournalBatch {
    /// Construct one exact journal batch at an already-completed boundary.
    #[must_use]
    pub(crate) fn new(
        id: JournalBatchId,
        scientific_event_sequence: Option<EventSequence>,
        command_lifecycle: Option<CommandLifecycleEvidence>,
        applied: AppliedCommand,
        scientific: Option<Arc<ScientificBoundary>>,
        persistence: Option<Arc<PersistenceBatch>>,
    ) -> Self {
        let retained_bytes = journal_batch_retained_bytes(
            command_lifecycle.as_ref(),
            scientific.as_deref(),
            persistence.as_deref(),
        );
        Self {
            id,
            scientific_event_sequence,
            command_lifecycle,
            applied,
            scientific,
            persistence,
            retained_bytes,
        }
    }

    /// Stable identity reused for every admission retry and later receipt.
    #[must_use]
    pub const fn id(&self) -> JournalBatchId {
        self.id
    }

    /// Canonical scientific-event sequence, present exactly for scientific boundaries.
    #[must_use]
    pub const fn scientific_event_sequence(&self) -> Option<EventSequence> {
        self.scientific_event_sequence
    }

    /// Command id associated with this batch, or `None` for automatic science.
    #[must_use]
    pub fn command_id(&self) -> Option<CommandId> {
        self.command().map(|command| command.command_id)
    }

    /// Exact command envelope captured in this lifecycle record.
    #[must_use]
    pub const fn command(&self) -> Option<&CommandEnvelope> {
        match &self.command_lifecycle {
            Some(lifecycle) => Some(&lifecycle.envelope),
            None => None,
        }
    }

    /// Immutable application-lifecycle evidence carried by command-driven work.
    #[must_use]
    pub const fn command_lifecycle(&self) -> Option<&CommandLifecycleEvidence> {
        self.command_lifecycle.as_ref()
    }

    /// Whether accepting this batch releases one bounded pre-admission audit slot.
    #[must_use]
    pub fn uses_ingress_audit_slot(&self) -> bool {
        self.command_lifecycle
            .as_ref()
            .is_some_and(|lifecycle| lifecycle.admission_sequence.is_none())
    }

    /// Whether receipts for this batch advance the command's runtime journal axis.
    #[must_use]
    pub fn requires_runtime_journal(&self) -> bool {
        self.command_lifecycle
            .as_ref()
            .is_some_and(CommandLifecycleEvidence::requires_runtime_journal)
    }

    /// Whether this batch is the successfully applied ordered shutdown barrier.
    #[must_use]
    pub fn is_applied_shutdown(&self) -> bool {
        self.command_lifecycle
            .as_ref()
            .is_some_and(CommandLifecycleEvidence::is_applied_shutdown)
    }

    /// Tick and typed revisions at the record's terminal host boundary.
    ///
    /// For legacy consumers this remains named `applied`; command consumers must
    /// inspect [`Self::command_lifecycle`] to distinguish application, rejection,
    /// and failure truthfully.
    #[must_use]
    pub const fn applied(&self) -> AppliedCommand {
        self.applied
    }

    /// Exact scientific boundary, or `None` for command-only work.
    #[must_use]
    pub const fn scientific(&self) -> Option<&Arc<ScientificBoundary>> {
        self.scientific.as_ref()
    }

    /// Exact immutable scientific persistence payload, when this boundary produced one.
    #[must_use]
    pub const fn persistence(&self) -> Option<&Arc<PersistenceBatch>> {
        self.persistence.as_ref()
    }

    /// Conservative bytes retained by this exact batch for bounded admission.
    ///
    /// The charge is computed once before the first admission attempt, uses saturating arithmetic,
    /// and includes owned vector capacity and nested dynamic payloads without serializing or
    /// allocating. Exact retries therefore reuse both this charge and the original batch allocation.
    #[must_use]
    pub const fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
}

/// Immediate result of one nonblocking journal admission attempt.
///
/// `Accepted` means only that the adapter took responsibility for the exact
/// batch. Commit and durability advance exclusively through [`JournalReceipt`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JournalAdmission {
    /// The adapter accepted the batch and owes a later receipt.
    Accepted {
        /// Identity accepted by the adapter.
        batch_id: JournalBatchId,
    },
    /// The bounded adapter had no admission capacity.
    Full {
        /// Identity that was not accepted.
        batch_id: JournalBatchId,
        /// Configured queue capacity at this boundary.
        capacity: usize,
    },
    /// The adapter permanently closed its admission gate.
    Closed {
        /// Identity that was not accepted.
        batch_id: JournalBatchId,
    },
}

impl JournalAdmission {
    /// Batch identity echoed by this admission result.
    #[must_use]
    pub const fn batch_id(self) -> JournalBatchId {
        match self {
            Self::Accepted { batch_id }
            | Self::Full { batch_id, .. }
            | Self::Closed { batch_id } => batch_id,
        }
    }

    /// Whether responsibility for the exact batch transferred to the adapter.
    #[must_use]
    pub const fn is_accepted(self) -> bool {
        matches!(self, Self::Accepted { .. })
    }
}

/// Terminal or progressive journal knowledge returned after admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum JournalReceiptState {
    /// The batch committed to volatile storage but is not crash durable.
    CommittedVolatile,
    /// The batch is durable according to the adapter's configured contract.
    Durable,
    /// The adapter can no longer complete this batch.
    Failed(JournalFailure),
}

/// Minimum journal commitment required before an ordered shutdown may finish.
///
/// The requirement applies to the shutdown batch and every earlier accepted
/// batch in the same host session. Durable is the safe default for adapters
/// backed by files or remote storage. Purely volatile adapters must opt in to
/// volatile shutdown explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShutdownCommitRequirement {
    /// In-memory commitment is sufficient for this adapter's contract.
    CommittedVolatile,
    /// Every ordered batch must reach crash-durable storage.
    Durable,
}

/// Typed acknowledgement for one previously accepted journal batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalReceipt {
    batch_id: JournalBatchId,
    state: JournalReceiptState,
}

impl JournalReceipt {
    /// Construct an acknowledgement for a stable batch identity.
    #[must_use]
    pub const fn new(batch_id: JournalBatchId, state: JournalReceiptState) -> Self {
        Self { batch_id, state }
    }

    /// Stable batch identity acknowledged by this receipt.
    #[must_use]
    pub const fn batch_id(&self) -> JournalBatchId {
        self.batch_id
    }

    /// Commit, durability, or terminal-failure knowledge carried by this receipt.
    #[must_use]
    pub const fn state(&self) -> &JournalReceiptState {
        &self.state
    }
}

/// Runtime-neutral, nonblocking adapter boundary for host journal work.
///
/// Implementations may enqueue work for another owner, but these methods must
/// not wait for database I/O, worker progress, or durability. A rejected batch
/// remains owned by the caller through the original [`Arc<JournalBatch>`].
pub trait JournalPort {
    /// Try to transfer responsibility for one exact immutable batch.
    fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission;

    /// Poll at most `limit` acknowledgements without blocking.
    fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt>;

    /// Optional detached capability for reconstructing evicted scientific-event records.
    ///
    /// A live-memory reader must never advertise crash durability. File-backed adapters added by
    /// the storage integration return a crash-durable reader only after their durable watermark
    /// covers the requested records.
    fn event_reader(&self, _session_id: HostSessionId) -> Option<Arc<dyn EventJournalReader>> {
        None
    }

    /// Commitment threshold that gates ordered host shutdown.
    ///
    /// This value must remain stable for the lifetime of a host. The durable
    /// default prevents a file-backed adapter from accidentally treating an
    /// intermediate volatile receipt as shutdown completion. Admission of the
    /// ordered shutdown [`JournalBatch`] is also the adapter-neutral flush
    /// barrier request: its qualifying receipt may be emitted only after that
    /// batch and every earlier accepted batch in the host session meet this
    /// threshold. A native lifecycle may time out while waiting, but the host
    /// retains the exact pending or inflight work so a later drive can resume
    /// the same barrier rather than inventing a second one.
    fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
        ShutdownCommitRequirement::Durable
    }
}

/// Typed reason scientific progress stopped at a manual-drive boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HostBlocker {
    /// Playback is intentionally paused and no explicit step was applied.
    PlaybackPaused,
    /// A retained journal batch could not enter the bounded adapter.
    JournalFull {
        /// Exact batch retained for retry.
        batch_id: JournalBatchId,
        /// Configured adapter capacity at the failed boundary.
        capacity: usize,
    },
    /// A retained journal batch reached a closed adapter.
    JournalClosed {
        /// Exact batch retained for retry or orderly failure handling.
        batch_id: JournalBatchId,
    },
    /// Canonical scientific-event hot ring is pinned before a lossless eviction.
    EventJournalHighWater {
        /// Configured hot-ring capacity.
        capacity: usize,
        /// Currently pinned pending records.
        pending: usize,
        /// Oldest pending batch, when any pending record remains in the ring.
        oldest_pending: Option<JournalBatchId>,
        /// Exact batch at the pinned front, even when it is no longer pending.
        pinned_batch: JournalBatchId,
        /// Canonical sequence at the pinned front.
        pinned_sequence: EventSequence,
        /// Why the front cannot currently leave the hot ring.
        reason: EventHighWaterReason,
    },
    /// The host is draining an ordered shutdown boundary.
    LifecycleStopping,
    /// The host has completed shutdown and cannot advance science.
    LifecycleStopped,
    /// A latched scientific fault prevents a later transition.
    ScientificFault,
}

/// Exact reason a full scientific-event hot ring cannot evict its front record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventHighWaterReason {
    /// The exact journal batch has not committed yet.
    Pending,
    /// The journal permanently failed the exact batch.
    Failed,
    /// No reader can reconstruct this non-durable record after eviction.
    NoReader,
    /// The observed commitment cannot satisfy the reader's advertised guarantee.
    GuaranteeMismatch,
    /// The reader's atomic retention snapshot does not contain the exact front sequence.
    RangeUnavailable,
}

/// Scheduling interest derived from the sole-owner host state.
///
/// Native and browser adapters use this value only to decide when to call
/// [`ManualHostDriver::drive`]. It never grants permission to mutate the world
/// or to infer scientific time from repaint activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostDriveInterest {
    /// Already-admitted command work can be processed at the current instant.
    ReadyNow,
    /// Healthy automatic playback needs the next fixed cadence deadline.
    Deadline,
    /// No periodic science deadline is useful; wait for an explicit wake.
    WakeOnly,
    /// Journal receipts or ordered shutdown finalization still need polling.
    Draining,
    /// The host reached its clean terminal lifecycle.
    Terminated,
    /// A queryable host fault prevents normal progress.
    Faulted,
}

/// Queryable host fault that is independent of frontend or transport state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HostFault {
    /// Core rejected or faulted a scientific transition.
    Scientific {
        /// Tick visible when the fault was observed.
        tick: Tick,
        /// Stable machine-readable category.
        code: String,
        /// Human-readable diagnostic detail.
        message: String,
    },
    /// An accepted journal batch later failed.
    Journal {
        /// Stable failed batch identity.
        batch_id: JournalBatchId,
        /// Typed journal failure detail.
        failure: JournalFailure,
    },
    /// The host detected an internal protocol invariant violation.
    Protocol {
        /// Stable machine-readable category.
        code: String,
        /// Human-readable diagnostic detail.
        message: String,
    },
}

/// Queryable health of the sole-owner host state machine.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum HostHealth {
    /// The host has no latched blocker or fault.
    #[default]
    Healthy,
    /// Progress is stopped by a typed, potentially recoverable condition.
    Blocked(HostBlocker),
    /// Progress is stopped by a queryable fault.
    Faulted(HostFault),
}

impl HostHealth {
    /// Recoverable blocker carried by this health value, if any.
    #[must_use]
    pub const fn blocker(&self) -> Option<HostBlocker> {
        match self {
            Self::Blocked(blocker) => Some(*blocker),
            Self::Healthy | Self::Faulted(_) => None,
        }
    }

    /// Fault carried by this health value, if any.
    #[must_use]
    pub const fn fault(&self) -> Option<&HostFault> {
        match self {
            Self::Faulted(fault) => Some(fault),
            Self::Healthy | Self::Blocked(_) => None,
        }
    }
}

/// Two-axis status for one stable command id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CommandStatus {
    command_id: CommandId,
    admission_sequence: Option<AdmissionSequence>,
    application: ApplicationState,
    journal: JournalState,
}

impl CommandStatus {
    /// Construct a status after validating cross-axis invariants.
    pub fn try_new(
        command_id: CommandId,
        admission_sequence: Option<AdmissionSequence>,
        application: ApplicationState,
        journal: JournalState,
    ) -> Result<Self, StatusCombinationError> {
        validate_status_combination(admission_sequence, &application, &journal)?;
        Ok(Self {
            command_id,
            admission_sequence,
            application,
            journal,
        })
    }

    /// Construct a pre-admission rejection.
    pub fn rejected(
        command_id: CommandId,
        reason: RejectionReason,
    ) -> Result<Self, StatusCombinationError> {
        Self::try_new(
            command_id,
            None,
            ApplicationState::Rejected(reason),
            JournalState::NotRequired,
        )
    }

    /// Stable command id represented by this status.
    #[must_use]
    pub const fn command_id(&self) -> CommandId {
        self.command_id
    }

    /// Total admission order, absent only for pre-admission rejection.
    #[must_use]
    pub const fn admission_sequence(&self) -> Option<AdmissionSequence> {
        self.admission_sequence
    }

    /// Current application-axis state.
    #[must_use]
    pub const fn application(&self) -> &ApplicationState {
        &self.application
    }

    /// Current journal-axis state.
    #[must_use]
    pub const fn journal(&self) -> &JournalState {
        &self.journal
    }

    /// Revalidate cross-axis invariants after transport deserialization.
    pub const fn validate(&self) -> Result<(), StatusCombinationError> {
        validate_status_combination(self.admission_sequence, &self.application, &self.journal)
    }
}

const fn validate_status_combination(
    admission_sequence: Option<AdmissionSequence>,
    application: &ApplicationState,
    journal: &JournalState,
) -> Result<(), StatusCombinationError> {
    match application {
        ApplicationState::Admitted => {
            if admission_sequence.is_none() {
                return Err(StatusCombinationError::MissingAdmissionSequence);
            }
            if !matches!(journal, JournalState::NotRequired | JournalState::Pending) {
                return Err(StatusCombinationError::AdmittedJournalAdvanced);
            }
        }
        ApplicationState::Applied(_) | ApplicationState::Failed(_) => {
            if admission_sequence.is_none() {
                return Err(StatusCombinationError::MissingAdmissionSequence);
            }
        }
        ApplicationState::Rejected(reason) => match reason {
            RejectionReason::ControlRevisionConflict { .. }
            | RejectionReason::ScientificRevisionConflict { .. }
            | RejectionReason::ConfigRevisionConflict { .. }
                if admission_sequence.is_none() =>
            {
                return Err(StatusCombinationError::ConflictMissingAdmission);
            }
            RejectionReason::Validation { .. }
            | RejectionReason::Overloaded { .. }
            | RejectionReason::HostStopping
                if admission_sequence.is_some() =>
            {
                return Err(StatusCombinationError::PreAdmissionRejectionWasAdmitted);
            }
            _ => {}
        },
    }
    Ok(())
}

/// Invalid cross-axis command status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum StatusCombinationError {
    /// Every admitted, applied, or failed application state needs an admission sequence.
    #[error("an admitted, applied, or failed command requires an admission sequence")]
    MissingAdmissionSequence,
    /// An application-pending command cannot claim a committed or failed journal outcome.
    #[error("an admitted command may only have journal state not_required or pending")]
    AdmittedJournalAdvanced,
    /// An ordered compare-and-set conflict must retain its admission position.
    #[error("a control revision conflict requires an admission sequence")]
    ConflictMissingAdmission,
    /// Validation, overload, and lifecycle rejection happen before admission.
    #[error("a pre-admission rejection cannot have an admission sequence")]
    PreAdmissionRejectionWasAdmitted,
}

#[derive(Deserialize)]
struct CommandStatusWire {
    command_id: CommandId,
    admission_sequence: Option<AdmissionSequence>,
    application: ApplicationState,
    journal: JournalState,
}

impl TryFrom<CommandStatusWire> for CommandStatus {
    type Error = StatusCombinationError;

    fn try_from(wire: CommandStatusWire) -> Result<Self, Self::Error> {
        Self::try_new(
            wire.command_id,
            wire.admission_sequence,
            wire.application,
            wire.journal,
        )
    }
}

impl<'de> Deserialize<'de> for CommandStatus {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = CommandStatusWire::deserialize(deserializer)?;
        Self::try_from(wire).map_err(serde::de::Error::custom)
    }
}

/// Protocol-level command validation failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CommandValidationError {
    /// Playback speed must be finite and non-negative.
    #[error("speed multiplier must be finite and non-negative")]
    InvalidSpeed,
    /// The core configuration contract rejected a replacement.
    #[error("{message}")]
    InvalidConfig {
        /// Core validation diagnostic.
        message: String,
    },
}

/// Ordered ephemeral notification emitted by the host protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostEvent {
    /// Host session that emitted this notification.
    pub session_id: HostSessionId,
    /// Total notification order within one host session.
    pub sequence: ProtocolEventSequence,
    /// Scientific tick visible when the event was emitted.
    pub tick: Tick,
    /// Event payload.
    pub kind: HostEventKind,
}

/// Renderer-neutral event payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum HostEventKind {
    /// One command's application or journal status changed.
    CommandStatusChanged(CommandStatus),
    /// Host lifecycle changed.
    LifecycleChanged(HostLifecycle),
    /// Queryable host health changed.
    HealthChanged(HostHealth),
}

/// Opaque cursor over bounded, reconstructibly lossy host notifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolEventCursor {
    session_id: HostSessionId,
    last_seen: ProtocolEventSequence,
}

impl ProtocolEventCursor {
    /// Start before the first notification in one host session.
    #[must_use]
    pub const fn beginning(session_id: HostSessionId) -> Self {
        Self {
            session_id,
            last_seen: ProtocolEventSequence::new(0),
        }
    }

    /// Resume after an already-observed notification.
    #[must_use]
    pub const fn after(session_id: HostSessionId, sequence: ProtocolEventSequence) -> Self {
        Self {
            session_id,
            last_seen: sequence,
        }
    }

    /// Bound host session.
    #[must_use]
    pub const fn session_id(self) -> HostSessionId {
        self.session_id
    }

    /// Last notification observed through this cursor.
    #[must_use]
    pub const fn last_seen(self) -> ProtocolEventSequence {
        self.last_seen
    }
}

/// Commitment state paired with one immutable scientific-event record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum EventCommitment {
    /// The exact journal batch has not reached a queryable commitment.
    Pending,
    /// The event is queryable only while its live in-memory journal survives.
    CommittedVolatile,
    /// The event is crash durable according to the configured journal contract.
    Durable,
    /// The journal can no longer commit this exact record.
    Failed(JournalFailure),
}

/// Immutable canonical record for one completed scientific boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificEvent {
    /// Host session that allocated this event.
    pub session_id: HostSessionId,
    /// Contiguous order among scientific boundaries only.
    pub sequence: EventSequence,
    /// Stable runtime-journal batch carrying the exact boundary.
    pub batch_id: JournalBatchId,
    /// Completed scientific tick.
    pub tick: Tick,
    /// Revisions captured at the boundary.
    pub revisions: HostRevisions,
    /// Exact complete engine-neutral scientific payload.
    #[serde(with = "serde_arc")]
    pub boundary: Arc<ScientificBoundary>,
}

/// One immutable event paired with current journal commitment knowledge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JournaledScientificEvent {
    /// Exact immutable event allocation.
    #[serde(with = "serde_arc")]
    pub event: Arc<ScientificEvent>,
    /// Current journal commitment observed by this page.
    pub commitment: EventCommitment,
}

/// Inclusive contiguous scientific-event range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSequenceRange {
    /// First available sequence.
    pub first: EventSequence,
    /// Last available sequence.
    pub last: EventSequence,
}

impl EventSequenceRange {
    /// Whether this inclusive range has at least one sequence.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.first.get() <= self.last.get()
    }

    /// Whether this valid range contains one sequence.
    #[must_use]
    pub const fn contains(self, sequence: EventSequence) -> bool {
        self.is_valid() && self.first.get() <= sequence.get() && sequence.get() <= self.last.get()
    }

    /// Whether this valid range completely contains another valid range.
    #[must_use]
    pub const fn contains_range(self, other: Self) -> bool {
        other.is_valid() && self.contains(other.first) && self.contains(other.last)
    }
}

/// Strength of an adapter-neutral scientific-event catch-up source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventCatchUpGuarantee {
    /// Queryability lasts only while the current in-memory journal survives.
    LiveMemory,
    /// Records are queryable after a writer/process restart.
    CrashDurable,
}

/// Atomic journal-retention evidence used while the hot ring evicts one record.
///
/// The opaque retained allocation is deliberately not exposed. Holding this value proves that
/// every sequence in [`Self::range`] remains physically retained for the snapshot lifetime, even
/// if the journal writer publishes or compacts concurrently.
#[derive(Clone)]
pub struct EventRetentionSnapshot {
    session_id: HostSessionId,
    guarantee: EventCatchUpGuarantee,
    range: EventSequenceRange,
    _retained: Arc<dyn Any + Send + Sync>,
}

impl fmt::Debug for EventRetentionSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EventRetentionSnapshot")
            .field("session_id", &self.session_id)
            .field("guarantee", &self.guarantee)
            .field("range", &self.range)
            .finish_non_exhaustive()
    }
}

impl EventRetentionSnapshot {
    /// Construct atomic retention evidence over one immutable reader allocation.
    ///
    /// # Errors
    ///
    /// Returns a protocol error when `range` is inverted.
    pub fn try_new<T>(
        session_id: HostSessionId,
        guarantee: EventCatchUpGuarantee,
        range: EventSequenceRange,
        retained: Arc<T>,
    ) -> Result<Self, HostAccessError>
    where
        T: Any + Send + Sync,
    {
        if !range.is_valid() {
            return Err(protocol_violation(
                "event retention snapshot range must be nonempty and ordered",
            ));
        }
        Ok(Self {
            session_id,
            guarantee,
            range,
            _retained: retained,
        })
    }

    /// Stable host session covered by this snapshot.
    #[must_use]
    pub const fn session_id(&self) -> HostSessionId {
        self.session_id
    }

    /// Queryability guarantee covered by this snapshot.
    #[must_use]
    pub const fn guarantee(&self) -> EventCatchUpGuarantee {
        self.guarantee
    }

    /// Exact inclusive range retained by the opaque allocation.
    #[must_use]
    pub const fn range(&self) -> EventSequenceRange {
        self.range
    }
}

/// Source that produced one validated event page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventPageSource {
    /// Bounded latest hot ring.
    Hot,
    /// Live in-memory journal catch-up.
    LiveMemory,
    /// Crash-durable journal catch-up.
    Durable,
}

/// Opaque session-bound locator for a precise missing scientific-event range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventCatchUpLocator {
    session_id: HostSessionId,
    range: EventSequenceRange,
    guarantee: EventCatchUpGuarantee,
}

impl EventCatchUpLocator {
    /// Host session whose journal may satisfy this locator.
    #[must_use]
    pub const fn session_id(self) -> HostSessionId {
        self.session_id
    }

    /// Exact contiguous range required before hot-ring reads can resume.
    #[must_use]
    pub const fn range(self) -> EventSequenceRange {
        self.range
    }

    /// Queryability guarantee advertised by the source.
    #[must_use]
    pub const fn guarantee(self) -> EventCatchUpGuarantee {
        self.guarantee
    }

    /// Return a locator for the exact unread suffix after one observed sequence.
    ///
    /// `None` means the sequence completed this locator or did not belong to it.
    #[must_use]
    pub fn remaining_after(self, sequence: EventSequence) -> Option<Self> {
        let first = sequence.checked_next()?;
        if !self.range.contains(sequence) || !self.range.contains(first) {
            return None;
        }
        Some(Self {
            session_id: self.session_id,
            range: EventSequenceRange {
                first,
                last: self.range.last,
            },
            guarantee: self.guarantee,
        })
    }
}

/// Why a gap cannot currently be repaired from a journal reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventCatchUpUnavailableReason {
    /// The host port exposes no compatible reader capability.
    NoReader,
    /// The requested prefix has expired from the available reader range.
    RangeExpired,
    /// Only a suffix of the exact missing range is available.
    PartialRange,
    /// The locator belongs to a different host session.
    SessionMismatch,
}

/// Catch-up state carried by an explicit hot-ring gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum EventCatchUpState {
    /// An exact adapter-neutral locator is available.
    Available(EventCatchUpLocator),
    /// The missing prefix cannot currently be reconstructed.
    Unavailable(EventCatchUpUnavailableReason),
}

/// Exact metadata returned when a scientific-event cursor fell behind the hot ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventGap {
    /// Host session whose event stream was read.
    pub session_id: HostSessionId,
    /// Next sequence required by the unchanged cursor.
    pub expected: EventSequence,
    /// Exact missing prefix before the current hot range.
    pub missing: EventSequenceRange,
    /// Current inclusive hot range.
    pub hot_available: EventSequenceRange,
    /// Newest sequence published by the host.
    pub latest: EventSequence,
    /// Catch-up capability for the exact missing range.
    pub catch_up: EventCatchUpState,
}

impl EventGap {
    /// Sequence to install only when a client explicitly accepts display truncation.
    #[must_use]
    pub const fn resume_after(self) -> EventSequence {
        EventSequence::new(self.hot_available.first.get().saturating_sub(1))
    }
}

/// One contiguous, bounded page of scientific events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventPage {
    /// Host session that owns every event in the page.
    pub session_id: HostSessionId,
    /// Read source and its durability meaning.
    pub source: EventPageSource,
    /// Contiguous events strictly after the requested cursor.
    pub events: Vec<JournaledScientificEvent>,
    /// Newest sequence known to the source at read time.
    pub latest: EventSequence,
}

/// Result of polling the bounded scientific-event hot ring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum EventPoll {
    /// The returned page begins exactly at the cursor's next sequence, or is empty at the tip.
    Contiguous(EventPage),
    /// The cursor is behind the retained hot range and remains unchanged.
    Gap(EventGap),
}

/// Result of resolving an explicit catch-up locator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum EventCatchUp {
    /// A contiguous catch-up page beginning at the locator's first sequence.
    Contiguous(EventPage),
    /// The exact locator cannot be satisfied; the client cursor remains unchanged.
    Unavailable {
        /// Exact unavailable range.
        range: EventSequenceRange,
        /// Stable reason the reader cannot serve it.
        reason: EventCatchUpUnavailableReason,
    },
}

/// Cloneable adapter-neutral reader for older canonical scientific events.
pub trait EventJournalReader: Send + Sync {
    /// Host session served by this reader, stable for the reader lifetime.
    fn session_id(&self) -> HostSessionId;

    /// Queryability guarantee of returned records, stable for the reader lifetime.
    fn guarantee(&self) -> EventCatchUpGuarantee;

    /// Current inclusive readable range, or `None` when empty.
    ///
    /// This is a cached nonblocking watermark query. Implementations must not perform file I/O,
    /// wait on a worker, or scan an unbounded journal; slow record reads belong in [`Self::read`].
    fn available_range(&self) -> Option<EventSequenceRange>;

    /// Atomically capture the readable range and an opaque allocation retaining every record in
    /// that range.
    ///
    /// A successful snapshot must close the range/compaction race: while the returned value is
    /// alive, a concurrent writer cannot destroy any covered record. This method is cached,
    /// nonblocking, and performs no file I/O. The hot ring holds the snapshot only across its exact
    /// eviction reservation and publication boundary.
    fn retention_snapshot(&self) -> Option<EventRetentionSnapshot>;

    /// Whether the cached reader index binds one exact sequence to one exact journal batch.
    ///
    /// This is a nonblocking identity query used for a later commitment receipt after the
    /// corresponding record has left the hot ring. Implementations must not perform file I/O.
    fn contains_event_identity(&self, sequence: EventSequence, batch_id: JournalBatchId) -> bool;

    /// Resolve a session-bound exact range without advancing client state.
    ///
    /// `limit` is a hard result and allocation bound. Implementations return at most `limit`
    /// records and must not allocate work proportional to an unbounded journal.
    fn read(
        &self,
        locator: EventCatchUpLocator,
        limit: usize,
    ) -> Result<EventCatchUp, HostAccessError>;
}

#[derive(Debug)]
struct EventHubState {
    capacity: usize,
    next_sequence: EventSequence,
    published_total: u64,
    reserved_front: Option<EventPublishReservation>,
    entries: VecDeque<JournaledScientificEvent>,
}

#[derive(Debug)]
struct EventPublishReservation {
    sequence: EventSequence,
    retention: Option<EventRetentionSnapshot>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EventHighWater {
    pub(crate) batch_id: JournalBatchId,
    pub(crate) sequence: EventSequence,
    pub(crate) reason: EventHighWaterReason,
}

#[derive(Debug, Clone)]
struct EventHotView {
    capacity: usize,
    next_sequence: EventSequence,
    published_total: u64,
    entries: VecDeque<JournaledScientificEvent>,
}

impl From<&EventHubState> for EventHotView {
    fn from(state: &EventHubState) -> Self {
        Self {
            capacity: state.capacity,
            next_sequence: state.next_sequence,
            published_total: state.published_total,
            entries: state.entries.clone(),
        }
    }
}

/// Cloneable detached bounded scientific-event hot ring and catch-up capability.
///
/// The hub owns no command sender, mutable world, database connection, or per-subscriber queue.
#[derive(Clone)]
pub struct EventHub {
    session_id: HostSessionId,
    reader: Option<Arc<dyn EventJournalReader>>,
    state: Arc<Mutex<EventHubState>>,
    hot: Arc<ArcSwap<EventHotView>>,
}

impl fmt::Debug for EventHub {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.hot.load();
        formatter
            .debug_struct("EventHub")
            .field("session_id", &self.session_id)
            .field("capacity", &state.capacity)
            .field("len", &state.entries.len())
            .field("published_total", &state.published_total)
            .finish_non_exhaustive()
    }
}

impl EventHub {
    /// Construct one bounded scientific-event hub.
    ///
    /// # Errors
    ///
    /// Returns a protocol error when capacity is zero or the reader serves another session.
    pub fn new(
        session_id: HostSessionId,
        capacity: usize,
        reader: Option<Arc<dyn EventJournalReader>>,
    ) -> Result<Self, HostAccessError> {
        if capacity == 0 {
            return Err(protocol_violation(
                "scientific event capacity must be nonzero",
            ));
        }
        if reader
            .as_ref()
            .is_some_and(|reader| reader.session_id() != session_id)
        {
            return Err(protocol_violation(
                "scientific event reader belongs to another host session",
            ));
        }
        let state = EventHubState {
            capacity,
            next_sequence: EventSequence::new(1),
            published_total: 0,
            reserved_front: None,
            entries: VecDeque::with_capacity(capacity),
        };
        let hot = Arc::new(ArcSwap::from_pointee(EventHotView::from(&state)));
        Ok(Self {
            session_id,
            reader,
            state: Arc::new(Mutex::new(state)),
            hot,
        })
    }

    /// Stable host session published by this hub.
    #[must_use]
    pub const fn session_id(&self) -> HostSessionId {
        self.session_id
    }

    /// Configured hot-ring capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.hot.load().capacity
    }

    /// Current retained hot-ring records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hot.load().entries.len()
    }

    /// Whether no scientific record is currently retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total canonical records ever published, independent of ring wrap.
    #[must_use]
    pub fn published_total(&self) -> u64 {
        self.hot.load().published_total
    }

    /// Pending records that may not be evicted.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.hot
            .load()
            .entries
            .iter()
            .filter(|entry| entry.commitment == EventCommitment::Pending)
            .count()
    }

    /// Oldest pending batch, when pressure pins the hot prefix.
    #[must_use]
    pub fn oldest_pending_batch(&self) -> Option<JournalBatchId> {
        self.hot
            .load()
            .entries
            .iter()
            .find(|entry| entry.commitment == EventCommitment::Pending)
            .map(|entry| entry.event.batch_id)
    }

    fn eviction_reservation(
        &self,
        entry: &JournaledScientificEvent,
    ) -> Result<EventPublishReservation, EventHighWaterReason> {
        match &entry.commitment {
            EventCommitment::Pending => return Err(EventHighWaterReason::Pending),
            EventCommitment::Failed(_) => return Err(EventHighWaterReason::Failed),
            EventCommitment::CommittedVolatile | EventCommitment::Durable => {}
        }
        let Some(reader) = &self.reader else {
            return if entry.commitment == EventCommitment::Durable {
                Ok(EventPublishReservation {
                    sequence: entry.event.sequence,
                    retention: None,
                })
            } else {
                Err(EventHighWaterReason::NoReader)
            };
        };
        let commitment_safe = match reader.guarantee() {
            EventCatchUpGuarantee::LiveMemory => matches!(
                entry.commitment,
                EventCommitment::CommittedVolatile | EventCommitment::Durable
            ),
            EventCatchUpGuarantee::CrashDurable => entry.commitment == EventCommitment::Durable,
        };
        if !commitment_safe {
            return Err(EventHighWaterReason::GuaranteeMismatch);
        }
        let Some(retention) = reader.retention_snapshot() else {
            return Err(EventHighWaterReason::RangeUnavailable);
        };
        if retention.session_id() != self.session_id || retention.guarantee() != reader.guarantee()
        {
            return Err(EventHighWaterReason::GuaranteeMismatch);
        }
        if !retention.range().contains(entry.event.sequence) {
            return Err(EventHighWaterReason::RangeUnavailable);
        }
        Ok(EventPublishReservation {
            sequence: entry.event.sequence,
            retention: Some(retention),
        })
    }

    pub(crate) fn prepare_publish(&self) -> Result<Option<EventHighWater>, HostAccessError> {
        let candidate = {
            let state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.next_sequence.checked_next().is_none() {
                return Err(protocol_violation("scientific event sequence exhausted"));
            }
            if state.entries.len() < state.capacity || state.reserved_front.is_some() {
                return Ok(None);
            }
            state.entries.front().cloned()
        };
        let Some(candidate) = candidate else {
            return Ok(None);
        };
        let reservation = match self.eviction_reservation(&candidate) {
            Ok(reservation) => reservation,
            Err(reason) => {
                return Ok(Some(EventHighWater {
                    batch_id: candidate.event.batch_id,
                    sequence: candidate.event.sequence,
                    reason,
                }));
            }
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.entries.len() < state.capacity || state.reserved_front.is_some() {
            return Ok(None);
        }
        let front_matches = state
            .entries
            .front()
            .is_some_and(|front| front.event.sequence == candidate.event.sequence);
        if front_matches {
            state.reserved_front = Some(reservation);
        }
        drop(state);
        if front_matches {
            Ok(None)
        } else {
            Err(protocol_violation(
                "scientific event hot ring changed while reserving its front slot",
            ))
        }
    }

    pub(crate) fn publish_pending(
        &self,
        batch_id: JournalBatchId,
        applied: AppliedCommand,
        boundary: Arc<ScientificBoundary>,
    ) -> Result<EventSequence, HostAccessError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.entries.len() >= state.capacity {
            let Some(reservation) = state.reserved_front.take() else {
                return Err(protocol_violation(
                    "scientific event published without a reserved hot-ring slot",
                ));
            };
            if state
                .entries
                .front()
                .is_none_or(|front| front.event.sequence != reservation.sequence)
            {
                return Err(protocol_violation(
                    "scientific event hot-ring reservation no longer names its front record",
                ));
            }
            let _retention = reservation.retention;
            state.entries.pop_front();
        }
        let sequence = state.next_sequence;
        state.next_sequence = sequence
            .checked_next()
            .ok_or_else(|| protocol_violation("scientific event sequence exhausted"))?;
        state.entries.push_back(JournaledScientificEvent {
            event: Arc::new(ScientificEvent {
                session_id: self.session_id,
                sequence,
                batch_id,
                tick: applied.tick,
                revisions: applied.revisions,
                boundary,
            }),
            commitment: EventCommitment::Pending,
        });
        state.published_total = state.published_total.saturating_add(1);
        self.hot.store(Arc::new(EventHotView::from(&*state)));
        drop(state);
        Ok(sequence)
    }

    pub(crate) fn cancel_publish_reservation(&self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .reserved_front = None;
    }

    pub(crate) fn update_commitment(
        &self,
        batch_id: JournalBatchId,
        event_sequence: EventSequence,
        commitment: EventCommitment,
    ) -> Result<(), HostAccessError> {
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(entry) = state
                .entries
                .iter_mut()
                .find(|entry| entry.event.batch_id == batch_id)
            {
                if entry.event.sequence != event_sequence {
                    return Err(protocol_violation(
                        "journal receipt event sequence does not match its retained batch",
                    ));
                }
                if entry.commitment == commitment
                    || matches!(
                        &entry.commitment,
                        EventCommitment::Durable | EventCommitment::Failed(_)
                    )
                    || matches!(
                        (&entry.commitment, &commitment),
                        (EventCommitment::CommittedVolatile, EventCommitment::Pending)
                    )
                {
                    return Ok(());
                }
                entry.commitment = commitment;
                self.hot.store(Arc::new(EventHotView::from(&*state)));
                drop(state);
                return Ok(());
            }
        }
        if self
            .reader
            .as_ref()
            .is_some_and(|reader| reader.contains_event_identity(event_sequence, batch_id))
        {
            Ok(())
        } else {
            Err(protocol_violation(
                "journal receipt referenced an unknown scientific event",
            ))
        }
    }

    /// Create an independent session-bound cursor before the first scientific event.
    #[must_use]
    pub const fn subscribe(&self) -> EventCursor {
        EventCursor::beginning(self.session_id)
    }

    /// Resume after one already-observed sequence in this exact host session.
    #[must_use]
    pub const fn resume_after(&self, sequence: EventSequence) -> EventCursor {
        EventCursor::after(self.session_id, sequence)
    }

    /// Poll the bounded hot ring without allocating per-subscriber state.
    ///
    /// # Errors
    ///
    /// Returns a typed access error for a cross-session or ahead-of-stream cursor.
    pub fn poll(&self, cursor: EventCursor, limit: usize) -> Result<EventPoll, HostAccessError> {
        if cursor.session_id != self.session_id {
            return Err(HostAccessError::EventSessionMismatch {
                expected: cursor.session_id,
                actual: self.session_id,
            });
        }
        let state = self.hot.load();
        let latest = EventSequence::new(state.next_sequence.get().saturating_sub(1));
        if cursor.last_seen > latest {
            return Err(protocol_violation(
                "scientific event cursor is ahead of the host",
            ));
        }
        let expected = cursor
            .last_seen
            .checked_next()
            .ok_or_else(|| protocol_violation("scientific event cursor sequence exhausted"))?;
        let Some(front) = state.entries.front() else {
            return Ok(EventPoll::Contiguous(EventPage {
                session_id: self.session_id,
                source: EventPageSource::Hot,
                events: Vec::new(),
                latest,
            }));
        };
        if expected < front.event.sequence {
            let missing = EventSequenceRange {
                first: expected,
                last: EventSequence::new(front.event.sequence.get() - 1),
            };
            let hot_available = EventSequenceRange {
                first: front.event.sequence,
                last: state
                    .entries
                    .back()
                    .map_or(front.event.sequence, |back| back.event.sequence),
            };
            let catch_up = self.reader.as_ref().map_or(
                EventCatchUpState::Unavailable(EventCatchUpUnavailableReason::NoReader),
                |reader| match reader.available_range() {
                    Some(range) if range.contains_range(missing) => {
                        EventCatchUpState::Available(EventCatchUpLocator {
                            session_id: self.session_id,
                            range: missing,
                            guarantee: reader.guarantee(),
                        })
                    }
                    Some(range) if range.contains(missing.last) => {
                        EventCatchUpState::Unavailable(EventCatchUpUnavailableReason::PartialRange)
                    }
                    _ => {
                        EventCatchUpState::Unavailable(EventCatchUpUnavailableReason::RangeExpired)
                    }
                },
            );
            return Ok(EventPoll::Gap(EventGap {
                session_id: self.session_id,
                expected,
                missing,
                hot_available,
                latest,
                catch_up,
            }));
        }
        let limit = limit.min(MAX_EVENT_PAGE_SIZE);
        let events = state
            .entries
            .iter()
            .filter(|entry| entry.event.sequence > cursor.last_seen)
            .take(limit)
            .cloned()
            .collect();
        Ok(EventPoll::Contiguous(EventPage {
            session_id: self.session_id,
            source: EventPageSource::Hot,
            events,
            latest,
        }))
    }

    /// Resolve one exact gap locator through the injected journal reader.
    ///
    /// # Errors
    ///
    /// Returns a typed access error when a reader violates the session or page contract.
    pub fn catch_up(
        &self,
        locator: EventCatchUpLocator,
        limit: usize,
    ) -> Result<EventCatchUp, HostAccessError> {
        if locator.session_id != self.session_id {
            return Ok(EventCatchUp::Unavailable {
                range: locator.range,
                reason: EventCatchUpUnavailableReason::SessionMismatch,
            });
        }
        let Some(reader) = &self.reader else {
            return Ok(EventCatchUp::Unavailable {
                range: locator.range,
                reason: EventCatchUpUnavailableReason::NoReader,
            });
        };
        let bounded_limit = limit.min(MAX_EVENT_PAGE_SIZE);
        let result = reader.read(locator, bounded_limit)?;
        let cursor = EventCursor::after(
            self.session_id,
            EventSequence::new(locator.range.first.get().saturating_sub(1)),
        );
        validate_event_catch_up(cursor, locator, &result, bounded_limit)?;
        Ok(result)
    }
}

/// Opaque host-session-bound cursor over canonical scientific events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventCursor {
    session_id: HostSessionId,
    last_seen: EventSequence,
}

impl EventCursor {
    /// Start before the first scientific event in one host session.
    #[must_use]
    pub const fn beginning(session_id: HostSessionId) -> Self {
        Self {
            session_id,
            last_seen: EventSequence::new(0),
        }
    }

    /// Resume after an already-observed event from one exact host session.
    #[must_use]
    pub const fn after(session_id: HostSessionId, sequence: EventSequence) -> Self {
        Self {
            session_id,
            last_seen: sequence,
        }
    }

    /// Bound host session.
    #[must_use]
    pub const fn session_id(self) -> HostSessionId {
        self.session_id
    }

    /// Last canonical event observed through this cursor.
    #[must_use]
    pub const fn last_seen(self) -> EventSequence {
        self.last_seen
    }
}

/// Opaque, host-session-bound cursor over latest-value render snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotSubscription {
    session_id: HostSessionId,
    last_seen: Option<SnapshotRevision>,
    skipped_revisions: u64,
}

impl SnapshotSubscription {
    /// Subscribe to the current publication from one exact host session.
    #[must_use]
    pub const fn current(session_id: HostSessionId) -> Self {
        Self {
            session_id,
            last_seen: None,
            skipped_revisions: 0,
        }
    }

    /// Resume after an already-observed publication from one exact host session.
    #[must_use]
    pub const fn after(session_id: HostSessionId, revision: SnapshotRevision) -> Self {
        Self {
            session_id,
            last_seen: Some(revision),
            skipped_revisions: 0,
        }
    }

    /// Host session to which this cursor is bound.
    #[must_use]
    pub const fn session_id(self) -> HostSessionId {
        self.session_id
    }

    /// Last snapshot observed through this subscription.
    #[must_use]
    pub const fn last_seen(self) -> Option<SnapshotRevision> {
        self.last_seen
    }

    /// Number of older render publications deliberately skipped by latest-value polling.
    #[must_use]
    pub const fn skipped_revisions(self) -> u64 {
        self.skipped_revisions
    }

    fn observe(
        &mut self,
        session_id: HostSessionId,
        revision: SnapshotRevision,
    ) -> Result<bool, HostAccessError> {
        if session_id != self.session_id {
            return Err(HostAccessError::SnapshotSessionMismatch {
                expected: self.session_id,
                actual: session_id,
            });
        }
        let Some(last_seen) = self.last_seen else {
            self.last_seen = Some(revision);
            return Ok(true);
        };
        if revision <= last_seen {
            return Ok(false);
        }
        self.skipped_revisions = self.skipped_revisions.saturating_add(
            revision
                .get()
                .saturating_sub(last_seen.get())
                .saturating_sub(1),
        );
        self.last_seen = Some(revision);
        Ok(true)
    }
}

/// Failure to reach or trust the opaque host port.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HostAccessError {
    /// The transport or in-process port is no longer connected.
    #[error("host port disconnected")]
    Disconnected,
    /// A stable command id was retried with a different canonical envelope.
    #[error("command id {command_id} was already used for a different command envelope")]
    CommandIdCollision {
        /// Reused id whose original payload remains authoritative.
        command_id: CommandId,
    },
    /// The bounded pre-admission evidence lane cannot accept another terminal status.
    #[error("command evidence backpressure at capacity {capacity}")]
    CommandEvidenceBackpressure {
        /// Maximum number of returned pre-admission statuses awaiting journal admission.
        capacity: usize,
    },
    /// The ordered shutdown barrier closed the command-evidence ingress path.
    #[error("command evidence ingress is closed while host lifecycle is {lifecycle:?}")]
    CommandEvidenceClosed {
        /// Lifecycle observed when the caller attempted submission.
        lifecycle: HostLifecycle,
    },
    /// A host implementation violated the public ordering contract.
    #[error("host protocol violation: {message}")]
    ProtocolViolation {
        /// Diagnostic identifying the violated invariant.
        message: String,
    },
    /// A snapshot cursor was reused against a different host session.
    #[error("snapshot session {actual:?} does not match subscription session {expected:?}")]
    SnapshotSessionMismatch {
        /// Session bound to the snapshot cursor.
        expected: HostSessionId,
        /// Session reported by the snapshot source.
        actual: HostSessionId,
    },
    /// A scientific-event cursor or locator was reused against another host session.
    #[error("event session {actual:?} does not match cursor session {expected:?}")]
    EventSessionMismatch {
        /// Session bound to the event cursor.
        expected: HostSessionId,
        /// Session reported by the event source.
        actual: HostSessionId,
    },
    /// A manual driver belongs to a different host than the frontend's ingress port.
    #[error("manual driver session {actual:?} does not match client session {expected:?}")]
    DriverSessionMismatch {
        /// Session bound to the frontend's client port.
        expected: HostSessionId,
        /// Session reported by the supplied manual driver.
        actual: HostSessionId,
    },
}

/// A null-frontend submission failure that preserves an indeterminate command envelope.
#[derive(Debug, Error)]
pub enum NullFrontendSubmissionError {
    /// The frontend exhausted its stable namespaced command-id sequence before submitting.
    #[error("command id sequence exhausted")]
    CommandIdExhausted,
    /// Host access failed after an exact retryable envelope had been prepared.
    #[error("null frontend command submission failed: {source}")]
    HostAccess {
        /// Exact envelope whose admission may be indeterminate.
        envelope: CommandEnvelope,
        /// Port failure observed by the frontend.
        #[source]
        source: HostAccessError,
    },
}

impl NullFrontendSubmissionError {
    /// Exact retryable envelope, when the failure happened after preparation.
    #[must_use]
    pub const fn envelope(&self) -> Option<&CommandEnvelope> {
        match self {
            Self::CommandIdExhausted => None,
            Self::HostAccess { envelope, .. } => Some(envelope),
        }
    }

    /// Consume the error and recover the exact retryable envelope.
    #[must_use]
    pub fn into_envelope(self) -> Option<CommandEnvelope> {
        match self {
            Self::CommandIdExhausted => None,
            Self::HostAccess { envelope, .. } => Some(envelope),
        }
    }
}

/// Synchronous, renderer-neutral client port implemented by a host handle.
///
/// Implementations may use channels internally, but the concrete transport is
/// intentionally hidden behind [`HostClient`].
pub trait HostPort {
    /// Stable identity of the host reached through this port.
    fn session_id(&self) -> HostSessionId;

    /// Submit or retry a logical command.
    fn submit(&mut self, envelope: CommandEnvelope) -> Result<CommandStatus, HostAccessError>;

    /// Look up the latest durable in-process knowledge for a command id.
    fn command_status(
        &mut self,
        command_id: CommandId,
    ) -> Result<Option<CommandStatus>, HostAccessError>;

    /// Return the newest snapshot when it is newer than `after`.
    fn snapshot_after(
        &mut self,
        after: Option<SnapshotRevision>,
    ) -> Result<Option<Arc<RenderSnapshot>>, HostAccessError>;

    /// Return at most `limit` events whose sequence is strictly greater than the cursor.
    fn events_after(
        &mut self,
        cursor: ProtocolEventSequence,
        limit: usize,
    ) -> Result<Vec<HostEvent>, HostAccessError>;

    /// Poll the bounded canonical scientific-event hot ring.
    fn poll_events(
        &mut self,
        cursor: EventCursor,
        limit: usize,
    ) -> Result<EventPoll, HostAccessError>;

    /// Resolve one exact scientific-event catch-up locator.
    fn catch_up_events(
        &mut self,
        locator: EventCatchUpLocator,
        limit: usize,
    ) -> Result<EventCatchUp, HostAccessError>;
}

/// Optional extension for deterministic same-thread and browser-owned hosts.
pub trait ManualHostDriver {
    /// Stable identity of the host owned by this driver.
    fn session_id(&self) -> HostSessionId;

    /// Drive the host to one explicit monotonic time boundary.
    fn drive(&mut self, now: ManualInstant) -> Result<DriveReceipt, HostAccessError>;
}

/// Result of one explicit manual-drive boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriveReceipt {
    /// Time boundary supplied by the driver.
    pub now: ManualInstant,
    /// Commands whose application completed during this drive.
    pub commands_completed: usize,
    /// Scientific transitions completed during this drive.
    pub scientific_steps: usize,
    /// Speed-scaled automatic science opportunities due at this boundary.
    ///
    /// This excludes an explicit [`HostCommand::Step`].
    pub automatic_steps_due: u64,
    /// Whole automatic opportunities deliberately discarded by the bounded
    /// catch-up policy. Fractional cadence credit is retained.
    pub automatic_steps_skipped: u64,
    /// Scientific revision visible after this drive.
    pub scientific_revision: ScientificRevision,
    /// Snapshots published during this drive.
    pub snapshots_published: usize,
    /// Canonical scientific events published during this drive.
    pub events_published: usize,
    /// Typed reason science could not make further progress, when applicable.
    pub blocker: Option<HostBlocker>,
}

/// Typed client that owns, but never exposes, its concrete host port.
#[derive(Clone)]
pub struct HostClient<P> {
    port: P,
}

impl<P: HostPort> HostClient<P> {
    /// Wrap one concrete port behind the typed client API.
    #[must_use]
    pub const fn new(port: P) -> Self {
        Self { port }
    }

    /// Submit a new command or retry an existing idempotency key.
    pub fn submit(&mut self, envelope: CommandEnvelope) -> Result<CommandStatus, HostAccessError> {
        let requested_id = envelope.command_id;
        let status = self.port.submit(envelope)?;
        if status.command_id() != requested_id {
            return Err(protocol_violation(
                "submission returned a different command id",
            ));
        }
        status
            .validate()
            .map_err(|error| protocol_violation(error.to_string()))?;
        Ok(status)
    }

    /// Look up a command after the submitting client has disconnected or restarted.
    pub fn command_status(
        &mut self,
        command_id: CommandId,
    ) -> Result<Option<CommandStatus>, HostAccessError> {
        let status = self.port.command_status(command_id)?;
        if status
            .as_ref()
            .is_some_and(|status| status.command_id() != command_id)
        {
            return Err(protocol_violation("lookup returned a different command id"));
        }
        if let Some(status) = &status {
            status
                .validate()
                .map_err(|error| protocol_violation(error.to_string()))?;
        }
        Ok(status)
    }

    /// Create a snapshot subscription starting with the current publication.
    #[must_use]
    pub fn subscribe_snapshots(&self) -> SnapshotSubscription {
        SnapshotSubscription::current(self.port.session_id())
    }

    /// Poll one immutable snapshot and advance the subscription only on success.
    pub fn poll_snapshot(
        &mut self,
        subscription: &mut SnapshotSubscription,
    ) -> Result<Option<Arc<RenderSnapshot>>, HostAccessError> {
        let session_id = self.port.session_id();
        if subscription.session_id() != session_id {
            return Err(HostAccessError::SnapshotSessionMismatch {
                expected: subscription.session_id(),
                actual: session_id,
            });
        }
        let snapshot = self.port.snapshot_after(subscription.last_seen)?;
        if let Some(snapshot) = snapshot {
            if snapshot.session_id != session_id {
                return Err(HostAccessError::SnapshotSessionMismatch {
                    expected: session_id,
                    actual: snapshot.session_id,
                });
            }
            if !subscription.observe(snapshot.session_id, snapshot.revision)? {
                return Err(protocol_violation(
                    "snapshot revision did not advance beyond the subscription",
                ));
            }
            Ok(Some(snapshot))
        } else {
            Ok(None)
        }
    }

    /// Create a cursor over bounded ephemeral command/lifecycle/health notifications.
    #[must_use]
    pub fn protocol_event_cursor(&self) -> ProtocolEventCursor {
        ProtocolEventCursor::beginning(self.port.session_id())
    }

    /// Read contiguous ephemeral notifications and advance only after full validation.
    pub fn read_protocol_events(
        &mut self,
        cursor: &mut ProtocolEventCursor,
        limit: usize,
    ) -> Result<Vec<HostEvent>, HostAccessError> {
        let session_id = self.port.session_id();
        if cursor.session_id != session_id {
            return Err(HostAccessError::EventSessionMismatch {
                expected: cursor.session_id,
                actual: session_id,
            });
        }
        let events = self.port.events_after(cursor.last_seen, limit)?;
        if events.len() > limit {
            return Err(protocol_violation(
                "event port exceeded the requested limit",
            ));
        }
        let mut previous = cursor.last_seen;
        for event in &events {
            let expected = previous
                .checked_next()
                .ok_or_else(|| protocol_violation("event cursor sequence exhausted"))?;
            if event.sequence != expected {
                return Err(protocol_violation("event sequence was not contiguous"));
            }
            if event.session_id != session_id {
                return Err(HostAccessError::EventSessionMismatch {
                    expected: session_id,
                    actual: event.session_id,
                });
            }
            if let HostEventKind::CommandStatusChanged(status) = &event.kind {
                status
                    .validate()
                    .map_err(|error| protocol_violation(error.to_string()))?;
            }
            previous = event.sequence;
        }
        cursor.last_seen = previous;
        Ok(events)
    }

    /// Create a session-bound cursor over canonical scientific events.
    #[must_use]
    pub fn event_cursor(&self) -> EventCursor {
        EventCursor::beginning(self.port.session_id())
    }

    /// Poll canonical scientific events, advancing only after a valid contiguous page.
    pub fn read_events(
        &mut self,
        cursor: &mut EventCursor,
        limit: usize,
    ) -> Result<EventPoll, HostAccessError> {
        let session_id = self.port.session_id();
        if cursor.session_id != session_id {
            return Err(HostAccessError::EventSessionMismatch {
                expected: cursor.session_id,
                actual: session_id,
            });
        }
        let poll = self.port.poll_events(*cursor, limit)?;
        match &poll {
            EventPoll::Contiguous(page) => {
                if page.source != EventPageSource::Hot {
                    return Err(protocol_violation(
                        "hot-ring poll returned a non-hot scientific event page",
                    ));
                }
                if limit.min(MAX_EVENT_PAGE_SIZE) != 0
                    && page.events.is_empty()
                    && page.latest > cursor.last_seen
                {
                    return Err(protocol_violation(
                        "hot-ring poll reported newer events without making bounded progress",
                    ));
                }
                cursor.last_seen = validate_event_page(*cursor, page, limit)?;
            }
            EventPoll::Gap(gap) => validate_event_gap(*cursor, *gap)?,
        }
        Ok(poll)
    }

    /// Resolve an exact gap locator and advance only through a valid contiguous catch-up page.
    pub fn catch_up_events(
        &mut self,
        cursor: &mut EventCursor,
        locator: EventCatchUpLocator,
        limit: usize,
    ) -> Result<EventCatchUp, HostAccessError> {
        let session_id = self.port.session_id();
        if cursor.session_id != session_id {
            return Err(HostAccessError::EventSessionMismatch {
                expected: cursor.session_id,
                actual: session_id,
            });
        }
        if locator.session_id != session_id {
            return Err(HostAccessError::EventSessionMismatch {
                expected: session_id,
                actual: locator.session_id,
            });
        }
        let expected = cursor
            .last_seen
            .checked_next()
            .ok_or_else(|| protocol_violation("scientific event cursor sequence exhausted"))?;
        if !locator.range.contains(expected) {
            return Err(protocol_violation(
                "catch-up locator does not begin at the cursor's required sequence",
            ));
        }
        let effective = EventCatchUpLocator {
            session_id,
            range: EventSequenceRange {
                first: expected,
                last: locator.range.last,
            },
            guarantee: locator.guarantee,
        };
        let result = self.port.catch_up_events(effective, limit)?;
        let next = validate_event_catch_up(*cursor, effective, &result, limit)?;
        if matches!(&result, EventCatchUp::Contiguous(_)) {
            cursor.last_seen = next;
        }
        Ok(result)
    }

    /// Explicitly accept display truncation and position the cursor before the hot range.
    ///
    /// Merely observing a gap never calls this method or advances the cursor.
    pub fn accept_event_gap(
        &self,
        cursor: &mut EventCursor,
        gap: EventGap,
    ) -> Result<(), HostAccessError> {
        let session_id = self.port.session_id();
        if cursor.session_id != session_id {
            return Err(HostAccessError::EventSessionMismatch {
                expected: cursor.session_id,
                actual: session_id,
            });
        }
        validate_event_gap(*cursor, gap)?;
        cursor.last_seen = gap.resume_after();
        Ok(())
    }
}

fn validate_event_page(
    cursor: EventCursor,
    page: &EventPage,
    limit: usize,
) -> Result<EventSequence, HostAccessError> {
    if page.session_id != cursor.session_id {
        return Err(HostAccessError::EventSessionMismatch {
            expected: cursor.session_id,
            actual: page.session_id,
        });
    }
    if page.events.len() > limit.min(MAX_EVENT_PAGE_SIZE) {
        return Err(protocol_violation(
            "scientific event port exceeded the bounded page limit",
        ));
    }
    let mut previous = cursor.last_seen;
    let mut previous_batch_sequence = None;
    for entry in &page.events {
        let expected = previous
            .checked_next()
            .ok_or_else(|| protocol_violation("scientific event cursor sequence exhausted"))?;
        if entry.event.sequence != expected {
            return Err(protocol_violation(
                "scientific event page was not contiguous",
            ));
        }
        if entry.event.session_id != cursor.session_id
            || entry.event.batch_id.session_id() != cursor.session_id
            || !scientific_boundary_matches_event(&entry.event)
        {
            return Err(protocol_violation(
                "scientific event page contains incoherent source identity",
            ));
        }
        if previous_batch_sequence
            .is_some_and(|sequence| sequence >= entry.event.batch_id.sequence())
        {
            return Err(protocol_violation(
                "scientific event page journal batches did not advance",
            ));
        }
        match page.source {
            EventPageSource::LiveMemory
                if !matches!(
                    entry.commitment,
                    EventCommitment::CommittedVolatile | EventCommitment::Durable
                ) =>
            {
                return Err(protocol_violation(
                    "live-memory catch-up returned an uncommitted event",
                ));
            }
            EventPageSource::Durable if entry.commitment != EventCommitment::Durable => {
                return Err(protocol_violation(
                    "durable catch-up returned a non-durable event",
                ));
            }
            EventPageSource::Hot | EventPageSource::LiveMemory | EventPageSource::Durable => {}
        }
        previous_batch_sequence = Some(entry.event.batch_id.sequence());
        previous = entry.event.sequence;
    }
    if previous > page.latest {
        return Err(protocol_violation(
            "scientific event page advanced beyond its latest watermark",
        ));
    }
    Ok(previous)
}

fn scientific_boundary_matches_event(event: &ScientificEvent) -> bool {
    let boundary = event.boundary.as_ref();
    boundary.events.tick == event.tick
        && boundary.summary.tick == event.tick
        && boundary.config_revision == event.revisions.config.get()
        && boundary
            .births
            .iter()
            .all(|record| record.tick == event.tick)
        && boundary
            .deaths
            .iter()
            .all(|record| record.tick == event.tick)
        && boundary
            .resource_tick
            .as_ref()
            .is_none_or(|record| record.tick == event.tick)
}

fn validate_event_catch_up(
    cursor: EventCursor,
    locator: EventCatchUpLocator,
    result: &EventCatchUp,
    limit: usize,
) -> Result<EventSequence, HostAccessError> {
    let expected = cursor
        .last_seen
        .checked_next()
        .ok_or_else(|| protocol_violation("scientific event cursor sequence exhausted"))?;
    if locator.session_id != cursor.session_id
        || !locator.range.is_valid()
        || locator.range.first != expected
    {
        return Err(protocol_violation(
            "catch-up locator is incoherent with the scientific event cursor",
        ));
    }
    match result {
        EventCatchUp::Contiguous(page) => {
            let expected_source = match locator.guarantee {
                EventCatchUpGuarantee::LiveMemory => EventPageSource::LiveMemory,
                EventCatchUpGuarantee::CrashDurable => EventPageSource::Durable,
            };
            if page.source != expected_source {
                return Err(protocol_violation(
                    "catch-up page source does not match its locator guarantee",
                ));
            }
            if limit.min(MAX_EVENT_PAGE_SIZE) != 0 && page.events.is_empty() {
                return Err(protocol_violation(
                    "catch-up reader returned no progress for a nonempty bounded request",
                ));
            }
            let next = validate_event_page(cursor, page, limit)?;
            if next > locator.range.last {
                return Err(protocol_violation(
                    "catch-up page advanced beyond its exact locator range",
                ));
            }
            Ok(next)
        }
        EventCatchUp::Unavailable { range, .. } => {
            if *range != locator.range {
                return Err(protocol_violation(
                    "catch-up reader reported an unrelated unavailable range",
                ));
            }
            Ok(cursor.last_seen)
        }
    }
}

fn validate_event_gap(cursor: EventCursor, gap: EventGap) -> Result<(), HostAccessError> {
    if gap.session_id != cursor.session_id {
        return Err(HostAccessError::EventSessionMismatch {
            expected: cursor.session_id,
            actual: gap.session_id,
        });
    }
    let expected = cursor
        .last_seen
        .checked_next()
        .ok_or_else(|| protocol_violation("scientific event cursor sequence exhausted"))?;
    if !gap.missing.is_valid()
        || !gap.hot_available.is_valid()
        || gap.expected != expected
        || gap.missing.first != expected
        || gap.missing.last.checked_next() != Some(gap.hot_available.first)
        || gap.hot_available.last != gap.latest
    {
        return Err(protocol_violation(
            "scientific event gap metadata is incoherent",
        ));
    }
    if let EventCatchUpState::Available(locator) = &gap.catch_up {
        let session_mismatch = locator.session_id != gap.session_id;
        let range_mismatch = locator.range != gap.missing;
        if session_mismatch {
            return Err(protocol_violation(
                "scientific event gap locator belongs to a different session",
            ));
        }
        if range_mismatch {
            return Err(protocol_violation(
                "scientific event gap locator does not cover its missing range",
            ));
        }
    }
    Ok(())
}

/// Headless reference frontend used by conformance tests and embedders.
///
/// It exercises only public client operations and owns no world, lock, storage
/// connection, renderer, server, or scheduler.
pub struct NullFrontend<P> {
    client: HostClient<P>,
    host_session_id: HostSessionId,
    client_namespace: u64,
    next_sequence: Option<u64>,
    snapshots: SnapshotSubscription,
    protocol_events: ProtocolEventCursor,
    events: EventCursor,
    last_drive: Option<ManualInstant>,
}

impl<P: HostPort> NullFrontend<P> {
    /// Construct a frontend with a stable command-id namespace.
    #[must_use]
    pub fn new(port: P, client_namespace: u64) -> Self {
        let host_session_id = port.session_id();
        Self {
            client: HostClient::new(port),
            host_session_id,
            client_namespace,
            next_sequence: Some(1),
            snapshots: SnapshotSubscription::current(host_session_id),
            protocol_events: ProtocolEventCursor::beginning(host_session_id),
            events: EventCursor::beginning(host_session_id),
            last_drive: None,
        }
    }

    /// Submit an arbitrary command with an optional control-revision guard.
    pub fn submit(
        &mut self,
        command: HostCommand,
        expected_control_revision: Option<ControlRevision>,
    ) -> Result<CommandStatus, NullFrontendSubmissionError> {
        let sequence = self
            .next_sequence
            .ok_or(NullFrontendSubmissionError::CommandIdExhausted)?;
        self.next_sequence = sequence.checked_add(1);
        let mut envelope = CommandEnvelope::new(
            CommandId::from_client_sequence(self.client_namespace, sequence),
            command,
        );
        envelope.expected_control_revision = expected_control_revision;
        self.submit_envelope(envelope)
    }

    /// Submit or retry an already prepared envelope without changing its stable identity.
    pub fn submit_envelope(
        &mut self,
        envelope: CommandEnvelope,
    ) -> Result<CommandStatus, NullFrontendSubmissionError> {
        let retry_envelope = envelope.clone();
        self.client
            .submit(envelope)
            .map_err(|source| NullFrontendSubmissionError::HostAccess {
                envelope: retry_envelope,
                source,
            })
    }

    /// Pause automatic ticks.
    pub fn pause(&mut self) -> Result<CommandStatus, NullFrontendSubmissionError> {
        self.submit(HostCommand::Pause, None)
    }

    /// Resume automatic ticks.
    pub fn resume(&mut self) -> Result<CommandStatus, NullFrontendSubmissionError> {
        self.submit(HostCommand::Resume, None)
    }

    /// Set the playback multiplier.
    pub fn set_speed(&mut self, speed: f32) -> Result<CommandStatus, NullFrontendSubmissionError> {
        self.submit(HostCommand::SetSpeed(speed), None)
    }

    /// Request exactly one scientific tick.
    pub fn step(&mut self) -> Result<CommandStatus, NullFrontendSubmissionError> {
        self.submit(HostCommand::Step, None)
    }

    /// Replace the active simulation configuration.
    pub fn update_config(
        &mut self,
        config: ScriptBotsConfig,
    ) -> Result<CommandStatus, NullFrontendSubmissionError> {
        self.submit(HostCommand::UpdateConfig(Box::new(config)), None)
    }

    /// Request orderly host shutdown.
    pub fn shutdown(&mut self) -> Result<CommandStatus, NullFrontendSubmissionError> {
        self.submit(HostCommand::Shutdown, None)
    }

    /// Look up the latest status for any command id.
    pub fn command_status(
        &mut self,
        command_id: CommandId,
    ) -> Result<Option<CommandStatus>, HostAccessError> {
        self.client.command_status(command_id)
    }

    /// Poll the next immutable snapshot for this frontend.
    pub fn poll_snapshot(&mut self) -> Result<Option<Arc<RenderSnapshot>>, HostAccessError> {
        self.client.poll_snapshot(&mut self.snapshots)
    }

    /// Read bounded ephemeral host notifications and advance this frontend's cursor.
    pub fn read_protocol_events(
        &mut self,
        limit: usize,
    ) -> Result<Vec<HostEvent>, HostAccessError> {
        self.client
            .read_protocol_events(&mut self.protocol_events, limit)
    }

    /// Poll canonical scientific events and advance only through contiguous pages.
    pub fn read_events(&mut self, limit: usize) -> Result<EventPoll, HostAccessError> {
        self.client.read_events(&mut self.events, limit)
    }

    /// Resolve one exact scientific-event gap and advance through a valid catch-up page.
    pub fn catch_up_events(
        &mut self,
        locator: EventCatchUpLocator,
        limit: usize,
    ) -> Result<EventCatchUp, HostAccessError> {
        self.client
            .catch_up_events(&mut self.events, locator, limit)
    }

    /// Explicitly accept an unavailable display prefix and resume at the hot window.
    pub fn accept_event_gap(&mut self, gap: EventGap) -> Result<(), HostAccessError> {
        self.client.accept_event_gap(&mut self.events, gap)
    }

    /// Drive a separately owned synchronous host to a caller-owned time boundary.
    pub fn drive_at(
        &mut self,
        driver: &mut impl ManualHostDriver,
        now: ManualInstant,
    ) -> Result<DriveReceipt, HostAccessError> {
        if self.last_drive.is_some_and(|last_drive| now < last_drive) {
            return Err(protocol_violation(
                "null frontend manual time moved backwards",
            ));
        }
        let driver_session_id = driver.session_id();
        if driver_session_id != self.host_session_id {
            return Err(HostAccessError::DriverSessionMismatch {
                expected: self.host_session_id,
                actual: driver_session_id,
            });
        }
        let receipt = driver.drive(now)?;
        if receipt.now != now {
            return Err(protocol_violation(
                "manual driver returned a receipt for a different time boundary",
            ));
        }
        self.last_drive = Some(receipt.now);
        Ok(receipt)
    }
}

fn protocol_violation(message: impl Into<String>) -> HostAccessError {
    HostAccessError::ProtocolViolation {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{
        AgentData, AgentId, AgentIdentity, AgentRuntime, AgentState, BirthOrigin, BrainBinding,
        BrainFamilyId, BrainGenomeEnvelope, BrainProvenance, CombatEventFlags, DeathCause,
        DynamicSnapshotSummary, DynamicSnapshotWorld, MetricSample, PersistenceEvent,
        PersistenceEventKind, Position, ReplayEvent, ReplayEventKind,
    };
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::hint::black_box;
    use std::mem::size_of;
    use std::sync::{
        Barrier, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    };
    use std::time::Instant;

    static NEXT_FAKE_SESSION_ID: AtomicU64 = AtomicU64::new(1);

    fn projection_agent(
        uid: u16,
        position: [f32; 2],
        energy: f32,
        health: f32,
        age: u32,
    ) -> DynamicAgentSnapshot {
        let uid_u64 = u64::from(uid);
        let uid_scalar = f32::from(uid);
        DynamicAgentSnapshot {
            id: uid_u64.saturating_add(10_000),
            uid: AgentUid(uid_u64),
            position,
            velocity: [uid_scalar, -uid_scalar],
            heading: uid_scalar * 0.1,
            health,
            energy,
            color: [0.2, 0.4, 0.6],
            spike_length: 0.25,
            boost: uid.is_multiple_of(2),
            age,
            generation: Generation(u32::from(uid)),
            herbivore_tendency: 0.25,
            brain_key: Some(uid_u64.saturating_add(100)),
        }
    }

    fn projection_summary(tick: u64) -> TickSummary {
        let fixture_tick =
            u16::try_from(tick).expect("projection fixture tick fits exact f32 integer range");
        let tick_scalar = f32::from(fixture_tick);
        TickSummary {
            tick: Tick(tick),
            agent_count: 3,
            births: usize::from(tick.is_multiple_of(2)),
            deaths: usize::from(tick.is_multiple_of(3)),
            total_energy: tick_scalar,
            average_energy: tick_scalar / 3.0,
            average_health: 0.75,
            max_age: u32::from(fixture_tick),
            spike_hits: u32::from(tick.is_multiple_of(4)),
        }
    }

    fn projection_snapshot() -> RenderSnapshot {
        let summary_history = Arc::new((1..=10).map(projection_summary).collect::<Vec<_>>());
        RenderSnapshot {
            session_id: HostSessionId::new(44),
            revision: SnapshotRevision::new(7),
            revisions: HostRevisions {
                control: ControlRevision::new(3),
                scientific: ScientificRevision::new(10),
                config: ConfigRevision::new(2),
            },
            playback: PlaybackSnapshot::default(),
            lifecycle: HostLifecycle::Running,
            health: HostHealth::Healthy,
            command_queue_depth: 0,
            last_applied_command: None,
            completed_summary: summary_history.last().cloned(),
            summary_history,
            layers: SnapshotLayers {
                revisions: SnapshotLayerRevisions {
                    terrain: LayerRevision::new(2),
                    food: LayerRevision::new(3),
                    hydrology: LayerRevision::new(0),
                },
                terrain: Arc::new(TerrainLayerSnapshot {
                    width: 1,
                    height: 1,
                    cell_size: 100,
                    tiles: vec![TerrainTileSnapshot {
                        kind: TerrainKind::Grass,
                        elevation: 0.5,
                        moisture: 0.5,
                        accent: 0.0,
                        fertility_bias: 0.5,
                        temperature_bias: 0.5,
                        palette_index: 3,
                    }],
                }),
                food: Arc::new(FoodLayerSnapshot {
                    width: 1,
                    height: 1,
                    cells: vec![1.0],
                }),
                hydrology: None,
            },
            build: SnapshotBuildStats::default(),
            world: DynamicWorldSnapshot {
                tick: 10,
                epoch: 1,
                world: DynamicSnapshotWorld {
                    width: 100,
                    height: 100,
                    closed: true,
                },
                summary: DynamicSnapshotSummary {
                    agent_count: 3,
                    births: 1,
                    deaths: 0,
                    total_energy: 25.0,
                    average_energy: 25.0 / 3.0,
                    average_health: 0.75,
                },
                agents: vec![
                    projection_agent(1, [99.0, 50.0], 10.0, 0.5, 5),
                    projection_agent(2, [2.0, 50.0], 10.0, 1.0, 7),
                    projection_agent(3, [50.0, 50.0], 5.0, 0.75, 20),
                ],
            },
        }
    }

    fn projection_request(client_id: u64) -> ProjectionRequest {
        ProjectionRequest {
            client_id: ProjectionClientId::new(client_id),
            viewport: ProjectionViewport {
                width: 20,
                height: 10,
            },
            camera: ProjectionCamera {
                center: [0.0, 50.0],
                zoom: 4.0,
            },
            selection: ProjectionSelection {
                focused: Some(AgentUid(1)),
                selected: vec![AgentUid(3), AgentUid(2), AgentUid(1), AgentUid(1)],
            },
            detail: ProjectionDetail::Kinematics,
            chart_window: 10,
            chart_points: 4,
            top_k: 2,
            ranking: ProjectionRanking::Energy,
        }
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one coherent two-client oracle keeps request isolation, source immutability, wrap, detail, chart, and cache evidence together"
    )]
    #[allow(
        clippy::float_cmp,
        reason = "camera centers are copied exactly and seam offsets use exactly representable fixture coordinates"
    )]
    fn pure_projections_isolate_clients_cover_wrap_and_preserve_source() {
        let snapshot = projection_snapshot();
        let source_before = snapshot.clone();
        let expected_source = projection_source_key(&snapshot);
        let request_a = projection_request(1);
        let mut request_b = projection_request(2);
        request_b.viewport = ProjectionViewport {
            width: 12,
            height: 8,
        };
        request_b.camera.center = [50.0, 50.0];
        request_b.camera.zoom = 2.0;
        request_b.selection = ProjectionSelection {
            focused: Some(AgentUid(3)),
            selected: vec![AgentUid(3)],
        };
        request_b.chart_window = 3;
        request_b.chart_points = 2;
        request_b.detail = ProjectionDetail::Vitals;
        request_b.ranking = ProjectionRanking::Age;

        assert_ne!(request_a.client_id, request_b.client_id);
        assert_ne!(request_a.viewport, request_b.viewport);
        assert_ne!(request_a.camera, request_b.camera);
        assert_ne!(request_a.selection, request_b.selection);
        assert_ne!(request_a.detail, request_b.detail);
        assert_ne!(request_a.chart_window, request_b.chart_window);
        assert_ne!(request_a.chart_points, request_b.chart_points);

        let mut broker = ProjectionBroker::new(2).expect("bounded broker");
        let a_first = broker
            .project(&snapshot, &request_a, ProjectionLimits::default())
            .expect("client A projection");
        let b = broker
            .project(&snapshot, &request_b, ProjectionLimits::default())
            .expect("client B projection");
        let a_second = broker
            .project(&snapshot, &request_a, ProjectionLimits::default())
            .expect("client A exact cache hit");

        assert!(Arc::ptr_eq(&a_first, &a_second));
        assert_ne!(a_first.as_ref(), b.as_ref());
        assert_eq!(a_first.source, expected_source);
        assert_eq!(b.source, expected_source);
        assert_eq!(a_first.request.client_id, request_a.client_id);
        assert_eq!(b.request.client_id, request_b.client_id);
        assert_eq!(a_first.transform.viewport, request_a.viewport);
        assert_eq!(b.transform.viewport, request_b.viewport);
        assert_eq!(a_first.transform.center, request_a.camera.center);
        assert_eq!(b.transform.center, request_b.camera.center);
        assert_eq!(a_first.request.detail, ProjectionDetail::Kinematics);
        assert_eq!(b.request.detail, ProjectionDetail::Vitals);
        assert_eq!(
            a_first.request.selection.selected,
            [AgentUid(1), AgentUid(2), AgentUid(3)]
        );
        assert_eq!(a_first.selected_agents.len(), 3);
        let offscreen_selected = a_first
            .selected_agents
            .iter()
            .find(|agent| agent.uid == AgentUid(3))
            .expect("client A selected offscreen agent");
        assert!(
            !(0.0..f64::from(request_a.viewport.width))
                .contains(&offscreen_selected.canvas_position[0])
        );
        let offscreen_detail = offscreen_selected
            .detail
            .as_ref()
            .expect("client A requested offscreen kinematics");
        assert_eq!(offscreen_detail.age, 20);
        assert!(offscreen_detail.velocity.is_some());
        assert!(offscreen_detail.heading.is_some());
        assert!(offscreen_detail.spike_length.is_some());
        assert!(offscreen_detail.boost.is_some());
        assert!(
            !a_first
                .visible_agents
                .iter()
                .any(|agent| agent.uid == offscreen_selected.uid)
        );
        assert_eq!(b.request.selection.selected, [AgentUid(3)]);
        let b_focused = b.focused_agent.as_ref().expect("client B focused agent");
        assert_eq!(b_focused.uid, AgentUid(3));
        assert!(b_focused.detail.as_ref().is_some_and(|detail| {
            detail.velocity.is_none()
                && detail.heading.is_none()
                && detail.spike_length.is_none()
                && detail.boost.is_none()
        }));
        assert_eq!(a_first.top_agents[0].uid, AgentUid(1));
        assert_eq!(a_first.top_agents[1].uid, AgentUid(2));
        assert_eq!(b.top_agents[0].uid, AgentUid(3));
        assert_eq!(
            a_first.chart.first().expect("first chart point").tick,
            Tick(1)
        );
        assert_eq!(
            a_first.chart.last().expect("last chart point").tick,
            Tick(10)
        );
        assert_eq!(b.chart.first().expect("B first chart point").tick, Tick(8));
        assert_eq!(b.chart.last().expect("B last chart point").tick, Tick(10));
        assert_eq!(
            a_first
                .focused_agent
                .as_ref()
                .expect("focused seam agent")
                .wrap_offset,
            [-100.0, 0.0]
        );
        assert!(a_first.build.top_k_peak <= usize::from(request_a.top_k));
        assert_eq!(a_first.build.agents_examined, snapshot.world.agents.len());
        assert_eq!(snapshot, source_before);
        assert_eq!(broker.hits(), 1);
        assert_eq!(broker.misses(), 2);
        assert_eq!(broker.len(), 2);
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one projection boundary oracle covers every validation class plus entry and byte cache bounds"
    )]
    #[allow(
        clippy::float_cmp,
        reason = "the multi-wrap fixture uses integer-valued coordinates that are exactly representable as f64"
    )]
    fn projection_validation_cache_eviction_and_chart_truncation_are_explicit() {
        let snapshot = projection_snapshot();
        let limits = ProjectionLimits::default();
        let mut invalid = projection_request(1);
        invalid.camera.zoom = f32::NAN;
        assert_eq!(
            project_snapshot(&snapshot, &invalid, limits),
            Err(ProjectionError::InvalidCamera)
        );
        invalid = projection_request(1);
        invalid.viewport.width = 0;
        assert_eq!(
            project_snapshot(&snapshot, &invalid, limits),
            Err(ProjectionError::EmptyViewport)
        );
        invalid = projection_request(1);
        invalid.top_k = limits.max_top_k.saturating_add(1);
        assert!(matches!(
            project_snapshot(&snapshot, &invalid, limits),
            Err(ProjectionError::TopKTooLarge { .. })
        ));
        invalid = projection_request(1);
        invalid.camera.zoom = f32::MIN_POSITIVE;
        assert_eq!(
            project_snapshot(&snapshot, &invalid, limits),
            Err(ProjectionError::ScaleOutOfRange)
        );
        invalid = projection_request(1);
        invalid.viewport = ProjectionViewport {
            width: 1_000,
            height: 1_000,
        };
        invalid.camera.zoom = f32::MAX;
        assert_eq!(
            project_snapshot(&snapshot, &invalid, limits),
            Err(ProjectionError::ScaleOutOfRange)
        );

        let mut multi_wrap = projection_request(1);
        multi_wrap.camera.center[0] = 1_000.0;
        let multi_wrap_result =
            project_snapshot(&snapshot, &multi_wrap, limits).expect("multi-wrap camera");
        assert_eq!(
            multi_wrap_result
                .focused_agent
                .expect("focused multi-wrap agent")
                .wrap_offset,
            [900.0, 0.0]
        );

        let mut exact_rank_snapshot = snapshot.clone();
        exact_rank_snapshot.world.agents[0].age = 16_777_216;
        exact_rank_snapshot.world.agents[1].age = 16_777_217;
        let mut exact_rank = projection_request(1);
        exact_rank.ranking = ProjectionRanking::Age;
        exact_rank.top_k = 1;
        assert_eq!(
            project_snapshot(&exact_rank_snapshot, &exact_rank, limits)
                .expect("exact integer ranking")
                .top_agents[0]
                .uid,
            AgentUid(2)
        );

        let mut broker = ProjectionBroker::new(2).expect("bounded broker");
        let mut plus_zero = projection_request(1);
        plus_zero.camera.center[0] = 0.0;
        let first = broker
            .project(&snapshot, &plus_zero, limits)
            .expect("positive-zero request");
        let weak = Arc::downgrade(&first);
        drop(first);
        let mut minus_zero = plus_zero.clone();
        minus_zero.camera.center[0] = -0.0;
        let canonical_hit = broker
            .project(&snapshot, &minus_zero, limits)
            .expect("negative-zero canonical hit");
        assert_eq!(broker.hits(), 1);
        drop(canonical_hit);

        let stricter_limits = ProjectionLimits {
            max_visible_agents: 1,
            ..limits
        };
        assert!(matches!(
            broker.project(&snapshot, &plus_zero, stricter_limits),
            Err(ProjectionError::VisibleAgentsTooLarge { limit: 1 })
        ));
        assert_eq!(
            broker.hits(),
            1,
            "a cached result built under looser bounds must not bypass stricter bounds"
        );

        for client in 2..=3 {
            broker
                .project(&snapshot, &projection_request(client), limits)
                .expect("bounded client projection");
        }
        assert_eq!(broker.len(), 2);
        assert_eq!(broker.evictions(), 1);
        assert!(
            weak.upgrade().is_none(),
            "evicted projection must be reclaimable"
        );
        assert!(broker.retained_output_capacity_bytes() > 0);
        assert!(broker.retained_output_capacity_bytes() <= broker.byte_capacity());

        let mut byte_bounded =
            ProjectionBroker::with_byte_capacity(2, 1).expect("one-byte cache budget");
        let uncached = byte_bounded
            .project(&snapshot, &projection_request(77), limits)
            .expect("oversized result remains usable without cache retention");
        assert!(uncached.build.output_capacity_bytes > byte_bounded.byte_capacity());
        assert!(byte_bounded.is_empty());
        assert_eq!(byte_bounded.uncached_oversize(), 1);
        assert_eq!(byte_bounded.retained_output_capacity_bytes(), 0);

        let mut truncated = projection_request(9);
        truncated.chart_window = 100;
        truncated.chart_points = 3;
        let result = project_snapshot(&snapshot, &truncated, limits).expect("truncated chart");
        assert!(result.chart_truncated);
        assert_eq!(result.chart.len(), 3);
        assert_eq!(result.chart.first().expect("chart first").tick, Tick(1));
        assert_eq!(result.chart.last().expect("chart last").tick, Tick(10));
    }

    #[derive(Debug)]
    struct TestDurableEventReader {
        session_id: HostSessionId,
        events: Mutex<Vec<JournaledScientificEvent>>,
    }

    impl TestDurableEventReader {
        fn push(&self, event: JournaledScientificEvent) {
            self.events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(event);
        }

        fn encoded(&self) -> Vec<u8> {
            serde_json::to_vec(
                &*self
                    .events
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )
            .expect("serialize durable event fixture")
        }

        fn reopen(session_id: HostSessionId, encoded: &[u8]) -> Self {
            Self {
                session_id,
                events: Mutex::new(
                    serde_json::from_slice(encoded).expect("reopen durable event fixture"),
                ),
            }
        }
    }

    impl EventJournalReader for TestDurableEventReader {
        fn session_id(&self) -> HostSessionId {
            self.session_id
        }

        fn guarantee(&self) -> EventCatchUpGuarantee {
            EventCatchUpGuarantee::CrashDurable
        }

        fn available_range(&self) -> Option<EventSequenceRange> {
            let events = self
                .events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            Some(EventSequenceRange {
                first: events.first()?.event.sequence,
                last: events.last()?.event.sequence,
            })
        }

        fn retention_snapshot(&self) -> Option<EventRetentionSnapshot> {
            let events = Arc::new(
                self.events
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clone(),
            );
            let range = EventSequenceRange {
                first: events.first()?.event.sequence,
                last: events.last()?.event.sequence,
            };
            EventRetentionSnapshot::try_new(
                self.session_id,
                EventCatchUpGuarantee::CrashDurable,
                range,
                events,
            )
            .ok()
        }

        fn contains_event_identity(
            &self,
            sequence: EventSequence,
            batch_id: JournalBatchId,
        ) -> bool {
            self.events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .iter()
                .any(|entry| entry.event.sequence == sequence && entry.event.batch_id == batch_id)
        }

        fn read(
            &self,
            locator: EventCatchUpLocator,
            limit: usize,
        ) -> Result<EventCatchUp, HostAccessError> {
            if locator.session_id != self.session_id {
                return Ok(EventCatchUp::Unavailable {
                    range: locator.range,
                    reason: EventCatchUpUnavailableReason::SessionMismatch,
                });
            }
            let events = self
                .events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(available) =
                events
                    .first()
                    .zip(events.last())
                    .map(|(first, last)| EventSequenceRange {
                        first: first.event.sequence,
                        last: last.event.sequence,
                    })
            else {
                return Ok(EventCatchUp::Unavailable {
                    range: locator.range,
                    reason: EventCatchUpUnavailableReason::RangeExpired,
                });
            };
            if !available.contains_range(locator.range) {
                return Ok(EventCatchUp::Unavailable {
                    range: locator.range,
                    reason: EventCatchUpUnavailableReason::RangeExpired,
                });
            }
            Ok(EventCatchUp::Contiguous(EventPage {
                session_id: self.session_id,
                source: EventPageSource::Durable,
                events: events
                    .iter()
                    .filter(|entry| locator.range.contains(entry.event.sequence))
                    .take(limit)
                    .cloned()
                    .collect(),
                latest: available.last,
            }))
        }
    }

    #[derive(Debug)]
    struct UnretainedEventReader {
        session_id: HostSessionId,
        range: EventSequenceRange,
    }

    impl EventJournalReader for UnretainedEventReader {
        fn session_id(&self) -> HostSessionId {
            self.session_id
        }

        fn guarantee(&self) -> EventCatchUpGuarantee {
            EventCatchUpGuarantee::LiveMemory
        }

        fn available_range(&self) -> Option<EventSequenceRange> {
            Some(self.range)
        }

        fn retention_snapshot(&self) -> Option<EventRetentionSnapshot> {
            None
        }

        fn contains_event_identity(
            &self,
            _sequence: EventSequence,
            _batch_id: JournalBatchId,
        ) -> bool {
            false
        }

        fn read(
            &self,
            locator: EventCatchUpLocator,
            _limit: usize,
        ) -> Result<EventCatchUp, HostAccessError> {
            Ok(EventCatchUp::Unavailable {
                range: locator.range,
                reason: EventCatchUpUnavailableReason::RangeExpired,
            })
        }
    }

    fn scientific_boundary(tick: u64) -> Arc<ScientificBoundary> {
        Arc::new(ScientificBoundary::new(
            TickEvents {
                tick: Tick(tick),
                charts_flushed: false,
                epoch_rolled: false,
                food_respawned: None,
            },
            projection_summary(tick),
            Vec::new(),
            Vec::new(),
            TickCombatSummary::default(),
            0,
            None,
        ))
    }

    fn charged_lifecycle_records(dynamic_text: &str) -> (BirthRecord, DeathRecord) {
        (
            BirthRecord {
                tick: Tick(1),
                agent_uid: AgentUid(1),
                spawn_ordinal: 1,
                birth_ordinal: None,
                origin: BirthOrigin::Seeded,
                parent_a: None,
                parent_b: None,
                brain_kind: Some(dynamic_text.to_owned()),
                brain_key: Some(7),
                herbivore_tendency: 0.5,
                generation: Generation(0),
                position: Position::default(),
                is_hybrid: false,
            },
            DeathRecord {
                tick: Tick(1),
                agent_uid: AgentUid(2),
                age: 1,
                generation: Generation(0),
                herbivore_tendency: 0.5,
                brain_kind: Some(dynamic_text.to_owned()),
                brain_key: Some(7),
                energy: 0.0,
                food_balance_total: 0.0,
                cause: DeathCause::Unknown,
                was_hybrid: false,
                combat_flags: CombatEventFlags::default(),
            },
        )
    }

    fn charged_agent(dynamic_text: &str, genome_bytes: usize) -> AgentState {
        let genome = BrainGenomeEnvelope::new(
            BrainFamilyId::new("charge-fixture").expect("valid fixture family"),
            1,
            1,
            vec![7; genome_bytes],
            BrainProvenance::default(),
        )
        .expect("bounded fixture genome");
        let mut runtime = AgentRuntime {
            brain: BrainBinding::Protocol {
                registry_key: 7,
                kind: dynamic_text.to_owned(),
                genome,
                evaluator: None,
            },
            ..AgentRuntime::default()
        };
        runtime.mutation_log.push(dynamic_text.to_owned());
        AgentState {
            id: AgentId::default(),
            identity: AgentIdentity {
                uid: AgentUid(1),
                spawn_ordinal: 1,
                birth_ordinal: None,
            },
            data: AgentData::default(),
            runtime,
        }
    }

    fn charged_persistence(
        dynamic_text: &str,
        replay_outputs: usize,
        genome_bytes: usize,
        birth: &BirthRecord,
        death: &DeathRecord,
    ) -> Arc<PersistenceBatch> {
        Arc::new(PersistenceBatch {
            summary: projection_summary(1),
            epoch: 0,
            closed: false,
            metrics: vec![MetricSample::new(dynamic_text.to_owned(), 1.0)],
            events: vec![PersistenceEvent::new(
                PersistenceEventKind::Custom(dynamic_text.to_owned().into()),
                1,
            )],
            agents: vec![charged_agent(dynamic_text, genome_bytes)],
            births: vec![birth.clone()],
            deaths: vec![death.clone()],
            replay_events: vec![ReplayEvent {
                agent_uid: Some(AgentUid(1)),
                kind: ReplayEventKind::BrainOutputs {
                    outputs: vec![0.0; replay_outputs],
                },
            }],
        })
    }

    fn charged_scientific(
        dynamic_text: &str,
        birth: BirthRecord,
        death: DeathRecord,
    ) -> Arc<ScientificBoundary> {
        Arc::new(
            ScientificBoundary::new(
                TickEvents {
                    tick: Tick(1),
                    charts_flushed: false,
                    epoch_rolled: false,
                    food_respawned: None,
                },
                projection_summary(1),
                vec![birth],
                vec![death],
                TickCombatSummary::default(),
                0,
                None,
            )
            .with_fault(ScientificBoundaryFault::new(dynamic_text, dynamic_text)),
        )
    }

    fn charged_config_command(config_layers: usize) -> CommandEnvelope {
        let mut config = ScriptBotsConfig::default();
        config.neuroflow.hidden_layers = vec![8; config_layers];
        CommandEnvelope::new(
            CommandId::new(1),
            HostCommand::UpdateConfig(Box::new(config)),
        )
    }

    fn charged_journal_batch(
        text_bytes: usize,
        replay_outputs: usize,
        genome_bytes: usize,
        config_layers: usize,
    ) -> Arc<JournalBatch> {
        let dynamic_text = "x".repeat(text_bytes);
        let (birth, death) = charged_lifecycle_records(&dynamic_text);
        let persistence =
            charged_persistence(&dynamic_text, replay_outputs, genome_bytes, &birth, &death);
        let scientific = charged_scientific(&dynamic_text, birth, death);
        let applied = AppliedCommand {
            tick: Tick(1),
            revisions: HostRevisions::default(),
        };
        let lifecycle = CommandLifecycleEvidence::from_terminal(
            charged_config_command(config_layers),
            Some(AdmissionSequence::new(1)),
            AppliedCommand {
                tick: Tick(0),
                revisions: HostRevisions::default(),
            },
            applied,
            ApplicationState::Applied(applied),
        )
        .expect("charged command lifecycle");
        Arc::new(JournalBatch::new(
            JournalBatchId::new(HostSessionId::new(70), 1),
            Some(EventSequence::new(1)),
            Some(lifecycle),
            applied,
            Some(scientific),
            Some(persistence),
        ))
    }

    #[test]
    fn journal_admission_charge_tracks_dynamic_strings_genomes_and_nested_outputs() {
        let baseline = charged_journal_batch(4, 4, 4, 1).retained_bytes();
        assert!(
            charged_journal_batch(128, 4, 4, 1).retained_bytes() > baseline,
            "longer owned strings must increase the precomputed charge"
        );
        assert!(
            charged_journal_batch(4, 128, 4, 1).retained_bytes() > baseline,
            "nested replay outputs must increase the precomputed charge"
        );
        assert!(
            charged_journal_batch(4, 4, 256, 1).retained_bytes() > baseline,
            "opaque genome bytes must increase the precomputed charge"
        );
        assert!(
            charged_journal_batch(4, 4, 4, 32).retained_bytes() > baseline,
            "UpdateConfig's boxed dynamic layers must increase the precomputed charge"
        );
    }

    #[derive(Default)]
    struct RejectOnceChargeJournal {
        first: Option<(*const JournalBatch, usize)>,
    }

    impl JournalPort for RejectOnceChargeJournal {
        fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
            if let Some((first_ptr, first_charge)) = self.first {
                assert!(
                    std::ptr::eq(Arc::as_ptr(batch), first_ptr),
                    "retry must reuse the exact batch allocation"
                );
                assert_eq!(
                    batch.retained_bytes(),
                    first_charge,
                    "retry must reuse the precomputed charge"
                );
                JournalAdmission::Accepted {
                    batch_id: batch.id(),
                }
            } else {
                self.first = Some((Arc::as_ptr(batch), batch.retained_bytes()));
                JournalAdmission::Full {
                    batch_id: batch.id(),
                    capacity: 1,
                }
            }
        }

        fn poll_receipts(&mut self, _limit: usize) -> Vec<JournalReceipt> {
            Vec::new()
        }
    }

    #[test]
    fn retained_retry_preserves_the_precomputed_charge_and_exact_arc() {
        let batch = charged_journal_batch(32, 16, 64, 3);
        let retained = Arc::clone(&batch);
        let charge = batch.retained_bytes();
        let mut journal = RejectOnceChargeJournal::default();

        assert!(matches!(
            journal.try_admit(&batch),
            JournalAdmission::Full { .. }
        ));
        assert!(Arc::ptr_eq(&batch, &retained));
        assert_eq!(retained.retained_bytes(), charge);
        drop(batch);
        assert!(matches!(
            journal.try_admit(&retained),
            JournalAdmission::Accepted { .. }
        ));
        assert_eq!(retained.retained_bytes(), charge);
    }

    fn publish_durable_fixture_event(
        hub: &EventHub,
        reader: &TestDurableEventReader,
        sequence: u64,
    ) {
        let session_id = reader.session_id;
        assert!(hub.prepare_publish().expect("durable slot").is_none());
        let batch_id = JournalBatchId::new(session_id, sequence);
        let event_sequence = hub
            .publish_pending(
                batch_id,
                AppliedCommand {
                    tick: Tick(sequence),
                    revisions: HostRevisions {
                        control: ControlRevision::new(0),
                        scientific: ScientificRevision::new(sequence),
                        config: ConfigRevision::new(0),
                    },
                },
                scientific_boundary(sequence),
            )
            .expect("durable event publish");
        let page = match hub
            .poll(
                EventCursor::after(session_id, EventSequence::new(sequence - 1)),
                1,
            )
            .expect("published event poll")
        {
            EventPoll::Contiguous(page) => Some(page),
            EventPoll::Gap(_) => None,
        }
        .expect("new event must be hot");
        let mut durable = page.events[0].clone();
        assert_eq!(durable.event.sequence, event_sequence);
        durable.commitment = EventCommitment::Durable;
        reader.push(durable);
        hub.update_commitment(batch_id, event_sequence, EventCommitment::Durable)
            .expect("durable commitment");
    }

    #[test]
    fn eviction_requires_atomic_retention_even_when_a_watermark_claims_coverage() {
        let session_id = HostSessionId::new(47);
        let reader: Arc<dyn EventJournalReader> = Arc::new(UnretainedEventReader {
            session_id,
            range: EventSequenceRange {
                first: EventSequence::new(1),
                last: EventSequence::new(1),
            },
        });
        let hub = EventHub::new(session_id, 1, Some(reader)).expect("retention-test hub");
        assert!(hub.prepare_publish().expect("initial slot").is_none());
        let batch_id = JournalBatchId::new(session_id, 1);
        let sequence = hub
            .publish_pending(
                batch_id,
                AppliedCommand {
                    tick: Tick(1),
                    revisions: HostRevisions {
                        scientific: ScientificRevision::new(1),
                        ..HostRevisions::default()
                    },
                },
                scientific_boundary(1),
            )
            .expect("retention-test event");
        hub.update_commitment(batch_id, sequence, EventCommitment::CommittedVolatile)
            .expect("volatile commitment");

        let pressure = hub
            .prepare_publish()
            .expect("retention refusal is typed")
            .expect("missing atomic retention must pin the front");
        assert_eq!(pressure.batch_id, batch_id);
        assert_eq!(pressure.sequence, sequence);
        assert_eq!(pressure.reason, EventHighWaterReason::RangeUnavailable);
        assert_eq!(hub.len(), 1);
        assert_eq!(hub.published_total(), 1);
    }

    #[test]
    fn catch_up_validation_rejects_wrong_source_boundary_and_bounds() {
        let session_id = HostSessionId::new(48);
        let cursor = EventCursor::beginning(session_id);
        let locator = EventCatchUpLocator {
            session_id,
            range: EventSequenceRange {
                first: EventSequence::new(1),
                last: EventSequence::new(1),
            },
            guarantee: EventCatchUpGuarantee::CrashDurable,
        };
        let valid_entry = JournaledScientificEvent {
            event: Arc::new(ScientificEvent {
                session_id,
                sequence: EventSequence::new(1),
                batch_id: JournalBatchId::new(session_id, 1),
                tick: Tick(1),
                revisions: HostRevisions {
                    scientific: ScientificRevision::new(1),
                    ..HostRevisions::default()
                },
                boundary: scientific_boundary(1),
            }),
            commitment: EventCommitment::Durable,
        };
        let valid = EventCatchUp::Contiguous(EventPage {
            session_id,
            source: EventPageSource::Durable,
            events: vec![valid_entry.clone()],
            latest: EventSequence::new(1),
        });
        assert_eq!(
            validate_event_catch_up(cursor, locator, &valid, 1).expect("valid durable page"),
            EventSequence::new(1)
        );

        let mut wrong_source = valid.clone();
        let EventCatchUp::Contiguous(page) = &mut wrong_source else {
            unreachable!("fixture is contiguous");
        };
        page.source = EventPageSource::LiveMemory;
        assert!(validate_event_catch_up(cursor, locator, &wrong_source, 1).is_err());

        let mut wrong_boundary = valid;
        let EventCatchUp::Contiguous(page) = &mut wrong_boundary else {
            unreachable!("fixture is contiguous");
        };
        page.events[0] = JournaledScientificEvent {
            event: Arc::new(ScientificEvent {
                boundary: scientific_boundary(2),
                ..valid_entry.event.as_ref().clone()
            }),
            commitment: EventCommitment::Durable,
        };
        assert!(validate_event_catch_up(cursor, locator, &wrong_boundary, 1).is_err());
        assert!(validate_event_catch_up(cursor, locator, &wrong_boundary, 0).is_err());
        assert_eq!(cursor.last_seen(), EventSequence::new(0));
    }

    #[test]
    #[ignore = "DSR-only reference-hardware scientific-event measurement"]
    #[allow(
        clippy::too_many_lines,
        reason = "the ignored evidence oracle emits raw latency and structural-memory samples together"
    )]
    fn dsr_measure_scientific_event_ring_latency_memory_and_consumer_scaling() {
        const WARMUPS: usize = 20;
        const SAMPLES: usize = 200;
        const CAPACITY: usize = 128;
        const CONSUMERS: usize = 10_000;
        const PUBLISH_P95_BUDGET_NS: u64 = 5_000_000;
        const POLL_P95_BUDGET_NS: u64 = 500_000;
        const FANOUT_BUDGET_NS: u64 = 500_000_000;

        let session_id = HostSessionId::new(49);
        let hub = EventHub::new(session_id, CAPACITY, None).expect("measurement event hub");
        let mut publish_samples = Vec::with_capacity(SAMPLES);
        for sample in 0..WARMUPS + SAMPLES {
            let sequence = u64::try_from(sample).expect("measurement sample fits u64") + 1;
            assert!(
                hub.prepare_publish()
                    .expect("measurement event slot")
                    .is_none()
            );
            let batch_id = JournalBatchId::new(session_id, sequence);
            let started = Instant::now();
            let event_sequence = hub
                .publish_pending(
                    batch_id,
                    AppliedCommand {
                        tick: Tick(sequence),
                        revisions: HostRevisions {
                            scientific: ScientificRevision::new(sequence),
                            ..HostRevisions::default()
                        },
                    },
                    scientific_boundary(sequence),
                )
                .expect("measurement event publish");
            hub.update_commitment(batch_id, event_sequence, EventCommitment::Durable)
                .expect("measurement durable commitment");
            let elapsed = u64::try_from(started.elapsed().as_nanos())
                .expect("measurement duration fits u64 nanoseconds");
            if sample >= WARMUPS {
                publish_samples.push(elapsed);
            }
        }
        assert_eq!(hub.len(), CAPACITY);
        assert_eq!(hub.pending_count(), 0);

        let held_old_view = hub.hot.load_full();
        let held_total = held_old_view.published_total;
        let held_first = held_old_view
            .entries
            .front()
            .expect("full held event view")
            .event
            .sequence;
        let next = u64::try_from(WARMUPS + SAMPLES).expect("sample total fits u64") + 1;
        assert!(
            hub.prepare_publish()
                .expect("post-hold event slot")
                .is_none()
        );
        let next_batch = JournalBatchId::new(session_id, next);
        let next_sequence = hub
            .publish_pending(
                next_batch,
                AppliedCommand {
                    tick: Tick(next),
                    revisions: HostRevisions {
                        scientific: ScientificRevision::new(next),
                        ..HostRevisions::default()
                    },
                },
                scientific_boundary(next),
            )
            .expect("post-hold event publish");
        hub.update_commitment(next_batch, next_sequence, EventCommitment::Durable)
            .expect("post-hold durable commitment");
        assert_eq!(held_old_view.published_total, held_total);
        assert_eq!(
            held_old_view
                .entries
                .front()
                .expect("held view remains populated")
                .event
                .sequence,
            held_first
        );
        assert_eq!(hub.published_total(), held_total + 1);

        let tip_cursor = EventCursor::after(session_id, next_sequence);
        let mut poll_samples = Vec::with_capacity(SAMPLES);
        for sample in 0..WARMUPS + SAMPLES {
            let started = Instant::now();
            let page = black_box(hub.poll(tip_cursor, 1).expect("measurement hot poll"));
            let elapsed = u64::try_from(started.elapsed().as_nanos())
                .expect("measurement duration fits u64 nanoseconds");
            if sample >= WARMUPS {
                poll_samples.push(elapsed);
            }
            let page = match page {
                EventPoll::Contiguous(page) => Some(page),
                EventPoll::Gap(_) => None,
            }
            .expect("tip cursor must remain contiguous");
            assert!(page.events.is_empty());
        }

        let cursors = vec![tip_cursor; CONSUMERS];
        let fanout_started = Instant::now();
        for cursor in &cursors {
            black_box(hub.poll(*cursor, 1).expect("independent consumer poll"));
        }
        let fanout_ns = u64::try_from(fanout_started.elapsed().as_nanos())
            .expect("fanout duration fits u64 nanoseconds");
        let cursor_capacity_bytes = cursors.capacity().saturating_mul(size_of::<EventCursor>());
        let hot = hub.hot.load();
        let hot_entry_capacity_bytes = hot
            .entries
            .capacity()
            .saturating_mul(size_of::<JournaledScientificEvent>());
        assert!(hub.len() <= CAPACITY);
        assert!(hot.entries.capacity() <= CAPACITY);
        assert_eq!(cursor_capacity_bytes, CONSUMERS * size_of::<EventCursor>());

        let percentile_95 = |samples: &[u64]| {
            let mut sorted = samples.to_vec();
            sorted.sort_unstable();
            sorted[(sorted.len() * 95).div_ceil(100).saturating_sub(1)]
        };
        let publish_p95_ns = percentile_95(&publish_samples);
        let poll_p95_ns = percentile_95(&poll_samples);
        let evidence = serde_json::json!({
            "schema": "scriptbots.scientific-event.measurement.v1",
            "hot_capacity": CAPACITY,
            "warmups_per_case": WARMUPS,
            "samples_per_case": SAMPLES,
            "consumer_count": CONSUMERS,
            "publish_raw_ns": publish_samples,
            "publish_p95_ns": publish_p95_ns,
            "publish_p95_budget_ns": PUBLISH_P95_BUDGET_NS,
            "tip_poll_raw_ns": poll_samples,
            "tip_poll_p95_ns": poll_p95_ns,
            "tip_poll_p95_budget_ns": POLL_P95_BUDGET_NS,
            "fanout_ns": fanout_ns,
            "fanout_budget_ns": FANOUT_BUDGET_NS,
            "hot_entry_capacity_bytes": hot_entry_capacity_bytes,
            "cursor_capacity_bytes": cursor_capacity_bytes,
            "published_total": hub.published_total(),
        });
        eprintln!(
            "{}",
            serde_json::to_string(&evidence).expect("serialize event measurement evidence")
        );
        assert!(publish_p95_ns < PUBLISH_P95_BUDGET_NS);
        assert!(poll_p95_ns < POLL_P95_BUDGET_NS);
        assert!(fanout_ns < FANOUT_BUDGET_NS);
    }

    #[test]
    fn durable_reader_repairs_hot_gap_and_survives_restart() {
        let session_id = HostSessionId::new(45);
        let reader = Arc::new(TestDurableEventReader {
            session_id,
            events: Mutex::new(Vec::new()),
        });
        let reader_capability: Arc<dyn EventJournalReader> = reader.clone();
        let hub = EventHub::new(session_id, 2, Some(reader_capability)).expect("durable hub");
        for sequence in 1..=3 {
            publish_durable_fixture_event(&hub, &reader, sequence);
        }
        hub.update_commitment(
            JournalBatchId::new(session_id, 1),
            EventSequence::new(1),
            EventCommitment::Durable,
        )
        .expect("evicted exact batch identity remains idempotent");
        assert!(
            hub.update_commitment(
                JournalBatchId::new(session_id, 99),
                EventSequence::new(1),
                EventCommitment::Durable,
            )
            .is_err(),
            "reader coverage alone must not authorize a different batch identity"
        );
        let gap = match hub
            .poll(EventCursor::beginning(session_id), usize::MAX)
            .expect("durable gap poll")
        {
            EventPoll::Gap(gap) => Some(gap),
            EventPoll::Contiguous(_) => None,
        }
        .expect("wrapped durable hot ring must report a gap");
        let locator = match gap.catch_up {
            EventCatchUpState::Available(locator) => Some(locator),
            EventCatchUpState::Unavailable(_) => None,
        }
        .expect("durable gap must expose a reader locator");
        assert_eq!(locator.guarantee(), EventCatchUpGuarantee::CrashDurable);
        let caught_up = match hub.catch_up(locator, usize::MAX).expect("durable catch-up") {
            EventCatchUp::Contiguous(page) => Some(page),
            EventCatchUp::Unavailable { .. } => None,
        }
        .expect("durable locator must be readable");
        assert_eq!(caught_up.source, EventPageSource::Durable);
        assert_eq!(caught_up.events.len(), 1);
        assert_eq!(caught_up.events[0].event.sequence, EventSequence::new(1));
        assert_eq!(caught_up.events[0].commitment, EventCommitment::Durable);

        let encoded = reader.encoded();
        let reopened_reader = Arc::new(TestDurableEventReader::reopen(session_id, &encoded));
        let reopened_capability: Arc<dyn EventJournalReader> = reopened_reader;
        let reopened_hub =
            EventHub::new(session_id, 2, Some(reopened_capability)).expect("reopened durable hub");
        let restart_locator = EventCatchUpLocator {
            session_id,
            range: EventSequenceRange {
                first: EventSequence::new(1),
                last: EventSequence::new(3),
            },
            guarantee: EventCatchUpGuarantee::CrashDurable,
        };
        let restarted = match reopened_hub
            .catch_up(restart_locator, usize::MAX)
            .expect("serialized durable restart catch-up")
        {
            EventCatchUp::Contiguous(page) => Some(page),
            EventCatchUp::Unavailable { .. } => None,
        }
        .expect("reopened durable reader must return a contiguous page");
        assert_eq!(restarted.source, EventPageSource::Durable);
        assert_eq!(restarted.events.len(), 3);
        assert_eq!(restarted.latest, EventSequence::new(3));
    }

    #[test]
    fn unavailable_gap_requires_explicit_skip() {
        let unavailable =
            EventHub::new(HostSessionId::new(46), 1, None).expect("unavailable-reader hub");
        for sequence in 1..=2 {
            assert!(
                unavailable
                    .prepare_publish()
                    .expect("durable slot without reader")
                    .is_none()
            );
            let batch_id = JournalBatchId::new(HostSessionId::new(46), sequence);
            unavailable
                .publish_pending(
                    batch_id,
                    AppliedCommand {
                        tick: Tick(sequence),
                        revisions: HostRevisions::default(),
                    },
                    scientific_boundary(sequence),
                )
                .expect("unavailable event publish");
            unavailable
                .update_commitment(
                    batch_id,
                    EventSequence::new(sequence),
                    EventCommitment::Durable,
                )
                .expect("unavailable event durable state");
        }
        let cursor = EventCursor::beginning(HostSessionId::new(46));
        let unavailable_gap = match unavailable
            .poll(cursor, usize::MAX)
            .expect("unavailable gap poll")
        {
            EventPoll::Gap(gap) => Some(gap),
            EventPoll::Contiguous(_) => None,
        }
        .expect("wrapped no-reader hub must report gap");
        assert_eq!(
            unavailable_gap.catch_up,
            EventCatchUpState::Unavailable(EventCatchUpUnavailableReason::NoReader)
        );
        assert_eq!(cursor.last_seen(), EventSequence::new(0));
        assert_eq!(unavailable_gap.resume_after(), EventSequence::new(1));
    }

    #[derive(Clone)]
    struct SharedFakeHost {
        inner: Arc<Mutex<FakeHost>>,
    }

    struct SharedFakeDriver {
        inner: Arc<Mutex<FakeHost>>,
    }

    struct LyingFakeDriver {
        session_id: HostSessionId,
    }

    impl SharedFakeHost {
        fn new() -> Self {
            let session_id =
                HostSessionId::new(NEXT_FAKE_SESSION_ID.fetch_add(1, Ordering::Relaxed));
            Self {
                inner: Arc::new(Mutex::new(FakeHost::new(session_id))),
            }
        }

        fn lock(&self) -> MutexGuard<'_, FakeHost> {
            self.inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }

        fn fail_on_application(&self, command_id: CommandId) {
            self.lock().fail_on_application.insert(command_id);
        }

        fn lose_next_submission_receipt(&self, command_id: CommandId) {
            self.lock().lost_submission_receipts.insert(command_id);
        }

        fn driver(&self) -> SharedFakeDriver {
            SharedFakeDriver {
                inner: Arc::clone(&self.inner),
            }
        }
    }

    impl HostPort for SharedFakeHost {
        fn session_id(&self) -> HostSessionId {
            self.lock().session_id
        }

        fn submit(&mut self, envelope: CommandEnvelope) -> Result<CommandStatus, HostAccessError> {
            let command_id = envelope.command_id;
            let mut host = self.lock();
            let status = host.submit(envelope)?;
            if host.lost_submission_receipts.remove(&command_id) {
                Err(HostAccessError::Disconnected)
            } else {
                Ok(status)
            }
        }

        fn command_status(
            &mut self,
            command_id: CommandId,
        ) -> Result<Option<CommandStatus>, HostAccessError> {
            Ok(self.lock().statuses.get(&command_id).cloned())
        }

        fn snapshot_after(
            &mut self,
            after: Option<SnapshotRevision>,
        ) -> Result<Option<Arc<RenderSnapshot>>, HostAccessError> {
            let host = self.lock();
            Ok(host.latest_snapshot.as_ref().and_then(|snapshot| {
                after
                    .is_none_or(|revision| snapshot.revision > revision)
                    .then(|| Arc::clone(snapshot))
            }))
        }

        fn events_after(
            &mut self,
            cursor: ProtocolEventSequence,
            limit: usize,
        ) -> Result<Vec<HostEvent>, HostAccessError> {
            Ok(self
                .lock()
                .events
                .iter()
                .filter(|event| event.sequence > cursor)
                .take(limit)
                .cloned()
                .collect())
        }

        fn poll_events(
            &mut self,
            cursor: EventCursor,
            _limit: usize,
        ) -> Result<EventPoll, HostAccessError> {
            let session_id = self.lock().session_id;
            if cursor.session_id != session_id {
                return Err(HostAccessError::EventSessionMismatch {
                    expected: cursor.session_id,
                    actual: session_id,
                });
            }
            Ok(EventPoll::Contiguous(EventPage {
                session_id,
                source: EventPageSource::Hot,
                events: Vec::new(),
                latest: EventSequence::new(0),
            }))
        }

        fn catch_up_events(
            &mut self,
            locator: EventCatchUpLocator,
            _limit: usize,
        ) -> Result<EventCatchUp, HostAccessError> {
            Ok(EventCatchUp::Unavailable {
                range: locator.range,
                reason: EventCatchUpUnavailableReason::NoReader,
            })
        }
    }

    impl ManualHostDriver for SharedFakeDriver {
        fn session_id(&self) -> HostSessionId {
            self.inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .session_id
        }

        fn drive(&mut self, now: ManualInstant) -> Result<DriveReceipt, HostAccessError> {
            self.inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .drive(now)
        }
    }

    impl ManualHostDriver for LyingFakeDriver {
        fn session_id(&self) -> HostSessionId {
            self.session_id
        }

        fn drive(&mut self, now: ManualInstant) -> Result<DriveReceipt, HostAccessError> {
            Ok(DriveReceipt {
                now: ManualInstant::from_nanos(
                    now.as_nanos()
                        .checked_add(1)
                        .expect("lying test time has headroom"),
                ),
                ..DriveReceipt::default()
            })
        }
    }

    struct FakeHost {
        session_id: HostSessionId,
        now: ManualInstant,
        next_admission: AdmissionSequence,
        next_event: ProtocolEventSequence,
        next_snapshot: SnapshotRevision,
        revisions: HostRevisions,
        tick: Tick,
        playback: PlaybackSnapshot,
        lifecycle: HostLifecycle,
        config: ScriptBotsConfig,
        queue: VecDeque<CommandEnvelope>,
        statuses: HashMap<CommandId, CommandStatus>,
        admission_order: Vec<CommandId>,
        latest_snapshot: Option<Arc<RenderSnapshot>>,
        events: Vec<HostEvent>,
        fail_on_application: HashSet<CommandId>,
        lost_submission_receipts: HashSet<CommandId>,
    }

    impl FakeHost {
        fn new(session_id: HostSessionId) -> Self {
            let config = ScriptBotsConfig::default();
            let mut host = Self {
                session_id,
                now: ManualInstant::default(),
                next_admission: AdmissionSequence::new(1),
                next_event: ProtocolEventSequence::new(1),
                next_snapshot: SnapshotRevision::new(1),
                revisions: HostRevisions::default(),
                tick: Tick(0),
                playback: PlaybackSnapshot::default(),
                lifecycle: HostLifecycle::Running,
                config,
                queue: VecDeque::new(),
                statuses: HashMap::new(),
                admission_order: Vec::new(),
                latest_snapshot: None,
                events: Vec::new(),
                fail_on_application: HashSet::new(),
                lost_submission_receipts: HashSet::new(),
            };
            host.publish_snapshot();
            host.events.clear();
            host.next_event = ProtocolEventSequence::new(1);
            host
        }

        fn submit(&mut self, envelope: CommandEnvelope) -> Result<CommandStatus, HostAccessError> {
            if let Some(status) = self.statuses.get(&envelope.command_id) {
                return Ok(status.clone());
            }

            let rejection = if self.lifecycle != HostLifecycle::Running {
                Some(RejectionReason::HostStopping)
            } else if let Err(error) = envelope.command.validate() {
                Some(RejectionReason::Validation {
                    message: error.to_string(),
                })
            } else {
                None
            };

            if let Some(reason) = rejection {
                let status = CommandStatus::try_new(
                    envelope.command_id,
                    None,
                    ApplicationState::Rejected(reason),
                    JournalState::Pending,
                )
                .map_err(|error| protocol_violation(error.to_string()))?;
                self.statuses.insert(envelope.command_id, status.clone());
                self.emit_status(status.clone());
                return Ok(status);
            }

            let admission = self.next_admission;
            self.next_admission = admission
                .checked_next()
                .ok_or_else(|| protocol_violation("admission sequence exhausted"))?;
            let status = CommandStatus::try_new(
                envelope.command_id,
                Some(admission),
                ApplicationState::Admitted,
                JournalState::Pending,
            )
            .map_err(|error| protocol_violation(error.to_string()))?;
            self.admission_order.push(envelope.command_id);
            self.queue.push_back(envelope);
            self.statuses.insert(status.command_id(), status.clone());
            self.emit_status(status.clone());
            Ok(status)
        }

        fn drive(&mut self, now: ManualInstant) -> Result<DriveReceipt, HostAccessError> {
            if now < self.now {
                return Err(protocol_violation("manual time moved backwards"));
            }
            self.now = now;
            let events_before = self.events.len();
            let scientific_before = self.revisions.scientific;
            let mut commands_completed = 0;
            while let Some(envelope) = self.queue.pop_front() {
                self.apply(envelope)?;
                commands_completed += 1;
            }
            let snapshots_published = usize::from(commands_completed != 0);
            if snapshots_published != 0 {
                self.publish_snapshot();
            }
            Ok(DriveReceipt {
                now,
                commands_completed,
                scientific_steps: usize::try_from(
                    self.revisions
                        .scientific
                        .get()
                        .saturating_sub(scientific_before.get()),
                )
                .expect("test scientific-step count fits usize"),
                automatic_steps_due: 0,
                automatic_steps_skipped: 0,
                scientific_revision: self.revisions.scientific,
                snapshots_published,
                events_published: self.events.len() - events_before,
                blocker: None,
            })
        }

        fn apply(&mut self, envelope: CommandEnvelope) -> Result<(), HostAccessError> {
            let admission = self
                .statuses
                .get(&envelope.command_id)
                .and_then(CommandStatus::admission_sequence)
                .ok_or_else(|| protocol_violation("queued command was not admitted"))?;

            let application = if let Some(expected) = envelope.expected_control_revision
                && expected != self.revisions.control
            {
                ApplicationState::Rejected(RejectionReason::ControlRevisionConflict {
                    expected,
                    actual: self.revisions.control,
                })
            } else if self.fail_on_application.remove(&envelope.command_id) {
                ApplicationState::Failed(ApplicationFailure {
                    code: "injected_conformance_failure".to_owned(),
                    message: "test host refused application".to_owned(),
                })
            } else {
                self.revisions.control = self
                    .revisions
                    .control
                    .checked_next()
                    .ok_or_else(|| protocol_violation("control revision exhausted"))?;
                match envelope.command {
                    HostCommand::Pause => self.playback.paused = true,
                    HostCommand::Resume => self.playback.paused = false,
                    HostCommand::SetSpeed(speed) => self.playback.speed_multiplier = speed,
                    HostCommand::Step => {
                        self.playback.paused = true;
                        self.tick.0 = self
                            .tick
                            .0
                            .checked_add(1)
                            .ok_or_else(|| protocol_violation("tick exhausted"))?;
                        self.revisions.scientific =
                            self.revisions.scientific.checked_next().ok_or_else(|| {
                                protocol_violation("scientific revision exhausted")
                            })?;
                    }
                    HostCommand::UpdateConfig(config) => {
                        self.config = *config;
                        self.revisions.config = self
                            .revisions
                            .config
                            .checked_next()
                            .ok_or_else(|| protocol_violation("config revision exhausted"))?;
                    }
                    HostCommand::Shutdown => self.lifecycle = HostLifecycle::Stopped,
                }
                ApplicationState::Applied(AppliedCommand {
                    tick: self.tick,
                    revisions: self.revisions,
                })
            };

            let status = CommandStatus::try_new(
                envelope.command_id,
                Some(admission),
                application,
                JournalState::Durable,
            )
            .map_err(|error| protocol_violation(error.to_string()))?;
            self.statuses.insert(envelope.command_id, status.clone());
            self.emit_status(status);
            if self.lifecycle == HostLifecycle::Stopped {
                self.emit(HostEventKind::LifecycleChanged(HostLifecycle::Stopped));
            }
            Ok(())
        }

        #[allow(
            clippy::too_many_lines,
            reason = "the fake protocol oracle keeps one complete coherent RenderSnapshot literal visible"
        )]
        fn publish_snapshot(&mut self) {
            let revision = self.next_snapshot;
            self.next_snapshot = revision
                .checked_next()
                .expect("test snapshot sequence must have headroom");
            let food_width = self.config.world_width / self.config.food_cell_size;
            let food_height = self.config.world_height / self.config.food_cell_size;
            let layers = self
                .latest_snapshot
                .as_ref()
                .map(|snapshot| snapshot.layers.clone())
                .filter(|layers| {
                    layers.terrain.width == food_width
                        && layers.terrain.height == food_height
                        && layers.terrain.cell_size == self.config.food_cell_size
                        && layers.food.width == food_width
                        && layers.food.height == food_height
                })
                .unwrap_or_else(|| {
                    let cell_count =
                        usize::try_from(u64::from(food_width) * u64::from(food_height))
                            .expect("fake snapshot dimensions fit usize");
                    let prior = self
                        .latest_snapshot
                        .as_ref()
                        .map(|snapshot| snapshot.layers.revisions)
                        .unwrap_or_default();
                    SnapshotLayers {
                        revisions: SnapshotLayerRevisions {
                            terrain: prior
                                .terrain
                                .checked_next()
                                .expect("fake terrain revision has headroom"),
                            food: prior
                                .food
                                .checked_next()
                                .expect("fake food revision has headroom"),
                            hydrology: prior.hydrology,
                        },
                        terrain: Arc::new(TerrainLayerSnapshot {
                            width: food_width,
                            height: food_height,
                            cell_size: self.config.food_cell_size,
                            tiles: vec![
                                TerrainTileSnapshot {
                                    kind: TerrainKind::Grass,
                                    elevation: 0.5,
                                    moisture: 0.5,
                                    accent: 0.0,
                                    fertility_bias: 0.5,
                                    temperature_bias: 0.5,
                                    palette_index: 3,
                                };
                                cell_count
                            ],
                        }),
                        food: Arc::new(FoodLayerSnapshot {
                            width: food_width,
                            height: food_height,
                            cells: vec![0.0; cell_count],
                        }),
                        hydrology: None,
                    }
                });
            let last_applied_command = self
                .admission_order
                .iter()
                .rev()
                .find(|command_id| {
                    self.statuses.get(command_id).is_some_and(|status| {
                        matches!(status.application(), ApplicationState::Applied(_))
                    })
                })
                .copied();
            let completed_summary = (self.tick != Tick::zero()).then_some(TickSummary {
                tick: self.tick,
                agent_count: 0,
                births: 0,
                deaths: 0,
                total_energy: 0.0,
                average_energy: 0.0,
                average_health: 0.0,
                max_age: 0,
                spike_hits: 0,
            });
            self.latest_snapshot = Some(Arc::new(RenderSnapshot {
                session_id: self.session_id,
                revision,
                revisions: self.revisions,
                playback: self.playback,
                lifecycle: self.lifecycle,
                health: HostHealth::Healthy,
                command_queue_depth: self.queue.len(),
                last_applied_command,
                completed_summary: completed_summary.clone(),
                summary_history: Arc::new(completed_summary.into_iter().collect()),
                layers,
                build: SnapshotBuildStats::default(),
                world: DynamicWorldSnapshot {
                    tick: self.tick.0,
                    epoch: 0,
                    world: DynamicSnapshotWorld {
                        width: self.config.world_width,
                        height: self.config.world_height,
                        closed: self.config.closed,
                    },
                    summary: DynamicSnapshotSummary {
                        agent_count: 0,
                        births: 0,
                        deaths: 0,
                        total_energy: 0.0,
                        average_energy: 0.0,
                        average_health: 0.0,
                    },
                    agents: Vec::new(),
                },
            }));
        }

        fn emit_status(&mut self, status: CommandStatus) {
            self.emit(HostEventKind::CommandStatusChanged(status));
        }

        fn emit(&mut self, kind: HostEventKind) {
            let sequence = self.next_event;
            self.next_event = sequence
                .checked_next()
                .expect("test event sequence must have headroom");
            self.events.push(HostEvent {
                session_id: self.session_id,
                sequence,
                tick: self.tick,
                kind,
            });
        }
    }

    const fn envelope(id: u128, command: HostCommand) -> CommandEnvelope {
        CommandEnvelope::new(CommandId::new(id), command)
    }

    fn submit_ok(
        client: &mut HostClient<SharedFakeHost>,
        envelope: CommandEnvelope,
    ) -> CommandStatus {
        client
            .submit(envelope)
            .expect("the conformance host should accept this request")
    }

    fn applied(status: &CommandStatus) -> Result<AppliedCommand, String> {
        match status.application() {
            ApplicationState::Applied(applied) => Ok(*applied),
            state => Err(format!("expected applied status, got {state:?}")),
        }
    }

    #[test]
    fn command_ids_use_fixed_width_json_strings() {
        for command_id in [
            CommandId::new(0),
            CommandId::from_client_sequence(u64::MAX, u64::MAX),
            CommandId::new(u128::MAX),
        ] {
            let encoded = serde_json::to_string(&command_id).expect("command id should encode");
            assert_eq!(encoded.len(), 34, "quotes plus 32 hexadecimal digits");
            assert!(encoded.starts_with('"') && encoded.ends_with('"'));
            let decoded: CommandId =
                serde_json::from_str(&encoded).expect("command id should round trip");
            assert_eq!(decoded, command_id);
        }
        assert_eq!(
            serde_json::to_string(&CommandId::new(u128::MAX)).expect("maximum id encodes"),
            "\"ffffffffffffffffffffffffffffffff\""
        );
        assert!(serde_json::from_str::<CommandId>("1").is_err());
        assert!(serde_json::from_str::<CommandId>("\"abc\"").is_err());
        assert!(serde_json::from_str::<CommandId>("\"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\"").is_err());
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the lifecycle wire test covers every independent causal boundary axis"
    )]
    fn command_lifecycle_evidence_round_trips_and_rejects_malformed_wire_state() {
        let command_id = CommandId::from_client_sequence(0xfeed_beef, 42);
        assert_eq!(command_id.client_namespace(), 0xfeed_beef);
        assert_eq!(command_id.client_sequence(), 42);
        let initial = AppliedCommand {
            tick: Tick(3),
            revisions: HostRevisions {
                control: ControlRevision::new(5),
                scientific: ScientificRevision::new(3),
                config: ConfigRevision::new(2),
            },
        };
        let terminal = AppliedCommand {
            tick: Tick(4),
            revisions: HostRevisions {
                control: ControlRevision::new(6),
                scientific: ScientificRevision::new(4),
                config: ConfigRevision::new(2),
            },
        };
        let envelope = CommandEnvelope::new(command_id, HostCommand::Step)
            .expecting_control_revision(ControlRevision::new(5))
            .expecting_scientific_revision(ScientificRevision::new(3))
            .expecting_config_revision(ConfigRevision::new(2));
        let evidence = CommandLifecycleEvidence::try_new(
            envelope.clone(),
            Some(AdmissionSequence::new(9)),
            vec![
                CommandLifecycleTransition::new(0, initial, ApplicationState::Admitted),
                CommandLifecycleTransition::new(1, terminal, ApplicationState::Applied(terminal)),
            ],
        )
        .expect("valid command lifecycle");

        assert_eq!(evidence.schema_version(), COMMAND_LIFECYCLE_SCHEMA_VERSION);
        assert_eq!(evidence.source_client_namespace(), 0xfeed_beef);
        assert_eq!(evidence.envelope(), &envelope);
        assert_eq!(
            evidence.admission_sequence(),
            Some(AdmissionSequence::new(9))
        );
        assert_eq!(evidence.transitions()[0].boundary(), initial);
        assert_eq!(evidence.transitions()[1].ordinal(), 1);
        assert!(evidence.requires_runtime_journal());

        let encoded = serde_json::to_vec(&evidence).expect("serialize lifecycle evidence");
        let decoded: CommandLifecycleEvidence =
            serde_json::from_slice(&encoded).expect("deserialize lifecycle evidence");
        assert_eq!(decoded, evidence);

        let mut malformed = serde_json::to_value(&evidence).expect("lifecycle JSON value");
        malformed["transitions"] = serde_json::json!([]);
        assert!(serde_json::from_value::<CommandLifecycleEvidence>(malformed).is_err());
        let mut wrong_source = serde_json::to_value(&evidence).expect("lifecycle JSON value");
        wrong_source["source_client_namespace"] = serde_json::json!(7);
        assert!(serde_json::from_value::<CommandLifecycleEvidence>(wrong_source).is_err());

        let failed = ApplicationState::Failed(ApplicationFailure {
            code: "test_failure".to_owned(),
            message: "typed terminal boundary test".to_owned(),
        });
        let valid_failed = CommandLifecycleEvidence::from_terminal(
            envelope.clone(),
            Some(AdmissionSequence::new(9)),
            initial,
            terminal,
            failed.clone(),
        )
        .expect("monotonic failed lifecycle");
        let regressions = [
            (
                "tick",
                AppliedCommand {
                    tick: Tick(2),
                    ..terminal
                },
                CommandLifecycleEvidenceError::TerminalTickRegressed,
                "terminal tick precedes",
            ),
            (
                "control",
                AppliedCommand {
                    revisions: HostRevisions {
                        control: ControlRevision::new(4),
                        ..terminal.revisions
                    },
                    ..terminal
                },
                CommandLifecycleEvidenceError::TerminalControlRevisionRegressed,
                "terminal control revision precedes",
            ),
            (
                "scientific",
                AppliedCommand {
                    revisions: HostRevisions {
                        scientific: ScientificRevision::new(2),
                        ..terminal.revisions
                    },
                    ..terminal
                },
                CommandLifecycleEvidenceError::TerminalScientificRevisionRegressed,
                "terminal scientific revision precedes",
            ),
            (
                "config",
                AppliedCommand {
                    revisions: HostRevisions {
                        config: ConfigRevision::new(1),
                        ..terminal.revisions
                    },
                    ..terminal
                },
                CommandLifecycleEvidenceError::TerminalConfigRevisionRegressed,
                "terminal configuration revision precedes",
            ),
        ];
        for (axis, regressed, expected, diagnostic) in regressions {
            assert_eq!(
                CommandLifecycleEvidence::from_terminal(
                    envelope.clone(),
                    Some(AdmissionSequence::new(9)),
                    initial,
                    regressed,
                    failed.clone(),
                ),
                Err(expected)
            );

            let mut malformed =
                serde_json::to_value(&valid_failed).expect("failed lifecycle JSON value");
            match axis {
                "tick" => malformed["transitions"][1]["boundary"]["tick"] = serde_json::json!(2),
                "control" => {
                    malformed["transitions"][1]["boundary"]["revisions"]["control"] =
                        serde_json::json!(4);
                }
                "scientific" => {
                    malformed["transitions"][1]["boundary"]["revisions"]["scientific"] =
                        serde_json::json!(2);
                }
                "config" => {
                    malformed["transitions"][1]["boundary"]["revisions"]["config"] =
                        serde_json::json!(1);
                }
                _ => unreachable!("regression fixture axis"),
            }
            let error = serde_json::from_value::<CommandLifecycleEvidence>(malformed)
                .expect_err("causally inverted lifecycle wire state must fail");
            assert!(
                error.to_string().contains(diagnostic),
                "{axis} regression returned unexpected error: {error}"
            );
        }
    }

    #[test]
    fn run_ids_use_canonical_fixed_width_strings() {
        let namespaced =
            RunId::from_namespace_sequence(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
        assert_eq!(namespaced.get(), 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);

        for (run_id, expected) in [
            (RunId::new(0), "00000000000000000000000000000000"),
            (namespaced, "0123456789abcdeffedcba9876543210"),
            (RunId::new(u128::MAX), "ffffffffffffffffffffffffffffffff"),
        ] {
            assert_eq!(run_id.to_string(), expected);
            assert_eq!(
                expected.parse::<RunId>().expect("canonical run id parses"),
                run_id
            );

            let encoded = serde_json::to_string(&run_id).expect("run id should encode");
            assert_eq!(encoded, format!("\"{expected}\""));
            let decoded: RunId = serde_json::from_str(&encoded).expect("run id should round trip");
            assert_eq!(decoded, run_id);
        }
    }

    #[test]
    fn run_id_parsing_rejects_noncanonical_text() {
        for invalid in [
            "",
            "0000000000000000000000000000000",
            "000000000000000000000000000000000",
            "0000000000000000000000000000000g",
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
            "0123456789abcdeFfedcba9876543210",
            " 0000000000000000000000000000000",
            "00000000000000000000000000000000 ",
        ] {
            assert_eq!(invalid.parse::<RunId>(), Err(RunIdParseError));
        }

        assert!(serde_json::from_str::<RunId>("0").is_err());
        assert!(serde_json::from_str::<RunId>("\"abc\"").is_err());
        assert!(serde_json::from_str::<RunId>("\"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\"").is_err());
    }

    #[test]
    fn every_command_class_requires_a_terminal_lifecycle_audit() {
        for command in [
            HostCommand::Pause,
            HostCommand::Resume,
            HostCommand::SetSpeed(1.0),
            HostCommand::Step,
            HostCommand::UpdateConfig(Box::default()),
            HostCommand::Shutdown,
        ] {
            assert!(command.requires_journal());
        }
    }

    #[test]
    fn admission_is_totally_ordered_and_duplicate_ids_never_reapply() {
        let shared = SharedFakeHost::new();
        let mut driver = shared.driver();
        let mut client = HostClient::new(shared.clone());
        let first = submit_ok(&mut client, envelope(1, HostCommand::Pause));
        let second_envelope = envelope(2, HostCommand::Resume);
        let second = submit_ok(&mut client, second_envelope.clone());
        let third = submit_ok(&mut client, envelope(3, HostCommand::Step));

        assert_eq!(first.admission_sequence(), Some(AdmissionSequence::new(1)));
        assert_eq!(second.admission_sequence(), Some(AdmissionSequence::new(2)));
        assert_eq!(third.admission_sequence(), Some(AdmissionSequence::new(3)));
        assert_eq!(first.journal(), &JournalState::Pending);
        assert_eq!(second.journal(), &JournalState::Pending);
        assert_eq!(third.journal(), &JournalState::Pending);
        assert_eq!(submit_ok(&mut client, second_envelope), second);

        let receipt = driver
            .drive(ManualInstant::from_nanos(1))
            .expect("drive should succeed");
        assert_eq!(receipt.commands_completed, 3);
        assert_eq!(receipt.scientific_steps, 1);
        assert_eq!(receipt.scientific_revision, ScientificRevision::new(1));
        assert_eq!(receipt.blocker, None);
        for command_id in [CommandId::new(1), CommandId::new(2)] {
            assert_eq!(
                client
                    .command_status(command_id)
                    .expect("playback status lookup")
                    .expect("playback status retained")
                    .journal(),
                &JournalState::Durable
            );
        }
        assert_eq!(
            client
                .command_status(CommandId::new(3))
                .expect("step status lookup")
                .expect("step status retained")
                .journal(),
            &JournalState::Durable
        );
        assert!(
            shared.lock().playback.paused,
            "Step must leave playback paused"
        );
        let retried = submit_ok(&mut client, envelope(2, HostCommand::SetSpeed(99.0)));
        assert!(matches!(
            retried.application(),
            ApplicationState::Applied(_)
        ));
        assert_eq!(retried.admission_sequence(), second.admission_sequence());
        let empty_receipt = driver
            .drive(ManualInstant::from_nanos(2))
            .expect("empty drive should succeed");
        assert_eq!(empty_receipt.commands_completed, 0);
        assert_eq!(empty_receipt.scientific_steps, 0);
        assert_eq!(
            empty_receipt.scientific_revision,
            ScientificRevision::new(1)
        );
        assert_eq!(
            shared.lock().admission_order,
            vec![CommandId::new(1), CommandId::new(2), CommandId::new(3)]
        );
    }

    #[test]
    fn concurrent_retry_of_one_id_gets_one_admission() {
        let shared = SharedFakeHost::new();
        let barrier = Arc::new(Barrier::new(3));
        let request = envelope(44, HostCommand::Step);

        let spawn_submitter = |port: SharedFakeHost| {
            let barrier = Arc::clone(&barrier);
            let request = request.clone();
            std::thread::spawn(move || {
                let mut client = HostClient::new(port);
                barrier.wait();
                client.submit(request)
            })
        };
        let left = spawn_submitter(shared.clone());
        let right = spawn_submitter(shared.clone());
        barrier.wait();

        let left = left
            .join()
            .expect("left submitter should not panic")
            .expect("left retry");
        let right = right
            .join()
            .expect("right submitter should not panic")
            .expect("right retry");
        assert_eq!(left, right);
        assert_eq!(left.admission_sequence(), Some(AdmissionSequence::new(1)));
        assert_eq!(shared.lock().admission_order, vec![CommandId::new(44)]);
    }

    #[test]
    fn null_frontend_preserves_id_after_an_indeterminate_submission_receipt() {
        let shared = SharedFakeHost::new();
        let command_id = CommandId::from_client_sequence(0x77, 1);
        shared.lose_next_submission_receipt(command_id);
        let mut frontend = NullFrontend::new(shared, 0x77);

        let failure = frontend
            .pause()
            .expect_err("the first admitted receipt should be lost");
        assert_eq!(
            failure.envelope().map(|envelope| envelope.command_id),
            Some(command_id)
        );
        let retry_envelope = failure
            .into_envelope()
            .expect("an indeterminate submission preserves its exact envelope");
        let admitted = frontend
            .submit_envelope(retry_envelope)
            .expect("retry should return the existing admission");
        assert_eq!(admitted.command_id(), command_id);
        assert_eq!(
            admitted.admission_sequence(),
            Some(AdmissionSequence::new(1))
        );

        let next = frontend
            .resume()
            .expect("a later command should use the next id");
        assert_eq!(next.command_id(), CommandId::from_client_sequence(0x77, 2));
        assert_eq!(next.admission_sequence(), Some(AdmissionSequence::new(2)));
    }

    #[test]
    fn null_frontend_rejects_unrelated_or_lying_manual_drivers() {
        let shared = SharedFakeHost::new();
        let mut frontend = NullFrontend::new(shared.clone(), 0x88);
        let unrelated = SharedFakeHost::new();
        let mut unrelated_driver = unrelated.driver();
        assert!(matches!(
            frontend.drive_at(&mut unrelated_driver, ManualInstant::from_nanos(1)),
            Err(HostAccessError::DriverSessionMismatch { .. })
        ));

        let mut lying_driver = LyingFakeDriver {
            session_id: shared.session_id(),
        };
        assert!(matches!(
            frontend.drive_at(&mut lying_driver, ManualInstant::from_nanos(1)),
            Err(HostAccessError::ProtocolViolation { .. })
        ));

        let mut matching_driver = shared.driver();
        assert_eq!(
            frontend
                .drive_at(&mut matching_driver, ManualInstant::from_nanos(1))
                .expect("matching driver should be accepted")
                .now,
            ManualInstant::from_nanos(1)
        );
    }

    #[test]
    fn compare_and_set_validation_and_application_failures_stay_distinct() {
        let shared = SharedFakeHost::new();
        let mut driver = shared.driver();
        let mut client = HostClient::new(shared.clone());
        let winner = client
            .submit(
                envelope(9, HostCommand::Pause).expecting_control_revision(ControlRevision::new(0)),
            )
            .expect("first compare-and-set candidate should be admitted");
        let conflict = client
            .submit(
                envelope(10, HostCommand::Resume)
                    .expecting_control_revision(ControlRevision::new(0)),
            )
            .expect("competing compare-and-set candidate should be admitted");
        assert!(matches!(winner.application(), ApplicationState::Admitted));
        assert!(matches!(conflict.application(), ApplicationState::Admitted));
        assert_eq!(
            conflict.admission_sequence(),
            Some(AdmissionSequence::new(2))
        );
        driver
            .drive(ManualInstant::from_nanos(1))
            .expect("conflict should resolve at the application boundary");
        let conflict = client
            .command_status(conflict.command_id())
            .expect("conflict lookup")
            .expect("conflict remains queryable");
        assert!(matches!(
            conflict.application(),
            ApplicationState::Rejected(RejectionReason::ControlRevisionConflict {
                expected,
                actual,
            }) if *expected == ControlRevision::new(0) && *actual == ControlRevision::new(1)
        ));
        assert_eq!(
            conflict.admission_sequence(),
            Some(AdmissionSequence::new(2))
        );
        assert_eq!(conflict.journal(), &JournalState::Durable);

        let invalid_config = ScriptBotsConfig {
            world_width: 0,
            ..ScriptBotsConfig::default()
        };
        let rejected = client
            .submit(envelope(
                11,
                HostCommand::UpdateConfig(Box::new(invalid_config)),
            ))
            .expect("validation rejection should be inspectable");
        assert!(matches!(
            rejected.application(),
            ApplicationState::Rejected(RejectionReason::Validation { .. })
        ));
        assert_eq!(rejected.journal(), &JournalState::Pending);

        let failed_id = CommandId::new(12);
        let admitted = submit_ok(&mut client, envelope(12, HostCommand::Step));
        assert!(matches!(admitted.application(), ApplicationState::Admitted));
        shared.fail_on_application(failed_id);
        driver
            .drive(ManualInstant::from_nanos(2))
            .expect("drive should report application through status");
        let failed = client
            .command_status(failed_id)
            .expect("lookup should succeed")
            .expect("failed command should remain queryable");
        assert!(matches!(failed.application(), ApplicationState::Failed(_)));
        assert_eq!(failed.journal(), &JournalState::Durable);
    }

    #[test]
    fn a_later_client_can_lookup_status_after_submitter_disconnects() {
        let shared = SharedFakeHost::new();
        let command_id = {
            let mut submitting_client = HostClient::new(shared.clone());
            let status = submit_ok(&mut submitting_client, envelope(70, HostCommand::Pause));
            status.command_id()
        };

        let mut later_client = HostClient::new(shared);
        let status = later_client
            .command_status(command_id)
            .expect("reconnected lookup should succeed")
            .expect("admitted command should still exist");
        assert_eq!(status.command_id(), command_id);
        assert_eq!(status.admission_sequence(), Some(AdmissionSequence::new(1)));
    }

    #[test]
    fn snapshot_from_another_session_is_rejected_without_advancing_cursor() {
        let shared = SharedFakeHost::new();
        let expected = shared.lock().session_id;
        let mut client = HostClient::new(shared.clone());
        let mut subscription = client.subscribe_snapshots();
        {
            let mut host = shared.lock();
            let snapshot = host
                .latest_snapshot
                .as_mut()
                .expect("fake initial snapshot");
            Arc::make_mut(snapshot).session_id = HostSessionId::new(9_999);
            drop(host);
        }

        assert_eq!(
            client.poll_snapshot(&mut subscription),
            Err(HostAccessError::SnapshotSessionMismatch {
                expected,
                actual: HostSessionId::new(9_999),
            })
        );
        assert_eq!(subscription.last_seen(), None);
    }

    #[test]
    fn snapshot_read_handles_are_thread_safe_and_runtime_has_no_storage_dependency() {
        fn assert_send<T: Send>() {}
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<SnapshotHub>();
        assert_send::<SnapshotSubscription>();

        let manifest = include_str!("../Cargo.toml");
        for forbidden in ["scriptbots-storage", "fsqlite"] {
            assert!(
                !manifest.contains(forbidden),
                "renderer-neutral runtime manifest must not depend on {forbidden}"
            );
        }
    }

    #[test]
    fn revisions_snapshots_and_events_are_monotonic_in_their_typed_domains() {
        let shared = SharedFakeHost::new();
        let mut driver = shared.driver();
        let mut client = HostClient::new(shared);
        let mut snapshots = client.subscribe_snapshots();
        let initial = client
            .poll_snapshot(&mut snapshots)
            .expect("initial snapshot poll")
            .expect("fake host publishes an initial snapshot");

        let pause = submit_ok(&mut client, envelope(80, HostCommand::Pause));
        driver
            .drive(ManualInstant::from_nanos(1))
            .expect("pause drive");
        let pause = client
            .command_status(pause.command_id())
            .expect("pause lookup")
            .expect("pause status");
        let pause = applied(&pause).expect("pause command must reach the applied state");
        let after_pause = client
            .poll_snapshot(&mut snapshots)
            .expect("pause snapshot poll")
            .expect("pause should publish");

        let config = submit_ok(
            &mut client,
            envelope(81, HostCommand::UpdateConfig(Box::default())),
        );
        driver
            .drive(ManualInstant::from_nanos(2))
            .expect("config drive");
        let config = applied(
            &client
                .command_status(config.command_id())
                .expect("config lookup")
                .expect("config status"),
        )
        .expect("config command must reach the applied state");
        let after_config = client
            .poll_snapshot(&mut snapshots)
            .expect("config snapshot poll")
            .expect("config should publish");

        let step = submit_ok(&mut client, envelope(82, HostCommand::Step));
        driver
            .drive(ManualInstant::from_nanos(3))
            .expect("step drive");
        let step = applied(
            &client
                .command_status(step.command_id())
                .expect("step lookup")
                .expect("step status"),
        )
        .expect("step command must reach the applied state");
        let after_step = client
            .poll_snapshot(&mut snapshots)
            .expect("step snapshot poll")
            .expect("step should publish");

        assert!(initial.revision < after_pause.revision);
        assert!(after_pause.revision < after_config.revision);
        assert!(after_config.revision < after_step.revision);
        assert_eq!(pause.revisions.control, ControlRevision::new(1));
        assert_eq!(pause.revisions.scientific, ScientificRevision::new(0));
        assert_eq!(config.revisions.control, ControlRevision::new(2));
        assert_eq!(config.revisions.config, ConfigRevision::new(1));
        assert_eq!(step.revisions.control, ControlRevision::new(3));
        assert_eq!(step.revisions.scientific, ScientificRevision::new(1));
        assert_eq!(step.tick, Tick(1));

        let mut cursor = client.protocol_event_cursor();
        let events = client
            .read_protocol_events(&mut cursor, usize::MAX)
            .expect("ordered event read");
        assert_eq!(events.len(), 6);
        assert!(
            events
                .windows(2)
                .all(|pair| pair[0].sequence.checked_next() == Some(pair[1].sequence))
        );
        assert_eq!(
            cursor.last_seen(),
            events.last().expect("events exist").sequence
        );
    }

    #[test]
    fn status_constructor_accepts_every_reachable_axis_combination_only() {
        let terminal_applications = [
            ApplicationState::Applied(AppliedCommand {
                tick: Tick(4),
                revisions: HostRevisions::default(),
            }),
            ApplicationState::Failed(ApplicationFailure {
                code: "apply".to_owned(),
                message: "failed".to_owned(),
            }),
        ];
        let journals = [
            JournalState::NotRequired,
            JournalState::Pending,
            JournalState::CommittedVolatile,
            JournalState::Durable,
            JournalState::Failed(JournalFailure {
                code: "journal".to_owned(),
                message: "failed".to_owned(),
            }),
        ];

        for journal in [JournalState::NotRequired, JournalState::Pending] {
            assert!(
                CommandStatus::try_new(
                    CommandId::new(1),
                    Some(AdmissionSequence::new(1)),
                    ApplicationState::Admitted,
                    journal,
                )
                .is_ok()
            );
        }
        for application in terminal_applications {
            for journal in journals.clone() {
                assert!(
                    CommandStatus::try_new(
                        CommandId::new(1),
                        Some(AdmissionSequence::new(1)),
                        application.clone(),
                        journal,
                    )
                    .is_ok()
                );
            }
        }
        for journal in journals {
            assert!(
                CommandStatus::try_new(
                    CommandId::new(2),
                    None,
                    ApplicationState::Rejected(RejectionReason::HostStopping),
                    journal,
                )
                .is_ok()
            );
        }
        assert_eq!(
            CommandStatus::try_new(
                CommandId::new(2),
                Some(AdmissionSequence::new(2)),
                ApplicationState::Rejected(RejectionReason::HostStopping),
                JournalState::Pending,
            ),
            Err(StatusCombinationError::PreAdmissionRejectionWasAdmitted)
        );
        let conflict = ApplicationState::Rejected(RejectionReason::ControlRevisionConflict {
            expected: ControlRevision::new(1),
            actual: ControlRevision::new(2),
        });
        assert!(
            CommandStatus::try_new(
                CommandId::new(2),
                Some(AdmissionSequence::new(2)),
                conflict.clone(),
                JournalState::Pending,
            )
            .is_ok()
        );
        assert_eq!(
            CommandStatus::try_new(CommandId::new(2), None, conflict, JournalState::Pending,),
            Err(StatusCombinationError::ConflictMissingAdmission)
        );
        assert_eq!(
            CommandStatus::rejected(CommandId::new(3), RejectionReason::HostStopping)
                .expect("generic non-runtime rejection")
                .journal(),
            &JournalState::NotRequired
        );
        assert_eq!(
            CommandStatus::try_new(
                CommandId::new(4),
                Some(AdmissionSequence::new(4)),
                ApplicationState::Admitted,
                JournalState::Durable,
            ),
            Err(StatusCombinationError::AdmittedJournalAdvanced)
        );
    }

    #[test]
    fn status_validation_rejects_missing_admission() {
        assert_eq!(
            CommandStatus::try_new(
                CommandId::new(3),
                None,
                ApplicationState::Admitted,
                JournalState::NotRequired,
            ),
            Err(StatusCombinationError::MissingAdmissionSequence)
        );
        assert_eq!(
            CommandStatus::try_from(CommandStatusWire {
                command_id: CommandId::new(5),
                admission_sequence: None,
                application: ApplicationState::Applied(AppliedCommand {
                    tick: Tick(0),
                    revisions: HostRevisions::default(),
                }),
                journal: JournalState::Durable,
            }),
            Err(StatusCombinationError::MissingAdmissionSequence)
        );
    }

    #[test]
    fn null_frontend_uses_only_commands_observation_and_manual_drive() {
        let shared = SharedFakeHost::new();
        let mut driver = shared.driver();
        let mut frontend = NullFrontend::new(shared, 0x51);
        let statuses = [
            frontend.pause().expect("pause submission"),
            frontend.resume().expect("resume submission"),
            frontend.set_speed(2.5).expect("speed submission"),
            frontend.step().expect("step submission"),
            frontend
                .update_config(ScriptBotsConfig::default())
                .expect("config submission"),
            frontend.shutdown().expect("shutdown submission"),
        ];
        assert!(statuses.iter().enumerate().all(|(index, status)| {
            let sequence = u64::try_from(index).expect("test command count fits u64") + 1;
            status.command_id() == CommandId::from_client_sequence(0x51, sequence)
        }));
        assert!(
            statuses
                .iter()
                .all(|status| status.journal() == &JournalState::Pending)
        );

        let receipt = frontend
            .drive_at(&mut driver, ManualInstant::from_nanos(10))
            .expect("manual drive");
        assert_eq!(receipt.commands_completed, statuses.len());
        let snapshot = frontend
            .poll_snapshot()
            .expect("snapshot poll")
            .expect("drive should publish a snapshot");
        assert!(
            (snapshot.playback.speed_multiplier - 2.5).abs() <= f32::EPSILON,
            "speed command must be reflected exactly in the host snapshot"
        );
        assert!(snapshot.playback.paused, "Step must leave playback paused");
        assert_eq!(snapshot.world.tick, 1);
        assert_eq!(snapshot.lifecycle, HostLifecycle::Stopped);
        for status in &statuses {
            assert_eq!(
                frontend
                    .command_status(status.command_id())
                    .expect("journalled status lookup")
                    .expect("journalled status retained")
                    .journal(),
                &JournalState::Durable
            );
        }
        assert!(
            !frontend
                .read_protocol_events(128)
                .expect("event observation")
                .is_empty()
        );
        assert!(matches!(
            frontend
                .command_status(statuses[3].command_id())
                .expect("status lookup")
                .expect("step remains queryable")
                .application(),
            ApplicationState::Applied(_)
        ));
        assert!(
            frontend
                .drive_at(&mut driver, ManualInstant::from_nanos(9))
                .is_err()
        );
    }
}
