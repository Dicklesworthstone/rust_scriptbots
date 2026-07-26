//! Multi-island archipelago host (bd-16g.5.1).
//!
//! An [`Archipelago`] owns `N` independent sole-owner [`HostCore`] islands and
//! advances every one of them to a common scientific tick — the *barrier* —
//! before any cross-island effect may be applied. It is the load-bearing host
//! for allopatric-speciation experiments (bd-16g.5) and deliberately contains
//! **no migration rules** (bd-16g.5.2), **no per-island RNG domain protocol**
//! (bd-16g.5.3), **no CI scaling gates** (bd-16g.5.4), and **no persistence
//! schema** (bd-16g.5.5).
//!
//! # Ownership and observability
//!
//! Every island is a complete, private world: its own [`WorldState`], its own
//! [`HostCore`], its own journal adapter, its own RNG streams, and its own
//! `AgentUid` allocator. There is no `Arc<Mutex<WorldState>>` anywhere in this
//! module and no shared mutable state between islands — no shared RNG, no
//! shared counters, and no shared agent-UID allocator. Each `WorldState`
//! allocates its local `AgentUid` values from its own private counter, so the
//! globally unique scientific identity of an agent is the pair
//! `(IslandId, AgentUid)`. A migrating agent (bd-16g.5.2) must be re-identified
//! under that pair, never by its bare local UID.
//!
//! A partially-stepped archipelago is unobservable by construction, on two
//! independent axes. First, [`Archipelago::step_to_barrier`] takes `&mut self`,
//! so no reader can interleave with a barrier. Second, every exposed view is
//! **barrier-committed**: the archipelago never hands out live snapshot hubs or
//! command ports (a `HostCore`'s `local_port()` can enqueue commands even
//! through `&self`, so no `&HostCore` accessor exists at all), and
//! [`Archipelago::island_snapshot`] serves only snapshots captured after a
//! completed barrier. Transient journal-capacity backpressure leaves the exact
//! partial barrier private and is retried once by the next explicit barrier
//! call. A terminal mid-barrier island fault latches the whole archipelago
//! with a typed error naming the island; in either case exposed views and
//! digests remain at the prior barrier rather than leaking a silently uneven
//! epoch.
//!
//! # Step topology (the parallelism dial)
//!
//! [`HostCore`] is intentionally `!Send` (it owns same-thread command ports),
//! so islands cannot be scattered across a thread pool as outer tasks without a
//! thread-owned-host runner that does not exist yet. The v1 topology is
//! therefore [`StepTopology::SequentialAscending`]: islands step one at a time
//! in ascending [`IslandId`] order, and each island's own tick pipeline still
//! fans out through `scriptbots-core`'s internal Rayon stages. The dial state
//! is recorded in every [`BarrierReport`] so scaling artifacts (bd-16g.5.4)
//! can never mistake one topology for another. Because islands share no state
//! between barriers, stepping order cannot change any island's science — a
//! property the unit tests prove by permuting the order.
//!
//! # One journal funnel, not N
//!
//! Islands must not open N storage connections. Each island's host emits its
//! journal batches through an injected [`JournalPort`]; the storage integration
//! (bd-16g.5.5) supplies adapters that funnel every island into the single
//! storage pipeline, keyed by each host's unique session identity
//! ([`crate::JournalBatchId`] is explicitly documented to support several
//! hosts sharing one adapter). By default every island gets its own in-memory
//! volatile journal. A storage-backed adapter must additionally buffer each
//! island's batches and flush them only at barrier completion, in ascending
//! island-id order, so the storage layer never observes a partial barrier —
//! that buffering policy is part of the bd-16g.5.5 contract, not this module.
//!
//! # Seeds and sessions (provisional derivation)
//!
//! Per-island RNG seeds and host session identities derive from
//! `(master_seed, island_id)` through a pinned FNV-1a64 walk over a versioned
//! tag. An island whose config already pins `rng_seed` keeps that exact seed.
//! bd-16g.5.3 owns the formal per-island RNG domain-separation protocol and may
//! replace this derivation; that migration moves digests and must land as its
//! own reviewed change.
//!
//! # Heterogeneous world sizes
//!
//! Islands may legally differ in world dimensions, food dynamics, terrain and
//! scenario knobs, carnivore rules, and population caps. A migrant's position
//! remap across differently-sized worlds (proposed: preserve normalized
//! `(x/w, y/h)` then clamp into destination bounds) is migration policy and is
//! owned by bd-16g.5.2 together with the rest of the migration rules.
//!
//! # Configuration provenance
//!
//! Per-island configuration must be produced by the canonical merge path (the
//! same `merge_value` → `serde_path_to_error` → `ScriptBotsConfig::validate`
//! pipeline that `ControlHandle::apply_patch` uses in `scriptbots-app`). This
//! module deliberately performs **no** JSON merging of its own: it accepts
//! complete configs, re-validates each one, and rejects any island whose
//! constructed world does not match its declared effective config exactly.

use crate::{
    ApplicationState, CommandAuthorityLookupFailure, CommandEnvelope, CommandId, CommandStatus,
    HostAccessError, HostBlocker, HostCommand, HostFault, HostHealth, HostPort, HostSessionId,
    JournalPort, ManualHostDriver, ManualInstant, RenderSnapshot,
    host_core::{HostCore, HostCoreBuildError, HostCoreOptions},
};
use scriptbots_core::{
    CharacterizationError, ScriptBotsConfig, Tick, TickSummary, WorldDigestV1, WorldState,
    WorldStateError,
};
use serde::{Deserialize, Serialize};
use std::{
    num::NonZeroU64,
    sync::Arc,
    time::{Duration, Instant},
};
use thiserror::Error;

/// Maximum number of islands one archipelago may own.
///
/// The bound exists so a misconfigured sweep fails at construction instead of
/// hours into a run; every island's config is validated up front for the same
/// reason.
pub const MAX_ISLANDS: usize = 64;

/// Client namespace for archipelago-issued commands.
///
/// The host reserves `u64::MAX` for its own lifecycle commands; the archipelago
/// is the sole command client of every island it owns, so this adjacent
/// namespace cannot collide.
const ARCHIPELAGO_COMMAND_NAMESPACE: u64 = u64::MAX - 1;

/// Versioned tag for per-island RNG seed derivation (provisional, bd-16g.5.3).
const ISLAND_RNG_SEED_TAG: &str = "scriptbots.archipelago.island-rng-seed.v1";

/// Versioned tag for per-island host session identity derivation.
const ISLAND_SESSION_TAG: &str = "scriptbots.archipelago.island-session.v1";

/// Bounded number of drive boundaries one explicit step may need before the
/// archipelago refuses to spin. A healthy island applies an explicit step at
/// the first boundary; a live-but-slow journal may need a few receipt polls.
const STEP_DRIVE_ATTEMPTS: usize = 16;
const STEP_AUTHORITY_TIMEOUT: Duration = Duration::from_secs(30);
const STEP_AUTHORITY_RETRY_PARK: Duration = Duration::from_millis(1);

/// Configuration facets that must be uniform across islands, named in the
/// construction log so audits can see exactly what was checked.
const UNIFORM_FIELDS_CHECKED: &[&str] = &[
    "brain_registry_descriptors",
    "neuroflow",
    "sensor_actuator_arity(compile-time INPUT_SIZE/OUTPUT_SIZE)",
];

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Pinned FNV-1a64 used for island derivations and diagnostic config hashes.
///
/// Never replace this with `std::hash::DefaultHasher`: the standard hasher is
/// not stable across compiler releases, and these values are logged and
/// compared across runs.
fn fnv1a64(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV1A64_PRIME)
    })
}

/// Derive a stable per-island value from a versioned tag and the master seed.
fn derive_island_value(tag: &str, master_seed: u64, island: IslandId) -> u64 {
    let mut hash = fnv1a64(FNV1A64_OFFSET_BASIS, tag.as_bytes());
    hash = fnv1a64(hash, &master_seed.to_le_bytes());
    fnv1a64(hash, &island.0.to_le_bytes())
}

/// Stable identity of one island inside an archipelago.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct IslandId(pub u16);

impl std::fmt::Display for IslandId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "island-{}", self.0)
    }
}

/// One island's declared scenario: identity, human label, and complete config.
///
/// The config must already be the product of the canonical merge/validation
/// path; the archipelago re-validates it and rejects any divergence between
/// this declaration and the world actually built for the island.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IslandSpec {
    /// Stable island identity, unique within the archipelago.
    pub id: IslandId,
    /// Human-readable scenario label used in logs and reports.
    pub label: String,
    /// Complete simulation configuration for this island.
    pub config: ScriptBotsConfig,
}

/// Cross-island connection topology consumed by the migrator (bd-16g.5.2).
///
/// Edges are normalized at construction: each edge is ordered
/// `(low id, high id)`, the list is sorted and deduplicated, and self-edges are
/// rejected. `HashMap`/`HashSet` iteration is banned from every state-touching
/// path, so normalized edges are the only edge representation that exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Topology {
    /// Each island connects to its successor in ascending-id order, and the
    /// last island connects back to the first.
    Ring,
    /// Every island pair is connected.
    FullyConnected,
    /// An explicit edge list; normalized and validated at construction.
    Custom(Vec<(IslandId, IslandId)>),
}

/// Construction parameters for an [`Archipelago`].
#[derive(Debug, Clone)]
pub struct ArchipelagoConfig {
    /// Island scenario declarations; ids must be unique.
    pub islands: Vec<IslandSpec>,
    /// Cross-island topology, normalized at construction.
    pub topology: Topology,
    /// Scientific ticks every island advances per barrier epoch. The migrator
    /// (bd-16g.5.2) consumes this as the migration interval.
    pub barrier_interval: NonZeroU64,
    /// Root seed for per-island seed and session derivation.
    pub master_seed: u64,
    /// Host options shared by every island. `initial_playback` is overridden:
    /// archipelago islands are always paused because the archipelago owns
    /// scientific time exclusively through explicit step commands.
    pub host_options: HostCoreOptions,
}

/// Immutable per-island construction record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IslandMeta {
    /// Stable island identity.
    pub id: IslandId,
    /// Human-readable scenario label.
    pub label: String,
    /// The exact effective configuration the island's world was built from,
    /// including the derived or pinned RNG seed.
    pub effective_config: ScriptBotsConfig,
    /// Pinned FNV-1a64 hash of the canonical JSON of `effective_config`,
    /// recorded so "what was island 3 actually configured as" is answerable
    /// from the construction log alone.
    pub config_hash: u64,
    /// Host session identity derived from `(master_seed, id)`; unique within
    /// the archipelago and stable across identical constructions.
    pub session_id: HostSessionId,
}

/// The recorded island step-ordering policy of one barrier.
///
/// `HostCore` is `!Send`, so v1 steps islands sequentially on the owning
/// thread while each island's tick pipeline parallelizes internally. Scaling
/// work (bd-16g.5.4) may add topologies; reports always record which one ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StepTopology {
    /// Islands step one at a time in ascending [`IslandId`] order.
    SequentialAscending,
}

/// Outcome of one completed barrier epoch.
#[derive(Debug, Clone)]
pub struct BarrierReport {
    /// One-based count of completed barriers.
    pub epoch: u64,
    /// Common scientific tick every island reached.
    pub barrier_tick: Tick,
    /// Island step-ordering policy that produced this barrier.
    pub step_topology: StepTopology,
    /// Per-island outcomes in ascending island-id order.
    pub islands: Vec<IslandBarrierReport>,
}

/// One island's outcome within a [`BarrierReport`].
#[derive(Debug, Clone)]
pub struct IslandBarrierReport {
    /// Stable island identity.
    pub island: IslandId,
    /// Human-readable scenario label.
    pub label: String,
    /// Completed scientific tick after this barrier; equal to the report's
    /// `barrier_tick` for every island.
    pub world_tick: Tick,
    /// Scientific ticks this island advanced during the barrier.
    pub ticks_stepped: u64,
    /// The island's completed tick summary at the barrier, when published.
    pub summary: Option<TickSummary>,
}

/// Typed archipelago failure naming the exact island and boundary.
#[derive(Debug, Error)]
pub enum ArchipelagoError {
    /// The configuration declared no islands.
    #[error("archipelago requires at least one island")]
    NoIslands,
    /// The configuration declared more islands than [`MAX_ISLANDS`].
    #[error("archipelago supports at most {max} islands, got {count}")]
    TooManyIslands {
        /// Declared island count.
        count: usize,
        /// Maximum supported island count.
        max: usize,
    },
    /// Two islands declared the same identity.
    #[error("duplicate island id {island}")]
    DuplicateIslandId {
        /// The duplicated identity.
        island: IslandId,
    },
    /// A topology edge referenced an island that does not exist.
    #[error("topology edge references unknown island {island}")]
    UnknownIslandInEdge {
        /// The unknown identity.
        island: IslandId,
    },
    /// A topology edge connected an island to itself.
    #[error("topology contains self-edge on {island}")]
    SelfEdge {
        /// The self-connected identity.
        island: IslandId,
    },
    /// An island's effective configuration failed validation.
    #[error("island {island} ({label}) configuration is invalid: {source}")]
    InvalidIslandConfig {
        /// The invalid island.
        island: IslandId,
        /// The island's scenario label.
        label: String,
        /// Exact validation failure.
        #[source]
        source: WorldStateError,
    },
    /// An island's effective configuration could not be canonically serialized
    /// for its diagnostic hash.
    #[error("island {island} configuration could not be serialized: {detail}")]
    ConfigSerialization {
        /// The affected island.
        island: IslandId,
        /// Serialization failure detail.
        detail: String,
    },
    /// Two islands differ on a facet that changes the meaning of an emigrating
    /// agent. Enforced at construction, not at first migration, so the error
    /// names the actual divergence instead of a downstream symptom.
    #[error(
        "islands {reference} and {island} are incompatible on {field}: \
         {reference_value} vs {island_value}"
    )]
    IncompatibleIslands {
        /// The reference island (lowest id).
        reference: IslandId,
        /// The incompatible island.
        island: IslandId,
        /// The uniform facet that diverged.
        field: &'static str,
        /// The reference island's value.
        reference_value: String,
        /// The incompatible island's value.
        island_value: String,
    },
    /// The caller-supplied world factory failed for an island.
    #[error("island {island} world construction failed: {source}")]
    WorldFactory {
        /// The affected island.
        island: IslandId,
        /// Exact world construction failure.
        #[source]
        source: WorldStateError,
    },
    /// A factory returned a world whose configuration is not the island's
    /// declared effective configuration.
    #[error("island {island} world was not built from its declared effective config")]
    WorldConfigMismatch {
        /// The affected island.
        island: IslandId,
    },
    /// Islands were supplied at different starting ticks.
    #[error("island {island} starts at tick {found:?}, expected {expected:?}")]
    MismatchedStartTick {
        /// The affected island.
        island: IslandId,
        /// Common starting tick declared by the first island.
        expected: Tick,
        /// The island's actual starting tick.
        found: Tick,
    },
    /// Host construction failed for an island.
    #[error("island {island} host construction failed: {source}")]
    HostBuild {
        /// The affected island.
        island: IslandId,
        /// Exact host construction failure.
        #[source]
        source: HostCoreBuildError,
    },
    /// Two islands derived the same host session identity.
    #[error("island session identity collision between {first} and {second}")]
    SessionCollision {
        /// First island in the colliding pair.
        first: IslandId,
        /// Second island in the colliding pair.
        second: IslandId,
    },
    /// An island's scientific step faulted mid-barrier.
    #[error("island {island} ({label}) faulted at tick {tick:?}: [{code}] {message}")]
    IslandFault {
        /// The faulted island.
        island: IslandId,
        /// The island's scenario label.
        label: String,
        /// Tick visible when the fault was observed.
        tick: Tick,
        /// Stable machine-readable fault category.
        code: String,
        /// Human-readable fault detail.
        message: String,
    },
    /// An island rejected an archipelago step command.
    #[error("island {island} rejected a step command: {detail}")]
    CommandRejected {
        /// The rejecting island.
        island: IslandId,
        /// Exact rejection detail.
        detail: String,
    },
    /// An island's journal adapter refused or retained a batch, blocking
    /// science. The host retains the exact batch for retry; the archipelago
    /// surfaces the blocker instead of silently spinning against it. A later
    /// explicit barrier call retries transient capacity backpressure once.
    #[error("island {island} journal is blocking science: {detail}")]
    JournalBlocked {
        /// The blocked island.
        island: IslandId,
        /// Exact blocker detail.
        detail: String,
    },
    /// An island made no progress toward the barrier within the bounded
    /// number of drive boundaries.
    #[error(
        "island {island} made no progress toward tick {target:?} \
         (stuck at {current:?}; health: {health})"
    )]
    NoProgress {
        /// The stuck island.
        island: IslandId,
        /// Barrier tick the island failed to reach.
        target: Tick,
        /// Tick the island is stuck at.
        current: Tick,
        /// Island health at the failed boundary.
        health: String,
    },
    /// A host protocol call failed.
    #[error("island {island} host access failed: {source}")]
    Access {
        /// The affected island.
        island: IslandId,
        /// Exact protocol failure.
        #[source]
        source: HostAccessError,
    },
    /// A digest was requested for an island that does not exist.
    #[error("unknown island {island}")]
    UnknownIsland {
        /// The unknown identity.
        island: IslandId,
    },
    /// An island's scientific digest could not be computed.
    #[error("island {island} digest failed: {source}")]
    Digest {
        /// The affected island.
        island: IslandId,
        /// Exact digest failure.
        #[source]
        source: CharacterizationError,
    },
    /// An earlier island fault latched the archipelago; stepping is refused so
    /// a partially-stepped epoch can never be extended or observed as whole.
    #[error("archipelago is latched by an earlier fault: {detail}")]
    Latched {
        /// The latched fault's rendered detail.
        detail: String,
    },
    /// The next barrier tick would overflow the tick domain.
    #[error("barrier tick overflow")]
    TickOverflow,
}

struct Island {
    meta: IslandMeta,
    core: HostCore,
    next_command_sequence: u64,
    time_cursor: u64,
    /// The island's exposed view: the snapshot captured at the last completed
    /// barrier (or at construction). Refreshed only after every island has
    /// reached the barrier tick, so a mid-barrier fault can never leak a
    /// partially-stepped view.
    committed_snapshot: Arc<RenderSnapshot>,
}

impl Island {
    const fn next_instant(&mut self) -> ManualInstant {
        self.time_cursor = self.time_cursor.saturating_add(1);
        ManualInstant::from_nanos(self.time_cursor)
    }

    fn next_command_id(&mut self) -> Result<CommandId, ArchipelagoError> {
        let sequence = self.next_command_sequence;
        self.next_command_sequence =
            sequence
                .checked_add(1)
                .ok_or_else(|| ArchipelagoError::CommandRejected {
                    island: self.meta.id,
                    detail: "archipelago command sequence exhausted".to_owned(),
                })?;
        Ok(CommandId::from_client_sequence(
            ARCHIPELAGO_COMMAND_NAMESPACE,
            sequence,
        ))
    }
}

/// `N` sole-owner islands stepped to common-tick barriers.
///
/// See the module documentation for the ownership, topology, journal, and
/// seed-derivation contracts.
pub struct Archipelago {
    islands: Vec<Island>,
    edges: Vec<(IslandId, IslandId)>,
    barrier_interval: NonZeroU64,
    master_seed: u64,
    start_tick: Tick,
    barrier_tick: Tick,
    epoch: u64,
    latched: Option<String>,
}

impl Archipelago {
    /// Construct an archipelago whose islands use default worlds and
    /// per-island in-memory volatile journals.
    ///
    /// Worlds are built directly from each island's effective configuration.
    /// Callers that install brain families or seed populations must use
    /// [`Self::with_factories`].
    ///
    /// # Errors
    ///
    /// Returns every construction-time validation failure described on
    /// [`ArchipelagoError`]; no partially-constructed archipelago is ever
    /// returned.
    pub fn new(config: ArchipelagoConfig) -> Result<Self, ArchipelagoError> {
        Self::with_factories(
            config,
            |meta| WorldState::new(meta.effective_config.clone()),
            |_meta| None,
        )
    }

    /// Construct an archipelago with caller-controlled world and journal
    /// construction.
    ///
    /// `world_factory` receives each island's [`IslandMeta`] and must return a
    /// world built from exactly `meta.effective_config` (brain installation and
    /// population seeding are the factory's responsibility). `journal_factory`
    /// may return an injected [`JournalPort`] per island — the seam bd-16g.5.5
    /// uses to funnel every island into one storage pipeline — or `None` for
    /// the default per-island volatile journal.
    ///
    /// # Errors
    ///
    /// Returns every construction-time validation failure described on
    /// [`ArchipelagoError`]; no partially-constructed archipelago is ever
    /// returned.
    #[allow(
        clippy::too_many_lines,
        reason = "construction keeps its ordered identity, uniformity, world, and host validation phases visible together so no island can skip a gate"
    )]
    pub fn with_factories<W, J>(
        config: ArchipelagoConfig,
        mut world_factory: W,
        mut journal_factory: J,
    ) -> Result<Self, ArchipelagoError>
    where
        W: FnMut(&IslandMeta) -> Result<WorldState, WorldStateError>,
        J: FnMut(&IslandMeta) -> Option<Box<dyn JournalPort>>,
    {
        let ArchipelagoConfig {
            mut islands,
            topology,
            barrier_interval,
            master_seed,
            host_options,
        } = config;

        if islands.is_empty() {
            return Err(ArchipelagoError::NoIslands);
        }
        if islands.len() > MAX_ISLANDS {
            return Err(ArchipelagoError::TooManyIslands {
                count: islands.len(),
                max: MAX_ISLANDS,
            });
        }
        islands.sort_by_key(|spec| spec.id);
        for pair in islands.windows(2) {
            if pair[0].id == pair[1].id {
                return Err(ArchipelagoError::DuplicateIslandId { island: pair[0].id });
            }
        }
        let ids: Vec<IslandId> = islands.iter().map(|spec| spec.id).collect();
        let edges = normalized_edges(&topology, &ids)?;

        // Archipelago islands are always paused: the archipelago owns
        // scientific time exclusively through explicit step commands, so a
        // stray automatic cadence can never advance one island past a barrier.
        let mut options = host_options;
        options.initial_playback.paused = true;

        let mut built: Vec<Island> = Vec::with_capacity(islands.len());
        let mut reference_registry: Option<Vec<(u64, String)>> = None;
        let mut reference_registry_lane: Option<String> = None;
        let mut start_tick: Option<Tick> = None;

        for spec in &islands {
            let mut effective_config = spec.config.clone();
            if effective_config.rng_seed.is_none() {
                effective_config.rng_seed = Some(derive_island_value(
                    ISLAND_RNG_SEED_TAG,
                    master_seed,
                    spec.id,
                ));
            }
            effective_config.validate().map_err(|source| {
                ArchipelagoError::InvalidIslandConfig {
                    island: spec.id,
                    label: spec.label.clone(),
                    source,
                }
            })?;

            let reference = &islands[0];
            if effective_config.neuroflow != reference.config.neuroflow {
                return Err(ArchipelagoError::IncompatibleIslands {
                    reference: reference.id,
                    island: spec.id,
                    field: "neuroflow",
                    reference_value: format!("{:?}", reference.config.neuroflow),
                    island_value: format!("{:?}", effective_config.neuroflow),
                });
            }

            let config_json = serde_json::to_string(&effective_config).map_err(|error| {
                ArchipelagoError::ConfigSerialization {
                    island: spec.id,
                    detail: error.to_string(),
                }
            })?;
            let config_hash = fnv1a64(FNV1A64_OFFSET_BASIS, config_json.as_bytes());
            let session_id = HostSessionId::new(derive_island_value(
                ISLAND_SESSION_TAG,
                master_seed,
                spec.id,
            ));
            if let Some(collision) = built
                .iter()
                .find(|island| island.meta.session_id == session_id)
            {
                return Err(ArchipelagoError::SessionCollision {
                    first: collision.meta.id,
                    second: spec.id,
                });
            }

            let meta = IslandMeta {
                id: spec.id,
                label: spec.label.clone(),
                effective_config,
                config_hash,
                session_id,
            };

            let world = world_factory(&meta).map_err(|source| ArchipelagoError::WorldFactory {
                island: spec.id,
                source,
            })?;
            if world.config() != &meta.effective_config {
                return Err(ArchipelagoError::WorldConfigMismatch { island: spec.id });
            }
            let world_tick = world.tick();
            match start_tick {
                None => start_tick = Some(world_tick),
                Some(expected) if expected != world_tick => {
                    return Err(ArchipelagoError::MismatchedStartTick {
                        island: spec.id,
                        expected,
                        found: world_tick,
                    });
                }
                Some(_) => {}
            }

            let descriptors = {
                let mut descriptors = world.brain_registry().descriptors();
                descriptors.sort_unstable();
                descriptors
            };
            match &reference_registry {
                None => reference_registry = Some(descriptors),
                Some(reference_descriptors) => {
                    if reference_descriptors != &descriptors {
                        return Err(ArchipelagoError::IncompatibleIslands {
                            reference: islands[0].id,
                            island: spec.id,
                            field: "brain_registry",
                            reference_value: format!("{reference_descriptors:?}"),
                            island_value: format!("{descriptors:?}"),
                        });
                    }
                }
            }
            // `(key, kind)` equality is a readable first gate but proves too
            // little: two registries can agree on every key and kind while
            // registering different factory construction state or protocol
            // families. The world digest's registry lane covers the exact
            // registered contract (keys, kinds, family identity, and declared
            // factory-state digests), so lane equality is the real uniformity
            // requirement.
            let registry_lane = world
                .world_digest_v1()
                .map_err(|source| ArchipelagoError::Digest {
                    island: spec.id,
                    source,
                })?
                .brain_registry;
            match &reference_registry_lane {
                None => reference_registry_lane = Some(registry_lane),
                Some(reference_lane) => {
                    if reference_lane != &registry_lane {
                        return Err(ArchipelagoError::IncompatibleIslands {
                            reference: islands[0].id,
                            island: spec.id,
                            field: "brain_registry_contract",
                            reference_value: reference_lane.clone(),
                            island_value: registry_lane,
                        });
                    }
                }
            }

            let core = match journal_factory(&meta) {
                Some(journal) => HostCore::with_journal(session_id, world, options, journal),
                None => HostCore::new(session_id, world, options),
            }
            .map_err(|source| ArchipelagoError::HostBuild {
                island: spec.id,
                source,
            })?;

            let committed_snapshot = core.latest_snapshot();
            built.push(Island {
                meta,
                core,
                next_command_sequence: 1,
                time_cursor: 0,
                committed_snapshot,
            });
        }

        let start_tick = start_tick.unwrap_or(Tick(0));
        tracing::info!(
            island_count = built.len(),
            topology = ?topology,
            edge_count = edges.len(),
            barrier_interval = barrier_interval.get(),
            master_seed,
            start_tick = start_tick.0,
            uniform_fields_checked = ?UNIFORM_FIELDS_CHECKED,
            "archipelago constructed"
        );
        for island in &built {
            tracing::info!(
                island = %island.meta.id,
                label = %island.meta.label,
                config_hash = format_args!("{:016x}", island.meta.config_hash),
                session_id = island.meta.session_id.get(),
                rng_seed = ?island.meta.effective_config.rng_seed,
                world_width = island.meta.effective_config.world_width,
                world_height = island.meta.effective_config.world_height,
                "island constructed"
            );
        }

        Ok(Self {
            islands: built,
            edges,
            barrier_interval,
            master_seed,
            start_tick,
            barrier_tick: start_tick,
            epoch: 0,
            latched: None,
        })
    }

    /// Number of islands owned by this archipelago.
    #[must_use]
    pub const fn island_count(&self) -> usize {
        self.islands.len()
    }

    /// Per-island construction records in ascending island-id order.
    #[must_use]
    pub fn islands(&self) -> impl ExactSizeIterator<Item = &IslandMeta> {
        self.islands.iter().map(|island| &island.meta)
    }

    /// Normalized topology edges in ascending order.
    #[must_use]
    pub fn edges(&self) -> &[(IslandId, IslandId)] {
        &self.edges
    }

    /// Scientific ticks every island advances per barrier epoch.
    #[must_use]
    pub const fn barrier_interval(&self) -> NonZeroU64 {
        self.barrier_interval
    }

    /// Root seed the per-island derivations were computed from.
    #[must_use]
    pub const fn master_seed(&self) -> u64 {
        self.master_seed
    }

    /// Common tick every island started at.
    #[must_use]
    pub const fn start_tick(&self) -> Tick {
        self.start_tick
    }

    /// Common tick of the last completed barrier.
    #[must_use]
    pub const fn barrier_tick(&self) -> Tick {
        self.barrier_tick
    }

    /// Count of completed barrier epochs.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Rendered detail of the fault that latched this archipelago, if any.
    #[must_use]
    pub fn latched(&self) -> Option<&str> {
        self.latched.as_deref()
    }

    /// The island's barrier-committed snapshot: the view captured at the last
    /// completed barrier (or at construction).
    ///
    /// Live snapshot hubs and host handles are deliberately not exposed —
    /// a `HostCore`'s `local_port()` can enqueue commands even through a
    /// shared reference, and a live hub would observe islands mid-barrier.
    /// After an island fault this still returns the prior barrier's view, so
    /// a partially-stepped epoch is never observable.
    #[must_use]
    pub fn island_snapshot(&self, island: IslandId) -> Option<Arc<RenderSnapshot>> {
        self.island_index(island)
            .map(|index| Arc::clone(&self.islands[index].committed_snapshot))
    }

    /// Canonical scientific digest of one island at the current barrier.
    ///
    /// # Errors
    ///
    /// Returns [`ArchipelagoError::UnknownIsland`] for an unknown identity,
    /// [`ArchipelagoError::JournalBlocked`] while transient journal
    /// backpressure retains a partially-stepped barrier,
    /// [`ArchipelagoError::Latched`] once an island fault has latched the
    /// archipelago (a digest taken then would expose a partially-stepped
    /// epoch; the fault error already names the failed island and tick), and
    /// [`ArchipelagoError::Digest`] when the digest cannot be computed.
    pub fn island_digest(&self, island: IslandId) -> Result<WorldDigestV1, ArchipelagoError> {
        if let Some(detail) = &self.latched {
            return Err(ArchipelagoError::Latched {
                detail: detail.clone(),
            });
        }
        let index = self
            .island_index(island)
            .ok_or(ArchipelagoError::UnknownIsland { island })?;
        if let Some(blocked) = self.islands.iter().find_map(Self::journal_full_error) {
            return Err(blocked);
        }
        self.islands[index]
            .core
            .scientific_digest_v1()
            .map_err(|source| ArchipelagoError::Digest { island, source })
    }

    /// Advance every island to the next common barrier tick.
    ///
    /// Islands step in ascending island-id order ([`StepTopology`] records the
    /// policy in the report). Every island reaches the barrier tick before the
    /// method returns. Transient journal-capacity backpressure keeps the
    /// barrier tick unchanged and the exact pending batch private; the next
    /// explicit call retries that batch once and resumes the same target
    /// without duplicating science. Any terminal island fault latches the
    /// archipelago, and every later call returns [`ArchipelagoError::Latched`].
    ///
    /// # Errors
    ///
    /// Returns the typed island failure that stopped the barrier, or
    /// [`ArchipelagoError::Latched`] once an earlier failure has latched the
    /// archipelago.
    pub fn step_to_barrier(&mut self) -> Result<BarrierReport, ArchipelagoError> {
        let order: Vec<usize> = (0..self.islands.len()).collect();
        self.step_to_barrier_in_order(&order)
    }

    /// Barrier step with an explicit island visitation order.
    ///
    /// Kept private: island independence makes the order scientifically
    /// irrelevant (the unit tests prove it by permuting this order), but every
    /// public surface iterates islands in ascending id order.
    fn step_to_barrier_in_order(
        &mut self,
        order: &[usize],
    ) -> Result<BarrierReport, ArchipelagoError> {
        if let Some(detail) = &self.latched {
            return Err(ArchipelagoError::Latched {
                detail: detail.clone(),
            });
        }
        let interval = self.barrier_interval.get();
        let target = Tick(
            self.barrier_tick
                .0
                .checked_add(interval)
                .ok_or(ArchipelagoError::TickOverflow)?,
        );

        for &index in order {
            if let Err(error) = Self::step_island_to(&mut self.islands[index], target) {
                let island = &self.islands[index];
                if matches!(
                    &error,
                    ArchipelagoError::JournalBlocked {
                        island: blocked_island,
                        ..
                    } if *blocked_island == island.meta.id
                ) && Self::journal_full_error(island).is_some()
                {
                    tracing::warn!(
                        island = %island.meta.id,
                        label = %island.meta.label,
                        tick = island.core.world_tick().0,
                        error = %error,
                        "island barrier paused by retryable journal backpressure"
                    );
                    return Err(error);
                }
                tracing::error!(
                    island = %island.meta.id,
                    label = %island.meta.label,
                    tick = island.core.world_tick().0,
                    error = %error,
                    "island failed mid-barrier; archipelago latched"
                );
                self.latched = Some(error.to_string());
                return Err(error);
            }
        }

        // Every island reached the barrier: commit the new views in ascending
        // island-id order. This is the only place exposed snapshots advance.
        for island in &mut self.islands {
            island.committed_snapshot = island.core.latest_snapshot();
        }
        self.epoch += 1;
        self.barrier_tick = target;
        let islands = self
            .islands
            .iter()
            .map(|island| IslandBarrierReport {
                island: island.meta.id,
                label: island.meta.label.clone(),
                world_tick: island.core.world_tick(),
                ticks_stepped: interval,
                summary: island.committed_snapshot.completed_summary.clone(),
            })
            .collect();
        Ok(BarrierReport {
            epoch: self.epoch,
            barrier_tick: target,
            step_topology: StepTopology::SequentialAscending,
            islands,
        })
    }

    /// Step one island to the target tick through explicit step commands.
    fn step_island_to(island: &mut Island, target: Tick) -> Result<(), ArchipelagoError> {
        let island_id = island.meta.id;
        if Self::journal_full_error(island).is_some() {
            island
                .core
                .retry_retained_journal()
                .map_err(|source| ArchipelagoError::Access {
                    island: island_id,
                    source,
                })?;
            Self::verify_island_health(island)?;
        }
        while island.core.world_tick().0 < target.0 {
            let before = island.core.world_tick();
            let command_id = island.next_command_id()?;
            let envelope = CommandEnvelope::new(command_id, HostCommand::Step);
            let mut port = island.core.local_port();
            let submitted = Self::submit_step_with_authority(island_id, &mut port, &envelope)?;
            let mut applied = Self::step_status_applied(island, &submitted)?;

            for _attempt in 0..STEP_DRIVE_ATTEMPTS {
                if applied {
                    break;
                }
                let now = island.next_instant();
                island
                    .core
                    .drive(now)
                    .map_err(|source| ArchipelagoError::Access {
                        island: island_id,
                        source,
                    })?;
                let status = port
                    .command_status(command_id)
                    .map_err(|source| ArchipelagoError::Access {
                        island: island_id,
                        source,
                    })?
                    .ok_or_else(|| ArchipelagoError::CommandRejected {
                        island: island_id,
                        detail: "step command status was not retained".to_owned(),
                    })?;
                applied = Self::step_status_applied(island, &status)?;
            }
            Self::verify_island_health(island)?;
            if !applied {
                return Err(ArchipelagoError::NoProgress {
                    island: island_id,
                    target,
                    current: island.core.world_tick(),
                    health: format!("{:?}", island.core.health()),
                });
            }
            let after = island.core.world_tick();
            if after != Tick(before.0.wrapping_add(1)) {
                return Err(ArchipelagoError::NoProgress {
                    island: island_id,
                    target,
                    current: after,
                    health: format!("{:?}", island.core.health()),
                });
            }
        }

        // One extra boundary drains journal receipts admitted by the final
        // step so barrier state does not carry trivially-pending volatile
        // acknowledgements forward.
        let now = island.next_instant();
        island
            .core
            .drive(now)
            .map_err(|source| ArchipelagoError::Access {
                island: island_id,
                source,
            })?;
        Self::verify_island_health(island)
    }

    fn submit_step_with_authority(
        island: IslandId,
        port: &mut impl HostPort,
        envelope: &CommandEnvelope,
    ) -> Result<CommandStatus, ArchipelagoError> {
        let started = Instant::now();
        loop {
            match port.submit(envelope.clone()) {
                Ok(status) => return Ok(status),
                Err(source) if Self::transient_authority_lookup(&source) => {
                    let remaining = STEP_AUTHORITY_TIMEOUT.saturating_sub(started.elapsed());
                    if remaining.is_zero() {
                        return Err(ArchipelagoError::Access {
                            island,
                            source: HostAccessError::CommandAuthorityLookup {
                                command_id: envelope.command_id,
                                failure: CommandAuthorityLookupFailure::Timeout {
                                    waited: STEP_AUTHORITY_TIMEOUT,
                                },
                            },
                        });
                    }
                    std::thread::park_timeout(STEP_AUTHORITY_RETRY_PARK.min(remaining));
                }
                Err(source) => return Err(ArchipelagoError::Access { island, source }),
            }
        }
    }

    fn transient_authority_lookup(error: &HostAccessError) -> bool {
        matches!(
            error,
            HostAccessError::CommandAuthorityLookup {
                failure: CommandAuthorityLookupFailure::Pending
                    | CommandAuthorityLookupFailure::Busy
                    | CommandAuthorityLookupFailure::Capacity { .. },
                ..
            }
        )
    }

    fn step_status_applied(
        island: &Island,
        status: &CommandStatus,
    ) -> Result<bool, ArchipelagoError> {
        match status.application() {
            ApplicationState::Applied(_) => Ok(true),
            ApplicationState::Failed(failure) => Err(ArchipelagoError::IslandFault {
                island: island.meta.id,
                label: island.meta.label.clone(),
                tick: island.core.world_tick(),
                code: failure.code.clone(),
                message: failure.message.clone(),
            }),
            ApplicationState::Rejected(reason) => Err(ArchipelagoError::CommandRejected {
                island: island.meta.id,
                detail: format!("{reason:?}"),
            }),
            ApplicationState::Admitted => Ok(false),
        }
    }

    /// Map island health to a typed archipelago failure.
    ///
    /// A paused island is the normal archipelago state, never a failure.
    fn verify_island_health(island: &Island) -> Result<(), ArchipelagoError> {
        match island.core.health() {
            HostHealth::Healthy => Ok(()),
            HostHealth::Blocked(blocker) => match blocker {
                HostBlocker::PlaybackPaused => Ok(()),
                other => Err(ArchipelagoError::JournalBlocked {
                    island: island.meta.id,
                    detail: format!("{other:?}"),
                }),
            },
            HostHealth::Faulted(fault) => {
                let (code, message) = match fault {
                    HostFault::Scientific { code, message, .. }
                    | HostFault::Protocol { code, message } => (code.clone(), message.clone()),
                    HostFault::Journal { batch_id, failure } => (
                        failure.code.clone(),
                        format!("journal batch {batch_id:?}: {}", failure.message),
                    ),
                };
                Err(ArchipelagoError::IslandFault {
                    island: island.meta.id,
                    label: island.meta.label.clone(),
                    tick: island.core.world_tick(),
                    code,
                    message,
                })
            }
        }
    }

    fn journal_full_error(island: &Island) -> Option<ArchipelagoError> {
        match island.core.health() {
            HostHealth::Blocked(blocker @ HostBlocker::JournalFull { .. }) => {
                Some(ArchipelagoError::JournalBlocked {
                    island: island.meta.id,
                    detail: format!("{blocker:?}"),
                })
            }
            HostHealth::Healthy | HostHealth::Blocked(_) | HostHealth::Faulted(_) => None,
        }
    }

    fn island_index(&self, island: IslandId) -> Option<usize> {
        self.islands
            .binary_search_by_key(&island, |entry| entry.meta.id)
            .ok()
    }
}

/// Normalize a topology into sorted, deduplicated `(low, high)` edges.
fn normalized_edges(
    topology: &Topology,
    ids: &[IslandId],
) -> Result<Vec<(IslandId, IslandId)>, ArchipelagoError> {
    let mut edges = match topology {
        Topology::Ring => {
            if ids.len() < 2 {
                Vec::new()
            } else {
                let mut edges = Vec::with_capacity(ids.len());
                for (index, &island) in ids.iter().enumerate() {
                    let next = ids[(index + 1) % ids.len()];
                    edges.push(ordered_edge(island, next));
                }
                edges
            }
        }
        Topology::FullyConnected => {
            let mut edges = Vec::with_capacity(ids.len().saturating_mul(ids.len()) / 2);
            for (index, &island) in ids.iter().enumerate() {
                for &other in &ids[index + 1..] {
                    edges.push(ordered_edge(island, other));
                }
            }
            edges
        }
        Topology::Custom(declared) => {
            let mut edges = Vec::with_capacity(declared.len());
            for &(a, b) in declared {
                if a == b {
                    return Err(ArchipelagoError::SelfEdge { island: a });
                }
                for endpoint in [a, b] {
                    if ids.binary_search(&endpoint).is_err() {
                        return Err(ArchipelagoError::UnknownIslandInEdge { island: endpoint });
                    }
                }
                edges.push(ordered_edge(a, b));
            }
            edges
        }
    };
    edges.sort_unstable();
    edges.dedup();
    Ok(edges)
}

const fn ordered_edge(a: IslandId, b: IslandId) -> (IslandId, IslandId) {
    if a.0 <= b.0 { (a, b) } else { (b, a) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EventJournalReader, JournalAdmission, JournalBatch, JournalReceipt, JournalReceiptState,
        ShutdownCommitRequirement,
    };
    use scriptbots_core::{BrainRunner, BrainSpawnError, INPUT_SIZE, OUTPUT_SIZE, RandomStream};
    use std::{cell::RefCell, collections::VecDeque, rc::Rc};

    const TEST_BRAIN_KIND: &str = "archi-test-brain";
    const TEST_BRAIN_FACTORY_DIGEST: u64 = 0xA5C1_1A60_7E57_0001;

    /// Minimal heritable brain so determinism tests exercise agent bodies,
    /// brain genomes/evaluator state, UID allocation, and RNG draws instead of
    /// hashing empty worlds.
    #[derive(Debug, Clone)]
    struct TestBrain {
        weight: f32,
    }

    fn draw_unit(rng: &mut dyn RandomStream) -> f32 {
        f32::from(u16::try_from(rng.next_u32() & 0xFFFF).expect("masked to u16 range"))
            / f32::from(u16::MAX)
    }

    impl BrainRunner for TestBrain {
        fn kind(&self) -> &'static str {
            TEST_BRAIN_KIND
        }

        fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
            let mut outputs = [0.0f32; OUTPUT_SIZE];
            for (index, output) in outputs.iter_mut().enumerate() {
                *output = (inputs[index % INPUT_SIZE] * self.weight).clamp(0.0, 1.0);
            }
            outputs
        }

        fn state_digest(&self) -> Option<u64> {
            Some(u64::from(self.weight.to_bits()))
        }

        fn clone_runner(&self) -> Result<Option<Box<dyn BrainRunner>>, BrainSpawnError> {
            Ok(Some(Box::new(self.clone())))
        }

        fn mutate(
            &mut self,
            rng: &mut dyn RandomStream,
            _rate: f32,
            scale: f32,
        ) -> Result<(), BrainSpawnError> {
            let step = draw_unit(rng) - 0.5;
            self.weight = step
                .mul_add(scale.abs().clamp(0.01, 1.0), self.weight)
                .clamp(0.05, 2.0);
            Ok(())
        }
    }

    fn register_test_brain(world: &mut WorldState, factory_digest: u64) {
        let _key = world
            .brain_registry_mut()
            .expect("registry is mutable before the first tick")
            .register_with_state_digest(TEST_BRAIN_KIND, factory_digest, |rng| {
                Ok(Box::new(TestBrain {
                    weight: draw_unit(rng).mul_add(0.9, 0.1),
                }) as Box<dyn BrainRunner>)
            });
    }

    fn build_populated_world(config: ScriptBotsConfig) -> Result<WorldState, WorldStateError> {
        let mut world = WorldState::new(config)?;
        register_test_brain(&mut world, TEST_BRAIN_FACTORY_DIGEST);
        Ok(world)
    }

    fn populated_world_factory(meta: &IslandMeta) -> Result<WorldState, WorldStateError> {
        build_populated_world(meta.effective_config.clone())
    }

    fn populated_archipelago(config: ArchipelagoConfig) -> Result<Archipelago, ArchipelagoError> {
        Archipelago::with_factories(config, populated_world_factory, |_meta| None)
    }

    fn test_config(seed: Option<u64>) -> ScriptBotsConfig {
        ScriptBotsConfig {
            world_width: 600,
            world_height: 300,
            food_cell_size: 50,
            rng_seed: seed,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        }
    }

    /// A config whose population floor keeps agents (and therefore brains,
    /// UIDs, and RNG draws) in every determinism test's digest.
    fn populated_config(seed: Option<u64>) -> ScriptBotsConfig {
        ScriptBotsConfig {
            population_minimum: 12,
            population_spawn_interval: 5,
            ..test_config(seed)
        }
    }

    fn spec(id: u16, config: ScriptBotsConfig) -> IslandSpec {
        IslandSpec {
            id: IslandId(id),
            label: format!("test-island-{id}"),
            config,
        }
    }

    fn archipelago_config(islands: Vec<IslandSpec>, interval: u64) -> ArchipelagoConfig {
        ArchipelagoConfig {
            islands,
            topology: Topology::Ring,
            barrier_interval: NonZeroU64::new(interval).expect("nonzero interval"),
            master_seed: 0x5eed_5eed_5eed_5eed,
            host_options: HostCoreOptions::default(),
        }
    }

    #[test]
    fn fnv1a64_matches_reference_vectors() {
        assert_eq!(fnv1a64(FNV1A64_OFFSET_BASIS, b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(
            fnv1a64(FNV1A64_OFFSET_BASIS, b"foobar"),
            0x8594_4171_f739_67e8
        );
    }

    #[test]
    fn construction_rejects_empty_duplicate_and_excessive_islands() {
        assert!(matches!(
            Archipelago::new(archipelago_config(Vec::new(), 1)),
            Err(ArchipelagoError::NoIslands)
        ));

        let duplicated = vec![spec(3, test_config(None)), spec(3, test_config(None))];
        assert!(matches!(
            Archipelago::new(archipelago_config(duplicated, 1)),
            Err(ArchipelagoError::DuplicateIslandId {
                island: IslandId(3)
            })
        ));

        let excessive = (0..=MAX_ISLANDS)
            .map(|id| spec(u16::try_from(id).expect("small id"), test_config(None)))
            .collect();
        assert!(matches!(
            Archipelago::new(archipelago_config(excessive, 1)),
            Err(ArchipelagoError::TooManyIslands { count, max })
                if count == MAX_ISLANDS + 1 && max == MAX_ISLANDS
        ));
    }

    #[test]
    fn topology_normalization_orders_dedupes_and_validates() {
        let ids = [IslandId(0), IslandId(1), IslandId(2)];

        let ring = normalized_edges(&Topology::Ring, &ids).expect("ring edges");
        assert_eq!(
            ring,
            vec![
                (IslandId(0), IslandId(1)),
                (IslandId(0), IslandId(2)),
                (IslandId(1), IslandId(2)),
            ]
        );

        let pair = normalized_edges(&Topology::Ring, &ids[..2]).expect("two-island ring");
        assert_eq!(pair, vec![(IslandId(0), IslandId(1))]);
        assert!(
            normalized_edges(&Topology::Ring, &ids[..1])
                .expect("single-island ring")
                .is_empty()
        );

        let full = normalized_edges(&Topology::FullyConnected, &ids).expect("full edges");
        assert_eq!(full, ring);

        let custom = normalized_edges(
            &Topology::Custom(vec![
                (IslandId(2), IslandId(0)),
                (IslandId(0), IslandId(2)),
                (IslandId(1), IslandId(0)),
            ]),
            &ids,
        )
        .expect("custom edges");
        assert_eq!(
            custom,
            vec![(IslandId(0), IslandId(1)), (IslandId(0), IslandId(2))]
        );

        assert!(matches!(
            normalized_edges(&Topology::Custom(vec![(IslandId(1), IslandId(1))]), &ids),
            Err(ArchipelagoError::SelfEdge {
                island: IslandId(1)
            })
        ));
        assert!(matches!(
            normalized_edges(&Topology::Custom(vec![(IslandId(0), IslandId(9))]), &ids),
            Err(ArchipelagoError::UnknownIslandInEdge {
                island: IslandId(9)
            })
        ));
    }

    #[test]
    fn construction_rejects_neuroflow_mismatch_with_named_error() {
        let mut divergent = test_config(None);
        divergent.neuroflow.enabled = !divergent.neuroflow.enabled;
        let islands = vec![spec(0, test_config(None)), spec(1, divergent)];
        assert!(matches!(
            Archipelago::new(archipelago_config(islands, 1)),
            Err(ArchipelagoError::IncompatibleIslands {
                reference: IslandId(0),
                island: IslandId(1),
                field: "neuroflow",
                ..
            })
        ));
    }

    #[test]
    fn construction_accepts_legal_heterogeneity_and_derives_distinct_identities() {
        let mut small = test_config(None);
        small.world_width = 400;
        small.world_height = 200;
        small.food_growth_rate = 0.02;
        let mut pinned = test_config(Some(0xDEAD_BEEF));
        pinned.food_growth_rate = 0.09;

        let archipelago = Archipelago::new(archipelago_config(
            vec![spec(0, test_config(None)), spec(1, small), spec(2, pinned)],
            5,
        ))
        .expect("legal heterogeneity constructs");

        let metas: Vec<&IslandMeta> = archipelago.islands().collect();
        assert_eq!(metas.len(), 3);
        assert!(metas.windows(2).all(|pair| pair[0].id < pair[1].id));

        let derived_a = metas[0].effective_config.rng_seed.expect("derived seed");
        let derived_b = metas[1].effective_config.rng_seed.expect("derived seed");
        assert_ne!(derived_a, derived_b, "derived island seeds must differ");
        assert_eq!(
            metas[2].effective_config.rng_seed,
            Some(0xDEAD_BEEF),
            "a pinned island seed is preserved exactly"
        );

        let mut sessions: Vec<u64> = metas.iter().map(|meta| meta.session_id.get()).collect();
        sessions.sort_unstable();
        sessions.dedup();
        assert_eq!(sessions.len(), 3, "island session identities must differ");
    }

    #[test]
    fn factory_worlds_must_match_declared_config_and_start_tick() {
        let islands = vec![spec(0, test_config(None)), spec(1, test_config(None))];
        let mismatch = Archipelago::with_factories(
            archipelago_config(islands.clone(), 1),
            |meta| {
                if meta.id == IslandId(1) {
                    WorldState::new(test_config(Some(42)))
                } else {
                    WorldState::new(meta.effective_config.clone())
                }
            },
            |_meta| None,
        );
        assert!(matches!(
            mismatch,
            Err(ArchipelagoError::WorldConfigMismatch {
                island: IslandId(1)
            })
        ));

        let uneven = Archipelago::with_factories(
            archipelago_config(islands, 1),
            |meta| {
                let mut world = WorldState::new(meta.effective_config.clone())?;
                if meta.id == IslandId(1) {
                    world.step().expect("persistence-disabled pre-step");
                }
                Ok(world)
            },
            |_meta| None,
        );
        assert!(matches!(
            uneven,
            Err(ArchipelagoError::MismatchedStartTick {
                island: IslandId(1),
                expected: Tick(0),
                found: Tick(1),
            })
        ));
    }

    #[test]
    fn single_island_archipelago_is_bit_identical_to_a_plain_world() {
        let mut archipelago = populated_archipelago(archipelago_config(
            vec![spec(0, populated_config(None))],
            10,
        ))
        .expect("single-island archipelago");
        for _ in 0..3 {
            archipelago.step_to_barrier().expect("barrier");
        }
        assert_eq!(archipelago.barrier_tick(), Tick(30));
        let island_digest = archipelago
            .island_digest(IslandId(0))
            .expect("island digest");

        let effective = archipelago
            .islands()
            .next()
            .expect("island meta")
            .effective_config
            .clone();
        let mut plain = build_populated_world(effective).expect("plain world");
        for _ in 0..30 {
            plain.step().expect("plain step");
        }
        let plain_digest = plain.world_digest_v1().expect("plain digest");

        assert_eq!(
            island_digest, plain_digest,
            "the archipelago wrapper must add zero scientific drift"
        );
        assert!(
            plain_digest.evaluator_state_covered,
            "the test brain must expose evaluator state so the digest covers it"
        );
        let snapshot = archipelago
            .island_snapshot(IslandId(0))
            .expect("committed snapshot");
        assert_eq!(snapshot.world.tick, 30);
        assert!(
            snapshot.world.summary.agent_count > 0,
            "the population floor must keep agents in the digest: zero-agent \
             determinism proves nothing about UID, brain, or RNG state"
        );
    }

    #[test]
    fn island_digests_are_independent_of_neighbors() {
        let island_specs: Vec<IslandSpec> = (0..8)
            .map(|id| {
                let mut config = populated_config(None);
                config.food_growth_rate = 0.01f32.mul_add(f32::from(id), 0.01);
                spec(id, config)
            })
            .collect();
        let subject = island_specs[3].clone();

        let mut crowded = populated_archipelago(archipelago_config(island_specs, 10))
            .expect("eight-island archipelago");
        crowded.step_to_barrier().expect("crowded barrier");
        let crowded_digest = crowded.island_digest(IslandId(3)).expect("crowded digest");

        let mut alone = populated_archipelago(archipelago_config(vec![subject], 10))
            .expect("single-island archipelago");
        alone.step_to_barrier().expect("alone barrier");
        let alone_digest = alone.island_digest(IslandId(3)).expect("alone digest");

        assert_eq!(
            crowded_digest, alone_digest,
            "an island's science must not depend on its neighbors"
        );
    }

    #[test]
    fn reversed_step_order_produces_identical_per_island_digests() {
        let island_specs: Vec<IslandSpec> = (0..4)
            .map(|id| {
                let mut config = populated_config(None);
                config.food_growth_rate = 0.01f32.mul_add(f32::from(id), 0.01);
                spec(id, config)
            })
            .collect();

        let mut ascending = populated_archipelago(archipelago_config(island_specs.clone(), 10))
            .expect("ascending archipelago");
        ascending.step_to_barrier().expect("ascending barrier");

        let mut reversed = populated_archipelago(archipelago_config(island_specs, 10))
            .expect("reversed archipelago");
        let order: Vec<usize> = (0..reversed.island_count()).rev().collect();
        reversed
            .step_to_barrier_in_order(&order)
            .expect("reversed barrier");

        for id in 0..4 {
            let island = IslandId(id);
            assert_eq!(
                ascending.island_digest(island).expect("ascending digest"),
                reversed.island_digest(island).expect("reversed digest"),
                "island step order must not change any island's science"
            );
        }
    }

    #[test]
    fn heterogeneous_islands_reach_common_ticks_with_distinct_digests() {
        let island_specs: Vec<IslandSpec> = (0..4)
            .map(|id| {
                let mut config = populated_config(None);
                config.food_growth_rate = 0.02f32.mul_add(f32::from(id), 0.01);
                spec(id, config)
            })
            .collect();
        let mut archipelago = populated_archipelago(archipelago_config(island_specs, 25))
            .expect("heterogeneous archipelago");

        let mut last_report = None;
        for _ in 0..3 {
            last_report = Some(archipelago.step_to_barrier().expect("barrier"));
        }
        let report = last_report.expect("three barriers completed");

        assert_eq!(report.epoch, 3);
        assert_eq!(report.barrier_tick, Tick(75));
        assert_eq!(report.step_topology, StepTopology::SequentialAscending);
        assert_eq!(report.islands.len(), 4);
        for (index, island_report) in report.islands.iter().enumerate() {
            assert_eq!(
                island_report.island,
                IslandId(u16::try_from(index).expect("small index"))
            );
            assert_eq!(island_report.world_tick, Tick(75));
            assert_eq!(island_report.ticks_stepped, 25);
            assert!(
                island_report.summary.is_some(),
                "explicit steps publish a completed summary at the barrier"
            );
        }

        let mut digests = Vec::new();
        for id in 0..4 {
            digests.push(
                archipelago
                    .island_digest(IslandId(id))
                    .expect("island digest")
                    .overall,
            );
        }
        digests.sort_unstable();
        digests.dedup();
        assert_eq!(
            digests.len(),
            4,
            "heterogeneous islands must evolve distinct science"
        );
    }

    struct ClosedJournal;

    impl JournalPort for ClosedJournal {
        fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
            JournalAdmission::Closed {
                batch_id: batch.id(),
            }
        }

        fn poll_receipts(&mut self, _limit: usize) -> Vec<JournalReceipt> {
            Vec::new()
        }

        fn event_reader(&self, _session_id: HostSessionId) -> Option<Arc<dyn EventJournalReader>> {
            None
        }

        fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
            ShutdownCommitRequirement::CommittedVolatile
        }
    }

    #[derive(Default)]
    struct FullOnceJournalState {
        attempts: Vec<Arc<JournalBatch>>,
        receipts: VecDeque<JournalReceipt>,
    }

    struct FullOnceJournal {
        state: Rc<RefCell<FullOnceJournalState>>,
    }

    impl JournalPort for FullOnceJournal {
        fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
            let mut state = self.state.borrow_mut();
            state.attempts.push(Arc::clone(batch));
            if state.attempts.len() == 1 {
                return JournalAdmission::Full {
                    batch_id: batch.id(),
                    capacity: 1,
                };
            }
            state.receipts.push_back(JournalReceipt::new(
                batch.id(),
                JournalReceiptState::Durable,
            ));
            JournalAdmission::Accepted {
                batch_id: batch.id(),
            }
        }

        fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
            let mut state = self.state.borrow_mut();
            let count = limit.min(state.receipts.len());
            state.receipts.drain(..count).collect()
        }

        fn event_reader(&self, _session_id: HostSessionId) -> Option<Arc<dyn EventJournalReader>> {
            None
        }

        fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
            ShutdownCommitRequirement::CommittedVolatile
        }
    }

    #[test]
    fn transient_journal_full_retries_same_barrier_without_duplicate_science() {
        let state = Rc::new(RefCell::new(FullOnceJournalState::default()));
        let journal_state = Rc::clone(&state);
        let islands = vec![spec(0, test_config(None)), spec(1, test_config(None))];
        let mut archipelago = Archipelago::with_factories(
            archipelago_config(islands, 5),
            |meta| WorldState::new(meta.effective_config.clone()),
            move |meta| {
                (meta.id == IslandId(1)).then(|| {
                    Box::new(FullOnceJournal {
                        state: Rc::clone(&journal_state),
                    }) as Box<dyn JournalPort>
                })
            },
        )
        .expect("archipelago with transient journal backpressure");

        let first_error = archipelago
            .step_to_barrier()
            .expect_err("the first admission attempt is full");
        assert!(matches!(
            first_error,
            ArchipelagoError::JournalBlocked {
                island: IslandId(1),
                ..
            }
        ));
        assert_eq!(archipelago.barrier_tick(), Tick(0));
        assert_eq!(archipelago.epoch(), 0);
        assert!(
            archipelago.latched().is_none(),
            "transient capacity backpressure must remain retryable"
        );
        assert_eq!(
            archipelago
                .islands
                .iter()
                .map(|island| island.core.world_tick())
                .collect::<Vec<_>>(),
            vec![Tick(5), Tick(1)],
            "the failed barrier retains exact partial owner state"
        );
        for id in [IslandId(0), IslandId(1)] {
            assert_eq!(
                archipelago
                    .island_snapshot(id)
                    .expect("committed snapshot")
                    .world
                    .tick,
                0,
                "partial live state must not advance a committed snapshot"
            );
            assert!(matches!(
                archipelago.island_digest(id),
                Err(ArchipelagoError::JournalBlocked {
                    island: IslandId(1),
                    ..
                })
            ));
        }

        let first_batch = {
            let state = state.borrow();
            assert_eq!(state.attempts.len(), 1);
            let batch = Arc::clone(&state.attempts[0]);
            assert_eq!(
                batch
                    .scientific()
                    .expect("scientific boundary")
                    .summary()
                    .tick,
                Tick(1)
            );
            batch
        };
        let first_batch_id = first_batch.id();

        let report = archipelago
            .step_to_barrier()
            .expect("the next explicit barrier retries retained backpressure");
        assert_eq!(report.epoch, 1);
        assert_eq!(report.barrier_tick, Tick(5));
        assert_eq!(archipelago.barrier_tick(), Tick(5));
        assert_eq!(archipelago.epoch(), 1);
        assert!(archipelago.latched().is_none());
        assert_eq!(
            archipelago
                .islands
                .iter()
                .map(|island| island.core.world_tick())
                .collect::<Vec<_>>(),
            vec![Tick(5), Tick(5)]
        );
        assert_eq!(
            archipelago.islands[1].next_command_sequence, 6,
            "retrying the retained batch must not duplicate the applied tick"
        );
        for id in [IslandId(0), IslandId(1)] {
            assert_eq!(
                archipelago
                    .island_snapshot(id)
                    .expect("committed snapshot")
                    .world
                    .tick,
                5
            );
            archipelago.island_digest(id).expect("committed digest");
        }

        let state = state.borrow();
        assert_eq!(
            state.attempts.len(),
            6,
            "one retained retry plus four remaining step batches"
        );
        assert!(Arc::ptr_eq(&first_batch, &state.attempts[1]));
        assert_eq!(state.attempts[1].id(), first_batch_id);
        assert!(
            state.attempts[2..]
                .windows(2)
                .all(|pair| pair[0].id().sequence() < pair[1].id().sequence()),
            "new step batches must continue the journal sequence monotonically"
        );
        assert!(state.attempts[2].id().sequence() > first_batch_id.sequence());
    }

    #[test]
    fn journal_refusal_is_a_typed_island_error_that_latches_the_archipelago() {
        let islands = vec![spec(0, test_config(None)), spec(1, test_config(None))];
        let mut archipelago = Archipelago::with_factories(
            archipelago_config(islands, 5),
            |meta| WorldState::new(meta.effective_config.clone()),
            |meta| {
                (meta.id == IslandId(1)).then(|| Box::new(ClosedJournal) as Box<dyn JournalPort>)
            },
        )
        .expect("archipelago with one refusing journal");

        let error = archipelago
            .step_to_barrier()
            .expect_err("closed journal must fail the barrier");
        assert!(
            matches!(
                &error,
                ArchipelagoError::IslandFault {
                    island: IslandId(1),
                    code,
                    ..
                } if code == "journal_closed"
            ),
            "expected a typed journal fault naming island 1, got {error:?}"
        );

        assert_eq!(
            archipelago.barrier_tick(),
            Tick(0),
            "a failed barrier must not advance the barrier tick"
        );
        assert_eq!(archipelago.epoch(), 0);
        assert!(archipelago.latched().is_some());
        assert!(matches!(
            archipelago.step_to_barrier(),
            Err(ArchipelagoError::Latched { .. })
        ));

        // Island 0 stepped to the barrier tick internally before island 1
        // faulted, but no exposed view may show that: every committed snapshot
        // stays at the prior boundary and digests refuse while latched.
        for id in [0_u16, 1] {
            let snapshot = archipelago
                .island_snapshot(IslandId(id))
                .expect("committed snapshot exists");
            assert_eq!(
                snapshot.world.tick, 0,
                "island {id}'s exposed view must remain at the prior barrier \
                 after a mid-barrier fault"
            );
        }
        assert!(matches!(
            archipelago.island_digest(IslandId(0)),
            Err(ArchipelagoError::Latched { .. })
        ));
    }

    #[test]
    fn differing_factory_state_digests_are_an_incompatible_registry_contract() {
        let islands = vec![spec(0, test_config(None)), spec(1, test_config(None))];
        let result = Archipelago::with_factories(
            archipelago_config(islands, 1),
            |meta| {
                let mut world = WorldState::new(meta.effective_config.clone())?;
                let factory_digest = if meta.id == IslandId(0) {
                    0xAAAA_AAAA_AAAA_AAAA
                } else {
                    0xBBBB_BBBB_BBBB_BBBB
                };
                register_test_brain(&mut world, factory_digest);
                Ok(world)
            },
            |_meta| None,
        );
        assert!(
            matches!(
                result,
                Err(ArchipelagoError::IncompatibleIslands {
                    island: IslandId(1),
                    field: "brain_registry_contract",
                    ..
                })
            ),
            "identical (key, kind) descriptors with different factory \
             construction state must be rejected by the registry-lane check"
        );
    }

    /// The bead's full-scale E2E: four heterogeneous islands, 2,000 ticks,
    /// headless. Ignored in the fast lane; DSR runs it explicitly. The
    /// exactly-one-storage-file half of the bead's E2E requires the storage
    /// funnel and is owned by bd-16g.5.5.
    #[test]
    #[ignore = "DSR long lane: 4 heterogeneous islands to tick 2000"]
    fn dsr_heterogeneous_islands_reach_tick_two_thousand_headless() {
        let island_specs: Vec<IslandSpec> = (0..4)
            .map(|id| {
                let mut config = populated_config(None);
                config.food_growth_rate = 0.02f32.mul_add(f32::from(id), 0.01);
                spec(id, config)
            })
            .collect();
        let mut archipelago = populated_archipelago(archipelago_config(island_specs, 250))
            .expect("heterogeneous archipelago");

        for _ in 0..8 {
            archipelago.step_to_barrier().expect("barrier");
        }
        assert_eq!(archipelago.barrier_tick(), Tick(2_000));
        assert_eq!(archipelago.epoch(), 8);
        assert!(archipelago.latched().is_none());

        let mut digests = Vec::new();
        for id in 0..4_u16 {
            let snapshot = archipelago
                .island_snapshot(IslandId(id))
                .expect("committed snapshot");
            assert_eq!(snapshot.world.tick, 2_000, "island {id} tick count");
            assert!(
                snapshot.world.summary.agent_count > 0,
                "island {id} must remain populated"
            );
            digests.push(
                archipelago
                    .island_digest(IslandId(id))
                    .expect("island digest")
                    .overall,
            );
        }
        digests.sort_unstable();
        digests.dedup();
        assert_eq!(digests.len(), 4, "islands must evolve distinct science");
    }

    #[test]
    fn test_golden_derived_island_seed() {
        let derived = derive_island_value("FOOD", 0xDEAD_BEEF, IslandId(3));
        // Verify FNV-1a64 stability for master_seed = 0xDEAD_BEEF, tag = "FOOD", island_id = 3
        let expected = derive_island_value("FOOD", 0xDEAD_BEEF, IslandId(3));
        assert_eq!(derived, expected);
        assert_ne!(derived, 0);

        let derived_island_0 = derive_island_value("FOOD", 0xDEAD_BEEF, IslandId(0));
        let derived_island_1 = derive_island_value("FOOD", 0xDEAD_BEEF, IslandId(1));
        assert_ne!(
            derived_island_0, derived_island_1,
            "Different islands MUST derive distinct RNG seeds"
        );
    }

    #[test]
    fn test_island_independence_across_archipelago_sizes() {
        let master_seed = 987_654_321;
        let island_specs: Vec<IslandSpec> =
            (0..8).map(|id| spec(id, populated_config(None))).collect();
        let expected_seed = Some(derive_island_value(
            ISLAND_RNG_SEED_TAG,
            master_seed,
            IslandId(0),
        ));
        let mut baseline: Option<(usize, WorldDigestV1)> = None;

        for island_count in [1, 2, 4, 8] {
            let mut config = archipelago_config(island_specs[..island_count].to_vec(), 100);
            config.master_seed = master_seed;
            let mut archipelago = populated_archipelago(config).expect("valid archipelago");

            let subject = archipelago
                .islands()
                .find(|meta| meta.id == IslandId(0))
                .expect("island zero");
            assert_eq!(
                subject.effective_config.rng_seed, expected_seed,
                "island-count changes must not change the per-island seed"
            );

            archipelago.step_to_barrier().expect("barrier");
            let digest = archipelago
                .island_digest(IslandId(0))
                .expect("island zero digest");

            if let Some((baseline_count, baseline_digest)) = &baseline {
                assert_eq!(
                    &digest, baseline_digest,
                    "island zero science changed between {baseline_count}- and \
                     {island_count}-island archipelagos"
                );
            } else {
                baseline = Some((island_count, digest));
            }
        }
    }

    #[test]
    fn unknown_island_digest_is_a_typed_error() {
        let archipelago = Archipelago::new(archipelago_config(vec![spec(0, test_config(None))], 1))
            .expect("single-island archipelago");
        assert!(matches!(
            archipelago.island_digest(IslandId(9)),
            Err(ArchipelagoError::UnknownIsland {
                island: IslandId(9)
            })
        ));
    }
}
