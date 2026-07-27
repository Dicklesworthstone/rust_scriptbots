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
    migrator::{
        CandidateAgent, EmigrantSelectionRule, MigrationConfig, MigrationError, MigrationTopology,
        select_emigrants,
    },
};
use scriptbots_core::{
    AgentUid, CharacterizationError, MigratingAgent, ScriptBotsConfig, Tick, TickSummary,
    WorldDigestV1, WorldState, WorldStateError, rng_domains::OrganismId,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
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
/// v2 (bd-cxcf): [`IslandId`] widened from `u16` to the canonical `u32`, so
/// `derive_island_value` now folds FOUR bytes of island id instead of two and every
/// island seed changes.
///
/// Per the bd-2z0.5.6 digest policy the change goes INTO the derivation and the version
/// tag moves with it, rather than being smuggled in silently: runs seeded under v1 are
/// NOT COMPARABLE to runs seeded under v2, which is a different statement from their
/// digests disagreeing. The tag is the only thing that lets a reader tell those apart.
const ISLAND_RNG_SEED_TAG: &str = "scriptbots.archipelago.island-rng-seed.v2";

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
///
/// Re-exported from core (bd-cxcf) so the archipelago, the migrator and the RNG
/// derivation all speak one type. This was a local `u16` newtype; widening it to the
/// canonical `u32` changes how many bytes the seed derivation hashes, which is why
/// [`ISLAND_RNG_SEED_TAG`] moved to v2.
pub use scriptbots_core::rng_domains::IslandId;

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
    /// Cross-island migration policy, or `None` for isolated islands
    /// (bd-16g.5.2).
    pub migration: Option<ArchipelagoMigration>,
}

/// Cross-island migration policy for an [`Archipelago`] (bd-16g.5.2).
///
/// # There is deliberately no topology field
///
/// [`MigrationConfig`] carries one, because the migrator is usable standalone.
/// Here it would be a second, independent description of which islands are
/// connected — and the moment two descriptions exist they can disagree. They
/// would disagree immediately, too: [`Topology::Ring`] normalizes to UNDIRECTED
/// edges, so on three islands it is the complete graph, while
/// [`MigrationTopology::Ring`] is the DIRECTED cycle `0 -> 1 -> 2 -> 0`. A
/// caller who wrote "Ring" in both places would get one of them and no
/// diagnostic.
///
/// The archipelago therefore DERIVES the migration graph from its own
/// [`ArchipelagoConfig::topology`]: every undirected edge `{a, b}` becomes both
/// `a -> b` and `b -> a`, since connectivity is symmetric but a move is not.
/// One description cannot contradict itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchipelagoMigration {
    /// Ticks between migration barriers.
    ///
    /// Must be a nonzero multiple of [`ArchipelagoConfig::barrier_interval`].
    /// Migration is only physically possible at a barrier — that is the only
    /// moment every island is at the same tick — so an interval that is not a
    /// multiple of the barrier would either never fire or fire at a tick the
    /// caller did not ask for. Construction rejects it rather than silently
    /// rounding, because a migration cadence that quietly differs from the
    /// configured one changes the science and nothing would say so.
    pub interval_ticks: u64,
    /// Emigrants selected per directed edge per barrier.
    pub emigrants_per_edge: usize,
    /// Rule that decides who leaves.
    pub selection_rule: EmigrantSelectionRule,
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
/// `HostCore` is `!Send`, so islands step sequentially on the owning thread while each island's
/// tick pipeline parallelizes internally. Reports always record which policy ran.
///
/// OUTER-ISLAND PARALLELISM IS STRUCTURALLY EXCLUDED, NOT MERELY UNIMPLEMENTED (bd-5tyo). This
/// doc used to say scaling work "may add topologies", which reads as pending — someone will get
/// to it. That is not the situation. An island IS a [`HostCore`], `HostCore` is `!Send` by
/// deliberate design because it owns same-thread command ports, and stepping islands on separate
/// threads therefore requires changing the host ownership model. That is bd-pcfj's decision to
/// make, not a gap in this module.
///
/// So there is ONE variant because one is correct under the current ownership model, and the
/// enum stays `#[non_exhaustive]` for the day that model changes rather than because more
/// variants are owed. Scaling work (bd-16g.5.4) should measure the INNER parallelism each island
/// already has; measuring it against an outer topology the ownership model forbids would be
/// comparing against something that cannot exist.
///
/// `outer_island_parallelism_stays_excluded_while_host_core_is_not_send` fails to compile if
/// `HostCore` ever becomes `Send`, so this justification expires visibly instead of silently
/// outliving the constraint it rests on.
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
    /// Migration outcome, when this barrier ran one (bd-16g.5.2).
    pub migration: Option<MigrationBarrierReport>,
}

/// One organism that actually moved during a barrier (bd-16g.5.2).
///
/// THIS IS THE WITNESS, and it exists because population state cannot be one.
/// Each island mints `AgentUid` from its own allocator, so an arrival
/// necessarily takes a fresh local UID — meaning a move is indistinguishable
/// from a death plus a birth when read from two censuses. Only a record that
/// names BOTH ends of the journey can say a specific organism left one island
/// and reached another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AppliedMigration {
    /// Island the organism departed.
    pub from: OrganismId,
    /// Island and freshly minted local identity it arrived under.
    pub to: OrganismId,
    /// Rule that selected it.
    pub rule: EmigrantSelectionRule,
    /// Rank within its island under that rule (0 is the strongest match).
    pub rank: usize,
    /// The energy, age, or speed value that got it selected — the field that
    /// makes "why did THIS agent leave?" answerable from the log alone.
    pub key_value: f64,
}

/// Outcome of one barrier's migration phase (bd-16g.5.2).
#[derive(Debug, Clone, PartialEq)]
pub struct MigrationBarrierReport {
    /// Common tick every island had reached when migration ran.
    pub barrier_tick: Tick,
    /// Selection rule in force.
    pub rule: EmigrantSelectionRule,
    /// Every organism that moved, in canonical `(to, from, uid)` order.
    pub moves: Vec<AppliedMigration>,
    /// Per-island population immediately before any move was applied.
    pub pop_before: BTreeMap<IslandId, u32>,
    /// Per-island population immediately after every move was applied.
    pub pop_after: BTreeMap<IslandId, u32>,
    /// Every organism's identity immediately before any move was applied.
    ///
    /// THE EVIDENCE, CARRIED RATHER THAN ASSERTED AWAY. Together with
    /// [`Self::census_after`] and [`Self::moves`] this is exactly the input
    /// [`verify_migration`] consumed, so a consumer can re-derive the
    /// conservation verdict instead of trusting that it was reached. A report
    /// that only said `conserved: true` would be a claim; this is a receipt.
    ///
    /// Note these bracket the MIGRATION PHASE, not the barrier. The barrier also
    /// steps every island, and births during that stepping are why a pre-barrier
    /// census is the wrong thing to compare against — an organism born this
    /// barrier can emigrate in it.
    pub census_before: BTreeSet<OrganismId>,
    /// Every organism's identity immediately after every move was applied.
    pub census_after: BTreeSet<OrganismId>,
}

impl MigrationBarrierReport {
    /// Total population across the archipelago before migration.
    #[must_use]
    pub fn total_before(&self) -> u32 {
        self.pop_before.values().sum()
    }

    /// Total population across the archipelago after migration.
    #[must_use]
    pub fn total_after(&self) -> u32 {
        self.pop_after.values().sum()
    }
}

/// An exact way a migration barrier can have corrupted the population
/// (bd-tfso).
///
/// Each variant names a distinct corruption so a failure says WHICH invariant
/// broke, not merely that one did. A single "migration failed" would be useless
/// here: the whole class of bug this guards against is silent, so the payload is
/// the investigation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MigrationViolation {
    /// The population changed across a phase that contains no ticks.
    #[error("population changed across migration: {before} organisms before, {after} after")]
    NotConserved {
        /// Organisms present before any move.
        before: usize,
        /// Organisms present after every move.
        after: usize,
    },
    /// Moves are not in ascending `(to island, from island, from uid)` order.
    #[error("move {index} is out of canonical (to, from, uid) order")]
    OutOfCanonicalOrder {
        /// Index of the first out-of-order move.
        index: usize,
    },
    /// A move claims to have taken an organism that was not there.
    #[error("move claims departure of {organism}, which was not present before migration")]
    DepartureNotPresent {
        /// The organism the move named.
        organism: OrganismId,
    },
    /// A move claims an arrival identity that already existed.
    #[error("arrival {organism} collides with an organism that already existed")]
    ArrivalCollides {
        /// The colliding identity.
        organism: OrganismId,
    },
    /// One organism emigrated more than once in a single barrier.
    #[error("{organism} emigrated twice in one barrier")]
    DepartedTwice {
        /// The doubly-departed organism.
        organism: OrganismId,
    },
    /// Two moves claim the same arrival identity.
    #[error("{organism} arrived twice in one barrier")]
    ArrivedTwice {
        /// The doubly-arrived identity.
        organism: OrganismId,
    },
    /// An organism exists afterwards that no move and no prior state explains.
    ///
    /// THIS IS THE DUPLICATION SIGNAL. An emigration that failed to remove the
    /// organism from its source leaves it on both islands, and the source copy
    /// is exactly an unexplained presence.
    #[error("{organism} exists after migration but nothing explains it")]
    UnexplainedPresence {
        /// The unexplained organism.
        organism: OrganismId,
    },
    /// An organism that should exist afterwards does not.
    ///
    /// THIS IS THE LOSS SIGNAL. A dropped emigrant — collected from its source
    /// and never delivered — is exactly an unexplained absence.
    #[error("{organism} should exist after migration but does not")]
    UnexplainedAbsence {
        /// The missing organism.
        organism: OrganismId,
    },
}

/// Prove one barrier's migration neither lost, duplicated, nor reordered an
/// organism (bd-tfso).
///
/// # Why this is a function and not a pile of assertions
///
/// A conservation check that lives only inside a test proves the test is
/// correct. This runs in production at every migration barrier AND is what the
/// mutation tests attack, so the thing being verified and the thing being
/// trusted are the same code.
///
/// # The identity it enforces
///
/// No scientific tick runs inside a migration phase, so births and deaths are
/// structurally zero across it and the population is not merely conserved in
/// COUNT — the exact set of organisms is determined:
///
/// ```text
/// after == (before \ departures) ∪ arrivals
/// ```
///
/// Set equality is strictly stronger than comparing totals, and the difference
/// is the whole point. An organism duplicated on one island and lost on another
/// leaves the total unchanged; it cannot leave this equation unchanged, because
/// the surviving copy is an identity nothing explains and the lost one is an
/// identity that should be there and is not.
///
/// # What it cannot catch, stated plainly
///
/// It compares identities, so it cannot see a corruption that preserves every
/// identity while damaging what the organism CARRIES — a brain that arrived
/// unbound, say. That is covered where it belongs, by `scriptbots-core`'s
/// transfer tests, not here.
///
/// # Errors
///
/// Returns the first [`MigrationViolation`] found, checked cheapest-first so the
/// reported violation is the most specific one available.
pub fn verify_migration(
    before: &BTreeSet<OrganismId>,
    after: &BTreeSet<OrganismId>,
    moves: &[AppliedMigration],
) -> Result<(), MigrationViolation> {
    for (index, pair) in moves.windows(2).enumerate() {
        let key =
            |applied: &AppliedMigration| (applied.to.island, applied.from.island, applied.from.uid);
        if key(&pair[0]) > key(&pair[1]) {
            return Err(MigrationViolation::OutOfCanonicalOrder { index: index + 1 });
        }
    }

    let mut departures = BTreeSet::new();
    let mut arrivals = BTreeSet::new();
    for applied in moves {
        if !before.contains(&applied.from) {
            return Err(MigrationViolation::DepartureNotPresent {
                organism: applied.from,
            });
        }
        if before.contains(&applied.to) {
            return Err(MigrationViolation::ArrivalCollides {
                organism: applied.to,
            });
        }
        if !departures.insert(applied.from) {
            return Err(MigrationViolation::DepartedTwice {
                organism: applied.from,
            });
        }
        if !arrivals.insert(applied.to) {
            return Err(MigrationViolation::ArrivedTwice {
                organism: applied.to,
            });
        }
    }

    if before.len() != after.len() {
        return Err(MigrationViolation::NotConserved {
            before: before.len(),
            after: after.len(),
        });
    }

    let expected: BTreeSet<OrganismId> = before
        .difference(&departures)
        .copied()
        .chain(arrivals.iter().copied())
        .collect();
    if let Some(&organism) = after.difference(&expected).next() {
        return Err(MigrationViolation::UnexplainedPresence { organism });
    }
    if let Some(&organism) = expected.difference(after).next() {
        return Err(MigrationViolation::UnexplainedAbsence { organism });
    }
    Ok(())
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
    /// The migration interval is not a nonzero multiple of the barrier interval.
    #[error(
        "migration interval {interval_ticks} must be a nonzero multiple of \
         barrier interval {barrier_interval}"
    )]
    MigrationIntervalNotBarrierAligned {
        /// Configured migration interval.
        interval_ticks: u64,
        /// Configured barrier interval.
        barrier_interval: u64,
    },
    /// A migration barrier corrupted the population.
    ///
    /// Distinct from [`Self::MigrationNotConserved`], which only compares
    /// totals: this carries the exact identity-level violation, including the
    /// duplicate-and-lose case that leaves the total untouched.
    #[error("migration corrupted the population at tick {tick}: {violation}")]
    MigrationCorrupted {
        /// Barrier tick the corruption was detected at.
        tick: u64,
        /// Exact identity-level violation.
        #[source]
        violation: MigrationViolation,
    },
    /// Emigrant selection failed.
    #[error("migration selection failed: {source}")]
    MigrationSelection {
        /// Exact selection failure.
        #[source]
        source: MigrationError,
    },
    /// Migration did not conserve population.
    ///
    /// No scientific tick runs inside the migration phase, so births and deaths
    /// are structurally zero across it and this identity is exact. A mismatch is
    /// therefore never explainable by ordinary population dynamics — it means an
    /// organism was duplicated or dropped.
    #[error(
        "migration did not conserve population: before={before}, after={after}, \
         moves={moves}"
    )]
    MigrationNotConserved {
        /// Total population before any move was applied.
        before: u32,
        /// Total population after every move was applied.
        after: u32,
        /// Number of moves applied.
        moves: usize,
    },
    /// A host applied an emigration but produced no organism to carry.
    #[error("island {island} emigrated agent {agent_uid} but parked no organism")]
    MigrationOrganismMissing {
        /// The source island.
        island: IslandId,
        /// The agent that was supposed to depart.
        agent_uid: u64,
    },
    /// An organism could not be delivered and could not be returned home.
    ///
    /// The one failure this module cannot repair, so it is named exactly rather
    /// than folded into a generic error: a living organism is held by no world
    /// and the run's population has genuinely changed.
    #[error(
        "organism {origin} could not reach {destination} and could not be \
         returned home: {detail}"
    )]
    MigrationOrganismLost {
        /// Where the organism came from.
        origin: OrganismId,
        /// Where it was going.
        destination: IslandId,
        /// Exact failure detail.
        detail: String,
    },
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
    migration: Option<ArchipelagoMigration>,
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
            migration,
        } = config;

        if islands.is_empty() {
            return Err(ArchipelagoError::NoIslands);
        }
        if let Some(migration) = &migration {
            let interval = migration.interval_ticks;
            if interval == 0 || !interval.is_multiple_of(barrier_interval.get()) {
                return Err(ArchipelagoError::MigrationIntervalNotBarrierAligned {
                    interval_ticks: interval,
                    barrier_interval: barrier_interval.get(),
                });
            }
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
            migration,
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

    /// Read one island's world at the current barrier boundary.
    ///
    /// THE SCIENCE READOUT THIS TYPE WAS MISSING. Before this, an archipelago could prove
    /// its islands were deterministic ([`Self::island_digest`]) but not observe what
    /// evolved on any of them -- and both halves of bd-16g.5's science acceptance
    /// (allopatric divergence, cross-island gene flow) are statements about biology, not
    /// about digests.
    ///
    /// Takes `&self`, so it cannot be called while [`Self::step_to_barrier`] holds
    /// `&mut self`. The partially-stepped world stays unobservable by construction, which
    /// is the invariant this module's header commits to.
    ///
    /// Errors identically to [`Self::island_digest`] on a latched archipelago or an
    /// unknown island: a caller must not be able to read science out of a failed run.
    pub fn with_island_world<R>(
        &self,
        island: IslandId,
        read: impl FnOnce(&WorldState) -> R,
    ) -> Result<R, ArchipelagoError> {
        if let Some(detail) = &self.latched {
            return Err(ArchipelagoError::Latched {
                detail: detail.clone(),
            });
        }
        let index = self
            .island_index(island)
            .ok_or(ArchipelagoError::UnknownIsland { island })?;
        Ok(self.islands[index].core.with_world(read))
    }

    /// Every living organism in the archipelago, keyed by its globally unique
    /// scientific identity.
    ///
    /// THE CORRECT PATH, GIVEN A HOME SO NOBODY HAND-ROLLS THE WRONG ONE
    /// (bd-8jlj). [`Self::with_island_world`] hands out one island's
    /// [`WorldState`], and the obvious way to survey the whole archipelago is to
    /// call it per island and union the results. Every lineage and species
    /// structure in `scriptbots-core` is keyed on a BARE [`AgentUid`], and each
    /// island mints UIDs from its own private counter — so that union silently
    /// collapses island 0's agent 1 and island 1's agent 1 into one organism. No
    /// panic, no typed error; just a phylogeny that claims two unrelated
    /// individuals are the same one.
    ///
    /// This returns [`OrganismId`] values instead, so the island travels with
    /// every element and the collapse cannot happen by accident. A caller that
    /// genuinely wants a cross-island gene pool has to discard the island axis
    /// deliberately, which is a decision someone can review.
    ///
    /// The set is barrier-consistent: it reads committed island worlds through
    /// `&self`, so it cannot interleave with [`Self::step_to_barrier`].
    ///
    /// # Errors
    ///
    /// Returns [`ArchipelagoError::Latched`] once an island fault has latched the
    /// archipelago — a census taken then would describe a partially-stepped
    /// epoch — or [`ArchipelagoError::Digest`] if an island's per-agent identity
    /// state cannot be read.
    pub fn organism_census(&self) -> Result<BTreeSet<OrganismId>, ArchipelagoError> {
        if let Some(detail) = &self.latched {
            return Err(ArchipelagoError::Latched {
                detail: detail.clone(),
            });
        }
        let mut census = BTreeSet::new();
        for island in &self.islands {
            let id = island.meta.id;
            let states = island
                .core
                .with_world(WorldState::ordered_agent_rng_counters_v1)
                .map_err(|source| ArchipelagoError::Digest { island: id, source })?;
            for state in states {
                let organism = OrganismId::new(id, state.agent_uid());
                debug_assert!(
                    !census.contains(&organism),
                    "one island cannot hold two agents with the same uid"
                );
                census.insert(organism);
            }
        }
        Ok(census)
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

        // EVERY ISLAND IS NOW AT `target`. That is the precondition the whole
        // migration protocol rests on, and it is satisfied here and nowhere
        // else: migration must not observe an island one tick behind another,
        // or which agents exist to be selected depends on stepping order.
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or(ArchipelagoError::TickOverflow)?;
        let migration = match self.migrate_at_barrier(epoch, target) {
            Ok(report) => report,
            Err(error) => {
                tracing::error!(
                    barrier_tick = target.0,
                    error = %error,
                    "migration failed mid-barrier; archipelago latched"
                );
                self.latched = Some(error.to_string());
                return Err(error);
            }
        };

        // Every island reached the barrier: commit the new views in ascending
        // island-id order. This is the only place exposed snapshots advance.
        for island in &mut self.islands {
            island.committed_snapshot = island.core.latest_snapshot();
        }
        self.epoch = epoch;
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
            migration,
        })
    }

    /// Run this barrier's migration phase, if one is due (bd-16g.5.2).
    ///
    /// # The two-phase discipline, and the bug it makes unrepresentable
    ///
    /// Selection reads a FROZEN pre-barrier census of every island and completes
    /// entirely before any move is applied. A per-edge select-and-apply loop
    /// would let island 1's fittest agent be chosen for the edge to island 0 AND
    /// the edge to island 2 — emigrating twice, i.e. being DUPLICATED — and
    /// would let an immigrant that has just landed be re-selected as an emigrant
    /// in the same barrier. Both become impossible when selection cannot observe
    /// application.
    ///
    /// # Why conservation is exactly checkable here, and only here
    ///
    /// I previously recorded that the accounting identity
    /// `after == before + births - deaths + immigrants - emigrants` was not
    /// closable, because `TickSummary::births`/`deaths` are per-tick and a
    /// barrier spans many ticks. That was measuring across the wrong window. NO
    /// SCIENTIFIC TICK RUNS INSIDE THIS PHASE — every island is already at
    /// `barrier_tick` and only migration commands are applied — so births and
    /// deaths are structurally zero across it and the identity collapses to
    /// `after == before`. The check below is therefore exact rather than
    /// approximate, and a mismatch cannot be explained by population dynamics.
    ///
    /// Population sums are still only half the proof: an agent duplicated on one
    /// island and lost on another leaves the total unchanged. The other half is
    /// uniqueness of `(IslandId, AgentUid)`, which [`Self::organism_census`]
    /// expresses and the tests assert at every barrier.
    fn migrate_at_barrier(
        &mut self,
        epoch: u64,
        barrier_tick: Tick,
    ) -> Result<Option<MigrationBarrierReport>, ArchipelagoError> {
        let Some(policy) = self.migration.clone() else {
            return Ok(None);
        };
        if !barrier_tick.0.is_multiple_of(policy.interval_ticks) {
            return Ok(None);
        }
        // The migration graph is DERIVED from the archipelago's own topology, so
        // there is no second description that could contradict it.
        let config = MigrationConfig {
            interval_ticks: policy.interval_ticks,
            emigrants_per_edge: policy.emigrants_per_edge,
            selection_rule: policy.selection_rule,
            topology: MigrationTopology::Custom(directed_edges(&self.edges)),
            replace: true,
        };

        // PHASE 1 — freeze. Every candidate list is uid-ascending by
        // construction, never slotmap handle order.
        let candidates = self.migration_candidates();
        let pop_before: BTreeMap<IslandId, u32> = candidates
            .iter()
            .map(|(&id, list)| (id, u32::try_from(list.len()).unwrap_or(u32::MAX)))
            .collect();
        let total_before: u32 = pop_before.values().sum();
        // The identity census taken from the SAME frozen snapshot selection
        // reads, so the proof at the end compares like with like.
        let census_before: BTreeSet<OrganismId> = candidates
            .iter()
            .flat_map(|(&island, list)| {
                list.iter()
                    .map(move |candidate| OrganismId::new(island, AgentUid(candidate.uid)))
            })
            .collect();

        // PHASE 2 — select everything before applying anything.
        let plan = select_emigrants(&candidates, &config, self.master_seed, epoch)
            .map_err(|source| ArchipelagoError::MigrationSelection { source })?;

        // PHASE 3 — apply in the plan's canonical (to, from, uid) order.
        let mut moves = Vec::with_capacity(plan.moves.len());
        for record in &plan.moves {
            let from = record.from_island;
            let to = record.to_island;
            let uid = AgentUid(record.agent_uid);
            let from_index = self
                .island_index(from)
                .ok_or(ArchipelagoError::UnknownIsland { island: from })?;
            let to_index = self
                .island_index(to)
                .ok_or(ArchipelagoError::UnknownIsland { island: to })?;

            Self::apply_island_command(
                &mut self.islands[from_index],
                HostCommand::Emigrate { agent_uid: uid },
            )?;
            let migrant = self.islands[from_index]
                .core
                .take_outbound_migrant()
                .ok_or(ArchipelagoError::MigrationOrganismMissing {
                    island: from,
                    agent_uid: record.agent_uid,
                })?;

            let arrival = self.deliver_migrant(to_index, from_index, from, uid, migrant)?;
            tracing::info!(
                barrier_tick = barrier_tick.0,
                from = %from,
                to = %to,
                origin_uid = uid.get(),
                local_uid = arrival.get(),
                rule = ?record.selection_rule,
                rank = record.rank,
                key_value = record.key_value,
                "migrated one organism"
            );
            moves.push(AppliedMigration {
                from: OrganismId::new(from, uid),
                to: OrganismId::new(to, arrival),
                rule: record.selection_rule,
                rank: record.rank,
                key_value: record.key_value,
            });
        }

        let pop_after: BTreeMap<IslandId, u32> = self
            .islands
            .iter()
            .map(|island| {
                (
                    island.meta.id,
                    u32::try_from(island.core.with_world(WorldState::agent_count))
                        .unwrap_or(u32::MAX),
                )
            })
            .collect();
        let total_after: u32 = pop_after.values().sum();
        if total_before != total_after {
            return Err(ArchipelagoError::MigrationNotConserved {
                before: total_before,
                after: total_after,
                moves: moves.len(),
            });
        }

        // THE IDENTITY-LEVEL PROOF, which the count above cannot give. An
        // organism duplicated on one island and lost on another leaves
        // `total_before == total_after` true; it cannot leave this true. Run in
        // production, not only in tests, because the failure it catches is
        // silent and a run that has corrupted its population must stop rather
        // than keep producing plausible science.
        let census_after = self.organism_census()?;
        verify_migration(&census_before, &census_after, &moves).map_err(|violation| {
            ArchipelagoError::MigrationCorrupted {
                tick: barrier_tick.0,
                violation,
            }
        })?;

        for (island, before) in &pop_before {
            let after = pop_after.get(island).copied().unwrap_or(0);
            if *before > 0 && after == 0 {
                tracing::warn!(
                    island = %island,
                    barrier_tick = barrier_tick.0,
                    "island reached zero population by emigration"
                );
            }
        }
        tracing::info!(
            barrier_tick = barrier_tick.0,
            rule = ?config.selection_rule,
            interval = config.interval_ticks,
            total_moves = moves.len(),
            total_before,
            total_after,
            conserved = true,
            "migration barrier complete"
        );

        Ok(Some(MigrationBarrierReport {
            barrier_tick,
            rule: config.selection_rule,
            moves,
            pop_before,
            pop_after,
            census_before,
            census_after,
        }))
    }

    /// Per-island emigration candidates in ascending `AgentUid` order.
    ///
    /// Ordering is the contract, not a convenience. Slotmap handle order is an
    /// implementation detail and handles are reused, so ranking over it would
    /// make the emigrant set depend on allocation history rather than on the
    /// selection rule.
    fn migration_candidates(&self) -> BTreeMap<IslandId, Vec<CandidateAgent>> {
        self.islands
            .iter()
            .map(|island| {
                let list = island.core.with_world(|world| {
                    let mut list: Vec<CandidateAgent> = world
                        .agents()
                        .iter_handles()
                        .filter_map(|id| {
                            let uid = world.agent_uid(id)?;
                            let data = world.agents().snapshot(id)?;
                            let runtime = world.agent_runtime(id)?;
                            let speed = runtime
                                .outputs
                                .iter()
                                .fold(0.0f32, |peak, value| peak.max(value.abs()));
                            Some(CandidateAgent {
                                uid: uid.get(),
                                energy: runtime.energy,
                                health: data.health,
                                age: data.age,
                                speed,
                            })
                        })
                        .collect();
                    list.sort_unstable_by_key(|candidate| candidate.uid);
                    list
                });
                (island.meta.id, list)
            })
            .collect()
    }

    /// Stage and admit one organism at its destination, returning it home if the
    /// destination refuses.
    ///
    /// The recovery path is the point. `HostCore` hands a refused organism back
    /// rather than dropping it, so a destination that cannot admit an arrival
    /// does not cost the run an agent — it costs the barrier, which is loud.
    /// Only when the SOURCE also refuses to take it back is an organism
    /// genuinely lost, and that gets its own named error.
    fn deliver_migrant(
        &mut self,
        to_index: usize,
        from_index: usize,
        from: IslandId,
        uid: AgentUid,
        migrant: MigratingAgent,
    ) -> Result<AgentUid, ArchipelagoError> {
        let to = self.islands[to_index].meta.id;
        if let Err(returned) = self.islands[to_index].core.stage_immigrant(migrant) {
            let detail = "destination already holds a staged organism".to_owned();
            return Err(self.return_migrant_home(from_index, from, uid, to, returned, detail));
        }
        let command = HostCommand::Immigrate {
            origin_island: from,
            origin_uid: uid,
        };
        if let Err(error) = Self::apply_island_command(&mut self.islands[to_index], command) {
            let detail = error.to_string();
            let Some(returned) = self.islands[to_index].core.unstage_immigrant() else {
                // The destination neither admitted nor retained it.
                return Err(ArchipelagoError::MigrationOrganismLost {
                    origin: OrganismId::new(from, uid),
                    destination: to,
                    detail,
                });
            };
            return Err(self.return_migrant_home(from_index, from, uid, to, returned, detail));
        }
        let arrival = self.islands[to_index].core.last_arrival().ok_or(
            ArchipelagoError::MigrationOrganismMissing {
                island: to,
                agent_uid: uid.get(),
            },
        )?;
        Ok(arrival.local_uid)
    }

    /// Put an undeliverable organism back on the island it came from.
    ///
    /// Always returns an error: the barrier has failed either way. What differs
    /// is whether the run still holds every organism, which is why the two
    /// outcomes are distinct error variants rather than one message.
    fn return_migrant_home(
        &mut self,
        from_index: usize,
        from: IslandId,
        uid: AgentUid,
        destination: IslandId,
        migrant: MigratingAgent,
        detail: String,
    ) -> ArchipelagoError {
        let origin = OrganismId::new(from, uid);
        if self.islands[from_index]
            .core
            .stage_immigrant(migrant)
            .is_err()
        {
            return ArchipelagoError::MigrationOrganismLost {
                origin,
                destination,
                detail: format!("{detail}; source could not re-stage it"),
            };
        }
        let command = HostCommand::Immigrate {
            origin_island: from,
            origin_uid: uid,
        };
        match Self::apply_island_command(&mut self.islands[from_index], command) {
            Ok(()) => {
                tracing::warn!(
                    origin = %origin,
                    destination = %destination,
                    detail = %detail,
                    "undeliverable organism returned to its source island"
                );
                ArchipelagoError::MigrationOrganismLost {
                    origin,
                    destination,
                    detail: format!("{detail}; organism was returned home"),
                }
            }
            Err(error) => ArchipelagoError::MigrationOrganismLost {
                origin,
                destination,
                detail: format!("{detail}; return home also failed: {error}"),
            },
        }
    }

    /// Submit one command to an island and drive until it applies.
    ///
    /// Shares the step path's authority-retry and health checks, so a migration
    /// command cannot bypass a gate an ordinary step honours.
    fn apply_island_command(
        island: &mut Island,
        command: HostCommand,
    ) -> Result<(), ArchipelagoError> {
        let island_id = island.meta.id;
        let command_id = island.next_command_id()?;
        let envelope = CommandEnvelope::new(command_id, command);
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
                    detail: "migration command status was not retained".to_owned(),
                })?;
            applied = Self::step_status_applied(island, &status)?;
        }
        Self::verify_island_health(island)?;
        if !applied {
            return Err(ArchipelagoError::NoProgress {
                island: island_id,
                target: island.core.world_tick(),
                current: island.core.world_tick(),
                health: format!("{:?}", island.core.health()),
            });
        }
        Ok(())
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

/// Expand undirected archipelago edges into the sorted directed edges migration
/// uses.
///
/// An archipelago edge `{a, b}` means the islands are connected; migration is
/// directional, so it means BOTH `a -> b` and `b -> a`. Sorting matches the
/// migrator's own normalization so the two representations are comparable.
fn directed_edges(edges: &[(IslandId, IslandId)]) -> Vec<(IslandId, IslandId)> {
    let mut directed = Vec::with_capacity(edges.len() * 2);
    for &(a, b) in edges {
        directed.push((a, b));
        directed.push((b, a));
    }
    directed.sort_unstable();
    directed.dedup();
    directed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EventJournalReader, JournalAdmission, JournalBatch, JournalReceipt, JournalReceiptState,
        ShutdownCommitRequirement,
    };
    use scriptbots_core::AgentUid;
    use scriptbots_core::{BrainRunner, BrainSpawnError, INPUT_SIZE, OUTPUT_SIZE, RandomStream};
    use std::{cell::RefCell, collections::VecDeque, rc::Rc};

    /// Pins the premise of bd-5tyo's parallel-topology rescope: `HostCore` is `!Send`.
    ///
    /// [`StepTopology`] has one variant because an island is a `HostCore`, `HostCore` owns
    /// same-thread command ports, and outer-island parallelism is therefore excluded by the
    /// ownership model rather than merely unimplemented. That reasoning is only honest while
    /// its premise holds, so the premise is CHECKED rather than asserted in prose.
    ///
    /// HOW IT FAILS, since a negative trait bound cannot be written directly: two blanket impls
    /// overlap only for `Send` types. Resolving `probe` for a `!Send` type picks the first impl
    /// unambiguously and compiles; for a `Send` type BOTH apply and the call is ambiguous, so
    /// this stops compiling. If someone makes `HostCore` `Send` — the change that would unlock
    /// outer parallelism — this breaks and the rescope's justification is reconsidered, instead
    /// of a stale "structurally excluded" comment outliving the structure it described.
    #[allow(
        dead_code,
        reason = "the guard is the trait resolution itself; nothing needs to call it at runtime"
    )]
    trait AmbiguousOnlyIfSend<Marker> {
        fn probe() {}
    }
    impl<T: ?Sized> AmbiguousOnlyIfSend<()> for T {}
    impl<T: ?Sized + Send> AmbiguousOnlyIfSend<u8> for T {}

    #[test]
    fn outer_island_parallelism_stays_excluded_while_host_core_is_not_send() {
        // Compiles only because HostCore is !Send. See the trait above for why.
        <HostCore as AmbiguousOnlyIfSend<_>>::probe();
    }

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

    fn spec(id: u32, config: ScriptBotsConfig) -> IslandSpec {
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
            migration: None,
        }
    }

    /// The same config with migration enabled at every barrier.
    fn migrating_config(
        islands: Vec<IslandSpec>,
        interval: u64,
        selection_rule: EmigrantSelectionRule,
        emigrants_per_edge: usize,
    ) -> ArchipelagoConfig {
        let mut config = archipelago_config(islands, interval);
        config.migration = Some(ArchipelagoMigration {
            interval_ticks: interval,
            emigrants_per_edge,
            selection_rule,
        });
        config
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
            .map(|id| spec(u32::try_from(id).expect("small id"), test_config(None)))
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
                config.food_growth_rate =
                    0.01f32.mul_add(f32::from(u16::try_from(id).expect("small id")), 0.01);
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
                config.food_growth_rate =
                    0.01f32.mul_add(f32::from(u16::try_from(id).expect("small id")), 0.01);
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
                config.food_growth_rate =
                    0.02f32.mul_add(f32::from(u16::try_from(id).expect("small id")), 0.01);
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
                IslandId(u32::try_from(index).expect("small index"))
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
        for id in [0_u32, 1] {
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
                config.food_growth_rate =
                    0.02f32.mul_add(f32::from(u16::try_from(id).expect("small id")), 0.01);
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
        for id in 0..4_u32 {
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

    /// PERMUTING THE CONFIG ORDER MUST NOT MOVE AN ISLAND'S SCIENCE.
    ///
    /// bd-16g.5.3 names this as the second half of its independence proof, and says
    /// explicitly that the first half misses it: `test_island_independence_across_
    /// archipelago_sizes` varies the island COUNT while island 0 stays first in the
    /// config, so a seeding rule that keyed off POSITION rather than IslandId would pass
    /// it. This test moves island 0 to the back and asserts nothing about it changed.
    ///
    /// The failure it exists to catch is quiet: position-keyed seeding still yields a
    /// perfectly deterministic archipelago, so every determinism gate stays green while
    /// island identity silently means "whatever slot it was declared in".
    #[test]
    fn island_science_is_independent_of_declaration_order() {
        let master_seed = 0x0BAD_1D3A_0BAD_1D3A;
        let subject = IslandId(0);

        let digest_for = |specs: Vec<IslandSpec>| {
            let mut config = archipelago_config(specs, 100);
            config.master_seed = master_seed;
            let mut archipelago = populated_archipelago(config).expect("valid archipelago");
            let seed = archipelago
                .islands()
                .find(|meta| meta.id == subject)
                .expect("subject island")
                .effective_config
                .rng_seed;
            archipelago.step_to_barrier().expect("barrier");
            let digest = archipelago.island_digest(subject).expect("subject digest");
            (seed, digest)
        };

        let forward: Vec<IslandSpec> = (0..4).map(|id| spec(id, populated_config(None))).collect();
        let mut reversed = forward.clone();
        reversed.reverse();

        let (forward_seed, forward_digest) = digest_for(forward);
        let (reversed_seed, reversed_digest) = digest_for(reversed);

        assert_eq!(
            forward_seed, reversed_seed,
            "the per-island seed must key off IslandId, never the config slot"
        );
        assert_eq!(
            forward_digest, reversed_digest,
            "island {subject:?} produced different science purely because it was declared \
             last instead of first"
        );
    }

    /// Agent identity in an archipelago is `(IslandId, AgentUid)`, and BARE UIDS COLLIDE.
    ///
    /// Each island allocates `AgentUid` from its own private counter, so island 0 and
    /// island 1 both hold an `AgentUid(1)` that are DIFFERENT ORGANISMS. This module's
    /// header states that contract; this test pins it, because the failure it prevents is
    /// silent and lands squarely on bd-16g.5's science acceptance.
    ///
    /// THE HAZARD, stated concretely because I walked into it writing this test: every
    /// lineage and species structure in `scriptbots-core` (bd-16g.3's ancestry DAG,
    /// `SpeciesTable::members`, the phylogeny event timeline) is keyed on a BARE
    /// `AgentUid`. Feed two islands' agents into one of those and distinct organisms merge
    /// into one node -- no panic, no error, just a phylogeny that quietly claims island 0's
    /// agent 1 and island 1's agent 1 are the same individual. Allopatric-speciation
    /// evidence built that way would be plausible, publishable and wrong.
    ///
    /// Also exercises `with_island_world`, the readout that makes per-island science
    /// possible at all: `island_digest` can prove two islands are identical but can never
    /// say what evolved on either.
    #[test]
    fn agent_identity_is_island_scoped_and_bare_uids_collide() {
        let specs: Vec<IslandSpec> = (0..3).map(|id| spec(id, populated_config(None))).collect();
        let mut archipelago =
            populated_archipelago(archipelago_config(specs, 40)).expect("valid archipelago");
        archipelago.step_to_barrier().expect("first barrier");
        archipelago.step_to_barrier().expect("second barrier");

        let mut per_island = Vec::new();
        for id in [IslandId(0), IslandId(1), IslandId(2)] {
            let uids = archipelago
                .with_island_world(id, |world| {
                    world
                        .ordered_agent_rng_counters_v1()
                        .map(|states| {
                            states
                                .iter()
                                .map(|state| state.agent_uid())
                                .collect::<BTreeSet<_>>()
                        })
                        .unwrap_or_default()
                })
                .expect("island world readable");
            // Guard against a vacuous pass: empty sets would satisfy anything below.
            assert!(
                !uids.is_empty(),
                "island {id:?} has no living agents, so nothing here would prove anything"
            );
            per_island.push((id, uids));
        }

        // Bare UIDs OVERLAP. That is the documented design, not a bug.
        let (_, first) = &per_island[0];
        let (_, second) = &per_island[1];
        assert!(
            first.intersection(second).next().is_some(),
            "islands allocate UIDs from private counters, so bare UIDs are expected to \
             collide; if this ever stops being true the identity contract changed"
        );

        // The COMPOUND key is what is actually unique, and every agent has exactly one.
        let mut compound = BTreeSet::new();
        let mut total = 0usize;
        for (id, uids) in &per_island {
            for uid in uids {
                total += 1;
                assert!(
                    compound.insert((*id, *uid)),
                    "(IslandId, AgentUid) must be unique across the whole archipelago"
                );
            }
        }
        assert_eq!(compound.len(), total);

        // And the merge hazard is real: keying on the bare UID loses organisms.
        let bare: BTreeSet<_> = per_island
            .iter()
            .flat_map(|(_, uids)| uids.iter().copied())
            .collect();
        assert!(
            bare.len() < total,
            "if bare UIDs did not collapse organisms this hazard would not need a test; \
             {} bare uids for {total} agents",
            bare.len()
        );
    }

    /// The science readout must refuse an unknown island rather than panic.
    #[test]
    fn with_island_world_rejects_an_unknown_island() {
        let archipelago = Archipelago::new(archipelago_config(vec![spec(0, test_config(None))], 1))
            .expect("single-island archipelago");
        assert!(matches!(
            archipelago.with_island_world(IslandId(7), |_| ()),
            Err(ArchipelagoError::UnknownIsland {
                island: IslandId(7)
            })
        ));
    }

    /// Census of every living agent's ARCHIPELAGO-WIDE identity, asserting uniqueness.
    ///
    /// THE CONSERVATION GUARD (bd-16g.5.2). Migration is the operation that gets this
    /// wrong silently, and population sums alone do not catch it: an agent duplicated on
    /// one island and lost on another leaves the total unchanged. Uniqueness of
    /// `(IslandId, AgentUid)` across all survivors is the half that does catch it.
    ///
    /// Returns the census so a caller can diff it across a barrier. Written before
    /// `emigrate`/`immigrate` exist deliberately: the digest-sensitive mutation should be
    /// born already guarded, so a wrong move fails loudly the first time it lands rather
    /// than corrupting lineage data that looks plausible.
    /// The census now comes from production code ([`Archipelago::organism_census`],
    /// bd-8jlj) rather than a test-local reimplementation. That matters: a guard
    /// that surveys the archipelago its own private way proves the guard is
    /// correct, not that the shipped path is.
    fn census(archipelago: &Archipelago, islands: &[IslandId]) -> BTreeSet<OrganismId> {
        let census = archipelago
            .organism_census()
            .expect("archipelago census readable");
        // Uniqueness is enforced inside `organism_census` by the set itself; what
        // this adds is that every island the caller named is actually represented,
        // so a census that silently skipped an island cannot pass as a full one.
        for &id in islands {
            assert!(
                census.iter().any(|organism| organism.island == id),
                "island {id:?} contributed nothing to the census"
            );
        }
        census
    }

    /// No agent is duplicated or lost across barriers.
    ///
    /// Today migration is not wired, so nothing moves and this holds trivially -- which is
    /// exactly why it is safe to commit now and worth committing now. The moment
    /// `emigrate`/`immigrate` land, a move that duplicates an agent, drops one, or fails
    /// to re-identify an arrival under the destination's allocator breaks this test
    /// LOUDLY instead of producing a plausible, wrong phylogeny.
    ///
    /// MIGRATION IS NOW ON IN THIS TEST, which is what turns it from a guard into a
    /// proof. It was committed while migration was unwired, so it held trivially; the
    /// whole reason to land it early was that the first wrong `emigrate`/`immigrate`
    /// would break it loudly. Running it against isolated islands from here on would be
    /// keeping the shape of a guard while removing everything it guards.
    #[test]
    fn bd_16g_5_2_no_agent_is_duplicated_or_lost_across_barriers() {
        let islands: Vec<IslandId> = (0..3).map(IslandId).collect();
        let specs: Vec<IslandSpec> = islands
            .iter()
            .map(|id| spec(id.0, populated_config(None)))
            .collect();
        let mut archipelago = populated_archipelago(migrating_config(
            specs,
            40,
            EmigrantSelectionRule::Fittest,
            1,
        ))
        .expect("valid archipelago");

        archipelago.step_to_barrier().expect("first barrier");
        let first = census(&archipelago, &islands);
        assert!(
            !first.is_empty(),
            "an empty archipelago would satisfy every assertion below vacuously"
        );

        // Uniqueness must hold at EVERY barrier, not just once: a migrator that
        // duplicates on the third exchange is the realistic bug.
        let mut previous = first;
        let mut total_moves = 0usize;
        for barrier in 0..4 {
            let stepped = archipelago.step_to_barrier();
            assert!(stepped.is_ok(), "barrier {barrier} must step: {stepped:?}");
            let report = stepped.expect("checked above");
            let current = census(&archipelago, &islands);
            assert!(
                !current.is_empty(),
                "population collapsed at barrier {barrier}, so later barriers prove nothing"
            );

            // CONSERVATION ACROSS THE MIGRATION ITSELF. I previously recorded that the
            // accounting identity was not closable because TickSummary births/deaths are
            // per tick and a barrier spans many. That was measuring across the wrong
            // window. NO TICK RUNS INSIDE THE MIGRATION PHASE -- every island is already
            // at the barrier tick and only migration commands are applied -- so births
            // and deaths are structurally zero across it and the identity collapses to
            // `after == before`. `migrate_at_barrier` enforces exactly that in
            // production; here we check the report agrees.
            assert!(
                report.migration.is_some(),
                "barrier {barrier} must have run migration"
            );
            let migration = report
                .migration
                .as_ref()
                .expect("checked immediately above");
            assert_eq!(
                migration.total_before(),
                migration.total_after(),
                "migration must conserve population exactly at barrier {barrier}"
            );
            total_moves += migration.moves.len();

            // THE MOVEMENT WITNESS, which population state cannot be. Each move names
            // both ends of the journey, and both ends must exist in the census: the
            // arrival under its fresh destination-local uid, and the departure NOT on
            // its source island any more.
            for applied in &migration.moves {
                assert!(
                    current.contains(&applied.to),
                    "barrier {barrier}: arrival {} is missing from the census",
                    applied.to
                );
                assert!(
                    !current.contains(&applied.from),
                    "barrier {barrier}: departed organism {} is still on its source island",
                    applied.from
                );
                assert_ne!(
                    applied.from.island, applied.to.island,
                    "a migration must cross islands"
                );
            }

            let _ = &previous;
            previous = current;
        }
        assert!(
            total_moves > 0,
            "no organism ever moved, so nothing above was actually exercised"
        );
    }

    /// Migration moves real organisms, conserves population exactly, and is
    /// reproducible barrier for barrier (bd-16g.5.2).
    ///
    /// The E2E proof the bead asks for, and the assertions are chosen for what they
    /// would CATCH rather than for what reads well:
    ///
    /// - Population sums alone are NOT sufficient and are not claimed to be. An agent
    ///   duplicated on one island and lost on another leaves the total unchanged, which
    ///   is exactly the silent failure. Uniqueness of `(IslandId, AgentUid)` across the
    ///   whole archipelago is the half that catches it, and `organism_census` enforces it
    ///   as the set is built.
    /// - A move is INDISTINGUISHABLE from a death plus a birth from population state,
    ///   because the arrival necessarily takes a fresh local uid. So the journaled move
    ///   record is the only witness that a specific organism travelled, and every move is
    ///   checked against the census at both ends.
    /// - Determinism is asserted against a SECOND archipelago built from the same seed,
    ///   not against a stored constant, so the property proven is reproducibility rather
    ///   than agreement with a number someone once wrote down.
    #[test]
    fn bd_16g_5_2_migration_conserves_population_and_repeats_exactly() {
        let islands: Vec<IslandId> = (0..4).map(IslandId).collect();
        let build = || {
            let specs: Vec<IslandSpec> = islands
                .iter()
                .map(|id| spec(id.0, populated_config(None)))
                .collect();
            populated_archipelago(migrating_config(
                specs,
                30,
                EmigrantSelectionRule::Fittest,
                2,
            ))
            .expect("valid migrating archipelago")
        };

        let mut first = build();
        let mut second = build();
        let mut observed_moves = 0usize;

        for barrier in 0..5 {
            let left = first.step_to_barrier().expect("first archipelago steps");
            let right = second.step_to_barrier().expect("second archipelago steps");

            assert!(
                left.migration.is_some() && right.migration.is_some(),
                "barrier {barrier} must run migration on both archipelagos"
            );
            let left_migration = left.migration.as_ref().expect("checked immediately above");
            let right_migration = right.migration.as_ref().expect("checked immediately above");

            assert_eq!(
                left_migration.total_before(),
                left_migration.total_after(),
                "barrier {barrier} must conserve population exactly"
            );

            // Same seed, same moves -- including which organism, in which order.
            assert_eq!(
                left_migration.moves, right_migration.moves,
                "two archipelagos with the same master seed must migrate identically \
                 at barrier {barrier}"
            );

            // Canonical application order is (to, from, uid). Anything else makes the
            // outcome depend on edge iteration order.
            let keys: Vec<_> = left_migration
                .moves
                .iter()
                .map(|applied| (applied.to.island, applied.from.island, applied.from.uid))
                .collect();
            let mut sorted = keys.clone();
            sorted.sort_unstable();
            assert_eq!(keys, sorted, "moves must be applied in canonical order");

            // No organism may move twice in one barrier: the two-phase select-all then
            // apply-all discipline is what makes that unrepresentable, and a per-edge
            // select-and-apply loop would duplicate the fittest agent across its edges.
            let departures: BTreeSet<_> = left_migration
                .moves
                .iter()
                .map(|applied| applied.from)
                .collect();
            assert_eq!(
                departures.len(),
                left_migration.moves.len(),
                "an organism emigrated twice in one barrier at {barrier}"
            );

            let census = first.organism_census().expect("census readable");
            let population: usize = islands
                .iter()
                .map(|&id| {
                    first
                        .with_island_world(id, WorldState::agent_count)
                        .expect("island readable")
                })
                .sum();
            assert_eq!(
                census.len(),
                population,
                "every living agent must have exactly one archipelago identity at \
                 barrier {barrier}"
            );

            observed_moves += left_migration.moves.len();
        }

        assert!(
            observed_moves > 0,
            "migration never moved anyone, so conservation held vacuously"
        );
    }

    /// Isolated islands stay bit-identical when migration is off.
    ///
    /// The control for the test above: it proves the migration machinery is inert when
    /// not configured, so a digest change elsewhere cannot be blamed on it, and that
    /// enabling migration is what actually perturbs the islands.
    #[test]
    fn bd_16g_5_2_migration_off_leaves_island_digests_untouched() {
        let specs = || {
            (0..3)
                .map(|id| spec(id, populated_config(None)))
                .collect::<Vec<_>>()
        };
        let mut isolated =
            populated_archipelago(archipelago_config(specs(), 30)).expect("isolated archipelago");
        let mut migrating = populated_archipelago(migrating_config(
            specs(),
            30,
            EmigrantSelectionRule::Fittest,
            2,
        ))
        .expect("migrating archipelago");

        for _ in 0..3 {
            isolated.step_to_barrier().expect("isolated steps");
            migrating.step_to_barrier().expect("migrating steps");
        }

        let mut differed = false;
        for id in (0..3).map(IslandId) {
            let quiet = isolated.island_digest(id).expect("isolated digest");
            let moved = migrating.island_digest(id).expect("migrating digest");
            if quiet != moved {
                differed = true;
            }
        }
        assert!(
            differed,
            "enabling migration must actually change island science; if the digests \
             match, nothing moved and every migration test above is vacuous"
        );
    }

    /// A migration interval that is not a multiple of the barrier is refused.
    ///
    /// Rounding it silently would run migration at a cadence the caller never asked
    /// for, and a wrong migration rate changes the science while every gate stays green.
    #[test]
    fn bd_16g_5_2_unaligned_migration_interval_is_a_construction_error() {
        let specs: Vec<IslandSpec> = (0..2).map(|id| spec(id, test_config(None))).collect();
        let mut config = archipelago_config(specs, 40);
        config.migration = Some(ArchipelagoMigration {
            interval_ticks: 30,
            emigrants_per_edge: 1,
            selection_rule: EmigrantSelectionRule::Fittest,
        });
        assert!(matches!(
            Archipelago::new(config),
            Err(ArchipelagoError::MigrationIntervalNotBarrierAligned {
                interval_ticks: 30,
                barrier_interval: 40,
            })
        ));

        let specs: Vec<IslandSpec> = (0..2).map(|id| spec(id, test_config(None))).collect();
        let mut zero = archipelago_config(specs, 40);
        zero.migration = Some(ArchipelagoMigration {
            interval_ticks: 0,
            emigrants_per_edge: 1,
            selection_rule: EmigrantSelectionRule::Fittest,
        });
        assert!(matches!(
            Archipelago::new(zero),
            Err(ArchipelagoError::MigrationIntervalNotBarrierAligned {
                interval_ticks: 0,
                ..
            })
        ));
    }

    // ---------------------------------------------------------------------
    // bd-tfso: born-red negative controls for the conservation invariant
    // ---------------------------------------------------------------------

    /// A real barrier's migration, and the two censuses that bracket it.
    ///
    /// Mutations below are applied to THIS — actual organisms from an actual
    /// migrating archipelago — rather than to invented identities, so a mutant
    /// is a corruption of something the migrator really produced.
    struct RealBarrier {
        before: BTreeSet<OrganismId>,
        after: BTreeSet<OrganismId>,
        moves: Vec<AppliedMigration>,
    }

    /// Run one migrating archipelago far enough to produce real moves.
    ///
    /// The censuses come from the report rather than from `organism_census()`
    /// around the call, and that distinction is load-bearing: a barrier STEPS
    /// every island before it migrates, so an organism born during this
    /// barrier's stepping phase can emigrate in it and would be absent from a
    /// pre-barrier census. I found that by writing the pre-barrier version
    /// first and watching it fail on `island-3/agent-24`.
    fn real_barrier() -> RealBarrier {
        let specs: Vec<IslandSpec> = (0..4).map(|id| spec(id, populated_config(None))).collect();
        let mut archipelago = populated_archipelago(migrating_config(
            specs,
            30,
            EmigrantSelectionRule::Fittest,
            2,
        ))
        .expect("valid migrating archipelago");

        archipelago.step_to_barrier().expect("warm-up barrier");
        let report = archipelago.step_to_barrier().expect("measured barrier");
        let migration = report.migration.expect("barrier migrated");
        assert!(
            !migration.moves.is_empty(),
            "no organism moved, so every mutation below would be vacuous"
        );
        RealBarrier {
            before: migration.census_before,
            after: migration.census_after,
            moves: migration.moves,
        }
    }

    /// The unmutated barrier must PASS, or every red result below is meaningless.
    ///
    /// This re-derives the verdict from the report's own evidence rather than
    /// trusting that production reached it — which is the point of the report
    /// carrying the censuses at all.
    #[test]
    fn bd_tfso_the_unmutated_barrier_passes_its_own_verifier() {
        let barrier = real_barrier();
        for applied in &barrier.moves {
            assert!(
                barrier.before.contains(&applied.from),
                "the departure {} must have existed before migration",
                applied.from
            );
            assert!(
                barrier.after.contains(&applied.to),
                "the arrival {} must exist after migration",
                applied.to
            );
            assert!(
                !barrier.after.contains(&applied.from),
                "the departure {} must be gone from its source island",
                applied.from
            );
        }
        assert_eq!(
            verify_migration(&barrier.before, &barrier.after, &barrier.moves),
            Ok(()),
            "a correct barrier must pass; if it does not, the red results below \
             prove nothing about corruption"
        );
        assert_eq!(
            barrier.before.len(),
            barrier.after.len(),
            "no tick runs inside migration, so the population cannot change"
        );
    }

    /// MUTANT 1 — DROP AN EMIGRANT. Caught by `UnexplainedAbsence`.
    ///
    /// The organism is collected from its source and never delivered: the move
    /// record says it arrived, the population says it did not. This is the
    /// failure a naive "did the totals match?" check reports as a mere count
    /// mismatch and a census-only check misses entirely.
    #[test]
    fn bd_tfso_a_dropped_emigrant_is_caught_as_an_unexplained_absence() {
        let barrier = real_barrier();
        let dropped = barrier.moves[0].to;
        let mut mutant = barrier.after.clone();
        assert!(mutant.remove(&dropped), "the arrival was there to drop");

        assert_eq!(
            verify_migration(&barrier.before, &mutant, &barrier.moves),
            Err(MigrationViolation::NotConserved {
                before: barrier.before.len(),
                after: barrier.before.len() - 1,
            }),
            "a dropped emigrant must be caught"
        );

        // AND WITH THE COUNT MADE TO MATCH by fabricating a replacement, the
        // identity check is what catches it — the case that matters, because a
        // sums-only conservation check is now perfectly satisfied.
        let fabricated = OrganismId::new(dropped.island, AgentUid(u64::MAX));
        let mut disguised = mutant;
        disguised.insert(fabricated);
        assert_eq!(
            disguised.len(),
            barrier.before.len(),
            "the disguise must restore the total, or it is not a disguise"
        );
        assert_eq!(
            verify_migration(&barrier.before, &disguised, &barrier.moves),
            Err(MigrationViolation::UnexplainedPresence {
                organism: fabricated,
            }),
            "a loss disguised by a fabricated organism must still be caught"
        );
    }

    /// MUTANT 2 — DUPLICATE AN ORGANISM ACROSS TWO ISLANDS. Caught by
    /// `UnexplainedPresence`.
    ///
    /// *** THE HALF THE OPERATOR NAMED, AND THE ONE POPULATION SUMS CANNOT SEE.
    /// *** An emigration that copied instead of moving leaves the organism on
    /// its source island AND on its destination. Pair it with a loss elsewhere
    /// and the total is unchanged, which is precisely why this test exists.
    #[test]
    fn bd_tfso_an_organism_duplicated_across_islands_is_caught() {
        let barrier = real_barrier();
        let duplicated = barrier.moves[0].from;

        // The emigration copied instead of moving: the organism is on its source
        // island AND on its destination.
        let mut duplicated_only = barrier.after.clone();
        assert!(
            duplicated_only.insert(duplicated),
            "the source copy must not already be there, or nothing was duplicated"
        );
        assert_eq!(
            verify_migration(&barrier.before, &duplicated_only, &barrier.moves),
            Err(MigrationViolation::NotConserved {
                before: barrier.before.len(),
                after: barrier.before.len() + 1,
            })
        );

        // NOW THE ONE THAT MATTERS: duplicate on one island, lose on another, so
        // THE TOTAL IS UNCHANGED. A conservation check built on sums passes this
        // happily; that is precisely the failure this bead's guard exists for.
        let mut balanced = duplicated_only;
        let victim = barrier
            .moves
            .last()
            .map(|applied| applied.to)
            .expect("at least one move");
        assert!(balanced.remove(&victim), "something else to lose");
        assert_eq!(
            balanced.len(),
            barrier.before.len(),
            "the mutant must be population-neutral, or it is not the hard case"
        );
        assert_eq!(
            verify_migration(&barrier.before, &balanced, &barrier.moves),
            Err(MigrationViolation::UnexplainedPresence {
                organism: duplicated,
            }),
            "duplicate-and-lose keeps the total unchanged and MUST still be caught"
        );
    }

    /// MUTANT 3 — REORDER SELECTION. Caught by `OutOfCanonicalOrder`.
    ///
    /// Application order is the contract. Out of canonical order the outcome
    /// depends on edge iteration order, which is how an archipelago stops being
    /// reproducible while still looking fine at one thread.
    #[test]
    fn bd_tfso_reordered_moves_are_caught() {
        let barrier = real_barrier();
        assert!(
            barrier.moves.len() >= 2,
            "reordering needs at least two moves"
        );
        let mut reordered = barrier.moves.clone();
        reordered.reverse();
        assert_eq!(
            verify_migration(&barrier.before, &barrier.after, &reordered),
            Err(MigrationViolation::OutOfCanonicalOrder { index: 1 }),
            "reversing a correctly ordered plan must be caught at the first pair"
        );
    }

    /// MUTANT 4 — EMIGRATE THE SAME ORGANISM TWICE IN ONE BARRIER.
    ///
    /// This is what a per-edge select-and-apply loop produces: island 1's
    /// fittest agent chosen for the edge to island 0 and again for the edge to
    /// island 2. The two-phase discipline makes it unreachable in production;
    /// this proves the verifier would catch it if that discipline were lost.
    #[test]
    fn bd_tfso_an_organism_emigrating_twice_is_caught() {
        let barrier = real_barrier();
        let mut moves = barrier.moves.clone();
        let mut cloned = moves[0];
        // Same departure, a different arrival identity, and placed so the plan
        // stays canonically ordered — otherwise the ORDER check would fire first
        // and this would not be testing what it claims to test.
        cloned.to = OrganismId::new(cloned.to.island, AgentUid(u64::MAX));
        moves.insert(1, cloned);
        moves.sort_by_key(|applied| (applied.to.island, applied.from.island, applied.from.uid));

        assert_eq!(
            verify_migration(&barrier.before, &barrier.after, &moves),
            Err(MigrationViolation::DepartedTwice {
                organism: barrier.moves[0].from,
            }),
            "one organism cannot leave twice in a single barrier"
        );
    }

    /// MUTANT 5 — AN ARRIVAL THAT REUSES AN EXISTING IDENTITY.
    ///
    /// The bd-8jlj hazard at the migration boundary: if an arrival took the
    /// source UID instead of a fresh destination one it could land on top of a
    /// living organism, and two individuals would silently become one.
    #[test]
    fn bd_tfso_an_arrival_colliding_with_a_living_organism_is_caught() {
        let barrier = real_barrier();
        let mut moves = barrier.moves.clone();
        let victim = *barrier
            .before
            .iter()
            .find(|organism| !moves.iter().any(|m| m.from == **organism))
            .expect("a bystander organism exists");
        moves[0].to = victim;

        assert_eq!(
            verify_migration(&barrier.before, &barrier.after, &moves),
            Err(MigrationViolation::ArrivalCollides { organism: victim }),
            "an arrival must never reuse a living organism's identity"
        );
    }

    /// DUPLICATION BY DOUBLE DELIVERY IS UNREPRESENTABLE, not merely untested.
    ///
    /// `MigratingAgent` owns a live `Box<dyn BrainRunner>` and is deliberately
    /// NOT `Clone`; `WorldState::immigrate` consumes it by value. So "deliver
    /// the same organism to two islands" cannot be written — it is a borrow
    /// checker error, not a runtime failure. This test documents the argument
    /// and pins the property it depends on, because a future `#[derive(Clone)]`
    /// added for convenience would silently reopen the hole.
    ///
    /// The verifier above covers the OTHER duplication route, where an
    /// emigration fails to remove the organism from its source.
    #[test]
    fn bd_tfso_double_delivery_is_prevented_by_ownership_not_by_checking() {
        fn assert_not_clone<T>() {
            // Compiles for any T; the point is the negative trait bound below.
        }
        assert_not_clone::<MigratingAgent>();
        // If `MigratingAgent` ever gains `Clone`, this stops compiling and the
        // reviewer is forced to think about why it was not `Clone`.
        trait NotClone {}
        impl<T: ?Sized> NotClone for T {}
        fn requires_no_clone<T: NotClone>() {}
        requires_no_clone::<MigratingAgent>();

        // The real evidence is structural and is asserted by the type system at
        // every call site: `deliver_migrant` takes the organism by value and
        // there is exactly one of it.
        let barrier = real_barrier();
        let arrivals: BTreeSet<_> = barrier.moves.iter().map(|applied| applied.to).collect();
        assert_eq!(
            arrivals.len(),
            barrier.moves.len(),
            "every arrival identity is distinct"
        );
    }

    /// A journal that keeps every batch so a run can be reconstructed from it.
    ///
    /// Receipts are DRAINED from a queue rather than regenerated per poll. My
    /// first version rebuilt them from the retained batches on every call, which
    /// re-acknowledged work the host had already accounted for and latched the
    /// island with `science_blocked` on the second barrier — a reminder that a
    /// test double still has to honour the protocol it stands in for.
    #[derive(Clone, Default)]
    struct RecordingJournal {
        batches: Rc<RefCell<Vec<Arc<JournalBatch>>>>,
        pending: Rc<RefCell<VecDeque<JournalReceipt>>>,
    }

    impl JournalPort for RecordingJournal {
        fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
            self.batches.borrow_mut().push(Arc::clone(batch));
            self.pending.borrow_mut().push_back(JournalReceipt::new(
                batch.id(),
                JournalReceiptState::Durable,
            ));
            JournalAdmission::Accepted {
                batch_id: batch.id(),
            }
        }

        fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
            let mut pending = self.pending.borrow_mut();
            let count = limit.min(pending.len());
            pending.drain(..count).collect()
        }

        fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
            ShutdownCommitRequirement::CommittedVolatile
        }
    }

    /// THE JOURNAL IS A RECORD, NOT A LOG: replaying it reconstructs every
    /// island's population exactly (bd-tfso).
    ///
    /// This is the other half of the conservation claim, and it composes with the
    /// mutation tests above. Those prove the in-memory invariant catches
    /// corruption; this proves the DURABLE record is sufficient to rebuild the
    /// state that invariant was checked against. A journal that cannot do that is
    /// a diagnostic log — useful for reading, useless for replay — and bd-16g.5.2
    /// requires migration to be replayable.
    ///
    /// The reconstruction closes the full accounting identity, per island:
    ///
    /// ```text
    /// population == initial + Σ births − Σ deaths − Σ emigrations + Σ immigrations
    /// ```
    ///
    /// Every term comes from the journal and nothing from live world state:
    /// births and deaths from each `Step` batch's `ScientificBoundary` summary,
    /// migrations from the `Emigrate`/`Immigrate` command envelopes. That is the
    /// identity I earlier recorded on bd-16g.5.2 as unclosable — it is closable,
    /// and this is where all four terms finally exist in one place.
    ///
    /// Only APPLIED commands count. Every terminal lifecycle is journaled,
    /// including failures, so counting a failed `Emigrate` as a departure would
    /// reconstruct a population that never existed from a record that is itself
    /// perfectly correct.
    #[test]
    fn bd_tfso_replaying_the_journal_reconstructs_every_island_population() {
        let islands: Vec<IslandId> = (0..3).map(IslandId).collect();
        let specs: Vec<IslandSpec> = islands
            .iter()
            .map(|id| spec(id.0, populated_config(None)))
            .collect();
        let journals: Vec<RecordingJournal> = islands
            .iter()
            .map(|_| RecordingJournal::default())
            .collect();
        let recorders = journals.clone();

        let mut archipelago = Archipelago::with_factories(
            migrating_config(specs, 30, EmigrantSelectionRule::Fittest, 2),
            populated_world_factory,
            |meta| {
                let index = meta.id.0 as usize;
                Some(Box::new(recorders[index].clone()) as Box<dyn JournalPort>)
            },
        )
        .expect("recording archipelago");

        let initial: Vec<i64> = islands
            .iter()
            .map(|&id| {
                i64::try_from(
                    archipelago
                        .with_island_world(id, WorldState::agent_count)
                        .expect("island readable"),
                )
                .expect("population fits i64")
            })
            .collect();

        let mut total_moves = 0usize;
        let mut last_barrier_arrivals: BTreeMap<IslandId, i64> = BTreeMap::new();
        for _ in 0..4 {
            let report = archipelago.step_to_barrier().expect("barrier steps");
            last_barrier_arrivals.clear();
            if let Some(migration) = report.migration {
                total_moves += migration.moves.len();
                for applied in &migration.moves {
                    *last_barrier_arrivals.entry(applied.to.island).or_insert(0) += 1;
                }
            }
        }
        assert!(
            total_moves > 0,
            "no organism migrated, so the migration terms below are all zero and \
             the reconstruction would succeed without proving anything"
        );

        for (index, &id) in islands.iter().enumerate() {
            let mut population = initial[index];
            let mut births = 0i64;
            let mut deaths = 0i64;
            let mut emigrations = 0i64;
            let mut immigrations = 0i64;

            for batch in journals[index].batches.borrow().iter() {
                if let Some(scientific) = batch.scientific() {
                    // The RECORDS, not `summary.births`. That counter is
                    // Born-only by design, so it omits population-floor
                    // injections AND arrivals; reconstructing from it gave 0
                    // against a live population of 36, which is how I found
                    // that. The record list carries every origin.
                    births += i64::try_from(scientific.births().len()).expect("births fit i64");
                    deaths += i64::try_from(scientific.deaths().len()).expect("deaths fit i64");
                }
                let Some(lifecycle) = batch.command_lifecycle() else {
                    continue;
                };
                if !lifecycle.was_applied() {
                    continue;
                }
                let Some(envelope) = batch.command() else {
                    continue;
                };
                match &envelope.command {
                    HostCommand::Emigrate { .. } => emigrations += 1,
                    HostCommand::Immigrate { .. } => immigrations += 1,
                    _ => {}
                }
            }

            // *** ARRIVALS ARE INSIDE `births`, AND THAT IS THE bd-it29 FIX. ***
            // `immigrate` records the arrival as an Injected BirthRecord in the
            // destination, and since bd-it29 that record reaches the next
            // scientific boundary like any other. So the arrival term must NOT
            // be added again — it is already counted. Before the fix it reached
            // no boundary at all and had to be added; this test asserted that
            // exclusion explicitly so the fix would break it, and it did.
            //
            // WHAT REMAINS IS THE DEFERRED TAIL, which is real rather than an
            // artifact: a barrier steps and THEN migrates, so the final
            // barrier's arrivals are recorded after the last step and no
            // boundary has reported them yet. They are present in the world and
            // absent from the journal, so the reconstruction must add exactly
            // those back. Naming the tail is better than stepping once more to
            // hide it, because every run has one.
            let deferred = last_barrier_arrivals.get(&id).copied().unwrap_or(0);
            population += births - deaths - emigrations + deferred;
            let actual = i64::try_from(
                archipelago
                    .with_island_world(id, WorldState::agent_count)
                    .expect("island readable"),
            )
            .expect("population fits i64");
            assert_eq!(
                population, actual,
                "island {id} reconstructed from its journal as {population} but holds \
                 {actual}: initial {}, births {births}, deaths {deaths}, emigrations \
                 {emigrations}, immigrations {immigrations}, deferred arrivals {deferred}",
                initial[index]
            );

            // The arrival records are genuinely there: every arrival except the
            // deferred tail must be inside `births`. This is the assertion that
            // would go red if bd-it29 regressed.
            assert!(
                births >= immigrations - deferred,
                "island {id}: {births} birth records cannot account for {immigrations} \
                 arrivals with {deferred} deferred; arrivals have stopped reaching the \
                 scientific boundary again (bd-it29)"
            );
        }

        // The migration terms must actually be non-zero SOMEWHERE, or the
        // identity above was proven only for births and deaths.
        let journaled_moves: usize = journals
            .iter()
            .map(|journal| {
                journal
                    .batches
                    .borrow()
                    .iter()
                    .filter(|batch| {
                        batch.command_lifecycle().is_some_and(|l| l.was_applied())
                            && batch.command().is_some_and(|e| {
                                matches!(&e.command, HostCommand::Immigrate { .. })
                            })
                    })
                    .count()
            })
            .sum();
        assert_eq!(
            journaled_moves, total_moves,
            "every applied migration must appear in exactly one island's journal"
        );
    }

    /// Every arrival is named in its destination's journaled birth records —
    /// and the conservation verifier CANNOT see when one is not (bd-it29).
    ///
    /// *** THE GAP THIS EXISTS TO CLOSE. *** bd-tfso's five mutation controls
    /// all attack `verify_migration`, which compares ORGANISM IDENTITIES taken
    /// from live world state. A missing birth record does not change a single
    /// identity: the agent is present on the destination, its census entry is
    /// correct, conservation holds exactly. So every one of those controls
    /// passes a world whose scientific record has silently lost the arrival —
    /// which is precisely the bd-it29 defect, and precisely the bd-0oro shape,
    /// where a success signal outruns what the record can show.
    ///
    /// This test asserts the record, then proves the gap by running the
    /// conservation verifier over the same barrier and showing it is content.
    #[test]
    fn bd_it29_every_arrival_is_named_in_the_destination_birth_records() {
        let islands: Vec<IslandId> = (0..3).map(IslandId).collect();
        let specs: Vec<IslandSpec> = islands
            .iter()
            .map(|id| spec(id.0, populated_config(None)))
            .collect();
        let journals: Vec<RecordingJournal> = islands
            .iter()
            .map(|_| RecordingJournal::default())
            .collect();
        let recorders = journals.clone();
        let mut archipelago = Archipelago::with_factories(
            migrating_config(specs, 30, EmigrantSelectionRule::Fittest, 2),
            populated_world_factory,
            |meta| {
                let index = meta.id.0 as usize;
                Some(Box::new(recorders[index].clone()) as Box<dyn JournalPort>)
            },
        )
        .expect("recording archipelago");

        // Arrivals from every barrier EXCEPT the last, whose records no step has
        // reported yet: a barrier steps and then migrates.
        let mut expected: Vec<AppliedMigration> = Vec::new();
        let mut last_barrier: Vec<AppliedMigration> = Vec::new();
        let mut final_verify: Option<MigrationBarrierReport> = None;
        for _ in 0..4 {
            let report = archipelago.step_to_barrier().expect("barrier steps");
            if let Some(migration) = report.migration {
                expected.extend(last_barrier.drain(..));
                last_barrier = migration.moves.clone();
                final_verify = Some(migration);
            }
        }
        assert!(
            !expected.is_empty(),
            "no reportable arrival occurred, so this proves nothing"
        );

        let journaled = |island: IslandId| -> BTreeSet<AgentUid> {
            let index = island.0 as usize;
            journals[index]
                .batches
                .borrow()
                .iter()
                .filter_map(|batch| batch.scientific().map(|s| s.births().to_vec()))
                .flatten()
                .map(|record| record.agent_uid)
                .collect()
        };

        for applied in &expected {
            assert!(
                journaled(applied.to.island).contains(&applied.to.uid),
                "arrival {} reached the world but no birth record in its destination's \
                 journal names it; the island's science cannot see that anyone arrived",
                applied.to
            );
        }

        // MUTANT: drop one arrival's record. The check above catches it...
        let victim = expected[0];
        let mut mutated = journaled(victim.to.island);
        assert!(
            mutated.remove(&victim.to.uid),
            "the record was there to drop"
        );
        assert!(
            !mutated.contains(&victim.to.uid),
            "dropping an arrival record must be detectable from the journal alone"
        );

        // ...AND THE CONSERVATION VERIFIER DOES NOT, which is the whole point.
        // The organism is present, its identity is correct, the population is
        // exactly conserved. Nothing about the missing record is expressible in
        // the identity sets bd-tfso's controls compare.
        let barrier = final_verify.expect("a barrier migrated");
        assert_eq!(
            verify_migration(
                &barrier.census_before,
                &barrier.census_after,
                &barrier.moves
            ),
            Ok(()),
            "the conservation verifier is content with this world, which is exactly \
             why a record-level assertion has to exist separately"
        );
    }

    /// Migrants between DIFFERENTLY SIZED islands land inside their destination
    /// (bd-tfso).
    ///
    /// `scriptbots-core` proves the normalized `(x/w, y/h)` remap in isolation,
    /// including a far-edge migrant from 4000x4000 into 100x100. What that
    /// cannot show is that the remap is actually reached through a real barrier
    /// between real islands of different sizes — an archipelago that never
    /// applied it would pass every core test and still drop agents outside the
    /// world, where the food and terrain grids have no cell for them.
    #[test]
    fn bd_tfso_migrants_between_differently_sized_islands_land_in_bounds() {
        let sizes = [(600u32, 300u32), (1200, 600), (300, 150)];
        let specs: Vec<IslandSpec> = sizes
            .iter()
            .enumerate()
            .map(|(index, &(width, height))| {
                let mut config = populated_config(None);
                config.world_width = width;
                config.world_height = height;
                spec(u32::try_from(index).expect("index fits u32"), config)
            })
            .collect();
        let mut archipelago = populated_archipelago(migrating_config(
            specs,
            30,
            EmigrantSelectionRule::Fittest,
            2,
        ))
        .expect("heterogeneous migrating archipelago");

        let mut moved = 0usize;
        for _ in 0..4 {
            let report = archipelago.step_to_barrier().expect("barrier steps");
            moved += report
                .migration
                .map_or(0, |migration| migration.moves.len());
        }
        assert!(
            moved > 0,
            "no organism crossed between differently sized worlds, so nothing was remapped"
        );

        for (index, &(width, height)) in sizes.iter().enumerate() {
            let id = IslandId(u32::try_from(index).expect("index fits u32"));
            let outside = archipelago
                .with_island_world(id, |world| {
                    world
                        .agents()
                        .iter_handles()
                        .filter_map(|handle| world.agents().snapshot(handle))
                        .filter(|data| {
                            let x = data.position.x;
                            let y = data.position.y;
                            !(x.is_finite() && y.is_finite())
                                || x < 0.0
                                || y < 0.0
                                || x >= width as f32
                                || y >= height as f32
                        })
                        .count()
                })
                .expect("island readable");
            assert_eq!(
                outside, 0,
                "island {id} ({width}x{height}) holds {outside} agents outside its own \
                 bounds; a migrant that kept its source coordinates would land here"
            );
        }
    }

    /// The migration graph is the archipelago's own topology, expanded both ways.
    ///
    /// Pins the reason [`ArchipelagoMigration`] has no topology field: `Topology::Ring`
    /// normalizes to UNDIRECTED edges, so on three islands it is the complete graph,
    /// while `MigrationTopology::Ring` is the directed cycle. Two descriptions would
    /// disagree here immediately and silently.
    #[test]
    fn bd_16g_5_2_migration_edges_are_the_archipelago_topology_both_ways() {
        let undirected = [
            (IslandId(0), IslandId(1)),
            (IslandId(0), IslandId(2)),
            (IslandId(1), IslandId(2)),
        ];
        assert_eq!(
            directed_edges(&undirected),
            vec![
                (IslandId(0), IslandId(1)),
                (IslandId(0), IslandId(2)),
                (IslandId(1), IslandId(0)),
                (IslandId(1), IslandId(2)),
                (IslandId(2), IslandId(0)),
                (IslandId(2), IslandId(1)),
            ],
            "connectivity is symmetric but a move is not, so each edge expands both ways"
        );
        assert_ne!(
            directed_edges(&undirected),
            MigrationTopology::Ring.build_edges(&[IslandId(0), IslandId(1), IslandId(2)]),
            "the two Ring meanings genuinely differ; that is why only one exists here"
        );
    }

    /// The archipelago-wide census keys on `(IslandId, AgentUid)`, and the bare-UID
    /// union it replaces provably loses organisms (bd-8jlj).
    ///
    /// THE DECISION THIS PINS. bd-8jlj asked whether core's lineage/species APIs should
    /// take an island-scoped key or whether an archipelago layer should key on the pair.
    /// The answer is the pair, because ancestry and species are properties of ONE
    /// interbreeding population and allopatry means the islands' pools are separate --
    /// so the core structures are right as they are, and the danger lives entirely in
    /// how a caller COMBINES them.
    ///
    /// That makes this test the real guard: it does the wrong thing and the right thing
    /// side by side over the same live archipelago and shows the wrong one is lossy. A
    /// test that only asserted the census is unique would pass just as happily against a
    /// single island, where bare UIDs are already unique and nothing is at stake.
    #[test]
    fn bd_8jlj_census_keys_on_the_island_pair_and_bare_uids_lose_organisms() {
        let islands: Vec<IslandId> = (0..3).map(IslandId).collect();
        let specs: Vec<IslandSpec> = islands
            .iter()
            .map(|id| spec(id.0, populated_config(None)))
            .collect();
        let mut archipelago =
            populated_archipelago(archipelago_config(specs, 40)).expect("valid archipelago");
        archipelago.step_to_barrier().expect("first barrier");

        let census = archipelago.organism_census().expect("census readable");
        assert!(
            !census.is_empty(),
            "an empty census would satisfy everything below vacuously"
        );
        for &id in &islands {
            assert!(
                census.iter().any(|organism| organism.island == id),
                "every island must contribute, or the union below is not a real union"
            );
        }

        // THE WRONG THING, written out so the loss is visible rather than argued.
        let bare: BTreeSet<AgentUid> = census.iter().map(|organism| organism.uid).collect();
        assert!(
            bare.len() < census.len(),
            "if discarding the island axis were harmless this hazard would need no type; \
             {} bare uids for {} organisms",
            bare.len(),
            census.len()
        );

        // Ordering is island-major, which is the canonical order every barrier and
        // migration surface uses. A caller that relied on UID-major order would read
        // one island's population as another's.
        let ordered: Vec<IslandId> = census.iter().map(|organism| organism.island).collect();
        let mut sorted = ordered.clone();
        sorted.sort_unstable();
        assert_eq!(
            ordered, sorted,
            "OrganismId must order island-major so a census iterates island by island"
        );

        // And the pair is genuinely unique: the set's own length is the population.
        let total: usize = islands
            .iter()
            .map(|&id| {
                archipelago
                    .with_island_world(id, WorldState::agent_count)
                    .expect("island world readable")
            })
            .sum();
        assert_eq!(
            census.len(),
            total,
            "the census must hold exactly one entry per living agent"
        );
    }
}
