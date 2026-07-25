//! Domain-separated random streams.
//!
//! # The problem this exists to solve
//!
//! Before the domain cutover the world drew every stochastic decision from ONE stream: food scatter, spawn
//! placement, mutation, crossover, reproduction rolls, cull selection — all of it, in sequence.
//! That has two consequences, and the second one is severe.
//!
//! **1. Any code change that adds or removes a single draw shifts every subsequent draw.** Add
//! one `rng.random_range(..)` to the food logic and every mutation, every spawn position, every
//! reproduction roll for the rest of the run lands somewhere else. The characterization digest
//! moves, and it moves for a reason that has nothing to do with the science. A reviewer then
//! cannot tell "I changed the physics" apart from "I added a draw", which is the single most
//! expensive kind of diff to reason about.
//!
//! **2. You cannot perturb one domain's draw schedule without varying every other domain.** Add a
//! mutation draw and the food continuation changes too. Every experiment that changes one
//! stochastic code path silently changes all of them, and measures something other than it claims.
//!
//! # The fix
//!
//! Each domain gets its own stream, derived from the root seed by a *domain-separated* function.
//! A draw in [`RngDomain::Food`] can no longer perturb [`RngDomain::Mutation`], because they are
//! different streams. The derivation is versioned ([`RNG_DOMAIN_DERIVATION_V1`]) so that a future
//! change to it is an announced act rather than a silent re-baseline of every run in the project.
//!
//! # Why not just seed each domain with `root + n`?
//!
//! Because nearby seeds are not guaranteed to produce independent streams. `seed_from_u64`
//! applies its own mixing, so in practice `root+1` is fine — but "in practice, probably" is not
//! a basis for a science oracle. The derivation below hashes a *stable domain tag* together with
//! the root seed and a schema tag, so the streams are separated by construction and the property
//! is testable rather than hoped for.

// bd-tqpj: deterministic-simulation policy — domain tags, byte order, and
// fixed-width hashing are part of the science contract. Function lengths mirror
// the legacy C++ parity layout and are reviewed as units.
#![allow(clippy::too_many_lines)]

use crate::{AgentUid, RandomStream, RandomStreamRestoreError, RandomStreamState, SmallRngStream};
use serde::{Deserialize, Serialize};

/// Identity of the derivation. Bump this ONLY when deliberately re-deriving every domain seed —
/// doing so moves every stochastic decision in the project, so it must be announced.
pub const RNG_DOMAIN_DERIVATION_V1: &str = "scriptbots.rng-domains.v1";

/// Version of the six-domain checkpoint envelope.
pub const DOMAIN_STREAMS_CHECKPOINT_VERSION: u16 = 1;
/// Codec version for the fixed domain-state wire object.
pub const DOMAIN_STREAMS_CHECKPOINT_CODEC_VERSION: u16 = 1;
/// Identity of the stable agent/offspring keyed-substream derivation.
///
/// Bump this only when deliberately changing a field, tag, byte order, or hash step in
/// [`derive_agent_substream_seed`] or [`derive_offspring_substream_seed`]. Such a change moves
/// agent-local stochastic decisions and therefore requires an announced science re-baseline.
pub const AGENT_SUBSTREAM_DERIVATION_V1: &str = "scriptbots.agent-rng-substreams.v1";
/// Version of the agent-keyed random-substream protocol envelope.
pub const AGENT_SUBSTREAM_PROTOCOL_VERSION: u16 = 1;
/// Codec version for agent-keyed random-substream counters and identities.
pub const AGENT_SUBSTREAM_PROTOCOL_CODEC_VERSION: u16 = 1;

/// The independent domains a stochastic decision can belong to.
///
/// These are not cosmetic labels. Two decisions belong in different domains when an experiment
/// might reasonably need one domain's draw schedule not to perturb the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RngDomain {
    /// Terrain, weather, interventions — everything about the world that is not food.
    Environment,
    /// Food growth and scatter.
    Food,
    /// Spawn placement, culling, population-floor injection.
    Population,
    /// Identity and lineage decisions (which parent, which partner).
    Lineage,
    /// Mutation of inherited brains and traits.
    Mutation,
    /// Crossover between two parents.
    Crossover,
}

impl RngDomain {
    /// Every domain, in the stable derivation and digest order. It must never be reordered: the
    /// order is part of the science wire and matches the fixed checkpoint object's field order.
    pub const ALL: [Self; 6] = [
        Self::Environment,
        Self::Food,
        Self::Population,
        Self::Lineage,
        Self::Mutation,
        Self::Crossover,
    ];

    /// The STABLE STRING that participates in seed derivation.
    ///
    /// Deliberately not the enum's discriminant: adding a variant in the middle of the enum would
    /// silently re-seed every domain after it, changing every stochastic decision in the project
    /// for no reason anyone could see in the diff. A string tag is immune to that — a new domain
    /// gets a new tag and perturbs nothing that already exists.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Environment => "environment",
            Self::Food => "food",
            Self::Population => "population",
            Self::Lineage => "lineage",
            Self::Mutation => "mutation",
            Self::Crossover => "crossover",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Environment => 0,
            Self::Food => 1,
            Self::Population => 2,
            Self::Lineage => 3,
            Self::Mutation => 4,
            Self::Crossover => 5,
        }
    }
}

/// FNV-1a over the derivation tag, the domain tag, and the root seed.
///
/// FNV-1a is a SPECIFIED algorithm, pinned forever. Deliberately not `std::hash::DefaultHasher`,
/// whose algorithm std explicitly does not promise across Rust releases — using it here would let
/// a *compiler upgrade* silently re-seed every domain in the project. That exact bug was found
/// feeding the characterization digest (bd-2z0.8.4); it is not repeated here.
#[must_use]
pub fn derive_domain_seed(root_seed: u64, domain: RngDomain) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET_BASIS;
    let mut absorb = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PRIME);
        }
    };

    // Length-prefixed so that the tags cannot be confused with one another by concatenation.
    absorb(&(RNG_DOMAIN_DERIVATION_V1.len() as u64).to_le_bytes());
    absorb(RNG_DOMAIN_DERIVATION_V1.as_bytes());
    absorb(&(domain.tag().len() as u64).to_le_bytes());
    absorb(domain.tag().as_bytes());
    absorb(&root_seed.to_le_bytes());

    hash
}

const FNV1A_64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy)]
struct StableSeedHash(u64);

impl StableSeedHash {
    const fn new() -> Self {
        Self(FNV1A_64_OFFSET_BASIS)
    }

    fn absorb(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(FNV1A_64_PRIME);
        }
    }

    fn absorb_field(&mut self, bytes: &[u8]) {
        self.absorb(&(bytes.len() as u64).to_le_bytes());
        self.absorb(bytes);
    }

    fn absorb_u64(&mut self, value: u64) {
        self.absorb(&value.to_le_bytes());
    }

    const fn finish(self) -> u64 {
        self.0
    }
}

fn begin_agent_substream_derivation(
    root_seed: u64,
    subject: &str,
    domain: RngDomain,
    operation: &str,
) -> StableSeedHash {
    let mut hash = StableSeedHash::new();
    hash.absorb_field(AGENT_SUBSTREAM_DERIVATION_V1.as_bytes());
    hash.absorb_field(subject.as_bytes());
    hash.absorb_field(domain.tag().as_bytes());
    hash.absorb_field(operation.as_bytes());
    hash.absorb_u64(root_seed);
    hash
}

/// A stochastic operation performed on behalf of one existing agent.
///
/// Each variant owns a stable tag and domain. Callers cannot accidentally derive the same
/// operation under two domains, and adding a new variant does not renumber or re-seed an existing
/// operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AgentRngOperationV1 {
    /// Decide whether an otherwise eligible agent reproduces in this window.
    ReproductionAdmission,
    /// Decide whether an admitted parent uses an eligible partner.
    ReproductionPartner,
    /// Construct or reconstruct an agent-owned brain from a registered family.
    BrainInitialization,
}

impl AgentRngOperationV1 {
    /// Stable derivation tag for this operation.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::ReproductionAdmission => "reproduction-admission",
            Self::ReproductionPartner => "reproduction-partner",
            Self::BrainInitialization => "brain-initialization",
        }
    }

    /// Random domain whose scientific meaning owns this operation.
    #[must_use]
    pub const fn domain(self) -> RngDomain {
        match self {
            Self::ReproductionAdmission | Self::ReproductionPartner => RngDomain::Lineage,
            Self::BrainInitialization => RngDomain::Population,
        }
    }
}

/// A stochastic operation performed while constructing one offspring.
///
/// Offspring operations use [`OffspringRngIdentityV1`] instead of the child's eventual
/// [`AgentUid`]. Construction therefore cannot depend on a global insertion schedule that may be
/// shifted by an unrelated agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OffspringRngOperationV1 {
    /// Derive spawn geometry and other population-owned body initialization.
    BodyPopulation,
    /// Cross scalar runtime traits inherited from two parents.
    RuntimeCrossover,
    /// Mutate scalar runtime traits inherited from the primary parent.
    RuntimeMutation,
    /// Cross compatible heritable brain genomes.
    BrainCrossover,
    /// Mutate a heritable brain genome.
    BrainMutation,
    /// Construct a fallback brain when no heritable evaluator can be reused.
    BrainInitialization,
    /// Construct evaluator state using a crossover/blend policy.
    BrainEvaluatorStateCrossover,
    /// Construct evaluator state using a reset/inherit/mutation policy.
    BrainEvaluatorStateMutation,
}

impl OffspringRngOperationV1 {
    /// Stable derivation tag for this operation.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::BodyPopulation => "body-population",
            Self::RuntimeCrossover => "runtime-crossover",
            Self::RuntimeMutation => "runtime-mutation",
            Self::BrainCrossover => "brain-crossover",
            Self::BrainMutation => "brain-mutation",
            Self::BrainInitialization => "brain-initialization",
            Self::BrainEvaluatorStateCrossover => "brain-evaluator-state-crossover",
            Self::BrainEvaluatorStateMutation => "brain-evaluator-state-mutation",
        }
    }

    /// Random domain whose scientific meaning owns this operation.
    #[must_use]
    pub const fn domain(self) -> RngDomain {
        match self {
            Self::BodyPopulation | Self::BrainInitialization => RngDomain::Population,
            Self::RuntimeCrossover | Self::BrainCrossover | Self::BrainEvaluatorStateCrossover => {
                RngDomain::Crossover
            }
            Self::RuntimeMutation | Self::BrainMutation | Self::BrainEvaluatorStateMutation => {
                RngDomain::Mutation
            }
        }
    }
}

/// Stable lineage identity used to derive every random stream for one offspring.
///
/// `birth_ordinal` is local to the primary parent. It must come from that parent's persisted
/// [`AgentRngCountersV1`], not from the world's global insertion or birth sequence. Parent order
/// is intentionally directional because the primary parent supplies the base runtime/genome while
/// the optional secondary parent contributes crossover material.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(deny_unknown_fields)]
pub struct OffspringRngIdentityV1 {
    primary_parent: AgentUid,
    secondary_parent: Option<AgentUid>,
    birth_ordinal: u64,
}

impl OffspringRngIdentityV1 {
    /// Construct an offspring identity from stable lineage and a parent-local birth ordinal.
    #[must_use]
    pub const fn new(
        primary_parent: AgentUid,
        secondary_parent: Option<AgentUid>,
        birth_ordinal: u64,
    ) -> Self {
        Self {
            primary_parent,
            secondary_parent,
            birth_ordinal,
        }
    }

    /// Primary parent whose persisted counter assigned this birth ordinal.
    #[must_use]
    pub const fn primary_parent(self) -> AgentUid {
        self.primary_parent
    }

    /// Optional secondary parent contributing crossover material.
    #[must_use]
    pub const fn secondary_parent(self) -> Option<AgentUid> {
        self.secondary_parent
    }

    /// Parent-local ordinal of this successful birth.
    #[must_use]
    pub const fn birth_ordinal(self) -> u64 {
        self.birth_ordinal
    }
}

/// Persisted agent-local continuation counters for keyed random substreams.
///
/// Each value is the next unused ordinal. A counter is advanced only through the checked `take_*`
/// methods, which return the claimed ordinal and refuse to wrap. Transactional callers must
/// restore the previous value when the scientific operation itself rolls back.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AgentRngCountersV1 {
    reproduction_attempt: u64,
    birth: u64,
    brain_initialization: u64,
}

impl AgentRngCountersV1 {
    /// Reconstruct exact persisted continuation values.
    #[must_use]
    pub const fn from_ordinals(
        reproduction_attempt: u64,
        birth: u64,
        brain_initialization: u64,
    ) -> Self {
        Self {
            reproduction_attempt,
            birth,
            brain_initialization,
        }
    }

    /// Next unused reproduction-attempt ordinal.
    #[must_use]
    pub const fn reproduction_attempt_ordinal(self) -> u64 {
        self.reproduction_attempt
    }

    /// Next unused parent-local birth ordinal.
    #[must_use]
    pub const fn birth_ordinal(self) -> u64 {
        self.birth
    }

    /// Next unused brain-initialization ordinal.
    #[must_use]
    pub const fn brain_initialization_ordinal(self) -> u64 {
        self.brain_initialization
    }

    /// Claim the next reproduction-attempt ordinal.
    pub const fn take_reproduction_attempt(&mut self) -> Result<u64, AgentRngCounterError> {
        Self::take_counter(&mut self.reproduction_attempt, "reproduction-attempt")
    }

    /// Claim the next successful-birth ordinal.
    pub const fn take_birth(&mut self) -> Result<u64, AgentRngCounterError> {
        Self::take_counter(&mut self.birth, "birth")
    }

    /// Claim the next brain-initialization ordinal.
    pub const fn take_brain_initialization(&mut self) -> Result<u64, AgentRngCounterError> {
        Self::take_counter(&mut self.brain_initialization, "brain-initialization")
    }

    const fn take_counter(
        counter: &mut u64,
        counter_name: &'static str,
    ) -> Result<u64, AgentRngCounterError> {
        let claimed = *counter;
        let Some(next) = claimed.checked_add(1) else {
            return Err(AgentRngCounterError::Exhausted {
                counter: counter_name,
            });
        };
        *counter = next;
        Ok(claimed)
    }
}

/// Failure to advance a persisted agent-local random continuation.
#[derive(Debug, Clone, Copy, thiserror::Error, PartialEq, Eq)]
pub enum AgentRngCounterError {
    /// The next ordinal would wrap and reuse an earlier random identity.
    #[error("agent random counter `{counter}` is exhausted")]
    Exhausted {
        /// Stable name of the exhausted counter (`reproduction-attempt`, `birth`, or
        /// `brain-initialization`).
        counter: &'static str,
    },
}

/// Versioned metadata binding an agent-keyed protocol to its root and concrete RNG lane.
///
/// Both the derivation and [`SmallRngStream`] are cross-target stable. Restore still validates the
/// exact generator identity before consuming persisted counters: compatible historical
/// Xoshiro256++ state may resume on either target, while legacy 32-bit Xoshiro128++ state is
/// rejected rather than silently reinterpreted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AgentSubstreamProtocolV1 {
    version: u16,
    algorithm: String,
    codec_version: u16,
    stream_algorithm: String,
    root_seed: u64,
}

impl AgentSubstreamProtocolV1 {
    /// Construct the protocol metadata for one world root seed and the portable RNG lane.
    #[must_use]
    pub fn from_root_seed(root_seed: u64) -> Self {
        Self {
            version: AGENT_SUBSTREAM_PROTOCOL_VERSION,
            algorithm: AGENT_SUBSTREAM_DERIVATION_V1.to_owned(),
            codec_version: AGENT_SUBSTREAM_PROTOCOL_CODEC_VERSION,
            stream_algorithm: SmallRngStream::algorithm().to_owned(),
            root_seed,
        }
    }

    /// Protocol envelope version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Stable keyed-substream derivation identity.
    #[must_use]
    pub fn algorithm(&self) -> &str {
        &self.algorithm
    }

    /// Counter/identity object codec version.
    #[must_use]
    pub const fn codec_version(&self) -> u16 {
        self.codec_version
    }

    /// Exact target-independent concrete generator algorithm.
    #[must_use]
    pub fn stream_algorithm(&self) -> &str {
        &self.stream_algorithm
    }

    /// Root seed from which all agent and offspring substreams are derived.
    #[must_use]
    pub const fn root_seed(&self) -> u64 {
        self.root_seed
    }

    /// Validate a decoded protocol envelope against the expected world and portable RNG lane.
    pub fn validate(&self, expected_root_seed: u64) -> Result<(), AgentSubstreamProtocolError> {
        if self.version != AGENT_SUBSTREAM_PROTOCOL_VERSION {
            return Err(AgentSubstreamProtocolError::Version {
                found: self.version,
                expected: AGENT_SUBSTREAM_PROTOCOL_VERSION,
            });
        }
        if self.algorithm != AGENT_SUBSTREAM_DERIVATION_V1 {
            return Err(AgentSubstreamProtocolError::Algorithm {
                found: self.algorithm.clone(),
                expected: AGENT_SUBSTREAM_DERIVATION_V1,
            });
        }
        if self.codec_version != AGENT_SUBSTREAM_PROTOCOL_CODEC_VERSION {
            return Err(AgentSubstreamProtocolError::CodecVersion {
                found: self.codec_version,
                expected: AGENT_SUBSTREAM_PROTOCOL_CODEC_VERSION,
            });
        }
        if self.stream_algorithm != SmallRngStream::algorithm() {
            return Err(AgentSubstreamProtocolError::StreamAlgorithm {
                found: self.stream_algorithm.clone(),
                expected: SmallRngStream::algorithm(),
            });
        }
        if self.root_seed != expected_root_seed {
            return Err(AgentSubstreamProtocolError::RootSeed {
                found: self.root_seed,
                expected: expected_root_seed,
            });
        }
        Ok(())
    }
}

/// Why decoded agent-keyed random-substream metadata is incompatible.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum AgentSubstreamProtocolError {
    /// The protocol envelope version is unsupported.
    #[error("agent random-substream protocol version {found} does not match {expected}")]
    Version {
        /// Envelope version decoded from the persisted metadata.
        found: u16,
        /// Envelope version this build supports.
        expected: u16,
    },

    /// The persisted keyed derivation algorithm is unsupported.
    #[error("agent random-substream algorithm `{found}` does not match `{expected}`")]
    Algorithm {
        /// Keyed-substream derivation identity decoded from the persisted metadata.
        found: String,
        /// Derivation identity this build implements.
        expected: &'static str,
    },

    /// The persisted counter/identity codec is unsupported.
    #[error("agent random-substream codec version {found} does not match {expected}")]
    CodecVersion {
        /// Counter/identity codec version decoded from the persisted metadata.
        found: u16,
        /// Codec version this build supports.
        expected: u16,
    },

    /// The concrete generator lane does not match this compilation target.
    #[error("agent random-substream generator `{found}` does not match `{expected}`")]
    StreamAlgorithm {
        /// Concrete generator algorithm recorded in the persisted metadata.
        found: String,
        /// Generator algorithm selected by this compilation target.
        expected: &'static str,
    },

    /// The envelope belongs to a different world root seed.
    #[error("agent random-substream root seed {found} does not match {expected}")]
    RootSeed {
        /// Root seed recorded in the persisted metadata.
        found: u64,
        /// Root seed of the world being validated against.
        expected: u64,
    },
}

/// Derive one existing agent's operation-local stream seed.
///
/// The complete identity is the protocol tag, subject kind, fixed domain, stable operation tag,
/// world root seed, stable [`AgentUid`], and agent-local operation ordinal. Dense storage position,
/// [`crate::AgentId`], iteration order, and wall-clock state are absent by construction.
#[must_use]
pub fn derive_agent_substream_seed(
    root_seed: u64,
    agent_uid: AgentUid,
    operation: AgentRngOperationV1,
    ordinal: u64,
) -> u64 {
    let mut hash =
        begin_agent_substream_derivation(root_seed, "agent", operation.domain(), operation.tag());
    hash.absorb_u64(agent_uid.get());
    hash.absorb_u64(ordinal);
    hash.finish()
}

/// Construct one existing agent's isolated operation-local random stream.
#[must_use]
pub fn agent_substream(
    root_seed: u64,
    agent_uid: AgentUid,
    operation: AgentRngOperationV1,
    ordinal: u64,
) -> SmallRngStream {
    SmallRngStream::seed_from_u64(derive_agent_substream_seed(
        root_seed, agent_uid, operation, ordinal,
    ))
}

/// Derive one offspring construction operation's isolated stream seed.
///
/// The partner-presence byte makes `None` distinct from `Some(AgentUid(0))`. Ordered parent UIDs
/// and the primary parent's local birth ordinal make the offspring identity independent of the
/// child's future dense handle, global UID, or global insertion ordinal.
#[must_use]
pub fn derive_offspring_substream_seed(
    root_seed: u64,
    identity: OffspringRngIdentityV1,
    operation: OffspringRngOperationV1,
) -> u64 {
    let mut hash = begin_agent_substream_derivation(
        root_seed,
        "offspring",
        operation.domain(),
        operation.tag(),
    );
    hash.absorb_u64(identity.primary_parent().get());
    match identity.secondary_parent() {
        Some(secondary_parent) => {
            hash.absorb(&[1]);
            hash.absorb_u64(secondary_parent.get());
        }
        None => hash.absorb(&[0]),
    }
    hash.absorb_u64(identity.birth_ordinal());
    hash.finish()
}

/// Construct one offspring operation's isolated random stream.
#[must_use]
pub fn offspring_substream(
    root_seed: u64,
    identity: OffspringRngIdentityV1,
    operation: OffspringRngOperationV1,
) -> SmallRngStream {
    SmallRngStream::seed_from_u64(derive_offspring_substream_seed(
        root_seed, identity, operation,
    ))
}

/// One independent [`SmallRngStream`] per [`RngDomain`].
#[derive(Debug, Clone)]
pub struct DomainStreams {
    root_seed: u64,
    streams: [SmallRngStream; RngDomain::ALL.len()],
}

/// Serializable continuation state for every random domain in one world.
///
/// The top-level metadata identifies the domain derivation and fixed-object codec; each field then
/// carries the underlying generator's own independently versioned [`RandomStreamState`]. Keeping
/// both layers explicit prevents either a domain-remapping change or a generator-state change from
/// being mistaken for a compatible continuation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DomainStreamsCheckpoint {
    /// Envelope format version; must equal [`DOMAIN_STREAMS_CHECKPOINT_VERSION`] on restore.
    pub version: u16,
    /// Domain-separation derivation identity; must equal [`RNG_DOMAIN_DERIVATION_V1`].
    pub algorithm: String,
    /// Codec version of the fixed domain-state wire object; must equal
    /// [`DOMAIN_STREAMS_CHECKPOINT_CODEC_VERSION`].
    pub codec_version: u16,
    /// World root seed every domain stream in this checkpoint was derived from.
    pub root_seed: u64,
    /// Per-domain generator continuation states, one named field per [`RngDomain`].
    pub streams: DomainStreamStates,
}

/// Fixed wire object containing exactly one checkpoint for each random domain.
///
/// Named fields make missing and future domains a decode error instead of allowing a partially
/// populated map to reach restore. Field order is also the canonical checkpoint serialization
/// order and deliberately matches [`RngDomain::ALL`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DomainStreamStates {
    /// Environment-domain continuation state.
    pub environment: RandomStreamState,
    /// Food-domain continuation state.
    pub food: RandomStreamState,
    /// Population-domain continuation state.
    pub population: RandomStreamState,
    /// Lineage-domain continuation state.
    pub lineage: RandomStreamState,
    /// Mutation-domain continuation state.
    pub mutation: RandomStreamState,
    /// Crossover-domain continuation state.
    pub crossover: RandomStreamState,
}

impl DomainStreamsCheckpoint {
    /// Return the checkpoint for one random domain.
    #[must_use]
    pub const fn stream(&self, domain: RngDomain) -> &RandomStreamState {
        self.streams.stream(domain)
    }
}

impl DomainStreamStates {
    /// Return the checkpoint for one random domain.
    #[must_use]
    pub const fn stream(&self, domain: RngDomain) -> &RandomStreamState {
        match domain {
            RngDomain::Environment => &self.environment,
            RngDomain::Food => &self.food,
            RngDomain::Population => &self.population,
            RngDomain::Lineage => &self.lineage,
            RngDomain::Mutation => &self.mutation,
            RngDomain::Crossover => &self.crossover,
        }
    }
}

impl DomainStreams {
    /// Derive every domain stream from one root seed.
    #[must_use]
    pub fn from_root_seed(root_seed: u64) -> Self {
        let streams = RngDomain::ALL
            .map(|domain| SmallRngStream::seed_from_u64(derive_domain_seed(root_seed, domain)));
        Self { root_seed, streams }
    }

    /// The root seed these streams were derived from.
    #[must_use]
    pub const fn root_seed(&self) -> u64 {
        self.root_seed
    }

    /// The stream for a domain.
    ///
    /// Infallible by construction: `from_root_seed` populates every variant of `RngDomain::ALL`,
    /// and the fixed checkpoint wire requires every domain before `restore` can run. There is
    /// deliberately no `Option` here — a caller forced to handle "this domain has no stream"
    /// would have nothing sensible to do but fall back to some other domain's stream, which is
    /// precisely the coupling this module exists to prevent.
    ///
    /// # Panics
    ///
    /// Never: `RngDomain::index` yields a lane index below `streams.len()` for every variant, so
    /// the indexing cannot go out of bounds.
    pub const fn stream(&mut self, domain: RngDomain) -> &mut SmallRngStream {
        &mut self.streams[domain.index()]
    }

    /// Capture a restorable checkpoint of every domain.
    ///
    /// # Panics
    ///
    /// Never: every lane is reached through `RngDomain::index`, which is in bounds by
    /// construction.
    #[must_use]
    pub fn checkpoint(&self) -> DomainStreamsCheckpoint {
        let checkpoint = |domain: RngDomain| self.streams[domain.index()].checkpoint();
        DomainStreamsCheckpoint {
            version: DOMAIN_STREAMS_CHECKPOINT_VERSION,
            algorithm: RNG_DOMAIN_DERIVATION_V1.to_owned(),
            codec_version: DOMAIN_STREAMS_CHECKPOINT_CODEC_VERSION,
            root_seed: self.root_seed,
            streams: DomainStreamStates {
                environment: checkpoint(RngDomain::Environment),
                food: checkpoint(RngDomain::Food),
                population: checkpoint(RngDomain::Population),
                lineage: checkpoint(RngDomain::Lineage),
                mutation: checkpoint(RngDomain::Mutation),
                crossover: checkpoint(RngDomain::Crossover),
            },
        }
    }

    /// Restore from a checkpoint.
    ///
    /// The fixed-field wire type guarantees every domain is present before this method can be
    /// called. Restore then validates the envelope, every generator state, and every embedded
    /// derived seed before returning any live stream.
    pub fn restore(checkpoint: &DomainStreamsCheckpoint) -> Result<Self, DomainStreamRestoreError> {
        if checkpoint.version != DOMAIN_STREAMS_CHECKPOINT_VERSION {
            return Err(DomainStreamRestoreError::Version {
                found: checkpoint.version,
                expected: DOMAIN_STREAMS_CHECKPOINT_VERSION,
            });
        }
        if checkpoint.algorithm != RNG_DOMAIN_DERIVATION_V1 {
            return Err(DomainStreamRestoreError::Algorithm {
                found: checkpoint.algorithm.clone(),
                expected: RNG_DOMAIN_DERIVATION_V1,
            });
        }
        if checkpoint.codec_version != DOMAIN_STREAMS_CHECKPOINT_CODEC_VERSION {
            return Err(DomainStreamRestoreError::CodecVersion {
                found: checkpoint.codec_version,
                expected: DOMAIN_STREAMS_CHECKPOINT_CODEC_VERSION,
            });
        }
        let restore_domain =
            |domain: RngDomain| -> Result<SmallRngStream, DomainStreamRestoreError> {
                let state = checkpoint.streams.stream(domain);
                let stream = SmallRngStream::from_state(state).map_err(|source| {
                    DomainStreamRestoreError::Stream {
                        domain: domain.tag(),
                        source,
                    }
                })?;
                let expected_seed = derive_domain_seed(checkpoint.root_seed, domain);
                if stream.seed() != expected_seed {
                    return Err(DomainStreamRestoreError::DerivedSeedMismatch {
                        domain: domain.tag(),
                        found: stream.seed(),
                        expected: expected_seed,
                    });
                }
                Ok(stream)
            };
        let streams = [
            restore_domain(RngDomain::Environment)?,
            restore_domain(RngDomain::Food)?,
            restore_domain(RngDomain::Population)?,
            restore_domain(RngDomain::Lineage)?,
            restore_domain(RngDomain::Mutation)?,
            restore_domain(RngDomain::Crossover)?,
        ];
        Ok(Self {
            streams,
            root_seed: checkpoint.root_seed,
        })
    }
}

/// Why a set of persisted domain states could not be restored.
///
/// Domain-specific variants name the affected domain. A stream failure that said only "a stream
/// was bad" would leave the reader unable to tell whether their mutations, food, or lineage
/// decisions were the ones that could not be resumed, and those have very different consequences
/// for a run.
#[derive(Debug, thiserror::Error)]
pub enum DomainStreamRestoreError {
    /// The checkpoint envelope version is unsupported.
    #[error("random-domain checkpoint version {found} does not match supported version {expected}")]
    Version {
        /// Envelope version found in the decoded checkpoint.
        found: u16,
        /// Envelope version this build supports.
        expected: u16,
    },

    /// The checkpoint was derived under a different domain-separation contract.
    #[error("random-domain checkpoint algorithm `{found}` does not match `{expected}`")]
    Algorithm {
        /// Domain derivation identity recorded in the checkpoint.
        found: String,
        /// Domain derivation identity this build implements.
        expected: &'static str,
    },

    /// The checkpoint state-object codec is unsupported.
    #[error(
        "random-domain checkpoint codec version {found} does not match supported version {expected}"
    )]
    CodecVersion {
        /// State-object codec version recorded in the checkpoint.
        found: u16,
        /// State-object codec version this build supports.
        expected: u16,
    },

    /// A stream state claims it belongs to a different root/domain derivation.
    #[error("the `{domain}` domain's embedded seed {found} does not match derived seed {expected}")]
    DerivedSeedMismatch {
        /// Stable tag of the domain whose stream failed seed validation.
        domain: &'static str,
        /// Seed embedded in the persisted stream state.
        found: u64,
        /// Seed re-derived from the checkpoint's root seed for this domain.
        expected: u64,
    },

    /// The domain's state was present but could not be restored.
    #[error("the `{domain}` domain's random stream could not be restored: {source}")]
    Stream {
        /// Stable tag of the domain whose state could not be restored.
        domain: &'static str,
        /// Underlying generator-level restore failure.
        #[source]
        source: RandomStreamRestoreError,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn a_draw_in_one_domain_does_not_disturb_another() {
        // THE ENTIRE POINT OF THIS MODULE.
        //
        // With a single shared stream, adding one draw to the food logic shifts every mutation,
        // every spawn position, and every reproduction roll for the rest of the run. Here, the
        // food domain is drawn from HARD, and the mutation domain must not notice.
        let mut untouched = DomainStreams::from_root_seed(4242);
        let mut hammered = DomainStreams::from_root_seed(4242);

        for _ in 0..1000 {
            let _ = hammered.stream(RngDomain::Food).next_u64();
        }

        let expected: Vec<u64> = (0..16)
            .map(|_| untouched.stream(RngDomain::Mutation).next_u64())
            .collect();
        let actual: Vec<u64> = (0..16)
            .map(|_| hammered.stream(RngDomain::Mutation).next_u64())
            .collect();

        assert_eq!(
            actual, expected,
            "a thousand draws from the FOOD domain changed what the MUTATION domain produces. \
             The streams are not independent, so adding a single draw anywhere in the simulator \
             would still shift every stochastic decision downstream of it — and no experiment \
             could hold one factor fixed while varying another."
        );
    }

    #[test]
    fn every_domain_gets_a_distinct_stream() {
        // Anti-vacuity for the test above: if all six domains were secretly the SAME stream,
        // independence would be trivially violated; if they were all seeded identically, the
        // "separation" would be a fiction that happens to pass the first test only because both
        // sides move together.
        let seeds: Vec<u64> = RngDomain::ALL
            .into_iter()
            .map(|domain| derive_domain_seed(7, domain))
            .collect();

        let mut unique = seeds.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            seeds.len(),
            "two domains derived the SAME seed from the same root. They would produce identical \
             streams, and 'domain separation' would be a label rather than a property: {seeds:?}"
        );

        // And the first draws must actually differ — equal seeds are the obvious failure, but
        // equal *output* is the one that matters.
        let mut streams = DomainStreams::from_root_seed(7);
        let firsts: Vec<u64> = RngDomain::ALL
            .into_iter()
            .map(|domain| streams.stream(domain).next_u64())
            .collect();
        let mut unique_firsts = firsts.clone();
        unique_firsts.sort_unstable();
        unique_firsts.dedup();
        assert_eq!(
            unique_firsts.len(),
            firsts.len(),
            "two domains produced the same first draw: {firsts:?}"
        );
    }

    #[test]
    fn derivation_is_a_pure_function_of_the_root_seed() {
        // Reproducibility rests on this. If derivation mixed in ANY ambient state — a clock, an
        // address, an iteration order — no run in this project could be repeated.
        for domain in RngDomain::ALL {
            assert_eq!(
                derive_domain_seed(99, domain),
                derive_domain_seed(99, domain),
                "derivation is not deterministic for {domain:?}"
            );
        }

        let mut left = DomainStreams::from_root_seed(1234);
        let mut right = DomainStreams::from_root_seed(1234);
        for domain in RngDomain::ALL {
            for draw in 0..8 {
                assert_eq!(
                    left.stream(domain).next_u64(),
                    right.stream(domain).next_u64(),
                    "two DomainStreams from the same root seed diverged in {domain:?} at draw \
                     {draw}: seeding is not a pure function of the seed"
                );
            }
        }
    }

    #[test]
    fn a_different_root_seed_moves_every_domain() {
        // The other half of the anti-vacuity guard: if the root seed were ignored, every test
        // above would still pass while every "independent replicate" in the lab was secretly the
        // same run.
        for domain in RngDomain::ALL {
            assert_ne!(
                derive_domain_seed(1, domain),
                derive_domain_seed(2, domain),
                "domain {domain:?} derives the same seed from two DIFFERENT root seeds — the root \
                 seed is being ignored, and every replicate is the same run"
            );
        }
    }

    #[test]
    fn a_domain_tag_is_stable_and_not_the_enum_discriminant() {
        // Seeds are derived from the TAG, never from the variant's position. If someone inserts a
        // new domain into the middle of the enum tomorrow, every existing domain must keep its
        // seed — otherwise a purely additive change would silently re-seed the whole simulator.
        assert_eq!(RngDomain::Environment.tag(), "environment");
        assert_eq!(RngDomain::Food.tag(), "food");
        assert_eq!(RngDomain::Population.tag(), "population");
        assert_eq!(RngDomain::Lineage.tag(), "lineage");
        assert_eq!(RngDomain::Mutation.tag(), "mutation");
        assert_eq!(RngDomain::Crossover.tag(), "crossover");

        // PINNED SEEDS. These are the derivation's golden values. If they move, every stochastic
        // decision in the project moves with them — so the move must be a deliberate, announced
        // act (bump RNG_DOMAIN_DERIVATION_V1), never a side effect.
        assert_eq!(
            derive_domain_seed(0, RngDomain::Environment),
            0xe935_ed64_0958_5b1b
        );
        assert_eq!(
            derive_domain_seed(0, RngDomain::Mutation),
            0xe734_d3a0_3c32_070a
        );
    }

    #[test]
    fn keyed_substream_tags_domains_and_golden_seeds_are_stable() {
        assert_eq!(
            AGENT_SUBSTREAM_DERIVATION_V1,
            "scriptbots.agent-rng-substreams.v1"
        );
        assert_eq!(AGENT_SUBSTREAM_PROTOCOL_VERSION, 1);
        assert_eq!(AGENT_SUBSTREAM_PROTOCOL_CODEC_VERSION, 1);

        assert_eq!(
            AgentRngOperationV1::ReproductionAdmission.tag(),
            "reproduction-admission"
        );
        assert_eq!(
            AgentRngOperationV1::ReproductionAdmission.domain(),
            RngDomain::Lineage
        );
        assert_eq!(
            AgentRngOperationV1::ReproductionPartner.tag(),
            "reproduction-partner"
        );
        assert_eq!(
            AgentRngOperationV1::ReproductionPartner.domain(),
            RngDomain::Lineage
        );
        assert_eq!(
            AgentRngOperationV1::BrainInitialization.tag(),
            "brain-initialization"
        );
        assert_eq!(
            AgentRngOperationV1::BrainInitialization.domain(),
            RngDomain::Population
        );

        let offspring_operations = [
            (
                OffspringRngOperationV1::BodyPopulation,
                "body-population",
                RngDomain::Population,
            ),
            (
                OffspringRngOperationV1::RuntimeCrossover,
                "runtime-crossover",
                RngDomain::Crossover,
            ),
            (
                OffspringRngOperationV1::RuntimeMutation,
                "runtime-mutation",
                RngDomain::Mutation,
            ),
            (
                OffspringRngOperationV1::BrainCrossover,
                "brain-crossover",
                RngDomain::Crossover,
            ),
            (
                OffspringRngOperationV1::BrainMutation,
                "brain-mutation",
                RngDomain::Mutation,
            ),
            (
                OffspringRngOperationV1::BrainInitialization,
                "brain-initialization",
                RngDomain::Population,
            ),
            (
                OffspringRngOperationV1::BrainEvaluatorStateCrossover,
                "brain-evaluator-state-crossover",
                RngDomain::Crossover,
            ),
            (
                OffspringRngOperationV1::BrainEvaluatorStateMutation,
                "brain-evaluator-state-mutation",
                RngDomain::Mutation,
            ),
        ];
        for (operation, tag, domain) in offspring_operations {
            assert_eq!(operation.tag(), tag);
            assert_eq!(operation.domain(), domain);
        }

        let root_seed = 0x0123_4567_89ab_cdef;
        assert_eq!(
            derive_agent_substream_seed(
                root_seed,
                AgentUid(42),
                AgentRngOperationV1::ReproductionAdmission,
                7,
            ),
            0x4a62_e9d3_894c_91c1
        );
        assert_eq!(
            derive_agent_substream_seed(
                root_seed,
                AgentUid(42),
                AgentRngOperationV1::ReproductionPartner,
                7,
            ),
            0x6c02_cda0_3a2f_2928
        );

        let identity = OffspringRngIdentityV1::new(AgentUid(42), Some(AgentUid(99)), 3);
        assert_eq!(
            derive_offspring_substream_seed(
                root_seed,
                identity,
                OffspringRngOperationV1::BrainMutation,
            ),
            0xd997_56b8_9852_5f01
        );
    }

    #[test]
    fn agent_substream_derivation_uses_uid_operation_and_local_ordinal() {
        let root_seed = 8181;
        let baseline = derive_agent_substream_seed(
            root_seed,
            AgentUid(12),
            AgentRngOperationV1::ReproductionAdmission,
            4,
        );
        assert_eq!(
            baseline,
            derive_agent_substream_seed(
                root_seed,
                AgentUid(12),
                AgentRngOperationV1::ReproductionAdmission,
                4,
            ),
            "the same stable identity did not reproduce the same local seed"
        );
        assert_ne!(
            baseline,
            derive_agent_substream_seed(
                root_seed,
                AgentUid(13),
                AgentRngOperationV1::ReproductionAdmission,
                4,
            ),
            "AgentUid is absent from the keyed derivation"
        );
        assert_ne!(
            baseline,
            derive_agent_substream_seed(
                root_seed,
                AgentUid(12),
                AgentRngOperationV1::ReproductionAdmission,
                5,
            ),
            "the persisted local ordinal is absent from the keyed derivation"
        );
        assert_ne!(
            baseline,
            derive_agent_substream_seed(
                root_seed,
                AgentUid(12),
                AgentRngOperationV1::ReproductionPartner,
                4,
            ),
            "two lineage operations share a seed despite their distinct stable tags"
        );
        assert_ne!(
            baseline,
            derive_agent_substream_seed(
                root_seed,
                AgentUid(12),
                AgentRngOperationV1::BrainInitialization,
                4,
            ),
            "the operation's fixed domain is absent from the keyed derivation"
        );
    }

    #[test]
    fn hammering_an_unrelated_agent_cannot_perturb_an_existing_agent_stream() {
        let root_seed = 7331;
        let target_uid = AgentUid(41);
        let operation = AgentRngOperationV1::ReproductionAdmission;
        let ordinal = 9;

        let mut untouched = agent_substream(root_seed, target_uid, operation, ordinal);
        let expected: Vec<u64> = (0..16).map(|_| untouched.next_u64()).collect();

        for unrelated_ordinal in 0..1000 {
            let mut unrelated = agent_substream(
                root_seed,
                AgentUid(9001),
                AgentRngOperationV1::ReproductionAdmission,
                unrelated_ordinal,
            );
            for _ in 0..8 {
                let _ = unrelated.next_u64();
            }
        }

        let mut after_unrelated_work = agent_substream(root_seed, target_uid, operation, ordinal);
        let actual: Vec<u64> = (0..16).map(|_| after_unrelated_work.next_u64()).collect();
        assert_eq!(
            actual, expected,
            "drawing from a distant agent changed an existing agent's local continuation"
        );
    }

    #[test]
    fn dense_permutation_cannot_change_agent_keyed_seeds() {
        let root_seed = 771;
        let operation = AgentRngOperationV1::ReproductionPartner;
        let canonical_order = [AgentUid(5), AgentUid(90), AgentUid(17), AgentUid(44)];
        let permuted_order = [AgentUid(44), AgentUid(17), AgentUid(5), AgentUid(90)];

        let derive_and_sort = |uids: &[AgentUid]| {
            let mut keyed: Vec<(u64, u64)> = uids
                .iter()
                .map(|uid| {
                    (
                        uid.get(),
                        derive_agent_substream_seed(root_seed, *uid, operation, 2),
                    )
                })
                .collect();
            keyed.sort_unstable_by_key(|(uid, _)| *uid);
            keyed
        };

        assert_eq!(
            derive_and_sort(&canonical_order),
            derive_and_sort(&permuted_order),
            "reordering dense storage changed seeds even though every stable AgentUid was unchanged"
        );
    }

    #[test]
    fn offspring_identity_is_lineage_order_and_parent_local_birth() {
        let root_seed = 616;
        let baseline_identity = OffspringRngIdentityV1::new(AgentUid(11), Some(AgentUid(29)), 6);
        let baseline = derive_offspring_substream_seed(
            root_seed,
            baseline_identity,
            OffspringRngOperationV1::RuntimeMutation,
        );

        assert_ne!(
            baseline,
            derive_offspring_substream_seed(
                root_seed,
                OffspringRngIdentityV1::new(AgentUid(11), Some(AgentUid(29)), 7),
                OffspringRngOperationV1::RuntimeMutation,
            ),
            "the primary parent's local birth ordinal is absent"
        );
        assert_ne!(
            baseline,
            derive_offspring_substream_seed(
                root_seed,
                OffspringRngIdentityV1::new(AgentUid(29), Some(AgentUid(11)), 6),
                OffspringRngOperationV1::RuntimeMutation,
            ),
            "ordered lineage was collapsed into an unordered parent set"
        );
        assert_ne!(
            baseline,
            derive_offspring_substream_seed(
                root_seed,
                OffspringRngIdentityV1::new(AgentUid(11), None, 6),
                OffspringRngOperationV1::RuntimeMutation,
            ),
            "the optional secondary parent is absent"
        );
        assert_ne!(
            derive_offspring_substream_seed(
                root_seed,
                OffspringRngIdentityV1::new(AgentUid(11), None, 6),
                OffspringRngOperationV1::RuntimeMutation,
            ),
            derive_offspring_substream_seed(
                root_seed,
                OffspringRngIdentityV1::new(AgentUid(11), Some(AgentUid(0)), 6),
                OffspringRngOperationV1::RuntimeMutation,
            ),
            "partner absence collided with a present zero-valued AgentUid"
        );
        assert_ne!(
            baseline,
            derive_offspring_substream_seed(
                root_seed,
                baseline_identity,
                OffspringRngOperationV1::BrainMutation,
            ),
            "distinct offspring operations share one seed"
        );

        let agent_brain_seed = derive_agent_substream_seed(
            root_seed,
            AgentUid(11),
            AgentRngOperationV1::BrainInitialization,
            6,
        );
        let offspring_brain_seed = derive_offspring_substream_seed(
            root_seed,
            OffspringRngIdentityV1::new(AgentUid(11), None, 6),
            OffspringRngOperationV1::BrainInitialization,
        );
        assert_ne!(
            agent_brain_seed, offspring_brain_seed,
            "agent and offspring subjects collided despite separate subject tags"
        );
    }

    #[test]
    fn offspring_identity_wire_is_strict_and_idempotent() {
        let identity = OffspringRngIdentityV1::new(AgentUid(8), Some(AgentUid(13)), 21);
        let encoded = serde_json::to_string(&identity).expect("offspring identity encodes");
        let decoded: OffspringRngIdentityV1 =
            serde_json::from_str(&encoded).expect("offspring identity decodes");
        assert_eq!(decoded, identity);
        assert_eq!(
            serde_json::to_string(&decoded).expect("decoded identity re-encodes"),
            encoded
        );

        let mut missing = serde_json::to_value(identity).expect("offspring identity encodes");
        missing
            .as_object_mut()
            .expect("offspring identity is an object")
            .remove("birth_ordinal");
        assert!(
            serde_json::from_value::<OffspringRngIdentityV1>(missing).is_err(),
            "an offspring identity missing its parent-local ordinal decoded"
        );

        let mut unknown = serde_json::to_value(identity).expect("offspring identity encodes");
        unknown
            .as_object_mut()
            .expect("offspring identity is an object")
            .insert("child_uid".to_owned(), serde_json::json!(999));
        assert!(
            serde_json::from_value::<OffspringRngIdentityV1>(unknown).is_err(),
            "a future child UID was accepted into the lineage-derived identity"
        );
    }

    #[test]
    fn agent_rng_counters_advance_independently_and_never_wrap() {
        let mut counters = AgentRngCountersV1::default();
        assert_eq!(counters.take_reproduction_attempt(), Ok(0));
        assert_eq!(counters.take_reproduction_attempt(), Ok(1));
        assert_eq!(counters.take_birth(), Ok(0));
        assert_eq!(counters.take_brain_initialization(), Ok(0));
        assert_eq!(counters.reproduction_attempt_ordinal(), 2);
        assert_eq!(counters.birth_ordinal(), 1);
        assert_eq!(counters.brain_initialization_ordinal(), 1);

        let mut exhausted = AgentRngCountersV1::from_ordinals(u64::MAX, 4, 5);
        let before = exhausted;
        assert_eq!(
            exhausted.take_reproduction_attempt(),
            Err(AgentRngCounterError::Exhausted {
                counter: "reproduction-attempt",
            })
        );
        assert_eq!(
            exhausted, before,
            "counter exhaustion mutated another continuation or wrapped the exhausted one"
        );
    }

    #[test]
    fn agent_rng_counter_wire_rejects_missing_unknown_and_duplicate_fields() {
        let counters = AgentRngCountersV1::from_ordinals(3, 4, 5);
        let encoded = serde_json::to_string(&counters).expect("agent counters encode");
        let decoded: AgentRngCountersV1 =
            serde_json::from_str(&encoded).expect("agent counters decode");
        assert_eq!(decoded, counters);
        assert_eq!(
            serde_json::to_string(&decoded).expect("decoded counters re-encode"),
            encoded
        );

        let mut missing = serde_json::to_value(counters).expect("agent counters encode");
        missing
            .as_object_mut()
            .expect("agent counters are an object")
            .remove("birth");
        assert!(
            serde_json::from_value::<AgentRngCountersV1>(missing).is_err(),
            "agent counters missing the birth continuation decoded"
        );

        let mut unknown = serde_json::to_value(counters).expect("agent counters encode");
        unknown
            .as_object_mut()
            .expect("agent counters are an object")
            .insert("dense_index".to_owned(), serde_json::json!(7));
        assert!(
            serde_json::from_value::<AgentRngCountersV1>(unknown).is_err(),
            "a dense-index continuation was accepted by the stable counter wire"
        );

        let birth_entry = "\"birth\":4";
        let duplicate_birth = format!("{birth_entry},{birth_entry}");
        let duplicated = encoded.replacen(birth_entry, &duplicate_birth, 1);
        assert_ne!(
            duplicated, encoded,
            "duplicate-counter fixture did not locate the birth field"
        );
        let error = serde_json::from_str::<AgentRngCountersV1>(&duplicated)
            .expect_err("duplicate birth counters must not decode");
        assert!(
            error.to_string().contains("duplicate field `birth`"),
            "duplicate-counter error did not name the repeated continuation: {error}"
        );
    }

    #[test]
    fn agent_substream_protocol_metadata_is_strictly_versioned() {
        let protocol = AgentSubstreamProtocolV1::from_root_seed(910);
        assert_eq!(protocol.version(), AGENT_SUBSTREAM_PROTOCOL_VERSION);
        assert_eq!(protocol.algorithm(), AGENT_SUBSTREAM_DERIVATION_V1);
        assert_eq!(
            protocol.codec_version(),
            AGENT_SUBSTREAM_PROTOCOL_CODEC_VERSION
        );
        assert_eq!(protocol.stream_algorithm(), SmallRngStream::algorithm());
        assert_eq!(protocol.root_seed(), 910);
        assert_eq!(protocol.validate(910), Ok(()));

        let encoded = serde_json::to_string(&protocol).expect("protocol metadata encodes");
        let decoded: AgentSubstreamProtocolV1 =
            serde_json::from_str(&encoded).expect("protocol metadata decodes");
        assert_eq!(decoded, protocol);
        assert_eq!(
            serde_json::to_string(&decoded).expect("protocol metadata re-encodes"),
            encoded
        );

        let mut wrong_version = protocol.clone();
        wrong_version.version += 1;
        assert!(matches!(
            wrong_version.validate(910),
            Err(AgentSubstreamProtocolError::Version { .. })
        ));

        let mut wrong_algorithm = protocol.clone();
        wrong_algorithm.algorithm = "other".to_owned();
        assert!(matches!(
            wrong_algorithm.validate(910),
            Err(AgentSubstreamProtocolError::Algorithm { .. })
        ));

        let mut wrong_codec = protocol.clone();
        wrong_codec.codec_version += 1;
        assert!(matches!(
            wrong_codec.validate(910),
            Err(AgentSubstreamProtocolError::CodecVersion { .. })
        ));

        let mut wrong_stream = protocol.clone();
        wrong_stream.stream_algorithm =
            "rand-0.9.5-smallrng-xoshiro128plusplus-32-seed-from-u64".to_owned();
        assert!(matches!(
            wrong_stream.validate(910),
            Err(AgentSubstreamProtocolError::StreamAlgorithm { .. })
        ));

        assert!(matches!(
            protocol.validate(911),
            Err(AgentSubstreamProtocolError::RootSeed { .. })
        ));

        let mut unknown = serde_json::to_value(protocol).expect("protocol metadata encodes");
        unknown
            .as_object_mut()
            .expect("protocol metadata is an object")
            .insert("target_pointer_width".to_owned(), serde_json::json!(64));
        assert!(
            serde_json::from_value::<AgentSubstreamProtocolV1>(unknown).is_err(),
            "unknown target metadata was silently accepted by the frozen envelope"
        );
    }

    #[test]
    fn keyed_substreams_record_the_same_portable_generator_lane_on_every_target() {
        let root_seed = 2026;
        let agent_seed = derive_agent_substream_seed(
            root_seed,
            AgentUid(55),
            AgentRngOperationV1::BrainInitialization,
            0,
        );
        let agent = agent_substream(
            root_seed,
            AgentUid(55),
            AgentRngOperationV1::BrainInitialization,
            0,
        );
        assert_eq!(agent.seed(), agent_seed);
        assert_eq!(agent.algorithm_id(), SmallRngStream::algorithm());

        let identity = OffspringRngIdentityV1::new(AgentUid(55), None, 0);
        let offspring_seed = derive_offspring_substream_seed(
            root_seed,
            identity,
            OffspringRngOperationV1::BrainEvaluatorStateMutation,
        );
        let offspring = offspring_substream(
            root_seed,
            identity,
            OffspringRngOperationV1::BrainEvaluatorStateMutation,
        );
        assert_eq!(offspring.seed(), offspring_seed);
        assert_eq!(offspring.algorithm_id(), SmallRngStream::algorithm());
    }

    #[test]
    fn a_checkpoint_restores_every_domain_exactly() {
        // A resumed run must be the run it claims to continue — in EVERY domain, not just the
        // ones that happen to be checked.
        let mut original = DomainStreams::from_root_seed(555);
        for domain in RngDomain::ALL {
            for _ in 0..(3 + domain as usize) {
                let _ = original.stream(domain).next_u64();
            }
        }

        let checkpoint = original.checkpoint();
        let expected: [Vec<u64>; RngDomain::ALL.len()] = RngDomain::ALL
            .map(|domain| (0..8).map(|_| original.stream(domain).next_u64()).collect());

        let mut restored =
            DomainStreams::restore(&checkpoint).expect("a checkpoint we just took must restore");
        for domain in RngDomain::ALL {
            let draws: Vec<u64> = (0..8).map(|_| restored.stream(domain).next_u64()).collect();
            assert_eq!(
                &draws,
                &expected[domain.index()],
                "domain {domain:?} diverged after restore — the resumed run is not the run it \
                 claims to continue"
            );
        }
    }

    #[test]
    fn a_checkpoint_missing_a_domain_is_rejected_during_decode() {
        // The dangerous failure. Silently re-deriving a missing domain from the root seed would
        // look like a successful restore while REWINDING that domain to tick zero — and the other
        // domains would match perfectly, so nothing downstream would necessarily notice.
        let checkpoint = DomainStreams::from_root_seed(9).checkpoint();
        let mut encoded = serde_json::to_value(checkpoint).expect("checkpoint encodes as JSON");
        encoded["streams"]
            .as_object_mut()
            .expect("domain streams encode as an object")
            .remove(RngDomain::Mutation.tag())
            .expect("mutation field is present in a complete checkpoint");
        let error = serde_json::from_value::<DomainStreamsCheckpoint>(encoded)
            .expect_err("a checkpoint without the mutation field must not decode");

        assert!(
            error.to_string().contains("mutation"),
            "a checkpoint with NO mutation stream was decoded. That domain would have been \
             silently rewound to its initial state while every other domain resumed correctly — \
             a resumed run that quietly re-rolls its mutations is not the run it claims to be: \
             {error}"
        );
    }

    #[test]
    fn a_checkpoint_with_an_unexpected_domain_is_rejected_during_decode() {
        let checkpoint = DomainStreams::from_root_seed(9).checkpoint();
        let mut encoded = serde_json::to_value(checkpoint).expect("checkpoint encodes as JSON");
        let future_state = encoded["streams"][RngDomain::Environment.tag()].clone();
        encoded["streams"]
            .as_object_mut()
            .expect("domain streams encode as an object")
            .insert("future-domain".to_owned(), future_state);
        let error = serde_json::from_value::<DomainStreamsCheckpoint>(encoded)
            .expect_err("a checkpoint with an unknown domain must not decode");

        assert!(
            error.to_string().contains("future-domain"),
            "unexpected domain decode failure did not name the rejected field: {error}"
        );
    }

    #[test]
    fn a_checkpoint_with_a_duplicate_domain_is_rejected_during_decode() {
        let checkpoint = DomainStreams::from_root_seed(9).checkpoint();
        let encoded = serde_json::to_string(&checkpoint).expect("checkpoint encodes as JSON");
        let food_entry = format!(
            "\"food\":{}",
            serde_json::to_string(&checkpoint.streams.food)
                .expect("Food-domain checkpoint encodes as JSON")
        );
        let duplicate_food = format!("{food_entry},{food_entry}");
        let duplicated = encoded.replacen(&food_entry, &duplicate_food, 1);
        assert_ne!(
            duplicated, encoded,
            "duplicate-field fixture failed to locate the Food-domain field"
        );

        let error = serde_json::from_str::<DomainStreamsCheckpoint>(&duplicated)
            .expect_err("a checkpoint with duplicate Food fields must not decode");
        assert!(
            error.to_string().contains("duplicate field `food`"),
            "duplicate-domain decode failure did not identify the repeated field: {error}"
        );
    }

    #[test]
    fn checkpoint_envelope_metadata_is_strictly_versioned() {
        let original = DomainStreams::from_root_seed(9);

        let mut wrong_version = original.checkpoint();
        wrong_version.version += 1;
        assert!(matches!(
            DomainStreams::restore(&wrong_version),
            Err(DomainStreamRestoreError::Version { .. })
        ));

        let mut wrong_algorithm = original.checkpoint();
        wrong_algorithm.algorithm = "other".to_owned();
        assert!(matches!(
            DomainStreams::restore(&wrong_algorithm),
            Err(DomainStreamRestoreError::Algorithm { .. })
        ));

        let mut wrong_codec = original.checkpoint();
        wrong_codec.codec_version += 1;
        assert!(matches!(
            DomainStreams::restore(&wrong_codec),
            Err(DomainStreamRestoreError::CodecVersion { .. })
        ));

        let legacy_wasm_algorithm = "rand-0.9.5-smallrng-xoshiro128plusplus-32-seed-from-u64";
        let mut legacy_wasm = original.checkpoint();
        legacy_wasm.streams.food.algorithm = legacy_wasm_algorithm.to_owned();
        assert!(matches!(
            DomainStreams::restore(&legacy_wasm),
            Err(DomainStreamRestoreError::Stream {
                domain: "food",
                source: RandomStreamRestoreError::UnsupportedAlgorithm {
                    ref found,
                    expected,
                },
            }) if found == legacy_wasm_algorithm && expected == SmallRngStream::algorithm()
        ));
    }

    #[test]
    fn checkpoint_rejects_a_stream_derived_from_another_root() {
        let mut checkpoint = DomainStreams::from_root_seed(9).checkpoint();
        checkpoint.streams.food =
            SmallRngStream::seed_from_u64(derive_domain_seed(10, RngDomain::Food)).checkpoint();

        assert!(matches!(
            DomainStreams::restore(&checkpoint),
            Err(DomainStreamRestoreError::DerivedSeedMismatch { domain: "food", .. })
        ));
    }
}
