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

use crate::{RandomStream, RandomStreamRestoreError, RandomStreamState, SmallRngStream};
use serde::{Deserialize, Serialize};

/// Identity of the derivation. Bump this ONLY when deliberately re-deriving every domain seed —
/// doing so moves every stochastic decision in the project, so it must be announced.
pub const RNG_DOMAIN_DERIVATION_V1: &str = "scriptbots.rng-domains.v1";

/// Version of the six-domain checkpoint envelope.
pub const DOMAIN_STREAMS_CHECKPOINT_VERSION: u16 = 1;
/// Codec version for the fixed domain-state wire object.
pub const DOMAIN_STREAMS_CHECKPOINT_CODEC_VERSION: u16 = 1;

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
    pub const ALL: [RngDomain; 6] = [
        RngDomain::Environment,
        RngDomain::Food,
        RngDomain::Population,
        RngDomain::Lineage,
        RngDomain::Mutation,
        RngDomain::Crossover,
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
            RngDomain::Environment => "environment",
            RngDomain::Food => "food",
            RngDomain::Population => "population",
            RngDomain::Lineage => "lineage",
            RngDomain::Mutation => "mutation",
            RngDomain::Crossover => "crossover",
        }
    }

    const fn index(self) -> usize {
        match self {
            RngDomain::Environment => 0,
            RngDomain::Food => 1,
            RngDomain::Population => 2,
            RngDomain::Lineage => 3,
            RngDomain::Mutation => 4,
            RngDomain::Crossover => 5,
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
    pub version: u16,
    pub algorithm: String,
    pub codec_version: u16,
    pub root_seed: u64,
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
        Self { streams, root_seed }
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
    pub fn stream(&mut self, domain: RngDomain) -> &mut SmallRngStream {
        &mut self.streams[domain.index()]
    }

    /// Capture a restorable checkpoint of every domain.
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
    Version { found: u16, expected: u16 },

    /// The checkpoint was derived under a different domain-separation contract.
    #[error("random-domain checkpoint algorithm `{found}` does not match `{expected}`")]
    Algorithm {
        found: String,
        expected: &'static str,
    },

    /// The checkpoint state-object codec is unsupported.
    #[error(
        "random-domain checkpoint codec version {found} does not match supported version {expected}"
    )]
    CodecVersion { found: u16, expected: u16 },

    /// A stream state claims it belongs to a different root/domain derivation.
    #[error("the `{domain}` domain's embedded seed {found} does not match derived seed {expected}")]
    DerivedSeedMismatch {
        domain: &'static str,
        found: u64,
        expected: u64,
    },

    /// The domain's state was present but could not be restored.
    #[error("the `{domain}` domain's random stream could not be restored: {source}")]
    Stream {
        domain: &'static str,
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
        let expected: [Vec<u64>; RngDomain::ALL.len()] = RngDomain::ALL.map(|domain| {
            (0..8)
                .map(|_| original.stream(domain).next_u64())
                .collect()
        });

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
