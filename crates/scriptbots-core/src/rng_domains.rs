//! Domain-separated random streams.
//!
//! # The problem this exists to solve
//!
//! Today the world draws every stochastic decision from ONE stream: food scatter, spawn
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
//! **2. You cannot hold one domain fixed while varying another.** That is not a nicety — it is
//! what a controlled experiment IS. "Same food layout, different mutation seed" is unaskable
//! with a single stream, because changing the mutation seed changes the food. Every "replicate"
//! that varies one factor silently varies all of them, and the experiment measures something
//! other than what it claims.
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
use std::collections::BTreeMap;

/// Identity of the derivation. Bump this ONLY when deliberately re-deriving every domain seed —
/// doing so moves every stochastic decision in the project, so it must be announced.
pub const RNG_DOMAIN_DERIVATION_V1: &str = "scriptbots.rng-domains.v1";

/// The independent domains a stochastic decision can belong to.
///
/// These are not cosmetic labels. Two decisions belong in different domains when an experiment
/// might reasonably want to hold one fixed while varying the other.
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
    /// Every domain, in a stable order. Used to derive and to checkpoint, so it must never be
    /// reordered — the order is part of the wire format.
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
    streams: BTreeMap<RngDomain, SmallRngStream>,
}

impl DomainStreams {
    /// Derive every domain stream from one root seed.
    #[must_use]
    pub fn from_root_seed(root_seed: u64) -> Self {
        let streams = RngDomain::ALL
            .into_iter()
            .map(|domain| {
                let seed = derive_domain_seed(root_seed, domain);
                (domain, SmallRngStream::seed_from_u64(seed))
            })
            .collect();
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
    /// and `restore` refuses any state that does not. There is deliberately no `Option` here —
    /// a caller forced to handle "this domain has no stream" would have nothing sensible to do
    /// but fall back to some other domain's stream, which is precisely the coupling this module
    /// exists to prevent.
    pub fn stream(&mut self, domain: RngDomain) -> &mut SmallRngStream {
        self.streams
            .get_mut(&domain)
            .expect("every RngDomain is populated at construction and validated at restore")
    }

    /// Capture a restorable checkpoint of every domain.
    #[must_use]
    pub fn checkpoint(&self) -> BTreeMap<String, RandomStreamState> {
        self.streams
            .iter()
            .map(|(domain, stream)| (domain.tag().to_owned(), stream.checkpoint()))
            .collect()
    }

    /// Restore from a checkpoint.
    ///
    /// A checkpoint MISSING a domain is refused rather than silently re-derived from the root
    /// seed. Re-deriving would look like a successful restore while quietly rewinding that
    /// domain's stream to tick zero — the resumed run would diverge from the one it claims to
    /// continue, and the digest would not necessarily catch it because the *other* domains would
    /// match perfectly.
    pub fn restore(
        root_seed: u64,
        states: &BTreeMap<String, RandomStreamState>,
    ) -> Result<Self, DomainStreamRestoreError> {
        let mut streams = BTreeMap::new();
        for domain in RngDomain::ALL {
            let state = states.get(domain.tag()).ok_or_else(|| {
                DomainStreamRestoreError::MissingDomain {
                    domain: domain.tag(),
                }
            })?;
            let stream = SmallRngStream::from_state(state).map_err(|source| {
                DomainStreamRestoreError::Stream {
                    domain: domain.tag(),
                    source,
                }
            })?;
            streams.insert(domain, stream);
        }
        Ok(Self { streams, root_seed })
    }
}

/// Why a set of persisted domain states could not be restored.
///
/// Both variants NAME THE DOMAIN. A restore failure that said only "a stream was bad" would leave
/// the reader unable to tell whether their mutations, their food, or their lineage decisions were
/// the ones that could not be resumed — and those have very different consequences for a run.
#[derive(Debug, thiserror::Error)]
pub enum DomainStreamRestoreError {
    /// The checkpoint carried no state for this domain.
    ///
    /// Refused rather than re-derived: silently re-deriving from the root seed would look like a
    /// successful restore while rewinding that one domain to tick zero, and every other domain
    /// would resume perfectly — so nothing downstream would necessarily notice that the resumed
    /// run had quietly begun re-rolling its mutations from the beginning.
    #[error(
        "the checkpoint carries no random stream for the `{domain}` domain; refusing to \
         re-derive it, because that would silently rewind {domain} to its initial state while \
         every other domain resumed correctly"
    )]
    MissingDomain { domain: &'static str },

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
        let expected: BTreeMap<RngDomain, Vec<u64>> = RngDomain::ALL
            .into_iter()
            .map(|domain| {
                let draws = (0..8).map(|_| original.stream(domain).next_u64()).collect();
                (domain, draws)
            })
            .collect();

        let mut restored = DomainStreams::restore(555, &checkpoint)
            .expect("a checkpoint we just took must restore");
        for domain in RngDomain::ALL {
            let draws: Vec<u64> = (0..8).map(|_| restored.stream(domain).next_u64()).collect();
            assert_eq!(
                &draws, &expected[&domain],
                "domain {domain:?} diverged after restore — the resumed run is not the run it \
                 claims to continue"
            );
        }
    }

    #[test]
    fn a_checkpoint_missing_a_domain_is_refused_rather_than_re_derived() {
        // The dangerous failure. Silently re-deriving a missing domain from the root seed would
        // look like a successful restore while REWINDING that domain to tick zero — and the other
        // domains would match perfectly, so nothing downstream would necessarily notice.
        let original = DomainStreams::from_root_seed(9);
        let mut checkpoint = original.checkpoint();
        checkpoint.remove(RngDomain::Mutation.tag());

        assert!(
            DomainStreams::restore(9, &checkpoint).is_err(),
            "a checkpoint with NO mutation stream was accepted. That domain would have been \
             silently rewound to its initial state while every other domain resumed correctly — \
             a resumed run that quietly re-rolls its mutations is not the run it claims to be."
        );
    }
}
