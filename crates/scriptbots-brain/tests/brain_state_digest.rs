//! A brain's digest must see what the brain has BECOME, not merely which family it belongs to.
//!
//! `CharacterizationDigestV0` — the oracle that decides whether two runs are the same run —
//! records only a brain's registry key, its family name, and whether it is bound. It records
//! nothing about the weights. So two populations that have evolved for a million ticks into
//! completely different brains produce the SAME v0 digest, provided their families and bindings
//! match. The oracle is blind to the only thing in the simulation that is actually evolving.
//!
//! That limitation is DECLARED rather than hidden — `RunManifestV3` carries
//! `CharacterizationLimitationsV0 { evaluator_state_covered: false, superseded_by: "WorldDigestV1" }`
//! — and `BrainRunner::state_digest` is the hook that closes it (bd-2z0.3.9).
//!
//! These tests pin the properties that make the hook trustworthy.

use scriptbots_brain::mlp::MlpBrain;
use scriptbots_core::{BrainRunner, SmallRngStream};

fn runner(seed: u64) -> Box<dyn BrainRunner> {
    let mut rng = SmallRngStream::seed_from_u64(seed);
    MlpBrain::runner(&mut rng)
}

#[test]
fn a_brain_that_has_mutated_reports_a_different_state_digest() {
    // THE POINT OF THE WHOLE LANE. If this failed, `WorldDigestV1` would inherit exactly the
    // blindness it exists to remove: an evolved population would be indistinguishable from an
    // untouched one.
    let mut brain = runner(1);
    let before = brain.state_digest().expect(
        "the default MLP family must expose its state; if it cannot, no run using it \
                 can be certified as the same run twice",
    );

    let mut rng = SmallRngStream::seed_from_u64(99);
    brain.mutate(&mut rng, 1.0, 1.0).expect("mutate");

    let after = brain
        .state_digest()
        .expect("state must remain observable after mutation");

    assert_ne!(
        before, after,
        "a brain mutated at FULL RATE reports the same state digest as before. The digest is \
         blind to the genome, so an evolved population would be indistinguishable from an \
         untouched one and every claim of run-identity would be worthless."
    );
}

#[test]
fn the_state_digest_is_deterministic_and_seed_sensitive() {
    // Deterministic: the same brain must always report the same digest, or nothing downstream
    // can compare anything.
    let brain = runner(7);
    assert_eq!(
        brain.state_digest(),
        brain.state_digest(),
        "the state digest varies between calls on an unchanged brain — no comparison built on \
         it could mean anything"
    );

    // Seed-sensitive: two independently seeded brains are DIFFERENT brains, and must not
    // collide. Without this, the test above is satisfiable by a constant.
    let left = runner(1);
    let right = runner(2);
    assert_ne!(
        left.state_digest(),
        right.state_digest(),
        "two differently-seeded brains share a state digest — the digest is not a function of \
         the genome, and distinct populations would compare as identical"
    );
}

#[test]
fn an_inherited_brain_carries_its_parents_state_digest() {
    // Heredity, stated in the digest's own terms: a child cloned from a parent IS that parent,
    // bit for bit, until it is mutated. This is the same property bd-2z0.3.6 proved over genomes
    // — restated here so `WorldDigestV1` can rely on it.
    let parent = runner(11);
    let parent_digest = parent.state_digest().expect("parent state");

    let child = parent
        .clone_runner()
        .expect("clone must not fail")
        .expect("the MLP family is heritable");

    assert_eq!(
        child.state_digest().expect("child state"),
        parent_digest,
        "a freshly cloned child does not carry its parent's state. Reproduction is handing \
         children something other than the parent's brain, and no lineage claim in this project \
         would hold."
    );
}

#[test]
fn the_evaluator_state_is_part_of_the_digest_not_just_the_genome() {
    // The subtle half, and the one a genome-only digest would miss.
    //
    // The MLP is RECURRENT: its next output depends on the previous one. Two brains with
    // IDENTICAL weights but different recurrent state are not the same brain — they will produce
    // different outputs on the very next tick. A checkpoint restored without the recurrent state
    // would pass a genome-only digest while silently resuming a DIFFERENT experiment.
    let mut ticked = runner(21);
    let untouched = runner(21);

    // Same seed, so identical genomes — and, before any tick, identical digests.
    assert_eq!(
        ticked.state_digest(),
        untouched.state_digest(),
        "two brains from the same seed must start identical, or this test proves nothing"
    );

    // Drive one of them so its recurrent state advances. The GENOME is untouched: `tick` does
    // not mutate weights.
    let inputs = [0.75; scriptbots_core::INPUT_SIZE];
    for _ in 0..4 {
        let _ = ticked.tick(&inputs);
    }

    assert_ne!(
        ticked.state_digest(),
        untouched.state_digest(),
        "a brain that has been TICKED reports the same digest as one that has not, even though \
         its recurrent state has advanced and it will now produce different outputs. The digest \
         covers the genome but not the evaluator state — so a checkpoint restored without that \
         state would compare as identical while resuming a different experiment."
    );
}
