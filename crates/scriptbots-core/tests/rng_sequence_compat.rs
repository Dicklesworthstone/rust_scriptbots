//! A `rand` upgrade must not be able to silently move the science.
//!
//! Every stochastic decision in this simulator — mutation, spawn placement, food
//! scatter, reproduction rolls — is drawn from one of six domain-separated
//! `RandomStream`s. Every domain currently uses `SmallRngStream`, and the canonical
//! world digest plus `RunManifestV3` cover their versioned checkpoint, so if `rand`
//! ever changed the algorithm behind `SmallRng`, every run's identity would move.
//!
//! It would move *quietly*. The symptom would be "the digest changed", which is
//! indistinguishable from "someone changed the physics" — the single most expensive
//! kind of failure to diagnose, because the evidence points at the wrong file.
//!
//! These goldens turn that into a loud, self-naming failure in a test whose entire
//! subject is the dependency. `bd-2z0.8.4` exists to permit a `rand` bump *safely*;
//! this file is what makes the bump safe to attempt, because it can be attempted and
//! then read off as compatible-or-not rather than guessed at.

use rand::RngCore;
use scriptbots_core::{RandomStream, SmallRngStream};

/// An arbitrary but FIXED seed. The value is meaningless; its constancy is the point.
const PINNED_SEED: u64 = 0x5CB1_B075_2026_0713;

/// The exact portable algorithm identity every run records in its manifest.
///
/// This string is not decoration: `SmallRngStream::from_state` REFUSES a checkpoint whose
/// algorithm id does not match, which is what stops a run from being resumed by a
/// generator that no longer produces its numbers. It also embeds the `rand` version, so
/// pinning it here means a `rand` bump that forgets to update the identity fails LOUDLY
/// instead of silently attributing two different streams to the same name.
const PINNED_ALGORITHM: &str = "rand-0.9.5-smallrng-xoshiro256plusplus-64-seed-from-u64";

/// The first draws of the project-owned stream. They preserve the 64-bit `rand 0.9.5`
/// `SmallRng` sequence that established this protocol identity. Regenerate ONLY when
/// deliberately accepting a new generator.
///
/// Xoshiro256++ is selected explicitly on every target. The `64` in the identity is the
/// generator word width, not the target pointer width. This sequence therefore applies equally
/// to native and `wasm32`; a legacy Xoshiro128++ checkpoint retains its different identifier and
/// is rejected rather than reseeded or reinterpreted.
const PINNED_SEQUENCE: [u64; 8] = [
    0x983b_a855_4b84_a81a,
    0x0da6_d2ad_c991_ed42,
    0x9a2d_5268_ac05_5d77,
    0x64cb_c9fe_908f_e17a,
    0xe84e_203f_90ac_64e3,
    0x4bcd_1316_1495_adde,
    0xbd09_ec00_a133_47f0,
    0x66a6_fcce_7bf9_5ce4,
];

#[test]
fn the_random_stream_produces_its_pinned_sequence() {
    let mut stream = SmallRngStream::seed_from_u64(PINNED_SEED);
    let actual: Vec<u64> = (0..PINNED_SEQUENCE.len())
        .map(|_| stream.next_u64())
        .collect();

    assert_eq!(
        actual,
        PINNED_SEQUENCE.to_vec(),
        "THE RANDOM STREAM CHANGED.\n\n\
         The generator behind `SmallRngStream` no longer produces the sequence this \
         project's science was recorded against. Every characterization digest, every \
         golden, and every reproducibility claim in the repository refers to the OLD \
         sequence.\n\n\
         This is almost certainly a `rand` upgrade. It is not automatically wrong — but \
         it is a deliberate, announced act, not a side effect of a version bump:\n\n\
         1. Bump the algorithm identity (`SmallRngStream::algorithm()`), so that a run \
            manifest written before the change cannot be mistaken for one written after.\n\
         2. Re-baseline the goldens that depend on the stream.\n\
         3. Record the old and new algorithm ids on the bead, so a historical run can \
            still be interpreted.\n\n\
         If you did NOT intend to change the generator, revert the dependency bump: the \
         upgrade is not sequence-compatible.\n\n\
         got: {actual:#018x?}"
    );
}

#[test]
fn the_algorithm_identity_is_stable() {
    // RunManifestV3 records this string for every fixed field in `random_streams.streams`, and
    // the restore path REFUSES a state whose algorithm id does not match. If the id drifted
    // silently, an old checkpoint would either be rejected for the wrong reason or — far
    // worse — be accepted by a generator that no longer produces the same numbers.
    let stream = SmallRngStream::seed_from_u64(PINNED_SEED);
    assert_eq!(
        stream.algorithm_id(),
        SmallRngStream::algorithm(),
        "the trait's algorithm id disagrees with the type's own constant"
    );
    assert_eq!(
        stream.checkpoint().algorithm,
        SmallRngStream::algorithm(),
        "the checkpoint records an algorithm id other than the one that produced it — a \
         restored run would be attributed to the wrong generator"
    );

    // And the id must be the EXACT string this project's recorded science was produced
    // under. It embeds the `rand` version, so a bump that changes the generator without
    // changing the name would let two different streams share one identity — and every
    // checkpoint written before the bump would be silently accepted by a generator that
    // no longer produces its numbers.
    assert_eq!(
        SmallRngStream::algorithm(),
        PINNED_ALGORITHM,
        "the random-stream algorithm identity changed. If you bumped `rand`, that is \
         correct and expected — re-baseline this pin and the sequence golden together, \
         and record both ids on the bead so historical runs stay interpretable."
    );
}

#[test]
fn a_restored_checkpoint_resumes_the_identical_sequence() {
    // The property that makes checkpoints worth having. If a restored stream diverged,
    // a resumed run would silently be a DIFFERENT experiment from the one it claims to
    // continue — and the manifest would still say they were the same.
    let mut original = SmallRngStream::seed_from_u64(PINNED_SEED);
    for _ in 0..17 {
        let _ = original.next_u64(); // advance to a non-trivial position in the stream
    }

    let checkpoint = original.checkpoint();
    let expected: Vec<u64> = (0..16).map(|_| original.next_u64()).collect();

    let mut restored = SmallRngStream::from_state(&checkpoint)
        .expect("a checkpoint this stream just produced must restore");
    let resumed: Vec<u64> = (0..16).map(|_| restored.next_u64()).collect();

    assert_eq!(
        resumed, expected,
        "a restored checkpoint diverged from the stream it was taken from — a resumed \
         run is not the run it claims to be"
    );
}

#[test]
fn two_streams_from_the_same_seed_agree() {
    // Seeding must be a pure function of the seed. If it mixed in ANY ambient state —
    // a clock, an address, thread-local entropy — reproducible runs would be impossible
    // and every manifest in the project would be lying.
    let mut left = SmallRngStream::seed_from_u64(PINNED_SEED);
    let mut right = SmallRngStream::seed_from_u64(PINNED_SEED);
    for draw in 0..64 {
        assert_eq!(
            left.next_u64(),
            right.next_u64(),
            "two streams seeded identically diverged at draw {draw}: seeding is not a \
             pure function of the seed, so no run in this project is reproducible"
        );
    }
}

#[test]
fn different_seeds_produce_different_streams() {
    // The anti-vacuity guard. If seeding were ignored entirely, every test above would
    // still pass — and every "independent replicate" in the lab would secretly be the
    // same run.
    let mut left = SmallRngStream::seed_from_u64(PINNED_SEED);
    let mut right = SmallRngStream::seed_from_u64(PINNED_SEED ^ 1);
    let diverged = (0..64).any(|_| left.next_u64() != right.next_u64());
    assert!(
        diverged,
        "two different seeds produced identical streams — the seed is being ignored, and \
         every 'independent replicate' is the same run"
    );
}
