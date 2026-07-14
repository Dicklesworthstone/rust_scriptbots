//! `WorldDigestV1` must see what `CharacterizationDigestV0` could not.
//!
//! v0's limitations were DECLARED rather than hidden — `RunManifestV1` carries
//! `CharacterizationLimitationsV0 { evaluator_state_covered: false, ..., superseded_by:
//! "WorldDigestV1" }`. This file is where that promise is kept.
//!
//! The headline test is a MATCHED PAIR: it shows v0 failing to notice something and v1 noticing
//! it. The second half is what makes the first half mean anything — on its own, "v1 changed"
//! would prove only that v1 is sensitive to *something*.

use scriptbots_brain::mlp::MlpBrain;
use scriptbots_core::{AgentData, ScriptBotsConfig, SmallRngStream, WorldState};

/// A world whose agents all carry an MLP brain built from `brain_seed`.
///
/// The registered factory deliberately IGNORES the world's RNG and seeds from a fixed value, so
/// that two worlds built with different `brain_seed`s differ in their brain GENOMES and in
/// nothing else whatsoever: same config, same world seed, same bodies, same registry key, same
/// family label, and — because the factory never draws from it — the same random-stream state.
///
/// That isolation is the whole experiment. Any digest difference between two such worlds is
/// attributable to the brains and to nothing else.
fn world_with_brain_seed(brain_seed: u64, agent_count: usize) -> WorldState {
    let mut world = WorldState::new(ScriptBotsConfig {
        rng_seed: Some(4242),
        ..ScriptBotsConfig::default()
    })
    .expect("world");

    let key = world
        .brain_registry_mut()
        .expect("digest registry mutation")
        .register(MlpBrain::KIND.as_str(), move |_world_rng| {
            let mut fixed = SmallRngStream::seed_from_u64(brain_seed);
            Ok(MlpBrain::runner(&mut fixed))
        });

    for _ in 0..agent_count {
        let id = world
            .try_spawn_agent(AgentData::default())
            .expect("spawn agent");
        world.bind_agent_brain(id, key).expect("bind brain");
    }
    world
}

#[test]
fn v0_is_blind_to_the_genomes_and_v1_is_not() {
    // THE HEADLINE, and the defect bd-2z0.3.9 exists to close — demonstrated, not asserted.
    //
    // Two worlds, identical in every respect except that their agents carry DIFFERENT BRAINS.
    // In the only sense that matters to an evolutionary simulator these are different
    // populations. v0 cannot tell them apart: it encodes each brain's registry key, family name
    // and bound-ness, and all three are identical here.
    let left = world_with_brain_seed(1, 8);
    let right = world_with_brain_seed(2, 8);

    let v0_left = left.characterization_digest_v0().expect("v0");
    let v0_right = right.characterization_digest_v0().expect("v0");
    let v1_left = left.world_digest_v1().expect("v1");
    let v1_right = right.world_digest_v1().expect("v1");

    // First, prove the two worlds really are identical apart from their brains. Without this the
    // test below could pass for the wrong reason.
    assert_eq!(
        v0_left.agents, v0_right.agents,
        "the two worlds differ in their AGENT BODIES, so any digest difference could not be \
         attributed to the brains and this experiment would prove nothing"
    );
    assert_eq!(
        v0_left.brain_registry, v0_right.brain_registry,
        "the two worlds differ in their brain REGISTRY (key or family label), so v0 could \
         distinguish them for a reason that has nothing to do with genomes"
    );

    // v0: BLIND. This assertion documents a real defect; it does not endorse it.
    assert_eq!(
        v0_left.overall, v0_right.overall,
        "EXPECTED BLINDNESS NOT PRESENT. v0 distinguished two populations that differ ONLY in \
         their brain genomes. If v0 has been taught to cover evaluator state, delete this \
         assertion — but then CharacterizationLimitationsV0::evaluator_state_covered must stop \
         reporting `false`, and the two must change together."
    );

    // v1: SEES IT. Without this half the assertion above is merely a curiosity.
    assert_ne!(
        v1_left.overall, v1_right.overall,
        "v1 could NOT distinguish two populations carrying entirely different brains. It has \
         inherited exactly the blindness it exists to remove: an evolved population would be \
         indistinguishable from an untouched one, and every claim of run-identity would be \
         worthless."
    );

    // And it must be the BRAINS lane that differs — not some incidental byte. A digest that
    // moved for the wrong reason is a coincidence, not a detection.
    assert_ne!(
        v1_left.brains, v1_right.brains,
        "the overall v1 digest differs but its BRAINS lane does not, so it distinguished the \
         worlds for some other reason. The lanes exist precisely so a difference can be \
         attributed."
    );
    assert_eq!(
        v1_left.agents, v1_right.agents,
        "the AGENTS (body) lane differs between worlds whose bodies are identical. The lanes are \
         not separating what they claim to separate, so 'the brains changed' could not be \
         distinguished from 'the bodies changed'."
    );
}

#[test]
fn v1_reports_honestly_how_much_of_the_world_it_can_see() {
    // The coverage flag is not decoration. A digest computed while blind to some brains must SAY
    // so — otherwise v1 repeats v0's sin while wearing a higher version number.
    let world = world_with_brain_seed(7, 4);
    let digest = world.world_digest_v1().expect("v1");

    assert!(
        digest.evaluator_state_covered,
        "the default MLP family exposes its state, so a world built only from MLP brains must \
         report FULL coverage. If this fails, either the family stopped exposing its state or the \
         coverage flag is computed wrongly — and either way every v1 digest in the project is \
         quietly weaker than it claims to be."
    );
    assert!(
        digest.uncovered_families.is_empty(),
        "a fully covered world named uncovered families: {:?}",
        digest.uncovered_families
    );
    assert_eq!(
        digest.agent_identity, "AgentUid",
        "the agent lane must be keyed on the STABLE uid, not the recycled slot key that v0 used"
    );
    assert_eq!(digest.schema, scriptbots_core::WORLD_DIGEST_V1_SCHEMA);
}

#[test]
fn the_digest_is_deterministic() {
    // Everything above rests on this. If the digest varied between calls on an unchanged world,
    // every comparison in the project would be noise.
    let world = world_with_brain_seed(3, 6);
    assert_eq!(
        world.world_digest_v1().expect("v1"),
        world.world_digest_v1().expect("v1"),
        "two digests of the SAME unchanged world disagree — the oracle is not a function of the \
         world, and nothing built on it means anything"
    );
}

#[test]
fn a_live_world_moves_and_the_lanes_say_where() {
    // The diagnostic value of keeping the lanes separate, on a real tick rather than a synthetic
    // poke. When two runs disagree, `overall` tells you only THAT they differ; the lanes tell you
    // WHERE — the difference between a bug you can find in an hour and one you cannot find at all.
    let mut world = world_with_brain_seed(5, 8);
    let before = world.world_digest_v1().expect("v1");

    for _ in 0..25 {
        world.step().expect("step");
    }
    let after = world.world_digest_v1().expect("v1");

    assert_ne!(
        before.overall, after.overall,
        "twenty-five ticks of a live world left the digest unchanged — the oracle is not \
         observing the simulation at all"
    );

    // The RNG lane must move: a stepped world has drawn from its stream, and two worlds whose
    // streams stand at different points have different futures from the very next stochastic
    // decision onward. v1 encodes the RESTORABLE checkpoint (algorithm identity included), not
    // v0's four-draw probe.
    assert_ne!(
        before.rng, after.rng,
        "the RNG lane did not move across 25 ticks, so the digest cannot tell apart two worlds \
         whose streams are poised at different points"
    );

    // The counters lane must move: ticks and allocation counters are exactly what it encodes.
    assert_ne!(
        before.counters, after.counters,
        "the counters lane did not move across 25 ticks"
    );

    // Anti-vacuity: the digest must not be a hash of the tick number alone. The bodies must have
    // moved too, or the lanes are not measuring the world.
    assert_ne!(
        before.agents, after.agents,
        "no agent body changed across 25 ticks of a live simulation — the agents lane is not \
         observing the agents"
    );
}
