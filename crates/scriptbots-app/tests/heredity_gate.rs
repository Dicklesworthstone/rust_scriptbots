//! A brain family that cannot inherit must never found a population.
//!
//! `BrainRunner`'s own defaults are the trap this guards:
//!
//! ```ignore
//! fn clone_runner(&self) -> Result<Option<Box<dyn BrainRunner>>, BrainSpawnError> { Ok(None) }
//! fn mutate(&mut self, ..) -> Result<(), BrainSpawnError> { Ok(()) }
//! ```
//!
//! A family that merely fails to OVERRIDE those is silently non-heritable: `clone_runner`
//! returns `Ok(None)`, so reproduction spawns a FRESH runner from the registry and the child
//! never receives its parent's brain, while `mutate` reports success having changed nothing.
//! Both failures are invisible, and both are reported to the caller as success.
//!
//! `ml.placeholder` fell straight into that hole — its `tick` copies sensors to outputs and its
//! `mutate` is a no-op — and `seed_agents` round-robins the founding population across every
//! registered family. So under `--features ml`, roughly a third of the founders were born
//! unable to think or evolve, their descendants inherited it, and nothing downstream said a
//! word. Every experiment run under that feature was quietly contaminated.
//!
//! The registry now PROVES the contract instead of trusting it. These tests hold it to that.

use scriptbots_core::{BrainRunner, RandomStream, ScriptBotsConfig, SmallRngStream, WorldState};

/// A family that honours heredity: it duplicates itself when asked.
struct HeritableRunner {
    weight: f32,
}

impl BrainRunner for HeritableRunner {
    fn kind(&self) -> &'static str {
        "test.heritable"
    }

    /// The output MUST depend on the genome (`weight`).
    ///
    /// An earlier version of this fixture returned zeros regardless, and the probe correctly
    /// judged it non-heritable: a brain whose behaviour does not depend on its genome is
    /// indistinguishable from one that never mutates. That is not a flaw in the probe — it is
    /// the probe working. A brain whose weights cannot affect its outputs has nothing for
    /// selection to act on.
    fn tick(
        &mut self,
        inputs: &[f32; scriptbots_core::INPUT_SIZE],
    ) -> [f32; scriptbots_core::OUTPUT_SIZE] {
        let mut outputs = [0.0; scriptbots_core::OUTPUT_SIZE];
        outputs[0] = inputs[0] * self.weight;
        outputs
    }

    fn clone_runner(
        &self,
    ) -> Result<Option<Box<dyn BrainRunner>>, scriptbots_core::BrainSpawnError> {
        Ok(Some(Box::new(HeritableRunner {
            weight: self.weight,
        })))
    }

    fn mutate(
        &mut self,
        _rng: &mut dyn RandomStream,
        rate: f32,
        scale: f32,
    ) -> Result<(), scriptbots_core::BrainSpawnError> {
        self.weight += rate * scale;
        Ok(())
    }
}

/// A family that cannot duplicate itself: it does not override `clone_runner`, so it inherits
/// the default `Ok(None)` — "my children get a fresh brain" — and reports that as success.
struct SilentlyNonHeritableRunner;

impl BrainRunner for SilentlyNonHeritableRunner {
    fn kind(&self) -> &'static str {
        "test.placeholder"
    }

    fn tick(
        &mut self,
        _inputs: &[f32; scriptbots_core::INPUT_SIZE],
    ) -> [f32; scriptbots_core::OUTPUT_SIZE] {
        [0.0; scriptbots_core::OUTPUT_SIZE]
    }

    // clone_runner and mutate are NOT overridden. That is the whole point.
}

/// THE CASE THAT FOOLED THE FIRST VERSION OF THIS GATE, and the one `ml.placeholder` actually
/// is: a family that **clones perfectly well but never mutates**.
///
/// A gate that only asks "can you duplicate yourself?" waves this straight through, because it
/// answers yes — honestly. Its `mutate` is the lie: it ignores the rate and returns `Ok(())`.
/// The resulting lineage is a set of exact copies that reports a successful mutation at every
/// generation while the population stands still.
///
/// This fixture exists because I built the clone-only gate first, and my own test caught it.
struct ClonesButNeverMutatesRunner {
    weight: f32,
}

impl BrainRunner for ClonesButNeverMutatesRunner {
    fn kind(&self) -> &'static str {
        "test.clones_but_never_mutates"
    }

    fn tick(
        &mut self,
        inputs: &[f32; scriptbots_core::INPUT_SIZE],
    ) -> [f32; scriptbots_core::OUTPUT_SIZE] {
        let mut outputs = [0.0; scriptbots_core::OUTPUT_SIZE];
        outputs[0] = inputs[0] * self.weight;
        outputs
    }

    fn clone_runner(
        &self,
    ) -> Result<Option<Box<dyn BrainRunner>>, scriptbots_core::BrainSpawnError> {
        // A genuine, faithful clone. This family is not lying here.
        Ok(Some(Box::new(ClonesButNeverMutatesRunner {
            weight: self.weight,
        })))
    }

    // `mutate` is NOT overridden: the default returns Ok(()) and changes nothing. THIS is the lie.
}

fn world() -> WorldState {
    WorldState::new(ScriptBotsConfig {
        rng_seed: Some(7),
        ..ScriptBotsConfig::default()
    })
    .expect("world")
}

/// The probe `install_brains` ships, reproduced here as the property it must satisfy.
///
/// Eligibility is decided by ASKING THE FAMILY TO PERFORM THE CONTRACT, never by trusting a
/// declaration — a missing declaration is exactly what caused the bug, and whoever forgets to
/// override `mutate` would equally forget to set a capability flag.
///
/// The contract has TWO halves and both are load-bearing:
///   1. it must duplicate itself (or children get a fresh brain), AND
///   2. mutating that duplicate must actually CHANGE it (or the lineage is exact copies while
///      `mutate` reports success every generation).
///
/// Checking only (1) is the mistake that let `ml.placeholder` through: it clones honestly.
fn heritable(prototype: &dyn BrainRunner) -> bool {
    let Ok(Some(mut child)) = prototype.clone_runner() else {
        return false;
    };
    let Ok(Some(mut baseline)) = prototype.clone_runner() else {
        return false;
    };
    let mut rng = SmallRngStream::seed_from_u64(0x0A11_CE5E);
    if child.mutate(&mut rng, 1.0, 1.0).is_err() {
        return false;
    }
    [0.25f32, 0.5, 0.75].into_iter().any(|probe| {
        let inputs = [probe; scriptbots_core::INPUT_SIZE];
        let before = baseline.tick(&inputs);
        let after = child.tick(&inputs);
        before
            .iter()
            .zip(after.iter())
            .any(|(lhs, rhs)| lhs.to_bits() != rhs.to_bits())
    })
}

#[test]
fn a_family_that_declines_to_duplicate_itself_is_detected_as_non_heritable() {
    // The detection itself. If this were wrong, everything below would be theatre.
    assert!(
        heritable(&HeritableRunner { weight: 1.0 }),
        "a family that DOES duplicate itself must be judged heritable, or the gate would \
         exclude every real brain and the simulator would refuse to start"
    );
    assert!(
        !heritable(&SilentlyNonHeritableRunner),
        "a family that inherits BrainRunner's default clone_runner() -> Ok(None) must be \
         judged NON-heritable. This is the exact defect: the default silently means 'my \
         children get a fresh brain', and it reports that as success."
    );
}

#[test]
fn a_family_that_clones_but_never_mutates_is_not_heritable() {
    // THE REGRESSION TEST FOR MY OWN FIRST FIX, and the case `ml.placeholder` really is.
    //
    // A gate that asks only "can you duplicate yourself?" passes this family, because it
    // answers yes — truthfully. Its lie is elsewhere: `mutate` returns Ok(()) and changes
    // nothing, so the lineage is a set of exact copies that reports a successful mutation
    // every generation while the population stands still.
    //
    // Heredity is COPY *and* VARY. A gate that checks only the copy half is not a gate.
    let clones_fine = ClonesButNeverMutatesRunner { weight: 2.0 };
    assert!(
        matches!(clones_fine.clone_runner(), Ok(Some(_))),
        "this fixture must genuinely clone — otherwise it would be caught by the structural \
         check and would not be testing what it claims to test"
    );
    assert!(
        !heritable(&clones_fine),
        "a family that clones faithfully but NEVER MUTATES was judged heritable. This is \
         exactly how `ml.placeholder` reached the founding population: it duplicates itself \
         honestly, and only its mutation is a no-op. Copying without varying is not heredity."
    );
}

#[test]
fn the_default_mutate_is_a_silent_lie_and_that_is_why_a_declaration_cannot_be_trusted() {
    // This is the anti-vacuity heart of the bead. A capability FLAG would not have saved us:
    // whoever forgot to override `mutate` would equally have forgotten to set the flag.
    //
    // Demonstrate the lie directly: the default mutate() returns Ok(()) — success — while
    // changing nothing. A population of these evolves in name only, and every downstream
    // consumer is told the mutation succeeded.
    let mut placeholder = SilentlyNonHeritableRunner;
    let mut rng = SmallRngStream::seed_from_u64(1);
    let result = placeholder.mutate(&mut rng, 1.0, 1.0);
    assert!(
        result.is_ok(),
        "the default mutate() reports SUCCESS — that is the bug, and this test documents it \
         rather than pretending otherwise"
    );

    // And the heritable family actually changes under the same call.
    let mut real = HeritableRunner { weight: 0.0 };
    real.mutate(&mut rng, 1.0, 0.5).expect("mutate");
    assert!(
        (real.weight - 0.5).abs() < 1e-6,
        "a family that claims heredity must actually change when mutated; otherwise the \
         'mutation on -> genome differs' proof in bd-2z0.3.6 is satisfiable by doing nothing"
    );
}

#[test]
fn a_non_heritable_family_never_founds_a_population() {
    // THE PROPERTY THE BEAD EXISTS FOR, stated over the registry the app actually uses.
    //
    // Register one real family and one silent placeholder — the precise situation under
    // `--features ml` — then apply the gate. The placeholder must be registered (so it can
    // still be selected explicitly for experiments) and must NOT be eligible to seed.
    let mut world = world();

    let good = world
        .brain_registry_mut()
        .expect("heritable registry mutation")
        .register("test.heritable", |_rng| {
            Ok(Box::new(HeritableRunner { weight: 1.0 }) as Box<dyn BrainRunner>)
        });
    let bad = world
        .brain_registry_mut()
        .expect("placeholder registry mutation")
        .register("test.placeholder", |_rng| {
            Ok(Box::new(SilentlyNonHeritableRunner) as Box<dyn BrainRunner>)
        });

    let mut rng = SmallRngStream::seed_from_u64(0);
    let mut population = Vec::new();
    let mut withheld = Vec::new();
    for (key, _label) in world.brain_registry().descriptors() {
        let prototype = world
            .brain_registry()
            .spawn(&mut rng, key)
            .expect("registered family must spawn")
            .expect("registered family must produce a runner");
        if heritable(prototype.as_ref()) {
            population.push(key);
        } else {
            withheld.push(key);
        }
    }

    assert!(
        population.contains(&good),
        "the heritable family must be eligible to found the population"
    );
    assert!(
        !population.contains(&bad),
        "A NON-HERITABLE FAMILY REACHED THE FOUNDING POPULATION. seed_agents round-robins \
         founders across every eligible key, so this agent — and every descendant that \
         inherits its family — cannot evolve, and no error is raised anywhere. This is the \
         defect that silently contaminated every `--features ml` run."
    );
    assert!(
        withheld.contains(&bad),
        "the placeholder must be recorded as WITHHELD rather than silently dropped: an \
         exclusion nobody can see is folklore, not a decision"
    );

    // It must remain REGISTERED — withheld is not the same as unregistered. An experiment
    // that deliberately wants to bind a placeholder brain must still be able to.
    assert!(
        world.brain_registry().contains(bad),
        "a withheld family must stay registered and selectable; withholding it from default \
         populations is not the same as deleting it"
    );
}

#[test]
fn a_world_with_only_non_heritable_families_has_no_eligible_founders() {
    // The refusal case. `install_brains` bails here rather than starting, because a run whose
    // entire population cannot evolve produces plausible-looking data and answers no question.
    // This test pins the condition that triggers that bail.
    let mut world = world();
    world
        .brain_registry_mut()
        .expect("placeholder registry mutation")
        .register("test.placeholder", |_rng| {
            Ok(Box::new(SilentlyNonHeritableRunner) as Box<dyn BrainRunner>)
        });

    let mut rng = SmallRngStream::seed_from_u64(0);
    let eligible: Vec<u64> = world
        .brain_registry()
        .descriptors()
        .into_iter()
        .filter(|(key, _)| {
            let prototype = world
                .brain_registry()
                .spawn(&mut rng, *key)
                .expect("spawn")
                .expect("runner");
            heritable(prototype.as_ref())
        })
        .map(|(key, _)| key)
        .collect();

    assert!(
        eligible.is_empty(),
        "a world whose only brain family is non-heritable must have NO eligible founders — \
         install_brains refuses to start such a run rather than produce data about a \
         population that is not evolving"
    );
}

#[test]
fn production_install_brains_exposes_canonical_heredity_capabilities() {
    let mut world = world();
    let installed = scriptbots_app::install_brains(&mut world, scriptbots_app::BrainPreset::Mixed)
        .expect("canonical production install succeeds");
    assert!(!installed.is_empty(), "installed brains must not be empty");
    assert!(
        installed.registered() >= 3,
        "at least 3 protocol families registered"
    );

    let snapshot = installed
        .heredity_capabilities(&world)
        .expect("heredity capability projection succeeds");
    assert_eq!(
        snapshot.descriptors.len(),
        installed.registered(),
        "capability descriptors count must match registered families"
    );
    assert_ne!(
        snapshot.capability_digest.as_bytes(),
        &[0u8; 32],
        "capability digest must be computed"
    );
}
