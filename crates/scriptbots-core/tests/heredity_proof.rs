//! The heredity proof (bd-16g.13.2).
//!
//! Heredity was fixed at e2d9aaa — offspring inherit the parent's weights — and until
//! this suite the evidence it stayed fixed was "it compiles". These tests go through
//! the REAL spawn path (a full `WorldState` driving its own reproduction stage, never
//! a hand-rolled `clone_box`) and prove, per registered family with locus support:
//!
//! * asexual children at mutation rate 0 are bit-identical to the parent;
//! * sexual children at mutation rate 0 are a per-locus bitwise mix of the two parents
//!   (never a blend, never a fresh node), with evaluator state reset, not inherited;
//! * the mixed-kind barrier holds (a mismatched partner produces a same-kind clone);
//! * changed-locus counts at a nonzero rate land inside an EXACT binomial tail

// bd-tqpj: deterministic-simulation policy — pinned floating-point evaluation
// order and fixed-width casts are part of the science contract; fma fusion,
// reassociation, or width changes alter world digests. The exact-binomial
// statistician casts counts to f64 deliberately.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
#![allow(clippy::float_cmp, clippy::while_float)]
//!   interval (mutation is neither dead nor total);
//! * spawn determinism is independent of the ambient Rayon schedule.
//!
//! The suite is table-driven over families with protocol locus support, so a new
//! family that stops inheriting correctly fails CI the day it is added.

use scriptbots_brain::assembly::{AssemblyBrain, AssemblyFamilyAdapter};
use scriptbots_brain::dwraon::DwraonFamilyAdapter;
use scriptbots_brain::mlp::MlpBrainFamily;
use scriptbots_core::genome_diff::{LocusValue, diff_genomes};
use scriptbots_core::{
    AgentData, AgentId, AgentUid, BirthOrigin, BrainAdapterIdentityV1, BrainEvaluator,
    BrainEvaluatorStateEnvelope, BrainFamilyAdapter, BrainFamilyCodec, BrainGenomeDerivation,
    BrainGenomeEnvelope, BrainGenomeMaterial, BrainHeredityCapabilityV1, BrainProtocolError,
    MutationRates, OffspringStatePolicy, Position, RandomStream, ScriptBotsConfig, WorldState,
};

const ASSEMBLY_KIND: &str = AssemblyBrain::KIND.as_str();
const DWRAON_KIND: &str = "dwraon-baseline";
const MLP_KIND: &str = "mlp-baseline";

fn reproduction_config(partner_chance: f32) -> ScriptBotsConfig {
    ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        initial_food: 0.0,
        food_respawn_interval: 0,
        food_intake_rate: 0.0,
        food_sharing_rate: 0.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        bot_speed: 0.0,
        spike_damage: 0.0,
        spike_energy_cost: 0.0,
        population_minimum: 0,
        population_spawn_interval: 0,
        reproduction_energy_threshold: 0.5,
        reproduction_energy_cost: 0.0,
        reproduction_cooldown: 1,
        reproduction_attempt_interval: 1,
        reproduction_attempt_chance: 1.0,
        reproduction_child_energy: 1.0,
        reproduction_spawn_jitter: 0.0,
        reproduction_color_jitter: 0.0,
        reproduction_spawn_back_distance: 0.0,
        reproduction_partner_chance: partner_chance,
        reproduction_meta_mutation_chance: 0.0,
        reproduction_meta_mutation_scale: 0.0,
        persistence_interval: 0,
        chart_flush_interval: 0,
        closed: true,
        rng_seed: Some(0xB2CA_2026),
        ..ScriptBotsConfig::default()
    }
}

fn register_family(world: &mut WorldState, family: &str) -> u64 {
    let adapter: Box<dyn BrainFamilyAdapter> = match family {
        MLP_KIND => Box::new(MlpBrainFamily::new()),
        DWRAON_KIND => Box::new(DwraonFamilyAdapter::default()),
        ASSEMBLY_KIND => {
            Box::new(AssemblyFamilyAdapter::new().expect("canonical Assembly adapter"))
        }
        other => panic!("no adapter fixture for {other}"),
    };
    world
        .register_brain_family(family.to_owned(), adapter)
        .expect("register brain family")
}

fn dyn_codec_for(family: &str) -> Box<dyn BrainFamilyCodec> {
    match family {
        MLP_KIND => Box::new(MlpBrainFamily::new()),
        DWRAON_KIND => Box::new(DwraonFamilyAdapter::default()),
        ASSEMBLY_KIND => {
            Box::new(AssemblyFamilyAdapter::new().expect("canonical Assembly adapter"))
        }
        other => panic!("no codec fixture for {other}"),
    }
}

fn spawn_parent(world: &mut WorldState, key: u64, x: f32, y: f32, rates: MutationRates) -> AgentId {
    let parent = world
        .try_spawn_agent(AgentData {
            position: Position::new(x, y),
            ..AgentData::default()
        })
        .expect("parent spawn is finite");
    assert!(
        world
            .bind_agent_brain(parent, key)
            .expect("bind parent brain"),
        "parent must accept its brain binding"
    );
    world
        .try_update_agent_runtime(parent, |runtime| {
            runtime.energy = 1.5;
            runtime.mutation_rates = rates;
        })
        .expect("parent energy and mutation rates");
    parent
}

/// A child observed through the world's own birth stream.
struct ObservedChild {
    id: AgentId,
    parent_a: AgentUid,
    parent_b: Option<AgentUid>,
}

/// Drive the world until `expected` births have been observed (bounded), returning
/// the children whose first parent is one of `tracked`.
fn drive_until_children(
    world: &mut WorldState,
    tracked: &[AgentUid],
    expected: usize,
    max_ticks: u64,
) -> Vec<ObservedChild> {
    let mut children = Vec::new();
    for _ in 0..max_ticks {
        let completion = world.step_outcome().expect("reproduction step");
        for birth in &completion.outcome.births {
            if birth.origin != BirthOrigin::Born {
                continue;
            }
            let Some(parent_a) = birth.parent_a else {
                continue;
            };
            if !tracked.contains(&parent_a) {
                continue;
            }
            let id = world
                .agents()
                .iter_handles()
                .find(|id| world.agent_uid(*id) == Some(birth.agent_uid))
                .expect("a just-recorded birth uid must resolve to one live agent");
            children.push(ObservedChild {
                id,
                parent_a,
                parent_b: birth.parent_b,
            });
            if children.len() >= expected {
                return children;
            }
        }
    }
    panic!(
        "expected {expected} tracked births within {max_ticks} ticks; observed {}",
        children.len()
    );
}

fn genome_of(world: &WorldState, id: AgentId) -> BrainGenomeEnvelope {
    world
        .agent_brain_genome(id)
        .expect("bound agent exposes a genome")
        .clone()
}

fn uid_of(world: &WorldState, id: AgentId) -> AgentUid {
    world.agent_uid(id).expect("live agent has a uid")
}

fn assert_bit_identical(
    codec: &dyn BrainFamilyCodec,
    parent: &BrainGenomeEnvelope,
    child: &BrainGenomeEnvelope,
    family: &str,
) {
    let diff = diff_genomes(codec, parent, child).expect("zero-rate diff");
    assert!(
        diff.deltas.is_empty(),
        "heredity_proof: family={family} offspring=1 changed_loci={} expected=0 — a child at \
         mutation rate 0 must be bit-identical to its parent; the spawn path re-randomized \
         the brain (the e2d9aaa regression)",
        diff.summary.changed_loci
    );
}

fn for_each_family(test: fn(&str)) {
    test(MLP_KIND);
    test(DWRAON_KIND);
    test(ASSEMBLY_KIND);
}

#[test]
fn asexual_child_at_zero_mutation_is_bit_identical_to_parent() {
    fn case(family: &str) {
        let mut world = WorldState::new(reproduction_config(0.0)).expect("world");
        let key = register_family(&mut world, family);
        let parent = spawn_parent(
            &mut world,
            key,
            100.0,
            100.0,
            MutationRates {
                primary: 0.0,
                secondary: 0.0,
            },
        );
        let parent_uid = uid_of(&world, parent);
        let children = drive_until_children(&mut world, &[parent_uid], 1, 64);
        let child = &children[0];
        assert!(
            child.parent_b.is_none(),
            "heredity_proof: family={family} partner_chance=0.0 produced a crossover child"
        );
        assert_bit_identical(
            &*dyn_codec_for(family),
            &genome_of(&world, parent),
            &genome_of(&world, child.id),
            family,
        );
    }
    for_each_family(case);
}

#[test]
fn identical_parent_crossover_at_zero_mutation_is_bit_identical() {
    fn case(family: &str) {
        let mut world = WorldState::new(reproduction_config(1.0)).expect("world");
        let key = register_family(&mut world, family);
        let rates = MutationRates {
            primary: 0.0,
            secondary: 0.0,
        };
        let parent = spawn_parent(&mut world, key, 100.0, 100.0, rates);
        let parent_uid = uid_of(&world, parent);
        let first_birth = drive_until_children(&mut world, &[parent_uid], 1, 64)
            .pop()
            .expect("one first-generation child");
        assert!(
            first_birth.parent_b.is_none(),
            "heredity_proof: family={family} lone founder unexpectedly found a crossover partner"
        );

        let codec = dyn_codec_for(family);
        let parent_genome = genome_of(&world, parent);
        let clone_genome = genome_of(&world, first_birth.id);
        assert_bit_identical(&*codec, &parent_genome, &clone_genome, family);
        assert_eq!(
            clone_genome.provenance().derivation,
            BrainGenomeDerivation::Clone,
            "heredity_proof: family={family} first zero-rate child must record an asexual clone"
        );

        world
            .try_update_agent_runtime(first_birth.id, |runtime| {
                runtime.energy = 1.5;
                runtime.mutation_rates = rates;
            })
            .expect("clone energy and zero mutation rates");
        let clone_uid = uid_of(&world, first_birth.id);
        let crossover = drive_until_children(&mut world, &[parent_uid, clone_uid], 1, 64)
            .pop()
            .expect("one second-generation child");
        let expected_partner = if crossover.parent_a == parent_uid {
            clone_uid
        } else {
            assert_eq!(
                crossover.parent_a, clone_uid,
                "heredity_proof: family={family} crossover descended from an untracked agent"
            );
            parent_uid
        };
        assert_eq!(
            crossover.parent_b,
            Some(expected_partner),
            "heredity_proof: family={family} two identical parents must exercise crossover"
        );

        let child_genome = genome_of(&world, crossover.id);
        assert_eq!(
            child_genome.provenance().derivation,
            BrainGenomeDerivation::Crossover,
            "heredity_proof: family={family} identical-parent birth must record a real crossover"
        );
        assert_eq!(
            child_genome.provenance().parents,
            [Some(crossover.parent_a), crossover.parent_b],
            "heredity_proof: family={family} crossover provenance must record both actual parents"
        );
        assert_eq!(
            child_genome.provenance().parent_genome_hashes,
            [
                Some(parent_genome.material_hash()),
                Some(parent_genome.material_hash()),
            ],
            "heredity_proof: family={family} crossover provenance must identify both identical \
             source genomes"
        );
        assert_bit_identical(&*codec, &parent_genome, &child_genome, family);
        assert_bit_identical(&*codec, &clone_genome, &child_genome, family);
    }

    for_each_family(case);
}

#[test]
fn sexual_child_at_zero_mutation_is_a_per_locus_bitwise_mix() {
    fn case(family: &str) {
        let mut world = WorldState::new(reproduction_config(1.0)).expect("world");
        let key = register_family(&mut world, family);
        let rates = MutationRates {
            primary: 0.0,
            secondary: 0.0,
        };
        let parent_a = spawn_parent(&mut world, key, 60.0, 100.0, rates);
        let parent_b = spawn_parent(&mut world, key, 140.0, 100.0, rates);
        let uid_a = uid_of(&world, parent_a);
        let uid_b = uid_of(&world, parent_b);
        let children = drive_until_children(&mut world, &[uid_a, uid_b], 1, 64);
        let child = &children[0];
        assert_eq!(
            child.parent_b,
            Some(if child.parent_a == uid_a {
                uid_b
            } else {
                uid_a
            }),
            "heredity_proof: family={family} partner_chance=1.0 child must carry both parents"
        );

        let codec = dyn_codec_for(family);
        let a_loci = codec
            .genome_loci(&genome_of(&world, parent_a))
            .expect("parent A loci");
        let b_loci = codec
            .genome_loci(&genome_of(&world, parent_b))
            .expect("parent B loci");
        let child_loci = codec
            .genome_loci(&genome_of(&world, child.id))
            .expect("child loci");
        assert_eq!(child_loci.len(), a_loci.len());
        assert_eq!(child_loci.len(), b_loci.len());

        let mut from_a = 0_usize;
        let mut from_b = 0_usize;
        for (index, (locus, child_value)) in child_loci.iter().enumerate() {
            let (_, a_value) = &a_loci[index];
            let (_, b_value) = &b_loci[index];
            let matches_a = locus_value_bit_eq(*child_value, *a_value);
            let matches_b = locus_value_bit_eq(*child_value, *b_value);
            assert!(
                matches_a || matches_b,
                "heredity_proof: family={family} locus {} of child came from NEITHER parent \
                 (child={child_value:?}, A={a_value:?}, B={b_value:?}) — crossover must be a \
                 per-locus bitwise mix, never a blend and never a fresh locus",
                locus.human()
            );
            if matches_a && !matches_b {
                from_a += 1;
            } else if matches_b && !matches_a {
                from_b += 1;
            }
        }
        assert!(
            from_a > 0,
            "heredity_proof: family={family} crossover child took NOTHING uniquely from parent \
             A — this 'crossover' is just cloning and nobody would notice"
        );
        assert!(
            from_b > 0,
            "heredity_proof: family={family} crossover child took NOTHING uniquely from parent \
             B — this 'crossover' is just cloning and nobody would notice"
        );

        // Evaluator state is reset, never inherited: the child's state must differ from the
        // parent's accumulated state (parents have ticked; the child just spawned).
        let child_state = world
            .agent_brain_evaluator_state(child.id)
            .expect("child evaluator state")
            .expect("child exposes evaluator state");
        let parent_state = world
            .agent_brain_evaluator_state(parent_a)
            .expect("parent evaluator state")
            .expect("parent exposes evaluator state");
        assert_ne!(
            child_state.payload(),
            parent_state.payload(),
            "heredity_proof: family={family} child inherited the parent's accumulated evaluator \
             state — offspring state must reset (bd-2z0.3.2's heritable/non-heritable split)"
        );
    }
    for_each_family(case);
}

const fn locus_value_bit_eq(left: LocusValue, right: LocusValue) -> bool {
    match (left, right) {
        (LocusValue::Scalar(a), LocusValue::Scalar(b)) => a.to_bits() == b.to_bits(),
        (LocusValue::Target(a), LocusValue::Target(b)) => a == b,
        (LocusValue::Kind(a), LocusValue::Kind(b)) => a == b,
        _ => false,
    }
}

#[test]
fn mixed_kind_mating_falls_back_to_same_kind_clone() {
    let mut world = WorldState::new(reproduction_config(1.0)).expect("world");
    let mlp_key = register_family(&mut world, MLP_KIND);
    let dwraon_key = register_family(&mut world, DWRAON_KIND);
    let rates = MutationRates {
        primary: 0.0,
        secondary: 0.0,
    };
    let mlp_parent = spawn_parent(&mut world, mlp_key, 60.0, 100.0, rates);
    let dwraon_parent = spawn_parent(&mut world, dwraon_key, 140.0, 100.0, rates);
    let uid_mlp = uid_of(&world, mlp_parent);
    let uid_dwraon = uid_of(&world, dwraon_parent);

    let children = drive_until_children(&mut world, &[uid_mlp, uid_dwraon], 1, 64);
    let child = &children[0];
    let (parent_kind, parent_id) = if child.parent_a == uid_mlp {
        (MLP_KIND, mlp_parent)
    } else {
        assert_eq!(
            child.parent_a, uid_dwraon,
            "heredity_proof: the barrier child must descend from one of the two parents"
        );
        (DWRAON_KIND, dwraon_parent)
    };
    let child_genome = genome_of(&world, child.id);
    let child_family = child_genome.family_id().clone();
    assert_eq!(
        child_family.as_str(),
        parent_kind,
        "heredity_proof: mixed-kind pairing produced a child of family {child_family} — the \
         species barrier must preserve the reproducing parent's family, never a hybrid"
    );
    // The barrier is brain-scoped by design (see the in-crate
    // `incompatible_body_partner_does_not_fabricate_a_brain_crossover` test): a cross-kind
    // body partner may blend runtime physiology and appear in the lineage record, but the
    // child's brain must come from `clone_runner` — the genome provenance must never claim
    // a crossover that did not happen.
    assert!(
        !matches!(
            child_genome.provenance().derivation,
            BrainGenomeDerivation::Crossover | BrainGenomeDerivation::CrossoverThenMutation
        ),
        "heredity_proof: family={parent_kind} barrier child's genome provenance claims a \
         crossover with a different brain family — `clone_runner` semantics broken"
    );
    assert_eq!(
        child_genome.provenance().parents[1],
        None,
        "heredity_proof: family={parent_kind} barrier child's genome records a second brain \
         parent — a cross-kind partner must never enter the genome lineage"
    );
    assert_eq!(
        child_genome.provenance().parent_genome_hashes[0],
        Some(genome_of(&world, parent_id).material_hash()),
        "heredity_proof: family={parent_kind} barrier child does not derive from the \
         reproducing parent's genome — the spawn path substituted a fresh brain"
    );
}

#[test]
fn scheduled_cross_kind_crossover_falls_back_to_a_fresh_random_registry_founder() {
    fn scheduled_world(crossover_chance: f32) -> WorldState {
        let mut config = reproduction_config(0.0);
        config.closed = false;
        config.population_spawn_interval = 1;
        config.population_spawn_count = 1;
        config.population_crossover_chance = crossover_chance;
        config.reproduction_energy_threshold = 0.0;

        let mut world = WorldState::new(config).expect("scheduled population world");
        let mlp_key = register_family(&mut world, MLP_KIND);
        let dwraon_key = register_family(&mut world, DWRAON_KIND);
        let rates = MutationRates {
            primary: 0.0,
            secondary: 0.0,
        };
        spawn_parent(&mut world, mlp_key, 60.0, 100.0, rates);
        spawn_parent(&mut world, dwraon_key, 140.0, 100.0, rates);
        world
    }

    let mut fallback_world = scheduled_world(1.0);
    let mut direct_random_world = scheduled_world(0.0);
    let fallback_outcome = fallback_world
        .step_outcome()
        .expect("cross-family scheduled fallback");
    let direct_random_outcome = direct_random_world
        .step_outcome()
        .expect("direct scheduled random spawn");

    let mut fallback_births = fallback_outcome
        .outcome
        .births
        .iter()
        .filter(|birth| birth.origin == BirthOrigin::Injected);
    let fallback_birth = fallback_births
        .next()
        .expect("scheduled fallback must record one injected arrival");
    assert!(
        fallback_births.next().is_none(),
        "one scheduled attempt must produce exactly one fallback arrival"
    );
    let mut direct_random_births = direct_random_outcome
        .outcome
        .births
        .iter()
        .filter(|birth| birth.origin == BirthOrigin::Injected);
    let direct_random_birth = direct_random_births
        .next()
        .expect("direct random path must record one injected arrival");
    assert!(
        direct_random_births.next().is_none(),
        "one direct scheduled spawn must produce exactly one arrival"
    );
    assert_eq!(
        [fallback_birth.parent_a, fallback_birth.parent_b],
        [None, None]
    );
    assert!(
        !fallback_birth.is_hybrid,
        "a rejected scheduled brain crossover must not fabricate hybrid lineage"
    );
    assert_eq!(
        fallback_birth.brain_key, direct_random_birth.brain_key,
        "cross-family scheduled fallback must use the same random registry selection as a \
         direct scheduled random spawn"
    );

    let fallback_child = fallback_world
        .agents()
        .iter_handles()
        .find(|id| fallback_world.agent_uid(*id) == Some(fallback_birth.agent_uid))
        .expect("fallback arrival uid resolves to one live agent");
    let direct_random_child = direct_random_world
        .agents()
        .iter_handles()
        .find(|id| direct_random_world.agent_uid(*id) == Some(direct_random_birth.agent_uid))
        .expect("direct random arrival uid resolves to one live agent");
    let fallback_genome = genome_of(&fallback_world, fallback_child);
    let direct_random_genome = genome_of(&direct_random_world, direct_random_child);
    assert_eq!(
        fallback_genome, direct_random_genome,
        "scheduled cross-family fallback must construct a fresh registry founder, not clone \
         either incompatible parent"
    );
    assert_eq!(
        fallback_genome.provenance().derivation,
        BrainGenomeDerivation::Founder
    );
    assert_eq!(fallback_genome.provenance().parents, [None, None]);
    assert_eq!(
        fallback_genome.provenance().parent_genome_hashes,
        [None, None]
    );
}

/// Exact binomial tail probabilities via a mode-anchored recurrence: unnormalized
/// weights are expanded outward from the mode with the exact PMF ratio and then
/// normalized, which stays numerically stable for the n and p used here.
struct ExactBinomial {
    n: u64,
    p: f64,
}

impl ExactBinomial {
    /// Probability mass function over the numerically relevant support, returned as
    /// `(start_k, weights)` where `weights[i]` = P(X = `start_k` + i).
    ///
    /// `f64` accumulation error bounds the exactness.
    fn pmf_support(&self) -> (u64, Vec<f64>) {
        let n = self.n;
        let p = self.p;
        let q = 1.0 - p;
        let mode = ((n as f64 + 1.0) * p).floor() as u64;
        let mut up = vec![1.0_f64];
        let mut k = mode;
        while k < n {
            let ratio = ((n - k) as f64 / (k + 1) as f64) * (p / q);
            let next = up.last().expect("nonempty") * ratio;
            if next < 1e-300 {
                break;
            }
            up.push(next);
            k += 1;
        }
        let mut down = Vec::new();
        let mut k = mode;
        let mut weight = 1.0_f64;
        while k > 0 {
            let ratio = (k as f64 / (n - k + 1) as f64) * (q / p);
            weight *= ratio;
            if weight < 1e-300 {
                break;
            }
            down.push(weight);
            k -= 1;
        }
        let start = mode - down.len() as u64;
        down.reverse();
        down.extend(up);
        let sum: f64 = down.iter().sum();
        for weight in &mut down {
            *weight /= sum;
        }
        (start, down)
    }

    /// P(X <= x), exact up to `f64` accumulation error.
    // bd-tqpj: retained as the method-level counterpart of the free `support_cdf`
    // used below; kept for future exact-tail assertions.
    #[allow(dead_code)]
    fn cdf(&self, x: u64) -> f64 {
        let (start, weights) = self.pmf_support();
        let mut mass = 0.0;
        for (offset, weight) in weights.iter().enumerate() {
            if start + offset as u64 <= x {
                mass += weight;
            }
        }
        mass
    }
}

/// Convolution of two discrete PMFs over shifted supports.
fn convolve(a: &(u64, Vec<f64>), b: &(u64, Vec<f64>)) -> (u64, Vec<f64>) {
    let (a_start, a_weights) = a;
    let (b_start, b_weights) = b;
    let mut out = vec![0.0_f64; a_weights.len() + b_weights.len() - 1];
    for (i, wa) in a_weights.iter().enumerate() {
        for (j, wb) in b_weights.iter().enumerate() {
            out[i + j] += wa * wb;
        }
    }
    (a_start + b_start, out)
}

fn support_cdf((start, weights): &(u64, Vec<f64>), x: u64) -> f64 {
    let mut mass = 0.0;
    for (offset, weight) in weights.iter().enumerate() {
        if start + offset as u64 <= x {
            mass += weight;
        }
    }
    mass
}

/// The exact distribution of changed-locus counts across `children` offspring for a
/// family: every mutation draw is an independent Bernoulli, so each field family is
/// binomial and the total is their convolution.
fn changed_locus_distribution(family: &str, children: usize, rate: f32) -> (u64, Vec<f64>) {
    let c = children as u64;
    let p = f64::from(rate);
    match family {
        // MLP: 6 Bernoulli(rate) draws per node × 200 nodes.
        MLP_KIND => ExactBinomial { n: 1200 * c, p }.pmf_support(),
        // DWRAON: 2 draws at 3×rate (bias, one weight) + 3 draws at rate (source,
        // inverted, kind) per node × 200 nodes.
        DWRAON_KIND => convolve(
            &ExactBinomial {
                n: 400 * c,
                p: 3.0 * p,
            }
            .pmf_support(),
            &ExactBinomial { n: 600 * c, p }.pmf_support(),
        ),
        // Assembly: one Bernoulli(rate) replacement draw per cell × 200 cells.
        ASSEMBLY_KIND => ExactBinomial { n: 200 * c, p }.pmf_support(),
        other => panic!("no band model for {other}"),
    }
}

#[test]
fn changed_locus_counts_land_inside_the_exact_binomial_band() {
    const CHILDREN: usize = 8;
    const RATE: f32 = 0.05;
    // Two-sided 99.9% band: a correct implementation passes deterministically; a dead
    // or total mutation path lands in the far tails with probability ≈ 1.
    const ALPHA: f64 = 0.0005;

    fn case(family: &str) {
        let mut world = WorldState::new(reproduction_config(0.0)).expect("world");
        let key = register_family(&mut world, family);
        let parent = spawn_parent(
            &mut world,
            key,
            100.0,
            100.0,
            MutationRates {
                primary: RATE,
                secondary: RATE,
            },
        );
        let parent_uid = uid_of(&world, parent);
        let children = drive_until_children(&mut world, &[parent_uid], CHILDREN, 256);
        let codec = dyn_codec_for(family);
        let parent_genome = genome_of(&world, parent);
        let mut observed = 0_u64;
        for child in &children {
            let diff = diff_genomes(&*codec, &parent_genome, &genome_of(&world, child.id))
                .expect("band diff");
            observed += diff.summary.changed_loci as u64;
        }

        let support = changed_locus_distribution(family, CHILDREN, RATE);
        let lower_tail = support_cdf(&support, observed);
        assert!(
            lower_tail > ALPHA,
            "heredity_proof: family={family} offspring={CHILDREN} changed_loci={observed} \
             lower_tail={lower_tail:.6} — far too few changed loci at rate {RATE}; mutation \
             is silently dead (a frozen gene pool that still looks alive)"
        );
        assert!(
            lower_tail < 1.0 - ALPHA,
            "heredity_proof: family={family} offspring={CHILDREN} changed_loci={observed} \
             upper_tail={:.6} — far too many changed loci at rate {RATE}; children are \
             unrelated to parents (search, not evolution)",
            1.0 - lower_tail
        );
    }
    for_each_family(case);
}

/// FAULT INJECTION for the negative control: an exact MLP delegate except that
/// `mutate_genome_material` returns FRESH founder material regardless of the parent --
/// the e2d9aaa bug (offspring brains replaced by a fresh registry brain), restored on
/// purpose. Registered as the world's only family so the MLP wire id stays unique.
struct SabotagedMutationFamily {
    inner: MlpBrainFamily,
}

impl BrainFamilyCodec for SabotagedMutationFamily {
    fn family_id(&self) -> &scriptbots_core::BrainFamilyId {
        self.inner.family_id()
    }
    fn adapter_identity(&self) -> BrainAdapterIdentityV1 {
        self.inner.adapter_identity()
    }
    fn heredity_capability(&self) -> BrainHeredityCapabilityV1 {
        self.inner.heredity_capability()
    }
    fn random_genome_material(
        &self,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        self.inner.random_genome_material(rng)
    }
    fn validate_genome(&self, genome: &BrainGenomeEnvelope) -> Result<(), BrainProtocolError> {
        self.inner.validate_genome(genome)
    }
    fn genome_loci(
        &self,
        genome: &BrainGenomeEnvelope,
    ) -> Result<Vec<(scriptbots_core::genome_diff::Locus, LocusValue)>, BrainProtocolError> {
        self.inner.genome_loci(genome)
    }
    fn validate_evaluator_state(
        &self,
        state: &BrainEvaluatorStateEnvelope,
    ) -> Result<(), BrainProtocolError> {
        self.inner.validate_evaluator_state(state)
    }
    fn mutate_genome_material(
        &self,
        _genome: &BrainGenomeEnvelope,
        _rates: MutationRates,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        // THE BUG: offspring material is a fresh random founder's, not the parent's.
        self.inner.random_genome_material(rng)
    }
    fn crossover_genomes_material(
        &self,
        left: &BrainGenomeEnvelope,
        right: &BrainGenomeEnvelope,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        self.inner.crossover_genomes_material(left, right, rng)
    }
    fn initial_state(
        &self,
        genome: &BrainGenomeEnvelope,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.inner.initial_state(genome, rng)
    }
    fn offspring_state_policy(&self) -> OffspringStatePolicy {
        self.inner.offspring_state_policy()
    }
    fn offspring_state(
        &self,
        child: &BrainGenomeEnvelope,
        parents: &[&BrainEvaluatorStateEnvelope],
        rng: &mut dyn RandomStream,
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.inner.offspring_state(child, parents, rng)
    }
    fn evaluator(
        &self,
        genome: &BrainGenomeEnvelope,
        state: &BrainEvaluatorStateEnvelope,
    ) -> Result<Box<dyn BrainEvaluator>, BrainProtocolError> {
        self.inner.evaluator(genome, state)
    }
}

#[test]
fn the_proof_detects_the_restored_e2d9aaa_bug() {
    // The zero-rate heredity proof MUST reject the saboteur: a proof that cannot
    // detect the historical bug is not a proof of anything.
    let mut world = WorldState::new(reproduction_config(0.0)).expect("world");
    let key = world
        .register_brain_family(
            "sabotaged-mlp".to_owned(),
            Box::new(SabotagedMutationFamily {
                inner: MlpBrainFamily::new(),
            }),
        )
        .expect("register saboteur");
    let parent = spawn_parent(
        &mut world,
        key,
        100.0,
        100.0,
        MutationRates {
            primary: 0.0,
            secondary: 0.0,
        },
    );
    let parent_uid = uid_of(&world, parent);
    let children = drive_until_children(&mut world, &[parent_uid], 1, 64);

    let codec = dyn_codec_for(MLP_KIND);
    let diff = diff_genomes(
        &*codec,
        &genome_of(&world, parent),
        &genome_of(&world, children[0].id),
    )
    .expect("sabotaged diff");
    assert!(
        !diff.deltas.is_empty(),
        "heredity_proof NEGATIVE CONTROL FAILED: the restored e2d9aaa bug (fresh founder \
         material substituted for the parent's) produced an EMPTY zero-rate diff — the \
         proof machinery cannot detect the exact regression it exists to catch"
    );
}

/// Drive a single asexual lineage for `generations` consecutive births, returning the
/// (parent genome, child genome) pairs. Only the designated chain child receives energy
/// above the reproduction threshold, so the population grows linearly instead of
/// exploding exponentially.
fn drive_lineage_chain(
    family: &str,
    rates: MutationRates,
    generations: usize,
) -> Vec<(BrainGenomeEnvelope, BrainGenomeEnvelope)> {
    let mut config = reproduction_config(0.0);
    config.reproduction_child_energy = 0.4; // below the 0.5 threshold: no uncontrolled growth
    let mut world = WorldState::new(config).expect("world");
    let key = register_family(&mut world, family);
    let founder = spawn_parent(&mut world, key, 100.0, 100.0, rates);
    let mut chain_parent = uid_of(&world, founder);

    let mut pairs = Vec::with_capacity(generations);
    for _ in 0..4096_u64 {
        if pairs.len() >= generations {
            return pairs;
        }
        let children = drive_until_children(&mut world, &[chain_parent], 1, 512);
        let child = &children[0];
        let parent_id = world
            .agents()
            .iter_handles()
            .find(|id| world.agent_uid(*id) == Some(chain_parent))
            .expect("chain parent alive (nothing drains energy)");
        pairs.push((genome_of(&world, parent_id), genome_of(&world, child.id)));
        world
            .try_update_agent_runtime(child.id, |runtime| {
                runtime.energy = 1.5;
            })
            .expect("chain child energy above reproduction threshold");
        chain_parent = world.agent_uid(child.id).expect("chain child uid");
    }
    panic!("lineage stalled after {} generations", pairs.len());
}

#[test]
fn e2e_mutation_total_changed_loci_is_in_band() {
    const GENERATIONS: usize = 20;
    const RATE: f32 = 0.05;
    const ALPHA: f64 = 0.0005;

    fn case(family: &str) {
        let pairs = drive_lineage_chain(
            family,
            MutationRates {
                primary: RATE,
                secondary: RATE,
            },
            GENERATIONS,
        );
        assert_eq!(pairs.len(), GENERATIONS);
        let codec = dyn_codec_for(family);
        let mut total_changed = 0_u64;
        for (parent, child) in &pairs {
            let diff = diff_genomes(&*codec, parent, child).expect("e2e diff");
            total_changed += diff.summary.changed_loci as u64;
        }
        let support = changed_locus_distribution(family, GENERATIONS, RATE);
        let lower_tail = support_cdf(&support, total_changed);
        assert!(
            lower_tail > ALPHA && lower_tail < 1.0 - ALPHA,
            "heredity_proof: family={family} {GENERATIONS} consecutive diffs changed \
             {total_changed} loci total (lower_tail={lower_tail:.6}) — outside the exact \
             binomial band at rate {RATE}"
        );
    }
    for_each_family(case);
}

#[test]
fn e2e_zero_rate_every_consecutive_diff_is_empty() {
    const GENERATIONS: usize = 20;

    fn case(family: &str) {
        let pairs = drive_lineage_chain(
            family,
            MutationRates {
                primary: 0.0,
                secondary: 0.0,
            },
            GENERATIONS,
        );
        assert_eq!(pairs.len(), GENERATIONS);
        let codec = dyn_codec_for(family);
        for (index, (parent, child)) in pairs.iter().enumerate() {
            let diff = diff_genomes(&*codec, parent, child).expect("mirror diff");
            assert!(
                diff.deltas.is_empty(),
                "heredity_proof: family={family} generation {index}: zero-rate lineage \
                 changed {} loci — either heredity is broken or something re-randomizes \
                 brains on the spawn path (both catastrophic and previously undetectable)",
                diff.summary.changed_loci
            );
        }
    }
    for_each_family(case);
}

#[test]
fn spawn_determinism_is_schedule_independent() {
    fn case(family: &str) {
        let run_once = || {
            let mut world = WorldState::new(reproduction_config(0.0)).expect("world");
            let key = register_family(&mut world, family);
            let parent = spawn_parent(
                &mut world,
                key,
                100.0,
                100.0,
                MutationRates {
                    primary: 0.05,
                    secondary: 0.05,
                },
            );
            let parent_uid = uid_of(&world, parent);
            let children = drive_until_children(&mut world, &[parent_uid], 3, 128);
            std::iter::once(genome_of(&world, parent))
                .chain(children.iter().map(|child| genome_of(&world, child.id)))
                .map(|genome| {
                    let mut bytes = genome.family_id().as_str().as_bytes().to_vec();
                    bytes.extend_from_slice(genome.payload());
                    bytes
                })
                .collect::<Vec<_>>()
        };

        let single = run_once();
        #[cfg(feature = "parallel")]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(8)
                .build()
                .expect("bounded Rayon test pool");
            let threaded = pool.install(run_once);
            assert_eq!(
                single, threaded,
                "heredity_proof: family={family} spawn determinism changed between 1 and 8 \
                 threads — the spawn path's RNG consumption must not depend on the Rayon \
                 schedule"
            );
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _ = single;
        }
    }
    for_each_family(case);
}
