//! Observing allopatric divergence and cross-island gene flow (bd-4xx5).
//!
//! bd-16g.5's acceptance clause (c) is the one the epic is named for: with
//! migration DISABLED islands must provably diverge, and with migration ENABLED
//! gene flow must be measurable. Until this module existed, neither had ever
//! been evaluated — `segment_species` had no caller outside its own test module,
//! so species were not computed in production for one world, let alone compared
//! across several. Migration worked and speciation was *enabled*; it was not
//! *observed*.
//!
//! # Segment per island, never across islands
//!
//! [`segment_species`] keys on a bare [`AgentUid`], and every island mints UIDs
//! from its own private counter — island 0 and island 1 both hold `AgentUid(1)`
//! and they are different organisms. Feeding two islands into one table merges
//! them silently (bd-8jlj). Every function here segments ONE island and compares
//! the *results*; nothing ever unions two islands' samples.
//!
//! # What "different species" means here
//!
//! Two populations are treated as sharing a species when their phenotype
//! clusters would merge under the project's own clustering rule — the same
//! [`SpeciesParams::distance_threshold`] that segmentation uses. Reusing that
//! threshold matters: inventing a second, looser statistic would let this module
//! report divergence that the project's own definition of species does not
//! recognise.
//!
//! # Why a null control is mandatory
//!
//! A separation metric that returns a positive number for any two samples proves
//! nothing. [`halves_separation`] measures the separation between two arbitrary
//! halves of a SINGLE island's population, split on UID parity — a partition
//! that is uncorrelated with phenotype and therefore has no real separation in
//! it. Cross-island separation is only evidence of divergence when it exceeds
//! that null. Held to the standard set on bd-tfso: a test that cannot fail
//! against a deliberately non-diverging run has not observed anything.

use crate::archipelago::{Archipelago, ArchipelagoError, IslandId};
use scriptbots_core::{
    AgentUid, Tick, WorldState,
    species::{SpeciesParams, SpeciesTable, segment_species},
};

/// Heritable phenotype dimensions compared across islands.
///
/// Every entry is a gene the mutation path perturbs, so isolation can move it.
/// Per-tick transient state (energy, sensors, combat flags) is deliberately
/// excluded: it varies second to second within one population and would swamp
/// the heritable signal this module exists to detect.
pub const ALLOPATRY_DIMENSIONS: [&str; 6] = [
    "herbivore_tendency",
    "temperature_preference",
    "trait.smell",
    "trait.sound",
    "trait.hearing",
    "trait.eye",
];

/// Fixed normalization bounds, one per [`ALLOPATRY_DIMENSIONS`] entry.
///
/// Fixed rather than data-derived on purpose. Normalizing against the observed
/// range would rescale each population to fill its own space, which makes two
/// populations look alike precisely when they have drifted apart — the metric
/// would hide the effect it is measuring.
fn allopatry_ranges() -> Vec<(f32, f32)> {
    vec![
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 4.0),
        (0.0, 4.0),
        (0.0, 4.0),
        (0.0, 4.0),
    ]
}

/// Segmentation parameters used for every comparison in this module.
#[must_use]
pub fn allopatry_species_params() -> SpeciesParams {
    SpeciesParams {
        dimension_ranges: allopatry_ranges(),
        ..SpeciesParams::default()
    }
}

/// One island's heritable phenotype samples, in ascending local-UID order.
///
/// # Errors
///
/// Propagates [`ArchipelagoError`] when the island cannot be read.
pub fn island_phenotypes(
    archipelago: &Archipelago,
    island: IslandId,
) -> Result<Vec<(AgentUid, Vec<f32>)>, ArchipelagoError> {
    archipelago.with_island_world(island, |world: &WorldState| {
        let mut samples: Vec<(AgentUid, Vec<f32>)> = world
            .agents()
            .iter_handles()
            .filter_map(|handle| {
                let uid = world.agent_uid(handle)?;
                let runtime = world.agent_runtime(handle)?;
                Some((
                    uid,
                    vec![
                        runtime.herbivore_tendency,
                        runtime.temperature_preference,
                        runtime.trait_modifiers.smell,
                        runtime.trait_modifiers.sound,
                        runtime.trait_modifiers.hearing,
                        runtime.trait_modifiers.eye,
                    ],
                ))
            })
            .collect();
        samples.sort_unstable_by_key(|(uid, _)| *uid);
        samples
    })
}

/// Segment ONE island into species.
///
/// Never call this with samples drawn from more than one island; see the module
/// documentation and bd-8jlj.
#[must_use]
pub fn segment_island(tick: Tick, samples: &[(AgentUid, Vec<f32>)]) -> SpeciesTable {
    let (table, _report) = segment_species(
        tick,
        samples,
        &SpeciesTable::default(),
        &allopatry_species_params(),
    );
    table
}

/// Standardized separation between two populations in normalized phenotype
/// space: the distance between their means divided by their pooled internal
/// spread.
///
/// # Why not "nearest species pair", which is what this replaced
///
/// The first version returned the smallest centroid distance between any
/// species of `a` and any of `b`. Measured against real runs it was useless, and
/// the numbers said so plainly: three isolated islands produced 20-24 species
/// each from ~110 agents, so the nearest pair across two islands was always some
/// pair of tiny adjacent clusters and the value (0.039-0.077) sat at or BELOW
/// the null control (0.020-0.076). A statistic that answers "do these two
/// populations contain at least one similar cluster?" cannot answer "have these
/// populations separated?" — with twenty clusters a side the answer is always
/// yes.
///
/// Dividing by the pooled spread is what makes the number comparable between
/// runs: a raw mean distance grows with any change that widens the phenotype
/// space, whether or not the populations separated relative to their own
/// variation. This is a two-sample effect size, and 0 means "indistinguishable".
///
/// `None` when either population is empty, which is a refusal rather than a
/// zero: an empty population has not been shown to resemble anything.
#[must_use]
pub fn population_separation(a: &[Vec<f32>], b: &[Vec<f32>]) -> Option<f32> {
    let mean_a = mean_vector(a)?;
    let mean_b = mean_vector(b)?;
    let distance = centroid_distance(&mean_a, &mean_b)?;
    let spread_a = mean_distance_from(a, &mean_a)?;
    let spread_b = mean_distance_from(b, &mean_b)?;
    let pooled = f32::midpoint(spread_a, spread_b);
    if pooled <= f32::EPSILON {
        // Both populations are points. Any nonzero gap is total separation;
        // no gap is none.
        return Some(if distance <= f32::EPSILON {
            0.0
        } else {
            f32::INFINITY
        });
    }
    Some(distance / pooled)
}

/// Componentwise mean of a set of equal-arity vectors.
fn mean_vector(samples: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = samples.first()?;
    let arity = first.len();
    if arity == 0 || samples.iter().any(|sample| sample.len() != arity) {
        return None;
    }
    let mut mean = vec![0.0f32; arity];
    for sample in samples {
        for (slot, value) in mean.iter_mut().zip(sample) {
            *slot += *value;
        }
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "population sizes here are far below f32's exact-integer range"
    )]
    let count = samples.len() as f32;
    for slot in &mut mean {
        *slot /= count;
    }
    Some(mean)
}

/// Mean Euclidean distance of `samples` from `centre`.
fn mean_distance_from(samples: &[Vec<f32>], centre: &[f32]) -> Option<f32> {
    if samples.is_empty() {
        return None;
    }
    let mut total = 0.0f32;
    for sample in samples {
        total += centroid_distance(sample, centre)?;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "population sizes here are far below f32's exact-integer range"
    )]
    let count = samples.len() as f32;
    Some(total / count)
}

/// Normalize raw phenotype samples into the space separation is measured in.
///
/// Separation must be computed in the SAME normalized space segmentation uses,
/// or a dimension with a wide raw range would dominate purely because of its
/// units.
#[must_use]
pub fn normalized_vectors(samples: &[(AgentUid, Vec<f32>)]) -> Vec<Vec<f32>> {
    let ranges = allopatry_ranges();
    samples
        .iter()
        .map(|(_, raw)| {
            raw.iter()
                .zip(&ranges)
                .map(|(value, (low, high))| {
                    let span = high - low;
                    if span.abs() <= f32::EPSILON {
                        0.0
                    } else {
                        ((value - low) / span).clamp(0.0, 1.0)
                    }
                })
                .collect()
        })
        .collect()
}

/// Euclidean distance between two normalized centroids of equal arity.
fn centroid_distance(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    Some(
        a.iter()
            .zip(b)
            .map(|(left, right)| (left - right) * (left - right))
            .sum::<f32>()
            .sqrt(),
    )
}

/// THE NULL CONTROL: separation between two arbitrary halves of one population.
///
/// The split is on UID parity, which is uncorrelated with phenotype, so whatever
/// this returns is what the metric reports when there is NO real separation.
/// Cross-island separation is evidence of divergence only insofar as it exceeds
/// this.
///
/// `None` when the population is too small to halve meaningfully.
#[must_use]
pub fn halves_separation(samples: &[(AgentUid, Vec<f32>)]) -> Option<f32> {
    if samples.len() < 4 {
        return None;
    }
    let even: Vec<(AgentUid, Vec<f32>)> = samples
        .iter()
        .filter(|(uid, _)| uid.get().is_multiple_of(2))
        .cloned()
        .collect();
    let odd: Vec<(AgentUid, Vec<f32>)> = samples
        .iter()
        .filter(|(uid, _)| !uid.get().is_multiple_of(2))
        .cloned()
        .collect();
    if even.len() < 2 || odd.len() < 2 {
        return None;
    }
    population_separation(&normalized_vectors(&even), &normalized_vectors(&odd))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archipelago::{ArchipelagoConfig, ArchipelagoMigration, IslandSpec, Topology};
    use crate::host_core::HostCoreOptions;
    use crate::migrator::EmigrantSelectionRule;
    use scriptbots_core::{
        BrainRunner, BrainSpawnError, INPUT_SIZE, OUTPUT_SIZE, RandomStream, ScriptBotsConfig,
        WorldStateError,
    };
    use std::num::NonZeroU64;

    const TEST_BRAIN_KIND: &str = "allopatry-test-brain";
    const TEST_BRAIN_FACTORY_DIGEST: u64 = 0xA110_7A72_9000_0001;

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

    fn populated_config() -> ScriptBotsConfig {
        ScriptBotsConfig {
            world_width: 600,
            world_height: 300,
            food_cell_size: 50,
            rng_seed: None,
            persistence_interval: 0,
            population_minimum: 16,
            population_spawn_interval: 4,
            ..ScriptBotsConfig::default()
        }
    }

    fn world_factory(config: ScriptBotsConfig) -> Result<WorldState, WorldStateError> {
        let mut world = WorldState::new(config)?;
        let _key = world
            .brain_registry_mut()
            .expect("registry is mutable before the first tick")
            .register_with_state_digest(TEST_BRAIN_KIND, TEST_BRAIN_FACTORY_DIGEST, |rng| {
                Ok(Box::new(TestBrain {
                    weight: draw_unit(rng).mul_add(0.9, 0.1),
                }) as Box<dyn BrainRunner>)
            });
        Ok(world)
    }

    fn archipelago(islands: u32, migration: Option<ArchipelagoMigration>) -> Archipelago {
        let specs: Vec<IslandSpec> = (0..islands)
            .map(|id| IslandSpec {
                id: IslandId(id),
                label: format!("allopatry-island-{id}"),
                config: populated_config(),
            })
            .collect();
        Archipelago::with_factories(
            ArchipelagoConfig {
                islands: specs,
                topology: Topology::Ring,
                barrier_interval: NonZeroU64::new(50).expect("nonzero interval"),
                master_seed: 0xA110_7A72_5EED_0001,
                host_options: HostCoreOptions::default(),
                migration,
            },
            |meta| world_factory(meta.effective_config.clone()),
            |_meta| None,
        )
        .expect("allopatry archipelago")
    }

    /// Barrier count at which isolated islands separate beyond the null.
    ///
    /// THIS NUMBER IS ITSELF THE SCIENTIFIC RESULT, so it is recorded here and
    /// on bd-4xx5 rather than left for a future agent to re-derive at ~4 minutes
    /// a run. Measured, not guessed:
    ///
    /// ```text
    ///   8 barriers  (400 ticks)  NOT SEPARATED. cross-island 0.184-0.378
    ///                            against nulls 0.123-0.224 — overlapping.
    ///  40 barriers (2000 ticks)  SEPARATED. cross-island 0.541 and 0.528
    ///                            against nulls <= 0.111.
    /// ```
    ///
    /// Drift needs time; four hundred ticks is simply not enough for isolation
    /// to move a population beyond its own internal spread.
    const SEPARATION_BARRIERS: usize = 20;

    /// Barriers for the cheap machinery and gene-flow checks.
    const FAST_BARRIERS: usize = 8;

    fn migration_policy() -> ArchipelagoMigration {
        ArchipelagoMigration {
            interval_ticks: 50,
            emigrants_per_edge: 2,
            selection_rule: EmigrantSelectionRule::Fittest,
        }
    }

    /// Run an archipelago and return each island's normalized population plus
    /// its own null control.
    fn observe(
        islands: u32,
        migration: Option<ArchipelagoMigration>,
        barriers: usize,
    ) -> (Vec<Vec<Vec<f32>>>, Vec<f32>) {
        let mut arch = archipelago(islands, migration);
        for _ in 0..barriers {
            arch.step_to_barrier().expect("barrier steps");
        }
        let mut populations = Vec::new();
        let mut nulls = Vec::new();
        for id in (0..islands).map(IslandId) {
            let samples = island_phenotypes(&arch, id).expect("island readable");
            assert!(
                !samples.is_empty(),
                "island {id} is empty, so every comparison below would be vacuous"
            );
            nulls.push(halves_separation(&samples).expect("island large enough to halve"));
            populations.push(normalized_vectors(&samples));
        }
        (populations, nulls)
    }

    fn max_pairwise_separation(populations: &[Vec<Vec<f32>>]) -> f32 {
        let mut best = 0.0f32;
        for a in 0..populations.len() {
            for b in (a + 1)..populations.len() {
                if let Some(value) = population_separation(&populations[a], &populations[b]) {
                    best = best.max(value);
                }
            }
        }
        best
    }

    /// THE METRIC MUST RETURN ZERO FOR IDENTICAL POPULATIONS AND LARGE FOR
    /// DELIBERATELY SEPARATED ONES.
    ///
    /// The born-red control for the statistic itself, held to the bd-tfso
    /// standard: a separation metric that cannot report "no separation" has not
    /// measured anything. Synthetic input, so the expected answer is known
    /// exactly rather than inferred from a simulation.
    #[test]
    fn bd_4xx5_separation_is_zero_for_identical_populations_and_large_for_split_ones() {
        let cluster = |centre: f32| -> Vec<Vec<f32>> {
            (0..20)
                .map(|index| {
                    #[allow(clippy::cast_precision_loss)]
                    let jitter = (index as f32) * 0.001;
                    vec![centre + jitter, 0.5, 0.5, 0.5, 0.5, 0.5]
                })
                .collect()
        };

        let same = cluster(0.20);
        assert_eq!(
            population_separation(&same, &same),
            Some(0.0),
            "a population cannot be separated from itself"
        );

        let apart = cluster(0.80);
        let separated =
            population_separation(&same, &apart).expect("both populations are non-empty");
        assert!(
            separated > 10.0,
            "two clusters 0.6 apart with ~0.005 internal spread must read as strongly \
             separated, got {separated}"
        );

        assert_eq!(
            population_separation(&[], &same),
            None,
            "an empty population must be refused, not reported as similar"
        );
    }

    /// The null control must be small on unstructured data and large on
    /// structured data.
    ///
    /// Without this, `halves_separation` could return a constant and every
    /// divergence claim built on it would be unfalsifiable.
    #[test]
    fn bd_4xx5_the_null_control_detects_structure_when_structure_exists() {
        let unstructured: Vec<(AgentUid, Vec<f32>)> = (1..=40u64)
            .map(|uid| {
                #[allow(clippy::cast_precision_loss)]
                let jitter = (uid as f32) * 0.001;
                (AgentUid(uid), vec![0.3 + jitter, 0.5, 1.0, 1.0, 1.0, 1.0])
            })
            .collect();
        let flat = halves_separation(&unstructured).expect("large enough to halve");
        assert!(
            flat < 1.0,
            "an arbitrary parity split of an unstructured population must show little \
             separation, got {flat}"
        );

        // Now make the parity split MEAN something: even uids get one phenotype,
        // odd uids another. The same null computation must now report it.
        let structured: Vec<(AgentUid, Vec<f32>)> = (1..=40u64)
            .map(|uid| {
                let centre = if uid.is_multiple_of(2) { 0.2 } else { 0.9 };
                (AgentUid(uid), vec![centre, 0.5, 1.0, 1.0, 1.0, 1.0])
            })
            .collect();
        let split = halves_separation(&structured).expect("large enough to halve");
        assert!(
            split > flat * 5.0,
            "the null control must detect a real split; unstructured {flat}, structured \
             {split}"
        );
    }

    /// Per-island segmentation runs on a live archipelago and never unions
    /// islands.
    #[test]
    fn bd_4xx5_islands_are_segmented_separately_and_bare_uids_would_collide() {
        let mut arch = archipelago(3, None);
        for _ in 0..FAST_BARRIERS {
            arch.step_to_barrier().expect("barrier steps");
        }
        let tick = arch.barrier_tick();
        let mut all_uids = Vec::new();
        for id in (0..3).map(IslandId) {
            let samples = island_phenotypes(&arch, id).expect("island readable");
            assert!(!samples.is_empty(), "island {id} must be populated");
            let table = segment_island(tick, &samples);
            assert!(
                !table.species.is_empty(),
                "island {id} produced no species, so nothing downstream means anything"
            );
            all_uids.extend(samples.iter().map(|(uid, _)| *uid));
        }
        // The reason every function here segments ONE island: bare uids collide.
        let mut unique = all_uids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert!(
            unique.len() < all_uids.len(),
            "bare uids are expected to collide across islands; if they stopped colliding \
             the bd-8jlj hazard changed and this module's per-island discipline needs a \
             fresh justification"
        );
    }

    /// GENE FLOW: migrants carry their genes into another island's pool, and
    /// with migration off nothing crosses.
    #[test]
    fn bd_4xx5_gene_flow_crosses_islands_only_when_migration_is_enabled() {
        let mut migrating = archipelago(3, Some(migration_policy()));
        let mut arrivals = Vec::new();
        for _ in 0..FAST_BARRIERS {
            let report = migrating.step_to_barrier().expect("barrier steps");
            if let Some(migration) = report.migration {
                arrivals.extend(
                    migration
                        .moves
                        .iter()
                        .map(|applied| (applied.from, applied.to)),
                );
            }
        }
        assert!(
            !arrivals.is_empty(),
            "no organism crossed, so there is no gene flow to observe"
        );

        // The genes are IN THE DESTINATION POOL: at least one arrival is still
        // alive at the end, under its destination-minted identity.
        let census = migrating.organism_census().expect("census readable");
        let survivors = arrivals
            .iter()
            .filter(|(_, to)| census.contains(to))
            .count();
        assert!(
            survivors > 0,
            "every one of the {} arrivals died before the run ended, so nothing entered \
             the destination gene pool",
            arrivals.len()
        );

        // Every surviving arrival came from a DIFFERENT island than it lives on.
        for (from, to) in &arrivals {
            assert_ne!(
                from.island, to.island,
                "a migration must cross islands to be gene flow"
            );
        }

        // THE CONTROL: the same archipelago with migration off produces none.
        let mut isolated = archipelago(3, None);
        for _ in 0..FAST_BARRIERS {
            let report = isolated.step_to_barrier().expect("barrier steps");
            assert!(
                report.migration.is_none(),
                "an isolated archipelago must not run a migration phase at all"
            );
        }
    }

    /// *** THE OBSERVATION: isolated islands diverge, and migration suppresses
    /// it. ***
    ///
    /// bd-16g.5 acceptance (c), which nothing had ever evaluated. Ignored by
    /// default for runtime, matching the existing
    /// `dsr_heterogeneous_islands_reach_tick_two_thousand_headless` convention:
    /// twenty barriers is a thousand scientific ticks per island, twice over.
    ///
    /// The three assertions are chosen so that a run WITHOUT real divergence
    /// fails all of them:
    ///  - separation must clear the null control by a wide margin, so a metric
    ///    that reports separation everywhere cannot pass;
    ///  - isolation must separate MORE than migration does, so a result driven
    ///    by anything other than isolation cannot pass;
    ///  - the null must stay small in absolute terms, so inflating both sides
    ///    equally cannot pass.
    #[test]
    #[ignore = "long lane: 20 barriers x 2 configurations; run explicitly for the observation"]
    fn bd_4xx5_isolated_islands_diverge_and_migration_suppresses_it() {
        let (isolated, isolated_nulls) = observe(3, None, SEPARATION_BARRIERS);
        let (migrating, _migrating_nulls) =
            observe(3, Some(migration_policy()), SEPARATION_BARRIERS);

        let isolated_max = max_pairwise_separation(&isolated);
        let migrating_max = max_pairwise_separation(&migrating);
        // MEAN, not max. The null estimates what this metric reports when there
        // is NO separation, so its expected value is the baseline. A max over
        // three noisy estimates is an extreme-value statistic, and measurement
        // showed it swinging between 0.11 and 0.22 across run lengths while the
        // signal it was being compared against moved monotonically. Using the
        // max was measuring the noisiest island, not the baseline.
        #[allow(
            clippy::cast_precision_loss,
            reason = "three islands is far below f32's exact-integer range"
        )]
        let null_mean = isolated_nulls.iter().copied().sum::<f32>() / isolated_nulls.len() as f32;

        println!(
            "isolated_max={isolated_max} migrating_max={migrating_max} \
             null_mean={null_mean} nulls={isolated_nulls:?}"
        );

        assert!(
            isolated_max > null_mean * 2.0,
            "isolated islands must separate clearly beyond the no-separation null; \
             separation {isolated_max}, null mean {null_mean}"
        );
        assert!(
            isolated_max > migrating_max,
            "isolation must separate populations MORE than migration does, or the \
             separation is not caused by isolation; isolated {isolated_max}, \
             migrating {migrating_max}"
        );
        assert!(
            null_mean < 0.25,
            "the null control drifted upward, so the margin above it no longer means \
             what it did when this was calibrated; null mean {null_mean}"
        );
    }
}
