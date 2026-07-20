//! Deterministic, bias-controlled brain-family tournament harness (bd-16g.12.1).
//!
//! This module owns the fairness core: a pure [`plan`] that turns a [`TournamentSpec`]
//! into match plans with equal cohorts, order-swap balance, and hash-allocated seeds —
//! plus the typed outcomes each match must emit. An unfair tournament is strictly worse
//! than none: it manufactures citable-looking numbers out of spawn order or cohort size,
//! and they will be believed. Every guard here exists to make a biased pairing
//! unrepresentable rather than merely unlikely.
//!
//! Execution deliberately drives the *existing* headless simulation driver (the same one
//! replay verification uses) and never downcasts or special-cases a brain family; the
//! matched-seed runner (`bd-2z0.5.5`) replaces that adapter when it lands.

use scriptbots_brain::BrainKind;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

pub use execution::{MatchRunReport, enforce_no_config_drift, run_match, run_tournament};

/// Largest match count one spec may emit (orders × seeds). Guards the factorial policy
/// from scheduling an untestable tournament by accident.
const MAX_MATCHES: usize = 4_096;

/// What a tournament runs. Config layers resolve exactly like the application's, and the
/// harness compares every arm's config digest against the plan's — one mutated knob
/// anywhere is a bug, not a finding.
#[derive(Debug, Clone, PartialEq)]
pub struct TournamentSpec {
    /// Families entered. Equal cohort size per family is enforced.
    pub families: Vec<BrainKind>,
    /// Root seeds; each produces the same order assignments.
    pub seeds: Vec<u64>,
    /// Tick budget per match.
    pub ticks: u64,
    /// Total agents spawned per match, split equally across families.
    pub cohort_size: usize,
    /// How spawn-order assignments are generated and balanced.
    pub order_policy: OrderPolicy,
    /// Tournaments default to closed worlds: an open world silently resurrects extinct
    /// families via `stage_population`, and "survival share" then measures the respawner.
    pub closed: bool,
    /// Config files layered beneath every arm, in application order.
    pub config_layers: Vec<PathBuf>,
}

/// Spawn-order assignment generation. Every policy must place each family in each
/// position an equal number of times per seed; `plan` refuses anything it cannot defend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderPolicy {
    /// Exactly the two assignments; valid only for two families.
    BothAssignments,
    /// Cyclic rotations; each family occupies each position once per seed.
    BalancedLatinSquare,
    /// Every permutation; size-guarded by `MAX_MATCHES`.
    AllPermutations,
}

/// Stable match identity, hash-derived from the root seed and match index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MatchId(pub u64);

/// One match: a seed, a spawn order, equal cohorts, and hash-allocated world/brain seeds.
/// The full composition is recorded so any suspicious result can be traced to the exact
/// world that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchPlan {
    pub match_id: MatchId,
    /// Root spec seed this match draws from.
    pub seed: u64,
    /// Family spawn order; position `spawn_order_index` is recorded on the outcome so an
    /// order effect can be measured rather than hidden.
    pub spawn_order: Vec<BrainKind>,
    /// Which assignment within the order set this match is, so the rating layer can test
    /// for an order effect instead of assuming it away.
    pub spawn_order_index: u32,
    /// Equal cohort composition per family.
    pub cohort: BTreeMap<BrainKind, usize>,
    pub world_seed: u64,
    pub brain_seed: u64,
}

/// One family's result inside one match.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyOutcome {
    /// Share of the match's final agent population belonging to this family.
    pub survival_share: f64,
    /// Share of the match's final total energy belonging to this family.
    pub biomass_share: f64,
    pub mean_lineage_depth: f64,
    pub max_lineage_depth: u32,
    /// First tick the family had no live agent, if it went extinct.
    pub extinct_at: Option<u64>,
    /// Reserved for the novelty archive (bd-16g.6); always `None` here.
    pub novelty_coverage: Option<f64>,
}

/// Typed per-match result.
///
/// `per_family` keys are owned `String`s, not `BrainKind`: `BrainKind`
/// wraps a `&'static str`, so a deserializable map keyed by it would need
/// either `'de: 'static` (rejected by serde's derive) or a `Box::leak`
/// intern table (banned by repo rule). Owned keys keep reports
/// round-trippable; the `BrainKind`-typed helpers preserve the ergonomic
/// API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchOutcome {
    pub match_id: MatchId,
    pub ticks_run: u64,
    /// Which assignment within the order set produced this outcome — the order effect is
    /// a measurement, not a nuisance to be hidden from the rating layer.
    pub spawn_order_index: u32,
    pub per_family: BTreeMap<String, FamilyOutcome>,
    /// Qualifiers the leaderboard must never lose (open-world respawn, early stop).
    pub warnings: Vec<String>,
}

impl MatchOutcome {
    /// Outcome for one brain family, if present.
    #[must_use]
    pub fn family(&self, kind: BrainKind) -> Option<&FamilyOutcome> {
        self.per_family.get(kind.as_str())
    }

    /// Record the outcome for one brain family.
    pub fn set_family(&mut self, kind: BrainKind, outcome: FamilyOutcome) {
        self.per_family.insert(kind.as_str().to_string(), outcome);
    }

    /// Iterate families by name (registration-order stable).
    pub fn families(&self) -> impl Iterator<Item = (&str, &FamilyOutcome)> + '_ {
        self.per_family
            .iter()
            .map(|(name, outcome)| (name.as_str(), outcome))
    }
}

/// A pairing the harness refuses to defend.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TournamentError {
    #[error("no families entered")]
    EmptyFamilies,
    #[error("no seeds supplied")]
    EmptySeeds,
    #[error("tick budget must be at least 1")]
    ZeroTicks,
    #[error("family {family} is entered more than once")]
    DuplicateFamilies { family: String },
    #[error(
        "cohort size {cohort_size} is not divisible by {families} families; unequal cohorts change the reproduction operator each family experiences"
    )]
    UnequalCohorts { cohort_size: usize, families: usize },
    #[error("config digest drift across arms: expected {expected}, found {found}")]
    ConfigDrift { expected: String, found: String },
    #[error("cross-kind mating: child {child} has parents of kinds {parent_a} and {parent_b}")]
    CrossKindMating {
        child: u64,
        parent_a: String,
        parent_b: String,
    },
    #[error("{reason}")]
    UnbalancedOrders { reason: String },
}

/// splitmix64 finalizer — the entire seed-allocation scheme is a pure hash of
/// (root_seed, match_index, lane), never a draw from a shared RNG, so parallelizing
/// matches with any `--jobs` cannot change an assignment.
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Lane-tagged index hash: lane 0 → world seed, 1 → brain seed, 2 → match id.
fn mix(index: u64, lane: u64) -> u64 {
    splitmix64(index.wrapping_mul(2).wrapping_add(lane))
}

fn world_seed(root_seed: u64, match_index: u64) -> u64 {
    splitmix64(root_seed ^ mix(match_index, 0))
}

fn brain_seed(root_seed: u64, match_index: u64) -> u64 {
    splitmix64(root_seed ^ mix(match_index, 1))
}

fn match_id(root_seed: u64, match_index: u64) -> MatchId {
    MatchId(splitmix64(root_seed ^ mix(match_index, 2)))
}

/// Spawn-order assignments for one seed under the policy, in deterministic order.
fn order_assignments(
    families: &[BrainKind],
    policy: OrderPolicy,
) -> Result<Vec<Vec<BrainKind>>, TournamentError> {
    let count = families.len();
    match policy {
        OrderPolicy::BothAssignments => {
            if count != 2 {
                return Err(TournamentError::UnbalancedOrders {
                    reason: format!("BothAssignments requires exactly 2 families, got {count}"),
                });
            }
            Ok(vec![families.to_vec(), {
                let mut swapped = families.to_vec();
                swapped.swap(0, 1);
                swapped
            }])
        }
        OrderPolicy::BalancedLatinSquare => {
            if count < 2 {
                return Err(TournamentError::UnbalancedOrders {
                    reason: "BalancedLatinSquare requires at least 2 families".to_owned(),
                });
            }
            Ok((0..count)
                .map(|rotation| {
                    families[rotation..]
                        .iter()
                        .chain(families[..rotation].iter())
                        .copied()
                        .collect()
                })
                .collect())
        }
        OrderPolicy::AllPermutations => {
            if count < 2 {
                return Err(TournamentError::UnbalancedOrders {
                    reason: "AllPermutations requires at least 2 families".to_owned(),
                });
            }
            // Permutations are enumerated from the lexicographically first ordering so
            // the plan is identical regardless of the caller's family order.
            let mut permutation = families.to_vec();
            permutation.sort_unstable();
            let mut assignments = vec![permutation.clone()];
            while lexicographic_next_permutation(&mut permutation) {
                assignments.push(permutation.clone());
            }
            Ok(assignments)
        }
    }
}

/// Advance to the next permutation in lexicographic order; false when exhausted.
fn lexicographic_next_permutation<T: Ord>(items: &mut [T]) -> bool {
    if items.len() < 2 {
        return false;
    }
    let mut pivot = items.len() - 2;
    while items[pivot] >= items[pivot + 1] {
        if pivot == 0 {
            return false;
        }
        pivot -= 1;
    }
    let mut successor = items.len() - 1;
    while items[successor] <= items[pivot] {
        successor -= 1;
    }
    items.swap(pivot, successor);
    items[pivot + 1..].reverse();
    true
}

/// Turn a spec into its full, defensible match plan set. Pure: no RNG is touched, and
/// two calls with the same spec return byte-identical plans.
pub fn plan(spec: &TournamentSpec) -> Result<Vec<MatchPlan>, TournamentError> {
    if spec.families.is_empty() {
        return Err(TournamentError::EmptyFamilies);
    }
    if spec.seeds.is_empty() {
        return Err(TournamentError::EmptySeeds);
    }
    if spec.ticks == 0 {
        return Err(TournamentError::ZeroTicks);
    }
    let mut sorted_families = spec.families.clone();
    sorted_families.sort_unstable();
    for window in sorted_families.windows(2) {
        if window[0] == window[1] {
            return Err(TournamentError::DuplicateFamilies {
                family: window[0].as_str().to_owned(),
            });
        }
    }
    let family_count = spec.families.len();
    if !spec.cohort_size.is_multiple_of(family_count) {
        return Err(TournamentError::UnequalCohorts {
            cohort_size: spec.cohort_size,
            families: family_count,
        });
    }
    let orders = if family_count == 1 {
        vec![spec.families.clone()]
    } else {
        order_assignments(&spec.families, spec.order_policy)?
    };
    let total_matches = orders.len() * spec.seeds.len();
    if total_matches > MAX_MATCHES {
        return Err(TournamentError::UnbalancedOrders {
            reason: format!(
                "{total_matches} matches exceeds the {MAX_MATCHES} affordability guard"
            ),
        });
    }

    // The whole point: assert the balance rather than trusting the generator. Every
    // family must occupy every spawn-order position the same number of times per seed.
    for position in 0..family_count {
        let mut counts = BTreeMap::new();
        for order in &orders {
            *counts.entry(order[position]).or_insert(0_usize) += 1;
        }
        let expected = counts.values().next().copied().unwrap_or(0);
        if counts.values().any(|count| *count != expected) {
            return Err(TournamentError::UnbalancedOrders {
                reason: format!(
                    "position {position} is not occupied equally by every family: {counts:?}"
                ),
            });
        }
    }

    // The plan is pure; the log is the audit trail: every pairing decision is visible
    // without re-running anything.
    tracing::info!(
        target: "scriptbots::tournament",
        families = spec.families.len(),
        seeds = spec.seeds.len(),
        ticks = spec.ticks,
        cohort_size = spec.cohort_size,
        matches = total_matches,
        order_policy = ?spec.order_policy,
        closed = spec.closed,
        "tournament plan accepted"
    );

    let per_family = spec.cohort_size / family_count;
    let mut plans = Vec::with_capacity(total_matches);
    for (seed_ordinal, root_seed) in spec.seeds.iter().enumerate() {
        for (order_ordinal, spawn_order) in orders.iter().enumerate() {
            let match_index = (seed_ordinal * orders.len() + order_ordinal) as u64;
            plans.push(MatchPlan {
                match_id: match_id(*root_seed, match_index),
                seed: *root_seed,
                spawn_order: spawn_order.clone(),
                spawn_order_index: order_ordinal as u32,
                cohort: spec
                    .families
                    .iter()
                    .map(|family| (*family, per_family))
                    .collect(),
                world_seed: world_seed(*root_seed, match_index),
                brain_seed: brain_seed(*root_seed, match_index),
            });
        }
    }
    Ok(plans)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MLP: BrainKind = BrainKind::new("mlp");
    const DWRAON: BrainKind = BrainKind::new("dwraon");
    const ASSEMBLY: BrainKind = BrainKind::new("assembly");
    const NEURO: BrainKind = BrainKind::new("neuro");

    fn spec(families: Vec<BrainKind>, seeds: Vec<u64>, cohort_size: usize) -> TournamentSpec {
        TournamentSpec {
            families,
            seeds,
            ticks: 2_000,
            cohort_size,
            order_policy: OrderPolicy::BalancedLatinSquare,
            closed: true,
            config_layers: Vec::new(),
        }
    }

    #[test]
    fn plan_balances_every_family_in_every_position() {
        for (families, expected_orders) in [
            (vec![MLP, DWRAON], 2_usize),
            (vec![MLP, DWRAON, ASSEMBLY], 3),
            (vec![MLP, DWRAON, ASSEMBLY, NEURO], 4),
        ] {
            let family_count = families.len();
            let planned = plan(&spec(families.clone(), vec![1], family_count * 10))
                .expect("balanced spec plans");
            assert_eq!(planned.len(), expected_orders, "one order set per seed");
            for position in 0..family_count {
                for family in &families {
                    let count = planned
                        .iter()
                        .filter(|plan| plan.spawn_order[position] == *family)
                        .count();
                    assert_eq!(
                        count,
                        1,
                        "family {} must occupy position {position} exactly once per seed",
                        family.as_str()
                    );
                }
            }
            for plan in &planned {
                assert!(
                    plan.cohort.values().all(|count| *count == 10),
                    "cohorts are equal"
                );
            }
        }
    }

    #[test]
    fn plan_is_pure_and_byte_identical_across_calls() {
        let input = spec(vec![MLP, DWRAON, ASSEMBLY], vec![7, 11, 13], 30);
        let first = plan(&input).expect("plan one");
        let second = plan(&input).expect("plan two");
        assert_eq!(first, second, "plan() is pure");
        assert_eq!(first.len(), 9, "3 seeds x 3 rotations");
        let ids: Vec<MatchId> = first.iter().map(|plan| plan.match_id).collect();
        let mut unique = ids.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(ids.len(), unique.len(), "match ids are unique");
    }

    #[test]
    fn seed_allocation_matches_the_golden_table() {
        // A hash change is a visible diff here, not a silent re-randomization of every
        // tournament that depends on this table.
        let golden: [(u64, u64, u64); 4] = [
            (0, 5_592_132_763_777_985_307, 9_129_838_320_742_759_465),
            (1, 2_139_811_525_164_838_579, 4_875_857_236_239_627_170),
            (2, 10_247_000_711_120_590_919, 14_481_633_987_070_419_362),
            (3, 3_420_186_050_861_303_280, 7_974_615_062_405_353_404),
        ];
        for (index, expected_world, expected_brain) in golden {
            assert_eq!(world_seed(42, index), expected_world, "world seed {index}");
            assert_eq!(brain_seed(42, index), expected_brain, "brain seed {index}");
        }
    }

    #[test]
    fn both_assignments_requires_exactly_two_families() {
        let mut input = spec(vec![MLP, DWRAON], vec![5], 20);
        input.order_policy = OrderPolicy::BothAssignments;
        let planned = plan(&input).expect("two families balance");
        assert_eq!(planned.len(), 2, "[A,B] and [B,A]");
        input.families.push(ASSEMBLY);
        input.cohort_size = 30;
        assert!(
            matches!(plan(&input), Err(TournamentError::UnbalancedOrders { .. })),
            "three families cannot use BothAssignments"
        );
    }

    #[test]
    fn unequal_cohorts_are_rejected_with_typed_error() {
        let error = plan(&spec(vec![MLP, DWRAON, ASSEMBLY], vec![1], 10))
            .expect_err("10 is not divisible by 3");
        assert!(
            matches!(
                error,
                TournamentError::UnequalCohorts {
                    cohort_size: 10,
                    families: 3
                }
            ),
            "expected typed UnequalCohorts, got {error}"
        );
    }

    #[test]
    fn duplicate_families_are_rejected() {
        let error = plan(&spec(vec![MLP, DWRAON, MLP], vec![1], 30))
            .expect_err("a family entered twice breaks balance semantics");
        assert!(
            matches!(error, TournamentError::DuplicateFamilies { .. }),
            "expected DuplicateFamilies, got {error}"
        );
    }

    #[test]
    fn empty_inputs_and_zero_ticks_are_rejected() {
        assert!(matches!(
            plan(&spec(Vec::new(), vec![1], 10)),
            Err(TournamentError::EmptyFamilies)
        ));
        assert!(matches!(
            plan(&spec(vec![MLP], Vec::new(), 10)),
            Err(TournamentError::EmptySeeds)
        ));
        let mut input = spec(vec![MLP], vec![1], 10);
        input.ticks = 0;
        assert!(matches!(plan(&input), Err(TournamentError::ZeroTicks)));
    }

    #[test]
    fn all_permutations_enumerates_factorial_orders() {
        let mut input = spec(vec![MLP, DWRAON, ASSEMBLY], vec![3], 30);
        input.order_policy = OrderPolicy::AllPermutations;
        let planned = plan(&input).expect("permutations plan");
        assert_eq!(planned.len(), 6, "3! = 6 orders for one seed");
        let first_positions: Vec<BrainKind> =
            planned.iter().map(|plan| plan.spawn_order[0]).collect();
        for family in [MLP, DWRAON, ASSEMBLY] {
            assert_eq!(
                first_positions
                    .iter()
                    .filter(|first| **first == family)
                    .count(),
                2,
                "each family leads exactly twice"
            );
        }
    }
}

/// Match execution and outcome computation.
pub mod execution {
    use super::{FamilyOutcome, MatchOutcome, MatchPlan, TournamentError};
    use scriptbots_brain::BrainKind;
    use scriptbots_core::{AgentData, ScriptBotsConfig, WorldState};
    use std::collections::BTreeMap;

    /// A completed match: the outcome plus the config digest it ran under, so every arm
    /// can be checked against the plan's digest before anyone reads a leaderboard.
    #[derive(Debug, Clone)]
    pub struct MatchRunReport {
        pub outcome: MatchOutcome,
        pub config_digest: String,
    }

    /// Register exactly the canonical adapters the spec's families need, deduplicated:
    /// two arm names resolving to one adapter produce ONE registry entry, because the
    /// registry refuses duplicate adapter identities by design. Arm attribution for the
    /// null tournament flows through founder slots and lineage, never through a second
    /// registration that would collide with that rule.
    fn register_entered_families(
        world: &mut WorldState,
        families: &[BrainKind],
    ) -> Result<BTreeMap<BrainKind, u64>, TournamentError> {
        let mut keys = BTreeMap::new();
        let mut registered: BTreeMap<String, u64> = BTreeMap::new();
        for family in families {
            let name = family.as_str();
            let canonical = canonical_kind_of(name);
            let key = match registered.get(canonical.as_str()) {
                Some(key) => *key,
                None => {
                    let adapter = adapter_for(name)?;
                    let key = world
                        .register_brain_family(canonical.clone(), adapter)
                        .map_err(|error| TournamentError::UnbalancedOrders {
                            reason: format!(
                                "registering adapter {canonical} for {name:?} failed: {error}"
                            ),
                        })?;
                    registered.insert(canonical.clone(), key);
                    key
                }
            };
            keys.insert(*family, key);
        }
        Ok(keys)
    }

    /// Resolve an entered family name to its canonical adapter constructor. Exact
    /// built-in names, or any name with the family's dotted/dashed prefix ("mlp-a",
    /// "mlp.baseline", ...) resolve to the same adapter.
    fn adapter_for(
        name: &str,
    ) -> Result<Box<dyn scriptbots_core::BrainFamilyAdapter>, TournamentError> {
        if name == "mlp" || name.starts_with("mlp.") || name.starts_with("mlp-") {
            return Ok(Box::new(scriptbots_brain::mlp::MlpBrainFamily::new()));
        }
        if name == "dwraon" || name.starts_with("dwraon.") || name.starts_with("dwraon-") {
            return Ok(Box::new(
                scriptbots_brain::dwraon::DwraonFamilyAdapter::default(),
            ));
        }
        if name == "assembly" || name.starts_with("assembly.") || name.starts_with("assembly-") {
            let adapter =
                scriptbots_brain::assembly::AssemblyFamilyAdapter::new().map_err(|error| {
                    TournamentError::UnbalancedOrders {
                        reason: format!("assembly adapter construction: {error}"),
                    }
                })?;
            return Ok(Box::new(adapter));
        }
        Err(TournamentError::UnbalancedOrders {
            reason: format!(
                "no built-in adapter for entered family {name:?}; enter one of mlp.baseline, dwraon.baseline, assembly.experimental"
            ),
        })
    }

    /// The canonical registry kind an entered family name resolves to.
    fn canonical_kind_of(name: &str) -> String {
        if name == "mlp" || name.starts_with("mlp.") || name.starts_with("mlp-") {
            scriptbots_brain::mlp::MlpBrain::KIND.as_str().to_owned()
        } else if name == "dwraon" || name.starts_with("dwraon.") || name.starts_with("dwraon-") {
            scriptbots_brain::dwraon::DwraonBrain::KIND
                .as_str()
                .to_owned()
        } else if name == "assembly"
            || name.starts_with("assembly.")
            || name.starts_with("assembly-")
        {
            scriptbots_brain::assembly::AssemblyBrain::KIND
                .as_str()
                .to_owned()
        } else {
            name.to_owned()
        }
    }

    /// Deterministic cohort placement: spawn-order slot index drives the grid, so the
    /// order effect is a measurement, not a nuisance. No RNG is touched.
    fn cohort_grid_positions(
        cohort_total: usize,
        world_width: f32,
        world_height: f32,
    ) -> Vec<(f32, f32)> {
        let cols = (cohort_total as f32).sqrt().ceil().max(1.0) as usize;
        let rows = cohort_total.div_ceil(cols);
        let spacing_x = world_width / (cols as f32 + 1.0);
        let spacing_y = world_height / (rows as f32 + 1.0);
        (0..cohort_total)
            .map(|slot| {
                let col = slot % cols;
                let row = slot / cols;
                (
                    spacing_x * (col as f32 + 1.0),
                    spacing_y * (row as f32 + 1.0),
                )
            })
            .collect()
    }

    /// Inherit arms into newly arrived agents while their parents are still alive. The
    /// reproduction stage runs after death cleanup, so a newborn's parent is always
    /// alive at step end: every child is registered while its parent's arm is known,
    /// which keeps the map complete even after founders die.
    fn register_offspring_arms(
        world: &WorldState,
        arm_by_uid: &mut std::collections::HashMap<scriptbots_core::AgentUid, BrainKind>,
    ) {
        for id in world.agents().iter_handles() {
            let Some(uid) = world.agent_uid(id) else {
                continue;
            };
            if arm_by_uid.contains_key(&uid) {
                continue;
            }
            let Some(runtime) = world.agent_runtime(id) else {
                continue;
            };
            let parent_uid = runtime.lineage[0].or(runtime.lineage[1]);
            let Some(arm) = parent_uid.and_then(|parent| arm_by_uid.get(&parent)) else {
                continue;
            };
            arm_by_uid.insert(uid, *arm);
        }
    }

    /// Live agents per arm from the founder/lineage map.
    fn arm_counts(
        world: &WorldState,
        arm_by_uid: &std::collections::HashMap<scriptbots_core::AgentUid, BrainKind>,
    ) -> BTreeMap<BrainKind, usize> {
        let mut counts = BTreeMap::new();
        for id in world.agents().iter_handles() {
            let Some(uid) = world.agent_uid(id) else {
                continue;
            };
            if let Some(family) = arm_by_uid.get(&uid) {
                *counts.entry(*family).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Execute one match headlessly and compute its outcome. The match world is
    /// single-seeded (`plan.world_seed`), closed unless the caller says otherwise, and
    /// independent of scheduling: `--jobs N` cannot change any number because every
    /// match owns its own world and seed.
    pub fn run_match(
        plan: &MatchPlan,
        ticks: u64,
        closed: bool,
        base_config: &ScriptBotsConfig,
    ) -> Result<MatchRunReport, TournamentError> {
        let mut config = base_config.clone();
        config.rng_seed = Some(plan.world_seed);
        config.closed = closed;
        // The open-world lifeline is deliberately NOT touched: a tournament that sets
        // closed=false must experience the respawner and carry the qualifier, or the
        // warning would certify a respawn-inflated result as clean.
        // The config digest arms the cross-arm drift guard: it covers the SHARED
        // configuration (seed-neutralized — each match's world_seed is per-match by
        // design, not drift), so any mutated knob anywhere looks like the bug it is.
        let mut digest_config = base_config.clone();
        digest_config.rng_seed = None;
        digest_config.closed = closed;
        let config_digest = blake3::hash(
            serde_json::to_string(&digest_config)
                .map_err(|error| TournamentError::UnbalancedOrders {
                    reason: format!("config serialization for the digest failed: {error}"),
                })?
                .as_bytes(),
        )
        .to_hex()
        .to_string();
        let mut world =
            WorldState::new(config).map_err(|error| TournamentError::UnbalancedOrders {
                reason: format!("match world construction failed: {error}"),
            })?;

        let family_keys = register_entered_families(&mut world, &plan.spawn_order)?;

        // Spawn the cohort in spawn order on a deterministic grid, recording each
        // founder's arm by stable uid; every later agent inherits its arm through its
        // primary parent's lineage link, so arm attribution never depends on brain
        // bindings (which the null tournament deliberately shares).
        let cohort_total: usize = plan.cohort.values().sum();
        let positions = cohort_grid_positions(
            cohort_total,
            world.config().world_width as f32,
            world.config().world_height as f32,
        );
        let mut slot = 0_usize;
        let mut arm_by_uid: std::collections::HashMap<scriptbots_core::AgentUid, BrainKind> =
            std::collections::HashMap::new();
        for family in &plan.spawn_order {
            let key = family_keys[family];
            let members = plan.cohort[family];
            for _ in 0..members {
                let (x, y) = positions[slot];
                slot += 1;
                let id = world
                    .try_spawn_agent(AgentData {
                        position: scriptbots_core::Position::new(x, y),
                        ..AgentData::default()
                    })
                    .map_err(|error| TournamentError::UnbalancedOrders {
                        reason: format!("cohort spawn failed at slot {slot}: {error}"),
                    })?;
                if !world.bind_agent_brain(id, key).map_err(|error| {
                    TournamentError::UnbalancedOrders {
                        reason: format!("cohort brain bind failed at slot {slot}: {error}"),
                    }
                })? {
                    return Err(TournamentError::UnbalancedOrders {
                        reason: format!("brain bind returned false for family {}", family.as_str()),
                    });
                }
                let uid = world
                    .agent_uid(id)
                    .expect("a just-spawned agent has a stable uid");
                arm_by_uid.insert(uid, *family);
            }
        }

        let mut extinct_at: BTreeMap<BrainKind, u64> = BTreeMap::new();
        let mut ticks_run = 0_u64;
        for tick in 1..=ticks {
            world
                .step()
                .map_err(|error| TournamentError::UnbalancedOrders {
                    reason: format!("match step {tick} failed: {error}"),
                })?;
            ticks_run = tick;
            register_offspring_arms(&world, &mut arm_by_uid);
            let counts = arm_counts(&world, &arm_by_uid);
            for family in &plan.spawn_order {
                if counts.get(family).copied().unwrap_or(0) == 0 {
                    extinct_at.entry(*family).or_insert(tick);
                }
            }
        }

        // Outcome from the final state.
        let columns = world.agents().columns();
        let generations = columns.generations();
        let mut total_live = 0_usize;
        let mut total_energy = 0.0_f64;
        let mut family_live: BTreeMap<BrainKind, usize> = BTreeMap::new();
        let mut family_energy: BTreeMap<BrainKind, f64> = BTreeMap::new();
        let mut family_generations: BTreeMap<BrainKind, Vec<u32>> = BTreeMap::new();
        for id in world.agents().iter_handles() {
            let Some(idx) = world.agents().index_of(id) else {
                continue;
            };
            let Some(runtime) = world.agent_runtime(id) else {
                continue;
            };
            let Some(uid) = world.agent_uid(id) else {
                continue;
            };
            let Some(family) = arm_by_uid.get(&uid) else {
                continue;
            };
            total_live += 1;
            let energy = f64::from(runtime.energy.max(0.0));
            total_energy += energy;
            *family_live.entry(*family).or_insert(0) += 1;
            *family_energy.entry(*family).or_insert(0.0) += energy;
            family_generations
                .entry(*family)
                .or_default()
                .push(generations[idx].0);
        }

        let mut outcome = MatchOutcome {
            match_id: plan.match_id,
            ticks_run,
            spawn_order_index: plan.spawn_order_index,
            per_family: BTreeMap::new(),
            warnings: Vec::new(),
        };
        for family in &plan.spawn_order {
            let live = family_live.get(family).copied().unwrap_or(0);
            let energy = family_energy.get(family).copied().unwrap_or(0.0);
            let gens = family_generations.get(family);
            let (mean_depth, max_depth) = gens.map_or((0.0, 0), |values| {
                let mean = if values.is_empty() {
                    0.0
                } else {
                    values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64
                };
                let max = values.iter().copied().max().unwrap_or(0);
                (mean, max)
            });
            outcome.set_family(
                *family,
                FamilyOutcome {
                    survival_share: if total_live == 0 {
                        0.0
                    } else {
                        live as f64 / total_live as f64
                    },
                    biomass_share: if total_energy <= 0.0 {
                        0.0
                    } else {
                        energy / total_energy
                    },
                    mean_lineage_depth: mean_depth,
                    max_lineage_depth: max_depth,
                    extinct_at: extinct_at.get(family).copied(),
                    novelty_coverage: None,
                },
            );
        }
        if !closed {
            outcome.warnings.push(
                "open-world respawn active; survival share includes respawned agents".to_owned(),
            );
        }
        for family in &plan.spawn_order {
            if let Some(tick) = extinct_at.get(family)
                && *tick <= ticks / 10
            {
                outcome.warnings.push(format!(
                    "family {} extinct at tick {tick} (<=10% of budget; likely a spawn bug, not a finding)",
                    family.as_str()
                ));
            }
        }

        // Species barrier re-proven at the end of every match: no child anywhere has
        // parents of two different arms.
        assert_no_cross_kind_mating(&world, &arm_by_uid)?;

        for family in &plan.spawn_order {
            let family_outcome = outcome.family(*family).expect("family recorded");
            tracing::info!(
                target: "scriptbots::tournament",
                match_id = plan.match_id.0,
                seed = plan.seed,
                family = family.as_str(),
                spawn_order_index = plan.spawn_order_index,
                survival_share = family_outcome.survival_share,
                biomass_share = family_outcome.biomass_share,
                mean_lineage_depth = family_outcome.mean_lineage_depth,
                extinct_at = ?family_outcome.extinct_at,
                "match family outcome"
            );
            if let Some(tick) = family_outcome.extinct_at
                && tick <= ticks / 10
            {
                tracing::warn!(
                    target: "scriptbots::tournament",
                    match_id = plan.match_id.0,
                    family = family.as_str(),
                    extinct_at = tick,
                    "family extinct before 10% of the tick budget (likely a spawn bug, not a finding)"
                );
            }
        }

        Ok(MatchRunReport {
            outcome,
            config_digest,
        })
    }

    /// Assert the species barrier held: no child anywhere has parents of two different
    /// arms. A mixed cohort is the only place the codebase routinely mixes families, so
    /// the barrier is re-proven at the end of every match rather than assumed.
    fn assert_no_cross_kind_mating(
        world: &WorldState,
        arm_by_uid: &std::collections::HashMap<scriptbots_core::AgentUid, BrainKind>,
    ) -> Result<(), TournamentError> {
        for id in world.agents().iter_handles() {
            let Some(uid) = world.agent_uid(id) else {
                continue;
            };
            let Some(runtime) = world.agent_runtime(id) else {
                continue;
            };
            let (Some(parent_a), Some(parent_b)) = (runtime.lineage[0], runtime.lineage[1]) else {
                continue;
            };
            let (Some(arm_a), Some(arm_b)) = (arm_by_uid.get(&parent_a), arm_by_uid.get(&parent_b))
            else {
                continue;
            };
            if arm_a != arm_b {
                return Err(TournamentError::CrossKindMating {
                    child: uid.get(),
                    parent_a: arm_a.as_str().to_owned(),
                    parent_b: arm_b.as_str().to_owned(),
                });
            }
        }
        Ok(())
    }

    /// Plan and execute the full tournament serially. Matches are independent worlds by
    /// construction, so serial execution is byte-identical to any parallel schedule —
    /// the `--jobs N` invariant is proven per match rather than claimed for a pool.
    pub fn run_tournament(
        spec: &super::TournamentSpec,
        base_config: &ScriptBotsConfig,
    ) -> Result<Vec<MatchRunReport>, TournamentError> {
        let plans = super::plan(spec)?;
        let mut reports = Vec::with_capacity(plans.len());
        for match_plan in &plans {
            reports.push(run_match(match_plan, spec.ticks, spec.closed, base_config)?);
        }
        enforce_no_config_drift(&reports)?;
        Ok(reports)
    }

    /// Cross-arm drift guard: every match in a tournament must have run the same
    /// effective config, so a mutated knob anywhere surfaces as `ConfigDrift`, never as
    /// a publishable finding.
    pub fn enforce_no_config_drift(reports: &[MatchRunReport]) -> Result<(), TournamentError> {
        let Some(expected) = reports.first().map(|report| report.config_digest.clone()) else {
            return Ok(());
        };
        for report in reports {
            if report.config_digest != expected {
                return Err(TournamentError::ConfigDrift {
                    expected: expected.clone(),
                    found: report.config_digest.clone(),
                });
            }
        }
        Ok(())
    }
}
