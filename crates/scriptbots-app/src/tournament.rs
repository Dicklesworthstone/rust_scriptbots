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
                    reason: format!(
                        "BothAssignments requires exactly 2 families, got {count}"
                    ),
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
    if spec.cohort_size % family_count != 0 {
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

    let per_family = spec.cohort_size / family_count;
    let mut plans = Vec::with_capacity(total_matches);
    for (seed_ordinal, root_seed) in spec.seeds.iter().enumerate() {
        for (order_ordinal, spawn_order) in orders.iter().enumerate() {
            let match_index = (seed_ordinal * orders.len() + order_ordinal) as u64;
            plans.push(MatchPlan {
                match_id: match_id(*root_seed, match_index),
                seed: *root_seed,
                spawn_order: spawn_order.clone(),
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
                        count, 1,
                        "family {} must occupy position {position} exactly once per seed",
                        family.as_str()
                    );
                }
            }
            for plan in &planned {
                assert!(
                    plan.cohort
                        .values()
                        .all(|count| *count == 10),
                    "cohorts are equal"
                );
            }
        }
    }

    #[test]
    fn plan_is_pure_and_byte_identical_across_calls() {
        let input = spec(
            vec![MLP, DWRAON, ASSEMBLY],
            vec![7, 11, 13],
            30,
        );
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
            matches!(
                plan(&input),
                Err(TournamentError::UnbalancedOrders { .. })
            ),
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
        assert!(matches!(
            plan(&input),
            Err(TournamentError::ZeroTicks)
        ));
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
