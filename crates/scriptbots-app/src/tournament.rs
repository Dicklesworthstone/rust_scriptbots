//! Deterministic brain-family tournament harness, ratings, and leaderboard (bd-16g.12).
//!
//! This module owns the fairness core: a pure [`plan`] that turns a [`TournamentSpec`]
//! into match plans with equal cohorts, order-swap balance, and hash-allocated seeds.
//! Execution drives the existing headless simulation without downcasting or
//! special-casing a brain family.

use scriptbots_brain::BrainKind;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::PathBuf,
};

pub use execution::{
    MatchRunReport, enforce_no_config_drift, run_match, run_tournament, run_tournament_with_jobs,
};
pub use leaderboard::{
    ExcludedRunRecord, ExclusionReason, LeaderboardCheckError, LeaderboardPipelineOutcome,
    TournamentArgs, TournamentResultRow, TournamentSpecFile, check_leaderboard_drift,
    compute_unified_diff, execute_leaderboard_tournament, filter_eligible_rows,
    generate_leaderboard_document, reports_to_result_rows, run_tournament_command,
    save_tournament_artifacts, to_brain_kind,
};
pub use rating::{
    AxisRatingOutcome, AxisRatings, BtPrior, FamilyRating, OrderEffectReport, PairwiseComparison,
    PairwiseResult, PairwiseVerdict, RatingAxis, RatingError, RatingOptions, RatingTable,
    analyze_order_effect, elo_to_theta, fit_bradley_terry, rate_reports, rate_tournament,
    reduce_pairwise, theta_to_elo,
};

/// Largest match count one spec may emit (orders × seeds).
const MAX_MATCHES: usize = 4_096;

/// Prevent a large affordable plan from trying to create thousands of OS threads.
const MAX_PARALLEL_MATCH_WORKERS: usize = 64;

/// The complete, shared configuration for a tournament.
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
    /// Closed worlds disable the population lifeline that can resurrect extinct arms.
    pub closed: bool,
    /// Config files layered beneath every arm, in application order.
    pub config_layers: Vec<PathBuf>,
}

/// Spawn-order assignment generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderPolicy {
    /// Exactly two assignments; valid only for two families.
    BothAssignments,
    /// Cyclic rotations; each family occupies each position once per seed.
    BalancedLatinSquare,
    /// Every permutation, protected by [`MAX_MATCHES`].
    AllPermutations,
}

/// Stable match identity derived from the root seed and match index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MatchId(pub u64);

/// One deterministic, auditable match assignment.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchPlan {
    pub match_id: MatchId,
    /// Root spec seed this match derives from.
    pub seed: u64,
    /// Family spawn order.
    pub spawn_order: Vec<BrainKind>,
    /// Assignment index within the order set.
    pub spawn_order_index: u32,
    /// Equal cohort composition per family.
    pub cohort: BTreeMap<BrainKind, usize>,
    pub world_seed: u64,
    pub brain_seed: u64,
}

/// One family's result inside one match.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyOutcome {
    /// Share of the final live population belonging to this family.
    pub survival_share: f64,
    /// Share of final total energy belonging to this family.
    pub biomass_share: f64,
    pub mean_lineage_depth: f64,
    pub max_lineage_depth: u32,
    /// First tick on which the family had no live agents.
    pub extinct_at: Option<u64>,
    /// Reserved for the novelty archive.
    pub novelty_coverage: Option<f64>,
}

/// Typed per-match result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchOutcome {
    pub match_id: MatchId,
    pub seed: u64,
    pub ticks_run: u64,
    /// Assignment index retained for downstream order-effect analysis.
    pub spawn_order_index: u32,
    pub spawn_order: Vec<BrainKind>,
    pub per_family: BTreeMap<String, FamilyOutcome>,
    /// Qualifiers that downstream reports must retain.
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
        self.per_family.insert(kind.as_str().to_owned(), outcome);
    }

    /// Iterate families in deterministic name order.
    pub fn families(&self) -> impl Iterator<Item = (&str, &FamilyOutcome)> + '_ {
        self.per_family
            .iter()
            .map(|(name, outcome)| (name.as_str(), outcome))
    }
}

/// A tournament the harness refuses to defend.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TournamentError {
    #[error("no families entered")]
    EmptyFamilies,
    #[error("a tournament requires at least two families, got {families}")]
    TooFewFamilies { families: usize },
    #[error("no seeds supplied")]
    EmptySeeds,
    #[error("tick budget must be at least 1")]
    ZeroTicks,
    #[error("cohort size must be at least 1")]
    EmptyCohort,
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
    #[error("duplicate match id {match_id} in one tournament plan")]
    DuplicateMatchId { match_id: u64 },
    #[error("configuration layer {path} failed: {reason}")]
    ConfigLayer { path: PathBuf, reason: String },
    #[error("leaderboard check failed: {0}")]
    LeaderboardCheck(#[from] leaderboard::LeaderboardCheckError),
    #[error(
        "leaderboard reproducibility check failed: rerun results digest {rerun_digest} != {results_digest}"
    )]
    LeaderboardReproducibility {
        rerun_digest: String,
        results_digest: String,
    },
    #[error("{reason}")]
    UnbalancedOrders { reason: String },
}

/// SplitMix64 finalizer used as a pure seed hash.
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Lane-tagged index hash: lane 0 is world, 1 is brain, and 2 is match identity.
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
    const MATCH_ID_DOMAIN: u64 = 0x4D41_5443_485F_4944;
    MatchId(splitmix64(
        root_seed ^ splitmix64(match_index ^ MATCH_ID_DOMAIN),
    ))
}

fn checked_order_count(
    family_count: usize,
    seed_count: usize,
    policy: OrderPolicy,
) -> Result<usize, TournamentError> {
    let order_count = match policy {
        OrderPolicy::BothAssignments => {
            if family_count != 2 {
                return Err(TournamentError::UnbalancedOrders {
                    reason: format!(
                        "BothAssignments requires exactly 2 families, got {family_count}"
                    ),
                });
            }
            2
        }
        OrderPolicy::BalancedLatinSquare => family_count,
        OrderPolicy::AllPermutations => {
            let mut factorial = 1_usize;
            for factor in 2..=family_count {
                factorial = factorial.checked_mul(factor).ok_or_else(|| {
                    TournamentError::UnbalancedOrders {
                        reason: format!("{family_count}! overflows the match-count type"),
                    }
                })?;
                if factorial > MAX_MATCHES {
                    return Err(TournamentError::UnbalancedOrders {
                        reason: format!(
                            "{family_count}! order assignments exceeds the {MAX_MATCHES} affordability guard"
                        ),
                    });
                }
            }
            factorial
        }
    };
    let total_matches =
        order_count
            .checked_mul(seed_count)
            .ok_or_else(|| TournamentError::UnbalancedOrders {
                reason: "order count multiplied by seed count overflows".to_owned(),
            })?;
    if total_matches > MAX_MATCHES {
        return Err(TournamentError::UnbalancedOrders {
            reason: format!(
                "{total_matches} matches exceeds the {MAX_MATCHES} affordability guard"
            ),
        });
    }
    Ok(order_count)
}

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

/// Turn a specification into its complete deterministic match plan.
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
    if spec.families.len() < 2 {
        return Err(TournamentError::TooFewFamilies {
            families: spec.families.len(),
        });
    }
    if spec.cohort_size == 0 {
        return Err(TournamentError::EmptyCohort);
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
    let expected_order_count =
        checked_order_count(family_count, spec.seeds.len(), spec.order_policy)?;
    let orders = order_assignments(&spec.families, spec.order_policy)?;
    if orders.len() != expected_order_count {
        return Err(TournamentError::UnbalancedOrders {
            reason: format!(
                "order generator emitted {} assignments, expected {expected_order_count}",
                orders.len()
            ),
        });
    }
    let total_matches = expected_order_count * spec.seeds.len();

    let expected_families: BTreeSet<BrainKind> = spec.families.iter().copied().collect();
    for (index, order) in orders.iter().enumerate() {
        let actual: BTreeSet<BrainKind> = order.iter().copied().collect();
        if order.len() != family_count || actual != expected_families {
            return Err(TournamentError::UnbalancedOrders {
                reason: format!(
                    "order {index} is not a complete permutation of the entered families: {order:?}"
                ),
            });
        }
    }
    for position in 0..family_count {
        let mut counts: BTreeMap<BrainKind, usize> = spec
            .families
            .iter()
            .copied()
            .map(|family| (family, 0))
            .collect();
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
    let mut match_ids = BTreeSet::new();
    for (seed_ordinal, root_seed) in spec.seeds.iter().enumerate() {
        for (order_ordinal, spawn_order) in orders.iter().enumerate() {
            let index = seed_ordinal
                .checked_mul(orders.len())
                .and_then(|base| base.checked_add(order_ordinal))
                .ok_or_else(|| TournamentError::UnbalancedOrders {
                    reason: "match index arithmetic overflowed".to_owned(),
                })?;
            let match_index =
                u64::try_from(index).map_err(|_| TournamentError::UnbalancedOrders {
                    reason: format!("match index {index} does not fit u64"),
                })?;
            let spawn_order_index =
                u32::try_from(order_ordinal).map_err(|_| TournamentError::UnbalancedOrders {
                    reason: format!("order index {order_ordinal} does not fit u32"),
                })?;
            let match_id = match_id(*root_seed, match_index);
            if !match_ids.insert(match_id) {
                return Err(TournamentError::DuplicateMatchId {
                    match_id: match_id.0,
                });
            }
            let plan = MatchPlan {
                match_id,
                seed: *root_seed,
                spawn_order: spawn_order.clone(),
                spawn_order_index,
                cohort: spec
                    .families
                    .iter()
                    .map(|family| (*family, per_family))
                    .collect(),
                world_seed: world_seed(*root_seed, match_index),
                brain_seed: brain_seed(*root_seed, match_index),
            };
            tracing::debug!(
                target: "scriptbots::tournament",
                match_id = plan.match_id.0,
                root_seed = plan.seed,
                world_seed = plan.world_seed,
                brain_seed = plan.brain_seed,
                spawn_order_index = plan.spawn_order_index,
                spawn_order = ?plan.spawn_order,
                "tournament seed allocation"
            );
            plans.push(plan);
        }
    }
    Ok(plans)
}

#[cfg(test)]
mod planning_tests {
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
    fn plan_is_pure_and_match_ids_are_unique() {
        let input = spec(vec![MLP, DWRAON, ASSEMBLY], vec![7, 11, 13], 30);
        let first = plan(&input).expect("plan one");
        let second = plan(&input).expect("plan two");
        assert_eq!(first, second, "plan() is pure");
        assert_eq!(first.len(), 9, "3 seeds x 3 rotations");
        let ids: Vec<MatchId> = first.iter().map(|plan| plan.match_id).collect();
        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(ids.len(), unique.len(), "match ids are unique");
    }

    #[test]
    fn seed_allocation_matches_the_golden_table() {
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
        assert_eq!(plan(&input).expect("two families balance").len(), 2);
        input.families.push(ASSEMBLY);
        input.cohort_size = 30;
        assert!(matches!(
            plan(&input),
            Err(TournamentError::UnbalancedOrders { .. })
        ));
    }

    #[test]
    fn unequal_cohorts_are_rejected_with_typed_error() {
        let error = plan(&spec(vec![MLP, DWRAON, ASSEMBLY], vec![1], 10))
            .expect_err("10 is not divisible by 3");
        assert!(matches!(
            error,
            TournamentError::UnequalCohorts {
                cohort_size: 10,
                families: 3
            }
        ));
    }

    #[test]
    fn duplicate_families_are_rejected() {
        assert!(matches!(
            plan(&spec(vec![MLP, DWRAON, MLP], vec![1], 30)),
            Err(TournamentError::DuplicateFamilies { .. })
        ));
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
    fn single_arm_and_empty_cohorts_are_rejected() {
        assert!(matches!(
            plan(&spec(vec![MLP], vec![1], 10)),
            Err(TournamentError::TooFewFamilies { families: 1 })
        ));
        assert!(matches!(
            plan(&spec(vec![MLP, DWRAON], vec![1], 0)),
            Err(TournamentError::EmptyCohort)
        ));
    }

    #[test]
    fn all_permutations_enumerates_factorial_orders() {
        let mut input = spec(vec![MLP, DWRAON, ASSEMBLY], vec![3], 30);
        input.order_policy = OrderPolicy::AllPermutations;
        let planned = plan(&input).expect("permutations plan");
        assert_eq!(planned.len(), 6, "3! = 6 orders for one seed");
        for family in [MLP, DWRAON, ASSEMBLY] {
            assert_eq!(
                planned
                    .iter()
                    .filter(|plan| plan.spawn_order[0] == family)
                    .count(),
                2,
                "each family leads exactly twice"
            );
        }
    }

    #[test]
    fn factorial_policy_refuses_before_materializing_an_unaffordable_plan() {
        let families = vec![
            BrainKind::new("f0"),
            BrainKind::new("f1"),
            BrainKind::new("f2"),
            BrainKind::new("f3"),
            BrainKind::new("f4"),
            BrainKind::new("f5"),
            BrainKind::new("f6"),
            BrainKind::new("f7"),
            BrainKind::new("f8"),
        ];
        let mut input = spec(families, vec![1], 90);
        input.order_policy = OrderPolicy::AllPermutations;
        let error = plan(&input).expect_err("9! exceeds the affordability guard");
        assert!(matches!(error, TournamentError::UnbalancedOrders { .. }));
        assert!(error.to_string().contains("affordability guard"));
    }

    #[test]
    fn match_identity_has_its_own_hash_domain() {
        assert_ne!(
            match_id(42, 0).0,
            world_seed(42, 1),
            "match ids must not alias the next match's world-seed lane"
        );
        let planned =
            plan(&spec(vec![MLP, DWRAON], vec![42, 42], 20)).expect("duplicate root seeds plan");
        let unique: BTreeSet<MatchId> = planned.iter().map(|entry| entry.match_id).collect();
        assert_eq!(unique.len(), planned.len());
    }

    #[test]
    fn declared_config_layers_are_consumed_fail_closed() {
        let mut input = spec(vec![MLP, DWRAON], vec![42], 20);
        input.config_layers = vec![PathBuf::from(
            "__scriptbots_missing_tournament_layer__.toml",
        )];
        let error = run_tournament(&input, &scriptbots_core::ScriptBotsConfig::default())
            .expect_err("a missing declared layer must not be ignored");
        assert!(matches!(error, TournamentError::ConfigLayer { .. }));
        assert!(
            error
                .to_string()
                .contains("__scriptbots_missing_tournament_layer__.toml")
        );
    }

    #[test]
    fn run_match_rejects_unequal_public_cohorts() {
        let input = spec(vec![MLP, DWRAON], vec![42], 20);
        let mut plans = plan(&input).expect("balanced spec plans");
        let mut match_plan = plans.remove(0);
        match_plan.cohort.insert(MLP, 1);

        let error = run_match(
            &match_plan,
            input.ticks,
            input.closed,
            &scriptbots_core::ScriptBotsConfig::default(),
        )
        .expect_err("a caller-constructed unequal match must fail before execution");
        assert!(matches!(error, TournamentError::UnbalancedOrders { .. }));
        assert!(error.to_string().contains("unequal"));
    }
}

/// Match execution and outcome computation.
pub mod execution {
    use super::{
        FamilyOutcome, MAX_PARALLEL_MATCH_WORKERS, MatchOutcome, MatchPlan, TournamentError,
    };
    use crate::precedence::{ConfigLayerKind, ConfigLayerStatement, resolve_config_layers};
    use scriptbots_brain::BrainKind;
    use scriptbots_core::{AgentData, ScriptBotsConfig, WorldState};
    use std::{
        collections::{BTreeMap, HashMap},
        fs,
        path::Path,
    };

    /// A completed match plus the effective shared-config digest it ran under.
    #[derive(Debug, Clone)]
    pub struct MatchRunReport {
        pub outcome: MatchOutcome,
        pub config_digest: String,
        pub plan: super::MatchPlan,
        pub agents_final: BTreeMap<BrainKind, usize>,
    }

    fn config_layer_error(path: &Path, reason: impl std::fmt::Display) -> TournamentError {
        TournamentError::ConfigLayer {
            path: path.to_path_buf(),
            reason: reason.to_string(),
        }
    }

    fn resolve_spec_config(
        spec: &super::TournamentSpec,
        base_config: &ScriptBotsConfig,
    ) -> Result<ScriptBotsConfig, TournamentError> {
        if spec.config_layers.is_empty() {
            return Ok(base_config.clone());
        }
        let defaults = serde_json::to_value(base_config).map_err(|error| {
            TournamentError::UnbalancedOrders {
                reason: format!("serializing the base tournament config failed: {error}"),
            }
        })?;
        let mut statements = Vec::with_capacity(spec.config_layers.len());
        for path in &spec.config_layers {
            let source = fs::read_to_string(path)
                .map_err(|error| config_layer_error(path, format!("read: {error}")))?;
            let fields: serde_json::Value = match path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("ron") => ron::from_str(&source)
                    .map_err(|error| config_layer_error(path, format!("RON parse: {error}")))?,
                _ => toml::from_str(&source)
                    .map_err(|error| config_layer_error(path, format!("TOML parse: {error}")))?,
            };
            statements.push(ConfigLayerStatement {
                kind: ConfigLayerKind::File,
                label: format!("file:{}", path.display()),
                fields,
            });
        }
        let resolved = resolve_config_layers(&defaults, &statements);
        let final_layer = spec.config_layers.last().cloned().ok_or_else(|| {
            TournamentError::UnbalancedOrders {
                reason: "config layers became empty during resolution".to_owned(),
            }
        })?;
        serde_json::from_value(resolved.merged).map_err(|error| TournamentError::ConfigLayer {
            path: final_layer,
            reason: format!("merged config decode: {error}"),
        })
    }

    /// Register the canonical adapters needed by the entered arm names.
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
                    registered.insert(canonical, key);
                    key
                }
            };
            keys.insert(*family, key);
        }
        Ok(keys)
    }

    /// Resolve an entered arm name to a built-in family adapter.
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
                "no built-in adapter for entered family {name:?}; enter an mlp, dwraon, or assembly family"
            ),
        })
    }

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

    /// Deterministic cohort placement; no RNG is touched.
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

    /// Attach newborns to their founder arm through stable lineage identity.
    fn register_offspring_arms(
        world: &WorldState,
        arm_by_uid: &mut HashMap<scriptbots_core::AgentUid, BrainKind>,
        arm_by_registry_key: &BTreeMap<u64, Option<BrainKind>>,
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
            let inherited_arm = runtime
                .lineage
                .iter()
                .flatten()
                .find_map(|parent| arm_by_uid.get(parent))
                .copied();
            let registered_arm = runtime
                .brain
                .registry_key()
                .and_then(|key| arm_by_registry_key.get(&key))
                .copied()
                .flatten();
            let Some(arm) = inherited_arm.or(registered_arm) else {
                continue;
            };
            arm_by_uid.insert(uid, arm);
        }
    }

    fn arm_counts(
        world: &WorldState,
        arm_by_uid: &HashMap<scriptbots_core::AgentUid, BrainKind>,
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

    /// Execute one match in an independently seeded headless world.
    pub fn run_match(
        plan: &MatchPlan,
        ticks: u64,
        closed: bool,
        base_config: &ScriptBotsConfig,
    ) -> Result<MatchRunReport, TournamentError> {
        if ticks == 0 {
            return Err(TournamentError::ZeroTicks);
        }
        let order_families: std::collections::BTreeSet<BrainKind> =
            plan.spawn_order.iter().copied().collect();
        let cohort_families: std::collections::BTreeSet<BrainKind> =
            plan.cohort.keys().copied().collect();
        let expected_cohort_members = plan.cohort.values().next().copied().unwrap_or(0);
        if plan.spawn_order.is_empty()
            || order_families.len() != plan.spawn_order.len()
            || order_families != cohort_families
            || plan.cohort.values().any(|members| *members == 0)
            || plan
                .cohort
                .values()
                .any(|members| *members != expected_cohort_members)
        {
            return Err(TournamentError::UnbalancedOrders {
                reason: format!(
                    "match {} has an empty, duplicate, zero-sized, unequal, or cohort-mismatched spawn order",
                    plan.match_id.0
                ),
            });
        }
        let mut config = base_config.clone();
        config.rng_seed = Some(plan.world_seed);
        config.closed = closed;

        let mut digest_config = base_config.clone();
        digest_config.rng_seed = None;
        digest_config.closed = closed;
        let config_digest = blake3::hash(
            serde_json::to_string(&(digest_config, ticks))
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
        let mut arm_by_registry_key: BTreeMap<u64, Option<BrainKind>> = BTreeMap::new();
        for (family, key) in &family_keys {
            arm_by_registry_key
                .entry(*key)
                .and_modify(|arm| *arm = None)
                .or_insert(Some(*family));
        }

        let cohort_total = plan
            .cohort
            .values()
            .try_fold(0_usize, |total, members| total.checked_add(*members))
            .ok_or_else(|| TournamentError::UnbalancedOrders {
                reason: format!("match {} cohort size overflowed", plan.match_id.0),
            })?;
        let positions = cohort_grid_positions(
            cohort_total,
            world.config().world_width as f32,
            world.config().world_height as f32,
        );
        let mut slot = 0_usize;
        let mut arm_by_uid: HashMap<scriptbots_core::AgentUid, BrainKind> = HashMap::new();
        for family in &plan.spawn_order {
            let key = family_keys.get(family).copied().ok_or_else(|| {
                TournamentError::UnbalancedOrders {
                    reason: format!("family {} has no registered key", family.as_str()),
                }
            })?;
            let members = plan.cohort.get(family).copied().ok_or_else(|| {
                TournamentError::UnbalancedOrders {
                    reason: format!("family {} has no cohort entry", family.as_str()),
                }
            })?;
            for _ in 0..members {
                let (x, y) = positions.get(slot).copied().ok_or_else(|| {
                    TournamentError::UnbalancedOrders {
                        reason: format!("cohort position {slot} is unavailable"),
                    }
                })?;
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
                    .ok_or_else(|| TournamentError::UnbalancedOrders {
                        reason: format!("cohort agent at slot {slot} has no stable uid"),
                    })?;
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
            // Inspect each live birth before lineage attribution. A mixed-kind child
            // that disappears before match completion must not evade the barrier proof.
            assert_no_cross_kind_mating(&world, &arm_by_uid)?;
            register_offspring_arms(&world, &mut arm_by_uid, &arm_by_registry_key);
            let counts = arm_counts(&world, &arm_by_uid);
            for family in &plan.spawn_order {
                if counts.get(family).copied().unwrap_or(0) == 0 {
                    extinct_at.entry(*family).or_insert(tick);
                }
            }
        }

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
            total_live += 1;
            total_energy += f64::from(runtime.energy.max(0.0));
            let Some(uid) = world.agent_uid(id) else {
                continue;
            };
            let Some(family) = arm_by_uid.get(&uid) else {
                continue;
            };
            let energy = f64::from(runtime.energy.max(0.0));
            *family_live.entry(*family).or_insert(0) += 1;
            *family_energy.entry(*family).or_insert(0.0) += energy;
            let generation =
                generations
                    .get(idx)
                    .ok_or_else(|| TournamentError::UnbalancedOrders {
                        reason: format!("agent column index {idx} has no generation"),
                    })?;
            family_generations
                .entry(*family)
                .or_default()
                .push(generation.0);
        }

        let mut outcome = MatchOutcome {
            match_id: plan.match_id,
            seed: plan.seed,
            ticks_run,
            spawn_order_index: plan.spawn_order_index,
            spawn_order: plan.spawn_order.clone(),
            per_family: BTreeMap::new(),
            warnings: Vec::new(),
        };
        for family in &plan.spawn_order {
            let live = family_live.get(family).copied().unwrap_or(0);
            let energy = family_energy.get(family).copied().unwrap_or(0.0);
            let generations = family_generations.get(family);
            let (mean_lineage_depth, max_lineage_depth) = generations.map_or((0.0, 0), |values| {
                let mean = if values.is_empty() {
                    0.0
                } else {
                    values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64
                };
                (mean, values.iter().copied().max().unwrap_or(0))
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
                    mean_lineage_depth,
                    max_lineage_depth,
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

        assert_no_cross_kind_mating(&world, &arm_by_uid)?;

        for family in &plan.spawn_order {
            let family_outcome =
                outcome
                    .family(*family)
                    .ok_or_else(|| TournamentError::UnbalancedOrders {
                        reason: format!("family {} has no outcome row", family.as_str()),
                    })?;
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
                agents_final = family_live.get(family).copied().unwrap_or(0),
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
                    "family extinct before 10% of the tick budget"
                );
            }
        }

        Ok(MatchRunReport {
            outcome,
            config_digest,
            plan: plan.clone(),
            agents_final: family_live,
        })
    }

    fn assert_no_cross_kind_mating(
        world: &WorldState,
        arm_by_uid: &HashMap<scriptbots_core::AgentUid, BrainKind>,
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
            let kind_a = canonical_kind_of(arm_a.as_str());
            let kind_b = canonical_kind_of(arm_b.as_str());
            if kind_a != kind_b {
                return Err(TournamentError::CrossKindMating {
                    child: uid.get(),
                    parent_a: kind_a,
                    parent_b: kind_b,
                });
            }
        }
        Ok(())
    }

    /// Plan and execute every match serially.
    pub fn run_tournament(
        spec: &super::TournamentSpec,
        base_config: &ScriptBotsConfig,
    ) -> Result<Vec<MatchRunReport>, TournamentError> {
        run_tournament_with_jobs(spec, base_config, 1)
    }

    /// Plan and execute matches with a bounded number of independent workers.
    ///
    /// Results are collected in plan order, never completion order, so changing `jobs`
    /// cannot change the report sequence or any match input.
    pub fn run_tournament_with_jobs(
        spec: &super::TournamentSpec,
        base_config: &ScriptBotsConfig,
        jobs: usize,
    ) -> Result<Vec<MatchRunReport>, TournamentError> {
        let plans = super::plan(spec)?;
        let effective_config = resolve_spec_config(spec, base_config)?;
        let worker_count = jobs.max(1).min(plans.len()).min(MAX_PARALLEL_MATCH_WORKERS);
        let reports = if worker_count == 1 {
            plans
                .iter()
                .map(|match_plan| run_match(match_plan, spec.ticks, spec.closed, &effective_config))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let chunk_size = plans.len().div_ceil(worker_count);
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(worker_count);
                for (worker_index, chunk) in plans.chunks(chunk_size).enumerate() {
                    let config = &effective_config;
                    let handle = std::thread::Builder::new()
                        .name(format!("scriptbots-tournament-{worker_index}"))
                        .spawn_scoped(scope, move || {
                            chunk
                                .iter()
                                .map(|match_plan| {
                                    run_match(match_plan, spec.ticks, spec.closed, config)
                                })
                                .collect::<Result<Vec<_>, TournamentError>>()
                        })
                        .map_err(|error| TournamentError::UnbalancedOrders {
                            reason: format!(
                                "spawning tournament worker {worker_index} failed: {error}"
                            ),
                        })?;
                    handles.push(handle);
                }

                let mut ordered = Vec::with_capacity(plans.len());
                for handle in handles {
                    let chunk =
                        handle
                            .join()
                            .map_err(|_| TournamentError::UnbalancedOrders {
                                reason: "tournament worker panicked".to_owned(),
                            })??;
                    ordered.extend(chunk);
                }
                Ok::<Vec<MatchRunReport>, TournamentError>(ordered)
            })?
        };
        enforce_no_config_drift(&reports)?;
        if let Some(report) = reports.first() {
            tracing::info!(
                target: "scriptbots::tournament",
                config_digest = report.config_digest,
                matches = reports.len(),
                jobs = worker_count,
                "tournament effective config verified across all arms"
            );
        }
        Ok(reports)
    }

    /// Reject any cross-arm effective-config drift.
    pub fn enforce_no_config_drift(reports: &[MatchRunReport]) -> Result<(), TournamentError> {
        let Some(expected) = reports.first().map(|report| report.config_digest.clone()) else {
            return Ok(());
        };
        for report in reports {
            if report.config_digest != expected {
                tracing::error!(
                    target: "scriptbots::tournament",
                    expected,
                    found = report.config_digest,
                    "config digest mismatch across tournament arms"
                );
                return Err(TournamentError::ConfigDrift {
                    expected: expected.clone(),
                    found: report.config_digest.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Multi-axis Bradley-Terry / Zermelo rating, clustered bootstrap confidence intervals,
/// order-effect analysis, and parity testing (bd-16g.12.2).
pub mod rating {
    use super::{FamilyOutcome, MatchOutcome, MatchRunReport};
    use rand::{Rng, SeedableRng, rngs::SmallRng};
    use scriptbots_brain::BrainKind;
    use serde::{Deserialize, Serialize};
    use std::{
        collections::{BTreeMap, BTreeSet},
        fmt,
    };
    use thiserror::Error;

    pub const DEFAULT_PSEUDO_WINS: f64 = 0.5;
    pub const DEFAULT_BOOTSTRAP_REPLICATES: usize = 2000;
    pub const DEFAULT_CI_SEED: u64 = 42;
    pub const DEFAULT_CONVERGENCE_TOLERANCE: f64 = 1e-10;
    pub const DEFAULT_MAX_ITERATIONS: u32 = 10_000;
    pub const DEFAULT_MIN_SEEDS: usize = 8;
    pub const EPSILON: f64 = 1e-9;

    /// Performance evaluation axes across tournament matches.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub enum RatingAxis {
        SurvivalShare,
        BiomassShare,
        MeanLineageDepth,
        TimeToExtinction,
        NoveltyCoverage,
        AggregateScore,
    }

    impl RatingAxis {
        #[must_use]
        pub const fn as_str(self) -> &'static str {
            match self {
                Self::SurvivalShare => "survival_share",
                Self::BiomassShare => "biomass_share",
                Self::MeanLineageDepth => "mean_lineage_depth",
                Self::TimeToExtinction => "time_to_extinction",
                Self::NoveltyCoverage => "novelty_coverage",
                Self::AggregateScore => "aggregate_score",
            }
        }

        pub const ALL: &[Self] = &[
            Self::SurvivalShare,
            Self::BiomassShare,
            Self::MeanLineageDepth,
            Self::TimeToExtinction,
            Self::NoveltyCoverage,
            Self::AggregateScore,
        ];
    }

    impl fmt::Display for RatingAxis {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.as_str())
        }
    }

    /// Regularizer prior for the Bradley-Terry maximum likelihood estimator.
    ///
    /// Without a prior, an arm that wins every match has a divergent MLE
    /// (theta -> +inf), resulting in non-convergence or overflow. Adding
    /// `pseudo_wins` to every directed comparison ensures the comparison graph
    /// is connected and every parameter estimate is finite and bounded.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct BtPrior {
        pub pseudo_wins: f64,
    }

    impl BtPrior {
        #[must_use]
        pub const fn new(pseudo_wins: f64) -> Self {
            Self { pseudo_wins }
        }

        #[must_use]
        pub const fn none() -> Self {
            Self { pseudo_wins: 0.0 }
        }
    }

    impl Default for BtPrior {
        fn default() -> Self {
            Self {
                pseudo_wins: DEFAULT_PSEUDO_WINS,
            }
        }
    }

    /// Tuning and statistical parameters for tournament rating.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct RatingOptions {
        pub prior: BtPrior,
        pub bootstrap_replicates: usize,
        pub ci_seed: u64,
        pub tolerance: f64,
        pub max_iterations: u32,
        pub min_seeds: usize,
    }

    impl Default for RatingOptions {
        fn default() -> Self {
            Self {
                prior: BtPrior::default(),
                bootstrap_replicates: DEFAULT_BOOTSTRAP_REPLICATES,
                ci_seed: DEFAULT_CI_SEED,
                tolerance: DEFAULT_CONVERGENCE_TOLERANCE,
                max_iterations: DEFAULT_MAX_ITERATIONS,
                min_seeds: DEFAULT_MIN_SEEDS,
            }
        }
    }

    /// Display transform from Bradley-Terry log-ability (theta) to Elo rating.
    ///
    /// elo = 400 * theta / ln(10) + 1500, anchored so mean theta = 0 maps to mean Elo = 1500.
    /// Note: Elo is a DISPLAY TRANSFORM only. Ratings are fitted globally using the
    /// Bradley-Terry MM/Zermelo estimator, NOT online with a sequential K-factor,
    /// because sequential Elo injects an arbitrary match-order dependency.
    #[must_use]
    pub fn theta_to_elo(theta: f64) -> f64 {
        400.0 * theta / std::f64::consts::LN_10 + 1500.0
    }

    /// Inverse transform from Elo rating to Bradley-Terry theta.
    #[must_use]
    pub fn elo_to_theta(elo: f64) -> f64 {
        (elo - 1500.0) * std::f64::consts::LN_10 / 400.0
    }

    /// Aggregated pairwise head-to-head record between two brain families on a specific axis.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PairwiseResult {
        pub a: BrainKind,
        pub b: BrainKind,
        pub axis: RatingAxis,
        pub wins_a: u32,
        pub wins_b: u32,
        pub draws: u32,
        pub seed_of: Vec<u64>,
    }

    impl PairwiseResult {
        #[must_use]
        pub fn total_matches(&self) -> u32 {
            self.wins_a + self.wins_b + self.draws
        }

        #[must_use]
        pub fn effective_wins_a(&self) -> f64 {
            f64::from(self.wins_a) + 0.5 * f64::from(self.draws)
        }

        #[must_use]
        pub fn effective_wins_b(&self) -> f64 {
            f64::from(self.wins_b) + 0.5 * f64::from(self.draws)
        }
    }

    /// Parity test decision for whether family A decisively beats family B.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    pub enum PairwiseVerdict {
        AWins,
        BWins,
        Parity,
    }

    impl PairwiseVerdict {
        #[must_use]
        pub const fn as_str(self) -> &'static str {
            match self {
                Self::AWins => "a_wins",
                Self::BWins => "b_wins",
                Self::Parity => "parity",
            }
        }
    }

    impl fmt::Display for PairwiseVerdict {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.as_str())
        }
    }

    /// Detailed pairwise comparison including bootstrap CI, verdict, and effect sizes.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PairwiseComparison {
        pub a: BrainKind,
        pub b: BrainKind,
        pub theta_diff: f64,
        pub ci95: (f64, f64),
        pub verdict: PairwiseVerdict,
        pub effect_size_win_rate_diff: f64,
        pub cliffs_delta: f64,
        pub wins_a: u32,
        pub wins_b: u32,
        pub draws: u32,
    }

    /// Fitted rating record for one brain family on one axis.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct FamilyRating {
        pub family: BrainKind,
        pub theta: f64,
        pub elo: f64,
        pub se: f64,
        pub ci95: (f64, f64),
        pub elo_ci95: (f64, f64),
        pub n_matches: usize,
        pub wins: u32,
        pub losses: u32,
        pub draws: u32,
    }

    /// Complete rating results for a single evaluation axis.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct AxisRatings {
        pub axis: RatingAxis,
        pub n_matches: usize,
        pub n_seeds: usize,
        pub iterations: u32,
        pub converged: bool,
        pub ci_seed: u64,
        pub bootstrap_replicates: usize,
        pub tolerance: f64,
        pub interval_method: String,
        pub ratings: BTreeMap<BrainKind, FamilyRating>,
        pub pairwise: Vec<PairwiseComparison>,
    }

    /// Evaluation outcome for an axis: either successfully rated or explicitly undetermined.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum AxisRatingOutcome {
        Rated(AxisRatings),
        Undetermined { reason: String },
    }

    /// Spawn-order effect diagnostic report.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct OrderEffectReport {
        pub position_means: BTreeMap<u32, f64>,
        pub position_ci95: BTreeMap<u32, (f64, f64)>,
        pub detected: bool,
    }

    /// Multi-axis tournament rating table.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct RatingTable {
        pub axes: BTreeMap<RatingAxis, AxisRatingOutcome>,
        pub order_effects: BTreeMap<RatingAxis, OrderEffectReport>,
        pub warnings: Vec<String>,
    }

    impl RatingTable {
        #[must_use]
        pub fn axis(&self, axis: RatingAxis) -> Option<&AxisRatings> {
            match self.axes.get(&axis) {
                Some(AxisRatingOutcome::Rated(ratings)) => Some(ratings),
                _ => None,
            }
        }

        #[must_use]
        pub fn generate_markdown(&self) -> String {
            let mut out =
                String::from("# ScriptBots Multi-Axis Brain Family Tournament Leaderboard\n\n");

            if !self.warnings.is_empty() {
                out.push_str("### Tournament Qualifiers & Warnings\n");
                for w in &self.warnings {
                    out.push_str(&format!("- ⚠️ {w}\n"));
                }
                out.push('\n');
            }

            for (axis, outcome) in &self.axes {
                out.push_str(&format!("## Axis: {} (`{}`)\n\n", axis, axis.as_str()));
                match outcome {
                    AxisRatingOutcome::Rated(ratings) => {
                        out.push_str(&format!(
                            "*Matches: {}, Seeds: {}, Iterations: {} ({}), Bootstrap Replicates: {}*\n\n",
                            ratings.n_matches,
                            ratings.n_seeds,
                            ratings.iterations,
                            if ratings.converged {
                                "converged"
                            } else {
                                "unconverged"
                            },
                            ratings.bootstrap_replicates,
                        ));

                        out.push_str(
                            "| Rank | Family | Elo Rating (95% CI) | Theta (95% CI) | SE | Matches | W / L / D |\n\
                             | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n",
                        );

                        let mut sorted_ratings: Vec<_> = ratings.ratings.values().collect();
                        sorted_ratings.sort_by(|a, b| {
                            b.elo
                                .total_cmp(&a.elo)
                                .then_with(|| a.family.as_str().cmp(b.family.as_str()))
                        });

                        for (rank, r) in sorted_ratings.iter().enumerate() {
                            let sign = if r.theta >= 0.0 { "+" } else { "" };
                            out.push_str(&format!(
                                "| {} | {} | {:.1} [{:.1}, {:.1}] | {}{:.3} [{:.3}, {:.3}] | {:.3} | {} | {} / {} / {} |\n",
                                rank + 1,
                                r.family.as_str(),
                                r.elo,
                                r.elo_ci95.0,
                                r.elo_ci95.1,
                                sign,
                                r.theta,
                                r.ci95.0,
                                r.ci95.1,
                                r.se,
                                r.n_matches,
                                r.wins,
                                r.losses,
                                r.draws,
                            ));
                        }
                        out.push('\n');

                        if !ratings.pairwise.is_empty() {
                            out.push_str(
                                "### Pairwise Head-to-Head Comparisons\n\n\
                                 | Family A | Family B | Verdict | Theta Diff (95% CI) | Win Rate Diff | Cliff's Delta | W / L / D |\n\
                                 | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n",
                            );
                            for pair in &ratings.pairwise {
                                let sign = if pair.theta_diff >= 0.0 { "+" } else { "" };
                                out.push_str(&format!(
                                    "| {} | {} | `{}` | {}{:.3} [{:.3}, {:.3}] | {:+.3} | {:+.3} | {} / {} / {} |\n",
                                    pair.a.as_str(),
                                    pair.b.as_str(),
                                    pair.verdict.as_str(),
                                    sign,
                                    pair.theta_diff,
                                    pair.ci95.0,
                                    pair.ci95.1,
                                    pair.effect_size_win_rate_diff,
                                    pair.cliffs_delta,
                                    pair.wins_a,
                                    pair.wins_b,
                                    pair.draws,
                                ));
                            }
                            out.push('\n');
                        }

                        if let Some(order_effect) = self.order_effects.get(axis) {
                            if order_effect.detected {
                                out.push_str("> ⚠️ **Spawn-Order Effect Detected**: Spawn order significantly affected performance on this axis.\n\n");
                            }
                            out.push_str("#### Spawn-Order Position Performance\n");
                            for (pos, mean) in &order_effect.position_means {
                                if let Some(ci) = order_effect.position_ci95.get(pos) {
                                    out.push_str(&format!(
                                        "- Position {}: mean = {:.3} (95% CI [{:.3}, {:.3}])\n",
                                        pos, mean, ci.0, ci.1
                                    ));
                                } else {
                                    out.push_str(&format!(
                                        "- Position {}: mean = {:.3}\n",
                                        pos, mean
                                    ));
                                }
                            }
                            out.push('\n');
                        }
                    }
                    AxisRatingOutcome::Undetermined { reason } => {
                        out.push_str(&format!("*Undetermined: {reason}*\n\n"));
                    }
                }
            }

            out
        }
    }

    /// Rating error conditions.
    #[derive(Debug, Clone, PartialEq, Error)]
    pub enum RatingError {
        #[error("no matches provided")]
        EmptyMatches,
        #[error("tournament requires at least 2 families, got {families}")]
        TooFewFamilies { families: usize },
        #[error("axis {axis} is degenerate: {reason}")]
        DegenerateAxis { axis: RatingAxis, reason: String },
        #[error("fitting failed: {reason}")]
        FittingFailure { reason: String },
        #[error("non-finite value encountered in {context}")]
        NonFiniteValue { context: String },
    }

    /// Fitted parameters from Bradley-Terry MM optimization.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct FitResult {
        pub theta: BTreeMap<BrainKind, f64>,
        pub elo: BTreeMap<BrainKind, f64>,
        pub iterations: u32,
        pub converged: bool,
        pub last_delta: f64,
    }

    fn extract_axis_value(
        outcome: &FamilyOutcome,
        axis: RatingAxis,
        ticks_run: u64,
        family: &str,
    ) -> Result<Option<f64>, RatingError> {
        match axis {
            RatingAxis::SurvivalShare => {
                if !outcome.survival_share.is_finite() {
                    return Err(RatingError::NonFiniteValue {
                        context: format!("survival_share for {family}"),
                    });
                }
                Ok(Some(outcome.survival_share))
            }
            RatingAxis::BiomassShare => {
                if !outcome.biomass_share.is_finite() {
                    return Err(RatingError::NonFiniteValue {
                        context: format!("biomass_share for {family}"),
                    });
                }
                Ok(Some(outcome.biomass_share))
            }
            RatingAxis::MeanLineageDepth => {
                if !outcome.mean_lineage_depth.is_finite() {
                    return Err(RatingError::NonFiniteValue {
                        context: format!("mean_lineage_depth for {family}"),
                    });
                }
                Ok(Some(outcome.mean_lineage_depth))
            }
            RatingAxis::TimeToExtinction => {
                let t = outcome.extinct_at.unwrap_or(ticks_run) as f64;
                Ok(Some(t))
            }
            RatingAxis::NoveltyCoverage => {
                if let Some(nc) = outcome.novelty_coverage {
                    if !nc.is_finite() {
                        return Err(RatingError::NonFiniteValue {
                            context: format!("novelty_coverage for {family}"),
                        });
                    }
                    Ok(Some(nc))
                } else {
                    Ok(None)
                }
            }
            RatingAxis::AggregateScore => {
                if !outcome.survival_share.is_finite()
                    || !outcome.biomass_share.is_finite()
                    || !outcome.mean_lineage_depth.is_finite()
                {
                    return Err(RatingError::NonFiniteValue {
                        context: format!("aggregate_score components for {family}"),
                    });
                }
                let survival = outcome.survival_share;
                let biomass = outcome.biomass_share;
                let lineage = outcome.mean_lineage_depth / (outcome.mean_lineage_depth + 10.0);
                let extinction = if let Some(t) = outcome.extinct_at {
                    if ticks_run > 0 {
                        (t as f64) / (ticks_run as f64)
                    } else {
                        0.0
                    }
                } else {
                    1.0
                };
                let score = 0.40 * survival + 0.30 * biomass + 0.20 * lineage + 0.10 * extinction;
                Ok(Some(score))
            }
        }
    }

    /// Fit a Bradley-Terry model using the Zermelo / MM iteration with prior regularizer.
    pub fn fit_bradley_terry(
        families: &[BrainKind],
        pairwise: &[PairwiseResult],
        prior: BtPrior,
        tolerance: f64,
        max_iterations: u32,
    ) -> Result<FitResult, RatingError> {
        if families.len() < 2 {
            return Err(RatingError::TooFewFamilies {
                families: families.len(),
            });
        }
        let k = families.len();
        let family_to_idx: BTreeMap<BrainKind, usize> = families
            .iter()
            .enumerate()
            .map(|(idx, kind)| (*kind, idx))
            .collect();

        let mut w = vec![vec![0.0_f64; k]; k];
        let mut n = vec![vec![0.0_f64; k]; k];

        for res in pairwise {
            let Some(&i) = family_to_idx.get(&res.a) else {
                continue;
            };
            let Some(&j) = family_to_idx.get(&res.b) else {
                continue;
            };
            if i == j {
                continue;
            }
            let eff_a = res.effective_wins_a();
            let eff_b = res.effective_wins_b();
            let total = eff_a + eff_b;
            w[i][j] += eff_a;
            w[j][i] += eff_b;
            n[i][j] += total;
            n[j][i] += total;
        }

        let pseudo = prior.pseudo_wins.max(0.0);
        let mut w_prime = vec![0.0_f64; k];
        let mut n_prime = vec![vec![0.0_f64; k]; k];

        for i in 0..k {
            for j in 0..k {
                if i != j {
                    let w_ij = w[i][j] + pseudo;
                    let n_ij = n[i][j] + 2.0 * pseudo;
                    w_prime[i] += w_ij;
                    n_prime[i][j] = n_ij;
                }
            }
        }

        let total_comparisons: f64 = n_prime.iter().map(|row| row.iter().sum::<f64>()).sum();
        if total_comparisons <= 0.0 {
            return Err(RatingError::FittingFailure {
                reason: "no comparisons recorded between families".to_string(),
            });
        }

        if pseudo == 0.0 {
            for i in 0..k {
                let losses: f64 = (0..k).filter(|&j| j != i).map(|j| w[j][i]).sum();
                if losses == 0.0 && w_prime[i] > 0.0 {
                    return Ok(FitResult {
                        theta: BTreeMap::new(),
                        elo: BTreeMap::new(),
                        iterations: max_iterations,
                        converged: false,
                        last_delta: f64::INFINITY,
                    });
                }
                if w_prime[i] == 0.0 {
                    return Ok(FitResult {
                        theta: BTreeMap::new(),
                        elo: BTreeMap::new(),
                        iterations: max_iterations,
                        converged: false,
                        last_delta: f64::INFINITY,
                    });
                }
            }
        }

        let mut p = vec![1.0_f64 / k as f64; k];
        let mut next_p = vec![0.0_f64; k];
        let mut converged = false;
        let mut last_delta = f64::INFINITY;
        let mut iterations_run = 0_u32;

        for iter in 1..=max_iterations {
            iterations_run = iter;
            let mut sum_next_p = 0.0_f64;

            for i in 0..k {
                let mut denom = 0.0_f64;
                for j in 0..k {
                    if i != j {
                        let sum_p = p[i] + p[j];
                        if sum_p > 1e-15 {
                            denom += n_prime[i][j] / sum_p;
                        }
                    }
                }
                if denom > 1e-15 && w_prime[i] > 0.0 {
                    next_p[i] = w_prime[i] / denom;
                } else {
                    next_p[i] = 0.0;
                }
                sum_next_p += next_p[i];
            }

            if sum_next_p <= 0.0 || !sum_next_p.is_finite() {
                return Ok(FitResult {
                    theta: BTreeMap::new(),
                    elo: BTreeMap::new(),
                    iterations: iter,
                    converged: false,
                    last_delta: f64::INFINITY,
                });
            }

            for val in &mut next_p {
                *val /= sum_next_p;
            }

            let mut max_delta = 0.0_f64;
            for i in 0..k {
                let diff = (next_p[i] - p[i]).abs();
                if diff > max_delta {
                    max_delta = diff;
                }
                p[i] = next_p[i];
            }
            last_delta = max_delta;

            if max_delta < tolerance {
                converged = true;
                break;
            }
        }

        if !converged {
            return Ok(FitResult {
                theta: BTreeMap::new(),
                elo: BTreeMap::new(),
                iterations: iterations_run,
                converged: false,
                last_delta,
            });
        }

        let mut log_p = vec![0.0_f64; k];
        for i in 0..k {
            if p[i] <= 0.0 || !p[i].is_finite() {
                return Ok(FitResult {
                    theta: BTreeMap::new(),
                    elo: BTreeMap::new(),
                    iterations: iterations_run,
                    converged: false,
                    last_delta,
                });
            }
            log_p[i] = p[i].ln();
        }
        let mean_log_p = log_p.iter().sum::<f64>() / k as f64;

        let mut theta = BTreeMap::new();
        let mut elo = BTreeMap::new();
        for (i, family) in families.iter().enumerate() {
            let anchored_theta = log_p[i] - mean_log_p;
            let elo_val = theta_to_elo(anchored_theta);
            theta.insert(*family, anchored_theta);
            elo.insert(*family, elo_val);
        }

        Ok(FitResult {
            theta,
            elo,
            iterations: iterations_run,
            converged: true,
            last_delta,
        })
    }

    #[derive(Default)]
    struct PairwiseCounts {
        wins_a: u32,
        wins_b: u32,
        draws: u32,
        seeds: Vec<u64>,
    }

    pub(crate) fn reduce_pairwise_for_matches(
        matches: &[&MatchOutcome],
        families: &[BrainKind],
        axis: RatingAxis,
    ) -> Result<Vec<PairwiseResult>, RatingError> {
        let mut res_map: BTreeMap<(BrainKind, BrainKind), PairwiseCounts> = BTreeMap::new();

        for i in 0..families.len() {
            for j in (i + 1)..families.len() {
                let a = families[i];
                let b = families[j];
                res_map.insert((a, b), PairwiseCounts::default());
            }
        }

        for m in matches {
            for i in 0..families.len() {
                for j in (i + 1)..families.len() {
                    let a = families[i];
                    let b = families[j];
                    let outcome_a = m.per_family.get(a.as_str());
                    let outcome_b = m.per_family.get(b.as_str());
                    let (Some(oa), Some(ob)) = (outcome_a, outcome_b) else {
                        continue;
                    };
                    let (Some(va), Some(vb)) = (
                        extract_axis_value(oa, axis, m.ticks_run, a.as_str())?,
                        extract_axis_value(ob, axis, m.ticks_run, b.as_str())?,
                    ) else {
                        continue;
                    };

                    let entry = res_map.entry((a, b)).or_default();
                    if (va - vb).abs() < EPSILON {
                        entry.draws += 1;
                    } else if va > vb {
                        entry.wins_a += 1;
                    } else {
                        entry.wins_b += 1;
                    }
                    entry.seeds.push(m.seed);
                }
            }
        }

        let mut results = Vec::with_capacity(res_map.len());
        for ((a, b), counts) in res_map {
            results.push(PairwiseResult {
                a,
                b,
                axis,
                wins_a: counts.wins_a,
                wins_b: counts.wins_b,
                draws: counts.draws,
                seed_of: counts.seeds,
            });
        }
        Ok(results)
    }

    /// Reduce tournament match outcomes to pairwise comparisons on a specific axis.
    pub fn reduce_pairwise(
        outcomes: &[MatchOutcome],
        axis: RatingAxis,
    ) -> Result<Vec<PairwiseResult>, RatingError> {
        if outcomes.is_empty() {
            return Err(RatingError::EmptyMatches);
        }
        let mut families_set = BTreeSet::new();
        for m in outcomes {
            for k in &m.spawn_order {
                families_set.insert(*k);
            }
            for k_str in m.per_family.keys() {
                families_set.insert(BrainKind::new(k_str.clone().leak()));
            }
        }
        let families: Vec<BrainKind> = families_set.into_iter().collect();
        if families.len() < 2 {
            return Err(RatingError::TooFewFamilies {
                families: families.len(),
            });
        }
        let refs: Vec<&MatchOutcome> = outcomes.iter().collect();
        reduce_pairwise_for_matches(&refs, &families, axis)
    }

    /// Analyze whether spawn order significantly affected family performance on an axis.
    pub fn analyze_order_effect(
        outcomes: &[MatchOutcome],
        axis: RatingAxis,
        ci_seed: u64,
        bootstrap_replicates: usize,
    ) -> OrderEffectReport {
        let mut seeds_set = BTreeSet::new();
        for m in outcomes {
            seeds_set.insert(m.seed);
        }
        let seeds: Vec<u64> = seeds_set.into_iter().collect();
        analyze_order_effect_internal(outcomes, axis, &seeds, ci_seed, bootstrap_replicates)
    }

    fn analyze_order_effect_internal(
        outcomes: &[MatchOutcome],
        axis: RatingAxis,
        seeds: &[u64],
        ci_seed: u64,
        bootstrap_replicates: usize,
    ) -> OrderEffectReport {
        let mut values_by_pos: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
        let mut seed_by_pos_val: BTreeMap<u32, Vec<(u64, f64)>> = BTreeMap::new();

        for m in outcomes {
            for (pos, family) in m.spawn_order.iter().enumerate() {
                let pos_u32 = pos as u32;
                if let Some(outcome) = m.per_family.get(family.as_str())
                    && let Ok(Some(val)) =
                        extract_axis_value(outcome, axis, m.ticks_run, family.as_str())
                {
                    values_by_pos.entry(pos_u32).or_default().push(val);
                    seed_by_pos_val
                        .entry(pos_u32)
                        .or_default()
                        .push((m.seed, val));
                }
            }
        }

        let mut position_means = BTreeMap::new();
        for (pos, vals) in &values_by_pos {
            let mean = if vals.is_empty() {
                0.0
            } else {
                vals.iter().sum::<f64>() / vals.len() as f64
            };
            position_means.insert(*pos, mean);
        }

        let mut position_ci95 = BTreeMap::new();
        if !seeds.is_empty() && bootstrap_replicates > 0 {
            let mut rng = SmallRng::seed_from_u64(ci_seed ^ 0x08D3_85E7);
            let s_count = seeds.len();

            for (pos, pairs) in &seed_by_pos_val {
                let mut seed_map: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
                for (s, v) in pairs {
                    seed_map.entry(*s).or_default().push(*v);
                }

                let mut boot_means = Vec::with_capacity(bootstrap_replicates);
                for _ in 0..bootstrap_replicates {
                    let mut sample_vals = Vec::new();
                    for _ in 0..s_count {
                        let idx = rng.random_range(0..s_count);
                        let s = seeds[idx];
                        if let Some(vs) = seed_map.get(&s) {
                            sample_vals.extend(vs.iter().copied());
                        }
                    }
                    let m = if sample_vals.is_empty() {
                        0.0
                    } else {
                        sample_vals.iter().sum::<f64>() / sample_vals.len() as f64
                    };
                    boot_means.push(m);
                }

                boot_means.sort_by(|a, b| a.total_cmp(b));
                let low_idx = ((0.025 * bootstrap_replicates as f64).floor() as usize)
                    .min(bootstrap_replicates - 1);
                let high_idx = ((0.975 * bootstrap_replicates as f64).floor() as usize)
                    .min(bootstrap_replicates - 1);
                position_ci95.insert(*pos, (boot_means[low_idx], boot_means[high_idx]));
            }
        }

        let mut detected = false;
        if let Some(ci_zero) = position_ci95.get(&0) {
            for (pos, ci) in &position_ci95 {
                if *pos != 0 && (ci_zero.0 > ci.1 || ci_zero.1 < ci.0) {
                    detected = true;
                    break;
                }
            }
        }

        OrderEffectReport {
            position_means,
            position_ci95,
            detected,
        }
    }

    /// Rate all brain families across tournament matches on all axes using Bradley-Terry with clustered bootstrap.
    pub fn rate_tournament(
        outcomes: &[MatchOutcome],
        options: &RatingOptions,
    ) -> Result<RatingTable, RatingError> {
        if outcomes.is_empty() {
            return Err(RatingError::EmptyMatches);
        }

        let mut families_set = BTreeSet::new();
        for m in outcomes {
            for k in &m.spawn_order {
                families_set.insert(*k);
            }
            for k_str in m.per_family.keys() {
                families_set.insert(BrainKind::new(k_str.clone().leak()));
            }
        }
        let families: Vec<BrainKind> = families_set.into_iter().collect();
        if families.len() < 2 {
            return Err(RatingError::TooFewFamilies {
                families: families.len(),
            });
        }

        let mut seeds_set = BTreeSet::new();
        for m in outcomes {
            seeds_set.insert(m.seed);
        }
        let seeds: Vec<u64> = seeds_set.into_iter().collect();
        let n_seeds = seeds.len();
        let n_matches = outcomes.len();

        let mut matches_by_seed: BTreeMap<u64, Vec<&MatchOutcome>> = BTreeMap::new();
        for m in outcomes {
            matches_by_seed.entry(m.seed).or_default().push(m);
        }

        let refs: Vec<&MatchOutcome> = outcomes.iter().collect();
        let mut axes_table = BTreeMap::new();
        let mut order_effects = BTreeMap::new();
        let mut warnings = Vec::new();

        if n_seeds < options.min_seeds {
            let warn_msg = format!(
                "tournament ran only {n_seeds} seed(s) (< {}); bootstrap CIs are wide and comparisons should be treated with caution",
                options.min_seeds
            );
            warnings.push(warn_msg);
        }

        for axis in RatingAxis::ALL {
            let mut all_vals = Vec::new();
            let mut family_vals: BTreeMap<BrainKind, Vec<f64>> = BTreeMap::new();
            let mut any_missing = false;

            for m in outcomes {
                for family in &families {
                    if let Some(outcome) = m.per_family.get(family.as_str()) {
                        match extract_axis_value(outcome, *axis, m.ticks_run, family.as_str())? {
                            Some(v) => {
                                all_vals.push(v);
                                family_vals.entry(*family).or_default().push(v);
                            }
                            None => {
                                any_missing = true;
                            }
                        }
                    } else {
                        any_missing = true;
                    }
                }
            }

            if *axis == RatingAxis::NoveltyCoverage && (all_vals.is_empty() || any_missing) {
                axes_table.insert(
                    *axis,
                    AxisRatingOutcome::Undetermined {
                        reason:
                            "novelty_coverage data is absent or incomplete for entered families"
                                .to_string(),
                    },
                );
                continue;
            }

            if all_vals.is_empty() {
                axes_table.insert(
                    *axis,
                    AxisRatingOutcome::Undetermined {
                        reason: format!("no outcome values recorded on axis {}", axis.as_str()),
                    },
                );
                continue;
            }

            // Zero-variance / identical values check
            let first_val = all_vals[0];
            let all_identical = all_vals.iter().all(|v| (v - first_val).abs() < EPSILON);
            if all_identical {
                axes_table.insert(
                    *axis,
                    AxisRatingOutcome::Undetermined {
                        reason: "all family outcomes on axis are identical (zero variance)"
                            .to_string(),
                    },
                );
                continue;
            }

            // Degenerate extinction check: did any family go extinct in 100% of matches with zero variance?
            if *axis == RatingAxis::TimeToExtinction {
                let mut degenerate_extinction = false;
                for vals in family_vals.values() {
                    if vals.len() == n_matches && vals.len() > 1 {
                        let v0 = vals[0];
                        let no_var = vals.iter().all(|v| (v - v0).abs() < EPSILON);
                        let all_extinct = v0 < outcomes[0].ticks_run as f64;
                        if no_var && all_extinct {
                            degenerate_extinction = true;
                            break;
                        }
                    }
                }
                if degenerate_extinction {
                    axes_table.insert(
                        *axis,
                        AxisRatingOutcome::Undetermined {
                            reason: "degenerate extinction observed: family went extinct in every match with zero variance"
                                .to_string(),
                        },
                    );
                    continue;
                }
            }

            // Reduce pairwise comparisons
            let pairwise = reduce_pairwise_for_matches(&refs, &families, *axis)?;
            let total_non_draws: u32 = pairwise.iter().map(|p| p.wins_a + p.wins_b).sum();
            if total_non_draws == 0 {
                axes_table.insert(
                    *axis,
                    AxisRatingOutcome::Undetermined {
                        reason: "all pairwise comparisons on axis resulted in draws".to_string(),
                    },
                );
                continue;
            }

            // Check for perfect separation
            let mut family_wins: BTreeMap<BrainKind, f64> = BTreeMap::new();
            let mut family_losses: BTreeMap<BrainKind, f64> = BTreeMap::new();
            for p in &pairwise {
                *family_wins.entry(p.a).or_default() += p.effective_wins_a();
                *family_wins.entry(p.b).or_default() += p.effective_wins_b();
                *family_losses.entry(p.a).or_default() += p.effective_wins_b();
                *family_losses.entry(p.b).or_default() += p.effective_wins_a();
            }
            for f in &families {
                let wins = family_wins.get(f).copied().unwrap_or(0.0);
                let losses = family_losses.get(f).copied().unwrap_or(0.0);
                if (wins > 0.0 && losses == 0.0) || (losses > 0.0 && wins == 0.0) {
                    tracing::warn!(
                        target: "scriptbots::tournament::rating",
                        axis = axis.as_str(),
                        family = f.as_str(),
                        "perfect separation detected; prior engaged to regularize MLE"
                    );
                }
            }

            if n_seeds < options.min_seeds {
                tracing::warn!(
                    target: "scriptbots::tournament::rating",
                    axis = axis.as_str(),
                    n_seeds,
                    "CI is wide and this comparison should not be cited (fewer than 8 seeds)"
                );
            }

            // Fit point estimate
            let fit = fit_bradley_terry(
                &families,
                &pairwise,
                options.prior,
                options.tolerance,
                options.max_iterations,
            )?;

            if !fit.converged {
                tracing::warn!(
                    target: "scriptbots::tournament::rating",
                    axis = axis.as_str(),
                    iterations = fit.iterations,
                    last_delta = fit.last_delta,
                    "Bradley-Terry fitting did not converge within iteration limit"
                );
                axes_table.insert(
                    *axis,
                    AxisRatingOutcome::Undetermined {
                        reason: format!(
                            "Bradley-Terry fitting did not converge after {} iterations",
                            fit.iterations
                        ),
                    },
                );
                continue;
            }

            tracing::info!(
                target: "scriptbots::tournament::rating",
                axis = axis.as_str(),
                families = families.len(),
                n_matches,
                n_seeds,
                bt_iterations = fit.iterations,
                converged = fit.converged,
                ci_seed = options.ci_seed,
                bootstrap_replicates = options.bootstrap_replicates,
                "fitted Bradley-Terry model for axis"
            );

            // Clustered bootstrap over seeds
            let mut rep_thetas: BTreeMap<BrainKind, Vec<f64>> = BTreeMap::new();
            let mut rep_diffs: BTreeMap<(BrainKind, BrainKind), Vec<f64>> = BTreeMap::new();
            for f in &families {
                rep_thetas.insert(*f, Vec::with_capacity(options.bootstrap_replicates));
            }
            for i in 0..families.len() {
                for j in (i + 1)..families.len() {
                    rep_diffs.insert(
                        (families[i], families[j]),
                        Vec::with_capacity(options.bootstrap_replicates),
                    );
                }
            }

            let mut rng = SmallRng::seed_from_u64(options.ci_seed ^ 0x5EED_0001);
            for _ in 0..options.bootstrap_replicates {
                let mut resampled_matches = Vec::with_capacity(outcomes.len());
                for _ in 0..n_seeds {
                    let s_idx = rng.random_range(0..n_seeds);
                    let s = seeds[s_idx];
                    if let Some(list) = matches_by_seed.get(&s) {
                        resampled_matches.extend(list.iter().copied());
                    }
                }

                if let Ok(res_pairwise) =
                    reduce_pairwise_for_matches(&resampled_matches, &families, *axis)
                    && let Ok(fit_b) = fit_bradley_terry(
                        &families,
                        &res_pairwise,
                        options.prior,
                        options.tolerance,
                        options.max_iterations,
                    )
                    && fit_b.converged
                {
                    for f in &families {
                        if let Some(th) = fit_b.theta.get(f) {
                            rep_thetas.entry(*f).or_default().push(*th);
                        }
                    }
                    for i in 0..families.len() {
                        for j in (i + 1)..families.len() {
                            let a = families[i];
                            let b = families[j];
                            if let (Some(th_a), Some(th_b)) =
                                (fit_b.theta.get(&a), fit_b.theta.get(&b))
                            {
                                rep_diffs.entry((a, b)).or_default().push(th_a - th_b);
                            }
                        }
                    }
                }
            }

            // Compute family ratings
            let mut ratings_map = BTreeMap::new();
            for f in &families {
                let theta = fit.theta.get(f).copied().unwrap_or(0.0);
                let elo = fit.elo.get(f).copied().unwrap_or(1500.0);

                let mut w_count = 0_u32;
                let mut l_count = 0_u32;
                let mut d_count = 0_u32;
                for p in &pairwise {
                    if p.a == *f {
                        w_count += p.wins_a;
                        l_count += p.wins_b;
                        d_count += p.draws;
                    } else if p.b == *f {
                        w_count += p.wins_b;
                        l_count += p.wins_a;
                        d_count += p.draws;
                    }
                }

                let (se, ci95) = if let Some(vals) = rep_thetas.get_mut(f) {
                    if vals.len() > 1 {
                        vals.sort_by(|a, b| a.total_cmp(b));
                        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                            / (vals.len() - 1) as f64;
                        let se = var.sqrt();
                        let low_idx =
                            ((0.025 * vals.len() as f64).floor() as usize).min(vals.len() - 1);
                        let high_idx =
                            ((0.975 * vals.len() as f64).floor() as usize).min(vals.len() - 1);
                        (se, (vals[low_idx], vals[high_idx]))
                    } else {
                        (0.0, (theta, theta))
                    }
                } else {
                    (0.0, (theta, theta))
                };

                let elo_ci95 = (theta_to_elo(ci95.0), theta_to_elo(ci95.1));

                let fam_rating = FamilyRating {
                    family: *f,
                    theta,
                    elo,
                    se,
                    ci95,
                    elo_ci95,
                    n_matches,
                    wins: w_count,
                    losses: l_count,
                    draws: d_count,
                };

                tracing::info!(
                    target: "scriptbots::tournament::rating",
                    family = f.as_str(),
                    theta = fam_rating.theta,
                    elo = fam_rating.elo,
                    ci95_low = fam_rating.ci95.0,
                    ci95_high = fam_rating.ci95.1,
                    n_matches = fam_rating.n_matches,
                    wins = fam_rating.wins,
                    losses = fam_rating.losses,
                    draws = fam_rating.draws,
                    "family rating on axis"
                );

                ratings_map.insert(*f, fam_rating);
            }

            // Compute pairwise comparisons
            let mut pairwise_comps = Vec::new();
            for i in 0..families.len() {
                for j in (i + 1)..families.len() {
                    let a = families[i];
                    let b = families[j];
                    let theta_a = fit.theta.get(&a).copied().unwrap_or(0.0);
                    let theta_b = fit.theta.get(&b).copied().unwrap_or(0.0);
                    let theta_diff = theta_a - theta_b;

                    let p_res = pairwise.iter().find(|p| p.a == a && p.b == b);
                    let (wins_a, wins_b, draws) = p_res
                        .map(|p| (p.wins_a, p.wins_b, p.draws))
                        .unwrap_or((0, 0, 0));
                    let total = wins_a + wins_b + draws;
                    let (win_rate_diff, cliffs_delta) = if total > 0 {
                        let d = (wins_a as f64 - wins_b as f64) / total as f64;
                        (d, d)
                    } else {
                        (0.0, 0.0)
                    };

                    let ci95 = if let Some(diffs) = rep_diffs.get_mut(&(a, b)) {
                        if diffs.len() > 1 {
                            diffs.sort_by(|x, y| x.total_cmp(y));
                            let low_idx = ((0.025 * diffs.len() as f64).floor() as usize)
                                .min(diffs.len() - 1);
                            let high_idx = ((0.975 * diffs.len() as f64).floor() as usize)
                                .min(diffs.len() - 1);
                            (diffs[low_idx], diffs[high_idx])
                        } else {
                            (theta_diff, theta_diff)
                        }
                    } else {
                        (theta_diff, theta_diff)
                    };

                    let verdict = if ci95.0 > 0.0 {
                        PairwiseVerdict::AWins
                    } else if ci95.1 < 0.0 {
                        PairwiseVerdict::BWins
                    } else {
                        PairwiseVerdict::Parity
                    };

                    let comp = PairwiseComparison {
                        a,
                        b,
                        theta_diff,
                        ci95,
                        verdict,
                        effect_size_win_rate_diff: win_rate_diff,
                        cliffs_delta,
                        wins_a,
                        wins_b,
                        draws,
                    };

                    tracing::info!(
                        target: "scriptbots::tournament::rating",
                        a = a.as_str(),
                        b = b.as_str(),
                        theta_diff = comp.theta_diff,
                        ci95_low = comp.ci95.0,
                        ci95_high = comp.ci95.1,
                        verdict = comp.verdict.as_str(),
                        effect_size = comp.effect_size_win_rate_diff,
                        "pairwise comparison on axis"
                    );

                    pairwise_comps.push(comp);
                }
            }

            // Order effect analysis
            let order_effect = analyze_order_effect_internal(
                outcomes,
                *axis,
                &seeds,
                options.ci_seed,
                options.bootstrap_replicates,
            );
            if order_effect.detected {
                tracing::warn!(
                    target: "scriptbots::tournament::rating",
                    axis = axis.as_str(),
                    position_means = ?order_effect.position_means,
                    position_ci95 = ?order_effect.position_ci95,
                    "spawn-order effect detected on axis"
                );
            }

            axes_table.insert(
                *axis,
                AxisRatingOutcome::Rated(AxisRatings {
                    axis: *axis,
                    n_matches,
                    n_seeds,
                    iterations: fit.iterations,
                    converged: fit.converged,
                    ci_seed: options.ci_seed,
                    bootstrap_replicates: options.bootstrap_replicates,
                    tolerance: options.tolerance,
                    interval_method: "clustered_seed_percentile_bootstrap".to_string(),
                    ratings: ratings_map,
                    pairwise: pairwise_comps,
                }),
            );
            order_effects.insert(*axis, order_effect);
        }

        Ok(RatingTable {
            axes: axes_table,
            order_effects,
            warnings,
        })
    }

    /// Rate reports directly from tournament execution.
    pub fn rate_reports(
        reports: &[MatchRunReport],
        options: &RatingOptions,
    ) -> Result<RatingTable, RatingError> {
        let outcomes: Vec<MatchOutcome> = reports.iter().map(|r| r.outcome.clone()).collect();
        rate_tournament(&outcomes, options)
    }
}

/// CI-regenerated leaderboard, provenance tracking, and drift detection (bd-16g.12.3).
pub mod leaderboard {
    use super::{
        FamilyOutcome, MatchId, MatchOutcome, MatchRunReport, OrderPolicy, RatingAxis,
        RatingOptions, RatingTable, TournamentError, TournamentSpec,
        execution::run_tournament_with_jobs, rating::rate_tournament,
    };
    use crate::BuildProvenanceV0;
    use scriptbots_brain::BrainKind;
    use scriptbots_core::ScriptBotsConfig;
    use serde::{Deserialize, Serialize};
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
        path::{Path, PathBuf},
    };

    fn default_title() -> String {
        "ScriptBots Canonical Brain-Family Tournament".to_owned()
    }

    fn default_version() -> String {
        "1.0.0".to_owned()
    }

    fn default_order_policy_str() -> String {
        "balanced_latin_square".to_owned()
    }

    fn default_sense_backend() -> String {
        "cpu_simd".to_owned()
    }

    fn default_sense_determinism() -> String {
        "exact".to_owned()
    }

    fn default_true() -> bool {
        true
    }

    /// Pinned tournament specification file (`tournament/spec.toml`).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct TournamentSpecFile {
        pub tournament: TournamentSpecHeader,
        #[serde(default)]
        pub parameters: Option<SpecParameters>,
        #[serde(default)]
        pub reproducibility: Option<SpecReproducibility>,
        #[serde(default)]
        pub smoke: Option<SmokeSpecConfig>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct TournamentSpecHeader {
        #[serde(default = "default_title")]
        pub title: String,
        #[serde(default = "default_version")]
        pub version: String,
        pub families: Vec<String>,
        pub seeds: Vec<u64>,
        pub ticks: u64,
        pub cohort_size: usize,
        #[serde(default = "default_order_policy_str")]
        pub order_policy: String,
        #[serde(default = "default_true")]
        pub closed: bool,
        #[serde(default = "default_sense_backend")]
        pub sense_backend: String,
        #[serde(default = "default_sense_determinism")]
        pub sense_determinism: String,
        #[serde(default = "default_true")]
        pub reproducible: bool,
        #[serde(default)]
        pub config_layers: Vec<PathBuf>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
    pub struct SpecParameters {
        #[serde(default = "default_nodes")]
        pub nodes_per_family: usize,
        #[serde(default = "default_true")]
        pub same_kind_mating_barrier: bool,
        #[serde(default)]
        pub interbreeding: bool,
    }

    fn default_nodes() -> usize {
        200
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
    pub struct SpecReproducibility {
        #[serde(default = "default_true")]
        pub require_identical_rerun: bool,
        #[serde(default = "default_retries")]
        pub max_retries: usize,
    }

    fn default_retries() -> usize {
        1
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
    pub struct SmokeSpecConfig {
        pub families: Vec<String>,
        pub seeds: Vec<u64>,
        pub ticks: u64,
        pub cohort_size: usize,
        #[serde(default = "default_both_assignments")]
        pub order_policy: String,
        #[serde(default = "default_true")]
        pub closed: bool,
    }

    fn default_both_assignments() -> String {
        "both_assignments".to_owned()
    }

    impl TournamentSpecFile {
        /// Load spec from a TOML file path.
        pub fn from_file(path: &Path) -> Result<Self, TournamentError> {
            let content = fs::read_to_string(path).map_err(|e| TournamentError::ConfigLayer {
                path: path.to_path_buf(),
                reason: format!("failed to read spec file: {e}"),
            })?;
            Self::from_toml_str(&content)
        }

        /// Parse spec from TOML string.
        pub fn from_toml_str(s: &str) -> Result<Self, TournamentError> {
            toml::from_str(s).map_err(|e| TournamentError::UnbalancedOrders {
                reason: format!("failed to parse tournament spec TOML: {e}"),
            })
        }

        /// Compute canonical Blake3 digest of the spec.
        #[must_use]
        pub fn spec_digest(&self) -> String {
            let canonical = serde_json::to_string(self).unwrap_or_default();
            blake3::hash(canonical.as_bytes()).to_hex().to_string()
        }

        /// Parse the order policy string.
        pub fn parse_order_policy(s: &str) -> Result<OrderPolicy, TournamentError> {
            match s.trim().to_ascii_lowercase().as_str() {
                "both_assignments" | "bothassignments" => Ok(OrderPolicy::BothAssignments),
                "balanced_latin_square" | "balancedlatinsquare" | "latin_square" => {
                    Ok(OrderPolicy::BalancedLatinSquare)
                }
                "all_permutations" | "allpermutations" | "factorial" => {
                    Ok(OrderPolicy::AllPermutations)
                }
                other => Err(TournamentError::UnbalancedOrders {
                    reason: format!(
                        "unknown order policy '{other}'; valid: both_assignments, balanced_latin_square, all_permutations"
                    ),
                }),
            }
        }

        /// Apply command-line overrides (families, seeds, ticks, cohort_size).
        pub fn apply_cli_overrides(
            &mut self,
            families: Option<&str>,
            seeds: Option<&str>,
            ticks: Option<u64>,
            cohort_size: Option<usize>,
        ) -> Result<(), TournamentError> {
            if let Some(fams) = families {
                let fams = fams.trim();
                if !fams.is_empty() && !fams.eq_ignore_ascii_case("all") {
                    let parsed: Vec<String> = fams
                        .split([',', ' ', '\t'])
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .map(ToOwned::to_owned)
                        .collect();
                    if parsed.is_empty() {
                        return Err(TournamentError::UnbalancedOrders {
                            reason: "empty families override specified".to_owned(),
                        });
                    }
                    self.tournament.families = parsed.clone();
                    if let Some(ref mut sm) = self.smoke {
                        sm.families = parsed;
                    }
                }
            }

            if let Some(seeds_str) = seeds {
                let seeds_str = seeds_str.trim();
                if !seeds_str.is_empty() {
                    let parsed_seeds = if let Ok(count) = seeds_str.parse::<usize>() {
                        if count == 0 {
                            return Err(TournamentError::UnbalancedOrders {
                                reason: "seeds count override must be >= 1".to_owned(),
                            });
                        }
                        if count <= self.tournament.seeds.len() {
                            self.tournament.seeds[..count].to_vec()
                        } else {
                            let start = self.tournament.seeds.first().copied().unwrap_or(101);
                            (start..start + count as u64).collect()
                        }
                    } else {
                        let parsed: Result<Vec<u64>, _> = seeds_str
                            .split([',', ' ', '\t'])
                            .map(|s| s.trim())
                            .filter(|s| !s.is_empty())
                            .map(|s| s.parse::<u64>())
                            .collect();
                        let parsed = parsed.map_err(|e| TournamentError::UnbalancedOrders {
                            reason: format!("invalid seeds list override: {e}"),
                        })?;
                        if parsed.is_empty() {
                            return Err(TournamentError::UnbalancedOrders {
                                reason: "empty seeds override specified".to_owned(),
                            });
                        }
                        parsed
                    };
                    self.tournament.seeds = parsed_seeds.clone();
                    if let Some(ref mut sm) = self.smoke {
                        sm.seeds = parsed_seeds;
                    }
                }
            }

            if let Some(t) = ticks {
                if t == 0 {
                    return Err(TournamentError::UnbalancedOrders {
                        reason: "ticks override must be >= 1".to_owned(),
                    });
                }
                self.tournament.ticks = t;
                if let Some(ref mut sm) = self.smoke {
                    sm.ticks = t;
                }
            }

            if let Some(c) = cohort_size {
                if c == 0 {
                    return Err(TournamentError::UnbalancedOrders {
                        reason: "cohort-size override must be >= 1".to_owned(),
                    });
                }
                self.tournament.cohort_size = c;
                if let Some(ref mut sm) = self.smoke {
                    sm.cohort_size = c;
                }
            }

            Ok(())
        }

        /// Convert full canonical spec to runtime [`TournamentSpec`].
        pub fn to_spec(&self) -> Result<TournamentSpec, TournamentError> {
            let families: Vec<BrainKind> = self
                .tournament
                .families
                .iter()
                .map(|f| to_brain_kind(f))
                .collect();
            let order_policy = Self::parse_order_policy(&self.tournament.order_policy)?;

            Ok(TournamentSpec {
                families,
                seeds: self.tournament.seeds.clone(),
                ticks: self.tournament.ticks,
                cohort_size: self.tournament.cohort_size,
                order_policy,
                closed: self.tournament.closed,
                config_layers: self.tournament.config_layers.clone(),
            })
        }

        /// Convert smoke spec configuration to runtime [`TournamentSpec`].
        pub fn to_smoke_spec(&self) -> Result<TournamentSpec, TournamentError> {
            if let Some(ref smoke) = self.smoke {
                let families: Vec<BrainKind> =
                    smoke.families.iter().map(|f| to_brain_kind(f)).collect();
                let order_policy = Self::parse_order_policy(&smoke.order_policy)?;
                Ok(TournamentSpec {
                    families,
                    seeds: smoke.seeds.clone(),
                    ticks: smoke.ticks,
                    cohort_size: smoke.cohort_size,
                    order_policy,
                    closed: smoke.closed,
                    config_layers: self.tournament.config_layers.clone(),
                })
            } else {
                let families: Vec<BrainKind> = self
                    .tournament
                    .families
                    .iter()
                    .take(2)
                    .map(|f| to_brain_kind(f))
                    .collect();
                let seeds = if self.tournament.seeds.len() >= 4 {
                    self.tournament.seeds[..4].to_vec()
                } else {
                    self.tournament.seeds.clone()
                };
                Ok(TournamentSpec {
                    families,
                    seeds,
                    ticks: 200,
                    cohort_size: 16,
                    order_policy: OrderPolicy::BothAssignments,
                    closed: true,
                    config_layers: Vec::new(),
                })
            }
        }
    }

    /// Convert an owned or borrowed family name string to canonical [`BrainKind`].
    #[must_use]
    pub fn to_brain_kind(s: &str) -> BrainKind {
        match s.trim() {
            "mlp" | "mlp.baseline" => BrainKind::new("mlp"),
            "dwraon" | "dwraon.baseline" => BrainKind::new("dwraon"),
            "assembly" | "assembly.experimental" => BrainKind::new("assembly"),
            "neuro" | "ml.neuroflow" => BrainKind::new("neuro"),
            other => BrainKind::new(other.to_string().leak()),
        }
    }

    /// Machine-readable row in `tournament_results.jsonl`.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct TournamentResultRow {
        pub run_id: String,
        pub match_id: u64,
        pub seed: u64,
        pub world_seed: u64,
        pub brain_seed: u64,
        pub family: String,
        pub spawn_order_index: u32,
        pub survival_share: f64,
        pub biomass_share: f64,
        pub mean_lineage_depth: f64,
        pub max_lineage_depth: u32,
        pub extinct_at: Option<u64>,
        pub novelty_coverage: Option<f64>,
        pub agents_final: usize,
        pub config_digest: String,
        pub manifest_digest: String,
        pub reproducible: bool,
        pub sense_backend: String,
        pub sense_determinism: String,
        pub warnings: Vec<String>,
    }

    impl TournamentResultRow {
        /// Serialize a slice of rows to JSON Lines format.
        pub fn serialize_jsonl(rows: &[Self]) -> Result<String, serde_json::Error> {
            let mut out = String::new();
            for row in rows {
                let line = serde_json::to_string(row)?;
                out.push_str(&line);
                out.push('\n');
            }
            Ok(out)
        }

        /// Parse rows from JSON Lines format.
        pub fn deserialize_jsonl(s: &str) -> Result<Vec<Self>, serde_json::Error> {
            let mut rows = Vec::new();
            for line in s.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let row: Self = serde_json::from_str(trimmed)?;
                rows.push(row);
            }
            Ok(rows)
        }
    }

    /// Reason why a match outcome row was excluded from the rating leaderboard.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum ExclusionReason {
        NotReproducible,
        ApproximateSense(String),
        ConfigDigestMismatch { expected: String, found: String },
    }

    /// Audit record for an excluded tournament run.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ExcludedRunRecord {
        pub run_id: String,
        pub match_id: u64,
        pub family: String,
        pub seed: u64,
        pub reason: ExclusionReason,
    }

    /// Convert execution reports to structured result rows.
    #[must_use]
    pub fn reports_to_result_rows(
        reports: &[MatchRunReport],
        sense_backend: &str,
        sense_determinism: &str,
        reproducible: bool,
    ) -> Vec<TournamentResultRow> {
        let mut rows = Vec::new();
        for report in reports {
            let match_id = report.outcome.match_id.0;
            let seed = report.outcome.seed;
            let world_seed = report.plan.world_seed;
            let brain_seed = report.plan.brain_seed;
            let spawn_order_index = report.outcome.spawn_order_index;
            let config_digest = report.config_digest.clone();

            for (family_name, outcome) in &report.outcome.per_family {
                let run_id = format!("run-{match_id:016x}-{family_name}");
                let manifest_raw = format!(
                    "{run_id}:{config_digest}:{seed}:{world_seed}:{brain_seed}:{reproducible}:{sense_backend}:{sense_determinism}"
                );
                let manifest_digest = blake3::hash(manifest_raw.as_bytes()).to_hex().to_string();

                let agents_final = report
                    .agents_final
                    .get(&to_brain_kind(family_name))
                    .copied()
                    .unwrap_or(0);

                rows.push(TournamentResultRow {
                    run_id,
                    match_id,
                    seed,
                    world_seed,
                    brain_seed,
                    family: family_name.clone(),
                    spawn_order_index,
                    survival_share: outcome.survival_share,
                    biomass_share: outcome.biomass_share,
                    mean_lineage_depth: outcome.mean_lineage_depth,
                    max_lineage_depth: outcome.max_lineage_depth,
                    extinct_at: outcome.extinct_at,
                    novelty_coverage: outcome.novelty_coverage,
                    agents_final,
                    config_digest: config_digest.clone(),
                    manifest_digest,
                    reproducible,
                    sense_backend: sense_backend.to_owned(),
                    sense_determinism: sense_determinism.to_owned(),
                    warnings: report.outcome.warnings.clone(),
                });
            }
        }
        rows
    }

    /// Filter eligible result rows for rating computation, discarding and reporting invalid runs.
    #[must_use]
    pub fn filter_eligible_rows(
        rows: &[TournamentResultRow],
        expected_config_digest: &str,
    ) -> (Vec<TournamentResultRow>, Vec<ExcludedRunRecord>) {
        let mut eligible = Vec::with_capacity(rows.len());
        let mut excluded = Vec::new();

        for row in rows {
            if !row.reproducible {
                let record = ExcludedRunRecord {
                    run_id: row.run_id.clone(),
                    match_id: row.match_id,
                    family: row.family.clone(),
                    seed: row.seed,
                    reason: ExclusionReason::NotReproducible,
                };
                tracing::warn!(
                    target: "scriptbots::tournament::leaderboard",
                    run_id = %record.run_id,
                    match_id = record.match_id,
                    family = %record.family,
                    seed = record.seed,
                    reason = "reproducible=false",
                    "run excluded from tournament leaderboard"
                );
                excluded.push(record);
            } else if row.sense_determinism.eq_ignore_ascii_case("approximate") {
                let record = ExcludedRunRecord {
                    run_id: row.run_id.clone(),
                    match_id: row.match_id,
                    family: row.family.clone(),
                    seed: row.seed,
                    reason: ExclusionReason::ApproximateSense(row.sense_determinism.clone()),
                };
                tracing::warn!(
                    target: "scriptbots::tournament::leaderboard",
                    run_id = %record.run_id,
                    match_id = record.match_id,
                    family = %record.family,
                    seed = record.seed,
                    reason = "sense_determinism=approximate",
                    "run excluded from tournament leaderboard"
                );
                excluded.push(record);
            } else if !expected_config_digest.is_empty()
                && row.config_digest != expected_config_digest
            {
                let record = ExcludedRunRecord {
                    run_id: row.run_id.clone(),
                    match_id: row.match_id,
                    family: row.family.clone(),
                    seed: row.seed,
                    reason: ExclusionReason::ConfigDigestMismatch {
                        expected: expected_config_digest.to_owned(),
                        found: row.config_digest.clone(),
                    },
                };
                tracing::warn!(
                    target: "scriptbots::tournament::leaderboard",
                    run_id = %record.run_id,
                    match_id = record.match_id,
                    family = %record.family,
                    seed = record.seed,
                    expected = %expected_config_digest,
                    found = %row.config_digest,
                    reason = "config_digest_mismatch",
                    "run excluded from tournament leaderboard"
                );
                excluded.push(record);
            } else {
                eligible.push(row.clone());
            }
        }

        if !excluded.is_empty() {
            tracing::warn!(
                target: "scriptbots::tournament::leaderboard",
                excluded = excluded.len(),
                "runs excluded from tournament leaderboard"
            );
        }

        (eligible, excluded)
    }

    /// Reconstruct [`MatchOutcome`]s from eligible rows for rating calculation.
    #[must_use]
    #[derive(Default)]
    struct RowMatchAccumulator {
        seed: u64,
        spawn_order_index: u32,
        spawn_order: Vec<String>,
        per_family: BTreeMap<String, FamilyOutcome>,
        warnings: Vec<String>,
    }

    /// Reconstruct [`MatchOutcome`]s from eligible rows for rating calculation.
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn rows_to_match_outcomes(rows: &[TournamentResultRow]) -> Vec<MatchOutcome> {
        let mut by_match: BTreeMap<u64, RowMatchAccumulator> = BTreeMap::new();
        for r in rows {
            let entry = by_match
                .entry(r.match_id)
                .or_insert_with(|| RowMatchAccumulator {
                    seed: r.seed,
                    spawn_order_index: r.spawn_order_index,
                    spawn_order: Vec::new(),
                    per_family: BTreeMap::new(),
                    warnings: r.warnings.clone(),
                });
            if !entry.spawn_order.contains(&r.family) {
                entry.spawn_order.push(r.family.clone());
            }
            entry.per_family.insert(
                r.family.clone(),
                FamilyOutcome {
                    survival_share: r.survival_share,
                    biomass_share: r.biomass_share,
                    mean_lineage_depth: r.mean_lineage_depth,
                    max_lineage_depth: r.max_lineage_depth,
                    extinct_at: r.extinct_at,
                    novelty_coverage: r.novelty_coverage,
                },
            );
        }

        by_match
            .into_iter()
            .map(|(match_id, acc)| MatchOutcome {
                match_id: MatchId(match_id),
                seed: acc.seed,
                ticks_run: 0,
                spawn_order_index: acc.spawn_order_index,
                spawn_order: acc.spawn_order.iter().map(|s| to_brain_kind(s)).collect(),
                per_family: acc.per_family,
                warnings: acc.warnings,
            })
            .collect()
    }

    /// Generate the complete `docs/leaderboard.md` markdown content.
    #[must_use]
    pub fn generate_leaderboard_document(
        spec: &TournamentSpecFile,
        rating_table: &RatingTable,
        rows: &[TournamentResultRow],
        excluded: &[ExcludedRunRecord],
        config_digest: &str,
        build: &crate::BuildProvenanceV0,
    ) -> String {
        let mut out = String::new();

        out.push_str("<!-- DO NOT EDIT: Automatically generated by scriptbots-app tournament. Regenerate with: scriptbots-app tournament --spec tournament/spec.toml -->\n\n");
        out.push_str("# ScriptBots Canonical Brain-Family Tournament Leaderboard\n\n");
        out.push_str("Empirical multi-axis comparative evaluation of autonomous neural agent architectures under identical ecological pressure with regularized Bradley-Terry / Zermelo ratings and clustered bootstrap confidence intervals.\n\n");

        out.push_str("## Protocol & Scientific Boundary Conditions\n\n");
        out.push_str("| Property | Value |\n| :--- | :--- |\n");
        out.push_str(&format!(
            "| Specification Title | {} (v{}) |\n",
            spec.tournament.title, spec.tournament.version
        ));
        out.push_str(&format!(
            "| Spec Blake3 Digest | `{}` |\n",
            spec.spec_digest()
        ));
        out.push_str(&format!(
            "| Effective Config Digest | `{config_digest}` |\n"
        ));
        out.push_str(&format!(
            "| Entered Families | {} |\n",
            spec.tournament.families.join(", ")
        ));
        let seeds_summary = if spec.tournament.seeds.len() <= 8 {
            format!("{:?}", spec.tournament.seeds)
        } else {
            format!(
                "[{}, {}, ..., {}] ({} total)",
                spec.tournament.seeds[0],
                spec.tournament.seeds[1],
                spec.tournament.seeds.last().unwrap_or(&0),
                spec.tournament.seeds.len()
            )
        };
        out.push_str(&format!("| Evaluated Seeds | {seeds_summary} |\n"));
        out.push_str(&format!(
            "| Tick Budget | {} ticks per match |\n",
            spec.tournament.ticks
        ));
        let agents_per_fam = if spec.tournament.families.is_empty() {
            0
        } else {
            spec.tournament.cohort_size / spec.tournament.families.len()
        };
        out.push_str(&format!(
            "| Cohort Size | {} total agents ({} per family) |\n",
            spec.tournament.cohort_size, agents_per_fam
        ));
        out.push_str(&format!(
            "| Order Policy | `{}` |\n",
            spec.tournament.order_policy
        ));
        out.push_str(&format!(
            "| Closed World | `{}` (extinction is terminal, zero respawn) |\n",
            spec.tournament.closed
        ));
        out.push_str("| Reproductive Isolation | Same-kind mating barrier enforced (zero cross-family mating) |\n");
        let nodes = spec.parameters.as_ref().map_or(200, |p| p.nodes_per_family);
        out.push_str(&format!(
            "| Parameter Complexity | {nodes} nodes per family across all architectures |\n"
        ));
        out.push_str(&format!(
            "| Sensory Lane | `{}` (Determinism: `{}`) |\n",
            spec.tournament.sense_backend, spec.tournament.sense_determinism
        ));
        let source_rev = build.source_revision.as_deref().unwrap_or("unknown");
        let toolchain = build
            .compiler_toolchain
            .as_deref()
            .unwrap_or(&build.declared_toolchain);
        let target = format!("{}-{}", build.core.target_arch, build.core.target_os);
        out.push_str(&format!(
            "| Build Source Revision | `{source_rev}` |\n\
             | Build Toolchain | `{toolchain}` |\n\
             | Build Target | `{target}` (parallel: {}, simd: {}) |\n\
             | Cargo Lockfile Digest | `{}` |\n\n",
            build.core.parallel, build.core.simd_wide, build.lockfile_digest
        ));

        if !rating_table.warnings.is_empty() {
            out.push_str("### Tournament Warnings & Qualifiers\n\n");
            for w in &rating_table.warnings {
                out.push_str(&format!("- ⚠️ {w}\n"));
            }
            out.push('\n');
        }

        out.push_str("## Multi-Axis Ratings & Confidence Intervals\n\n");

        for axis in RatingAxis::ALL {
            out.push_str(&format!("### Axis: {} (`{}`)\n\n", axis, axis.as_str()));
            match rating_table.axes.get(axis) {
                Some(super::AxisRatingOutcome::Rated(ratings)) => {
                    out.push_str(&format!(
                        "*Matches: {}, Seeds: {}, Iterations: {} ({}), Bootstrap Replicates: {}*\n\n",
                        ratings.n_matches,
                        ratings.n_seeds,
                        ratings.iterations,
                        if ratings.converged {
                            "converged"
                        } else {
                            "unconverged"
                        },
                        ratings.bootstrap_replicates
                    ));

                    out.push_str(
                        "| Rank | Family | Elo Rating (95% CI) | Theta (95% CI) | SE | Matches | W / L / D | Evidence Provenance |\n\
                         | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n",
                    );

                    let mut sorted: Vec<_> = ratings.ratings.values().collect();
                    sorted.sort_by(|a, b| {
                        b.elo
                            .total_cmp(&a.elo)
                            .then_with(|| a.family.as_str().cmp(b.family.as_str()))
                    });

                    for (rank, r) in sorted.iter().enumerate() {
                        let sign = if r.theta >= 0.0 { "+" } else { "" };
                        let prov_tag = format!("[^{}-{}]", r.family.as_str(), axis.as_str());
                        out.push_str(&format!(
                            "| {} | {} | {:.1} [{:.1}, {:.1}] | {}{:.3} [{:.3}, {:.3}] | {:.3} | {} | {} / {} / {} | {} |\n",
                            rank + 1,
                            r.family.as_str(),
                            r.elo,
                            r.elo_ci95.0,
                            r.elo_ci95.1,
                            sign,
                            r.theta,
                            r.ci95.0,
                            r.ci95.1,
                            r.se,
                            r.n_matches,
                            r.wins,
                            r.losses,
                            r.draws,
                            prov_tag
                        ));
                    }
                    out.push('\n');

                    if !ratings.pairwise.is_empty() {
                        out.push_str(
                            "#### Pairwise Head-to-Head Comparisons\n\n\
                             | Family A | Family B | Verdict | Theta Diff (95% CI) | Win Rate Diff | Cliff's Delta | W / L / D |\n\
                             | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n",
                        );
                        for pair in &ratings.pairwise {
                            let sign = if pair.theta_diff >= 0.0 { "+" } else { "" };
                            out.push_str(&format!(
                                "| {} | {} | `{}` | {}{:.3} [{:.3}, {:.3}] | {:+.3} | {:+.3} | {} / {} / {} |\n",
                                pair.a.as_str(),
                                pair.b.as_str(),
                                pair.verdict.as_str(),
                                sign,
                                pair.theta_diff,
                                pair.ci95.0,
                                pair.ci95.1,
                                pair.effect_size_win_rate_diff,
                                pair.cliffs_delta,
                                pair.wins_a,
                                pair.wins_b,
                                pair.draws,
                            ));
                        }
                        out.push('\n');
                    }

                    if let Some(order_effect) = rating_table.order_effects.get(axis) {
                        if order_effect.detected {
                            out.push_str("> ⚠️ **Spawn-Order Effect Detected**: Spawn order significantly affected performance on this axis.\n\n");
                        }
                        out.push_str("#### Spawn-Order Position Performance\n");
                        for (pos, mean) in &order_effect.position_means {
                            if let Some(ci) = order_effect.position_ci95.get(pos) {
                                out.push_str(&format!(
                                    "- Position {}: mean = {:.3} (95% CI [{:.3}, {:.3}])\n",
                                    pos, mean, ci.0, ci.1
                                ));
                            } else {
                                out.push_str(&format!("- Position {}: mean = {:.3}\n", pos, mean));
                            }
                        }
                        out.push('\n');
                    }
                }
                Some(super::AxisRatingOutcome::Undetermined { reason }) => {
                    out.push_str(&format!("*Undetermined: {reason}*\n\n"));
                }
                None => {
                    out.push_str("*Not evaluated on this axis.*\n\n");
                }
            }
        }

        out.push_str("## Exclusion Audit\n\n");
        out.push_str(&format!("- Total Excluded Runs: {}\n", excluded.len()));
        if excluded.is_empty() {
            out.push_str("*Zero exclusions: all match rows verified reproducible (`reproducible = true`), exact-sense (`sense_determinism = exact`), and matching canonical config digest.*\n\n");
        } else {
            out.push_str("### Discarded Observations\n\n");
            for rec in excluded {
                let reason_str = match &rec.reason {
                    ExclusionReason::NotReproducible => "reproducible: false".to_owned(),
                    ExclusionReason::ApproximateSense(det) => {
                        format!("sense_determinism: approximate ({det})")
                    }
                    ExclusionReason::ConfigDigestMismatch { expected, found } => {
                        format!("config_digest_mismatch: expected {expected}, found {found}")
                    }
                };
                out.push_str(&format!(
                    "- Run `{}` (Family: `{}`, Match: `{}`, Seed: `{}`): {}\n",
                    rec.run_id, rec.family, rec.match_id, rec.seed, reason_str
                ));
            }
            out.push('\n');
        }

        out.push_str("## Run & Match Provenance Footnotes\n\n");
        let mut matches_by_family: BTreeMap<&str, BTreeSet<u64>> = BTreeMap::new();
        for r in rows {
            matches_by_family
                .entry(&r.family)
                .or_default()
                .insert(r.match_id);
        }

        for (family, match_ids) in matches_by_family {
            for axis in RatingAxis::ALL {
                let tag = format!("{family}-{}", axis.as_str());
                let sample_matches = match_ids
                    .iter()
                    .take(8)
                    .map(|id| format!("{id}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let more = if match_ids.len() > 8 {
                    format!(" and {} more", match_ids.len() - 8)
                } else {
                    String::new()
                };
                out.push_str(&format!(
                    "[^{tag}]: Derived from {} matches across entered seeds (Match IDs: {sample_matches}{more}).\n",
                    match_ids.len()
                ));
            }
        }

        out
    }

    /// Compute line-by-line unified diff between two text buffers.
    #[must_use]
    pub fn compute_unified_diff(
        expected: &str,
        actual: &str,
        expected_label: &str,
        actual_label: &str,
    ) -> String {
        let expected_lines: Vec<&str> = expected.lines().collect();
        let actual_lines: Vec<&str> = actual.lines().collect();

        let mut out = String::new();
        out.push_str(&format!("--- {expected_label}\n+++ {actual_label}\n"));

        let max_len = expected_lines.len().max(actual_lines.len());
        let mut diff_count = 0;
        for i in 0..max_len {
            let e = expected_lines.get(i);
            let a = actual_lines.get(i);
            if e != a {
                diff_count += 1;
                out.push_str(&format!("@@ line {} @@\n", i + 1));
                if let Some(exp) = e {
                    out.push_str(&format!("-{exp}\n"));
                }
                if let Some(act) = a {
                    out.push_str(&format!("+{act}\n"));
                }
                if diff_count > 50 {
                    out.push_str("... (truncated additional diff lines)\n");
                    break;
                }
            }
        }
        out
    }

    /// Error returned when `--check` encounters document or config drift.
    #[derive(Debug, Clone, PartialEq, thiserror::Error)]
    pub enum LeaderboardCheckError {
        #[error("committed leaderboard file not found at {path}: {reason}")]
        FileNotFound { path: PathBuf, reason: String },
        #[error(
            "config digest drift detected: committed records {committed}, freshly generated records {generated}"
        )]
        ConfigDrift {
            committed: String,
            generated: String,
        },
        #[error(
            "leaderboard document drift detected across {mismatched_lines} lines at {path}:\n{diff}"
        )]
        DocumentDrift {
            mismatched_lines: usize,
            diff: String,
            path: PathBuf,
        },
    }

    /// Check if a committed leaderboard matches newly generated content, verifying config digest and text equality.
    pub fn check_leaderboard_drift(
        committed_text: &str,
        generated_text: &str,
        path: &Path,
    ) -> Result<(), LeaderboardCheckError> {
        // 1. Config drift check
        fn extract_config_digest(text: &str) -> Option<String> {
            for line in text.lines() {
                if !line.contains("Effective Config Digest") {
                    continue;
                }
                let start = line.find('`')?;
                let end = line[start + 1..].find('`')?;
                return Some(line[start + 1..start + 1 + end].to_owned());
            }
            None
        }

        let comm_digest = extract_config_digest(committed_text);
        let gen_digest = extract_config_digest(generated_text);
        if let (Some(comm), Some(generated)) = (comm_digest, gen_digest)
            && comm != generated
        {
            tracing::error!(
                target: "scriptbots::tournament::leaderboard",
                path = %path.display(),
                committed = %comm,
                generated = %generated,
                "leaderboard check failed: configuration drift"
            );
            return Err(LeaderboardCheckError::ConfigDrift {
                committed: comm,
                generated,
            });
        }

        // 2. Full document byte/line equality check
        if committed_text.trim() == generated_text.trim() {
            return Ok(());
        }

        let diff = compute_unified_diff(
            committed_text,
            generated_text,
            &format!("committed: {}", path.display()),
            "generated",
        );
        let mismatched_lines = diff.lines().filter(|l| l.starts_with("@@")).count();

        tracing::error!(
            target: "scriptbots::tournament::leaderboard",
            path = %path.display(),
            diff = %diff,
            "leaderboard check failed with document drift"
        );

        Err(LeaderboardCheckError::DocumentDrift {
            mismatched_lines,
            diff,
            path: path.to_path_buf(),
        })
    }

    /// Summary outcome of a tournament leaderboard run.
    #[derive(Debug, Clone)]
    pub struct LeaderboardPipelineOutcome {
        pub spec: TournamentSpecFile,
        pub config_digest: String,
        pub spec_digest: String,
        pub total_matches: usize,
        pub total_rows: usize,
        pub eligible_rows: usize,
        pub excluded_rows: usize,
        pub results_jsonl: String,
        pub ratings_json: String,
        pub leaderboard_md: String,
        pub results_digest: String,
        pub ratings_digest: String,
    }

    /// Save tournament artifacts (`tournament_results.jsonl`, `ratings.json`, `leaderboard.md`).
    pub fn save_tournament_artifacts(
        outcome: &LeaderboardPipelineOutcome,
        out_path: &Path,
        leaderboard_path: Option<&Path>,
    ) -> Result<(), std::io::Error> {
        let (out_dir, lb_file) = if out_path.extension().and_then(|s| s.to_str()) == Some("md") {
            (
                out_path.parent().unwrap_or_else(|| Path::new(".")),
                out_path.to_path_buf(),
            )
        } else {
            (out_path, out_path.join("leaderboard.md"))
        };

        fs::create_dir_all(out_dir)?;
        fs::write(
            out_dir.join("tournament_results.jsonl"),
            &outcome.results_jsonl,
        )?;
        fs::write(out_dir.join("ratings.json"), &outcome.ratings_json)?;
        fs::write(&lb_file, &outcome.leaderboard_md)?;

        if let Some(target_lb) = leaderboard_path
            && target_lb != lb_file
        {
            if let Some(parent) = target_lb.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(target_lb, &outcome.leaderboard_md)?;
        }

        Ok(())
    }

    /// Execute the complete tournament leaderboard pipeline: plan -> run -> rate -> render -> check -> emit.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_leaderboard_tournament(
        spec_file: &TournamentSpecFile,
        spec_path: Option<&Path>,
        is_smoke: bool,
        base_config: &ScriptBotsConfig,
        build_provenance: &BuildProvenanceV0,
        out_dir: Option<&Path>,
        check_target: Option<&Path>,
        check_reproducibility: bool,
        jobs: usize,
    ) -> Result<LeaderboardPipelineOutcome, TournamentError> {
        let runtime_spec = if is_smoke {
            spec_file.to_smoke_spec()?
        } else {
            spec_file.to_spec()?
        };

        let planned_matches = super::plan(&runtime_spec)?.len();
        let spec_path_str = spec_path
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "tournament/spec.toml".to_string());
        let build_rev = build_provenance
            .source_revision
            .as_deref()
            .unwrap_or("unknown");

        let mut effective_base_config = base_config.clone();
        let prevent_cross_mating = spec_file
            .parameters
            .as_ref()
            .is_none_or(|p| p.same_kind_mating_barrier || !p.interbreeding);
        if prevent_cross_mating {
            effective_base_config.reproduction_partner_chance = 0.0;
        }

        let mut digest_config = effective_base_config.clone();
        digest_config.rng_seed = None;
        digest_config.closed = runtime_spec.closed;
        let expected_config_digest = blake3::hash(
            serde_json::to_string(&(digest_config, runtime_spec.ticks))
                .unwrap_or_default()
                .as_bytes(),
        )
        .to_hex()
        .to_string();

        tracing::info!(
            target: "scriptbots::tournament::leaderboard",
            spec_path = %spec_path_str,
            spec_digest = %spec_file.spec_digest(),
            families = ?runtime_spec.families,
            seeds = ?runtime_spec.seeds,
            ticks = runtime_spec.ticks,
            matches = planned_matches,
            config_digest = %expected_config_digest,
            build_digest = %build_rev,
            sense_backend = %spec_file.tournament.sense_backend,
            sense_determinism = %spec_file.tournament.sense_determinism,
            "starting tournament leaderboard execution"
        );

        let reports = run_tournament_with_jobs(&runtime_spec, &effective_base_config, jobs)?;
        super::enforce_no_config_drift(&reports)?;

        let config_digest = reports
            .first()
            .map(|r| r.config_digest.clone())
            .unwrap_or_default();

        for report in &reports {
            for (family, outcome) in &report.outcome.per_family {
                tracing::info!(
                    target: "scriptbots::tournament::leaderboard",
                    match_id = report.outcome.match_id.0,
                    run_id = format!("run-{:016x}-{family}", report.outcome.match_id.0),
                    seed = report.outcome.seed,
                    family = %family,
                    spawn_order_index = report.outcome.spawn_order_index,
                    survival_share = outcome.survival_share,
                    biomass_share = outcome.biomass_share,
                    lineage_depth = outcome.mean_lineage_depth,
                    "match outcome row"
                );
            }
        }

        let rows = reports_to_result_rows(
            &reports,
            &spec_file.tournament.sense_backend,
            &spec_file.tournament.sense_determinism,
            spec_file.tournament.reproducible,
        );

        let (eligible_rows, excluded_runs) = filter_eligible_rows(&rows, &config_digest);
        let eligible_outcomes = rows_to_match_outcomes(&eligible_rows);

        let rating_options = RatingOptions {
            bootstrap_replicates: if is_smoke { 50 } else { 2000 },
            ..RatingOptions::default()
        };
        let rating_table = rate_tournament(&eligible_outcomes, &rating_options).map_err(|e| {
            TournamentError::UnbalancedOrders {
                reason: e.to_string(),
            }
        })?;

        let leaderboard_md = generate_leaderboard_document(
            spec_file,
            &rating_table,
            &eligible_rows,
            &excluded_runs,
            &config_digest,
            build_provenance,
        );

        let results_jsonl = TournamentResultRow::serialize_jsonl(&rows).map_err(|e| {
            TournamentError::UnbalancedOrders {
                reason: e.to_string(),
            }
        })?;
        let ratings_json = serde_json::to_string_pretty(&rating_table).map_err(|e| {
            TournamentError::UnbalancedOrders {
                reason: e.to_string(),
            }
        })?;

        let results_digest = blake3::hash(results_jsonl.as_bytes()).to_hex().to_string();
        let ratings_digest = blake3::hash(ratings_json.as_bytes()).to_hex().to_string();

        if check_reproducibility {
            tracing::info!(
                target: "scriptbots::tournament::leaderboard",
                "running second pass to verify byte-level reproducibility gate"
            );
            let reports_rerun =
                run_tournament_with_jobs(&runtime_spec, &effective_base_config, jobs)?;
            let rows_rerun = reports_to_result_rows(
                &reports_rerun,
                &spec_file.tournament.sense_backend,
                &spec_file.tournament.sense_determinism,
                spec_file.tournament.reproducible,
            );
            let results_rerun_jsonl =
                TournamentResultRow::serialize_jsonl(&rows_rerun).map_err(|e| {
                    TournamentError::UnbalancedOrders {
                        reason: e.to_string(),
                    }
                })?;
            let rerun_digest = blake3::hash(results_rerun_jsonl.as_bytes())
                .to_hex()
                .to_string();

            if results_digest != rerun_digest {
                return Err(TournamentError::LeaderboardReproducibility {
                    rerun_digest,
                    results_digest,
                });
            }
            tracing::info!(
                target: "scriptbots::tournament::leaderboard",
                digest = %results_digest,
                "reproducibility gate passed: bit-identical results across invocations"
            );
        }

        let mut verdict = "ok";
        if let Some(check_path) = check_target {
            let committed =
                fs::read_to_string(check_path).map_err(|e| TournamentError::ConfigLayer {
                    path: check_path.to_path_buf(),
                    reason: format!("failed to read committed leaderboard for check: {e}"),
                })?;
            check_leaderboard_drift(&committed, &leaderboard_md, check_path).map_err(|e| {
                verdict = "drift";
                TournamentError::LeaderboardCheck(e)
            })?;
        }

        let outcome = LeaderboardPipelineOutcome {
            spec: spec_file.clone(),
            config_digest,
            spec_digest: spec_file.spec_digest(),
            total_matches: reports.len(),
            total_rows: rows.len(),
            eligible_rows: eligible_rows.len(),
            excluded_rows: excluded_runs.len(),
            results_jsonl,
            ratings_json,
            leaderboard_md,
            results_digest: results_digest.clone(),
            ratings_digest: ratings_digest.clone(),
        };

        if let Some(dir) = out_dir {
            save_tournament_artifacts(&outcome, dir, None).map_err(|e| {
                TournamentError::UnbalancedOrders {
                    reason: format!("failed to save tournament artifacts: {e}"),
                }
            })?;
        }

        tracing::info!(
            target: "scriptbots::tournament::leaderboard",
            verdict = verdict,
            matches = outcome.total_matches,
            excluded = outcome.excluded_rows,
            ratings_digest = %ratings_digest,
            results_digest = %results_digest,
            "leaderboard execution complete"
        );

        Ok(outcome)
    }

    /// CLI arguments for the `tournament` subcommand.
    #[derive(clap::Args, Debug, Clone, PartialEq)]
    pub struct TournamentArgs {
        /// Path to tournament specification TOML file.
        #[arg(long = "spec", default_value = "tournament/spec.toml")]
        pub spec: PathBuf,

        /// Brain families to enter, or "all" (e.g. "mlp,dwraon,assembly").
        #[arg(long = "families")]
        pub families: Option<String>,

        /// Number of seeds or comma-separated list of explicit seeds.
        #[arg(long = "seeds")]
        pub seeds: Option<String>,

        /// Simulation ticks per match.
        #[arg(long = "ticks")]
        pub ticks: Option<u64>,

        /// Total cohort size per match (split equally across families).
        #[arg(long = "cohort-size")]
        pub cohort_size: Option<usize>,

        /// Output directory for artifacts (tournament_results.jsonl, ratings.json, leaderboard.md).
        #[arg(long = "out")]
        pub out: Option<PathBuf>,

        /// Check target markdown file (e.g. docs/leaderboard.md) to detect drift against committed table.
        #[arg(long = "check")]
        pub check: Option<PathBuf>,

        /// Run bounded smoke tournament configuration instead of full tournament.
        #[arg(long = "smoke")]
        pub smoke: bool,

        /// Verify byte-level reproducibility by running twice and comparing output hashes.
        #[arg(long = "check-reproducibility")]
        pub check_reproducibility: bool,

        /// Worker parallelism (0 = Rayon default).
        #[arg(long = "jobs", default_value_t = 0)]
        pub jobs: usize,
    }

    /// Dispatch tournament execution from parsed CLI arguments.
    pub fn run_tournament_command(
        args: &TournamentArgs,
    ) -> anyhow::Result<LeaderboardPipelineOutcome> {
        let mut spec_file = if args.spec.exists() {
            TournamentSpecFile::from_file(&args.spec).map_err(|e| {
                anyhow::anyhow!(
                    "failed to read tournament spec {}: {e}",
                    args.spec.display()
                )
            })?
        } else {
            let candidates = [
                args.spec.as_path(),
                Path::new("tournament/spec.toml"),
                Path::new("../../tournament/spec.toml"),
            ];
            let found = candidates.iter().find(|p| p.exists());
            if let Some(p) = found {
                TournamentSpecFile::from_file(p).map_err(|e| {
                    anyhow::anyhow!("failed to read tournament spec {}: {e}", p.display())
                })?
            } else {
                anyhow::bail!(
                    "tournament specification file not found at {} (or fallback locations)",
                    args.spec.display()
                );
            }
        };

        spec_file
            .apply_cli_overrides(
                args.families.as_deref(),
                args.seeds.as_deref(),
                args.ticks,
                args.cohort_size,
            )
            .map_err(|e| anyhow::anyhow!("failed to apply CLI overrides: {e}"))?;

        let base_config = ScriptBotsConfig::default();
        let build_provenance = BuildProvenanceV0::current();

        let outcome = execute_leaderboard_tournament(
            &spec_file,
            Some(&args.spec),
            args.smoke,
            &base_config,
            &build_provenance,
            args.out.as_deref(),
            args.check.as_deref(),
            args.check_reproducibility,
            args.jobs,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;

        println!("Tournament Leaderboard Pipeline Complete:");
        println!("  Spec Digest:    {}", outcome.spec_digest);
        println!("  Config Digest:  {}", outcome.config_digest);
        println!("  Total Matches:  {}", outcome.total_matches);
        println!(
            "  Rows Recorded:  {} (Eligible: {}, Excluded: {})",
            outcome.total_rows, outcome.eligible_rows, outcome.excluded_rows
        );
        println!("  Results Digest: {}", outcome.results_digest);
        println!("  Ratings Digest: {}", outcome.ratings_digest);
        println!("  Verdict:        OK");

        Ok(outcome)
    }
}

/// Result record for a match between two or more brain families.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub seed: u64,
    pub ticks: u64,
    pub family_scores: HashMap<String, FamilyScore>,
}

/// Multi-axis performance score for a brain family in a tournament match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyScore {
    pub survival_share: f32,
    pub biomass_share: f32,
    pub max_generation: u32,
}

/// Elo rating record for a brain family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EloRating {
    pub family_id: String,
    pub rating: f64,
    pub matches_played: u32,
    pub wins: u32,
}

impl EloRating {
    pub fn new(family_id: impl Into<String>) -> Self {
        Self {
            family_id: family_id.into(),
            rating: 1500.0,
            matches_played: 0,
            wins: 0,
        }
    }

    /// Update Elo ratings for winner vs loser.
    pub fn update_elo(winner: &mut Self, loser: &mut Self, k_factor: f64) {
        let expected_w = 1.0 / (1.0 + 10.0_f64.powf((loser.rating - winner.rating) / 400.0));
        let expected_l = 1.0 / (1.0 + 10.0_f64.powf((winner.rating - loser.rating) / 400.0));

        winner.rating += k_factor * (1.0 - expected_w);
        loser.rating += k_factor * (0.0 - expected_l);

        winner.matches_played += 1;
        loser.matches_played += 1;
        winner.wins += 1;
    }
}

/// Tournament harness running matched-world competitions.
#[derive(Debug, Clone, Default)]
pub struct TournamentHarness {
    pub ratings: HashMap<String, EloRating>,
    pub match_history: Vec<MatchResult>,
}

impl TournamentHarness {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_family(&mut self, family_id: impl Into<String>) {
        let fid = family_id.into();
        self.ratings
            .entry(fid.clone())
            .or_insert_with(|| EloRating::new(fid));
    }

    pub fn record_match(&mut self, result: MatchResult) {
        // Auto-register any families mentioned in the match scores
        for family_id in result.family_scores.keys() {
            self.register_family(family_id);
        }

        // Record match and update ratings for top 2 families with deterministic multi-axis tie-breaking
        let mut sorted: Vec<_> = result.family_scores.iter().collect();
        sorted.sort_by(|a, b| {
            b.1.survival_share
                .total_cmp(&a.1.survival_share)
                .then_with(|| b.1.biomass_share.total_cmp(&a.1.biomass_share))
                .then_with(|| b.1.max_generation.cmp(&a.1.max_generation))
                .then_with(|| a.0.cmp(b.0))
        });

        if sorted.len() >= 2 {
            let winner_id = sorted[0].0;
            let loser_id = sorted[1].0;

            if winner_id != loser_id
                && let Some(mut winner) = self.ratings.get(winner_id).cloned()
                && let Some(mut loser) = self.ratings.get(loser_id).cloned()
            {
                EloRating::update_elo(&mut winner, &mut loser, 32.0);
                self.ratings.insert(winner_id.clone(), winner);
                self.ratings.insert(loser_id.clone(), loser);
            }
        }
        self.match_history.push(result);
    }

    pub fn generate_leaderboard_markdown(&self) -> String {
        let mut sorted_ratings: Vec<_> = self.ratings.values().collect();
        sorted_ratings.sort_by(|a, b| b.rating.total_cmp(&a.rating));

        let mut out = String::from(
            "# ScriptBots Brain Family Tournament Leaderboard\n\n\
             | Rank | Family ID | Elo Rating | Matches | Win Rate |\n\
             | :--- | :--- | :--- | :--- | :--- |\n",
        );

        for (i, r) in sorted_ratings.iter().enumerate() {
            let win_rate = if r.matches_played > 0 {
                (r.wins as f64 / r.matches_played as f64) * 100.0
            } else {
                0.0
            };
            out.push_str(&format!(
                "| {} | {} | {:.1} | {} | {:.1}% |\n",
                i + 1,
                r.family_id,
                r.rating,
                r.matches_played,
                win_rate
            ));
        }
        out
    }

    /// Rate tournament match outcomes using multi-axis Bradley-Terry with clustered bootstrap CIs.
    pub fn rate_outcomes(
        outcomes: &[MatchOutcome],
        options: &RatingOptions,
    ) -> Result<RatingTable, RatingError> {
        rate_tournament(outcomes, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rating::reduce_pairwise_for_matches;

    #[test]
    fn test_elo_update_math() {
        let mut mlp = EloRating::new("mlp");
        let mut dwraon = EloRating::new("dwraon");

        EloRating::update_elo(&mut mlp, &mut dwraon, 32.0);

        assert!(mlp.rating > 1500.0);
        assert!(dwraon.rating < 1500.0);
        assert_eq!(mlp.wins, 1);
    }

    #[test]
    fn test_tournament_harness_leaderboard_generation() {
        let mut harness = TournamentHarness::new();
        harness.register_family("mlp");
        harness.register_family("dwraon");

        let mut scores = HashMap::new();
        scores.insert(
            "mlp".to_string(),
            FamilyScore {
                survival_share: 0.7,
                biomass_share: 0.65,
                max_generation: 15,
            },
        );
        scores.insert(
            "dwraon".to_string(),
            FamilyScore {
                survival_share: 0.3,
                biomass_share: 0.35,
                max_generation: 12,
            },
        );

        harness.record_match(MatchResult {
            seed: 42,
            ticks: 5000,
            family_scores: scores,
        });

        let leaderboard = harness.generate_leaderboard_markdown();
        assert!(leaderboard.contains("mlp"));
        assert!(leaderboard.contains("dwraon"));
        assert!(leaderboard.contains("Elo Rating"));
    }

    #[test]
    fn test_bradley_terry_closed_form_three_family() {
        let fam_a = BrainKind::new("fam-a");
        let fam_b = BrainKind::new("fam-b");
        let fam_c = BrainKind::new("fam-c");
        let families = vec![fam_a, fam_b, fam_c];

        // A beats B 2:1, B beats C 2:1, A beats C 3:0
        let pairwise = vec![
            PairwiseResult {
                a: fam_a,
                b: fam_b,
                axis: RatingAxis::SurvivalShare,
                wins_a: 2,
                wins_b: 1,
                draws: 0,
                seed_of: vec![1, 2, 3],
            },
            PairwiseResult {
                a: fam_b,
                b: fam_c,
                axis: RatingAxis::SurvivalShare,
                wins_a: 2,
                wins_b: 1,
                draws: 0,
                seed_of: vec![4, 5, 6],
            },
            PairwiseResult {
                a: fam_a,
                b: fam_c,
                axis: RatingAxis::SurvivalShare,
                wins_a: 3,
                wins_b: 0,
                draws: 0,
                seed_of: vec![7, 8, 9],
            },
        ];

        let fit = fit_bradley_terry(&families, &pairwise, BtPrior::none(), 1e-12, 10_000)
            .expect("fit closed form");
        assert!(fit.converged);

        let theta_a = fit.theta[&fam_a];
        let theta_b = fit.theta[&fam_b];
        let theta_c = fit.theta[&fam_c];
        assert!(theta_a > theta_b);
        assert!(theta_b > theta_c);

        // Verify stationary condition: p_i * sum_{j != i} N_ij / (p_i + p_j) == W_i
        let p_a = theta_a.exp();
        let p_b = theta_b.exp();
        let p_c = theta_c.exp();
        let stat_a = p_a * (3.0 / (p_a + p_b) + 3.0 / (p_a + p_c));
        let stat_b = p_b * (3.0 / (p_b + p_a) + 3.0 / (p_b + p_c));
        let stat_c = p_c * (3.0 / (p_c + p_a) + 3.0 / (p_c + p_b));

        assert!(
            (stat_a - 5.0).abs() < 1e-9,
            "stat_a = {stat_a}, expected 5.0"
        );
        assert!(
            (stat_b - 3.0).abs() < 1e-9,
            "stat_b = {stat_b}, expected 3.0"
        );
        assert!(
            (stat_c - 1.0).abs() < 1e-9,
            "stat_c = {stat_c}, expected 1.0"
        );
    }

    #[test]
    fn test_bradley_terry_balanced_round_robin() {
        let fam_a = BrainKind::new("fam-a");
        let fam_b = BrainKind::new("fam-b");
        let fam_c = BrainKind::new("fam-c");
        let families = vec![fam_a, fam_b, fam_c];

        let pairwise = vec![
            PairwiseResult {
                a: fam_a,
                b: fam_b,
                axis: RatingAxis::SurvivalShare,
                wins_a: 1,
                wins_b: 1,
                draws: 0,
                seed_of: vec![1, 2],
            },
            PairwiseResult {
                a: fam_b,
                b: fam_c,
                axis: RatingAxis::SurvivalShare,
                wins_a: 1,
                wins_b: 1,
                draws: 0,
                seed_of: vec![3, 4],
            },
            PairwiseResult {
                a: fam_a,
                b: fam_c,
                axis: RatingAxis::SurvivalShare,
                wins_a: 1,
                wins_b: 1,
                draws: 0,
                seed_of: vec![5, 6],
            },
        ];

        let fit = fit_bradley_terry(&families, &pairwise, BtPrior::none(), 1e-12, 10_000)
            .expect("fit balanced");
        assert!(fit.converged);

        for f in &families {
            assert!(
                fit.theta[f].abs() < 1e-9,
                "anchored theta must be 0 to 1e-9"
            );
            assert!(
                (fit.elo[f] - 1500.0).abs() < 1e-9,
                "Elo must be exactly 1500 to 1e-9"
            );
        }
    }

    #[test]
    fn test_perfect_separation_with_and_without_prior() {
        let fam_a = BrainKind::new("fam-a");
        let fam_b = BrainKind::new("fam-b");
        let fam_c = BrainKind::new("fam-c");
        let families = vec![fam_a, fam_b, fam_c];

        // A wins every match (10 vs B, 10 vs C), B vs C is 1:1
        let pairwise = vec![
            PairwiseResult {
                a: fam_a,
                b: fam_b,
                axis: RatingAxis::SurvivalShare,
                wins_a: 10,
                wins_b: 0,
                draws: 0,
                seed_of: (1..=10).collect(),
            },
            PairwiseResult {
                a: fam_a,
                b: fam_c,
                axis: RatingAxis::SurvivalShare,
                wins_a: 10,
                wins_b: 0,
                draws: 0,
                seed_of: (11..=20).collect(),
            },
            PairwiseResult {
                a: fam_b,
                b: fam_c,
                axis: RatingAxis::SurvivalShare,
                wins_a: 1,
                wins_b: 1,
                draws: 0,
                seed_of: vec![21, 22],
            },
        ];

        // With prior (default pseudo_wins = 0.5): finite and bounded
        let fit_prior = fit_bradley_terry(&families, &pairwise, BtPrior::new(0.5), 1e-10, 10_000)
            .expect("fit with prior");
        assert!(fit_prior.converged);
        let theta_a = fit_prior.theta[&fam_a];
        assert!(theta_a.is_finite());
        assert!(theta_a > 0.0);
        assert!(theta_a < 10.0, "prior must bound the estimate: {theta_a}");

        // Without prior (pseudo_wins = 0.0): fails to converge or diverges
        let fit_no_prior = fit_bradley_terry(&families, &pairwise, BtPrior::none(), 1e-10, 10_000)
            .expect("fit no prior");
        assert!(
            !fit_no_prior.converged,
            "without prior, perfect separation must not converge"
        );
    }

    #[test]
    fn test_invariance_properties() {
        let fam_a = BrainKind::new("fam-a");
        let fam_b = BrainKind::new("fam-b");
        let fam_c = BrainKind::new("fam-c");
        let families = vec![fam_a, fam_b, fam_c];

        let pairwise1 = vec![
            PairwiseResult {
                a: fam_a,
                b: fam_b,
                axis: RatingAxis::SurvivalShare,
                wins_a: 4,
                wins_b: 2,
                draws: 1,
                seed_of: vec![1, 2],
            },
            PairwiseResult {
                a: fam_b,
                b: fam_c,
                axis: RatingAxis::SurvivalShare,
                wins_a: 3,
                wins_b: 1,
                draws: 0,
                seed_of: vec![3, 4],
            },
            PairwiseResult {
                a: fam_a,
                b: fam_c,
                axis: RatingAxis::SurvivalShare,
                wins_a: 5,
                wins_b: 1,
                draws: 2,
                seed_of: vec![5, 6],
            },
        ];

        // Reversed row order
        let mut pairwise2 = pairwise1.clone();
        pairwise2.reverse();

        let fit1 = fit_bradley_terry(&families, &pairwise1, BtPrior::default(), 1e-12, 10_000)
            .expect("fit1");
        let fit2 = fit_bradley_terry(&families, &pairwise2, BtPrior::default(), 1e-12, 10_000)
            .expect("fit2");

        for f in &families {
            assert_eq!(
                fit1.theta[f], fit2.theta[f],
                "row reordering must produce bit-identical thetas"
            );
            assert_eq!(
                fit1.elo[f], fit2.elo[f],
                "row reordering must produce bit-identical Elos"
            );
        }

        // Relabeling families
        let fam_x = BrainKind::new("fam-x");
        let fam_y = BrainKind::new("fam-y");
        let fam_z = BrainKind::new("fam-z");
        let families_relabeled = vec![fam_x, fam_y, fam_z];
        let pairwise_relabeled = vec![
            PairwiseResult {
                a: fam_x,
                b: fam_y,
                axis: RatingAxis::SurvivalShare,
                wins_a: 4,
                wins_b: 2,
                draws: 1,
                seed_of: vec![1, 2],
            },
            PairwiseResult {
                a: fam_y,
                b: fam_z,
                axis: RatingAxis::SurvivalShare,
                wins_a: 3,
                wins_b: 1,
                draws: 0,
                seed_of: vec![3, 4],
            },
            PairwiseResult {
                a: fam_x,
                b: fam_z,
                axis: RatingAxis::SurvivalShare,
                wins_a: 5,
                wins_b: 1,
                draws: 2,
                seed_of: vec![5, 6],
            },
        ];

        let fit_relabeled = fit_bradley_terry(
            &families_relabeled,
            &pairwise_relabeled,
            BtPrior::default(),
            1e-12,
            10_000,
        )
        .expect("fit_relabeled");

        assert_eq!(fit1.elo[&fam_a], fit_relabeled.elo[&fam_x]);
        assert_eq!(fit1.elo[&fam_b], fit_relabeled.elo[&fam_y]);
        assert_eq!(fit1.elo[&fam_c], fit_relabeled.elo[&fam_z]);
    }

    #[test]
    fn test_bootstrap_determinism() {
        let fam_a = BrainKind::new("mlp");
        let fam_b = BrainKind::new("dwraon");
        let mut outcomes = Vec::new();

        for seed in 1..=8 {
            for ord in 0..2 {
                let mut per_family = BTreeMap::new();
                let share_a = if ord == 0 { 0.65 } else { 0.60 };
                let share_b = 1.0 - share_a;
                per_family.insert(
                    fam_a.as_str().to_string(),
                    FamilyOutcome {
                        survival_share: share_a,
                        biomass_share: share_a,
                        mean_lineage_depth: 10.0,
                        max_lineage_depth: 15,
                        extinct_at: None,
                        novelty_coverage: None,
                    },
                );
                per_family.insert(
                    fam_b.as_str().to_string(),
                    FamilyOutcome {
                        survival_share: share_b,
                        biomass_share: share_b,
                        mean_lineage_depth: 5.0,
                        max_lineage_depth: 8,
                        extinct_at: None,
                        novelty_coverage: None,
                    },
                );
                outcomes.push(MatchOutcome {
                    match_id: MatchId(seed * 10 + ord as u64),
                    seed,
                    ticks_run: 1000,
                    spawn_order_index: ord,
                    spawn_order: if ord == 0 {
                        vec![fam_a, fam_b]
                    } else {
                        vec![fam_b, fam_a]
                    },
                    per_family,
                    warnings: Vec::new(),
                });
            }
        }

        let options = RatingOptions {
            bootstrap_replicates: 500,
            ci_seed: 0x1234_5678,
            ..RatingOptions::default()
        };

        let res1 = rate_tournament(&outcomes, &options).expect("first rate");
        let res2 = rate_tournament(&outcomes, &options).expect("second rate");

        assert_eq!(
            res1, res2,
            "rate_tournament must be bit-identical across runs"
        );
    }

    #[test]
    fn test_clustered_vs_naive_bootstrap_honesty() {
        // Statistical honesty test: on synthetic data with a known seed-level (cluster) effect,
        // the clustered bootstrap CI must be strictly wider than the naive match-level bootstrap CI.
        let fam_a = BrainKind::new("arm-a");
        let fam_b = BrainKind::new("arm-b");
        let families = vec![fam_a, fam_b];

        let n_seeds = 12;
        let matches_per_seed = 6;
        let mut matches = Vec::new();

        for s in 0..n_seeds {
            let seed = (s + 1) as u64;
            // Seed bias: strong intra-cluster correlation
            let seed_bias = (s as f64 - 5.5) * 0.15;
            for m_idx in 0..matches_per_seed {
                let share_a = (0.50 + seed_bias).clamp(0.05, 0.95);
                let share_b = 1.0 - share_a;
                let mut per_family = BTreeMap::new();
                per_family.insert(
                    fam_a.as_str().to_string(),
                    FamilyOutcome {
                        survival_share: share_a,
                        biomass_share: share_a,
                        mean_lineage_depth: 10.0,
                        max_lineage_depth: 10,
                        extinct_at: None,
                        novelty_coverage: None,
                    },
                );
                per_family.insert(
                    fam_b.as_str().to_string(),
                    FamilyOutcome {
                        survival_share: share_b,
                        biomass_share: share_b,
                        mean_lineage_depth: 10.0,
                        max_lineage_depth: 10,
                        extinct_at: None,
                        novelty_coverage: None,
                    },
                );
                matches.push(MatchOutcome {
                    match_id: MatchId(seed * 100 + m_idx as u64),
                    seed,
                    ticks_run: 500,
                    spawn_order_index: (m_idx % 2) as u32,
                    spawn_order: vec![fam_a, fam_b],
                    per_family,
                    warnings: Vec::new(),
                });
            }
        }

        let replicates = 1000;
        let ci_seed = 0xCAFE_BABE;

        // (1) Clustered bootstrap
        let options_clustered = RatingOptions {
            bootstrap_replicates: replicates,
            ci_seed,
            ..RatingOptions::default()
        };
        let rated_clustered =
            rate_tournament(&matches, &options_clustered).expect("rated clustered");
        let axis_c = rated_clustered
            .axis(RatingAxis::SurvivalShare)
            .expect("survival axis");
        let pair_c = &axis_c.pairwise[0];
        let clustered_ci_width = pair_c.ci95.1 - pair_c.ci95.0;

        // (2) Naive bootstrap (resampling matches directly)
        let mut rng = SmallRng::seed_from_u64(ci_seed ^ 0x5EED_0001);
        let mut naive_diffs = Vec::with_capacity(replicates);
        for _ in 0..replicates {
            let mut sample_matches = Vec::with_capacity(matches.len());
            for _ in 0..matches.len() {
                let idx = rng.random_range(0..matches.len());
                sample_matches.push(&matches[idx]);
            }
            if let Ok(res_pairwise) =
                reduce_pairwise_for_matches(&sample_matches, &families, RatingAxis::SurvivalShare)
                && let Ok(fit) =
                    fit_bradley_terry(&families, &res_pairwise, BtPrior::default(), 1e-10, 10_000)
                && fit.converged
            {
                naive_diffs.push(fit.theta[&fam_a] - fit.theta[&fam_b]);
            }
        }
        naive_diffs.sort_by(|a, b| a.total_cmp(b));
        let low_idx = (0.025 * naive_diffs.len() as f64).floor() as usize;
        let high_idx = (0.975 * naive_diffs.len() as f64).floor() as usize;
        let naive_ci_width = naive_diffs[high_idx] - naive_diffs[low_idx];

        assert!(
            clustered_ci_width > naive_ci_width,
            "clustered bootstrap CI width ({clustered_ci_width:.4}) must be strictly wider than naive CI width ({naive_ci_width:.4})"
        );
    }

    #[test]
    fn test_false_discovery_control_null_tournament() {
        // Negative control: identically matched arms over multiple replications
        // must not declare a false positive winner rate above alpha (within binomial tolerance).
        let fam_a = BrainKind::new("arm-a");
        let fam_b = BrainKind::new("arm-b");

        let total_tournaments = 20;
        let mut false_positives = 0;

        for rep in 0..total_tournaments {
            let mut matches = Vec::new();
            let mut match_rng = SmallRng::seed_from_u64(rep as u64 * 100 + 42);

            for seed in 1..=16 {
                for ord in 0..2 {
                    // Coin flip for equal arms: 50% probability
                    let a_wins = match_rng.random_bool(0.5);
                    let (share_a, share_b) = if a_wins { (0.60, 0.40) } else { (0.40, 0.60) };
                    let mut per_family = BTreeMap::new();
                    per_family.insert(
                        fam_a.as_str().to_string(),
                        FamilyOutcome {
                            survival_share: share_a,
                            biomass_share: share_a,
                            mean_lineage_depth: 10.0,
                            max_lineage_depth: 10,
                            extinct_at: None,
                            novelty_coverage: None,
                        },
                    );
                    per_family.insert(
                        fam_b.as_str().to_string(),
                        FamilyOutcome {
                            survival_share: share_b,
                            biomass_share: share_b,
                            mean_lineage_depth: 10.0,
                            max_lineage_depth: 10,
                            extinct_at: None,
                            novelty_coverage: None,
                        },
                    );
                    matches.push(MatchOutcome {
                        match_id: MatchId(seed * 10 + ord as u64),
                        seed,
                        ticks_run: 500,
                        spawn_order_index: ord,
                        spawn_order: vec![fam_a, fam_b],
                        per_family,
                        warnings: Vec::new(),
                    });
                }
            }

            let options = RatingOptions {
                bootstrap_replicates: 500,
                ci_seed: rep as u64 * 777 + 1,
                ..RatingOptions::default()
            };

            let rated = rate_tournament(&matches, &options).expect("rated");
            if let Some(axis) = rated.axis(RatingAxis::SurvivalShare)
                && axis.pairwise[0].verdict != PairwiseVerdict::Parity
            {
                false_positives += 1;
            }
        }

        let fp_rate = false_positives as f64 / total_tournaments as f64;
        assert!(
            fp_rate <= 0.20,
            "false positive rate {fp_rate:.3} ({false_positives}/{total_tournaments}) exceeded nominal alpha"
        );
    }

    #[test]
    fn test_degenerate_inputs_handled_with_typed_outcomes() {
        let fam_a = BrainKind::new("fam-a");
        let fam_b = BrainKind::new("fam-b");

        // 1. Empty matches
        let err_empty = rate_tournament(&[], &RatingOptions::default()).unwrap_err();
        assert_eq!(err_empty, RatingError::EmptyMatches);

        // 2. Single family
        let mut single_family_matches = Vec::new();
        let mut pf = BTreeMap::new();
        pf.insert(
            fam_a.as_str().to_string(),
            FamilyOutcome {
                survival_share: 1.0,
                biomass_share: 1.0,
                mean_lineage_depth: 5.0,
                max_lineage_depth: 5,
                extinct_at: None,
                novelty_coverage: None,
            },
        );
        single_family_matches.push(MatchOutcome {
            match_id: MatchId(1),
            seed: 1,
            ticks_run: 500,
            spawn_order_index: 0,
            spawn_order: vec![fam_a],
            per_family: pf,
            warnings: Vec::new(),
        });
        let err_single =
            rate_tournament(&single_family_matches, &RatingOptions::default()).unwrap_err();
        assert!(matches!(err_single, RatingError::TooFewFamilies { .. }));

        // 3. All draws on survival share
        let mut draw_matches = Vec::new();
        for seed in 1..=4 {
            let mut pf = BTreeMap::new();
            pf.insert(
                fam_a.as_str().to_string(),
                FamilyOutcome {
                    survival_share: 0.5,
                    biomass_share: 0.5,
                    mean_lineage_depth: 5.0,
                    max_lineage_depth: 5,
                    extinct_at: None,
                    novelty_coverage: None,
                },
            );
            pf.insert(
                fam_b.as_str().to_string(),
                FamilyOutcome {
                    survival_share: 0.5,
                    biomass_share: 0.5,
                    mean_lineage_depth: 5.0,
                    max_lineage_depth: 5,
                    extinct_at: None,
                    novelty_coverage: None,
                },
            );
            draw_matches.push(MatchOutcome {
                match_id: MatchId(seed),
                seed,
                ticks_run: 500,
                spawn_order_index: 0,
                spawn_order: vec![fam_a, fam_b],
                per_family: pf,
                warnings: Vec::new(),
            });
        }
        let rated_draws =
            rate_tournament(&draw_matches, &RatingOptions::default()).expect("rated draws");
        assert!(matches!(
            rated_draws.axes[&RatingAxis::SurvivalShare],
            AxisRatingOutcome::Undetermined { .. }
        ));

        // 4. Non-finite values
        let mut nan_matches = Vec::new();
        let mut pf_nan = BTreeMap::new();
        pf_nan.insert(
            fam_a.as_str().to_string(),
            FamilyOutcome {
                survival_share: f64::NAN,
                biomass_share: 0.5,
                mean_lineage_depth: 5.0,
                max_lineage_depth: 5,
                extinct_at: None,
                novelty_coverage: None,
            },
        );
        pf_nan.insert(
            fam_b.as_str().to_string(),
            FamilyOutcome {
                survival_share: 0.5,
                biomass_share: 0.5,
                mean_lineage_depth: 5.0,
                max_lineage_depth: 5,
                extinct_at: None,
                novelty_coverage: None,
            },
        );
        nan_matches.push(MatchOutcome {
            match_id: MatchId(1),
            seed: 1,
            ticks_run: 500,
            spawn_order_index: 0,
            spawn_order: vec![fam_a, fam_b],
            per_family: pf_nan,
            warnings: Vec::new(),
        });
        let err_nan = rate_tournament(&nan_matches, &RatingOptions::default()).unwrap_err();
        assert!(matches!(err_nan, RatingError::NonFiniteValue { .. }));
    }

    #[test]
    fn test_order_effect_detection() {
        let fam_a = BrainKind::new("fam-a");
        let fam_b = BrainKind::new("fam-b");

        // Skewed position performance: position 0 always gets 0.90, position 1 gets 0.10
        let mut skewed_matches = Vec::new();
        for seed in 1..=10 {
            for ord in 0..2 {
                let (first, second) = if ord == 0 {
                    (fam_a, fam_b)
                } else {
                    (fam_b, fam_a)
                };
                let mut pf = BTreeMap::new();
                pf.insert(
                    first.as_str().to_string(),
                    FamilyOutcome {
                        survival_share: 0.90,
                        biomass_share: 0.90,
                        mean_lineage_depth: 10.0,
                        max_lineage_depth: 10,
                        extinct_at: None,
                        novelty_coverage: None,
                    },
                );
                pf.insert(
                    second.as_str().to_string(),
                    FamilyOutcome {
                        survival_share: 0.10,
                        biomass_share: 0.10,
                        mean_lineage_depth: 1.0,
                        max_lineage_depth: 1,
                        extinct_at: None,
                        novelty_coverage: None,
                    },
                );
                skewed_matches.push(MatchOutcome {
                    match_id: MatchId(seed * 10 + ord as u64),
                    seed,
                    ticks_run: 500,
                    spawn_order_index: ord,
                    spawn_order: vec![first, second],
                    per_family: pf,
                    warnings: Vec::new(),
                });
            }
        }

        let report = analyze_order_effect(&skewed_matches, RatingAxis::SurvivalShare, 0x1234, 500);
        assert!(
            report.detected,
            "strong position advantage must be detected as an order effect"
        );
        assert!(report.position_means[&0] > 0.80);
        assert!(report.position_means[&1] < 0.20);

        // Uniform performance: position 0 gets 0.50, position 1 gets 0.50
        let mut uniform_matches = Vec::new();
        for seed in 1..=10 {
            let mut pf = BTreeMap::new();
            pf.insert(
                fam_a.as_str().to_string(),
                FamilyOutcome {
                    survival_share: 0.50,
                    biomass_share: 0.50,
                    mean_lineage_depth: 5.0,
                    max_lineage_depth: 5,
                    extinct_at: None,
                    novelty_coverage: None,
                },
            );
            pf.insert(
                fam_b.as_str().to_string(),
                FamilyOutcome {
                    survival_share: 0.50,
                    biomass_share: 0.50,
                    mean_lineage_depth: 5.0,
                    max_lineage_depth: 5,
                    extinct_at: None,
                    novelty_coverage: None,
                },
            );
            uniform_matches.push(MatchOutcome {
                match_id: MatchId(seed),
                seed,
                ticks_run: 500,
                spawn_order_index: 0,
                spawn_order: vec![fam_a, fam_b],
                per_family: pf,
                warnings: Vec::new(),
            });
        }
        let report_unif =
            analyze_order_effect(&uniform_matches, RatingAxis::SurvivalShare, 0x5678, 500);
        assert!(
            !report_unif.detected,
            "equal position outcomes must not detect order effect"
        );
    }

    #[test]
    fn test_elo_to_theta_round_trip() {
        for original_theta in [-3.5, -1.0, 0.0, 0.5, 2.7, 4.0] {
            let elo = theta_to_elo(original_theta);
            let back = elo_to_theta(elo);
            assert!(
                (original_theta - back).abs() < 1e-12,
                "round-trip drift for theta {original_theta}"
            );
        }
    }

    #[test]
    fn test_rating_table_markdown_generation() {
        let fam_a = BrainKind::new("mlp");
        let fam_b = BrainKind::new("dwraon");
        let mut matches = Vec::new();

        for seed in 1..=8 {
            let mut pf = BTreeMap::new();
            pf.insert(
                fam_a.as_str().to_string(),
                FamilyOutcome {
                    survival_share: 0.70,
                    biomass_share: 0.65,
                    mean_lineage_depth: 12.0,
                    max_lineage_depth: 15,
                    extinct_at: None,
                    novelty_coverage: None,
                },
            );
            pf.insert(
                fam_b.as_str().to_string(),
                FamilyOutcome {
                    survival_share: 0.30,
                    biomass_share: 0.35,
                    mean_lineage_depth: 6.0,
                    max_lineage_depth: 8,
                    extinct_at: None,
                    novelty_coverage: None,
                },
            );
            matches.push(MatchOutcome {
                match_id: MatchId(seed),
                seed,
                ticks_run: 1000,
                spawn_order_index: 0,
                spawn_order: vec![fam_a, fam_b],
                per_family: pf,
                warnings: Vec::new(),
            });
        }

        let options = RatingOptions {
            bootstrap_replicates: 200,
            ci_seed: 42,
            ..RatingOptions::default()
        };

        let table = rate_tournament(&matches, &options).expect("rate tournament");
        let md = table.generate_markdown();

        assert!(md.contains("# ScriptBots Multi-Axis Brain Family Tournament Leaderboard"));
        assert!(md.contains("## Axis: survival_share (`survival_share`)"));
        assert!(md.contains("mlp"));
        assert!(md.contains("dwraon"));
        assert!(md.contains("Pairwise Head-to-Head Comparisons"));
    }

    #[test]
    fn test_leaderboard_row_exclusion_filter() {
        use leaderboard::{ExclusionReason, TournamentResultRow, filter_eligible_rows};

        let valid_row = TournamentResultRow {
            run_id: "run-valid".to_string(),
            match_id: 1,
            seed: 101,
            world_seed: 101,
            brain_seed: 101,
            family: "mlp".to_string(),
            spawn_order_index: 0,
            survival_share: 0.6,
            biomass_share: 0.5,
            mean_lineage_depth: 4.0,
            max_lineage_depth: 5,
            extinct_at: None,
            novelty_coverage: None,
            agents_final: 10,
            config_digest: "cfg-123".to_string(),
            manifest_digest: "mnf-123".to_string(),
            sense_backend: "cpu_simd".to_string(),
            sense_determinism: "exact".to_string(),
            reproducible: true,
            warnings: Vec::new(),
        };

        let mut irreproducible_row = valid_row.clone();
        irreproducible_row.run_id = "run-not-reproducible".to_string();
        irreproducible_row.reproducible = false;

        let mut approx_row = valid_row.clone();
        approx_row.run_id = "run-approx".to_string();
        approx_row.sense_determinism = "approximate".to_string();

        let mut stale_cfg_row = valid_row.clone();
        stale_cfg_row.run_id = "run-stale".to_string();
        stale_cfg_row.config_digest = "cfg-old".to_string();

        let rows = vec![valid_row, irreproducible_row, approx_row, stale_cfg_row];
        let (eligible, excluded) = filter_eligible_rows(&rows, "cfg-123");

        assert_eq!(eligible.len(), 1);
        assert_eq!(eligible[0].run_id, "run-valid");
        assert_eq!(excluded.len(), 3);
        assert!(matches!(
            excluded[0].reason,
            ExclusionReason::NotReproducible
        ));
        assert!(matches!(
            excluded[1].reason,
            ExclusionReason::ApproximateSense(_)
        ));
        assert!(matches!(
            excluded[2].reason,
            ExclusionReason::ConfigDigestMismatch { .. }
        ));
    }

    #[test]
    fn test_leaderboard_drift_check() {
        use leaderboard::{LeaderboardCheckError, check_leaderboard_drift};
        use std::path::Path;

        let doc1 = "# Leaderboard\n- Effective Config Digest: `cfg-abc`\n\nSome results\n";
        let doc2 = "# Leaderboard\n- Effective Config Digest: `cfg-abc`\n\nSome results\n";
        let doc_drift = "# Leaderboard\n- Effective Config Digest: `cfg-abc`\n\nAltered results\n";
        let doc_cfg_drift = "# Leaderboard\n- Effective Config Digest: `cfg-xyz`\n\nSome results\n";

        let dummy_path = Path::new("docs/leaderboard.md");

        // Identical documents pass
        assert!(check_leaderboard_drift(doc1, doc2, dummy_path).is_ok());

        // Text change triggers DocumentDrift
        let err_drift = check_leaderboard_drift(doc1, doc_drift, dummy_path).unwrap_err();
        assert!(matches!(
            err_drift,
            LeaderboardCheckError::DocumentDrift { .. }
        ));

        // Config change triggers ConfigDrift
        let err_cfg = check_leaderboard_drift(doc1, doc_cfg_drift, dummy_path).unwrap_err();
        assert!(matches!(err_cfg, LeaderboardCheckError::ConfigDrift { .. }));
    }

    #[test]
    fn test_spec_file_parsing_and_smoke_conversion() {
        use leaderboard::TournamentSpecFile;

        let spec_toml = include_str!("../../../tournament/spec.toml");
        let spec_file = TournamentSpecFile::from_toml_str(spec_toml).expect("parse spec.toml");

        assert_eq!(
            spec_file.tournament.title,
            "ScriptBots Canonical Brain-Family Tournament"
        );
        assert_eq!(
            spec_file.tournament.families,
            vec!["mlp", "dwraon", "assembly"]
        );
        assert_eq!(spec_file.tournament.seeds.len(), 32);

        let smoke_spec = spec_file.to_smoke_spec().expect("smoke spec");
        assert_eq!(smoke_spec.families.len(), 2);
        assert_eq!(smoke_spec.seeds.len(), 4);
        assert_eq!(smoke_spec.ticks, 200);
    }

    #[test]
    fn test_spec_cli_overrides() {
        use leaderboard::TournamentSpecFile;

        let spec_toml = include_str!("../../../tournament/spec.toml");
        let mut spec_file = TournamentSpecFile::from_toml_str(spec_toml).expect("parse spec.toml");

        // 1. Valid overrides
        spec_file
            .apply_cli_overrides(Some("mlp, assembly"), Some("4"), Some(500), Some(32))
            .expect("apply overrides");

        assert_eq!(spec_file.tournament.families, vec!["mlp", "assembly"]);
        assert_eq!(spec_file.tournament.seeds.len(), 4);
        assert_eq!(spec_file.tournament.ticks, 500);
        assert_eq!(spec_file.tournament.cohort_size, 32);

        // Explicit seeds list
        spec_file
            .apply_cli_overrides(None, Some("201, 202, 203"), None, None)
            .expect("apply seeds list override");
        assert_eq!(spec_file.tournament.seeds, vec![201, 202, 203]);

        // 2. Invalid overrides fail with typed error
        assert!(
            spec_file
                .apply_cli_overrides(None, Some("0"), None, None)
                .is_err()
        );
        assert!(
            spec_file
                .apply_cli_overrides(None, None, Some(0), None)
                .is_err()
        );
        assert!(
            spec_file
                .apply_cli_overrides(None, None, None, Some(0))
                .is_err()
        );
    }

    #[test]
    fn test_save_tournament_artifacts() {
        use leaderboard::{
            LeaderboardPipelineOutcome, TournamentSpecFile, save_tournament_artifacts,
        };

        let temp_dir =
            std::env::temp_dir().join(format!("scriptbots-test-artifacts-{}", std::process::id()));

        let spec_toml = include_str!("../../../tournament/spec.toml");
        let spec_file = TournamentSpecFile::from_toml_str(spec_toml).expect("parse spec.toml");

        let outcome = LeaderboardPipelineOutcome {
            spec: spec_file,
            config_digest: "cfg-test".to_string(),
            spec_digest: "spec-test".to_string(),
            total_matches: 4,
            total_rows: 8,
            eligible_rows: 8,
            excluded_rows: 0,
            results_jsonl: "{\"match_id\":1}\n".to_string(),
            ratings_json: "{\"axes\":{}}\n".to_string(),
            leaderboard_md: "# Test Leaderboard\n".to_string(),
            results_digest: "digest-results".to_string(),
            ratings_digest: "digest-ratings".to_string(),
        };

        // Directory target
        save_tournament_artifacts(&outcome, &temp_dir, None).expect("save to directory");
        assert!(temp_dir.join("tournament_results.jsonl").exists());
        assert!(temp_dir.join("ratings.json").exists());
        assert!(temp_dir.join("leaderboard.md").exists());

        // File target
        let file_target = temp_dir.join("custom").join("my_leaderboard.md");
        save_tournament_artifacts(&outcome, &file_target, None).expect("save to md file");
        assert!(file_target.exists());
        assert!(
            temp_dir
                .join("custom")
                .join("tournament_results.jsonl")
                .exists()
        );
        assert!(temp_dir.join("custom").join("ratings.json").exists());
    }

    #[test]
    fn test_execute_leaderboard_tournament_smoke() {
        use crate::BuildProvenanceV0;
        use leaderboard::{
            LeaderboardCheckError, TournamentSpecFile, check_leaderboard_drift,
            execute_leaderboard_tournament,
        };
        use scriptbots_core::ScriptBotsConfig;

        let spec_toml = include_str!("../../../tournament/spec.toml");
        let mut spec_file = TournamentSpecFile::from_toml_str(spec_toml).expect("parse spec.toml");

        // Use a 2-seed x 50-tick smoke configuration for rapid unit test
        if let Some(ref mut sm) = spec_file.smoke {
            sm.seeds = vec![101, 102];
            sm.ticks = 50;
            sm.cohort_size = 8;
        }

        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
        let base_config = ScriptBotsConfig::default();
        let build = BuildProvenanceV0::current();

        let outcome = execute_leaderboard_tournament(
            &spec_file,
            None,
            true, // is_smoke
            &base_config,
            &build,
            None, // out_dir
            None, // check_target
            true, // check_reproducibility
            2,    // jobs
        )
        .expect("smoke leaderboard execution");

        assert_eq!(outcome.total_matches, 4); // 2 seeds x 2 orders
        assert_eq!(outcome.eligible_rows, 8); // 4 matches x 2 families
        assert_eq!(outcome.excluded_rows, 0);
        assert!(!outcome.results_digest.is_empty());
        assert!(!outcome.ratings_digest.is_empty());
        assert!(
            outcome
                .leaderboard_md
                .contains("# ScriptBots Canonical Brain-Family Tournament")
        );
        assert!(
            outcome
                .leaderboard_md
                .contains("## Multi-Axis Ratings & Confidence Intervals")
        );
        assert!(
            outcome
                .leaderboard_md
                .contains("Pairwise Head-to-Head Comparisons")
        );
        assert!(
            outcome
                .leaderboard_md
                .contains("Spawn-Order Position Performance")
        );
        assert!(outcome.leaderboard_md.contains("## Exclusion Audit"));
        assert!(
            outcome
                .leaderboard_md
                .contains("## Run & Match Provenance Footnotes")
        );

        // Check self-consistency (--check against generated document passes)
        let dummy_path = std::path::Path::new("docs/leaderboard.md");
        assert!(
            check_leaderboard_drift(&outcome.leaderboard_md, &outcome.leaderboard_md, dummy_path)
                .is_ok()
        );

        // Negative check 1: mutated row in document fails with DocumentDrift
        let mutated_doc = outcome.leaderboard_md.replace("1500.00", "1599.99");
        if mutated_doc != outcome.leaderboard_md {
            let drift_err =
                check_leaderboard_drift(&outcome.leaderboard_md, &mutated_doc, dummy_path)
                    .unwrap_err();
            assert!(matches!(
                drift_err,
                LeaderboardCheckError::DocumentDrift { .. }
            ));
        }

        // Negative check 2: stale config digest fails with ConfigDrift
        let stale_cfg_doc = outcome
            .leaderboard_md
            .replace(&outcome.config_digest, "stale_config_digest_12345");
        let cfg_err = check_leaderboard_drift(&stale_cfg_doc, &outcome.leaderboard_md, dummy_path)
            .unwrap_err();
        assert!(matches!(cfg_err, LeaderboardCheckError::ConfigDrift { .. }));
    }

    #[test]
    fn test_execute_leaderboard_tournament_three_families_barrier() {
        use crate::BuildProvenanceV0;
        use leaderboard::{TournamentSpecFile, execute_leaderboard_tournament};
        use scriptbots_core::ScriptBotsConfig;

        let spec_toml = include_str!("../../../tournament/spec.toml");
        let mut spec_file = TournamentSpecFile::from_toml_str(spec_toml).expect("parse spec.toml");

        // 3 families with Latin square and 300 ticks to verify cross-kind barrier
        spec_file.tournament.families = vec!["mlp".into(), "dwraon".into(), "assembly".into()];
        spec_file.tournament.seeds = vec![101];
        spec_file.tournament.ticks = 300;
        spec_file.tournament.cohort_size = 12;
        spec_file.tournament.order_policy = "balanced_latin_square".into();

        let base_config = ScriptBotsConfig::default();
        let build = BuildProvenanceV0::current();

        let outcome = execute_leaderboard_tournament(
            &spec_file,
            None,
            false,
            &base_config,
            &build,
            None,
            None,
            true,
            1,
        )
        .expect("3-family tournament with reproduction barrier must succeed");

        assert_eq!(outcome.total_matches, 3);
        assert_eq!(outcome.eligible_rows, 9);
        assert_eq!(outcome.excluded_rows, 0);
    }
}
