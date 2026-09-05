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
    pub ticks_run: u64,
    /// Assignment index retained for downstream order-effect analysis.
    pub spawn_order_index: u32,
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
            ticks_run,
            spawn_order_index: plan.spawn_order_index,
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
