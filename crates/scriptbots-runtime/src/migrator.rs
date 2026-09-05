//! Deterministic Migrator for multi-island archipelagos (`bd-16g.5.2`).
//!
//! Plans migration barriers across islands through two-phase selection.
//! Checks projected population conservation and uses deterministic total ordering and total
//! tie-breaking (no `partial_cmp().unwrap()`, no `HashMap` iteration).

use rand::Rng;
use scriptbots_core::{SmallRngStream, rng_domains::derive_migration_seed};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub use scriptbots_core::rng_domains::IslandId;

/// Emigrant selection strategy for migration barriers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmigrantSelectionRule {
    /// Select agents at random using a dedicated seeded PRNG stream.
    Random,
    /// Select agents with highest energy (tie-break by `AgentUid` ASC).
    Fittest,
    /// Select youngest agents by age ASC (tie-break by `AgentUid` ASC).
    Youngest,
    /// Select agents with highest realized speed (tie-break by `AgentUid` ASC).
    Wanderer,
}

/// Archipelago migration topology configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationTopology {
    /// Islands connected in a ring (0 -> 1 -> ... -> N-1 -> 0).
    Ring,
    /// Fully connected directed graph (every island connected to every other).
    FullyConnected,
    /// Explicit list of directed edges `(from_island, to_island)`.
    Custom(Vec<(IslandId, IslandId)>),
}

impl MigrationTopology {
    /// Returns sorted, deduped list of directed edges for `island_ids`.
    #[must_use]
    pub fn build_edges(&self, island_ids: &[IslandId]) -> Vec<(IslandId, IslandId)> {
        let mut sorted_ids = island_ids.to_vec();
        sorted_ids.sort_unstable();
        sorted_ids.dedup();

        let n = sorted_ids.len();
        if n < 2 {
            return Vec::new();
        }

        let mut edges = Vec::new();
        match self {
            Self::Ring => {
                for i in 0..n {
                    let from = sorted_ids[i];
                    let to = sorted_ids[(i + 1) % n];
                    if from != to {
                        edges.push((from, to));
                    }
                }
            }
            Self::FullyConnected => {
                for &from in &sorted_ids {
                    for &to in &sorted_ids {
                        if from != to {
                            edges.push((from, to));
                        }
                    }
                }
            }
            Self::Custom(custom) => {
                for &(from, to) in custom {
                    if from != to && sorted_ids.contains(&from) && sorted_ids.contains(&to) {
                        edges.push((from, to));
                    }
                }
            }
        }

        edges.sort_unstable();
        edges.dedup();
        edges
    }
}

/// Configuration for deterministic migration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationConfig {
    /// Interval in ticks between migration barriers (0 = disabled).
    pub interval_ticks: u64,
    /// Number of emigrants per directed edge per barrier.
    pub emigrants_per_edge: usize,
    /// Emigrant selection rule.
    pub selection_rule: EmigrantSelectionRule,
    /// Topology of island migration edges.
    pub topology: MigrationTopology,
    /// Conservation guarantee flag (must always be true).
    pub replace: bool,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            interval_ticks: 500,
            emigrants_per_edge: 2,
            selection_rule: EmigrantSelectionRule::Fittest,
            topology: MigrationTopology::Ring,
            replace: true,
        }
    }
}

/// One selected organism's planned destination and ranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmigrantRecord {
    /// Island from which the candidate was selected.
    pub from_island: IslandId,
    /// Island to which the candidate is assigned by the migration plan.
    pub to_island: IslandId,
    /// Candidate identity, scoped to its source island.
    pub agent_uid: u64,
    /// Rule used to rank this candidate on its source island.
    pub selection_rule: EmigrantSelectionRule,
    /// Zero-based position in the source island's complete candidate ranking.
    pub rank: usize,
    /// Reporting projection of energy, age, speed, or the random rank under the selected rule.
    /// Random ranks are rounded to `f64` here; selection orders their original `u64` values.
    pub key_value: f64,
}

/// Selection plan and projected population counts for one migration barrier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BarrierReport {
    /// Barrier index supplied to [`select_emigrants`], despite the historical field name.
    pub tick: u64,
    /// Rule applied to all candidate rankings in this plan.
    pub selection_rule: EmigrantSelectionRule,
    /// Planned moves in canonical destination, source, and agent identity order.
    pub moves: Vec<EmigrantRecord>,
    /// Candidate population of each island before applying the planned moves.
    pub pop_before: BTreeMap<IslandId, u32>,
    /// Projected population of each island after applying the planned moves.
    pub pop_after: BTreeMap<IslandId, u32>,
    /// Whether the plan's projected total population equals its initial total.
    pub conserved: bool,
}

/// Errors occurring during migration processing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum MigrationError {
    /// Planned population updates changed the archipelago's total population.
    #[error("population not conserved across migration barrier: before={before}, after={after}")]
    PopulationNotConserved {
        /// Total population before the planned moves.
        before: u64,
        /// Total projected population after the planned moves.
        after: u64,
        /// Number of moves in the rejected plan.
        moves_count: usize,
    },
    /// A planned migration edge has a missing endpoint or refers to the same island twice.
    #[error("invalid edge in topology: from={from}, to={to}")]
    InvalidEdge {
        /// Source island of the invalid edge.
        from: IslandId,
        /// Destination island of the invalid edge.
        to: IslandId,
    },
    /// No islands were supplied for migration selection.
    #[error("empty archipelago passed to migrator")]
    EmptyArchipelago,
    /// An island's candidate or projected population cannot fit the report's `u32` field.
    #[error("migration population count exceeds u32 capacity for island {island}")]
    IslandPopulationOverflow {
        /// Island whose population could not be represented.
        island: IslandId,
    },
    /// A planned emigration would remove an agent from a zero-population island.
    #[error("migration would subtract an agent from empty island {island}")]
    IslandPopulationUnderflow {
        /// Source island whose projected population was already zero.
        island: IslandId,
    },
}

/// Snapshot of a single agent's state for migration selection.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateAgent {
    /// Agent identity, scoped to its containing island.
    pub uid: u64,
    /// Energy used by [`EmigrantSelectionRule::Fittest`].
    pub energy: f32,
    /// Health retained in the candidate snapshot; current selection rules do not read it.
    pub health: f32,
    /// Age in simulation ticks, used by [`EmigrantSelectionRule::Youngest`].
    pub age: u32,
    /// Realized speed used by [`EmigrantSelectionRule::Wanderer`].
    pub speed: f32,
}

/// Compares two candidate agents deterministically under `rule`.
#[must_use]
pub fn compare_candidates(
    a: &CandidateAgent,
    b: &CandidateAgent,
    rule: EmigrantSelectionRule,
    random_rank_a: u64,
    random_rank_b: u64,
) -> Ordering {
    match rule {
        EmigrantSelectionRule::Fittest => b
            .energy
            .total_cmp(&a.energy)
            .then_with(|| a.uid.cmp(&b.uid)),
        EmigrantSelectionRule::Youngest => a.age.cmp(&b.age).then_with(|| a.uid.cmp(&b.uid)),
        EmigrantSelectionRule::Wanderer => {
            b.speed.total_cmp(&a.speed).then_with(|| a.uid.cmp(&b.uid))
        }
        EmigrantSelectionRule::Random => random_rank_a
            .cmp(&random_rank_b)
            .then_with(|| a.uid.cmp(&b.uid)),
    }
}

pub(crate) fn checked_island_population(
    island: IslandId,
    population: usize,
) -> Result<u32, MigrationError> {
    u32::try_from(population).map_err(|_| MigrationError::IslandPopulationOverflow { island })
}

pub(crate) fn total_population(populations: &BTreeMap<IslandId, u32>) -> u64 {
    // Unique IslandId(u32) keys permit at most u32::MAX + 1 islands, each with
    // a u32 population. Their aggregate bound is therefore below u64::MAX.
    populations
        .values()
        .map(|&population| u64::from(population))
        .sum()
}

fn apply_population_move(
    populations: &mut BTreeMap<IslandId, u32>,
    from: IslandId,
    to: IslandId,
) -> Result<(), MigrationError> {
    if from == to {
        return Err(MigrationError::InvalidEdge { from, to });
    }
    let source = populations
        .get(&from)
        .ok_or(MigrationError::InvalidEdge { from, to })?;
    let destination = populations
        .get(&to)
        .ok_or(MigrationError::InvalidEdge { from, to })?;
    let remaining = source
        .checked_sub(1)
        .ok_or(MigrationError::IslandPopulationUnderflow { island: from })?;
    let arriving = destination
        .checked_add(1)
        .ok_or(MigrationError::IslandPopulationOverflow { island: to })?;
    // Validate both counts before changing either endpoint of this planned move.
    populations.insert(from, remaining);
    populations.insert(to, arriving);
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "key_value is a serialized f64 reporting projection; candidate ordering uses the original u64 rank, and changing this cast would change existing report bytes"
)]
const fn random_rank_report_value(rank: u64) -> f64 {
    rank as f64
}

/// Selects emigrants across `islands` according to two-phase deterministic rules.
///
/// Returns a list of `EmigrantRecord` moves and the updated population counts.
/// # Migration RNG domain
///
/// The migration stream is ITS OWN domain, derived from `(master_seed, barrier_index)`
/// through [`derive_migration_seed`]. bd-16g.5.3 requires exactly this and explains why:
/// if emigrant choice were drawn from an island's generator, the selection on the
/// island-4-to-island-5 edge would depend on how many agents happened to reproduce on
/// island 0 that epoch -- coupling the islands through the one path nobody inspects.
///
/// It previously seeded with `seed.wrapping_add(tick)`. That is derivation by ADDITION,
/// which bd-16g.5.3 forbids by name: generators seeded with adjacent integers can emit
/// correlated early output, so consecutive barriers could select correlated emigrants
/// while every determinism gate stayed green. The KDF diffuses instead.
pub fn select_emigrants(
    islands_candidates: &BTreeMap<IslandId, Vec<CandidateAgent>>,
    config: &MigrationConfig,
    master_seed: u64,
    barrier_index: u64,
) -> Result<BarrierReport, MigrationError> {
    if islands_candidates.is_empty() {
        return Err(MigrationError::EmptyArchipelago);
    }

    let island_ids: Vec<IslandId> = islands_candidates.keys().copied().collect();
    let edges = config.topology.build_edges(&island_ids);

    let pop_before: BTreeMap<IslandId, u32> = islands_candidates
        .iter()
        .map(|(&id, candidates)| {
            checked_island_population(id, candidates.len()).map(|population| (id, population))
        })
        .collect::<Result<_, _>>()?;

    let total_before = total_population(&pop_before);

    // Rank candidates per island
    let mut ranked_candidates: BTreeMap<IslandId, Vec<(CandidateAgent, f64)>> = BTreeMap::new();
    let mut rng = SmallRngStream::seed_from_u64(derive_migration_seed(master_seed, barrier_index));

    for (&island_id, candidates) in islands_candidates {
        let mut sorted = candidates.clone();
        let mut random_ranks: BTreeMap<u64, u64> = BTreeMap::new();
        if config.selection_rule == EmigrantSelectionRule::Random {
            for c in &sorted {
                random_ranks.insert(c.uid, rng.random::<u64>());
            }
        }

        sorted.sort_by(|a, b| {
            let r_a = random_ranks.get(&a.uid).copied().unwrap_or(0);
            let r_b = random_ranks.get(&b.uid).copied().unwrap_or(0);
            compare_candidates(a, b, config.selection_rule, r_a, r_b)
        });

        let scored: Vec<(CandidateAgent, f64)> = sorted
            .into_iter()
            .map(|c| {
                let val = match config.selection_rule {
                    EmigrantSelectionRule::Fittest => f64::from(c.energy),
                    EmigrantSelectionRule::Youngest => f64::from(c.age),
                    EmigrantSelectionRule::Wanderer => f64::from(c.speed),
                    EmigrantSelectionRule::Random => {
                        random_rank_report_value(random_ranks.get(&c.uid).copied().unwrap_or(0))
                    }
                };
                (c, val)
            })
            .collect();

        ranked_candidates.insert(island_id, scored);
    }

    // Two-Phase Selection: claim candidates without double-emigration.
    // Candidate organisms are scoped to their home island (IslandId, AgentUid),
    // so claiming must be keyed on (IslandId, u64) rather than raw UID alone (bd-16g.5.5.1).
    let mut claimed_uids: BTreeSet<(IslandId, u64)> = BTreeSet::new();
    let mut moves: Vec<EmigrantRecord> = Vec::new();

    for &(from_island, to_island) in &edges {
        if let Some(candidates) = ranked_candidates.get(&from_island) {
            let mut selected_for_edge = 0usize;
            for (rank, (candidate, key_val)) in candidates.iter().enumerate() {
                if selected_for_edge >= config.emigrants_per_edge {
                    break;
                }
                if claimed_uids.insert((from_island, candidate.uid)) {
                    moves.push(EmigrantRecord {
                        from_island,
                        to_island,
                        agent_uid: candidate.uid,
                        selection_rule: config.selection_rule,
                        rank,
                        key_value: *key_val,
                    });
                    selected_for_edge += 1;
                }
            }
        }
    }

    // Sort moves canonically: (to_island, from_island, agent_uid)
    moves.sort_by(|a, b| {
        a.to_island
            .cmp(&b.to_island)
            .then_with(|| a.from_island.cmp(&b.from_island))
            .then_with(|| a.agent_uid.cmp(&b.agent_uid))
    });

    // Compute pop_after
    let mut pop_after = pop_before.clone();
    for rec in &moves {
        apply_population_move(&mut pop_after, rec.from_island, rec.to_island)?;
    }

    let total_after = total_population(&pop_after);
    let conserved = total_before == total_after;

    if !conserved {
        return Err(MigrationError::PopulationNotConserved {
            before: total_before,
            after: total_after,
            moves_count: moves.len(),
        });
    }

    Ok(BarrierReport {
        // The report's `tick` field carries the barrier index it was produced for.
        tick: barrier_index,
        selection_rule: config.selection_rule,
        moves,
        pop_before,
        pop_after,
        conserved: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn island_population_count_preserves_zero_and_u32_max() {
        let island = IslandId(7);
        let maximum = usize::try_from(u32::MAX).expect("population maximum fits target usize");
        assert_eq!(checked_island_population(island, 0), Ok(0));
        assert_eq!(checked_island_population(island, maximum), Ok(u32::MAX));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn island_population_count_rejects_value_above_u32_max() {
        // Exercise the production conversion directly without allocating a huge candidate list.
        let island = IslandId(7);
        let maximum = usize::try_from(u32::MAX).expect("population maximum fits target usize");
        assert_eq!(
            checked_island_population(island, maximum + 1),
            Err(MigrationError::IslandPopulationOverflow { island })
        );
    }

    #[test]
    fn total_population_preserves_valid_counts_above_u32_max() {
        assert_eq!(total_population(&BTreeMap::new()), 0);
        let mut populations = BTreeMap::from([(IslandId(0), u32::MAX), (IslandId(1), 0)]);
        assert_eq!(total_population(&populations), u64::from(u32::MAX));
        populations.insert(IslandId(1), 1);
        assert_eq!(total_population(&populations), u64::from(u32::MAX) + 1);
        populations.insert(IslandId(1), u32::MAX);
        assert_eq!(total_population(&populations), 2 * u64::from(u32::MAX));
    }

    #[test]
    fn population_move_preserves_total_when_destination_reaches_u32_max() {
        let source = IslandId(0);
        let destination = IslandId(1);
        let mut populations = BTreeMap::from([(source, 1), (destination, u32::MAX - 1)]);
        let before = total_population(&populations);
        apply_population_move(&mut populations, source, destination).expect("bounded move");
        assert_eq!(
            populations,
            BTreeMap::from([(source, 0), (destination, u32::MAX)])
        );
        assert_eq!(total_population(&populations), before);
    }

    #[test]
    fn population_move_refuses_overflow_and_underflow_without_partial_updates() {
        let source = IslandId(0);
        let destination = IslandId(1);
        // These are direct arithmetic boundary cases, not claims about allocated live worlds.
        for (source_count, destination_count, expected) in [
            (
                0,
                1,
                MigrationError::IslandPopulationUnderflow { island: source },
            ),
            (
                1,
                u32::MAX,
                MigrationError::IslandPopulationOverflow {
                    island: destination,
                },
            ),
        ] {
            let mut populations =
                BTreeMap::from([(source, source_count), (destination, destination_count)]);
            let before = populations.clone();
            assert_eq!(
                apply_population_move(&mut populations, source, destination),
                Err(expected)
            );
            assert_eq!(
                populations, before,
                "a refused move must preserve both endpoint counts"
            );
        }
    }

    #[test]
    fn population_move_rejects_invalid_edges_without_changing_counts() {
        let mut populations = BTreeMap::from([(IslandId(0), 1), (IslandId(1), 1)]);
        let before = populations.clone();
        for (from, to) in [
            (IslandId(2), IslandId(1)),
            (IslandId(0), IslandId(2)),
            (IslandId(0), IslandId(0)),
        ] {
            assert_eq!(
                apply_population_move(&mut populations, from, to),
                Err(MigrationError::InvalidEdge { from, to })
            );
            assert_eq!(populations, before);
        }
    }

    #[test]
    fn random_selection_orders_exact_ranks_despite_rounded_report_values() {
        let lower_rank = 1_u64 << f64::MANTISSA_DIGITS;
        let higher_rank = lower_rank + 1;
        assert_eq!(
            random_rank_report_value(lower_rank).to_bits(),
            random_rank_report_value(higher_rank).to_bits(),
            "adjacent ranks at this magnitude share the same f64 report projection"
        );
        let lower = CandidateAgent {
            uid: 2,
            energy: 0.0,
            health: 1.0,
            age: 0,
            speed: 0.0,
        };
        let higher = CandidateAgent { uid: 1, ..lower };
        assert_eq!(
            compare_candidates(
                &lower,
                &higher,
                EmigrantSelectionRule::Random,
                lower_rank,
                higher_rank
            ),
            Ordering::Less,
            "exact random ranks must take precedence over the opposite UID ordering"
        );
        assert_eq!(
            compare_candidates(
                &higher,
                &lower,
                EmigrantSelectionRule::Random,
                higher_rank,
                lower_rank
            ),
            Ordering::Greater
        );
    }

    #[test]
    fn test_ring_topology_build_edges() {
        let topo = MigrationTopology::Ring;
        let edges = topo.build_edges(&[IslandId(0), IslandId(1), IslandId(2)]);
        assert_eq!(
            edges,
            vec![
                (IslandId(0), IslandId(1)),
                (IslandId(1), IslandId(2)),
                (IslandId(2), IslandId(0)),
            ]
        );
    }

    #[test]
    fn test_fittest_selection_with_uid_tiebreak() {
        let mut candidates = BTreeMap::new();
        candidates.insert(
            IslandId(0),
            vec![
                CandidateAgent {
                    uid: 10,
                    energy: 1.0,
                    health: 1.0,
                    age: 100,
                    speed: 0.5,
                },
                CandidateAgent {
                    uid: 5,
                    energy: 1.0,
                    health: 1.0,
                    age: 100,
                    speed: 0.5,
                },
                CandidateAgent {
                    uid: 2,
                    energy: 2.0,
                    health: 1.0,
                    age: 100,
                    speed: 0.5,
                },
            ],
        );
        candidates.insert(IslandId(1), vec![]);

        let config = MigrationConfig {
            interval_ticks: 100,
            emigrants_per_edge: 2,
            selection_rule: EmigrantSelectionRule::Fittest,
            topology: MigrationTopology::Ring,
            replace: true,
        };

        let report = select_emigrants(&candidates, &config, 42, 100).unwrap();
        assert_eq!(report.moves.len(), 2);
        // First fittest: uid 2 (energy 2.0), Second fittest (tied energy 1.0): uid 5 (lower uid)
        assert_eq!(report.moves[0].agent_uid, 2);
        assert_eq!(report.moves[1].agent_uid, 5);
        assert!(report.conserved);
    }

    #[test]
    fn test_two_phase_no_duplicate_emigration() {
        let mut candidates = BTreeMap::new();
        candidates.insert(
            IslandId(0),
            vec![CandidateAgent {
                uid: 1,
                energy: 1.0,
                health: 1.0,
                age: 10,
                speed: 0.1,
            }],
        );
        candidates.insert(IslandId(1), vec![]);
        candidates.insert(IslandId(2), vec![]);

        let config = MigrationConfig {
            interval_ticks: 100,
            emigrants_per_edge: 1,
            selection_rule: EmigrantSelectionRule::Fittest,
            topology: MigrationTopology::FullyConnected,
            replace: true,
        };

        let report = select_emigrants(&candidates, &config, 42, 100).unwrap();
        // Agent 1 should emigrate to ONE island only, not both
        assert_eq!(report.moves.len(), 1);
        assert_eq!(report.moves[0].agent_uid, 1);
        assert!(report.conserved);
    }

    #[test]
    fn test_population_conservation_property() {
        let mut candidates = BTreeMap::new();
        for id in 0..4 {
            let agents: Vec<CandidateAgent> = (0..20)
                .map(|i| CandidateAgent {
                    uid: (id * 100 + i) as u64,
                    energy: (i as f32) * 0.1,
                    health: 1.0,
                    age: i * 5,
                    speed: (i as f32) * 0.05,
                })
                .collect();
            candidates.insert(IslandId(id), agents);
        }

        let config = MigrationConfig::default();
        let report = select_emigrants(&candidates, &config, 12345, 500).unwrap();
        let total_before: u32 = report.pop_before.values().sum();
        let total_after: u32 = report.pop_after.values().sum();
        assert_eq!(total_before, total_after);
        assert!(report.conserved);
    }

    /// The migration stream must NOT be `master_seed + barrier_index`.
    ///
    /// bd-16g.5.3 forbids seed derivation by addition by name: generators seeded with
    /// adjacent integers can emit correlated early output, so consecutive barriers could
    /// pick correlated emigrants while every determinism gate stayed green. This pins the
    /// KDF against the construction it replaced, which is the only way the regression
    /// stays visible -- reverting to addition breaks nothing else.
    #[test]
    fn bd_16g_5_2_migration_seed_is_derived_not_added() {
        let master = 0x5EED_0000_5EED_0000_u64;
        for barrier in 0..8u64 {
            assert_ne!(
                derive_migration_seed(master, barrier),
                master.wrapping_add(barrier),
                "barrier {barrier} fell back to additive derivation"
            );
        }
        // Adjacent barriers must land far apart, not one apart.
        let a = derive_migration_seed(master, 4);
        let b = derive_migration_seed(master, 5);
        assert_ne!(a, b);
        assert!(
            a.abs_diff(b) > 1,
            "adjacent barrier seeds must diffuse, got {a} and {b}"
        );
    }

    /// Emigrant choice must depend on the barrier index, and only through the KDF.
    #[test]
    fn bd_16g_5_2_random_selection_changes_across_barriers() {
        let mut candidates = BTreeMap::new();
        candidates.insert(
            IslandId(0),
            (1..=12u64)
                .map(|uid| CandidateAgent {
                    uid,
                    energy: 1.0,
                    health: 1.0,
                    age: 10,
                    speed: 1.0,
                })
                .collect::<Vec<_>>(),
        );
        candidates.insert(
            IslandId(1),
            (100..=111u64)
                .map(|uid| CandidateAgent {
                    uid,
                    energy: 1.0,
                    health: 1.0,
                    age: 10,
                    speed: 1.0,
                })
                .collect::<Vec<_>>(),
        );
        let config = MigrationConfig {
            selection_rule: EmigrantSelectionRule::Random,
            emigrants_per_edge: 3,
            ..MigrationConfig::default()
        };
        let master = 0xA11C_E000_A11C_E000_u64;

        let first = select_emigrants(&candidates, &config, master, 0).expect("barrier 0");
        let second = select_emigrants(&candidates, &config, master, 1).expect("barrier 1");

        // Every candidate is identical on every axis, so ONLY the random rank can order
        // them -- if the barrier index did not reach the stream these would match.
        let uids =
            |report: &BarrierReport| report.moves.iter().map(|m| m.agent_uid).collect::<Vec<_>>();
        assert_ne!(
            uids(&first),
            uids(&second),
            "barrier index must reach the migration stream"
        );

        // And it must be reproducible: same inputs, same answer.
        let repeat = select_emigrants(&candidates, &config, master, 0).expect("barrier 0 again");
        assert_eq!(
            uids(&first),
            uids(&repeat),
            "the same barrier must select the same emigrants"
        );
    }

    /// Migration selection must not change when an island's population history differs.
    ///
    /// This is the coupling bd-16g.5.3 exists to prevent, stated for the migrator: the
    /// choice on the 0->1 edge must not depend on what happened on island 2.
    #[test]
    fn bd_16g_5_2_selection_on_one_edge_ignores_unrelated_islands() {
        let agents = |base: u64| {
            (0..6u64)
                .map(|i| CandidateAgent {
                    uid: base + i,
                    energy: 1.0 + i as f32,
                    health: 1.0,
                    age: 10,
                    speed: 1.0,
                })
                .collect::<Vec<_>>()
        };
        let config = MigrationConfig {
            selection_rule: EmigrantSelectionRule::Fittest,
            emigrants_per_edge: 2,
            topology: MigrationTopology::Ring,
            ..MigrationConfig::default()
        };
        let master = 0xFEED_FEED_FEED_FEED_u64;

        let mut two = BTreeMap::new();
        two.insert(IslandId(0), agents(1));
        two.insert(IslandId(1), agents(100));
        let small = select_emigrants(&two, &config, master, 3).expect("two islands");

        let mut three = two.clone();
        three.insert(IslandId(2), agents(200));
        let large = select_emigrants(&three, &config, master, 3).expect("three islands");

        let edge_moves = |report: &BarrierReport| {
            report
                .moves
                .iter()
                .filter(|m| m.from_island == IslandId(0) && m.to_island == IslandId(1))
                .map(|m| m.agent_uid)
                .collect::<Vec<_>>()
        };
        assert!(
            !edge_moves(&small).is_empty(),
            "the 0->1 edge must move agents"
        );
        assert_eq!(
            edge_moves(&small),
            edge_moves(&large),
            "who leaves island 0 for island 1 must not depend on island 2 existing"
        );
    }
}
