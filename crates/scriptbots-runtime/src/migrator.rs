//! Deterministic Migrator for multi-island archipelagos (`bd-16g.5.2`).
//!
//! Owns two-phase selection and application of migration barriers across islands.
//! Guarantees population conservation, deterministic total ordering, and total
//! tie-breaking (no `partial_cmp().unwrap()`, no HashMap iteration).

use rand::Rng;
use scriptbots_core::SmallRngStream;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// Unique identifier for an island within an archipelago.
pub type IslandId = u32;

/// Emigrant selection strategy for migration barriers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmigrantSelectionRule {
    /// Select agents at random using a dedicated seeded PRNG stream.
    Random,
    /// Select agents with highest energy (tie-break by AgentUid ASC).
    Fittest,
    /// Select youngest agents by age ASC (tie-break by AgentUid ASC).
    Youngest,
    /// Select agents with highest realized speed (tie-break by AgentUid ASC).
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

/// Record of a single agent migration event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmigrantRecord {
    pub from_island: IslandId,
    pub to_island: IslandId,
    pub agent_uid: u64,
    pub selection_rule: EmigrantSelectionRule,
    pub rank: usize,
    pub key_value: f64,
}

/// Summary report produced by a migration barrier execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BarrierReport {
    pub tick: u64,
    pub selection_rule: EmigrantSelectionRule,
    pub moves: Vec<EmigrantRecord>,
    pub pop_before: BTreeMap<IslandId, u32>,
    pub pop_after: BTreeMap<IslandId, u32>,
    pub conserved: bool,
}

/// Errors occurring during migration processing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum MigrationError {
    #[error("population not conserved across migration barrier: before={before}, after={after}")]
    PopulationNotConserved {
        before: u32,
        after: u32,
        moves_count: usize,
    },
    #[error("invalid edge in topology: from={from}, to={to}")]
    InvalidEdge { from: IslandId, to: IslandId },
    #[error("empty archipelago passed to migrator")]
    EmptyArchipelago,
}

/// Snapshot of a single agent's state for migration selection.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateAgent {
    pub uid: u64,
    pub energy: f32,
    pub health: f32,
    pub age: u32,
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

/// Selects emigrants across `islands` according to two-phase deterministic rules.
///
/// Returns a list of `EmigrantRecord` moves and the updated population counts.
pub fn select_emigrants(
    islands_candidates: &BTreeMap<IslandId, Vec<CandidateAgent>>,
    config: &MigrationConfig,
    seed: u64,
    tick: u64,
) -> Result<BarrierReport, MigrationError> {
    if islands_candidates.is_empty() {
        return Err(MigrationError::EmptyArchipelago);
    }

    let island_ids: Vec<IslandId> = islands_candidates.keys().copied().collect();
    let edges = config.topology.build_edges(&island_ids);

    let pop_before: BTreeMap<IslandId, u32> = islands_candidates
        .iter()
        .map(|(&id, c)| (id, c.len() as u32))
        .collect();

    let total_before: u32 = pop_before.values().sum();

    // Rank candidates per island
    let mut ranked_candidates: BTreeMap<IslandId, Vec<(CandidateAgent, f64)>> = BTreeMap::new();
    let mut rng = SmallRngStream::seed_from_u64(seed.wrapping_add(tick));

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
                    EmigrantSelectionRule::Fittest => c.energy as f64,
                    EmigrantSelectionRule::Youngest => c.age as f64,
                    EmigrantSelectionRule::Wanderer => c.speed as f64,
                    EmigrantSelectionRule::Random => {
                        random_ranks.get(&c.uid).copied().unwrap_or(0) as f64
                    }
                };
                (c, val)
            })
            .collect();

        ranked_candidates.insert(island_id, scored);
    }

    // Two-Phase Selection: claim candidates without double-emigration
    let mut claimed_uids: BTreeSet<u64> = BTreeSet::new();
    let mut moves: Vec<EmigrantRecord> = Vec::new();

    for &(from_island, to_island) in &edges {
        if let Some(candidates) = ranked_candidates.get(&from_island) {
            let mut selected_for_edge = 0usize;
            for (rank, (candidate, key_val)) in candidates.iter().enumerate() {
                if selected_for_edge >= config.emigrants_per_edge {
                    break;
                }
                if claimed_uids.insert(candidate.uid) {
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
        if let Some(c) = pop_after.get_mut(&rec.from_island) {
            *c = c.saturating_sub(1);
        }
        if let Some(c) = pop_after.get_mut(&rec.to_island) {
            *c = c.saturating_add(1);
        }
    }

    let total_after: u32 = pop_after.values().sum();
    let conserved = total_before == total_after;

    if !conserved {
        return Err(MigrationError::PopulationNotConserved {
            before: total_before,
            after: total_after,
            moves_count: moves.len(),
        });
    }

    Ok(BarrierReport {
        tick,
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
    fn test_ring_topology_build_edges() {
        let topo = MigrationTopology::Ring;
        let edges = topo.build_edges(&[0, 1, 2]);
        assert_eq!(edges, vec![(0, 1), (1, 2), (2, 0)]);
    }

    #[test]
    fn test_fittest_selection_with_uid_tiebreak() {
        let mut candidates = BTreeMap::new();
        candidates.insert(
            0,
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
        candidates.insert(1, vec![]);

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
            0,
            vec![CandidateAgent {
                uid: 1,
                energy: 1.0,
                health: 1.0,
                age: 10,
                speed: 0.1,
            }],
        );
        candidates.insert(1, vec![]);
        candidates.insert(2, vec![]);

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
            candidates.insert(id, agents);
        }

        let config = MigrationConfig::default();
        let report = select_emigrants(&candidates, &config, 12345, 500).unwrap();
        let total_before: u32 = report.pop_before.values().sum();
        let total_after: u32 = report.pop_after.values().sum();
        assert_eq!(total_before, total_after);
        assert!(report.conserved);
    }
}
