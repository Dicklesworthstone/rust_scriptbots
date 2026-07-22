//! MAP-Elites behavioral archive, Quality-Diversity (QD) metrics, and novelty search (bd-16g.6).

use crate::{AgentUid, Tick};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Discretized behavior cell coordinate in N-dimensional phenotype space.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MapElitesCellKey(pub Vec<i32>);

/// Record for an elite individual stored in a MAP-Elites grid cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapElitesRecord {
    pub agent_uid: AgentUid,
    pub birth_tick: Tick,
    pub fitness: f32,
    pub phenotype: Vec<f32>,
    pub genome_data: Vec<u8>,
}

/// Selection mode for evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SelectionMode {
    #[default]
    Fitness,
    Novelty,
    CuriosityHybrid,
}

/// MAP-Elites behavioral grid archive.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MapElitesArchive {
    /// Number of discrete bins per behavioral dimension.
    pub grid_bins_per_dim: usize,
    /// Min/max bounds for each behavioral dimension.
    pub dimension_ranges: Vec<(f32, f32)>,
    /// Map of discretized cell coordinates to elite individual records.
    pub cells: BTreeMap<MapElitesCellKey, MapElitesRecord>,
}

impl MapElitesArchive {
    /// Create a new MAP-Elites archive with specified resolution and dimension bounds.
    pub fn new(grid_bins: usize, ranges: Vec<(f32, f32)>) -> Self {
        Self {
            grid_bins_per_dim: grid_bins,
            dimension_ranges: ranges,
            cells: BTreeMap::new(),
        }
    }

    /// Discretize a continuous phenotype vector into a discrete cell key.
    pub fn discretize(&self, phenotype: &[f32]) -> MapElitesCellKey {
        let mut coords = Vec::with_capacity(phenotype.len());
        for (i, &val) in phenotype.iter().enumerate() {
            let (min, max) = self.dimension_ranges.get(i).copied().unwrap_or((0.0, 1.0));
            let span = (max - min).max(1e-6);
            let val_clean = if val.is_nan() { min } else { val };
            let normalized = ((val_clean - min) / span).clamp(0.0, 0.9999);
            let bin = (normalized * self.grid_bins_per_dim as f32).floor() as i32;
            coords.push(bin);
        }
        MapElitesCellKey(coords)
    }

    /// Try inserting an agent into the archive. Replaces existing elite if fitness is higher.
    pub fn insert(&mut self, record: MapElitesRecord) -> bool {
        let key = self.discretize(&record.phenotype);
        match self.cells.get(&key) {
            Some(existing) => match record.fitness.total_cmp(&existing.fitness) {
                std::cmp::Ordering::Greater => {
                    self.cells.insert(key, record);
                    true
                }
                std::cmp::Ordering::Equal if record.agent_uid < existing.agent_uid => {
                    self.cells.insert(key, record);
                    true
                }
                _ => false,
            },
            None => {
                self.cells.insert(key, record);
                true
            }
        }
    }

    /// Compute Quality-Diversity (QD) score (sum of elite fitnesses).
    pub fn qd_score(&self) -> f64 {
        self.cells.values().map(|r| r.fitness as f64).sum()
    }

    /// Compute percentage of total grid cells filled.
    pub fn coverage_ratio(&self, num_dims: usize) -> f32 {
        let total_cells = self
            .grid_bins_per_dim
            .checked_pow(num_dims as u32)
            .unwrap_or(usize::MAX);
        if total_cells == 0 {
            0.0
        } else {
            self.cells.len() as f32 / total_cells as f32
        }
    }
}

/// Compute k-NN novelty score for a candidate phenotype against the archive.
pub fn compute_novelty_score(candidate: &[f32], archive: &MapElitesArchive, k: usize) -> f32 {
    if archive.cells.is_empty() || k == 0 {
        return 0.0;
    }

    let mut distances: Vec<f32> = archive
        .cells
        .values()
        .map(|r| {
            candidate
                .iter()
                .zip(r.phenotype.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .collect();

    distances.sort_by(f32::total_cmp);
    let take_k = k.min(distances.len());
    let sum: f32 = distances.iter().take(take_k).sum();
    sum / take_k as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_elites_cell_discretization_and_insertion() {
        let mut archive = MapElitesArchive::new(5, vec![(0.0, 1.0), (0.0, 2.0)]);

        let r1 = MapElitesRecord {
            agent_uid: AgentUid(1),
            birth_tick: Tick(10),
            fitness: 50.0,
            phenotype: vec![0.1, 0.5],
            genome_data: vec![1, 2, 3],
        };
        assert!(archive.insert(r1));
        assert_eq!(archive.cells.len(), 1);

        // Lower fitness at same cell -> rejected
        let r2 = MapElitesRecord {
            agent_uid: AgentUid(2),
            birth_tick: Tick(12),
            fitness: 30.0,
            phenotype: vec![0.12, 0.52],
            genome_data: vec![1, 2, 3],
        };
        assert!(!archive.insert(r2));
        assert_eq!(archive.cells.len(), 1);

        // Higher fitness at same cell -> replaces
        let r3 = MapElitesRecord {
            agent_uid: AgentUid(3),
            birth_tick: Tick(15),
            fitness: 75.0,
            phenotype: vec![0.11, 0.51],
            genome_data: vec![1, 2, 4],
        };
        assert!(archive.insert(r3));
        assert_eq!(archive.qd_score(), 75.0);
    }

    #[test]
    fn test_novelty_score_knn() {
        let mut archive = MapElitesArchive::new(5, vec![(0.0, 1.0)]);
        archive.insert(MapElitesRecord {
            agent_uid: AgentUid(1),
            birth_tick: Tick(0),
            fitness: 10.0,
            phenotype: vec![0.0],
            genome_data: vec![],
        });

        let nov = compute_novelty_score(&[1.0], &archive, 1);
        assert!((nov - 1.0).abs() < 1e-4);
    }
}
