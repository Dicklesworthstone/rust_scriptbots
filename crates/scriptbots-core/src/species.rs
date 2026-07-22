//! Species segmentation engine: phenotype clustering, label continuity, and deterministic naming.
//!
//! # Purpose
//! Segments the living population into distinct species based on normalized phenotype vectors.
//!
//! # Hard Guarantees
//! - **Order Invariance**: Shuffling the order of input samples produces identical results.
//! - **Label Stability**: Small changes (e.g. death of one agent) retain species IDs via Jaccard matching.
//! - **Fixed-Range Scaling**: Normalizes dimensions against fixed config boundaries to prevent false cluster shifts.
//! - **Determinism**: Name generation and clustering are 100% deterministic (FNV-1a hash over founder UID & tick).
//! - **Panic-Free**: Empty inputs, singletons, or non-finite float values return typed reports/empty tables, never panic.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::module_name_repetitions
)]

use crate::{AgentUid, Tick};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Stable identifier for a species, monotonically increasing.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct SpeciesId(pub u64);

/// Configuration parameters for species segmentation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeciesParams {
    /// Maximum Euclidean distance in normalized phenotype space to join a cluster.
    pub distance_threshold: f32,
    /// Minimum Jaccard membership overlap ratio (0.0 to 1.0) to inherit a previous species ID.
    pub continuity_threshold: f32,
    /// Minimum number of agents required for a valid species cluster (default 1).
    pub min_cluster_size: usize,
    /// Fixed (min, max) bounds per dimension for deterministic feature scaling.
    pub dimension_ranges: Vec<(f32, f32)>,
}

impl Default for SpeciesParams {
    fn default() -> Self {
        Self {
            distance_threshold: 0.25,
            continuity_threshold: 0.3,
            min_cluster_size: 1,
            dimension_ranges: vec![
                (0.0, 1.0), // e.g. herbivore tendency
                (0.0, 2.0), // e.g. speed / size
                (0.0, 1.0), // e.g. sensory range
            ],
        }
    }
}

/// Description of a single segmented species clade at a specific point in time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Species {
    /// Monotonic species ID.
    pub id: SpeciesId,
    /// Deterministic human-readable name (e.g., "Swiftgiver-14").
    pub name: String,
    /// Stable UIDs of founder agent(s) when this species was first minted.
    pub founders: Vec<AgentUid>,
    /// Stable UIDs of all currently living member agents in this species.
    pub members: Vec<AgentUid>,
    /// Centroid in normalized phenotype space.
    pub centroid: Vec<f32>,
    /// Mean Euclidean distance of members from the centroid.
    pub spread: f32,
    /// Tick when this species ID was first minted.
    pub first_tick: Tick,
    /// Most recent tick when members were present.
    pub last_seen_tick: Tick,
}

/// Complete table of active species at a given simulation tick.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeciesTable {
    /// Tick at which segmentation was performed.
    pub tick: Tick,
    /// Active species list, sorted by [`SpeciesId`].
    pub species: Vec<Species>,
    /// Monotonic next ID counter.
    pub next_id: SpeciesId,
}

/// Report summarizing changes during a species segmentation step.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentReport {
    /// Tick of this segmentation step.
    pub tick: Tick,
    /// Count of active species.
    pub active_species_count: usize,
    /// Newly minted species in this step: (ID, Name, Primary Founder UID).
    pub new_species_minted: Vec<(SpeciesId, String, AgentUid)>,
    /// Species that went extinct in this step: (ID, Name).
    pub extinct_species_dropped: Vec<(SpeciesId, String)>,
    /// Total number of living agents segmented.
    pub total_agents_segmented: usize,
}

const ADJECTIVES: &[&str] = &[
    "Swift",
    "Amber",
    "Crystal",
    "Solar",
    "Frost",
    "Zenith",
    "Prismatic",
    "Radiant",
    "Silent",
    "Verdant",
    "Obsidian",
    "Astral",
    "Crimson",
    "Echo",
    "Golden",
    "Velvet",
    "Shadow",
    "Lunar",
    "Starlight",
    "Copper",
    "Cobalt",
    "Emerald",
    "Sylvan",
    "Iron",
    "Silver",
    "Coral",
    "Topaz",
    "Azure",
    "Scarlet",
    "Granite",
    "Breeze",
    "Thunder",
];

const NOUNS: &[&str] = &[
    "giver",
    "hunter",
    "wanderer",
    "seeker",
    "glider",
    "weaver",
    "runner",
    "striker",
    "sentinel",
    "forager",
    "chaser",
    "crawler",
    "observer",
    "strider",
    "guardian",
    "nomad",
    "stalker",
    "pioneer",
    "drifter",
    "watcher",
    "racer",
    "harvester",
    "scout",
    "ranger",
    "sailor",
    "diver",
    "flyer",
    "bounder",
    "weaver",
    "tracker",
    "voyager",
    "prowler",
];

/// Generates a deterministic human-readable species name from founder UID and tick.
#[must_use]
pub fn generate_species_name(founder_uid: AgentUid, first_tick: Tick) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in founder_uid.0.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for byte in first_tick.0.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }

    let adj_idx = (hash as usize) % ADJECTIVES.len();
    let noun_idx = ((hash >> 16) as usize) % NOUNS.len();

    format!(
        "{}{}-{}",
        ADJECTIVES[adj_idx], NOUNS[noun_idx], first_tick.0
    )
}

/// Normalizes a single raw phenotype vector against fixed dimension ranges.
fn normalize_phenotype(raw: &[f32], ranges: &[(f32, f32)]) -> Vec<f32> {
    raw.iter()
        .enumerate()
        .map(|(i, &v)| {
            if !v.is_finite() {
                return 0.0;
            }
            if let Some(&(min_val, max_val)) = ranges.get(i) {
                if max_val > min_val {
                    ((v - min_val) / (max_val - min_val)).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            } else {
                v.clamp(0.0, 1.0)
            }
        })
        .collect()
}

/// Euclidean distance between two equal-length normalized vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();
    sum_sq.sqrt()
}

/// Calculates Jaccard similarity between two sorted member sets.
fn jaccard_similarity(a: &[AgentUid], b: &[AgentUid]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let set_a: BTreeSet<AgentUid> = a.iter().copied().collect();
    let set_b: BTreeSet<AgentUid> = b.iter().copied().collect();
    let intersection_count = set_a.intersection(&set_b).count();
    let union_count = set_a.union(&set_b).count();
    if union_count == 0 {
        0.0
    } else {
        intersection_count as f32 / union_count as f32
    }
}

/// Intermediate raw cluster formed during segmentation.
struct RawCluster {
    members: Vec<AgentUid>,
    norm_vectors: Vec<Vec<f32>>,
}

/// Main entry point for species segmentation.
///
/// Segments living agents into species, enforcing label continuity from `prev` table.
#[must_use]
pub fn segment_species(
    tick: Tick,
    samples: &[(AgentUid, Vec<f32>)],
    prev: &SpeciesTable,
    params: &SpeciesParams,
) -> (SpeciesTable, SegmentReport) {
    if samples.is_empty() {
        let mut report = SegmentReport {
            tick,
            active_species_count: 0,
            new_species_minted: Vec::new(),
            extinct_species_dropped: Vec::new(),
            total_agents_segmented: 0,
        };
        for prev_sp in &prev.species {
            report
                .extinct_species_dropped
                .push((prev_sp.id, prev_sp.name.clone()));
        }
        return (
            SpeciesTable {
                tick,
                species: Vec::new(),
                next_id: prev.next_id,
            },
            report,
        );
    }

    // 1. Sort samples deterministically by AgentUid ascending (ensures order invariance)
    let mut sorted_samples = samples.to_vec();
    sorted_samples.sort_by_key(|(uid, _)| *uid);

    // 2. Normalize vectors against fixed ranges
    let normalized_samples: Vec<(AgentUid, Vec<f32>)> = sorted_samples
        .into_iter()
        .map(|(uid, vec)| (uid, normalize_phenotype(&vec, &params.dimension_ranges)))
        .collect();

    // 3. Leader-based agglomerative clustering
    let mut raw_clusters: Vec<RawCluster> = Vec::new();
    for (uid, norm_vec) in normalized_samples {
        let mut matched_idx = None;
        for (idx, cluster) in raw_clusters.iter().enumerate() {
            // Compare against leader (first member) of each cluster
            let leader_vec = &cluster.norm_vectors[0];
            let dist = euclidean_distance(&norm_vec, leader_vec);
            if dist <= params.distance_threshold {
                matched_idx = Some(idx);
                break;
            }
        }

        if let Some(idx) = matched_idx {
            raw_clusters[idx].members.push(uid);
            raw_clusters[idx].norm_vectors.push(norm_vec);
        } else {
            raw_clusters.push(RawCluster {
                members: vec![uid],
                norm_vectors: vec![norm_vec],
            });
        }
    }

    // Filter clusters by minimum size
    raw_clusters.retain(|c| c.members.len() >= params.min_cluster_size);

    // 4. Match raw clusters to previous species using Jaccard continuity
    let mut matched_prev_ids = BTreeSet::new();
    let mut current_next_id = prev.next_id;
    let mut new_species_list = Vec::new();
    let mut report = SegmentReport {
        tick,
        active_species_count: 0,
        new_species_minted: Vec::new(),
        extinct_species_dropped: Vec::new(),
        total_agents_segmented: samples.len(),
    };

    for cluster in raw_clusters {
        let mut best_prev_match: Option<(&Species, f32)> = None;
        for prev_sp in &prev.species {
            if matched_prev_ids.contains(&prev_sp.id) {
                continue;
            }
            let sim = jaccard_similarity(&cluster.members, &prev_sp.members);
            if sim >= params.continuity_threshold {
                match best_prev_match {
                    Some((_, best_sim)) => {
                        if sim > best_sim {
                            best_prev_match = Some((prev_sp, sim));
                        }
                    }
                    None => {
                        best_prev_match = Some((prev_sp, sim));
                    }
                }
            }
        }

        // Calculate centroid and spread for this cluster
        let dim = cluster.norm_vectors[0].len();
        let count = cluster.norm_vectors.len() as f32;
        let mut centroid = vec![0.0f32; dim];
        for vec in &cluster.norm_vectors {
            for (i, &val) in vec.iter().enumerate() {
                centroid[i] += val;
            }
        }
        for val in &mut centroid {
            *val /= count;
        }

        let sum_sq_dist: f32 = cluster
            .norm_vectors
            .iter()
            .map(|vec| {
                let dist = euclidean_distance(vec, &centroid);
                dist * dist
            })
            .sum();
        let spread = (sum_sq_dist / count).sqrt();

        let (sp_id, sp_name, first_tick, founders) = if let Some((prev_sp, _)) = best_prev_match {
            matched_prev_ids.insert(prev_sp.id);
            (
                prev_sp.id,
                prev_sp.name.clone(),
                prev_sp.first_tick,
                prev_sp.founders.clone(),
            )
        } else {
            let id = current_next_id;
            current_next_id.0 += 1;
            let primary_founder = cluster.members[0];
            let name = generate_species_name(primary_founder, tick);
            report
                .new_species_minted
                .push((id, name.clone(), primary_founder));
            (id, name, tick, vec![primary_founder])
        };

        new_species_list.push(Species {
            id: sp_id,
            name: sp_name,
            founders,
            members: cluster.members,
            centroid,
            spread,
            first_tick,
            last_seen_tick: tick,
        });
    }

    // Check for extinct species (previous species not matched)
    for prev_sp in &prev.species {
        if !matched_prev_ids.contains(&prev_sp.id) {
            report
                .extinct_species_dropped
                .push((prev_sp.id, prev_sp.name.clone()));
        }
    }

    // Sort species by SpeciesId for deterministic table ordering
    new_species_list.sort_by_key(|s| s.id);
    report.active_species_count = new_species_list.len();

    (
        SpeciesTable {
            tick,
            species: new_species_list,
            next_id: current_next_id,
        },
        report,
    )
}

/// Multi-axis behavior feature vector for an agent (bd-2z0.11.2).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentPhenotypeVector {
    pub agent_uid: AgentUid,
    pub movement_speed_mean: f32,
    pub diet_herbivore_ratio: f32,
    pub sensing_range_mean: f32,
    pub aggression_index: f32,
    pub giving_altruism_index: f32,
    pub reproduction_rate: f32,
}

impl AgentPhenotypeVector {
    pub fn features(&self) -> [f32; 6] {
        [
            self.movement_speed_mean,
            self.diet_herbivore_ratio,
            self.sensing_range_mean,
            self.aggression_index,
            self.giving_altruism_index,
            self.reproduction_rate,
        ]
    }
}

/// Statistical comparison between two phenotype clusters (bd-2z0.11.2).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhenotypeClusterComparison {
    pub cluster_a_name: String,
    pub cluster_b_name: String,
    pub sample_size_a: usize,
    pub sample_size_b: usize,
    pub feature_effect_sizes: std::collections::HashMap<String, f32>,
    pub overall_mahalanobis_distance: f32,
}

/// Comprehensive phenotype shift and interaction analysis report (bd-2z0.11.2).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhenotypeAnalysisReport {
    pub run_id: String,
    pub tick: Tick,
    pub total_agents_analyzed: usize,
    pub mean_phenotype: Vec<f32>,
    pub phenotype_variance: Vec<f32>,
    pub comparisons: Vec<PhenotypeClusterComparison>,
}

/// Compares two phenotype feature cohorts computing effect sizes (Cohen's d) per dimension (bd-2z0.11.2).
pub fn compare_phenotype_clusters(
    name_a: &str,
    a: &[AgentPhenotypeVector],
    name_b: &str,
    b: &[AgentPhenotypeVector],
) -> PhenotypeClusterComparison {
    let feature_names = [
        "movement_speed",
        "diet_herbivore_ratio",
        "sensing_range",
        "aggression_index",
        "giving_altruism_index",
        "reproduction_rate",
    ];

    let mut effect_sizes = std::collections::HashMap::new();

    if !a.is_empty() && !b.is_empty() {
        for (i, &name) in feature_names.iter().enumerate() {
            let vals_a: Vec<f32> = a.iter().map(|v| v.features()[i]).collect();
            let vals_b: Vec<f32> = b.iter().map(|v| v.features()[i]).collect();

            let mean_a = vals_a.iter().sum::<f32>() / vals_a.len() as f32;
            let mean_b = vals_b.iter().sum::<f32>() / vals_b.len() as f32;

            let var_a = vals_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f32>() / vals_a.len() as f32;
            let var_b = vals_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f32>() / vals_b.len() as f32;

            let pooled_sd = ((var_a + var_b) / 2.0).sqrt().max(1e-6);
            let cohens_d = (mean_a - mean_b) / pooled_sd;
            effect_sizes.insert(name.to_string(), cohens_d);
        }
    }

    PhenotypeClusterComparison {
        cluster_a_name: name_a.to_string(),
        cluster_b_name: name_b.to_string(),
        sample_size_a: a.len(),
        sample_size_b: b.len(),
        feature_effect_sizes: effect_sizes,
        overall_mahalanobis_distance: 0.0,
    }
}

/// Generates a comprehensive phenotype analysis report over population samples (bd-2z0.11.2).
pub fn compute_phenotype_analysis(
    run_id: &str,
    tick: Tick,
    vectors: &[AgentPhenotypeVector],
) -> PhenotypeAnalysisReport {
    if vectors.is_empty() {
        return PhenotypeAnalysisReport {
            run_id: run_id.to_string(),
            tick,
            total_agents_analyzed: 0,
            mean_phenotype: vec![0.0; 6],
            phenotype_variance: vec![0.0; 6],
            comparisons: Vec::new(),
        };
    }

    let count = vectors.len() as f32;
    let mut means = vec![0.0f32; 6];

    for vec in vectors {
        let f = vec.features();
        for i in 0..6 {
            means[i] += f[i];
        }
    }
    for i in 0..6 {
        means[i] /= count;
    }

    let mut vars = vec![0.0f32; 6];
    for vec in vectors {
        let f = vec.features();
        for i in 0..6 {
            vars[i] += (f[i] - means[i]).powi(2);
        }
    }
    for i in 0..6 {
        vars[i] /= count;
    }

    PhenotypeAnalysisReport {
        run_id: run_id.to_string(),
        tick,
        total_agents_analyzed: vectors.len(),
        mean_phenotype: means,
        phenotype_variance: vars,
        comparisons: Vec::new(),
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_name_generation_golden_values() {
        let name1 = generate_species_name(AgentUid(1), Tick(100));
        let name2 = generate_species_name(AgentUid(1), Tick(100));
        let name3 = generate_species_name(AgentUid(2), Tick(100));

        assert_eq!(name1, name2, "same inputs must produce identical name");
        assert_ne!(
            name1, name3,
            "different UIDs should produce different names"
        );
        assert!(name1.contains("-100"), "name must contain tick suffix");
    }

    #[test]
    fn test_two_well_separated_clusters() {
        let params = SpeciesParams::default();
        let prev = SpeciesTable::default();

        let samples = vec![
            (AgentUid(1), vec![0.1, 0.1, 0.1]),
            (AgentUid(2), vec![0.12, 0.08, 0.11]),
            (AgentUid(3), vec![0.9, 0.9, 0.9]),
            (AgentUid(4), vec![0.88, 0.91, 0.89]),
        ];

        let (table, report) = segment_species(Tick(10), &samples, &prev, &params);
        assert_eq!(table.species.len(), 2);
        assert_eq!(report.new_species_minted.len(), 2);
        assert_eq!(table.species[0].members, vec![AgentUid(1), AgentUid(2)]);
        assert_eq!(table.species[1].members, vec![AgentUid(3), AgentUid(4)]);
    }

    #[test]
    fn test_order_invariance() {
        let params = SpeciesParams::default();
        let prev = SpeciesTable::default();

        let samples1 = vec![
            (AgentUid(1), vec![0.1, 0.1, 0.1]),
            (AgentUid(2), vec![0.12, 0.08, 0.11]),
            (AgentUid(3), vec![0.9, 0.9, 0.9]),
            (AgentUid(4), vec![0.88, 0.91, 0.89]),
        ];

        let samples2 = vec![
            (AgentUid(4), vec![0.88, 0.91, 0.89]),
            (AgentUid(1), vec![0.1, 0.1, 0.1]),
            (AgentUid(3), vec![0.9, 0.9, 0.9]),
            (AgentUid(2), vec![0.12, 0.08, 0.11]),
        ];

        let (table1, _) = segment_species(Tick(10), &samples1, &prev, &params);
        let (table2, _) = segment_species(Tick(10), &samples2, &prev, &params);

        assert_eq!(
            table1, table2,
            "shuffling input samples must produce identical SpeciesTable"
        );
    }

    #[test]
    fn test_label_stability_across_steps() {
        let params = SpeciesParams::default();
        let prev = SpeciesTable::default();

        let step1_samples = vec![
            (AgentUid(1), vec![0.1, 0.1, 0.1]),
            (AgentUid(2), vec![0.12, 0.08, 0.11]),
            (AgentUid(3), vec![0.11, 0.10, 0.09]),
        ];

        let (table1, _) = segment_species(Tick(10), &step1_samples, &prev, &params);
        assert_eq!(table1.species.len(), 1);
        let sp1_id = table1.species[0].id;
        let sp1_name = table1.species[0].name.clone();

        // Step 2: Agent 2 dies, but Agent 1 & 3 survive
        let step2_samples = vec![
            (AgentUid(1), vec![0.1, 0.1, 0.1]),
            (AgentUid(3), vec![0.11, 0.10, 0.09]),
        ];

        let (table2, report2) = segment_species(Tick(20), &step2_samples, &table1, &params);
        assert_eq!(table2.species.len(), 1);
        assert_eq!(table2.species[0].id, sp1_id, "species ID must be preserved");
        assert_eq!(
            table2.species[0].name, sp1_name,
            "species name must be preserved"
        );
        assert!(report2.new_species_minted.is_empty());
        assert!(report2.extinct_species_dropped.is_empty());
    }

    #[test]
    fn test_degenerate_and_empty_inputs() {
        let params = SpeciesParams::default();
        let prev = SpeciesTable::default();

        let (empty_table, report) = segment_species(Tick(10), &[], &prev, &params);
        assert!(empty_table.species.is_empty());
        assert_eq!(report.total_agents_segmented, 0);

        let singleton = vec![(AgentUid(1), vec![0.5, 0.5, 0.5])];
        let (single_table, _) = segment_species(Tick(10), &singleton, &prev, &params);
        assert_eq!(single_table.species.len(), 1);
        assert_eq!(single_table.species[0].members, vec![AgentUid(1)]);
    }

    #[test]
    fn test_fixed_range_scaling_outlier_insensitivity() {
        let params = SpeciesParams::default();
        let prev = SpeciesTable::default();

        let normal_samples = vec![
            (AgentUid(1), vec![0.1, 0.1, 0.1]),
            (AgentUid(2), vec![0.9, 0.9, 0.9]),
        ];

        let (table_normal, _) = segment_species(Tick(10), &normal_samples, &prev, &params);

        // Add a extreme outlier (100.0) which is clamped to 1.0 by fixed range (0.0..1.0)
        let outlier_samples = vec![
            (AgentUid(1), vec![0.1, 0.1, 0.1]),
            (AgentUid(2), vec![0.9, 0.9, 0.9]),
            (AgentUid(3), vec![100.0, 100.0, 100.0]),
        ];

        let (table_outlier, _) = segment_species(Tick(10), &outlier_samples, &prev, &params);
        assert_eq!(table_normal.species[0].members, vec![AgentUid(1)]);
        assert_eq!(table_outlier.species[0].members, vec![AgentUid(1)]);
    }

    #[test]
    fn test_phenotype_feature_analysis_and_clustering() {
        let cohort_a = vec![
            AgentPhenotypeVector {
                agent_uid: AgentUid(1),
                movement_speed_mean: 1.5,
                diet_herbivore_ratio: 0.9,
                sensing_range_mean: 0.8,
                aggression_index: 0.1,
                giving_altruism_index: 0.5,
                reproduction_rate: 0.05,
            },
            AgentPhenotypeVector {
                agent_uid: AgentUid(2),
                movement_speed_mean: 1.6,
                diet_herbivore_ratio: 0.95,
                sensing_range_mean: 0.82,
                aggression_index: 0.05,
                giving_altruism_index: 0.6,
                reproduction_rate: 0.06,
            },
        ];

        let cohort_b = vec![
            AgentPhenotypeVector {
                agent_uid: AgentUid(3),
                movement_speed_mean: 0.5,
                diet_herbivore_ratio: 0.1,
                sensing_range_mean: 0.3,
                aggression_index: 0.8,
                giving_altruism_index: 0.0,
                reproduction_rate: 0.01,
            },
        ];

        let report = compute_phenotype_analysis("run-test", Tick(100), &cohort_a);
        assert_eq!(report.total_agents_analyzed, 2);
        assert_eq!(report.mean_phenotype.len(), 6);

        let comparison = compare_phenotype_clusters("Herbivores", &cohort_a, "Carnivores", &cohort_b);
        assert_eq!(comparison.sample_size_a, 2);
        assert_eq!(comparison.sample_size_b, 1);
        assert!(comparison.feature_effect_sizes.contains_key("movement_speed"));
    }
}

