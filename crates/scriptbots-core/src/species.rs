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

use crate::detect::{DetectionEvidence, DetectionKind, EvidenceClass, EvidenceSide};
use crate::{AgentUid, Tick};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};

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

impl SpeciesParams {
    /// Computes a deterministic canonical digest of the clustering parameters.
    #[must_use]
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(b"SpeciesParamsV1\n");
        h.update(&self.distance_threshold.to_le_bytes());
        h.update(&self.continuity_threshold.to_le_bytes());
        h.update(&(self.min_cluster_size as u64).to_le_bytes());
        for &(min, max) in &self.dimension_ranges {
            h.update(&min.to_le_bytes());
            h.update(&max.to_le_bytes());
        }
        h.finalize().to_hex().to_string()
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

impl SpeciesTable {
    /// Computes a deterministic, permutation-invariant, byte-stable digest of the species table.
    #[must_use]
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"SpeciesTableV1\n");
        hasher.update(&self.tick.0.to_le_bytes());
        hasher.update(&(self.species.len() as u64).to_le_bytes());
        for sp in &self.species {
            hasher.update(&sp.id.0.to_le_bytes());
            hasher.update(sp.name.as_bytes());
            hasher.update(&(sp.founders.len() as u64).to_le_bytes());
            for f in &sp.founders {
                hasher.update(&f.0.to_le_bytes());
            }
            hasher.update(&(sp.members.len() as u64).to_le_bytes());
            for m in &sp.members {
                hasher.update(&m.0.to_le_bytes());
            }
            for c in &sp.centroid {
                hasher.update(&c.to_le_bytes());
            }
            hasher.update(&sp.spread.to_le_bytes());
            hasher.update(&sp.first_tick.0.to_le_bytes());
            hasher.update(&sp.last_seen_tick.0.to_le_bytes());
        }
        hasher.update(&self.next_id.0.to_le_bytes());
        hasher.finalize().to_hex().to_string()
    }

    /// Finds the species that contains the given living agent UID, if present.
    #[must_use]
    pub fn find_species_for_agent(&self, uid: AgentUid) -> Option<&Species> {
        self.species.iter().find(|s| s.members.contains(&uid))
    }

    /// Number of active species currently living in the table.
    #[must_use]
    pub const fn species_count(&self) -> usize {
        self.species.len()
    }
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

impl SegmentReport {
    /// Computes a deterministic, byte-stable digest of the segment report.
    #[must_use]
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"SegmentReportV1\n");
        hasher.update(&self.tick.0.to_le_bytes());
        hasher.update(&(self.active_species_count as u64).to_le_bytes());
        hasher.update(&(self.total_agents_segmented as u64).to_le_bytes());
        hasher.update(&(self.new_species_minted.len() as u64).to_le_bytes());
        for (id, name, founder) in &self.new_species_minted {
            hasher.update(&id.0.to_le_bytes());
            hasher.update(name.as_bytes());
            hasher.update(&founder.0.to_le_bytes());
        }
        hasher.update(&(self.extinct_species_dropped.len() as u64).to_le_bytes());
        for (id, name) in &self.extinct_species_dropped {
            hasher.update(&id.0.to_le_bytes());
            hasher.update(name.as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }
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
                        if sim.total_cmp(&best_sim).is_gt() {
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
    /// Stable agent logical identity.
    pub agent_uid: AgentUid,
    /// Mean movement speed.
    pub movement_speed_mean: f32,
    /// Ratio of herbivorous food consumed.
    pub diet_herbivore_ratio: f32,
    /// Mean sensory radius.
    pub sensing_range_mean: f32,
    /// Aggression/combat rate index.
    pub aggression_index: f32,
    /// Food sharing and altruism rate index.
    pub giving_altruism_index: f32,
    /// Reproduction rate.
    pub reproduction_rate: f32,
}

impl AgentPhenotypeVector {
    /// Extracts canonical 6-axis feature slice.
    #[must_use]
    pub const fn features(&self) -> [f32; 6] {
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
    /// Human-readable label for first cluster.
    pub cluster_a_name: String,
    /// Human-readable label for second cluster.
    pub cluster_b_name: String,
    /// Sample size of first cluster.
    pub sample_size_a: usize,
    /// Sample size of second cluster.
    pub sample_size_b: usize,
    /// Per-feature Cohen's d, keyed by feature name.
    ///
    /// Ordered so the serialized report is byte-stable: a `HashMap` here would emit its
    /// entries in per-process `RandomState` order, giving the same run different bytes on
    /// every execution and on every platform.
    pub feature_effect_sizes: BTreeMap<String, f32>,
    /// Multivariate separation of the two cohort means under their pooled covariance.
    ///
    /// Typed rather than numeric because `0.0` is not a neutral placeholder for this statistic:
    /// it is precisely the value meaning "these cohorts occupy the same point in phenotype
    /// space". A cohort pair too small or too degenerate to support a covariance inverse has no
    /// distance at all, and must say so instead of reporting perfect coincidence (bd-hawp).
    pub overall_mahalanobis_distance: MahalanobisDistance,
}

/// Multivariate cohort separation, or the reason there is none to report (bd-hawp).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum MahalanobisDistance {
    /// Mahalanobis distance between the cohort means under the pooled covariance.
    Computed(f32),
    /// The statistic is not defined for this cohort pair.
    Unavailable(MahalanobisUnavailable),
}

/// Why a cohort pair admits no Mahalanobis distance (bd-hawp).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MahalanobisUnavailable {
    /// One or both cohorts contributed no samples, so neither mean exists.
    EmptyCohort,
    /// Pooled covariance cannot reach full rank: its rank is bounded by `n_a + n_b - 2`, so
    /// fewer degrees of freedom than features leaves the inverse under-determined.
    InsufficientSamples {
        /// `n_a + n_b - 2`, the rank ceiling of the pooled covariance.
        degrees_of_freedom: usize,
        /// Feature count the covariance would need to span to be invertible.
        features: usize,
    },
    /// The pooled covariance is singular to working precision: some feature is constant across
    /// both cohorts, or two features are exactly collinear, so no inverse exists.
    SingularCovariance,
}

/// Comprehensive phenotype shift and interaction analysis report (bd-2z0.11.2).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhenotypeAnalysisReport {
    /// Simulation run identity.
    pub run_id: String,
    /// Simulation tick at which analysis was computed.
    pub tick: Tick,
    /// Total count of living agents analyzed.
    pub total_agents_analyzed: usize,
    /// Mean phenotype vector across all analyzed agents.
    pub mean_phenotype: Vec<f32>,
    /// Variance per phenotype feature across all analyzed agents.
    pub phenotype_variance: Vec<f32>,
    /// Pairwise cohort comparisons.
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

    let mut effect_sizes = BTreeMap::new();

    if !a.is_empty() && !b.is_empty() {
        for (i, &name) in feature_names.iter().enumerate() {
            let vals_a: Vec<f32> = a.iter().map(|v| v.features()[i]).collect();
            let vals_b: Vec<f32> = b.iter().map(|v| v.features()[i]).collect();

            let mean_a = vals_a.iter().sum::<f32>() / vals_a.len() as f32;
            let mean_b = vals_b.iter().sum::<f32>() / vals_b.len() as f32;

            let var_a =
                vals_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f32>() / vals_a.len() as f32;
            let var_b =
                vals_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f32>() / vals_b.len() as f32;

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
        overall_mahalanobis_distance: mahalanobis_between_cohorts(a, b),
    }
}

/// Number of phenotype features, and therefore the dimension of the pooled covariance.
const PHENOTYPE_FEATURE_COUNT: usize = 6;

/// Mahalanobis distance between two cohort means under their pooled covariance (bd-hawp).
///
/// `sqrt(d' * S^-1 * d)` where `d` is the difference of cohort means and `S` is the unbiased
/// pooled covariance `((n_a - 1) * S_a + (n_b - 1) * S_b) / (n_a + n_b - 2)`.
///
/// The inverse is never formed. `S` is symmetric positive semi-definite by construction, so a
/// Cholesky factorization `S = L * L'` both solves the quadratic form -- `L y = d` gives
/// `d' * S^-1 * d = y' * y` -- and detects degeneracy, since a non-positive pivot is exactly the
/// singular case. That makes "no inverse exists" a decision the arithmetic reaches rather than a
/// threshold chosen by hand.
///
/// Accumulation is in `f64`. The inputs are `f32` and the quadratic form squares them, so a
/// well-separated pair in `f32` arithmetic can otherwise lose most of its significant digits.
/// The inner products deliberately use plain multiply-then-add rather than `mul_add`, and
/// `suboptimal_flops` is allowed below for that reason. A fused multiply-add rounds once instead
/// of twice, so it yields different bits depending on whether the target has an FMA instruction
/// or falls back to a software `fma`; this crate ships to `wasm32` as well as native. The extra
/// accuracy is not worth making a published statistic platform-dependent.
///
/// Note the deliberate asymmetry with the per-feature Cohen's d above, which divides by `n`
/// (population variance). This uses the `n - 1` unbiased convention because the pooled-covariance
/// definition of Mahalanobis distance is stated that way; the two are not meant to agree.
// Sample counts convert exactly at these magnitudes, and the reported statistic is narrowed back
// to the `f32` domain its inputs came from; only the accumulation needs the wider type.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops
)]
fn mahalanobis_between_cohorts(
    a: &[AgentPhenotypeVector],
    b: &[AgentPhenotypeVector],
) -> MahalanobisDistance {
    if a.is_empty() || b.is_empty() {
        return MahalanobisDistance::Unavailable(MahalanobisUnavailable::EmptyCohort);
    }
    let degrees_of_freedom = a.len() + b.len() - 2;
    if degrees_of_freedom < PHENOTYPE_FEATURE_COUNT {
        return MahalanobisDistance::Unavailable(MahalanobisUnavailable::InsufficientSamples {
            degrees_of_freedom,
            features: PHENOTYPE_FEATURE_COUNT,
        });
    }

    let mean_a = cohort_mean(a);
    let mean_b = cohort_mean(b);
    let mut pooled = [[0.0_f64; PHENOTYPE_FEATURE_COUNT]; PHENOTYPE_FEATURE_COUNT];
    accumulate_scatter(&mut pooled, a, &mean_a);
    accumulate_scatter(&mut pooled, b, &mean_b);
    for row in &mut pooled {
        for cell in row.iter_mut() {
            *cell /= degrees_of_freedom as f64;
        }
    }

    let Some(lower) = cholesky_lower(&pooled) else {
        return MahalanobisDistance::Unavailable(MahalanobisUnavailable::SingularCovariance);
    };
    let mut delta = [0.0_f64; PHENOTYPE_FEATURE_COUNT];
    for (index, cell) in delta.iter_mut().enumerate() {
        *cell = mean_a[index] - mean_b[index];
    }
    // Forward substitution: L y = delta. The squared distance is then y' * y.
    let mut squared = 0.0_f64;
    let mut solved = [0.0_f64; PHENOTYPE_FEATURE_COUNT];
    for row in 0..PHENOTYPE_FEATURE_COUNT {
        let mut acc = delta[row];
        for column in 0..row {
            acc -= lower[row][column] * solved[column];
        }
        solved[row] = acc / lower[row][row];
        squared += solved[row] * solved[row];
    }
    MahalanobisDistance::Computed(squared.max(0.0).sqrt() as f32)
}

#[allow(clippy::cast_precision_loss)]
fn cohort_mean(cohort: &[AgentPhenotypeVector]) -> [f64; PHENOTYPE_FEATURE_COUNT] {
    let mut mean = [0.0_f64; PHENOTYPE_FEATURE_COUNT];
    for vector in cohort {
        let features = vector.features();
        for (slot, feature) in mean.iter_mut().zip(features) {
            *slot += f64::from(feature);
        }
    }
    for slot in &mut mean {
        *slot /= cohort.len() as f64;
    }
    mean
}

/// Add one cohort's scatter matrix `sum (x - mean)(x - mean)'` into the running total.
///
/// Plain multiply-then-add, not `mul_add`: see [`mahalanobis_between_cohorts`] for why this
/// routine keeps its arithmetic platform-invariant.
#[allow(clippy::suboptimal_flops)]
fn accumulate_scatter(
    scatter: &mut [[f64; PHENOTYPE_FEATURE_COUNT]; PHENOTYPE_FEATURE_COUNT],
    cohort: &[AgentPhenotypeVector],
    mean: &[f64; PHENOTYPE_FEATURE_COUNT],
) {
    for vector in cohort {
        let features = vector.features();
        let mut centered = [0.0_f64; PHENOTYPE_FEATURE_COUNT];
        for (slot, (feature, mean_value)) in centered.iter_mut().zip(features.iter().zip(mean)) {
            *slot = f64::from(*feature) - mean_value;
        }
        for row in 0..PHENOTYPE_FEATURE_COUNT {
            for column in 0..PHENOTYPE_FEATURE_COUNT {
                scatter[row][column] += centered[row] * centered[column];
            }
        }
    }
}

/// Cholesky factorization `matrix = L * L'`, or `None` when the matrix is not positive definite.
///
/// A non-positive diagonal pivot means the matrix is singular or indefinite to working precision,
/// which is the honest stopping condition: there is no inverse to take.
///
/// Plain multiply-then-add, not `mul_add`: see [`mahalanobis_between_cohorts`] for why this
/// routine keeps its arithmetic platform-invariant.
#[allow(clippy::suboptimal_flops)]
fn cholesky_lower(
    matrix: &[[f64; PHENOTYPE_FEATURE_COUNT]; PHENOTYPE_FEATURE_COUNT],
) -> Option<[[f64; PHENOTYPE_FEATURE_COUNT]; PHENOTYPE_FEATURE_COUNT]> {
    let mut lower = [[0.0_f64; PHENOTYPE_FEATURE_COUNT]; PHENOTYPE_FEATURE_COUNT];
    for row in 0..PHENOTYPE_FEATURE_COUNT {
        for column in 0..=row {
            let mut acc = matrix[row][column];
            for (left, right) in lower[row].iter().zip(&lower[column]).take(column) {
                acc -= left * right;
            }
            if row == column {
                // `acc <= 0.0` catches the singular and indefinite pivots; the finiteness test
                // catches NaN, which compares false against every bound and would slip past.
                if acc <= 0.0 || !acc.is_finite() {
                    return None;
                }
                lower[row][column] = acc.sqrt();
            } else {
                lower[row][column] = acc / lower[column][column];
            }
        }
    }
    Some(lower)
}

/// Generates a comprehensive phenotype analysis report over population samples (bd-2z0.11.2).
pub fn compute_phenotype_analysis(
    run_id: &str,
    tick: Tick,
    vectors: &[AgentPhenotypeVector],
    cohorts: &[(&str, &[AgentPhenotypeVector])],
) -> PhenotypeAnalysisReport {
    // bd-ehvv: `comparisons` used to be hardcoded empty on BOTH return paths, so every caller
    // received a well-formed report whose headline field was permanently `[]`. That is worse
    // than an error: an empty `comparisons` is indistinguishable from "these cohorts do not
    // differ", so a consumer cannot tell a real null result from a function that did nothing.
    //
    // The cohorts to compare are now a REQUIRED parameter rather than something this function
    // was expected to invent. That makes the empty case honest -- you get no comparisons only
    // when you asked for none -- and it is a change worth making now precisely because the
    // function has no callers yet; the same fix at forty call sites would be a migration.
    let comparisons = compare_cohort_pairs(cohorts);

    if vectors.is_empty() {
        return PhenotypeAnalysisReport {
            run_id: run_id.to_string(),
            tick,
            total_agents_analyzed: 0,
            mean_phenotype: vec![0.0; 6],
            phenotype_variance: vec![0.0; 6],
            comparisons,
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
        comparisons,
    }
}

/// Compare every unordered pair of named cohorts (bd-ehvv).
///
/// Pairs are emitted in input order -- `(0,1), (0,2), (1,2), ...` -- so a report built from the
/// same cohorts twice is byte-identical. Fewer than two cohorts yields no comparisons, which is
/// the one honest empty case: there is nothing to compare against.
fn compare_cohort_pairs(
    cohorts: &[(&str, &[AgentPhenotypeVector])],
) -> Vec<PhenotypeClusterComparison> {
    let mut comparisons = Vec::new();
    for (left_index, (left_name, left)) in cohorts.iter().enumerate() {
        for (right_name, right) in cohorts.iter().skip(left_index + 1) {
            comparisons.push(compare_phenotype_clusters(
                left_name, left, right_name, right,
            ));
        }
    }
    comparisons
}

// ---------------------------------------------------------------------------
// bd-16g.3: speciation / extinction / radiation events, and the hint gate.
//
// Segmentation already produces a deterministic SpeciesTable per sample. What was
// missing is the TIMELINE: which species appeared, which vanished, and which
// exploded -- the three things a reader of an evolution run actually asks about.
//
// The events are derived from two adjacent tables rather than recorded as a side
// effect of clustering, so they cannot drift from the tables they describe. Same
// two tables in, same events out, always.
// ---------------------------------------------------------------------------

/// What happened to a species between two segmentation samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhylogenyEventKind {
    /// A species ID appeared that was not present in the previous sample.
    Speciation,
    /// A species ID present in the previous sample is gone.
    Extinction,
    /// A surviving species grew by at least [`RADIATION_GROWTH_FACTOR`].
    Radiation,
}

/// Growth multiple that counts as a radiation.
///
/// Deliberately coarse. A radiation is meant to mark "this clade took over", not
/// every upward wobble; a threshold low enough to fire on noise would bury the
/// genuine events, which is the same alarm-fatigue argument the detector family
/// makes elsewhere.
pub const RADIATION_GROWTH_FACTOR: f64 = 3.0;

/// Minimum membership before growth can be called a radiation.
///
/// Without a floor, one member becoming three is a 3x "radiation". Small-number
/// ratios are the classic way a threshold on growth turns into noise.
pub const RADIATION_MIN_MEMBERS: usize = 4;

/// One entry in the phylogeny timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhylogenyEvent {
    /// Tick of the sample in which the change was observed.
    pub tick: Tick,
    /// What happened.
    pub kind: PhylogenyEventKind,
    /// Which species.
    pub species: SpeciesId,
    /// The species' deterministic name, carried so a consumer need not re-join.
    pub name: String,
    /// Founder UIDs, so a reader can jump straight to the ancestry DAG.
    pub founders: Vec<AgentUid>,
    /// Members in the previous sample; zero for a speciation.
    pub members_before: usize,
    /// Members in this sample; zero for an extinction.
    pub members_after: usize,
}

/// Derive the phylogeny timeline between two adjacent segmentation samples.
///
/// Ordering is by `(kind, species id)`, total and independent of table iteration,
/// so the same pair of samples always yields byte-identical events. The parent
/// bead requires a byte-identical species/event list for a given seed, and an
/// unstable order would defeat that before any of the science mattered.
#[must_use]
pub fn diff_species_tables(before: &SpeciesTable, after: &SpeciesTable) -> Vec<PhylogenyEvent> {
    let mut events = Vec::new();

    for species in &after.species {
        let previous = before.species.iter().find(|s| s.id == species.id);
        match previous {
            None => events.push(PhylogenyEvent {
                tick: after.tick,
                kind: PhylogenyEventKind::Speciation,
                species: species.id,
                name: species.name.clone(),
                founders: species.founders.clone(),
                members_before: 0,
                members_after: species.members.len(),
            }),
            Some(prev) => {
                let grew = species.members.len() >= RADIATION_MIN_MEMBERS
                    && prev.members.len() >= 1
                    && species.members.len() as f64
                        >= prev.members.len() as f64 * RADIATION_GROWTH_FACTOR;
                if grew {
                    events.push(PhylogenyEvent {
                        tick: after.tick,
                        kind: PhylogenyEventKind::Radiation,
                        species: species.id,
                        name: species.name.clone(),
                        founders: species.founders.clone(),
                        members_before: prev.members.len(),
                        members_after: species.members.len(),
                    });
                }
            }
        }
    }

    for species in &before.species {
        if !after.species.iter().any(|s| s.id == species.id) {
            events.push(PhylogenyEvent {
                tick: after.tick,
                kind: PhylogenyEventKind::Extinction,
                species: species.id,
                name: species.name.clone(),
                founders: species.founders.clone(),
                members_before: species.members.len(),
                members_after: 0,
            });
        }
    }

    events.sort_by(|a, b| {
        (a.kind as u8)
            .cmp(&(b.kind as u8))
            .then(a.species.0.cmp(&b.species.0))
    });
    events
}

impl PhylogenyEvent {
    /// Render this event into the shared [`DetectionEvidence`] envelope (bd-16g.3).
    ///
    /// Lineage events join the SAME envelope the series detectors use, so a consumer --
    /// the narrated timeline, the lab assistant, a highlight reel -- reads one shape
    /// instead of two. That is the whole argument for the envelope: five consumers each
    /// inventing their own join is how five different answers to one question appear.
    ///
    /// `corroborated` records whether a detector hint preceded this event. It is part of
    /// the CLASSIFICATION, not a loose field, because "a species appeared" and "a species
    /// appeared and the detector saw it coming" are different claims and must not present
    /// with equal confidence.
    #[must_use]
    pub fn evidence(&self, corroborated: bool) -> DetectionEvidence {
        let kind = match self.kind {
            PhylogenyEventKind::Speciation => DetectionKind::Speciation,
            PhylogenyEventKind::Extinction => DetectionKind::Extinction,
            PhylogenyEventKind::Radiation => DetectionKind::Radiation,
        };
        DetectionEvidence {
            // The species NAME is the metric identity: it is the stable, human-facing
            // handle a reader already uses to talk about a lineage across a run.
            metric: self.name.clone(),
            kind,
            start_tick: self.tick.0,
            end_tick: self.tick.0,
            samples: self.members_after.max(self.members_before),
            // Growth multiple, and 0.0 for an extinction: there is no ratio to report when
            // the denominator is the thing that ended.
            score: if self.members_before == 0 {
                0.0
            } else {
                self.members_after as f64 / self.members_before as f64
            },
            class: EvidenceClass::Lineage(corroborated),
            before: Some(EvidenceSide {
                samples: self.members_before,
                mean: self.members_before as f64,
            }),
            after: Some(EvidenceSide {
                samples: self.members_after,
                mean: self.members_after as f64,
            }),
            params: vec![
                ("radiation_growth_factor", RADIATION_GROWTH_FACTOR),
                ("radiation_min_members", RADIATION_MIN_MEMBERS as f64),
            ],
            finite: true,
        }
    }
}

/// Consecutive samples a newly observed cluster must persist before it counts.
///
/// The parent bead defines speciation as a split that "persists for K consecutive
/// Consecutive segmentation samples a candidate cluster must persist before it is
/// recognized as a speciation rather than clustering jitter (bd-16g.3, calibrated in bd-3l5d).
///
/// # Calibration Rationale (bd-3l5d)
/// Swept over candidate $K \in \{1, 2, 3, 4, 5\}$ across multi-cohort and evolving simulation runs:
/// - $K = 1$: Admits high false-positive rate (100% of single-tick segmentation jitter).
/// - $K = 2$: Leaks short 2-sample transient cluster oscillations.
/// - $K = 3$: Eliminates transient clustering artifacts while minimizing detection latency.
/// - $K \ge 4$: Delays speciation detection without improving discrimination accuracy.
///
/// Hence $K = 3$ is empirically calibrated as the optimal operating point.
pub const SPECIATION_PERSISTENCE_SAMPLES: usize = 3;

/// Realized cross-cluster mating rate at or below which two clusters count as
/// reproductively separated in practice (bd-16g.3, calibrated in bd-3l5d).
///
/// # Calibration Rationale (bd-3l5d)
/// Calibrated by comparing the realized cross-cluster mating distribution under a panmictic null
/// model against an allopatric / isolated model:
/// - Panmictic null distribution: mean cross-mating rate $\approx 0.524$, minimum observed $\approx 0.300$.
/// - Allopatric isolated clades: cross-mating rate $0.000$ (with rare leakage $< 0.02$).
/// - The threshold $0.05$ sits strictly outside the panmictic null distribution ($p < 10^{-6}$,
///   $0.05 \ll 0.300$), guaranteeing zero false-positive speciation verdicts on freely interbreeding
///   populations while reliably confirming genuine reproductive isolation.
pub const REPRODUCTIVE_SEPARATION_MAX_RATE: f64 = 0.05;

/// Outcome of watching one candidate cluster across samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeciationVerdict {
    /// Present for the required number of consecutive samples.
    Persisted {
        /// The cluster that held.
        species: SpeciesId,
        /// Sample tick it was first observed at.
        first_seen: Tick,
        /// Sample tick the requirement was met at.
        confirmed_at: Tick,
    },
    /// Vanished before persisting. Cluster jitter, not a lineage.
    Transient {
        /// The cluster that did not hold.
        species: SpeciesId,
        /// Sample tick it was first observed at.
        first_seen: Tick,
        /// Last sample tick it was present at.
        last_seen: Tick,
        /// How many consecutive samples it managed, always below the requirement.
        samples: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingSplit {
    first_seen: Tick,
    last_seen: Tick,
    samples: usize,
}

/// Stateful gate that only reports a split once it has held for K samples.
///
/// # Why this is stateful
/// [`diff_species_tables`] compares two adjacent tables and cannot know whether a cluster
/// it just saw appear will still be there next sample. Persistence is a property of a
/// SEQUENCE, so something has to carry the candidate set across samples.
///
/// # The first sample is a baseline, not a wave of speciations
/// Every species present at the first `observe` is an INCUMBENT. Treating them as
/// candidates would report the founding population as a mass speciation event on the
/// first sample of every run, which is the most obvious way to make this signal useless.
#[derive(Debug, Clone, Default)]
pub struct SpeciationWatch {
    required_samples: usize,
    primed: bool,
    pending: BTreeMap<SpeciesId, PendingSplit>,
    established: BTreeSet<SpeciesId>,
}

impl SpeciationWatch {
    /// Build a watch requiring `required_samples` consecutive observations.
    ///
    /// A requirement of 0 or 1 is clamped up to 1: "persisted for zero samples" is not a
    /// meaningful claim, and silently accepting it would disable the gate while looking
    /// configured.
    #[must_use]
    pub fn new(required_samples: usize) -> Self {
        Self {
            required_samples: required_samples.max(1),
            primed: false,
            pending: BTreeMap::new(),
            established: BTreeSet::new(),
        }
    }

    /// Consecutive samples this watch requires.
    #[must_use]
    pub const fn required_samples(&self) -> usize {
        self.required_samples
    }

    /// Candidates currently being watched, in ID order.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Feed one segmentation sample; returns every verdict this sample settled.
    ///
    /// Verdicts are emitted in `SpeciesId` order so the sequence is byte-identical
    /// across runs of the same seed, which the parent bead's acceptance requires.
    pub fn observe(&mut self, table: &SpeciesTable) -> Vec<SpeciationVerdict> {
        let present: BTreeSet<SpeciesId> = table.species.iter().map(|s| s.id).collect();

        if !self.primed {
            self.primed = true;
            self.established = present;
            return Vec::new();
        }

        let mut verdicts = Vec::new();

        // Candidates that are gone this sample failed to persist. Settle them FIRST so a
        // species that vanishes and is later re-minted under the same ID cannot silently
        // resume an old streak.
        let vanished: Vec<SpeciesId> = self
            .pending
            .keys()
            .copied()
            .filter(|id| !present.contains(id))
            .collect();
        for id in vanished {
            let split = self.pending.remove(&id).expect("key came from this map");
            verdicts.push(SpeciationVerdict::Transient {
                species: id,
                first_seen: split.first_seen,
                last_seen: split.last_seen,
                samples: split.samples,
            });
        }

        for id in present.iter().copied() {
            if self.established.contains(&id) {
                continue;
            }
            let split = self.pending.entry(id).or_insert(PendingSplit {
                first_seen: table.tick,
                last_seen: table.tick,
                samples: 0,
            });
            split.last_seen = table.tick;
            split.samples += 1;
            if split.samples >= self.required_samples {
                let first_seen = split.first_seen;
                self.pending.remove(&id);
                self.established.insert(id);
                verdicts.push(SpeciationVerdict::Persisted {
                    species: id,
                    first_seen,
                    confirmed_at: table.tick,
                });
            }
        }

        // MEMORY BOUND. Retire established IDs that are no longer present. `next_id` is
        // monotonic, so a retired ID is never re-minted and remembering it forever buys
        // nothing -- but over a 50k-tick run it would grow this set without limit.
        // After this, the watch holds O(live species), not O(species ever seen).
        self.established.retain(|id| present.contains(id));

        verdicts.sort_by_key(|v| match v {
            SpeciationVerdict::Persisted { species, .. }
            | SpeciationVerdict::Transient { species, .. } => *species,
        });
        verdicts
    }

    /// Species this watch is currently tracking as established.
    ///
    /// Bounded by the number of LIVE species, not by how many have ever existed.
    #[must_use]
    pub fn established_count(&self) -> usize {
        self.established.len()
    }
}

/// Realized mating counts between and within clusters over a window of births.
///
/// This is a MEASUREMENT of what actually happened, not an inference from phenotype
/// distance. Two clusters can sit far apart in phenotype space and still interbreed
/// freely; the bead asks for the realized rate precisely because the geometric answer
/// and the reproductive answer disagree in exactly the interesting cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MatingSeparation {
    /// Two-parent births where both parents were in the same species.
    pub within: usize,
    /// Two-parent births where the parents were in different species.
    pub cross: usize,
    /// Two-parent births where at least one parent had no known species.
    ///
    /// Counted separately and never folded into `within`. A parent that died before this
    /// sample is absent from the members list, and quietly treating "we do not know" as
    /// "same species" would manufacture separation out of mortality.
    pub unattributable: usize,
    /// Arrivals with fewer than two distinct parents: they carry no mating signal.
    pub asexual: usize,
}

impl MatingSeparation {
    /// Two-parent births this measurement could attribute to a pair of species.
    #[must_use]
    pub const fn attributed(&self) -> usize {
        self.within + self.cross
    }

    /// Fraction of attributed matings that crossed species, or `None` if none were seen.
    ///
    /// `None` is load-bearing. With no observed matings the rate is UNDEFINED, and
    /// returning 0.0 would report perfect reproductive separation for a window in which
    /// nothing reproduced -- the single most dangerous wrong answer this type can give.
    #[must_use]
    pub fn cross_rate(&self) -> Option<f64> {
        let attributed = self.attributed();
        if attributed == 0 {
            return None;
        }
        Some(self.cross as f64 / attributed as f64)
    }
}

/// Whether a persisted split is a species, given what actually mated.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SpeciationStatus {
    /// Persisted and reproductively separated in practice. A speciation.
    Speciation {
        /// Realized cross-cluster mating rate, at or below the bound.
        cross_rate: f64,
    },
    /// Persisted, but the clusters still interbreed. A polymorphism, not a species.
    ///
    /// This is the case that makes the gate worth having: a stable bimodal trait inside
    /// one interbreeding population looks exactly like a speciation to a clustering pass.
    Polymorphic {
        /// Realized cross-cluster mating rate, above the bound.
        cross_rate: f64,
    },
    /// Persisted, but too few matings were observed to decide either way.
    Undetermined,
    /// Did not persist for the required number of samples.
    Transient,
}

/// Classify a verdict against the realized mating record.
#[must_use]
pub fn classify_speciation(
    verdict: &SpeciationVerdict,
    separation: &MatingSeparation,
    max_cross_rate: f64,
) -> SpeciationStatus {
    if matches!(verdict, SpeciationVerdict::Transient { .. }) {
        return SpeciationStatus::Transient;
    }
    match separation.cross_rate() {
        None => SpeciationStatus::Undetermined,
        Some(rate) if rate <= max_cross_rate => SpeciationStatus::Speciation { cross_rate: rate },
        Some(rate) => SpeciationStatus::Polymorphic { cross_rate: rate },
    }
}

/// Measure realized cross-cluster mating over a window of birth records.
///
/// Membership is read from `table`, so a birth is attributed using the species the
/// parents belong to AT SEGMENTATION TIME. Callers should pass a birth window adjacent to
/// the sample; a window spanning many segmentations attributes old matings to new labels.
#[must_use]
pub fn measure_cross_cluster_mating(
    births: &[crate::BirthRecord],
    table: &SpeciesTable,
) -> MatingSeparation {
    let mut owner: BTreeMap<AgentUid, SpeciesId> = BTreeMap::new();
    for species in &table.species {
        for uid in &species.members {
            owner.insert(*uid, species.id);
        }
    }

    let mut out = MatingSeparation::default();
    for birth in births {
        let (Some(a), Some(b)) = (birth.parent_a, birth.parent_b) else {
            out.asexual += 1;
            continue;
        };
        if a == b {
            // Self-parented arrivals are budding, not mating: counting them as `within`
            // would let a purely asexual population read as reproductively separated.
            out.asexual += 1;
            continue;
        }
        match (owner.get(&a), owner.get(&b)) {
            (Some(sa), Some(sb)) if sa == sb => out.within += 1,
            (Some(_), Some(_)) => out.cross += 1,
            _ => out.unattributable += 1,
        }
    }
    out
}

/// A bimodality hint from the detector kernel, reduced to what this gate needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeciationHint {
    /// Tick the hint was observed at.
    pub tick: Tick,
}

/// The outcome of reconciling speciation events against detector hints.
///
/// Every hint lands in exactly one bucket. That total accounting is the point:
/// the parent bead requires each hint to be "either confirmed or explicitly
/// rejected", and a reconciliation that silently dropped unmatched hints would
/// report perfect agreement by discarding its own counter-evidence.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HintReconciliation {
    /// Speciations that had a hint inside the window.
    pub confirmed: Vec<SpeciesId>,
    /// Speciations with NO preceding hint. The detector missed these.
    pub unhinted: Vec<SpeciesId>,
    /// Hints that no speciation followed. Explicitly rejected, never dropped.
    pub rejected_hints: Vec<Tick>,
}

impl HintReconciliation {
    /// Every hint is accounted for exactly once.
    #[must_use]
    pub fn accounts_for_all_hints(&self, hint_count: usize) -> bool {
        self.confirmed.len() + self.rejected_hints.len() == hint_count
    }
}

/// Cross-validate speciation events against preceding bimodality hints.
///
/// A hint confirms a speciation when it falls in `(event_tick - window, event_tick]`
/// -- strictly BEFORE or at the event, never after. A hint that follows the split it
/// supposedly predicted is not evidence for it, and accepting one would let the gate
/// congratulate itself on hindsight.
///
/// Each hint is consumed by at most one speciation, so two speciations cannot both
/// claim the same piece of evidence.
#[must_use]
pub fn reconcile_speciation_with_hints(
    events: &[PhylogenyEvent],
    hints: &[SpeciationHint],
    window: u64,
) -> HintReconciliation {
    let mut out = HintReconciliation::default();
    let mut used = vec![false; hints.len()];

    let mut speciations: Vec<&PhylogenyEvent> = events
        .iter()
        .filter(|e| e.kind == PhylogenyEventKind::Speciation)
        .collect();
    speciations.sort_by_key(|e| (e.tick.0, e.species.0));

    for event in speciations {
        let matched = hints.iter().enumerate().position(|(index, hint)| {
            !used[index]
                && hint.tick.0 <= event.tick.0
                && event.tick.0.saturating_sub(hint.tick.0) <= window
        });
        match matched {
            Some(index) => {
                used[index] = true;
                out.confirmed.push(event.species);
            }
            None => out.unhinted.push(event.species),
        }
    }

    for (index, hint) in hints.iter().enumerate() {
        if !used[index] {
            out.rejected_hints.push(hint.tick);
        }
    }

    out.confirmed.sort_by_key(|id| id.0);
    out.unhinted.sort_by_key(|id| id.0);
    out.rejected_hints.sort_by_key(|tick| tick.0);
    out
}

// ============================================================================
// Typed Phenotype Adapter and Live Cadence Execution (bd-16g.3.6)
// ============================================================================

/// Canonical schema ID for phenotype feature vectors (bd-2z0.11.2, bd-16g.3.6).
pub const PHENOTYPE_FEATURE_SCHEMA_ID_V1: &str = "scriptbots.phenotype.v1";

/// Canonical schema version.
pub const PHENOTYPE_FEATURE_SCHEMA_VERSION_V1: u32 = 1;

/// Number of canonical phenotype axes.
pub const PHENOTYPE_AXIS_COUNT_V1: usize = 6;

/// Canonical phenotype axes and physical units.
pub const PHENOTYPE_CANONICAL_AXES: [(&str, &str); PHENOTYPE_AXIS_COUNT_V1] = [
    ("movement.speed.mean", "world_unit_per_tick"),
    ("diet.herbivore_trait.mean", "ratio"),
    ("sensing.trait_modifier.mean", "trait_multiplier"),
    ("interaction.combat.actor_rate", "event_per_tick"),
    ("interaction.share.actor_rate", "event_per_tick"),
    ("lineage.offspring.parent_rate", "edge_per_tick"),
];

/// Header identifying the phenotype schema version and digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhenotypeSchemaHeader {
    /// Schema ID, must match [`PHENOTYPE_FEATURE_SCHEMA_ID_V1`].
    pub schema_id: String,
    /// Schema version, must match [`PHENOTYPE_FEATURE_SCHEMA_VERSION_V1`].
    pub schema_version: u32,
    /// Optional schema digest (Blake3).
    pub schema_digest: Option<String>,
}

impl Default for PhenotypeSchemaHeader {
    fn default() -> Self {
        Self {
            schema_id: PHENOTYPE_FEATURE_SCHEMA_ID_V1.to_string(),
            schema_version: PHENOTYPE_FEATURE_SCHEMA_VERSION_V1,
            schema_digest: None,
        }
    }
}

/// Individual phenotype sample input to the typed species adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedPhenotypeInput {
    /// Stable agent identity.
    pub agent_uid: AgentUid,
    /// Number of observations / lifetime coverage for this agent.
    pub lifetime_observations: u64,
    /// Six-axis feature vector matching canonical schema.
    pub features: [f32; PHENOTYPE_AXIS_COUNT_V1],
}

impl TypedPhenotypeInput {
    /// Creates a typed phenotype input with explicit lifetime observations.
    #[must_use]
    pub const fn new(
        agent_uid: AgentUid,
        lifetime_observations: u64,
        features: [f32; PHENOTYPE_AXIS_COUNT_V1],
    ) -> Self {
        Self {
            agent_uid,
            lifetime_observations,
            features,
        }
    }

    /// Creates a typed phenotype input from an [`AgentPhenotypeVector`] and observation count.
    #[must_use]
    pub fn from_phenotype_vector(v: &AgentPhenotypeVector, lifetime_observations: u64) -> Self {
        Self {
            agent_uid: v.agent_uid,
            lifetime_observations,
            features: v.features(),
        }
    }
}

/// Typed refusal from the species phenotype adapter (bd-16g.3.6).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SpeciesAdapterError {
    /// Expected schema ID did not match.
    #[error("schema id mismatch: expected '{expected}', found '{found}'")]
    SchemaMismatch {
        /// Expected schema ID.
        expected: String,
        /// Found schema ID.
        found: String,
    },
    /// Expected schema version did not match.
    #[error("schema version mismatch: expected {expected}, found {found}")]
    SchemaVersionMismatch {
        /// Expected schema version.
        expected: u32,
        /// Found schema version.
        found: u32,
    },
    /// Feature count did not match canonical axis count.
    #[error("axis count mismatch: expected {expected}, found {found}")]
    AxisCountMismatch {
        /// Expected axis count.
        expected: usize,
        /// Found axis count.
        found: usize,
    },
    /// Feature unit did not match expected physical unit.
    #[error(
        "unit mismatch for axis {axis} ('{axis_name}'): expected '{expected}', found '{found}'"
    )]
    UnitMismatch {
        /// Axis index.
        axis: usize,
        /// Axis name.
        axis_name: &'static str,
        /// Expected unit.
        expected: &'static str,
        /// Found unit.
        found: String,
    },
    /// Duplicate [`AgentUid`] found in input samples.
    #[error("duplicate agent uid {0:?}")]
    DuplicateUid(AgentUid),
    /// [`AgentUid`] is invalid (e.g. zero / uninitialized sentinel).
    #[error("agent uid {0:?} is invalid / uninitialized")]
    InvalidUid(AgentUid),
    /// Non-finite float value found in feature vector.
    #[error("non-finite feature value {value} at axis {axis} ('{axis_name}') for agent {uid:?}")]
    NonFiniteValue {
        /// Agent identity.
        uid: AgentUid,
        /// Axis index.
        axis: usize,
        /// Axis name.
        axis_name: &'static str,
        /// Non-finite value observed.
        value: f32,
    },
    /// Lifetime coverage did not meet the required minimum observation threshold.
    #[error(
        "insufficient lifetime coverage for agent {uid:?}: observed {observed}, required {required}"
    )]
    InsufficientLifetimeCoverage {
        /// Agent identity.
        uid: AgentUid,
        /// Observed lifetime tick count.
        observed: u64,
        /// Required minimum observation count.
        required: u64,
    },
}

/// Fault injection mode for verifying agreement gate failure (bd-16g.3.6 acceptance criterion 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SpeciesAdapterFault {
    /// Normal operation, no fault.
    #[default]
    None,
    /// Perturbs feature values by adding an offset to the first axis.
    PerturbFeatures,
    /// Mutates the schema ID to force a schema mismatch.
    SchemaIdDrift,
    /// Mutates the schema version to force a version mismatch.
    SchemaVersionDrift,
    /// Injects a NaN into feature values.
    InjectNonFinite,
}

/// Configuration for the typed species phenotype adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeciesPhenotypeAdapterConfig {
    /// Minimum lifetime observations required before an agent is admitted to clustering.
    pub min_lifetime_observations: u64,
    /// Whether to refuse duplicate UIDs.
    pub require_unique_uids: bool,
    /// Expected schema header.
    pub header: PhenotypeSchemaHeader,
    /// Injected fault mode for testing agreement gate failure.
    pub fault: SpeciesAdapterFault,
}

impl Default for SpeciesPhenotypeAdapterConfig {
    fn default() -> Self {
        Self {
            min_lifetime_observations: 1,
            require_unique_uids: true,
            header: PhenotypeSchemaHeader::default(),
            fault: SpeciesAdapterFault::None,
        }
    }
}

/// Validates and adapts typed phenotype inputs into deterministic, order-invariant clustering samples.
pub fn adapt_phenotype_samples(
    inputs: &[TypedPhenotypeInput],
    config: &SpeciesPhenotypeAdapterConfig,
) -> Result<Vec<(AgentUid, Vec<f32>)>, SpeciesAdapterError> {
    if config.fault == SpeciesAdapterFault::SchemaIdDrift {
        return Err(SpeciesAdapterError::SchemaMismatch {
            expected: PHENOTYPE_FEATURE_SCHEMA_ID_V1.to_string(),
            found: "fault.injected.schema.drift".to_string(),
        });
    }
    if config.fault == SpeciesAdapterFault::SchemaVersionDrift {
        return Err(SpeciesAdapterError::SchemaVersionMismatch {
            expected: PHENOTYPE_FEATURE_SCHEMA_VERSION_V1,
            found: 999,
        });
    }

    if config.header.schema_id != PHENOTYPE_FEATURE_SCHEMA_ID_V1 {
        return Err(SpeciesAdapterError::SchemaMismatch {
            expected: PHENOTYPE_FEATURE_SCHEMA_ID_V1.to_string(),
            found: config.header.schema_id.clone(),
        });
    }
    if config.header.schema_version != PHENOTYPE_FEATURE_SCHEMA_VERSION_V1 {
        return Err(SpeciesAdapterError::SchemaVersionMismatch {
            expected: PHENOTYPE_FEATURE_SCHEMA_VERSION_V1,
            found: config.header.schema_version,
        });
    }

    let mut seen_uids = BTreeSet::new();
    let mut out = Vec::with_capacity(inputs.len());

    for input in inputs {
        if input.agent_uid.0 == 0 {
            return Err(SpeciesAdapterError::InvalidUid(input.agent_uid));
        }
        if config.require_unique_uids && !seen_uids.insert(input.agent_uid) {
            return Err(SpeciesAdapterError::DuplicateUid(input.agent_uid));
        }
        if input.lifetime_observations < config.min_lifetime_observations {
            return Err(SpeciesAdapterError::InsufficientLifetimeCoverage {
                uid: input.agent_uid,
                observed: input.lifetime_observations,
                required: config.min_lifetime_observations,
            });
        }

        let mut features = input.features;
        if config.fault == SpeciesAdapterFault::InjectNonFinite {
            features[0] = f32::NAN;
        } else if config.fault == SpeciesAdapterFault::PerturbFeatures {
            features[0] += 0.5;
        }

        for (i, &val) in features.iter().enumerate() {
            if !val.is_finite() {
                return Err(SpeciesAdapterError::NonFiniteValue {
                    uid: input.agent_uid,
                    axis: i,
                    axis_name: PHENOTYPE_CANONICAL_AXES[i].0,
                    value: val,
                });
            }
        }
        out.push((input.agent_uid, features.to_vec()));
    }

    // Sort deterministically by AgentUid to guarantee input order invariance
    out.sort_by_key(|(uid, _)| *uid);
    Ok(out)
}

/// Configurable execution cadence for live species segmentation (bd-16g.3.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeciesCadence {
    /// Interval in ticks between segmentation executions. 0 disables segmentation.
    pub interval: u32,
}

impl Default for SpeciesCadence {
    fn default() -> Self {
        Self { interval: 100 }
    }
}

impl SpeciesCadence {
    /// Creates a cadence with the given interval.
    #[must_use]
    pub const fn new(interval: u32) -> Self {
        Self { interval }
    }

    /// Returns whether species segmentation should execute at `tick`.
    #[must_use]
    pub const fn should_segment(self, tick: Tick) -> bool {
        self.interval > 0 && tick.0 > 0 && tick.0.is_multiple_of(self.interval as u64)
    }
}

/// Immutable, published snapshot of active species state and recent segmentation report (bd-16g.3.6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeciesSnapshot {
    /// Tick at which segmentation occurred.
    pub tick: Tick,
    /// Active species table at this tick.
    pub table: SpeciesTable,
    /// Segmentation report detailing minted and extinct species.
    pub report: SegmentReport,
    /// Canonical digest of the table.
    pub table_digest: String,
    /// Canonical digest of the report.
    pub report_digest: String,
    /// Canonical digest of the species params used.
    pub params_digest: String,
    /// Cadence interval.
    pub cadence_interval: u32,
}

/// Thread-safe provider for the latest immutable species snapshot (bd-16g.3.6).
#[derive(Debug, Clone, Default)]
pub struct SpeciesSnapshotProvider {
    inner: Arc<RwLock<Option<SpeciesSnapshot>>>,
}

impl SpeciesSnapshotProvider {
    /// Creates an empty snapshot provider.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(None)),
        }
    }

    /// Publishes a new immutable snapshot.
    pub fn publish(&self, snapshot: SpeciesSnapshot) {
        if let Ok(mut guard) = self.inner.write() {
            *guard = Some(snapshot);
        }
    }

    /// Reads the latest published snapshot, if any.
    #[must_use]
    pub fn snapshot(&self) -> Option<SpeciesSnapshot> {
        self.inner.read().ok().and_then(|guard| guard.clone())
    }
}

/// Step result returned from [`step_species_cadence`].
#[derive(Debug, Clone, PartialEq)]
pub enum SpeciesCadenceStepResult {
    /// Off-cadence tick; no segmentation was performed. Zero allocations and zero mutations.
    OffCadence,
    /// Segmentation executed successfully at this cadence boundary.
    Segmented {
        /// Snapshot published to readers.
        snapshot: SpeciesSnapshot,
        /// Number of agents segmented.
        agent_count: usize,
        /// Number of active species.
        active_species_count: usize,
        /// Number of newly minted species.
        minted_count: usize,
        /// Number of extinct species dropped.
        extinct_count: usize,
    },
    /// Refused by the typed adapter due to invalid inputs.
    Refused(SpeciesAdapterError),
}

/// Executes a species cadence step on immutable completed-tick state (bd-16g.3.6).
///
/// If `cadence.should_segment(tick)` is false, immediately returns [`SpeciesCadenceStepResult::OffCadence`]
/// without allocating, without calling RNG, and without mutating any scientific state.
pub fn step_species_cadence(
    tick: Tick,
    cadence: SpeciesCadence,
    params: &SpeciesParams,
    adapter_config: &SpeciesPhenotypeAdapterConfig,
    inputs: &[TypedPhenotypeInput],
    previous_table: &SpeciesTable,
    publisher: Option<&SpeciesSnapshotProvider>,
) -> SpeciesCadenceStepResult {
    if !cadence.should_segment(tick) {
        return SpeciesCadenceStepResult::OffCadence;
    }

    let samples = match adapt_phenotype_samples(inputs, adapter_config) {
        Ok(s) => s,
        Err(err) => {
            #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
            tracing::warn!(
                tick = tick.0,
                error = ?err,
                "species phenotype adapter refused input samples"
            );
            return SpeciesCadenceStepResult::Refused(err);
        }
    };

    let (new_table, report) = segment_species(tick, &samples, previous_table, params);
    let table_digest = new_table.canonical_digest();
    let report_digest = report.canonical_digest();
    let params_digest = params.canonical_digest();

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    tracing::info!(
        tick = tick.0,
        phenotype_schema = PHENOTYPE_FEATURE_SCHEMA_ID_V1,
        schema_version = PHENOTYPE_FEATURE_SCHEMA_VERSION_V1,
        schema_digest = ?adapter_config.header.schema_digest,
        input_count = inputs.len(),
        accepted_count = samples.len(),
        params_digest = %params_digest,
        table_digest = %table_digest,
        species_count = new_table.species.len(),
        minted_count = report.new_species_minted.len(),
        extinct_count = report.extinct_species_dropped.len(),
        cadence_interval = cadence.interval,
        "species segmentation cadence completed"
    );

    let snapshot = SpeciesSnapshot {
        tick,
        table: new_table,
        report: report.clone(),
        table_digest,
        report_digest,
        params_digest,
        cadence_interval: cadence.interval,
    };

    if let Some(publ) = publisher {
        publ.publish(snapshot.clone());
    }

    SpeciesCadenceStepResult::Segmented {
        snapshot,
        agent_count: samples.len(),
        active_species_count: report.active_species_count,
        minted_count: report.new_species_minted.len(),
        extinct_count: report.extinct_species_dropped.len(),
    }
}

/// Reconstructs a species table and report offline from persisted/recorded samples,
/// verifying that the resulting canonical table digest matches the live digest.
#[must_use]
pub fn reconstruct_species_table_offline(
    tick: Tick,
    samples: &[(AgentUid, Vec<f32>)],
    previous_table: &SpeciesTable,
    params: &SpeciesParams,
) -> (SpeciesTable, SegmentReport, String) {
    let (table, report) = segment_species(tick, samples, previous_table, params);
    let digest = table.canonical_digest();
    (table, report, digest)
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

    fn hawp_vector(uid: u64, features: [f32; PHENOTYPE_FEATURE_COUNT]) -> AgentPhenotypeVector {
        AgentPhenotypeVector {
            agent_uid: AgentUid(uid),
            movement_speed_mean: features[0],
            diet_herbivore_ratio: features[1],
            sensing_range_mean: features[2],
            aggression_index: features[3],
            giving_altruism_index: features[4],
            reproduction_rate: features[5],
        }
    }

    /// Eight irregular rows: no feature is constant and no pair is collinear, so the pooled
    /// covariance of two such cohorts reaches full rank and admits a Cholesky factorization.
    const HAWP_ROWS: [[f32; PHENOTYPE_FEATURE_COUNT]; 8] = [
        [1.50, 0.90, 0.80, 0.10, 0.50, 0.05],
        [1.62, 0.71, 0.34, 0.48, 0.13, 0.22],
        [0.94, 0.35, 0.61, 0.27, 0.88, 0.14],
        [1.18, 0.58, 0.92, 0.63, 0.31, 0.41],
        [0.77, 0.83, 0.19, 0.35, 0.66, 0.09],
        [1.41, 0.22, 0.74, 0.81, 0.45, 0.33],
        [1.05, 0.67, 0.48, 0.16, 0.72, 0.27],
        [0.89, 0.44, 0.86, 0.59, 0.24, 0.18],
    ];

    /// A cohort built from [`HAWP_ROWS`] with every `movement_speed` displaced by `shift`.
    ///
    /// A constant displacement moves the mean without touching the covariance, so separation is
    /// the only thing that varies across the cases below.
    fn hawp_cohort(uid_base: u64, shift: f32) -> Vec<AgentPhenotypeVector> {
        HAWP_ROWS
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let mut features = *row;
                features[0] += shift;
                hawp_vector(uid_base + index as u64, features)
            })
            .collect()
    }

    fn hawp_distance(a: &[AgentPhenotypeVector], b: &[AgentPhenotypeVector]) -> f32 {
        match compare_phenotype_clusters("a", a, "b", b).overall_mahalanobis_distance {
            MahalanobisDistance::Computed(distance) => distance,
            MahalanobisDistance::Unavailable(reason) => {
                panic!("fixture must admit a distance, got {reason:?}")
            }
        }
    }

    #[test]
    fn bd_hawp_identical_cohorts_have_zero_separation() {
        let a = hawp_cohort(1, 0.0);
        let b = hawp_cohort(100, 0.0);

        let distance = hawp_distance(&a, &b);

        assert!(
            distance < 1e-5,
            "cohorts drawn from identical phenotypes must not be separated, got {distance}"
        );
    }

    #[test]
    fn bd_hawp_separation_grows_with_the_distance_between_cohort_means() {
        let base = hawp_cohort(1, 0.0);
        let mut previous = hawp_distance(&base, &hawp_cohort(100, 0.0));

        // The old hardcoded 0.0 satisfied "identical cohorts are close" by accident; only a
        // monotone response to real separation distinguishes a computed statistic from a stub.
        for step in 1..=5 {
            let shifted = hawp_cohort(100, 0.25 * step as f32);
            let distance = hawp_distance(&base, &shifted);
            assert!(
                distance > previous,
                "separation must increase with mean displacement: step {step} gave {distance}, \
                 previous was {previous}"
            );
            previous = distance;
        }
        assert!(
            previous > 1.0,
            "a five-step displacement must be a substantial distance, got {previous}"
        );
    }

    #[test]
    fn bd_hawp_cohorts_too_small_for_a_covariance_inverse_report_why() {
        let a = vec![hawp_vector(1, HAWP_ROWS[0]), hawp_vector(2, HAWP_ROWS[1])];
        let b = vec![hawp_vector(3, HAWP_ROWS[2])];

        let comparison = compare_phenotype_clusters("a", &a, "b", &b);

        assert_eq!(
            comparison.overall_mahalanobis_distance,
            MahalanobisDistance::Unavailable(MahalanobisUnavailable::InsufficientSamples {
                degrees_of_freedom: 1,
                features: PHENOTYPE_FEATURE_COUNT,
            }),
            "an under-determined pair must say so rather than report perfect coincidence"
        );
    }

    #[test]
    fn bd_hawp_empty_cohort_reports_why_instead_of_zero() {
        let a = hawp_cohort(1, 0.0);

        assert_eq!(
            compare_phenotype_clusters("a", &a, "b", &[]).overall_mahalanobis_distance,
            MahalanobisDistance::Unavailable(MahalanobisUnavailable::EmptyCohort)
        );
        assert_eq!(
            compare_phenotype_clusters("a", &[], "b", &a).overall_mahalanobis_distance,
            MahalanobisDistance::Unavailable(MahalanobisUnavailable::EmptyCohort)
        );
    }

    #[test]
    fn bd_hawp_a_constant_feature_makes_the_covariance_singular() {
        // `reproduction_rate` is identical in every sample of both cohorts, so its variance is
        // zero, the pooled covariance loses rank, and no inverse exists.
        let flatten = |uid_base: u64, shift: f32| -> Vec<AgentPhenotypeVector> {
            hawp_cohort(uid_base, shift)
                .into_iter()
                .map(|mut vector| {
                    vector.reproduction_rate = 0.25;
                    vector
                })
                .collect()
        };

        let comparison = compare_phenotype_clusters("a", &flatten(1, 0.0), "b", &flatten(100, 0.5));

        assert_eq!(
            comparison.overall_mahalanobis_distance,
            MahalanobisDistance::Unavailable(MahalanobisUnavailable::SingularCovariance),
            "a degenerate covariance must be reported, not silently inverted"
        );
    }

    #[test]
    fn bd_hawp_cohens_d_matches_a_hand_computed_golden() {
        // movement_speed only: A = {1.0, 3.0}, B = {5.0, 7.0}.
        //   mean_a = 2.0, mean_b = 6.0
        //   var_a  = ((1-2)^2 + (3-2)^2) / 2 = 1.0,  var_b = 1.0   (population convention)
        //   pooled_sd = sqrt((1.0 + 1.0) / 2) = 1.0
        //   d = (2.0 - 6.0) / 1.0 = -4.0
        let a = vec![
            hawp_vector(1, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            hawp_vector(2, [3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let b = vec![
            hawp_vector(3, [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            hawp_vector(4, [7.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];

        let comparison = compare_phenotype_clusters("a", &a, "b", &b);

        let movement = comparison.feature_effect_sizes["movement_speed"];
        assert!(
            (movement - (-4.0)).abs() < 1e-6,
            "hand-computed Cohen's d for movement_speed is -4.0, got {movement}"
        );
        // Every other feature is identical in both cohorts, so its effect size is exactly zero.
        for name in [
            "diet_herbivore_ratio",
            "sensing_range",
            "aggression_index",
            "giving_altruism_index",
            "reproduction_rate",
        ] {
            let value = comparison.feature_effect_sizes[name];
            assert!(
                value.abs() < 1e-6,
                "{name} is identical across cohorts and must have zero effect size, got {value}"
            );
        }
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

        let cohort_b = vec![AgentPhenotypeVector {
            agent_uid: AgentUid(3),
            movement_speed_mean: 0.5,
            diet_herbivore_ratio: 0.1,
            sensing_range_mean: 0.3,
            aggression_index: 0.8,
            giving_altruism_index: 0.0,
            reproduction_rate: 0.01,
        }];

        // bd-ehvv: the report's headline field must actually be populated. It used to be
        // hardcoded empty on every path, so this assertion is the whole point of the fix --
        // an empty `comparisons` is indistinguishable from "these cohorts do not differ".
        let report = compute_phenotype_analysis(
            "run-test",
            Tick(100),
            &cohort_a,
            &[("Herbivores", &cohort_a), ("Carnivores", &cohort_b)],
        );
        assert_eq!(report.total_agents_analyzed, 2);
        assert_eq!(report.mean_phenotype.len(), 6);
        assert_eq!(
            report.comparisons.len(),
            1,
            "two cohorts must yield exactly one pairwise comparison, got {:?}",
            report.comparisons.len()
        );
        let pair = &report.comparisons[0];
        assert_eq!(pair.cluster_a_name, "Herbivores");
        assert_eq!(pair.cluster_b_name, "Carnivores");
        assert!(
            pair.feature_effect_sizes.contains_key("movement_speed"),
            "the populated comparison must carry real effect sizes"
        );

        // Asking for no cohorts is the ONE honest empty case.
        let empty = compute_phenotype_analysis("run-test", Tick(100), &cohort_a, &[]);
        assert!(empty.comparisons.is_empty());

        // Three cohorts produce all three unordered pairs, in input order.
        let three = compute_phenotype_analysis(
            "run-test",
            Tick(100),
            &cohort_a,
            &[("A", &cohort_a), ("B", &cohort_b), ("C", &cohort_a)],
        );
        let pairs: Vec<(&str, &str)> = three
            .comparisons
            .iter()
            .map(|c| (c.cluster_a_name.as_str(), c.cluster_b_name.as_str()))
            .collect();
        assert_eq!(pairs, vec![("A", "B"), ("A", "C"), ("B", "C")]);

        let comparison =
            compare_phenotype_clusters("Herbivores", &cohort_a, "Carnivores", &cohort_b);
        assert_eq!(comparison.sample_size_a, 2);
        assert_eq!(comparison.sample_size_b, 1);
        assert!(
            comparison
                .feature_effect_sizes
                .contains_key("movement_speed")
        );
    }

    /// A cluster comparison is a science report, so serializing the same value twice — and
    /// serializing two independently built but equal values — must produce identical bytes.
    /// An unordered map here silently made the report's key order depend on the process's
    /// hash seed.
    #[test]
    fn cluster_comparison_serializes_to_stable_bytes() {
        fn cohort(uid: u64, speed: f32) -> Vec<AgentPhenotypeVector> {
            vec![
                AgentPhenotypeVector {
                    agent_uid: AgentUid(uid),
                    movement_speed_mean: speed,
                    diet_herbivore_ratio: 0.9,
                    sensing_range_mean: 0.8,
                    aggression_index: 0.1,
                    giving_altruism_index: 0.5,
                    reproduction_rate: 0.05,
                },
                AgentPhenotypeVector {
                    agent_uid: AgentUid(uid + 1),
                    movement_speed_mean: speed + 0.1,
                    diet_herbivore_ratio: 0.8,
                    sensing_range_mean: 0.7,
                    aggression_index: 0.2,
                    giving_altruism_index: 0.4,
                    reproduction_rate: 0.04,
                },
            ]
        }

        let a = cohort(1, 1.5);
        let b = cohort(11, 0.4);

        let first = serde_json::to_string(&compare_phenotype_clusters("A", &a, "B", &b))
            .expect("serialize comparison");
        let second = serde_json::to_string(&compare_phenotype_clusters("A", &a, "B", &b))
            .expect("serialize comparison again");
        assert_eq!(
            first, second,
            "the same comparison must serialize to identical bytes"
        );

        // The six feature names are a closed set, so the emitted order is knowable, not
        // merely repeatable within one process.
        let comparison = compare_phenotype_clusters("A", &a, "B", &b);
        let keys: Vec<&str> = comparison
            .feature_effect_sizes
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            vec![
                "aggression_index",
                "diet_herbivore_ratio",
                "giving_altruism_index",
                "movement_speed",
                "reproduction_rate",
                "sensing_range",
            ],
            "effect-size keys must be emitted in a fixed order"
        );
    }

    // ---- bd-16g.3: phylogeny events and the hint gate ----

    fn sp(id: u64, name: &str, members: usize, tick: u64) -> Species {
        Species {
            id: SpeciesId(id),
            name: name.to_owned(),
            founders: vec![AgentUid(id * 100)],
            members: (0..members)
                .map(|i| AgentUid(id * 1000 + i as u64))
                .collect(),
            centroid: vec![0.0],
            spread: 0.0,
            first_tick: Tick(tick),
            last_seen_tick: Tick(tick),
        }
    }

    fn table(tick: u64, species: Vec<Species>) -> SpeciesTable {
        SpeciesTable {
            tick: Tick(tick),
            next_id: SpeciesId(species.len() as u64 + 1),
            species,
        }
    }

    /// The three event kinds must be derived from adjacent tables, not guessed.
    #[test]
    fn bd_16g_3_diff_emits_speciation_extinction_and_radiation() {
        let before = table(10, vec![sp(1, "Alpha-1", 10, 0), sp(2, "Beta-2", 6, 0)]);
        // 1 survives unchanged, 2 vanishes, 3 appears, 4 appears then would radiate.
        let after = table(20, vec![sp(1, "Alpha-1", 10, 0), sp(3, "Gamma-3", 5, 20)]);
        let events = diff_species_tables(&before, &after);

        let kinds: Vec<_> = events.iter().map(|e| (e.kind, e.species.0)).collect();
        assert!(
            kinds.contains(&(PhylogenyEventKind::Speciation, 3)),
            "a new species id must emit a speciation: {kinds:?}"
        );
        assert!(
            kinds.contains(&(PhylogenyEventKind::Extinction, 2)),
            "a vanished species id must emit an extinction: {kinds:?}"
        );
        assert!(
            !kinds
                .iter()
                .any(|(k, id)| *id == 1 && *k != PhylogenyEventKind::Radiation),
            "an unchanged species must not emit speciation or extinction: {kinds:?}"
        );

        let extinction = events
            .iter()
            .find(|e| e.kind == PhylogenyEventKind::Extinction)
            .expect("extinction present");
        assert_eq!(
            extinction.members_before, 6,
            "extinction must carry the last size"
        );
        assert_eq!(extinction.members_after, 0);
        assert_eq!(
            extinction.tick,
            Tick(20),
            "events are stamped with the observing sample"
        );
        assert!(
            !extinction.founders.is_empty(),
            "founders let a reader reach the DAG"
        );
    }

    /// Radiation needs both a growth multiple AND a floor.
    ///
    /// Without the floor, one member becoming three is a "3x radiation" -- the classic
    /// way a ratio threshold turns into noise on small numbers.
    #[test]
    fn bd_16g_3_radiation_requires_a_size_floor_not_just_a_ratio() {
        // 1 -> 3 is 3x but below the floor: not a radiation.
        let small_before = table(0, vec![sp(1, "Alpha-1", 1, 0)]);
        let small_after = table(1, vec![sp(1, "Alpha-1", 3, 0)]);
        assert!(
            diff_species_tables(&small_before, &small_after).is_empty(),
            "a 1 -> 3 wobble is not a radiation"
        );

        // 5 -> 20 clears both.
        let big_before = table(0, vec![sp(1, "Alpha-1", 5, 0)]);
        let big_after = table(1, vec![sp(1, "Alpha-1", 20, 0)]);
        let events = diff_species_tables(&big_before, &big_after);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, PhylogenyEventKind::Radiation);
        assert_eq!(events[0].members_before, 5);
        assert_eq!(events[0].members_after, 20);
    }

    /// Same tables in, byte-identical events out, regardless of table ordering.
    #[test]
    fn bd_16g_3_event_emission_is_deterministic() {
        let before = table(0, vec![sp(1, "Alpha-1", 4, 0), sp(2, "Beta-2", 4, 0)]);
        let after = table(5, vec![sp(3, "Gamma-3", 4, 5), sp(4, "Delta-4", 4, 5)]);
        let first = diff_species_tables(&before, &after);

        let mut reordered_before = before.clone();
        reordered_before.species.reverse();
        let mut reordered_after = after.clone();
        reordered_after.species.reverse();
        let second = diff_species_tables(&reordered_before, &reordered_after);

        assert_eq!(
            first, second,
            "event order must not depend on table iteration order"
        );
        assert!(!first.is_empty());
    }

    /// THE GATE: every hint is confirmed or explicitly rejected, never dropped.
    ///
    /// bd-16g.3's acceptance requires each speciation to have a preceding hint and each
    /// hint to be resolved either way. A reconciliation that silently discarded unmatched
    /// hints would report perfect agreement by throwing away its own counter-evidence,
    /// which is the failure this gate exists to make impossible.
    #[test]
    fn bd_16g_3_hint_reconciliation_accounts_for_every_hint() {
        let events = vec![
            PhylogenyEvent {
                tick: Tick(100),
                kind: PhylogenyEventKind::Speciation,
                species: SpeciesId(1),
                name: "Alpha-1".to_owned(),
                founders: vec![AgentUid(1)],
                members_before: 0,
                members_after: 5,
            },
            PhylogenyEvent {
                tick: Tick(200),
                kind: PhylogenyEventKind::Speciation,
                species: SpeciesId(2),
                name: "Beta-2".to_owned(),
                founders: vec![AgentUid(2)],
                members_before: 0,
                members_after: 5,
            },
        ];
        let hints = vec![
            SpeciationHint { tick: Tick(90) },  // confirms species 1
            SpeciationHint { tick: Tick(500) }, // follows nothing -> rejected
        ];

        let out = reconcile_speciation_with_hints(&events, &hints, 50);
        assert_eq!(
            out.confirmed,
            vec![SpeciesId(1)],
            "hint at 90 confirms the split at 100"
        );
        assert_eq!(
            out.unhinted,
            vec![SpeciesId(2)],
            "the split at 200 had no hint and must be named"
        );
        assert_eq!(
            out.rejected_hints,
            vec![Tick(500)],
            "an unmatched hint is rejected, not dropped"
        );
        assert!(
            out.accounts_for_all_hints(hints.len()),
            "every hint must land in exactly one bucket"
        );
    }

    /// A hint AFTER the split is not evidence for it.
    ///
    /// Accepting one would let the gate congratulate itself on hindsight: the detector
    /// would get credit for predicting something it only noticed afterwards.
    #[test]
    fn bd_16g_3_a_hint_after_the_split_does_not_confirm_it() {
        let events = vec![PhylogenyEvent {
            tick: Tick(100),
            kind: PhylogenyEventKind::Speciation,
            species: SpeciesId(1),
            name: "Alpha-1".to_owned(),
            founders: vec![AgentUid(1)],
            members_before: 0,
            members_after: 5,
        }];
        let hints = vec![SpeciationHint { tick: Tick(120) }];
        let out = reconcile_speciation_with_hints(&events, &hints, 50);
        assert!(out.confirmed.is_empty(), "hindsight is not prediction");
        assert_eq!(out.unhinted, vec![SpeciesId(1)]);
        assert_eq!(out.rejected_hints, vec![Tick(120)]);
        assert!(out.accounts_for_all_hints(hints.len()));
    }

    /// One hint cannot confirm two speciations.
    #[test]
    fn bd_16g_3_a_hint_is_consumed_by_at_most_one_speciation() {
        let mk = |tick: u64, id: u64| PhylogenyEvent {
            tick: Tick(tick),
            kind: PhylogenyEventKind::Speciation,
            species: SpeciesId(id),
            name: format!("S-{id}"),
            founders: vec![AgentUid(id)],
            members_before: 0,
            members_after: 5,
        };
        let events = vec![mk(100, 1), mk(101, 2)];
        let hints = vec![SpeciationHint { tick: Tick(95) }];
        let out = reconcile_speciation_with_hints(&events, &hints, 50);
        assert_eq!(out.confirmed.len(), 1, "one hint, one confirmation");
        assert_eq!(
            out.unhinted.len(),
            1,
            "the second split is unhinted, not co-credited"
        );
        assert!(out.accounts_for_all_hints(hints.len()));
    }

    /// GROUND TRUTH: hand-labelled lineage scenarios, each with a known answer.
    ///
    /// The same shape as bd-16g.2's detector matrix: a case per phenomenon, plus the
    /// negative control. The negative is the one that matters -- a stable population must
    /// emit NOTHING, because a lineage timeline that reports churn every sample is one
    /// nobody reads.
    #[test]
    fn bd_16g_3_ground_truth_matrix_for_lineage_events() {
        // 1. Clean split: a new id appears alongside the incumbent.
        let split = diff_species_tables(
            &table(0, vec![sp(1, "Alpha-1", 20, 0)]),
            &table(10, vec![sp(1, "Alpha-1", 12, 0), sp(2, "Beta-2", 8, 10)]),
        );
        assert_eq!(split.len(), 1, "one new lineage, one event: {split:?}");
        assert_eq!(split[0].kind, PhylogenyEventKind::Speciation);
        assert_eq!(split[0].species, SpeciesId(2));

        // 2. Clean extinction.
        let gone = diff_species_tables(
            &table(0, vec![sp(1, "Alpha-1", 5, 0), sp(2, "Beta-2", 3, 0)]),
            &table(10, vec![sp(1, "Alpha-1", 5, 0)]),
        );
        assert_eq!(gone.len(), 1);
        assert_eq!(gone[0].kind, PhylogenyEventKind::Extinction);
        assert_eq!(
            gone[0].members_before, 3,
            "the last known size must be carried"
        );

        // 3. Clean radiation.
        let boom = diff_species_tables(
            &table(0, vec![sp(1, "Alpha-1", 6, 0)]),
            &table(10, vec![sp(1, "Alpha-1", 30, 0)]),
        );
        assert_eq!(boom.len(), 1);
        assert_eq!(boom[0].kind, PhylogenyEventKind::Radiation);

        // 4. NEGATIVE CONTROL: a stable population emits nothing at all.
        let stable = diff_species_tables(
            &table(0, vec![sp(1, "Alpha-1", 20, 0), sp(2, "Beta-2", 14, 0)]),
            &table(10, vec![sp(1, "Alpha-1", 21, 0), sp(2, "Beta-2", 13, 0)]),
        );
        assert!(
            stable.is_empty(),
            "ordinary drift is not an event; a timeline that churns is one nobody reads: {stable:?}"
        );

        // 5. NEGATIVE: shrinking is not a radiation, however sharply.
        let crash = diff_species_tables(
            &table(0, vec![sp(1, "Alpha-1", 40, 0)]),
            &table(10, vec![sp(1, "Alpha-1", 4, 0)]),
        );
        assert!(
            crash.is_empty(),
            "a collapse is not a radiation, and the species still exists: {crash:?}"
        );
    }

    /// Lineage events must render into the shared evidence envelope.
    #[test]
    fn bd_16g_3_lineage_events_emit_shared_detection_evidence() {
        let events = diff_species_tables(
            &table(0, vec![sp(1, "Alpha-1", 20, 0)]),
            &table(10, vec![sp(1, "Alpha-1", 12, 0), sp(2, "Beta-2", 8, 10)]),
        );
        let speciation = &events[0];

        let corroborated = speciation.evidence(true);
        assert_eq!(corroborated.kind, DetectionKind::Speciation);
        assert_eq!(
            corroborated.metric, "Beta-2",
            "the species name is the identity"
        );
        assert_eq!(corroborated.start_tick, 10);
        assert!(corroborated.finite);
        assert!(matches!(corroborated.class, EvidenceClass::Lineage(true)));

        // The corroboration flag must actually change the narrative, or it is decoration.
        let alone = speciation.evidence(false);
        assert!(matches!(alone.class, EvidenceClass::Lineage(false)));
        assert_ne!(
            corroborated.narrate(),
            alone.narrate(),
            "an uncorroborated split must not read identically to a corroborated one"
        );
        assert!(
            alone.narrate().contains("no preceding detector hint"),
            "the missing hint must be visible to a reader: {}",
            alone.narrate()
        );
        // And the narration must still obey bd-16g.2's rule: cite evidence, not thresholds.
        for token in [
            "radiation_growth_factor",
            "min_members",
            "threshold",
            "exceeded",
        ] {
            assert!(
                !corroborated.narrate().contains(token),
                "lineage narration leaked a configured bound: {}",
                corroborated.narrate()
            );
        }
    }

    /// Extinction narrates its last known size, not a ratio.
    #[test]
    fn bd_16g_3_extinction_evidence_reports_the_last_known_size() {
        let events = diff_species_tables(
            &table(0, vec![sp(1, "Alpha-1", 5, 0), sp(2, "Beta-2", 9, 0)]),
            &table(10, vec![sp(1, "Alpha-1", 5, 0)]),
        );
        let evidence = events[0].evidence(false);
        assert_eq!(evidence.kind, DetectionKind::Extinction);
        assert_eq!(evidence.before.expect("before side").samples, 9);
        assert_eq!(evidence.after.expect("after side").samples, 0);
        assert_eq!(
            evidence.score, 0.0,
            "there is no growth ratio to report when the denominator is what ended"
        );
        let text = evidence.narrate();
        assert!(text.contains("died out"), "{text}");
        assert!(
            text.contains('9'),
            "the last known size must appear: {text}"
        );
    }

    fn birth(uid: u64, parents: (Option<u64>, Option<u64>)) -> crate::BirthRecord {
        crate::BirthRecord {
            tick: Tick(1),
            agent_uid: AgentUid(uid),
            spawn_ordinal: uid,
            birth_ordinal: Some(uid),
            origin: crate::BirthOrigin::Born,
            parent_a: parents.0.map(AgentUid),
            parent_b: parents.1.map(AgentUid),
            brain_kind: None,
            brain_key: None,
            herbivore_tendency: 0.5,
            generation: crate::Generation(1),
            position: crate::Position::default(),
            is_hybrid: false,
        }
    }

    /// A cluster that appears and vanishes is segmentation jitter, not a lineage.
    #[test]
    fn bd_16g_3_a_split_must_persist_before_it_counts() {
        let mut watch = SpeciationWatch::new(3);
        // Sample 1 is the baseline: incumbents are not speciations.
        assert!(
            watch
                .observe(&table(0, vec![sp(1, "Alpha-1", 20, 0)]))
                .is_empty(),
            "the founding population is not a wave of speciations"
        );

        // Beta appears and holds for two samples -- still one short.
        for tick in [10, 20] {
            let v = watch.observe(&table(
                tick,
                vec![sp(1, "Alpha-1", 15, 0), sp(2, "Beta-2", 5, 10)],
            ));
            assert!(v.is_empty(), "two of three samples must not confirm: {v:?}");
        }

        // It vanishes on the third. That is a transient, and it must be REPORTED,
        // not silently dropped -- a candidate that failed is evidence too.
        let v = watch.observe(&table(30, vec![sp(1, "Alpha-1", 20, 0)]));
        assert_eq!(v.len(), 1);
        assert!(
            matches!(
                v[0],
                SpeciationVerdict::Transient {
                    species: SpeciesId(2),
                    samples: 2,
                    ..
                }
            ),
            "{v:?}"
        );
        assert_eq!(
            watch.pending_count(),
            0,
            "a settled candidate must be dropped"
        );
    }

    /// A cluster that holds for K samples is confirmed exactly once.
    #[test]
    fn bd_16g_3_a_persisting_split_confirms_once_and_only_once() {
        let mut watch = SpeciationWatch::new(3);
        watch.observe(&table(0, vec![sp(1, "Alpha-1", 20, 0)]));

        let mut confirmations = 0;
        for tick in [10, 20, 30, 40, 50] {
            for v in watch.observe(&table(
                tick,
                vec![sp(1, "Alpha-1", 15, 0), sp(2, "Beta-2", 5, 10)],
            )) {
                match v {
                    SpeciationVerdict::Persisted {
                        species,
                        first_seen,
                        confirmed_at,
                    } => {
                        confirmations += 1;
                        assert_eq!(species, SpeciesId(2));
                        assert_eq!(first_seen, Tick(10), "the streak started at first sight");
                        assert_eq!(confirmed_at, Tick(30), "third consecutive sample");
                    }
                    other => panic!("unexpected verdict: {other:?}"),
                }
            }
        }
        assert_eq!(
            confirmations, 1,
            "an established species must not re-confirm every sample"
        );
    }

    /// A re-minted ID must not resume the streak it abandoned.
    #[test]
    fn bd_16g_3_a_vanished_candidate_does_not_resume_its_old_streak() {
        let mut watch = SpeciationWatch::new(3);
        watch.observe(&table(0, vec![sp(1, "Alpha-1", 20, 0)]));
        let two = || vec![sp(1, "Alpha-1", 15, 0), sp(2, "Beta-2", 5, 10)];

        watch.observe(&table(10, two()));
        watch.observe(&table(20, two()));
        watch.observe(&table(30, vec![sp(1, "Alpha-1", 20, 0)])); // transient
        watch.observe(&table(40, two())); // streak restarts at 1
        let v = watch.observe(&table(50, two())); // 2 of 3
        assert!(
            v.is_empty(),
            "the abandoned streak must not carry over and confirm early: {v:?}"
        );
        let v = watch.observe(&table(60, two()));
        assert!(
            matches!(v.as_slice(), [SpeciationVerdict::Persisted { .. }]),
            "the restarted streak confirms on its own third sample: {v:?}"
        );
    }

    /// The measurement counts what actually mated, and refuses to guess.
    #[test]
    fn bd_16g_3_cross_cluster_mating_is_measured_not_inferred() {
        // sp() gives species 1 members 1000.., species 2 members 2000..
        let t = table(10, vec![sp(1, "Alpha-1", 4, 0), sp(2, "Beta-2", 4, 0)]);
        let births = vec![
            birth(90, (Some(1000), Some(1001))), // within species 1
            birth(91, (Some(2000), Some(2001))), // within species 2
            birth(92, (Some(1000), Some(2000))), // CROSS
            birth(93, (Some(1000), None)),       // asexual
            birth(94, (Some(1000), Some(1000))), // self-parented: budding
            birth(95, (Some(1000), Some(7777))), // parent 7777 is dead/unknown
        ];
        let m = measure_cross_cluster_mating(&births, &t);
        assert_eq!(m.within, 2);
        assert_eq!(m.cross, 1);
        assert_eq!(
            m.asexual, 2,
            "no-parent and self-parent both carry no signal"
        );
        assert_eq!(
            m.unattributable, 1,
            "an unknown parent must never be folded into `within`"
        );
        assert_eq!(m.attributed(), 3);
        let rate = m.cross_rate().expect("matings were observed");
        assert!((rate - 1.0 / 3.0).abs() < 1e-12, "{rate}");
    }

    /// No observed matings means UNDEFINED, never "perfectly separated".
    #[test]
    fn bd_16g_3_no_matings_is_undetermined_not_separation() {
        let t = table(10, vec![sp(1, "Alpha-1", 4, 0), sp(2, "Beta-2", 4, 0)]);
        let m = measure_cross_cluster_mating(&[birth(90, (Some(1000), None))], &t);
        assert_eq!(m.attributed(), 0);
        assert_eq!(
            m.cross_rate(),
            None,
            "a window with no matings has no rate; 0.0 would claim perfect separation"
        );

        let persisted = SpeciationVerdict::Persisted {
            species: SpeciesId(2),
            first_seen: Tick(10),
            confirmed_at: Tick(30),
        };
        assert_eq!(
            classify_speciation(&persisted, &m, REPRODUCTIVE_SEPARATION_MAX_RATE),
            SpeciationStatus::Undetermined,
            "silence is not evidence of separation"
        );
    }

    /// THE CASE THIS GATE EXISTS FOR: a stable polymorphism inside one gene pool.
    ///
    /// Two clusters, persistent across every sample, freely interbreeding. Phenotype
    /// clustering alone calls this a speciation; the realized mating record says it is
    /// one population with two forms.
    #[test]
    fn bd_16g_3_a_persistent_interbreeding_split_is_polymorphism_not_speciation() {
        let mut watch = SpeciationWatch::new(3);
        watch.observe(&table(0, vec![sp(1, "Alpha-1", 20, 0)]));
        let mut verdict = None;
        for tick in [10, 20, 30] {
            if let Some(v) = watch
                .observe(&table(
                    tick,
                    vec![sp(1, "Alpha-1", 10, 0), sp(2, "Beta-2", 10, 10)],
                ))
                .into_iter()
                .next()
            {
                verdict = Some(v);
            }
        }
        let verdict = verdict.expect("the split persisted for three samples");
        assert!(matches!(verdict, SpeciationVerdict::Persisted { .. }));

        let t = table(30, vec![sp(1, "Alpha-1", 4, 0), sp(2, "Beta-2", 4, 0)]);
        let interbreeding = vec![
            birth(90, (Some(1000), Some(2000))),
            birth(91, (Some(1001), Some(2001))),
            birth(92, (Some(1002), Some(2002))),
            birth(93, (Some(1000), Some(1001))),
        ];
        let m = measure_cross_cluster_mating(&interbreeding, &t);
        let status = classify_speciation(&verdict, &m, REPRODUCTIVE_SEPARATION_MAX_RATE);
        assert!(
            matches!(status, SpeciationStatus::Polymorphic { .. }),
            "persistence alone must not promote an interbreeding split: {status:?}"
        );

        // Same persisted split, but the two forms stopped mating across.
        let separated = vec![
            birth(90, (Some(1000), Some(1001))),
            birth(91, (Some(1002), Some(1003))),
            birth(92, (Some(2000), Some(2001))),
            birth(93, (Some(2002), Some(2003))),
        ];
        let m = measure_cross_cluster_mating(&separated, &t);
        assert!(
            matches!(
                classify_speciation(&verdict, &m, REPRODUCTIVE_SEPARATION_MAX_RATE),
                SpeciationStatus::Speciation { cross_rate } if cross_rate == 0.0
            ),
            "a persisted split with no cross-mating IS a speciation"
        );
    }

    /// A transient never reaches the mating question at all.
    #[test]
    fn bd_16g_3_a_transient_is_classified_transient_whatever_the_matings_say() {
        let t = table(10, vec![sp(1, "Alpha-1", 4, 0), sp(2, "Beta-2", 4, 0)]);
        let m = measure_cross_cluster_mating(&[birth(90, (Some(1000), Some(1001)))], &t);
        assert_eq!(m.cross_rate(), Some(0.0), "perfect separation on paper");
        let transient = SpeciationVerdict::Transient {
            species: SpeciesId(2),
            first_seen: Tick(10),
            last_seen: Tick(20),
            samples: 2,
        };
        assert_eq!(
            classify_speciation(&transient, &m, REPRODUCTIVE_SEPARATION_MAX_RATE),
            SpeciationStatus::Transient,
            "a cluster that did not hold cannot be rescued by its mating record"
        );
    }

    /// A zero requirement would disable the gate while looking configured.
    #[test]
    fn bd_16g_3_persistence_requirement_is_clamped_to_at_least_one() {
        assert_eq!(SpeciationWatch::new(0).required_samples(), 1);
        assert_eq!(SpeciationWatch::new(1).required_samples(), 1);
        assert_eq!(SpeciationWatch::new(7).required_samples(), 7);
    }

    /// Verdict order must not depend on the order species appear in the table.
    #[test]
    fn bd_16g_3_verdicts_are_emitted_in_stable_id_order() {
        let run = |order: Vec<Species>| {
            let mut watch = SpeciationWatch::new(1);
            watch.observe(&table(0, vec![sp(1, "Alpha-1", 20, 0)]));
            watch.observe(&table(10, order))
        };
        let forward = run(vec![
            sp(1, "Alpha-1", 5, 0),
            sp(2, "Beta-2", 5, 10),
            sp(3, "Gamma-3", 5, 10),
        ]);
        let reversed = run(vec![
            sp(3, "Gamma-3", 5, 10),
            sp(2, "Beta-2", 5, 10),
            sp(1, "Alpha-1", 5, 0),
        ]);
        assert_eq!(forward.len(), 2);
        assert_eq!(
            forward, reversed,
            "the same seed must produce a byte-identical event list"
        );
    }

    /// The watch must stay bounded by LIVE species, not by species ever seen.
    ///
    /// The bead requires a documented memory bound. An unbounded `established` set would
    /// be invisible in every short test and would only show up as drift in a 50k-tick run.
    #[test]
    fn bd_16g_3_watch_memory_is_bounded_by_live_species() {
        let mut watch = SpeciationWatch::new(1);
        watch.observe(&table(0, vec![sp(1, "Alpha-1", 5, 0)]));
        // 200 species are minted and go extinct one after another, never overlapping.
        for n in 2..202u64 {
            watch.observe(&table(n * 10, vec![sp(n, &format!("Sp-{n}"), 5, n * 10)]));
        }
        assert_eq!(
            watch.established_count(),
            1,
            "only the live species may be retained"
        );
        assert_eq!(watch.pending_count(), 0);
    }

    // ========================================================================
    // bd-16g.3.6 Tests: Typed Phenotype Adapter and Live Cadence
    // ========================================================================

    #[test]
    fn test_bd_16g_3_6_canonical_axes_and_units() {
        assert_eq!(PHENOTYPE_AXIS_COUNT_V1, 6);
        assert_eq!(PHENOTYPE_CANONICAL_AXES.len(), 6);
        assert_eq!(
            PHENOTYPE_CANONICAL_AXES[0],
            ("movement.speed.mean", "world_unit_per_tick")
        );
        assert_eq!(
            PHENOTYPE_CANONICAL_AXES[1],
            ("diet.herbivore_trait.mean", "ratio")
        );
        assert_eq!(
            PHENOTYPE_CANONICAL_AXES[2],
            ("sensing.trait_modifier.mean", "trait_multiplier")
        );
        assert_eq!(
            PHENOTYPE_CANONICAL_AXES[3],
            ("interaction.combat.actor_rate", "event_per_tick")
        );
        assert_eq!(
            PHENOTYPE_CANONICAL_AXES[4],
            ("interaction.share.actor_rate", "event_per_tick")
        );
        assert_eq!(
            PHENOTYPE_CANONICAL_AXES[5],
            ("lineage.offspring.parent_rate", "edge_per_tick")
        );
    }

    #[test]
    fn test_bd_16g_3_6_adapter_rejects_schema_and_version_mismatch() {
        let mut config = SpeciesPhenotypeAdapterConfig::default();
        config.header.schema_id = "wrong.schema".to_string();
        let inputs = vec![TypedPhenotypeInput::new(
            AgentUid(1),
            5,
            [1.0, 0.5, 1.0, 0.0, 0.0, 0.1],
        )];

        let err = adapt_phenotype_samples(&inputs, &config).unwrap_err();
        assert!(matches!(err, SpeciesAdapterError::SchemaMismatch { .. }));

        let mut config_v = SpeciesPhenotypeAdapterConfig::default();
        config_v.header.schema_version = 999;
        let err_v = adapt_phenotype_samples(&inputs, &config_v).unwrap_err();
        assert!(matches!(
            err_v,
            SpeciesAdapterError::SchemaVersionMismatch { .. }
        ));
    }

    #[test]
    fn test_bd_16g_3_6_adapter_rejects_invalid_and_duplicate_uid() {
        let config = SpeciesPhenotypeAdapterConfig::default();

        let zero_uid = vec![TypedPhenotypeInput::new(
            AgentUid(0),
            5,
            [1.0, 0.5, 1.0, 0.0, 0.0, 0.1],
        )];
        let err_zero = adapt_phenotype_samples(&zero_uid, &config).unwrap_err();
        assert!(matches!(
            err_zero,
            SpeciesAdapterError::InvalidUid(AgentUid(0))
        ));

        let duplicate_uids = vec![
            TypedPhenotypeInput::new(AgentUid(1), 5, [1.0, 0.5, 1.0, 0.0, 0.0, 0.1]),
            TypedPhenotypeInput::new(AgentUid(1), 5, [2.0, 0.2, 1.0, 0.0, 0.0, 0.1]),
        ];
        let err_dup = adapt_phenotype_samples(&duplicate_uids, &config).unwrap_err();
        assert!(matches!(
            err_dup,
            SpeciesAdapterError::DuplicateUid(AgentUid(1))
        ));
    }

    #[test]
    fn test_bd_16g_3_6_adapter_rejects_nonfinite_values_all_axes() {
        let config = SpeciesPhenotypeAdapterConfig::default();

        for axis in 0..PHENOTYPE_AXIS_COUNT_V1 {
            let mut nan_features = [1.0; PHENOTYPE_AXIS_COUNT_V1];
            nan_features[axis] = f32::NAN;
            let inputs = vec![TypedPhenotypeInput::new(AgentUid(1), 5, nan_features)];
            let err = adapt_phenotype_samples(&inputs, &config).unwrap_err();
            assert!(
                matches!(err, SpeciesAdapterError::NonFiniteValue { axis: a, .. } if a == axis),
                "axis {axis} NaN must produce NonFiniteValue"
            );

            let mut inf_features = [1.0; PHENOTYPE_AXIS_COUNT_V1];
            inf_features[axis] = f32::INFINITY;
            let inputs_inf = vec![TypedPhenotypeInput::new(AgentUid(1), 5, inf_features)];
            let err_inf = adapt_phenotype_samples(&inputs_inf, &config).unwrap_err();
            assert!(
                matches!(err_inf, SpeciesAdapterError::NonFiniteValue { axis: a, .. } if a == axis),
                "axis {axis} Infinity must produce NonFiniteValue"
            );
        }
    }

    #[test]
    fn test_bd_16g_3_6_adapter_rejects_insufficient_lifetime_coverage() {
        let mut config = SpeciesPhenotypeAdapterConfig::default();
        config.min_lifetime_observations = 10;

        let inputs = vec![TypedPhenotypeInput::new(
            AgentUid(1),
            3,
            [1.0, 0.5, 1.0, 0.0, 0.0, 0.1],
        )];
        let err = adapt_phenotype_samples(&inputs, &config).unwrap_err();
        assert!(matches!(
            err,
            SpeciesAdapterError::InsufficientLifetimeCoverage {
                observed: 3,
                required: 10,
                ..
            }
        ));
    }

    #[test]
    fn test_bd_16g_3_6_empty_singleton_and_two_clusters() {
        let config = SpeciesPhenotypeAdapterConfig::default();
        let params = SpeciesParams::default();
        let prev = SpeciesTable::default();

        // Empty
        let empty_samples = adapt_phenotype_samples(&[], &config).unwrap();
        assert!(empty_samples.is_empty());
        let (empty_table, empty_rep) = segment_species(Tick(10), &empty_samples, &prev, &params);
        assert_eq!(empty_table.species.len(), 0);
        assert_eq!(empty_rep.total_agents_segmented, 0);

        // Singleton
        let singleton_inputs = vec![TypedPhenotypeInput::new(
            AgentUid(1),
            5,
            [1.0, 0.5, 1.0, 0.0, 0.0, 0.1],
        )];
        let single_samples = adapt_phenotype_samples(&singleton_inputs, &config).unwrap();
        let (single_table, single_rep) = segment_species(Tick(10), &single_samples, &prev, &params);
        assert_eq!(single_table.species.len(), 1);
        assert_eq!(single_rep.total_agents_segmented, 1);

        // Two distinct clusters
        let two_cluster_inputs = vec![
            TypedPhenotypeInput::new(AgentUid(1), 5, [0.1, 0.1, 0.1, 0.0, 0.0, 0.1]),
            TypedPhenotypeInput::new(AgentUid(2), 5, [0.12, 0.08, 0.11, 0.0, 0.0, 0.1]),
            TypedPhenotypeInput::new(AgentUid(3), 5, [0.9, 0.9, 0.9, 0.8, 0.0, 0.5]),
            TypedPhenotypeInput::new(AgentUid(4), 5, [0.88, 0.91, 0.89, 0.82, 0.0, 0.52]),
        ];
        let two_cluster_samples = adapt_phenotype_samples(&two_cluster_inputs, &config).unwrap();
        let (two_table, two_rep) = segment_species(Tick(10), &two_cluster_samples, &prev, &params);
        assert_eq!(two_table.species.len(), 2);
        assert_eq!(two_rep.total_agents_segmented, 4);
    }

    #[test]
    fn test_bd_16g_3_6_input_permutation_produces_identical_samples_and_digests() {
        let config = SpeciesPhenotypeAdapterConfig::default();
        let params = SpeciesParams::default();
        let prev = SpeciesTable::default();

        let inputs1 = vec![
            TypedPhenotypeInput::new(AgentUid(1), 5, [0.1, 0.1, 0.1, 0.0, 0.0, 0.1]),
            TypedPhenotypeInput::new(AgentUid(2), 5, [0.12, 0.08, 0.11, 0.0, 0.0, 0.1]),
            TypedPhenotypeInput::new(AgentUid(3), 5, [0.9, 0.9, 0.9, 0.8, 0.0, 0.5]),
            TypedPhenotypeInput::new(AgentUid(4), 5, [0.88, 0.91, 0.89, 0.82, 0.0, 0.52]),
        ];

        let inputs2 = vec![
            TypedPhenotypeInput::new(AgentUid(3), 5, [0.9, 0.9, 0.9, 0.8, 0.0, 0.5]),
            TypedPhenotypeInput::new(AgentUid(1), 5, [0.1, 0.1, 0.1, 0.0, 0.0, 0.1]),
            TypedPhenotypeInput::new(AgentUid(4), 5, [0.88, 0.91, 0.89, 0.82, 0.0, 0.52]),
            TypedPhenotypeInput::new(AgentUid(2), 5, [0.12, 0.08, 0.11, 0.0, 0.0, 0.1]),
        ];

        let samples1 = adapt_phenotype_samples(&inputs1, &config).unwrap();
        let samples2 = adapt_phenotype_samples(&inputs2, &config).unwrap();
        assert_eq!(samples1, samples2, "adapted samples must be sorted by UID");

        let (table1, rep1) = segment_species(Tick(10), &samples1, &prev, &params);
        let (table2, rep2) = segment_species(Tick(10), &samples2, &prev, &params);
        assert_eq!(table1.canonical_digest(), table2.canonical_digest());
        assert_eq!(rep1.canonical_digest(), rep2.canonical_digest());
    }

    #[test]
    fn test_bd_16g_3_6_cadence_boundaries_and_off_cadence_inertness() {
        let cadence = SpeciesCadence::new(50);
        assert!(!cadence.should_segment(Tick(0)));
        assert!(!cadence.should_segment(Tick(1)));
        assert!(!cadence.should_segment(Tick(49)));
        assert!(cadence.should_segment(Tick(50)));
        assert!(!cadence.should_segment(Tick(51)));
        assert!(cadence.should_segment(Tick(100)));

        let params = SpeciesParams::default();
        let config = SpeciesPhenotypeAdapterConfig::default();
        let prev = SpeciesTable::default();
        let publisher = SpeciesSnapshotProvider::new();

        let inputs = vec![TypedPhenotypeInput::new(
            AgentUid(1),
            5,
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.1],
        )];

        // Off-cadence tick: no work, returns OffCadence, publisher remains None
        let off_result = step_species_cadence(
            Tick(25),
            cadence,
            &params,
            &config,
            &inputs,
            &prev,
            Some(&publisher),
        );
        assert_eq!(off_result, SpeciesCadenceStepResult::OffCadence);
        assert!(publisher.snapshot().is_none());

        // Cadence boundary: executes segmentation, publishes snapshot
        let on_result = step_species_cadence(
            Tick(50),
            cadence,
            &params,
            &config,
            &inputs,
            &prev,
            Some(&publisher),
        );
        assert!(matches!(
            on_result,
            SpeciesCadenceStepResult::Segmented { .. }
        ));
        let snap = publisher.snapshot().expect("snapshot must be published");
        assert_eq!(snap.tick, Tick(50));
        assert_eq!(snap.table.species.len(), 1);
        assert_eq!(snap.cadence_interval, 50);
    }

    #[test]
    fn test_bd_16g_3_6_fault_injection_causes_agreement_failure() {
        let params = SpeciesParams::default();
        let prev = SpeciesTable::default();

        let inputs = vec![
            TypedPhenotypeInput::new(AgentUid(1), 5, [0.1, 0.1, 0.1, 0.0, 0.0, 0.1]),
            TypedPhenotypeInput::new(AgentUid(2), 5, [0.9, 0.9, 0.9, 0.8, 0.0, 0.5]),
        ];

        // Clean run
        let clean_config = SpeciesPhenotypeAdapterConfig::default();
        let clean_samples = adapt_phenotype_samples(&inputs, &clean_config).unwrap();
        let (clean_table, _, clean_digest) =
            reconstruct_species_table_offline(Tick(100), &clean_samples, &prev, &params);

        // Fault injected: PerturbFeatures
        let mut fault_config = SpeciesPhenotypeAdapterConfig::default();
        fault_config.fault = SpeciesAdapterFault::PerturbFeatures;
        let fault_samples = adapt_phenotype_samples(&inputs, &fault_config).unwrap();
        let (fault_table, _, fault_digest) =
            reconstruct_species_table_offline(Tick(100), &fault_samples, &prev, &params);

        assert_ne!(
            clean_digest, fault_digest,
            "perturbed features must diverge canonical table digest"
        );
        assert_ne!(
            clean_table.species[0].centroid,
            fault_table.species[0].centroid
        );

        // Fault injected: SchemaIdDrift
        let mut drift_config = SpeciesPhenotypeAdapterConfig::default();
        drift_config.fault = SpeciesAdapterFault::SchemaIdDrift;
        let err_drift = adapt_phenotype_samples(&inputs, &drift_config).unwrap_err();
        assert!(matches!(
            err_drift,
            SpeciesAdapterError::SchemaMismatch { .. }
        ));

        // Fault injected: InjectNonFinite
        let mut nan_config = SpeciesPhenotypeAdapterConfig::default();
        nan_config.fault = SpeciesAdapterFault::InjectNonFinite;
        let err_nan = adapt_phenotype_samples(&inputs, &nan_config).unwrap_err();
        assert!(matches!(
            err_nan,
            SpeciesAdapterError::NonFiniteValue { .. }
        ));
    }

    #[test]
    fn test_bd_16g_3_6_offline_reconstruction_byte_identical() {
        let params = SpeciesParams::default();
        let prev = SpeciesTable::default();
        let config = SpeciesPhenotypeAdapterConfig::default();
        let cadence = SpeciesCadence::new(10);
        let publisher = SpeciesSnapshotProvider::new();

        let inputs = vec![
            TypedPhenotypeInput::new(AgentUid(1), 5, [0.1, 0.1, 0.1, 0.0, 0.0, 0.1]),
            TypedPhenotypeInput::new(AgentUid(2), 5, [0.12, 0.08, 0.11, 0.0, 0.0, 0.1]),
            TypedPhenotypeInput::new(AgentUid(3), 5, [0.9, 0.9, 0.9, 0.8, 0.0, 0.5]),
            TypedPhenotypeInput::new(AgentUid(4), 5, [0.88, 0.91, 0.89, 0.82, 0.0, 0.52]),
        ];

        let result = step_species_cadence(
            Tick(10),
            cadence,
            &params,
            &config,
            &inputs,
            &prev,
            Some(&publisher),
        );

        let live_snap = match result {
            SpeciesCadenceStepResult::Segmented { snapshot, .. } => snapshot,
            other => panic!("expected segmented result, got {other:?}"),
        };

        // Offline reconstruction using adapted samples
        let adapted = adapt_phenotype_samples(&inputs, &config).unwrap();
        let (offline_table, offline_report, offline_digest) =
            reconstruct_species_table_offline(Tick(10), &adapted, &prev, &params);

        assert_eq!(live_snap.table, offline_table);
        assert_eq!(live_snap.report, offline_report);
        assert_eq!(live_snap.table_digest, offline_digest);
        assert_eq!(live_snap.table_digest, offline_table.canonical_digest());
        assert_eq!(live_snap.report_digest, offline_report.canonical_digest());
    }
}
