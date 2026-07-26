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
use std::collections::{BTreeMap, BTreeSet};

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
}
