//! Run-scoped lineage fitness, uncertainty, and evolutionary explanations (bd-2z0.11.10).
//!
//! This module fulfills the completion debt left by bd-2z0.11.1:
//! - **Reconciliation against persisted storage**: Rebuilds the [`AncestryGraph`] from
//!   persisted births and deaths, verifying the accounting identity:
//!   `total_arrivals == living_agents + total_deaths`.
//! - **Explicit missing and censored semantics**: Right-censored lifespans for surviving
//!   agents (`censored: true`, bounded by `latest_tick - birth_tick`) are clearly separated from
//!   completed lifespans (`censored: false`). Root agents (seeded or floor-respawned) with no
//!   parents are explicitly typed rather than assigning phantom identities.
//! - **Uncertainty quantification**: Lineage lifespan distributions are summarized with mean,
//!   standard deviation, median, min, max, and a deterministic moving-block/i.i.d. bootstrap 95%
//!   confidence interval.
//! - **Evolutionary change explanation**: Generation-by-generation survivorship, founder lineage
//!   extinction vs persistence (turnover rate), and Shannon/Simpson diversity indices over the
//!   living population.
//! - **Deterministic provenance exports**: Versioned schema (`scriptbots.lineage-fitness.v1`),
//!   database path, latest tick, and deterministic sort orders.

#![allow(clippy::cast_precision_loss)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::time::Instant;

use scriptbots_core::ancestry::AncestryGraph;
use scriptbots_storage::{PersistedAncestryBirth, rebuild_ancestry};
use serde::{Deserialize, Serialize};

use crate::stats;
use crate::{
    AnalyticsError, ReaderCtx, Report, ReportOutput, ReportParams, base_output, log_report_stage,
};

/// Stable schema identifier for the lineage fitness report.
pub const LINEAGE_FITNESS_SCHEMA_ID_V1: &str = "scriptbots.lineage-fitness.v1";

/// Accounting reconciliation between persisted storage records and the reconstructed graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LineageRunReconciliation {
    /// Total agent arrivals recorded in the persisted births table.
    pub total_arrivals: usize,
    /// Total agent removals recorded in the persisted deaths table.
    pub total_deaths: usize,
    /// Total living agents present in the reconstructed ancestry graph.
    pub living_agents: usize,
    /// Whether the closed accounting identity `total_arrivals == living_agents + total_deaths` holds.
    pub arrivals_accounted: bool,
    /// Total distinct founder roots (agents entering without lineage parents).
    pub founder_count: usize,
    /// Number of founder roots that are currently alive themselves.
    pub living_founders: usize,
    /// Sum of living contribution shares across all founders.
    pub total_contribution_share: f64,
    /// Whether the total living contribution share reconciles (>= 1.0 if living > 0, 0.0 if empty).
    pub contribution_share_reconciled: bool,
}

/// Statistical summary of descendant lifespans with uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LineageLifespanSummary {
    /// Total descendants included in the lifespan analysis.
    pub count: usize,
    /// Number of descendants whose lifespans are right-censored (still alive at run end).
    pub censored_count: usize,
    /// Empirical mean lifespan in ticks.
    pub mean: f64,
    /// Empirical standard deviation of lifespan in ticks.
    pub std_dev: f64,
    /// Median lifespan in ticks.
    pub median: f64,
    /// Minimum observed lifespan in ticks.
    pub min: f64,
    /// Maximum observed lifespan in ticks.
    pub max: f64,
    /// Lower bound of the bootstrap confidence interval for mean lifespan.
    pub ci_low: f64,
    /// Upper bound of the bootstrap confidence interval for mean lifespan.
    pub ci_high: f64,
    /// Nominal confidence level (e.g. 0.95).
    pub confidence: f64,
}

/// Lineage fitness and demographic contribution record for a single founder root.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FounderLineageRecord {
    /// Logical unique identifier of the founder root.
    pub founder_uid: u64,
    /// Simulation tick at which the founder entered the world.
    pub birth_tick: u64,
    /// Typed origin of the founder (e.g., "seeded", "born", "floorrespawn").
    pub origin: String,
    /// Simulation tick at which the founder died, if dead.
    pub death_tick: Option<u64>,
    /// Typed cause of death, if dead.
    pub death_cause: Option<String>,
    /// Whether the founder agent itself is currently alive at run end.
    pub is_living: bool,
    /// Observed lifespan in ticks (right-censored if `is_living` is true).
    pub lifespan_ticks: u64,
    /// True if the founder's lifespan is right-censored (survived to run end).
    pub censored: bool,
    /// Number of direct 1st-generation offspring produced by this founder.
    pub direct_offspring_count: usize,
    /// Total descendants produced across all generations (excluding the founder).
    pub total_descendants: usize,
    /// Descendants produced that are currently alive at run end.
    pub living_descendants: usize,
    /// Total living lineage members (living descendants + founder if alive).
    pub living_lineage_members: usize,
    /// Share of the total living population represented by this founder lineage.
    pub contribution_share: f64,
    /// Maximum lineage depth (generations past founder) reached by any descendant.
    pub max_generation_depth: u32,
    /// Descendant lifespan distribution and uncertainty summary, when descendants exist.
    pub descendant_lifespan: Option<LineageLifespanSummary>,
}

/// Metrics for a single genealogical generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenerationMetricRow {
    /// Generation number (0 = founders, 1 = direct offspring of founders, etc.).
    pub generation: u32,
    /// Total individuals recorded arriving in this generation.
    pub total_born: usize,
    /// Individuals in this generation alive at run end.
    pub living: usize,
    /// Individuals in this generation confirmed dead.
    pub dead: usize,
    /// Fraction of individuals in this generation surviving to run end (`living / total_born`).
    pub survival_rate: f64,
    /// Mean observed lifespan of individuals in this generation.
    pub mean_lifespan: f64,
    /// Fraction of individuals in this generation with right-censored lifespans.
    pub censored_ratio: f64,
}

/// Numerical explanation of measured evolutionary change and demographic turnover.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvolutionaryChangeExplanation {
    /// Generation dynamics progression across all observed generations.
    pub generations: Vec<GenerationMetricRow>,
    /// Number of founder lineages with zero living members.
    pub extinct_lineages: usize,
    /// Number of founder lineages with living representatives (founder or descendants).
    pub surviving_lineages: usize,
    /// Lineage turnover rate: fraction of founder lineages that went extinct (`extinct / total`).
    pub turnover_rate: f64,
    /// Shannon diversity index of the living population across founder lineages (-sum(p ln p)).
    pub shannon_diversity: f64,
    /// Simpson diversity index (1 - sum(p^2)).
    pub simpson_diversity: f64,
    /// Dominance: maximum contribution share among all founder lineages.
    pub max_founder_dominance: f64,
}

/// Machine-readable payload for `lineage-fitness`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LineageFitnessMachine {
    /// Schema identifier ([`LINEAGE_FITNESS_SCHEMA_ID_V1`]).
    pub schema: String,
    /// Latest simulation tick observed in the run database.
    pub latest_tick: u64,
    /// Full run reconciliation accounting.
    pub reconciliation: LineageRunReconciliation,
    /// Detailed founder lineage records, sorted by contribution share descending.
    pub founders: Vec<FounderLineageRecord>,
    /// Multi-generation dynamics and evolutionary turnover explanation.
    pub evolutionary_change: EvolutionaryChangeExplanation,
}

/// `lineage-fitness`: Reconciled run-scoped founder fitness, right-censored lifespan uncertainty,
/// and evolutionary turnover explanations.
pub struct LineageFitness;

impl Report for LineageFitness {
    fn name(&self) -> &'static str {
        "lineage-fitness"
    }

    fn description(&self) -> &'static str {
        "Reconciled run-scoped founder fitness, right-censored lifespan uncertainty, and evolutionary explanations"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let read_started = Instant::now();
        let births = cx.reader.load_ancestry_births()?;
        let deaths = cx.reader.load_ancestry_deaths()?;
        let max_tick_opt = cx.reader.max_tick()?;
        log_report_stage("read", &read_started, births.len() + deaths.len());

        let top_founders = params.get_usize("top_founders")?;
        let resamples = params.get_usize("resamples")?.unwrap_or(1000);
        let seed = params.get_u64("seed")?.unwrap_or(0x0114_EA9E);
        let confidence = params.get_f64("confidence")?.unwrap_or(0.95);

        let compute_started = Instant::now();
        let max_observed_tick = births
            .iter()
            .map(|b| b.tick.0)
            .max()
            .unwrap_or(0)
            .max(deaths.iter().map(|d| d.tick.0).max().unwrap_or(0));
        let latest_tick = max_tick_opt
            .unwrap_or(max_observed_tick)
            .max(max_observed_tick);

        let graph = rebuild_ancestry(&births, &deaths).map_err(AnalyticsError::Ancestry)?;
        let total_arrivals = births.len();
        let total_deaths = deaths.len();
        let living_agents = graph.living();
        let arrivals_accounted = total_arrivals == (living_agents + total_deaths);

        let roots = graph.roots();
        let founder_count = roots.len();
        let mut living_founders = 0usize;

        for &r in &roots {
            if let Some(node) = graph.node(r)
                && node.death_tick.is_none()
                && !node.pruned
            {
                living_founders += 1;
            }
        }

        let founder_records = compute_founder_records(
            &graph,
            &roots,
            latest_tick,
            living_agents,
            resamples,
            seed,
            confidence,
        )?;

        let total_share: f64 = founder_records.iter().map(|f| f.contribution_share).sum();
        let contribution_share_reconciled = if living_agents > 0 {
            total_share >= 1.0 - 1e-4
        } else {
            total_share.abs() < 1e-4
        };

        let reconciliation = LineageRunReconciliation {
            total_arrivals,
            total_deaths,
            living_agents,
            arrivals_accounted,
            founder_count,
            living_founders,
            total_contribution_share: total_share,
            contribution_share_reconciled,
        };

        let generation_metrics = compute_generation_metrics(&graph, &births, latest_tick);
        let evolutionary_change =
            compute_evolutionary_turnover(founder_count, &founder_records, generation_metrics);

        let md = render_markdown(
            cx,
            latest_tick,
            &reconciliation,
            &evolutionary_change,
            &founder_records,
            top_founders,
        );

        let machine = LineageFitnessMachine {
            schema: LINEAGE_FITNESS_SCHEMA_ID_V1.to_owned(),
            latest_tick,
            reconciliation,
            founders: founder_records,
            evolutionary_change,
        };

        let row_count = machine.founders.len();
        let output = base_output(
            self.name(),
            cx,
            row_count,
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("compute_and_render", &compute_started, row_count);
        Ok(output)
    }
}

fn compute_founder_records(
    graph: &AncestryGraph,
    roots: &[scriptbots_core::AgentUid],
    latest_tick: u64,
    living_agents: usize,
    resamples: usize,
    seed: u64,
    confidence: f64,
) -> Result<Vec<FounderLineageRecord>, AnalyticsError> {
    let mut founder_records = Vec::with_capacity(roots.len());
    for &founder_uid in roots {
        let Some(node) = graph.node(founder_uid) else {
            continue;
        };
        let birth_tick = node.birth_tick.0;
        let death_tick = node.death_tick.map(|t| t.0);
        let is_living = death_tick.is_none() && !node.pruned;
        let death_cause = node.death_cause.map(|c| format!("{c:?}"));
        let lifespan_ticks = if is_living {
            latest_tick.saturating_sub(birth_tick)
        } else {
            death_tick.unwrap_or(latest_tick).saturating_sub(birth_tick)
        };
        let censored = is_living;
        let direct_offspring_count = node.children.len();

        let mut visited = BTreeSet::new();
        let mut stack = vec![(founder_uid, 0u32)];
        let mut living_descendants = 0usize;
        let mut max_depth = 0u32;
        let mut descendant_lifespans = Vec::new();
        let mut censored_descendants = 0usize;

        while let Some((curr, depth)) = stack.pop() {
            if !visited.insert(curr) {
                continue;
            }
            max_depth = max_depth.max(depth);

            if curr == founder_uid {
                for &child in &node.children {
                    stack.push((child, 1));
                }
            } else if let Some(curr_node) = graph.node(curr) {
                let curr_birth = curr_node.birth_tick.0;
                let curr_lifespan = curr_node.death_tick.map_or_else(
                    || {
                        censored_descendants += 1;
                        living_descendants += 1;
                        latest_tick.saturating_sub(curr_birth) as f64
                    },
                    |dt| dt.0.saturating_sub(curr_birth) as f64,
                );
                descendant_lifespans.push(curr_lifespan);

                for &child in &curr_node.children {
                    stack.push((child, depth + 1));
                }
            }
        }

        let total_descendants = visited.len().saturating_sub(1);
        let living_lineage_members = living_descendants + usize::from(is_living);
        let contribution_share = if living_agents > 0 {
            living_lineage_members as f64 / living_agents as f64
        } else {
            0.0
        };

        let descendant_lifespan = summarize_descendant_lifespans(
            &descendant_lifespans,
            censored_descendants,
            resamples,
            seed.wrapping_add(founder_uid.0),
            confidence,
        )?;

        let origin_str = format!("{:?}", node.origin).to_lowercase();

        founder_records.push(FounderLineageRecord {
            founder_uid: founder_uid.0,
            birth_tick,
            origin: origin_str,
            death_tick,
            death_cause,
            is_living,
            lifespan_ticks,
            censored,
            direct_offspring_count,
            total_descendants,
            living_descendants,
            living_lineage_members,
            contribution_share,
            max_generation_depth: max_depth,
            descendant_lifespan,
        });
    }

    founder_records.sort_by(|a, b| {
        b.contribution_share
            .partial_cmp(&a.contribution_share)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.living_lineage_members.cmp(&a.living_lineage_members))
            .then_with(|| b.total_descendants.cmp(&a.total_descendants))
            .then_with(|| a.founder_uid.cmp(&b.founder_uid))
    });

    Ok(founder_records)
}

fn summarize_descendant_lifespans(
    descendant_lifespans: &[f64],
    censored_descendants: usize,
    resamples: usize,
    founder_seed: u64,
    confidence: f64,
) -> Result<Option<LineageLifespanSummary>, AnalyticsError> {
    if descendant_lifespans.is_empty() {
        return Ok(None);
    }

    let mean_val = stats::mean(descendant_lifespans).map_err(|e| {
        AnalyticsError::Storage(scriptbots_storage::StorageError::InvalidData {
            context: "analytics.lineage_fitness.mean",
            reason: e.to_string(),
        })
    })?;
    let std_dev_val = stats::std_dev(descendant_lifespans).map_err(|e| {
        AnalyticsError::Storage(scriptbots_storage::StorageError::InvalidData {
            context: "analytics.lineage_fitness.std_dev",
            reason: e.to_string(),
        })
    })?;
    let median_val = stats::quantile(descendant_lifespans, 0.50).map_err(|e| {
        AnalyticsError::Storage(scriptbots_storage::StorageError::InvalidData {
            context: "analytics.lineage_fitness.median",
            reason: e.to_string(),
        })
    })?;
    let min_val = descendant_lifespans
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_val = descendant_lifespans
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let (ci_low, ci_high) = if descendant_lifespans.len() >= 2 && resamples > 0 {
        stats::moving_block_bootstrap_mean_ci(
            descendant_lifespans,
            1,
            resamples,
            confidence,
            founder_seed,
        )
        .map_or((mean_val, mean_val), |ci| (ci.lower, ci.upper))
    } else {
        (mean_val, mean_val)
    };

    Ok(Some(LineageLifespanSummary {
        count: descendant_lifespans.len(),
        censored_count: censored_descendants,
        mean: mean_val,
        std_dev: std_dev_val,
        median: median_val,
        min: min_val,
        max: max_val,
        ci_low,
        ci_high,
        confidence,
    }))
}

fn compute_generation_metrics(
    graph: &AncestryGraph,
    births: &[PersistedAncestryBirth],
    latest_tick: u64,
) -> Vec<GenerationMetricRow> {
    let mut by_gen: BTreeMap<u32, (usize, usize, usize, Vec<f64>, usize)> = BTreeMap::new();
    let mut seen_uids = BTreeSet::new();
    for birth in births {
        if !seen_uids.insert(birth.agent_uid) {
            continue;
        }
        if let Some(node) = graph.node(birth.agent_uid) {
            let generation_idx = node.generation.0;
            let entry = by_gen
                .entry(generation_idx)
                .or_insert((0, 0, 0, Vec::new(), 0));
            entry.0 += 1;
            if node.death_tick.is_none() && !node.pruned {
                entry.1 += 1;
                entry.4 += 1;
                let lifespan = latest_tick.saturating_sub(node.birth_tick.0) as f64;
                entry.3.push(lifespan);
            } else {
                entry.2 += 1;
                let lifespan = node
                    .death_tick
                    .unwrap_or(node.birth_tick)
                    .0
                    .saturating_sub(node.birth_tick.0) as f64;
                entry.3.push(lifespan);
            }
        }
    }

    let mut generation_metrics = Vec::with_capacity(by_gen.len());
    for (generation_idx, (total_born, living, dead, lifespans, censored_count)) in by_gen {
        let survival_rate = if total_born > 0 {
            living as f64 / total_born as f64
        } else {
            0.0
        };
        let censored_ratio = if total_born > 0 {
            censored_count as f64 / total_born as f64
        } else {
            0.0
        };
        let mean_lifespan = stats::mean(&lifespans).unwrap_or(0.0);
        generation_metrics.push(GenerationMetricRow {
            generation: generation_idx,
            total_born,
            living,
            dead,
            survival_rate,
            mean_lifespan,
            censored_ratio,
        });
    }
    generation_metrics
}

fn compute_evolutionary_turnover(
    founder_count: usize,
    founder_records: &[FounderLineageRecord],
    generations: Vec<GenerationMetricRow>,
) -> EvolutionaryChangeExplanation {
    let mut extinct_count = 0usize;
    let mut surviving_count = 0usize;
    let mut max_dominance = 0.0f64;
    for f in founder_records {
        if f.living_lineage_members == 0 {
            extinct_count += 1;
        } else {
            surviving_count += 1;
        }
        if f.contribution_share > max_dominance {
            max_dominance = f.contribution_share;
        }
    }

    let turnover_rate = if founder_count > 0 {
        extinct_count as f64 / founder_count as f64
    } else {
        0.0
    };

    let shannon_diversity = founder_records
        .iter()
        .filter(|f| f.contribution_share > 0.0)
        .map(|f| -f.contribution_share * f.contribution_share.ln())
        .sum::<f64>();

    let simpson_diversity = 1.0
        - founder_records
            .iter()
            .map(|f| f.contribution_share.powi(2))
            .sum::<f64>();

    EvolutionaryChangeExplanation {
        generations,
        extinct_lineages: extinct_count,
        surviving_lineages: surviving_count,
        turnover_rate,
        shannon_diversity,
        simpson_diversity,
        max_founder_dominance: max_dominance,
    }
}

fn render_markdown(
    cx: &ReaderCtx,
    latest_tick: u64,
    reconciliation: &LineageRunReconciliation,
    evolutionary_change: &EvolutionaryChangeExplanation,
    founder_records: &[FounderLineageRecord],
    top_founders: Option<usize>,
) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Lineage Fitness & Evolutionary Report\n");
    let _ = writeln!(md, "- **Database**: `{}`", cx.db_path);
    let _ = writeln!(md, "- **Latest Tick**: `{latest_tick}`");
    let _ = writeln!(
        md,
        "- **Reconciliation**: Arrivals = {} | Deaths = {} | Living = {} (Identity: {})",
        reconciliation.total_arrivals,
        reconciliation.total_deaths,
        reconciliation.living_agents,
        if reconciliation.arrivals_accounted {
            "RECONCILED"
        } else {
            "BREACH"
        }
    );
    let _ = writeln!(
        md,
        "- **Demographic Diversity**: Shannon H' = {:.4} | Simpson D = {:.4} | Turnover = {:.2}% ({} / {} extinct)\n",
        evolutionary_change.shannon_diversity,
        evolutionary_change.simpson_diversity,
        evolutionary_change.turnover_rate * 100.0,
        evolutionary_change.extinct_lineages,
        reconciliation.founder_count
    );

    let _ = writeln!(md, "## Founder Lineage Fitness\n");
    if founder_records.is_empty() {
        let _ = writeln!(md, "_No founder lineages recorded in this run._\n");
    } else {
        let _ = writeln!(
            md,
            "| Founder UID | Origin | Direct Offspring | Total Desc | Living Desc | Living Members | Share (%) | Max Gen | Mean Lifespan (ticks) | 95% Bootstrap CI | Censored |"
        );
        let _ = writeln!(md, "|---|---|---|---|---|---|---|---|---|---|---|");

        let display_limit = top_founders.unwrap_or(founder_records.len());
        for f in founder_records.iter().take(display_limit) {
            let (mean_str, ci_str, censored_str) = f.descendant_lifespan.as_ref().map_or_else(
                || ("-".to_owned(), "-".to_owned(), "-".to_owned()),
                |ls| {
                    (
                        format!("{:.2}", ls.mean),
                        format!("[{:.2}, {:.2}]", ls.ci_low, ls.ci_high),
                        format!("{}/{}", ls.censored_count, ls.count),
                    )
                },
            );

            let _ = writeln!(
                md,
                "| {} | {} | {} | {} | {} | {} | {:.2}% | {} | {} | {} | {} |",
                f.founder_uid,
                f.origin,
                f.direct_offspring_count,
                f.total_descendants,
                f.living_descendants,
                f.living_lineage_members,
                f.contribution_share * 100.0,
                f.max_generation_depth,
                mean_str,
                ci_str,
                censored_str
            );
        }
        if founder_records.len() > display_limit {
            let _ = writeln!(
                md,
                "\n_Displaying top {} of {} founder lineages._",
                display_limit,
                founder_records.len()
            );
        }
        let _ = writeln!(md);
    }

    let _ = writeln!(md, "## Generation Dynamics\n");
    if evolutionary_change.generations.is_empty() {
        let _ = writeln!(md, "_No generations recorded._\n");
    } else {
        let _ = writeln!(
            md,
            "| Generation | Total Born | Living | Dead | Survival Rate | Mean Lifespan | Censored Ratio |"
        );
        let _ = writeln!(md, "|---|---|---|---|---|---|---|");
        for g in &evolutionary_change.generations {
            let _ = writeln!(
                md,
                "| {} | {} | {} | {} | {:.2}% | {:.2} | {:.2}% |",
                g.generation,
                g.total_born,
                g.living,
                g.dead,
                g.survival_rate * 100.0,
                g.mean_lifespan,
                g.censored_ratio * 100.0
            );
        }
        let _ = writeln!(md);
    }
    md
}
