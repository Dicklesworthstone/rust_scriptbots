//! NetworkX-parity lineage and interaction graph reports backed by `fnx` (bd-2z0.11.7).
//!
//! Provides:
//! - **lineage-structure**: Reconstructs the ancestry directed acyclic graph (DAG) via
//!   [`fnx_classes::digraph::DiGraph`], computing weakly connected components (founder families),
//!   strongly connected components (asserting acyclicity), longest generational chains
//!   ([`fnx_algorithms::dag_longest_path`]), in/out-degree distributions, and founder subgraphs.
//! - **dynasty-communities**: Detects modular communities on the undirected projection
//!   of the mating/ancestry graph using Louvain community detection ([`fnx_algorithms::louvain_communities`]),
//!   computes Newman-Girvan modularity ([`fnx_algorithms::modularity`]), and evaluates agreement with
//!   founder lineages.
//! - **interaction-centrality**: Extracts directed pairwise interaction networks (from persisted
//!   pairwise interactions in SQLite or replay events), computing directed degree centrality,
//!   betweenness centrality, and PageRank, while documenting storage persistence gaps.
//! - **Graph exports**: Serializes graphs to GraphML and Edge-List formats via
//!   [`fnx_readwrite::EdgeListEngine`].

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::suboptimal_flops,
    clippy::must_use_candidate,
    clippy::derive_partial_eq_without_eq,
    clippy::doc_markdown,
    clippy::map_unwrap_or,
    clippy::similar_names
)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::time::Instant;

use fnx_classes::digraph::DiGraph;
use fnx_classes::{AttrMap, Graph};
use scriptbots_core::AgentUid;
use scriptbots_storage::{
    InteractionGraphBudget, InteractionGraphEvent, InteractionGraphSelection,
    PersistedAncestryBirth, PersistedAncestryDeath, PersistedInteractionCapture,
};
use serde::{Deserialize, Serialize};

use crate::{
    AnalyticsError, ReaderCtx, Report, ReportOutput, ReportParams, base_output, log_report_stage,
    stats,
};

/// Schema identifier for the lineage structure report.
pub const LINEAGE_STRUCTURE_SCHEMA_ID_V1: &str = "scriptbots.lineage-structure.v1";

/// Schema identifier for the dynasty communities report.
pub const DYNASTY_COMMUNITIES_SCHEMA_ID_V1: &str = "scriptbots.dynasty-communities.v1";

/// Schema identifier for the interaction centrality report.
pub const INTERACTION_CENTRALITY_SCHEMA_ID_V1: &str = "scriptbots.interaction-centrality.v1";

/// Formats an [`AgentUid`] as a canonical graph node identifier string.
#[must_use]
pub fn agent_uid_to_node_id(uid: AgentUid) -> String {
    format!("agent_{}", uid.0)
}

/// Parses an [`AgentUid`] back from a canonical node string key ("agent_123" or "123").
#[must_use]
pub fn node_id_to_agent_uid(node: &str) -> Option<AgentUid> {
    node.strip_prefix("agent_")
        .unwrap_or(node)
        .parse::<u64>()
        .ok()
        .map(AgentUid)
}

// ---------------------------------------------------------------------------
// 1. Lineage Structure Report
// ---------------------------------------------------------------------------

/// Execution timings across graph building and analytical algorithm stages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LineageStructureTimings {
    /// Time spent formatting integer UIDs into string keys.
    pub format_node_keys_ms: f64,
    /// Time spent populating DiGraph edges via `extend_edges_unrecorded`.
    pub bulk_build_edges_ms: f64,
    /// Time spent computing weakly connected components.
    pub weakly_connected_ms: f64,
    /// Time spent verifying strongly connected components (acyclicity).
    pub strongly_connected_ms: f64,
    /// Time spent determining DAG longest path / generation depth.
    pub dag_longest_path_ms: f64,
    /// Time spent computing in-degree and out-degree distributions.
    pub degree_distribution_ms: f64,
    /// Total wall-clock time in milliseconds.
    pub total_ms: f64,
}

/// Degree distribution summary over the number of parents per agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InDegreeSummary {
    /// Agents with 0 parents (seeded founders or floor respawns).
    pub zero_parents_founders: usize,
    /// Agents with 1 parent (asexual reproduction or mutation).
    pub one_parent: usize,
    /// Agents with 2 parents (sexual reproduction).
    pub two_parents_sexual: usize,
    /// Maximum in-degree observed in the graph.
    pub max_in_degree: usize,
}

/// Individual agent reproductive fecundity record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentReproductionRecord {
    /// Logical unique agent identity.
    pub agent_uid: u64,
    /// Total direct children produced (out-degree).
    pub offspring_count: usize,
    /// Generation number.
    pub generation: u32,
    /// Simulation tick at birth.
    pub birth_tick: u64,
    /// Whether the agent is still alive at run end.
    pub is_living: bool,
}

/// Out-degree (offspring count) distribution summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutDegreeSummary {
    /// Total agents that produced zero offspring.
    pub zero_offspring: usize,
    /// Maximum offspring produced by any single agent.
    pub max_offspring: usize,
    /// Mean number of offspring per agent.
    pub mean_offspring: f64,
    /// Standard deviation of offspring count.
    pub std_dev_offspring: f64,
    /// Median number of offspring per agent.
    pub median_offspring: f64,
    /// Top reproductive agents by out-degree.
    pub top_reproducers: Vec<AgentReproductionRecord>,
}

/// Summary of a single founder family (weakly connected component).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FounderFamilyRecord {
    /// Ordinal rank by total family size descending.
    pub rank: usize,
    /// Root founder identities belonging to this component (in-degree == 0).
    pub founder_uids: Vec<u64>,
    /// Primary root founder identity (smallest UID or earliest birth).
    pub primary_founder_uid: u64,
    /// Total agents in this family across all generations.
    pub total_members: usize,
    /// Members of this family still living at run end.
    pub living_members: usize,
    /// Whether the lineage has surviving representatives.
    pub surviving: bool,
    /// Maximum generation depth observed within this component.
    pub max_generation: u32,
}

/// Extinction and turnover dynamics summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtinctionDepthSummary {
    /// Total founder families (weakly connected components).
    pub total_founder_families: usize,
    /// Number of founder families with zero living representatives.
    pub extinct_families: usize,
    /// Number of founder families with living representatives.
    pub surviving_families: usize,
    /// Fraction of founder families that went extinct (`extinct / total`).
    pub turnover_rate: f64,
    /// Maximum generation reached among extinct families.
    pub max_generation_extinct: u32,
    /// Maximum generation reached among surviving families.
    pub max_generation_surviving: u32,
}

/// Machine-readable payload for `lineage-structure`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LineageStructurePayload {
    /// Schema identifier ([`LINEAGE_STRUCTURE_SCHEMA_ID_V1`]).
    pub schema: String,
    /// Latest simulation tick observed in the run database.
    pub latest_tick: u64,
    /// Total nodes (agents) in the lineage graph.
    pub node_count: usize,
    /// Total directed parent-to-child edges in the graph.
    pub edge_count: usize,
    /// Whether the graph is a verified directed acyclic graph.
    pub is_dag: bool,
    /// Length of the longest generation path (edge count).
    pub longest_path_length: usize,
    /// Sequence of agent UIDs along the longest generational path.
    pub longest_path_sample: Vec<u64>,
    /// Total weakly connected components (founder dynasties).
    pub weakly_connected_components_count: usize,
    /// Total strongly connected components.
    pub strongly_connected_components_count: usize,
    /// Number of non-trivial SCCs (must be 0 for an acyclic phylogeny).
    pub non_trivial_scc_count: usize,
    /// In-degree (parents) distribution.
    pub in_degree_summary: InDegreeSummary,
    /// Out-degree (offspring) distribution.
    pub out_degree_summary: OutDegreeSummary,
    /// Founder family records, sorted by size descending.
    pub founder_families: Vec<FounderFamilyRecord>,
    /// Extinction depth and demographic turnover summary.
    pub extinction_depth: ExtinctionDepthSummary,
    /// Wall-clock timing breakdown for telemetry.
    pub timings: LineageStructureTimings,
}

/// Builds a directed lineage [`DiGraph`] from ancestry birth records.
///
/// In this graph:
/// - Nodes represent logical agent identities (`agent_{uid}`).
/// - Directed edges point from **parent to child** (`(parent, child)`).
/// - Roots have in-degree 0; childless leaves have out-degree 0.
/// - Isolated nodes (founders who produced 0 children) are guaranteed present.
pub fn build_lineage_digraph(
    births: &[PersistedAncestryBirth],
) -> (DiGraph, LineageStructureTimings) {
    let t_start = Instant::now();

    // 1. Format string keys for bulk insertion
    let t_fmt = Instant::now();
    let mut edges: Vec<(String, String)> = Vec::with_capacity(births.len() * 2);
    let mut all_nodes: Vec<String> = Vec::with_capacity(births.len());

    for birth in births {
        let child_key = agent_uid_to_node_id(birth.agent_uid);
        all_nodes.push(child_key.clone());
        if let Some(pa) = birth.parent_a {
            edges.push((agent_uid_to_node_id(pa), child_key.clone()));
        }
        if let Some(pb) = birth.parent_b {
            edges.push((agent_uid_to_node_id(pb), child_key));
        }
    }
    let format_node_keys_ms = t_fmt.elapsed().as_secs_f64() * 1000.0;

    // 2. Bulk build via extend_edges_unrecorded
    let t_bulk = Instant::now();
    let mut digraph = DiGraph::strict();
    let _inserted_edges = digraph.extend_edges_unrecorded(edges);

    // Ensure all arriving agents (including isolated founders) exist in graph
    for node_key in all_nodes {
        if !digraph.has_node(&node_key) {
            digraph.add_node(node_key);
        }
    }
    let bulk_build_edges_ms = t_bulk.elapsed().as_secs_f64() * 1000.0;

    tracing::info!(
        nodes = digraph.node_count(),
        edges = digraph.edge_count(),
        fmt_ms = format_node_keys_ms,
        bulk_ms = bulk_build_edges_ms,
        "lineage DiGraph built from births"
    );

    let timings = LineageStructureTimings {
        format_node_keys_ms,
        bulk_build_edges_ms,
        weakly_connected_ms: 0.0,
        strongly_connected_ms: 0.0,
        dag_longest_path_ms: 0.0,
        degree_distribution_ms: 0.0,
        total_ms: t_start.elapsed().as_secs_f64() * 1000.0,
    };

    (digraph, timings)
}

/// Analyzes the topological structure, components, DAG height, and degrees of a lineage graph.
pub fn analyze_lineage_structure(
    digraph: &DiGraph,
    births: &[PersistedAncestryBirth],
    deaths: &[PersistedAncestryDeath],
    latest_tick: u64,
    mut timings: LineageStructureTimings,
) -> LineageStructurePayload {
    let t_analysis_start = Instant::now();
    let node_count = digraph.node_count();
    let edge_count = digraph.edge_count();

    // Map of dead agent UIDs for survivorship determination
    let dead_uids: HashSet<AgentUid> = deaths.iter().map(|d| d.agent_uid).collect();
    let birth_map: HashMap<AgentUid, &PersistedAncestryBirth> =
        births.iter().map(|b| (b.agent_uid, b)).collect();

    // 1. Weakly connected components (founder families)
    let t_wcc = Instant::now();
    let mut wccs = fnx_algorithms::weakly_connected_components(digraph);
    timings.weakly_connected_ms = t_wcc.elapsed().as_secs_f64() * 1000.0;
    tracing::debug!(
        wcc_count = wccs.len(),
        elapsed_ms = timings.weakly_connected_ms,
        "computed weakly connected components"
    );

    // 2. Strongly connected components (verify DAG / 0 non-trivial cycles)
    let t_scc = Instant::now();
    let sccs = fnx_algorithms::strongly_connected_components(digraph);
    let is_dag = fnx_algorithms::is_directed_acyclic_graph(digraph);
    let scc_count = sccs.len();
    let non_trivial_scc_count = sccs.iter().filter(|c| c.len() > 1).count();
    timings.strongly_connected_ms = t_scc.elapsed().as_secs_f64() * 1000.0;

    // 3. DAG longest path (generational chain height)
    let t_dag = Instant::now();
    let longest_path = if is_dag && node_count > 0 {
        fnx_algorithms::dag_longest_path(digraph).unwrap_or_default()
    } else {
        Vec::new()
    };
    let longest_path_length = if longest_path.is_empty() {
        0
    } else {
        longest_path.len().saturating_sub(1)
    };
    let longest_path_sample: Vec<u64> = longest_path
        .iter()
        .filter_map(|s| node_id_to_agent_uid(s).map(|u| u.0))
        .collect();
    timings.dag_longest_path_ms = t_dag.elapsed().as_secs_f64() * 1000.0;

    // 4. Degree distributions
    let degree_start = Instant::now();
    let mut in_zero_parents = 0usize;
    let mut in_one_parent = 0usize;
    let mut in_two_parents = 0usize;
    let mut max_in_degree = 0usize;

    let mut zero_offspring = 0usize;
    let mut max_offspring = 0usize;
    let mut offspring_counts: Vec<f64> = Vec::with_capacity(node_count);
    let mut reproducers: Vec<AgentReproductionRecord> = Vec::new();

    for node_name in digraph.nodes_ordered() {
        let in_deg = digraph.in_degree(node_name);
        let out_deg = digraph.out_degree(node_name);

        max_in_degree = max_in_degree.max(in_deg);
        match in_deg {
            0 => in_zero_parents += 1,
            1 => in_one_parent += 1,
            2 => in_two_parents += 1,
            _ => {}
        }

        max_offspring = max_offspring.max(out_deg);
        if out_deg == 0 {
            zero_offspring += 1;
        }
        offspring_counts.push(out_deg as f64);

        if let Some(uid) = node_id_to_agent_uid(node_name) {
            let (generation, birth_tick) = birth_map
                .get(&uid)
                .map(|b| (b.generation.0, b.tick.0))
                .unwrap_or((0, 0));
            let is_living = !dead_uids.contains(&uid);
            if out_deg > 0 {
                reproducers.push(AgentReproductionRecord {
                    agent_uid: uid.0,
                    offspring_count: out_deg,
                    generation,
                    birth_tick,
                    is_living,
                });
            }
        }
    }
    reproducers.sort_by(|a, b| {
        b.offspring_count
            .cmp(&a.offspring_count)
            .then(a.agent_uid.cmp(&b.agent_uid))
    });
    reproducers.truncate(20);

    let mean_offspring = stats::mean(&offspring_counts).unwrap_or(0.0);
    let std_dev_offspring = stats::std_dev(&offspring_counts).unwrap_or(0.0);
    let median_offspring = stats::quantile(&offspring_counts, 0.5).unwrap_or(0.0);
    timings.degree_distribution_ms = degree_start.elapsed().as_secs_f64() * 1000.0;

    // 5. Founder families summary
    // Sort components by size descending
    wccs.sort_by_key(|a| std::cmp::Reverse(a.len()));

    let mut founder_families: Vec<FounderFamilyRecord> = Vec::with_capacity(wccs.len());
    let mut extinct_families = 0usize;
    let mut surviving_families = 0usize;
    let mut max_gen_extinct = 0u32;
    let mut max_gen_surviving = 0u32;

    for (rank, comp) in wccs.iter().enumerate() {
        let mut founder_uids = Vec::new();
        let mut living_members = 0usize;
        let mut max_generation = 0u32;

        for node_str in comp {
            if let Some(uid) = node_id_to_agent_uid(node_str) {
                if digraph.in_degree(node_str) == 0 {
                    founder_uids.push(uid.0);
                }
                if !dead_uids.contains(&uid) {
                    living_members += 1;
                }
                if let Some(b) = birth_map.get(&uid) {
                    max_generation = max_generation.max(b.generation.0);
                }
            }
        }
        founder_uids.sort_unstable();
        let primary_founder_uid = founder_uids.first().copied().unwrap_or(0);
        let surviving = living_members > 0;

        if surviving {
            surviving_families += 1;
            max_gen_surviving = max_gen_surviving.max(max_generation);
        } else {
            extinct_families += 1;
            max_gen_extinct = max_gen_extinct.max(max_generation);
        }

        founder_families.push(FounderFamilyRecord {
            rank: rank + 1,
            founder_uids,
            primary_founder_uid,
            total_members: comp.len(),
            living_members,
            surviving,
            max_generation,
        });
    }

    let total_founder_families = founder_families.len();
    let turnover_rate = if total_founder_families > 0 {
        extinct_families as f64 / total_founder_families as f64
    } else {
        0.0
    };

    timings.total_ms += t_analysis_start.elapsed().as_secs_f64() * 1000.0;

    LineageStructurePayload {
        schema: LINEAGE_STRUCTURE_SCHEMA_ID_V1.to_owned(),
        latest_tick,
        node_count,
        edge_count,
        is_dag,
        longest_path_length,
        longest_path_sample,
        weakly_connected_components_count: wccs.len(),
        strongly_connected_components_count: scc_count,
        non_trivial_scc_count,
        in_degree_summary: InDegreeSummary {
            zero_parents_founders: in_zero_parents,
            one_parent: in_one_parent,
            two_parents_sexual: in_two_parents,
            max_in_degree,
        },
        out_degree_summary: OutDegreeSummary {
            zero_offspring,
            max_offspring,
            mean_offspring,
            std_dev_offspring,
            median_offspring,
            top_reproducers: reproducers,
        },
        founder_families,
        extinction_depth: ExtinctionDepthSummary {
            total_founder_families,
            extinct_families,
            surviving_families,
            turnover_rate,
            max_generation_extinct: max_gen_extinct,
            max_generation_surviving: max_gen_surviving,
        },
        timings,
    }
}

/// Formats the human-readable markdown representation of [`LineageStructurePayload`].
pub fn render_lineage_structure_md(payload: &LineageStructurePayload) -> String {
    let mut out = String::with_capacity(4096);
    let _ = writeln!(out, "# Lineage Structure Report");
    let _ = writeln!(
        out,
        "\n**Schema:** `{}` | **Latest Tick:** {}",
        payload.schema, payload.latest_tick
    );

    let _ = writeln!(out, "\n## Graph Topology Overview");
    let _ = writeln!(out, "| Metric | Value | Interpretation |");
    let _ = writeln!(out, "| :--- | :--- | :--- |");
    let _ = writeln!(
        out,
        "| Total Nodes (Agents) | {} | Total agents recorded in phylogeny |",
        payload.node_count
    );
    let _ = writeln!(
        out,
        "| Total Directed Edges | {} | Parent-to-offspring relationships |",
        payload.edge_count
    );
    let _ = writeln!(
        out,
        "| Is Valid DAG | {} | Phylogeny contains no temporal cycles |",
        payload.is_dag
    );
    let _ = writeln!(
        out,
        "| Max Generational Depth | {} | Length of longest lineage chain |",
        payload.longest_path_length
    );
    let _ = writeln!(
        out,
        "| Founder Dynasties (WCC) | {} | Distinct disconnected founder trees |",
        payload.weakly_connected_components_count
    );
    let _ = writeln!(
        out,
        "| Strongly Connected (SCC) | {} | Non-trivial cycles: {} |",
        payload.strongly_connected_components_count, payload.non_trivial_scc_count
    );

    let _ = writeln!(out, "\n## Generational Path & Degree Distributions");
    let _ = writeln!(
        out,
        "- **In-Degree (Parents):** {} founders (0 parents), {} single-parent, {} sexual (2 parents), max={}",
        payload.in_degree_summary.zero_parents_founders,
        payload.in_degree_summary.one_parent,
        payload.in_degree_summary.two_parents_sexual,
        payload.in_degree_summary.max_in_degree,
    );
    let _ = writeln!(
        out,
        "- **Out-Degree (Offspring):** mean={:.2}, std={:.2}, median={:.1}, max={}, childless={}",
        payload.out_degree_summary.mean_offspring,
        payload.out_degree_summary.std_dev_offspring,
        payload.out_degree_summary.median_offspring,
        payload.out_degree_summary.max_offspring,
        payload.out_degree_summary.zero_offspring,
    );

    if !payload.longest_path_sample.is_empty() {
        let sample_str = payload
            .longest_path_sample
            .iter()
            .take(8)
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(" -> ");
        let suffix = if payload.longest_path_sample.len() > 8 {
            " -> ..."
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "- **Longest Generational Chain Sample:** {sample_str}{suffix}"
        );
    }

    let _ = writeln!(out, "\n## Founder Dynasties (Top 10)");
    let _ = writeln!(
        out,
        "| Rank | Primary Founder | Total Members | Living Members | Status | Max Gen |"
    );
    let _ = writeln!(out, "| :--- | :--- | :--- | :--- | :--- | :--- |");
    for f in payload.founder_families.iter().take(10) {
        let status = if f.surviving { "Surviving" } else { "Extinct" };
        let _ = writeln!(
            out,
            "| {} | agent_{} | {} | {} | {} | {} |",
            f.rank,
            f.primary_founder_uid,
            f.total_members,
            f.living_members,
            status,
            f.max_generation
        );
    }

    let _ = writeln!(out, "\n## Demographic Turnover");
    let _ = writeln!(
        out,
        "- Total Founder Families: {} (Surviving: {}, Extinct: {})",
        payload.extinction_depth.total_founder_families,
        payload.extinction_depth.surviving_families,
        payload.extinction_depth.extinct_families
    );
    let _ = writeln!(
        out,
        "- Lineage Turnover Rate: {:.2}%",
        payload.extinction_depth.turnover_rate * 100.0
    );
    let _ = writeln!(
        out,
        "- Max Generation: Surviving={}, Extinct={}",
        payload.extinction_depth.max_generation_surviving,
        payload.extinction_depth.max_generation_extinct
    );

    let _ = writeln!(out, "\n## Execution Telemetry");
    let _ = writeln!(
        out,
        "- Node key formatting: {:.2}ms",
        payload.timings.format_node_keys_ms
    );
    let _ = writeln!(
        out,
        "- Bulk edge insertion: {:.2}ms",
        payload.timings.bulk_build_edges_ms
    );
    let _ = writeln!(
        out,
        "- Weakly connected components: {:.2}ms",
        payload.timings.weakly_connected_ms
    );
    let _ = writeln!(
        out,
        "- DAG longest path: {:.2}ms",
        payload.timings.dag_longest_path_ms
    );
    let _ = writeln!(
        out,
        "- Degree distributions: {:.2}ms",
        payload.timings.degree_distribution_ms
    );
    let _ = writeln!(out, "- Total time: {:.2}ms", payload.timings.total_ms);

    out
}

/// `lineage-structure`: Reports weakly/strongly connected components, DAG height, degree distributions,
/// and extinction depths over the ancestry graph.
pub struct LineageStructure;

impl Report for LineageStructure {
    fn name(&self) -> &'static str {
        "lineage-structure"
    }

    fn description(&self) -> &'static str {
        "Lineage graph structure: founder families, DAG height, degree distributions, and extinction depth"
    }

    fn run(&self, cx: &ReaderCtx, _params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let read_start = Instant::now();
        let births = cx.reader.load_ancestry_births()?;
        let deaths = cx.reader.load_ancestry_deaths()?;
        let max_tick = cx.reader.max_tick()?.unwrap_or(0);
        log_report_stage("read", &read_start, births.len() + deaths.len());

        let (digraph, timings) = build_lineage_digraph(&births);
        let payload = analyze_lineage_structure(&digraph, &births, &deaths, max_tick, timings);
        let md = render_lineage_structure_md(&payload);
        let machine = serde_json::to_value(&payload)?;

        base_output("lineage-structure", cx, payload.node_count, machine, md)
    }
}

// ---------------------------------------------------------------------------
// 2. Dynasty Communities Report (Louvain Detection)
// ---------------------------------------------------------------------------

/// Execution timings for dynasty community detection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DynastyCommunitiesTimings {
    /// Time spent constructing the directed graph.
    pub build_digraph_ms: f64,
    /// Time spent converting to undirected projection.
    pub to_undirected_ms: f64,
    /// Time spent running Louvain community detection.
    pub louvain_ms: f64,
    /// Time spent computing Newman-Girvan modularity.
    pub modularity_ms: f64,
    /// Total wall-clock time in milliseconds.
    pub total_ms: f64,
}

/// Record of an individual detected community.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DynastyCommunityRecord {
    /// 1-based rank by community member count.
    pub community_id: usize,
    /// Number of total agents in this community.
    pub member_count: usize,
    /// Number of living agents in this community.
    pub living_count: usize,
    /// Founder root identities present in this community.
    pub founder_uids: Vec<u64>,
    /// Dominant founder UID (most common founder family among members).
    pub dominant_founder_uid: Option<u64>,
    /// Fraction of members belonging to the dominant founder family.
    pub dominant_founder_fraction: f64,
    /// Sample member agent UIDs (up to 10).
    pub sample_members: Vec<u64>,
}

/// Machine-readable payload for `dynasty-communities`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DynastyCommunitiesPayload {
    /// Schema identifier ([`DYNASTY_COMMUNITIES_SCHEMA_ID_V1`]).
    pub schema: String,
    /// Latest simulation tick observed.
    pub latest_tick: u64,
    /// Total nodes in the analyzed graph.
    pub node_count: usize,
    /// Total undirected edges in the projection.
    pub edge_count: usize,
    /// Louvain resolution parameter used.
    pub resolution: f64,
    /// Deterministic RNG seed used for Louvain initialization.
    pub seed: u64,
    /// Threshold parameter for modularity gain convergence.
    pub threshold: f64,
    /// Max levels parameter.
    pub max_level: Option<usize>,
    /// Number of communities detected.
    pub community_count: usize,
    /// Newman-Girvan modularity score.
    pub modularity: f64,
    /// Purity agreement rate with founder families (weakly connected components).
    pub agreement_rate_with_founders: f64,
    /// Adjusted Rand Index with founder families.
    pub rand_index_with_founders: f64,
    /// Detected communities, sorted by size descending.
    pub communities: Vec<DynastyCommunityRecord>,
    /// Wall-clock timings for audit.
    pub timings: DynastyCommunitiesTimings,
}

/// Evaluates agreement between Louvain communities and founder families (WCCs).
///
/// Returns `(purity, adjusted_rand_index)`.
pub fn compute_contingency_agreement(
    wccs: &[Vec<String>],
    communities: &[Vec<String>],
    total_nodes: usize,
) -> (f64, f64) {
    if total_nodes == 0 || communities.is_empty() || wccs.is_empty() {
        return (1.0, 1.0);
    }

    // Map each node to (wcc_id, comm_id)
    let mut node_to_wcc: HashMap<&str, usize> = HashMap::with_capacity(total_nodes);
    for (i, comp) in wccs.iter().enumerate() {
        for node in comp {
            node_to_wcc.insert(node.as_str(), i);
        }
    }

    let mut contingency: HashMap<(usize, usize), usize> = HashMap::new();
    let mut comm_sizes: Vec<usize> = Vec::with_capacity(communities.len());
    let mut wcc_sizes: Vec<usize> = vec![0; wccs.len()];

    for (c_idx, comm) in communities.iter().enumerate() {
        comm_sizes.push(comm.len());
        for node in comm {
            if let Some(&w_idx) = node_to_wcc.get(node.as_str()) {
                *contingency.entry((w_idx, c_idx)).or_insert(0) += 1;
                wcc_sizes[w_idx] += 1;
            }
        }
    }

    // Purity: sum(max_wcc_overlap_in_each_community) / total_nodes
    let mut max_overlap_sum = 0usize;
    for (c_idx, &size) in comm_sizes.iter().enumerate() {
        let mut max_in_comm = 0usize;
        for w_idx in 0..wccs.len() {
            if let Some(&count) = contingency.get(&(w_idx, c_idx)) {
                max_in_comm = max_in_comm.max(count);
            }
        }
        if size > 0 {
            max_overlap_sum += max_in_comm;
        }
    }
    let purity = max_overlap_sum as f64 / total_nodes as f64;

    // Adjusted Rand Index (ARI) via contingency table
    let sum_comb_n_ij: f64 = contingency
        .values()
        .map(|&n| if n >= 2 { (n * (n - 1)) / 2 } else { 0 } as f64)
        .sum();

    let sum_comb_a_i: f64 = wcc_sizes
        .iter()
        .map(|&a| if a >= 2 { (a * (a - 1)) / 2 } else { 0 } as f64)
        .sum();

    let sum_comb_b_j: f64 = comm_sizes
        .iter()
        .map(|&b| if b >= 2 { (b * (b - 1)) / 2 } else { 0 } as f64)
        .sum();

    let comb_total = if total_nodes >= 2 {
        ((total_nodes * (total_nodes - 1)) / 2) as f64
    } else {
        1.0
    };

    let expected_index = (sum_comb_a_i * sum_comb_b_j) / comb_total;
    let max_index = f64::midpoint(sum_comb_a_i, sum_comb_b_j);
    let denominator = max_index - expected_index;

    let ari = if denominator.abs() < 1e-12 {
        if (sum_comb_n_ij - expected_index).abs() < 1e-12 {
            1.0
        } else {
            0.0
        }
    } else {
        (sum_comb_n_ij - expected_index) / denominator
    };

    (purity, ari)
}

/// Detects dynasty communities using Louvain community detection on the undirected mating graph.
pub fn analyze_dynasty_communities(
    digraph: &DiGraph,
    births: &[PersistedAncestryBirth],
    deaths: &[PersistedAncestryDeath],
    latest_tick: u64,
    resolution: f64,
    seed: u64,
    threshold: f64,
    max_level: Option<usize>,
    mut timings: DynastyCommunitiesTimings,
) -> Result<DynastyCommunitiesPayload, AnalyticsError> {
    let t_start = Instant::now();

    // 1. Convert to undirected graph
    let t_undir = Instant::now();
    let undirected = digraph.to_undirected();
    timings.to_undirected_ms = t_undir.elapsed().as_secs_f64() * 1000.0;
    tracing::debug!(
        nodes = undirected.node_count(),
        edges = undirected.edge_count(),
        elapsed_ms = timings.to_undirected_ms,
        "converted lineage DiGraph to undirected Graph"
    );

    let node_count = undirected.node_count();
    let edge_count = undirected.edge_count();

    if node_count == 0 {
        timings.total_ms += t_start.elapsed().as_secs_f64() * 1000.0;
        return Ok(DynastyCommunitiesPayload {
            schema: DYNASTY_COMMUNITIES_SCHEMA_ID_V1.to_owned(),
            latest_tick,
            node_count: 0,
            edge_count: 0,
            resolution,
            seed,
            threshold,
            max_level,
            community_count: 0,
            modularity: 0.0,
            agreement_rate_with_founders: 1.0,
            rand_index_with_founders: 1.0,
            communities: Vec::new(),
            timings,
        });
    }

    // 2. Run Louvain community detection
    let t_louvain = Instant::now();
    let mut communities = fnx_algorithms::louvain_communities(
        &undirected,
        resolution,
        "weight",
        threshold,
        max_level,
        Some(seed),
    );
    timings.louvain_ms = t_louvain.elapsed().as_secs_f64() * 1000.0;

    // Sort communities by size descending
    communities.sort_by_key(|a| std::cmp::Reverse(a.len()));

    // 3. Modularity calculation
    let t_mod = Instant::now();
    let modularity = fnx_algorithms::modularity(&undirected, &communities, resolution, "weight")
        .map_err(|e| AnalyticsError::Graph(e.to_string()))?;
    timings.modularity_ms = t_mod.elapsed().as_secs_f64() * 1000.0;

    // 4. Founder components for agreement cross-check
    let wccs = fnx_algorithms::weakly_connected_components(digraph);
    let (purity, ari) = compute_contingency_agreement(&wccs, &communities, node_count);

    tracing::info!(
        communities = communities.len(),
        modularity,
        purity,
        ari,
        "Louvain dynasty communities detected"
    );

    // Build member records
    let dead_uids: HashSet<AgentUid> = deaths.iter().map(|d| d.agent_uid).collect();
    let _birth_map: HashMap<AgentUid, &PersistedAncestryBirth> =
        births.iter().map(|b| (b.agent_uid, b)).collect();

    // Map each agent to its founder WCC index
    let mut node_to_founder_wcc: HashMap<&str, usize> = HashMap::with_capacity(node_count);
    for (w_idx, comp) in wccs.iter().enumerate() {
        for node in comp {
            node_to_founder_wcc.insert(node.as_str(), w_idx);
        }
    }

    let mut community_records: Vec<DynastyCommunityRecord> = Vec::with_capacity(communities.len());

    for (rank, comm) in communities.iter().enumerate() {
        let mut founder_uids = Vec::new();
        let mut living_count = 0usize;
        let mut sample_members = Vec::new();
        let mut founder_wcc_counts: HashMap<usize, usize> = HashMap::new();

        for node_str in comm {
            if let Some(uid) = node_id_to_agent_uid(node_str) {
                if digraph.in_degree(node_str) == 0 {
                    founder_uids.push(uid.0);
                }
                if !dead_uids.contains(&uid) {
                    living_count += 1;
                }
                if sample_members.len() < 10 {
                    sample_members.push(uid.0);
                }
                if let Some(&w_idx) = node_to_founder_wcc.get(node_str.as_str()) {
                    *founder_wcc_counts.entry(w_idx).or_insert(0) += 1;
                }
            }
        }
        founder_uids.sort_unstable();

        // Dominant founder family
        let (dominant_w_idx, dominant_count) = founder_wcc_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .unwrap_or((0, 0));

        let dominant_founder_uid = wccs
            .get(dominant_w_idx)
            .and_then(|comp| {
                comp.iter()
                    .filter_map(|s| node_id_to_agent_uid(s))
                    .find(|u| digraph.in_degree(&agent_uid_to_node_id(*u)) == 0)
            })
            .map(|u| u.0);

        let dominant_fraction = if comm.is_empty() {
            0.0
        } else {
            dominant_count as f64 / comm.len() as f64
        };

        community_records.push(DynastyCommunityRecord {
            community_id: rank + 1,
            member_count: comm.len(),
            living_count,
            founder_uids,
            dominant_founder_uid,
            dominant_founder_fraction: dominant_fraction,
            sample_members,
        });
    }

    timings.total_ms += t_start.elapsed().as_secs_f64() * 1000.0;

    Ok(DynastyCommunitiesPayload {
        schema: DYNASTY_COMMUNITIES_SCHEMA_ID_V1.to_owned(),
        latest_tick,
        node_count,
        edge_count,
        resolution,
        seed,
        threshold,
        max_level,
        community_count: communities.len(),
        modularity,
        agreement_rate_with_founders: purity,
        rand_index_with_founders: ari,
        communities: community_records,
        timings,
    })
}

/// Formats the human-readable markdown representation of [`DynastyCommunitiesPayload`].
pub fn render_dynasty_communities_md(payload: &DynastyCommunitiesPayload) -> String {
    let mut out = String::with_capacity(4096);
    let _ = writeln!(out, "# Dynasty Communities Report");
    let _ = writeln!(
        out,
        "\n**Schema:** `{}` | **Latest Tick:** {}",
        payload.schema, payload.latest_tick
    );

    let _ = writeln!(out, "\n## Community Detection Overview");
    let _ = writeln!(out, "| Metric | Value | Details |");
    let _ = writeln!(out, "| :--- | :--- | :--- |");
    let _ = writeln!(
        out,
        "| Nodes (Agents) | {} | Agents in mating projection |",
        payload.node_count
    );
    let _ = writeln!(
        out,
        "| Undirected Edges | {} | Mating/ancestry connections |",
        payload.edge_count
    );
    let _ = writeln!(
        out,
        "| Communities Detected | {} | Distinct Louvain modules |",
        payload.community_count
    );
    let _ = writeln!(
        out,
        "| Modularity (Q) | {:.4} | Newman-Girvan modularity index |",
        payload.modularity
    );
    let _ = writeln!(
        out,
        "| Founder Purity Agreement | {:.4} | Overlap with genealogical founder trees |",
        payload.agreement_rate_with_founders
    );
    let _ = writeln!(
        out,
        "| Adjusted Rand Index | {:.4} | Agreement between Louvain and founder families |",
        payload.rand_index_with_founders
    );
    let _ = writeln!(
        out,
        "| Resolution | {:.2} | Louvain resolution parameter |",
        payload.resolution
    );
    let _ = writeln!(
        out,
        "| RNG Seed | 0x{:08X} | Deterministic partition seed |",
        payload.seed
    );

    let _ = writeln!(out, "\n## Top Detected Communities");
    let _ = writeln!(
        out,
        "| ID | Total Members | Living Members | Dominant Founder | Founder Share | Sample Members |"
    );
    let _ = writeln!(out, "| :--- | :--- | :--- | :--- | :--- | :--- |");
    for c in payload.communities.iter().take(10) {
        let dom_str = c
            .dominant_founder_uid
            .map(|u| format!("agent_{u}"))
            .unwrap_or_else(|| "none".to_owned());
        let sample_str = c
            .sample_members
            .iter()
            .take(4)
            .map(|u| format!("{u}"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {:.1}% | [{sample_str}] |",
            c.community_id,
            c.member_count,
            c.living_count,
            dom_str,
            c.dominant_founder_fraction * 100.0
        );
    }

    let _ = writeln!(out, "\n## Performance Telemetry");
    let _ = writeln!(
        out,
        "- Graph construction: {:.2}ms",
        payload.timings.build_digraph_ms
    );
    let _ = writeln!(
        out,
        "- Undirected projection: {:.2}ms",
        payload.timings.to_undirected_ms
    );
    let _ = writeln!(
        out,
        "- Louvain execution: {:.2}ms",
        payload.timings.louvain_ms
    );
    let _ = writeln!(
        out,
        "- Modularity evaluation: {:.2}ms",
        payload.timings.modularity_ms
    );
    let _ = writeln!(out, "- Total time: {:.2}ms", payload.timings.total_ms);

    out
}

/// `dynasty-communities`: Louvain community detection on the mating/ancestry graph.
pub struct DynastyCommunities;

impl Report for DynastyCommunities {
    fn name(&self) -> &'static str {
        "dynasty-communities"
    }

    fn description(&self) -> &'static str {
        "Louvain community detection over the mating graph with modularity and founder cross-check"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let read_start = Instant::now();
        let births = cx.reader.load_ancestry_births()?;
        let deaths = cx.reader.load_ancestry_deaths()?;
        let max_tick = cx.reader.max_tick()?.unwrap_or(0);
        log_report_stage("read", &read_start, births.len() + deaths.len());

        let resolution = params.get_f64("resolution")?.unwrap_or(1.0);
        let seed = params.get_u64("seed")?.unwrap_or(0x0114_EA9E);
        let threshold = params.get_f64("threshold")?.unwrap_or(1e-7);
        let max_level = params.get_usize("max_level")?;

        let (digraph, timings_base) = build_lineage_digraph(&births);
        let timings = DynastyCommunitiesTimings {
            build_digraph_ms: timings_base.bulk_build_edges_ms + timings_base.format_node_keys_ms,
            to_undirected_ms: 0.0,
            louvain_ms: 0.0,
            modularity_ms: 0.0,
            total_ms: 0.0,
        };

        let payload = analyze_dynasty_communities(
            &digraph, &births, &deaths, max_tick, resolution, seed, threshold, max_level, timings,
        )?;
        let md = render_dynasty_communities_md(&payload);
        let machine = serde_json::to_value(&payload)?;

        base_output("dynasty-communities", cx, payload.node_count, machine, md)
    }
}

// ---------------------------------------------------------------------------
// 3. Interaction Centrality Report
// ---------------------------------------------------------------------------

/// Execution timings for interaction centrality computation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionCentralityTimings {
    /// Time spent loading interactions from storage or replay events.
    pub load_interactions_ms: f64,
    /// Time spent bulk-building the directed interaction network.
    pub bulk_build_ms: f64,
    /// Time spent computing degree centrality.
    pub degree_centrality_ms: f64,
    /// Time spent computing betweenness centrality.
    pub betweenness_centrality_ms: f64,
    /// Time spent computing PageRank.
    pub pagerank_ms: f64,
    /// Total wall-clock time in milliseconds.
    pub total_ms: f64,
}

/// Centrality ranking record for an individual agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentCentralityScore {
    /// Rank ordinal (1-based).
    pub rank: usize,
    /// Logical unique agent identity.
    pub agent_uid: u64,
    /// Normalized centrality score.
    pub score: f64,
}

/// Degree ranking record for an individual agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentDegreeScore {
    /// Rank ordinal (1-based).
    pub rank: usize,
    /// Logical unique agent identity.
    pub agent_uid: u64,
    /// Raw integer degree count.
    pub degree: usize,
}

/// Telemetry on storage persistence and gaps for pairwise interaction records.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionPersistenceGap {
    /// Source table queried ("interactions" or "replay_events").
    pub source_table: String,
    /// Number of interaction edge rows loaded.
    pub interaction_rows_read: usize,
    /// Documented capabilities and unlock paths for pairwise persistence.
    pub documented_gap: String,
}

/// Selection and capture evidence shared verbatim by reports and graph exports.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionGraphEvidence {
    /// Canonical run identity carried by the finished reader.
    pub run_id: String,
    /// Effective population selection, including half-open bounds when requested.
    pub selection: InteractionGraphSelection,
    /// Requested limits checked before materializing graph events.
    pub budget: InteractionGraphBudget,
    /// Conservative node/edge visits for all-source unweighted betweenness.
    /// This bounds graph input work, not SQL execution time or total process memory.
    pub max_graph_work: usize,
    /// Explicit absence of a hard bound on synchronous offline SQL execution.
    pub sql_execution_bound: String,
    /// Ordering of selected event identities.
    pub ordering: String,
    /// Canonical table actually queried.
    pub source_table: String,
    /// Exact selected (tick, sequence) keys within this run.
    pub selected_event_ids: Vec<(u64, u64)>,
    /// An additional older row was observed outside the requested recent page.
    pub omitted_older_rows: bool,
    /// Count of all canonical interaction rows in the selected run.
    pub run_persisted_rows: u64,
    /// Run-wide accounting; not a fabricated count for the selected sub-window.
    pub run_capture: Option<PersistedInteractionCapture>,
    /// Scope over which the persisted counters were accumulated.
    pub capture_scope: String,
    /// Observed omissions, a consistent complete counter set, or missing evidence.
    pub capture_status: String,
    /// Whether centrality algorithms consume edge weights or simple topology.
    pub centrality_semantics: String,
    /// Meaning and units of the count and weight attributes.
    pub edge_semantics: String,
}

/// Validate selection and allocation/work bounds before any graph-input query.
pub fn load_interaction_graph_input(
    cx: &ReaderCtx,
    params: &ReportParams,
) -> Result<(Vec<InteractionGraphEvent>, InteractionGraphEvidence), AnalyticsError> {
    params.validate_keys(&[
        "start_tick",
        "end_tick",
        "limit",
        "max_projected_bytes",
        "max_graph_work",
        "sample_k",
        "seed",
        "fallback",
        "deadline_ms",
    ])?;
    let bad = |name: &str, reason: &str| AnalyticsError::BadParam {
        name: name.to_owned(),
        reason: reason.to_owned(),
    };
    let selection = match (params.get_u64("start_tick")?, params.get_u64("end_tick")?) {
        (None, None) => InteractionGraphSelection::RecentPage,
        (Some(start_tick), Some(end_tick)) if start_tick < end_tick => {
            InteractionGraphSelection::CompleteWindow {
                start_tick,
                end_tick,
            }
        }
        (Some(_), Some(_)) => return Err(bad("window", "expected start_tick < end_tick")),
        _ => {
            return Err(bad(
                "window",
                "start_tick and end_tick must be supplied together",
            ));
        }
    };
    if params.get("fallback").is_some() {
        return Err(bad(
            "fallback",
            "replay fallback is unsupported: it selects a different population",
        ));
    }
    if params.get("deadline_ms").is_some() {
        return Err(bad(
            "deadline_ms",
            "offline SQL execution has no hard deadline",
        ));
    }
    let defaults = InteractionGraphBudget::default();
    let budget = InteractionGraphBudget {
        max_rows: params.get_usize("limit")?.unwrap_or(defaults.max_rows),
        max_projected_bytes: params
            .get_usize("max_projected_bytes")?
            .unwrap_or(defaults.max_projected_bytes),
    };
    let max_graph_work = params
        .get_usize("max_graph_work")?
        .unwrap_or(6 * defaults.max_rows * defaults.max_rows);
    let required_work = budget
        .max_rows
        .checked_mul(budget.max_rows)
        .and_then(|value| value.checked_mul(6))
        .ok_or_else(|| bad("limit", "graph work bound overflows usize"))?;
    if required_work > max_graph_work {
        return Err(bad(
            "max_graph_work",
            "row limit exceeds the declared graph work budget",
        ));
    }
    let page = cx.reader.load_interaction_graph(selection, budget)?;
    let capture = cx.reader.load_interaction_capture_evidence()?;
    if let Some(capture) = capture
        && capture.persisted != page.run_persisted_rows
    {
        return Err(AnalyticsError::Graph(format!(
            "capture accounts for {} persisted interactions but the run contains {} rows",
            capture.persisted, page.run_persisted_rows
        )));
    }
    let capture_status = match capture {
        None => "unknown",
        Some(c) if c.sampled_out > 0 && c.truncated > 0 => "sampled_and_truncated",
        Some(c) if c.sampled_out > 0 => "sampled",
        Some(c) if c.truncated > 0 => "truncated",
        Some(_) => "counters_report_complete_run",
    };
    let evidence = InteractionGraphEvidence {
        run_id: cx.reader.run_id().to_string(),
        selection, budget, max_graph_work,
        sql_execution_bound: "unbounded_offline_query".to_owned(),
        ordering: "tick_ascending_then_seq_ascending".to_owned(),
        source_table: "interactions".to_owned(),
        selected_event_ids: page.events.iter().map(|row| (row.tick, row.seq)).collect(),
        omitted_older_rows: page.omitted_older_rows,
        run_persisted_rows: page.run_persisted_rows,
        run_capture: capture,
        capture_scope: "whole_run; counters do not localize omissions to a sub-window".to_owned(),
        capture_status: capture_status.to_owned(),
        centrality_semantics: "unweighted_simple_directed_graph; only interacting agents are nodes".to_owned(),
        edge_semantics: "count=selected event multiplicity; weight=sum of recorded magnitudes (combat damage plus food-share energy); not a normalized physical quantity".to_owned(),
    };
    Ok((page.events, evidence))
}

/// Machine-readable payload for `interaction-centrality`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionCentralityPayload {
    /// Schema identifier ([`INTERACTION_CENTRALITY_SCHEMA_ID_V1`]).
    pub schema: String,
    /// Latest simulation tick observed.
    pub latest_tick: u64,
    /// Start tick analyzed, if windowed.
    pub window_start_tick: Option<u64>,
    /// End tick analyzed, if windowed.
    pub window_end_tick: Option<u64>,
    /// Number of active interacting agents (nodes).
    pub node_count: usize,
    /// Number of directed interaction edges.
    pub edge_count: usize,
    /// Graph density (`edges / (nodes * (nodes - 1))`).
    pub density: f64,
    /// Top hubs by directed degree centrality.
    pub top_by_degree_centrality: Vec<AgentCentralityScore>,
    /// Top interacting agents by out-degree (most active actors).
    pub top_by_out_degree: Vec<AgentDegreeScore>,
    /// Top interacting agents by in-degree (most targeted recipients).
    pub top_by_in_degree: Vec<AgentDegreeScore>,
    /// Top interaction bridges by betweenness centrality.
    pub top_by_betweenness: Vec<AgentCentralityScore>,
    /// Top interaction authorities by PageRank.
    pub top_by_pagerank: Vec<AgentCentralityScore>,
    /// Storage persistence gap evidence and description.
    pub persistence_gap: InteractionPersistenceGap,
    /// Selection, capture and budget evidence from the actual reader.
    pub input: InteractionGraphEvidence,
    /// Exact source UIDs used by betweenness (all nodes for an exact calculation).
    pub betweenness_source_uids: Vec<u64>,
    /// Seed used to rank candidate betweenness source nodes.
    pub betweenness_seed: u64,
    /// None for an empty graph; otherwise the observed fnx convergence result.
    pub pagerank_converged: Option<bool>,
    /// Wall-clock timings for audit.
    pub timings: InteractionCentralityTimings,
}

/// Build one attributed edge per directed pair without losing event multiplicity.
/// Missing/nonfinite magnitudes and nonfinite aggregate weights are errors.
pub fn build_interaction_digraph(
    interactions: &[InteractionGraphEvent],
) -> Result<(DiGraph, f64, f64), AnalyticsError> {
    let t_fmt = Instant::now();
    let mut aggregates: BTreeMap<(AgentUid, AgentUid), (i64, f64)> = BTreeMap::new();
    for row in interactions {
        let magnitude = row
            .magnitude
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                AnalyticsError::Graph(format!(
                    "missing or nonfinite magnitude at ({}, {})",
                    row.tick, row.seq
                ))
            })?;
        let (count, weight) = aggregates.entry((row.actor, row.target)).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| AnalyticsError::Graph("edge count overflow".to_owned()))?;
        *weight += magnitude;
        if !weight.is_finite() {
            return Err(AnalyticsError::Graph(format!(
                "nonfinite aggregate magnitude at ({}, {})",
                row.tick, row.seq
            )));
        }
    }
    let format_ms = t_fmt.elapsed().as_secs_f64() * 1000.0;
    let t_bulk = Instant::now();
    let mut digraph = DiGraph::strict();
    for ((actor, target), (count, weight)) in aggregates {
        let attrs = AttrMap::from([
            ("count".to_owned(), count.into()),
            ("weight".to_owned(), weight.into()),
        ]);
        digraph
            .add_edge_with_attrs(
                agent_uid_to_node_id(actor),
                agent_uid_to_node_id(target),
                attrs,
            )
            .map_err(|error| AnalyticsError::Graph(error.to_string()))?;
    }
    let bulk_ms = t_bulk.elapsed().as_secs_f64() * 1000.0;
    Ok((digraph, format_ms, bulk_ms))
}

/// Computes interaction centrality metrics over the directed interaction network.
pub fn analyze_interaction_centrality(
    digraph: &DiGraph,
    input: InteractionGraphEvidence,
    latest_tick: u64,
    sample_k: Option<usize>,
    seed: u64,
    mut timings: InteractionCentralityTimings,
) -> InteractionCentralityPayload {
    let t_start = Instant::now();
    let node_count = digraph.node_count();
    let edge_count = digraph.edge_count();
    let rows_read = input.selected_event_ids.len();
    let (start_tick, end_tick) = match input.selection {
        InteractionGraphSelection::RecentPage => (None, None),
        InteractionGraphSelection::CompleteWindow {
            start_tick,
            end_tick,
        } => (Some(start_tick), Some(end_tick)),
    };

    let density = if node_count > 1 {
        edge_count as f64 / (node_count * (node_count - 1)) as f64
    } else {
        0.0
    };

    if node_count == 0 {
        timings.total_ms += t_start.elapsed().as_secs_f64() * 1000.0;
        return InteractionCentralityPayload {
            schema: INTERACTION_CENTRALITY_SCHEMA_ID_V1.to_owned(),
            latest_tick,
            window_start_tick: start_tick,
            window_end_tick: end_tick,
            node_count: 0,
            edge_count: 0,
            density: 0.0,
            top_by_degree_centrality: Vec::new(),
            top_by_out_degree: Vec::new(),
            top_by_in_degree: Vec::new(),
            top_by_betweenness: Vec::new(),
            top_by_pagerank: Vec::new(),
            persistence_gap: InteractionPersistenceGap {
                source_table: input.source_table.clone(),
                interaction_rows_read: rows_read,
                documented_gap: "No persisted interaction rows selected; this alone does not establish absence of encounters.".to_owned(),
            },
            input,
            betweenness_source_uids: Vec::new(),
            betweenness_seed: seed,
            pagerank_converged: None,
            timings,
        };
    }

    // 1. Degree Centrality
    let t_deg = Instant::now();
    let deg_result = fnx_algorithms::degree_centrality_directed(digraph);
    timings.degree_centrality_ms = t_deg.elapsed().as_secs_f64() * 1000.0;

    let mut deg_scores: Vec<AgentCentralityScore> = deg_result
        .scores
        .into_iter()
        .filter_map(|s| {
            node_id_to_agent_uid(&s.node).map(|u| AgentCentralityScore {
                rank: 0,
                agent_uid: u.0,
                score: s.score,
            })
        })
        .collect();
    deg_scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.agent_uid.cmp(&b.agent_uid))
    });
    for (i, item) in deg_scores.iter_mut().enumerate() {
        item.rank = i + 1;
    }
    deg_scores.truncate(10);

    // Raw in/out degrees
    let mut in_degrees: Vec<AgentDegreeScore> = Vec::with_capacity(node_count);
    let mut out_degrees: Vec<AgentDegreeScore> = Vec::with_capacity(node_count);

    for node_name in digraph.nodes_ordered() {
        if let Some(u) = node_id_to_agent_uid(node_name) {
            in_degrees.push(AgentDegreeScore {
                rank: 0,
                agent_uid: u.0,
                degree: digraph.in_degree(node_name),
            });
            out_degrees.push(AgentDegreeScore {
                rank: 0,
                agent_uid: u.0,
                degree: digraph.out_degree(node_name),
            });
        }
    }
    in_degrees.sort_by(|a, b| b.degree.cmp(&a.degree).then(a.agent_uid.cmp(&b.agent_uid)));
    for (i, item) in in_degrees.iter_mut().enumerate() {
        item.rank = i + 1;
    }
    in_degrees.truncate(10);

    out_degrees.sort_by(|a, b| b.degree.cmp(&a.degree).then(a.agent_uid.cmp(&b.agent_uid)));
    for (i, item) in out_degrees.iter_mut().enumerate() {
        item.rank = i + 1;
    }
    out_degrees.truncate(10);

    // 2. Betweenness Centrality
    let t_bet = Instant::now();
    let mut sources = digraph.nodes_ordered();
    let sampled = node_count > 1000 || sample_k.is_some();
    if sampled {
        let k = sample_k.unwrap_or(100).min(node_count);
        sources.sort_by_cached_key(|node| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&seed.to_le_bytes());
            hasher.update(node.as_bytes());
            (*hasher.finalize().as_bytes(), *node)
        });
        sources.truncate(k);
    }
    let betweenness_source_uids = sources
        .iter()
        .filter_map(|node| node_id_to_agent_uid(node).map(|uid| uid.0))
        .collect();
    let bet_result = if sampled {
        fnx_algorithms::betweenness_centrality_sampled_directed_with_params(
            digraph, &sources, true, false,
        )
    } else {
        fnx_algorithms::betweenness_centrality_directed(digraph)
    };
    timings.betweenness_centrality_ms = t_bet.elapsed().as_secs_f64() * 1000.0;

    let mut bet_scores: Vec<AgentCentralityScore> = bet_result
        .scores
        .into_iter()
        .filter_map(|s| {
            node_id_to_agent_uid(&s.node).map(|u| AgentCentralityScore {
                rank: 0,
                agent_uid: u.0,
                score: s.score,
            })
        })
        .collect();
    bet_scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.agent_uid.cmp(&b.agent_uid))
    });
    for (i, item) in bet_scores.iter_mut().enumerate() {
        item.rank = i + 1;
    }
    bet_scores.truncate(10);

    // 3. PageRank
    let t_pr = Instant::now();
    let pr_result = fnx_algorithms::pagerank_directed(digraph);
    let pagerank_converged = Some(pr_result.converged);
    timings.pagerank_ms = t_pr.elapsed().as_secs_f64() * 1000.0;

    let mut pr_scores: Vec<AgentCentralityScore> = pr_result
        .scores
        .into_iter()
        .filter_map(|s| {
            node_id_to_agent_uid(&s.node).map(|u| AgentCentralityScore {
                rank: 0,
                agent_uid: u.0,
                score: s.score,
            })
        })
        .collect();
    pr_scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.agent_uid.cmp(&b.agent_uid))
    });
    for (i, item) in pr_scores.iter_mut().enumerate() {
        item.rank = i + 1;
    }
    pr_scores.truncate(10);

    timings.total_ms += t_start.elapsed().as_secs_f64() * 1000.0;

    let documented_gap = format!(
        "Selected {rows_read} canonical persisted interaction records. Run capture: {}. \
        Older rows omitted by selection: {}. Capture counters describe the whole run; \
        degree, betweenness and PageRank use unweighted unique directed edges.",
        input.capture_status, input.omitted_older_rows
    );

    InteractionCentralityPayload {
        schema: INTERACTION_CENTRALITY_SCHEMA_ID_V1.to_owned(),
        latest_tick,
        window_start_tick: start_tick,
        window_end_tick: end_tick,
        node_count,
        edge_count,
        density,
        top_by_degree_centrality: deg_scores,
        top_by_out_degree: out_degrees,
        top_by_in_degree: in_degrees,
        top_by_betweenness: bet_scores,
        top_by_pagerank: pr_scores,
        persistence_gap: InteractionPersistenceGap {
            source_table: input.source_table.clone(),
            interaction_rows_read: rows_read,
            documented_gap,
        },
        input,
        betweenness_source_uids,
        betweenness_seed: seed,
        pagerank_converged,
        timings,
    }
}

/// Formats the human-readable markdown representation of [`InteractionCentralityPayload`].
pub fn render_interaction_centrality_md(payload: &InteractionCentralityPayload) -> String {
    let mut out = String::with_capacity(4096);
    let _ = writeln!(out, "# Interaction Centrality Report");
    let _ = writeln!(
        out,
        "\n**Schema:** `{}` | **Latest Tick:** {}",
        payload.schema, payload.latest_tick
    );

    let _ = writeln!(out, "\n## Interaction Network Overview");
    let _ = writeln!(
        out,
        "\nSelection: `{:?}`; ordering: `{}`. Capture: `{}` (whole run).",
        payload.input.selection, payload.input.ordering, payload.input.capture_status
    );
    let _ = writeln!(
        out,
        "\n{}\n\n{}",
        payload.input.centrality_semantics, payload.input.edge_semantics
    );
    let _ = writeln!(
        out,
        "\nInput budgets: {} events, {} projected bytes, {} graph-work units. SQL execution has no hard deadline.",
        payload.input.budget.max_rows,
        payload.input.budget.max_projected_bytes,
        payload.input.max_graph_work
    );
    let _ = writeln!(out, "| Metric | Value | Interpretation |");
    let _ = writeln!(out, "| :--- | :--- | :--- |");
    let _ = writeln!(
        out,
        "| Interacting Agents (Nodes) | {} | Agents with combat or food-share actions |",
        payload.node_count
    );
    let _ = writeln!(
        out,
        "| Directed Interaction Edges | {} | Unique directed actor-target connections |",
        payload.edge_count
    );
    let _ = writeln!(
        out,
        "| Graph Density | {:.6} | Directed edge density |",
        payload.density
    );
    let _ = writeln!(
        out,
        "| Source Data Table | {} | Total records read: {} |",
        payload.persistence_gap.source_table, payload.persistence_gap.interaction_rows_read
    );

    let _ = writeln!(out, "\n## Top Hubs by Centrality");

    let _ = writeln!(out, "\n### Top Instigators (Out-Degree)");
    let _ = writeln!(
        out,
        "| Rank | Agent | Out-Degree (Targets Interacted With) |"
    );
    let _ = writeln!(out, "| :--- | :--- | :--- |");
    for h in payload.top_by_out_degree.iter().take(5) {
        let _ = writeln!(out, "| {} | agent_{} | {} |", h.rank, h.agent_uid, h.degree);
    }

    let _ = writeln!(out, "\n### Top Recipients (In-Degree)");
    let _ = writeln!(out, "| Rank | Agent | In-Degree (Actors Interacted By) |");
    let _ = writeln!(out, "| :--- | :--- | :--- |");
    for h in payload.top_by_in_degree.iter().take(5) {
        let _ = writeln!(out, "| {} | agent_{} | {} |", h.rank, h.agent_uid, h.degree);
    }

    let _ = writeln!(out, "\n### Top Network Bridges (Betweenness Centrality)");
    let _ = writeln!(out, "| Rank | Agent | Betweenness Score |");
    let _ = writeln!(out, "| :--- | :--- | :--- |");
    for h in payload.top_by_betweenness.iter().take(5) {
        let _ = writeln!(
            out,
            "| {} | agent_{} | {:.6} |",
            h.rank, h.agent_uid, h.score
        );
    }

    let _ = writeln!(out, "\n### Top Influencers (PageRank)");
    let _ = writeln!(out, "| Rank | Agent | PageRank Score |");
    let _ = writeln!(out, "| :--- | :--- | :--- |");
    for h in payload.top_by_pagerank.iter().take(5) {
        let _ = writeln!(
            out,
            "| {} | agent_{} | {:.6} |",
            h.rank, h.agent_uid, h.score
        );
    }

    let _ = writeln!(out, "\n## Persistence Gap Analysis");
    let _ = writeln!(out, "> {}", payload.persistence_gap.documented_gap);

    let _ = writeln!(out, "\n## Performance Telemetry");
    let _ = writeln!(
        out,
        "- Interaction loading: {:.2}ms",
        payload.timings.load_interactions_ms
    );
    let _ = writeln!(
        out,
        "- Graph bulk build: {:.2}ms",
        payload.timings.bulk_build_ms
    );
    let _ = writeln!(
        out,
        "- Degree centrality: {:.2}ms",
        payload.timings.degree_centrality_ms
    );
    let _ = writeln!(
        out,
        "- Betweenness centrality: {:.2}ms",
        payload.timings.betweenness_centrality_ms
    );
    let _ = writeln!(out, "- PageRank: {:.2}ms", payload.timings.pagerank_ms);
    let _ = writeln!(out, "- Total time: {:.2}ms", payload.timings.total_ms);

    out
}

/// `interaction-centrality`: Directed interaction network centrality analysis (degree, betweenness, `PageRank`).
pub struct InteractionCentrality;

impl Report for InteractionCentrality {
    fn name(&self) -> &'static str {
        "interaction-centrality"
    }

    fn description(&self) -> &'static str {
        "Pairwise interaction networks: degree centrality, betweenness, PageRank, and persistence gaps"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let read_start = Instant::now();
        let sample_k = params.get_usize("sample_k")?;
        if sample_k == Some(0) {
            return Err(AnalyticsError::BadParam {
                name: "sample_k".to_owned(),
                reason: "must be positive when supplied".to_owned(),
            });
        }
        let seed = params.get_u64("seed")?.unwrap_or(0x0114_EA9E);
        let (interactions, evidence) = load_interaction_graph_input(cx, params)?;
        let max_tick = cx.reader.max_tick()?.unwrap_or(0);

        let load_interactions_ms = read_start.elapsed().as_secs_f64() * 1000.0;
        log_report_stage("read_interactions", &read_start, interactions.len());

        let (digraph, format_ms, bulk_ms) = build_interaction_digraph(&interactions)?;

        let timings = InteractionCentralityTimings {
            load_interactions_ms,
            bulk_build_ms: format_ms + bulk_ms,
            degree_centrality_ms: 0.0,
            betweenness_centrality_ms: 0.0,
            pagerank_ms: 0.0,
            total_ms: load_interactions_ms + format_ms + bulk_ms,
        };

        let payload =
            analyze_interaction_centrality(&digraph, evidence, max_tick, sample_k, seed, timings);

        let md = render_interaction_centrality_md(&payload);
        let machine = serde_json::to_value(&payload)?;

        base_output(
            "interaction-centrality",
            cx,
            payload.node_count,
            machine,
            md,
        )
    }
}

// ---------------------------------------------------------------------------
// 4. GraphML and Edge-List Export Functions
// ---------------------------------------------------------------------------

/// Formats that preserve both interaction edge attributes and selection metadata.
#[derive(Clone, Copy)]
pub enum InteractionGraphFormat {
    /// GraphML with typed edge attributes and graph-level JSON evidence.
    GraphMl,
    /// fnx attributed edge-list syntax with a JSON evidence comment header.
    EdgeList,
}

/// Export attributed interactions with the exact selection/capture provenance.
pub fn export_interaction_graph(
    digraph: &DiGraph,
    evidence: &InteractionGraphEvidence,
    format: InteractionGraphFormat,
) -> Result<String, AnalyticsError> {
    let metadata = serde_json::to_string(evidence)?;
    match format {
        InteractionGraphFormat::GraphMl => {
            let attrs = AttrMap::from([(
                "scriptbots_interaction_evidence".to_owned(),
                metadata.into(),
            )]);
            fnx_readwrite::EdgeListEngine::strict()
                .write_digraph_graphml_with_graph_attrs(digraph, &attrs)
                .map_err(|error| AnalyticsError::Graph(error.to_string()))
        }
        InteractionGraphFormat::EdgeList => {
            let body = export_digraph_edgelist(digraph)?;
            Ok(format!(
                "# scriptbots.interaction-edgelist.v1 {metadata}\n{body}\n"
            ))
        }
    }
}

/// Serializes a directed graph to standard `GraphML` format using [`fnx_readwrite::EdgeListEngine`].
pub fn export_digraph_graphml(digraph: &DiGraph) -> Result<String, AnalyticsError> {
    let mut engine = fnx_readwrite::EdgeListEngine::strict();
    engine
        .write_digraph_graphml(digraph)
        .map_err(|e| AnalyticsError::Graph(e.to_string()))
}

/// Serializes an undirected graph to standard `GraphML` format using [`fnx_readwrite::EdgeListEngine`].
pub fn export_graph_graphml(graph: &Graph) -> Result<String, AnalyticsError> {
    let mut engine = fnx_readwrite::EdgeListEngine::strict();
    engine
        .write_graphml(graph)
        .map_err(|e| AnalyticsError::Graph(e.to_string()))
}

/// Serializes a directed graph to standard space-delimited edge-list format.
pub fn export_digraph_edgelist(digraph: &DiGraph) -> Result<String, AnalyticsError> {
    let mut engine = fnx_readwrite::EdgeListEngine::strict();
    engine
        .write_digraph_edgelist(digraph)
        .map_err(|e| AnalyticsError::Graph(e.to_string()))
}

/// Serializes an undirected graph to standard space-delimited edge-list format.
pub fn export_graph_edgelist(graph: &Graph) -> Result<String, AnalyticsError> {
    let mut engine = fnx_readwrite::EdgeListEngine::strict();
    engine
        .write_edgelist(graph)
        .map_err(|e| AnalyticsError::Graph(e.to_string()))
}
