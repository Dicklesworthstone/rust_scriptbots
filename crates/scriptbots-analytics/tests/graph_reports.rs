#![allow(clippy::all, clippy::pedantic, clippy::nursery)]

//! Integration and unit tests for fnx-backed graph reports (bd-2z0.11.7).
//!
//! Tests:
//! 1. Golden mini genealogy (3 founders, known components/communities) asserting exact fnx outputs.
//! 2. Determinism: seeded Louvain community detection produces identical results across runs.
//! 3. Graph exports: `GraphML` and Edge-List serialization and structure verification.
//! 4. Interaction centrality: directed degree, betweenness, `PageRank`, and persistence gap docs.
//! 5. Scale smoke: 100k-edge synthetic DAG under time budget with stage timings logged.
//! 6. Real SQLite fixture DB integration: running reports through Registry and `sb-analyze` CLI.

use std::process::Command;
use std::time::Instant;

use scriptbots_analytics::{
    DYNASTY_COMMUNITIES_SCHEMA_ID_V1, INTERACTION_CENTRALITY_SCHEMA_ID_V1,
    LINEAGE_STRUCTURE_SCHEMA_ID_V1, REPORT_SCHEMA_VERSION, ReaderCtx, Registry, ReportParams,
    analyze_dynasty_communities, analyze_interaction_centrality, analyze_lineage_structure,
    build_interaction_digraph, build_lineage_digraph, export_digraph_edgelist,
    export_digraph_graphml, export_graph_edgelist, export_graph_graphml,
};
use scriptbots_core::{
    AgentUid, BirthOrigin, BirthRecord, DeathCause, DeathRecord, Generation, PersistenceBatch,
    PersistenceEvent, PersistenceEventKind, Position, Tick, TickSummary,
};
use scriptbots_storage::{
    PersistedAncestryBirth, PersistedAncestryDeath, PersistedInteraction, Storage,
};

fn sample_birth(
    uid: u64,
    parent_a: Option<u64>,
    parent_b: Option<u64>,
    generation: u32,
    tick: u64,
) -> PersistedAncestryBirth {
    PersistedAncestryBirth {
        tick: Tick(tick),
        agent_uid: AgentUid(uid),
        spawn_ordinal: uid,
        birth_ordinal: if parent_a.is_some() { Some(uid) } else { None },
        parent_a: parent_a.map(AgentUid),
        parent_b: parent_b.map(AgentUid),
        generation: Generation(generation),
        brain_key: Some(1),
        is_hybrid: parent_a.is_some() && parent_b.is_some(),
        origin: if parent_a.is_some() {
            BirthOrigin::Born
        } else {
            BirthOrigin::Seeded
        },
    }
}

/// Hand-built mini genealogy with 3 founders and known ground-truth structure:
/// - Family 1 (Founder 100):
///   - 101 born to (100)
///   - 102 born to (100, 101)
///   - 103 born to (101, 102)
///   Total: 4 agents, longest path: 100 -> 101 -> 102 -> 103 (length 3).
/// - Family 2 (Founder 200):
///   - 201 born to (200)
///   Total: 2 agents, longest path: 200 -> 201 (length 1).
/// - Family 3 (Founder 300):
///   - Isolated founder with 0 children.
///   Total: 1 agent, longest path length 0.
///
/// Total nodes = 7, total directed edges = 6.
fn mini_genealogy() -> (Vec<PersistedAncestryBirth>, Vec<PersistedAncestryDeath>) {
    let births = vec![
        sample_birth(100, None, None, 0, 0),
        sample_birth(200, None, None, 0, 0),
        sample_birth(300, None, None, 0, 0),
        sample_birth(101, Some(100), None, 1, 5),
        sample_birth(201, Some(200), None, 1, 6),
        sample_birth(102, Some(100), Some(101), 2, 10),
        sample_birth(103, Some(101), Some(102), 3, 15),
    ];

    // Deaths: 100 and 200 died, 201 died (family 2 extinct).
    // Living: 101, 102, 103 (family 1 surviving) and 300 (family 3 surviving).
    let deaths = vec![
        PersistedAncestryDeath {
            tick: Tick(12),
            agent_uid: AgentUid(100),
            cause: DeathCause::Aging,
        },
        PersistedAncestryDeath {
            tick: Tick(14),
            agent_uid: AgentUid(200),
            cause: DeathCause::Starvation,
        },
        PersistedAncestryDeath {
            tick: Tick(18),
            agent_uid: AgentUid(201),
            cause: DeathCause::Starvation,
        },
    ];

    (births, deaths)
}

#[test]
fn golden_mini_genealogy_asserts_exact_fnx_outputs() {
    let (births, deaths) = mini_genealogy();
    let (digraph, timings) = build_lineage_digraph(&births);

    assert_eq!(digraph.node_count(), 7, "exact 7 nodes");
    assert_eq!(digraph.edge_count(), 6, "exact 6 edges");

    let payload = analyze_lineage_structure(&digraph, &births, &deaths, 20, timings);

    assert_eq!(payload.schema, LINEAGE_STRUCTURE_SCHEMA_ID_V1);
    assert_eq!(payload.node_count, 7);
    assert_eq!(payload.edge_count, 6);
    assert!(payload.is_dag, "lineage must be an acyclic graph");
    assert_eq!(payload.longest_path_length, 3, "max generation depth 3");
    assert_eq!(
        payload.longest_path_sample,
        vec![100, 101, 102, 103],
        "exact longest path sequence"
    );

    // Connected components
    assert_eq!(
        payload.weakly_connected_components_count, 3,
        "3 distinct founder families"
    );
    assert_eq!(
        payload.strongly_connected_components_count, 7,
        "7 trivial SCCs"
    );
    assert_eq!(
        payload.non_trivial_scc_count, 0,
        "zero non-trivial SCCs (no cycles)"
    );

    // In-degree distribution
    assert_eq!(payload.in_degree_summary.zero_parents_founders, 3); // 100, 200, 300
    assert_eq!(payload.in_degree_summary.one_parent, 2); // 101, 201
    assert_eq!(payload.in_degree_summary.two_parents_sexual, 2); // 102, 103
    assert_eq!(payload.in_degree_summary.max_in_degree, 2);

    // Out-degree distribution
    assert_eq!(payload.out_degree_summary.zero_offspring, 3); // 103, 201, 300
    assert_eq!(payload.out_degree_summary.max_offspring, 2); // 100 and 101 each have 2 children

    // Founder families ranking
    assert_eq!(payload.founder_families.len(), 3);
    let f1 = &payload.founder_families[0];
    assert_eq!(f1.primary_founder_uid, 100);
    assert_eq!(f1.total_members, 4);
    assert_eq!(f1.living_members, 3);
    assert!(f1.surviving);
    assert_eq!(f1.max_generation, 3);

    let f2 = &payload.founder_families[1];
    assert_eq!(f2.primary_founder_uid, 200);
    assert_eq!(f2.total_members, 2);
    assert_eq!(f2.living_members, 0);
    assert!(!f2.surviving, "family 2 is extinct");
    assert_eq!(f2.max_generation, 1);

    let f3 = &payload.founder_families[2];
    assert_eq!(f3.primary_founder_uid, 300);
    assert_eq!(f3.total_members, 1);
    assert_eq!(f3.living_members, 1);
    assert!(f3.surviving, "family 3 founder is still alive");
    assert_eq!(f3.max_generation, 0);

    // Demographic turnover
    assert_eq!(payload.extinction_depth.total_founder_families, 3);
    assert_eq!(payload.extinction_depth.extinct_families, 1);
    assert_eq!(payload.extinction_depth.surviving_families, 2);
    assert!(
        (payload.extinction_depth.turnover_rate - 1.0 / 3.0).abs() < 1e-6,
        "turnover rate is 1/3"
    );
}

#[test]
fn dynasty_communities_louvain_determinism() {
    let (births, deaths) = mini_genealogy();
    let (digraph, timings_base) = build_lineage_digraph(&births);

    let timings1 = scriptbots_analytics::DynastyCommunitiesTimings {
        build_digraph_ms: timings_base.bulk_build_edges_ms,
        to_undirected_ms: 0.0,
        louvain_ms: 0.0,
        modularity_ms: 0.0,
        total_ms: 0.0,
    };
    let timings2 = timings1.clone();

    let res1 = analyze_dynasty_communities(
        &digraph,
        &births,
        &deaths,
        20,
        1.0,
        0x1234_5678,
        1e-7,
        None,
        timings1,
    )
    .expect("run 1");

    let res2 = analyze_dynasty_communities(
        &digraph,
        &births,
        &deaths,
        20,
        1.0,
        0x1234_5678,
        1e-7,
        None,
        timings2,
    )
    .expect("run 2");

    assert_eq!(res1.schema, DYNASTY_COMMUNITIES_SCHEMA_ID_V1);
    assert_eq!(res1.community_count, res2.community_count);
    assert_eq!(res1.modularity, res2.modularity);
    assert!(
        res1.modularity > 0.0,
        "modularity must be strictly positive on modular network"
    );
    assert_eq!(
        res1.agreement_rate_with_founders, 1.0,
        "perfect agreement between communities and disconnected founder trees"
    );
    assert_eq!(
        res1.rand_index_with_founders, 1.0,
        "perfect adjusted rand index"
    );

    // Assert exact community membership equality
    for (c1, c2) in res1.communities.iter().zip(res2.communities.iter()) {
        assert_eq!(c1.community_id, c2.community_id);
        assert_eq!(c1.member_count, c2.member_count);
        assert_eq!(c1.dominant_founder_uid, c2.dominant_founder_uid);
        assert_eq!(c1.sample_members, c2.sample_members);
    }
}

#[test]
fn graph_export_graphml_and_edgelist_roundtrip() {
    let (births, _) = mini_genealogy();
    let (digraph, _) = build_lineage_digraph(&births);
    let undirected = digraph.to_undirected();

    // 1. Directed GraphML export
    let digraph_graphml = export_digraph_graphml(&digraph).expect("export digraph graphml");
    assert!(digraph_graphml.contains("<graphml"));
    assert!(digraph_graphml.contains("edgedefault=\"directed\""));
    assert!(digraph_graphml.contains("agent_100"));
    assert!(digraph_graphml.contains("agent_103"));

    // 2. Directed edge list export
    let digraph_edgelist = export_digraph_edgelist(&digraph).expect("export digraph edgelist");
    assert!(
        digraph_edgelist.contains("agent_100 agent_101")
            || digraph_edgelist.contains("agent_100\tagent_101")
    );
    assert!(
        digraph_edgelist.contains("agent_200 agent_201")
            || digraph_edgelist.contains("agent_200\tagent_201")
    );

    // 3. Undirected GraphML export
    let graph_graphml = export_graph_graphml(&undirected).expect("export undirected graphml");
    assert!(graph_graphml.contains("<graphml"));
    assert!(graph_graphml.contains("edgedefault=\"undirected\""));

    // 4. Undirected edge list export
    let graph_edgelist = export_graph_edgelist(&undirected).expect("export undirected edgelist");
    assert!(!graph_edgelist.is_empty());
}

#[test]
fn interaction_centrality_on_synthetic_interactions() {
    let interactions = vec![
        PersistedInteraction {
            tick: 1,
            seq: 1,
            kind: "combat".to_owned(),
            actor: AgentUid(1),
            target: AgentUid(2),
            value: Some(10.0),
            actor_position: None,
            target_position: None,
            payload: String::new(),
        },
        PersistedInteraction {
            tick: 2,
            seq: 2,
            kind: "combat".to_owned(),
            actor: AgentUid(1),
            target: AgentUid(3),
            value: Some(15.0),
            actor_position: None,
            target_position: None,
            payload: String::new(),
        },
        PersistedInteraction {
            tick: 3,
            seq: 3,
            kind: "combat".to_owned(),
            actor: AgentUid(1),
            target: AgentUid(4),
            value: Some(20.0),
            actor_position: None,
            target_position: None,
            payload: String::new(),
        },
        PersistedInteraction {
            tick: 4,
            seq: 4,
            kind: "food_share".to_owned(),
            actor: AgentUid(2),
            target: AgentUid(3),
            value: Some(5.0),
            actor_position: None,
            target_position: None,
            payload: String::new(),
        },
        PersistedInteraction {
            tick: 5,
            seq: 5,
            kind: "combat".to_owned(),
            actor: AgentUid(3),
            target: AgentUid(4),
            value: Some(25.0),
            actor_position: None,
            target_position: None,
            payload: String::new(),
        },
    ];

    let (digraph, format_ms, bulk_ms) = build_interaction_digraph(&interactions);
    assert_eq!(digraph.node_count(), 4);
    assert_eq!(digraph.edge_count(), 5);

    let timings = scriptbots_analytics::InteractionCentralityTimings {
        load_interactions_ms: 1.0,
        bulk_build_ms: format_ms + bulk_ms,
        degree_centrality_ms: 0.0,
        betweenness_centrality_ms: 0.0,
        pagerank_ms: 0.0,
        total_ms: 0.0,
    };

    let payload = analyze_interaction_centrality(
        &digraph,
        "interactions",
        interactions.len(),
        10,
        None,
        None,
        None,
        0x114_EA9E,
        timings,
    );

    assert_eq!(payload.schema, INTERACTION_CENTRALITY_SCHEMA_ID_V1);
    assert_eq!(payload.node_count, 4);
    assert_eq!(payload.edge_count, 5);

    // Out-degree: agent 1 has out-degree 3
    assert_eq!(payload.top_by_out_degree[0].agent_uid, 1);
    assert_eq!(payload.top_by_out_degree[0].degree, 3);

    // In-degree: agents 3 and 4 each have in-degree 2
    assert!(payload.top_by_in_degree[0].degree == 2);

    // Centralities computed
    assert!(!payload.top_by_degree_centrality.is_empty());
    assert!(!payload.top_by_betweenness.is_empty());
    assert!(!payload.top_by_pagerank.is_empty());

    // Persistence gap documented
    assert!(
        payload
            .persistence_gap
            .documented_gap
            .contains("bd-2z0.5.9")
    );
}

#[test]
fn scale_smoke_100k_edge_synthetic_dag() {
    let num_edges = 100_000usize;
    let mut births = Vec::with_capacity(num_edges + 1);

    // Create a 100k linear chain: 0 -> 1 -> 2 -> ... -> 100,000
    births.push(sample_birth(0, None, None, 0, 0));
    for i in 1..=num_edges as u64 {
        births.push(sample_birth(i, Some(i - 1), None, i as u32, i));
    }

    let t_total = Instant::now();
    let (digraph, timings) = build_lineage_digraph(&births);

    assert_eq!(digraph.node_count(), num_edges + 1);
    assert_eq!(digraph.edge_count(), num_edges);

    let t_wcc = Instant::now();
    let wccs = fnx_algorithms::weakly_connected_components(&digraph);
    let wcc_ms = t_wcc.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(wccs.len(), 1, "single connected component");

    let t_dag = Instant::now();
    let is_dag = fnx_algorithms::is_directed_acyclic_graph(&digraph);
    let is_dag_ms = t_dag.elapsed().as_secs_f64() * 1000.0;
    assert!(is_dag);

    let total_elapsed = t_total.elapsed();
    eprintln!(
        "[SCALE SMOKE 100K EDGES] format_fmt={:.2}ms, bulk_build={:.2}ms, wcc={:.2}ms, is_dag={:.2}ms, total={:?}",
        timings.format_node_keys_ms, timings.bulk_build_edges_ms, wcc_ms, is_dag_ms, total_elapsed
    );

    // Assert time budget: total build + WCC + DAG check runs well within 15 seconds
    assert!(
        total_elapsed.as_secs() < 15,
        "100k edge DAG analysis exceeded 15s budget: {total_elapsed:?}"
    );
}

fn birth_rec(
    tick: u64,
    uid: u64,
    parent_a: Option<u64>,
    parent_b: Option<u64>,
    generation: u32,
    origin: BirthOrigin,
) -> BirthRecord {
    BirthRecord {
        tick: Tick(tick),
        agent_uid: AgentUid(uid),
        spawn_ordinal: uid,
        birth_ordinal: if origin == BirthOrigin::Born {
            Some(uid)
        } else {
            None
        },
        origin,
        parent_a: parent_a.map(AgentUid),
        parent_b: parent_b.map(AgentUid),
        brain_kind: Some("mlp".to_owned()),
        brain_key: Some(1),
        herbivore_tendency: 0.5,
        generation: Generation(generation),
        position: Position::new(0.0, 0.0),
        is_hybrid: parent_a.is_some() && parent_b.is_some(),
    }
}

fn death_rec(tick: u64, uid: u64, age: u32, cause: DeathCause) -> DeathRecord {
    DeathRecord {
        tick: Tick(tick),
        agent_uid: AgentUid(uid),
        age,
        generation: Generation(0),
        herbivore_tendency: 0.5,
        brain_kind: Some("mlp".to_owned()),
        brain_key: Some(1),
        energy: 0.0,
        food_balance_total: 0.0,
        cause,
        was_hybrid: false,
        combat_flags: scriptbots_core::CombatEventFlags::default(),
    }
}

fn make_tick_batch(tick: u64, births_count: usize, deaths_count: usize) -> PersistenceBatch {
    let mut events = Vec::new();
    if births_count > 0 {
        events.push(PersistenceEvent::new(
            PersistenceEventKind::Births,
            births_count,
        ));
    }
    if deaths_count > 0 {
        events.push(PersistenceEvent::new(
            PersistenceEventKind::Deaths,
            deaths_count,
        ));
    }
    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(tick),
            agent_count: 0,
            births: births_count,
            deaths: deaths_count,
            total_energy: 0.0,
            average_energy: 0.0,
            average_health: 1.0,
            max_age: 0,
            spike_hits: 0,
        },
        epoch: 1,
        closed: false,
        metrics: Vec::new(),
        events,
        agents: Vec::new(),
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events: Vec::new(),
        narrative_events: Vec::new(),
        genomes: Vec::new(),
    }
}

#[test]
fn fixture_db_integration_e2e() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir
        .path()
        .join("graph_fixture.sqlite")
        .display()
        .to_string();
    let mut storage = Storage::create_unattributed_file(&db_path).expect("create test db");

    // Tick 0: 3 founders (seeded arrivals do not count as demographic births)
    let mut b0 = make_tick_batch(0, 0, 0);
    b0.births
        .push(birth_rec(0, 10, None, None, 0, BirthOrigin::Seeded));
    b0.births
        .push(birth_rec(0, 20, None, None, 0, BirthOrigin::Seeded));
    b0.births
        .push(birth_rec(0, 30, None, None, 0, BirthOrigin::Seeded));
    storage.persist(&b0).expect("persist tick 0");

    // Tick 5: 2 offspring born
    let mut b5 = make_tick_batch(5, 2, 0);
    b5.births
        .push(birth_rec(5, 11, Some(10), None, 1, BirthOrigin::Born));
    b5.births
        .push(birth_rec(5, 21, Some(20), None, 1, BirthOrigin::Born));
    storage.persist(&b5).expect("persist tick 5");

    // Tick 10: 1 sexual offspring + replay interaction event
    let mut b10 = make_tick_batch(10, 1, 0);
    b10.births
        .push(birth_rec(10, 12, Some(10), Some(11), 2, BirthOrigin::Born));
    b10.replay_events.push(scriptbots_core::ReplayEvent {
        agent_uid: Some(AgentUid(10)),
        position: Some(Position::new(1.0, 2.0)),
        counterpart: Some(AgentUid(11)),
        counterpart_position: Some(Position::new(3.0, 4.0)),
        kind: scriptbots_core::ReplayEventKind::Interaction {
            tick: Tick(10),
            ordinal: 0,
            kind: scriptbots_core::ReplayInteractionKind::Combat,
            magnitude: 5.0,
        },
    });
    storage.persist(&b10).expect("persist tick 10");

    // Tick 15: 1 death
    let mut b15 = make_tick_batch(15, 0, 1);
    b15.deaths.push(death_rec(15, 20, 15, DeathCause::Aging));
    storage.persist(&b15).expect("persist tick 15");

    storage.flush().expect("flush");
    storage.close().expect("close");

    // Open read-only context
    let cx = ReaderCtx::open(&db_path).expect("open reader context");
    let registry = Registry::builtin();

    // 1. Run lineage-structure
    let lineage_out = registry
        .run("lineage-structure", &cx, &ReportParams::default())
        .expect("run lineage-structure");
    assert_eq!(lineage_out.schema_version, REPORT_SCHEMA_VERSION);
    assert_eq!(lineage_out.report, "lineage-structure");
    assert_eq!(lineage_out.row_count, 6); // 6 agents total
    assert!(lineage_out.human_md.contains("# Lineage Structure Report"));
    assert_eq!(lineage_out.machine["weakly_connected_components_count"], 3);
    assert_eq!(lineage_out.machine["is_dag"], true);

    // 2. Run dynasty-communities
    let dynasty_out = registry
        .run("dynasty-communities", &cx, &ReportParams::default())
        .expect("run dynasty-communities");
    assert_eq!(dynasty_out.schema_version, REPORT_SCHEMA_VERSION);
    assert_eq!(dynasty_out.report, "dynasty-communities");
    assert!(
        dynasty_out
            .human_md
            .contains("# Dynasty Communities Report")
    );
    assert!(dynasty_out.machine["community_count"].as_u64().unwrap() >= 1);

    // 3. Run interaction-centrality
    let interaction_out = registry
        .run("interaction-centrality", &cx, &ReportParams::default())
        .expect("run interaction-centrality");
    assert_eq!(interaction_out.schema_version, REPORT_SCHEMA_VERSION);
    assert_eq!(interaction_out.report, "interaction-centrality");
    assert!(
        interaction_out
            .human_md
            .contains("# Interaction Centrality Report")
    );

    // Drop in-process context to release database lease before spawning CLI process
    drop(cx);

    // 4. Test sb-analyze CLI execution via Command
    let bin_path = env!("CARGO_BIN_EXE_sb-analyze");
    let output = Command::new(bin_path)
        .arg(&db_path)
        .arg("run")
        .arg("lineage-structure")
        .output()
        .expect("run sb-analyze CLI");
    assert!(
        output.status.success(),
        "sb-analyze run lineage-structure failed: {output:?}"
    );

    let export_output = Command::new(bin_path)
        .arg(&db_path)
        .arg("export-graph")
        .arg("--graph")
        .arg("lineage")
        .arg("--format")
        .arg("graphml")
        .output()
        .expect("run sb-analyze export-graph");
    assert!(
        export_output.status.success(),
        "sb-analyze export-graph failed: {export_output:?}"
    );
    let stdout = String::from_utf8_lossy(&export_output.stdout);
    assert!(
        stdout.contains("<graphml"),
        "exported graphml must contain <graphml"
    );
    assert!(
        stdout.contains("agent_10"),
        "exported graphml must contain agent_10"
    );
}
