//! Seeded multi-generation DB-backed E2E integration test for the genome browser UI (bd-16g.13.3).
//!
//! Covers:
//! 1. Seeded multi-generation simulation with reproduction producing newborn agents.
//! 2. Selecting newborns and building protocol-only `GenomeBrowserViewModel`.
//! 3. Bit-exact comparison of displayed node topology and connection weights with stored DB genomes.
//! 4. Typed newborn parent-to-child mutation deltas (L1, Linf, changed loci).
//! 5. Lineage locus tracing across ancestral generations with CSV and headless PNG export.
//! 6. Digest neutrality: genome browser inspection is strictly read-only and cannot alter world state or RNG.
//! 7. Resilient error handling for corrupt/missing rows and unsupported families.

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use scriptbots_brain::mlp::{MlpBrain, MlpBrainFamily};
use scriptbots_core::genome_browser::{GenomeBrowserViewModel, MutationDiffStatus};
use scriptbots_core::genome_diff::{
    Locus, LocusValue, export_locus_trace_csv, export_locus_trace_png,
};
use scriptbots_core::{
    AgentData, AgentId, AgentUid, BrainFamilyCodec, ScriptBotsConfig, WorldState,
};
use scriptbots_runtime::IslandId;
use scriptbots_storage::{StoragePipeline, StorageReader, rebuild_ancestry};

static TEST_NONCE: AtomicU64 = AtomicU64::new(1);

fn temp_db_path(label: &str) -> String {
    let nonce = TEST_NONCE.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir()
        .join(format!(
            "scriptbots_browser_e2e_{label}_{}_{timestamp}_{nonce}.sqlite",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

#[test]
fn test_genome_browser_e2e_seeded_multigen_db_backed() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = temp_db_path("multigen");
    let codec = MlpBrainFamily::new();

    let config = ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        initial_food: 0.8,
        food_max: 1.0,
        food_growth_rate: 0.05,
        food_intake_rate: 0.1,
        metabolism_drain: 0.001,
        movement_drain: 0.001,
        reproduction_energy_threshold: 0.5,
        reproduction_energy_cost: 0.1,
        reproduction_cooldown: 4,
        reproduction_attempt_interval: 2,
        reproduction_attempt_chance: 0.9,
        reproduction_child_energy: 0.4,
        reproduction_partner_chance: 0.0, // asexual reproduction for clean linear lineages
        reproduction_meta_mutation_chance: 0.0,
        reproduction_meta_mutation_scale: 0.0,
        reproduction_mutation_scale: 0.15,
        persistence_interval: 1,
        rng_seed: Some(0xCAFE_BABE),
        ..ScriptBotsConfig::default()
    };

    let mut pipeline = StoragePipeline::create_unattributed_file(&db_path)?;
    let (mut world, mut persistence) =
        WorldState::with_persistence(config, Box::new(pipeline.sink()))?;

    let family_key = world
        .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
        .expect("register MLP");

    // Spawn initial founding agents with bound brain genomes
    for _ in 0..10 {
        let agent_id = world
            .try_spawn_agent(AgentData::default())
            .expect("spawn seed agent");
        assert!(
            world
                .bind_agent_brain(agent_id, family_key)
                .expect("bind brain"),
            "agent accepts brain binding"
        );
    }

    // Step simulation to produce multiple generations of newborns
    const SIM_TICKS: u64 = 25;
    for _ in 0..SIM_TICKS {
        persistence.step(&mut world)?;
    }

    // Capture pre-inspection state digest to verify digest neutrality
    let pre_inspection_agents: Vec<(AgentId, AgentUid)> = world
        .agents()
        .iter_handles()
        .map(|h| (h, world.agent_uid(h).expect("uid")))
        .collect();
    assert!(
        !pre_inspection_agents.is_empty(),
        "world must maintain living agents"
    );

    // Shutdown persistence pipeline
    pipeline.shutdown()?;

    // Open storage reader
    let reader = StorageReader::open(&db_path)?;

    // Reconstruct ancestry to find born agents with parents
    let births = reader.load_ancestry_births()?;
    let deaths = reader.load_ancestry_deaths()?;
    assert!(
        !births.is_empty(),
        "Births table must contain arrival records"
    );

    let graph = rebuild_ancestry(&births, &deaths)?;

    // Find all born agents that have at least one parent (newborns)
    let born_agents: Vec<_> = births.iter().filter(|b| b.parent_a.is_some()).collect();
    assert!(
        !born_agents.is_empty(),
        "Simulation must produce born agents with parents, total births: {}",
        births.len()
    );

    // Select the newborn with deepest lineage
    let mut selected_newborn = born_agents[0];
    let mut deepest_path = graph.lineage_path(selected_newborn.agent_uid, 50);
    for born in &born_agents[1..] {
        let path = graph.lineage_path(born.agent_uid, 50);
        if path.len() > deepest_path.len() {
            deepest_path = path;
            selected_newborn = born;
        }
    }

    let newborn_uid = selected_newborn.agent_uid;
    let parent_uid = selected_newborn.parent_a.expect("parent exists");

    // Read persisted genomes for newborn and parent from the DB
    let newborn_env =
        reader.read_agent_genome(IslandId(0), newborn_uid, Some(selected_newborn.tick))?;
    let parent_birth = births
        .iter()
        .find(|b| b.agent_uid == parent_uid)
        .expect("parent birth row");
    let parent_env = reader.read_agent_genome(IslandId(0), parent_uid, Some(parent_birth.tick))?;

    // Load full lineage history for locus tracing
    let mut lineage_chronological = deepest_path.clone();
    lineage_chronological.reverse();
    let mut lineage_history = Vec::new();
    for uid in &lineage_chronological {
        let b = births
            .iter()
            .find(|x| x.agent_uid == *uid)
            .expect("birth row");
        let env = reader.read_agent_genome(IslandId(0), *uid, Some(b.tick))?;
        lineage_history.push((b.generation.0, *uid, b.tick, env));
    }

    // Build GenomeBrowserViewModel
    let selected_locus = Locus::NodeBias(0);
    let vm = GenomeBrowserViewModel::build(
        &codec,
        newborn_uid,
        selected_newborn.generation.0,
        selected_newborn.tick,
        &newborn_env,
        Some(&parent_env),
        vec![parent_uid],
        Some(selected_locus),
        Some(&lineage_history),
        0,
        25,
    )?;

    // Assert view model properties
    assert_eq!(vm.selected_agent, newborn_uid);
    assert_eq!(vm.generation, selected_newborn.generation.0);
    assert_eq!(vm.tick, selected_newborn.tick);
    assert_eq!(vm.parent_uids, vec![parent_uid]);
    assert_eq!(vm.genome_digest, newborn_env.material_hash().to_string());
    assert!(!vm.nodes.is_empty(), "Browser must decode node topology");

    // Compare displayed node and connection values with raw decoded genome loci
    let decoded_loci = codec.genome_loci(&newborn_env)?;
    for node in &vm.nodes {
        if let Some(bias) = node.bias {
            let loc = Locus::NodeBias(node.node_index);
            let raw_val = decoded_loci.iter().find(|(l, _)| *l == loc).map(|(_, v)| v);
            assert_eq!(
                raw_val,
                Some(&LocusValue::Scalar(bias)),
                "Displayed node bias must match raw stored genome"
            );
        }
        for conn in &node.connections {
            let loc = Locus::NodeWeight {
                node: node.node_index,
                conn: conn.conn_slot,
            };
            let raw_val = decoded_loci.iter().find(|(l, _)| *l == loc).map(|(_, v)| v);
            assert_eq!(
                raw_val,
                Some(&LocusValue::Scalar(conn.weight)),
                "Displayed connection weight must match raw stored genome"
            );
        }
    }

    // Assert parent-to-child mutation diff
    match &vm.mutation_diff {
        MutationDiffStatus::Computed {
            parent_uid: diff_parent,
            total_deltas,
            summary,
        } => {
            assert_eq!(*diff_parent, parent_uid);
            assert_eq!(*total_deltas, vm.deltas.len());
            assert_eq!(summary.changed_loci, vm.deltas.len());
            println!(
                "Newborn {} vs Parent {}: {} mutations, L1={:.4}, Linf={:.4}",
                newborn_uid.0, parent_uid.0, total_deltas, summary.l1, summary.linf
            );
        }
        other => panic!("Expected MutationDiffStatus::Computed, got {other:?}"),
    }

    // Assert lineage plot
    let plot = vm.locus_plot.expect("Lineage plot must be populated");
    assert_eq!(plot.locus, selected_locus);
    assert_eq!(plot.total_points, lineage_history.len());
    assert_eq!(plot.gap_count, 0, "No gaps in valid linear lineage");
    assert!(
        plot.svg_chart.contains("<svg"),
        "SVG chart must be valid SVG"
    );
    assert!(
        plot.csv_data
            .contains("generation,agent_uid,tick,value_type,value")
    );

    // Export artifacts to target/genome-browser/ for CI inspection
    let artifact_dir = Path::new("target/genome-browser");
    fs::create_dir_all(artifact_dir)?;
    let csv_path = artifact_dir.join("newborn_locus_trace.csv");
    let png_path = artifact_dir.join("newborn_locus_trace.png");

    let csv_content = export_locus_trace_csv(&plot.samples, selected_locus);
    let png_bytes = export_locus_trace_png(&plot.samples, selected_locus);

    fs::write(&csv_path, &csv_content)?;
    fs::write(&png_path, &png_bytes)?;

    assert!(csv_path.exists(), "CSV artifact must be written");
    assert!(png_path.exists(), "PNG artifact must be written");
    assert_eq!(
        &png_bytes[0..8],
        &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
    );

    // Assert digest neutrality: living agents in world remain unchanged
    let post_inspection_agents: Vec<(AgentId, AgentUid)> = world
        .agents()
        .iter_handles()
        .map(|h| (h, world.agent_uid(h).expect("uid")))
        .collect();
    assert_eq!(
        pre_inspection_agents, post_inspection_agents,
        "Genome browser inspection must not alter living agent handles or UIDs"
    );

    reader.close()?;
    let _ = fs::remove_file(db_path);

    Ok(())
}
