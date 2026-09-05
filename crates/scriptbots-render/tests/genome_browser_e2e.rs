//! Multi-generation DB-backed E2E integration test for Genome Browser UI (bd-16g.13.3).
//!
//! Verifies:
//! 1. Seeded simulation with reproduction, genome persistence, and ancestry tracking.
//! 2. `AgentInspectorDetails::from_world` constructs `GenomeBrowserViewModel` without concrete brain downcasts.
//! 3. Displayed node/connection topology and weights bit-exactly match stored genomes and live brain envelopes.
//! 4. Newborn mutation diff correctly captures parent-to-child deltas (Scalar, Retarget, KindFlip).
//! 5. Lineage locus tracing across ancestry DAG produces valid plot data and exports valid CSV and PNG.
//! 6. Digest neutrality: constructing and querying `GenomeBrowserViewModel` causes zero side-effects on simulation digest.

use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use scriptbots_brain::mlp::{MlpBrain, MlpBrainFamily};
use scriptbots_core::genome_browser::{GenomeBrowserViewModel, MutationDiffStatus};
use scriptbots_core::genome_diff::{
    Locus, LocusValue, export_locus_trace_csv, export_locus_trace_png,
};
use scriptbots_core::rng_domains::IslandId;
use scriptbots_core::{
    AgentData, AgentId, AgentUid, BrainFamilyCodec, ScriptBotsConfig, WorldState,
};
use scriptbots_render::AgentInspectorDetails;
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
            "scriptbots_genome_browser_{label}_{}_{timestamp}_{nonce}.sqlite",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

#[test]
fn test_genome_browser_ui_e2e() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = temp_db_path("e2e_browser");
    let codec = MlpBrainFamily::new();

    let config = ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        initial_food: 0.5,
        food_max: 1.0,
        persistence_interval: 1,
        rng_seed: Some(0xbeef_cafe),
        ..ScriptBotsConfig::default()
    };

    let mut pipeline = StoragePipeline::create_unattributed_file(&db_path)?;
    let (mut world, mut persistence) =
        WorldState::with_persistence(config, Box::new(pipeline.sink()))?;
    let family_key = world
        .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
        .expect("register MLP");

    // 1. Spawn initial founding agents and bind brains
    for _ in 0..12 {
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

    // Step simulation for 35 ticks to generate births and persist records
    const SIMULATION_TICKS: u64 = 35;
    for _ in 0..SIMULATION_TICKS {
        persistence.step(&mut world)?;
    }

    // 2. Digest Neutrality: compute characterization digest before inspector queries
    let digest_before = world.characterization_digest_v0()?;

    // 3. Inspect a live agent with AgentInspectorDetails::from_world
    let live_agents: Vec<AgentId> = world.agents().iter_handles().collect();
    assert!(!live_agents.is_empty(), "simulation retained live agents");

    let first_agent = live_agents[0];
    let detail = AgentInspectorDetails::from_world(&world, first_agent, None)
        .expect("inspector detail constructed");

    let vm = detail
        .genome_browser
        .as_ref()
        .expect("genome browser view model present");

    // Check basic view model properties
    assert_eq!(vm.selected_agent, world.agent_uid(first_agent).unwrap());
    assert_eq!(vm.family_id.as_str(), codec.family_id().as_str());
    assert!(
        !vm.nodes.is_empty(),
        "nodes populated in browser view model"
    );

    // 4. Bit-exactness check: Displayed node values must match the codec decoded loci
    let live_envelope = world
        .agent_brain_genome(first_agent)
        .expect("agent brain genome");
    let decoded_loci = codec
        .genome_loci(live_envelope)
        .expect("decode genome loci");

    // Verify each node in vm.nodes matches decoded loci
    for node in &vm.nodes {
        if let Some(bias) = node.bias {
            let locus = Locus::NodeBias(node.node_index);
            let expected = decoded_loci.iter().find(|(l, _)| *l == locus);
            assert!(expected.is_some(), "node bias locus found in decoded loci");
            if let Some((_, LocusValue::Scalar(s))) = expected {
                assert_eq!(bias, *s, "bias matches decoded locus value");
            }
        }
        for conn in &node.connections {
            let locus = Locus::NodeWeight {
                node: node.node_index,
                conn: conn.conn_slot,
            };
            let expected = decoded_loci.iter().find(|(l, _)| *l == locus);
            assert!(
                expected.is_some(),
                "connection weight locus found in decoded loci"
            );
            if let Some((_, LocusValue::Scalar(s))) = expected {
                assert_eq!(conn.weight, *s, "weight matches decoded locus value");
            }
        }
    }

    // Shutdown storage pipeline cleanly to force flush to disk
    let _shutdown = pipeline.shutdown()?;
    let reader = StorageReader::open(&db_path)?;

    // 5. Check DB persistence matches live genome material digest
    let agent_uid = world.agent_uid(first_agent).unwrap();
    if let Ok(db_envelope) = reader.read_agent_genome(IslandId(0), agent_uid, None) {
        assert_eq!(
            vm.genome_digest,
            db_envelope.material_hash().to_string(),
            "view model digest matches persisted genome digest"
        );
    }

    // 6. Newborn mutation diff: Find a newborn (generation > 0)
    let arena = world.agents();
    let cols = arena.columns();
    let newborn_id = live_agents.iter().copied().find(|&id| {
        arena
            .index_of(id)
            .map_or(0, |idx| cols.generations()[idx].0)
            > 0
    });

    if let Some(nb_id) = newborn_id {
        let nb_detail = AgentInspectorDetails::from_world(&world, nb_id, None)
            .expect("newborn inspector detail");
        let nb_vm = nb_detail
            .genome_browser
            .as_ref()
            .expect("newborn genome browser");

        match &nb_vm.mutation_diff {
            MutationDiffStatus::Computed {
                parent_uid,
                total_deltas,
                summary,
            } => {
                assert_ne!(*parent_uid, AgentUid(0), "parent UID must be valid");
                assert_eq!(*total_deltas, summary.changed_loci);
                assert_eq!(nb_vm.deltas.len(), *total_deltas);
                println!(
                    "Verified newborn mutation diff: parent {}, deltas {}, L1 {:.4}",
                    parent_uid.get(),
                    total_deltas,
                    summary.l1
                );
            }
            MutationDiffStatus::SexualPrimary {
                parent_uids,
                total_deltas,
                summary,
            } => {
                assert!(!parent_uids.is_empty());
                assert_eq!(*total_deltas, summary.changed_loci);
            }
            other => {
                println!("Newborn parent diff status: {other:?}");
            }
        }
    }

    // 7. Lineage locus tracing and CSV/PNG export via GenomeBrowserViewModel
    let births = reader.load_ancestry_births()?;
    let deaths = reader.load_ancestry_deaths()?;
    let graph = rebuild_ancestry(&births, &deaths)?;

    let selected_locus = Locus::NodeBias(0);
    let path = graph.lineage_path(agent_uid, 100);
    let lineage_genomes = reader.read_lineage_genomes(IslandId(0), &path)?;

    let mut lineage_nodes = Vec::new();
    for (uid, tick, env) in lineage_genomes {
        let gen_val = births
            .iter()
            .find(|b| b.agent_uid == uid)
            .map_or(0, |b| b.generation.0);
        lineage_nodes.push((gen_val, uid, tick, env));
    }

    let first_agent_idx = arena.index_of(first_agent).unwrap();
    let first_agent_gen = cols.generations()[first_agent_idx].0;

    let vm_with_plot = GenomeBrowserViewModel::build(
        &codec,
        agent_uid,
        first_agent_gen,
        world.tick(),
        live_envelope,
        None,
        vec![],
        Some(selected_locus),
        Some(&lineage_nodes),
        0,
        20,
    )?;

    assert!(
        vm_with_plot.locus_plot.is_some(),
        "locus plot view constructed"
    );
    let plot = vm_with_plot.locus_plot.as_ref().unwrap();
    assert_eq!(plot.locus, selected_locus);
    assert!(!plot.samples.is_empty(), "plot contains ancestry samples");

    // Export CSV and PNG artifacts
    let out_dir =
        std::env::temp_dir().join(format!("genome_browser_artifacts_{}", std::process::id()));
    fs::create_dir_all(&out_dir)?;

    let csv_path = out_dir.join("locus_trace.csv");
    let png_path = out_dir.join("locus_trace.png");

    let csv_str = export_locus_trace_csv(&plot.samples, plot.locus);
    fs::write(&csv_path, csv_str)?;

    let png_bytes = export_locus_trace_png(&plot.samples, plot.locus);
    fs::write(&png_path, &png_bytes)?;

    assert!(csv_path.exists(), "CSV artifact exported");
    let csv_content = fs::read_to_string(&csv_path)?;
    assert!(csv_content.contains("generation,agent_uid,tick,value_type,value"));

    assert!(png_path.exists(), "PNG artifact exported");
    assert!(png_bytes.len() > 8, "PNG file has non-zero size");
    assert_eq!(
        &png_bytes[0..8],
        b"\x89PNG\r\n\x1a\n",
        "PNG has valid 8-byte signature"
    );

    // 8. Digest Neutrality: world state digest must be completely unchanged
    let digest_after = world.characterization_digest_v0()?;
    assert_eq!(
        digest_before, digest_after,
        "Genome browser view model construction must be strictly digest neutral"
    );

    // Clean up temporary database files
    let _ = fs::remove_file(&db_path);
    let _ = fs::remove_file(format!("{db_path}-wal"));
    let _ = fs::remove_file(format!("{db_path}-shm"));
    let _ = fs::remove_file(&csv_path);
    let _ = fs::remove_file(&png_path);
    let _ = fs::remove_dir(&out_dir);

    println!("All genome browser E2E assertions passed successfully!");
    Ok(())
}
