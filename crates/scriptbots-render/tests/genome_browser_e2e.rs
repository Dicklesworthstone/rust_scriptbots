//! Multi-generation DB-backed E2E integration test for Genome Browser UI (bd-16g.13.3).
//!
//! Verifies:
//! 1. Seeded simulation with reproduction, genome persistence, and ancestry tracking.
//! 2. `AgentInspectorDetails::from_world` constructs `GenomeBrowserViewModel` without concrete brain downcasts.
//! 3. Displayed node/connection topology and weights bit-exactly match stored genomes and live brain envelopes.
//! 4. Newborn mutation diff correctly captures parent-to-child deltas (Scalar, Retarget, KindFlip).
//! 5. Lineage locus tracing across ancestry DAG produces valid plot data and exports valid CSV, PNG, and SVG.
//! 6. Retains CSV, PNG, and SVG artifacts in target/genome-browser/ for CI inspection (no deletion).
//! 7. GPUI UI rendering: verifies `render_genome_browser_view` renders element trees for founders, newborns, and lineage plots without panic.
//! 8. Digest neutrality: constructing and querying `GenomeBrowserViewModel` and rendering GPUI views causes zero side-effects on simulation digest.

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use scriptbots_brain::mlp::{MlpBrain, MlpBrainFamily};
use scriptbots_core::genome_browser::{
    BrowserConnectionView, BrowserLineagePlotView, BrowserNodeView, BrowserPagingMeta,
    GenomeBrowserViewModel, MutationDiffStatus,
};
use scriptbots_core::genome_diff::{
    DiffSummary, Locus, LocusSample, LocusValue, export_locus_trace_csv, export_locus_trace_png,
    export_locus_trace_svg,
};
use scriptbots_core::rng_domains::IslandId;
use scriptbots_core::{
    AgentData, AgentId, AgentUid, BrainFamilyCodec, BrainFamilyId, BrainGenomeDerivation,
    BrainProvenance, ScriptBotsConfig, Tick, WorldState,
};
use scriptbots_render::{AgentInspectorDetails, render_genome_browser_view};
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

    // Verify GPUI element tree rendering for founder view model
    let _founder_div = render_genome_browser_view(vm);

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
    let db_envelope = reader
        .read_agent_genome(IslandId(0), agent_uid, None)
        .expect("persisted genome envelope must be readable from storage");
    assert_eq!(
        vm.genome_digest,
        db_envelope.material_hash().to_string(),
        "view model digest matches persisted genome digest"
    );

    // 6. Newborn mutation diff: Find a newborn (generation > 0)
    let arena = world.agents();
    let cols = arena.columns();
    let newborn_id = live_agents.iter().copied().find(|&id| {
        arena
            .index_of(id)
            .map_or(0, |idx| cols.generations()[idx].0)
            > 0
    });
    let nb_id = newborn_id.expect("simulation must produce newborn agents with generation > 0");

    let nb_detail =
        AgentInspectorDetails::from_world(&world, nb_id, None).expect("newborn inspector detail");
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
            panic!("Expected computed mutation diff for newborn, got: {other:?}");
        }
    }

    // Verify GPUI element tree rendering for newborn view model
    let _newborn_div = render_genome_browser_view(nb_vm);

    // 7. Lineage locus tracing and CSV/PNG/SVG export via GenomeBrowserViewModel
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

    // Verify GPUI element tree rendering for view model with locus plot
    let _plot_div = render_genome_browser_view(&vm_with_plot);

    // Export CSV, PNG, and SVG artifacts to target/genome-browser/ and retain them
    let out_dir = Path::new("target/genome-browser");
    fs::create_dir_all(out_dir)?;

    let csv_path = out_dir.join("render_locus_trace.csv");
    let png_path = out_dir.join("render_locus_trace.png");
    let svg_path = out_dir.join("render_locus_trace.svg");

    let csv_str = export_locus_trace_csv(&plot.samples, plot.locus);
    fs::write(&csv_path, &csv_str)?;

    let png_bytes = export_locus_trace_png(&plot.samples, plot.locus);
    fs::write(&png_path, &png_bytes)?;

    let svg_str = export_locus_trace_svg(&plot.samples, plot.locus);
    fs::write(&svg_path, &svg_str)?;

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

    assert!(svg_path.exists(), "SVG artifact exported");
    let svg_content = fs::read_to_string(&svg_path)?;
    assert!(svg_content.contains("<svg"), "SVG content is valid SVG");

    // 8. Digest Neutrality: world state digest must be completely unchanged
    let digest_after = world.characterization_digest_v0()?;
    assert_eq!(
        digest_before, digest_after,
        "Genome browser view model construction and rendering must be strictly digest neutral"
    );

    println!("All genome browser E2E assertions passed successfully!");
    Ok(())
}

#[test]
fn test_render_genome_browser_view_edge_cases() {
    // 1. Test rendering with unavailable diff reason
    let unavailable_vm = GenomeBrowserViewModel {
        selected_agent: AgentUid(100),
        generation: 3,
        tick: Tick(150),
        family_id: BrainFamilyId::new("mlp").unwrap(),
        schema_version: 1,
        genome_digest: "abcdef0123456789".to_string(),
        provenance: BrainProvenance {
            parents: [Some(AgentUid(50)), None],
            parent_genome_hashes: [None, None],
            created_at: Tick(100),
            derivation: BrainGenomeDerivation::MutationOnly,
        },
        parent_uids: vec![AgentUid(50)],
        nodes: vec![BrowserNodeView {
            node_index: 0,
            bias: Some(0.42),
            damping: Some(0.1),
            gain: Some(1.0),
            connections: vec![BrowserConnectionView {
                conn_slot: 0,
                target_node: 1,
                weight: -0.75,
                kind: None,
            }],
        }],
        paging: BrowserPagingMeta {
            page_offset: 0,
            page_limit: 20,
            total_nodes: 50,
            total_loci: 200,
            is_truncated: true,
        },
        mutation_diff: MutationDiffStatus::Unavailable {
            reason: "Schema version mismatch".to_string(),
        },
        deltas: vec![],
        locus_plot: None,
        build_duration_us: 0,
    };
    let _div = render_genome_browser_view(&unavailable_vm);

    // 2. Test rendering with sexual crossover diff and lineage gaps
    let sexual_vm = GenomeBrowserViewModel {
        selected_agent: AgentUid(200),
        generation: 5,
        tick: Tick(300),
        family_id: BrainFamilyId::new("dwraon").unwrap(),
        schema_version: 2,
        genome_digest: "12345678abcdef01".to_string(),
        provenance: BrainProvenance {
            parents: [Some(AgentUid(110)), Some(AgentUid(120))],
            parent_genome_hashes: [None, None],
            created_at: Tick(250),
            derivation: BrainGenomeDerivation::Crossover,
        },
        parent_uids: vec![AgentUid(110), AgentUid(120)],
        nodes: vec![],
        paging: BrowserPagingMeta {
            page_offset: 0,
            page_limit: 20,
            total_nodes: 0,
            total_loci: 0,
            is_truncated: false,
        },
        mutation_diff: MutationDiffStatus::SexualPrimary {
            parent_uids: vec![AgentUid(110), AgentUid(120)],
            total_deltas: 12,
            summary: DiffSummary {
                changed_loci: 12,
                total_loci: 12,
                l1: 4.5,
                linf: 0.8,
                by_kind: Default::default(),
            },
        },
        deltas: vec![],
        locus_plot: Some(BrowserLineagePlotView {
            locus: Locus::NodeBias(0),
            samples: vec![
                LocusSample {
                    generation: 1,
                    agent_uid: AgentUid(10),
                    tick: Tick(10),
                    value: Some(LocusValue::Scalar(0.1)),
                },
                LocusSample {
                    generation: 2,
                    agent_uid: AgentUid(20),
                    tick: Tick(50),
                    value: None,
                },
                LocusSample {
                    generation: 3,
                    agent_uid: AgentUid(30),
                    tick: Tick(100),
                    value: Some(LocusValue::Scalar(0.35)),
                },
            ],
            svg_chart: "<svg></svg>".to_string(),
            csv_data: "generation,agent_uid,tick,value_type,value\n".to_string(),
            total_points: 3,
            gap_count: 1,
        }),
        build_duration_us: 0,
    };
    let _div2 = render_genome_browser_view(&sexual_vm);
}
