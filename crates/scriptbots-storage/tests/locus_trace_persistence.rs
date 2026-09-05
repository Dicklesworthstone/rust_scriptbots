//! E2E tests for DB-backed lineage locus tracing with CSV/PNG export (bd-wdyu).
//!
//! Covers:
//! 1. `trace_locus(db, island, codec, lineage, locus) -> Vec<LocusSample>` reading persisted genomes from the run DB
//!    and walking the ancestry DAG.
//! 2. Explicit gaps (never phantom 0.0) for loci absent in an ancestor schema or unresolvable agents.
//! 3. Live-world vs DB cross-check on overlapping ticks (bit-exact genomes and locus values).
//! 4. Exporting `locus_trace.csv` and headless PNG for CI artifacts.
//! 5. E2E: seeded simulation run with reproduction, identifying deepest lineage, asserting traced
//!    values match persisted genomes.

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use scriptbots_brain::mlp::{MlpBrain, MlpBrainFamily};
use scriptbots_core::genome_diff::{
    Locus, LocusValue, export_locus_trace_csv, export_locus_trace_png,
};
use scriptbots_core::{
    AgentData, AgentId, AgentUid, BrainFamilyCodec, ScriptBotsConfig, WorldState,
};
use scriptbots_runtime::IslandId;
use scriptbots_storage::{StoragePipeline, StorageReader, rebuild_ancestry, trace_locus};

static TEST_NONCE: AtomicU64 = AtomicU64::new(1);

fn temp_db_path(label: &str) -> String {
    let nonce = TEST_NONCE.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir()
        .join(format!(
            "scriptbots_locus_trace_{label}_{}_{timestamp}_{nonce}.sqlite",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

#[test]
fn test_db_backed_lineage_locus_tracing_e2e() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = temp_db_path("e2e_lineage");
    let codec = MlpBrainFamily::new();

    let config = ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        initial_food: 0.5,
        food_max: 1.0,
        persistence_interval: 1,
        rng_seed: Some(0x10c0_ffee),
        ..ScriptBotsConfig::default()
    };

    let mut pipeline = StoragePipeline::create_unattributed_file(&db_path)?;
    let (mut world, mut persistence) =
        WorldState::with_persistence(config, Box::new(pipeline.sink()))?;
    let family_key = world
        .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
        .expect("register MLP");

    // Spawn initial founding agents and bind brains
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

    // Step the simulation for 30 ticks
    const SIMULATION_TICKS: u64 = 30;
    for _ in 0..SIMULATION_TICKS {
        persistence.step(&mut world)?;
    }

    // =========================================================================
    // 1. LIVE-VS-DB CROSS-CHECK on overlapping ticks
    // =========================================================================
    // Capture living agents and their live genome envelopes
    let live_handles: Vec<AgentId> = world.agents().iter_handles().collect();
    assert!(
        !live_handles.is_empty(),
        "world should maintain active population"
    );

    let mut live_agents = Vec::new();
    for handle in live_handles {
        let uid = world.agent_uid(handle).expect("agent exists");
        if let Some(envelope) = world.agent_brain_genome(handle) {
            live_agents.push((uid, envelope.clone()));
        }
    }
    assert!(
        !live_agents.is_empty(),
        "live agents should have brain genomes"
    );

    // Shutdown storage pipeline cleanly (forces flush to disk)
    let shutdown_receipt = pipeline.shutdown()?;
    assert!(
        shutdown_receipt.committed_tick.is_some(),
        "persistence pipeline must commit durable records"
    );

    // Reopen DB via StorageReader
    let reader = StorageReader::open(&db_path)?;

    // Cross-check: every living agent genome must be bit-identical to DB readback
    for (uid, live_envelope) in &live_agents {
        let db_envelope = reader.read_agent_genome(IslandId(0), *uid, None)?;
        assert_eq!(
            db_envelope.payload(),
            live_envelope.payload(),
            "Agent {uid:?} payload mismatch between live and persisted DB"
        );
        assert_eq!(
            db_envelope.material_hash(),
            live_envelope.material_hash(),
            "Agent {uid:?} material hash mismatch between live and persisted DB"
        );
        assert_eq!(
            db_envelope.provenance(),
            live_envelope.provenance(),
            "Agent {uid:?} provenance mismatch between live and persisted DB"
        );

        // Cross-check locus values extracted from live vs DB
        let test_locus = Locus::NodeBias(0);
        let live_loci = codec.genome_loci(live_envelope)?;
        let db_loci = codec.genome_loci(&db_envelope)?;
        let live_val = live_loci
            .iter()
            .find(|(l, _)| *l == test_locus)
            .map(|(_, v)| v);
        let db_val = db_loci
            .iter()
            .find(|(l, _)| *l == test_locus)
            .map(|(_, v)| v);
        assert_eq!(
            live_val, db_val,
            "Locus value mismatch between live world and persisted DB for agent {uid:?}"
        );
    }

    // =========================================================================
    // 2. RECONSTRUCT ANCESTRY DAG & FIND DEEPEST LINEAGE
    // =========================================================================
    let births = reader.load_ancestry_births()?;
    let deaths = reader.load_ancestry_deaths()?;
    assert!(!births.is_empty(), "Births table must contain arrival rows");

    let graph = rebuild_ancestry(&births, &deaths)?;

    // Find living agent with the deepest lineage
    let mut deepest_target = live_agents[0].0;
    let mut longest_path: Vec<AgentUid> = Vec::new();
    for (uid, _) in &live_agents {
        let path = graph.lineage_path(*uid, 100);
        if path.len() > longest_path.len() {
            longest_path = path;
            deepest_target = *uid;
        }
    }

    assert!(
        !longest_path.is_empty(),
        "Must find at least one lineage path, found {}",
        longest_path.len()
    );

    // =========================================================================
    // 3. TRACE LOCUS FROM PERSISTED DB ALONG THE DEEPEST LINEAGE
    // =========================================================================
    // Trace via StorageReader method
    let target_locus = Locus::NodeBias(0);
    let samples = reader.trace_agent_lineage_locus(&codec, deepest_target, target_locus, 100)?;

    assert_eq!(
        samples.len(),
        longest_path.len(),
        "Traced locus samples count must match lineage path length"
    );

    // Verify chronological order: generations should be non-decreasing from founder to target
    for i in 1..samples.len() {
        assert!(
            samples[i].generation >= samples[i - 1].generation,
            "Traced lineage samples must be ordered chronologically (generation {} >= {})",
            samples[i].generation,
            samples[i - 1].generation
        );
    }

    // Top-level trace_locus function parity check
    let path_chronological: Vec<AgentUid> = {
        let mut p = graph.lineage_path(deepest_target, 100);
        p.reverse();
        p
    };
    let free_samples = trace_locus(
        &reader,
        IslandId(0),
        &codec,
        &path_chronological,
        target_locus,
    )?;
    assert_eq!(
        samples, free_samples,
        "reader.trace_locus and top-level trace_locus must yield identical samples"
    );

    // Verify each traced sample matches the persisted genome envelope in DB
    for sample in &samples {
        let env = reader.read_agent_genome(IslandId(0), sample.agent_uid, None)?;
        let loci = codec.genome_loci(&env)?;
        let expected_val = loci
            .into_iter()
            .find(|(loc, _)| *loc == target_locus)
            .map(|(_, val)| val);
        assert_eq!(
            sample.value, expected_val,
            "Traced locus value at agent {:?} must equal persisted genome value",
            sample.agent_uid
        );
    }

    // =========================================================================
    // 4. EXPLICIT GAPS TEST: absent loci and nonexistent agents are never 0.0
    // =========================================================================
    // A locus absent from the MLP schema (e.g. Cell locus) must return None (explicit gap)
    let absent_locus = Locus::Cell(99);
    let absent_samples =
        reader.trace_agent_lineage_locus(&codec, deepest_target, absent_locus, 100)?;
    assert_eq!(absent_samples.len(), longest_path.len());
    for s in &absent_samples {
        assert!(
            s.value.is_none(),
            "Absent locus must evaluate to None (explicit gap), never phantom 0.0"
        );
    }

    // A nonexistent agent in a synthetic lineage must produce an explicit gap
    let ghost_lineage = vec![deepest_target, AgentUid(999_999_999)];
    let ghost_samples = reader.trace_locus(IslandId(0), &codec, &ghost_lineage, target_locus)?;
    assert_eq!(ghost_samples.len(), 2);
    assert!(ghost_samples[0].value.is_some());
    assert_eq!(
        ghost_samples[1].value, None,
        "Nonexistent agent in lineage must produce an explicit gap (None)"
    );

    // =========================================================================
    // 5. EXPORT CSV & HEADLESS PNG ARTIFACTS
    // =========================================================================
    let csv_output = export_locus_trace_csv(&samples, target_locus);
    assert!(csv_output.starts_with("# Locus Trace: node 0 bias"));
    assert!(csv_output.contains("generation,agent_uid,tick,value_type,value"));
    for s in &samples {
        let expected_row = match s.value {
            Some(LocusValue::Scalar(v)) => {
                format!("{},{},{},scalar,{v}", s.generation, s.agent_uid.0, s.tick.0)
            }
            _ => format!("{},{},{},gap,GAP", s.generation, s.agent_uid.0, s.tick.0),
        };
        assert!(
            csv_output.contains(&expected_row),
            "CSV must contain row: {expected_row}"
        );
    }

    // Export headless PNG and verify format
    let png_output = export_locus_trace_png(&samples, target_locus);
    assert!(
        png_output.len() > 100,
        "PNG output must be non-empty and well-sized"
    );
    assert_eq!(
        &png_output[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
        "PNG output must start with valid 8-byte PNG header"
    );

    // Write artifacts to target/locus-trace/ for CI inspection
    let artifact_dir = Path::new("target/locus-trace");
    fs::create_dir_all(artifact_dir)?;
    let csv_file = artifact_dir.join("locus_trace.csv");
    let png_file = artifact_dir.join("locus_trace.png");
    fs::write(&csv_file, &csv_output)?;
    fs::write(&png_file, &png_output)?;

    assert!(csv_file.exists(), "CSV artifact file must exist");
    assert!(png_file.exists(), "PNG artifact file must exist");

    println!(
        "locus-trace: PASS; artifacts written to {}",
        artifact_dir.display()
    );

    Ok(())
}
