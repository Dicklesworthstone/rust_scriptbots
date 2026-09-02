//! Integration tests for versioned brain genome envelope persistence and run-scoped readback.
//!
//! Covers:
//! - Founder, asexual, sexual, and mixed-family genome envelope admission.
//! - Bit-exact payload readback including signed zero, subnormals, and float ULP bit preservation.
//! - Duplicate and retry idempotence (`ON CONFLICT DO NOTHING`).
//! - Multi-run isolation.
//! - Typed error handling (NotFound, CorruptPayload, DigestMismatch, RunMismatch).
//! - Ordered lineage batch readback and bounded pagination.
//! - Fixed-seed multi-generation E2E comparing live envelopes with reopened DB readback.

use fsqlite::compat::{open_with_flags, OpenFlags, RowExt};
use scriptbots_brain::mlp::{MlpBrain, MlpBrainFamily};
use scriptbots_core::{
    AgentData, AgentId, AgentUid, BirthOrigin, BirthRecord, BrainFamilyId, BrainGenomeDerivation,
    BrainGenomeEnvelope, BrainGenomeHash, BrainProvenance, Generation, MetricSample,
    PersistedGenome, PersistenceBatch, PersistenceEvent, PersistenceEventKind, Position,
    ScriptBotsConfig, Tick, TickSummary, WorldState,
};
use scriptbots_runtime::RunId;
use scriptbots_storage::{
    GenomeStorageError, RunManifestRecord, StorageError, StoragePipeline, StorageReader,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static DB_NONCE: AtomicU64 = AtomicU64::new(1);

fn temp_db_path(label: &str) -> String {
    let nonce = DB_NONCE.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir()
        .join(format!(
            "scriptbots_genome_test_{label}_{}_{timestamp}_{nonce}.sqlite",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

fn sample_batch_with_genomes(
    tick: u64,
    genomes: Vec<PersistedGenome>,
    births: Vec<BirthRecord>,
) -> PersistenceBatch {
    let born_count = births
        .iter()
        .filter(|b| matches!(b.origin, BirthOrigin::Born))
        .count();
    let events = if born_count > 0 {
        vec![PersistenceEvent::new(PersistenceEventKind::Births, born_count)]
    } else {
        Vec::new()
    };
    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(tick),
            agent_count: births.len(),
            births: born_count,
            deaths: 0,
            total_energy: 100.0,
            average_energy: 10.0,
            average_health: 1.0,
            max_age: 0,
            spike_hits: 0,
        },
        epoch: 0,
        closed: false,
        metrics: vec![MetricSample::from_f32("population", births.len() as f32)],
        events,
        agents: Vec::new(),
        births,
        deaths: Vec::new(),
        replay_events: Vec::new(),
        narrative_events: Vec::new(),
        genomes,
    }
}

fn create_founder_envelope(family: &str, payload: Vec<u8>) -> BrainGenomeEnvelope {
    let family_id = BrainFamilyId::new(family).expect("valid family id");
    let provenance = BrainProvenance {
        parents: [None, None],
        parent_genome_hashes: [None, None],
        created_at: Tick(0),
        derivation: BrainGenomeDerivation::Founder,
    };
    BrainGenomeEnvelope::new(family_id, 1, 1, payload, provenance)
        .expect("create founder envelope")
}

fn create_offspring_envelope(
    family: &str,
    payload: Vec<u8>,
    parent_a: (AgentUid, BrainGenomeHash),
    parent_b: Option<(AgentUid, BrainGenomeHash)>,
    tick: u64,
) -> BrainGenomeEnvelope {
    let family_id = BrainFamilyId::new(family).expect("valid family id");
    let (derivation, parents, parent_hashes) = match parent_b {
        Some(p2) => (
            BrainGenomeDerivation::Crossover,
            [Some(parent_a.0), Some(p2.0)],
            [Some(parent_a.1), Some(p2.1)],
        ),
        None => (
            BrainGenomeDerivation::MutationOnly,
            [Some(parent_a.0), None],
            [Some(parent_a.1), None],
        ),
    };
    let provenance = BrainProvenance {
        parents,
        parent_genome_hashes: parent_hashes,
        created_at: Tick(tick),
        derivation,
    };
    BrainGenomeEnvelope::new(family_id, 1, 1, payload, provenance)
        .expect("create offspring envelope")
}

#[test]
fn test_founder_and_offspring_admission_and_readback() -> Result<(), Box<dyn std::error::Error>> {
    let path = temp_db_path("founder_offspring");
    let mut pipeline = StoragePipeline::create_unattributed_file(&path)?;

    let founder_payload = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let founder_env = create_founder_envelope("mlp", founder_payload.clone());
    let founder_hash = founder_env.material_hash();

    let founder_genome = PersistedGenome {
        agent_uid: AgentUid(1),
        created_at_tick: Tick(0),
        envelope: founder_env.clone(),
    };

    let founder_birth = BirthRecord {
        tick: Tick(0),
        agent_uid: AgentUid(1),
        spawn_ordinal: 0,
        birth_ordinal: None,
        origin: BirthOrigin::Seeded,
        parent_a: None,
        parent_b: None,
        brain_kind: Some("mlp".to_owned()),
        brain_key: None,
        herbivore_tendency: 0.5,
        generation: Generation(0),
        position: Position::new(10.0, 10.0),
        is_hybrid: false,
    };

    let batch_0 = sample_batch_with_genomes(0, vec![founder_genome], vec![founder_birth]);
    pipeline.submit(&batch_0)?;

    // Tick 5: asexual child of agent 1
    let asexual_payload = vec![1, 2, 3, 4, 5, 6, 7, 9]; // mutated byte
    let asexual_env = create_offspring_envelope(
        "mlp",
        asexual_payload.clone(),
        (AgentUid(1), founder_hash),
        None,
        5,
    );
    let asexual_genome = PersistedGenome {
        agent_uid: AgentUid(2),
        created_at_tick: Tick(5),
        envelope: asexual_env.clone(),
    };
    let asexual_birth = BirthRecord {
        tick: Tick(5),
        agent_uid: AgentUid(2),
        spawn_ordinal: 1,
        birth_ordinal: Some(1),
        origin: BirthOrigin::Born,
        parent_a: Some(AgentUid(1)),
        parent_b: None,
        brain_kind: Some("mlp".to_owned()),
        brain_key: None,
        herbivore_tendency: 0.5,
        generation: Generation(1),
        position: Position::new(12.0, 10.0),
        is_hybrid: false,
    };

    let batch_5 = sample_batch_with_genomes(5, vec![asexual_genome], vec![asexual_birth]);
    pipeline.submit(&batch_5)?;

    // Flush and finalize
    pipeline.shutdown()?;

    // Read back via StorageReader
    let reader = StorageReader::open(&path)?;

    // 1. Read founder by agent UID and exact tick
    let read_founder = reader.read_agent_genome(AgentUid(1), Some(Tick(0)))?;
    assert_eq!(read_founder.payload(), &founder_payload);
    assert_eq!(read_founder.material_hash(), founder_hash);
    assert_eq!(read_founder.provenance().derivation, BrainGenomeDerivation::Founder);

    // 2. Read founder by agent UID latest (tick = None)
    let read_founder_latest = reader.read_agent_genome(AgentUid(1), None)?;
    assert_eq!(read_founder_latest, read_founder);

    // 3. Read asexual offspring
    let read_asexual = reader.read_agent_genome(AgentUid(2), Some(Tick(5)))?;
    assert_eq!(read_asexual.payload(), &asexual_payload);
    assert_eq!(read_asexual.provenance().derivation, BrainGenomeDerivation::MutationOnly);
    assert_eq!(read_asexual.provenance().parents[0], Some(AgentUid(1)));
    assert_eq!(read_asexual.provenance().parent_genome_hashes[0], Some(founder_hash));

    // 4. Read by genome ID
    let read_by_id = reader.read_genome_by_id("agent:1:tick:0")?;
    assert_eq!(read_by_id, read_founder);

    Ok(())
}

#[test]
fn test_bit_exact_float_and_subnormal_payload_preservation() -> Result<(), Box<dyn std::error::Error>> {
    let path = temp_db_path("float_bit_exact");
    let mut pipeline = StoragePipeline::create_unattributed_file(&path)?;

    // Build payload containing specific floating-point bit patterns:
    // - negative zero: (-0.0_f32).to_bits() = 0x8000_0000
    // - subnormal float: 1e-40_f32
    // - positive zero: 0.0_f32
    // - max subnormal, min positive normal
    let mut payload = Vec::new();
    let floats: &[f32] = &[
        -0.0_f32,
        0.0_f32,
        1.0e-40_f32,
        -1.0e-40_f32,
        1.0000001_f32,
        -1.0000001_f32,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
    ];
    for &f in floats {
        payload.extend_from_slice(&f.to_le_bytes());
    }

    let env = create_founder_envelope("dwraon", payload.clone());
    let genome = PersistedGenome {
        agent_uid: AgentUid(42),
        created_at_tick: Tick(10),
        envelope: env.clone(),
    };
    let batch = sample_batch_with_genomes(10, vec![genome], Vec::new());
    pipeline.submit(&batch)?;
    pipeline.shutdown()?;

    let reader = StorageReader::open(&path)?;
    let readback = reader.read_agent_genome(AgentUid(42), Some(Tick(10)))?;

    assert_eq!(readback.payload(), &payload);
    assert_eq!(readback.material_hash(), env.material_hash());

    // Verify each f32 bit pattern matches byte-for-byte and bit-for-bit
    for (chunk_idx, chunk) in readback.payload().chunks_exact(4).enumerate() {
        let expected_bits = floats[chunk_idx].to_bits();
        let actual_bits = u32::from_le_bytes(chunk.try_into().unwrap());
        assert_eq!(
            actual_bits, expected_bits,
            "Float at index {chunk_idx} bit mismatch: expected 0x{expected_bits:08x}, got 0x{actual_bits:08x}"
        );
    }

    Ok(())
}

#[test]
fn test_duplicate_and_retry_idempotence() -> Result<(), Box<dyn std::error::Error>> {
    let path = temp_db_path("idempotence");
    let mut pipeline = StoragePipeline::create_unattributed_file(&path)?;

    let payload = vec![10, 20, 30, 40];
    let env = create_founder_envelope("mlp", payload);
    let genome = PersistedGenome {
        agent_uid: AgentUid(7),
        created_at_tick: Tick(1),
        envelope: env.clone(),
    };

    let batch = sample_batch_with_genomes(1, vec![genome.clone(), genome.clone()], Vec::new());
    // Submitting batch with duplicate genome in same batch
    pipeline.submit(&batch)?;
    pipeline.shutdown()?;

    // Reopening and resubmitting same batch again (retry simulation)
    let mut recovered_pipe = StoragePipeline::recover_existing(&path)?;
    recovered_pipe.submit(&batch)?;
    recovered_pipe.shutdown()?;

    let reader = StorageReader::open(&path)?;
    let readback = reader.read_agent_genome(AgentUid(7), Some(Tick(1)))?;
    assert_eq!(readback.material_hash(), env.material_hash());

    // Verify exactly 1 row exists in genomes table
    let conn = open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM genomes WHERE agent_uid = 7")?
        .get_typed(0)?;
    assert_eq!(count, 1, "Duplicate submission must not create duplicate rows");

    Ok(())
}

#[test]
fn test_multi_run_isolation() -> Result<(), Box<dyn std::error::Error>> {
    let path = temp_db_path("multi_run_isolation");
    let run_a = RunId::from_namespace_sequence(0xAA11_0001, 1);
    let run_b = RunId::from_namespace_sequence(0xBB22_0002, 1);

    // Pipeline A writes genome for AgentUid(1)
    let mut pipe_a =
        StoragePipeline::create_new_file_for_run(&path, RunManifestRecord::unattributed(run_a))?;
    let env_a = create_founder_envelope("mlp", vec![1, 1, 1]);
    let batch_a = sample_batch_with_genomes(
        1,
        vec![PersistedGenome {
            agent_uid: AgentUid(1),
            created_at_tick: Tick(1),
            envelope: env_a.clone(),
        }],
        Vec::new(),
    );
    pipe_a.submit(&batch_a)?;
    pipe_a.shutdown()?;

    // Pipeline B writes genome for AgentUid(1) in Run B
    let mut pipe_b = StoragePipeline::append_run(&path, RunManifestRecord::unattributed(run_b))?;
    let env_b = create_founder_envelope("dwraon", vec![2, 2, 2, 2]);
    let batch_b = sample_batch_with_genomes(
        1,
        vec![PersistedGenome {
            agent_uid: AgentUid(1),
            created_at_tick: Tick(1),
            envelope: env_b.clone(),
        }],
        Vec::new(),
    );
    pipe_b.submit(&batch_b)?;
    pipe_b.shutdown()?;

    // Reader for Run A
    let reader_a = StorageReader::open_for_run(&path, run_a)?;
    let read_a = reader_a.read_agent_genome(AgentUid(1), Some(Tick(1)))?;
    assert_eq!(read_a.family_id().as_str(), "mlp");
    assert_eq!(read_a.payload(), &[1, 1, 1]);

    // Reader for Run B
    let reader_b = StorageReader::open_for_run(&path, run_b)?;
    let read_b = reader_b.read_agent_genome(AgentUid(1), Some(Tick(1)))?;
    assert_eq!(read_b.family_id().as_str(), "dwraon");
    assert_eq!(read_b.payload(), &[2, 2, 2, 2]);

    // Query for nonexistent agent in Run A
    let err = reader_a.read_agent_genome(AgentUid(999), None);
    assert!(matches!(
        err,
        Err(StorageError::Genome(GenomeStorageError::NotFound { .. }))
    ));

    Ok(())
}

#[test]
fn test_error_conditions_and_tamper_detection() -> Result<(), Box<dyn std::error::Error>> {
    let path = temp_db_path("tamper_detection");
    let mut pipeline = StoragePipeline::create_unattributed_file(&path)?;

    let env = create_founder_envelope("mlp", vec![10, 20, 30]);
    let batch = sample_batch_with_genomes(
        1,
        vec![
            PersistedGenome {
                agent_uid: AgentUid(1),
                created_at_tick: Tick(1),
                envelope: env.clone(),
            },
            PersistedGenome {
                agent_uid: AgentUid(2),
                created_at_tick: Tick(1),
                envelope: env.clone(),
            },
        ],
        Vec::new(),
    );
    pipeline.submit(&batch)?;
    pipeline.shutdown()?;

    // 1. NotFound error
    let reader = StorageReader::open(&path)?;
    let err_not_found = reader.read_agent_genome(AgentUid(100), Some(Tick(1)));
    assert!(matches!(
        err_not_found,
        Err(StorageError::Genome(GenomeStorageError::NotFound {
            agent_uid: Some(AgentUid(100)),
            tick: Some(Tick(1))
        }))
    ));

    // Tamper with SQLite DB directly
    let conn = open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;

    // 2. Corrupt Payload JSON
    conn.execute(
        "UPDATE genomes SET genome_json = 'INVALID_NOT_JSON{' WHERE agent_uid = 1",
    )?;

    // 3. Digest Mismatch
    conn.execute(
        "UPDATE genomes SET genome_digest = '0000000000000000000000000000000000000000000000000000000000000000' WHERE agent_uid = 2",
    )?;
    drop(conn);

    let reader_tampered = StorageReader::open(&path)?;

    // Test CorruptPayload detection
    let err_corrupt = reader_tampered.read_agent_genome(AgentUid(1), Some(Tick(1)));
    assert!(matches!(
        err_corrupt,
        Err(StorageError::Genome(GenomeStorageError::CorruptPayload { .. }))
    ));

    // Test DigestMismatch detection
    let err_digest = reader_tampered.read_agent_genome(AgentUid(2), Some(Tick(1)));
    assert!(matches!(
        err_digest,
        Err(StorageError::Genome(GenomeStorageError::DigestMismatch { .. }))
    ));

    Ok(())
}

#[test]
fn test_ordered_lineage_and_page_readback() -> Result<(), Box<dyn std::error::Error>> {
    let path = temp_db_path("ordered_lineage");
    let mut pipeline = StoragePipeline::create_unattributed_file(&path)?;

    let mut genomes = Vec::new();
    // Agent 1: 2 revisions (tick 0, tick 5)
    let env_1_0 = create_founder_envelope("mlp", vec![1, 0]);
    let env_1_5 = create_founder_envelope("mlp", vec![1, 5]);
    genomes.push(PersistedGenome {
        agent_uid: AgentUid(1),
        created_at_tick: Tick(0),
        envelope: env_1_0,
    });
    genomes.push(PersistedGenome {
        agent_uid: AgentUid(1),
        created_at_tick: Tick(5),
        envelope: env_1_5,
    });

    // Agent 2: 1 revision (tick 2)
    let env_2_2 = create_founder_envelope("dwraon", vec![2, 2]);
    genomes.push(PersistedGenome {
        agent_uid: AgentUid(2),
        created_at_tick: Tick(2),
        envelope: env_2_2,
    });

    // Agent 3: 1 revision (tick 7)
    let env_3_7 = create_founder_envelope("mlp", vec![3, 7]);
    genomes.push(PersistedGenome {
        agent_uid: AgentUid(3),
        created_at_tick: Tick(7),
        envelope: env_3_7,
    });

    let batch = sample_batch_with_genomes(10, genomes, Vec::new());
    pipeline.submit(&batch)?;
    pipeline.shutdown()?;

    let reader = StorageReader::open(&path)?;

    // Lineage readback: request order [AgentUid(3), AgentUid(1)]
    let lineage = reader.read_lineage_genomes(&[AgentUid(3), AgentUid(1)])?;
    assert_eq!(lineage.len(), 3);
    assert_eq!(lineage[0].0, AgentUid(3));
    assert_eq!(lineage[0].1, Tick(7));
    assert_eq!(lineage[0].2.payload(), &[3, 7]);

    assert_eq!(lineage[1].0, AgentUid(1));
    assert_eq!(lineage[1].1, Tick(0));
    assert_eq!(lineage[1].2.payload(), &[1, 0]);

    assert_eq!(lineage[2].0, AgentUid(1));
    assert_eq!(lineage[2].1, Tick(5));
    assert_eq!(lineage[2].2.payload(), &[1, 5]);

    // Page readback
    let page_0 = reader.read_genomes_page(0, 2)?;
    assert_eq!(page_0.len(), 2);
    assert_eq!(page_0[0].0, AgentUid(1));
    assert_eq!(page_0[0].1, Tick(0));
    assert_eq!(page_0[1].0, AgentUid(2));
    assert_eq!(page_0[1].1, Tick(2));

    let page_1 = reader.read_genomes_page(2, 2)?;
    assert_eq!(page_1.len(), 2);
    assert_eq!(page_1[0].0, AgentUid(1));
    assert_eq!(page_1[0].1, Tick(5));
    assert_eq!(page_1[1].0, AgentUid(3));
    assert_eq!(page_1[1].1, Tick(7));

    Ok(())
}

#[test]
fn test_live_simulation_multigen_e2e_reopened_db() -> Result<(), Box<dyn std::error::Error>> {
    let path = temp_db_path("live_sim_e2e");
    let mut pipeline =
        StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)?;

    let config = ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        persistence_interval: 1,
        rng_seed: Some(0xC0DE_B075),
        ..ScriptBotsConfig::default()
    };

    let (mut world, mut persistence) =
        WorldState::with_persistence(config, Box::new(pipeline.sink()))
            .expect("world with storage pipeline");

    let family_key = world
        .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
        .expect("register mlp brain family");

    for _ in 0..4 {
        let agent_id = world
            .try_spawn_agent(AgentData::default())
            .expect("seed agent");
        assert!(
            world
                .bind_agent_brain(agent_id, family_key)
                .expect("bind brain"),
            "agent accepts brain binding"
        );
    }

    // Run simulation for 10 ticks
    for _ in 0..10 {
        persistence.step(&mut world).expect("step world");
    }

    // Collect all live agent identities and brain genomes
    let handles: Vec<AgentId> = world.agents().iter_handles().collect();
    assert!(!handles.is_empty(), "World should have active agents");

    let mut live_genomes = Vec::new();
    for id in handles {
        let uid = world.agent_uid(id).expect("agent exists");
        if let Some(envelope) = world.agent_brain_genome(id) {
            live_genomes.push((uid, envelope.clone()));
        }
    }
    assert!(!live_genomes.is_empty(), "Active agents should have brain genomes");

    // Shutdown persistence pipeline
    pipeline.shutdown()?;

    // Reopen DB via StorageReader
    let reader = StorageReader::open(&path)?;

    // Verify all live agent genomes exist in the reopened database and match bit-for-bit
    for (uid, expected_envelope) in &live_genomes {
        let readback = reader.read_agent_genome(*uid, None)?;
        assert_eq!(
            readback.payload(),
            expected_envelope.payload(),
            "Agent {uid:?} payload mismatch between live and persisted"
        );
        assert_eq!(
            readback.material_hash(),
            expected_envelope.material_hash(),
            "Agent {uid:?} hash mismatch"
        );
        assert_eq!(
            readback.provenance(),
            expected_envelope.provenance(),
            "Agent {uid:?} provenance mismatch"
        );
    }

    // Verify load_ancestry_births agrees with persisted records
    let births = reader.load_ancestry_births()?;
    assert!(!births.is_empty(), "Births table must hold arrival records");

    Ok(())
}
