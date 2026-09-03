//! E2E tests for the archipelago reconstruction report CLI and per-island population conservation audit (bd-16g.5.5.5).

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use scriptbots_app::archipelago_report::{ReportArchipelagoArgs, run_archipelago_report};
use scriptbots_core::{AgentData, PersistenceBatch, ScriptBotsConfig, WorldState};
use scriptbots_runtime::migrator::EmigrantSelectionRule;
use scriptbots_runtime::{
    Archipelago, ArchipelagoConfig, ArchipelagoMigration, HostCoreOptions, IslandId, IslandSpec,
    Topology,
};
use scriptbots_storage::{ArchipelagoBarrierSink, ArchipelagoReport, Storage, StorageReader};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn temp_db_path(label: &str) -> PathBuf {
    let nonce = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "scriptbots-{label}-{}-{nonce}.db",
        std::process::id()
    ))
}

struct HarnessJournal {
    island: IslandId,
    captured: Rc<RefCell<Vec<(IslandId, PersistenceBatch)>>>,
    pending: Rc<RefCell<VecDeque<scriptbots_runtime::JournalReceipt>>>,
}

impl scriptbots_runtime::JournalPort for HarnessJournal {
    fn try_admit(
        &mut self,
        batch: &Arc<scriptbots_runtime::JournalBatch>,
    ) -> scriptbots_runtime::JournalAdmission {
        if let Some(payload) = batch.persistence() {
            self.captured
                .borrow_mut()
                .push((self.island, (**payload).clone()));
        }
        self.pending
            .borrow_mut()
            .push_back(scriptbots_runtime::JournalReceipt::new(
                batch.id(),
                scriptbots_runtime::JournalReceiptState::Durable,
            ));
        scriptbots_runtime::JournalAdmission::Accepted {
            batch_id: batch.id(),
        }
    }

    fn poll_receipts(&mut self, limit: usize) -> Vec<scriptbots_runtime::JournalReceipt> {
        let mut pending = self.pending.borrow_mut();
        let count = limit.min(pending.len());
        pending.drain(..count).collect()
    }

    fn shutdown_commit_requirement(&self) -> scriptbots_runtime::ShutdownCommitRequirement {
        scriptbots_runtime::ShutdownCommitRequirement::CommittedVolatile
    }
}

#[test]
fn test_archipelago_report_reconstructs_and_verifies_conservation_e2e()
-> Result<(), Box<dyn std::error::Error>> {
    const ISLANDS: u32 = 4;
    const BARRIERS: u32 = 3;

    let db_path = temp_db_path("archipelago-report-e2e");
    let json_path = temp_db_path("archipelago-report-out").with_extension("json");
    let path_str = db_path.to_string_lossy().to_string();

    let mut storage = Storage::create_unattributed_file_with_thresholds(&path_str, 1, 1, 1, 1)?;
    let captured: Rc<RefCell<Vec<(IslandId, PersistenceBatch)>>> =
        Rc::new(RefCell::new(Vec::new()));

    let base = ScriptBotsConfig {
        world_width: 64,
        world_height: 64,
        food_cell_size: 16,
        rng_seed: Some(0x0151_a4d0),
        closed: true,
        history_capacity: 8,
        persistence_interval: 1,
        narrative_interval: 0,
        ..ScriptBotsConfig::default()
    };

    let specs: Vec<IslandSpec> = (0..ISLANDS)
        .map(|id| IslandSpec {
            id: IslandId(id),
            label: format!("island-{id}"),
            config: base.clone(),
        })
        .collect();

    let capture_handle = Rc::clone(&captured);
    let mut archipelago = Archipelago::with_factories(
        ArchipelagoConfig {
            islands: specs,
            topology: Topology::Ring,
            barrier_interval: std::num::NonZeroU64::new(1).expect("nonzero"),
            master_seed: 0x00c0_ffee,
            host_options: HostCoreOptions::default(),
            migration: Some(ArchipelagoMigration {
                interval_ticks: 1,
                emigrants_per_edge: 1,
                selection_rule: EmigrantSelectionRule::Fittest,
            }),
        },
        |meta| {
            let mut world = WorldState::new(meta.effective_config.clone())?;
            for _ in 0..5 {
                world
                    .try_spawn_agent(AgentData::default())
                    .expect("deterministic founding agent is finite");
            }
            Ok(world)
        },
        |meta| {
            Some(Box::new(HarnessJournal {
                island: meta.id,
                captured: Rc::clone(&capture_handle),
                pending: Rc::new(RefCell::new(VecDeque::new())),
            }) as Box<dyn scriptbots_runtime::JournalPort>)
        },
    )?;

    // Persist static metadata once at start
    let metas: Vec<scriptbots_runtime::IslandMeta> = archipelago.islands().cloned().collect();
    storage.persist_islands(&metas)?;

    for _ in 0..BARRIERS {
        let report = archipelago.step_to_barrier()?;

        let drained: Vec<(IslandId, PersistenceBatch)> = captured.borrow_mut().drain(..).collect();
        let mut by_tick: BTreeMap<u64, Vec<(IslandId, PersistenceBatch)>> = BTreeMap::new();
        for (island, batch) in drained {
            by_tick
                .entry(batch.summary.tick.0)
                .or_default()
                .push((island, batch));
        }

        for (_tick, island_batches) in by_tick {
            if u32::try_from(island_batches.len())? != ISLANDS {
                continue;
            }
            let mut sink = ArchipelagoBarrierSink::new((0..ISLANDS).map(IslandId))?;
            for (island, batch) in island_batches {
                sink.admit(island, batch)?;
            }
            if let Some(migration_report) = &report.migration {
                sink.admit_migrations(migration_report.moves.clone());
            }
            storage.persist_barrier_from(&sink)?;
        }
    }

    storage.flush()?;
    storage.close()?;

    // =======================================================================
    // 1. Run CLI report in strict conservation-verification mode
    // =======================================================================
    let args = ReportArchipelagoArgs {
        db: db_path.clone(),
        json: Some(json_path.clone()),
        verify_conservation: true,
    };

    let pass = run_archipelago_report(&args)?;
    assert!(
        pass,
        "run_archipelago_report must pass when conservation holds"
    );

    // =======================================================================
    // 2. Validate JSON output artifact
    // =======================================================================
    let json_bytes = std::fs::read(&json_path)?;
    let reconstructed: ArchipelagoReport = serde_json::from_slice(&json_bytes)?;

    assert_eq!(reconstructed.islands.len(), ISLANDS as usize);
    assert_eq!(reconstructed.histories.len(), ISLANDS as usize);
    for island in 0..ISLANDS {
        let hist = reconstructed
            .histories
            .get(&island)
            .expect("must have history for each island");
        assert_eq!(hist.points.len(), BARRIERS as usize);
    }

    // Migration graph must have 8 edges for Ring topology of 4 islands
    assert_eq!(reconstructed.migration_graph.len(), 8);
    assert_eq!(
        reconstructed.migrations.len(),
        8 * (BARRIERS as usize),
        "total migrations across 3 barriers with 8 edges and 1 emigrant/edge"
    );

    // Conservation audit must have passed
    assert!(reconstructed.conservation_audit.passed);
    assert_eq!(
        reconstructed.conservation_audit.total_islands_checked,
        ISLANDS as usize
    );
    assert_eq!(reconstructed.conservation_audit.breaches.len(), 0);

    // =======================================================================
    // 3. Negative control: inject a phantom death to break conservation
    // =======================================================================
    {
        let reader = StorageReader::open(&path_str)?;
        let run_id_str = reader.run_id().to_string();
        reader.close()?;

        let conn = fsqlite::Connection::open(&path_str)?;
        conn.execute_with_params(
            "INSERT INTO deaths (run_id, tick, agent_uid, age, generation, herbivore_tendency, brain_kind, brain_key, energy, food_balance_total, cause, was_hybrid, spike_attacker, spike_victim, hit_carnivore, hit_herbivore, hit_by_carnivore, hit_by_herbivore, island_id)
             VALUES (?1, 2, 999999, 10, 1, 0.5, NULL, NULL, 1.0, 0.0, 'phantom_fault', 0, 0, 0, 0, 0, 0, 0, 0)",
            &[run_id_str.into()],
        )?;
        conn.close()?;
    }

    let fail_pass = run_archipelago_report(&args)?;
    assert!(
        !fail_pass,
        "run_archipelago_report must return false when a conservation breach is injected"
    );

    Ok(())
}
