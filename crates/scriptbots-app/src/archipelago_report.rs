//! Recorded isolated-island runs and offline archipelago reports.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use scriptbots_core::{PersistenceBatch, ScriptBotsConfig, WorldState, WorldStateError};
use scriptbots_runtime::{
    Archipelago, ArchipelagoConfig, EventJournalReader, HostCoreOptions, HostSessionId, IslandId,
    IslandSpec, JournalAdmission, JournalBatch, JournalBatchId, JournalPort, JournalReceipt,
    ShutdownCommitRequirement, Topology, VolatileJournal,
};
use scriptbots_storage::{ArchipelagoBarrierSink, RunManifestRecord, Storage, StorageReader};

use crate::{BrainPreset, BuildProvenanceV0, install_brains, seed_founding_population};

const MAX_CAPTURE_BYTES: usize = 64 * 1024 * 1024;
const MAX_RECORDED_TICKS: u64 = 10_000_000;

#[derive(Default)]
struct BarrierCapture {
    batches: BTreeMap<IslandId, (JournalBatchId, Arc<PersistenceBatch>)>,
    bytes: usize,
}

/// The host journal commits to bounded live memory. Science is persisted separately,
/// only after every island reaches the barrier; no receipt here claims database durability.
struct IslandCaptureJournal {
    island: IslandId,
    capture: Rc<RefCell<BarrierCapture>>,
    journal: VolatileJournal,
}

impl JournalPort for IslandCaptureJournal {
    fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
        let mut capture = self.capture.borrow_mut();
        if let Some(payload) = batch.persistence() {
            if let Some((identity, retained)) = capture.batches.get(&self.island) {
                if *identity != batch.id() {
                    return JournalAdmission::Full {
                        batch_id: batch.id(),
                        capacity: 1,
                    };
                }
                if !Arc::ptr_eq(retained, payload) {
                    return JournalAdmission::Closed {
                        batch_id: batch.id(),
                    };
                }
            } else if batch.retained_bytes() > MAX_CAPTURE_BYTES.saturating_sub(capture.bytes) {
                return JournalAdmission::Full {
                    batch_id: batch.id(),
                    capacity: MAX_CAPTURE_BYTES,
                };
            }
        }
        let admission = self.journal.try_admit(batch);
        if matches!(admission, JournalAdmission::Accepted { .. })
            && let Some(payload) = batch.persistence()
            && !capture.batches.contains_key(&self.island)
        {
            capture.bytes += batch.retained_bytes();
            capture
                .batches
                .insert(self.island, (batch.id(), Arc::clone(payload)));
        }
        admission
    }

    fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
        self.journal.poll_receipts(limit)
    }

    fn event_reader(&self, session: HostSessionId) -> Option<Arc<dyn EventJournalReader>> {
        self.journal.event_reader(session)
    }

    fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
        ShutdownCommitRequirement::CommittedVolatile
    }
}

fn build_recorded_archipelago(
    config: &ScriptBotsConfig,
    island_count: u32,
    root_seed: u64,
    preset: BrainPreset,
    capture: &Rc<RefCell<BarrierCapture>>,
) -> Result<Archipelago> {
    let options = HostCoreOptions::default();
    let mut construction_error = None;
    let constructed = Archipelago::with_factories(
        ArchipelagoConfig {
            islands: (0..island_count)
                .map(|id| IslandSpec {
                    id: IslandId(id),
                    label: format!("island-{id}"),
                    config: config.clone(),
                })
                .collect(),
            topology: Topology::Custom(Vec::new()),
            barrier_interval: NonZeroU64::MIN,
            master_seed: root_seed,
            host_options: options,
            migration: None,
        },
        |meta| {
            let result = (|| -> Result<WorldState> {
                let mut world = WorldState::new(meta.effective_config.clone())?;
                let installed = install_brains(&mut world, preset)?;
                seed_founding_population(&mut world, installed.population())?;
                Ok(world)
            })();
            result.map_err(|error| {
                // Preserve the full application error across the runtime's typed core
                // factory boundary. Construction aborts at this first failing island.
                construction_error = Some(error.context(format!("construct island {}", meta.id.0)));
                WorldStateError::InvalidConfig("application brain/founder construction failed")
            })
        },
        |meta| {
            Some(Box::new(IslandCaptureJournal {
                island: meta.id,
                capture: Rc::clone(capture),
                journal: VolatileJournal::default(),
            }))
        },
    );
    if let Some(error) = construction_error {
        return Err(error);
    }
    constructed.context("construct recorded archipelago")
}

/// Run isolated populations through sole-owner hosts and one atomic science writer.
///
/// The supplied manifest describes the first island's launch world. The embedded
/// `archipelago` extension and `islands` table identify the complete experiment. This
/// mode records science, not a durable host-command session or checkpoint-resume artifact.
///
/// # Errors
///
/// Refuses invalid run bounds, missing per-tick capture, unsupported brain construction,
/// inconsistent provenance, existing output files, incomplete barriers, and storage failures.
pub fn run_recorded_archipelago(
    mut config: ScriptBotsConfig,
    island_count: u32,
    ticks: u64,
    preset: BrainPreset,
    path: &str,
    manifest_for_world: impl FnOnce(&WorldState) -> Result<RunManifestRecord>,
) -> Result<serde_json::Value> {
    ensure!(
        (1..=scriptbots_runtime::archipelago::MAX_ISLANDS)
            .contains(&usize::try_from(island_count)?),
        "island count must be in 1..={}",
        scriptbots_runtime::archipelago::MAX_ISLANDS
    );
    ensure!(
        (1..=MAX_RECORDED_TICKS).contains(&ticks),
        "archipelago ticks must be in 1..={MAX_RECORDED_TICKS}"
    );
    ensure!(
        config.persistence_interval == 1,
        "recorded archipelago requires persistence_interval=1 so every island supplies every tick"
    );
    let root_seed = config
        .rng_seed
        .context("recorded archipelago requires an explicit rng_seed")?;
    // The launch seed belongs to the archipelago. Each effective island config receives
    // the runtime's versioned derivation instead of accidentally pinning identical seeds.
    config.rng_seed = None;
    let capture = Rc::new(RefCell::new(BarrierCapture::default()));
    let mut archipelago =
        build_recorded_archipelago(&config, island_count, root_seed, preset, &capture)?;
    let islands: Vec<_> = archipelago.islands().cloned().collect();
    let mut manifest = archipelago.with_island_world(IslandId(0), manifest_for_world)??;
    let mut manifest_json: serde_json::Value = serde_json::from_str(&manifest.manifest_json)?;
    let manifest_object = manifest_json
        .as_object_mut()
        .context("run manifest must be a JSON object")?;
    manifest_object.insert(
        "archipelago".to_owned(),
        serde_json::json!({
            "schema": "scriptbots.recorded-archipelago.v1", "root_seed": root_seed,
            "primary_manifest_island": 0, "island_count": island_count,
            "ticks": ticks, "barrier_interval": 1, "migration": "disabled",
            "step_topology": "sequential_ascending", "host_journal": "committed_volatile",
            "science_storage": "complete_barriers", "islands": islands,
        }),
    );
    manifest.manifest_json = serde_json::to_string(&manifest_json)?;
    let run_id = manifest.run_id;
    let mut storage = Storage::create_new_file_for_run(path, manifest)?;
    let execution = (|| -> Result<serde_json::Value> {
        storage.persist_islands(&islands)?;
        let mut sink = ArchipelagoBarrierSink::new(islands.iter().map(|island| island.id))?;
        for expected_tick in 1..=ticks {
            let barrier = archipelago
                .step_to_barrier()
                .with_context(|| format!("advance barrier {expected_tick}"))?;
            ensure!(
                barrier.barrier_tick.0 == expected_tick,
                "barrier tick differs from requested tick"
            );
            {
                let captured = capture.borrow();
                for (island, (_, payload)) in &captured.batches {
                    ensure!(
                        payload.summary.tick.0 == expected_tick,
                        "island {} supplied stale tick {} at barrier {expected_tick}",
                        island.0,
                        payload.summary.tick.0
                    );
                    sink.admit(*island, (**payload).clone())?;
                }
            }
            // An incomplete barrier refuses here without writing any of its rows.
            // Retain both captures until application and durability have been observed.
            storage
                .persist_barrier_from(&sink)
                .with_context(|| format!("persist complete barrier {expected_tick}"))?;
            let watermarks = storage.flush_with_watermarks()?;
            ensure!(
                watermarks.admitted.map(|id| id.get()) == Some(expected_tick)
                    && watermarks.admitted == watermarks.applied
                    && watermarks.applied == watermarks.durable,
                "barrier {expected_tick} has incomplete persistence watermarks: {watermarks:?}"
            );
            tracing::info!(
                tick = expected_tick,
                islands = island_count,
                durable_batch = watermarks.durable.map(|id| id.get()),
                "persisted complete island barrier"
            );
            sink.clear();
            let mut captured = capture.borrow_mut();
            captured.batches.clear();
            captured.bytes = 0;
        }
        let final_islands = islands.iter().map(|island| {
            let digest = archipelago.island_digest(island.id)?;
            let snapshot = archipelago.island_snapshot(island.id).context("missing final island snapshot")?;
            let summary = snapshot.completed_summary.as_ref().context("missing completed island summary")?;
            Ok(serde_json::json!({"island_id": island.id.0, "digest": digest,
                "population": snapshot.world.agents.len(), "total_energy": f64::from(summary.total_energy)}))
        }).collect::<Result<Vec<_>>>()?;
        let watermarks = storage.persistence_watermarks()?;
        Ok(serde_json::json!({
            "schema": "scriptbots.recorded-archipelago.v1", "status": "complete",
            "db": path, "run_id": run_id, "root_seed": root_seed, "ticks": ticks,
            "island_count": island_count, "migration": "disabled",
            "step_topology": "sequential_ascending", "host_journal": "committed_volatile",
            "source": BuildProvenanceV0::current().source_revision,
            "watermarks": {"admitted": watermarks.admitted.map(|id| id.get()),
                "applied": watermarks.applied.map(|id| id.get()), "durable": watermarks.durable.map(|id| id.get())},
            "islands": final_islands,
        }))
    })();
    let close = storage.close();
    match (execution, close) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error.into()),
        (Err(error), Err(close_error)) => {
            bail!("{error:#}; storage close also failed: {close_error}")
        }
    }
}

/// CLI arguments for the `report-archipelago` subcommand or mode.
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct ReportArchipelagoArgs {
    /// Path to the SQLite run database to inspect.
    #[arg(value_name = "DB")]
    pub db: PathBuf,

    /// Optional path to write report JSON.
    #[arg(long, value_name = "JSON_PATH")]
    pub json: Option<PathBuf>,

    /// Require per-island population conservation to close (exit nonzero on violation).
    #[arg(long, default_value_t = false)]
    pub verify_conservation: bool,
}

/// Generate a human-readable text report and optional JSON for an archipelago run database.
pub fn run_archipelago_report(args: &ReportArchipelagoArgs) -> Result<bool> {
    let path_str = args.db.to_string_lossy().to_string();
    let reader = StorageReader::open(&path_str)
        .with_context(|| format!("failed to open run database at {}", args.db.display()))?;
    let report = reader
        .archipelago_report()
        .context("failed to reconstruct archipelago report from database")?;

    println!("=== ARCHIPELAGO OFFLINE RECONSTRUCTION REPORT ===");
    println!("Database:   {}", args.db.display());
    println!("Run ID:     {}", report.run_id);
    println!("Islands:    {}", report.islands.len());
    println!("Migrations: {}", report.migrations.len());
    println!();

    println!("--- CONFIGURED ISLANDS ---");
    for island in &report.islands {
        let hist_len = report
            .histories
            .get(&island.island_id)
            .map(|h| h.points.len())
            .unwrap_or(0);
        println!(
            "  Island #{:<2} [label: {:<12}] config_hash: 0x{:016x} ({} history points)",
            island.island_id, island.label, island.config_hash, hist_len
        );
    }
    println!();

    println!("--- MIGRATION MULTIGRAPH EDGES ---");
    if report.migration_graph.is_empty() {
        println!("  (No migrations recorded)");
    } else {
        for edge in &report.migration_graph {
            println!(
                "  Island {} -> Island {}: {} emigrant(s)",
                edge.from, edge.to, edge.count
            );
        }
    }
    println!();

    println!("--- POPULATION CONSERVATION AUDIT ---");
    println!("  Passed:          {}", report.conservation_audit.passed);
    println!(
        "  Islands Checked: {}",
        report.conservation_audit.total_islands_checked
    );
    println!(
        "  Ticks Checked:   {}",
        report.conservation_audit.total_ticks_checked
    );
    println!(
        "  Breaches:        {}",
        report.conservation_audit.breaches.len()
    );

    if !report.conservation_audit.breaches.is_empty() {
        eprintln!();
        eprintln!("CONSERVATION BREACHES DETECTED:");
        for breach in &report.conservation_audit.breaches {
            eprintln!(
                "  Island {} at tick {}: expected {}, recorded {} (births: {}, deaths: {}, immigrations: {}, emigrations: {})",
                breach.island_id,
                breach.tick,
                breach.expected_population,
                breach.recorded_population,
                breach.births,
                breach.deaths,
                breach.immigrations,
                breach.emigrations
            );
        }
    }

    if let Some(json_path) = &args.json {
        let json_str = serde_json::to_string_pretty(&report)?;
        if let Some(parent) = json_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(json_path, json_str)?;
        println!();
        println!("Wrote archipelago report JSON to {}", json_path.display());
    }

    Ok(!args.verify_conservation || report.conservation_audit.passed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_runtime::{
        CommandEnvelope, CommandId, HostCommand, HostCore, HostPort, JournalState,
        ManualHostDriver, ManualInstant, PlaybackSnapshot,
    };

    #[test]
    fn recorded_capture_retries_the_same_tick_and_acknowledges_only_volatile_storage() -> Result<()>
    {
        let capture = Rc::new(RefCell::new(BarrierCapture {
            bytes: MAX_CAPTURE_BYTES,
            ..Default::default()
        }));
        let options = HostCoreOptions {
            initial_playback: PlaybackSnapshot {
                paused: true,
                speed_multiplier: 1.0,
            },
            ..Default::default()
        };
        let world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(42),
            persistence_interval: 1,
            closed: true,
            world_width: 64,
            world_height: 64,
            food_cell_size: 16,
            ..Default::default()
        })?;
        let mut host = HostCore::with_journal(
            HostSessionId::new(42),
            world,
            options,
            Box::new(IslandCaptureJournal {
                island: IslandId(7),
                capture: Rc::clone(&capture),
                journal: VolatileJournal::default(),
            }),
        )?;
        let mut port = host.local_port();
        let command = CommandId::new(1);
        port.submit(CommandEnvelope::new(command, HostCommand::Step))?;
        host.drive(ManualInstant::from_nanos(0))?;
        assert!(
            capture.borrow().batches.is_empty(),
            "the byte bound must refuse before capturing science"
        );
        let retained = host
            .pending_journal_batch()
            .context("bounded capture must retain the completed batch")?;
        let tick = host.world_tick();
        assert_eq!(tick.0, 1);
        capture.borrow_mut().bytes = 0;
        host.retry_retained_journal()?;
        host.drive(ManualInstant::from_nanos(1))?;
        let capture = capture.borrow();
        assert_eq!(capture.batches.len(), 1);
        let (id, payload) = capture
            .batches
            .get(&IslandId(7))
            .context("retry must capture the island")?;
        assert_eq!(*id, retained.id());
        assert!(Arc::ptr_eq(
            payload,
            retained.persistence().context("science payload")?
        ));
        assert_eq!(
            host.world_tick(),
            tick,
            "retry must not execute a second science tick"
        );
        assert_eq!(
            port.command_status(command)?
                .context("command receipt")?
                .journal(),
            &JournalState::CommittedVolatile
        );
        Ok(())
    }
}
