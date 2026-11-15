//! Recorded island runs, optional deterministic migration and offline reports.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use scriptbots_core::{PersistenceBatch, ScriptBotsConfig, WorldState, WorldStateError};
use scriptbots_runtime::archipelago::{ArchipelagoCheckpointV1, MAX_ARCHIPELAGO_CHECKPOINT_BYTES};
use scriptbots_runtime::migrator::EmigrantSelectionRule;
use scriptbots_runtime::{
    Archipelago, ArchipelagoConfig, ArchipelagoMigration, EventJournalReader, HostCheckpointError,
    HostCoreOptions, HostJournalCutV1, HostSessionId, IslandId, IslandSpec, JournalAdmission,
    JournalBatch, JournalBatchId, JournalPort, JournalReceipt, ShutdownCommitRequirement, Topology,
    VolatileJournal, VolatileJournalCheckpointV1,
};
use scriptbots_storage::{ArchipelagoBarrierSink, RunManifestRecord, Storage, StorageReader};

use crate::{
    BrainPreset, BuildProvenanceV0, FounderRecipe, create_brain_registry_for_config,
    install_brains, seed_founders,
};

const MAX_CAPTURE_BYTES: usize = 64 * 1024 * 1024;
const MAX_RECORDED_TICKS: u64 = 10_000_000;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MigrationSelection {
    Random,
    Fittest,
    Youngest,
    Wanderer,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MigrationConnections {
    Ring,
    FullyConnected,
}

/// Optional policies for recorded islands; migration is disabled unless requested.
#[derive(Args, Debug, Clone, Default)]
pub struct RecordedArchipelagoOptions {
    /// Save a complete portable owner checkpoint after the final recorded barrier.
    #[arg(
        long = "archipelago-checkpoint",
        value_name = "FILE",
        requires = "run_archipelago"
    )]
    checkpoint_out: Option<PathBuf>,
    /// Per-island scenario overlay, repeatable as ID:JSON_OR_PATH.
    #[arg(long = "island-overlay", value_name = "ID:SPEC", action = clap::ArgAction::Append)]
    island_overlays: Vec<String>,
    /// Migrate every N recorded ticks (positive; requires at least two islands).
    #[arg(long, value_name = "N", requires = "run_archipelago")]
    migration_interval: Option<NonZeroU64>,
    /// Emigrants per directed edge per migration (default: 1).
    #[arg(long, value_name = "N", requires = "migration_interval")]
    migration_emigrants: Option<NonZeroU32>,
    /// Emigrant selection (default: fittest).
    #[arg(long, value_enum, requires = "migration_interval")]
    migration_selection: Option<MigrationSelection>,
    /// Bidirectional island connectivity (default: ring).
    #[arg(long, value_enum, requires = "migration_interval")]
    migration_topology: Option<MigrationConnections>,
}

impl RecordedArchipelagoOptions {
    fn migration_policy(
        &self,
        island_count: u32,
    ) -> Result<(Topology, Option<ArchipelagoMigration>)> {
        let Some(interval) = self.migration_interval else {
            ensure!(
                self.migration_emigrants.is_none()
                    && self.migration_selection.is_none()
                    && self.migration_topology.is_none(),
                "migration settings require --migration-interval"
            );
            return Ok((Topology::Custom(Vec::new()), None));
        };
        ensure!(island_count >= 2, "migration requires at least two islands");
        let topology = match self
            .migration_topology
            .unwrap_or(MigrationConnections::Ring)
        {
            MigrationConnections::Ring => Topology::Ring,
            MigrationConnections::FullyConnected => Topology::FullyConnected,
        };
        let selection_rule = match self
            .migration_selection
            .unwrap_or(MigrationSelection::Fittest)
        {
            MigrationSelection::Random => EmigrantSelectionRule::Random,
            MigrationSelection::Fittest => EmigrantSelectionRule::Fittest,
            MigrationSelection::Youngest => EmigrantSelectionRule::Youngest,
            MigrationSelection::Wanderer => EmigrantSelectionRule::Wanderer,
        };
        Ok((
            topology,
            Some(ArchipelagoMigration {
                interval_ticks: interval.get(),
                emigrants_per_edge: usize::try_from(
                    self.migration_emigrants.map_or(1, NonZeroU32::get),
                )?,
                selection_rule,
            }),
        ))
    }
}

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
    fn capture_volatile_checkpoint_v1(
        &self,
        session_id: HostSessionId,
    ) -> Result<VolatileJournalCheckpointV1, HostCheckpointError> {
        self.journal.capture_volatile_checkpoint_v1(session_id)
    }

    fn continuation_cut(&self) -> Option<HostJournalCutV1> {
        self.journal.continuation_cut()
    }

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

/// Parse a CLI island overlay argument in `ID:JSON_OR_PATH` format.
pub fn parse_island_overlay(spec: &str) -> Result<(u32, serde_json::Value)> {
    let (id_str, rest) = spec
        .split_once(':')
        .context("expected island overlay format ID:JSON_OR_PATH")?;
    let id: u32 = id_str
        .trim()
        .parse()
        .with_context(|| format!("invalid island id '{id_str}' in overlay spec"))?;
    let rest = rest.trim();
    let value = if rest.starts_with('{') {
        serde_json::from_str(rest)
            .with_context(|| format!("invalid overlay JSON for island {id}: '{rest}'"))?
    } else {
        let content = fs::read_to_string(rest)
            .with_context(|| format!("failed to read overlay file '{rest}' for island {id}"))?;
        if rest.ends_with(".toml") {
            let toml_val: toml::Value = toml::from_str(&content)
                .with_context(|| format!("invalid overlay TOML in '{rest}'"))?;
            serde_json::to_value(toml_val)
                .with_context(|| format!("failed to convert TOML overlay '{rest}' to JSON"))?
        } else {
            serde_json::from_str(&content)
                .with_context(|| format!("invalid overlay JSON in '{rest}'"))?
        }
    };
    Ok((id, value))
}

#[allow(
    clippy::too_many_arguments,
    reason = "explicit launch inputs retain the selected brain and founder recipes at construction"
)]
fn build_recorded_archipelago(
    config: &ScriptBotsConfig,
    island_count: u32,
    root_seed: u64,
    preset: BrainPreset,
    founders: FounderRecipe,
    overlays: &BTreeMap<u32, serde_json::Value>,
    capture: &Rc<RefCell<BarrierCapture>>,
    policy: (Topology, Option<ArchipelagoMigration>),
) -> Result<Archipelago> {
    let options = HostCoreOptions::default();
    let mut construction_error = None;
    let specs: Result<Vec<IslandSpec>> = (0..island_count)
        .map(|id| {
            let label = format!("island-{id}");
            if let Some(overlay) = overlays.get(&id) {
                IslandSpec::with_overlay(IslandId(id), label, config, overlay)
                    .with_context(|| format!("failed to apply overlay for island {id}"))
            } else {
                Ok(IslandSpec::new(IslandId(id), label, config.clone()))
            }
        })
        .collect();
    let island_specs = specs?;
    let constructed = Archipelago::with_factories(
        ArchipelagoConfig {
            islands: island_specs,
            topology: policy.0,
            barrier_interval: NonZeroU64::MIN,
            master_seed: root_seed,
            host_options: options,
            migration: policy.1,
        },
        |meta| {
            let result = (|| -> Result<WorldState> {
                let mut world = WorldState::new(meta.effective_config.clone())?;
                let installed = install_brains(&mut world, preset)?;
                seed_founders(&mut world, installed.population(), founders)?;
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

/// Run populations through sole-owner hosts and one atomic science/migration writer.
///
/// The supplied manifest describes the first island's launch world. The embedded
/// `archipelago` extension and `islands` table identify the complete experiment. This
/// mode records science, not a durable host-command session. An optional portable checkpoint
/// preserves the complete owners and retained volatile journals; continuing it does not append
/// to this database or turn its volatile command receipts into database durability claims.
///
/// # Errors
///
/// Refuses invalid run bounds, missing per-tick capture, unsupported brain construction,
/// inconsistent provenance, existing output files, incomplete barriers, and storage failures.
#[allow(
    clippy::too_many_arguments,
    reason = "the recorded launch binds brain and founder selection to its manifest and output"
)]
pub fn run_recorded_archipelago(
    mut config: ScriptBotsConfig,
    island_count: u32,
    ticks: u64,
    preset: BrainPreset,
    founders: FounderRecipe,
    path: &str,
    options: &RecordedArchipelagoOptions,
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
    if let Some(output) = &options.checkpoint_out {
        ensure_new_checkpoint_destination(output)?;
        ensure!(
            output != Path::new(path),
            "archipelago checkpoint and database must have distinct paths"
        );
    }
    let policy = options.migration_policy(island_count)?;
    let rayon_threads = rayon::current_num_threads();
    let migration_status = if policy.1.is_some() {
        "enabled"
    } else {
        "disabled"
    };
    let migration_policy = policy.1.as_ref().map(|migration| {
        serde_json::json!({
            "interval_ticks": migration.interval_ticks,
            "emigrants_per_directed_edge": migration.emigrants_per_edge,
            "selection_rule": migration.selection_rule,
            "topology": policy.0, "edge_direction": "bidirectional",
        })
    });
    let mut parsed_overlays = BTreeMap::new();
    for spec in &options.island_overlays {
        let (id, val) = parse_island_overlay(spec)?;
        ensure!(
            id < island_count,
            "overlay island {id} is outside the configured islands"
        );
        ensure!(
            parsed_overlays.insert(id, val).is_none(),
            "duplicate overlay for island {id}"
        );
    }
    let root_seed = config
        .rng_seed
        .context("recorded archipelago requires an explicit rng_seed")?;
    // The launch seed belongs to the archipelago. Each effective island config receives
    // the runtime's versioned derivation instead of accidentally pinning identical seeds.
    config.rng_seed = None;
    let capture = Rc::new(RefCell::new(BarrierCapture::default()));
    let mut archipelago = build_recorded_archipelago(
        &config,
        island_count,
        root_seed,
        preset,
        founders,
        &parsed_overlays,
        &capture,
        policy,
    )?;
    let islands: Vec<_> = archipelago.islands().cloned().collect();
    let mut manifest = archipelago.with_island_world(IslandId(0), manifest_for_world)??;
    let mut manifest_json: serde_json::Value = serde_json::from_str(&manifest.manifest_json)?;
    let population_recipe = manifest_json
        .pointer("/scenario/population_recipe")
        .and_then(serde_json::Value::as_str)
        .context("recorded archipelago manifest must declare scenario.population_recipe")?;
    ensure!(
        FounderRecipe::from_population_recipe(population_recipe) == Some(founders),
        "recorded archipelago founder recipe {} differs from manifest declaration {population_recipe}",
        founders.id()
    );
    let manifest_object = manifest_json
        .as_object_mut()
        .context("run manifest must be a JSON object")?;
    manifest_object.insert(
        "archipelago".to_owned(),
        serde_json::json!({
            "schema": "scriptbots.recorded-archipelago.v1", "root_seed": root_seed,
            "primary_manifest_island": 0, "island_count": island_count,
            "founder_recipe": founders.id(),
            "ticks": ticks, "barrier_interval": 1, "migration": migration_status,
            "migration_policy": migration_policy,
            "tick_summary_phase": "before_migration", "membership_phase": "after_migration",
            "rayon_threads": rayon_threads,
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
        let mut migration_count = 0_u64;
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
            if let Some(migration) = &barrier.migration {
                sink.admit_migrations(migration.moves.iter().copied());
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
                migrations = sink.migrations().len(),
                "persisted complete island barrier"
            );
            migration_count = migration_count
                .checked_add(u64::try_from(sink.migrations().len())?)
                .context("recorded migration count overflow")?;
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
                "population": snapshot.world.agents.len(), "population_phase": "after_migration",
                "recorded_population": summary.agent_count,
                "total_energy": f64::from(summary.total_energy), "total_energy_phase": "before_migration"}))
        }).collect::<Result<Vec<_>>>()?;
        let watermarks = storage.persistence_watermarks()?;
        Ok(serde_json::json!({
            "schema": "scriptbots.recorded-archipelago.v1", "status": "complete",
            "db": path, "run_id": run_id, "root_seed": root_seed, "ticks": ticks,
            "island_count": island_count, "migration": migration_status,
            "founder_recipe": founders.id(),
            "migration_policy": migration_policy, "migration_count": migration_count,
            "rayon_threads": rayon_threads,
            "step_topology": "sequential_ascending", "host_journal": "committed_volatile",
            "source": BuildProvenanceV0::current().source_revision,
            "watermarks": {"admitted": watermarks.admitted.map(|id| id.get()),
                "applied": watermarks.applied.map(|id| id.get()), "durable": watermarks.durable.map(|id| id.get())},
            "islands": final_islands,
        }))
    })();
    let close = storage.close();
    match (execution, close) {
        (Ok(mut result), Ok(())) => {
            if let Some(output) = &options.checkpoint_out {
                let checkpoint = archipelago.checkpoint_v1().context(
                    "recorded science is saved, but complete island owner capture refused",
                )?;
                result["checkpoint"] = write_archipelago_checkpoint(output, &checkpoint).context(
                    "recorded science is saved, but checkpoint publication is unconfirmed",
                )?;
            }
            Ok(result)
        }
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error.into()),
        (Err(error), Err(close_error)) => {
            bail!("{error:#}; storage close also failed: {close_error}")
        }
    }
}

/// Continue a portable multi-island owner cut without modifying its original run database.
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct ContinueArchipelagoArgs {
    /// Complete portable archipelago checkpoint saved by a previous run.
    #[arg(value_name = "CHECKPOINT")]
    pub checkpoint: PathBuf,
    /// New checkpoint destination; existing files are never replaced.
    #[arg(long, short = 'o', value_name = "FILE")]
    pub output: PathBuf,
    /// Additional complete barriers, using the saved barrier and migration intervals.
    #[arg(long, value_name = "N")]
    pub barriers: NonZeroU64,
    /// Original executable brain preset; a mismatched registry refuses restoration.
    #[arg(long, value_enum, default_value_t = BrainPreset::Mixed)]
    pub brain: BrainPreset,
    /// Maximum Rayon worker threads for the continued scientific kernel (1..=64).
    #[arg(long, default_value = "1", value_name = "N")]
    pub threads: NonZeroUsize,
}

/// Execute a bounded portable continuation and save its exact final owner cut.
///
/// Both input and output are bounded checksummed artifacts. The restored islands retain their
/// original command/session identities and actual volatile history. No database is opened, no
/// founding population is recreated, and no elapsed wall time is converted into science ticks.
pub fn continue_archipelago(args: &ContinueArchipelagoArgs) -> Result<serde_json::Value> {
    ensure_new_checkpoint_destination(&args.output)?;
    ensure!(
        args.threads.get() <= 64,
        "continuation threads must be in 1..=64"
    );
    let (checkpoint, source_blake3) = read_archipelago_checkpoint(&args.checkpoint)?;
    let source_tick = checkpoint.barrier_tick().0;
    let source_epoch = checkpoint.epoch();
    let island_count = checkpoint.island_metadata().len();
    let ticks_per_island = args
        .barriers
        .get()
        .checked_mul(checkpoint.barrier_interval_ticks())
        .context("continued tick horizon overflowed")?;
    let transitions = ticks_per_island
        .checked_mul(u64::try_from(island_count)?)
        .context("continued island-transition budget overflowed")?;
    ensure!(
        transitions <= MAX_RECORDED_TICKS,
        "continuation requires {transitions} island transitions; maximum is {MAX_RECORDED_TICKS}"
    );
    let target_tick = source_tick
        .checked_add(ticks_per_island)
        .context("continued target tick overflowed")?;
    let target_epoch = source_epoch
        .checked_add(args.barriers.get())
        .context("continued target epoch overflowed")?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads.get())
        .build()?;
    pool.install(|| {
        let mut archipelago = Archipelago::restore_checkpoint_v1(&checkpoint, |meta| {
            create_brain_registry_for_config(args.brain, &meta.effective_config)
                .map_err(|error| format!("{error:#}"))
        })
        .context("saved archipelago refused executable owner restoration")?;
        let mut migration_count = 0_u64;
        for _ in 0..args.barriers.get() {
            let report = archipelago.step_to_barrier().context(
                "continued archipelago did not complete its next barrier; source is unchanged",
            )?;
            if let Some(migration) = &report.migration {
                migration_count = migration_count
                    .checked_add(u64::try_from(migration.moves.len())?)
                    .context("continued migration count overflowed")?;
            }
        }
        let final_cut = archipelago
            .checkpoint_v1()
            .context("continued islands could not settle a complete owner checkpoint")?;
        ensure!(
            final_cut.barrier_tick().0 == target_tick && final_cut.epoch() == target_epoch,
            "continued owner cut does not match the requested barrier horizon"
        );
        let islands = archipelago
            .islands()
            .map(|island| {
                let snapshot = archipelago
                    .island_snapshot(island.id)
                    .context("continued island has no committed snapshot")?;
                Ok(serde_json::json!({
                    "island_id": island.id.0,
                    "session_id": island.session_id.get(),
                    "digest": archipelago.island_digest(island.id)?,
                    "population": snapshot.world.agents.len(),
                    "population_phase": "after_migration",
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        let saved = write_archipelago_checkpoint(&args.output, &final_cut)?;
        Ok(serde_json::json!({
            "schema": "scriptbots.archipelago-continuation.v1", "status": "complete",
            "source_checkpoint": args.checkpoint, "source_blake3": source_blake3,
            "source_tick": source_tick, "source_epoch": source_epoch,
            "tick": target_tick, "epoch": target_epoch,
            "additional_barriers": args.barriers.get(), "ticks_per_island": ticks_per_island,
            "island_transitions": transitions, "island_count": island_count,
            "root_seed": checkpoint.master_seed(), "migration_count": migration_count,
            "brain_preset": args.brain.as_str(), "rayon_threads": rayon::current_num_threads(),
            "step_topology": "sequential_ascending", "host_journal": "committed_volatile",
            "science_database_continued": false,
            "source": BuildProvenanceV0::current().source_revision,
            "checkpoint": saved, "islands": islands,
        }))
    })
}

fn checkpoint_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

fn ensure_new_checkpoint_destination(path: &Path) -> Result<()> {
    ensure!(
        path.file_name().is_some(),
        "checkpoint output must name a file"
    );
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("inspect checkpoint output"),
        Ok(_) => bail!("checkpoint output already exists: {}", path.display()),
    }
    ensure!(
        fs::metadata(checkpoint_parent(path))?.is_dir(),
        "checkpoint output parent must be an existing directory"
    );
    Ok(())
}

fn read_archipelago_checkpoint(path: &Path) -> Result<(ArchipelagoCheckpointV1, String)> {
    ensure!(
        fs::symlink_metadata(path)?.is_file(),
        "checkpoint input must be a regular file"
    );
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options
        .open(path)
        .context("open portable archipelago checkpoint")?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file(),
        "checkpoint input must be a regular file"
    );
    let limit = u64::try_from(MAX_ARCHIPELAGO_CHECKPOINT_BYTES)?;
    ensure!(
        metadata.len() <= limit,
        "checkpoint input exceeds {limit} bytes"
    );
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= MAX_ARCHIPELAGO_CHECKPOINT_BYTES,
        "checkpoint input grew beyond its byte bound"
    );
    let checkpoint = ArchipelagoCheckpointV1::decode(&bytes)
        .context("portable archipelago checkpoint validation failed")?;
    Ok((checkpoint, blake3::hash(&bytes).to_hex().to_string()))
}

fn write_archipelago_checkpoint(
    path: &Path,
    checkpoint: &ArchipelagoCheckpointV1,
) -> Result<serde_json::Value> {
    let bytes = checkpoint
        .encode()
        .context("encode complete archipelago checkpoint")?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "reserve new checkpoint {}; existing files are preserved",
                path.display()
            )
        })?;
    file.write_all(&bytes)
        .with_context(|| format!("incomplete checkpoint retained at {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("checkpoint file sync is unconfirmed at {}", path.display()))?;
    #[cfg(unix)]
    fs::File::open(checkpoint_parent(path))?
        .sync_all()
        .context("checkpoint parent-directory sync is unconfirmed")?;
    Ok(serde_json::json!({
        "schema": "scriptbots.archipelago-checkpoint-file.v1", "path": path,
        "bytes": bytes.len(), "blake3": blake3::hash(&bytes).to_hex().to_string(),
        "tick": checkpoint.barrier_tick().0, "epoch": checkpoint.epoch(),
        "island_count": checkpoint.island_metadata().len(),
        "file_synced": true, "host_journal": "committed_volatile",
        "directory_synced": cfg!(unix),
    }))
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
    fn portable_checkpoint_files_preserve_existing_evidence_and_bound_input() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let mut archipelago = Archipelago::new(ArchipelagoConfig {
            islands: vec![IslandSpec::new(
                IslandId(7),
                "saved-island",
                ScriptBotsConfig {
                    world_width: 64,
                    world_height: 64,
                    food_cell_size: 16,
                    closed: true,
                    persistence_interval: 0,
                    ..Default::default()
                },
            )],
            topology: Topology::Custom(Vec::new()),
            barrier_interval: NonZeroU64::new(2).unwrap(),
            master_seed: 17,
            host_options: HostCoreOptions::default(),
            migration: None,
        })?;
        let cut = archipelago.checkpoint_v1()?;
        let path = directory.path().join("source.checkpoint");
        let receipt = write_archipelago_checkpoint(&path, &cut)?;
        let original = fs::read(&path)?;
        let (restored, hash) = read_archipelago_checkpoint(&path)?;
        assert_eq!(restored.encode()?, original);
        assert_eq!(receipt["blake3"], hash);
        assert!(write_archipelago_checkpoint(&path, &cut).is_err());
        assert_eq!(fs::read(&path)?, original);

        let oversized = directory.path().join("oversized.checkpoint");
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&oversized)?
            .set_len(u64::try_from(MAX_ARCHIPELAGO_CHECKPOINT_BYTES)? + 1)?;
        assert!(
            read_archipelago_checkpoint(&oversized)
                .unwrap_err()
                .to_string()
                .contains("exceeds")
        );
        let corrupt = directory.path().join("corrupt.checkpoint");
        let mut changed = original.clone();
        let last = changed.last_mut().unwrap();
        *last ^= 1;
        fs::write(&corrupt, changed)?;
        assert!(read_archipelago_checkpoint(&corrupt).is_err());
        assert!(read_archipelago_checkpoint(directory.path()).is_err());

        // Destination refusal happens before reconstruction or stepping, including input=output.
        let args = ContinueArchipelagoArgs {
            checkpoint: path.clone(),
            output: path.clone(),
            barriers: NonZeroU64::new(1).unwrap(),
            brain: BrainPreset::Mlp,
            threads: NonZeroUsize::new(1).unwrap(),
        };
        assert!(
            continue_archipelago(&args)
                .unwrap_err()
                .to_string()
                .contains("already exists")
        );
        assert_eq!(fs::read(&path)?, original);
        #[cfg(unix)]
        {
            let alias = directory.path().join("alias.checkpoint");
            std::os::unix::fs::symlink(&path, &alias)?;
            assert!(read_archipelago_checkpoint(&alias).is_err());
            assert!(write_archipelago_checkpoint(&alias, &cut).is_err());
            assert_eq!(fs::read(&path)?, original);
        }
        Ok(())
    }

    #[test]
    fn recorded_founders_must_match_the_manifest_before_storage_creation() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let config = ScriptBotsConfig {
            rng_seed: Some(42),
            persistence_interval: 1,
            closed: true,
            world_width: 600,
            world_height: 600,
            ..Default::default()
        };
        for (index, (executed, declared)) in [
            (FounderRecipe::Spread, FounderRecipe::CornerGrid.id()),
            (FounderRecipe::CornerGrid, FounderRecipe::Spread.id()),
            (FounderRecipe::Spread, "unknown-founder-recipe"),
        ]
        .into_iter()
        .enumerate()
        {
            let database = directory.path().join(format!("mismatched-{index}.sqlite"));
            let error = run_recorded_archipelago(
                config.clone(),
                2,
                1,
                BrainPreset::Mlp,
                executed,
                database.to_str().context("test database path")?,
                &RecordedArchipelagoOptions::default(),
                |world| {
                    let mut scenario =
                        crate::ScenarioIdentityV0::caller_seeded("recorded-founders");
                    scenario.population_recipe = format!("{declared};brain=mlp");
                    let manifest = crate::RunManifestV3::from_world_with_provenance(
                        crate::RunIdentityV1::new(
                            scriptbots_runtime::RunId::new(1),
                            0,
                            Some(1),
                            None,
                        ),
                        scenario,
                        world,
                        BuildProvenanceV0::current(),
                    )?;
                    Ok(manifest.to_storage_record()?)
                },
            )
            .expect_err("contradictory founder provenance must not create a run database");
            assert!(error.to_string().contains("founder recipe"), "{error:#}");
            assert!(!database.exists());
        }
        assert!(directory.path().read_dir()?.next().is_none());
        Ok(())
    }

    #[test]
    fn recorded_migration_is_opt_in_with_explicit_runtime_defaults() -> Result<()> {
        let mut options = RecordedArchipelagoOptions::default();
        let (topology, migration) = options.migration_policy(1)?;
        assert_eq!(topology, Topology::Custom(Vec::new()));
        assert!(migration.is_none());
        options.migration_interval = NonZeroU64::new(2);
        assert!(options.migration_policy(1).is_err());
        let (topology, migration) = options.migration_policy(4)?;
        assert_eq!(topology, Topology::Ring);
        assert_eq!(
            migration,
            Some(ArchipelagoMigration {
                interval_ticks: 2,
                emigrants_per_edge: 1,
                selection_rule: EmigrantSelectionRule::Fittest,
            })
        );
        options.migration_interval = None;
        options.migration_emigrants = NonZeroU32::new(1);
        assert!(options.migration_policy(4).is_err());
        Ok(())
    }

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
