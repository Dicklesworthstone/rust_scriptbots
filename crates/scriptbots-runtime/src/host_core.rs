//! Deterministic sole-owner simulation host.

use super::{
    AdmissionSequence, ApplicationFailure, ApplicationState, AppliedCommand, CommandEnvelope,
    CommandId, CommandStatus, ConfigRevision, ControlRevision, DriveReceipt, EventSequence,
    FoodLayerSnapshot, HostAccessError, HostBlocker, HostCommand, HostDriveInterest, HostEvent,
    HostEventKind, HostFault, HostHealth, HostLifecycle, HostPort, HostRevisions, HostSessionId,
    HydrologyLayerSnapshot, HydrologyTileSnapshot, JournalAdmission, JournalBatch, JournalBatchId,
    JournalFailure, JournalPort, JournalReceipt, JournalReceiptState, JournalState, LayerRevision,
    ManualHostDriver, ManualInstant, PlaybackSnapshot, RejectionReason, RenderSnapshot,
    ScientificBoundary, ScientificBoundaryFault, ScientificRevision, ShutdownCommitRequirement,
    SnapshotBuildStats, SnapshotHub, SnapshotLayerRevisions, SnapshotLayers, SnapshotRevision,
    StatusCombinationError, TerrainLayerSnapshot, TerrainTileSnapshot,
};
use scriptbots_core::{
    CharacterizationError, CompletedStepFault, DynamicAgentSnapshot, DynamicWorldSnapshot,
    NullPersistence, PersistenceAdmissionSession, PersistenceSessionError, ScriptBotsConfig, Tick,
    TickSummary, WorldDigestV1, WorldState,
};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
    mem::size_of,
    rc::Rc,
    sync::Arc,
};
use thiserror::Error;

const SPEED_SCALE: u128 = 1_000_000;
const DEFAULT_TICK_PERIOD_NANOS: u64 = 16_666_667;
const DEFAULT_COMMAND_CAPACITY: usize = 32;
const DEFAULT_MAX_AUTOMATIC_STEPS: usize = 8;
const DEFAULT_SNAPSHOT_INTERVAL_TICKS: u64 = 1;
const RECEIPT_POLL_LIMIT: usize = 4_096;
const LIFECYCLE_COMMAND_NAMESPACE: u64 = u64::MAX;

/// Construction options for a synchronous [`HostCore`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostCoreOptions {
    /// Playback state visible before the first manual-drive boundary.
    pub initial_playback: PlaybackSnapshot,
    /// Maximum number of admitted envelopes waiting for ordered application.
    pub command_capacity: usize,
    /// Nominal duration of one automatic scientific step.
    pub tick_period_nanos: u64,
    /// Maximum automatic catch-up work performed by one drive call.
    pub max_automatic_steps_per_drive: usize,
    /// Completed automatic science revisions between render publications.
    ///
    /// Control, lifecycle, health, configuration, and explicit-step changes still publish
    /// immediately. This deterministic stride changes presentation work only.
    pub snapshot_interval_ticks: u64,
}

impl Default for HostCoreOptions {
    fn default() -> Self {
        Self {
            initial_playback: PlaybackSnapshot::default(),
            command_capacity: DEFAULT_COMMAND_CAPACITY,
            tick_period_nanos: DEFAULT_TICK_PERIOD_NANOS,
            max_automatic_steps_per_drive: DEFAULT_MAX_AUTOMATIC_STEPS,
            snapshot_interval_ticks: DEFAULT_SNAPSHOT_INTERVAL_TICKS,
        }
    }
}

/// Failure to construct a sole-owner host without changing the supplied world.
#[derive(Debug, Error)]
pub enum HostCoreBuildError {
    /// One of the deterministic scheduling bounds was zero or invalid.
    #[error("invalid HostCore option: {message}")]
    InvalidOptions {
        /// Actionable option diagnostic.
        message: String,
    },
    /// The world had already surrendered its one lifetime persistence binding.
    #[error(transparent)]
    Persistence(#[from] PersistenceSessionError),
}

/// Same-thread volatile journal used by [`HostCore::new`].
///
/// Admission is immediate and nonblocking. A volatile receipt is queued for the
/// next drive boundary so application and journal state still advance on their
/// independent protocol axes.
#[derive(Debug, Default)]
pub struct VolatileJournal {
    accepted: HashSet<JournalBatchId>,
    batches: Vec<Arc<JournalBatch>>,
    receipts: VecDeque<JournalReceipt>,
}

impl VolatileJournal {
    /// Accepted immutable batches in journal order.
    #[must_use]
    pub fn batches(&self) -> &[Arc<JournalBatch>] {
        &self.batches
    }
}

impl JournalPort for VolatileJournal {
    fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
        let batch_id = batch.id();
        if self.accepted.insert(batch_id) {
            self.batches.push(Arc::clone(batch));
            self.receipts.push_back(JournalReceipt::new(
                batch_id,
                JournalReceiptState::CommittedVolatile,
            ));
        }
        JournalAdmission::Accepted { batch_id }
    }

    fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
        let count = limit.min(self.receipts.len());
        self.receipts.drain(..count).collect()
    }

    fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
        ShutdownCommitRequirement::CommittedVolatile
    }
}

#[derive(Debug, Clone)]
struct AdmittedEnvelope {
    admission: AdmissionSequence,
    envelope: CommandEnvelope,
}

struct SharedHostState {
    session_id: HostSessionId,
    command_capacity: usize,
    next_admission: AdmissionSequence,
    next_event: EventSequence,
    admission_lifecycle: HostLifecycle,
    shutdown_command_id: Option<CommandId>,
    queue: VecDeque<AdmittedEnvelope>,
    statuses: HashMap<CommandId, CommandStatus>,
    last_applied: Option<(AdmissionSequence, CommandId)>,
    events: Vec<HostEvent>,
    visible_tick: Tick,
}

impl SharedHostState {
    fn emit(&mut self, kind: HostEventKind) -> Result<(), HostAccessError> {
        let sequence = self.next_event;
        self.next_event = sequence
            .checked_next()
            .ok_or_else(|| protocol_violation("event sequence exhausted"))?;
        self.events.push(HostEvent {
            sequence,
            tick: self.visible_tick,
            kind,
        });
        Ok(())
    }

    fn store_status(&mut self, status: CommandStatus) -> Result<(), HostAccessError> {
        if matches!(status.application(), ApplicationState::Applied(_))
            && let Some(admission) = status.admission_sequence()
            && self
                .last_applied
                .is_none_or(|(current, _)| admission > current)
        {
            self.last_applied = Some((admission, status.command_id()));
        }
        self.statuses.insert(status.command_id(), status.clone());
        self.emit(HostEventKind::CommandStatusChanged(status))
    }

    fn submit(
        &mut self,
        envelope: CommandEnvelope,
        reserve_lifecycle_slot: bool,
    ) -> Result<CommandStatus, HostAccessError> {
        if let Some(status) = self.statuses.get(&envelope.command_id) {
            return Ok(status.clone());
        }

        if let Err(error) = envelope.command.validate() {
            let status = CommandStatus::rejected(
                envelope.command_id,
                RejectionReason::Validation {
                    message: error.to_string(),
                },
            )
            .map_err(status_violation)?;
            self.store_status(status.clone())?;
            return Ok(status);
        }
        if self.admission_lifecycle != HostLifecycle::Running {
            let status =
                CommandStatus::rejected(envelope.command_id, RejectionReason::HostStopping)
                    .map_err(status_violation)?;
            self.store_status(status.clone())?;
            return Ok(status);
        }

        let closes_gate = matches!(&envelope.command, HostCommand::Shutdown);
        if self.queue.len() >= self.command_capacity && !(reserve_lifecycle_slot && closes_gate) {
            let status = CommandStatus::rejected(
                envelope.command_id,
                RejectionReason::Overloaded {
                    capacity: self.command_capacity,
                },
            )
            .map_err(status_violation)?;
            self.store_status(status.clone())?;
            return Ok(status);
        }

        let admission = self.next_admission;
        self.next_admission = admission
            .checked_next()
            .ok_or_else(|| protocol_violation("admission sequence exhausted"))?;
        let journal = if envelope.command.requires_journal() {
            JournalState::Pending
        } else {
            JournalState::NotRequired
        };
        let status = CommandStatus::try_new(
            envelope.command_id,
            Some(admission),
            ApplicationState::Admitted,
            journal,
        )
        .map_err(status_violation)?;
        if closes_gate {
            self.admission_lifecycle = HostLifecycle::Stopping;
            self.shutdown_command_id = Some(envelope.command_id);
        }
        self.queue.push_back(AdmittedEnvelope {
            admission,
            envelope,
        });
        self.store_status(status.clone())?;
        Ok(status)
    }
}

/// Cloneable same-thread command and observation handle for [`HostCore`].
///
/// The handle shares only bounded ingress and immutable/query DTO state. It
/// never owns, borrows, locks, or exposes the mutable [`WorldState`].
#[derive(Clone)]
pub struct LocalHostPort {
    shared: Rc<RefCell<SharedHostState>>,
    snapshots: SnapshotHub,
}

impl LocalHostPort {
    /// Number of admitted envelopes not yet processed by the owner.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.shared.borrow().queue.len()
    }
}

impl HostPort for LocalHostPort {
    fn session_id(&self) -> HostSessionId {
        self.shared.borrow().session_id
    }

    fn submit(&mut self, envelope: CommandEnvelope) -> Result<CommandStatus, HostAccessError> {
        self.shared.borrow_mut().submit(envelope, false)
    }

    fn command_status(
        &mut self,
        command_id: CommandId,
    ) -> Result<Option<CommandStatus>, HostAccessError> {
        Ok(self.shared.borrow().statuses.get(&command_id).cloned())
    }

    fn snapshot_after(
        &mut self,
        after: Option<SnapshotRevision>,
    ) -> Result<Option<Arc<RenderSnapshot>>, HostAccessError> {
        Ok(self.snapshots.snapshot_after(after))
    }

    fn events_after(
        &mut self,
        cursor: EventSequence,
        limit: usize,
    ) -> Result<Vec<HostEvent>, HostAccessError> {
        Ok(self
            .shared
            .borrow()
            .events
            .iter()
            .filter(|event| event.sequence > cursor)
            .take(limit)
            .cloned()
            .collect())
    }
}

#[derive(Debug, Clone, Copy)]
struct InflightJournal {
    command_id: Option<CommandId>,
    shutdown_requirement: Option<ShutdownCommitRequirement>,
    committed_volatile: bool,
}

#[derive(Clone)]
struct SnapshotLayerCache {
    revisions: SnapshotLayerRevisions,
    terrain: Arc<TerrainLayerSnapshot>,
    food: Arc<FoodLayerSnapshot>,
    hydrology: Option<Arc<HydrologyLayerSnapshot>>,
}

#[derive(Debug, Clone, Copy, Default)]
struct LayerRefreshStats {
    bulk_allocations: usize,
    newly_allocated_capacity_bytes: usize,
}

impl LayerRefreshStats {
    fn add_vector<T>(&mut self, capacity: usize) {
        if capacity != 0 {
            self.bulk_allocations = self.bulk_allocations.saturating_add(1);
            self.newly_allocated_capacity_bytes = self
                .newly_allocated_capacity_bytes
                .saturating_add(capacity.saturating_mul(size_of::<T>()));
        }
    }
}

impl SnapshotLayerCache {
    fn new(world: &WorldState) -> (Self, LayerRefreshStats) {
        let terrain = Arc::new(capture_terrain_layer(world));
        let food = Arc::new(capture_food_layer(world));
        let hydrology = world.hydrology().map(capture_hydrology_layer).map(Arc::new);
        let cache = Self {
            revisions: SnapshotLayerRevisions {
                terrain: LayerRevision::new(1),
                food: LayerRevision::new(1),
                hydrology: LayerRevision::new(u64::from(hydrology.is_some())),
            },
            terrain,
            food,
            hydrology,
        };
        let mut stats = LayerRefreshStats::default();
        cache.add_allocation_stats(&mut stats);
        (cache, stats)
    }

    fn refresh(&mut self, world: &WorldState) -> Result<LayerRefreshStats, HostAccessError> {
        let mut stats = LayerRefreshStats::default();
        if !terrain_layer_matches(&self.terrain, world) {
            let terrain = Arc::new(capture_terrain_layer(world));
            stats.add_vector::<TerrainTileSnapshot>(terrain.tiles.capacity());
            self.terrain = terrain;
            self.revisions.terrain = next_layer_revision(self.revisions.terrain, "terrain")?;
        }
        if !food_layer_matches(&self.food, world) {
            let food = Arc::new(capture_food_layer(world));
            stats.add_vector::<f32>(food.cells.capacity());
            self.food = food;
            self.revisions.food = next_layer_revision(self.revisions.food, "food")?;
        }
        match (&self.hydrology, world.hydrology()) {
            (None, None) => {}
            (Some(current), Some(world_hydrology))
                if hydrology_layer_matches(current, world_hydrology) => {}
            (_, next) => {
                self.hydrology = next.map(capture_hydrology_layer).map(Arc::new);
                if let Some(hydrology) = &self.hydrology {
                    add_hydrology_allocation_stats(hydrology, &mut stats);
                }
                self.revisions.hydrology =
                    next_layer_revision(self.revisions.hydrology, "hydrology")?;
            }
        }
        Ok(stats)
    }

    fn snapshot(&self) -> SnapshotLayers {
        SnapshotLayers {
            revisions: self.revisions,
            terrain: Arc::clone(&self.terrain),
            food: Arc::clone(&self.food),
            hydrology: self.hydrology.as_ref().map(Arc::clone),
        }
    }

    fn total_capacity_bytes(&self) -> usize {
        let terrain = self
            .terrain
            .tiles
            .capacity()
            .saturating_mul(size_of::<TerrainTileSnapshot>());
        let food = self.food.cells.capacity().saturating_mul(size_of::<f32>());
        let hydrology = self
            .hydrology
            .as_deref()
            .map_or(0, hydrology_capacity_bytes);
        terrain.saturating_add(food).saturating_add(hydrology)
    }

    fn add_allocation_stats(&self, stats: &mut LayerRefreshStats) {
        stats.add_vector::<TerrainTileSnapshot>(self.terrain.tiles.capacity());
        stats.add_vector::<f32>(self.food.cells.capacity());
        if let Some(hydrology) = &self.hydrology {
            add_hydrology_allocation_stats(hydrology, stats);
        }
    }
}

fn next_layer_revision(
    revision: LayerRevision,
    layer: &'static str,
) -> Result<LayerRevision, HostAccessError> {
    revision
        .checked_next()
        .ok_or_else(|| protocol_violation(format!("{layer} layer revision exhausted")))
}

fn capture_terrain_layer(world: &WorldState) -> TerrainLayerSnapshot {
    let terrain = world.terrain();
    let mut tiles = Vec::with_capacity(terrain.tiles().len());
    tiles.extend(terrain.tiles().iter().map(|tile| TerrainTileSnapshot {
        kind: tile.kind,
        elevation: tile.elevation,
        moisture: tile.moisture,
        accent: tile.accent,
        fertility_bias: tile.fertility_bias,
        temperature_bias: tile.temperature_bias,
        palette_index: tile.palette_index,
    }));
    TerrainLayerSnapshot {
        width: terrain.width(),
        height: terrain.height(),
        cell_size: terrain.cell_size(),
        tiles,
    }
}

fn capture_food_layer(world: &WorldState) -> FoodLayerSnapshot {
    let food = world.food();
    FoodLayerSnapshot {
        width: food.width(),
        height: food.height(),
        cells: food.cells().to_vec(),
    }
}

fn capture_hydrology_layer(hydrology: &scriptbots_core::HydrologyState) -> HydrologyLayerSnapshot {
    let tiles = hydrology.tiles();
    let field = hydrology.field();
    HydrologyLayerSnapshot {
        width: hydrology.width(),
        height: hydrology.height(),
        tiles: tiles
            .tiles()
            .iter()
            .map(|tile| HydrologyTileSnapshot {
                permeability: tile.permeability,
                runoff_bias: tile.runoff_bias,
                basin_rank: tile.basin_rank,
                channel_priority: tile.channel_priority,
                swim_cost: tile.swim_cost,
            })
            .collect(),
        flow_directions: field.flow_directions().to_vec(),
        accumulation: field.accumulation().to_vec(),
        spill_elevation: field.spill_elevation().to_vec(),
        basin_ids: field.basin_ids().to_vec(),
        water_depth: hydrology.water_depth().to_vec(),
    }
}

fn terrain_layer_matches(snapshot: &TerrainLayerSnapshot, world: &WorldState) -> bool {
    let terrain = world.terrain();
    snapshot.width == terrain.width()
        && snapshot.height == terrain.height()
        && snapshot.cell_size == terrain.cell_size()
        && snapshot.tiles.len() == terrain.tiles().len()
        && snapshot
            .tiles
            .iter()
            .zip(terrain.tiles())
            .all(|(snapshot, tile)| {
                snapshot.kind == tile.kind
                    && same_f32(snapshot.elevation, tile.elevation)
                    && same_f32(snapshot.moisture, tile.moisture)
                    && same_f32(snapshot.accent, tile.accent)
                    && same_f32(snapshot.fertility_bias, tile.fertility_bias)
                    && same_f32(snapshot.temperature_bias, tile.temperature_bias)
                    && snapshot.palette_index == tile.palette_index
            })
}

fn food_layer_matches(snapshot: &FoodLayerSnapshot, world: &WorldState) -> bool {
    let food = world.food();
    snapshot.width == food.width()
        && snapshot.height == food.height()
        && same_f32_slice(&snapshot.cells, food.cells())
}

fn hydrology_layer_matches(
    snapshot: &HydrologyLayerSnapshot,
    hydrology: &scriptbots_core::HydrologyState,
) -> bool {
    let tiles = hydrology.tiles();
    let field = hydrology.field();
    snapshot.width == hydrology.width()
        && snapshot.height == hydrology.height()
        && snapshot.tiles.len() == tiles.tiles().len()
        && snapshot
            .tiles
            .iter()
            .zip(tiles.tiles())
            .all(|(snapshot, tile)| {
                same_f32(snapshot.permeability, tile.permeability)
                    && same_f32(snapshot.runoff_bias, tile.runoff_bias)
                    && same_f32(snapshot.basin_rank, tile.basin_rank)
                    && same_f32(snapshot.channel_priority, tile.channel_priority)
                    && same_f32(snapshot.swim_cost, tile.swim_cost)
            })
        && snapshot.flow_directions == field.flow_directions()
        && same_f32_slice(&snapshot.accumulation, field.accumulation())
        && same_f32_slice(&snapshot.spill_elevation, field.spill_elevation())
        && snapshot.basin_ids == field.basin_ids()
        && same_f32_slice(&snapshot.water_depth, hydrology.water_depth())
}

fn same_f32(left: f32, right: f32) -> bool {
    left.to_bits() == right.to_bits()
}

fn same_f32_slice(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| same_f32(*left, *right))
}

fn hydrology_capacity_bytes(snapshot: &HydrologyLayerSnapshot) -> usize {
    snapshot
        .tiles
        .capacity()
        .saturating_mul(size_of::<HydrologyTileSnapshot>())
        .saturating_add(
            snapshot
                .flow_directions
                .capacity()
                .saturating_mul(size_of::<scriptbots_core::HydrologyFlowDirection>()),
        )
        .saturating_add(
            snapshot
                .accumulation
                .capacity()
                .saturating_mul(size_of::<f32>()),
        )
        .saturating_add(
            snapshot
                .spill_elevation
                .capacity()
                .saturating_mul(size_of::<f32>()),
        )
        .saturating_add(
            snapshot
                .basin_ids
                .capacity()
                .saturating_mul(size_of::<u32>()),
        )
        .saturating_add(
            snapshot
                .water_depth
                .capacity()
                .saturating_mul(size_of::<f32>()),
        )
}

fn add_hydrology_allocation_stats(
    snapshot: &HydrologyLayerSnapshot,
    stats: &mut LayerRefreshStats,
) {
    stats.add_vector::<HydrologyTileSnapshot>(snapshot.tiles.capacity());
    stats
        .add_vector::<scriptbots_core::HydrologyFlowDirection>(snapshot.flow_directions.capacity());
    stats.add_vector::<f32>(snapshot.accumulation.capacity());
    stats.add_vector::<f32>(snapshot.spill_elevation.capacity());
    stats.add_vector::<u32>(snapshot.basin_ids.capacity());
    stats.add_vector::<f32>(snapshot.water_depth.capacity());
}

fn snapshot_build_stats(
    world: &DynamicWorldSnapshot,
    layers: &SnapshotLayerCache,
    refresh: LayerRefreshStats,
) -> SnapshotBuildStats {
    let dynamic_agent_bytes = world
        .agents
        .capacity()
        .saturating_mul(size_of::<DynamicAgentSnapshot>());
    let layer_bytes = layers.total_capacity_bytes();
    SnapshotBuildStats {
        dynamic_agent_count: world.agents.len(),
        bulk_allocations: refresh
            .bulk_allocations
            .saturating_add(usize::from(world.agents.capacity() != 0)),
        newly_allocated_capacity_bytes: refresh
            .newly_allocated_capacity_bytes
            .saturating_add(dynamic_agent_bytes),
        reused_layer_capacity_bytes: layer_bytes
            .saturating_sub(refresh.newly_allocated_capacity_bytes),
        total_payload_capacity_bytes: dynamic_agent_bytes.saturating_add(layer_bytes),
    }
}

/// Pure synchronous authority for command order and scientific time.
///
/// `HostCore` owns its world and persistence-admission session by value. It
/// spawns no threads and contains no platform runtime, renderer, server,
/// database connection, or mutable-world lock.
pub struct HostCore {
    session_id: HostSessionId,
    world: WorldState,
    persistence: PersistenceAdmissionSession,
    journal: Box<dyn JournalPort>,
    shared: Rc<RefCell<SharedHostState>>,
    snapshots: SnapshotHub,
    snapshot_layers: SnapshotLayerCache,
    options: HostCoreOptions,
    playback: PlaybackSnapshot,
    lifecycle: HostLifecycle,
    health: HostHealth,
    revisions: HostRevisions,
    last_now: Option<ManualInstant>,
    cadence_credit: u128,
    next_snapshot: SnapshotRevision,
    last_published_scientific: ScientificRevision,
    latest_completed_summary: Option<TickSummary>,
    next_journal_sequence: u64,
    next_lifecycle_command_sequence: u64,
    active_command: Option<AdmittedEnvelope>,
    active_journal_batch: Option<Arc<JournalBatch>>,
    indeterminate_journal_batch: Option<Arc<JournalBatch>>,
    retained_journal: Option<Arc<JournalBatch>>,
    retained_blocker: Option<HostBlocker>,
    inflight_journal: HashMap<JournalBatchId, InflightJournal>,
    shutdown_receipt: Option<(JournalBatchId, ShutdownCommitRequirement)>,
    failed_journal_batches: HashSet<JournalBatchId>,
    latched_fault: Option<HostFault>,
}

impl HostCore {
    /// Construct a host with a nonblocking in-memory volatile journal.
    pub fn new(
        session_id: HostSessionId,
        world: WorldState,
        options: HostCoreOptions,
    ) -> Result<Self, HostCoreBuildError> {
        Self::with_journal(
            session_id,
            world,
            options,
            Box::<VolatileJournal>::default(),
        )
    }

    /// Construct a host with an injected runtime-neutral journal adapter.
    pub fn with_journal(
        session_id: HostSessionId,
        world: WorldState,
        options: HostCoreOptions,
        journal: Box<dyn JournalPort>,
    ) -> Result<Self, HostCoreBuildError> {
        validate_options(options)?;
        let persistence = world.bind_persistence(Box::new(NullPersistence))?;
        let revisions = HostRevisions {
            control: ControlRevision::new(0),
            scientific: ScientificRevision::new(world.tick().0),
            config: ConfigRevision::new(world.config_revision()),
        };
        let playback = options.initial_playback;
        let lifecycle = HostLifecycle::Running;
        let health = HostHealth::Healthy;
        let (snapshot_layers, layer_refresh) = SnapshotLayerCache::new(&world);
        let dynamic_world = DynamicWorldSnapshot::from_world(&world);
        let build = snapshot_build_stats(&dynamic_world, &snapshot_layers, layer_refresh);
        let latest_completed_summary = world
            .history()
            .next_back()
            .filter(|summary| summary.tick == world.tick())
            .cloned();
        let initial_snapshot = Arc::new(RenderSnapshot {
            session_id,
            revision: SnapshotRevision::new(1),
            revisions,
            playback,
            lifecycle,
            health: health.clone(),
            command_queue_depth: 0,
            last_applied_command: None,
            completed_summary: latest_completed_summary.clone(),
            layers: snapshot_layers.snapshot(),
            build,
            world: dynamic_world,
        });
        let snapshots = SnapshotHub::new(initial_snapshot);
        let shared = Rc::new(RefCell::new(SharedHostState {
            session_id,
            command_capacity: options.command_capacity,
            next_admission: AdmissionSequence::new(1),
            next_event: EventSequence::new(1),
            admission_lifecycle: HostLifecycle::Running,
            shutdown_command_id: None,
            queue: VecDeque::with_capacity(options.command_capacity),
            statuses: HashMap::new(),
            last_applied: None,
            events: Vec::new(),
            visible_tick: world.tick(),
        }));
        Ok(Self {
            session_id,
            world,
            persistence,
            journal,
            shared,
            snapshots,
            snapshot_layers,
            options,
            playback,
            lifecycle,
            health,
            revisions,
            last_now: None,
            cadence_credit: 0,
            next_snapshot: SnapshotRevision::new(2),
            last_published_scientific: revisions.scientific,
            latest_completed_summary,
            next_journal_sequence: 1,
            next_lifecycle_command_sequence: session_id.get(),
            active_command: None,
            active_journal_batch: None,
            indeterminate_journal_batch: None,
            retained_journal: None,
            retained_blocker: None,
            inflight_journal: HashMap::new(),
            shutdown_receipt: None,
            failed_journal_batches: HashSet::new(),
            latched_fault: None,
        })
    }

    /// Create another same-thread handle to the bounded host port.
    #[must_use]
    pub fn local_port(&self) -> LocalHostPort {
        LocalHostPort {
            shared: Rc::clone(&self.shared),
            snapshots: self.snapshots.clone(),
        }
    }

    /// Clone the thread-safe latest-value snapshot read handle.
    ///
    /// The returned hub owns no command sender and cannot keep native ingress alive.
    #[must_use]
    pub fn snapshot_hub(&self) -> SnapshotHub {
        self.snapshots.clone()
    }

    /// Admit or reuse the host-owned ordered shutdown command.
    ///
    /// Native cancellation uses one reserved lifecycle slot after every
    /// command already admitted through [`HostPort`]. The method closes normal
    /// ingress immediately, never bypasses that existing total order, and
    /// returns the same command status on every later call.
    pub fn request_shutdown(&mut self) -> Result<CommandStatus, HostAccessError> {
        if let Some(command_id) = self.shared.borrow().shutdown_command_id {
            return self
                .shared
                .borrow()
                .statuses
                .get(&command_id)
                .cloned()
                .ok_or_else(|| protocol_violation("shutdown command status is missing"));
        }

        let command_id = loop {
            let sequence = self.next_lifecycle_command_sequence;
            self.next_lifecycle_command_sequence = sequence
                .checked_add(1)
                .ok_or_else(|| protocol_violation("lifecycle command sequence exhausted"))?;
            let candidate = CommandId::from_client_sequence(LIFECYCLE_COMMAND_NAMESPACE, sequence);
            if !self.shared.borrow().statuses.contains_key(&candidate) {
                break candidate;
            }
        };
        self.shared.borrow_mut().submit(
            CommandEnvelope::new(command_id, HostCommand::Shutdown),
            true,
        )
    }

    /// Stable identity of the admitted ordered shutdown, when one exists.
    #[must_use]
    pub fn shutdown_command_id(&self) -> Option<CommandId> {
        self.shared.borrow().shutdown_command_id
    }

    /// Current two-axis status of the admitted ordered shutdown, when present.
    #[must_use]
    pub fn shutdown_command_status(&self) -> Option<CommandStatus> {
        let shared = self.shared.borrow();
        shared
            .shutdown_command_id
            .and_then(|command_id| shared.statuses.get(&command_id).cloned())
    }

    /// Latest immutable host publication.
    #[must_use]
    pub fn latest_snapshot(&self) -> Arc<RenderSnapshot> {
        self.snapshots.latest()
    }

    /// Current queryable health.
    #[must_use]
    pub const fn health(&self) -> &HostHealth {
        &self.health
    }

    /// Current completed scientific tick.
    #[must_use]
    pub const fn world_tick(&self) -> Tick {
        self.world.tick()
    }

    /// Canonical full scientific digest without exposing the mutable world.
    pub fn scientific_digest_v1(&self) -> Result<WorldDigestV1, CharacterizationError> {
        self.world.world_digest_v1()
    }

    /// Nominal automatic cadence configured for this host.
    #[must_use]
    pub const fn tick_period_nanos(&self) -> u64 {
        self.options.tick_period_nanos
    }

    /// Maximum automatic science work one drive boundary may perform.
    #[must_use]
    pub const fn max_automatic_steps_per_drive(&self) -> usize {
        self.options.max_automatic_steps_per_drive
    }

    /// Current scheduling interest for a platform-owned driver.
    #[must_use]
    pub fn drive_interest(&self) -> HostDriveInterest {
        if self.lifecycle == HostLifecycle::Stopped {
            return HostDriveInterest::Terminated;
        }
        if self.health.fault().is_some() {
            return HostDriveInterest::Faulted;
        }
        if self.retained_journal.is_some() {
            return HostDriveInterest::WakeOnly;
        }
        if !self.shared.borrow().queue.is_empty() {
            return HostDriveInterest::ReadyNow;
        }
        let journal_drain_required = self.inflight_journal.values().any(|entry| {
            !entry.committed_volatile
                || matches!(
                    entry.shutdown_requirement,
                    Some(ShutdownCommitRequirement::Durable)
                )
        });
        if self.lifecycle == HostLifecycle::Stopping
            || (journal_drain_required
                && (self.playback.paused || speed_units(self.playback.speed_multiplier) == 0))
        {
            return HostDriveInterest::Draining;
        }
        if self.playback.paused || speed_units(self.playback.speed_multiplier) == 0 {
            HostDriveInterest::WakeOnly
        } else {
            HostDriveInterest::Deadline
        }
    }

    /// Exact immutable batch retained after non-admission.
    #[must_use]
    pub fn pending_journal_batch(&self) -> Option<Arc<JournalBatch>> {
        self.retained_journal.as_ref().map(Arc::clone)
    }

    /// Exact command whose owner boundary did not return normally.
    ///
    /// This is diagnostic evidence only. A returned envelope has already
    /// entered the host's admission order and must never be resubmitted.
    #[must_use]
    pub fn panicked_command(&self) -> Option<&CommandEnvelope> {
        self.active_command
            .as_ref()
            .map(|admitted| &admitted.envelope)
    }

    /// Exact journal batch whose adapter handoff panicked after application.
    ///
    /// The adapter may or may not have accepted this batch before unwinding,
    /// so this evidence is deliberately separate from retryable
    /// [`Self::pending_journal_batch`] state.
    #[must_use]
    pub fn indeterminate_journal_batch(&self) -> Option<Arc<JournalBatch>> {
        self.indeterminate_journal_batch.as_ref().map(Arc::clone)
    }

    /// Seal admissions and publish exact fault evidence after a driver unwind.
    #[cfg_attr(
        not(all(feature = "native-asupersync", not(target_arch = "wasm32"))),
        allow(
            dead_code,
            reason = "the default/browser host retains panic evidence for its native adapter"
        )
    )]
    pub(crate) fn record_panicked_boundary(
        &mut self,
        message: &str,
    ) -> Result<(), HostAccessError> {
        if self.indeterminate_journal_batch.is_none()
            && let Some(batch) = self.active_journal_batch.take()
        {
            if self
                .retained_journal
                .as_ref()
                .is_some_and(|retained| retained.id() == batch.id())
            {
                self.retained_journal = None;
                self.retained_blocker = None;
            }
            self.indeterminate_journal_batch = Some(batch);
        }
        if self.lifecycle == HostLifecycle::Running {
            self.lifecycle = HostLifecycle::Stopping;
            let mut shared = self.shared.borrow_mut();
            shared.admission_lifecycle = HostLifecycle::Stopping;
            shared.emit(HostEventKind::LifecycleChanged(HostLifecycle::Stopping))?;
        }
        self.latched_fault = Some(HostFault::Protocol {
            code: "native_lifecycle_panic".to_owned(),
            message: message.to_owned(),
        });
        let changed = self.synchronize_health()?;
        if changed {
            self.publish_snapshot()?;
        }
        Ok(())
    }

    /// Explicitly retry the exact retained journal allocation once.
    ///
    /// A successful retry seals the corresponding core persistence boundary,
    /// but its command's `JournalState` still advances only through a later
    /// receipt polled by [`ManualHostDriver::drive`].
    pub fn retry_retained_journal(&mut self) -> Result<Option<JournalAdmission>, HostAccessError> {
        let Some(batch) = self.retained_journal.as_ref().map(Arc::clone) else {
            return Ok(None);
        };
        self.active_journal_batch = Some(Arc::clone(&batch));
        let admission = self.journal.try_admit(&batch);
        let shutdown = batch
            .command()
            .is_some_and(|command| matches!(&command.command, HostCommand::Shutdown));
        let result = self.finish_journal_admission(&batch, admission, shutdown, true);
        if result.is_ok() {
            self.active_journal_batch = None;
        }
        result?;
        let changed = self.synchronize_health()?;
        if changed {
            self.publish_snapshot()?;
        }
        Ok(Some(admission))
    }

    fn retain_identity_violation(
        &mut self,
        batch: &Arc<JournalBatch>,
        admission: JournalAdmission,
    ) -> Result<bool, HostAccessError> {
        if admission.batch_id() == batch.id() {
            return Ok(false);
        }
        self.retained_journal = Some(Arc::clone(batch));
        self.retained_blocker = None;
        if let Some(command_id) = batch.command_id() {
            self.update_command_journal(
                command_id,
                JournalState::Failed(JournalFailure {
                    code: "journal_identity_mismatch".to_owned(),
                    message: "journal response echoed a different batch identity".to_owned(),
                }),
            )?;
        }
        self.failed_journal_batches.insert(batch.id());
        self.latched_fault = Some(HostFault::Protocol {
            code: "journal_identity_mismatch".to_owned(),
            message: format!(
                "journal response for {:?} echoed {:?}",
                batch.id(),
                admission.batch_id()
            ),
        });
        self.synchronize_health()?;
        Ok(true)
    }

    fn seal_core_persistence(&mut self) -> Result<(), HostAccessError> {
        self.persistence
            .admit_pending(&mut self.world)
            .map(|_| ())
            .map_err(|error| {
                protocol_violation(format!("could not seal core persistence: {error}"))
            })
    }

    fn fail_closed_batch(&mut self, batch: &Arc<JournalBatch>) -> Result<(), HostAccessError> {
        self.retained_journal = Some(Arc::clone(batch));
        self.retained_blocker = Some(HostBlocker::JournalClosed {
            batch_id: batch.id(),
        });
        let failure = JournalFailure {
            code: "journal_closed".to_owned(),
            message: "journal admission gate is permanently closed".to_owned(),
        };
        if let Some(command_id) = batch.command_id() {
            self.update_command_journal(command_id, JournalState::Failed(failure.clone()))?;
        }
        self.failed_journal_batches.insert(batch.id());
        self.latched_fault = Some(HostFault::Journal {
            batch_id: batch.id(),
            failure,
        });
        self.synchronize_health().map(|_| ())
    }

    fn poll_journal_receipts(&mut self) -> Result<bool, HostAccessError> {
        let mut changed = false;
        for receipt in self.journal.poll_receipts(RECEIPT_POLL_LIMIT) {
            let batch_id = receipt.batch_id();
            let Some(inflight) = self.inflight_journal.get(&batch_id).copied() else {
                self.latch_protocol_fault(
                    "unknown_journal_receipt",
                    format!("journal acknowledged unknown batch {batch_id:?}"),
                )?;
                changed = true;
                continue;
            };

            let journal_state = match receipt.state() {
                JournalReceiptState::CommittedVolatile => JournalState::CommittedVolatile,
                JournalReceiptState::Durable => JournalState::Durable,
                JournalReceiptState::Failed(failure) => JournalState::Failed(failure.clone()),
            };
            if let Some(command_id) = inflight.command_id {
                changed |= self.update_command_journal(command_id, journal_state)?;
            }

            match receipt.state() {
                JournalReceiptState::CommittedVolatile => {
                    if let Some(entry) = self.inflight_journal.get_mut(&batch_id) {
                        entry.committed_volatile = true;
                    }
                    if let Some(requirement @ ShutdownCommitRequirement::CommittedVolatile) =
                        inflight.shutdown_requirement
                    {
                        self.shutdown_receipt = Some((batch_id, requirement));
                    }
                }
                JournalReceiptState::Durable => {
                    self.inflight_journal.remove(&batch_id);
                    if let Some(requirement) = inflight.shutdown_requirement {
                        self.shutdown_receipt = Some((batch_id, requirement));
                    }
                }
                JournalReceiptState::Failed(failure) => {
                    self.inflight_journal.remove(&batch_id);
                    self.failed_journal_batches.insert(batch_id);
                    self.latched_fault = Some(HostFault::Journal {
                        batch_id,
                        failure: failure.clone(),
                    });
                    changed = true;
                }
            }
            changed |= self.try_finish_shutdown()?;
        }
        changed |= self.synchronize_health()?;
        Ok(changed)
    }

    fn update_command_journal(
        &self,
        command_id: CommandId,
        journal: JournalState,
    ) -> Result<bool, HostAccessError> {
        let current = self
            .shared
            .borrow()
            .statuses
            .get(&command_id)
            .cloned()
            .ok_or_else(|| protocol_violation("journal receipt command status is missing"))?;
        if current.journal() == &journal
            || matches!(
                current.journal(),
                JournalState::Durable | JournalState::Failed(_)
            )
        {
            return Ok(false);
        }
        let status = CommandStatus::try_new(
            command_id,
            current.admission_sequence(),
            current.application().clone(),
            journal,
        )
        .map_err(status_violation)?;
        self.shared.borrow_mut().store_status(status)?;
        Ok(true)
    }

    fn finish_shutdown(&mut self) -> Result<bool, HostAccessError> {
        if self.lifecycle == HostLifecycle::Stopped {
            return Ok(false);
        }
        self.lifecycle = HostLifecycle::Stopped;
        let mut shared = self.shared.borrow_mut();
        shared.admission_lifecycle = HostLifecycle::Stopped;
        shared.emit(HostEventKind::LifecycleChanged(HostLifecycle::Stopped))?;
        Ok(true)
    }

    fn try_finish_shutdown(&mut self) -> Result<bool, HostAccessError> {
        let Some((shutdown_id, requirement)) = self.shutdown_receipt else {
            return Ok(false);
        };
        let earlier_work_failed = self.failed_journal_batches.iter().any(|batch_id| {
            batch_id.session_id() == shutdown_id.session_id()
                && batch_id.sequence() <= shutdown_id.sequence()
        });
        if earlier_work_failed {
            return Ok(false);
        }
        let earlier_work_pending = self.inflight_journal.iter().any(|(batch_id, inflight)| {
            batch_id.session_id() == shutdown_id.session_id()
                && batch_id.sequence() <= shutdown_id.sequence()
                && match requirement {
                    ShutdownCommitRequirement::CommittedVolatile => !inflight.committed_volatile,
                    ShutdownCommitRequirement::Durable => true,
                }
        });
        if earlier_work_pending {
            Ok(false)
        } else {
            self.finish_shutdown()
        }
    }

    fn latch_protocol_fault(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), HostAccessError> {
        self.latched_fault = Some(HostFault::Protocol {
            code: code.into(),
            message: message.into(),
        });
        self.synchronize_health().map(|_| ())
    }

    fn synchronize_health(&mut self) -> Result<bool, HostAccessError> {
        let next = if let Some(fault) = &self.latched_fault {
            HostHealth::Faulted(fault.clone())
        } else if let Some(blocker) = self.retained_blocker {
            HostHealth::Blocked(blocker)
        } else {
            match self.lifecycle {
                HostLifecycle::Running => HostHealth::Healthy,
                HostLifecycle::Stopping => HostHealth::Blocked(HostBlocker::LifecycleStopping),
                HostLifecycle::Stopped => HostHealth::Blocked(HostBlocker::LifecycleStopped),
            }
        };
        if next == self.health {
            return Ok(false);
        }
        self.health = next.clone();
        self.shared
            .borrow_mut()
            .emit(HostEventKind::HealthChanged(next))?;
        Ok(true)
    }

    const fn current_blocker(&self) -> Option<HostBlocker> {
        if let Some(blocker) = self.retained_blocker {
            return Some(blocker);
        }
        if let Some(blocker) = self.health.blocker() {
            return Some(blocker);
        }
        if self.health.fault().is_some() {
            return Some(HostBlocker::ScientificFault);
        }
        match self.lifecycle {
            HostLifecycle::Stopping => Some(HostBlocker::LifecycleStopping),
            HostLifecycle::Stopped => Some(HostBlocker::LifecycleStopped),
            HostLifecycle::Running if self.playback.paused => Some(HostBlocker::PlaybackPaused),
            HostLifecycle::Running => None,
        }
    }

    fn publish_snapshot(&mut self) -> Result<(), HostAccessError> {
        let revision = self.next_snapshot;
        let following_revision = revision
            .checked_next()
            .ok_or_else(|| protocol_violation("snapshot revision exhausted"))?;
        let mut layers = self.snapshot_layers.clone();
        let refresh = layers.refresh(&self.world)?;
        let dynamic_world = DynamicWorldSnapshot::from_world(&self.world);
        let build = snapshot_build_stats(&dynamic_world, &layers, refresh);
        let (command_queue_depth, last_applied_command) = {
            let shared = self.shared.borrow();
            (
                shared.queue.len(),
                shared.last_applied.map(|(_, command_id)| command_id),
            )
        };
        let snapshot = Arc::new(RenderSnapshot {
            session_id: self.session_id,
            revision,
            revisions: self.revisions,
            playback: self.playback,
            lifecycle: self.lifecycle,
            health: self.health.clone(),
            command_queue_depth,
            last_applied_command,
            completed_summary: self.latest_completed_summary.clone(),
            layers: layers.snapshot(),
            build,
            world: dynamic_world,
        });
        self.snapshots.publish(snapshot)?;
        self.snapshot_layers = layers;
        self.next_snapshot = following_revision;
        self.last_published_scientific = self.revisions.scientific;
        self.shared.borrow_mut().visible_tick = self.world.tick();
        Ok(())
    }

    fn pop_command(&self) -> Option<AdmittedEnvelope> {
        self.shared.borrow_mut().queue.pop_front()
    }

    fn complete_status(&self, status: CommandStatus) -> Result<(), HostAccessError> {
        let mut shared = self.shared.borrow_mut();
        shared.visible_tick = self.world.tick();
        shared.store_status(status)
    }

    fn apply_command(
        &mut self,
        admitted: AdmittedEnvelope,
    ) -> Result<ApplyResult, HostAccessError> {
        let AdmittedEnvelope {
            admission,
            envelope,
        } = admitted;
        if let Some(expected) = envelope.expected_control_revision
            && expected != self.revisions.control
        {
            if matches!(&envelope.command, HostCommand::Shutdown) {
                let mut shared = self.shared.borrow_mut();
                shared.admission_lifecycle = HostLifecycle::Running;
                shared.shutdown_command_id = None;
            }
            let status = CommandStatus::try_new(
                envelope.command_id,
                Some(admission),
                ApplicationState::Rejected(RejectionReason::ControlRevisionConflict {
                    expected,
                    actual: self.revisions.control,
                }),
                JournalState::NotRequired,
            )
            .map_err(status_violation)?;
            self.complete_status(status)?;
            return Ok(ApplyResult::completed(false));
        }

        if self.latched_fault.is_some()
            && matches!(
                &envelope.command,
                HostCommand::Step | HostCommand::UpdateConfig(_)
            )
        {
            self.complete_failed(
                envelope.command_id,
                admission,
                "science_blocked",
                "host science is stopped by a latched fault".to_owned(),
            )?;
            return Ok(ApplyResult::completed(false));
        }

        let next_control = self
            .revisions
            .control
            .checked_next()
            .ok_or_else(|| protocol_violation("control revision exhausted"))?;
        let retry_envelope = envelope.clone();
        match envelope.command {
            HostCommand::Pause => {
                self.playback.paused = true;
                self.revisions.control = next_control;
                self.complete_applied(retry_envelope.command_id, admission, false)?;
                Ok(ApplyResult::completed(false))
            }
            HostCommand::Resume => {
                self.playback.paused = false;
                self.revisions.control = next_control;
                self.complete_applied(retry_envelope.command_id, admission, false)?;
                Ok(ApplyResult::completed(false))
            }
            HostCommand::SetSpeed(speed) => {
                self.playback.speed_multiplier = speed;
                self.revisions.control = next_control;
                self.complete_applied(retry_envelope.command_id, admission, false)?;
                Ok(ApplyResult::completed(false))
            }
            HostCommand::UpdateSelection(update) => {
                self.world.apply_selection_update(update);
                self.revisions.control = next_control;
                self.complete_applied(retry_envelope.command_id, admission, false)?;
                Ok(ApplyResult::completed(false))
            }
            HostCommand::UpdateConfig(config) => {
                self.apply_config_command(admission, retry_envelope, config, next_control)
            }
            HostCommand::Step => self.apply_step_command(admission, retry_envelope, next_control),
            HostCommand::Shutdown => {
                self.apply_shutdown_command(admission, retry_envelope, next_control)
            }
        }
    }

    fn apply_config_command(
        &mut self,
        admission: AdmissionSequence,
        envelope: CommandEnvelope,
        config: Box<ScriptBotsConfig>,
        next_control: ControlRevision,
    ) -> Result<ApplyResult, HostAccessError> {
        if let Err(error) = self.world.apply_config_update(*config) {
            self.complete_failed(
                envelope.command_id,
                admission,
                "config_application",
                error.to_string(),
            )?;
            return Ok(ApplyResult::completed(false));
        }
        self.revisions.control = next_control;
        self.revisions.config = ConfigRevision::new(self.world.config_revision());
        let applied = self.applied_boundary();
        self.complete_applied_with(envelope.command_id, admission, applied, true)?;
        let blocked = self.offer_journal(envelope, applied, None, None)?;
        Ok(ApplyResult::completed(blocked))
    }

    fn apply_step_command(
        &mut self,
        admission: AdmissionSequence,
        envelope: CommandEnvelope,
        next_control: ControlRevision,
    ) -> Result<ApplyResult, HostAccessError> {
        let next_scientific = self
            .revisions
            .scientific
            .checked_next()
            .ok_or_else(|| protocol_violation("scientific revision exhausted"))?;
        let completion = match self.persistence.step_outcome(&mut self.world) {
            Ok(completion) => completion,
            Err(error) => {
                self.complete_failed(
                    envelope.command_id,
                    admission,
                    "world_step",
                    error.to_string(),
                )?;
                self.latched_fault = Some(HostFault::Scientific {
                    tick: self.world.tick(),
                    code: "world_step".to_owned(),
                    message: error.to_string(),
                });
                self.synchronize_health()?;
                return Ok(ApplyResult::completed(false));
            }
        };
        self.playback.paused = true;
        self.cadence_credit = 0;
        let scriptbots_core::StepCompletion {
            outcome,
            fault: completed_fault,
        } = completion;
        let scriptbots_core::StepOutcome {
            events,
            summary,
            births,
            deaths,
            combat,
            config_revision,
            resource_tick,
            persistence,
        } = outcome;
        self.revisions.control = next_control;
        self.revisions.scientific = next_scientific;
        self.revisions.config = ConfigRevision::new(config_revision);
        self.latest_completed_summary = Some(summary.clone());
        let tick = summary.tick;
        let completed_fault = completed_fault.as_ref().map(completed_fault_record);
        let mut scientific = ScientificBoundary::new(
            events,
            summary,
            births,
            deaths,
            combat,
            config_revision,
            resource_tick,
        );
        if let Some(fault) = completed_fault.clone() {
            scientific = scientific.with_fault(fault);
        }
        let scientific = Arc::new(scientific);
        let persistence = persistence.into_batch();
        let applied = AppliedCommand {
            tick,
            revisions: self.revisions,
        };
        self.complete_applied_with(envelope.command_id, admission, applied, true)?;
        let blocked = self.offer_journal(envelope, applied, Some(scientific), persistence)?;
        if let Some(fault) = completed_fault {
            self.latch_completed_step_fault(tick, &fault)?;
        }
        Ok(ApplyResult::completed(blocked).with_science())
    }

    fn apply_shutdown_command(
        &mut self,
        admission: AdmissionSequence,
        envelope: CommandEnvelope,
        next_control: ControlRevision,
    ) -> Result<ApplyResult, HostAccessError> {
        let persistence = match self.persistence.stage_final_batch(&mut self.world) {
            Ok(persistence) => persistence,
            Err(error) => {
                self.lifecycle = HostLifecycle::Stopping;
                self.shared
                    .borrow_mut()
                    .emit(HostEventKind::LifecycleChanged(HostLifecycle::Stopping))?;
                self.complete_failed(
                    envelope.command_id,
                    admission,
                    "shutdown_finalization",
                    error.to_string(),
                )?;
                self.latched_fault = Some(HostFault::Scientific {
                    tick: self.world.tick(),
                    code: "shutdown_finalization".to_owned(),
                    message: error.to_string(),
                });
                self.synchronize_health()?;
                return Ok(ApplyResult::completed(false));
            }
        };
        self.lifecycle = HostLifecycle::Stopping;
        self.shared
            .borrow_mut()
            .emit(HostEventKind::LifecycleChanged(HostLifecycle::Stopping))?;
        self.revisions.control = next_control;
        let applied = self.applied_boundary();
        self.complete_applied_with(envelope.command_id, admission, applied, true)?;
        let blocked = self.offer_journal(envelope, applied, None, persistence)?;
        self.synchronize_health()?;
        Ok(ApplyResult::completed(blocked))
    }

    const fn applied_boundary(&self) -> AppliedCommand {
        AppliedCommand {
            tick: self.world.tick(),
            revisions: self.revisions,
        }
    }

    fn complete_applied(
        &self,
        command_id: CommandId,
        admission: AdmissionSequence,
        requires_journal: bool,
    ) -> Result<(), HostAccessError> {
        self.complete_applied_with(
            command_id,
            admission,
            self.applied_boundary(),
            requires_journal,
        )
    }

    fn complete_applied_with(
        &self,
        command_id: CommandId,
        admission: AdmissionSequence,
        applied: AppliedCommand,
        requires_journal: bool,
    ) -> Result<(), HostAccessError> {
        let journal = if requires_journal {
            JournalState::Pending
        } else {
            JournalState::NotRequired
        };
        let status = CommandStatus::try_new(
            command_id,
            Some(admission),
            ApplicationState::Applied(applied),
            journal,
        )
        .map_err(status_violation)?;
        self.complete_status(status)
    }

    fn complete_failed(
        &self,
        command_id: CommandId,
        admission: AdmissionSequence,
        code: &str,
        message: String,
    ) -> Result<(), HostAccessError> {
        let status = CommandStatus::try_new(
            command_id,
            Some(admission),
            ApplicationState::Failed(ApplicationFailure {
                code: code.to_owned(),
                message,
            }),
            JournalState::NotRequired,
        )
        .map_err(status_violation)?;
        self.complete_status(status)
    }

    fn offer_journal(
        &mut self,
        envelope: CommandEnvelope,
        applied: AppliedCommand,
        scientific: Option<Arc<ScientificBoundary>>,
        persistence: Option<Arc<scriptbots_core::PersistenceBatch>>,
    ) -> Result<bool, HostAccessError> {
        let batch_id = JournalBatchId::new(self.session_id, self.next_journal_sequence);
        self.next_journal_sequence = self
            .next_journal_sequence
            .checked_add(1)
            .ok_or_else(|| protocol_violation("journal batch sequence exhausted"))?;
        let shutdown = matches!(&envelope.command, HostCommand::Shutdown);
        let batch = Arc::new(JournalBatch::new(
            batch_id,
            Some(envelope),
            applied,
            scientific,
            persistence,
        ));
        self.active_journal_batch = Some(Arc::clone(&batch));
        let admission = self.journal.try_admit(&batch);
        let result = self.finish_journal_admission(&batch, admission, shutdown, false);
        if result.is_ok() {
            self.active_journal_batch = None;
        }
        result
    }

    fn offer_automatic_journal(
        &mut self,
        applied: AppliedCommand,
        scientific: Arc<ScientificBoundary>,
        persistence: Option<Arc<scriptbots_core::PersistenceBatch>>,
    ) -> Result<bool, HostAccessError> {
        let batch_id = JournalBatchId::new(self.session_id, self.next_journal_sequence);
        self.next_journal_sequence = self
            .next_journal_sequence
            .checked_add(1)
            .ok_or_else(|| protocol_violation("journal batch sequence exhausted"))?;
        let batch = Arc::new(JournalBatch::new(
            batch_id,
            None,
            applied,
            Some(scientific),
            persistence,
        ));
        self.active_journal_batch = Some(Arc::clone(&batch));
        let admission = self.journal.try_admit(&batch);
        let result = self.finish_journal_admission(&batch, admission, false, false);
        if result.is_ok() {
            self.active_journal_batch = None;
        }
        result
    }

    fn finish_journal_admission(
        &mut self,
        batch: &Arc<JournalBatch>,
        admission: JournalAdmission,
        shutdown: bool,
        was_retained: bool,
    ) -> Result<bool, HostAccessError> {
        if self.retain_identity_violation(batch, admission)? {
            return Ok(true);
        }
        match admission {
            JournalAdmission::Accepted { .. } => {
                self.seal_core_persistence()?;
                let shutdown_requirement = if shutdown {
                    Some(self.journal.shutdown_commit_requirement())
                } else {
                    None
                };
                self.inflight_journal.insert(
                    batch.id(),
                    InflightJournal {
                        command_id: batch.command_id(),
                        shutdown_requirement,
                        committed_volatile: false,
                    },
                );
                if was_retained {
                    self.retained_journal = None;
                    self.retained_blocker = None;
                }
                Ok(false)
            }
            JournalAdmission::Full { capacity, .. } => {
                self.retained_journal = Some(Arc::clone(batch));
                self.retained_blocker = Some(HostBlocker::JournalFull {
                    batch_id: batch.id(),
                    capacity,
                });
                self.synchronize_health()?;
                Ok(true)
            }
            JournalAdmission::Closed { .. } => {
                self.fail_closed_batch(batch)?;
                Ok(true)
            }
        }
    }

    fn latch_completed_step_fault(
        &mut self,
        tick: Tick,
        fault: &ScientificBoundaryFault,
    ) -> Result<(), HostAccessError> {
        self.latched_fault = Some(HostFault::Scientific {
            tick,
            code: fault.code().to_owned(),
            message: fault.message().to_owned(),
        });
        self.synchronize_health().map(|_| ())
    }

    fn automatic_step(&mut self) -> Result<ApplyResult, HostAccessError> {
        let next_scientific = self
            .revisions
            .scientific
            .checked_next()
            .ok_or_else(|| protocol_violation("scientific revision exhausted"))?;
        let completion = match self.persistence.step_outcome(&mut self.world) {
            Ok(completion) => completion,
            Err(error) => {
                self.latched_fault = Some(HostFault::Scientific {
                    tick: self.world.tick(),
                    code: "world_step".to_owned(),
                    message: error.to_string(),
                });
                self.synchronize_health()?;
                return Ok(ApplyResult::blocked());
            }
        };
        let scriptbots_core::StepCompletion {
            outcome,
            fault: completed_fault,
        } = completion;
        let scriptbots_core::StepOutcome {
            events,
            summary,
            births,
            deaths,
            combat,
            config_revision,
            resource_tick,
            persistence,
        } = outcome;
        self.revisions.scientific = next_scientific;
        self.revisions.config = ConfigRevision::new(config_revision);
        self.latest_completed_summary = Some(summary.clone());
        let tick = summary.tick;
        self.shared.borrow_mut().visible_tick = tick;
        let completed_fault = completed_fault.as_ref().map(completed_fault_record);
        let mut scientific = ScientificBoundary::new(
            events,
            summary,
            births,
            deaths,
            combat,
            config_revision,
            resource_tick,
        );
        if let Some(fault) = completed_fault.clone() {
            scientific = scientific.with_fault(fault);
        }
        let scientific = Arc::new(scientific);
        let persistence = persistence.into_batch();
        let applied = AppliedCommand {
            tick,
            revisions: self.revisions,
        };
        let blocked = self.offer_automatic_journal(applied, scientific, persistence)?;
        if let Some(fault) = completed_fault {
            self.latch_completed_step_fault(tick, &fault)?;
        }
        Ok(ApplyResult::science(
            blocked || self.latched_fault.is_some(),
        ))
    }

    fn automatic_budget(&mut self, elapsed_nanos: u64, speed_multiplier: f32) -> AutomaticBudget {
        let speed_units = speed_units(speed_multiplier);
        let threshold = u128::from(self.options.tick_period_nanos) * SPEED_SCALE;
        let maximum_steps =
            u128::try_from(self.options.max_automatic_steps_per_drive).unwrap_or(u128::MAX);
        let total_credit = self
            .cadence_credit
            .saturating_add(u128::from(elapsed_nanos).saturating_mul(u128::from(speed_units)));
        let due = total_credit / threshold;
        let admitted = due.min(maximum_steps);
        let steps = usize::try_from(admitted)
            .unwrap_or(self.options.max_automatic_steps_per_drive)
            .min(self.options.max_automatic_steps_per_drive);
        let skipped = due.saturating_sub(admitted);
        self.cadence_credit =
            (total_credit % threshold).saturating_add(admitted.saturating_mul(threshold));
        AutomaticBudget {
            steps,
            due: u64::try_from(due).unwrap_or(u64::MAX),
            skipped: u64::try_from(skipped).unwrap_or(u64::MAX),
        }
    }

    fn consume_automatic_credit(&mut self) {
        let threshold = u128::from(self.options.tick_period_nanos) * SPEED_SCALE;
        self.cadence_credit = self.cadence_credit.saturating_sub(threshold);
    }
}

impl ManualHostDriver for HostCore {
    fn session_id(&self) -> HostSessionId {
        self.session_id
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the sole-owner boundary keeps its ordered command, science, health, and publication phases visible together"
    )]
    fn drive(&mut self, now: ManualInstant) -> Result<DriveReceipt, HostAccessError> {
        if self.last_now.is_some_and(|last| now < last) {
            return Err(protocol_violation("manual time moved backwards"));
        }
        let elapsed_nanos = self
            .last_now
            .map_or(0, |last| now.as_nanos().saturating_sub(last.as_nanos()));
        let first_boundary = self.last_now.is_none();
        let was_running = self.lifecycle == HostLifecycle::Running && !self.playback.paused;
        let prior_speed = self.playback.speed_multiplier;
        self.last_now = Some(now);

        let events_before = self.shared.borrow().events.len();
        self.poll_journal_receipts()?;
        let mut commands_completed = 0;
        let mut scientific_steps = 0;
        let mut automatic_steps_due = 0;
        let mut automatic_steps_skipped = 0;
        let mut explicit_step_applied = false;

        if self.retained_journal.is_none() {
            while let Some(admitted) = self.pop_command() {
                self.active_command = Some(admitted.clone());
                let result = self.apply_command(admitted);
                if result.is_ok() {
                    self.active_command = None;
                }
                let result = result?;
                commands_completed += usize::from(result.command_completed);
                scientific_steps += usize::from(result.science_completed);
                explicit_step_applied |= result.science_completed;
                if result.blocked {
                    break;
                }
            }
        }

        if !first_boundary
            && !explicit_step_applied
            && was_running
            && self.lifecycle == HostLifecycle::Running
            && !self.playback.paused
            && self.retained_journal.is_none()
            && self.latched_fault.is_none()
        {
            let budget = self.automatic_budget(elapsed_nanos, prior_speed);
            automatic_steps_due = budget.due;
            automatic_steps_skipped = budget.skipped;
            for _ in 0..budget.steps {
                let result = self.automatic_step()?;
                if result.science_completed {
                    self.consume_automatic_credit();
                    scientific_steps += 1;
                }
                if result.blocked {
                    break;
                }
            }
        } else if self.playback.paused || explicit_step_applied {
            self.cadence_credit = 0;
        }

        self.synchronize_health()?;
        let latest = self.snapshots.latest();
        let (command_queue_depth, last_applied_command) = {
            let shared = self.shared.borrow();
            (
                shared.queue.len(),
                shared.last_applied.map(|(_, command_id)| command_id),
            )
        };
        let presentation_changed = latest.revisions.control != self.revisions.control
            || latest.revisions.config != self.revisions.config
            || latest.playback != self.playback
            || latest.lifecycle != self.lifecycle
            || latest.health != self.health
            || latest.command_queue_depth != command_queue_depth
            || latest.last_applied_command != last_applied_command;
        let scientific_since_publication = self
            .revisions
            .scientific
            .get()
            .saturating_sub(self.last_published_scientific.get());
        let science_cadence_due = scientific_steps != 0
            && scientific_since_publication >= self.options.snapshot_interval_ticks;
        let should_publish = presentation_changed || science_cadence_due;
        let snapshots_published = usize::from(should_publish);
        if should_publish {
            self.publish_snapshot()?;
        }
        let events_published = self.shared.borrow().events.len() - events_before;
        Ok(DriveReceipt {
            now,
            commands_completed,
            scientific_steps,
            automatic_steps_due,
            automatic_steps_skipped,
            scientific_revision: self.revisions.scientific,
            snapshots_published,
            events_published,
            blocker: self.current_blocker(),
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct AutomaticBudget {
    steps: usize,
    due: u64,
    skipped: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ApplyResult {
    command_completed: bool,
    science_completed: bool,
    blocked: bool,
}

impl ApplyResult {
    const fn completed(blocked: bool) -> Self {
        Self {
            command_completed: true,
            science_completed: false,
            blocked,
        }
    }

    const fn with_science(mut self) -> Self {
        self.science_completed = true;
        self
    }

    const fn science(blocked: bool) -> Self {
        Self {
            command_completed: false,
            science_completed: true,
            blocked,
        }
    }

    const fn blocked() -> Self {
        Self {
            command_completed: false,
            science_completed: false,
            blocked: true,
        }
    }
}

fn validate_options(options: HostCoreOptions) -> Result<(), HostCoreBuildError> {
    if options.command_capacity == 0 {
        return Err(HostCoreBuildError::InvalidOptions {
            message: "command_capacity must be nonzero".to_owned(),
        });
    }
    if options.tick_period_nanos == 0 {
        return Err(HostCoreBuildError::InvalidOptions {
            message: "tick_period_nanos must be nonzero".to_owned(),
        });
    }
    if options.max_automatic_steps_per_drive == 0 {
        return Err(HostCoreBuildError::InvalidOptions {
            message: "max_automatic_steps_per_drive must be nonzero".to_owned(),
        });
    }
    if options.snapshot_interval_ticks == 0 {
        return Err(HostCoreBuildError::InvalidOptions {
            message: "snapshot_interval_ticks must be nonzero".to_owned(),
        });
    }
    if !options.initial_playback.speed_multiplier.is_finite()
        || options.initial_playback.speed_multiplier < 0.0
    {
        return Err(HostCoreBuildError::InvalidOptions {
            message: "initial speed multiplier must be finite and non-negative".to_owned(),
        });
    }
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn speed_units(speed: f32) -> u64 {
    let scaled = f64::from(speed) * 1_000_000.0;
    if scaled >= u64::MAX as f64 {
        u64::MAX
    } else {
        scaled.round() as u64
    }
}

fn status_violation(error: StatusCombinationError) -> HostAccessError {
    protocol_violation(error.to_string())
}

fn completed_fault_record(fault: &CompletedStepFault) -> ScientificBoundaryFault {
    let code = match fault {
        CompletedStepFault::BrainSpawn(_) => "brain_spawn",
        CompletedStepFault::ScientificState(_) => "scientific_state",
    };
    ScientificBoundaryFault::new(code, fault.to_string())
}

fn protocol_violation(message: impl Into<String>) -> HostAccessError {
    HostAccessError::ProtocolViolation {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{
        AgentData, Generation, HydrologyField, HydrologyFlowDirection, HydrologyTile,
        HydrologyTileLayer, MapArtifact, MapArtifactMetadata, MapGeneratorKind, Position,
        ScriptBotsConfig, TerrainLayer, Velocity,
    };
    use std::{hint::black_box, time::Instant};

    fn world(persistence_interval: u32) -> WorldState {
        WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0x5eed_cafe),
            persistence_interval,
            ..ScriptBotsConfig::default()
        })
        .expect("deterministic test world")
    }

    fn snapshot_map_artifacts(world: &WorldState) -> (MapArtifact, MapArtifact, MapArtifact) {
        let base_terrain = world.terrain().clone();
        let width = base_terrain.width();
        let height = base_terrain.height();
        let cell_size = base_terrain.cell_size();
        let cell_count = base_terrain.tiles().len();
        let mut changed_tiles = base_terrain.tiles().to_vec();
        let first = changed_tiles
            .first_mut()
            .expect("snapshot map has at least one terrain tile");
        first.accent = if first.accent.to_bits() == 0.25_f32.to_bits() {
            0.75
        } else {
            0.25
        };
        let changed_terrain = TerrainLayer::from_tiles(width, height, cell_size, changed_tiles)
            .expect("changed snapshot terrain");

        let artifact = |terrain: TerrainLayer, changed: bool| {
            let mut hydrology_tiles = vec![
                HydrologyTile {
                    permeability: 0.4,
                    runoff_bias: 0.2,
                    basin_rank: 0.5,
                    channel_priority: 0.3,
                    swim_cost: 0.6,
                };
                cell_count
            ];
            let mut water_depth = vec![0.0; cell_count];
            if changed {
                hydrology_tiles[0].permeability = 0.8;
                water_depth[0] = 0.125;
            }
            MapArtifact::new(
                terrain,
                None,
                None,
                Some(
                    HydrologyTileLayer::new(width, height, hydrology_tiles)
                        .expect("snapshot hydrology tiles"),
                ),
                Some(
                    HydrologyField::new(
                        width,
                        height,
                        vec![HydrologyFlowDirection::None; cell_count],
                        vec![1.0; cell_count],
                        vec![0.5; cell_count],
                        vec![0; cell_count],
                        water_depth,
                    )
                    .expect("snapshot hydrology field"),
                ),
                MapArtifactMetadata {
                    generator: MapGeneratorKind::RuleBased,
                    tileset_id: if changed {
                        "snapshot-layer-b"
                    } else {
                        "snapshot-layer-a"
                    }
                    .to_owned(),
                    tileset_hash: u64::from(changed),
                    seed: 0x5a4f_4d41 + u64::from(changed),
                    width,
                    height,
                    attempt_count: 1,
                    succeeded_on: 1,
                    generated_at_epoch_ms: 0,
                },
            )
            .expect("snapshot map artifact")
        };

        let without_hydrology = MapArtifact::new(
            changed_terrain.clone(),
            None,
            None,
            None,
            None,
            MapArtifactMetadata {
                generator: MapGeneratorKind::RuleBased,
                tileset_id: "snapshot-layer-no-hydrology".to_owned(),
                tileset_hash: 2,
                seed: 0x5a4f_4d43,
                width,
                height,
                attempt_count: 1,
                succeeded_on: 1,
                generated_at_epoch_ms: 0,
            },
        )
        .expect("snapshot map without hydrology");

        (
            artifact(base_terrain, false),
            artifact(changed_terrain, true),
            without_hydrology,
        )
    }

    fn options(paused: bool) -> HostCoreOptions {
        HostCoreOptions {
            initial_playback: PlaybackSnapshot {
                paused,
                speed_multiplier: 1.0,
            },
            command_capacity: 32,
            tick_period_nanos: 10,
            max_automatic_steps_per_drive: 4,
            snapshot_interval_ticks: 1,
        }
    }

    fn host(paused: bool) -> (HostCore, LocalHostPort) {
        let core = HostCore::new(HostSessionId::new(7), world(0), options(paused))
            .expect("host construction");
        let port = core.local_port();
        (core, port)
    }

    fn envelope(id: u128, command: HostCommand) -> CommandEnvelope {
        CommandEnvelope::new(CommandId::new(id), command)
    }

    fn submit(port: &mut LocalHostPort, id: u128, command: HostCommand) -> CommandStatus {
        port.submit(envelope(id, command))
            .expect("local submission")
    }

    fn status(port: &mut LocalHostPort, id: u128) -> CommandStatus {
        port.command_status(CommandId::new(id))
            .expect("status lookup")
            .expect("status retained")
    }

    fn applied(status: &CommandStatus) -> AppliedCommand {
        match status.application() {
            ApplicationState::Applied(applied) => *applied,
            application => panic!("expected applied status, got {application:?}"),
        }
    }

    #[test]
    fn construction_and_first_drive_never_hide_a_warmup_tick() {
        for paused in [false, true] {
            let (mut core, _port) = host(paused);
            let initial = core.latest_snapshot();
            assert_eq!(initial.world.tick, 0);
            assert_eq!(initial.revisions.scientific, ScientificRevision::new(0));
            assert_eq!(initial.playback.paused, paused);

            let receipt = core
                .drive(ManualInstant::from_nanos(1_000))
                .expect("first manual boundary");
            assert_eq!(receipt.scientific_steps, 0);
            assert_eq!(core.world_tick(), Tick(0));
        }
    }

    #[test]
    fn snapshot_hub_keeps_independent_latest_only_cursors_without_a_backlog() {
        let (mut core, mut port) = host(true);
        let hub = core.snapshot_hub();
        let mut fast_a = hub.subscribe();
        let mut fast_b = hub.subscribe();
        let mut stalled = hub.subscribe();
        let mut dropped = hub.subscribe();

        let initial_a = hub
            .poll_latest(&mut fast_a)
            .expect("first fast poll")
            .expect("initial publication");
        let initial_b = hub
            .poll_latest(&mut fast_b)
            .expect("second fast poll")
            .expect("same initial publication");
        let stalled_initial = hub
            .poll_latest(&mut stalled)
            .expect("stalled initial poll")
            .expect("stalled cursor starts current");
        let dropped_initial = hub
            .poll_latest(&mut dropped)
            .expect("dropped initial poll")
            .expect("dropped cursor starts current");
        assert!(Arc::ptr_eq(&initial_a, &initial_b));
        assert!(Arc::ptr_eq(&initial_a, &stalled_initial));
        let initial_weak = Arc::downgrade(&initial_a);
        drop(initial_a);
        drop(initial_b);
        drop(stalled_initial);
        drop(dropped_initial);
        drop(dropped);

        let mut newest = SnapshotRevision::new(1);
        for sequence in 1_u128..=3 {
            submit(&mut port, sequence, HostCommand::Step);
            core.drive(ManualInstant::from_nanos(
                u64::try_from(sequence).expect("small test sequence"),
            ))
            .expect("explicit step publication");
            let snapshot_a = hub
                .poll_latest(&mut fast_a)
                .expect("fast A poll")
                .expect("fast A sees every publication");
            let snapshot_b = hub
                .poll_latest(&mut fast_b)
                .expect("fast B poll")
                .expect("fast B sees every publication");
            assert!(Arc::ptr_eq(&snapshot_a, &snapshot_b));
            newest = snapshot_a.revision;
        }

        assert!(
            initial_weak.upgrade().is_none(),
            "a scalar stalled cursor must not retain an obsolete snapshot Arc"
        );
        let stalled_latest = hub
            .poll_latest(&mut stalled)
            .expect("stalled latest poll")
            .expect("stalled cursor jumps to newest");
        assert_eq!(stalled_latest.revision, newest);
        assert_eq!(stalled.skipped_revisions(), 2);
        assert!(
            hub.poll_latest(&mut stalled)
                .expect("idempotent stalled repoll")
                .is_none()
        );

        let mut reconnecting = hub.resume_after(SnapshotRevision::new(2));
        let reconnected = hub
            .poll_latest(&mut reconnecting)
            .expect("reconnected poll")
            .expect("reconnected cursor receives newest");
        assert_eq!(reconnected.revision, newest);
        assert_eq!(reconnecting.skipped_revisions(), 1);
    }

    #[test]
    fn snapshots_publish_exact_completed_summary_without_persistence_cadence() {
        let mut core = HostCore::new(HostSessionId::new(71), world(3), options(true))
            .expect("host with sparse persistence cadence");
        let mut port = core.local_port();
        assert!(core.latest_snapshot().completed_summary.is_none());

        for tick in 1_u128..=2 {
            submit(&mut port, tick, HostCommand::Step);
            core.drive(ManualInstant::from_nanos(
                u64::try_from(tick).expect("small tick"),
            ))
            .expect("completed explicit step");
            let snapshot = core.latest_snapshot();
            let summary = snapshot
                .completed_summary
                .as_ref()
                .expect("completed summary is retained independently of persistence");
            let expected = core
                .world
                .history()
                .next_back()
                .expect("current summary remains in history");
            assert_eq!(summary, expected);
            assert_eq!(summary.tick, Tick(u64::try_from(tick).expect("small tick")));
            assert_eq!(snapshot.world.tick, summary.tick.0);
            assert_eq!(snapshot.world.summary.agent_count, summary.agent_count);
            assert_eq!(snapshot.world.summary.births, summary.births);
            assert_eq!(snapshot.world.summary.deaths, summary.deaths);
        }
    }

    #[test]
    fn construction_preserves_an_existing_current_tick_summary_exactly() {
        let mut prestepped = world(0);
        prestepped.step().expect("pre-host scientific tick");
        let expected = prestepped
            .history()
            .next_back()
            .expect("pre-host summary")
            .clone();

        let core = HostCore::new(HostSessionId::new(76), prestepped, options(true))
            .expect("host from prestepped world");
        let initial = core.latest_snapshot();
        assert_eq!(initial.world.tick, expected.tick.0);
        assert_eq!(initial.completed_summary.as_ref(), Some(&expected));
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one acceptance scenario makes every terrain, food, and hydrology Arc/revision transition explicit"
    )]
    fn layer_arcs_and_revisions_change_together_on_exact_content_bits() {
        let mut config = ScriptBotsConfig {
            rng_seed: Some(0x51a7_1c5),
            initial_food: 0.5,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        };
        let world = WorldState::new(config.clone()).expect("layer test world");
        let (base_map, changed_map, map_without_hydrology) = snapshot_map_artifacts(&world);
        let mut core =
            HostCore::new(HostSessionId::new(72), world, options(true)).expect("layer test host");
        let mut port = core.local_port();
        let initial = core.latest_snapshot();
        assert!(initial.layers.hydrology.is_none());
        assert_eq!(initial.layers.revisions.hydrology, LayerRevision::new(0));

        submit(&mut port, 1, HostCommand::Pause);
        core.drive(ManualInstant::from_nanos(1))
            .expect("control-only publication");
        let control_only = core.latest_snapshot();
        assert!(Arc::ptr_eq(
            &initial.layers.terrain,
            &control_only.layers.terrain
        ));
        assert!(Arc::ptr_eq(&initial.layers.food, &control_only.layers.food));
        assert_eq!(initial.layers.revisions, control_only.layers.revisions);
        assert_eq!(control_only.build.bulk_allocations, 0);

        core.world
            .apply_map_artifact(&base_map)
            .expect("install initial hydrology");
        core.publish_snapshot().expect("hydrology-add publication");
        let hydrology_added = core.latest_snapshot();
        assert!(Arc::ptr_eq(
            &control_only.layers.terrain,
            &hydrology_added.layers.terrain
        ));
        assert!(Arc::ptr_eq(
            &control_only.layers.food,
            &hydrology_added.layers.food
        ));
        assert_eq!(
            hydrology_added.layers.revisions.terrain,
            control_only.layers.revisions.terrain
        );
        assert_eq!(
            hydrology_added.layers.revisions.food,
            control_only.layers.revisions.food
        );
        assert_eq!(
            hydrology_added.layers.revisions.hydrology,
            control_only
                .layers
                .revisions
                .hydrology
                .checked_next()
                .expect("hydrology-add revision headroom")
        );
        let initial_hydrology = hydrology_added
            .layers
            .hydrology
            .as_ref()
            .expect("hydrology payload added");

        config.food_max = 0.25;
        config.initial_food = 0.25;
        config.food_respawn_amount = 0.25;
        submit(&mut port, 2, HostCommand::UpdateConfig(Box::new(config)));
        core.drive(ManualInstant::from_nanos(2))
            .expect("food-changing configuration publication");
        let food_changed = core.latest_snapshot();
        assert!(Arc::ptr_eq(
            &hydrology_added.layers.terrain,
            &food_changed.layers.terrain
        ));
        assert_eq!(
            hydrology_added.layers.revisions.terrain,
            food_changed.layers.revisions.terrain
        );
        assert!(!Arc::ptr_eq(
            &hydrology_added.layers.food,
            &food_changed.layers.food
        ));
        assert!(Arc::ptr_eq(
            initial_hydrology,
            food_changed
                .layers
                .hydrology
                .as_ref()
                .expect("food-only change retains hydrology")
        ));
        assert_eq!(
            food_changed.layers.revisions.food,
            hydrology_added
                .layers
                .revisions
                .food
                .checked_next()
                .expect("food revision headroom")
        );
        assert_eq!(food_changed.build.bulk_allocations, 1);
        assert!(food_changed.build.newly_allocated_capacity_bytes > 0);
        assert!(food_changed.build.reused_layer_capacity_bytes > 0);
        assert!(!same_f32_slice(&[0.0], &[-0.0]));

        core.world
            .apply_map_artifact(&changed_map)
            .expect("change terrain and hydrology");
        core.publish_snapshot()
            .expect("terrain and hydrology publication");
        let terrain_and_hydrology_changed = core.latest_snapshot();
        assert!(!Arc::ptr_eq(
            &food_changed.layers.terrain,
            &terrain_and_hydrology_changed.layers.terrain
        ));
        assert!(Arc::ptr_eq(
            &food_changed.layers.food,
            &terrain_and_hydrology_changed.layers.food
        ));
        assert!(!Arc::ptr_eq(
            food_changed
                .layers
                .hydrology
                .as_ref()
                .expect("old hydrology"),
            terrain_and_hydrology_changed
                .layers
                .hydrology
                .as_ref()
                .expect("changed hydrology")
        ));
        let changed_hydrology = terrain_and_hydrology_changed
            .layers
            .hydrology
            .as_ref()
            .expect("changed hydrology payload");
        assert_eq!(
            changed_hydrology.tiles[0].permeability.to_bits(),
            0.8_f32.to_bits()
        );
        assert_eq!(
            changed_hydrology.water_depth[0].to_bits(),
            0.125_f32.to_bits()
        );
        assert_eq!(
            terrain_and_hydrology_changed.layers.revisions.terrain,
            food_changed
                .layers
                .revisions
                .terrain
                .checked_next()
                .expect("terrain revision headroom")
        );
        assert_eq!(
            terrain_and_hydrology_changed.layers.revisions.hydrology,
            food_changed
                .layers
                .revisions
                .hydrology
                .checked_next()
                .expect("hydrology revision headroom")
        );

        core.world
            .apply_map_artifact(&map_without_hydrology)
            .expect("remove hydrology");
        core.publish_snapshot()
            .expect("hydrology-removal publication");
        let hydrology_removed = core.latest_snapshot();
        assert!(Arc::ptr_eq(
            &terrain_and_hydrology_changed.layers.terrain,
            &hydrology_removed.layers.terrain
        ));
        assert!(Arc::ptr_eq(
            &terrain_and_hydrology_changed.layers.food,
            &hydrology_removed.layers.food
        ));
        assert!(hydrology_removed.layers.hydrology.is_none());
        assert_eq!(
            hydrology_removed.layers.revisions.hydrology,
            terrain_and_hydrology_changed
                .layers
                .revisions
                .hydrology
                .checked_next()
                .expect("hydrology-removal revision headroom")
        );

        let encoded =
            serde_json::to_vec(hydrology_removed.as_ref()).expect("snapshot serialization");
        let decoded: RenderSnapshot =
            serde_json::from_slice(&encoded).expect("snapshot deserialization");
        assert_eq!(decoded, *hydrology_removed);
    }

    #[test]
    fn snapshot_stride_and_subscriber_load_cannot_change_science() {
        let config = ScriptBotsConfig {
            rng_seed: Some(0x5a4f_5707),
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        };
        let mut every_options = options(false);
        every_options.snapshot_interval_ticks = 1;
        let mut sparse_options = every_options;
        sparse_options.snapshot_interval_ticks = 3;
        let mut every = HostCore::new(
            HostSessionId::new(73),
            WorldState::new(config.clone()).expect("every world"),
            every_options,
        )
        .expect("every host");
        let mut sparse = HostCore::new(
            HostSessionId::new(74),
            WorldState::new(config).expect("sparse world"),
            sparse_options,
        )
        .expect("sparse host");
        let busy_hub = every.snapshot_hub();
        let mut readers = (0..128).map(|_| busy_hub.subscribe()).collect::<Vec<_>>();
        for reader in &mut readers {
            assert!(
                busy_hub
                    .poll_latest(reader)
                    .expect("initial busy-reader poll")
                    .is_some()
            );
        }

        every
            .drive(ManualInstant::from_nanos(0))
            .expect("every epoch");
        sparse
            .drive(ManualInstant::from_nanos(0))
            .expect("sparse epoch");
        let mut every_publications = 0usize;
        let mut sparse_publications = 0usize;
        for tick in 1_u64..=6 {
            every_publications += every
                .drive(ManualInstant::from_nanos(tick * 10))
                .expect("every cadence drive")
                .snapshots_published;
            for reader in &mut readers {
                let _ = busy_hub
                    .poll_latest(reader)
                    .expect("busy-reader latest poll");
            }
            sparse_publications += sparse
                .drive(ManualInstant::from_nanos(tick * 10))
                .expect("sparse cadence drive")
                .snapshots_published;
            assert_eq!(every.world_tick(), sparse.world_tick());
            assert_eq!(
                every.scientific_digest_v1().expect("every digest"),
                sparse.scientific_digest_v1().expect("sparse digest")
            );
        }
        assert_eq!(every.world_tick(), Tick(6));
        assert_eq!(every_publications, 6);
        assert_eq!(sparse_publications, 2);
    }

    fn snapshot_measurement_config() -> ScriptBotsConfig {
        ScriptBotsConfig {
            world_width: 800,
            world_height: 800,
            food_cell_size: 20,
            rng_seed: Some(0x5eed_ba5e),
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            population_crossover_chance: 0.0,
            reproduction_attempt_interval: 1,
            reproduction_attempt_chance: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            metabolism_ramp_floor: 0.0,
            metabolism_ramp_rate: 0.0,
            metabolism_boost_penalty: 0.0,
            temperature_discomfort_rate: 0.0,
            aging_health_decay_rate: 0.0,
            aging_health_decay_max: 0.0,
            aging_energy_penalty_rate: 0.0,
            spike_damage: 0.0,
            spike_energy_cost: 0.0,
            food_max: 2.0,
            food_growth_rate: 0.02,
            food_decay_rate: 0.0,
            food_diffusion_rate: 0.1,
            food_waste_rate: 0.0,
            history_capacity: 1,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn snapshot_measurement_world(agent_count: usize) -> WorldState {
        let mut world =
            WorldState::new(snapshot_measurement_config()).expect("snapshot measurement world");
        for ordinal in 0..agent_count {
            let x = (ordinal % 800) as f32;
            let y = ((ordinal * 37) % 800) as f32;
            world
                .try_spawn_agent(AgentData::new(
                    Position::new(x, y),
                    Velocity::default(),
                    0.0,
                    1.0,
                    [0.5, 0.5, 0.5],
                    0.0,
                    false,
                    0,
                    Generation(0),
                ))
                .unwrap_or_else(|error| panic!("snapshot agent {ordinal} failed: {error}"));
        }
        world
    }

    fn measured_snapshot_publication(core: &mut HostCore) -> (u64, SnapshotBuildStats) {
        let started = Instant::now();
        core.publish_snapshot()
            .expect("measured full snapshot publication");
        let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let snapshot = core.latest_snapshot();
        black_box(&snapshot);
        (elapsed, snapshot.build)
    }

    fn nearest_rank(samples: &[u64], percentile: usize) -> u64 {
        let mut ordered = samples.to_vec();
        ordered.sort_unstable();
        let rank = ordered
            .len()
            .saturating_mul(percentile)
            .div_ceil(100)
            .saturating_sub(1)
            .min(ordered.len().saturating_sub(1));
        ordered[rank]
    }

    #[test]
    #[ignore = "DSR-only reference-hardware snapshot measurement"]
    #[allow(
        clippy::too_many_lines,
        reason = "one DSR evidence record keeps warmups, raw samples, digests, capacity accounting, and both reference scales coherent"
    )]
    fn measure_snapshot_builds_at_1k_and_10k_agents() {
        const WARMUPS: usize = 20;
        const SAMPLES: usize = 200;
        for agent_count in [1_000, 10_000] {
            let mut world = snapshot_measurement_world(agent_count);
            let (base_map, changed_map, _) = snapshot_map_artifacts(&world);
            world
                .apply_map_artifact(&base_map)
                .expect("install measured base map");
            let scenario_config = world.config().clone();
            let session_id = HostSessionId::new(
                u64::try_from(agent_count).expect("measured agent count fits session id"),
            );
            let mut core =
                HostCore::new(session_id, world, options(true)).expect("snapshot measurement host");
            let initial_stats = core.latest_snapshot().build;
            assert_eq!(initial_stats.dynamic_agent_count, agent_count);
            let initial_digest = core
                .scientific_digest_v1()
                .expect("initial measurement digest");

            for _ in 0..WARMUPS {
                black_box(measured_snapshot_publication(&mut core));
            }

            let mut steady_samples = Vec::with_capacity(SAMPLES);
            let mut steady_stats = SnapshotBuildStats::default();
            for _ in 0..SAMPLES {
                let (elapsed, stats) = measured_snapshot_publication(&mut core);
                steady_stats = stats;
                steady_samples.push(elapsed);
            }
            assert_eq!(steady_stats.dynamic_agent_count, agent_count);
            assert_eq!(steady_stats.bulk_allocations, 1);
            assert_eq!(
                core.scientific_digest_v1()
                    .expect("post-steady measurement digest"),
                initial_digest,
                "steady snapshot publication must not alter science"
            );

            for sample in 0..WARMUPS {
                let map = if sample.is_multiple_of(2) {
                    &changed_map
                } else {
                    &base_map
                };
                core.world
                    .apply_map_artifact(map)
                    .expect("changed-layer warmup map");
                black_box(measured_snapshot_publication(&mut core));
            }
            let mut changed_layer_samples = Vec::with_capacity(SAMPLES);
            let mut changed_layer_stats = SnapshotBuildStats::default();
            for sample in 0..SAMPLES {
                let map = if sample.is_multiple_of(2) {
                    &changed_map
                } else {
                    &base_map
                };
                core.world
                    .apply_map_artifact(map)
                    .expect("changed-layer measurement map");
                let (elapsed, stats) = measured_snapshot_publication(&mut core);
                changed_layer_stats = stats;
                changed_layer_samples.push(elapsed);
            }
            assert_eq!(changed_layer_stats.dynamic_agent_count, agent_count);
            assert!(changed_layer_stats.bulk_allocations > steady_stats.bulk_allocations);

            core.world
                .apply_map_artifact(&changed_map)
                .expect("publication-invariance probe map");
            let digest_before_probe = core
                .scientific_digest_v1()
                .expect("pre-publication probe digest");
            black_box(measured_snapshot_publication(&mut core));
            let digest_after_probe = core
                .scientific_digest_v1()
                .expect("post-publication probe digest");
            assert_eq!(digest_before_probe, digest_after_probe);

            let steady_p50_ns = nearest_rank(&steady_samples, 50);
            let steady_p95_ns = nearest_rank(&steady_samples, 95);
            let changed_layer_p50_ns = nearest_rank(&changed_layer_samples, 50);
            let changed_layer_p95_ns = nearest_rank(&changed_layer_samples, 95);
            let budget_p95_ns = if agent_count == 1_000 {
                4_000_000
            } else {
                16_000_000
            };
            let evidence = serde_json::json!({
                "schema": "scriptbots.render_snapshot.measurement.v1",
                "scenario_contract": "standard-800x800-food20-seed-0x5eedba5e",
                "agent_count": agent_count,
                "warmups_per_case": WARMUPS,
                "samples_per_case": SAMPLES,
                "config": scenario_config,
                "initial_scientific_digest": initial_digest.overall,
                "changed_layer_scientific_digest": digest_after_probe.overall,
                "budget_p95_ns": budget_p95_ns,
                "steady": {
                    "raw_ns": steady_samples,
                    "p50_ns": steady_p50_ns,
                    "p95_ns": steady_p95_ns,
                    "bulk_vector_capacity_stats": steady_stats,
                },
                "changed_layer": {
                    "raw_ns": changed_layer_samples,
                    "p50_ns": changed_layer_p50_ns,
                    "p95_ns": changed_layer_p95_ns,
                    "bulk_vector_capacity_stats": changed_layer_stats,
                },
                "initial_bulk_vector_capacity_stats": initial_stats,
            });
            eprintln!(
                "{}",
                serde_json::to_string(&evidence).expect("serialize snapshot measurement evidence")
            );
            assert!(
                steady_p95_ns < budget_p95_ns,
                "steady snapshot p95 {steady_p95_ns}ns exceeded {budget_p95_ns}ns"
            );
            assert!(
                changed_layer_p95_ns < budget_p95_ns,
                "changed-layer snapshot p95 {changed_layer_p95_ns}ns exceeded {budget_p95_ns}ns"
            );
        }
    }

    #[test]
    fn volatile_journal_retains_the_exact_accepted_allocation_after_receipt_polling() {
        let mut journal = VolatileJournal::default();
        let batch = Arc::new(JournalBatch::new(
            JournalBatchId::new(HostSessionId::new(1), 1),
            None,
            AppliedCommand {
                tick: Tick(0),
                revisions: HostRevisions::default(),
            },
            None,
            None,
        ));

        assert!(journal.try_admit(&batch).is_accepted());
        assert_eq!(journal.poll_receipts(1).len(), 1);
        assert_eq!(journal.batches().len(), 1);
        assert!(Arc::ptr_eq(&journal.batches()[0], &batch));
    }

    #[test]
    fn playback_commands_and_bounded_cadence_use_injected_time_only() {
        let (mut core, mut port) = host(false);
        core.drive(ManualInstant::from_nanos(0))
            .expect("establish epoch");
        let receipt = core
            .drive(ManualInstant::from_nanos(100))
            .expect("bounded catch-up");
        assert_eq!(receipt.scientific_steps, 4);
        assert_eq!(core.world_tick(), Tick(4));

        submit(&mut port, 1, HostCommand::Pause);
        let pause = core
            .drive(ManualInstant::from_nanos(200))
            .expect("pause boundary");
        assert_eq!(pause.scientific_steps, 0);
        assert_eq!(core.world_tick(), Tick(4));

        submit(&mut port, 2, HostCommand::Resume);
        submit(&mut port, 3, HostCommand::SetSpeed(2.5));
        let resume = core
            .drive(ManualInstant::from_nanos(300))
            .expect("resume boundary");
        assert_eq!(resume.scientific_steps, 0);
        assert_eq!(core.world_tick(), Tick(4));
        assert_eq!(
            core.latest_snapshot().playback.speed_multiplier.to_bits(),
            2.5_f32.to_bits()
        );

        let accelerated = core
            .drive(ManualInstant::from_nanos(310))
            .expect("accelerated boundary");
        assert_eq!(accelerated.scientific_steps, 2);
        assert_eq!(core.world_tick(), Tick(6));
    }

    #[test]
    fn explicit_step_order_is_exact_and_suppresses_due_cadence() {
        let (mut core, mut port) = host(false);
        core.drive(ManualInstant::from_nanos(0))
            .expect("establish epoch");
        submit(&mut port, 1, HostCommand::Step);
        submit(&mut port, 2, HostCommand::Resume);
        let step_then_resume = core
            .drive(ManualInstant::from_nanos(1_000))
            .expect("step then resume");
        assert_eq!(step_then_resume.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(1));
        assert!(!core.latest_snapshot().playback.paused);

        submit(&mut port, 3, HostCommand::Resume);
        submit(&mut port, 4, HostCommand::Step);
        let resume_then_step = core
            .drive(ManualInstant::from_nanos(2_000))
            .expect("resume then step");
        assert_eq!(resume_then_step.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(2));
        assert!(core.latest_snapshot().playback.paused);

        submit(&mut port, 5, HostCommand::Step);
        submit(&mut port, 6, HostCommand::Step);
        let two_steps = core
            .drive(ManualInstant::from_nanos(3_000))
            .expect("two explicit steps");
        assert_eq!(two_steps.scientific_steps, 2);
        assert_eq!(core.world_tick(), Tick(4));
        assert_eq!(applied(&status(&mut port, 5)).tick, Tick(3));
        assert_eq!(applied(&status(&mut port, 6)).tick, Tick(4));
    }

    #[test]
    fn rejected_step_does_not_suppress_the_due_automatic_boundary() {
        let (mut core, mut port) = host(false);
        core.drive(ManualInstant::from_nanos(0))
            .expect("establish epoch");
        port.submit(
            envelope(1, HostCommand::Step).expecting_control_revision(ControlRevision::new(99)),
        )
        .expect("guarded step admission");

        let receipt = core
            .drive(ManualInstant::from_nanos(10))
            .expect("conflict plus cadence");
        assert_eq!(receipt.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(1));
        assert_eq!(
            core.latest_snapshot().revisions.control,
            ControlRevision::new(0)
        );
        assert!(matches!(
            status(&mut port, 1).application(),
            ApplicationState::Rejected(RejectionReason::ControlRevisionConflict { .. })
        ));
    }

    #[test]
    fn speed_change_does_not_erase_elapsed_time_at_the_prior_rate() {
        let (mut core, mut port) = host(false);
        core.drive(ManualInstant::from_nanos(0))
            .expect("establish epoch");
        submit(&mut port, 1, HostCommand::SetSpeed(2.0));

        let receipt = core
            .drive(ManualInstant::from_nanos(10))
            .expect("speed boundary");
        assert_eq!(receipt.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(1));
    }

    #[test]
    fn mixed_command_classes_apply_at_their_true_ordered_tick() {
        let updated = ScriptBotsConfig {
            food_growth_rate: 0.025,
            ..ScriptBotsConfig::default()
        };

        let (mut before, mut before_port) = host(true);
        submit(
            &mut before_port,
            1,
            HostCommand::UpdateConfig(Box::new(updated.clone())),
        );
        submit(&mut before_port, 2, HostCommand::Step);
        before
            .drive(ManualInstant::from_nanos(0))
            .expect("config then step");
        assert_eq!(applied(&status(&mut before_port, 1)).tick, Tick(0));
        assert_eq!(applied(&status(&mut before_port, 2)).tick, Tick(1));

        let (mut after, mut after_port) = host(true);
        submit(&mut after_port, 1, HostCommand::Step);
        submit(
            &mut after_port,
            2,
            HostCommand::UpdateConfig(Box::new(updated)),
        );
        after
            .drive(ManualInstant::from_nanos(0))
            .expect("step then config");
        assert_eq!(applied(&status(&mut after_port, 1)).tick, Tick(1));
        assert_eq!(applied(&status(&mut after_port, 2)).tick, Tick(1));
    }

    #[test]
    fn deduplication_and_control_cas_never_reapply_or_advance_on_conflict() {
        let (mut core, mut port) = host(true);
        let original = submit(&mut port, 1, HostCommand::Pause);
        let duplicate = submit(&mut port, 1, HostCommand::Step);
        assert_eq!(duplicate, original);

        let winner =
            envelope(2, HostCommand::Resume).expecting_control_revision(ControlRevision::new(1));
        let conflict =
            envelope(3, HostCommand::Step).expecting_control_revision(ControlRevision::new(1));
        port.submit(winner).expect("winner admission");
        port.submit(conflict).expect("conflict admission");
        core.drive(ManualInstant::from_nanos(0))
            .expect("ordered CAS boundary");

        assert_eq!(core.world_tick(), Tick(0));
        assert_eq!(
            core.latest_snapshot().revisions.control,
            ControlRevision::new(2)
        );
        assert!(matches!(
            status(&mut port, 3).application(),
            ApplicationState::Rejected(RejectionReason::ControlRevisionConflict {
                expected,
                actual,
            }) if *expected == ControlRevision::new(1) && *actual == ControlRevision::new(2)
        ));
    }

    #[test]
    fn bounded_admission_has_explicit_results_for_1_32_33_and_1000() {
        for requested in [1_usize, 32, 33, 1_000] {
            let (mut core, mut port) = host(true);
            let mut admitted = 0;
            let mut overloaded = 0;
            for offset in 0..requested {
                let id = u128::try_from(offset + 1).expect("test id");
                let result = submit(&mut port, id, HostCommand::Step);
                match result.application() {
                    ApplicationState::Admitted => admitted += 1,
                    ApplicationState::Rejected(RejectionReason::Overloaded { capacity: 32 }) => {
                        overloaded += 1;
                        assert_eq!(result.admission_sequence(), None);
                    }
                    other => panic!("unexpected burst result: {other:?}"),
                }
            }
            assert_eq!(admitted, requested.min(32));
            assert_eq!(overloaded, requested.saturating_sub(32));
            assert_eq!(port.queue_depth(), requested.min(32));
            let receipt = core
                .drive(ManualInstant::from_nanos(0))
                .expect("burst drive");
            assert_eq!(receipt.commands_completed, requested.min(32));
            assert_eq!(receipt.scientific_steps, requested.min(32));
            assert_eq!(
                core.world_tick().0,
                u64::try_from(requested.min(32)).expect("bounded burst count fits u64")
            );
        }
    }

    #[derive(Default)]
    struct FakeJournalState {
        full: bool,
        closed: bool,
        suppress_receipts: bool,
        attempts: Vec<Arc<JournalBatch>>,
        receipts: VecDeque<JournalReceipt>,
    }

    struct FakeJournal {
        state: Rc<RefCell<FakeJournalState>>,
    }

    impl JournalPort for FakeJournal {
        fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
            let mut state = self.state.borrow_mut();
            state.attempts.push(Arc::clone(batch));
            if state.closed {
                JournalAdmission::Closed {
                    batch_id: batch.id(),
                }
            } else if state.full {
                JournalAdmission::Full {
                    batch_id: batch.id(),
                    capacity: 1,
                }
            } else {
                if !state.suppress_receipts {
                    state.receipts.push_back(JournalReceipt::new(
                        batch.id(),
                        JournalReceiptState::Durable,
                    ));
                }
                JournalAdmission::Accepted {
                    batch_id: batch.id(),
                }
            }
        }

        fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
            let mut state = self.state.borrow_mut();
            let count = limit.min(state.receipts.len());
            state.receipts.drain(..count).collect()
        }
    }

    #[test]
    fn delayed_old_journal_receipt_cannot_rewind_last_applied_command() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState {
            suppress_receipts: true,
            ..FakeJournalState::default()
        }));
        let mut core = HostCore::with_journal(
            HostSessionId::new(75),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("host with delayed receipts");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);
        submit(&mut port, 2, HostCommand::Pause);
        core.drive(ManualInstant::from_nanos(0))
            .expect("step then newer control application");
        assert_eq!(
            core.latest_snapshot().last_applied_command,
            Some(CommandId::new(2))
        );

        let old_batch = journal_state.borrow().attempts[0].id();
        journal_state
            .borrow_mut()
            .receipts
            .push_back(JournalReceipt::new(old_batch, JournalReceiptState::Durable));
        core.drive(ManualInstant::from_nanos(1))
            .expect("delayed old receipt");
        core.publish_snapshot().expect("diagnostic republish");
        assert_eq!(
            core.latest_snapshot().last_applied_command,
            Some(CommandId::new(2))
        );
    }

    #[test]
    fn scientific_boundary_survives_when_core_persistence_is_disabled() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState::default()));
        let mut core = HostCore::with_journal(
            HostSessionId::new(8),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("host with recording journal");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);
        core.drive(ManualInstant::from_nanos(0))
            .expect("recorded step");

        let journal = journal_state.borrow();
        let batch = journal.attempts.first().expect("one journal attempt");
        assert!(batch.persistence().is_none());
        let boundary = batch.scientific().expect("lossless scientific outcome");
        assert_eq!(boundary.summary().tick, Tick(1));
        assert_eq!(boundary.events().tick, Tick(1));
        assert_eq!(boundary.births().len(), boundary.summary().births);
        assert_eq!(boundary.deaths().len(), boundary.summary().deaths);
    }

    #[test]
    fn full_journal_retains_the_exact_arc_and_stops_later_science_until_retry() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState {
            full: true,
            ..FakeJournalState::default()
        }));
        let mut core = HostCore::with_journal(
            HostSessionId::new(9),
            world(1),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("host with fake journal");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);
        submit(&mut port, 2, HostCommand::Resume);

        let blocked = core
            .drive(ManualInstant::from_nanos(0))
            .expect("completed step with backpressure");
        assert_eq!(blocked.scientific_steps, 1);
        assert!(matches!(
            blocked.blocker,
            Some(HostBlocker::JournalFull { .. })
        ));
        assert_eq!(core.world_tick(), Tick(1));
        assert_eq!(port.queue_depth(), 1);
        assert_eq!(status(&mut port, 1).journal(), &JournalState::Pending);

        let retained = core.pending_journal_batch().expect("exact retained batch");
        assert_eq!(
            retained
                .scientific()
                .expect("lossless scientific boundary")
                .summary()
                .tick,
            Tick(1)
        );
        assert!(retained.persistence().is_some());
        assert!(Arc::ptr_eq(&retained, &journal_state.borrow().attempts[0]));

        let blocked_now = DEFAULT_TICK_PERIOD_NANOS.saturating_mul(4);
        let still_blocked = core
            .drive(ManualInstant::from_nanos(blocked_now))
            .expect("elapsed time cannot bypass retained journal work");
        assert_eq!(still_blocked.scientific_steps, 0);
        assert_eq!(core.world_tick(), Tick(1));
        assert_eq!(port.queue_depth(), 1);
        assert!(Arc::ptr_eq(
            &retained,
            &core
                .pending_journal_batch()
                .expect("same batch remains retained")
        ));

        journal_state.borrow_mut().full = false;
        let retried = core
            .retry_retained_journal()
            .expect("retry call")
            .expect("retained work existed");
        assert!(retried.is_accepted());
        assert!(Arc::ptr_eq(
            &journal_state.borrow().attempts[0],
            &journal_state.borrow().attempts[1]
        ));

        let resumed = core
            .drive(ManualInstant::from_nanos(blocked_now.saturating_add(1)))
            .expect("receipt and queued resume");
        assert_eq!(resumed.scientific_steps, 0);
        assert_eq!(core.world_tick(), Tick(1));
        assert!(!core.latest_snapshot().playback.paused);
        assert_eq!(status(&mut port, 1).journal(), &JournalState::Durable);
    }

    #[test]
    fn closed_journal_is_terminal_on_the_journal_axis_and_queryable_in_health() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState {
            closed: true,
            ..FakeJournalState::default()
        }));
        let mut core = HostCore::with_journal(
            HostSessionId::new(10),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: journal_state,
            }),
        )
        .expect("host with closed journal");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);

        let receipt = core
            .drive(ManualInstant::from_nanos(0))
            .expect("step completed before closed journal result");
        assert_eq!(receipt.scientific_steps, 1);
        assert!(matches!(
            receipt.blocker,
            Some(HostBlocker::JournalClosed { .. })
        ));
        assert!(matches!(
            status(&mut port, 1).journal(),
            JournalState::Failed(_)
        ));
        assert!(matches!(
            core.health(),
            HostHealth::Faulted(HostFault::Journal { .. })
        ));
        assert!(core.pending_journal_batch().is_some());
    }

    #[test]
    fn shutdown_closes_admission_in_order_and_stops_after_its_receipt() {
        let (mut core, mut port) = host(true);
        let shutdown = submit(&mut port, 1, HostCommand::Shutdown);
        assert!(matches!(shutdown.application(), ApplicationState::Admitted));
        let rejected = submit(&mut port, 2, HostCommand::Pause);
        assert!(matches!(
            rejected.application(),
            ApplicationState::Rejected(RejectionReason::HostStopping)
        ));

        core.drive(ManualInstant::from_nanos(0))
            .expect("shutdown application");
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopping);
        core.drive(ManualInstant::from_nanos(1))
            .expect("volatile shutdown receipt");
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);
        assert_eq!(
            status(&mut port, 1).journal(),
            &JournalState::CommittedVolatile
        );
    }

    #[test]
    fn durable_adapter_shutdown_waits_past_a_volatile_receipt() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState {
            suppress_receipts: true,
            ..FakeJournalState::default()
        }));
        let mut core = HostCore::with_journal(
            HostSessionId::new(12),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("host with progressive durable journal");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Shutdown);
        core.drive(ManualInstant::from_nanos(0))
            .expect("shutdown application");

        let shutdown_id = journal_state.borrow().attempts[0].id();
        journal_state
            .borrow_mut()
            .receipts
            .push_back(JournalReceipt::new(
                shutdown_id,
                JournalReceiptState::CommittedVolatile,
            ));
        core.drive(ManualInstant::from_nanos(1))
            .expect("progressive volatile receipt");
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopping);
        assert_eq!(
            status(&mut port, 1).journal(),
            &JournalState::CommittedVolatile
        );

        journal_state
            .borrow_mut()
            .receipts
            .push_back(JournalReceipt::new(
                shutdown_id,
                JournalReceiptState::Durable,
            ));
        core.drive(ManualInstant::from_nanos(2))
            .expect("durable shutdown receipt");
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);
        assert_eq!(status(&mut port, 1).journal(), &JournalState::Durable);
    }

    #[test]
    fn earlier_journal_failure_blocks_shutdown_even_if_health_fault_is_overwritten() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState {
            suppress_receipts: true,
            ..FakeJournalState::default()
        }));
        let session_id = HostSessionId::new(13);
        let mut core = HostCore::with_journal(
            session_id,
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("host with manually acknowledged journal");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);
        submit(&mut port, 2, HostCommand::Shutdown);
        core.drive(ManualInstant::from_nanos(0))
            .expect("ordered step and shutdown application");

        let earlier_id = journal_state.borrow().attempts[0].id();
        let shutdown_id = journal_state.borrow().attempts[1].id();
        journal_state.borrow_mut().receipts.extend([
            JournalReceipt::new(
                earlier_id,
                JournalReceiptState::Failed(JournalFailure {
                    code: "write_failed".to_owned(),
                    message: "injected terminal failure".to_owned(),
                }),
            ),
            JournalReceipt::new(
                JournalBatchId::new(session_id, 999),
                JournalReceiptState::Durable,
            ),
            JournalReceipt::new(shutdown_id, JournalReceiptState::Durable),
        ]);
        core.drive(ManualInstant::from_nanos(1))
            .expect("failure, malformed receipt, and shutdown durability");

        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopping);
        assert!(matches!(
            status(&mut port, 1).journal(),
            JournalState::Failed(_)
        ));
        assert_eq!(status(&mut port, 2).journal(), &JournalState::Durable);
        assert!(matches!(
            core.health(),
            HostHealth::Faulted(HostFault::Protocol { .. })
        ));
    }

    #[test]
    fn stale_shutdown_cas_reopens_ingress_without_advancing_control() {
        let (mut core, mut port) = host(true);
        port.submit(
            envelope(1, HostCommand::Shutdown).expecting_control_revision(ControlRevision::new(7)),
        )
        .expect("guarded shutdown admission");
        core.drive(ManualInstant::from_nanos(0))
            .expect("shutdown conflict");

        assert!(matches!(
            status(&mut port, 1).application(),
            ApplicationState::Rejected(RejectionReason::ControlRevisionConflict { .. })
        ));
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Running);
        assert_eq!(
            core.latest_snapshot().revisions.control,
            ControlRevision::new(0)
        );
        assert!(matches!(
            submit(&mut port, 2, HostCommand::Pause).application(),
            ApplicationState::Admitted
        ));
    }

    #[test]
    fn shutdown_journals_the_exact_partial_persistence_tail() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState::default()));
        let mut core = HostCore::with_journal(
            HostSessionId::new(11),
            world(3),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("host with cadence journal");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);
        submit(&mut port, 2, HostCommand::Step);
        core.drive(ManualInstant::from_nanos(0))
            .expect("two deferred persistence steps");
        journal_state.borrow_mut().full = true;
        submit(&mut port, 3, HostCommand::Shutdown);

        let receipt = core
            .drive(ManualInstant::from_nanos(1))
            .expect("shutdown tail backpressure");
        assert!(matches!(
            receipt.blocker,
            Some(HostBlocker::JournalFull { .. })
        ));
        let retained = core
            .pending_journal_batch()
            .expect("shutdown tail retained");
        assert!(
            retained
                .command()
                .is_some_and(|command| matches!(&command.command, HostCommand::Shutdown))
        );
        assert_eq!(
            retained
                .persistence()
                .expect("partial cadence persistence payload")
                .summary
                .tick,
            Tick(2)
        );
        assert!(Arc::ptr_eq(
            &retained,
            journal_state
                .borrow()
                .attempts
                .last()
                .expect("shutdown admission attempt")
        ));
    }

    #[test]
    fn backwards_time_is_rejected_before_any_state_change() {
        let (mut core, _port) = host(true);
        core.drive(ManualInstant::from_nanos(5))
            .expect("initial boundary");
        let before = core.latest_snapshot();
        assert!(matches!(
            core.drive(ManualInstant::from_nanos(4)),
            Err(HostAccessError::ProtocolViolation { .. })
        ));
        assert_eq!(core.latest_snapshot(), before);
    }

    #[test]
    fn lifecycle_shutdown_uses_one_reserved_ordered_slot_and_is_idempotent() {
        let mut constrained = options(true);
        constrained.command_capacity = 1;
        let mut core =
            HostCore::new(HostSessionId::new(21), world(0), constrained).expect("constrained host");
        let mut port = core.local_port();
        let first = submit(&mut port, 1, HostCommand::Pause);
        assert_eq!(first.admission_sequence(), Some(AdmissionSequence::new(1)));

        let shutdown = core.request_shutdown().expect("reserved shutdown slot");
        assert_eq!(
            shutdown.admission_sequence(),
            Some(AdmissionSequence::new(2))
        );
        let repeated = core.request_shutdown().expect("idempotent shutdown");
        assert_eq!(repeated, shutdown);
        assert_eq!(core.shutdown_command_id(), Some(shutdown.command_id()));
        assert!(matches!(
            submit(&mut port, 2, HostCommand::Resume).application(),
            ApplicationState::Rejected(RejectionReason::HostStopping)
        ));

        let applied = core
            .drive(ManualInstant::from_nanos(0))
            .expect("ordered pause then shutdown");
        assert_eq!(applied.commands_completed, 2);
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopping);
        core.drive(ManualInstant::from_nanos(0))
            .expect("volatile shutdown barrier");
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);
    }

    #[test]
    fn bounded_catch_up_reports_skips_and_preserves_fractional_credit() {
        let (mut core, _port) = host(false);
        core.drive(ManualInstant::from_nanos(0))
            .expect("initial boundary");

        let late = core
            .drive(ManualInstant::from_nanos(105))
            .expect("bounded late boundary");
        assert_eq!(late.automatic_steps_due, 10);
        assert_eq!(late.automatic_steps_skipped, 6);
        assert_eq!(late.scientific_steps, 4);
        assert_eq!(core.world_tick(), Tick(4));

        let fractional = core
            .drive(ManualInstant::from_nanos(110))
            .expect("fractional credit boundary");
        assert_eq!(fractional.automatic_steps_due, 1);
        assert_eq!(fractional.automatic_steps_skipped, 0);
        assert_eq!(fractional.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(5));
    }

    #[test]
    fn drive_interest_disarms_paused_science_and_exposes_immediate_work() {
        let (mut core, mut port) = host(true);
        assert_eq!(core.drive_interest(), HostDriveInterest::WakeOnly);
        submit(&mut port, 1, HostCommand::Step);
        assert_eq!(core.drive_interest(), HostDriveInterest::ReadyNow);
        core.drive(ManualInstant::from_nanos(0))
            .expect("explicit step");
        assert_eq!(core.drive_interest(), HostDriveInterest::Draining);
        core.drive(ManualInstant::from_nanos(0))
            .expect("volatile step receipt");
        assert_eq!(core.drive_interest(), HostDriveInterest::WakeOnly);
        assert!(core.scientific_digest_v1().is_ok());
    }

    #[test]
    fn host_core_source_has_no_shared_mutable_world_escape_hatch() {
        let source = include_str!("host_core.rs");
        let shared_world_lock = ["Arc<", "Mutex<", "WorldState>>"].concat();
        let thread_spawn = ["thread::", "spawn"].concat();
        assert!(!source.contains(&shared_world_lock));
        assert!(!source.contains(&thread_spawn));
    }
}
