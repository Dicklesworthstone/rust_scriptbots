use std::cmp::Reverse;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
// removed duplicate import

use arc_swap::ArcSwapOption;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;
// removed duplicate import

use scriptbots_core::{
    AgentDebugInfo, AgentDebugQuery, ControlCommand, DietClass, HydrologyFlowDirection,
    ScriptBotsConfig, SelectionMode, SelectionState, SelectionUpdate, TerrainKind, Tick,
    WorldState,
};

use scriptbots_core::ConfigAuditEntry;
use scriptbots_core::check_knob_ranges;
#[cfg(feature = "gui")]
use scriptbots_render::{OffscreenScene, render_offscreen_scene};
use scriptbots_runtime::{
    ApplicationState, CommandEnvelope, CommandId, HostCommand, HostPort, JournalState,
    RenderSnapshot, channel::ChannelHostPort,
};
use smallvec::SmallVec;

/// Snapshot of configuration state returned to external clients.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ConfigSnapshot {
    pub tick: u64,
    pub config: Value,
}

impl ConfigSnapshot {
    fn from_world(config: &ScriptBotsConfig, tick: Tick) -> Result<Self, ControlError> {
        let config_value = serde_json::to_value(config).map_err(ControlError::serialization)?;
        Ok(Self {
            tick: tick.0,
            config: config_value,
        })
    }
}

/// Status summary of the running simulation for control clients.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SimulationStatusDto {
    pub tick: u64,
    pub agent_count: usize,
    pub is_closed: bool,
    pub config_revision: u64,
}

impl SimulationStatusDto {
    fn from_world(world: &WorldState) -> Self {
        Self {
            tick: world.tick().0,
            agent_count: world.agent_count(),
            is_closed: world.is_closed(),
            config_revision: world.config_revision(),
        }
    }
}

/// Snapshot describing the current hydrology state.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct HydrologySnapshot {
    pub width: u32,
    pub height: u32,
    pub total_water_depth: f32,
    pub mean_water_depth: f32,
    pub flooded_shallow_count: u32,
    pub flooded_deep_count: u32,
    pub shallow_threshold: f32,
    pub deep_threshold: f32,
    #[schema(value_type = Vec<f32>)]
    pub water_depth: Vec<f32>,
    #[schema(value_type = Vec<String>)]
    pub flow_directions: Vec<String>,
    #[schema(value_type = Vec<u32>)]
    pub basin_ids: Vec<u32>,
    #[schema(value_type = Vec<f32>)]
    pub accumulation: Vec<f32>,
    #[schema(value_type = Vec<f32>)]
    pub spill_elevation: Vec<f32>,
}

impl HydrologySnapshot {
    const SHALLOW_THRESHOLD: f32 = 0.05;
    const DEEP_THRESHOLD: f32 = 0.2;

    fn from_snapshot(state: &scriptbots_runtime::HydrologyLayerSnapshot) -> Self {
        let total_water_depth: f32 = state.water_depth.iter().sum();
        let cell_count = state.water_depth.len().max(1) as f32;
        let shallow = state
            .water_depth
            .iter()
            .filter(|&&depth| depth >= Self::SHALLOW_THRESHOLD)
            .count();
        let deep = state
            .water_depth
            .iter()
            .filter(|&&depth| depth >= Self::DEEP_THRESHOLD)
            .count();

        let flow_directions = state
            .flow_directions
            .iter()
            .map(|direction| {
                match direction {
                    HydrologyFlowDirection::North => "N",
                    HydrologyFlowDirection::South => "S",
                    HydrologyFlowDirection::East => "E",
                    HydrologyFlowDirection::West => "W",
                    HydrologyFlowDirection::None => "-",
                }
                .to_owned()
            })
            .collect();

        Self {
            width: state.width,
            height: state.height,
            total_water_depth,
            mean_water_depth: total_water_depth / cell_count,
            flooded_shallow_count: saturating_u32(shallow),
            flooded_deep_count: saturating_u32(deep),
            shallow_threshold: Self::SHALLOW_THRESHOLD,
            deep_threshold: Self::DEEP_THRESHOLD,
            water_depth: state.water_depth.clone(),
            flow_directions,
            basin_ids: state.basin_ids.clone(),
            accumulation: state.accumulation.clone(),
            spill_elevation: state.spill_elevation.clone(),
        }
    }
}

fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Enumeration describing the primitive type of a knob.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum KnobKind {
    Number,
    Integer,
    Boolean,
    String,
    Array,
    Object,
    Null,
}

/// Public descriptor for a single configuration knob.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KnobEntry {
    pub path: String,
    pub kind: KnobKind,
    pub value: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Request payload for updating a configuration knob.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KnobUpdate {
    pub path: String,
    #[schema(value_type = Object, nullable = false)]
    pub value: Value,
}

/// Errors produced by the control domain when mutating configuration.
#[derive(Debug, Error)]
pub enum ControlError {
    #[error(transparent)]
    Host(#[from] scriptbots_runtime::HostAccessError),
    #[error("failed to lock world state")]
    Lock,
    #[error("{0}")]
    InvalidPatch(String),
    #[error("unknown knob path: {0}")]
    UnknownPath(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("command queue is full; retry later")]
    CommandQueueFull,
    #[error("command queue has been closed")]
    CommandQueueClosed,
}

impl ControlError {
    fn serialization(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

impl From<PoisonError<MutexGuard<'_, WorldState>>> for ControlError {
    fn from(_: PoisonError<MutexGuard<'_, WorldState>>) -> Self {
        ControlError::Lock
    }
}

type KnobsCache = std::sync::Arc<Mutex<Option<(u64, Vec<KnobEntry>)>>>;

/// Lock a derived cache, adopting the contents even if a previous holder panicked.
///
/// The world mutex propagates poisoning as [`ControlError::Lock`] because a panic
/// mid-tick can leave scientific state torn. The knob and command-status caches are
/// different: both are pure projections whose only invariants are enforced on read
/// (knobs are revalidated against `config_revision`, statuses are looked up by exact
/// command ID), so a poisoned guard holds a structurally intact value. Unwrapping
/// here instead panicked the axum worker on every later `/api/knobs`, `/api/config`,
/// and `/api/status` request, turning one unrelated panic into a permanently dead
/// control plane (bd-2t3k).
fn lock_cache<T>(cache: &Mutex<T>) -> MutexGuard<'_, T> {
    cache.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Status and completed summary captured together while the owner can read the world.
/// The status can exist at bootstrap before the first completed tick summary.
#[derive(Debug, Clone)]
pub struct PublishedWorldObservation {
    summary: Option<scriptbots_core::TickSummary>,
    status: SimulationStatusDto,
}

/// Latest owner observation; readers load it without acquiring the world mutex.
pub type SharedLatestSummary = Arc<ArcSwapOption<PublishedWorldObservation>>;

/// Publish the world's actual fields at an owner boundary, including failed steps.
pub fn publish_world_observation(slot: &SharedLatestSummary, world: &WorldState) {
    slot.store(Some(Arc::new(PublishedWorldObservation {
        summary: world.history().next_back().cloned(),
        status: SimulationStatusDto::from_world(world),
    })));
}

/// Fresh, empty published-summary slot.
#[must_use]
pub fn empty_latest_summary() -> SharedLatestSummary {
    Arc::new(ArcSwapOption::empty())
}

/// Wire tag of `scriptbots_runtime::ApplicationState::Admitted`.
///
/// The legacy app-owned `CommandBus` can honestly report only this value: it hands a
/// command an admission order and nothing on that path observes application.
pub const APPLICATION_STATE_ADMITTED: &str = "admitted";

/// Wire tag of `scriptbots_runtime::ApplicationState::Applied`.
///
/// Reachable only when something OBSERVES a command being applied to the world
/// and reports it back. Nothing on the legacy path does that yet, which is why
/// this constant exists alongside a ledger that can hold it rather than being
/// written anywhere on submission (bd-k7nq).
pub const APPLICATION_STATE_APPLIED: &str = "applied";

/// Wire tag of `scriptbots_runtime::ApplicationState::Rejected`.
///
/// A command that reached the applier and was refused there — distinct from one
/// refused at admission, which never gets a receipt at all.
pub const APPLICATION_STATE_REJECTED: &str = "rejected";

/// Wire tag of `scriptbots_runtime::JournalState::NotRequired`.
///
/// That variant exists precisely "for non-runtime and historical producers", which is
/// what the legacy bus is — it never writes a lifecycle record, so `pending` would be
/// a promise nothing keeps.
pub const JOURNAL_STATE_NOT_REQUIRED: &str = "not_required";

/// Two-axis status representation returned by REST, MCP, and CLI interfaces for commands.
///
/// The two axes are independent by design: application tracks
/// `admitted`/`applied`/`rejected`/`failed`, journal tracks
/// `not_required`/`pending`/`committed_volatile`/`durable`. Commands issued through the
/// legacy bus stay at `admitted`/`not_required` for their whole life; only the
/// `HostCore` path can advance them (bd-f65w).
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CommandStatusDto {
    pub command_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub admission_sequence: Option<u64>,
    pub application_state: String,
    pub journal_state: String,
    pub control_revision: u64,
    pub scientific_revision: u64,
}

/// What the applier observed when it applied a drained command.
///
/// Deliberately only the two outcomes an applier can actually witness. It knows
/// whether the world took the command; it does not know whether a journal
/// commit will follow, so it is given no way to claim one (bd-k7nq).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandOutcome {
    /// The world accepted the command.
    Applied,
    /// The world refused it at application time.
    Rejected,
}

/// Records an applied command's outcome against the identity it travelled with.
///
/// Held by whatever drains and applies, so the report comes FROM the applier
/// rather than from the submitter guessing.
pub type CommandReporter = std::sync::Arc<dyn Fn(&str, CommandOutcome) + Send + Sync>;

/// The two-axis record of what has happened to each submitted command.
///
/// Eight instances of one defect were fixed across five surfaces by teaching
/// each of them to say `admitted` instead of asserting an outcome the host
/// never acknowledged. That was the right correction, but it left `admitted` as
/// the END of the road rather than the first step: nothing could ever move a
/// command past it, so a caller still had no way to learn what actually
/// happened next (bd-k7nq).
///
/// This is the mechanism that lets a receipt advance. It replaces a bare
/// `HashMap<String, CommandStatusDto>` whose only operation was insert, which
/// is why nothing could advance: there was no notion of a transition at all,
/// so there was nothing to call when a command was applied.
///
/// TRANSITIONS ARE VALIDATED rather than assumed. `admitted` may become
/// `applied` or `rejected`; a terminal state may not silently change to
/// another; an unknown id is refused rather than invented. A ledger that
/// accepted any write would let a caller report an outcome it had not observed
/// — the same defect one level down, which is exactly the trap this whole line
/// of work has been about.
#[derive(Debug, Default)]
pub struct CommandLedger {
    entries: std::collections::HashMap<String, CommandStatusDto>,
}

/// Why a ledger transition was refused.
#[derive(Debug, PartialEq, Eq)]
pub enum LedgerError {
    /// No command was ever admitted under this id.
    Unknown(String),
    /// The command already reached a terminal state.
    AlreadyTerminal {
        command_id: String,
        current: String,
        attempted: String,
    },
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(id) => write!(
                f,
                "no command was admitted under id `{id}`; a receipt cannot be advanced for a \
                 command that was never submitted"
            ),
            Self::AlreadyTerminal {
                command_id,
                current,
                attempted,
            } => write!(
                f,
                "command `{command_id}` is already `{current}` and cannot become `{attempted}`; \
                 a terminal outcome is not revisable"
            ),
        }
    }
}

impl CommandLedger {
    /// Is this application state one that can no longer change?
    fn is_terminal(state: &str) -> bool {
        state == APPLICATION_STATE_APPLIED || state == APPLICATION_STATE_REJECTED
    }

    /// Record a freshly admitted command.
    fn admit(&mut self, status: CommandStatusDto) {
        self.entries.insert(status.command_id.clone(), status);
    }

    /// The receipt for an id, if one was ever admitted.
    fn get(&self, command_id: &str) -> Option<&CommandStatusDto> {
        self.entries.get(command_id)
    }

    /// Advance a command to a terminal application state.
    ///
    /// Returns the updated receipt so a caller reports what the ledger now
    /// holds rather than what it assumed it would hold.
    fn resolve(
        &mut self,
        command_id: &str,
        application_state: &str,
    ) -> Result<CommandStatusDto, LedgerError> {
        let entry = self
            .entries
            .get_mut(command_id)
            .ok_or_else(|| LedgerError::Unknown(command_id.to_owned()))?;
        if Self::is_terminal(&entry.application_state) {
            // Re-reporting the SAME terminal state is a retry, not a conflict:
            // an applier that replays its report must not be punished for it.
            if entry.application_state == application_state {
                return Ok(entry.clone());
            }
            return Err(LedgerError::AlreadyTerminal {
                command_id: command_id.to_owned(),
                current: entry.application_state.clone(),
                attempted: application_state.to_owned(),
            });
        }
        entry.application_state = application_state.to_owned();
        Ok(entry.clone())
    }
}

/// Request payload for setting simulation speed multiplier.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SpeedRequest {
    pub speed: f32,
}

/// Shared handle used by REST, CLI, and MCP surfaces to access the running world.
#[derive(Clone)]
pub struct ControlHandle {
    host: ChannelHostPort,
    knobs_cache: KnobsCache,
    command_counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
    command_namespace: u64,
}

impl ControlHandle {
    pub fn new(host: ChannelHostPort) -> Self {
        static NEXT_NAMESPACE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            host,
            knobs_cache: std::sync::Arc::new(Mutex::new(None)),
            command_counter: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            command_namespace: NEXT_NAMESPACE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Produce a PNG snapshot of the world without a live window.
    pub fn snapshot_png(&self, width: u32, height: u32) -> Result<Vec<u8>, ControlError> {
        #[cfg(feature = "gui")]
        {
            const MAX_PIXELS: u64 = 64 * 1024 * 1024; // 64M px guardrail
            if (width as u64) * (height as u64) > MAX_PIXELS {
                return Err(ControlError::InvalidPatch(
                    "requested image too large".into(),
                ));
            }
            // Rasterization holds an immutable publication, never the owner.
            let scene = OffscreenScene::capture(&self.read_snapshot()?);
            Ok(render_offscreen_scene(&scene, width, height))
        }
        #[cfg(not(feature = "gui"))]
        {
            // Reference params to avoid unused warnings in non-GUI builds
            let _ = (width, height);
            Err(ControlError::InvalidPatch(
                "PNG snapshot requires gui feature".into(),
            ))
        }
    }

    pub fn read_snapshot(&self) -> Result<Arc<RenderSnapshot>, ControlError> {
        self.host
            .clone()
            .snapshot_after(None)?
            .ok_or(ControlError::Lock)
    }

    /// Retrieve the current configuration snapshot.
    pub fn snapshot(&self) -> Result<ConfigSnapshot, ControlError> {
        let snapshot = self.read_snapshot()?;
        ConfigSnapshot::from_world(&snapshot.config, Tick(snapshot.world.tick))
    }

    /// Retrieve the latest tick summary from the running world.
    pub fn latest_summary(&self) -> Result<scriptbots_core::TickSummary, ControlError> {
        let snapshot = self.read_snapshot()?;
        Ok(snapshot
            .completed_summary
            .clone()
            .or_else(|| snapshot.summary_history.last().cloned())
            .unwrap_or_else(|| scriptbots_core::TickSummary {
                tick: Tick(snapshot.world.tick),
                agent_count: snapshot.world.agents.len(),
                births: snapshot.world.summary.births,
                deaths: snapshot.world.summary.deaths,
                total_energy: snapshot.world.summary.total_energy,
                average_energy: snapshot.world.summary.average_energy,
                average_health: snapshot.world.summary.average_health,
                max_age: snapshot.world.agents.iter().map(|agent| agent.age).max().unwrap_or(0),
                spike_hits: 0,
            }))
    }

    /// Retrieve a filtered debug listing of agents.
    pub fn debug_agents(
        &self,
        query: AgentDebugQuery,
    ) -> Result<Vec<AgentDebugInfo>, ControlError> {
        Ok(self.host.debug_agents(query)?)
    }

    /// Submit a selection update and return its admission receipt.
    ///
    /// This used to return `Result<(), _>`, which made selection the only
    /// control surface a client could not follow. Every other command hands
    /// back a [`CommandStatusDto`] carrying a command id, the admission
    /// sequence and both revision axes; selection handed back nothing, so the
    /// REST layer had no identity to report and invented a bare `queued: true`
    /// instead. A client could not poll the outcome, could not tell one
    /// selection from another, and could not distinguish a command that was
    /// applied from one that was admitted and then dropped (bd-2z0.4.9).
    ///
    /// Poll the returned identity for the host's independent application and
    /// journal progress; admission itself does not establish application.
    pub fn update_selection(
        &self,
        update: SelectionUpdate,
        idempotency_key: Option<&str>,
    ) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command_with_key(
            ControlCommand::UpdateSelection(update),
            idempotency_key,
        )
    }

    /// Enqueue step commands for the simulation driver to advance `count` ticks.
    pub fn step_count(&self, count: u64) -> Result<CommandStatusDto, ControlError> {
        let iterations = count.max(1);
        let mut last_status = None;
        for _ in 0..iterations {
            last_status = Some(self.submit_control_command(ControlCommand::Step)?);
        }
        Ok(last_status.expect("at least one iteration"))
    }

    /// Read current status without waiting for a busy world owner.
    /// During contention, return the last published boundary with its observed tick.
    pub fn status(&self) -> Result<SimulationStatusDto, ControlError> {
        let snapshot = self.read_snapshot()?;
        Ok(SimulationStatusDto {
            tick: snapshot.world.tick,
            agent_count: snapshot.world.agents.len(),
            is_closed: snapshot.config.closed,
            config_revision: snapshot.revisions.config.get(),
        })
    }

    /// Retrieve a snapshot of the current hydrology state, if available.
    pub fn hydrology_snapshot(&self) -> Result<Option<HydrologySnapshot>, ControlError> {
        Ok(self
            .read_snapshot()?
            .layers
            .hydrology
            .as_deref()
            .map(HydrologySnapshot::from_snapshot))
    }

    /// Flatten the configuration into individual knob descriptors for discovery.
    pub fn list_knobs(&self) -> Result<Vec<KnobEntry>, ControlError> {
        let snapshot = self.read_snapshot()?;
        let rev = snapshot.revisions.config.get();
        if let Some((cached_rev, cached)) = lock_cache(&self.knobs_cache).as_ref()
            && *cached_rev == rev
        {
            return Ok(cached.clone());
        }
        let config_value =
            serde_json::to_value(snapshot.config.as_ref()).map_err(ControlError::serialization)?;
        let mut entries = Vec::with_capacity(256);
        let mut prefix = String::new();
        flatten_value(&mut prefix, &config_value, &mut entries);
        *lock_cache(&self.knobs_cache) = Some((rev, entries.clone()));
        Ok(entries)
    }

    /// Retrieve the configuration audit log accumulated since startup.
    pub fn audit(&self) -> Result<Vec<ConfigAuditEntry>, ControlError> {
        Ok(self.read_snapshot()?.config_audit.as_ref().clone())
    }

    /// Build a tail of recent narrative events from the world's tick history.
    /// Events include births, deaths, and combat spike hits.
    pub fn events_tail(&self, limit: usize) -> Result<Vec<EventEntry>, ControlError> {
        // Answered before the lock is taken. This used to sit AFTER it, so a
        // limit=0 request contended on the world mutex to return an empty vec.
        if limit == 0 {
            return Ok(Vec::new());
        }
        let snapshot = self.read_snapshot()?;
        // The limit arrives unclamped from the query string; cap it so a hostile
        // request cannot reserve unbounded memory (history yields ≤3 events/tick).
        let limit = limit.min(snapshot.summary_history.len().saturating_mul(3).max(1));
        let mut events = Vec::with_capacity(limit);
        for summary in snapshot.summary_history.iter().rev() {
            if summary.births > 0 {
                events.push(EventEntry::new(
                    summary.tick.0,
                    EventKind::Birth,
                    saturating_u32(summary.births),
                ));
                if events.len() >= limit {
                    break;
                }
            }
            if summary.deaths > 0 {
                events.push(EventEntry::new(
                    summary.tick.0,
                    EventKind::Death,
                    saturating_u32(summary.deaths),
                ));
                if events.len() >= limit {
                    break;
                }
            }
            if summary.spike_hits > 0 {
                events.push(EventEntry::new(
                    summary.tick.0,
                    EventKind::Combat,
                    summary.spike_hits,
                ));
                if events.len() >= limit {
                    break;
                }
            }
        }
        Ok(events)
    }

    /// Render a coarse ASCII map of terrain, food, and agents — the server-side
    /// equivalent of the terminal renderer's saved snapshots.
    pub fn ascii_map(&self) -> Result<String, ControlError> {
        let snapshot = self.read_snapshot()?;
        let food = &snapshot.layers.food;
        let terrain = &snapshot.layers.terrain;
        let grid_w = food.width.max(1) as usize;
        let grid_h = food.height.max(1) as usize;
        let width = grid_w.clamp(16, 96);
        let height = grid_h.clamp(8, 48);
        let food_max = snapshot.config.food_max.max(f32::EPSILON);
        let world_w = (snapshot.config.world_width as f32).max(1.0);
        let world_h = (snapshot.config.world_height as f32).max(1.0);
        let tiles = &terrain.tiles;
        let cells = &food.cells;

        let mut rows = vec![vec![' '; width]; height];
        for (y, row) in rows.iter_mut().enumerate() {
            for (x, slot) in row.iter_mut().enumerate() {
                let cell_x = (x * grid_w) / width;
                let cell_y = (y * grid_h) / height;
                let idx = cell_y * grid_w + cell_x;
                let kind = tiles.get(idx).map(|tile| tile.kind);
                let food_level = cells.get(idx).copied().unwrap_or(0.0) / food_max;
                let base = match kind {
                    Some(TerrainKind::DeepWater) => '~',
                    Some(TerrainKind::ShallowWater) => '=',
                    Some(TerrainKind::Sand) => '.',
                    Some(TerrainKind::Grass) => ',',
                    Some(TerrainKind::Bloom) => '*',
                    Some(TerrainKind::Rock) => '^',
                    None => ' ',
                };
                *slot = if food_level > 0.66 {
                    '#'
                } else if food_level > 0.33 {
                    '+'
                } else {
                    base
                };
            }
        }
        for agent in &snapshot.world.agents {
            let x = (((agent.position[0] / world_w) * width as f32) as usize).min(width - 1);
            let y = (((agent.position[1] / world_h) * height as f32) as usize).min(height - 1);
            rows[y][x] = '@';
        }

        let mut out = format!("ScriptBots tick {}\n", snapshot.world.tick);
        for row in rows {
            out.extend(row);
            out.push('\n');
        }
        Ok(out)
    }

    /// Compute scoreboard snapshots: top predators (carnivores) by energy and oldest living agents.
    pub fn compute_scoreboard(&self, limit: usize) -> Result<Scoreboard, ControlError> {
        // Collection happens inside the seam and ranking happens outside it, so
        // the expensive sort provably cannot hold the world lock. That used to
        // rest on a hand-placed `drop(world)` with a comment; now the borrow
        // ends at the closure boundary and the compiler enforces it (bd-88yj).
        let snapshot = self.read_snapshot()?;
        let mut carnivores = Vec::with_capacity(snapshot.world.agents.len() / 2 + 1);
        let mut oldest = Vec::with_capacity(snapshot.world.agents.len());

        for agent in &snapshot.world.agents {
            let diet_core = DietClass::from_tendency(agent.herbivore_tendency);
            let diet = DietClassDto::from(diet_core);

            let entry = AgentScoreEntry {
                agent_id: agent.id,
                energy: agent.energy,
                health: agent.health,
                age: agent.age,
                generation: agent.generation.0,
                diet,
            };

            if matches!(diet_core, DietClass::Carnivore) {
                carnivores.push(entry.clone());
            }
            oldest.push(entry);
        }

        if limit == 0 {
            return Ok(Scoreboard {
                top_predators: Vec::new(),
                oldest: Vec::new(),
            });
        }

        partial_top_k(&mut carnivores, limit, cmp_score);
        if oldest.len() > limit {
            let nth = limit - 1;
            oldest.select_nth_unstable_by_key(nth, |e| Reverse(e.age));
            oldest.truncate(limit);
            oldest.sort_unstable_by_key(|e| Reverse(e.age));
        } else {
            oldest.sort_unstable_by_key(|e| Reverse(e.age));
        }

        Ok(Scoreboard {
            top_predators: carnivores,
            oldest,
        })
    }

    /// Apply a structured JSON patch object onto the configuration.
    pub fn apply_patch(&self, patch: Value) -> Result<CommandStatusDto, ControlError> {
        if !patch.is_object() {
            return Err(ControlError::InvalidPatch(
                "configuration patch must be a JSON object".into(),
            ));
        }

        // Every world read this function needs, captured ATOMICALLY in one
        // borrow. The original held the lock from here through validation,
        // merge and deserialization, reading config again near the end; holding
        // one guard is what made those two reads consistent. Capturing both up
        // front preserves that consistency exactly while moving the expensive
        // work - a JSON merge and a full config deserialization - off the lock
        // instead of running it under one (bd-88yj).
        let snapshot = self.read_snapshot()?;
        let (config_value, current_dims, current_bounds) = {
            let config = &snapshot.config;
            (
                serde_json::to_value(config.as_ref()),
                (snapshot.layers.food.width, snapshot.layers.food.height),
                (
                    config.world_width,
                    config.world_height,
                    config.food_cell_size,
                ),
            )
        };
        let mut config_value = config_value.map_err(ControlError::serialization)?;
        // Range-check the REQUESTED knobs before merging them. `validate()`
        // proves admissibility (finite, non-negative) but declares no upper
        // bounds, so `food_regrowth_rate = 1e9` used to sail through from REST,
        // from MCP, and therefore from any agent driving them. Every violation
        // is reported at once: a caller who has to fix one knob per round trip
        // gives up, and an autonomous one burns its entire budget doing it.
        let requested = flatten_numeric_assignments(&patch);
        let violations = check_knob_ranges(&requested);
        if !violations.is_empty() {
            let detail = violations
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ");
            return Err(ControlError::InvalidPatch(detail));
        }

        let mut path = SmallVec::<[&str; 8]>::new();
        merge_value(&mut config_value, &patch, &mut path)?;
        let json_str = serde_json::to_string(&config_value).map_err(ControlError::serialization)?;
        let mut de = serde_json::Deserializer::from_str(&json_str);
        let new_config: ScriptBotsConfig = serde_path_to_error::deserialize::<_, ScriptBotsConfig>(
            &mut de,
        )
        .map_err(|e: serde_path_to_error::Error<serde_json::Error>| {
            ControlError::InvalidPatch(format!("{} at {}", e, e.path()))
        })?;
        let (food_w, food_h) = new_config
            .food_dimensions()
            .map_err(|err| ControlError::InvalidPatch(err.to_string()))?;
        let (current_width, current_height, current_cell_size) = current_bounds;
        if current_dims != (food_w, food_h)
            || new_config.world_width != current_width
            || new_config.world_height != current_height
            || new_config.food_cell_size != current_cell_size
        {
            return Err(ControlError::InvalidPatch(
                "changing world dimensions at runtime is not supported; restart the simulation with the new configuration"
                    .into(),
            ));
        }
        // No explicit drop is needed any more: the borrow ended at the seam
        // above, so the submit below - which reads tick and revision under the
        // same non-reentrant mutex - cannot deadlock against a guard this
        // function is still holding.
        // Return the receipt, not a projection. This used to build a
        // ConfigSnapshot from the REQUESTED config and hand it back as though
        // it were current, so a client was told the new configuration was in
        // effect when the command had only been admitted to a bounded queue. It
        // was worse than a plain projection: the requested config was stamped
        // with `current_tick`, the tick at which those values were NOT in
        // effect, making the response a chimera of a config that had not been
        // applied and a tick at which it had not been applied (bd-k7nq).
        //
        // Acceptance criterion 4 of the migration is explicit that reads use
        // immutable snapshots and never project future config, so the caller
        // now gets a command id it can poll and reads the configuration back
        // through /api/config when it wants the applied truth.
        let status =
            self.submit_control_command(ControlCommand::UpdateConfig(Box::new(new_config)))?;
        *lock_cache(&self.knobs_cache) = None;
        Ok(status)
    }

    /// Apply a list of knob updates by path.
    pub fn apply_updates(&self, updates: &[KnobUpdate]) -> Result<CommandStatusDto, ControlError> {
        let mut patch_map = Map::new();
        for update in updates {
            insert_path(&mut patch_map, &update.path, update.value.clone())?;
        }
        self.apply_patch(Value::Object(patch_map))
    }

    /// Pause simulation ticks.
    pub fn pause(&self, idempotency_key: Option<&str>) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command_with_key(ControlCommand::Pause, idempotency_key)
    }

    /// Resume simulation ticks.
    pub fn resume(&self, idempotency_key: Option<&str>) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command_with_key(ControlCommand::Resume, idempotency_key)
    }

    /// Step simulation by one tick.
    pub fn step(&self) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command(ControlCommand::Step)
    }

    /// Set simulation playback speed multiplier.
    pub fn set_speed(
        &self,
        speed: f32,
        idempotency_key: Option<&str>,
    ) -> Result<CommandStatusDto, ControlError> {
        if !speed.is_finite() || speed < 0.0 {
            return Err(ControlError::InvalidPatch(
                "invalid speed multiplier".into(),
            ));
        }
        self.submit_control_command_with_key(ControlCommand::SetSpeed(speed), idempotency_key)
    }

    /// Issue graceful shutdown command.
    pub fn shutdown(&self) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command(ControlCommand::Shutdown)
    }

    /// Look up status of a command by ID.
    pub fn command_status(
        &self,
        command_id: &str,
    ) -> Result<Option<CommandStatusDto>, ControlError> {
        let id: CommandId = serde_json::from_value(Value::String(command_id.to_owned()))
            .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
        self.host
            .clone()
            .command_status(id)?
            .map(|status| self.status_dto(status))
            .transpose()
    }

    /// Submit a command, honouring an idempotency key when one is supplied.
    ///
    /// [`scriptbots_runtime::HostPort::submit`] is documented as "submit or
    /// retry a logical command", with the command id as a "stable idempotency
    /// key" whose retry "returns its existing status". This path met none of
    /// that: ids were minted from a server-side counter, so a client that timed
    /// out and retried submitted a SECOND command. For `Pause` that is
    /// harmless, but a retried `Step` advances the simulation twice and a
    /// retried config patch applies twice — the client cannot tell, because
    /// both attempts return a cheerful receipt with different ids (bd-k7nq).
    ///
    /// Keyed retries always reach the authoritative host ledger: the same
    /// envelope reuses its identity, while a changed payload is a typed conflict.
    fn submit_control_command_with_key(
        &self,
        cmd: ControlCommand,
        idempotency_key: Option<&str>,
    ) -> Result<CommandStatusDto, ControlError> {
        cmd.validate()
            .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
        let command = HostCommand::try_from(cmd)
            .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
        let id = if let Some(key) = idempotency_key {
            if key.is_empty() || key.len() > 1024 {
                return Err(ControlError::InvalidPatch(
                    "idempotency key must contain 1..=1024 bytes".into(),
                ));
            }
            let mut hasher = blake3::Hasher::new_derive_key("scriptbots.control.idempotency.v1");
            hasher.update(&self.host.session_id().get().to_le_bytes());
            hasher.update(key.as_bytes());
            let mut bytes = [0_u8; 16];
            bytes.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
            u128::from_le_bytes(bytes) | (1_u128 << 127)
        } else {
            let sequence = self
                .command_counter
                .fetch_update(
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                    |value| value.checked_add(1),
                )
                .map_err(|_| {
                    ControlError::InvalidPatch("command identity sequence exhausted".into())
                })?;
            (u128::from(self.command_namespace) << 64) | u128::from(sequence)
        };
        let status = self
            .host
            .clone()
            .submit(CommandEnvelope::new(CommandId::new(id), command))?;
        self.status_dto(status)
    }

    fn status_dto(
        &self,
        status: scriptbots_runtime::CommandStatus,
    ) -> Result<CommandStatusDto, ControlError> {
        let revisions = match status.application() {
            ApplicationState::Applied(applied) => applied.revisions,
            _ => self.read_snapshot()?.revisions,
        };
        Ok(CommandStatusDto {
            command_id: status.command_id().to_string(),
            admission_sequence: status.admission_sequence().map(|sequence| sequence.get()),
            application_state: match status.application() {
                ApplicationState::Admitted => "admitted",
                ApplicationState::Applied(_) => "applied",
                ApplicationState::Rejected(_) => "rejected",
                ApplicationState::Failed(_) => "failed",
            }
            .to_owned(),
            journal_state: match status.journal() {
                JournalState::NotRequired => "not_required",
                JournalState::Pending => "pending",
                JournalState::CommittedVolatile => "committed_volatile",
                JournalState::Durable => "durable",
                JournalState::Failed(_) => "failed",
            }
            .to_owned(),
            control_revision: revisions.control.get(),
            scientific_revision: revisions.scientific.get(),
        })
    }

    /// Submit a command without an idempotency key.
    ///
    /// Every existing caller keeps its previous behaviour: a retry here is a
    /// new command, because the caller supplied nothing to recognise it by.
    fn submit_control_command(
        &self,
        cmd: ControlCommand,
    ) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command_with_key(cmd, None)
    }

    /// Submit any control command, optionally keyed for safe retry.
    ///
    /// This is the [`scriptbots_runtime::HostPort`]-shaped entry point: one
    /// command, one optional stable key, one receipt. Surfaces that can carry a
    /// client-supplied key (an `Idempotency-Key` header, an MCP argument)
    /// should use it so a timeout-and-retry cannot double-apply. The
    /// command-specific helpers below remain for callers that have no key to
    /// offer.
    pub fn submit_command(
        &self,
        command: ControlCommand,
        idempotency_key: Option<&str>,
    ) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command_with_key(command, idempotency_key)
    }
}

fn insert_path(map: &mut Map<String, Value>, path: &str, value: Value) -> Result<(), ControlError> {
    let mut segments = path.split('.').filter(|s| !s.is_empty());
    let Some(mut seg) = segments.next() else {
        return Err(ControlError::InvalidPatch("empty knob path".into()));
    };
    let mut cur = map;

    for next in segments {
        // Always use Entry API to avoid double-borrow; require objects for intermediate segments
        let entry = cur
            .entry(seg.to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        cur = entry.as_object_mut().ok_or_else(|| {
            ControlError::InvalidPatch(format!("intermediate segment '{seg}' is not an object"))
        })?;
        seg = next;
    }

    cur.insert(seg.to_owned(), value);
    Ok(())
}

fn path_display(path: &[&str]) -> String {
    path.join(".")
}

fn set_f64(target: &mut Value, v: f64, path: &[&str]) -> Result<(), ControlError> {
    if !v.is_finite() {
        return Err(ControlError::InvalidPatch(format!(
            "non-finite float at {}",
            path_display(path)
        )));
    }
    *target = Value::Number(serde_json::Number::from_f64(v).expect("checked finite above"));
    Ok(())
}

fn merge_value<'a>(
    target: &mut Value,
    patch: &'a Value,
    path: &mut SmallVec<[&'a str; 8]>,
) -> Result<(), ControlError> {
    match target {
        Value::Object(target_map) => {
            let Value::Object(patch_map) = patch else {
                return Err(ControlError::InvalidPatch(format!(
                    "type mismatch at {}",
                    path_display(path),
                )));
            };

            for (key, patch_value) in patch_map {
                path.push(key);
                let Some(target_value) = target_map.get_mut(key) else {
                    return Err(ControlError::UnknownPath(path_display(path)));
                };
                merge_value(target_value, patch_value, path)?;
                path.pop();
            }
            Ok(())
        }
        Value::Array(_) => {
            if matches!(patch, Value::Array(_)) {
                *target = patch.clone();
                Ok(())
            } else {
                Err(ControlError::InvalidPatch(format!(
                    "type mismatch at {}",
                    path_display(path),
                )))
            }
        }
        Value::Number(_) => match patch {
            Value::Number(n) => {
                *target = Value::Number(n.clone());
                Ok(())
            }
            Value::String(s) => {
                let s = s.trim();
                if target.as_i64().is_some() {
                    let v: i64 = s
                        .parse()
                        .map_err(|_| ControlError::InvalidPatch(path_display(path)))?;
                    *target = Value::from(v);
                } else if target.as_u64().is_some() {
                    let v: u64 = s
                        .parse()
                        .map_err(|_| ControlError::InvalidPatch(path_display(path)))?;
                    *target = Value::from(v);
                } else {
                    let v: f64 = s
                        .parse()
                        .map_err(|_| ControlError::InvalidPatch(path_display(path)))?;
                    set_f64(target, v, path)?;
                }
                Ok(())
            }
            Value::Null => {
                *target = Value::Null;
                Ok(())
            }
            _ => Err(ControlError::InvalidPatch(format!(
                "type mismatch at {}",
                path_display(path),
            ))),
        },
        Value::String(_) => match patch {
            Value::String(_) | Value::Null => {
                *target = patch.clone();
                Ok(())
            }
            _ => Err(ControlError::InvalidPatch(format!(
                "type mismatch at {}",
                path_display(path),
            ))),
        },
        Value::Bool(_) => match patch {
            Value::Bool(_) | Value::Null => {
                *target = patch.clone();
                Ok(())
            }
            Value::String(_) => {
                let parsed = match patch.as_str().map(|s| s.trim().to_ascii_lowercase()) {
                    Some(s) if matches!(s.as_str(), "true" | "1" | "yes" | "on" | "t" | "y") => {
                        true
                    }
                    Some(s) if matches!(s.as_str(), "false" | "0" | "no" | "off" | "f" | "n") => {
                        false
                    }
                    _ => {
                        return Err(ControlError::InvalidPatch(format!(
                            "cannot coerce '{:?}' to bool for {}",
                            patch,
                            path_display(path),
                        )));
                    }
                };
                *target = Value::from(parsed);
                Ok(())
            }
            _ => Err(ControlError::InvalidPatch(format!(
                "type mismatch at {}",
                path_display(path),
            ))),
        },
        Value::Null => {
            *target = patch.clone();
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Birth,
    Death,
    Combat,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct EventEntry {
    pub tick: u64,
    pub kind: EventKind,
    pub count: u32,
}

impl EventEntry {
    pub fn new(tick: u64, kind: EventKind, count: u32) -> Self {
        Self { tick, kind, count }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DietClassDto {
    Herbivore,
    Omnivore,
    Carnivore,
}

impl From<DietClass> for DietClassDto {
    fn from(value: DietClass) -> Self {
        match value {
            DietClass::Herbivore => Self::Herbivore,
            DietClass::Omnivore => Self::Omnivore,
            DietClass::Carnivore => Self::Carnivore,
        }
    }
}

impl From<DietClassDto> for DietClass {
    fn from(value: DietClassDto) -> Self {
        match value {
            DietClassDto::Herbivore => DietClass::Herbivore,
            DietClassDto::Omnivore => DietClass::Omnivore,
            DietClassDto::Carnivore => DietClass::Carnivore,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, utoipa::ToSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStateDto {
    None,
    Hovered,
    #[default]
    Selected,
}

impl From<SelectionState> for SelectionStateDto {
    fn from(value: SelectionState) -> Self {
        match value {
            SelectionState::None => Self::None,
            SelectionState::Hovered => Self::Hovered,
            SelectionState::Selected => Self::Selected,
        }
    }
}

impl From<SelectionStateDto> for SelectionState {
    fn from(value: SelectionStateDto) -> Self {
        match value {
            SelectionStateDto::None => SelectionState::None,
            SelectionStateDto::Hovered => SelectionState::Hovered,
            SelectionStateDto::Selected => SelectionState::Selected,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum SelectionModeDto {
    Replace,
    Add,
    Clear,
}

impl From<SelectionModeDto> for SelectionMode {
    fn from(value: SelectionModeDto) -> Self {
        match value {
            SelectionModeDto::Replace => SelectionMode::Replace,
            SelectionModeDto::Add => SelectionMode::Add,
            SelectionModeDto::Clear => SelectionMode::Clear,
        }
    }
}

impl From<SelectionMode> for SelectionModeDto {
    fn from(value: SelectionMode) -> Self {
        match value {
            SelectionMode::Replace => SelectionModeDto::Replace,
            SelectionMode::Add => SelectionModeDto::Add,
            SelectionMode::Clear => SelectionModeDto::Clear,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct AgentScoreEntry {
    pub agent_id: u64,
    pub energy: f32,
    pub health: f32,
    pub age: u32,
    pub generation: u32,
    pub diet: DietClassDto,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Scoreboard {
    pub top_predators: Vec<AgentScoreEntry>,
    pub oldest: Vec<AgentScoreEntry>,
}

fn cmp_score(a: &AgentScoreEntry, b: &AgentScoreEntry) -> std::cmp::Ordering {
    b.energy
        .total_cmp(&a.energy)
        .then_with(|| b.health.total_cmp(&a.health))
        .then_with(|| b.age.cmp(&a.age))
}

fn partial_top_k<T, F: Fn(&T, &T) -> std::cmp::Ordering>(v: &mut Vec<T>, k: usize, cmp: F) {
    if v.len() <= k {
        v.sort_by(cmp);
        return;
    }
    let nth = k.saturating_sub(1);
    v.select_nth_unstable_by(nth, &cmp);
    v.truncate(k);
    v.sort_by(cmp);
}

/// Flatten a JSON patch into dotted-path numeric assignments for range checking.
///
/// Only numbers are collected: strings, booleans and structural values are left
/// to serde, which already rejects type mismatches with a precise path.
fn flatten_numeric_assignments(patch: &Value) -> Vec<(String, f64)> {
    fn walk(prefix: &str, value: &Value, out: &mut Vec<(String, f64)>) {
        match value {
            Value::Object(map) => {
                for (key, child) in map {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    walk(&path, child, out);
                }
            }
            Value::Number(number) => {
                if let Some(as_f64) = number.as_f64() {
                    out.push((prefix.to_owned(), as_f64));
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk("", patch, &mut out);
    out
}

fn flatten_value(prefix: &mut String, value: &Value, entries: &mut Vec<KnobEntry>) {
    match value {
        Value::Object(map) => {
            let base = prefix.len();
            for (k, v) in map {
                if base != 0 {
                    prefix.push('.');
                }
                prefix.push_str(k);
                flatten_value(prefix, v, entries);
                prefix.truncate(base);
            }
        }
        _ => entries.push(KnobEntry {
            path: prefix.clone(),
            kind: knob_kind(value),
            value: value.clone(),
            description: None,
        }),
    }
}

fn knob_kind(value: &Value) -> KnobKind {
    match value {
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                KnobKind::Integer
            } else {
                KnobKind::Number
            }
        }
        Value::String(_) => KnobKind::String,
        Value::Bool(_) => KnobKind::Boolean,
        Value::Array(_) => KnobKind::Array,
        Value::Object(_) => KnobKind::Object,
        Value::Null => KnobKind::Null,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use slotmap::{Key, KeyData};
    use std::sync::{Arc, Mutex};

    pub(crate) struct TestHost {
        pub(crate) port: ChannelHostPort,
        worker: Option<std::thread::JoinHandle<()>>,
    }

    impl TestHost {
        pub(crate) fn spawn(world: WorldState) -> Self {
            let (send, receive) = std::sync::mpsc::sync_channel(1);
            let worker = std::thread::spawn(move || {
                use scriptbots_runtime::{
                    FixedDeadlineHost, HostCore, HostCoreOptions, HostSessionId, ManualInstant,
                    PlaybackSnapshot,
                    channel::{ChannelHostDriver, ChannelHostOptions},
                };
                let core = HostCore::new(
                    HostSessionId::new(0xc017),
                    world,
                    HostCoreOptions {
                        initial_playback: PlaybackSnapshot {
                            paused: true,
                            speed_multiplier: 1.0,
                        },
                        capture_agent_visuals: true,
                        ..HostCoreOptions::default()
                    },
                )
                .expect("test host");
                let (mut driver, port) = ChannelHostDriver::new(
                    FixedDeadlineHost::new(core),
                    ChannelHostOptions::default(),
                )
                .expect("channel owner");
                send.send(port).expect("publish test port");
                let epoch = std::time::Instant::now();
                driver
                    .run(|| {
                        ManualInstant::from_nanos(
                            u64::try_from(epoch.elapsed().as_nanos()).expect("test duration"),
                        )
                    })
                    .expect("test owner run");
            });
            Self {
                port: receive.recv().expect("test host rendezvous"),
                worker: Some(worker),
            }
        }

        pub(crate) fn handle(&self) -> ControlHandle {
            ControlHandle::new(self.port.clone())
        }

        pub(crate) fn wait_applied(&self, status: &CommandStatusDto) -> CommandStatusDto {
            let handle = self.handle();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                let observed = handle
                    .command_status(&status.command_id)
                    .expect("host status")
                    .expect("retained command");
                if observed.application_state != "admitted" && observed.journal_state != "pending" {
                    assert_eq!(observed.application_state, "applied");
                    return observed;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "host command did not finish"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    impl Drop for TestHost {
        fn drop(&mut self) {
            let _ = self.port.submit(CommandEnvelope::new(
                CommandId::new(u128::MAX - 1),
                HostCommand::Shutdown,
            ));
            if let Some(worker) = self.worker.take() {
                worker.join().expect("test host joined");
            }
        }
    }

    fn handle() -> (ControlHandle, TestHost) {
        let host = TestHost::spawn(WorldState::new(ScriptBotsConfig { rng_seed: Some(42), ..ScriptBotsConfig::default() }).expect("world"));
        (host.handle(), host)
    }

    fn drain_and_apply(receiver: &crate::command::CommandReceiver, world: &mut WorldState) {
        for bus in crate::command::drain_pending_commands(receiver) {
            let _ = scriptbots_core::apply_control_command(world, bus.command)
                .expect("drained test command applies");
        }
    }

    fn read_status_before_releasing_owner(
        handle: ControlHandle,
        owner: MutexGuard<'_, WorldState>,
    ) -> Result<SimulationStatusDto, ControlError> {
        let (reply, receipt) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || reply.send(handle.status()));
        let result = receipt.recv_timeout(std::time::Duration::from_secs(2));
        drop(owner);
        reader
            .join()
            .expect("status reader joins")
            .expect("reply retained");
        result.expect("status must return while the owner still holds its lock")
    }

    #[test]
    fn status_contention_serves_observed_fields_and_refreshes_after_release() {
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0x513A),
            persistence_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 0,
            closed: true,
            ..ScriptBotsConfig::default()
        })
        .expect("status world");
        world.step().expect("first observed tick");
        world
            .try_inject_agent(scriptbots_core::AgentData::default())
            .expect("one observed agent");
        let mut config = world.config().clone();
        config.closed = false;
        world.apply_config_update(config).expect("first revision");
        let expected = SimulationStatusDto {
            tick: 1,
            agent_count: 1,
            is_closed: false,
            config_revision: 1,
        };
        assert_eq!(SimulationStatusDto::from_world(&world), expected);
        let slot = empty_latest_summary();
        publish_world_observation(&slot, &world);
        let shared_world = Arc::new(Mutex::new(world));
        let (sender, _receiver) = crate::command::create_command_bus(4);
        let handle = ControlHandle::new(Arc::clone(&shared_world), sender, Arc::clone(&slot));

        let mut owner = shared_world.lock().expect("hold the actual owner mutex");
        let mut config = owner.config().clone();
        config.closed = true;
        owner.apply_config_update(config).expect("second revision");
        assert!(matches!(
            shared_world.try_lock(),
            Err(TryLockError::WouldBlock)
        ));
        let current = SimulationStatusDto::from_world(&owner);
        assert_eq!(current.config_revision, 2);
        assert!(current.is_closed);
        assert_ne!(
            current, expected,
            "the cached observation must actually be stale"
        );
        assert_eq!(
            read_status_before_releasing_owner(handle.clone(), owner)
                .expect("nonblocking observed status"),
            expected
        );
        assert_eq!(handle.status().expect("current unlocked status"), current);

        let owner = shared_world
            .lock()
            .expect("republish the actual owner boundary");
        publish_world_observation(&slot, &owner);
        assert_eq!(
            read_status_before_releasing_owner(handle, owner).expect("refreshed observed status"),
            current
        );
    }

    #[test]
    fn status_without_an_observation_refuses_a_busy_world() {
        let (handle, _receiver) = handle();
        let owner = handle.shared_world.lock().expect("hold unobserved world");
        assert!(handle.latest_summary.load_full().is_none());
        assert!(matches!(
            read_status_before_releasing_owner(handle.clone(), owner),
            Err(ControlError::Lock)
        ));
    }

    /// bd-134: a published summary is served wait-free — even a POISONED world
    /// mutex must not take the latest-summary endpoint (and the SSE/NDJSON
    /// streams built on it) down with it.
    #[test]
    fn latest_summary_reads_the_published_slot_without_the_world_mutex() {
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0xB134_5EED),
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .expect("world");
        world.step().expect("persistence-disabled step");
        let published = world
            .history()
            .next_back()
            .expect("completed tick summary")
            .clone();

        let slot = empty_latest_summary();
        publish_world_observation(&slot, &world);
        let shared_world: SharedWorld = Arc::new(Mutex::new(world));
        let (sender, _receiver) = crate::command::create_command_bus(4);
        let handle = ControlHandle::new(Arc::clone(&shared_world), sender, slot);

        // Poison the world mutex on purpose.
        let poisoner = Arc::clone(&shared_world);
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.lock().expect("pre-poison lock");
            panic!("deliberate poison for bd-134 latest-summary test");
        })
        .join();
        assert!(
            shared_world.lock().is_err(),
            "the world mutex must actually be poisoned for this test to prove anything"
        );

        let served = handle
            .latest_summary()
            .expect("published summary served despite the poisoned mutex");
        assert_eq!(served, published);

        // Endpoints that genuinely need the world still fail typed.
        assert!(matches!(handle.snapshot(), Err(ControlError::Lock)));
        assert!(matches!(handle.status(), Err(ControlError::Lock)));
    }

    /// bd-2t3k: the derived caches are not scientific state, so one unrelated panic
    /// while holding either of them must not permanently disable `/api/knobs`,
    /// `/api/config`, or `/api/status`. Before the fix these call sites unwrapped the
    /// guard, so every later request panicked its axum worker.
    #[test]
    fn poisoned_derived_caches_keep_serving_knobs_and_command_status() {
        let (handle, _receiver) = handle();

        let before = handle.list_knobs().expect("knobs list before poisoning");
        assert!(!before.is_empty(), "config must flatten to some knobs");
        let issued = handle.pause(None).expect("pause command accepted");

        let knobs_poisoner = Arc::clone(&handle.knobs_cache);
        let _ = std::thread::spawn(move || {
            let _guard = knobs_poisoner.lock().expect("pre-poison knobs cache");
            panic!("deliberate poison for the bd-2t3k knobs-cache test");
        })
        .join();
        let status_poisoner = Arc::clone(&handle.status_cache);
        let _ = std::thread::spawn(move || {
            let _guard = status_poisoner.lock().expect("pre-poison status cache");
            panic!("deliberate poison for the bd-2t3k status-cache test");
        })
        .join();
        assert!(
            handle.knobs_cache.lock().is_err() && handle.status_cache.lock().is_err(),
            "both cache mutexes must actually be poisoned for this test to prove anything"
        );

        let after = handle
            .list_knobs()
            .expect("knobs still served from a poisoned cache");
        assert_eq!(
            serde_json::to_value(&after).expect("knobs serialize"),
            serde_json::to_value(&before).expect("knobs serialize")
        );
        let looked_up = handle
            .command_status(&issued.command_id)
            .expect("status still served from a poisoned cache")
            .expect("the issued command is cached");
        assert_eq!(looked_up.command_id, issued.command_id);
        // Writes recover too: a later command must still land in the status cache.
        let next = handle.resume(None).expect("resume command accepted");
        assert!(
            handle
                .command_status(&next.command_id)
                .expect("lookup after a poisoned write")
                .is_some(),
            "a command issued after poisoning must still be recorded"
        );
    }

    #[test]
    fn patch_updates_single_field() {
        let (handle, receiver) = handle();
        let updates = vec![KnobUpdate {
            path: "food_max".to_string(),
            value: Value::from(0.6),
        }];
        let receipt = handle.apply_updates(&updates).expect("patch");

        // A receipt, not a projection. This test used to read food_max ≈ 0.6
        // out of the RETURNED value and call that success, which encoded the
        // defect as the contract: the response was built from the requested
        // config and stamped with the tick at which it was not yet in effect
        // (bd-k7nq).
        assert!(
            receipt.admission_sequence.is_some(),
            "a config update must report the order it took on the bus"
        );

        // The world has NOT changed yet. This is the assertion the old shape
        // could not make, because it was handed the requested config and had
        // nothing to compare against.
        {
            let world = handle.lock_world().expect("world lock");
            assert!(
                (world.config().food_max - 0.5).abs() < f32::EPSILON,
                "config changed before the command was drained; the update was only admitted"
            );
        }

        let mut world = handle.lock_world().expect("world lock");
        drain_and_apply(&receiver, &mut world);
        assert!(
            (world.config().food_max - 0.6).abs() < f32::EPSILON,
            "draining the admitted command must actually apply it"
        );
    }

    #[test]
    fn patch_render_quality_and_post_stack_round_trip() {
        let (handle, receiver) = handle();
        let snapshot = handle
            .apply_patch(serde_json::json!({
                "render": {
                    "quality": "high",
                    "theme": "nordic_frost",
                    "palette": "tritanopia",
                    "post": {
                        "bloom": { "enabled": false, "threshold": 1.2 },
                        "fog": { "mode": "low" }
                    },
                    "day_night": { "cycle_ticks": 24000, "stars": true }
                }
            }))
            .expect("render patch applies");
        assert!(
            snapshot.admission_sequence.is_some(),
            "the render patch must report its admission order"
        );

        // Read the round trip back from the world AFTER draining, rather than
        // from the response. The response no longer echoes the requested config
        // (bd-k7nq), and reading it back through the authoritative path is what
        // actually proves the patch survived serialization.
        {
            let mut world = handle.lock_world().expect("world lock");
            drain_and_apply(&receiver, &mut world);
        }
        let applied = handle.snapshot().expect("config after drain");
        let render = &applied.config["render"];
        assert_eq!(render["quality"], serde_json::json!("high"));
        assert_eq!(render["theme"], serde_json::json!("nordic_frost"));
        assert_eq!(render["palette"], serde_json::json!("tritanopia"));
        assert_eq!(render["post"]["bloom"]["enabled"], serde_json::json!(false));
        assert!(
            (render["post"]["bloom"]["threshold"]
                .as_f64()
                .expect("threshold")
                - 1.2)
                .abs()
                < 1e-6
        );
        assert_eq!(render["post"]["fog"]["mode"], serde_json::json!("low"));
        assert_eq!(render["day_night"]["cycle_ticks"], serde_json::json!(24000));

        // The applied world config carries the same values after the command drains.
        let mut world = handle.lock_world().expect("world lock");
        drain_and_apply(&receiver, &mut world);
        assert_eq!(
            world.config().render.quality,
            Some(scriptbots_core::RenderQuality::High)
        );
        assert_eq!(
            world
                .config()
                .render
                .post
                .as_ref()
                .and_then(|post| post.bloom.as_ref())
                .map(|bloom| bloom.enabled),
            Some(false)
        );
    }

    #[test]
    fn patch_render_rejects_invalid_enum_and_range() {
        let (handle, _receiver) = handle();
        let bad_enum = handle.apply_patch(serde_json::json!({
            "render": { "quality": "ludicrous" }
        }));
        assert!(
            matches!(bad_enum, Err(ControlError::InvalidPatch(_))),
            "unknown quality tier must fail closed: {bad_enum:?}"
        );

        let out_of_range = handle.apply_patch(serde_json::json!({
            "render": { "post": { "bloom": { "enabled": true, "intensity": 2.0 } } }
        }));
        assert!(
            matches!(out_of_range, Err(ControlError::InvalidPatch(_))),
            "bloom intensity 2.0 must be rejected by the knob range table: {out_of_range:?}"
        );

        let bad_nested = handle.apply_patch(serde_json::json!({
            "render": { "day_night": { "start_phase": 1.5 } }
        }));
        assert!(
            matches!(bad_nested, Err(ControlError::InvalidPatch(_))),
            "start_phase 1.5 must be rejected: {bad_nested:?}"
        );
    }

    /// The config response reports only applied state.
    ///
    /// This was a `#[should_panic]` target test carrying "KNOWN DEFECT
    /// bd-2z0.4.1: config response projects unapplied future state". The defect
    /// is fixed under bd-k7nq, so the target is now asserted directly rather
    /// than pinned as known-failing — a should_panic marker that no longer
    /// panics is worse than no test, because it fails for the right reason and
    /// reads like a regression.
    ///
    /// The response can no longer project, structurally: it is a receipt and
    /// carries no config field at all. What is left to prove is that the
    /// authoritative read still shows the OLD value while the command is merely
    /// admitted.
    #[test]
    fn config_response_reports_only_applied_state() {
        let (handle, receiver) = handle();
        let before = handle.snapshot().expect("config before");
        let baseline = before.config["food_max"].as_f64().expect("food_max");

        let receipt = handle
            .apply_updates(&[KnobUpdate {
                path: "food_max".to_owned(),
                value: Value::from(0.6),
            }])
            .expect("accepted config patch");
        assert!(
            receipt.admission_sequence.is_some(),
            "an accepted config patch must report its admission order"
        );

        let observed = handle.snapshot().expect("current config snapshot");
        let observed_food_max = observed.config["food_max"].as_f64().expect("food_max");
        assert!(
            (observed_food_max - baseline).abs() < 1.0e-6,
            "the authoritative read moved on an admitted-but-undrained command: {baseline} -> \
             {observed_food_max}"
        );
        assert!(
            (observed_food_max - 0.6).abs() > 1.0e-6,
            "the read reports the REQUESTED value, which is the projection this fix removed"
        );

        // Positive control: the command is real and does apply once drained, so
        // the assertions above describe timing rather than a dropped command.
        {
            let mut world = handle.lock_world().expect("world lock");
            drain_and_apply(&receiver, &mut world);
        }
        let applied = handle.snapshot().expect("config after drain");
        let applied_food_max = applied.config["food_max"].as_f64().expect("food_max");
        assert!(
            (applied_food_max - 0.6).abs() < 1.0e-6,
            "the admitted command never applied, got {applied_food_max}"
        );
    }

    #[test]
    fn absurd_knob_values_are_rejected_at_the_control_boundary() {
        // The end-to-end proof that the hole is closed: this exact request used
        // to be ACCEPTED. ScriptBotsConfig::validate() checks that values are
        // finite and non-negative but declares no upper bounds, so a growth rate
        // of one billion was admissible from REST, from MCP, and therefore from
        // any agent driving them.
        let (handle, _receiver) = handle();
        let err = handle
            .apply_updates(&[KnobUpdate {
                path: "food_growth_rate".into(),
                value: Value::from(1e9),
            }])
            .expect_err("an absurd growth rate must be refused");
        let message = err.to_string();
        assert!(
            message.contains("food_growth_rate") && message.contains("range"),
            "the rejection must name the knob and its range, got: {message}"
        );
    }

    #[test]
    fn every_violation_in_one_patch_is_reported_at_once() {
        // A caller who must fix one knob per round trip gives up; an autonomous
        // one burns its whole budget doing it.
        let (handle, _receiver) = handle();
        let err = handle
            .apply_updates(&[
                KnobUpdate {
                    path: "food_growth_rate".into(),
                    value: Value::from(1e9),
                },
                KnobUpdate {
                    path: "metabolism_drain".into(),
                    value: Value::from(50.0),
                },
            ])
            .expect_err("both knobs are out of range");
        let message = err.to_string();
        assert!(message.contains("food_growth_rate"), "{message}");
        assert!(message.contains("metabolism_drain"), "{message}");
    }

    #[test]
    fn a_harsh_but_sane_world_is_still_expressible() {
        // The bounds exist to reject the absurd, not to enforce taste: a
        // researcher must still be able to build a brutal world.
        let (handle, receiver) = handle();
        handle
            .apply_updates(&[
                KnobUpdate {
                    path: "metabolism_drain".into(),
                    value: Value::from(0.9),
                },
                KnobUpdate {
                    path: "spike_damage".into(),
                    value: Value::from(9.0),
                },
            ])
            .expect("a hostile world is a legitimate experiment");
        let mut world = handle.lock_world().expect("world lock");
        drain_and_apply(&receiver, &mut world);
    }

    #[test]
    fn unknown_path_errors() {
        let (handle, _receiver) = handle();
        let err = handle
            .apply_updates(&[KnobUpdate {
                path: "does.not.exist".into(),
                value: Value::from(1),
            }])
            .expect_err("unknown path");
        assert!(matches!(err, ControlError::UnknownPath(_)));
    }

    #[test]
    fn bd_yw1j_retired_neighbor_normalizer_is_not_discoverable_or_mutable() {
        let (handle, _receiver) = handle();
        let knobs = handle.list_knobs().expect("list public knobs");
        assert!(
            knobs.iter().all(|knob| knob.path != "sense_max_neighbors"),
            "the retired no-op normalizer must not remain visible as a scientific control"
        );

        let err = handle
            .apply_updates(&[KnobUpdate {
                path: "sense_max_neighbors".into(),
                value: Value::from(12.0),
            }])
            .expect_err("the retired normalizer must fail closed");
        assert!(matches!(
            err,
            ControlError::UnknownPath(ref path) if path == "sense_max_neighbors"
        ));
    }

    #[test]
    fn dimension_updates_are_rejected() {
        let (handle, _receiver) = handle();
        let err = handle
            .apply_updates(&[KnobUpdate {
                path: "world_width".into(),
                value: Value::from(8_000),
            }])
            .expect_err("dimension update should fail");
        match err {
            ControlError::InvalidPatch(message) => {
                assert!(
                    message.contains("changing world dimensions")
                        || message.contains("world dimensions must be divisible"),
                    "unexpected error message: {message}"
                );
            }
            other => panic!("expected InvalidPatch, got {other:?}"),
        }
    }

    #[test]
    fn non_finite_knob_update_is_field_specific_and_not_admitted() {
        let (handle, receiver) = handle();
        let err = handle
            .apply_updates(&[KnobUpdate {
                path: "food_growth_rate".into(),
                value: Value::String("NaN".into()),
            }])
            .expect_err("non-finite string coercion must fail");
        assert!(
            matches!(&err, ControlError::InvalidPatch(_)),
            "expected InvalidPatch, got {err:?}"
        );
        let ControlError::InvalidPatch(message) = err else {
            return;
        };
        assert!(
            message.contains("food_growth_rate"),
            "error did not identify field: {message}"
        );
        assert!(matches!(
            receiver.try_recv(),
            Err(crate::command::CommandRecvError::Empty)
        ));
        let value = handle
            .snapshot()
            .expect("snapshot")
            .config
            .get("food_growth_rate")
            .and_then(Value::as_f64)
            .expect("food_growth_rate");
        assert!(
            (value - f64::from(ScriptBotsConfig::default().food_growth_rate)).abs() < f64::EPSILON,
            "rejected update changed food_growth_rate to {value}"
        );
    }

    #[test]
    fn unrepresentable_nested_float_reports_exact_path_without_partial_admission() {
        let (handle, receiver) = handle();
        let err = handle
            .apply_updates(&[
                KnobUpdate {
                    path: "food_max".into(),
                    value: Value::from(0.6),
                },
                KnobUpdate {
                    path: "render.auto_exposure.enabled".into(),
                    value: Value::from(true),
                },
                KnobUpdate {
                    path: "render.auto_exposure.speed_brighten".into(),
                    value: Value::from(1.0e40_f64),
                },
            ])
            .expect_err("f64 value outside the f32 domain must fail");
        assert!(
            matches!(&err, ControlError::InvalidPatch(_)),
            "expected InvalidPatch, got {err:?}"
        );
        let ControlError::InvalidPatch(message) = err else {
            return;
        };
        assert!(
            message.contains("render.auto_exposure.speed_brighten"),
            "error did not identify nested field: {message}"
        );
        assert!(matches!(
            receiver.try_recv(),
            Err(crate::command::CommandRecvError::Empty)
        ));
        assert_eq!(
            handle.snapshot().expect("snapshot").config.get("food_max"),
            Some(&Value::from(0.5))
        );
    }

    #[test]
    fn full_command_queue_returns_no_optimistic_config_snapshot() {
        let (handle, receiver) = handle();
        for _ in 0..4 {
            handle
                .update_selection(
                    SelectionUpdate {
                        mode: SelectionMode::Clear,
                        agent_ids: Vec::new(),
                        state: SelectionState::None,
                    },
                    None,
                )
                .expect("fill bounded command queue");
        }

        let error = handle
            .apply_updates(&[KnobUpdate {
                path: "food_max".into(),
                value: Value::from(0.6),
            }])
            .expect_err("full queue must reject config update");
        assert!(matches!(error, ControlError::CommandQueueFull));
        assert_eq!(
            handle.snapshot().expect("snapshot").config.get("food_max"),
            Some(&Value::from(0.5))
        );

        let mut queued = 0;
        while receiver.try_recv().is_ok() {
            queued += 1;
        }
        assert_eq!(queued, 4, "invalid optimistic config command reached queue");
    }

    /// END TO END: a real submission becomes `applied` because the applier said so.
    ///
    /// Every earlier test in this file proved a piece. This proves the
    /// behaviour, which is the only thing that was ever claimed to be missing:
    ///
    ///   1. a surface submits through the real ControlHandle path,
    ///   2. the identity travels on the real bounded bus inside `BusCommand`,
    ///   3. the real `scriptbots_core::apply_control_command` applies it,
    ///   4. the APPLIER - not the submitter - reports the outcome through the
    ///      reporter seam, using the id that arrived with the command,
    ///   5. polling the receipt shows `applied`.
    ///
    /// Nothing here simulates a step. The submitter never touches the ledger
    /// after admission, which is the distinction this whole line of work rests
    /// on: an outcome is observed, not assumed (bd-k7nq).
    #[test]
    fn an_applied_command_is_reported_by_the_applier_and_visible_to_the_submitter() {
        let (handle, receiver) = handle();
        let reporter = handle.command_reporter();

        let admitted = handle.pause(None).expect("pause admitted");
        assert_eq!(admitted.application_state, APPLICATION_STATE_ADMITTED);

        // The applier: drains the real bus and applies through core. It holds a
        // reporter and the id, and nothing else from the control plane.
        let mut applied_ids = Vec::new();
        {
            let mut world = handle.lock_world().expect("world lock");
            for bus in crate::command::drain_pending_commands(&receiver) {
                let outcome = match scriptbots_core::apply_control_command(&mut world, bus.command)
                {
                    Ok(_) => CommandOutcome::Applied,
                    Err(_) => CommandOutcome::Rejected,
                };
                reporter(&bus.id, outcome);
                applied_ids.push(bus.id);
            }
        }

        assert_eq!(
            applied_ids,
            vec![admitted.command_id.clone()],
            "the id the applier saw must be the id the submitter was given, or the report \
             cannot be correlated with the submission"
        );

        let polled = handle
            .command_status(&admitted.command_id)
            .expect("lookup")
            .expect("the command is known");
        assert_eq!(
            polled.application_state, APPLICATION_STATE_APPLIED,
            "the receipt did not advance, so `admitted` is still the end of the road"
        );
    }

    /// The same seam records a refusal, and the refusal is the applier's.
    ///
    /// Positive control for the test above: without it, a reporter that wrote
    /// `applied` unconditionally would pass, which would be the submitter
    /// guessing wearing the applier's clothes.
    #[test]
    fn a_rejected_command_is_reported_as_rejected_not_applied() {
        let (handle, _receiver) = handle();
        let reporter = handle.command_reporter();
        let admitted = handle.resume(None).expect("resume admitted");

        reporter(&admitted.command_id, CommandOutcome::Rejected);

        assert_eq!(
            handle
                .command_status(&admitted.command_id)
                .expect("lookup")
                .expect("known")
                .application_state,
            APPLICATION_STATE_REJECTED
        );
    }

    /// A receipt can advance from admitted to applied.
    ///
    /// This is the guarantee that did not exist before: every prior fix taught
    /// a surface to say `admitted` honestly, but nothing could move a command
    /// past it, so a caller had no way to learn what happened next.
    #[test]
    fn a_receipt_advances_from_admitted_to_applied() {
        let (handle, _receiver) = handle();
        let admitted = handle.pause(None).expect("pause admitted");
        assert_eq!(admitted.application_state, APPLICATION_STATE_ADMITTED);

        let applied = handle
            .mark_applied(&admitted.command_id)
            .expect("the applier reports application");
        assert_eq!(applied.application_state, APPLICATION_STATE_APPLIED);
        assert_eq!(
            applied.command_id, admitted.command_id,
            "advancing must not mint a new identity"
        );

        // And the advance is visible to anyone polling the id, which is the
        // only reason a caller was given an identity in the first place.
        let polled = handle
            .command_status(&admitted.command_id)
            .expect("lookup")
            .expect("the command is known");
        assert_eq!(polled.application_state, APPLICATION_STATE_APPLIED);
    }

    /// A terminal outcome is not revisable, and an unknown id is refused.
    ///
    /// Without both, the ledger would let a caller record an outcome it never
    /// observed — the same defect one level down from the one this whole line
    /// of work removed from five surfaces.
    #[test]
    fn the_ledger_refuses_invented_and_revised_outcomes() {
        let (handle, _receiver) = handle();

        let unknown = handle.mark_applied("never-submitted");
        assert!(
            unknown.is_err(),
            "a receipt was advanced for a command that was never submitted"
        );

        let admitted = handle.resume(None).expect("resume admitted");
        handle
            .mark_applied(&admitted.command_id)
            .expect("first report");

        // Replaying the SAME report is a retry, not a conflict.
        let replay = handle
            .mark_applied(&admitted.command_id)
            .expect("an applier replaying its report must not be punished");
        assert_eq!(replay.application_state, APPLICATION_STATE_APPLIED);

        // Changing a terminal outcome is refused.
        let contradiction = handle.mark_rejected(&admitted.command_id);
        assert!(
            contradiction.is_err(),
            "an applied command was quietly re-reported as rejected"
        );
        assert_eq!(
            handle
                .command_status(&admitted.command_id)
                .expect("lookup")
                .expect("known")
                .application_state,
            APPLICATION_STATE_APPLIED,
            "the refused write must not have changed the stored outcome"
        );
    }

    /// A keyed retry returns the original receipt and enqueues nothing.
    ///
    /// The receipt equality alone would be satisfied by a cache that answered
    /// correctly while still submitting a duplicate, so the queue depth is what
    /// actually carries this test: a retried `Step` that reaches the bus twice
    /// advances the simulation twice, and the client cannot tell (bd-k7nq).
    #[test]
    fn a_keyed_retry_does_not_submit_a_second_command() {
        let (handle, receiver) = handle();

        let first = handle
            .submit_command(ControlCommand::Step, Some("client-abc"))
            .expect("first submit");
        let retry = handle
            .submit_command(ControlCommand::Step, Some("client-abc"))
            .expect("retry of the same logical command");

        assert_eq!(first.command_id.len(), CommandId::new(1).to_string().len());
        assert_eq!(
            first.command_id, retry.command_id,
            "a retry must return the original receipt"
        );
        assert_eq!(
            first.admission_sequence, retry.admission_sequence,
            "a retry must not take a second admission order"
        );

        receiver.wait_applied(&retry);
        assert_eq!(handle.status().expect("status").tick, 1, "a retried Step must advance once");
    }

    /// Positive control: distinct keys are distinct commands.
    ///
    /// Without this, the test above would pass against an implementation that
    /// deduplicated everything and only ever submitted once.
    #[test]
    fn distinct_keys_submit_distinct_commands() {
        let (handle, receiver) = handle();

        let first = handle
            .submit_command(ControlCommand::Step, Some("step-1"))
            .expect("first");
        let second = handle
            .submit_command(ControlCommand::Step, Some("step-2"))
            .expect("second");

        assert_ne!(first.command_id, second.command_id);
        assert!(
            second.admission_sequence > first.admission_sequence,
            "distinct commands must take distinct admission orders"
        );

        receiver.wait_applied(&second);
        assert_eq!(handle.status().expect("status").tick, 2, "distinct Steps must both apply");
    }

    /// An unkeyed submit stays non-idempotent, as every existing caller expects.
    #[test]
    fn an_unkeyed_submit_is_still_a_new_command_each_time() {
        let (handle, receiver) = handle();

        let first = handle
            .submit_command(ControlCommand::Step, None)
            .expect("a");
        let second = handle
            .submit_command(ControlCommand::Step, None)
            .expect("b");

        assert_ne!(
            first.command_id, second.command_id,
            "without a key there is nothing to recognise a retry by, so these are two commands"
        );

        receiver.wait_applied(&second);
        assert_eq!(handle.status().expect("status").tick, 2);
    }

    /// A selection submission must hand back a receipt a client can follow.
    ///
    /// Selection used to return `()`, so the REST layer had no identity to
    /// report and answered with a hardcoded `queued: true`. This asserts the
    /// three properties that made that answer useless: there is a command id,
    /// it is distinct per submission, and the receipt is retrievable by that id
    /// afterwards (bd-2z0.4.9).
    #[test]
    fn selection_submission_returns_a_followable_receipt() {
        let (handle, receiver) = handle();
        let update = || SelectionUpdate {
            mode: SelectionMode::Clear,
            agent_ids: Vec::new(),
            state: SelectionState::None,
        };

        let first = handle
            .update_selection(update(), None)
            .expect("first selection");
        let second = handle
            .update_selection(update(), None)
            .expect("second selection");

        assert_ne!(
            first.command_id, second.command_id,
            "two selections must be distinguishable; a client correlating receipts \
             cannot work with a shared id"
        );
        assert!(
            first.admission_sequence.is_some(),
            "an admitted command must report the order it took on the bus"
        );
        assert!(
            second.admission_sequence > first.admission_sequence,
            "admission order must advance, got {:?} then {:?}",
            first.admission_sequence,
            second.admission_sequence
        );

        let looked_up = handle
            .command_status(&first.command_id)
            .expect("status lookup")
            .expect("the receipt must be retrievable by its own id");
        assert_eq!(looked_up.command_id, first.command_id);

        let terminal = receiver.wait_applied(&looked_up);
        assert_eq!(terminal.application_state, APPLICATION_STATE_APPLIED);
        assert_eq!(terminal.journal_state, "committed_volatile");
    }

    #[test]
    fn debug_agents_lists_selection() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let raw_id = {
            let id = world
                .try_spawn_agent(scriptbots_core::AgentData::default())
                .expect("default agent is finite");
            let _ = world.apply_selection_update(SelectionUpdate {
                mode: SelectionMode::Replace,
                agent_ids: vec![id.data().as_ffi()],
                state: SelectionState::Selected,
            });
            id.data().as_ffi()
        };
        let host = TestHost::spawn(world);
        let handle = host.handle();

        let entries = handle
            .debug_agents(AgentDebugQuery {
                selection: Some(SelectionState::Selected),
                ids: Some(vec![raw_id]),
                ..AgentDebugQuery::default()
            })
            .expect("debug agents");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].agent_id, raw_id);
    }

    #[test]
    fn update_selection_enqueues_and_applies() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let raw_id = {
            let id = world
                .try_spawn_agent(scriptbots_core::AgentData::default())
                .expect("default agent is finite");
            id.data().as_ffi()
        };
        let host = TestHost::spawn(world);
        let handle = host.handle();
        let receipt = handle
            .update_selection(
                SelectionUpdate {
                    mode: SelectionMode::Replace,
                    agent_ids: vec![raw_id],
                    state: SelectionState::Selected,
                },
                None,
            )
            .expect("enqueue selection command");

        host.wait_applied(&receipt);
        let entries = handle.debug_agents(AgentDebugQuery { ids: Some(vec![raw_id]), selection: Some(SelectionState::Selected), ..AgentDebugQuery::default() }).expect("selected agent query");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].agent_id, raw_id);
    }

    #[test]
    fn control_commands_generate_status_dtos_and_lookup() {
        let (handle, receiver) = handle();

        let status_pause = handle.pause(None).expect("pause command");
        // Enqueueing proves admission order, nothing more: the driver has not drained
        // this command and the legacy bus journals nothing (bd-f65w).
        assert_eq!(status_pause.application_state, APPLICATION_STATE_ADMITTED);
        assert_eq!(status_pause.journal_state, "pending");
        assert_eq!(status_pause.command_id.len(), CommandId::new(1).to_string().len());

        let status_resume = handle.resume(None).expect("resume command");
        assert_ne!(status_pause.command_id, status_resume.command_id);

        let status_step = handle.step().expect("step command");
        assert_eq!(status_step.application_state, APPLICATION_STATE_ADMITTED);

        let status_speed = handle.set_speed(2.5, None).expect("speed command");
        assert_eq!(status_speed.application_state, APPLICATION_STATE_ADMITTED);

        receiver.wait_applied(&status_speed);

        let status_shutdown = handle.shutdown().expect("shutdown command");
        assert_eq!(
            status_shutdown.application_state,
            APPLICATION_STATE_ADMITTED
        );

        let looked_up = handle
            .command_status(&status_pause.command_id)
            .expect("lookup")
            .expect("found status");
        assert_eq!(looked_up.command_id, status_pause.command_id);

        let non_existent = handle.command_status(&CommandId::new(9999).to_string()).expect("lookup");
        assert!(non_existent.is_none());

        let err = handle
            .set_speed(-1.0, None)
            .expect_err("negative speed must fail");
        assert!(matches!(err, ControlError::InvalidPatch(_)));
    }
}
