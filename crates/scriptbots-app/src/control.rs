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
    HydrologyState, ScriptBotsConfig, SelectionMode, SelectionState, SelectionUpdate, TerrainKind,
    Tick, WorldState,
};

use crate::SharedWorld;
use crate::command::{CommandSendError, CommandSender};
use scriptbots_core::ConfigAuditEntry;
use scriptbots_core::check_knob_ranges;
#[cfg(feature = "gui")]
use scriptbots_render::{OffscreenScene, render_offscreen_scene};
use slotmap::Key; // offscreen PNG renderer
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

    fn from_config(config: ScriptBotsConfig, tick: Tick) -> Result<Self, ControlError> {
        let config_value = serde_json::to_value(config).map_err(ControlError::serialization)?;
        Ok(Self {
            tick: tick.0,
            config: config_value,
        })
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

    fn from_state(state: &HydrologyState) -> Self {
        let total_water_depth = state.total_water_depth();
        let cell_count = state.cell_count().max(1) as f32;
        let (shallow, deep) =
            state.flooded_cell_counts(Self::SHALLOW_THRESHOLD, Self::DEEP_THRESHOLD);

        let flow_directions = state
            .field()
            .flow_directions()
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
            width: state.width(),
            height: state.height(),
            total_water_depth,
            mean_water_depth: total_water_depth / cell_count,
            flooded_shallow_count: saturating_u32(shallow),
            flooded_deep_count: saturating_u32(deep),
            shallow_threshold: Self::SHALLOW_THRESHOLD,
            deep_threshold: Self::DEEP_THRESHOLD,
            water_depth: state.water_depth().to_vec(),
            flow_directions,
            basin_ids: state.field().basin_ids().to_vec(),
            accumulation: state.field().accumulation().to_vec(),
            spill_elevation: state.field().spill_elevation().to_vec(),
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

/// Latest completed tick summary, published by the simulation step drivers
/// outside the world mutex (bd-134).
///
/// Reads are wait-free: control surfaces serve `latest_summary` from this slot
/// even while the world mutex is held by a long tick — or poisoned outright.
pub type SharedLatestSummary = Arc<ArcSwapOption<scriptbots_core::TickSummary>>;

/// Fresh, empty published-summary slot.
#[must_use]
pub fn empty_latest_summary() -> SharedLatestSummary {
    Arc::new(ArcSwapOption::empty())
}

/// Shared handle used by REST, CLI, and MCP surfaces to access the running world.
#[derive(Clone)]
pub struct ControlHandle {
    shared_world: SharedWorld,
    commands: CommandSender,
    knobs_cache: KnobsCache,
    latest_summary: SharedLatestSummary,
}

impl ControlHandle {
    pub fn new(
        shared_world: SharedWorld,
        commands: CommandSender,
        latest_summary: SharedLatestSummary,
    ) -> Self {
        Self {
            shared_world,
            commands,
            knobs_cache: std::sync::Arc::new(Mutex::new(None)),
            latest_summary,
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
            // Capture the render-relevant state under a short lock, then
            // rasterize with no lock held at all (bd-134): a slow PNG render
            // must never stall the simulation or other control reads.
            let scene = {
                let world = self.lock_world()?;
                OffscreenScene::capture(&world)
            };
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

    fn lock_world(&self) -> Result<MutexGuard<'_, WorldState>, ControlError> {
        self.shared_world.lock().map_err(|err| err.into())
    }

    /// Retrieve the current configuration snapshot.
    pub fn snapshot(&self) -> Result<ConfigSnapshot, ControlError> {
        let world = self.lock_world()?;
        ConfigSnapshot::from_world(world.config(), world.tick())
    }

    /// Retrieve the latest tick summary from the running world.
    pub fn latest_summary(&self) -> Result<scriptbots_core::TickSummary, ControlError> {
        // Wait-free published read first (bd-134): the step drivers store each
        // completed summary here, so a contended — or poisoned — world mutex
        // cannot stall this endpoint or the SSE/NDJSON streams built on it.
        if let Some(summary) = self.latest_summary.load_full() {
            return Ok((*summary).clone());
        }
        // Nothing published yet (before the first completed tick, or a driver
        // that does not publish): fall back to the world itself.
        let world = self.lock_world()?;
        if let Some(latest) = world.history().last() {
            Ok(latest.clone())
        } else {
            Ok(scriptbots_core::TickSummary {
                tick: world.tick(),
                agent_count: world.agent_count(),
                births: 0,
                deaths: 0,
                total_energy: 0.0,
                average_energy: 0.0,
                average_health: 0.0,
                max_age: 0,
                spike_hits: 0,
            })
        }
    }

    /// Retrieve a filtered debug listing of agents.
    pub fn debug_agents(
        &self,
        query: AgentDebugQuery,
    ) -> Result<Vec<AgentDebugInfo>, ControlError> {
        let world = self.lock_world()?;
        Ok(world.agent_debug_view(query))
    }

    /// Enqueue a selection update command.
    pub fn update_selection(&self, update: SelectionUpdate) -> Result<(), ControlError> {
        self.enqueue(ControlCommand::UpdateSelection(update))
    }

    /// Retrieve a snapshot of the current hydrology state, if available.
    pub fn hydrology_snapshot(&self) -> Result<Option<HydrologySnapshot>, ControlError> {
        let world = self.lock_world()?;
        Ok(world.hydrology().map(HydrologySnapshot::from_state))
    }

    /// Flatten the configuration into individual knob descriptors for discovery.
    pub fn list_knobs(&self) -> Result<Vec<KnobEntry>, ControlError> {
        let rev = {
            let world = self.lock_world()?;
            world.config_revision()
        };
        if let Some((cached_rev, cached)) = self.knobs_cache.lock().unwrap().as_ref()
            && *cached_rev == rev
        {
            return Ok(cached.clone());
        }
        let (rev2, config_value) = {
            let world = self.lock_world()?;
            let rev2 = world.config_revision();
            let value =
                serde_json::to_value(world.config()).map_err(ControlError::serialization)?;
            (rev2, value)
        };
        let mut entries = Vec::with_capacity(256);
        let mut prefix = String::new();
        flatten_value(&mut prefix, &config_value, &mut entries);
        *self.knobs_cache.lock().unwrap() = Some((rev2, entries.clone()));
        Ok(entries)
    }

    /// Retrieve the configuration audit log accumulated since startup.
    pub fn audit(&self) -> Result<Vec<ConfigAuditEntry>, ControlError> {
        let world = self.lock_world()?;
        Ok(world.config_audit().to_vec())
    }

    /// Build a tail of recent narrative events from the world's tick history.
    /// Events include births, deaths, and combat spike hits.
    pub fn events_tail(&self, limit: usize) -> Result<Vec<EventEntry>, ControlError> {
        let world = self.lock_world()?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        // The limit arrives unclamped from the query string; cap it so a hostile
        // request cannot reserve unbounded memory (history yields ≤3 events/tick).
        let limit = limit.min(world.history().count().saturating_mul(3).max(1));
        let mut events = Vec::with_capacity(limit);
        for summary in world.history().rev() {
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
        let world = self.lock_world()?;
        let food = world.food();
        let terrain = world.terrain();
        let grid_w = food.width().max(1) as usize;
        let grid_h = food.height().max(1) as usize;
        let width = grid_w.clamp(16, 96);
        let height = grid_h.clamp(8, 48);
        let food_max = world.config().food_max.max(f32::EPSILON);
        let world_w = (world.config().world_width as f32).max(1.0);
        let world_h = (world.config().world_height as f32).max(1.0);
        let tiles = terrain.tiles();
        let cells = food.cells();

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
        for pos in world.agents().columns().positions() {
            let x = (((pos.x / world_w) * width as f32) as usize).min(width - 1);
            let y = (((pos.y / world_h) * height as f32) as usize).min(height - 1);
            rows[y][x] = '@';
        }

        let mut out = format!("ScriptBots tick {}\n", world.tick().0);
        for row in rows {
            out.extend(row);
            out.push('\n');
        }
        Ok(out)
    }

    /// Compute scoreboard snapshots: top predators (carnivores) by energy and oldest living agents.
    pub fn compute_scoreboard(&self, limit: usize) -> Result<Scoreboard, ControlError> {
        let world = self.lock_world()?;

        let handles: Vec<scriptbots_core::AgentId> = world.agents().iter_handles().collect();
        let columns = world.agents().columns();
        let runtimes = world.runtime();

        let mut carnivores = Vec::with_capacity(handles.len() / 2 + 1);
        let mut oldest = Vec::with_capacity(handles.len());

        for (idx, id) in handles.iter().enumerate() {
            let runtime = runtimes.get(*id);
            let tendency = runtime.map(|rt| rt.herbivore_tendency).unwrap_or(0.5);
            let diet_core = DietClass::from_tendency(tendency);
            let diet = DietClassDto::from(diet_core);
            let energy = runtime.map(|rt| rt.energy).unwrap_or(0.0);
            let health = columns.health()[idx];
            let age = columns.ages()[idx];
            let generation = columns.generations()[idx].0;

            let entry = AgentScoreEntry {
                agent_id: id.data().as_ffi(),
                energy,
                health,
                age,
                generation,
                diet,
            };

            if matches!(diet_core, DietClass::Carnivore) {
                carnivores.push(entry.clone());
            }
            oldest.push(entry);
        }

        drop(world); // release lock before sorting

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
    pub fn apply_patch(&self, patch: Value) -> Result<ConfigSnapshot, ControlError> {
        if !patch.is_object() {
            return Err(ControlError::InvalidPatch(
                "configuration patch must be a JSON object".into(),
            ));
        }

        let world = self.lock_world()?;
        let current_tick = world.tick();
        let mut config_value =
            serde_json::to_value(world.config()).map_err(ControlError::serialization)?;
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
        let current_dims = (world.food().width(), world.food().height());
        let current = world.config();
        if current_dims != (food_w, food_h)
            || new_config.world_width != current.world_width
            || new_config.world_height != current.world_height
            || new_config.food_cell_size != current.food_cell_size
        {
            return Err(ControlError::InvalidPatch(
                "changing world dimensions at runtime is not supported; restart the simulation with the new configuration"
                    .into(),
            ));
        }
        let snapshot = ConfigSnapshot::from_config(new_config.clone(), current_tick)?;
        drop(world);
        self.enqueue(ControlCommand::UpdateConfig(Box::new(new_config)))?;
        *self.knobs_cache.lock().unwrap() = None;
        Ok(snapshot)
    }

    /// Apply a list of knob updates by path.
    pub fn apply_updates(&self, updates: &[KnobUpdate]) -> Result<ConfigSnapshot, ControlError> {
        let mut patch_map = Map::new();
        for update in updates {
            insert_path(&mut patch_map, &update.path, update.value.clone())?;
        }
        self.apply_patch(Value::Object(patch_map))
    }

    fn enqueue(&self, command: ControlCommand) -> Result<(), ControlError> {
        if let Err(error) = command.validate() {
            self.commands.record_validation_rejection();
            return Err(ControlError::InvalidPatch(error.to_string()));
        }
        match self.commands.try_send(command) {
            Ok(()) => Ok(()),
            Err(CommandSendError::Full(_command)) => Err(ControlError::CommandQueueFull),
            Err(CommandSendError::Disconnected(_command)) => Err(ControlError::CommandQueueClosed),
        }
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
mod tests {
    use super::*;
    use slotmap::{Key, KeyData};
    use std::sync::{Arc, Mutex};

    fn handle() -> (ControlHandle, crate::command::CommandReceiver) {
        let world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let (sender, receiver) = crate::command::create_command_bus(4);
        let handle =
            ControlHandle::new(Arc::new(Mutex::new(world)), sender, empty_latest_summary());
        (handle, receiver)
    }

    fn drain_and_apply(receiver: &crate::command::CommandReceiver, world: &mut WorldState) {
        for command in crate::command::drain_pending_commands(receiver) {
            let _ = scriptbots_core::apply_control_command(world, command)
                .expect("drained test command applies");
        }
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

        let shared_world: SharedWorld = Arc::new(Mutex::new(world));
        let (sender, _receiver) = crate::command::create_command_bus(4);
        let slot = empty_latest_summary();
        slot.store(Some(Arc::new(published.clone())));
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
    }

    #[test]
    fn patch_updates_single_field() {
        let (handle, receiver) = handle();
        let updates = vec![KnobUpdate {
            path: "food_max".to_string(),
            value: Value::from(0.6),
        }];
        let snapshot = handle.apply_updates(&updates).expect("patch");
        let value = snapshot
            .config
            .get("food_max")
            .and_then(Value::as_f64)
            .expect("food_max");
        assert!(
            (value - 0.6).abs() < 1e-6,
            "expected food_max ≈ 0.6 in snapshot, got {value}"
        );

        // ensure queue drained for consistency
        let mut world = handle.lock_world().expect("world lock");
        drain_and_apply(&receiver, &mut world);
        assert!((world.config().food_max - 0.6).abs() < f32::EPSILON);
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

        let render = &snapshot.config["render"];
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

    #[test]
    #[should_panic(
        expected = "KNOWN DEFECT bd-2z0.4.1: config response projects unapplied future state"
    )]
    fn target_config_response_reports_only_applied_state() {
        let (handle, _receiver) = handle();
        let projected = handle
            .apply_updates(&[KnobUpdate {
                path: "food_max".to_owned(),
                value: Value::from(0.6),
            }])
            .expect("accepted config patch");
        let observed = handle.snapshot().expect("current config snapshot");
        let projected_food_max = projected.config["food_max"]
            .as_f64()
            .expect("projected food_max");
        let observed_food_max = observed.config["food_max"]
            .as_f64()
            .expect("observed food_max");

        assert!((projected_food_max - 0.6).abs() < 1.0e-6);
        assert!((observed_food_max - 0.6).abs() > 1.0e-6);
        assert_eq!(
            projected_food_max, observed_food_max,
            "KNOWN DEFECT bd-2z0.4.1: config response projects unapplied future state"
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
                .update_selection(SelectionUpdate {
                    mode: SelectionMode::Clear,
                    agent_ids: Vec::new(),
                    state: SelectionState::None,
                })
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

    #[test]
    fn debug_agents_lists_selection() {
        let (handle, receiver) = handle();
        let raw_id = {
            let mut world = handle.lock_world().expect("world lock");
            let id = world
                .try_spawn_agent(scriptbots_core::AgentData::default())
                .expect("default agent is finite");
            let _ = world.apply_selection_update(SelectionUpdate {
                mode: SelectionMode::Replace,
                agent_ids: vec![id.data().as_ffi()],
                state: SelectionState::Selected,
            });
            drain_and_apply(&receiver, &mut world);
            id.data().as_ffi()
        };

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
        let (handle, receiver) = handle();
        let raw_id = {
            let mut world = handle.lock_world().expect("world lock");
            let id = world
                .try_spawn_agent(scriptbots_core::AgentData::default())
                .expect("default agent is finite");
            id.data().as_ffi()
        };
        handle
            .update_selection(SelectionUpdate {
                mode: SelectionMode::Replace,
                agent_ids: vec![raw_id],
                state: SelectionState::Selected,
            })
            .expect("enqueue selection command");

        let mut world = handle.lock_world().expect("world lock");
        drain_and_apply(&receiver, &mut world);
        let agent_id = scriptbots_core::AgentId::from(KeyData::from_ffi(raw_id));
        let runtime = world.agent_runtime(agent_id).expect("runtime");
        assert!(matches!(runtime.selection, SelectionState::Selected));
    }
}
