use std::sync::{Mutex, MutexGuard, PoisonError};
use std::cmp::Reverse;
// removed duplicate import

use crossfire::TrySendError;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;
// removed duplicate import

use scriptbots_core::{
    AgentDebugInfo, AgentDebugQuery, ControlCommand, DietClass, HydrologyFlowDirection,
    HydrologyState, ScriptBotsConfig, SelectionMode, SelectionState, SelectionUpdate, Tick,
    WorldState,
};

use crate::SharedWorld;
use crate::command::CommandSender;
use scriptbots_core::ConfigAuditEntry;
#[cfg(feature = "gui")]
use scriptbots_render::render_png_offscreen;
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
        Ok(Self { tick: tick.0, config: config_value })
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
            flooded_shallow_count: shallow as u32,
            flooded_deep_count: deep as u32,
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

type KnobsCache = std::sync::Arc<Mutex<Option<(usize, Vec<KnobEntry>)>>>;

/// Shared handle used by REST, CLI, and MCP surfaces to access the running world.
#[derive(Clone)]
pub struct ControlHandle {
    shared_world: SharedWorld,
    commands: CommandSender,
    knobs_cache: KnobsCache,
}

impl ControlHandle {
    pub fn new(shared_world: SharedWorld, commands: CommandSender) -> Self {
        Self {
            shared_world,
            commands,
            knobs_cache: std::sync::Arc::new(Mutex::new(None)),
        }
    }

    /// Produce a PNG snapshot of the world without a live window.
    pub fn snapshot_png(&self, width: u32, height: u32) -> Result<Vec<u8>, ControlError> {
        #[cfg(feature = "gui")]
        {
            const MAX_PIXELS: u64 = 64 * 1024 * 1024; // 64M px guardrail
            if (width as u64) * (height as u64) > MAX_PIXELS {
                return Err(ControlError::InvalidPatch("requested image too large".into()));
            }
            let world = self.lock_world()?;
            let bytes = render_png_offscreen(&world, width, height);
            Ok(bytes)
        }
        #[cfg(not(feature = "gui"))]
        {
            // Reference params to avoid unused warnings in non-GUI builds
            let _ = (width, height);
            Err(ControlError::InvalidPatch("PNG snapshot requires gui feature".into()))
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
        let rev = { let world = self.lock_world()?; world.config_audit().len() };
        if let Some((cached_rev, cached)) = self.knobs_cache.lock().unwrap().as_ref()
            && *cached_rev == rev
        {
            return Ok(cached.clone());
        }
        let (rev2, config_value) = {
            let world = self.lock_world()?;
            let rev2 = world.config_audit().len();
            let value = serde_json::to_value(world.config()).map_err(ControlError::serialization)?;
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
        let mut events = Vec::with_capacity(limit);
        for summary in world.history().rev() {
            if summary.births > 0 {
                events.push(EventEntry::new(
                    summary.tick.0,
                    EventKind::Birth,
                    summary.births as u32,
                ));
                if events.len() >= limit { break; }
            }
            if summary.deaths > 0 {
                events.push(EventEntry::new(
                    summary.tick.0,
                    EventKind::Death,
                    summary.deaths as u32,
                ));
                if events.len() >= limit { break; }
            }
            if summary.spike_hits > 0 {
                events.push(EventEntry::new(
                    summary.tick.0,
                    EventKind::Combat,
                    summary.spike_hits as u32,
                ));
                if events.len() >= limit { break; }
            }
        }
        Ok(events)
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
            return Ok(Scoreboard { top_predators: Vec::new(), oldest: Vec::new() });
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
        let mut config_value = serde_json::to_value(world.config()).map_err(ControlError::serialization)?;
        let mut path = SmallVec::<[&str; 8]>::new();
        merge_value(&mut config_value, &patch, &mut path)?;
        let json_str = serde_json::to_string(&config_value).map_err(ControlError::serialization)?;
        let mut de = serde_json::Deserializer::from_str(&json_str);
        let new_config: ScriptBotsConfig = serde_path_to_error::deserialize::<_, ScriptBotsConfig>(&mut de)
            .map_err(|e: serde_path_to_error::Error<serde_json::Error>| ControlError::InvalidPatch(format!("{} at {}", e, e.path())))?;
        let (food_w, food_h) = new_config
            .food_dimensions()
            .map_err(|err| ControlError::InvalidPatch(err.to_string()))?;
        let current_dims = (world.food().width(), world.food().height());
        if current_dims != (food_w, food_h) {
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
        match self.commands.try_send(command) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_msg)) => Err(ControlError::CommandQueueFull),
            Err(TrySendError::Disconnected(_msg)) => Err(ControlError::CommandQueueClosed),
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
            ControlError::InvalidPatch(format!(
                "intermediate segment '{seg}' is not an object"
            ))
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
    *target = Value::Number(
        serde_json::Number::from_f64(v).expect("checked finite above"),
    );
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
                    Some(s) if matches!(s.as_str(), "true" | "1" | "yes" | "on" | "t" | "y") => true,
                    Some(s) if matches!(s.as_str(), "false" | "0" | "no" | "off" | "f" | "n") => false,
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

fn flatten_value(prefix: &mut String, value: &Value, entries: &mut Vec<KnobEntry>) {
    match value {
        Value::Object(map) => {
            let base = prefix.len();
            for (k, v) in map {
                if base != 0 { prefix.push('.'); }
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
        let handle = ControlHandle::new(Arc::new(Mutex::new(world)), sender);
        (handle, receiver)
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
        crate::command::drain_pending_commands(&receiver, &mut world);
        assert!((world.config().food_max - 0.6).abs() < f32::EPSILON);
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
    fn debug_agents_lists_selection() {
        let (handle, receiver) = handle();
        let raw_id = {
            let mut world = handle.lock_world().expect("world lock");
            let id = world.spawn_agent(scriptbots_core::AgentData::default());
            world.apply_selection_update(SelectionUpdate {
                mode: SelectionMode::Replace,
                agent_ids: vec![id.data().as_ffi()],
                state: SelectionState::Selected,
            });
            crate::command::drain_pending_commands(&receiver, &mut world);
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
            let id = world.spawn_agent(scriptbots_core::AgentData::default());
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
        crate::command::drain_pending_commands(&receiver, &mut world);
        let agent_id = scriptbots_core::AgentId::from(KeyData::from_ffi(raw_id));
        let runtime = world.agent_runtime(agent_id).expect("runtime");
        assert!(matches!(runtime.selection, SelectionState::Selected));
    }
}
