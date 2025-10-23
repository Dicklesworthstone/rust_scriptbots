use std::sync::{MutexGuard, PoisonError};

use crossfire::TrySendError;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use scriptbots_core::{ControlCommand, ScriptBotsConfig, Tick, WorldState};

use crate::SharedWorld;
use crate::command::CommandSender;
use scriptbots_core::ConfigAuditEntry;

/// Snapshot of configuration state returned to external clients.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ConfigSnapshot {
    pub tick: u64,
    pub config: Value,
}

impl ConfigSnapshot {
    fn from_world(config: &ScriptBotsConfig, tick: Tick) -> Result<Self, ControlError> {
        Self::from_config(config.clone(), tick)
    }

    fn from_config(config: ScriptBotsConfig, tick: Tick) -> Result<Self, ControlError> {
        let config_value = serde_json::to_value(config).map_err(ControlError::serialization)?;
        Ok(Self {
            tick: tick.0,
            config: config_value,
        })
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

/// Shared handle used by REST, CLI, and MCP surfaces to access the running world.
#[derive(Clone)]
pub struct ControlHandle {
    shared_world: SharedWorld,
    commands: CommandSender,
}

impl ControlHandle {
    pub fn new(shared_world: SharedWorld, commands: CommandSender) -> Self {
        Self {
            shared_world,
            commands,
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
            })
        }
    }

    /// Flatten the configuration into individual knob descriptors for discovery.
    pub fn list_knobs(&self) -> Result<Vec<KnobEntry>, ControlError> {
        let world = self.lock_world()?;
        let mut entries = Vec::new();
        let config_value =
            serde_json::to_value(world.config().clone()).map_err(ControlError::serialization)?;
        flatten_value("", &config_value, &mut entries);
        Ok(entries)
    }

    /// Retrieve the configuration audit log accumulated since startup.
    pub fn audit(&self) -> Result<Vec<ConfigAuditEntry>, ControlError> {
        let world = self.lock_world()?;
        Ok(world.config_audit().to_vec())
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
            serde_json::to_value(world.config().clone()).map_err(ControlError::serialization)?;
        merge_value(&mut config_value, &patch, &mut Vec::new())?;
        let new_config: ScriptBotsConfig =
            serde_json::from_value(config_value).map_err(ControlError::serialization)?;
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
        self.enqueue(ControlCommand::UpdateConfig(new_config))?;
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
    let mut segments = path.split('.').filter(|segment| !segment.is_empty());
    let Some(first) = segments.next() else {
        return Err(ControlError::InvalidPatch("empty knob path".into()));
    };

    let mut current = map;
    let mut segment = first.to_string();

    for next in segments {
        let entry = current
            .entry(segment.clone())
            .or_insert_with(|| Value::Object(Map::new()));
        current = entry.as_object_mut().ok_or_else(|| {
            ControlError::InvalidPatch(format!("intermediate segment '{segment}' is not an object"))
        })?;
        segment = next.to_string();
    }

    current.insert(segment, value);
    Ok(())
}

fn merge_value(
    target: &mut Value,
    patch: &Value,
    path: &mut Vec<String>,
) -> Result<(), ControlError> {
    match target {
        Value::Object(target_map) => {
            let Value::Object(patch_map) = patch else {
                return Err(ControlError::InvalidPatch(format!(
                    "type mismatch at {}",
                    path.join("."),
                )));
            };

            for (key, patch_value) in patch_map {
                path.push(key.clone());
                let Some(target_value) = target_map.get_mut(key) else {
                    return Err(ControlError::UnknownPath(path.join(".")));
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
                    path.join("."),
                )))
            }
        }
        Value::Number(_) => match patch {
            Value::Number(_) => {
                *target = patch.clone();
                Ok(())
            }
            Value::String(_) => {
                let Some(text) = patch.as_str() else {
                    return Err(ControlError::InvalidPatch(path.join(".")));
                };
                if target.as_i64().is_some() {
                    let parsed: i64 = text
                        .trim()
                        .parse()
                        .map_err(|_| ControlError::InvalidPatch(path.join(".")))?;
                    *target = Value::from(parsed);
                } else if target.as_u64().is_some() {
                    let parsed: u64 = text
                        .trim()
                        .parse()
                        .map_err(|_| ControlError::InvalidPatch(path.join(".")))?;
                    *target = Value::from(parsed);
                } else {
                    let parsed: f64 = text
                        .trim()
                        .parse()
                        .map_err(|_| ControlError::InvalidPatch(path.join(".")))?;
                    *target = Value::from(parsed);
                }
                Ok(())
            }
            Value::Null => {
                *target = Value::Null;
                Ok(())
            }
            _ => Err(ControlError::InvalidPatch(format!(
                "type mismatch at {}",
                path.join("."),
            ))),
        },
        Value::String(_) => match patch {
            Value::String(_) | Value::Null => {
                *target = patch.clone();
                Ok(())
            }
            _ => Err(ControlError::InvalidPatch(format!(
                "type mismatch at {}",
                path.join("."),
            ))),
        },
        Value::Bool(_) => match patch {
            Value::Bool(_) | Value::Null => {
                *target = patch.clone();
                Ok(())
            }
            Value::String(_) => {
                let parsed = match patch.as_str().map(|s| s.trim().to_ascii_lowercase()) {
                    Some(s) if matches!(s.as_str(), "true" | "1" | "yes" | "on") => true,
                    Some(s) if matches!(s.as_str(), "false" | "0" | "no" | "off") => false,
                    _ => {
                        return Err(ControlError::InvalidPatch(format!(
                            "cannot coerce '{:?}' to bool for {}",
                            patch,
                            path.join("."),
                        )));
                    }
                };
                *target = Value::from(parsed);
                Ok(())
            }
            _ => Err(ControlError::InvalidPatch(format!(
                "type mismatch at {}",
                path.join("."),
            ))),
        },
        Value::Null => {
            *target = patch.clone();
            Ok(())
        }
    }
}

fn flatten_value(prefix: &str, value: &Value, entries: &mut Vec<KnobEntry>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let new_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten_value(&new_prefix, child, entries);
            }
        }
        _ => entries.push(KnobEntry {
            path: prefix.to_string(),
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
}
