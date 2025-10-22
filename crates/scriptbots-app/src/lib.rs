//! Shared application plumbing for ScriptBots control surfaces.

use std::sync::{Arc, Mutex};

use scriptbots_core::WorldState;
use scriptbots_storage::Storage;

pub type SharedWorld = Arc<Mutex<WorldState>>;
pub type SharedStorage = Arc<Mutex<Storage>>;

pub mod control;
pub mod servers;

pub use control::{ConfigSnapshot, ControlError, ControlHandle, KnobEntry, KnobKind, KnobUpdate};
pub use servers::{
    ConfigPatchRequest, ControlRuntime, ControlServerConfig, KnobApplyRequest, McpTransportConfig,
};
