//! Shared application plumbing for ScriptBots control surfaces.

use std::sync::{Arc, Mutex};

use scriptbots_core::WorldState;
use scriptbots_storage::Storage;

pub type SharedWorld = Arc<Mutex<WorldState>>;
pub type SharedStorage = Arc<Mutex<Storage>>;

pub mod control;
pub mod servers;
pub mod terminal;

pub mod renderer {
    use anyhow::Result;

    use crate::{ControlRuntime, SharedStorage, SharedWorld};

    /// Shared context passed to renderer implementations.
    pub struct RendererContext<'a> {
        pub world: SharedWorld,
        pub storage: SharedStorage,
        pub control_runtime: &'a ControlRuntime,
    }

    pub trait Renderer {
        /// Stable identifier describing the renderer implementation (e.g., "gpui", "terminal").
        fn name(&self) -> &'static str;

        /// Launch the renderer; blocks until the rendering session completes.
        fn run(&self, ctx: RendererContext<'_>) -> Result<()>;
    }
}

pub use control::{ConfigSnapshot, ControlError, ControlHandle, KnobEntry, KnobKind, KnobUpdate};
pub use servers::{
    ConfigPatchRequest, ControlRuntime, ControlServerConfig, KnobApplyRequest, McpTransportConfig,
};
