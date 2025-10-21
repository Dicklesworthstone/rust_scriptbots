//! Core data structures and traits shared across the ScriptBots workspace.

use serde::{Deserialize, Serialize};

/// Unique identifier assigned to each agent at creation time.
pub type AgentId = u64;

/// High level simulation clock (ticks processed since boot).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Tick(pub u64);

impl Tick {
    /// Returns the next sequential tick.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }
}
