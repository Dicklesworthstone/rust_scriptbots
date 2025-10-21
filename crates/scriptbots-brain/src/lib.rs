//! Traits and baseline implementations for ScriptBots brains.

use rand::Rng;
use scriptbots_core::{AgentId, Tick};
use serde::{Deserialize, Serialize};

/// Shared interface implemented by all agent brains.
pub trait Brain {
    /// Immutable brain identifier (useful for analytics).
    fn kind(&self) -> &'static str;

    /// Evaluate brain outputs given the latest sensor input vector.
    fn tick(&mut self, inputs: &[f32]) -> Vec<f32>;

    /// Mutate the brain's internal state given mutation rates.
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32, scale: f32);
}

/// Summary emitted after each brain evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainTelemetry {
    pub agent: AgentId,
    pub tick: Tick,
    pub energy_spent: f32,
}
