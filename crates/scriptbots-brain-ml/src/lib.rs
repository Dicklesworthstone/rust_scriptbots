//! Placeholder NeuroFlow-backed brain integrations.

use std::marker::PhantomData;

use neuroflow::FeedForward;
use rand::Rng;
use scriptbots_brain::Brain;

/// Temporary wrapper that reserves space for a future NeuroFlow network.
#[derive(Default)]
pub struct NeuroBrain {
    _marker: PhantomData<FeedForward>,
}

impl NeuroBrain {
    /// Construct a placeholder NeuroFlow brain.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Brain for NeuroBrain {
    fn kind(&self) -> &'static str {
        "neuroflow"
    }

    fn tick(&mut self, inputs: &[f32]) -> Vec<f32> {
        inputs.to_vec()
    }

    fn mutate<R: Rng>(&mut self, _rng: &mut R, _rate: f32, _scale: f32) {
        // Mutation logic will be implemented alongside the full NeuroFlow integration.
    }
}
