//! Traits and adapters for ScriptBots brain implementations.

use rand::RngCore;
use scriptbots_core::{AgentId, BrainActivations, BrainRunner, INPUT_SIZE, OUTPUT_SIZE, Tick};
use serde::{Deserialize, Serialize};
use std::any::Any;

#[cfg(feature = "mlp")]
pub mod mlp;
#[cfg(feature = "mlp")]
pub use mlp::MlpBrain;

#[cfg(feature = "dwraon")]
pub mod dwraon;
#[cfg(feature = "dwraon")]
pub use dwraon::DwraonBrain;

#[cfg(feature = "assembly")]
pub mod assembly;
#[cfg(feature = "assembly")]
pub use assembly::AssemblyBrain;

/// Small newtype wrapper identifying brain families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BrainKind(&'static str);

impl BrainKind {
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl From<&'static str> for BrainKind {
    fn from(value: &'static str) -> Self {
        Self::new(value)
    }
}

/// Shared interface implemented by all agent brains.
pub trait Brain: Send + Sync + Any {
    /// Unique identifier for analytics/registry display.
    fn kind(&self) -> BrainKind;

    /// Evaluate brain outputs given the latest sensor input vector.
    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE];

    /// Mutate the brain's internal state given mutation rates.
    fn mutate(&mut self, rng: &mut dyn RngCore, rate: f32, scale: f32);

    /// Optional crossover hook; return `None` when unsupported.
    fn crossover(&self, _other: &dyn Brain, _rng: &mut dyn RngCore) -> Option<Box<dyn Brain>> {
        None
    }

    /// Duplicate this brain including all evolved parameters. Heredity relies
    /// on this: offspring receive the parent's weights, not a fresh network.
    fn clone_box(&self) -> Box<dyn Brain>;

    /// Downcast support for concrete brain logic.
    fn as_any(&self) -> &(dyn Any + Send + Sync);

    /// Mutable downcast support for concrete brain logic.
    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync);

    /// Optional introspection hook for visualization/debug UIs.
    /// Default returns `None` for brains that do not expose activations.
    fn snapshot_activations(&self) -> Option<BrainActivations> {
        None
    }
}

/// Summary emitted after each brain evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainTelemetry {
    pub agent: AgentId,
    pub tick: Tick,
    pub energy_spent: f32,
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct EchoBrain;

    impl Brain for EchoBrain {
        fn kind(&self) -> BrainKind {
            BrainKind::new("echo")
        }

        fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
            let mut outputs = [0.0; OUTPUT_SIZE];
            outputs[..OUTPUT_SIZE.min(INPUT_SIZE)]
                .copy_from_slice(&inputs[..OUTPUT_SIZE.min(INPUT_SIZE)]);
            outputs
        }

        fn mutate(&mut self, _rng: &mut dyn RngCore, _rate: f32, _scale: f32) {}

        fn clone_box(&self) -> Box<dyn Brain> {
            Box::new(self.clone())
        }

        fn as_any(&self) -> &(dyn Any + Send + Sync) {
            self
        }

        fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) {
            self
        }
    }

    #[test]
    fn adapter_forward_outputs() {
        let mut runner = BrainRunnerAdapter::new(EchoBrain);
        let mut inputs = [0.0; INPUT_SIZE];
        inputs[0] = 1.0;
        inputs[1] = 2.0;
        let outputs = runner.tick(&inputs);
        assert!((outputs[0] - 1.0).abs() < f32::EPSILON);
        assert!((outputs[1] - 2.0).abs() < f32::EPSILON);
    }
}

/// Adapter bridging a [`Brain`] implementation into the simulation registry.
///
/// Holds the brain as a trait object so that crossover offspring (which are
/// produced as `Box<dyn Brain>`) stay heritable across generations.
pub struct BrainRunnerAdapter {
    brain: Box<dyn Brain>,
}

impl BrainRunnerAdapter {
    #[must_use]
    pub fn new<B: Brain + 'static>(brain: B) -> Self {
        Self {
            brain: Box::new(brain),
        }
    }

    #[must_use]
    pub fn from_boxed(brain: Box<dyn Brain>) -> Self {
        Self { brain }
    }

    #[must_use]
    pub fn into_inner(self) -> Box<dyn Brain> {
        self.brain
    }
}

impl BrainRunner for BrainRunnerAdapter {
    fn kind(&self) -> &'static str {
        self.brain.kind().as_str()
    }

    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        self.brain.tick(inputs)
    }

    fn snapshot_activations(&self) -> Option<BrainActivations> {
        self.brain.snapshot_activations()
    }

    fn clone_runner(&self) -> Option<Box<dyn BrainRunner>> {
        Some(Box::new(Self {
            brain: self.brain.clone_box(),
        }))
    }

    fn mutate(&mut self, rng: &mut dyn RngCore, rate: f32, scale: f32) {
        self.brain.mutate(rng, rate, scale);
    }

    fn crossover(
        &self,
        partner: &dyn BrainRunner,
        rng: &mut dyn RngCore,
    ) -> Option<Box<dyn BrainRunner>> {
        let partner = partner.as_any()?.downcast_ref::<Self>()?;
        let child = self.brain.crossover(&*partner.brain, rng)?;
        Some(Box::new(Self { brain: child }))
    }

    fn as_any(&self) -> Option<&(dyn Any + Send + Sync)> {
        Some(self)
    }
}

/// Convenience helper to box a brain as a [`BrainRunner`].
#[must_use]
pub fn into_runner<B: Brain + 'static>(brain: B) -> Box<dyn BrainRunner> {
    Box::new(BrainRunnerAdapter::new(brain))
}
