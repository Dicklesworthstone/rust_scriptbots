//! NeuroFlow-backed brain implementation gated behind an opt-in feature.
//!
//! This module wraps the `neuroflow` crate’s [`FeedForward`] network so that it can participate in
//! the ScriptBots brain registry. The integration intentionally keeps configuration minimal while
//! remaining forward-compatible with richer training workflows. The implementation focuses on
//! inference; mutation currently randomizes weights using the recorded architecture.

use neuroflow::FeedForward;
use neuroflow::activators::Type;
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};

use scriptbots_brain::{Brain, BrainKind, into_runner};
use scriptbots_core::{BrainRunner, BrainActivations, ActivationLayer, NeuroflowActivationKind, NeuroflowSettings, WorldState};
use std::sync::Arc;

/// Number of inputs inherited from the simulation sensors.
const INPUT_SIZE: usize = scriptbots_core::INPUT_SIZE;
/// Number of outputs consumed by the actuation stage.
const OUTPUT_SIZE: usize = scriptbots_core::OUTPUT_SIZE;

/// Activation families supported by NeuroFlow.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum NeuroflowActivation {
    /// Hyperbolic tangent activation.
    #[default]
    Tanh,
    /// Logistic sigmoid activation.
    Sigmoid,
    /// Rectified linear unit (ReLU).
    Relu,
}

impl NeuroflowActivation {
    fn to_type(self) -> Type {
        match self {
            Self::Tanh => Type::Tanh,
            Self::Sigmoid => Type::Sigmoid,
            Self::Relu => Type::Relu,
        }
    }
}

/// Configuration options for constructing a NeuroFlow-backed brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroflowBrainConfig {
    /// Sizes of hidden layers between the fixed input/output layers.
    pub hidden_layers: Vec<usize>,
    /// Activation function applied to hidden/output layers.
    pub activation: NeuroflowActivation,
    /// Learning rate baked into the network (relevant if online learning is enabled later).
    pub learning_rate: f64,
    /// Momentum factor used by NeuroFlow’s trainer.
    pub momentum: f64,
}

impl NeuroflowBrainConfig {
    #[must_use]
    pub fn from_settings(settings: &NeuroflowSettings) -> Self {
        let mut config = Self::default();
        if !settings.hidden_layers.is_empty() {
            config.hidden_layers = settings.hidden_layers.clone();
        }
        config.activation = match settings.activation {
            NeuroflowActivationKind::Tanh => NeuroflowActivation::Tanh,
            NeuroflowActivationKind::Sigmoid => NeuroflowActivation::Sigmoid,
            NeuroflowActivationKind::Relu => NeuroflowActivation::Relu,
        };
        config
    }
}

impl Default for NeuroflowBrainConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![48, 32, 24],
            activation: NeuroflowActivation::Tanh,
            learning_rate: 0.01,
            momentum: 0.05,
        }
    }
}

/// Runtime brain leveraging NeuroFlow's feed-forward network.
pub struct NeuroflowBrain {
    network: FeedForward,
    config: NeuroflowBrainConfig,
    inputs: Vec<f64>,
}

#[derive(Serialize)]
struct LayerSeed {
    v: Vec<f64>,
    y: Vec<f64>,
    delta: Vec<f64>,
    prev_delta: Vec<f64>,
    w: Vec<Vec<f64>>,
}

#[derive(Serialize)]
struct FeedForwardSeed {
    layers: Vec<LayerSeed>,
    learn_rate: f64,
    momentum: f64,
    error: f64,
    act_type: Type,
}

impl std::fmt::Debug for NeuroflowBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuroflowBrain")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl NeuroflowBrain {
    /// Identifier for the brain registry.
    pub const KIND: BrainKind = BrainKind::new("ml.neuroflow");

    /// Construct a new brain with random weights using the supplied configuration.
    #[must_use]
    pub fn new(config: NeuroflowBrainConfig, rng: &mut dyn RngCore) -> Self {
        let network = Self::build_network(&config, rng);
        Self {
            network,
            config,
            inputs: vec![0.0; INPUT_SIZE],
        }
    }

    /// Convenience helper to box the brain into a [`BrainRunner`].
    #[must_use]
    pub fn runner(config: NeuroflowBrainConfig, rng: &mut dyn RngCore) -> Box<dyn BrainRunner> {
        into_runner(Self::new(config, rng))
    }

    /// Register a NeuroFlow brain into the world registry and return its key.
    #[must_use]
    pub fn register(world: &mut WorldState, config: NeuroflowBrainConfig) -> u64 {
        let config = Arc::new(config);
        world
            .brain_registry_mut()
            .register(Self::KIND.as_str(), move |rng| {
                Self::runner((*config).clone(), rng)
            })
    }

    fn build_network(config: &NeuroflowBrainConfig, rng: &mut dyn RngCore) -> FeedForward {
        let mut architecture: Vec<i32> = Vec::with_capacity(config.hidden_layers.len() + 2);
        architecture.push(INPUT_SIZE as i32);
        architecture.extend(
            config
                .hidden_layers
                .iter()
                .copied()
                .map(|layer| layer as i32),
        );
        architecture.push(OUTPUT_SIZE as i32);

        let mut layers = Vec::with_capacity(architecture.len().saturating_sub(1));
        for window in architecture.windows(2) {
            let inputs = window[0] as usize;
            let outputs = window[1] as usize;
            let mut neurons = Vec::with_capacity(outputs);
            for _ in 0..outputs {
                let mut weights = Vec::with_capacity(inputs + 1);
                for _ in 0..=inputs {
                    weights.push(rng.random_range(-1.0..1.0));
                }
                neurons.push(weights);
            }

            layers.push(LayerSeed {
                v: vec![0.0; outputs],
                y: vec![0.0; outputs],
                delta: vec![0.0; outputs],
                prev_delta: vec![0.0; outputs],
                w: neurons,
            });
        }

        let seed = FeedForwardSeed {
            layers,
            learn_rate: config.learning_rate,
            momentum: config.momentum,
            error: 0.0,
            act_type: config.activation.to_type(),
        };

        let value = serde_json::to_value(&seed).expect("serialize neuroflow seed");
        let mut network: FeedForward =
            serde_json::from_value(value).expect("construct neuroflow network");
        network
            .activation(config.activation.to_type())
            .learning_rate(config.learning_rate)
            .momentum(config.momentum);
        network
    }
}

impl Brain for NeuroflowBrain {
    fn kind(&self) -> BrainKind {
        Self::KIND
    }

    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        for (slot, value) in self.inputs.iter_mut().zip(inputs.iter()) {
            *slot = (*value) as f64;
        }
        let outputs = self.network.calc(&self.inputs);
        let mut result = [0.0; OUTPUT_SIZE];
        for (dst, src) in result.iter_mut().zip(outputs.iter()) {
            *dst = (*src) as f32;
        }
        result
    }

    fn mutate(&mut self, rng: &mut dyn RngCore, rate: f32, _scale: f32) {
        if rate <= 0.0 {
            return;
        }
        if rng.random::<f32>() <= rate {
            self.network = Self::build_network(&self.config, rng);
        }
    }

    fn as_any(&self) -> &(dyn std::any::Any + Send + Sync) {
        self
    }

    fn as_any_mut(&mut self) -> &mut (dyn std::any::Any + Send + Sync) {
        self
    }
}

impl BrainRunner for scriptbots_brain::BrainRunnerAdapter<NeuroflowBrain> {
    fn kind(&self) -> &'static str { self.brain.kind().as_str() }
    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] { self.brain.tick(inputs) }
    fn snapshot_activations(&self) -> Option<BrainActivations> {
        // Extract layer outputs (y) from the network by serializing the seed back out.
        // This is a pragmatic approach since NeuroFlow does not expose public getters for y.
        // Note: serde reflection of internal state relies on current NeuroFlow structure.
        let value = serde_json::to_value(&self.brain.network).ok()?;
        let layers = value.get("layers")?.as_array()?.to_vec();
        let mut result_layers: Vec<ActivationLayer> = Vec::new();
        for (li, layer_val) in layers.iter().enumerate() {
            let y = layer_val.get("y").and_then(|v| v.as_array()).cloned().unwrap_or_default();
            let values: Vec<f32> = y.into_iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect();
            let width = (values.len() as f32).sqrt().ceil() as usize;
            let height = if width == 0 { 0 } else { (values.len() + width - 1) / width };
            result_layers.push(ActivationLayer {
                name: format!("nf.layer.{li}"),
                width,
                height,
                values,
            });
        }
        Some(BrainActivations { layers: result_layers, connections: Vec::new() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn runner_executes_and_returns_outputs() {
        let mut rng = SmallRng::seed_from_u64(0xBEEF);
        let mut runner = NeuroflowBrain::runner(NeuroflowBrainConfig::default(), &mut rng);
        let outputs = runner.tick(&[0.0; INPUT_SIZE]);
        assert_eq!(outputs.len(), OUTPUT_SIZE);
        assert!(outputs.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mutate_regenerates_network() {
        let mut rng = SmallRng::seed_from_u64(0xCAFE);
        let config = NeuroflowBrainConfig::default();
        let mut brain = NeuroflowBrain::new(config.clone(), &mut rng);
        let baseline = brain.tick(&[0.0; INPUT_SIZE]);
        brain.mutate(&mut rng, 1.0, 0.5);
        let after = brain.tick(&[0.0; INPUT_SIZE]);
        // The outputs are likely to differ after reinitialization.
        assert_ne!(baseline, after);
    }
}
