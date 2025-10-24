//! Multi-layer perceptron brain mirroring the legacy ScriptBots baseline.

use rand::Rng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::any::Any;

use scriptbots_core::{BrainRunner, BrainActivations, ActivationLayer, INPUT_SIZE, OUTPUT_SIZE};

use crate::{Brain, BrainKind, into_runner};

const BRAIN_SIZE: usize = 200;
const CONNECTIONS: usize = 4;

/// Identifies how a synapse samples its source neuron.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum SynapseKind {
    Regular,
    ChangeSensitive,
}

impl SynapseKind {
    fn random(rng: &mut dyn RngCore) -> Self {
        if rng.random::<f32>() < 0.05 {
            Self::ChangeSensitive
        } else {
            Self::Regular
        }
    }

    fn flip(self) -> Self {
        match self {
            Self::Regular => Self::ChangeSensitive,
            Self::ChangeSensitive => Self::Regular,
        }
    }
}

/// Immutable parameters describing a node in the MLP network.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeParams {
    weights: [f32; CONNECTIONS],
    targets: [usize; CONNECTIONS],
    kinds: [SynapseKind; CONNECTIONS],
    gain: f32,
    damping: f32,
    bias: f32,
}

impl NodeParams {
    fn random(rng: &mut dyn RngCore) -> Self {
        let mut weights = [0.0; CONNECTIONS];
        for weight in &mut weights {
            let value = rng.random_range(-3.0..3.0);
            *weight = if rng.random::<f32>() < 0.5 {
                0.0
            } else {
                value
            };
        }

        let mut targets = [0usize; CONNECTIONS];
        for target in &mut targets {
            *target = rng.random_range(0..BRAIN_SIZE);
            if rng.random::<f32>() < 0.2 {
                *target = rng.random_range(0..INPUT_SIZE);
            }
        }

        let mut kinds = [SynapseKind::Regular; CONNECTIONS];
        for kind in &mut kinds {
            *kind = SynapseKind::random(rng);
        }

        Self {
            weights,
            targets,
            kinds,
            gain: rng.random_range(0.0..5.0),
            damping: rng.random_range(0.9..1.1),
            bias: rng.random_range(-2.0..2.0),
        }
    }
}

/// Dynamic state for each node.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeState {
    output: f32,
    previous_output: f32,
    target: f32,
}

impl Default for NodeState {
    fn default() -> Self {
        Self {
            output: 0.0,
            previous_output: 0.0,
            target: 0.0,
        }
    }
}

/// Baseline ScriptBots MLP brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpBrain {
    nodes: Vec<NodeParams>,
    state: Vec<NodeState>,
}

impl MlpBrain {
    /// Trait identifier for this brain family.
    pub const KIND: BrainKind = BrainKind::new("mlp.baseline");

    /// Construct a randomly initialized brain.
    #[must_use]
    pub fn random(rng: &mut dyn RngCore) -> Self {
        let mut nodes = Vec::with_capacity(BRAIN_SIZE);
        for _ in 0..BRAIN_SIZE {
            nodes.push(NodeParams::random(rng));
        }
        let mut brain = Self {
            nodes,
            state: vec![NodeState::default(); BRAIN_SIZE],
        };
        brain.reset_state();
        brain
    }

    /// Return a boxed brain runner wrapping a randomly initialized MLP.
    #[must_use]
    pub fn runner(rng: &mut dyn RngCore) -> Box<dyn BrainRunner> {
        into_runner(Self::random(rng))
    }

    fn reset_state(&mut self) {
        for node in &mut self.state {
            *node = NodeState::default();
        }
    }

    fn logistic(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    fn gaussian(rng: &mut dyn RngCore) -> f32 {
        const TWO_PI: f32 = std::f32::consts::TAU;
        let u1 = (rng.random::<f32>()).clamp(f32::MIN_POSITIVE, 1.0);
        let u2 = rng.random::<f32>();
        (-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).cos()
    }

    pub(crate) fn activations(&self) -> BrainActivations {
        // Map internal node outputs into a single-layer activation map for now.
        let width = 20usize;
        let height = 10usize;
        let mut values = vec![0.0_f32; width * height];
        for (i, node) in self.state.iter().enumerate().take(values.len()) {
            values[i] = node.output;
        }
        BrainActivations {
            layers: vec![ActivationLayer {
                name: "mlp.state".to_string(),
                width,
                height,
                values,
            }],
            connections: Vec::new(),
        }
    }
}

impl Brain for MlpBrain {
    fn kind(&self) -> BrainKind {
        Self::KIND
    }

    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        for (idx, input) in inputs.iter().enumerate() {
            if let Some(node) = self.state.get_mut(idx) {
                node.output = *input;
            }
        }

        for idx in INPUT_SIZE..self.nodes.len() {
            let params = &self.nodes[idx];
            let mut acc = 0.0_f32;
            for conn in 0..CONNECTIONS {
                let target_idx = params.targets[conn];
                let source = self
                    .state
                    .get(target_idx)
                    .map(|node| node.output)
                    .unwrap_or(0.0);
                let delta = match params.kinds[conn] {
                    SynapseKind::Regular => source,
                    SynapseKind::ChangeSensitive => {
                        let previous = self
                            .state
                            .get(target_idx)
                            .map(|node| node.previous_output)
                            .unwrap_or(0.0);
                        (source - previous) * 10.0
                    }
                };
                acc += delta * params.weights[conn];
            }

            acc *= params.gain;
            acc += params.bias;
            let target = Self::logistic(acc);
            if let Some(node) = self.state.get_mut(idx) {
                node.target = target;
            }
        }

        for node in &mut self.state {
            node.previous_output = node.output;
        }

        for idx in INPUT_SIZE..self.nodes.len() {
            let params = &self.nodes[idx];
            if let Some(node) = self.state.get_mut(idx) {
                let delta = node.target - node.output;
                node.output += delta * params.damping.clamp(0.01, 1.0);
            }
        }

        let mut result = [0.0; OUTPUT_SIZE];
        for (output, node) in result.iter_mut().zip(self.state.iter().rev()) {
            *output = node.output;
        }
        result
    }

    fn mutate(&mut self, rng: &mut dyn RngCore, rate: f32, scale: f32) {
        let sigma = scale.max(1e-5);
        for params in &mut self.nodes {
            if rng.random::<f32>() < rate {
                params.bias += Self::gaussian(rng) * sigma;
            }
            if rng.random::<f32>() < rate {
                params.damping = (params.damping + Self::gaussian(rng) * sigma).clamp(0.01, 1.0);
            }
            if rng.random::<f32>() < rate {
                params.gain = (params.gain + Self::gaussian(rng) * sigma).max(0.0);
            }
            if rng.random::<f32>() < rate {
                let idx = rng.random_range(0..CONNECTIONS);
                params.weights[idx] += Self::gaussian(rng) * sigma;
            }
            if rng.random::<f32>() < rate {
                let idx = rng.random_range(0..CONNECTIONS);
                params.kinds[idx] = params.kinds[idx].flip();
            }
            if rng.random::<f32>() < rate {
                let idx = rng.random_range(0..CONNECTIONS);
                let target = if rng.random::<f32>() < 0.2 {
                    rng.random_range(0..INPUT_SIZE)
                } else {
                    rng.random_range(0..BRAIN_SIZE)
                };
                params.targets[idx] = target;
            }
        }
    }

    fn crossover(&self, other: &dyn Brain, rng: &mut dyn RngCore) -> Option<Box<dyn Brain>> {
        if other.kind() != Self::KIND {
            return None;
        }
        let other = other.as_any().downcast_ref::<Self>()?;
        let mut child = self.clone();
        for (child_params, other_params) in child.nodes.iter_mut().zip(&other.nodes) {
            if rng.random::<f32>() < 0.5 {
                continue;
            }
            *child_params = other_params.clone();
        }
        child.reset_state();
        Some(Box::new(child))
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) {
        self
    }

    fn snapshot_activations(&self) -> Option<BrainActivations> { Some(self.activations()) }
}

// Specialized adapter impl removed; generic adapter in lib.rs downcasts to call `activations()`.

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn random_brain_has_expected_structure() {
        let mut rng = SmallRng::seed_from_u64(0xDEADBEEF);
        let brain = MlpBrain::random(&mut rng);
        assert_eq!(brain.nodes.len(), BRAIN_SIZE);
        assert_eq!(brain.state.len(), BRAIN_SIZE);
    }

    #[test]
    fn tick_produces_stable_outputs() {
        let mut rng = SmallRng::seed_from_u64(123);
        let mut brain = MlpBrain::random(&mut rng);
        let mut inputs = [0.0; INPUT_SIZE];
        inputs[0] = 1.0;
        let outputs = brain.tick(&inputs);
        assert!(outputs.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mutate_changes_parameters() {
        let mut rng = SmallRng::seed_from_u64(456);
        let mut brain = MlpBrain::random(&mut rng);
        let original = brain.nodes[10].bias;
        brain.mutate(&mut rng, 1.0, 0.5);
        assert_ne!(brain.nodes[10].bias, original);
    }

    #[test]
    fn crossover_combines_parents() {
        let mut rng = SmallRng::seed_from_u64(789);
        let brain_a = MlpBrain::random(&mut rng);
        let brain_b = MlpBrain::random(&mut rng);
        let mut rng = SmallRng::seed_from_u64(101112);
        let child = brain_a
            .crossover(&brain_b, &mut rng)
            .expect("crossover should succeed");
        assert_eq!(child.kind(), MlpBrain::KIND);
    }

    #[test]
    fn runner_bridge_executes() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut runner = MlpBrain::runner(&mut rng);
        let inputs = [0.0; INPUT_SIZE];
        let outputs = runner.tick(&inputs);
        assert!(outputs.iter().all(|v| v.is_finite()));
        assert_eq!(runner.kind(), MlpBrain::KIND.as_str());
    }
}
