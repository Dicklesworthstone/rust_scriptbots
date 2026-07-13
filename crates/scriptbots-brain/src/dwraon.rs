//! Feature-gated DWRAON brain (Damped Weighted Recurrent AND/OR Network).

use rand::Rng;
use scriptbots_core::RandomStream;
use serde::{Deserialize, Serialize};
use std::any::Any;

use scriptbots_core::{BrainRunner, INPUT_SIZE, OUTPUT_SIZE};

use crate::{Brain, BrainKind, into_runner};

const BRAIN_SIZE: usize = 200;
const CONNECTIONS: usize = 4;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum NodeKind {
    And,
    Or,
}

impl NodeKind {
    fn random(rng: &mut dyn RandomStream) -> Self {
        if rng.random::<f32>() < 0.5 {
            Self::And
        } else {
            Self::Or
        }
    }

    fn toggle(self) -> Self {
        match self {
            Self::And => Self::Or,
            Self::Or => Self::And,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeParams {
    kind: NodeKind,
    damping: f32,
    bias: f32,
    weights: [f32; CONNECTIONS],
    sources: [usize; CONNECTIONS],
    inverted: [bool; CONNECTIONS],
}

impl NodeParams {
    fn random(rng: &mut dyn RandomStream) -> Self {
        let mut weights = [0.0; CONNECTIONS];
        for weight in &mut weights {
            *weight = rng.random_range(0.1..2.0);
        }

        let mut sources = [0usize; CONNECTIONS];
        for source in &mut sources {
            *source = rng.random_range(0..BRAIN_SIZE);
            if rng.random::<f32>() < 0.2 {
                *source = rng.random_range(0..INPUT_SIZE);
            }
        }

        let mut inverted = [false; CONNECTIONS];
        for flag in &mut inverted {
            *flag = rng.random::<f32>() < 0.5;
        }

        Self {
            kind: NodeKind::random(rng),
            damping: rng.random_range(0.8..1.0),
            bias: rng.random_range(-1.0..1.0),
            weights,
            sources,
            inverted,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct NodeState {
    output: f32,
    target: f32,
}

impl Default for NodeState {
    fn default() -> Self {
        Self {
            output: 0.0,
            target: 0.0,
        }
    }
}

/// DWRAON implementation closely mirroring the legacy C++ behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DwraonBrain {
    nodes: Vec<NodeParams>,
    state: Vec<NodeState>,
}

impl DwraonBrain {
    /// Trait identifier for this brain family.
    pub const KIND: BrainKind = BrainKind::new("dwraon.baseline");

    /// Construct a randomly initialized brain.
    #[must_use]
    pub fn random(rng: &mut dyn RandomStream) -> Self {
        let mut nodes = Vec::with_capacity(BRAIN_SIZE);
        for idx in 0..BRAIN_SIZE {
            let mut params = NodeParams::random(rng);
            // legacy DWRAONBrain: the first half of the brain reads sensors
            // directly, guaranteeing a large reactive (non-recurrent) core.
            if idx < BRAIN_SIZE / 2 {
                for source in &mut params.sources {
                    *source = rng.random_range(0..INPUT_SIZE);
                }
            }
            nodes.push(params);
        }

        let mut brain = Self {
            nodes,
            state: vec![NodeState::default(); BRAIN_SIZE],
        };
        brain.reset_state();
        brain
    }

    /// Return a boxed runner for this brain implementation.
    #[must_use]
    pub fn runner(rng: &mut dyn RandomStream) -> Box<dyn BrainRunner> {
        into_runner(Self::random(rng))
    }

    fn reset_state(&mut self) {
        for node in &mut self.state {
            *node = NodeState::default();
        }
    }

    fn gaussian(rng: &mut dyn RandomStream) -> f32 {
        const TWO_PI: f32 = std::f32::consts::TAU;
        let u1 = (rng.random::<f32>()).clamp(f32::MIN_POSITIVE, 1.0);
        let u2 = rng.random::<f32>();
        (-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).cos()
    }

    fn source_output(&self, index: usize) -> f32 {
        self.state
            .get(index)
            .map(|node| node.output)
            .unwrap_or_default()
    }
}

impl Brain for DwraonBrain {
    fn kind(&self) -> BrainKind {
        Self::KIND
    }

    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        for (idx, input) in inputs.iter().enumerate() {
            if let Some(node) = self.state.get_mut(idx) {
                node.output = input.clamp(0.0, 1.0);
            }
        }

        for idx in INPUT_SIZE..self.nodes.len() {
            let params = &self.nodes[idx];
            let mut target = match params.kind {
                NodeKind::And => {
                    let mut product = 1.0;
                    for conn in 0..CONNECTIONS {
                        let mut value = self.source_output(params.sources[conn]);
                        if params.inverted[conn] {
                            value = 1.0 - value;
                        }
                        product *= value.clamp(0.0, 1.0);
                    }
                    product * params.bias
                }
                NodeKind::Or => {
                    let mut sum = 0.0;
                    for conn in 0..CONNECTIONS {
                        let mut value = self.source_output(params.sources[conn]);
                        if params.inverted[conn] {
                            value = 1.0 - value;
                        }
                        sum += value.clamp(0.0, 1.0) * params.weights[conn];
                    }
                    sum + params.bias
                }
            };

            target = target.clamp(0.0, 1.0);
            if let Some(node) = self.state.get_mut(idx) {
                node.target = target;
            }
        }

        for idx in INPUT_SIZE..self.state.len() {
            let params = &self.nodes[idx];
            if let Some(node) = self.state.get_mut(idx) {
                let delta = node.target - node.output;
                node.output += delta * params.damping.clamp(0.01, 1.0);
                node.output = node.output.clamp(0.0, 1.0);
            }
        }

        let mut outputs = [0.0; OUTPUT_SIZE];
        for (offset, output) in outputs.iter_mut().enumerate() {
            let idx = self.state.len() - 1 - offset;
            *output = self.state[idx].output.clamp(0.0, 1.0);
        }
        outputs
    }

    fn mutate(
        &mut self,
        rng: &mut dyn RandomStream,
        rate: f32,
        scale: f32,
    ) -> Result<(), crate::BrainMutationError> {
        let sigma = scale.max(1e-5);
        for params in &mut self.nodes {
            if rng.random::<f32>() < rate * 3.0 {
                params.bias += Self::gaussian(rng) * sigma;
            }
            if rng.random::<f32>() < rate * 3.0 {
                let idx = rng.random_range(0..CONNECTIONS);
                let weight = params.weights[idx] + Self::gaussian(rng) * sigma;
                params.weights[idx] = weight.max(0.01);
            }
            if rng.random::<f32>() < rate {
                let idx = rng.random_range(0..CONNECTIONS);
                params.sources[idx] = rng.random_range(0..BRAIN_SIZE);
            }
            if rng.random::<f32>() < rate {
                let idx = rng.random_range(0..CONNECTIONS);
                params.inverted[idx] = !params.inverted[idx];
            }
            if rng.random::<f32>() < rate {
                params.kind = params.kind.toggle();
            }
        }
        Ok(())
    }

    fn crossover(&self, other: &dyn Brain, rng: &mut dyn RandomStream) -> Option<Box<dyn Brain>> {
        if other.kind() != Self::KIND {
            return None;
        }
        let other = other.as_any().downcast_ref::<Self>()?;

        // legacy DWRAONBrain recombines per FIELD, not per node: bias, kind,
        // damping, and every connection's weight/source/inversion each flip
        // an independent coin, so co-adapted fields can recombine within a box.
        let mut child = self.clone();
        for (child_params, other_params) in child.nodes.iter_mut().zip(&other.nodes) {
            if rng.random::<f32>() < 0.5 {
                child_params.bias = other_params.bias;
            }
            if rng.random::<f32>() < 0.5 {
                child_params.kind = other_params.kind;
            }
            if rng.random::<f32>() < 0.5 {
                child_params.damping = other_params.damping;
            }
            for conn in 0..CONNECTIONS {
                if rng.random::<f32>() < 0.5 {
                    child_params.weights[conn] = other_params.weights[conn];
                }
                if rng.random::<f32>() < 0.5 {
                    child_params.sources[conn] = other_params.sources[conn];
                }
                if rng.random::<f32>() < 0.5 {
                    child_params.inverted[conn] = other_params.inverted[conn];
                }
            }
        }

        child.reset_state();
        Some(Box::new(child))
    }

    fn clone_box(&self) -> Result<Box<dyn Brain>, crate::BrainCloneError> {
        Ok(Box::new(self.clone()))
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::SmallRngStream;

    #[test]
    fn random_brain_builds_expected_layout() {
        let mut rng = SmallRngStream::seed_from_u64(0x5A5A5A5A);
        let brain = DwraonBrain::random(&mut rng);
        assert_eq!(brain.nodes.len(), BRAIN_SIZE);
        assert_eq!(brain.state.len(), BRAIN_SIZE);
    }

    #[test]
    fn tick_emits_bounded_outputs() {
        let mut rng = SmallRngStream::seed_from_u64(1234);
        let mut brain = DwraonBrain::random(&mut rng);
        let inputs = [0.25; INPUT_SIZE];
        let outputs = brain.tick(&inputs);
        assert!(outputs.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn mutate_adjusts_parameters() {
        let mut rng = SmallRngStream::seed_from_u64(5678);
        let mut brain = DwraonBrain::random(&mut rng);
        let before = brain.nodes[5].bias;
        brain
            .mutate(&mut rng, 1.0, 0.5)
            .expect("DWRAON mutation is infallible");
        assert_ne!(brain.nodes[5].bias, before);
    }

    #[test]
    fn crossover_combines_parents() {
        let mut rng = SmallRngStream::seed_from_u64(42);
        let brain_a = DwraonBrain::random(&mut rng);
        let brain_b = DwraonBrain::random(&mut rng);
        let mut rng = SmallRngStream::seed_from_u64(84);
        let child = brain_a.crossover(&brain_b, &mut rng).expect("same kind");
        assert_eq!(child.kind(), DwraonBrain::KIND);
    }

    #[test]
    fn runner_bridge_invokes_brain() {
        let mut rng = SmallRngStream::seed_from_u64(9001);
        let mut runner = DwraonBrain::runner(&mut rng);
        let inputs = [0.1; INPUT_SIZE];
        let outputs = runner.tick(&inputs);
        assert!(outputs.iter().all(|v| v.is_finite()));
    }
}
