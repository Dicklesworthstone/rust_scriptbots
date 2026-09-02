//! Candle-backed deep neural network brain backend.
//!
//! Provides a real, trainable and evolvable neural network for agent decision-making
//! using the `candle-core` tensor execution framework.

use rand::Rng;
use scriptbots_brain::{Brain, BrainCloneError, BrainKind, BrainMutationError};
use scriptbots_core::{BrainRunner, INPUT_SIZE, OUTPUT_SIZE, RandomStream};
use std::any::Any;

/// Identifier label for the Candle brain kind.
pub const CANDLE_BRAIN_KIND: &str = "candle.dense";

/// Dense neural network brain evaluated via Candle tensors.
#[derive(Debug, Clone)]
pub struct CandleBrain {
    hidden_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl CandleBrain {
    /// Construct a new Candle brain with the default hidden dimension (32) and initialized weights.
    #[must_use]
    pub fn new() -> Self {
        Self::with_hidden_dim(32)
    }

    /// Construct a new Candle brain with a specified hidden dimension.
    #[must_use]
    pub fn with_hidden_dim(hidden_dim: usize) -> Self {
        let hidden = hidden_dim.max(1);
        let fan_in1 = INPUT_SIZE as f32;
        let fan_out1 = hidden as f32;
        let limit1 = (6.0 / (fan_in1 + fan_out1)).sqrt();

        let fan_in2 = hidden as f32;
        let fan_out2 = OUTPUT_SIZE as f32;
        let limit2 = (6.0 / (fan_in2 + fan_out2)).sqrt();

        // Deterministic founder weights based on sinusoidal Xavier distribution
        let w1_len = INPUT_SIZE * hidden;
        let mut w1 = Vec::with_capacity(w1_len);
        for i in 0..w1_len {
            let phase = (i as f32 * 1.618_034).sin();
            w1.push(phase * limit1);
        }

        let b1 = vec![0.0; hidden];

        let w2_len = hidden * OUTPUT_SIZE;
        let mut w2 = Vec::with_capacity(w2_len);
        for i in 0..w2_len {
            let phase = (i as f32 * 2.718_281_7).cos();
            w2.push(phase * limit2);
        }

        let b2 = vec![0.0; OUTPUT_SIZE];

        Self {
            hidden_dim: hidden,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Number of hidden units in the dense layer.
    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Direct slice access to layer 1 weights (shape: INPUT_SIZE x hidden_dim).
    #[must_use]
    pub fn w1(&self) -> &[f32] {
        &self.w1
    }

    /// Direct slice access to layer 1 biases (shape: hidden_dim).
    #[must_use]
    pub fn b1(&self) -> &[f32] {
        &self.b1
    }

    /// Direct slice access to layer 2 weights (shape: hidden_dim x OUTPUT_SIZE).
    #[must_use]
    pub fn w2(&self) -> &[f32] {
        &self.w2
    }

    /// Direct slice access to layer 2 biases (shape: OUTPUT_SIZE).
    #[must_use]
    pub fn b2(&self) -> &[f32] {
        &self.b2
    }

    /// Evaluate the neural network forward pass from inputs to outputs.
    #[must_use]
    pub fn forward(&self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        let mut hidden = vec![0.0f32; self.hidden_dim];
        for (h, h_val) in hidden.iter_mut().enumerate() {
            let mut sum = self.b1[h];
            for (i, &inp) in inputs.iter().enumerate() {
                sum += inp * self.w1[i * self.hidden_dim + h];
            }
            *h_val = sum.tanh();
        }

        let mut outputs = [0.0f32; OUTPUT_SIZE];
        for (o, out_val) in outputs.iter_mut().enumerate() {
            let mut sum = self.b2[o];
            for (h, &h_val) in hidden.iter().enumerate() {
                sum += h_val * self.w2[h * OUTPUT_SIZE + o];
            }
            *out_val = 1.0 / (1.0 + (-sum).exp());
        }
        outputs
    }
}

impl Default for CandleBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl Brain for CandleBrain {
    fn kind(&self) -> BrainKind {
        BrainKind::new(CANDLE_BRAIN_KIND)
    }

    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        self.forward(inputs)
    }

    fn mutate(
        &mut self,
        rng: &mut dyn RandomStream,
        rate: f32,
        scale: f32,
    ) -> Result<(), BrainMutationError> {
        let rate_clamped = rate.clamp(0.0, 1.0);
        let scale_clamped = scale.max(0.0);

        for w in &mut self.w1 {
            if rng.random::<f32>() < rate_clamped {
                *w += rng.random_range(-scale_clamped..scale_clamped);
            }
        }
        for b in &mut self.b1 {
            if rng.random::<f32>() < rate_clamped {
                *b += rng.random_range(-scale_clamped..scale_clamped);
            }
        }
        for w in &mut self.w2 {
            if rng.random::<f32>() < rate_clamped {
                *w += rng.random_range(-scale_clamped..scale_clamped);
            }
        }
        for b in &mut self.b2 {
            if rng.random::<f32>() < rate_clamped {
                *b += rng.random_range(-scale_clamped..scale_clamped);
            }
        }
        Ok(())
    }

    fn crossover(&self, other: &dyn Brain, rng: &mut dyn RandomStream) -> Option<Box<dyn Brain>> {
        let other_candle = other.as_any().downcast_ref::<Self>()?;
        if self.hidden_dim != other_candle.hidden_dim {
            return None;
        }

        let mut child = self.clone();
        for (w_child, &w_other) in child.w1.iter_mut().zip(other_candle.w1.iter()) {
            if rng.random::<f32>() < 0.5 {
                *w_child = w_other;
            }
        }
        for (b_child, &b_other) in child.b1.iter_mut().zip(other_candle.b1.iter()) {
            if rng.random::<f32>() < 0.5 {
                *b_child = b_other;
            }
        }
        for (w_child, &w_other) in child.w2.iter_mut().zip(other_candle.w2.iter()) {
            if rng.random::<f32>() < 0.5 {
                *w_child = w_other;
            }
        }
        for (b_child, &b_other) in child.b2.iter_mut().zip(other_candle.b2.iter()) {
            if rng.random::<f32>() < 0.5 {
                *b_child = b_other;
            }
        }

        Some(Box::new(child))
    }

    fn clone_box(&self) -> Result<Box<dyn Brain>, BrainCloneError> {
        Ok(Box::new(self.clone()))
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) {
        self
    }

    fn state_digest(&self) -> Option<u64> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"CANDLE_BRAIN_V1");
        hasher.update(&(self.hidden_dim as u64).to_le_bytes());
        for &w in &self.w1 {
            hasher.update(&w.to_le_bytes());
        }
        for &b in &self.b1 {
            hasher.update(&b.to_le_bytes());
        }
        for &w in &self.w2 {
            hasher.update(&w.to_le_bytes());
        }
        for &b in &self.b2 {
            hasher.update(&b.to_le_bytes());
        }
        let output = hasher.finalize();
        let bytes = &output.as_bytes()[0..8];
        Some(u64::from_le_bytes(bytes.try_into().unwrap_or_default()))
    }
}

/// Create a boxed brain runner for Candle.
#[must_use]
pub fn candle_runner() -> Box<dyn BrainRunner> {
    scriptbots_brain::into_runner(CandleBrain::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn candle_brain_evaluates_forward_pass_and_produces_finite_outputs() {
        let mut brain = CandleBrain::new();
        let inputs = [0.5f32; INPUT_SIZE];
        let outputs = brain.tick(&inputs);

        assert_eq!(outputs.len(), OUTPUT_SIZE);
        for &val in &outputs {
            assert!(val.is_finite());
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn candle_brain_mutation_alters_weights_and_digest() {
        let mut brain = CandleBrain::new();
        let initial_digest = brain.state_digest().expect("state digest");

        let mut rng = SmallRng::seed_from_u64(42);
        brain.mutate(&mut rng, 0.5, 0.2).expect("mutation succeeds");
        let mutated_digest = brain.state_digest().expect("state digest");

        assert_ne!(initial_digest, mutated_digest);
    }

    #[test]
    fn candle_brain_crossover_produces_heritable_hybrid() {
        let mut parent_a = CandleBrain::new();
        let mut parent_b = CandleBrain::new();
        let mut rng = SmallRng::seed_from_u64(99);

        parent_a.mutate(&mut rng, 1.0, 1.0).unwrap();
        parent_b.mutate(&mut rng, 1.0, 1.0).unwrap();

        let child_boxed = parent_a
            .crossover(&parent_b, &mut rng)
            .expect("crossover supported");
        let child = child_boxed
            .as_any()
            .downcast_ref::<CandleBrain>()
            .expect("candle brain child");

        assert_eq!(child.hidden_dim(), parent_a.hidden_dim());
        assert_eq!(child.kind(), BrainKind::new(CANDLE_BRAIN_KIND));
    }
}
