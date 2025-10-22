//! Experimental assembly-style brain gated behind the `experimental` feature.

use rand::Rng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::any::Any;

use scriptbots_core::{BrainRunner, INPUT_SIZE, OUTPUT_SIZE};

use crate::{Brain, BrainKind, into_runner};

const BRAIN_SIZE: usize = 200;

/// Assembly-like instruction brain mirroring the legacy implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyBrain {
    cells: Vec<f32>,
}

impl AssemblyBrain {
    /// Trait identifier for this brain.
    pub const KIND: BrainKind = BrainKind::new("assembly.experimental");

    /// Construct a randomly initialized assembly brain.
    #[must_use]
    pub fn random(rng: &mut dyn RngCore) -> Self {
        let mut cells = Vec::with_capacity(BRAIN_SIZE);
        for _ in 0..BRAIN_SIZE {
            let mut value = rng.random_range(-3.0..3.0);
            if rng.random::<f32>() < 0.1 {
                value = rng.random_range(0.0..0.5);
            }
            if rng.random::<f32>() < 0.1 {
                value = rng.random_range(0.8..1.0);
            }
            cells.push(value);
        }

        Self { cells }
    }

    /// Return a boxed runner for this brain implementation.
    #[must_use]
    pub fn runner(rng: &mut dyn RngCore) -> Box<dyn BrainRunner> {
        into_runner(Self::random(rng))
    }

    fn clamp_index(value: f32) -> usize {
        let abs_value = value.abs();
        let fractional = abs_value - abs_value.floor();
        let idx = (fractional * BRAIN_SIZE as f32).floor() as isize;
        idx.clamp(0, (BRAIN_SIZE - 1) as isize) as usize
    }

    fn clamp_cells(cells: &mut [f32]) {
        for value in cells {
            if *value > 10.0 {
                *value = 10.0;
            } else if *value < -10.0 {
                *value = -10.0;
            }
        }
    }
}

impl Brain for AssemblyBrain {
    fn kind(&self) -> BrainKind {
        Self::KIND
    }

    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        for (idx, input) in inputs.iter().enumerate() {
            self.cells[idx] = *input;
        }

        for i in INPUT_SIZE..(BRAIN_SIZE - OUTPUT_SIZE) {
            let op = self.cells[i];
            if !(2.0..3.0).contains(&op) {
                continue;
            }

            let v1 = self.cells.get(i + 1).copied().unwrap_or_default();
            let v2 = self.cells.get(i + 2).copied().unwrap_or_default();
            let v3 = self.cells.get(i + 3).copied().unwrap_or_default();

            let d1 = Self::clamp_index(v1);
            let d2 = Self::clamp_index(v2);
            let d3 = Self::clamp_index(v3);

            if op < 2.1 {
                self.cells[d3] = self.cells[d1] + self.cells[d2];
                continue;
            }
            if op < 2.2 {
                self.cells[d3] = self.cells[d1] - self.cells[d2];
                continue;
            }
            if op < 2.3 {
                self.cells[d3] = self.cells[d1] * self.cells[d2];
                continue;
            }
            if op < 2.4 {
                if self.cells[d3] > 0.0 {
                    self.cells[d1] = 0.0;
                }
                continue;
            }
            if op < 2.5 {
                if self.cells[d3] > 0.0 {
                    self.cells[d1] = -self.cells[d1];
                }
                continue;
            }
            if op < 2.7 {
                if self.cells[d3] > 0.0 {
                    self.cells[d1] += v2;
                }
                continue;
            }
            if self.cells[d3] > 0.0 {
                self.cells[d1] = self.cells[d2];
            }
        }

        Self::clamp_cells(&mut self.cells[INPUT_SIZE..(BRAIN_SIZE - OUTPUT_SIZE)]);

        let mut outputs = [0.0; OUTPUT_SIZE];
        for (offset, output) in outputs.iter_mut().enumerate() {
            let idx = BRAIN_SIZE - 1 - offset;
            let value = self.cells[idx].clamp(0.0, 1.0);
            *output = value;
        }

        outputs
    }

    fn mutate(&mut self, rng: &mut dyn RngCore, rate: f32, _scale: f32) {
        for cell in &mut self.cells {
            if rng.random::<f32>() < rate {
                *cell = rng.random_range(-3.0..3.0);
            }
        }
    }

    fn crossover(&self, other: &dyn Brain, rng: &mut dyn RngCore) -> Option<Box<dyn Brain>> {
        if other.kind() != Self::KIND {
            return None;
        }

        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return None;
        };

        let mut child = self.clone();
        for (value, other_value) in child.cells.iter_mut().zip(&other.cells) {
            if rng.random::<f32>() < 0.5 {
                *value = *other_value;
            }
        }

        Some(Box::new(child))
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
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn random_brain_has_expected_length() {
        let mut rng = SmallRng::seed_from_u64(0xABCD);
        let brain = AssemblyBrain::random(&mut rng);
        assert_eq!(brain.cells.len(), BRAIN_SIZE);
    }

    #[test]
    fn tick_outputs_in_range() {
        let mut rng = SmallRng::seed_from_u64(4242);
        let mut brain = AssemblyBrain::random(&mut rng);
        let inputs = [0.5; INPUT_SIZE];
        let outputs = brain.tick(&inputs);
        assert!(outputs.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn mutate_changes_cells() {
        let mut rng = SmallRng::seed_from_u64(1717);
        let mut brain = AssemblyBrain::random(&mut rng);
        let before = brain.cells[10];
        brain.mutate(&mut rng, 1.0, 0.5);
        assert_ne!(brain.cells[10], before);
    }

    #[test]
    fn crossover_selects_values() {
        let mut rng = SmallRng::seed_from_u64(9999);
        let brain_a = AssemblyBrain::random(&mut rng);
        let brain_b = AssemblyBrain::random(&mut rng);
        let mut rng = SmallRng::seed_from_u64(1111);
        let child = brain_a
            .crossover(&brain_b, &mut rng)
            .expect("matching kinds");
        assert_eq!(child.kind(), AssemblyBrain::KIND);
    }

    #[test]
    fn runner_executes_program() {
        let mut rng = SmallRng::seed_from_u64(2025);
        let mut runner = AssemblyBrain::runner(&mut rng);
        let inputs = [0.0; INPUT_SIZE];
        let outputs = runner.tick(&inputs);
        assert!(outputs.iter().all(|v| v.is_finite()));
    }
}
