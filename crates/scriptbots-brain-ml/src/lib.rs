//! Optional modern ML brain backends (Candle, tract-onnx, tch).

use scriptbots_brain::{Brain, BrainKind, into_runner};
use scriptbots_core::{BrainRunner, INPUT_SIZE, OUTPUT_SIZE};
use std::any::Any;

/// Supported ML backends selected at build time.
#[derive(Debug, Clone, Copy, Default)]
pub enum MlBackendKind {
    Candle,
    Tract,
    Tch,
    #[default]
    None,
}

/// Placeholder structure that will host the chosen ML model.
#[derive(Debug, Default)]
pub struct MlBrain {
    kind: MlBackendKind,
}

impl MlBrain {
    /// Construct a new ML brain instance using the active backend feature.
    #[must_use]
    pub fn new() -> Self {
        let kind = if cfg!(feature = "candle") {
            MlBackendKind::Candle
        } else if cfg!(feature = "tract") {
            MlBackendKind::Tract
        } else if cfg!(feature = "tch") {
            MlBackendKind::Tch
        } else {
            MlBackendKind::None
        };

        Self { kind }
    }

    /// Returns which backend is active.
    #[must_use]
    pub const fn backend(&self) -> MlBackendKind {
        self.kind
    }
}

impl Brain for MlBrain {
    fn kind(&self) -> BrainKind {
        let name = match self.kind {
            MlBackendKind::Candle => "ml.candle",
            MlBackendKind::Tract => "ml.tract-onnx",
            MlBackendKind::Tch => "ml.tch",
            MlBackendKind::None => "ml.placeholder",
        };
        BrainKind::new(name)
    }

    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        // Placeholder: copy the first OUTPUT_SIZE sensors to outputs.
        let mut outputs = [0.0; OUTPUT_SIZE];
        let len = OUTPUT_SIZE.min(INPUT_SIZE);
        outputs[..len].copy_from_slice(&inputs[..len]);
        outputs
    }

    fn mutate(&mut self, _rng: &mut dyn rand::RngCore, _rate: f32, _scale: f32) {
        // Mutation behavior will be implemented per-backend as we integrate models.
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) {
        self
    }
}

/// Create a boxed brain runner for the active ML backend.
#[must_use]
pub fn runner() -> Box<dyn BrainRunner> {
    into_runner(MlBrain::new())
}
