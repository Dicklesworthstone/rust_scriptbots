//! Optional modern ML brain backends (Candle, tract-onnx, tch).

use rand::Rng;
use scriptbots_brain::Brain;

/// Supported ML backends selected at build time.
#[derive(Debug, Clone, Copy)]
pub enum MlBackendKind {
    #[cfg(feature = "candle")]
    Candle,
    #[cfg(feature = "tract")]
    Tract,
    #[cfg(feature = "tch")]
    Tch,
    #[cfg(not(any(feature = "candle", feature = "tract", feature = "tch")))]
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
    fn kind(&self) -> &'static str {
        match self.kind {
            #[cfg(feature = "candle")]
            MlBackendKind::Candle => "candle",
            #[cfg(feature = "tract")]
            MlBackendKind::Tract => "tract-onnx",
            #[cfg(feature = "tch")]
            MlBackendKind::Tch => "tch",
            #[cfg(not(any(feature = "candle", feature = "tract", feature = "tch")))]
            MlBackendKind::None => "ml-placeholder",
        }
    }

    fn tick(&mut self, inputs: &[f32]) -> Vec<f32> {
        // Placeholder: pass-through until concrete integrations land.
        inputs.to_vec()
    }

    fn mutate<R: Rng>(&mut self, _rng: &mut R, _rate: f32, _scale: f32) {
        // Mutation behavior will be implemented per-backend as we integrate models.
    }
}
