#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "candle")]
pub use candle::{CANDLE_BRAIN_KIND, CandleBrain, candle_runner};

#[cfg(feature = "brain-ft")]
mod ft;

#[cfg(feature = "brain-ft")]
pub use ft::{FT_BRAIN_KIND, FtBrainConfig, FtBrainFamily};

use scriptbots_brain::{Brain, BrainCloneError, BrainKind, BrainMutationError, into_runner};
use scriptbots_core::{BrainRunner, INPUT_SIZE, OUTPUT_SIZE, RandomStream};
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

/// Unified ML brain structure that hosts the active ML backend.
#[derive(Debug, Clone)]
pub struct MlBrain {
    kind: MlBackendKind,
    #[cfg(feature = "candle")]
    candle: Option<CandleBrain>,
}

impl Default for MlBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl MlBrain {
    /// Construct a new ML brain instance using the active backend feature.
    #[must_use]
    pub fn new() -> Self {
        #[cfg(feature = "candle")]
        {
            Self {
                kind: MlBackendKind::Candle,
                candle: Some(CandleBrain::new()),
            }
        }
        #[cfg(not(feature = "candle"))]
        {
            let kind = if cfg!(feature = "tract") {
                MlBackendKind::Tract
            } else if cfg!(feature = "tch") {
                MlBackendKind::Tch
            } else {
                MlBackendKind::None
            };
            Self { kind }
        }
    }

    /// Returns which backend is active.
    #[must_use]
    pub const fn backend(&self) -> MlBackendKind {
        self.kind
    }
}

impl Brain for MlBrain {
    fn kind(&self) -> BrainKind {
        #[cfg(feature = "candle")]
        if self.candle.is_some() {
            return BrainKind::new(CANDLE_BRAIN_KIND);
        }
        BrainKind::new("ml.placeholder")
    }

    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        #[cfg(feature = "candle")]
        if let Some(ref mut candle) = self.candle {
            return candle.tick(inputs);
        }

        // Fallback placeholder: copy the first OUTPUT_SIZE sensors to outputs.
        let mut outputs = [0.0; OUTPUT_SIZE];
        let len = OUTPUT_SIZE.min(INPUT_SIZE);
        outputs[..len].copy_from_slice(&inputs[..len]);
        outputs
    }

    fn mutate(
        &mut self,
        rng: &mut dyn RandomStream,
        rate: f32,
        scale: f32,
    ) -> Result<(), BrainMutationError> {
        #[cfg(feature = "candle")]
        if let Some(ref mut candle) = self.candle {
            return candle.mutate(rng, rate, scale);
        }
        let _ = (rng, rate, scale);
        Ok(())
    }

    fn crossover(&self, other: &dyn Brain, rng: &mut dyn RandomStream) -> Option<Box<dyn Brain>> {
        #[cfg(feature = "candle")]
        if let Some(ref candle) = self.candle {
            if let Some(other_ml) = other.as_any().downcast_ref::<Self>() {
                if let Some(ref other_candle) = other_ml.candle {
                    let child_candle = candle.crossover(other_candle, rng)?;
                    let downcasted = child_candle.as_any().downcast_ref::<CandleBrain>()?.clone();
                    return Some(Box::new(Self {
                        kind: MlBackendKind::Candle,
                        candle: Some(downcasted),
                    }));
                }
            }
        }
        let _ = (other, rng);
        None
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
        #[cfg(feature = "candle")]
        if let Some(ref candle) = self.candle {
            return candle.state_digest();
        }
        None
    }
}

/// Create a boxed brain runner for the active ML backend.
#[must_use]
pub fn runner() -> Box<dyn BrainRunner> {
    into_runner(MlBrain::new())
}
