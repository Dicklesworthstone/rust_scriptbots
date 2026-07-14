//! NeuroFlow-backed brain implementation gated behind an opt-in feature.
//!
//! This module wraps the `neuroflow` crate’s [`FeedForward`] network so that it can participate in
//! the ScriptBots brain registry. The integration intentionally keeps configuration minimal while
//! remaining forward-compatible with richer training workflows. The implementation focuses on
//! inference; mutation perturbs inherited weights while explicit regeneration rebuilds the recorded
//! architecture through the same validated constructor.

use neuroflow::FeedForward;
use neuroflow::activators::Type;
use rand::Rng;
use serde::{Deserialize, Serialize};

use scriptbots_brain::{Brain, BrainCloneError, BrainKind, BrainMutationError, into_runner};
use scriptbots_core::{
    ActivationLayer, BrainActivations, BrainRunner, BrainSpawnError, NeuroflowActivationKind,
    NeuroflowSettings, RandomStream, ScientificStateError, WorldState,
};
use std::sync::Arc;

/// Number of inputs inherited from the simulation sensors.
const INPUT_SIZE: usize = scriptbots_core::INPUT_SIZE;
/// Number of outputs consumed by the actuation stage.
const OUTPUT_SIZE: usize = scriptbots_core::OUTPUT_SIZE;
/// Defensive ceiling on configurable hidden-layer count.
const MAX_HIDDEN_LAYERS: usize = 64;
/// Defensive ceiling on a single hidden layer.
const MAX_LAYER_NEURONS: usize = 65_536;
/// Defensive ceiling on the total deterministic weight matrix.
const MAX_NETWORK_WEIGHTS: usize = 1_048_576;

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
    /// Activation supplied to NeuroFlow (non-final layers, or the sole direct-output layer).
    pub activation: NeuroflowActivation,
    /// Learning rate baked into the network (relevant if online learning is enabled later).
    pub learning_rate: f64,
    /// Momentum factor used by NeuroFlow’s trainer.
    pub momentum: f64,
}

/// Failure while validating or constructing a NeuroFlow-backed brain.
#[derive(Debug)]
pub enum NeuroflowBrainError {
    /// The world rejected registry mutation at an unresolved persistence boundary.
    ScientificState(ScientificStateError),
    /// A public floating-point field was NaN or infinite.
    NonFinite { field: &'static str, value: f64 },
    /// A layer dimension cannot be represented by NeuroFlow or is zero.
    InvalidDimension {
        field: String,
        value: usize,
        requirement: &'static str,
    },
    /// Architecture arithmetic overflowed before allocation or library construction.
    ArchitectureOverflow { context: &'static str },
    /// Rust could not reserve storage for the validated architecture.
    Allocation {
        context: String,
        source: std::collections::TryReserveError,
    },
    /// The deterministic RNG produced a weight outside the promised finite interval.
    InvalidGeneratedWeight {
        layer: usize,
        neuron: usize,
        weight: usize,
        value: f64,
    },
    /// The validated deterministic seed could not be encoded.
    SeedSerialization { source: serde_json::Error },
    /// NeuroFlow rejected the validated deterministic seed representation.
    NetworkConstruction { source: serde_json::Error },
}

impl NeuroflowBrainError {
    /// Public configuration field implicated by a validation error, when applicable.
    #[must_use]
    pub fn field(&self) -> Option<&str> {
        match self {
            Self::NonFinite { field, .. } => Some(field),
            Self::InvalidDimension { field, .. } => Some(field),
            _ => None,
        }
    }
}

impl std::fmt::Display for NeuroflowBrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScientificState(error) => std::fmt::Display::fmt(error, f),
            Self::NonFinite { field, value } => {
                write!(f, "NeuroFlow `{field}` must be finite, got {value}")
            }
            Self::InvalidDimension {
                field,
                value,
                requirement,
            } => write!(
                f,
                "NeuroFlow `{field}` has invalid dimension {value}: {requirement}"
            ),
            Self::ArchitectureOverflow { context } => {
                write!(
                    f,
                    "NeuroFlow architecture overflow while computing {context}"
                )
            }
            Self::Allocation { context, source } => {
                write!(f, "NeuroFlow allocation failed for {context}: {source}")
            }
            Self::InvalidGeneratedWeight {
                layer,
                neuron,
                weight,
                value,
            } => write!(
                f,
                "NeuroFlow generated weight layers[{layer}].neurons[{neuron}].weights[{weight}] must be finite and within [-1, 1), got {value}"
            ),
            Self::SeedSerialization { source } => {
                write!(f, "failed to serialize validated NeuroFlow seed: {source}")
            }
            Self::NetworkConstruction { source } => {
                write!(
                    f,
                    "failed to construct NeuroFlow network from validated seed: {source}"
                )
            }
        }
    }
}

impl std::error::Error for NeuroflowBrainError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ScientificState(error) => Some(error),
            Self::Allocation { source, .. } => Some(source),
            Self::SeedSerialization { source } | Self::NetworkConstruction { source } => {
                Some(source)
            }
            _ => None,
        }
    }
}

impl From<ScientificStateError> for NeuroflowBrainError {
    fn from(error: ScientificStateError) -> Self {
        Self::ScientificState(error)
    }
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

    /// Validate the complete adapter boundary before invoking NeuroFlow.
    pub fn validate(&self) -> Result<(), NeuroflowBrainError> {
        for (field, value) in [
            ("learning_rate", self.learning_rate),
            ("momentum", self.momentum),
        ] {
            if !value.is_finite() {
                return Err(NeuroflowBrainError::NonFinite { field, value });
            }
        }

        let _ = self.hidden_layers.len().checked_add(2).ok_or(
            NeuroflowBrainError::ArchitectureOverflow {
                context: "layer count",
            },
        )?;
        if self.hidden_layers.len() > MAX_HIDDEN_LAYERS {
            return Err(NeuroflowBrainError::InvalidDimension {
                field: "hidden_layers".to_owned(),
                value: self.hidden_layers.len(),
                requirement: "at most 64 hidden layers are supported",
            });
        }
        let mut previous = INPUT_SIZE;
        let mut total_weights = 0usize;
        for (index, &size) in self.hidden_layers.iter().enumerate() {
            let field = format!("hidden_layers[{index}]");
            if size == 0 {
                return Err(NeuroflowBrainError::InvalidDimension {
                    field,
                    value: size,
                    requirement: "layer sizes must be non-zero",
                });
            }
            if size > MAX_LAYER_NEURONS {
                return Err(NeuroflowBrainError::InvalidDimension {
                    field,
                    value: size,
                    requirement: "layer sizes must not exceed 65,536 neurons",
                });
            }
            if i32::try_from(size).is_err() {
                return Err(NeuroflowBrainError::InvalidDimension {
                    field,
                    value: size,
                    requirement: "layer sizes must fit NeuroFlow's positive i32 domain",
                });
            }
            let weights_per_neuron =
                previous
                    .checked_add(1)
                    .ok_or(NeuroflowBrainError::ArchitectureOverflow {
                        context: "weights per neuron",
                    })?;
            let layer_weights = size.checked_mul(weights_per_neuron).ok_or(
                NeuroflowBrainError::ArchitectureOverflow {
                    context: "hidden-layer weight count",
                },
            )?;
            total_weights = total_weights.checked_add(layer_weights).ok_or(
                NeuroflowBrainError::ArchitectureOverflow {
                    context: "total weight count",
                },
            )?;
            if total_weights > MAX_NETWORK_WEIGHTS {
                return Err(NeuroflowBrainError::InvalidDimension {
                    field: "total_weights".to_owned(),
                    value: total_weights,
                    requirement: "network must not exceed 1,048,576 weights",
                });
            }
            previous = size;
        }
        let output_weights = OUTPUT_SIZE
            .checked_mul(previous.checked_add(1).ok_or(
                NeuroflowBrainError::ArchitectureOverflow {
                    context: "output weights per neuron",
                },
            )?)
            .ok_or(NeuroflowBrainError::ArchitectureOverflow {
                context: "output-layer weight count",
            })?;
        let total_weights = total_weights.checked_add(output_weights).ok_or(
            NeuroflowBrainError::ArchitectureOverflow {
                context: "total weight count",
            },
        )?;
        if total_weights > MAX_NETWORK_WEIGHTS {
            return Err(NeuroflowBrainError::InvalidDimension {
                field: "total_weights".to_owned(),
                value: total_weights,
                requirement: "network must not exceed 1,048,576 weights",
            });
        }
        Ok(())
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
    pub fn new(
        config: NeuroflowBrainConfig,
        rng: &mut dyn RandomStream,
    ) -> Result<Self, NeuroflowBrainError> {
        let network = Self::build_network(&config, rng)?;
        Ok(Self {
            network,
            config,
            inputs: vec![0.0; INPUT_SIZE],
        })
    }

    /// Convenience helper to box the brain into a [`BrainRunner`].
    pub fn runner(
        config: NeuroflowBrainConfig,
        rng: &mut dyn RandomStream,
    ) -> Result<Box<dyn BrainRunner>, NeuroflowBrainError> {
        Self::new(config, rng).map(into_runner)
    }

    /// Register a NeuroFlow brain into the world registry and return its key.
    pub fn register(
        world: &mut WorldState,
        config: NeuroflowBrainConfig,
    ) -> Result<u64, NeuroflowBrainError> {
        config.validate()?;
        let config = Arc::new(config);
        let key = world
            .brain_registry_mut()?
            .register(Self::KIND.as_str(), move |rng| {
                Self::runner((*config).clone(), rng)
                    .map_err(|source| BrainSpawnError::new(Self::KIND.as_str(), source))
            });
        Ok(key)
    }

    /// Rebuild weights through the same validated, fallible constructor used at startup.
    pub fn try_regenerate(
        &mut self,
        rng: &mut dyn RandomStream,
    ) -> Result<(), NeuroflowBrainError> {
        self.network = Self::build_network(&self.config, rng)?;
        Ok(())
    }

    fn build_network(
        config: &NeuroflowBrainConfig,
        rng: &mut dyn RandomStream,
    ) -> Result<FeedForward, NeuroflowBrainError> {
        config.validate()?;
        let architecture_len = config.hidden_layers.len().checked_add(2).ok_or(
            NeuroflowBrainError::ArchitectureOverflow {
                context: "layer count",
            },
        )?;
        let mut architecture: Vec<i32> = Vec::new();
        architecture
            .try_reserve_exact(architecture_len)
            .map_err(|source| NeuroflowBrainError::Allocation {
                context: "architecture".to_owned(),
                source,
            })?;
        architecture.push(i32::try_from(INPUT_SIZE).map_err(|_| {
            NeuroflowBrainError::InvalidDimension {
                field: "input_size".to_owned(),
                value: INPUT_SIZE,
                requirement: "input size must fit NeuroFlow's positive i32 domain",
            }
        })?);
        for (index, &layer) in config.hidden_layers.iter().enumerate() {
            architecture.push(i32::try_from(layer).map_err(|_| {
                NeuroflowBrainError::InvalidDimension {
                    field: format!("hidden_layers[{index}]"),
                    value: layer,
                    requirement: "layer sizes must fit NeuroFlow's positive i32 domain",
                }
            })?);
        }
        architecture.push(i32::try_from(OUTPUT_SIZE).map_err(|_| {
            NeuroflowBrainError::InvalidDimension {
                field: "output_size".to_owned(),
                value: OUTPUT_SIZE,
                requirement: "output size must fit NeuroFlow's positive i32 domain",
            }
        })?);

        let layer_count =
            architecture
                .len()
                .checked_sub(1)
                .ok_or(NeuroflowBrainError::ArchitectureOverflow {
                    context: "layer count",
                })?;
        let mut layers = Vec::new();
        layers.try_reserve_exact(layer_count).map_err(|source| {
            NeuroflowBrainError::Allocation {
                context: "network layers".to_owned(),
                source,
            }
        })?;
        for (layer_index, window) in architecture.windows(2).enumerate() {
            let inputs = usize::try_from(window[0]).map_err(|_| {
                NeuroflowBrainError::ArchitectureOverflow {
                    context: "input dimension conversion",
                }
            })?;
            let outputs = usize::try_from(window[1]).map_err(|_| {
                NeuroflowBrainError::ArchitectureOverflow {
                    context: "output dimension conversion",
                }
            })?;
            let mut neurons = Vec::new();
            neurons.try_reserve_exact(outputs).map_err(|source| {
                NeuroflowBrainError::Allocation {
                    context: format!("layers[{layer_index}] neurons"),
                    source,
                }
            })?;
            let weight_count =
                inputs
                    .checked_add(1)
                    .ok_or(NeuroflowBrainError::ArchitectureOverflow {
                        context: "weights per neuron",
                    })?;
            for neuron_index in 0..outputs {
                let mut weights = Vec::new();
                weights.try_reserve_exact(weight_count).map_err(|source| {
                    NeuroflowBrainError::Allocation {
                        context: format!("layers[{layer_index}].neurons[{neuron_index}] weights"),
                        source,
                    }
                })?;
                for weight_index in 0..weight_count {
                    let value: f64 = rng.random_range(-1.0..1.0);
                    if !value.is_finite() || !(-1.0..1.0).contains(&value) {
                        return Err(NeuroflowBrainError::InvalidGeneratedWeight {
                            layer: layer_index,
                            neuron: neuron_index,
                            weight: weight_index,
                            value,
                        });
                    }
                    weights.push(value);
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

        let value = serde_json::to_value(&seed)
            .map_err(|source| NeuroflowBrainError::SeedSerialization { source })?;
        let mut network: FeedForward = serde_json::from_value(value)
            .map_err(|source| NeuroflowBrainError::NetworkConstruction { source })?;
        network
            .activation(config.activation.to_type())
            .learning_rate(config.learning_rate)
            .momentum(config.momentum);
        Ok(network)
    }

    fn gaussian(rng: &mut dyn RandomStream) -> f64 {
        let u1 = rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
        let u2 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
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

    fn mutate(
        &mut self,
        rng: &mut dyn RandomStream,
        rate: f32,
        scale: f32,
    ) -> Result<(), BrainMutationError> {
        if rate <= 0.0 {
            return Ok(());
        }
        // Perturb existing weights in place (per-weight coin at `rate`,
        // Gaussian step scaled by `scale`). Regenerating the whole network
        // would erase all inherited structure in one event.
        let sigma = f64::from(scale.max(1e-5));
        let mut value = serde_json::to_value(&self.network).map_err(|source| {
            BrainMutationError::new(NeuroflowBrainError::SeedSerialization { source })
        })?;
        let mut changed = false;
        if let Some(layers) = value.get_mut("layers").and_then(|v| v.as_array_mut()) {
            for (layer_index, layer) in layers.iter_mut().enumerate() {
                if let Some(neurons) = layer.get_mut("w").and_then(|v| v.as_array_mut()) {
                    for (neuron_index, weights) in neurons
                        .iter_mut()
                        .filter_map(|n| n.as_array_mut())
                        .enumerate()
                    {
                        for (weight_index, weight) in weights.iter_mut().enumerate() {
                            if rng.random::<f32>() < rate
                                && let Some(current) = weight.as_f64()
                            {
                                let next = current + Self::gaussian(rng) * sigma;
                                let number =
                                    serde_json::Number::from_f64(next).ok_or_else(|| {
                                        BrainMutationError::new(
                                            NeuroflowBrainError::InvalidGeneratedWeight {
                                                layer: layer_index,
                                                neuron: neuron_index,
                                                weight: weight_index,
                                                value: next,
                                            },
                                        )
                                    })?;
                                *weight = serde_json::Value::Number(number);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        if changed {
            let mut network = serde_json::from_value::<FeedForward>(value).map_err(|source| {
                BrainMutationError::new(NeuroflowBrainError::NetworkConstruction { source })
            })?;
            network
                .activation(self.config.activation.to_type())
                .learning_rate(self.config.learning_rate)
                .momentum(self.config.momentum);
            self.network = network;
        }
        Ok(())
    }

    fn clone_box(&self) -> Result<Box<dyn Brain>, BrainCloneError> {
        let value = serde_json::to_value(&self.network).map_err(|source| {
            BrainCloneError::new(NeuroflowBrainError::SeedSerialization { source })
        })?;
        let mut network: FeedForward = serde_json::from_value(value).map_err(|source| {
            BrainCloneError::new(NeuroflowBrainError::NetworkConstruction { source })
        })?;
        network
            .activation(self.config.activation.to_type())
            .learning_rate(self.config.learning_rate)
            .momentum(self.config.momentum);
        Ok(Box::new(Self {
            network,
            config: self.config.clone(),
            inputs: vec![0.0; INPUT_SIZE],
        }))
    }

    fn as_any(&self) -> &(dyn std::any::Any + Send + Sync) {
        self
    }

    fn as_any_mut(&mut self) -> &mut (dyn std::any::Any + Send + Sync) {
        self
    }

    fn snapshot_activations(&self) -> Option<BrainActivations> {
        let value = serde_json::to_value(&self.network).ok()?;
        let layers = value.get("layers")?.as_array()?.to_vec();
        let mut result_layers: Vec<ActivationLayer> = Vec::new();
        for (li, layer_val) in layers.iter().enumerate() {
            let y = layer_val
                .get("y")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let values: Vec<f32> = y
                .into_iter()
                .filter_map(|v| v.as_f64())
                .map(|v| v as f32)
                .collect();
            let width = (values.len() as f32).sqrt().ceil() as usize;
            let height = if width == 0 {
                0
            } else {
                values.len().div_ceil(width)
            };
            result_layers.push(ActivationLayer {
                name: format!("nf.layer.{li}"),
                width,
                height,
                values,
            });
        }
        Some(BrainActivations {
            layers: result_layers,
            connections: Vec::new(),
            truncated: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::SmallRngStream;

    #[test]
    fn runner_executes_and_returns_outputs() {
        let mut rng = SmallRngStream::seed_from_u64(0xBEEF);
        let mut runner = NeuroflowBrain::runner(NeuroflowBrainConfig::default(), &mut rng)
            .expect("default NeuroFlow runner");
        let outputs = runner.tick(&[0.0; INPUT_SIZE]);
        assert_eq!(outputs.len(), OUTPUT_SIZE);
        assert!(outputs.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fallible_regeneration_changes_network() {
        let mut rng = SmallRngStream::seed_from_u64(0xCAFE);
        let config = NeuroflowBrainConfig::default();
        let mut brain = NeuroflowBrain::new(config, &mut rng).expect("default NeuroFlow brain");
        let baseline = brain.tick(&[0.0; INPUT_SIZE]);
        brain
            .try_regenerate(&mut rng)
            .expect("regenerate validated NeuroFlow brain");
        let after = brain.tick(&[0.0; INPUT_SIZE]);
        assert_ne!(baseline, after);
    }

    #[test]
    fn exact_clone_preserves_outputs_and_mutates_independently() {
        let mut rng = SmallRngStream::seed_from_u64(0xC10E);
        let mut original = NeuroflowBrain::new(NeuroflowBrainConfig::default(), &mut rng)
            .expect("validated NeuroFlow brain");
        let inputs = [0.25; INPUT_SIZE];
        let baseline = original.tick(&inputs);
        let mut inherited = original
            .clone_box()
            .expect("exact NeuroFlow snapshot must round-trip");

        assert_eq!(inherited.tick(&inputs), baseline);
        inherited
            .mutate(&mut rng, 1.0, 0.5)
            .expect("finite inherited mutation must rebuild exactly");
        assert_ne!(inherited.tick(&inputs), baseline);
        assert_eq!(original.tick(&inputs), baseline);
    }

    #[test]
    fn non_finite_public_rates_name_the_exact_field() {
        for (field, value) in [
            ("learning_rate", f64::NAN),
            ("learning_rate", f64::INFINITY),
            ("learning_rate", f64::NEG_INFINITY),
            ("momentum", f64::NAN),
            ("momentum", f64::INFINITY),
            ("momentum", f64::NEG_INFINITY),
        ] {
            let mut config = NeuroflowBrainConfig::default();
            match field {
                "learning_rate" => config.learning_rate = value,
                "momentum" => config.momentum = value,
                _ => unreachable!("test field table is exhaustive"),
            }
            let mut rng = SmallRngStream::seed_from_u64(7);
            let error = NeuroflowBrain::new(config, &mut rng)
                .expect_err("non-finite public rate must fail construction");
            assert_eq!(error.field(), Some(field));
            assert!(
                error.to_string().contains(field),
                "diagnostic did not name `{field}`: {error}"
            );
        }
    }

    #[test]
    fn representative_finite_signed_rates_preserve_construction() {
        for (learning_rate, momentum) in [(-0.25, -0.5), (-0.0, 0.0), (0.01, 0.05), (1.25, 2.0)] {
            let config = NeuroflowBrainConfig {
                hidden_layers: vec![1],
                learning_rate,
                momentum,
                ..NeuroflowBrainConfig::default()
            };
            let mut rng = SmallRngStream::seed_from_u64(11);
            let mut brain =
                NeuroflowBrain::new(config, &mut rng).expect("finite signed rates remain valid");
            assert!(
                brain
                    .tick(&[0.0; INPUT_SIZE])
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
    }

    #[test]
    fn layer_dimensions_are_checked_before_network_construction() {
        let zero = NeuroflowBrainConfig {
            hidden_layers: vec![8, 0, 4],
            ..NeuroflowBrainConfig::default()
        };
        let error = zero.validate().expect_err("zero-sized layer must fail");
        assert_eq!(error.field(), Some("hidden_layers[1]"));

        let too_wide = NeuroflowBrainConfig {
            hidden_layers: vec![(i32::MAX as usize) + 1],
            ..NeuroflowBrainConfig::default()
        };
        let error = too_wide
            .validate()
            .expect_err("layer beyond the defensive adapter domain must fail");
        assert_eq!(error.field(), Some("hidden_layers[0]"));

        let too_many_layers = NeuroflowBrainConfig {
            hidden_layers: vec![1; MAX_HIDDEN_LAYERS + 1],
            ..NeuroflowBrainConfig::default()
        };
        let error = too_many_layers
            .validate()
            .expect_err("excessive layer count must fail before allocation");
        assert_eq!(error.field(), Some("hidden_layers"));

        let too_many_weights = NeuroflowBrainConfig {
            hidden_layers: vec![1024, 1024],
            ..NeuroflowBrainConfig::default()
        };
        let error = too_many_weights
            .validate()
            .expect_err("excessive total weight count must fail before allocation");
        assert_eq!(error.field(), Some("total_weights"));

        let no_hidden = NeuroflowBrainConfig {
            hidden_layers: Vec::new(),
            ..NeuroflowBrainConfig::default()
        };
        let mut rng = SmallRngStream::seed_from_u64(13);
        NeuroflowBrain::new(no_hidden, &mut rng)
            .expect("direct input-to-output architecture remains valid");
    }

    #[test]
    fn empty_settings_preserve_the_existing_default_topology() {
        let settings = NeuroflowSettings {
            enabled: true,
            hidden_layers: Vec::new(),
            activation: NeuroflowActivationKind::Relu,
        };
        let config = NeuroflowBrainConfig::from_settings(&settings);
        assert_eq!(
            config.hidden_layers,
            NeuroflowBrainConfig::default().hidden_layers
        );
        assert!(matches!(config.activation, NeuroflowActivation::Relu));
    }

    #[test]
    fn invalid_registration_is_typed_and_leaves_registry_unchanged() {
        let mut world =
            WorldState::new(scriptbots_core::ScriptBotsConfig::default()).expect("test world");
        let before = world.brain_registry().descriptors();
        let invalid = NeuroflowBrainConfig {
            momentum: f64::NAN,
            ..NeuroflowBrainConfig::default()
        };
        let error = NeuroflowBrain::register(&mut world, invalid)
            .expect_err("invalid registration must fail before mutating the registry");
        assert_eq!(error.field(), Some("momentum"));
        assert_eq!(world.brain_registry().descriptors(), before);
    }
}
