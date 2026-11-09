//! Frankentorch-backed dense feed-forward brain family.
//!
//! The heritable representation is a bounded, versioned, flat `Vec<f32>`. Founder
//! construction uses `ft_nn::parameters_to_vector` as the canonical flattening
//! oracle. The pinned Frankentorch `vector_to_parameters` implementation rebuilds
//! tensors through F64 storage, so inference deliberately materializes validated
//! F32 layer slices directly until that upstream API is F32-safe.

use ft_api::FrankenTorchSession;
use ft_core::{DType, Device, ExecutionMode, TensorMeta};
use scriptbots_core::{
    ActivationEdge, ActivationLayer, BrainActivations, BrainActivitySource, BrainAdapterIdentityV1,
    BrainBatchArchitectureKey, BrainBatchEvaluator, BrainEnvelopeKind, BrainEvaluator,
    BrainEvaluatorStateEnvelope, BrainFamilyCodec, BrainFamilyId, BrainGenomeEnvelope,
    BrainGenomeMaterial, BrainHeredityCapabilityV1, BrainInspection, BrainInspectionContext,
    BrainInspectionError, BrainInspectionLimits, BrainInspectionSnapshot,
    BrainInspectionUnavailable, BrainLocusSchemaIdentityV1, BrainMutationTrialGroupV1,
    BrainProtocolError, INPUT_SIZE, MAX_BRAIN_GENOME_PAYLOAD_BYTES, MutationRates, OUTPUT_SIZE,
    OffspringStatePolicy, OutputChannel, RandomStream, Tick, bound_brain_inspection,
};

/// Stable registry label for the Frankentorch brain family.
pub const FT_BRAIN_KIND: &str = "ft";

const FAMILY_ID: &str = "frankentorch-dense-v1";
const GENOME_MAGIC: [u8; 4] = *b"FTG1";
const STATE_MAGIC: [u8; 4] = *b"FTS1";
const GENOME_SCHEMA_VERSION: u32 = 1;
const GENOME_CODEC_VERSION: u16 = 1;
const STATE_SCHEMA_VERSION: u32 = 1;
const STATE_CODEC_VERSION: u16 = 1;
const PARAMETER_LAYOUT_VERSION: u16 = 1;
const OUTPUT_HEAD_SIGMOID: u8 = 1;
const MAX_HIDDEN_LAYERS: usize = 8;
const MAX_LAYER_WIDTH: usize = 256;
const GENOME_FIXED_HEADER_BYTES: usize = 16;
const STATE_PAYLOAD_BYTES: usize = STATE_MAGIC.len() + 32;
const ADAPTER_SEMANTIC_VERSION: u32 = 3;
const LOCUS_SCHEMA_SEMANTIC_VERSION: u32 = 1;
const LOCUS_SCHEMA_SEMANTIC_PREFIX: &[u8] = b"dense-f32-loci;inputs=25;outputs=9;\
    nodes=inputs-then-hidden-then-output;order=layer,node,bias,connection-weight-target;\
    connection-slot=incoming-column;target=previous-layer-source;\
    scalar-bits=canonical-weight-out-in-then-bias;structural-targets=immutable;\
    mutation-trials=one-next-u32-high24-less-than-primary-per-parameter";
const ADAPTER_SEMANTIC_PREFIX: &[u8] = b"dense-f32;\
    scalar-kernel=ft-kernel-cpu::linear_tensor_f32;\
    cohort-kernel=ft-kernel-cpu::bmm_tensor_contiguous_f32_into;\
    layer-order=weight[out,in],bias[out];hidden=tanh;heads=typed-all-sigmoid;\
    equivalence=scalar-packed-cohort-bit-identical;state=genome-digest-only;layout=v1;\
    founder=xavier-uniform-next-u32-bias-zero;\
    mutation=per-gene-box-muller-draw-preserving-zero-sigma-identity;\
    crossover=per-gene-uniform-next-u32";

/// Dense FtBrain architecture configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FtBrainConfig {
    hidden_layers: Vec<usize>,
}

impl FtBrainConfig {
    /// Construct a bounded dense architecture.
    pub fn new(hidden_layers: Vec<usize>) -> Result<Self, BrainProtocolError> {
        let config = Self { hidden_layers };
        validate_config(&config)?;
        Ok(config)
    }

    /// Hidden-layer widths in evaluation order.
    #[must_use]
    pub fn hidden_layers(&self) -> &[usize] {
        &self.hidden_layers
    }
}

impl Default for FtBrainConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![48, 32, 24],
        }
    }
}

/// Canonical versioned protocol family for the Frankentorch dense brain.
#[derive(Debug, Clone)]
pub struct FtBrainFamily {
    family_id: BrainFamilyId,
    config: FtBrainConfig,
}

#[derive(Debug, Clone, PartialEq)]
struct DecodedGenome {
    config: FtBrainConfig,
    parameters: Vec<f32>,
}

#[derive(Debug, Clone)]
struct DenseLayer {
    input_width: usize,
    output_width: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Debug, Clone)]
struct FtNetwork {
    layers: Vec<DenseLayer>,
}

struct PayloadReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PayloadReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let end = self.offset.checked_add(N)?;
        let bytes = self.bytes.get(self.offset..end)?;
        self.offset = end;
        bytes.try_into().ok()
    }

    fn exhausted(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

impl FtBrainFamily {
    /// Construct a family using the supplied dense architecture.
    pub fn new(config: FtBrainConfig) -> Result<Self, BrainProtocolError> {
        validate_config(&config)?;
        Ok(Self {
            family_id: canonical_family_id(),
            config,
        })
    }

    fn invalid(&self, kind: BrainEnvelopeKind, detail: impl Into<String>) -> BrainProtocolError {
        BrainProtocolError::InvalidPayload {
            kind,
            family_id: self.family_id.clone(),
            detail: detail.into(),
        }
    }

    fn read<const N: usize>(
        &self,
        reader: &mut PayloadReader<'_>,
        kind: BrainEnvelopeKind,
        label: &str,
    ) -> Result<[u8; N], BrainProtocolError> {
        reader
            .take::<N>()
            .ok_or_else(|| self.invalid(kind, format!("truncated while reading {label}")))
    }

    fn encode_genome_payload(
        &self,
        config: &FtBrainConfig,
        parameters: &[f32],
    ) -> Result<Vec<u8>, BrainProtocolError> {
        let parameter_count = validate_config(config)?;
        if config != &self.config {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                "genome architecture does not match this registered FtBrain family",
            ));
        }
        if parameters.len() != parameter_count {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                format!(
                    "FtBrain parameter vector requires {parameter_count} values, found {}",
                    parameters.len()
                ),
            ));
        }
        if let Some(index) = parameters.iter().position(|value| !value.is_finite()) {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                format!("FtBrain parameter {index} is not finite"),
            ));
        }

        let hidden_count = u8::try_from(config.hidden_layers.len()).map_err(|_| {
            self.invalid(
                BrainEnvelopeKind::Genome,
                "hidden-layer count exceeds the u8 wire field",
            )
        })?;
        let input_size = u16::try_from(INPUT_SIZE).map_err(|_| {
            self.invalid(
                BrainEnvelopeKind::Genome,
                "input size exceeds the u16 wire field",
            )
        })?;
        let output_size = u16::try_from(OUTPUT_SIZE).map_err(|_| {
            self.invalid(
                BrainEnvelopeKind::Genome,
                "output size exceeds the u16 wire field",
            )
        })?;
        let parameter_count_wire = u32::try_from(parameter_count).map_err(|_| {
            self.invalid(
                BrainEnvelopeKind::Genome,
                "parameter count exceeds the u32 wire field",
            )
        })?;
        let mut payload = Vec::with_capacity(
            GENOME_FIXED_HEADER_BYTES + config.hidden_layers.len() * 2 + parameters.len() * 4,
        );
        payload.extend_from_slice(&GENOME_MAGIC);
        payload.extend_from_slice(&input_size.to_le_bytes());
        payload.extend_from_slice(&output_size.to_le_bytes());
        payload.push(hidden_count);
        payload.push(OUTPUT_HEAD_SIGMOID);
        payload.extend_from_slice(&PARAMETER_LAYOUT_VERSION.to_le_bytes());
        payload.extend_from_slice(&parameter_count_wire.to_le_bytes());
        for &width in &config.hidden_layers {
            let width = u16::try_from(width).map_err(|_| {
                self.invalid(
                    BrainEnvelopeKind::Genome,
                    "hidden width exceeds the u16 wire field",
                )
            })?;
            payload.extend_from_slice(&width.to_le_bytes());
        }
        for &parameter in parameters {
            payload.extend_from_slice(&parameter.to_bits().to_le_bytes());
        }
        Ok(payload)
    }

    fn decode_genome(
        &self,
        genome: &BrainGenomeEnvelope,
    ) -> Result<DecodedGenome, BrainProtocolError> {
        genome.require_protocol(&self.family_id, GENOME_SCHEMA_VERSION, GENOME_CODEC_VERSION)?;
        let mut reader = PayloadReader::new(genome.payload());
        let magic = self.read::<4>(&mut reader, BrainEnvelopeKind::Genome, "genome magic")?;
        let input_size = u16::from_le_bytes(self.read::<2>(
            &mut reader,
            BrainEnvelopeKind::Genome,
            "input size",
        )?);
        let output_size = u16::from_le_bytes(self.read::<2>(
            &mut reader,
            BrainEnvelopeKind::Genome,
            "output size",
        )?);
        let hidden_count =
            usize::from(self.read::<1>(&mut reader, BrainEnvelopeKind::Genome, "hidden count")?[0]);
        let output_head = self.read::<1>(&mut reader, BrainEnvelopeKind::Genome, "output head")?[0];
        let layout = u16::from_le_bytes(self.read::<2>(
            &mut reader,
            BrainEnvelopeKind::Genome,
            "parameter layout",
        )?);
        let parameter_count = usize::try_from(u32::from_le_bytes(self.read::<4>(
            &mut reader,
            BrainEnvelopeKind::Genome,
            "parameter count",
        )?))
        .map_err(|_| {
            self.invalid(
                BrainEnvelopeKind::Genome,
                "parameter count does not fit this platform",
            )
        })?;
        if magic != GENOME_MAGIC
            || usize::from(input_size) != INPUT_SIZE
            || usize::from(output_size) != OUTPUT_SIZE
            || output_head != OUTPUT_HEAD_SIGMOID
            || layout != PARAMETER_LAYOUT_VERSION
        {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                format!(
                    "unsupported FtBrain header: magic={magic:?}, inputs={input_size}, outputs={output_size}, head={output_head}, layout={layout}"
                ),
            ));
        }
        if hidden_count == 0 || hidden_count > MAX_HIDDEN_LAYERS {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                format!("unsupported hidden-layer count {hidden_count}"),
            ));
        }
        let mut hidden_layers = Vec::with_capacity(hidden_count);
        for _ in 0..hidden_count {
            hidden_layers.push(usize::from(u16::from_le_bytes(self.read::<2>(
                &mut reader,
                BrainEnvelopeKind::Genome,
                "hidden width",
            )?)));
        }
        let config = FtBrainConfig::new(hidden_layers)?;
        if config != self.config {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                "genome architecture does not match this registered FtBrain family",
            ));
        }
        let expected_count = validate_config(&config)?;
        if parameter_count != expected_count {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                format!(
                    "header declares {parameter_count} parameters, architecture requires {expected_count}"
                ),
            ));
        }
        let mut parameters = Vec::with_capacity(parameter_count);
        for parameter_index in 0..parameter_count {
            let value = f32::from_bits(u32::from_le_bytes(self.read::<4>(
                &mut reader,
                BrainEnvelopeKind::Genome,
                "parameter",
            )?));
            if !value.is_finite() {
                return Err(self.invalid(
                    BrainEnvelopeKind::Genome,
                    format!("FtBrain parameter {parameter_index} is not finite"),
                ));
            }
            parameters.push(value);
        }
        if !reader.exhausted() {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                "FtBrain genome contains trailing bytes",
            ));
        }
        Ok(DecodedGenome { config, parameters })
    }

    fn genome_material(
        &self,
        config: &FtBrainConfig,
        parameters: &[f32],
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        BrainGenomeMaterial::new(
            GENOME_SCHEMA_VERSION,
            GENOME_CODEC_VERSION,
            self.encode_genome_payload(config, parameters)?,
        )
    }

    fn encode_state_payload(&self, genome: &BrainGenomeEnvelope) -> Vec<u8> {
        let mut payload = Vec::with_capacity(STATE_PAYLOAD_BYTES);
        payload.extend_from_slice(&STATE_MAGIC);
        payload.extend_from_slice(genome.material_hash().as_bytes());
        payload
    }

    fn decode_state(
        &self,
        state: &BrainEvaluatorStateEnvelope,
    ) -> Result<[u8; 32], BrainProtocolError> {
        state.require_protocol(&self.family_id, STATE_SCHEMA_VERSION, STATE_CODEC_VERSION)?;
        if state.payload().len() != STATE_PAYLOAD_BYTES {
            return Err(self.invalid(
                BrainEnvelopeKind::EvaluatorState,
                format!(
                    "FtBrain evaluator state requires {STATE_PAYLOAD_BYTES} bytes, found {}",
                    state.payload().len()
                ),
            ));
        }
        let mut reader = PayloadReader::new(state.payload());
        let magic = self.read::<4>(
            &mut reader,
            BrainEnvelopeKind::EvaluatorState,
            "state magic",
        )?;
        let digest = self.read::<32>(
            &mut reader,
            BrainEnvelopeKind::EvaluatorState,
            "genome digest",
        )?;
        if magic != STATE_MAGIC || !reader.exhausted() {
            return Err(self.invalid(
                BrainEnvelopeKind::EvaluatorState,
                format!("unsupported FtBrain evaluator-state magic {magic:?}"),
            ));
        }
        Ok(digest)
    }

    fn state_for(
        &self,
        genome: &BrainGenomeEnvelope,
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        BrainEvaluatorStateEnvelope::new(
            self.family_id.clone(),
            STATE_SCHEMA_VERSION,
            STATE_CODEC_VERSION,
            self.encode_state_payload(genome),
        )
    }

    fn validate_pair(
        &self,
        genome: &BrainGenomeEnvelope,
        state: &BrainEvaluatorStateEnvelope,
    ) -> Result<DecodedGenome, BrainProtocolError> {
        let decoded = self.decode_genome(genome)?;
        let state_digest = self.decode_state(state)?;
        if state_digest != *genome.material_hash().as_bytes() {
            return Err(self.invalid(
                BrainEnvelopeKind::EvaluatorState,
                "FtBrain evaluator state belongs to a different genome",
            ));
        }
        Ok(decoded)
    }

    fn validate_mutation_rates(&self, rates: MutationRates) -> Result<(), BrainProtocolError> {
        if !rates.primary.is_finite() || !(0.0..=1.0).contains(&rates.primary) {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                "primary mutation probability must be finite and within [0, 1]",
            ));
        }
        if !rates.secondary.is_finite() || rates.secondary < 0.0 {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                "secondary mutation sigma must be finite and nonnegative",
            ));
        }
        Ok(())
    }

    fn random_parameters(
        &self,
        rng: &mut dyn RandomStream,
    ) -> Result<Vec<f32>, BrainProtocolError> {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let mut parameter_nodes = Vec::new();
        for (input_width, output_width) in layer_shapes(&self.config) {
            let denominator = f32::from(
                u16::try_from(input_width + output_width)
                    .expect("validated FtBrain widths fit in u16"),
            );
            let limit = (6.0_f32 / denominator).sqrt();
            let weights: Vec<f32> = (0..input_width * output_width)
                .map(|_| signed_unit_f32(rng) * limit)
                .collect();
            let bias = vec![0.0; output_width];
            let weight_node = session
                .tensor_variable_f32(weights, vec![output_width, input_width], false)
                .map_err(|error| {
                    self.invalid(
                        BrainEnvelopeKind::Genome,
                        format!("failed to create F32 Frankentorch weight tensor: {error}"),
                    )
                })?;
            let bias_node = session
                .tensor_variable_f32(bias, vec![output_width], false)
                .map_err(|error| {
                    self.invalid(
                        BrainEnvelopeKind::Genome,
                        format!("failed to create F32 Frankentorch bias tensor: {error}"),
                    )
                })?;
            parameter_nodes.extend([weight_node, bias_node]);
        }
        let vector =
            ft_nn::parameters_to_vector(&mut session, &parameter_nodes).map_err(|error| {
                self.invalid(
                    BrainEnvelopeKind::Genome,
                    format!("Frankentorch parameters_to_vector failed: {error}"),
                )
            })?;
        let parameters = session.tensor_values_f32(vector).map_err(|error| {
            self.invalid(
                BrainEnvelopeKind::Genome,
                format!("failed to read flattened F32 parameters: {error}"),
            )
        })?;
        let expected = validate_config(&self.config)?;
        if parameters.len() != expected {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                format!(
                    "parameters_to_vector returned {} values, expected {expected}",
                    parameters.len()
                ),
            ));
        }
        Ok(parameters)
    }

    fn architecture_key(&self) -> Result<BrainBatchArchitectureKey, BrainProtocolError> {
        let mut key = Vec::with_capacity(8 + self.config.hidden_layers.len() * 2);
        key.extend_from_slice(&PARAMETER_LAYOUT_VERSION.to_le_bytes());
        key.push(OUTPUT_HEAD_SIGMOID);
        key.push(u8::try_from(self.config.hidden_layers.len()).map_err(|_| {
            self.invalid(
                BrainEnvelopeKind::Genome,
                "hidden-layer count exceeds the batch-key wire field",
            )
        })?);
        key.extend_from_slice(
            &u16::try_from(INPUT_SIZE)
                .map_err(|_| self.invalid(BrainEnvelopeKind::Genome, "input size overflow"))?
                .to_le_bytes(),
        );
        key.extend_from_slice(
            &u16::try_from(OUTPUT_SIZE)
                .map_err(|_| self.invalid(BrainEnvelopeKind::Genome, "output size overflow"))?
                .to_le_bytes(),
        );
        for &width in &self.config.hidden_layers {
            key.extend_from_slice(
                &u16::try_from(width)
                    .map_err(|_| self.invalid(BrainEnvelopeKind::Genome, "hidden width overflow"))?
                    .to_le_bytes(),
            );
        }
        BrainBatchArchitectureKey::new(key)
    }
}

impl Default for FtBrainFamily {
    fn default() -> Self {
        Self::new(FtBrainConfig::default()).expect("the built-in FtBrain topology is valid")
    }
}

impl FtNetwork {
    fn materialize(decoded: &DecodedGenome) -> Self {
        let mut offset = 0;
        let mut layers = Vec::new();
        for (input_width, output_width) in layer_shapes(&decoded.config) {
            let weight_count = input_width * output_width;
            let weight_end = offset + weight_count;
            let bias_end = weight_end + output_width;
            layers.push(DenseLayer {
                input_width,
                output_width,
                weights: decoded.parameters[offset..weight_end].to_vec(),
                bias: decoded.parameters[weight_end..bias_end].to_vec(),
            });
            offset = bias_end;
        }
        debug_assert_eq!(offset, decoded.parameters.len());
        Self { layers }
    }

    fn evaluate(
        &self,
        family: &FtBrainFamily,
        sensors: &[f32; INPUT_SIZE],
    ) -> Result<[f32; OUTPUT_SIZE], BrainProtocolError> {
        if sensors.iter().any(|value| !value.is_finite()) {
            return Err(family.invalid(
                BrainEnvelopeKind::EvaluatorState,
                "FtBrain sensor input contains a non-finite value",
            ));
        }
        let mut activations = sensors.to_vec();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            activations = ft_kernel_cpu::linear_tensor_f32(
                &activations,
                &layer.weights,
                Some(&layer.bias),
                1,
                layer.input_width,
                layer.output_width,
            );
            if layer_index + 1 == self.layers.len() {
                for value in &mut activations {
                    *value = sigmoid(*value);
                }
            } else {
                for value in &mut activations {
                    *value = value.tanh();
                }
            }
        }
        if activations.len() != OUTPUT_SIZE || activations.iter().any(|value| !value.is_finite()) {
            return Err(family.invalid(
                BrainEnvelopeKind::EvaluatorState,
                "FtBrain evaluation produced an invalid output vector",
            ));
        }
        let mut outputs = [0.0; OUTPUT_SIZE];
        for channel in OutputChannel::ALL {
            outputs[channel.index()] = activations[channel.index()];
        }
        Ok(outputs)
    }

    // Reuse the exact scalar kernels and activation order; this is observational work only.
    #[allow(clippy::too_many_lines)]
    fn inspect_recorded(
        &self,
        context: BrainInspectionContext<'_>,
        limits: BrainInspectionLimits,
    ) -> Result<BrainInspectionSnapshot, BrainInspectionError> {
        if context.source_tick == Tick::zero()
            || context
                .sensors
                .iter()
                .chain(context.outputs)
                .all(|value| *value == 0.0)
        {
            // A newly inserted agent has this pair. Even if an extreme sigmoid could also
            // produce it, no stored history distinguishes that response from initialization.
            return Err(BrainInspectionError::ActivityUnavailable {
                reason: BrainInspectionUnavailable::NoRecordedEvaluation,
            });
        }
        let node_count = INPUT_SIZE
            + self
                .layers
                .iter()
                .map(|layer| layer.output_width)
                .sum::<usize>();
        let source_edges = self
            .layers
            .iter()
            .map(|layer| layer.weights.len())
            .sum::<usize>();
        let source_scalars = node_count
            + self
                .layers
                .iter()
                .map(|layer| layer.weights.len() + layer.bias.len())
                .sum::<usize>();
        if source_scalars > limits.max_source_scalars() {
            return Err(BrainInspectionError::SourceScalarLimitExceeded {
                family: FT_BRAIN_KIND.to_owned(),
                required: source_scalars,
                limit: limits.max_source_scalars(),
            });
        }
        if context
            .sensors
            .iter()
            .chain(context.outputs)
            .any(|value| !value.is_finite())
        {
            return Err(BrainInspectionError::ActivityUnavailable {
                reason: BrainInspectionUnavailable::RecordedResponseMismatch,
            });
        }

        let mut flat_values = Vec::with_capacity(node_count);
        flat_values.extend_from_slice(context.sensors);
        let mut layers = Vec::with_capacity(self.layers.len() + 2);
        layers.push(ActivationLayer {
            name: "Reconstructed input".to_owned(),
            width: INPUT_SIZE,
            height: 1,
            values: context.sensors.to_vec(),
        });
        let mut activations = context.sensors.to_vec();
        for (index, layer) in self.layers.iter().enumerate() {
            activations = ft_kernel_cpu::linear_tensor_f32(
                &activations,
                &layer.weights,
                Some(&layer.bias),
                1,
                layer.input_width,
                layer.output_width,
            );
            let output_layer = index + 1 == self.layers.len();
            for value in &mut activations {
                *value = if output_layer {
                    sigmoid(*value)
                } else {
                    value.tanh()
                };
            }
            if activations.iter().any(|value| !value.is_finite()) {
                return Err(BrainInspectionError::ActivityUnavailable {
                    reason: BrainInspectionUnavailable::RecordedResponseMismatch,
                });
            }
            flat_values.extend_from_slice(&activations);
            layers.push(ActivationLayer {
                name: if output_layer {
                    "Reconstructed output".to_owned()
                } else {
                    format!("Reconstructed hidden {}", index + 1)
                },
                width: layer.output_width,
                height: 1,
                values: activations.clone(),
            });
        }
        if activations.len() != OUTPUT_SIZE
            || activations
                .iter()
                .zip(context.outputs)
                .any(|(reconstructed, recorded)| reconstructed.to_bits() != recorded.to_bits())
        {
            return Err(BrainInspectionError::ActivityUnavailable {
                reason: BrainInspectionUnavailable::RecordedResponseMismatch,
            });
        }

        // Core attribution indexes the final layer's flat state. Keep the named layer views
        // as well, but never reinterpret hidden-local indices as global connection endpoints.
        layers.push(ActivationLayer {
            name: "Reconstructed flat state".to_owned(),
            width: node_count,
            height: 1,
            values: flat_values,
        });
        let source_layers = layers.len();
        let source_name_bytes = layers.iter().map(|layer| layer.name.len()).sum();
        let source_values = node_count * 2;
        let edge_limit = source_edges.min(limits.max_edges());
        let mut connections = Vec::with_capacity(edge_limit);
        let mut source_start = 0;
        let mut node_start = INPUT_SIZE;
        for layer in &self.layers {
            for (weight_index, &weight) in layer.weights.iter().enumerate() {
                if connections.len() == edge_limit {
                    break;
                }
                connections.push(ActivationEdge {
                    from: source_start + weight_index % layer.input_width,
                    to: node_start + weight_index / layer.input_width,
                    weight,
                });
            }
            source_start = node_start;
            node_start += layer.output_width;
        }
        let mut snapshot = bound_brain_inspection(
            FT_BRAIN_KIND,
            BrainActivations {
                layers,
                connections,
                output_slots: (node_count - OUTPUT_SIZE..node_count).collect(),
                truncated: edge_limit < source_edges,
            },
            source_scalars,
            limits,
        )?;
        snapshot.activity_source = BrainActivitySource::ReconstructedFeedForward;
        snapshot.build.source_layers = source_layers;
        snapshot.build.source_name_bytes = source_name_bytes;
        snapshot.build.source_values = source_values;
        snapshot.build.source_edges = source_edges;
        Ok(snapshot)
    }
}

struct FtEvaluator {
    family: FtBrainFamily,
    genome_digest: [u8; 32],
    network: FtNetwork,
}

impl BrainEvaluator for FtEvaluator {
    fn family_id(&self) -> &BrainFamilyId {
        &self.family.family_id
    }

    fn evaluate(
        &mut self,
        sensors: &[f32; INPUT_SIZE],
    ) -> Result<[f32; OUTPUT_SIZE], BrainProtocolError> {
        self.network.evaluate(&self.family, sensors)
    }

    fn inspect(
        &self,
        _request: BrainInspection,
    ) -> Result<Option<BrainInspectionSnapshot>, BrainInspectionError> {
        Err(BrainInspectionError::ActivityUnavailable {
            reason: BrainInspectionUnavailable::NoRecordedEvaluation,
        })
    }

    fn inspect_with_context(
        &self,
        request: BrainInspection,
        context: BrainInspectionContext<'_>,
    ) -> Result<Option<BrainInspectionSnapshot>, BrainInspectionError> {
        let BrainInspection::Activations(limits) = request;
        self.network.inspect_recorded(context, limits).map(Some)
    }

    fn checkpoint_state(&self) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        state_from_digest(&self.family, self.genome_digest)
    }
}

struct FtBatchEvaluator {
    family: FtBrainFamily,
    genome_digests: Vec<[u8; 32]>,
    layers: Vec<FtBatchLayer>,
    activations: Vec<f32>,
    candidates: Vec<f32>,
}

/// All lanes share an architecture, but each lane has its own heritable weights.
/// A layer contracts `[lane, 1, input] @ [lane, input, output]`; sharing one
/// `[input, output]` matrix across the cohort would silently erase that diversity.
struct FtBatchLayer {
    input_width: usize,
    output_width: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    input_meta: TensorMeta,
    weight_meta: TensorMeta,
}

impl FtBatchLayer {
    fn new(
        family: &FtBrainFamily,
        lanes: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<Self, BrainProtocolError> {
        let weight_count = input_width
            .checked_mul(output_width)
            .and_then(|count| count.checked_mul(lanes))
            .ok_or_else(|| config_error("FtBrain cohort weight footprint overflow"))?;
        let bias_count = output_width
            .checked_mul(lanes)
            .ok_or_else(|| config_error("FtBrain cohort bias footprint overflow"))?;
        Ok(Self {
            input_width,
            output_width,
            weights: cohort_buffer(family, weight_count)?,
            biases: cohort_buffer(family, bias_count)?,
            input_meta: TensorMeta::from_shape(
                vec![lanes, 1, input_width],
                DType::F32,
                Device::Cpu,
            ),
            weight_meta: TensorMeta::from_shape(
                vec![lanes, input_width, output_width],
                DType::F32,
                Device::Cpu,
            ),
        })
    }

    fn pack_lane(&mut self, lane: usize, parameters: &[f32]) {
        let weight_count = self.input_width * self.output_width;
        let weights = &mut self.weights[lane * weight_count..(lane + 1) * weight_count];
        // The canonical genome stays weight[out, in], bias[out]. Only this
        // execution-owned copy is transposed, with no floating-point arithmetic.
        for output in 0..self.output_width {
            for input in 0..self.input_width {
                weights[input * self.output_width + output] =
                    parameters[output * self.input_width + input];
            }
        }
        self.biases[lane * self.output_width..(lane + 1) * self.output_width]
            .copy_from_slice(&parameters[weight_count..weight_count + self.output_width]);
    }
}

fn cohort_buffer(family: &FtBrainFamily, count: usize) -> Result<Vec<f32>, BrainProtocolError> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|error| {
        family.invalid(
            BrainEnvelopeKind::EvaluatorState,
            format!("cannot allocate FtBrain cohort buffer for {count} values: {error}"),
        )
    })?;
    values.resize(count, 0.0);
    Ok(values)
}

impl BrainBatchEvaluator for FtBatchEvaluator {
    fn family_id(&self) -> &BrainFamilyId {
        &self.family.family_id
    }

    fn evaluate_batch(
        &mut self,
        sensors: &[[f32; INPUT_SIZE]],
        outputs: &mut [[f32; OUTPUT_SIZE]],
    ) -> Result<(), BrainProtocolError> {
        let lanes = self.genome_digests.len();
        if lanes != sensors.len() || sensors.len() != outputs.len() {
            return Err(BrainProtocolError::BatchCardinalityMismatch {
                evaluators: lanes,
                inputs: sensors.len(),
                outputs: outputs.len(),
            });
        }
        if lanes == 0 {
            return Ok(());
        }
        if let Some(lane) = sensors
            .iter()
            .position(|input| input.iter().any(|value| !value.is_finite()))
        {
            return Err(self.family.invalid(
                BrainEnvelopeKind::EvaluatorState,
                format!("FtBrain sensor input in cohort lane {lane} contains a non-finite value"),
            ));
        }
        for (destination, input) in self.activations.chunks_exact_mut(INPUT_SIZE).zip(sensors) {
            destination.copy_from_slice(input);
        }

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let output_count = lanes * layer.output_width;
            let candidate = &mut self.candidates[..output_count];
            ft_kernel_cpu::bmm_tensor_contiguous_f32_into(
                &self.activations[..lanes * layer.input_width],
                &layer.weights,
                &layer.input_meta,
                &layer.weight_meta,
                candidate,
            )
            .map_err(|error| {
                self.family.invalid(
                    BrainEnvelopeKind::EvaluatorState,
                    format!("FtBrain cohort layer {layer_index} evaluation failed: {error}"),
                )
            })?;

            // Bias is added AFTER the kernel's reduction, exactly as in the
            // scalar linear_tensor_f32 oracle. Fusing it into the contraction
            // would change rounding and break scalar/cohort replay equivalence.
            let output_layer = layer_index + 1 == self.layers.len();
            for (value, bias) in candidate.iter_mut().zip(&layer.biases) {
                *value += *bias;
                *value = if output_layer {
                    sigmoid(*value)
                } else {
                    value.tanh()
                };
            }
            std::mem::swap(&mut self.activations, &mut self.candidates);
        }

        let final_activations = &self.activations[..lanes * OUTPUT_SIZE];
        if let Some(index) = final_activations
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(self.family.invalid(
                BrainEnvelopeKind::EvaluatorState,
                format!(
                    "FtBrain evaluation produced an invalid output in cohort lane {}",
                    index / OUTPUT_SIZE
                ),
            ));
        }
        // No caller output is published until every lane has succeeded. Scratch
        // is not evaluator state and is overwritten before the next attempt.
        for (output, activation) in outputs
            .iter_mut()
            .zip(final_activations.chunks_exact(OUTPUT_SIZE))
        {
            for channel in OutputChannel::ALL {
                output[channel.index()] = activation[channel.index()];
            }
        }
        Ok(())
    }

    fn checkpoint_states(&self) -> Result<Vec<BrainEvaluatorStateEnvelope>, BrainProtocolError> {
        self.genome_digests
            .iter()
            .map(|digest| state_from_digest(&self.family, *digest))
            .collect()
    }
}

impl BrainFamilyCodec for FtBrainFamily {
    fn family_id(&self) -> &BrainFamilyId {
        &self.family_id
    }

    fn adapter_identity(&self) -> BrainAdapterIdentityV1 {
        let mut descriptor = ADAPTER_SEMANTIC_PREFIX.to_vec();
        descriptor.push(u8::try_from(self.config.hidden_layers.len()).unwrap_or(u8::MAX));
        for &width in &self.config.hidden_layers {
            descriptor.extend_from_slice(&u16::try_from(width).unwrap_or(u16::MAX).to_le_bytes());
        }
        BrainAdapterIdentityV1::from_semantic_descriptor(
            &self.family_id,
            ADAPTER_SEMANTIC_VERSION,
            &descriptor,
        )
    }

    fn heredity_capability(&self) -> BrainHeredityCapabilityV1 {
        let mut descriptor = LOCUS_SCHEMA_SEMANTIC_PREFIX.to_vec();
        descriptor
            .push(u8::try_from(self.config.hidden_layers.len()).expect("validated layer count"));
        for &width in &self.config.hidden_layers {
            descriptor.extend_from_slice(
                &u16::try_from(width)
                    .expect("validated layer width")
                    .to_le_bytes(),
            );
        }
        let trials =
            u32::try_from(validate_config(&self.config).expect("validated FtBrain architecture"))
                .expect("bounded parameter count fits u32");
        BrainHeredityCapabilityV1::locus_capable(
            BrainLocusSchemaIdentityV1::from_semantic_descriptor(
                &self.family_id,
                LOCUS_SCHEMA_SEMANTIC_VERSION,
                &descriptor,
            ),
            vec![BrainMutationTrialGroupV1::new(trials, 1, 1)],
        )
    }

    fn random_genome_material(
        &self,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        self.genome_material(&self.config, &self.random_parameters(rng)?)
    }

    fn validate_genome(&self, genome: &BrainGenomeEnvelope) -> Result<(), BrainProtocolError> {
        self.decode_genome(genome).map(|_| ())
    }

    fn genome_loci(
        &self,
        genome: &BrainGenomeEnvelope,
    ) -> Result<
        Vec<(
            scriptbots_core::genome_diff::Locus,
            scriptbots_core::genome_diff::LocusValue,
        )>,
        BrainProtocolError,
    > {
        use scriptbots_core::genome_diff::{Locus, LocusValue};
        let decoded = self.decode_genome(genome)?;
        let mut loci = Vec::with_capacity(decoded.parameters.len() * 2);
        let mut parameter_start = 0;
        let mut source_start = 0;
        let mut node_start = INPUT_SIZE;
        for (input_width, output_width) in layer_shapes(&decoded.config) {
            let weights_end = parameter_start + input_width * output_width;
            for row in 0..output_width {
                let node = u32::try_from(node_start + row).expect("bounded FtBrain node index");
                loci.push((
                    Locus::NodeBias(node),
                    LocusValue::Scalar(decoded.parameters[weights_end + row]),
                ));
                for column in 0..input_width {
                    let conn = u8::try_from(column).expect("at most 256 incoming connections");
                    loci.push((
                        Locus::NodeWeight { node, conn },
                        LocusValue::Scalar(
                            decoded.parameters[parameter_start + row * input_width + column],
                        ),
                    ));
                    // As in the recurrent families, NodeTarget names the incoming source.
                    // These structural loci never consume mutation trials or RNG draws.
                    loci.push((
                        Locus::NodeTarget { node, conn },
                        LocusValue::Target(
                            u32::try_from(source_start + column)
                                .expect("bounded FtBrain source index"),
                        ),
                    ));
                }
            }
            parameter_start = weights_end + output_width;
            source_start = node_start;
            node_start += output_width;
        }
        Ok(loci)
    }

    fn genome_loci_upper_bound(&self) -> Option<usize> {
        std::iter::once(INPUT_SIZE)
            .chain(self.config.hidden_layers.iter().copied())
            .zip(
                self.config
                    .hidden_layers
                    .iter()
                    .copied()
                    .chain(std::iter::once(OUTPUT_SIZE)),
            )
            .try_fold(0_usize, |total, (inputs, outputs)| {
                // Each output owns one bias and a weight/source pair for each input.
                let per_output = inputs.checked_mul(2)?.checked_add(1)?;
                total.checked_add(per_output.checked_mul(outputs)?)
            })
    }

    fn validate_evaluator_state(
        &self,
        state: &BrainEvaluatorStateEnvelope,
    ) -> Result<(), BrainProtocolError> {
        self.decode_state(state).map(|_| ())
    }

    fn mutate_genome_material(
        &self,
        genome: &BrainGenomeEnvelope,
        rates: MutationRates,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        self.validate_mutation_rates(rates)?;
        let mut decoded = self.decode_genome(genome)?;
        for parameter in &mut decoded.parameters {
            if unit_f32(rng) < rates.primary {
                let gaussian = gaussian_f32(rng);
                if rates.secondary > 0.0 {
                    *parameter += gaussian * rates.secondary;
                }
            }
        }
        self.genome_material(&decoded.config, &decoded.parameters)
    }

    fn crossover_genomes_material(
        &self,
        left: &BrainGenomeEnvelope,
        right: &BrainGenomeEnvelope,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        let mut left = self.decode_genome(left)?;
        let right = self.decode_genome(right)?;
        if left.config != right.config || left.parameters.len() != right.parameters.len() {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                "FtBrain crossover requires identical architectures",
            ));
        }
        for (child, &right_gene) in left.parameters.iter_mut().zip(&right.parameters) {
            if unit_f32(rng) >= 0.5 {
                *child = right_gene;
            }
        }
        self.genome_material(&left.config, &left.parameters)
    }

    fn initial_state(
        &self,
        genome: &BrainGenomeEnvelope,
        _rng: &mut dyn RandomStream,
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.validate_genome(genome)?;
        self.state_for(genome)
    }

    fn offspring_state_policy(&self) -> OffspringStatePolicy {
        OffspringStatePolicy::Reset
    }

    fn offspring_state(
        &self,
        child: &BrainGenomeEnvelope,
        _parents: &[&BrainEvaluatorStateEnvelope],
        rng: &mut dyn RandomStream,
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.initial_state(child, rng)
    }

    fn evaluator(
        &self,
        genome: &BrainGenomeEnvelope,
        state: &BrainEvaluatorStateEnvelope,
    ) -> Result<Box<dyn BrainEvaluator>, BrainProtocolError> {
        let decoded = self.validate_pair(genome, state)?;
        Ok(Box::new(FtEvaluator {
            family: self.clone(),
            genome_digest: *genome.material_hash().as_bytes(),
            network: FtNetwork::materialize(&decoded),
        }))
    }

    fn batch_architecture_key(
        &self,
        genome: &BrainGenomeEnvelope,
    ) -> Result<Option<BrainBatchArchitectureKey>, BrainProtocolError> {
        self.validate_genome(genome)?;
        self.architecture_key().map(Some)
    }

    fn batch_evaluator(
        &self,
        genomes: &[&BrainGenomeEnvelope],
        states: &[&BrainEvaluatorStateEnvelope],
    ) -> Result<Option<Box<dyn BrainBatchEvaluator>>, BrainProtocolError> {
        if genomes.len() != states.len() {
            return Err(BrainProtocolError::BatchCardinalityMismatch {
                evaluators: genomes.len(),
                inputs: states.len(),
                outputs: states.len(),
            });
        }
        let lanes = genomes.len();
        let scratch_width = self
            .config
            .hidden_layers
            .iter()
            .copied()
            .chain([INPUT_SIZE, OUTPUT_SIZE])
            .max()
            .unwrap_or(INPUT_SIZE);
        let scratch_count = lanes
            .checked_mul(scratch_width)
            .ok_or_else(|| config_error("FtBrain cohort activation footprint overflow"))?;
        let mut layers = layer_shapes(&self.config)
            .into_iter()
            .map(|(input, output)| FtBatchLayer::new(self, lanes, input, output))
            .collect::<Result<Vec<_>, _>>()?;
        let mut genome_digests = Vec::with_capacity(genomes.len());
        for (lane, (&genome, &state)) in genomes.iter().zip(states).enumerate() {
            let decoded = self.validate_pair(genome, state)?;
            let mut offset = 0;
            for layer in &mut layers {
                let parameter_count = (layer.input_width + 1) * layer.output_width;
                layer.pack_lane(lane, &decoded.parameters[offset..offset + parameter_count]);
                offset += parameter_count;
            }
            debug_assert_eq!(offset, decoded.parameters.len());
            genome_digests.push(*genome.material_hash().as_bytes());
        }
        Ok(Some(Box::new(FtBatchEvaluator {
            family: self.clone(),
            genome_digests,
            layers,
            activations: cohort_buffer(self, scratch_count)?,
            candidates: cohort_buffer(self, scratch_count)?,
        })))
    }
}

fn canonical_family_id() -> BrainFamilyId {
    BrainFamilyId::new(FAMILY_ID).expect("the built-in FtBrain family ID is canonical")
}

fn config_error(detail: impl Into<String>) -> BrainProtocolError {
    BrainProtocolError::InvalidPayload {
        kind: BrainEnvelopeKind::Genome,
        family_id: canonical_family_id(),
        detail: detail.into(),
    }
}

fn validate_config(config: &FtBrainConfig) -> Result<usize, BrainProtocolError> {
    if config.hidden_layers.is_empty() || config.hidden_layers.len() > MAX_HIDDEN_LAYERS {
        return Err(config_error(format!(
            "FtBrain requires 1..={MAX_HIDDEN_LAYERS} hidden layers"
        )));
    }
    if let Some((index, width)) = config
        .hidden_layers
        .iter()
        .copied()
        .enumerate()
        .find(|(_, width)| *width == 0 || *width > MAX_LAYER_WIDTH)
    {
        return Err(config_error(format!(
            "hidden layer {index} width {width} is outside 1..={MAX_LAYER_WIDTH}"
        )));
    }
    let mut parameter_count = 0usize;
    for (input_width, output_width) in layer_shapes(config) {
        parameter_count = parameter_count
            .checked_add(
                input_width
                    .checked_mul(output_width)
                    .and_then(|weights| weights.checked_add(output_width))
                    .ok_or_else(|| config_error("FtBrain parameter footprint overflow"))?,
            )
            .ok_or_else(|| config_error("FtBrain parameter footprint overflow"))?;
    }
    let payload_bytes = GENOME_FIXED_HEADER_BYTES
        .checked_add(config.hidden_layers.len() * 2)
        .and_then(|header| header.checked_add(parameter_count.checked_mul(4)?))
        .ok_or_else(|| config_error("FtBrain genome payload size overflow"))?;
    if payload_bytes > MAX_BRAIN_GENOME_PAYLOAD_BYTES {
        return Err(config_error(format!(
            "FtBrain genome needs {payload_bytes} bytes, exceeding the {MAX_BRAIN_GENOME_PAYLOAD_BYTES}-byte protocol bound"
        )));
    }
    Ok(parameter_count)
}

fn layer_shapes(config: &FtBrainConfig) -> Vec<(usize, usize)> {
    let mut widths = Vec::with_capacity(config.hidden_layers.len() + 2);
    widths.push(INPUT_SIZE);
    widths.extend_from_slice(&config.hidden_layers);
    widths.push(OUTPUT_SIZE);
    widths.windows(2).map(|pair| (pair[0], pair[1])).collect()
}

#[allow(clippy::cast_precision_loss)]
fn unit_f32(rng: &mut dyn RandomStream) -> f32 {
    const SCALE: f32 = 1.0 / 16_777_216.0;
    ((rng.next_u32() >> 8) as f32) * SCALE
}

fn signed_unit_f32(rng: &mut dyn RandomStream) -> f32 {
    unit_f32(rng) * 2.0 - 1.0
}

fn gaussian_f32(rng: &mut dyn RandomStream) -> f32 {
    let u1 = unit_f32(rng).clamp(f32::MIN_POSITIVE, 1.0);
    let u2 = unit_f32(rng);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn state_from_digest(
    family: &FtBrainFamily,
    digest: [u8; 32],
) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
    let mut payload = Vec::with_capacity(STATE_PAYLOAD_BYTES);
    payload.extend_from_slice(&STATE_MAGIC);
    payload.extend_from_slice(&digest);
    BrainEvaluatorStateEnvelope::new(
        family.family_id.clone(),
        STATE_SCHEMA_VERSION,
        STATE_CODEC_VERSION,
        payload,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;
    use scriptbots_core::{
        AgentUid, BrainFamilyAdapter, BrainGenomeDerivation, BrainProvenance,
        RANDOM_STREAM_STATE_VERSION, RandomStreamState, SmallRngStream, Tick,
    };

    #[derive(Debug)]
    struct ScriptedStream {
        draws: Vec<u32>,
        cursor: usize,
    }

    impl ScriptedStream {
        fn selected_positive_gaussian(parameter_count: usize) -> Self {
            let mut draws = Vec::with_capacity(parameter_count * 3);
            for _ in 0..parameter_count {
                draws.extend_from_slice(&[0, 1_u32 << 31, 0]);
            }
            Self { draws, cursor: 0 }
        }

        const fn consumed(&self) -> usize {
            self.cursor
        }

        const fn remaining(&self) -> usize {
            self.draws.len() - self.cursor
        }
    }

    impl RngCore for ScriptedStream {
        fn next_u32(&mut self) -> u32 {
            let draw = *self
                .draws
                .get(self.cursor)
                .expect("FtBrain consumed more next_u32 draws than scripted");
            self.cursor += 1;
            draw
        }

        fn next_u64(&mut self) -> u64 {
            panic!("FtBrain mutation must preserve its next_u32 draw schedule")
        }

        fn fill_bytes(&mut self, _destination: &mut [u8]) {
            panic!("FtBrain mutation must preserve its next_u32 draw schedule")
        }
    }

    impl RandomStream for ScriptedStream {
        fn algorithm_id(&self) -> &'static str {
            "test.ft-scripted-next-u32"
        }

        fn checkpoint(&self) -> RandomStreamState {
            RandomStreamState {
                version: RANDOM_STREAM_STATE_VERSION,
                algorithm: self.algorithm_id().to_owned(),
                codec_version: 1,
                state: u64::try_from(self.cursor)
                    .expect("script cursor fits in u64")
                    .to_le_bytes()
                    .to_vec(),
            }
        }
    }

    fn founder(
        family: &FtBrainFamily,
        seed: u64,
    ) -> (BrainGenomeEnvelope, BrainEvaluatorStateEnvelope) {
        let mut rng = SmallRngStream::seed_from_u64(seed);
        let genome = family
            .random_genome(BrainProvenance::default(), &mut rng)
            .expect("founder genome");
        let state = family
            .initial_state(&genome, &mut rng)
            .expect("initial state");
        (genome, state)
    }

    fn genome_with_parameters(family: &FtBrainFamily, parameters: &[f32]) -> BrainGenomeEnvelope {
        BrainGenomeEnvelope::new(
            family.family_id.clone(),
            GENOME_SCHEMA_VERSION,
            GENOME_CODEC_VERSION,
            family
                .encode_genome_payload(&family.config, parameters)
                .expect("encode parameters"),
            BrainProvenance::default(),
        )
        .expect("genome envelope")
    }

    #[test]
    fn ft_loci_are_architecture_bound_and_match_every_mutated_parameter() {
        use scriptbots_core::genome_diff::{Locus, LocusValue};
        use std::collections::BTreeMap;
        let family = FtBrainFamily::new(FtBrainConfig::new(vec![2]).unwrap()).unwrap();
        let count = validate_config(&family.config).unwrap();
        let mut parameters = vec![0.25; count];
        parameters[1] = -0.0;
        parameters[50] = -0.5;
        parameters[78] = 0.75;
        let genome = genome_with_parameters(&family, &parameters);
        let loci = family.genome_loci(&genome).unwrap();
        assert_eq!(family.genome_loci_upper_bound(), Some(loci.len()));
        let by_locus: BTreeMap<_, _> = loci.iter().copied().collect();
        assert_eq!(
            by_locus.len(),
            loci.len(),
            "canonical addresses cannot alias"
        );
        assert_eq!(loci.len(), 2 * (50 + 18) + 11);
        assert_eq!(loci[0], (Locus::NodeBias(25), LocusValue::Scalar(-0.5)));
        let LocusValue::Scalar(signed_zero) = by_locus[&Locus::NodeWeight { node: 25, conn: 1 }]
        else {
            panic!("a dense weight must remain a scalar locus");
        };
        assert_eq!(signed_zero.to_bits(), (-0.0_f32).to_bits());
        assert_eq!(
            by_locus[&Locus::NodeTarget { node: 25, conn: 24 }],
            LocusValue::Target(24)
        );
        assert_eq!(
            by_locus[&Locus::NodeTarget { node: 35, conn: 1 }],
            LocusValue::Target(26)
        );
        assert_eq!(by_locus[&Locus::NodeBias(35)], LocusValue::Scalar(0.75));
        assert_eq!(family.genome_loci(&genome).unwrap(), loci);

        let BrainHeredityCapabilityV1::LocusCapable {
            locus_schema,
            mutation_trials,
        } = family.heredity_capability()
        else {
            panic!("Ft must participate in typed heredity discovery");
        };
        assert_eq!(locus_schema.family_id(), family.family_id());
        assert_eq!(
            mutation_trials,
            vec![BrainMutationTrialGroupV1::new(79, 1, 1)]
        );
        let other = FtBrainFamily::new(FtBrainConfig::new(vec![3]).unwrap()).unwrap();
        let other_count = validate_config(&other.config).unwrap();
        let other_genome = genome_with_parameters(&other, &vec![0.0; other_count]);
        assert_eq!(
            other.genome_loci_upper_bound(),
            Some(other.genome_loci(&other_genome).unwrap().len())
        );
        assert_ne!(
            family.genome_loci_upper_bound(),
            other.genome_loci_upper_bound()
        );
        let BrainHeredityCapabilityV1::LocusCapable {
            locus_schema: other_schema,
            ..
        } = other.heredity_capability()
        else {
            panic!("other architecture is locus-capable");
        };
        assert_ne!(
            locus_schema, other_schema,
            "different layouts cannot reuse a locus schema"
        );
        assert!(
            other.genome_loci(&genome).is_err(),
            "loci must use the validated architecture"
        );

        let mut rng = ScriptedStream::selected_positive_gaussian(count);
        let child = family
            .mutate_genome(
                &genome,
                MutationRates {
                    primary: 1.0,
                    secondary: 0.125,
                },
                BrainProvenance {
                    parents: [Some(AgentUid(1)), None],
                    parent_genome_hashes: [Some(genome.material_hash()), None],
                    created_at: Tick(1),
                    derivation: BrainGenomeDerivation::MutationOnly,
                },
                &mut rng,
            )
            .unwrap();
        let child_loci = family.genome_loci(&child).unwrap();
        let mut changed_scalars = 0;
        for ((address, old), (next_address, new)) in loci.iter().zip(&child_loci) {
            assert_eq!(address, next_address);
            match (old, new) {
                (LocusValue::Scalar(old), LocusValue::Scalar(new)) => {
                    assert_ne!(old.to_bits(), new.to_bits());
                    changed_scalars += 1;
                }
                (LocusValue::Target(old), LocusValue::Target(new)) => assert_eq!(old, new),
                _ => panic!("dense loci must preserve their scalar or target type"),
            }
        }
        assert_eq!(changed_scalars, count);
        assert_eq!(rng.consumed(), count * 3);
        assert_eq!(rng.remaining(), 0);

        let wide = FtBrainFamily::new(FtBrainConfig::new(vec![256]).unwrap()).unwrap();
        let wide_genome =
            genome_with_parameters(&wide, &vec![0.0; validate_config(&wide.config).unwrap()]);
        let wide_loci: BTreeMap<_, _> = wide
            .genome_loci(&wide_genome)
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(
            wide_loci[&Locus::NodeTarget {
                node: 281,
                conn: 255
            }],
            LocusValue::Target(280)
        );
    }

    #[test]
    fn ft_inspection_reconstructs_global_activity_and_preserves_continuation() {
        let family = FtBrainFamily::new(FtBrainConfig::new(vec![2]).unwrap()).unwrap();
        let mut parameters = vec![0.0; 79];
        parameters[0] = 1.0;
        parameters[24] = -0.5;
        parameters[26] = 0.5;
        parameters[50] = 0.25;
        parameters[51] = -0.25;
        parameters[52] = 2.0;
        parameters[53] = -1.0;
        parameters[70] = 0.125;
        let genome = genome_with_parameters(&family, &parameters);
        let state = family.state_for(&genome).unwrap();
        let mut evaluator = family.evaluator(&genome, &state).unwrap();
        let mut sensors = [0.0; INPUT_SIZE];
        sensors[0] = 0.25;
        sensors[1] = 0.5;
        sensors[24] = 0.5;
        let outputs = evaluator.evaluate(&sensors).unwrap();
        let before = evaluator.checkpoint_state().unwrap();
        let inspection = evaluator
            .inspect_with_context(
                BrainInspection::Activations(BrainInspectionLimits::hard()),
                BrainInspectionContext {
                    source_tick: Tick(1),
                    sensors: &sensors,
                    outputs: &outputs,
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            inspection.activity_source,
            BrainActivitySource::ReconstructedFeedForward
        );
        assert!(!inspection.build.truncated);
        assert_eq!(inspection.build.source_scalars, 79 + 36);
        assert_eq!(inspection.build.source_values, 72);
        assert_eq!(inspection.build.source_edges, 68);
        assert_eq!(inspection.activations.layers.len(), 4);
        assert!(
            inspection
                .activations
                .layers
                .iter()
                .all(|layer| layer.name.starts_with("Reconstructed "))
        );
        assert_eq!(inspection.activations.layers[0].values, sensors);
        assert_eq!(
            inspection.activations.layers[1].values[0].to_bits(),
            0.25_f32.tanh().to_bits()
        );
        assert_eq!(
            inspection.activations.layers[1].values[1].to_bits(),
            0.0_f32.to_bits()
        );
        let flat = &inspection.activations.layers.last().unwrap().values;
        assert_eq!(&flat[..INPUT_SIZE], &sensors);
        assert_eq!(
            inspection.activations.output_slots,
            (27..36).collect::<Vec<_>>()
        );
        for channel in OutputChannel::ALL {
            assert_eq!(
                flat[27 + channel.index()].to_bits(),
                outputs[channel.index()].to_bits()
            );
        }
        assert_eq!(
            inspection.activations.connections[0],
            ActivationEdge {
                from: 0,
                to: 25,
                weight: 1.0
            }
        );
        assert_eq!(
            inspection.activations.connections[50],
            ActivationEdge {
                from: 25,
                to: 27,
                weight: 2.0
            }
        );
        let explanation =
            scriptbots_core::attribution::attribute_output(&inspection.activations, 0, 3).unwrap();
        assert!(matches!(
            explanation.method,
            scriptbots_core::attribution::AttributionMethod::PathProduct { .. }
        ));
        assert_eq!(explanation.raw_value.to_bits(), outputs[0].to_bits());
        assert_eq!(evaluator.checkpoint_state().unwrap(), before);
        let mut control = family.evaluator(&genome, &before).unwrap();
        let next_sensors = [0.75; INPUT_SIZE];
        assert_output_bits(
            &[evaluator.evaluate(&next_sensors).unwrap()],
            &[control.evaluate(&next_sensors).unwrap()],
        );
        assert_eq!(
            evaluator.checkpoint_state().unwrap(),
            control.checkpoint_state().unwrap()
        );

        // Core rehydrates scalar evaluators after committing a batch; they must not need a
        // private last-input cache to inspect the owner's recorded batch response.
        let mut cohort = family
            .batch_evaluator(&[&genome], &[&state])
            .unwrap()
            .unwrap();
        let mut cohort_outputs = [[0.0; OUTPUT_SIZE]];
        cohort
            .evaluate_batch(&[sensors], &mut cohort_outputs)
            .unwrap();
        let restored = family
            .evaluator(&genome, &cohort.checkpoint_states().unwrap()[0])
            .unwrap();
        let reconstructed = restored
            .inspect_with_context(
                BrainInspection::Activations(BrainInspectionLimits::hard()),
                BrainInspectionContext {
                    source_tick: Tick(1),
                    sensors: &sensors,
                    outputs: &cohort_outputs[0],
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(reconstructed, inspection);
    }

    #[test]
    fn ft_inspection_refuses_unrecorded_mismatched_and_over_budget_activity() {
        let family = FtBrainFamily::new(FtBrainConfig::new(vec![1]).unwrap()).unwrap();
        let (genome, state) = founder(&family, 9_100);
        let mut evaluator = family.evaluator(&genome, &state).unwrap();
        let sensors = [0.25; INPUT_SIZE];
        let outputs = evaluator.evaluate(&sensors).unwrap();
        let request = BrainInspection::Activations(BrainInspectionLimits::hard());
        assert!(matches!(
            evaluator.inspect(request),
            Err(BrainInspectionError::ActivityUnavailable {
                reason: BrainInspectionUnavailable::NoRecordedEvaluation,
            })
        ));
        for (source_tick, recorded_sensors, recorded_outputs) in [
            (Tick::zero(), sensors, outputs),
            (Tick(10), [0.0; INPUT_SIZE], [0.0; OUTPUT_SIZE]),
        ] {
            assert!(matches!(
                evaluator.inspect_with_context(
                    request,
                    BrainInspectionContext {
                        source_tick,
                        sensors: &recorded_sensors,
                        outputs: &recorded_outputs,
                    }
                ),
                Err(BrainInspectionError::ActivityUnavailable {
                    reason: BrainInspectionUnavailable::NoRecordedEvaluation
                })
            ));
        }
        let mut altered = outputs;
        altered[8] = f32::from_bits(altered[8].to_bits() ^ 1);
        assert!(matches!(
            evaluator.inspect_with_context(
                request,
                BrainInspectionContext {
                    source_tick: Tick(1),
                    sensors: &sensors,
                    outputs: &altered,
                }
            ),
            Err(BrainInspectionError::ActivityUnavailable {
                reason: BrainInspectionUnavailable::RecordedResponseMismatch
            })
        ));
        assert_eq!(evaluator.checkpoint_state().unwrap(), state);

        let large = FtBrainFamily::new(FtBrainConfig::new(vec![256, 256]).unwrap()).unwrap();
        let large_genome =
            genome_with_parameters(&large, &vec![0.0; validate_config(&large.config).unwrap()]);
        let large_evaluator = large
            .evaluator(&large_genome, &large.state_for(&large_genome).unwrap())
            .unwrap();
        assert!(
            matches!(large_evaluator.inspect_with_context(request, BrainInspectionContext {
            source_tick: Tick(1), sensors: &sensors, outputs: &[0.5; OUTPUT_SIZE],
        }), Err(BrainInspectionError::SourceScalarLimitExceeded { required, limit, .. }) if required > limit)
        );
        let tight = BrainInspectionLimits::tightened(2, 32, 4, 1, 256, 1_024);
        let clipped = evaluator
            .inspect_with_context(
                BrainInspection::Activations(tight),
                BrainInspectionContext {
                    source_tick: Tick(1),
                    sensors: &sensors,
                    outputs: &outputs,
                },
            )
            .unwrap()
            .unwrap();
        assert!(clipped.build.truncated);
        assert!(clipped.build.retained_payload_bytes <= tight.max_payload_bytes());
        assert!(clipped.activations.layers.is_empty());
        assert!(clipped.activations.connections.is_empty());
    }

    #[test]
    fn ft_world_inspection_keeps_clipped_weights_and_is_checkpoint_neutral() {
        use scriptbots_core::{
            AgentData, BrainInspectionClientId, BrainInspectionRequest, BrainInspectionRevision,
            BrainRegistry, Position, ScriptBotsConfig, SelectedBrainTelemetryOutcome, WorldState,
        };
        let family = FtBrainFamily::default();
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            closed: true,
            persistence_interval: 0,
            rng_seed: Some(9_200),
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        let key = world
            .register_brain_family(FT_BRAIN_KIND, Box::new(family.clone()))
            .unwrap();
        let agent = world
            .try_spawn_agent(AgentData {
                position: Position::new(60.0, 60.0),
                ..AgentData::default()
            })
            .unwrap();
        world.bind_agent_brain(agent, key).unwrap();
        let uid = world.agent_uid(agent).unwrap();
        let mut request = BrainInspectionRequest::single(
            BrainInspectionClientId::new(3),
            BrainInspectionRevision::new(1),
            uid,
        );
        assert!(matches!(
            world.inspect_brains(&request).unwrap().telemetry.as_slice(),
            [SelectedBrainTelemetryOutcome::Unavailable {
                reason: BrainInspectionUnavailable::NoRecordedEvaluation,
                ..
            }]
        ));
        assert_eq!(
            world
                .brain_registry()
                .heredity_capabilities()
                .unwrap()
                .descriptors[0]
                .capability,
            family.heredity_capability()
        );
        world.step().unwrap();
        let checkpoint = world.checkpoint_v1().unwrap();
        let before_bytes = checkpoint.encode().unwrap();
        let before_digest = world.world_digest_v1().unwrap();
        let response = world.inspect_brains(&request).unwrap();
        let ready = response.ready_for(uid).unwrap();
        assert_eq!(
            ready.inspection.activity_source,
            BrainActivitySource::ReconstructedFeedForward
        );
        assert_eq!(
            ready.inspection.activations.connections.len(),
            request.limits.max_edges()
        );
        assert!(ready.inspection.build.source_edges > ready.inspection.build.retained_edges);
        assert!(ready.inspection.activations.truncated);
        assert_eq!(world.world_digest_v1().unwrap(), before_digest);
        assert_eq!(
            world.checkpoint_v1().unwrap().encode().unwrap(),
            before_bytes
        );
        let mut registry = BrainRegistry::default();
        registry
            .register_family(FT_BRAIN_KIND, Box::new(family.clone()))
            .unwrap();
        let mut control = WorldState::restore_checkpoint_v1(&checkpoint, registry).unwrap();
        world.step().unwrap();
        control.step().unwrap();
        assert_eq!(
            world.world_digest_v1().unwrap(),
            control.world_digest_v1().unwrap()
        );

        let fresh = world
            .try_inject_agent(AgentData {
                position: Position::new(140.0, 140.0),
                ..AgentData::default()
            })
            .unwrap();
        world.bind_agent_brain(fresh, key).unwrap();
        let fresh_uid = world.agent_uid(fresh).unwrap();
        request.targets.push(fresh_uid);
        let mixed = world.inspect_brains(&request).unwrap();
        assert!(mixed.ready_for(uid).is_some());
        assert!(matches!(
            mixed.telemetry[1],
            SelectedBrainTelemetryOutcome::Unavailable {
                reason: BrainInspectionUnavailable::NoRecordedEvaluation,
                ..
            }
        ));
        let different_genome = genome_with_parameters(
            &family,
            &vec![0.0; validate_config(&family.config).unwrap()],
        );
        world
            .bind_agent_brain_genome(agent, &different_genome)
            .unwrap();
        let mismatched = world.inspect_brains(&request).unwrap();
        assert!(matches!(
            mismatched.telemetry[0],
            SelectedBrainTelemetryOutcome::Unavailable {
                reason: BrainInspectionUnavailable::RecordedResponseMismatch,
                ..
            }
        ));
        assert!(world.agent_runtime(agent).unwrap().energy.is_finite());
    }

    #[test]
    fn default_topology_has_the_expected_flat_f32_footprint() {
        let family = FtBrainFamily::default();
        assert_eq!(family.config.hidden_layers(), &[48, 32, 24]);
        assert_eq!(
            validate_config(&family.config).expect("valid config"),
            3_833
        );

        let (genome, _) = founder(&family, 17);
        let decoded = family.decode_genome(&genome).expect("decode genome");
        assert_eq!(decoded.parameters.len(), 3_833);
        assert!(decoded.parameters.iter().all(|value| value.is_finite()));
        assert_eq!(
            family
                .encode_genome_payload(&decoded.config, &decoded.parameters)
                .expect("re-encode"),
            genome.payload(),
            "the bounded codec must round-trip byte-for-byte"
        );
    }

    #[test]
    fn frankentorch_f32_flattening_oracle_preserves_parameter_order() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let first = session
            .tensor_variable_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .expect("first F32 parameter");
        let second = session
            .tensor_variable_f32(vec![5.0, 6.0], vec![2], false)
            .expect("second F32 parameter");
        let flat = ft_nn::parameters_to_vector(&mut session, &[first, second])
            .expect("flatten F32 parameters");
        assert_eq!(
            session.tensor_values_f32(flat).expect("flat F32 values"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn adapter_identity_changes_with_architecture() {
        let baseline = FtBrainFamily::default().adapter_identity();
        assert_eq!(baseline, FtBrainFamily::default().adapter_identity());
        let narrow = FtBrainFamily::new(FtBrainConfig::new(vec![8]).expect("config"))
            .expect("family")
            .adapter_identity();
        assert_ne!(baseline, narrow);
        assert_eq!(baseline.semantic_version(), ADAPTER_SEMANTIC_VERSION);
    }

    #[test]
    fn known_single_hidden_unit_matches_an_independent_reference() {
        let family =
            FtBrainFamily::new(FtBrainConfig::new(vec![1]).expect("config")).expect("family");
        let mut parameters = vec![0.0; validate_config(&family.config).expect("count")];
        parameters[0] = 2.0;
        parameters[INPUT_SIZE] = 0.1;
        let output_weights_start = INPUT_SIZE + 1;
        let output_bias_start = output_weights_start + OUTPUT_SIZE;
        for channel in OutputChannel::ALL {
            parameters[output_weights_start + channel.index()] =
                f32::from(u8::try_from(channel.index() + 1).expect("channel")) * 0.1;
            parameters[output_bias_start + channel.index()] = -0.05;
        }

        let genome = genome_with_parameters(&family, &parameters);
        let mut rng = SmallRngStream::seed_from_u64(9);
        let state = family.initial_state(&genome, &mut rng).expect("state");
        let mut evaluator = family.evaluator(&genome, &state).expect("evaluator");
        let mut sensors = [0.0; INPUT_SIZE];
        sensors[0] = 0.25;
        let actual = evaluator.evaluate(&sensors).expect("forward");
        let hidden = (sensors[0] * 2.0 + 0.1).tanh();
        for channel in OutputChannel::ALL {
            let weight = f32::from(u8::try_from(channel.index() + 1).expect("channel")) * 0.1;
            assert_eq!(actual[channel.index()], sigmoid(hidden * weight - 0.05));
        }
    }

    #[test]
    fn evaluator_state_is_bound_to_exact_genome_material() {
        let family = FtBrainFamily::default();
        let (left, left_state) = founder(&family, 3);
        let (right, _) = founder(&family, 4);
        family
            .evaluator(&left, &left_state)
            .expect("matching state must reconstruct");
        let error = family
            .evaluator(&right, &left_state)
            .err()
            .expect("state from another genome must fail");
        assert!(error.to_string().contains("different genome"));
    }

    #[test]
    fn evaluator_state_accepts_same_material_with_different_provenance() {
        let family = FtBrainFamily::default();
        let (original, state) = founder(&family, 7);
        let rewrapped = BrainGenomeEnvelope::new(
            family.family_id.clone(),
            GENOME_SCHEMA_VERSION,
            GENOME_CODEC_VERSION,
            original.payload().to_vec(),
            BrainProvenance {
                created_at: Tick(99),
                ..BrainProvenance::default()
            },
        )
        .expect("same material with later founder provenance");
        assert_eq!(original.material_hash(), rewrapped.material_hash());
        family
            .evaluator(&rewrapped, &state)
            .expect("state binding is material-based, not provenance-based");
    }

    #[test]
    fn batch_and_scalar_paths_are_bit_identical() {
        let family = FtBrainFamily::default();
        let (left, left_state) = founder(&family, 31);
        let (right, right_state) = founder(&family, 32);
        let mut inputs = [[0.0; INPUT_SIZE]; 2];
        for (index, value) in inputs[0].iter_mut().enumerate() {
            *value = f32::from(u8::try_from(index).expect("sensor index")) / 25.0;
        }
        for (index, value) in inputs[1].iter_mut().enumerate() {
            *value = f32::from(u8::try_from(INPUT_SIZE - index).expect("sensor index")) / 25.0;
        }

        let mut left_scalar = family.evaluator(&left, &left_state).expect("left scalar");
        let mut right_scalar = family
            .evaluator(&right, &right_state)
            .expect("right scalar");
        let expected = [
            left_scalar.evaluate(&inputs[0]).expect("left output"),
            right_scalar.evaluate(&inputs[1]).expect("right output"),
        ];

        let mut batch = family
            .batch_evaluator(&[&left, &right], &[&left_state, &right_state])
            .expect("batch construction")
            .expect("FtBrain opts into batching");
        let mut actual = [[0.0; OUTPUT_SIZE]; 2];
        batch
            .evaluate_batch(&inputs, &mut actual)
            .expect("batch evaluation");
        assert_eq!(actual, expected);
        assert_eq!(
            batch.checkpoint_states().expect("batch checkpoints"),
            vec![left_state, right_state]
        );
    }

    fn cohort_inputs(lanes: usize, phase: usize) -> Vec<[f32; INPUT_SIZE]> {
        (0..lanes)
            .map(|lane| {
                std::array::from_fn(|sensor| {
                    let value = (lane * 11 + sensor * 7 + phase) % 37;
                    f32::from(u8::try_from(value).expect("bounded fixture value")) / 18.0 - 1.0
                })
            })
            .collect()
    }

    fn scalar_cohort_outputs(
        family: &FtBrainFamily,
        founders: &[(BrainGenomeEnvelope, BrainEvaluatorStateEnvelope)],
        inputs: &[[f32; INPUT_SIZE]],
    ) -> Vec<[f32; OUTPUT_SIZE]> {
        founders
            .iter()
            .zip(inputs)
            .map(|((genome, state), input)| {
                family
                    .evaluator(genome, state)
                    .expect("scalar evaluator")
                    .evaluate(input)
                    .expect("scalar output")
            })
            .collect()
    }

    fn assert_output_bits(actual: &[[f32; OUTPUT_SIZE]], expected: &[[f32; OUTPUT_SIZE]]) {
        assert_eq!(actual.len(), expected.len());
        for (lane, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            for channel in OutputChannel::ALL {
                assert_eq!(
                    actual[channel.index()].to_bits(),
                    expected[channel.index()].to_bits(),
                    "first differing cohort lane {lane}, channel {channel:?}"
                );
            }
        }
    }

    #[test]
    fn packed_cohort_matches_scalar_across_shapes_and_parallel_dispatch_boundary() {
        // 7/8 straddles the pinned BMM kernel's serial/parallel lane dispatch.
        // Asymmetric widths expose transpose/stride mistakes that square layers hide.
        for hidden in [vec![1], vec![7, 13], vec![48, 32, 24], vec![31, 8, 17, 2]] {
            let family =
                FtBrainFamily::new(FtBrainConfig::new(hidden).expect("config")).expect("family");
            for lanes in [0, 1, 7, 8, 17] {
                let founders: Vec<_> = (0..lanes)
                    .map(|lane| founder(&family, 1_001 + u64::try_from(lane).expect("lane")))
                    .collect();
                let genomes: Vec<_> = founders.iter().map(|(genome, _)| genome).collect();
                let states: Vec<_> = founders.iter().map(|(_, state)| state).collect();
                let mut cohort = family
                    .batch_evaluator(&genomes, &states)
                    .expect("construct packed cohort")
                    .expect("cohort supported");
                let initial_states = cohort.checkpoint_states().expect("initial states");
                let mut actual = vec![[f32::NAN; OUTPUT_SIZE]; lanes];

                // A second, different forward catches stale activation scratch.
                for phase in [0, 19] {
                    let inputs = cohort_inputs(lanes, phase);
                    let expected = scalar_cohort_outputs(&family, &founders, &inputs);
                    cohort
                        .evaluate_batch(&inputs, &mut actual)
                        .expect("packed evaluation");
                    assert_output_bits(&actual, &expected);
                }
                assert_eq!(
                    cohort.checkpoint_states().expect("final states"),
                    initial_states,
                    "execution scratch must never enter stateless evaluator checkpoints"
                );
            }
        }
    }

    #[test]
    fn packed_cohort_preserves_lane_identity_under_permutation_and_partition() {
        let family = FtBrainFamily::default();
        let founders: Vec<_> = (0..17).map(|lane| founder(&family, 2_000 + lane)).collect();
        let inputs = cohort_inputs(founders.len(), 5);
        let expected = scalar_cohort_outputs(&family, &founders, &inputs);
        assert!(
            expected.windows(2).any(|pair| pair[0] != pair[1]),
            "a shared-weights implementation must not pass on identical fixture lanes"
        );
        let permutation: Vec<_> = (0..founders.len())
            .map(|lane| (lane * 7) % founders.len())
            .collect();
        let permuted_inputs: Vec<_> = permutation.iter().map(|&lane| inputs[lane]).collect();
        let permuted_expected: Vec<_> = permutation.iter().map(|&lane| expected[lane]).collect();

        // Same biological lanes, different batching. These partitions include
        // partial tails and cross the BMM backend's parallelism threshold.
        for partition in [1, 3, 7, 8, 17] {
            let mut actual = Vec::new();
            for (lane_order, inputs) in permutation
                .chunks(partition)
                .zip(permuted_inputs.chunks(partition))
            {
                let genomes: Vec<_> = lane_order.iter().map(|&lane| &founders[lane].0).collect();
                let states: Vec<_> = lane_order.iter().map(|&lane| &founders[lane].1).collect();
                let mut cohort = family
                    .batch_evaluator(&genomes, &states)
                    .expect("construct partition")
                    .expect("cohort supported");
                let mut output = vec![[f32::NAN; OUTPUT_SIZE]; inputs.len()];
                cohort
                    .evaluate_batch(inputs, &mut output)
                    .expect("evaluate partition");
                assert_eq!(
                    cohort.checkpoint_states().expect("partition states"),
                    states.into_iter().cloned().collect::<Vec<_>>()
                );
                actual.extend(output);
            }
            assert_output_bits(&actual, &permuted_expected);
        }
    }

    #[test]
    fn packed_cohort_rejects_invalid_lanes_without_publishing_or_poisoning_retry() {
        let family = FtBrainFamily::default();
        let founders: Vec<_> = (0..9).map(|lane| founder(&family, 3_000 + lane)).collect();
        let genomes: Vec<_> = founders.iter().map(|(genome, _)| genome).collect();
        let states: Vec<_> = founders.iter().map(|(_, state)| state).collect();
        let mut cohort = family
            .batch_evaluator(&genomes, &states)
            .expect("construct cohort")
            .expect("cohort supported");
        let inputs = cohort_inputs(founders.len(), 2);
        let expected = scalar_cohort_outputs(&family, &founders, &inputs);
        let initial_states = cohort.checkpoint_states().expect("initial states");
        let sentinel = vec![[0.125; OUTPUT_SIZE]; founders.len()];

        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut malformed = inputs.clone();
            malformed[8][INPUT_SIZE - 1] = invalid;
            let mut outputs = sentinel.clone();
            let error = cohort
                .evaluate_batch(&malformed, &mut outputs)
                .expect_err("a late malformed lane must reject the entire call");
            assert!(error.to_string().contains("lane 8"));
            assert_eq!(outputs, sentinel);
            assert_eq!(
                cohort.checkpoint_states().expect("error states"),
                initial_states
            );
            cohort
                .evaluate_batch(&inputs, &mut outputs)
                .expect("clean retry after rejection");
            assert_output_bits(&outputs, &expected);
        }

        let mut outputs = sentinel.clone();
        assert!(matches!(
            cohort.evaluate_batch(&inputs[..8], &mut outputs),
            Err(BrainProtocolError::BatchCardinalityMismatch {
                evaluators: 9,
                inputs: 8,
                outputs: 9,
            })
        ));
        assert_eq!(outputs, sentinel);
        let mut short_outputs = sentinel[..8].to_vec();
        assert!(matches!(
            cohort.evaluate_batch(&inputs, &mut short_outputs),
            Err(BrainProtocolError::BatchCardinalityMismatch {
                evaluators: 9,
                inputs: 9,
                outputs: 8,
            })
        ));
        assert_eq!(short_outputs, sentinel[..8]);
    }

    #[test]
    fn packed_cohort_rejects_mismatched_genome_state_and_architecture() {
        let family = FtBrainFamily::default();
        let (left, left_state) = founder(&family, 4_000);
        let (right, right_state) = founder(&family, 4_001);
        assert!(
            family
                .batch_evaluator(&[&left, &right], &[&left_state])
                .is_err()
        );
        let error = family
            .batch_evaluator(&[&left, &right], &[&left_state, &left_state])
            .err()
            .expect("lane state must remain bound to its own genome");
        assert!(error.to_string().contains("different genome"));

        let narrow = FtBrainFamily::new(FtBrainConfig::new(vec![5]).expect("config"))
            .expect("narrow family");
        let (other, other_state) = founder(&narrow, 4_002);
        assert!(
            family
                .batch_evaluator(&[&left, &other], &[&left_state, &other_state])
                .is_err()
        );
        assert!(
            family
                .batch_evaluator(&[&left, &right], &[&left_state, &right_state])
                .expect("valid ordered cohort remains accepted")
                .is_some()
        );
    }

    #[test]
    fn packed_cohort_preserves_scalar_extreme_finite_inputs_and_reuses_scratch() {
        let family =
            FtBrainFamily::new(FtBrainConfig::new(vec![1]).expect("config")).expect("family");
        let (healthy, healthy_state) = founder(&family, 4_100);
        let mut parameters = vec![0.0; validate_config(&family.config).expect("count")];
        parameters[0] = f32::MAX;
        parameters[1] = -f32::MAX;
        for channel in OutputChannel::ALL {
            let index = u8::try_from(channel.index()).expect("bounded output channel");
            parameters[INPUT_SIZE + 1 + channel.index()] = (f32::from(index) - 4.0) * 0.125;
        }
        let explosive = genome_with_parameters(&family, &parameters);
        let mut rng = SmallRngStream::seed_from_u64(4_101);
        let explosive_state = family.initial_state(&explosive, &mut rng).expect("state");
        let founders = [(healthy, healthy_state), (explosive, explosive_state)];
        let genomes: Vec<_> = founders.iter().map(|(genome, _)| genome).collect();
        let states: Vec<_> = founders.iter().map(|(_, state)| state).collect();
        let mut cohort = family
            .batch_evaluator(&genomes, &states)
            .expect("finite weights are valid genomes")
            .expect("cohort supported");
        let initial_states = cohort.checkpoint_states().expect("initial states");
        let sentinel = [[0.125; OUTPUT_SIZE]; 2];
        let valid_inputs = [[0.0; INPUT_SIZE]; 2];
        let expected_retry = scalar_cohort_outputs(&family, &founders, &valid_inputs);
        // Fused multiply-add can retain an infinite accumulator through the
        // opposing finite product, after which tanh saturates to a valid value.
        // Unfused kernels can instead form inf + -inf and reject the NaN.
        // Preserve the actual scalar backend's outcome, including its rejection
        // boundary, rather than imposing a different overflow policy here.
        for sensors in [
            [2.0; INPUT_SIZE],
            [-2.0; INPUT_SIZE],
            [f32::MAX; INPUT_SIZE],
            [-f32::MAX; INPUT_SIZE],
            std::array::from_fn(|index| if index % 2 == 0 { f32::MAX } else { -f32::MAX }),
        ] {
            let inputs = [sensors; 2];
            let expected: Vec<_> = founders
                .iter()
                .zip(&inputs)
                .map(|((genome, state), input)| {
                    family
                        .evaluator(genome, state)
                        .expect("scalar evaluator")
                        .evaluate(input)
                })
                .collect();
            let mut outputs = sentinel;
            let result = cohort.evaluate_batch(&inputs, &mut outputs);
            if let Some(lane) = expected.iter().position(Result::is_err) {
                let error = result.expect_err("cohort must preserve scalar rejection");
                assert!(error.to_string().contains(&format!("lane {lane}")));
                assert_eq!(outputs, sentinel, "no partial cohort may be published");
            } else {
                result.expect("finite scalar outputs must remain accepted");
                let expected: Vec<_> = expected.into_iter().map(Result::unwrap).collect();
                assert_output_bits(&outputs, &expected);
            }
            assert_eq!(
                cohort
                    .checkpoint_states()
                    .expect("states after extreme inputs"),
                initial_states
            );
            cohort
                .evaluate_batch(&valid_inputs, &mut outputs)
                .expect("retry must overwrite previous scratch contents");
            assert_output_bits(&outputs, &expected_retry);
        }
    }

    #[test]
    fn architecture_and_mutation_bounds_fail_closed() {
        assert!(FtBrainConfig::new(Vec::new()).is_err());
        assert!(FtBrainConfig::new(vec![MAX_LAYER_WIDTH + 1]).is_err());

        let family = FtBrainFamily::default();
        let (genome, _) = founder(&family, 91);
        let mut rng = SmallRngStream::seed_from_u64(92);
        assert!(
            family
                .mutate_genome_material(
                    &genome,
                    MutationRates {
                        primary: 1.1,
                        secondary: 0.1,
                    },
                    &mut rng,
                )
                .is_err()
        );
    }

    #[test]
    fn zero_sigma_mutation_preserves_negative_zero_bits_and_each_gene_draw() {
        let family =
            FtBrainFamily::new(FtBrainConfig::new(vec![1]).expect("config")).expect("family");
        let parameter_count = validate_config(&family.config).expect("parameter count");
        let signed_zero_index = parameter_count / 2;
        let mut parameters = vec![0.25; parameter_count];
        parameters[signed_zero_index] = -0.0;
        let genome = genome_with_parameters(&family, &parameters);
        assert_eq!(
            family
                .decode_genome(&genome)
                .expect("decode signed-zero parent")
                .parameters[signed_zero_index]
                .to_bits(),
            (-0.0_f32).to_bits()
        );
        let provenance = BrainProvenance {
            parents: [Some(AgentUid(1)), None],
            parent_genome_hashes: [Some(genome.material_hash()), None],
            created_at: Tick(1),
            derivation: BrainGenomeDerivation::MutationOnly,
        };
        let mut actual_rng = ScriptedStream::selected_positive_gaussian(parameter_count);
        let child = family
            .mutate_genome(
                &genome,
                MutationRates {
                    primary: 1.0,
                    secondary: 0.0,
                },
                provenance,
                &mut actual_rng,
            )
            .expect("zero-sigma mutation");
        assert_eq!(child.material_hash(), genome.material_hash());
        assert_eq!(child.payload(), genome.payload());
        assert_eq!(
            family
                .decode_genome(&child)
                .expect("decode zero-sigma child")
                .parameters[signed_zero_index]
                .to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(actual_rng.consumed(), parameter_count * 3);
        assert_eq!(actual_rng.remaining(), 0);
    }

    #[test]
    fn crossover_mixes_each_flat_gene_in_stream_order() {
        let family =
            FtBrainFamily::new(FtBrainConfig::new(vec![1]).expect("config")).expect("family");
        let (left, _) = founder(&family, 201);
        let (right, _) = founder(&family, 202);
        let left_decoded = family.decode_genome(&left).expect("left");
        let right_decoded = family.decode_genome(&right).expect("right");
        let provenance = BrainProvenance {
            parents: [Some(AgentUid(1)), Some(AgentUid(2))],
            parent_genome_hashes: [Some(left.material_hash()), Some(right.material_hash())],
            created_at: Tick(1),
            derivation: BrainGenomeDerivation::Crossover,
        };
        let mut actual_rng = SmallRngStream::seed_from_u64(203);
        let child = family
            .crossover_genomes(&left, &right, provenance, &mut actual_rng)
            .expect("crossover");
        let child_parameters = family.decode_genome(&child).expect("child").parameters;

        let mut oracle_rng = SmallRngStream::seed_from_u64(203);
        let expected: Vec<f32> = left_decoded
            .parameters
            .iter()
            .zip(&right_decoded.parameters)
            .map(|(&left_gene, &right_gene)| {
                if unit_f32(&mut oracle_rng) < 0.5 {
                    left_gene
                } else {
                    right_gene
                }
            })
            .collect();
        assert_eq!(child_parameters, expected);
        assert_eq!(actual_rng.checkpoint(), oracle_rng.checkpoint());
    }
}
