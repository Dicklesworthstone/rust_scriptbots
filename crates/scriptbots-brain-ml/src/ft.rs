//! Frankentorch-backed dense feed-forward brain family.
//!
//! The heritable representation is a bounded, versioned, flat `Vec<f32>`. Founder
//! construction uses `ft_nn::parameters_to_vector` as the canonical flattening
//! oracle. The pinned Frankentorch `vector_to_parameters` implementation rebuilds
//! tensors through F64 storage, so inference deliberately materializes validated
//! F32 layer slices directly until that upstream API is F32-safe.

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use scriptbots_core::{
    BrainAdapterIdentityV1, BrainBatchArchitectureKey, BrainBatchEvaluator, BrainEnvelopeKind,
    BrainEvaluator, BrainEvaluatorStateEnvelope, BrainFamilyCodec, BrainFamilyId,
    BrainGenomeEnvelope, BrainGenomeMaterial, BrainInspection, BrainInspectionError,
    BrainInspectionSnapshot, BrainProtocolError, INPUT_SIZE, MAX_BRAIN_GENOME_PAYLOAD_BYTES,
    MutationRates, OUTPUT_SIZE, OffspringStatePolicy, OutputChannel, RandomStream,
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
const ADAPTER_SEMANTIC_VERSION: u32 = 1;
const ADAPTER_SEMANTIC_PREFIX: &[u8] = b"dense-f32;ft-kernel-cpu::linear_tensor_f32;\
    layer-order=weight[out,in],bias[out];hidden=tanh;heads=typed-all-sigmoid;\
    scalar=batch-loop-identical;state=genome-digest-only;layout=v1;\
    founder=xavier-uniform-next-u32-bias-zero;mutation=per-gene-box-muller;\
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
        Ok(None)
    }

    fn checkpoint_state(&self) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        state_from_digest(&self.family, self.genome_digest)
    }
}

struct FtBatchEvaluator {
    family: FtBrainFamily,
    genome_digests: Vec<[u8; 32]>,
    networks: Vec<FtNetwork>,
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
        if self.networks.len() != sensors.len() || sensors.len() != outputs.len() {
            return Err(BrainProtocolError::BatchCardinalityMismatch {
                evaluators: self.networks.len(),
                inputs: sensors.len(),
                outputs: outputs.len(),
            });
        }
        let mut candidates = Vec::with_capacity(outputs.len());
        for (network, input) in self.networks.iter().zip(sensors) {
            candidates.push(network.evaluate(&self.family, input)?);
        }
        outputs.copy_from_slice(&candidates);
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

    fn random_genome_material(
        &self,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        self.genome_material(&self.config, &self.random_parameters(rng)?)
    }

    fn validate_genome(&self, genome: &BrainGenomeEnvelope) -> Result<(), BrainProtocolError> {
        self.decode_genome(genome).map(|_| ())
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
                *parameter += gaussian_f32(rng) * rates.secondary;
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
        let mut networks = Vec::with_capacity(genomes.len());
        let mut genome_digests = Vec::with_capacity(genomes.len());
        for (&genome, &state) in genomes.iter().zip(states) {
            let decoded = self.validate_pair(genome, state)?;
            networks.push(FtNetwork::materialize(&decoded));
            genome_digests.push(*genome.material_hash().as_bytes());
        }
        Ok(Some(Box::new(FtBatchEvaluator {
            family: self.clone(),
            genome_digests,
            networks,
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
    use scriptbots_core::{
        AgentUid, BrainFamilyAdapter, BrainGenomeDerivation, BrainProvenance, SmallRngStream, Tick,
    };

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
    fn zero_sigma_mutation_is_material_stable_and_consumes_each_gene_draw() {
        let family =
            FtBrainFamily::new(FtBrainConfig::new(vec![1]).expect("config")).expect("family");
        let (genome, _) = founder(&family, 101);
        let parameter_count = family
            .decode_genome(&genome)
            .expect("decode parent")
            .parameters
            .len();
        let provenance = BrainProvenance {
            parents: [Some(AgentUid(1)), None],
            parent_genome_hashes: [Some(genome.material_hash()), None],
            created_at: Tick(1),
            derivation: BrainGenomeDerivation::MutationOnly,
        };
        let mut actual_rng = SmallRngStream::seed_from_u64(102);
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

        let mut oracle_rng = SmallRngStream::seed_from_u64(102);
        for _ in 0..parameter_count {
            let _probability = oracle_rng.next_u32();
            let _gaussian_u1 = oracle_rng.next_u32();
            let _gaussian_u2 = oracle_rng.next_u32();
        }
        assert_eq!(actual_rng.checkpoint(), oracle_rng.checkpoint());
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
