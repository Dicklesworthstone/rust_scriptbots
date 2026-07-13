//! Multi-layer perceptron brain mirroring the legacy ScriptBots baseline.

use rand::Rng;
use scriptbots_core::{
    BrainEnvelopeKind, BrainEvaluator, BrainEvaluatorStateEnvelope, BrainFamilyAdapter,
    BrainFamilyId, BrainGenomeEnvelope, BrainInspection, BrainProtocolError, BrainProvenance,
    MutationRates, OffspringStatePolicy, RandomStream,
};
use serde::{Deserialize, Serialize};
use std::any::Any;

use scriptbots_core::{ActivationLayer, BrainActivations, BrainRunner, INPUT_SIZE, OUTPUT_SIZE};

use crate::{Brain, BrainKind, into_runner};

const BRAIN_SIZE: usize = 200;
const CONNECTIONS: usize = 4;
const BRAIN_SIZE_WIRE: u16 = 200;
const CONNECTIONS_WIRE: u8 = 4;
const MLP_FAMILY_ID: &str = "mlp-baseline";
const GENOME_SCHEMA_VERSION: u32 = 1;
const GENOME_CODEC_VERSION: u16 = 1;
const STATE_SCHEMA_VERSION: u32 = 1;
const STATE_CODEC_VERSION: u16 = 1;
const GENOME_MAGIC: [u8; 4] = *b"MLPG";
const STATE_MAGIC: [u8; 4] = *b"MLPS";
const GENOME_HEADER_BYTES: usize = 12;
const GENOME_NODE_BYTES: usize = CONNECTIONS * 4 + CONNECTIONS * 2 + CONNECTIONS + 3 * 4;
const GENOME_PAYLOAD_BYTES: usize = GENOME_HEADER_BYTES + BRAIN_SIZE * GENOME_NODE_BYTES;
const STATE_HEADER_BYTES: usize = 6;
const STATE_NODE_BYTES: usize = 3 * 4;
const STATE_PAYLOAD_BYTES: usize = STATE_HEADER_BYTES + BRAIN_SIZE * STATE_NODE_BYTES;
const INITIAL_DAMPING_MIN: f32 = 0.9;
const INITIAL_DAMPING_MAX: f32 = 1.1;
const MUTATED_DAMPING_MIN: f32 = 0.01;
const MUTATED_DAMPING_MAX: f32 = 1.0;

/// Identifies how a synapse samples its source neuron.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum SynapseKind {
    Regular,
    ChangeSensitive,
}

impl SynapseKind {
    fn random(rng: &mut dyn RandomStream) -> Self {
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct NodeParams {
    weights: [f32; CONNECTIONS],
    targets: [usize; CONNECTIONS],
    kinds: [SynapseKind; CONNECTIONS],
    gain: f32,
    damping: f32,
    bias: f32,
}

impl NodeParams {
    fn random(rng: &mut dyn RandomStream) -> Self {
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
            damping: rng.random_range(INITIAL_DAMPING_MIN..INITIAL_DAMPING_MAX),
            bias: rng.random_range(-2.0..2.0),
        }
    }
}

/// Dynamic state for each node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    pub fn random(rng: &mut dyn RandomStream) -> Self {
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
    pub fn runner(rng: &mut dyn RandomStream) -> Box<dyn BrainRunner> {
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

    fn gaussian(rng: &mut dyn RandomStream) -> f32 {
        const TWO_PI: f32 = std::f32::consts::TAU;
        let u1 = (rng.random::<f32>()).clamp(f32::MIN_POSITIVE, 1.0);
        let u2 = rng.random::<f32>();
        (-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).cos()
    }

    fn apply_damping(output: f32, target: f32, damping: f32) -> f32 {
        output + (target - output) * damping
    }

    fn mutate_parameters(&mut self, rng: &mut dyn RandomStream, rate: f32, scale: f32) {
        let sigma = scale.max(1e-5);
        for params in &mut self.nodes {
            if rng.random::<f32>() < rate {
                params.bias += Self::gaussian(rng) * sigma;
            }
            if rng.random::<f32>() < rate {
                params.damping = (params.damping + Self::gaussian(rng) * sigma)
                    .clamp(MUTATED_DAMPING_MIN, MUTATED_DAMPING_MAX);
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
                // Legacy retargets uniformly over the whole brain.
                params.targets[idx] = rng.random_range(0..BRAIN_SIZE);
            }
        }
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
            truncated: false,
        }
    }
}

#[derive(Debug)]
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
        let value = bytes.try_into().ok()?;
        self.offset = end;
        Some(value)
    }
}

/// Versioned protocol adapter for the baseline recurrent MLP family.
///
/// Heritable node parameters and future-affecting node dynamics use separate bounded codecs.
/// Children always start from a reset dynamic state; recurrent activity is checkpointed for run
/// continuation but is deliberately not inherited by offspring.
#[derive(Debug, Clone)]
pub struct MlpBrainFamily {
    family_id: BrainFamilyId,
}

impl MlpBrainFamily {
    /// Construct the canonical MLP family adapter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            family_id: BrainFamilyId::new(MLP_FAMILY_ID)
                .expect("the built-in MLP family identifier is canonical"),
        }
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

    fn validate_nodes(&self, nodes: &[NodeParams]) -> Result<(), BrainProtocolError> {
        if nodes.len() != BRAIN_SIZE {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                format!(
                    "MLP topology requires exactly {BRAIN_SIZE} nodes, found {}",
                    nodes.len()
                ),
            ));
        }
        for (node_index, node) in nodes.iter().enumerate() {
            for (connection, weight) in node.weights.iter().enumerate() {
                if !weight.is_finite() {
                    return Err(self.invalid(
                        BrainEnvelopeKind::Genome,
                        format!("node {node_index} weight {connection} is not finite"),
                    ));
                }
            }
            for (connection, target) in node.targets.iter().enumerate() {
                if *target >= BRAIN_SIZE {
                    return Err(self.invalid(
                        BrainEnvelopeKind::Genome,
                        format!(
                            "node {node_index} target {connection} is {target}, outside 0..{BRAIN_SIZE}"
                        ),
                    ));
                }
            }
            if !node.gain.is_finite() || node.gain < 0.0 {
                return Err(self.invalid(
                    BrainEnvelopeKind::Genome,
                    format!("node {node_index} gain must be finite and nonnegative"),
                ));
            }
            if !node.damping.is_finite()
                || !(MUTATED_DAMPING_MIN..=INITIAL_DAMPING_MAX).contains(&node.damping)
            {
                return Err(self.invalid(
                    BrainEnvelopeKind::Genome,
                    format!(
                        "node {node_index} damping must be within [{MUTATED_DAMPING_MIN}, {INITIAL_DAMPING_MAX}]"
                    ),
                ));
            }
            if !node.bias.is_finite() {
                return Err(self.invalid(
                    BrainEnvelopeKind::Genome,
                    format!("node {node_index} bias is not finite"),
                ));
            }
        }
        Ok(())
    }

    fn validate_state_values(&self, state: &[NodeState]) -> Result<(), BrainProtocolError> {
        if state.len() != BRAIN_SIZE {
            return Err(self.invalid(
                BrainEnvelopeKind::EvaluatorState,
                format!(
                    "MLP evaluator state requires exactly {BRAIN_SIZE} nodes, found {}",
                    state.len()
                ),
            ));
        }
        for (node_index, node) in state.iter().enumerate() {
            if !node.output.is_finite()
                || !node.previous_output.is_finite()
                || !node.target.is_finite()
            {
                return Err(self.invalid(
                    BrainEnvelopeKind::EvaluatorState,
                    format!("node {node_index} dynamic state contains a non-finite value"),
                ));
            }
        }
        Ok(())
    }

    fn encode_genome_payload(&self, nodes: &[NodeParams]) -> Result<Vec<u8>, BrainProtocolError> {
        self.validate_nodes(nodes)?;
        let input_size = u16::try_from(INPUT_SIZE).map_err(|_| {
            self.invalid(
                BrainEnvelopeKind::Genome,
                "input topology exceeds the u16 MLP wire field",
            )
        })?;
        let output_size = u16::try_from(OUTPUT_SIZE).map_err(|_| {
            self.invalid(
                BrainEnvelopeKind::Genome,
                "output topology exceeds the u16 MLP wire field",
            )
        })?;
        let mut payload = Vec::with_capacity(GENOME_PAYLOAD_BYTES);
        payload.extend_from_slice(&GENOME_MAGIC);
        payload.extend_from_slice(&BRAIN_SIZE_WIRE.to_le_bytes());
        payload.push(CONNECTIONS_WIRE);
        payload.extend_from_slice(&input_size.to_le_bytes());
        payload.extend_from_slice(&output_size.to_le_bytes());
        payload.push(0);
        for node in nodes {
            for weight in node.weights {
                payload.extend_from_slice(&weight.to_bits().to_le_bytes());
            }
            for target in node.targets {
                let target = u16::try_from(target).map_err(|_| {
                    self.invalid(
                        BrainEnvelopeKind::Genome,
                        "MLP connection target exceeds the u16 wire field",
                    )
                })?;
                payload.extend_from_slice(&target.to_le_bytes());
            }
            for kind in node.kinds {
                payload.push(match kind {
                    SynapseKind::Regular => 0,
                    SynapseKind::ChangeSensitive => 1,
                });
            }
            for value in [node.gain, node.damping, node.bias] {
                payload.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        debug_assert_eq!(payload.len(), GENOME_PAYLOAD_BYTES);
        Ok(payload)
    }

    fn decode_genome(
        &self,
        genome: &BrainGenomeEnvelope,
    ) -> Result<Vec<NodeParams>, BrainProtocolError> {
        genome.require_protocol(&self.family_id, GENOME_SCHEMA_VERSION, GENOME_CODEC_VERSION)?;
        let payload = genome.payload();
        if payload.len() != GENOME_PAYLOAD_BYTES {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                format!(
                    "MLP genome payload requires exactly {GENOME_PAYLOAD_BYTES} bytes, found {}",
                    payload.len()
                ),
            ));
        }
        let mut reader = PayloadReader::new(payload);
        let magic = self.read::<4>(&mut reader, BrainEnvelopeKind::Genome, "genome magic")?;
        let node_count = u16::from_le_bytes(self.read::<2>(
            &mut reader,
            BrainEnvelopeKind::Genome,
            "node count",
        )?);
        let connection_count =
            self.read::<1>(&mut reader, BrainEnvelopeKind::Genome, "connection count")?[0];
        let input_size = u16::from_le_bytes(self.read::<2>(
            &mut reader,
            BrainEnvelopeKind::Genome,
            "input count",
        )?);
        let output_size = u16::from_le_bytes(self.read::<2>(
            &mut reader,
            BrainEnvelopeKind::Genome,
            "output count",
        )?);
        let flags = self.read::<1>(&mut reader, BrainEnvelopeKind::Genome, "flags")?[0];
        if magic != GENOME_MAGIC
            || node_count != BRAIN_SIZE_WIRE
            || connection_count != CONNECTIONS_WIRE
            || usize::from(input_size) != INPUT_SIZE
            || usize::from(output_size) != OUTPUT_SIZE
            || flags != 0
        {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                format!(
                    "unsupported MLP topology header: magic={magic:?}, nodes={node_count}, connections={connection_count}, inputs={input_size}, outputs={output_size}, flags={flags}"
                ),
            ));
        }

        let mut nodes = Vec::with_capacity(BRAIN_SIZE);
        for node_index in 0..BRAIN_SIZE {
            let mut weights = [0.0; CONNECTIONS];
            for weight in &mut weights {
                *weight = f32::from_bits(u32::from_le_bytes(self.read::<4>(
                    &mut reader,
                    BrainEnvelopeKind::Genome,
                    "weight",
                )?));
            }
            let mut targets = [0; CONNECTIONS];
            for target in &mut targets {
                *target = usize::from(u16::from_le_bytes(self.read::<2>(
                    &mut reader,
                    BrainEnvelopeKind::Genome,
                    "target",
                )?));
            }
            let mut kinds = [SynapseKind::Regular; CONNECTIONS];
            for kind in &mut kinds {
                *kind = match self.read::<1>(
                    &mut reader,
                    BrainEnvelopeKind::Genome,
                    "synapse kind",
                )?[0]
                {
                    0 => SynapseKind::Regular,
                    1 => SynapseKind::ChangeSensitive,
                    value => {
                        return Err(self.invalid(
                            BrainEnvelopeKind::Genome,
                            format!("node {node_index} has unsupported synapse kind {value}"),
                        ));
                    }
                };
            }
            let gain = f32::from_bits(u32::from_le_bytes(self.read::<4>(
                &mut reader,
                BrainEnvelopeKind::Genome,
                "gain",
            )?));
            let damping = f32::from_bits(u32::from_le_bytes(self.read::<4>(
                &mut reader,
                BrainEnvelopeKind::Genome,
                "damping",
            )?));
            let bias = f32::from_bits(u32::from_le_bytes(self.read::<4>(
                &mut reader,
                BrainEnvelopeKind::Genome,
                "bias",
            )?));
            nodes.push(NodeParams {
                weights,
                targets,
                kinds,
                gain,
                damping,
                bias,
            });
        }
        self.validate_nodes(&nodes)?;
        Ok(nodes)
    }

    fn encode_state_payload(&self, state: &[NodeState]) -> Result<Vec<u8>, BrainProtocolError> {
        self.validate_state_values(state)?;
        let mut payload = Vec::with_capacity(STATE_PAYLOAD_BYTES);
        payload.extend_from_slice(&STATE_MAGIC);
        payload.extend_from_slice(&BRAIN_SIZE_WIRE.to_le_bytes());
        for node in state {
            for value in [node.output, node.previous_output, node.target] {
                payload.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        debug_assert_eq!(payload.len(), STATE_PAYLOAD_BYTES);
        Ok(payload)
    }

    fn decode_state(
        &self,
        state: &BrainEvaluatorStateEnvelope,
    ) -> Result<Vec<NodeState>, BrainProtocolError> {
        state.require_protocol(&self.family_id, STATE_SCHEMA_VERSION, STATE_CODEC_VERSION)?;
        let payload = state.payload();
        if payload.len() != STATE_PAYLOAD_BYTES {
            return Err(self.invalid(
                BrainEnvelopeKind::EvaluatorState,
                format!(
                    "MLP evaluator-state payload requires exactly {STATE_PAYLOAD_BYTES} bytes, found {}",
                    payload.len()
                ),
            ));
        }
        let mut reader = PayloadReader::new(payload);
        let magic = self.read::<4>(
            &mut reader,
            BrainEnvelopeKind::EvaluatorState,
            "state magic",
        )?;
        let node_count = u16::from_le_bytes(self.read::<2>(
            &mut reader,
            BrainEnvelopeKind::EvaluatorState,
            "state node count",
        )?);
        if magic != STATE_MAGIC || node_count != BRAIN_SIZE_WIRE {
            return Err(self.invalid(
                BrainEnvelopeKind::EvaluatorState,
                format!("unsupported MLP state header: magic={magic:?}, nodes={node_count}"),
            ));
        }
        let mut decoded = Vec::with_capacity(BRAIN_SIZE);
        for _ in 0..BRAIN_SIZE {
            let output = f32::from_bits(u32::from_le_bytes(self.read::<4>(
                &mut reader,
                BrainEnvelopeKind::EvaluatorState,
                "node output",
            )?));
            let previous_output = f32::from_bits(u32::from_le_bytes(self.read::<4>(
                &mut reader,
                BrainEnvelopeKind::EvaluatorState,
                "node previous output",
            )?));
            let target = f32::from_bits(u32::from_le_bytes(self.read::<4>(
                &mut reader,
                BrainEnvelopeKind::EvaluatorState,
                "node target",
            )?));
            decoded.push(NodeState {
                output,
                previous_output,
                target,
            });
        }
        self.validate_state_values(&decoded)?;
        Ok(decoded)
    }

    fn genome(
        &self,
        nodes: &[NodeParams],
        provenance: BrainProvenance,
    ) -> Result<BrainGenomeEnvelope, BrainProtocolError> {
        BrainGenomeEnvelope::new(
            self.family_id.clone(),
            GENOME_SCHEMA_VERSION,
            GENOME_CODEC_VERSION,
            self.encode_genome_payload(nodes)?,
            provenance,
        )
    }

    fn state(
        &self,
        state: &[NodeState],
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        BrainEvaluatorStateEnvelope::new(
            self.family_id.clone(),
            STATE_SCHEMA_VERSION,
            STATE_CODEC_VERSION,
            self.encode_state_payload(state)?,
        )
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
                "secondary mutation scale must be finite and nonnegative",
            ));
        }
        Ok(())
    }
}

impl Default for MlpBrainFamily {
    fn default() -> Self {
        Self::new()
    }
}

struct MlpEvaluator {
    family: MlpBrainFamily,
    brain: MlpBrain,
}

impl BrainEvaluator for MlpEvaluator {
    fn family_id(&self) -> &BrainFamilyId {
        &self.family.family_id
    }

    fn evaluate(
        &mut self,
        sensors: &[f32; INPUT_SIZE],
    ) -> Result<[f32; OUTPUT_SIZE], BrainProtocolError> {
        if sensors.iter().any(|value| !value.is_finite()) {
            return Err(self.family.invalid(
                BrainEnvelopeKind::EvaluatorState,
                "MLP sensor input contains a non-finite value",
            ));
        }
        let outputs = self.brain.tick(sensors);
        self.family.validate_state_values(&self.brain.state)?;
        if outputs.iter().any(|value| !value.is_finite()) {
            return Err(self.family.invalid(
                BrainEnvelopeKind::EvaluatorState,
                "MLP evaluation produced a non-finite output",
            ));
        }
        Ok(outputs)
    }

    fn inspect(
        &self,
        request: BrainInspection,
    ) -> Result<Option<BrainActivations>, BrainProtocolError> {
        match request {
            BrainInspection::Activations => Ok(Some(self.brain.activations())),
        }
    }

    fn checkpoint_state(&self) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.family.state(&self.brain.state)
    }
}

impl BrainFamilyAdapter for MlpBrainFamily {
    fn family_id(&self) -> &BrainFamilyId {
        &self.family_id
    }

    fn random_genome(
        &self,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeEnvelope, BrainProtocolError> {
        let brain = MlpBrain::random(rng);
        self.genome(&brain.nodes, BrainProvenance::default())
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

    fn mutate_genome(
        &self,
        genome: &BrainGenomeEnvelope,
        rates: MutationRates,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeEnvelope, BrainProtocolError> {
        self.validate_mutation_rates(rates)?;
        let mut brain = MlpBrain {
            nodes: self.decode_genome(genome)?,
            state: vec![NodeState::default(); BRAIN_SIZE],
        };
        brain.mutate_parameters(rng, rates.primary, rates.secondary);
        self.genome(&brain.nodes, genome.provenance().clone())
    }

    fn crossover_genomes(
        &self,
        left: &BrainGenomeEnvelope,
        right: &BrainGenomeEnvelope,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeEnvelope, BrainProtocolError> {
        let left_nodes = self.decode_genome(left)?;
        let right_nodes = self.decode_genome(right)?;
        let mut child_nodes = left_nodes;
        for (child, right) in child_nodes.iter_mut().zip(right_nodes) {
            if rng.random::<f32>() >= 0.5 {
                *child = right;
            }
        }
        let provenance = BrainProvenance {
            parents: [left.provenance().parents[0], right.provenance().parents[0]],
            created_at: scriptbots_core::Tick(
                left.provenance()
                    .created_at
                    .0
                    .max(right.provenance().created_at.0),
            ),
        };
        self.genome(&child_nodes, provenance)
    }

    fn initial_state(
        &self,
        genome: &BrainGenomeEnvelope,
        _rng: &mut dyn RandomStream,
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.validate_genome(genome)?;
        self.state(&vec![NodeState::default(); BRAIN_SIZE])
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
        let nodes = self.decode_genome(genome)?;
        let state = self.decode_state(state)?;
        Ok(Box::new(MlpEvaluator {
            family: self.clone(),
            brain: MlpBrain { nodes, state },
        }))
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
                // damping is applied raw: legacy kp ranges up to 1.1, and the
                // resulting overshoot is a real dynamical feature. Bounds are
                // enforced at init/mutation time instead.
                node.output = Self::apply_damping(node.output, node.target, params.damping);
            }
        }

        let mut result = [0.0; OUTPUT_SIZE];
        for (output, node) in result.iter_mut().zip(self.state.iter().rev()) {
            *output = node.output;
        }
        result
    }

    fn mutate(
        &mut self,
        rng: &mut dyn RandomStream,
        rate: f32,
        scale: f32,
    ) -> Result<(), crate::BrainMutationError> {
        self.mutate_parameters(rng, rate, scale);
        Ok(())
    }

    fn crossover(&self, other: &dyn Brain, rng: &mut dyn RandomStream) -> Option<Box<dyn Brain>> {
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

    fn clone_box(&self) -> Result<Box<dyn Brain>, crate::BrainCloneError> {
        Ok(Box::new(self.clone()))
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) {
        self
    }

    fn snapshot_activations(&self) -> Option<BrainActivations> {
        Some(self.activations())
    }
}

// Specialized adapter impl removed; generic adapter in lib.rs downcasts to call `activations()`.

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{AgentUid, SmallRngStream, Tick};

    fn fixture_nodes() -> Vec<NodeParams> {
        (0..BRAIN_SIZE)
            .map(|index| {
                let index_wire = u16::try_from(index).expect("fixture index fits u16");
                let scalar = f32::from(index_wire);
                NodeParams {
                    weights: [scalar + 0.25, -scalar - 0.5, scalar * 0.125, -0.0],
                    targets: [
                        index,
                        (index + 1) % BRAIN_SIZE,
                        (index + 17) % BRAIN_SIZE,
                        (BRAIN_SIZE - 1) - index,
                    ],
                    kinds: [
                        SynapseKind::Regular,
                        SynapseKind::ChangeSensitive,
                        if index % 2 == 0 {
                            SynapseKind::Regular
                        } else {
                            SynapseKind::ChangeSensitive
                        },
                        SynapseKind::Regular,
                    ],
                    gain: scalar * 0.03125,
                    damping: if index % 3 == 0 { 1.1 } else { 0.75 },
                    bias: scalar * -0.0625 + 1.25,
                }
            })
            .collect()
    }

    fn fixture_state() -> Vec<NodeState> {
        (0..BRAIN_SIZE)
            .map(|index| {
                let scalar = f32::from(u16::try_from(index).expect("fixture state index fits u16"));
                NodeState {
                    output: scalar * 0.01,
                    previous_output: scalar * -0.02,
                    target: scalar * 0.005 + 0.125,
                }
            })
            .collect()
    }

    fn fixture_provenance(left: u64, right: u64, tick: u64) -> BrainProvenance {
        BrainProvenance {
            parents: [Some(AgentUid(left)), Some(AgentUid(right))],
            created_at: Tick(tick),
        }
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    #[test]
    fn random_brain_has_expected_structure() {
        let mut rng = SmallRngStream::seed_from_u64(0xDEADBEEF);
        let brain = MlpBrain::random(&mut rng);
        assert_eq!(brain.nodes.len(), BRAIN_SIZE);
        assert_eq!(brain.state.len(), BRAIN_SIZE);
    }

    #[test]
    fn tick_produces_stable_outputs() {
        let mut rng = SmallRngStream::seed_from_u64(123);
        let mut brain = MlpBrain::random(&mut rng);
        let mut inputs = [0.0; INPUT_SIZE];
        inputs[0] = 1.0;
        let outputs = brain.tick(&inputs);
        assert!(outputs.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mutate_changes_parameters() {
        let mut rng = SmallRngStream::seed_from_u64(456);
        let mut brain = MlpBrain::random(&mut rng);
        let original = brain.nodes[10].bias;
        brain
            .mutate(&mut rng, 1.0, 0.5)
            .expect("MLP mutation is infallible");
        assert_ne!(brain.nodes[10].bias, original);
    }

    #[test]
    fn crossover_combines_parents() {
        let mut rng = SmallRngStream::seed_from_u64(789);
        let brain_a = MlpBrain::random(&mut rng);
        let brain_b = MlpBrain::random(&mut rng);
        let mut rng = SmallRngStream::seed_from_u64(101112);
        let child = brain_a
            .crossover(&brain_b, &mut rng)
            .expect("crossover should succeed");
        assert_eq!(child.kind(), MlpBrain::KIND);
    }

    #[test]
    fn runner_bridge_executes() {
        let mut rng = SmallRngStream::seed_from_u64(42);
        let mut runner = MlpBrain::runner(&mut rng);
        let inputs = [0.0; INPUT_SIZE];
        let outputs = runner.tick(&inputs);
        assert!(outputs.iter().all(|v| v.is_finite()));
        assert_eq!(runner.kind(), MlpBrain::KIND.as_str());
    }

    #[test]
    fn protocol_codecs_freeze_exact_topology_parameters_and_dynamic_state() {
        let family = MlpBrainFamily::new();
        let nodes = fixture_nodes();
        let provenance = fixture_provenance(41, 99, 12);
        let genome = family
            .genome(&nodes, provenance.clone())
            .expect("fixture genome");
        assert_eq!(genome.family_id(), family.family_id());
        assert_eq!(genome.schema_version(), GENOME_SCHEMA_VERSION);
        assert_eq!(genome.codec_version(), GENOME_CODEC_VERSION);
        assert_eq!(genome.provenance(), &provenance);
        assert_eq!(genome.payload().len(), GENOME_PAYLOAD_BYTES);
        assert_eq!(
            &genome.payload()[..GENOME_HEADER_BYTES],
            &[b'M', b'L', b'P', b'G', 200, 0, 4, 25, 0, 9, 0, 0]
        );
        assert_eq!(fnv1a64(genome.payload()), 0xa790_cc98_8dd9_29c7);
        let decoded_nodes = family.decode_genome(&genome).expect("decode genome");
        assert_eq!(decoded_nodes, nodes);
        assert_eq!(
            family
                .genome(&decoded_nodes, provenance)
                .expect("re-encode genome")
                .payload(),
            genome.payload()
        );

        let dynamic_state = fixture_state();
        let state = family.state(&dynamic_state).expect("fixture state");
        assert_eq!(state.family_id(), family.family_id());
        assert_eq!(state.schema_version(), STATE_SCHEMA_VERSION);
        assert_eq!(state.codec_version(), STATE_CODEC_VERSION);
        assert_eq!(state.payload().len(), STATE_PAYLOAD_BYTES);
        assert_eq!(
            &state.payload()[..STATE_HEADER_BYTES],
            &[b'M', b'L', b'P', b'S', 200, 0]
        );
        assert_eq!(fnv1a64(state.payload()), 0x3187_c3e9_4404_77c5);
        let decoded_state = family.decode_state(&state).expect("decode state");
        assert_eq!(decoded_state, dynamic_state);
        assert_eq!(
            family
                .state(&decoded_state)
                .expect("re-encode state")
                .payload(),
            state.payload()
        );
    }

    #[test]
    fn protocol_rejects_malformed_topology_parameters_and_state() {
        let family = MlpBrainFamily::new();
        let genome = family
            .genome(&fixture_nodes(), BrainProvenance::default())
            .expect("valid genome");

        let make_genome = |payload| {
            BrainGenomeEnvelope::new(
                family.family_id.clone(),
                GENOME_SCHEMA_VERSION,
                GENOME_CODEC_VERSION,
                payload,
                BrainProvenance::default(),
            )
            .expect("bounded malformed genome")
        };
        assert!(matches!(
            family.validate_genome(&make_genome(genome.payload()[..40].to_vec())),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::Genome,
                ..
            })
        ));

        let mut invalid_target = genome.payload().to_vec();
        invalid_target[GENOME_HEADER_BYTES + CONNECTIONS * 4..][..2]
            .copy_from_slice(&BRAIN_SIZE_WIRE.to_le_bytes());
        assert!(matches!(
            family.validate_genome(&make_genome(invalid_target)),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::Genome,
                ..
            })
        ));

        let mut invalid_kind = genome.payload().to_vec();
        invalid_kind[GENOME_HEADER_BYTES + CONNECTIONS * 4 + CONNECTIONS * 2] = 2;
        assert!(matches!(
            family.validate_genome(&make_genome(invalid_kind)),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::Genome,
                ..
            })
        ));

        let mut non_finite_weight = genome.payload().to_vec();
        non_finite_weight[GENOME_HEADER_BYTES..][..4]
            .copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        assert!(matches!(
            family.validate_genome(&make_genome(non_finite_weight)),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::Genome,
                ..
            })
        ));

        let state = family.state(&fixture_state()).expect("valid state");
        let mut non_finite_state = state.payload().to_vec();
        non_finite_state[STATE_HEADER_BYTES..][..4]
            .copy_from_slice(&f32::INFINITY.to_bits().to_le_bytes());
        let non_finite_state = BrainEvaluatorStateEnvelope::new(
            family.family_id.clone(),
            STATE_SCHEMA_VERSION,
            STATE_CODEC_VERSION,
            non_finite_state,
        )
        .expect("bounded malformed state");
        assert!(matches!(
            family.validate_evaluator_state(&non_finite_state),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::EvaluatorState,
                ..
            })
        ));
    }

    #[test]
    fn protocol_mutation_crossover_and_reset_touch_only_declared_fields() {
        let family = MlpBrainFamily::new();
        let left_nodes = fixture_nodes();
        let mut right_nodes = fixture_nodes();
        for node in &mut right_nodes {
            for weight in &mut node.weights {
                *weight = -*weight + 3.0;
            }
            for target in &mut node.targets {
                *target = (*target + 53) % BRAIN_SIZE;
            }
            for kind in &mut node.kinds {
                *kind = kind.flip();
            }
            node.gain += 0.5;
            node.damping = 0.5;
            node.bias -= 2.0;
        }
        let left_provenance = fixture_provenance(10, 11, 4);
        let right_provenance = fixture_provenance(20, 21, 6);
        let left = family
            .genome(&left_nodes, left_provenance.clone())
            .expect("left genome");
        let right = family
            .genome(&right_nodes, right_provenance)
            .expect("right genome");

        let mut mutation_rng = SmallRngStream::seed_from_u64(0xA11C_E5E5);
        let mutated = family
            .mutate_genome(
                &left,
                MutationRates {
                    primary: 1.0,
                    secondary: 0.25,
                },
                &mut mutation_rng,
            )
            .expect("mutated genome");
        assert_eq!(mutated.provenance(), &left_provenance);
        let mutated_nodes = family.decode_genome(&mutated).expect("mutated nodes");
        assert_eq!(mutated_nodes.len(), BRAIN_SIZE);
        assert_ne!(mutated_nodes, left_nodes);
        for (before, after) in left_nodes.iter().zip(&mutated_nodes) {
            assert!(
                before
                    .weights
                    .iter()
                    .zip(after.weights)
                    .filter(|(left, right)| left != &right)
                    .count()
                    <= 1
            );
            assert!(
                before
                    .targets
                    .iter()
                    .zip(after.targets)
                    .filter(|(left, right)| left != &right)
                    .count()
                    <= 1
            );
            assert!(
                before
                    .kinds
                    .iter()
                    .zip(after.kinds)
                    .filter(|(left, right)| left != &right)
                    .count()
                    <= 1
            );
            assert!(after.targets.iter().all(|target| *target < BRAIN_SIZE));
            assert!((MUTATED_DAMPING_MIN..=MUTATED_DAMPING_MAX).contains(&after.damping));
        }

        let mut crossover_rng = SmallRngStream::seed_from_u64(0xC205_50FE);
        let child = family
            .crossover_genomes(&left, &right, &mut crossover_rng)
            .expect("child genome");
        let child_nodes = family.decode_genome(&child).expect("child nodes");
        let mut selected_left = false;
        let mut selected_right = false;
        for ((child, left), right) in child_nodes.iter().zip(&left_nodes).zip(&right_nodes) {
            assert!(child == left || child == right);
            selected_left |= child == left;
            selected_right |= child == right;
        }
        assert!(selected_left && selected_right);
        assert_eq!(
            child.provenance(),
            &BrainProvenance {
                parents: [Some(AgentUid(10)), Some(AgentUid(20))],
                created_at: Tick(6),
            }
        );

        let first_parent_state = family.state(&fixture_state()).expect("first parent state");
        let mut second_dynamic = fixture_state();
        second_dynamic[0].output = 99.0;
        let second_parent_state = family.state(&second_dynamic).expect("second parent state");
        assert_eq!(family.offspring_state_policy(), OffspringStatePolicy::Reset);
        let reset = family
            .offspring_state(
                &child,
                &[&first_parent_state, &second_parent_state],
                &mut crossover_rng,
            )
            .expect("reset offspring state");
        assert_eq!(
            family.decode_state(&reset).expect("decoded reset state"),
            vec![NodeState::default(); BRAIN_SIZE]
        );
    }

    #[test]
    fn checkpoint_reconstruction_preserves_the_exact_next_output() {
        let family = MlpBrainFamily::new();
        let mut rng = SmallRngStream::seed_from_u64(0x5EED_F00D);
        let genome = family.random_genome(&mut rng).expect("random genome");
        let state = family
            .initial_state(&genome, &mut rng)
            .expect("initial state");
        let mut evaluator = family.evaluator(&genome, &state).expect("evaluator");

        let mut primer = [0.0; INPUT_SIZE];
        for (index, value) in primer.iter_mut().enumerate() {
            *value = f32::from(u16::try_from(index).expect("input index fits u16")) * 0.03125;
        }
        evaluator.evaluate(&primer).expect("primer output");
        let checkpoint = family
            .checkpoint_evaluator(evaluator.as_ref())
            .expect("validated checkpoint");
        let mut restored = family
            .evaluator(&genome, &checkpoint)
            .expect("restored evaluator");

        let mut next_inputs = [0.0; INPUT_SIZE];
        for (index, value) in next_inputs.iter_mut().enumerate() {
            *value =
                f32::from(u16::try_from(index).expect("input index fits u16")) * -0.015625 + 0.75;
        }
        assert_eq!(
            evaluator.evaluate(&next_inputs).expect("continued output"),
            restored.evaluate(&next_inputs).expect("restored output")
        );
        assert_eq!(
            family
                .checkpoint_evaluator(evaluator.as_ref())
                .expect("continued checkpoint"),
            family
                .checkpoint_evaluator(restored.as_ref())
                .expect("restored checkpoint")
        );
        assert!(
            restored
                .inspect(BrainInspection::Activations)
                .expect("bounded inspection")
                .is_some()
        );
    }

    #[test]
    fn legacy_damping_oracle_preserves_initial_overshoot_then_mutation_cap() {
        assert_eq!(INITIAL_DAMPING_MIN, 0.9);
        assert_eq!(INITIAL_DAMPING_MAX, 1.1);
        assert_eq!(MUTATED_DAMPING_MIN, 0.01);
        assert_eq!(MUTATED_DAMPING_MAX, 1.0);

        let family = MlpBrainFamily::new();
        let node = NodeParams {
            weights: [0.0; CONNECTIONS],
            targets: [0; CONNECTIONS],
            kinds: [SynapseKind::Regular; CONNECTIONS],
            gain: 0.0,
            damping: INITIAL_DAMPING_MAX,
            bias: 0.0,
        };
        let genome = family
            .genome(&vec![node; BRAIN_SIZE], BrainProvenance::default())
            .expect("overshooting genome");
        let mut rng = SmallRngStream::seed_from_u64(77);
        let state = family.initial_state(&genome, &mut rng).expect("zero state");
        let mut evaluator = family.evaluator(&genome, &state).expect("evaluator");
        let output = evaluator
            .evaluate(&[0.0; INPUT_SIZE])
            .expect("legacy damping output")[0];
        let legacy_target = MlpBrain::logistic(0.0);
        let legacy_expected = MlpBrain::apply_damping(0.0, legacy_target, INITIAL_DAMPING_MAX);
        assert_eq!(output.to_bits(), legacy_expected.to_bits());
        assert!(output > legacy_target, "1.1 damping must overshoot target");

        let mutated = family
            .mutate_genome(
                &genome,
                MutationRates {
                    primary: 1.0,
                    secondary: 0.0,
                },
                &mut rng,
            )
            .expect("legacy-clamped mutation");
        assert!(
            family
                .decode_genome(&mutated)
                .expect("mutated nodes")
                .iter()
                .all(|node| (MUTATED_DAMPING_MIN..=MUTATED_DAMPING_MAX).contains(&node.damping))
        );
    }
}
