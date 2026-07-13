//! Feature-gated DWRAON brain (Damped Weighted Recurrent AND/OR Network).

use rand::Rng;
use scriptbots_core::{
    ActivationLayer, BrainActivations, BrainEnvelopeKind, BrainEvaluator,
    BrainEvaluatorStateEnvelope, BrainFamilyAdapter, BrainFamilyId, BrainGenomeEnvelope,
    BrainInspection, BrainProtocolError, BrainProvenance, MutationRates, OffspringStatePolicy,
    RandomStream, Tick,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::sync::LazyLock;

use scriptbots_core::{BrainRunner, INPUT_SIZE, OUTPUT_SIZE};

use crate::{Brain, BrainKind, into_runner};

const BRAIN_SIZE: usize = 200;
const CONNECTIONS: usize = 4;
const DWRAON_FAMILY_NAME: &str = "dwraon-baseline";
const GENOME_SCHEMA_VERSION: u32 = 1;
const GENOME_CODEC_VERSION: u16 = 1;
const STATE_SCHEMA_VERSION: u32 = 1;
const STATE_CODEC_VERSION: u16 = 1;
const GENOME_MAGIC: [u8; 4] = *b"DWGN";
const STATE_MAGIC: [u8; 4] = *b"DWST";
const GENOME_HEADER_BYTES: usize = 8;
const GENOME_NODE_BYTES: usize = 1 + 4 + 4 + CONNECTIONS * (4 + 2 + 1);
const GENOME_PAYLOAD_BYTES: usize = GENOME_HEADER_BYTES + BRAIN_SIZE * GENOME_NODE_BYTES;
const STATE_HEADER_BYTES: usize = 6;
const STATE_NODE_BYTES: usize = 4;
const STATE_PAYLOAD_BYTES: usize = STATE_HEADER_BYTES + BRAIN_SIZE * STATE_NODE_BYTES;

static DWRAON_FAMILY_ID: LazyLock<BrainFamilyId> = LazyLock::new(|| {
    BrainFamilyId::new(DWRAON_FAMILY_NAME).expect("the built-in DWRAON family id is canonical")
});

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum NodeKind {
    And,
    Or,
}

impl NodeKind {
    fn random(rng: &mut dyn RandomStream) -> Self {
        if rng.random::<f32>() > 0.5 {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
        let mut sources = [0usize; CONNECTIONS];
        let mut inverted = [false; CONNECTIONS];
        // The preserved constructor initializes each connection atomically in
        // weight/source/sensor-bias/inversion order. Keep that sampling order;
        // changing to one pass per field produces the same distributions but
        // a different deterministic genome for the same stream.
        for connection in 0..CONNECTIONS {
            weights[connection] = rng.random_range(0.1..2.0);
            sources[connection] = rng.random_range(0..BRAIN_SIZE);
            if rng.random::<f32>() < 0.2 {
                sources[connection] = rng.random_range(0..INPUT_SIZE);
            }
            inverted[connection] = rng.random::<f32>() < 0.5;
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
            // Preserve the legacy constructor's exact source policy: first generate the
            // provisional full-brain/20%-sensor-biased sources above, then overwrite all four
            // sources in the first half with `[0, INPUT_SIZE)` sensor indices. Nodes 0..INPUT_SIZE
            // are input latches whose source genes are dormant; nodes INPUT_SIZE..100 form the
            // large reactive core. The redundant draws and dormant genes remain intentional so
            // genome construction has the same shape and sampling order as the preserved C++.
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

    fn activations(&self) -> BrainActivations {
        BrainActivations {
            layers: vec![ActivationLayer {
                name: "dwraon.state".to_owned(),
                width: 20,
                height: 10,
                values: self.state.iter().map(|node| node.output).collect(),
            }],
            connections: Vec::new(),
            truncated: false,
        }
    }
}

fn invalid_payload(kind: BrainEnvelopeKind, detail: impl Into<String>) -> BrainProtocolError {
    BrainProtocolError::InvalidPayload {
        kind,
        family_id: DWRAON_FAMILY_ID.clone(),
        detail: detail.into(),
    }
}

fn validate_nodes(nodes: &[NodeParams]) -> Result<(), BrainProtocolError> {
    if nodes.len() != BRAIN_SIZE {
        return Err(invalid_payload(
            BrainEnvelopeKind::Genome,
            format!(
                "DWRAON genome has {} nodes; expected exactly {BRAIN_SIZE}",
                nodes.len()
            ),
        ));
    }

    for (node_index, node) in nodes.iter().enumerate() {
        if !node.damping.is_finite() || !(0.01..=1.0).contains(&node.damping) {
            return Err(invalid_payload(
                BrainEnvelopeKind::Genome,
                format!(
                    "node {node_index} damping {} is outside the finite [0.01, 1] range",
                    node.damping
                ),
            ));
        }
        if !node.bias.is_finite() {
            return Err(invalid_payload(
                BrainEnvelopeKind::Genome,
                format!("node {node_index} bias is not finite"),
            ));
        }
        for connection in 0..CONNECTIONS {
            let weight = node.weights[connection];
            if !weight.is_finite() || weight < 0.01 {
                return Err(invalid_payload(
                    BrainEnvelopeKind::Genome,
                    format!(
                        "node {node_index} connection {connection} weight {weight} is not finite and >= 0.01"
                    ),
                ));
            }
            let source = node.sources[connection];
            if source >= BRAIN_SIZE {
                return Err(invalid_payload(
                    BrainEnvelopeKind::Genome,
                    format!(
                        "node {node_index} connection {connection} source {source} is outside 0..{BRAIN_SIZE}"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_state(state: &[NodeState]) -> Result<(), BrainProtocolError> {
    if state.len() != BRAIN_SIZE {
        return Err(invalid_payload(
            BrainEnvelopeKind::EvaluatorState,
            format!(
                "DWRAON evaluator state has {} nodes; expected exactly {BRAIN_SIZE}",
                state.len()
            ),
        ));
    }
    for (node_index, node) in state.iter().enumerate() {
        if !node.output.is_finite() || !(0.0..=1.0).contains(&node.output) {
            return Err(invalid_payload(
                BrainEnvelopeKind::EvaluatorState,
                format!(
                    "node {node_index} output {} is outside the finite [0, 1] range",
                    node.output
                ),
            ));
        }
    }
    Ok(())
}

fn encode_genome_payload(nodes: &[NodeParams]) -> Result<Vec<u8>, BrainProtocolError> {
    validate_nodes(nodes)?;
    let brain_size = u16::try_from(BRAIN_SIZE).expect("DWRAON brain size fits the v1 wire field");
    let connections =
        u8::try_from(CONNECTIONS).expect("DWRAON connection count fits the v1 wire field");
    let input_size = u8::try_from(INPUT_SIZE).expect("DWRAON input count fits the v1 wire field");
    let mut payload = Vec::with_capacity(GENOME_PAYLOAD_BYTES);
    payload.extend_from_slice(&GENOME_MAGIC);
    payload.extend_from_slice(&brain_size.to_le_bytes());
    payload.push(connections);
    payload.push(input_size);
    for node in nodes {
        payload.push(match node.kind {
            NodeKind::And => 0,
            NodeKind::Or => 1,
        });
        payload.extend_from_slice(&node.damping.to_bits().to_le_bytes());
        payload.extend_from_slice(&node.bias.to_bits().to_le_bytes());
        for connection in 0..CONNECTIONS {
            payload.extend_from_slice(&node.weights[connection].to_bits().to_le_bytes());
            let source = u16::try_from(node.sources[connection])
                .expect("validated DWRAON source fits the v1 wire field");
            payload.extend_from_slice(&source.to_le_bytes());
            payload.push(u8::from(node.inverted[connection]));
        }
    }
    debug_assert_eq!(payload.len(), GENOME_PAYLOAD_BYTES);
    Ok(payload)
}

fn decode_genome_payload(payload: &[u8]) -> Result<Vec<NodeParams>, BrainProtocolError> {
    if payload.len() != GENOME_PAYLOAD_BYTES {
        return Err(invalid_payload(
            BrainEnvelopeKind::Genome,
            format!(
                "DWRAON v1 genome payload is {} bytes; expected exactly {GENOME_PAYLOAD_BYTES}",
                payload.len()
            ),
        ));
    }
    if payload[..4] != GENOME_MAGIC {
        return Err(invalid_payload(
            BrainEnvelopeKind::Genome,
            "DWRAON v1 genome magic does not match DWGN",
        ));
    }
    let encoded_brain_size = usize::from(u16::from_le_bytes([payload[4], payload[5]]));
    if encoded_brain_size != BRAIN_SIZE
        || usize::from(payload[6]) != CONNECTIONS
        || usize::from(payload[7]) != INPUT_SIZE
    {
        return Err(invalid_payload(
            BrainEnvelopeKind::Genome,
            format!(
                "DWRAON v1 layout declares {encoded_brain_size} nodes, {} connections, and {} inputs",
                payload[6], payload[7]
            ),
        ));
    }

    let mut cursor = GENOME_HEADER_BYTES;
    let mut nodes = Vec::with_capacity(BRAIN_SIZE);
    for node_index in 0..BRAIN_SIZE {
        let kind = match payload[cursor] {
            0 => NodeKind::And,
            1 => NodeKind::Or,
            value => {
                return Err(invalid_payload(
                    BrainEnvelopeKind::Genome,
                    format!("node {node_index} has unknown kind tag {value}"),
                ));
            }
        };
        cursor += 1;
        let damping = decode_f32(payload, cursor);
        cursor += 4;
        let bias = decode_f32(payload, cursor);
        cursor += 4;
        let mut weights = [0.0; CONNECTIONS];
        let mut sources = [0; CONNECTIONS];
        let mut inverted = [false; CONNECTIONS];
        for connection in 0..CONNECTIONS {
            weights[connection] = decode_f32(payload, cursor);
            cursor += 4;
            sources[connection] =
                usize::from(u16::from_le_bytes([payload[cursor], payload[cursor + 1]]));
            cursor += 2;
            inverted[connection] = match payload[cursor] {
                0 => false,
                1 => true,
                value => {
                    return Err(invalid_payload(
                        BrainEnvelopeKind::Genome,
                        format!(
                            "node {node_index} connection {connection} has invalid inversion tag {value}"
                        ),
                    ));
                }
            };
            cursor += 1;
        }
        nodes.push(NodeParams {
            kind,
            damping,
            bias,
            weights,
            sources,
            inverted,
        });
    }
    debug_assert_eq!(cursor, payload.len());
    validate_nodes(&nodes)?;
    Ok(nodes)
}

fn decode_f32(payload: &[u8], offset: usize) -> f32 {
    f32::from_bits(u32::from_le_bytes([
        payload[offset],
        payload[offset + 1],
        payload[offset + 2],
        payload[offset + 3],
    ]))
}

fn encode_state_payload(state: &[NodeState]) -> Result<Vec<u8>, BrainProtocolError> {
    validate_state(state)?;
    let brain_size = u16::try_from(BRAIN_SIZE).expect("DWRAON brain size fits the v1 wire field");
    let mut payload = Vec::with_capacity(STATE_PAYLOAD_BYTES);
    payload.extend_from_slice(&STATE_MAGIC);
    payload.extend_from_slice(&brain_size.to_le_bytes());
    // `target` is intentionally absent. Every target is recomputed from the
    // checkpointed outputs before the evaluator reads it, so it is scratch,
    // not future-affecting recurrent state.
    for node in state {
        payload.extend_from_slice(&node.output.to_bits().to_le_bytes());
    }
    debug_assert_eq!(payload.len(), STATE_PAYLOAD_BYTES);
    Ok(payload)
}

fn decode_state_payload(payload: &[u8]) -> Result<Vec<NodeState>, BrainProtocolError> {
    if payload.len() != STATE_PAYLOAD_BYTES {
        return Err(invalid_payload(
            BrainEnvelopeKind::EvaluatorState,
            format!(
                "DWRAON v1 evaluator-state payload is {} bytes; expected exactly {STATE_PAYLOAD_BYTES}",
                payload.len()
            ),
        ));
    }
    if payload[..4] != STATE_MAGIC {
        return Err(invalid_payload(
            BrainEnvelopeKind::EvaluatorState,
            "DWRAON v1 evaluator-state magic does not match DWST",
        ));
    }
    let encoded_brain_size = usize::from(u16::from_le_bytes([payload[4], payload[5]]));
    if encoded_brain_size != BRAIN_SIZE {
        return Err(invalid_payload(
            BrainEnvelopeKind::EvaluatorState,
            format!("DWRAON v1 state declares {encoded_brain_size} nodes"),
        ));
    }
    let state = (0..BRAIN_SIZE)
        .map(|index| NodeState {
            output: decode_f32(payload, STATE_HEADER_BYTES + index * STATE_NODE_BYTES),
            target: 0.0,
        })
        .collect::<Vec<_>>();
    validate_state(&state)?;
    Ok(state)
}

fn decode_genome(genome: &BrainGenomeEnvelope) -> Result<Vec<NodeParams>, BrainProtocolError> {
    genome.require_protocol(
        &DWRAON_FAMILY_ID,
        GENOME_SCHEMA_VERSION,
        GENOME_CODEC_VERSION,
    )?;
    decode_genome_payload(genome.payload())
}

fn decode_state(state: &BrainEvaluatorStateEnvelope) -> Result<Vec<NodeState>, BrainProtocolError> {
    state.require_protocol(&DWRAON_FAMILY_ID, STATE_SCHEMA_VERSION, STATE_CODEC_VERSION)?;
    decode_state_payload(state.payload())
}

fn genome_envelope(
    nodes: &[NodeParams],
    provenance: BrainProvenance,
) -> Result<BrainGenomeEnvelope, BrainProtocolError> {
    BrainGenomeEnvelope::new(
        DWRAON_FAMILY_ID.clone(),
        GENOME_SCHEMA_VERSION,
        GENOME_CODEC_VERSION,
        encode_genome_payload(nodes)?,
        provenance,
    )
}

fn state_envelope(state: &[NodeState]) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
    BrainEvaluatorStateEnvelope::new(
        DWRAON_FAMILY_ID.clone(),
        STATE_SCHEMA_VERSION,
        STATE_CODEC_VERSION,
        encode_state_payload(state)?,
    )
}

fn validate_mutation_rates(rates: MutationRates) -> Result<(), BrainProtocolError> {
    if !rates.primary.is_finite() || !(0.0..=1.0).contains(&rates.primary) {
        return Err(invalid_payload(
            BrainEnvelopeKind::Genome,
            format!(
                "DWRAON primary mutation probability {} is outside the finite [0, 1] range",
                rates.primary
            ),
        ));
    }
    if !rates.secondary.is_finite() || rates.secondary < 0.0 {
        return Err(invalid_payload(
            BrainEnvelopeKind::Genome,
            format!(
                "DWRAON secondary mutation scale {} is not finite and nonnegative",
                rates.secondary
            ),
        ));
    }
    Ok(())
}

fn mutate_nodes(nodes: &mut [NodeParams], rng: &mut dyn RandomStream, rate: f32, scale: f32) {
    for params in nodes {
        if rng.random::<f32>() < rate * 3.0 {
            params.bias += DwraonBrain::gaussian(rng) * scale;
        }
        // Legacy contains a permanently disabled damping mutation branch; do
        // not consume a random draw or silently activate it here.
        if rng.random::<f32>() < rate * 3.0 {
            let index = rng.random_range(0..CONNECTIONS);
            let weight = params.weights[index] + DwraonBrain::gaussian(rng) * scale;
            params.weights[index] = weight.max(0.01);
        }
        if rng.random::<f32>() < rate {
            let index = rng.random_range(0..CONNECTIONS);
            params.sources[index] = rng.random_range(0..BRAIN_SIZE);
        }
        if rng.random::<f32>() < rate {
            let index = rng.random_range(0..CONNECTIONS);
            params.inverted[index] = !params.inverted[index];
        }
        if rng.random::<f32>() < rate {
            params.kind = params.kind.toggle();
        }
    }
}

fn crossover_nodes(
    left: &[NodeParams],
    right: &[NodeParams],
    rng: &mut dyn RandomStream,
) -> Vec<NodeParams> {
    // Match `DWRAONBrain::crossover` field-for-field and in draw order. A
    // sample below 0.5 selects the left parent; bias, damping, kind, then each
    // source/inversion/weight receive independent coins. This intentionally
    // permits a node to contain co-adapted fields from both parents.
    left.iter()
        .zip(right)
        .map(|(left_node, right_node)| {
            let mut child = left_node.clone();
            if rng.random::<f32>() >= 0.5 {
                child.bias = right_node.bias;
            }
            if rng.random::<f32>() >= 0.5 {
                child.damping = right_node.damping;
            }
            if rng.random::<f32>() >= 0.5 {
                child.kind = right_node.kind;
            }
            for connection in 0..CONNECTIONS {
                if rng.random::<f32>() >= 0.5 {
                    child.sources[connection] = right_node.sources[connection];
                }
                if rng.random::<f32>() >= 0.5 {
                    child.inverted[connection] = right_node.inverted[connection];
                }
                if rng.random::<f32>() >= 0.5 {
                    child.weights[connection] = right_node.weights[connection];
                }
            }
            child
        })
        .collect()
}

/// Versioned genome/evaluator-state protocol adapter for the DWRAON family.
#[derive(Debug, Clone)]
pub struct DwraonFamilyAdapter {
    family_id: BrainFamilyId,
}

impl Default for DwraonFamilyAdapter {
    fn default() -> Self {
        Self {
            family_id: DWRAON_FAMILY_ID.clone(),
        }
    }
}

#[derive(Debug)]
struct DwraonEvaluator {
    family_id: BrainFamilyId,
    brain: DwraonBrain,
}

impl BrainEvaluator for DwraonEvaluator {
    fn family_id(&self) -> &BrainFamilyId {
        &self.family_id
    }

    fn evaluate(
        &mut self,
        sensors: &[f32; INPUT_SIZE],
    ) -> Result<[f32; OUTPUT_SIZE], BrainProtocolError> {
        if let Some((index, value)) = sensors
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(invalid_payload(
                BrainEnvelopeKind::EvaluatorState,
                format!("sensor {index} is non-finite ({value})"),
            ));
        }

        // A rejected evaluation cannot leave a partially advanced recurrent state. Evaluate a
        // candidate, validate every future-affecting output, and publish it atomically.
        let mut candidate = self.brain.clone();
        let outputs = candidate.tick(sensors);
        validate_state(&candidate.state)?;
        self.brain = candidate;
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
        state_envelope(&self.brain.state)
    }
}

impl BrainFamilyAdapter for DwraonFamilyAdapter {
    fn family_id(&self) -> &BrainFamilyId {
        &self.family_id
    }

    fn random_genome(
        &self,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeEnvelope, BrainProtocolError> {
        let brain = DwraonBrain::random(rng);
        genome_envelope(&brain.nodes, BrainProvenance::default())
    }

    fn validate_genome(&self, genome: &BrainGenomeEnvelope) -> Result<(), BrainProtocolError> {
        decode_genome(genome).map(|_| ())
    }

    fn validate_evaluator_state(
        &self,
        state: &BrainEvaluatorStateEnvelope,
    ) -> Result<(), BrainProtocolError> {
        decode_state(state).map(|_| ())
    }

    fn mutate_genome(
        &self,
        genome: &BrainGenomeEnvelope,
        rates: MutationRates,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeEnvelope, BrainProtocolError> {
        validate_mutation_rates(rates)?;
        let mut nodes = decode_genome(genome)?;
        mutate_nodes(&mut nodes, rng, rates.primary, rates.secondary);
        genome_envelope(&nodes, genome.provenance().clone())
    }

    fn crossover_genomes(
        &self,
        left: &BrainGenomeEnvelope,
        right: &BrainGenomeEnvelope,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeEnvelope, BrainProtocolError> {
        let left_nodes = decode_genome(left)?;
        let right_nodes = decode_genome(right)?;
        let child = crossover_nodes(&left_nodes, &right_nodes, rng);
        let provenance = BrainProvenance {
            parents: [left.provenance().parents[0], right.provenance().parents[0]],
            created_at: Tick(
                left.provenance()
                    .created_at
                    .0
                    .max(right.provenance().created_at.0),
            ),
        };
        genome_envelope(&child, provenance)
    }

    fn initial_state(
        &self,
        genome: &BrainGenomeEnvelope,
        _rng: &mut dyn RandomStream,
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.validate_genome(genome)?;
        state_envelope(&vec![NodeState::default(); BRAIN_SIZE])
    }

    fn offspring_state_policy(&self) -> OffspringStatePolicy {
        // The legacy C++ copy constructor accidentally copied a parent's
        // outputs into offspring. Rust treats acquired recurrent activations as
        // evaluator state, not heredity, and resets them explicitly.
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
        Ok(Box::new(DwraonEvaluator {
            family_id: self.family_id.clone(),
            brain: DwraonBrain {
                nodes: decode_genome(genome)?,
                state: decode_state(state)?,
            },
        }))
    }
}

impl Brain for DwraonBrain {
    fn kind(&self) -> BrainKind {
        Self::KIND
    }

    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        for (idx, input) in inputs.iter().enumerate() {
            if let Some(node) = self.state.get_mut(idx) {
                // Core's sensor contract is [0, 1]. Unlike the C++ method,
                // retain a defensive clamp at this public Rust boundary so an
                // out-of-contract caller cannot inject unbounded recurrent
                // state. It is behaviorally identical for real sensor frames.
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
                        // The legacy evaluator relied on its input and node
                        // invariants here. The explicit clamp makes that same
                        // invariant fail-closed for malformed Rust callers.
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
        mutate_nodes(&mut self.nodes, rng, rate, scale);
        Ok(())
    }

    fn crossover(&self, other: &dyn Brain, rng: &mut dyn RandomStream) -> Option<Box<dyn Brain>> {
        if other.kind() != Self::KIND {
            return None;
        }
        let other = other.as_any().downcast_ref::<Self>()?;

        let nodes = crossover_nodes(&self.nodes, &other.nodes, rng);

        Some(Box::new(Self {
            nodes,
            state: vec![NodeState::default(); BRAIN_SIZE],
        }))
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

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{AgentUid, SmallRngStream};

    fn protocol_genome(nodes: &[NodeParams], provenance: BrainProvenance) -> BrainGenomeEnvelope {
        genome_envelope(nodes, provenance).expect("valid test genome")
    }

    fn contrasting_parents() -> (Vec<NodeParams>, Vec<NodeParams>) {
        let left = NodeParams {
            kind: NodeKind::And,
            damping: 0.25,
            bias: -1.0,
            weights: [0.1, 0.2, 0.3, 0.4],
            sources: [0, 1, 2, 3],
            inverted: [false; CONNECTIONS],
        };
        let right = NodeParams {
            kind: NodeKind::Or,
            damping: 0.75,
            bias: 1.0,
            weights: [1.1, 1.2, 1.3, 1.4],
            sources: [100, 101, 102, 103],
            inverted: [true; CONNECTIONS],
        };
        (vec![left; BRAIN_SIZE], vec![right; BRAIN_SIZE])
    }

    #[test]
    fn random_brain_builds_expected_layout() {
        let mut rng = SmallRngStream::seed_from_u64(0x5A5A5A5A);
        let brain = DwraonBrain::random(&mut rng);
        assert_eq!(brain.nodes.len(), BRAIN_SIZE);
        assert_eq!(brain.state.len(), BRAIN_SIZE);
        assert!(
            brain.nodes[..BRAIN_SIZE / 2]
                .iter()
                .flat_map(|node| node.sources)
                .all(|source| source < INPUT_SIZE),
            "the legacy constructor overwrites every first-half source with a sensor index"
        );
        assert!(
            brain.nodes[BRAIN_SIZE / 2..]
                .iter()
                .flat_map(|node| node.sources)
                .any(|source| source >= INPUT_SIZE),
            "the recurrent half retains full-brain sources"
        );
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

    #[test]
    fn protocol_genome_codec_is_fixed_width_deterministic_and_source_exact() {
        let family = DwraonFamilyAdapter::default();
        let mut first_rng = SmallRngStream::seed_from_u64(0x0A11_CE55);
        let first = family
            .random_genome(&mut first_rng)
            .expect("first deterministic genome");
        let mut second_rng = SmallRngStream::seed_from_u64(0x0A11_CE55);
        let second = family
            .random_genome(&mut second_rng)
            .expect("second deterministic genome");
        assert_eq!(first, second);
        assert_eq!(first.payload().len(), GENOME_PAYLOAD_BYTES);
        assert_eq!(
            &first.payload()[..GENOME_HEADER_BYTES],
            &[b'D', b'W', b'G', b'N', 200, 0, 4, 25]
        );

        let nodes = decode_genome(&first).expect("decode exact genome");
        assert_eq!(nodes[0].sources, [4, 17, 13, 16]);
        assert_eq!(nodes[99].sources, [18, 18, 22, 1]);
        assert_eq!(nodes[100].sources, [7, 182, 58, 152]);
        assert!(
            nodes[..BRAIN_SIZE / 2]
                .iter()
                .flat_map(|node| node.sources)
                .all(|source| source < INPUT_SIZE)
        );
        assert_eq!(
            encode_genome_payload(&nodes).expect("re-encode exact genome"),
            first.payload(),
            "fixed-width source fields and float bits must round-trip byte-for-byte"
        );
    }

    #[test]
    fn protocol_rejects_malformed_genomes_states_and_mutation_rates() {
        let family = DwraonFamilyAdapter::default();
        let mut rng = SmallRngStream::seed_from_u64(17);
        let genome = family.random_genome(&mut rng).expect("valid genome");

        let mut invalid_source_payload = genome.payload().to_vec();
        let first_source = GENOME_HEADER_BYTES + 1 + 4 + 4 + 4;
        invalid_source_payload[first_source..first_source + 2]
            .copy_from_slice(&(BRAIN_SIZE as u16).to_le_bytes());
        let invalid_source = BrainGenomeEnvelope::new(
            DWRAON_FAMILY_ID.clone(),
            GENOME_SCHEMA_VERSION,
            GENOME_CODEC_VERSION,
            invalid_source_payload,
            BrainProvenance::default(),
        )
        .expect("generic envelope accepts family-owned malformed bytes");
        assert!(matches!(
            family.validate_genome(&invalid_source),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::Genome,
                ..
            })
        ));
        assert!(matches!(
            family.mutate_genome(&invalid_source, MutationRates::default(), &mut rng),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::Genome,
                ..
            })
        ));
        assert!(matches!(
            family.mutate_genome(
                &genome,
                MutationRates {
                    primary: f32::NAN,
                    secondary: 0.1,
                },
                &mut rng,
            ),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::Genome,
                ..
            })
        ));

        let state = family
            .initial_state(&genome, &mut rng)
            .expect("valid initial state");
        let mut nonfinite_state_payload = state.payload().to_vec();
        nonfinite_state_payload[STATE_HEADER_BYTES..STATE_HEADER_BYTES + 4]
            .copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        let nonfinite_state = BrainEvaluatorStateEnvelope::new(
            DWRAON_FAMILY_ID.clone(),
            STATE_SCHEMA_VERSION,
            STATE_CODEC_VERSION,
            nonfinite_state_payload,
        )
        .expect("generic envelope accepts family-owned malformed state");
        assert!(matches!(
            family.validate_evaluator_state(&nonfinite_state),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::EvaluatorState,
                ..
            })
        ));
        assert!(matches!(
            family.evaluator(&genome, &nonfinite_state),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::EvaluatorState,
                ..
            })
        ));

        let mut evaluator = family.evaluator(&genome, &state).expect("valid evaluator");
        let before = family
            .checkpoint_evaluator(evaluator.as_ref())
            .expect("checkpoint before rejected input");
        let mut nonfinite_sensors = [0.0; INPUT_SIZE];
        nonfinite_sensors[3] = f32::INFINITY;
        assert!(matches!(
            evaluator.evaluate(&nonfinite_sensors),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::EvaluatorState,
                ..
            })
        ));
        assert_eq!(
            family
                .checkpoint_evaluator(evaluator.as_ref())
                .expect("checkpoint after rejected input"),
            before,
            "a rejected sensor frame must not partially advance recurrent state"
        );
    }

    #[test]
    fn protocol_mutation_is_deterministic_changes_genome_and_stays_valid() {
        let family = DwraonFamilyAdapter::default();
        let mut genome_rng = SmallRngStream::seed_from_u64(44);
        let genome = family.random_genome(&mut genome_rng).expect("base genome");
        let rates = MutationRates {
            primary: 0.2,
            secondary: 0.05,
        };
        let mut first_rng = SmallRngStream::seed_from_u64(99);
        let first = family
            .mutate_genome(&genome, rates, &mut first_rng)
            .expect("first mutation");
        let mut second_rng = SmallRngStream::seed_from_u64(99);
        let second = family
            .mutate_genome(&genome, rates, &mut second_rng)
            .expect("second mutation");
        assert_eq!(first, second);
        assert_ne!(first.payload(), genome.payload());
        family
            .validate_genome(&first)
            .expect("mutation must return a valid offspring genome");
        assert_eq!(first.provenance(), genome.provenance());
    }

    #[test]
    fn protocol_crossover_matches_legacy_field_order_not_whole_nodes() {
        let family = DwraonFamilyAdapter::default();
        let (left_nodes, right_nodes) = contrasting_parents();
        let left = protocol_genome(
            &left_nodes,
            BrainProvenance {
                parents: [Some(AgentUid(11)), None],
                created_at: Tick(4),
            },
        );
        let right = protocol_genome(
            &right_nodes,
            BrainProvenance {
                parents: [Some(AgentUid(22)), None],
                created_at: Tick(9),
            },
        );
        let mut crossover_rng = SmallRngStream::seed_from_u64(0xC205_50E2);
        let child = family
            .crossover_genomes(&left, &right, &mut crossover_rng)
            .expect("field-wise child");
        let child_nodes = decode_genome(&child).expect("decode child");

        let mut oracle_rng = SmallRngStream::seed_from_u64(0xC205_50E2);
        let mut expected = left_nodes.clone();
        let mut selected_left = false;
        let mut selected_right = false;
        for (node, (left_node, right_node)) in
            expected.iter_mut().zip(left_nodes.iter().zip(&right_nodes))
        {
            let choose_left = oracle_rng.random::<f32>() < 0.5;
            node.bias = if choose_left {
                selected_left = true;
                left_node.bias
            } else {
                selected_right = true;
                right_node.bias
            };
            let choose_left = oracle_rng.random::<f32>() < 0.5;
            node.damping = if choose_left {
                selected_left = true;
                left_node.damping
            } else {
                selected_right = true;
                right_node.damping
            };
            let choose_left = oracle_rng.random::<f32>() < 0.5;
            node.kind = if choose_left {
                selected_left = true;
                left_node.kind
            } else {
                selected_right = true;
                right_node.kind
            };
            for connection in 0..CONNECTIONS {
                let choose_left = oracle_rng.random::<f32>() < 0.5;
                node.sources[connection] = if choose_left {
                    selected_left = true;
                    left_node.sources[connection]
                } else {
                    selected_right = true;
                    right_node.sources[connection]
                };
                let choose_left = oracle_rng.random::<f32>() < 0.5;
                node.inverted[connection] = if choose_left {
                    selected_left = true;
                    left_node.inverted[connection]
                } else {
                    selected_right = true;
                    right_node.inverted[connection]
                };
                let choose_left = oracle_rng.random::<f32>() < 0.5;
                node.weights[connection] = if choose_left {
                    selected_left = true;
                    left_node.weights[connection]
                } else {
                    selected_right = true;
                    right_node.weights[connection]
                };
            }
        }
        assert!(selected_left && selected_right);
        assert_eq!(child_nodes, expected);
        assert!(child_nodes.iter().any(|node| {
            node.bias == left_nodes[0].bias && node.kind == right_nodes[0].kind
                || node.bias == right_nodes[0].bias && node.kind == left_nodes[0].kind
        }));
        assert_eq!(
            child.provenance(),
            &BrainProvenance {
                parents: [Some(AgentUid(11)), Some(AgentUid(22))],
                created_at: Tick(9),
            }
        );
        family
            .validate_genome(&child)
            .expect("field-wise child must remain valid");
    }

    #[test]
    fn evaluator_checkpoint_restore_has_identical_next_output_and_state() {
        let family = DwraonFamilyAdapter::default();
        let mut rng = SmallRngStream::seed_from_u64(8080);
        let genome = family.random_genome(&mut rng).expect("genome");
        let initial = family.initial_state(&genome, &mut rng).expect("state");
        assert_eq!(initial.payload().len(), STATE_PAYLOAD_BYTES);
        assert_eq!(&initial.payload()[..6], &[b'D', b'W', b'S', b'T', 200, 0]);
        let mut original = family
            .evaluator(&genome, &initial)
            .expect("original evaluator");
        let mut warmup = [0.0; INPUT_SIZE];
        for (index, sensor) in warmup.iter_mut().enumerate() {
            *sensor = index as f32 / INPUT_SIZE as f32;
        }
        for _ in 0..5 {
            original.evaluate(&warmup).expect("warmup evaluation");
        }
        let checkpoint = family
            .checkpoint_evaluator(original.as_ref())
            .expect("validated checkpoint");
        let mut restored = family
            .evaluator(&genome, &checkpoint)
            .expect("restored evaluator");
        let mut next = warmup;
        next.reverse();
        assert_eq!(
            original.evaluate(&next).expect("original continuation"),
            restored.evaluate(&next).expect("restored continuation")
        );
        assert_eq!(
            family
                .checkpoint_evaluator(original.as_ref())
                .expect("original next checkpoint"),
            family
                .checkpoint_evaluator(restored.as_ref())
                .expect("restored next checkpoint")
        );
        assert!(
            restored
                .inspect(BrainInspection::Activations)
                .expect("bounded inspection")
                .is_some()
        );
    }

    #[test]
    fn offspring_recurrent_state_explicitly_resets_instead_of_legacy_copying() {
        let family = DwraonFamilyAdapter::default();
        assert_eq!(family.offspring_state_policy(), OffspringStatePolicy::Reset);
        let mut rng = SmallRngStream::seed_from_u64(303);
        let genome = family.random_genome(&mut rng).expect("genome");
        let initial = family.initial_state(&genome, &mut rng).expect("initial");
        let mut parent = family
            .evaluator(&genome, &initial)
            .expect("parent evaluator");
        parent
            .evaluate(&[0.75; INPUT_SIZE])
            .expect("advance parent state");
        let parent_state = family
            .checkpoint_evaluator(parent.as_ref())
            .expect("parent checkpoint");
        let child_state = family
            .offspring_state(&genome, &[&parent_state], &mut rng)
            .expect("reset child state");
        assert_ne!(child_state, parent_state);
        assert!(
            decode_state(&child_state)
                .expect("decode reset child")
                .iter()
                .all(|node| node.output == 0.0)
        );

        let malformed_child = BrainGenomeEnvelope::new(
            DWRAON_FAMILY_ID.clone(),
            GENOME_SCHEMA_VERSION,
            GENOME_CODEC_VERSION,
            vec![0; GENOME_PAYLOAD_BYTES - 1],
            BrainProvenance::default(),
        )
        .expect("generic envelope accepts malformed family payload");
        assert!(
            family
                .offspring_state(&malformed_child, &[&parent_state], &mut rng)
                .is_err()
        );
    }
}
