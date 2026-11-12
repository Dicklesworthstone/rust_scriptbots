//! Versioned, bounded `NeuroFlow` heredity and continuation.
//!
//! The genome stores the configured architecture and exact f64 weights. Evaluator state
//! stores the four per-neuron working arrays, the network error and the input cache, bound
//! to both the exact genome and configured adapter. Offspring reset that working state.
//! The legacy runner remains available through `NeuroflowBrain::register`.

use super::{
    FeedForwardSeed, INPUT_SIZE, LayerSeed, NeuroflowActivation, NeuroflowBrain,
    NeuroflowBrainConfig, NeuroflowBrainError, OUTPUT_SIZE,
};
use rand::Rng;
use scriptbots_brain::Brain;
use scriptbots_core::{
    BrainAdapterIdentityV1, BrainEnvelopeKind, BrainEvaluator, BrainEvaluatorStateEnvelope,
    BrainFamilyCodec, BrainFamilyId, BrainGenomeEnvelope, BrainGenomeMaterial,
    BrainHeredityCapabilityV1, BrainHeredityExclusionV1, BrainInspection, BrainInspectionError,
    BrainInspectionSnapshot, BrainProtocolError, MAX_BRAIN_EVALUATOR_STATE_PAYLOAD_BYTES,
    MAX_BRAIN_GENOME_PAYLOAD_BYTES, MutationRates, OffspringStatePolicy, RandomStream,
};

const FAMILY_ID: &str = "neuroflow-f64-v1";
const SEMANTIC_VERSION: u32 = 1;
const SEMANTIC_DESCRIPTOR: &[u8] = b"scriptbots.neuroflow-f64.adapter.v1:neuroflow-0.2.0;legacy-constructor-and-inference;f64-le;reset-offspring;per-weight-uniform-crossover;legacy-gaussian-mutation;transactional-evaluation;configured-founder";
const GENOME_SCHEMA_VERSION: u32 = 1;
const GENOME_CODEC_VERSION: u16 = 1;
const STATE_SCHEMA_VERSION: u32 = 1;
const STATE_CODEC_VERSION: u16 = 1;
const GENOME_MAGIC: [u8; 4] = *b"NFG1";
const STATE_MAGIC: [u8; 4] = *b"NFS1";
const GENOME_FIXED_HEADER_BYTES: usize = 32;
const DIGEST_BYTES: usize = 32;
const STATE_HEADER_BYTES: usize = STATE_MAGIC.len() + 2 * DIGEST_BYTES;
const SCALAR_BYTES: usize = std::mem::size_of::<f64>();

/// Checkpointable `NeuroFlow` family with exact f64 heredity.
///
/// The configured constructor is part of the executable identity. Genomes from a different
/// architecture, activation, learning rate or momentum are refused, including otherwise valid
/// `NeuroFlow` genomes. Existing envelope limits impose tighter architecture bounds than the
/// legacy runner; construction checks these before allocating a network or drawing randomness.
///
/// Typed genome loci remain unavailable: the shared scalar-locus vocabulary stores f32 and
/// cannot represent this family's f64 genes without losing heritable information.
#[derive(Debug, Clone)]
pub struct NeuroflowFamilyAdapter {
    family_id: BrainFamilyId,
    config: NeuroflowBrainConfig,
    identity: BrainAdapterIdentityV1,
    genome_header: Vec<u8>,
    genome_bytes: usize,
    state_bytes: usize,
    inspection_source_scalars: usize,
}

impl NeuroflowFamilyAdapter {
    /// Validate the configured architecture and both envelope budgets before constructing a family.
    ///
    /// # Panics
    /// Panics only if the built-in family identifier ceases to be a valid protocol identifier.
    #[allow(
        clippy::too_many_lines,
        reason = "constructor keeps checked architecture budgets and exact configured header identity together"
    )]
    pub fn new(config: NeuroflowBrainConfig) -> Result<Self, NeuroflowBrainError> {
        config.validate()?;
        let overflow = || NeuroflowBrainError::ArchitectureOverflow {
            context: "versioned NeuroFlow payload size",
        };
        let mut previous = INPUT_SIZE;
        let mut weights = 0_usize;
        let mut neurons = 0_usize;
        for outputs in config
            .hidden_layers
            .iter()
            .copied()
            .chain(std::iter::once(OUTPUT_SIZE))
        {
            weights = weights
                .checked_add(
                    outputs
                        .checked_mul(previous.checked_add(1).ok_or_else(overflow)?)
                        .ok_or_else(overflow)?,
                )
                .ok_or_else(overflow)?;
            neurons = neurons.checked_add(outputs).ok_or_else(overflow)?;
            previous = outputs;
        }
        let header_bytes = GENOME_FIXED_HEADER_BYTES
            .checked_add(
                config
                    .hidden_layers
                    .len()
                    .checked_mul(4)
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
        let genome_bytes = weights
            .checked_mul(SCALAR_BYTES)
            .and_then(|bytes| bytes.checked_add(header_bytes))
            .ok_or_else(overflow)?;
        let state_bytes = neurons
            .checked_mul(4)
            .and_then(|values| values.checked_add(INPUT_SIZE + 1))
            .and_then(|values| values.checked_mul(SCALAR_BYTES))
            .and_then(|bytes| bytes.checked_add(STATE_HEADER_BYTES))
            .ok_or_else(overflow)?;
        for (field, value, maximum, requirement) in [
            (
                "genome_payload_bytes",
                genome_bytes,
                MAX_BRAIN_GENOME_PAYLOAD_BYTES,
                "versioned NeuroFlow genomes must fit the 1 MiB envelope",
            ),
            (
                "evaluator_state_payload_bytes",
                state_bytes,
                MAX_BRAIN_EVALUATOR_STATE_PAYLOAD_BYTES,
                "versioned NeuroFlow state must fit the 256 KiB envelope",
            ),
        ] {
            if value > maximum {
                return Err(NeuroflowBrainError::InvalidDimension {
                    field: field.to_owned(),
                    value,
                    requirement,
                });
            }
        }

        let mut genome_header = Vec::with_capacity(header_bytes);
        genome_header.extend_from_slice(&GENOME_MAGIC);
        genome_header.extend_from_slice(
            &u16::try_from(INPUT_SIZE)
                .map_err(|_| overflow())?
                .to_le_bytes(),
        );
        genome_header.extend_from_slice(
            &u16::try_from(OUTPUT_SIZE)
                .map_err(|_| overflow())?
                .to_le_bytes(),
        );
        genome_header.extend_from_slice(
            &u16::try_from(config.hidden_layers.len())
                .map_err(|_| overflow())?
                .to_le_bytes(),
        );
        genome_header.push(match config.activation {
            NeuroflowActivation::Tanh => 0,
            NeuroflowActivation::Sigmoid => 1,
            NeuroflowActivation::Relu => 2,
        });
        genome_header.push(0); // Reserved; an alternative representation is not canonical.
        genome_header.extend_from_slice(&config.learning_rate.to_bits().to_le_bytes());
        genome_header.extend_from_slice(&config.momentum.to_bits().to_le_bytes());
        genome_header.extend_from_slice(
            &u32::try_from(weights)
                .map_err(|_| overflow())?
                .to_le_bytes(),
        );
        for &width in &config.hidden_layers {
            genome_header
                .extend_from_slice(&u32::try_from(width).map_err(|_| overflow())?.to_le_bytes());
        }
        debug_assert_eq!(genome_header.len(), header_bytes);
        let family_id = BrainFamilyId::new(FAMILY_ID)
            .expect("the built-in NeuroFlow family identifier is canonical");
        let mut semantic_material = SEMANTIC_DESCRIPTOR.to_vec();
        semantic_material.extend_from_slice(&genome_header);
        let identity = BrainAdapterIdentityV1::from_semantic_descriptor(
            &family_id,
            SEMANTIC_VERSION,
            &semantic_material,
        );
        let inspection_source_scalars = config.inspection_source_scalars()?;
        Ok(Self {
            family_id,
            config,
            identity,
            genome_header,
            genome_bytes,
            state_bytes,
            inspection_source_scalars,
        })
    }

    /// Count the actual inspection source scalars in a validated founding genome.
    ///
    /// This structural count includes bias-inclusive weights, four working arrays per layer,
    /// and learning rate, momentum and error. It performs no evaluation, allocation of network
    /// state, or random draws and is available before the first tick.
    pub fn genome_source_scalars(
        &self,
        genome: &BrainGenomeEnvelope,
    ) -> Result<usize, BrainProtocolError> {
        self.validate_genome(genome)?;
        Ok(self.inspection_source_scalars)
    }

    fn invalid(&self, kind: BrainEnvelopeKind, detail: impl Into<String>) -> BrainProtocolError {
        BrainProtocolError::InvalidPayload {
            kind,
            family_id: self.family_id.clone(),
            detail: detail.into(),
        }
    }

    fn finite_scalars(
        &self,
        bytes: &[u8],
        kind: BrainEnvelopeKind,
    ) -> Result<(), BrainProtocolError> {
        if !bytes.len().is_multiple_of(SCALAR_BYTES) {
            return Err(self.invalid(kind, "truncated f64 scalar"));
        }
        for (index, chunk) in bytes.as_chunks::<SCALAR_BYTES>().0.iter().enumerate() {
            let value = f64::from_le_bytes(*chunk);
            if !value.is_finite() {
                return Err(self.invalid(kind, format!("scalar {index} is not finite")));
            }
        }
        Ok(())
    }

    fn validate_genome_payload(&self, bytes: &[u8]) -> Result<(), BrainProtocolError> {
        let kind = BrainEnvelopeKind::Genome;
        if bytes.len() != self.genome_bytes {
            return Err(self.invalid(
                kind,
                format!(
                    "expected {} bytes, found {}",
                    self.genome_bytes,
                    bytes.len()
                ),
            ));
        }
        if !bytes.starts_with(&self.genome_header) {
            return Err(self.invalid(
                kind,
                "genome header differs from the configured architecture or parameters",
            ));
        }
        self.finite_scalars(&bytes[self.genome_header.len()..], kind)
    }

    fn validate_state_payload(&self, bytes: &[u8]) -> Result<(), BrainProtocolError> {
        let kind = BrainEnvelopeKind::EvaluatorState;
        if bytes.len() != self.state_bytes {
            return Err(self.invalid(
                kind,
                format!("expected {} bytes, found {}", self.state_bytes, bytes.len()),
            ));
        }
        if !bytes.starts_with(&STATE_MAGIC)
            || &bytes[STATE_MAGIC.len() + DIGEST_BYTES..STATE_HEADER_BYTES]
                != self.identity.digest().as_slice()
        {
            return Err(self.invalid(kind, "state header differs from the configured adapter"));
        }
        self.finite_scalars(&bytes[STATE_HEADER_BYTES..], kind)
    }

    /// Decode only after the complete payload's shape, length and values have been checked.
    fn genome_snapshot(
        &self,
        genome: &BrainGenomeEnvelope,
    ) -> Result<FeedForwardSeed, BrainProtocolError> {
        self.validate_genome(genome)?;
        let mut scalars = genome.payload()[self.genome_header.len()..]
            .as_chunks::<SCALAR_BYTES>()
            .0
            .iter();
        let mut previous = INPUT_SIZE;
        let mut layers = Vec::with_capacity(self.config.hidden_layers.len() + 1);
        for outputs in self
            .config
            .hidden_layers
            .iter()
            .copied()
            .chain(std::iter::once(OUTPUT_SIZE))
        {
            let mut w = Vec::with_capacity(outputs);
            for _ in 0..outputs {
                w.push((0..=previous).map(|_| next_scalar(&mut scalars)).collect());
            }
            layers.push(LayerSeed {
                v: vec![0.0; outputs],
                y: vec![0.0; outputs],
                delta: vec![0.0; outputs],
                prev_delta: vec![0.0; outputs],
                w,
            });
            previous = outputs;
        }
        debug_assert_eq!(scalars.as_slice(), [] as [[u8; SCALAR_BYTES]; 0]);
        Ok(FeedForwardSeed {
            layers,
            learn_rate: self.config.learning_rate,
            momentum: self.config.momentum,
            error: 0.0,
            act_type: self.config.activation.to_type(),
        })
    }

    fn genome_material(
        &self,
        snapshot: &FeedForwardSeed,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        NeuroflowBrain::validate_network_snapshot(snapshot, &self.config)
            .map_err(|error| self.invalid(BrainEnvelopeKind::Genome, error.to_string()))?;
        let mut payload = Vec::with_capacity(self.genome_bytes);
        payload.extend_from_slice(&self.genome_header);
        for layer in &snapshot.layers {
            for row in &layer.w {
                for weight in row {
                    payload.extend_from_slice(&weight.to_bits().to_le_bytes());
                }
            }
        }
        self.validate_genome_payload(&payload)?;
        BrainGenomeMaterial::new(GENOME_SCHEMA_VERSION, GENOME_CODEC_VERSION, payload)
    }

    fn copy_snapshot(&self, snapshot: &FeedForwardSeed) -> FeedForwardSeed {
        FeedForwardSeed {
            layers: snapshot
                .layers
                .iter()
                .map(|layer| LayerSeed {
                    v: layer.v.clone(),
                    y: layer.y.clone(),
                    delta: layer.delta.clone(),
                    prev_delta: layer.prev_delta.clone(),
                    w: layer.w.clone(),
                })
                .collect(),
            learn_rate: snapshot.learn_rate,
            momentum: snapshot.momentum,
            error: snapshot.error,
            act_type: self.config.activation.to_type(),
        }
    }

    fn brain_from_snapshot(
        &self,
        snapshot: FeedForwardSeed,
        inputs: Vec<f64>,
    ) -> Result<NeuroflowBrain, BrainProtocolError> {
        let network = NeuroflowBrain::restore_network_snapshot(snapshot, &self.config)
            .map_err(|error| self.invalid(BrainEnvelopeKind::EvaluatorState, error.to_string()))?;
        Ok(NeuroflowBrain {
            network,
            config: self.config.clone(),
            inputs,
            inspection_source_scalars: self.inspection_source_scalars,
            #[cfg(test)]
            inspection_json_conversions: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn state_header(&self, genome_digest: &[u8; DIGEST_BYTES]) -> Vec<u8> {
        let mut payload = Vec::with_capacity(self.state_bytes);
        payload.extend_from_slice(&STATE_MAGIC);
        payload.extend_from_slice(genome_digest);
        payload.extend_from_slice(self.identity.digest());
        payload
    }

    fn encode_state(
        &self,
        genome_digest: &[u8; DIGEST_BYTES],
        brain: &NeuroflowBrain,
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        let snapshot = brain
            .network_snapshot()
            .map_err(|error| self.invalid(BrainEnvelopeKind::EvaluatorState, error.to_string()))?;
        if brain.inputs.len() != INPUT_SIZE || brain.inputs.iter().any(|value| !value.is_finite()) {
            return Err(self.invalid(BrainEnvelopeKind::EvaluatorState, "invalid input cache"));
        }
        let mut payload = self.state_header(genome_digest);
        payload.extend_from_slice(&snapshot.error.to_bits().to_le_bytes());
        for value in &brain.inputs {
            payload.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        for layer in &snapshot.layers {
            for values in [&layer.v, &layer.y, &layer.delta, &layer.prev_delta] {
                for value in values {
                    payload.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
        }
        self.validate_state_payload(&payload)?;
        BrainEvaluatorStateEnvelope::new(
            self.family_id.clone(),
            STATE_SCHEMA_VERSION,
            STATE_CODEC_VERSION,
            payload,
        )
    }

    fn validate_mutation_rates(&self, rates: MutationRates) -> Result<(), BrainProtocolError> {
        if !rates.primary.is_finite()
            || rates.primary < 0.0
            || !rates.secondary.is_finite()
            || rates.secondary < 0.0
        {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                "mutation threshold and scale must be finite and nonnegative",
            ));
        }
        Ok(())
    }
}

impl Default for NeuroflowFamilyAdapter {
    fn default() -> Self {
        Self::new(NeuroflowBrainConfig::default())
            .expect("the default NeuroFlow architecture fits the versioned envelope budgets")
    }
}

fn next_scalar(chunks: &mut std::slice::Iter<'_, [u8; SCALAR_BYTES]>) -> f64 {
    f64::from_le_bytes(
        *chunks
            .next()
            .expect("complete payload was validated before decoding"),
    )
}

struct NeuroflowEvaluator {
    family: NeuroflowFamilyAdapter,
    genome_digest: [u8; DIGEST_BYTES],
    brain: NeuroflowBrain,
    // Execution scratch only. Its v/y and input cache are overwritten completely by calc;
    // immutable weights and the unused training state match the accepted brain exactly.
    standby: NeuroflowBrain,
}

impl BrainEvaluator for NeuroflowEvaluator {
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
                "sensor input is not finite",
            ));
        }
        // NeuroFlow 0.2.0 calc overwrites every v/y entry and never trains. A failed
        // standby therefore needs no reconstruction before retry; all partially written
        // values are overwritten by the next call. Only a validated result becomes visible.
        // This keeps the actual engine and its operation order while avoiding per-evaluation
        // JSON reconstruction. State validation/checkpointing still have serialization cost.
        let outputs = self.standby.tick(sensors);
        self.standby.network_snapshot().map_err(|error| {
            self.family
                .invalid(BrainEnvelopeKind::EvaluatorState, error.to_string())
        })?;
        if outputs.iter().any(|value| !value.is_finite()) {
            return Err(self.family.invalid(
                BrainEnvelopeKind::EvaluatorState,
                "evaluation produced a non-finite output",
            ));
        }
        std::mem::swap(&mut self.brain, &mut self.standby);
        Ok(outputs)
    }

    fn inspect(
        &self,
        request: BrainInspection,
    ) -> Result<Option<BrainInspectionSnapshot>, BrainInspectionError> {
        self.brain.inspect(request)
    }

    fn checkpoint_state(&self) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.family.encode_state(&self.genome_digest, &self.brain)
    }
}

impl BrainFamilyCodec for NeuroflowFamilyAdapter {
    fn family_id(&self) -> &BrainFamilyId {
        &self.family_id
    }

    fn adapter_identity(&self) -> BrainAdapterIdentityV1 {
        self.identity
    }

    fn heredity_capability(&self) -> BrainHeredityCapabilityV1 {
        BrainHeredityCapabilityV1::excluded(BrainHeredityExclusionV1::NoCanonicalLocusSchema)
    }

    fn random_genome_material(
        &self,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        let brain = NeuroflowBrain::new(self.config.clone(), rng)
            .map_err(|error| self.invalid(BrainEnvelopeKind::Genome, error.to_string()))?;
        let snapshot = brain
            .network_snapshot()
            .map_err(|error| self.invalid(BrainEnvelopeKind::Genome, error.to_string()))?;
        self.genome_material(&snapshot)
    }

    fn validate_genome(&self, genome: &BrainGenomeEnvelope) -> Result<(), BrainProtocolError> {
        genome.require_protocol(&self.family_id, GENOME_SCHEMA_VERSION, GENOME_CODEC_VERSION)?;
        self.validate_genome_payload(genome.payload())
    }

    fn validate_evaluator_state(
        &self,
        state: &BrainEvaluatorStateEnvelope,
    ) -> Result<(), BrainProtocolError> {
        state.require_protocol(&self.family_id, STATE_SCHEMA_VERSION, STATE_CODEC_VERSION)?;
        self.validate_state_payload(state.payload())
    }

    fn mutate_genome_material(
        &self,
        genome: &BrainGenomeEnvelope,
        rates: MutationRates,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        self.validate_mutation_rates(rates)?;
        let snapshot = self.genome_snapshot(genome)?;
        let mut brain = self.brain_from_snapshot(snapshot, vec![0.0; INPUT_SIZE])?;
        brain
            .mutate(rng, rates.primary, rates.secondary)
            .map_err(|error| self.invalid(BrainEnvelopeKind::Genome, error.to_string()))?;
        let snapshot = brain
            .network_snapshot()
            .map_err(|error| self.invalid(BrainEnvelopeKind::Genome, error.to_string()))?;
        self.genome_material(&snapshot)
    }

    fn crossover_genomes_material(
        &self,
        left: &BrainGenomeEnvelope,
        right: &BrainGenomeEnvelope,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        // Refuse either parent before allocating decoded matrices or consuming crossover RNG.
        self.validate_genome(left)?;
        self.validate_genome(right)?;
        let mut child = self.genome_snapshot(left)?;
        let right = self.genome_snapshot(right)?;
        for (child_layer, right_layer) in child.layers.iter_mut().zip(right.layers) {
            for (child_row, right_row) in child_layer.w.iter_mut().zip(right_layer.w) {
                for (child_weight, right_weight) in child_row.iter_mut().zip(right_row) {
                    if rng.random::<f32>() >= 0.5 {
                        *child_weight = right_weight;
                    }
                }
            }
        }
        self.genome_material(&child)
    }

    fn initial_state(
        &self,
        genome: &BrainGenomeEnvelope,
        _rng: &mut dyn RandomStream,
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.validate_genome(genome)?;
        let mut payload = self.state_header(genome.material_hash().as_bytes());
        payload.resize(self.state_bytes, 0);
        BrainEvaluatorStateEnvelope::new(
            self.family_id.clone(),
            STATE_SCHEMA_VERSION,
            STATE_CODEC_VERSION,
            payload,
        )
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
        self.validate_genome(genome)?;
        self.validate_evaluator_state(state)?;
        let genome_digest = *genome.material_hash().as_bytes();
        if &state.payload()[STATE_MAGIC.len()..STATE_MAGIC.len() + DIGEST_BYTES]
            != genome_digest.as_slice()
        {
            return Err(self.invalid(
                BrainEnvelopeKind::EvaluatorState,
                "state belongs to a different genome",
            ));
        }
        let mut snapshot = self.genome_snapshot(genome)?;
        let mut scalars = state.payload()[STATE_HEADER_BYTES..]
            .as_chunks::<SCALAR_BYTES>()
            .0
            .iter();
        snapshot.error = next_scalar(&mut scalars);
        let inputs: Vec<f64> = (0..INPUT_SIZE).map(|_| next_scalar(&mut scalars)).collect();
        for layer in &mut snapshot.layers {
            for values in [
                &mut layer.v,
                &mut layer.y,
                &mut layer.delta,
                &mut layer.prev_delta,
            ] {
                for value in values {
                    *value = next_scalar(&mut scalars);
                }
            }
        }
        debug_assert_eq!(scalars.as_slice(), [] as [[u8; SCALAR_BYTES]; 0]);
        let standby = self.brain_from_snapshot(self.copy_snapshot(&snapshot), inputs.clone())?;
        let brain = self.brain_from_snapshot(snapshot, inputs)?;
        Ok(Box::new(NeuroflowEvaluator {
            family: self.clone(),
            genome_digest,
            brain,
            standby,
        }))
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::too_many_lines,
        reason = "complete heredity, malformed-input and real-world fixtures retain their positive and negative controls"
    )]
    use super::*;
    use scriptbots_core::{
        AgentData, AgentUid, BrainFamilyAdapter, BrainGenomeDerivation, BrainInspectionLimits,
        BrainProvenance, BrainRegistry, Position, ScriptBotsConfig, SmallRngStream, Tick,
        WorldState,
    };

    fn small_config() -> NeuroflowBrainConfig {
        NeuroflowBrainConfig {
            hidden_layers: vec![4, 3],
            ..NeuroflowBrainConfig::default()
        }
    }

    fn genome_with_payload(
        family: &NeuroflowFamilyAdapter,
        payload: Vec<u8>,
    ) -> BrainGenomeEnvelope {
        BrainGenomeEnvelope::new(
            family.family_id.clone(),
            GENOME_SCHEMA_VERSION,
            GENOME_CODEC_VERSION,
            payload,
            BrainProvenance::default(),
        )
        .unwrap()
    }

    fn state_with_payload(
        family: &NeuroflowFamilyAdapter,
        payload: Vec<u8>,
    ) -> BrainEvaluatorStateEnvelope {
        BrainEvaluatorStateEnvelope::new(
            family.family_id.clone(),
            STATE_SCHEMA_VERSION,
            STATE_CODEC_VERSION,
            payload,
        )
        .unwrap()
    }

    fn sensors(tick: u16) -> [f32; INPUT_SIZE] {
        std::array::from_fn(|index| f32::from((u16::try_from(index).unwrap() + tick) % 17) / 16.0)
    }

    #[test]
    fn configured_identity_and_payload_budgets_are_checked_at_construction() {
        let baseline = NeuroflowFamilyAdapter::default();
        assert_eq!(
            baseline.adapter_identity(),
            NeuroflowFamilyAdapter::default().adapter_identity()
        );
        for config in [
            NeuroflowBrainConfig {
                hidden_layers: vec![4, 3],
                ..NeuroflowBrainConfig::default()
            },
            NeuroflowBrainConfig {
                activation: NeuroflowActivation::Sigmoid,
                ..NeuroflowBrainConfig::default()
            },
            NeuroflowBrainConfig {
                learning_rate: 0.02,
                ..NeuroflowBrainConfig::default()
            },
            NeuroflowBrainConfig {
                momentum: 0.1,
                ..NeuroflowBrainConfig::default()
            },
        ] {
            assert_ne!(
                baseline.adapter_identity(),
                NeuroflowFamilyAdapter::new(config)
                    .unwrap()
                    .adapter_identity()
            );
        }
        assert!(matches!(
            NeuroflowFamilyAdapter::new(NeuroflowBrainConfig {
                hidden_layers: vec![4096],
                ..small_config()
            }),
            Err(NeuroflowBrainError::InvalidDimension { field, .. })
                if field == "genome_payload_bytes"
        ));
        assert!(matches!(
            NeuroflowFamilyAdapter::new(NeuroflowBrainConfig {
                hidden_layers: [1000, 1].repeat(9),
                ..small_config()
            }),
            Err(NeuroflowBrainError::InvalidDimension { field, .. })
                if field == "evaluator_state_payload_bytes"
        ));
        assert!(
            NeuroflowFamilyAdapter::new(NeuroflowBrainConfig {
                hidden_layers: vec![usize::MAX],
                ..small_config()
            })
            .is_err()
        );
        assert!(
            NeuroflowFamilyAdapter::new(NeuroflowBrainConfig {
                momentum: f64::NAN,
                ..small_config()
            })
            .is_err()
        );
        assert_eq!(
            baseline.heredity_capability(),
            BrainHeredityCapabilityV1::excluded(BrainHeredityExclusionV1::NoCanonicalLocusSchema)
        );
    }

    #[test]
    fn versioned_inference_preserves_legacy_constructor_draws_and_outputs() {
        for activation in [
            NeuroflowActivation::Tanh,
            NeuroflowActivation::Sigmoid,
            NeuroflowActivation::Relu,
        ] {
            for hidden_layers in [Vec::new(), vec![4, 3]] {
                let config = NeuroflowBrainConfig {
                    hidden_layers,
                    activation,
                    ..small_config()
                };
                let family = NeuroflowFamilyAdapter::new(config.clone()).unwrap();
                let mut legacy_rng = SmallRngStream::seed_from_u64(0x4E46_0001);
                let mut protocol_rng = SmallRngStream::seed_from_u64(0x4E46_0001);
                let mut legacy = NeuroflowBrain::new(config.clone(), &mut legacy_rng).unwrap();
                let genome = family
                    .random_genome(BrainProvenance::default(), &mut protocol_rng)
                    .unwrap();
                let state = family.initial_state(&genome, &mut protocol_rng).unwrap();
                assert_eq!(legacy_rng.random::<u64>(), protocol_rng.random::<u64>());
                assert_eq!(
                    family.genome_source_scalars(&genome).unwrap(),
                    config.inspection_source_scalars().unwrap()
                );
                let mut evaluator = family.evaluator(&genome, &state).unwrap();
                for tick in 0..12 {
                    assert_eq!(
                        legacy.tick(&sensors(tick)).map(f32::to_bits),
                        evaluator
                            .evaluate(&sensors(tick))
                            .unwrap()
                            .map(f32::to_bits),
                    );
                    let request = BrainInspection::Activations(BrainInspectionLimits::default());
                    assert_eq!(
                        legacy.inspect(request).unwrap(),
                        evaluator.inspect(request).unwrap()
                    );
                    let saved = evaluator.checkpoint_state().unwrap();
                    evaluator = family.evaluator(&genome, &saved).unwrap();
                    assert_eq!(saved, evaluator.checkpoint_state().unwrap());
                }
            }
        }
    }

    #[test]
    fn genome_and_complete_state_preserve_f64_bits_across_reconstruction() {
        let family = NeuroflowFamilyAdapter::new(small_config()).unwrap();
        let mut rng = SmallRngStream::seed_from_u64(0x4E46_0002);
        let source = family
            .random_genome(BrainProvenance::default(), &mut rng)
            .unwrap();
        let precise = [
            -0.0_f64,
            f64::from_bits(1.0_f64.to_bits() + 1),
            f64::from_bits(1),
            -f64::from_bits(1.0_f64.to_bits() - 1),
        ];
        let mut payload = source.payload().to_vec();
        for (chunk, value) in payload[family.genome_header.len()..]
            .chunks_exact_mut(SCALAR_BYTES)
            .zip(precise.iter().cycle())
        {
            chunk.copy_from_slice(&value.to_bits().to_le_bytes());
        }
        let genome = genome_with_payload(&family, payload);
        let snapshot = family.genome_snapshot(&genome).unwrap();
        let brain = family
            .brain_from_snapshot(snapshot, vec![0.0; INPUT_SIZE])
            .unwrap();
        let reconstructed = brain.network_snapshot().unwrap();
        assert_eq!(
            family.genome_material(&reconstructed).unwrap(),
            BrainGenomeMaterial::new(
                GENOME_SCHEMA_VERSION,
                GENOME_CODEC_VERSION,
                genome.payload().to_vec()
            )
            .unwrap(),
        );

        let initial = family.initial_state(&genome, &mut rng).unwrap();
        let mut payload = initial.payload().to_vec();
        for (chunk, value) in payload[STATE_HEADER_BYTES..]
            .chunks_exact_mut(SCALAR_BYTES)
            .zip(precise.iter().cycle())
        {
            chunk.copy_from_slice(&value.to_bits().to_le_bytes());
        }
        let exact = state_with_payload(&family, payload);
        let mut uninterrupted = family.evaluator(&genome, &exact).unwrap();
        assert_eq!(uninterrupted.checkpoint_state().unwrap(), exact);
        let mut restored = family
            .evaluator(&genome, &uninterrupted.checkpoint_state().unwrap())
            .unwrap();
        for tick in 0..8 {
            assert_eq!(
                uninterrupted
                    .evaluate(&sensors(tick))
                    .unwrap()
                    .map(f32::to_bits),
                restored.evaluate(&sensors(tick)).unwrap().map(f32::to_bits),
            );
            assert_eq!(
                uninterrupted.checkpoint_state().unwrap(),
                restored.checkpoint_state().unwrap()
            );
            let request = BrainInspection::Activations(BrainInspectionLimits::default());
            assert_eq!(
                uninterrupted.inspect(request).unwrap(),
                restored.inspect(request).unwrap()
            );
        }
    }

    #[test]
    fn mutation_matches_legacy_draws_and_inherits_weights_while_children_reset_state() {
        let config = small_config();
        let family = NeuroflowFamilyAdapter::new(config.clone()).unwrap();
        let mut legacy_rng = SmallRngStream::seed_from_u64(0x4E46_0003);
        let mut protocol_rng = SmallRngStream::seed_from_u64(0x4E46_0003);
        let legacy = NeuroflowBrain::new(config, &mut legacy_rng).unwrap();
        let parent = family
            .random_genome(BrainProvenance::default(), &mut protocol_rng)
            .unwrap();
        let parent_provenance = BrainProvenance {
            parents: [Some(AgentUid(17)), None],
            parent_genome_hashes: [Some(parent.material_hash()), None],
            created_at: Tick(9),
            derivation: BrainGenomeDerivation::MutationOnly,
        };
        for rates in [
            MutationRates {
                primary: 0.0,
                secondary: 0.5,
            },
            MutationRates {
                primary: 0.35,
                secondary: 0.0,
            },
            MutationRates {
                primary: 0.35,
                secondary: 0.1,
            },
            MutationRates {
                primary: 2.0,
                secondary: 0.1,
            },
        ] {
            let snapshot = legacy.network_snapshot().unwrap();
            let mut legacy_child = family
                .brain_from_snapshot(snapshot, vec![0.0; INPUT_SIZE])
                .unwrap();
            let mut left_rng = SmallRngStream::seed_from_u64(0x4E46_0013);
            let mut right_rng = SmallRngStream::seed_from_u64(0x4E46_0013);
            legacy_child
                .mutate(&mut left_rng, rates.primary, rates.secondary)
                .unwrap();
            let child = family
                .mutate_genome(&parent, rates, parent_provenance.clone(), &mut right_rng)
                .unwrap();
            assert_eq!(left_rng.random::<u64>(), right_rng.random::<u64>());
            assert_eq!(
                family
                    .genome_material(&legacy_child.network_snapshot().unwrap())
                    .unwrap(),
                BrainGenomeMaterial::new(
                    GENOME_SCHEMA_VERSION,
                    GENOME_CODEC_VERSION,
                    child.payload().to_vec()
                )
                .unwrap(),
            );
            assert_eq!(child.provenance(), &parent_provenance);
            if rates.primary > 0.0 && rates.secondary > 0.0 {
                assert_ne!(child.material_hash(), parent.material_hash());
            } else {
                assert_eq!(child.material_hash(), parent.material_hash());
            }
        }
        let clone = family
            .clone_genome(
                &parent,
                BrainProvenance {
                    derivation: BrainGenomeDerivation::Clone,
                    ..parent_provenance
                },
            )
            .unwrap();
        assert_eq!(clone.material_hash(), parent.material_hash());
        let initial = family.initial_state(&parent, &mut protocol_rng).unwrap();
        let mut active = family.evaluator(&parent, &initial).unwrap();
        active.evaluate(&sensors(5)).unwrap();
        let active_state = active.checkpoint_state().unwrap();
        assert_ne!(active_state, initial);
        let reset = family
            .offspring_state(&clone, &[&active_state], &mut protocol_rng)
            .unwrap();
        assert_eq!(family.offspring_state_policy(), OffspringStatePolicy::Reset);
        assert_eq!(reset, initial);
    }

    #[test]
    fn crossover_is_deterministic_and_preserves_exact_parent_weight_choices() {
        let family = NeuroflowFamilyAdapter::new(small_config()).unwrap();
        let mut rng = SmallRngStream::seed_from_u64(0x4E46_0004);
        let left = family
            .random_genome(BrainProvenance::default(), &mut rng)
            .unwrap();
        let right = family
            .random_genome(BrainProvenance::default(), &mut rng)
            .unwrap();
        let provenance = BrainProvenance {
            parents: [Some(AgentUid(1)), Some(AgentUid(2))],
            parent_genome_hashes: [Some(left.material_hash()), Some(right.material_hash())],
            created_at: Tick(10),
            derivation: BrainGenomeDerivation::Crossover,
        };
        let mut first_rng = SmallRngStream::seed_from_u64(71);
        let mut second_rng = SmallRngStream::seed_from_u64(71);
        let child = family
            .crossover_genomes(&left, &right, provenance.clone(), &mut first_rng)
            .unwrap();
        let repeated = family
            .crossover_genomes(&left, &right, provenance.clone(), &mut second_rng)
            .unwrap();
        assert_eq!(child, repeated);
        assert_eq!(first_rng.random::<u64>(), second_rng.random::<u64>());
        assert_eq!(child.provenance(), &provenance);
        let start = family.genome_header.len();
        let mut chose_left = false;
        let mut chose_right = false;
        for ((value, from_left), from_right) in child.payload()[start..]
            .chunks_exact(SCALAR_BYTES)
            .zip(left.payload()[start..].chunks_exact(SCALAR_BYTES))
            .zip(right.payload()[start..].chunks_exact(SCALAR_BYTES))
        {
            assert!(value == from_left || value == from_right);
            chose_left |= value == from_left;
            chose_right |= value == from_right;
        }
        assert!(chose_left && chose_right);
    }

    #[test]
    fn malformed_noncanonical_and_cross_genome_state_are_refused() {
        let family = NeuroflowFamilyAdapter::new(small_config()).unwrap();
        let mut rng = SmallRngStream::seed_from_u64(0x4E46_0005);
        let genome = family
            .random_genome(BrainProvenance::default(), &mut rng)
            .unwrap();
        let state = family.initial_state(&genome, &mut rng).unwrap();
        for length in [0, GENOME_FIXED_HEADER_BYTES - 1, genome.payload().len() - 1] {
            assert!(
                family
                    .validate_genome(&genome_with_payload(
                        &family,
                        genome.payload()[..length].to_vec()
                    ))
                    .is_err()
            );
        }
        let mut trailing = genome.payload().to_vec();
        trailing.push(0);
        assert!(
            family
                .validate_genome(&genome_with_payload(&family, trailing))
                .is_err()
        );
        for offset in [0, 4, 8, 10, 11, 12, 20, 28, GENOME_FIXED_HEADER_BYTES] {
            let mut bytes = genome.payload().to_vec();
            bytes[offset] ^= 1;
            assert!(
                family
                    .validate_genome(&genome_with_payload(&family, bytes))
                    .is_err(),
                "header byte {offset}"
            );
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut bytes = genome.payload().to_vec();
            let start = family.genome_header.len();
            bytes[start..start + SCALAR_BYTES].copy_from_slice(&value.to_bits().to_le_bytes());
            assert!(
                family
                    .validate_genome(&genome_with_payload(&family, bytes))
                    .is_err()
            );
            let mut bytes = state.payload().to_vec();
            bytes[STATE_HEADER_BYTES..STATE_HEADER_BYTES + SCALAR_BYTES]
                .copy_from_slice(&value.to_bits().to_le_bytes());
            assert!(
                family
                    .validate_evaluator_state(&state_with_payload(&family, bytes))
                    .is_err()
            );
        }
        for length in [0, STATE_HEADER_BYTES - 1, state.payload().len() - 1] {
            assert!(
                family
                    .validate_evaluator_state(&state_with_payload(
                        &family,
                        state.payload()[..length].to_vec()
                    ))
                    .is_err()
            );
        }
        let mut bytes = state.payload().to_vec();
        bytes[STATE_MAGIC.len() + DIGEST_BYTES] ^= 1;
        assert!(
            family
                .validate_evaluator_state(&state_with_payload(&family, bytes))
                .is_err()
        );
        let other = family
            .random_genome(BrainProvenance::default(), &mut rng)
            .unwrap();
        assert!(family.evaluator(&other, &state).is_err());
        let different = NeuroflowFamilyAdapter::new(NeuroflowBrainConfig {
            activation: NeuroflowActivation::Sigmoid,
            ..small_config()
        })
        .unwrap();
        assert!(different.validate_genome(&genome).is_err());
        assert!(different.validate_evaluator_state(&state).is_err());
        let wrong_version = BrainGenomeEnvelope::new(
            family.family_id.clone(),
            2,
            GENOME_CODEC_VERSION,
            genome.payload().to_vec(),
            BrainProvenance::default(),
        )
        .unwrap();
        assert!(matches!(
            family.validate_genome(&wrong_version),
            Err(BrainProtocolError::SchemaVersionMismatch { .. })
        ));
        assert!(matches!(
            family.genome_loci(&genome),
            Err(BrainProtocolError::GenomeLociUnavailable { .. })
        ));
    }

    #[test]
    fn rejected_inputs_and_overflow_leave_accepted_state_exact_and_permit_retry() {
        let family = NeuroflowFamilyAdapter::new(NeuroflowBrainConfig {
            hidden_layers: Vec::new(),
            activation: NeuroflowActivation::Relu,
            ..small_config()
        })
        .unwrap();
        let mut rng = SmallRngStream::seed_from_u64(0x4E46_0006);
        let genome = family
            .random_genome(BrainProvenance::default(), &mut rng)
            .unwrap();
        let initial = family.initial_state(&genome, &mut rng).unwrap();
        let mut evaluator = family.evaluator(&genome, &initial).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(evaluator.evaluate(&[value; INPUT_SIZE]).is_err());
            assert_eq!(evaluator.checkpoint_state().unwrap(), initial);
        }

        let mut bytes = genome.payload().to_vec();
        bytes[family.genome_header.len()..].fill(0);
        let start = family.genome_header.len() + SCALAR_BYTES;
        bytes[start..start + SCALAR_BYTES].copy_from_slice(&f64::MAX.to_bits().to_le_bytes());
        let large = genome_with_payload(&family, bytes);
        family.validate_genome(&large).unwrap();
        let initial = family.initial_state(&large, &mut rng).unwrap();
        let mut evaluator = family.evaluator(&large, &initial).unwrap();
        let mut untouched = family.evaluator(&large, &initial).unwrap();
        let request = BrainInspection::Activations(BrainInspectionLimits::default());
        for input in [f32::MAX, 1.0] {
            assert!(evaluator.evaluate(&[input; INPUT_SIZE]).is_err());
            assert_eq!(evaluator.checkpoint_state().unwrap(), initial);
            assert_eq!(
                evaluator.inspect(request).unwrap(),
                untouched.inspect(request).unwrap()
            );
        }
        for _ in 0..3 {
            assert_eq!(
                evaluator
                    .evaluate(&[0.0; INPUT_SIZE])
                    .unwrap()
                    .map(f32::to_bits),
                untouched
                    .evaluate(&[0.0; INPUT_SIZE])
                    .unwrap()
                    .map(f32::to_bits)
            );
            assert_eq!(
                evaluator.checkpoint_state().unwrap(),
                untouched.checkpoint_state().unwrap()
            );
        }
    }

    #[test]
    fn real_world_birth_and_checkpoint_continue_with_exact_versioned_neuroflow_state() {
        let family = NeuroflowFamilyAdapter::new(small_config()).unwrap();
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            closed: true,
            population_spawn_interval: 0,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            food_sharing_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            aging_health_decay_rate: 0.0,
            spike_damage: 0.0,
            spike_energy_cost: 0.0,
            reproduction_energy_threshold: 0.4,
            reproduction_energy_cost: 0.1,
            reproduction_cooldown: 1,
            reproduction_attempt_interval: 1,
            reproduction_attempt_chance: 1.0,
            reproduction_partner_chance: 0.0,
            reproduction_meta_mutation_chance: 0.0,
            reproduction_meta_mutation_scale: 0.0,
            persistence_interval: 0,
            rng_seed: Some(0x4E46_0007),
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        let key = world
            .register_brain_family(NeuroflowBrain::KIND.as_str(), Box::new(family.clone()))
            .unwrap();
        let parent = world
            .try_spawn_agent(AgentData {
                position: Position::new(60.0, 60.0),
                health: 2.0,
                ..AgentData::default()
            })
            .unwrap();
        world.bind_agent_brain(parent, key).unwrap();
        world
            .try_update_agent_runtime(parent, |runtime| {
                runtime.energy = 1.0;
                runtime.reproduction_counter = 1.0;
                runtime.mutation_rates = MutationRates {
                    primary: 0.0,
                    secondary: 0.0,
                };
            })
            .unwrap();
        let parent_genome = world.agent_brain_genome(parent).unwrap().clone();
        world.step().unwrap();
        assert_eq!(
            world.agent_count(),
            2,
            "the fixture must actually reproduce"
        );
        let child = world
            .agents()
            .iter_handles()
            .find(|id| *id != parent)
            .unwrap();
        let child_genome = world.agent_brain_genome(child).unwrap();
        assert_eq!(child_genome.material_hash(), parent_genome.material_hash());
        assert_eq!(
            child_genome.provenance().parents[0],
            world.agent_uid(parent)
        );
        let mut rng = SmallRngStream::seed_from_u64(0);
        assert_eq!(
            world.agent_brain_evaluator_state(child).unwrap().unwrap(),
            family.initial_state(child_genome, &mut rng).unwrap()
        );
        let mut next = world.config().clone();
        next.reproduction_energy_threshold = f32::MAX;
        world.apply_config_update(next).unwrap();
        let checkpoint = world.checkpoint_v1().unwrap();
        let bytes = checkpoint.encode().unwrap();
        let checkpoint = scriptbots_core::WorldCheckpointV1::decode(&bytes).unwrap();
        let mut registry = BrainRegistry::default();
        registry
            .register_family(NeuroflowBrain::KIND.as_str(), Box::new(family))
            .unwrap();
        let mut resumed = WorldState::restore_checkpoint_v1(&checkpoint, registry).unwrap();
        assert_eq!(
            world.world_digest_v1().unwrap(),
            resumed.world_digest_v1().unwrap()
        );
        for _ in 0..6 {
            world.step().unwrap();
            resumed.step().unwrap();
            assert_eq!(
                world.world_digest_v1().unwrap(),
                resumed.world_digest_v1().unwrap()
            );
            assert_eq!(
                world.checkpoint_v1().unwrap().encode().unwrap(),
                resumed.checkpoint_v1().unwrap().encode().unwrap()
            );
        }
    }
}
