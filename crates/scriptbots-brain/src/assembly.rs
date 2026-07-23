//! Experimental assembly-style brain gated behind the `experimental` feature.

use rand::Rng;
use scriptbots_core::{
    BrainAdapterIdentityV1, BrainEnvelopeKind, BrainEvaluator, BrainEvaluatorStateEnvelope,
    BrainFamilyCodec, BrainFamilyId, BrainGenomeEnvelope, BrainGenomeMaterial, BrainInspection,
    BrainInspectionError, BrainInspectionSnapshot, BrainProtocolError, MutationRates,
    OffspringStatePolicy, RandomStream,
};
use std::any::Any;

#[cfg(test)]
use scriptbots_core::BrainProvenance;

use scriptbots_core::{BrainRunner, INPUT_SIZE, OUTPUT_SIZE};

use crate::{Brain, BrainKind, into_runner};

const BRAIN_SIZE: usize = 200;
const ASSEMBLY_FAMILY_ID: &str = "assembly";
const ADAPTER_SEMANTIC_VERSION: u32 = 1;
const ADAPTER_SEMANTIC_DESCRIPTOR: &[u8] = b"scriptbots.assembly.adapter-semantics.v1";
const ASSEMBLY_GENOME_SCHEMA_VERSION: u32 = 1;
const ASSEMBLY_GENOME_CODEC_VERSION: u16 = 1;
const ASSEMBLY_STATE_SCHEMA_VERSION: u32 = 1;
const ASSEMBLY_STATE_CODEC_VERSION: u16 = 1;
const ASSEMBLY_CELL_BYTES: usize = std::mem::size_of::<f32>();
const ASSEMBLY_GENOME_PAYLOAD_BYTES: usize = BRAIN_SIZE * ASSEMBLY_CELL_BYTES;
const ASSEMBLY_STATE_MAGIC: [u8; 8] = *b"ASMBST01";
const ASSEMBLY_GENOME_DIGEST_BYTES: usize = blake3::OUT_LEN;
const ASSEMBLY_STATE_HEADER_BYTES: usize =
    ASSEMBLY_STATE_MAGIC.len() + ASSEMBLY_GENOME_DIGEST_BYTES;
const ASSEMBLY_STATE_PAYLOAD_BYTES: usize =
    ASSEMBLY_STATE_HEADER_BYTES + ASSEMBLY_GENOME_PAYLOAD_BYTES;
/// Maximum instruction slots scanned by one Assembly evaluator tick.
///
/// The language deliberately preserves the legacy single-pass program: every middle cell is
/// visited at most once, so data-dependent jumps and unbounded loops are impossible.
pub const ASSEMBLY_INSTRUCTION_BUDGET: usize = BRAIN_SIZE - INPUT_SIZE - OUTPUT_SIZE;

/// Assembly-like instruction brain mirroring the legacy implementation.
///
/// Persistence deliberately goes through [`AssemblyFamilyAdapter`]'s bounded, versioned protocol
/// envelopes. Exposing raw serde for this type would let callers construct short or non-finite
/// programs that bypass those invariants.
#[derive(Debug, Clone)]
pub struct AssemblyBrain {
    cells: [f32; BRAIN_SIZE],
}

impl AssemblyBrain {
    /// Trait identifier for this brain.
    pub const KIND: BrainKind = BrainKind::new("assembly.experimental");

    /// Construct a randomly initialized assembly brain.
    #[must_use]
    pub fn random(rng: &mut dyn RandomStream) -> Self {
        let mut cells = [0.0; BRAIN_SIZE];
        for cell in &mut cells {
            let mut value = rng.random_range(-3.0..3.0);
            if rng.random::<f32>() < 0.1 {
                value = rng.random_range(0.0..0.5);
            }
            if rng.random::<f32>() < 0.1 {
                value = rng.random_range(0.8..1.0);
            }
            *cell = value;
        }

        Self { cells }
    }

    /// Return a boxed runner for this brain implementation.
    #[must_use]
    pub fn runner(rng: &mut dyn RandomStream) -> Box<dyn BrainRunner> {
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
            if value.is_nan() {
                *value = 0.0;
            } else {
                *value = (*value).clamp(-10.0, 10.0);
            }
        }
    }

    fn from_cells(cells: [f32; BRAIN_SIZE]) -> Result<Self, BrainProtocolError> {
        validate_cells(&cells, BrainEnvelopeKind::EvaluatorState)?;
        Ok(Self { cells })
    }

    fn tick_with_budget(&mut self, inputs: &[f32; INPUT_SIZE]) -> ([f32; OUTPUT_SIZE], usize) {
        for (idx, input) in inputs.iter().enumerate() {
            self.cells[idx] = if input.is_nan() { 0.0 } else { *input };
        }

        let mut scanned = 0;
        for i in INPUT_SIZE..(BRAIN_SIZE - OUTPUT_SIZE) {
            scanned += 1;
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
        debug_assert_eq!(scanned, ASSEMBLY_INSTRUCTION_BUDGET);

        Self::clamp_cells(&mut self.cells[INPUT_SIZE..(BRAIN_SIZE - OUTPUT_SIZE)]);

        let mut outputs = [0.0; OUTPUT_SIZE];
        for (offset, output) in outputs.iter_mut().enumerate() {
            let idx = BRAIN_SIZE - 1 - offset;
            let val = self.cells[idx];
            *output = if val.is_nan() {
                0.0
            } else {
                val.clamp(0.0, 1.0)
            };
        }

        (outputs, scanned)
    }
}

/// Versioned protocol adapter for the Assembly brain family.
///
/// The genome owns the initial 200-cell program. The evaluator state owns all 200 live cells
/// because the legacy single-pass language is intentionally self-modifying: instructions may
/// write into later instruction cells, input cells, or output cells and thereby affect later
/// ticks. Offspring reset working state from their mutated/crossed genome instead of inheriting a
/// parent's partially executed program.
#[derive(Debug, Clone)]
pub struct AssemblyFamilyAdapter {
    family_id: BrainFamilyId,
}

#[derive(Debug, Clone, PartialEq)]
struct DecodedAssemblyState {
    genome_digest: [u8; ASSEMBLY_GENOME_DIGEST_BYTES],
    cells: [f32; BRAIN_SIZE],
}

impl AssemblyFamilyAdapter {
    /// Construct the canonical Assembly protocol adapter.
    pub fn new() -> Result<Self, BrainProtocolError> {
        Ok(Self {
            family_id: BrainFamilyId::new(ASSEMBLY_FAMILY_ID)?,
        })
    }

    fn decode_genome(
        &self,
        genome: &BrainGenomeEnvelope,
    ) -> Result<[f32; BRAIN_SIZE], BrainProtocolError> {
        genome.require_protocol(
            &self.family_id,
            ASSEMBLY_GENOME_SCHEMA_VERSION,
            ASSEMBLY_GENOME_CODEC_VERSION,
        )?;
        decode_cells(genome.payload(), BrainEnvelopeKind::Genome, &self.family_id)
    }

    fn decode_state(
        &self,
        state: &BrainEvaluatorStateEnvelope,
    ) -> Result<DecodedAssemblyState, BrainProtocolError> {
        state.require_protocol(
            &self.family_id,
            ASSEMBLY_STATE_SCHEMA_VERSION,
            ASSEMBLY_STATE_CODEC_VERSION,
        )?;
        decode_state_payload(state.payload(), &self.family_id)
    }

    #[cfg(test)]
    fn genome(
        &self,
        cells: &[f32; BRAIN_SIZE],
        provenance: BrainProvenance,
    ) -> Result<BrainGenomeEnvelope, BrainProtocolError> {
        BrainGenomeEnvelope::new(
            self.family_id.clone(),
            ASSEMBLY_GENOME_SCHEMA_VERSION,
            ASSEMBLY_GENOME_CODEC_VERSION,
            encode_cells(cells, BrainEnvelopeKind::Genome, &self.family_id)?,
            provenance,
        )
    }

    fn genome_material(
        &self,
        cells: &[f32; BRAIN_SIZE],
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        BrainGenomeMaterial::new(
            ASSEMBLY_GENOME_SCHEMA_VERSION,
            ASSEMBLY_GENOME_CODEC_VERSION,
            encode_cells(cells, BrainEnvelopeKind::Genome, &self.family_id)?,
        )
    }

    fn state(
        &self,
        genome: &BrainGenomeEnvelope,
        cells: &[f32; BRAIN_SIZE],
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        genome.require_protocol(
            &self.family_id,
            ASSEMBLY_GENOME_SCHEMA_VERSION,
            ASSEMBLY_GENOME_CODEC_VERSION,
        )?;
        BrainEvaluatorStateEnvelope::new(
            self.family_id.clone(),
            ASSEMBLY_STATE_SCHEMA_VERSION,
            ASSEMBLY_STATE_CODEC_VERSION,
            encode_state_payload(
                genome_digest(genome),
                cells,
                BrainEnvelopeKind::EvaluatorState,
                &self.family_id,
            )?,
        )
    }
}

struct AssemblyProtocolEvaluator {
    family_id: BrainFamilyId,
    genome_digest: [u8; ASSEMBLY_GENOME_DIGEST_BYTES],
    brain: AssemblyBrain,
}

impl BrainEvaluator for AssemblyProtocolEvaluator {
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
                &self.family_id,
                format!("sensor {index} is non-finite ({value})"),
            ));
        }

        // Execute transactionally: an overflow/non-finite instruction result must not leave a
        // poisoned evaluator that can later be checkpointed as if it were valid.
        let mut candidate = self.brain.clone();
        let (outputs, scanned) = candidate.tick_with_budget(sensors);
        if scanned != ASSEMBLY_INSTRUCTION_BUDGET {
            return Err(invalid_payload(
                BrainEnvelopeKind::EvaluatorState,
                &self.family_id,
                format!(
                    "instruction scan consumed {scanned} slots; budget is {ASSEMBLY_INSTRUCTION_BUDGET}"
                ),
            ));
        }
        validate_cells_for_family(
            &candidate.cells,
            BrainEnvelopeKind::EvaluatorState,
            &self.family_id,
        )?;
        self.brain = candidate;
        Ok(outputs)
    }

    fn inspect(
        &self,
        request: BrainInspection,
    ) -> Result<Option<BrainInspectionSnapshot>, BrainInspectionError> {
        match request {
            BrainInspection::Activations(_) => Ok(None),
        }
    }

    fn checkpoint_state(&self) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        BrainEvaluatorStateEnvelope::new(
            self.family_id.clone(),
            ASSEMBLY_STATE_SCHEMA_VERSION,
            ASSEMBLY_STATE_CODEC_VERSION,
            encode_state_payload(
                self.genome_digest,
                &self.brain.cells,
                BrainEnvelopeKind::EvaluatorState,
                &self.family_id,
            )?,
        )
    }
}

impl BrainFamilyCodec for AssemblyFamilyAdapter {
    fn family_id(&self) -> &BrainFamilyId {
        &self.family_id
    }

    fn adapter_identity(&self) -> BrainAdapterIdentityV1 {
        BrainAdapterIdentityV1::from_semantic_descriptor(
            self.family_id(),
            ADAPTER_SEMANTIC_VERSION,
            ADAPTER_SEMANTIC_DESCRIPTOR,
        )
    }

    fn random_genome_material(
        &self,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        let brain = AssemblyBrain::random(rng);
        self.genome_material(&brain.cells)
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
        validate_mutation_probability(rates.primary, &self.family_id)?;
        validate_mutation_scale(rates.secondary, &self.family_id)?;
        let mut cells = self.decode_genome(genome)?;
        for cell in &mut cells {
            if rng.random::<f32>() < rates.primary {
                *cell = rng.random_range(-3.0..3.0);
            }
        }
        // Legacy Assembly mutation deliberately ignores MR2. `secondary` is retained in the
        // shared rate structure but cannot silently acquire a different meaning for this family.
        self.genome_material(&cells)
    }

    fn crossover_genomes_material(
        &self,
        left: &BrainGenomeEnvelope,
        right: &BrainGenomeEnvelope,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        let left_cells = self.decode_genome(left)?;
        let right_cells = self.decode_genome(right)?;
        let mut child = right_cells;
        for (cell, left_cell) in child.iter_mut().zip(left_cells) {
            if rng.random::<f32>() < 0.5 {
                *cell = left_cell;
            }
        }
        self.genome_material(&child)
    }

    fn initial_state(
        &self,
        genome: &BrainGenomeEnvelope,
        _rng: &mut dyn RandomStream,
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.state(genome, &self.decode_genome(genome)?)
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
        let state = self.decode_state(state)?;
        let expected_digest = genome_digest(genome);
        if state.genome_digest != expected_digest {
            return Err(invalid_payload(
                BrainEnvelopeKind::EvaluatorState,
                &self.family_id,
                format!(
                    "Assembly state belongs to genome {}, but evaluator received {}",
                    state_digest_hex(&state.genome_digest),
                    state_digest_hex(&expected_digest)
                ),
            ));
        }
        Ok(Box::new(AssemblyProtocolEvaluator {
            family_id: self.family_id.clone(),
            genome_digest: expected_digest,
            brain: AssemblyBrain::from_cells(state.cells)?,
        }))
    }
}

fn encode_cells(
    cells: &[f32; BRAIN_SIZE],
    kind: BrainEnvelopeKind,
    family_id: &BrainFamilyId,
) -> Result<Vec<u8>, BrainProtocolError> {
    validate_cells_for_family(cells, kind, family_id)?;
    let mut payload = Vec::with_capacity(ASSEMBLY_GENOME_PAYLOAD_BYTES);
    for cell in cells {
        payload.extend_from_slice(&cell.to_bits().to_le_bytes());
    }
    Ok(payload)
}

fn decode_cells(
    payload: &[u8],
    kind: BrainEnvelopeKind,
    family_id: &BrainFamilyId,
) -> Result<[f32; BRAIN_SIZE], BrainProtocolError> {
    if payload.len() != ASSEMBLY_GENOME_PAYLOAD_BYTES {
        return Err(invalid_payload(
            kind,
            family_id,
            format!(
                "Assembly cell payload is {} bytes; expected exactly {ASSEMBLY_GENOME_PAYLOAD_BYTES}",
                payload.len()
            ),
        ));
    }
    let (encoded_cells, remainder) = payload.as_chunks::<ASSEMBLY_CELL_BYTES>();
    debug_assert!(remainder.is_empty());
    let cells =
        std::array::from_fn(|index| f32::from_bits(u32::from_le_bytes(encoded_cells[index])));
    validate_cells_for_family(&cells, kind, family_id)?;
    Ok(cells)
}

fn encode_state_payload(
    genome_digest: [u8; ASSEMBLY_GENOME_DIGEST_BYTES],
    cells: &[f32; BRAIN_SIZE],
    kind: BrainEnvelopeKind,
    family_id: &BrainFamilyId,
) -> Result<Vec<u8>, BrainProtocolError> {
    validate_cells_for_family(cells, kind, family_id)?;
    let mut payload = Vec::with_capacity(ASSEMBLY_STATE_PAYLOAD_BYTES);
    payload.extend_from_slice(&ASSEMBLY_STATE_MAGIC);
    payload.extend_from_slice(&genome_digest);
    for cell in cells {
        payload.extend_from_slice(&cell.to_bits().to_le_bytes());
    }
    debug_assert_eq!(payload.len(), ASSEMBLY_STATE_PAYLOAD_BYTES);
    Ok(payload)
}

fn decode_state_payload(
    payload: &[u8],
    family_id: &BrainFamilyId,
) -> Result<DecodedAssemblyState, BrainProtocolError> {
    if payload.len() != ASSEMBLY_STATE_PAYLOAD_BYTES {
        return Err(invalid_payload(
            BrainEnvelopeKind::EvaluatorState,
            family_id,
            format!(
                "Assembly state payload is {} bytes; expected exactly {ASSEMBLY_STATE_PAYLOAD_BYTES}",
                payload.len()
            ),
        ));
    }
    if payload[..ASSEMBLY_STATE_MAGIC.len()] != ASSEMBLY_STATE_MAGIC {
        return Err(invalid_payload(
            BrainEnvelopeKind::EvaluatorState,
            family_id,
            "Assembly state payload has an invalid ASMBST01 codec magic".to_owned(),
        ));
    }
    let digest_start = ASSEMBLY_STATE_MAGIC.len();
    let digest_end = ASSEMBLY_STATE_HEADER_BYTES;
    let mut genome_digest = [0; ASSEMBLY_GENOME_DIGEST_BYTES];
    genome_digest.copy_from_slice(&payload[digest_start..digest_end]);
    let cells = decode_cells(
        &payload[ASSEMBLY_STATE_HEADER_BYTES..],
        BrainEnvelopeKind::EvaluatorState,
        family_id,
    )?;
    Ok(DecodedAssemblyState {
        genome_digest,
        cells,
    })
}

fn genome_digest(genome: &BrainGenomeEnvelope) -> [u8; ASSEMBLY_GENOME_DIGEST_BYTES] {
    *genome.material_hash().as_bytes()
}

fn state_digest_hex(digest: &[u8; ASSEMBLY_GENOME_DIGEST_BYTES]) -> String {
    blake3::Hash::from_bytes(*digest).to_hex().to_string()
}

fn validate_cells(
    cells: &[f32; BRAIN_SIZE],
    kind: BrainEnvelopeKind,
) -> Result<(), BrainProtocolError> {
    let family_id = BrainFamilyId::new(ASSEMBLY_FAMILY_ID)?;
    validate_cells_for_family(cells, kind, &family_id)
}

fn validate_cells_for_family(
    cells: &[f32; BRAIN_SIZE],
    kind: BrainEnvelopeKind,
    family_id: &BrainFamilyId,
) -> Result<(), BrainProtocolError> {
    if let Some((index, value)) = cells
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(invalid_payload(
            kind,
            family_id,
            format!("Assembly cell {index} is non-finite ({value})"),
        ));
    }
    Ok(())
}

fn validate_mutation_probability(
    probability: f32,
    family_id: &BrainFamilyId,
) -> Result<(), BrainProtocolError> {
    if (0.0..=1.0).contains(&probability) {
        Ok(())
    } else {
        Err(invalid_payload(
            BrainEnvelopeKind::Genome,
            family_id,
            format!(
                "Assembly mutation probability must be finite and in [0, 1], found {probability}"
            ),
        ))
    }
}

fn validate_mutation_scale(
    scale: f32,
    family_id: &BrainFamilyId,
) -> Result<(), BrainProtocolError> {
    if scale.is_finite() && scale >= 0.0 {
        Ok(())
    } else {
        Err(invalid_payload(
            BrainEnvelopeKind::Genome,
            family_id,
            format!(
                "Assembly secondary mutation scale must be finite and nonnegative, found {scale}"
            ),
        ))
    }
}

fn invalid_payload(
    kind: BrainEnvelopeKind,
    family_id: &BrainFamilyId,
    detail: String,
) -> BrainProtocolError {
    BrainProtocolError::InvalidPayload {
        kind,
        family_id: family_id.clone(),
        detail,
    }
}

impl Brain for AssemblyBrain {
    fn kind(&self) -> BrainKind {
        Self::KIND
    }

    fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        self.tick_with_budget(inputs).0
    }

    fn mutate(
        &mut self,
        rng: &mut dyn RandomStream,
        rate: f32,
        _scale: f32,
    ) -> Result<(), crate::BrainMutationError> {
        for cell in &mut self.cells {
            if rng.random::<f32>() < rate {
                *cell = rng.random_range(-3.0..3.0);
            }
        }
        Ok(())
    }

    fn crossover(&self, other: &dyn Brain, rng: &mut dyn RandomStream) -> Option<Box<dyn Brain>> {
        if other.kind() != Self::KIND {
            return None;
        }

        let other = other.as_any().downcast_ref::<Self>()?;

        let mut child = other.clone();
        for (value, self_value) in child.cells.iter_mut().zip(&self.cells) {
            if rng.random::<f32>() < 0.5 {
                *value = *self_value;
            }
        }

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

    fn inspect(
        &self,
        request: BrainInspection,
    ) -> Result<Option<BrainInspectionSnapshot>, BrainInspectionError> {
        match request {
            BrainInspection::Activations(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;
    use scriptbots_core::{
        AgentUid, BrainFamilyAdapter, BrainGenomeDerivation, BrainInspectionLimits, SmallRngStream,
        Tick,
    };

    #[test]
    fn adapter_semantic_identity_v1_is_pinned() {
        let identity = AssemblyFamilyAdapter::new()
            .expect("canonical Assembly adapter")
            .adapter_identity();
        assert_eq!(identity.semantic_version(), ADAPTER_SEMANTIC_VERSION);
        assert_eq!(
            identity.to_string(),
            "f0f345102059ff017681acdc92998a4de7817a8ee1afe579bae300b6ccfe92b7",
            "update only after reviewing an intentional Assembly executable-semantics change"
        );
    }

    #[derive(Clone, Debug, Default)]
    struct AlternatingThresholdStream {
        sample_index: usize,
    }

    impl AlternatingThresholdStream {
        fn next_scripted_u64(&mut self) -> u64 {
            let sample = if self.sample_index.is_multiple_of(2) {
                0
            } else {
                u64::MAX
            };
            self.sample_index += 1;
            sample
        }
    }

    impl RngCore for AlternatingThresholdStream {
        fn next_u32(&mut self) -> u32 {
            self.next_scripted_u64() as u32
        }

        fn next_u64(&mut self) -> u64 {
            self.next_scripted_u64()
        }

        fn fill_bytes(&mut self, destination: &mut [u8]) {
            let (chunks, remainder) = destination.as_chunks_mut::<8>();
            for chunk in chunks {
                *chunk = self.next_u64().to_le_bytes();
            }
            if !remainder.is_empty() {
                let sample = self.next_u64().to_le_bytes();
                remainder.copy_from_slice(&sample[..remainder.len()]);
            }
        }
    }

    impl RandomStream for AlternatingThresholdStream {
        fn algorithm_id(&self) -> &'static str {
            "test.alternating-threshold"
        }

        fn checkpoint(&self) -> scriptbots_core::RandomStreamState {
            scriptbots_core::RandomStreamState {
                version: 1,
                algorithm: self.algorithm_id().to_owned(),
                codec_version: 1,
                state: self.sample_index.to_le_bytes().to_vec(),
            }
        }
    }

    fn fixture_cells() -> [f32; BRAIN_SIZE] {
        let mut cells = [0.0; BRAIN_SIZE];
        let accumulate = INPUT_SIZE;
        cells[accumulate] = 2.05;
        cells[accumulate + 1] = 0.0; // input 0
        cells[accumulate + 2] = 0.25; // persistent cell 50
        cells[accumulate + 3] = 0.25; // persistent cell 50
        let publish = INPUT_SIZE + 6;
        cells[publish] = 2.05;
        cells[publish + 1] = 0.25; // persistent cell 50
        cells[publish + 2] = 0.30; // zero cell 60
        cells[publish + 3] = 0.995; // output cell 199
        cells[50] = 0.125;
        cells
    }

    fn fixture_provenance() -> BrainProvenance {
        BrainProvenance {
            created_at: Tick(31),
            ..BrainProvenance::default()
        }
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    #[test]
    fn random_brain_has_expected_length() {
        let mut rng = SmallRngStream::seed_from_u64(0xABCD);
        let brain = AssemblyBrain::random(&mut rng);
        assert_eq!(brain.cells.len(), BRAIN_SIZE);
    }

    #[test]
    fn tick_outputs_in_range() {
        let mut rng = SmallRngStream::seed_from_u64(4242);
        let mut brain = AssemblyBrain::random(&mut rng);
        let inputs = [0.5; INPUT_SIZE];
        let outputs = brain.tick(&inputs);
        assert!(outputs.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn mutate_changes_cells() {
        let mut rng = SmallRngStream::seed_from_u64(1717);
        let mut brain = AssemblyBrain::random(&mut rng);
        let before = brain.cells[10];
        brain
            .mutate(&mut rng, 1.0, 0.5)
            .expect("assembly mutation is infallible");
        assert_ne!(brain.cells[10], before);
    }

    #[test]
    fn crossover_selects_values() {
        let mut rng = SmallRngStream::seed_from_u64(9999);
        let brain_a = AssemblyBrain::random(&mut rng);
        let brain_b = AssemblyBrain::random(&mut rng);
        let mut rng = SmallRngStream::seed_from_u64(1111);
        let child = brain_a
            .crossover(&brain_b, &mut rng)
            .expect("matching kinds");
        assert_eq!(child.kind(), AssemblyBrain::KIND);
    }

    #[test]
    fn crossover_matches_cpp_left_for_each_low_scripted_draw() {
        let family = AssemblyFamilyAdapter::new().expect("canonical Assembly family");
        let left_cells = [-1.0; BRAIN_SIZE];
        let right_cells = [1.0; BRAIN_SIZE];
        let expected =
            std::array::from_fn(|index| if index.is_multiple_of(2) { -1.0 } else { 1.0 });
        let left = family
            .genome(&left_cells, fixture_provenance())
            .expect("left genome");
        let right = family
            .genome(&right_cells, fixture_provenance())
            .expect("right genome");

        let mut protocol_rng = AlternatingThresholdStream::default();
        let protocol_child = family
            .crossover_genomes(&left, &right, BrainProvenance::default(), &mut protocol_rng)
            .expect("protocol crossover");
        assert_eq!(
            family
                .decode_genome(&protocol_child)
                .expect("protocol child cells"),
            expected,
            "C++ chooses the left parent when the per-locus draw is below 0.5"
        );
        assert_eq!(protocol_rng.sample_index, BRAIN_SIZE);

        let legacy_left = AssemblyBrain { cells: left_cells };
        let legacy_right = AssemblyBrain { cells: right_cells };
        let mut legacy_rng = AlternatingThresholdStream::default();
        let legacy_child = legacy_left
            .crossover(&legacy_right, &mut legacy_rng)
            .expect("legacy crossover")
            .as_any()
            .downcast_ref::<AssemblyBrain>()
            .expect("Assembly child")
            .clone();
        assert_eq!(legacy_child.cells, expected);
        assert_eq!(legacy_rng.sample_index, BRAIN_SIZE);
    }

    #[test]
    fn runner_executes_program() {
        let mut rng = SmallRngStream::seed_from_u64(2025);
        let mut runner = AssemblyBrain::runner(&mut rng);
        let inputs = [0.0; INPUT_SIZE];
        let outputs = runner.tick(&inputs);
        assert!(outputs.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn protocol_codec_is_exact_and_checkpoint_restore_preserves_the_next_output() {
        let family = AssemblyFamilyAdapter::new().expect("canonical Assembly family");
        let cells = fixture_cells();
        let genome = family
            .genome(&cells, fixture_provenance())
            .expect("fixture genome");
        assert_eq!(genome.payload().len(), ASSEMBLY_GENOME_PAYLOAD_BYTES);
        assert_eq!(
            &genome.payload()[..ASSEMBLY_CELL_BYTES],
            &0.0_f32.to_bits().to_le_bytes()
        );
        assert_eq!(
            fnv1a64(genome.payload()),
            0x2af8_892d_a84c_d282,
            "update only after reviewing an intentional Assembly codec change"
        );
        assert_eq!(family.decode_genome(&genome).expect("decode genome"), cells);

        let mut rng = SmallRngStream::seed_from_u64(71);
        let state = family
            .initial_state(&genome, &mut rng)
            .expect("initial state");
        assert_eq!(state.payload().len(), ASSEMBLY_STATE_PAYLOAD_BYTES);
        assert_eq!(
            &state.payload()[..ASSEMBLY_STATE_MAGIC.len()],
            ASSEMBLY_STATE_MAGIC
        );
        assert_eq!(
            &state.payload()[ASSEMBLY_STATE_MAGIC.len()..ASSEMBLY_STATE_HEADER_BYTES],
            genome_digest(&genome)
        );
        assert_eq!(
            &state.payload()[ASSEMBLY_STATE_HEADER_BYTES..],
            genome.payload()
        );
        assert_eq!(
            fnv1a64(state.payload()),
            0x0004_9030_3a2f_eadf,
            "update only after reviewing an intentional Assembly state codec change"
        );
        let mut evaluator = family.evaluator(&genome, &state).expect("evaluator");
        let mut first_inputs = [0.0; INPUT_SIZE];
        first_inputs[0] = 0.25;
        let first_outputs = evaluator.evaluate(&first_inputs).expect("first output");
        assert_eq!(first_outputs[0].to_bits(), 0.375_f32.to_bits());

        let checkpoint = family
            .checkpoint_evaluator(evaluator.as_ref())
            .expect("validated checkpoint");
        assert!(
            evaluator
                .inspect(BrainInspection::Activations(BrainInspectionLimits::hard(),))
                .expect("Assembly inspection refusal")
                .is_none(),
            "Assembly must explicitly report that activations are unsupported"
        );
        assert_eq!(
            family
                .checkpoint_evaluator(evaluator.as_ref())
                .expect("checkpoint after unsupported inspection"),
            checkpoint,
            "unsupported inspection must not alter Assembly working state"
        );
        assert_ne!(checkpoint.payload(), state.payload());
        let mut restored = family
            .evaluator(&genome, &checkpoint)
            .expect("restored evaluator");
        let mut next_inputs = [0.0; INPUT_SIZE];
        next_inputs[0] = 0.5;
        let expected = evaluator.evaluate(&next_inputs).expect("continued output");
        let actual = restored.evaluate(&next_inputs).expect("restored output");
        assert_eq!(
            expected.map(f32::to_bits),
            actual.map(f32::to_bits),
            "checkpoint reconstruction must preserve every future-affecting working cell"
        );
        assert_eq!(expected[0].to_bits(), 0.875_f32.to_bits());
    }

    #[test]
    fn protocol_mutation_crossover_and_offspring_reset_only_touch_heritable_cells() {
        let family = AssemblyFamilyAdapter::new().expect("canonical Assembly family");
        let left_cells = [-1.0; BRAIN_SIZE];
        let right_cells = [1.0; BRAIN_SIZE];
        let left = family
            .genome(&left_cells, fixture_provenance())
            .expect("left genome");
        let right = family
            .genome(&right_cells, fixture_provenance())
            .expect("right genome");

        let mut rng = SmallRngStream::seed_from_u64(91);
        let unchanged_provenance = BrainProvenance {
            parents: [Some(AgentUid(101)), None],
            parent_genome_hashes: [Some(left.material_hash()), None],
            created_at: Tick(40),
            derivation: BrainGenomeDerivation::Clone,
        };
        let unchanged = family
            .mutate_genome(
                &left,
                MutationRates {
                    primary: 0.0,
                    secondary: 999.0,
                },
                unchanged_provenance.clone(),
                &mut rng,
            )
            .expect("zero-rate mutation");
        assert_eq!(unchanged.payload(), left.payload());
        assert_eq!(unchanged.provenance(), &unchanged_provenance);

        let mutated_provenance = BrainProvenance {
            parents: [Some(AgentUid(102)), None],
            parent_genome_hashes: [Some(left.material_hash()), None],
            created_at: Tick(41),
            derivation: BrainGenomeDerivation::MutationOnly,
        };
        let mutated = family
            .mutate_genome(
                &left,
                MutationRates {
                    primary: 1.0,
                    secondary: 0.0,
                },
                mutated_provenance.clone(),
                &mut rng,
            )
            .expect("full mutation");
        assert_eq!(mutated.provenance(), &mutated_provenance);
        let mutated_cells = family.decode_genome(&mutated).expect("mutated cells");
        assert_ne!(mutated_cells, left_cells);
        assert!(mutated_cells.iter().all(|cell| (-3.0..3.0).contains(cell)));

        let child_provenance = BrainProvenance {
            parents: [Some(AgentUid(103)), Some(AgentUid(104))],
            parent_genome_hashes: [Some(left.material_hash()), Some(right.material_hash())],
            created_at: Tick(42),
            derivation: BrainGenomeDerivation::Crossover,
        };
        let child = family
            .crossover_genomes(&left, &right, child_provenance.clone(), &mut rng)
            .expect("field-wise crossover");
        assert_eq!(child.provenance(), &child_provenance);
        let child_cells = family.decode_genome(&child).expect("child cells");
        assert!(child_cells.iter().all(|cell| *cell == -1.0 || *cell == 1.0));
        assert!(child_cells.contains(&-1.0));
        assert!(child_cells.contains(&1.0));

        assert_eq!(family.offspring_state_policy(), OffspringStatePolicy::Reset);
        let parent_genome = family
            .genome(&[7.0; BRAIN_SIZE], fixture_provenance())
            .expect("parent genome");
        let parent_state = family
            .state(&parent_genome, &[7.0; BRAIN_SIZE])
            .expect("parent state");
        let reset = family
            .offspring_state(&child, &[&parent_state], &mut rng)
            .expect("reset offspring state");
        assert_eq!(
            &reset.payload()[ASSEMBLY_STATE_HEADER_BYTES..],
            child.payload()
        );
        assert_ne!(reset, parent_state);
    }

    #[test]
    fn protocol_rejects_invalid_shapes_nonfinite_cells_rates_and_inputs_transactionally() {
        let family = AssemblyFamilyAdapter::new().expect("canonical Assembly family");
        let genome = family
            .genome(&fixture_cells(), fixture_provenance())
            .expect("fixture genome");

        let short = BrainGenomeEnvelope::new(
            family.family_id.clone(),
            ASSEMBLY_GENOME_SCHEMA_VERSION,
            ASSEMBLY_GENOME_CODEC_VERSION,
            vec![0; ASSEMBLY_GENOME_PAYLOAD_BYTES - 1],
            BrainProvenance::default(),
        )
        .expect("generic envelope accepts family-owned shape");
        assert!(matches!(
            family.validate_genome(&short),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::Genome,
                ..
            })
        ));

        let mut nan_payload = genome.payload().to_vec();
        nan_payload[..ASSEMBLY_CELL_BYTES].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        let nan_genome = BrainGenomeEnvelope::new(
            family.family_id.clone(),
            ASSEMBLY_GENOME_SCHEMA_VERSION,
            ASSEMBLY_GENOME_CODEC_VERSION,
            nan_payload,
            BrainProvenance::default(),
        )
        .expect("generic envelope accepts family-owned values");
        assert!(matches!(
            family.validate_genome(&nan_genome),
            Err(BrainProtocolError::InvalidPayload { .. })
        ));
        for invalid_rate in [f32::NAN, -0.01, 1.01] {
            let mut rng = SmallRngStream::seed_from_u64(1);
            assert!(matches!(
                family.mutate_genome(
                    &genome,
                    MutationRates {
                        primary: invalid_rate,
                        secondary: 0.0,
                    },
                    BrainProvenance::default(),
                    &mut rng,
                ),
                Err(BrainProtocolError::InvalidPayload { .. })
            ));
        }
        for invalid_scale in [f32::NAN, -0.01, f32::INFINITY, f32::NEG_INFINITY] {
            let mut rng = SmallRngStream::seed_from_u64(1);
            assert!(matches!(
                family.mutate_genome(
                    &genome,
                    MutationRates {
                        primary: 0.0,
                        secondary: invalid_scale,
                    },
                    BrainProvenance::default(),
                    &mut rng,
                ),
                Err(BrainProtocolError::InvalidPayload { .. })
            ));
        }

        let mut rng = SmallRngStream::seed_from_u64(2);
        let state = family
            .initial_state(&genome, &mut rng)
            .expect("initial state");

        let mut invalid_magic_payload = state.payload().to_vec();
        invalid_magic_payload[0] ^= 0xff;
        let invalid_magic_state = BrainEvaluatorStateEnvelope::new(
            family.family_id.clone(),
            ASSEMBLY_STATE_SCHEMA_VERSION,
            ASSEMBLY_STATE_CODEC_VERSION,
            invalid_magic_payload,
        )
        .expect("generic envelope accepts family-owned state bytes");
        assert!(matches!(
            family.validate_evaluator_state(&invalid_magic_state),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::EvaluatorState,
                ..
            })
        ));

        let mut different_cells = fixture_cells();
        different_cells[50] = 0.25;
        let different_genome = family
            .genome(&different_cells, fixture_provenance())
            .expect("different genome");
        assert!(matches!(
            family.evaluator(&different_genome, &state),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::EvaluatorState,
                ..
            })
        ));

        let mut evaluator = family.evaluator(&genome, &state).expect("evaluator");
        let before = family
            .checkpoint_evaluator(evaluator.as_ref())
            .expect("checkpoint before rejection");
        let mut invalid_inputs = [0.0; INPUT_SIZE];
        invalid_inputs[0] = f32::INFINITY;
        assert!(matches!(
            evaluator.evaluate(&invalid_inputs),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::EvaluatorState,
                ..
            })
        ));
        assert_eq!(
            family
                .checkpoint_evaluator(evaluator.as_ref())
                .expect("checkpoint after rejection"),
            before,
            "a rejected tick must not poison live working state"
        );

        let mut overflowing_cells = [0.0; BRAIN_SIZE];
        overflowing_cells[INPUT_SIZE] = 2.25; // multiply
        overflowing_cells[INPUT_SIZE + 1] = 0.25; // cell 50
        overflowing_cells[INPUT_SIZE + 2] = 0.50; // cell 100
        overflowing_cells[INPUT_SIZE + 3] = 0.995; // output cell 199 (not clamped)
        overflowing_cells[50] = f32::MAX;
        overflowing_cells[100] = f32::MAX;
        let overflowing_genome = family
            .genome(&overflowing_cells, fixture_provenance())
            .expect("finite overflow program");
        let overflowing_state = family
            .initial_state(&overflowing_genome, &mut rng)
            .expect("overflow program initial state");
        let mut overflowing_evaluator = family
            .evaluator(&overflowing_genome, &overflowing_state)
            .expect("overflow evaluator");
        let before_overflow = family
            .checkpoint_evaluator(overflowing_evaluator.as_ref())
            .expect("checkpoint before interpreter overflow");
        assert!(matches!(
            overflowing_evaluator.evaluate(&[0.0; INPUT_SIZE]),
            Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::EvaluatorState,
                ..
            })
        ));
        assert_eq!(
            family
                .checkpoint_evaluator(overflowing_evaluator.as_ref())
                .expect("checkpoint after interpreter overflow"),
            before_overflow,
            "an instruction-produced non-finite result must roll back byte-for-byte"
        );
    }

    #[test]
    fn interpreter_opcode_bands_match_the_cpp_oracle() {
        let cases = [
            (2.05, 150, 3.0, "add"),
            (2.15, 150, 5.0, "subtract"),
            (2.25, 150, -4.0, "multiply"),
            (2.35, 50, 0.0, "conditional clear"),
            (2.45, 50, -4.0, "conditional negate"),
            (2.55, 50, 4.5, "conditional add-immediate"),
            (2.75, 50, -1.0, "conditional copy"),
        ];

        for (opcode, result_index, expected, name) in cases {
            let mut cells = [0.0; BRAIN_SIZE];
            cells[INPUT_SIZE] = opcode;
            cells[INPUT_SIZE + 1] = 0.25; // d1 = cell 50
            cells[INPUT_SIZE + 2] = 0.50; // d2 = cell 100, immediate = 0.5
            cells[INPUT_SIZE + 3] = 0.75; // d3 = cell 150
            cells[50] = 4.0;
            cells[100] = -1.0;
            cells[150] = 1.0;

            let mut brain = AssemblyBrain::from_cells(cells).expect("finite opcode fixture");
            let (_, scanned) = brain.tick_with_budget(&[0.0; INPUT_SIZE]);
            assert_eq!(scanned, ASSEMBLY_INSTRUCTION_BUDGET, "{name}");
            assert_eq!(
                brain.cells[result_index].to_bits(),
                f32::to_bits(expected),
                "opcode {opcode} ({name}) diverged from the C++ instruction semantics"
            );
        }
    }

    #[test]
    fn interpreter_scans_a_fixed_budget_and_unknown_cells_are_safe_noops() {
        let mut cells = [0.0; BRAIN_SIZE];
        cells[INPUT_SIZE] = 3.5;
        cells[INPUT_SIZE + 1] = -7.0;
        cells[BRAIN_SIZE - 1] = 0.75;
        let mut brain = AssemblyBrain::from_cells(cells).expect("valid finite program");
        let (outputs, scanned) = brain.tick_with_budget(&[0.0; INPUT_SIZE]);
        assert_eq!(scanned, ASSEMBLY_INSTRUCTION_BUDGET);
        assert_eq!(outputs[0].to_bits(), 0.75_f32.to_bits());
        assert_eq!(brain.cells[INPUT_SIZE], 3.5);
        assert_eq!(brain.cells[INPUT_SIZE + 1], -7.0);
    }
}
