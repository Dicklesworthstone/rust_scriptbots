use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use scriptbots_brain::MlpBrain;
use scriptbots_brain_neuro::{NeuroflowBrain, NeuroflowBrainConfig};
use scriptbots_core::{
    AgentData, AgentId, AnalyticsStride, BrainAdapterIdentityV1, BrainBatchArchitectureKey,
    BrainBatchEvaluator, BrainEnvelopeKind, BrainEvaluator, BrainEvaluatorStateEnvelope,
    BrainFamilyCodec, BrainFamilyId, BrainGenomeEnvelope, BrainGenomeMaterial, BrainInspection,
    BrainInspectionError, BrainInspectionSnapshot, BrainProtocolError,
    DYNAMIC_WORLD_SNAPSHOT_SCHEMA, DynamicWorldSnapshot, Generation, INPUT_SIZE, MutationRates,
    OUTPUT_SIZE, OffspringStatePolicy, Position, RandomStream, RuleBasedMapGenerator,
    ScriptBotsConfig, TerrainKind, TileSpec, TilesetSpec, Velocity, WORLD_STEP_PROFILE_SCHEMA,
    WorldState, WorldStepProfiler, WorldStepStage,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DENSE_BENCH_FAMILY_ID: &str = "bench-dense-25x16x9";
const DENSE_BENCH_KIND: &str = "bench.dense.25x16x9";
const DENSE_BENCH_HIDDEN: usize = 16;
const DENSE_BENCH_INPUT_WIRE: u16 = 25;
const DENSE_BENCH_HIDDEN_WIRE: u16 = 16;
const DENSE_BENCH_OUTPUT_WIRE: u16 = 9;
const DENSE_BENCH_GENOME_SCHEMA: u32 = 1;
const DENSE_BENCH_GENOME_CODEC: u16 = 1;
const DENSE_BENCH_STATE_SCHEMA: u32 = 1;
const DENSE_BENCH_STATE_CODEC: u16 = 1;
const DENSE_BENCH_GENOME_MAGIC: [u8; 4] = *b"DB25";
const DENSE_BENCH_STATE_MAGIC: [u8; 4] = *b"DBST";
const DENSE_BENCH_INPUT_HIDDEN: usize = INPUT_SIZE * DENSE_BENCH_HIDDEN;
const DENSE_BENCH_HIDDEN_OUTPUT: usize = DENSE_BENCH_HIDDEN * OUTPUT_SIZE;
const DENSE_BENCH_PARAMETER_COUNT: usize =
    DENSE_BENCH_INPUT_HIDDEN + DENSE_BENCH_HIDDEN + DENSE_BENCH_HIDDEN_OUTPUT + OUTPUT_SIZE;
const DENSE_BENCH_GENOME_HEADER_BYTES: usize = DENSE_BENCH_GENOME_MAGIC.len() + 3 * 2;
const DENSE_BENCH_GENOME_PAYLOAD_BYTES: usize =
    DENSE_BENCH_GENOME_HEADER_BYTES + DENSE_BENCH_PARAMETER_COUNT * 4;
const DENSE_BENCH_STATE_PAYLOAD_BYTES: usize = DENSE_BENCH_STATE_MAGIC.len() + blake3::OUT_LEN + 8;

#[derive(Clone, Copy)]
enum DenseBenchMode {
    Scalar,
    Batch,
}

impl DenseBenchMode {
    const fn batch_enabled(self) -> bool {
        matches!(self, Self::Batch)
    }
}

#[derive(Clone)]
struct DenseBenchParameters {
    input_hidden: Vec<f32>,
    hidden_bias: [f32; DENSE_BENCH_HIDDEN],
    hidden_output: Vec<f32>,
    output_bias: [f32; OUTPUT_SIZE],
}

impl DenseBenchParameters {
    fn random(rng: &mut dyn RandomStream) -> Self {
        let mut sample = || {
            let upper = u16::try_from(rng.next_u32() >> 16)
                .expect("the upper half of a u32 always fits u16");
            (f32::from(upper) / f32::from(u16::MAX) - 0.5) * 0.5
        };
        Self {
            input_hidden: (0..DENSE_BENCH_INPUT_HIDDEN).map(|_| sample()).collect(),
            hidden_bias: std::array::from_fn(|_| sample()),
            hidden_output: (0..DENSE_BENCH_HIDDEN_OUTPUT).map(|_| sample()).collect(),
            output_bias: std::array::from_fn(|_| sample()),
        }
    }
}

struct DenseBenchFamily {
    family_id: BrainFamilyId,
    mode: DenseBenchMode,
    batch_calls: Option<Arc<AtomicUsize>>,
}

impl DenseBenchFamily {
    fn new(mode: DenseBenchMode, batch_calls: Option<Arc<AtomicUsize>>) -> Self {
        Self {
            family_id: BrainFamilyId::new(DENSE_BENCH_FAMILY_ID)
                .expect("the benchmark family identifier is canonical"),
            mode,
            batch_calls,
        }
    }

    fn invalid(&self, kind: BrainEnvelopeKind, detail: impl Into<String>) -> BrainProtocolError {
        dense_bench_invalid(&self.family_id, kind, detail)
    }

    fn encode_genome(parameters: &DenseBenchParameters) -> Vec<u8> {
        debug_assert_eq!(INPUT_SIZE, usize::from(DENSE_BENCH_INPUT_WIRE));
        debug_assert_eq!(DENSE_BENCH_HIDDEN, usize::from(DENSE_BENCH_HIDDEN_WIRE));
        debug_assert_eq!(OUTPUT_SIZE, usize::from(DENSE_BENCH_OUTPUT_WIRE));
        let mut payload = Vec::with_capacity(DENSE_BENCH_GENOME_PAYLOAD_BYTES);
        payload.extend_from_slice(&DENSE_BENCH_GENOME_MAGIC);
        payload.extend_from_slice(&DENSE_BENCH_INPUT_WIRE.to_le_bytes());
        payload.extend_from_slice(&DENSE_BENCH_HIDDEN_WIRE.to_le_bytes());
        payload.extend_from_slice(&DENSE_BENCH_OUTPUT_WIRE.to_le_bytes());
        for value in &parameters.input_hidden {
            payload.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        for value in parameters.hidden_bias {
            payload.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        for value in &parameters.hidden_output {
            payload.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        for value in parameters.output_bias {
            payload.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        debug_assert_eq!(payload.len(), DENSE_BENCH_GENOME_PAYLOAD_BYTES);
        payload
    }

    fn decode_genome(
        &self,
        genome: &BrainGenomeEnvelope,
    ) -> Result<DenseBenchParameters, BrainProtocolError> {
        genome.require_protocol(
            &self.family_id,
            DENSE_BENCH_GENOME_SCHEMA,
            DENSE_BENCH_GENOME_CODEC,
        )?;
        let payload = genome.payload();
        if payload.len() != DENSE_BENCH_GENOME_PAYLOAD_BYTES {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                format!(
                    "dense benchmark genome requires exactly {DENSE_BENCH_GENOME_PAYLOAD_BYTES} bytes, found {}",
                    payload.len()
                ),
            ));
        }
        let mut cursor = 0;
        let magic = dense_bench_take::<4>(payload, &mut cursor)
            .ok_or_else(|| self.invalid(BrainEnvelopeKind::Genome, "truncated genome magic"))?;
        let inputs =
            u16::from_le_bytes(dense_bench_take::<2>(payload, &mut cursor).ok_or_else(|| {
                self.invalid(BrainEnvelopeKind::Genome, "truncated input dimension")
            })?);
        let hidden =
            u16::from_le_bytes(dense_bench_take::<2>(payload, &mut cursor).ok_or_else(|| {
                self.invalid(BrainEnvelopeKind::Genome, "truncated hidden dimension")
            })?);
        let outputs =
            u16::from_le_bytes(dense_bench_take::<2>(payload, &mut cursor).ok_or_else(|| {
                self.invalid(BrainEnvelopeKind::Genome, "truncated output dimension")
            })?);
        if magic != DENSE_BENCH_GENOME_MAGIC
            || inputs != DENSE_BENCH_INPUT_WIRE
            || hidden != DENSE_BENCH_HIDDEN_WIRE
            || outputs != DENSE_BENCH_OUTPUT_WIRE
        {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                format!(
                    "unsupported dense benchmark header: magic={magic:?}, inputs={inputs}, hidden={hidden}, outputs={outputs}"
                ),
            ));
        }

        let mut parameters = Vec::with_capacity(DENSE_BENCH_PARAMETER_COUNT);
        for index in 0..DENSE_BENCH_PARAMETER_COUNT {
            let bits = dense_bench_take::<4>(payload, &mut cursor).ok_or_else(|| {
                self.invalid(
                    BrainEnvelopeKind::Genome,
                    format!("truncated dense parameter {index}"),
                )
            })?;
            let value = f32::from_bits(u32::from_le_bytes(bits));
            if !value.is_finite() {
                return Err(self.invalid(
                    BrainEnvelopeKind::Genome,
                    format!("dense parameter {index} is not finite"),
                ));
            }
            parameters.push(value);
        }
        if cursor != payload.len() {
            return Err(self.invalid(
                BrainEnvelopeKind::Genome,
                "dense benchmark genome contains trailing bytes",
            ));
        }

        let mut offset = 0;
        let input_hidden = parameters[offset..offset + DENSE_BENCH_INPUT_HIDDEN].to_vec();
        offset += DENSE_BENCH_INPUT_HIDDEN;
        let hidden_bias = parameters[offset..offset + DENSE_BENCH_HIDDEN]
            .try_into()
            .expect("the validated dense hidden-bias slice has exact length");
        offset += DENSE_BENCH_HIDDEN;
        let hidden_output = parameters[offset..offset + DENSE_BENCH_HIDDEN_OUTPUT].to_vec();
        offset += DENSE_BENCH_HIDDEN_OUTPUT;
        let output_bias = parameters[offset..offset + OUTPUT_SIZE]
            .try_into()
            .expect("the validated dense output-bias slice has exact length");
        offset += OUTPUT_SIZE;
        debug_assert_eq!(offset, parameters.len());
        Ok(DenseBenchParameters {
            input_hidden,
            hidden_bias,
            hidden_output,
            output_bias,
        })
    }

    fn state(
        &self,
        genome: &BrainGenomeEnvelope,
        evaluations: u64,
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.validate_genome(genome)?;
        dense_bench_state_envelope(
            &self.family_id,
            *genome.material_hash().as_bytes(),
            evaluations,
        )
    }

    fn decode_state(
        &self,
        state: &BrainEvaluatorStateEnvelope,
    ) -> Result<([u8; blake3::OUT_LEN], u64), BrainProtocolError> {
        state.require_protocol(
            &self.family_id,
            DENSE_BENCH_STATE_SCHEMA,
            DENSE_BENCH_STATE_CODEC,
        )?;
        let payload = state.payload();
        if payload.len() != DENSE_BENCH_STATE_PAYLOAD_BYTES {
            return Err(self.invalid(
                BrainEnvelopeKind::EvaluatorState,
                format!(
                    "dense benchmark state requires exactly {DENSE_BENCH_STATE_PAYLOAD_BYTES} bytes, found {}",
                    payload.len()
                ),
            ));
        }
        let mut cursor = 0;
        let magic = dense_bench_take::<4>(payload, &mut cursor).ok_or_else(|| {
            self.invalid(
                BrainEnvelopeKind::EvaluatorState,
                "truncated evaluator-state magic",
            )
        })?;
        let genome_digest = dense_bench_take::<{ blake3::OUT_LEN }>(payload, &mut cursor)
            .ok_or_else(|| {
                self.invalid(
                    BrainEnvelopeKind::EvaluatorState,
                    "truncated evaluator-state genome digest",
                )
            })?;
        let evaluations =
            u64::from_le_bytes(dense_bench_take::<8>(payload, &mut cursor).ok_or_else(|| {
                self.invalid(
                    BrainEnvelopeKind::EvaluatorState,
                    "truncated evaluator-state counter",
                )
            })?);
        if magic != DENSE_BENCH_STATE_MAGIC || cursor != payload.len() {
            return Err(self.invalid(
                BrainEnvelopeKind::EvaluatorState,
                "unsupported dense benchmark evaluator-state header",
            ));
        }
        Ok((genome_digest, evaluations))
    }
}

impl BrainFamilyCodec for DenseBenchFamily {
    fn family_id(&self) -> &BrainFamilyId {
        &self.family_id
    }

    fn adapter_identity(&self) -> BrainAdapterIdentityV1 {
        BrainAdapterIdentityV1::from_semantic_descriptor(
            &self.family_id,
            1,
            b"scriptbots.bench-dense-25x16x9.adapter-semantics.v1",
        )
    }

    fn random_genome_material(
        &self,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        BrainGenomeMaterial::new(
            DENSE_BENCH_GENOME_SCHEMA,
            DENSE_BENCH_GENOME_CODEC,
            Self::encode_genome(&DenseBenchParameters::random(rng)),
        )
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
        _rates: MutationRates,
        _rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        self.validate_genome(genome)?;
        BrainGenomeMaterial::new(
            DENSE_BENCH_GENOME_SCHEMA,
            DENSE_BENCH_GENOME_CODEC,
            genome.payload().to_vec(),
        )
    }

    fn crossover_genomes_material(
        &self,
        left: &BrainGenomeEnvelope,
        right: &BrainGenomeEnvelope,
        _rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        self.validate_genome(left)?;
        self.validate_genome(right)?;
        BrainGenomeMaterial::new(
            DENSE_BENCH_GENOME_SCHEMA,
            DENSE_BENCH_GENOME_CODEC,
            left.payload().to_vec(),
        )
    }

    fn initial_state(
        &self,
        genome: &BrainGenomeEnvelope,
        _rng: &mut dyn RandomStream,
    ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.state(genome, 0)
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
        let parameters = self.decode_genome(genome)?;
        let (state_genome, evaluations) = self.decode_state(state)?;
        let genome_digest = *genome.material_hash().as_bytes();
        if state_genome != genome_digest {
            return Err(self.invalid(
                BrainEnvelopeKind::EvaluatorState,
                "dense evaluator state belongs to a different genome",
            ));
        }
        Ok(Box::new(DenseBenchEvaluator {
            family_id: self.family_id.clone(),
            genome_digest,
            parameters,
            evaluations,
        }))
    }

    fn batch_architecture_key(
        &self,
        genome: &BrainGenomeEnvelope,
    ) -> Result<Option<BrainBatchArchitectureKey>, BrainProtocolError> {
        if !self.mode.batch_enabled() {
            return Ok(None);
        }
        self.validate_genome(genome)?;
        BrainBatchArchitectureKey::new(b"dense-f32-25x16x9-distinct-lanes-v1".to_vec()).map(Some)
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
        if !self.mode.batch_enabled() {
            return Ok(None);
        }
        let evaluators = genomes
            .iter()
            .zip(states)
            .map(|(genome, state)| {
                let parameters = self.decode_genome(genome)?;
                let (state_genome, evaluations) = self.decode_state(state)?;
                let genome_digest = *genome.material_hash().as_bytes();
                if state_genome != genome_digest {
                    return Err(self.invalid(
                        BrainEnvelopeKind::EvaluatorState,
                        "dense batch lane state belongs to a different genome",
                    ));
                }
                Ok(DenseBenchEvaluator {
                    family_id: self.family_id.clone(),
                    genome_digest,
                    parameters,
                    evaluations,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(batch_calls) = &self.batch_calls {
            batch_calls.fetch_add(1, AtomicOrdering::Relaxed);
        }
        Ok(Some(Box::new(DenseBenchBatchEvaluator {
            family_id: self.family_id.clone(),
            evaluators,
        })))
    }
}

struct DenseBenchEvaluator {
    family_id: BrainFamilyId,
    genome_digest: [u8; blake3::OUT_LEN],
    parameters: DenseBenchParameters,
    evaluations: u64,
}

impl DenseBenchEvaluator {
    fn evaluate_dense(
        &mut self,
        sensors: &[f32; INPUT_SIZE],
    ) -> Result<[f32; OUTPUT_SIZE], BrainProtocolError> {
        if sensors.iter().any(|value| !value.is_finite()) {
            return Err(dense_bench_invalid(
                &self.family_id,
                BrainEnvelopeKind::EvaluatorState,
                "dense benchmark sensor input contains a non-finite value",
            ));
        }
        let next_evaluations = self.evaluations.checked_add(1).ok_or_else(|| {
            dense_bench_invalid(
                &self.family_id,
                BrainEnvelopeKind::EvaluatorState,
                "dense benchmark evaluation counter is exhausted",
            )
        })?;
        let outputs = dense_bench_forward(&self.parameters, self.evaluations, sensors);
        if outputs.iter().any(|value| !value.is_finite()) {
            return Err(dense_bench_invalid(
                &self.family_id,
                BrainEnvelopeKind::EvaluatorState,
                "dense benchmark evaluation produced a non-finite output",
            ));
        }
        self.evaluations = next_evaluations;
        Ok(outputs)
    }
}

impl BrainEvaluator for DenseBenchEvaluator {
    fn family_id(&self) -> &BrainFamilyId {
        &self.family_id
    }

    fn evaluate(
        &mut self,
        sensors: &[f32; INPUT_SIZE],
    ) -> Result<[f32; OUTPUT_SIZE], BrainProtocolError> {
        self.evaluate_dense(sensors)
    }

    fn inspect(
        &self,
        _request: BrainInspection,
    ) -> Result<Option<BrainInspectionSnapshot>, BrainInspectionError> {
        Ok(None)
    }

    fn checkpoint_state(&self) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        dense_bench_state_envelope(&self.family_id, self.genome_digest, self.evaluations)
    }
}

struct DenseBenchBatchEvaluator {
    family_id: BrainFamilyId,
    // Each lane retains independently decoded weights and state. This measures the complete
    // protocol-batch roundtrip, not a shared-weight shortcut or an optimized matrix kernel.
    evaluators: Vec<DenseBenchEvaluator>,
}

impl BrainBatchEvaluator for DenseBenchBatchEvaluator {
    fn family_id(&self) -> &BrainFamilyId {
        &self.family_id
    }

    fn evaluate_batch(
        &mut self,
        sensors: &[[f32; INPUT_SIZE]],
        outputs: &mut [[f32; OUTPUT_SIZE]],
    ) -> Result<(), BrainProtocolError> {
        if self.evaluators.len() != sensors.len() || sensors.len() != outputs.len() {
            return Err(BrainProtocolError::BatchCardinalityMismatch {
                evaluators: self.evaluators.len(),
                inputs: sensors.len(),
                outputs: outputs.len(),
            });
        }
        for ((evaluator, input), output) in self
            .evaluators
            .iter_mut()
            .zip(sensors)
            .zip(outputs.iter_mut())
        {
            *output = evaluator.evaluate_dense(input)?;
        }
        Ok(())
    }

    fn checkpoint_states(&self) -> Result<Vec<BrainEvaluatorStateEnvelope>, BrainProtocolError> {
        self.evaluators
            .iter()
            .map(BrainEvaluator::checkpoint_state)
            .collect()
    }
}

fn dense_bench_invalid(
    family_id: &BrainFamilyId,
    kind: BrainEnvelopeKind,
    detail: impl Into<String>,
) -> BrainProtocolError {
    BrainProtocolError::InvalidPayload {
        kind,
        family_id: family_id.clone(),
        detail: detail.into(),
    }
}

fn dense_bench_take<const N: usize>(payload: &[u8], cursor: &mut usize) -> Option<[u8; N]> {
    let end = cursor.checked_add(N)?;
    let bytes = payload.get(*cursor..end)?;
    *cursor = end;
    bytes.try_into().ok()
}

fn dense_bench_state_envelope(
    family_id: &BrainFamilyId,
    genome_digest: [u8; blake3::OUT_LEN],
    evaluations: u64,
) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
    let mut payload = Vec::with_capacity(DENSE_BENCH_STATE_PAYLOAD_BYTES);
    payload.extend_from_slice(&DENSE_BENCH_STATE_MAGIC);
    payload.extend_from_slice(&genome_digest);
    payload.extend_from_slice(&evaluations.to_le_bytes());
    BrainEvaluatorStateEnvelope::new(
        family_id.clone(),
        DENSE_BENCH_STATE_SCHEMA,
        DENSE_BENCH_STATE_CODEC,
        payload,
    )
}

fn dense_bench_forward(
    parameters: &DenseBenchParameters,
    evaluations: u64,
    sensors: &[f32; INPUT_SIZE],
) -> [f32; OUTPUT_SIZE] {
    let mut hidden = [0.0; DENSE_BENCH_HIDDEN];
    for (hidden_index, hidden_output) in hidden.iter_mut().enumerate() {
        let mut accumulator = parameters.hidden_bias[hidden_index];
        let row = hidden_index * INPUT_SIZE;
        for (sensor, weight) in sensors
            .iter()
            .zip(&parameters.input_hidden[row..row + INPUT_SIZE])
        {
            accumulator += sensor * weight;
        }
        *hidden_output = accumulator.tanh();
    }

    let phase_bits =
        u16::try_from(evaluations & 1023).expect("the bounded evaluation phase fits u16");
    let state_bias = (f32::from(phase_bits) - 512.0) * 0.000_001;
    let mut outputs = [0.0; OUTPUT_SIZE];
    for (output_index, output) in outputs.iter_mut().enumerate() {
        let mut accumulator = parameters.output_bias[output_index] + state_bias;
        let row = output_index * DENSE_BENCH_HIDDEN;
        for (hidden_value, weight) in hidden
            .iter()
            .zip(&parameters.hidden_output[row..row + DENSE_BENCH_HIDDEN])
        {
            accumulator += hidden_value * weight;
        }
        *output = accumulator.tanh();
    }
    outputs
}

struct DenseBenchWorld {
    world: WorldState,
    agents: Vec<AgentId>,
    batch_calls: Option<Arc<AtomicUsize>>,
}

fn dense_bench_config() -> ScriptBotsConfig {
    ScriptBotsConfig {
        world_width: 1_000,
        world_height: 1_000,
        food_cell_size: 20,
        initial_food: 0.0,
        rng_seed: Some(0x25_16_09_ba_7c_2026),
        chart_flush_interval: 0,
        food_respawn_interval: 0,
        food_respawn_amount: 0.0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        sense_radius: 1.0,
        sense_max_neighbors: 1.0,
        bot_speed: 0.0,
        bot_radius: 1.0,
        boost_multiplier: 1.0,
        spike_growth_rate: 0.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        metabolism_ramp_rate: 0.0,
        metabolism_boost_penalty: 0.0,
        temperature_discomfort_rate: 0.0,
        food_intake_rate: 0.0,
        food_waste_rate: 0.0,
        food_sharing_rate: 0.0,
        food_transfer_rate: 0.0,
        reproduction_attempt_chance: 0.0,
        reproduction_food_bonus: 0.0,
        reproduction_fertility_bonus: 0.0,
        reproduction_partner_chance: 0.0,
        reproduction_meta_mutation_chance: 0.0,
        aging_tick_interval: u32::MAX,
        aging_health_decay_rate: 0.0,
        aging_health_decay_max: 0.0,
        aging_energy_penalty_rate: 0.0,
        carcass_distribution_radius: 0.0,
        carcass_health_reward: 0.0,
        carcass_reproduction_reward: 0.0,
        carcass_energy_share_rate: 0.0,
        carcass_indicator_scale: 0.0,
        closed: true,
        population_minimum: 0,
        population_spawn_interval: 0,
        population_crossover_chance: 0.0,
        spike_damage: 0.0,
        spike_energy_cost: 0.0,
        spike_speed_damage_bonus: 0.0,
        spike_length_damage_bonus: 0.0,
        history_capacity: 1,
        narrative_interval: 0,
        narrative_capacity: 0,
        persistence_interval: 0,
        analytics_stride: AnalyticsStride {
            macro_metrics: 0,
            behavior_metrics: 0,
            lifecycle_events: 0,
        },
        ..ScriptBotsConfig::default()
    }
}

fn build_dense_bench_world(
    agent_count: usize,
    mode: DenseBenchMode,
    probe_batch_calls: bool,
) -> DenseBenchWorld {
    let mut world = WorldState::new(dense_bench_config()).expect("dense benchmark world");
    let batch_calls = probe_batch_calls.then(|| Arc::new(AtomicUsize::new(0)));
    let family_key = world
        .register_brain_family(
            DENSE_BENCH_KIND,
            Box::new(DenseBenchFamily::new(mode, batch_calls.clone())),
        )
        .expect("register dense benchmark family");
    let mut agents = Vec::with_capacity(agent_count);
    for ordinal in 0..agent_count {
        let column = u16::try_from(ordinal % 100).expect("benchmark column fits u16");
        let row = u16::try_from(ordinal / 100).expect("benchmark row fits u16");
        let agent = world
            .try_spawn_agent(AgentData::new(
                Position::new(f32::from(column) * 10.0 + 0.5, f32::from(row) * 10.0 + 0.5),
                Velocity::default(),
                0.0,
                1.0,
                [0.25, 0.5, 0.75],
                0.0,
                false,
                0,
                Generation(0),
            ))
            .expect("dense benchmark agent is finite");
        assert!(
            world
                .bind_agent_brain(agent, family_key)
                .expect("bind dense benchmark brain")
        );
        agents.push(agent);
    }
    DenseBenchWorld {
        world,
        agents,
        batch_calls,
    }
}

fn dense_bench_genome_hashes(world: &WorldState, agents: &[AgentId]) -> Vec<[u8; 32]> {
    agents
        .iter()
        .map(|agent| {
            *world
                .agent_brain_genome(*agent)
                .expect("dense benchmark agent has a genome")
                .material_hash()
                .as_bytes()
        })
        .collect()
}

fn dense_bench_output_bits(world: &WorldState, agents: &[AgentId]) -> Vec<[u32; OUTPUT_SIZE]> {
    agents
        .iter()
        .map(|agent| {
            world
                .agent_runtime(*agent)
                .expect("dense benchmark agent has runtime")
                .outputs
                .map(f32::to_bits)
        })
        .collect()
}

fn dense_bench_states(world: &WorldState, agents: &[AgentId]) -> Vec<BrainEvaluatorStateEnvelope> {
    agents
        .iter()
        .map(|agent| {
            world
                .agent_brain_evaluator_state(*agent)
                .expect("checkpoint dense benchmark evaluator")
                .expect("dense benchmark agent has protocol state")
        })
        .collect()
}

fn assert_dense_bench_preflight(agent_count: usize) {
    let DenseBenchWorld {
        world: mut scalar,
        agents: scalar_agents,
        batch_calls: scalar_calls,
    } = build_dense_bench_world(agent_count, DenseBenchMode::Scalar, true);
    let DenseBenchWorld {
        world: mut batch,
        agents: batch_agents,
        batch_calls,
    } = build_dense_bench_world(agent_count, DenseBenchMode::Batch, true);

    let scalar_hashes = dense_bench_genome_hashes(&scalar, &scalar_agents);
    let batch_hashes = dense_bench_genome_hashes(&batch, &batch_agents);
    assert_eq!(
        scalar_hashes, batch_hashes,
        "matched scalar and batch worlds must carry the same ordered genomes"
    );
    assert_eq!(
        scalar_hashes.iter().copied().collect::<HashSet<_>>().len(),
        agent_count,
        "the benchmark cohort must carry distinct per-agent weights"
    );

    let mut scalar_profiler = WorldStepProfiler::default();
    let mut batch_profiler = WorldStepProfiler::default();
    scalar
        .step_profiled(&mut scalar_profiler)
        .expect("scalar dense benchmark preflight step");
    batch
        .step_profiled(&mut batch_profiler)
        .expect("batch dense benchmark preflight step");
    assert_eq!(
        scalar_calls
            .expect("scalar preflight probe")
            .load(AtomicOrdering::Relaxed),
        0,
        "the scalar benchmark family must never enter the batch hook"
    );
    assert_eq!(
        batch_calls
            .expect("batch preflight probe")
            .load(AtomicOrdering::Relaxed),
        1,
        "one homogeneous architecture must execute as one batch cohort"
    );
    assert_eq!(
        dense_bench_output_bits(&scalar, &scalar_agents),
        dense_bench_output_bits(&batch, &batch_agents),
        "batch outputs must be bit-identical to scalar outputs"
    );
    assert_eq!(
        dense_bench_states(&scalar, &scalar_agents),
        dense_bench_states(&batch, &batch_agents),
        "batch evaluator states must be bit-identical to scalar states"
    );
    assert_eq!(
        scalar
            .world_digest_v1()
            .expect("scalar dense benchmark digest"),
        batch
            .world_digest_v1()
            .expect("batch dense benchmark digest"),
        "batch and scalar transitions must produce the same science digest"
    );
}

fn bench_brain_protocol_cohorts(c: &mut Criterion) {
    let samples = env::var("SB_BATCH_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 10)
        .unwrap_or(10);
    let warmup_seconds = env::var("SB_BATCH_BENCH_WARMUP_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1);
    let measurement_seconds = env::var("SB_BATCH_BENCH_MEASURE_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(5);

    let mut group = c.benchmark_group("brain_protocol_cohort");
    group.sample_size(samples);
    group.warm_up_time(Duration::from_secs(warmup_seconds));
    group.measurement_time(Duration::from_secs(measurement_seconds));

    for agent_count in [2_000_usize, 5_000, 10_000] {
        assert_dense_bench_preflight(agent_count);
        group.throughput(Throughput::Elements(
            u64::try_from(agent_count).expect("benchmark population fits u64"),
        ));

        {
            let DenseBenchWorld {
                world: mut scalar, ..
            } = build_dense_bench_world(agent_count, DenseBenchMode::Scalar, false);
            group.bench_with_input(
                BenchmarkId::new("scalar_live", agent_count),
                &agent_count,
                |bencher, _| {
                    let mut profiler = WorldStepProfiler::default();
                    bencher.iter_custom(|iterations| {
                        let mut measured = Duration::ZERO;
                        for _ in 0..iterations {
                            scalar
                                .step_profiled(&mut profiler)
                                .expect("profile scalar dense brain stage");
                            let elapsed_ns = profiler
                                .latest()
                                .and_then(|profile| profile.elapsed_ns(WorldStepStage::Brains))
                                .expect("profile includes the scalar brain stage");
                            measured = measured.saturating_add(Duration::from_nanos(elapsed_ns));
                            black_box(scalar.tick());
                        }
                        measured
                    });
                },
            );
        }

        {
            let DenseBenchWorld {
                world: mut batch, ..
            } = build_dense_bench_world(agent_count, DenseBenchMode::Batch, false);
            group.bench_with_input(
                BenchmarkId::new("batch_roundtrip", agent_count),
                &agent_count,
                |bencher, _| {
                    let mut profiler = WorldStepProfiler::default();
                    bencher.iter_custom(|iterations| {
                        let mut measured = Duration::ZERO;
                        for _ in 0..iterations {
                            batch
                                .step_profiled(&mut profiler)
                                .expect("profile batch dense brain stage");
                            let elapsed_ns = profiler
                                .latest()
                                .and_then(|profile| profile.elapsed_ns(WorldStepStage::Brains))
                                .expect("profile includes the batch brain stage");
                            measured = measured.saturating_add(Duration::from_nanos(elapsed_ns));
                            black_box(batch.tick());
                        }
                        measured
                    });
                },
            );
        }
    }
    group.finish();
}

#[allow(clippy::field_reassign_with_default)]
fn bench_world_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("world_step");
    // Increase iteration time for more stable results and allow env overrides
    let samples: usize = std::env::var("SB_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(30);
    let warm: u64 = std::env::var("SB_BENCH_WARMUP_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(2);
    let measure: u64 = std::env::var("SB_BENCH_MEASURE_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(10);
    group.sample_size(samples);
    group.warm_up_time(Duration::from_secs(warm));
    group.measurement_time(Duration::from_secs(measure));
    // Steps per bench iteration (can override via SB_BENCH_STEPS)
    let steps: usize = std::env::var("SB_BENCH_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(64);
    let agents_list: Vec<usize> = std::env::var("SB_BENCH_AGENTS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![2000_usize, 5000, 10000]);
    for &agents in &agents_list {
        group.bench_function(format!("steps{}_agents{}_ticks", steps, agents), |b| {
            b.iter_batched(
                || {
                    let mut config = ScriptBotsConfig::default();
                    // Smaller world to stress neighbor density
                    config.world_width = 800;
                    config.world_height = 800;
                    config.food_cell_size = 20;
                    config.rng_seed = Some(0xBEEFu64);
                    config.history_capacity = 0;
                    // Ensure all validated fields are set to sane values
                    config.food_max = 2.0;
                    config.food_growth_rate = 0.02;
                    config.food_decay_rate = 0.0;
                    config.food_diffusion_rate = 0.1;
                    config.food_waste_rate = 0.0;
                    config.reproduction_attempt_interval = 1;
                    config.reproduction_attempt_chance = 0.2;
                    config.reproduction_rate_herbivore = 1.0;
                    config.reproduction_rate_carnivore = 1.0;
                    config.spike_radius = 4.0;
                    config.spike_damage = 0.0;
                    config.spike_energy_cost = 0.0;
                    config.spike_min_length = 0.0;
                    config.spike_alignment_cosine = 0.5;
                    config.spike_speed_damage_bonus = 0.0;
                    config.spike_length_damage_bonus = 0.0;
                    config.carnivore_threshold = 0.5;
                    config.history_capacity = 1;
                    config.metabolism_drain = 0.001;
                    config.movement_drain = 0.001;
                    config.metabolism_ramp_floor = 0.0;
                    config.metabolism_ramp_rate = 0.0;
                    config.temperature_discomfort_rate = 0.0;
                    config.temperature_comfort_band = 0.2;
                    config.temperature_gradient_exponent = 1.0;
                    config.temperature_discomfort_exponent = 1.0;
                    config.aging_tick_interval = 1;
                    config.aging_health_decay_rate = 0.0;
                    config.aging_health_decay_max = 0.0;
                    config.aging_energy_penalty_rate = 0.0;
                    config.carcass_distribution_radius = 0.0;
                    config.carcass_health_reward = 0.0;
                    config.carcass_reproduction_reward = 0.0;
                    config.carcass_energy_share_rate = 0.0;
                    config.carcass_indicator_scale = 0.0;
                    config.boost_multiplier = 1.2;
                    config.spike_growth_rate = 0.01;
                    config.population_spawn_count = 1;
                    config.population_crossover_chance = 0.0;
                    config.sense_radius = 20.0;
                    config.sense_max_neighbors = 16.0;
                    config.bot_radius = 2.0;
                    config.bot_speed = 1.0;
                    let mut world = WorldState::new(config).expect("world");
                    for seed in 0..agents as u32 {
                        let pos_x = (seed % 800) as f32;
                        let pos_y = ((seed * 37) % 800) as f32;
                        let data = scriptbots_core::AgentData::new(
                            scriptbots_core::Position::new(pos_x, pos_y),
                            scriptbots_core::Velocity::default(),
                            0.0,
                            1.0,
                            [0.5, 0.5, 0.5],
                            0.0,
                            false,
                            0,
                            scriptbots_core::Generation(0),
                        );
                        world
                            .try_spawn_agent(data)
                            .expect("benchmark agent is finite");
                    }
                    world
                },
                |mut world| {
                    for _ in 0..steps {
                        world
                            .step()
                            .expect("benchmark world should accept each simulation step");
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_hydrology_map_generation(c: &mut Criterion) {
    let generator = RuleBasedMapGenerator::new(TilesetSpec {
        id: "hydrology-benchmark".to_owned(),
        label: None,
        description: None,
        tiles: vec![TileSpec {
            id: "channel".to_owned(),
            label: None,
            weight: 1,
            terrain_kind: TerrainKind::Grass,
            fertility_bias: Some(0.5),
            temperature_bias: Some(0.5),
            elevation: Some(0.5),
            moisture: Some(0.5),
            accent: Some(0.5),
            palette_index: Some(0),
            permeability: Some(0.35),
            runoff_bias: Some(0.2),
            basin_rank: Some(0.55),
            channel_priority: Some(0.4),
            swim_cost: Some(1.2),
        }],
        adjacency: Vec::new(),
    })
    .expect("compile deterministic one-tile benchmark tileset");

    let mut group = c.benchmark_group("hydrology_map_generation");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("single_tile_128x128", |b| {
        b.iter(|| {
            generator
                .generate(128, 128, 1, 0x5eed_cafe)
                .expect("generate deterministic hydrology benchmark map")
        });
    });
    group.finish();
}

const PERF_GATE_SCHEMA: &str = "scriptbots.perf-gate.v3";
const PERF_VERDICT_SCHEMA: &str = "scriptbots.perf-verdict.v1";
const PERF_SCENARIO_CONTRACT: &str = "scriptbots.perf-scenario.v3";
const PERF_SEED: u64 = 0x5eed_ba5e;
const PERF_WARMUPS: usize = 3;
const PERF_REPETITIONS: usize = 5;
const PERF_TICKS: usize = 200;
const PERF_WINDOW_TICKS: usize = 20;
const PERF_SNAPSHOT_SAMPLES_PER_TICK: usize = 5;
const MAX_CV_PCT: f64 = 5.0;
const MAX_TPS_REGRESSION_PCT: f64 = 10.0;
const MIN_TPS_1K: f64 = 60.0;
const MAX_SNAPSHOT_P95_NS_1K: u64 = 4_000_000;
const MEMORY_CLASS_BUCKET_MIB: u64 = 256;

type GateResult<T> = Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateMode {
    Short,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateBrainFamily {
    Mlp,
    Neuroflow,
}

impl GateBrainFamily {
    const ALL: [Self; 2] = [Self::Mlp, Self::Neuroflow];

    const fn label(self) -> &'static str {
        match self {
            Self::Mlp => MlpBrain::KIND.as_str(),
            Self::Neuroflow => NeuroflowBrain::KIND.as_str(),
        }
    }

    const fn scenario_slug(self) -> &'static str {
        match self {
            Self::Mlp => "mlp-baseline",
            Self::Neuroflow => "ml-neuroflow",
        }
    }

    fn compiled_feature_set() -> String {
        let core_features = [
            ("parallel", cfg!(feature = "parallel")),
            ("simd_wide", cfg!(feature = "simd_wide")),
        ]
        .into_iter()
        .filter_map(|(feature, enabled)| enabled.then_some(feature))
        .collect::<Vec<_>>();
        let core = if core_features.is_empty() {
            "none".to_owned()
        } else {
            core_features.join(",")
        };
        format!("scriptbots-core/{core}+scriptbots-brain/mlp+scriptbots-brain-neuro/base")
    }

    fn register(self, world: &mut WorldState) -> GateResult<u64> {
        match self {
            Self::Mlp => Ok(world
                .brain_registry_mut()
                .map_err(|error| format!("failed to access brain registry: {error}"))?
                .register(self.label(), |rng| Ok(MlpBrain::runner(rng)))),
            Self::Neuroflow => {
                let config = NeuroflowBrainConfig::from_settings(&world.config().neuroflow);
                NeuroflowBrain::register(world, config)
                    .map_err(|error| format!("failed to register {}: {error}", self.label()))
            }
        }
    }
}

impl GateMode {
    fn parse(raw: &str) -> GateResult<Self> {
        match raw {
            "short" => Ok(Self::Short),
            "full" => Ok(Self::Full),
            _ => Err(format!(
                "unsupported perf-gate mode `{raw}`; expected short|full"
            )),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Full => "full",
        }
    }

    fn agent_counts(self) -> &'static [usize] {
        match self {
            Self::Short => &[1_000],
            Self::Full => &[1_000, 5_000],
        }
    }
}

#[derive(Debug)]
struct GateArgs {
    mode: GateMode,
    output_dir: PathBuf,
    baseline: Option<PathBuf>,
    record_baseline: bool,
    justification: Option<String>,
    ticks: usize,
    synthetic_sleep_us: u64,
    self_test: bool,
}

impl GateArgs {
    fn parse(args: &[String]) -> GateResult<Self> {
        let mut parsed = Self {
            mode: GateMode::Short,
            output_dir: PathBuf::new(),
            baseline: None,
            record_baseline: false,
            justification: None,
            ticks: PERF_TICKS,
            synthetic_sleep_us: 0,
            self_test: false,
        };
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--mode" => {
                    let raw = args
                        .get(index + 1)
                        .ok_or_else(|| "--mode requires short|full".to_owned())?;
                    parsed.mode = GateMode::parse(raw)?;
                    index += 2;
                }
                "--output-dir" => {
                    let raw = args
                        .get(index + 1)
                        .ok_or_else(|| "--output-dir requires a path".to_owned())?;
                    parsed.output_dir = PathBuf::from(raw);
                    index += 2;
                }
                "--baseline" => {
                    let raw = args
                        .get(index + 1)
                        .ok_or_else(|| "--baseline requires a path".to_owned())?;
                    parsed.baseline = Some(PathBuf::from(raw));
                    index += 2;
                }
                "--record-baseline" => {
                    parsed.record_baseline = true;
                    index += 1;
                }
                "--justification" => {
                    let raw = args
                        .get(index + 1)
                        .ok_or_else(|| "--justification requires text".to_owned())?;
                    parsed.justification = Some(raw.clone());
                    index += 2;
                }
                "--ticks" => {
                    let raw = args
                        .get(index + 1)
                        .ok_or_else(|| "--ticks requires a positive integer".to_owned())?;
                    parsed.ticks = raw
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --ticks `{raw}`: {error}"))?;
                    index += 2;
                }
                "--synthetic-sleep-us" => {
                    let raw = args.get(index + 1).ok_or_else(|| {
                        "--synthetic-sleep-us requires a non-negative integer".to_owned()
                    })?;
                    parsed.synthetic_sleep_us = raw.parse::<u64>().map_err(|error| {
                        format!("invalid --synthetic-sleep-us `{raw}`: {error}")
                    })?;
                    index += 2;
                }
                "--self-test" => {
                    parsed.self_test = true;
                    index += 1;
                }
                other => return Err(format!("unknown perf-gate argument `{other}`")),
            }
        }
        if parsed.self_test
            && (parsed.record_baseline
                || parsed.baseline.is_some()
                || parsed.justification.is_some()
                || parsed.synthetic_sleep_us != 0
                || parsed.ticks != PERF_TICKS)
        {
            return Err(
                "--self-test cannot be combined with baseline, tick, justification, or synthetic-delay options"
                    .to_owned(),
            );
        }
        if parsed.record_baseline && parsed.baseline.is_some() {
            return Err("--record-baseline cannot be combined with --baseline".to_owned());
        }
        if !parsed.self_test && parsed.ticks < 200 {
            return Err(format!(
                "--ticks must be at least 200 so each repetition has a defensible p95; got {}",
                parsed.ticks
            ));
        }
        if !parsed.ticks.is_multiple_of(PERF_WINDOW_TICKS) {
            return Err(format!(
                "--ticks must be divisible by the {PERF_WINDOW_TICKS}-tick window; got {}",
                parsed.ticks
            ));
        }
        if parsed.output_dir.as_os_str().is_empty() {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| format!("system clock precedes Unix epoch: {error}"))?
                .as_millis();
            parsed.output_dir = PathBuf::from(format!("tests/artifacts/perf/{timestamp}"));
        }
        let workspace_root = workspace_root()?;
        if parsed.output_dir.is_relative() {
            parsed.output_dir = workspace_root.join(&parsed.output_dir);
        }
        if let Some(baseline) = parsed.baseline.as_mut()
            && baseline.is_relative()
        {
            *baseline = workspace_root.join(&*baseline);
        }
        if parsed.record_baseline {
            if parsed.mode != GateMode::Full {
                return Err("baseline recording requires --mode full".to_owned());
            }
            if parsed.synthetic_sleep_us != 0 {
                return Err("a synthetic delay can never be blessed as a baseline".to_owned());
            }
            if parsed
                .justification
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
            {
                return Err("--record-baseline requires a non-empty --justification".to_owned());
            }
        }
        Ok(parsed)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct MachineClass {
    provider: String,
    runner_os: String,
    runner_arch: String,
    image_os: String,
    image_version: String,
    cpu_vendor: String,
    cpu_family: String,
    cpu_model: String,
    cpu_brand: String,
    logical_cpus: usize,
    cpu_quota: String,
    filesystem: String,
    kernel: String,
    memory: String,
    rust_release: String,
    rust_host: String,
    rustc_verbose: String,
    linker: String,
    cargo_config_hash: String,
    build_target: String,
    build_profile: String,
    thread_budget: usize,
    parallel_min_split: usize,
    rustflags: String,
}

fn validate_machine_class_identity(class: &MachineClass) -> GateResult<()> {
    if class.provider.trim().is_empty()
        || class.runner_os.trim().is_empty()
        || class.runner_arch.trim().is_empty()
        || class.cpu_vendor.trim().is_empty()
        || class.cpu_family.trim().is_empty()
        || class.cpu_model.trim().is_empty()
        || class.cpu_brand.trim().is_empty()
        || class.filesystem.trim().is_empty()
        || class.kernel.trim().is_empty()
        || class.memory.trim().is_empty()
        || class.rust_release.trim().is_empty()
        || class.rust_host.trim().is_empty()
        || class.rustc_verbose.trim().is_empty()
        || class.linker.trim().is_empty()
        || class.cargo_config_hash.trim().is_empty()
        || class.build_target.trim().is_empty()
        || class.build_profile.trim().is_empty()
        || class.logical_cpus == 0
        || class.thread_budget == 0
        || class.parallel_min_split == 0
    {
        return Err("machine class is missing required host identity".to_owned());
    }
    if class.provider != "local"
        && (class.image_os.trim().is_empty() || class.image_version.trim().is_empty())
    {
        return Err("hosted-runner machine class has no exact image identity".to_owned());
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct Fingerprint {
    machine_class_id: String,
    class: MachineClass,
    cpu_brand: String,
    kernel: String,
    memory: String,
    cargo_lock_git_blob: String,
    git_commit: String,
    git_dirty: bool,
    generated_unix_ms: u128,
    github_run_id: Option<String>,
    github_run_attempt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct GatePolicy {
    warmup_repetitions: usize,
    measured_repetitions: usize,
    ticks_per_repetition: usize,
    snapshot_samples_per_tick: usize,
    snapshot_samples_per_repetition: usize,
    median_window_ticks: usize,
    max_cv_pct: f64,
    max_tps_regression_pct: f64,
    min_tps_1k: f64,
    max_snapshot_p95_ns_1k: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerfArtifact {
    schema: String,
    scenario_contract: String,
    artifact_kind: String,
    mode: String,
    source_commit: String,
    baseline_justification: Option<String>,
    dynamic_snapshot_schema: String,
    world_step_profile_schema: String,
    synthetic_sleep_us: u64,
    fingerprint: Fingerprint,
    policy: GatePolicy,
    scenarios: Vec<ScenarioResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScenarioResult {
    id: String,
    agents: usize,
    brain_family: String,
    seed: u64,
    ticks_per_repetition: usize,
    feature_set: String,
    scenario_config_hash: String,
    initial_agent_count: usize,
    warmups: Vec<RepetitionResult>,
    measurements: Vec<RepetitionResult>,
    median_of_run_total_tps: f64,
    run_tps_cv_pct: f64,
    median_of_run_snapshot_p95_ns: u64,
    snapshot_p95_cv_pct: f64,
    stage_median_of_run_medians_ns: BTreeMap<String, Option<u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RepetitionResult {
    index: usize,
    window_elapsed_ns: Vec<u64>,
    window_tps: Vec<f64>,
    total_step_elapsed_ns: u64,
    total_tps: f64,
    median_window_tps: f64,
    snapshot_ns: Vec<u64>,
    snapshot_p50_ns: u64,
    snapshot_p95_ns: u64,
    snapshot_max_ns: u64,
    profiled_step_total_ns: Vec<u64>,
    stages: Vec<StageResult>,
    final_agent_count: usize,
    final_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StageResult {
    stage: String,
    raw_ns: Vec<Option<u64>>,
    executions: usize,
    median_ns: Option<u64>,
}

fn perf_config() -> ScriptBotsConfig {
    ScriptBotsConfig {
        world_width: 800,
        world_height: 800,
        food_cell_size: 20,
        rng_seed: Some(PERF_SEED),
        closed: true,
        population_minimum: 0,
        population_spawn_interval: 0,
        population_crossover_chance: 0.0,
        reproduction_attempt_interval: 1,
        reproduction_attempt_chance: 0.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        metabolism_ramp_floor: 0.0,
        metabolism_ramp_rate: 0.0,
        metabolism_boost_penalty: 0.0,
        temperature_discomfort_rate: 0.0,
        aging_health_decay_rate: 0.0,
        aging_health_decay_max: 0.0,
        aging_energy_penalty_rate: 0.0,
        spike_damage: 0.0,
        spike_energy_cost: 0.0,
        food_max: 2.0,
        food_growth_rate: 0.02,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.1,
        food_waste_rate: 0.0,
        history_capacity: 1,
        persistence_interval: 0,
        ..ScriptBotsConfig::default()
    }
}

fn scenario_config_hash(brain_family: GateBrainFamily) -> GateResult<String> {
    let world_config = perf_config();
    let neuroflow_config = (brain_family == GateBrainFamily::Neuroflow)
        .then(|| NeuroflowBrainConfig::from_settings(&world_config.neuroflow));
    let descriptor = (
        world_config,
        brain_family.label(),
        GateBrainFamily::compiled_feature_set(),
        neuroflow_config,
    );
    serde_json::to_vec(&descriptor)
        .map(|bytes| stable_hash(&bytes))
        .map_err(|error| format!("failed to serialize perf scenario contract: {error}"))
}

fn build_perf_world(agents: usize, brain_family: GateBrainFamily) -> GateResult<WorldState> {
    let mut world = WorldState::new(perf_config()).map_err(|error| error.to_string())?;
    let brain_key = brain_family.register(&mut world)?;
    for ordinal in 0..agents {
        let x = (ordinal % 800) as f32;
        let y = ((ordinal * 37) % 800) as f32;
        let agent = scriptbots_core::AgentData::new(
            scriptbots_core::Position::new(x, y),
            scriptbots_core::Velocity::default(),
            0.0,
            1.0,
            [0.5, 0.5, 0.5],
            0.0,
            false,
            0,
            scriptbots_core::Generation(0),
        );
        let id = world
            .try_spawn_agent(agent)
            .map_err(|error| format!("failed to spawn perf agent {ordinal}: {error}"))?;
        let bound = world
            .bind_agent_brain(id, brain_key)
            .map_err(|error| format!("failed to bind perf agent {ordinal}: {error}"))?;
        if !bound {
            return Err(format!(
                "brain registry key {brain_key} disappeared while binding agent {ordinal}"
            ));
        }
    }
    if world.agent_count() != agents {
        return Err(format!(
            "scenario requested {agents} agents but initialized {}",
            world.agent_count()
        ));
    }
    Ok(world)
}

fn run_repetition(
    agents: usize,
    brain_family: GateBrainFamily,
    ticks: usize,
    index: usize,
    synthetic_sleep_us: u64,
) -> GateResult<RepetitionResult> {
    let mut throughput_world = build_perf_world(agents, brain_family)?;
    let mut window_elapsed_ns = Vec::with_capacity(ticks / PERF_WINDOW_TICKS);
    for _ in 0..ticks / PERF_WINDOW_TICKS {
        let started = Instant::now();
        for _ in 0..PERF_WINDOW_TICKS {
            throughput_world
                .step()
                .map_err(|error| format!("pure TPS step failed: {error}"))?;
            if synthetic_sleep_us != 0 {
                thread::sleep(Duration::from_micros(synthetic_sleep_us));
            }
        }
        window_elapsed_ns.push(duration_ns(started.elapsed()));
    }
    let total_step_elapsed_ns = window_elapsed_ns
        .iter()
        .copied()
        .fold(0_u64, u64::saturating_add);
    let window_tps: Vec<f64> = window_elapsed_ns
        .iter()
        .map(|elapsed_ns| PERF_WINDOW_TICKS as f64 * 1_000_000_000.0 / *elapsed_ns as f64)
        .collect();
    let total_tps = ticks as f64 * 1_000_000_000.0 / total_step_elapsed_ns as f64;
    let median_window_tps = median_f64(&window_tps)?;
    let throughput_digest = throughput_world
        .characterization_digest_v0()
        .map_err(|error| format!("failed to digest throughput world: {error}"))?;

    let mut profiled_world = build_perf_world(agents, brain_family)?;
    let mut profiler = WorldStepProfiler::default();
    let snapshot_sample_capacity = ticks
        .checked_mul(PERF_SNAPSHOT_SAMPLES_PER_TICK)
        .ok_or_else(|| "snapshot sample count overflowed usize".to_owned())?;
    let mut snapshot_ns = Vec::with_capacity(snapshot_sample_capacity);
    let mut profiled_step_total_ns = Vec::with_capacity(ticks);
    let mut stage_samples: Vec<Vec<Option<u64>>> = WorldStepStage::all()
        .iter()
        .map(|_| Vec::with_capacity(ticks))
        .collect();
    for _ in 0..ticks {
        profiled_world
            .step_profiled(&mut profiler)
            .map_err(|error| format!("profiled step failed: {error}"))?;
        let profile = profiler
            .latest()
            .ok_or_else(|| "profiled step produced no timing report".to_owned())?;
        profiled_step_total_ns.push(profile.total_ns);
        for (sample_index, stage) in WorldStepStage::all().iter().copied().enumerate() {
            stage_samples[sample_index].push(profile.elapsed_ns(stage));
        }

        for _ in 0..PERF_SNAPSHOT_SAMPLES_PER_TICK {
            let snapshot_started = Instant::now();
            let snapshot = DynamicWorldSnapshot::from_world(&profiled_world);
            let elapsed = duration_ns(snapshot_started.elapsed());
            if snapshot.agents.len() != agents {
                return Err(format!(
                    "dynamic snapshot drifted from {agents} to {} agents",
                    snapshot.agents.len()
                ));
            }
            black_box(snapshot);
            snapshot_ns.push(elapsed);
        }
    }
    let profiled_digest = profiled_world
        .characterization_digest_v0()
        .map_err(|error| format!("failed to digest profiled world: {error}"))?;
    if throughput_digest != profiled_digest {
        return Err(format!(
            "profiled and pure-TPS passes diverged: {} != {}",
            throughput_digest.overall, profiled_digest.overall
        ));
    }
    if throughput_world.agent_count() != agents || profiled_world.agent_count() != agents {
        return Err(format!(
            "scenario population drift: pure={} profiled={} expected={agents}",
            throughput_world.agent_count(),
            profiled_world.agent_count()
        ));
    }

    let stages = WorldStepStage::all()
        .iter()
        .copied()
        .zip(stage_samples)
        .map(|(stage, raw_ns)| {
            let executed_ns: Vec<u64> = raw_ns.iter().flatten().copied().collect();
            let executions = executed_ns.len();
            let median_ns = (!executed_ns.is_empty()).then(|| median_u64(&executed_ns));
            StageResult {
                stage: stage.as_str().to_owned(),
                raw_ns,
                executions,
                median_ns,
            }
        })
        .collect();

    Ok(RepetitionResult {
        index,
        window_elapsed_ns,
        window_tps,
        total_step_elapsed_ns,
        total_tps,
        median_window_tps,
        snapshot_p50_ns: nearest_rank(&snapshot_ns, 50),
        snapshot_p95_ns: nearest_rank(&snapshot_ns, 95),
        snapshot_max_ns: snapshot_ns.iter().copied().max().unwrap_or_default(),
        snapshot_ns,
        profiled_step_total_ns,
        stages,
        final_agent_count: throughput_world.agent_count(),
        final_digest: throughput_digest.overall,
    })
}

fn run_scenario(
    agents: usize,
    brain_family: GateBrainFamily,
    ticks: usize,
    synthetic_sleep_us: u64,
) -> GateResult<ScenarioResult> {
    eprintln!(
        "perf-gate: agents={agents} brain={} warmups={PERF_WARMUPS} measurements={PERF_REPETITIONS} ticks={ticks}",
        brain_family.label()
    );
    let mut warmups = Vec::with_capacity(PERF_WARMUPS);
    for index in 0..PERF_WARMUPS {
        eprintln!("perf-gate: warmup {}/{}", index + 1, PERF_WARMUPS);
        warmups.push(run_repetition(
            agents,
            brain_family,
            ticks,
            index,
            synthetic_sleep_us,
        )?);
    }
    let mut measurements = Vec::with_capacity(PERF_REPETITIONS);
    for index in 0..PERF_REPETITIONS {
        eprintln!("perf-gate: measurement {}/{}", index + 1, PERF_REPETITIONS);
        measurements.push(run_repetition(
            agents,
            brain_family,
            ticks,
            index,
            synthetic_sleep_us,
        )?);
    }

    let run_total_tps: Vec<f64> = measurements.iter().map(|run| run.total_tps).collect();
    let snapshot_p95s: Vec<u64> = measurements.iter().map(|run| run.snapshot_p95_ns).collect();
    let mut stage_median_of_run_medians_ns = BTreeMap::new();
    for stage in WorldStepStage::all() {
        let run_stage_medians: Vec<u64> = measurements
            .iter()
            .filter_map(|run| {
                run.stages
                    .iter()
                    .find(|entry| entry.stage == stage.as_str())
                    .and_then(|entry| entry.median_ns)
            })
            .collect();
        stage_median_of_run_medians_ns.insert(
            stage.as_str().to_owned(),
            (!run_stage_medians.is_empty()).then(|| median_u64(&run_stage_medians)),
        );
    }
    Ok(ScenarioResult {
        id: scenario_id(agents, brain_family, ticks),
        agents,
        brain_family: brain_family.label().to_owned(),
        seed: PERF_SEED,
        ticks_per_repetition: ticks,
        feature_set: GateBrainFamily::compiled_feature_set(),
        scenario_config_hash: scenario_config_hash(brain_family)?,
        initial_agent_count: agents,
        warmups,
        measurements,
        median_of_run_total_tps: median_f64(&run_total_tps)?,
        run_tps_cv_pct: coefficient_of_variation_pct(&run_total_tps),
        median_of_run_snapshot_p95_ns: median_u64(&snapshot_p95s),
        snapshot_p95_cv_pct: coefficient_of_variation_pct(
            &snapshot_p95s
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>(),
        ),
        stage_median_of_run_medians_ns,
    })
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn median_f64(values: &[f64]) -> GateResult<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("median requires finite, non-empty samples".to_owned());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Ok((sorted[middle - 1] + sorted[middle]) / 2.0)
    } else {
        Ok(sorted[middle])
    }
}

fn median_u64(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        sorted[middle - 1].saturating_add(sorted[middle]) / 2
    } else {
        sorted[middle]
    }
}

fn nearest_rank(values: &[u64], percentile: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = percentile
        .saturating_mul(sorted.len())
        .div_ceil(100)
        .clamp(1, sorted.len());
    sorted[rank - 1]
}

fn coefficient_of_variation_pct(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean == 0.0 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt() / mean.abs() * 100.0
}

fn stable_hash(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn workspace_root() -> GateResult<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "scriptbots-core manifest has no workspace-root ancestor".to_owned())
}

fn command_output(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn checked_command_output(program: &str, args: &[&str], allow_empty: bool) -> GateResult<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("failed to run `{program} {}`: {error}", args.join(" ")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "`{program} {}` failed with {}: {}",
            args.join(" "),
            output.status,
            stderr.trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !allow_empty && stdout.is_empty() {
        return Err(format!(
            "`{program} {}` returned no identity",
            args.join(" ")
        ));
    }
    Ok(stdout)
}

fn cargo_config_value(workspace_root: &Path, key: &str) -> GateResult<String> {
    let output = Command::new("cargo")
        .current_dir(workspace_root)
        .args([
            "-Z",
            "unstable-options",
            "config",
            "get",
            "--format",
            "json",
            key,
        ])
        .output()
        .map_err(|error| format!("failed to inspect effective Cargo config `{key}`: {error}"))?;
    if output.status.success() {
        let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if value.is_empty() {
            return Err(format!(
                "effective Cargo config query `{key}` returned no identity"
            ));
        }
        return Ok(value);
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains("is not set") {
        return Ok("null".to_owned());
    }
    Err(format!(
        "effective Cargo config query `{key}` failed with {}: {}",
        output.status,
        stderr.trim()
    ))
}

fn read_trimmed(path: impl AsRef<Path>) -> String {
    fs::read_to_string(path)
        .map(|value| value.trim().to_owned())
        .unwrap_or_default()
}

fn proc_cpu_field(label: &str) -> String {
    let prefix = format!("{label}\t:");
    read_trimmed("/proc/cpuinfo")
        .lines()
        .find_map(|line| {
            line.strip_prefix(&prefix)
                .map(|value| value.trim().to_owned())
        })
        .unwrap_or_default()
}

fn rustc_field(rustc_verbose: &str, label: &str) -> GateResult<String> {
    rustc_verbose
        .lines()
        .find_map(|line| {
            line.strip_prefix(label)
                .map(|value| value.trim().to_owned())
        })
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("rustc -Vv did not report `{label}`"))
}

fn macos_mount_filesystem_kind(device: &str, mounts: &str) -> Option<String> {
    let prefix = format!("{device} on ");
    mounts
        .lines()
        .find(|line| line.starts_with(&prefix))
        .and_then(|line| line.rsplit_once(" ("))
        .and_then(|(_, properties)| properties.strip_suffix(')'))
        .and_then(|properties| properties.split(',').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn filesystem_kind() -> String {
    if cfg!(target_os = "linux") {
        command_output("stat", &["-f", "-c", "%T", "."])
    } else if cfg!(target_os = "macos") {
        let device = command_output("df", &["-P", "."])
            .lines()
            .last()
            .and_then(|line| line.split_whitespace().next())
            .unwrap_or_default()
            .to_owned();
        if device.is_empty() {
            return String::new();
        }
        let diskutil_kind = command_output("diskutil", &["info", &device])
            .lines()
            .find_map(|line| {
                line.trim()
                    .strip_prefix("Type (Bundle):")
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned)
            });
        diskutil_kind.unwrap_or_else(|| {
            macos_mount_filesystem_kind(&device, &command_output("mount", &[])).unwrap_or_default()
        })
    } else {
        String::new()
    }
}

fn cpu_brand() -> String {
    let linux = proc_cpu_field("model name");
    if !linux.is_empty() {
        linux
    } else {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
    }
}

fn memory_fingerprint() -> String {
    let linux = read_trimmed("/proc/meminfo")
        .lines()
        .find(|line| line.starts_with("MemTotal:"))
        .unwrap_or_default()
        .to_owned();
    if !linux.is_empty() {
        linux
    } else {
        command_output("sysctl", &["-n", "hw.memsize"])
    }
}

fn memory_class_identity(raw: &str) -> GateResult<String> {
    let raw = raw.trim();
    let total_bytes = if let Some(fields) = raw.strip_prefix("MemTotal:") {
        let mut fields = fields.split_whitespace();
        let kib = fields
            .next()
            .ok_or_else(|| "Linux memory identity has no capacity".to_owned())?
            .parse::<u64>()
            .map_err(|error| format!("invalid Linux memory capacity: {error}"))?;
        if fields.next() != Some("kB") || fields.next().is_some() {
            return Err(format!(
                "unsupported Linux memory identity `{raw}`; expected `MemTotal: <KiB> kB`"
            ));
        }
        kib.checked_mul(1_024)
            .ok_or_else(|| "Linux memory capacity overflowed bytes".to_owned())?
    } else {
        raw.parse::<u64>()
            .map_err(|error| format!("invalid byte-count memory identity `{raw}`: {error}"))?
    };
    let bucket_bytes = MEMORY_CLASS_BUCKET_MIB
        .checked_mul(1_024 * 1_024)
        .ok_or_else(|| "memory class bucket overflowed bytes".to_owned())?;
    let buckets = (total_bytes / bucket_bytes)
        .checked_add(u64::from(total_bytes % bucket_bytes != 0))
        .ok_or_else(|| "memory capacity overflowed its class bucket".to_owned())?;
    let rounded_mib = buckets
        .checked_mul(MEMORY_CLASS_BUCKET_MIB)
        .filter(|value| *value > 0)
        .ok_or_else(|| "memory capacity is below the minimum class bucket".to_owned())?;
    Ok(format!(
        "{rounded_mib} MiB (ceiling {MEMORY_CLASS_BUCKET_MIB} MiB)"
    ))
}

fn capture_fingerprint() -> GateResult<Fingerprint> {
    let workspace_root = workspace_root()?;
    let logical_cpus = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let scriptbots_thread_budget = env::var("SCRIPTBOTS_MAX_THREADS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0);
    let rayon_thread_budget = env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0);
    if let (Some(scriptbots), Some(rayon)) = (scriptbots_thread_budget, rayon_thread_budget)
        && scriptbots != rayon
    {
        return Err(format!(
            "thread-budget mismatch: SCRIPTBOTS_MAX_THREADS={scriptbots}, RAYON_NUM_THREADS={rayon}"
        ));
    }
    let thread_budget = rayon_thread_budget
        .or(scriptbots_thread_budget)
        .unwrap_or(logical_cpus);
    let parallel_min_split = env::var("SCRIPTBOTS_PAR_MIN_SPLIT")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1024);
    let brand = cpu_brand();
    let cpu_vendor = {
        let linux = proc_cpu_field("vendor_id");
        if linux.is_empty() {
            brand
                .split_whitespace()
                .next()
                .unwrap_or("unknown")
                .to_owned()
        } else {
            linux
        }
    };
    let cpu_family = {
        let value = proc_cpu_field("cpu family");
        if value.is_empty() {
            env::consts::ARCH.to_owned()
        } else {
            value
        }
    };
    let cpu_model = {
        let value = proc_cpu_field("model");
        if value.is_empty() {
            brand.clone()
        } else {
            value
        }
    };
    let cpu_quota = [
        read_trimmed("/sys/fs/cgroup/cpu.max"),
        read_trimmed("/sys/fs/cgroup/cpuset.cpus.effective"),
    ]
    .into_iter()
    .filter(|value| !value.is_empty())
    .collect::<Vec<_>>()
    .join(";");
    let rustc_verbose = checked_command_output("rustc", &["-Vv"], false)?;
    let rust_release = rustc_field(&rustc_verbose, "release:")?;
    let rust_host = rustc_field(&rustc_verbose, "host:")?;
    let kernel = command_output("uname", &["-srvmo"]);
    let memory = memory_fingerprint();
    let memory_class = memory_class_identity(&memory)?;
    let clang_identity = checked_command_output("clang", &["--version"], false)?;
    let mold_identity = command_output("mold", &["--version"]);
    let linker = format!(
        "clang={clang_identity};mold={}",
        if mold_identity.is_empty() {
            "unavailable"
        } else {
            mold_identity.as_str()
        }
    );
    let build_target = env::var("SCRIPTBOTS_PERF_BUILD_TARGET")
        .or_else(|_| env::var("CARGO_BUILD_TARGET"))
        .unwrap_or_else(|_| rust_host.clone());
    let effective_cargo_config = format!(
        "build.target={}\nbuild.rustflags={}\ntarget={}",
        cargo_config_value(&workspace_root, "build.target")?,
        cargo_config_value(&workspace_root, "build.rustflags")?,
        cargo_config_value(&workspace_root, &format!("target.{build_target}"))?
    );
    let cargo_config_hash = stable_hash(effective_cargo_config.as_bytes());
    let class = MachineClass {
        provider: env::var("RUNNER_ENVIRONMENT").unwrap_or_else(|_| "local".to_owned()),
        runner_os: env::var("RUNNER_OS").unwrap_or_else(|_| env::consts::OS.to_owned()),
        runner_arch: env::var("RUNNER_ARCH").unwrap_or_else(|_| env::consts::ARCH.to_owned()),
        image_os: env::var("ImageOS").unwrap_or_default(),
        image_version: env::var("ImageVersion").unwrap_or_default(),
        cpu_vendor,
        cpu_family,
        cpu_model,
        cpu_brand: brand.clone(),
        logical_cpus,
        cpu_quota,
        filesystem: filesystem_kind(),
        kernel: kernel.clone(),
        memory: memory_class,
        rust_release,
        build_target,
        rust_host,
        rustc_verbose,
        linker,
        cargo_config_hash,
        build_profile: "cargo-bench-release".to_owned(),
        thread_budget,
        parallel_min_split,
        rustflags: env::var("CARGO_ENCODED_RUSTFLAGS")
            .or_else(|_| env::var("RUSTFLAGS"))
            .unwrap_or_default(),
    };
    validate_machine_class_identity(&class)?;
    let class_bytes = serde_json::to_vec(&class)
        .map_err(|error| format!("failed to serialize machine class: {error}"))?;
    let workspace_root_text = workspace_root.to_string_lossy();
    let cargo_lock = workspace_root.join("Cargo.lock");
    let cargo_lock_text = cargo_lock.to_string_lossy();
    let git_status = checked_command_output(
        "git",
        &["-C", workspace_root_text.as_ref(), "status", "--porcelain"],
        true,
    )?;
    let cargo_lock_git_blob = checked_command_output(
        "git",
        &[
            "-C",
            workspace_root_text.as_ref(),
            "hash-object",
            cargo_lock_text.as_ref(),
        ],
        false,
    )?;
    let git_commit = checked_command_output(
        "git",
        &["-C", workspace_root_text.as_ref(), "rev-parse", "HEAD"],
        false,
    )?;
    let generated_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock precedes Unix epoch: {error}"))?
        .as_millis();

    Ok(Fingerprint {
        machine_class_id: stable_hash(&class_bytes),
        class,
        cpu_brand: brand,
        kernel,
        memory,
        cargo_lock_git_blob,
        git_commit,
        git_dirty: !git_status.is_empty(),
        generated_unix_ms,
        github_run_id: env::var("GITHUB_RUN_ID").ok(),
        github_run_attempt: env::var("GITHUB_RUN_ATTEMPT").ok(),
    })
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum VerdictStatus {
    Pass,
    Fail,
    Advisory,
    Refused,
    BootstrapRequired,
    BaselineCandidate,
}

impl VerdictStatus {
    const fn exit_code(self) -> i32 {
        match self {
            Self::Pass | Self::Advisory | Self::BootstrapRequired | Self::BaselineCandidate => 0,
            Self::Fail => 1,
            Self::Refused => 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScenarioVerdict {
    id: String,
    baseline_tps: f64,
    candidate_tps: f64,
    tps_regression_pct: f64,
    baseline_snapshot_p95_ns: u64,
    candidate_snapshot_p95_ns: u64,
    candidate_tps_cv_pct: f64,
    candidate_snapshot_cv_pct: f64,
    noisy: bool,
    tps_noisy: bool,
    snapshot_noisy: bool,
    would_fail_tps_regression: bool,
    would_fail_absolute_tps: bool,
    would_fail_snapshot_budget: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerfVerdict {
    schema: String,
    status: VerdictStatus,
    exit_code: i32,
    baseline_machine_class_id: Option<String>,
    candidate_machine_class_id: String,
    reasons: Vec<String>,
    scenarios: Vec<ScenarioVerdict>,
}

fn scenario_id(agents: usize, brain_family: GateBrainFamily, ticks: usize) -> String {
    format!(
        "agents-{agents}__brain-{}__ticks-{ticks}__snapshot-dynamic-v2",
        brain_family.scenario_slug()
    )
}

fn expected_scenarios(mode: GateMode, ticks: usize) -> Vec<(String, usize, GateBrainFamily)> {
    mode.agent_counts()
        .iter()
        .flat_map(|agents| {
            GateBrainFamily::ALL.into_iter().map(|brain_family| {
                (
                    scenario_id(*agents, brain_family, ticks),
                    *agents,
                    brain_family,
                )
            })
        })
        .collect()
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= scale * 1e-9
}

fn validate_repetition(
    scenario: &ScenarioResult,
    repetition: &RepetitionResult,
    expected_index: usize,
    policy: &GatePolicy,
) -> GateResult<()> {
    let ticks = scenario.ticks_per_repetition;
    let expected_windows = ticks / policy.median_window_ticks;
    if repetition.index != expected_index {
        return Err(format!(
            "scenario `{}` repetition index {} should be {expected_index}",
            scenario.id, repetition.index
        ));
    }
    if repetition.window_elapsed_ns.len() != expected_windows
        || repetition.window_tps.len() != expected_windows
    {
        return Err(format!(
            "scenario `{}` repetition {} has the wrong window sample count",
            scenario.id, repetition.index
        ));
    }
    if repetition.window_elapsed_ns.contains(&0)
        || repetition.window_tps.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "scenario `{}` repetition {} contains invalid throughput samples",
            scenario.id, repetition.index
        ));
    }
    for (elapsed_ns, recorded_tps) in repetition
        .window_elapsed_ns
        .iter()
        .zip(&repetition.window_tps)
    {
        let recomputed_tps =
            policy.median_window_ticks as f64 * 1_000_000_000.0 / *elapsed_ns as f64;
        if !approximately_equal(recomputed_tps, *recorded_tps) {
            return Err(format!(
                "scenario `{}` repetition {} has a window TPS value that does not match raw elapsed time",
                scenario.id, repetition.index
            ));
        }
    }
    let total_elapsed_ns = repetition
        .window_elapsed_ns
        .iter()
        .copied()
        .fold(0_u64, u64::saturating_add);
    let total_tps = ticks as f64 * 1_000_000_000.0 / total_elapsed_ns as f64;
    if repetition.total_step_elapsed_ns != total_elapsed_ns
        || !approximately_equal(repetition.total_tps, total_tps)
        || !approximately_equal(
            repetition.median_window_tps,
            median_f64(&repetition.window_tps)?,
        )
    {
        return Err(format!(
            "scenario `{}` repetition {} throughput summary does not match raw samples",
            scenario.id, repetition.index
        ));
    }

    let expected_snapshot_samples = ticks
        .checked_mul(policy.snapshot_samples_per_tick)
        .ok_or_else(|| {
            format!(
                "scenario `{}` snapshot sample count overflowed usize",
                scenario.id
            )
        })?;
    if repetition.snapshot_ns.len() != expected_snapshot_samples
        || repetition.snapshot_ns.contains(&0)
    {
        return Err(format!(
            "scenario `{}` repetition {} has invalid snapshot samples: got {}, expected {expected_snapshot_samples}",
            scenario.id,
            repetition.index,
            repetition.snapshot_ns.len()
        ));
    }
    if repetition.snapshot_p50_ns != nearest_rank(&repetition.snapshot_ns, 50)
        || repetition.snapshot_p95_ns != nearest_rank(&repetition.snapshot_ns, 95)
        || repetition.snapshot_max_ns
            != repetition
                .snapshot_ns
                .iter()
                .copied()
                .max()
                .unwrap_or_default()
    {
        return Err(format!(
            "scenario `{}` repetition {} snapshot summary does not match raw samples",
            scenario.id, repetition.index
        ));
    }
    if repetition.profiled_step_total_ns.len() != ticks
        || repetition.profiled_step_total_ns.contains(&0)
    {
        return Err(format!(
            "scenario `{}` repetition {} has invalid profiled-step samples",
            scenario.id, repetition.index
        ));
    }
    if repetition.stages.len() != WorldStepStage::all().len() {
        return Err(format!(
            "scenario `{}` repetition {} has {} stages; expected {}",
            scenario.id,
            repetition.index,
            repetition.stages.len(),
            WorldStepStage::all().len()
        ));
    }
    for stage in WorldStepStage::all() {
        let matching: Vec<&StageResult> = repetition
            .stages
            .iter()
            .filter(|entry| entry.stage == stage.as_str())
            .collect();
        if matching.len() != 1 {
            return Err(format!(
                "scenario `{}` repetition {} must contain exactly one `{}` stage",
                scenario.id,
                repetition.index,
                stage.as_str()
            ));
        }
        let result = matching[0];
        if result.raw_ns.len() != ticks {
            return Err(format!(
                "scenario `{}` repetition {} stage `{}` does not match its raw samples",
                scenario.id,
                repetition.index,
                stage.as_str()
            ));
        }
        let expected_execution = |sample_index: usize| {
            *stage != WorldStepStage::Aging
                || (sample_index + 1).is_multiple_of(perf_config().aging_tick_interval as usize)
        };
        if result
            .raw_ns
            .iter()
            .enumerate()
            .any(|(sample_index, sample)| sample.is_some() != expected_execution(sample_index))
        {
            return Err(format!(
                "scenario `{}` repetition {} stage `{}` has an invalid execution cadence",
                scenario.id,
                repetition.index,
                stage.as_str()
            ));
        }
        let executed_ns: Vec<u64> = result.raw_ns.iter().flatten().copied().collect();
        let expected_median = (!executed_ns.is_empty()).then(|| median_u64(&executed_ns));
        if result.executions != executed_ns.len() || result.median_ns != expected_median {
            return Err(format!(
                "scenario `{}` repetition {} stage `{}` summary does not match its executed samples",
                scenario.id,
                repetition.index,
                stage.as_str()
            ));
        }
    }
    for sample_index in 0..ticks {
        let stage_total_ns = repetition.stages.iter().fold(0_u64, |total, stage| {
            total.saturating_add(stage.raw_ns[sample_index].unwrap_or_default())
        });
        if repetition.profiled_step_total_ns[sample_index] < stage_total_ns {
            return Err(format!(
                "scenario `{}` repetition {} profiled-step sample {} is shorter than its stage samples",
                scenario.id, repetition.index, sample_index
            ));
        }
    }
    if repetition.final_agent_count != scenario.initial_agent_count
        || repetition.final_digest.trim().is_empty()
    {
        return Err(format!(
            "scenario `{}` repetition {} has population drift or no final digest",
            scenario.id, repetition.index
        ));
    }
    Ok(())
}

fn validate_artifact(artifact: &PerfArtifact, require_baseline: bool) -> GateResult<()> {
    if artifact.schema != PERF_GATE_SCHEMA {
        return Err(format!(
            "artifact schema mismatch: expected {PERF_GATE_SCHEMA}, got {}",
            artifact.schema
        ));
    }
    if artifact.scenario_contract != PERF_SCENARIO_CONTRACT {
        return Err(format!(
            "scenario contract mismatch: expected {PERF_SCENARIO_CONTRACT}, got {}",
            artifact.scenario_contract
        ));
    }
    if artifact.dynamic_snapshot_schema != DYNAMIC_WORLD_SNAPSHOT_SCHEMA {
        return Err("dynamic snapshot schema mismatch".to_owned());
    }
    if artifact.world_step_profile_schema != WORLD_STEP_PROFILE_SCHEMA {
        return Err("world-step profile schema mismatch".to_owned());
    }
    let class_bytes = serde_json::to_vec(&artifact.fingerprint.class)
        .map_err(|error| format!("failed to serialize artifact machine class: {error}"))?;
    if artifact.fingerprint.machine_class_id != stable_hash(&class_bytes) {
        return Err("machine-class ID does not match the serialized class".to_owned());
    }
    if artifact.fingerprint.git_commit.trim().is_empty()
        || artifact.fingerprint.cargo_lock_git_blob.trim().is_empty()
    {
        return Err("artifact has no verified Git commit or Cargo.lock identity".to_owned());
    }
    validate_machine_class_identity(&artifact.fingerprint.class)?;
    let expected_memory_class = memory_class_identity(&artifact.fingerprint.memory)?;
    if artifact.fingerprint.cpu_brand != artifact.fingerprint.class.cpu_brand
        || artifact.fingerprint.kernel != artifact.fingerprint.class.kernel
        || expected_memory_class != artifact.fingerprint.class.memory
    {
        return Err("fingerprint host identity disagrees with its machine class".to_owned());
    }
    if artifact.source_commit != artifact.fingerprint.git_commit {
        return Err("source commit does not match the fingerprint".to_owned());
    }
    let ticks = artifact.policy.ticks_per_repetition;
    if ticks < 200 || !ticks.is_multiple_of(PERF_WINDOW_TICKS) {
        return Err(format!("invalid gate tick count {ticks}"));
    }
    let expected_snapshot_samples = ticks
        .checked_mul(artifact.policy.snapshot_samples_per_tick)
        .ok_or_else(|| "artifact snapshot sample count overflowed usize".to_owned())?;
    if artifact.policy.snapshot_samples_per_tick == 0
        || artifact.policy.snapshot_samples_per_repetition != expected_snapshot_samples
    {
        return Err("artifact snapshot sampling policy is internally inconsistent".to_owned());
    }
    if artifact.policy != gate_policy(ticks) {
        return Err("artifact policy differs from the compiled gate policy".to_owned());
    }
    let mode = GateMode::parse(&artifact.mode)?;
    if require_baseline {
        if artifact.artifact_kind != "baseline" {
            return Err(format!(
                "comparison input is not a reviewed baseline: artifact_kind={}",
                artifact.artifact_kind
            ));
        }
        if artifact.synthetic_sleep_us != 0 {
            return Err("baseline contains a synthetic delay".to_owned());
        }
        if artifact.fingerprint.git_dirty {
            return Err("baseline was recorded from a dirty checkout".to_owned());
        }
        if artifact
            .baseline_justification
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err("baseline has no reviewed justification".to_owned());
        }
        if mode != GateMode::Full {
            return Err("reviewed baselines must contain the full scenario matrix".to_owned());
        }
    } else if artifact.artifact_kind != "candidate" {
        return Err(format!(
            "comparison candidate has unexpected artifact_kind={}",
            artifact.artifact_kind
        ));
    }

    let expected = expected_scenarios(mode, ticks);
    if artifact.scenarios.len() != expected.len() {
        return Err(format!(
            "mode `{}` requires {} exact scenarios, found {}",
            artifact.mode,
            expected.len(),
            artifact.scenarios.len()
        ));
    }
    let mut ids = BTreeMap::new();
    for scenario in &artifact.scenarios {
        if ids.insert(&scenario.id, ()).is_some() {
            return Err(format!("duplicate scenario id `{}`", scenario.id));
        }
        let Some((_, expected_agents, brain_family)) =
            expected.iter().find(|(id, _, _)| id == &scenario.id)
        else {
            return Err(format!(
                "scenario `{}` is not part of mode `{}`",
                scenario.id, artifact.mode
            ));
        };
        if scenario.agents != *expected_agents
            || scenario.initial_agent_count != *expected_agents
            || scenario.brain_family != brain_family.label()
            || scenario.seed != PERF_SEED
            || scenario.ticks_per_repetition != ticks
            || scenario.feature_set != GateBrainFamily::compiled_feature_set()
            || scenario.scenario_config_hash != scenario_config_hash(*brain_family)?
        {
            return Err(format!(
                "scenario `{}` does not match its compiled workload contract",
                scenario.id
            ));
        }
        if scenario.warmups.len() != artifact.policy.warmup_repetitions {
            return Err(format!(
                "scenario `{}` has {} warmups; expected {}",
                scenario.id,
                scenario.warmups.len(),
                artifact.policy.warmup_repetitions
            ));
        }
        if scenario.measurements.len() != artifact.policy.measured_repetitions {
            return Err(format!(
                "scenario `{}` has {} measurements; expected {}",
                scenario.id,
                scenario.measurements.len(),
                artifact.policy.measured_repetitions
            ));
        }
        for (index, repetition) in scenario.warmups.iter().enumerate() {
            validate_repetition(scenario, repetition, index, &artifact.policy)?;
        }
        for (index, repetition) in scenario.measurements.iter().enumerate() {
            validate_repetition(scenario, repetition, index, &artifact.policy)?;
        }
        let mut digests = scenario
            .warmups
            .iter()
            .chain(&scenario.measurements)
            .map(|repetition| repetition.final_digest.as_str());
        let first_digest = digests.next().ok_or_else(|| {
            format!(
                "scenario `{}` has no deterministic digest evidence",
                scenario.id
            )
        })?;
        if digests.any(|digest| digest != first_digest) {
            return Err(format!(
                "scenario `{}` repetitions produced different final digests",
                scenario.id
            ));
        }
        let measured_tps: Vec<f64> = scenario
            .measurements
            .iter()
            .map(|run| run.total_tps)
            .collect();
        let measured_snapshot: Vec<u64> = scenario
            .measurements
            .iter()
            .map(|run| run.snapshot_p95_ns)
            .collect();
        let recomputed_tps = median_f64(&measured_tps)?;
        let recomputed_snapshot = median_u64(&measured_snapshot);
        let recomputed_tps_cv = coefficient_of_variation_pct(&measured_tps);
        let recomputed_snapshot_cv = coefficient_of_variation_pct(
            &measured_snapshot
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>(),
        );
        if !approximately_equal(recomputed_tps, scenario.median_of_run_total_tps)
            || recomputed_snapshot != scenario.median_of_run_snapshot_p95_ns
            || !approximately_equal(recomputed_tps_cv, scenario.run_tps_cv_pct)
            || !approximately_equal(recomputed_snapshot_cv, scenario.snapshot_p95_cv_pct)
        {
            return Err(format!(
                "scenario `{}` summary does not match its raw measurements",
                scenario.id
            ));
        }
        if scenario.stage_median_of_run_medians_ns.len() != WorldStepStage::all().len() {
            return Err(format!(
                "scenario `{}` has the wrong aggregate stage set",
                scenario.id
            ));
        }
        for stage in WorldStepStage::all() {
            let run_medians: Vec<u64> = scenario
                .measurements
                .iter()
                .filter_map(|repetition| {
                    repetition
                        .stages
                        .iter()
                        .find(|entry| entry.stage == stage.as_str())
                        .and_then(|entry| entry.median_ns)
                })
                .collect();
            let expected_median = (!run_medians.is_empty()).then(|| median_u64(&run_medians));
            if scenario
                .stage_median_of_run_medians_ns
                .get(stage.as_str())
                .copied()
                != Some(expected_median)
            {
                return Err(format!(
                    "scenario `{}` aggregate stage `{}` does not match raw repetitions",
                    scenario.id,
                    stage.as_str()
                ));
            }
        }
    }
    Ok(())
}

fn compare_artifacts(baseline: &PerfArtifact, candidate: &PerfArtifact) -> PerfVerdict {
    let refused = |reason: String| PerfVerdict {
        schema: PERF_VERDICT_SCHEMA.to_owned(),
        status: VerdictStatus::Refused,
        exit_code: VerdictStatus::Refused.exit_code(),
        baseline_machine_class_id: Some(baseline.fingerprint.machine_class_id.clone()),
        candidate_machine_class_id: candidate.fingerprint.machine_class_id.clone(),
        reasons: vec![reason],
        scenarios: Vec::new(),
    };
    if let Err(error) = validate_artifact(baseline, true) {
        return refused(format!("invalid baseline: {error}"));
    }
    if let Err(error) = validate_artifact(candidate, false) {
        return refused(format!("invalid candidate: {error}"));
    }
    if baseline.policy != candidate.policy {
        return refused("gate policy mismatch; rebaseline review is required".to_owned());
    }
    if baseline.fingerprint.machine_class_id != candidate.fingerprint.machine_class_id
        || baseline.fingerprint.class != candidate.fingerprint.class
    {
        return refused(format!(
            "machine class mismatch: baseline={} candidate={}; no cross-class delta was calculated",
            baseline.fingerprint.machine_class_id, candidate.fingerprint.machine_class_id
        ));
    }

    let mut reasons = Vec::new();
    let mut scenario_verdicts = Vec::new();
    let mut any_stable_failure = false;
    let mut any_noise = candidate.fingerprint.git_dirty;
    if candidate.fingerprint.git_dirty {
        reasons.push("candidate checkout is dirty; result is advisory".to_owned());
    }
    for candidate_scenario in &candidate.scenarios {
        let Some(baseline_scenario) = baseline
            .scenarios
            .iter()
            .find(|scenario| scenario.id == candidate_scenario.id)
        else {
            return refused(format!(
                "baseline has no exact scenario `{}`",
                candidate_scenario.id
            ));
        };
        if baseline_scenario.agents != candidate_scenario.agents
            || baseline_scenario.brain_family != candidate_scenario.brain_family
            || baseline_scenario.seed != candidate_scenario.seed
            || baseline_scenario.ticks_per_repetition != candidate_scenario.ticks_per_repetition
            || baseline_scenario.feature_set != candidate_scenario.feature_set
            || baseline_scenario.scenario_config_hash != candidate_scenario.scenario_config_hash
            || baseline_scenario.initial_agent_count != candidate_scenario.initial_agent_count
            || baseline_scenario.measurements[0].final_digest
                != candidate_scenario.measurements[0].final_digest
        {
            return refused(format!(
                "scenario contract mismatch for `{}`; no workload delta was calculated",
                candidate_scenario.id
            ));
        }
        if baseline_scenario.run_tps_cv_pct > MAX_CV_PCT
            || baseline_scenario.snapshot_p95_cv_pct > MAX_CV_PCT
        {
            return refused(format!(
                "baseline scenario `{}` is noisy (TPS CV {:.2}%, snapshot CV {:.2}%); it must be re-recorded",
                baseline_scenario.id,
                baseline_scenario.run_tps_cv_pct,
                baseline_scenario.snapshot_p95_cv_pct
            ));
        }
        let tps_noisy = candidate_scenario.run_tps_cv_pct > MAX_CV_PCT;
        let snapshot_noisy = candidate_scenario.snapshot_p95_cv_pct > MAX_CV_PCT;
        let noisy = tps_noisy || snapshot_noisy;
        any_noise |= noisy;
        if noisy {
            reasons.push(format!(
                "scenario `{}` has advisory metrics: TPS CV {:.2}%, snapshot CV {:.2}%",
                candidate_scenario.id,
                candidate_scenario.run_tps_cv_pct,
                candidate_scenario.snapshot_p95_cv_pct
            ));
        }
        let baseline_tps = baseline_scenario.median_of_run_total_tps;
        let candidate_tps = candidate_scenario.median_of_run_total_tps;
        let tps_regression_pct = (baseline_tps - candidate_tps) / baseline_tps * 100.0;
        let would_fail_tps_regression = tps_regression_pct > MAX_TPS_REGRESSION_PCT;
        let would_fail_absolute_tps =
            candidate_scenario.agents == 1_000 && candidate_tps < MIN_TPS_1K;
        let would_fail_snapshot_budget = candidate_scenario.agents == 1_000
            && candidate_scenario.median_of_run_snapshot_p95_ns >= MAX_SNAPSHOT_P95_NS_1K;
        any_stable_failure |= (!tps_noisy
            && (would_fail_tps_regression || would_fail_absolute_tps))
            || (!snapshot_noisy && would_fail_snapshot_budget);
        if tps_noisy && (would_fail_tps_regression || would_fail_absolute_tps) {
            reasons.push(format!(
                "scenario `{}` would fail a TPS budget, but only that noisy TPS measurement is advisory",
                candidate_scenario.id
            ));
        }
        if snapshot_noisy && would_fail_snapshot_budget {
            reasons.push(format!(
                "scenario `{}` would fail the snapshot budget, but only that noisy snapshot measurement is advisory",
                candidate_scenario.id
            ));
        }
        scenario_verdicts.push(ScenarioVerdict {
            id: candidate_scenario.id.clone(),
            baseline_tps,
            candidate_tps,
            tps_regression_pct,
            baseline_snapshot_p95_ns: baseline_scenario.median_of_run_snapshot_p95_ns,
            candidate_snapshot_p95_ns: candidate_scenario.median_of_run_snapshot_p95_ns,
            candidate_tps_cv_pct: candidate_scenario.run_tps_cv_pct,
            candidate_snapshot_cv_pct: candidate_scenario.snapshot_p95_cv_pct,
            noisy,
            tps_noisy,
            snapshot_noisy,
            would_fail_tps_regression,
            would_fail_absolute_tps,
            would_fail_snapshot_budget,
        });
    }
    let status = if candidate.fingerprint.git_dirty {
        if any_stable_failure {
            reasons.push(
                "stable thresholds would fail, but dirty-checkout policy makes the whole run advisory"
                    .to_owned(),
            );
        }
        VerdictStatus::Advisory
    } else if any_stable_failure {
        VerdictStatus::Fail
    } else if any_noise {
        VerdictStatus::Advisory
    } else {
        VerdictStatus::Pass
    };
    if reasons.is_empty() {
        reasons.push(match status {
            VerdictStatus::Pass => "all exact-class performance budgets passed".to_owned(),
            VerdictStatus::Fail => "one or more stable performance budgets failed".to_owned(),
            _ => "performance comparison completed".to_owned(),
        });
    }
    PerfVerdict {
        schema: PERF_VERDICT_SCHEMA.to_owned(),
        status,
        exit_code: status.exit_code(),
        baseline_machine_class_id: Some(baseline.fingerprint.machine_class_id.clone()),
        candidate_machine_class_id: candidate.fingerprint.machine_class_id.clone(),
        reasons,
        scenarios: scenario_verdicts,
    }
}

fn bootstrap_verdict(candidate: &PerfArtifact, reason: String) -> PerfVerdict {
    let status = VerdictStatus::BootstrapRequired;
    PerfVerdict {
        schema: PERF_VERDICT_SCHEMA.to_owned(),
        status,
        exit_code: status.exit_code(),
        baseline_machine_class_id: None,
        candidate_machine_class_id: candidate.fingerprint.machine_class_id.clone(),
        reasons: vec![reason],
        scenarios: Vec::new(),
    }
}

fn baseline_candidate_verdict(candidate: &PerfArtifact) -> PerfVerdict {
    let status = VerdictStatus::BaselineCandidate;
    PerfVerdict {
        schema: PERF_VERDICT_SCHEMA.to_owned(),
        status,
        exit_code: status.exit_code(),
        baseline_machine_class_id: Some(candidate.fingerprint.machine_class_id.clone()),
        candidate_machine_class_id: candidate.fingerprint.machine_class_id.clone(),
        reasons: vec![
            "baseline candidate passed noise and absolute-budget admission; review raw samples and commit it deliberately"
                .to_owned(),
        ],
        scenarios: Vec::new(),
    }
}

fn ensure_baseline_is_blessable(artifact: &PerfArtifact) -> GateResult<()> {
    validate_artifact(artifact, true)?;
    for scenario in &artifact.scenarios {
        if scenario.run_tps_cv_pct > MAX_CV_PCT || scenario.snapshot_p95_cv_pct > MAX_CV_PCT {
            return Err(format!(
                "refusing noisy baseline `{}`: TPS CV {:.2}%, snapshot CV {:.2}%",
                scenario.id, scenario.run_tps_cv_pct, scenario.snapshot_p95_cv_pct
            ));
        }
        if scenario.agents == 1_000 && scenario.median_of_run_total_tps < MIN_TPS_1K {
            return Err(format!(
                "refusing baseline `{}` below the absolute {:.0} TPS budget: {:.2} TPS",
                scenario.id, MIN_TPS_1K, scenario.median_of_run_total_tps
            ));
        }
        if scenario.agents == 1_000
            && scenario.median_of_run_snapshot_p95_ns >= MAX_SNAPSHOT_P95_NS_1K
        {
            return Err(format!(
                "refusing baseline `{}` at or above the strict 4ms snapshot budget: {}ns",
                scenario.id, scenario.median_of_run_snapshot_p95_ns
            ));
        }
    }
    Ok(())
}

fn write_json(path: &Path, value: &impl Serialize) -> GateResult<()> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to encode {}: {error}", path.display()))?;
    bytes.push(b'\n');
    fs::write(path, bytes).map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn summary_markdown(artifact: &PerfArtifact, verdict: &PerfVerdict) -> String {
    let mut output = String::new();
    output.push_str("# ScriptBots performance gate\n\n");
    output.push_str(&format!(
        "- Verdict: `{:?}`\n- Machine class: `{}`\n- Commit: `{}`\n- Mode: `{}`\n- Snapshot operation: `{}`\n- Snapshot sampling: `{} separately timed operations/tick; {} raw samples/repetition`\n- Stage schema: `{}`\n\n",
        verdict.status,
        artifact.fingerprint.machine_class_id,
        artifact.source_commit,
        artifact.mode,
        artifact.dynamic_snapshot_schema,
        artifact.policy.snapshot_samples_per_tick,
        artifact.policy.snapshot_samples_per_repetition,
        artifact.world_step_profile_schema
    ));
    for reason in &verdict.reasons {
        output.push_str(&format!("- {reason}\n"));
    }
    output.push_str("\n## Measurements\n\n");
    output.push_str(
        "| Scenario | Median TPS | TPS CV | Snapshot p95 | Snapshot CV | Raw run TPS |\n",
    );
    output.push_str("|---|---:|---:|---:|---:|---|\n");
    for scenario in &artifact.scenarios {
        let raw = scenario
            .measurements
            .iter()
            .map(|run| format!("{:.2}", run.total_tps))
            .collect::<Vec<_>>()
            .join(", ");
        output.push_str(&format!(
            "| `{}` | {:.2} | {:.2}% | {:.3} ms | {:.2}% | {} |\n",
            scenario.id,
            scenario.median_of_run_total_tps,
            scenario.run_tps_cv_pct,
            scenario.median_of_run_snapshot_p95_ns as f64 / 1_000_000.0,
            scenario.snapshot_p95_cv_pct,
            raw
        ));
    }
    if !verdict.scenarios.is_empty() {
        output.push_str("\n## Comparison\n\n");
        output.push_str("| Scenario | Baseline TPS | Candidate TPS | Delta | Budget flags |\n");
        output.push_str("|---|---:|---:|---:|---|\n");
        for scenario in &verdict.scenarios {
            let mut flags = Vec::new();
            if scenario.would_fail_tps_regression {
                flags.push(">10% TPS regression");
            }
            if scenario.would_fail_absolute_tps {
                flags.push("<60 TPS at 1k");
            }
            if scenario.would_fail_snapshot_budget {
                flags.push("snapshot p95 >=4ms at 1k");
            }
            if scenario.noisy {
                flags.push("CV advisory");
            }
            if flags.is_empty() {
                flags.push("pass");
            }
            output.push_str(&format!(
                "| `{}` | {:.2} | {:.2} | {:+.2}% | {} |\n",
                scenario.id,
                scenario.baseline_tps,
                scenario.candidate_tps,
                -scenario.tps_regression_pct,
                flags.join(", ")
            ));
        }
    }
    output.push_str(
        "\nThe five measured runs are a regression sentinel, not a publishable performance claim. Raw nanosecond samples and the full fingerprint are in `perf_result.json`.\n",
    );
    output
}

fn gate_policy(ticks: usize) -> GatePolicy {
    GatePolicy {
        warmup_repetitions: PERF_WARMUPS,
        measured_repetitions: PERF_REPETITIONS,
        ticks_per_repetition: ticks,
        snapshot_samples_per_tick: PERF_SNAPSHOT_SAMPLES_PER_TICK,
        snapshot_samples_per_repetition: ticks.saturating_mul(PERF_SNAPSHOT_SAMPLES_PER_TICK),
        median_window_ticks: PERF_WINDOW_TICKS,
        max_cv_pct: MAX_CV_PCT,
        max_tps_regression_pct: MAX_TPS_REGRESSION_PCT,
        min_tps_1k: MIN_TPS_1K,
        max_snapshot_p95_ns_1k: MAX_SNAPSHOT_P95_NS_1K,
    }
}

fn run_perf_gate(args: GateArgs) -> GateResult<i32> {
    if args.self_test {
        return run_self_test().map(|()| 0);
    }
    let fingerprint = capture_fingerprint()?;
    let mut scenarios = Vec::new();
    for agents in args.mode.agent_counts() {
        for brain_family in GateBrainFamily::ALL {
            scenarios.push(run_scenario(
                *agents,
                brain_family,
                args.ticks,
                args.synthetic_sleep_us,
            )?);
        }
    }
    let artifact_kind = if args.record_baseline {
        "baseline"
    } else {
        "candidate"
    };
    let artifact = PerfArtifact {
        schema: PERF_GATE_SCHEMA.to_owned(),
        scenario_contract: PERF_SCENARIO_CONTRACT.to_owned(),
        artifact_kind: artifact_kind.to_owned(),
        mode: args.mode.as_str().to_owned(),
        source_commit: fingerprint.git_commit.clone(),
        baseline_justification: args.justification.clone(),
        dynamic_snapshot_schema: DYNAMIC_WORLD_SNAPSHOT_SCHEMA.to_owned(),
        world_step_profile_schema: WORLD_STEP_PROFILE_SCHEMA.to_owned(),
        synthetic_sleep_us: args.synthetic_sleep_us,
        fingerprint,
        policy: gate_policy(args.ticks),
        scenarios,
    };

    fs::create_dir_all(&args.output_dir).map_err(|error| {
        format!(
            "failed to create artifact directory {}: {error}",
            args.output_dir.display()
        )
    })?;
    write_json(&args.output_dir.join("perf_result.json"), &artifact)?;
    write_json(
        &args.output_dir.join("fingerprint.json"),
        &artifact.fingerprint,
    )?;

    let verdict = if args.record_baseline {
        ensure_baseline_is_blessable(&artifact)?;
        write_json(&args.output_dir.join("perf_baseline.json"), &artifact)?;
        baseline_candidate_verdict(&artifact)
    } else if let Some(path) = args.baseline {
        if path.is_file() {
            let bytes = fs::read(&path)
                .map_err(|error| format!("failed to read baseline {}: {error}", path.display()))?;
            let baseline: PerfArtifact = serde_json::from_slice(&bytes)
                .map_err(|error| format!("failed to parse baseline {}: {error}", path.display()))?;
            compare_artifacts(&baseline, &artifact)
        } else {
            bootstrap_verdict(
                &artifact,
                format!(
                    "no checked-in exact baseline exists at {}; run the reviewed baseline-candidate workflow",
                    path.display()
                ),
            )
        }
    } else {
        bootstrap_verdict(
            &artifact,
            "no baseline was supplied; raw results are informational only".to_owned(),
        )
    };
    write_json(&args.output_dir.join("perf_verdict.json"), &verdict)?;
    let summary = summary_markdown(&artifact, &verdict);
    fs::write(args.output_dir.join("perf_summary.md"), &summary)
        .map_err(|error| format!("failed to write performance summary: {error}"))?;
    print!("{summary}");
    Ok(verdict.exit_code)
}

fn synthetic_repetition(
    index: usize,
    agents: usize,
    brain_family: GateBrainFamily,
    requested_tps: f64,
    snapshot_p95_ns: u64,
) -> RepetitionResult {
    let window_elapsed = (PERF_WINDOW_TICKS as f64 * 1_000_000_000.0 / requested_tps) as u64;
    let window_tps = PERF_WINDOW_TICKS as f64 * 1_000_000_000.0 / window_elapsed as f64;
    let window_count = PERF_TICKS / PERF_WINDOW_TICKS;
    let window_elapsed_ns = vec![window_elapsed; window_count];
    let total_step_elapsed_ns = window_elapsed.saturating_mul(window_count as u64);
    let total_tps = PERF_TICKS as f64 * 1_000_000_000.0 / total_step_elapsed_ns as f64;
    let stages = WorldStepStage::all()
        .iter()
        .map(|stage| StageResult {
            stage: stage.as_str().to_owned(),
            raw_ns: (0..PERF_TICKS)
                .map(|sample_index| {
                    (*stage != WorldStepStage::Aging
                        || (sample_index + 1)
                            .is_multiple_of(perf_config().aging_tick_interval as usize))
                    .then_some(1)
                })
                .collect(),
            executions: if *stage == WorldStepStage::Aging {
                PERF_TICKS / perf_config().aging_tick_interval as usize
            } else {
                PERF_TICKS
            },
            median_ns: Some(1),
        })
        .collect();
    RepetitionResult {
        index,
        window_elapsed_ns,
        window_tps: vec![window_tps; window_count],
        total_step_elapsed_ns,
        total_tps,
        median_window_tps: window_tps,
        snapshot_ns: vec![snapshot_p95_ns; PERF_TICKS * PERF_SNAPSHOT_SAMPLES_PER_TICK],
        snapshot_p50_ns: snapshot_p95_ns,
        snapshot_p95_ns,
        snapshot_max_ns: snapshot_p95_ns,
        profiled_step_total_ns: vec![WorldStepStage::all().len() as u64; PERF_TICKS],
        stages,
        final_agent_count: agents,
        final_digest: format!("synthetic-{}-{agents}", brain_family.scenario_slug()),
    }
}

fn synthetic_artifact(
    artifact_kind: &str,
    class_suffix: &str,
    tps: [f64; PERF_REPETITIONS],
    snapshot_p95_ns: [u64; PERF_REPETITIONS],
) -> PerfArtifact {
    let memory = "MemTotal: 16777216 kB".to_owned();
    let memory_class = memory_class_identity(&memory).expect("normalize test memory class");
    let class = MachineClass {
        provider: format!("self-test-{class_suffix}"),
        runner_os: "linux".to_owned(),
        runner_arch: "x64".to_owned(),
        image_os: "ubuntu24".to_owned(),
        image_version: "test-image-v1".to_owned(),
        cpu_vendor: "Synthetic".to_owned(),
        cpu_family: "1".to_owned(),
        cpu_model: "1".to_owned(),
        cpu_brand: "Synthetic CPU".to_owned(),
        logical_cpus: 4,
        cpu_quota: "4".to_owned(),
        filesystem: "ext2/ext3".to_owned(),
        kernel: "test-kernel".to_owned(),
        memory: memory_class,
        rust_release: "nightly-test".to_owned(),
        rust_host: "x86_64-unknown-linux-gnu".to_owned(),
        rustc_verbose: "rustc nightly-test (self-test)\nbinary: rustc\ncommit-hash: test\ncommit-date: test\nhost: x86_64-unknown-linux-gnu\nrelease: nightly-test\nLLVM version: test".to_owned(),
        linker: "clang=test;mold=test".to_owned(),
        cargo_config_hash: "fnv1a64:0000000000000000".to_owned(),
        build_target: "x86_64-unknown-linux-gnu".to_owned(),
        build_profile: "cargo-bench-release".to_owned(),
        thread_budget: 4,
        parallel_min_split: 1024,
        rustflags: String::new(),
    };
    let machine_class_id = stable_hash(&serde_json::to_vec(&class).expect("serialize test class"));
    let scenarios = expected_scenarios(GateMode::Full, PERF_TICKS)
        .into_iter()
        .map(|(id, agents, brain_family)| {
            let measurements: Vec<RepetitionResult> = tps
                .iter()
                .copied()
                .zip(snapshot_p95_ns)
                .enumerate()
                .map(|(index, (run_tps, snapshot))| {
                    synthetic_repetition(index, agents, brain_family, run_tps, snapshot)
                })
                .collect();
            let warmups = (0..PERF_WARMUPS)
                .map(|index| {
                    synthetic_repetition(index, agents, brain_family, tps[0], snapshot_p95_ns[0])
                })
                .collect();
            let measured_tps: Vec<f64> = measurements
                .iter()
                .map(|repetition| repetition.total_tps)
                .collect();
            let measured_snapshot: Vec<u64> = measurements
                .iter()
                .map(|repetition| repetition.snapshot_p95_ns)
                .collect();
            let snapshot_values: Vec<f64> = measured_snapshot
                .iter()
                .map(|value| *value as f64)
                .collect();
            let stage_median_of_run_medians_ns = WorldStepStage::all()
                .iter()
                .map(|stage| (stage.as_str().to_owned(), Some(1)))
                .collect();
            ScenarioResult {
                id,
                agents,
                brain_family: brain_family.label().to_owned(),
                seed: PERF_SEED,
                ticks_per_repetition: PERF_TICKS,
                feature_set: GateBrainFamily::compiled_feature_set(),
                scenario_config_hash: scenario_config_hash(brain_family)
                    .expect("serialize self-test scenario"),
                initial_agent_count: agents,
                warmups,
                measurements,
                median_of_run_total_tps: median_f64(&measured_tps).expect("test median"),
                run_tps_cv_pct: coefficient_of_variation_pct(&measured_tps),
                median_of_run_snapshot_p95_ns: median_u64(&measured_snapshot),
                snapshot_p95_cv_pct: coefficient_of_variation_pct(&snapshot_values),
                stage_median_of_run_medians_ns,
            }
        })
        .collect();
    PerfArtifact {
        schema: PERF_GATE_SCHEMA.to_owned(),
        scenario_contract: PERF_SCENARIO_CONTRACT.to_owned(),
        artifact_kind: artifact_kind.to_owned(),
        mode: "full".to_owned(),
        source_commit: "self-test".to_owned(),
        baseline_justification: (artifact_kind == "baseline")
            .then(|| "self-test baseline".to_owned()),
        dynamic_snapshot_schema: DYNAMIC_WORLD_SNAPSHOT_SCHEMA.to_owned(),
        world_step_profile_schema: WORLD_STEP_PROFILE_SCHEMA.to_owned(),
        synthetic_sleep_us: 0,
        fingerprint: Fingerprint {
            machine_class_id,
            class,
            cpu_brand: "Synthetic CPU".to_owned(),
            kernel: "test-kernel".to_owned(),
            memory,
            cargo_lock_git_blob: "test".to_owned(),
            git_commit: "self-test".to_owned(),
            git_dirty: false,
            generated_unix_ms: 0,
            github_run_id: None,
            github_run_attempt: None,
        },
        policy: gate_policy(PERF_TICKS),
        scenarios,
    }
}

fn replace_synthetic_snapshot_runs(
    scenario: &mut ScenarioResult,
    snapshot_p95_ns: [u64; PERF_REPETITIONS],
) {
    for (repetition, snapshot_ns) in scenario.measurements.iter_mut().zip(snapshot_p95_ns) {
        repetition.snapshot_ns.fill(snapshot_ns);
        repetition.snapshot_p50_ns = snapshot_ns;
        repetition.snapshot_p95_ns = snapshot_ns;
        repetition.snapshot_max_ns = snapshot_ns;
    }
    scenario.median_of_run_snapshot_p95_ns = median_u64(&snapshot_p95_ns);
    scenario.snapshot_p95_cv_pct = coefficient_of_variation_pct(
        &snapshot_p95_ns
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>(),
    );
}

fn assert_self_test_status(label: &str, verdict: &PerfVerdict, expected: VerdictStatus) {
    assert_eq!(
        verdict.status, expected,
        "self-test `{label}` returned {:?}: {:?}",
        verdict.status, verdict.reasons
    );
    println!("perf-gate self-test: {label}: {:?}", verdict.status);
}

fn run_self_test() -> GateResult<()> {
    let live_fingerprint = capture_fingerprint()?;
    let live_class = serde_json::to_vec(&live_fingerprint.class)
        .map_err(|error| format!("failed to serialize live self-test machine class: {error}"))?;
    if live_fingerprint.machine_class_id != stable_hash(&live_class) {
        return Err("live fingerprint machine-class ID is not reproducible".to_owned());
    }
    println!(
        "perf-gate self-test: live Git and machine identity: {}",
        live_fingerprint.machine_class_id
    );

    let mount_fixture = concat!(
        "/dev/disk3s5 on /System/Volumes/Data (apfs, local, journaled)\n",
        "tmpfs on /Volumes/ScriptBotsRAM (tmpfs, local)"
    );
    if macos_mount_filesystem_kind("tmpfs", mount_fixture).as_deref() != Some("tmpfs")
        || macos_mount_filesystem_kind("/dev/disk3s5", mount_fixture).as_deref() != Some("apfs")
        || macos_mount_filesystem_kind("missing", mount_fixture).is_some()
    {
        return Err("macOS mount-table filesystem identity parsing is unstable".to_owned());
    }
    println!("perf-gate self-test: macOS mount-table filesystem identity");

    let linux_memory = memory_class_identity("MemTotal: 16377688 kB")?;
    let page_drift_memory = memory_class_identity("MemTotal: 16377692 kB")?;
    let old_midpoint_memory = memory_class_identity("MemTotal: 16384004 kB")?;
    let macos_memory = memory_class_identity("16770752512")?;
    let larger_memory = memory_class_identity("MemTotal: 33554432 kB")?;
    if linux_memory != page_drift_memory
        || linux_memory != old_midpoint_memory
        || linux_memory != macos_memory
    {
        return Err(
            "reserved-page or platform-format memory drift changed the machine class".to_owned(),
        );
    }
    if linux_memory == larger_memory {
        return Err("materially different memory capacities collapsed into one class".to_owned());
    }
    if memory_class_identity("MemTotal: 16377688 MB").is_ok() {
        return Err("unsupported Linux memory units were accepted".to_owned());
    }
    println!("perf-gate self-test: stable memory capacity class: {linux_memory}");

    let baseline = synthetic_artifact("baseline", "same", [100.0; 5], [1_000_000; 5]);

    let regression = synthetic_artifact("candidate", "same", [89.0; 5], [1_000_000; 5]);
    assert_self_test_status(
        "stable 11% TPS regression",
        &compare_artifacts(&baseline, &regression),
        VerdictStatus::Fail,
    );

    let boundary = synthetic_artifact("candidate", "same", [90.0; 5], [1_000_000; 5]);
    assert_self_test_status(
        "exactly 10% TPS regression",
        &compare_artifacts(&baseline, &boundary),
        VerdictStatus::Pass,
    );

    let snapshot_boundary = synthetic_artifact("candidate", "same", [100.0; 5], [4_000_000; 5]);
    assert_self_test_status(
        "strict 4ms snapshot boundary",
        &compare_artifacts(&baseline, &snapshot_boundary),
        VerdictStatus::Fail,
    );

    let noisy = synthetic_artifact(
        "candidate",
        "same",
        [70.0, 80.0, 80.0, 80.0, 120.0],
        [1_000_000; 5],
    );
    assert_self_test_status(
        "high-CV candidate is advisory",
        &compare_artifacts(&baseline, &noisy),
        VerdictStatus::Advisory,
    );

    let snapshot_noise = [920_000, 1_000_000, 1_000_000, 1_000_000, 1_080_000];
    let mut snapshot_noisy_candidate =
        synthetic_artifact("candidate", "same", [100.0; 5], [1_000_000; 5]);
    replace_synthetic_snapshot_runs(&mut snapshot_noisy_candidate.scenarios[0], snapshot_noise);
    assert_self_test_status(
        "snapshot-only high-CV candidate is advisory",
        &compare_artifacts(&baseline, &snapshot_noisy_candidate),
        VerdictStatus::Advisory,
    );

    let mut snapshot_noisy_baseline = baseline.clone();
    replace_synthetic_snapshot_runs(&mut snapshot_noisy_baseline.scenarios[0], snapshot_noise);
    let stable_candidate = synthetic_artifact("candidate", "same", [100.0; 5], [1_000_000; 5]);
    assert_self_test_status(
        "snapshot-only high-CV baseline is refused",
        &compare_artifacts(&snapshot_noisy_baseline, &stable_candidate),
        VerdictStatus::Refused,
    );

    let mut localized_noise = synthetic_artifact("candidate", "same", [89.0; 5], [1_000_000; 5]);
    let noisy_5k = localized_noise
        .scenarios
        .iter_mut()
        .find(|scenario| scenario.agents == 5_000)
        .expect("full matrix contains 5k scenario");
    replace_synthetic_snapshot_runs(
        noisy_5k,
        [500_000, 1_000_000, 1_000_000, 1_000_000, 2_000_000],
    );
    assert_self_test_status(
        "noisy 5k snapshot does not suppress stable TPS failure",
        &compare_artifacts(&baseline, &localized_noise),
        VerdictStatus::Fail,
    );

    let cross_class = synthetic_artifact("candidate", "different", [100.0; 5], [1_000_000; 5]);
    assert_self_test_status(
        "cross-class comparison is refused",
        &compare_artifacts(&baseline, &cross_class),
        VerdictStatus::Refused,
    );

    let mut tampered = synthetic_artifact("candidate", "same", [100.0; 5], [1_000_000; 5]);
    tampered.scenarios[0].median_of_run_total_tps = 999.0;
    assert_self_test_status(
        "tampered aggregate is refused",
        &compare_artifacts(&baseline, &tampered),
        VerdictStatus::Refused,
    );

    let mut tampered_raw = synthetic_artifact("candidate", "same", [100.0; 5], [1_000_000; 5]);
    tampered_raw.scenarios[0].measurements[0].window_tps[0] = 999.0;
    assert_self_test_status(
        "tampered per-run raw derivation is refused",
        &compare_artifacts(&baseline, &tampered_raw),
        VerdictStatus::Refused,
    );

    let mut missing_snapshot_sample =
        synthetic_artifact("candidate", "same", [100.0; 5], [1_000_000; 5]);
    if missing_snapshot_sample.scenarios[0].measurements[0]
        .snapshot_ns
        .pop()
        .is_none()
    {
        return Err("synthetic snapshot fixture unexpectedly had no raw samples".to_owned());
    }
    assert_self_test_status(
        "missing separately timed snapshot sample is refused",
        &compare_artifacts(&baseline, &missing_snapshot_sample),
        VerdictStatus::Refused,
    );

    let mut tampered_snapshot_policy =
        synthetic_artifact("candidate", "same", [100.0; 5], [1_000_000; 5]);
    tampered_snapshot_policy.policy.snapshot_samples_per_tick = 1;
    tampered_snapshot_policy
        .policy
        .snapshot_samples_per_repetition = PERF_TICKS;
    assert_self_test_status(
        "tampered snapshot sampling policy is refused",
        &compare_artifacts(&baseline, &tampered_snapshot_policy),
        VerdictStatus::Refused,
    );

    let mut tampered_profile_total =
        synthetic_artifact("candidate", "same", [100.0; 5], [1_000_000; 5]);
    tampered_profile_total.scenarios[0].measurements[0].profiled_step_total_ns[0] = 1;
    assert_self_test_status(
        "profile total shorter than aligned stages is refused",
        &compare_artifacts(&baseline, &tampered_profile_total),
        VerdictStatus::Refused,
    );

    let mut tampered_cadence = synthetic_artifact("candidate", "same", [100.0; 5], [1_000_000; 5]);
    let aging = tampered_cadence.scenarios[0].measurements[0]
        .stages
        .iter_mut()
        .find(|stage| stage.stage == WorldStepStage::Aging.as_str())
        .expect("synthetic artifact contains aging stage");
    aging.raw_ns[0] = Some(1);
    aging.executions += 1;
    assert_self_test_status(
        "impossible stage cadence is refused",
        &compare_artifacts(&baseline, &tampered_cadence),
        VerdictStatus::Refused,
    );

    let mut missing_git_identity =
        synthetic_artifact("candidate", "same", [100.0; 5], [1_000_000; 5]);
    missing_git_identity.source_commit.clear();
    missing_git_identity.fingerprint.git_commit.clear();
    assert_self_test_status(
        "missing Git identity is refused",
        &compare_artifacts(&baseline, &missing_git_identity),
        VerdictStatus::Refused,
    );

    let mut changed_science = synthetic_artifact("candidate", "same", [100.0; 5], [1_000_000; 5]);
    let changed_scenario = &mut changed_science.scenarios[0];
    for repetition in changed_scenario
        .warmups
        .iter_mut()
        .chain(&mut changed_scenario.measurements)
    {
        repetition.final_digest = "synthetic-different-science".to_owned();
    }
    assert_self_test_status(
        "cross-artifact science drift is refused",
        &compare_artifacts(&baseline, &changed_science),
        VerdictStatus::Refused,
    );

    let mut empty = synthetic_artifact("candidate", "same", [100.0; 5], [1_000_000; 5]);
    empty.scenarios.clear();
    assert_self_test_status(
        "empty scenario set is refused",
        &compare_artifacts(&baseline, &empty),
        VerdictStatus::Refused,
    );
    Ok(())
}

fn run_criterion() {
    let mut criterion = Criterion::default().configure_from_args();
    bench_brain_protocol_cohorts(&mut criterion);
    bench_world_steps(&mut criterion);
    bench_hydrology_map_generation(&mut criterion);
    criterion.final_summary();
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if let Some(perf_gate_index) = args.iter().position(|arg| arg == "--perf-gate") {
        let mut gate_args = &args[perf_gate_index + 1..];
        if gate_args.last().is_some_and(|arg| arg == "--bench") {
            gate_args = &gate_args[..gate_args.len() - 1];
        }
        let result = GateArgs::parse(gate_args).and_then(run_perf_gate);
        match result {
            Ok(exit_code) => std::process::exit(exit_code),
            Err(error) => {
                eprintln!("perf-gate error: {error}");
                std::process::exit(2);
            }
        }
    }
    run_criterion();
}
