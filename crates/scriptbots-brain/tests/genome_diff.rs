//! Protocol genome-diff integration proofs (bd-16g.13.1).
//!
//! These tests exercise `BrainFamilyCodec::genome_loci` and `diff_genomes` through the
//! real MLP family adapter. They live in the brain crate deliberately: the brain crate's
//! dependency on scriptbots-core uses `default-features = false`, so trait identities
//! stay single-version here, while inline core tests would build a second core instance
//! (default features) and make the same trait name two different types.

use scriptbots_brain::assembly::AssemblyFamilyAdapter;
use scriptbots_brain::dwraon::DwraonFamilyAdapter;
use scriptbots_brain::mlp::MlpBrainFamily;
use scriptbots_core::genome_diff::{
    DeltaKind, GenomeDelta, GenomeDiffError, Locus, LocusValue, diff_genomes,
};
use scriptbots_core::{
    AgentData, BrainAdapterIdentityV1, BrainEvaluator, BrainEvaluatorStateEnvelope,
    BrainFamilyAdapter, BrainFamilyCodec, BrainFamilyId, BrainGenomeEnvelope, BrainGenomeMaterial,
    BrainHeredityCapabilityV1, BrainHeredityExclusionV1, BrainProtocolError, BrainProvenance,
    MutationRates, OffspringStatePolicy, Position, RandomStream, ScriptBotsConfig, SmallRngStream,
    WorldState,
};

fn adapter() -> MlpBrainFamily {
    MlpBrainFamily::new()
}

fn seeded_stream(seed: u64) -> SmallRngStream {
    SmallRngStream::seed_from_u64(seed)
}

#[test]
fn identical_genomes_produce_an_empty_diff_meaning_no_mutations() {
    let adapter = adapter();
    let parent = adapter
        .random_genome(BrainProvenance::default(), &mut seeded_stream(0xA11CE))
        .expect("parent genome");
    let child = parent.clone();
    let diff = diff_genomes(&adapter, &parent, &child).expect("diff");
    assert!(
        diff.deltas.is_empty(),
        "an identical genome must diff to exactly zero deltas — the heredity proof \
         depends on this sentence being true"
    );
    assert_eq!(diff.summary.changed_loci, 0);
    assert!(diff.summary.total_loci > 0);
    assert!(diff.summary.by_kind.is_empty());
}

#[test]
fn a_full_rate_mutation_produces_typed_deltas_matching_manual_loci() {
    let adapter = adapter();
    let parent = adapter
        .random_genome(BrainProvenance::default(), &mut seeded_stream(0xB0B))
        .expect("parent genome");
    let child = adapter
        .mutate_genome(
            &parent,
            MutationRates {
                primary: 1.0,
                secondary: 1.0,
            },
            BrainProvenance::default(),
            &mut seeded_stream(0xC1C1),
        )
        .expect("mutated child genome");

    let parent_loci = adapter.genome_loci(&parent).expect("parent loci");
    let child_loci = adapter.genome_loci(&child).expect("child loci");
    let diff = diff_genomes(&adapter, &parent, &child).expect("diff");

    let expected: Vec<GenomeDelta> = parent_loci
        .iter()
        .zip(child_loci.iter())
        .filter_map(|((locus, before), (_, after))| match (*before, *after) {
            (LocusValue::Scalar(b), LocusValue::Scalar(a)) if b.to_bits() != a.to_bits() => {
                Some(GenomeDelta::Scalar {
                    locus: *locus,
                    before: b,
                    after: a,
                })
            }
            (LocusValue::Target(b), LocusValue::Target(a)) if b != a => {
                Some(GenomeDelta::Retarget {
                    locus: *locus,
                    before: b,
                    after: a,
                })
            }
            (LocusValue::Kind(b), LocusValue::Kind(a)) if b != a => Some(GenomeDelta::KindFlip {
                locus: *locus,
                before: b,
                after: a,
            }),
            _ => None,
        })
        .collect();
    assert_eq!(
        diff.deltas, expected,
        "the diff must equal the manual locus comparison exactly"
    );
    assert!(
        diff.deltas.iter().any(|delta| matches!(
            delta,
            GenomeDelta::KindFlip { .. } | GenomeDelta::Retarget { .. }
        )),
        "a full-rate mutation must exercise kind and target loci"
    );
    assert_eq!(diff.summary.changed_loci, expected.len());
    assert!(diff.summary.l1 > 0.0);
    assert!(diff.summary.linf > 0.0 && diff.summary.linf <= diff.summary.l1);
}

#[test]
fn cross_family_and_cross_schema_diffs_are_typed_errors() {
    let adapter = adapter();
    let parent = adapter
        .random_genome(BrainProvenance::default(), &mut seeded_stream(7))
        .expect("parent genome");
    let other_family = BrainGenomeEnvelope::new(
        BrainFamilyId::new("dwraon-baseline").expect("family id"),
        1,
        1,
        vec![0, 1, 2, 3],
        BrainProvenance::default(),
    )
    .expect("other family envelope");
    let error =
        diff_genomes(&adapter, &parent, &other_family).expect_err("cross-family diff must refuse");
    assert!(
        matches!(error, GenomeDiffError::FamilyMismatch { .. }),
        "expected FamilyMismatch, got {error}"
    );

    let other_schema = BrainGenomeEnvelope::new(
        parent.family_id().clone(),
        parent.schema_version() + 1,
        parent.codec_version(),
        parent.payload().to_vec(),
        BrainProvenance::default(),
    )
    .expect("other schema envelope");
    let error =
        diff_genomes(&adapter, &parent, &other_schema).expect_err("cross-schema diff must refuse");
    assert!(
        matches!(error, GenomeDiffError::SchemaMismatch { .. }),
        "expected SchemaMismatch, got {error}"
    );
}

#[test]
fn dwraon_full_rate_mutation_produces_typed_deltas() {
    let adapter = DwraonFamilyAdapter::default();
    let parent = adapter
        .random_genome(BrainProvenance::default(), &mut seeded_stream(0xD00D))
        .expect("dwraon parent genome");
    let child = adapter
        .mutate_genome(
            &parent,
            MutationRates {
                primary: 1.0,
                secondary: 1.0,
            },
            BrainProvenance::default(),
            &mut seeded_stream(0xDAD),
        )
        .expect("dwraon mutated child genome");

    let parent_loci = adapter.genome_loci(&parent).expect("dwraon parent loci");
    let child_loci = adapter.genome_loci(&child).expect("dwraon child loci");
    let diff = diff_genomes(&adapter, &parent, &child).expect("dwraon diff");

    let expected: Vec<GenomeDelta> = parent_loci
        .iter()
        .zip(child_loci.iter())
        .filter_map(|((locus, before), (_, after))| match (*before, *after) {
            (LocusValue::Scalar(b), LocusValue::Scalar(a)) if b.to_bits() != a.to_bits() => {
                Some(GenomeDelta::Scalar {
                    locus: *locus,
                    before: b,
                    after: a,
                })
            }
            (LocusValue::Target(b), LocusValue::Target(a)) if b != a => {
                Some(GenomeDelta::Retarget {
                    locus: *locus,
                    before: b,
                    after: a,
                })
            }
            (LocusValue::Kind(b), LocusValue::Kind(a)) if b != a => Some(GenomeDelta::KindFlip {
                locus: *locus,
                before: b,
                after: a,
            }),
            _ => None,
        })
        .collect();
    assert_eq!(
        diff.deltas, expected,
        "the dwraon diff must equal the manual locus comparison exactly"
    );
    assert!(
        diff.deltas.iter().any(|delta| matches!(
            delta,
            GenomeDelta::KindFlip { .. } | GenomeDelta::Retarget { .. }
        )),
        "a full-rate dwraon mutation must exercise kind and source loci"
    );
    assert_eq!(diff.summary.changed_loci, expected.len());
}

#[test]
fn assembly_full_rate_mutation_produces_typed_cell_deltas() {
    let adapter = AssemblyFamilyAdapter::new().expect("assembly adapter");
    let parent = adapter
        .random_genome(BrainProvenance::default(), &mut seeded_stream(0xABBA))
        .expect("assembly parent genome");
    let child = adapter
        .mutate_genome(
            &parent,
            MutationRates {
                primary: 1.0,
                secondary: 1.0,
            },
            BrainProvenance::default(),
            &mut seeded_stream(0xCAFE),
        )
        .expect("assembly mutated child genome");

    let parent_loci = adapter.genome_loci(&parent).expect("assembly parent loci");
    let child_loci = adapter.genome_loci(&child).expect("assembly child loci");
    let diff = diff_genomes(&adapter, &parent, &child).expect("assembly diff");

    let expected: Vec<GenomeDelta> = parent_loci
        .iter()
        .zip(child_loci.iter())
        .filter_map(|((locus, before), (_, after))| match (*before, *after) {
            (LocusValue::Scalar(b), LocusValue::Scalar(a)) if b.to_bits() != a.to_bits() => {
                Some(GenomeDelta::Scalar {
                    locus: *locus,
                    before: b,
                    after: a,
                })
            }
            _ => None,
        })
        .collect();

    assert_eq!(
        diff.deltas, expected,
        "the assembly diff must equal the manual cell locus comparison"
    );
    assert_eq!(diff.summary.changed_loci, expected.len());
    assert!(diff.summary.changed_loci > 0);
    assert!(diff.deltas.iter().all(|d| matches!(
        d,
        GenomeDelta::Scalar {
            locus: Locus::Cell(_),
            ..
        }
    )));
}

#[test]
fn single_locus_mutation_fixtures_produce_exact_typed_deltas() {
    struct MockCodec {
        family_id: BrainFamilyId,
        parent_loci: Vec<(Locus, LocusValue)>,
        child_loci: Vec<(Locus, LocusValue)>,
    }

    impl BrainFamilyCodec for MockCodec {
        fn family_id(&self) -> &BrainFamilyId {
            &self.family_id
        }
        fn adapter_identity(&self) -> BrainAdapterIdentityV1 {
            BrainAdapterIdentityV1::from_semantic_descriptor(&self.family_id, 1, b"mock-adapter")
        }
        fn heredity_capability(&self) -> BrainHeredityCapabilityV1 {
            BrainHeredityCapabilityV1::excluded(BrainHeredityExclusionV1::NoCanonicalLocusSchema)
        }
        fn random_genome_material(
            &self,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
            BrainGenomeMaterial::new(1, 1, vec![0; 4])
        }
        fn validate_genome(&self, _genome: &BrainGenomeEnvelope) -> Result<(), BrainProtocolError> {
            Ok(())
        }
        fn genome_loci(
            &self,
            genome: &BrainGenomeEnvelope,
        ) -> Result<Vec<(Locus, LocusValue)>, BrainProtocolError> {
            if genome.payload() == [0] {
                Ok(self.parent_loci.clone())
            } else {
                Ok(self.child_loci.clone())
            }
        }
        fn validate_evaluator_state(
            &self,
            _state: &BrainEvaluatorStateEnvelope,
        ) -> Result<(), BrainProtocolError> {
            Ok(())
        }
        fn mutate_genome_material(
            &self,
            _genome: &BrainGenomeEnvelope,
            _rates: MutationRates,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
            BrainGenomeMaterial::new(1, 1, vec![1; 4])
        }
        fn crossover_genomes_material(
            &self,
            _left: &BrainGenomeEnvelope,
            _right: &BrainGenomeEnvelope,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
            BrainGenomeMaterial::new(1, 1, vec![2; 4])
        }
        fn initial_state(
            &self,
            _genome: &BrainGenomeEnvelope,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
            BrainEvaluatorStateEnvelope::new(self.family_id.clone(), 1, 1, vec![0])
        }
        fn offspring_state_policy(&self) -> OffspringStatePolicy {
            OffspringStatePolicy::Reset
        }
        fn offspring_state(
            &self,
            genome: &BrainGenomeEnvelope,
            _parents: &[&BrainEvaluatorStateEnvelope],
            rng: &mut dyn RandomStream,
        ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
            self.initial_state(genome, rng)
        }
        fn evaluator(
            &self,
            _genome: &BrainGenomeEnvelope,
            _state: &BrainEvaluatorStateEnvelope,
        ) -> Result<Box<dyn BrainEvaluator>, BrainProtocolError> {
            unimplemented!()
        }
    }

    let family_id = BrainFamilyId::new("mock-family").expect("valid family id");
    let parent_env =
        BrainGenomeEnvelope::new(family_id.clone(), 1, 1, vec![0], BrainProvenance::default())
            .expect("parent envelope");
    let child_env =
        BrainGenomeEnvelope::new(family_id.clone(), 1, 1, vec![1], BrainProvenance::default())
            .expect("child envelope");

    // 1. Bias fixture
    let codec = MockCodec {
        family_id: family_id.clone(),
        parent_loci: vec![(Locus::NodeBias(3), LocusValue::Scalar(0.2))],
        child_loci: vec![(Locus::NodeBias(3), LocusValue::Scalar(0.7))],
    };
    let diff = diff_genomes(&codec, &parent_env, &child_env).expect("bias diff");
    assert_eq!(
        diff.deltas,
        vec![GenomeDelta::Scalar {
            locus: Locus::NodeBias(3),
            before: 0.2,
            after: 0.7,
        }]
    );
    assert_eq!(diff.summary.by_kind.get(&DeltaKind::Scalar), Some(&1));

    // 2. Damping boundary fixture
    let codec = MockCodec {
        family_id: family_id.clone(),
        parent_loci: vec![(Locus::NodeDamping(5), LocusValue::Scalar(0.0))],
        child_loci: vec![(Locus::NodeDamping(5), LocusValue::Scalar(1.0))],
    };
    let diff = diff_genomes(&codec, &parent_env, &child_env).expect("damping diff");
    assert_eq!(
        diff.deltas,
        vec![GenomeDelta::Scalar {
            locus: Locus::NodeDamping(5),
            before: 0.0,
            after: 1.0,
        }]
    );

    // 3. Gain floor fixture
    let codec = MockCodec {
        family_id: family_id.clone(),
        parent_loci: vec![(Locus::NodeGain(2), LocusValue::Scalar(0.01))],
        child_loci: vec![(Locus::NodeGain(2), LocusValue::Scalar(5.0))],
    };
    let diff = diff_genomes(&codec, &parent_env, &child_env).expect("gain diff");
    assert_eq!(
        diff.deltas,
        vec![GenomeDelta::Scalar {
            locus: Locus::NodeGain(2),
            before: 0.01,
            after: 5.0,
        }]
    );

    // 4. Weight fixture
    let codec = MockCodec {
        family_id: family_id.clone(),
        parent_loci: vec![(
            Locus::NodeWeight { node: 10, conn: 1 },
            LocusValue::Scalar(-0.5),
        )],
        child_loci: vec![(
            Locus::NodeWeight { node: 10, conn: 1 },
            LocusValue::Scalar(0.8),
        )],
    };
    let diff = diff_genomes(&codec, &parent_env, &child_env).expect("weight diff");
    assert_eq!(
        diff.deltas,
        vec![GenomeDelta::Scalar {
            locus: Locus::NodeWeight { node: 10, conn: 1 },
            before: -0.5,
            after: 0.8,
        }]
    );

    // 5. Kind flip fixture
    let codec = MockCodec {
        family_id: family_id.clone(),
        parent_loci: vec![(Locus::NodeKind { node: 8, conn: 0 }, LocusValue::Kind(1))],
        child_loci: vec![(Locus::NodeKind { node: 8, conn: 0 }, LocusValue::Kind(4))],
    };
    let diff = diff_genomes(&codec, &parent_env, &child_env).expect("kind flip diff");
    assert_eq!(
        diff.deltas,
        vec![GenomeDelta::KindFlip {
            locus: Locus::NodeKind { node: 8, conn: 0 },
            before: 1,
            after: 4,
        }]
    );
    assert_eq!(diff.summary.by_kind.get(&DeltaKind::KindFlip), Some(&1));

    // 6. Retarget fixture
    let codec = MockCodec {
        family_id: family_id.clone(),
        parent_loci: vec![(
            Locus::NodeTarget { node: 8, conn: 0 },
            LocusValue::Target(88),
        )],
        child_loci: vec![(
            Locus::NodeTarget { node: 8, conn: 0 },
            LocusValue::Target(14),
        )],
    };
    let diff = diff_genomes(&codec, &parent_env, &child_env).expect("retarget diff");
    assert_eq!(
        diff.deltas,
        vec![GenomeDelta::Retarget {
            locus: Locus::NodeTarget { node: 8, conn: 0 },
            before: 88,
            after: 14,
        }]
    );
    assert_eq!(diff.summary.by_kind.get(&DeltaKind::Retarget), Some(&1));

    // 7. Assembly cell fixture
    let codec = MockCodec {
        family_id: family_id.clone(),
        parent_loci: vec![(Locus::Cell(42), LocusValue::Scalar(10.0))],
        child_loci: vec![(Locus::Cell(42), LocusValue::Scalar(25.0))],
    };
    let diff = diff_genomes(&codec, &parent_env, &child_env).expect("cell diff");
    assert_eq!(
        diff.deltas,
        vec![GenomeDelta::Scalar {
            locus: Locus::Cell(42),
            before: 10.0,
            after: 25.0,
        }]
    );

    // 8. Hyperparameter fixture
    let codec = MockCodec {
        family_id: family_id.clone(),
        parent_loci: vec![(Locus::Hyper(0), LocusValue::Scalar(0.01))],
        child_loci: vec![(Locus::Hyper(0), LocusValue::Scalar(0.05))],
    };
    let diff = diff_genomes(&codec, &parent_env, &child_env).expect("hyper diff");
    assert_eq!(
        diff.deltas,
        vec![GenomeDelta::Scalar {
            locus: Locus::Hyper(0),
            before: 0.01,
            after: 0.05,
        }]
    );

    // 9. Exactly one f32 ULP difference
    let before_ulp = 1.0_f32;
    let after_ulp = f32::from_bits(before_ulp.to_bits() + 1);
    let codec = MockCodec {
        family_id: family_id.clone(),
        parent_loci: vec![(Locus::NodeBias(0), LocusValue::Scalar(before_ulp))],
        child_loci: vec![(Locus::NodeBias(0), LocusValue::Scalar(after_ulp))],
    };
    let diff = diff_genomes(&codec, &parent_env, &child_env).expect("1-ulp diff");
    assert_eq!(diff.summary.changed_loci, 1);
    assert!(diff.summary.l1 > 0.0);
    assert_eq!(
        diff.deltas,
        vec![GenomeDelta::Scalar {
            locus: Locus::NodeBias(0),
            before: before_ulp,
            after: after_ulp,
        }]
    );
}

#[test]
fn negative_control_swapped_locus_order_fails_closed_in_release_mode() {
    // A sabotaged codec emits loci out-of-order in the child:
    // Parent has [NodeBias(0), NodeBias(1)]
    // Child has [NodeBias(1), NodeBias(0)]
    // Equal length, matching value kinds, but mismatched locus address sequence.
    // In release mode (without debug_assert), diff_genomes MUST return GenomeDiffError::LocusMismatch.
    struct SwappingCodec {
        family_id: BrainFamilyId,
    }

    impl BrainFamilyCodec for SwappingCodec {
        fn family_id(&self) -> &BrainFamilyId {
            &self.family_id
        }
        fn adapter_identity(&self) -> BrainAdapterIdentityV1 {
            BrainAdapterIdentityV1::from_semantic_descriptor(
                &self.family_id,
                1,
                b"swapping-adapter",
            )
        }
        fn heredity_capability(&self) -> BrainHeredityCapabilityV1 {
            BrainHeredityCapabilityV1::excluded(BrainHeredityExclusionV1::NoCanonicalLocusSchema)
        }
        fn random_genome_material(
            &self,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
            BrainGenomeMaterial::new(1, 1, vec![0; 4])
        }
        fn validate_genome(&self, _genome: &BrainGenomeEnvelope) -> Result<(), BrainProtocolError> {
            Ok(())
        }
        fn genome_loci(
            &self,
            genome: &BrainGenomeEnvelope,
        ) -> Result<Vec<(Locus, LocusValue)>, BrainProtocolError> {
            if genome.payload() == [0] {
                Ok(vec![
                    (Locus::NodeBias(0), LocusValue::Scalar(1.0)),
                    (Locus::NodeBias(1), LocusValue::Scalar(2.0)),
                ])
            } else {
                // Swapped address order!
                Ok(vec![
                    (Locus::NodeBias(1), LocusValue::Scalar(1.0)),
                    (Locus::NodeBias(0), LocusValue::Scalar(2.0)),
                ])
            }
        }
        fn validate_evaluator_state(
            &self,
            _state: &BrainEvaluatorStateEnvelope,
        ) -> Result<(), BrainProtocolError> {
            Ok(())
        }
        fn mutate_genome_material(
            &self,
            _genome: &BrainGenomeEnvelope,
            _rates: MutationRates,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
            BrainGenomeMaterial::new(1, 1, vec![1; 4])
        }
        fn crossover_genomes_material(
            &self,
            _left: &BrainGenomeEnvelope,
            _right: &BrainGenomeEnvelope,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
            BrainGenomeMaterial::new(1, 1, vec![2; 4])
        }
        fn initial_state(
            &self,
            _genome: &BrainGenomeEnvelope,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
            BrainEvaluatorStateEnvelope::new(self.family_id.clone(), 1, 1, vec![0])
        }
        fn offspring_state_policy(&self) -> OffspringStatePolicy {
            OffspringStatePolicy::Reset
        }
        fn offspring_state(
            &self,
            genome: &BrainGenomeEnvelope,
            _parents: &[&BrainEvaluatorStateEnvelope],
            rng: &mut dyn RandomStream,
        ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
            self.initial_state(genome, rng)
        }
        fn evaluator(
            &self,
            _genome: &BrainGenomeEnvelope,
            _state: &BrainEvaluatorStateEnvelope,
        ) -> Result<Box<dyn BrainEvaluator>, BrainProtocolError> {
            unimplemented!()
        }
    }

    let family_id = BrainFamilyId::new("swapped-family").expect("valid family id");
    let parent_env =
        BrainGenomeEnvelope::new(family_id.clone(), 1, 1, vec![0], BrainProvenance::default())
            .expect("parent envelope");
    let child_env =
        BrainGenomeEnvelope::new(family_id.clone(), 1, 1, vec![1], BrainProvenance::default())
            .expect("child envelope");

    let codec = SwappingCodec { family_id };
    let error =
        diff_genomes(&codec, &parent_env, &child_env).expect_err("must reject swapped locus order");
    match error {
        GenomeDiffError::LocusMismatch {
            index,
            parent,
            child,
            ..
        } => {
            assert_eq!(index, 0);
            assert_eq!(parent, Locus::NodeBias(0));
            assert_eq!(child, Locus::NodeBias(1));
        }
        other => panic!("expected LocusMismatch, got {other:?}"),
    }
}

#[test]
fn negative_control_value_type_mismatch_fails_closed() {
    struct TypeMismatchCodec {
        family_id: BrainFamilyId,
    }

    impl BrainFamilyCodec for TypeMismatchCodec {
        fn family_id(&self) -> &BrainFamilyId {
            &self.family_id
        }
        fn adapter_identity(&self) -> BrainAdapterIdentityV1 {
            BrainAdapterIdentityV1::from_semantic_descriptor(
                &self.family_id,
                1,
                b"type-mismatch-adapter",
            )
        }
        fn heredity_capability(&self) -> BrainHeredityCapabilityV1 {
            BrainHeredityCapabilityV1::excluded(BrainHeredityExclusionV1::NoCanonicalLocusSchema)
        }
        fn random_genome_material(
            &self,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
            BrainGenomeMaterial::new(1, 1, vec![0; 4])
        }
        fn validate_genome(&self, _genome: &BrainGenomeEnvelope) -> Result<(), BrainProtocolError> {
            Ok(())
        }
        fn genome_loci(
            &self,
            genome: &BrainGenomeEnvelope,
        ) -> Result<Vec<(Locus, LocusValue)>, BrainProtocolError> {
            if genome.payload() == [0] {
                Ok(vec![(Locus::NodeBias(0), LocusValue::Scalar(1.0))])
            } else {
                Ok(vec![(Locus::NodeBias(0), LocusValue::Target(5))])
            }
        }
        fn validate_evaluator_state(
            &self,
            _state: &BrainEvaluatorStateEnvelope,
        ) -> Result<(), BrainProtocolError> {
            Ok(())
        }
        fn mutate_genome_material(
            &self,
            _genome: &BrainGenomeEnvelope,
            _rates: MutationRates,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
            BrainGenomeMaterial::new(1, 1, vec![1; 4])
        }
        fn crossover_genomes_material(
            &self,
            _left: &BrainGenomeEnvelope,
            _right: &BrainGenomeEnvelope,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
            BrainGenomeMaterial::new(1, 1, vec![2; 4])
        }
        fn initial_state(
            &self,
            _genome: &BrainGenomeEnvelope,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
            BrainEvaluatorStateEnvelope::new(self.family_id.clone(), 1, 1, vec![0])
        }
        fn offspring_state_policy(&self) -> OffspringStatePolicy {
            OffspringStatePolicy::Reset
        }
        fn offspring_state(
            &self,
            genome: &BrainGenomeEnvelope,
            _parents: &[&BrainEvaluatorStateEnvelope],
            rng: &mut dyn RandomStream,
        ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
            self.initial_state(genome, rng)
        }
        fn evaluator(
            &self,
            _genome: &BrainGenomeEnvelope,
            _state: &BrainEvaluatorStateEnvelope,
        ) -> Result<Box<dyn BrainEvaluator>, BrainProtocolError> {
            unimplemented!()
        }
    }

    let family_id = BrainFamilyId::new("type-mismatch-family").expect("valid family id");
    let parent_env =
        BrainGenomeEnvelope::new(family_id.clone(), 1, 1, vec![0], BrainProvenance::default())
            .expect("parent envelope");
    let child_env =
        BrainGenomeEnvelope::new(family_id.clone(), 1, 1, vec![1], BrainProvenance::default())
            .expect("child envelope");

    let codec = TypeMismatchCodec { family_id };
    let error =
        diff_genomes(&codec, &parent_env, &child_env).expect_err("must reject value type mismatch");
    match error {
        GenomeDiffError::ValueTypeMismatch {
            index,
            locus,
            parent_type,
            child_type,
        } => {
            assert_eq!(index, 0);
            assert_eq!(locus, Locus::NodeBias(0));
            assert_eq!(parent_type, "scalar");
            assert_eq!(child_type, "target");
        }
        other => panic!("expected ValueTypeMismatch, got {other:?}"),
    }
}

#[test]
fn sanctioned_world_path_e2e_diff_and_logging() {
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        reproduction_energy_threshold: 0.5,
        reproduction_energy_cost: 0.0,
        reproduction_cooldown: 1,
        reproduction_attempt_interval: 1,
        reproduction_attempt_chance: 1.0,
        reproduction_child_energy: 1.0,
        reproduction_spawn_jitter: 0.0,
        reproduction_color_jitter: 0.0,
        reproduction_partner_chance: 0.0,
        reproduction_meta_mutation_chance: 0.0,
        reproduction_meta_mutation_scale: 0.0,
        closed: true,
        rng_seed: Some(0xDEAD_BEEF),
        ..ScriptBotsConfig::default()
    })
    .expect("world");

    let family_kind = MlpBrainFamily::new().family_id().as_str().to_owned();
    let family_key = world
        .register_brain_family(family_kind, Box::new(MlpBrainFamily::new()))
        .expect("register family");

    let parent_id = world
        .try_spawn_agent(AgentData {
            position: Position::new(100.0, 100.0),
            ..AgentData::default()
        })
        .expect("parent spawned");
    assert!(
        world
            .bind_agent_brain(parent_id, family_key)
            .expect("bind parent brain"),
        "must bind parent brain"
    );
    world
        .try_update_agent_runtime(parent_id, |runtime| {
            runtime.energy = 1.5;
            runtime.mutation_rates = MutationRates {
                primary: 0.1,
                secondary: 0.1,
            };
        })
        .expect("parent runtime updated");

    let parent_uid = world.agent_uid(parent_id).expect("parent uid");
    let mut child_id = None;
    for _ in 0..64 {
        let completion = world.step_outcome().expect("reproduction step");
        for birth in &completion.outcome.births {
            if birth.parent_a == Some(parent_uid) {
                let id = world
                    .agents()
                    .iter_handles()
                    .find(|h| world.agent_uid(*h) == Some(birth.agent_uid))
                    .expect("child id");
                child_id = Some(id);
                break;
            }
        }
        if child_id.is_some() {
            break;
        }
    }
    let child_id = child_id.expect("child must be born");

    // Obtain parent and child genome envelopes through the sanctioned world path
    let parent_genome = world.agent_brain_genome(parent_id).expect("parent genome");
    let child_genome = world.agent_brain_genome(child_id).expect("child genome");
    let codec = world
        .brain_registry()
        .family(family_key)
        .expect("family codec");

    let diff =
        diff_genomes(codec, parent_genome, child_genome).expect("diff via sanctioned world path");
    assert_eq!(diff.family, *parent_genome.family_id());
    assert_eq!(diff.schema_version, parent_genome.schema_version());
    assert_eq!(diff.summary.total_loci, 3000);
    assert!(diff.summary.changed_loci > 0);

    // Logging verification
    let max_display_deltas = 5;
    let truncated = diff.deltas.len() > max_display_deltas;
    let mut log_lines = Vec::new();
    log_lines.push(format!("family={}", diff.family));
    log_lines.push(format!("schema_version={}", diff.schema_version));
    log_lines.push(format!(
        "changed_loci={}/{}",
        diff.summary.changed_loci, diff.summary.total_loci
    ));
    log_lines.push(format!("by_kind={:?}", diff.summary.by_kind));
    for (i, delta) in diff.deltas.iter().take(max_display_deltas).enumerate() {
        log_lines.push(format!("  delta[{i}]={delta:?}"));
    }
    if truncated {
        log_lines.push(format!(
            "  ... ({} deltas truncated)",
            diff.deltas.len() - max_display_deltas
        ));
    }
    let formatted_log = log_lines.join("\n");
    assert!(formatted_log.contains("family=mlp-baseline"));
    assert!(formatted_log.contains("schema_version=1"));
    assert!(formatted_log.contains("changed_loci="));
}

#[test]
fn locus_round_trip_is_bit_exact() {
    let mlp = adapter();
    let parent_mlp = mlp
        .random_genome(BrainProvenance::default(), &mut seeded_stream(99))
        .expect("parent genome");
    let first = mlp.genome_loci(&parent_mlp).expect("first decode");
    let second = mlp.genome_loci(&parent_mlp).expect("second decode");
    assert_eq!(first, second, "mlp decode is deterministic");

    let dwraon = DwraonFamilyAdapter::default();
    let parent_dwraon = dwraon
        .random_genome(BrainProvenance::default(), &mut seeded_stream(99))
        .expect("parent dwraon");
    let first = dwraon.genome_loci(&parent_dwraon).expect("first decode");
    let second = dwraon.genome_loci(&parent_dwraon).expect("second decode");
    assert_eq!(first, second, "dwraon decode is deterministic");

    let assembly = AssemblyFamilyAdapter::new().expect("assembly adapter");
    let parent_assembly = assembly
        .random_genome(BrainProvenance::default(), &mut seeded_stream(99))
        .expect("parent assembly");
    let first = assembly
        .genome_loci(&parent_assembly)
        .expect("first decode");
    let second = assembly
        .genome_loci(&parent_assembly)
        .expect("second decode");
    assert_eq!(first, second, "assembly decode is deterministic");
}
