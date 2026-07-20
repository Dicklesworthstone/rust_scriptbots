//! Protocol genome-diff integration proofs (bd-16g.13.1).
//!
//! These tests exercise `BrainFamilyCodec::genome_loci` and `diff_genomes` through the
//! real MLP family adapter. They live in the brain crate deliberately: the brain crate's
//! dependency on scriptbots-core uses `default-features = false`, so trait identities
//! stay single-version here, while inline core tests would build a second core instance
//! (default features) and make the same trait name two different types.

use scriptbots_brain::dwraon::DwraonFamilyAdapter;
use scriptbots_brain::mlp::MlpBrainFamily;
use scriptbots_core::genome_diff::{GenomeDelta, GenomeDiffError, LocusValue, diff_genomes};
use scriptbots_core::{
    BrainFamilyAdapter, BrainFamilyCodec, BrainFamilyId, BrainGenomeEnvelope, BrainProvenance,
    MutationRates, SmallRngStream,
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
fn locus_round_trip_is_bit_exact() {
    let adapter = adapter();
    let parent = adapter
        .random_genome(BrainProvenance::default(), &mut seeded_stream(99))
        .expect("parent genome");
    let first = adapter.genome_loci(&parent).expect("first decode");
    let second = adapter.genome_loci(&parent).expect("second decode");
    assert_eq!(first, second, "decode is deterministic");
    assert!(
        first
            .iter()
            .zip(second.iter())
            .all(|((_, a), (_, b))| match (a, b) {
                (LocusValue::Scalar(x), LocusValue::Scalar(y)) => x.to_bits() == y.to_bits(),
                other => other.0 == other.1,
            }),
        "every locus value is bit-exact across decodes"
    );
}
