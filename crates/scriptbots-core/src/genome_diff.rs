//! Protocol-level genome locus addressing and structural diffing (bd-16g.13.1).
//!
//! Evolution in this project happens at a level nothing could observe until now: the
//! genome. This module defines the canonical locus address space ([`Locus`]), the typed
//! delta vocabulary ([`GenomeDelta`]), and the only sanctioned comparison
//! ([`diff_genomes`]) — all computed from protocol envelopes through the family codec's
//! [`BrainFamilyCodec::genome_loci`], never from live brain objects, never by text-diffing
//! serialized dumps. A reordered field must never read as a mutation, and a retarget from
//! 88 to 14 must read as exactly one typed [`GenomeDelta::Retarget`].

use crate::{BrainFamilyCodec, BrainFamilyId, BrainGenomeEnvelope};
use std::collections::BTreeMap;

#[cfg(test)]
mod tests {
    use super::Locus;

    #[test]
    fn locus_human_strings_name_the_gene() {
        assert_eq!(Locus::NodeBias(47).human(), "node 47 bias");
        assert_eq!(
            Locus::NodeWeight { node: 47, conn: 2 }.human(),
            "node 47 weight 2"
        );
        assert_eq!(
            Locus::NodeKind { node: 130, conn: 1 }.human(),
            "node 130 conn 1 kind"
        );
        assert_eq!(
            Locus::NodeTarget { node: 3, conn: 0 }.human(),
            "node 3 conn 0 target"
        );
    }
}

/// One addressable genome locus, in canonical order (index order, never a hash walk).
/// Two genomes of the same family and schema produce loci in the same order, so the diff
/// is a single aligned pass and its bytes are reproducible on any platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Locus {
    NodeBias(u32),
    NodeDamping(u32),
    NodeGain(u32),
    NodeWeight {
        node: u32,
        conn: u8,
    },
    NodeKind {
        node: u32,
        conn: u8,
    },
    NodeTarget {
        node: u32,
        conn: u8,
    },
    /// A family-level hyperparameter locus (reserved; families name their own).
    Hyper(u8),
}

impl Locus {
    /// The string the UI shows ("node 47 weight 2", "node 130 conn 1 kind").
    #[must_use]
    pub fn human(&self) -> String {
        match *self {
            Self::NodeBias(node) => format!("node {node} bias"),
            Self::NodeDamping(node) => format!("node {node} damping"),
            Self::NodeGain(node) => format!("node {node} gain"),
            Self::NodeWeight { node, conn } => format!("node {node} weight {conn}"),
            Self::NodeKind { node, conn } => format!("node {node} conn {conn} kind"),
            Self::NodeTarget { node, conn } => format!("node {node} conn {conn} target"),
            Self::Hyper(id) => format!("hyperparameter {id}"),
        }
    }
}

/// One decoded locus value. Float comparison for scalars is BITWISE (`f32::to_bits`) —
/// heredity is an exact claim, and an epsilon-tolerant diff would hide exactly the
/// corruption it exists to find.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocusValue {
    Scalar(f32),
    Target(u32),
    Kind(u8),
}

/// One typed change between two aligned genomes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GenomeDelta {
    Scalar {
        locus: Locus,
        before: f32,
        after: f32,
    },
    Retarget {
        locus: Locus,
        before: u32,
        after: u32,
    },
    KindFlip {
        locus: Locus,
        before: u8,
        after: u8,
    },
}

/// The kind of a delta, for the by-kind summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeltaKind {
    Scalar,
    Retarget,
    KindFlip,
}

/// Aggregate diff statistics, computed in one pass over the canonical delta list.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffSummary {
    pub changed_loci: usize,
    pub total_loci: usize,
    /// Sum of |after - before| over scalar deltas.
    pub l1: f64,
    /// Max |after - before| over scalar deltas.
    pub linf: f64,
    pub by_kind: BTreeMap<DeltaKind, usize>,
}

/// The complete structural diff between two genomes of one family and schema.
#[derive(Debug, Clone, PartialEq)]
pub struct GenomeDiff {
    pub family: BrainFamilyId,
    pub schema_version: u32,
    pub deltas: Vec<GenomeDelta>,
    pub summary: DiffSummary,
}

/// Degenerate inputs are typed, never a panic and never a silent empty diff.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GenomeDiffError {
    #[error("cannot diff across brain families: {a} vs {b}")]
    FamilyMismatch { a: BrainFamilyId, b: BrainFamilyId },
    #[error("cannot diff across genome schemas: {a} vs {b}")]
    SchemaMismatch { a: u32, b: u32 },
    #[error("genome shape mismatch: {a_loci} loci vs {b_loci} loci")]
    ShapeMismatch { a_loci: usize, b_loci: usize },
    #[error("family {family} cannot expose typed loci: {reason}")]
    Unsupported {
        family: BrainFamilyId,
        reason: String,
    },
}

/// Compute the typed structural diff between two genomes of the same family and schema,
/// decoded through the family codec's locus view. Deltas are emitted in canonical locus
/// order, so two invocations produce identical bytes on any platform at any thread count.
/// An empty delta list means "no mutations" — and nothing else.
pub fn diff_genomes(
    codec: &dyn BrainFamilyCodec,
    parent: &BrainGenomeEnvelope,
    child: &BrainGenomeEnvelope,
) -> Result<GenomeDiff, GenomeDiffError> {
    if parent.family_id() != child.family_id() {
        return Err(GenomeDiffError::FamilyMismatch {
            a: parent.family_id().clone(),
            b: child.family_id().clone(),
        });
    }
    if parent.schema_version() != child.schema_version() {
        return Err(GenomeDiffError::SchemaMismatch {
            a: parent.schema_version(),
            b: child.schema_version(),
        });
    }
    let family = parent.family_id().clone();
    let schema_version = parent.schema_version();
    let parent_loci = codec
        .genome_loci(parent)
        .map_err(|error| GenomeDiffError::Unsupported {
            family: family.clone(),
            reason: error.to_string(),
        })?;
    let child_loci = codec
        .genome_loci(child)
        .map_err(|error| GenomeDiffError::Unsupported {
            family: family.clone(),
            reason: error.to_string(),
        })?;
    if parent_loci.len() != child_loci.len() {
        return Err(GenomeDiffError::ShapeMismatch {
            a_loci: parent_loci.len(),
            b_loci: child_loci.len(),
        });
    }

    let mut deltas = Vec::new();
    let mut l1 = 0.0_f64;
    let mut linf = 0.0_f64;
    let mut by_kind: BTreeMap<DeltaKind, usize> = BTreeMap::new();
    for ((locus, before), (child_locus, after)) in parent_loci.iter().zip(child_loci.iter()) {
        debug_assert_eq!(
            locus, child_locus,
            "family codec must emit loci in canonical order"
        );
        match (before, after) {
            (LocusValue::Scalar(before), LocusValue::Scalar(after)) => {
                if before.to_bits() != after.to_bits() {
                    let magnitude = f64::from((*after - *before).abs());
                    l1 += magnitude;
                    linf = linf.max(magnitude);
                    *by_kind.entry(DeltaKind::Scalar).or_insert(0) += 1;
                    deltas.push(GenomeDelta::Scalar {
                        locus: *locus,
                        before: *before,
                        after: *after,
                    });
                }
            }
            (LocusValue::Target(before), LocusValue::Target(after)) => {
                if before != after {
                    *by_kind.entry(DeltaKind::Retarget).or_insert(0) += 1;
                    deltas.push(GenomeDelta::Retarget {
                        locus: *locus,
                        before: *before,
                        after: *after,
                    });
                }
            }
            (LocusValue::Kind(before), LocusValue::Kind(after)) => {
                if before != after {
                    *by_kind.entry(DeltaKind::KindFlip).or_insert(0) += 1;
                    deltas.push(GenomeDelta::KindFlip {
                        locus: *locus,
                        before: *before,
                        after: *after,
                    });
                }
            }
            _ => {
                return Err(GenomeDiffError::ShapeMismatch {
                    a_loci: parent_loci.len(),
                    b_loci: child_loci.len(),
                });
            }
        }
    }

    Ok(GenomeDiff {
        family,
        schema_version,
        summary: DiffSummary {
            changed_loci: deltas.len(),
            total_loci: parent_loci.len(),
            l1,
            linf,
            by_kind,
        },
        deltas,
    })
}
