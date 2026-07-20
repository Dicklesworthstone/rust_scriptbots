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
///
/// Two genomes of the same family and schema produce loci in the same order, so the diff
/// is a single aligned pass and its bytes are reproducible on any platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Locus {
    /// Bias of the node with the given index.
    NodeBias(u32),
    /// Damping factor of the node with the given index.
    NodeDamping(u32),
    /// Gain of the node with the given index.
    NodeGain(u32),
    /// Weight of connection `conn` feeding node `node`.
    NodeWeight {
        /// Index of the node that owns the connection.
        node: u32,
        /// Incoming connection slot within the node.
        conn: u8,
    },
    /// Function kind of connection `conn` feeding node `node`.
    NodeKind {
        /// Index of the node that owns the connection.
        node: u32,
        /// Incoming connection slot within the node.
        conn: u8,
    },
    /// Target node of connection `conn` leaving node `node`.
    NodeTarget {
        /// Index of the node that owns the connection.
        node: u32,
        /// Outgoing connection slot within the node.
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
    /// A continuous scalar parameter; compared bitwise via `f32::to_bits`.
    Scalar(f32),
    /// A connection target: the index of the node a connection points at.
    Target(u32),
    /// A connection function-kind tag.
    Kind(u8),
}

/// One typed change between two aligned genomes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GenomeDelta {
    /// A scalar parameter whose bits changed.
    Scalar {
        /// The locus that changed.
        locus: Locus,
        /// Parent value.
        before: f32,
        /// Child value.
        after: f32,
    },
    /// A connection whose target node changed.
    Retarget {
        /// The locus that changed.
        locus: Locus,
        /// Parent target node index.
        before: u32,
        /// Child target node index.
        after: u32,
    },
    /// A connection whose function kind changed.
    KindFlip {
        /// The locus that changed.
        locus: Locus,
        /// Parent kind tag.
        before: u8,
        /// Child kind tag.
        after: u8,
    },
}

/// The kind of a delta, for the by-kind summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeltaKind {
    /// Scalar value change.
    Scalar,
    /// Connection retarget.
    Retarget,
    /// Function-kind change.
    KindFlip,
}

/// Aggregate diff statistics, computed in one pass over the canonical delta list.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffSummary {
    /// Number of loci that differ (equal to the delta count).
    pub changed_loci: usize,
    /// Total loci compared (genome length).
    pub total_loci: usize,
    /// Sum of |after - before| over scalar deltas.
    pub l1: f64,
    /// Max |after - before| over scalar deltas.
    pub linf: f64,
    /// Delta count per kind.
    pub by_kind: BTreeMap<DeltaKind, usize>,
}

/// The complete structural diff between two genomes of one family and schema.
#[derive(Debug, Clone, PartialEq)]
pub struct GenomeDiff {
    /// Brain family both genomes belong to.
    pub family: BrainFamilyId,
    /// Genome schema version both envelopes share.
    pub schema_version: u32,
    /// Typed deltas in canonical locus order.
    pub deltas: Vec<GenomeDelta>,
    /// Aggregate statistics over `deltas`.
    pub summary: DiffSummary,
}

/// Degenerate inputs are typed, never a panic and never a silent empty diff.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GenomeDiffError {
    /// The two genomes belong to different brain families.
    #[error("cannot diff across brain families: {a} vs {b}")]
    FamilyMismatch {
        /// Family of the parent genome.
        a: BrainFamilyId,
        /// Family of the child genome.
        b: BrainFamilyId,
    },
    /// The two genomes use different schema versions.
    #[error("cannot diff across genome schemas: {a} vs {b}")]
    SchemaMismatch {
        /// Schema version of the parent genome.
        a: u32,
        /// Schema version of the child genome.
        b: u32,
    },
    /// The decoded locus lists differ in length or type alignment.
    #[error("genome shape mismatch: {a_loci} loci vs {b_loci} loci")]
    ShapeMismatch {
        /// Locus count of the parent genome.
        a_loci: usize,
        /// Locus count of the child genome.
        b_loci: usize,
    },
    /// The family codec could not expose typed loci.
    #[error("family {family} cannot expose typed loci: {reason}")]
    Unsupported {
        /// Family that failed to decode.
        family: BrainFamilyId,
        /// Codec-reported failure detail.
        reason: String,
    },
}

/// Compute the typed structural diff between two genomes of the same family and schema.
///
/// Decoded through the family codec's locus view. Deltas are emitted in canonical locus
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
