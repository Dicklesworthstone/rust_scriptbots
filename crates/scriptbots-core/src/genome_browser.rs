//! Bounded, protocol-only genome browser view model (bd-16g.13.3).
//!
//! # Architecture & Boundary Guarantees
//!
//! * **No Brain-Internal Couplings**: Consumes only [`BrainGenomeEnvelope`] and [`BrainFamilyCodec`].
//!   Never downcasts to concrete brain families or implementations.
//! * **Frontend-Neutral**: Represents node topology, connection weights, parent-child mutation deltas,
//!   and lineage locus traces in a pure, serializable view model suitable for GPUI, TUI, WASM, or CLI.
//! * **Strict Bounded Paging & LOD**: Enforces a locus and node budget so huge networks do not blow up UI
//!   memory or frame times. Paging is explicit (`page_offset`, `page_limit`, `total_nodes`, `is_truncated`).
//! * **Read-Only / Digest-Neutral**: Inspecting a genome never mutates world state, RNG streams, or
//!   activation telemetry.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

use crate::genome_diff::{
    DiffSummary, GenomeDelta, GenomeDiffError, Locus, LocusSample, LocusValue, diff_genomes,
    export_locus_trace_csv, export_locus_trace_svg, trace_lineage_locus,
};
use crate::{
    AgentUid, BrainFamilyCodec, BrainFamilyId, BrainGenomeEnvelope, BrainProtocolError,
    BrainProvenance, Tick,
};

/// One outgoing or incoming connection from a node in the genome topology.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrowserConnectionView {
    /// Slot index of connection within node.
    pub conn_slot: u8,
    /// Target node index.
    pub target_node: u32,
    /// Connection weight.
    pub weight: f32,
    /// Optional function kind tag (e.g. for DWRAON).
    pub kind: Option<String>,
}

/// A single node decoded properties and connections.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrowserNodeView {
    /// Node index in the network.
    pub node_index: u32,
    /// Bias scalar, if present.
    pub bias: Option<f32>,
    /// Damping factor, if present.
    pub damping: Option<f32>,
    /// Gain scalar, if present.
    pub gain: Option<f32>,
    /// Connections associated with this node.
    pub connections: Vec<BrowserConnectionView>,
}

/// Status of the parent-child mutation diff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MutationDiffStatus {
    /// Agent is a founding ancestor or injected seed with no parent.
    FounderNoParent,
    /// Parent envelope was provided and diff computed successfully.
    Computed {
        /// Parent UID.
        parent_uid: AgentUid,
        /// Total count of detected mutations.
        total_deltas: usize,
        /// Diff statistical summary.
        summary: DiffSummary,
    },
    /// Multiple parents (sexual reproduction), diff against primary parent.
    SexualPrimary {
        /// All parent UIDs.
        parent_uids: Vec<AgentUid>,
        /// Total count of detected mutations against primary parent.
        total_deltas: usize,
        /// Diff statistical summary against primary parent.
        summary: DiffSummary,
    },
    /// Diff could not be computed due to a typed error (e.g. mismatched family or schema).
    Unavailable {
        /// Explanation of why the diff could not be computed.
        reason: String,
    },
}

/// Lineage locus trace sub-view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrowserLineagePlotView {
    /// The traced locus.
    pub locus: Locus,
    /// Ordered samples along the lineage.
    pub samples: Vec<LocusSample>,
    /// Standalone SVG chart representation.
    pub svg_chart: String,
    /// Standalone CSV representation.
    pub csv_data: String,
    /// Total data points.
    pub total_points: usize,
    /// Number of explicit gaps (missing or incompatible schemas).
    pub gap_count: usize,
}

/// Paging metadata for bounded UI rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrowserPagingMeta {
    /// Starting node offset.
    pub page_offset: usize,
    /// Maximum nodes included per page.
    pub page_limit: usize,
    /// Total nodes in the full genome.
    pub total_nodes: usize,
    /// Total loci in the full genome.
    pub total_loci: usize,
    /// Whether nodes were truncated to fit the page limit.
    pub is_truncated: bool,
}

/// Errors that can occur when building a `GenomeBrowserViewModel`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GenomeBrowserError {
    /// Protocol error decoding genome.
    #[error("protocol error decoding genome: {0}")]
    Protocol(#[from] BrainProtocolError),
    /// Error diffing genomes.
    #[error("diff error: {0}")]
    Diff(#[from] GenomeDiffError),
    /// Missing required genome envelope.
    #[error("missing required genome envelope")]
    MissingGenome,
}

/// Frontend-neutral bounded view model for the genome browser.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeBrowserViewModel {
    /// Selected agent UID.
    pub selected_agent: AgentUid,
    /// Simulation generation of the selected agent.
    pub generation: u32,
    /// Simulation tick of observation.
    pub tick: Tick,
    /// Brain family ID.
    pub family_id: BrainFamilyId,
    /// Schema version of the genome protocol envelope.
    pub schema_version: u32,
    /// BLAKE3 material hash of the genome payload in hex.
    pub genome_digest: String,
    /// Brain provenance metadata.
    pub provenance: BrainProvenance,
    /// Parent agent UIDs.
    pub parent_uids: Vec<AgentUid>,

    /// Decoded node topology for the current page/LOD slice.
    pub nodes: Vec<BrowserNodeView>,
    /// Paging metadata.
    pub paging: BrowserPagingMeta,

    /// Parent-to-newborn mutation diff and deltas.
    pub mutation_diff: MutationDiffStatus,
    /// Complete list of typed deltas against the parent.
    pub deltas: Vec<GenomeDelta>,

    /// Selected-locus lineage plot (if a locus was selected and history supplied).
    pub locus_plot: Option<BrowserLineagePlotView>,

    /// Diagnostic build duration in microseconds.
    pub build_duration_us: u64,
}

#[derive(Default)]
struct NodeAccumulator {
    bias: Option<f32>,
    damping: Option<f32>,
    gain: Option<f32>,
    connections: BTreeMap<u8, ConnAccumulator>,
}

#[derive(Default)]
struct ConnAccumulator {
    target_node: u32,
    weight: f32,
    kind: Option<String>,
}

fn accumulate_nodes(loci: &[(Locus, LocusValue)]) -> BTreeMap<u32, NodeAccumulator> {
    let mut nodes: BTreeMap<u32, NodeAccumulator> = BTreeMap::new();
    for (locus, val) in loci {
        match (*locus, *val) {
            (Locus::NodeBias(node), LocusValue::Scalar(s)) => {
                nodes.entry(node).or_default().bias = Some(s);
            }
            (Locus::NodeDamping(node), LocusValue::Scalar(s)) => {
                nodes.entry(node).or_default().damping = Some(s);
            }
            (Locus::NodeGain(node), LocusValue::Scalar(s)) => {
                nodes.entry(node).or_default().gain = Some(s);
            }
            (Locus::NodeWeight { node, conn }, LocusValue::Scalar(s)) => {
                nodes
                    .entry(node)
                    .or_default()
                    .connections
                    .entry(conn)
                    .or_default()
                    .weight = s;
            }
            (Locus::NodeTarget { node, conn }, LocusValue::Target(t)) => {
                nodes
                    .entry(node)
                    .or_default()
                    .connections
                    .entry(conn)
                    .or_default()
                    .target_node = t;
            }
            (Locus::NodeKind { node, conn }, LocusValue::Kind(k)) => {
                nodes
                    .entry(node)
                    .or_default()
                    .connections
                    .entry(conn)
                    .or_default()
                    .kind = Some(format!("kind_{k}"));
            }
            (Locus::Cell(cell_idx), LocusValue::Scalar(s)) => {
                nodes.entry(cell_idx).or_default().bias = Some(s);
            }
            (Locus::Hyper(id), LocusValue::Scalar(s)) => {
                nodes.entry(100_000 + u32::from(id)).or_default().bias = Some(s);
            }
            _ => {}
        }
    }
    nodes
}

fn mutation_diff(
    codec: &dyn BrainFamilyCodec,
    envelope: &BrainGenomeEnvelope,
    parent_envelope: Option<&BrainGenomeEnvelope>,
    parent_uids: &[AgentUid],
) -> (MutationDiffStatus, Vec<GenomeDelta>) {
    parent_envelope.map_or_else(
        || (MutationDiffStatus::FounderNoParent, Vec::new()),
        |parent| match diff_genomes(codec, parent, envelope) {
            Ok(diff) => {
                let status = if parent_uids.len() > 1 {
                    MutationDiffStatus::SexualPrimary {
                        parent_uids: parent_uids.to_vec(),
                        total_deltas: diff.deltas.len(),
                        summary: diff.summary,
                    }
                } else {
                    let parent_uid = parent_uids.first().copied().unwrap_or(AgentUid(0));
                    MutationDiffStatus::Computed {
                        parent_uid,
                        total_deltas: diff.deltas.len(),
                        summary: diff.summary,
                    }
                };
                (status, diff.deltas)
            }
            Err(e) => (
                MutationDiffStatus::Unavailable {
                    reason: e.to_string(),
                },
                Vec::new(),
            ),
        },
    )
}

impl GenomeBrowserViewModel {
    /// Build a frontend-neutral bounded genome browser view model from versioned protocol envelopes.
    ///
    /// Consumes only [`BrainFamilyCodec`] and [`BrainGenomeEnvelope`]; never downcasts to concrete brain types.
    #[expect(
        clippy::too_many_arguments,
        reason = "each input supplies distinct view evidence: codec/genome, agent/generation/tick identity, parent envelope/identities, locus/history, and retained paging offset/limit"
    )]
    pub fn build(
        codec: &dyn BrainFamilyCodec,
        agent_uid: AgentUid,
        generation: u32,
        tick: Tick,
        envelope: &BrainGenomeEnvelope,
        parent_envelope: Option<&BrainGenomeEnvelope>,
        parent_uids: Vec<AgentUid>,
        selected_locus: Option<Locus>,
        lineage_history: Option<&[(u32, AgentUid, Tick, BrainGenomeEnvelope)]>,
        page_offset: usize,
        page_limit: usize,
    ) -> Result<Self, GenomeBrowserError> {
        let start_time = Instant::now();

        // Decode loci from protocol envelope
        let loci = codec.genome_loci(envelope)?;
        let total_loci = loci.len();

        // Group into node topology
        let node_accum = accumulate_nodes(&loci);

        let total_nodes = node_accum.len();
        let effective_limit = if page_limit == 0 {
            usize::MAX
        } else {
            page_limit
        };

        let paged_nodes: Vec<BrowserNodeView> = node_accum
            .into_iter()
            .skip(page_offset)
            .take(effective_limit)
            .map(|(node_index, acc)| {
                let connections = acc
                    .connections
                    .into_iter()
                    .map(|(conn_slot, c)| BrowserConnectionView {
                        conn_slot,
                        target_node: c.target_node,
                        weight: c.weight,
                        kind: c.kind,
                    })
                    .collect();
                BrowserNodeView {
                    node_index,
                    bias: acc.bias,
                    damping: acc.damping,
                    gain: acc.gain,
                    connections,
                }
            })
            .collect();

        let is_truncated = page_offset + paged_nodes.len() < total_nodes;
        let paging = BrowserPagingMeta {
            page_offset,
            page_limit: effective_limit,
            total_nodes,
            total_loci,
            is_truncated,
        };

        // Compute mutation diff
        let (mutation_diff, deltas) = mutation_diff(codec, envelope, parent_envelope, &parent_uids);

        // Lineage locus plot
        let locus_plot = if let (Some(locus), Some(history)) = (selected_locus, lineage_history) {
            let samples = trace_lineage_locus(codec, history, locus);
            let gap_count = samples.iter().filter(|s| s.value.is_none()).count();
            let total_points = samples.len();
            let svg_chart = export_locus_trace_svg(&samples, locus);
            let csv_data = export_locus_trace_csv(&samples, locus);
            Some(BrowserLineagePlotView {
                locus,
                samples,
                svg_chart,
                csv_data,
                total_points,
                gap_count,
            })
        } else {
            None
        };

        let build_duration_us = u64::try_from(start_time.elapsed().as_micros()).unwrap_or(u64::MAX);

        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        tracing::debug!(
            target: "scriptbots::genome_browser",
            selected_agent = agent_uid.get(),
            generation,
            family = envelope.family_id().as_str(),
            schema = envelope.schema_version(),
            total_nodes = paging.total_nodes,
            total_loci = paging.total_loci,
            visible_nodes = paged_nodes.len(),
            delta_count = deltas.len(),
            has_locus_plot = locus_plot.is_some(),
            build_duration_us,
            "built GenomeBrowserViewModel"
        );

        Ok(Self {
            selected_agent: agent_uid,
            generation,
            tick,
            family_id: envelope.family_id().clone(),
            schema_version: envelope.schema_version(),
            genome_digest: envelope.material_hash().to_string(),
            provenance: envelope.provenance().clone(),
            parent_uids,
            nodes: paged_nodes,
            paging,
            mutation_diff,
            deltas,
            locus_plot,
            build_duration_us,
        })
    }

    /// Attach externally-traced locus samples to the view model.
    #[must_use]
    pub fn with_locus_samples(mut self, locus: Locus, samples: Vec<LocusSample>) -> Self {
        let gap_count = samples.iter().filter(|s| s.value.is_none()).count();
        let total_points = samples.len();
        let svg_chart = export_locus_trace_svg(&samples, locus);
        let csv_data = export_locus_trace_csv(&samples, locus);
        self.locus_plot = Some(BrowserLineagePlotView {
            locus,
            samples,
            svg_chart,
            csv_data,
            total_points,
            gap_count,
        });
        self
    }

    /// Lookup a specific node by its index within the visible page.
    #[must_use]
    pub fn node(&self, index: u32) -> Option<&BrowserNodeView> {
        self.nodes.iter().find(|n| n.node_index == index)
    }

    /// Total number of outgoing connections across all visible nodes.
    #[must_use]
    pub fn visible_connection_count(&self) -> usize {
        self.nodes.iter().map(|n| n.connections.len()).sum()
    }

    /// Lookup a delta by its locus, if one occurred between parent and child.
    #[must_use]
    pub fn delta_for_locus(&self, locus: &Locus) -> Option<&GenomeDelta> {
        self.deltas.iter().find(|d| match d {
            GenomeDelta::Scalar { locus: l, .. }
            | GenomeDelta::Retarget { locus: l, .. }
            | GenomeDelta::KindFlip { locus: l, .. } => l == locus,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome_diff::DeltaKind;
    use crate::{
        BrainAdapterIdentityV1, BrainEvaluator, BrainEvaluatorStateEnvelope, BrainGenomeDerivation,
        BrainGenomeMaterial, BrainHeredityCapabilityV1, BrainHeredityExclusionV1, MutationRates,
        OffspringStatePolicy, RandomStream,
    };

    struct MockCodec {
        family: BrainFamilyId,
        loci: Vec<(Locus, LocusValue)>,
    }

    impl BrainFamilyCodec for MockCodec {
        fn family_id(&self) -> &BrainFamilyId {
            &self.family
        }
        fn adapter_identity(&self) -> BrainAdapterIdentityV1 {
            BrainAdapterIdentityV1::from_semantic_descriptor(&self.family, 1, b"mock")
        }
        fn heredity_capability(&self) -> BrainHeredityCapabilityV1 {
            BrainHeredityCapabilityV1::excluded(BrainHeredityExclusionV1::NoCanonicalLocusSchema)
        }
        fn random_genome_material(
            &self,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
            BrainGenomeMaterial::new(1, 1, vec![0])
        }
        fn validate_genome(&self, _genome: &BrainGenomeEnvelope) -> Result<(), BrainProtocolError> {
            Ok(())
        }
        fn genome_loci(
            &self,
            _genome: &BrainGenomeEnvelope,
        ) -> Result<Vec<(Locus, LocusValue)>, BrainProtocolError> {
            Ok(self.loci.clone())
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
            BrainGenomeMaterial::new(1, 1, vec![0])
        }
        fn crossover_genomes_material(
            &self,
            _left: &BrainGenomeEnvelope,
            _right: &BrainGenomeEnvelope,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
            BrainGenomeMaterial::new(1, 1, vec![0])
        }
        fn initial_state(
            &self,
            _genome: &BrainGenomeEnvelope,
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
            BrainEvaluatorStateEnvelope::new(self.family.clone(), 1, 1, vec![0])
        }
        fn offspring_state_policy(&self) -> OffspringStatePolicy {
            OffspringStatePolicy::Reset
        }
        fn offspring_state(
            &self,
            _child: &BrainGenomeEnvelope,
            _parents: &[&BrainEvaluatorStateEnvelope],
            _rng: &mut dyn RandomStream,
        ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
            BrainEvaluatorStateEnvelope::new(self.family.clone(), 1, 1, vec![0])
        }
        fn evaluator(
            &self,
            _genome: &BrainGenomeEnvelope,
            _state: &BrainEvaluatorStateEnvelope,
        ) -> Result<Box<dyn BrainEvaluator>, BrainProtocolError> {
            unimplemented!()
        }
    }

    fn founder_provenance() -> BrainProvenance {
        BrainProvenance {
            parents: [None, None],
            parent_genome_hashes: [None, None],
            created_at: Tick(0),
            derivation: BrainGenomeDerivation::Founder,
        }
    }

    #[test]
    fn test_table_driven_view_model_goldens_delta_kinds() {
        let family = BrainFamilyId::new("mock_family").expect("family id");
        let parent_loci = vec![
            (Locus::NodeBias(0), LocusValue::Scalar(1.0)),
            (Locus::NodeDamping(0), LocusValue::Scalar(0.5)),
            (Locus::NodeGain(0), LocusValue::Scalar(2.0)),
            (
                Locus::NodeWeight { node: 0, conn: 0 },
                LocusValue::Scalar(0.25),
            ),
            (
                Locus::NodeTarget { node: 0, conn: 0 },
                LocusValue::Target(4),
            ),
            (Locus::NodeKind { node: 0, conn: 0 }, LocusValue::Kind(1)),
        ];
        let child_loci = vec![
            (Locus::NodeBias(0), LocusValue::Scalar(1.5)), // Scalar delta
            (Locus::NodeDamping(0), LocusValue::Scalar(0.5)),
            (Locus::NodeGain(0), LocusValue::Scalar(2.0)),
            (
                Locus::NodeWeight { node: 0, conn: 0 },
                LocusValue::Scalar(0.25),
            ),
            (
                Locus::NodeTarget { node: 0, conn: 0 },
                LocusValue::Target(9),
            ), // Retarget delta
            (Locus::NodeKind { node: 0, conn: 0 }, LocusValue::Kind(2)), // KindFlip delta
        ];

        let parent_env =
            BrainGenomeEnvelope::new(family.clone(), 1, 1, vec![1, 2, 3], founder_provenance())
                .expect("envelope");
        let child_env =
            BrainGenomeEnvelope::new(family.clone(), 1, 1, vec![1, 2, 4], founder_provenance())
                .expect("envelope");

        struct DynamicMockCodec {
            family: BrainFamilyId,
            parent_loci: Vec<(Locus, LocusValue)>,
            child_loci: Vec<(Locus, LocusValue)>,
        }

        impl BrainFamilyCodec for DynamicMockCodec {
            fn family_id(&self) -> &BrainFamilyId {
                &self.family
            }
            fn adapter_identity(&self) -> BrainAdapterIdentityV1 {
                BrainAdapterIdentityV1::from_semantic_descriptor(&self.family, 1, b"mock")
            }
            fn heredity_capability(&self) -> BrainHeredityCapabilityV1 {
                BrainHeredityCapabilityV1::excluded(
                    BrainHeredityExclusionV1::NoCanonicalLocusSchema,
                )
            }
            fn random_genome_material(
                &self,
                _rng: &mut dyn RandomStream,
            ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
                BrainGenomeMaterial::new(1, 1, vec![0])
            }
            fn validate_genome(
                &self,
                _genome: &BrainGenomeEnvelope,
            ) -> Result<(), BrainProtocolError> {
                Ok(())
            }
            fn genome_loci(
                &self,
                env: &BrainGenomeEnvelope,
            ) -> Result<Vec<(Locus, LocusValue)>, BrainProtocolError> {
                if env.payload() == &[1, 2, 3] {
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
                BrainGenomeMaterial::new(1, 1, vec![0])
            }
            fn crossover_genomes_material(
                &self,
                _left: &BrainGenomeEnvelope,
                _right: &BrainGenomeEnvelope,
                _rng: &mut dyn RandomStream,
            ) -> Result<BrainGenomeMaterial, BrainProtocolError> {
                BrainGenomeMaterial::new(1, 1, vec![0])
            }
            fn initial_state(
                &self,
                _genome: &BrainGenomeEnvelope,
                _rng: &mut dyn RandomStream,
            ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
                BrainEvaluatorStateEnvelope::new(self.family.clone(), 1, 1, vec![0])
            }
            fn offspring_state_policy(&self) -> OffspringStatePolicy {
                OffspringStatePolicy::Reset
            }
            fn offspring_state(
                &self,
                _child: &BrainGenomeEnvelope,
                _parents: &[&BrainEvaluatorStateEnvelope],
                _rng: &mut dyn RandomStream,
            ) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
                BrainEvaluatorStateEnvelope::new(self.family.clone(), 1, 1, vec![0])
            }
            fn evaluator(
                &self,
                _genome: &BrainGenomeEnvelope,
                _state: &BrainEvaluatorStateEnvelope,
            ) -> Result<Box<dyn BrainEvaluator>, BrainProtocolError> {
                unimplemented!()
            }
        }

        let codec = DynamicMockCodec {
            family,
            parent_loci,
            child_loci,
        };

        let vm = GenomeBrowserViewModel::build(
            &codec,
            AgentUid(42),
            2,
            Tick(200),
            &child_env,
            Some(&parent_env),
            vec![AgentUid(10)],
            Some(Locus::NodeBias(0)),
            None,
            0,
            10,
        )
        .expect("build view model");

        assert_eq!(vm.selected_agent, AgentUid(42));
        assert_eq!(vm.generation, 2);
        assert_eq!(vm.tick, Tick(200));
        assert_eq!(vm.nodes.len(), 1);
        let node0 = &vm.nodes[0];
        assert_eq!(node0.node_index, 0);
        assert_eq!(node0.bias, Some(1.5));
        assert_eq!(node0.damping, Some(0.5));
        assert_eq!(node0.gain, Some(2.0));
        assert_eq!(node0.connections.len(), 1);
        assert_eq!(node0.connections[0].target_node, 9);
        assert_eq!(node0.connections[0].weight, 0.25);
        assert_eq!(node0.connections[0].kind, Some("kind_2".to_string()));

        // Check deltas
        assert_eq!(vm.deltas.len(), 3);
        assert!(matches!(
            vm.delta_for_locus(&Locus::NodeBias(0)),
            Some(GenomeDelta::Scalar { before, after, .. }) if *before == 1.0 && *after == 1.5
        ));
        assert!(matches!(
            vm.delta_for_locus(&Locus::NodeTarget { node: 0, conn: 0 }),
            Some(GenomeDelta::Retarget {
                before: 4,
                after: 9,
                ..
            })
        ));
        assert!(matches!(
            vm.delta_for_locus(&Locus::NodeKind { node: 0, conn: 0 }),
            Some(GenomeDelta::KindFlip {
                before: 1,
                after: 2,
                ..
            })
        ));

        // Check diff summary
        match &vm.mutation_diff {
            MutationDiffStatus::Computed {
                parent_uid,
                total_deltas,
                summary,
            } => {
                assert_eq!(*parent_uid, AgentUid(10));
                assert_eq!(*total_deltas, 3);
                assert_eq!(summary.changed_loci, 3);
                assert_eq!(summary.by_kind.get(&DeltaKind::Scalar), Some(&1));
                assert_eq!(summary.by_kind.get(&DeltaKind::Retarget), Some(&1));
                assert_eq!(summary.by_kind.get(&DeltaKind::KindFlip), Some(&1));
            }
            other => panic!("unexpected diff status: {other:?}"),
        }
    }

    #[test]
    fn test_browser_empty_and_founder_no_parent() {
        let family = BrainFamilyId::new("empty_fam").expect("family id");
        let codec = MockCodec {
            family: family.clone(),
            loci: vec![],
        };
        let env =
            BrainGenomeEnvelope::new(family, 1, 1, vec![], founder_provenance()).expect("envelope");

        let vm = GenomeBrowserViewModel::build(
            &codec,
            AgentUid(1),
            0,
            Tick(0),
            &env,
            None,
            vec![],
            None,
            None,
            0,
            10,
        )
        .expect("build empty view model");

        assert_eq!(vm.nodes.len(), 0);
        assert_eq!(vm.paging.total_nodes, 0);
        assert_eq!(vm.paging.total_loci, 0);
        assert!(!vm.paging.is_truncated);
        assert_eq!(vm.mutation_diff, MutationDiffStatus::FounderNoParent);
        assert!(vm.deltas.is_empty());
        assert!(vm.locus_plot.is_none());
    }

    #[test]
    fn test_browser_long_genomes_paging_and_lod() {
        let family = BrainFamilyId::new("large_net").expect("family id");
        let mut loci = Vec::new();
        for node in 0..50 {
            loci.push((Locus::NodeBias(node), LocusValue::Scalar(node as f32 * 0.1)));
            loci.push((Locus::NodeWeight { node, conn: 0 }, LocusValue::Scalar(1.0)));
            loci.push((
                Locus::NodeTarget { node, conn: 0 },
                LocusValue::Target((node + 1) % 50),
            ));
        }

        let codec = MockCodec {
            family: family.clone(),
            loci,
        };
        let env = BrainGenomeEnvelope::new(family, 1, 1, vec![9; 100], founder_provenance())
            .expect("envelope");

        // Request page 2: offset 20, limit 10
        let vm = GenomeBrowserViewModel::build(
            &codec,
            AgentUid(5),
            1,
            Tick(50),
            &env,
            None,
            vec![],
            None,
            None,
            20,
            10,
        )
        .expect("build paged view model");

        assert_eq!(vm.paging.total_nodes, 50);
        assert_eq!(vm.paging.total_loci, 150);
        assert_eq!(vm.paging.page_offset, 20);
        assert_eq!(vm.paging.page_limit, 10);
        assert!(vm.paging.is_truncated);

        assert_eq!(vm.nodes.len(), 10);
        assert_eq!(vm.nodes[0].node_index, 20);
        assert_eq!(vm.nodes[9].node_index, 29);
    }

    #[test]
    fn test_browser_zero_rate_empty_diff() {
        let family = BrainFamilyId::new("cloned_fam").expect("family id");
        let loci = vec![
            (Locus::NodeBias(0), LocusValue::Scalar(0.7)),
            (
                Locus::NodeWeight { node: 0, conn: 0 },
                LocusValue::Scalar(1.2),
            ),
        ];

        let codec = MockCodec {
            family: family.clone(),
            loci,
        };
        let env = BrainGenomeEnvelope::new(family, 1, 1, vec![1, 2], founder_provenance())
            .expect("envelope");

        let vm = GenomeBrowserViewModel::build(
            &codec,
            AgentUid(2),
            1,
            Tick(10),
            &env,
            Some(&env), // Same parent envelope
            vec![AgentUid(1)],
            None,
            None,
            0,
            10,
        )
        .expect("build zero diff view model");

        assert!(vm.deltas.is_empty());
        match &vm.mutation_diff {
            MutationDiffStatus::Computed {
                total_deltas,
                summary,
                ..
            } => {
                assert_eq!(*total_deltas, 0);
                assert_eq!(summary.changed_loci, 0);
                assert_eq!(summary.l1, 0.0);
            }
            other => panic!("expected computed diff, got {other:?}"),
        }
    }

    #[test]
    fn test_browser_mechanical_dependency_boundary() {
        // Mechanically verify that genome_browser.rs production code does not contain concrete brain downcasts
        let file_contents = include_str!("genome_browser.rs");
        let prod_code = file_contents.split("#[cfg(test)]").next().unwrap();
        assert!(
            !prod_code.contains(".downcast"),
            "genome_browser production code must not perform downcasts"
        );
        assert!(
            !prod_code.contains("downcast_ref"),
            "genome_browser production code must not downcast"
        );
        assert!(
            !prod_code.contains("MlpBrain"),
            "genome_browser must not reference MlpBrain"
        );
        assert!(
            !prod_code.contains("DwraonBrain"),
            "genome_browser must not reference DwraonBrain"
        );
        assert!(
            !prod_code.contains("AssemblyBrain"),
            "genome_browser must not reference AssemblyBrain"
        );
    }
}
