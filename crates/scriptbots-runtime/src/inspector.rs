//! Bounded, renderer-neutral selected-creature observations for browser and HTTP clients.
//!
//! The owner captures all values together. Identifiers are decimal strings so JavaScript
//! cannot round a UID into another creature, and published and inspected revisions remain
//! distinct when the scientific owner is ahead of the presentation stream.

use scriptbots_core::channels::{OutputChannel, SENSOR_LAYOUT};
use scriptbots_core::genome_diff::{
    DiffSummary, GenomeDelta, GenomeDiffError, Locus, LocusValue, diff_genome_loci,
};
use scriptbots_core::{
    AgentUid, BrainActivations, BrainActivitySource, BrainGenomeDerivation, BrainGenomeEnvelope,
    BrainInspectionBuildStats, BrainInspectionLimits, BrainInspectionUnavailable,
    BrainProtocolError, INPUT_SIZE, MutationRates, OUTPUT_SIZE, SelectedBrainTelemetryOutcome,
    TraitModifiers, WorldState,
};
use serde::Serialize;
use std::collections::BTreeSet;

use crate::{
    BrainProjection, BrainProjectionSource, HostAccessError, HostRevisions,
    channel::AgentInspectorData,
};

/// Per-output contribution rows retained by the browser brain panel.
pub const BRAIN_ACTIVITY_CONTRIBUTORS: usize = 3;

/// Transport ceiling for the bounded brain JSON observation, including its provenance.
pub const MAX_BRAIN_ACTIVITY_JSON_BYTES: usize = 512 * 1024;

/// Maximum loci decoded from either the selected genome or its primary parent.
pub const MAX_GENOME_SOURCE_LOCI: usize = 65_536;
/// Number of complete canonical locus rows in one interactive page.
pub const GENOME_LOCUS_PAGE_SIZE: usize = 64;
/// Number of complete typed parent differences in one independently paged result.
pub const GENOME_DELTA_PAGE_SIZE: usize = 32;
/// Transport ceiling including both pages and exact source/provenance identities.
pub const MAX_GENOME_INSPECTION_JSON_BYTES: usize = 64 * 1024;

/// Parse one canonical, exact stable creature identity without numeric coercion.
#[must_use]
pub fn parse_agent_uid(value: &str) -> Option<AgentUid> {
    let raw = value.parse::<u64>().ok()?;
    (raw.to_string() == value).then_some(AgentUid(raw))
}

/// Exact revisions observed at one owner or publication boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InspectionRevisions {
    /// Scientific-state revision.
    pub scientific: String,
    /// Playback and command-visible revision.
    pub control: String,
    /// Active configuration revision.
    pub config: String,
}

impl From<HostRevisions> for InspectionRevisions {
    fn from(value: HostRevisions) -> Self {
        Self {
            scientific: value.scientific.get().to_string(),
            control: value.control.get().to_string(),
            config: value.config.get().to_string(),
        }
    }
}

/// Source identity retained independently from whichever canvas frame a client is showing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InspectionSource {
    /// Owning host session.
    pub session_id: String,
    /// Evolution epoch inspected by the owner.
    pub epoch: String,
    /// Latest immutable publication visible when the owner answered.
    pub publication_revision: String,
    /// Revisions in that immutable publication.
    pub published: InspectionRevisions,
    /// Revisions in the owner state actually inspected.
    pub inspected: InspectionRevisions,
    /// Completed scientific tick actually inspected.
    pub tick: String,
}

impl InspectionSource {
    fn new(source: BrainProjectionSource, epoch: u64) -> Self {
        Self {
            session_id: source.session_id.get().to_string(),
            epoch: epoch.to_string(),
            publication_revision: source.published_snapshot.get().to_string(),
            published: source.published_host.into(),
            inspected: source.inspected_host.into(),
            tick: source.inspected_tick.0.to_string(),
        }
    }
}

/// An explicitly requested activity observation for one stable creature identity.
///
/// The core bounds the producer work and retained activation graph before this projection.
/// Output explanations consume that same bounded graph and preserve their method/refusal.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrainActivityInspection {
    /// Version of the browser/HTTP activity contract.
    pub schema: &'static str,
    /// Exact owner boundary, including missing and unsupported results.
    pub source: InspectionSource,
    /// Stable creature identity, never a reusable arena handle.
    pub uid: String,
    /// Actual independent producer and payload limits used for this observation.
    pub limits: BrainInspectionLimits,
    /// One typed result; an unavailable brain is never presented as an empty network.
    #[serde(flatten)]
    pub outcome: BrainActivityOutcome,
}

/// Available activity or an explicit scientific reason it cannot be observed.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(
    tag = "status",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum BrainActivityOutcome {
    /// The current bounded evaluator observation and explanations derived from it.
    Ready {
        /// Whether values are retained evaluator state or a read-only feed-forward reconstruction.
        activity_source: BrainActivitySource,
        /// Named layers, row-major values, actual weighted connections and output slots.
        activations: BrainActivations,
        /// Source and retained work, including explicit clipping evidence.
        build: BrainInspectionBuildStats,
        /// All canonical output channels and at most three named input contributions each.
        outputs: Vec<scriptbots_core::attribution::OutputExplanation>,
    },
    /// The subject is absent, unbound or does not support this observation.
    Unavailable {
        /// Typed reason supplied by core, not inferred from an empty payload.
        reason: BrainInspectionUnavailable,
    },
}

impl BrainActivityInspection {
    pub(crate) fn from_projection(
        epoch: u64,
        uid: AgentUid,
        projection: BrainProjection,
    ) -> Result<Self, HostAccessError> {
        let invalid = || HostAccessError::ProtocolViolation {
            message: "brain activity projection did not contain its one requested subject"
                .to_owned(),
        };
        if projection.request.targets != [uid] || projection.inspection.telemetry.len() != 1 {
            return Err(invalid());
        }
        let outcome = match projection
            .inspection
            .telemetry
            .into_iter()
            .next()
            .ok_or_else(invalid)?
        {
            SelectedBrainTelemetryOutcome::Ready { telemetry } if telemetry.agent_uid == uid => {
                let outputs = scriptbots_core::attribution::explain_outputs(
                    &telemetry.outputs,
                    true,
                    Some(&telemetry.inspection.activations),
                    BRAIN_ACTIVITY_CONTRIBUTORS,
                );
                BrainActivityOutcome::Ready {
                    activity_source: telemetry.inspection.activity_source,
                    activations: telemetry.inspection.activations,
                    build: telemetry.inspection.build,
                    outputs,
                }
            }
            SelectedBrainTelemetryOutcome::Unavailable { agent_uid, reason }
                if agent_uid == uid =>
            {
                BrainActivityOutcome::Unavailable { reason }
            }
            _ => return Err(invalid()),
        };
        let detail = Self {
            schema: "scriptbots.brain-activity.v1",
            source: InspectionSource::new(projection.source, epoch),
            uid: uid.to_string(),
            limits: projection.request.limits,
            outcome,
        };
        let encoded =
            serde_json::to_vec(&detail).map_err(|error| HostAccessError::ProtocolViolation {
                message: format!("brain activity could not be encoded: {error}"),
            })?;
        if encoded.len() > MAX_BRAIN_ACTIVITY_JSON_BYTES {
            return Err(HostAccessError::ProtocolViolation {
                message: "brain activity exceeds its JSON transport bound".to_owned(),
            });
        }
        Ok(detail)
    }
}

/// Independent producer, row and transport budgets for explicit genome reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenomeInspectionLimits {
    /// Checked against a family declaration before either genome is decoded.
    pub max_source_loci: usize,
    /// Complete selected-genome rows retained in one page.
    pub locus_page_size: usize,
    /// Complete parent-difference rows retained in one page.
    pub delta_page_size: usize,
    /// Final serialized JSON ceiling.
    pub max_json_bytes: usize,
}

impl GenomeInspectionLimits {
    /// The actual fixed budgets applied by the owner, independent of UI choices.
    #[must_use]
    pub const fn hard() -> Self {
        Self {
            max_source_loci: MAX_GENOME_SOURCE_LOCI,
            locus_page_size: GENOME_LOCUS_PAGE_SIZE,
            delta_page_size: GENOME_DELTA_PAGE_SIZE,
            max_json_bytes: MAX_GENOME_INSPECTION_JSON_BYTES,
        }
    }
}

/// Metadata for an independently navigable page, with no silently omitted rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenomeInspectionPage {
    /// Canonical zero-based row offset requested by the caller.
    pub offset: usize,
    /// Maximum complete rows on this page.
    pub limit: usize,
    /// Exact total number of rows before paging.
    pub total: usize,
    /// At least one later page exists.
    pub has_more: bool,
}

impl GenomeInspectionPage {
    const fn new(offset: usize, limit: usize, total: usize) -> Self {
        Self {
            offset,
            limit,
            total,
            has_more: total.saturating_sub(offset) > limit,
        }
    }
}

/// One complete value at an exact canonical locus address.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GenomeInspectionLocus {
    /// Index in the family's canonical ordered locus vector.
    pub index: usize,
    /// Bounded human label derived from the typed address, never opaque family text.
    pub label: String,
    /// Canonical protocol address; incoming source/weight/kind entries remain distinct.
    pub locus: Locus,
    /// Exact recorded scalar, node index or kind tag, without UI normalization.
    pub value: LocusValue,
}

/// Genome provenance with exact JavaScript-safe identities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenomeInspectionProvenance {
    /// Completed scientific tick when the selected genome was created.
    pub created_at: String,
    /// Material-contributing brain parents, which can differ from whole-agent parents.
    pub parent_uids: [Option<String>; 2],
    /// Recorded material hashes aligned with the two brain-parent slots.
    pub parent_genome_digests: [Option<String>; 2],
    /// Core-owned derivation; a crossover difference is not mislabelled as mutation alone.
    pub derivation: BrainGenomeDerivation,
}

/// Why a selected genome cannot be inspected at this owner boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GenomeInspectionUnavailable {
    /// Stable UID is absent from this world.
    MissingAgent,
    /// The living agent has no bound brain.
    UnboundBrain,
    /// No versioned genome is attached to the bound brain.
    GenomeUnavailable,
    /// The family does not declare bounded canonical locus decoding.
    Unsupported,
    /// The declared source work exceeds the interactive budget.
    SourceBudgetExceeded,
    /// Decoding failed or violated the family's bounds/address/value contract.
    InvalidGenome,
    /// The requested nonzero page starts beyond the current locus list.
    PageOutOfRange,
}

/// Why an exact primary-parent comparison is unavailable; the selected loci stay usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GenomeParentDiffUnavailable {
    /// The primary parent's genome is not retained in the living world.
    ParentUnavailable,
    /// Its observed material differs from the hash recorded at reproduction.
    MaterialMismatch,
    /// Parent and selected genome belong to different families.
    FamilyMismatch,
    /// Parent and selected genome use different schema versions.
    SchemaMismatch,
    /// The parent's codec has no bounded canonical locus capability.
    Unsupported,
    /// The parent's independent source budget was exceeded.
    SourceBudgetExceeded,
    /// The parent genome could not be decoded into valid bounded loci.
    InvalidGenome,
    /// Canonical addresses, shapes or value kinds do not align.
    IncompatibleLoci,
    /// The requested nonzero difference page starts beyond the full difference list.
    PageOutOfRange,
}

/// Exact whole-genome summary and one bounded page of differences against the primary parent.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(
    tag = "status",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum GenomeParentDiff {
    /// A founder has no parent comparison, rather than an invented empty comparison.
    Founder,
    /// Every canonical locus was compared; only the returned difference rows are paged.
    Ready {
        /// Exact primary brain-parent identity.
        parent_uid: String,
        /// Hash verified against the selected genome's retained provenance.
        parent_genome_digest: String,
        /// Two parents contributed material; differences include crossover effects.
        primary_only: bool,
        /// Statistics over all differences, not just this page.
        summary: DiffSummary,
        /// Independent navigation over the complete difference sequence.
        paging: GenomeInspectionPage,
        /// At most `GENOME_DELTA_PAGE_SIZE` complete typed rows in canonical order.
        deltas: Vec<GenomeDelta>,
    },
    /// No partial or guessed comparison is substituted.
    Unavailable {
        /// Stable bounded reason, without unbounded codec error text.
        reason: GenomeParentDiffUnavailable,
    },
}

/// Complete canonical values and independent parent comparison for one genome page.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenomeInspectionReady {
    /// Versioned family identity, bounded by the core envelope contract.
    pub family_id: String,
    /// Family-owned schema version of the inspected genome.
    pub schema_version: u32,
    /// Immutable material identity for binding successive pages.
    pub genome_digest: String,
    /// Selected creature's generation at the inspected boundary.
    pub generation: u32,
    /// Exact genome-level creation and ancestry metadata.
    pub provenance: GenomeInspectionProvenance,
    /// Navigation over the complete canonical locus sequence.
    pub paging: GenomeInspectionPage,
    /// At most `GENOME_LOCUS_PAGE_SIZE` complete locus rows.
    pub loci: Vec<GenomeInspectionLocus>,
    /// Independently bounded exact primary-parent comparison.
    pub parent_diff: GenomeParentDiff,
}

/// A selected genome page or an explicit typed refusal from one owner boundary.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(
    tag = "status",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum GenomeInspectionOutcome {
    /// Complete canonical values for the requested page.
    Ready {
        /// Boxed to keep owner channels small; the JSON retains flat ready fields.
        #[serde(flatten)]
        genome: Box<GenomeInspectionReady>,
    },
    /// A missing or unsupported subject is not an empty genome.
    Unavailable {
        /// Stable bounded scientific or producer-budget reason.
        reason: GenomeInspectionUnavailable,
    },
}

/// Explicit, read-only browser/HTTP genome inspection; it installs no probe or cache.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenomeInspection {
    /// Version of this bounded presentation contract.
    pub schema: &'static str,
    /// Owner source retained for every ready or unavailable result.
    pub source: InspectionSource,
    /// Requested stable subject, never a reusable arena handle.
    pub uid: String,
    /// Applied producer and transport limits.
    pub limits: GenomeInspectionLimits,
    /// Exactly one outcome.
    #[serde(flatten)]
    pub outcome: GenomeInspectionOutcome,
}

/// Refuse malformed or impossible page arithmetic before allocating or enqueueing owner work.
#[expect(
    clippy::suspicious_operation_groupings,
    reason = "loci align to GENOME_LOCUS_PAGE_SIZE and deltas to GENOME_DELTA_PAGE_SIZE"
)]
pub fn validate_genome_page_offsets(
    locus_offset: usize,
    delta_offset: usize,
) -> Result<(), HostAccessError> {
    if !locus_offset.is_multiple_of(GENOME_LOCUS_PAGE_SIZE)
        || !delta_offset.is_multiple_of(GENOME_DELTA_PAGE_SIZE)
        || locus_offset > MAX_GENOME_SOURCE_LOCI
        || delta_offset > MAX_GENOME_SOURCE_LOCI
    {
        return Err(HostAccessError::ProtocolViolation {
            message: "invalid genome locus or difference page offset".to_owned(),
        });
    }
    Ok(())
}

fn decode_bounded_loci(
    declared_bound: Option<usize>,
    decode: impl FnOnce() -> Result<Vec<(Locus, LocusValue)>, BrainProtocolError>,
) -> Result<Vec<(Locus, LocusValue)>, GenomeInspectionUnavailable> {
    let bound = declared_bound.ok_or(GenomeInspectionUnavailable::Unsupported)?;
    if bound > MAX_GENOME_SOURCE_LOCI {
        return Err(GenomeInspectionUnavailable::SourceBudgetExceeded);
    }
    if bound == 0 {
        return Err(GenomeInspectionUnavailable::InvalidGenome);
    }
    let loci = decode().map_err(|error| match error {
        BrainProtocolError::GenomeLociUnavailable { .. } => {
            GenomeInspectionUnavailable::Unsupported
        }
        _ => GenomeInspectionUnavailable::InvalidGenome,
    })?;
    if loci.is_empty() || loci.len() > bound {
        return Err(GenomeInspectionUnavailable::InvalidGenome);
    }
    let mut seen = BTreeSet::new();
    if loci.iter().any(|(locus, value)| {
        !seen.insert(*locus) || matches!(value, LocusValue::Scalar(value) if !value.is_finite())
    }) {
        return Err(GenomeInspectionUnavailable::InvalidGenome);
    }
    Ok(loci)
}

fn inspect_parent_diff(
    genome: &BrainGenomeEnvelope,
    parent: Option<&BrainGenomeEnvelope>,
    child_loci: &[(Locus, LocusValue)],
    delta_offset: usize,
    declared_bound: Option<usize>,
    decode: impl FnOnce(&BrainGenomeEnvelope) -> Result<Vec<(Locus, LocusValue)>, BrainProtocolError>,
) -> GenomeParentDiff {
    let unavailable = |reason| GenomeParentDiff::Unavailable { reason };
    let provenance = genome.provenance();
    let Some(parent_uid) = provenance.parents[0] else {
        return GenomeParentDiff::Founder;
    };
    let Some(parent) = parent else {
        return unavailable(GenomeParentDiffUnavailable::ParentUnavailable);
    };
    if provenance.parent_genome_hashes[0] != Some(parent.material_hash()) {
        return unavailable(GenomeParentDiffUnavailable::MaterialMismatch);
    }
    if parent.family_id() != genome.family_id() {
        return unavailable(GenomeParentDiffUnavailable::FamilyMismatch);
    }
    if parent.schema_version() != genome.schema_version() {
        return unavailable(GenomeParentDiffUnavailable::SchemaMismatch);
    }
    let parent_loci = match decode_bounded_loci(declared_bound, || decode(parent)) {
        Ok(loci) => loci,
        Err(reason) => {
            return unavailable(match reason {
                GenomeInspectionUnavailable::Unsupported => {
                    GenomeParentDiffUnavailable::Unsupported
                }
                GenomeInspectionUnavailable::SourceBudgetExceeded => {
                    GenomeParentDiffUnavailable::SourceBudgetExceeded
                }
                _ => GenomeParentDiffUnavailable::InvalidGenome,
            });
        }
    };
    let diff = match diff_genome_loci(parent, genome, &parent_loci, child_loci) {
        Ok(diff) => diff,
        Err(error) => {
            return unavailable(match error {
                GenomeDiffError::FamilyMismatch { .. } => {
                    GenomeParentDiffUnavailable::FamilyMismatch
                }
                GenomeDiffError::SchemaMismatch { .. } => {
                    GenomeParentDiffUnavailable::SchemaMismatch
                }
                GenomeDiffError::Unsupported { .. } => GenomeParentDiffUnavailable::Unsupported,
                _ => GenomeParentDiffUnavailable::IncompatibleLoci,
            });
        }
    };
    if delta_offset != 0 && delta_offset >= diff.deltas.len() {
        return unavailable(GenomeParentDiffUnavailable::PageOutOfRange);
    }
    let paging = GenomeInspectionPage::new(delta_offset, GENOME_DELTA_PAGE_SIZE, diff.deltas.len());
    GenomeParentDiff::Ready {
        parent_uid: parent_uid.to_string(),
        parent_genome_digest: parent.material_hash().to_string(),
        primary_only: provenance.parents[1].is_some(),
        summary: diff.summary,
        paging,
        deltas: diff
            .deltas
            .into_iter()
            .skip(delta_offset)
            .take(GENOME_DELTA_PAGE_SIZE)
            .collect(),
    }
}

fn inspect_genome_outcome(
    world: &WorldState,
    uid: AgentUid,
    locus_offset: usize,
    delta_offset: usize,
) -> Result<GenomeInspectionOutcome, GenomeInspectionUnavailable> {
    let id = world
        .agents()
        .iter_handles()
        .find(|id| world.agent_uid(*id) == Some(uid))
        .ok_or(GenomeInspectionUnavailable::MissingAgent)?;
    let runtime = world
        .agent_runtime(id)
        .ok_or(GenomeInspectionUnavailable::MissingAgent)?;
    if !runtime.brain.is_bound() {
        return Err(GenomeInspectionUnavailable::UnboundBrain);
    }
    let genome = world
        .agent_brain_genome(id)
        .ok_or(GenomeInspectionUnavailable::GenomeUnavailable)?;
    let codec = world
        .brain_registry()
        .family_by_id(genome.family_id())
        .ok_or(GenomeInspectionUnavailable::Unsupported)?;
    let loci = decode_bounded_loci(codec.genome_loci_upper_bound(), || {
        codec.genome_loci(genome)
    })?;
    if locus_offset != 0 && locus_offset >= loci.len() {
        return Err(GenomeInspectionUnavailable::PageOutOfRange);
    }
    let index = world
        .agents()
        .index_of(id)
        .ok_or(GenomeInspectionUnavailable::MissingAgent)?;
    let provenance = genome.provenance();
    let parent = provenance.parents[0].and_then(|parent_uid| {
        world.agents().iter_handles().find_map(|handle| {
            (world.agent_uid(handle) == Some(parent_uid))
                .then(|| world.agent_brain_genome(handle))
                .flatten()
        })
    });
    let parent_diff = inspect_parent_diff(
        genome,
        parent,
        &loci,
        delta_offset,
        codec.genome_loci_upper_bound(),
        |parent| codec.genome_loci(parent),
    );
    let paging = GenomeInspectionPage::new(locus_offset, GENOME_LOCUS_PAGE_SIZE, loci.len());
    Ok(GenomeInspectionOutcome::Ready {
        genome: Box::new(GenomeInspectionReady {
            family_id: genome.family_id().as_str().to_owned(),
            schema_version: genome.schema_version(),
            genome_digest: genome.material_hash().to_string(),
            generation: world.agents().columns().generations()[index].0,
            provenance: GenomeInspectionProvenance {
                created_at: provenance.created_at.0.to_string(),
                parent_uids: provenance
                    .parents
                    .map(|parent| parent.map(|uid| uid.to_string())),
                parent_genome_digests: provenance
                    .parent_genome_hashes
                    .map(|hash| hash.map(|hash| hash.to_string())),
                derivation: provenance.derivation,
            },
            paging,
            loci: loci
                .into_iter()
                .enumerate()
                .skip(locus_offset)
                .take(GENOME_LOCUS_PAGE_SIZE)
                .map(|(index, (locus, value))| GenomeInspectionLocus {
                    index,
                    label: locus.human(),
                    locus,
                    value,
                })
                .collect(),
            parent_diff,
        }),
    })
}

impl GenomeInspection {
    pub(crate) fn from_world(
        world: &WorldState,
        source: BrainProjectionSource,
        uid: AgentUid,
        locus_offset: usize,
        delta_offset: usize,
    ) -> Result<Self, HostAccessError> {
        validate_genome_page_offsets(locus_offset, delta_offset)?;
        let outcome = inspect_genome_outcome(world, uid, locus_offset, delta_offset)
            .unwrap_or_else(|reason| GenomeInspectionOutcome::Unavailable { reason });
        let detail = Self {
            schema: "scriptbots.genome-inspection.v1",
            source: InspectionSource::new(source, world.epoch()),
            uid: uid.to_string(),
            limits: GenomeInspectionLimits::hard(),
            outcome,
        };
        let encoded =
            serde_json::to_vec(&detail).map_err(|error| HostAccessError::ProtocolViolation {
                message: format!("genome inspection could not be encoded: {error}"),
            })?;
        if encoded.len() > MAX_GENOME_INSPECTION_JSON_BYTES {
            return Err(HostAccessError::ProtocolViolation {
                message: "genome inspection exceeds its JSON transport bound".to_owned(),
            });
        }
        Ok(detail)
    }
}

/// One value labelled by the core's canonical channel table.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InspectionChannel {
    /// Stable channel name from the scientific input/output layout.
    pub name: &'static str,
    /// Most recently recorded value, without frontend normalization.
    pub value: f32,
}

/// Fixed-size scientific detail for one living creature.
///
/// No evaluator, genome payload, unbounded mutation history or world collection is exposed.
/// The brain label is capped at 256 characters; sensor/actuator counts are compile-time fixed.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentInspection {
    /// Version of this presentation-only JSON contract.
    pub schema: &'static str,
    /// Exact owner and publication boundary.
    pub source: InspectionSource,
    /// Stable identity, independent of reusable arena handles.
    pub uid: String,
    /// Current health from the inspected owner boundary.
    pub health: f32,
    /// Completed ticks lived by this creature.
    pub age: u32,
    /// Heritable lineage generation.
    pub generation: u32,
    /// Bounded human-readable brain description.
    pub brain: String,
    /// Whether a brain is attached at this boundary.
    pub brain_bound: bool,
    /// Current energy reserve.
    pub energy: f32,
    /// Accumulated net food/energy balance, including losses and outgoing sharing.
    pub food_balance_total: f32,
    /// Diet on the carnivore (0) to herbivore (1) axis.
    pub herbivore_tendency: f32,
    /// Preferred normalized temperature.
    pub temperature_preference: f32,
    /// Realized emitted sound.
    pub sound_output: f32,
    /// Current sharing intent.
    pub give_intent: f32,
    /// Current progress toward reproduction.
    pub reproduction_counter: f32,
    /// Inherited clock divisors.
    pub clock_scales: [f32; 2],
    /// Whether this creature has two parents.
    pub hybrid: bool,
    /// Up to two exact parent UIDs.
    pub lineage: [Option<String>; 2],
    /// Inherited mutation threshold and magnitude.
    pub mutation_rates: MutationRates,
    /// Inherited sensory and physiology multipliers.
    pub trait_modifiers: TraitModifiers,
    /// Recorded inputs in canonical channel order.
    pub sensors: [InspectionChannel; INPUT_SIZE],
    /// Recorded brain outputs in canonical actuator order.
    pub outputs: [InspectionChannel; OUTPUT_SIZE],
}

impl From<AgentInspectorData> for AgentInspection {
    fn from(value: AgentInspectorData) -> Self {
        Self {
            schema: "scriptbots.agent-inspection.v1",
            source: InspectionSource::new(value.source, value.epoch),
            uid: value.uid.to_string(),
            health: value.health,
            age: value.age,
            generation: value.generation,
            brain: value.brain_descriptor.chars().take(256).collect(),
            brain_bound: value.brain_bound,
            energy: value.physiology.energy,
            food_balance_total: value.life.food_balance_total,
            herbivore_tendency: value.physiology.herbivore_tendency,
            temperature_preference: value.physiology.temperature_preference,
            sound_output: value.physiology.sound_output,
            give_intent: value.physiology.give_intent,
            reproduction_counter: value.life.reproduction_counter,
            clock_scales: value.life.clock_scales,
            hybrid: value.life.hybrid,
            lineage: value.life.lineage.map(|uid| uid.map(|uid| uid.to_string())),
            mutation_rates: value.mutation_rates,
            trait_modifiers: value.trait_modifiers,
            sensors: std::array::from_fn(|index| InspectionChannel {
                name: SENSOR_LAYOUT[index].name,
                value: value.sensors[index],
            }),
            outputs: std::array::from_fn(|index| InspectionChannel {
                name: OutputChannel::ALL[index].name(),
                value: value.outputs[index],
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HostCore, HostCoreOptions, HostSessionId};
    use scriptbots_core::{
        AgentData, BrainFamilyId, BrainProvenance, ScriptBotsConfig, Tick, WorldState,
    };

    #[test]
    fn genome_decoder_refuses_work_before_calling_unsupported_or_oversize_producers() {
        assert_eq!(
            decode_bounded_loci(None, || panic!("undeclared producer must not run")),
            Err(GenomeInspectionUnavailable::Unsupported)
        );
        assert_eq!(
            decode_bounded_loci(Some(MAX_GENOME_SOURCE_LOCI + 1), || panic!(
                "over-budget producer must not run"
            )),
            Err(GenomeInspectionUnavailable::SourceBudgetExceeded)
        );
        assert_eq!(
            decode_bounded_loci(Some(0), || panic!("empty declared producer must not run")),
            Err(GenomeInspectionUnavailable::InvalidGenome)
        );
        let locus = (Locus::NodeBias(0), LocusValue::Scalar(0.0));
        for loci in [
            vec![],
            vec![locus, locus],
            vec![(Locus::Cell(0), LocusValue::Scalar(f32::NAN))],
        ] {
            assert_eq!(
                decode_bounded_loci(Some(2), || Ok(loci)),
                Err(GenomeInspectionUnavailable::InvalidGenome)
            );
        }
        assert_eq!(
            decode_bounded_loci(Some(1), || Ok(vec![
                locus,
                (Locus::Cell(1), LocusValue::Scalar(1.0))
            ])),
            Err(GenomeInspectionUnavailable::InvalidGenome)
        );
        assert_eq!(
            decode_bounded_loci(Some(1), || Ok(vec![locus])).unwrap(),
            vec![locus]
        );
    }

    fn genome_pair() -> (BrainGenomeEnvelope, BrainGenomeEnvelope) {
        let family = BrainFamilyId::new("inspection-test").unwrap();
        let parent =
            BrainGenomeEnvelope::new(family.clone(), 1, 1, vec![0], BrainProvenance::default())
                .unwrap();
        let child = BrainGenomeEnvelope::new(
            family,
            1,
            1,
            vec![1],
            BrainProvenance {
                parents: [Some(AgentUid(u64::MAX)), None],
                parent_genome_hashes: [Some(parent.material_hash()), None],
                created_at: Tick(u64::MAX),
                derivation: BrainGenomeDerivation::MutationOnly,
            },
        )
        .unwrap();
        (parent, child)
    }

    #[test]
    fn genome_parent_difference_pages_keep_whole_summary_and_exact_identity() {
        let (parent, child) = genome_pair();
        let before: Vec<_> = (0..70)
            .map(|node| (Locus::NodeBias(node), LocusValue::Scalar(0.0)))
            .collect();
        let after: Vec<_> = (0..70)
            .map(|node| (Locus::NodeBias(node), LocusValue::Scalar(1.0)))
            .collect();
        for (offset, expected_length) in [(0, 32), (32, 32), (64, 6)] {
            let diff = inspect_parent_diff(
                &child,
                Some(&parent),
                &after,
                offset,
                Some(70),
                |observed| {
                    assert_eq!(observed.material_hash(), parent.material_hash());
                    Ok(before.clone())
                },
            );
            let encoded = serde_json::to_value(&diff).unwrap();
            assert_eq!(encoded["parentUid"], u64::MAX.to_string());
            let GenomeParentDiff::Ready {
                parent_uid,
                primary_only,
                summary,
                paging,
                deltas,
                ..
            } = diff
            else {
                panic!("exact compatible parent must be compared");
            };
            assert_eq!(parent_uid, u64::MAX.to_string());
            assert!(!primary_only);
            assert_eq!(summary.total_loci, 70);
            assert_eq!(summary.changed_loci, 70);
            assert_eq!(summary.l1.to_bits(), 70.0_f64.to_bits());
            assert_eq!(paging, GenomeInspectionPage::new(offset, 32, 70));
            assert_eq!(paging.has_more, offset < 64);
            assert_eq!(deltas.len(), expected_length);
            assert!(
                matches!(deltas[0], GenomeDelta::Scalar { locus: Locus::NodeBias(node), .. } if usize::try_from(node).unwrap() == offset)
            );
        }
        assert!(matches!(
            inspect_parent_diff(&child, Some(&parent), &after, 96, Some(70), |_| Ok(before)),
            GenomeParentDiff::Unavailable {
                reason: GenomeParentDiffUnavailable::PageOutOfRange
            }
        ));
    }

    #[test]
    fn genome_parent_diff_refuses_missing_changed_over_budget_and_misaligned_material() {
        let (parent, child) = genome_pair();
        let loci = [(Locus::NodeBias(0), LocusValue::Scalar(1.0))];
        assert!(matches!(
            inspect_parent_diff(&parent, None, &loci, 0, None, |_| panic!(
                "founder has no parent work"
            )),
            GenomeParentDiff::Founder
        ));
        assert!(matches!(
            inspect_parent_diff(&child, None, &loci, 0, Some(1), |_| panic!(
                "missing parent cannot decode"
            )),
            GenomeParentDiff::Unavailable {
                reason: GenomeParentDiffUnavailable::ParentUnavailable
            }
        ));
        assert!(matches!(
            inspect_parent_diff(&child, Some(&child), &loci, 0, Some(1), |_| panic!(
                "changed material cannot decode"
            )),
            GenomeParentDiff::Unavailable {
                reason: GenomeParentDiffUnavailable::MaterialMismatch
            }
        ));
        assert!(matches!(
            inspect_parent_diff(
                &child,
                Some(&parent),
                &loci,
                0,
                Some(MAX_GENOME_SOURCE_LOCI + 1),
                |_| panic!("parent has an independent source budget")
            ),
            GenomeParentDiff::Unavailable {
                reason: GenomeParentDiffUnavailable::SourceBudgetExceeded
            }
        ));
        assert!(matches!(
            inspect_parent_diff(&child, Some(&parent), &loci, 0, Some(1), |_| Ok(vec![(
                Locus::NodeBias(1),
                LocusValue::Scalar(1.0)
            )])),
            GenomeParentDiff::Unavailable {
                reason: GenomeParentDiffUnavailable::IncompatibleLoci
            }
        ));
    }

    #[test]
    fn genome_offsets_are_bounded_and_align_to_their_independent_page_widths() {
        for (loci, deltas) in [
            (0, 0),
            (64, 32),
            (MAX_GENOME_SOURCE_LOCI, MAX_GENOME_SOURCE_LOCI),
        ] {
            validate_genome_page_offsets(loci, deltas).unwrap();
        }
        for (loci, deltas) in [
            (1, 0),
            (0, 1),
            (32, 64),
            (MAX_GENOME_SOURCE_LOCI + 64, 0),
            (0, usize::MAX),
        ] {
            assert!(validate_genome_page_offsets(loci, deltas).is_err());
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one ready-wire fixture: full locus and delta pages plus large exact provenance"
    )]
    fn genome_ready_wire_keeps_flat_complete_pages_and_exact_large_provenance() {
        let identity = u64::MAX.to_string();
        let revisions = InspectionRevisions {
            scientific: identity.clone(),
            control: identity.clone(),
            config: identity.clone(),
        };
        let locus_at = |index: usize| Locus::NodeWeight {
            node: u32::MAX - u32::try_from(index).unwrap(),
            conn: u8::MAX,
        };
        let mut by_kind = std::collections::BTreeMap::new();
        by_kind.insert(
            scriptbots_core::genome_diff::DeltaKind::Scalar,
            MAX_GENOME_SOURCE_LOCI,
        );
        let detail = GenomeInspection {
            schema: "scriptbots.genome-inspection.v1",
            source: InspectionSource {
                session_id: identity.clone(),
                epoch: identity.clone(),
                publication_revision: identity.clone(),
                published: revisions.clone(),
                inspected: revisions,
                tick: identity.clone(),
            },
            uid: identity.clone(),
            limits: GenomeInspectionLimits::hard(),
            outcome: GenomeInspectionOutcome::Ready {
                genome: Box::new(GenomeInspectionReady {
                    family_id: "a".repeat(scriptbots_core::MAX_BRAIN_FAMILY_ID_BYTES),
                    schema_version: u32::MAX,
                    genome_digest: "f".repeat(64),
                    generation: u32::MAX,
                    provenance: GenomeInspectionProvenance {
                        created_at: identity.clone(),
                        parent_uids: [
                            Some((u64::MAX - 1).to_string()),
                            Some((u64::MAX - 2).to_string()),
                        ],
                        parent_genome_digests: [Some("e".repeat(64)), Some("d".repeat(64))],
                        derivation: BrainGenomeDerivation::CrossoverThenMutation,
                    },
                    paging: GenomeInspectionPage::new(
                        MAX_GENOME_SOURCE_LOCI - GENOME_LOCUS_PAGE_SIZE,
                        GENOME_LOCUS_PAGE_SIZE,
                        MAX_GENOME_SOURCE_LOCI,
                    ),
                    loci: (0..GENOME_LOCUS_PAGE_SIZE)
                        .map(|index| {
                            let locus = locus_at(index);
                            GenomeInspectionLocus {
                                index: MAX_GENOME_SOURCE_LOCI - GENOME_LOCUS_PAGE_SIZE + index,
                                label: locus.human(),
                                locus,
                                value: LocusValue::Scalar(-f32::MAX),
                            }
                        })
                        .collect(),
                    parent_diff: GenomeParentDiff::Ready {
                        parent_uid: (u64::MAX - 1).to_string(),
                        parent_genome_digest: "e".repeat(64),
                        primary_only: true,
                        summary: DiffSummary {
                            changed_loci: MAX_GENOME_SOURCE_LOCI,
                            total_loci: MAX_GENOME_SOURCE_LOCI,
                            l1: f64::from(f32::MAX)
                                * 2.0
                                * f64::from(u32::try_from(MAX_GENOME_SOURCE_LOCI).unwrap()),
                            linf: f64::from(f32::MAX) * 2.0,
                            by_kind,
                        },
                        paging: GenomeInspectionPage::new(
                            MAX_GENOME_SOURCE_LOCI - GENOME_DELTA_PAGE_SIZE,
                            GENOME_DELTA_PAGE_SIZE,
                            MAX_GENOME_SOURCE_LOCI,
                        ),
                        deltas: (0..GENOME_DELTA_PAGE_SIZE)
                            .map(|index| GenomeDelta::Scalar {
                                locus: locus_at(index),
                                before: -f32::MAX,
                                after: f32::MAX,
                            })
                            .collect(),
                    },
                }),
            },
        };
        let encoded = serde_json::to_vec(&detail).unwrap();
        assert!(encoded.len() < MAX_GENOME_INSPECTION_JSON_BYTES);
        let json: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(json["status"], "ready");
        assert!(
            json.get("genome").is_none(),
            "boxed storage must not change the flat wire shape"
        );
        assert_eq!(json["uid"], identity);
        assert_eq!(json["provenance"]["createdAt"], u64::MAX.to_string());
        assert_eq!(
            json["loci"].as_array().unwrap().len(),
            GENOME_LOCUS_PAGE_SIZE
        );
        assert_eq!(
            json["parentDiff"]["deltas"].as_array().unwrap().len(),
            GENOME_DELTA_PAGE_SIZE
        );
        assert_eq!(
            json["parentDiff"]["summary"]["changed_loci"],
            MAX_GENOME_SOURCE_LOCI
        );
        assert!(
            json["parentDiff"]["summary"]["l1"]
                .as_f64()
                .unwrap()
                .is_finite()
        );
    }

    #[test]
    fn exact_uid_parser_refuses_rounding_and_noncanonical_forms() {
        assert_eq!(
            parse_agent_uid("18446744073709551615"),
            Some(AgentUid(u64::MAX))
        );
        assert_eq!(
            parse_agent_uid("9007199254740993"),
            Some(AgentUid(9_007_199_254_740_993))
        );
        for value in [
            "",
            "01",
            "+1",
            "-1",
            "1.0",
            "1e3",
            " 1",
            "18446744073709551616",
        ] {
            assert_eq!(parse_agent_uid(value), None, "{value}");
        }
    }

    #[test]
    fn inspector_projects_real_owner_values_without_advancing_science() {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            rng_seed: Some(91),
            persistence_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        let id = world.try_spawn_agent(AgentData::default()).unwrap();
        let uid = world.agent_uid(id).unwrap();
        let owner = HostCore::new(
            HostSessionId::new(u64::MAX),
            world,
            HostCoreOptions::default(),
        )
        .unwrap();
        let digest = owner.scientific_digest_v1().unwrap();
        let publication = owner.latest_snapshot();
        let mut detail = owner.inspect_agent_basic(uid).unwrap().unwrap();
        assert!(detail.source.matches_snapshot(&publication));
        assert_eq!(detail.epoch, publication.world.epoch);
        assert!(detail.sensor_attribution.is_none());
        assert!(detail.genome_browser.is_none());
        detail.uid = AgentUid(u64::MAX);
        detail.life.lineage = [Some(AgentUid(u64::MAX - 1)), None];
        detail.brain_descriptor = "λ".repeat(300);
        detail.sensors[4] = 0.375;
        detail.outputs[6] = 0.875;
        let view = AgentInspection::from(detail);
        assert_eq!(view.uid, u64::MAX.to_string());
        assert_eq!(view.source.session_id, u64::MAX.to_string());
        assert_eq!(view.lineage[0], Some((u64::MAX - 1).to_string()));
        assert_eq!(view.brain.chars().count(), 256);
        assert_eq!(
            view.sensors[4],
            InspectionChannel {
                name: "food",
                value: 0.375
            }
        );
        assert_eq!(
            view.outputs[6],
            InspectionChannel {
                name: "boost",
                value: 0.875
            }
        );
        assert!(serde_json::to_vec(&view).unwrap().len() < 16 * 1024);
        assert!(
            owner
                .inspect_agent_basic(AgentUid(u64::MAX))
                .unwrap()
                .is_none()
        );
        let missing = owner.inspect_genome_page(AgentUid(u64::MAX), 0, 0).unwrap();
        assert_eq!(missing.uid, u64::MAX.to_string());
        assert_eq!(missing.source.session_id, u64::MAX.to_string());
        assert_eq!(missing.source.epoch, publication.world.epoch.to_string());
        assert!(matches!(
            missing.outcome,
            GenomeInspectionOutcome::Unavailable {
                reason: GenomeInspectionUnavailable::MissingAgent
            }
        ));
        let unbound = owner.inspect_genome_page(uid, 0, 0).unwrap();
        assert!(matches!(
            unbound.outcome,
            GenomeInspectionOutcome::Unavailable {
                reason: GenomeInspectionUnavailable::UnboundBrain
            }
        ));
        assert_eq!(owner.scientific_digest_v1().unwrap(), digest);
        assert_eq!(owner.latest_snapshot().revision, publication.revision);
    }
}
