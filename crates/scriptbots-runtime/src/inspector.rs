//! Bounded, renderer-neutral selected-creature observations for browser and HTTP clients.
//!
//! The owner captures all values together. Identifiers are decimal strings so JavaScript
//! cannot round a UID into another creature, and published and inspected revisions remain
//! distinct when the scientific owner is ahead of the presentation stream.

use scriptbots_core::channels::{OutputChannel, SENSOR_LAYOUT};
use scriptbots_core::{
    AgentUid, BrainActivations, BrainActivitySource, BrainInspectionBuildStats,
    BrainInspectionLimits, BrainInspectionUnavailable, INPUT_SIZE, MutationRates, OUTPUT_SIZE,
    SelectedBrainTelemetryOutcome, TraitModifiers,
};
use serde::Serialize;

use crate::{
    BrainProjection, BrainProjectionSource, HostAccessError, HostRevisions,
    channel::AgentInspectorData,
};

/// Per-output contribution rows retained by the browser brain panel.
pub const BRAIN_ACTIVITY_CONTRIBUTORS: usize = 3;

/// Transport ceiling for the bounded brain JSON observation, including its provenance.
pub const MAX_BRAIN_ACTIVITY_JSON_BYTES: usize = 512 * 1024;

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
    use scriptbots_core::{AgentData, ScriptBotsConfig, WorldState};

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
        assert_eq!(owner.scientific_digest_v1().unwrap(), digest);
        assert_eq!(owner.latest_snapshot().revision, publication.revision);
    }
}
