//! Production replay-event emission for deterministic replay verification (bd-2z0.8.9.8).
//!
//! After the actuation stage finalizes each agent's output vector, the tick pipeline records
//! the decision as a [`ReplayEventKind::Action`] event in canonical handle order — the dense
//! layout is normalized to ascending stable `AgentUid` before every successful tick, so the
//! recorded stream is deterministic. Emission is a read-only projection of outputs the
//! actuation stage already computed: it never mutates science state, so identical
//! configurations produce identical streams and the world digest is unaffected.
//!
//! Optional action emission is bounded by `ScriptBotsConfig::replay_event_tick_cap`, which
//! defaults to `DEFAULT_REPLAY_EVENT_TICK_CAP` (512). THE ACTION STREAM IS ON BY DEFAULT.
//! Runs with larger populations should raise the cap to at or above their peak agent count so
//! every live agent is recorded every tick; the reported drop count is how they learn they need
//! to. The mandatory narrative input record is independent of this optional budget and remains
//! exactly one per persisted tick even when the cap is zero.
//! Setting the cap to zero is an explicit opt-OUT, not the resting state.
//!
//! WHAT "BYTE-IDENTICAL" MEANS NOW, because this sentence used to promise something else.
//! The old zero default was justified as keeping production runs byte-identical to their
//! pre-instrumentation baselines. With the stream on, that is NO LONGER TRUE OF PERSISTED
//! BYTES: runs now record replay events, so stored batches and the config hash both differ
//! from a pre-instrumentation baseline, and that is a deliberate provenance move requiring
//! a re-bless.
//!
//! The property that survives is the one that matters, and it never depended on the cap:
//! SCIENCE IS UNAFFECTED. `replay_events` is excluded from `WorldDigestV1` (bd-zpoa), so
//! emission cannot perturb simulation state at any cap value. That guarantee comes from the
//! digest exclusion, not from recording nothing -- which is precisely why shipping the
//! stream inert was protecting nothing the exclusion does not already protect.
//!
//! The cap bounds each tick's own contribution, not the retained buffer. `replay_events`
//! drains only when a persistence boundary projects a batch, so a whole-buffer cap would
//! let the first tick of a `persistence_interval` window spend the entire budget and leave
//! every later tick in that window recording nothing — a stream that is nonempty, and
//! therefore passes the empty-vs-empty vacuity guard, while missing most of the run. The
//! honest cost of per-tick semantics is that up to `replay_event_tick_cap *
//! persistence_interval` events are retained between drains.

use crate::{
    PendingReplayInteraction, ReplayEvent, ReplayEventKind, ScientificStateError, Tick,
    TickSummary, WorldState,
    channels::{OutputChannel, OutputsExt},
    narrative::{NarrativeInputError, NarrativeInputRecordV1},
};

impl WorldState {
    /// Append the exact production narrative input outside optional replay sampling budgets.
    pub(crate) fn record_narrative_input(&mut self, summary: &TickSummary) {
        if self.config.persistence_interval == 0 {
            return;
        }
        match NarrativeInputRecordV1::from_summary(summary, self.config_revision, &self.config) {
            Ok(record) => self.replay_events.push(ReplayEvent {
                agent_uid: None,
                position: None,
                counterpart: None,
                counterpart_position: None,
                kind: ReplayEventKind::NarrativeInputV1 { record },
            }),
            Err(error) => {
                let fault = match &error {
                    NarrativeInputError::NonFinite { field, .. } => {
                        ScientificStateError::NonFinite {
                            path: format!("narrative_input.{field}"),
                        }
                    }
                    _ => ScientificStateError::DimensionOverflow {
                        path: "narrative_input.persistence_record".to_owned(),
                    },
                };
                diag_error!(
                    tick = summary.tick.0,
                    %error,
                    "complete narrative input could not enter persistence"
                );
                self.latch_scientific_fault(fault);
            }
        }
    }

    fn replay_interaction_tick_selected(&self, tick: Tick) -> bool {
        let stride = self.config.interaction_event_tick_stride;
        self.config.interaction_event_tick_cap > 0
            && stride > 0
            && tick.0.is_multiple_of(u64::from(stride))
    }

    const fn begin_replay_tick(&mut self, tick: Tick) {
        if self.replay_tick != tick.0 {
            self.replay_tick = tick.0;
            self.replay_interactions_this_tick = 0;
        }
    }

    /// Number of pairwise interaction slots still available for this simulation tick.
    pub(crate) fn replay_interaction_slots(&mut self, tick: Tick) -> usize {
        self.begin_replay_tick(tick);
        if !self.replay_interaction_tick_selected(tick) {
            return 0;
        }
        self.config
            .interaction_event_tick_cap
            .saturating_sub(self.replay_interactions_this_tick)
    }

    /// Append an already bounded, canonical sequence of pairwise interaction facts.
    pub(crate) fn record_replay_interaction_events(
        &mut self,
        tick: Tick,
        events: Vec<PendingReplayInteraction>,
        dropped: usize,
    ) {
        self.begin_replay_tick(tick);
        let remaining = self.replay_interaction_slots(tick);
        let observed = events.len().saturating_add(dropped);
        let persisted = events.len().min(remaining);
        let omitted = observed.saturating_sub(persisted);
        self.replay_interactions_observed_pending = self
            .replay_interactions_observed_pending
            .saturating_add(observed);
        self.replay_interactions_persisted_pending = self
            .replay_interactions_persisted_pending
            .saturating_add(persisted);
        if self.replay_interaction_tick_selected(tick) {
            self.replay_interactions_truncated_pending = self
                .replay_interactions_truncated_pending
                .saturating_add(omitted);
        } else {
            self.replay_interactions_sampled_out_pending = self
                .replay_interactions_sampled_out_pending
                .saturating_add(omitted);
        }
        for event in events.into_iter().take(persisted) {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "usize is at most u64 on every supported ScriptBots target"
            )]
            let ordinal = self.replay_interactions_this_tick as u64;
            self.replay_events.push(ReplayEvent {
                agent_uid: Some(event.actor),
                position: Some(event.actor_position),
                counterpart: Some(event.target),
                counterpart_position: Some(event.target_position),
                kind: ReplayEventKind::Interaction {
                    tick,
                    ordinal,
                    kind: event.kind,
                    magnitude: event.magnitude,
                },
            });
            self.replay_interactions_this_tick += 1;
        }
        let dropped = u64::try_from(omitted).unwrap_or(u64::MAX);
        self.replay_interactions_dropped_total = self
            .replay_interactions_dropped_total
            .saturating_add(dropped);
        if dropped > 0 {
            diag_debug!(
                tick = tick.0,
                emitted = self.replay_interactions_this_tick,
                dropped,
                cap = self.config.interaction_event_tick_cap,
                stride = self.config.interaction_event_tick_stride,
                "pairwise interaction replay sample omitted events under its configured policy"
            );
        }
    }

    /// Total pairwise facts omitted by the configured stride or hard cap.
    #[must_use]
    pub const fn replay_interaction_events_dropped(&self) -> u64 {
        self.replay_interactions_dropped_total
    }

    /// Record per-agent actuation decisions into the tick's replay stream.
    ///
    /// The spike target is not an actuation output — combat resolves spike victims after
    /// this stage — so `spike_target` records `None` rather than fabricating a target.
    pub(crate) fn record_replay_action_events(&mut self, tick: Tick) {
        self.begin_replay_tick(tick);
        let cap = self.config.replay_event_tick_cap;
        if cap == 0 {
            return;
        }
        // Budget this tick against the buffer as it stands now, so events already retained
        // for earlier ticks in the same persistence window cannot starve this one.
        let budget_end = self.replay_events.len().saturating_add(cap);
        for id in self.agents.iter_handles() {
            if self.replay_events.len() >= budget_end {
                break;
            }
            let Some(runtime) = self.runtime.get(id) else {
                continue;
            };
            let Some(uid) = self.identities.get(id).map(|identity| identity.uid) else {
                continue;
            };
            let outputs = &runtime.outputs;
            // Position is read HERE, at emission, from the same agent whose outputs we are
            // recording -- not resolved later by a consumer against live state.
            let position = self
                .agents
                .index_of(id)
                .and_then(|index| self.agents.columns().positions().get(index).copied());
            self.replay_events.push(ReplayEvent {
                agent_uid: Some(uid),
                position,
                counterpart: None,
                counterpart_position: None,
                kind: ReplayEventKind::Action {
                    left_wheel: outputs.channel_clamped(OutputChannel::WheelLeft),
                    right_wheel: outputs.channel_clamped(OutputChannel::WheelRight),
                    boost: outputs.boost_engaged(),
                    spike_target: None,
                    sound_level: outputs.channel_clamped(OutputChannel::SoundLevel),
                    give_intent: outputs.channel_clamped(OutputChannel::GiveIntent),
                },
            });
        }
    }

    /// Ask the world to bind the canonical world digest into the replay stream of the next
    /// projected batch (see [`ReplayEventKind::WorldDigest`]). Drivers call this before the
    /// final science tick of a recorded run; the projection consumes the request after that
    /// tick completes, so the digest covers the final post-tick state and rides the same
    /// admitted batch as the tick's action events. The request is a no-op only when a run has
    /// explicitly opted OUT by setting `replay_event_tick_cap` to zero; the default cap is
    /// non-zero, so this is live unless a caller deliberately disabled it.
    pub const fn request_replay_world_digest(&mut self) {
        if self.config.replay_event_tick_cap > 0 {
            self.replay_world_digest_pending = true;
        }
    }

    /// Compute and append the digest event; invoked once per projection by
    /// `prepare_persistence`, where it consumes any pending driver request.
    pub(crate) fn append_requested_replay_world_digest(&mut self, tick: Tick) {
        self.begin_replay_tick(tick);
        if !self.replay_world_digest_pending {
            return;
        }
        self.replay_world_digest_pending = false;
        // Only emission being switched off suppresses the anchor. Refusing it because the
        // action buffer is full would drop it precisely in the runs that record every agent
        // every tick — the ones whose verification depends on having a digest to check the
        // stream against — and one extra event per batch cannot change the footprint.
        if self.config.replay_event_tick_cap == 0 {
            return;
        }
        let previous_tick = self.tick;
        self.tick = tick;
        let digest_result = self.world_digest_v1();
        self.tick = previous_tick;
        match digest_result {
            // The digest anchor is a world-level fact, not an agent's: no participants and
            // no position. Deliberately not given the boundary's centroid or similar --
            // an invented position would be a field a consumer could mistake for real.
            Ok(digest) => self.replay_events.push(ReplayEvent {
                agent_uid: None,
                position: None,
                counterpart: None,
                counterpart_position: None,
                kind: ReplayEventKind::WorldDigest {
                    overall: digest.overall,
                },
            }),
            Err(error) => {
                diag_warn!(
                    %error,
                    "replay world digest request could not be satisfied at this boundary"
                );
            }
        }
    }

    /// Replay scrub starting from current world state to `target_tick` with probe set (bd-16g.4.4).
    pub fn scrub_to_tick(
        &mut self,
        target_tick: Tick,
        target_uid: crate::AgentUid,
        max_tick: Tick,
    ) -> Result<ReplayScrubOutcome, ReplayScrubError> {
        if target_tick < self.tick || target_tick > max_tick {
            return Err(ReplayScrubError::UnreachableTick {
                requested: target_tick,
                base: self.tick,
                max: max_tick,
            });
        }

        let mut death_tick = None;
        let mut born = false;

        if let Some(agent_id) = self.find_agent_by_uid(target_uid) {
            born = true;
            self.set_activation_probe_with_reason(Some(agent_id), "replay scrub");
        }

        while self.tick < target_tick {
            let pre_probe = self.active_activation_probe();
            self.step().map_err(|e| ReplayScrubError::Simulation {
                tick: self.tick,
                error: e.to_string(),
            })?;

            if pre_probe.is_some()
                && self.active_activation_probe().is_none()
                && death_tick.is_none()
            {
                death_tick = self
                    .last_probed_agent_death
                    .map(|(_, t)| t)
                    .or(Some(self.tick));
            }

            if !born && let Some(agent_id) = self.find_agent_by_uid(target_uid) {
                born = true;
                self.set_activation_probe_with_reason(Some(agent_id), "replay scrub");
            }
        }

        if let Some(agent_id) = self.find_agent_by_uid(target_uid) {
            let runtime = self.runtime.get(agent_id);
            let sensors = runtime.map_or([0.0; crate::INPUT_SIZE], |r| r.sensors);
            let outputs = runtime.map_or([0.0; crate::OUTPUT_SIZE], |r| r.outputs);
            let boost_engaged =
                outputs[OutputChannel::Boost.index()] > crate::channels::BOOST_THRESHOLD;

            let activations = self
                .inspect_brains(&crate::BrainInspectionRequest::single(
                    crate::BrainInspectionClientId::new(1),
                    crate::BrainInspectionRevision::new(1),
                    target_uid,
                ))
                .ok()
                .and_then(|resp| {
                    resp.telemetry.into_iter().find_map(|item| match item {
                        crate::SelectedBrainTelemetryOutcome::Ready { telemetry } => {
                            Some(*telemetry)
                        }
                        crate::SelectedBrainTelemetryOutcome::Unavailable { .. } => None,
                    })
                });

            let sensor_attribution = self.explain_sensors(agent_id, 10);
            let brain_bound = runtime.is_some_and(|r| r.brain.is_bound());
            let act_ref = activations.as_ref().map(|a| &a.inspection.activations);
            let output_explanations =
                crate::attribution::explain_outputs(&outputs, brain_bound, act_ref, 4);

            Ok(ReplayScrubOutcome::Inspected(Box::new(ReplayScrubFrame {
                tick: self.tick,
                agent_uid: target_uid,
                agent_id,
                sensors,
                outputs,
                boost_engaged,
                activations,
                sensor_attribution,
                output_explanations,
            })))
        } else if let Some(tick) = death_tick {
            Ok(ReplayScrubOutcome::AgentDied {
                death_tick: Some(tick),
            })
        } else if !born {
            Ok(ReplayScrubOutcome::AgentUnborn)
        } else {
            Ok(ReplayScrubOutcome::AgentDied { death_tick: None })
        }
    }
}

/// Typed error for replay scrub operations (bd-16g.4.4).
#[derive(Debug, thiserror::Error)]
pub enum ReplayScrubError {
    /// Requested tick is outside reachable bounds (e.g. earlier than base checkpoint or past run max).
    #[error("requested tick {requested:?} is unreachable (base {base:?}, max {max:?})")]
    UnreachableTick {
        /// Requested scrub tick.
        requested: Tick,
        /// Starting base tick.
        base: Tick,
        /// Maximum reachable tick.
        max: Tick,
    },
    /// Checkpoint restoration failed.
    #[error("checkpoint error: {0}")]
    Checkpoint(#[from] crate::WorldCheckpointError),
    /// Simulation step error.
    #[error("simulation step error at tick {tick:?}: {error}")]
    Simulation {
        /// Tick where failure occurred.
        tick: Tick,
        /// Stringified error description.
        error: String,
    },
}

/// Re-derived inspection snapshot produced by replay scrub (bd-16g.4.4).
#[derive(Debug, Clone)]
pub struct ReplayScrubFrame {
    /// Simulation tick of this snapshot.
    pub tick: Tick,
    /// Stable agent UID.
    pub agent_uid: crate::AgentUid,
    /// Transient agent handle.
    pub agent_id: crate::AgentId,
    /// Re-derived sensor vector.
    pub sensors: [f32; crate::INPUT_SIZE],
    /// Re-derived output vector.
    pub outputs: [f32; crate::OUTPUT_SIZE],
    /// Whether boost actuator is engaged.
    pub boost_engaged: bool,
    /// Brain activation inspection telemetry.
    pub activations: Option<crate::SelectedBrainTelemetry>,
    /// Sensor attribution summary.
    pub sensor_attribution: Option<crate::SensorAttribution>,
    /// Output explanation summary.
    pub output_explanations: Vec<crate::attribution::OutputExplanation>,
}

/// Scrub outcome at target tick for the requested agent (bd-16g.4.4).
#[derive(Debug, Clone)]
pub enum ReplayScrubOutcome {
    /// Agent is alive at target tick with re-derived inspection telemetry.
    Inspected(Box<ReplayScrubFrame>),
    /// Agent died before or at target tick.
    AgentDied {
        /// Tick of death if known.
        death_tick: Option<Tick>,
    },
    /// Agent was not yet born at target tick.
    AgentUnborn,
}

/// Replay scrub starting from a recorded checkpoint to `target_tick` with probe set (bd-16g.4.4).
pub fn replay_scrub_from_checkpoint(
    checkpoint: &crate::WorldCheckpointV1,
    registry: crate::BrainRegistry,
    target_tick: Tick,
    target_uid: crate::AgentUid,
    max_tick: Tick,
) -> Result<ReplayScrubOutcome, ReplayScrubError> {
    let base_tick = checkpoint.tick();
    if target_tick < base_tick || target_tick > max_tick {
        return Err(ReplayScrubError::UnreachableTick {
            requested: target_tick,
            base: base_tick,
            max: max_tick,
        });
    }

    let mut world = WorldState::restore_checkpoint_v1(checkpoint, registry)?;
    world.scrub_to_tick(target_tick, target_uid, max_tick)
}

#[cfg(test)]
mod tests {
    use crate::{
        AgentData, AgentId, AgentUid, PendingReplayInteraction, PersistenceAdmissionError,
        PersistenceBatch, Position, ReplayEvent, ReplayEventKind, ReplayInteractionKind,
        ScriptBotsConfig, Tick, WorldPersistence, WorldState, channels::OutputChannel,
    };
    use std::sync::{Arc, Mutex};

    /// Retains every admitted batch so a test can inspect the exact projected replay stream.
    struct CollectingSink {
        batches: Arc<Mutex<Vec<PersistenceBatch>>>,
    }

    impl WorldPersistence for CollectingSink {
        fn on_tick(&mut self, payload: &PersistenceBatch) -> Result<(), PersistenceAdmissionError> {
            self.batches
                .lock()
                .expect("sink mutex")
                .push(payload.clone());
            Ok(())
        }
    }

    /// A closed world whose seeded population neither dies, reproduces, nor is restocked, so
    /// per-tick replay counts are exactly the seeded agent count.
    fn quiescent_config(persistence_interval: u32, cap: usize, seed: u64) -> ScriptBotsConfig {
        ScriptBotsConfig {
            world_width: 400,
            world_height: 400,
            closed: true,
            // No drains, so nobody starves out of the population mid-test.
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            // No births and no restocking, so the population cannot grow either.
            reproduction_energy_threshold: 1_000.0,
            reproduction_attempt_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval,
            replay_event_tick_cap: cap,
            rng_seed: Some(seed),
            ..ScriptBotsConfig::default()
        }
    }

    /// Seed `count` well-separated agents so nobody spikes anybody during the test window.
    fn seed_agents(world: &mut WorldState, count: usize) -> Vec<AgentUid> {
        (0..count)
            .map(|index| {
                let index = u16::try_from(index).expect("fixture seed index fits exactly in f32");
                #[expect(
                    clippy::suboptimal_flops,
                    reason = "These seeded positions define interaction separation; preserve the fixture's multiply-then-add coordinate order"
                )]
                let offset = 40.0 + f32::from(index) * 60.0;
                let id = world
                    .try_spawn_agent(AgentData {
                        position: Position::new(offset, offset),
                        health: 2.0,
                        ..AgentData::default()
                    })
                    .expect("seed agent");
                world.agent_uid(id).expect("seeded agent has a stable uid")
            })
            .collect()
    }

    fn world_with_sink(
        config: ScriptBotsConfig,
    ) -> (
        WorldState,
        crate::PersistenceAdmissionSession,
        Arc<Mutex<Vec<PersistenceBatch>>>,
    ) {
        let batches = Arc::new(Mutex::new(Vec::new()));
        let sink = CollectingSink {
            batches: Arc::clone(&batches),
        };
        let (world, session) =
            WorldState::with_persistence(config, Box::new(sink)).expect("world and session");
        (world, session, batches)
    }

    fn action_events(batch: &PersistenceBatch) -> Vec<&ReplayEvent> {
        batch
            .replay_events
            .iter()
            .filter(|event| matches!(event.kind, ReplayEventKind::Action { .. }))
            .collect()
    }

    fn actions_for(batch: &PersistenceBatch, uid: AgentUid) -> usize {
        action_events(batch)
            .into_iter()
            .filter(|event| event.agent_uid == Some(uid))
            .count()
    }

    /// The regression guard for the cross-tick truncation defect: with a cap comfortably above
    /// the population, every tick in the persistence window must contribute its own events.
    /// Budgeting against the retained buffer instead recorded only the window's first tick.
    #[test]
    fn every_tick_in_a_persistence_window_records_its_own_actions() {
        const INTERVAL: u32 = 4;
        const AGENTS: usize = 3;

        let (mut world, mut session, batches) =
            world_with_sink(quiescent_config(INTERVAL, 64, 0x5EED_0001));
        let uids = seed_agents(&mut world, AGENTS);

        for _ in 0..INTERVAL {
            session.step(&mut world).expect("quiescent tick");
        }

        let batches = batches.lock().expect("sink mutex");
        let batch = batches
            .last()
            .expect("the window boundary projected a batch");
        assert_eq!(
            action_events(batch).len(),
            AGENTS * INTERVAL as usize,
            "each of the {INTERVAL} ticks must contribute one action per live agent"
        );
        for uid in uids {
            assert_eq!(
                actions_for(batch, uid),
                INTERVAL as usize,
                "agent {uid:?} survived the whole window and must appear once per tick"
            );
        }
        drop(batches);
    }

    /// The cap is a per-tick budget, so a cap below the population truncates each tick
    /// identically rather than spending the whole run's allowance on the first tick.
    #[test]
    fn a_cap_below_the_population_truncates_each_tick_not_the_run() {
        const INTERVAL: u32 = 3;
        const AGENTS: usize = 4;
        const CAP: usize = 2;

        let (mut world, mut session, batches) =
            world_with_sink(quiescent_config(INTERVAL, CAP, 0x5EED_0002));
        seed_agents(&mut world, AGENTS);

        for _ in 0..INTERVAL {
            session.step(&mut world).expect("quiescent tick");
        }

        let batches = batches.lock().expect("sink mutex");
        let batch = batches
            .last()
            .expect("the window boundary projected a batch");
        assert_eq!(
            action_events(batch).len(),
            CAP * INTERVAL as usize,
            "every tick spends its own budget of {CAP}"
        );
        drop(batches);
    }

    /// An explicit zero action cap must suppress optional action and digest records without
    /// suppressing the mandatory narrative input needed for scientific replay.
    #[test]
    fn a_zero_action_cap_still_records_complete_narrative_inputs() {
        const INTERVAL: u32 = 2;

        let (mut world, mut session, batches) =
            world_with_sink(quiescent_config(INTERVAL, 0, 0x5EED_0003));
        seed_agents(&mut world, 3);
        world.request_replay_world_digest();

        for _ in 0..INTERVAL {
            session.step(&mut world).expect("quiescent tick");
        }

        let batches = batches.lock().expect("sink mutex");
        let batch = batches
            .last()
            .expect("the window boundary projected a batch");
        assert_eq!(action_events(batch), Vec::<&ReplayEvent>::new());
        assert!(
            !batch
                .replay_events
                .iter()
                .any(|event| matches!(event.kind, ReplayEventKind::WorldDigest { .. })),
            "the optional digest anchor follows the action replay opt-out"
        );
        let narrative_inputs = batch
            .replay_events
            .iter()
            .filter_map(|event| match event.kind {
                ReplayEventKind::NarrativeInputV1 { record } => Some(record.input.tick),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            narrative_inputs,
            vec![Tick(1), Tick(2)],
            "mandatory detector inputs must cover every completed tick"
        );
        drop(batches);
    }

    /// Emission is documented as a read-only projection. Two identically seeded runs that
    /// differ only in whether they record must reach the same canonical science state.
    ///
    /// Compared at a drained persistence boundary: `replay_events` is itself part of the
    /// digest, so mid-window the recording run legitimately carries its pending stream.
    /// What must never differ is the settled state, which is what a verifier compares.
    #[test]
    fn recording_does_not_perturb_the_world_digest() {
        const INTERVAL: u32 = 3;
        const TICKS: u32 = 6;

        let mut digests = Vec::new();
        for cap in [0usize, 64] {
            let (mut world, mut session, _batches) =
                world_with_sink(quiescent_config(INTERVAL, cap, 0x5EED_0004));
            seed_agents(&mut world, 3);
            for _ in 0..TICKS {
                session.step(&mut world).expect("quiescent tick");
            }
            digests.push(
                world
                    .world_digest_v1()
                    .expect("completed boundary digest")
                    .overall,
            );
        }

        assert_eq!(
            digests[0], digests[1],
            "replay emission must not change science state"
        );
    }

    /// The anchor the verifier compares against must survive a saturated action buffer —
    /// that is exactly the run which records every agent every tick.
    #[test]
    fn the_world_digest_anchor_survives_a_saturated_action_buffer() {
        const INTERVAL: u32 = 2;
        const AGENTS: usize = 3;

        // Cap equals the population, so each tick fills its budget exactly.
        let (mut world, mut session, batches) =
            world_with_sink(quiescent_config(INTERVAL, AGENTS, 0x5EED_0005));
        seed_agents(&mut world, AGENTS);

        for tick in 0..INTERVAL {
            if tick + 1 == INTERVAL {
                world.request_replay_world_digest();
            }
            session.step(&mut world).expect("quiescent tick");
        }

        let batches = batches.lock().expect("sink mutex");
        let batch = batches
            .last()
            .expect("the window boundary projected a batch");
        let digest_count = batch
            .replay_events
            .iter()
            .filter(|event| matches!(event.kind, ReplayEventKind::WorldDigest { .. }))
            .count();
        assert_eq!(
            digest_count, 1,
            "the requested digest anchor must ride the batch even with a full action budget"
        );
        assert_eq!(
            action_events(batch).len(),
            AGENTS * INTERVAL as usize,
            "the anchor must not displace any action event"
        );
        drop(batches);
    }

    /// Determinism rests on the canonical ascending-`AgentUid` layout; the recorded stream
    /// must expose that order rather than physical slot order.
    #[test]
    fn recorded_actions_follow_ascending_stable_uid_order() {
        let (mut world, mut session, batches) =
            world_with_sink(quiescent_config(1, 64, 0x5EED_0006));
        seed_agents(&mut world, 5);

        session.step(&mut world).expect("quiescent tick");

        let batches = batches.lock().expect("sink mutex");
        let batch = batches.last().expect("the boundary projected a batch");
        let uids: Vec<AgentUid> = action_events(batch)
            .into_iter()
            .filter_map(|event| event.agent_uid)
            .collect();
        drop(batches);
        assert_eq!(uids.len(), 5, "every live agent is recorded");
        assert!(
            uids.windows(2).all(|pair| pair[0] < pair[1]),
            "actions must be recorded in ascending stable uid order, got {uids:?}"
        );
    }

    fn pairwise_world(cap: usize) -> (WorldState, AgentId, AgentId, AgentUid, AgentUid) {
        let config = ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            initial_food: 0.0,
            food_max: 1.0,
            food_intake_rate: 0.0,
            food_waste_rate: 0.0,
            food_transfer_rate: 0.125,
            food_sharing_distance: 20.0,
            spike_radius: 20.0,
            spike_damage: 0.5,
            spike_energy_cost: 0.0,
            spike_min_length: 0.1,
            spike_alignment_cosine: 0.9,
            spike_speed_damage_bonus: 0.0,
            spike_length_damage_bonus: 0.0,
            interaction_event_tick_cap: cap,
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(0x1A7E_2AC7),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("pairwise fixture world");
        let actor = world
            .try_spawn_agent(AgentData::default())
            .expect("interaction actor");
        let target = world
            .try_spawn_agent(AgentData::default())
            .expect("interaction target");
        let actor_index = world.agents.index_of(actor).expect("actor index");
        let target_index = world.agents.index_of(target).expect("target index");
        {
            let columns = world.agents.columns_mut();
            columns.positions_mut()[actor_index] = Position::new(10.0, 10.0);
            columns.positions_mut()[target_index] = Position::new(12.0, 10.0);
            columns.headings_mut()[actor_index] = 0.0;
            columns.spike_lengths_mut()[actor_index] = 1.0;
            columns.health_mut()[target_index] = 2.0;
        }
        {
            let runtime = world.runtime.get_mut(actor).expect("actor runtime");
            runtime.energy = 1.0;
            runtime.give_intent = 1.0;
            runtime.herbivore_tendency = 0.1;
            runtime.outputs[OutputChannel::SpikeTarget.index()] = 1.0;
        }
        {
            let runtime = world.runtime.get_mut(target).expect("target runtime");
            runtime.energy = 1.0;
            runtime.give_intent = 0.0;
            runtime.herbivore_tendency = 0.2;
        }
        let actor_uid = world.agent_uid(actor).expect("actor uid");
        let target_uid = world.agent_uid(target).expect("target uid");
        (world, actor, target, actor_uid, target_uid)
    }

    fn exercise_pairwise_stages(cap: usize) -> (WorldState, AgentUid, AgentUid) {
        let (mut world, _actor, _target, actor_uid, target_uid) = pairwise_world(cap);
        world.stage_food();
        world.stage_combat();
        (world, actor_uid, target_uid)
    }

    #[test]
    fn food_share_and_combat_emit_exact_typed_pairwise_facts() {
        let (world, actor, target) = exercise_pairwise_stages(8);
        let interactions = world
            .replay_events
            .iter()
            .filter(|event| matches!(event.kind, ReplayEventKind::Interaction { .. }))
            .collect::<Vec<_>>();
        assert_eq!(interactions.len(), 2);

        let share = interactions[0];
        assert_eq!(share.agent_uid, Some(actor));
        assert_eq!(share.counterpart, Some(target));
        assert_eq!(share.position, Some(Position::new(10.0, 10.0)));
        assert_eq!(share.counterpart_position, Some(Position::new(12.0, 10.0)));
        assert!(matches!(
            share.kind,
            ReplayEventKind::Interaction {
                tick: Tick(1),
                ordinal: 0,
                kind: ReplayInteractionKind::FoodShare,
                magnitude,
            } if magnitude.to_bits() == 0.125_f32.to_bits()
        ));

        let combat = interactions[1];
        assert_eq!(combat.agent_uid, Some(actor));
        assert_eq!(combat.counterpart, Some(target));
        assert!(matches!(
            combat.kind,
            ReplayEventKind::Interaction {
                tick: Tick(1),
                ordinal: 1,
                kind: ReplayInteractionKind::Combat,
                magnitude,
            } if magnitude.to_bits() == 0.5_f32.to_bits()
        ));
        assert_eq!(
            world.combat_spike_hits, 1,
            "one emitted combat edge must match the in-sim hit counter"
        );
    }

    #[test]
    fn interaction_cap_is_per_tick_bounded_and_science_neutral() {
        let mut observations = Vec::new();
        for cap in [0usize, 1, 2] {
            let (world, _, _) = exercise_pairwise_stages(cap);
            let kinds = world
                .replay_events
                .iter()
                .filter_map(|event| match event.kind {
                    ReplayEventKind::Interaction { kind, .. } => Some(kind),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let digest = world.world_digest_v1().expect("science digest").overall;
            observations.push((kinds, digest, world.replay_interaction_events_dropped()));
        }

        assert_eq!(observations[0].0, Vec::<ReplayInteractionKind>::new());
        assert_eq!(observations[1].0, vec![ReplayInteractionKind::FoodShare]);
        assert_eq!(
            observations[2].0,
            vec![
                ReplayInteractionKind::FoodShare,
                ReplayInteractionKind::Combat
            ]
        );
        assert_eq!(observations[0].1, observations[1].1);
        assert_eq!(observations[1].1, observations[2].1);
        assert_eq!(observations[0].2, 2);
        assert_eq!(observations[1].2, 1);
        assert_eq!(observations[2].2, 0);
        assert_eq!(
            ScriptBotsConfig::default().interaction_event_tick_cap,
            crate::DEFAULT_INTERACTION_EVENT_TICK_CAP
        );
    }

    #[test]
    fn interaction_edges_keep_stable_uids_when_slotmap_storage_recycles() {
        let (mut world, actor_handle, removed_handle, actor_uid, removed_uid) = pairwise_world(8);
        world
            .remove_agent(removed_handle)
            .expect("remove the original target");
        let replacement_handle = world
            .try_spawn_agent(AgentData::default())
            .expect("spawn replacement target");
        let replacement_uid = world
            .agent_uid(replacement_handle)
            .expect("replacement uid");
        assert_ne!(
            removed_handle, replacement_handle,
            "slot generation must advance"
        );
        assert_eq!(
            removed_handle.raw() & u64::from(u32::MAX),
            replacement_handle.raw() & u64::from(u32::MAX),
            "the test premise requires the replacement to reuse the removed slot"
        );
        assert_ne!(
            removed_uid, replacement_uid,
            "stable UIDs must never recycle"
        );

        let actor_index = world.agents.index_of(actor_handle).expect("actor index");
        let replacement_index = world
            .agents
            .index_of(replacement_handle)
            .expect("replacement index");
        {
            let columns = world.agents.columns_mut();
            columns.positions_mut()[actor_index] = Position::new(10.0, 10.0);
            columns.positions_mut()[replacement_index] = Position::new(12.0, 10.0);
            columns.headings_mut()[actor_index] = 0.0;
            columns.spike_lengths_mut()[actor_index] = 1.0;
            columns.health_mut()[replacement_index] = 2.0;
        }
        {
            let actor = world.runtime.get_mut(actor_handle).expect("actor runtime");
            actor.energy = 1.0;
            actor.give_intent = 1.0;
            actor.herbivore_tendency = 0.1;
            actor.outputs[OutputChannel::SpikeTarget.index()] = 1.0;
        }
        {
            let replacement = world
                .runtime
                .get_mut(replacement_handle)
                .expect("replacement runtime");
            replacement.energy = 1.0;
            replacement.give_intent = 0.0;
            replacement.herbivore_tendency = 0.2;
        }

        world.stage_food();
        world.stage_combat();

        let interactions = world
            .replay_events
            .iter()
            .filter(|event| matches!(event.kind, ReplayEventKind::Interaction { .. }))
            .collect::<Vec<_>>();
        assert_eq!(interactions.len(), 2);
        assert!(
            interactions.iter().all(|event| {
                event.agent_uid == Some(actor_uid)
                    && event.counterpart == Some(replacement_uid)
                    && event.counterpart != Some(removed_uid)
            }),
            "interaction rows must name stable live UIDs, never a recycled slot's former owner"
        );
    }

    #[test]
    fn interaction_stride_resets_ordinals_and_persists_completeness_counts() {
        let mut world = WorldState::new(ScriptBotsConfig {
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 1,
            interaction_event_tick_cap: 1,
            interaction_event_tick_stride: 2,
            ..ScriptBotsConfig::default()
        })
        .expect("sampling world");
        assert!(
            !world.has_pending_persistence_material(),
            "the empty sampling world starts without a persistence tail"
        );
        let pending = || PendingReplayInteraction {
            actor: AgentUid(1),
            actor_position: Position::new(1.0, 2.0),
            target: AgentUid(2),
            target_position: Position::new(3.0, 4.0),
            kind: ReplayInteractionKind::Combat,
            magnitude: 0.25,
        };

        for tick in 1..=4 {
            let tick = Tick(tick);
            if world.replay_interaction_slots(tick) == 0 {
                world.record_replay_interaction_events(tick, Vec::new(), 1);
            } else {
                world.record_replay_interaction_events(tick, vec![pending(), pending()], 0);
            }
        }
        assert!(
            world.has_pending_persistence_material(),
            "sampled-out interaction accounting is persistence material even when a tick emits no \
             edge row"
        );

        let projection = world.prepare_persistence(Tick(4), true);
        let batch = projection.batch().expect("forced persistence batch");
        assert!(
            !world.has_pending_persistence_material(),
            "projection drains the exact interaction accounting tail"
        );
        let sampled = batch
            .replay_events
            .iter()
            .filter_map(|event| match event.kind {
                ReplayEventKind::Interaction { tick, ordinal, .. } => Some((tick, ordinal)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            sampled,
            vec![(Tick(2), 0), (Tick(4), 0)],
            "stride two retains complete selected windows and resets the source-tick ordinal"
        );
        let count = |kind: &str| {
            batch
                .events
                .iter()
                .find_map(|event| match &event.kind {
                    crate::PersistenceEventKind::Custom(name) if name == kind => Some(event.count),
                    _ => None,
                })
                .unwrap_or(0)
        };
        let observed = count(crate::INTERACTION_EVENTS_OBSERVED_KIND);
        let persisted = count(crate::INTERACTION_EVENTS_PERSISTED_KIND);
        let sampled_out = count(crate::INTERACTION_EVENTS_SAMPLED_OUT_KIND);
        let truncated = count(crate::INTERACTION_EVENTS_TRUNCATED_KIND);
        assert_eq!(observed, 6);
        assert_eq!(persisted, 2);
        assert_eq!(sampled_out, 2);
        assert_eq!(truncated, 2);
        assert_eq!(
            observed.saturating_sub(sampled_out),
            persisted + truncated,
            "every selected-tick interaction is either persisted or explicitly truncated"
        );
        assert_eq!(observed, persisted + sampled_out + truncated);
        assert_eq!(world.replay_interaction_events_dropped(), 4);
    }

    #[test]
    fn interaction_stride_roundtrips_zero_and_refuses_the_digest_sentinel() {
        for stride in [0, 1, 7] {
            let config = ScriptBotsConfig {
                interaction_event_tick_stride: stride,
                ..ScriptBotsConfig::default()
            };
            let json = serde_json::to_value(&config).expect("serialize config JSON");
            assert_eq!(
                json.get("interaction_event_tick_stride"),
                Some(&serde_json::json!(stride)),
                "every real stride, including aggregates-only zero, belongs in provenance"
            );
            let from_json: ScriptBotsConfig =
                serde_json::from_value(json).expect("deserialize config JSON");
            assert_eq!(from_json.interaction_event_tick_stride, stride);

            let postcard = postcard::to_allocvec(&config).expect("serialize config postcard");
            let from_postcard: ScriptBotsConfig =
                postcard::from_bytes(&postcard).expect("deserialize config postcard");
            assert_eq!(from_postcard.interaction_event_tick_stride, stride);
        }

        let sentinel = ScriptBotsConfig {
            interaction_event_tick_stride: u32::MAX,
            ..ScriptBotsConfig::default()
        };
        assert!(matches!(
            WorldState::new(sentinel),
            Err(crate::WorldStateError::InvalidConfig(
                "interaction_event_tick_stride must be less than u32::MAX"
            ))
        ));
    }

    #[test]
    fn seeded_2k_food_share_stage_has_exact_interaction_accounting() {
        const AGENTS: usize = 2_000;
        const PAIRS: usize = AGENTS / 2;
        let mut world = WorldState::new(ScriptBotsConfig {
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 1,
            food_intake_rate: 0.0,
            food_waste_rate: 0.0,
            food_transfer_rate: 0.01,
            food_sharing_distance: 2.0,
            interaction_event_tick_cap: PAIRS,
            interaction_event_tick_stride: 1,
            rng_seed: Some(0x2000_5EED),
            ..ScriptBotsConfig::default()
        })
        .expect("2k seeded world");

        for pair in 0..PAIRS {
            let grid_x = u16::try_from(pair % 50).expect("2k grid x fits u16");
            let grid_y = u16::try_from(pair / 50).expect("2k grid y fits u16");
            #[expect(
                clippy::suboptimal_flops,
                reason = "The 2k fixture's exact pair spacing is part of its interaction-accounting premise; retain the existing coordinate evaluation order"
            )]
            let (x, y) = (
                50.0 + f32::from(grid_x) * 100.0,
                50.0 + f32::from(grid_y) * 100.0,
            );
            let giver = world
                .try_spawn_agent(AgentData {
                    position: Position::new(x, y),
                    ..AgentData::default()
                })
                .expect("seed giver");
            let recipient = world
                .try_spawn_agent(AgentData {
                    position: Position::new(x + 1.0, y),
                    ..AgentData::default()
                })
                .expect("seed recipient");
            let giver_runtime = world.runtime.get_mut(giver).expect("giver runtime");
            giver_runtime.energy = 1.0;
            giver_runtime.give_intent = 1.0;
            let recipient_runtime = world.runtime.get_mut(recipient).expect("recipient runtime");
            recipient_runtime.energy = 1.0;
            recipient_runtime.give_intent = 0.0;
        }

        world.stage_food();
        assert_eq!(world.agents.len(), AGENTS);
        assert_eq!(world.replay_interactions_observed_pending, PAIRS);
        assert_eq!(world.replay_interactions_persisted_pending, PAIRS);
        assert_eq!(world.replay_interactions_sampled_out_pending, 0);
        assert_eq!(world.replay_interactions_truncated_pending, 0);

        let projection = world.prepare_persistence(Tick(1), true);
        let batch = projection.batch().expect("2k persistence batch");
        let persisted_edges = batch
            .replay_events
            .iter()
            .filter(|event| matches!(event.kind, ReplayEventKind::Interaction { .. }))
            .count();
        assert_eq!(batch.summary.agent_count, AGENTS);
        assert_eq!(
            persisted_edges, PAIRS,
            "the persisted edge stream must exactly match the in-sim interaction counter"
        );
        assert_eq!(
            batch.events.iter().find_map(|event| match &event.kind {
                crate::PersistenceEventKind::Custom(name)
                    if name == crate::INTERACTION_EVENTS_PERSISTED_KIND =>
                {
                    Some(event.count)
                }
                _ => None,
            }),
            Some(PAIRS)
        );
    }

    #[test]
    fn test_replay_scrub_alive_agent_produces_full_frame() {
        let config = ScriptBotsConfig {
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(0x1234),
            spike_damage: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            aging_health_decay_rate: 0.0,
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("world init");
        let target_handle = world.spawn_agent(AgentData {
            position: Position::new(50.0, 50.0),
            ..AgentData::default()
        });
        let target_uid = world.agent_uid(target_handle).expect("uid");

        let outcome = world
            .scrub_to_tick(Tick(5), target_uid, Tick(10))
            .expect("scrub success");
        match outcome {
            super::ReplayScrubOutcome::Inspected(frame) => {
                assert_eq!(frame.tick, Tick(5));
                assert_eq!(frame.agent_uid, target_uid);
                assert_eq!(frame.agent_id, target_handle);
                assert_eq!(world.tick, Tick(5));
                assert_eq!(world.active_activation_probe(), Some(target_handle));
                assert_eq!(frame.sensors.len(), crate::INPUT_SIZE);
                assert_eq!(frame.outputs.len(), crate::OUTPUT_SIZE);
            }
            other => panic!("expected Inspected outcome, got {other:?}"),
        }
    }

    #[test]
    fn test_replay_scrub_agent_death_records_death_tick() {
        let config = ScriptBotsConfig {
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(0x5678),
            spike_damage: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            aging_health_decay_rate: 0.0,
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("world init");
        let target_handle = world.spawn_agent(AgentData {
            position: Position::new(50.0, 50.0),
            ..AgentData::default()
        });
        let target_uid = world.agent_uid(target_handle).expect("uid");

        let _ = world.step();
        let _ = world.step();
        assert_eq!(world.tick, Tick(2));

        let idx = world
            .agents
            .index_of(target_handle)
            .expect("index of agent");
        world.agents.columns_mut().health_mut()[idx] = 0.0;

        let outcome = world
            .scrub_to_tick(Tick(5), target_uid, Tick(10))
            .expect("scrub should return Ok(AgentDied)");
        match outcome {
            super::ReplayScrubOutcome::AgentDied { death_tick } => {
                assert_eq!(death_tick, Some(Tick(3)));
            }
            other => panic!("expected AgentDied outcome, got {other:?}"),
        }
        assert_eq!(world.tick, Tick(5));
        assert_eq!(world.active_activation_probe(), None);
    }

    #[test]
    fn test_replay_scrub_unborn_agent() {
        let config = ScriptBotsConfig {
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(0x9ABC),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("world init");
        let nonexistent_uid = AgentUid(0xDEAD_BEEF);

        let outcome = world
            .scrub_to_tick(Tick(3), nonexistent_uid, Tick(10))
            .expect("scrub outcome");
        assert!(matches!(outcome, super::ReplayScrubOutcome::AgentUnborn));
        assert_eq!(world.tick, Tick(3));
    }

    #[test]
    fn test_replay_scrub_unreachable_ticks() {
        let config = ScriptBotsConfig {
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(0xDEF0),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("world init");
        let target_handle = world.spawn_agent(AgentData::default());
        let target_uid = world.agent_uid(target_handle).expect("uid");

        let _ = world.step();
        let _ = world.step();
        let _ = world.step();
        assert_eq!(world.tick, Tick(3));

        let err_earlier = world.scrub_to_tick(Tick(2), target_uid, Tick(10));
        assert!(matches!(
            err_earlier,
            Err(super::ReplayScrubError::UnreachableTick {
                requested: Tick(2),
                base: Tick(3),
                max: Tick(10),
            })
        ));

        let err_past_max = world.scrub_to_tick(Tick(15), target_uid, Tick(10));
        assert!(matches!(
            err_past_max,
            Err(super::ReplayScrubError::UnreachableTick {
                requested: Tick(15),
                base: Tick(3),
                max: Tick(10),
            })
        ));
    }

    #[test]
    fn test_replay_scrub_from_checkpoint_success_and_bounds() {
        let config = ScriptBotsConfig {
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            rng_seed: Some(0x1337),
            spike_damage: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            aging_health_decay_rate: 0.0,
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("world init");
        let target_handle = world.spawn_agent(AgentData {
            position: Position::new(25.0, 25.0),
            ..AgentData::default()
        });
        let target_uid = world.agent_uid(target_handle).expect("uid");

        let _ = world.step();
        let _ = world.step();
        assert_eq!(world.tick, Tick(2));

        let checkpoint = world.checkpoint_v1().expect("capture checkpoint");

        let outcome = super::replay_scrub_from_checkpoint(
            &checkpoint,
            crate::BrainRegistry::new(),
            Tick(4),
            target_uid,
            Tick(10),
        )
        .expect("checkpoint scrub success");

        match outcome {
            super::ReplayScrubOutcome::Inspected(frame) => {
                assert_eq!(frame.tick, Tick(4));
                assert_eq!(frame.agent_uid, target_uid);
            }
            other => panic!("expected Inspected outcome, got {other:?}"),
        }

        let err_before = super::replay_scrub_from_checkpoint(
            &checkpoint,
            crate::BrainRegistry::new(),
            Tick(1),
            target_uid,
            Tick(10),
        );
        assert!(matches!(
            err_before,
            Err(super::ReplayScrubError::UnreachableTick {
                requested: Tick(1),
                base: Tick(2),
                max: Tick(10),
            })
        ));
    }

    #[test]
    fn test_probe_and_selection_neutrality_over_replay_event_stream() {
        const INTERVAL: u32 = 1;
        const AGENTS: usize = 8;
        const SEED: u64 = 0xCAFE_BABE;

        let (mut world_a, mut session_a, batches_a) =
            world_with_sink(quiescent_config(INTERVAL, 64, SEED));
        let _uids_a = seed_agents(&mut world_a, AGENTS);

        let (mut world_b, mut session_b, batches_b) =
            world_with_sink(quiescent_config(INTERVAL, 64, SEED));
        let uids_b = seed_agents(&mut world_b, AGENTS);

        let probed_handle = world_b.find_agent_by_uid(uids_b[0]).expect("probed handle");
        world_b.set_activation_probe(Some(probed_handle));
        world_b.set_capture_budget(crate::CaptureBudget { max_agents: 2 });
        let selected_handle = world_b
            .find_agent_by_uid(uids_b[2])
            .expect("selected handle");
        let _ = world_b.apply_selection_update(crate::SelectionUpdate {
            mode: crate::SelectionMode::Replace,
            agent_ids: vec![selected_handle.raw()],
            state: crate::SelectionState::Selected,
        });

        for _ in 0..20 {
            session_a.step(&mut world_a).expect("step a");
            session_b.step(&mut world_b).expect("step b");
        }

        let digest_a = world_a.world_digest_v1().expect("digest a");
        let digest_b = world_b.world_digest_v1().expect("digest b");
        assert_eq!(digest_a, digest_b, "world digests must be identical");

        let b_a = batches_a.lock().expect("batches a");
        let b_b = batches_b.lock().expect("batches b");
        assert_eq!(b_a.len(), 20, "20 batches emitted");
        assert_eq!(b_a.len(), b_b.len());
        for (batch_a, batch_b) in b_a.iter().zip(b_b.iter()) {
            assert_eq!(batch_a.replay_events, batch_b.replay_events);
            assert_ne!(batch_a.replay_events.len(), 0);
        }
    }
}
