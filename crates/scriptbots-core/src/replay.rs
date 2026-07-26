//! Production replay-event emission for deterministic replay verification (bd-2z0.8.9.8).
//!
//! After the actuation stage finalizes each agent's output vector, the tick pipeline records
//! the decision as a [`ReplayEventKind::Action`] event in canonical handle order — the dense
//! layout is normalized to ascending stable `AgentUid` before every successful tick, so the
//! recorded stream is deterministic. Emission is a read-only projection of outputs the
//! actuation stage already computed: it never mutates science state, so identical
//! configurations produce identical streams and the world digest is unaffected.
//!
//! Emission is bounded by `ScriptBotsConfig::replay_event_tick_cap`, which defaults to
//! `DEFAULT_REPLAY_EVENT_TICK_CAP` (512). THE STREAM IS ON BY DEFAULT. Runs with larger
//! populations should raise the cap to at or above their peak agent count so every live
//! agent is recorded every tick; the reported drop count is how they learn they need to.
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
    PendingReplayInteraction, ReplayEvent, ReplayEventKind, Tick, WorldState,
    channels::{OutputChannel, OutputsExt},
};

impl WorldState {
    fn begin_replay_tick(&mut self, tick: Tick) {
        if self.replay_tick != tick.0 {
            self.replay_tick = tick.0;
            self.replay_interactions_this_tick = 0;
        }
    }

    /// Number of pairwise interaction slots still available for this simulation tick.
    pub(crate) fn replay_interaction_slots(&mut self, tick: Tick) -> usize {
        self.begin_replay_tick(tick);
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
        let remaining = self
            .config
            .interaction_event_tick_cap
            .saturating_sub(self.replay_interactions_this_tick);
        for event in events.into_iter().take(remaining) {
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
        let dropped = u64::try_from(dropped).unwrap_or(u64::MAX);
        self.replay_interactions_dropped_total = self
            .replay_interactions_dropped_total
            .saturating_add(dropped);
        if dropped > 0 {
            diag_debug!(
                tick = tick.0,
                emitted = self.replay_interactions_this_tick,
                dropped,
                cap = self.config.interaction_event_tick_cap,
                "pairwise interaction replay sample reached its per-tick cap"
            );
        }
    }

    /// Total pairwise facts omitted by the deterministic per-tick interaction cap.
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
        match self.world_digest_v1() {
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
}

#[cfg(test)]
mod tests {
    use crate::{
        AgentData, AgentId, AgentUid, PersistenceAdmissionError, PersistenceBatch, Position,
        ReplayEvent, ReplayEventKind, ReplayInteractionKind, ScriptBotsConfig, WorldPersistence,
        WorldState, channels::OutputChannel,
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
                let offset = 40.0 + (index as f32) * 60.0;
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
    }

    /// An explicit zero cap -- the opt-OUT, no longer the default -- must leave the stream
    /// completely empty. This still matters after the default moved to 512: a run that
    /// deliberately disables emission must actually get silence, or "opt out" would be as
    /// hollow as the inert default it replaced.
    #[test]
    fn a_zero_cap_records_nothing() {
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
        assert!(
            batch.replay_events.is_empty(),
            "disabled emission must not record actions or a digest anchor"
        );
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
        let digests: Vec<_> = batch
            .replay_events
            .iter()
            .filter(|event| matches!(event.kind, ReplayEventKind::WorldDigest { .. }))
            .collect();
        assert_eq!(
            digests.len(),
            1,
            "the requested digest anchor must ride the batch even with a full action budget"
        );
        assert_eq!(
            action_events(batch).len(),
            AGENTS * INTERVAL as usize,
            "the anchor must not displace any action event"
        );
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

        assert!(observations[0].0.is_empty());
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
}
