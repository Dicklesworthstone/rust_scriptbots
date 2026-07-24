//! Production replay-event emission for deterministic replay verification (bd-2z0.8.9.8).
//!
//! After the actuation stage finalizes each agent's output vector, the tick pipeline records
//! the decision as a [`ReplayEventKind::Action`] event in canonical handle order — the dense
//! layout is normalized to ascending stable `AgentUid` before every successful tick, so the
//! recorded stream is deterministic. Emission is a read-only projection of outputs the
//! actuation stage already computed: it never mutates science state, so identical
//! configurations produce identical streams and the world digest is unaffected.
//!
//! Emission is bounded by `ScriptBotsConfig::replay_event_tick_cap`; the default of zero
//! keeps production runs byte-identical to their pre-instrumentation baselines. Runs that
//! opt in (for example replay-verification fixtures) should set the cap at or above their
//! peak agent count so every live agent is recorded every tick.
//!
//! The cap bounds each tick's own contribution, not the retained buffer. `replay_events`
//! drains only when a persistence boundary projects a batch, so a whole-buffer cap would
//! let the first tick of a `persistence_interval` window spend the entire budget and leave
//! every later tick in that window recording nothing — a stream that is nonempty, and
//! therefore passes the empty-vs-empty vacuity guard, while missing most of the run. The
//! honest cost of per-tick semantics is that up to `replay_event_tick_cap *
//! persistence_interval` events are retained between drains.

use crate::{
    ReplayEvent, ReplayEventKind, WorldState,
    channels::{OutputChannel, OutputsExt},
};

impl WorldState {
    /// Record per-agent actuation decisions into the tick's replay stream.
    ///
    /// The spike target is not an actuation output — combat resolves spike victims after
    /// this stage — so `spike_target` records `None` rather than fabricating a target.
    pub(crate) fn record_replay_action_events(&mut self) {
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
            self.replay_events.push(ReplayEvent {
                agent_uid: Some(uid),
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
    /// admitted batch as the tick's action events. The request is a no-op while emission is
    /// disabled by `replay_event_tick_cap == 0`.
    pub const fn request_replay_world_digest(&mut self) {
        if self.config.replay_event_tick_cap > 0 {
            self.replay_world_digest_pending = true;
        }
    }

    /// Compute and append the digest event; invoked once per projection by
    /// `prepare_persistence`, where it consumes any pending driver request.
    pub(crate) fn append_requested_replay_world_digest(&mut self) {
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
            Ok(digest) => self.replay_events.push(ReplayEvent {
                agent_uid: None,
                kind: ReplayEventKind::WorldDigest {
                    overall: digest.overall,
                },
            }),
            Err(error) => {
                tracing::warn!(
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
        AgentData, AgentUid, PersistenceAdmissionError, PersistenceBatch, Position, ReplayEvent,
        ReplayEventKind, ScriptBotsConfig, WorldPersistence, WorldState,
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

    /// The zero default must leave the stream completely empty.
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
}
