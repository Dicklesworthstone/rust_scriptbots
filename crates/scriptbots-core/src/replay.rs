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
        if cap == 0 || self.replay_events.len() >= cap {
            return;
        }
        for id in self.agents.iter_handles() {
            if self.replay_events.len() >= cap {
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
    pub fn request_replay_world_digest(&mut self) {
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
        let cap = self.config.replay_event_tick_cap;
        if cap == 0 || self.replay_events.len() >= cap {
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
