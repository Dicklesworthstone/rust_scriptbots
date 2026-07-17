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

    /// Record the canonical final world digest as one world-scoped replay event.
    ///
    /// Replay verification compares this recorded digest against the digest the verifying
    /// driver records after re-simulation, proving the entire final science state — not
    /// just the event stream — reproduced exactly. Drivers call this once at a clean
    /// boundary, after the last science tick and before the final partial batch is
    /// projected; both the recording and the verifying driver must emit it so the two
    /// streams stay structurally aligned for ordered diffing.
    pub fn record_replay_world_digest(&mut self, overall: String) {
        let cap = self.config.replay_event_tick_cap;
        if cap == 0 || self.replay_events.len() >= cap {
            return;
        }
        self.replay_events.push(ReplayEvent {
            agent_uid: None,
            kind: ReplayEventKind::WorldDigest { overall },
        });
    }
}
