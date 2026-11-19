//! Presentation-only sonification of immutable owner publications.
//!
//! A latest-value subscriber can miss completed ticks. This adapter reports those
//! gaps and never reconstructs missing events from differences between per-tick
//! counters. It owns no simulation, database, audio device, clock, or worker.

use crate::{HostHealth, HostLifecycle, HostSessionId, RenderSnapshot};
use scriptbots_core::audio::{AudioFrame, SpatialPosition};
use scriptbots_core::visual::WorldVisualEvent;

/// One observation for a frontend's bounded soundscape consumer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotAudioUpdate {
    /// New observed input, or `None` when silent or already consumed.
    pub frame: Option<AudioFrame>,
    /// Discard old queued input and DSP tails before accepting this observation.
    pub reset: bool,
    /// The frontend must silence its device for this observation.
    pub muted: bool,
    /// Completed ticks skipped since the previous audible observation.
    ///
    /// These ticks have no reconstructed cues. Time spent explicitly muted is
    /// excluded because resuming establishes a new observation boundary.
    pub skipped_ticks: u64,
}

/// Deduplicates owner cuts and projects genuine per-tick activity for audio.
#[derive(Debug, Default)]
pub struct SnapshotAudioProjector {
    source: Option<(HostSessionId, u64)>,
    audible: bool,
    skipped_ticks: u64,
}

impl SnapshotAudioProjector {
    /// Total skipped completed ticks while this consumer was audible.
    #[must_use]
    pub const fn skipped_ticks(&self) -> u64 {
        self.skipped_ticks
    }

    /// Observe a publication without modifying its owner or scientific state.
    ///
    /// `presentation_audible` combines the user's opt-in with the frontend's
    /// live-view and foreground policy. Paused, stopping, stopped and unhealthy
    /// owners are always silent. Re-enabling or changing owners starts with
    /// current ambience only; already completed births, deaths and combat are
    /// not replayed. Normal epoch boundaries do not reset a continuing owner.
    #[must_use]
    pub fn observe(
        &mut self,
        snapshot: &RenderSnapshot,
        presentation_audible: bool,
    ) -> SnapshotAudioUpdate {
        let tick = snapshot.world.tick;
        let changed_source = self.source.is_none_or(|(session, previous_tick)| {
            session != snapshot.session_id || tick < previous_tick
        });
        let coherent = snapshot
            .completed_summary
            .as_ref()
            .map_or(tick == 0, |summary| summary.tick.0 == tick);
        let audible = presentation_audible
            && coherent
            && !snapshot.playback.paused
            && snapshot.playback.speed_multiplier.is_finite()
            && snapshot.playback.speed_multiplier > 0.0
            && snapshot.lifecycle == HostLifecycle::Running
            && matches!(snapshot.health, HostHealth::Healthy);
        let rebase = changed_source || !self.audible;
        let reset = changed_source || audible != self.audible;
        let advanced = self
            .source
            .is_none_or(|(_, previous_tick)| tick > previous_tick);
        let skipped_ticks = if audible && !rebase && advanced {
            self.source
                .map_or(0, |(_, previous_tick)| tick - previous_tick - 1)
        } else {
            0
        };
        self.skipped_ticks = self.skipped_ticks.saturating_add(skipped_ticks);
        self.source = Some((snapshot.session_id, tick));
        self.audible = audible;

        SnapshotAudioUpdate {
            frame: (audible && (rebase || advanced)).then(|| frame_from_snapshot(snapshot, !rebase)),
            reset,
            muted: !audible,
            skipped_ticks,
        }
    }
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "presentation diet share rounds bounded snapshot counts to f32"
)]
fn frame_from_snapshot(snapshot: &RenderSnapshot, include_events: bool) -> AudioFrame {
    let agents = &snapshot.world.agents;
    let population = u32::try_from(agents.len()).unwrap_or(u32::MAX);
    let herbivore_share = if agents.is_empty() {
        0.0
    } else {
        let herbivores = agents
            .iter()
            .filter(|agent| agent.herbivore_tendency >= snapshot.config.carnivore_threshold)
            .count();
        (herbivores as f64 / agents.len() as f64) as f32
    };
    let mut frame = AudioFrame {
        tick: snapshot.world.tick,
        population,
        herbivore_share,
        mean_energy: snapshot.world.summary.average_energy,
        mean_health: snapshot.world.summary.average_health,
        // The current snapshot has no measured crowding statistic. This field
        // is unused by the shared mapper; do not invent a spatial observation.
        crowding: 0.0,
        ..AudioFrame::default()
    };
    if include_events {
        if let Some(summary) = &snapshot.completed_summary {
            frame.births = u32::try_from(summary.births).unwrap_or(u32::MAX);
            frame.deaths = u32::try_from(summary.deaths).unwrap_or(u32::MAX);
            frame.spike_hits = summary.spike_hits;
        }
        for event in snapshot.visual_events.iter() {
            if event.tick.0 == frame.tick
                && matches!(event.event, WorldVisualEvent::CombatHit { .. })
                && event.position.x.is_finite()
                && event.position.y.is_finite()
                && !frame.add_spike_position(SpatialPosition::new(
                    event.position.x,
                    event.position.y,
                ))
            {
                break;
            }
        }
    }
    frame
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CommandEnvelope, CommandId, HostCommand, HostCore, HostCoreOptions, HostPort,
        ManualHostDriver, ManualInstant, PlaybackSnapshot, SnapshotRevision,
    };
    use scriptbots_core::{AgentData, ScriptBotsConfig, WorldState};

    fn owner(session: u64) -> HostCore {
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(42),
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            reproduction_energy_threshold: f32::MAX,
            carnivore_threshold: 0.75,
            ..ScriptBotsConfig::default()
        })
        .expect("valid compact world");
        for _ in 0..3 {
            world
                .try_spawn_agent_with(AgentData::default(), |runtime| {
                    runtime.herbivore_tendency = 0.6;
                })
                .expect("real founder");
        }
        HostCore::new(
            HostSessionId::new(session),
            world,
            HostCoreOptions {
                tick_period_nanos: 1,
                snapshot_interval_ticks: 1,
                initial_playback: PlaybackSnapshot {
                    paused: false,
                    speed_multiplier: 1.0,
                },
                ..HostCoreOptions::default()
            },
        )
        .expect("real owner")
    }

    fn advance(owner: &mut HostCore, now: u64) {
        for _ in 0..4 {
            owner
                .drive(ManualInstant::from_nanos(now))
                .expect("real owner boundary");
        }
    }

    #[test]
    fn real_owner_projection_is_observational_and_control_publications_do_not_replay() {
        let mut owner = owner(1);
        let mut projector = SnapshotAudioProjector::default();
        advance(&mut owner, 0);
        let initial = owner.latest_snapshot();
        assert_eq!(initial.world.agents.len(), 3);
        let initial_audio = projector.observe(&initial, true);
        assert!(initial_audio.reset);
        assert_eq!(
            initial_audio.frame.expect("initial ambience").herbivore_share.to_bits(),
            0.0_f32.to_bits(),
            "diet follows the observed configuration threshold"
        );
        advance(&mut owner, 1);
        let snapshot = owner.latest_snapshot();
        assert_eq!(snapshot.world.tick, 1);
        let digest = owner.world().world_digest_v1().expect("scientific digest");
        let output = projector.observe(&snapshot, true);
        let frame = output.frame.expect("new completed tick");
        let summary = snapshot.completed_summary.as_ref().expect("actual summary");
        assert_eq!(frame.tick, summary.tick.0);
        assert_eq!(frame.births, u32::try_from(summary.births).expect("small count"));
        assert_eq!(frame.deaths, u32::try_from(summary.deaths).expect("small count"));
        assert_eq!(frame.spike_hits, summary.spike_hits);
        assert_eq!(frame.mean_energy.to_bits(), snapshot.world.summary.average_energy.to_bits());
        assert_eq!(frame.mean_health.to_bits(), snapshot.world.summary.average_health.to_bits());
        assert!(projector.observe(&snapshot, true).frame.is_none());

        let mut port = owner.local_port();
        port.submit(CommandEnvelope::new(CommandId::new(1), HostCommand::SetSpeed(2.0)))
            .expect("real control admission");
        advance(&mut owner, 1);
        let changed = owner.latest_snapshot();
        assert_eq!(changed.world.tick, snapshot.world.tick);
        assert!(changed.revision > snapshot.revision);
        assert!(projector.observe(&changed, true).frame.is_none());
        assert_eq!(owner.world().world_digest_v1().expect("scientific digest"), digest);
    }

    #[test]
    fn repeated_equal_tick_counts_are_new_events_and_skipped_ticks_are_not_synthesized() {
        let mut owner = owner(2);
        let mut projector = SnapshotAudioProjector::default();
        advance(&mut owner, 0);
        let _ = projector.observe(&owner.latest_snapshot(), true);
        advance(&mut owner, 1);
        let mut snapshot = (*owner.latest_snapshot()).clone();
        let summary = snapshot.completed_summary.as_mut().expect("real summary");
        summary.births = 5;
        summary.deaths = 2;
        summary.spike_hits = 3;
        let first = projector.observe(&snapshot, true).frame.expect("first events");
        snapshot.world.tick = 2;
        snapshot.completed_summary.as_mut().expect("summary").tick.0 = 2;
        let second = projector.observe(&snapshot, true).frame.expect("second events");
        assert_eq!((first.births, first.deaths, first.spike_hits), (5, 2, 3));
        assert_eq!((second.births, second.deaths, second.spike_hits), (5, 2, 3));
        snapshot.world.tick = 9;
        snapshot.completed_summary.as_mut().expect("summary").tick.0 = 9;
        let gap = projector.observe(&snapshot, true);
        assert_eq!(gap.skipped_ticks, 6);
        assert_eq!(projector.skipped_ticks(), 6);
        assert_eq!(gap.frame.expect("latest observed events").births, 5);
    }

    #[test]
    fn pause_background_replay_and_owner_replacement_clear_history() {
        let mut owner = owner(3);
        advance(&mut owner, 0);
        advance(&mut owner, 1);
        let mut snapshot = (*owner.latest_snapshot()).clone();
        snapshot.completed_summary.as_mut().expect("summary").births = 7;
        let mut projector = SnapshotAudioProjector::default();
        let initial = projector.observe(&snapshot, true);
        assert_eq!(initial.frame.expect("ambient rebase").births, 0);
        for paused in [true, false] {
            snapshot.playback.paused = paused;
            let silent = projector.observe(&snapshot, false);
            assert!(silent.muted);
            assert!(silent.frame.is_none());
            snapshot.playback.paused = false;
            let resumed = projector.observe(&snapshot, true);
            assert!(resumed.reset);
            assert_eq!(resumed.frame.expect("no historical birth").births, 0);
        }
        snapshot.playback.paused = true;
        let paused = projector.observe(&snapshot, true);
        assert!(paused.reset && paused.muted);
        assert!(paused.frame.is_none());
        snapshot.playback.paused = false;
        snapshot.session_id = HostSessionId::new(4);
        let replaced = projector.observe(&snapshot, true);
        assert!(replaced.reset);
        assert_eq!(replaced.frame.expect("new owner ambience").births, 0);
        assert_eq!(projector.skipped_ticks(), 0);
    }

    #[test]
    fn epoch_changes_do_not_repeat_cues_and_incoherent_summaries_silence() {
        let mut owner = owner(4);
        advance(&mut owner, 0);
        advance(&mut owner, 1);
        let mut snapshot = (*owner.latest_snapshot()).clone();
        let mut projector = SnapshotAudioProjector::default();
        let _ = projector.observe(&snapshot, true);
        snapshot.world.epoch += 1;
        snapshot.revision = SnapshotRevision::new(snapshot.revision.get() + 1);
        let epoch = projector.observe(&snapshot, true);
        assert!(!epoch.reset && !epoch.muted);
        assert!(epoch.frame.is_none());
        snapshot.completed_summary.as_mut().expect("summary").tick.0 += 1;
        let incoherent = projector.observe(&snapshot, true);
        assert!(incoherent.reset && incoherent.muted);
        assert!(incoherent.frame.is_none());
        snapshot.completed_summary.as_mut().expect("summary").tick.0 -= 1;
        snapshot.lifecycle = HostLifecycle::Stopped;
        assert!(projector.observe(&snapshot, true).muted);
    }

    #[test]
    fn missing_completed_summary_frozen_speed_and_owner_health_silence() {
        let mut owner = owner(5);
        advance(&mut owner, 0);
        advance(&mut owner, 1);
        let mut snapshot = (*owner.latest_snapshot()).clone();
        let mut projector = SnapshotAudioProjector::default();
        assert!(!projector.observe(&snapshot, true).muted);
        let summary = snapshot.completed_summary.take();
        let missing = projector.observe(&snapshot, true);
        assert!(missing.muted && missing.reset);
        assert!(missing.frame.is_none());
        snapshot.completed_summary = summary;
        for speed in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            snapshot.playback.speed_multiplier = speed;
            let frozen = projector.observe(&snapshot, true);
            assert!(frozen.muted && frozen.frame.is_none());
        }
        snapshot.playback.speed_multiplier = 1.0;
        let resumed = projector.observe(&snapshot, true);
        assert!(resumed.reset && !resumed.muted);
        assert_eq!(resumed.frame.expect("ambience after frozen playback").births, 0);
        snapshot.health = HostHealth::Blocked(crate::HostBlocker::ScientificFault);
        assert!(projector.observe(&snapshot, true).muted);
        snapshot.health = HostHealth::Faulted(crate::HostFault::Protocol {
            code: "test_fault".to_owned(),
            message: "owner cannot advance".to_owned(),
        });
        assert!(projector.observe(&snapshot, true).frame.is_none());
    }
}
