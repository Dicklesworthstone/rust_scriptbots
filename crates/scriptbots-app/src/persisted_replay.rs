//! Shared reconstruction of persisted host inputs and full checkpoint science.
//!
//! Native verification and offline visual exports use this interpreter so a
//! scheduled patch cannot acquire a different order in an exported frame.

use anyhow::{Context, Result, bail};
use scriptbots_core::{
    PersistenceAdmissionSession, ReplayEventKind, ScriptBotsConfig, WorldCheckpointV1,
    WorldPersistence, WorldState,
};
use scriptbots_runtime::ScheduledConfigPatch;
use scriptbots_storage::{PersistedReplayEvent, StorageReader};
use std::collections::{BTreeMap, BTreeSet};

/// Ticks carrying a canonical world digest, up to `tick_limit`.
pub fn recorded_digest_ticks(events: &[PersistedReplayEvent], tick_limit: u64) -> BTreeSet<u64> {
    events
        .iter()
        .filter(|entry| {
            entry.tick <= tick_limit
                && matches!(entry.event.kind, ReplayEventKind::WorldDigest { .. })
        })
        .map(|entry| entry.tick)
        .collect()
}

/// A science command, scheduled patch or persistence cut in its actual journal order.
#[derive(Debug, Clone, PartialEq)]
pub enum ReplayOperatorCommand {
    /// Operator input applied through the same core entry point as the host.
    World(scriptbots_core::ControlCommand),
    /// Explicit batch boundary, ordered relative to science inputs.
    FlushPersistence,
    /// Applied statement from the recorded owner's schedule.
    ScheduledConfigPatch(ScheduledConfigPatch),
}

/// What replay reproduces from the recording beyond the scenario itself.
#[derive(Debug, Clone, Default)]
pub struct ReplayMirror {
    /// Ticks whose batch carries a canonical world digest (`None`: the final tick only).
    pub digest_ticks: Option<BTreeSet<u64>>,
    /// Off-cadence batch boundaries; see [`recorded_flush_ticks`].
    pub flush_ticks: BTreeSet<u64>,
    /// Operator commands, checkpoint schedule patches and explicit flushes in journal order.
    pub operator_commands: BTreeMap<u64, Vec<ReplayOperatorCommand>>,
}

/// Recorded inputs and whether the host journal is their complete authority.
#[derive(Debug, Clone, Default)]
pub struct RecordedInputHistory {
    /// Commands, scheduled patches and explicit flushes in actual journal order.
    pub operator_commands: BTreeMap<u64, Vec<ReplayOperatorCommand>>,
    /// A recorded host owns its schedule even when no scheduled patch was applied.
    /// External scenario schedules are used only by legacy recordings with no host journal.
    pub has_host_journal: bool,
}

/// Every science-changing input a recorded host applied, by applied tick.
///
/// Flushes preserve the cut relative to science commands at the same tick.
/// Playback-only commands are skipped; migrations require an archipelago and are
/// refused. Genesis and checkpoint replay both join applied scheduled patches by
/// their journal identity. Distinct sessions may only cover disjoint tick ranges:
/// session identifiers do not define a scientific order at an overlapping boundary.
#[allow(
    clippy::too_many_lines,
    reason = "one journal read preserves the exact checkpoint cut, command outcomes, scheduled input order and durable boundary refusals"
)]
pub fn recorded_input_history(
    storage: &StorageReader,
    checkpoint: Option<(u64, scriptbots_runtime::HostJournalCutV1)>,
) -> Result<RecordedInputHistory> {
    use scriptbots_core::ControlCommand;
    use scriptbots_runtime::{ApplicationState, HostCommand};

    let sessions = storage.host_journal_sessions()?;
    let has_host_journal = !sessions.is_empty();
    if let Some((tick, cut)) = checkpoint {
        if !matches!(sessions.as_slice(), [session] if *session == cut.session_id) {
            bail!("checkpoint replay requires its exact recorded host session");
        }
        let after = cut
            .last_journal_sequence
            .checked_sub(1)
            .filter(|sequence| *sequence != 0)
            .map(|sequence| scriptbots_runtime::JournalBatchId::new(cut.session_id, sequence));
        let page = storage.host_journal_session_conformance_page(
            cut.session_id,
            after,
            1,
            scriptbots_storage::MAX_HOST_JOURNAL_ARCHIVE_BYTES,
        )?;
        let boundary_matches = if cut.last_journal_sequence == 0 {
            tick == 0 && cut.last_scientific_event_sequence == 0
        } else {
            page.records.first().is_some_and(|record| {
                record.batch_id.sequence() == cut.last_journal_sequence
                    && record.applied.tick.0 == tick
            })
        };
        if !boundary_matches
            || cut.commitment_requirement != scriptbots_runtime::ShutdownCommitRequirement::Durable
            || cut.last_journal_sequence > page.progress.journal.durable
            || cut.last_scientific_event_sequence > page.progress.events.durable
        {
            bail!("checkpoint replay journal cut differs from the recorded durable boundary");
        }
    }
    let mut commands: BTreeMap<u64, Vec<ReplayOperatorCommand>> = BTreeMap::new();
    let mut session_ranges: Vec<(scriptbots_runtime::HostSessionId, u64, u64)> = Vec::new();
    for session in sessions {
        let mut session_commands: BTreeMap<u64, Vec<(u64, ReplayOperatorCommand)>> =
            BTreeMap::new();
        let mut after = None;
        loop {
            let page = storage
                .command_journal_page(
                    session,
                    after,
                    1024,
                    scriptbots_storage::MAX_HOST_JOURNAL_ARCHIVE_BYTES,
                )
                .with_context(|| {
                    format!(
                        "failed to read the commands of host session {}",
                        session.get()
                    )
                })?;
            for record in page.commands {
                if let Some((tick, cut)) = checkpoint {
                    if record.batch_id.sequence() <= cut.last_journal_sequence {
                        if record.terminal_boundary.tick.0 > tick {
                            bail!("checkpoint journal prefix contains a later command boundary");
                        }
                        continue;
                    }
                    if record.terminal_boundary.tick.0 < tick {
                        bail!("checkpoint journal suffix contains an earlier command boundary");
                    }
                }
                let applied = record.lifecycle.terminal().is_some_and(|transition| {
                    matches!(transition.application(), ApplicationState::Applied(_))
                });
                if !applied {
                    continue;
                }
                let command = match record.lifecycle.envelope().command.clone() {
                    HostCommand::Intervention(command) => ControlCommand::Intervention(command),
                    HostCommand::UpdateConfig(config) => ControlCommand::UpdateConfig(config),
                    HostCommand::UpdateSelection(update) => ControlCommand::UpdateSelection(update),
                    HostCommand::AdjustAgentMutationRates {
                        agent_uid,
                        delta_primary,
                        delta_secondary,
                    } => ControlCommand::AdjustAgentMutationRates {
                        agent_uid,
                        delta_primary,
                        delta_secondary,
                    },
                    HostCommand::SpawnAgent { herbivore_tendency } => {
                        ControlCommand::SpawnAgent { herbivore_tendency }
                    }
                    HostCommand::SpawnCrossover { parent_a, parent_b } => {
                        ControlCommand::SpawnCrossover { parent_a, parent_b }
                    }
                    HostCommand::ApplyMap {
                        artifact,
                        existing_agent_policy,
                    } => ControlCommand::ApplyMap {
                        artifact,
                        existing_agent_policy,
                    },
                    HostCommand::FlushPersistence => {
                        session_commands
                            .entry(record.terminal_boundary.tick.0)
                            .or_default()
                            .push((
                                record.batch_id.sequence(),
                                ReplayOperatorCommand::FlushPersistence,
                            ));
                        continue;
                    }
                    HostCommand::Emigrate { .. } | HostCommand::Immigrate { .. } => bail!(
                        "the recording migrated agents between islands at tick {}; replaying an \
                         archipelago run is not supported by --replay-db",
                        record.terminal_boundary.tick.0
                    ),
                    HostCommand::Pause
                    | HostCommand::Resume
                    | HostCommand::SetSpeed(_)
                    | HostCommand::Step
                    | HostCommand::UpdateSimulation(_)
                    | HostCommand::Shutdown => continue,
                };
                session_commands
                    .entry(record.terminal_boundary.tick.0)
                    .or_default()
                    .push((
                        record.batch_id.sequence(),
                        ReplayOperatorCommand::World(command),
                    ));
            }
            match page.next_after {
                Some(cursor) => after = Some(cursor),
                None => break,
            }
        }
        // Scheduled patches are separate journal records, not operator commands.
        // Their actual position relative to same-tick commands is authoritative.
        let mut after = checkpoint.and_then(|(_, cut)| {
            (cut.last_journal_sequence != 0).then_some(
                scriptbots_runtime::JournalBatchId::new(cut.session_id, cut.last_journal_sequence),
            )
        });
        let minimum_tick = checkpoint.map_or(0, |(tick, _)| tick);
        let mut session_range: Option<(u64, u64)> = None;
        loop {
            let page = storage.host_journal_session_conformance_page(
                session,
                after,
                1024,
                scriptbots_storage::MAX_HOST_JOURNAL_ARCHIVE_BYTES,
            )?;
            for record in page.records {
                if record.state != scriptbots_storage::HostJournalRecordState::Durable
                    || record.applied.tick.0 < minimum_tick
                {
                    bail!("replay input history differs from its durable journal boundary");
                }
                let tick = record.applied.tick.0;
                session_range = Some(session_range.map_or((tick, tick), |(start, end)| {
                    (start.min(tick), end.max(tick))
                }));
                let Some(evidence) = record.scheduled_patch else {
                    continue;
                };
                if evidence.scheduled.due_tick != record.applied.tick {
                    bail!("checkpoint replay schedule differs from its durable journal boundary");
                }
                if matches!(
                    evidence.outcome,
                    scriptbots_runtime::ScheduledPatchOutcome::Applied { .. }
                ) {
                    session_commands.entry(tick).or_default().push((
                        record.batch_id.sequence(),
                        ReplayOperatorCommand::ScheduledConfigPatch(evidence.scheduled),
                    ));
                }
            }
            match page.next_after {
                Some(cursor) => after = Some(cursor),
                None => break,
            }
        }
        if let Some((start, end)) = session_range {
            if let Some((other, _, _)) = session_ranges
                .iter()
                .find(|(_, other_start, other_end)| start <= *other_end && *other_start <= end)
            {
                bail!(
                    "replay host sessions {} and {} overlap ticks; their input order is ambiguous",
                    other.get(),
                    session.get()
                );
            }
            session_ranges.push((session, start, end));
        }
        for (tick, mut actions) in session_commands {
            actions.sort_by_key(|(sequence, _)| *sequence);
            if actions.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                bail!("replay journal sequence contains multiple inputs without an ordering");
            }
            commands
                .entry(tick)
                .or_default()
                .extend(actions.into_iter().map(|(_, action)| action));
        }
    }
    Ok(RecordedInputHistory {
        operator_commands: commands,
        has_host_journal,
    })
}

/// Re-apply inputs at `tick` through the same core entry point the host uses.
pub fn apply_recorded_operator_commands(
    world: &mut WorldState,
    persistence: &mut PersistenceAdmissionSession,
    current_config: &mut serde_json::Value,
    mirror: &ReplayMirror,
    tick: u64,
) -> Result<()> {
    let Some(commands) = mirror.operator_commands.get(&tick) else {
        return Ok(());
    };
    for command in commands {
        match command {
            ReplayOperatorCommand::World(command) => {
                scriptbots_core::apply_control_command(world, command.clone()).with_context(
                    || format!("failed to re-apply a recorded operator command at tick {tick}"),
                )?;
            }
            ReplayOperatorCommand::FlushPersistence => {
                persistence.finalize(world).with_context(|| {
                    format!("failed to re-apply a recorded persistence flush at tick {tick}")
                })?;
            }
            ReplayOperatorCommand::ScheduledConfigPatch(statement) => {
                if statement.due_tick.0 != tick {
                    bail!("recorded schedule patch is attached to the wrong replay tick");
                }
                let patch = statement.decode_patch().map_err(anyhow::Error::msg)?;
                let config = crate::resolve_scheduled_config_patch(world.config(), &patch)
                    .map_err(anyhow::Error::msg)?;
                world.apply_config_update(config).with_context(|| {
                    format!(
                        "failed to re-apply recorded schedule patch {} at tick {tick}",
                        statement.sequence
                    )
                })?;
            }
        }
    }
    // Scenario patches must merge onto the most recent operator configuration.
    *current_config = serde_json::to_value(world.config())?;
    Ok(())
}

/// Recorded batch boundaries that the cadence alone would not produce.
pub fn recorded_flush_ticks(
    batch_ticks: &[u64],
    config: &ScriptBotsConfig,
    tick_limit: u64,
) -> BTreeSet<u64> {
    let interval = u64::from(config.persistence_interval.max(1));
    batch_ticks
        .iter()
        .copied()
        .filter(|&tick| tick > 0 && tick < tick_limit && !tick.is_multiple_of(interval))
        .collect()
}

/// Cut a partial batch when no explicit journaled flush orders it against inputs.
pub fn mirror_recorded_flush(
    world: &mut WorldState,
    persistence: &mut PersistenceAdmissionSession,
    mirror: &ReplayMirror,
) -> Result<()> {
    let tick = world.tick().0;
    let ordered_flush = mirror.operator_commands.get(&tick).is_some_and(|commands| {
        commands
            .iter()
            .any(|command| matches!(command, ReplayOperatorCommand::FlushPersistence))
    });
    if mirror.flush_ticks.contains(&tick) && !ordered_flush {
        persistence
            .finalize(world)
            .context("failed to mirror a recorded off-cadence persistence batch")?;
    }
    Ok(())
}

/// Whether the batch completing `completed_tick` must carry a canonical digest.
pub fn digest_due(
    digest_ticks: Option<&BTreeSet<u64>>,
    completed_tick: u64,
    final_tick: u64,
) -> bool {
    digest_ticks.map_or(completed_tick == final_tick, |ticks| {
        ticks.contains(&completed_tick)
    })
}

/// A full owner source retains the auxiliary science absent from its inner cut.
pub enum ReplayCheckpointSource {
    /// A raw scientific fixture, with its own stricter restoration limitations.
    Science(Box<WorldCheckpointV1>),
    /// Complete owner checkpoint, including the durable journal cut and schedule.
    Host(Box<scriptbots_runtime::HostCheckpointV1>),
}

impl ReplayCheckpointSource {
    /// Scientific cut inside either artifact.
    pub fn world_checkpoint(&self) -> &WorldCheckpointV1 {
        match self {
            Self::Science(checkpoint) => checkpoint,
            Self::Host(checkpoint) => checkpoint.world_checkpoint(),
        }
    }

    /// Exact durable journal prefix captured by a full owner.
    pub fn journal_cut(&self) -> Option<scriptbots_runtime::HostJournalCutV1> {
        match self {
            Self::Science(_) => None,
            Self::Host(checkpoint) => Some(checkpoint.journal_cut()),
        }
    }

    /// Original recording configuration when the complete owner retained it.
    ///
    /// A raw science checkpoint normalizes persistence to zero, so only that
    /// legacy lane still needs the explicitly supplied recording configuration.
    pub fn recording_config<'a>(&'a self, fallback: &'a ScriptBotsConfig) -> &'a ScriptBotsConfig {
        match self {
            Self::Science(_) => fallback,
            Self::Host(checkpoint) => checkpoint.config(),
        }
    }

    /// Bind a replay sink without rewriting full-owner recording continuity.
    pub fn bind_restored_persistence(
        &self,
        world: &mut WorldState,
        sink: Box<dyn WorldPersistence>,
        fallback: &ScriptBotsConfig,
    ) -> Result<PersistenceAdmissionSession> {
        match self {
            Self::Host(checkpoint) => {
                let mut persistence = world
                    .bind_persistence(sink)
                    .context("failed to bind full-owner replay persistence")?;
                persistence
                    .restore_admission_cursor(world, checkpoint.persistence_last_admitted_tick())
                    .context("failed to restore full-owner replay admission cursor")?;
                Ok(persistence)
            }
            Self::Science(checkpoint) => {
                let interval = fallback.persistence_interval.max(1);
                let replay_cap = if fallback.replay_event_tick_cap == 0 {
                    checkpoint.config().replay_event_tick_cap.max(65536)
                } else {
                    fallback.replay_event_tick_cap
                };
                world
                    .bind_continuation_persistence(sink, interval, replay_cap)
                    .context("failed to bind scientific-checkpoint replay persistence")
            }
        }
    }

    /// Restore science without discarding any full-owner companion state.
    pub fn restore_world(&self) -> Result<WorldState> {
        let registry = crate::brains::create_brain_registry_for_checkpoint(self.world_checkpoint())
            .context("failed to reconstruct the checkpoint's exact brain registry")?;
        match self {
            Self::Science(checkpoint) => WorldState::restore_checkpoint_v1(checkpoint, registry)
                .context(
                    "failed to restore scientific checkpoint; auxiliary state requires a full owner checkpoint",
                ),
            Self::Host(checkpoint) => checkpoint
                .restore_world(registry)
                .context("failed to restore complete checkpoint science and continuity"),
        }
    }

    /// Bind applied scheduled inputs to the exact captured pending suffix.
    pub fn validate_recorded_schedule(&self, mirror: &ReplayMirror, tick_limit: u64) -> Result<()> {
        let Self::Host(checkpoint) = self else {
            return Ok(());
        };
        let mut pending = checkpoint.remaining_schedule().unwrap_or_default().iter();
        for (&tick, actions) in &mirror.operator_commands {
            for action in actions {
                if let ReplayOperatorCommand::ScheduledConfigPatch(statement) = action {
                    let expected = pending.next().context(
                        "recorded schedule has an action absent from the checkpoint's pending suffix",
                    )?;
                    if statement != expected || statement.due_tick.0 != tick {
                        bail!(
                            "recorded schedule differs from the checkpoint's exact pending suffix"
                        );
                    }
                }
            }
        }
        if pending
            .next()
            .is_some_and(|statement| statement.due_tick.0 < tick_limit)
        {
            bail!(
                "checkpoint replay would cross a due schedule patch without its applied journal record"
            );
        }
        Ok(())
    }
}
