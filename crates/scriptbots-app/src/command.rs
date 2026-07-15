use crossfire::mpmc;
use crossfire::{MAsyncTx, MRx, TryRecvError, TrySendError, detect_backoff_cfg};
use scriptbots_core::ControlCommand;
use std::sync::Arc;
use tracing::warn;

pub type CommandSender = MAsyncTx<ControlCommand>;
pub type CommandReceiver = MRx<ControlCommand>;
pub type CommandDrain = Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync>;
pub type CommandSubmit = Arc<dyn Fn(ControlCommand) -> bool + Send + Sync>;

pub fn create_command_bus(capacity: usize) -> (CommandSender, CommandReceiver) {
    detect_backoff_cfg();
    mpmc::bounded_tx_async_rx_blocking(capacity)
}

#[must_use]
pub fn drain_pending_commands(receiver: &CommandReceiver) -> Vec<ControlCommand> {
    let mut commands = Vec::new();
    loop {
        match receiver.try_recv() {
            Ok(command) => commands.push(command),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => break,
        }
    }
    commands
}

pub fn make_command_drain(receiver: CommandReceiver) -> CommandDrain {
    let receiver = Arc::new(receiver);
    Arc::new(move || drain_pending_commands(&receiver))
}

pub fn make_command_submit(sender: CommandSender) -> CommandSubmit {
    let sender = Arc::new(sender);
    Arc::new(move |command: ControlCommand| {
        if let Err(error) = command.validate() {
            warn!(%error, "rejected invalid control command before queue admission");
            return false;
        }
        match sender.try_send(command) {
            Ok(()) => true,
            Err(TrySendError::Full(cmd)) => {
                warn!(?cmd, "control command queue full; dropping command");
                false
            }
            Err(TrySendError::Disconnected(cmd)) => {
                warn!(?cmd, "control command queue disconnected");
                false
            }
        }
    })
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use scriptbots_core::{
        ControlDisposition, ScriptBotsConfig, SimulationCommand, WorldState, apply_control_command,
    };

    #[test]
    fn submit_rejects_non_finite_speed_before_queue_admission() {
        let (sender, receiver) = create_command_bus(1);
        let submit = make_command_submit(sender);
        assert!(!(submit)(ControlCommand::UpdateSimulation(
            SimulationCommand {
                paused: Some(false),
                speed_multiplier: Some(f32::NAN),
                step_once: false,
            }
        )));
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn finite_speed_is_admitted_and_clamped_when_applied() {
        let (sender, receiver) = create_command_bus(1);
        let submit = make_command_submit(sender);
        assert!((submit)(ControlCommand::UpdateSimulation(
            SimulationCommand {
                paused: Some(false),
                speed_multiplier: Some(128.0),
                step_once: false,
            }
        )));

        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let pending = drain_pending_commands(&receiver);
        assert_eq!(pending.len(), 1);
        let disposition = apply_control_command(
            &mut world,
            pending.into_iter().next().expect("one playback command"),
        )
        .expect("normalized playback disposition");
        assert_eq!(
            disposition,
            ControlDisposition::Playback(SimulationCommand {
                paused: Some(false),
                speed_multiplier: Some(32.0),
                step_once: false,
            })
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{
        ControlDisposition, ScriptBotsConfig, SimulationCommand, WorldState, apply_control_command,
    };
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TargetRunState {
        Running,
        Paused,
        Stopped,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TargetScienceTicks {
        NoneAtFrozenBoundary,
        Exactly(u16),
        TriggeringTickThenStop,
        NoAdditionalTick,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TargetApplicationState {
        Applied {
            count: u16,
        },
        AppliedAfterDisconnect {
            count: u16,
        },
        ExistingStatus,
        Rejected {
            count: u16,
            reason: &'static str,
        },
        AppliedAndRejected {
            applied: u16,
            rejected: u16,
            reason: &'static str,
        },
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TargetJournalState {
        NotRequired {
            count: u16,
        },
        PendingThenModeCommit {
            count: u16,
        },
        Mixed {
            not_required: u16,
            pending_then_mode_commit: u16,
        },
        ExistingStatus,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TargetCommandCase {
        name: &'static str,
        envelope_count: u16,
        initial: TargetRunState,
        final_state: TargetRunState,
        control_revision_delta: u16,
        science_ticks: TargetScienceTicks,
        application: TargetApplicationState,
        journal: TargetJournalState,
    }

    // This is the checked Phase 2.1 contract, not an implementation of HostCore. Queue burst
    // rows describe an unserviced capacity-32 admission window containing Step envelopes. An
    // overload rejection is terminal and queryable, but it has no AdmissionSequence and requires
    // no journal record. "Mode commit" means CommittedVolatile in memory mode or Durable in file
    // mode; application and journal progress remain independent axes.
    const TARGET_COMMAND_TRUTH_TABLE: &[TargetCommandCase] = &[
        TargetCommandCase {
            name: "pause",
            envelope_count: 1,
            initial: TargetRunState::Running,
            final_state: TargetRunState::Paused,
            control_revision_delta: 1,
            science_ticks: TargetScienceTicks::NoneAtFrozenBoundary,
            application: TargetApplicationState::Applied { count: 1 },
            journal: TargetJournalState::NotRequired { count: 1 },
        },
        TargetCommandCase {
            name: "resume",
            envelope_count: 1,
            initial: TargetRunState::Paused,
            final_state: TargetRunState::Running,
            control_revision_delta: 1,
            science_ticks: TargetScienceTicks::NoneAtFrozenBoundary,
            application: TargetApplicationState::Applied { count: 1 },
            journal: TargetJournalState::NotRequired { count: 1 },
        },
        TargetCommandCase {
            name: "speed",
            envelope_count: 1,
            initial: TargetRunState::Running,
            final_state: TargetRunState::Running,
            control_revision_delta: 1,
            science_ticks: TargetScienceTicks::NoneAtFrozenBoundary,
            application: TargetApplicationState::Applied { count: 1 },
            journal: TargetJournalState::NotRequired { count: 1 },
        },
        TargetCommandCase {
            name: "step",
            envelope_count: 1,
            initial: TargetRunState::Running,
            final_state: TargetRunState::Paused,
            control_revision_delta: 1,
            science_ticks: TargetScienceTicks::Exactly(1),
            application: TargetApplicationState::Applied { count: 1 },
            journal: TargetJournalState::PendingThenModeCommit { count: 1 },
        },
        TargetCommandCase {
            name: "step_then_resume",
            envelope_count: 2,
            initial: TargetRunState::Running,
            final_state: TargetRunState::Running,
            control_revision_delta: 2,
            science_ticks: TargetScienceTicks::Exactly(1),
            application: TargetApplicationState::Applied { count: 2 },
            journal: TargetJournalState::Mixed {
                not_required: 1,
                pending_then_mode_commit: 1,
            },
        },
        TargetCommandCase {
            name: "resume_then_step",
            envelope_count: 2,
            initial: TargetRunState::Paused,
            final_state: TargetRunState::Paused,
            control_revision_delta: 2,
            science_ticks: TargetScienceTicks::Exactly(1),
            application: TargetApplicationState::Applied { count: 2 },
            journal: TargetJournalState::Mixed {
                not_required: 1,
                pending_then_mode_commit: 1,
            },
        },
        TargetCommandCase {
            name: "config",
            envelope_count: 1,
            initial: TargetRunState::Paused,
            final_state: TargetRunState::Paused,
            control_revision_delta: 1,
            science_ticks: TargetScienceTicks::NoneAtFrozenBoundary,
            application: TargetApplicationState::Applied { count: 1 },
            journal: TargetJournalState::PendingThenModeCommit { count: 1 },
        },
        TargetCommandCase {
            name: "selection",
            envelope_count: 1,
            initial: TargetRunState::Paused,
            final_state: TargetRunState::Paused,
            control_revision_delta: 1,
            science_ticks: TargetScienceTicks::NoneAtFrozenBoundary,
            application: TargetApplicationState::Applied { count: 1 },
            journal: TargetJournalState::NotRequired { count: 1 },
        },
        TargetCommandCase {
            name: "auto_pause",
            envelope_count: 1,
            initial: TargetRunState::Running,
            final_state: TargetRunState::Paused,
            control_revision_delta: 1,
            science_ticks: TargetScienceTicks::TriggeringTickThenStop,
            application: TargetApplicationState::Applied { count: 1 },
            journal: TargetJournalState::NotRequired { count: 1 },
        },
        TargetCommandCase {
            name: "duplicate_command_id",
            envelope_count: 1,
            initial: TargetRunState::Paused,
            final_state: TargetRunState::Paused,
            control_revision_delta: 0,
            science_ticks: TargetScienceTicks::NoAdditionalTick,
            application: TargetApplicationState::ExistingStatus,
            journal: TargetJournalState::ExistingStatus,
        },
        TargetCommandCase {
            name: "expected_revision_conflict",
            envelope_count: 1,
            initial: TargetRunState::Paused,
            final_state: TargetRunState::Paused,
            control_revision_delta: 0,
            science_ticks: TargetScienceTicks::NoneAtFrozenBoundary,
            application: TargetApplicationState::Rejected {
                count: 1,
                reason: "expected_control_revision_conflict",
            },
            journal: TargetJournalState::NotRequired { count: 1 },
        },
        TargetCommandCase {
            name: "disconnected_client",
            envelope_count: 1,
            initial: TargetRunState::Paused,
            final_state: TargetRunState::Paused,
            control_revision_delta: 1,
            science_ticks: TargetScienceTicks::NoneAtFrozenBoundary,
            application: TargetApplicationState::AppliedAfterDisconnect { count: 1 },
            journal: TargetJournalState::PendingThenModeCommit { count: 1 },
        },
        TargetCommandCase {
            name: "unserviced_step_burst_1_capacity_32",
            envelope_count: 1,
            initial: TargetRunState::Running,
            final_state: TargetRunState::Paused,
            control_revision_delta: 1,
            science_ticks: TargetScienceTicks::Exactly(1),
            application: TargetApplicationState::Applied { count: 1 },
            journal: TargetJournalState::PendingThenModeCommit { count: 1 },
        },
        TargetCommandCase {
            name: "unserviced_step_burst_32_capacity_32",
            envelope_count: 32,
            initial: TargetRunState::Running,
            final_state: TargetRunState::Paused,
            control_revision_delta: 32,
            science_ticks: TargetScienceTicks::Exactly(32),
            application: TargetApplicationState::Applied { count: 32 },
            journal: TargetJournalState::PendingThenModeCommit { count: 32 },
        },
        TargetCommandCase {
            name: "unserviced_step_burst_33_capacity_32",
            envelope_count: 33,
            initial: TargetRunState::Running,
            final_state: TargetRunState::Paused,
            control_revision_delta: 32,
            science_ticks: TargetScienceTicks::Exactly(32),
            application: TargetApplicationState::AppliedAndRejected {
                applied: 32,
                rejected: 1,
                reason: "overloaded_before_admission",
            },
            journal: TargetJournalState::Mixed {
                not_required: 1,
                pending_then_mode_commit: 32,
            },
        },
        TargetCommandCase {
            name: "unserviced_step_burst_1000_capacity_32",
            envelope_count: 1_000,
            initial: TargetRunState::Running,
            final_state: TargetRunState::Paused,
            control_revision_delta: 32,
            science_ticks: TargetScienceTicks::Exactly(32),
            application: TargetApplicationState::AppliedAndRejected {
                applied: 32,
                rejected: 968,
                reason: "overloaded_before_admission",
            },
            journal: TargetJournalState::Mixed {
                not_required: 968,
                pending_then_mode_commit: 32,
            },
        },
        TargetCommandCase {
            name: "shutdown_empty",
            envelope_count: 1,
            initial: TargetRunState::Paused,
            final_state: TargetRunState::Stopped,
            control_revision_delta: 1,
            science_ticks: TargetScienceTicks::NoneAtFrozenBoundary,
            application: TargetApplicationState::Applied { count: 1 },
            journal: TargetJournalState::PendingThenModeCommit { count: 1 },
        },
        TargetCommandCase {
            name: "shutdown_after_pending_step_and_config",
            envelope_count: 3,
            initial: TargetRunState::Running,
            final_state: TargetRunState::Stopped,
            control_revision_delta: 3,
            science_ticks: TargetScienceTicks::Exactly(1),
            application: TargetApplicationState::Applied { count: 3 },
            journal: TargetJournalState::PendingThenModeCommit { count: 3 },
        },
    ];

    fn application_counts(expectation: TargetApplicationState) -> (u16, u16, u16) {
        match expectation {
            TargetApplicationState::Applied { count }
            | TargetApplicationState::AppliedAfterDisconnect { count } => (count, 0, 0),
            TargetApplicationState::ExistingStatus => (0, 0, 1),
            TargetApplicationState::Rejected { count, reason } => {
                assert!(!reason.is_empty());
                (0, count, 0)
            }
            TargetApplicationState::AppliedAndRejected {
                applied,
                rejected,
                reason,
            } => {
                assert!(!reason.is_empty());
                (applied, rejected, 0)
            }
        }
    }

    fn journal_counts(expectation: TargetJournalState) -> (u16, u16, u16) {
        match expectation {
            TargetJournalState::NotRequired { count } => (count, 0, 0),
            TargetJournalState::PendingThenModeCommit { count } => (0, count, 0),
            TargetJournalState::Mixed {
                not_required,
                pending_then_mode_commit,
            } => (not_required, pending_then_mode_commit, 0),
            TargetJournalState::ExistingStatus => (0, 0, 1),
        }
    }

    #[test]
    fn target_command_truth_table_is_complete_and_self_consistent() {
        const REQUIRED_CASES: &[&str] = &[
            "pause",
            "resume",
            "speed",
            "step",
            "step_then_resume",
            "resume_then_step",
            "config",
            "selection",
            "auto_pause",
            "duplicate_command_id",
            "expected_revision_conflict",
            "disconnected_client",
            "unserviced_step_burst_1_capacity_32",
            "unserviced_step_burst_32_capacity_32",
            "unserviced_step_burst_33_capacity_32",
            "unserviced_step_burst_1000_capacity_32",
            "shutdown_empty",
            "shutdown_after_pending_step_and_config",
        ];

        assert_eq!(
            TARGET_COMMAND_TRUTH_TABLE
                .iter()
                .map(|case| case.name)
                .collect::<Vec<_>>(),
            REQUIRED_CASES
        );

        for case in TARGET_COMMAND_TRUTH_TABLE {
            let (applied, rejected, existing) = application_counts(case.application);
            let (not_required, pending, existing_journal) = journal_counts(case.journal);
            assert_eq!(
                applied + rejected + existing,
                case.envelope_count,
                "every envelope needs one terminal application result for {}",
                case.name
            );
            assert_eq!(
                not_required + pending + existing_journal,
                case.envelope_count,
                "every envelope needs an explicit journal result for {}",
                case.name
            );
            assert_eq!(
                case.control_revision_delta, applied,
                "ControlRevision changes once per successfully applied envelope for {}",
                case.name
            );
            assert!(
                !matches!(case.initial, TargetRunState::Stopped),
                "no case starts after shutdown"
            );
        }

        let case = |name| {
            TARGET_COMMAND_TRUTH_TABLE
                .iter()
                .find(|case| case.name == name)
                .expect("required target command case")
        };
        assert_eq!(
            (
                case("step_then_resume").final_state,
                case("step_then_resume").science_ticks,
            ),
            (TargetRunState::Running, TargetScienceTicks::Exactly(1))
        );
        assert_eq!(
            (
                case("resume_then_step").final_state,
                case("resume_then_step").science_ticks,
            ),
            (TargetRunState::Paused, TargetScienceTicks::Exactly(1))
        );
        assert_eq!(
            application_counts(case("unserviced_step_burst_33_capacity_32").application),
            (32, 1, 0)
        );
        assert_eq!(
            application_counts(case("unserviced_step_burst_1000_capacity_32").application),
            (32, 968, 0)
        );
    }

    #[test]
    fn current_capacity_32_bus_accepts_32_and_rejects_the_33rd() {
        let (sender, _receiver) = create_command_bus(32);
        for index in 0..32 {
            let result = sender.try_send(ControlCommand::UpdateSimulation(
                SimulationCommand::default(),
            ));
            assert!(result.is_ok(), "command {index} should fit: {result:?}");
        }
        assert!(matches!(
            sender.try_send(ControlCommand::UpdateSimulation(
                SimulationCommand::default()
            )),
            Err(TrySendError::Full(_))
        ));
    }

    #[test]
    fn drained_mixed_command_classes_preserve_enqueue_order() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let (sender, receiver) = create_command_bus(2);
        sender
            .try_send(ControlCommand::UpdateSimulation(SimulationCommand {
                paused: Some(true),
                speed_multiplier: None,
                step_once: false,
            }))
            .expect("pause fits");
        let mut updated = world.config().clone();
        updated.food_max = 0.73;
        sender
            .try_send(ControlCommand::UpdateConfig(Box::new(updated)))
            .expect("config fits");

        let commands = drain_pending_commands(&receiver);
        assert_eq!(commands.len(), 2);
        assert!(matches!(commands[0], ControlCommand::UpdateSimulation(_)));
        assert!(matches!(commands[1], ControlCommand::UpdateConfig(_)));

        let dispositions = commands
            .into_iter()
            .map(|command| {
                apply_control_command(&mut world, command).expect("ordered command application")
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            dispositions.as_slice(),
            [
                ControlDisposition::Playback(_),
                ControlDisposition::WorldApplied
            ]
        ));
        assert!((world.config().food_max - 0.73).abs() < f32::EPSILON);
    }

    #[test]
    #[should_panic(
        expected = "KNOWN DEFECT bd-2z0.4.1: mixed command classes do not apply in enqueue order"
    )]
    fn target_mixed_command_classes_apply_atomically_in_enqueue_order() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let (sender, receiver) = create_command_bus(2);
        sender
            .try_send(ControlCommand::UpdateSimulation(SimulationCommand {
                paused: Some(true),
                speed_multiplier: None,
                step_once: true,
            }))
            .expect("step fits");
        let mut updated = world.config().clone();
        updated.food_max = 0.73;
        sender
            .try_send(ControlCommand::UpdateConfig(Box::new(updated)))
            .expect("config fits");

        let mut deferred_playback = Vec::new();
        for command in drain_pending_commands(&receiver) {
            match apply_control_command(&mut world, command).expect("apply drained command") {
                ControlDisposition::WorldApplied => {}
                ControlDisposition::Playback(command) => deferred_playback.push(command),
            }
        }

        assert!((world.config().food_max - 0.73).abs() < f32::EPSILON);
        assert_eq!(
            deferred_playback.len(),
            1,
            "current semantic-order defect must stay visible"
        );
        assert!(
            deferred_playback.is_empty(),
            "KNOWN DEFECT bd-2z0.4.1: mixed command classes do not apply in enqueue order"
        );
    }

    #[test]
    #[should_panic(
        expected = "KNOWN DEFECT bd-2z0.4.1: shutdown returns while admitted command work is pending"
    )]
    fn target_shutdown_terminally_resolves_pending_commands() {
        let world = Arc::new(Mutex::new(
            WorldState::new(ScriptBotsConfig::default()).expect("world"),
        ));
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let mut updated = world.lock().expect("world lock").config().clone();
        updated.food_max = 0.73;
        assert!(submit(ControlCommand::UpdateConfig(Box::new(updated))));

        runtime.shutdown().expect("control runtime shutdown");
        let observed_at_shutdown = world.lock().expect("world lock").config().food_max;
        assert!((observed_at_shutdown - 0.73).abs() > f32::EPSILON);

        let mut world_guard = world.lock().expect("world lock");
        for command in (drain.as_ref())() {
            let _ = apply_control_command(&mut world_guard, command)
                .expect("apply command after shutdown");
        }
        drop(world_guard);
        assert!((world.lock().expect("world lock").config().food_max - 0.73).abs() < f32::EPSILON);
        assert!(
            (observed_at_shutdown - 0.73).abs() < f32::EPSILON,
            "KNOWN DEFECT bd-2z0.4.1: shutdown returns while admitted command work is pending"
        );
    }
}
