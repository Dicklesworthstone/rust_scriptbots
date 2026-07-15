use asupersync::channel::mpsc::{self, MpscTelemetrySnapshot};
use scriptbots_core::ControlCommand;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use tracing::{trace, trace_span, warn};

const COMMAND_BUS_TELEMETRY_ID: u64 = 0x7362_636d_6462_7573;

/// Failure to enqueue one exact legacy control command.
#[derive(Debug)]
pub enum CommandSendError {
    /// The bounded queue has no unreserved capacity.
    Full(ControlCommand),
    /// The sole receiver closed before admission committed.
    Disconnected(ControlCommand),
}

/// Nonblocking receive outcome for the legacy control command bus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandRecvError {
    /// The queue is open but currently empty.
    Empty,
    /// Every sender is gone or the receiver was closed.
    Disconnected,
}

/// Payload-redacted command-bus state and cumulative counters.
///
/// This diagnostic view is best-effort rather than a linearizable ledger: the
/// channel state and cumulative atomics are sampled independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandBusTelemetry {
    /// Stable deterministic identifier for this bus class.
    pub channel_id: u64,
    /// Maximum committed plus reserved queue slots.
    pub capacity: usize,
    /// Committed commands waiting for the receiver.
    pub queued_commands: usize,
    /// Reserved slots not yet committed or aborted.
    pub reserved_commands: usize,
    /// Commands admitted since bus creation.
    pub admitted_commands: u64,
    /// Commands removed by the sole consumer since bus creation.
    pub drained_commands: u64,
    /// Commands rejected because the bounded queue was full.
    pub full_rejections: u64,
    /// Commands rejected because the receiver was closed.
    pub disconnected_rejections: u64,
    /// Invalid commands rejected before queue admission.
    pub validation_rejections: u64,
    /// Reservation abort/cancellation events observed by Asupersync.
    pub cancellation_count: u64,
    /// Redacted receiver health reported by Asupersync.
    pub receiver_health: &'static str,
    /// Whether the underlying channel has reached a closed state.
    pub closed: bool,
}

#[derive(Debug, Default)]
struct CommandBusCounters {
    admitted_commands: AtomicU64,
    drained_commands: AtomicU64,
    full_rejections: AtomicU64,
    disconnected_rejections: AtomicU64,
    validation_rejections: AtomicU64,
}

impl CommandBusCounters {
    fn snapshot(&self, channel: MpscTelemetrySnapshot) -> CommandBusTelemetry {
        CommandBusTelemetry {
            channel_id: channel.channel_id,
            capacity: channel.capacity,
            queued_commands: channel.queued_messages,
            reserved_commands: channel.reserved_uncommitted_obligations,
            admitted_commands: self.admitted_commands.load(Ordering::Relaxed),
            drained_commands: self.drained_commands.load(Ordering::Relaxed),
            full_rejections: self.full_rejections.load(Ordering::Relaxed),
            disconnected_rejections: self.disconnected_rejections.load(Ordering::Relaxed),
            validation_rejections: self.validation_rejections.load(Ordering::Relaxed),
            cancellation_count: channel.cancellation_count,
            receiver_health: channel.receiver_health,
            closed: channel.closed,
        }
    }
}

/// Cloneable producer for the bounded control-command ingress.
#[derive(Clone, Debug)]
pub struct CommandSender {
    inner: mpsc::Sender<ControlCommand>,
    counters: Arc<CommandBusCounters>,
}

impl CommandSender {
    /// Attempt to enqueue one command without waiting for capacity.
    pub fn try_send(&self, command: ControlCommand) -> Result<(), CommandSendError> {
        let span = trace_span!(
            "control_command_bus.try_send",
            channel_id = COMMAND_BUS_TELEMETRY_ID
        );
        let _entered = span.enter();
        let result = match self.inner.try_send(command) {
            Ok(()) => {
                self.counters
                    .admitted_commands
                    .fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(mpsc::SendError::Full(command)) => {
                self.counters
                    .full_rejections
                    .fetch_add(1, Ordering::Relaxed);
                Err(CommandSendError::Full(command))
            }
            Err(mpsc::SendError::Disconnected(command) | mpsc::SendError::Cancelled(command)) => {
                self.counters
                    .disconnected_rejections
                    .fetch_add(1, Ordering::Relaxed);
                Err(CommandSendError::Disconnected(command))
            }
        };
        let outcome = match &result {
            Ok(()) => "admitted",
            Err(CommandSendError::Full(_)) => "full",
            Err(CommandSendError::Disconnected(_)) => "disconnected",
        };
        if tracing::enabled!(tracing::Level::TRACE) {
            let telemetry = self.telemetry_snapshot();
            trace!(
                outcome,
                queued_commands = telemetry.queued_commands,
                reserved_commands = telemetry.reserved_commands,
                capacity = telemetry.capacity,
                full_rejections = telemetry.full_rejections,
                disconnected_rejections = telemetry.disconnected_rejections,
                "control command admission observed"
            );
        }
        result
    }

    pub(crate) fn record_validation_rejection(&self) {
        self.counters
            .validation_rejections
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Return the fixed capacity of this command bus.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Return a payload-redacted snapshot of the command bus state.
    #[must_use]
    pub fn telemetry_snapshot(&self) -> CommandBusTelemetry {
        self.counters
            .snapshot(self.inner.telemetry_snapshot(COMMAND_BUS_TELEMETRY_ID))
    }

    #[cfg(test)]
    fn try_reserve(&self) -> Result<mpsc::SendPermit<'_, ControlCommand>, mpsc::SendError<()>> {
        self.inner.try_reserve()
    }
}

/// Single-consumer endpoint for the bounded control-command ingress.
///
/// Asupersync's receiver deliberately requires mutable access. The legacy
/// frontend callback is `Send + Sync`, so this adapter serializes only the
/// receiver handle; the channel itself remains the bounded Asupersync queue.
#[derive(Debug)]
pub struct CommandReceiver {
    inner: Mutex<mpsc::Receiver<ControlCommand>>,
    counters: Arc<CommandBusCounters>,
}

impl CommandReceiver {
    /// Attempt to receive one command without blocking.
    pub fn try_recv(&self) -> Result<ControlCommand, CommandRecvError> {
        let result = self
            .inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .try_recv();
        match result {
            Ok(command) => {
                self.counters
                    .drained_commands
                    .fetch_add(1, Ordering::Relaxed);
                Ok(command)
            }
            Err(mpsc::RecvError::Empty) => Err(CommandRecvError::Empty),
            Err(mpsc::RecvError::Disconnected | mpsc::RecvError::Cancelled) => {
                Err(CommandRecvError::Disconnected)
            }
        }
    }

    fn drain_bounded(&self) -> Vec<ControlCommand> {
        let mut receiver = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let mut commands = Vec::new();
        for _ in 0..receiver.capacity() {
            match receiver.try_recv() {
                Ok(command) => commands.push(command),
                Err(
                    mpsc::RecvError::Empty
                    | mpsc::RecvError::Disconnected
                    | mpsc::RecvError::Cancelled,
                ) => break,
            }
        }
        self.counters.drained_commands.fetch_add(
            u64::try_from(commands.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        commands
    }

    /// Return the fixed capacity of this command bus.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .capacity()
    }

    /// Return a payload-redacted snapshot of the command bus state.
    #[must_use]
    pub fn telemetry_snapshot(&self) -> CommandBusTelemetry {
        let channel = self
            .inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .telemetry_snapshot(COMMAND_BUS_TELEMETRY_ID);
        self.counters.snapshot(channel)
    }
}

/// Cloneable callback for the one logical simulation consumer.
///
/// Calls are mutex-serialized so one preloaded batch cannot split between
/// callbacks. Application-order FIFO still requires exactly one logical
/// consumer; concurrent callers must not apply returned batches out of order.
pub type CommandDrain = Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync>;
pub type CommandSubmit = Arc<dyn Fn(ControlCommand) -> bool + Send + Sync>;

/// Create a bounded legacy control-command bus.
///
/// # Panics
///
/// Panics when `capacity` is zero, matching the Asupersync bounded-channel
/// contract.
pub fn create_command_bus(capacity: usize) -> (CommandSender, CommandReceiver) {
    let (sender, receiver) = mpsc::channel(capacity);
    let counters = Arc::new(CommandBusCounters::default());
    (
        CommandSender {
            inner: sender,
            counters: Arc::clone(&counters),
        },
        CommandReceiver {
            inner: Mutex::new(receiver),
            counters,
        },
    )
}

#[must_use]
pub fn drain_pending_commands(receiver: &CommandReceiver) -> Vec<ControlCommand> {
    let span = trace_span!(
        "control_command_bus.drain",
        channel_id = COMMAND_BUS_TELEMETRY_ID
    );
    let _entered = span.enter();
    let commands = receiver.drain_bounded();
    if tracing::enabled!(tracing::Level::TRACE) {
        let telemetry = receiver.telemetry_snapshot();
        trace!(
            drained_commands = commands.len(),
            queued_commands = telemetry.queued_commands,
            capacity = telemetry.capacity,
            receiver_health = telemetry.receiver_health,
            "control command drain completed"
        );
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
            sender.record_validation_rejection();
            warn!(%error, "rejected invalid control command before queue admission");
            return false;
        }
        match sender.try_send(command) {
            Ok(()) => true,
            Err(CommandSendError::Full(_command)) => {
                warn!("control command queue full; command rejected");
                false
            }
            Err(CommandSendError::Disconnected(_command)) => {
                warn!("control command queue disconnected; command rejected");
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
        let telemetry = sender.clone();
        let submit = make_command_submit(sender);
        assert!(!(submit)(ControlCommand::UpdateSimulation(
            SimulationCommand {
                paused: Some(false),
                speed_multiplier: Some(f32::NAN),
                step_once: false,
            }
        )));
        assert!(matches!(receiver.try_recv(), Err(CommandRecvError::Empty)));
        assert_eq!(telemetry.telemetry_snapshot().validation_rejections, 1);
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
    use std::sync::{Arc, Barrier, Mutex};

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
        let (sender, receiver) = create_command_bus(32);
        for index in 0..32 {
            let result = sender.try_send(ControlCommand::UpdateSimulation(
                SimulationCommand::default(),
            ));
            assert!(result.is_ok(), "command {index} should fit: {result:?}");
        }
        let overflow = ControlCommand::UpdateSimulation(SimulationCommand {
            paused: Some(true),
            speed_multiplier: Some(33.0),
            step_once: true,
        });
        let error = sender
            .try_send(overflow)
            .expect_err("the thirty-third command must be returned");
        let recovered = match error {
            CommandSendError::Full(ControlCommand::UpdateSimulation(recovered)) => recovered,
            other => panic!("expected exact full playback command, got {other:?}"),
        };
        assert_eq!(recovered.speed_multiplier, Some(33.0));
        assert!(recovered.step_once);
        let telemetry = sender.telemetry_snapshot();
        assert_eq!(telemetry.admitted_commands, 32);
        assert_eq!(telemetry.full_rejections, 1);
        assert_eq!(drain_pending_commands(&receiver).len(), 32);
    }

    #[test]
    #[should_panic(expected = "channel capacity must be non-zero")]
    fn zero_capacity_is_rejected_at_bus_construction() {
        let _ = create_command_bus(0);
    }

    #[test]
    fn telemetry_reports_exact_bounded_depth_and_drain_progress() {
        let (sender, receiver) = create_command_bus(2);
        assert_eq!(sender.capacity(), 2);
        assert_eq!(sender.telemetry_snapshot().queued_commands, 0);

        sender
            .try_send(ControlCommand::UpdateSimulation(SimulationCommand {
                paused: Some(true),
                speed_multiplier: None,
                step_once: false,
            }))
            .expect("pause fits");
        sender
            .try_send(ControlCommand::UpdateSimulation(SimulationCommand {
                paused: None,
                speed_multiplier: Some(2.0),
                step_once: false,
            }))
            .expect("speed fits");

        let full = sender.telemetry_snapshot();
        assert_eq!(full.capacity, 2);
        assert_eq!(full.queued_commands, 2);
        assert_eq!(full.reserved_commands, 0);
        assert_eq!(full.admitted_commands, 2);

        let drained = drain_pending_commands(&receiver);
        assert_eq!(drained.len(), 2);
        let empty = receiver.telemetry_snapshot();
        assert_eq!(empty.queued_commands, 0);
        assert_eq!(empty.drained_commands, 2);
    }

    #[test]
    fn dropped_reservation_aborts_without_a_phantom_command() {
        let (sender, receiver) = create_command_bus(1);
        let permit = sender.try_reserve().expect("one slot can be reserved");
        let reserved = permit.telemetry_snapshot(COMMAND_BUS_TELEMETRY_ID);
        assert_eq!(reserved.queued_messages, 0);
        assert_eq!(reserved.reserved_uncommitted_obligations, 1);

        drop(permit);

        let aborted = sender.telemetry_snapshot();
        assert_eq!(aborted.queued_commands, 0);
        assert_eq!(aborted.reserved_commands, 0);
        assert_eq!(aborted.cancellation_count, 1);
        assert!(matches!(receiver.try_recv(), Err(CommandRecvError::Empty)));
        sender
            .try_send(ControlCommand::UpdateSimulation(
                SimulationCommand::default(),
            ))
            .expect("aborted reservation released capacity");
    }

    #[test]
    fn reserved_commit_after_receiver_drop_recovers_exact_command() {
        let (sender, receiver) = create_command_bus(1);
        let permit = sender.try_reserve().expect("one slot can be reserved");
        drop(receiver);
        let command = ControlCommand::UpdateSimulation(SimulationCommand {
            paused: Some(true),
            speed_multiplier: Some(3.0),
            step_once: true,
        });

        let error = permit
            .try_send(command)
            .expect_err("closed receiver must reject reserved commit");
        let recovered = match error {
            mpsc::SendError::Disconnected(ControlCommand::UpdateSimulation(recovered)) => recovered,
            other => panic!("expected exact disconnected playback command, got {other:?}"),
        };
        assert_eq!(
            recovered,
            SimulationCommand {
                paused: Some(true),
                speed_multiplier: Some(3.0),
                step_once: true,
            }
        );
    }

    #[test]
    fn receiver_drop_returns_exact_command_and_counts_disconnect() {
        let (sender, receiver) = create_command_bus(1);
        drop(receiver);
        let error = sender
            .try_send(ControlCommand::UpdateSimulation(SimulationCommand {
                paused: Some(false),
                speed_multiplier: Some(7.0),
                step_once: true,
            }))
            .expect_err("closed receiver rejects command");
        let recovered = match error {
            CommandSendError::Disconnected(ControlCommand::UpdateSimulation(recovered)) => {
                recovered
            }
            other => panic!("expected exact disconnected playback command, got {other:?}"),
        };
        assert_eq!(recovered.speed_multiplier, Some(7.0));
        assert!(recovered.step_once);
        let telemetry = sender.telemetry_snapshot();
        assert_eq!(telemetry.admitted_commands, 0);
        assert_eq!(telemetry.disconnected_rejections, 1);
        assert!(telemetry.closed);
    }

    #[test]
    fn dropping_last_sender_drains_fifo_then_reports_typed_disconnect() {
        let (sender, receiver) = create_command_bus(2);
        sender
            .try_send(ControlCommand::UpdateSimulation(SimulationCommand {
                paused: Some(true),
                speed_multiplier: None,
                step_once: false,
            }))
            .expect("pause fits");
        sender
            .try_send(ControlCommand::UpdateSimulation(SimulationCommand {
                paused: None,
                speed_multiplier: Some(4.0),
                step_once: false,
            }))
            .expect("speed fits");
        drop(sender);

        let drained = drain_pending_commands(&receiver);
        assert_eq!(drained.len(), 2);
        let ControlCommand::UpdateSimulation(first) = &drained[0] else {
            panic!("first command should be playback");
        };
        let ControlCommand::UpdateSimulation(second) = &drained[1] else {
            panic!("second command should be playback");
        };
        assert_eq!(first.paused, Some(true));
        assert_eq!(second.speed_multiplier, Some(4.0));
        assert!(matches!(
            receiver.try_recv(),
            Err(CommandRecvError::Disconnected)
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tokio_producer_uses_sync_admission_without_runtime_coupling() {
        let (sender, receiver) = create_command_bus(1);
        tokio::spawn(async move {
            sender.try_send(ControlCommand::UpdateSimulation(SimulationCommand {
                paused: Some(true),
                speed_multiplier: None,
                step_once: false,
            }))
        })
        .await
        .expect("tokio producer task")
        .expect("sync try_send from tokio worker");

        let ControlCommand::UpdateSimulation(received) = receiver
            .try_recv()
            .expect("plain-thread consumer receives command")
        else {
            panic!("expected playback command");
        };
        assert_eq!(received.paused, Some(true));
    }

    #[test]
    fn concurrent_drain_callbacks_do_not_split_one_preloaded_batch() {
        const SPEEDS: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (sender, receiver) = create_command_bus(SPEEDS.len());
        for speed_multiplier in SPEEDS {
            sender
                .try_send(ControlCommand::UpdateSimulation(SimulationCommand {
                    paused: None,
                    speed_multiplier: Some(speed_multiplier),
                    step_once: false,
                }))
                .expect("batch command fits");
        }
        drop(sender);

        let drain = make_command_drain(receiver);
        let barrier = Arc::new(Barrier::new(3));
        let first_drain = Arc::clone(&drain);
        let first_barrier = Arc::clone(&barrier);
        let first = std::thread::spawn(move || {
            first_barrier.wait();
            first_drain()
        });
        let second_drain = Arc::clone(&drain);
        let second_barrier = Arc::clone(&barrier);
        let second = std::thread::spawn(move || {
            second_barrier.wait();
            second_drain()
        });
        barrier.wait();

        let batches = [
            first.join().expect("first drain callback"),
            second.join().expect("second drain callback"),
        ];
        assert_eq!(
            batches.iter().filter(|batch| !batch.is_empty()).count(),
            1,
            "one receiver lock owns the whole preloaded bounded batch"
        );
        let mut observed = batches
            .into_iter()
            .flatten()
            .map(|command| {
                let ControlCommand::UpdateSimulation(playback) = command else {
                    panic!("expected playback command");
                };
                playback
                    .speed_multiplier
                    .expect("scripted speed is present")
            })
            .collect::<Vec<_>>();
        observed.sort_by(f32::total_cmp);
        assert_eq!(observed, SPEEDS);
    }

    #[test]
    fn seeded_command_schedule_matches_direct_world_digest_each_tick() {
        const FOOD_MAX_SCHEDULE: &[&[f32]] = &[
            &[0.61, 0.47],
            &[],
            &[0.58],
            &[],
            &[0.52, 0.63],
        ];
        let config = ScriptBotsConfig::default();
        let mut direct = WorldState::new(config.clone()).expect("direct seeded world");
        let mut queued = WorldState::new(config).expect("queued seeded world");
        let (sender, receiver) = create_command_bus(4);

        for (boundary, food_max_values) in FOOD_MAX_SCHEDULE.iter().enumerate() {
            for &food_max in *food_max_values {
                let mut updated = direct.config().clone();
                updated.food_max = food_max;
                let command = ControlCommand::UpdateConfig(Box::new(updated));
                let direct_disposition = apply_control_command(&mut direct, command.clone())
                    .expect("direct scheduled command applies");
                assert!(matches!(
                    direct_disposition,
                    ControlDisposition::WorldApplied
                ));
                sender
                    .try_send(command)
                    .expect("scheduled command enters bounded bus");
            }

            for command in drain_pending_commands(&receiver) {
                let queued_disposition = apply_control_command(&mut queued, command)
                    .expect("queued scheduled command applies");
                assert!(matches!(
                    queued_disposition,
                    ControlDisposition::WorldApplied
                ));
            }

            direct.step().expect("direct seeded tick");
            queued.step().expect("queued seeded tick");
            assert_eq!(
                queued.world_digest_v1().expect("queued world digest"),
                direct.world_digest_v1().expect("direct world digest"),
                "command transport changed scientific state at boundary {boundary}"
            );
        }
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
