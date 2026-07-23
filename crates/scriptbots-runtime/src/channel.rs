//! Cross-thread channel-backed host port and sole-owner driver.
//!
//! [`HostCore`](crate::HostCore) is intentionally same-thread: its admission
//! state lives in `Rc<RefCell<SharedHostState>>`, so the authoritative
//! [`HostPort`](crate::HostPort) (`LocalHostPort`) cannot leave the owner
//! thread. Servers and other transports still need the full client contract —
//! synchronous admission, command-status lookup, snapshot subscriptions, and
//! scientific-event cursors — from their own threads.
//!
//! [`ChannelHostPort`] closes that gap without relaxing the ownership model:
//!
//! - The client port is `Send + Sync + Clone` and implements
//!   [`HostPort`](crate::HostPort). `submit` performs a bounded round-trip:
//!   the envelope enters a bounded ingress channel, the owner thread runs the
//!   exact same `SharedHostState::submit` admission path as
//!   [`LocalHostPort`](crate::LocalHostPort), and the authoritative
//!   [`CommandStatus`] (including stored pre-admission rejections) is returned
//!   to the caller. An admission that never reaches the owner is an error,
//!   never a fabricated status.
//! - Command statuses and ordered protocol events are mirrored into bounded,
//!   lock-guarded boards by the owner after every admission and drive
//!   boundary, so `command_status` and `events_after` never touch the host.
//! - Snapshot and scientific-event reads delegate to the thread-safe
//!   [`SnapshotHub`] and [`EventHub`] handles already published by the host.
//! - [`ChannelHostDriver`] owns the [`FixedDeadlineHost`] on the owner thread:
//!   it drains ingress, drives at fixed cadence deadlines with bounded
//!   catch-up, parks without a periodic timer while the world is quiescent,
//!   converts a full client disconnect into one ordered shutdown, and returns
//!   an explicit receipt when the host lifecycle terminates.

use crate::{
    CommandEnvelope, CommandId, CommandStatus, EventCatchUp, EventCatchUpLocator, EventCursor,
    EventHub, EventPoll, FixedDeadlineHost, HostAccessError, HostDriveInterest, HostEvent,
    HostPort, HostSessionId, LocalHostPort, ManualInstant, NativeDriveTrigger, NativeScheduleError,
    ProtocolEventSequence, RenderSnapshot, SnapshotHub, SnapshotRevision,
};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use thiserror::Error;

/// Default bound for the pre-admission ingress channel.
pub const DEFAULT_CHANNEL_INGRESS_CAPACITY: usize = 64;
/// Default bound for mirrored command statuses and protocol events.
pub const DEFAULT_CHANNEL_BOARD_CAPACITY: usize = 4_096;
/// Default worst-case wait for ingress space or an admission reply.
pub const DEFAULT_CHANNEL_SUBMIT_DEADLINE: Duration = Duration::from_millis(2_000);
/// Default polling cadence while journal or shutdown work drains.
pub const DEFAULT_CHANNEL_MAINTENANCE_PERIOD: Duration = Duration::from_millis(20);

/// Bounds and deadlines for one channel host driver and its ports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelHostOptions {
    /// Bounded pre-admission ingress channel capacity.
    pub ingress_capacity: usize,
    /// Maximum mirrored command statuses retained for cross-thread lookup.
    pub status_board_capacity: usize,
    /// Maximum mirrored ordered protocol events retained for cross-thread catch-up.
    pub protocol_event_capacity: usize,
    /// Worst-case wait for ingress space or an authoritative admission reply.
    pub submit_deadline: Duration,
    /// Polling cadence while journal or shutdown work drains.
    pub maintenance_period: Duration,
}

impl Default for ChannelHostOptions {
    fn default() -> Self {
        Self {
            ingress_capacity: DEFAULT_CHANNEL_INGRESS_CAPACITY,
            status_board_capacity: DEFAULT_CHANNEL_BOARD_CAPACITY,
            protocol_event_capacity: DEFAULT_CHANNEL_BOARD_CAPACITY,
            submit_deadline: DEFAULT_CHANNEL_SUBMIT_DEADLINE,
            maintenance_period: DEFAULT_CHANNEL_MAINTENANCE_PERIOD,
        }
    }
}

impl ChannelHostOptions {
    const fn validate(self) -> Result<Self, ChannelHostOptionsError> {
        if self.ingress_capacity == 0 {
            return Err(ChannelHostOptionsError::EmptyIngress);
        }
        if self.status_board_capacity == 0 {
            return Err(ChannelHostOptionsError::EmptyStatusBoard);
        }
        if self.protocol_event_capacity == 0 {
            return Err(ChannelHostOptionsError::EmptyProtocolEventBoard);
        }
        if self.submit_deadline.is_zero() {
            return Err(ChannelHostOptionsError::MissingSubmitDeadline);
        }
        if self.maintenance_period.is_zero() {
            return Err(ChannelHostOptionsError::MissingMaintenancePeriod);
        }
        Ok(self)
    }
}

/// Invalid bound supplied to [`ChannelHostDriver::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ChannelHostOptionsError {
    /// Bounded ingress must retain at least one envelope.
    #[error("channel ingress_capacity must be nonzero")]
    EmptyIngress,
    /// The mirrored status board must retain at least one status.
    #[error("channel status_board_capacity must be nonzero")]
    EmptyStatusBoard,
    /// The mirrored protocol-event board must retain at least one event.
    #[error("channel protocol_event_capacity must be nonzero")]
    EmptyProtocolEventBoard,
    /// Admission round-trips must have a positive deadline.
    #[error("channel submit_deadline must be nonzero")]
    MissingSubmitDeadline,
    /// Drain polling must have a positive period.
    #[error("channel maintenance_period must be nonzero")]
    MissingMaintenancePeriod,
}

/// One pre-admission message offered to the owner thread.
enum IngressMessage {
    /// A command envelope plus the one-shot admission reply lane.
    Command {
        /// Exact client envelope forwarded untouched to host admission.
        envelope: CommandEnvelope,
        /// Single-use reply lane carrying the authoritative admission status.
        reply: Sender<Result<CommandStatus, HostAccessError>>,
    },
    /// A coalesced non-command observation request.
    Wake,
}

/// Bounded mirror of host command statuses for cross-thread lookup.
///
/// Insertion order is retained so eviction can prefer terminal statuses:
/// a finished command (application resolved and journal axis settled) is
/// evicted before any still-admitted command, and only when every retained
/// status is unfinished does the oldest unfinished status yield.
#[derive(Debug, Default)]
struct StatusBoard {
    order: VecDeque<CommandId>,
    statuses: HashMap<CommandId, CommandStatus>,
}

impl StatusBoard {
    fn insert(&mut self, status: CommandStatus, capacity: usize) {
        let command_id = status.command_id();
        if self.statuses.insert(command_id, status).is_none() {
            self.order.push_back(command_id);
        }
        while self.statuses.len() > capacity {
            let evict = self
                .order
                .iter()
                .position(|id| {
                    self.statuses
                        .get(id)
                        .is_some_and(|status| status_is_finished(status))
                })
                .unwrap_or(0);
            let Some(removed) = self.order.remove(evict) else {
                break;
            };
            self.statuses.remove(&removed);
        }
    }

    fn get(&self, command_id: CommandId) -> Option<CommandStatus> {
        self.statuses.get(&command_id).cloned()
    }
}

/// Whether both axes of a command status reached a terminal state.
fn status_is_finished(status: &CommandStatus) -> bool {
    let application_finished = !matches!(status.application(), crate::ApplicationState::Admitted);
    let journal_finished = !matches!(status.journal(), crate::JournalState::Pending);
    application_finished && journal_finished
}

/// Bounded mirror of the ordered protocol-event ring.
#[derive(Debug, Default)]
struct ProtocolEventBoard {
    events: VecDeque<HostEvent>,
}

impl ProtocolEventBoard {
    fn push(&mut self, event: HostEvent, capacity: usize) {
        self.events.push_back(event);
        while self.events.len() > capacity {
            self.events.pop_front();
        }
    }

    fn after(&self, cursor: ProtocolEventSequence, limit: usize) -> Vec<HostEvent> {
        self.events
            .iter()
            .filter(|event| event.sequence > cursor)
            .take(limit)
            .cloned()
            .collect()
    }
}

/// Cloneable, thread-safe client port for a sole-owner host driven elsewhere.
///
/// Successful submission means the owner thread admitted (or truthfully
/// rejected) the exact envelope through the same ordered path as
/// [`LocalHostPort`]; it is not merely a channel enqueue. Status, snapshot,
/// and event reads never block the owner.
pub struct ChannelHostPort {
    sender: SyncSender<IngressMessage>,
    statuses: Arc<RwLock<StatusBoard>>,
    protocol_events: Arc<RwLock<ProtocolEventBoard>>,
    snapshots: SnapshotHub,
    events: EventHub,
    session_id: HostSessionId,
    submit_deadline: Duration,
}

impl Clone for ChannelHostPort {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            statuses: Arc::clone(&self.statuses),
            protocol_events: Arc::clone(&self.protocol_events),
            snapshots: self.snapshots.clone(),
            events: self.events.clone(),
            session_id: self.session_id,
            submit_deadline: self.submit_deadline,
        }
    }
}

impl ChannelHostPort {
    fn protocol_violation(message: impl Into<String>) -> HostAccessError {
        HostAccessError::ProtocolViolation {
            message: message.into(),
        }
    }
}

impl HostPort for ChannelHostPort {
    fn session_id(&self) -> HostSessionId {
        self.session_id
    }

    fn submit(&mut self, envelope: CommandEnvelope) -> Result<CommandStatus, HostAccessError> {
        let (reply, reply_rx) = std::sync::mpsc::channel();
        let message = IngressMessage::Command { envelope, reply };
        match self.sender.try_send(message) {
            Ok(()) => {}
            Err(TrySendError::Full(message)) => {
                // `SyncSender::send_timeout` is unstable; park briefly between
                // retries inside the configured deadline instead of blocking
                // without bound. The owner drains ingress every boundary, so
                // a full channel for the whole deadline means it is wedged.
                const RETRY_PARK: Duration = Duration::from_millis(2);
                let started = std::time::Instant::now();
                let mut pending = message;
                loop {
                    match self.sender.try_send(pending) {
                        Ok(()) => break,
                        Err(TrySendError::Full(returned)) => {
                            if started.elapsed() >= self.submit_deadline {
                                return Err(Self::protocol_violation(format!(
                                    "channel host did not drain ingress within {:?}",
                                    self.submit_deadline
                                )));
                            }
                            pending = returned;
                            std::thread::park_timeout(RETRY_PARK);
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            return Err(HostAccessError::Disconnected);
                        }
                    }
                }
            }
            Err(TrySendError::Disconnected(_)) => return Err(HostAccessError::Disconnected),
        }
        reply_rx
            .recv_timeout(self.submit_deadline)
            .map_err(|error| match error {
                RecvTimeoutError::Timeout => Self::protocol_violation(format!(
                    "channel host admission reply exceeded {:?}",
                    self.submit_deadline
                )),
                RecvTimeoutError::Disconnected => HostAccessError::Disconnected,
            })?
    }

    fn command_status(
        &mut self,
        command_id: CommandId,
    ) -> Result<Option<CommandStatus>, HostAccessError> {
        let board = self
            .statuses
            .read()
            .map_err(|_| Self::protocol_violation("channel status board poisoned"))?;
        Ok(board.get(command_id))
    }

    fn snapshot_after(
        &mut self,
        after: Option<SnapshotRevision>,
    ) -> Result<Option<Arc<RenderSnapshot>>, HostAccessError> {
        Ok(self.snapshots.snapshot_after(after))
    }

    fn events_after(
        &mut self,
        cursor: ProtocolEventSequence,
        limit: usize,
    ) -> Result<Vec<HostEvent>, HostAccessError> {
        let board = self
            .protocol_events
            .read()
            .map_err(|_| Self::protocol_violation("channel protocol-event board poisoned"))?;
        Ok(board.after(cursor, limit))
    }

    fn poll_events(
        &mut self,
        cursor: EventCursor,
        limit: usize,
    ) -> Result<EventPoll, HostAccessError> {
        self.events.poll(cursor, limit)
    }

    fn catch_up_events(
        &mut self,
        locator: EventCatchUpLocator,
        limit: usize,
    ) -> Result<EventCatchUp, HostAccessError> {
        self.events.catch_up(locator, limit)
    }
}

/// Why one driver iteration woke the owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelStepReport {
    /// Command envelopes admitted during this iteration.
    pub admitted: usize,
    /// Whether the host advanced one drive boundary.
    pub drove: bool,
    /// Scheduling interest observed after the boundary.
    pub interest: HostDriveInterest,
}

/// Terminal classification of one channel driver run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelRunOutcome {
    /// An explicit ordered shutdown completed (host lifecycle stopped).
    Stopped,
    /// Losing every client producer triggered the fail-safe ordered shutdown.
    ControllerDisconnected,
    /// The host latched a fault before an ordered shutdown completed.
    Faulted,
}

/// Exact accounting returned when the driver exits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelRunReceipt {
    /// Why the run ended.
    pub outcome: ChannelRunOutcome,
    /// Command envelopes admitted by the owner.
    pub commands_admitted: u64,
    /// Host drive boundaries completed.
    pub drives: u64,
}

/// Failure advancing one driver boundary.
#[derive(Debug, Error)]
pub enum ChannelDriveError {
    /// Host admission or shutdown failed.
    #[error(transparent)]
    Access(#[from] HostAccessError),
    /// The fixed-deadline schedule failed.
    #[error(transparent)]
    Schedule(#[from] NativeScheduleError),
}

/// Sole-owner driver for a [`FixedDeadlineHost`] serving [`ChannelHostPort`]s.
///
/// The driver is deliberately not `Send`: it owns the same-thread host and
/// must be constructed and run on the thread that owns the world. Clients
/// receive [`ChannelHostPort`] handles from [`Self::new`].
pub struct ChannelHostDriver {
    host: FixedDeadlineHost,
    port: LocalHostPort,
    receiver: Receiver<IngressMessage>,
    statuses: Arc<RwLock<StatusBoard>>,
    protocol_events: Arc<RwLock<ProtocolEventBoard>>,
    tracked: VecDeque<CommandId>,
    last_protocol_event: ProtocolEventSequence,
    status_board_capacity: usize,
    protocol_event_capacity: usize,
    maintenance_period: Duration,
    controller_disconnected: bool,
    shutdown_requested: bool,
}

impl ChannelHostDriver {
    /// Pair one owner-side driver with its first cross-thread client port.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelHostOptionsError`] when any bound is zero.
    pub fn new(
        host: FixedDeadlineHost,
        options: ChannelHostOptions,
    ) -> Result<(Self, ChannelHostPort), ChannelHostOptionsError> {
        let options = options.validate()?;
        let (sender, receiver) = std::sync::mpsc::sync_channel(options.ingress_capacity);
        let statuses = Arc::new(RwLock::new(StatusBoard::default()));
        let protocol_events = Arc::new(RwLock::new(ProtocolEventBoard::default()));
        let local = host.local_port();
        let port = ChannelHostPort {
            sender,
            statuses: Arc::clone(&statuses),
            protocol_events: Arc::clone(&protocol_events),
            snapshots: host.snapshot_hub(),
            events: host.event_hub(),
            session_id: local.session_id(),
            submit_deadline: options.submit_deadline,
        };
        let last_protocol_event = ProtocolEventSequence::new(0);
        let driver = Self {
            host,
            port: local,
            receiver,
            statuses,
            protocol_events,
            tracked: VecDeque::new(),
            last_protocol_event,
            status_board_capacity: options.status_board_capacity,
            protocol_event_capacity: options.protocol_event_capacity,
            maintenance_period: options.maintenance_period,
            controller_disconnected: false,
            shutdown_requested: false,
        };
        Ok((driver, port))
    }

    fn track(&mut self, command_id: CommandId) {
        if self.tracked.contains(&command_id) {
            return;
        }
        self.tracked.push_back(command_id);
        while self.tracked.len() > self.status_board_capacity {
            self.tracked.pop_front();
        }
    }

    fn mirror_one(&mut self, command_id: CommandId) {
        let Ok(Some(status)) = self.port.command_status(command_id) else {
            return;
        };
        if let Ok(mut board) = self.statuses.write() {
            board.insert(status, self.status_board_capacity);
        }
    }

    fn mirror_tracked(&mut self) {
        for index in 0..self.tracked.len() {
            let Some(command_id) = self.tracked.get(index).copied() else {
                continue;
            };
            self.mirror_one(command_id);
        }
    }

    fn mirror_protocol_events(&mut self) {
        let Ok(events) = self.port.events_after(self.last_protocol_event, usize::MAX) else {
            return;
        };
        if let Some(last) = events.last() {
            self.last_protocol_event = last.sequence;
        }
        if let Ok(mut board) = self.protocol_events.write() {
            for event in events {
                board.push(event, self.protocol_event_capacity);
            }
        }
    }

    fn drain_ingress(&mut self) -> Result<usize, ChannelDriveError> {
        let mut admitted = 0;
        loop {
            match self.receiver.try_recv() {
                Ok(IngressMessage::Command { envelope, reply }) => {
                    let command_id = envelope.command_id;
                    let result = self.host.submit(envelope);
                    if let Ok(status) = &result {
                        self.track(command_id);
                        if let Ok(mut board) = self.statuses.write() {
                            board.insert(status.clone(), self.status_board_capacity);
                        }
                    }
                    // A client that timed out is unreachable; the admission
                    // remains authoritative and mirrored either way.
                    let _ = reply.send(result);
                    admitted += 1;
                }
                Ok(IngressMessage::Wake) => {}
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.controller_disconnected = true;
                    break;
                }
            }
        }
        Ok(admitted)
    }

    /// Drain ingress, drive the host when due, and mirror all boards once.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelDriveError`] when admission or the schedule fails.
    pub fn step(&mut self, now: ManualInstant) -> Result<ChannelStepReport, ChannelDriveError> {
        let admitted = self.drain_ingress()?;
        let mut interest = self.host.drive_interest();
        let mut drove = false;
        match interest {
            HostDriveInterest::ReadyNow => {
                self.host.drive_at(now, NativeDriveTrigger::Command)?;
                drove = true;
            }
            HostDriveInterest::Deadline => {
                let due = self
                    .host
                    .next_deadline()
                    .is_none_or(|deadline| now >= deadline);
                if due {
                    self.host.drive_at(now, NativeDriveTrigger::Deadline)?;
                    drove = true;
                }
            }
            HostDriveInterest::Draining => {
                self.host.drive_at(now, NativeDriveTrigger::Maintenance)?;
                drove = true;
            }
            HostDriveInterest::WakeOnly
            | HostDriveInterest::Terminated
            | HostDriveInterest::Faulted => {}
        }
        if self.controller_disconnected && !self.shutdown_requested {
            let _ = self.host.request_shutdown()?;
            self.shutdown_requested = true;
        }
        self.mirror_tracked();
        self.mirror_protocol_events();
        interest = self.host.drive_interest();
        Ok(ChannelStepReport {
            admitted,
            drove,
            interest,
        })
    }

    /// Drive the host to its terminal lifecycle and return exact accounting.
    ///
    /// The loop parks without a periodic timer while the host reports
    /// [`HostDriveInterest::WakeOnly`]. A full client disconnect converts into
    /// one ordered shutdown rather than abandoning the world.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelDriveError`] when a boundary cannot advance.
    pub fn run(
        mut self,
        mut now: impl FnMut() -> ManualInstant,
    ) -> Result<ChannelRunReceipt, ChannelDriveError> {
        let mut drives = 0_u64;
        let mut commands_admitted = 0_u64;
        loop {
            let report = self.step(now())?;
            commands_admitted = commands_admitted.saturating_add(report.admitted as u64);
            drives = drives.saturating_add(u64::from(report.drove));
            match report.interest {
                HostDriveInterest::Terminated => {
                    return Ok(ChannelRunReceipt {
                        outcome: if self.controller_disconnected {
                            ChannelRunOutcome::ControllerDisconnected
                        } else {
                            ChannelRunOutcome::Stopped
                        },
                        commands_admitted,
                        drives,
                    });
                }
                HostDriveInterest::Faulted => {
                    return Ok(ChannelRunReceipt {
                        outcome: ChannelRunOutcome::Faulted,
                        commands_admitted,
                        drives,
                    });
                }
                HostDriveInterest::WakeOnly => {
                    // Quiescent: park until a client wakes us; no periodic timer.
                    if self.receiver.recv().is_err() {
                        self.controller_disconnected = true;
                    }
                }
                HostDriveInterest::Draining => {
                    if matches!(
                        self.receiver.recv_timeout(self.maintenance_period),
                        Err(RecvTimeoutError::Disconnected)
                    ) {
                        self.controller_disconnected = true;
                    }
                }
                HostDriveInterest::ReadyNow | HostDriveInterest::Deadline => {
                    let wait = self
                        .host
                        .next_deadline()
                        .map(|deadline| {
                            let current = now();
                            if deadline > current {
                                Duration::from_nanos(deadline.as_nanos() - current.as_nanos())
                            } else {
                                Duration::ZERO
                            }
                        })
                        .unwrap_or(self.maintenance_period)
                        .min(self.maintenance_period.max(Duration::from_millis(1)));
                    if matches!(
                        self.receiver.recv_timeout(wait),
                        Err(RecvTimeoutError::Disconnected)
                    ) {
                        self.controller_disconnected = true;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ApplicationState, HostCommand, HostCore, HostCoreOptions, HostLifecycle, PlaybackSnapshot,
        RejectionReason,
    };
    use scriptbots_core::ScriptBotsConfig;

    fn test_host(paused: bool) -> FixedDeadlineHost {
        let world = scriptbots_core::WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0x5eed_cafe),
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .expect("deterministic test world");
        let options = HostCoreOptions {
            initial_playback: PlaybackSnapshot {
                paused,
                ..PlaybackSnapshot::default()
            },
            tick_period_nanos: 1_000_000_000,
            snapshot_interval_ticks: 1,
            ..HostCoreOptions::default()
        };
        let core = HostCore::new(HostSessionId::new(7), world, options).expect("host core builds");
        FixedDeadlineHost::new(core)
    }

    fn fast_options() -> ChannelHostOptions {
        ChannelHostOptions {
            submit_deadline: Duration::from_millis(500),
            maintenance_period: Duration::from_millis(2),
            ..ChannelHostOptions::default()
        }
    }

    /// Spawn the !Send driver on its owner thread and hand the port back.
    /// This is the exact production deployment shape: the host is constructed
    /// and driven on its own thread; only the cross-thread port travels.
    fn spawn_driver(
        paused: bool,
    ) -> (
        std::thread::JoinHandle<Result<ChannelRunReceipt, ChannelDriveError>>,
        ChannelHostPort,
    ) {
        let (port_tx, port_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let (driver, port) =
                ChannelHostDriver::new(test_host(paused), fast_options()).expect("driver");
            port_tx.send(port).expect("port handoff");
            let start = std::time::Instant::now();
            driver.run(move || {
                ManualInstant::from_nanos(1_000_000 + start.elapsed().as_nanos() as u64)
            })
        });
        let port = port_rx.recv().expect("port handoff");
        (worker, port)
    }

    /// Wait until a command reaches a terminal application state.
    fn wait_resolved(port: &mut ChannelHostPort, command_id: CommandId) -> CommandStatus {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = port.command_status(command_id).expect("status lookup") {
                if !matches!(status.application(), ApplicationState::Admitted) {
                    return status;
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "command never resolved"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn shutdown_and_join(
        worker: std::thread::JoinHandle<Result<ChannelRunReceipt, ChannelDriveError>>,
        port: &mut ChannelHostPort,
    ) -> ChannelRunReceipt {
        port.submit(CommandEnvelope::new(
            CommandId::new(999_999),
            HostCommand::Shutdown,
        ))
        .expect("shutdown admission");
        worker.join().expect("driver thread").expect("clean run")
    }

    #[test]
    fn pause_step_shutdown_round_trip_across_threads() {
        let (worker, mut port) = spawn_driver(false);

        let pause = port
            .submit(CommandEnvelope::new(CommandId::new(11), HostCommand::Pause))
            .expect("pause admission");
        assert!(matches!(pause.application(), ApplicationState::Admitted));
        let applied = wait_resolved(&mut port, CommandId::new(11));
        assert!(matches!(
            applied.application(),
            ApplicationState::Applied(_)
        ));

        let step = port
            .submit(CommandEnvelope::new(CommandId::new(12), HostCommand::Step))
            .expect("step admission");
        assert!(matches!(step.application(), ApplicationState::Admitted));
        let stepped = wait_resolved(&mut port, CommandId::new(12));
        assert!(matches!(
            stepped.application(),
            ApplicationState::Applied(_)
        ));

        let receipt = shutdown_and_join(worker, &mut port);
        assert_eq!(receipt.outcome, ChannelRunOutcome::Stopped);
        assert!(receipt.commands_admitted >= 3);

        // The mirrored board survives the driver's exit.
        let status = port
            .command_status(CommandId::new(999_999))
            .expect("post-exit status")
            .expect("shutdown status retained");
        assert!(matches!(status.application(), ApplicationState::Applied(_)));
    }

    #[test]
    fn dedup_preserves_one_admission() {
        let (worker, mut port) = spawn_driver(false);
        let envelope = CommandEnvelope::new(CommandId::new(21), HostCommand::Pause);
        let first = port.submit(envelope.clone()).expect("first submit");
        let second = port.submit(envelope).expect("retry submit");
        assert_eq!(first.command_id(), second.command_id());
        assert_eq!(first.admission_sequence(), second.admission_sequence());
        let _ = wait_resolved(&mut port, CommandId::new(21));
        let receipt = shutdown_and_join(worker, &mut port);
        assert_eq!(receipt.outcome, ChannelRunOutcome::Stopped);
    }

    #[test]
    fn cas_conflict_retains_admission_sequence() {
        let (worker, mut port) = spawn_driver(false);
        let snapshot = port
            .snapshot_after(None)
            .expect("snapshot")
            .expect("initial snapshot");
        let wrong = snapshot.revisions.control.checked_next().expect("next");
        let envelope = CommandEnvelope::new(
            CommandId::new(31),
            HostCommand::UpdateConfig(Box::new(ScriptBotsConfig::default())),
        )
        .expecting_control_revision(wrong);
        port.submit(envelope).expect("submit");
        let resolved = wait_resolved(&mut port, CommandId::new(31));
        match resolved.application() {
            ApplicationState::Rejected(RejectionReason::ControlRevisionConflict { .. }) => {}
            other => panic!("expected control-revision conflict, got {other:?}"),
        }
        assert!(resolved.admission_sequence().is_some());
        let receipt = shutdown_and_join(worker, &mut port);
        assert_eq!(receipt.outcome, ChannelRunOutcome::Stopped);
    }

    #[test]
    fn validation_rejection_is_pre_admission() {
        let (worker, mut port) = spawn_driver(false);
        let status = port
            .submit(CommandEnvelope::new(
                CommandId::new(41),
                HostCommand::SetSpeed(-1.0),
            ))
            .expect("submit");
        match status.application() {
            ApplicationState::Rejected(RejectionReason::Validation { .. }) => {}
            other => panic!("expected validation rejection, got {other:?}"),
        }
        assert!(status.admission_sequence().is_none());
        let receipt = shutdown_and_join(worker, &mut port);
        assert_eq!(receipt.outcome, ChannelRunOutcome::Stopped);
    }

    #[test]
    fn update_config_applies_and_advances_revisions() {
        let (worker, mut port) = spawn_driver(false);
        let before = port
            .snapshot_after(None)
            .expect("snapshot")
            .expect("initial snapshot");
        port.submit(CommandEnvelope::new(
            CommandId::new(51),
            HostCommand::UpdateConfig(Box::new(ScriptBotsConfig {
                rng_seed: Some(0x5eed_cafe),
                ..ScriptBotsConfig::default()
            })),
        ))
        .expect("submit");
        let applied = wait_resolved(&mut port, CommandId::new(51));
        assert!(matches!(
            applied.application(),
            ApplicationState::Applied(_)
        ));
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let after = loop {
            if let Some(snapshot) = port
                .snapshot_after(Some(before.revision))
                .expect("snapshot")
            {
                break snapshot;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no snapshot published after config application"
            );
            std::thread::sleep(Duration::from_millis(1));
        };
        assert!(after.revisions.config > before.revisions.config);
        assert!(matches!(after.lifecycle, HostLifecycle::Running));
        let receipt = shutdown_and_join(worker, &mut port);
        assert_eq!(receipt.outcome, ChannelRunOutcome::Stopped);
    }

    #[test]
    fn step_command_applies_through_control_lane() {
        let (worker, mut port) = spawn_driver(false);
        let status = port
            .submit(CommandEnvelope::new(CommandId::new(81), HostCommand::Step))
            .expect("submit");
        assert!(matches!(status.application(), ApplicationState::Admitted));
        let resolved = wait_resolved(&mut port, CommandId::new(81));
        assert!(matches!(
            resolved.application(),
            ApplicationState::Applied(_)
        ));
        let receipt = shutdown_and_join(worker, &mut port);
        assert_eq!(receipt.outcome, ChannelRunOutcome::Stopped);
    }

    #[test]
    fn controller_disconnect_requests_ordered_shutdown() {
        let (driver, port) =
            ChannelHostDriver::new(test_host(false), fast_options()).expect("driver");
        drop(port);
        let start = std::time::Instant::now();
        let receipt = driver
            .run(move || ManualInstant::from_nanos(1_000_000 + start.elapsed().as_nanos() as u64))
            .expect("run");
        assert_eq!(receipt.outcome, ChannelRunOutcome::ControllerDisconnected);
    }

    #[test]
    fn ingress_flood_times_out_truthfully_when_owner_stalled() {
        let (_driver, mut port) = ChannelHostDriver::new(
            test_host(false),
            ChannelHostOptions {
                ingress_capacity: 1,
                submit_deadline: Duration::from_millis(25),
                ..fast_options()
            },
        )
        .expect("driver");
        // Owner never runs: a wake occupies the only ingress slot.
        let _ = port.sender.try_send(IngressMessage::Wake);
        let error = port
            .submit(CommandEnvelope::new(CommandId::new(61), HostCommand::Pause))
            .expect_err("stalled owner must fail truthfully");
        assert!(matches!(error, HostAccessError::ProtocolViolation { .. }));
    }

    #[test]
    fn status_board_evicts_finished_first() {
        let mut board = StatusBoard::default();
        let finished = CommandStatus::rejected(
            CommandId::new(71),
            RejectionReason::Validation {
                message: "x".to_owned(),
            },
        )
        .expect("valid rejection");
        board.insert(finished, 2);
        let admitted = CommandStatus::try_new(
            CommandId::new(72),
            Some(crate::AdmissionSequence::new(1)),
            ApplicationState::Admitted,
            crate::JournalState::NotRequired,
        )
        .expect("valid admitted");
        board.insert(admitted, 2);
        let admitted_two = CommandStatus::try_new(
            CommandId::new(73),
            Some(crate::AdmissionSequence::new(2)),
            ApplicationState::Admitted,
            crate::JournalState::NotRequired,
        )
        .expect("valid admitted");
        board.insert(admitted_two, 2);
        assert!(board.get(CommandId::new(71)).is_none());
        assert!(board.get(CommandId::new(72)).is_some());
        assert!(board.get(CommandId::new(73)).is_some());
    }

    #[test]
    fn host_accepts_caller_bound_persistence_session() {
        /// Minimal same-thread journal double for constructor tests.
        #[derive(Debug, Default)]
        struct NullTestJournal;

        impl crate::JournalPort for NullTestJournal {
            fn try_admit(
                &mut self,
                _batch: &std::sync::Arc<crate::JournalBatch>,
            ) -> crate::JournalAdmission {
                crate::JournalAdmission::Full {
                    batch_id: crate::JournalBatchId::default(),
                    capacity: 0,
                }
            }

            fn poll_receipts(&mut self, _limit: usize) -> Vec<crate::JournalReceipt> {
                Vec::new()
            }
        }

        let world = scriptbots_core::WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0x5eed_cafe),
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .expect("deterministic test world");
        let _persistence = world
            .bind_persistence(Box::new(scriptbots_core::NullPersistence))
            .expect("one-shot binding");
        let options = HostCoreOptions::default();
        let core = HostCore::with_journal(
            HostSessionId::new(9),
            world,
            options,
            Box::new(NullTestJournal),
        );
        if let Err(error) = &core {
            panic!("caller-bound session must construct: {error:?}");
        }
    }
}
