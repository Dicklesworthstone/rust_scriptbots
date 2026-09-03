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
//!   boundary. A status-board miss makes one bounded owner round-trip so an
//!   evicted id can still consult durable command authority; event reads stay
//!   mirror-only.
//! - Snapshot and scientific-event reads delegate to the thread-safe
//!   [`SnapshotHub`] and [`EventHub`] handles already published by the host.
//! - [`ChannelHostDriver`] owns the [`FixedDeadlineHost`] on the owner thread:
//!   it processes at most the configured ingress budget before each drive,
//!   drives at fixed cadence deadlines with bounded catch-up, parks without a
//!   periodic timer while the world is quiescent, converts a full client
//!   disconnect into one ordered shutdown, and returns an explicit receipt
//!   when the host lifecycle terminates.

use crate::{
    CommandEnvelope, CommandId, CommandStatus, EventCatchUp, EventCatchUpLocator, EventCursor,
    EventHub, EventPoll, FixedDeadlineHost, HostAccessError, HostDriveInterest, HostEvent,
    HostPort, HostSessionId, LocalHostPort, ManualInstant, NativeDriveTrigger, NativeScheduleError,
    ProtocolEventSequence, RenderSnapshot, SnapshotHub, SnapshotRevision,
};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Default bound for the pre-admission ingress channel.
pub const DEFAULT_CHANNEL_INGRESS_CAPACITY: usize = 64;
/// Default maximum ingress messages processed before one host drive boundary.
pub const DEFAULT_CHANNEL_INGRESS_DRAIN_BUDGET: usize = 64;
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
    /// Maximum ingress messages processed before one host drive boundary.
    pub ingress_drain_budget: usize,
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
            ingress_drain_budget: DEFAULT_CHANNEL_INGRESS_DRAIN_BUDGET,
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
        if self.ingress_drain_budget == 0 {
            return Err(ChannelHostOptionsError::EmptyIngressDrainBudget);
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
    /// Each boundary must process at least one ingress message.
    #[error("channel ingress_drain_budget must be nonzero")]
    EmptyIngressDrainBudget,
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
    /// A cache-miss status query that must consult the owner-side durable authority.
    CommandStatus {
        /// Stable command identity requested by the client.
        command_id: CommandId,
        /// Single-use reply carrying authoritative found/absent/fail-closed knowledge.
        reply: Sender<Result<Option<CommandStatus>, HostAccessError>>,
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
/// and event reads use bounded mirrors, except that a status-board miss makes
/// one bounded owner round-trip to durable command authority.
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

    fn command_authority_timeout(command_id: CommandId, waited: Duration) -> HostAccessError {
        HostAccessError::CommandAuthorityLookup {
            command_id,
            failure: crate::CommandAuthorityLookupFailure::Timeout { waited },
        }
    }

    fn enqueue(&self, message: IngressMessage, started: Instant) -> Result<(), HostAccessError> {
        // `SyncSender::send_timeout` is unstable; park briefly between retries inside the
        // caller's original deadline instead of blocking without bound or resetting the budget.
        const RETRY_PARK: Duration = Duration::from_millis(2);
        let mut pending = message;
        loop {
            let remaining = self.submit_deadline.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Err(Self::protocol_violation(format!(
                    "channel host did not drain ingress within {:?}",
                    self.submit_deadline
                )));
            }
            match self.sender.try_send(pending) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Full(returned)) => {
                    pending = returned;
                    std::thread::park_timeout(RETRY_PARK.min(remaining));
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(HostAccessError::Disconnected);
                }
            }
        }
    }
}

impl HostPort for ChannelHostPort {
    fn session_id(&self) -> HostSessionId {
        self.session_id
    }

    fn submit(&mut self, envelope: CommandEnvelope) -> Result<CommandStatus, HostAccessError> {
        const RETRY_PARK: Duration = Duration::from_millis(2);
        let command_id = envelope.command_id;
        let started = Instant::now();
        let mut retrying_authority = false;
        loop {
            let (reply, reply_rx) = std::sync::mpsc::channel();
            if let Err(error) = self.enqueue(
                IngressMessage::Command {
                    envelope: envelope.clone(),
                    reply,
                },
                started,
            ) {
                if retrying_authority && matches!(&error, HostAccessError::ProtocolViolation { .. })
                {
                    return Err(Self::command_authority_timeout(
                        command_id,
                        self.submit_deadline,
                    ));
                }
                return Err(error);
            }
            let remaining = self.submit_deadline.saturating_sub(started.elapsed());
            let result = reply_rx
                .recv_timeout(remaining)
                .map_err(|error| match error {
                    RecvTimeoutError::Timeout if retrying_authority => {
                        Self::command_authority_timeout(command_id, self.submit_deadline)
                    }
                    RecvTimeoutError::Timeout => Self::protocol_violation(format!(
                        "channel host admission reply exceeded {:?}",
                        self.submit_deadline
                    )),
                    RecvTimeoutError::Disconnected => HostAccessError::Disconnected,
                })?;
            match result {
                Err(HostAccessError::CommandAuthorityLookup {
                    failure:
                        crate::CommandAuthorityLookupFailure::Pending
                        | crate::CommandAuthorityLookupFailure::Busy
                        | crate::CommandAuthorityLookupFailure::Capacity { .. },
                    ..
                }) => {
                    retrying_authority = true;
                    let remaining = self.submit_deadline.saturating_sub(started.elapsed());
                    if remaining.is_zero() {
                        return Err(Self::command_authority_timeout(
                            command_id,
                            self.submit_deadline,
                        ));
                    }
                    std::thread::park_timeout(RETRY_PARK.min(remaining));
                }
                result => return result,
            }
        }
    }

    fn command_status(
        &mut self,
        command_id: CommandId,
    ) -> Result<Option<CommandStatus>, HostAccessError> {
        if let Some(status) = self
            .statuses
            .read()
            .map_err(|_| Self::protocol_violation("channel status board poisoned"))?
            .get(command_id)
        {
            return Ok(Some(status));
        }
        const RETRY_PARK: Duration = Duration::from_millis(2);
        let started = Instant::now();
        loop {
            let (reply, reply_rx) = std::sync::mpsc::channel();
            if let Err(error) =
                self.enqueue(IngressMessage::CommandStatus { command_id, reply }, started)
            {
                if matches!(&error, HostAccessError::ProtocolViolation { .. }) {
                    return Err(Self::command_authority_timeout(
                        command_id,
                        self.submit_deadline,
                    ));
                }
                return Err(error);
            }
            let remaining = self.submit_deadline.saturating_sub(started.elapsed());
            let result = reply_rx
                .recv_timeout(remaining)
                .map_err(|error| match error {
                    RecvTimeoutError::Timeout => {
                        Self::command_authority_timeout(command_id, self.submit_deadline)
                    }
                    RecvTimeoutError::Disconnected => HostAccessError::Disconnected,
                })?;
            match result {
                Err(HostAccessError::CommandAuthorityLookup {
                    failure:
                        crate::CommandAuthorityLookupFailure::Pending
                        | crate::CommandAuthorityLookupFailure::Busy
                        | crate::CommandAuthorityLookupFailure::Capacity { .. },
                    ..
                }) => {
                    let remaining = self.submit_deadline.saturating_sub(started.elapsed());
                    if remaining.is_zero() {
                        return Err(Self::command_authority_timeout(
                            command_id,
                            self.submit_deadline,
                        ));
                    }
                    std::thread::park_timeout(RETRY_PARK.min(remaining));
                }
                result => return result,
            }
        }
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
    pending_ingress: Option<IngressMessage>,
    statuses: Arc<RwLock<StatusBoard>>,
    protocol_events: Arc<RwLock<ProtocolEventBoard>>,
    mirror_poll_ids: Vec<CommandId>,
    last_protocol_event: ProtocolEventSequence,
    ingress_drain_budget: usize,
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
            pending_ingress: None,
            statuses,
            protocol_events,
            mirror_poll_ids: Vec::new(),
            last_protocol_event,
            ingress_drain_budget: options.ingress_drain_budget,
            status_board_capacity: options.status_board_capacity,
            protocol_event_capacity: options.protocol_event_capacity,
            maintenance_period: options.maintenance_period,
            controller_disconnected: false,
            shutdown_requested: false,
        };
        Ok((driver, port))
    }

    fn mirror_one(&mut self, command_id: CommandId) {
        let Ok(Some(status)) = self.port.command_status(command_id) else {
            return;
        };
        if let Ok(mut board) = self.statuses.write() {
            board.insert(status, self.status_board_capacity);
        }
    }

    fn mirror_retained_statuses(&mut self) {
        self.mirror_poll_ids.clear();
        let Ok(board) = self.statuses.read() else {
            return;
        };
        self.mirror_poll_ids.extend(board.order.iter().copied());
        drop(board);

        for index in 0..self.mirror_poll_ids.len() {
            let command_id = self.mirror_poll_ids[index];
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

    fn process_ingress(&mut self, message: IngressMessage) {
        match message {
            IngressMessage::Command { envelope, reply } => {
                let result = self.host.submit(envelope);
                if let Ok(status) = &result {
                    if let Ok(mut board) = self.statuses.write() {
                        board.insert(status.clone(), self.status_board_capacity);
                    }
                }
                // A client that timed out is unreachable; the admission
                // remains authoritative and mirrored either way.
                let _ = reply.send(result);
            }
            IngressMessage::CommandStatus { command_id, reply } => {
                let result = self.port.command_status(command_id);
                if let Ok(Some(status)) = &result
                    && let Ok(mut board) = self.statuses.write()
                {
                    board.insert(status.clone(), self.status_board_capacity);
                }
                let _ = reply.send(result);
            }
            IngressMessage::Wake => {}
        }
    }

    fn drain_ingress(&mut self) {
        let mut processed = 0;
        if let Some(message) = self.pending_ingress.take() {
            processed += 1;
            self.process_ingress(message);
        }
        while processed < self.ingress_drain_budget {
            match self.receiver.try_recv() {
                Ok(message) => {
                    processed += 1;
                    self.process_ingress(message);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.controller_disconnected = true;
                    break;
                }
            }
        }
    }

    fn retain_waited_ingress(&mut self, message: IngressMessage) -> Result<(), HostAccessError> {
        if self.pending_ingress.is_some() {
            return Err(ChannelHostPort::protocol_violation(
                "driver tried to retain two waited ingress messages",
            ));
        }
        self.pending_ingress = Some(message);
        Ok(())
    }

    fn wait_for_timed_ingress(
        &mut self,
        interest: HostDriveInterest,
        now: &mut impl FnMut() -> ManualInstant,
    ) -> Result<(), HostAccessError> {
        let wait = match interest {
            HostDriveInterest::Draining => self.maintenance_period,
            HostDriveInterest::ReadyNow | HostDriveInterest::Deadline => self
                .host
                .next_deadline()
                .map_or(self.maintenance_period, |deadline| {
                    let current = now();
                    if deadline > current {
                        Duration::from_nanos(deadline.as_nanos() - current.as_nanos())
                    } else {
                        Duration::ZERO
                    }
                })
                .min(self.maintenance_period.max(Duration::from_millis(1))),
            HostDriveInterest::WakeOnly
            | HostDriveInterest::Terminated
            | HostDriveInterest::Faulted => {
                return Err(ChannelHostPort::protocol_violation(format!(
                    "driver requested a timed wait for non-timed interest {interest:?}"
                )));
            }
        };
        match self.receiver.recv_timeout(wait) {
            Ok(message) => self.retain_waited_ingress(message)?,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                self.controller_disconnected = true;
            }
        }
        Ok(())
    }

    /// Process at most the configured ingress budget, drive the host when due,
    /// and mirror all boards once.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelDriveError`] when admission or the schedule fails.
    pub fn step(&mut self, now: ManualInstant) -> Result<ChannelStepReport, ChannelDriveError> {
        let admission_before = self.host.core().admission_cursor();
        self.drain_ingress();
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
            match self.host.request_shutdown() {
                Ok(_) => self.shutdown_requested = true,
                Err(HostAccessError::CommandAuthorityLookup {
                    failure:
                        crate::CommandAuthorityLookupFailure::Pending
                        | crate::CommandAuthorityLookupFailure::Busy
                        | crate::CommandAuthorityLookupFailure::Capacity { .. },
                    ..
                }) => {}
                Err(error) => return Err(error.into()),
            }
        }
        self.mirror_retained_statuses();
        self.mirror_protocol_events();
        interest = self.host.drive_interest();
        if self.controller_disconnected
            && !self.shutdown_requested
            && interest == HostDriveInterest::WakeOnly
        {
            interest = HostDriveInterest::Draining;
        }
        let admission_after = self.host.core().admission_cursor();
        let admitted = admission_after
            .get()
            .checked_sub(admission_before.get())
            .and_then(|count| usize::try_from(count).ok())
            .ok_or_else(|| {
                ChannelHostPort::protocol_violation(
                    "channel host admission cursor regressed or exceeded platform usize",
                )
            })?;
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
    /// one ordered shutdown rather than abandoning the world. An error leaves
    /// this driver and its exact host state owned by the caller for an explicit
    /// [`Self::step`] recovery or a later run retry; retained client ports stay
    /// available for status inspection.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelDriveError`] when a boundary cannot advance.
    pub fn run(
        &mut self,
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
                    match self.receiver.recv() {
                        Ok(message) => self.retain_waited_ingress(message)?,
                        Err(_) => self.controller_disconnected = true,
                    }
                }
                HostDriveInterest::Draining
                | HostDriveInterest::ReadyNow
                | HostDriveInterest::Deadline => {
                    self.wait_for_timed_ingress(report.interest, &mut now)?;
                }
            }
        }
    }

    /// Immutable access to the driver-owned host.
    #[must_use]
    pub const fn host(&self) -> &FixedDeadlineHost {
        &self.host
    }

    /// Mutable access to the driver-owned host.
    pub fn host_mut(&mut self) -> &mut FixedDeadlineHost {
        &mut self.host
    }

    /// Retry the exact retained journal allocation once.
    pub fn retry_retained_journal(
        &mut self,
    ) -> Result<Option<crate::JournalAdmission>, HostAccessError> {
        self.host.retry_retained_journal()
    }

    /// Consume the driver and return the retained owner-pinned host.
    #[must_use]
    pub fn into_host(self) -> FixedDeadlineHost {
        self.host
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ApplicationState, CommandAuthorityLookup, CommandAuthorityMode, CommandAuthorityReader,
        CommandClaimPolicy, HostCommand, HostCore, HostCoreOptions, HostLifecycle,
        JournalAdmission, JournalBatch, JournalPort, JournalReceipt, PlaybackSnapshot,
        RejectionReason, ShutdownCommitRequirement, VolatileJournal,
    };
    use scriptbots_core::ScriptBotsConfig;
    use std::sync::Mutex;

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

    struct ScriptedChannelAuthority {
        outcomes: Mutex<HashMap<CommandId, VecDeque<CommandAuthorityLookup>>>,
    }

    impl CommandAuthorityReader for ScriptedChannelAuthority {
        fn resolve_for_submit(
            &self,
            envelope: &CommandEnvelope,
            _envelope_digest: [u8; blake3::OUT_LEN],
            _policy: CommandClaimPolicy,
        ) -> CommandAuthorityLookup {
            self.resolve_status(envelope.command_id)
        }

        fn resolve_status(&self, command_id: CommandId) -> CommandAuthorityLookup {
            self.outcomes
                .lock()
                .expect("scripted channel authority lock")
                .get_mut(&command_id)
                .and_then(VecDeque::pop_front)
                .unwrap_or(CommandAuthorityLookup::Absent)
        }
    }

    struct ChannelAuthorityJournal {
        inner: VolatileJournal,
        authority: Arc<dyn CommandAuthorityReader>,
    }

    impl JournalPort for ChannelAuthorityJournal {
        fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
            self.inner.try_admit(batch)
        }

        fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
            self.inner.poll_receipts(limit)
        }

        fn command_authority_reader(
            &self,
            _session_id: HostSessionId,
        ) -> Option<Arc<dyn CommandAuthorityReader>> {
            Some(Arc::clone(&self.authority))
        }

        fn command_authority_mode(&self) -> CommandAuthorityMode {
            CommandAuthorityMode::DurableRequired
        }

        fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
            self.inner.shutdown_commit_requirement()
        }
    }

    fn authority_test_host(authority: Arc<dyn CommandAuthorityReader>) -> FixedDeadlineHost {
        let world = scriptbots_core::WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0x5eed_cafe),
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .expect("deterministic authority world");
        let options = HostCoreOptions {
            initial_playback: PlaybackSnapshot {
                paused: true,
                ..PlaybackSnapshot::default()
            },
            tick_period_nanos: 1_000_000_000,
            snapshot_interval_ticks: 1,
            ..HostCoreOptions::default()
        };
        let core = HostCore::with_journal(
            HostSessionId::new(0x7171),
            world,
            options,
            Box::new(ChannelAuthorityJournal {
                inner: VolatileJournal::with_capacity(64),
                authority,
            }),
        )
        .expect("channel authority host");
        FixedDeadlineHost::new(core)
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
            let (mut driver, port) =
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
    fn quiescent_driver_wait_preserves_the_command_that_wakes_it() {
        let (worker, mut port) = spawn_driver(true);

        // A paused host reports WakeOnly, so the owner is blocked in recv()
        // until this exact command arrives. The wait path must retain the
        // message for the next ordered ingress drain instead of consuming it
        // merely as a wake notification.
        std::thread::sleep(Duration::from_millis(20));
        let step = port
            .submit(CommandEnvelope::new(CommandId::new(13), HostCommand::Step))
            .expect("wake command reaches authoritative admission");
        assert!(matches!(step.application(), ApplicationState::Admitted));
        let stepped = wait_resolved(&mut port, CommandId::new(13));
        assert!(matches!(
            stepped.application(),
            ApplicationState::Applied(_)
        ));

        let receipt = shutdown_and_join(worker, &mut port);
        assert_eq!(receipt.outcome, ChannelRunOutcome::Stopped);
        assert!(receipt.commands_admitted >= 2);
    }

    type AdmissionReceipt = Receiver<Result<CommandStatus, HostAccessError>>;

    fn timed_wait_driver(draining: bool) -> (ChannelHostDriver, ChannelHostPort) {
        ChannelHostDriver::new(
            test_host(draining),
            ChannelHostOptions {
                ingress_capacity: 1,
                ingress_drain_budget: 1,
                ..fast_options()
            },
        )
        .expect("driver")
    }

    fn queue_timed_wait_setup(
        port: &ChannelHostPort,
        draining: bool,
        setup_id: CommandId,
    ) -> Option<AdmissionReceipt> {
        if draining {
            let (reply, receipt) = std::sync::mpsc::channel();
            assert!(
                port.sender
                    .try_send(IngressMessage::Command {
                        envelope: CommandEnvelope::new(setup_id, HostCommand::Step),
                        reply,
                    })
                    .is_ok()
            );
            Some(receipt)
        } else {
            assert!(port.sender.try_send(IngressMessage::Wake).is_ok());
            None
        }
    }

    fn hold_final_mirror_until_waking_queued(
        port: &ChannelHostPort,
    ) -> (std::thread::JoinHandle<()>, Sender<()>) {
        // Hold the final mirror write while the first step drains the setup
        // message. This opens exactly one queue slot after the drain budget is
        // spent, so the waking command can be queued but cannot be consumed
        // until run() executes the interest-specific recv_timeout branch.
        let protocol_events = Arc::clone(&port.protocol_events);
        let (mirror_locked_tx, mirror_locked_rx) = std::sync::mpsc::channel();
        let (waking_queued_tx, waking_queued_rx) = std::sync::mpsc::channel();
        let mirror_lock = std::thread::spawn(move || {
            let guard = protocol_events.read().expect("protocol-event board lock");
            mirror_locked_tx
                .send(())
                .expect("mirror-lock acquisition signal");
            waking_queued_rx
                .recv()
                .expect("waking-command queue signal");
            drop(guard);
        });
        mirror_locked_rx
            .recv()
            .expect("protocol-event board lock acquired");
        (mirror_lock, waking_queued_tx)
    }

    fn spawn_timed_wait_client(
        port: &ChannelHostPort,
        waking_queued_tx: Sender<()>,
        waking_id: CommandId,
        shutdown_id: CommandId,
    ) -> std::thread::JoinHandle<(CommandStatus, CommandStatus)> {
        let sender = port.sender.clone();
        std::thread::spawn(move || {
            let (waking_reply, waking_receipt) = std::sync::mpsc::channel();
            assert!(
                sender
                    .send(IngressMessage::Command {
                        envelope: CommandEnvelope::new(waking_id, HostCommand::SetSpeed(2.0),),
                        reply: waking_reply,
                    })
                    .is_ok()
            );
            waking_queued_tx
                .send(())
                .expect("waking-command queue signal");

            let (shutdown_reply, shutdown_receipt) = std::sync::mpsc::channel();
            assert!(
                sender
                    .send(IngressMessage::Command {
                        envelope: CommandEnvelope::new(shutdown_id, HostCommand::Shutdown),
                        reply: shutdown_reply,
                    })
                    .is_ok()
            );

            let waking = waking_receipt
                .recv_timeout(Duration::from_secs(2))
                .expect("waking command reply")
                .expect("waking command admission");
            let shutdown = shutdown_receipt
                .recv_timeout(Duration::from_secs(2))
                .expect("shutdown command reply")
                .expect("shutdown command admission");
            (waking, shutdown)
        })
    }

    fn assert_admission(status: &CommandStatus, command_id: CommandId, sequence: u64) {
        assert_eq!(status.command_id(), command_id);
        assert_eq!(
            status.admission_sequence(),
            Some(crate::AdmissionSequence::new(sequence))
        );
    }

    fn assert_final_applied_status(
        port: &mut ChannelHostPort,
        command_id: CommandId,
        sequence: u64,
    ) {
        let status = port
            .command_status(command_id)
            .expect("final status lookup")
            .expect("final status retained");
        assert_admission(&status, command_id, sequence);
        assert!(matches!(status.application(), ApplicationState::Applied(_)));
    }

    fn assert_run_timed_wait_preserves_waking_command(interest: HostDriveInterest) {
        assert!(matches!(
            interest,
            HostDriveInterest::Draining | HostDriveInterest::Deadline
        ));
        let draining = interest == HostDriveInterest::Draining;
        let (mut driver, mut port) = timed_wait_driver(draining);
        let setup_id = CommandId::new(if draining { 14 } else { 16 });
        let waking_id = CommandId::new(if draining { 15 } else { 17 });
        let shutdown_id = CommandId::new(if draining { 18 } else { 19 });
        let setup_receipt = queue_timed_wait_setup(&port, draining, setup_id);
        let (mirror_lock, waking_queued_tx) = hold_final_mirror_until_waking_queued(&port);
        let client = spawn_timed_wait_client(&port, waking_queued_tx, waking_id, shutdown_id);

        let start = std::time::Instant::now();
        let receipt = driver
            .run(move || {
                let elapsed_nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
                ManualInstant::from_nanos(1_000_000_u64.saturating_add(elapsed_nanos))
            })
            .expect("driver reaches ordered shutdown");
        mirror_lock.join().expect("mirror-lock holder");
        let (waking, shutdown) = client.join().expect("timed-wait client");

        let first_waking_sequence = if draining { 2 } else { 1 };
        assert_admission(&waking, waking_id, first_waking_sequence);
        assert_admission(&shutdown, shutdown_id, first_waking_sequence + 1);
        assert_eq!(receipt.outcome, ChannelRunOutcome::Stopped);
        assert_eq!(
            receipt.commands_admitted,
            if draining { 3 } else { 2 },
            "the retained waking command must be counted exactly once"
        );

        let mut expected = vec![
            (waking_id, first_waking_sequence),
            (shutdown_id, first_waking_sequence + 1),
        ];
        if let Some(setup_receipt) = setup_receipt {
            let setup = setup_receipt
                .recv_timeout(Duration::from_secs(2))
                .expect("setup command reply")
                .expect("setup command admission");
            assert_admission(&setup, setup_id, 1);
            expected.insert(0, (setup_id, 1));
        }
        for (command_id, sequence) in expected {
            assert_final_applied_status(&mut port, command_id, sequence);
        }
    }

    #[test]
    fn draining_timed_wait_preserves_the_command_that_wakes_it() {
        assert_run_timed_wait_preserves_waking_command(HostDriveInterest::Draining);
    }

    #[test]
    fn deadline_timed_wait_preserves_the_command_that_wakes_it() {
        assert_run_timed_wait_preserves_waking_command(HostDriveInterest::Deadline);
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
    fn run_error_preserves_driver_and_pending_journal_work() {
        use std::cell::{Cell, RefCell};
        use std::rc::Rc;

        #[derive(Default)]
        struct DeferredJournalState {
            suppress_receipts: bool,
            attempts: Vec<crate::JournalBatchId>,
            receipts: VecDeque<crate::JournalReceipt>,
        }

        struct DeferredJournal {
            state: Rc<RefCell<DeferredJournalState>>,
            dropped: Rc<Cell<bool>>,
        }

        impl Drop for DeferredJournal {
            fn drop(&mut self) {
                self.dropped.set(true);
            }
        }

        impl crate::JournalPort for DeferredJournal {
            fn try_admit(
                &mut self,
                batch: &std::sync::Arc<crate::JournalBatch>,
            ) -> crate::JournalAdmission {
                let mut state = self.state.borrow_mut();
                let batch_id = batch.id();
                state.attempts.push(batch_id);
                if !state.suppress_receipts {
                    state.receipts.push_back(crate::JournalReceipt::new(
                        batch_id,
                        crate::JournalReceiptState::Durable,
                    ));
                }
                crate::JournalAdmission::Accepted { batch_id }
            }

            fn poll_receipts(&mut self, limit: usize) -> Vec<crate::JournalReceipt> {
                let mut state = self.state.borrow_mut();
                let count = limit.min(state.receipts.len());
                state.receipts.drain(..count).collect()
            }

            fn command_authority_mode(&self) -> crate::CommandAuthorityMode {
                crate::CommandAuthorityMode::ProcessLocal
            }
        }

        let journal_state = Rc::new(RefCell::new(DeferredJournalState {
            suppress_receipts: true,
            ..DeferredJournalState::default()
        }));
        let journal_dropped = Rc::new(Cell::new(false));
        let world = scriptbots_core::WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0x5eed_cafe),
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .expect("deterministic test world");
        let core = HostCore::with_journal(
            HostSessionId::new(10),
            world,
            HostCoreOptions {
                initial_playback: PlaybackSnapshot {
                    paused: true,
                    ..PlaybackSnapshot::default()
                },
                tick_period_nanos: 10,
                snapshot_interval_ticks: 1,
                ..HostCoreOptions::default()
            },
            Box::new(DeferredJournal {
                state: Rc::clone(&journal_state),
                dropped: Rc::clone(&journal_dropped),
            }),
        )
        .expect("host with deferred journal");
        let (mut driver, mut port) = ChannelHostDriver::new(
            FixedDeadlineHost::new(core),
            ChannelHostOptions {
                submit_deadline: Duration::from_secs(5),
                ..fast_options()
            },
        )
        .expect("driver");

        let client = std::thread::spawn(move || {
            let admitted = port
                .submit(CommandEnvelope::new(CommandId::new(82), HostCommand::Step))
                .expect("step admission");
            (port, admitted)
        });
        let mut observations = [10_u64, 9, 8].into_iter();
        let error = driver
            .run(|| {
                ManualInstant::from_nanos(
                    observations
                        .next()
                        .expect("backwards-clock run must fail promptly"),
                )
            })
            .expect_err("backwards clock must fail the run");
        assert!(matches!(
            error,
            ChannelDriveError::Schedule(NativeScheduleError::ClockMovedBackwards { .. })
        ));

        let (mut port, admitted) = client.join().expect("client thread");
        assert_eq!(admitted.journal(), &crate::JournalState::Pending);
        let pending = port
            .command_status(CommandId::new(82))
            .expect("status lookup")
            .expect("step status retained");
        assert!(matches!(
            pending.application(),
            ApplicationState::Applied(_)
        ));
        assert_eq!(pending.journal(), &crate::JournalState::Pending);
        assert!(
            !journal_dropped.get(),
            "run error dropped the sole-owner host and its pending journal work"
        );
        assert_eq!(driver.host.drive_interest(), HostDriveInterest::Draining);

        {
            let mut state = journal_state.borrow_mut();
            let batch_id = *state.attempts.first().expect("accepted journal batch");
            state.receipts.push_back(crate::JournalReceipt::new(
                batch_id,
                crate::JournalReceiptState::Durable,
            ));
            state.suppress_receipts = false;
        }
        driver
            .step(ManualInstant::from_nanos(11))
            .expect("retained driver settles the pending journal receipt");
        let durable = port
            .command_status(CommandId::new(82))
            .expect("status lookup")
            .expect("step status retained");
        assert_eq!(durable.journal(), &crate::JournalState::Durable);

        drop(port);
        driver
            .step(ManualInstant::from_nanos(12))
            .expect("disconnect requests ordered shutdown");
        driver
            .step(ManualInstant::from_nanos(13))
            .expect("ordered shutdown applies");
        let stopped = driver
            .step(ManualInstant::from_nanos(14))
            .expect("ordered shutdown journal settles");
        assert_eq!(stopped.interest, HostDriveInterest::Terminated);
        assert!(
            !journal_dropped.get(),
            "retained driver must still own its settled journal"
        );
    }

    #[test]
    fn controller_disconnect_requests_ordered_shutdown() {
        let (mut driver, port) =
            ChannelHostDriver::new(test_host(false), fast_options()).expect("driver");
        drop(port);
        let start = std::time::Instant::now();
        let receipt = driver
            .run(move || ManualInstant::from_nanos(1_000_000 + start.elapsed().as_nanos() as u64))
            .expect("run");
        assert_eq!(receipt.outcome, ChannelRunOutcome::ControllerDisconnected);
        assert_eq!(receipt.commands_admitted, 1);
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
    fn submit_reply_timeout_after_pending_authority_is_typed_and_keeps_command_id() {
        let command_id = CommandId::new(611);
        let submit_deadline = Duration::from_millis(500);
        let (driver, mut port) = ChannelHostDriver::new(
            test_host(true),
            ChannelHostOptions {
                submit_deadline,
                ..fast_options()
            },
        )
        .expect("driver");
        let caller = std::thread::spawn(move || {
            port.submit(CommandEnvelope::new(command_id, HostCommand::Pause))
        });

        let first = driver
            .receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("initial authority submission");
        match first {
            IngressMessage::Command { envelope, reply } => {
                assert_eq!(envelope.command_id, command_id);
                reply
                    .send(Err(HostAccessError::CommandAuthorityLookup {
                        command_id,
                        failure: crate::CommandAuthorityLookupFailure::Pending,
                    }))
                    .expect("pending authority reply");
            }
            IngressMessage::CommandStatus { .. } | IngressMessage::Wake => {
                panic!("expected command submission")
            }
        }

        let stalled_retry = driver
            .receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("exact authority retry");
        assert!(matches!(
            &stalled_retry,
            IngressMessage::Command { envelope, .. } if envelope.command_id == command_id
        ));
        let error = caller
            .join()
            .expect("submission caller")
            .expect_err("stalled authority reply must time out");
        assert_eq!(
            error,
            HostAccessError::CommandAuthorityLookup {
                command_id,
                failure: crate::CommandAuthorityLookupFailure::Timeout {
                    waited: submit_deadline,
                },
            }
        );
        drop(stalled_retry);
    }

    #[test]
    fn status_reply_timeout_after_busy_authority_is_typed_and_keeps_command_id() {
        let command_id = CommandId::new(612);
        let submit_deadline = Duration::from_millis(500);
        let (driver, mut port) = ChannelHostDriver::new(
            test_host(true),
            ChannelHostOptions {
                submit_deadline,
                ..fast_options()
            },
        )
        .expect("driver");
        let caller = std::thread::spawn(move || port.command_status(command_id));

        let first = driver
            .receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("initial authority lookup");
        match first {
            IngressMessage::CommandStatus {
                command_id: requested_id,
                reply,
            } => {
                assert_eq!(requested_id, command_id);
                reply
                    .send(Err(HostAccessError::CommandAuthorityLookup {
                        command_id,
                        failure: crate::CommandAuthorityLookupFailure::Busy,
                    }))
                    .expect("busy authority reply");
            }
            IngressMessage::Command { .. } | IngressMessage::Wake => {
                panic!("expected command-status lookup")
            }
        }

        let stalled_retry = driver
            .receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("exact status retry");
        assert!(matches!(
            &stalled_retry,
            IngressMessage::CommandStatus {
                command_id: requested_id,
                ..
            } if *requested_id == command_id
        ));
        let error = caller
            .join()
            .expect("status caller")
            .expect_err("stalled authority lookup must time out");
        assert_eq!(
            error,
            HostAccessError::CommandAuthorityLookup {
                command_id,
                failure: crate::CommandAuthorityLookupFailure::Timeout {
                    waited: submit_deadline,
                },
            }
        );
        drop(stalled_retry);
    }

    #[test]
    fn enqueue_cannot_restart_an_expired_outer_submission_deadline() {
        let (mut driver, port) =
            ChannelHostDriver::new(test_host(false), fast_options()).expect("driver");
        let expired = Instant::now()
            .checked_sub(port.submit_deadline)
            .expect("test deadline subtraction");

        let error = port
            .enqueue(IngressMessage::Wake, expired)
            .expect_err("expired submission must not enter ingress");
        assert!(matches!(error, HostAccessError::ProtocolViolation { .. }));
        assert!(matches!(
            driver.receiver.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn admission_accounting_includes_claim_resolved_by_drive_but_not_exact_retry() {
        let command_id = CommandId::new(62);
        let envelope = CommandEnvelope::new(command_id, HostCommand::Pause);
        let authority = Arc::new(ScriptedChannelAuthority {
            outcomes: Mutex::new(HashMap::from([(
                command_id,
                VecDeque::from([
                    CommandAuthorityLookup::Failed(crate::CommandAuthorityLookupFailure::Pending),
                    CommandAuthorityLookup::Claimed,
                ]),
            )])),
        });
        let (mut driver, port) =
            ChannelHostDriver::new(authority_test_host(authority), fast_options())
                .expect("channel authority driver");

        let (first_reply, first_receipt) = std::sync::mpsc::channel();
        port.sender
            .try_send(IngressMessage::Command {
                envelope: envelope.clone(),
                reply: first_reply,
            })
            .expect("first authority ingress");
        let first = driver
            .step(ManualInstant::from_nanos(0))
            .expect("claim resolves during owner drive");
        assert_eq!(first.admitted, 1);
        assert!(matches!(
            first_receipt.recv().expect("first authority reply"),
            Err(HostAccessError::CommandAuthorityLookup {
                command_id: pending_id,
                failure: crate::CommandAuthorityLookupFailure::Pending,
            }) if pending_id == command_id
        ));

        let (retry_reply, retry_receipt) = std::sync::mpsc::channel();
        port.sender
            .try_send(IngressMessage::Command {
                envelope,
                reply: retry_reply,
            })
            .expect("exact retry ingress");
        let retry = driver
            .step(ManualInstant::from_nanos(1))
            .expect("exact retry boundary");
        assert_eq!(retry.admitted, 0);
        let status = retry_receipt
            .recv()
            .expect("exact retry reply")
            .expect("exact retry status");
        assert_eq!(status.command_id(), command_id);
        assert!(matches!(status.application(), ApplicationState::Applied(_)));
    }

    #[test]
    fn zero_ingress_drain_budget_is_rejected() {
        let result = ChannelHostDriver::new(
            test_host(false),
            ChannelHostOptions {
                ingress_drain_budget: 0,
                ..ChannelHostOptions::default()
            },
        );
        assert!(matches!(
            result,
            Err(ChannelHostOptionsError::EmptyIngressDrainBudget)
        ));
    }

    #[test]
    fn retained_wake_consumes_budget_before_newer_ingress() {
        let (mut driver, port) = ChannelHostDriver::new(
            test_host(false),
            ChannelHostOptions {
                ingress_capacity: 1,
                ingress_drain_budget: 1,
                ..ChannelHostOptions::default()
            },
        )
        .expect("driver");
        driver
            .retain_waited_ingress(IngressMessage::Wake)
            .expect("one retained wake");

        let (reply, receipt) = std::sync::mpsc::channel();
        assert!(
            port.sender
                .try_send(IngressMessage::Command {
                    envelope: CommandEnvelope::new(
                        CommandId::new(9_999),
                        HostCommand::SetSpeed(-1.0),
                    ),
                    reply,
                })
                .is_ok()
        );

        let first = driver
            .step(ManualInstant::from_nanos(1))
            .expect("retained-wake boundary");
        assert_eq!(first.admitted, 0);
        assert!(first.drove);
        assert!(matches!(
            receipt.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));

        let next_due = driver.host.next_deadline().expect("next running deadline");
        let second = driver.step(next_due).expect("queued-command boundary");
        assert_eq!(second.admitted, 0);
        let status = receipt
            .recv_timeout(Duration::from_secs(1))
            .expect("queued command reply")
            .expect("authoritative validation rejection");
        assert_eq!(status.command_id(), CommandId::new(9_999));
    }

    #[test]
    fn producer_refill_cannot_starve_a_due_drive_boundary() {
        let (mut driver, port) =
            ChannelHostDriver::new(test_host(false), ChannelHostOptions::default())
                .expect("driver");
        let primed = driver
            .step(ManualInstant::from_nanos(1))
            .expect("prime fixed cadence");
        assert!(primed.drove);
        let due = driver.host.next_deadline().expect("armed fixed deadline");
        let deadlines_before = driver.host.total_scheduled_deadlines();

        for offset in 0..DEFAULT_CHANNEL_INGRESS_CAPACITY {
            let (reply, _receipt) = std::sync::mpsc::channel();
            let command_id =
                CommandId::new(10_000 + u128::try_from(offset).expect("test offset fits u128"));
            assert!(
                port.sender
                    .try_send(IngressMessage::Command {
                        envelope: CommandEnvelope::new(command_id, HostCommand::SetSpeed(-1.0),),
                        reply,
                    })
                    .is_ok(),
                "the initial ingress fill must fit exactly"
            );
        }

        // Hold the first mirrored-status write until a producer has refilled
        // the slot freed by the first receive. This makes the sustained refill
        // deterministic without an infinite producer or timing assumptions.
        let statuses = Arc::clone(&port.statuses);
        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let (refilled_tx, refilled_rx) = std::sync::mpsc::channel();
        let status_lock = std::thread::spawn(move || {
            let guard = statuses.read().expect("status board lock");
            locked_tx.send(()).expect("lock acquisition signal");
            refilled_rx.recv().expect("producer refill signal");
            drop(guard);
        });
        locked_rx.recv().expect("status board lock acquired");

        let sender = port.sender.clone();
        let (last_reply, last_receipt) = std::sync::mpsc::channel();
        let refill = std::thread::spawn(move || {
            assert!(
                sender
                    .send(IngressMessage::Command {
                        envelope: CommandEnvelope::new(
                            CommandId::new(20_000),
                            HostCommand::SetSpeed(-1.0),
                        ),
                        reply: last_reply,
                    })
                    .is_ok(),
                "owner must remain connected"
            );
            refilled_tx.send(()).expect("producer refill signal");
        });

        let first = driver.step(due).expect("first bounded boundary");
        refill.join().expect("refill producer");
        status_lock.join().expect("status lock holder");

        assert_eq!(
            first.admitted, 0,
            "validation rejections must not be counted as admissions"
        );
        assert!(first.drove, "the due host deadline must still advance");
        assert_eq!(
            driver.host.total_scheduled_deadlines(),
            deadlines_before + 1,
            "the refilled ingress queue must not hide the elapsed cadence deadline"
        );
        assert!(
            driver.host.next_deadline().is_some_and(|next| next > due),
            "the fixed deadline must advance beyond the flooded boundary"
        );
        assert!(
            matches!(
                last_receipt.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Empty)
            ),
            "the refilled command must preserve FIFO for the next boundary"
        );

        let next_due = driver.host.next_deadline().expect("next running deadline");
        let second = driver.step(next_due).expect("second bounded boundary");
        assert_eq!(second.admitted, 0);
        let status = last_receipt
            .recv_timeout(Duration::from_secs(1))
            .expect("refilled command reply")
            .expect("authoritative validation rejection");
        assert_eq!(status.command_id(), CommandId::new(20_000));
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
    fn archived_retry_cannot_strand_a_retained_admitted_status() {
        let (mut driver, mut port) = ChannelHostDriver::new(
            test_host(false),
            ChannelHostOptions {
                status_board_capacity: 1,
                ..fast_options()
            },
        )
        .expect("driver");

        fn process(driver: &mut ChannelHostDriver, envelope: CommandEnvelope) -> CommandStatus {
            let (reply, receipt) = std::sync::mpsc::channel();
            driver.process_ingress(IngressMessage::Command { envelope, reply });
            receipt
                .recv()
                .expect("authoritative admission reply")
                .expect("host remains available")
        }

        let archived_id = CommandId::new(74);
        let archived_envelope = CommandEnvelope::new(archived_id, HostCommand::Pause);
        let admitted = process(&mut driver, archived_envelope.clone());
        assert!(matches!(admitted.application(), ApplicationState::Admitted));

        driver
            .step(ManualInstant::from_nanos(1))
            .expect("pause applies");
        driver
            .step(ManualInstant::from_nanos(2))
            .expect("volatile journal receipt settles");
        let archived = port
            .command_status(archived_id)
            .expect("mirrored lookup")
            .expect("settled pause retained");
        assert!(matches!(
            archived.application(),
            ApplicationState::Applied(_)
        ));
        assert!(status_is_finished(&archived));

        // Fresh A displaces terminal B from both retention structures. An
        // archived retry of B then made the old FIFO drop A while the board
        // protected admitted A by immediately evicting terminal B again.
        let retained_id = CommandId::new(75);
        let retained = process(
            &mut driver,
            CommandEnvelope::new(retained_id, HostCommand::SetSpeed(2.0)),
        );
        assert!(matches!(retained.application(), ApplicationState::Admitted));

        let retry = process(&mut driver, archived_envelope);
        assert_eq!(retry, archived);
        assert!(
            port.statuses
                .read()
                .expect("status board lock")
                .get(archived_id)
                .is_none(),
            "the terminal retry must evict itself before the retained admission"
        );
        assert!(matches!(
            port.command_status(retained_id)
                .expect("mirrored lookup")
                .expect("admitted status retained")
                .application(),
            ApplicationState::Admitted
        ));

        let report = driver
            .step(ManualInstant::from_nanos(3))
            .expect("retained command applies");
        assert!(report.drove);
        let authoritative = driver
            .port
            .command_status(retained_id)
            .expect("authoritative lookup")
            .expect("authoritative status retained");
        assert!(matches!(
            authoritative.application(),
            ApplicationState::Applied(_)
        ));
        let mirrored = port
            .command_status(retained_id)
            .expect("mirrored lookup")
            .expect("board-retained status must remain available");
        assert_eq!(
            mirrored, authoritative,
            "the board must mirror the exact authoritative terminal status"
        );
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

        fn persistence_disabled_world() -> scriptbots_core::WorldState {
            scriptbots_core::WorldState::new(ScriptBotsConfig {
                rng_seed: Some(0x5eed_cafe),
                persistence_interval: 0,
                ..ScriptBotsConfig::default()
            })
            .expect("deterministic test world")
        }

        let world = persistence_disabled_world();
        let persistence = world
            .bind_persistence(Box::new(scriptbots_core::NullPersistence))
            .expect("one-shot binding");
        let core = HostCore::with_journal_and_persistence(
            HostSessionId::new(9),
            world,
            HostCoreOptions::default(),
            Box::new(NullTestJournal),
            persistence,
        )
        .expect("caller-bound session must construct");
        assert_eq!(core.world_tick(), scriptbots_core::Tick(0));

        let session_world = persistence_disabled_world();
        let wrong_session = session_world
            .bind_persistence(Box::new(scriptbots_core::NullPersistence))
            .expect("wrong-world fixture binding");
        let wrong_world = persistence_disabled_world();
        let wrong_pair = HostCore::with_journal_and_persistence(
            HostSessionId::new(10),
            wrong_world,
            HostCoreOptions::default(),
            Box::new(NullTestJournal),
            wrong_session,
        );
        assert!(matches!(
            wrong_pair,
            Err(crate::HostCoreBuildError::Persistence(
                scriptbots_core::PersistenceSessionError::WrongWorld
            ))
        ));

        let conflicting_world = persistence_disabled_world();
        let _existing_session = conflicting_world
            .bind_persistence(Box::new(scriptbots_core::NullPersistence))
            .expect("first binding");
        let conflicting = HostCore::with_journal(
            HostSessionId::new(11),
            conflicting_world,
            HostCoreOptions::default(),
            Box::new(NullTestJournal),
        );
        assert!(matches!(
            conflicting,
            Err(crate::HostCoreBuildError::Persistence(
                scriptbots_core::PersistenceSessionError::AlreadyBound
            ))
        ));
    }
}
