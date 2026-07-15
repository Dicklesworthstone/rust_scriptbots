//! Fixed-deadline native ownership and lifecycle adapter.
//!
//! The scheduling kernel in this module stays usable without an async runtime.
//! The optional Asupersync runner owns the exact same kernel as a root future:
//! [`HostCore`](crate::HostCore) is deliberately `!Send`, so it is never moved
//! into a spawned task or hidden behind a shared mutable-world lock.

use super::{
    CommandEnvelope, CommandStatus, DriveReceipt, HostAccessError, HostCore, HostDriveInterest,
    HostPort, JournalAdmission, LocalHostPort, ManualHostDriver, ManualInstant,
};
use thiserror::Error;

/// Reason a native owner observed and drove one host boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDriveTrigger {
    /// Establish the initial time boundary without a hidden scientific tick.
    Startup,
    /// The next absolute science deadline became due.
    Deadline,
    /// One or more command envelopes woke the owner.
    Command,
    /// A coalesced non-command wake requested an observation.
    SyntheticWake,
    /// A journal adapter reported that retained or inflight work may progress.
    JournalReady,
    /// Structured cancellation requested ordered shutdown.
    Cancellation,
    /// Shutdown or journal receipt maintenance requested another boundary.
    Maintenance,
}

#[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
impl NativeDriveTrigger {
    const fn priority(self) -> u8 {
        match self {
            Self::Startup => 0,
            Self::Maintenance => 1,
            Self::Deadline => 2,
            Self::SyntheticWake => 3,
            Self::Command => 4,
            Self::JournalReady => 5,
            Self::Cancellation => 6,
        }
    }

    const fn combine(self, other: Self) -> Self {
        if other.priority() > self.priority() {
            other
        } else {
            self
        }
    }
}

/// Result of one native fixed-deadline observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeDriveReceipt {
    /// Cause with the highest boundary-order priority.
    pub trigger: NativeDriveTrigger,
    /// Monotonic instant supplied to the exact [`HostCore`].
    pub observed_at: ManualInstant,
    /// Fixed cadence deadlines crossed since the preceding observation.
    pub scheduled_deadlines_elapsed: u64,
    /// Whole wall-clock deadline opportunities beyond the host catch-up cap.
    pub scheduled_deadlines_skipped: u64,
    /// First absolute cadence deadline strictly after this observation.
    pub next_deadline: ManualInstant,
    /// Exact receipt returned by [`HostCore::drive`](ManualHostDriver::drive).
    pub host: DriveReceipt,
}

/// Failure to advance the native fixed-deadline kernel.
#[derive(Debug, Error)]
pub enum NativeScheduleError {
    /// The injected monotonic clock regressed before the host was touched.
    #[error("native monotonic time moved backwards from {previous}ns to {observed}ns")]
    ClockMovedBackwards {
        /// Last successfully observed instant.
        previous: u64,
        /// Regressing instant.
        observed: u64,
    },
    /// No later absolute deadline can be represented by the protocol clock.
    #[error("native fixed deadline overflowed after {at}ns with period {period_nanos}ns")]
    DeadlineOverflow {
        /// Absolute deadline that could not be advanced.
        at: u64,
        /// Configured host cadence.
        period_nanos: u64,
    },
    /// The sole-owner host rejected the boundary.
    #[error(transparent)]
    Host(#[from] HostAccessError),
}

/// Owner-pinned adapter that drives one exact [`HostCore`] at absolute deadlines.
///
/// Early wakes call the host at the observed monotonic instant but never rebase
/// `next_deadline`. Late observations advance directly to the first future
/// deadline, while `HostCore` remains the sole authority for bounded scientific
/// catch-up and command ordering.
pub struct FixedDeadlineHost {
    core: HostCore,
    port: LocalHostPort,
    last_observed: Option<ManualInstant>,
    next_deadline: Option<ManualInstant>,
    total_scheduled_deadlines: u64,
    total_scheduled_deadlines_skipped: u64,
}

impl FixedDeadlineHost {
    /// Wrap a sole-owner host without driving it or changing its world.
    #[must_use]
    pub fn new(core: HostCore) -> Self {
        let port = core.local_port();
        Self {
            core,
            port,
            last_observed: None,
            next_deadline: None,
            total_scheduled_deadlines: 0,
            total_scheduled_deadlines_skipped: 0,
        }
    }

    /// Clone the same-thread protocol handle without exposing mutable world state.
    #[must_use]
    pub fn local_port(&self) -> LocalHostPort {
        self.port.clone()
    }

    /// Read the exact sole-owner host for diagnostics and immutable projections.
    #[must_use]
    pub const fn core(&self) -> &HostCore {
        &self.core
    }

    /// Consume the adapter and return its exact retained host state.
    #[must_use]
    pub fn into_core(self) -> HostCore {
        self.core
    }

    /// First absolute cadence deadline, once startup has been observed.
    #[must_use]
    pub const fn next_deadline(&self) -> Option<ManualInstant> {
        self.next_deadline
    }

    /// Last successfully observed monotonic instant.
    #[must_use]
    pub const fn last_observed(&self) -> Option<ManualInstant> {
        self.last_observed
    }

    /// Total fixed cadence deadlines crossed by successful observations.
    #[must_use]
    pub const fn total_scheduled_deadlines(&self) -> u64 {
        self.total_scheduled_deadlines
    }

    /// Total fixed cadence opportunities discarded by bounded catch-up.
    #[must_use]
    pub const fn total_scheduled_deadlines_skipped(&self) -> u64 {
        self.total_scheduled_deadlines_skipped
    }

    /// Submit one envelope through the exact same ordered local host port.
    pub fn submit(&mut self, envelope: CommandEnvelope) -> Result<CommandStatus, HostAccessError> {
        self.port.submit(envelope)
    }

    /// Admit or reuse the host-owned ordered shutdown envelope.
    pub fn request_shutdown(&mut self) -> Result<CommandStatus, HostAccessError> {
        self.core.request_shutdown()
    }

    /// Retry the exact retained journal allocation after an explicit ready wake.
    pub fn retry_retained_journal(&mut self) -> Result<Option<JournalAdmission>, HostAccessError> {
        self.core.retry_retained_journal()
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    fn record_panicked_boundary(&mut self, message: &str) -> Result<(), HostAccessError> {
        self.core.record_panicked_boundary(message)
    }

    /// Current platform scheduling interest derived by the exact host.
    #[must_use]
    pub fn drive_interest(&self) -> HostDriveInterest {
        self.core.drive_interest()
    }

    /// Drive one observation without deriving time from repaint or client count.
    pub fn drive_at(
        &mut self,
        observed_at: ManualInstant,
        trigger: NativeDriveTrigger,
    ) -> Result<NativeDriveReceipt, NativeScheduleError> {
        if let Some(previous) = self.last_observed
            && observed_at < previous
        {
            return Err(NativeScheduleError::ClockMovedBackwards {
                previous: previous.as_nanos(),
                observed: observed_at.as_nanos(),
            });
        }

        let period_nanos = self.core.tick_period_nanos();
        let (scheduled_deadlines_elapsed, next_deadline) = match self.next_deadline {
            None => {
                let next = observed_at.as_nanos().checked_add(period_nanos).ok_or(
                    NativeScheduleError::DeadlineOverflow {
                        at: observed_at.as_nanos(),
                        period_nanos,
                    },
                )?;
                (0, ManualInstant::from_nanos(next))
            }
            Some(deadline) if observed_at < deadline => (0, deadline),
            Some(deadline) => {
                let elapsed = observed_at.as_nanos().saturating_sub(deadline.as_nanos());
                let crossed = elapsed
                    .checked_div(period_nanos)
                    .and_then(|whole| whole.checked_add(1))
                    .ok_or(NativeScheduleError::DeadlineOverflow {
                        at: deadline.as_nanos(),
                        period_nanos,
                    })?;
                let advance = period_nanos.checked_mul(crossed).ok_or(
                    NativeScheduleError::DeadlineOverflow {
                        at: deadline.as_nanos(),
                        period_nanos,
                    },
                )?;
                let next = deadline.as_nanos().checked_add(advance).ok_or(
                    NativeScheduleError::DeadlineOverflow {
                        at: deadline.as_nanos(),
                        period_nanos,
                    },
                )?;
                (crossed, ManualInstant::from_nanos(next))
            }
        };

        let host = self.core.drive(observed_at)?;
        let catch_up_limit =
            u64::try_from(self.core.max_automatic_steps_per_drive()).unwrap_or(u64::MAX);
        let scheduled_deadlines_skipped =
            scheduled_deadlines_elapsed.saturating_sub(catch_up_limit);
        self.last_observed = Some(observed_at);
        self.next_deadline = Some(next_deadline);
        self.total_scheduled_deadlines = self
            .total_scheduled_deadlines
            .saturating_add(scheduled_deadlines_elapsed);
        self.total_scheduled_deadlines_skipped = self
            .total_scheduled_deadlines_skipped
            .saturating_add(scheduled_deadlines_skipped);

        Ok(NativeDriveReceipt {
            trigger,
            observed_at,
            scheduled_deadlines_elapsed,
            scheduled_deadlines_skipped,
            next_deadline,
            host,
        })
    }
}

#[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
#[allow(
    clippy::future_not_send,
    reason = "the sole-owner HostCore is intentionally !Send and is driven only by Runtime::block_on"
)]
mod asupersync_runner {
    use super::{
        CommandEnvelope, FixedDeadlineHost, HostDriveInterest, ManualInstant, NativeDriveReceipt,
        NativeDriveTrigger, NativeScheduleError,
    };
    use crate::{ApplicationState, CommandId, HostFault, HostHealth, JournalBatchId};
    use asupersync::Cx;
    use asupersync::channel::mpsc::{self, Receiver, RecvError, SendError, Sender};
    use asupersync::runtime::{Runtime, RuntimeBuilder};
    use asupersync::time::sleep_until;
    use asupersync::types::{CancelKind, Time};
    use std::any::Any;
    use std::future::{Future, poll_fn};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Poll, Waker};
    use thiserror::Error;

    const DEFAULT_INGRESS_CAPACITY: usize = 64;
    const DEFAULT_MAINTENANCE_PERIOD_NANOS: u64 = 50_000_000;
    const DEFAULT_SHUTDOWN_TIMEOUT_NANOS: u64 = 30_000_000_000;
    const DEFAULT_MAX_SAME_INSTANT_DRIVES: usize = 4;

    #[derive(Debug)]
    enum NativeMessage {
        Command(CommandEnvelope),
        Wake,
        JournalReady,
    }

    struct NativeControlState {
        active_cx: Mutex<Option<Cx>>,
        runner_waker: Mutex<Option<Waker>>,
        command_ingress_open: AtomicBool,
        runner_closed: AtomicBool,
        cancel_requested: AtomicBool,
        wake_enqueued: AtomicBool,
        journal_wake_enqueued: AtomicBool,
        owner_wait_generation: AtomicU64,
    }

    impl NativeControlState {
        const fn new() -> Self {
            Self {
                active_cx: Mutex::new(None),
                runner_waker: Mutex::new(None),
                command_ingress_open: AtomicBool::new(true),
                runner_closed: AtomicBool::new(false),
                cancel_requested: AtomicBool::new(false),
                wake_enqueued: AtomicBool::new(false),
                journal_wake_enqueued: AtomicBool::new(false),
                owner_wait_generation: AtomicU64::new(0),
            }
        }

        fn active_cx(&self) -> std::sync::MutexGuard<'_, Option<Cx>> {
            self.active_cx
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }

        fn runner_waker(&self) -> std::sync::MutexGuard<'_, Option<Waker>> {
            self.runner_waker
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }

        fn register_runner_waker(&self, waker: &Waker) {
            let mut slot = self.runner_waker();
            if slot
                .as_ref()
                .is_none_or(|registered| !registered.will_wake(waker))
            {
                *slot = Some(waker.clone());
            }
        }

        fn begin_owner_wait(&self) {
            let _ = self.owner_wait_generation.fetch_add(1, Ordering::AcqRel);
        }

        fn wake_runner(&self) {
            let waker = self.runner_waker().take();
            if let Some(waker) = waker {
                waker.wake();
            }
        }

        fn clear_runner_waker(&self) {
            drop(self.runner_waker().take());
        }
    }

    struct ActiveCxGuard {
        state: Arc<NativeControlState>,
    }

    impl ActiveCxGuard {
        fn install(state: Arc<NativeControlState>, cx: &Cx) -> Self {
            *state.active_cx() = Some(cx.clone());
            Self { state }
        }
    }

    impl Drop for ActiveCxGuard {
        fn drop(&mut self) {
            *self.state.active_cx() = None;
            self.state.clear_runner_waker();
        }
    }

    /// Bounds and deadlines for one optional Asupersync native runner.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NativeRunnerOptions {
        /// Bounded pre-admission command and wake capacity.
        pub ingress_capacity: usize,
        /// Absolute polling cadence while journal or shutdown work drains.
        pub maintenance_period_nanos: u64,
        /// Maximum monotonic wait for one ordered shutdown barrier.
        pub shutdown_timeout_nanos: u64,
        /// Maximum immediate same-instant maintenance boundaries per wake.
        pub max_same_instant_drives: usize,
    }

    impl Default for NativeRunnerOptions {
        fn default() -> Self {
            Self {
                ingress_capacity: DEFAULT_INGRESS_CAPACITY,
                maintenance_period_nanos: DEFAULT_MAINTENANCE_PERIOD_NANOS,
                shutdown_timeout_nanos: DEFAULT_SHUTDOWN_TIMEOUT_NANOS,
                max_same_instant_drives: DEFAULT_MAX_SAME_INSTANT_DRIVES,
            }
        }
    }

    /// Invalid bound supplied to [`NativeRunner::new`].
    #[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
    pub enum NativeRunnerOptionsError {
        /// Bounded ingress must retain at least one message.
        #[error("native ingress_capacity must be nonzero")]
        EmptyIngress,
        /// Maintenance polling must have a positive duration.
        #[error("native maintenance_period_nanos must be nonzero")]
        MissingMaintenancePeriod,
        /// Ordered shutdown must have a positive retry window.
        #[error("native shutdown_timeout_nanos must be nonzero")]
        MissingShutdownTimeout,
        /// Same-instant pumping must permit at least one boundary.
        #[error("native max_same_instant_drives must be nonzero")]
        DisabledSameInstantDrive,
    }

    impl NativeRunnerOptions {
        const fn validate(self) -> Result<Self, NativeRunnerOptionsError> {
            if self.ingress_capacity == 0 {
                return Err(NativeRunnerOptionsError::EmptyIngress);
            }
            if self.maintenance_period_nanos == 0 {
                return Err(NativeRunnerOptionsError::MissingMaintenancePeriod);
            }
            if self.shutdown_timeout_nanos == 0 {
                return Err(NativeRunnerOptionsError::MissingShutdownTimeout);
            }
            if self.max_same_instant_drives == 0 {
                return Err(NativeRunnerOptionsError::DisabledSameInstantDrive);
            }
            Ok(self)
        }
    }

    /// Exact envelope retained when native ingress cannot accept it.
    #[derive(Debug, Error)]
    pub enum NativeIngressError {
        /// The bounded native ingress is currently full.
        #[error("native ingress is full")]
        Full(CommandEnvelope),
        /// The runner closed ingress for ordered shutdown.
        #[error("native ingress is closed")]
        Closed(CommandEnvelope),
    }

    impl NativeIngressError {
        /// Recover the exact envelope that did not enter native ingress.
        #[must_use]
        pub fn into_envelope(self) -> CommandEnvelope {
            match self {
                Self::Full(envelope) | Self::Closed(envelope) => envelope,
            }
        }
    }

    /// Result of a level-triggered non-command wake request.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum NativeWakeResult {
        /// One wake token entered the bounded ingress.
        Enqueued,
        /// An existing wake or command already guarantees an observation.
        Coalesced,
        /// The runner closed and disconnected its ingress.
        Closed,
    }

    /// Cloneable, thread-safe producer for the owner-pinned native runner.
    ///
    /// Successful `try_submit` means only that the envelope entered bounded
    /// native ingress. Host admission and [`CommandStatus`](crate::CommandStatus)
    /// remain authoritative only after the owner processes it.
    #[derive(Clone)]
    pub struct NativeControl {
        sender: Sender<NativeMessage>,
        state: Arc<NativeControlState>,
    }

    impl NativeControl {
        /// Try to enqueue one exact command without waiting.
        ///
        /// # Panics
        ///
        /// Panics only if the private channel returns a different internal
        /// message variant than the command supplied to that same call.
        pub fn try_submit(&self, envelope: CommandEnvelope) -> Result<(), NativeIngressError> {
            if !self.state.command_ingress_open.load(Ordering::Acquire) {
                return Err(NativeIngressError::Closed(envelope));
            }
            match self.sender.try_send(NativeMessage::Command(envelope)) {
                Ok(()) => {
                    self.state.wake_runner();
                    Ok(())
                }
                Err(SendError::Full(NativeMessage::Command(envelope))) => {
                    Err(NativeIngressError::Full(envelope))
                }
                // Pinned Asupersync 0.3.6 `try_send` has no cancellation
                // context, but the shared error enum still requires this
                // exhaustive defensive arm.
                Err(SendError::Disconnected(NativeMessage::Command(envelope)))
                | Err(SendError::Cancelled(NativeMessage::Command(envelope))) => {
                    Err(NativeIngressError::Closed(envelope))
                }
                Err(SendError::Full(_) | SendError::Disconnected(_) | SendError::Cancelled(_)) => {
                    unreachable!("try_submit sent only a command message")
                }
            }
        }

        /// Coalesce a synthetic observation wake.
        #[must_use]
        pub fn wake(&self) -> NativeWakeResult {
            self.coalesced_wake(false)
        }

        /// Notify the owner that a retained batch or receipt may progress.
        #[must_use]
        pub fn journal_ready(&self) -> NativeWakeResult {
            self.coalesced_wake(true)
        }

        fn coalesced_wake(&self, journal: bool) -> NativeWakeResult {
            if self.state.runner_closed.load(Ordering::Acquire) {
                return NativeWakeResult::Closed;
            }
            let flag = if journal {
                &self.state.journal_wake_enqueued
            } else {
                &self.state.wake_enqueued
            };
            if flag.swap(true, Ordering::AcqRel) {
                self.state.wake_runner();
                self.sender.wake_receiver();
                return NativeWakeResult::Coalesced;
            }
            let message = if journal {
                NativeMessage::JournalReady
            } else {
                NativeMessage::Wake
            };
            match self.sender.try_send(message) {
                Ok(()) => {
                    self.state.wake_runner();
                    NativeWakeResult::Enqueued
                }
                Err(SendError::Full(_)) => {
                    if !journal {
                        flag.store(false, Ordering::Release);
                    }
                    self.state.wake_runner();
                    self.sender.wake_receiver();
                    NativeWakeResult::Coalesced
                }
                Err(SendError::Disconnected(_) | SendError::Cancelled(_)) => {
                    flag.store(false, Ordering::Release);
                    NativeWakeResult::Closed
                }
            }
        }

        /// Request structured cancellation and wake any blocked receiver.
        ///
        /// Returns `true` only for the first request. The owner converts the
        /// signal into one stable ordered shutdown envelope. Cancellation is
        /// boundary-cooperative: an already-running synchronous world step
        /// completes and publishes its exact outcome before shutdown applies.
        #[must_use]
        pub fn cancel(&self) -> bool {
            let first = !self.state.cancel_requested.swap(true, Ordering::AcqRel);
            let active_cx = self.state.active_cx().clone();
            if let Some(cx) = active_cx {
                cx.cancel_with(
                    CancelKind::Shutdown,
                    Some("ScriptBots native lifecycle cancellation"),
                );
            }
            self.state.wake_runner();
            self.sender.wake_receiver();
            first
        }

        /// Whether structured cancellation has been requested.
        #[must_use]
        pub fn is_cancel_requested(&self) -> bool {
            self.state.cancel_requested.load(Ordering::Acquire)
        }

        /// Whether the owner root future has registered its current wait waker.
        ///
        /// This is diagnostic state only; it is not a command-admission or
        /// lifecycle synchronization guarantee.
        #[must_use]
        pub fn is_owner_waiting(&self) -> bool {
            self.state.runner_waker().is_some()
        }

        /// Monotonic count of distinct owner wait registrations.
        ///
        /// This diagnostic counter lets supervisors distinguish a fresh wait
        /// after a wake from the wait that wake originally interrupted. It is
        /// not a command admission or scientific revision.
        #[must_use]
        pub fn owner_wait_generation(&self) -> u64 {
            self.state.owner_wait_generation.load(Ordering::Acquire)
        }
    }

    /// Cumulative, bounded-memory native lifecycle telemetry.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct NativeLifecycleMetrics {
        /// Root runtime invocations, including retries after typed timeout.
        pub runtime_runs: u64,
        /// Exact `HostCore` drive boundaries.
        pub drive_calls: u64,
        /// Command envelopes removed from native ingress.
        pub command_wakes: u64,
        /// Coalesced synthetic wake tokens processed.
        pub synthetic_wakes: u64,
        /// Explicit journal-ready tokens processed.
        pub journal_wakes: u64,
        /// Fixed cadence deadline wakes.
        pub deadline_wakes: u64,
        /// Structured cancellation observations.
        pub cancellation_wakes: u64,
        /// Immediate receipt/shutdown maintenance boundaries.
        pub same_instant_maintenance_drives: u64,
        /// Fixed cadence deadlines crossed.
        pub scheduled_deadlines_elapsed: u64,
        /// Fixed cadence opportunities discarded by bounded catch-up.
        pub scheduled_deadlines_skipped: u64,
        /// Host automatic science opportunities discarded by bounded catch-up.
        pub automatic_steps_skipped: u64,
        /// Host-owned shutdown envelopes created or reused.
        pub shutdown_requests: u64,
        /// Simulation or scheduler child tasks spawned by this runner.
        pub owned_tasks_started: u64,
        /// Owned child tasks explicitly joined before return.
        pub owned_tasks_joined: u64,
    }

    /// Clean terminal result of one native lifecycle.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum NativeRunOutcome {
        /// An explicit ordered Shutdown command completed.
        Stopped {
            /// Stable host command that formed the durability barrier.
            shutdown_command_id: CommandId,
        },
        /// Asupersync cancellation drained through ordered shutdown.
        Cancelled {
            /// Stable host command that formed the durability barrier.
            shutdown_command_id: CommandId,
        },
        /// Losing every producer triggered fail-safe ordered shutdown.
        ControllerDisconnected {
            /// Stable host command that formed the durability barrier.
            shutdown_command_id: CommandId,
        },
    }

    /// Typed failure that leaves [`NativeRunner`] and its exact `HostCore` owned
    /// by the caller for diagnostics or an explicit retry.
    #[derive(Debug, Error)]
    pub enum NativeRunError {
        /// Asupersync runtime construction failed before the host moved.
        #[error("native Asupersync runtime construction failed: {message}")]
        RuntimeBuild {
            /// Runtime diagnostic.
            message: String,
        },
        /// `Runtime::block_on` did not install its documented ambient context.
        #[error("native Asupersync runtime did not install a Cx")]
        MissingContext,
        /// Fixed-deadline or `HostCore` boundary failure.
        #[error(transparent)]
        Schedule(#[from] NativeScheduleError),
        /// Host journal or scientific state reached a queryable fault.
        #[error("native host faulted during lifecycle: {fault:?}")]
        HostFault {
            /// Exact published host fault.
            fault: HostFault,
        },
        /// The ordered shutdown barrier remains retryable after its deadline.
        #[error("native shutdown timed out after {waited_nanos}ns")]
        ShutdownTimedOut {
            /// Monotonic duration already spent draining.
            waited_nanos: u64,
            /// Exact retained batch when admission itself remains blocked.
            pending_batch: Option<JournalBatchId>,
            /// Last published host health.
            health: HostHealth,
        },
        /// A panic was caught at the root boundary; no scheduler task detached.
        #[error("native lifecycle panicked: {message}")]
        Panicked {
            /// Panic payload normalized for diagnostics.
            message: String,
        },
        /// The host stopped, but at least one racing envelope could not receive
        /// its terminal status and remains queryable on the runner.
        #[error(
            "native terminal ingress drain retained {unresolved_envelopes} unresolved envelope(s): {message}"
        )]
        TerminalDrainFailed {
            /// Exact envelopes retained by [`NativeRunner::unresolved_envelopes`].
            unresolved_envelopes: usize,
            /// First host protocol diagnostic observed while reconciling them.
            message: String,
        },
    }

    enum DriverWake {
        Message(NativeMessage),
        Immediate,
        Deadline,
        Cancellation,
        Disconnected,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ControllerState {
        Connected,
        DisconnectPending,
        Disconnected,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TerminalDrainFailure {
        unresolved_envelopes: usize,
        message: String,
    }

    /// Owner-pinned Asupersync lifecycle for one [`FixedDeadlineHost`].
    ///
    /// The runner spawns no simulation or scheduler task. Its root future is
    /// driven directly by `Runtime::block_on`, and all failures leave `host`
    /// retained inside this value.
    pub struct NativeRunner {
        host: FixedDeadlineHost,
        receiver: Receiver<NativeMessage>,
        state: Arc<NativeControlState>,
        options: NativeRunnerOptions,
        metrics: NativeLifecycleMetrics,
        shutdown_started_at: Option<ManualInstant>,
        provisional_shutdown_started_at: Option<ManualInstant>,
        shutdown_command_id: Option<CommandId>,
        cancellation_observed: bool,
        controller_state: ControllerState,
        receiver_closed: bool,
        maintenance_deadline: Option<ManualInstant>,
        terminal: Option<NativeRunOutcome>,
        panic_message: Option<String>,
        terminal_drain_failure: Option<TerminalDrainFailure>,
        unresolved_envelopes: Vec<CommandEnvelope>,
    }

    impl NativeRunner {
        /// Construct a bounded native ingress and owner-pinned runner.
        pub fn new(
            host: FixedDeadlineHost,
            options: NativeRunnerOptions,
        ) -> Result<(Self, NativeControl), NativeRunnerOptionsError> {
            let options = options.validate()?;
            let (sender, receiver) = mpsc::channel(options.ingress_capacity);
            let state = Arc::new(NativeControlState::new());
            let control = NativeControl {
                sender,
                state: Arc::clone(&state),
            };
            Ok((
                Self {
                    host,
                    receiver,
                    state,
                    options,
                    metrics: NativeLifecycleMetrics::default(),
                    shutdown_started_at: None,
                    provisional_shutdown_started_at: None,
                    shutdown_command_id: None,
                    cancellation_observed: false,
                    controller_state: ControllerState::Connected,
                    receiver_closed: false,
                    maintenance_deadline: None,
                    terminal: None,
                    panic_message: None,
                    terminal_drain_failure: None,
                    unresolved_envelopes: Vec::new(),
                },
                control,
            ))
        }

        /// Immutable access to the retained owner-pinned host.
        #[must_use]
        pub const fn host(&self) -> &FixedDeadlineHost {
            &self.host
        }

        /// Cumulative lifecycle telemetry.
        #[must_use]
        pub const fn metrics(&self) -> NativeLifecycleMetrics {
            self.metrics
        }

        /// Exact bounded envelopes retained when panic or terminal protocol
        /// failure made truthful host status reconciliation impossible.
        #[must_use]
        pub fn unresolved_envelopes(&self) -> &[CommandEnvelope] {
            &self.unresolved_envelopes
        }

        /// Consume the runner and recover the exact retained host together
        /// with every envelope whose terminal status could not be reconciled.
        #[must_use]
        pub fn into_parts(self) -> (FixedDeadlineHost, Vec<CommandEnvelope>) {
            (self.host, self.unresolved_envelopes)
        }

        #[cfg(test)]
        /// Seed the sticky terminal-drain record for preflight retry proof.
        pub(super) fn inject_terminal_drain_failure_for_test(
            &mut self,
            envelope: CommandEnvelope,
            message: &str,
        ) {
            self.seal_runner_ingress();
            self.unresolved_envelopes.push(envelope);
            let _ = self.cache_terminal_drain_failure(message.to_owned());
        }

        /// Build the selected current-thread Asupersync runtime and run until
        /// clean stop or a typed retryable/terminal failure.
        pub fn run_until_terminal(&mut self) -> Result<NativeRunOutcome, NativeRunError> {
            if let Some(result) = self.cached_terminal_result() {
                return result;
            }
            let runtime = RuntimeBuilder::current_thread()
                .enable_time()
                .enable_platform_reactor(false)
                .build()
                .map_err(|error| NativeRunError::RuntimeBuild {
                    message: error.to_string(),
                })?;
            self.run_on_runtime(&runtime)
        }

        /// Run on a caller-configured runtime, including a deterministic
        /// virtual-clock runtime used by conformance tests.
        pub fn run_on_runtime(
            &mut self,
            runtime: &Runtime,
        ) -> Result<NativeRunOutcome, NativeRunError> {
            if let Some(result) = self.cached_terminal_result() {
                return result;
            }
            self.metrics.runtime_runs = self.metrics.runtime_runs.saturating_add(1);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.block_on(async {
                    let cx = Cx::current().ok_or(NativeRunError::MissingContext)?;
                    self.run_loop(cx).await
                })
            }));
            match result {
                Ok(result) => {
                    if matches!(&result, Err(NativeRunError::HostFault { .. })) {
                        self.close_runner_after_failure();
                    }
                    result
                }
                Err(payload) => {
                    let mut message = panic_message(payload.as_ref());
                    if let Err(error) = self.host.record_panicked_boundary(&message) {
                        message = format!("{message}; panic bookkeeping failed: {error}");
                    }
                    self.panic_message = Some(message.clone());
                    self.close_runner_after_failure();
                    Err(NativeRunError::Panicked { message })
                }
            }
        }

        fn cached_terminal_result(&self) -> Option<Result<NativeRunOutcome, NativeRunError>> {
            if let Some(outcome) = self.terminal {
                return Some(Ok(outcome));
            }
            if let Some(message) = &self.panic_message {
                return Some(Err(NativeRunError::Panicked {
                    message: message.clone(),
                }));
            }
            self.terminal_drain_failure.as_ref().map(|failure| {
                Err(NativeRunError::TerminalDrainFailed {
                    unresolved_envelopes: failure.unresolved_envelopes,
                    message: failure.message.clone(),
                })
            })
        }

        async fn run_loop(&mut self, cx: Cx) -> Result<NativeRunOutcome, NativeRunError> {
            let _active = ActiveCxGuard::install(Arc::clone(&self.state), &cx);
            if self.state.cancel_requested.load(Ordering::Acquire) {
                cx.cancel_with(
                    CancelKind::Shutdown,
                    Some("ScriptBots native lifecycle cancellation"),
                );
            }

            let initial_trigger = if self.host.last_observed().is_none() {
                NativeDriveTrigger::Startup
            } else {
                NativeDriveTrigger::Maintenance
            };
            let now = manual_now(&cx);
            self.observe(&cx, now, initial_trigger, None)?;

            loop {
                if let Some(outcome) = self.check_terminal(manual_now(&cx))? {
                    self.close_runner_cleanly()?;
                    self.terminal = Some(outcome);
                    return Ok(outcome);
                }

                let wake = self.wait_for_work(&cx).await;
                let now = manual_now(&cx);
                let (trigger, first_message) = match wake {
                    DriverWake::Message(message) => {
                        (NativeDriveTrigger::SyntheticWake, Some(message))
                    }
                    DriverWake::Immediate => (NativeDriveTrigger::Maintenance, None),
                    DriverWake::Deadline => {
                        self.metrics.deadline_wakes = self.metrics.deadline_wakes.saturating_add(1);
                        (NativeDriveTrigger::Deadline, None)
                    }
                    DriverWake::Cancellation => {
                        self.metrics.cancellation_wakes =
                            self.metrics.cancellation_wakes.saturating_add(1);
                        (NativeDriveTrigger::Cancellation, None)
                    }
                    DriverWake::Disconnected => {
                        self.controller_state = ControllerState::DisconnectPending;
                        (NativeDriveTrigger::Cancellation, None)
                    }
                };
                self.observe(&cx, now, trigger, first_message)?;
            }
        }

        fn observe(
            &mut self,
            cx: &Cx,
            now: ManualInstant,
            mut trigger: NativeDriveTrigger,
            first_message: Option<NativeMessage>,
        ) -> Result<(), NativeRunError> {
            if self.should_cancel(cx) {
                self.begin_shutdown(now, true)?;
                trigger = trigger.combine(NativeDriveTrigger::Cancellation);
            } else if self.controller_state == ControllerState::Disconnected {
                self.begin_shutdown(now, false)?;
                trigger = trigger.combine(NativeDriveTrigger::Cancellation);
            }

            let mut remaining_messages = self.options.ingress_capacity;
            if let Some(message) = first_message {
                trigger = self.process_message(message, trigger)?;
                remaining_messages = remaining_messages.saturating_sub(1);
            }
            self.drain_available(now, &mut trigger, remaining_messages)?;
            trigger = self.consume_latched_journal_ready(trigger)?;
            self.track_provisional_shutdown(now);

            if self.should_cancel(cx) {
                self.begin_shutdown(now, true)?;
                trigger = trigger.combine(NativeDriveTrigger::Cancellation);
            }
            if self.controller_state == ControllerState::DisconnectPending
                && self.host.core().shutdown_command_id().is_none()
            {
                self.controller_state = ControllerState::Disconnected;
                self.begin_shutdown(now, false)?;
                trigger = trigger.combine(NativeDriveTrigger::Cancellation);
            }

            self.drive(now, trigger)?;
            let cancellation_after_drive = self.should_cancel(cx);
            self.reconcile_provisional_shutdown();
            self.reconcile_controller_disconnect();
            let forced_stop =
                cancellation_after_drive || self.controller_state != ControllerState::Connected;
            if self.shutdown_is_applied()
                || (forced_stop && self.host.core().shutdown_command_id().is_some())
            {
                self.begin_shutdown(now, cancellation_after_drive)?;
                let limit = self.options.ingress_capacity;
                self.drain_available(now, &mut trigger, limit)?;
                self.consume_latched_journal_ready(trigger)?;
            }

            self.replace_missing_shutdown_if_required(now, cancellation_after_drive, trigger)?;
            self.pump_same_instant_maintenance(now)?;
            self.reconcile_provisional_shutdown();
            self.reconcile_controller_disconnect();
            if self.replace_missing_shutdown_if_required(now, cancellation_after_drive, trigger)? {
                self.pump_same_instant_maintenance(now)?;
                self.reconcile_provisional_shutdown();
                self.reconcile_controller_disconnect();
            }
            Ok(())
        }

        fn reconcile_controller_disconnect(&mut self) {
            if self.controller_state != ControllerState::DisconnectPending {
                return;
            }
            match self.host.core().shutdown_command_status() {
                Some(status) if matches!(status.application(), ApplicationState::Admitted) => {}
                Some(status) if matches!(status.application(), ApplicationState::Applied(_)) => {
                    self.controller_state = ControllerState::Connected;
                }
                Some(_) | None => {
                    self.controller_state = ControllerState::Disconnected;
                }
            }
        }

        fn shutdown_is_applied(&self) -> bool {
            self.host
                .core()
                .shutdown_command_status()
                .is_some_and(|status| matches!(status.application(), ApplicationState::Applied(_)))
        }

        fn track_provisional_shutdown(&mut self, now: ManualInstant) {
            if self.shutdown_started_at.is_none()
                && self.provisional_shutdown_started_at.is_none()
                && self
                    .host
                    .core()
                    .shutdown_command_status()
                    .is_some_and(|status| {
                        matches!(status.application(), ApplicationState::Admitted)
                    })
            {
                self.provisional_shutdown_started_at = Some(now);
            }
        }

        fn reconcile_provisional_shutdown(&mut self) {
            if self.provisional_shutdown_started_at.is_none() {
                return;
            }
            match self.host.core().shutdown_command_status() {
                Some(status)
                    if matches!(
                        status.application(),
                        ApplicationState::Admitted | ApplicationState::Applied(_)
                    ) => {}
                Some(_) | None if self.controller_state == ControllerState::Connected => {
                    self.provisional_shutdown_started_at = None;
                }
                Some(_) | None => {}
            }
        }

        const fn shutdown_wait_started_at(&self) -> Option<ManualInstant> {
            match (
                self.shutdown_started_at,
                self.provisional_shutdown_started_at,
            ) {
                (Some(started), Some(provisional)) => {
                    if started.as_nanos() <= provisional.as_nanos() {
                        Some(started)
                    } else {
                        Some(provisional)
                    }
                }
                (Some(started), None) => Some(started),
                (None, provisional) => provisional,
            }
        }

        fn replace_missing_shutdown_if_required(
            &mut self,
            now: ManualInstant,
            cancellation: bool,
            mut trigger: NativeDriveTrigger,
        ) -> Result<bool, NativeRunError> {
            let ordered_stop_required = cancellation
                || self.controller_state == ControllerState::Disconnected
                || self.shutdown_started_at.is_some();
            if !ordered_stop_required || self.host.core().shutdown_command_id().is_some() {
                return Ok(false);
            }
            // A conditional external Shutdown can close the host gate at
            // admission and then be rejected at its ordered CAS boundary.
            // Replace a selected provisional identity with the unconditional
            // host-owned shutdown while preserving the original stop cause.
            self.shutdown_command_id = None;
            self.begin_shutdown(now, cancellation)?;
            let limit = self.options.ingress_capacity;
            self.drain_available(now, &mut trigger, limit)?;
            self.consume_latched_journal_ready(trigger)?;
            self.drive(now, NativeDriveTrigger::Cancellation)?;
            Ok(true)
        }

        fn pump_same_instant_maintenance(
            &mut self,
            now: ManualInstant,
        ) -> Result<(), NativeRunError> {
            if !matches!(self.host.drive_interest(), HostDriveInterest::Draining) {
                return Ok(());
            }
            for _ in 0..self.options.max_same_instant_drives {
                let maintenance = self.drive(now, NativeDriveTrigger::Maintenance)?;
                self.metrics.same_instant_maintenance_drives = self
                    .metrics
                    .same_instant_maintenance_drives
                    .saturating_add(1);
                let progressed = maintenance.host.commands_completed > 0
                    || maintenance.host.scientific_steps > 0
                    || maintenance.host.snapshots_published > 0
                    || maintenance.host.events_published > 0;
                if !progressed
                    || !matches!(
                        self.host.drive_interest(),
                        HostDriveInterest::ReadyNow | HostDriveInterest::Draining
                    )
                {
                    break;
                }
            }
            Ok(())
        }

        fn drain_available(
            &mut self,
            now: ManualInstant,
            trigger: &mut NativeDriveTrigger,
            limit: usize,
        ) -> Result<(), NativeRunError> {
            for _ in 0..limit {
                match self.receiver.try_recv() {
                    Ok(message) => *trigger = self.process_message(message, *trigger)?,
                    Err(RecvError::Empty) => return Ok(()),
                    Err(RecvError::Disconnected) => {
                        if !self.receiver_closed
                            && self.controller_state != ControllerState::Disconnected
                            && !self.shutdown_is_applied()
                        {
                            self.controller_state = ControllerState::DisconnectPending;
                            *trigger = trigger.combine(NativeDriveTrigger::Cancellation);
                        }
                        return Ok(());
                    }
                    Err(RecvError::Cancelled) => {
                        self.begin_shutdown(now, true)?;
                        *trigger = trigger.combine(NativeDriveTrigger::Cancellation);
                        return Ok(());
                    }
                }
            }
            Ok(())
        }

        fn process_message(
            &mut self,
            message: NativeMessage,
            trigger: NativeDriveTrigger,
        ) -> Result<NativeDriveTrigger, NativeRunError> {
            match message {
                NativeMessage::Command(envelope) => {
                    self.metrics.command_wakes = self.metrics.command_wakes.saturating_add(1);
                    self.host
                        .submit(envelope)
                        .map_err(NativeScheduleError::from)?;
                    Ok(trigger.combine(NativeDriveTrigger::Command))
                }
                NativeMessage::Wake => {
                    self.state.wake_enqueued.store(false, Ordering::Release);
                    self.metrics.synthetic_wakes = self.metrics.synthetic_wakes.saturating_add(1);
                    Ok(trigger.combine(NativeDriveTrigger::SyntheticWake))
                }
                NativeMessage::JournalReady => {
                    self.state
                        .journal_wake_enqueued
                        .store(false, Ordering::Release);
                    self.metrics.journal_wakes = self.metrics.journal_wakes.saturating_add(1);
                    self.host
                        .retry_retained_journal()
                        .map_err(NativeScheduleError::from)?;
                    Ok(trigger.combine(NativeDriveTrigger::JournalReady))
                }
            }
        }

        fn consume_latched_journal_ready(
            &mut self,
            trigger: NativeDriveTrigger,
        ) -> Result<NativeDriveTrigger, NativeRunError> {
            if !self
                .state
                .journal_wake_enqueued
                .swap(false, Ordering::AcqRel)
            {
                return Ok(trigger);
            }
            self.metrics.journal_wakes = self.metrics.journal_wakes.saturating_add(1);
            self.host
                .retry_retained_journal()
                .map_err(NativeScheduleError::from)?;
            Ok(trigger.combine(NativeDriveTrigger::JournalReady))
        }

        fn drive(
            &mut self,
            now: ManualInstant,
            trigger: NativeDriveTrigger,
        ) -> Result<NativeDriveReceipt, NativeRunError> {
            let receipt = self.host.drive_at(now, trigger)?;
            self.metrics.drive_calls = self.metrics.drive_calls.saturating_add(1);
            self.metrics.scheduled_deadlines_elapsed = self
                .metrics
                .scheduled_deadlines_elapsed
                .saturating_add(receipt.scheduled_deadlines_elapsed);
            self.metrics.scheduled_deadlines_skipped = self
                .metrics
                .scheduled_deadlines_skipped
                .saturating_add(receipt.scheduled_deadlines_skipped);
            self.metrics.automatic_steps_skipped = self
                .metrics
                .automatic_steps_skipped
                .saturating_add(receipt.host.automatic_steps_skipped);
            Ok(receipt)
        }

        fn should_cancel(&self, cx: &Cx) -> bool {
            self.state.cancel_requested.load(Ordering::Acquire) || cx.is_cancel_requested()
        }

        fn begin_shutdown(
            &mut self,
            now: ManualInstant,
            cancellation: bool,
        ) -> Result<(), NativeRunError> {
            if cancellation {
                self.cancellation_observed = true;
            }
            if self.shutdown_command_id.is_none() {
                let status = self
                    .host
                    .request_shutdown()
                    .map_err(NativeScheduleError::from)?;
                self.shutdown_command_id = Some(status.command_id());
                self.metrics.shutdown_requests = self.metrics.shutdown_requests.saturating_add(1);
            }
            let started_at = self.provisional_shutdown_started_at.take().unwrap_or(now);
            self.shutdown_started_at = Some(
                self.shutdown_started_at
                    .map_or(started_at, |existing| existing.min(started_at)),
            );
            self.state
                .command_ingress_open
                .store(false, Ordering::Release);
            Ok(())
        }

        fn seal_runner_ingress(&mut self) {
            self.state
                .command_ingress_open
                .store(false, Ordering::Release);
            self.state.runner_closed.store(true, Ordering::Release);
            self.receiver.close();
            self.receiver_closed = true;
            self.state.wake_runner();
        }

        fn close_runner_cleanly(&mut self) -> Result<(), NativeRunError> {
            self.seal_runner_ingress();
            let mut first_error = None;
            while let Ok(message) = self.receiver.try_recv() {
                match message {
                    NativeMessage::Command(envelope) => {
                        self.metrics.command_wakes = self.metrics.command_wakes.saturating_add(1);
                        let retained = envelope.clone();
                        if let Err(error) = self.host.submit(envelope) {
                            first_error.get_or_insert_with(|| error.to_string());
                            self.unresolved_envelopes.push(retained);
                        }
                    }
                    NativeMessage::Wake => {
                        self.state.wake_enqueued.store(false, Ordering::Release);
                        self.metrics.synthetic_wakes =
                            self.metrics.synthetic_wakes.saturating_add(1);
                    }
                    NativeMessage::JournalReady => {
                        self.state
                            .journal_wake_enqueued
                            .store(false, Ordering::Release);
                        self.metrics.journal_wakes = self.metrics.journal_wakes.saturating_add(1);
                    }
                }
            }
            if let Some(message) = first_error {
                return Err(self.cache_terminal_drain_failure(message));
            }
            Ok(())
        }

        fn cache_terminal_drain_failure(&mut self, message: String) -> NativeRunError {
            let unresolved_envelopes = self.unresolved_envelopes.len();
            self.terminal_drain_failure = Some(TerminalDrainFailure {
                unresolved_envelopes,
                message: message.clone(),
            });
            NativeRunError::TerminalDrainFailed {
                unresolved_envelopes,
                message,
            }
        }

        fn close_runner_after_failure(&mut self) {
            self.seal_runner_ingress();
            while let Ok(message) = self.receiver.try_recv() {
                match message {
                    NativeMessage::Command(envelope) => {
                        self.metrics.command_wakes = self.metrics.command_wakes.saturating_add(1);
                        self.unresolved_envelopes.push(envelope);
                    }
                    NativeMessage::Wake => {
                        self.state.wake_enqueued.store(false, Ordering::Release);
                        self.metrics.synthetic_wakes =
                            self.metrics.synthetic_wakes.saturating_add(1);
                    }
                    NativeMessage::JournalReady => {
                        self.state
                            .journal_wake_enqueued
                            .store(false, Ordering::Release);
                        self.metrics.journal_wakes = self.metrics.journal_wakes.saturating_add(1);
                    }
                }
            }
        }

        fn check_terminal(
            &self,
            now: ManualInstant,
        ) -> Result<Option<NativeRunOutcome>, NativeRunError> {
            match self.host.drive_interest() {
                HostDriveInterest::Terminated => {
                    let shutdown_command_id = self
                        .shutdown_command_id
                        .or_else(|| self.host.core().shutdown_command_id())
                        .ok_or_else(|| NativeRunError::HostFault {
                            fault: HostFault::Protocol {
                                code: "missing_shutdown_identity".to_owned(),
                                message: "stopped native host has no shutdown command identity"
                                    .to_owned(),
                            },
                        })?;
                    let outcome = if self.controller_state == ControllerState::Disconnected {
                        NativeRunOutcome::ControllerDisconnected {
                            shutdown_command_id,
                        }
                    } else if self.cancellation_observed {
                        NativeRunOutcome::Cancelled {
                            shutdown_command_id,
                        }
                    } else {
                        NativeRunOutcome::Stopped {
                            shutdown_command_id,
                        }
                    };
                    Ok(Some(outcome))
                }
                HostDriveInterest::Faulted => Err(NativeRunError::HostFault {
                    fault: self
                        .host
                        .core()
                        .health()
                        .fault()
                        .cloned()
                        .unwrap_or_else(|| HostFault::Protocol {
                            code: "missing_fault_detail".to_owned(),
                            message: "host reported faulted scheduling interest without detail"
                                .to_owned(),
                        }),
                }),
                HostDriveInterest::ReadyNow
                | HostDriveInterest::Deadline
                | HostDriveInterest::WakeOnly
                | HostDriveInterest::Draining => {
                    if let Some(started) = self.shutdown_wait_started_at() {
                        let waited_nanos = now.as_nanos().saturating_sub(started.as_nanos());
                        if waited_nanos >= self.options.shutdown_timeout_nanos {
                            return Err(NativeRunError::ShutdownTimedOut {
                                waited_nanos,
                                pending_batch: self
                                    .host
                                    .core()
                                    .pending_journal_batch()
                                    .map(|batch| batch.id()),
                                health: self.host.core().health().clone(),
                            });
                        }
                    }
                    Ok(None)
                }
            }
        }

        async fn wait_for_work(&mut self, cx: &Cx) -> DriverWake {
            match self.host.drive_interest() {
                HostDriveInterest::ReadyNow => DriverWake::Immediate,
                HostDriveInterest::Deadline => {
                    let cadence = self
                        .host
                        .next_deadline()
                        .expect("startup establishes a fixed deadline");
                    let deadline = self.shutdown_wait_started_at().map_or(cadence, |started| {
                        cadence.min(ManualInstant::from_nanos(
                            started
                                .as_nanos()
                                .saturating_add(self.options.shutdown_timeout_nanos),
                        ))
                    });
                    self.wait_message_or_deadline(cx, deadline).await
                }
                HostDriveInterest::WakeOnly => {
                    if let Some(started) = self.shutdown_wait_started_at() {
                        let timeout = ManualInstant::from_nanos(
                            started
                                .as_nanos()
                                .saturating_add(self.options.shutdown_timeout_nanos),
                        );
                        self.wait_message_or_deadline(cx, timeout).await
                    } else {
                        self.wait_message(cx).await
                    }
                }
                HostDriveInterest::Draining => {
                    let now = manual_now(cx);
                    let maintenance = advance_absolute_deadline(
                        self.maintenance_deadline,
                        now,
                        self.options.maintenance_period_nanos,
                    );
                    self.maintenance_deadline = Some(maintenance);
                    let deadline = self
                        .shutdown_wait_started_at()
                        .map_or(maintenance, |started| {
                            let timeout = ManualInstant::from_nanos(
                                started
                                    .as_nanos()
                                    .saturating_add(self.options.shutdown_timeout_nanos),
                            );
                            maintenance.min(timeout)
                        });
                    self.wait_message_or_deadline(cx, deadline).await
                }
                HostDriveInterest::Terminated | HostDriveInterest::Faulted => DriverWake::Deadline,
            }
        }

        async fn wait_message(&mut self, cx: &Cx) -> DriverWake {
            self.state.begin_owner_wait();
            let state = Arc::clone(&self.state);
            let wake = poll_fn(|task_cx| {
                state.register_runner_waker(task_cx.waker());
                if !self.cancellation_observed && self.should_cancel(cx) {
                    return Poll::Ready(DriverWake::Cancellation);
                }
                let received = if self.cancellation_observed {
                    match self.receiver.try_recv() {
                        Ok(message) => Poll::Ready(Ok(message)),
                        Err(RecvError::Empty) => Poll::Pending,
                        Err(error) => Poll::Ready(Err(error)),
                    }
                } else {
                    self.receiver.poll_recv(cx, task_cx)
                };
                match received {
                    Poll::Ready(Ok(message)) => Poll::Ready(DriverWake::Message(message)),
                    Poll::Ready(Err(RecvError::Disconnected)) => {
                        Poll::Ready(DriverWake::Disconnected)
                    }
                    Poll::Ready(Err(RecvError::Cancelled)) => Poll::Ready(DriverWake::Cancellation),
                    Poll::Ready(Err(RecvError::Empty)) | Poll::Pending => Poll::Pending,
                }
            })
            .await;
            self.state.clear_runner_waker();
            wake
        }

        async fn wait_message_or_deadline(
            &mut self,
            cx: &Cx,
            deadline: ManualInstant,
        ) -> DriverWake {
            self.state.begin_owner_wait();
            let mut sleep = Box::pin(sleep_until(Time::from_nanos(deadline.as_nanos())));
            let state = Arc::clone(&self.state);
            let wake = poll_fn(|task_cx| {
                state.register_runner_waker(task_cx.waker());
                if !self.cancellation_observed && self.should_cancel(cx) {
                    return Poll::Ready(DriverWake::Cancellation);
                }
                let received = if self.cancellation_observed {
                    match self.receiver.try_recv() {
                        Ok(message) => Poll::Ready(Ok(message)),
                        Err(RecvError::Empty) => Poll::Pending,
                        Err(error) => Poll::Ready(Err(error)),
                    }
                } else {
                    self.receiver.poll_recv(cx, task_cx)
                };
                match received {
                    Poll::Ready(Ok(message)) => {
                        return Poll::Ready(DriverWake::Message(message));
                    }
                    Poll::Ready(Err(RecvError::Disconnected)) => {
                        // A provisional timeout deliberately leaves ingress
                        // reversible. Producer loss must wake once so observe
                        // can promote that same deadline into definitive stop.
                        if self.shutdown_started_at.is_none() {
                            return Poll::Ready(DriverWake::Disconnected);
                        }
                    }
                    Poll::Ready(Err(RecvError::Cancelled)) => {
                        if !self.cancellation_observed {
                            return Poll::Ready(DriverWake::Cancellation);
                        }
                    }
                    Poll::Ready(Err(RecvError::Empty)) | Poll::Pending => {}
                }
                match sleep.as_mut().poll(task_cx) {
                    Poll::Ready(()) => Poll::Ready(DriverWake::Deadline),
                    Poll::Pending => Poll::Pending,
                }
            })
            .await;
            self.state.clear_runner_waker();
            wake
        }
    }

    fn manual_now(cx: &Cx) -> ManualInstant {
        ManualInstant::from_nanos(cx.now().as_nanos())
    }

    fn advance_absolute_deadline(
        previous: Option<ManualInstant>,
        now: ManualInstant,
        period_nanos: u64,
    ) -> ManualInstant {
        let first = previous.map_or_else(
            || now.as_nanos().saturating_add(period_nanos),
            ManualInstant::as_nanos,
        );
        if first > now.as_nanos() {
            return ManualInstant::from_nanos(first);
        }
        let elapsed = now.as_nanos().saturating_sub(first);
        let crossed = (elapsed / period_nanos).saturating_add(1);
        ManualInstant::from_nanos(first.saturating_add(period_nanos.saturating_mul(crossed)))
    }

    fn panic_message(payload: &(dyn Any + Send)) -> String {
        payload.downcast_ref::<&str>().map_or_else(
            || {
                payload
                    .downcast_ref::<String>()
                    .cloned()
                    .unwrap_or_else(|| "non-string panic payload".to_owned())
            },
            |message| (*message).to_owned(),
        )
    }
}

#[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
pub use asupersync_runner::{
    NativeControl, NativeIngressError, NativeLifecycleMetrics, NativeRunError, NativeRunOutcome,
    NativeRunner, NativeRunnerOptions, NativeRunnerOptionsError, NativeWakeResult,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AppliedCommand, CommandId, EventSequence, HostCommand, HostCoreOptions, HostEvent,
        HostSessionId, JournalBatch, JournalBatchId, JournalPort, JournalReceipt,
        JournalReceiptState, PlaybackSnapshot, ScientificBoundary, ShutdownCommitRequirement,
    };
    use scriptbots_core::{ScriptBotsConfig, Tick, WorldState};
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;
    use std::sync::Arc;

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    use crate::{HostLifecycle, JournalFailure};
    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    use asupersync::runtime::RuntimeBuilder;
    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    use asupersync::time::{TimerDriverHandle, VirtualClock};
    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    use asupersync::types::Time;
    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug, Clone, Copy, Default)]
    enum ReceiptMode {
        #[default]
        Immediate,
        #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
        Never,
        #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
        Failed,
        #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
        Panic,
    }

    #[derive(Default)]
    struct CaptureState {
        batches: Vec<Arc<JournalBatch>>,
        receipts: VecDeque<JournalReceipt>,
        mode: ReceiptMode,
        #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
        cancel_on_science: Option<NativeControl>,
        #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
        enqueue_before_panic: Option<(NativeControl, CommandEnvelope)>,
    }

    struct CaptureJournal {
        state: Rc<RefCell<CaptureState>>,
    }

    impl JournalPort for CaptureJournal {
        fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
            let batch_id = batch.id();
            self.state.borrow_mut().batches.push(Arc::clone(batch));
            #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
            if batch.scientific().is_some()
                && let Some(control) = self.state.borrow_mut().cancel_on_science.take()
            {
                assert!(
                    control.cancel(),
                    "science hook must request first cancellation"
                );
            }
            let mode = self.state.borrow().mode;
            match mode {
                ReceiptMode::Immediate => {
                    self.state
                        .borrow_mut()
                        .receipts
                        .push_back(JournalReceipt::new(
                            batch_id,
                            JournalReceiptState::CommittedVolatile,
                        ));
                }
                #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
                ReceiptMode::Never => {}
                #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
                ReceiptMode::Failed => {
                    self.state
                        .borrow_mut()
                        .receipts
                        .push_back(JournalReceipt::new(
                            batch_id,
                            JournalReceiptState::Failed(JournalFailure {
                                code: "injected_failure".to_owned(),
                                message: "injected native journal failure".to_owned(),
                            }),
                        ));
                }
                #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
                ReceiptMode::Panic => {
                    if let Some((control, envelope)) =
                        self.state.borrow_mut().enqueue_before_panic.take()
                    {
                        control
                            .try_submit(envelope)
                            .expect("panic hook command must remain in native ingress");
                    }
                    panic!("injected native journal panic");
                }
            }
            JournalAdmission::Accepted { batch_id }
        }

        fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
            let count = limit.min(self.state.borrow().receipts.len());
            self.state.borrow_mut().receipts.drain(..count).collect()
        }

        fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
            ShutdownCommitRequirement::CommittedVolatile
        }
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    struct ReadinessJournal {
        ready: Arc<AtomicBool>,
        blocked: Arc<AtomicBool>,
        on_full: Rc<RefCell<Option<NativeControl>>>,
        wake_results: Rc<RefCell<Option<(NativeWakeResult, NativeWakeResult)>>>,
        receipts: VecDeque<JournalReceipt>,
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    impl JournalPort for ReadinessJournal {
        fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
            let batch_id = batch.id();
            if !self.ready.load(Ordering::Acquire) {
                self.blocked.store(true, Ordering::Release);
                if let Some(control) = self.on_full.borrow_mut().take() {
                    self.ready.store(true, Ordering::Release);
                    let wake = control.wake();
                    let journal_ready = control.journal_ready();
                    *self.wake_results.borrow_mut() = Some((wake, journal_ready));
                }
                return JournalAdmission::Full {
                    batch_id,
                    capacity: 1,
                };
            }
            self.receipts.push_back(JournalReceipt::new(
                batch_id,
                JournalReceiptState::CommittedVolatile,
            ));
            JournalAdmission::Accepted { batch_id }
        }

        fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
            let count = limit.min(self.receipts.len());
            self.receipts.drain(..count).collect()
        }

        fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
            ShutdownCommitRequirement::CommittedVolatile
        }
    }

    fn world() -> WorldState {
        world_with_persistence_interval(0)
    }

    fn world_with_persistence_interval(persistence_interval: u32) -> WorldState {
        WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0x4e41_5449_5645),
            persistence_interval,
            ..ScriptBotsConfig::default()
        })
        .expect("deterministic native test world")
    }

    fn options(paused: bool) -> HostCoreOptions {
        HostCoreOptions {
            initial_playback: PlaybackSnapshot {
                paused,
                speed_multiplier: 1.0,
            },
            command_capacity: 32,
            tick_period_nanos: 10,
            max_automatic_steps_per_drive: 4,
        }
    }

    fn captured_host(
        session: u64,
        paused: bool,
        mode: ReceiptMode,
    ) -> (HostCore, Rc<RefCell<CaptureState>>) {
        captured_host_with_options(session, options(paused), mode)
    }

    fn captured_host_with_options(
        session: u64,
        host_options: HostCoreOptions,
        mode: ReceiptMode,
    ) -> (HostCore, Rc<RefCell<CaptureState>>) {
        captured_host_with_world(session, world(), host_options, mode)
    }

    fn captured_host_with_world(
        session: u64,
        world: WorldState,
        host_options: HostCoreOptions,
        mode: ReceiptMode,
    ) -> (HostCore, Rc<RefCell<CaptureState>>) {
        let state = Rc::new(RefCell::new(CaptureState {
            mode,
            ..CaptureState::default()
        }));
        let host = HostCore::with_journal(
            HostSessionId::new(session),
            world,
            host_options,
            Box::new(CaptureJournal {
                state: Rc::clone(&state),
            }),
        )
        .expect("captured native host");
        (host, state)
    }

    fn envelope(sequence: u64, command: HostCommand) -> CommandEnvelope {
        CommandEnvelope::new(CommandId::from_client_sequence(7, sequence), command)
    }

    fn events(port: &mut LocalHostPort) -> Vec<HostEvent> {
        port.events_after(EventSequence::new(0), usize::MAX)
            .expect("host events")
    }

    type JournalTrace = (
        JournalBatchId,
        Option<serde_json::Value>,
        AppliedCommand,
        Option<ScientificBoundary>,
        Option<PersistenceTrace>,
    );

    #[derive(Debug, Clone, PartialEq)]
    struct PersistenceTrace {
        summary: scriptbots_core::TickSummary,
        epoch: u64,
        closed: bool,
        metrics: Vec<scriptbots_core::MetricSample>,
        events: Vec<scriptbots_core::PersistenceEvent>,
        agents: serde_json::Value,
        births: Vec<scriptbots_core::BirthRecord>,
        deaths: Vec<scriptbots_core::DeathRecord>,
        replay_events: Vec<scriptbots_core::ReplayEvent>,
    }

    fn persistence_value(batch: &Arc<scriptbots_core::PersistenceBatch>) -> PersistenceTrace {
        PersistenceTrace {
            summary: batch.summary.clone(),
            epoch: batch.epoch,
            closed: batch.closed,
            metrics: batch.metrics.clone(),
            events: batch.events.clone(),
            agents: serde_json::to_value(&batch.agents).expect("agent persistence trace JSON"),
            births: batch.births.clone(),
            deaths: batch.deaths.clone(),
            replay_events: batch.replay_events.clone(),
        }
    }

    fn journal_trace(state: &Rc<RefCell<CaptureState>>) -> Vec<JournalTrace> {
        state
            .borrow()
            .batches
            .iter()
            .map(|batch| {
                (
                    batch.id(),
                    batch
                        .command()
                        .map(|command| serde_json::to_value(command).expect("command trace JSON")),
                    batch.applied(),
                    batch.scientific().map(std::convert::AsRef::as_ref).cloned(),
                    batch.persistence().map(persistence_value),
                )
            })
            .collect()
    }

    #[test]
    fn fixed_deadline_and_manual_drivers_produce_identical_host_traces() {
        let (mut manual, manual_journal) = captured_host(31, false, ReceiptMode::Immediate);
        let mut manual_port = manual.local_port();
        let (native_core, native_journal) = captured_host(31, false, ReceiptMode::Immediate);
        let mut native = FixedDeadlineHost::new(native_core);
        let mut native_port = native.local_port();

        let schedule = [
            (0, Vec::new()),
            (
                5,
                vec![
                    envelope(1, HostCommand::Pause),
                    envelope(2, HostCommand::Step),
                ],
            ),
            (5, Vec::new()),
            (10, vec![envelope(3, HostCommand::Resume)]),
            (20, Vec::new()),
            (20, Vec::new()),
        ];

        for (now, commands) in schedule {
            for command in commands {
                let manual_status = manual_port
                    .submit(command.clone())
                    .expect("manual command admission");
                let native_status = native.submit(command).expect("native command admission");
                assert_eq!(native_status, manual_status);
            }
            let instant = ManualInstant::from_nanos(now);
            let manual_receipt = manual.drive(instant).expect("manual drive");
            let native_receipt = native
                .drive_at(instant, NativeDriveTrigger::Command)
                .expect("native drive");
            assert_eq!(native_receipt.host, manual_receipt);
            assert_eq!(native.core().latest_snapshot(), manual.latest_snapshot());
            assert_eq!(
                native.core().scientific_digest_v1(),
                manual.scientific_digest_v1()
            );
            assert_eq!(events(&mut native_port), events(&mut manual_port));
        }

        assert_eq!(
            journal_trace(&native_journal),
            journal_trace(&manual_journal)
        );
        assert_eq!(native.core().world_tick(), Tick(2));
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one end-to-end trace keeps the manual and native schedules visibly adjacent"
    )]
    fn actual_native_runner_matches_manual_full_journal_trace() {
        let (mut manual, manual_journal) = captured_host_with_world(
            52,
            world_with_persistence_interval(1),
            options(false),
            ReceiptMode::Immediate,
        );
        let mut manual_port = manual.local_port();
        let (native_core, native_journal) = captured_host_with_world(
            52,
            world_with_persistence_interval(1),
            options(false),
            ReceiptMode::Immediate,
        );
        let (mut native, control) = NativeRunner::new(
            FixedDeadlineHost::new(native_core),
            NativeRunnerOptions::default(),
        )
        .expect("native parity runner");
        native_journal.borrow_mut().cancel_on_science = Some(control.clone());

        let clock = Arc::new(VirtualClock::starting_at(Time::ZERO));
        let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let runtime = RuntimeBuilder::current_thread()
            .with_timer_driver(timer.clone())
            .enable_platform_reactor(false)
            .build()
            .expect("native parity runtime");
        let failsafe = control;
        let advancer = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while timer.pending_count() == 0 {
                if std::time::Instant::now() >= deadline {
                    clock.advance_to(Time::from_nanos(10));
                    let _ = timer.process_timers();
                    let _ = failsafe.cancel();
                    return Err("native parity deadline was never registered");
                }
                std::thread::yield_now();
            }
            clock.advance_to(Time::from_nanos(10));
            Ok(timer.process_timers())
        });
        let native_run = native.run_on_runtime(&runtime);
        let timers_fired = advancer
            .join()
            .expect("native parity advancer")
            .expect("native parity deadline registration");
        let native_shutdown_id = match native_run.expect("native parity cancellation") {
            NativeRunOutcome::Cancelled {
                shutdown_command_id,
            } => shutdown_command_id,
            other => panic!("unexpected native parity outcome: {other:?}"),
        };
        assert_eq!(timers_fired, 1);

        manual
            .drive(ManualInstant::from_nanos(0))
            .expect("manual parity startup");
        manual
            .drive(ManualInstant::from_nanos(10))
            .expect("manual parity deadline");
        let manual_shutdown = manual.request_shutdown().expect("manual parity shutdown");
        for _ in 0..4 {
            manual
                .drive(ManualInstant::from_nanos(10))
                .expect("manual parity shutdown drain");
            if matches!(manual.drive_interest(), HostDriveInterest::Terminated) {
                break;
            }
        }
        assert!(matches!(
            manual.drive_interest(),
            HostDriveInterest::Terminated
        ));
        assert_eq!(manual_shutdown.command_id(), native_shutdown_id);

        let mut native_port = native.host().local_port();
        assert_eq!(
            native.host().core().latest_snapshot(),
            manual.latest_snapshot()
        );
        assert_eq!(
            native.host().core().scientific_digest_v1(),
            manual.scientific_digest_v1()
        );
        assert_eq!(events(&mut native_port), events(&mut manual_port));
        assert_eq!(
            journal_trace(&native_journal),
            journal_trace(&manual_journal)
        );
        assert!(native_journal.borrow().batches[0].persistence().is_some());
        assert_eq!(
            native_port
                .command_status(native_shutdown_id)
                .expect("native parity shutdown status query"),
            manual_port
                .command_status(manual_shutdown.command_id())
                .expect("manual parity shutdown status query")
        );
    }

    #[test]
    fn early_wakes_preserve_absolute_deadline_and_backward_time_is_atomic() {
        let (core, _) = captured_host(32, false, ReceiptMode::Immediate);
        let mut native = FixedDeadlineHost::new(core);
        let startup = native
            .drive_at(ManualInstant::from_nanos(100), NativeDriveTrigger::Startup)
            .expect("startup");
        assert_eq!(startup.next_deadline, ManualInstant::from_nanos(110));

        let early = native
            .drive_at(
                ManualInstant::from_nanos(105),
                NativeDriveTrigger::SyntheticWake,
            )
            .expect("early wake");
        assert_eq!(early.scheduled_deadlines_elapsed, 0);
        assert_eq!(early.host.scientific_steps, 0);
        assert_eq!(early.next_deadline, ManualInstant::from_nanos(110));
        let before = native.core().latest_snapshot();

        assert!(matches!(
            native.drive_at(
                ManualInstant::from_nanos(104),
                NativeDriveTrigger::SyntheticWake
            ),
            Err(NativeScheduleError::ClockMovedBackwards { .. })
        ));
        assert_eq!(native.last_observed(), Some(ManualInstant::from_nanos(105)));
        assert_eq!(native.next_deadline(), Some(ManualInstant::from_nanos(110)));
        assert_eq!(native.core().latest_snapshot(), before);
    }

    #[test]
    fn late_observation_bounds_catch_up_reports_skips_and_drops_no_fraction() {
        let (core, _) = captured_host(33, false, ReceiptMode::Immediate);
        let mut native = FixedDeadlineHost::new(core);
        native
            .drive_at(ManualInstant::from_nanos(0), NativeDriveTrigger::Startup)
            .expect("startup");
        let late = native
            .drive_at(ManualInstant::from_nanos(105), NativeDriveTrigger::Deadline)
            .expect("late deadline");
        assert_eq!(late.scheduled_deadlines_elapsed, 10);
        assert_eq!(late.scheduled_deadlines_skipped, 6);
        assert_eq!(late.host.automatic_steps_due, 10);
        assert_eq!(late.host.automatic_steps_skipped, 6);
        assert_eq!(late.host.scientific_steps, 4);
        assert_eq!(late.next_deadline, ManualInstant::from_nanos(110));

        let fractional = native
            .drive_at(ManualInstant::from_nanos(110), NativeDriveTrigger::Deadline)
            .expect("fractional deadline");
        assert_eq!(fractional.host.scientific_steps, 1);
        assert_eq!(native.core().world_tick(), Tick(5));
    }

    #[test]
    fn wake_storms_clients_and_repaints_cannot_change_science() {
        let (storm_core, _) = captured_host(34, true, ReceiptMode::Immediate);
        let mut storm = FixedDeadlineHost::new(storm_core);
        let (control_core, _) = captured_host(34, true, ReceiptMode::Immediate);
        let mut control = FixedDeadlineHost::new(control_core);
        storm
            .drive_at(ManualInstant::from_nanos(0), NativeDriveTrigger::Startup)
            .expect("storm startup");
        control
            .drive_at(ManualInstant::from_nanos(0), NativeDriveTrigger::Startup)
            .expect("control startup");

        let clients: [LocalHostPort; 128] = std::array::from_fn(|_| storm.local_port());
        assert_eq!(clients.len(), 128);
        for _ in 0..1_000 {
            let receipt = storm
                .drive_at(
                    ManualInstant::from_nanos(0),
                    NativeDriveTrigger::SyntheticWake,
                )
                .expect("coalesced-time wake");
            assert_eq!(receipt.host.scientific_steps, 0);
        }
        control
            .drive_at(
                ManualInstant::from_nanos(0),
                NativeDriveTrigger::SyntheticWake,
            )
            .expect("single control wake");

        assert_eq!(storm.core().world_tick(), Tick(0));
        assert_eq!(
            storm.core().scientific_digest_v1(),
            control.core().scientific_digest_v1()
        );
        assert_eq!(storm.next_deadline(), control.next_deadline());
        assert_eq!(storm.drive_interest(), HostDriveInterest::WakeOnly);
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn asupersync_cancel_before_start_drains_once_without_detached_tasks() {
        let (core, _) = captured_host(35, true, ReceiptMode::Immediate);
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("native runner");
        assert!(control.cancel());
        assert!(!control.cancel());

        let outcome = runner.run_until_terminal().expect("cancel-clean runner");
        assert!(matches!(outcome, NativeRunOutcome::Cancelled { .. }));
        assert_eq!(
            runner.host().core().latest_snapshot().lifecycle,
            HostLifecycle::Stopped
        );
        assert_eq!(runner.metrics().shutdown_requests, 1);
        assert_eq!(runner.metrics().owned_tasks_started, 0);
        assert_eq!(runner.metrics().owned_tasks_joined, 0);
        assert_eq!(
            runner.run_until_terminal().expect("idempotent join"),
            outcome
        );
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn cancellation_during_and_after_a_completed_step_preserves_exact_science() {
        let (during_core, during_journal) = captured_host(39, true, ReceiptMode::Immediate);
        let (mut during, during_control) = NativeRunner::new(
            FixedDeadlineHost::new(during_core),
            NativeRunnerOptions::default(),
        )
        .expect("during-step runner");
        during_journal.borrow_mut().cancel_on_science = Some(during_control.clone());
        let during_step = envelope(1, HostCommand::Step);
        during_control
            .try_submit(during_step.clone())
            .expect("during-step enqueue");

        assert!(matches!(
            during
                .run_until_terminal()
                .expect("during-step cancellation"),
            NativeRunOutcome::Cancelled { .. }
        ));
        assert_eq!(during.host().core().world_tick(), Tick(1));
        assert_eq!(
            during_journal
                .borrow()
                .batches
                .iter()
                .filter(|batch| batch.scientific().is_some())
                .count(),
            1
        );
        let mut during_port = during.host().local_port();
        let during_status = during_port
            .command_status(during_step.command_id)
            .expect("during-step status query")
            .expect("during-step status");
        assert!(matches!(
            during_status.application(),
            crate::ApplicationState::Applied(applied) if applied.tick == Tick(1)
        ));

        let (after_core, after_journal) = captured_host(40, true, ReceiptMode::Immediate);
        let mut after_host = FixedDeadlineHost::new(after_core);
        let after_step = envelope(1, HostCommand::Step);
        after_host
            .submit(after_step)
            .expect("after-step host admission");
        after_host
            .drive_at(ManualInstant::from_nanos(0), NativeDriveTrigger::Command)
            .expect("completed step boundary");
        after_host
            .drive_at(
                ManualInstant::from_nanos(0),
                NativeDriveTrigger::Maintenance,
            )
            .expect("completed step receipt");
        let digest_after_step = after_host
            .core()
            .scientific_digest_v1()
            .expect("post-step digest");
        let (mut after, after_control) =
            NativeRunner::new(after_host, NativeRunnerOptions::default())
                .expect("after-step runner");
        assert!(after_control.cancel());
        assert!(matches!(
            after.run_until_terminal().expect("after-step cancellation"),
            NativeRunOutcome::Cancelled { .. }
        ));
        assert_eq!(after.host().core().world_tick(), Tick(1));
        assert_eq!(
            after
                .host()
                .core()
                .scientific_digest_v1()
                .expect("post-cancellation digest"),
            digest_after_step
        );
        assert_eq!(
            after_journal
                .borrow()
                .batches
                .iter()
                .filter(|batch| batch.scientific().is_some())
                .count(),
            1
        );
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn full_host_queue_stops_after_its_ordered_scientific_boundary() {
        let mut constrained = options(true);
        constrained.command_capacity = 1;
        let (core, journal) = captured_host_with_options(41, constrained, ReceiptMode::Immediate);
        let mut host = FixedDeadlineHost::new(core);
        let step = envelope(1, HostCommand::Step);
        assert!(matches!(
            host.submit(step.clone())
                .expect("fill exact host queue")
                .application(),
            crate::ApplicationState::Admitted
        ));
        let (mut runner, control) =
            NativeRunner::new(host, NativeRunnerOptions::default()).expect("queue-full runner");
        assert!(control.cancel());

        assert!(matches!(
            runner.run_until_terminal().expect("queue-full shutdown"),
            NativeRunOutcome::Cancelled { .. }
        ));
        assert_eq!(runner.host().core().world_tick(), Tick(1));
        assert_eq!(
            journal
                .borrow()
                .batches
                .iter()
                .filter(|batch| batch.scientific().is_some())
                .count(),
            1
        );
        let mut port = runner.host().local_port();
        assert!(matches!(
            port.command_status(step.command_id)
                .expect("queue-full status query")
                .expect("queue-full step status")
                .application(),
            crate::ApplicationState::Applied(applied) if applied.tick == Tick(1)
        ));
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn explicit_shutdown_is_idempotent_for_empty_and_nonempty_queues() {
        let (empty_core, _) = captured_host(42, true, ReceiptMode::Immediate);
        let (mut empty, empty_control) = NativeRunner::new(
            FixedDeadlineHost::new(empty_core),
            NativeRunnerOptions::default(),
        )
        .expect("empty shutdown runner");
        let empty_shutdown = envelope(1, HostCommand::Shutdown);
        empty_control
            .try_submit(empty_shutdown.clone())
            .expect("empty shutdown enqueue");
        let empty_outcome = empty.run_until_terminal().expect("empty shutdown");
        assert_eq!(
            empty_outcome,
            NativeRunOutcome::Stopped {
                shutdown_command_id: empty_shutdown.command_id,
            }
        );
        assert_eq!(
            empty.run_until_terminal().expect("repeat empty shutdown"),
            empty_outcome
        );

        let (ordered_core, _) = captured_host(43, true, ReceiptMode::Immediate);
        let (mut ordered, ordered_control) = NativeRunner::new(
            FixedDeadlineHost::new(ordered_core),
            NativeRunnerOptions::default(),
        )
        .expect("ordered shutdown runner");
        let step = envelope(1, HostCommand::Step);
        let shutdown = envelope(2, HostCommand::Shutdown);
        let after_shutdown = envelope(3, HostCommand::Resume);
        ordered_control
            .try_submit(step.clone())
            .expect("ordered step enqueue");
        ordered_control
            .try_submit(shutdown.clone())
            .expect("ordered shutdown enqueue");
        ordered_control
            .try_submit(after_shutdown.clone())
            .expect("post-shutdown enqueue remains pre-admission only");

        assert_eq!(
            ordered.run_until_terminal().expect("ordered shutdown"),
            NativeRunOutcome::Stopped {
                shutdown_command_id: shutdown.command_id,
            }
        );
        assert_eq!(ordered.host().core().world_tick(), Tick(1));
        let mut port = ordered.host().local_port();
        assert!(matches!(
            port.command_status(step.command_id)
                .expect("step status query")
                .expect("step status")
                .application(),
            crate::ApplicationState::Applied(applied) if applied.tick == Tick(1)
        ));
        assert!(matches!(
            port.command_status(after_shutdown.command_id)
                .expect("post-shutdown status query")
                .expect("post-shutdown status")
                .application(),
            crate::ApplicationState::Rejected(crate::RejectionReason::HostStopping)
        ));
        let ordered_events = events(&mut port);
        assert!(
            ordered_events
                .windows(2)
                .all(|pair| pair[0].sequence < pair[1].sequence)
        );
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn stale_shutdown_cas_reopens_native_command_ingress() {
        let (core, _) = captured_host(44, true, ReceiptMode::Immediate);
        let runner_options = NativeRunnerOptions {
            ingress_capacity: 1,
            ..NativeRunnerOptions::default()
        };
        let (mut runner, control) = NativeRunner::new(FixedDeadlineHost::new(core), runner_options)
            .expect("stale shutdown runner");
        let stale = envelope(1, HostCommand::Shutdown)
            .expecting_control_revision(crate::ControlRevision::new(99));
        control
            .try_submit(stale.clone())
            .expect("stale shutdown enqueue");
        let producer = control;
        let followup = std::thread::spawn(move || {
            let mut candidate = envelope(2, HostCommand::Shutdown);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            loop {
                let command_id = candidate.command_id;
                match producer.try_submit(candidate) {
                    Ok(()) => return Ok(command_id),
                    Err(NativeIngressError::Full(returned)) => candidate = returned,
                    Err(NativeIngressError::Closed(_)) => {
                        let _ = producer.cancel();
                        return Err("native command ingress closed before post-CAS shutdown");
                    }
                }
                if std::time::Instant::now() >= deadline {
                    let _ = producer.cancel();
                    return Err("post-CAS shutdown never entered native ingress");
                }
                std::thread::yield_now();
            }
        });

        let run = runner.run_until_terminal();
        let followup_id = followup
            .join()
            .expect("follow-up shutdown producer")
            .expect("follow-up shutdown admission");
        let outcome = run.expect("post-CAS shutdown must stop");
        let shutdown_command_id = match outcome {
            NativeRunOutcome::Stopped {
                shutdown_command_id,
            } => shutdown_command_id,
            other => panic!("unexpected stale-CAS outcome: {other:?}"),
        };
        assert_eq!(shutdown_command_id, followup_id);
        assert_ne!(shutdown_command_id, stale.command_id);
        let mut port = runner.host().local_port();
        assert!(matches!(
            port.command_status(stale.command_id)
                .expect("stale status query")
                .expect("stale status")
                .application(),
            crate::ApplicationState::Rejected(
                crate::RejectionReason::ControlRevisionConflict { .. }
            )
        ));
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn cancellation_replaces_a_pre_admitted_stale_shutdown_identity() {
        let (core, _) = captured_host(50, true, ReceiptMode::Immediate);
        let stale = envelope(1, HostCommand::Shutdown)
            .expecting_control_revision(crate::ControlRevision::new(99));
        let mut port = core.local_port();
        port.submit(stale.clone())
            .expect("pre-admit stale shutdown");
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("stale cancellation runner");
        assert!(control.cancel());

        let outcome = runner
            .run_until_terminal()
            .expect("cancellation must replace stale shutdown");
        let shutdown_command_id = match outcome {
            NativeRunOutcome::Cancelled {
                shutdown_command_id,
            } => shutdown_command_id,
            other => panic!("unexpected stale-cancellation outcome: {other:?}"),
        };
        assert_ne!(shutdown_command_id, stale.command_id);
        assert_eq!(
            runner.host().core().shutdown_command_id(),
            Some(shutdown_command_id)
        );
        assert_eq!(
            runner.host().core().latest_snapshot().lifecycle,
            HostLifecycle::Stopped
        );
        let mut port = runner.host().local_port();
        assert!(matches!(
            port.command_status(stale.command_id)
                .expect("stale cancellation status query")
                .expect("stale cancellation status")
                .application(),
            crate::ApplicationState::Rejected(
                crate::RejectionReason::ControlRevisionConflict { .. }
            )
        ));
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn retained_shutdown_batch_recovers_from_full_on_journal_ready() {
        let ready = Arc::new(AtomicBool::new(false));
        let blocked = Arc::new(AtomicBool::new(false));
        let on_full = Rc::new(RefCell::new(None));
        let wake_results = Rc::new(RefCell::new(None));
        let core = HostCore::with_journal(
            HostSessionId::new(45),
            world(),
            options(true),
            Box::new(ReadinessJournal {
                ready: Arc::clone(&ready),
                blocked: Arc::clone(&blocked),
                on_full: Rc::clone(&on_full),
                wake_results: Rc::clone(&wake_results),
                receipts: VecDeque::new(),
            }),
        )
        .expect("readiness-gated host");
        let (mut runner, control) = NativeRunner::new(
            FixedDeadlineHost::new(core),
            NativeRunnerOptions {
                ingress_capacity: 1,
                ..NativeRunnerOptions::default()
            },
        )
        .expect("readiness-gated runner");
        *on_full.borrow_mut() = Some(control.clone());
        assert!(control.cancel());

        assert!(matches!(
            runner
                .run_until_terminal()
                .expect("journal-ready recovery must stop"),
            NativeRunOutcome::Cancelled { .. }
        ));
        assert!(blocked.load(Ordering::Acquire));
        assert!(ready.load(Ordering::Acquire));
        assert_eq!(
            *wake_results.borrow(),
            Some((NativeWakeResult::Enqueued, NativeWakeResult::Coalesced))
        );
        assert!(runner.host().core().pending_journal_batch().is_none());
        assert_eq!(
            runner.host().core().latest_snapshot().lifecycle,
            HostLifecycle::Stopped
        );
        assert!(runner.metrics().journal_wakes >= 1);
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one virtual-time scenario proves the complete JournalFull wake transition"
    )]
    fn journal_full_disarms_native_deadlines_until_explicit_ready() {
        let ready = Arc::new(AtomicBool::new(false));
        let blocked = Arc::new(AtomicBool::new(false));
        let core = HostCore::with_journal(
            HostSessionId::new(53),
            world(),
            options(false),
            Box::new(ReadinessJournal {
                ready: Arc::clone(&ready),
                blocked: Arc::clone(&blocked),
                on_full: Rc::new(RefCell::new(None)),
                wake_results: Rc::new(RefCell::new(None)),
                receipts: VecDeque::new(),
            }),
        )
        .expect("automatic journal-full host");
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("automatic journal-full runner");
        let pause = envelope(20, HostCommand::Pause);
        let shutdown = envelope(21, HostCommand::Shutdown);

        let clock = Arc::new(VirtualClock::starting_at(Time::ZERO));
        let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let runtime = RuntimeBuilder::current_thread()
            .with_timer_driver(timer.clone())
            .enable_platform_reactor(false)
            .build()
            .expect("automatic journal-full runtime");
        let lifecycle = control;
        let expected_pause = pause.clone();
        let expected_shutdown = shutdown.clone();
        let coordinator = std::thread::spawn(move || {
            let first_deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while timer.pending_count() == 0 {
                if std::time::Instant::now() >= first_deadline {
                    clock.advance_to(Time::from_nanos(10));
                    let _ = timer.process_timers();
                    ready.store(true, Ordering::Release);
                    let _ = lifecycle.journal_ready();
                    let _ = lifecycle.cancel();
                    return Err("initial native deadline was never registered");
                }
                std::thread::yield_now();
            }
            clock.advance_to(Time::from_nanos(10));
            let first_fired = timer.process_timers();

            let blocked_deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while !(blocked.load(Ordering::Acquire) && lifecycle.is_owner_waiting()) {
                if std::time::Instant::now() >= blocked_deadline {
                    ready.store(true, Ordering::Release);
                    let _ = lifecycle.journal_ready();
                    let _ = lifecycle.cancel();
                    return Err("journal-full host never reached its wake-only wait");
                }
                std::thread::yield_now();
            }
            if timer.pending_count() != 0 {
                ready.store(true, Ordering::Release);
                let _ = lifecycle.journal_ready();
                let _ = lifecycle.cancel();
                return Err("journal-full wake-only wait retained a cadence timer");
            }
            clock.advance_to(Time::from_nanos(1_000));
            let paused_fired = timer.process_timers();
            if lifecycle.try_submit(pause).is_err() || lifecycle.try_submit(shutdown).is_err() {
                ready.store(true, Ordering::Release);
                let _ = lifecycle.journal_ready();
                let _ = lifecycle.cancel();
                return Err("journal-full recovery commands did not enter native ingress");
            }
            ready.store(true, Ordering::Release);
            let ready_wake = lifecycle.journal_ready();
            Ok((first_fired, paused_fired, ready_wake))
        });

        let run = runner.run_on_runtime(&runtime);
        let (first_fired, paused_fired, ready_wake) = coordinator
            .join()
            .expect("journal-full coordinator")
            .expect("journal-full coordination");
        assert_eq!(first_fired, 1);
        assert_eq!(paused_fired, 0);
        assert!(matches!(
            ready_wake,
            NativeWakeResult::Enqueued | NativeWakeResult::Coalesced
        ));
        assert_eq!(
            run.expect("journal-full recovery must stop"),
            NativeRunOutcome::Stopped {
                shutdown_command_id: expected_shutdown.command_id,
            }
        );
        assert_eq!(runner.host().core().world_tick(), Tick(1));
        assert_eq!(runner.metrics().deadline_wakes, 1);
        assert_eq!(runner.metrics().automatic_steps_skipped, 0);
        assert!(runner.host().core().pending_journal_batch().is_none());
        let mut port = runner.host().local_port();
        for command in [expected_pause, expected_shutdown] {
            assert!(matches!(
                port.command_status(command.command_id)
                    .expect("journal-full command status query")
                    .expect("journal-full command status")
                    .application(),
                crate::ApplicationState::Applied(applied) if applied.tick == Tick(1)
            ));
        }
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn controller_disconnect_uses_the_same_ordered_shutdown_barrier() {
        let (core, _) = captured_host(46, true, ReceiptMode::Immediate);
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("disconnect runner");
        drop(control);

        assert!(matches!(
            runner.run_until_terminal().expect("disconnect shutdown"),
            NativeRunOutcome::ControllerDisconnected { .. }
        ));
        assert_eq!(
            runner.host().core().latest_snapshot().lifecycle,
            HostLifecycle::Stopped
        );
        assert_eq!(runner.metrics().shutdown_requests, 1);
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn disconnect_wins_a_ready_deadline_tie_without_extra_science() {
        let (core, _) = captured_host(57, false, ReceiptMode::Immediate);
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("disconnect deadline-tie runner");
        let clock = Arc::new(VirtualClock::starting_at(Time::ZERO));
        let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let runtime = RuntimeBuilder::current_thread()
            .with_timer_driver(timer.clone())
            .enable_platform_reactor(false)
            .build()
            .expect("disconnect deadline-tie runtime");
        let coordinator = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while timer.pending_count() == 0 {
                if std::time::Instant::now() >= deadline {
                    drop(control);
                    clock.advance_to(Time::from_nanos(10));
                    let _ = timer.process_timers();
                    return Err("disconnect deadline was never registered");
                }
                std::thread::yield_now();
            }
            drop(control);
            clock.advance_to(Time::from_nanos(10));
            let _ = timer.process_timers();
            Ok(())
        });

        let run = runner.run_on_runtime(&runtime);
        coordinator
            .join()
            .expect("disconnect deadline-tie coordinator")
            .expect("disconnect deadline registration");
        assert!(matches!(
            run,
            Ok(NativeRunOutcome::ControllerDisconnected { .. })
        ));
        assert_eq!(runner.host().core().world_tick(), Tick(0));
        assert_eq!(runner.metrics().automatic_steps_skipped, 0);
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn disconnect_after_valid_explicit_shutdown_preserves_stopped_outcome() {
        let (core, _) = captured_host(54, true, ReceiptMode::Immediate);
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("explicit shutdown disconnect runner");
        let shutdown = envelope(1, HostCommand::Shutdown);
        control
            .try_submit(shutdown.clone())
            .expect("explicit shutdown enqueue");
        drop(control);

        assert_eq!(
            runner
                .run_until_terminal()
                .expect("explicit shutdown before disconnect"),
            NativeRunOutcome::Stopped {
                shutdown_command_id: shutdown.command_id,
            }
        );
        assert_eq!(runner.host().core().world_tick(), Tick(0));
        assert_eq!(runner.metrics().shutdown_requests, 1);
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn disconnect_replaces_stale_queued_shutdown_with_fail_safe_identity() {
        let (core, _) = captured_host(55, true, ReceiptMode::Immediate);
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("stale shutdown disconnect runner");
        let stale = envelope(1, HostCommand::Shutdown)
            .expecting_control_revision(crate::ControlRevision::new(99));
        control
            .try_submit(stale.clone())
            .expect("stale shutdown enqueue");
        drop(control);

        let fail_safe_id = match runner
            .run_until_terminal()
            .expect("disconnect must replace stale shutdown")
        {
            NativeRunOutcome::ControllerDisconnected {
                shutdown_command_id,
            } => shutdown_command_id,
            other => panic!("unexpected stale-disconnect outcome: {other:?}"),
        };
        assert_ne!(fail_safe_id, stale.command_id);
        assert_eq!(
            runner.host().core().shutdown_command_id(),
            Some(fail_safe_id)
        );
        let mut port = runner.host().local_port();
        assert!(matches!(
            port.command_status(stale.command_id)
                .expect("stale disconnect status query")
                .expect("stale disconnect status")
                .application(),
            crate::ApplicationState::Rejected(
                crate::RejectionReason::ControlRevisionConflict { .. }
            )
        ));
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn disconnect_provenance_survives_journal_full_before_stale_shutdown() {
        let ready = Arc::new(AtomicBool::new(false));
        let blocked = Arc::new(AtomicBool::new(false));
        let on_full = Rc::new(RefCell::new(None));
        let wake_results = Rc::new(RefCell::new(None));
        let core = HostCore::with_journal(
            HostSessionId::new(56),
            world(),
            options(true),
            Box::new(ReadinessJournal {
                ready: Arc::clone(&ready),
                blocked: Arc::clone(&blocked),
                on_full: Rc::clone(&on_full),
                wake_results,
                receipts: VecDeque::new(),
            }),
        )
        .expect("journal-full stale-shutdown host");
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("journal-full stale-shutdown runner");
        *on_full.borrow_mut() = Some(control.clone());
        let step = envelope(1, HostCommand::Step);
        let stale = envelope(2, HostCommand::Shutdown)
            .expecting_control_revision(crate::ControlRevision::new(99));
        control
            .try_submit(step.clone())
            .expect("journal-full step enqueue");
        control
            .try_submit(stale.clone())
            .expect("journal-full stale shutdown enqueue");
        drop(control);

        let fail_safe_id = match runner
            .run_until_terminal()
            .expect("journal-full disconnect must install fail-safe shutdown")
        {
            NativeRunOutcome::ControllerDisconnected {
                shutdown_command_id,
            } => shutdown_command_id,
            other => panic!("unexpected delayed stale-disconnect outcome: {other:?}"),
        };
        assert!(blocked.load(Ordering::Acquire));
        assert!(ready.load(Ordering::Acquire));
        assert_ne!(fail_safe_id, stale.command_id);
        assert_eq!(runner.host().core().world_tick(), Tick(1));
        let mut port = runner.host().local_port();
        assert!(matches!(
            port.command_status(step.command_id)
                .expect("delayed step status query")
                .expect("delayed step status")
                .application(),
            crate::ApplicationState::Applied(applied) if applied.tick == Tick(1)
        ));
        assert!(matches!(
            port.command_status(stale.command_id)
                .expect("delayed stale status query")
                .expect("delayed stale status")
                .application(),
            crate::ApplicationState::Rejected(
                crate::RejectionReason::ControlRevisionConflict { .. }
            )
        ));
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one coordinated trace proves delayed stale-CAS recovery before valid shutdown"
    )]
    fn connected_stale_shutdown_reopens_for_later_valid_shutdown() {
        let ready = Arc::new(AtomicBool::new(false));
        let blocked = Arc::new(AtomicBool::new(false));
        let core = HostCore::with_journal(
            HostSessionId::new(58),
            world(),
            options(true),
            Box::new(ReadinessJournal {
                ready: Arc::clone(&ready),
                blocked: Arc::clone(&blocked),
                on_full: Rc::new(RefCell::new(None)),
                wake_results: Rc::new(RefCell::new(None)),
                receipts: VecDeque::new(),
            }),
        )
        .expect("connected stale-shutdown host");
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("connected stale-shutdown runner");
        let step = envelope(1, HostCommand::Step);
        let stale = envelope(2, HostCommand::Shutdown)
            .expecting_control_revision(crate::ControlRevision::new(99));
        let valid = envelope(3, HostCommand::Shutdown);
        control
            .try_submit(step.clone())
            .expect("connected delayed step enqueue");
        control
            .try_submit(stale.clone())
            .expect("connected delayed stale shutdown enqueue");
        let coordinator = std::thread::spawn(move || {
            let first_wait_deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while !(blocked.load(Ordering::Acquire) && control.is_owner_waiting()) {
                if std::time::Instant::now() >= first_wait_deadline {
                    ready.store(true, Ordering::Release);
                    let _ = control.journal_ready();
                    let _ = control.cancel();
                    return Err("connected stale shutdown never reached JournalFull wait");
                }
                std::thread::yield_now();
            }
            let first_wait_generation = control.owner_wait_generation();
            ready.store(true, Ordering::Release);
            let _ = control.journal_ready();

            let reopened_deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while control.owner_wait_generation() <= first_wait_generation
                || !control.is_owner_waiting()
            {
                if std::time::Instant::now() >= reopened_deadline {
                    let _ = control.cancel();
                    return Err("stale shutdown never returned to a connected wake-only wait");
                }
                std::thread::yield_now();
            }
            let valid_id = valid.command_id;
            if control.try_submit(valid).is_err() {
                let _ = control.cancel();
                return Err("valid shutdown could not enter reopened native ingress");
            }
            Ok(valid_id)
        });

        let run = runner.run_until_terminal();
        let valid_id = coordinator
            .join()
            .expect("connected stale-shutdown coordinator")
            .expect("connected stale-shutdown recovery");
        assert_eq!(
            run.expect("later valid shutdown must stop"),
            NativeRunOutcome::Stopped {
                shutdown_command_id: valid_id,
            }
        );
        assert_eq!(runner.host().core().world_tick(), Tick(1));
        let mut port = runner.host().local_port();
        assert!(matches!(
            port.command_status(step.command_id)
                .expect("connected delayed step status query")
                .expect("connected delayed step status")
                .application(),
            crate::ApplicationState::Applied(applied) if applied.tick == Tick(1)
        ));
        assert!(matches!(
            port.command_status(stale.command_id)
                .expect("connected delayed stale status query")
                .expect("connected delayed stale status")
                .application(),
            crate::ApplicationState::Rejected(
                crate::RejectionReason::ControlRevisionConflict { .. }
            )
        ));
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one virtual-time trace proves provisional shutdown remains timeout-bounded"
    )]
    fn disconnected_provisional_shutdown_times_out_without_journal_ready() {
        let ready = Arc::new(AtomicBool::new(false));
        let blocked = Arc::new(AtomicBool::new(false));
        let core = HostCore::with_journal(
            HostSessionId::new(59),
            world(),
            options(true),
            Box::new(ReadinessJournal {
                ready: Arc::clone(&ready),
                blocked: Arc::clone(&blocked),
                on_full: Rc::new(RefCell::new(None)),
                wake_results: Rc::new(RefCell::new(None)),
                receipts: VecDeque::new(),
            }),
        )
        .expect("provisional-timeout host");
        let (mut runner, control) = NativeRunner::new(
            FixedDeadlineHost::new(core),
            NativeRunnerOptions {
                maintenance_period_nanos: 5,
                shutdown_timeout_nanos: 20,
                ..NativeRunnerOptions::default()
            },
        )
        .expect("provisional-timeout runner");
        let step = envelope(1, HostCommand::Step);
        let shutdown = envelope(2, HostCommand::Shutdown);
        control
            .try_submit(step)
            .expect("provisional-timeout step enqueue");
        control
            .try_submit(shutdown.clone())
            .expect("provisional-timeout shutdown enqueue");

        let clock = Arc::new(VirtualClock::starting_at(Time::ZERO));
        let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let runtime = RuntimeBuilder::current_thread()
            .with_timer_driver(timer.clone())
            .enable_platform_reactor(false)
            .build()
            .expect("provisional-timeout runtime");
        let blocked_for_coordinator = Arc::clone(&blocked);
        let ready_for_coordinator = Arc::clone(&ready);
        let coordinator = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while !(blocked_for_coordinator.load(Ordering::Acquire) && timer.pending_count() > 0) {
                if std::time::Instant::now() >= deadline {
                    ready_for_coordinator.store(true, Ordering::Release);
                    let _ = control.journal_ready();
                    let _ = control.cancel();
                    drop(control);
                    clock.advance_to(Time::from_nanos(20));
                    let _ = timer.process_timers();
                    return Err("provisional shutdown timeout was never registered");
                }
                std::thread::yield_now();
            }
            drop(control);
            clock.advance_to(Time::from_nanos(20));
            let _ = timer.process_timers();
            Ok(())
        });

        let run = runner.run_on_runtime(&runtime);
        coordinator
            .join()
            .expect("provisional-timeout coordinator")
            .expect("provisional timeout registration");
        let pending_batch = match run {
            Err(NativeRunError::ShutdownTimedOut {
                waited_nanos,
                pending_batch,
                ..
            }) => {
                assert_eq!(waited_nanos, 20);
                pending_batch
            }
            other => panic!("unexpected provisional-timeout result: {other:?}"),
        };
        assert!(pending_batch.is_some());
        assert!(blocked.load(Ordering::Acquire));
        assert_eq!(runner.host().core().world_tick(), Tick(1));
        assert!(runner.host().core().pending_journal_batch().is_some());
        let mut port = runner.host().local_port();
        let shutdown_status = port
            .command_status(shutdown.command_id)
            .expect("provisional shutdown status query")
            .expect("provisional shutdown status");
        assert!(matches!(
            shutdown_status.application(),
            crate::ApplicationState::Admitted
        ));
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn root_panic_retains_racing_native_envelopes_without_detaching() {
        let (core, journal) = captured_host(47, true, ReceiptMode::Panic);
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("panic runner");
        let step = envelope(1, HostCommand::Step);
        let racing = envelope(2, HostCommand::Pause);
        journal.borrow_mut().enqueue_before_panic = Some((control.clone(), racing.clone()));
        control
            .try_submit(step.clone())
            .expect("panic step enqueue");

        assert!(matches!(
            runner.run_until_terminal(),
            Err(NativeRunError::Panicked { .. })
        ));
        assert_eq!(runner.unresolved_envelopes().len(), 1);
        assert_eq!(
            runner.unresolved_envelopes()[0].command_id,
            racing.command_id
        );
        let indeterminate = runner
            .host()
            .core()
            .indeterminate_journal_batch()
            .expect("panicked active batch must remain exact evidence");
        assert_eq!(indeterminate.command_id(), Some(step.command_id));
        assert_eq!(
            indeterminate
                .scientific()
                .expect("step boundary evidence")
                .summary()
                .tick,
            Tick(1)
        );
        assert!(indeterminate.persistence().is_some());
        assert!(runner.host().core().pending_journal_batch().is_none());
        assert_eq!(
            runner
                .host()
                .core()
                .panicked_command()
                .map(|command| command.command_id),
            Some(step.command_id)
        );
        assert!(matches!(
            runner.host().core().health(),
            crate::HostHealth::Faulted(crate::HostFault::Protocol { code, .. })
                if code == "native_lifecycle_panic"
        ));
        let captured = journal.borrow();
        assert!(Arc::ptr_eq(&indeterminate, &captured.batches[0]));
        drop(captured);
        let mut port = runner.host().local_port();
        let step_status = port
            .command_status(step.command_id)
            .expect("panicked step status query")
            .expect("panicked step status");
        assert!(matches!(
            step_status.application(),
            crate::ApplicationState::Applied(applied) if applied.tick == Tick(1)
        ));
        assert_eq!(step_status.journal(), &crate::JournalState::Pending);
        assert_eq!(runner.metrics().owned_tasks_started, 0);
        assert_eq!(runner.metrics().owned_tasks_joined, 0);
        assert!(matches!(
            control.try_submit(envelope(3, HostCommand::Resume)),
            Err(NativeIngressError::Closed(_))
        ));
        assert!(matches!(
            runner.run_until_terminal(),
            Err(NativeRunError::Panicked { .. })
        ));
        let repeated = runner
            .host()
            .core()
            .indeterminate_journal_batch()
            .expect("panic evidence remains sticky");
        assert!(Arc::ptr_eq(&indeterminate, &repeated));
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn terminal_drain_failure_is_sticky_before_any_runtime_retry() {
        let (core, _) = captured_host(51, true, ReceiptMode::Immediate);
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("terminal-drain sticky runner");
        let unresolved = envelope(1, HostCommand::Pause);
        runner.inject_terminal_drain_failure_for_test(
            unresolved.clone(),
            "injected terminal reconciliation failure",
        );
        let metrics_before = runner.metrics();

        for _ in 0..2 {
            assert!(matches!(
                runner.run_until_terminal(),
                Err(NativeRunError::TerminalDrainFailed {
                    unresolved_envelopes: 1,
                    ref message,
                }) if message == "injected terminal reconciliation failure"
            ));
            assert_eq!(runner.metrics(), metrics_before);
        }
        assert_eq!(runner.unresolved_envelopes().len(), 1);
        assert_eq!(
            runner.unresolved_envelopes()[0].command_id,
            unresolved.command_id
        );
        assert!(matches!(
            control.try_submit(envelope(2, HostCommand::Resume)),
            Err(NativeIngressError::Closed(_))
        ));
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn virtual_deadline_drives_one_exact_step_then_cancel_cleanly() {
        let (core, journal) = captured_host(48, false, ReceiptMode::Immediate);
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("virtual deadline runner");
        journal.borrow_mut().cancel_on_science = Some(control.clone());
        let clock = Arc::new(VirtualClock::starting_at(Time::ZERO));
        let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let runtime = RuntimeBuilder::current_thread()
            .with_timer_driver(timer.clone())
            .enable_platform_reactor(false)
            .build()
            .expect("virtual deadline runtime");
        let failsafe = control;
        let advancer = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while timer.pending_count() == 0 {
                if std::time::Instant::now() >= deadline {
                    clock.advance_to(Time::from_nanos(10));
                    let _ = timer.process_timers();
                    let _ = failsafe.cancel();
                    return Err("native deadline sleep was never registered");
                }
                std::thread::yield_now();
            }
            clock.advance_to(Time::from_nanos(10));
            Ok(timer.process_timers())
        });

        let run = runner.run_on_runtime(&runtime);
        let timers_fired = advancer
            .join()
            .expect("virtual deadline advancer")
            .expect("virtual deadline registration");
        assert!(matches!(run, Ok(NativeRunOutcome::Cancelled { .. })));
        assert!(timers_fired >= 1);
        assert_eq!(runner.host().core().world_tick(), Tick(1));
        assert_eq!(runner.metrics().deadline_wakes, 1);
        assert_eq!(runner.metrics().automatic_steps_skipped, 0);
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn long_virtual_pause_has_no_periodic_drive_or_science() {
        let (core, _) = captured_host(49, true, ReceiptMode::Immediate);
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), NativeRunnerOptions::default())
                .expect("paused virtual runner");
        assert_eq!(control.wake(), NativeWakeResult::Enqueued);
        for _ in 0..999 {
            assert_eq!(control.wake(), NativeWakeResult::Coalesced);
        }
        let clock = Arc::new(VirtualClock::starting_at(Time::ZERO));
        let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let runtime = RuntimeBuilder::current_thread()
            .with_timer_driver(timer.clone())
            .enable_platform_reactor(false)
            .build()
            .expect("paused virtual runtime");
        let lifecycle = control;
        let advancer = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while !lifecycle.is_owner_waiting() {
                if std::time::Instant::now() >= deadline {
                    clock.advance_to(Time::from_nanos(1_000_000_000_000));
                    let _ = timer.process_timers();
                    let _ = lifecycle.cancel();
                    return Err("paused native owner never reached its event wait");
                }
                std::thread::yield_now();
            }
            clock.advance_to(Time::from_nanos(1_000_000_000_000));
            let fired = timer.process_timers();
            let first_cancel = lifecycle.cancel();
            Ok((fired, first_cancel))
        });

        let run = runner.run_on_runtime(&runtime);
        let (timers_fired, first_cancel) = advancer
            .join()
            .expect("paused virtual advancer")
            .expect("paused owner wait registration");
        assert!(matches!(run, Ok(NativeRunOutcome::Cancelled { .. })));
        assert_eq!(timers_fired, 0);
        assert!(first_cancel);
        assert_eq!(runner.host().core().world_tick(), Tick(0));
        assert_eq!(runner.metrics().deadline_wakes, 0);
        assert_eq!(runner.metrics().synthetic_wakes, 1);
        assert!(runner.metrics().drive_calls <= 4);
        assert_eq!(runner.metrics().owned_tasks_started, 0);
        assert_eq!(runner.metrics().owned_tasks_joined, 0);
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn native_ingress_retains_full_envelope_and_shutdown_rejects_queued_work() {
        let (core, _) = captured_host(36, true, ReceiptMode::Immediate);
        let options = NativeRunnerOptions {
            ingress_capacity: 1,
            ..NativeRunnerOptions::default()
        };
        let (mut runner, control) =
            NativeRunner::new(FixedDeadlineHost::new(core), options).expect("bounded runner");
        let first = envelope(1, HostCommand::Step);
        let overflow = envelope(2, HostCommand::Step);
        control
            .try_submit(first.clone())
            .expect("first native enqueue");
        let returned = control
            .try_submit(overflow)
            .expect_err("second envelope must be retained")
            .into_envelope();
        assert_eq!(returned.command_id, CommandId::from_client_sequence(7, 2));
        assert!(control.cancel());

        assert!(matches!(
            runner.run_until_terminal().expect("ordered cancellation"),
            NativeRunOutcome::Cancelled { .. }
        ));
        let mut port = runner.host().local_port();
        let status = port
            .command_status(first.command_id)
            .expect("first status query")
            .expect("first status retained");
        assert!(matches!(
            status.application(),
            crate::ApplicationState::Rejected(crate::RejectionReason::HostStopping)
        ));
    }

    #[cfg(all(feature = "native-asupersync", not(target_arch = "wasm32")))]
    #[test]
    fn journal_failure_and_timeout_return_with_exact_host_retained() {
        let (failed_core, _) = captured_host(37, true, ReceiptMode::Failed);
        let (mut failed, failed_control) = NativeRunner::new(
            FixedDeadlineHost::new(failed_core),
            NativeRunnerOptions::default(),
        )
        .expect("failed journal runner");
        assert!(failed_control.cancel());
        assert!(matches!(
            failed.run_until_terminal(),
            Err(NativeRunError::HostFault { .. })
        ));
        assert!(failed.host().core().health().fault().is_some());

        let (timeout_core, _) = captured_host(38, true, ReceiptMode::Never);
        let timeout_options = NativeRunnerOptions {
            maintenance_period_nanos: 1_000_000,
            shutdown_timeout_nanos: 2_000_000,
            ..NativeRunnerOptions::default()
        };
        let (mut timed_out, timeout_control) =
            NativeRunner::new(FixedDeadlineHost::new(timeout_core), timeout_options)
                .expect("timeout runner");
        assert!(timeout_control.cancel());
        let clock = Arc::new(VirtualClock::starting_at(Time::ZERO));
        let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let runtime = RuntimeBuilder::current_thread()
            .with_timer_driver(timer.clone())
            .enable_platform_reactor(false)
            .build()
            .expect("virtual timeout runtime");
        let failsafe = timeout_control;
        let advancer = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while timer.pending_count() == 0 {
                if std::time::Instant::now() >= deadline {
                    clock.advance_to(Time::from_nanos(2_000_000));
                    let _ = timer.process_timers();
                    let _ = failsafe.cancel();
                    return Err("shutdown timeout sleep was never registered");
                }
                std::thread::yield_now();
            }
            clock.advance_to(Time::from_nanos(2_000_000));
            Ok(timer.process_timers())
        });
        let first_run = timed_out.run_on_runtime(&runtime);
        let timers_fired = advancer
            .join()
            .expect("virtual timeout advancer")
            .expect("virtual timeout registration");
        assert!(matches!(
            first_run,
            Err(NativeRunError::ShutdownTimedOut { .. })
        ));
        assert!(timers_fired >= 1);
        assert_eq!(
            timed_out.host().core().latest_snapshot().lifecycle,
            HostLifecycle::Stopping
        );
        assert_eq!(timed_out.metrics().shutdown_requests, 1);
        assert!(matches!(
            timed_out.run_on_runtime(&runtime),
            Err(NativeRunError::ShutdownTimedOut { .. })
        ));
        assert_eq!(timed_out.metrics().shutdown_requests, 1);
    }
}
