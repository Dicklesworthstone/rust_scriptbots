//! Renderer-neutral host protocol for ScriptBots.
//!
//! This crate deliberately stops at the command, status, snapshot, and event
//! boundary. It does not own a [`scriptbots_core::WorldState`], schedule ticks,
//! open storage, or depend on any renderer, server, or platform runtime. The
//! sole-owner host state machine is built on this contract by the next recovery
//! phase.

#![warn(missing_docs, unsafe_code)]

use scriptbots_core::{DynamicWorldSnapshot, ScriptBotsConfig, SelectionUpdate, Tick};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

macro_rules! monotonic_newtype {
    ($(#[$metadata:meta])* $name:ident) => {
        $(#[$metadata])*
        #[derive(
            Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize,
            Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Construct a value from its protocol representation.
            #[must_use]
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            /// Return the protocol representation.
            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }

            /// Return the following value, or `None` at the end of the domain.
            #[must_use]
            pub const fn checked_next(self) -> Option<Self> {
                match self.0.checked_add(1) {
                    Some(value) => Some(Self(value)),
                    None => None,
                }
            }
        }
    };
}

/// Stable idempotency key supplied by a client for one logical command.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct CommandId(u128);

impl CommandId {
    /// Construct an identifier from its protocol representation.
    #[must_use]
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// Construct a collision-resistant identifier from a stable client namespace and sequence.
    #[must_use]
    pub fn from_client_sequence(client_namespace: u64, sequence: u64) -> Self {
        Self((u128::from(client_namespace) << 64) | u128::from(sequence))
    }

    /// Return the protocol representation.
    #[must_use]
    pub const fn get(self) -> u128 {
        self.0
    }
}

monotonic_newtype!(
    /// Total order assigned to successfully admitted commands.
    AdmissionSequence
);
monotonic_newtype!(
    /// Revision of externally visible control state.
    ControlRevision
);
monotonic_newtype!(
    /// Revision of scientific world state.
    ScientificRevision
);
monotonic_newtype!(
    /// Revision of the active simulation configuration.
    ConfigRevision
);
monotonic_newtype!(
    /// Revision of the immutable snapshot publication stream.
    SnapshotRevision
);
monotonic_newtype!(
    /// Sequence number in the ordered host event stream.
    EventSequence
);

/// Monotonic time supplied by a deterministic or browser-owned driver.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ManualInstant(u64);

impl ManualInstant {
    /// Construct an instant measured in monotonically increasing nanoseconds.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Return the instant in nanoseconds.
    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }
}

/// The independent revision domains observed at one host boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostRevisions {
    /// Revision of playback, lifecycle, and command-visible control state.
    pub control: ControlRevision,
    /// Revision of the scientific simulation state.
    pub scientific: ScientificRevision,
    /// Revision of the active configuration.
    pub config: ConfigRevision,
}

/// Playback state included in every host snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlaybackSnapshot {
    /// Whether automatic ticking is paused.
    pub paused: bool,
    /// Requested tick-rate multiplier.
    pub speed_multiplier: f32,
}

impl Default for PlaybackSnapshot {
    fn default() -> Self {
        Self {
            paused: false,
            speed_multiplier: 1.0,
        }
    }
}

/// Lifecycle visible to clients without exposing the concrete host implementation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostLifecycle {
    /// The host accepts commands and may advance science.
    #[default]
    Running,
    /// Shutdown was admitted but finalization is not complete.
    Stopping,
    /// Finalization completed and no new command may be admitted.
    Stopped,
}

/// Immutable renderer-neutral publication from the sole-owner host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostSnapshot {
    /// Monotonic publication revision.
    pub revision: SnapshotRevision,
    /// Revisions captured atomically with the payload.
    pub revisions: HostRevisions,
    /// Playback state captured at this boundary.
    pub playback: PlaybackSnapshot,
    /// Host lifecycle captured at this boundary.
    pub lifecycle: HostLifecycle,
    /// Existing renderer-neutral dynamic world projection.
    pub world: DynamicWorldSnapshot,
}

/// A state-changing request understood by the runtime boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum HostCommand {
    /// Pause automatic simulation ticks.
    Pause,
    /// Resume automatic simulation ticks.
    Resume,
    /// Set the requested playback multiplier.
    SetSpeed(f32),
    /// Advance exactly one scientific tick while paused.
    Step,
    /// Atomically replace the active simulation configuration.
    UpdateConfig(Box<ScriptBotsConfig>),
    /// Update the selected-agent set at the next ordered command boundary.
    UpdateSelection(SelectionUpdate),
    /// Begin orderly host shutdown.
    Shutdown,
}

impl HostCommand {
    /// Validate input that can be rejected before admission.
    pub fn validate(&self) -> Result<(), CommandValidationError> {
        match self {
            Self::SetSpeed(speed) if !speed.is_finite() || *speed < 0.0 => {
                Err(CommandValidationError::InvalidSpeed)
            }
            Self::UpdateConfig(config) => config.validate().map_err(|error| {
                CommandValidationError::InvalidConfig {
                    message: error.to_string(),
                }
            }),
            _ => Ok(()),
        }
    }
}

/// A command plus its stable identity and optional control-revision compare-and-set guard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandEnvelope {
    /// Stable idempotency key. Retrying this id returns its existing status.
    pub command_id: CommandId,
    /// Reject unless this is the host's current control revision.
    pub expected_control_revision: Option<ControlRevision>,
    /// Requested operation.
    pub command: HostCommand,
}

impl CommandEnvelope {
    /// Construct an unguarded command envelope.
    #[must_use]
    pub const fn new(command_id: CommandId, command: HostCommand) -> Self {
        Self {
            command_id,
            expected_control_revision: None,
            command,
        }
    }

    /// Add an expected control revision for compare-and-set admission.
    #[must_use]
    pub const fn expecting_control_revision(mut self, revision: ControlRevision) -> Self {
        self.expected_control_revision = Some(revision);
        self
    }
}

/// Reason a command was rejected before admission or at its ordered application boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RejectionReason {
    /// The request failed protocol-level validation.
    Validation {
        /// Actionable validation detail.
        message: String,
    },
    /// The optimistic control-revision guard did not match.
    ControlRevisionConflict {
        /// Revision requested by the client.
        expected: ControlRevision,
        /// Current host revision.
        actual: ControlRevision,
    },
    /// The bounded host admission queue had no capacity for this command.
    Overloaded {
        /// Configured admission capacity at the rejected boundary.
        capacity: usize,
    },
    /// The host lifecycle no longer admits new work.
    HostStopping,
}

/// Failure encountered after a command had been admitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplicationFailure {
    /// Stable machine-readable failure category.
    pub code: String,
    /// Human-readable diagnostic detail.
    pub message: String,
}

/// Actual boundary at which an admitted command finished applying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedCommand {
    /// Scientific tick visible after application.
    pub tick: Tick,
    /// Typed revisions visible after application.
    pub revisions: HostRevisions,
}

/// Application axis of a command's status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum ApplicationState {
    /// The command has an admission order but has not finished applying.
    Admitted,
    /// The command applied exactly once.
    Applied(AppliedCommand),
    /// The command was rejected before admission or at its ordered application boundary.
    Rejected(RejectionReason),
    /// The command was admitted but application failed.
    Failed(ApplicationFailure),
}

/// Failure of the independent command-journal axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalFailure {
    /// Stable machine-readable failure category.
    pub code: String,
    /// Human-readable diagnostic detail.
    pub message: String,
}

/// Journal axis of a command's status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum JournalState {
    /// This command does not require a journal record.
    NotRequired,
    /// A required journal record has not committed yet.
    Pending,
    /// The record committed to volatile storage.
    CommittedVolatile,
    /// The record is durable according to the configured storage contract.
    Durable,
    /// Journal persistence failed independently of application.
    Failed(JournalFailure),
}

/// Two-axis status for one stable command id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CommandStatus {
    command_id: CommandId,
    admission_sequence: Option<AdmissionSequence>,
    application: ApplicationState,
    journal: JournalState,
}

impl CommandStatus {
    /// Construct a status after validating cross-axis invariants.
    pub fn try_new(
        command_id: CommandId,
        admission_sequence: Option<AdmissionSequence>,
        application: ApplicationState,
        journal: JournalState,
    ) -> Result<Self, StatusCombinationError> {
        validate_status_combination(admission_sequence, &application, &journal)?;
        Ok(Self {
            command_id,
            admission_sequence,
            application,
            journal,
        })
    }

    /// Construct a pre-admission rejection.
    pub fn rejected(
        command_id: CommandId,
        reason: RejectionReason,
    ) -> Result<Self, StatusCombinationError> {
        Self::try_new(
            command_id,
            None,
            ApplicationState::Rejected(reason),
            JournalState::NotRequired,
        )
    }

    /// Stable command id represented by this status.
    #[must_use]
    pub const fn command_id(&self) -> CommandId {
        self.command_id
    }

    /// Total admission order, absent only for pre-admission rejection.
    #[must_use]
    pub const fn admission_sequence(&self) -> Option<AdmissionSequence> {
        self.admission_sequence
    }

    /// Current application-axis state.
    #[must_use]
    pub const fn application(&self) -> &ApplicationState {
        &self.application
    }

    /// Current journal-axis state.
    #[must_use]
    pub const fn journal(&self) -> &JournalState {
        &self.journal
    }

    /// Revalidate cross-axis invariants after transport deserialization.
    pub fn validate(&self) -> Result<(), StatusCombinationError> {
        validate_status_combination(
            self.admission_sequence,
            &self.application,
            &self.journal,
        )
    }
}

fn validate_status_combination(
    admission_sequence: Option<AdmissionSequence>,
    application: &ApplicationState,
    journal: &JournalState,
) -> Result<(), StatusCombinationError> {
    match application {
        ApplicationState::Admitted => {
            if admission_sequence.is_none() {
                return Err(StatusCombinationError::MissingAdmissionSequence);
            }
            if !matches!(journal, JournalState::NotRequired | JournalState::Pending) {
                return Err(StatusCombinationError::AdmittedJournalAdvanced);
            }
        }
        ApplicationState::Applied(_) | ApplicationState::Failed(_) => {
            if admission_sequence.is_none() {
                return Err(StatusCombinationError::MissingAdmissionSequence);
            }
        }
        ApplicationState::Rejected(reason) => {
            if journal != &JournalState::NotRequired {
                return Err(StatusCombinationError::RejectedWasJournaled);
            }
            match reason {
                RejectionReason::ControlRevisionConflict { .. }
                    if admission_sequence.is_none() =>
                {
                    return Err(StatusCombinationError::ConflictMissingAdmission);
                }
                RejectionReason::Validation { .. }
                | RejectionReason::Overloaded { .. }
                | RejectionReason::HostStopping
                    if admission_sequence.is_some() =>
                {
                    return Err(StatusCombinationError::PreAdmissionRejectionWasAdmitted);
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Invalid cross-axis command status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum StatusCombinationError {
    /// Every admitted, applied, or failed application state needs an admission sequence.
    #[error("an admitted, applied, or failed command requires an admission sequence")]
    MissingAdmissionSequence,
    /// An application-pending command cannot claim a committed or failed journal outcome.
    #[error("an admitted command may only have journal state not_required or pending")]
    AdmittedJournalAdvanced,
    /// A rejected command cannot have journal work.
    #[error("a rejected command must have journal state not_required")]
    RejectedWasJournaled,
    /// An ordered compare-and-set conflict must retain its admission position.
    #[error("a control revision conflict requires an admission sequence")]
    ConflictMissingAdmission,
    /// Validation, overload, and lifecycle rejection happen before admission.
    #[error("a pre-admission rejection cannot have an admission sequence")]
    PreAdmissionRejectionWasAdmitted,
}

#[derive(Deserialize)]
struct CommandStatusWire {
    command_id: CommandId,
    admission_sequence: Option<AdmissionSequence>,
    application: ApplicationState,
    journal: JournalState,
}

impl TryFrom<CommandStatusWire> for CommandStatus {
    type Error = StatusCombinationError;

    fn try_from(wire: CommandStatusWire) -> Result<Self, Self::Error> {
        Self::try_new(
            wire.command_id,
            wire.admission_sequence,
            wire.application,
            wire.journal,
        )
    }
}

impl<'de> Deserialize<'de> for CommandStatus {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = CommandStatusWire::deserialize(deserializer)?;
        Self::try_from(wire).map_err(serde::de::Error::custom)
    }
}

/// Protocol-level command validation failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CommandValidationError {
    /// Playback speed must be finite and non-negative.
    #[error("speed multiplier must be finite and non-negative")]
    InvalidSpeed,
    /// The core configuration contract rejected a replacement.
    #[error("{message}")]
    InvalidConfig {
        /// Core validation diagnostic.
        message: String,
    },
}

/// Ordered event emitted by the host protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostEvent {
    /// Total event order.
    pub sequence: EventSequence,
    /// Scientific tick visible when the event was emitted.
    pub tick: Tick,
    /// Event payload.
    pub kind: HostEventKind,
}

/// Renderer-neutral event payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum HostEventKind {
    /// One command's application or journal status changed.
    CommandStatusChanged(CommandStatus),
    /// A new immutable snapshot became visible.
    SnapshotPublished(SnapshotRevision),
    /// Host lifecycle changed.
    LifecycleChanged(HostLifecycle),
}

/// Opaque client-side cursor into the host event stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EventCursor {
    last_seen: EventSequence,
}

impl EventCursor {
    /// Start before the first event.
    #[must_use]
    pub const fn beginning() -> Self {
        Self {
            last_seen: EventSequence::new(0),
        }
    }

    /// Resume after an already-observed event.
    #[must_use]
    pub const fn after(sequence: EventSequence) -> Self {
        Self {
            last_seen: sequence,
        }
    }

    /// Last event observed through this cursor.
    #[must_use]
    pub const fn last_seen(self) -> EventSequence {
        self.last_seen
    }
}

/// Opaque client-side subscription to immutable snapshots.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SnapshotSubscription {
    last_seen: Option<SnapshotRevision>,
}

impl SnapshotSubscription {
    /// Subscribe to the current publication followed by all newer revisions.
    #[must_use]
    pub const fn current() -> Self {
        Self { last_seen: None }
    }

    /// Resume after an already-observed snapshot.
    #[must_use]
    pub const fn after(revision: SnapshotRevision) -> Self {
        Self {
            last_seen: Some(revision),
        }
    }

    /// Last snapshot observed through this subscription.
    #[must_use]
    pub const fn last_seen(self) -> Option<SnapshotRevision> {
        self.last_seen
    }
}

/// Failure to reach or trust the opaque host port.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HostAccessError {
    /// The transport or in-process port is no longer connected.
    #[error("host port disconnected")]
    Disconnected,
    /// A host implementation violated the public ordering contract.
    #[error("host protocol violation: {message}")]
    ProtocolViolation {
        /// Diagnostic identifying the violated invariant.
        message: String,
    },
    /// A null frontend exhausted its namespaced command-id sequence.
    #[error("command id sequence exhausted")]
    CommandIdExhausted,
}

/// Synchronous, renderer-neutral client port implemented by a host handle.
///
/// Implementations may use channels internally, but the concrete transport is
/// intentionally hidden behind [`HostClient`].
pub trait HostPort {
    /// Submit or retry a logical command.
    fn submit(&mut self, envelope: CommandEnvelope) -> Result<CommandStatus, HostAccessError>;

    /// Look up the latest durable in-process knowledge for a command id.
    fn command_status(
        &mut self,
        command_id: CommandId,
    ) -> Result<Option<CommandStatus>, HostAccessError>;

    /// Return a snapshot newer than `after`, or the current snapshot when `after` is `None`.
    fn snapshot_after(
        &mut self,
        after: Option<SnapshotRevision>,
    ) -> Result<Option<Arc<HostSnapshot>>, HostAccessError>;

    /// Return at most `limit` events whose sequence is strictly greater than the cursor.
    fn events_after(
        &mut self,
        cursor: EventSequence,
        limit: usize,
    ) -> Result<Vec<HostEvent>, HostAccessError>;
}

/// Optional extension for deterministic same-thread and browser-owned hosts.
pub trait ManualHostDriver {
    /// Drive the host to one explicit monotonic time boundary.
    fn drive(&mut self, now: ManualInstant) -> Result<DriveReceipt, HostAccessError>;
}

/// Result of one explicit manual-drive boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriveReceipt {
    /// Time boundary supplied by the driver.
    pub now: ManualInstant,
    /// Commands whose application completed during this drive.
    pub commands_completed: usize,
    /// Snapshots published during this drive.
    pub snapshots_published: usize,
    /// Events published during this drive.
    pub events_published: usize,
}

/// Typed client that owns, but never exposes, its concrete host port.
#[derive(Clone)]
pub struct HostClient<P> {
    port: P,
}

impl<P: HostPort> HostClient<P> {
    /// Wrap one concrete port behind the typed client API.
    #[must_use]
    pub const fn new(port: P) -> Self {
        Self { port }
    }

    /// Submit a new command or retry an existing idempotency key.
    pub fn submit(
        &mut self,
        envelope: CommandEnvelope,
    ) -> Result<CommandStatus, HostAccessError> {
        let requested_id = envelope.command_id;
        let status = self.port.submit(envelope)?;
        if status.command_id() != requested_id {
            return Err(protocol_violation("submission returned a different command id"));
        }
        status
            .validate()
            .map_err(|error| protocol_violation(error.to_string()))?;
        Ok(status)
    }

    /// Look up a command after the submitting client has disconnected or restarted.
    pub fn command_status(
        &mut self,
        command_id: CommandId,
    ) -> Result<Option<CommandStatus>, HostAccessError> {
        let status = self.port.command_status(command_id)?;
        if status
            .as_ref()
            .is_some_and(|status| status.command_id() != command_id)
        {
            return Err(protocol_violation("lookup returned a different command id"));
        }
        if let Some(status) = &status {
            status
                .validate()
                .map_err(|error| protocol_violation(error.to_string()))?;
        }
        Ok(status)
    }

    /// Create a snapshot subscription starting with the current publication.
    #[must_use]
    pub const fn subscribe_snapshots(&self) -> SnapshotSubscription {
        SnapshotSubscription::current()
    }

    /// Poll one immutable snapshot and advance the subscription only on success.
    pub fn poll_snapshot(
        &mut self,
        subscription: &mut SnapshotSubscription,
    ) -> Result<Option<Arc<HostSnapshot>>, HostAccessError> {
        let snapshot = self.port.snapshot_after(subscription.last_seen)?;
        if let Some(snapshot) = snapshot {
            if subscription
                .last_seen
                .is_some_and(|seen| snapshot.revision <= seen)
            {
                return Err(protocol_violation(
                    "snapshot revision did not advance beyond the subscription",
                ));
            }
            subscription.last_seen = Some(snapshot.revision);
            Ok(Some(snapshot))
        } else {
            Ok(None)
        }
    }

    /// Create an event cursor positioned before the first event.
    #[must_use]
    pub const fn event_cursor(&self) -> EventCursor {
        EventCursor::beginning()
    }

    /// Read ordered events and advance the cursor after the last valid event.
    pub fn read_events(
        &mut self,
        cursor: &mut EventCursor,
        limit: usize,
    ) -> Result<Vec<HostEvent>, HostAccessError> {
        let events = self.port.events_after(cursor.last_seen, limit)?;
        if events.len() > limit {
            return Err(protocol_violation("event port exceeded the requested limit"));
        }
        let mut previous = cursor.last_seen;
        for event in &events {
            let expected = previous
                .checked_next()
                .ok_or_else(|| protocol_violation("event cursor sequence exhausted"))?;
            if event.sequence != expected {
                return Err(protocol_violation("event sequence was not contiguous"));
            }
            if let HostEventKind::CommandStatusChanged(status) = &event.kind {
                status
                    .validate()
                    .map_err(|error| protocol_violation(error.to_string()))?;
            }
            previous = event.sequence;
        }
        cursor.last_seen = previous;
        Ok(events)
    }
}

/// Headless reference frontend used by conformance tests and embedders.
///
/// It exercises only public client operations and owns no world, lock, storage
/// connection, renderer, server, or scheduler.
pub struct NullFrontend<P> {
    client: HostClient<P>,
    client_namespace: u64,
    next_sequence: Option<u64>,
    snapshots: SnapshotSubscription,
    events: EventCursor,
    last_drive: Option<ManualInstant>,
}

impl<P: HostPort> NullFrontend<P> {
    /// Construct a frontend with a stable command-id namespace.
    #[must_use]
    pub const fn new(port: P, client_namespace: u64) -> Self {
        Self {
            client: HostClient::new(port),
            client_namespace,
            next_sequence: Some(1),
            snapshots: SnapshotSubscription::current(),
            events: EventCursor::beginning(),
            last_drive: None,
        }
    }

    /// Submit an arbitrary command with an optional control-revision guard.
    pub fn submit(
        &mut self,
        command: HostCommand,
        expected_control_revision: Option<ControlRevision>,
    ) -> Result<CommandStatus, HostAccessError> {
        let sequence = self
            .next_sequence
            .ok_or(HostAccessError::CommandIdExhausted)?;
        self.next_sequence = sequence.checked_add(1);
        let mut envelope = CommandEnvelope::new(
            CommandId::from_client_sequence(self.client_namespace, sequence),
            command,
        );
        envelope.expected_control_revision = expected_control_revision;
        self.client.submit(envelope)
    }

    /// Pause automatic ticks.
    pub fn pause(&mut self) -> Result<CommandStatus, HostAccessError> {
        self.submit(HostCommand::Pause, None)
    }

    /// Resume automatic ticks.
    pub fn resume(&mut self) -> Result<CommandStatus, HostAccessError> {
        self.submit(HostCommand::Resume, None)
    }

    /// Set the playback multiplier.
    pub fn set_speed(&mut self, speed: f32) -> Result<CommandStatus, HostAccessError> {
        self.submit(HostCommand::SetSpeed(speed), None)
    }

    /// Request exactly one scientific tick.
    pub fn step(&mut self) -> Result<CommandStatus, HostAccessError> {
        self.submit(HostCommand::Step, None)
    }

    /// Replace the active simulation configuration.
    pub fn update_config(
        &mut self,
        config: ScriptBotsConfig,
    ) -> Result<CommandStatus, HostAccessError> {
        self.submit(HostCommand::UpdateConfig(Box::new(config)), None)
    }

    /// Request orderly host shutdown.
    pub fn shutdown(&mut self) -> Result<CommandStatus, HostAccessError> {
        self.submit(HostCommand::Shutdown, None)
    }

    /// Look up the latest status for any command id.
    pub fn command_status(
        &mut self,
        command_id: CommandId,
    ) -> Result<Option<CommandStatus>, HostAccessError> {
        self.client.command_status(command_id)
    }

    /// Poll the next immutable snapshot for this frontend.
    pub fn poll_snapshot(&mut self) -> Result<Option<Arc<HostSnapshot>>, HostAccessError> {
        self.client.poll_snapshot(&mut self.snapshots)
    }

    /// Read ordered host events and advance this frontend's cursor.
    pub fn read_events(&mut self, limit: usize) -> Result<Vec<HostEvent>, HostAccessError> {
        self.client.read_events(&mut self.events, limit)
    }

    /// Drive a separately owned synchronous host to a caller-owned time boundary.
    pub fn drive_at(
        &mut self,
        driver: &mut impl ManualHostDriver,
        now: ManualInstant,
    ) -> Result<DriveReceipt, HostAccessError> {
        if self.last_drive.is_some_and(|last_drive| now < last_drive) {
            return Err(protocol_violation("null frontend manual time moved backwards"));
        }
        let receipt = driver.drive(now)?;
        self.last_drive = Some(now);
        Ok(receipt)
    }
}

fn protocol_violation(message: impl Into<String>) -> HostAccessError {
    HostAccessError::ProtocolViolation {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{DynamicSnapshotSummary, DynamicSnapshotWorld};
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::sync::{Barrier, Mutex, MutexGuard};

    #[derive(Clone)]
    struct SharedFakeHost {
        inner: Arc<Mutex<FakeHost>>,
    }

    struct SharedFakeDriver {
        inner: Arc<Mutex<FakeHost>>,
    }

    impl SharedFakeHost {
        fn new() -> Self {
            Self {
                inner: Arc::new(Mutex::new(FakeHost::new())),
            }
        }

        fn lock(&self) -> MutexGuard<'_, FakeHost> {
            self.inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }

        fn fail_on_application(&self, command_id: CommandId) {
            self.lock().fail_on_application.insert(command_id);
        }

        fn driver(&self) -> SharedFakeDriver {
            SharedFakeDriver {
                inner: Arc::clone(&self.inner),
            }
        }
    }

    impl HostPort for SharedFakeHost {
        fn submit(
            &mut self,
            envelope: CommandEnvelope,
        ) -> Result<CommandStatus, HostAccessError> {
            self.lock().submit(envelope)
        }

        fn command_status(
            &mut self,
            command_id: CommandId,
        ) -> Result<Option<CommandStatus>, HostAccessError> {
            Ok(self.lock().statuses.get(&command_id).cloned())
        }

        fn snapshot_after(
            &mut self,
            after: Option<SnapshotRevision>,
        ) -> Result<Option<Arc<HostSnapshot>>, HostAccessError> {
            let host = self.lock();
            Ok(match after {
                None => host.snapshots.last().cloned(),
                Some(revision) => host
                    .snapshots
                    .iter()
                    .find(|snapshot| snapshot.revision > revision)
                    .cloned(),
            })
        }

        fn events_after(
            &mut self,
            cursor: EventSequence,
            limit: usize,
        ) -> Result<Vec<HostEvent>, HostAccessError> {
            Ok(self
                .lock()
                .events
                .iter()
                .filter(|event| event.sequence > cursor)
                .take(limit)
                .cloned()
                .collect())
        }
    }

    impl ManualHostDriver for SharedFakeDriver {
        fn drive(&mut self, now: ManualInstant) -> Result<DriveReceipt, HostAccessError> {
            self.inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .drive(now)
        }
    }

    struct FakeHost {
        now: ManualInstant,
        next_admission: AdmissionSequence,
        next_event: EventSequence,
        next_snapshot: SnapshotRevision,
        revisions: HostRevisions,
        tick: Tick,
        playback: PlaybackSnapshot,
        lifecycle: HostLifecycle,
        config: ScriptBotsConfig,
        queue: VecDeque<CommandEnvelope>,
        statuses: HashMap<CommandId, CommandStatus>,
        admission_order: Vec<CommandId>,
        snapshots: Vec<Arc<HostSnapshot>>,
        events: Vec<HostEvent>,
        fail_on_application: HashSet<CommandId>,
    }

    impl FakeHost {
        fn new() -> Self {
            let config = ScriptBotsConfig::default();
            let mut host = Self {
                now: ManualInstant::default(),
                next_admission: AdmissionSequence::new(1),
                next_event: EventSequence::new(1),
                next_snapshot: SnapshotRevision::new(1),
                revisions: HostRevisions::default(),
                tick: Tick(0),
                playback: PlaybackSnapshot::default(),
                lifecycle: HostLifecycle::Running,
                config,
                queue: VecDeque::new(),
                statuses: HashMap::new(),
                admission_order: Vec::new(),
                snapshots: Vec::new(),
                events: Vec::new(),
                fail_on_application: HashSet::new(),
            };
            host.publish_snapshot();
            host.events.clear();
            host.next_event = EventSequence::new(1);
            host
        }

        fn submit(
            &mut self,
            envelope: CommandEnvelope,
        ) -> Result<CommandStatus, HostAccessError> {
            if let Some(status) = self.statuses.get(&envelope.command_id) {
                return Ok(status.clone());
            }

            let rejection = if self.lifecycle != HostLifecycle::Running {
                Some(RejectionReason::HostStopping)
            } else if let Err(error) = envelope.command.validate() {
                Some(RejectionReason::Validation {
                    message: error.to_string(),
                })
            } else {
                None
            };

            if let Some(reason) = rejection {
                let status = CommandStatus::rejected(envelope.command_id, reason)
                    .map_err(|error| protocol_violation(error.to_string()))?;
                self.statuses.insert(envelope.command_id, status.clone());
                self.emit_status(status.clone());
                return Ok(status);
            }

            let admission = self.next_admission;
            self.next_admission = admission
                .checked_next()
                .ok_or_else(|| protocol_violation("admission sequence exhausted"))?;
            let status = CommandStatus::try_new(
                envelope.command_id,
                Some(admission),
                ApplicationState::Admitted,
                JournalState::Pending,
            )
            .map_err(|error| protocol_violation(error.to_string()))?;
            self.admission_order.push(envelope.command_id);
            self.queue.push_back(envelope);
            self.statuses.insert(status.command_id(), status.clone());
            self.emit_status(status.clone());
            Ok(status)
        }

        fn drive(&mut self, now: ManualInstant) -> Result<DriveReceipt, HostAccessError> {
            if now < self.now {
                return Err(protocol_violation("manual time moved backwards"));
            }
            self.now = now;
            let events_before = self.events.len();
            let mut commands_completed = 0;
            while let Some(envelope) = self.queue.pop_front() {
                self.apply(envelope)?;
                commands_completed += 1;
            }
            let snapshots_published = usize::from(commands_completed != 0);
            if snapshots_published != 0 {
                self.publish_snapshot();
            }
            Ok(DriveReceipt {
                now,
                commands_completed,
                snapshots_published,
                events_published: self.events.len() - events_before,
            })
        }

        fn apply(&mut self, envelope: CommandEnvelope) -> Result<(), HostAccessError> {
            let admission = self
                .statuses
                .get(&envelope.command_id)
                .and_then(CommandStatus::admission_sequence)
                .ok_or_else(|| protocol_violation("queued command was not admitted"))?;

            let application = if let Some(expected) = envelope.expected_control_revision
                && expected != self.revisions.control
            {
                ApplicationState::Rejected(RejectionReason::ControlRevisionConflict {
                    expected,
                    actual: self.revisions.control,
                })
            } else if self.fail_on_application.remove(&envelope.command_id) {
                ApplicationState::Failed(ApplicationFailure {
                    code: "injected_conformance_failure".to_owned(),
                    message: "test host refused application".to_owned(),
                })
            } else {
                self.revisions.control = self
                    .revisions
                    .control
                    .checked_next()
                    .ok_or_else(|| protocol_violation("control revision exhausted"))?;
                match envelope.command {
                    HostCommand::Pause => self.playback.paused = true,
                    HostCommand::Resume => self.playback.paused = false,
                    HostCommand::SetSpeed(speed) => self.playback.speed_multiplier = speed,
                    HostCommand::Step => {
                        self.tick.0 = self
                            .tick
                            .0
                            .checked_add(1)
                            .ok_or_else(|| protocol_violation("tick exhausted"))?;
                        self.revisions.scientific = self
                            .revisions
                            .scientific
                            .checked_next()
                            .ok_or_else(|| protocol_violation("scientific revision exhausted"))?;
                    }
                    HostCommand::UpdateConfig(config) => {
                        self.config = *config;
                        self.revisions.config = self
                            .revisions
                            .config
                            .checked_next()
                            .ok_or_else(|| protocol_violation("config revision exhausted"))?;
                    }
                    HostCommand::UpdateSelection(_) => {}
                    HostCommand::Shutdown => self.lifecycle = HostLifecycle::Stopped,
                }
                ApplicationState::Applied(AppliedCommand {
                    tick: self.tick,
                    revisions: self.revisions,
                })
            };

            let journal = if matches!(&application, ApplicationState::Rejected(_)) {
                JournalState::NotRequired
            } else {
                JournalState::Durable
            };
            let status = CommandStatus::try_new(
                envelope.command_id,
                Some(admission),
                application,
                journal,
            )
            .map_err(|error| protocol_violation(error.to_string()))?;
            self.statuses.insert(envelope.command_id, status.clone());
            self.emit_status(status);
            if self.lifecycle == HostLifecycle::Stopped {
                self.emit(HostEventKind::LifecycleChanged(HostLifecycle::Stopped));
            }
            Ok(())
        }

        fn publish_snapshot(&mut self) {
            let revision = self.next_snapshot;
            self.next_snapshot = revision
                .checked_next()
                .expect("test snapshot sequence must have headroom");
            self.snapshots.push(Arc::new(HostSnapshot {
                revision,
                revisions: self.revisions,
                playback: self.playback,
                lifecycle: self.lifecycle,
                world: DynamicWorldSnapshot {
                    tick: self.tick.0,
                    epoch: 0,
                    world: DynamicSnapshotWorld {
                        width: self.config.world_width,
                        height: self.config.world_height,
                        closed: self.config.closed,
                    },
                    summary: DynamicSnapshotSummary {
                        agent_count: 0,
                        births: 0,
                        deaths: 0,
                        total_energy: 0.0,
                        average_energy: 0.0,
                        average_health: 0.0,
                    },
                    agents: Vec::new(),
                },
            }));
            self.emit(HostEventKind::SnapshotPublished(revision));
        }

        fn emit_status(&mut self, status: CommandStatus) {
            self.emit(HostEventKind::CommandStatusChanged(status));
        }

        fn emit(&mut self, kind: HostEventKind) {
            let sequence = self.next_event;
            self.next_event = sequence
                .checked_next()
                .expect("test event sequence must have headroom");
            self.events.push(HostEvent {
                sequence,
                tick: self.tick,
                kind,
            });
        }
    }

    const fn envelope(id: u128, command: HostCommand) -> CommandEnvelope {
        CommandEnvelope::new(CommandId::new(id), command)
    }

    fn submit_ok(client: &mut HostClient<SharedFakeHost>, envelope: CommandEnvelope) -> CommandStatus {
        client
            .submit(envelope)
            .expect("the conformance host should accept this request")
    }

    fn applied(status: &CommandStatus) -> AppliedCommand {
        match status.application() {
            ApplicationState::Applied(applied) => *applied,
            state => panic!("expected applied status, got {state:?}"),
        }
    }

    #[test]
    fn admission_is_totally_ordered_and_duplicate_ids_never_reapply() {
        let shared = SharedFakeHost::new();
        let mut driver = shared.driver();
        let mut client = HostClient::new(shared.clone());
        let first = submit_ok(&mut client, envelope(1, HostCommand::Pause));
        let second_envelope = envelope(2, HostCommand::Resume);
        let second = submit_ok(&mut client, second_envelope.clone());
        let third = submit_ok(&mut client, envelope(3, HostCommand::Step));

        assert_eq!(first.admission_sequence(), Some(AdmissionSequence::new(1)));
        assert_eq!(second.admission_sequence(), Some(AdmissionSequence::new(2)));
        assert_eq!(third.admission_sequence(), Some(AdmissionSequence::new(3)));
        assert_eq!(submit_ok(&mut client, second_envelope), second);

        let receipt = driver
            .drive(ManualInstant::from_nanos(1))
            .expect("drive should succeed");
        assert_eq!(receipt.commands_completed, 3);
        let retried = submit_ok(
            &mut client,
            envelope(2, HostCommand::SetSpeed(99.0)),
        );
        assert!(matches!(retried.application(), ApplicationState::Applied(_)));
        assert_eq!(retried.admission_sequence(), second.admission_sequence());
        assert_eq!(
            driver
                .drive(ManualInstant::from_nanos(2))
                .expect("empty drive should succeed")
                .commands_completed,
            0
        );
        assert_eq!(
            shared.lock().admission_order,
            vec![CommandId::new(1), CommandId::new(2), CommandId::new(3)]
        );
    }

    #[test]
    fn concurrent_retry_of_one_id_gets_one_admission() {
        let shared = SharedFakeHost::new();
        let barrier = Arc::new(Barrier::new(3));
        let request = envelope(44, HostCommand::Step);

        let spawn_submitter = |port: SharedFakeHost| {
            let barrier = Arc::clone(&barrier);
            let request = request.clone();
            std::thread::spawn(move || {
                let mut client = HostClient::new(port);
                barrier.wait();
                client.submit(request)
            })
        };
        let left = spawn_submitter(shared.clone());
        let right = spawn_submitter(shared.clone());
        barrier.wait();

        let left = left.join().expect("left submitter should not panic").expect("left retry");
        let right = right
            .join()
            .expect("right submitter should not panic")
            .expect("right retry");
        assert_eq!(left, right);
        assert_eq!(left.admission_sequence(), Some(AdmissionSequence::new(1)));
        assert_eq!(shared.lock().admission_order, vec![CommandId::new(44)]);
    }

    #[test]
    fn compare_and_set_validation_and_application_failures_stay_distinct() {
        let shared = SharedFakeHost::new();
        let mut driver = shared.driver();
        let mut client = HostClient::new(shared.clone());
        let winner = client
            .submit(
                envelope(9, HostCommand::Pause)
                    .expecting_control_revision(ControlRevision::new(0)),
            )
            .expect("first compare-and-set candidate should be admitted");
        let conflict = client
            .submit(
                envelope(10, HostCommand::Resume)
                    .expecting_control_revision(ControlRevision::new(0)),
            )
            .expect("competing compare-and-set candidate should be admitted");
        assert!(matches!(winner.application(), ApplicationState::Admitted));
        assert!(matches!(conflict.application(), ApplicationState::Admitted));
        assert_eq!(
            conflict.admission_sequence(),
            Some(AdmissionSequence::new(2))
        );
        driver
            .drive(ManualInstant::from_nanos(1))
            .expect("conflict should resolve at the application boundary");
        let conflict = client
            .command_status(conflict.command_id())
            .expect("conflict lookup")
            .expect("conflict remains queryable");
        assert!(matches!(
            conflict.application(),
            ApplicationState::Rejected(RejectionReason::ControlRevisionConflict {
                expected,
                actual,
            }) if *expected == ControlRevision::new(0) && *actual == ControlRevision::new(1)
        ));
        assert_eq!(
            conflict.admission_sequence(),
            Some(AdmissionSequence::new(2))
        );
        assert_eq!(conflict.journal(), &JournalState::NotRequired);

        let mut invalid_config = ScriptBotsConfig::default();
        invalid_config.world_width = 0;
        let rejected = client
            .submit(envelope(
                11,
                HostCommand::UpdateConfig(Box::new(invalid_config)),
            ))
            .expect("validation rejection should be inspectable");
        assert!(matches!(
            rejected.application(),
            ApplicationState::Rejected(RejectionReason::Validation { .. })
        ));
        assert_eq!(rejected.journal(), &JournalState::NotRequired);

        let failed_id = CommandId::new(12);
        let admitted = submit_ok(&mut client, envelope(12, HostCommand::Step));
        assert!(matches!(admitted.application(), ApplicationState::Admitted));
        shared.fail_on_application(failed_id);
        driver
            .drive(ManualInstant::from_nanos(2))
            .expect("drive should report application through status");
        let failed = client
            .command_status(failed_id)
            .expect("lookup should succeed")
            .expect("failed command should remain queryable");
        assert!(matches!(failed.application(), ApplicationState::Failed(_)));
        assert_eq!(failed.journal(), &JournalState::Durable);
    }

    #[test]
    fn a_later_client_can_lookup_status_after_submitter_disconnects() {
        let shared = SharedFakeHost::new();
        let command_id = {
            let mut submitting_client = HostClient::new(shared.clone());
            let status = submit_ok(&mut submitting_client, envelope(70, HostCommand::Pause));
            status.command_id()
        };

        let mut later_client = HostClient::new(shared);
        let status = later_client
            .command_status(command_id)
            .expect("reconnected lookup should succeed")
            .expect("admitted command should still exist");
        assert_eq!(status.command_id(), command_id);
        assert_eq!(status.admission_sequence(), Some(AdmissionSequence::new(1)));
    }

    #[test]
    fn revisions_snapshots_and_events_are_monotonic_in_their_typed_domains() {
        let shared = SharedFakeHost::new();
        let mut driver = shared.driver();
        let mut client = HostClient::new(shared);
        let mut snapshots = client.subscribe_snapshots();
        let initial = client
            .poll_snapshot(&mut snapshots)
            .expect("initial snapshot poll")
            .expect("fake host publishes an initial snapshot");

        let pause = submit_ok(&mut client, envelope(80, HostCommand::Pause));
        driver
            .drive(ManualInstant::from_nanos(1))
            .expect("pause drive");
        let pause = client
            .command_status(pause.command_id())
            .expect("pause lookup")
            .expect("pause status");
        let pause = applied(&pause);
        let after_pause = client
            .poll_snapshot(&mut snapshots)
            .expect("pause snapshot poll")
            .expect("pause should publish");

        let config = submit_ok(
            &mut client,
            envelope(
                81,
                HostCommand::UpdateConfig(Box::new(ScriptBotsConfig::default())),
            ),
        );
        driver
            .drive(ManualInstant::from_nanos(2))
            .expect("config drive");
        let config = applied(
            &client
                .command_status(config.command_id())
                .expect("config lookup")
                .expect("config status"),
        );
        let after_config = client
            .poll_snapshot(&mut snapshots)
            .expect("config snapshot poll")
            .expect("config should publish");

        let step = submit_ok(&mut client, envelope(82, HostCommand::Step));
        driver
            .drive(ManualInstant::from_nanos(3))
            .expect("step drive");
        let step = applied(
            &client
                .command_status(step.command_id())
                .expect("step lookup")
                .expect("step status"),
        );
        let after_step = client
            .poll_snapshot(&mut snapshots)
            .expect("step snapshot poll")
            .expect("step should publish");

        assert!(initial.revision < after_pause.revision);
        assert!(after_pause.revision < after_config.revision);
        assert!(after_config.revision < after_step.revision);
        assert_eq!(pause.revisions.control, ControlRevision::new(1));
        assert_eq!(pause.revisions.scientific, ScientificRevision::new(0));
        assert_eq!(config.revisions.control, ControlRevision::new(2));
        assert_eq!(config.revisions.config, ConfigRevision::new(1));
        assert_eq!(step.revisions.control, ControlRevision::new(3));
        assert_eq!(step.revisions.scientific, ScientificRevision::new(1));
        assert_eq!(step.tick, Tick(1));

        let mut cursor = client.event_cursor();
        let events = client
            .read_events(&mut cursor, usize::MAX)
            .expect("ordered event read");
        assert!(events.len() >= 9);
        assert!(events
            .windows(2)
            .all(|pair| pair[0].sequence.checked_next() == Some(pair[1].sequence)));
        assert_eq!(cursor.last_seen(), events.last().expect("events exist").sequence);
    }

    #[test]
    fn status_constructor_accepts_every_reachable_axis_combination_only() {
        let terminal_applications = [
            ApplicationState::Applied(AppliedCommand {
                tick: Tick(4),
                revisions: HostRevisions::default(),
            }),
            ApplicationState::Failed(ApplicationFailure {
                code: "apply".to_owned(),
                message: "failed".to_owned(),
            }),
        ];
        let journals = [
            JournalState::NotRequired,
            JournalState::Pending,
            JournalState::CommittedVolatile,
            JournalState::Durable,
            JournalState::Failed(JournalFailure {
                code: "journal".to_owned(),
                message: "failed".to_owned(),
            }),
        ];

        for journal in [JournalState::NotRequired, JournalState::Pending] {
            assert!(CommandStatus::try_new(
                CommandId::new(1),
                Some(AdmissionSequence::new(1)),
                ApplicationState::Admitted,
                journal,
            )
            .is_ok());
        }
        for application in terminal_applications {
            for journal in journals.clone() {
                assert!(CommandStatus::try_new(
                    CommandId::new(1),
                    Some(AdmissionSequence::new(1)),
                    application.clone(),
                    journal,
                )
                .is_ok());
            }
        }
        let rejected = ApplicationState::Rejected(RejectionReason::HostStopping);
        assert!(CommandStatus::try_new(
            CommandId::new(2),
            None,
            rejected.clone(),
            JournalState::NotRequired,
        )
        .is_ok());
        assert_eq!(
            CommandStatus::try_new(
                CommandId::new(2),
                Some(AdmissionSequence::new(2)),
                rejected.clone(),
                JournalState::NotRequired,
            ),
            Err(StatusCombinationError::PreAdmissionRejectionWasAdmitted)
        );
        let conflict = ApplicationState::Rejected(
            RejectionReason::ControlRevisionConflict {
                expected: ControlRevision::new(1),
                actual: ControlRevision::new(2),
            },
        );
        assert!(CommandStatus::try_new(
            CommandId::new(2),
            Some(AdmissionSequence::new(2)),
            conflict.clone(),
            JournalState::NotRequired,
        )
        .is_ok());
        assert_eq!(
            CommandStatus::try_new(
                CommandId::new(2),
                None,
                conflict,
                JournalState::NotRequired,
            ),
            Err(StatusCombinationError::ConflictMissingAdmission)
        );
        assert_eq!(
            CommandStatus::try_new(
                CommandId::new(2),
                None,
                rejected,
                JournalState::Pending,
            ),
            Err(StatusCombinationError::RejectedWasJournaled)
        );
        assert_eq!(
            CommandStatus::try_new(
                CommandId::new(3),
                None,
                ApplicationState::Admitted,
                JournalState::NotRequired,
            ),
            Err(StatusCombinationError::MissingAdmissionSequence)
        );
        assert_eq!(
            CommandStatus::try_new(
                CommandId::new(4),
                Some(AdmissionSequence::new(4)),
                ApplicationState::Admitted,
                JournalState::Durable,
            ),
            Err(StatusCombinationError::AdmittedJournalAdvanced)
        );
        assert_eq!(
            CommandStatus::try_from(CommandStatusWire {
                command_id: CommandId::new(5),
                admission_sequence: None,
                application: ApplicationState::Applied(AppliedCommand {
                    tick: Tick(0),
                    revisions: HostRevisions::default(),
                }),
                journal: JournalState::Durable,
            }),
            Err(StatusCombinationError::MissingAdmissionSequence)
        );
    }

    #[test]
    fn null_frontend_uses_only_commands_observation_and_manual_drive() {
        let shared = SharedFakeHost::new();
        let mut driver = shared.driver();
        let mut frontend = NullFrontend::new(shared, 0x51);
        let statuses = [
            frontend.pause().expect("pause submission"),
            frontend.resume().expect("resume submission"),
            frontend.set_speed(2.5).expect("speed submission"),
            frontend.step().expect("step submission"),
            frontend
                .update_config(ScriptBotsConfig::default())
                .expect("config submission"),
            frontend.shutdown().expect("shutdown submission"),
        ];
        assert!(statuses.iter().enumerate().all(|(index, status)| {
            let sequence = u64::try_from(index).expect("test command count fits u64") + 1;
            status.command_id() == CommandId::from_client_sequence(0x51, sequence)
        }));

        let receipt = frontend
            .drive_at(&mut driver, ManualInstant::from_nanos(10))
            .expect("manual drive");
        assert_eq!(receipt.commands_completed, statuses.len());
        let snapshot = frontend
            .poll_snapshot()
            .expect("snapshot poll")
            .expect("drive should publish a snapshot");
        assert_eq!(snapshot.playback.speed_multiplier, 2.5);
        assert_eq!(snapshot.world.tick, 1);
        assert_eq!(snapshot.lifecycle, HostLifecycle::Stopped);
        assert!(!frontend
            .read_events(128)
            .expect("event observation")
            .is_empty());
        assert!(matches!(
            frontend
                .command_status(statuses[3].command_id())
                .expect("status lookup")
                .expect("step remains queryable")
                .application(),
            ApplicationState::Applied(_)
        ));
        assert!(frontend
            .drive_at(&mut driver, ManualInstant::from_nanos(9))
            .is_err());
    }
}
