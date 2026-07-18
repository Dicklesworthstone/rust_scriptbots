//! Deterministic sole-owner simulation host.

use super::{
    AdmissionSequence, ApplicationFailure, ApplicationState, AppliedCommand, BrainProjection,
    BrainProjectionRequest, BrainProjectionSource, CommandEnvelope, CommandId,
    CommandLifecycleEvidence, CommandStatus, ConfigRevision, ControlRevision, DriveReceipt,
    EventCatchUp, EventCatchUpGuarantee, EventCatchUpLocator, EventCatchUpUnavailableReason,
    EventCommitment, EventHub, EventJournalReader, EventPage, EventPageSource,
    EventRetentionSnapshot, EventSequence, EventSequenceRange, FoodLayerSnapshot, HostAccessError,
    HostBlocker, HostCommand, HostDriveInterest, HostEvent, HostEventKind, HostFault, HostHealth,
    HostLifecycle, HostPort, HostRevisions, HostSessionId, HydrologyLayerSnapshot,
    HydrologyTileSnapshot, JournalAdmission, JournalBatch, JournalBatchId, JournalFailure,
    JournalPort, JournalReceipt, JournalReceiptState, JournalState, JournaledScientificEvent,
    LayerRevision, ManualHostDriver, ManualInstant, PlaybackSnapshot, ProtocolEventSequence,
    RejectionReason, RenderSnapshot, ScientificBoundary, ScientificBoundaryFault, ScientificEvent,
    ScientificRevision, ShutdownCommitRequirement, SnapshotBuildStats, SnapshotHub,
    SnapshotLayerRevisions, SnapshotLayers, SnapshotRevision, StatusCombinationError,
    TerrainLayerSnapshot, TerrainTileSnapshot,
};
use arc_swap::ArcSwap;
use scriptbots_core::{
    ACTIVATION_CAPTURE_BUDGET, BrainInspectionClientId, BrainInspectionError,
    BrainInspectionRequest, BrainInspectionRevision, CharacterizationError, CompletedStepFault,
    DynamicAgentSnapshot, DynamicWorldSnapshot, NullPersistence, PersistenceAdmissionSession,
    PersistenceSessionError, ScriptBotsConfig, Tick, TickSummary, WorldDigestV1, WorldState,
};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
    mem::size_of,
    rc::Rc,
    sync::Arc,
};
use thiserror::Error;

const SPEED_SCALE: u128 = 1_000_000;
const DEFAULT_TICK_PERIOD_NANOS: u64 = 16_666_667;
const DEFAULT_COMMAND_CAPACITY: usize = 32;
const DEFAULT_MAX_AUTOMATIC_STEPS: usize = 8;
const DEFAULT_SNAPSHOT_INTERVAL_TICKS: u64 = 1;
const DEFAULT_PROTOCOL_EVENT_CAPACITY: usize = 256;
const DEFAULT_SCIENTIFIC_EVENT_CAPACITY: usize = 64;
const DEFAULT_VOLATILE_EVENT_HISTORY_CAPACITY: usize = 512;
const RECEIPT_POLL_LIMIT: usize = 4_096;
const LIFECYCLE_COMMAND_NAMESPACE: u64 = u64::MAX;

/// Construction options for a synchronous [`HostCore`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostCoreOptions {
    /// Playback state visible before the first manual-drive boundary.
    pub initial_playback: PlaybackSnapshot,
    /// Maximum number of admitted envelopes waiting for ordered application.
    pub command_capacity: usize,
    /// Nominal duration of one automatic scientific step.
    pub tick_period_nanos: u64,
    /// Maximum automatic catch-up work performed by one drive call.
    pub max_automatic_steps_per_drive: usize,
    /// Completed automatic science revisions between render publications.
    ///
    /// Control, lifecycle, health, configuration, and explicit-step changes still publish
    /// immediately. This deterministic stride changes presentation work only.
    pub snapshot_interval_ticks: u64,
    /// Ephemeral command/lifecycle/health notifications retained for diagnostics.
    pub protocol_event_capacity: usize,
    /// Canonical scientific records retained by the detached latest hot ring.
    pub scientific_event_capacity: usize,
    /// Exact committed batches retained by the default live-memory catch-up journal.
    pub volatile_event_history_capacity: usize,
}

impl Default for HostCoreOptions {
    fn default() -> Self {
        Self {
            initial_playback: PlaybackSnapshot::default(),
            command_capacity: DEFAULT_COMMAND_CAPACITY,
            tick_period_nanos: DEFAULT_TICK_PERIOD_NANOS,
            max_automatic_steps_per_drive: DEFAULT_MAX_AUTOMATIC_STEPS,
            snapshot_interval_ticks: DEFAULT_SNAPSHOT_INTERVAL_TICKS,
            protocol_event_capacity: DEFAULT_PROTOCOL_EVENT_CAPACITY,
            scientific_event_capacity: DEFAULT_SCIENTIFIC_EVENT_CAPACITY,
            volatile_event_history_capacity: DEFAULT_VOLATILE_EVENT_HISTORY_CAPACITY,
        }
    }
}

/// Failure to construct a sole-owner host without changing the supplied world.
#[derive(Debug, Error)]
pub enum HostCoreBuildError {
    /// One of the deterministic scheduling bounds was zero or invalid.
    #[error("invalid HostCore option: {message}")]
    InvalidOptions {
        /// Actionable option diagnostic.
        message: String,
    },
    /// The world had already surrendered its one lifetime persistence binding.
    #[error(transparent)]
    Persistence(#[from] PersistenceSessionError),
}

/// Same-thread volatile journal used by [`HostCore::new`].
///
/// Admission is immediate and nonblocking. A volatile receipt is queued for the
/// next drive boundary so application and journal state still advance on their
/// independent protocol axes.
#[derive(Debug)]
pub struct VolatileJournal {
    highest_accepted: Option<JournalBatchId>,
    archive: VolatileJournalArchive,
    event_view: Arc<ArcSwap<VolatileEventArchiveView>>,
    receipts: VecDeque<JournalReceipt>,
}

#[derive(Debug)]
struct VolatileJournalEntry {
    batch: Arc<JournalBatch>,
}

#[derive(Debug, Clone)]
struct VolatileScientificEntry {
    batch_id: JournalBatchId,
    sequence: EventSequence,
    applied: AppliedCommand,
    boundary: Arc<ScientificBoundary>,
    committed: bool,
}

#[derive(Debug)]
struct VolatileJournalArchive {
    capacity: usize,
    entries: VecDeque<VolatileJournalEntry>,
    scientific_entries: VecDeque<VolatileScientificEntry>,
}

#[derive(Debug, Default)]
struct VolatileEventArchiveView {
    entries: Vec<VolatileScientificEntry>,
    available_range: Option<EventSequenceRange>,
}

impl Default for VolatileJournal {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_VOLATILE_EVENT_HISTORY_CAPACITY)
    }
}

impl VolatileJournal {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            highest_accepted: None,
            archive: VolatileJournalArchive {
                capacity,
                entries: VecDeque::with_capacity(capacity),
                scientific_entries: VecDeque::with_capacity(capacity),
            },
            event_view: Arc::new(ArcSwap::from_pointee(VolatileEventArchiveView::default())),
            receipts: VecDeque::new(),
        }
    }

    /// Exact admitted batches whose volatile commitment receipt is still pending.
    ///
    /// Committed scientific history is retained separately as a lightweight immutable
    /// boundary record so catch-up readers do not pin complete persistence payloads.
    #[must_use]
    pub fn batches(&self) -> Vec<Arc<JournalBatch>> {
        self.archive
            .entries
            .iter()
            .map(|entry| Arc::clone(&entry.batch))
            .collect()
    }
}

#[derive(Debug)]
struct VolatileEventReader {
    session_id: HostSessionId,
    view: Arc<ArcSwap<VolatileEventArchiveView>>,
}

fn volatile_available_range(
    archive: &VolatileJournalArchive,
    session_id: HostSessionId,
) -> Option<EventSequenceRange> {
    let mut sequences = archive.scientific_entries.iter().filter_map(|entry| {
        (entry.committed && entry.batch_id.session_id() == session_id).then_some(entry.sequence)
    });
    let first = sequences.next()?;
    let last = sequences.next_back().unwrap_or(first);
    Some(EventSequenceRange { first, last })
}

fn volatile_event_view(
    archive: &VolatileJournalArchive,
    session_id: HostSessionId,
) -> VolatileEventArchiveView {
    VolatileEventArchiveView {
        entries: archive
            .scientific_entries
            .iter()
            .filter(|entry| entry.committed && entry.batch_id.session_id() == session_id)
            .cloned()
            .collect(),
        available_range: volatile_available_range(archive, session_id),
    }
}

impl EventJournalReader for VolatileEventReader {
    fn session_id(&self) -> HostSessionId {
        self.session_id
    }

    fn guarantee(&self) -> EventCatchUpGuarantee {
        EventCatchUpGuarantee::LiveMemory
    }

    fn available_range(&self) -> Option<EventSequenceRange> {
        self.view.load().available_range
    }

    fn retention_snapshot(&self) -> Option<EventRetentionSnapshot> {
        let view = self.view.load_full();
        let range = view.available_range?;
        EventRetentionSnapshot::try_new(
            self.session_id,
            EventCatchUpGuarantee::LiveMemory,
            range,
            view,
        )
        .ok()
    }

    fn contains_event_identity(&self, sequence: EventSequence, batch_id: JournalBatchId) -> bool {
        self.view.load().entries.iter().any(|entry| {
            entry.sequence == sequence
                && entry.batch_id == batch_id
                && entry.batch_id.session_id() == self.session_id
        })
    }

    fn read(
        &self,
        locator: EventCatchUpLocator,
        limit: usize,
    ) -> Result<EventCatchUp, HostAccessError> {
        if locator.session_id() != self.session_id {
            return Ok(EventCatchUp::Unavailable {
                range: locator.range(),
                reason: EventCatchUpUnavailableReason::SessionMismatch,
            });
        }
        let view = self.view.load();
        let Some(available) = view.available_range else {
            return Ok(EventCatchUp::Unavailable {
                range: locator.range(),
                reason: EventCatchUpUnavailableReason::RangeExpired,
            });
        };
        if !available.contains_range(locator.range()) {
            let reason = if available.contains(locator.range().last) {
                EventCatchUpUnavailableReason::PartialRange
            } else {
                EventCatchUpUnavailableReason::RangeExpired
            };
            return Ok(EventCatchUp::Unavailable {
                range: locator.range(),
                reason,
            });
        }
        let events = view
            .entries
            .iter()
            .filter(|entry| locator.range().contains(entry.sequence))
            .map(|entry| JournaledScientificEvent {
                event: Arc::new(ScientificEvent {
                    session_id: self.session_id,
                    sequence: entry.sequence,
                    batch_id: entry.batch_id,
                    tick: entry.applied.tick,
                    revisions: entry.applied.revisions,
                    boundary: Arc::clone(&entry.boundary),
                }),
                commitment: EventCommitment::CommittedVolatile,
            })
            .take(limit)
            .collect();
        Ok(EventCatchUp::Contiguous(EventPage {
            session_id: self.session_id,
            source: EventPageSource::LiveMemory,
            events,
            latest: available.last,
        }))
    }
}

impl JournalPort for VolatileJournal {
    fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
        let batch_id = batch.id();
        if self.highest_accepted.is_some_and(|highest| {
            highest.session_id() == batch_id.session_id()
                && highest.sequence() >= batch_id.sequence()
        }) {
            return JournalAdmission::Accepted { batch_id };
        }
        if self
            .highest_accepted
            .is_some_and(|highest| highest.session_id() != batch_id.session_id())
        {
            return JournalAdmission::Closed { batch_id };
        }
        let scientific_entry = match batch.scientific_event_sequence() {
            None => None,
            Some(sequence) => {
                let Some(boundary) = batch.scientific() else {
                    return JournalAdmission::Closed { batch_id };
                };
                Some(VolatileScientificEntry {
                    batch_id,
                    sequence,
                    applied: batch.applied(),
                    boundary: Arc::clone(boundary),
                    committed: false,
                })
            }
        };
        let archive = &mut self.archive;
        let scientific = scientific_entry.is_some();
        let general_has_room = archive.entries.len() < archive.capacity;
        let scientific_has_room = !scientific
            || archive.scientific_entries.len() < archive.capacity
            || archive
                .scientific_entries
                .front()
                .is_some_and(|entry| entry.committed);
        if !general_has_room || !scientific_has_room {
            return JournalAdmission::Full {
                batch_id,
                capacity: archive.capacity,
            };
        }
        if scientific && archive.scientific_entries.len() >= archive.capacity {
            archive.scientific_entries.pop_front();
        }
        archive.entries.push_back(VolatileJournalEntry {
            batch: Arc::clone(batch),
        });
        if let Some(scientific_entry) = scientific_entry {
            archive.scientific_entries.push_back(scientific_entry);
        }
        if scientific {
            let event_view = volatile_event_view(archive, batch_id.session_id());
            self.event_view.store(Arc::new(event_view));
        }
        self.highest_accepted = Some(batch_id);
        self.receipts.push_back(JournalReceipt::new(
            batch_id,
            JournalReceiptState::CommittedVolatile,
        ));
        JournalAdmission::Accepted { batch_id }
    }

    fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
        let count = limit.min(self.receipts.len());
        let receipts: Vec<_> = self.receipts.drain(..count).collect();
        let archive = &mut self.archive;
        let mut scientific_changed = false;
        for receipt in &receipts {
            if let Some(index) = archive
                .entries
                .iter()
                .position(|entry| entry.batch.id() == receipt.batch_id())
            {
                archive.entries.remove(index);
            }
            if let Some(entry) = archive
                .scientific_entries
                .iter_mut()
                .find(|entry| entry.batch_id == receipt.batch_id())
            {
                entry.committed = true;
                scientific_changed = true;
            }
        }
        if scientific_changed && let Some(highest) = self.highest_accepted {
            self.event_view
                .store(Arc::new(volatile_event_view(archive, highest.session_id())));
        }
        receipts
    }

    fn event_reader(&self, session_id: HostSessionId) -> Option<Arc<dyn EventJournalReader>> {
        Some(Arc::new(VolatileEventReader {
            session_id,
            view: Arc::clone(&self.event_view),
        }))
    }

    fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
        ShutdownCommitRequirement::CommittedVolatile
    }
}

#[derive(Debug, Clone)]
struct AdmittedEnvelope {
    admission: AdmissionSequence,
    envelope: CommandEnvelope,
}

#[derive(Debug, Clone)]
struct CommandAuthority {
    envelope: Option<CommandEnvelope>,
    envelope_digest: [u8; blake3::OUT_LEN],
    status: CommandStatus,
    initial_boundary: AppliedCommand,
    application_boundary: AppliedCommand,
    pending_audit_order: Option<u64>,
}

impl CommandAuthority {
    fn lifecycle_evidence(&self) -> Result<CommandLifecycleEvidence, HostAccessError> {
        let envelope = self.envelope.clone().ok_or_else(|| {
            protocol_violation("command lifecycle envelope was already compacted")
        })?;
        CommandLifecycleEvidence::from_terminal(
            envelope,
            self.status.admission_sequence(),
            self.initial_boundary,
            self.application_boundary,
            self.status.application().clone(),
        )
        .map_err(|error| protocol_violation(format!("invalid command lifecycle evidence: {error}")))
    }
}

fn command_envelope_digest(
    envelope: &CommandEnvelope,
) -> Result<[u8; blake3::OUT_LEN], HostAccessError> {
    let bytes = postcard::to_allocvec(envelope).map_err(|error| {
        protocol_violation(format!("could not encode command identity: {error}"))
    })?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

struct SharedHostState {
    session_id: HostSessionId,
    command_capacity: usize,
    next_admission: AdmissionSequence,
    next_event: ProtocolEventSequence,
    protocol_event_capacity: usize,
    admission_lifecycle: HostLifecycle,
    audit_gate_closed: bool,
    shutdown_command_id: Option<CommandId>,
    queue: VecDeque<AdmittedEnvelope>,
    commands: HashMap<CommandId, CommandAuthority>,
    /// Bounded post-archival idempotency index: digest + terminal status for commands
    /// whose lifecycle is durably archived. This, not the unbounded live map, is the
    /// long-window dedup authority (bd-2z0.5.2.1); the durable journal outranks it.
    archived_idempotency: HashMap<CommandId, ArchivedIdempotency>,
    /// Admission-ordered ids of archived commands, for oldest-first eviction.
    archived_order: VecDeque<CommandId>,
    /// Retention bound for the archived index; a field so tests can shrink it without
    /// touching production defaults.
    archived_retention: usize,
    next_audit_order: u64,
    pending_audit_count: usize,
    pending_audits: VecDeque<(u64, CommandId)>,
    last_applied: Option<(AdmissionSequence, CommandId)>,
    events: VecDeque<HostEvent>,
    visible_boundary: AppliedCommand,
}

/// Retained proof that one terminal command already ran, kept after its full
/// `CommandAuthority` is released at durable archival.
#[derive(Debug, Clone)]
struct ArchivedIdempotency {
    envelope_digest: [u8; blake3::OUT_LEN],
    status: CommandStatus,
}

/// Maximum archived terminal commands retained for idempotent retry answers. Beyond
/// this bound the oldest archived records are evicted and the durable journal is the
/// only authority (bd-2z0.5.2.1).
const ARCHIVED_IDEMPOTENCY_RETENTION: usize = 4_096;

impl SharedHostState {
    fn emit(&mut self, kind: HostEventKind) -> Result<(), HostAccessError> {
        let sequence = self.next_event;
        self.next_event = sequence
            .checked_next()
            .ok_or_else(|| protocol_violation("event sequence exhausted"))?;
        if self.events.len() == self.protocol_event_capacity {
            self.events.pop_front();
        }
        self.events.push_back(HostEvent {
            session_id: self.session_id,
            sequence,
            tick: self.visible_boundary.tick,
            kind,
        });
        Ok(())
    }

    fn insert_status(
        &mut self,
        envelope: CommandEnvelope,
        envelope_digest: [u8; blake3::OUT_LEN],
        status: CommandStatus,
        pending_audit_order: Option<u64>,
    ) -> Result<(), HostAccessError> {
        let command_id = status.command_id();
        let boundary = match status.application() {
            ApplicationState::Applied(applied) => *applied,
            ApplicationState::Admitted
            | ApplicationState::Rejected(_)
            | ApplicationState::Failed(_) => self.visible_boundary,
        };
        let previous = self.commands.insert(
            command_id,
            CommandAuthority {
                envelope: Some(envelope),
                envelope_digest,
                status: status.clone(),
                initial_boundary: boundary,
                application_boundary: boundary,
                pending_audit_order,
            },
        );
        if previous.is_some() {
            return Err(protocol_violation("command authority was inserted twice"));
        }
        self.emit(HostEventKind::CommandStatusChanged(status))
    }

    fn store_status(&mut self, status: CommandStatus) -> Result<(), HostAccessError> {
        if matches!(status.application(), ApplicationState::Applied(_))
            && let Some(admission) = status.admission_sequence()
            && self
                .last_applied
                .is_none_or(|(current, _)| admission > current)
        {
            self.last_applied = Some((admission, status.command_id()));
        }
        let visible_boundary = self.visible_boundary;
        let authority = self
            .commands
            .get_mut(&status.command_id())
            .ok_or_else(|| protocol_violation("command status has no envelope authority"))?;
        if authority.status.application() != status.application() {
            authority.application_boundary = match status.application() {
                ApplicationState::Applied(applied) => *applied,
                ApplicationState::Admitted
                | ApplicationState::Rejected(_)
                | ApplicationState::Failed(_) => visible_boundary,
            };
        }
        authority.status = status.clone();
        self.emit(HostEventKind::CommandStatusChanged(status))
    }

    /// Move one terminal, journal-committed command out of the live authority map into
    /// the bounded archived-idempotency index, evicting the oldest archived records
    /// beyond the retention bound (bd-2z0.5.2.1).
    ///
    /// Eviction guards: the command must be terminal (Applied/Rejected/Failed), must
    /// not be the pending shutdown command, and must not hold an outstanding audit
    /// order. Those stay live; every other durably archived record eventually moves.
    fn archive_terminal_command(&mut self, command_id: CommandId) {
        let Some(authority) = self.commands.get(&command_id) else {
            return;
        };
        let terminal = matches!(
            authority.status.application(),
            ApplicationState::Applied(_)
                | ApplicationState::Rejected(_)
                | ApplicationState::Failed(_)
        );
        if !terminal
            || self.shutdown_command_id == Some(command_id)
            || authority.pending_audit_order.is_some()
        {
            return;
        }
        let archived = ArchivedIdempotency {
            envelope_digest: authority.envelope_digest,
            status: authority.status.clone(),
        };
        self.archived_idempotency.insert(command_id, archived);
        self.archived_order.push_back(command_id);
        self.commands.remove(&command_id);
        while self.archived_order.len() > self.archived_retention {
            if let Some(oldest) = self.archived_order.pop_front() {
                self.archived_idempotency.remove(&oldest);
            }
        }
    }

    fn reserve_pre_admission_audit(&mut self) -> Result<u64, HostAccessError> {
        if self.pending_audit_count >= self.command_capacity {
            return Err(HostAccessError::CommandEvidenceBackpressure {
                capacity: self.command_capacity,
            });
        }
        let order = self.next_audit_order;
        self.next_audit_order = order
            .checked_add(1)
            .ok_or_else(|| protocol_violation("command audit order exhausted"))?;
        self.pending_audit_count += 1;
        Ok(order)
    }

    fn insert_pre_admission_rejection(
        &mut self,
        envelope: CommandEnvelope,
        envelope_digest: [u8; blake3::OUT_LEN],
        reason: RejectionReason,
    ) -> Result<CommandStatus, HostAccessError> {
        let audit_order = self.reserve_pre_admission_audit()?;
        let status = CommandStatus::try_new(
            envelope.command_id,
            None,
            ApplicationState::Rejected(reason),
            JournalState::Pending,
        )
        .map_err(status_violation)?;
        let command_id = envelope.command_id;
        // Make the evidence reachable before the fallible protocol-event
        // notification. A saturated notification lane must not orphan a
        // terminal status whose audit slot has already been reserved.
        self.pending_audits.push_back((audit_order, command_id));
        self.insert_status(envelope, envelope_digest, status.clone(), Some(audit_order))?;
        Ok(status)
    }

    fn next_pending_audit(
        &self,
    ) -> Result<Option<(CommandId, u64, CommandLifecycleEvidence)>, HostAccessError> {
        let Some((order, command_id)) = self.pending_audits.front().copied() else {
            return Ok(None);
        };
        let authority = self
            .commands
            .get(&command_id)
            .ok_or_else(|| protocol_violation("pending command audit authority is missing"))?;
        if authority.pending_audit_order != Some(order) {
            return Err(protocol_violation("pending command audit lost its order"));
        }
        Ok(Some((command_id, order, authority.lifecycle_evidence()?)))
    }

    fn claim_pending_audit(
        &mut self,
        command_id: CommandId,
        expected_order: u64,
    ) -> Result<(), HostAccessError> {
        let pending_order = self
            .commands
            .get(&command_id)
            .ok_or_else(|| protocol_violation("pending command audit authority is missing"))?
            .pending_audit_order;
        if pending_order != Some(expected_order) {
            return Err(protocol_violation("pending command audit order changed"));
        }
        if self.pending_audits.pop_front() != Some((expected_order, command_id)) {
            return Err(protocol_violation("pending command audit queue changed"));
        }
        self.commands
            .get_mut(&command_id)
            .ok_or_else(|| protocol_violation("pending command audit authority disappeared"))?
            .pending_audit_order = None;
        Ok(())
    }

    fn release_pre_admission_audit(&mut self) -> Result<(), HostAccessError> {
        self.pending_audit_count = self
            .pending_audit_count
            .checked_sub(1)
            .ok_or_else(|| protocol_violation("pre-admission audit slot released twice"))?;
        Ok(())
    }

    fn compact_command_envelope(&mut self, command_id: CommandId) -> Result<(), HostAccessError> {
        self.commands
            .get_mut(&command_id)
            .ok_or_else(|| protocol_violation("command authority disappeared before compaction"))?
            .envelope = None;
        Ok(())
    }

    fn lifecycle_evidence(
        &self,
        command_id: CommandId,
    ) -> Result<CommandLifecycleEvidence, HostAccessError> {
        self.commands
            .get(&command_id)
            .ok_or_else(|| protocol_violation("command lifecycle authority is missing"))?
            .lifecycle_evidence()
    }

    fn submit(
        &mut self,
        envelope: CommandEnvelope,
        reserve_lifecycle_slot: bool,
    ) -> Result<CommandStatus, HostAccessError> {
        let envelope_digest = command_envelope_digest(&envelope)?;
        if let Some(authority) = self.commands.get(&envelope.command_id) {
            if authority.envelope_digest == envelope_digest {
                return Ok(authority.status.clone());
            }
            return Err(HostAccessError::CommandIdCollision {
                command_id: envelope.command_id,
            });
        }
        if let Some(archived) = self.archived_idempotency.get(&envelope.command_id) {
            // The command is durably archived: an exact retry replays the archived
            // terminal status, a changed payload collides (bd-2z0.5.2.1).
            if archived.envelope_digest == envelope_digest {
                return Ok(archived.status.clone());
            }
            return Err(HostAccessError::CommandIdCollision {
                command_id: envelope.command_id,
            });
        }

        if self.audit_gate_closed || self.admission_lifecycle == HostLifecycle::Stopped {
            return Err(HostAccessError::CommandEvidenceClosed {
                lifecycle: self.admission_lifecycle,
            });
        }

        if let Err(error) = envelope.command.validate() {
            return self.insert_pre_admission_rejection(
                envelope,
                envelope_digest,
                RejectionReason::Validation {
                    message: error.to_string(),
                },
            );
        }
        if self.admission_lifecycle != HostLifecycle::Running {
            return self.insert_pre_admission_rejection(
                envelope,
                envelope_digest,
                RejectionReason::HostStopping,
            );
        }

        let closes_gate = matches!(&envelope.command, HostCommand::Shutdown);
        if self.queue.len() >= self.command_capacity && !(reserve_lifecycle_slot && closes_gate) {
            return self.insert_pre_admission_rejection(
                envelope,
                envelope_digest,
                RejectionReason::Overloaded {
                    capacity: self.command_capacity,
                },
            );
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
        .map_err(status_violation)?;
        if closes_gate {
            self.admission_lifecycle = HostLifecycle::Stopping;
            self.shutdown_command_id = Some(envelope.command_id);
        }
        self.queue.push_back(AdmittedEnvelope {
            admission,
            envelope: envelope.clone(),
        });
        self.insert_status(envelope, envelope_digest, status.clone(), None)?;
        Ok(status)
    }
}

/// Cloneable same-thread command and observation handle for [`HostCore`].
///
/// The handle shares only bounded ingress and immutable/query DTO state. It
/// never owns, borrows, locks, or exposes the mutable [`WorldState`].
#[derive(Clone)]
pub struct LocalHostPort {
    shared: Rc<RefCell<SharedHostState>>,
    snapshots: SnapshotHub,
    events: EventHub,
}

impl LocalHostPort {
    /// Number of admitted envelopes not yet processed by the owner.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.shared.borrow().queue.len()
    }
}

impl HostPort for LocalHostPort {
    fn session_id(&self) -> HostSessionId {
        self.shared.borrow().session_id
    }

    fn submit(&mut self, envelope: CommandEnvelope) -> Result<CommandStatus, HostAccessError> {
        self.shared.borrow_mut().submit(envelope, false)
    }

    fn command_status(
        &mut self,
        command_id: CommandId,
    ) -> Result<Option<CommandStatus>, HostAccessError> {
        let shared = self.shared.borrow();
        Ok(shared
            .commands
            .get(&command_id)
            .map(|authority| authority.status.clone())
            .or_else(|| {
                shared
                    .archived_idempotency
                    .get(&command_id)
                    .map(|archived| archived.status.clone())
            }))
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
        Ok(self
            .shared
            .borrow()
            .events
            .iter()
            .filter(|event| event.sequence > cursor)
            .take(limit)
            .cloned()
            .collect())
    }

    fn poll_events(
        &mut self,
        cursor: super::EventCursor,
        limit: usize,
    ) -> Result<super::EventPoll, HostAccessError> {
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

#[derive(Debug, Clone, Copy)]
struct InflightJournal {
    command_id: Option<CommandId>,
    scientific_event: Option<EventSequence>,
    shutdown_requirement: Option<ShutdownCommitRequirement>,
    committed_volatile: bool,
}

#[derive(Clone)]
struct SnapshotLayerCache {
    revisions: SnapshotLayerRevisions,
    terrain: Arc<TerrainLayerSnapshot>,
    food: Arc<FoodLayerSnapshot>,
    hydrology: Option<Arc<HydrologyLayerSnapshot>>,
}

#[derive(Debug, Clone, Copy, Default)]
struct LayerRefreshStats {
    bulk_allocations: usize,
    newly_allocated_capacity_bytes: usize,
}

impl LayerRefreshStats {
    const fn add_vector<T>(&mut self, capacity: usize) {
        if capacity != 0 {
            self.bulk_allocations = self.bulk_allocations.saturating_add(1);
            self.newly_allocated_capacity_bytes = self
                .newly_allocated_capacity_bytes
                .saturating_add(capacity.saturating_mul(size_of::<T>()));
        }
    }
}

impl SnapshotLayerCache {
    fn new(world: &WorldState) -> (Self, LayerRefreshStats) {
        let terrain = Arc::new(capture_terrain_layer(world));
        let food = Arc::new(capture_food_layer(world));
        let hydrology = world.hydrology().map(capture_hydrology_layer).map(Arc::new);
        let cache = Self {
            revisions: SnapshotLayerRevisions {
                terrain: LayerRevision::new(1),
                food: LayerRevision::new(1),
                hydrology: LayerRevision::new(u64::from(hydrology.is_some())),
            },
            terrain,
            food,
            hydrology,
        };
        let mut stats = LayerRefreshStats::default();
        cache.add_allocation_stats(&mut stats);
        (cache, stats)
    }

    fn refresh(&mut self, world: &WorldState) -> Result<LayerRefreshStats, HostAccessError> {
        let mut stats = LayerRefreshStats::default();
        if !terrain_layer_matches(&self.terrain, world) {
            let terrain = Arc::new(capture_terrain_layer(world));
            stats.add_vector::<TerrainTileSnapshot>(terrain.tiles.capacity());
            self.terrain = terrain;
            self.revisions.terrain = next_layer_revision(self.revisions.terrain, "terrain")?;
        }
        if !food_layer_matches(&self.food, world) {
            let food = Arc::new(capture_food_layer(world));
            stats.add_vector::<f32>(food.cells.capacity());
            self.food = food;
            self.revisions.food = next_layer_revision(self.revisions.food, "food")?;
        }
        match (&self.hydrology, world.hydrology()) {
            (None, None) => {}
            (Some(current), Some(world_hydrology))
                if hydrology_layer_matches(current, world_hydrology) => {}
            (_, next) => {
                self.hydrology = next.map(capture_hydrology_layer).map(Arc::new);
                if let Some(hydrology) = &self.hydrology {
                    add_hydrology_allocation_stats(hydrology, &mut stats);
                }
                self.revisions.hydrology =
                    next_layer_revision(self.revisions.hydrology, "hydrology")?;
            }
        }
        Ok(stats)
    }

    fn snapshot(&self) -> SnapshotLayers {
        SnapshotLayers {
            revisions: self.revisions,
            terrain: Arc::clone(&self.terrain),
            food: Arc::clone(&self.food),
            hydrology: self.hydrology.as_ref().map(Arc::clone),
        }
    }

    fn total_capacity_bytes(&self) -> usize {
        let terrain = self
            .terrain
            .tiles
            .capacity()
            .saturating_mul(size_of::<TerrainTileSnapshot>());
        let food = self.food.cells.capacity().saturating_mul(size_of::<f32>());
        let hydrology = self
            .hydrology
            .as_deref()
            .map_or(0, hydrology_capacity_bytes);
        terrain.saturating_add(food).saturating_add(hydrology)
    }

    fn add_allocation_stats(&self, stats: &mut LayerRefreshStats) {
        stats.add_vector::<TerrainTileSnapshot>(self.terrain.tiles.capacity());
        stats.add_vector::<f32>(self.food.cells.capacity());
        if let Some(hydrology) = &self.hydrology {
            add_hydrology_allocation_stats(hydrology, stats);
        }
    }
}

fn next_layer_revision(
    revision: LayerRevision,
    layer: &'static str,
) -> Result<LayerRevision, HostAccessError> {
    revision
        .checked_next()
        .ok_or_else(|| protocol_violation(format!("{layer} layer revision exhausted")))
}

fn capture_terrain_layer(world: &WorldState) -> TerrainLayerSnapshot {
    let terrain = world.terrain();
    let mut tiles = Vec::with_capacity(terrain.tiles().len());
    tiles.extend(terrain.tiles().iter().map(|tile| TerrainTileSnapshot {
        kind: tile.kind,
        elevation: tile.elevation,
        moisture: tile.moisture,
        accent: tile.accent,
        fertility_bias: tile.fertility_bias,
        temperature_bias: tile.temperature_bias,
        palette_index: tile.palette_index,
    }));
    TerrainLayerSnapshot {
        width: terrain.width(),
        height: terrain.height(),
        cell_size: terrain.cell_size(),
        tiles,
    }
}

fn capture_food_layer(world: &WorldState) -> FoodLayerSnapshot {
    let food = world.food();
    FoodLayerSnapshot {
        width: food.width(),
        height: food.height(),
        cells: food.cells().to_vec(),
    }
}

fn capture_hydrology_layer(hydrology: &scriptbots_core::HydrologyState) -> HydrologyLayerSnapshot {
    let tiles = hydrology.tiles();
    let field = hydrology.field();
    HydrologyLayerSnapshot {
        width: hydrology.width(),
        height: hydrology.height(),
        tiles: tiles
            .tiles()
            .iter()
            .map(|tile| HydrologyTileSnapshot {
                permeability: tile.permeability,
                runoff_bias: tile.runoff_bias,
                basin_rank: tile.basin_rank,
                channel_priority: tile.channel_priority,
                swim_cost: tile.swim_cost,
            })
            .collect(),
        flow_directions: field.flow_directions().to_vec(),
        accumulation: field.accumulation().to_vec(),
        spill_elevation: field.spill_elevation().to_vec(),
        basin_ids: field.basin_ids().to_vec(),
        water_depth: hydrology.water_depth().to_vec(),
    }
}

fn terrain_layer_matches(snapshot: &TerrainLayerSnapshot, world: &WorldState) -> bool {
    let terrain = world.terrain();
    snapshot.width == terrain.width()
        && snapshot.height == terrain.height()
        && snapshot.cell_size == terrain.cell_size()
        && snapshot.tiles.len() == terrain.tiles().len()
        && snapshot
            .tiles
            .iter()
            .zip(terrain.tiles())
            .all(|(snapshot, tile)| {
                snapshot.kind == tile.kind
                    && same_f32(snapshot.elevation, tile.elevation)
                    && same_f32(snapshot.moisture, tile.moisture)
                    && same_f32(snapshot.accent, tile.accent)
                    && same_f32(snapshot.fertility_bias, tile.fertility_bias)
                    && same_f32(snapshot.temperature_bias, tile.temperature_bias)
                    && snapshot.palette_index == tile.palette_index
            })
}

fn food_layer_matches(snapshot: &FoodLayerSnapshot, world: &WorldState) -> bool {
    let food = world.food();
    snapshot.width == food.width()
        && snapshot.height == food.height()
        && same_f32_slice(&snapshot.cells, food.cells())
}

fn hydrology_layer_matches(
    snapshot: &HydrologyLayerSnapshot,
    hydrology: &scriptbots_core::HydrologyState,
) -> bool {
    let tiles = hydrology.tiles();
    let field = hydrology.field();
    snapshot.width == hydrology.width()
        && snapshot.height == hydrology.height()
        && snapshot.tiles.len() == tiles.tiles().len()
        && snapshot
            .tiles
            .iter()
            .zip(tiles.tiles())
            .all(|(snapshot, tile)| {
                same_f32(snapshot.permeability, tile.permeability)
                    && same_f32(snapshot.runoff_bias, tile.runoff_bias)
                    && same_f32(snapshot.basin_rank, tile.basin_rank)
                    && same_f32(snapshot.channel_priority, tile.channel_priority)
                    && same_f32(snapshot.swim_cost, tile.swim_cost)
            })
        && snapshot.flow_directions == field.flow_directions()
        && same_f32_slice(&snapshot.accumulation, field.accumulation())
        && same_f32_slice(&snapshot.spill_elevation, field.spill_elevation())
        && snapshot.basin_ids == field.basin_ids()
        && same_f32_slice(&snapshot.water_depth, hydrology.water_depth())
}

const fn same_f32(left: f32, right: f32) -> bool {
    left.to_bits() == right.to_bits()
}

fn same_f32_slice(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| same_f32(*left, *right))
}

const fn hydrology_capacity_bytes(snapshot: &HydrologyLayerSnapshot) -> usize {
    snapshot
        .tiles
        .capacity()
        .saturating_mul(size_of::<HydrologyTileSnapshot>())
        .saturating_add(
            snapshot
                .flow_directions
                .capacity()
                .saturating_mul(size_of::<scriptbots_core::HydrologyFlowDirection>()),
        )
        .saturating_add(
            snapshot
                .accumulation
                .capacity()
                .saturating_mul(size_of::<f32>()),
        )
        .saturating_add(
            snapshot
                .spill_elevation
                .capacity()
                .saturating_mul(size_of::<f32>()),
        )
        .saturating_add(
            snapshot
                .basin_ids
                .capacity()
                .saturating_mul(size_of::<u32>()),
        )
        .saturating_add(
            snapshot
                .water_depth
                .capacity()
                .saturating_mul(size_of::<f32>()),
        )
}

const fn add_hydrology_allocation_stats(
    snapshot: &HydrologyLayerSnapshot,
    stats: &mut LayerRefreshStats,
) {
    stats.add_vector::<HydrologyTileSnapshot>(snapshot.tiles.capacity());
    stats
        .add_vector::<scriptbots_core::HydrologyFlowDirection>(snapshot.flow_directions.capacity());
    stats.add_vector::<f32>(snapshot.accumulation.capacity());
    stats.add_vector::<f32>(snapshot.spill_elevation.capacity());
    stats.add_vector::<u32>(snapshot.basin_ids.capacity());
    stats.add_vector::<f32>(snapshot.water_depth.capacity());
}

fn snapshot_build_stats(
    world: &DynamicWorldSnapshot,
    summary_history: &[TickSummary],
    summary_history_capacity: usize,
    summary_history_allocated: bool,
    layers: &SnapshotLayerCache,
    refresh: LayerRefreshStats,
) -> SnapshotBuildStats {
    let dynamic_agent_bytes = world
        .agents
        .capacity()
        .saturating_mul(size_of::<DynamicAgentSnapshot>());
    let layer_bytes = layers.total_capacity_bytes();
    let summary_history_bytes = summary_history_capacity.saturating_mul(size_of::<TickSummary>());
    SnapshotBuildStats {
        dynamic_agent_count: world.agents.len(),
        summary_history_count: summary_history.len(),
        bulk_allocations: refresh
            .bulk_allocations
            .saturating_add(usize::from(world.agents.capacity() != 0))
            .saturating_add(usize::from(
                summary_history_allocated && !summary_history.is_empty(),
            )),
        newly_allocated_capacity_bytes: refresh
            .newly_allocated_capacity_bytes
            .saturating_add(dynamic_agent_bytes)
            .saturating_add(if summary_history_allocated {
                summary_history_bytes
            } else {
                0
            }),
        reused_layer_capacity_bytes: layer_bytes
            .saturating_sub(refresh.newly_allocated_capacity_bytes),
        total_payload_capacity_bytes: dynamic_agent_bytes
            .saturating_add(summary_history_bytes)
            .saturating_add(layer_bytes),
    }
}

/// Pure synchronous authority for command order and scientific time.
///
/// `HostCore` owns its world and persistence-admission session by value. It
/// spawns no threads and contains no platform runtime, renderer, server,
/// database connection, or mutable-world lock.
pub struct HostCore {
    session_id: HostSessionId,
    world: WorldState,
    persistence: PersistenceAdmissionSession,
    journal: Box<dyn JournalPort>,
    shared: Rc<RefCell<SharedHostState>>,
    snapshots: SnapshotHub,
    events: EventHub,
    summary_history: Arc<Vec<TickSummary>>,
    snapshot_layers: SnapshotLayerCache,
    options: HostCoreOptions,
    playback: PlaybackSnapshot,
    lifecycle: HostLifecycle,
    health: HostHealth,
    revisions: HostRevisions,
    last_now: Option<ManualInstant>,
    cadence_credit: u128,
    next_snapshot: SnapshotRevision,
    last_published_scientific: ScientificRevision,
    latest_completed_summary: Option<TickSummary>,
    next_journal_sequence: u64,
    next_lifecycle_command_sequence: u64,
    active_command: Option<AdmittedEnvelope>,
    active_journal_batch: Option<Arc<JournalBatch>>,
    indeterminate_journal_batch: Option<Arc<JournalBatch>>,
    retained_journal: Option<Arc<JournalBatch>>,
    retained_blocker: Option<HostBlocker>,
    event_pressure: Option<HostBlocker>,
    inflight_journal: HashMap<JournalBatchId, InflightJournal>,
    shutdown_receipt: Option<(JournalBatchId, ShutdownCommitRequirement)>,
    failed_journal_batches: HashSet<JournalBatchId>,
    latched_fault: Option<HostFault>,
}

impl HostCore {
    /// Construct a host with a nonblocking in-memory volatile journal.
    pub fn new(
        session_id: HostSessionId,
        world: WorldState,
        options: HostCoreOptions,
    ) -> Result<Self, HostCoreBuildError> {
        if options.volatile_event_history_capacity == 0
            || options.volatile_event_history_capacity <= options.scientific_event_capacity
        {
            return Err(HostCoreBuildError::InvalidOptions {
                message: "volatile_event_history_capacity must exceed scientific_event_capacity"
                    .to_owned(),
            });
        }
        Self::with_journal(
            session_id,
            world,
            options,
            Box::new(VolatileJournal::with_capacity(
                options.volatile_event_history_capacity,
            )),
        )
    }

    /// Construct a host with an injected runtime-neutral journal adapter.
    #[allow(clippy::too_many_lines)]
    pub fn with_journal(
        session_id: HostSessionId,
        world: WorldState,
        options: HostCoreOptions,
        journal: Box<dyn JournalPort>,
    ) -> Result<Self, HostCoreBuildError> {
        validate_options(options)?;
        let events = EventHub::new(
            session_id,
            options.scientific_event_capacity,
            journal.event_reader(session_id),
        )
        .map_err(|error| HostCoreBuildError::InvalidOptions {
            message: error.to_string(),
        })?;
        let persistence = world.bind_persistence(Box::new(NullPersistence))?;
        let revisions = HostRevisions {
            control: ControlRevision::new(0),
            scientific: ScientificRevision::new(world.tick().0),
            config: ConfigRevision::new(world.config_revision()),
        };
        let playback = options.initial_playback;
        let lifecycle = HostLifecycle::Running;
        let health = HostHealth::Healthy;
        let (snapshot_layers, layer_refresh) = SnapshotLayerCache::new(&world);
        let dynamic_world = DynamicWorldSnapshot::from_world(&world);
        let summary_history = Arc::new(world.history().cloned().collect::<Vec<_>>());
        let build = snapshot_build_stats(
            &dynamic_world,
            &summary_history,
            summary_history.capacity(),
            true,
            &snapshot_layers,
            layer_refresh,
        );
        let latest_completed_summary = world
            .history()
            .next_back()
            .filter(|summary| summary.tick == world.tick())
            .cloned();
        let initial_snapshot = Arc::new(RenderSnapshot {
            session_id,
            revision: SnapshotRevision::new(1),
            revisions,
            playback,
            lifecycle,
            health: health.clone(),
            command_queue_depth: 0,
            last_applied_command: None,
            completed_summary: latest_completed_summary.clone(),
            summary_history: Arc::clone(&summary_history),
            layers: snapshot_layers.snapshot(),
            build,
            world: dynamic_world,
        });
        let snapshots = SnapshotHub::new(initial_snapshot);
        let shared = Rc::new(RefCell::new(SharedHostState {
            session_id,
            command_capacity: options.command_capacity,
            next_admission: AdmissionSequence::new(1),
            next_event: ProtocolEventSequence::new(1),
            protocol_event_capacity: options.protocol_event_capacity,
            admission_lifecycle: HostLifecycle::Running,
            audit_gate_closed: false,
            shutdown_command_id: None,
            queue: VecDeque::with_capacity(options.command_capacity),
            commands: HashMap::new(),
            archived_idempotency: HashMap::new(),
            archived_order: VecDeque::new(),
            archived_retention: ARCHIVED_IDEMPOTENCY_RETENTION,
            next_audit_order: 1,
            pending_audit_count: 0,
            pending_audits: VecDeque::with_capacity(options.command_capacity),
            last_applied: None,
            events: VecDeque::with_capacity(options.protocol_event_capacity),
            visible_boundary: AppliedCommand {
                tick: world.tick(),
                revisions,
            },
        }));
        Ok(Self {
            session_id,
            world,
            persistence,
            journal,
            shared,
            snapshots,
            events,
            summary_history: Arc::clone(&summary_history),
            snapshot_layers,
            options,
            playback,
            lifecycle,
            health,
            revisions,
            last_now: None,
            cadence_credit: 0,
            next_snapshot: SnapshotRevision::new(2),
            last_published_scientific: revisions.scientific,
            latest_completed_summary,
            next_journal_sequence: 1,
            next_lifecycle_command_sequence: session_id.get(),
            active_command: None,
            active_journal_batch: None,
            indeterminate_journal_batch: None,
            retained_journal: None,
            retained_blocker: None,
            event_pressure: None,
            inflight_journal: HashMap::new(),
            shutdown_receipt: None,
            failed_journal_batches: HashSet::new(),
            latched_fault: None,
        })
    }

    /// Create another same-thread handle to the bounded host port.
    #[must_use]
    pub fn local_port(&self) -> LocalHostPort {
        LocalHostPort {
            shared: Rc::clone(&self.shared),
            snapshots: self.snapshots.clone(),
            events: self.events.clone(),
        }
    }

    /// Clone the thread-safe latest-value snapshot read handle.
    ///
    /// The returned hub owns no command sender and cannot keep native ingress alive.
    #[must_use]
    pub fn snapshot_hub(&self) -> SnapshotHub {
        self.snapshots.clone()
    }

    /// Clone the detached bounded scientific-event reader.
    ///
    /// The returned handle owns no command ingress and cannot keep a native controller alive.
    #[must_use]
    pub fn event_hub(&self) -> EventHub {
        self.events.clone()
    }

    /// Admit or reuse the host-owned ordered shutdown command.
    ///
    /// Native cancellation uses one reserved lifecycle slot after every
    /// command already admitted through [`HostPort`]. The method closes normal
    /// ingress immediately, never bypasses that existing total order, and
    /// returns the same command status on every later call.
    pub fn request_shutdown(&mut self) -> Result<CommandStatus, HostAccessError> {
        if let Some(command_id) = self.shared.borrow().shutdown_command_id {
            return self
                .shared
                .borrow()
                .commands
                .get(&command_id)
                .map(|authority| authority.status.clone())
                .ok_or_else(|| protocol_violation("shutdown command status is missing"));
        }

        let command_id = loop {
            let sequence = self.next_lifecycle_command_sequence;
            self.next_lifecycle_command_sequence = sequence
                .checked_add(1)
                .ok_or_else(|| protocol_violation("lifecycle command sequence exhausted"))?;
            let candidate = CommandId::from_client_sequence(LIFECYCLE_COMMAND_NAMESPACE, sequence);
            if !self.shared.borrow().commands.contains_key(&candidate) {
                break candidate;
            }
        };
        self.shared.borrow_mut().submit(
            CommandEnvelope::new(command_id, HostCommand::Shutdown),
            true,
        )
    }

    /// Stable identity of the admitted ordered shutdown, when one exists.
    #[must_use]
    pub fn shutdown_command_id(&self) -> Option<CommandId> {
        self.shared.borrow().shutdown_command_id
    }

    /// Current two-axis status of the admitted ordered shutdown, when present.
    #[must_use]
    pub fn shutdown_command_status(&self) -> Option<CommandStatus> {
        let shared = self.shared.borrow();
        shared
            .shutdown_command_id
            .and_then(|command_id| shared.commands.get(&command_id))
            .map(|authority| authority.status.clone())
    }

    /// Latest immutable host publication.
    #[must_use]
    pub fn latest_snapshot(&self) -> Arc<RenderSnapshot> {
        self.snapshots.latest()
    }

    /// Pull bounded selected-brain detail for one client without mutating host or science state.
    pub fn inspect_brains(
        &self,
        request: &BrainProjectionRequest,
    ) -> Result<BrainProjection, BrainInspectionError> {
        if request.targets.len() > ACTIVATION_CAPTURE_BUDGET {
            return Err(BrainInspectionError::TargetLimitExceeded {
                requested: request.targets.len(),
                limit: ACTIVATION_CAPTURE_BUDGET,
            });
        }
        let latest = self.latest_snapshot();
        let inspection = self.world.inspect_brains(&BrainInspectionRequest {
            client_id: BrainInspectionClientId::new(request.client_id.get()),
            revision: BrainInspectionRevision::new(request.revision.get()),
            targets: request.targets.clone(),
            limits: request.limits,
        })?;
        Ok(BrainProjection {
            source: BrainProjectionSource {
                session_id: self.session_id,
                published_snapshot: latest.revision,
                published_host: latest.revisions,
                inspected_host: self.revisions,
                inspected_tick: self.world.tick(),
            },
            request: request.clone(),
            inspection,
        })
    }

    /// Current queryable health.
    #[must_use]
    pub const fn health(&self) -> &HostHealth {
        &self.health
    }

    /// Current completed scientific tick.
    #[must_use]
    pub const fn world_tick(&self) -> Tick {
        self.world.tick()
    }

    /// Canonical full scientific digest without exposing the mutable world.
    pub fn scientific_digest_v1(&self) -> Result<WorldDigestV1, CharacterizationError> {
        self.world.world_digest_v1()
    }

    /// Nominal automatic cadence configured for this host.
    #[must_use]
    pub const fn tick_period_nanos(&self) -> u64 {
        self.options.tick_period_nanos
    }

    /// Maximum automatic science work one drive boundary may perform.
    #[must_use]
    pub const fn max_automatic_steps_per_drive(&self) -> usize {
        self.options.max_automatic_steps_per_drive
    }

    /// Current scheduling interest for a platform-owned driver.
    #[must_use]
    pub fn drive_interest(&self) -> HostDriveInterest {
        if self.lifecycle == HostLifecycle::Stopped {
            return HostDriveInterest::Terminated;
        }
        if self.health.fault().is_some() {
            return HostDriveInterest::Faulted;
        }
        if self.retained_journal.is_some() {
            return HostDriveInterest::WakeOnly;
        }
        if self.event_pressure.is_some() {
            return HostDriveInterest::Draining;
        }
        let shared = self.shared.borrow();
        if shared.pending_audit_count != 0 || !shared.queue.is_empty() {
            return HostDriveInterest::ReadyNow;
        }
        drop(shared);
        let journal_drain_required = self.inflight_journal.values().any(|entry| {
            !entry.committed_volatile
                || matches!(
                    entry.shutdown_requirement,
                    Some(ShutdownCommitRequirement::Durable)
                )
        });
        if self.lifecycle == HostLifecycle::Stopping
            || (journal_drain_required
                && (self.playback.paused || speed_units(self.playback.speed_multiplier) == 0))
        {
            return HostDriveInterest::Draining;
        }
        if self.playback.paused || speed_units(self.playback.speed_multiplier) == 0 {
            HostDriveInterest::WakeOnly
        } else {
            HostDriveInterest::Deadline
        }
    }

    /// Exact immutable batch retained after non-admission.
    #[must_use]
    pub fn pending_journal_batch(&self) -> Option<Arc<JournalBatch>> {
        self.retained_journal.as_ref().map(Arc::clone)
    }

    /// Exact command whose owner boundary did not return normally.
    ///
    /// This is diagnostic evidence only. A returned envelope has already
    /// entered the host's admission order and must never be resubmitted.
    #[must_use]
    pub fn panicked_command(&self) -> Option<&CommandEnvelope> {
        self.active_command
            .as_ref()
            .map(|admitted| &admitted.envelope)
    }

    /// Exact journal batch whose adapter handoff panicked after application.
    ///
    /// The adapter may or may not have accepted this batch before unwinding,
    /// so this evidence is deliberately separate from retryable
    /// [`Self::pending_journal_batch`] state.
    #[must_use]
    pub fn indeterminate_journal_batch(&self) -> Option<Arc<JournalBatch>> {
        self.indeterminate_journal_batch.as_ref().map(Arc::clone)
    }

    /// Seal admissions and publish exact fault evidence after a driver unwind.
    #[cfg_attr(
        not(all(feature = "native-asupersync", not(target_arch = "wasm32"))),
        allow(
            dead_code,
            reason = "the default/browser host retains panic evidence for its native adapter"
        )
    )]
    pub(crate) fn record_panicked_boundary(
        &mut self,
        message: &str,
    ) -> Result<(), HostAccessError> {
        self.events.cancel_publish_reservation();
        if self.indeterminate_journal_batch.is_none()
            && let Some(batch) = self.active_journal_batch.take()
        {
            if self
                .retained_journal
                .as_ref()
                .is_some_and(|retained| retained.id() == batch.id())
            {
                self.retained_journal = None;
                self.retained_blocker = None;
            }
            self.indeterminate_journal_batch = Some(batch);
        }
        if self.lifecycle == HostLifecycle::Running {
            self.lifecycle = HostLifecycle::Stopping;
            let mut shared = self.shared.borrow_mut();
            shared.admission_lifecycle = HostLifecycle::Stopping;
            shared.emit(HostEventKind::LifecycleChanged(HostLifecycle::Stopping))?;
        }
        self.latched_fault = Some(HostFault::Protocol {
            code: "native_lifecycle_panic".to_owned(),
            message: message.to_owned(),
        });
        let changed = self.synchronize_health()?;
        if changed {
            self.publish_snapshot()?;
        }
        Ok(())
    }

    /// Explicitly retry the exact retained journal allocation once.
    ///
    /// A successful retry seals the corresponding core persistence boundary,
    /// but its command's `JournalState` still advances only through a later
    /// receipt polled by [`ManualHostDriver::drive`].
    pub fn retry_retained_journal(&mut self) -> Result<Option<JournalAdmission>, HostAccessError> {
        let Some(batch) = self.retained_journal.as_ref().map(Arc::clone) else {
            return Ok(None);
        };
        self.active_journal_batch = Some(Arc::clone(&batch));
        let admission = self.journal.try_admit(&batch);
        let result = self.finish_journal_admission(&batch, admission, true);
        if result.is_ok() {
            self.active_journal_batch = None;
        }
        result?;
        let changed = self.synchronize_health()?;
        if changed {
            self.publish_snapshot()?;
        }
        Ok(Some(admission))
    }

    fn retain_identity_violation(
        &mut self,
        batch: &Arc<JournalBatch>,
        admission: JournalAdmission,
    ) -> Result<bool, HostAccessError> {
        if admission.batch_id() == batch.id() {
            return Ok(false);
        }
        self.retained_journal = Some(Arc::clone(batch));
        self.retained_blocker = None;
        if let Some(command_id) = batch.command_id() {
            self.shared
                .borrow_mut()
                .compact_command_envelope(command_id)?;
        }
        let failure = JournalFailure {
            code: "journal_identity_mismatch".to_owned(),
            message: "journal response echoed a different batch identity".to_owned(),
        };
        if batch.requires_runtime_journal()
            && let Some(command_id) = batch.command_id()
        {
            self.update_command_journal(command_id, JournalState::Failed(failure.clone()))?;
        }
        if let Some(event_sequence) = batch.scientific_event_sequence() {
            self.events.update_commitment(
                batch.id(),
                event_sequence,
                EventCommitment::Failed(failure),
            )?;
        }
        self.failed_journal_batches.insert(batch.id());
        self.latched_fault = Some(HostFault::Protocol {
            code: "journal_identity_mismatch".to_owned(),
            message: format!(
                "journal response for {:?} echoed {:?}",
                batch.id(),
                admission.batch_id()
            ),
        });
        self.synchronize_health()?;
        Ok(true)
    }

    fn seal_core_persistence(&mut self) -> Result<(), HostAccessError> {
        self.persistence
            .admit_pending(&mut self.world)
            .map(|_| ())
            .map_err(|error| {
                protocol_violation(format!("could not seal core persistence: {error}"))
            })
    }

    fn fail_closed_batch(&mut self, batch: &Arc<JournalBatch>) -> Result<(), HostAccessError> {
        self.retained_journal = Some(Arc::clone(batch));
        if let Some(command_id) = batch.command_id() {
            self.shared
                .borrow_mut()
                .compact_command_envelope(command_id)?;
        }
        self.retained_blocker = Some(HostBlocker::JournalClosed {
            batch_id: batch.id(),
        });
        let failure = JournalFailure {
            code: "journal_closed".to_owned(),
            message: "journal admission gate is permanently closed".to_owned(),
        };
        if batch.requires_runtime_journal()
            && let Some(command_id) = batch.command_id()
        {
            self.update_command_journal(command_id, JournalState::Failed(failure.clone()))?;
        }
        if let Some(event_sequence) = batch.scientific_event_sequence() {
            self.events.update_commitment(
                batch.id(),
                event_sequence,
                EventCommitment::Failed(failure.clone()),
            )?;
        }
        self.failed_journal_batches.insert(batch.id());
        self.latched_fault = Some(HostFault::Journal {
            batch_id: batch.id(),
            failure,
        });
        self.synchronize_health().map(|_| ())
    }

    fn poll_journal_receipts(&mut self) -> Result<bool, HostAccessError> {
        let mut changed = false;
        for receipt in self.journal.poll_receipts(RECEIPT_POLL_LIMIT) {
            let batch_id = receipt.batch_id();
            let Some(inflight) = self.inflight_journal.get(&batch_id).copied() else {
                self.latch_protocol_fault(
                    "unknown_journal_receipt",
                    format!("journal acknowledged unknown batch {batch_id:?}"),
                )?;
                changed = true;
                continue;
            };

            let journal_state = match receipt.state() {
                JournalReceiptState::CommittedVolatile => JournalState::CommittedVolatile,
                JournalReceiptState::Durable => JournalState::Durable,
                JournalReceiptState::Failed(failure) => JournalState::Failed(failure.clone()),
            };
            if let Some(command_id) = inflight.command_id {
                changed |= self.update_command_journal(command_id, journal_state.clone())?;
            }
            if let Some(event_sequence) = inflight.scientific_event {
                let commitment = match receipt.state() {
                    JournalReceiptState::CommittedVolatile => EventCommitment::CommittedVolatile,
                    JournalReceiptState::Durable => EventCommitment::Durable,
                    JournalReceiptState::Failed(failure) => {
                        EventCommitment::Failed(failure.clone())
                    }
                };
                self.events
                    .update_commitment(batch_id, event_sequence, commitment)?;
                changed = true;
            }

            match receipt.state() {
                JournalReceiptState::CommittedVolatile => {
                    let terminal = self.journal.shutdown_commit_requirement()
                        == ShutdownCommitRequirement::CommittedVolatile;
                    if terminal {
                        self.inflight_journal.remove(&batch_id);
                    } else if let Some(entry) = self.inflight_journal.get_mut(&batch_id) {
                        entry.committed_volatile = true;
                    }
                    if let Some(requirement @ ShutdownCommitRequirement::CommittedVolatile) =
                        inflight.shutdown_requirement
                    {
                        self.shutdown_receipt = Some((batch_id, requirement));
                    }
                }
                JournalReceiptState::Durable => {
                    self.inflight_journal.remove(&batch_id);
                    if let Some(requirement) = inflight.shutdown_requirement {
                        self.shutdown_receipt = Some((batch_id, requirement));
                    }
                }
                JournalReceiptState::Failed(failure) => {
                    self.inflight_journal.remove(&batch_id);
                    self.failed_journal_batches.insert(batch_id);
                    self.latched_fault = Some(HostFault::Journal {
                        batch_id,
                        failure: failure.clone(),
                    });
                    changed = true;
                }
            }
            changed |= self.try_finish_shutdown()?;
        }
        changed |= self.synchronize_health()?;
        Ok(changed)
    }

    fn update_command_journal(
        &self,
        command_id: CommandId,
        journal: JournalState,
    ) -> Result<bool, HostAccessError> {
        let current = self
            .shared
            .borrow()
            .commands
            .get(&command_id)
            .map(|authority| authority.status.clone());
        let Some(current) = current else {
            // The record already moved to the archived idempotency index at durable
            // archival; a duplicate or later receipt upgrades it without resurrecting
            // the live authority (bd-2z0.5.2.1).
            let mut shared = self.shared.borrow_mut();
            let Some(archived) = shared.archived_idempotency.get_mut(&command_id) else {
                return Err(protocol_violation(
                    "journal receipt command status is missing",
                ));
            };
            if archived.status.journal() == &journal
                || matches!(
                    archived.status.journal(),
                    JournalState::Durable | JournalState::Failed(_)
                )
            {
                return Ok(false);
            }
            let upgraded = CommandStatus::try_new(
                command_id,
                archived.status.admission_sequence(),
                archived.status.application().clone(),
                journal,
            )
            .map_err(status_violation)?;
            archived.status = upgraded;
            return Ok(true);
        };
        if current.journal() == &journal
            || matches!(
                current.journal(),
                JournalState::Durable | JournalState::Failed(_)
            )
        {
            return Ok(false);
        }
        let status = CommandStatus::try_new(
            command_id,
            current.admission_sequence(),
            current.application().clone(),
            journal.clone(),
        )
        .map_err(status_violation)?;
        self.shared.borrow_mut().store_status(status)?;
        if matches!(
            journal,
            JournalState::CommittedVolatile | JournalState::Durable
        ) {
            self.shared
                .borrow_mut()
                .archive_terminal_command(command_id);
        }
        Ok(true)
    }

    fn finish_shutdown(&mut self) -> Result<bool, HostAccessError> {
        if self.lifecycle == HostLifecycle::Stopped {
            return Ok(false);
        }
        self.lifecycle = HostLifecycle::Stopped;
        let mut shared = self.shared.borrow_mut();
        shared.admission_lifecycle = HostLifecycle::Stopped;
        shared.emit(HostEventKind::LifecycleChanged(HostLifecycle::Stopped))?;
        Ok(true)
    }

    fn try_finish_shutdown(&mut self) -> Result<bool, HostAccessError> {
        let Some((shutdown_id, requirement)) = self.shutdown_receipt else {
            return Ok(false);
        };
        let earlier_work_failed = self.failed_journal_batches.iter().any(|batch_id| {
            batch_id.session_id() == shutdown_id.session_id()
                && batch_id.sequence() <= shutdown_id.sequence()
        });
        if earlier_work_failed {
            return Ok(false);
        }
        let earlier_work_pending = self.inflight_journal.iter().any(|(batch_id, inflight)| {
            batch_id.session_id() == shutdown_id.session_id()
                && batch_id.sequence() <= shutdown_id.sequence()
                && match requirement {
                    ShutdownCommitRequirement::CommittedVolatile => !inflight.committed_volatile,
                    ShutdownCommitRequirement::Durable => true,
                }
        });
        if earlier_work_pending {
            Ok(false)
        } else {
            self.finish_shutdown()
        }
    }

    fn latch_protocol_fault(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), HostAccessError> {
        self.latched_fault = Some(HostFault::Protocol {
            code: code.into(),
            message: message.into(),
        });
        self.synchronize_health().map(|_| ())
    }

    fn synchronize_health(&mut self) -> Result<bool, HostAccessError> {
        let next = if let Some(fault) = &self.latched_fault {
            HostHealth::Faulted(fault.clone())
        } else if let Some(blocker) = self.retained_blocker {
            HostHealth::Blocked(blocker)
        } else if let Some(blocker) = self.event_pressure {
            HostHealth::Blocked(blocker)
        } else {
            match self.lifecycle {
                HostLifecycle::Running => HostHealth::Healthy,
                HostLifecycle::Stopping => HostHealth::Blocked(HostBlocker::LifecycleStopping),
                HostLifecycle::Stopped => HostHealth::Blocked(HostBlocker::LifecycleStopped),
            }
        };
        if next == self.health {
            return Ok(false);
        }
        self.health = next.clone();
        self.shared
            .borrow_mut()
            .emit(HostEventKind::HealthChanged(next))?;
        Ok(true)
    }

    const fn current_blocker(&self) -> Option<HostBlocker> {
        if let Some(blocker) = self.retained_blocker {
            return Some(blocker);
        }
        if let Some(blocker) = self.event_pressure {
            return Some(blocker);
        }
        if let Some(blocker) = self.health.blocker() {
            return Some(blocker);
        }
        if self.health.fault().is_some() {
            return Some(HostBlocker::ScientificFault);
        }
        match self.lifecycle {
            HostLifecycle::Stopping => Some(HostBlocker::LifecycleStopping),
            HostLifecycle::Stopped => Some(HostBlocker::LifecycleStopped),
            HostLifecycle::Running if self.playback.paused => Some(HostBlocker::PlaybackPaused),
            HostLifecycle::Running => None,
        }
    }

    fn prepare_scientific_event_slot(&mut self) -> Result<bool, HostAccessError> {
        let pressure = self.events.prepare_publish()?;
        let ready = pressure.is_none();
        if !ready {
            self.cadence_credit = 0;
        }
        self.event_pressure = pressure.map(|pressure| HostBlocker::EventJournalHighWater {
            capacity: self.events.capacity(),
            pending: self.events.pending_count(),
            oldest_pending: self.events.oldest_pending_batch(),
            pinned_batch: pressure.batch_id,
            pinned_sequence: pressure.sequence,
            reason: pressure.reason,
        });
        self.synchronize_health()?;
        Ok(ready)
    }

    fn publish_snapshot(&mut self) -> Result<(), HostAccessError> {
        let revision = self.next_snapshot;
        let following_revision = revision
            .checked_next()
            .ok_or_else(|| protocol_violation("snapshot revision exhausted"))?;
        let mut layers = self.snapshot_layers.clone();
        let refresh = layers.refresh(&self.world)?;
        let dynamic_world = DynamicWorldSnapshot::from_world(&self.world);
        let summary_history_allocated = self.revisions.scientific != self.last_published_scientific;
        let summary_history = if summary_history_allocated {
            Arc::new(self.world.history().cloned().collect::<Vec<_>>())
        } else {
            Arc::clone(&self.summary_history)
        };
        let build = snapshot_build_stats(
            &dynamic_world,
            &summary_history,
            summary_history.capacity(),
            summary_history_allocated,
            &layers,
            refresh,
        );
        let (command_queue_depth, last_applied_command) = {
            let shared = self.shared.borrow();
            (
                shared.queue.len(),
                shared.last_applied.map(|(_, command_id)| command_id),
            )
        };
        let snapshot = Arc::new(RenderSnapshot {
            session_id: self.session_id,
            revision,
            revisions: self.revisions,
            playback: self.playback,
            lifecycle: self.lifecycle,
            health: self.health.clone(),
            command_queue_depth,
            last_applied_command,
            completed_summary: self.latest_completed_summary.clone(),
            summary_history: Arc::clone(&summary_history),
            layers: layers.snapshot(),
            build,
            world: dynamic_world,
        });
        self.snapshots.publish(snapshot)?;
        self.snapshot_layers = layers;
        self.summary_history = summary_history;
        self.next_snapshot = following_revision;
        self.last_published_scientific = self.revisions.scientific;
        self.shared.borrow_mut().visible_boundary = self.applied_boundary();
        Ok(())
    }

    fn pop_command(&self) -> Option<AdmittedEnvelope> {
        self.shared.borrow_mut().queue.pop_front()
    }

    fn drain_pending_command_audits(&mut self) -> Result<bool, HostAccessError> {
        loop {
            let Some((command_id, order, lifecycle)) = self.shared.borrow().next_pending_audit()?
            else {
                return Ok(false);
            };
            self.ensure_journal_sequence_available()?;
            self.shared
                .borrow_mut()
                .claim_pending_audit(command_id, order)?;
            if self.offer_command_lifecycle(lifecycle)? {
                return Ok(true);
            }
        }
    }

    fn next_command_requires_scientific_event(&self) -> bool {
        self.shared
            .borrow()
            .queue
            .front()
            .is_some_and(|admitted| matches!(&admitted.envelope.command, HostCommand::Step))
    }

    fn ensure_journal_sequence_available(&self) -> Result<(), HostAccessError> {
        self.next_journal_sequence
            .checked_add(1)
            .map(|_| ())
            .ok_or_else(|| protocol_violation("journal batch sequence exhausted"))
    }

    fn complete_status(&self, status: CommandStatus) -> Result<(), HostAccessError> {
        let mut shared = self.shared.borrow_mut();
        shared.visible_boundary = self.applied_boundary();
        shared.store_status(status)
    }

    fn apply_command(
        &mut self,
        admitted: AdmittedEnvelope,
    ) -> Result<ApplyResult, HostAccessError> {
        let AdmittedEnvelope {
            admission,
            envelope,
        } = admitted;
        self.ensure_journal_sequence_available()?;
        let revision_conflict = if let Some(expected) = envelope.expected_control_revision
            && expected != self.revisions.control
        {
            Some(RejectionReason::ControlRevisionConflict {
                expected,
                actual: self.revisions.control,
            })
        } else if let Some(expected) = envelope.expected_scientific_revision
            && expected != self.revisions.scientific
        {
            Some(RejectionReason::ScientificRevisionConflict {
                expected,
                actual: self.revisions.scientific,
            })
        } else if let Some(expected) = envelope.expected_config_revision
            && expected != self.revisions.config
        {
            Some(RejectionReason::ConfigRevisionConflict {
                expected,
                actual: self.revisions.config,
            })
        } else {
            None
        };
        if let Some(reason) = revision_conflict {
            if matches!(&envelope.command, HostCommand::Shutdown) {
                let mut shared = self.shared.borrow_mut();
                shared.admission_lifecycle = HostLifecycle::Running;
                shared.shutdown_command_id = None;
            }
            let status = CommandStatus::try_new(
                envelope.command_id,
                Some(admission),
                ApplicationState::Rejected(reason),
                JournalState::Pending,
            )
            .map_err(status_violation)?;
            self.complete_status(status)?;
            let blocked = self.offer_terminal_command_audit(envelope.command_id)?;
            return Ok(ApplyResult::completed(blocked));
        }

        if self.latched_fault.is_some()
            && matches!(
                &envelope.command,
                HostCommand::Step | HostCommand::UpdateConfig(_)
            )
        {
            let blocked = self.complete_failed(
                envelope.command_id,
                admission,
                "science_blocked",
                "host science is stopped by a latched fault".to_owned(),
            )?;
            return Ok(ApplyResult::completed(blocked));
        }

        let next_control = self
            .revisions
            .control
            .checked_next()
            .ok_or_else(|| protocol_violation("control revision exhausted"))?;
        let retry_envelope = envelope.clone();
        match envelope.command {
            HostCommand::Pause => {
                self.playback.paused = true;
                self.revisions.control = next_control;
                self.complete_applied(retry_envelope.command_id, admission)?;
                let blocked = self.offer_terminal_command_audit(retry_envelope.command_id)?;
                Ok(ApplyResult::completed(blocked))
            }
            HostCommand::Resume => {
                self.playback.paused = false;
                self.revisions.control = next_control;
                self.complete_applied(retry_envelope.command_id, admission)?;
                let blocked = self.offer_terminal_command_audit(retry_envelope.command_id)?;
                Ok(ApplyResult::completed(blocked))
            }
            HostCommand::SetSpeed(speed) => {
                self.playback.speed_multiplier = speed;
                self.revisions.control = next_control;
                self.complete_applied(retry_envelope.command_id, admission)?;
                let blocked = self.offer_terminal_command_audit(retry_envelope.command_id)?;
                Ok(ApplyResult::completed(blocked))
            }
            HostCommand::UpdateConfig(config) => {
                self.apply_config_command(admission, retry_envelope, config, next_control)
            }
            HostCommand::Step => self.apply_step_command(admission, retry_envelope, next_control),
            HostCommand::Shutdown => {
                self.apply_shutdown_command(admission, retry_envelope, next_control)
            }
        }
    }

    fn apply_config_command(
        &mut self,
        admission: AdmissionSequence,
        envelope: CommandEnvelope,
        config: Box<ScriptBotsConfig>,
        next_control: ControlRevision,
    ) -> Result<ApplyResult, HostAccessError> {
        if let Err(error) = self.world.apply_config_update(*config) {
            let blocked = self.complete_failed(
                envelope.command_id,
                admission,
                "config_application",
                error.to_string(),
            )?;
            return Ok(ApplyResult::completed(blocked));
        }
        self.revisions.control = next_control;
        self.revisions.config = ConfigRevision::new(self.world.config_revision());
        let applied = self.applied_boundary();
        self.complete_applied_with(envelope.command_id, admission, applied)?;
        let blocked = self.offer_journal(&envelope, applied, None, None)?;
        Ok(ApplyResult::completed(blocked))
    }

    fn apply_step_command(
        &mut self,
        admission: AdmissionSequence,
        envelope: CommandEnvelope,
        next_control: ControlRevision,
    ) -> Result<ApplyResult, HostAccessError> {
        let next_scientific = self
            .revisions
            .scientific
            .checked_next()
            .ok_or_else(|| protocol_violation("scientific revision exhausted"))?;
        let completion = match self.persistence.step_outcome(&mut self.world) {
            Ok(completion) => completion,
            Err(error) => {
                let blocked = self.complete_failed(
                    envelope.command_id,
                    admission,
                    "world_step",
                    error.to_string(),
                )?;
                self.latched_fault = Some(HostFault::Scientific {
                    tick: self.world.tick(),
                    code: "world_step".to_owned(),
                    message: error.to_string(),
                });
                self.synchronize_health()?;
                return Ok(ApplyResult::completed(blocked));
            }
        };
        self.playback.paused = true;
        self.cadence_credit = 0;
        let scriptbots_core::StepCompletion {
            outcome,
            fault: completed_fault,
        } = completion;
        let scriptbots_core::StepOutcome {
            events,
            summary,
            births,
            deaths,
            combat,
            config_revision,
            resource_tick,
            persistence,
        } = outcome;
        self.revisions.control = next_control;
        self.revisions.scientific = next_scientific;
        self.revisions.config = ConfigRevision::new(config_revision);
        self.latest_completed_summary = Some(summary.clone());
        let tick = summary.tick;
        let completed_fault = completed_fault.as_ref().map(completed_fault_record);
        let mut scientific = ScientificBoundary::new(
            events,
            summary,
            births,
            deaths,
            combat,
            config_revision,
            resource_tick,
        );
        if let Some(fault) = completed_fault.clone() {
            scientific = scientific.with_fault(fault);
        }
        let scientific = Arc::new(scientific);
        let persistence = persistence.into_batch();
        let applied = AppliedCommand {
            tick,
            revisions: self.revisions,
        };
        self.complete_applied_with(envelope.command_id, admission, applied)?;
        let blocked = self.offer_journal(&envelope, applied, Some(scientific), persistence)?;
        if let Some(fault) = completed_fault {
            self.latch_completed_step_fault(tick, &fault)?;
        }
        Ok(ApplyResult::completed(blocked).with_science())
    }

    fn apply_shutdown_command(
        &mut self,
        admission: AdmissionSequence,
        envelope: CommandEnvelope,
        next_control: ControlRevision,
    ) -> Result<ApplyResult, HostAccessError> {
        {
            let mut shared = self.shared.borrow_mut();
            if shared.pending_audit_count != 0 {
                return Err(protocol_violation(
                    "shutdown application began before pre-admission audits drained",
                ));
            }
            shared.audit_gate_closed = true;
        }
        let persistence = match self.persistence.stage_final_batch(&mut self.world) {
            Ok(persistence) => persistence,
            Err(error) => {
                self.lifecycle = HostLifecycle::Stopping;
                self.shared
                    .borrow_mut()
                    .emit(HostEventKind::LifecycleChanged(HostLifecycle::Stopping))?;
                let blocked = self.complete_failed(
                    envelope.command_id,
                    admission,
                    "shutdown_finalization",
                    error.to_string(),
                )?;
                self.latched_fault = Some(HostFault::Scientific {
                    tick: self.world.tick(),
                    code: "shutdown_finalization".to_owned(),
                    message: error.to_string(),
                });
                self.synchronize_health()?;
                return Ok(ApplyResult::completed(blocked));
            }
        };
        self.lifecycle = HostLifecycle::Stopping;
        self.shared
            .borrow_mut()
            .emit(HostEventKind::LifecycleChanged(HostLifecycle::Stopping))?;
        self.revisions.control = next_control;
        let applied = self.applied_boundary();
        self.complete_applied_with(envelope.command_id, admission, applied)?;
        let blocked = self.offer_journal(&envelope, applied, None, persistence)?;
        self.synchronize_health()?;
        Ok(ApplyResult::completed(blocked))
    }

    const fn applied_boundary(&self) -> AppliedCommand {
        AppliedCommand {
            tick: self.world.tick(),
            revisions: self.revisions,
        }
    }

    fn complete_applied(
        &self,
        command_id: CommandId,
        admission: AdmissionSequence,
    ) -> Result<(), HostAccessError> {
        self.complete_applied_with(command_id, admission, self.applied_boundary())
    }

    fn complete_applied_with(
        &self,
        command_id: CommandId,
        admission: AdmissionSequence,
        applied: AppliedCommand,
    ) -> Result<(), HostAccessError> {
        let status = CommandStatus::try_new(
            command_id,
            Some(admission),
            ApplicationState::Applied(applied),
            JournalState::Pending,
        )
        .map_err(status_violation)?;
        self.complete_status(status)
    }

    fn complete_failed(
        &mut self,
        command_id: CommandId,
        admission: AdmissionSequence,
        code: &str,
        message: String,
    ) -> Result<bool, HostAccessError> {
        let status = CommandStatus::try_new(
            command_id,
            Some(admission),
            ApplicationState::Failed(ApplicationFailure {
                code: code.to_owned(),
                message,
            }),
            JournalState::Pending,
        )
        .map_err(status_violation)?;
        self.complete_status(status)?;
        self.offer_terminal_command_audit(command_id)
    }

    fn offer_journal(
        &mut self,
        envelope: &CommandEnvelope,
        applied: AppliedCommand,
        scientific: Option<Arc<ScientificBoundary>>,
        persistence: Option<Arc<scriptbots_core::PersistenceBatch>>,
    ) -> Result<bool, HostAccessError> {
        let batch_id = JournalBatchId::new(self.session_id, self.next_journal_sequence);
        self.next_journal_sequence = self
            .next_journal_sequence
            .checked_add(1)
            .ok_or_else(|| protocol_violation("journal batch sequence exhausted"))?;
        let lifecycle = self
            .shared
            .borrow()
            .lifecycle_evidence(envelope.command_id)?;
        if command_envelope_digest(lifecycle.envelope())? != command_envelope_digest(&envelope)? {
            return Err(protocol_violation(
                "journal command differs from its retained lifecycle envelope",
            ));
        }
        let scientific_event_sequence = scientific
            .as_ref()
            .map(|boundary| {
                self.events
                    .publish_pending(batch_id, applied, Arc::clone(boundary))
            })
            .transpose()?;
        let batch = Arc::new(JournalBatch::new(
            batch_id,
            scientific_event_sequence,
            Some(lifecycle),
            applied,
            scientific,
            persistence,
        ));
        self.active_journal_batch = Some(Arc::clone(&batch));
        let admission = self.journal.try_admit(&batch);
        let result = self.finish_journal_admission(&batch, admission, false);
        if result.is_ok() {
            self.active_journal_batch = None;
        }
        result
    }

    fn offer_terminal_command_audit(
        &mut self,
        command_id: CommandId,
    ) -> Result<bool, HostAccessError> {
        let lifecycle = self.shared.borrow().lifecycle_evidence(command_id)?;
        self.offer_command_lifecycle(lifecycle)
    }

    fn offer_command_lifecycle(
        &mut self,
        lifecycle: CommandLifecycleEvidence,
    ) -> Result<bool, HostAccessError> {
        let applied = lifecycle
            .terminal()
            .map(CommandLifecycleTransition::boundary)
            .ok_or_else(|| protocol_violation("terminal command lifecycle is empty"))?;
        let batch_id = JournalBatchId::new(self.session_id, self.next_journal_sequence);
        self.next_journal_sequence = self
            .next_journal_sequence
            .checked_add(1)
            .ok_or_else(|| protocol_violation("journal batch sequence exhausted"))?;
        let batch = Arc::new(JournalBatch::new(
            batch_id,
            None,
            Some(lifecycle),
            applied,
            None,
            None,
        ));
        self.active_journal_batch = Some(Arc::clone(&batch));
        let admission = self.journal.try_admit(&batch);
        let result = self.finish_journal_admission(&batch, admission, false);
        if result.is_ok() {
            self.active_journal_batch = None;
        }
        result
    }

    fn offer_automatic_journal(
        &mut self,
        applied: AppliedCommand,
        scientific: Arc<ScientificBoundary>,
        persistence: Option<Arc<scriptbots_core::PersistenceBatch>>,
    ) -> Result<bool, HostAccessError> {
        let batch_id = JournalBatchId::new(self.session_id, self.next_journal_sequence);
        self.next_journal_sequence = self
            .next_journal_sequence
            .checked_add(1)
            .ok_or_else(|| protocol_violation("journal batch sequence exhausted"))?;
        let scientific_event_sequence =
            self.events
                .publish_pending(batch_id, applied, Arc::clone(&scientific))?;
        let batch = Arc::new(JournalBatch::new(
            batch_id,
            Some(scientific_event_sequence),
            None,
            applied,
            Some(scientific),
            persistence,
        ));
        self.active_journal_batch = Some(Arc::clone(&batch));
        let admission = self.journal.try_admit(&batch);
        let result = self.finish_journal_admission(&batch, admission, false);
        if result.is_ok() {
            self.active_journal_batch = None;
        }
        result
    }

    fn finish_journal_admission(
        &mut self,
        batch: &Arc<JournalBatch>,
        admission: JournalAdmission,
        was_retained: bool,
    ) -> Result<bool, HostAccessError> {
        if self.retain_identity_violation(batch, admission)? {
            return Ok(true);
        }
        match admission {
            JournalAdmission::Accepted { .. } => {
                if let Some(command_id) = batch.command_id() {
                    self.shared
                        .borrow_mut()
                        .compact_command_envelope(command_id)?;
                }
                if batch.uses_ingress_audit_slot() {
                    self.shared.borrow_mut().release_pre_admission_audit()?;
                }
                self.seal_core_persistence()?;
                let shutdown_requirement = if batch.is_applied_shutdown() {
                    Some(self.journal.shutdown_commit_requirement())
                } else {
                    None
                };
                self.inflight_journal.insert(
                    batch.id(),
                    InflightJournal {
                        command_id: batch.command_id(),
                        scientific_event: batch.scientific_event_sequence(),
                        shutdown_requirement,
                        committed_volatile: false,
                    },
                );
                if was_retained {
                    self.retained_journal = None;
                    self.retained_blocker = None;
                }
                Ok(false)
            }
            JournalAdmission::Full { capacity, .. } => {
                self.retained_journal = Some(Arc::clone(batch));
                if let Some(command_id) = batch.command_id() {
                    self.shared
                        .borrow_mut()
                        .compact_command_envelope(command_id)?;
                }
                self.retained_blocker = Some(HostBlocker::JournalFull {
                    batch_id: batch.id(),
                    capacity,
                });
                self.synchronize_health()?;
                Ok(true)
            }
            JournalAdmission::Closed { .. } => {
                self.fail_closed_batch(batch)?;
                Ok(true)
            }
        }
    }

    fn latch_completed_step_fault(
        &mut self,
        tick: Tick,
        fault: &ScientificBoundaryFault,
    ) -> Result<(), HostAccessError> {
        self.latched_fault = Some(HostFault::Scientific {
            tick,
            code: fault.code().to_owned(),
            message: fault.message().to_owned(),
        });
        self.synchronize_health().map(|_| ())
    }

    fn automatic_step(&mut self) -> Result<ApplyResult, HostAccessError> {
        self.ensure_journal_sequence_available()?;
        let next_scientific = self
            .revisions
            .scientific
            .checked_next()
            .ok_or_else(|| protocol_violation("scientific revision exhausted"))?;
        let completion = match self.persistence.step_outcome(&mut self.world) {
            Ok(completion) => completion,
            Err(error) => {
                self.latched_fault = Some(HostFault::Scientific {
                    tick: self.world.tick(),
                    code: "world_step".to_owned(),
                    message: error.to_string(),
                });
                self.synchronize_health()?;
                return Ok(ApplyResult::blocked());
            }
        };
        let scriptbots_core::StepCompletion {
            outcome,
            fault: completed_fault,
        } = completion;
        let scriptbots_core::StepOutcome {
            events,
            summary,
            births,
            deaths,
            combat,
            config_revision,
            resource_tick,
            persistence,
        } = outcome;
        self.revisions.scientific = next_scientific;
        self.revisions.config = ConfigRevision::new(config_revision);
        self.latest_completed_summary = Some(summary.clone());
        let tick = summary.tick;
        self.shared.borrow_mut().visible_boundary = AppliedCommand {
            tick,
            revisions: self.revisions,
        };
        let completed_fault = completed_fault.as_ref().map(completed_fault_record);
        let mut scientific = ScientificBoundary::new(
            events,
            summary,
            births,
            deaths,
            combat,
            config_revision,
            resource_tick,
        );
        if let Some(fault) = completed_fault.clone() {
            scientific = scientific.with_fault(fault);
        }
        let scientific = Arc::new(scientific);
        let persistence = persistence.into_batch();
        let applied = AppliedCommand {
            tick,
            revisions: self.revisions,
        };
        let blocked = self.offer_automatic_journal(applied, scientific, persistence)?;
        if let Some(fault) = completed_fault {
            self.latch_completed_step_fault(tick, &fault)?;
        }
        Ok(ApplyResult::science(
            blocked || self.latched_fault.is_some(),
        ))
    }

    fn automatic_budget(&mut self, elapsed_nanos: u64, speed_multiplier: f32) -> AutomaticBudget {
        let speed_units = speed_units(speed_multiplier);
        let threshold = u128::from(self.options.tick_period_nanos) * SPEED_SCALE;
        let maximum_steps =
            u128::try_from(self.options.max_automatic_steps_per_drive).unwrap_or(u128::MAX);
        let total_credit = self
            .cadence_credit
            .saturating_add(u128::from(elapsed_nanos).saturating_mul(u128::from(speed_units)));
        let due = total_credit / threshold;
        let admitted = due.min(maximum_steps);
        let steps = usize::try_from(admitted)
            .unwrap_or(self.options.max_automatic_steps_per_drive)
            .min(self.options.max_automatic_steps_per_drive);
        let skipped = due.saturating_sub(admitted);
        self.cadence_credit =
            (total_credit % threshold).saturating_add(admitted.saturating_mul(threshold));
        AutomaticBudget {
            steps,
            due: u64::try_from(due).unwrap_or(u64::MAX),
            skipped: u64::try_from(skipped).unwrap_or(u64::MAX),
        }
    }

    fn consume_automatic_credit(&mut self) {
        let threshold = u128::from(self.options.tick_period_nanos) * SPEED_SCALE;
        self.cadence_credit = self.cadence_credit.saturating_sub(threshold);
    }
}

impl ManualHostDriver for HostCore {
    fn session_id(&self) -> HostSessionId {
        self.session_id
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the sole-owner boundary keeps its ordered command, science, health, and publication phases visible together"
    )]
    fn drive(&mut self, now: ManualInstant) -> Result<DriveReceipt, HostAccessError> {
        if self.last_now.is_some_and(|last| now < last) {
            return Err(protocol_violation("manual time moved backwards"));
        }
        let elapsed_nanos = self
            .last_now
            .map_or(0, |last| now.as_nanos().saturating_sub(last.as_nanos()));
        let first_boundary = self.last_now.is_none();
        let was_running = self.lifecycle == HostLifecycle::Running && !self.playback.paused;
        let prior_speed = self.playback.speed_multiplier;
        self.last_now = Some(now);

        self.events.cancel_publish_reservation();
        let events_before = self.events.published_total();
        let event_was_pressured = self.event_pressure.is_some();
        self.poll_journal_receipts()?;
        if self.event_pressure.is_some() {
            self.prepare_scientific_event_slot()?;
        }
        let mut commands_completed = 0;
        let mut scientific_steps = 0;
        let mut automatic_steps_due = 0;
        let mut automatic_steps_skipped = 0;
        let mut explicit_step_applied = false;

        let audit_blocked = if self.retained_journal.is_none() {
            self.drain_pending_command_audits()?
        } else {
            true
        };
        if !audit_blocked {
            loop {
                if self.next_command_requires_scientific_event()
                    && !self.prepare_scientific_event_slot()?
                {
                    break;
                }
                let Some(admitted) = self.pop_command() else {
                    break;
                };
                self.active_command = Some(admitted.clone());
                let result = self.apply_command(admitted);
                let reservation_unused = result
                    .as_ref()
                    .map_or(true, |result| !result.science_completed);
                if reservation_unused {
                    self.events.cancel_publish_reservation();
                }
                if result.is_ok() {
                    self.active_command = None;
                }
                let result = result?;
                commands_completed += usize::from(result.command_completed);
                scientific_steps += usize::from(result.science_completed);
                explicit_step_applied |= result.science_completed;
                if result.blocked {
                    break;
                }
            }
        }

        if !first_boundary
            && !explicit_step_applied
            && was_running
            && self.lifecycle == HostLifecycle::Running
            && !self.playback.paused
            && self.retained_journal.is_none()
            && self.latched_fault.is_none()
            && self.event_pressure.is_none()
            && !event_was_pressured
        {
            let budget = self.automatic_budget(elapsed_nanos, prior_speed);
            automatic_steps_due = budget.due;
            automatic_steps_skipped = budget.skipped;
            for _ in 0..budget.steps {
                if !self.prepare_scientific_event_slot()? {
                    break;
                }
                let result = self.automatic_step();
                let reservation_unused = result
                    .as_ref()
                    .map_or(true, |result| !result.science_completed);
                if reservation_unused {
                    self.events.cancel_publish_reservation();
                }
                let result = result?;
                if result.science_completed {
                    self.consume_automatic_credit();
                    scientific_steps += 1;
                }
                if result.blocked {
                    break;
                }
            }
        } else if self.playback.paused || explicit_step_applied {
            self.cadence_credit = 0;
        }

        self.synchronize_health()?;
        let latest = self.snapshots.latest();
        let (command_queue_depth, last_applied_command) = {
            let shared = self.shared.borrow();
            (
                shared.queue.len(),
                shared.last_applied.map(|(_, command_id)| command_id),
            )
        };
        let presentation_changed = latest.revisions.control != self.revisions.control
            || latest.revisions.config != self.revisions.config
            || latest.playback != self.playback
            || latest.lifecycle != self.lifecycle
            || latest.health != self.health
            || latest.command_queue_depth != command_queue_depth
            || latest.last_applied_command != last_applied_command;
        let scientific_since_publication = self
            .revisions
            .scientific
            .get()
            .saturating_sub(self.last_published_scientific.get());
        let science_cadence_due = scientific_steps != 0
            && scientific_since_publication >= self.options.snapshot_interval_ticks;
        let should_publish = presentation_changed || science_cadence_due;
        let snapshots_published = usize::from(should_publish);
        if should_publish {
            self.publish_snapshot()?;
        }
        let events_published =
            usize::try_from(self.events.published_total().saturating_sub(events_before))
                .unwrap_or(usize::MAX);
        self.events.cancel_publish_reservation();
        Ok(DriveReceipt {
            now,
            commands_completed,
            scientific_steps,
            automatic_steps_due,
            automatic_steps_skipped,
            scientific_revision: self.revisions.scientific,
            snapshots_published,
            events_published,
            blocker: self.current_blocker(),
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct AutomaticBudget {
    steps: usize,
    due: u64,
    skipped: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ApplyResult {
    command_completed: bool,
    science_completed: bool,
    blocked: bool,
}

impl ApplyResult {
    const fn completed(blocked: bool) -> Self {
        Self {
            command_completed: true,
            science_completed: false,
            blocked,
        }
    }

    const fn with_science(mut self) -> Self {
        self.science_completed = true;
        self
    }

    const fn science(blocked: bool) -> Self {
        Self {
            command_completed: false,
            science_completed: true,
            blocked,
        }
    }

    const fn blocked() -> Self {
        Self {
            command_completed: false,
            science_completed: false,
            blocked: true,
        }
    }
}

fn validate_options(options: HostCoreOptions) -> Result<(), HostCoreBuildError> {
    if options.command_capacity == 0 {
        return Err(HostCoreBuildError::InvalidOptions {
            message: "command_capacity must be nonzero".to_owned(),
        });
    }
    if options.tick_period_nanos == 0 {
        return Err(HostCoreBuildError::InvalidOptions {
            message: "tick_period_nanos must be nonzero".to_owned(),
        });
    }
    if options.max_automatic_steps_per_drive == 0 {
        return Err(HostCoreBuildError::InvalidOptions {
            message: "max_automatic_steps_per_drive must be nonzero".to_owned(),
        });
    }
    if options.snapshot_interval_ticks == 0 {
        return Err(HostCoreBuildError::InvalidOptions {
            message: "snapshot_interval_ticks must be nonzero".to_owned(),
        });
    }
    if options.scientific_event_capacity == 0 {
        return Err(HostCoreBuildError::InvalidOptions {
            message: "scientific_event_capacity must be nonzero".to_owned(),
        });
    }
    if options.protocol_event_capacity == 0 {
        return Err(HostCoreBuildError::InvalidOptions {
            message: "protocol_event_capacity must be nonzero".to_owned(),
        });
    }
    if !options.initial_playback.speed_multiplier.is_finite()
        || options.initial_playback.speed_multiplier < 0.0
    {
        return Err(HostCoreBuildError::InvalidOptions {
            message: "initial speed multiplier must be finite and non-negative".to_owned(),
        });
    }
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn speed_units(speed: f32) -> u64 {
    let scaled = f64::from(speed) * 1_000_000.0;
    if scaled >= u64::MAX as f64 {
        u64::MAX
    } else {
        scaled.round() as u64
    }
}

fn status_violation(error: StatusCombinationError) -> HostAccessError {
    protocol_violation(error.to_string())
}

fn completed_fault_record(fault: &CompletedStepFault) -> ScientificBoundaryFault {
    let code = match fault {
        CompletedStepFault::BrainSpawn(_) => "brain_spawn",
        CompletedStepFault::ScientificState(_) => "scientific_state",
    };
    ScientificBoundaryFault::new(code, fault.to_string())
}

fn protocol_violation(message: impl Into<String>) -> HostAccessError {
    HostAccessError::ProtocolViolation {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BrainProjectionRequest, CommandLifecycleTransition, ProjectionBroker, ProjectionCamera,
        ProjectionClientId, ProjectionDetail, ProjectionLimits, ProjectionRanking,
        ProjectionRequest, ProjectionRequestRevision, ProjectionSelection, ProjectionViewport,
        project_snapshot,
    };
    use scriptbots_core::{
        ActivationLayer, AgentData, AgentUid, BrainActivations, BrainInspection,
        BrainInspectionLimits, BrainInspectionSnapshot, BrainRunner, Generation, HydrologyField,
        HydrologyFlowDirection, HydrologyTile, HydrologyTileLayer, INPUT_SIZE, MapArtifact,
        MapArtifactMetadata, MapGeneratorKind, OUTPUT_SIZE, Position, ScriptBotsConfig,
        TerrainLayer, Velocity, bound_brain_inspection,
    };
    use std::{hint::black_box, time::Instant};

    fn world(persistence_interval: u32) -> WorldState {
        WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0x5eed_cafe),
            persistence_interval,
            ..ScriptBotsConfig::default()
        })
        .expect("deterministic test world")
    }

    fn snapshot_map_artifacts(world: &WorldState) -> (MapArtifact, MapArtifact, MapArtifact) {
        let base_terrain = world.terrain().clone();
        let width = base_terrain.width();
        let height = base_terrain.height();
        let cell_size = base_terrain.cell_size();
        let cell_count = base_terrain.tiles().len();
        let mut changed_tiles = base_terrain.tiles().to_vec();
        let first = changed_tiles
            .first_mut()
            .expect("snapshot map has at least one terrain tile");
        first.accent = if first.accent.to_bits() == 0.25_f32.to_bits() {
            0.75
        } else {
            0.25
        };
        let changed_terrain = TerrainLayer::from_tiles(width, height, cell_size, changed_tiles)
            .expect("changed snapshot terrain");

        let artifact = |terrain: TerrainLayer, changed: bool| {
            let mut hydrology_tiles = vec![
                HydrologyTile {
                    permeability: 0.4,
                    runoff_bias: 0.2,
                    basin_rank: 0.5,
                    channel_priority: 0.3,
                    swim_cost: 0.6,
                };
                cell_count
            ];
            let mut water_depth = vec![0.0; cell_count];
            if changed {
                hydrology_tiles[0].permeability = 0.8;
                water_depth[0] = 0.125;
            }
            MapArtifact::new(
                terrain,
                None,
                None,
                Some(
                    HydrologyTileLayer::new(width, height, hydrology_tiles)
                        .expect("snapshot hydrology tiles"),
                ),
                Some(
                    HydrologyField::new(
                        width,
                        height,
                        vec![HydrologyFlowDirection::None; cell_count],
                        vec![1.0; cell_count],
                        vec![0.5; cell_count],
                        vec![0; cell_count],
                        water_depth,
                    )
                    .expect("snapshot hydrology field"),
                ),
                MapArtifactMetadata {
                    generator: MapGeneratorKind::RuleBased,
                    tileset_id: if changed {
                        "snapshot-layer-b"
                    } else {
                        "snapshot-layer-a"
                    }
                    .to_owned(),
                    tileset_hash: u64::from(changed),
                    seed: 0x5a4f_4d41 + u64::from(changed),
                    width,
                    height,
                    attempt_count: 1,
                    succeeded_on: 1,
                    generated_at_epoch_ms: 0,
                },
            )
            .expect("snapshot map artifact")
        };

        let without_hydrology = MapArtifact::new(
            changed_terrain.clone(),
            None,
            None,
            None,
            None,
            MapArtifactMetadata {
                generator: MapGeneratorKind::RuleBased,
                tileset_id: "snapshot-layer-no-hydrology".to_owned(),
                tileset_hash: 2,
                seed: 0x5a4f_4d43,
                width,
                height,
                attempt_count: 1,
                succeeded_on: 1,
                generated_at_epoch_ms: 0,
            },
        )
        .expect("snapshot map without hydrology");

        (
            artifact(base_terrain, false),
            artifact(changed_terrain, true),
            without_hydrology,
        )
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
            snapshot_interval_ticks: 1,
            protocol_event_capacity: 256,
            scientific_event_capacity: 64,
            volatile_event_history_capacity: 512,
        }
    }

    fn host(paused: bool) -> (HostCore, LocalHostPort) {
        let core = HostCore::new(HostSessionId::new(7), world(0), options(paused))
            .expect("host construction");
        let port = core.local_port();
        (core, port)
    }

    fn envelope(id: u128, command: HostCommand) -> CommandEnvelope {
        CommandEnvelope::new(CommandId::new(id), command)
    }

    fn submit(port: &mut LocalHostPort, id: u128, command: HostCommand) -> CommandStatus {
        port.submit(envelope(id, command))
            .expect("local submission")
    }

    fn status(port: &mut LocalHostPort, id: u128) -> CommandStatus {
        port.command_status(CommandId::new(id))
            .expect("status lookup")
            .expect("status retained")
    }

    fn applied(status: &CommandStatus) -> AppliedCommand {
        match status.application() {
            ApplicationState::Applied(applied) => *applied,
            application => panic!("expected applied status, got {application:?}"),
        }
    }

    #[test]
    fn construction_and_first_drive_never_hide_a_warmup_tick() {
        for paused in [false, true] {
            let (mut core, _port) = host(paused);
            let initial = core.latest_snapshot();
            assert_eq!(initial.world.tick, 0);
            assert_eq!(initial.revisions.scientific, ScientificRevision::new(0));
            assert_eq!(initial.playback.paused, paused);

            let receipt = core
                .drive(ManualInstant::from_nanos(1_000))
                .expect("first manual boundary");
            assert_eq!(receipt.scientific_steps, 0);
            assert_eq!(core.world_tick(), Tick(0));
        }
    }

    #[test]
    fn snapshot_hub_keeps_independent_latest_only_cursors_without_a_backlog() {
        let (mut core, mut port) = host(true);
        let hub = core.snapshot_hub();
        let mut fast_a = hub.subscribe();
        let mut fast_b = hub.subscribe();
        let mut stalled = hub.subscribe();

        let initial_a = hub
            .poll_latest(&mut fast_a)
            .expect("first fast poll")
            .expect("initial publication");
        let initial_b = hub
            .poll_latest(&mut fast_b)
            .expect("second fast poll")
            .expect("same initial publication");
        let stalled_initial = hub
            .poll_latest(&mut stalled)
            .expect("stalled initial poll")
            .expect("stalled cursor starts current");
        let dropped_initial = {
            let mut dropped = hub.subscribe();
            hub.poll_latest(&mut dropped)
                .expect("dropped initial poll")
                .expect("dropped cursor starts current")
        };
        assert!(Arc::ptr_eq(&initial_a, &initial_b));
        assert!(Arc::ptr_eq(&initial_a, &stalled_initial));
        let initial_weak = Arc::downgrade(&initial_a);
        drop(initial_a);
        drop(initial_b);
        drop(stalled_initial);
        drop(dropped_initial);

        let mut newest = SnapshotRevision::new(1);
        for sequence in 1_u128..=3 {
            submit(&mut port, sequence, HostCommand::Step);
            core.drive(ManualInstant::from_nanos(
                u64::try_from(sequence).expect("small test sequence"),
            ))
            .expect("explicit step publication");
            let snapshot_a = hub
                .poll_latest(&mut fast_a)
                .expect("fast A poll")
                .expect("fast A sees every publication");
            let snapshot_b = hub
                .poll_latest(&mut fast_b)
                .expect("fast B poll")
                .expect("fast B sees every publication");
            assert!(Arc::ptr_eq(&snapshot_a, &snapshot_b));
            newest = snapshot_a.revision;
        }

        assert!(
            initial_weak.upgrade().is_none(),
            "a scalar stalled cursor must not retain an obsolete snapshot Arc"
        );
        let stalled_latest = hub
            .poll_latest(&mut stalled)
            .expect("stalled latest poll")
            .expect("stalled cursor jumps to newest");
        assert_eq!(stalled_latest.revision, newest);
        assert_eq!(stalled.skipped_revisions(), 2);
        assert!(
            hub.poll_latest(&mut stalled)
                .expect("idempotent stalled repoll")
                .is_none()
        );

        let mut reconnecting = hub.resume_after(SnapshotRevision::new(2));
        let reconnected = hub
            .poll_latest(&mut reconnecting)
            .expect("reconnected poll")
            .expect("reconnected cursor receives newest");
        assert_eq!(reconnected.revision, newest);
        assert_eq!(reconnecting.skipped_revisions(), 1);
    }

    #[test]
    fn snapshots_publish_exact_completed_summary_without_persistence_cadence() {
        let mut core = HostCore::new(HostSessionId::new(71), world(3), options(true))
            .expect("host with sparse persistence cadence");
        let mut port = core.local_port();
        assert!(core.latest_snapshot().completed_summary.is_none());

        for tick in 1_u128..=2 {
            submit(&mut port, tick, HostCommand::Step);
            core.drive(ManualInstant::from_nanos(
                u64::try_from(tick).expect("small tick"),
            ))
            .expect("completed explicit step");
            let snapshot = core.latest_snapshot();
            let summary = snapshot
                .completed_summary
                .as_ref()
                .expect("completed summary is retained independently of persistence");
            let expected = core
                .world
                .history()
                .next_back()
                .expect("current summary remains in history");
            assert_eq!(summary, expected);
            assert_eq!(summary.tick, Tick(u64::try_from(tick).expect("small tick")));
            assert_eq!(snapshot.world.tick, summary.tick.0);
            assert_eq!(snapshot.world.summary.agent_count, summary.agent_count);
            assert_eq!(snapshot.world.summary.births, summary.births);
            assert_eq!(snapshot.world.summary.deaths, summary.deaths);
        }
    }

    #[test]
    fn construction_preserves_an_existing_current_tick_summary_exactly() {
        let mut prestepped = world(0);
        prestepped.step().expect("pre-host scientific tick");
        let expected = prestepped
            .history()
            .next_back()
            .expect("pre-host summary")
            .clone();

        let core = HostCore::new(HostSessionId::new(76), prestepped, options(true))
            .expect("host from prestepped world");
        let initial = core.latest_snapshot();
        assert_eq!(initial.world.tick, expected.tick.0);
        assert_eq!(initial.completed_summary.as_ref(), Some(&expected));
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one acceptance scenario makes every terrain, food, and hydrology Arc/revision transition explicit"
    )]
    fn layer_arcs_and_revisions_change_together_on_exact_content_bits() {
        let mut config = ScriptBotsConfig {
            rng_seed: Some(0x051a_71c5),
            initial_food: 0.5,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        };
        let world = WorldState::new(config.clone()).expect("layer test world");
        let (base_map, changed_map, map_without_hydrology) = snapshot_map_artifacts(&world);
        let mut core =
            HostCore::new(HostSessionId::new(72), world, options(true)).expect("layer test host");
        let mut port = core.local_port();
        let initial = core.latest_snapshot();
        assert!(initial.layers.hydrology.is_none());
        assert_eq!(initial.layers.revisions.hydrology, LayerRevision::new(0));

        submit(&mut port, 1, HostCommand::Pause);
        core.drive(ManualInstant::from_nanos(1))
            .expect("control-only publication");
        let control_only = core.latest_snapshot();
        assert!(Arc::ptr_eq(
            &initial.layers.terrain,
            &control_only.layers.terrain
        ));
        assert!(Arc::ptr_eq(&initial.layers.food, &control_only.layers.food));
        assert_eq!(initial.layers.revisions, control_only.layers.revisions);
        assert_eq!(control_only.build.bulk_allocations, 0);

        core.world
            .apply_map_artifact(&base_map)
            .expect("install initial hydrology");
        core.publish_snapshot().expect("hydrology-add publication");
        let hydrology_added = core.latest_snapshot();
        assert!(Arc::ptr_eq(
            &control_only.layers.terrain,
            &hydrology_added.layers.terrain
        ));
        assert!(Arc::ptr_eq(
            &control_only.layers.food,
            &hydrology_added.layers.food
        ));
        assert_eq!(
            hydrology_added.layers.revisions.terrain,
            control_only.layers.revisions.terrain
        );
        assert_eq!(
            hydrology_added.layers.revisions.food,
            control_only.layers.revisions.food
        );
        assert_eq!(
            hydrology_added.layers.revisions.hydrology,
            control_only
                .layers
                .revisions
                .hydrology
                .checked_next()
                .expect("hydrology-add revision headroom")
        );
        let initial_hydrology = hydrology_added
            .layers
            .hydrology
            .as_ref()
            .expect("hydrology payload added");

        config.food_max = 0.25;
        config.initial_food = 0.25;
        config.food_respawn_amount = 0.25;
        submit(&mut port, 2, HostCommand::UpdateConfig(Box::new(config)));
        core.drive(ManualInstant::from_nanos(2))
            .expect("food-changing configuration publication");
        let food_changed = core.latest_snapshot();
        assert!(Arc::ptr_eq(
            &hydrology_added.layers.terrain,
            &food_changed.layers.terrain
        ));
        assert_eq!(
            hydrology_added.layers.revisions.terrain,
            food_changed.layers.revisions.terrain
        );
        assert!(!Arc::ptr_eq(
            &hydrology_added.layers.food,
            &food_changed.layers.food
        ));
        assert!(Arc::ptr_eq(
            initial_hydrology,
            food_changed
                .layers
                .hydrology
                .as_ref()
                .expect("food-only change retains hydrology")
        ));
        assert_eq!(
            food_changed.layers.revisions.food,
            hydrology_added
                .layers
                .revisions
                .food
                .checked_next()
                .expect("food revision headroom")
        );
        assert_eq!(food_changed.build.bulk_allocations, 1);
        assert!(food_changed.build.newly_allocated_capacity_bytes > 0);
        assert!(food_changed.build.reused_layer_capacity_bytes > 0);
        assert!(!same_f32_slice(&[0.0], &[-0.0]));

        core.world
            .apply_map_artifact(&changed_map)
            .expect("change terrain and hydrology");
        core.publish_snapshot()
            .expect("terrain and hydrology publication");
        let terrain_and_hydrology_changed = core.latest_snapshot();
        assert!(!Arc::ptr_eq(
            &food_changed.layers.terrain,
            &terrain_and_hydrology_changed.layers.terrain
        ));
        assert!(Arc::ptr_eq(
            &food_changed.layers.food,
            &terrain_and_hydrology_changed.layers.food
        ));
        assert!(!Arc::ptr_eq(
            food_changed
                .layers
                .hydrology
                .as_ref()
                .expect("old hydrology"),
            terrain_and_hydrology_changed
                .layers
                .hydrology
                .as_ref()
                .expect("changed hydrology")
        ));
        let changed_hydrology = terrain_and_hydrology_changed
            .layers
            .hydrology
            .as_ref()
            .expect("changed hydrology payload");
        assert_eq!(
            changed_hydrology.tiles[0].permeability.to_bits(),
            0.8_f32.to_bits()
        );
        assert_eq!(
            changed_hydrology.water_depth[0].to_bits(),
            0.125_f32.to_bits()
        );
        assert_eq!(
            terrain_and_hydrology_changed.layers.revisions.terrain,
            food_changed
                .layers
                .revisions
                .terrain
                .checked_next()
                .expect("terrain revision headroom")
        );
        assert_eq!(
            terrain_and_hydrology_changed.layers.revisions.hydrology,
            food_changed
                .layers
                .revisions
                .hydrology
                .checked_next()
                .expect("hydrology revision headroom")
        );

        core.world
            .apply_map_artifact(&map_without_hydrology)
            .expect("remove hydrology");
        core.publish_snapshot()
            .expect("hydrology-removal publication");
        let hydrology_removed = core.latest_snapshot();
        assert!(Arc::ptr_eq(
            &terrain_and_hydrology_changed.layers.terrain,
            &hydrology_removed.layers.terrain
        ));
        assert!(Arc::ptr_eq(
            &terrain_and_hydrology_changed.layers.food,
            &hydrology_removed.layers.food
        ));
        assert!(hydrology_removed.layers.hydrology.is_none());
        assert_eq!(
            hydrology_removed.layers.revisions.hydrology,
            terrain_and_hydrology_changed
                .layers
                .revisions
                .hydrology
                .checked_next()
                .expect("hydrology-removal revision headroom")
        );

        let encoded =
            serde_json::to_vec(hydrology_removed.as_ref()).expect("snapshot serialization");
        let decoded: RenderSnapshot =
            serde_json::from_slice(&encoded).expect("snapshot deserialization");
        assert_eq!(decoded, *hydrology_removed);
    }

    #[test]
    fn snapshot_stride_and_subscriber_load_cannot_change_science() {
        let config = ScriptBotsConfig {
            rng_seed: Some(0x5a4f_5707),
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        };
        let mut every_options = options(false);
        every_options.snapshot_interval_ticks = 1;
        let mut sparse_options = every_options;
        sparse_options.snapshot_interval_ticks = 3;
        let mut every = HostCore::new(
            HostSessionId::new(73),
            WorldState::new(config.clone()).expect("every world"),
            every_options,
        )
        .expect("every host");
        let mut sparse = HostCore::new(
            HostSessionId::new(74),
            WorldState::new(config).expect("sparse world"),
            sparse_options,
        )
        .expect("sparse host");
        let busy_hub = every.snapshot_hub();
        let mut readers = (0..128).map(|_| busy_hub.subscribe()).collect::<Vec<_>>();
        for reader in &mut readers {
            assert!(
                busy_hub
                    .poll_latest(reader)
                    .expect("initial busy-reader poll")
                    .is_some()
            );
        }

        every
            .drive(ManualInstant::from_nanos(0))
            .expect("every epoch");
        sparse
            .drive(ManualInstant::from_nanos(0))
            .expect("sparse epoch");
        let mut every_publications = 0usize;
        let mut sparse_publications = 0usize;
        for tick in 1_u64..=6 {
            every_publications += every
                .drive(ManualInstant::from_nanos(tick * 10))
                .expect("every cadence drive")
                .snapshots_published;
            for reader in &mut readers {
                let _ = busy_hub
                    .poll_latest(reader)
                    .expect("busy-reader latest poll");
            }
            sparse_publications += sparse
                .drive(ManualInstant::from_nanos(tick * 10))
                .expect("sparse cadence drive")
                .snapshots_published;
            assert_eq!(every.world_tick(), sparse.world_tick());
            assert_eq!(
                every.scientific_digest_v1().expect("every digest"),
                sparse.scientific_digest_v1().expect("sparse digest")
            );
        }
        assert_eq!(every.world_tick(), Tick(6));
        assert_eq!(every_publications, 6);
        assert_eq!(sparse_publications, 2);
    }

    fn projection_matrix_request(client: u16) -> ProjectionRequest {
        let uid = u64::from(client) + 1;
        let next_uid = u64::from((client + 1) % 128) + 1;
        ProjectionRequest {
            client_id: ProjectionClientId::new(u64::from(client) + 1),
            viewport: ProjectionViewport {
                width: 80,
                height: 45,
            },
            camera: ProjectionCamera {
                center: [
                    f32::from(client % 32) * 25.0,
                    f32::from(client / 32) * 200.0,
                ],
                zoom: 1.0 + f32::from(client % 4) * 0.5,
            },
            selection: ProjectionSelection {
                focused: Some(AgentUid(uid)),
                selected: vec![AgentUid(next_uid), AgentUid(uid)],
            },
            detail: match client % 3 {
                0 => ProjectionDetail::Minimal,
                1 => ProjectionDetail::Vitals,
                _ => ProjectionDetail::Kinematics,
            },
            chart_window: u32::from(client % 64) + 1,
            chart_points: client % 16 + 1,
            top_k: client % 32 + 1,
            ranking: match client % 4 {
                0 => ProjectionRanking::Energy,
                1 => ProjectionRanking::Health,
                2 => ProjectionRanking::Age,
                _ => ProjectionRanking::Generation,
            },
        }
    }

    #[test]
    fn one_hundred_twenty_eight_projection_clients_are_isolated_and_science_invariant() {
        let core = HostCore::new(
            HostSessionId::new(76),
            snapshot_measurement_world(128),
            options(true),
        )
        .expect("projection matrix host");
        let source = core.latest_snapshot();
        let source_before = source.as_ref().clone();
        let digest_before = core
            .scientific_digest_v1()
            .expect("projection matrix digest");
        let revisions_before = source.revisions;
        let mut broker = ProjectionBroker::with_byte_capacity(16, 8 * 1024 * 1024)
            .expect("bounded projection matrix broker");
        let mut first = None;
        let mut second = None;

        for client in 0_u16..128 {
            let projection = broker
                .project(
                    &source,
                    &projection_matrix_request(client),
                    ProjectionLimits::default(),
                )
                .unwrap_or_else(|error| panic!("projection client {client} failed: {error}"));
            assert_eq!(projection.source.snapshot, source.revision);
            assert_eq!(projection.source.host, revisions_before);
            if client == 0 {
                first = Some(projection.as_ref().clone());
            } else if client == 1 {
                second = Some(projection.as_ref().clone());
            }
        }

        let first = first.expect("first client projection");
        assert_ne!(first, second.expect("second client projection"));
        let rebuilt_first = broker
            .project(
                &source,
                &projection_matrix_request(0),
                ProjectionLimits::default(),
            )
            .expect("deterministic first-client rebuild");
        assert_eq!(rebuilt_first.as_ref(), &first);
        assert!(broker.len() <= 16);
        assert!(broker.retained_output_capacity_bytes() <= broker.byte_capacity());
        assert!(broker.evictions() > 0);
        assert_eq!(source.as_ref(), &source_before);
        assert_eq!(core.latest_snapshot().revisions, revisions_before);
        assert_eq!(
            core.scientific_digest_v1()
                .expect("post-projection matrix digest"),
            digest_before
        );
    }

    const PROJECTION_BRAIN_DIGEST: u64 = 0x5255_4e54_494d_4501;

    #[derive(Debug)]
    struct ProjectionBrain;

    impl BrainRunner for ProjectionBrain {
        fn kind(&self) -> &'static str {
            "runtime.projection"
        }

        fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
            [0.0; OUTPUT_SIZE]
        }

        fn inspect(
            &self,
            request: BrainInspection,
        ) -> Result<Option<BrainInspectionSnapshot>, BrainInspectionError> {
            let BrainInspection::Activations(limits) = request;
            bound_brain_inspection(
                self.kind(),
                BrainActivations {
                    layers: vec![ActivationLayer {
                        name: "runtime".to_owned(),
                        width: 2,
                        height: 1,
                        values: vec![0.25, 0.75],
                    }],
                    connections: Vec::new(),
                    truncated: false,
                },
                2,
                limits,
            )
            .map(Some)
        }

        fn state_digest(&self) -> Option<u64> {
            Some(PROJECTION_BRAIN_DIGEST)
        }
    }

    fn projection_brain_host() -> HostCore {
        let mut world = snapshot_measurement_world(2);
        let family = world
            .brain_registry_mut()
            .expect("runtime projection registry mutation")
            .register_with_state_digest("runtime.projection", PROJECTION_BRAIN_DIGEST, |_rng| {
                Ok(Box::new(ProjectionBrain))
            });
        let agent_ids: Vec<_> = world.agents().iter_handles().collect();
        for agent_id in agent_ids {
            world
                .bind_agent_brain(agent_id, family)
                .expect("bind runtime projection brain");
        }
        HostCore::new(HostSessionId::new(0x0042_5241_494e), world, options(true))
            .expect("brain projection host")
    }

    #[test]
    fn selected_brain_projection_is_client_isolated_revisioned_and_science_neutral() {
        let core = projection_brain_host();
        let source = core.latest_snapshot();
        let first_uid = source.world.agents[0].uid;
        let second_uid = source.world.agents[1].uid;
        let digest_before = core
            .scientific_digest_v1()
            .expect("pre-inspection scientific digest");

        let first = core
            .inspect_brains(&BrainProjectionRequest::focused(
                ProjectionClientId::new(1),
                ProjectionRequestRevision::new(7),
                first_uid,
            ))
            .expect("first client brain projection");
        let second = core
            .inspect_brains(&BrainProjectionRequest::focused(
                ProjectionClientId::new(2),
                ProjectionRequestRevision::new(11),
                second_uid,
            ))
            .expect("second client brain projection");

        assert!(first.source.matches_snapshot(&source));
        assert!(second.source.matches_snapshot(&source));
        assert_eq!(first.request.client_id, ProjectionClientId::new(1));
        assert_eq!(first.request.revision, ProjectionRequestRevision::new(7));
        assert_eq!(first.inspection.client_id.get(), 1);
        assert_eq!(first.inspection.request_revision.get(), 7);
        assert_eq!(second.request.client_id, ProjectionClientId::new(2));
        assert_eq!(second.request.revision, ProjectionRequestRevision::new(11));
        assert_eq!(second.inspection.client_id.get(), 2);
        assert_eq!(second.inspection.request_revision.get(), 11);
        assert_ne!(first.request.targets, second.request.targets);
        let first_ready = first
            .inspection
            .ready_for(first_uid)
            .expect("first runtime projection payload");
        assert_eq!(first_ready.inspection.build.source_values, 2);
        assert_eq!(first_ready.inspection.build.retained_values, 2);
        assert!(!first_ready.inspection.build.truncated);

        let clipped = core
            .inspect_brains(&BrainProjectionRequest {
                client_id: ProjectionClientId::new(3),
                revision: ProjectionRequestRevision::new(12),
                targets: vec![first_uid],
                limits: BrainInspectionLimits::tightened(1, 1, 1, 0, 128, 2),
            })
            .expect("clipped runtime projection");
        let clipped_ready = clipped
            .inspection
            .ready_for(first_uid)
            .expect("clipped ready payload");
        assert!(clipped_ready.inspection.build.truncated);
        assert!(clipped_ready.inspection.activations.truncated);
        assert!(clipped_ready.inspection.build.retained_payload_bytes <= 128);
        assert_eq!(
            core.scientific_digest_v1()
                .expect("post-inspection scientific digest"),
            digest_before
        );

        let oversized = BrainProjectionRequest {
            client_id: ProjectionClientId::new(3),
            revision: ProjectionRequestRevision::new(1),
            targets: vec![first_uid; ACTIVATION_CAPTURE_BUDGET + 1],
            limits: scriptbots_core::BrainInspectionLimits::hard(),
        };
        assert!(matches!(
            core.inspect_brains(&oversized),
            Err(BrainInspectionError::TargetLimitExceeded { .. })
        ));
        let oversized_json = serde_json::to_vec(&oversized).expect("encode oversized projection");
        assert!(serde_json::from_slice::<BrainProjectionRequest>(&oversized_json).is_err());
    }

    #[test]
    fn brain_projection_source_rejects_an_unpublished_owner_advance() {
        let mut sparse_options = options(false);
        sparse_options.snapshot_interval_ticks = 3;
        let mut core = HostCore::new(
            HostSessionId::new(0x4252_4149_4e02),
            snapshot_measurement_world(1),
            sparse_options,
        )
        .expect("sparse brain projection host");
        core.drive(ManualInstant::from_nanos(0))
            .expect("establish sparse-host clock origin");
        core.drive(ManualInstant::from_nanos(10))
            .expect("advance without snapshot publication");
        let published = core.latest_snapshot();
        let projection = core
            .inspect_brains(&BrainProjectionRequest {
                client_id: ProjectionClientId::new(9),
                revision: ProjectionRequestRevision::new(1),
                targets: Vec::new(),
                limits: BrainInspectionLimits::hard(),
            })
            .expect("empty sparse-source inspection");
        assert_ne!(projection.source.inspected_host, published.revisions);
        assert!(!projection.source.matches_snapshot(&published));
    }

    fn snapshot_measurement_config() -> ScriptBotsConfig {
        ScriptBotsConfig {
            world_width: 800,
            world_height: 800,
            food_cell_size: 20,
            rng_seed: Some(0x5eed_ba5e),
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            population_crossover_chance: 0.0,
            reproduction_attempt_interval: 1,
            reproduction_attempt_chance: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            metabolism_ramp_floor: 0.0,
            metabolism_ramp_rate: 0.0,
            metabolism_boost_penalty: 0.0,
            temperature_discomfort_rate: 0.0,
            aging_health_decay_rate: 0.0,
            aging_health_decay_max: 0.0,
            aging_energy_penalty_rate: 0.0,
            spike_damage: 0.0,
            spike_energy_cost: 0.0,
            food_max: 2.0,
            food_growth_rate: 0.02,
            food_decay_rate: 0.0,
            food_diffusion_rate: 0.1,
            food_waste_rate: 0.0,
            history_capacity: 1,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "bounded synthetic world coordinates intentionally narrow into the f32 simulation domain"
    )]
    fn populate_snapshot_measurement_agents(world: &mut WorldState, agent_count: usize) {
        for ordinal in 0..agent_count {
            let x = (ordinal % 800) as f32;
            let y = ((ordinal * 37) % 800) as f32;
            world
                .try_spawn_agent(AgentData::new(
                    Position::new(x, y),
                    Velocity::default(),
                    0.0,
                    1.0,
                    [0.5, 0.5, 0.5],
                    0.0,
                    false,
                    0,
                    Generation(0),
                ))
                .unwrap_or_else(|error| panic!("snapshot agent {ordinal} failed: {error}"));
        }
    }

    fn snapshot_measurement_world(agent_count: usize) -> WorldState {
        let mut world =
            WorldState::new(snapshot_measurement_config()).expect("snapshot measurement world");
        populate_snapshot_measurement_agents(&mut world, agent_count);
        world
    }

    fn projection_measurement_world(agent_count: usize) -> WorldState {
        const HISTORY_SAMPLES: usize = 64;
        let mut config = snapshot_measurement_config();
        config.history_capacity = HISTORY_SAMPLES;
        let mut world = WorldState::new(config).expect("projection measurement world");
        populate_snapshot_measurement_agents(&mut world, agent_count);
        for _ in 0..HISTORY_SAMPLES {
            world.step().expect("projection measurement history step");
        }
        assert_eq!(world.history().count(), HISTORY_SAMPLES);
        world
    }

    fn measured_snapshot_publication(core: &mut HostCore) -> (u64, SnapshotBuildStats) {
        let started = Instant::now();
        core.publish_snapshot()
            .expect("measured full snapshot publication");
        let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let snapshot = core.latest_snapshot();
        black_box(&snapshot);
        (elapsed, snapshot.build)
    }

    fn nearest_rank(samples: &[u64], percentile: usize) -> u64 {
        let mut ordered = samples.to_vec();
        ordered.sort_unstable();
        let rank = ordered
            .len()
            .saturating_mul(percentile)
            .div_ceil(100)
            .saturating_sub(1)
            .min(ordered.len().saturating_sub(1));
        ordered[rank]
    }

    fn projection_measurement_request(snapshot: &RenderSnapshot) -> ProjectionRequest {
        let focused = snapshot
            .world
            .agents
            .first()
            .map(|agent| agent.uid)
            .expect("projection measurement source has agents");
        let selected = snapshot
            .world
            .agents
            .get(1)
            .map_or_else(|| vec![focused], |agent| vec![focused, agent.uid]);
        ProjectionRequest {
            client_id: ProjectionClientId::new(0x004d_4541_5355_5245),
            viewport: ProjectionViewport {
                width: 160,
                height: 90,
            },
            camera: ProjectionCamera {
                center: [400.0, 400.0],
                zoom: 1.25,
            },
            selection: ProjectionSelection {
                focused: Some(focused),
                selected,
            },
            detail: ProjectionDetail::Kinematics,
            chart_window: 64,
            chart_points: 32,
            top_k: 32,
            ranking: ProjectionRanking::Energy,
        }
    }

    fn moving_projection_request(base: &ProjectionRequest, ordinal: usize) -> ProjectionRequest {
        let ordinal = u16::try_from(ordinal).expect("bounded projection sample ordinal");
        let mut request = base.clone();
        request.camera.center = [
            (f32::from(ordinal) * 17.0).rem_euclid(800.0),
            (f32::from(ordinal) * 31.0).rem_euclid(800.0),
        ];
        request
    }

    #[test]
    #[ignore = "DSR-only reference-hardware snapshot measurement"]
    #[allow(
        clippy::too_many_lines,
        reason = "one DSR evidence record keeps warmups, raw samples, digests, capacity accounting, and both reference scales coherent"
    )]
    fn measure_snapshot_builds_at_1k_and_10k_agents() {
        const WARMUPS: usize = 20;
        const SAMPLES: usize = 200;
        for agent_count in [1_000, 10_000] {
            let mut world = snapshot_measurement_world(agent_count);
            let (base_map, changed_map, _) = snapshot_map_artifacts(&world);
            world
                .apply_map_artifact(&base_map)
                .expect("install measured base map");
            let scenario_config = world.config().clone();
            let session_id = HostSessionId::new(
                u64::try_from(agent_count).expect("measured agent count fits session id"),
            );
            let mut core =
                HostCore::new(session_id, world, options(true)).expect("snapshot measurement host");
            let initial_stats = core.latest_snapshot().build;
            assert_eq!(initial_stats.dynamic_agent_count, agent_count);
            let initial_digest = core
                .scientific_digest_v1()
                .expect("initial measurement digest");

            for _ in 0..WARMUPS {
                black_box(measured_snapshot_publication(&mut core));
            }

            let mut steady_samples = Vec::with_capacity(SAMPLES);
            let mut steady_stats = SnapshotBuildStats::default();
            for _ in 0..SAMPLES {
                let (elapsed, stats) = measured_snapshot_publication(&mut core);
                steady_stats = stats;
                steady_samples.push(elapsed);
            }
            assert_eq!(steady_stats.dynamic_agent_count, agent_count);
            assert_eq!(steady_stats.bulk_allocations, 1);
            assert_eq!(
                core.scientific_digest_v1()
                    .expect("post-steady measurement digest"),
                initial_digest,
                "steady snapshot publication must not alter science"
            );

            for sample in 0..WARMUPS {
                let map = if sample.is_multiple_of(2) {
                    &changed_map
                } else {
                    &base_map
                };
                core.world
                    .apply_map_artifact(map)
                    .expect("changed-layer warmup map");
                black_box(measured_snapshot_publication(&mut core));
            }
            let mut changed_layer_samples = Vec::with_capacity(SAMPLES);
            let mut changed_layer_stats = SnapshotBuildStats::default();
            for sample in 0..SAMPLES {
                let map = if sample.is_multiple_of(2) {
                    &changed_map
                } else {
                    &base_map
                };
                core.world
                    .apply_map_artifact(map)
                    .expect("changed-layer measurement map");
                let (elapsed, stats) = measured_snapshot_publication(&mut core);
                changed_layer_stats = stats;
                changed_layer_samples.push(elapsed);
            }
            assert_eq!(changed_layer_stats.dynamic_agent_count, agent_count);
            assert!(changed_layer_stats.bulk_allocations > steady_stats.bulk_allocations);

            core.world
                .apply_map_artifact(&changed_map)
                .expect("publication-invariance probe map");
            let digest_before_probe = core
                .scientific_digest_v1()
                .expect("pre-publication probe digest");
            black_box(measured_snapshot_publication(&mut core));
            let digest_after_probe = core
                .scientific_digest_v1()
                .expect("post-publication probe digest");
            assert_eq!(digest_before_probe, digest_after_probe);

            let steady_p50_ns = nearest_rank(&steady_samples, 50);
            let steady_p95_ns = nearest_rank(&steady_samples, 95);
            let changed_layer_p50_ns = nearest_rank(&changed_layer_samples, 50);
            let changed_layer_p95_ns = nearest_rank(&changed_layer_samples, 95);
            let budget_p95_ns = if agent_count == 1_000 {
                4_000_000
            } else {
                16_000_000
            };
            let evidence = serde_json::json!({
                "schema": "scriptbots.render_snapshot.measurement.v1",
                "scenario_contract": "standard-800x800-food20-seed-0x5eedba5e",
                "agent_count": agent_count,
                "warmups_per_case": WARMUPS,
                "samples_per_case": SAMPLES,
                "config": scenario_config,
                "initial_scientific_digest": initial_digest.overall,
                "changed_layer_scientific_digest": digest_after_probe.overall,
                "budget_p95_ns": budget_p95_ns,
                "steady": {
                    "raw_ns": steady_samples,
                    "p50_ns": steady_p50_ns,
                    "p95_ns": steady_p95_ns,
                    "bulk_vector_capacity_stats": steady_stats,
                },
                "changed_layer": {
                    "raw_ns": changed_layer_samples,
                    "p50_ns": changed_layer_p50_ns,
                    "p95_ns": changed_layer_p95_ns,
                    "bulk_vector_capacity_stats": changed_layer_stats,
                },
                "initial_bulk_vector_capacity_stats": initial_stats,
            });
            eprintln!(
                "{}",
                serde_json::to_string(&evidence).expect("serialize snapshot measurement evidence")
            );
            assert!(
                steady_p95_ns < budget_p95_ns,
                "steady snapshot p95 {steady_p95_ns}ns exceeded {budget_p95_ns}ns"
            );
            assert!(
                changed_layer_p95_ns < budget_p95_ns,
                "changed-layer snapshot p95 {changed_layer_p95_ns}ns exceeded {budget_p95_ns}ns"
            );
        }
    }

    #[test]
    #[ignore = "DSR-only reference-hardware per-client projection measurement"]
    #[allow(
        clippy::too_many_lines,
        reason = "one emitted DSR evidence record keeps raw cold, cached, moving-camera, and 128-client fanout samples with their exact structural accounting"
    )]
    fn measure_projections_at_1k_and_10k_agents() {
        const WARMUPS: usize = 20;
        const SAMPLES: usize = 200;
        const FANOUT_CLIENTS: u16 = 128;
        const SINGLE_CACHE_BYTES: usize = 64 * 1024 * 1024;
        const FANOUT_CACHE_BYTES: usize = 512 * 1024 * 1024;
        let limits = ProjectionLimits::default();

        for agent_count in [1_000, 10_000] {
            let core = HostCore::new(
                HostSessionId::new(
                    u64::try_from(agent_count).expect("projection agent count fits session id"),
                ),
                projection_measurement_world(agent_count),
                options(true),
            )
            .expect("projection measurement host");
            let source = core.latest_snapshot();
            assert_eq!(source.world.agents.len(), agent_count);
            assert_eq!(source.summary_history.len(), 64);
            let digest_before = core
                .scientific_digest_v1()
                .expect("pre-projection measurement digest");
            let request = projection_measurement_request(&source);

            for _ in 0..WARMUPS {
                black_box(
                    project_snapshot(&source, &request, limits).expect("cold projection warmup"),
                );
            }
            let mut cold_samples = Vec::with_capacity(SAMPLES);
            let mut cold_stats = crate::ProjectionBuildStats::default();
            for _ in 0..SAMPLES {
                let started = Instant::now();
                let projection =
                    project_snapshot(&source, &request, limits).expect("measured cold projection");
                cold_samples.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
                cold_stats = projection.build;
                black_box(projection);
            }

            let mut warm_broker = ProjectionBroker::with_byte_capacity(4, SINGLE_CACHE_BYTES)
                .expect("warm projection broker");
            for _ in 0..WARMUPS {
                black_box(
                    warm_broker
                        .project(&source, &request, limits)
                        .expect("warm cached projection warmup"),
                );
            }
            let mut warm_samples = Vec::with_capacity(SAMPLES);
            let mut warm_stats = crate::ProjectionBuildStats::default();
            for _ in 0..SAMPLES {
                let started = Instant::now();
                let projection = warm_broker
                    .project(&source, &request, limits)
                    .expect("measured warm cached projection");
                warm_samples.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
                warm_stats = projection.build;
                black_box(projection);
            }
            assert_eq!(warm_broker.misses(), 1);
            assert_eq!(
                warm_broker.hits(),
                u64::try_from(WARMUPS + SAMPLES - 1).expect("bounded warm hit count")
            );

            let mut moving_broker = ProjectionBroker::with_byte_capacity(16, SINGLE_CACHE_BYTES)
                .expect("moving projection broker");
            for sample in 0..WARMUPS {
                let moving = moving_projection_request(&request, sample);
                black_box(
                    moving_broker
                        .project(&source, &moving, limits)
                        .expect("moving-camera projection warmup"),
                );
            }
            let mut moving_samples = Vec::with_capacity(SAMPLES);
            let mut moving_stats = crate::ProjectionBuildStats::default();
            for sample in 0..SAMPLES {
                let moving = moving_projection_request(&request, WARMUPS + sample);
                let started = Instant::now();
                let projection = moving_broker
                    .project(&source, &moving, limits)
                    .expect("measured moving-camera projection");
                moving_samples
                    .push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
                moving_stats = projection.build;
                black_box(projection);
            }
            assert_eq!(
                moving_broker.misses(),
                u64::try_from(WARMUPS + SAMPLES).expect("bounded moving miss count")
            );

            let mut fanout_broker = ProjectionBroker::with_byte_capacity(
                usize::from(FANOUT_CLIENTS),
                FANOUT_CACHE_BYTES,
            )
            .expect("128-client projection broker");
            let fanout_started = Instant::now();
            let mut fanout_agents_examined = 0usize;
            let mut fanout_visible_agents = 0usize;
            let mut fanout_canvas_cells = 0usize;
            let mut fanout_top_k_peak = 0usize;
            let mut fanout_chart_points = 0usize;
            let mut fanout_output_capacity_bytes = 0usize;
            for client in 0..FANOUT_CLIENTS {
                let projection = fanout_broker
                    .project(&source, &projection_matrix_request(client), limits)
                    .unwrap_or_else(|error| {
                        panic!("cold fanout projection {client} failed: {error}")
                    });
                fanout_agents_examined =
                    fanout_agents_examined.saturating_add(projection.build.agents_examined);
                fanout_visible_agents =
                    fanout_visible_agents.saturating_add(projection.build.visible_agents);
                fanout_canvas_cells =
                    fanout_canvas_cells.saturating_add(projection.build.canvas_cells);
                fanout_top_k_peak = fanout_top_k_peak.saturating_add(projection.build.top_k_peak);
                fanout_chart_points =
                    fanout_chart_points.saturating_add(projection.build.chart_points_emitted);
                fanout_output_capacity_bytes = fanout_output_capacity_bytes
                    .saturating_add(projection.build.output_capacity_bytes);
                black_box(projection);
            }
            let fanout_cold_ns =
                u64::try_from(fanout_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            assert_eq!(fanout_broker.len(), usize::from(FANOUT_CLIENTS));
            assert_eq!(fanout_broker.misses(), u64::from(FANOUT_CLIENTS));
            assert_eq!(
                fanout_broker.retained_output_capacity_bytes(),
                fanout_output_capacity_bytes
            );
            assert!(fanout_broker.retained_output_capacity_bytes() <= FANOUT_CACHE_BYTES);

            for _ in 0..WARMUPS {
                for client in 0..FANOUT_CLIENTS {
                    black_box(
                        fanout_broker
                            .project(&source, &projection_matrix_request(client), limits)
                            .expect("warm fanout projection"),
                    );
                }
            }
            let mut fanout_warm_samples = Vec::with_capacity(SAMPLES);
            for _ in 0..SAMPLES {
                let started = Instant::now();
                for client in 0..FANOUT_CLIENTS {
                    black_box(
                        fanout_broker
                            .project(&source, &projection_matrix_request(client), limits)
                            .expect("measured warm fanout projection"),
                    );
                }
                fanout_warm_samples
                    .push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
            }

            assert!(cold_stats.chart_samples_examined > 0);
            assert!(cold_stats.chart_points_emitted > 0);
            assert_eq!(cold_stats.top_k_peak, usize::from(request.top_k));
            assert_eq!(cold_stats.canvas_cells, 160 * 90);
            assert_eq!(warm_stats, cold_stats);
            assert_eq!(moving_stats.agents_examined, agent_count);
            assert_eq!(moving_stats.top_k_peak, usize::from(request.top_k));
            assert_eq!(moving_stats.canvas_cells, cold_stats.canvas_cells);
            assert!(cold_stats.output_capacity_bytes > 0);

            let cold_p50_ns = nearest_rank(&cold_samples, 50);
            let cold_p95_ns = nearest_rank(&cold_samples, 95);
            let warm_p50_ns = nearest_rank(&warm_samples, 50);
            let warm_p95_ns = nearest_rank(&warm_samples, 95);
            let moving_p50_ns = nearest_rank(&moving_samples, 50);
            let moving_p95_ns = nearest_rank(&moving_samples, 95);
            let fanout_warm_p50_ns = nearest_rank(&fanout_warm_samples, 50);
            let fanout_warm_p95_ns = nearest_rank(&fanout_warm_samples, 95);
            let (
                cold_budget_p95_ns,
                warm_budget_p95_ns,
                moving_budget_p95_ns,
                fanout_cold_budget_ns,
                fanout_warm_budget_p95_ns,
            ) = if agent_count == 1_000 {
                (16_000_000, 500_000, 20_000_000, 250_000_000, 20_000_000)
            } else {
                (
                    80_000_000,
                    1_000_000,
                    100_000_000,
                    3_000_000_000,
                    25_000_000,
                )
            };
            let digest_after = core
                .scientific_digest_v1()
                .expect("post-projection measurement digest");
            assert_eq!(digest_after, digest_before);

            let evidence = serde_json::json!({
                "schema": "scriptbots.client_projection.measurement.v1",
                "scenario_contract": "standard-800x800-food20-seed-0x5eedba5e-history64",
                "agent_count": agent_count,
                "warmups_per_case": WARMUPS,
                "samples_per_case": SAMPLES,
                "fanout_clients": FANOUT_CLIENTS,
                "source": {
                    "snapshot_revision": source.revision,
                    "host_revisions": source.revisions,
                    "history_samples": source.summary_history.len(),
                    "scientific_digest": digest_before.overall,
                },
                "request": request,
                "limits": limits,
                "budgets_ns": {
                    "cold_p95": cold_budget_p95_ns,
                    "warm_p95": warm_budget_p95_ns,
                    "moving_camera_p95": moving_budget_p95_ns,
                    "fanout_cold": fanout_cold_budget_ns,
                    "fanout_warm_p95": fanout_warm_budget_p95_ns,
                },
                "cold": {
                    "raw_ns": cold_samples,
                    "p50_ns": cold_p50_ns,
                    "p95_ns": cold_p95_ns,
                    "build": cold_stats,
                },
                "warm_cache": {
                    "raw_ns": warm_samples,
                    "p50_ns": warm_p50_ns,
                    "p95_ns": warm_p95_ns,
                    "build": warm_stats,
                    "hits": warm_broker.hits(),
                    "misses": warm_broker.misses(),
                    "retained_output_capacity_bytes": warm_broker.retained_output_capacity_bytes(),
                    "cache_byte_capacity": warm_broker.byte_capacity(),
                },
                "moving_camera": {
                    "raw_ns": moving_samples,
                    "p50_ns": moving_p50_ns,
                    "p95_ns": moving_p95_ns,
                    "build": moving_stats,
                    "hits": moving_broker.hits(),
                    "misses": moving_broker.misses(),
                    "evictions": moving_broker.evictions(),
                    "retained_output_capacity_bytes": moving_broker.retained_output_capacity_bytes(),
                    "cache_byte_capacity": moving_broker.byte_capacity(),
                },
                "fanout_128": {
                    "cold_ns": fanout_cold_ns,
                    "warm_raw_ns": fanout_warm_samples,
                    "warm_p50_ns": fanout_warm_p50_ns,
                    "warm_p95_ns": fanout_warm_p95_ns,
                    "agents_examined": fanout_agents_examined,
                    "visible_agents": fanout_visible_agents,
                    "canvas_cells": fanout_canvas_cells,
                    "top_k_peak_total": fanout_top_k_peak,
                    "chart_points": fanout_chart_points,
                    "output_capacity_bytes": fanout_output_capacity_bytes,
                    "retained_output_capacity_bytes": fanout_broker.retained_output_capacity_bytes(),
                    "cache_byte_capacity": fanout_broker.byte_capacity(),
                    "hits": fanout_broker.hits(),
                    "misses": fanout_broker.misses(),
                    "evictions": fanout_broker.evictions(),
                },
            });
            eprintln!(
                "{}",
                serde_json::to_string(&evidence)
                    .expect("serialize client projection measurement evidence")
            );

            assert!(
                cold_p95_ns < cold_budget_p95_ns,
                "cold projection p95 {cold_p95_ns}ns exceeded {cold_budget_p95_ns}ns"
            );
            assert!(
                warm_p95_ns < warm_budget_p95_ns,
                "warm projection p95 {warm_p95_ns}ns exceeded {warm_budget_p95_ns}ns"
            );
            assert!(
                moving_p95_ns < moving_budget_p95_ns,
                "moving projection p95 {moving_p95_ns}ns exceeded {moving_budget_p95_ns}ns"
            );
            assert!(
                fanout_cold_ns < fanout_cold_budget_ns,
                "cold 128-client fanout {fanout_cold_ns}ns exceeded {fanout_cold_budget_ns}ns"
            );
            assert!(
                fanout_warm_p95_ns < fanout_warm_budget_p95_ns,
                "warm 128-client fanout p95 {fanout_warm_p95_ns}ns exceeded {fanout_warm_budget_p95_ns}ns"
            );
        }
    }

    #[test]
    fn volatile_journal_releases_the_exact_accepted_allocation_after_receipt_polling() {
        let mut journal = VolatileJournal::default();
        let batch = Arc::new(JournalBatch::new(
            JournalBatchId::new(HostSessionId::new(1), 1),
            None,
            None,
            AppliedCommand {
                tick: Tick(0),
                revisions: HostRevisions::default(),
            },
            None,
            None,
        ));
        let released = Arc::downgrade(&batch);

        assert!(journal.try_admit(&batch).is_accepted());
        let pending = journal.batches();
        assert_eq!(pending.len(), 1);
        assert!(Arc::ptr_eq(&pending[0], &batch));
        drop(pending);
        assert_eq!(Arc::strong_count(&batch), 2);
        assert_eq!(journal.poll_receipts(1).len(), 1);
        assert!(journal.batches().is_empty());
        assert_eq!(Arc::strong_count(&batch), 1);
        drop(batch);
        assert!(released.upgrade().is_none());
    }

    #[test]
    fn non_scientific_churn_cannot_evict_lightweight_scientific_catch_up() {
        let mut event_options = options(true);
        event_options.scientific_event_capacity = 1;
        event_options.volatile_event_history_capacity = 3;
        let mut core = HostCore::new(HostSessionId::new(78), world(0), event_options)
            .expect("churn-isolation host");
        let mut port = core.local_port();
        let mut client = crate::HostClient::new(port.clone());
        let mut slow = client.event_cursor();

        submit(&mut port, 1, HostCommand::Step);
        core.drive(ManualInstant::from_nanos(0))
            .expect("first scientific boundary");
        core.drive(ManualInstant::from_nanos(1))
            .expect("first volatile commitment");
        submit(&mut port, 2, HostCommand::Step);
        core.drive(ManualInstant::from_nanos(2))
            .expect("second scientific boundary");
        core.drive(ManualInstant::from_nanos(3))
            .expect("second volatile commitment");

        let gap = match client
            .read_events(&mut slow, usize::MAX)
            .expect("slow event read")
        {
            crate::EventPoll::Gap(gap) => gap,
            other @ crate::EventPoll::Contiguous(_) => {
                panic!("wrapped hot ring must report a gap, got {other:?}")
            }
        };
        let locator = match gap.catch_up {
            crate::EventCatchUpState::Available(locator) => locator,
            other @ crate::EventCatchUpState::Unavailable(_) => {
                panic!("lightweight live catch-up must be available, got {other:?}")
            }
        };
        assert_eq!(locator.range().first, EventSequence::new(1));
        assert_eq!(locator.range().last, EventSequence::new(1));

        for ordinal in 0_u16..32 {
            let mut config = core.world.config().clone();
            config.food_growth_rate = (f32::from(ordinal) + 1.0) / 1_000.0;
            let command_id = u128::from(ordinal) + 100;
            submit(
                &mut port,
                command_id,
                HostCommand::UpdateConfig(Box::new(config)),
            );
            let boundary = u64::from(ordinal) * 2 + 10;
            core.drive(ManualInstant::from_nanos(boundary))
                .expect("non-scientific config journal boundary");
            core.drive(ManualInstant::from_nanos(boundary + 1))
                .expect("non-scientific config receipt boundary");
        }

        let caught_up = client
            .catch_up_events(&mut slow, locator, 1)
            .expect("catch up after config churn");
        let EventCatchUp::Contiguous(page) = caught_up else {
            panic!("committed scientific record must survive config churn");
        };
        assert_eq!(page.source, EventPageSource::LiveMemory);
        assert_eq!(page.events.len(), 1);
        assert_eq!(page.events[0].event.sequence, EventSequence::new(1));
        assert_eq!(
            page.events[0].commitment,
            EventCommitment::CommittedVolatile
        );
        assert_eq!(slow.last_seen(), EventSequence::new(1));
    }

    #[test]
    fn archived_idempotency_stays_bounded_and_answers_exact_retries_after_durable_archival() {
        let mut core = HostCore::new(HostSessionId::new(91), world(0), options(true))
            .expect("bounded-idempotency host");
        let mut port = core.local_port();
        // Shrink the retention bound so the eviction proof runs in tens of commands
        // rather than thousands.
        core.shared.borrow_mut().archived_retention = 8;

        // Drive 40 unique commands to terminal + durable archival: far more than the
        // live map may retain once lifecycle evidence is durably archived.
        for ordinal in 1_u128..=40 {
            submit(&mut port, ordinal, HostCommand::Step);
            let boundary = u64::try_from(ordinal * 2).expect("test boundary within u64");
            core.drive(ManualInstant::from_nanos(boundary))
                .expect("scientific boundary");
            core.drive(ManualInstant::from_nanos(boundary + 1))
                .expect("volatile commitment");
        }

        {
            let shared = core.shared.borrow();
            assert_eq!(
                shared.commands.len(),
                0,
                "durably-archived terminal commands must not stay in the live map"
            );
            assert_eq!(
                shared.archived_idempotency.len(),
                8,
                "the archived index must be bounded at the retention limit"
            );
            assert_eq!(shared.archived_order.len(), 8);
            // Oldest-first eviction: ids 33..=40 survive, ids 1..=32 are gone.
            for evicted in 1_u128..=32 {
                assert!(
                    !shared
                        .archived_idempotency
                        .contains_key(&CommandId::new(evicted)),
                    "id {evicted} should have been evicted beyond the bound"
                );
            }
        }

        // An exact retry of an archived command replays its archived terminal status.
        let replayed = submit(&mut port, 40, HostCommand::Step);
        assert!(
            matches!(replayed.application(), ApplicationState::Applied(_)),
            "exact archived retry must replay the applied terminal status, got {:?}",
            replayed.application()
        );
        assert!(matches!(
            replayed.journal(),
            JournalState::CommittedVolatile
        ));

        // A changed payload on an archived id collides instead of silently re-executing.
        let mut config = core.world.config().clone();
        config.food_growth_rate = 0.5;
        let collision = port.submit(envelope(40, HostCommand::UpdateConfig(Box::new(config))));
        assert!(
            matches!(collision, Err(HostAccessError::CommandIdCollision { command_id }) if command_id == CommandId::new(40)),
            "changed payload on an archived id must collide, got {collision:?}"
        );

        // An evicted-beyond-the-bound id reads as a fresh command again: bounded
        // idempotency is explicit, and the durable journal outranks it.
        let fresh = submit(&mut port, 1, HostCommand::Step);
        assert!(
            matches!(fresh.application(), ApplicationState::Admitted),
            "an id evicted beyond the retention bound is admitted as fresh"
        );
    }

    #[test]
    fn shutdown_command_is_never_evicted_from_the_live_map() {
        let mut core = HostCore::new(HostSessionId::new(92), world(0), options(true))
            .expect("shutdown-retention host");
        let mut port = core.local_port();
        core.shared.borrow_mut().archived_retention = 1;

        submit(&mut port, 1, HostCommand::Step);
        core.drive(ManualInstant::from_nanos(2))
            .expect("scientific boundary");
        core.drive(ManualInstant::from_nanos(3))
            .expect("volatile commitment");
        submit(&mut port, 2, HostCommand::Shutdown);
        core.drive(ManualInstant::from_nanos(4))
            .expect("shutdown boundary");
        core.drive(ManualInstant::from_nanos(5))
            .expect("shutdown commitment");

        let shared = core.shared.borrow();
        assert!(
            shared.commands.contains_key(&CommandId::new(2)),
            "the pending shutdown command must never be evicted"
        );
    }

    #[test]
    fn playback_commands_and_bounded_cadence_use_injected_time_only() {
        let (mut core, mut port) = host(false);
        core.drive(ManualInstant::from_nanos(0))
            .expect("establish epoch");
        let receipt = core
            .drive(ManualInstant::from_nanos(100))
            .expect("bounded catch-up");
        assert_eq!(receipt.scientific_steps, 4);
        assert_eq!(core.world_tick(), Tick(4));

        submit(&mut port, 1, HostCommand::Pause);
        let pause = core
            .drive(ManualInstant::from_nanos(200))
            .expect("pause boundary");
        assert_eq!(pause.scientific_steps, 0);
        assert_eq!(core.world_tick(), Tick(4));

        submit(&mut port, 2, HostCommand::Resume);
        submit(&mut port, 3, HostCommand::SetSpeed(2.5));
        let resume = core
            .drive(ManualInstant::from_nanos(300))
            .expect("resume boundary");
        assert_eq!(resume.scientific_steps, 0);
        assert_eq!(core.world_tick(), Tick(4));
        assert_eq!(
            core.latest_snapshot().playback.speed_multiplier.to_bits(),
            2.5_f32.to_bits()
        );

        let accelerated = core
            .drive(ManualInstant::from_nanos(310))
            .expect("accelerated boundary");
        assert_eq!(accelerated.scientific_steps, 2);
        assert_eq!(core.world_tick(), Tick(6));
    }

    #[test]
    fn explicit_step_order_is_exact_and_suppresses_due_cadence() {
        let (mut core, mut port) = host(false);
        core.drive(ManualInstant::from_nanos(0))
            .expect("establish epoch");
        submit(&mut port, 1, HostCommand::Step);
        submit(&mut port, 2, HostCommand::Resume);
        let step_then_resume = core
            .drive(ManualInstant::from_nanos(1_000))
            .expect("step then resume");
        assert_eq!(step_then_resume.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(1));
        assert!(!core.latest_snapshot().playback.paused);

        submit(&mut port, 3, HostCommand::Resume);
        submit(&mut port, 4, HostCommand::Step);
        let resume_then_step = core
            .drive(ManualInstant::from_nanos(2_000))
            .expect("resume then step");
        assert_eq!(resume_then_step.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(2));
        assert!(core.latest_snapshot().playback.paused);

        submit(&mut port, 5, HostCommand::Step);
        submit(&mut port, 6, HostCommand::Step);
        let two_steps = core
            .drive(ManualInstant::from_nanos(3_000))
            .expect("two explicit steps");
        assert_eq!(two_steps.scientific_steps, 2);
        assert_eq!(core.world_tick(), Tick(4));
        assert_eq!(applied(&status(&mut port, 5)).tick, Tick(3));
        assert_eq!(applied(&status(&mut port, 6)).tick, Tick(4));
    }

    #[test]
    fn rejected_step_does_not_suppress_the_due_automatic_boundary() {
        let (mut core, mut port) = host(false);
        core.drive(ManualInstant::from_nanos(0))
            .expect("establish epoch");
        port.submit(
            envelope(1, HostCommand::Step).expecting_control_revision(ControlRevision::new(99)),
        )
        .expect("guarded step admission");

        let receipt = core
            .drive(ManualInstant::from_nanos(10))
            .expect("conflict plus cadence");
        assert_eq!(receipt.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(1));
        assert_eq!(
            core.latest_snapshot().revisions.control,
            ControlRevision::new(0)
        );
        assert!(matches!(
            status(&mut port, 1).application(),
            ApplicationState::Rejected(RejectionReason::ControlRevisionConflict { .. })
        ));
    }

    #[test]
    fn speed_change_does_not_erase_elapsed_time_at_the_prior_rate() {
        let (mut core, mut port) = host(false);
        core.drive(ManualInstant::from_nanos(0))
            .expect("establish epoch");
        submit(&mut port, 1, HostCommand::SetSpeed(2.0));

        let receipt = core
            .drive(ManualInstant::from_nanos(10))
            .expect("speed boundary");
        assert_eq!(receipt.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(1));
    }

    #[test]
    fn mixed_command_classes_apply_at_their_true_ordered_tick() {
        let updated = ScriptBotsConfig {
            food_growth_rate: 0.025,
            ..ScriptBotsConfig::default()
        };

        let (mut before, mut before_port) = host(true);
        submit(
            &mut before_port,
            1,
            HostCommand::UpdateConfig(Box::new(updated.clone())),
        );
        submit(&mut before_port, 2, HostCommand::Step);
        before
            .drive(ManualInstant::from_nanos(0))
            .expect("config then step");
        assert_eq!(applied(&status(&mut before_port, 1)).tick, Tick(0));
        assert_eq!(applied(&status(&mut before_port, 2)).tick, Tick(1));

        let (mut after, mut after_port) = host(true);
        submit(&mut after_port, 1, HostCommand::Step);
        submit(
            &mut after_port,
            2,
            HostCommand::UpdateConfig(Box::new(updated)),
        );
        after
            .drive(ManualInstant::from_nanos(0))
            .expect("step then config");
        assert_eq!(applied(&status(&mut after_port, 1)).tick, Tick(1));
        assert_eq!(applied(&status(&mut after_port, 2)).tick, Tick(1));
    }

    #[test]
    fn deduplication_and_control_cas_never_reapply_or_advance_on_conflict() {
        let (mut core, mut port) = host(true);
        let original = submit(&mut port, 1, HostCommand::Pause);
        let duplicate = submit(&mut port, 1, HostCommand::Pause);
        assert_eq!(duplicate, original);
        assert_eq!(port.queue_depth(), 1);
        assert_eq!(
            port.submit(envelope(1, HostCommand::Step)),
            Err(HostAccessError::CommandIdCollision {
                command_id: CommandId::new(1),
            })
        );
        assert_eq!(port.queue_depth(), 1);

        let winner =
            envelope(2, HostCommand::Resume).expecting_control_revision(ControlRevision::new(1));
        let conflict =
            envelope(3, HostCommand::Step).expecting_control_revision(ControlRevision::new(1));
        port.submit(winner).expect("winner admission");
        port.submit(conflict).expect("conflict admission");
        core.drive(ManualInstant::from_nanos(0))
            .expect("ordered CAS boundary");

        assert_eq!(core.world_tick(), Tick(0));
        assert_eq!(
            core.latest_snapshot().revisions.control,
            ControlRevision::new(2)
        );
        assert!(matches!(
            status(&mut port, 3).application(),
            ApplicationState::Rejected(RejectionReason::ControlRevisionConflict {
                expected,
                actual,
            }) if *expected == ControlRevision::new(1) && *actual == ControlRevision::new(2)
        ));
    }

    #[test]
    fn bounded_admission_has_explicit_results_for_1_32_33_and_1000() {
        for requested in [1_usize, 32, 33, 1_000] {
            let (mut core, mut port) = host(true);
            let mut admitted = 0;
            let mut overloaded = 0;
            let mut evidence_backpressured = 0;
            for offset in 0..requested {
                let id = u128::try_from(offset + 1).expect("test id");
                match port.submit(envelope(id, HostCommand::Step)) {
                    Ok(result) => match result.application() {
                        ApplicationState::Admitted => admitted += 1,
                        ApplicationState::Rejected(RejectionReason::Overloaded {
                            capacity: 32,
                        }) => {
                            overloaded += 1;
                            assert_eq!(result.admission_sequence(), None);
                        }
                        other => panic!("unexpected burst result: {other:?}"),
                    },
                    Err(HostAccessError::CommandEvidenceBackpressure { capacity: 32 }) => {
                        evidence_backpressured += 1;
                    }
                    Err(other) => panic!("unexpected burst error: {other:?}"),
                }
            }
            let expected_admitted = requested.min(32);
            let expected_overloaded = requested.saturating_sub(32).min(32);
            assert_eq!(admitted, expected_admitted);
            assert_eq!(overloaded, expected_overloaded);
            assert_eq!(
                evidence_backpressured,
                requested - expected_admitted - expected_overloaded
            );
            assert_eq!(port.queue_depth(), requested.min(32));
            let receipt = core
                .drive(ManualInstant::from_nanos(0))
                .expect("burst drive");
            assert_eq!(receipt.commands_completed, requested.min(32));
            assert_eq!(receipt.scientific_steps, requested.min(32));
            assert_eq!(
                core.world_tick().0,
                u64::try_from(requested.min(32)).expect("bounded burst count fits u64")
            );
        }
    }

    #[derive(Default)]
    struct FakeJournalState {
        full: bool,
        closed: bool,
        suppress_receipts: bool,
        attempts: Vec<Arc<JournalBatch>>,
        receipts: VecDeque<JournalReceipt>,
    }

    struct FakeJournal {
        state: Rc<RefCell<FakeJournalState>>,
    }

    impl JournalPort for FakeJournal {
        fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
            let mut state = self.state.borrow_mut();
            state.attempts.push(Arc::clone(batch));
            if state.closed {
                JournalAdmission::Closed {
                    batch_id: batch.id(),
                }
            } else if state.full {
                JournalAdmission::Full {
                    batch_id: batch.id(),
                    capacity: 1,
                }
            } else {
                if !state.suppress_receipts {
                    state.receipts.push_back(JournalReceipt::new(
                        batch.id(),
                        JournalReceiptState::Durable,
                    ));
                }
                JournalAdmission::Accepted {
                    batch_id: batch.id(),
                }
            }
        }

        fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
            let mut state = self.state.borrow_mut();
            let count = limit.min(state.receipts.len());
            state.receipts.drain(..count).collect()
        }
    }

    #[test]
    fn control_successes_emit_ordered_application_lifecycles_and_advance_on_receipt() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState::default()));
        let mut core = HostCore::with_journal(
            HostSessionId::new(90),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("control lifecycle host");
        let mut port = core.local_port();
        let commands = [
            (CommandId::from_client_sequence(11, 1), HostCommand::Pause),
            (CommandId::from_client_sequence(11, 2), HostCommand::Resume),
            (
                CommandId::from_client_sequence(11, 3),
                HostCommand::SetSpeed(2.0),
            ),
        ];
        for (command_id, command) in &commands {
            port.submit(CommandEnvelope::new(*command_id, command.clone()))
                .expect("control admission");
        }

        let receipt = core
            .drive(ManualInstant::from_nanos(0))
            .expect("control lifecycle boundary");
        assert_eq!(receipt.commands_completed, commands.len());
        let journal = journal_state.borrow();
        assert_eq!(journal.attempts.len(), commands.len());
        for (index, ((command_id, _), batch)) in commands.iter().zip(&journal.attempts).enumerate()
        {
            let lifecycle = batch
                .command_lifecycle()
                .expect("control command lifecycle");
            assert_eq!(
                lifecycle.schema_version(),
                crate::COMMAND_LIFECYCLE_SCHEMA_VERSION
            );
            assert_eq!(lifecycle.source_client_namespace(), 11);
            assert_eq!(lifecycle.envelope().command_id, *command_id);
            assert_eq!(lifecycle.transitions().len(), 2);
            assert_eq!(lifecycle.transitions()[0].ordinal(), 0);
            assert_eq!(
                lifecycle.transitions()[0].boundary().revisions.control,
                ControlRevision::new(0)
            );
            let terminal = lifecycle.terminal().expect("terminal control transition");
            assert!(matches!(
                terminal.application(),
                ApplicationState::Applied(_)
            ));
            assert_eq!(
                terminal.boundary().revisions.control,
                ControlRevision::new(u64::try_from(index + 1).expect("control revision"))
            );
            assert!(batch.requires_runtime_journal());
            assert_eq!(
                status(&mut port, command_id.get()).journal(),
                &JournalState::Pending
            );
        }
        drop(journal);
        core.drive(ManualInstant::from_nanos(1))
            .expect("control lifecycle receipts");
        for (command_id, _) in &commands {
            assert_eq!(
                status(&mut port, command_id.get()).journal(),
                &JournalState::Durable
            );
        }
    }

    #[test]
    fn validation_rejection_is_a_command_only_audit_with_no_admission() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState::default()));
        let mut core = HostCore::with_journal(
            HostSessionId::new(91),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("validation lifecycle host");
        let mut port = core.local_port();
        let rejected = submit(&mut port, 1, HostCommand::SetSpeed(-1.0));
        let duplicate = submit(&mut port, 1, HostCommand::SetSpeed(-1.0));
        assert_eq!(duplicate, rejected);
        assert_eq!(rejected.journal(), &JournalState::Pending);
        assert!(matches!(
            rejected.application(),
            ApplicationState::Rejected(RejectionReason::Validation { .. })
        ));

        core.drive(ManualInstant::from_nanos(0))
            .expect("validation audit boundary");
        let journal = journal_state.borrow();
        assert_eq!(journal.attempts.len(), 1);
        let lifecycle = journal.attempts[0]
            .command_lifecycle()
            .expect("validation lifecycle");
        assert_eq!(lifecycle.admission_sequence(), None);
        assert_eq!(lifecycle.transitions().len(), 1);
        assert!(matches!(
            lifecycle.transitions()[0].application(),
            ApplicationState::Rejected(RejectionReason::Validation { .. })
        ));
        assert_eq!(lifecycle.transitions()[0].boundary().tick, Tick(0));
        assert!(journal.attempts[0].scientific().is_none());
        assert!(journal.attempts[0].persistence().is_none());
        drop(journal);
        core.drive(ManualInstant::from_nanos(1))
            .expect("validation audit receipt");
        assert_eq!(status(&mut port, 1).journal(), &JournalState::Durable);
    }

    #[test]
    fn config_identity_preserves_nan_bits_and_compacts_audited_envelopes() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState::default()));
        let mut core = HostCore::with_journal(
            HostSessionId::new(97),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("config identity host");
        let mut port = core.local_port();
        let mut envelopes = Vec::new();
        for sequence in 1_u128..=8 {
            let config = ScriptBotsConfig {
                initial_food: f32::from_bits(0x7fc0_0001),
                neuroflow: scriptbots_core::NeuroflowSettings {
                    hidden_layers: vec![64; 64],
                    ..scriptbots_core::NeuroflowSettings::default()
                },
                ..ScriptBotsConfig::default()
            };
            let envelope = CommandEnvelope::new(
                CommandId::new(sequence),
                HostCommand::UpdateConfig(Box::new(config)),
            );
            let rejected = port
                .submit(envelope.clone())
                .expect("non-finite config has inspectable rejection");
            assert!(matches!(
                rejected.application(),
                ApplicationState::Rejected(RejectionReason::Validation { .. })
            ));
            assert_eq!(rejected.journal(), &JournalState::Pending);
            assert_eq!(
                port.submit(envelope.clone())
                    .expect("same-bit NaN retry is idempotent"),
                rejected
            );
            envelopes.push(envelope);
        }

        core.drive(ManualInstant::from_nanos(0))
            .expect("config rejection audits");
        assert_eq!(journal_state.borrow().attempts.len(), envelopes.len());
        for envelope in &envelopes {
            let command_id = envelope.command_id;
            let authority_status = {
                let shared = core.shared.borrow();
                let authority = shared
                    .commands
                    .get(&command_id)
                    .expect("config command authority");
                assert!(
                    authority.envelope.is_none(),
                    "accepted lifecycle handoff must compact the full config envelope"
                );
                authority.status.clone()
            };
            assert_eq!(
                port.submit(envelope.clone())
                    .expect("exact retry uses compact identity"),
                authority_status
            );

            let mut changed = envelope.clone();
            let HostCommand::UpdateConfig(config) = &mut changed.command else {
                panic!("test envelope is an update-config command");
            };
            config.initial_food = f32::from_bits(0x7fc0_0002);
            assert_eq!(
                port.submit(changed),
                Err(HostAccessError::CommandIdCollision { command_id })
            );
        }

        core.drive(ManualInstant::from_nanos(1))
            .expect("config rejection audit receipts");
        for envelope in &envelopes {
            assert_eq!(
                port.submit(envelope.clone())
                    .expect("exact retry remains idempotent after receipt")
                    .journal(),
                &JournalState::Durable
            );
        }
    }

    #[test]
    fn overload_audit_drains_before_the_earlier_admitted_science_command() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState::default()));
        let mut test_options = options(true);
        test_options.command_capacity = 1;
        let mut core = HostCore::with_journal(
            HostSessionId::new(92),
            world(0),
            test_options,
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("overload audit host");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);
        let overloaded = submit(&mut port, 2, HostCommand::Pause);
        assert!(matches!(
            overloaded.application(),
            ApplicationState::Rejected(RejectionReason::Overloaded { capacity: 1 })
        ));

        core.drive(ManualInstant::from_nanos(0))
            .expect("ordered overload audit");
        let journal = journal_state.borrow();
        assert_eq!(journal.attempts.len(), 2);
        assert_eq!(journal.attempts[0].command_id(), Some(CommandId::new(2)));
        assert!(matches!(
            journal.attempts[0]
                .command_lifecycle()
                .and_then(CommandLifecycleEvidence::terminal)
                .map(CommandLifecycleTransition::application),
            Some(ApplicationState::Rejected(RejectionReason::Overloaded {
                capacity: 1
            }))
        ));
        assert_eq!(journal.attempts[1].command_id(), Some(CommandId::new(1)));
        assert!(journal.attempts[1].scientific().is_some());
        assert!(matches!(
            journal.attempts[1]
                .command_lifecycle()
                .and_then(CommandLifecycleEvidence::terminal)
                .map(CommandLifecycleTransition::application),
            Some(ApplicationState::Applied(_))
        ));
    }

    #[test]
    fn successful_step_and_config_batches_carry_applied_lifecycle_evidence() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState::default()));
        let mut core = HostCore::with_journal(
            HostSessionId::new(96),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("successful lifecycle host");
        let mut port = core.local_port();
        let mut config = core.world.config().clone();
        config.food_growth_rate = 0.02;
        submit(&mut port, 1, HostCommand::Step);
        submit(&mut port, 2, HostCommand::UpdateConfig(Box::new(config)));

        let receipt = core
            .drive(ManualInstant::from_nanos(0))
            .expect("step and config lifecycle boundary");
        assert_eq!(receipt.commands_completed, 2);
        let journal = journal_state.borrow();
        assert_eq!(journal.attempts.len(), 2);
        for batch in &journal.attempts {
            let lifecycle = batch
                .command_lifecycle()
                .expect("successful command lifecycle");
            assert_eq!(lifecycle.transitions().len(), 2);
            assert!(matches!(
                lifecycle
                    .terminal()
                    .map(CommandLifecycleTransition::application),
                Some(ApplicationState::Applied(_))
            ));
            assert!(batch.requires_runtime_journal());
        }
        assert!(matches!(
            &journal.attempts[0]
                .command_lifecycle()
                .expect("step lifecycle")
                .envelope()
                .command,
            HostCommand::Step
        ));
        assert!(journal.attempts[0].scientific().is_some());
        assert!(matches!(
            &journal.attempts[1]
                .command_lifecycle()
                .expect("config lifecycle")
                .envelope()
                .command,
            HostCommand::UpdateConfig(_)
        ));
        assert!(journal.attempts[1].scientific().is_none());
    }

    #[test]
    fn admitted_revision_conflicts_and_failures_keep_their_application_boundaries() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState::default()));
        let mut core = HostCore::with_journal(
            HostSessionId::new(93),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("terminal lifecycle host");
        let mut port = core.local_port();
        port.submit(
            envelope(1, HostCommand::Pause)
                .expecting_scientific_revision(ScientificRevision::new(7)),
        )
        .expect("conflicting admission");
        core.latched_fault = Some(HostFault::Scientific {
            tick: Tick(0),
            code: "injected".to_owned(),
            message: "test fault".to_owned(),
        });
        submit(&mut port, 2, HostCommand::Step);

        core.drive(ManualInstant::from_nanos(0))
            .expect("conflict and failure audits");
        let journal = journal_state.borrow();
        assert_eq!(journal.attempts.len(), 2);
        let conflict = journal.attempts[0]
            .command_lifecycle()
            .expect("conflict lifecycle");
        assert!(matches!(
            conflict.terminal().map(CommandLifecycleTransition::application),
            Some(ApplicationState::Rejected(
                RejectionReason::ScientificRevisionConflict {
                    expected,
                    actual,
                }
            )) if *expected == ScientificRevision::new(7)
                && *actual == ScientificRevision::new(0)
        ));
        let failed = journal.attempts[1]
            .command_lifecycle()
            .expect("failure lifecycle");
        assert!(matches!(
            failed.terminal().map(CommandLifecycleTransition::application),
            Some(ApplicationState::Failed(ApplicationFailure { code, .. }))
                if code == "science_blocked"
        ));
        assert_eq!(
            conflict.transitions()[0].boundary(),
            conflict.transitions()[1].boundary()
        );
        assert_eq!(
            failed.transitions()[0].boundary(),
            failed.transitions()[1].boundary()
        );
        assert_eq!(status(&mut port, 1).journal(), &JournalState::Pending);
        assert_eq!(status(&mut port, 2).journal(), &JournalState::Pending);
        drop(journal);
        core.drive(ManualInstant::from_nanos(1))
            .expect("conflict and failure audit receipts");
        assert_eq!(status(&mut port, 1).journal(), &JournalState::Durable);
        assert_eq!(status(&mut port, 2).journal(), &JournalState::Durable);
    }

    #[test]
    fn full_control_audit_retries_the_exact_arc() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState {
            full: true,
            ..FakeJournalState::default()
        }));
        let mut core = HostCore::with_journal(
            HostSessionId::new(94),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("full control audit host");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Pause);
        core.drive(ManualInstant::from_nanos(0))
            .expect("full control audit boundary");
        let retained = core
            .pending_journal_batch()
            .expect("retained control audit");
        assert!(Arc::ptr_eq(&retained, &journal_state.borrow().attempts[0]));
        assert!(
            core.shared
                .borrow()
                .commands
                .get(&CommandId::new(1))
                .expect("control authority")
                .envelope
                .is_none(),
            "the retained exact batch supersedes the authority envelope"
        );
        assert_eq!(
            port.submit(envelope(1, HostCommand::Pause))
                .expect("exact retry uses compact identity")
                .journal(),
            &JournalState::Pending
        );
        assert_eq!(
            port.submit(envelope(1, HostCommand::Resume)),
            Err(HostAccessError::CommandIdCollision {
                command_id: CommandId::new(1)
            })
        );

        journal_state.borrow_mut().full = false;
        let retried = core
            .retry_retained_journal()
            .expect("control audit retry")
            .expect("retained admission result");
        assert!(retried.is_accepted());
        assert!(Arc::ptr_eq(
            &journal_state.borrow().attempts[0],
            &journal_state.borrow().attempts[1]
        ));
        assert!(core.pending_journal_batch().is_none());
        assert_eq!(status(&mut port, 1).journal(), &JournalState::Pending);
        core.drive(ManualInstant::from_nanos(1))
            .expect("control audit receipt");
        assert_eq!(status(&mut port, 1).journal(), &JournalState::Durable);
    }

    #[test]
    fn pending_host_stopping_audit_blocks_shutdown_then_stopped_ingress_fails_typed() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState {
            full: true,
            ..FakeJournalState::default()
        }));
        let mut core = HostCore::with_journal(
            HostSessionId::new(95),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("shutdown audit host");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Shutdown);
        let stopping = submit(&mut port, 2, HostCommand::Pause);
        assert!(matches!(
            stopping.application(),
            ApplicationState::Rejected(RejectionReason::HostStopping)
        ));

        let blocked = core
            .drive(ManualInstant::from_nanos(0))
            .expect("blocked shutdown audit ordering");
        assert!(matches!(
            blocked.blocker,
            Some(HostBlocker::JournalFull { .. })
        ));
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Running);
        assert!(matches!(
            status(&mut port, 1).application(),
            ApplicationState::Admitted
        ));
        {
            let journal = journal_state.borrow();
            assert_eq!(journal.attempts.len(), 1);
            assert_eq!(journal.attempts[0].command_id(), Some(CommandId::new(2)));
            assert!(matches!(
                journal.attempts[0]
                    .command_lifecycle()
                    .and_then(CommandLifecycleEvidence::terminal)
                    .map(CommandLifecycleTransition::application),
                Some(ApplicationState::Rejected(RejectionReason::HostStopping))
            ));
        }
        journal_state.borrow_mut().full = false;
        core.retry_retained_journal()
            .expect("stopping audit retry")
            .expect("retained stopping audit");
        core.drive(ManualInstant::from_nanos(1))
            .expect("shutdown applies after audit admission");
        {
            let journal = journal_state.borrow();
            assert_eq!(journal.attempts.len(), 3);
            assert!(Arc::ptr_eq(&journal.attempts[0], &journal.attempts[1]));
            assert_eq!(journal.attempts[2].command_id(), Some(CommandId::new(1)));
            assert!(journal.attempts[2].is_applied_shutdown());
            assert!(matches!(
                journal.attempts[2]
                    .command_lifecycle()
                    .and_then(CommandLifecycleEvidence::terminal)
                    .map(CommandLifecycleTransition::application),
                Some(ApplicationState::Applied(_))
            ));
        }
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopping);
        core.drive(ManualInstant::from_nanos(2))
            .expect("durable shutdown audit receipts");
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);
        assert_eq!(
            port.submit(envelope(3, HostCommand::Pause)),
            Err(HostAccessError::CommandEvidenceClosed {
                lifecycle: HostLifecycle::Stopped,
            })
        );
    }

    #[test]
    fn scientific_event_ring_wraps_with_exact_live_memory_catch_up() {
        let mut event_options = options(true);
        event_options.scientific_event_capacity = 2;
        event_options.volatile_event_history_capacity = 8;
        let mut core = HostCore::new(HostSessionId::new(70), world(0), event_options)
            .expect("event catch-up host");
        let mut port = core.local_port();
        let mut client = crate::HostClient::new(port.clone());
        let mut slow = client.event_cursor();

        for id in 1..=3 {
            submit(&mut port, id, HostCommand::Step);
            core.drive(ManualInstant::from_nanos(
                u64::try_from(id).expect("small event sequence"),
            ))
            .expect("scientific event drive");
        }

        let gap = match client
            .read_events(&mut slow, usize::MAX)
            .expect("slow event poll")
        {
            crate::EventPoll::Gap(gap) => gap,
            other @ crate::EventPoll::Contiguous(_) => {
                panic!("slow cursor must receive a gap, got {other:?}")
            }
        };
        assert_eq!(gap.expected, EventSequence::new(1));
        assert_eq!(gap.missing.first, EventSequence::new(1));
        assert_eq!(gap.missing.last, EventSequence::new(1));
        assert_eq!(gap.hot_available.first, EventSequence::new(2));
        assert_eq!(gap.hot_available.last, EventSequence::new(3));
        assert_eq!(slow.last_seen(), EventSequence::new(0));
        let locator = match gap.catch_up {
            crate::EventCatchUpState::Available(locator) => locator,
            other @ crate::EventCatchUpState::Unavailable(_) => {
                panic!("live memory locator expected, got {other:?}")
            }
        };
        assert_eq!(locator.guarantee(), EventCatchUpGuarantee::LiveMemory);
        let caught_up = client
            .catch_up_events(&mut slow, locator, 1)
            .expect("live memory catch-up");
        let crate::EventCatchUp::Contiguous(page) = caught_up else {
            panic!("live memory catch-up must be contiguous");
        };
        assert_eq!(page.source, EventPageSource::LiveMemory);
        assert_eq!(page.events.len(), 1);
        assert_eq!(page.events[0].event.sequence, EventSequence::new(1));
        assert_eq!(
            page.events[0].commitment,
            EventCommitment::CommittedVolatile
        );
        assert_eq!(slow.last_seen(), EventSequence::new(1));

        let crate::EventPoll::Contiguous(hot) = client
            .read_events(&mut slow, usize::MAX)
            .expect("hot suffix")
        else {
            panic!("caught-up cursor must rejoin hot ring");
        };
        assert_eq!(
            hot.events
                .iter()
                .map(|entry| entry.event.sequence)
                .collect::<Vec<_>>(),
            [EventSequence::new(2), EventSequence::new(3)]
        );
        assert_eq!(slow.last_seen(), EventSequence::new(3));
        let crate::EventPoll::Contiguous(duplicate) = client
            .read_events(&mut slow, usize::MAX)
            .expect("duplicate tip poll")
        else {
            panic!("tip poll must remain contiguous");
        };
        assert!(duplicate.events.is_empty());
        assert_eq!(slow.last_seen(), EventSequence::new(3));

        let mut wrong_session = crate::EventCursor::beginning(HostSessionId::new(71));
        let before = wrong_session;
        assert!(client.read_events(&mut wrong_session, 1).is_err());
        assert_eq!(wrong_session, before);
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one public-boundary test contrasts recoverable and explicitly accepted event gaps through the same frontend API"
    )]
    fn null_frontend_exercises_live_catch_up_and_explicit_unavailable_gap_acceptance() {
        let mut live_options = options(true);
        live_options.scientific_event_capacity = 1;
        live_options.volatile_event_history_capacity = 4;
        let mut live_core = HostCore::new(HostSessionId::new(81), world(0), live_options)
            .expect("null-frontend live-memory host");
        let mut live_frontend = crate::NullFrontend::new(live_core.local_port(), 11);

        live_frontend.step().expect("first live frontend step");
        live_frontend
            .drive_at(&mut live_core, ManualInstant::from_nanos(0))
            .expect("first live frontend drive");
        live_frontend
            .drive_at(&mut live_core, ManualInstant::from_nanos(1))
            .expect("first live frontend commitment");
        live_frontend.step().expect("second live frontend step");
        live_frontend
            .drive_at(&mut live_core, ManualInstant::from_nanos(2))
            .expect("second live frontend drive");
        live_frontend
            .drive_at(&mut live_core, ManualInstant::from_nanos(3))
            .expect("second live frontend commitment");

        let live_gap = match live_frontend
            .read_events(usize::MAX)
            .expect("public live frontend event read")
        {
            crate::EventPoll::Gap(gap) => gap,
            other @ crate::EventPoll::Contiguous(_) => {
                panic!("wrapped live frontend must observe a gap, got {other:?}")
            }
        };
        let live_locator = match live_gap.catch_up {
            crate::EventCatchUpState::Available(locator) => locator,
            other @ crate::EventCatchUpState::Unavailable(_) => {
                panic!("live frontend must receive a catch-up locator, got {other:?}")
            }
        };
        let EventCatchUp::Contiguous(caught_up) = live_frontend
            .catch_up_events(live_locator, 1)
            .expect("public live frontend catch-up")
        else {
            panic!("live frontend catch-up must be contiguous");
        };
        assert_eq!(caught_up.source, EventPageSource::LiveMemory);
        assert_eq!(caught_up.events.len(), 1);
        assert_eq!(caught_up.events[0].event.sequence, EventSequence::new(1));
        let crate::EventPoll::Contiguous(live_hot) = live_frontend
            .read_events(1)
            .expect("public live frontend hot suffix")
        else {
            panic!("live frontend must rejoin the hot ring");
        };
        assert_eq!(live_hot.events.len(), 1);
        assert_eq!(live_hot.events[0].event.sequence, EventSequence::new(2));

        let state = Rc::new(RefCell::new(FakeJournalState::default()));
        let mut unavailable_options = options(true);
        unavailable_options.scientific_event_capacity = 1;
        unavailable_options.volatile_event_history_capacity = 4;
        let mut unavailable_core = HostCore::with_journal(
            HostSessionId::new(82),
            world(0),
            unavailable_options,
            Box::new(FakeJournal { state }),
        )
        .expect("null-frontend no-reader host");
        let mut unavailable_frontend = crate::NullFrontend::new(unavailable_core.local_port(), 12);
        unavailable_frontend
            .step()
            .expect("first unavailable frontend step");
        unavailable_frontend
            .drive_at(&mut unavailable_core, ManualInstant::from_nanos(0))
            .expect("first unavailable frontend drive");
        unavailable_frontend
            .drive_at(&mut unavailable_core, ManualInstant::from_nanos(1))
            .expect("first unavailable frontend commitment");
        unavailable_frontend
            .step()
            .expect("second unavailable frontend step");
        unavailable_frontend
            .drive_at(&mut unavailable_core, ManualInstant::from_nanos(2))
            .expect("second unavailable frontend drive");
        unavailable_frontend
            .drive_at(&mut unavailable_core, ManualInstant::from_nanos(3))
            .expect("second unavailable frontend commitment");

        let unavailable_gap = match unavailable_frontend
            .read_events(1)
            .expect("public unavailable frontend event read")
        {
            crate::EventPoll::Gap(gap) => gap,
            other @ crate::EventPoll::Contiguous(_) => {
                panic!("no-reader frontend must observe a gap, got {other:?}")
            }
        };
        assert_eq!(
            unavailable_gap.catch_up,
            crate::EventCatchUpState::Unavailable(EventCatchUpUnavailableReason::NoReader)
        );
        unavailable_frontend
            .accept_event_gap(unavailable_gap)
            .expect("explicitly accept unavailable prefix");
        let crate::EventPoll::Contiguous(unavailable_hot) = unavailable_frontend
            .read_events(1)
            .expect("public hot read after accepting gap")
        else {
            panic!("accepted no-reader gap must resume at the hot ring");
        };
        assert_eq!(unavailable_hot.events.len(), 1);
        assert_eq!(
            unavailable_hot.events[0].event.sequence,
            EventSequence::new(2)
        );
    }

    #[test]
    fn pending_event_high_water_stops_before_loss_and_resumes_exact_step() {
        let state = Rc::new(RefCell::new(FakeJournalState {
            suppress_receipts: true,
            ..FakeJournalState::default()
        }));
        let mut event_options = options(true);
        event_options.scientific_event_capacity = 2;
        event_options.volatile_event_history_capacity = 8;
        let mut core = HostCore::with_journal(
            HostSessionId::new(72),
            world(0),
            event_options,
            Box::new(FakeJournal {
                state: Rc::clone(&state),
            }),
        )
        .expect("high-water host");
        let mut port = core.local_port();
        for id in 1..=3 {
            let status = submit(&mut port, id, HostCommand::Step);
            assert!(matches!(status.application(), ApplicationState::Admitted));
        }

        let first = core
            .drive(ManualInstant::from_nanos(0))
            .expect("high-water drive");
        assert_eq!(first.commands_completed, 2);
        assert_eq!(first.scientific_steps, 2);
        assert_eq!(core.world_tick(), Tick(2));
        assert_eq!(port.queue_depth(), 1);
        assert_eq!(core.event_hub().len(), 2);
        assert_eq!(core.event_hub().pending_count(), 2);
        assert!(matches!(
            first.blocker,
            Some(HostBlocker::EventJournalHighWater {
                capacity: 2,
                pending: 2,
                oldest_pending: Some(_),
                ..
            })
        ));
        let blocked_digest = core
            .scientific_digest_v1()
            .expect("blocked scientific digest");
        let blocked_revision = core.latest_snapshot().revisions.scientific;
        let second = core
            .drive(ManualInstant::from_nanos(1_000))
            .expect("repeated high-water drive");
        assert_eq!(second.scientific_steps, 0);
        assert_eq!(port.queue_depth(), 1);
        assert_eq!(
            core.scientific_digest_v1()
                .expect("repeated blocked digest"),
            blocked_digest
        );
        assert_eq!(
            core.latest_snapshot().revisions.scientific,
            blocked_revision
        );

        let batch_ids = state
            .borrow()
            .attempts
            .iter()
            .map(|batch| batch.id())
            .collect::<Vec<_>>();
        assert_eq!(batch_ids.len(), 2);
        state.borrow_mut().receipts.extend(
            batch_ids
                .iter()
                .copied()
                .map(|batch_id| JournalReceipt::new(batch_id, JournalReceiptState::Durable)),
        );
        let resumed = core
            .drive(ManualInstant::from_nanos(1_001))
            .expect("event-pressure recovery");
        assert_eq!(resumed.commands_completed, 1);
        assert_eq!(resumed.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(3));
        assert_eq!(port.queue_depth(), 0);
        assert_eq!(state.borrow().attempts.len(), 3);
        assert_eq!(
            status(&mut port, 3).application(),
            &ApplicationState::Applied(AppliedCommand {
                tick: Tick(3),
                revisions: core.latest_snapshot().revisions,
            })
        );
    }

    #[test]
    fn running_automatic_high_water_discards_blocked_time_and_resumes_one_step() {
        let state = Rc::new(RefCell::new(FakeJournalState {
            suppress_receipts: true,
            ..FakeJournalState::default()
        }));
        let mut event_options = options(false);
        event_options.scientific_event_capacity = 2;
        event_options.volatile_event_history_capacity = 8;
        let mut core = HostCore::with_journal(
            HostSessionId::new(79),
            world(0),
            event_options,
            Box::new(FakeJournal {
                state: Rc::clone(&state),
            }),
        )
        .expect("automatic high-water host");

        core.drive(ManualInstant::from_nanos(0))
            .expect("automatic epoch");
        assert_eq!(
            core.drive(ManualInstant::from_nanos(10))
                .expect("first automatic boundary")
                .scientific_steps,
            1
        );
        assert_eq!(
            core.drive(ManualInstant::from_nanos(20))
                .expect("second automatic boundary")
                .scientific_steps,
            1
        );
        assert_eq!(core.world_tick(), Tick(2));

        let pressure = core
            .drive(ManualInstant::from_nanos(1_000))
            .expect("automatic high-water boundary");
        assert_eq!(pressure.scientific_steps, 0);
        assert_eq!(pressure.automatic_steps_due, 98);
        assert_eq!(pressure.automatic_steps_skipped, 94);
        assert!(matches!(
            pressure.blocker,
            Some(HostBlocker::EventJournalHighWater {
                capacity: 2,
                pending: 2,
                reason: crate::EventHighWaterReason::Pending,
                ..
            })
        ));
        assert_eq!(core.world_tick(), Tick(2));

        let still_blocked = core
            .drive(ManualInstant::from_nanos(2_000))
            .expect("blocked time does not accumulate");
        assert_eq!(still_blocked.scientific_steps, 0);
        assert_eq!(still_blocked.automatic_steps_due, 0);
        assert_eq!(still_blocked.automatic_steps_skipped, 0);

        let batch_ids = state
            .borrow()
            .attempts
            .iter()
            .map(|batch| batch.id())
            .collect::<Vec<_>>();
        assert_eq!(batch_ids.len(), 2);
        state.borrow_mut().receipts.extend(
            batch_ids
                .iter()
                .copied()
                .map(|batch_id| JournalReceipt::new(batch_id, JournalReceiptState::Durable)),
        );
        let recovery = core
            .drive(ManualInstant::from_nanos(3_000))
            .expect("clear automatic event pressure");
        assert_eq!(recovery.scientific_steps, 0);
        assert_eq!(recovery.automatic_steps_due, 0);
        assert_eq!(core.world_tick(), Tick(2));

        let resumed = core
            .drive(ManualInstant::from_nanos(3_010))
            .expect("resume automatic cadence");
        assert_eq!(resumed.automatic_steps_due, 1);
        assert_eq!(resumed.automatic_steps_skipped, 0);
        assert_eq!(resumed.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(3));
        assert_eq!(state.borrow().attempts.len(), 3);
    }

    #[test]
    fn maintenance_drive_cancels_an_unused_full_ring_reservation() {
        let state = Rc::new(RefCell::new(FakeJournalState::default()));
        let mut event_options = options(true);
        event_options.scientific_event_capacity = 2;
        event_options.volatile_event_history_capacity = 8;
        let mut core = HostCore::with_journal(
            HostSessionId::new(80),
            world(0),
            event_options,
            Box::new(FakeJournal {
                state: Rc::clone(&state),
            }),
        )
        .expect("reservation-cancellation host");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);
        submit(&mut port, 2, HostCommand::Step);
        core.drive(ManualInstant::from_nanos(0))
            .expect("fill event ring");
        core.drive(ManualInstant::from_nanos(1))
            .expect("make full ring durable");
        assert_eq!(core.events.len(), 2);
        assert_eq!(core.events.pending_count(), 0);

        let retained_batch = Arc::clone(
            state
                .borrow()
                .attempts
                .first()
                .expect("first scientific journal batch"),
        );
        let boundary = Arc::clone(
            retained_batch
                .scientific()
                .expect("first batch scientific boundary"),
        );
        assert!(
            core.events
                .prepare_publish()
                .expect("reserve durable full-ring front")
                .is_none()
        );
        let published_before = core.events.published_total();
        core.drive(ManualInstant::from_nanos(2))
            .expect("maintenance-only drive cancels reservation");

        assert!(matches!(
            core.events.publish_pending(
                JournalBatchId::new(core.session_id, 99),
                retained_batch.applied(),
                boundary,
            ),
            Err(HostAccessError::ProtocolViolation { .. })
        ));
        assert_eq!(core.events.published_total(), published_before);
        assert_eq!(core.events.len(), 2);

        submit(&mut port, 3, HostCommand::Step);
        let legitimate = core
            .drive(ManualInstant::from_nanos(3))
            .expect("host obtains a fresh reservation");
        assert_eq!(legitimate.scientific_steps, 1);
        assert_eq!(core.events.published_total(), published_before + 1);
        assert_eq!(core.events.len(), 2);
    }

    #[test]
    fn delayed_old_journal_receipt_cannot_rewind_last_applied_command() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState {
            suppress_receipts: true,
            ..FakeJournalState::default()
        }));
        let mut core = HostCore::with_journal(
            HostSessionId::new(75),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("host with delayed receipts");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);
        submit(&mut port, 2, HostCommand::Pause);
        core.drive(ManualInstant::from_nanos(0))
            .expect("step then newer control application");
        assert_eq!(
            core.latest_snapshot().last_applied_command,
            Some(CommandId::new(2))
        );

        let old_batch = journal_state.borrow().attempts[0].id();
        journal_state
            .borrow_mut()
            .receipts
            .push_back(JournalReceipt::new(old_batch, JournalReceiptState::Durable));
        core.drive(ManualInstant::from_nanos(1))
            .expect("delayed old receipt");
        core.publish_snapshot().expect("diagnostic republish");
        assert_eq!(
            core.latest_snapshot().last_applied_command,
            Some(CommandId::new(2))
        );
    }

    #[test]
    fn scientific_boundary_survives_when_core_persistence_is_disabled() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState::default()));
        let mut core = HostCore::with_journal(
            HostSessionId::new(8),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("host with recording journal");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);
        core.drive(ManualInstant::from_nanos(0))
            .expect("recorded step");

        let journal = journal_state.borrow();
        let batch = journal.attempts.first().expect("one journal attempt");
        assert!(batch.persistence().is_none());
        let boundary = batch.scientific().expect("lossless scientific outcome");
        assert_eq!(boundary.summary().tick, Tick(1));
        assert_eq!(boundary.events().tick, Tick(1));
        assert_eq!(boundary.births().len(), boundary.summary().births);
        assert_eq!(boundary.deaths().len(), boundary.summary().deaths);
    }

    #[test]
    fn full_journal_retains_the_exact_arc_and_stops_later_science_until_retry() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState {
            full: true,
            ..FakeJournalState::default()
        }));
        let mut core = HostCore::with_journal(
            HostSessionId::new(9),
            world(1),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("host with fake journal");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);
        submit(&mut port, 2, HostCommand::Resume);

        let blocked = core
            .drive(ManualInstant::from_nanos(0))
            .expect("completed step with backpressure");
        assert_eq!(blocked.scientific_steps, 1);
        assert!(matches!(
            blocked.blocker,
            Some(HostBlocker::JournalFull { .. })
        ));
        assert_eq!(core.world_tick(), Tick(1));
        assert_eq!(port.queue_depth(), 1);
        assert_eq!(status(&mut port, 1).journal(), &JournalState::Pending);

        let retained = core.pending_journal_batch().expect("exact retained batch");
        assert_eq!(
            retained
                .scientific()
                .expect("lossless scientific boundary")
                .summary()
                .tick,
            Tick(1)
        );
        assert!(retained.persistence().is_some());
        assert!(Arc::ptr_eq(&retained, &journal_state.borrow().attempts[0]));

        let blocked_now = DEFAULT_TICK_PERIOD_NANOS.saturating_mul(4);
        let still_blocked = core
            .drive(ManualInstant::from_nanos(blocked_now))
            .expect("elapsed time cannot bypass retained journal work");
        assert_eq!(still_blocked.scientific_steps, 0);
        assert_eq!(core.world_tick(), Tick(1));
        assert_eq!(port.queue_depth(), 1);
        assert!(Arc::ptr_eq(
            &retained,
            &core
                .pending_journal_batch()
                .expect("same batch remains retained")
        ));

        journal_state.borrow_mut().full = false;
        let retried = core
            .retry_retained_journal()
            .expect("retry call")
            .expect("retained work existed");
        assert!(retried.is_accepted());
        assert!(Arc::ptr_eq(
            &journal_state.borrow().attempts[0],
            &journal_state.borrow().attempts[1]
        ));

        let resumed = core
            .drive(ManualInstant::from_nanos(blocked_now.saturating_add(1)))
            .expect("receipt and queued resume");
        assert_eq!(resumed.scientific_steps, 0);
        assert_eq!(core.world_tick(), Tick(1));
        assert!(!core.latest_snapshot().playback.paused);
        assert_eq!(status(&mut port, 1).journal(), &JournalState::Durable);
    }

    #[test]
    fn closed_journal_is_terminal_on_the_journal_axis_and_queryable_in_health() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState {
            closed: true,
            ..FakeJournalState::default()
        }));
        let mut core = HostCore::with_journal(
            HostSessionId::new(10),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: journal_state,
            }),
        )
        .expect("host with closed journal");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);

        let receipt = core
            .drive(ManualInstant::from_nanos(0))
            .expect("step completed before closed journal result");
        assert_eq!(receipt.scientific_steps, 1);
        assert!(matches!(
            receipt.blocker,
            Some(HostBlocker::JournalClosed { .. })
        ));
        assert!(matches!(
            status(&mut port, 1).journal(),
            JournalState::Failed(_)
        ));
        assert!(matches!(
            core.health(),
            HostHealth::Faulted(HostFault::Journal { .. })
        ));
        assert!(core.pending_journal_batch().is_some());
    }

    #[test]
    fn shutdown_closes_admission_in_order_and_stops_after_its_receipt() {
        let (mut core, mut port) = host(true);
        let shutdown = submit(&mut port, 1, HostCommand::Shutdown);
        assert!(matches!(shutdown.application(), ApplicationState::Admitted));
        let rejected = submit(&mut port, 2, HostCommand::Pause);
        assert!(matches!(
            rejected.application(),
            ApplicationState::Rejected(RejectionReason::HostStopping)
        ));

        core.drive(ManualInstant::from_nanos(0))
            .expect("shutdown application");
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopping);
        core.drive(ManualInstant::from_nanos(1))
            .expect("volatile shutdown receipt");
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);
        assert_eq!(
            status(&mut port, 1).journal(),
            &JournalState::CommittedVolatile
        );
    }

    #[test]
    fn durable_adapter_shutdown_waits_past_a_volatile_receipt() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState {
            suppress_receipts: true,
            ..FakeJournalState::default()
        }));
        let mut core = HostCore::with_journal(
            HostSessionId::new(12),
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("host with progressive durable journal");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Shutdown);
        core.drive(ManualInstant::from_nanos(0))
            .expect("shutdown application");

        let shutdown_id = journal_state.borrow().attempts[0].id();
        journal_state
            .borrow_mut()
            .receipts
            .push_back(JournalReceipt::new(
                shutdown_id,
                JournalReceiptState::CommittedVolatile,
            ));
        core.drive(ManualInstant::from_nanos(1))
            .expect("progressive volatile receipt");
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopping);
        assert_eq!(
            status(&mut port, 1).journal(),
            &JournalState::CommittedVolatile
        );

        journal_state
            .borrow_mut()
            .receipts
            .push_back(JournalReceipt::new(
                shutdown_id,
                JournalReceiptState::Durable,
            ));
        core.drive(ManualInstant::from_nanos(2))
            .expect("durable shutdown receipt");
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);
        assert_eq!(status(&mut port, 1).journal(), &JournalState::Durable);
    }

    #[test]
    fn earlier_journal_failure_blocks_shutdown_even_if_health_fault_is_overwritten() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState {
            suppress_receipts: true,
            ..FakeJournalState::default()
        }));
        let session_id = HostSessionId::new(13);
        let mut core = HostCore::with_journal(
            session_id,
            world(0),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("host with manually acknowledged journal");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);
        submit(&mut port, 2, HostCommand::Shutdown);
        core.drive(ManualInstant::from_nanos(0))
            .expect("ordered step and shutdown application");

        let earlier_id = journal_state.borrow().attempts[0].id();
        let shutdown_id = journal_state.borrow().attempts[1].id();
        journal_state.borrow_mut().receipts.extend([
            JournalReceipt::new(
                earlier_id,
                JournalReceiptState::Failed(JournalFailure {
                    code: "write_failed".to_owned(),
                    message: "injected terminal failure".to_owned(),
                }),
            ),
            JournalReceipt::new(
                JournalBatchId::new(session_id, 999),
                JournalReceiptState::Durable,
            ),
            JournalReceipt::new(shutdown_id, JournalReceiptState::Durable),
        ]);
        core.drive(ManualInstant::from_nanos(1))
            .expect("failure, malformed receipt, and shutdown durability");

        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopping);
        assert!(matches!(
            status(&mut port, 1).journal(),
            JournalState::Failed(_)
        ));
        assert_eq!(status(&mut port, 2).journal(), &JournalState::Durable);
        assert!(matches!(
            core.health(),
            HostHealth::Faulted(HostFault::Protocol { .. })
        ));
    }

    #[test]
    fn stale_shutdown_cas_reopens_ingress_without_advancing_control() {
        let (mut core, mut port) = host(true);
        port.submit(
            envelope(1, HostCommand::Shutdown).expecting_control_revision(ControlRevision::new(7)),
        )
        .expect("guarded shutdown admission");
        core.drive(ManualInstant::from_nanos(0))
            .expect("shutdown conflict");

        assert!(matches!(
            status(&mut port, 1).application(),
            ApplicationState::Rejected(RejectionReason::ControlRevisionConflict { .. })
        ));
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Running);
        assert_eq!(
            core.latest_snapshot().revisions.control,
            ControlRevision::new(0)
        );
        assert!(matches!(
            submit(&mut port, 2, HostCommand::Pause).application(),
            ApplicationState::Admitted
        ));
    }

    #[test]
    fn shutdown_journals_the_exact_partial_persistence_tail() {
        let journal_state = Rc::new(RefCell::new(FakeJournalState::default()));
        let mut core = HostCore::with_journal(
            HostSessionId::new(11),
            world(3),
            options(true),
            Box::new(FakeJournal {
                state: Rc::clone(&journal_state),
            }),
        )
        .expect("host with cadence journal");
        let mut port = core.local_port();
        submit(&mut port, 1, HostCommand::Step);
        submit(&mut port, 2, HostCommand::Step);
        core.drive(ManualInstant::from_nanos(0))
            .expect("two deferred persistence steps");
        journal_state.borrow_mut().full = true;
        submit(&mut port, 3, HostCommand::Shutdown);

        let receipt = core
            .drive(ManualInstant::from_nanos(1))
            .expect("shutdown tail backpressure");
        assert!(matches!(
            receipt.blocker,
            Some(HostBlocker::JournalFull { .. })
        ));
        let retained = core
            .pending_journal_batch()
            .expect("shutdown tail retained");
        assert!(
            retained
                .command()
                .is_some_and(|command| matches!(&command.command, HostCommand::Shutdown))
        );
        assert_eq!(
            retained
                .persistence()
                .expect("partial cadence persistence payload")
                .summary
                .tick,
            Tick(2)
        );
        assert!(Arc::ptr_eq(
            &retained,
            journal_state
                .borrow()
                .attempts
                .last()
                .expect("shutdown admission attempt")
        ));
    }

    #[test]
    fn backwards_time_is_rejected_before_any_state_change() {
        let (mut core, _port) = host(true);
        core.drive(ManualInstant::from_nanos(5))
            .expect("initial boundary");
        let before = core.latest_snapshot();
        assert!(matches!(
            core.drive(ManualInstant::from_nanos(4)),
            Err(HostAccessError::ProtocolViolation { .. })
        ));
        assert_eq!(core.latest_snapshot(), before);
    }

    #[test]
    fn exhausted_journal_sequence_fails_before_config_or_science_mutation() {
        let (mut config_core, mut config_port) = host(true);
        let config_before = config_core.world.config().clone();
        let snapshot_before = config_core.latest_snapshot();
        let digest_before = config_core
            .scientific_digest_v1()
            .expect("pre-exhaustion config digest");
        let mut changed_config = config_before.clone();
        changed_config.food_growth_rate = 0.03125;
        config_core.next_journal_sequence = u64::MAX;
        submit(
            &mut config_port,
            1,
            HostCommand::UpdateConfig(Box::new(changed_config)),
        );
        assert!(matches!(
            config_core.drive(ManualInstant::from_nanos(0)),
            Err(HostAccessError::ProtocolViolation { .. })
        ));
        assert_eq!(config_core.world.config(), &config_before);
        assert_eq!(config_core.world_tick(), Tick(0));
        assert_eq!(config_core.latest_snapshot(), snapshot_before);
        assert_eq!(
            config_core
                .scientific_digest_v1()
                .expect("post-exhaustion config digest"),
            digest_before
        );
        assert_eq!(config_core.events.published_total(), 0);

        let (mut step_core, mut step_port) = host(true);
        let step_snapshot_before = step_core.latest_snapshot();
        let step_digest_before = step_core
            .scientific_digest_v1()
            .expect("pre-exhaustion step digest");
        step_core.next_journal_sequence = u64::MAX;
        submit(&mut step_port, 1, HostCommand::Step);
        assert!(matches!(
            step_core.drive(ManualInstant::from_nanos(0)),
            Err(HostAccessError::ProtocolViolation { .. })
        ));
        assert_eq!(step_core.world_tick(), Tick(0));
        assert_eq!(step_core.latest_snapshot(), step_snapshot_before);
        assert_eq!(
            step_core
                .scientific_digest_v1()
                .expect("post-exhaustion step digest"),
            step_digest_before
        );
        assert_eq!(step_core.events.published_total(), 0);
    }

    #[test]
    fn lifecycle_shutdown_uses_one_reserved_ordered_slot_and_is_idempotent() {
        let mut constrained = options(true);
        constrained.command_capacity = 1;
        let mut core =
            HostCore::new(HostSessionId::new(21), world(0), constrained).expect("constrained host");
        let mut port = core.local_port();
        let first = submit(&mut port, 1, HostCommand::Pause);
        assert_eq!(first.admission_sequence(), Some(AdmissionSequence::new(1)));

        let shutdown = core.request_shutdown().expect("reserved shutdown slot");
        assert_eq!(
            shutdown.admission_sequence(),
            Some(AdmissionSequence::new(2))
        );
        let repeated = core.request_shutdown().expect("idempotent shutdown");
        assert_eq!(repeated, shutdown);
        assert_eq!(core.shutdown_command_id(), Some(shutdown.command_id()));
        assert!(matches!(
            submit(&mut port, 2, HostCommand::Resume).application(),
            ApplicationState::Rejected(RejectionReason::HostStopping)
        ));

        let applied = core
            .drive(ManualInstant::from_nanos(0))
            .expect("ordered pause then shutdown");
        assert_eq!(applied.commands_completed, 2);
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopping);
        core.drive(ManualInstant::from_nanos(0))
            .expect("volatile shutdown barrier");
        assert_eq!(core.latest_snapshot().lifecycle, HostLifecycle::Stopped);
    }

    #[test]
    fn bounded_catch_up_reports_skips_and_preserves_fractional_credit() {
        let (mut core, _port) = host(false);
        core.drive(ManualInstant::from_nanos(0))
            .expect("initial boundary");

        let late = core
            .drive(ManualInstant::from_nanos(105))
            .expect("bounded late boundary");
        assert_eq!(late.automatic_steps_due, 10);
        assert_eq!(late.automatic_steps_skipped, 6);
        assert_eq!(late.scientific_steps, 4);
        assert_eq!(core.world_tick(), Tick(4));

        let fractional = core
            .drive(ManualInstant::from_nanos(110))
            .expect("fractional credit boundary");
        assert_eq!(fractional.automatic_steps_due, 1);
        assert_eq!(fractional.automatic_steps_skipped, 0);
        assert_eq!(fractional.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(5));
    }

    #[test]
    fn drive_interest_disarms_paused_science_and_exposes_immediate_work() {
        let (mut core, mut port) = host(true);
        assert_eq!(core.drive_interest(), HostDriveInterest::WakeOnly);
        submit(&mut port, 1, HostCommand::Step);
        assert_eq!(core.drive_interest(), HostDriveInterest::ReadyNow);
        core.drive(ManualInstant::from_nanos(0))
            .expect("explicit step");
        assert_eq!(core.drive_interest(), HostDriveInterest::Draining);
        core.drive(ManualInstant::from_nanos(0))
            .expect("volatile step receipt");
        assert_eq!(core.drive_interest(), HostDriveInterest::WakeOnly);
        assert!(core.scientific_digest_v1().is_ok());
    }

    #[test]
    fn host_core_source_has_no_shared_mutable_world_escape_hatch() {
        let source = include_str!("host_core.rs");
        let shared_world_lock = ["Arc<", "Mutex<", "WorldState>>"].concat();
        let thread_spawn = ["thread::", "spawn"].concat();
        assert!(!source.contains(&shared_world_lock));
        assert!(!source.contains(&thread_spawn));
    }
}
