//! Nonblocking HostCore journal adapter and detached event-reader surface.

use super::{
    AdmissionState, CommandAuthorityRequest, DEFAULT_COMMAND_CAPACITY, ExistingStorageLease,
    InFlightPermit, MAX_COMMAND_ENVELOPE_BYTES, MAX_STORAGE_QUERY_PAGE, MAX_STORAGE_WAIT_TIMEOUT,
    Storage, StorageBuffer, StorageCommand, StorageError, load_host_journal_index,
    read_host_journal_events,
};
use arc_swap::ArcSwap;
use crossbeam_channel as xchan;
use scriptbots_core::{
    AgentUid, BirthRecord, DeathRecord, ScriptBotsConfig, SelectionUpdate, Tick, TickCombatSummary,
};
use scriptbots_runtime::{
    AdmissionSequence, ApplicationFailure, ApplicationState, AppliedCommand,
    COMMAND_LIFECYCLE_SCHEMA_VERSION, CommandAuthorityLookup, CommandAuthorityLookupFailure,
    CommandAuthorityReader, CommandClaimPolicy, CommandEnvelope, CommandId,
    CommandLifecycleEvidence, CommandLifecycleTransition, ConfigRevision, ControlRevision,
    EventCatchUp, EventCatchUpGuarantee, EventCatchUpLocator, EventCatchUpUnavailableReason,
    EventCommitment, EventJournalReader, EventPage, EventPageSource, EventRetentionSnapshot,
    EventSequence, EventSequenceRange, HostAccessError, HostCommand, HostRevisions, HostSessionId,
    JournalAdmission, JournalBatch, JournalBatchId, JournalFailure, JournalPort, JournalReceipt,
    JournalReceiptState, JournaledScientificEvent, RejectionReason, RunId, ScientificBoundary,
    ScientificEvent, ScientificRevision, ShutdownCommitRequirement,
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    io::Write,
    sync::{
        Arc, Mutex, TryLockError,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

const DEFAULT_JOURNAL_CAPACITY: usize = DEFAULT_COMMAND_CAPACITY;
const DEFAULT_JOURNAL_RECEIPT_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_JOURNAL_BATCH_BYTES: usize = 64 << 20;
const DEFAULT_JOURNAL_INFLIGHT_BYTES: usize = 256 << 20;
const DEFAULT_JOURNAL_IDENTITY_CAPACITY: usize = 512;
const DEFAULT_EVENT_PAGE_BYTES: usize = 64 << 20;
const MAX_JOURNAL_BYTES: usize = 1 << 30;
pub(super) const HOST_JOURNAL_ARCHIVE_VERSION: u32 = 2;

/// Bounded admission and catch-up limits for one HostCore journal adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageJournalOptions {
    /// Most accepted batches that may await receipt polling.
    pub admission_capacity: usize,
    /// Monotonic age at which an unresolved batch becomes eligible for timeout resolution on the
    /// next nonblocking receipt poll.
    ///
    /// This is not an autonomous timer: expiry is observed only when the host polls, and
    /// contention on the authoritative terminal-truth cache may defer resolution until a later
    /// uncontended poll.
    pub receipt_timeout: Duration,
    /// Largest exact [`JournalBatch`] allocation accepted by the adapter.
    pub max_batch_bytes: usize,
    /// Largest total accepted journal and durable-authority allocation awaiting worker progress.
    pub max_inflight_bytes: usize,
    /// Recent durable event identities retained for nonblocking eviction checks.
    pub event_cache_capacity: usize,
    /// Largest encoded event page and retained in-memory catch-up window.
    pub max_event_page_bytes: usize,
}

impl Default for StorageJournalOptions {
    fn default() -> Self {
        Self {
            admission_capacity: DEFAULT_JOURNAL_CAPACITY,
            receipt_timeout: DEFAULT_JOURNAL_RECEIPT_TIMEOUT,
            max_batch_bytes: DEFAULT_JOURNAL_BATCH_BYTES,
            max_inflight_bytes: DEFAULT_JOURNAL_INFLIGHT_BYTES,
            event_cache_capacity: DEFAULT_JOURNAL_IDENTITY_CAPACITY,
            max_event_page_bytes: DEFAULT_EVENT_PAGE_BYTES,
        }
    }
}

impl StorageJournalOptions {
    pub(super) fn validate(self) -> Result<Self, &'static str> {
        if self.admission_capacity == 0 {
            return Err("journal admission_capacity must be nonzero");
        }
        if self.admission_capacity > DEFAULT_COMMAND_CAPACITY {
            return Err("journal admission_capacity exceeds the storage command lane");
        }
        if self.receipt_timeout.is_zero() {
            return Err("journal receipt_timeout must be nonzero");
        }
        if self.receipt_timeout > MAX_STORAGE_WAIT_TIMEOUT {
            return Err("journal receipt_timeout exceeds the storage wait ceiling");
        }
        if self.max_batch_bytes == 0 {
            return Err("journal max_batch_bytes must be nonzero");
        }
        if self.max_batch_bytes > MAX_JOURNAL_BYTES {
            return Err("journal max_batch_bytes exceeds the hard allocation ceiling");
        }
        if self.max_inflight_bytes < self.max_batch_bytes {
            return Err("journal max_inflight_bytes must cover at least one maximum batch");
        }
        if self.max_inflight_bytes > MAX_JOURNAL_BYTES {
            return Err("journal max_inflight_bytes exceeds the hard allocation ceiling");
        }
        if self.event_cache_capacity == 0 {
            return Err("journal event_cache_capacity must be nonzero");
        }
        if self.event_cache_capacity > MAX_STORAGE_QUERY_PAGE {
            return Err("journal event_cache_capacity exceeds the bounded query ceiling");
        }
        if self.max_event_page_bytes < self.max_batch_bytes {
            return Err("journal max_event_page_bytes must cover at least one maximum batch");
        }
        if self.max_event_page_bytes > MAX_JOURNAL_BYTES {
            return Err("journal max_event_page_bytes exceeds the hard allocation ceiling");
        }
        Ok(self)
    }
}

/// Monotonic prefixes proven for either host-journal batches or scientific events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostJournalPrefixes {
    /// Highest contiguously admitted sequence.
    pub admitted: u64,
    /// Highest contiguously applied sequence.
    pub applied: u64,
    /// Highest contiguous sequence committed to volatile storage.
    pub committed_volatile: u64,
    /// Highest crash-durable contiguous sequence.
    pub durable: u64,
}

/// Immutable progress for one validated host-journal session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostJournalSessionProgress {
    /// Admission, application, volatile-commit, and durability prefixes for journal batches.
    pub journal: HostJournalPrefixes,
    /// Admission, application, volatile-commit, and durability prefixes for scientific events.
    pub events: HostJournalPrefixes,
    /// Final ordered shutdown batch, when the session durably reached shutdown.
    pub shutdown: Option<JournalBatchId>,
}

/// Persisted state of one canonical host-journal record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostJournalRecordState {
    /// The exact archive and ledger identity were admitted atomically.
    Admitted,
    /// The archived boundary was applied to its durable scientific tables.
    Applied,
    /// Application committed, but file durability was not yet established.
    CommittedVolatile,
    /// The exact record and every preceding record are crash durable.
    Durable,
}

impl HostJournalRecordState {
    const fn event_commitment(self) -> EventCommitment {
        match self {
            Self::Admitted | Self::Applied => EventCommitment::Pending,
            Self::CommittedVolatile => EventCommitment::CommittedVolatile,
            Self::Durable => EventCommitment::Durable,
        }
    }
}

/// Typed public projection of one digest- and canonical-form-validated host-journal archive.
#[derive(Debug, Clone)]
pub struct HostJournalRecord {
    /// Stable session-scoped archive identity and total journal order.
    pub batch_id: JournalBatchId,
    /// Complete validated command application lifecycle, when command-driven.
    pub command_lifecycle: Option<CommandLifecycleEvidence>,
    /// Exact terminal tick and revision boundary of this record.
    ///
    /// For rejected and failed commands this is an observation boundary, not an application.
    pub applied: AppliedCommand,
    /// Complete canonical scientific event, when this record advanced science.
    pub event: Option<JournaledScientificEvent>,
    /// Persisted ledger state cross-checked against the session prefixes.
    pub state: HostJournalRecordState,
}

/// Storage-ledger transition recorded after a command's application lifecycle was archived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandStorageTransitionKind {
    /// The archive transaction committed to the configured volatile storage boundary.
    CommittedVolatile,
    /// The archive reached the file-backed crash-durability boundary.
    Durable,
}

impl CommandStorageTransitionKind {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::CommittedVolatile => "committed_volatile",
            Self::Durable => "durable",
        }
    }

    pub(super) fn decode(value: &str) -> Result<Self, StorageError> {
        match value {
            "committed_volatile" => Ok(Self::CommittedVolatile),
            "durable" => Ok(Self::Durable),
            _ => Err(StorageError::InvalidData {
                context: "host_command_storage_transitions.storage_state",
                reason: format!("unknown command storage transition {value:?}"),
            }),
        }
    }
}

/// One ordered storage-ledger transition bound to a command's canonical archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandStorageTransition {
    /// Zero-based position on the storage axis.
    pub ordinal: u32,
    /// Storage state established at this transition.
    pub kind: CommandStorageTransitionKind,
}

/// Exact cursor for one command row in total host-journal order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandJournalCursor {
    /// Host session that owns the command.
    pub session_id: HostSessionId,
    /// Total host-journal position of the command archive.
    pub journal_sequence: u64,
    /// Stable idempotency identity at that exact position.
    pub command_id: CommandId,
}

/// Fully validated normalized command evidence from one canonical host archive.
#[derive(Debug, Clone, PartialEq)]
pub struct CommandJournalRecord {
    /// Total host-journal identity whose canonical archive supplied this command.
    pub batch_id: JournalBatchId,
    /// Complete runtime-validated application lifecycle.
    pub lifecycle: CommandLifecycleEvidence,
    /// Terminal tick and revisions, whether applied, rejected, or failed.
    pub terminal_boundary: AppliedCommand,
    /// Scientific event produced by an applied step, when present.
    pub scientific_event_sequence: Option<EventSequence>,
    /// Ordered storage-ledger transitions, distinct from application transitions.
    pub storage_transitions: Vec<CommandStorageTransition>,
    /// BLAKE3 digest of the canonical host-journal archive payload.
    ///
    /// This binds the projection to its source archive. It is not a world-state digest.
    pub archive_payload_digest: String,
}

impl CommandJournalRecord {
    /// Exact cursor that resumes after this command row.
    #[must_use]
    pub fn cursor(&self) -> CommandJournalCursor {
        CommandJournalCursor {
            session_id: self.batch_id.session_id(),
            journal_sequence: self.batch_id.sequence(),
            command_id: self.lifecycle.envelope().command_id,
        }
    }
}

/// Complete validated command-projection counts for one finished host session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandJournalEvidence {
    /// Number of normalized command records.
    pub command_count: u64,
    /// Number of ordered runtime application transitions.
    pub application_transition_count: u64,
    /// Number of ordered storage-ledger transitions.
    pub storage_transition_count: u64,
}

/// One bounded page of normalized command evidence in host-journal order.
#[derive(Debug, Clone, PartialEq)]
pub struct CommandJournalPage {
    /// Durable run selected by the finished storage reader.
    pub run_id: RunId,
    /// Host session whose commands this page describes.
    pub session_id: HostSessionId,
    /// Commands ordered by total host-journal sequence.
    pub commands: Vec<CommandJournalRecord>,
    /// Exact cursor for the next page, or `None` at the tip.
    pub next_after: Option<CommandJournalCursor>,
    /// Complete non-vacuous evidence counts for the session.
    pub evidence: CommandJournalEvidence,
}

/// Successful result of the offline FrankenSQLite integrity conformance gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageIntegrityCheckResult {
    /// `PRAGMA integrity_check` returned exactly `ok`.
    Ok,
}

/// One bounded, immutable page from a validated finished host-journal session.
#[derive(Debug, Clone)]
pub struct HostJournalSessionPage {
    /// Durable run selected by the finished storage reader.
    pub run_id: RunId,
    /// Host session whose records and progress this page describes.
    pub session_id: HostSessionId,
    /// Complete session prefixes and ordered shutdown marker.
    pub progress: HostJournalSessionProgress,
    /// Canonically decoded records in strictly increasing journal order.
    pub records: Vec<HostJournalRecord>,
    /// Cursor to pass as `after` for the next page, or `None` when this page reaches the tip.
    pub next_after: Option<JournalBatchId>,
    /// Typed evidence that the offline integrity conformance gate passed.
    pub integrity_check: StorageIntegrityCheckResult,
}

/// Stable normalized kind of one scientific domain event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainEventKind {
    /// Complete agent-arrival record from the scientific boundary.
    Birth,
    /// Complete agent-removal record from the scientific boundary.
    Death,
    /// Nonzero aggregate combat counters for one scientific tick.
    Combat,
}

impl DomainEventKind {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Birth => "birth",
            Self::Death => "death",
            Self::Combat => "combat",
        }
    }

    pub(super) fn decode(value: &str) -> Result<Self, StorageError> {
        match value {
            "birth" => Ok(Self::Birth),
            "death" => Ok(Self::Death),
            "combat" => Ok(Self::Combat),
            _ => Err(StorageError::InvalidData {
                context: "host_domain_events.kind",
                reason: format!("unknown normalized domain-event kind {value:?}"),
            }),
        }
    }
}

/// Complete typed payload retained for one normalized domain event.
#[derive(Debug, Clone, PartialEq)]
pub enum DomainEventPayload {
    /// Full birth/arrival record, including lineage and origin metadata.
    Birth(BirthRecord),
    /// Full death record, including cause and combat flags.
    Death(DeathRecord),
    /// Aggregate combat counters for the tick. Pairwise edges belong to the interaction journal.
    Combat(TickCombatSummary),
}

impl DomainEventPayload {
    /// Stable normalized kind corresponding to this payload.
    #[must_use]
    pub const fn kind(&self) -> DomainEventKind {
        match self {
            Self::Birth(_) => DomainEventKind::Birth,
            Self::Death(_) => DomainEventKind::Death,
            Self::Combat(_) => DomainEventKind::Combat,
        }
    }

    pub(super) const fn actor_agent_uid(&self) -> Option<AgentUid> {
        match self {
            Self::Birth(record) => Some(record.agent_uid),
            Self::Death(record) => Some(record.agent_uid),
            Self::Combat(_) => None,
        }
    }

    pub(super) fn encode_json(&self) -> Result<String, StorageError> {
        let encoded = match self {
            Self::Birth(record) => serde_json::to_string(record),
            Self::Death(record) => serde_json::to_string(record),
            Self::Combat(record) => serde_json::to_string(record),
        };
        encoded.map_err(|error| StorageError::InvalidData {
            context: "host_domain_events.payload_json",
            reason: error.to_string(),
        })
    }

    pub(super) fn decode_json(
        kind: DomainEventKind,
        payload_json: &str,
    ) -> Result<Self, StorageError> {
        let payload = match kind {
            DomainEventKind::Birth => serde_json::from_str(payload_json).map(Self::Birth),
            DomainEventKind::Death => serde_json::from_str(payload_json).map(Self::Death),
            DomainEventKind::Combat => serde_json::from_str(payload_json).map(Self::Combat),
        }
        .map_err(|error| StorageError::InvalidData {
            context: "host_domain_events.payload_json",
            reason: error.to_string(),
        })?;
        if payload.encode_json()? != payload_json {
            return Err(StorageError::InvalidData {
                context: "host_domain_events.payload_json",
                reason: "payload is valid JSON but not in canonical normalized form".to_owned(),
            });
        }
        Ok(payload)
    }
}

/// Exact cursor for one normalized domain-event row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainEventCursor {
    /// Host session that owns the scientific sequence.
    pub session_id: HostSessionId,
    /// Contiguous scientific boundary sequence.
    pub scientific_event_sequence: EventSequence,
    /// Zero-based deterministic ordinal within that boundary.
    pub event_ordinal: u64,
}

/// One normalized lifecycle or aggregate-combat event bound to its canonical archive.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainEventRecord {
    /// Host session that owns this event.
    pub session_id: HostSessionId,
    /// Contiguous scientific boundary sequence.
    pub scientific_event_sequence: EventSequence,
    /// Zero-based deterministic ordinal within the scientific boundary.
    pub event_ordinal: u64,
    /// Total journal identity whose canonical archive supplied this row.
    pub journal_batch_id: JournalBatchId,
    /// Scientific tick recorded by the canonical boundary.
    pub tick: Tick,
    /// Full typed domain payload.
    pub payload: DomainEventPayload,
    /// BLAKE3 digest of the canonical host-journal archive payload.
    ///
    /// This binds the projection to its source archive. It is not a world-state digest.
    pub archive_payload_digest: String,
}

impl DomainEventRecord {
    /// Cursor that resumes immediately after this exact row.
    #[must_use]
    pub const fn cursor(&self) -> DomainEventCursor {
        DomainEventCursor {
            session_id: self.session_id,
            scientific_event_sequence: self.scientific_event_sequence,
            event_ordinal: self.event_ordinal,
        }
    }
}

/// Whether a domain-evidence query permits a genuinely empty projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainEventExpectation {
    /// Empty evidence is valid for scenarios that do not promise domain events.
    AllowEmpty,
    /// At least one normalized domain event is required; zero rows fail closed.
    RequireNonEmpty,
}

/// Validated counts for one finished host session's normalized domain evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainEventEvidence {
    /// Number of contiguous durable scientific boundaries covered by projections.
    pub scientific_event_count: u64,
    /// Total normalized birth, death, and aggregate-combat rows.
    pub domain_event_count: u64,
}

/// One bounded page of normalized domain events in scientific-sequence order.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainEventPage {
    /// Durable run selected by the finished storage reader.
    pub run_id: RunId,
    /// Host session whose normalized events this page describes.
    pub session_id: HostSessionId,
    /// Rows ordered by scientific sequence then deterministic local ordinal.
    pub events: Vec<DomainEventRecord>,
    /// Cursor for the next page, or `None` when the durable projection is exhausted.
    pub next_after: Option<DomainEventCursor>,
    /// Complete validated evidence counts for the session.
    pub evidence: DomainEventEvidence,
}

#[derive(Debug, Clone)]
pub(super) struct PreparedDomainEvent {
    pub(super) ordinal: u64,
    pub(super) payload: DomainEventPayload,
    pub(super) payload_json: String,
}

#[derive(Debug, Clone)]
pub(super) struct PreparedDomainEventProjection {
    pub(super) batch_id: JournalBatchId,
    pub(super) scientific_event_sequence: EventSequence,
    pub(super) tick: Tick,
    pub(super) archive_payload_digest: String,
    pub(super) events: Vec<PreparedDomainEvent>,
}

impl PreparedDomainEventProjection {
    fn new(
        batch_id: JournalBatchId,
        scientific_event_sequence: EventSequence,
        tick: Tick,
        scientific: &ScientificBoundary,
        archive_payload_digest: &str,
    ) -> Result<Self, StorageError> {
        let combat = scientific.combat();
        let event_capacity = scientific
            .births()
            .len()
            .saturating_add(scientific.deaths().len())
            .saturating_add(usize::from(
                combat.spike_attempts != 0 || combat.spike_hits != 0,
            ));
        let mut events = Vec::with_capacity(event_capacity);
        for birth in scientific.births() {
            let payload = DomainEventPayload::Birth(birth.clone());
            let payload_json = payload.encode_json()?;
            events.push(PreparedDomainEvent {
                ordinal: u64::try_from(events.len()).map_err(|error| {
                    StorageError::InvalidData {
                        context: "host_domain_events.event_ordinal",
                        reason: error.to_string(),
                    }
                })?,
                payload,
                payload_json,
            });
        }
        for death in scientific.deaths() {
            let payload = DomainEventPayload::Death(death.clone());
            let payload_json = payload.encode_json()?;
            events.push(PreparedDomainEvent {
                ordinal: u64::try_from(events.len()).map_err(|error| {
                    StorageError::InvalidData {
                        context: "host_domain_events.event_ordinal",
                        reason: error.to_string(),
                    }
                })?,
                payload,
                payload_json,
            });
        }
        if combat.spike_attempts != 0 || combat.spike_hits != 0 {
            let payload = DomainEventPayload::Combat(combat);
            let payload_json = payload.encode_json()?;
            events.push(PreparedDomainEvent {
                ordinal: u64::try_from(events.len()).map_err(|error| {
                    StorageError::InvalidData {
                        context: "host_domain_events.event_ordinal",
                        reason: error.to_string(),
                    }
                })?,
                payload,
                payload_json,
            });
        }
        Ok(Self {
            batch_id,
            scientific_event_sequence,
            tick,
            archive_payload_digest: archive_payload_digest.to_owned(),
            events,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct PreparedCommandApplicationTransition {
    pub(super) ordinal: u32,
    pub(super) boundary: AppliedCommand,
    pub(super) application: ApplicationState,
    pub(super) application_postcard_hex: String,
}

#[derive(Debug, Clone)]
pub(super) struct PreparedCommandProjection {
    pub(super) batch_id: JournalBatchId,
    pub(super) lifecycle: CommandLifecycleEvidence,
    pub(super) envelope_postcard_hex: String,
    pub(super) command_payload_postcard_hex: String,
    pub(super) terminal_boundary: AppliedCommand,
    pub(super) scientific_event_sequence: Option<EventSequence>,
    pub(super) archive_payload_digest: String,
    pub(super) application_transitions: Vec<PreparedCommandApplicationTransition>,
}

impl PreparedCommandProjection {
    fn new(
        batch_id: JournalBatchId,
        lifecycle: &CommandLifecycleEvidence,
        terminal_boundary: AppliedCommand,
        scientific_event_sequence: Option<EventSequence>,
        archive_payload_digest: &str,
    ) -> Result<Self, StorageError> {
        lifecycle
            .validate()
            .map_err(|error| StorageError::InvalidData {
                context: "host_command_records.lifecycle",
                reason: error.to_string(),
            })?;
        let terminal = lifecycle.terminal().ok_or(StorageError::InvalidData {
            context: "host_command_records.terminal_boundary",
            reason: "validated lifecycle has no terminal transition".to_owned(),
        })?;
        if terminal.boundary() != terminal_boundary {
            return Err(StorageError::InvalidData {
                context: "host_command_records.terminal_boundary",
                reason: "command terminal boundary differs from its host-journal archive"
                    .to_owned(),
            });
        }
        let envelope_postcard_hex = encode_command_envelope_postcard_hex(
            "host_command_records.envelope_postcard_hex",
            lifecycle.envelope(),
        )?;
        let command_payload_postcard_hex = encode_host_command_postcard_hex(
            "host_command_records.command_payload_postcard_hex",
            &lifecycle.envelope().command,
        )?;
        let application_transitions = lifecycle
            .transitions()
            .iter()
            .map(|transition| {
                let application_postcard_hex = encode_application_state_postcard_hex(
                    "host_command_application_transitions.application_postcard_hex",
                    transition.application(),
                )?;
                Ok(PreparedCommandApplicationTransition {
                    ordinal: transition.ordinal(),
                    boundary: transition.boundary(),
                    application: transition.application().clone(),
                    application_postcard_hex,
                })
            })
            .collect::<Result<Vec<_>, StorageError>>()?;
        Ok(Self {
            batch_id,
            lifecycle: lifecycle.clone(),
            envelope_postcard_hex,
            command_payload_postcard_hex,
            terminal_boundary,
            scientific_event_sequence,
            archive_payload_digest: archive_payload_digest.to_owned(),
            application_transitions,
        })
    }

    pub(super) const fn command_id(&self) -> CommandId {
        self.lifecycle.envelope().command_id
    }
}

/// Postcard-safe wire mirror for [`HostCommand`].
///
/// The runtime enum is internally tagged for its JSON protocol, a representation
/// that asks binary deserializers to implement `deserialize_any`. Postcard
/// deliberately does not. This externally tagged mirror keeps the durable binary
/// contract independent of that JSON representation and stores `SetSpeed` by raw
/// IEEE-754 bits so even a rejected non-finite request remains exact evidence.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
enum HostCommandPostcardV1 {
    Pause,
    Resume,
    SetSpeedBits(u32),
    Step,
    UpdateConfig(Box<ScriptBotsConfig>),
    Shutdown,
    // The durable discriminant order is append-only. New command kinds belong
    // after the original six variants so existing V1 archives remain decodable.
    UpdateSelection(SelectionUpdate),
    AdjustAgentMutationRates {
        agent_uid: u64,
        delta_primary_bits: u32,
        delta_secondary_bits: u32,
    },
    SpawnAgent {
        herbivore_tendency_bits: u32,
    },
    SpawnCrossover {
        parent_a: u64,
        parent_b: u64,
    },
}

impl HostCommandPostcardV1 {
    fn from_runtime(command: &HostCommand) -> Self {
        match command {
            HostCommand::Pause => Self::Pause,
            HostCommand::Resume => Self::Resume,
            HostCommand::SetSpeed(speed) => Self::SetSpeedBits(speed.to_bits()),
            HostCommand::Step => Self::Step,
            HostCommand::UpdateConfig(config) => Self::UpdateConfig(config.clone()),
            HostCommand::Shutdown => Self::Shutdown,
            HostCommand::UpdateSelection(update) => Self::UpdateSelection(update.clone()),
            HostCommand::AdjustAgentMutationRates {
                agent_uid,
                delta_primary,
                delta_secondary,
            } => Self::AdjustAgentMutationRates {
                agent_uid: agent_uid.get(),
                delta_primary_bits: delta_primary.to_bits(),
                delta_secondary_bits: delta_secondary.to_bits(),
            },
            HostCommand::SpawnAgent { herbivore_tendency } => Self::SpawnAgent {
                herbivore_tendency_bits: herbivore_tendency.to_bits(),
            },
            HostCommand::SpawnCrossover { parent_a, parent_b } => Self::SpawnCrossover {
                parent_a: parent_a.get(),
                parent_b: parent_b.get(),
            },
        }
    }

    fn into_runtime(self) -> HostCommand {
        match self {
            Self::Pause => HostCommand::Pause,
            Self::Resume => HostCommand::Resume,
            Self::SetSpeedBits(bits) => HostCommand::SetSpeed(f32::from_bits(bits)),
            Self::Step => HostCommand::Step,
            Self::UpdateConfig(config) => HostCommand::UpdateConfig(config),
            Self::Shutdown => HostCommand::Shutdown,
            Self::UpdateSelection(update) => HostCommand::UpdateSelection(update),
            Self::AdjustAgentMutationRates {
                agent_uid,
                delta_primary_bits,
                delta_secondary_bits,
            } => HostCommand::AdjustAgentMutationRates {
                agent_uid: AgentUid(agent_uid),
                delta_primary: f32::from_bits(delta_primary_bits),
                delta_secondary: f32::from_bits(delta_secondary_bits),
            },
            Self::SpawnAgent {
                herbivore_tendency_bits,
            } => HostCommand::SpawnAgent {
                herbivore_tendency: f32::from_bits(herbivore_tendency_bits),
            },
            Self::SpawnCrossover { parent_a, parent_b } => HostCommand::SpawnCrossover {
                parent_a: AgentUid(parent_a),
                parent_b: AgentUid(parent_b),
            },
        }
    }
}

#[derive(serde::Serialize)]
enum HostCommandPostcardRefV1<'a> {
    Pause,
    Resume,
    SetSpeedBits(u32),
    Step,
    UpdateConfig(&'a ScriptBotsConfig),
    Shutdown,
    UpdateSelection(&'a SelectionUpdate),
    AdjustAgentMutationRates {
        agent_uid: u64,
        delta_primary_bits: u32,
        delta_secondary_bits: u32,
    },
    SpawnAgent {
        herbivore_tendency_bits: u32,
    },
    SpawnCrossover {
        parent_a: u64,
        parent_b: u64,
    },
}

impl<'a> HostCommandPostcardRefV1<'a> {
    fn from_runtime(command: &'a HostCommand) -> Self {
        match command {
            HostCommand::Pause => Self::Pause,
            HostCommand::Resume => Self::Resume,
            HostCommand::SetSpeed(speed) => Self::SetSpeedBits(speed.to_bits()),
            HostCommand::Step => Self::Step,
            HostCommand::UpdateConfig(config) => Self::UpdateConfig(config),
            HostCommand::Shutdown => Self::Shutdown,
            HostCommand::UpdateSelection(update) => Self::UpdateSelection(update),
            HostCommand::AdjustAgentMutationRates {
                agent_uid,
                delta_primary,
                delta_secondary,
            } => Self::AdjustAgentMutationRates {
                agent_uid: agent_uid.get(),
                delta_primary_bits: delta_primary.to_bits(),
                delta_secondary_bits: delta_secondary.to_bits(),
            },
            HostCommand::SpawnAgent { herbivore_tendency } => Self::SpawnAgent {
                herbivore_tendency_bits: herbivore_tendency.to_bits(),
            },
            HostCommand::SpawnCrossover { parent_a, parent_b } => Self::SpawnCrossover {
                parent_a: parent_a.get(),
                parent_b: parent_b.get(),
            },
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct CommandEnvelopePostcardV1 {
    command_id: u128,
    expected_control_revision: Option<u64>,
    expected_scientific_revision: Option<u64>,
    expected_config_revision: Option<u64>,
    command: HostCommandPostcardV1,
}

impl CommandEnvelopePostcardV1 {
    fn from_runtime(envelope: &CommandEnvelope) -> Self {
        Self {
            command_id: envelope.command_id.get(),
            expected_control_revision: envelope.expected_control_revision.map(|value| value.get()),
            expected_scientific_revision: envelope
                .expected_scientific_revision
                .map(|value| value.get()),
            expected_config_revision: envelope.expected_config_revision.map(|value| value.get()),
            command: HostCommandPostcardV1::from_runtime(&envelope.command),
        }
    }

    fn into_runtime(self) -> CommandEnvelope {
        CommandEnvelope {
            command_id: CommandId::new(self.command_id),
            expected_control_revision: self.expected_control_revision.map(ControlRevision::new),
            expected_scientific_revision: self
                .expected_scientific_revision
                .map(ScientificRevision::new),
            expected_config_revision: self.expected_config_revision.map(ConfigRevision::new),
            command: self.command.into_runtime(),
        }
    }
}

#[derive(serde::Serialize)]
struct CommandEnvelopePostcardRefV1<'a> {
    command_id: u128,
    expected_control_revision: Option<u64>,
    expected_scientific_revision: Option<u64>,
    expected_config_revision: Option<u64>,
    command: HostCommandPostcardRefV1<'a>,
}

impl<'a> CommandEnvelopePostcardRefV1<'a> {
    fn from_runtime(envelope: &'a CommandEnvelope) -> Self {
        Self {
            command_id: envelope.command_id.get(),
            expected_control_revision: envelope.expected_control_revision.map(|value| value.get()),
            expected_scientific_revision: envelope
                .expected_scientific_revision
                .map(|value| value.get()),
            expected_config_revision: envelope.expected_config_revision.map(|value| value.get()),
            command: HostCommandPostcardRefV1::from_runtime(&envelope.command),
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct AppliedCommandPostcardV1 {
    tick: u64,
    control_revision: u64,
    scientific_revision: u64,
    config_revision: u64,
}

impl AppliedCommandPostcardV1 {
    const fn from_runtime(applied: AppliedCommand) -> Self {
        Self {
            tick: applied.tick.0,
            control_revision: applied.revisions.control.get(),
            scientific_revision: applied.revisions.scientific.get(),
            config_revision: applied.revisions.config.get(),
        }
    }

    const fn into_runtime(self) -> AppliedCommand {
        AppliedCommand {
            tick: Tick(self.tick),
            revisions: HostRevisions {
                control: ControlRevision::new(self.control_revision),
                scientific: ScientificRevision::new(self.scientific_revision),
                config: ConfigRevision::new(self.config_revision),
            },
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
enum RejectionReasonPostcardV1 {
    Validation { message: String },
    ControlRevisionConflict { expected: u64, actual: u64 },
    ScientificRevisionConflict { expected: u64, actual: u64 },
    ConfigRevisionConflict { expected: u64, actual: u64 },
    Overloaded { capacity: u64 },
    HostStopping,
}

impl RejectionReasonPostcardV1 {
    fn from_runtime(
        rejection: &RejectionReason,
        context: &'static str,
    ) -> Result<Self, StorageError> {
        match rejection {
            RejectionReason::Validation { message } => Ok(Self::Validation {
                message: message.clone(),
            }),
            RejectionReason::ControlRevisionConflict { expected, actual } => {
                Ok(Self::ControlRevisionConflict {
                    expected: expected.get(),
                    actual: actual.get(),
                })
            }
            RejectionReason::ScientificRevisionConflict { expected, actual } => {
                Ok(Self::ScientificRevisionConflict {
                    expected: expected.get(),
                    actual: actual.get(),
                })
            }
            RejectionReason::ConfigRevisionConflict { expected, actual } => {
                Ok(Self::ConfigRevisionConflict {
                    expected: expected.get(),
                    actual: actual.get(),
                })
            }
            RejectionReason::Overloaded { capacity } => Ok(Self::Overloaded {
                capacity: u64::try_from(*capacity).map_err(|error| StorageError::InvalidData {
                    context,
                    reason: format!("overload capacity cannot be represented on the wire: {error}"),
                })?,
            }),
            RejectionReason::HostStopping => Ok(Self::HostStopping),
        }
    }

    fn into_runtime(self, context: &'static str) -> Result<RejectionReason, StorageError> {
        match self {
            Self::Validation { message } => Ok(RejectionReason::Validation { message }),
            Self::ControlRevisionConflict { expected, actual } => {
                Ok(RejectionReason::ControlRevisionConflict {
                    expected: ControlRevision::new(expected),
                    actual: ControlRevision::new(actual),
                })
            }
            Self::ScientificRevisionConflict { expected, actual } => {
                Ok(RejectionReason::ScientificRevisionConflict {
                    expected: ScientificRevision::new(expected),
                    actual: ScientificRevision::new(actual),
                })
            }
            Self::ConfigRevisionConflict { expected, actual } => {
                Ok(RejectionReason::ConfigRevisionConflict {
                    expected: ConfigRevision::new(expected),
                    actual: ConfigRevision::new(actual),
                })
            }
            Self::Overloaded { capacity } => Ok(RejectionReason::Overloaded {
                capacity: usize::try_from(capacity).map_err(|error| StorageError::InvalidData {
                    context,
                    reason: format!(
                        "overload capacity cannot be represented on this host: {error}"
                    ),
                })?,
            }),
            Self::HostStopping => Ok(RejectionReason::HostStopping),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct ApplicationFailurePostcardV1 {
    code: String,
    message: String,
}

impl ApplicationFailurePostcardV1 {
    fn from_runtime(failure: &ApplicationFailure) -> Self {
        Self {
            code: failure.code.clone(),
            message: failure.message.clone(),
        }
    }

    fn into_runtime(self) -> ApplicationFailure {
        ApplicationFailure {
            code: self.code,
            message: self.message,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
enum ApplicationStatePostcardV1 {
    Admitted,
    Applied(AppliedCommandPostcardV1),
    Rejected(RejectionReasonPostcardV1),
    Failed(ApplicationFailurePostcardV1),
}

impl ApplicationStatePostcardV1 {
    fn from_runtime(
        application: &ApplicationState,
        context: &'static str,
    ) -> Result<Self, StorageError> {
        match application {
            ApplicationState::Admitted => Ok(Self::Admitted),
            ApplicationState::Applied(applied) => Ok(Self::Applied(
                AppliedCommandPostcardV1::from_runtime(*applied),
            )),
            ApplicationState::Rejected(rejection) => Ok(Self::Rejected(
                RejectionReasonPostcardV1::from_runtime(rejection, context)?,
            )),
            ApplicationState::Failed(failure) => Ok(Self::Failed(
                ApplicationFailurePostcardV1::from_runtime(failure),
            )),
        }
    }

    fn into_runtime(self, context: &'static str) -> Result<ApplicationState, StorageError> {
        match self {
            Self::Admitted => Ok(ApplicationState::Admitted),
            Self::Applied(applied) => Ok(ApplicationState::Applied(applied.into_runtime())),
            Self::Rejected(rejection) => {
                Ok(ApplicationState::Rejected(rejection.into_runtime(context)?))
            }
            Self::Failed(failure) => Ok(ApplicationState::Failed(failure.into_runtime())),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct CommandLifecycleTransitionPostcardV1 {
    ordinal: u32,
    boundary: AppliedCommandPostcardV1,
    application: ApplicationStatePostcardV1,
}

impl CommandLifecycleTransitionPostcardV1 {
    fn from_runtime(
        transition: &CommandLifecycleTransition,
        context: &'static str,
    ) -> Result<Self, StorageError> {
        Ok(Self {
            ordinal: transition.ordinal(),
            boundary: AppliedCommandPostcardV1::from_runtime(transition.boundary()),
            application: ApplicationStatePostcardV1::from_runtime(
                transition.application(),
                context,
            )?,
        })
    }

    fn into_runtime(
        self,
        context: &'static str,
    ) -> Result<CommandLifecycleTransition, StorageError> {
        Ok(CommandLifecycleTransition::new(
            self.ordinal,
            self.boundary.into_runtime(),
            self.application.into_runtime(context)?,
        ))
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct CommandLifecyclePostcardV1 {
    schema_version: u16,
    source_client_namespace: u64,
    envelope: CommandEnvelopePostcardV1,
    admission_sequence: Option<u64>,
    transitions: Vec<CommandLifecycleTransitionPostcardV1>,
}

impl CommandLifecyclePostcardV1 {
    fn from_runtime(
        lifecycle: &CommandLifecycleEvidence,
        context: &'static str,
    ) -> Result<Self, StorageError> {
        lifecycle
            .validate()
            .map_err(|error| StorageError::InvalidData {
                context,
                reason: error.to_string(),
            })?;
        Ok(Self {
            schema_version: lifecycle.schema_version(),
            source_client_namespace: lifecycle.source_client_namespace(),
            envelope: CommandEnvelopePostcardV1::from_runtime(lifecycle.envelope()),
            admission_sequence: lifecycle.admission_sequence().map(|value| value.get()),
            transitions: lifecycle
                .transitions()
                .iter()
                .map(|transition| {
                    CommandLifecycleTransitionPostcardV1::from_runtime(transition, context)
                })
                .collect::<Result<Vec<_>, StorageError>>()?,
        })
    }

    fn into_runtime(self, context: &'static str) -> Result<CommandLifecycleEvidence, StorageError> {
        if self.schema_version != COMMAND_LIFECYCLE_SCHEMA_VERSION {
            return Err(StorageError::InvalidData {
                context,
                reason: format!(
                    "unsupported command lifecycle schema version {}, expected {COMMAND_LIFECYCLE_SCHEMA_VERSION}",
                    self.schema_version
                ),
            });
        }
        let envelope = self.envelope.into_runtime();
        if self.source_client_namespace != envelope.command_id.client_namespace() {
            return Err(StorageError::InvalidData {
                context,
                reason: format!(
                    "command source namespace {} does not match command id namespace {}",
                    self.source_client_namespace,
                    envelope.command_id.client_namespace()
                ),
            });
        }
        let admission_sequence = self.admission_sequence.map(AdmissionSequence::new);
        let transitions = self
            .transitions
            .into_iter()
            .map(|transition| transition.into_runtime(context))
            .collect::<Result<Vec<_>, StorageError>>()?;
        CommandLifecycleEvidence::try_new(envelope, admission_sequence, transitions).map_err(
            |error| StorageError::InvalidData {
                context,
                reason: error.to_string(),
            },
        )
    }
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        encoded.push(char::from(DIGITS[usize::from(byte >> 4)]));
        encoded.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    encoded
}

struct LowerHexWriter {
    encoded: String,
}

impl LowerHexWriter {
    fn for_postcard_bytes(
        context: &'static str,
        postcard_bytes: usize,
    ) -> Result<Self, StorageError> {
        let capacity = postcard_bytes
            .checked_mul(2)
            .ok_or_else(|| StorageError::InvalidData {
                context,
                reason: "postcard hex length exceeds the platform allocation range".to_owned(),
            })?;
        Ok(Self {
            encoded: String::with_capacity(capacity),
        })
    }
}

impl Write for LowerHexWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        for byte in bytes {
            self.encoded
                .push(char::from(DIGITS[usize::from(byte >> 4)]));
            self.encoded
                .push(char::from(DIGITS[usize::from(byte & 0x0f)]));
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn decode_lower_hex(context: &'static str, encoded: &str) -> Result<Vec<u8>, StorageError> {
    fn nibble(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        }
    }

    if encoded.is_empty() || !encoded.len().is_multiple_of(2) {
        return Err(StorageError::InvalidData {
            context,
            reason: "postcard hex must have a nonzero even length".to_owned(),
        });
    }
    let mut decoded = Vec::with_capacity(encoded.len() / 2);
    for pair in encoded.as_bytes().as_chunks::<2>().0 {
        let high = nibble(pair[0]).ok_or(StorageError::InvalidData {
            context,
            reason: "postcard hex must use lowercase hexadecimal characters".to_owned(),
        })?;
        let low = nibble(pair[1]).ok_or(StorageError::InvalidData {
            context,
            reason: "postcard hex must use lowercase hexadecimal characters".to_owned(),
        })?;
        decoded.push((high << 4) | low);
    }
    Ok(decoded)
}

fn encode_postcard_hex<T: serde::Serialize + ?Sized>(
    context: &'static str,
    value: &T,
) -> Result<String, StorageError> {
    postcard::to_allocvec(value)
        .map(|bytes| encode_lower_hex(&bytes))
        .map_err(|error| StorageError::InvalidData {
            context,
            reason: error.to_string(),
        })
}

fn decode_postcard_hex<T: serde::de::DeserializeOwned>(
    context: &'static str,
    encoded: &str,
) -> Result<T, StorageError> {
    let bytes = decode_lower_hex(context, encoded)?;
    postcard::from_bytes(&bytes).map_err(|error| StorageError::InvalidData {
        context,
        reason: error.to_string(),
    })
}

pub(super) fn encode_command_envelope_postcard_hex(
    context: &'static str,
    envelope: &CommandEnvelope,
) -> Result<String, StorageError> {
    let postcard_bytes = command_envelope_postcard_size(context, envelope)?;
    encode_command_envelope_postcard_hex_with_size(context, envelope, postcard_bytes)
}

fn command_envelope_postcard_size(
    context: &'static str,
    envelope: &CommandEnvelope,
) -> Result<usize, StorageError> {
    postcard::experimental::serialized_size(&CommandEnvelopePostcardRefV1::from_runtime(envelope))
        .map_err(|error| StorageError::InvalidData {
            context,
            reason: error.to_string(),
        })
}

fn encode_command_envelope_postcard_hex_with_size(
    context: &'static str,
    envelope: &CommandEnvelope,
    postcard_bytes: usize,
) -> Result<String, StorageError> {
    let writer = LowerHexWriter::for_postcard_bytes(context, postcard_bytes)?;
    let writer = postcard::to_io(
        &CommandEnvelopePostcardRefV1::from_runtime(envelope),
        writer,
    )
    .map_err(|error| StorageError::InvalidData {
        context,
        reason: error.to_string(),
    })?;
    let expected_hex_bytes =
        postcard_bytes
            .checked_mul(2)
            .ok_or_else(|| StorageError::InvalidData {
                context,
                reason: "postcard hex length exceeds the platform allocation range".to_owned(),
            })?;
    if writer.encoded.len() != expected_hex_bytes {
        return Err(StorageError::InvalidData {
            context,
            reason: format!(
                "postcard encoder produced {} hex bytes after sizing {postcard_bytes} binary bytes",
                writer.encoded.len()
            ),
        });
    }
    Ok(writer.encoded)
}

pub(super) fn decode_command_envelope_postcard_hex(
    context: &'static str,
    encoded: &str,
) -> Result<CommandEnvelope, StorageError> {
    decode_postcard_hex::<CommandEnvelopePostcardV1>(context, encoded)
        .map(CommandEnvelopePostcardV1::into_runtime)
}

pub(super) fn encode_host_command_postcard_hex(
    context: &'static str,
    command: &HostCommand,
) -> Result<String, StorageError> {
    encode_postcard_hex(context, &HostCommandPostcardV1::from_runtime(command))
}

pub(super) fn decode_host_command_postcard_hex(
    context: &'static str,
    encoded: &str,
) -> Result<HostCommand, StorageError> {
    decode_postcard_hex::<HostCommandPostcardV1>(context, encoded)
        .map(HostCommandPostcardV1::into_runtime)
}

pub(super) fn encode_application_state_postcard_hex(
    context: &'static str,
    application: &ApplicationState,
) -> Result<String, StorageError> {
    encode_postcard_hex(
        context,
        &ApplicationStatePostcardV1::from_runtime(application, context)?,
    )
}

pub(super) fn decode_application_state_postcard_hex(
    context: &'static str,
    encoded: &str,
) -> Result<ApplicationState, StorageError> {
    decode_postcard_hex::<ApplicationStatePostcardV1>(context, encoded)?.into_runtime(context)
}

fn encode_command_lifecycle_postcard_hex(
    context: &'static str,
    lifecycle: &CommandLifecycleEvidence,
) -> Result<String, StorageError> {
    encode_postcard_hex(
        context,
        &CommandLifecyclePostcardV1::from_runtime(lifecycle, context)?,
    )
}

fn decode_command_lifecycle_postcard_hex(
    context: &'static str,
    encoded: &str,
) -> Result<CommandLifecycleEvidence, StorageError> {
    decode_postcard_hex::<CommandLifecyclePostcardV1>(context, encoded)?.into_runtime(context)
}

pub(super) fn encode_journal_u64(value: u64) -> String {
    format!("{value:016x}")
}

pub(super) fn decode_journal_u64(
    context: &'static str,
    encoded: &str,
) -> Result<u64, StorageError> {
    if encoded.len() != 16
        || !encoded
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(StorageError::InvalidData {
            context,
            reason: format!("expected 16 lowercase hexadecimal characters, found {encoded:?}"),
        });
    }
    u64::from_str_radix(encoded, 16).map_err(|error| StorageError::InvalidData {
        context,
        reason: error.to_string(),
    })
}

struct BoundedJsonWriter {
    bytes: Vec<u8>,
    maximum: usize,
}

impl BoundedJsonWriter {
    fn new(maximum: usize) -> Self {
        Self {
            bytes: Vec::new(),
            maximum,
        }
    }

    fn finish(self) -> Result<String, StorageError> {
        String::from_utf8(self.bytes).map_err(|error| StorageError::InvalidData {
            context: "host_journal_archive.payload_json",
            reason: error.to_string(),
        })
    }
}

impl Write for BoundedJsonWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = self.bytes.len().checked_add(bytes.len()).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "host journal archive length overflow",
            )
        })?;
        if next > self.maximum {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "host journal archive exceeds the configured {} byte maximum",
                    self.maximum
                ),
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct CanonicalJsonComparator<'a> {
    expected: &'a [u8],
    offset: usize,
    maximum: usize,
}

impl<'a> CanonicalJsonComparator<'a> {
    fn new(expected: &'a str, maximum: usize) -> Self {
        Self {
            expected: expected.as_bytes(),
            offset: 0,
            maximum,
        }
    }

    fn finish(self) -> bool {
        self.offset == self.expected.len()
    }
}

impl Write for CanonicalJsonComparator<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = self.offset.checked_add(bytes.len()).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "host journal canonical comparison length overflow",
            )
        })?;
        if next > self.maximum
            || self
                .expected
                .get(self.offset..next)
                .is_none_or(|expected| expected != bytes)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "host journal archive is not in canonical JSON form",
            ));
        }
        self.offset = next;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct EncodedCommandLifecycle {
    schema_version: u16,
    postcard_hex: String,
}

impl EncodedCommandLifecycle {
    fn encode(lifecycle: &CommandLifecycleEvidence) -> Result<Self, StorageError> {
        lifecycle
            .validate()
            .map_err(|error| StorageError::InvalidData {
                context: "host_journal_archive.command_lifecycle",
                reason: error.to_string(),
            })?;
        Ok(Self {
            schema_version: lifecycle.schema_version(),
            postcard_hex: encode_command_lifecycle_postcard_hex(
                "host_journal_archive.command_lifecycle.postcard_hex",
                lifecycle,
            )?,
        })
    }

    fn decode(&self) -> Result<CommandLifecycleEvidence, StorageError> {
        if self.schema_version != COMMAND_LIFECYCLE_SCHEMA_VERSION {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.command_lifecycle.schema_version",
                reason: format!(
                    "unsupported version {}, expected {COMMAND_LIFECYCLE_SCHEMA_VERSION}",
                    self.schema_version
                ),
            });
        }
        let lifecycle = decode_command_lifecycle_postcard_hex(
            "host_journal_archive.command_lifecycle.postcard_hex",
            &self.postcard_hex,
        )?;
        lifecycle
            .validate()
            .map_err(|error| StorageError::InvalidData {
                context: "host_journal_archive.command_lifecycle",
                reason: error.to_string(),
            })?;
        if lifecycle.schema_version() != self.schema_version
            || encode_command_lifecycle_postcard_hex(
                "host_journal_archive.command_lifecycle.postcard_hex",
                &lifecycle,
            )? != self.postcard_hex
        {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.command_lifecycle.postcard_hex",
                reason: "command lifecycle is not in canonical postcard form".to_owned(),
            });
        }
        Ok(lifecycle)
    }
}

#[derive(serde::Serialize)]
struct HostJournalArchiveRef<'a> {
    version: u32,
    run_id: RunId,
    host_session_id: &'a str,
    journal_sequence: &'a str,
    scientific_event_sequence: Option<&'a str>,
    command_lifecycle: Option<&'a EncodedCommandLifecycle>,
    applied: AppliedCommand,
    scientific: Option<&'a ScientificBoundary>,
    persistence: Option<&'a StorageBuffer>,
}

fn validate_scientific_archive_boundary(
    journal_sequence: u64,
    event_sequence: Option<EventSequence>,
    applied: AppliedCommand,
    scientific: Option<&ScientificBoundary>,
    command_lifecycle: Option<&CommandLifecycleEvidence>,
    has_persistence: bool,
) -> Result<(), StorageError> {
    match command_lifecycle {
        None if scientific.is_none() => {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.payload_json",
                reason:
                    "journal batch must contain command lifecycle evidence or a scientific boundary"
                        .to_owned(),
            });
        }
        Some(lifecycle) => {
            lifecycle
                .validate()
                .map_err(|error| StorageError::InvalidData {
                    context: "host_journal_archive.command_lifecycle",
                    reason: error.to_string(),
                })?;
            let terminal = lifecycle.terminal().ok_or(StorageError::InvalidData {
                context: "host_journal_archive.command_lifecycle",
                reason: "validated command lifecycle has no terminal transition".to_owned(),
            })?;
            if terminal.boundary() != applied {
                return Err(StorageError::InvalidData {
                    context: "host_journal_archive.applied",
                    reason: "archive terminal boundary differs from command lifecycle evidence"
                        .to_owned(),
                });
            }
            match terminal.application() {
                ApplicationState::Applied(_) => match &lifecycle.envelope().command {
                    HostCommand::Pause | HostCommand::Resume | HostCommand::SetSpeed(_)
                        if scientific.is_some() || event_sequence.is_some() || has_persistence =>
                    {
                        return Err(StorageError::InvalidData {
                            context: "host_journal_archive.command_lifecycle",
                            reason: "an applied control command must be command-only".to_owned(),
                        });
                    }
                    HostCommand::Step if scientific.is_none() || event_sequence.is_none() => {
                        return Err(StorageError::InvalidData {
                            context: "host_journal_archive.scientific",
                            reason: "an applied step command requires its scientific boundary"
                                .to_owned(),
                        });
                    }
                    HostCommand::UpdateConfig(_)
                        if scientific.is_some() || event_sequence.is_some() || has_persistence =>
                    {
                        return Err(StorageError::InvalidData {
                            context: "host_journal_archive.command_lifecycle",
                            reason: "an applied update-config command must be command-only"
                                .to_owned(),
                        });
                    }
                    HostCommand::UpdateSelection(_)
                    | HostCommand::AdjustAgentMutationRates { .. }
                    | HostCommand::SpawnAgent { .. }
                    | HostCommand::SpawnCrossover { .. }
                        if scientific.is_some() || event_sequence.is_some() || has_persistence =>
                    {
                        return Err(StorageError::InvalidData {
                            context: "host_journal_archive.command_lifecycle",
                            reason:
                                "an applied selection or agent-edit command must be command-only"
                                    .to_owned(),
                        });
                    }
                    HostCommand::Shutdown if scientific.is_some() || event_sequence.is_some() => {
                        return Err(StorageError::InvalidData {
                            context: "host_journal_archive.command_lifecycle",
                            reason: "an applied shutdown may carry only its final persistence tail"
                                .to_owned(),
                        });
                    }
                    _ => {}
                },
                ApplicationState::Rejected(_) | ApplicationState::Failed(_)
                    if scientific.is_some() || event_sequence.is_some() || has_persistence =>
                {
                    return Err(StorageError::InvalidData {
                        context: "host_journal_archive.command_lifecycle",
                        reason: "a rejected or failed command must be command-only".to_owned(),
                    });
                }
                ApplicationState::Admitted => {
                    return Err(StorageError::InvalidData {
                        context: "host_journal_archive.command_lifecycle",
                        reason: "archive command lifecycle is not terminal".to_owned(),
                    });
                }
                ApplicationState::Rejected(_) | ApplicationState::Failed(_) => {}
            }
        }
        None => {}
    }
    if has_persistence
        && scientific.is_none()
        && !command_lifecycle.is_some_and(CommandLifecycleEvidence::is_applied_shutdown)
    {
        return Err(StorageError::InvalidData {
            context: "host_journal_archive.persistence",
            reason: "persistence without science is reserved for an applied shutdown's final tail"
                .to_owned(),
        });
    }
    if event_sequence.is_some() != scientific.is_some() {
        return Err(StorageError::InvalidData {
            context: "host_journal_archive.scientific_event_sequence",
            reason: "scientific payload and event sequence must be present together".to_owned(),
        });
    }
    if event_sequence.is_some_and(|event| event.get() == 0 || event.get() > journal_sequence) {
        return Err(StorageError::InvalidData {
            context: "host_journal_archive.scientific_event_sequence",
            reason: "scientific event sequence must be nonzero and no later than its journal batch"
                .to_owned(),
        });
    }
    let Some(scientific) = scientific else {
        return Ok(());
    };
    let tick = applied.tick;
    if scientific.events().tick != tick
        || scientific.summary().tick != tick
        || scientific.config_revision() != applied.revisions.config.get()
        || scientific.births().iter().any(|record| record.tick != tick)
        || scientific.deaths().iter().any(|record| record.tick != tick)
        || scientific
            .resource_tick()
            .is_some_and(|record| record.tick != tick)
    {
        return Err(StorageError::InvalidData {
            context: "host_journal_archive.scientific",
            reason: "scientific payload does not match its applied tick and revisions".to_owned(),
        });
    }
    Ok(())
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HostJournalArchive {
    version: u32,
    run_id: RunId,
    host_session_id: String,
    journal_sequence: String,
    scientific_event_sequence: Option<String>,
    command_lifecycle: Option<EncodedCommandLifecycle>,
    applied: AppliedCommand,
    scientific: Option<ScientificBoundary>,
    persistence: Option<StorageBuffer>,
}

impl HostJournalArchive {
    pub(super) fn decode(
        payload_json: &str,
        payload_digest: &str,
        expected_run_id: RunId,
        expected_batch_id: JournalBatchId,
        maximum_bytes: usize,
    ) -> Result<Self, StorageError> {
        if payload_json.len() > maximum_bytes {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.payload_json",
                reason: format!(
                    "payload has {} bytes, exceeding the bounded {maximum_bytes} byte reader",
                    payload_json.len()
                ),
            });
        }
        let actual_digest = blake3::hash(payload_json.as_bytes()).to_hex().to_string();
        if actual_digest != payload_digest {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.payload_digest",
                reason: format!("expected {payload_digest}, computed {actual_digest}"),
            });
        }
        let archive: Self =
            serde_json::from_str(payload_json).map_err(|error| StorageError::InvalidData {
                context: "host_journal_archive.payload_json",
                reason: error.to_string(),
            })?;
        let mut canonical = CanonicalJsonComparator::new(payload_json, maximum_bytes);
        serde_json::to_writer(&mut canonical, &archive).map_err(|error| {
            StorageError::InvalidData {
                context: "host_journal_archive.payload_json",
                reason: error.to_string(),
            }
        })?;
        if !canonical.finish() {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.payload_json",
                reason: "payload is valid JSON but not in canonical archive form".to_owned(),
            });
        }
        archive.validate(expected_run_id, expected_batch_id)?;
        Ok(archive)
    }

    fn validate(
        &self,
        expected_run_id: RunId,
        expected_batch_id: JournalBatchId,
    ) -> Result<(), StorageError> {
        if self.version != HOST_JOURNAL_ARCHIVE_VERSION {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.payload_version",
                reason: format!(
                    "unsupported version {}, expected {HOST_JOURNAL_ARCHIVE_VERSION}",
                    self.version
                ),
            });
        }
        if self.run_id != expected_run_id {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.run_id",
                reason: format!(
                    "archive belongs to {}, expected {expected_run_id}",
                    self.run_id
                ),
            });
        }
        let session = decode_journal_u64(
            "host_journal_archive.host_session_id",
            &self.host_session_id,
        )?;
        let sequence = decode_journal_u64(
            "host_journal_archive.journal_sequence",
            &self.journal_sequence,
        )?;
        if sequence == 0 {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.identity",
                reason: "journal sequence must be nonzero".to_owned(),
            });
        }
        if JournalBatchId::new(HostSessionId::new(session), sequence) != expected_batch_id {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.journal_sequence",
                reason: format!(
                    "archive identity ({session}, {sequence}) does not match {expected_batch_id:?}"
                ),
            });
        }
        let event_sequence = self
            .scientific_event_sequence
            .as_deref()
            .map(|encoded| {
                decode_journal_u64("host_journal_archive.scientific_event_sequence", encoded)
                    .map(EventSequence::new)
            })
            .transpose()?;
        let command_lifecycle = self
            .command_lifecycle
            .as_ref()
            .map(EncodedCommandLifecycle::decode)
            .transpose()?;
        validate_scientific_archive_boundary(
            sequence,
            event_sequence,
            self.applied,
            self.scientific.as_ref(),
            command_lifecycle.as_ref(),
            self.persistence.is_some(),
        )?;
        if let Some(persistence) = &self.persistence {
            persistence.validate_contents(self.applied.tick.0)?;
        }
        Ok(())
    }

    pub(super) fn batch_id(&self) -> Result<JournalBatchId, StorageError> {
        Ok(JournalBatchId::new(
            HostSessionId::new(decode_journal_u64(
                "host_journal_archive.host_session_id",
                &self.host_session_id,
            )?),
            decode_journal_u64(
                "host_journal_archive.journal_sequence",
                &self.journal_sequence,
            )?,
        ))
    }

    pub(super) fn event_sequence(&self) -> Result<Option<EventSequence>, StorageError> {
        self.scientific_event_sequence
            .as_deref()
            .map(|encoded| {
                decode_journal_u64("host_journal_archive.scientific_event_sequence", encoded)
                    .map(EventSequence::new)
            })
            .transpose()
    }

    pub(super) const fn applied(&self) -> AppliedCommand {
        self.applied
    }

    pub(super) fn prepare_domain_event_projection(
        &self,
        archive_payload_digest: &str,
    ) -> Result<Option<PreparedDomainEventProjection>, StorageError> {
        match (self.event_sequence()?, self.scientific.as_ref()) {
            (Some(sequence), Some(scientific)) => PreparedDomainEventProjection::new(
                self.batch_id()?,
                sequence,
                self.applied.tick,
                scientific,
                archive_payload_digest,
            )
            .map(Some),
            (None, None) => Ok(None),
            (Some(_), None) => Err(StorageError::InvalidData {
                context: "host_journal_archive.scientific",
                reason: "scientific event sequence has no canonical payload".to_owned(),
            }),
            (None, Some(_)) => Err(StorageError::InvalidData {
                context: "host_journal_archive.scientific_event_sequence",
                reason: "scientific payload has no canonical event sequence".to_owned(),
            }),
        }
    }

    pub(super) fn prepare_command_projection(
        &self,
        archive_payload_digest: &str,
    ) -> Result<Option<PreparedCommandProjection>, StorageError> {
        self.command_lifecycle
            .as_ref()
            .map(|encoded| {
                let lifecycle = encoded.decode()?;
                PreparedCommandProjection::new(
                    self.batch_id()?,
                    &lifecycle,
                    self.applied,
                    self.event_sequence()?,
                    archive_payload_digest,
                )
            })
            .transpose()
    }

    pub(super) fn is_applied_shutdown(&self) -> Result<bool, StorageError> {
        self.command_lifecycle
            .as_ref()
            .map(EncodedCommandLifecycle::decode)
            .transpose()
            .map(|lifecycle| lifecycle.is_some_and(|value| value.is_applied_shutdown()))
    }

    pub(super) fn take_persistence(&mut self) -> Option<StorageBuffer> {
        self.persistence.take()
    }

    pub(super) fn into_journaled_event(
        self,
        commitment: EventCommitment,
    ) -> Result<Option<JournaledScientificEvent>, StorageError> {
        let Some(sequence) = self.event_sequence()? else {
            return Ok(None);
        };
        let batch_id = self.batch_id()?;
        let boundary = self.scientific.ok_or(StorageError::InvalidData {
            context: "host_journal_archive.scientific",
            reason: "scientific event sequence has no payload".to_owned(),
        })?;
        Ok(Some(JournaledScientificEvent {
            event: Arc::new(ScientificEvent {
                session_id: batch_id.session_id(),
                sequence,
                batch_id,
                tick: self.applied.tick,
                revisions: self.applied.revisions,
                boundary: Arc::new(boundary),
            }),
            commitment,
        }))
    }

    pub(super) fn into_public_record(
        self,
        state: HostJournalRecordState,
    ) -> Result<HostJournalRecord, StorageError> {
        let event_sequence = self.event_sequence()?;
        let batch_id = self.batch_id()?;
        let command_lifecycle = self
            .command_lifecycle
            .as_ref()
            .map(EncodedCommandLifecycle::decode)
            .transpose()?;
        let Self {
            applied,
            scientific,
            ..
        } = self;
        let event = match (event_sequence, scientific) {
            (Some(sequence), Some(boundary)) => Some(JournaledScientificEvent {
                event: Arc::new(ScientificEvent {
                    session_id: batch_id.session_id(),
                    sequence,
                    batch_id,
                    tick: applied.tick,
                    revisions: applied.revisions,
                    boundary: Arc::new(boundary),
                }),
                commitment: state.event_commitment(),
            }),
            (None, None) => None,
            (Some(_), None) => {
                return Err(StorageError::InvalidData {
                    context: "host_journal_archive.scientific",
                    reason: "scientific event sequence has no canonical payload".to_owned(),
                });
            }
            (None, Some(_)) => {
                return Err(StorageError::InvalidData {
                    context: "host_journal_archive.scientific_event_sequence",
                    reason: "scientific payload has no canonical event sequence".to_owned(),
                });
            }
        };
        Ok(HostJournalRecord {
            batch_id,
            command_lifecycle,
            applied,
            event,
            state,
        })
    }
}

#[derive(Debug)]
pub(super) struct PreparedHostJournalArchive {
    pub(super) payload_json: String,
    pub(super) payload_digest: String,
    pub(super) persistence: Option<StorageBuffer>,
    pub(super) domain_events: Option<PreparedDomainEventProjection>,
    pub(super) command: Option<PreparedCommandProjection>,
}

fn encode_host_journal_archive(
    archive: &HostJournalArchiveRef<'_>,
    maximum_bytes: usize,
) -> Result<(String, String), StorageError> {
    let mut writer = BoundedJsonWriter::new(maximum_bytes);
    serde_json::to_writer(&mut writer, archive).map_err(|error| StorageError::InvalidData {
        context: "host_journal_archive.payload_json",
        reason: error.to_string(),
    })?;
    let payload_json = writer.finish()?;
    let payload_digest = blake3::hash(payload_json.as_bytes()).to_hex().to_string();
    Ok((payload_json, payload_digest))
}

pub(super) fn prepare_host_journal_archive(
    run_id: RunId,
    batch: &JournalBatch,
    maximum_bytes: usize,
) -> Result<PreparedHostJournalArchive, StorageError> {
    let batch_id = batch.id();
    if batch_id.sequence() == 0 {
        return Err(StorageError::InvalidData {
            context: "host_journal_archive.identity",
            reason: "journal sequence must be nonzero".to_owned(),
        });
    }
    validate_scientific_archive_boundary(
        batch_id.sequence(),
        batch.scientific_event_sequence(),
        batch.applied(),
        batch.scientific().map(Arc::as_ref),
        batch.command_lifecycle(),
        batch.persistence().is_some(),
    )?;
    let persistence = batch
        .persistence()
        .map(|payload| {
            if payload.summary.tick != batch.applied().tick {
                return Err(StorageError::InvalidData {
                    context: "host_journal_archive.persistence.tick",
                    reason: format!(
                        "persistence tick {} does not match applied tick {}",
                        payload.summary.tick.0,
                        batch.applied().tick.0
                    ),
                });
            }
            Storage::prepare_batch(payload)
        })
        .transpose()?;
    let host_session_id = encode_journal_u64(batch_id.session_id().get());
    let journal_sequence = encode_journal_u64(batch_id.sequence());
    let scientific_event_sequence = batch
        .scientific_event_sequence()
        .map(|sequence| encode_journal_u64(sequence.get()));
    let command_lifecycle = batch
        .command_lifecycle()
        .map(EncodedCommandLifecycle::encode)
        .transpose()?;
    let archive = HostJournalArchiveRef {
        version: HOST_JOURNAL_ARCHIVE_VERSION,
        run_id,
        host_session_id: &host_session_id,
        journal_sequence: &journal_sequence,
        scientific_event_sequence: scientific_event_sequence.as_deref(),
        command_lifecycle: command_lifecycle.as_ref(),
        applied: batch.applied(),
        scientific: batch.scientific().map(Arc::as_ref),
        persistence: persistence.as_ref(),
    };
    let (payload_json, payload_digest) = encode_host_journal_archive(&archive, maximum_bytes)?;
    let domain_events = match (
        batch.scientific_event_sequence(),
        batch.scientific().map(Arc::as_ref),
    ) {
        (Some(sequence), Some(scientific)) => Some(PreparedDomainEventProjection::new(
            batch_id,
            sequence,
            batch.applied().tick,
            scientific,
            &payload_digest,
        )?),
        (None, None) => None,
        _ => {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.scientific_event_sequence",
                reason: "scientific payload and event sequence must be present together".to_owned(),
            });
        }
    };
    let command = batch
        .command_lifecycle()
        .map(|lifecycle| {
            PreparedCommandProjection::new(
                batch_id,
                lifecycle,
                batch.applied(),
                batch.scientific_event_sequence(),
                &payload_digest,
            )
        })
        .transpose()?;
    Ok(PreparedHostJournalArchive {
        payload_json,
        payload_digest,
        persistence,
        domain_events,
        command,
    })
}

#[derive(Clone)]
pub(super) enum JournalReaderBackend {
    File {
        path: Arc<str>,
        run_id: RunId,
        lease: Arc<ExistingStorageLease>,
    },
    Memory,
}

impl std::fmt::Debug for JournalReaderBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File { path, run_id, .. } => formatter
                .debug_struct("File")
                .field("path", path)
                .field("run_id", run_id)
                .finish_non_exhaustive(),
            Self::Memory => formatter.write_str("Memory"),
        }
    }
}

#[derive(Debug, Clone)]
struct JournalReaderView {
    available: Option<EventSequenceRange>,
    identities: VecDeque<(EventSequence, JournalBatchId)>,
    memory_events: VecDeque<(JournaledScientificEvent, usize)>,
    memory_bytes: usize,
}

impl JournalReaderView {
    fn empty() -> Self {
        Self {
            available: None,
            identities: VecDeque::new(),
            memory_events: VecDeque::new(),
            memory_bytes: 0,
        }
    }
}

struct FileRetentionToken {
    path: Arc<str>,
    run_id: RunId,
    session_id: HostSessionId,
    lease: Arc<ExistingStorageLease>,
}

impl std::fmt::Debug for FileRetentionToken {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FileRetentionToken")
            .field("path", &self.path)
            .field("run_id", &self.run_id)
            .field("session_id", &self.session_id)
            .field("lease_references", &Arc::strong_count(&self.lease))
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct JournalReaderInner {
    session_id: HostSessionId,
    guarantee: EventCatchUpGuarantee,
    identity_capacity: usize,
    max_event_page_bytes: usize,
    backend: JournalReaderBackend,
    view: ArcSwap<JournalReaderView>,
    file_retention: Option<Arc<FileRetentionToken>>,
}

/// Worker-owned publication handle for one detached event reader.
#[derive(Debug, Clone)]
pub(super) struct JournalReaderPublisher {
    inner: Arc<JournalReaderInner>,
}

fn journal_event_commitment(
    guarantee: EventCatchUpGuarantee,
    state: &JournalReceiptState,
) -> Result<EventCommitment, StorageError> {
    match (guarantee, state) {
        (EventCatchUpGuarantee::CrashDurable, JournalReceiptState::Durable) => {
            Ok(EventCommitment::Durable)
        }
        (EventCatchUpGuarantee::LiveMemory, JournalReceiptState::CommittedVolatile) => {
            Ok(EventCommitment::CommittedVolatile)
        }
        (guarantee, invalid) => Err(StorageError::InvalidData {
            context: "host_journal_reader.commitment",
            reason: format!("reader guarantee {guarantee:?} cannot publish receipt {invalid:?}"),
        }),
    }
}

impl JournalReaderPublisher {
    pub(super) fn new(
        session_id: HostSessionId,
        backend: JournalReaderBackend,
        options: StorageJournalOptions,
    ) -> Self {
        let (guarantee, file_retention) = match &backend {
            JournalReaderBackend::File {
                path,
                run_id,
                lease,
            } => (
                EventCatchUpGuarantee::CrashDurable,
                Some(Arc::new(FileRetentionToken {
                    path: Arc::clone(path),
                    run_id: *run_id,
                    session_id,
                    lease: Arc::clone(lease),
                })),
            ),
            JournalReaderBackend::Memory => (EventCatchUpGuarantee::LiveMemory, None),
        };
        Self {
            inner: Arc::new(JournalReaderInner {
                session_id,
                guarantee,
                identity_capacity: options.event_cache_capacity,
                max_event_page_bytes: options.max_event_page_bytes,
                backend,
                view: ArcSwap::from_pointee(JournalReaderView::empty()),
                file_retention,
            }),
        }
    }

    pub(super) fn reader(&self) -> Arc<dyn EventJournalReader> {
        Arc::new(StorageEventJournalReader {
            inner: Arc::clone(&self.inner),
        })
    }

    pub(super) fn install_recovered_index(
        &self,
        available: Option<EventSequenceRange>,
        identities: VecDeque<(EventSequence, JournalBatchId)>,
    ) {
        self.inner.view.store(Arc::new(JournalReaderView {
            available,
            identities,
            memory_events: VecDeque::new(),
            memory_bytes: 0,
        }));
    }

    pub(super) fn publish(
        &self,
        batch: &JournalBatch,
        state: &JournalReceiptState,
    ) -> Result<(), StorageError> {
        let commitment = journal_event_commitment(self.inner.guarantee, state)?;
        let (Some(sequence), Some(boundary)) =
            (batch.scientific_event_sequence(), batch.scientific())
        else {
            return Ok(());
        };
        let current = self.inner.view.load_full();
        if current.identities.contains(&(sequence, batch.id())) {
            return Ok(());
        }
        let contiguous = current.available.map_or(sequence.get() == 1, |range| {
            range.last.checked_next() == Some(sequence)
        });
        if !contiguous {
            return Err(StorageError::InvalidData {
                context: "host_journal_reader.event_sequence",
                reason: format!(
                    "event {sequence:?} does not extend cached range {:?}",
                    current.available
                ),
            });
        }
        let entry = JournaledScientificEvent {
            event: Arc::new(ScientificEvent {
                session_id: self.inner.session_id,
                sequence,
                batch_id: batch.id(),
                tick: batch.applied().tick,
                revisions: batch.applied().revisions,
                boundary: Arc::clone(boundary),
            }),
            commitment,
        };
        let mut next = (*current).clone();
        next.identities.push_back((sequence, batch.id()));
        while next.identities.len() > self.inner.identity_capacity {
            next.identities.pop_front();
        }
        match self.inner.guarantee {
            EventCatchUpGuarantee::CrashDurable => {
                next.available = Some(next.available.map_or(
                    EventSequenceRange {
                        first: sequence,
                        last: sequence,
                    },
                    |range| EventSequenceRange {
                        first: range.first,
                        last: sequence,
                    },
                ));
            }
            EventCatchUpGuarantee::LiveMemory => {
                let retained_bytes = batch.retained_bytes();
                next.memory_bytes = next.memory_bytes.saturating_add(retained_bytes);
                next.memory_events.push_back((entry, retained_bytes));
                while next.memory_events.len() > self.inner.identity_capacity
                    || next.memory_bytes > self.inner.max_event_page_bytes
                {
                    let Some((_event, evicted_bytes)) = next.memory_events.pop_front() else {
                        break;
                    };
                    next.memory_bytes = next.memory_bytes.saturating_sub(evicted_bytes);
                }
                next.identities = next
                    .memory_events
                    .iter()
                    .map(|(event, _bytes)| (event.event.sequence, event.event.batch_id))
                    .collect();
                next.available = next
                    .memory_events
                    .front()
                    .zip(next.memory_events.back())
                    .map(|(first, last)| EventSequenceRange {
                        first: first.0.event.sequence,
                        last: last.0.event.sequence,
                    });
            }
        }
        self.inner.view.store(Arc::new(next));
        Ok(())
    }
}

/// Detached, thread-safe reader for canonical scientific events committed by storage.
#[derive(Debug)]
pub struct StorageEventJournalReader {
    inner: Arc<JournalReaderInner>,
}

impl StorageEventJournalReader {
    /// Open a crash-durable reader over one previously registered host session.
    ///
    /// This constructor performs bounded metadata I/O. The resulting cached range,
    /// identity, and retention queries remain nonblocking; only [`Self::read`] opens
    /// a short-lived read-only FrankenSQLite connection.
    pub fn open_file(
        path: &str,
        run_id: RunId,
        session_id: HostSessionId,
        options: StorageJournalOptions,
    ) -> Result<Self, StorageError> {
        let options = options
            .validate()
            .map_err(|reason| StorageError::InvalidData {
                context: "storage.journal_options",
                reason: reason.to_owned(),
            })?;
        let lease = Arc::new(ExistingStorageLease::open(path)?);
        let (available, identities) = load_host_journal_index(
            path,
            &lease,
            run_id,
            session_id,
            options.event_cache_capacity,
        )?;
        let publisher = JournalReaderPublisher::new(
            session_id,
            JournalReaderBackend::File {
                path: Arc::from(path),
                run_id,
                lease,
            },
            options,
        );
        publisher.install_recovered_index(available, identities);
        Ok(Self {
            inner: publisher.inner,
        })
    }

    fn unavailable(
        locator: EventCatchUpLocator,
        reason: EventCatchUpUnavailableReason,
    ) -> EventCatchUp {
        EventCatchUp::Unavailable {
            range: locator.range(),
            reason,
        }
    }
}

impl EventJournalReader for StorageEventJournalReader {
    fn session_id(&self) -> HostSessionId {
        self.inner.session_id
    }

    fn guarantee(&self) -> EventCatchUpGuarantee {
        self.inner.guarantee
    }

    fn available_range(&self) -> Option<EventSequenceRange> {
        self.inner.view.load().available
    }

    fn retention_snapshot(&self) -> Option<EventRetentionSnapshot> {
        let view = self.inner.view.load_full();
        let range = view.available?;
        match &self.inner.file_retention {
            Some(token) => EventRetentionSnapshot::try_new(
                self.inner.session_id,
                self.inner.guarantee,
                range,
                Arc::clone(token),
            )
            .ok(),
            None => EventRetentionSnapshot::try_new(
                self.inner.session_id,
                self.inner.guarantee,
                range,
                view,
            )
            .ok(),
        }
    }

    fn contains_event_identity(&self, sequence: EventSequence, batch_id: JournalBatchId) -> bool {
        self.inner
            .view
            .load()
            .identities
            .contains(&(sequence, batch_id))
    }

    fn read(
        &self,
        locator: EventCatchUpLocator,
        limit: usize,
    ) -> Result<EventCatchUp, HostAccessError> {
        if locator.session_id() != self.inner.session_id {
            return Ok(Self::unavailable(
                locator,
                EventCatchUpUnavailableReason::SessionMismatch,
            ));
        }
        if locator.guarantee() != self.inner.guarantee {
            return Err(HostAccessError::ProtocolViolation {
                message: "event locator guarantee does not match the storage reader".to_owned(),
            });
        }
        let view = self.inner.view.load_full();
        let Some(available) = view.available else {
            return Ok(Self::unavailable(
                locator,
                EventCatchUpUnavailableReason::RangeExpired,
            ));
        };
        if !available.contains_range(locator.range()) {
            let reason = if available.contains(locator.range().last) {
                EventCatchUpUnavailableReason::PartialRange
            } else {
                EventCatchUpUnavailableReason::RangeExpired
            };
            return Ok(Self::unavailable(locator, reason));
        }
        let limit = limit.min(MAX_STORAGE_QUERY_PAGE);
        let events = match &self.inner.backend {
            JournalReaderBackend::File {
                path,
                run_id,
                lease,
            } => read_host_journal_events(
                path,
                lease,
                *run_id,
                self.inner.session_id,
                locator.range(),
                limit,
                self.inner.max_event_page_bytes,
            )
            .map_err(|error| HostAccessError::ProtocolViolation {
                message: format!("durable journal read failed: {error}"),
            })?,
            JournalReaderBackend::Memory => view
                .memory_events
                .iter()
                .filter(|(entry, _bytes)| locator.range().contains(entry.event.sequence))
                .take(limit)
                .map(|(entry, _bytes)| entry.clone())
                .collect(),
        };
        Ok(EventCatchUp::Contiguous(EventPage {
            session_id: self.inner.session_id,
            source: match self.inner.guarantee {
                EventCatchUpGuarantee::LiveMemory => EventPageSource::LiveMemory,
                EventCatchUpGuarantee::CrashDurable => EventPageSource::Durable,
            },
            events,
            latest: available.last,
        }))
    }
}

/// Bounded terminal truth and cancellation state shared by one port and the storage worker.
#[derive(Debug)]
pub(super) struct JournalSessionShared {
    terminal_receipts: Mutex<BTreeMap<JournalBatchId, JournalReceiptState>>,
    capacity: usize,
    cancelled: AtomicBool,
    final_accepted_sequence: AtomicU64,
    completed_sequence: AtomicU64,
    acknowledged_sequence: AtomicU64,
    worker_closed: AtomicBool,
}

impl JournalSessionShared {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            terminal_receipts: Mutex::new(BTreeMap::new()),
            capacity,
            cancelled: AtomicBool::new(false),
            final_accepted_sequence: AtomicU64::new(0),
            completed_sequence: AtomicU64::new(0),
            acknowledged_sequence: AtomicU64::new(0),
            worker_closed: AtomicBool::new(false),
        }
    }

    pub(super) fn cache_terminal(
        &self,
        batch_id: JournalBatchId,
        state: &JournalReceiptState,
    ) -> Result<(), StorageError> {
        if !matches!(
            state,
            JournalReceiptState::CommittedVolatile | JournalReceiptState::Durable
        ) {
            return Err(StorageError::InvalidData {
                context: "host_journal_receipt.terminal_cache",
                reason: format!("cannot cache nonterminal receipt state {state:?}"),
            });
        }
        let mut cache = self
            .terminal_receipts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let acknowledged = self.acknowledged_sequence.load(Ordering::Acquire);
        cache.retain(|cached_id, _state| cached_id.sequence() > acknowledged);
        if let Some(existing) = cache.get(&batch_id) {
            if existing == state {
                return Ok(());
            }
            return Err(StorageError::InvalidData {
                context: "host_journal_receipt.terminal_cache",
                reason: format!(
                    "batch {batch_id:?} cached {existing:?}, refusing conflicting {state:?}"
                ),
            });
        }
        if cache.len() >= self.capacity {
            return Err(StorageError::InvalidData {
                context: "host_journal_receipt.terminal_cache",
                reason: format!(
                    "terminal cache reached its bounded {}-batch capacity",
                    self.capacity
                ),
            });
        }
        cache.insert(batch_id, state.clone());
        Ok(())
    }

    pub(super) fn cancel_after(&self, final_accepted_sequence: u64) {
        self.final_accepted_sequence
            .store(final_accepted_sequence, Ordering::Release);
        self.cancelled.store(true, Ordering::Release);
    }

    pub(super) fn cancellation_boundary(&self) -> Option<u64> {
        self.cancelled
            .load(Ordering::Acquire)
            .then(|| self.final_accepted_sequence.load(Ordering::Acquire))
    }

    pub(super) fn mark_completed(&self, sequence: u64) {
        self.completed_sequence.store(sequence, Ordering::Release);
    }

    pub(super) fn acknowledge(&self, sequence: u64) {
        self.acknowledged_sequence
            .fetch_max(sequence, Ordering::Release);
    }

    pub(super) fn mark_worker_closed(&self) {
        self.worker_closed.store(true, Ordering::Release);
    }

    fn resolution_ready(&self, sequence: u64) -> bool {
        sequence <= self.completed_sequence.load(Ordering::Acquire)
            || self.worker_closed.load(Ordering::Acquire)
    }
}

/// Nonblocking, bounded adapter from [`JournalBatch`] to the storage owner thread.
pub struct StorageJournalPort {
    session_id: HostSessionId,
    tx: xchan::Sender<StorageCommand>,
    admission: Arc<Mutex<AdmissionState>>,
    shared: Arc<JournalSessionShared>,
    receipts: xchan::Receiver<JournalReceipt>,
    outstanding: BTreeMap<JournalBatchId, OutstandingJournalBatch>,
    // IDs resolved from the cache or a local timeout before their lane notification arrived.
    // Tombstones count against `capacity` until a matching notification removes them; a timeout
    // also seals admission.
    suppressed_receipts: BTreeSet<JournalBatchId>,
    capacity: usize,
    receipt_timeout: Duration,
    max_batch_bytes: usize,
    max_inflight_bytes: usize,
    inflight_bytes: Arc<AtomicUsize>,
    reader: Arc<dyn EventJournalReader>,
    command_reader: Option<Arc<dyn CommandAuthorityReader>>,
    shutdown_requirement: ShutdownCommitRequirement,
    expected_sequence: u64,
    last_accepted_sequence: u64,
    open: bool,
}

struct PendingCommandAuthorityLookup {
    requested_at: Instant,
    reply: xchan::Receiver<CommandAuthorityLookup>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CommandAuthorityLookupKey {
    Status(CommandId),
    Submit {
        command_id: CommandId,
        envelope_digest: [u8; blake3::OUT_LEN],
        policy: CommandClaimPolicy,
    },
}

struct StorageCommandAuthorityReader {
    session_id: HostSessionId,
    tx: xchan::Sender<StorageCommand>,
    capacity: usize,
    timeout: Duration,
    max_envelope_bytes: usize,
    inflight_bytes: Arc<AtomicUsize>,
    max_inflight_bytes: usize,
    pending: Mutex<HashMap<CommandAuthorityLookupKey, PendingCommandAuthorityLookup>>,
}

impl StorageCommandAuthorityReader {
    fn new(
        session_id: HostSessionId,
        tx: xchan::Sender<StorageCommand>,
        capacity: usize,
        timeout: Duration,
        max_envelope_bytes: usize,
        inflight_bytes: Arc<AtomicUsize>,
        max_inflight_bytes: usize,
    ) -> Self {
        Self {
            session_id,
            tx,
            capacity,
            timeout,
            max_envelope_bytes: max_envelope_bytes
                .min(max_inflight_bytes / 2)
                .min(MAX_COMMAND_ENVELOPE_BYTES),
            inflight_bytes,
            max_inflight_bytes,
            pending: Mutex::new(HashMap::with_capacity(capacity)),
        }
    }

    fn resolve<F>(
        &self,
        key: CommandAuthorityLookupKey,
        command_id: CommandId,
        build_request: F,
    ) -> CommandAuthorityLookup
    where
        F: FnOnce() -> Result<
            (CommandAuthorityRequest, Option<InFlightPermit>),
            CommandAuthorityLookup,
        >,
    {
        let mut pending = match self.pending.try_lock() {
            Ok(pending) => pending,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => {
                return CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Busy);
            }
        };
        if let Some(existing) = pending.get(&key) {
            match existing.reply.try_recv() {
                Ok(outcome) => {
                    pending.remove(&key);
                    return outcome;
                }
                Err(xchan::TryRecvError::Empty)
                    if existing.requested_at.elapsed() >= self.timeout =>
                {
                    pending.remove(&key);
                    return CommandAuthorityLookup::Failed(
                        CommandAuthorityLookupFailure::Timeout {
                            waited: self.timeout,
                        },
                    );
                }
                Err(xchan::TryRecvError::Empty) => {
                    return CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Pending);
                }
                Err(xchan::TryRecvError::Disconnected) => {
                    pending.remove(&key);
                    return CommandAuthorityLookup::Failed(
                        CommandAuthorityLookupFailure::Unavailable {
                            message: "storage command-authority reply lane disconnected".to_owned(),
                        },
                    );
                }
            }
        }
        // Polling another key's one-shot reply here would consume its ready authority truth.
        // Only the exact-key branch above may observe that outcome; this sweep expires deadlines.
        pending.retain(|_, existing| existing.requested_at.elapsed() < self.timeout);
        if pending.len() >= self.capacity {
            return CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Capacity {
                capacity: self.capacity,
            });
        }
        let (request, permit) = match build_request() {
            Ok(request) => request,
            Err(outcome) => return outcome,
        };
        let (reply_tx, reply_rx) = xchan::bounded(1);
        let command = StorageCommand::ResolveCommandAuthority {
            session_id: self.session_id,
            command_id,
            request,
            reply: reply_tx,
            permit,
        };
        match self.tx.try_send(command) {
            Ok(()) => {
                pending.insert(
                    key,
                    PendingCommandAuthorityLookup {
                        requested_at: Instant::now(),
                        reply: reply_rx,
                    },
                );
                CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Pending)
            }
            Err(xchan::TrySendError::Full(_)) => {
                CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Busy)
            }
            Err(xchan::TrySendError::Disconnected(_)) => {
                CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Unavailable {
                    message: "storage command lane disconnected".to_owned(),
                })
            }
        }
    }
}

impl CommandAuthorityReader for StorageCommandAuthorityReader {
    fn resolve_for_submit(
        &self,
        envelope: &CommandEnvelope,
        envelope_digest: [u8; blake3::OUT_LEN],
        policy: CommandClaimPolicy,
    ) -> CommandAuthorityLookup {
        self.resolve(
            CommandAuthorityLookupKey::Submit {
                command_id: envelope.command_id,
                envelope_digest,
                policy,
            },
            envelope.command_id,
            || {
                let envelope_bytes = command_envelope_postcard_size(
                    "host_command_claims.envelope_postcard_hex",
                    envelope,
                )
                .map_err(|error| {
                    CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Corrupt {
                        message: error.to_string(),
                    })
                })?;
                if envelope_bytes > self.max_envelope_bytes {
                    return Err(CommandAuthorityLookup::Failed(
                        CommandAuthorityLookupFailure::Oversized {
                            bytes: envelope_bytes,
                            limit: self.max_envelope_bytes,
                        },
                    ));
                }
                let Some(hex_bytes) = envelope_bytes.checked_mul(2) else {
                    return Err(CommandAuthorityLookup::Failed(
                        CommandAuthorityLookupFailure::Oversized {
                            bytes: usize::MAX,
                            limit: self.max_envelope_bytes,
                        },
                    ));
                };
                let permit = InFlightPermit::try_acquire(
                    &self.inflight_bytes,
                    hex_bytes,
                    self.max_inflight_bytes,
                )
                .map_err(|_| {
                    CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Capacity {
                        capacity: self.capacity,
                    })
                })?;
                let envelope_postcard_hex = encode_command_envelope_postcard_hex_with_size(
                    "host_command_claims.envelope_postcard_hex",
                    envelope,
                    envelope_bytes,
                )
                .map_err(|error| {
                    CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Corrupt {
                        message: error.to_string(),
                    })
                })?;
                Ok((
                    CommandAuthorityRequest::Submit {
                        envelope_postcard_hex,
                        policy,
                        max_envelope_bytes: self.max_envelope_bytes,
                    },
                    Some(permit),
                ))
            },
        )
    }

    fn resolve_status(&self, command_id: CommandId) -> CommandAuthorityLookup {
        self.resolve(
            CommandAuthorityLookupKey::Status(command_id),
            command_id,
            || {
                Ok((
                    CommandAuthorityRequest::Status {
                        max_envelope_bytes: self.max_envelope_bytes,
                    },
                    None,
                ))
            },
        )
    }
}

#[derive(Debug)]
struct OutstandingJournalBatch {
    batch: Arc<JournalBatch>,
    accepted_at: Instant,
    _permit: InFlightPermit,
}

impl std::fmt::Debug for StorageJournalPort {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StorageJournalPort")
            .field("session_id", &self.session_id)
            .field("outstanding", &self.outstanding.len())
            .field("suppressed_receipts", &self.suppressed_receipts.len())
            .field("capacity", &self.capacity)
            .field("receipt_timeout", &self.receipt_timeout)
            .field("max_batch_bytes", &self.max_batch_bytes)
            .field("max_inflight_bytes", &self.max_inflight_bytes)
            .finish_non_exhaustive()
    }
}

impl StorageJournalPort {
    pub(super) fn new(
        tx: xchan::Sender<StorageCommand>,
        admission: Arc<Mutex<AdmissionState>>,
        shared: Arc<JournalSessionShared>,
        receipts: xchan::Receiver<JournalReceipt>,
        publisher: &JournalReaderPublisher,
        options: StorageJournalOptions,
        shutdown_requirement: ShutdownCommitRequirement,
    ) -> Self {
        let inflight_bytes = Arc::new(AtomicUsize::new(0));
        let command_reader = Some(Arc::new(StorageCommandAuthorityReader::new(
            publisher.inner.session_id,
            tx.clone(),
            options.admission_capacity,
            options.receipt_timeout,
            options.max_batch_bytes,
            Arc::clone(&inflight_bytes),
            options.max_inflight_bytes,
        )) as Arc<dyn CommandAuthorityReader>);
        Self {
            session_id: publisher.inner.session_id,
            tx,
            admission,
            shared,
            receipts,
            outstanding: BTreeMap::new(),
            suppressed_receipts: BTreeSet::new(),
            capacity: options.admission_capacity,
            receipt_timeout: options.receipt_timeout,
            max_batch_bytes: options.max_batch_bytes,
            max_inflight_bytes: options.max_inflight_bytes,
            inflight_bytes,
            reader: publisher.reader(),
            command_reader,
            shutdown_requirement,
            expected_sequence: 1,
            last_accepted_sequence: 0,
            open: true,
        }
    }

    /// Exact journal and durable-authority bytes accepted but not yet released by worker progress.
    #[must_use]
    pub fn inflight_bytes(&self) -> usize {
        self.inflight_bytes
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    fn failed_receipt(batch_id: JournalBatchId, message: &str) -> JournalReceipt {
        JournalReceipt::new(
            batch_id,
            JournalReceiptState::Failed(JournalFailure {
                code: "storage_journal_disconnected".to_owned(),
                message: message.to_owned(),
            }),
        )
    }

    fn timed_out_receipt(&self, batch_id: JournalBatchId) -> JournalReceipt {
        JournalReceipt::new(
            batch_id,
            JournalReceiptState::Failed(JournalFailure {
                code: "storage_journal_receipt_timeout".to_owned(),
                message: format!(
                    "storage journal did not publish terminal receipt truth within {:?}",
                    self.receipt_timeout
                ),
            }),
        )
    }

    fn release_terminal(&mut self, batch_id: JournalBatchId) {
        self.shared.acknowledge(batch_id.sequence());
        self.outstanding.remove(&batch_id);
        match self.shared.terminal_receipts.try_lock() {
            Ok(mut cache) => {
                cache.remove(&batch_id);
            }
            Err(TryLockError::Poisoned(poisoned)) => {
                poisoned.into_inner().remove(&batch_id);
            }
            Err(TryLockError::WouldBlock) => {}
        }
    }

    fn append_expired_receipts(
        &mut self,
        now: Instant,
        limit: usize,
        receipts: &mut Vec<JournalReceipt>,
    ) {
        let remaining = limit.saturating_sub(receipts.len());
        let expired = self
            .outstanding
            .iter()
            .filter_map(|(batch_id, outstanding)| {
                (now.saturating_duration_since(outstanding.accepted_at) >= self.receipt_timeout)
                    .then_some(*batch_id)
            })
            .take(remaining)
            .collect::<Vec<_>>();
        if expired.is_empty() {
            return;
        }

        // The worker caches terminal truth before it notifies the receipt lane. Reconcile that
        // authoritative cache while selecting the timeout outcome so a preemption between those
        // two publications cannot turn a real commitment into a synthetic failure. Contention is
        // resolved by deferring this bounded poll; blocking here would violate JournalPort.
        let reconciled = {
            let mut cache = match self.shared.terminal_receipts.try_lock() {
                Ok(cache) => cache,
                Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                Err(TryLockError::WouldBlock) => return,
            };
            expired
                .into_iter()
                .map(|batch_id| {
                    let terminal = cache.remove(&batch_id);
                    (batch_id, terminal)
                })
                .collect::<Vec<_>>()
        };
        if reconciled.iter().any(|(_batch_id, state)| state.is_none()) {
            self.open = false;
            self.shared.cancel_after(self.last_accepted_sequence);
        }
        for (batch_id, terminal) in reconciled {
            // The worker may enqueue the matching notification immediately after the lane-first
            // drain above. Keep one bounded tombstone so that duplicate cannot reach HostCore.
            self.suppressed_receipts.insert(batch_id);
            self.release_terminal(batch_id);
            match terminal {
                Some(state) => receipts.push(JournalReceipt::new(batch_id, state)),
                None => receipts.push(self.timed_out_receipt(batch_id)),
            }
        }
    }
}

impl Drop for StorageJournalPort {
    fn drop(&mut self) {
        self.shared.cancel_after(self.last_accepted_sequence);
    }
}

impl JournalPort for StorageJournalPort {
    fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
        let batch_id = batch.id();
        if let Some(outstanding) = self.outstanding.get(&batch_id) {
            if Arc::ptr_eq(&outstanding.batch, batch) {
                return JournalAdmission::Accepted { batch_id };
            }
            self.open = false;
            return JournalAdmission::Closed { batch_id };
        }
        if !self.open {
            return JournalAdmission::Closed { batch_id };
        }
        if batch_id.session_id() != self.session_id
            || batch_id.sequence() != self.expected_sequence
            || batch.retained_bytes() > self.max_batch_bytes
        {
            self.open = false;
            return JournalAdmission::Closed { batch_id };
        }
        if self
            .outstanding
            .len()
            .saturating_add(self.suppressed_receipts.len())
            >= self.capacity
        {
            return JournalAdmission::Full {
                batch_id,
                capacity: self.capacity,
            };
        }
        let Ok(permit) = InFlightPermit::try_acquire(
            &self.inflight_bytes,
            batch.retained_bytes(),
            self.max_inflight_bytes,
        ) else {
            return JournalAdmission::Full {
                batch_id,
                capacity: self.capacity,
            };
        };
        let admission = Arc::clone(&self.admission);
        let gate = match admission.try_lock() {
            Ok(gate) if gate.open => gate,
            Ok(_) | Err(TryLockError::Poisoned(_)) => {
                self.open = false;
                return JournalAdmission::Closed { batch_id };
            }
            Err(TryLockError::WouldBlock) => {
                return JournalAdmission::Full {
                    batch_id,
                    capacity: self.capacity,
                };
            }
        };
        let command = StorageCommand::JournalAdmit {
            batch: Arc::clone(batch),
        };
        let result = match self.tx.try_send(command) {
            Ok(()) => {
                self.outstanding.insert(
                    batch_id,
                    OutstandingJournalBatch {
                        batch: Arc::clone(batch),
                        accepted_at: Instant::now(),
                        _permit: permit,
                    },
                );
                self.last_accepted_sequence = batch_id.sequence();
                self.expected_sequence = match self.expected_sequence.checked_add(1) {
                    Some(next) => next,
                    None => {
                        self.open = false;
                        u64::MAX
                    }
                };
                if batch.is_applied_shutdown() {
                    self.open = false;
                }
                JournalAdmission::Accepted { batch_id }
            }
            Err(xchan::TrySendError::Full(_command)) => JournalAdmission::Full {
                batch_id,
                capacity: self.capacity,
            },
            Err(xchan::TrySendError::Disconnected(_command)) => {
                self.open = false;
                JournalAdmission::Closed { batch_id }
            }
        };
        drop(gate);
        result
    }

    fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
        if limit == 0 {
            return Vec::new();
        }
        let now = Instant::now();
        let mut receipts = Vec::with_capacity(limit.min(self.capacity));
        // Truth already present in the nonblocking receipt lane wins a deadline tie. Once the
        // lane is drained, the authoritative terminal cache is reconciled before any expired
        // remainder becomes one sticky local terminal failure.
        while receipts.len() < limit {
            match self.receipts.try_recv() {
                Ok(receipt) => {
                    let batch_id = receipt.batch_id();
                    if self.suppressed_receipts.remove(&batch_id) {
                        self.release_terminal(batch_id);
                        continue;
                    }
                    if matches!(receipt.state(), JournalReceiptState::Failed(_)) {
                        self.open = false;
                    }
                    let terminal = matches!(
                        receipt.state(),
                        JournalReceiptState::Durable | JournalReceiptState::Failed(_)
                    ) || self.shutdown_requirement
                        == ShutdownCommitRequirement::CommittedVolatile;
                    if terminal {
                        self.release_terminal(batch_id);
                    }
                    receipts.push(receipt);
                }
                Err(xchan::TryRecvError::Empty) => break,
                Err(xchan::TryRecvError::Disconnected) => {
                    self.open = false;
                    let remaining = limit.saturating_sub(receipts.len());
                    let mut cache = match self.shared.terminal_receipts.try_lock() {
                        Ok(cache) => cache,
                        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                        Err(TryLockError::WouldBlock) => break,
                    };
                    let resolvable = self
                        .outstanding
                        .keys()
                        .copied()
                        .take_while(|batch_id| self.shared.resolution_ready(batch_id.sequence()))
                        .take(remaining)
                        .collect::<Vec<_>>();
                    for batch_id in resolvable {
                        self.shared.acknowledge(batch_id.sequence());
                        self.outstanding.remove(&batch_id);
                        receipts.push(match cache.remove(&batch_id) {
                            Some(state) => JournalReceipt::new(batch_id, state),
                            None => Self::failed_receipt(
                                batch_id,
                                "storage worker exited before terminal journal truth was published",
                            ),
                        });
                    }
                    break;
                }
            }
        }

        self.append_expired_receipts(now, limit, &mut receipts);

        match self.shared.terminal_receipts.try_lock() {
            Ok(mut cache) => {
                cache.retain(|batch_id, _state| self.outstanding.contains_key(batch_id));
            }
            Err(TryLockError::Poisoned(poisoned)) => {
                poisoned
                    .into_inner()
                    .retain(|batch_id, _state| self.outstanding.contains_key(batch_id));
            }
            Err(TryLockError::WouldBlock) => {}
        }
        receipts
    }

    fn event_reader(&self, session_id: HostSessionId) -> Option<Arc<dyn EventJournalReader>> {
        (session_id == self.session_id).then(|| Arc::clone(&self.reader))
    }

    fn command_authority_reader(
        &self,
        session_id: HostSessionId,
    ) -> Option<Arc<dyn CommandAuthorityReader>> {
        if session_id != self.session_id {
            return None;
        }
        self.command_reader.as_ref().map(Arc::clone)
    }

    fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
        self.shutdown_requirement
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{
        MetricSample, PersistenceBatch, ScriptBotsConfig, Tick, TickCombatSummary, TickEvents,
        TickSummary, WorldState,
    };
    use scriptbots_runtime::{
        AdmissionSequence, ApplicationFailure, ApplicationState, CommandId,
        CommandLifecycleTransition, HostBlocker, HostCore, HostCoreOptions, HostFault, HostHealth,
        HostRevisions, JournalState, ManualInstant, NullFrontend, PlaybackSnapshot,
        RejectionReason,
    };

    fn applied() -> AppliedCommand {
        AppliedCommand {
            tick: Tick(1),
            revisions: HostRevisions::default(),
        }
    }

    fn scientific() -> ScientificBoundary {
        ScientificBoundary::new(
            TickEvents {
                tick: Tick(1),
                charts_flushed: false,
                epoch_rolled: false,
                food_respawned: None,
            },
            TickSummary {
                tick: Tick(1),
                agent_count: 0,
                births: 0,
                deaths: 0,
                total_energy: 0.0,
                average_energy: 0.0,
                average_health: 0.0,
                max_age: 0,
                spike_hits: 0,
            },
            Vec::new(),
            Vec::new(),
            TickCombatSummary::default(),
            0,
            None,
        )
    }

    fn applied_lifecycle(
        envelope: CommandEnvelope,
        boundary: AppliedCommand,
    ) -> CommandLifecycleEvidence {
        CommandLifecycleEvidence::try_new(
            envelope,
            Some(AdmissionSequence::new(1)),
            vec![
                CommandLifecycleTransition::new(0, boundary, ApplicationState::Admitted),
                CommandLifecycleTransition::new(1, boundary, ApplicationState::Applied(boundary)),
            ],
        )
        .expect("valid applied command lifecycle")
    }

    fn rejected_lifecycle(
        envelope: CommandEnvelope,
        boundary: AppliedCommand,
    ) -> CommandLifecycleEvidence {
        CommandLifecycleEvidence::try_new(
            envelope,
            None,
            vec![CommandLifecycleTransition::new(
                0,
                boundary,
                ApplicationState::Rejected(RejectionReason::Validation {
                    message: "invalid command".to_owned(),
                }),
            )],
        )
        .expect("valid pre-admission rejection lifecycle")
    }

    fn failed_lifecycle(
        envelope: CommandEnvelope,
        boundary: AppliedCommand,
    ) -> CommandLifecycleEvidence {
        CommandLifecycleEvidence::try_new(
            envelope,
            Some(AdmissionSequence::new(1)),
            vec![
                CommandLifecycleTransition::new(0, boundary, ApplicationState::Admitted),
                CommandLifecycleTransition::new(
                    1,
                    boundary,
                    ApplicationState::Failed(ApplicationFailure {
                        code: "application_failed".to_owned(),
                        message: "deterministic failure".to_owned(),
                    }),
                ),
            ],
        )
        .expect("valid failed command lifecycle")
    }

    fn timeout_test_world() -> WorldState {
        WorldState::new(ScriptBotsConfig {
            world_width: 64,
            world_height: 64,
            food_cell_size: 16,
            rng_seed: Some(0x5eed),
            closed: true,
            history_capacity: 8,
            persistence_interval: 1,
            ..ScriptBotsConfig::default()
        })
        .expect("compact deterministic timeout world")
    }

    fn timeout_host_options() -> HostCoreOptions {
        HostCoreOptions {
            initial_playback: PlaybackSnapshot {
                paused: true,
                speed_multiplier: 1.0,
            },
            scientific_event_capacity: 2,
            volatile_event_history_capacity: 4,
            ..HostCoreOptions::default()
        }
    }

    fn admitted_journal_batch(command: StorageCommand) -> Option<Arc<JournalBatch>> {
        match command {
            StorageCommand::JournalAdmit { batch } => Some(batch),
            _ => None,
        }
    }

    fn measured_scientific_batch_bytes() -> (usize, usize) {
        let session_id = HostSessionId::new(0x404);
        let options = StorageJournalOptions::default();
        let (worker_tx, worker_rx) = xchan::bounded(DEFAULT_COMMAND_CAPACITY);
        let admission = Arc::new(Mutex::new(AdmissionState { open: true }));
        let shared = Arc::new(JournalSessionShared::new(options.admission_capacity));
        let (_receipt_tx, receipt_rx) = xchan::bounded(options.admission_capacity);
        let publisher =
            JournalReaderPublisher::new(session_id, JournalReaderBackend::Memory, options);
        let mut journal = StorageJournalPort::new(
            worker_tx,
            admission,
            shared,
            receipt_rx,
            &publisher,
            options,
            ShutdownCommitRequirement::CommittedVolatile,
        );
        journal.command_reader = None;
        let mut core = HostCore::with_journal(
            session_id,
            timeout_test_world(),
            timeout_host_options(),
            Box::new(journal),
        )
        .expect("host used to measure its exact production journal allocation");
        let mut frontend = NullFrontend::new(core.local_port(), 0x5004);

        frontend.step().expect("first measurement step admission");
        let first_applied = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(0))
            .expect("first measurement step reaches the production journal port");
        assert_eq!(first_applied.scientific_steps, 1);
        let first_queued = worker_rx
            .try_recv()
            .expect("measurement worker lane receives the first accepted journal batch");
        let first = admitted_journal_batch(first_queued)
            .expect("measurement worker lane returns the first journal admission");

        frontend.step().expect("second measurement step admission");
        let second_applied = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(1))
            .expect("second measurement step reaches the production journal port");
        assert_eq!(second_applied.scientific_steps, 1);
        let second_queued = worker_rx
            .try_recv()
            .expect("measurement worker lane receives the second accepted journal batch");
        let second = admitted_journal_batch(second_queued)
            .expect("measurement worker lane returns the second journal admission");
        (first.retained_bytes(), second.retained_bytes())
    }

    #[test]
    fn command_authority_sweep_preserves_ready_outcome_for_exact_key() {
        let session_id = HostSessionId::new(0x407);
        let first_id = CommandId::new(1);
        let second_id = CommandId::new(2);
        let (worker_tx, worker_rx) = xchan::bounded(2);
        let inflight_bytes = Arc::new(AtomicUsize::new(0));
        let reader = StorageCommandAuthorityReader::new(
            session_id,
            worker_tx,
            1,
            Duration::from_secs(60),
            1_024,
            inflight_bytes,
            1_024,
        );

        assert_eq!(
            reader.resolve_status(first_id),
            CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Pending)
        );
        let first_reply = match worker_rx
            .try_recv()
            .expect("first authority lookup reaches the storage worker")
        {
            StorageCommand::ResolveCommandAuthority {
                command_id, reply, ..
            } => {
                assert_eq!(command_id, first_id);
                reply
            }
            other => panic!("unexpected storage command: {other:?}"),
        };
        first_reply
            .try_send(CommandAuthorityLookup::Collision)
            .expect("worker publishes first authority outcome");

        assert_eq!(
            reader.resolve_status(second_id),
            CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Capacity { capacity: 1 })
        );
        assert!(
            matches!(worker_rx.try_recv(), Err(xchan::TryRecvError::Empty)),
            "an unrelated lookup must not replace the ready first outcome"
        );
        assert_eq!(
            reader.resolve_status(first_id),
            CommandAuthorityLookup::Collision
        );
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one allocation-boundary test covers frozen encoding, pre-allocation oversize refusal, shared byte-capacity refusal, and permit release"
    )]
    fn command_authority_sizes_before_encoding_and_holds_a_global_byte_permit() {
        let session_id = HostSessionId::new(0x408);
        let (worker_tx, worker_rx) = xchan::bounded(2);
        let inflight_bytes = Arc::new(AtomicUsize::new(0));
        let reader = StorageCommandAuthorityReader::new(
            session_id,
            worker_tx.clone(),
            2,
            Duration::from_secs(60),
            32,
            Arc::clone(&inflight_bytes),
            1_024,
        );
        let compatibility = CommandEnvelope::new(
            CommandId::new(9),
            HostCommand::UpdateSelection(SelectionUpdate {
                mode: scriptbots_core::SelectionMode::Replace,
                agent_ids: vec![7, 11],
                state: scriptbots_core::SelectionState::Selected,
            }),
        );
        assert_eq!(
            encode_command_envelope_postcard_hex(
                "host_command_claims.envelope_postcard_hex",
                &compatibility,
            )
            .expect("borrowed envelope encoding"),
            encode_postcard_hex(
                "host_command_claims.envelope_postcard_hex",
                &CommandEnvelopePostcardV1::from_runtime(&compatibility),
            )
            .expect("owned envelope encoding"),
            "bounded borrowed encoding changed the durable postcard contract"
        );
        let oversized = CommandEnvelope::new(
            CommandId::new(1),
            HostCommand::UpdateSelection(SelectionUpdate {
                mode: scriptbots_core::SelectionMode::Replace,
                agent_ids: vec![7; 128],
                state: scriptbots_core::SelectionState::Selected,
            }),
        );
        assert!(matches!(
            reader.resolve_for_submit(
                &oversized,
                [0; blake3::OUT_LEN],
                CommandClaimPolicy::ReserveIfAbsent,
            ),
            CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Oversized {
                limit: 32,
                ..
            })
        ));
        assert_eq!(inflight_bytes.load(Ordering::SeqCst), 0);
        assert!(matches!(
            worker_rx.try_recv(),
            Err(xchan::TryRecvError::Empty)
        ));

        let step = CommandEnvelope::new(CommandId::new(2), HostCommand::Step);
        let step_bytes =
            command_envelope_postcard_size("host_command_claims.envelope_postcard_hex", &step)
                .expect("step envelope size");
        let step_hex_bytes = step_bytes.checked_mul(2).expect("step hex size");
        let impossible = StorageCommandAuthorityReader::new(
            session_id,
            worker_tx.clone(),
            2,
            Duration::from_secs(60),
            1_024,
            Arc::new(AtomicUsize::new(0)),
            step_hex_bytes - 1,
        );
        assert!(matches!(
            impossible.resolve_for_submit(
                &step,
                [1; blake3::OUT_LEN],
                CommandClaimPolicy::ReserveIfAbsent,
            ),
            CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Oversized {
                bytes,
                limit,
            }) if bytes == step_bytes && limit == step_bytes - 1
        ));
        assert!(matches!(
            worker_rx.try_recv(),
            Err(xchan::TryRecvError::Empty)
        ));

        let preoccupied_bytes = Arc::new(AtomicUsize::new(1));
        let byte_starved = StorageCommandAuthorityReader::new(
            session_id,
            worker_tx.clone(),
            2,
            Duration::from_secs(60),
            1_024,
            Arc::clone(&preoccupied_bytes),
            step_hex_bytes,
        );
        assert_eq!(
            byte_starved.resolve_for_submit(
                &step,
                [1; blake3::OUT_LEN],
                CommandClaimPolicy::ReserveIfAbsent,
            ),
            CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Capacity { capacity: 2 })
        );
        assert_eq!(preoccupied_bytes.load(Ordering::SeqCst), 1);
        assert!(matches!(
            worker_rx.try_recv(),
            Err(xchan::TryRecvError::Empty)
        ));

        let exact_bytes = Arc::new(AtomicUsize::new(0));
        let exact_reader = StorageCommandAuthorityReader::new(
            session_id,
            worker_tx,
            2,
            Duration::from_secs(60),
            1_024,
            Arc::clone(&exact_bytes),
            step_hex_bytes,
        );
        assert_eq!(
            exact_reader.resolve_for_submit(
                &step,
                [1; blake3::OUT_LEN],
                CommandClaimPolicy::ReserveIfAbsent,
            ),
            CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Pending)
        );
        assert_eq!(exact_bytes.load(Ordering::SeqCst), step_hex_bytes);
        assert_eq!(
            exact_reader.resolve_for_submit(
                &step,
                [1; blake3::OUT_LEN],
                CommandClaimPolicy::ReserveIfAbsent,
            ),
            CommandAuthorityLookup::Failed(CommandAuthorityLookupFailure::Pending),
            "an exact poll must not reacquire a byte permit before checking its pending reply"
        );
        let (reply, permit) = match worker_rx
            .try_recv()
            .expect("accepted authority request reaches the worker lane")
        {
            StorageCommand::ResolveCommandAuthority {
                reply,
                permit: Some(permit),
                ..
            } => (reply, permit),
            other => panic!("unexpected storage command: {other:?}"),
        };
        reply
            .try_send(CommandAuthorityLookup::Collision)
            .expect("worker publishes exact authority truth");
        assert_eq!(
            exact_reader.resolve_for_submit(
                &step,
                [1; blake3::OUT_LEN],
                CommandClaimPolicy::ReserveIfAbsent,
            ),
            CommandAuthorityLookup::Collision,
            "a ready exact result must remain readable while its request owns all byte capacity"
        );
        drop(permit);
        assert_eq!(exact_bytes.load(Ordering::SeqCst), 0);
        assert_eq!(inflight_bytes.load(Ordering::SeqCst), 0);
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one production-port test proves oversize refusal, exact Arc retention, zero charge, and fail-closed science together"
    )]
    fn oversize_journal_batch_closes_without_charge_and_blocks_later_science() {
        let session_id = HostSessionId::new(0x405);
        let options = StorageJournalOptions {
            admission_capacity: 2,
            max_batch_bytes: 1,
            max_inflight_bytes: 1,
            ..StorageJournalOptions::default()
        };
        assert_eq!(options.validate(), Ok(options));
        let (worker_tx, worker_rx) = xchan::bounded(DEFAULT_COMMAND_CAPACITY);
        let admission = Arc::new(Mutex::new(AdmissionState { open: true }));
        let shared = Arc::new(JournalSessionShared::new(options.admission_capacity));
        let (_receipt_tx, receipt_rx) = xchan::bounded(options.admission_capacity);
        let publisher =
            JournalReaderPublisher::new(session_id, JournalReaderBackend::Memory, options);
        let mut journal = StorageJournalPort::new(
            worker_tx,
            admission,
            shared,
            receipt_rx,
            &publisher,
            options,
            ShutdownCommitRequirement::CommittedVolatile,
        );
        journal.command_reader = None;
        let inflight_bytes = Arc::clone(&journal.inflight_bytes);
        let mut core = HostCore::with_journal(
            session_id,
            timeout_test_world(),
            timeout_host_options(),
            Box::new(journal),
        )
        .expect("host with a one-byte production journal batch ceiling");
        let mut frontend = NullFrontend::new(core.local_port(), 0x5005);

        let first = frontend.step().expect("oversize step enters host order");
        let rejected = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(0))
            .expect("oversize admission fails closed without waiting on storage");
        assert_eq!(rejected.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(1));
        assert!(matches!(
            rejected.blocker,
            Some(HostBlocker::JournalClosed { .. })
        ));
        let retained = core
            .pending_journal_batch()
            .expect("oversize result retains the exact completed batch");
        assert_eq!(retained.id().session_id(), session_id);
        assert_eq!(retained.id().sequence(), 1);
        assert!(retained.retained_bytes() > options.max_batch_bytes);
        assert_eq!(Arc::strong_count(&retained), 2);
        let same_retained = core
            .pending_journal_batch()
            .expect("oversize batch remains retained for diagnostics");
        assert!(Arc::ptr_eq(&retained, &same_retained));
        drop(same_retained);
        assert_eq!(Arc::strong_count(&retained), 2);
        assert_eq!(inflight_bytes.load(Ordering::SeqCst), 0);
        assert!(matches!(
            worker_rx.try_recv(),
            Err(xchan::TryRecvError::Empty)
        ));
        let first_status = frontend
            .command_status(first.command_id())
            .expect("oversize command status query")
            .expect("oversize command remains queryable");
        assert!(matches!(
            first_status.journal(),
            JournalState::Failed(failure) if failure.code == "journal_closed"
        ));

        let later = frontend
            .step()
            .expect("later science command enters host order");
        let blocked = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(1))
            .expect("closed size gate blocks later science deterministically");
        assert_eq!(blocked.scientific_steps, 0);
        assert_eq!(blocked.commands_completed, 0);
        assert_eq!(
            blocked.blocker,
            Some(HostBlocker::JournalClosed {
                batch_id: retained.id(),
            })
        );
        assert!(matches!(
            core.health(),
            HostHealth::Faulted(HostFault::Journal { batch_id, failure })
                if *batch_id == retained.id() && failure.code == "journal_closed"
        ));
        assert_eq!(core.world_tick(), Tick(1));
        let still_retained = core
            .pending_journal_batch()
            .expect("later science preserves the exact closed batch");
        assert!(Arc::ptr_eq(&retained, &still_retained));
        drop(still_retained);
        assert!(matches!(
            worker_rx.try_recv(),
            Err(xchan::TryRecvError::Empty)
        ));
        let later_status = frontend
            .command_status(later.command_id())
            .expect("later command status query")
            .expect("later command remains queryable");
        assert_eq!(later_status.application(), &ApplicationState::Admitted);
        assert_eq!(later_status.journal(), &JournalState::Pending);

        drop(core);
        assert_eq!(inflight_bytes.load(Ordering::SeqCst), 0);
        assert_eq!(Arc::strong_count(&retained), 1);
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one production-port test proves byte backpressure, exact retry identity, receipt release, and drop release together"
    )]
    fn inflight_byte_limit_retains_exact_batch_until_permit_release_and_retry() {
        let (first_batch_bytes, second_batch_bytes) = measured_scientific_batch_bytes();
        assert!(first_batch_bytes > 0);
        assert!(second_batch_bytes > 0);
        let byte_limit = first_batch_bytes.max(second_batch_bytes);
        assert!(
            first_batch_bytes.saturating_add(second_batch_bytes) > byte_limit,
            "two nonempty exact charges must exceed the one-batch byte limit"
        );
        let defaults = StorageJournalOptions::default();
        let options = StorageJournalOptions {
            admission_capacity: 4,
            max_batch_bytes: byte_limit,
            max_inflight_bytes: byte_limit,
            max_event_page_bytes: defaults.max_event_page_bytes.max(byte_limit),
            ..defaults
        };
        assert_eq!(options.validate(), Ok(options));
        let session_id = HostSessionId::new(0x406);
        let (worker_tx, worker_rx) = xchan::bounded(DEFAULT_COMMAND_CAPACITY);
        let admission = Arc::new(Mutex::new(AdmissionState { open: true }));
        let shared = Arc::new(JournalSessionShared::new(options.admission_capacity));
        let (receipt_tx, receipt_rx) = xchan::bounded(options.admission_capacity);
        let publisher =
            JournalReaderPublisher::new(session_id, JournalReaderBackend::Memory, options);
        let mut journal = StorageJournalPort::new(
            worker_tx,
            admission,
            shared,
            receipt_rx,
            &publisher,
            options,
            ShutdownCommitRequirement::CommittedVolatile,
        );
        journal.command_reader = None;
        let inflight_bytes = Arc::clone(&journal.inflight_bytes);
        let mut core = HostCore::with_journal(
            session_id,
            timeout_test_world(),
            timeout_host_options(),
            Box::new(journal),
        )
        .expect("host whose byte ceiling admits exactly one scientific batch");
        let mut frontend = NullFrontend::new(core.local_port(), 0x5006);

        frontend.step().expect("first byte-budgeted step admission");
        let first_applied = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(0))
            .expect("first byte-budgeted step reaches storage");
        assert_eq!(first_applied.scientific_steps, 1);
        let first_queued = worker_rx
            .try_recv()
            .expect("first exact batch reaches the live worker lane");
        let first_worker_batch = admitted_journal_batch(first_queued)
            .expect("first byte-budgeted command is a journal admission");
        assert_eq!(first_worker_batch.retained_bytes(), first_batch_bytes);
        assert_eq!(inflight_bytes.load(Ordering::SeqCst), first_batch_bytes);

        let second = frontend
            .step()
            .expect("second byte-budgeted step enters host order");
        let full = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(1))
            .expect("second batch reaches the nonblocking byte gate");
        assert_eq!(full.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(2));
        assert!(matches!(
            full.blocker,
            Some(HostBlocker::JournalFull { capacity: 4, .. })
        ));
        let retained = core
            .pending_journal_batch()
            .expect("aggregate byte pressure retains the exact second batch");
        assert_eq!(retained.retained_bytes(), second_batch_bytes);
        assert_eq!(Arc::strong_count(&retained), 2);
        assert_eq!(inflight_bytes.load(Ordering::SeqCst), first_batch_bytes);
        assert!(matches!(
            worker_rx.try_recv(),
            Err(xchan::TryRecvError::Empty)
        ));
        let second_status = frontend
            .command_status(second.command_id())
            .expect("second command status query under byte pressure")
            .expect("second command remains queryable");
        assert_eq!(second_status.journal(), &JournalState::Pending);

        frontend
            .step()
            .expect("later science command enters host order while bytes are full");
        let still_full = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(2))
            .expect("retained second batch blocks all later science");
        assert_eq!(still_full.scientific_steps, 0);
        assert_eq!(core.world_tick(), Tick(2));
        assert!(matches!(
            still_full.blocker,
            Some(HostBlocker::JournalFull { capacity: 4, .. })
        ));
        let same_retained = core
            .pending_journal_batch()
            .expect("later drive preserves the identical retained allocation");
        assert!(Arc::ptr_eq(&retained, &same_retained));
        drop(same_retained);
        assert_eq!(Arc::strong_count(&retained), 2);

        receipt_tx
            .try_send(JournalReceipt::new(
                first_worker_batch.id(),
                JournalReceiptState::CommittedVolatile,
            ))
            .expect("first terminal receipt fits the bounded lane");
        let released = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(3))
            .expect("receipt polling releases the first exact byte permit");
        assert_eq!(released.scientific_steps, 0);
        assert_eq!(inflight_bytes.load(Ordering::SeqCst), 0);
        assert_eq!(Arc::strong_count(&first_worker_batch), 1);
        assert!(matches!(
            released.blocker,
            Some(HostBlocker::JournalFull { capacity: 4, .. })
        ));

        let retried = core
            .retry_retained_journal()
            .expect("exact retained retry preserves host invariants")
            .expect("second retained batch exists");
        assert!(matches!(
            retried,
            JournalAdmission::Accepted { batch_id } if batch_id == retained.id()
        ));
        assert!(core.pending_journal_batch().is_none());
        assert_eq!(inflight_bytes.load(Ordering::SeqCst), second_batch_bytes);
        let second_queued = worker_rx
            .try_recv()
            .expect("released bytes admit the exact retained second batch");
        let second_worker_batch = admitted_journal_batch(second_queued)
            .expect("retried byte-budgeted command is a journal admission");
        assert!(Arc::ptr_eq(&retained, &second_worker_batch));
        assert_eq!(second_worker_batch.retained_bytes(), second_batch_bytes);

        drop(core);
        assert_eq!(inflight_bytes.load(Ordering::SeqCst), 0);
        assert_eq!(Arc::strong_count(&retained), 2);
    }

    #[test]
    fn journal_receipt_timeout_is_positive_and_storage_wait_bounded() {
        let zero = StorageJournalOptions {
            receipt_timeout: Duration::ZERO,
            ..StorageJournalOptions::default()
        };
        assert_eq!(
            zero.validate(),
            Err("journal receipt_timeout must be nonzero")
        );

        let oversized = StorageJournalOptions {
            receipt_timeout: MAX_STORAGE_WAIT_TIMEOUT.saturating_add(Duration::from_nanos(1)),
            ..StorageJournalOptions::default()
        };
        assert_eq!(
            oversized.validate(),
            Err("journal receipt_timeout exceeds the storage wait ceiling")
        );
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one race test proves cache-first truth, bounded duplicate suppression, exact retry, and healthy recovery together"
    )]
    fn cached_terminal_truth_wins_expired_deadline_without_channel_notification() {
        let session_id = HostSessionId::new(0x403);
        let options = StorageJournalOptions {
            admission_capacity: 1,
            receipt_timeout: Duration::from_millis(1),
            ..StorageJournalOptions::default()
        };
        let (worker_tx, worker_rx) = xchan::bounded(DEFAULT_COMMAND_CAPACITY);
        let admission = Arc::new(Mutex::new(AdmissionState { open: true }));
        let shared = Arc::new(JournalSessionShared::new(options.admission_capacity));
        let (receipt_tx, receipt_rx) = xchan::bounded(options.admission_capacity);
        let publisher =
            JournalReaderPublisher::new(session_id, JournalReaderBackend::Memory, options);
        let mut journal = StorageJournalPort::new(
            worker_tx,
            admission,
            Arc::clone(&shared),
            receipt_rx,
            &publisher,
            options,
            ShutdownCommitRequirement::CommittedVolatile,
        );
        journal.command_reader = None;
        let inflight_bytes = Arc::clone(&journal.inflight_bytes);
        let mut core = HostCore::with_journal(
            session_id,
            timeout_test_world(),
            timeout_host_options(),
            Box::new(journal),
        )
        .expect("host whose worker caches truth before notifying the receipt lane");
        let mut frontend = NullFrontend::new(core.local_port(), 0x5003);

        let first = frontend.step().expect("first step admission");
        let applied = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(0))
            .expect("first step completes before its asynchronous receipt");
        assert_eq!(applied.scientific_steps, 1);
        let queued = worker_rx
            .try_recv()
            .expect("live worker lane owns the accepted batch");
        assert!(matches!(&queued, StorageCommand::JournalAdmit { .. }));
        let StorageCommand::JournalAdmit {
            batch: worker_batch,
        } = queued
        else {
            return;
        };
        let batch_id = worker_batch.id();
        assert_eq!(
            inflight_bytes.load(Ordering::SeqCst),
            worker_batch.retained_bytes()
        );

        std::thread::sleep(
            options
                .receipt_timeout
                .saturating_add(Duration::from_millis(1)),
        );
        shared
            .cache_terminal(batch_id, &JournalReceiptState::CommittedVolatile)
            .expect("worker publishes terminal truth before its channel notification");
        let reconciled = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(1))
            .expect("cached truth is reconciled without blocking on a channel notification");
        assert_eq!(reconciled.scientific_steps, 0);
        let first_status = frontend
            .command_status(first.command_id())
            .expect("status query after cached-truth reconciliation")
            .expect("first command remains queryable");
        assert_eq!(first_status.journal(), &JournalState::CommittedVolatile);
        assert_eq!(core.health(), &HostHealth::Healthy);
        assert_eq!(shared.cancellation_boundary(), None);
        assert_eq!(inflight_bytes.load(Ordering::SeqCst), 0);
        assert_eq!(Arc::strong_count(&worker_batch), 1);
        assert!(
            shared
                .terminal_receipts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_empty()
        );

        let second = frontend.step().expect("second step enters host order");
        let bounded = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(2))
            .expect("suppression tombstone preserves bounded journal admission");
        assert_eq!(bounded.scientific_steps, 1);
        assert!(matches!(
            bounded.blocker,
            Some(HostBlocker::JournalFull { capacity: 1, .. })
        ));
        let retained = core
            .pending_journal_batch()
            .expect("second exact batch remains retained while the tombstone consumes capacity");
        assert!(matches!(
            worker_rx.try_recv(),
            Err(xchan::TryRecvError::Empty)
        ));

        receipt_tx
            .try_send(JournalReceipt::new(
                batch_id,
                JournalReceiptState::CommittedVolatile,
            ))
            .expect("preempted worker eventually sends its matching channel notification");
        frontend
            .drive_at(&mut core, ManualInstant::from_nanos(3))
            .expect("locally reconciled receipt suppresses its later channel duplicate");
        assert!(matches!(
            core.health(),
            HostHealth::Blocked(HostBlocker::JournalFull { capacity: 1, .. })
        ));
        let retried = core
            .retry_retained_journal()
            .expect("retry after duplicate suppression preserves host invariants")
            .expect("second retained batch exists");
        assert!(matches!(
            retried,
            JournalAdmission::Accepted { batch_id: accepted } if accepted == retained.id()
        ));
        let queued_second = worker_rx
            .try_recv()
            .expect("cleared tombstone admits the exact retained batch");
        assert!(matches!(
            &queued_second,
            StorageCommand::JournalAdmit { batch } if Arc::ptr_eq(batch, &retained)
        ));
        assert!(core.pending_journal_batch().is_none());
        assert_eq!(core.health(), &HostHealth::Healthy);
        let unchanged_status = frontend
            .command_status(first.command_id())
            .expect("status query after duplicate suppression")
            .expect("first command remains queryable");
        assert_eq!(unchanged_status.journal(), &JournalState::CommittedVolatile);
        let second_status = frontend
            .command_status(second.command_id())
            .expect("second status query after exact retry")
            .expect("second command remains queryable");
        assert_eq!(second_status.journal(), &JournalState::Pending);
        assert_eq!(shared.cancellation_boundary(), None);
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one lifecycle test proves timeout, accounting release, late-receipt suppression, and host fail-closed behavior together"
    )]
    fn stuck_live_worker_times_out_once_releases_batch_and_stops_later_science() {
        let session_id = HostSessionId::new(0x402);
        let options = StorageJournalOptions {
            admission_capacity: 2,
            receipt_timeout: Duration::from_millis(1),
            ..StorageJournalOptions::default()
        };
        let (worker_tx, worker_rx) = xchan::bounded(DEFAULT_COMMAND_CAPACITY);
        let admission = Arc::new(Mutex::new(AdmissionState { open: true }));
        let shared = Arc::new(JournalSessionShared::new(options.admission_capacity));
        let (receipt_tx, receipt_rx) = xchan::bounded(options.admission_capacity);
        let publisher =
            JournalReaderPublisher::new(session_id, JournalReaderBackend::Memory, options);
        let mut journal = StorageJournalPort::new(
            worker_tx,
            admission,
            Arc::clone(&shared),
            receipt_rx,
            &publisher,
            options,
            ShutdownCommitRequirement::CommittedVolatile,
        );
        journal.command_reader = None;
        let inflight_bytes = Arc::clone(&journal.inflight_bytes);
        let mut core = HostCore::with_journal(
            session_id,
            timeout_test_world(),
            timeout_host_options(),
            Box::new(journal),
        )
        .expect("host with a live but deliberately unserviced storage worker lane");
        let mut frontend = NullFrontend::new(core.local_port(), 0x5002);

        let first = frontend.step().expect("first step admission");
        let applied = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(0))
            .expect("first step completes before its asynchronous receipt");
        assert_eq!(applied.scientific_steps, 1);
        assert_eq!(core.world_tick(), Tick(1));

        let queued = worker_rx
            .try_recv()
            .expect("live worker lane owns the accepted batch without servicing it");
        assert!(matches!(&queued, StorageCommand::JournalAdmit { .. }));
        let StorageCommand::JournalAdmit {
            batch: worker_batch,
        } = queued
        else {
            return;
        };
        let batch_id = worker_batch.id();
        assert_eq!(
            inflight_bytes.load(Ordering::SeqCst),
            worker_batch.retained_bytes()
        );

        std::thread::sleep(
            options
                .receipt_timeout
                .saturating_add(Duration::from_millis(1)),
        );
        let timed_out = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(1))
            .expect("deadline polling is nonblocking and returns terminal failure");
        assert_eq!(timed_out.scientific_steps, 0);
        assert_eq!(core.world_tick(), Tick(1));
        let first_status = frontend
            .command_status(first.command_id())
            .expect("status query after timeout")
            .expect("first command remains queryable");
        assert!(matches!(
            first_status.journal(),
            JournalState::Failed(failure)
                if failure.code == "storage_journal_receipt_timeout"
        ));
        assert!(matches!(
            core.health(),
            HostHealth::Faulted(HostFault::Journal { failure, .. })
                if failure.code == "storage_journal_receipt_timeout"
        ));
        assert_eq!(inflight_bytes.load(Ordering::SeqCst), 0);
        assert_eq!(Arc::strong_count(&worker_batch), 1);
        assert!(core.pending_journal_batch().is_none());
        assert!(
            core.retry_retained_journal()
                .expect("accepted timed-out work has no retryable retained batch")
                .is_none()
        );

        shared
            .cache_terminal(batch_id, &JournalReceiptState::CommittedVolatile)
            .expect("simulate terminal truth published after the local deadline");
        receipt_tx
            .try_send(JournalReceipt::new(
                batch_id,
                JournalReceiptState::CommittedVolatile,
            ))
            .expect("late receipt fits the still-live bounded lane");
        frontend
            .drive_at(&mut core, ManualInstant::from_nanos(2))
            .expect("late receipt is suppressed instead of becoming an unknown receipt");
        assert!(matches!(
            core.health(),
            HostHealth::Faulted(HostFault::Journal { failure, .. })
                if failure.code == "storage_journal_receipt_timeout"
        ));
        assert!(
            shared
                .terminal_receipts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_empty()
        );

        let second = frontend
            .step()
            .expect("later step enters host command order");
        let blocked = frontend
            .drive_at(&mut core, ManualInstant::from_nanos(3))
            .expect("latched journal failure rejects later science deterministically");
        assert_eq!(blocked.scientific_steps, 0);
        // Exactly one blocker is possible here, and it is `JournalClosed`.
        // `HostCore::current_blocker` is a `const fn` with strict precedence —
        // `retained_blocker`, then `event_pressure`, then `health.blocker()`, then
        // `ScientificFault` — and the receipt timeout above ran `fail_closed_batch`, which
        // latches `retained_blocker`. The accepted timed-out batch leaves no *retryable*
        // retained batch (asserted above), but the blocker latch is separate state and
        // stays. Asserting `ScientificFault` here was stale: it names the fallback arm that
        // the latch preempts.
        assert!(
            matches!(blocked.blocker, Some(HostBlocker::JournalClosed { .. })),
            "a latched journal failure must block later science as JournalClosed, got {:?}",
            blocked.blocker
        );
        assert_eq!(core.world_tick(), Tick(1));
        let second_status = frontend
            .command_status(second.command_id())
            .expect("later command status query")
            .expect("later command remains queryable");
        assert!(matches!(
            second_status.application(),
            ApplicationState::Failed(failure) if failure.code == "science_blocked"
        ));
        // Same reasoning: `fail_closed_batch` writes the `journal_closed` failure, so the
        // journal axis is deterministically `Failed`, not `Pending`.
        assert!(
            matches!(
                second_status.journal(),
                JournalState::Failed(failure) if failure.code == "journal_closed"
            ),
            "a latched journal failure must report journal_closed, got {:?}",
            second_status.journal()
        );
    }

    #[test]
    fn prepared_archive_preserves_exact_f64_during_canonical_decode() {
        const EXACT_F64: f64 = 0.025_496_361_777_186_394;
        let persistence = PersistenceBatch {
            summary: TickSummary {
                tick: Tick(1),
                agent_count: 0,
                births: 0,
                deaths: 0,
                total_energy: 0.0,
                average_energy: 0.0,
                average_health: 0.0,
                max_age: 0,
                spike_hits: 0,
            },
            epoch: 0,
            closed: true,
            metrics: vec![MetricSample::new("food.max", EXACT_F64)],
            events: Vec::new(),
            agents: Vec::new(),
            births: Vec::new(),
            deaths: Vec::new(),
            replay_events: Vec::new(),
            narrative_events: Vec::new(),
        };
        let persistence = Storage::prepare_batch(&persistence).expect("prepare exact f64 metric");
        let run_id = RunId::new(1);
        let session_id = HostSessionId::new(0x401);
        let batch_id = JournalBatchId::new(session_id, 1);
        let host_session_id = encode_journal_u64(session_id.get());
        let journal_sequence = encode_journal_u64(batch_id.sequence());
        let scientific_event_sequence = encode_journal_u64(1);
        let command = applied_lifecycle(
            CommandEnvelope::new(CommandId::new(1), HostCommand::Step),
            applied(),
        );
        let encoded_command =
            EncodedCommandLifecycle::encode(&command).expect("encode command lifecycle");
        let scientific = scientific();
        let archive = HostJournalArchiveRef {
            version: HOST_JOURNAL_ARCHIVE_VERSION,
            run_id,
            host_session_id: &host_session_id,
            journal_sequence: &journal_sequence,
            scientific_event_sequence: Some(&scientific_event_sequence),
            command_lifecycle: Some(&encoded_command),
            applied: applied(),
            scientific: Some(&scientific),
            persistence: Some(&persistence),
        };
        let (payload_json, payload_digest) =
            encode_host_journal_archive(&archive, MAX_JOURNAL_BYTES)
                .expect("encode canonical archive");
        assert!(payload_json.contains("\"value\":0.025496361777186394"));

        let decoded = HostJournalArchive::decode(
            &payload_json,
            &payload_digest,
            run_id,
            batch_id,
            MAX_JOURNAL_BYTES,
        )
        .expect("precisely parsed f64 remains canonical");
        let decoded_value = decoded
            .persistence
            .expect("decoded persistence payload")
            .metrics[0]
            .value;
        assert_eq!(decoded_value.to_bits(), EXACT_F64.to_bits());
    }

    #[test]
    fn archive_shape_matrix_accepts_only_runtime_producible_batches() {
        let applied = applied();
        let scientific = scientific();
        let step = applied_lifecycle(
            CommandEnvelope::new(CommandId::new(1), HostCommand::Step),
            applied,
        );
        let update = applied_lifecycle(
            CommandEnvelope::new(CommandId::new(2), HostCommand::UpdateConfig(Box::default())),
            applied,
        );
        let shutdown = applied_lifecycle(
            CommandEnvelope::new(CommandId::new(3), HostCommand::Shutdown),
            applied,
        );
        let control = applied_lifecycle(
            CommandEnvelope::new(CommandId::new(4), HostCommand::Pause),
            applied,
        );
        let world_edits = [
            applied_lifecycle(
                CommandEnvelope::new(
                    CommandId::new(8),
                    HostCommand::UpdateSelection(SelectionUpdate {
                        mode: scriptbots_core::SelectionMode::Clear,
                        agent_ids: Vec::new(),
                        state: scriptbots_core::SelectionState::Selected,
                    }),
                ),
                applied,
            ),
            applied_lifecycle(
                CommandEnvelope::new(
                    CommandId::new(9),
                    HostCommand::AdjustAgentMutationRates {
                        agent_uid: AgentUid(1),
                        delta_primary: 0.002,
                        delta_secondary: -0.01,
                    },
                ),
                applied,
            ),
            applied_lifecycle(
                CommandEnvelope::new(
                    CommandId::new(10),
                    HostCommand::SpawnAgent {
                        herbivore_tendency: 1.0,
                    },
                ),
                applied,
            ),
            applied_lifecycle(
                CommandEnvelope::new(
                    CommandId::new(11),
                    HostCommand::SpawnCrossover {
                        parent_a: AgentUid(1),
                        parent_b: AgentUid(2),
                    },
                ),
                applied,
            ),
        ];
        let rejected_step = rejected_lifecycle(
            CommandEnvelope::new(CommandId::new(5), HostCommand::Step),
            applied,
        );
        let rejected_shutdown = rejected_lifecycle(
            CommandEnvelope::new(CommandId::new(6), HostCommand::Shutdown),
            applied,
        );
        let failed_shutdown = failed_lifecycle(
            CommandEnvelope::new(CommandId::new(7), HostCommand::Shutdown),
            applied,
        );

        assert!(
            validate_scientific_archive_boundary(
                1,
                Some(EventSequence::new(1)),
                applied,
                Some(&scientific),
                None,
                true,
            )
            .is_ok()
        );
        assert!(
            validate_scientific_archive_boundary(
                1,
                Some(EventSequence::new(1)),
                applied,
                Some(&scientific),
                Some(&step),
                true,
            )
            .is_ok()
        );
        assert!(
            validate_scientific_archive_boundary(1, None, applied, None, Some(&update), false,)
                .is_ok()
        );
        assert!(
            validate_scientific_archive_boundary(1, None, applied, None, Some(&shutdown), true,)
                .is_ok()
        );
        assert!(
            validate_scientific_archive_boundary(1, None, applied, None, Some(&control), false,)
                .is_ok()
        );
        for world_edit in &world_edits {
            assert!(
                validate_scientific_archive_boundary(
                    1,
                    None,
                    applied,
                    None,
                    Some(world_edit),
                    false,
                )
                .is_ok(),
                "an applied GUI world edit is a terminal command-only lifecycle"
            );
        }
        assert!(
            validate_scientific_archive_boundary(
                1,
                None,
                applied,
                None,
                Some(&rejected_step),
                false,
            )
            .is_ok()
        );
        assert!(
            validate_scientific_archive_boundary(
                1,
                None,
                applied,
                None,
                Some(&rejected_shutdown),
                false,
            )
            .is_ok()
        );
        assert!(!rejected_shutdown.is_applied_shutdown());
        assert!(
            validate_scientific_archive_boundary(
                1,
                None,
                applied,
                None,
                Some(&failed_shutdown),
                false,
            )
            .is_ok()
        );

        assert!(
            validate_scientific_archive_boundary(1, None, applied, None, Some(&step), false,)
                .is_err()
        );
        assert!(
            validate_scientific_archive_boundary(
                1,
                Some(EventSequence::new(1)),
                applied,
                Some(&scientific),
                Some(&update),
                false,
            )
            .is_err()
        );
        assert!(
            validate_scientific_archive_boundary(
                1,
                Some(EventSequence::new(1)),
                applied,
                Some(&scientific),
                Some(&shutdown),
                false,
            )
            .is_err()
        );
        assert!(
            validate_scientific_archive_boundary(1, None, applied, None, Some(&update), true,)
                .is_err()
        );
        for world_edit in &world_edits {
            assert!(
                validate_scientific_archive_boundary(
                    1,
                    Some(EventSequence::new(1)),
                    applied,
                    Some(&scientific),
                    Some(world_edit),
                    false,
                )
                .is_err(),
                "a GUI world edit cannot claim a scientific boundary/event payload"
            );
            assert!(
                validate_scientific_archive_boundary(
                    1,
                    None,
                    applied,
                    None,
                    Some(world_edit),
                    true,
                )
                .is_err(),
                "a GUI world edit cannot claim a persistence payload"
            );
        }
        assert!(
            validate_scientific_archive_boundary(
                1,
                Some(EventSequence::new(1)),
                applied,
                Some(&scientific),
                Some(&rejected_step),
                false,
            )
            .is_err()
        );
        assert!(
            validate_scientific_archive_boundary(
                1,
                None,
                applied,
                None,
                Some(&rejected_shutdown),
                true,
            )
            .is_err()
        );
    }

    #[test]
    fn rejected_nonfinite_command_round_trips_losslessly_through_archive_v2() {
        let nan_bits = 0x7fc0_1234_u32;
        let speed = f32::from_bits(nan_bits);
        let lifecycle = rejected_lifecycle(
            CommandEnvelope::new(CommandId::new(0x44), HostCommand::SetSpeed(speed)),
            applied(),
        );
        let encoded =
            EncodedCommandLifecycle::encode(&lifecycle).expect("encode nonfinite rejection");
        let noncanonical = EncodedCommandLifecycle {
            schema_version: encoded.schema_version,
            postcard_hex: format!("{}00", encoded.postcard_hex),
        };
        let noncanonical_error = noncanonical
            .decode()
            .expect_err("postcard payload with ignored trailing bytes was accepted as canonical");
        assert!(matches!(
            noncanonical_error,
            StorageError::InvalidData {
                context: "host_journal_archive.command_lifecycle.postcard_hex",
                ..
            }
        ));
        let run_id = RunId::new(0x44);
        let session_id = HostSessionId::new(0x44);
        let batch_id = JournalBatchId::new(session_id, 1);
        let host_session_id = encode_journal_u64(session_id.get());
        let journal_sequence = encode_journal_u64(1);
        let archive = HostJournalArchiveRef {
            version: HOST_JOURNAL_ARCHIVE_VERSION,
            run_id,
            host_session_id: &host_session_id,
            journal_sequence: &journal_sequence,
            scientific_event_sequence: None,
            command_lifecycle: Some(&encoded),
            applied: applied(),
            scientific: None,
            persistence: None,
        };
        let (payload_json, payload_digest) =
            encode_host_journal_archive(&archive, MAX_JOURNAL_BYTES)
                .expect("encode lossless command archive");
        assert!(payload_json.contains("\"postcard_hex\""));
        let decoded = HostJournalArchive::decode(
            &payload_json,
            &payload_digest,
            run_id,
            batch_id,
            MAX_JOURNAL_BYTES,
        )
        .expect("decode lossless command archive");
        let projection = decoded
            .prepare_command_projection(&payload_digest)
            .expect("prepare normalized command")
            .expect("command projection exists");
        let HostCommand::SetSpeed(decoded_speed) = &projection.lifecycle.envelope().command else {
            panic!("decoded lifecycle changed rejected command kind");
        };
        assert_eq!(decoded_speed.to_bits(), nan_bits);
        assert_eq!(
            projection.command_payload_postcard_hex,
            encode_host_command_postcard_hex(
                "host_command_records.command_payload_postcard_hex",
                &HostCommand::SetSpeed(speed),
            )
            .expect("encode original rejected command")
        );
    }

    #[test]
    fn gui_world_command_postcard_round_trips_exact_payloads() {
        let primary_bits = 0x8000_0000_u32;
        let secondary_bits = 0x7fc0_37a1_u32;
        let mutation = HostCommand::AdjustAgentMutationRates {
            agent_uid: AgentUid(0x37),
            delta_primary: f32::from_bits(primary_bits),
            delta_secondary: f32::from_bits(secondary_bits),
        };
        let mutation_hex = encode_host_command_postcard_hex(
            "host_command_records.command_payload_postcard_hex",
            &mutation,
        )
        .expect("encode exact mutation command");
        let decoded_mutation = decode_host_command_postcard_hex(
            "host_command_records.command_payload_postcard_hex",
            &mutation_hex,
        )
        .expect("decode exact mutation command");
        let HostCommand::AdjustAgentMutationRates {
            agent_uid,
            delta_primary,
            delta_secondary,
        } = decoded_mutation
        else {
            panic!("mutation command changed durable kind");
        };
        assert_eq!(agent_uid, AgentUid(0x37));
        assert_eq!(delta_primary.to_bits(), primary_bits);
        assert_eq!(delta_secondary.to_bits(), secondary_bits);

        let tendency_bits = 0x7fc0_37b2_u32;
        let spawn = HostCommand::SpawnAgent {
            herbivore_tendency: f32::from_bits(tendency_bits),
        };
        let spawn_hex = encode_host_command_postcard_hex(
            "host_command_records.command_payload_postcard_hex",
            &spawn,
        )
        .expect("encode exact spawn command");
        let decoded_spawn = decode_host_command_postcard_hex(
            "host_command_records.command_payload_postcard_hex",
            &spawn_hex,
        )
        .expect("decode exact spawn command");
        let HostCommand::SpawnAgent { herbivore_tendency } = decoded_spawn else {
            panic!("spawn command changed durable kind");
        };
        assert_eq!(herbivore_tendency.to_bits(), tendency_bits);

        for command in [
            HostCommand::UpdateSelection(SelectionUpdate {
                mode: scriptbots_core::SelectionMode::Replace,
                agent_ids: vec![7, 11],
                state: scriptbots_core::SelectionState::Selected,
            }),
            HostCommand::SpawnCrossover {
                parent_a: AgentUid(7),
                parent_b: AgentUid(11),
            },
        ] {
            let encoded = encode_host_command_postcard_hex(
                "host_command_records.command_payload_postcard_hex",
                &command,
            )
            .expect("encode GUI world command");
            assert_eq!(
                decode_host_command_postcard_hex(
                    "host_command_records.command_payload_postcard_hex",
                    &encoded,
                )
                .expect("decode GUI world command"),
                command
            );
        }
    }

    #[test]
    fn journal_reader_rejects_receipts_from_the_wrong_storage_mode() {
        assert_eq!(
            journal_event_commitment(
                EventCatchUpGuarantee::CrashDurable,
                &JournalReceiptState::Durable,
            )
            .expect("file readers accept durable truth"),
            EventCommitment::Durable
        );
        assert_eq!(
            journal_event_commitment(
                EventCatchUpGuarantee::LiveMemory,
                &JournalReceiptState::CommittedVolatile,
            )
            .expect("memory readers accept volatile truth"),
            EventCommitment::CommittedVolatile
        );
        assert!(
            journal_event_commitment(
                EventCatchUpGuarantee::CrashDurable,
                &JournalReceiptState::CommittedVolatile,
            )
            .is_err()
        );
        assert!(
            journal_event_commitment(
                EventCatchUpGuarantee::LiveMemory,
                &JournalReceiptState::Durable,
            )
            .is_err()
        );
    }
}
