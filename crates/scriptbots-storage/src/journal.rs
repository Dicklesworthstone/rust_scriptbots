//! Nonblocking HostCore journal adapter and detached event-reader surface.

use super::{
    AdmissionState, DEFAULT_COMMAND_CAPACITY, ExistingStorageLease, InFlightPermit,
    MAX_STORAGE_QUERY_PAGE, Storage, StorageBuffer, StorageCommand, StorageError,
    load_host_journal_index, read_host_journal_events,
};
use arc_swap::ArcSwap;
use crossbeam_channel as xchan;
use scriptbots_runtime::{
    AppliedCommand, CommandEnvelope, EventCatchUp, EventCatchUpGuarantee, EventCatchUpLocator,
    EventCatchUpUnavailableReason, EventCommitment, EventJournalReader, EventPage, EventPageSource,
    EventRetentionSnapshot, EventSequence, EventSequenceRange, HostAccessError, HostCommand,
    HostSessionId, JournalAdmission, JournalBatch, JournalBatchId, JournalFailure, JournalPort,
    JournalReceipt, JournalReceiptState, JournaledScientificEvent, RunId, ScientificBoundary,
    ScientificEvent, ShutdownCommitRequirement,
};
use std::{
    collections::{BTreeMap, VecDeque},
    io::Write,
    sync::{
        Arc, Mutex, TryLockError,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};

const DEFAULT_JOURNAL_CAPACITY: usize = DEFAULT_COMMAND_CAPACITY;
const DEFAULT_JOURNAL_BATCH_BYTES: usize = 64 << 20;
const DEFAULT_JOURNAL_INFLIGHT_BYTES: usize = 256 << 20;
const DEFAULT_JOURNAL_IDENTITY_CAPACITY: usize = 512;
const DEFAULT_EVENT_PAGE_BYTES: usize = 64 << 20;
const MAX_JOURNAL_BYTES: usize = 1 << 30;
pub(super) const HOST_JOURNAL_ARCHIVE_VERSION: u32 = 1;

/// Bounded admission and catch-up limits for one HostCore journal adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageJournalOptions {
    /// Most accepted batches that may await receipt polling.
    pub admission_capacity: usize,
    /// Largest exact [`JournalBatch`] allocation accepted by the adapter.
    pub max_batch_bytes: usize,
    /// Largest total accepted allocation awaiting terminal receipts.
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

#[derive(serde::Serialize)]
struct HostJournalArchiveRef<'a> {
    version: u32,
    run_id: RunId,
    host_session_id: &'a str,
    journal_sequence: &'a str,
    scientific_event_sequence: Option<&'a str>,
    command: Option<&'a CommandEnvelope>,
    applied: AppliedCommand,
    scientific: Option<&'a ScientificBoundary>,
    persistence: Option<&'a StorageBuffer>,
}

fn validate_scientific_archive_boundary(
    journal_sequence: u64,
    event_sequence: Option<EventSequence>,
    applied: AppliedCommand,
    scientific: Option<&ScientificBoundary>,
    command: Option<&CommandEnvelope>,
    has_persistence: bool,
) -> Result<(), StorageError> {
    match command.map(|envelope| &envelope.command) {
        None if scientific.is_none() => {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.payload_json",
                reason: "journal batch must contain a command or scientific boundary".to_owned(),
            });
        }
        Some(command) if !command.requires_journal() => {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.command",
                reason: "command does not require a journal record".to_owned(),
            });
        }
        Some(HostCommand::Step) if scientific.is_none() => {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.scientific",
                reason: "a successful step command requires its scientific boundary".to_owned(),
            });
        }
        Some(HostCommand::UpdateConfig(_)) if scientific.is_some() || has_persistence => {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.command",
                reason: "an update-config journal batch must be command-only".to_owned(),
            });
        }
        Some(HostCommand::Shutdown) if scientific.is_some() => {
            return Err(StorageError::InvalidData {
                context: "host_journal_archive.command",
                reason: "a shutdown journal batch may carry only its final persistence tail"
                    .to_owned(),
            });
        }
        _ => {}
    }
    let shutdown =
        command.is_some_and(|envelope| matches!(&envelope.command, HostCommand::Shutdown));
    if has_persistence && scientific.is_none() && !shutdown {
        return Err(StorageError::InvalidData {
            context: "host_journal_archive.persistence",
            reason:
                "persistence without a scientific boundary is reserved for the final shutdown tail"
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
    command: Option<CommandEnvelope>,
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
        validate_scientific_archive_boundary(
            sequence,
            event_sequence,
            self.applied,
            self.scientific.as_ref(),
            self.command.as_ref(),
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

    pub(super) fn is_shutdown(&self) -> bool {
        self.command
            .as_ref()
            .is_some_and(|envelope| matches!(&envelope.command, HostCommand::Shutdown))
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
}

#[derive(Debug)]
pub(super) struct PreparedHostJournalArchive {
    pub(super) payload_json: String,
    pub(super) payload_digest: String,
    pub(super) persistence: Option<StorageBuffer>,
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
        batch.command(),
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
    let archive = HostJournalArchiveRef {
        version: HOST_JOURNAL_ARCHIVE_VERSION,
        run_id,
        host_session_id: &host_session_id,
        journal_sequence: &journal_sequence,
        scientific_event_sequence: scientific_event_sequence.as_deref(),
        command: batch.command(),
        applied: batch.applied(),
        scientific: batch.scientific().map(Arc::as_ref),
        persistence: persistence.as_ref(),
    };
    let (payload_json, payload_digest) = encode_host_journal_archive(&archive, maximum_bytes)?;
    Ok(PreparedHostJournalArchive {
        payload_json,
        payload_digest,
        persistence,
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
    capacity: usize,
    max_batch_bytes: usize,
    max_inflight_bytes: usize,
    inflight_bytes: Arc<AtomicUsize>,
    reader: Arc<dyn EventJournalReader>,
    shutdown_requirement: ShutdownCommitRequirement,
    expected_sequence: u64,
    last_accepted_sequence: u64,
    open: bool,
}

#[derive(Debug)]
struct OutstandingJournalBatch {
    batch: Arc<JournalBatch>,
    _permit: InFlightPermit,
}

impl std::fmt::Debug for StorageJournalPort {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StorageJournalPort")
            .field("session_id", &self.session_id)
            .field("outstanding", &self.outstanding.len())
            .field("capacity", &self.capacity)
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
        Self {
            session_id: publisher.inner.session_id,
            tx,
            admission,
            shared,
            receipts,
            outstanding: BTreeMap::new(),
            capacity: options.admission_capacity,
            max_batch_bytes: options.max_batch_bytes,
            max_inflight_bytes: options.max_inflight_bytes,
            inflight_bytes: Arc::new(AtomicUsize::new(0)),
            reader: publisher.reader(),
            shutdown_requirement,
            expected_sequence: 1,
            last_accepted_sequence: 0,
            open: true,
        }
    }

    /// Exact bytes accepted but not yet observed through terminal receipts.
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
        if self.outstanding.len() >= self.capacity {
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
                if batch
                    .command()
                    .is_some_and(|envelope| matches!(&envelope.command, HostCommand::Shutdown))
                {
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
        let mut receipts = Vec::with_capacity(limit.min(self.capacity));
        while receipts.len() < limit {
            match self.receipts.try_recv() {
                Ok(receipt) => {
                    if matches!(receipt.state(), JournalReceiptState::Failed(_)) {
                        self.open = false;
                    }
                    let terminal = matches!(
                        receipt.state(),
                        JournalReceiptState::Durable | JournalReceiptState::Failed(_)
                    ) || self.shutdown_requirement
                        == ShutdownCommitRequirement::CommittedVolatile;
                    if terminal {
                        self.shared.acknowledge(receipt.batch_id().sequence());
                        self.outstanding.remove(&receipt.batch_id());
                        match self.shared.terminal_receipts.try_lock() {
                            Ok(mut cache) => {
                                cache.remove(&receipt.batch_id());
                            }
                            Err(TryLockError::Poisoned(poisoned)) => {
                                poisoned.into_inner().remove(&receipt.batch_id());
                            }
                            Err(TryLockError::WouldBlock) => {}
                        }
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

    fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
        self.shutdown_requirement
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{
        MetricSample, PersistenceBatch, Tick, TickCombatSummary, TickEvents, TickSummary,
    };
    use scriptbots_runtime::{CommandId, HostRevisions};

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
        };
        let persistence = Storage::prepare_batch(&persistence).expect("prepare exact f64 metric");
        let run_id = RunId::new(1);
        let session_id = HostSessionId::new(0x401);
        let batch_id = JournalBatchId::new(session_id, 1);
        let host_session_id = encode_journal_u64(session_id.get());
        let journal_sequence = encode_journal_u64(batch_id.sequence());
        let scientific_event_sequence = encode_journal_u64(1);
        let command = CommandEnvelope::new(CommandId::new(1), HostCommand::Step);
        let scientific = scientific();
        let archive = HostJournalArchiveRef {
            version: HOST_JOURNAL_ARCHIVE_VERSION,
            run_id,
            host_session_id: &host_session_id,
            journal_sequence: &journal_sequence,
            scientific_event_sequence: Some(&scientific_event_sequence),
            command: Some(&command),
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
        let step = CommandEnvelope::new(CommandId::new(1), HostCommand::Step);
        let update =
            CommandEnvelope::new(CommandId::new(2), HostCommand::UpdateConfig(Box::default()));
        let shutdown = CommandEnvelope::new(CommandId::new(3), HostCommand::Shutdown);

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
