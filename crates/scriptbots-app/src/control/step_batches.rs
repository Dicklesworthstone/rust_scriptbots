//! Bounded, process-scoped plans for independently journalled owner steps.
//!
//! Reservation performs no science. Only an existing opaque batch identity can
//! advance its plan, and every attempted child identity is retained before
//! submission. A missing acknowledgement is resolved by lookup, never by
//! resubmitting a child that may have outlived volatile owner history.

use super::{
    CommandFailureDto, CommandStatusDto, ControlError, ControlHandle, MAX_STEP_BATCH,
    allocate_control_command_namespace,
};
use scriptbots_core::ControlCommand;
use scriptbots_runtime::{
    ApplicationState, CommandEnvelope, CommandStatus, HostAccessError, HostPort, JournalState,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

/// Maximum byte length of an opaque batch identity.
pub const MAX_STEP_BATCH_ID_BYTES: usize = 64;
/// Maximum constituent receipts requested on one page.
pub const MAX_STEP_BATCH_PAGE_SIZE: usize = 16;
/// Maximum serialized batch body, leaving room for the WebSocket envelope.
pub const MAX_STEP_BATCH_PAGE_BYTES: usize = 9 * 1024;
const MAX_RETAINED_STEP_BATCHES: usize = 128;
const MAX_RESERVED_STEP_SLOTS: u64 = 100_000;
const MAX_ADVANCE_ADMISSIONS: usize = 32;
const ADVANCE_BUDGET: Duration = Duration::from_millis(250);
const OBSERVATION_PARK: Duration = Duration::from_millis(1);

/// Progress of a non-atomic sequence; journal strength remains in each receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum StepBatchPhase {
    Reserved,
    Ready,
    Waiting,
    Indeterminate,
    Stopped,
    Complete,
}

/// Knowledge of one submission, separate from its owner's application state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum StepBatchSubmissionState {
    InFlight,
    Observed,
    Indeterminate,
    /// Typed owner evidence-lane pressure proved this child was not admitted.
    NotAdmitted,
}

/// One retained child identity and the last authoritative receipt observed for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct StepBatchEntryDto {
    pub index: u64,
    pub command_id: String,
    pub submission_state: StepBatchSubmissionState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt: Option<CommandStatusDto>,
    /// Transport or projection diagnostic; this never replaces an owner state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation_error: Option<CommandFailureDto>,
}

/// A retained plan and one bounded page of actual attempted constituents.
///
/// Batches belong to one control-service lifetime, including all handle clones.
/// They are not checkpointed, durable jobs, or atomic world transitions. Retain
/// `batch_id` before advancing; an unknown identity must never be recreated as
/// an execution retry. Registry capacity is reserved up front and never evicted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct StepBatchDto {
    pub batch_id: String,
    pub session_id: String,
    pub requested_count: u64,
    pub attempted_count: u64,
    /// Constituents positively observed with an owner admission sequence.
    pub admitted_count: u64,
    /// Constituents positively observed applied; unknown attempts are not counted.
    pub applied_count: u64,
    /// Terminal application and non-pending journal, including rejected/failed.
    pub settled_count: u64,
    pub phase: StepBatchPhase,
    pub advancing: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocking_index: Option<u64>,
    pub offset: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_offset: Option<u64>,
    pub entries: Vec<StepBatchEntryDto>,
}

#[derive(Default)]
struct RegistryState {
    sequence: u64,
    reserved_steps: u64,
    batches: BTreeMap<String, Arc<ReservedBatch>>,
    keys: BTreeMap<String, String>,
}

pub(super) struct StepBatchRegistry {
    scope: u64,
    state: Mutex<RegistryState>,
}

impl Default for StepBatchRegistry {
    fn default() -> Self {
        Self {
            scope: allocate_control_command_namespace(),
            state: Mutex::new(RegistryState::default()),
        }
    }
}

struct ReservedBatch {
    id: String,
    session_id: String,
    count: u64,
    advancing: AtomicBool,
    progress: Mutex<BatchProgress>,
}

#[derive(Default)]
struct BatchProgress {
    entries: Vec<RetainedStep>,
    admitted: u64,
    applied: u64,
    settled: u64,
}

struct RetainedStep {
    envelope: CommandEnvelope,
    dto: StepBatchEntryDto,
    observed: Option<ObservedProgress>,
    history_missing: bool,
}

#[derive(Clone, Copy)]
struct ObservedProgress {
    admitted: bool,
    applied: bool,
    settled: bool,
    failed: bool,
}

impl ObservedProgress {
    fn from_status(status: &CommandStatus) -> Self {
        Self {
            admitted: status.admission_sequence().is_some(),
            applied: matches!(status.application(), ApplicationState::Applied(_)),
            settled: !matches!(status.application(), ApplicationState::Admitted)
                && !matches!(status.journal(), JournalState::Pending),
            failed: matches!(
                status.application(),
                ApplicationState::Rejected(_) | ApplicationState::Failed(_)
            ) || matches!(status.journal(), JournalState::Failed(_)),
        }
    }
}

impl BatchProgress {
    fn phase(&self, count: u64) -> StepBatchPhase {
        let Some(last) = self.entries.last() else {
            return StepBatchPhase::Reserved;
        };
        if last.observed.is_some_and(|observed| observed.failed) {
            StepBatchPhase::Stopped
        } else if last.dto.submission_state == StepBatchSubmissionState::NotAdmitted {
            StepBatchPhase::Ready
        } else if last.history_missing
            || last.dto.submission_state == StepBatchSubmissionState::Indeterminate
        {
            StepBatchPhase::Indeterminate
        } else if !last.observed.is_some_and(|observed| observed.settled) {
            StepBatchPhase::Waiting
        } else if self.entries.len() as u64 == count {
            StepBatchPhase::Complete
        } else {
            StepBatchPhase::Ready
        }
    }

    fn begin_attempt(&mut self, envelope: CommandEnvelope) -> usize {
        let index = self.entries.len();
        self.entries.push(RetainedStep {
            dto: StepBatchEntryDto {
                index: index as u64,
                command_id: envelope.command_id.to_string(),
                submission_state: StepBatchSubmissionState::InFlight,
                receipt: None,
                observation_error: None,
            },
            envelope,
            observed: None,
            history_missing: false,
        });
        index
    }

    fn submission_failed(&mut self, index: usize, error: HostAccessError) {
        let entry = &mut self.entries[index];
        let (state, code) = if matches!(&error, HostAccessError::CommandEvidenceBackpressure { .. })
        {
            // HostCore reserves this audit slot before inserting a rejection or
            // queueing the envelope. This typed refusal proves non-admission;
            // timeout strings and missing status lookups prove nothing similar.
            (
                StepBatchSubmissionState::NotAdmitted,
                "step_submission_not_admitted",
            )
        } else {
            (
                StepBatchSubmissionState::Indeterminate,
                "step_submission_unconfirmed",
            )
        };
        entry.dto.submission_state = state;
        entry.dto.observation_error =
            Some(CommandFailureDto::from_diagnostic(code, &error.to_string()));
    }

    fn observe(
        &mut self,
        index: usize,
        observed: ObservedProgress,
        receipt: Result<CommandStatusDto, ControlError>,
    ) {
        let entry = &mut self.entries[index];
        if let Some(previous) = entry.observed {
            self.admitted -= u64::from(previous.admitted);
            self.applied -= u64::from(previous.applied);
            self.settled -= u64::from(previous.settled);
        }
        self.admitted += u64::from(observed.admitted);
        self.applied += u64::from(observed.applied);
        self.settled += u64::from(observed.settled);
        entry.observed = Some(observed);
        entry.history_missing = false;
        entry.dto.submission_state = StepBatchSubmissionState::Observed;
        match receipt {
            Ok(receipt) => {
                entry.dto.receipt = Some(receipt);
                entry.dto.observation_error = None;
            }
            Err(error) => {
                entry.dto.receipt = None;
                entry.dto.observation_error = Some(CommandFailureDto::from_diagnostic(
                    "step_receipt_projection_failed",
                    &error.to_string(),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{ScriptBotsConfig, WorldState};
    use scriptbots_runtime::{
        CommandAuthorityMode, CommandId, FixedDeadlineHost, HostCommand, HostCore, HostCoreOptions,
        HostSessionId, JournalAdmission, JournalBatch, JournalFailure, JournalPort, JournalReceipt,
        JournalReceiptState, ManualInstant, PlaybackSnapshot, ShutdownCommitRequirement,
        channel::{ChannelHostDriver, ChannelHostOptions},
    };
    use std::{collections::BTreeSet, collections::VecDeque, thread::JoinHandle};

    /// The test owns and drives the real sole owner. Control calls use its real
    /// channel port from worker threads; withholding a drive loses an actual
    /// admission acknowledgement rather than substituting a fake host status.
    struct Rig {
        driver: ChannelHostDriver,
        handle: ControlHandle,
        now: u64,
    }

    impl Rig {
        fn new(command_capacity: usize, journal: Option<Box<dyn JournalPort>>) -> Self {
            let world = WorldState::new(ScriptBotsConfig {
                world_width: 64,
                world_height: 64,
                food_cell_size: 16,
                rng_seed: Some(42),
                persistence_interval: 0,
                population_minimum: 0,
                population_spawn_interval: 0,
                closed: true,
                ..ScriptBotsConfig::default()
            })
            .expect("small deterministic world");
            let options = HostCoreOptions {
                initial_playback: PlaybackSnapshot {
                    paused: true,
                    speed_multiplier: 1.0,
                },
                command_capacity,
                archived_command_capacity: 1,
                ..HostCoreOptions::default()
            };
            let session = HostSessionId::new(0x57eb);
            let core = match journal {
                Some(journal) => HostCore::with_journal(session, world, options, journal),
                None => HostCore::new(session, world, options),
            }
            .expect("real host");
            let (driver, port) = ChannelHostDriver::new(
                FixedDeadlineHost::new(core),
                ChannelHostOptions {
                    status_board_capacity: 1,
                    ..ChannelHostOptions::default()
                },
            )
            .expect("real owner channel");
            Self {
                driver,
                handle: ControlHandle::new(port),
                now: 0,
            }
        }

        fn tick(&self) -> u64 {
            self.driver.host().core().latest_snapshot().world.tick
        }

        fn pump(&mut self) {
            self.now += 1_000_000;
            self.driver
                .step(ManualInstant::from_nanos(self.now))
                .expect("drive actual owner");
        }

        fn join<T>(&mut self, worker: JoinHandle<T>) -> T {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !worker.is_finished() {
                assert!(
                    Instant::now() < deadline,
                    "bounded control worker did not finish"
                );
                self.pump();
                std::thread::yield_now();
            }
            worker.join().expect("control worker joined")
        }

        fn call<T: Send + 'static>(
            &mut self,
            operation: impl FnOnce(ControlHandle) -> T + Send + 'static,
        ) -> T {
            let handle = self.handle.clone();
            self.join(std::thread::spawn(move || operation(handle)))
        }

        fn finish(&mut self, batch_id: &str) -> StepBatchDto {
            for _ in 0..32 {
                let id = batch_id.to_owned();
                let status = self
                    .call(move |handle| handle.advance_step_batch(&id))
                    .expect("advance retained batch");
                if matches!(
                    status.phase,
                    StepBatchPhase::Complete | StepBatchPhase::Stopped
                ) {
                    return status;
                }
                assert_ne!(status.phase, StepBatchPhase::Indeterminate);
            }
            panic!("small real-owner batch did not settle");
        }

        fn lose_admission_reply(&self, batch_id: &str) -> StepBatchDto {
            let handle = self.handle.clone();
            let id = batch_id.to_owned();
            let started = Instant::now();
            let status = std::thread::spawn(move || handle.advance_step_batch(&id))
                .join()
                .expect("bounded blocked submitter joined")
                .expect("indeterminate progress remains a receipt");
            assert!(started.elapsed() < Duration::from_secs(2));
            assert_eq!(status.phase, StepBatchPhase::Indeterminate);
            assert_eq!(status.attempted_count, 1);
            assert_eq!(status.admitted_count, 0);
            assert_eq!(status.applied_count, 0);
            assert_eq!(self.tick(), 0);
            assert!(
                status.entries[0]
                    .observation_error
                    .as_ref()
                    .expect("actual channel timeout")
                    .message
                    .contains("admission reply exceeded"),
                "the envelope must actually have entered ingress"
            );
            status
        }
    }

    #[test]
    fn batch_reservation_binds_count_and_capacity_without_science() {
        let rig = Rig::new(32, None);
        for key in [None, Some("legacy-key")] {
            assert!(rig.handle.step_count(2, key).is_err());
        }
        assert!(rig.handle.create_step_batch(0, None).is_err());
        assert!(
            rig.handle
                .create_step_batch(MAX_STEP_BATCH + 1, None)
                .is_err()
        );
        let reserved = rig
            .handle
            .create_step_batch(MAX_STEP_BATCH, Some("full-capacity"))
            .expect("reserve maximum plan");
        assert_eq!(reserved.phase, StepBatchPhase::Reserved);
        assert_eq!(reserved.attempted_count, 0);
        assert!(reserved.entries.is_empty());
        assert_eq!(rig.tick(), 0);
        assert!(matches!(
            rig.handle.create_step_batch(1, Some("another")),
            Err(ControlError::StepBatchCapacity(_))
        ));
        assert_eq!(
            rig.handle
                .clone()
                .create_step_batch(MAX_STEP_BATCH, Some("full-capacity"))
                .unwrap()
                .batch_id,
            reserved.batch_id,
        );
        assert!(matches!(
            rig.handle.create_step_batch(1, Some("full-capacity")),
            Err(ControlError::Conflict(_))
        ));
        let another_service = ControlHandle::new(rig.handle.host.clone());
        assert!(matches!(
            another_service.advance_step_batch(&reserved.batch_id),
            Err(ControlError::NotFound(_))
        ));
        for index in 0..MAX_RETAINED_STEP_BATCHES {
            another_service
                .create_step_batch(1, Some(&format!("key-{index}")))
                .unwrap();
        }
        assert!(matches!(
            another_service.create_step_batch(1, None),
            Err(ControlError::StepBatchCapacity(_))
        ));
        assert!(another_service.create_step_batch(1, Some("key-0")).is_ok());
        assert_eq!(rig.tick(), 0);
    }

    #[test]
    fn advance_is_bounded_and_concurrent_retries_keep_every_owner_receipt() {
        let mut rig = Rig::new(32, None);
        let reserved = rig
            .handle
            .create_step_batch(40, Some("concurrent"))
            .unwrap();
        let id = reserved.batch_id.clone();
        let first = rig
            .call(move |handle| handle.advance_step_batch(&id))
            .unwrap();
        assert!(first.attempted_count > 0);
        assert!(first.attempted_count <= MAX_ADVANCE_ADMISSIONS as u64);
        assert_ne!(first.phase, StepBatchPhase::Complete);
        assert_eq!(first.entries.len(), 1);
        assert_eq!(first.offset + 1, first.attempted_count);

        let workers: Vec<_> = (0..4)
            .map(|_| {
                let handle = rig.handle.clone();
                let id = reserved.batch_id.clone();
                std::thread::spawn(move || handle.advance_step_batch(&id))
            })
            .collect();
        for worker in workers {
            rig.join(worker).expect("concurrent advance");
        }
        let completed = rig.finish(&reserved.batch_id);
        assert_eq!(completed.phase, StepBatchPhase::Complete);
        assert_eq!(completed.attempted_count, 40);
        assert_eq!(completed.admitted_count, 40);
        assert_eq!(completed.applied_count, 40);
        assert_eq!(completed.settled_count, 40);
        assert_eq!(rig.tick(), 40);

        let mut offset = 0;
        let mut ids = BTreeSet::new();
        loop {
            let page = rig
                .handle
                .step_batch_status(&reserved.batch_id, offset, MAX_STEP_BATCH_PAGE_SIZE)
                .unwrap();
            assert!(!page.entries.is_empty());
            for (position, entry) in page.entries.iter().enumerate() {
                assert_eq!(entry.index, offset + position as u64);
                assert!(ids.insert(entry.command_id.clone()));
                let receipt = entry.receipt.as_ref().unwrap();
                assert_eq!(receipt.applied_tick, Some(entry.index + 1));
                assert_eq!(receipt.application_state, "applied");
                assert_eq!(receipt.journal_state, "committed_volatile");
            }
            if let Some(next) = page.next_offset {
                assert_eq!(next, offset + page.entries.len() as u64);
                offset = next;
            } else {
                break;
            }
        }
        assert_eq!(ids.len(), 40);
        let old_id: CommandId = serde_json::from_value(serde_json::Value::String(
            rig.handle
                .step_batch_status(&reserved.batch_id, 0, 1)
                .unwrap()
                .entries[0]
                .command_id
                .clone(),
        ))
        .unwrap();
        assert!(
            rig.driver
                .host()
                .core()
                .local_port()
                .command_status(old_id)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            rig.handle
                .advance_step_batch(&reserved.batch_id)
                .unwrap()
                .phase,
            StepBatchPhase::Complete
        );
        assert_eq!(
            rig.tick(),
            40,
            "evicted owner receipts must not cause replay"
        );
    }

    #[test]
    fn late_admission_acknowledgement_is_reconciled_without_an_extra_step() {
        let mut rig = Rig::new(32, None);
        let reserved = rig.handle.create_step_batch(2, None).unwrap();
        let unknown = rig.lose_admission_reply(&reserved.batch_id);
        let first_id = unknown.entries[0].command_id.clone();
        rig.pump();
        rig.pump();
        assert_eq!(rig.tick(), 1, "the timed-out envelope really applied");
        let completed = rig.finish(&reserved.batch_id);
        assert_eq!(completed.phase, StepBatchPhase::Complete);
        assert_eq!(completed.attempted_count, 2);
        assert_eq!(rig.tick(), 2);
        let page = rig
            .handle
            .step_batch_status(&reserved.batch_id, 0, 2)
            .unwrap();
        assert_eq!(page.entries[0].command_id, first_id);
        assert_eq!(
            page.entries[0].receipt.as_ref().unwrap().applied_tick,
            Some(1)
        );
        assert_ne!(page.entries[0].command_id, page.entries[1].command_id);
    }

    #[test]
    fn unknown_attempt_stays_unresolved_after_real_volatile_history_eviction() {
        let mut rig = Rig::new(32, None);
        let reserved = rig.handle.create_step_batch(2, None).unwrap();
        rig.lose_admission_reply(&reserved.batch_id);
        let batch = rig.handle.retained_step_batch(&reserved.batch_id).unwrap();
        let original = batch.progress().unwrap().entries[0].envelope.clone();
        rig.pump();
        rig.pump();
        for index in 0..3 {
            rig.driver
                .host()
                .core()
                .local_port()
                .submit(CommandEnvelope::new(
                    CommandId::new(0x8000 + index),
                    HostCommand::Step,
                ))
                .unwrap();
            rig.pump();
            rig.pump();
        }
        assert_eq!(rig.tick(), 4);
        assert!(
            rig.driver
                .host()
                .core()
                .local_port()
                .command_status(original.command_id)
                .unwrap()
                .is_none()
        );
        assert!(
            rig.handle
                .host
                .try_published_command_status(original.command_id)
                .unwrap()
                .is_none()
        );
        for _ in 0..3 {
            let id = reserved.batch_id.clone();
            let unresolved = rig
                .call(move |handle| handle.advance_step_batch(&id))
                .unwrap();
            assert_eq!(unresolved.phase, StepBatchPhase::Indeterminate);
            assert_eq!(unresolved.attempted_count, 1);
            assert_eq!(
                unresolved.applied_count, 0,
                "application was never observed by this batch"
            );
            assert_eq!(
                unresolved.entries[0].command_id,
                original.command_id.to_string()
            );
            assert_eq!(rig.tick(), 4);
        }
        // Negative control: the owner really would admit this evicted identity
        // again. The batch's refusal to resubmit is what prevents another tick.
        let repeated = rig
            .driver
            .host()
            .core()
            .local_port()
            .submit(original)
            .unwrap();
        assert!(matches!(repeated.application(), ApplicationState::Admitted));
        rig.pump();
        assert_eq!(rig.tick(), 5);
    }

    #[test]
    fn a_known_pre_admission_rejection_stops_the_plan_when_capacity_returns() {
        let mut rig = Rig::new(1, None);
        let mut owner = rig.driver.host().core().local_port();
        owner
            .submit(CommandEnvelope::new(
                CommandId::new(0x801),
                HostCommand::Pause,
            ))
            .unwrap();
        let reserved = rig.handle.create_step_batch(2, None).unwrap();
        let batch = rig.handle.retained_step_batch(&reserved.batch_id).unwrap();
        let envelope = rig
            .handle
            .prepare_control_command(ControlCommand::Step, None)
            .unwrap();
        let index = batch.progress().unwrap().begin_attempt(envelope.clone());
        let rejected = owner.submit(envelope.clone()).unwrap();
        assert!(matches!(
            rejected.application(),
            ApplicationState::Rejected(_)
        ));
        assert!(rejected.admission_sequence().is_none());
        let observed = ObservedProgress::from_status(&rejected);
        batch
            .progress()
            .unwrap()
            .observe(index, observed, rig.handle.status_dto(rejected));
        let stopped = rig.finish(&reserved.batch_id);
        assert_eq!(stopped.phase, StepBatchPhase::Stopped);
        assert_eq!(stopped.attempted_count, 1);
        assert_eq!(stopped.admitted_count, 0);
        assert_eq!(stopped.applied_count, 0);
        assert_eq!(
            stopped.entries[0].command_id,
            envelope.command_id.to_string()
        );
        assert_eq!(
            stopped.entries[0]
                .receipt
                .as_ref()
                .unwrap()
                .application_state,
            "rejected"
        );
        assert_eq!(
            owner.queue_depth(),
            0,
            "the original queue pressure has cleared"
        );
        assert_eq!(rig.tick(), 0);
        let id = reserved.batch_id.clone();
        let again = rig
            .call(move |handle| handle.advance_step_batch(&id))
            .unwrap();
        assert_eq!(again.phase, StepBatchPhase::Stopped);
        assert_eq!(again.attempted_count, 1);
        assert_eq!(
            rig.tick(),
            0,
            "a recorded refusal is not a new retry opportunity"
        );
    }

    #[test]
    fn definite_evidence_pressure_retries_the_exact_unadmitted_envelope() {
        let mut rig = Rig::new(1, None);
        let mut owner = rig.driver.host().core().local_port();
        let first = owner
            .submit(CommandEnvelope::new(
                CommandId::new(0x901),
                HostCommand::Pause,
            ))
            .unwrap();
        let refused = owner
            .submit(CommandEnvelope::new(
                CommandId::new(0x902),
                HostCommand::Step,
            ))
            .unwrap();
        assert!(matches!(
            refused.application(),
            ApplicationState::Rejected(_)
        ));
        assert!(refused.admission_sequence().is_none());
        let reserved = rig.handle.create_step_batch(1, None).unwrap();
        let batch = rig.handle.retained_step_batch(&reserved.batch_id).unwrap();
        let envelope = rig
            .handle
            .prepare_control_command(ControlCommand::Step, None)
            .unwrap();
        let index = batch.progress().unwrap().begin_attempt(envelope.clone());
        let pressure = owner
            .submit(envelope.clone())
            .expect_err("actual audit lane is full");
        assert!(matches!(
            &pressure,
            HostAccessError::CommandEvidenceBackpressure { capacity: 1 }
        ));
        assert!(owner.command_status(envelope.command_id).unwrap().is_none());
        assert_eq!(owner.queue_depth(), 1);
        assert_eq!(rig.tick(), 0);
        batch.progress().unwrap().submission_failed(index, pressure);
        let ready = rig
            .handle
            .step_batch_status(&reserved.batch_id, 0, 1)
            .unwrap();
        assert_eq!(ready.phase, StepBatchPhase::Ready);
        assert_eq!(
            ready.entries[0].submission_state,
            StepBatchSubmissionState::NotAdmitted
        );
        assert_eq!(ready.admitted_count, 0);
        assert_eq!(ready.applied_count, 0);
        assert_eq!(owner.queue_depth(), 1, "status must not retry a submission");
        rig.pump();
        rig.pump();
        let completed = rig.finish(&reserved.batch_id);
        assert_eq!(completed.phase, StepBatchPhase::Complete);
        assert_eq!(
            completed.attempted_count, 1,
            "network retry retains one constituent"
        );
        assert_eq!(
            completed.entries[0].command_id,
            envelope.command_id.to_string()
        );
        assert_eq!(
            completed.entries[0]
                .receipt
                .as_ref()
                .unwrap()
                .admission_sequence,
            Some(first.admission_sequence().unwrap().get() + 1)
        );
        assert_eq!(rig.tick(), 1);
        rig.handle.advance_step_batch(&reserved.batch_id).unwrap();
        assert_eq!(rig.tick(), 1);
    }

    struct FailSecondStepJournal {
        receipts: VecDeque<JournalReceipt>,
    }

    impl JournalPort for FailSecondStepJournal {
        fn try_admit(&mut self, batch: &Arc<JournalBatch>) -> JournalAdmission {
            let state = if batch.applied().tick.0 == 2 {
                JournalReceiptState::Failed(JournalFailure {
                    code: "journal_test_failure".into(),
                    message: "the second step applied but its journal failed".into(),
                })
            } else {
                JournalReceiptState::CommittedVolatile
            };
            self.receipts
                .push_back(JournalReceipt::new(batch.id(), state));
            JournalAdmission::Accepted {
                batch_id: batch.id(),
            }
        }

        fn poll_receipts(&mut self, limit: usize) -> Vec<JournalReceipt> {
            let count = limit.min(self.receipts.len());
            self.receipts.drain(..count).collect()
        }

        fn command_authority_mode(&self) -> CommandAuthorityMode {
            CommandAuthorityMode::ProcessLocal
        }

        fn shutdown_commit_requirement(&self) -> ShutdownCommitRequirement {
            ShutdownCommitRequirement::CommittedVolatile
        }
    }

    #[test]
    fn actual_journal_failure_keeps_the_applied_prefix_and_stops_later_steps() {
        let mut rig = Rig::new(
            32,
            Some(Box::new(FailSecondStepJournal {
                receipts: VecDeque::new(),
            })),
        );
        let reserved = rig.handle.create_step_batch(3, None).unwrap();
        let stopped = rig.finish(&reserved.batch_id);
        assert_eq!(stopped.phase, StepBatchPhase::Stopped);
        assert_eq!(stopped.attempted_count, 2);
        assert_eq!(stopped.admitted_count, 2);
        assert_eq!(stopped.applied_count, 2);
        assert_eq!(stopped.blocking_index, Some(1));
        assert_eq!(rig.tick(), 2);
        let receipt = stopped.entries[0].receipt.as_ref().unwrap();
        assert_eq!(receipt.applied_tick, Some(2));
        assert_eq!(receipt.application_state, "applied");
        assert_eq!(receipt.journal_state, "failed");
        assert_eq!(
            receipt.journal_failure.as_ref().unwrap().code,
            "journal_test_failure"
        );
        let again = rig.handle.advance_step_batch(&reserved.batch_id).unwrap();
        assert_eq!(again.attempted_count, 2);
        assert_eq!(rig.tick(), 2);
        assert_eq!(
            rig.handle
                .step_batch_status(&reserved.batch_id, 0, 1)
                .unwrap()
                .entries[0]
                .receipt
                .as_ref()
                .unwrap()
                .applied_tick,
            Some(1)
        );
    }

    #[test]
    fn receipt_pages_bound_exact_encoded_bytes_and_always_advance_the_cursor() {
        let failure = CommandFailureDto {
            code: "\u{1}".repeat(CommandFailureDto::MAX_CODE_BYTES),
            message: "\u{1}".repeat(CommandFailureDto::MAX_MESSAGE_BYTES),
            diagnostic_truncated: true,
        };
        let mut progress = BatchProgress::default();
        for index in 0..3 {
            let envelope = CommandEnvelope::new(CommandId::new(index + 1), HostCommand::Step);
            let id = envelope.command_id.to_string();
            let at = progress.begin_attempt(envelope);
            progress.observe(
                at,
                ObservedProgress {
                    admitted: true,
                    applied: false,
                    settled: true,
                    failed: true,
                },
                Ok(CommandStatusDto {
                    command_id: id,
                    admission_sequence: Some(u64::MAX),
                    application_state: "failed".into(),
                    rejection_reason: None,
                    application_failure: Some(failure.clone()),
                    journal_state: "failed".into(),
                    journal_failure: Some(failure.clone()),
                    applied_tick: None,
                    control_revision: u64::MAX,
                    scientific_revision: u64::MAX,
                    config_revision: u64::MAX,
                }),
            );
        }
        // Declared wire fixture for maximum escaped diagnostics; the owner
        // tests above cover actual science, failures, and application ordering.
        let batch = ReservedBatch {
            id: "b".repeat(MAX_STEP_BATCH_ID_BYTES),
            session_id: u64::MAX.to_string(),
            count: 3,
            advancing: AtomicBool::new(false),
            progress: Mutex::new(progress),
        };
        for offset in 0..3 {
            let page = batch.page(offset, MAX_STEP_BATCH_PAGE_SIZE).unwrap();
            assert_eq!(
                page.entries.len(),
                1,
                "the byte bound must shrink an item-limited page"
            );
            assert_eq!(page.entries[0].index, offset);
            assert_eq!(page.next_offset, (offset < 2).then_some(offset + 1));
            assert!(serde_json::to_vec(&page).unwrap().len() <= MAX_STEP_BATCH_PAGE_BYTES);
        }
        assert!(batch.page(0, 0).is_err());
        assert!(batch.page(0, MAX_STEP_BATCH_PAGE_SIZE + 1).is_err());
        assert!(batch.page(4, 1).is_err());
    }
}

struct AdvanceLease<'a>(&'a AtomicBool);

impl Drop for AdvanceLease<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl ReservedBatch {
    fn progress(&self) -> Result<MutexGuard<'_, BatchProgress>, ControlError> {
        self.progress.lock().map_err(|_| {
            ControlError::Conflict(
                "step batch progress is unavailable after a poisoned lock".into(),
            )
        })
    }

    fn try_advance(&self) -> Option<AdvanceLease<'_>> {
        self.advancing
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| AdvanceLease(&self.advancing))
    }

    fn page(&self, offset: u64, limit: usize) -> Result<StepBatchDto, ControlError> {
        let progress = self.progress()?;
        self.page_from_progress(&progress, offset, limit)
    }

    fn page_from_progress(
        &self,
        progress: &BatchProgress,
        offset: u64,
        limit: usize,
    ) -> Result<StepBatchDto, ControlError> {
        if !(1..=MAX_STEP_BATCH_PAGE_SIZE).contains(&limit) {
            return Err(ControlError::InvalidPatch(format!(
                "step batch page limit must be 1..={MAX_STEP_BATCH_PAGE_SIZE}"
            )));
        }
        let attempted = progress.entries.len() as u64;
        if offset > attempted {
            return Err(ControlError::InvalidPatch(format!(
                "step batch offset {offset} exceeds {attempted} attempted entries"
            )));
        }
        let phase = progress.phase(self.count);
        let mut page = StepBatchDto {
            batch_id: self.id.clone(),
            session_id: self.session_id.clone(),
            requested_count: self.count,
            attempted_count: attempted,
            admitted_count: progress.admitted,
            applied_count: progress.applied,
            settled_count: progress.settled,
            phase,
            advancing: self.advancing.load(Ordering::Acquire),
            blocking_index: matches!(
                phase,
                StepBatchPhase::Waiting | StepBatchPhase::Indeterminate | StepBatchPhase::Stopped
            )
            .then(|| attempted - 1),
            offset,
            next_offset: None,
            entries: Vec::new(),
        };
        let start = usize::try_from(offset).map_err(|_| {
            ControlError::InvalidPatch("step batch offset does not fit this target".into())
        })?;
        for entry in progress.entries.iter().skip(start).take(limit) {
            page.entries.push(entry.dto.clone());
            let next = offset + page.entries.len() as u64;
            page.next_offset = (next < attempted).then_some(next);
            if serde_json::to_vec(&page)
                .map_err(ControlError::serialization)?
                .len()
                > MAX_STEP_BATCH_PAGE_BYTES
            {
                page.entries.pop();
                if page.entries.is_empty() {
                    return Err(ControlError::PayloadTooLarge(
                        "a retained step receipt exceeds the batch page byte budget".into(),
                    ));
                }
                page.next_offset = Some(offset + page.entries.len() as u64);
                break;
            }
        }
        Ok(page)
    }

    fn last_page(&self) -> Result<StepBatchDto, ControlError> {
        let progress = self.progress()?;
        let offset = progress.entries.len().saturating_sub(1) as u64;
        self.page_from_progress(&progress, offset, 1)
    }
}

impl ControlHandle {
    /// Reserve bounded metadata without submitting a scientific command.
    ///
    /// A key binds the count only for this retained registry lifetime. Once the
    /// identity is returned, execution retries must use that exact batch ID.
    pub fn create_step_batch(
        &self,
        count: u64,
        idempotency_key: Option<&str>,
    ) -> Result<StepBatchDto, ControlError> {
        if !(1..=MAX_STEP_BATCH).contains(&count) {
            return Err(ControlError::InvalidPatch(format!(
                "step batch count must be 1..={MAX_STEP_BATCH}"
            )));
        }
        if let Some(key) = idempotency_key
            && (key.is_empty() || key.len() > 1024)
        {
            return Err(ControlError::InvalidPatch(
                "idempotency key must contain 1..=1024 bytes".into(),
            ));
        }
        let mut registry = self.step_batches.state.lock().map_err(|_| {
            ControlError::Conflict(
                "step batch registry is unavailable after a poisoned lock".into(),
            )
        })?;
        // A retained retry still works when all new-reservation capacity is used.
        if let Some(id) = idempotency_key.and_then(|key| registry.keys.get(key)) {
            let batch = Arc::clone(registry.batches.get(id).ok_or_else(|| {
                ControlError::Conflict("step batch key has no retained reservation".into())
            })?);
            if batch.count != count {
                return Err(ControlError::Conflict(format!(
                    "step batch '{}' already binds this key to count {}",
                    batch.id, batch.count,
                )));
            }
            drop(registry);
            return batch.last_page();
        }
        if registry.batches.len() >= MAX_RETAINED_STEP_BATCHES
            || count > MAX_RESERVED_STEP_SLOTS.saturating_sub(registry.reserved_steps)
        {
            return Err(ControlError::StepBatchCapacity(format!(
                "this control-service lifetime retains at most {MAX_RETAINED_STEP_BATCHES} batches and {MAX_RESERVED_STEP_SLOTS} reserved steps; existing batches remain readable and resumable"
            )));
        }
        let sequence = registry.sequence.checked_add(1).ok_or_else(|| {
            ControlError::StepBatchCapacity("batch identity sequence exhausted".into())
        })?;
        let id = format!("sb1-{:016x}-{sequence:016x}", self.step_batches.scope);
        let batch = Arc::new(ReservedBatch {
            id: id.clone(),
            session_id: self.host.session_id().get().to_string(),
            count,
            advancing: AtomicBool::new(false),
            progress: Mutex::new(BatchProgress::default()),
        });
        registry.sequence = sequence;
        registry.reserved_steps += count;
        registry.batches.insert(id.clone(), Arc::clone(&batch));
        if let Some(key) = idempotency_key {
            registry.keys.insert(key.to_owned(), id);
        }
        drop(registry);
        batch.page(0, 1)
    }

    fn retained_step_batch(&self, batch_id: &str) -> Result<Arc<ReservedBatch>, ControlError> {
        if batch_id.is_empty() || batch_id.len() > MAX_STEP_BATCH_ID_BYTES {
            return Err(ControlError::InvalidPatch(format!(
                "batch_id must contain 1..={MAX_STEP_BATCH_ID_BYTES} bytes"
            )));
        }
        self.step_batches
            .state
            .lock()
            .map_err(|_| {
                ControlError::Conflict("step batch registry is unavailable after a poisoned lock".into())
            })?
            .batches
            .get(batch_id)
            .cloned()
            .ok_or_else(|| {
                ControlError::NotFound(format!(
                    "step batch '{batch_id}' is not retained by this control service; do not recreate it as an execution retry"
                ))
            })
    }

    /// Look up the current unresolved constituent, without ever resubmitting it.
    fn observe_step_batch_tail(
        &self,
        batch: &ReservedBatch,
        deadline: Instant,
    ) -> Result<bool, ControlError> {
        let pending = {
            let progress = batch.progress()?;
            progress.entries.last().and_then(|entry| {
                (entry.dto.submission_state != StepBatchSubmissionState::NotAdmitted
                    && !entry.observed.is_some_and(|observed| observed.settled))
                .then_some((progress.entries.len() - 1, entry.envelope.command_id))
            })
        };
        let Some((index, command_id)) = pending else {
            return Ok(true);
        };
        match self.host.command_status_before(command_id, deadline) {
            Ok(Some(status)) => {
                let observed = ObservedProgress::from_status(&status);
                let receipt = self.status_dto(status);
                batch.progress()?.observe(index, observed, receipt);
                Ok(true)
            }
            Ok(None) => {
                let mut progress = batch.progress()?;
                let entry = &mut progress.entries[index];
                entry.history_missing = true;
                entry.dto.observation_error = Some(CommandFailureDto::from_diagnostic(
                    "step_receipt_unavailable",
                    "the owner does not retain this attempted identity; absence does not prove non-admission and this child will not be resubmitted",
                ));
                Ok(false)
            }
            Err(error) => {
                let mut progress = batch.progress()?;
                progress.entries[index].dto.observation_error =
                    Some(CommandFailureDto::from_diagnostic(
                        "step_receipt_lookup_failed",
                        &error.to_string(),
                    ));
                Ok(false)
            }
        }
    }

    /// Admit a bounded prefix of the remaining plan, preserving every attempt.
    ///
    /// One absolute deadline bounds all host calls in this turn. A child must
    /// finish application and journal settlement before the next is submitted.
    /// Other commands may interleave; already completed science is never undone.
    pub fn advance_step_batch(&self, batch_id: &str) -> Result<StepBatchDto, ControlError> {
        let batch = self.retained_step_batch(batch_id)?;
        let Some(lease) = batch.try_advance() else {
            return batch.last_page();
        };
        let deadline = Instant::now() + ADVANCE_BUDGET;
        let mut admitted_this_turn = 0;
        while Instant::now() < deadline {
            if !self.observe_step_batch_tail(&batch, deadline)? {
                break;
            }
            let phase = batch.progress()?.phase(batch.count);
            match phase {
                StepBatchPhase::Stopped | StepBatchPhase::Complete => break,
                StepBatchPhase::Indeterminate => break,
                StepBatchPhase::Waiting => {
                    std::thread::park_timeout(
                        OBSERVATION_PARK.min(deadline.saturating_duration_since(Instant::now())),
                    );
                    continue;
                }
                StepBatchPhase::Reserved | StepBatchPhase::Ready => {}
            }
            if admitted_this_turn >= MAX_ADVANCE_ADMISSIONS || Instant::now() >= deadline {
                break;
            }
            let retained_retry = {
                let mut progress = batch.progress()?;
                let index = progress.entries.len().saturating_sub(1);
                progress.entries.last_mut().and_then(|entry| {
                    (entry.dto.submission_state == StepBatchSubmissionState::NotAdmitted).then(
                        || {
                            entry.dto.submission_state = StepBatchSubmissionState::InFlight;
                            entry.dto.observation_error = None;
                            (index, entry.envelope.clone())
                        },
                    )
                })
            };
            let (index, envelope) = if let Some(retained) = retained_retry {
                retained
            } else {
                let envelope = self.prepare_control_command(ControlCommand::Step, None)?;
                let index = batch.progress()?.begin_attempt(envelope.clone());
                (index, envelope)
            };
            // The identity is now visible to readers even if admission's reply
            // is lost, this caller disconnects, or unwinding releases the lease.
            admitted_this_turn += 1;
            match self.host.submit_before(envelope, deadline) {
                Ok(status) => {
                    let observed = ObservedProgress::from_status(&status);
                    let receipt = self.status_dto(status);
                    batch.progress()?.observe(index, observed, receipt);
                }
                Err(error) => {
                    batch.progress()?.submission_failed(index, error);
                    break;
                }
            }
        }
        drop(lease);
        batch.last_page()
    }

    /// Read a bounded page; refresh only the unsettled tail and never admit work.
    pub fn step_batch_status(
        &self,
        batch_id: &str,
        offset: u64,
        limit: usize,
    ) -> Result<StepBatchDto, ControlError> {
        let batch = self.retained_step_batch(batch_id)?;
        // Validate the requested page before doing an owner round-trip.
        batch.page(offset, limit)?;
        if let Some(lease) = batch.try_advance() {
            self.observe_step_batch_tail(&batch, Instant::now() + ADVANCE_BUDGET)?;
            drop(lease);
        }
        batch.page(offset, limit)
    }
}
