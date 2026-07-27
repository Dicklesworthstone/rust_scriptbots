// Deterministic persistence chaos coverage driven by asupersync's LabRuntime.
//
// Lab tasks never own `Storage`, `StoragePipeline`, `HostCore`, or a FrankenSQLite
// connection. They enqueue typed checkpoints; the caller thread drains those
// checkpoints after every deterministic Lab step and applies them to the real
// production worker. This keeps the deliberately `!Send + !Sync` connection on
// its owner thread while preserving a replayable scheduler boundary.

use super::*;
use asupersync::{
    lab::{DporExplorer, ExplorerConfig, LabConfig, LabRuntime, ScheduleExplorer},
    runtime::yield_now,
    types::Budget,
};
use serde::Serialize;
use std::{
    collections::VecDeque,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
};

const LAB_MAX_STEPS: u64 = 512;
const LAB_WORKERS: usize = 2;
const DPOR_MAX_RUNS: usize = 8;
const STABILITY_REPETITIONS: usize = 50;
const FIXED_SEEDS: [u64; 2] = [0x5eed_0000, 0x5eed_0001];
const TRACE_SCHEMA: &str = "scriptbots.persistence-lab-chaos.v1";

static LAB_CHAOS_SERIAL: Mutex<()> = Mutex::new(());

fn lab_chaos_serial_guard() -> MutexGuard<'static, ()> {
    LAB_CHAOS_SERIAL
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProtocolAction {
    Admit,
    RetryExactA,
    Flush,
    RetryExactB,
}

impl ProtocolAction {
    const ALL: [Self; 4] = [
        Self::Admit,
        Self::RetryExactA,
        Self::Flush,
        Self::RetryExactB,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::Admit => "admit",
            Self::RetryExactA => "retry_exact_a",
            Self::Flush => "flush",
            Self::RetryExactB => "retry_exact_b",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProtocolMode {
    File,
    Memory,
}

impl ProtocolMode {
    const fn label(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Memory => "memory",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaultGroup {
    AdmissionTransaction,
    ApplicationTransaction,
    LostRollbackAcknowledgement,
    AdmittedBeforeApplication,
    LostPostCommitReceipt,
    DurableFinalizeAndPublication,
    FlushBeforePersistence,
    AnalyticsPublication,
    ShutdownCheckpoint,
    RecoveryArchiveScan,
    DroppedAcknowledgement,
    QueueFullAndGateTimeout,
    AdmissionAcknowledgementTimeout,
    FlushAndShutdownTimeout,
    CancelledShutdownReaper,
    TerminalApplyFailure,
    ChangedPayloadCollision,
}

impl FaultGroup {
    const ALL: [Self; 17] = [
        Self::AdmissionTransaction,
        Self::ApplicationTransaction,
        Self::LostRollbackAcknowledgement,
        Self::AdmittedBeforeApplication,
        Self::LostPostCommitReceipt,
        Self::DurableFinalizeAndPublication,
        Self::FlushBeforePersistence,
        Self::AnalyticsPublication,
        Self::ShutdownCheckpoint,
        Self::RecoveryArchiveScan,
        Self::DroppedAcknowledgement,
        Self::QueueFullAndGateTimeout,
        Self::AdmissionAcknowledgementTimeout,
        Self::FlushAndShutdownTimeout,
        Self::CancelledShutdownReaper,
        Self::TerminalApplyFailure,
        Self::ChangedPayloadCollision,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::AdmissionTransaction => "admission_transaction",
            Self::ApplicationTransaction => "application_transaction",
            Self::LostRollbackAcknowledgement => "lost_rollback_acknowledgement",
            Self::AdmittedBeforeApplication => "admitted_before_application",
            Self::LostPostCommitReceipt => "lost_post_commit_receipt",
            Self::DurableFinalizeAndPublication => "durable_finalize_and_publication",
            Self::FlushBeforePersistence => "flush_before_persistence",
            Self::AnalyticsPublication => "analytics_publication",
            Self::ShutdownCheckpoint => "shutdown_checkpoint",
            Self::RecoveryArchiveScan => "recovery_archive_scan",
            Self::DroppedAcknowledgement => "dropped_acknowledgement",
            Self::QueueFullAndGateTimeout => "queue_full_and_gate_timeout",
            Self::AdmissionAcknowledgementTimeout => "admission_acknowledgement_timeout",
            Self::FlushAndShutdownTimeout => "flush_and_shutdown_timeout",
            Self::CancelledShutdownReaper => "cancelled_shutdown_reaper",
            Self::TerminalApplyFailure => "terminal_apply_failure",
            Self::ChangedPayloadCollision => "changed_payload_collision",
        }
    }
}

#[derive(Debug, Clone)]
struct LabRunMeta {
    seed: u64,
    steps: u64,
    virtual_time_nanos: u64,
    trace_fingerprint: u64,
    schedule_hash: u64,
    schedule: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
struct TraceWatermarks {
    admitted: Option<u64>,
    applied: Option<u64>,
    durable: Option<u64>,
}

impl From<PersistenceWatermarks> for TraceWatermarks {
    fn from(value: PersistenceWatermarks) -> Self {
        Self {
            admitted: value.admitted.map(PersistenceBatchId::get),
            applied: value.applied.map(PersistenceBatchId::get),
            durable: value.durable.map(PersistenceBatchId::get),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
struct TraceDomainState {
    batch_id: Option<u64>,
    watermarks: TraceWatermarks,
    exact_retry_batch_ids: Vec<u64>,
    application_count: Option<u64>,
    outbox_before_finalize: Option<u64>,
    outbox_after_finalize: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ChaosTraceV1 {
    schema: &'static str,
    seed: u64,
    mode: &'static str,
    checkpoint: String,
    fault: String,
    schedule: Vec<String>,
    virtual_time_nanos: u64,
    lab_steps: u64,
    trace_fingerprint: u64,
    schedule_hash: u64,
    exploration_runs: usize,
    exploration_classes: usize,
    domain: TraceDomainState,
    first_divergence: Option<String>,
    replay_command: String,
}

#[derive(Serialize)]
struct ChaosTraceArtifact<'a> {
    trace: &'a ChaosTraceV1,
    digest: &'a str,
}

#[derive(Debug)]
struct ProtocolObservation {
    mode: ProtocolMode,
    initial_batch_id: PersistenceBatchId,
    retry_batch_ids: Vec<PersistenceBatchId>,
    final_watermarks: PersistenceWatermarks,
    application_count: u64,
    outbox_before_finalize: Option<u64>,
    outbox_after_finalize: Option<u64>,
}

struct ProtocolRig {
    mode: ProtocolMode,
    pipeline: Option<StoragePipeline>,
    path: Option<String>,
    batch: PersistenceBatch,
    initial_batch_id: Option<PersistenceBatchId>,
    retry_batch_ids: Vec<PersistenceBatchId>,
    watermark_history: Vec<PersistenceWatermarks>,
    outbox_before_finalize: Option<u64>,
}

impl ProtocolRig {
    fn new(mode: ProtocolMode, seed: u64) -> Self {
        let (pipeline, path) = match mode {
            ProtocolMode::File => {
                let path = temp_db_path(&format!("lab-runtime-chaos-{seed:016x}"));
                let path = path.to_string_lossy().into_owned();
                let pipeline = StoragePipeline::create_unattributed_file_with_thresholds(
                    &path, 64, 4096, 1024, 1024,
                )
                .expect("create file-backed LabRuntime persistence rig");
                (pipeline, Some(path))
            }
            ProtocolMode::Memory => (
                StoragePipeline::unattributed_memory_with_thresholds(64, 4096, 1024, 1024)
                    .expect("create memory LabRuntime persistence rig"),
                None,
            ),
        };
        Self {
            mode,
            pipeline: Some(pipeline),
            path,
            batch: sample_batch(7, 2.5),
            initial_batch_id: None,
            retry_batch_ids: Vec::with_capacity(2),
            watermark_history: Vec::with_capacity(6),
            outbox_before_finalize: None,
        }
    }

    fn apply(&mut self, action: ProtocolAction) {
        let pipeline = self
            .pipeline
            .as_ref()
            .expect("protocol checkpoint applied after shutdown");
        match action {
            ProtocolAction::Admit => {
                assert!(
                    self.initial_batch_id.is_none(),
                    "LabRuntime admitted the same fresh payload twice"
                );
                let admission = pipeline
                    .submit_with_receipt(&self.batch)
                    .expect("LabRuntime admission checkpoint");
                assert_eq!(admission.tick, 7);
                assert_eq!(admission.watermarks.admitted, Some(admission.batch_id));
                assert_eq!(admission.watermarks.applied, None);
                assert_eq!(admission.watermarks.durable, None);
                self.initial_batch_id = Some(admission.batch_id);
                self.watermark_history.push(admission.watermarks);
                if let Some(path) = self.path.as_deref() {
                    let outbox_count = storage_outbox_count(path);
                    assert_eq!(
                        outbox_count, 1,
                        "an admitted-but-not-durable payload was compacted early"
                    );
                    self.outbox_before_finalize = Some(outbox_count);
                }
            }
            ProtocolAction::RetryExactA | ProtocolAction::RetryExactB => {
                let expected = self
                    .initial_batch_id
                    .expect("exact retry checkpoint ran before admission");
                let retry = pipeline
                    .submit_with_receipt(&self.batch)
                    .expect("exact LabRuntime retry");
                assert_eq!(
                    retry.batch_id, expected,
                    "identical retry allocated a second durable identity"
                );
                self.retry_batch_ids.push(retry.batch_id);
            }
            ProtocolAction::Flush => {
                let flush = pipeline
                    .flush_and_wait()
                    .expect("LabRuntime flush checkpoint");
                self.watermark_history.push(flush.watermarks);
            }
        }
    }

    fn finish(mut self) -> ProtocolObservation {
        let initial_batch_id = self
            .initial_batch_id
            .expect("LabRuntime schedule omitted admission");
        assert_eq!(
            self.retry_batch_ids.len(),
            2,
            "LabRuntime schedule omitted an exact retry"
        );
        let mut pipeline = self
            .pipeline
            .take()
            .expect("LabRuntime persistence rig already finished");
        let flush = pipeline
            .flush_and_wait()
            .expect("final LabRuntime persistence flush");
        self.watermark_history.push(flush.watermarks);
        let shutdown = pipeline
            .shutdown()
            .expect("clean LabRuntime persistence shutdown");
        self.watermark_history.push(shutdown.watermarks);
        assert_watermark_history(&self.watermark_history);
        assert_eq!(shutdown.watermarks.admitted, Some(initial_batch_id));
        assert_eq!(shutdown.watermarks.applied, Some(initial_batch_id));

        let (application_count, outbox_after_finalize) = match self.path.as_deref() {
            Some(path) => {
                assert_eq!(
                    shutdown.watermarks.durable,
                    Some(initial_batch_id),
                    "clean file shutdown did not make every admission durable"
                );
                let reader =
                    StorageReader::open(path).expect("open completed LabRuntime persistence run");
                let ledger = reader
                    .run_ledger_summary()
                    .expect("read LabRuntime persistence ledger");
                let reader_watermarks = reader
                    .persistence_watermarks()
                    .expect("read LabRuntime persistence watermarks");
                reader.close().expect("close LabRuntime persistence reader");
                assert_eq!(reader_watermarks, shutdown.watermarks);
                assert_eq!(
                    ledger.tick_count, 1,
                    "an identical retry applied or persisted the batch twice"
                );
                let outbox_count = storage_outbox_count(path);
                assert_eq!(
                    outbox_count, 0,
                    "a payload covered by the durable marker was not compacted"
                );
                (ledger.tick_count, Some(outbox_count))
            }
            None => {
                assert_eq!(
                    shutdown.watermarks.durable, None,
                    "memory storage advertised a file durability marker"
                );
                assert_eq!(shutdown.committed_tick, Some(7));
                (1, None)
            }
        };

        ProtocolObservation {
            mode: self.mode,
            initial_batch_id,
            retry_batch_ids: self.retry_batch_ids,
            final_watermarks: shutdown.watermarks,
            application_count,
            outbox_before_finalize: self.outbox_before_finalize,
            outbox_after_finalize,
        }
    }
}

fn storage_outbox_count(path: &str) -> u64 {
    let connection = open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("open LabRuntime outbox read-only");
    let count = connection
        .query_row("SELECT COUNT(*) FROM storage_outbox")
        .expect("query LabRuntime outbox count")
        .get_typed::<i64>(0)
        .expect("decode LabRuntime outbox count");
    connection.close().expect("close LabRuntime outbox reader");
    u64::try_from(count).expect("storage outbox count is nonnegative")
}

fn watermark_raw(value: Option<PersistenceBatchId>) -> u64 {
    value.map_or(0, PersistenceBatchId::get)
}

fn assert_watermark_history(history: &[PersistenceWatermarks]) {
    assert!(!history.is_empty(), "watermark history must not be empty");
    let mut previous = PersistenceWatermarks::default();
    for current in history {
        let admitted = watermark_raw(current.admitted);
        let applied = watermark_raw(current.applied);
        let durable = watermark_raw(current.durable);
        assert!(
            admitted >= applied && applied >= durable,
            "invalid persistence prefix ordering admitted={admitted} applied={applied} durable={durable}"
        );
        assert!(
            admitted >= watermark_raw(previous.admitted)
                && applied >= watermark_raw(previous.applied)
                && durable >= watermark_raw(previous.durable),
            "persistence watermarks moved backward: previous={previous:?} current={current:?}"
        );
        previous = *current;
    }
}

fn push_action<T>(pending: &Arc<Mutex<VecDeque<T>>>, action: T) {
    pending
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push_back(action);
}

fn drain_actions<T>(pending: &Arc<Mutex<VecDeque<T>>>) -> Vec<T> {
    pending
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .drain(..)
        .collect()
}

fn install_protocol_actors(
    runtime: &mut LabRuntime,
    pending: &Arc<Mutex<VecDeque<ProtocolAction>>>,
) {
    let region = runtime.state.create_root_region(Budget::INFINITE);
    let admitted = Arc::new(AtomicBool::new(false));

    let admission_queue = Arc::clone(pending);
    let admission_gate = Arc::clone(&admitted);
    let (admit_task, _) = runtime
        .state
        .create_task(region, Budget::INFINITE, async move {
            yield_now().await;
            push_action(&admission_queue, ProtocolAction::Admit);
            admission_gate.store(true, Ordering::Release);
            yield_now().await;
        })
        .expect("create LabRuntime admission actor");
    runtime.scheduler.lock().schedule(admit_task, 0);

    for (index, action) in [
        ProtocolAction::RetryExactA,
        ProtocolAction::Flush,
        ProtocolAction::RetryExactB,
    ]
    .into_iter()
    .enumerate()
    {
        let queue = Arc::clone(pending);
        let gate = Arc::clone(&admitted);
        let (task, _) = runtime
            .state
            .create_task(region, Budget::INFINITE, async move {
                while !gate.load(Ordering::Acquire) {
                    yield_now().await;
                }
                for _ in 0..=index {
                    yield_now().await;
                }
                push_action(&queue, action);
                yield_now().await;
            })
            .expect("create LabRuntime protocol actor");
        runtime.scheduler.lock().schedule(task, 0);
    }
}

fn drive_protocol_on_runtime(runtime: &mut LabRuntime, rig: &mut ProtocolRig) -> Vec<String> {
    let pending = Arc::new(Mutex::new(VecDeque::new()));
    install_protocol_actors(runtime, &pending);
    let mut executed = Vec::with_capacity(ProtocolAction::ALL.len());
    while !runtime.is_quiescent() {
        assert!(
            runtime.steps() < LAB_MAX_STEPS,
            "LabRuntime protocol exceeded its bounded step budget"
        );
        runtime.step_for_test();
        for action in drain_actions(&pending) {
            rig.apply(action);
            executed.push(action.label().to_owned());
            runtime.advance_time(1);
        }
    }
    for action in drain_actions(&pending) {
        rig.apply(action);
        executed.push(action.label().to_owned());
        runtime.advance_time(1);
    }

    let mut expected = ProtocolAction::ALL
        .into_iter()
        .map(|action| action.label())
        .collect::<Vec<_>>();
    let mut actual = executed.iter().map(String::as_str).collect::<Vec<_>>();
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(
        actual, expected,
        "LabRuntime lost or duplicated a checkpoint"
    );
    executed
}

fn run_protocol(seed: u64, mode: ProtocolMode) -> (LabRunMeta, ProtocolObservation) {
    let config = LabConfig::new(seed)
        .worker_count(LAB_WORKERS)
        .max_steps(LAB_MAX_STEPS)
        .with_default_replay_recording();
    let mut runtime = LabRuntime::new(config);
    runtime.advance_time(1_000 + seed % 97);
    let mut rig = ProtocolRig::new(mode, seed);
    let schedule = drive_protocol_on_runtime(&mut runtime, &mut rig);
    let observation = rig.finish();
    let report = runtime.report();
    assert!(
        report.lab_test_passed(),
        "LabRuntime protocol report failed: {}",
        report.to_json()
    );
    (
        LabRunMeta {
            seed,
            steps: report.steps_total,
            virtual_time_nanos: report.now_nanos,
            trace_fingerprint: report.trace_fingerprint,
            schedule_hash: runtime.certificate().hash(),
            schedule,
        },
        observation,
    )
}

fn protocol_trace(
    meta: LabRunMeta,
    observation: ProtocolObservation,
    test_name: &'static str,
) -> ChaosTraceV1 {
    ChaosTraceV1 {
        schema: TRACE_SCHEMA,
        seed: meta.seed,
        mode: observation.mode.label(),
        checkpoint: "shutdown".to_owned(),
        fault: "none".to_owned(),
        schedule: meta.schedule,
        virtual_time_nanos: meta.virtual_time_nanos,
        lab_steps: meta.steps,
        trace_fingerprint: meta.trace_fingerprint,
        schedule_hash: meta.schedule_hash,
        exploration_runs: 0,
        exploration_classes: 0,
        domain: TraceDomainState {
            batch_id: Some(observation.initial_batch_id.get()),
            watermarks: observation.final_watermarks.into(),
            exact_retry_batch_ids: observation
                .retry_batch_ids
                .into_iter()
                .map(PersistenceBatchId::get)
                .collect(),
            application_count: Some(observation.application_count),
            outbox_before_finalize: observation.outbox_before_finalize,
            outbox_after_finalize: observation.outbox_after_finalize,
        },
        first_divergence: None,
        replay_command: replay_command(meta.seed, test_name),
    }
}

fn replay_command(seed: u64, test_name: &str) -> String {
    format!(
        "SCRIPTBOTS_LAB_SEED={seed} rch exec -- cargo test -p scriptbots-storage \
         {test_name} -- --exact --nocapture --test-threads=1"
    )
}

fn trace_digest(trace: &ChaosTraceV1) -> String {
    let canonical = serde_json::to_vec(trace).expect("serialize canonical LabRuntime trace");
    blake3::hash(&canonical).to_hex().to_string()
}

fn emit_trace(trace: &ChaosTraceV1, artifact_label: &str) {
    let digest = trace_digest(trace);
    let artifact = ChaosTraceArtifact {
        trace,
        digest: &digest,
    };
    let canonical = serde_json::to_string(&artifact).expect("serialize LabRuntime trace artifact");
    println!("{canonical}");
    let Some(root) = std::env::var_os("SCRIPTBOTS_LAB_TRACE_DIR") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    std::fs::create_dir_all(&root).expect("create LabRuntime trace artifact directory");
    let path = root.join(format!(
        "{artifact_label}-seed-{:016x}-{}.json",
        trace.seed,
        &digest[..16]
    ));
    std::fs::write(path, format!("{canonical}\n")).expect("write LabRuntime trace artifact");
}

fn first_domain_divergence(trace: &ChaosTraceV1) -> Option<String> {
    let admitted = trace.domain.watermarks.admitted.unwrap_or(0);
    let applied = trace.domain.watermarks.applied.unwrap_or(0);
    let durable = trace.domain.watermarks.durable.unwrap_or(0);
    if admitted < applied || applied < durable {
        return Some("watermark_order".to_owned());
    }
    let Some(batch_id) = trace.domain.batch_id else {
        return Some("missing_batch_id".to_owned());
    };
    if trace
        .domain
        .exact_retry_batch_ids
        .iter()
        .any(|retry| *retry != batch_id)
    {
        return Some("retry_identity".to_owned());
    }
    if trace.domain.application_count != Some(1) {
        return Some("application_count".to_owned());
    }
    if trace.mode == ProtocolMode::File.label() {
        if trace.domain.watermarks.admitted != trace.domain.watermarks.durable {
            return Some("clean_shutdown_durability".to_owned());
        }
        if trace.domain.outbox_before_finalize != Some(1)
            || trace.domain.outbox_after_finalize != Some(0)
        {
            return Some("outbox_compaction_boundary".to_owned());
        }
    }
    None
}

fn install_fault_actors(runtime: &mut LabRuntime, pending: &Arc<Mutex<VecDeque<FaultGroup>>>) {
    let region = runtime.state.create_root_region(Budget::INFINITE);
    for (index, group) in FaultGroup::ALL.into_iter().enumerate() {
        let queue = Arc::clone(pending);
        let (task, _) = runtime
            .state
            .create_task(region, Budget::INFINITE, async move {
                for _ in 0..=index % 3 {
                    yield_now().await;
                }
                push_action(&queue, group);
                yield_now().await;
            })
            .expect("create LabRuntime fault actor");
        runtime.scheduler.lock().schedule(task, 0);
    }
}

fn run_fault_group(group: FaultGroup) -> Result<(), Box<dyn std::error::Error>> {
    match group {
        FaultGroup::AdmissionTransaction => {
            super::host_journal_receive_and_admission_transaction_fault_matrix_rolls_back_exactly();
        }
        FaultGroup::ApplicationTransaction => {
            super::host_journal_scientific_table_transaction_fault_matrix_recovers_exactly_once();
        }
        FaultGroup::LostRollbackAcknowledgement => {
            super::host_journal_lost_rollback_ack_is_indeterminate_but_reopen_safe();
        }
        FaultGroup::AdmittedBeforeApplication => {
            super::host_journal_post_archive_fault_recovers_and_applies_exactly_once();
        }
        FaultGroup::LostPostCommitReceipt => {
            super::host_journal_post_commit_pre_receipt_fault_reopens_without_duplicate_effects();
        }
        FaultGroup::DurableFinalizeAndPublication => {
            super::host_journal_durable_marker_and_publication_faults_fail_closed();
        }
        FaultGroup::FlushBeforePersistence => {
            super::host_journal_flush_fault_recovers_the_final_shutdown_persistence_tail();
        }
        FaultGroup::AnalyticsPublication => {
            super::host_journal_analytics_publication_fault_keeps_the_durable_event_exactly_once();
        }
        FaultGroup::ShutdownCheckpoint => {
            super::host_journal_shutdown_checkpoint_close_fault_is_typed_and_reopen_safe();
        }
        FaultGroup::RecoveryArchiveScan => {
            super::host_journal_reopen_scan_fault_releases_the_writer_without_mutation();
        }
        FaultGroup::DroppedAcknowledgement => {
            super::world_persistence_preserves_indeterminate_acknowledgement_loss()?;
        }
        FaultGroup::QueueFullAndGateTimeout => {
            super::full_queue_and_contended_gate_have_bounded_definite_non_admission()?;
        }
        FaultGroup::AdmissionAcknowledgementTimeout => {
            super::admission_ack_timeout_is_indeterminate_and_exact_retry_is_idempotent()?;
        }
        FaultGroup::FlushAndShutdownTimeout => {
            super::flush_and_shutdown_timeouts_are_bounded_and_retry_the_original_barrier()?;
        }
        FaultGroup::CancelledShutdownReaper => {
            super::timed_out_shutdown_drop_hands_worker_to_supervised_reaper()?;
        }
        FaultGroup::TerminalApplyFailure => {
            super::terminal_flush_failure_root_cause_survives_worker_join()?;
        }
        FaultGroup::ChangedPayloadCollision => {
            super::terminal_admission_failure_status_survives_worker_join()?;
        }
    }
    Ok(())
}

fn fault_trace(
    runtime: &mut LabRuntime,
    seed: u64,
    schedule: Vec<String>,
    fault: FaultGroup,
    first_divergence: Option<String>,
) -> ChaosTraceV1 {
    let report = runtime.report();
    ChaosTraceV1 {
        schema: TRACE_SCHEMA,
        seed,
        mode: "fault-matrix",
        checkpoint: fault.label().to_owned(),
        fault: fault.label().to_owned(),
        schedule,
        virtual_time_nanos: report.now_nanos,
        lab_steps: report.steps_total,
        trace_fingerprint: report.trace_fingerprint,
        schedule_hash: runtime.certificate().hash(),
        exploration_runs: 0,
        exploration_classes: 0,
        domain: TraceDomainState::default(),
        first_divergence,
        replay_command: replay_command(
            seed,
            "tests::lab_runtime_chaos::lab_runtime_fault_matrix_covers_every_persistence_boundary",
        ),
    }
}

fn panic_payload(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_owned()
    } else {
        "non-string panic payload".to_owned()
    }
}

#[test]
fn lab_runtime_dpor_and_fixed_seed_corpus_drive_real_persistence() {
    let _serial = lab_chaos_serial_guard();

    let explorer_config = ExplorerConfig::new(FIXED_SEEDS[0], DPOR_MAX_RUNS)
        .worker_count(LAB_WORKERS)
        .max_steps(LAB_MAX_STEPS);
    let mut dpor = DporExplorer::new(explorer_config.clone());
    let dpor_report = dpor.explore(|runtime| {
        let mut rig = ProtocolRig::new(ProtocolMode::Memory, runtime.config().seed);
        let _schedule = drive_protocol_on_runtime(runtime, &mut rig);
        let observation = rig.finish();
        assert_eq!(observation.application_count, 1);
    });
    assert!(
        !dpor_report.has_violations(),
        "bounded DPOR-guided persistence schedules failed at seeds {:?}",
        dpor_report.violation_seeds()
    );
    assert!(dpor_report.total_runs >= 1);
    assert!(dpor_report.total_runs <= DPOR_MAX_RUNS);
    assert!(dpor_report.certificates_consistent());

    let mut seed_explorer = ScheduleExplorer::new(
        ExplorerConfig::new(FIXED_SEEDS[0], FIXED_SEEDS.len())
            .worker_count(LAB_WORKERS)
            .max_steps(LAB_MAX_STEPS),
    );
    let seed_report = seed_explorer.explore(|runtime| {
        let mut rig = ProtocolRig::new(ProtocolMode::Memory, runtime.config().seed);
        let _schedule = drive_protocol_on_runtime(runtime, &mut rig);
        let observation = rig.finish();
        assert_eq!(observation.application_count, 1);
    });
    assert!(!seed_report.has_violations());
    assert_eq!(seed_report.total_runs, FIXED_SEEDS.len());
    assert!(seed_report.certificates_consistent());

    for seed in FIXED_SEEDS {
        let (meta, observation) = run_protocol(seed, ProtocolMode::File);
        let mut trace = protocol_trace(
            meta,
            observation,
            "tests::lab_runtime_chaos::lab_runtime_dpor_and_fixed_seed_corpus_drive_real_persistence",
        );
        trace.exploration_runs = dpor_report.total_runs + seed_report.total_runs;
        trace.exploration_classes = dpor_report.unique_classes + seed_report.unique_classes;
        assert_eq!(
            first_domain_divergence(&trace),
            None,
            "production persistence oracle rejected seed {seed}"
        );
        emit_trace(&trace, "dpor-fixed-corpus");
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the exhaustive adapter maps every production persistence fault group to one deterministic Lab checkpoint"
)]
fn lab_runtime_fault_matrix_covers_every_persistence_boundary() {
    let _serial = lab_chaos_serial_guard();
    let seed = 0xfa17_0001;
    let mut runtime = LabRuntime::new(
        LabConfig::new(seed)
            .worker_count(LAB_WORKERS)
            .max_steps(LAB_MAX_STEPS)
            .with_default_replay_recording(),
    );
    runtime.advance_time(2_000);
    let pending = Arc::new(Mutex::new(VecDeque::new()));
    install_fault_actors(&mut runtime, &pending);
    let mut schedule = Vec::with_capacity(FaultGroup::ALL.len());

    while !runtime.is_quiescent() {
        assert!(
            runtime.steps() < LAB_MAX_STEPS,
            "LabRuntime fault matrix exceeded its bounded step budget"
        );
        runtime.step_for_test();
        for group in drain_actions(&pending) {
            schedule.push(group.label().to_owned());
            let outcome = catch_unwind(AssertUnwindSafe(|| run_fault_group(group)));
            match outcome {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    let trace = fault_trace(
                        &mut runtime,
                        seed,
                        schedule.clone(),
                        group,
                        Some(error.to_string()),
                    );
                    emit_trace(&trace, "fault-matrix-failure");
                    panic!("LabRuntime fault group {} failed: {error}", group.label());
                }
                Err(payload) => {
                    let trace = fault_trace(
                        &mut runtime,
                        seed,
                        schedule.clone(),
                        group,
                        Some(panic_payload(payload.as_ref())),
                    );
                    emit_trace(&trace, "fault-matrix-panic");
                    resume_unwind(payload);
                }
            }
            runtime.advance_time(1);
        }
    }
    let mut expected = FaultGroup::ALL
        .into_iter()
        .map(|group| group.label())
        .collect::<Vec<_>>();
    let mut actual = schedule.iter().map(String::as_str).collect::<Vec<_>>();
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(actual, expected, "LabRuntime skipped a persistence fault");
    let report = runtime.report();
    assert!(
        report.lab_test_passed(),
        "LabRuntime fault report failed: {}",
        report.to_json()
    );
    let last = *FaultGroup::ALL
        .last()
        .expect("fault matrix is intentionally nonempty");
    let trace = fault_trace(&mut runtime, seed, schedule, last, None);
    emit_trace(&trace, "fault-matrix-pass");
}

#[test]
fn lab_runtime_fixed_corpus_digest_is_stable_for_fifty_repetitions() {
    let _serial = lab_chaos_serial_guard();
    for seed in FIXED_SEEDS {
        let (meta, observation) = run_protocol(seed, ProtocolMode::Memory);
        let expected = protocol_trace(
            meta,
            observation,
            "tests::lab_runtime_chaos::lab_runtime_fixed_corpus_digest_is_stable_for_fifty_repetitions",
        );
        assert_eq!(first_domain_divergence(&expected), None);
        let expected_digest = trace_digest(&expected);
        emit_trace(&expected, "stable-corpus");

        for repetition in 1..STABILITY_REPETITIONS {
            let (meta, observation) = run_protocol(seed, ProtocolMode::Memory);
            let actual = protocol_trace(
                meta,
                observation,
                "tests::lab_runtime_chaos::lab_runtime_fixed_corpus_digest_is_stable_for_fifty_repetitions",
            );
            let actual_digest = trace_digest(&actual);
            if actual_digest != expected_digest {
                let mut divergent = actual;
                divergent.first_divergence = Some(format!("stable_digest_repetition_{repetition}"));
                emit_trace(&divergent, "stable-corpus-divergence");
            }
            assert_eq!(
                actual_digest, expected_digest,
                "seed {seed} diverged at repetition {repetition}"
            );
        }
    }
}

#[test]
fn lab_runtime_mutated_negative_control_is_detected() {
    let _serial = lab_chaos_serial_guard();
    let seed = FIXED_SEEDS[0];
    let (meta, observation) = run_protocol(seed, ProtocolMode::Memory);
    let trace = protocol_trace(
        meta,
        observation,
        "tests::lab_runtime_chaos::lab_runtime_mutated_negative_control_is_detected",
    );
    assert_eq!(first_domain_divergence(&trace), None);

    let mut mutated = trace;
    mutated.domain.application_count = Some(2);
    let divergence = first_domain_divergence(&mutated);
    assert_eq!(divergence.as_deref(), Some("application_count"));
    mutated.first_divergence = divergence;
    emit_trace(&mutated, "negative-control-detected");
}

#[test]
fn lab_runtime_torn_write_refuses_recovery_without_mutation()
-> Result<(), Box<dyn std::error::Error>> {
    let _serial = lab_chaos_serial_guard();
    let path_string = super::create_complete_narrative_database("lab-torn-write", 4)?;
    let corruptor = Connection::open(&path_string)?;
    corruptor.execute_with_params(
        "UPDATE replay_events SET payload = ?1
         WHERE run_id = ?2 AND tick = ?3 AND seq = ?4",
        &[
            "{".into(),
            sqlite_run_id(RunId::new(1)),
            encode_u64("lab.torn_write.tick", 4)?.into(),
            encode_u64("lab.torn_write.seq", NARRATIVE_INPUT_REPLAY_SEQ)?.into(),
        ],
    )?;
    corruptor.close()?;

    let error = match StorageReader::open_finished(&path_string) {
        Ok(reader) => {
            reader.close()?;
            return Err("a torn JSON write was accepted".into());
        }
        Err(error) => error,
    };
    assert!(error.to_string().contains("JSON") || error.to_string().contains("json"));
    let reader = StorageReader::open_finished(&path_string);
    assert!(reader.is_err(), "torn payload must remain refused on retry");
    Ok(())
}

#[test]
fn lab_runtime_truncated_tail_refuses_recovery_without_mutation()
-> Result<(), Box<dyn std::error::Error>> {
    let _serial = lab_chaos_serial_guard();
    let path_string = super::create_complete_narrative_database("lab-truncated-tail", 4)?;
    let corruptor = Connection::open(&path_string)?;
    corruptor.execute_with_params(
        "DELETE FROM replay_events WHERE run_id = ?1 AND tick = ?2 AND seq = ?3",
        &[
            sqlite_run_id(RunId::new(1)),
            encode_u64("lab.truncated_tail.tick", 4)?.into(),
            encode_u64("lab.truncated_tail.seq", NARRATIVE_INPUT_REPLAY_SEQ)?.into(),
        ],
    )?;
    corruptor.close()?;

    let reader = StorageReader::open_finished(&path_string)
        .expect("finished reader opens the database before bounded page validation");
    let error = reader
        .narrative_input_page_v1(None, 4, 0)
        .expect_err("a missing terminal row must report a truncated tail");
    assert!(matches!(
        error,
        StorageError::NarrativeInputStream(NarrativeInputStreamError::Truncated {
            first_offending_tick: 4,
            terminal_tick: 4,
            ..
        })
    ));
    reader.close()?;
    let retry = StorageReader::open_finished(&path_string);
    assert!(
        retry.is_ok(),
        "truncated tail remains readable for bounded diagnostics"
    );
    retry?.close()?;
    Ok(())
}
