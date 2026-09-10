//! Sole-owner host thread: the only place that owns `WorldState`.
//!
//! bd-pcfj transfers world ownership from `Arc<Mutex<WorldState>>` to a
//! `HostCore`. `HostCore` is deliberately same-thread — its admission state is
//! `Rc<RefCell<SharedHostState>>` and it is therefore `!Send` — so it cannot be
//! built on the main thread and moved here. It has to be constructed *inside*
//! the thread that will own it, which is what this module does.
//!
//! Everything else reaches the host through [`ChannelHostPort`], which is
//! `Send + Sync + Clone` and carries the full client contract: submit,
//! command-status lookup, snapshot subscriptions and event cursors. That is the
//! ownership model `scriptbots_runtime::channel` was built for, rather than one
//! invented here.
//!

use anyhow::{Context, Result, anyhow};
use scriptbots_core::{
    CharacterizationError, PersistenceAdmissionSession, WorldDigestV1, WorldState,
};
use scriptbots_runtime::channel::{
    ChannelHostDriver, ChannelHostOptions, ChannelHostPort, ChannelRunOutcome, ChannelRunReceipt,
};
use scriptbots_runtime::{
    ApplicationState, CommandEnvelope, CommandId, FixedDeadlineHost, HostCommand, HostCore,
    HostCoreOptions, HostFault, HostPort, HostSessionId, JournalPort, JournalState,
    ManualHostDriver, ManualInstant, ScheduledConfigPatch, ShutdownCommitRequirement,
};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::{Builder, JoinHandle};
use std::time::{Duration, Instant};

type BootstrapObserver = Box<dyn FnOnce(&WorldState) -> Result<()> + Send>;

/// Work completed by the sole owner before any external client receives a port.
#[derive(Default)]
pub struct HostBootstrap {
    /// Explicit startup science transitions; zero keeps the initial world at tick zero.
    pub ticks: u64,
    /// Immutable tick-ordered launch schedule, also retained for the interactive run.
    pub schedule: Vec<ScheduledConfigPatch>,
    /// Read-only observation after bootstrap and its journal commitments complete.
    pub on_completed: Option<BootstrapObserver>,
}

/// A running host thread and the handle everyone else talks to it through.
pub struct HostThread {
    port: ChannelHostPort,
    handle: JoinHandle<Result<HostThreadReceipt>>,
}

/// Owner observations when the drive loop exits; a fault does not prove finalization.
#[derive(Debug)]
pub struct HostThreadReceipt {
    pub run: ChannelRunReceipt,
    pub snapshot: std::sync::Arc<scriptbots_runtime::RenderSnapshot>,
    pub sense_saturations_total: u64,
    pub required_persistence_tick: Option<u64>,
    /// Canonical digest captured on the owner after the loop exits. A blocked
    /// scientific boundary retains its typed refusal instead of a stale digest.
    pub final_digest: Result<WorldDigestV1, CharacterizationError>,
}

/// A terminal host fault together with its last owner observations.
///
/// This receipt describes the failed run, not a successfully persisted tail.
#[derive(Debug, thiserror::Error)]
#[error("host drive loop faulted: {fault:?}")]
pub struct HostThreadFaultError {
    pub fault: HostFault,
    pub receipt: HostThreadReceipt,
}

impl HostThread {
    /// Move a bootstrapped world onto its own thread and start driving it.
    ///
    /// The world is passed by value because this call is the ownership
    /// transfer: after it returns, nothing outside the host thread can reach
    /// `WorldState` except through the returned port.
    ///
    /// # Errors
    ///
    /// Returns an error if the thread cannot be spawned, or if host
    /// construction fails inside it. Construction failures are reported through
    /// the same rendezvous that carries the port, so a failed build surfaces as
    /// an error here rather than as a thread that never answers.
    pub fn spawn(
        session_id: HostSessionId,
        world: WorldState,
        persistence: PersistenceAdmissionSession,
        journal: Box<dyn JournalPort + Send>,
        core_options: HostCoreOptions,
        channel_options: ChannelHostOptions,
    ) -> Result<Self> {
        Self::spawn_with_bootstrap(
            session_id,
            world,
            persistence,
            journal,
            core_options,
            channel_options,
            HostBootstrap::default(),
        )
    }

    /// Start one owner for bootstrap and the subsequent interactive lifetime.
    pub fn spawn_with_bootstrap(
        session_id: HostSessionId,
        world: WorldState,
        persistence: PersistenceAdmissionSession,
        journal: Box<dyn JournalPort + Send>,
        core_options: HostCoreOptions,
        channel_options: ChannelHostOptions,
        bootstrap: HostBootstrap,
    ) -> Result<Self> {
        // Rendezvous of exactly one message. The port cannot exist until the
        // host does, and the host cannot exist off this thread, so the caller
        // has to wait for the thread to hand it back. Sending the RESULT rather
        // than the port means a construction failure arrives as an error
        // instead of as a hang followed by a confusing join error.
        let (ready_tx, ready_rx) = sync_channel::<Result<ChannelHostPort, String>>(1);
        let handle = Builder::new()
            .name("scriptbots-host".to_owned())
            .spawn(move || {
                Self::own_and_drive(
                    session_id,
                    world,
                    persistence,
                    journal,
                    core_options,
                    channel_options,
                    bootstrap,
                    &ready_tx,
                )
            })
            .context("failed to spawn the scriptbots-host thread")?;

        match ready_rx.recv() {
            Ok(Ok(port)) => Ok(Self { port, handle }),
            Ok(Err(reason)) => match handle.join() {
                Ok(Err(error)) => Err(error.context(format!("host construction failed: {reason}"))),
                Ok(Ok(_)) => Err(anyhow!("host construction failed: {reason}")),
                Err(_) => Err(anyhow!(
                    "host construction failed: {reason}; owner panicked while exiting"
                )),
            },
            // The thread died before reporting either way. Join to recover the
            // real cause rather than reporting the closed channel, which would
            // describe the symptom and hide the panic.
            Err(_) => match handle.join() {
                Ok(Ok(_)) => Err(anyhow!(
                    "host thread exited before publishing its port, with no error"
                )),
                Ok(Err(error)) => Err(error.context("host thread failed before publishing a port")),
                Err(_) => Err(anyhow!("host thread panicked before publishing a port")),
            },
        }
    }

    /// The owner-thread body: construct, publish the port, then drive forever.
    #[allow(clippy::too_many_arguments)]
    fn own_and_drive(
        session_id: HostSessionId,
        world: WorldState,
        persistence: PersistenceAdmissionSession,
        journal: Box<dyn JournalPort + Send>,
        core_options: HostCoreOptions,
        channel_options: ChannelHostOptions,
        bootstrap: HostBootstrap,
        ready_tx: &SyncSender<Result<ChannelHostPort, String>>,
    ) -> Result<HostThreadReceipt> {
        let build = (|| -> Result<(ChannelHostDriver, ChannelHostPort)> {
            let requirement = journal.shutdown_commit_requirement();
            let mut core = HostCore::with_journal_and_persistence(
                session_id,
                world,
                core_options,
                journal,
                persistence,
            )
            .context("HostCore construction failed")?;
            core.install_schedule(bootstrap.schedule, crate::resolve_scheduled_config_patch)
                .context("invalid owner scenario schedule")?;
            let completed =
                drive_bootstrap(&mut core, bootstrap.ticks, requirement).and_then(|()| {
                    match bootstrap.on_completed {
                        Some(observe) => core.with_world(observe),
                        None => Ok(()),
                    }
                });
            if let Err(error) = completed {
                // Preserve the original failure; ordered shutdown attempts to retain the
                // completed persistence tail before the storage controller closes.
                let cleanup = drain_bootstrap_shutdown(&mut core);
                return match cleanup {
                    Ok(()) => Err(error),
                    Err(cleanup) => {
                        Err(error.context(format!("bootstrap shutdown also failed: {cleanup:#}")))
                    }
                };
            }
            ChannelHostDriver::new(FixedDeadlineHost::new(core), channel_options)
                .context("channel host driver rejected its options")
        })();

        let (mut driver, port) = match build {
            Ok(parts) => parts,
            Err(error) => {
                // Report before returning, or the caller blocks on a rendezvous
                // that will never be answered.
                let _ = ready_tx.send(Err(format!("{error:#}")));
                return Err(error);
            }
        };
        if ready_tx.send(Ok(port)).is_err() {
            return Err(anyhow!(
                "host built successfully but the caller stopped waiting for its port"
            ));
        }

        // A monotonic clock read per drive. ManualInstant is nanosecond-based and
        // the driver only ever compares and orders these, so an arbitrary epoch
        // is fine as long as it never goes backwards.
        let epoch = Instant::now();
        let run = driver
            .run(|| {
                ManualInstant::from_nanos(
                    u64::try_from(epoch.elapsed().as_nanos()).unwrap_or(u64::MAX),
                )
            })
            .context("host drive loop stopped")?;
        let core = driver.host().core();
        let snapshot = core.latest_snapshot();
        let receipt = HostThreadReceipt {
            run,
            snapshot,
            sense_saturations_total: core.world().sense_saturations_total(),
            required_persistence_tick: core.persistence().last_admitted_tick().map(|tick| tick.0),
            final_digest: core.scientific_digest_v1(),
        };
        if run.outcome == ChannelRunOutcome::Faulted {
            let fault = core
                .health()
                .fault()
                .cloned()
                .context("host drive loop reported a fault without a recorded cause")?;
            return Err(HostThreadFaultError { fault, receipt }.into());
        }
        Ok(receipt)
    }

    /// A cross-thread handle to the host.
    ///
    /// Cloneable on purpose: the control server, the frontend and any future
    /// transport each hold their own.
    #[must_use]
    pub fn port(&self) -> ChannelHostPort {
        self.port.clone()
    }

    /// Wait for the host to finish and recover its run receipt.
    ///
    /// Dropping every port is what tells the driver to stop, so a caller that
    /// joins while still holding one will wait forever. That is the caller's
    /// ordering to get right and is why this consumes `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if the host thread panicked or its drive loop failed.
    pub fn join(self) -> Result<HostThreadReceipt> {
        drop(self.port);
        match self.handle.join() {
            Ok(result) => result,
            Err(_) => Err(anyhow!("host thread panicked")),
        }
    }
}

fn journal_committed(state: &JournalState, requirement: ShutdownCommitRequirement) -> bool {
    matches!(state, JournalState::Durable)
        || (requirement == ShutdownCommitRequirement::CommittedVolatile
            && matches!(state, JournalState::CommittedVolatile))
}

const BOOTSTRAP_STALL_LIMIT: Duration = Duration::from_secs(30);

/// Bounds an unchanged observation, not synchronous driver execution. A storage
/// call already running on the owner cannot be cancelled by this waiter. Observe
/// its result after it returns; only a changed status starts another wait window.
struct ProgressDeadline<T> {
    observed: T,
    deadline: Instant,
}

impl<T: PartialEq> ProgressDeadline<T> {
    fn new(now: Instant, observed: T) -> Self {
        Self {
            observed,
            deadline: now + BOOTSTRAP_STALL_LIMIT,
        }
    }

    fn observe(&mut self, now: Instant, observed: T) {
        if self.observed != observed {
            self.observed = observed;
            self.deadline = now + BOOTSTRAP_STALL_LIMIT;
        }
    }

    fn expired(&self, now: Instant) -> bool {
        now >= self.deadline
    }
}

fn drive_bootstrap(
    core: &mut HostCore,
    ticks: u64,
    requirement: ShutdownCommitRequirement,
) -> Result<()> {
    let mut port = core.local_port();
    for sequence in 1..=ticks {
        let envelope = CommandEnvelope::new(
            CommandId::from_client_sequence(u64::MAX - 1, sequence),
            HostCommand::Step,
        );
        let authority_deadline = Instant::now() + BOOTSTRAP_STALL_LIMIT;
        let admitted = loop {
            match port.submit(envelope.clone()) {
                Ok(status) => break status,
                Err(scriptbots_runtime::HostAccessError::CommandAuthorityLookup {
                    failure:
                        scriptbots_runtime::CommandAuthorityLookupFailure::Pending
                        | scriptbots_runtime::CommandAuthorityLookupFailure::Busy
                        | scriptbots_runtime::CommandAuthorityLookupFailure::Capacity { .. },
                    ..
                }) => {}
                Err(error) => return Err(error.into()),
            }
            core.drive(ManualInstant::from_nanos(0))?;
            if core.pending_journal_batch().is_some() {
                core.retry_retained_journal()?;
            }
            if Instant::now() >= authority_deadline {
                return Err(anyhow!(
                    "bootstrap command authority timed out at step {sequence}"
                ));
            }
            std::thread::park_timeout(Duration::from_millis(1));
        };
        // Authority admission and application/journal progress have distinct
        // budgets. In particular, waiting for authority consumes no Step budget.
        let mut progress = ProgressDeadline::new(
            Instant::now(),
            (admitted, core.latest_snapshot().scheduled_patches.clone()),
        );
        loop {
            core.drive(ManualInstant::from_nanos(0))?;
            if core.pending_journal_batch().is_some() {
                core.retry_retained_journal()?;
            }
            let status = port
                .command_status(envelope.command_id)?
                .context("bootstrap Step lost its command identity")?;
            if let ApplicationState::Failed(error) = status.application() {
                return Err(anyhow!(
                    "bootstrap Step {sequence} failed: {}: {}",
                    error.code,
                    error.message
                ));
            }
            if let ApplicationState::Rejected(reason) = status.application() {
                return Err(anyhow!("bootstrap Step {sequence} rejected: {reason:?}"));
            }
            if let Some(fault) = core.health().fault() {
                return Err(anyhow!("bootstrap stopped at step {sequence}: {fault:?}"));
            }
            if matches!(status.application(), ApplicationState::Applied(_))
                && journal_committed(status.journal(), requirement)
            {
                tracing::debug!(sequence, tick = core.latest_snapshot().world.tick,
                    journal = ?status.journal(), "Owner bootstrap Step applied and journal committed");
                break;
            }
            let now = Instant::now();
            progress.observe(
                now,
                (status, core.latest_snapshot().scheduled_patches.clone()),
            );
            if progress.expired(now) {
                return Err(anyhow!(
                    "bootstrap Step {sequence} made no application or journal progress for {BOOTSTRAP_STALL_LIMIT:?}"
                ));
            }
            std::thread::park_timeout(Duration::from_millis(1));
        }
    }
    Ok(())
}

fn drain_bootstrap_shutdown(core: &mut HostCore) -> Result<()> {
    let initial = core.latest_snapshot();
    let mut progress = ProgressDeadline::new(
        Instant::now(),
        (None, initial.lifecycle, initial.scheduled_patches.clone()),
    );
    let mut requested = false;
    let mut authority_wait = None;
    loop {
        if !requested {
            match core.request_shutdown() {
                Ok(_) => {
                    requested = true;
                    authority_wait = None;
                }
                Err(
                    error @ scriptbots_runtime::HostAccessError::CommandAuthorityLookup {
                        failure:
                            scriptbots_runtime::CommandAuthorityLookupFailure::Pending
                            | scriptbots_runtime::CommandAuthorityLookupFailure::Busy
                            | scriptbots_runtime::CommandAuthorityLookupFailure::Capacity { .. },
                        ..
                    },
                ) => authority_wait = Some(error),
                Err(error) => return Err(error.into()),
            }
        }
        // File-backed authority lookups complete asynchronously. Drive retained
        // failure evidence while waiting, then retry the same shutdown identity.
        core.drive(ManualInstant::from_nanos(0))?;
        if core.latest_snapshot().lifecycle == scriptbots_runtime::HostLifecycle::Stopped {
            return Ok(());
        }
        if core.pending_journal_batch().is_some() {
            core.retry_retained_journal()?;
        }
        let status = if requested {
            let id = core
                .shutdown_command_id()
                .context("requested shutdown lost its identity")?;
            Some(
                core.local_port()
                    .command_status(id)?
                    .context("requested shutdown lost its status")?,
            )
        } else {
            None
        };
        let snapshot = core.latest_snapshot();
        let now = Instant::now();
        // None -> Some marks authority admission; later lifecycle or journal
        // changes measure actual cleanup progress, never the number of polls.
        progress.observe(
            now,
            (
                status,
                snapshot.lifecycle,
                snapshot.scheduled_patches.clone(),
            ),
        );
        if progress.expired(now) {
            if let Some(error) = authority_wait {
                return Err(anyhow::Error::new(error)
                    .context("bootstrap shutdown authority made no progress before its deadline"));
            }
            return Err(anyhow!(
                "bootstrap shutdown made no lifecycle or journal progress for {BOOTSTRAP_STALL_LIMIT:?}"
            ));
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{NullPersistence, ScriptBotsConfig};
    use scriptbots_runtime::{
        ApplicationState, CommandEnvelope, CommandId, HostCommand, HostPort, JournalState,
        PlaybackSnapshot, VolatileJournal,
    };

    fn world_and_session() -> (WorldState, PersistenceAdmissionSession) {
        WorldState::with_persistence(
            ScriptBotsConfig {
                rng_seed: Some(0x5eed_cafe),
                ..ScriptBotsConfig::default()
            },
            Box::new(NullPersistence),
        )
        .expect("deterministic test world")
    }

    #[test]
    fn bootstrap_progress_deadline_restarts_after_observed_application_not_elapsed_driver_work() {
        let start = Instant::now();
        let admitted = ApplicationState::Admitted;
        let mut progress = ProgressDeadline::new(start, admitted);
        let returned = start + Duration::from_secs(52);
        assert!(
            progress.expired(returned),
            "the old whole-Step deadline has elapsed"
        );
        progress.observe(
            returned,
            ApplicationState::Applied(scriptbots_runtime::AppliedCommand {
                tick: scriptbots_core::Tick(1),
                revisions: scriptbots_runtime::HostRevisions::default(),
            }),
        );
        assert!(
            !progress.expired(returned),
            "completed science gives journal polling its wait window"
        );
        assert!(!progress.expired(returned + BOOTSTRAP_STALL_LIMIT - Duration::from_nanos(1)));
        assert!(progress.expired(returned + BOOTSTRAP_STALL_LIMIT));
    }

    #[test]
    fn bootstrap_progress_deadline_does_not_restart_for_unchanged_polls() {
        let start = Instant::now();
        let mut progress = ProgressDeadline::new(start, JournalState::Pending);
        for second in [1, 15, 29, 30, 52] {
            let now = start + Duration::from_secs(second);
            progress.observe(now, JournalState::Pending);
            assert_eq!(
                progress.expired(now),
                second >= BOOTSTRAP_STALL_LIMIT.as_secs()
            );
        }
    }

    #[test]
    fn bootstrap_progress_deadline_tracks_scheduled_identity_and_commitment() {
        let start = Instant::now();
        let mut progress = ProgressDeadline::new(start, vec![(1_u64, JournalState::Pending)]);
        let next_patch = start + Duration::from_secs(29);
        progress.observe(
            next_patch,
            vec![(1, JournalState::Pending), (2, JournalState::Pending)],
        );
        assert!(!progress.expired(start + BOOTSTRAP_STALL_LIMIT));
        let commitment = next_patch + Duration::from_secs(29);
        let committed = vec![(1, JournalState::Durable), (2, JournalState::Pending)];
        progress.observe(commitment, committed.clone());
        assert!(!progress.expired(next_patch + BOOTSTRAP_STALL_LIMIT));
        progress.observe(commitment + Duration::from_secs(29), committed);
        assert!(
            progress.expired(commitment + BOOTSTRAP_STALL_LIMIT),
            "unchanged scheduled status must still time out"
        );
    }

    #[test]
    fn bootstrap_application_wait_starts_after_authority_admission() {
        let start = Instant::now();
        let authority = ProgressDeadline::new(start, ());
        let admitted_at = start + Duration::from_secs(29);
        assert!(!authority.expired(admitted_at));
        let application = ProgressDeadline::new(admitted_at, ApplicationState::Admitted);
        assert!(authority.expired(start + BOOTSTRAP_STALL_LIMIT));
        assert!(!application.expired(start + BOOTSTRAP_STALL_LIMIT));
        assert!(application.expired(admitted_at + BOOTSTRAP_STALL_LIMIT));
    }

    #[test]
    fn bootstrap_shutdown_wait_tracks_admission_and_lifecycle_without_poll_resets() {
        use scriptbots_runtime::HostLifecycle;

        let start = Instant::now();
        let mut progress = ProgressDeadline::new(start, (None, HostLifecycle::Running));
        let admission = start + Duration::from_secs(29);
        progress.observe(
            admission,
            (Some(JournalState::Pending), HostLifecycle::Running),
        );
        assert!(!progress.expired(start + BOOTSTRAP_STALL_LIMIT));
        let returned = admission + Duration::from_secs(52);
        assert!(progress.expired(returned));
        let stopping = (Some(JournalState::Pending), HostLifecycle::Stopping);
        progress.observe(returned, stopping.clone());
        assert!(!progress.expired(returned));
        progress.observe(returned + Duration::from_secs(29), stopping);
        assert!(progress.expired(returned + BOOTSTRAP_STALL_LIMIT));
    }

    #[test]
    fn bootstrap_and_interactive_steps_share_the_owner_schedule() {
        let (mut world, persistence) = world_and_session();
        let mut config = world.config().clone();
        config.food_respawn_amount = 0.1;
        world
            .apply_config_update(config)
            .expect("respawn fits every scheduled food capacity");
        let host = HostThread::spawn_with_bootstrap(
            HostSessionId::new(81),
            world,
            persistence,
            Box::new(VolatileJournal::default()),
            HostCoreOptions {
                initial_playback: PlaybackSnapshot {
                    paused: true,
                    speed_multiplier: 1.0,
                },
                ..HostCoreOptions::default()
            },
            ChannelHostOptions::default(),
            HostBootstrap {
                ticks: 2,
                schedule: vec![
                    ScheduledConfigPatch::new(
                        1,
                        scriptbots_core::Tick(0),
                        serde_json::json!({"food_max": 0.31}),
                    )
                    .unwrap(),
                    ScheduledConfigPatch::new(
                        2,
                        scriptbots_core::Tick(1),
                        serde_json::json!({"food_max": 0.42}),
                    )
                    .unwrap(),
                    ScheduledConfigPatch::new(
                        3,
                        scriptbots_core::Tick(2),
                        serde_json::json!({"food_max": 0.53}),
                    )
                    .unwrap(),
                ],
                on_completed: Some(Box::new(|world| {
                    assert_eq!(world.tick().0, 2);
                    assert!((world.config().food_max - 0.42).abs() < f32::EPSILON);
                    Ok(())
                })),
            },
        )
        .expect("bootstrap completes before publishing the client port");
        let mut port = host.port();
        let initial = port.snapshot_after(None).unwrap().unwrap();
        assert_eq!(initial.world.tick, 2);
        assert_eq!(initial.scheduled_patches.len(), 2);
        assert!(
            initial
                .scheduled_patches
                .iter()
                .all(|status| matches!(status.journal, JournalState::CommittedVolatile))
        );
        let id = CommandId::new(82);
        port.submit(CommandEnvelope::new(id, HostCommand::Step))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let status = port.command_status(id).unwrap().unwrap();
            if matches!(status.journal(), JournalState::CommittedVolatile) {
                assert!(
                    matches!(status.application(), ApplicationState::Applied(boundary) if boundary.tick.0 == 3)
                );
                break;
            }
            assert!(
                Instant::now() < deadline,
                "interactive step stalled: {status:?}"
            );
            std::thread::park_timeout(Duration::from_millis(1));
        }
        drop(port);
        let receipt = host.join().expect("same owner shuts down");
        assert_eq!(receipt.snapshot.world.tick, 3);
        assert_eq!(receipt.snapshot.scheduled_patches.len(), 3);
        assert!((receipt.snapshot.config.food_max - 0.53).abs() < f32::EPSILON);
        for (index, status) in receipt.snapshot.scheduled_patches.iter().enumerate() {
            assert_eq!(
                status.evidence.scheduled.sequence,
                u64::try_from(index).unwrap() + 1
            );
            assert_eq!(
                status.evidence.boundary.tick.0,
                u64::try_from(index).unwrap()
            );
        }
    }

    #[test]
    fn failed_bootstrap_patch_does_not_publish_a_client_port() {
        let (world, persistence) = world_and_session();
        let result = HostThread::spawn_with_bootstrap(
            HostSessionId::new(83),
            world,
            persistence,
            Box::new(VolatileJournal::default()),
            HostCoreOptions::default(),
            ChannelHostOptions::default(),
            HostBootstrap {
                ticks: 1,
                schedule: vec![
                    ScheduledConfigPatch::new(
                        1,
                        scriptbots_core::Tick(0),
                        serde_json::json!({"food_max": "invalid"}),
                    )
                    .unwrap(),
                ],
                on_completed: Some(Box::new(|_| {
                    panic!("failed bootstrap must not invoke completion observer")
                })),
            },
        );
        let Err(error) = result else {
            panic!("invalid patch published a port")
        };
        assert!(
            format!("{error:#}").contains("scheduled_config_resolution"),
            "{error:#}"
        );
    }

    #[test]
    fn failed_file_bootstrap_drains_durable_failure_and_shutdown_without_science() {
        use scriptbots_storage::{PersistenceGuarantee, StoragePipeline, StorageReader};

        let directory = tempfile::tempdir()
            .expect("bootstrap failure artifacts")
            .keep();
        let path = directory.join("failed-bootstrap.sqlite");
        let path = path.to_str().expect("UTF-8 database path");
        eprintln!("bootstrap failure database retained at {path}");
        let mut pipeline =
            StoragePipeline::create_unattributed_file(path).expect("real file worker");
        let run_id = pipeline.run_id();
        let session = HostSessionId::new(84);
        let (world, persistence) = WorldState::with_persistence(
            ScriptBotsConfig {
                world_width: 64,
                world_height: 64,
                food_cell_size: 16,
                rng_seed: Some(84),
                persistence_interval: 1,
                ..ScriptBotsConfig::default()
            },
            Box::new(pipeline.sink()),
        )
        .expect("world with real persistence session");
        let journal = pipeline
            .journal_port(session, Default::default())
            .expect("file journal authority");
        let result = HostThread::spawn_with_bootstrap(
            session,
            world,
            persistence,
            Box::new(journal),
            HostCoreOptions {
                initial_playback: PlaybackSnapshot {
                    paused: true,
                    speed_multiplier: 1.0,
                },
                ..HostCoreOptions::default()
            },
            ChannelHostOptions::default(),
            HostBootstrap {
                ticks: 1,
                schedule: vec![
                    ScheduledConfigPatch::new(
                        1,
                        scriptbots_core::Tick(0),
                        serde_json::json!({"food_max": "invalid"}),
                    )
                    .expect("structurally valid patch"),
                ],
                on_completed: Some(Box::new(|_| {
                    panic!("failed bootstrap published completion")
                })),
            },
        );
        let Err(error) = result else {
            panic!("invalid bootstrap published a client port")
        };
        let diagnostic = format!("{error:#}");
        assert!(
            diagnostic.contains("scheduled_config_resolution"),
            "original failure lost: {diagnostic}"
        );
        assert!(
            !diagnostic.contains("bootstrap shutdown also failed"),
            "cleanup failed: {diagnostic}"
        );
        let shutdown = pipeline.shutdown().expect("file worker shutdown receipt");
        assert_eq!(shutdown.guarantee, PersistenceGuarantee::Durable);
        drop(pipeline);
        let reader =
            StorageReader::open_finished_for_run(path, run_id).expect("finished file readback");
        assert_failed_bootstrap_journal(&reader, session);
        reader.close().expect("close finished reader");
    }

    fn assert_failed_bootstrap_journal(
        reader: &scriptbots_storage::StorageReader,
        session: HostSessionId,
    ) {
        use scriptbots_runtime::ScheduledPatchOutcome;
        use scriptbots_storage::{HostJournalRecordState, StorageIntegrityCheckResult};

        assert_eq!(reader.max_tick().expect("scientific tick rows"), None);
        let page = reader
            .host_journal_session_conformance_page(session, None, 16, 1024 * 1024)
            .expect("canonical durable journal readback");
        assert_eq!(page.integrity_check, StorageIntegrityCheckResult::Ok);
        assert_eq!(page.next_after, None);
        assert_eq!(
            page.records.len(),
            3,
            "scheduled failure, terminal Step, and shutdown"
        );
        assert_eq!(page.progress.journal.durable, 3);
        assert_eq!(page.progress.events.durable, 0);
        for record in &page.records {
            assert_eq!(record.state, HostJournalRecordState::Durable);
            assert_eq!(record.applied.tick.0, 0);
            assert!(
                record.event.is_none(),
                "failed bootstrap must not record science"
            );
        }
        let [scheduled, step, shutdown] = page.records.as_slice() else {
            panic!("wrong journal shape")
        };
        let evidence = scheduled
            .scheduled_patch
            .as_ref()
            .expect("actual scheduled failure evidence");
        assert_eq!(evidence.scheduled.sequence, 1);
        assert_eq!(evidence.scheduled.due_tick.0, 0);
        assert!(
            matches!(&evidence.outcome, ScheduledPatchOutcome::Failed(failure)
            if failure.code == "scheduled_config_resolution")
        );
        let step = step
            .command_lifecycle
            .as_ref()
            .expect("terminal bootstrap Step lifecycle");
        assert_eq!(
            step.envelope().command_id,
            CommandId::from_client_sequence(u64::MAX - 1, 1)
        );
        assert!(matches!(step.envelope().command, HostCommand::Step));
        assert!(matches!(
            step.terminal()
                .expect("terminal Step observation")
                .application(),
            ApplicationState::Failed(_)
        ));
        assert!(
            shutdown
                .command_lifecycle
                .as_ref()
                .expect("shutdown lifecycle")
                .is_applied_shutdown()
        );
        assert_eq!(page.progress.shutdown, Some(shutdown.batch_id));
    }

    /// The host thread owns the world and hands back a usable port.
    ///
    /// This is the whole point of the module: `HostCore` is `!Send`, so the
    /// only proof that the ownership transfer works is that a world moved into
    /// the thread comes back as a cross-thread port rather than as a hang or a
    /// join error.
    #[test]
    fn a_world_moved_into_the_host_thread_yields_a_working_port() {
        let (world, persistence) = world_and_session();
        let host = HostThread::spawn(
            HostSessionId::new(1),
            world,
            persistence,
            Box::new(VolatileJournal::default()),
            HostCoreOptions {
                initial_playback: PlaybackSnapshot {
                    paused: true,
                    speed_multiplier: 1.0,
                },
                ..HostCoreOptions::default()
            },
            ChannelHostOptions::default(),
        )
        .expect("host thread starts and publishes its port");

        let mut port = host.port();
        assert_eq!(
            port.session_id(),
            HostSessionId::new(1),
            "the port must speak for the host that built it"
        );

        let initial = port
            .snapshot_after(None)
            .expect("snapshot access")
            .expect("initial snapshot");
        let initial_digest = port.scientific_digest_v1().expect("initial digest");
        let command_id = CommandId::new(17);
        port.submit(CommandEnvelope::new(command_id, HostCommand::Step))
            .expect("real step admission");
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        let applied = loop {
            let status = port
                .command_status(command_id)
                .expect("status access")
                .expect("known step");
            if matches!(status.journal(), JournalState::CommittedVolatile) {
                let ApplicationState::Applied(applied) = status.application() else {
                    panic!("step must actually apply: {status:?}");
                };
                break *applied;
            }
            assert!(
                Instant::now() < deadline,
                "step did not complete: {status:?}"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert_eq!(applied.tick.0, initial.world.tick + 1);
        let applied_digest = port
            .scientific_digest_v1()
            .expect("applied boundary digest");
        assert_ne!(applied_digest.overall, initial_digest.overall);

        // The join receipt must retain the completed science observation after
        // the final client disconnects and the owner acknowledges shutdown.
        drop(port);
        let receipt = host.join().expect("host thread stops cleanly");
        assert_eq!(
            receipt.run.outcome,
            ChannelRunOutcome::ControllerDisconnected
        );
        assert_eq!(receipt.snapshot.world.tick, applied.tick.0);
        assert_eq!(
            receipt.final_digest.expect("final owner digest"),
            applied_digest,
            "shutdown must return the exact last scientific boundary, not startup state"
        );
        assert_eq!(
            receipt.snapshot.revisions.scientific,
            applied.revisions.scientific
        );
        assert_eq!(
            receipt.snapshot.lifecycle,
            scriptbots_runtime::HostLifecycle::Stopped
        );
        assert!(receipt.run.drives > 0);
    }

    #[test]
    fn construction_rejects_a_session_bound_to_a_different_world() {
        let (world, _own_session) = world_and_session();
        let (_other_world, foreign_session) = world_and_session();
        let result = HostThread::spawn(
            HostSessionId::new(3),
            world,
            foreign_session,
            Box::new(VolatileJournal::default()),
            HostCoreOptions::default(),
            ChannelHostOptions::default(),
        );
        let Err(error) = result else {
            panic!("a foreign persistence session must not publish a usable host");
        };
        assert!(
            error.to_string().contains("host construction failed"),
            "{error:#}"
        );
        assert!(error.to_string().contains("different world"), "{error:#}");
        assert!(matches!(
            error.downcast_ref::<scriptbots_runtime::HostCoreBuildError>(),
            Some(scriptbots_runtime::HostCoreBuildError::Persistence(
                scriptbots_core::PersistenceSessionError::WrongWorld
            ))
        ));
    }

    /// A host that cannot be built reports an error rather than hanging.
    ///
    /// The rendezvous carries a Result precisely so a construction failure
    /// surfaces here. Without it the caller would block forever on a port that
    /// is never sent, and the real cause would only appear later as a confusing
    /// join error - a failure mode that looks like a deadlock.
    #[test]
    fn a_host_that_cannot_be_built_reports_instead_of_hanging() {
        let (world, persistence) = world_and_session();
        let refused = ChannelHostOptions {
            ingress_capacity: 0,
            ..ChannelHostOptions::default()
        };
        let outcome = HostThread::spawn(
            HostSessionId::new(2),
            world,
            persistence,
            Box::new(VolatileJournal::default()),
            HostCoreOptions::default(),
            refused,
        );
        // Matched rather than `expect_err`: HostThread owns a JoinHandle and is
        // deliberately not Debug, so the failure has to be destructured.
        let Err(error) = outcome else {
            panic!("zero ingress capacity must be rejected");
        };
        assert!(
            error.to_string().contains("host construction failed"),
            "the failure must name construction, got: {error}"
        );
        assert!(matches!(
            error.downcast_ref::<scriptbots_runtime::channel::ChannelHostOptionsError>(),
            Some(scriptbots_runtime::channel::ChannelHostOptionsError::EmptyIngress)
        ));
    }
}
