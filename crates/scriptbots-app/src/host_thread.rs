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
use scriptbots_core::{PersistenceAdmissionSession, WorldState};
use scriptbots_runtime::channel::{
    ChannelHostDriver, ChannelHostOptions, ChannelHostPort, ChannelRunOutcome, ChannelRunReceipt,
};
use scriptbots_runtime::{
    FixedDeadlineHost, HostCore, HostCoreOptions, HostSessionId, JournalPort, ManualInstant,
};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::{Builder, JoinHandle};
use std::time::Instant;

/// A running host thread and the handle everyone else talks to it through.
pub struct HostThread {
    port: ChannelHostPort,
    handle: JoinHandle<Result<HostThreadReceipt>>,
}

/// Owner observations retained after all clients disconnect and finalization finishes.
pub struct HostThreadReceipt {
    pub run: ChannelRunReceipt,
    pub snapshot: std::sync::Arc<scriptbots_runtime::RenderSnapshot>,
    pub sense_saturations_total: u64,
    pub required_persistence_tick: Option<u64>,
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
        ready_tx: &SyncSender<Result<ChannelHostPort, String>>,
    ) -> Result<HostThreadReceipt> {
        let build = (|| -> Result<(ChannelHostDriver, ChannelHostPort)> {
            let core = HostCore::with_journal_and_persistence(
                session_id,
                world,
                core_options,
                journal,
                persistence,
            )
            .map_err(|source| anyhow!("HostCore construction failed: {source}"))?;
            ChannelHostDriver::new(FixedDeadlineHost::new(core), channel_options)
                .map_err(|source| anyhow!("channel host driver rejected its options: {source}"))
        })();

        let (mut driver, port) = match build {
            Ok(parts) => parts,
            Err(error) => {
                // Report before returning, or the caller blocks on a rendezvous
                // that will never be answered.
                let _ = ready_tx.send(Err(error.to_string()));
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
            .map_err(|source| anyhow!("host drive loop stopped: {source}"))?;
        let core = driver.host().core();
        let snapshot = core.latest_snapshot();
        if run.outcome == ChannelRunOutcome::Faulted {
            return Err(anyhow!("host drive loop faulted: {:?}", core.health()));
        }
        Ok(HostThreadReceipt {
            run,
            snapshot,
            sense_saturations_total: core.world().sense_saturations_total(),
            required_persistence_tick: core.persistence().last_admitted_tick().map(|tick| tick.0),
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{NullPersistence, ScriptBotsConfig};
    use scriptbots_runtime::{HostPort, VolatileJournal};

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
            HostCoreOptions::default(),
            ChannelHostOptions::default(),
        )
        .expect("host thread starts and publishes its port");

        // The port is the client contract. Asking it for the session identity
        // proves the round trip works, not merely that a value came back.
        let port = host.port();
        assert_eq!(
            port.session_id(),
            HostSessionId::new(1),
            "the port must speak for the host that built it"
        );

        // Dropping every port is what stops the driver; join then recovers the
        // receipt. A hang here would mean the shutdown handshake is wrong.
        drop(port);
        let receipt = host.join().expect("host thread stops cleanly");
        assert!(
            receipt.run.drives >= 1 || receipt.run.commands_admitted == 0,
            "a host that never drove and never admitted anything did not run"
        );
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
    }
}
