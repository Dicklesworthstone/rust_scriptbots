//! Region-owned background services and structured shutdown (bd-2z0.4.13).
//!
//! The app entrypoints (GPUI, terminal, Bevy, headless) own an [`AppRoot`] whose child
//! regions wrap each background service: the control-server bridge and the storage
//! pipeline bridge today. Closing the root runs each region's finalizer in reverse
//! dependency order with an explicit [`Budget`], records a per-region
//! [`Outcome`](asupersync::types::Outcome) (`Ok`/`Err`/`Cancelled`/`Panicked`), and
//! logs every outcome at the exit boundary. A wedged finalizer exhausts its budget as
//! a typed `Cancelled` outcome instead of hanging the process forever, and every
//! finalizer runs on the orderly path so `Drop` remains a last-resort guard.
//!
//! Semantic contract types come from the asupersync ecosystem
//! (`asupersync::types::{Budget, Outcome, CancelReason}`); the runtime's own scopes
//! remain the longer-term owner of async service topology (bd-2z0.4.12's bus and the
//! native HostCore runner already live on that spine).

use asupersync::types::{Budget, Outcome};
use std::time::Instant;
use tracing::{error, info, warn};

/// One background service owned by the application root.
///
/// The finalizer performs the service's drain-and-join contract on the orderly
/// teardown path. It receives the region's budget; services with internal wait loops
/// must honor the budget's deadline and return `Outcome::Cancelled` on exhaustion
/// rather than overrunning.
pub struct ServiceRegion {
    name: &'static str,
    budget: Budget,
    finalizer: Box<dyn FnOnce(&Budget) -> Outcome<String, String> + Send>,
}

impl ServiceRegion {
    /// Register a service with a name, an explicit teardown budget, and its finalizer.
    #[must_use]
    pub fn new(
        name: &'static str,
        budget: Budget,
        finalizer: impl FnOnce(&Budget) -> Outcome<String, String> + Send + 'static,
    ) -> Self {
        Self {
            name,
            budget,
            finalizer: Box::new(finalizer),
        }
    }
}

/// The recorded result of one region's teardown: its outcome, wall time, and whether
/// the teardown budget was exhausted.
#[derive(Debug)]
pub struct RegionOutcome {
    pub name: &'static str,
    pub outcome: Outcome<String, String>,
    pub elapsed: std::time::Duration,
    pub budget_exhausted: bool,
}

/// The application root: owns every background service and drives ordered, budgeted
/// teardown with per-region outcome logging.
pub struct AppRoot {
    regions: Vec<ServiceRegion>,
}

impl AppRoot {
    #[must_use]
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// Register a child region. Regions close in REVERSE registration order, so
    /// register producers before the services that drain them (storage goes last).
    pub fn register(&mut self, region: ServiceRegion) {
        self.regions.push(region);
    }

    /// Ordered, budgeted teardown of every registered region.
    ///
    /// Children close in reverse registration order. Each finalizer runs on the
    /// calling thread (orderly path), is wrapped in panic isolation so one panicking
    /// service cannot skip its siblings' teardown, and is logged with its outcome,
    /// elapsed time, and budget state.
    pub fn close(self) -> Vec<RegionOutcome> {
        let mut outcomes = Vec::with_capacity(self.regions.len());
        for region in self.regions.into_iter().rev() {
            let name = region.name;
            let started = Instant::now();
            let outcome = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (region.finalizer)(&region.budget)
            })) {
                Ok(outcome) => outcome,
                Err(payload) => {
                    let detail = payload
                        .downcast_ref::<&str>()
                        .map(ToString::to_string)
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic".to_owned());
                    Outcome::Panicked(asupersync::types::PanicPayload::new(detail))
                }
            };
            let elapsed = started.elapsed();
            let budget_exhausted = matches!(outcome, Outcome::Cancelled(_));
            match &outcome {
                Outcome::Ok(detail) => info!(
                    region = name,
                    elapsed_ms = elapsed.as_millis() as u64,
                    %detail,
                    "region closed"
                ),
                Outcome::Err(error) => error!(
                    region = name,
                    elapsed_ms = elapsed.as_millis() as u64,
                    %error,
                    "region closed with an error"
                ),
                Outcome::Cancelled(reason) => warn!(
                    region = name,
                    elapsed_ms = elapsed.as_millis() as u64,
                    ?reason,
                    "region teardown exhausted its budget"
                ),
                Outcome::Panicked(payload) => error!(
                    region = name,
                    elapsed_ms = elapsed.as_millis() as u64,
                    %payload,
                    "region finalizer panicked"
                ),
            }
            outcomes.push(RegionOutcome {
                name,
                outcome,
                elapsed,
                budget_exhausted,
            });
        }
        outcomes
    }
}

impl Default for AppRoot {
    fn default() -> Self {
        Self::new()
    }
}
