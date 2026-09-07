//! Region-owned background services and structured shutdown (bd-2z0.4.13).
//!
//! [`AppRoot`] closes registered service finalizers in reverse
//! dependency order with an explicit [`Budget`], records a per-region
//! [`Outcome`](asupersync::types::Outcome) (`Ok`/`Err`/`Cancelled`/`Panicked`), and
//! logs every outcome at the exit boundary. Finalizers run synchronously and must
//! enforce their own budgets: this root cannot interrupt a wedged finalizer.
//! Production currently registers the control-server region; complete host/storage
//! region ownership remains part of bd-pcfj.
//!
//! Semantic contract types come from the asupersync ecosystem
//! (`asupersync::types::{Budget, Outcome, CancelReason}`); the runtime's own scopes
//! remain the longer-term owner of async service topology (bd-2z0.4.12's bus and the
//! native HostCore runner already live on that spine).

use asupersync::types::{Budget, Outcome};
use std::time::Instant;
use tracing::{error, info, warn};

/// Finalizer contract for one owned background service: performs the drain-and-join
/// work on the orderly teardown path, honoring the region's budget.
type RegionFinalizer = Box<dyn FnOnce(&Budget) -> Outcome<String, String> + Send>;

/// One background service owned by the application root.
///
/// The finalizer performs the service's drain-and-join contract on the orderly
/// teardown path. It receives the region's budget; services with internal wait loops
/// must honor the budget's deadline and return `Outcome::Cancelled` on exhaustion
/// rather than overrunning.
pub struct ServiceRegion {
    name: &'static str,
    budget: Budget,
    finalizer: RegionFinalizer,
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
/// the finalizer reported budget exhaustion.
#[derive(Debug)]
pub struct RegionOutcome {
    pub name: &'static str,
    pub outcome: Outcome<String, String>,
    pub elapsed: std::time::Duration,
    /// The cancellation reason names deadline, poll-quota, or cost-budget
    /// exhaustion. This is reported by the finalizer, not measured by the root.
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
    /// register downstream services first (storage), then their producers (host,
    /// control). Producers therefore stop before the services that drain them.
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
            let budget_exhausted = matches!(
                &outcome,
                Outcome::Cancelled(reason) if reason.is_budget_exceeded()
            );
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
                    budget_exhausted,
                    "region finalizer reported cancellation"
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

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::types::{CancelKind, CancelReason};

    #[test]
    fn cancellation_reports_distinguish_budget_exhaustion_from_other_reasons() {
        for (kind, expected_exhaustion) in [
            (CancelKind::User, false),
            (CancelKind::Timeout, false),
            (CancelKind::Shutdown, false),
            (CancelKind::Deadline, true),
            (CancelKind::PollQuota, true),
            (CancelKind::CostBudget, true),
        ] {
            let mut root = AppRoot::new();
            root.register(ServiceRegion::new("service", Budget::new(), move |_| {
                Outcome::Cancelled(CancelReason::new(kind))
            }));
            let outcomes = root.close();
            assert_eq!(outcomes.len(), 1);
            assert_eq!(
                outcomes[0].budget_exhausted, expected_exhaustion,
                "{kind:?}"
            );
            let Outcome::Cancelled(reason) = &outcomes[0].outcome else {
                panic!("the actual cancellation must remain in the outcome");
            };
            assert_eq!(reason.kind, kind);
        }
    }

    #[test]
    fn panicking_producer_does_not_skip_its_downstream_finalizer() {
        let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut root = AppRoot::new();
        let storage_observed = std::sync::Arc::clone(&observed);
        root.register(ServiceRegion::new("storage", Budget::new(), move |_| {
            storage_observed.lock().expect("trace").push("storage");
            Outcome::ok("drained".to_owned())
        }));
        let control_observed = std::sync::Arc::clone(&observed);
        root.register(ServiceRegion::new("control", Budget::new(), move |_| {
            control_observed.lock().expect("trace").push("control");
            panic!("finalizer failure");
        }));
        let outcomes = root.close();
        assert_eq!(*observed.lock().expect("trace"), ["control", "storage"]);
        assert_eq!(outcomes.len(), 2);
        assert!(matches!(outcomes[0].outcome, Outcome::Panicked(_)));
        assert!(matches!(outcomes[1].outcome, Outcome::Ok(_)));
        assert!(outcomes.iter().all(|outcome| !outcome.budget_exhausted));
    }
}
