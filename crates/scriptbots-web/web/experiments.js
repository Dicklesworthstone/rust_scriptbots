// A browser experiment is a separate matched-seed cohort on the connected host.
// This controller owns requests and observations, never a world or a scientific clock.
import { exactIdentity } from "./inspector.js";
import { interventionRequestId } from "./interventions.js";
import { prepareExperimentRequest, validateExperimentCatalog, validateExperimentStatus,
  validateExperimentComparison, experimentCanResume } from "./experiment-protocol.js";

export const EXPERIMENT_PAGE_SIZE = 12;
export const EXPERIMENT_RUN_PAGE_SIZE = 16;
const POLL_MS = 2000;
const ACTIVE = new Set(["pending", "running", "cancelling"]);
const errorText = (error) => String(error?.message || error || "Unknown experiment failure").slice(0, 1024);

function freeze(value) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}
const copy = (value) => freeze(structuredClone(value));

function completedVariants(status) {
  const variants = new Map();
  for (const run of status?.runs ?? []) {
    variants.set(run.variant_id, (variants.get(run.variant_id) ?? true) && run.state === "completed");
  }
  return new Set([...variants].filter(([, complete]) => complete).map(([id]) => id));
}

/** One in-flight operation and one exact uncertain creation request per observed source. */
export class ExperimentController {
  constructor({ list, create, status, cancel, resume, compare, onChange = () => {},
    id = interventionRequestId, now = () => performance.now() }) {
    this.services = { list, create, status, cancel, resume, compare };
    this.onChange = onChange;
    this.id = id;
    this.now = now;
    this.enabled = false;
    this.source = null;
    this.generation = 0;
    this.operation = null;
    this.clear();
  }

  clear() {
    this.page = null;
    this.cursor = null;
    this.selected = null;
    this.creation = null;
    this.comparison = null;
    this.runPageIndex = 0;
    this.lastCatalogRead = -Infinity;
    this.lastStatusRead = -Infinity;
    this.notice = "Connect to a remote host to run independent matched-seed experiments.";
  }

  get busy() { return this.operation !== null; }
  get canOperate() { return this.enabled && this.source !== null && !this.busy; }
  get canSelect() { return this.canOperate && this.creation?.state !== "unconfirmed"; }
  get canLaunch() { return this.canSelect; }
  get canRetry() { return this.canOperate && this.creation?.state === "unconfirmed"
    && this.creation.sessionId === this.source.sessionId; }
  get canCancel() { return this.canOperate && ACTIVE.has(this.selected?.status); }
  get canResume() { return this.canOperate && this.selected !== null && experimentCanResume(this.selected); }
  get canCompare() { return this.canOperate && completedVariants(this.selected).size >= 2; }
  get totalRunPages() { return Math.max(1, Math.ceil((this.selected?.runs.length ?? 0) / EXPERIMENT_RUN_PAGE_SIZE)); }
  get visibleRuns() {
    const start = this.runPageIndex * EXPERIMENT_RUN_PAGE_SIZE;
    return this.selected?.runs.slice(start, start + EXPERIMENT_RUN_PAGE_SIZE) ?? [];
  }

  emit() { this.onChange(this); }

  invalidate() {
    this.generation += 1;
    this.operation?.abort.abort();
    if (this.creation?.state === "pending") {
      this.creation.state = "unconfirmed";
      this.notice = "Experiment admission is unconfirmed. Reconnect to this host and retry the unchanged request.";
    }
  }

  reset(message) {
    const uncertain = this.creation?.state === "pending" || this.creation?.state === "unconfirmed";
    this.invalidate();
    this.source = null;
    this.enabled = false;
    this.clear();
    if (message) this.notice = message;
    if (uncertain) this.notice += " An earlier request may still finish on its original host; inspect that host's retained experiments before starting another.";
    this.emit();
  }

  observe(frame, sourceToken) {
    const sessionId = exactIdentity(frame?.identity?.sessionId);
    if (sourceToken === null || sourceToken === undefined) throw new Error("Experiments require an explicit remote source identity");
    if (!this.source || this.source.sessionId !== sessionId || this.source.token !== sourceToken) {
      const replaced = this.source !== null;
      this.invalidate();
      this.clear();
      this.source = { sessionId, token: sourceToken };
      if (replaced) this.notice = "The remote source changed. Prior experiments may still run on the previous host; its observations were cleared.";
      this.emit();
    }
    this.poll();
  }

  setEnabled(value) {
    const enabled = Boolean(value);
    if (enabled === this.enabled) return;
    this.enabled = enabled;
    if (!enabled) this.invalidate();
    this.emit();
    if (enabled) this.poll();
  }

  async run(kind, action, accept, failure) {
    if (!this.canOperate) return null;
    const operation = { kind, abort: new AbortController(), generation: this.generation };
    this.operation = operation;
    this.emit();
    try {
      const value = await action(operation.abort.signal);
      if (operation.generation !== this.generation || !this.enabled) return null;
      return accept(value) ?? true;
    } catch (error) {
      if (operation.generation === this.generation && this.enabled) {
        const label = failure?.(error) ?? kind;
        this.notice = label + ": " + errorText(error);
      }
      return null;
    } finally {
      if (this.operation === operation) this.operation = null;
      this.emit();
    }
  }

  poll() {
    if (!this.canOperate) return;
    if (this.selected && ACTIVE.has(this.selected.status)) {
      if (this.now() - this.lastStatusRead >= POLL_MS) void this.refreshStatus();
    } else if (!this.page && this.now() - this.lastCatalogRead >= POLL_MS) {
      void this.refreshCatalog();
    }
  }

  refreshCatalog(force = false) {
    if (!this.canOperate || (!force && this.now() - this.lastCatalogRead < POLL_MS)) return Promise.resolve(null);
    this.lastCatalogRead = this.now();
    const request = { sessionId: this.source.sessionId, cursor: this.cursor, limit: EXPERIMENT_PAGE_SIZE };
    return this.run("Experiment catalog unavailable", (signal) => this.services.list(request, signal), (page) => {
      this.page = copy(validateExperimentCatalog(page, { cursor: request.cursor, limit: request.limit }));
      // A summary cannot replace the verified per-run observation or its plan identity.
    });
  }

  firstPage() {
    if (!this.canOperate) return Promise.resolve(null);
    this.cursor = null;
    this.page = null;
    return this.refreshCatalog(true);
  }

  nextPage() {
    if (!this.canOperate || !this.page?.next_cursor) return Promise.resolve(null);
    this.cursor = this.page.next_cursor;
    this.page = null;
    return this.refreshCatalog(true);
  }

  open(experimentId) {
    if (!this.canSelect) return Promise.resolve(null);
    const known = this.page?.items.find((row) => row.experiment_id === experimentId);
    if (!known && this.selected?.experiment_id !== experimentId) return Promise.resolve(null);
    const request = { sessionId: this.source.sessionId, experimentId };
    return this.run("Retained experiment unavailable", (signal) => this.services.status(request, signal), (value) => {
      this.acceptStatus(value, experimentId);
      this.runPageIndex = 0;
    });
  }

  launch(input) {
    if (!this.canLaunch) return Promise.resolve(null);
    try {
      // Inputs are captured once before admission. Form edits cannot affect retry bytes.
      const prepared = prepareExperimentRequest({ ...input, idempotency_key: this.id() });
      this.creation = { prepared, sessionId: this.source.sessionId, state: "ready" };
    } catch (error) {
      this.notice = "Experiment request refused before submission: " + errorText(error);
      this.emit();
      return Promise.resolve(null);
    }
    this.comparison = null;
    return this.submitCreation();
  }

  retryLaunch() {
    if (!this.canRetry) return Promise.resolve(null);
    return this.submitCreation();
  }

  submitCreation() {
    if (!this.canOperate || !this.creation) return Promise.resolve(null);
    const creation = this.creation;
    const priorUncertainty = creation.state === "unconfirmed";
    const request = copy({ sessionId: creation.sessionId, request: creation.prepared.request, body: creation.prepared.body });
    creation.state = "pending";
    this.notice = "Submitting fresh matched-seed worlds; awaiting an experiment receipt.";
    return this.run("Experiment admission unconfirmed; retry the unchanged request",
      (signal) => this.services.create(request, signal), (value) => {
        const checked = validateExperimentStatus(value, { request: creation.prepared.request });
        this.acceptStatus(checked, checked.experiment_id, creation.prepared.request);
        creation.state = "observed";
        creation.experimentId = checked.experiment_id;
        this.page = null;
        this.cursor = null;
        this.runPageIndex = 0;
      }, (error) => {
        creation.state = !priorUncertainty && error?.admission === "not_admitted" ? "refused" : "unconfirmed";
        return creation.state === "refused" ? "Experiment refused; correct the inputs before submitting again"
          : "Experiment admission unconfirmed; retry the unchanged request";
      });
  }

  acceptStatus(value, experimentId, request) {
    const previous = this.selected?.experiment_id === experimentId ? this.selected : null;
    const capturedRequest = request ?? (this.creation?.state === "observed"
      && this.creation.experimentId === experimentId ? this.creation.prepared.request : undefined);
    const checked = validateExperimentStatus(value, { experimentId, request: capturedRequest,
      planDigest: previous?.plan_digest, minimumGeneration: previous?.generation, previousStatus: previous });
    if (this.comparison && (this.comparison.experiment_id !== checked.experiment_id
        || this.comparison.plan_digest !== checked.plan_digest || this.comparison.status_generation !== checked.generation)) {
      this.comparison = null;
    }
    this.selected = copy(checked);
    this.runPageIndex = Math.min(this.runPageIndex, this.totalRunPages - 1);
    this.lastStatusRead = this.now();
    this.notice = checked.status === "completed"
      ? "All experiment runs completed. Choose two variants and an endpoint to request a verified comparison."
      : checked.status === "failed" ? "Experiment reported a failure: " + (checked.error_reason || "inspect its run records") + ". Completed runs remain visible."
        : checked.status === "cancelled" || checked.status === "stopped"
          ? "Experiment stopped. Retained progress is visible; resume explicitly to continue unfinished runs."
          : checked.status === "cancelling" ? "Cancellation requested; waiting for workers to stop at their saved boundaries."
            : "Experiment admitted. Its independent worlds run on the host without stepping the displayed world.";
    return this.selected;
  }

  refreshStatus(force = false) {
    if (!this.canOperate || !this.selected || (!force && this.now() - this.lastStatusRead < POLL_MS)) return Promise.resolve(null);
    this.lastStatusRead = this.now();
    const request = { sessionId: this.source.sessionId, experimentId: this.selected.experiment_id };
    return this.run("Experiment status unavailable", (signal) => this.services.status(request, signal),
      (value) => this.acceptStatus(value, request.experimentId));
  }

  cancel() {
    if (!this.canCancel) return Promise.resolve(null);
    const request = { sessionId: this.source.sessionId, experimentId: this.selected.experiment_id };
    this.notice = "Requesting cancellation; displayed progress is the last observed state.";
    return this.run("Cancellation outcome unconfirmed; refresh status before retrying",
      (signal) => this.services.cancel(request, signal), (value) => this.acceptStatus(value, request.experimentId));
  }

  resume() {
    if (!this.canResume) return Promise.resolve(null);
    const request = { sessionId: this.source.sessionId, experimentId: this.selected.experiment_id };
    this.notice = "Requesting continuation of this experiment's unfinished runs.";
    return this.run("Resume outcome unconfirmed; refresh status before retrying",
      (signal) => this.services.resume(request, signal), (value) => this.acceptStatus(value, request.experimentId));
  }

  compare({ baselineVariant, challengerVariant, metric }) {
    if (!this.canCompare) return Promise.resolve(null);
    const status = this.selected;
    const complete = completedVariants(status);
    if (baselineVariant === challengerVariant || !complete.has(baselineVariant) || !complete.has(challengerVariant)) {
      this.notice = "Choose two distinct variants with every declared seed completed.";
      this.emit();
      return Promise.resolve(null);
    }
    const request = { sessionId: this.source.sessionId, experimentId: status.experiment_id,
      baseline_variant: baselineVariant, challenger_variant: challengerVariant, metric };
    this.comparison = null;
    this.notice = "The host is verifying endpoint evidence and comparing independent matched seeds.";
    return this.run("Comparison unavailable", (signal) => this.services.compare(request, signal), (value) => {
      this.comparison = copy(validateExperimentComparison(value, { status, baselineVariant, challengerVariant, metric }));
      this.notice = this.comparison.descriptive_only
        ? "One matched seed: descriptive difference only. No confidence interval or significance conclusion is available."
        : "Verified endpoint comparison loaded. A larger endpoint is not automatically better fitness.";
      return this.comparison;
    });
  }

  clearComparison() { this.comparison = null; this.emit(); }

  previousRuns() {
    if (this.runPageIndex > 0) { this.runPageIndex -= 1; this.emit(); }
  }

  nextRuns() {
    if (this.runPageIndex + 1 < this.totalRunPages) { this.runPageIndex += 1; this.emit(); }
  }
}
