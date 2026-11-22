// A browser experiment is a separate matched-seed cohort on the connected host.
// This controller owns requests and observations, never a world or a scientific clock.
import { exactIdentity } from "./inspector.js";
import { interventionRequestId } from "./interventions.js";
import { EXPERIMENT_LIMITS, prepareExperimentRequest, validateExperimentCatalog, validateExperimentStatus,
  validateExperimentComparison, validateExperimentRunArtifacts, experimentRunArtifactPath,
  experimentCanResume } from "./experiment-protocol.js";

export const EXPERIMENT_PAGE_SIZE = 12;
export const EXPERIMENT_RUN_PAGE_SIZE = 16;
const POLL_MS = 2000;
const ACTIVE = new Set(["pending", "running", "cancelling"]);
const ZIP_CRC_TABLE = Uint32Array.from({ length: 256 }, (_, value) => {
  for (let bit = 0; bit < 8; bit += 1) value = (value >>> 1) ^ (0xedb88320 & -(value & 1));
  return value >>> 0;
});
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

function requireDownloadActive(signal) {
  if (signal.aborted) throw new Error("Run download was cancelled; no file was exported");
}

async function verifiedRunFile(result, expected, signal) {
  requireDownloadActive(signal);
  if (!result || !(result.blob instanceof Blob) || result.blob.size !== expected.byte_size
      || result.filename !== expected.filename
      || Object.keys(expected).some((key) => result.metadata?.[key] !== expected[key])) {
    throw new Error("Downloaded file does not match the selected run's immutable inventory");
  }
  const bytes = new Uint8Array(await result.blob.arrayBuffer());
  requireDownloadActive(signal);
  if (!globalThis.crypto?.subtle) throw new Error("Verified downloads require Web Crypto on a secure connection");
  const digest = new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", bytes));
  requireDownloadActive(signal);
  const actual = Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
  if (actual !== expected.checksum_sha256) throw new Error("Downloaded run file failed SHA-256 verification");
  return bytes;
}

// Store original verified bytes without recompression or renaming. The bounded
// inventory fits classic ZIP; no ZIP64, host paths, or external archive dependency.
function runBundleZip(files) {
  const local = [], central = [];
  let offset = 0, centralBytes = 0;
  for (const { path, bytes } of files) {
    const name = new TextEncoder().encode(path);
    let crc = 0xffffffff;
    for (const byte of bytes) crc = (crc >>> 8) ^ ZIP_CRC_TABLE[(crc ^ byte) & 0xff];
    crc = (crc ^ 0xffffffff) >>> 0;
    const header = new Uint8Array(30), h = new DataView(header.buffer);
    h.setUint32(0, 0x04034b50, true);
    h.setUint16(4, 20, true);
    h.setUint16(6, 0x800, true);
    h.setUint16(12, 33, true); // 1980-01-01; the manifest retains scientific provenance.
    h.setUint32(14, crc, true);
    h.setUint32(18, bytes.length, true);
    h.setUint32(22, bytes.length, true);
    h.setUint16(26, name.length, true);
    const entry = new Uint8Array(46), c = new DataView(entry.buffer);
    c.setUint32(0, 0x02014b50, true);
    c.setUint16(4, 20, true);
    c.setUint16(6, 20, true);
    c.setUint16(8, 0x800, true);
    c.setUint16(14, 33, true);
    c.setUint32(16, crc, true);
    c.setUint32(20, bytes.length, true);
    c.setUint32(24, bytes.length, true);
    c.setUint16(28, name.length, true);
    c.setUint32(42, offset, true);
    local.push(header, name, bytes);
    central.push(entry, name);
    offset += header.length + name.length + bytes.length;
    centralBytes += entry.length + name.length;
  }
  const end = new Uint8Array(22), e = new DataView(end.buffer);
  e.setUint32(0, 0x06054b50, true);
  e.setUint16(8, files.length, true);
  e.setUint16(10, files.length, true);
  e.setUint32(12, centralBytes, true);
  e.setUint32(16, offset, true);
  return new Blob([...local, ...central, end], { type: "application/zip" });
}

function verifyBundleIndex(files, inventory) {
  const bytes = files.find((file) => file.path === "bundle_manifest.json")?.bytes;
  // Inspect only the portable artifact index. Preserve all original bytes;
  // full-width seeds inside the manifest are never interpreted or reserialized.
  const manifest = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  const expected = new Map(inventory.artifacts.filter((artifact) =>
    experimentRunArtifactPath(artifact, inventory) !== "bundle_manifest.json")
    .map((artifact) => [experimentRunArtifactPath(artifact, inventory), artifact]));
  if (manifest?.bundle_version !== "scriptbots.run-bundle.v1" || !Array.isArray(manifest.artifacts)
      || manifest.artifacts.length !== expected.size || manifest.digests?.max_tick !== inventory.run.total_ticks) {
    throw new Error("Run bundle manifest does not match the complete downloaded inventory");
  }
  for (const entry of manifest.artifacts) {
    const artifact = expected.get(entry?.relative_path);
    if (!artifact || entry.bytes_len !== artifact.byte_size || entry.blake3_hex !== artifact.checksum_blake3) {
      throw new Error("Run bundle manifest has a missing, replaced, or duplicated artifact");
    }
    expected.delete(entry.relative_path);
  }
}

/** One in-flight operation and one exact uncertain creation request per observed source. */
export class ExperimentController {
  constructor({ list, create, status, cancel, resume, retryFailed, compare, artifacts, download, onChange = () => {},
    id = interventionRequestId, now = () => performance.now() }) {
    this.services = { list, create, status, cancel, resume, retryFailed, compare, artifacts, download };
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
    this.failedRetry = null;
    this.comparison = null;
    this.artifactRunId = null;
    this.runArtifacts = null;
    this.selectedArtifact = null;
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
  get canRetryFailed() { return this.canOperate
    && (!this.failedRetry || (this.failedRetry.refreshed
      && this.failedRetry.experimentId === this.selected?.experiment_id
      && this.failedRetry.generation === this.selected?.generation))
    && this.selected?.failed_runs > 0
    && ["stopped", "cancelled", "failed"].includes(this.selected.status); }
  get canCompare() { return this.canOperate && completedVariants(this.selected).size >= 2; }
  get completedRuns() { return this.selected?.runs.filter((run) => run.state === "completed") ?? []; }
  get canReadArtifacts() { return this.canOperate && this.completedRuns.some((run) => run.run_id === this.artifactRunId); }
  get canDownloadBundle() { return this.canReadArtifacts && this.runArtifacts?.run.run_id === this.artifactRunId; }
  get canDownloadArtifact() { return this.canDownloadBundle && this.selectedArtifact !== null; }
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
      planDigest: previous?.plan_digest, minimumGeneration: previous?.generation, previousStatus: previous,
      retryGeneration: this.failedRetry?.experimentId === experimentId ? this.failedRetry.generation : undefined });
    if (this.comparison && (this.comparison.experiment_id !== checked.experiment_id
        || this.comparison.plan_digest !== checked.plan_digest || this.comparison.status_generation !== checked.generation)) {
      this.comparison = null;
    }
    this.selected = copy(checked);
    if (!previous || !this.completedRuns.some((run) => run.run_id === this.artifactRunId)) {
      this.artifactRunId = this.completedRuns[0]?.run_id ?? null;
      this.runArtifacts = null;
      this.selectedArtifact = null;
    }
    if (this.failedRetry) {
      if (checked.experiment_id !== this.failedRetry.experimentId
          || checked.generation > this.failedRetry.generation) this.failedRetry = null;
      // A lost POST may still be validating when this read sees the old status.
      // Preserve its authority for a later observation; only an explicit reissue
      // of this same generation is now allowed.
      else if (checked.generation === this.failedRetry.generation) this.failedRetry.refreshed = true;
    }
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

  retryFailed() {
    if (!this.canRetryFailed) return Promise.resolve(null);
    const request = { sessionId: this.source.sessionId, experimentId: this.selected.experiment_id,
      expectedGeneration: this.selected.generation };
    // Keep this authority after a lost response so an explicit status refresh
    // can observe the retry, without silently starting another failed attempt.
    this.failedRetry = { experimentId: request.experimentId, generation: request.expectedGeneration, refreshed: false };
    this.notice = "Verifying saved continuations and preserving the failed attempt before retrying.";
    return this.run("Failed-run retry outcome unconfirmed; refresh status before trying again",
      (signal) => this.services.retryFailed(request, signal), (value) => {
        if (!Number.isSafeInteger(value?.generation) || value.generation <= request.expectedGeneration) {
          throw new Error("The host did not observe a newer retry generation");
        }
        return this.acceptStatus(value, request.experimentId);
      });
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

  selectArtifactRun(runId) {
    if (!this.canOperate || !this.completedRuns.some((run) => run.run_id === runId)) return false;
    if (this.artifactRunId !== runId) {
      this.artifactRunId = runId;
      this.runArtifacts = null;
      this.selectedArtifact = null;
    }
    this.emit();
    return true;
  }

  loadRunArtifacts() {
    if (!this.canReadArtifacts) return Promise.resolve(null);
    const status = this.selected;
    const request = copy({ sessionId: this.source.sessionId, experimentId: status.experiment_id, runId: this.artifactRunId });
    this.runArtifacts = null;
    this.selectedArtifact = null;
    this.notice = "Verifying the selected completed run and discovering its portable files.";
    return this.run("Completed run files unavailable", (signal) => this.services.artifacts(request, signal), (value) => {
      this.runArtifacts = copy(validateExperimentRunArtifacts(value, { status, runId: request.runId }));
      this.selectedArtifact = this.runArtifacts.artifacts.find((artifact) =>
        experimentRunArtifactPath(artifact, this.runArtifacts) === "bundle_manifest.json");
      this.notice = "The host verified this completed run. Choose an individual file or download the complete portable bundle.";
      return this.runArtifacts;
    });
  }

  selectArtifact(artifactId) {
    if (!this.canDownloadBundle) return false;
    const selected = this.runArtifacts.artifacts.find((artifact) => artifact.artifact_id === artifactId);
    if (!selected) return false;
    this.selectedArtifact = selected;
    this.emit();
    return true;
  }

  downloadSelectedArtifact() {
    if (!this.canDownloadArtifact) return Promise.resolve(null);
    const expected = this.selectedArtifact;
    const request = copy({ sessionId: this.source.sessionId, artifactId: expected.artifact_id,
      expected, maxBytes: EXPERIMENT_LIMITS.maxRunArtifactBytes });
    this.notice = "Downloading and checking the selected run file.";
    return this.run("Run file download unavailable", async (signal) => {
      const result = await this.services.download(request, signal);
      await verifiedRunFile(result, expected, signal);
      return { blob: result.blob, filename: expected.filename };
    }, (result) => {
      this.notice = "Verified run file is ready to save. Your browser manages the final file save.";
      return result;
    });
  }

  downloadRunBundle() {
    if (!this.canDownloadBundle) return Promise.resolve(null);
    const inventory = this.runArtifacts;
    const sessionId = this.source.sessionId;
    this.notice = "Downloading and checking every file before preparing the portable run bundle.";
    return this.run("Run bundle download unavailable", async (signal) => {
      const files = [];
      for (const expected of inventory.artifacts) {
        requireDownloadActive(signal);
        const request = copy({ sessionId, artifactId: expected.artifact_id, expected,
          maxBytes: EXPERIMENT_LIMITS.maxRunArtifactBytes });
        const result = await this.services.download(request, signal);
        const bytes = await verifiedRunFile(result, expected, signal);
        files.push({ path: experimentRunArtifactPath(expected, inventory), bytes });
      }
      requireDownloadActive(signal);
      verifyBundleIndex(files, inventory);
      return { blob: runBundleZip(files), filename: inventory.run.run_id + ".zip" };
    }, (result) => {
      this.notice = "Complete verified run bundle is ready to save. Extract the ZIP to preserve its manifest and relative paths.";
      return result;
    });
  }

  previousRuns() {
    if (this.runPageIndex > 0) { this.runPageIndex -= 1; this.emit(); }
  }

  nextRuns() {
    if (this.runPageIndex + 1 < this.totalRunPages) { this.runPageIndex += 1; this.emit(); }
  }
}
