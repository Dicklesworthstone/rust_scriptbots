// Remote saved-world studies reuse the application's checkpoint and branch services.
// A saved scientific cut is neither a browser continuation nor a new live owner.
import { exactIdentity } from "./inspector.js";
import { interventionRequestId, validateInterventionAction } from "./interventions.js";

export const CHECKPOINT_LAB_PAGE_SIZE = 12;
export const MAX_LAB_ARTIFACT_BYTES = 64 * 1024 * 1024;
const MAX_BRANCH_TICKS = 10000;
const POLL_MS = 2000;
const DIGEST = /^[0-9a-f]{64}$/;
const ID = /^[A-Za-z0-9_-]{1,128}$/;
const digest = (value) => typeof value === "string" && DIGEST.test(value);
const identifier = (value) => typeof value === "string" && ID.test(value);
const TERMINAL = new Set(["completed", "cancelled", "failed", "stopped"]);
const integer = (value, min = 0, max = Number.MAX_SAFE_INTEGER) =>
  Number.isSafeInteger(value) && value >= min && value <= max;
const scalar = (value) => typeof value === "number" && Number.isFinite(value) && Number.isFinite(Math.fround(value));
const text = (value, limit) => typeof value === "string" && value.length <= limit;
const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const keys = (value, required, optional = []) => object(value)
  && required.every((key) => Object.hasOwn(value, key))
  && Object.keys(value).every((key) => required.includes(key) || optional.includes(key));

function freeze(value) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

function copy(value) { return freeze(structuredClone(value)); }
function errorText(error) { return String(error?.message || error || "Unknown service failure").slice(0, 2048); }

/** Scientific checkpoint metadata is checked before it can become a study source. */
export function validateLabCheckpoint(value) {
  if (!object(value) || !identifier(value.checkpoint_id) || !integer(value.tick)
      || !integer(value.agent_count) || !integer(value.byte_size, 1)
      || value.agent_count > value.byte_size || !digest(value.world_digest_v1)
      || !digest(value.checksum_blake3) || !digest(value.checksum_sha256)
      || !text(value.schema, 128) || !value.schema.startsWith("scriptbots.world-checkpoint.")
      || !text(value.created_at_utc, 128) || !value.created_at_utc
      || (value.description != null && !text(value.description, 4096))) {
    throw new Error("The host returned invalid or inexact checkpoint metadata");
  }
  return value;
}

function sameCheckpoint(left, right) {
  return ["checkpoint_id", "tick", "agent_count", "byte_size", "world_digest_v1", "checksum_blake3", "checksum_sha256", "schema"]
    .every((key) => left[key] === right[key]);
}

function validatePage(page, cursor, validateItem) {
  const offset = cursor === null ? 0 : Number(cursor);
  if (!object(page) || !Array.isArray(page.items) || page.items.length > CHECKPOINT_LAB_PAGE_SIZE
      || page.limit !== CHECKPOINT_LAB_PAGE_SIZE || !integer(page.total) || !integer(page.offset)
      || page.offset !== offset || page.offset > page.total || page.items.length > page.total - page.offset
      || typeof page.has_more !== "boolean") throw new Error("The host returned an invalid bounded catalog page");
  const end = page.offset + page.items.length;
  if (page.has_more !== (end < page.total) || (page.has_more && page.items.length !== CHECKPOINT_LAB_PAGE_SIZE)
      || (page.has_more ? page.next_cursor !== String(end) : page.next_cursor != null)) {
    throw new Error("The catalog cursor disagrees with its returned rows");
  }
  const ids = new Set();
  for (const row of page.items) {
    validateItem(row);
    const id = row.checkpoint_id ?? row.artifact_id;
    if (ids.has(id)) throw new Error("The catalog repeats an artifact identity");
    ids.add(id);
  }
  return page;
}

export function validateLabCheckpointPage(page, cursor = null) {
  validatePage(page, cursor, validateLabCheckpoint);
  const retention = page.retention;
  if (!object(retention) || !["max_count", "max_bytes", "file_count", "file_bytes"]
    .every((key) => integer(retention[key]))) throw new Error("Checkpoint retention usage is unavailable or inexact");
  return page;
}

/** The browser offers explicit scalar treatments; the application validates the saved world too. */
export function validateLabIntervention(value) {
  if (value === null) return value;
  if (!keys(value, ["type", "params"])) throw new Error("Choose a supported checkpoint treatment");
  const p = value.params;
  if (value.type === "Drought" && keys(p, ["duration_ticks", "growth_scale"], ["region"])) {
    validateInterventionAction({ kind: "drought", region: p.region ?? { shape: "all" }, ticks: p.duration_ticks, growth_scale: p.growth_scale });
  } else if (value.type === "ResourceShock" && keys(p, ["bloom_amount", "scorch"], ["region"])) {
    validateInterventionAction({ kind: "bloom", region: p.region ?? { shape: "all" }, amount: p.bloom_amount });
    validateInterventionAction({ kind: "meteor", region: p.region ?? { shape: "all" }, lethality: 0, scorch: p.scorch });
    if (p.bloom_amount === 0 && p.scorch === 0) throw new Error("A resource shock must add or remove food");
  } else if (value.type === "TemperatureShift" && keys(p, ["discomfort_rate", "comfort_band"])) {
    if (![p.discomfort_rate, p.comfort_band].every((n) => scalar(n) && n >= 0)) {
      throw new Error("Temperature discomfort and comfort band must be finite and nonnegative");
    }
  } else if (value.type === "ClosedWorld" && keys(p, ["closed"]) && typeof p.closed === "boolean") {
    return value;
  } else if (value.type === "Core") {
    validateInterventionAction(p);
  } else {
    throw new Error("Choose a supported scalar checkpoint treatment without extra fields");
  }
  return value;
}

/** Reject changed source identities, impossible tick bounds and lifecycle regressions. */
export function validateLabBranchStatus(value, request, checkpoint, previous = null) {
  if (!object(value) || !identifier(value.branch_id) || value.checkpoint_id !== request.checkpoint_id
      || value.branch_name !== request.branch_name || !["running", "cancelling", ...TERMINAL].includes(value.status)
      || value.source_tick !== checkpoint.tick || !integer(value.target_tick)
      || value.target_tick !== checkpoint.tick + request.ticks
      || value.world_digest_v1 !== checkpoint.world_digest_v1
      || (value.error_reason != null && !text(value.error_reason, 4096))
      || (value.status === "completed" ? !identifier(value.artifact_id) || value.error_reason != null : value.artifact_id != null)) {
    throw new Error("The study observation does not match its saved source and requested future");
  }
  if (previous && (value.branch_id !== previous.branch_id
      || (TERMINAL.has(previous.status) && value.status !== previous.status)
      || (previous.status === "cancelling" && value.status === "running")
      || (previous.artifact_id != null && value.artifact_id !== previous.artifact_id))) {
    throw new Error("The study lifecycle regressed or changed its identity");
  }
  return value;
}

/** Catalog data supplies the identity and filename; no download URL comes from a report. */
export function validateLabArtifact(value, branchId = null) {
  if (!object(value) || !identifier(value.artifact_id) || !integer(value.byte_size)
      || !digest(value.checksum_blake3) || !digest(value.checksum_sha256)
      || !text(value.filename, 255) || !value.filename || /[/\\\u0000-\u001f\u007f]/.test(value.filename)
      || !text(value.content_type, 128) || !/^[a-z0-9.+-]+\/[a-z0-9.+-]+$/.test(value.content_type)
      || !text(value.relative_path, 2048) || !value.relative_path
      || value.relative_path.split("/").some((part) => !part || part === "." || part === ".." || /[\\\u0000-\u001f\u007f]/.test(part))
      || value.relative_path.split("/").at(-1) !== value.filename
      || (branchId !== null && !value.relative_path.startsWith(`branches/${branchId}/`))) {
    throw new Error("The host returned invalid or unrelated artifact metadata");
  }
  return value;
}

export function validateLabArtifactPage(page, branchId, cursor = null) {
  if (!identifier(branchId)) throw new Error("Invalid study identity");
  return validatePage(page, cursor, (row) => validateLabArtifact(row, branchId));
}

/** One bounded request slot, with explicit retries for potentially admitted mutations. */
export class CheckpointLabController {
  constructor({ list, save, launch, status, cancel, artifacts, download, onChange = () => {},
    id = interventionRequestId, now = () => performance.now() }) {
    this.services = { list, save, launch, status, cancel, artifacts, download };
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
    this.saveRequest = null;
    this.saveReceipt = null;
    this.saveState = "idle";
    this.branch = null;
    this.branchState = "idle";
    this.artifactsPage = null;
    this.artifactCursor = null;
    this.selectedArtifact = null;
    this.lastPageRead = -Infinity;
    this.lastBranchRead = -Infinity;
    this.lastArtifactRead = -Infinity;
    this.notice = null;
  }

  get busy() { return this.operation !== null; }
  get canOperate() { return this.enabled && this.source !== null && !this.busy; }
  get canSave() { return this.canOperate && this.saveState !== "unconfirmed"; }
  get canLaunch() { return this.canOperate && this.selected !== null
    && (!this.branch || this.branchState === "refused" || TERMINAL.has(this.branch.status?.status)); }

  emit() { this.onChange(this); }

  invalidate() {
    this.generation += 1;
    this.operation?.abort.abort();
    if (this.saveState === "pending") this.saveState = "unconfirmed";
    if (this.branchState === "pending") this.branchState = "unconfirmed";
  }

  reset(message = "Connect to a remote host to save scientific checkpoints and study their futures.") {
    this.invalidate();
    this.source = null;
    this.enabled = false;
    this.clear();
    this.notice = message;
    this.emit();
  }

  observe(frame, sourceToken) {
    const sessionId = exactIdentity(frame?.identity?.sessionId);
    const epoch = exactIdentity(frame?.snapshot?.epoch ?? "0");
    exactIdentity(frame?.identity?.tick);
    if (sourceToken === null || sourceToken === undefined) throw new Error("The Lab requires an explicit remote source identity");
    if (!this.source || this.source.sessionId !== sessionId || this.source.token !== sourceToken) {
      const replaced = this.source !== null;
      this.invalidate();
      this.clear();
      this.source = { sessionId, epoch, token: sourceToken };
      if (replaced) this.notice = "The remote source changed. Earlier requests may still finish on the previous host.";
      this.emit();
    } else {
      this.source.epoch = epoch;
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
      const result = await action(operation.abort.signal);
      if (operation.generation !== this.generation || !this.enabled) return null;
      const accepted = accept(result);
      return accepted ?? true;
    } catch (error) {
      if (operation.generation === this.generation && this.enabled) {
        const label = failure?.(error) ?? kind;
        this.notice = `${label}: ${errorText(error)}`;
      }
      return null;
    } finally {
      if (this.operation === operation) this.operation = null;
      this.emit();
    }
  }

  poll() {
    if (!this.canOperate) return;
    if (this.branch?.status && !TERMINAL.has(this.branch.status.status)) {
      if (this.now() - this.lastBranchRead >= POLL_MS) void this.refreshBranch();
    } else if (this.branch?.status?.status === "completed" && !this.artifactsPage) {
      if (this.now() - this.lastArtifactRead >= POLL_MS) void this.refreshArtifacts();
    } else if (!this.page && this.now() - this.lastPageRead >= POLL_MS) void this.refresh();
  }

  refresh(force = false) {
    if (!this.canOperate || (!force && this.now() - this.lastPageRead < POLL_MS)) return Promise.resolve(null);
    this.lastPageRead = this.now();
    const request = { sessionId: this.source.sessionId, cursor: this.cursor, limit: CHECKPOINT_LAB_PAGE_SIZE };
    return this.run("Checkpoint catalog", (signal) => this.services.list(request, signal), (page) => {
      validateLabCheckpointPage(page, request.cursor);
      if (this.selected) {
        const same = page.items.find((row) => row.checkpoint_id === this.selected.checkpoint_id);
        if (same && !sameCheckpoint(same, this.selected)) throw new Error("A retained checkpoint changed its scientific identity");
      }
      this.page = copy(page);
    });
  }

  nextPage() {
    if (!this.canOperate || !this.page?.next_cursor) return Promise.resolve(null);
    this.cursor = this.page.next_cursor;
    this.page = null;
    return this.refresh(true);
  }

  firstPage() {
    if (!this.canOperate) return Promise.resolve(null);
    this.cursor = null;
    this.page = null;
    return this.refresh(true);
  }

  select(checkpointId) {
    if (this.busy) return;
    const selected = this.page?.items.find((row) => row.checkpoint_id === checkpointId)
      ?? (this.saveReceipt?.checkpoint_id === checkpointId ? this.saveReceipt : null);
    if (!selected) return;
    this.selected = copy(selected);
    this.emit();
  }

  save() {
    if (!this.canSave) return Promise.resolve(null);
    try {
      const key = this.id();
      if (!text(key, 1024) || !key.trim()) throw new Error("A stable checkpoint retry identity is unavailable");
      this.saveRequest = copy({ sessionId: this.source.sessionId, description: "Saved from the browser Evolution Lab", idempotency_key: key });
    } catch (error) { this.notice = errorText(error); this.emit(); return Promise.resolve(null); }
    this.saveReceipt = null;
    return this.submitSave();
  }

  retrySave() {
    if (!this.canOperate || this.saveState !== "unconfirmed" || this.saveRequest?.sessionId !== this.source.sessionId) return Promise.resolve(null);
    return this.submitSave();
  }

  submitSave() {
    const request = this.saveRequest;
    const priorUncertainty = this.saveState === "unconfirmed";
    this.saveState = "pending";
    this.notice = "Saving a scientific checkpoint on the remote host; awaiting its verified receipt.";
    return this.run("Checkpoint outcome unconfirmed; retry the same save", (signal) => this.services.save(request, signal), (value) => {
      validateLabCheckpoint(value);
      this.saveReceipt = copy(value);
      this.selected = this.saveReceipt;
      this.saveState = "saved";
      this.page = null;
      this.cursor = null;
      this.notice = `Saved checkpoint at tick ${value.tick} with ${value.agent_count} agents.`;
    }, (error) => {
      this.saveState = !priorUncertainty && error?.admission === "not_admitted" ? "refused" : "unconfirmed";
      return this.saveState === "refused" ? "Checkpoint save refused; a new save can be requested"
        : "Checkpoint outcome unconfirmed; retry the same save";
    });
  }

  launch({ branchName, ticks, intervention = null }) {
    if (!this.canLaunch) return Promise.resolve(null);
    try {
      if (typeof branchName !== "string" || !/^[A-Za-z0-9_-]{1,64}$/.test(branchName)) throw new Error("Study name needs 1–64 letters, digits, underscores or hyphens");
      if (!integer(ticks, 1, MAX_BRANCH_TICKS) || !integer(this.selected.tick + ticks)) throw new Error("Choose 1–10,000 future ticks within the exact browser integer range");
      validateLabIntervention(intervention);
      const key = this.id();
      if (!text(key, 1024) || !key.trim()) throw new Error("A stable study retry identity is unavailable");
      const request = copy({ sessionId: this.source.sessionId, checkpoint_id: this.selected.checkpoint_id,
        branch_name: branchName, ticks, intervention, idempotency_key: key });
      this.branch = { request, checkpoint: this.selected, status: null };
      this.artifactsPage = null;
      this.selectedArtifact = null;
      this.artifactCursor = null;
    } catch (error) { this.notice = errorText(error); this.emit(); return Promise.resolve(null); }
    return this.submitBranch();
  }

  retryLaunch() {
    if (!this.canOperate || this.branchState !== "unconfirmed" || this.branch?.request.sessionId !== this.source.sessionId) return Promise.resolve(null);
    return this.submitBranch();
  }

  submitBranch() {
    const branch = this.branch;
    const priorUncertainty = this.branchState === "unconfirmed";
    this.branchState = "pending";
    this.notice = "Submitting an independent saved-world study; awaiting job admission.";
    return this.run("Study admission unconfirmed; retry the unchanged request", (signal) => this.services.launch(branch.request, signal), (value) => {
      this.acceptStatus(value, branch);
    }, (error) => {
      this.branchState = !priorUncertainty && error?.admission === "not_admitted" ? "refused" : "unconfirmed";
      return this.branchState === "refused" ? "Study request refused; correct the inputs and start a new request"
        : "Study admission unconfirmed; retry the unchanged request";
    });
  }

  acceptStatus(value, branch) {
    validateLabBranchStatus(value, branch.request, branch.checkpoint, branch.status);
    branch.status = copy(value);
    this.branchState = "observed";
    this.lastBranchRead = this.now();
    this.notice = value.status === "completed"
      ? "Study completed. Its report and retained branch artifacts are available after server integrity verification."
      : value.status === "stopped" ? "The study stopped after a host interruption and will not rerun automatically. Start a new study explicitly to run this saved source again."
        : value.status === "cancelled" ? "The study stopped after cancellation; interrupted output remains on the host."
          : value.status === "failed" ? `Study failed: ${value.error_reason || "the host reported no additional detail"}`
          : value.status === "cancelling" ? "Cancellation requested; waiting for the worker to stop."
            : "Study admitted and running independently of the displayed world.";
  }

  refreshBranch(force = false) {
    if (!this.canOperate || !this.branch?.status || (!force && this.now() - this.lastBranchRead < POLL_MS)) return Promise.resolve(null);
    this.lastBranchRead = this.now();
    const branch = this.branch;
    const request = { sessionId: this.source.sessionId, branchId: branch.status.branch_id };
    return this.run("Study status unavailable", (signal) => this.services.status(request, signal), (value) => this.acceptStatus(value, branch));
  }

  cancelBranch() {
    if (!this.canOperate || !this.branch?.status || TERMINAL.has(this.branch.status.status)) return Promise.resolve(null);
    const branch = this.branch;
    const request = { sessionId: this.source.sessionId, branchId: branch.status.branch_id };
    this.notice = "Requesting cancellation; the previous status remains the last observed outcome.";
    return this.run("Cancellation outcome unconfirmed; refresh status or retry cancellation", (signal) => this.services.cancel(request, signal), (value) => this.acceptStatus(value, branch));
  }

  refreshArtifacts(force = false) {
    if (!this.canOperate || this.branch?.status?.status !== "completed" || typeof this.services.artifacts !== "function"
        || (!force && this.now() - this.lastArtifactRead < POLL_MS)) return Promise.resolve(null);
    this.lastArtifactRead = this.now();
    const request = { sessionId: this.source.sessionId, branchId: this.branch.status.branch_id,
      cursor: this.artifactCursor, limit: CHECKPOINT_LAB_PAGE_SIZE };
    return this.run("Study artifacts unavailable", (signal) => this.services.artifacts(request, signal), (page) => {
      validateLabArtifactPage(page, request.branchId, request.cursor);
      this.artifactsPage = copy(page);
      if (this.selectedArtifact && !page.items.some((row) => row.artifact_id === this.selectedArtifact.artifact_id)) this.selectedArtifact = null;
    });
  }

  firstArtifactPage() {
    if (!this.canOperate) return Promise.resolve(null);
    this.artifactCursor = null;
    this.artifactsPage = null;
    this.selectedArtifact = null;
    return this.refreshArtifacts(true);
  }

  nextArtifactPage() {
    if (!this.canOperate || !this.artifactsPage?.next_cursor) return Promise.resolve(null);
    this.artifactCursor = this.artifactsPage.next_cursor;
    this.artifactsPage = null;
    this.selectedArtifact = null;
    return this.refreshArtifacts(true);
  }

  selectArtifact(id) {
    if (this.busy) return;
    this.selectedArtifact = this.artifactsPage?.items.find((row) => row.artifact_id === id) ?? null;
    this.emit();
  }

  downloadCheckpoint() {
    if (!this.selected) return Promise.resolve(null);
    return this.download(this.selected.checkpoint_id, this.selected, null);
  }

  downloadReport() {
    if (this.branch?.status?.status !== "completed") return Promise.resolve(null);
    return this.download(this.branch.status.artifact_id, null, this.branch.status.branch_id);
  }

  downloadSelectedArtifact() {
    if (!this.selectedArtifact || this.branch?.status?.status !== "completed") return Promise.resolve(null);
    return this.download(this.selectedArtifact.artifact_id, this.selectedArtifact, this.branch.status.branch_id);
  }

  download(artifactId, expected, branchId) {
    if (!this.canOperate) return Promise.resolve(null);
    const request = { sessionId: this.source.sessionId, artifactId, expected };
    return this.run("Artifact download refused", (signal) => this.services.download(request, signal), (result) => {
      validateLabArtifact(result?.metadata, branchId);
      const metadata = result.metadata;
      if (metadata.artifact_id !== artifactId || !(result.blob instanceof Blob)
          || result.blob.size !== metadata.byte_size || result.filename !== metadata.filename
          || (expected && ["byte_size", "checksum_blake3", "checksum_sha256"].some((key) => metadata[key] !== expected[key]))) {
        throw new Error("Downloaded artifact does not match the selected verified metadata");
      }
      this.notice = `Prepared ${metadata.filename}; downloaded bytes match its recorded size and SHA-256.`;
      return result;
    });
  }
}

function updateSelect(select, rows, selected, key, label) {
  if (!select) return;
  const entries = [["", "Choose a saved item"], ...rows.map((row) => [row[key], label(row)])];
  const optionsJson = JSON.stringify(entries);
  if (select.dataset.options !== optionsJson) {
    select.replaceChildren(...entries.map(([value, title]) => {
      const option = select.ownerDocument.createElement("option");
      option.value = value;
      option.textContent = title;
      return option;
    }));
    select.dataset.options = optionsJson;
  }
  select.value = selected ?? "";
}

/** Presentation only: callers attach events and handle the returned download Blob. */
export function renderCheckpointLab(panel, controller) {
  if (!panel) return;
  const find = (id) => panel.querySelector(`#${id}`);
  const write = (id, value) => { const node = find(id); if (node && node.textContent !== value) node.textContent = value; };
  const disable = (id, value) => { const node = find(id); if (node) node.disabled = value; };
  const ready = controller.canOperate;
  panel.dataset.state = controller.busy ? "busy" : controller.enabled ? "ready" : "inactive";
  write("lab-source", controller.source
    ? `Remote host session ${controller.source.sessionId}${controller.enabled ? "" : " — connect and open this panel to continue"}. Saved studies run on that host.`
    : "Connect to a remote simulation to save a scientific checkpoint and explore an independent future.");
  write("lab-notice", controller.notice || "Save the current remote world or select an existing scientific checkpoint.");
  const rows = [...(controller.page?.items ?? [])];
  for (const row of [controller.saveReceipt, controller.selected]) {
    if (row && !rows.some((item) => item.checkpoint_id === row.checkpoint_id)) rows.unshift(row);
  }
  updateSelect(find("lab-checkpoint-select"), rows, controller.selected?.checkpoint_id, "checkpoint_id",
    (row) => `Tick ${row.tick} · ${row.agent_count} agents · ${row.checkpoint_id}`);
  const selected = controller.selected;
  const retention = controller.page?.retention;
  write("lab-checkpoint-detail", selected
    ? `Saved tick ${selected.tick}; ${selected.agent_count} agents; ${selected.byte_size} bytes. World digest ${selected.world_digest_v1}. Scientific state only; the live owner remains unchanged.`
    : retention ? `${controller.page.total} saved checkpoints; ${retention.file_bytes} of ${retention.max_bytes} retained bytes. Choose one to study.` : "No saved checkpoint selected.");
  disable("lab-checkpoint-select", !ready);
  disable("lab-checkpoint-save", !controller.canSave);
  disable("lab-checkpoint-retry", !ready || controller.saveState !== "unconfirmed");
  disable("lab-checkpoint-refresh", !ready);
  disable("lab-checkpoint-first", !ready || controller.cursor === null);
  disable("lab-checkpoint-next", !ready || !controller.page?.next_cursor);
  disable("lab-checkpoint-download", !ready || !selected);
  disable("lab-branch-launch", !controller.canLaunch);
  disable("lab-branch-retry", !ready || controller.branchState !== "unconfirmed");
  const branch = controller.branch, status = branch?.status;
  disable("lab-branch-refresh", !ready || !status);
  disable("lab-branch-cancel", !ready || !status || TERMINAL.has(status.status));
  disable("lab-report-download", !ready || status?.status !== "completed");
  write("lab-branch-detail", branch
    ? `${status ? `Study ${status.branch_id}: ${status.status}` : controller.branchState === "refused" ? "Study request refused; no job admitted by this request" : "Study admission is unconfirmed"}. Saved tick ${branch.checkpoint.tick} → target ${branch.checkpoint.tick + branch.request.ticks}. ${branch.request.intervention ? "Treatment and unchanged control start from the identical saved state." : "One unchanged scientific future."} These are bounds, not live progress. One source gives a descriptive trajectory comparison, not independent-seed statistical inference.`
    : "Choose a horizon and, optionally, a treatment beside an unchanged control. The remote live world is neither advanced nor replaced by this study.");
  updateSelect(find("lab-artifact-select"), controller.artifactsPage?.items ?? [], controller.selectedArtifact?.artifact_id,
    "artifact_id", (row) => `${row.relative_path.split("/").slice(2).join("/")} · ${row.byte_size} bytes`);
  write("lab-artifact-detail", controller.selectedArtifact
    ? `${controller.selectedArtifact.filename}; SHA-256 ${controller.selectedArtifact.checksum_sha256}. Downloads verify the recorded byte count and SHA-256.`
    : controller.artifactsPage ? `${controller.artifactsPage.total} retained study artifacts; page begins at ${controller.artifactsPage.offset + 1}. Bundle verification describes artifact integrity.` : "Completed studies publish their report, saved source, plan and branch bundle files here.");
  disable("lab-artifact-select", !ready || status?.status !== "completed");
  disable("lab-artifact-refresh", !ready || status?.status !== "completed");
  disable("lab-artifact-first", !ready || controller.artifactCursor === null);
  disable("lab-artifact-next", !ready || !controller.artifactsPage?.next_cursor);
  disable("lab-artifact-download", !ready || !controller.selectedArtifact);
}
