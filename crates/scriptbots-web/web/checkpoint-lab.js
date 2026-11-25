// Remote saved-world studies reuse the application's checkpoint and branch services.
// A saved scientific cut is neither a browser continuation nor a new live owner.
import { exactIdentity } from "./inspector.js";
import { interventionRequestId, validateInterventionAction } from "./interventions.js";
import { MAX_STUDY_REPORT_BYTES, parseSavedStudyReport } from "./study-report.js";

export const CHECKPOINT_LAB_PAGE_SIZE = 12;
export const MAX_LAB_ARTIFACT_BYTES = 64 * 1024 * 1024;
const MAX_BRANCH_TICKS = 10000;
const POLL_MS = 2000;
// WorldDigestV1 uses FNV-1a64. Artifact integrity uses BLAKE3/SHA-256;
// admitting one format must never make the other format interchangeable.
const WORLD_DIGEST = /^[0-9a-f]{16}$/;
const CHECKSUM = /^[0-9a-f]{64}$/;
const ID = /^[A-Za-z0-9_-]{1,128}$/;
const WORLD_CHECKPOINT_SCHEMA = "scriptbots.world-checkpoint.v1.3";
const HOST_CHECKPOINT_SCHEMA = "scriptbots.host-checkpoint.v1";
const CHECKPOINT_SCHEMAS = new Set([WORLD_CHECKPOINT_SCHEMA, HOST_CHECKPOINT_SCHEMA]);
const worldDigest = (value) => typeof value === "string" && WORLD_DIGEST.test(value);
const checksum = (value) => typeof value === "string" && CHECKSUM.test(value);
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
      || value.agent_count > value.byte_size || !worldDigest(value.world_digest_v1)
      || !checksum(value.checksum_blake3) || !checksum(value.checksum_sha256)
      || !CHECKPOINT_SCHEMAS.has(value.schema)
      || !text(value.created_at_utc, 128) || !value.created_at_utc
      || (value.description != null && !text(value.description, 4096))) {
    throw new Error("The host returned invalid or inexact checkpoint metadata");
  }
  return value;
}

function sameCheckpoint(left, right) {
  return ["checkpoint_id", "tick", "agent_count", "byte_size", "world_digest_v1", "checksum_blake3", "checksum_sha256", "schema", "created_at_utc", "description"]
    .every((key) => left[key] === right[key]);
}

/** A reconstructed canvas is a saved scientific cut, never a live publication. */
export function validateCheckpointCanvas(value, { sessionId, checkpoint }) {
  validateLabCheckpoint(checkpoint);
  validateLabCheckpoint(value?.checkpoint);
  exactIdentity(sessionId, 128);
  if (value?.schema !== "scriptbots.checkpoint-canvas.v1"
      || value.sourceSessionId !== sessionId || !sameCheckpoint(value.checkpoint, checkpoint)) {
    throw new Error("The saved-world preview does not match the selected checkpoint and host");
  }
  const frame = value.frame, identity = frame?.identity, snapshot = frame?.snapshot;
  for (const field of ["sessionId", "publicationRevision", "scientificRevision",
    "controlRevision", "configRevision", "tick"]) exactIdentity(identity?.[field], field === "sessionId" ? 128 : 64);
  if (frame?.schema !== "scriptbots.browser-frame.v1" || identity.sessionId !== sessionId
      || identity.tick !== String(checkpoint.tick) || snapshot?.tick !== checkpoint.tick
      || snapshot.summary?.agentCount !== checkpoint.agent_count || !Array.isArray(snapshot.agents)
      || snapshot.agents.length !== checkpoint.agent_count || frame.playback?.paused !== true
      || frame.playback.speedMultiplier !== 1 || !object(frame.environment)) {
    throw new Error("The reconstructed canvas disagrees with its saved scientific cut");
  }
  return value;
}

/** Preserve decimal targets exactly; the current browser snapshot uses safe Numbers. */
export function parseReplayTarget(value) {
  try { exactIdentity(value); }
  catch (_) { throw new Error("Enter a whole target tick using decimal digits without signs, spaces, exponents or leading zeroes"); }
  const tick = BigInt(value);
  if (tick > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error("The browser cannot display ticks above 9007199254740991 exactly; use a native replay client for larger ticks");
  }
  return tick;
}

/** A replay carries the selected source and the actual scope of its digest check. */
export function validateCheckpointReplay(value, { sessionId, checkpoint, targetTick }) {
  validateLabCheckpoint(checkpoint);
  validateLabCheckpoint(value?.checkpoint);
  exactIdentity(sessionId, 128);
  const target = parseReplayTarget(targetTick), source = BigInt(checkpoint.tick);
  if (value?.schema !== "scriptbots.checkpoint-replay.v1"
      || value.sourceSessionId !== sessionId || !sameCheckpoint(value.checkpoint, checkpoint)
      || value.targetTick !== targetTick || target < source) {
    throw new Error("The historical replay does not match its selected checkpoint, target and host");
  }
  const bound = BigInt(exactIdentity(value.maxVerificationTicks));
  if (bound === 0n) throw new Error("The replay verification bound is invalid");
  const verification = value.verification;
  if (target === source) {
    if (!keys(verification, ["kind"]) || verification.kind !== "saved_cut") {
      throw new Error("A saved-cut review requires saved-cut verification");
    }
  } else {
    if (!keys(verification, ["kind", "verifiedThroughTick", "capturedCurrentDigest"])
        || verification.kind !== "captured_interval" || !worldDigest(verification.capturedCurrentDigest)) {
      throw new Error("Historical replay requires captured-current interval verification");
    }
    const endpoint = BigInt(exactIdentity(verification.verifiedThroughTick));
    if (endpoint < target || endpoint - source > bound) {
      throw new Error("The complete replay verification interval is outside its observed bounds");
    }
  }
  const frame = value.frame, identity = frame?.identity, snapshot = frame?.snapshot;
  for (const field of ["sessionId", "publicationRevision", "scientificRevision",
    "controlRevision", "configRevision", "tick"]) exactIdentity(identity?.[field], field === "sessionId" ? 128 : 64);
  if (frame?.schema !== "scriptbots.browser-frame.v1" || identity.sessionId !== sessionId
      || identity.tick !== targetTick || snapshot?.tick !== Number(target)
      || !integer(snapshot.summary?.agentCount) || !Array.isArray(snapshot.agents)
      || snapshot.agents.length !== snapshot.summary.agentCount
      || (target === source && snapshot.agents.length !== checkpoint.agent_count)
      || frame.playback?.paused !== true || frame.playback.speedMultiplier !== 1 || !object(frame.environment)) {
    throw new Error("The reconstructed canvas disagrees with its verified historical target");
  }
  return value;
}

function validatePage(page, cursor, validateItem, identityKey) {
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
    const id = row[identityKey];
    if (ids.has(id)) throw new Error("The catalog repeats an artifact identity");
    ids.add(id);
  }
  return page;
}

export function validateLabCheckpointPage(page, cursor = null) {
  validatePage(page, cursor, validateLabCheckpoint, "checkpoint_id");
  const retention = page.retention;
  if (!object(retention) || !["max_count", "max_bytes", "file_count", "file_bytes"]
    .every((key) => integer(retention[key]))) throw new Error("Checkpoint retention usage is unavailable or inexact");
  return page;
}

/** Build the bounded archive treatment offered by the saved-world form. */
export function archiveStudyTreatment(count, seed) {
  return validateLabIntervention({ type: "Core", params: {
    kind: "spawn_from_archive", selector: { TopKByQuality: count },
    placement: { placement: "seeded", region: { shape: "all" }, seed },
    headroom: count, herbivore_tendency: null,
  } });
}

/** The application also validates source capabilities and the actual saved population. */
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
  } else if (value.type === "Core" && p?.kind === "spawn_from_archive") {
    if (!keys(p, ["kind", "selector", "placement", "headroom", "herbivore_tendency"])
        || !keys(p.selector, ["TopKByQuality"]) || !integer(p.selector.TopKByQuality, 1, 1000)
        || !keys(p.placement, ["placement", "region", "seed"]) || p.placement.placement !== "seeded"
        || !keys(p.placement.region, ["shape"]) || p.placement.region.shape !== "all"
        || !integer(p.placement.seed, 0, 0xffff_ffff)
        || !integer(p.headroom, p.selector.TopKByQuality, 1000)
        || (p.herbivore_tendency !== null && (!scalar(p.herbivore_tendency)
          || p.herbivore_tendency < 0 || p.herbivore_tendency > 1))) {
      throw new Error("Archive studies need 1–1000 top occupied cells, a whole-world placement seed from 0 through 4294967295, and sufficient population headroom");
    }
  } else if (value.type === "Core") {
    validateInterventionAction(p);
  } else {
    throw new Error("Choose a supported checkpoint treatment without extra fields");
  }
  return value;
}

/** A delay counts completed source transitions and must leave a treatment transition. */
export function validateLabTreatmentDelay(delay, ticks, intervention) {
  if (!integer(ticks, 1, MAX_BRANCH_TICKS) || !integer(delay, 0, ticks - 1)) {
    throw new Error("Treatment delay must be a whole number from zero through one less than the future tick horizon");
  }
  if (delay > 0 && intervention === null) {
    throw new Error("Choose a treatment before delaying its application");
  }
  return delay;
}

/** A retained receipt describes observed science without reconstructing its request. */
export function validateLabBranchObservation(value, previous = null) {
  if (!object(value) || !identifier(value.branch_id) || !identifier(value.checkpoint_id)
      || !text(value.branch_name, 64) || !/^[A-Za-z0-9_-]{1,64}$/.test(value.branch_name)
      || !["running", "cancelling", ...TERMINAL].includes(value.status)
      || !integer(value.source_tick) || !integer(value.target_tick)
      || !integer(value.target_tick - value.source_tick, 1, MAX_BRANCH_TICKS)
      || !worldDigest(value.world_digest_v1)
      || (value.error_reason != null && !text(value.error_reason, 4096))
      || (value.status === "completed" ? !identifier(value.artifact_id) || value.error_reason != null : value.artifact_id != null)) {
    throw new Error("The host returned invalid or inexact retained study metadata");
  }
  if (previous && ["branch_id", "checkpoint_id", "branch_name", "source_tick", "target_tick", "world_digest_v1"]
    .some((key) => value[key] !== previous[key])) {
    throw new Error("The study observation does not match its retained source identity");
  }
  if (previous && ((TERMINAL.has(previous.status) && value.status !== previous.status)
      || (previous.status === "cancelling" && value.status === "running")
      || (previous.artifact_id != null && value.artifact_id !== previous.artifact_id))) {
    throw new Error("The study lifecycle regressed or changed its identity");
  }
  return value;
}

/** Reject changed source identities and bounds for a request made by this browser. */
export function validateLabBranchStatus(value, request, checkpoint, previous = null) {
  validateLabBranchObservation(value, previous);
  if (value.checkpoint_id !== request.checkpoint_id || value.branch_name !== request.branch_name
      || value.source_tick !== checkpoint.tick || value.target_tick !== checkpoint.tick + request.ticks
      || value.world_digest_v1 !== checkpoint.world_digest_v1) {
    throw new Error("The study observation does not match its saved source and requested future");
  }
  return value;
}

export function validateLabBranchPage(page, cursor = null) {
  return validatePage(page, cursor, validateLabBranchObservation, "branch_id");
}

/** Catalog data supplies the identity and filename; no download URL comes from a report. */
export function validateLabArtifact(value, branchId = null) {
  if (!object(value) || !identifier(value.artifact_id) || !integer(value.byte_size)
      || !checksum(value.checksum_blake3) || !checksum(value.checksum_sha256)
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
  return validatePage(page, cursor, (row) => validateLabArtifact(row, branchId), "artifact_id");
}

function validateDownload(result, artifactId, expected, branchId) {
  validateLabArtifact(result?.metadata, branchId);
  const metadata = result.metadata;
  if (metadata.artifact_id !== artifactId || !(result.blob instanceof Blob)
      || result.blob.size !== metadata.byte_size || result.filename !== metadata.filename
      || (expected && ["byte_size", "checksum_blake3", "checksum_sha256"].some((key) => metadata[key] !== expected[key]))) {
    throw new Error("Downloaded artifact does not match the selected verified metadata");
  }
  return result;
}

/** One bounded request slot, with explicit retries for potentially admitted mutations. */
export class CheckpointLabController {
  constructor({ list, listStudies, save, review, replay, launch, status, cancel, artifacts, download, onChange = () => {},
    id = interventionRequestId, now = () => performance.now(), parseReport = parseSavedStudyReport }) {
    this.services = { list, listStudies, save, review, replay, launch, status, cancel, artifacts, download };
    this.onChange = onChange;
    this.id = id;
    this.now = now;
    this.parseReport = parseReport;
    this.enabled = false;
    this.ownerWritable = true;
    this.source = null;
    this.generation = 0;
    this.operation = null;
    this.clear();
  }

  clear() {
    this.page = null;
    this.cursor = null;
    this.selected = null;
    this.canvasPreview = null;
    this.replayTickText = "";
    this.saveRequest = null;
    this.saveReceipt = null;
    this.saveState = "idle";
    this.branch = null;
    this.branchState = "idle";
    this.historyPage = null;
    this.historyCursor = null;
    this.reportModel = null;
    this.artifactsPage = null;
    this.artifactCursor = null;
    this.selectedArtifact = null;
    this.lastPageRead = -Infinity;
    this.lastBranchRead = -Infinity;
    this.lastHistoryRead = -Infinity;
    this.lastArtifactRead = -Infinity;
    this.notice = null;
  }

  get busy() { return this.operation !== null; }
  get canOperate() { return this.enabled && this.source !== null && !this.busy; }
  get canSave() { return this.canOperate && this.ownerWritable && this.canvasPreview === null && this.saveState !== "unconfirmed"; }
  get canReview() { return this.canOperate && this.selected !== null && typeof this.services.review === "function"; }
  get canImportLocally() { return this.canOperate && this.selected?.schema === WORLD_CHECKPOINT_SCHEMA; }
  get reviewPending() { return ["Saved-world review unavailable", "Historical replay unavailable"].includes(this.operation?.kind); }
  get canEditReplay() { return this.enabled && this.source !== null && this.selected !== null && (!this.busy || this.reviewPending); }
  get replayTarget() {
    try {
      const tick = parseReplayTarget(this.replayTickText);
      return this.selected && tick >= BigInt(this.selected.tick) ? String(tick) : null;
    } catch (_) { return null; }
  }
  get canSeekReplay() { return this.canOperate && this.replayTarget !== null && typeof this.services.replay === "function"; }
  get canSelectStudy() { return this.canOperate && this.branchState !== "unconfirmed"; }
  get canLaunch() { return this.canOperate && this.canvasPreview === null && this.selected !== null
    && (!this.branch || this.branchState === "refused" || TERMINAL.has(this.branch.status?.status)); }

  emit() { this.onChange(this); }

  setOwnerWritable(value) {
    const writable = Boolean(value);
    if (writable === this.ownerWritable) return;
    this.ownerWritable = writable;
    this.emit();
  }

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
    const sessionId = exactIdentity(frame?.identity?.sessionId, 128);
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
    } else if (!this.page && this.now() - this.lastPageRead >= POLL_MS) {
      void this.refresh();
    } else if (!this.historyPage && typeof this.services.listStudies === "function"
        && this.now() - this.lastHistoryRead >= POLL_MS) void this.refreshHistory();
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
    if (!this.selected || !sameCheckpoint(this.selected, selected)) {
      this.canvasPreview = null;
      this.replayTickText = String(selected.tick);
    }
    this.selected = copy(selected);
    this.emit();
  }

  /** An edit supersedes pending viewing intent without releasing its occupied worker. */
  editReplayTick(value) {
    if (!this.canEditReplay || typeof value !== "string" || value === this.replayTickText) return;
    if (this.reviewPending) this.invalidate();
    // An overlong paste remains invalid instead of truncating to an admissible tick.
    this.replayTickText = value.slice(0, 64);
    try {
      const target = parseReplayTarget(this.replayTickText);
      if (target < BigInt(this.selected.tick)) throw new Error("The replay target precedes the selected checkpoint");
      this.notice = `Target tick ${target}; seek explicitly to reconstruct it. The displayed world is unchanged.`;
    } catch (error) { this.notice = errorText(error); }
    this.emit();
  }

  stepReplayTick(delta) {
    if (!this.canEditReplay || ![-10, -1, 1, 10].includes(delta)) return;
    try {
      const next = parseReplayTarget(this.replayTickText) + BigInt(delta);
      if (next < BigInt(this.selected.tick) || next > BigInt(Number.MAX_SAFE_INTEGER)) {
        throw new Error("The stepped target is outside the selected checkpoint and exact browser tick range");
      }
      this.editReplayTick(String(next));
    } catch (error) { this.notice = errorText(error); this.emit(); }
  }

  /** Verify the selected source through the captured current endpoint before displaying a target. */
  seekReplay() {
    if (!this.canSeekReplay) return Promise.resolve(null);
    const request = copy({ sessionId: this.source.sessionId, checkpoint: this.selected, targetTick: this.replayTarget });
    this.notice = `Reconstructing tick ${request.targetTick} from the selected checkpoint; the live server continues independently.`;
    return this.run("Historical replay unavailable", (signal) => this.services.replay(request, signal), (value) => {
      validateCheckpointReplay(value, request);
      if (!this.selected || !sameCheckpoint(this.selected, request.checkpoint) || this.replayTarget !== request.targetTick) {
        throw new Error("The selected replay source or target changed while reconstruction was loading");
      }
      this.canvasPreview = freeze(value);
      this.notice = value.verification.kind === "saved_cut"
        ? `Saved cut verified at tick ${value.targetTick}. The live owner is unchanged.`
        : `Historical tick ${value.targetTick} ready; the complete interval was checked through captured current tick ${value.verification.verifiedThroughTick}, digest ${value.verification.capturedCurrentDigest}. This is not a separate target digest.`;
      return this.canvasPreview;
    });
  }

  /** One explicit background reconstruction, through the existing bounded request slot. */
  reviewSelected() {
    if (!this.canReview) return Promise.resolve(null);
    const request = { sessionId: this.source.sessionId, checkpoint: this.selected };
    this.notice = "Reconstructing the selected saved world; the live server continues independently.";
    return this.run("Saved-world review unavailable", (signal) => this.services.review(request, signal), (value) => {
      validateCheckpointCanvas(value, request);
      if (!this.selected || !sameCheckpoint(this.selected, request.checkpoint)) {
        throw new Error("The selected checkpoint changed while its preview was loading");
      }
      this.canvasPreview = freeze(value);
      this.notice = `Saved-world preview ready at tick ${value.checkpoint.tick}. Return to live before controlling or inspecting the current population.`;
      return this.canvasPreview;
    });
  }

  /** Leaving a preview cancels a pending replacement; its late result cannot replace live. */
  returnToLive() {
    const pending = this.reviewPending;
    if (pending) this.invalidate();
    if (this.canvasPreview === null && !pending) { this.emit(); return; }
    this.canvasPreview = null;
    this.notice = "Returned to the current live view. No simulation command was submitted.";
    this.emit();
  }

  checkStudySource(value) {
    for (const source of [this.selected, this.saveReceipt, this.branch?.checkpoint, ...(this.page?.items ?? [])]) {
      if (source?.checkpoint_id === value.checkpoint_id
          && (source.tick !== value.source_tick || source.world_digest_v1 !== value.world_digest_v1)) {
        throw new Error("The retained study does not match the known saved checkpoint identity");
      }
    }
  }

  refreshHistory(force = false) {
    if (!this.canOperate || typeof this.services.listStudies !== "function"
        || (!force && this.now() - this.lastHistoryRead < POLL_MS)) return Promise.resolve(null);
    this.lastHistoryRead = this.now();
    const request = { sessionId: this.source.sessionId, cursor: this.historyCursor, limit: CHECKPOINT_LAB_PAGE_SIZE };
    return this.run("Study history unavailable", (signal) => this.services.listStudies(request, signal), (page) => {
      validateLabBranchPage(page, request.cursor);
      for (const row of page.items) {
        this.checkStudySource(row);
        const previous = row.branch_id === this.branch?.status?.branch_id ? this.branch.status
          : this.historyPage?.items.find((item) => item.branch_id === row.branch_id);
        if (previous) validateLabBranchObservation(row, previous);
      }
      this.historyPage = copy(page);
      const current = page.items.find((row) => row.branch_id === this.branch?.status?.branch_id);
      if (current) this.acceptStatus(current, this.branch);
    });
  }

  firstHistoryPage() {
    if (!this.canOperate) return Promise.resolve(null);
    this.historyCursor = null;
    this.historyPage = null;
    return this.refreshHistory(true);
  }

  nextHistoryPage() {
    if (!this.canOperate || !this.historyPage?.next_cursor) return Promise.resolve(null);
    this.historyCursor = this.historyPage.next_cursor;
    this.historyPage = null;
    return this.refreshHistory(true);
  }

  selectStudy(branchId) {
    if (!this.canSelectStudy) {
      if (this.canOperate && this.branchState === "unconfirmed") {
        this.notice = "Study admission remains unconfirmed. Retry the unchanged request before selecting another study.";
        this.emit();
      }
      return Promise.resolve(null);
    }
    const receipt = this.branch?.status?.branch_id === branchId ? this.branch.status
      : this.historyPage?.items.find((row) => row.branch_id === branchId) ?? null;
    if (!receipt) return Promise.resolve(null);
    const observed = copy(receipt);
    const request = { sessionId: this.source.sessionId, branchId: observed.branch_id };
    return this.run("Retained study unavailable", (signal) => this.services.status(request, signal), (value) => {
      validateLabBranchObservation(value, observed);
      this.checkStudySource(value);
      const branch = { request: null, checkpoint: null, status: observed, origin: "history" };
      this.acceptStatus(value, branch);
      this.branch = branch;
      this.reportModel = null;
      this.artifactsPage = null;
      this.selectedArtifact = null;
      this.artifactCursor = null;
      this.lastArtifactRead = -Infinity;
    });
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
    if (!this.canOperate || !this.ownerWritable || this.canvasPreview !== null || this.saveState !== "unconfirmed"
        || this.saveRequest?.sessionId !== this.source.sessionId) return Promise.resolve(null);
    return this.submitSave();
  }

  submitSave() {
    if (!this.ownerWritable || this.canvasPreview !== null) return Promise.resolve(null);
    const request = this.saveRequest;
    const priorUncertainty = this.saveState === "unconfirmed";
    this.saveState = "pending";
    this.notice = "Saving a scientific checkpoint on the remote host; awaiting its verified receipt.";
    return this.run("Checkpoint outcome unconfirmed; retry the same save", (signal) => this.services.save(request, signal), (value) => {
      validateLabCheckpoint(value);
      this.saveReceipt = copy(value);
      this.selected = this.saveReceipt;
      this.replayTickText = String(value.tick);
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

  launch({ branchName, ticks, intervention = null, interventionDelayTicks = 0 }) {
    if (!this.canLaunch) return Promise.resolve(null);
    try {
      if (typeof branchName !== "string" || !/^[A-Za-z0-9_-]{1,64}$/.test(branchName)) throw new Error("Study name needs 1–64 letters, digits, underscores or hyphens");
      if (!integer(ticks, 1, MAX_BRANCH_TICKS) || !integer(this.selected.tick + ticks)) throw new Error("Choose 1–10,000 future ticks within the exact browser integer range");
      validateLabIntervention(intervention);
      validateLabTreatmentDelay(interventionDelayTicks, ticks, intervention);
      if (intervention?.type === "Core" && intervention.params.kind === "spawn_from_archive"
          && this.selected.schema !== HOST_CHECKPOINT_SCHEMA) {
        throw new Error("Archive resurrection requires a complete checkpoint with a retained behavioral archive");
      }
      const key = this.id();
      if (!text(key, 1024) || !key.trim()) throw new Error("A stable study retry identity is unavailable");
      const request = copy({ sessionId: this.source.sessionId, checkpoint_id: this.selected.checkpoint_id,
        branch_name: branchName, ticks, intervention,
        ...(interventionDelayTicks > 0 ? { intervention_delay_ticks: interventionDelayTicks } : {}),
        idempotency_key: key });
      this.branch = { request, checkpoint: this.selected, status: null };
      this.reportModel = null;
      this.artifactsPage = null;
      this.selectedArtifact = null;
      this.artifactCursor = null;
    } catch (error) { this.notice = errorText(error); this.emit(); return Promise.resolve(null); }
    return this.submitBranch();
  }

  retryLaunch() {
    if (!this.canOperate || this.canvasPreview !== null || this.branchState !== "unconfirmed" || this.branch?.request?.sessionId !== this.source.sessionId) return Promise.resolve(null);
    return this.submitBranch();
  }

  submitBranch() {
    if (!this.canOperate || this.canvasPreview !== null || !this.branch?.request || !this.branch.checkpoint) return Promise.resolve(null);
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
    if (branch.request && branch.checkpoint) validateLabBranchStatus(value, branch.request, branch.checkpoint, branch.status);
    else if (branch.origin === "history" && branch.status) validateLabBranchObservation(value, branch.status);
    else throw new Error("A study requires its submitted request or an observed retained identity");
    branch.status = copy(value);
    this.branchState = "observed";
    this.lastBranchRead = this.now();
    this.notice = value.status === "completed"
      ? "Study completed. Its report and retained branch artifacts are available after server integrity verification."
      : value.status === "stopped" ? "The study stopped after a host interruption and will not rerun automatically. Choose a retained checkpoint and start a new study explicitly."
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
    if (!this.canOperate || this.canvasPreview !== null || !this.branch?.status || TERMINAL.has(this.branch.status.status)) return Promise.resolve(null);
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

  /** Read one verified scientific file for an explicit new local owner. */
  prepareLocalImport(maxBytes) {
    if (!this.canOperate || !this.selected) return Promise.resolve(null);
    if (!integer(maxBytes, 1)) throw new Error("The browser package did not provide a valid scientific checkpoint limit");
    if (!this.canImportLocally) {
      this.notice = "This complete owner checkpoint requires native restoration. Review, download and server studies remain available; local import accepts standalone world checkpoints.";
      this.emit();
      return Promise.resolve(null);
    }
    const limit = Math.min(maxBytes, MAX_LAB_ARTIFACT_BYTES);
    const checkpoint = this.selected;
    if (checkpoint.byte_size > limit) {
      this.notice = `This checkpoint exceeds the ${limit} byte browser transfer limit. It remains available for server studies.`;
      this.emit();
      return Promise.resolve(null);
    }
    const request = { sessionId: this.source.sessionId, artifactId: checkpoint.checkpoint_id,
      expected: checkpoint, maxBytes: limit };
    const sourceToken = this.source.token;
    this.notice = "Reading the selected checkpoint for a new paused local world; the server continues independently.";
    return this.run("Local checkpoint transfer unavailable", async (signal) => {
      const result = validateDownload(await this.services.download(request, signal), request.artifactId, checkpoint, null);
      signal.throwIfAborted();
      if (result.blob.size > limit) throw new Error("Scientific checkpoint exceeds the browser transfer byte limit");
      const bytes = new Uint8Array(await result.blob.arrayBuffer());
      signal.throwIfAborted();
      return bytes;
    }, (bytes) => {
      if (!this.selected || !sameCheckpoint(this.selected, checkpoint) || bytes.length !== checkpoint.byte_size) {
        throw new Error("The selected checkpoint changed during local transfer");
      }
      this.notice = "Scientific bytes verified. The browser still must validate and restore their scientific state.";
      return { bytes, checkpoint, sourceSessionId: request.sessionId, sourceToken };
    });
  }

  downloadReport() {
    if (this.branch?.status?.status !== "completed") return Promise.resolve(null);
    return this.download(this.branch.status.artifact_id, null, this.branch.status.branch_id);
  }

  previewReport() {
    if (!this.canOperate || this.branch?.status?.status !== "completed") return Promise.resolve(null);
    const status = this.branch.status, checkpoint = this.branch.checkpoint;
    const submittedRequest = this.branch.request == null ? null : copy(this.branch.request);
    const request = { sessionId: this.source.sessionId, artifactId: status.artifact_id,
      expected: null, maxBytes: MAX_STUDY_REPORT_BYTES };
    return this.run("Study report unavailable", async (signal) => {
      const result = validateDownload(await this.services.download(request, signal),
        status.artifact_id, null, status.branch_id);
      signal.throwIfAborted();
      if (result.blob.size > MAX_STUDY_REPORT_BYTES) throw new Error("Study report exceeds the browser preview byte limit");
      return this.parseReport(result.blob, { status, checkpoint, request: submittedRequest, artifact: result.metadata });
    }, (model) => {
      this.reportModel = model;
      this.notice = "Verified saved-study report loaded. These descriptive trajectories come from one saved source.";
      return model;
    });
  }

  downloadSelectedArtifact() {
    if (!this.selectedArtifact || this.branch?.status?.status !== "completed") return Promise.resolve(null);
    return this.download(this.selectedArtifact.artifact_id, this.selectedArtifact, this.branch.status.branch_id);
  }

  download(artifactId, expected, branchId) {
    if (!this.canOperate) return Promise.resolve(null);
    const request = { sessionId: this.source.sessionId, artifactId, expected };
    return this.run("Artifact download refused", (signal) => this.services.download(request, signal), (result) => {
      validateDownload(result, artifactId, expected, branchId);
      const metadata = result.metadata;
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
    ? `Saved tick ${selected.tick}; ${selected.agent_count} agents; ${selected.byte_size} bytes. World digest ${selected.world_digest_v1}. ${selected.schema === HOST_CHECKPOINT_SCHEMA ? "Complete owner checkpoint. Review and server studies preserve its saved science. Independent studies do not inherit owner commands or schedules; local import requires a standalone world checkpoint." : "Standalone scientific checkpoint; the live owner remains unchanged."}`
    : retention ? `${controller.page.total} saved checkpoints; ${retention.file_bytes} of ${retention.max_bytes} retained bytes. Choose one to study.` : "No saved checkpoint selected.");
  disable("lab-checkpoint-select", !ready);
  disable("lab-checkpoint-save", !controller.canSave);
  disable("lab-checkpoint-retry", !ready || !controller.ownerWritable || controller.canvasPreview !== null || controller.saveState !== "unconfirmed");
  disable("lab-checkpoint-refresh", !ready);
  disable("lab-checkpoint-first", !ready || controller.cursor === null);
  disable("lab-checkpoint-next", !ready || !controller.page?.next_cursor);
  disable("lab-checkpoint-download", !ready || !selected);
  disable("lab-checkpoint-review", !controller.canReview);
  const targetInput = find("lab-replay-tick");
  if (targetInput && targetInput.value !== controller.replayTickText) targetInput.value = controller.replayTickText;
  disable("lab-replay-tick", !controller.canEditReplay);
  disable("lab-replay-seek", !controller.canSeekReplay);
  for (const [id, delta] of [["lab-replay-back-ten", -10], ["lab-replay-back-one", -1],
    ["lab-replay-forward-one", 1], ["lab-replay-forward-ten", 10]]) {
    const target = controller.replayTarget === null ? null : BigInt(controller.replayTarget) + BigInt(delta);
    disable(id, !controller.canEditReplay || target === null || target < BigInt(selected?.tick ?? 0)
      || target > BigInt(Number.MAX_SAFE_INTEGER));
  }
  disable("lab-checkpoint-local", !controller.canImportLocally);
  disable("lab-branch-launch", !controller.canLaunch);
  disable("lab-branch-retry", !ready || controller.canvasPreview !== null || controller.branchState !== "unconfirmed");
  const branch = controller.branch, status = branch?.status;
  const history = (controller.historyPage?.items ?? []).map((row) => row.branch_id === status?.branch_id ? status : row);
  if (status && !history.some((row) => row.branch_id === status.branch_id)) history.unshift(status);
  updateSelect(find("lab-history-select"), history, status?.branch_id, "branch_id",
    (row) => `${row.branch_name} · ${row.status} · ${row.source_tick} → ${row.target_tick} · ${row.branch_id}`);
  disable("lab-history-select", !controller.canSelectStudy);
  disable("lab-history-refresh", !ready);
  disable("lab-history-first", !ready || controller.historyCursor === null);
  disable("lab-history-next", !ready || !controller.historyPage?.next_cursor);
  write("lab-history-detail", controller.branchState === "unconfirmed"
    ? "Retry the unchanged pending admission before selecting a different retained study. Reading history does not resolve that request's identity."
    : controller.historyPage ? `${controller.historyPage.total} retained studies; page begins at ${controller.historyPage.offset + 1}. Selecting a study reads its current status without starting it again.`
      : "Load retained studies from this host to reconnect to their status and downloads after a browser refresh or host restart.");
  disable("lab-branch-refresh", !ready || !status);
  disable("lab-branch-cancel", !ready || controller.canvasPreview !== null || !status || TERMINAL.has(status.status));
  disable("lab-report-download", !ready || status?.status !== "completed");
  disable("lab-report-preview", !ready || status?.status !== "completed");
  const studyDesign = branch?.request
    ? branch.request.intervention
      ? `Treatment and unchanged control start from the identical saved state. Treatment requested at tick ${branch.checkpoint.tick + (branch.request.intervention_delay_ticks ?? 0)}, after ${branch.request.intervention_delay_ticks ?? 0} future ticks.`
      : "One unchanged scientific future."
    : "Selected from retained history. The original treatment is unknown from this receipt; inspect the verified plan or report for its design.";
  write("lab-branch-detail", branch
    ? `${status ? `Study ${status.branch_id}: ${status.status}` : controller.branchState === "refused" ? "Study request refused; no job admitted by this request" : "Study admission is unconfirmed"}. Saved tick ${status?.source_tick ?? branch.checkpoint.tick} → target ${status?.target_tick ?? branch.checkpoint.tick + branch.request.ticks}. Source ${status?.checkpoint_id ?? branch.request.checkpoint_id}; world digest ${status?.world_digest_v1 ?? branch.checkpoint.world_digest_v1}. ${studyDesign} These are bounds, not live progress. One source gives a descriptive trajectory comparison, not independent-seed statistical inference.`
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
