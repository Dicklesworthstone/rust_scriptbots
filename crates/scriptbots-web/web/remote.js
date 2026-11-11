// Remote presentation never owns or drives a local SimHandle. Binary decoding is
// supplied by the shared Rust/WASM protocol; this module owns connection and
// consumer cursors, with no queue of historical canvas frames.
import { MAX_LAB_ARTIFACT_BYTES, validateLabArtifact, validateLabIntervention } from "./checkpoint-lab.js";

const VERSION = 1;
const STREAM_VERSION = 2;
const MAX_FRAME_BYTES = 8 * 1024 * 1024;
const MAX_CONTROL_BYTES = 16 * 1024;
const MAX_SEND_BUFFER_BYTES = 64 * 1024;
const MAX_REQUESTS = 32;
const MAX_INSPECTION_BYTES = 16 * 1024;
const MAX_BRAIN_ACTIVITY_BYTES = 512 * 1024;
const CHANNELS = ["canvas", "telemetry", "events"];
const IDENTITY_FIELDS = ["sessionId", "publicationRevision", "scientificRevision",
  "controlRevision", "configRevision", "tick"];
const encoder = new TextEncoder();

/** Preserve actual HTTP refusal separately from transport failures and lost receipts. */
export class RemoteOwnerHttpError extends Error {
  constructor(status, message) {
    super(message);
    this.name = "RemoteOwnerHttpError";
    this.status = status;
    // 503 includes both capacity refusal and unknown owner outcomes. Checkpoint
    // 507 can follow a retained file write, so both require the original retry key.
    this.admission = [400, 404, 409, 413, 415, 422].includes(status) ? "not_admitted" : "unknown";
  }
}

/** One deadline and one actual-byte ceiling cover every JSON owner response. */
async function boundedOwnerJson(url, byteLimit, { signal, method = "GET", body, missingMessage,
  sessionId, timeoutMs = 2000 } = {}) {
  const abort = new AbortController();
  const cancel = () => abort.abort();
  if (signal?.aborted) cancel();
  signal?.addEventListener("abort", cancel, { once: true });
  const timeout = setTimeout(cancel, timeoutMs);
  try {
    const headers = { ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      ...(sessionId === undefined ? {} : { "X-ScriptBots-Session": sessionId }) };
    const response = await fetch(url, { method, body, cache: "no-store", credentials: "omit",
      ...(Object.keys(headers).length ? { headers } : {}), signal: abort.signal });
    const advertised = response.headers.get("content-length");
    if (advertised !== null && (!/^\d+$/.test(advertised) || Number(advertised) > byteLimit)) {
      await response.body?.cancel();
      throw new Error(`Remote response exceeds its ${byteLimit / 1024} KiB bound`);
    }
    if (!response.body) throw new Error("Remote owner returned no response body");
    const reader = response.body.getReader();
    const chunks = [];
    let length = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        length += value.byteLength;
        if (length > byteLimit) { await reader.cancel(); throw new Error(`Remote response exceeds its ${byteLimit / 1024} KiB bound`); }
        chunks.push(value);
      }
    } finally { reader.releaseLock(); }
    const bytes = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    let detail;
    try { detail = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)); }
    catch (error) {
      if (!response.ok) throw new RemoteOwnerHttpError(response.status, `Remote owner request failed (${response.status})`);
      throw error;
    }
    if (response.status === 404 && missingMessage && detail.message === missingMessage) return null;
    if (!response.ok) throw new RemoteOwnerHttpError(response.status, detail.error || detail.message || `Remote owner request failed (${response.status})`);
    return detail;
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", cancel);
  }
}

function labIdentifier(value) {
  if (typeof value !== "string" || !/^[A-Za-z0-9_-]{1,128}$/.test(value)) throw new Error("Invalid bounded Lab artifact identity");
  return value;
}

function labPageQuery(cursor, limit) {
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > 100) throw new Error("Invalid bounded Lab page size");
  if (cursor !== null && (decimal(cursor) > BigInt(Number.MAX_SAFE_INTEGER))) throw new Error("Lab page cursor exceeds exact browser integers");
  return { limit: String(limit), ...(cursor === null ? {} : { cursor }) };
}

/** Bytes are bounded before allocation, then checked against the catalog's exact SHA-256. */
async function downloadLabBytes(url, metadata, sessionId, signal) {
  if (!globalThis.crypto?.subtle) throw new Error("SHA-256 verification requires a secure browser context");
  const abort = new AbortController();
  const cancel = () => abort.abort();
  if (signal?.aborted) cancel();
  signal?.addEventListener("abort", cancel, { once: true });
  const timeout = setTimeout(cancel, 15000);
  try {
    const response = await fetch(url, { cache: "no-store", credentials: "omit",
      headers: { "X-ScriptBots-Session": sessionId }, signal: abort.signal });
    const limit = response.ok ? metadata.byte_size : 16 * 1024;
    const advertised = response.headers.get("content-length");
    if (advertised !== null && (!/^(0|[1-9][0-9]*)$/.test(advertised)
        || Number(advertised) > limit || (response.ok && Number(advertised) !== metadata.byte_size))) {
      await response.body?.cancel();
      throw new Error("Artifact response size differs from its catalog metadata");
    }
    if (!response.body) throw new Error("The host returned no artifact bytes");
    const reader = response.body.getReader(), chunks = [];
    let length = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        length += value.byteLength;
        if (length > limit || length > MAX_LAB_ARTIFACT_BYTES) {
          await reader.cancel();
          throw new Error("Artifact response exceeds its exact byte bound");
        }
        chunks.push(value);
      }
    } finally { reader.releaseLock(); }
    const bytes = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    if (!response.ok) {
      let detail;
      try { detail = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)); } catch (_) { /* Report bounded HTTP status when no JSON error exists. */ }
      throw new Error(detail?.message || detail?.error || `Artifact download failed (${response.status})`);
    }
    if (length !== metadata.byte_size) throw new Error("Artifact response is truncated");
    const checksum = Array.from(new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", bytes)),
      (byte) => byte.toString(16).padStart(2, "0")).join("");
    if (checksum !== metadata.checksum_sha256) throw new Error("Downloaded artifact SHA-256 differs from its verified catalog entry");
    return new Blob([bytes], { type: metadata.content_type });
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", cancel);
  }
}

function decimal(value, bits = 64) {
  const maxDigits = bits === 128 ? 39 : 20;
  if (typeof value !== "string" || value.length > maxDigits || !/^(0|[1-9][0-9]*)$/.test(value)) {
    throw new Error("Remote identity must be a canonical decimal string");
  }
  const number = BigInt(value);
  if (number >= (1n << BigInt(bits))) throw new Error("Remote identity exceeds its integer range");
  return number;
}

function validateChannels(channels) {
  if (!Array.isArray(channels) || channels.length < 1 || channels.length > CHANNELS.length
      || new Set(channels).size !== channels.length || channels.some((channel) => !CHANNELS.includes(channel))) {
    throw new Error("Choose one to three distinct remote channels: canvas, telemetry, events");
  }
  return [...channels];
}

function validateRate(maxFps) {
  if (!Number.isInteger(maxFps) || maxFps < 1 || maxFps > 120) {
    throw new Error("Remote frame rate must be an integer from 1 through 120");
  }
  return maxFps;
}

function samePublication(left, right) {
  return IDENTITY_FIELDS.every((field) => left.identity[field] === right.identity[field])
    && left.playback.paused === right.playback.paused
    && left.playback.speedMultiplier === right.playback.speedMultiplier;
}

function metadata(frame) {
  return { identity: { ...frame.identity }, playback: { ...frame.playback } };
}

/** Pure consumer cursor state, shared by the socket adapter and focused tests. */
export class RemoteFrameTracker {
  constructor(channels = CHANNELS) {
    this.channels = validateChannels(channels);
    this.sessionId = null;
    this.afterEvent = 0n;
    this.observed = new Map();
    this.freshSnapshots = new Set();
  }

  /** Revalidate each new connection without discarding its earlier source floors. */
  beginConnection(channels = this.channels) {
    this.freshSnapshots = new Set(validateChannels(channels).filter((channel) => channel !== "events"));
  }

  acknowledge(sessionId, channels = this.channels) {
    decimal(sessionId, 128);
    if (this.sessionId !== null && this.sessionId !== sessionId) {
      const error = new Error("The server session changed; explicitly connect to the new session to start a fresh view");
      error.code = "session_changed";
      throw error;
    }
    this.channels = validateChannels(channels);
    this.sessionId = sessionId;
  }

  resume(channels = this.channels) {
    if (this.sessionId === null) return undefined;
    const snapshots = validateChannels(channels).filter((channel) => channel !== "events");
    // One scalar resume cursor must not skip a canvas that arrived after its
    // sibling telemetry. An unseen channel has no cursor: revision zero itself
    // may still need delivering from a paused, freshly started host.
    const resume = { session_id: this.sessionId, after_event: String(this.afterEvent) };
    if (snapshots.length && snapshots.every((channel) => this.observed.has(channel) && !this.freshSnapshots.has(channel))) {
      const revisions = snapshots.map((channel) => BigInt(this.observed.get(channel).identity.publicationRevision));
      resume.after_revision = String(revisions.reduce((minimum, value) => value < minimum ? value : minimum));
    }
    return resume;
  }

  /** Deliver synchronously, then commit cursors. Refusal leaves all cursors unchanged. */
  accept(frame, consume) {
    const freshSnapshots = this.freshSnapshots;
    if (frame?.schema !== "scriptbots.remote-frame.v2" || !this.channels.includes(frame.kind)
        || this.sessionId === null || frame.identity?.sessionId !== this.sessionId) {
      throw new Error("Remote frame has an unnegotiated schema, channel or session");
    }
    for (const field of IDENTITY_FIELDS) decimal(frame.identity[field], field === "sessionId" ? 128 : 64);
    decimal(frame.coalescedPublications);
    if (typeof frame.playback?.paused !== "boolean" || !Number.isFinite(frame.playback.speedMultiplier)
        || frame.playback.speedMultiplier < 0) throw new Error("Invalid observed remote playback state");
    const revision = BigInt(frame.identity.publicationRevision);
    const previous = this.observed.get(frame.kind);
    if (previous) {
      const priorRevision = BigInt(previous.identity.publicationRevision);
      if (revision < priorRevision) throw new Error("Remote channel publication regressed");
      for (const field of ["scientificRevision", "controlRevision", "configRevision", "tick"]) {
        if (BigInt(frame.identity[field]) < BigInt(previous.identity[field])) {
          throw new Error("Remote source revision regressed within a session");
        }
      }
    }
    for (const observed of this.observed.values()) {
      if (observed.identity.publicationRevision === frame.identity.publicationRevision
          && !samePublication(observed, frame)) {
        throw new Error("Remote channels disagree about the same publication");
      }
    }
    let nextEvent = this.afterEvent;
    if (frame.kind === "events") {
      const page = frame.payload;
      const after = decimal(page?.afterEvent);
      const earliest = decimal(page.earliestRetained);
      const latest = decimal(page.latestEvent);
      if (after > latest || earliest === 0n || earliest > latest + 1n
          || !Array.isArray(page.events) || page.events.length > 64) {
        throw new Error("Invalid remote event retention bounds");
      }
      const missing = after + 1n < earliest;
      if (missing !== (page.gap != null) || (missing
          && (decimal(page.gap.firstMissing) !== after + 1n || decimal(page.gap.lastMissing) !== earliest - 1n))) {
        throw new Error("Remote event loss was not described by an exact gap");
      }
      let next = after + 1n > earliest ? after + 1n : earliest;
      for (const event of page.events) {
        if (decimal(event.sequence) !== next || next > latest || event.record?.schemaVersion !== 1
            || decimal(event.record.tick) > BigInt(frame.identity.tick)) {
          throw new Error("Remote event page is not a contiguous source-bound sequence");
        }
        next += 1n;
      }
      nextEvent = page.events.length ? next - 1n : (missing ? earliest - 1n : after);
      if (after < this.afterEvent && nextEvent <= this.afterEvent) return false;
      if (after !== this.afterEvent) throw new Error("Remote event page does not follow the consumed cursor");
    } else {
      if (!Number.isSafeInteger(frame.payload?.tick) || frame.payload.tick < 0
          || String(frame.payload.tick) !== frame.identity.tick) {
        throw new Error("Remote payload tick differs from its exact publication identity");
      }
      if (previous?.identity.publicationRevision === frame.identity.publicationRevision && !freshSnapshots.has(frame.kind)) return false;
    }
    const accepted = consume(frame);
    if (accepted && typeof accepted.then === "function") {
      throw new Error("Remote frame consumers must commit synchronously");
    }
    if (accepted === false) throw new Error("Remote frame consumer refused the publication");
    this.afterEvent = nextEvent;
    this.observed.set(frame.kind, metadata(frame));
    freshSnapshots.delete(frame.kind);
    return true;
  }
}

/** Return the server's single-frame credit only after synchronous consumption. */
export function consumeRemoteFrame(tracker, frame, consume, acknowledge) {
  const delivered = tracker.accept(frame, consume);
  const ack = { protocol_version: VERSION, command: "stream_ack",
    session_id: frame.identity.sessionId, channel: frame.kind,
    publication_revision: frame.identity.publicationRevision };
  if (frame.kind === "events") {
    const page = frame.payload;
    ack.after_event = page.events.at(-1)?.sequence ?? page.gap?.lastMissing ?? page.afterEvent;
  }
  acknowledge(ack);
  return delivered;
}

function uniqueRequestId(kind) {
  const random = new Uint8Array(16);
  if (!globalThis.crypto?.getRandomValues) throw new Error("Secure request identities are unavailable");
  globalThis.crypto.getRandomValues(random);
  return `web-${kind}-${Array.from(random, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function controlText(command, options) {
  const { speed, commandId } = options;
  const requestId = options.requestId ?? uniqueRequestId("control");
  if (typeof requestId !== "string" || requestId.trim() !== requestId || !requestId
      || encoder.encode(requestId).length > 1024) throw new Error("Invalid remote request identity");
  const request = { protocol_version: VERSION, request_id: requestId, command };
  if (["pause", "resume", "step"].includes(command)) {
    if (speed !== undefined || commandId !== undefined) throw new Error("Unrelated remote command fields");
  } else if (command === "speed") {
    if (!Number.isFinite(speed) || speed < 0 || commandId !== undefined) throw new Error("Invalid remote speed request");
    request.speed = speed;
  } else if (command === "status") {
    if (speed !== undefined || typeof commandId !== "string" || !commandId
        || encoder.encode(commandId).length > 128) throw new Error("Invalid remote status request");
    request.command_id = commandId;
  } else {
    throw new Error("Unknown remote control command");
  }
  return { requestId, text: JSON.stringify(request), commandId: command === "status" ? commandId : null };
}

/** Bounded browser WebSocket client. Reconnect and command retries are explicit. */
export class RemoteStreamClient {
  constructor({ url, decodeFrame, channels = CHANNELS, maxFps = 60,
    onFrame = () => {}, onState = () => {}, onReceipt = () => {} }) {
    const parsed = new URL(url, globalThis.location?.href);
    if (!["ws:", "wss:"].includes(parsed.protocol) || parsed.username || parsed.password) {
      throw new Error("Use a ws:// or wss:// endpoint without URL credentials");
    }
    if (typeof decodeFrame !== "function") throw new Error("The remote WASM decoder is unavailable");
    this.url = parsed.href;
    this.decodeFrame = decodeFrame;
    this.channels = validateChannels(channels);
    this.maxFps = validateRate(maxFps);
    this.onFrame = onFrame;
    this.onState = onState;
    this.onReceipt = onReceipt;
    this.tracker = new RemoteFrameTracker(this.channels);
    this.requests = new Map();
    this.socket = null;
    this.generation = 0;
    this.subscription = null;
    this.state = "disconnected";
  }

  get resume() { return this.tracker.resume(this.channels); }

  /** Read bounded scientific detail from the same host as the negotiated stream. */
  async inspectAgent(uid, { sessionId, epoch, signal } = {}) {
    return this.readInspection("inspect", MAX_INSPECTION_BYTES, uid, { sessionId, epoch, signal });
  }

  /** Explicit activity reads share the owner endpoint and never subscribe a background probe. */
  async inspectBrain(uid, { sessionId, epoch, signal } = {}) {
    return this.readInspection("brain", MAX_BRAIN_ACTIVITY_BYTES, uid, { sessionId, epoch, signal });
  }

  /** One explicit immutable genome page, independently bounded from activity reads. */
  async inspectGenome(uid, { sessionId, epoch, locusOffset = 0, deltaOffset = 0, signal } = {}) {
    if (!Number.isSafeInteger(locusOffset) || locusOffset < 0 || locusOffset > 65536 || locusOffset % 64 !== 0
        || !Number.isSafeInteger(deltaOffset) || deltaOffset < 0 || deltaOffset > 65536 || deltaOffset % 32 !== 0) {
      throw new Error("Invalid bounded genome page offsets");
    }
    return this.readInspection("genome", 65536, uid, { sessionId, epoch, signal,
      query: { locus_offset: String(locusOffset), delta_offset: String(deltaOffset) } });
  }

  /** Stage or explicitly retry the same source-bound request through the canonical owner. */
  async intervene(request, { signal } = {}) {
    const url = this.apiUrl("/api/control/browser-intervention", request?.sessionId);
    const body = JSON.stringify(request);
    if (typeof body !== "string" || encoder.encode(body).length > 4096) throw new Error("Intervention request exceeds its 4 KiB bound");
    return boundedOwnerJson(url, 16 * 1024, { signal, method: "POST", body });
  }

  async interventionStatus(commandId, { sessionId, signal } = {}) {
    if (typeof commandId !== "string" || !/^[0-9a-f]{32}$/.test(commandId)) throw new Error("Invalid exact intervention command identity");
    const url = this.apiUrl(`/api/control/browser-intervention/${commandId}`, sessionId, { session_id: sessionId });
    return boundedOwnerJson(url, 16 * 1024, { signal, missingMessage: "no retained command receipt; admission remains unknown" });
  }

  async interventionHistory({ sessionId, epoch, afterSeq, limit = 32, signal } = {}) {
    decimal(epoch); decimal(afterSeq);
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 64) throw new Error("Invalid bounded intervention history page size");
    const url = this.apiUrl("/api/interventions/browser", sessionId,
      { session_id: sessionId, epoch, after_seq: afterSeq, limit: String(limit) });
    return boundedOwnerJson(url, 65536, { signal });
  }

  /** Every Lab read/write is bound to the negotiated owner, including at the HTTP server. */
  async labJson(path, sessionId, byteLimit, { signal, body, method = "GET", query = {} } = {}) {
    const url = this.apiUrl(path, sessionId, query);
    const generation = this.generation;
    const value = await boundedOwnerJson(url, byteLimit, { signal, body, method, sessionId, timeoutMs: 15000 });
    this.requireLabSource(sessionId, generation);
    return value;
  }

  requireLabSource(sessionId, generation) {
    if (this.state !== "connected" || this.tracker.sessionId !== sessionId || this.generation !== generation) {
      throw new Error("The remote connection changed while the Lab request was pending; its outcome is unconfirmed");
    }
  }

  async listCheckpoints({ sessionId, cursor = null, limit = 12, signal } = {}) {
    return this.labJson("/api/v1/checkpoints", sessionId, 128 * 1024,
      { signal, query: labPageQuery(cursor, limit) });
  }

  async listCheckpointBranches({ sessionId, cursor = null, limit = 12, signal } = {}) {
    return this.labJson("/api/v1/checkpoints/branches", sessionId, 128 * 1024,
      { signal, query: labPageQuery(cursor, limit) });
  }

  async saveCheckpoint({ sessionId, description, idempotency_key }, { signal } = {}) {
    if (typeof idempotency_key !== "string" || !idempotency_key.trim() || encoder.encode(idempotency_key).length > 1024
        || (description != null && (typeof description !== "string" || encoder.encode(description).length > 4096))) {
      throw new Error("Invalid bounded checkpoint save or retry identity");
    }
    const body = JSON.stringify({ description, idempotency_key });
    return this.labJson("/api/v1/checkpoints", sessionId, 16 * 1024, { signal, method: "POST", body });
  }

  async launchCheckpointBranch(request, { signal } = {}) {
    const { sessionId, ...payload } = request;
    labIdentifier(payload.checkpoint_id);
    if (typeof payload.branch_name !== "string" || !/^[A-Za-z0-9_-]{1,64}$/.test(payload.branch_name)
        || !Number.isSafeInteger(payload.ticks) || payload.ticks < 1 || payload.ticks > 10000
        || typeof payload.idempotency_key !== "string" || !payload.idempotency_key.trim()
        || encoder.encode(payload.idempotency_key).length > 1024) throw new Error("Invalid bounded checkpoint study request");
    validateLabIntervention(payload.intervention ?? null);
    const body = JSON.stringify(payload);
    if (encoder.encode(body).length > 16 * 1024) throw new Error("Checkpoint study request exceeds 16 KiB");
    return this.labJson("/api/v1/checkpoints/branches", sessionId, 16 * 1024, { signal, method: "POST", body });
  }

  async checkpointBranchStatus(branchId, { sessionId, signal } = {}) {
    return this.labJson(`/api/v1/checkpoints/branches/${labIdentifier(branchId)}`, sessionId, 16 * 1024, { signal });
  }

  async cancelCheckpointBranch(branchId, { sessionId, signal } = {}) {
    return this.labJson(`/api/v1/checkpoints/branches/${labIdentifier(branchId)}/cancel`, sessionId, 16 * 1024,
      { signal, method: "POST" });
  }

  async checkpointBranchArtifacts(branchId, { sessionId, cursor = null, limit = 12, signal } = {}) {
    return this.labJson(`/api/v1/checkpoints/branches/${labIdentifier(branchId)}/artifacts`, sessionId, 128 * 1024,
      { signal, query: labPageQuery(cursor, limit) });
  }

  /** Download only an actual catalog entry, preserving its filename and verifying byte integrity. */
  async downloadArtifact(artifactId, { sessionId, signal, expected = null, maxBytes = MAX_LAB_ARTIFACT_BYTES } = {}) {
    labIdentifier(artifactId);
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > MAX_LAB_ARTIFACT_BYTES) {
      throw new Error("Invalid bounded artifact download size");
    }
    const generation = this.generation;
    const metadata = await this.labJson(`/api/v1/artifacts/${artifactId}`, sessionId, 16 * 1024, { signal });
    validateLabArtifact(metadata);
    if (metadata.byte_size > maxBytes) {
      throw new Error(`Artifact exceeds the ${maxBytes} byte browser download limit; the saved source remains available for server studies`);
    }
    if (metadata.artifact_id !== artifactId || (expected && (
      (expected.artifact_id ?? expected.checkpoint_id) !== artifactId
      || ["byte_size", "checksum_blake3", "checksum_sha256"].some((key) => metadata[key] !== expected[key])))) {
      throw new Error("Artifact catalog entry differs from the selected saved source");
    }
    const url = this.apiUrl(`/api/v1/artifacts/${artifactId}/download`, sessionId);
    const blob = await downloadLabBytes(url, metadata, sessionId, signal);
    this.requireLabSource(sessionId, generation);
    return { blob, filename: metadata.filename, metadata: Object.freeze(metadata) };
  }

  apiUrl(path, sessionId, query = {}) {
    decimal(sessionId);
    if (this.state !== "connected" || this.tracker.sessionId !== sessionId) {
      throw new Error("Connect to the observed world’s host before reading or submitting an intervention");
    }
    const url = new URL(this.url);
    if (!url.pathname.endsWith("/api/ws/stream")) throw new Error("Owner APIs require the host’s /api/ws/stream endpoint");
    url.protocol = url.protocol === "wss:" ? "https:" : "http:";
    url.pathname = `${url.pathname.slice(0, -"/api/ws/stream".length)}${path}`;
    url.search = new URLSearchParams(query).toString();
    url.hash = "";
    return url;
  }

  async readInspection(endpoint, byteLimit, uid, { sessionId, epoch, signal, query = {} }) {
    decimal(uid);
    decimal(sessionId, 128);
    decimal(epoch);
    if (this.state !== "connected" || this.tracker.sessionId !== sessionId) {
      throw new Error("Connect to the selected creature's host before inspecting it");
    }
    const url = new URL(this.url);
    if (!url.pathname.endsWith("/api/ws/stream")) {
      throw new Error("The inspector requires the host's /api/ws/stream endpoint");
    }
    url.protocol = url.protocol === "wss:" ? "https:" : "http:";
    url.pathname = `${url.pathname.slice(0, -"/api/ws/stream".length)}/api/agents/${endpoint}`;
    url.search = new URLSearchParams({ ...query, uid, session_id: sessionId, epoch }).toString();
    url.hash = "";
    return boundedOwnerJson(url, byteLimit, { signal,
      missingMessage: endpoint === "inspect" ? "creature is no longer present in this host" : undefined });
  }

  report(state, message, extra = {}) {
    this.state = state;
    this.onState({ state, message, sessionId: this.tracker.sessionId, ...extra });
  }

  connect() {
    if (this.socket) throw new Error("A remote connection is already active");
    this.tracker.beginConnection(this.channels);
    const generation = ++this.generation;
    this.report("connecting", "Connecting to the remote host…");
    let socket;
    try {
      socket = new WebSocket(this.url);
    } catch (error) {
      this.fail(error);
      return;
    }
    this.socket = socket;
    socket.binaryType = "arraybuffer";
    const current = () => this.socket === socket && this.generation === generation;
    socket.addEventListener("open", () => {
      if (!current()) return;
      try { this.subscribe(); } catch (error) { this.fail(error); }
    });
    socket.addEventListener("message", (event) => {
      if (!current()) return;
      try { this.receive(event.data); } catch (error) { this.fail(error); }
    });
    socket.addEventListener("error", () => {
      if (current()) this.fail(new Error("Connection failed; check the endpoint and the server's allowed browser Origin"));
    });
    socket.addEventListener("close", (event) => {
      if (!current()) return;
      this.socket = null;
      this.subscription = null;
      this.report("disconnected", `Remote connection closed (${event.code}); reconnect explicitly to continue`);
    });
  }

  disconnect() {
    const socket = this.socket;
    this.socket = null;
    this.subscription = null;
    this.generation += 1;
    if (socket) socket.close(1000, "Client disconnected");
    this.report("disconnected", "Remote connection disconnected");
  }

  reconnect() {
    this.disconnect();
    this.connect();
  }

  setSubscriptions({ channels = this.channels, maxFps = this.maxFps } = {}) {
    if (this.subscription) throw new Error("A subscription acknowledgement is still pending");
    const checkedChannels = validateChannels(channels);
    const checkedRate = validateRate(maxFps);
    this.channels = checkedChannels;
    this.maxFps = checkedRate;
    // A replacement connection discards the old channel credits safely. Sending
    // a new mask on a busy socket could race with a frame still in transit.
    if (this.socket) this.reconnect();
  }

  subscribe() {
    const requestId = uniqueRequestId("subscribe");
    const request = { protocol_version: VERSION, request_id: requestId, command: "subscribe",
      channels: this.channels, max_fps: this.maxFps };
    const resume = this.resume;
    if (resume) request.resume = resume;
    this.sendText(JSON.stringify(request));
    this.subscription = request;
    this.report("subscribing", "Waiting for the remote stream subscription…");
  }

  sendText(text) {
    const bytes = encoder.encode(text).length;
    if (bytes > MAX_CONTROL_BYTES) throw new Error("Remote control request exceeds 16 KiB");
    if (!this.socket || this.socket.readyState !== 1) throw new Error("Remote socket is not open");
    if (this.socket.bufferedAmount + bytes > MAX_SEND_BUFFER_BYTES) {
      throw new Error("Remote send buffer is full; wait before retrying this request");
    }
    this.socket.send(text);
  }

  sendControl(command, options = {}) {
    if (this.state !== "connected") throw new Error("Wait for the remote subscription before sending controls");
    const request = controlText(command, options);
    const existing = this.requests.get(request.requestId);
    if (existing && existing.text !== request.text) throw new Error("A request identity cannot be reused for a different command");
    if (!existing && this.requests.size === MAX_REQUESTS) {
      const settled = [...this.requests].find(([, item]) => item.received);
      if (!settled) throw new Error("32 control acknowledgements are pending; inspect or retry an existing request");
      this.requests.delete(settled[0]);
    }
    // Retain the exact key and payload even when browser send throws: an
    // explicit retry must never allocate another Step identity.
    this.requests.set(request.requestId, existing ?? { ...request, received: false });
    try {
      this.sendText(request.text);
    } catch (error) {
      error.requestId = request.requestId;
      throw error;
    }
    return request.requestId;
  }

  retryControl(requestId) {
    if (this.state !== "connected") throw new Error("Reconnect and subscribe before retrying a control request");
    const request = this.requests.get(requestId);
    if (!request) throw new Error("The exact request is no longer retained; it cannot be retried safely");
    this.sendText(request.text);
    return requestId;
  }

  receive(data) {
    if (typeof data !== "string") {
      // Existing servers can send a legacy TickSummary before processing the
      // subscription. Its schema is never guessed or decoded as the new wire.
      if (this.state !== "connected") return;
      if (!(data instanceof ArrayBuffer) || data.byteLength > MAX_FRAME_BYTES) {
        throw new Error("Remote binary frame exceeds its size or type bound");
      }
      const frame = this.decodeFrame(new Uint8Array(data));
      const socket = this.socket;
      const generation = this.generation;
      consumeRemoteFrame(this.tracker, frame, this.onFrame, (ack) => {
        // A consumer may explicitly disconnect while presenting the frame. Its
        // credit belongs to this connection, never a replacement socket.
        if (this.socket === socket && this.generation === generation) this.sendText(JSON.stringify(ack));
      });
      return;
    }
    if (encoder.encode(data).length > MAX_CONTROL_BYTES) throw new Error("Remote reply exceeds 16 KiB");
    const reply = JSON.parse(data);
    if (reply.protocol_version !== VERSION) throw new Error("Unsupported remote control protocol version");
    if (reply.type === "subscribed") {
      const pending = this.subscription;
      if (!pending || reply.request_id !== pending.request_id || reply.stream_protocol_version !== STREAM_VERSION
          || reply.max_fps !== pending.max_fps
          || JSON.stringify(validateChannels(reply.channels).sort()) !== JSON.stringify([...pending.channels].sort())) {
        throw new Error("Remote subscription acknowledgement does not match the request");
      }
      this.tracker.acknowledge(reply.session_id, pending.channels);
      this.subscription = null;
      this.report("connected", "Remote stream connected", { channels: [...pending.channels], maxFps: reply.max_fps });
      return;
    }
    if (reply.type === "stream_rate") {
      // An observation for the previous subscription can precede the matching
      // replacement acknowledgement on the same ordered socket.
      if (this.state === "subscribing") return;
      if (this.state !== "connected" || !["slow_consumer", "recovering"].includes(reply.reason)
          || !Number.isSafeInteger(reply.requested_interval_us) || reply.requested_interval_us < 1
          || !Number.isSafeInteger(reply.effective_interval_us)
          || reply.effective_interval_us < reply.requested_interval_us) {
        throw new Error("Invalid remote stream-rate observation");
      }
      this.report("connected", `Remote stream interval: ${(reply.effective_interval_us / 1000).toFixed(1)} ms (${reply.reason})`,
        { requestedIntervalUs: reply.requested_interval_us, effectiveIntervalUs: reply.effective_interval_us, reason: reply.reason });
      return;
    }
    if (reply.type !== "receipt" && reply.type !== "error") throw new Error("Unknown remote reply type");
    if (reply.type === "error" && reply.request_id === this.subscription?.request_id) {
      throw new Error(`Remote subscription refused: ${reply.message}`);
    }
    const request = this.requests.get(reply.request_id);
    if (!request) throw new Error(`Uncorrelated remote reply: ${reply.message ?? reply.type}`);
    if (reply.type === "receipt") {
      const commandId = reply.receipt?.command_id;
      if (typeof commandId !== "string" || !/^[0-9a-f]{32}$/.test(commandId)
          || (request.commandId && request.commandId !== commandId)) {
        throw new Error("Remote receipt changed its canonical command identity");
      }
      request.commandId = commandId;
    }
    request.received = true;
    this.onReceipt(reply);
  }

  fail(error) {
    const socket = this.socket;
    this.socket = null;
    this.subscription = null;
    this.generation += 1;
    if (socket) socket.close(1002, "Remote protocol refused");
    this.report(error.code === "session_changed" ? "session_changed" : "error", error.message ?? String(error));
  }
}
