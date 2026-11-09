// Remote presentation never owns or drives a local SimHandle. Binary decoding is
// supplied by the shared Rust/WASM protocol; this module owns connection and
// consumer cursors, with no queue of historical canvas frames.

const VERSION = 1;
const STREAM_VERSION = 2;
const MAX_FRAME_BYTES = 8 * 1024 * 1024;
const MAX_CONTROL_BYTES = 16 * 1024;
const MAX_SEND_BUFFER_BYTES = 64 * 1024;
const MAX_REQUESTS = 32;
const CHANNELS = ["canvas", "telemetry", "events"];
const IDENTITY_FIELDS = ["sessionId", "publicationRevision", "scientificRevision",
  "controlRevision", "configRevision", "tick"];
const encoder = new TextEncoder();

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
    if (snapshots.length && snapshots.every((channel) => this.observed.has(channel))) {
      const revisions = snapshots.map((channel) => BigInt(this.observed.get(channel).identity.publicationRevision));
      resume.after_revision = String(revisions.reduce((minimum, value) => value < minimum ? value : minimum));
    }
    return resume;
  }

  /** Deliver synchronously, then commit cursors. Refusal leaves all cursors unchanged. */
  accept(frame, consume) {
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
      if (previous?.identity.publicationRevision === frame.identity.publicationRevision) return false;
    }
    const accepted = consume(frame);
    if (accepted && typeof accepted.then === "function") {
      throw new Error("Remote frame consumers must commit synchronously");
    }
    if (accepted === false) throw new Error("Remote frame consumer refused the publication");
    this.afterEvent = nextEvent;
    this.observed.set(frame.kind, metadata(frame));
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

  report(state, message, extra = {}) {
    this.state = state;
    this.onState({ state, message, sessionId: this.tracker.sessionId, ...extra });
  }

  connect() {
    if (this.socket) throw new Error("A remote connection is already active");
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
