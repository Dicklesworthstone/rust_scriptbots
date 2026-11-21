import { RemoteFrameTracker } from "./remote.js";

// Narrative reads consume the same canonical pages as the remote stream. They
// never submit controls, drive the owner clock, or reconstruct detector prose.
export const NARRATIVE_ROW_LIMIT = 48;
const REFRESH_MS = 250;
const IDENTITY_FIELDS = ["sessionId", "publicationRevision", "scientificRevision",
  "controlRevision", "configRevision", "tick"];
const encoder = new TextEncoder();

function decimal(value, bits = 64) {
  if (typeof value !== "string" || value.length > (bits === 128 ? 39 : 20)
      || !/^(0|[1-9][0-9]*)$/.test(value) || BigInt(value) >= (1n << BigInt(bits))) {
    throw new Error("Narrative identities must be exact canonical decimal strings");
  }
  return BigInt(value);
}

function identity(value) {
  for (const field of IDENTITY_FIELDS) decimal(value?.[field], field === "sessionId" ? 128 : 64);
  return { ...value };
}

function boundedText(value, maximum) {
  return typeof value === "string" && value.length <= maximum && encoder.encode(value).length <= maximum;
}

/** Validate the scientific values rendered below, in addition to stream cursors. */
export function validateNarrativePage(frame) {
  if (frame?.schema !== "scriptbots.remote-frame.v2" || frame.kind !== "events") {
    throw new Error("Unsupported scientific narrative page");
  }
  identity(frame.identity);
  const page = frame.payload;
  if (!Array.isArray(page?.events) || page.events.length > 64) throw new Error("Narrative page exceeds its record bound");
  const after = decimal(page.afterEvent), earliest = decimal(page.earliestRetained), latest = decimal(page.latestEvent);
  const start = after + 1n > earliest ? after + 1n : earliest;
  const remaining = latest >= start ? latest - start + 1n : 0n;
  if (page.events.length !== Number(remaining > 64n ? 64n : remaining)) {
    throw new Error("Narrative page omits records before its declared latest event");
  }
  for (const { record } of page.events) {
    if (!record || record.schemaVersion !== 1 || !boundedText(record.humanText, 4096)
        || !boundedText(record.metric, 256) || !boundedText(record.kind, 64)
        || (record.subject !== null && !boundedText(record.subject, 256))
        || ![record.before, record.after, record.magnitude, record.score, record.severity].every(Number.isFinite)
        || record.severity < 0 || record.severity > 1 || !Array.isArray(record.window) || record.window.length !== 2
        || decimal(record.window[0]) > decimal(record.window[1]) || decimal(record.window[1]) > decimal(frame.identity.tick)
        || decimal(record.tick) > decimal(frame.identity.tick)) {
      throw new Error("Narrative record has invalid scientific values or text bounds");
    }
  }
  return frame;
}

/** One bounded timeline for the active owner; reconnect keeps its consumed cursor. */
export class NarrativeController {
  constructor({ read, onChange = () => {}, onFrame = () => {}, now = () => performance.now() } = {}) {
    this.read = read;
    this.onChange = onChange;
    this.onFrame = onFrame;
    this.now = now;
    this.enabled = true;
    this.pending = null;
    this.token = 0;
    this.reset();
  }

  reset(mode = "local") {
    this.token += 1;
    this.mode = mode;
    this.ownerGeneration = null;
    this.sessionId = null;
    this.frameIdentity = null;
    this.pageIdentity = null;
    this.tracker = new RemoteFrameTracker(["events"]);
    this.rows = [];
    this.earliestRetained = "1";
    this.latestEvent = "0";
    this.lastReadRevision = null;
    this.lastStarted = -Infinity;
    this.lastGap = null;
    this.missingEvents = 0n;
    this.omittedRows = 0n;
    this.error = null;
    this.connection = mode === "remote" ? "Waiting for server history." : null;
    this.onChange(this);
  }

  get afterEvent() { return String(this.tracker.afterEvent); }
  get hasMore() { return this.tracker.afterEvent < BigInt(this.latestEvent); }

  setEnabled(enabled) {
    if (this.enabled === Boolean(enabled)) return;
    this.enabled = Boolean(enabled);
    this.onChange(this);
  }

  setConnection(message) {
    this.connection = message;
    this.onChange(this);
  }

  /** Observe a canonical publication; extra presentation frames do not add reads. */
  observe(frame, generation) {
    const observed = identity(frame?.identity);
    if ((this.ownerGeneration !== null && this.ownerGeneration !== generation)
        || (this.sessionId !== null && this.sessionId !== observed.sessionId)) this.reset(this.mode);
    this.ownerGeneration = generation;
    if (this.sessionId === null) {
      this.tracker.acknowledge(observed.sessionId);
      this.sessionId = observed.sessionId;
    }
    if (this.frameIdentity && BigInt(observed.publicationRevision) < BigInt(this.frameIdentity.publicationRevision)) return;
    const changed = this.frameIdentity?.publicationRevision !== observed.publicationRevision;
    this.frameIdentity = observed;
    if (changed) this.onFrame(this);
    if (this.mode === "local") void this.refresh();
  }

  /** Consume synchronously before the socket returns its per-channel credit. */
  accept(frame) {
    validateNarrativePage(frame);
    const tracker = this.sessionId === null ? new RemoteFrameTracker(["events"]) : this.tracker;
    if (this.sessionId === null) tracker.acknowledge(frame.identity.sessionId);
    const accepted = tracker.accept(frame, (value) => {
      const page = value.payload;
      if (page.gap) {
        this.lastGap = { ...page.gap };
        this.missingEvents += BigInt(page.gap.lastMissing) - BigInt(page.gap.firstMissing) + 1n;
      }
      this.rows.push(...page.events.map((event) => ({ sequence: event.sequence,
        record: { ...event.record, window: [...event.record.window] } })));
      const excess = Math.max(0, this.rows.length - NARRATIVE_ROW_LIMIT);
      if (excess) {
        this.rows.splice(0, excess);
        this.omittedRows += BigInt(excess);
      }
      this.earliestRetained = page.earliestRetained;
      this.latestEvent = page.latestEvent;
      this.pageIdentity = identity(value.identity);
      this.lastReadRevision = value.identity.publicationRevision;
      this.error = null;
    });
    if (accepted) {
      this.tracker = tracker;
      this.sessionId = frame.identity.sessionId;
      this.onChange(this);
    }
    return accepted;
  }

  /** At most one page per 250 ms, including a paused owner's retained backlog. */
  async refresh(force = false) {
    if (this.mode !== "local" || !this.enabled || !this.frameIdentity || this.pending
        || this.now() - this.lastStarted < REFRESH_MS) return false;
    const alreadyRead = this.lastReadRevision !== null
      && BigInt(this.lastReadRevision) >= BigInt(this.frameIdentity.publicationRevision);
    if (!force && alreadyRead && (!this.hasMore || this.error)) return false;
    const request = { token: this.token, afterEvent: this.afterEvent, identity: { ...this.frameIdentity } };
    this.pending = request;
    this.lastStarted = this.now();
    this.error = null;
    this.onChange(this);
    try {
      if (typeof this.read !== "function") throw new Error("This browser package does not expose scientific narrative history");
      const page = await this.read(request.afterEvent, request.identity.sessionId);
      if (request.token !== this.token) return false;
      for (const field of IDENTITY_FIELDS.slice(1)) {
        if (decimal(page?.identity?.[field]) < decimal(request.identity[field])) {
          throw new Error("Narrative response is older than the requested owner publication");
        }
      }
      return this.accept(page);
    } catch (error) {
      if (request.token !== this.token) return false;
      this.error = String(error?.message || error).slice(0, 512);
      this.lastReadRevision = request.identity.publicationRevision;
      this.onChange(this);
      return false;
    } finally {
      if (this.pending === request) this.pending = null;
      if (request.token === this.token) this.onChange(this);
    }
  }
}

function setText(element, text) { if (element && element.textContent !== text) element.textContent = text; }

export function renderNarrativeSource(panel, controller) {
  if (!panel) return;
  const page = controller.pageIdentity, world = controller.frameIdentity;
  let text = page ? `${controller.mode === "local" ? "Local" : "Server"} history · session ${page.sessionId} · observed tick ${page.tick}, publication ${page.publicationRevision}.` : "Waiting for a scientific narrative observation.";
  if (page && world && page.publicationRevision !== world.publicationRevision) {
    text += ` Latest world observation: tick ${world.tick}, publication ${world.publicationRevision}.`;
  }
  setText(panel.querySelector("#narrative-source"), text);
}

/** Canonical prose is inserted only as text; empty history stays an empty list. */
export function renderNarrative(panel, controller) {
  if (!panel) return;
  const state = controller.error ? "error" : controller.connection ? "disconnected"
    : controller.pending ? "reading" : controller.pageIdentity ? "ready" : "waiting";
  panel.dataset.state = state;
  panel.dataset.mode = controller.mode;
  panel.dataset.afterEvent = controller.afterEvent;
  panel.dataset.sessionId = controller.sessionId ?? "";
  let message = controller.error ? `History unavailable: ${controller.error}. Refresh to retry.`
    : controller.connection || (controller.pending ? "Reading scientific history…"
      : controller.hasMore ? "Reading more retained events…"
        : controller.pageIdentity ? controller.rows.length ? `${controller.rows.length} recent scientific events shown.`
          : "No scientific narrative events have been observed in the retained history."
          : "Waiting for scientific history.");
  if (!controller.enabled && controller.mode === "local") message += " Local reads pause while the page is hidden.";
  setText(panel.querySelector("#narrative-status"), message);
  renderNarrativeSource(panel, controller);
  let retention = controller.mode === "local"
    ? "Local history is retained in memory and saved continuations; this view does not confirm durable journal storage."
    : "Server-retained history; this view does not confirm durable journal storage.";
  if (controller.pageIdentity) retention += BigInt(controller.earliestRetained) > BigInt(controller.latestEvent)
    ? ` Source retains no events; latest ordinal #${controller.latestEvent}; consumed through #${controller.afterEvent}.`
    : ` Source retains #${controller.earliestRetained}–#${controller.latestEvent}; consumed through #${controller.afterEvent}.`;
  if (controller.omittedRows > 0n) retention += ` ${controller.omittedRows} older viewed events are outside this ${NARRATIVE_ROW_LIMIT}-row display.`;
  setText(panel.querySelector("#narrative-retention"), retention);
  setText(panel.querySelector("#narrative-gap"), controller.lastGap
    ? `History gap: narrative events ${controller.lastGap.firstMissing}–${controller.lastGap.lastMissing} are no longer retained by this ${controller.mode === "local" ? "owner" : "server"}. ${controller.missingEvents} requested events unavailable in this view.` : "");
  const refresh = panel.querySelector("#narrative-refresh");
  if (refresh) refresh.disabled = controller.mode !== "local" || !controller.enabled || !controller.frameIdentity || !!controller.pending;
  const list = panel.querySelector("#remote-events");
  if (!list) return;
  const ownerKey = `${controller.mode}/${controller.sessionId}/${controller.token}`;
  const previousLast = list.lastElementChild?.dataset.sequence;
  // No event nodes are rebuilt for a repeated paused observation or read status.
  if (list.dataset.narrativeOwner === ownerKey && list.children.length === controller.rows.length
      && previousLast === controller.rows.at(-1)?.sequence) return;
  const follow = list.scrollHeight - list.scrollTop - list.clientHeight < 24;
  list.replaceChildren();
  list.dataset.narrativeOwner = ownerKey;
  for (const event of controller.rows) {
    const item = panel.ownerDocument.createElement("li");
    item.dataset.sequence = event.sequence;
    item.textContent = `#${event.sequence} · tick ${event.record.tick}: ${event.record.humanText}`;
    item.title = `${event.record.kind} · ${event.record.metric}: ${event.record.before} → ${event.record.after} · window ${event.record.window[0]}–${event.record.window[1]}`;
    list.append(item);
  }
  if (follow) list.scrollTop = list.scrollHeight;
}
