import { exactIdentity } from "./inspector.js";

const COMMAND_ID = /^[0-9a-f]{32}$/;
const SOURCE_FIELDS = ["epoch", "publicationRevision", "scientificRevision", "controlRevision", "configRevision", "tick"];
const EFFECT_KINDS = new Set(["drought", "embargo", "bloom", "meteor", "paint_terrain", "set_closed_world",
  "inject_cohort", "spawn_from_archive", "expiry:drought", "expiry:embargo"]);
const scalar = (value) => typeof value === "number" && Number.isFinite(value) && Number.isFinite(Math.fround(value));
const unit = (value) => scalar(value) && value >= 0 && value <= 1;
const exactKeys = (value, keys) => value && typeof value === "object" && !Array.isArray(value)
  && Object.keys(value).length === keys.length && keys.every((key) => Object.hasOwn(value, key));

function validateRegion(region) {
  if (region?.shape === "all" && exactKeys(region, ["shape"])) return;
  if (region?.shape === "disc" && exactKeys(region, ["shape", "x", "y", "radius"])
      && [region.x, region.y, region.radius].every(scalar) && region.radius > 0) return;
  if (region?.shape === "rect" && exactKeys(region, ["shape", "x", "y", "w", "h"])
      && [region.x, region.y, region.w, region.h].every(scalar) && region.w > 0 && region.h > 0) return;
  throw new Error("Choose a finite region with positive dimensions");
}

/** The toolbox only constructs canonical scalar ecological actions. */
export function validateInterventionAction(action) {
  if (action?.kind === "set_closed_world") {
    if (!exactKeys(action, ["kind", "closed"]) || typeof action.closed !== "boolean") throw new Error("Choose closed or open world mode");
    return action;
  }
  validateRegion(action?.region);
  const keys = { drought: ["ticks", "growth_scale"], embargo: ["ticks"], bloom: ["amount"], meteor: ["lethality", "scorch"] }[action.kind];
  if (!keys || !exactKeys(action, ["kind", "region", ...keys])) throw new Error("Unsupported intervention or unexpected action fields");
  if (keys.includes("ticks") && (!Number.isSafeInteger(action.ticks) || action.ticks < 1 || action.ticks > 0xffff_ffff)) {
    throw new Error("Duration must be an integer from 1 through 4294967295 ticks");
  }
  if (action.kind === "drought" && !unit(action.growth_scale)) throw new Error("Growth scale must be between 0 and 1");
  if (action.kind === "bloom" && (!scalar(action.amount) || action.amount < 0)) throw new Error("Food added per cell must be finite and nonnegative");
  if (action.kind === "meteor" && (!scalar(action.lethality) || action.lethality < 0 || !unit(action.scorch))) {
    throw new Error("Health removed must be finite and nonnegative; scorch must be between 0 and 1");
  }
  return action;
}

export function interventionRequestId() {
  if (!globalThis.crypto?.getRandomValues) throw new Error("Secure random request identities are unavailable in this browser");
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return `web-intervention-${Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

/** Freeze exactly the displayed paused boundary and payload for every explicit retry. */
export function makeInterventionRequest(frame, action, requestId = interventionRequestId()) {
  if (!frame || frame.playback?.paused !== true) throw new Error("Pause the simulation and wait for its observed paused frame before staging an intervention");
  if (typeof requestId !== "string" || !/^[A-Za-z0-9_.:-]{1,128}$/.test(requestId)) throw new Error("Invalid intervention retry identity");
  validateInterventionAction(action);
  const intervention = structuredClone(action);
  if (intervention.region) Object.freeze(intervention.region);
  Object.freeze(intervention);
  return Object.freeze({ requestId, sessionId: exactIdentity(frame.identity.sessionId),
    epoch: exactIdentity(frame.snapshot.epoch), expectedControlRevision: exactIdentity(frame.identity.controlRevision), intervention });
}

function validateSource(source, request, { sameEpoch = true } = {}) {
  if (!source || source.sessionId !== request.sessionId || (sameEpoch && source.epoch !== request.epoch)
      || typeof source.paused !== "boolean") throw new Error("Intervention response belongs to another owner or epoch");
  exactIdentity(source.sessionId);
  for (const key of SOURCE_FIELDS) exactIdentity(source[key]);
  if (BigInt(source.epoch) < BigInt(request.epoch)) throw new Error("Intervention source epoch regressed");
  if (request.identity && ["tick", "publicationRevision", "scientificRevision", "controlRevision", "configRevision"]
    .some((key) => BigInt(source[key]) < BigInt(request.identity[key]))) throw new Error("Intervention observation predates its requested source");
}

/** Admission, queue application, and journal commitment remain independent observations. */
export function validateInterventionReceipt(receipt, request, previous = null) {
  if (!receipt || receipt.schema !== "scriptbots.browser-intervention-receipt.v1"
      || !COMMAND_ID.test(receipt.commandId) || (previous && receipt.commandId !== previous.commandId)
      || !["admitted", "applied", "rejected", "failed"].includes(receipt.applicationState)
      || !["not_required", "pending", "committed_volatile", "durable", "failed"].includes(receipt.journalState)
      || typeof receipt.messageTruncated !== "boolean") throw new Error("Invalid intervention command receipt");
  validateSource(receipt.source, request, { sameEpoch: false });
  if (receipt.admissionSequence !== null) exactIdentity(receipt.admissionSequence);
  if (receipt.applicationState !== "rejected" && receipt.admissionSequence === null) throw new Error("Admitted intervention lacks its admission sequence");
  if (receipt.applicationState === "admitted" && !["not_required", "pending"].includes(receipt.journalState)) {
    throw new Error("Unapplied intervention invented a completed lifecycle journal");
  }
  for (const text of [receipt.message, receipt.ownerError]) {
    if (text !== null && (typeof text !== "string" || [...text].length > 1024)) throw new Error("Unbounded intervention diagnostic");
  }
  if (receipt.applicationState === "applied") {
    if (!receipt.applicationBoundary || receipt.admissionSequence === null) throw new Error("Applied intervention lacks its observed queue boundary");
    for (const key of ["tick", "controlRevision", "scientificRevision", "configRevision"]) {
      exactIdentity(receipt.applicationBoundary[key]);
      if (BigInt(receipt.applicationBoundary[key]) > BigInt(receipt.source[key])) throw new Error("Intervention queue boundary follows the owner observation");
    }
  } else if (receipt.applicationBoundary !== null) throw new Error("Unapplied intervention invented a queue boundary");
  if (previous) {
    if (["epoch", "tick", "publicationRevision", "scientificRevision", "controlRevision", "configRevision"]
      .some((key) => BigInt(receipt.source[key]) < BigInt(previous.source[key]))
        || (previous.applicationState !== "admitted" && receipt.applicationState !== previous.applicationState)
        || receipt.admissionSequence !== previous.admissionSequence
        || (previous.applicationBoundary && JSON.stringify(receipt.applicationBoundary) !== JSON.stringify(previous.applicationBoundary))) {
      throw new Error("Intervention lifecycle observation regressed or changed its identity");
    }
    if ((["durable", "failed"].includes(previous.journalState) && receipt.journalState !== previous.journalState)
        || (previous.journalState === "committed_volatile" && !["committed_volatile", "durable", "failed"].includes(receipt.journalState))) {
      throw new Error("Intervention journal observation regressed");
    }
  }
  return receipt;
}

/** Verify the whole page, including an exact missing prefix, before accepting a cursor. */
export function validateInterventionHistory(page, request) {
  if (!page || page.schema !== "scriptbots.browser-intervention-history.v1" || typeof page.closed !== "boolean"
      || typeof page.hasMore !== "boolean" || page.afterSeq !== request.afterSeq
      || !Array.isArray(page.records) || page.records.length > request.limit || page.records.length > 64) {
    throw new Error("Invalid bounded intervention history page");
  }
  validateSource(page.source, request);
  for (const key of ["afterSeq", "earliestRetained", "latestSeq", "nextAfterSeq"]) exactIdentity(page[key]);
  const after = BigInt(page.afterSeq), earliest = BigInt(page.earliestRetained), latest = BigInt(page.latestSeq);
  if (earliest < 1n || earliest > latest + 1n || after > latest) throw new Error("Invalid intervention history range");
  const missing = after + 1n < earliest;
  if (missing) {
    if (!page.gap || page.gap.firstMissing !== (after + 1n).toString()
        || page.gap.lastMissing !== (earliest - 1n).toString()) throw new Error("Intervention history silently lost its missing prefix");
  } else if (page.gap !== null) throw new Error("Intervention history invented a missing prefix");
  let next = after > earliest - 1n ? after : earliest - 1n;
  for (const record of page.records) {
    for (const key of ["seq", "tick", "agentsAffected", "cellsAffected"]) exactIdentity(record?.[key]);
    if (BigInt(record.seq) !== next + 1n || BigInt(record.seq) > latest || BigInt(record.tick) > BigInt(page.source.tick)
        || !EFFECT_KINDS.has(record.kind) || typeof record.expired !== "boolean"
        || record.expired !== record.kind.startsWith("expiry:")) {
      throw new Error("Intervention history is not a contiguous observed sequence");
    }
    if (record.expiresAt !== null) exactIdentity(record.expiresAt);
    validateRegion(record.region);
    next += 1n;
  }
  if (page.nextAfterSeq !== next.toString() || page.hasMore !== (next < latest)
      || (page.hasMore && page.records.length !== request.limit)) throw new Error("Intervention history cursor disagrees with its page");
  return page;
}

export function interventionReceiptText(receipt) {
  const application = receipt.applicationState === "applied"
    ? `Queued at completed tick ${receipt.applicationBoundary.tick}. The ecological effect runs on the next scientific tick; this receipt does not identify an effect-history row.`
    : receipt.applicationState === "admitted" ? "Admitted by the owner; queue application has not yet been observed."
      : `Owner ${receipt.applicationState} this command.`;
  return `${application} Journal: ${receipt.journalState.replaceAll("_", " ")}.${receipt.message ? ` ${receipt.message}${receipt.messageTruncated ? "…" : ""}` : ""}${receipt.ownerError ? ` Owner drive reported: ${receipt.ownerError}` : ""}`;
}

/** One retained command and one history page; rendering never advances scientific time. */
export class InterventionController {
  constructor({ send, status, history, onChange = () => {}, now = () => performance.now(), id = interventionRequestId }) {
    Object.assign(this, { send, status, history, onChange, now, id, enabled: false,
      frame: null, context: null, retained: null, receipt: null, commandPending: null,
      historyPending: null, historyPage: null, historyAfter: "0", commandError: null, historyError: null, attachmentError: null,
      lastHistoryAt: -Infinity, lastStatusAt: -Infinity, lastHistoryPublication: null, lastStatusPublication: null });
  }
  emit() { this.onChange(this); }
  reset() {
    this.commandPending?.abort.abort(); this.historyPending?.abort.abort();
    Object.assign(this, { frame: null, context: null, retained: null, receipt: null, commandPending: null,
      historyPending: null, historyPage: null, historyAfter: "0", commandError: null, historyError: null, attachmentError: null,
      lastHistoryAt: -Infinity, lastStatusAt: -Infinity, lastHistoryPublication: null, lastStatusPublication: null });
    this.emit();
  }
  setEnabled(enabled) {
    enabled = Boolean(enabled) && !this.attachmentError;
    if (this.enabled === enabled) return;
    this.enabled = enabled;
    if (!enabled) {
      this.historyPending?.abort.abort(); this.historyPending = null;
      if (this.commandPending) {
        this.commandPending.abort.abort(); this.commandPending = null;
        this.commandError = "Command delivery or refresh was interrupted. The original request is retained; refresh its outcome or retry with the same ID.";
      }
    } else { void this.refreshHistory(true); }
    this.emit();
  }
  /** Reopen a saved handle/endpoint view; the caller must first match that owner. */
  reattachOwner(frame, generation) {
    this.setEnabled(false);
    if (this.frame?.identity.sessionId === frame.identity.sessionId) {
      const current = { ...frame.identity, epoch: frame.snapshot.epoch };
      const observed = [{ ...this.frame.identity, epoch: this.frame.snapshot.epoch },
        this.receipt?.source, this.historyPage?.source].filter(Boolean);
      const regressed = SOURCE_FIELDS.filter((field) => observed.some((source) => BigInt(current[field]) < BigInt(source[field])));
      if (regressed.length) {
        const error = `Owner source moved behind previously observed ${regressed.join(", ")}. Intervention actions are disabled until a coherent boundary returns or the owner is explicitly replaced. Any original request remains retained; nothing was resubmitted.`;
        if (this.attachmentError !== error) { this.attachmentError = error; this.emit(); }
        return false;
      }
      // Only presentation generation changes. The command's original source,
      // revision guard and retry identity must survive a round trip to another view.
      this.context = `${generation}/${frame.identity.sessionId}`;
      if (this.retained) this.retained = { ...this.retained, context: this.context };
    }
    this.attachmentError = null;
    this.observe(frame, generation);
    this.emit();
    return true;
  }
  observe(frame, generation) {
    const changed = this.context !== `${generation}/${frame.identity.sessionId}`
      || this.frame?.identity.publicationRevision !== frame.identity.publicationRevision;
    const context = `${generation}/${frame.identity.sessionId}`;
    if (this.context !== context) this.reset();
    if (this.frame && this.frame.snapshot.epoch !== frame.snapshot.epoch) {
      this.historyPending?.abort.abort(); this.historyPending = null; this.lastHistoryPublication = null;
    }
    this.context = context; this.frame = frame;
    if (this.enabled) {
      void this.refreshHistory();
      if (this.receipt && (this.receipt.applicationState === "admitted"
          || ["pending", "committed_volatile"].includes(this.receipt.journalState)) && this.now() - this.lastStatusAt >= 250
          && this.lastStatusPublication !== frame.identity.publicationRevision) void this.refreshReceipt();
    }
    if (changed) this.emit();
  }
  get canStage() {
    return this.enabled && this.frame?.playback?.paused === true && !this.commandPending
      && (!this.retained || (this.receipt && ["rejected", "failed"].includes(this.receipt.applicationState))
        || (this.receipt?.applicationState === "applied"
          && BigInt(this.frame.identity.tick) > BigInt(this.receipt.applicationBoundary.tick)));
  }
  async stage(action) {
    if (!this.canStage) throw new Error("Wait for a paused owner and a known outcome for the previous request");
    const request = makeInterventionRequest(this.frame, action, this.id());
    this.retained = { request, identity: { ...this.frame.identity }, context: this.context };
    this.receipt = null;
    return this.deliver();
  }
  async retry() {
    if (!this.enabled || !this.retained || this.commandPending) return;
    return this.deliver();
  }
  async deliver() {
    const retained = this.retained;
    const pending = { abort: new AbortController() };
    this.commandPending = pending; this.commandError = null; this.emit();
    try {
      const reply = await this.send(retained.request, pending.abort.signal);
      if (this.commandPending !== pending || this.retained !== retained || this.context !== retained.context) return;
      this.receipt = validateInterventionReceipt(reply, { ...retained.request, identity: retained.identity }, this.receipt);
    } catch (error) {
      if (this.commandPending === pending && this.retained === retained) this.commandError =
        `Delivery outcome unconfirmed: ${error.message || error}. Keep this request ID when retrying.`;
    } finally {
      if (this.commandPending === pending) { this.commandPending = null; this.emit(); }
    }
  }
  async refreshReceipt() {
    if (!this.enabled || !this.receipt || this.commandPending || !this.retained) return;
    const retained = this.retained, previous = this.receipt;
    const pending = { abort: new AbortController() };
    this.commandPending = pending; this.lastStatusAt = this.now();
    this.lastStatusPublication = this.frame.identity.publicationRevision; this.commandError = null; this.emit();
    try {
      const reply = await this.status(previous.commandId, retained.request.sessionId, pending.abort.signal);
      if (this.commandPending !== pending || this.retained !== retained || this.context !== retained.context) return;
      if (reply === null) this.commandError = "No retained receipt is available. Absence does not prove that the command was never admitted; the original request remains available for a guarded retry.";
      else this.receipt = validateInterventionReceipt(reply, { ...retained.request, identity: retained.identity }, previous);
    } catch (error) {
      if (this.commandPending === pending) this.commandError = `Outcome refresh failed: ${error.message || error}`;
    } finally {
      if (this.commandPending === pending) { this.commandPending = null; this.emit(); }
    }
  }
  async refreshHistory(force = false) {
    if (!this.enabled || !this.frame || this.historyPending) return;
    if (!force && (this.lastHistoryPublication === this.frame.identity.publicationRevision || this.now() - this.lastHistoryAt < 250)) return;
    const request = { sessionId: this.frame.identity.sessionId, epoch: this.frame.snapshot.epoch,
      identity: { ...this.frame.identity }, afterSeq: this.historyAfter, limit: 32,
      context: this.context, abort: new AbortController() };
    this.historyPending = request; this.lastHistoryAt = this.now();
    this.lastHistoryPublication = this.frame.identity.publicationRevision; this.historyError = null;
    try {
      const reply = await this.history(request, request.abort.signal);
      if (this.historyPending !== request || this.context !== request.context) return;
      this.historyPage = validateInterventionHistory(reply, request);
    } catch (error) {
      if (this.historyPending === request) this.historyError = `History read failed: ${error.message || error}`;
    } finally {
      if (this.historyPending === request) { this.historyPending = null; this.emit(); }
    }
  }
  historyPageAfter(afterSeq) {
    exactIdentity(afterSeq);
    this.historyPending?.abort.abort(); this.historyPending = null;
    this.historyAfter = afterSeq;
    return this.refreshHistory(true);
  }
}

const renderedHistory = new WeakMap();
export function renderInterventions(panel, controller) {
  if (!panel) return;
  const query = (id) => panel.querySelector(`#intervention-${id}`);
  const { frame, retained, receipt, commandPending, commandError, historyPage, historyError, attachmentError } = controller;
  query("stage").disabled = !controller.canStage;
  query("retry").disabled = !controller.enabled || !retained || !!commandPending;
  query("status-refresh").disabled = !controller.enabled || !receipt || !!commandPending;
  query("history-refresh").disabled = !frame || !controller.enabled;
  query("history-first").disabled = !frame || !controller.enabled;
  query("history-next").disabled = !historyPage?.hasMore || !!controller.historyPending || !controller.enabled;
  query("pause").disabled = !controller.enabled || !frame || frame.playback?.paused === true;
  query("step").disabled = !controller.enabled || !frame || frame.playback?.paused !== true || !!commandPending;
  query("resume").disabled = !controller.enabled || !frame || frame.playback?.paused !== true || !!commandPending;
  query("owner").textContent = attachmentError ? "Displayed owner has regressed behind an earlier observation. Intervention actions remain disabled."
    : frame ? `Displayed owner: ${frame.playback?.paused ? "paused" : "running"} · tick ${frame.identity.tick} · control revision ${frame.identity.controlRevision}. Pause to stage; Step or Resume applies queued effects.` : "Waiting for an observed world.";
  query("request").textContent = retained ? `${retained.request.intervention.kind.replaceAll("_", " ")} · request ${retained.request.requestId}${receipt ? ` · command ${receipt.commandId}` : ""}` : "No intervention request in this owner session.";
  query("receipt").textContent = `${commandPending ? "Waiting for owner response. " : ""}${receipt ? interventionReceiptText(receipt) : ""}${commandError ? ` ${commandError}` : ""}${attachmentError ? ` ${attachmentError}` : ""}`;
  query("history-source").textContent = historyError || (historyPage
    ? `Global effects observed at tick ${historyPage.source.tick}; world ${historyPage.closed ? "closed" : "open"}. Showing ${historyPage.records.length} records after #${historyPage.afterSeq}; latest observed #${historyPage.latestSeq}. These records are not correlated to the command above.`
    : "Open the toolbox to read global ecological effects.");
  query("history-gap").textContent = historyPage?.gap
    ? `History gap: effects #${historyPage.gap.firstMissing}–#${historyPage.gap.lastMissing} are no longer retained by this owner.` : "";
  if (renderedHistory.has(panel) && renderedHistory.get(panel) === historyPage) return;
  renderedHistory.set(panel, historyPage);
  query("history").replaceChildren();
  for (const record of historyPage?.records || []) {
    const item = panel.ownerDocument.createElement("li");
    const region = record.region.shape === "all" ? "whole world" : record.region.shape === "disc"
      ? `disc (${record.region.x}, ${record.region.y}), radius ${record.region.radius}`
      : `rectangle (${record.region.x}, ${record.region.y}), ${record.region.w}×${record.region.h}`;
    item.textContent = `#${record.seq} · tick ${record.tick} · ${record.kind.replaceAll("_", " ")}${record.expired ? " expired" : ""} · ${region} · ${record.agentsAffected} agents / ${record.cellsAffected} cells${record.expiresAt !== null ? ` · expiry tick ${record.expiresAt}` : ""}`;
    query("history").append(item);
  }
}
