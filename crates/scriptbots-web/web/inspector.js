// Presentation-owned creature selection. Every scientific value comes from an
// owner observation; picking, refreshes and highlighting never submit commands.

export function exactIdentity(value, bits = 64) {
  if (typeof value !== "string" || !/^(0|[1-9][0-9]*)$/.test(value)
      || value.length > (bits === 128 ? 39 : 20) || BigInt(value) >= (1n << BigInt(bits))) {
    throw new Error("Creature and source identities must be exact canonical decimal strings");
  }
  return value;
}

export function agentUid(agent) {
  try { return exactIdentity(agent?.uid); } catch { return null; }
}

export function agentRadius(agent) {
  return Math.max(1.5, 3.5 * Math.sqrt(Math.max(0, agent.health + 0.15)));
}

/** Pick in CSS pixels using the same projection/radius as both renderers. */
export function pickAgent(snapshot, x, y, { width, height, canvasWidth, canvasHeight }) {
  if (!snapshot || !(width > 0 && height > 0 && canvasWidth > 0 && canvasHeight > 0)
      || !Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x > width || y > height) return null;
  const scale = Math.max(width / canvasWidth, height / canvasHeight);
  let picked = null;
  let nearest = Infinity;
  for (const agent of snapshot.agents) {
    const uid = agentUid(agent);
    if (uid === null) continue;
    const dx = x - agent.position[0] / snapshot.world.width * width;
    const dy = y - agent.position[1] / snapshot.world.height * height;
    const distance = dx * dx + dy * dy;
    const radius = Math.max(8, (agentRadius(agent) + 5) * scale);
    if (distance <= radius * radius && (distance < nearest
        || (distance === nearest && (picked === null || BigInt(uid) < BigInt(picked))))) {
      nearest = distance;
      picked = uid;
    }
  }
  return picked;
}

function epochOf(frame) {
  return exactIdentity(frame.snapshot.epoch);
}

const REVISION_FIELDS = ["scientific", "control", "config"];
const NUMERIC_FIELDS = ["health", "energy", "foodBalanceTotal", "herbivoreTendency",
  "temperaturePreference", "soundOutput", "giveIntent", "reproductionCounter"];

/** Refuse malformed, oversized or differently sourced inspector responses. */
export function validateInspection(detail, request) {
  if (detail == null) return null;
  if (detail.schema !== "scriptbots.agent-inspection.v1" || detail.uid !== request.uid
      || typeof detail.brain !== "string" || [...detail.brain].length > 256
      || typeof detail.brainBound !== "boolean" || typeof detail.hybrid !== "boolean"
      || [detail.age, detail.generation].some((value) => !Number.isInteger(value) || value < 0 || value > 0xffff_ffff)) {
    throw new Error("Creature inspection has an invalid schema or subject");
  }
  exactIdentity(detail.uid);
  const source = detail.source;
  if (source?.sessionId !== request.sessionId || source.epoch !== request.epoch) {
    throw new Error("Creature inspection belongs to another host session or epoch");
  }
  exactIdentity(source.sessionId, 128);
  exactIdentity(source.epoch);
  exactIdentity(source.tick);
  exactIdentity(source.publicationRevision);
  if (BigInt(source.tick) < BigInt(request.identity.tick)
      || BigInt(source.publicationRevision) < BigInt(request.identity.publicationRevision)) {
    throw new Error("Creature inspection predates the selected publication");
  }
  for (const field of REVISION_FIELDS) {
    exactIdentity(source.published?.[field]);
    exactIdentity(source.inspected?.[field]);
    if (BigInt(source.published[field]) < BigInt(request.identity[`${field}Revision`])
        || BigInt(source.inspected[field]) < BigInt(source.published[field])) {
      throw new Error("Creature inspection source revisions regressed");
    }
  }
  if (NUMERIC_FIELDS.some((field) => !Number.isFinite(detail[field]))
      || !Array.isArray(detail.clockScales) || detail.clockScales.length !== 2
      || detail.clockScales.some((value) => !Number.isFinite(value))
      || !Array.isArray(detail.lineage) || detail.lineage.length !== 2) {
    throw new Error("Creature inspection contains invalid physiology or lineage");
  }
  for (const uid of detail.lineage) if (uid !== null) exactIdentity(uid);
  for (const [object, fields] of [[detail.mutationRates, ["primary", "secondary"]],
    [detail.traitModifiers, ["smell", "sound", "hearing", "eye", "blood"]]]) {
    if (fields.some((field) => !Number.isFinite(object?.[field]))) {
      throw new Error("Creature inspection contains invalid inherited traits");
    }
  }
  for (const [channels, count] of [[detail.sensors, 25], [detail.outputs, 9]]) {
    if (!Array.isArray(channels) || channels.length !== count
        || channels.some((channel) => typeof channel?.name !== "string"
          || !/^[a-z0-9_]{1,32}$/.test(channel.name) || !Number.isFinite(channel.value))
        || new Set(channels.map((channel) => channel.name)).size !== count) {
      throw new Error("Creature inspection has invalid named sensor/actuator channels");
    }
  }
  return detail;
}

/** One selected UID and at most one in-flight read. No history queue or timers drive science. */
export class InspectionController {
  constructor({ read, onChange = () => {}, onFrame = () => {}, now = () => performance.now() }) {
    this.read = read;
    this.onChange = onChange;
    this.onFrame = onFrame;
    this.now = now;
    this.frame = null;
    this.context = null;
    this.selectedUid = null;
    this.pending = null;
    this.lastRequested = null;
    this.lastReadAt = -Infinity;
    this.state = { status: "empty", uid: null, detail: null };
  }

  emit(state) {
    this.state = state;
    this.onChange(state, this.frame);
  }

  cancel() {
    const pending = this.pending;
    this.pending = null;
    pending?.abort.abort();
  }

  clear() {
    this.cancel();
    this.selectedUid = null;
    this.lastRequested = null;
    this.lastReadAt = -Infinity;
    this.emit({ status: "empty", uid: null, detail: null });
  }

  reset() {
    this.frame = null;
    this.context = null;
    this.clear();
  }

  selectionFor(frame, generation) {
    return this.context === `${generation}/${frame.identity.sessionId}/${epochOf(frame)}` ? this.selectedUid : null;
  }

  observe(frame, generation) {
    const previous = this.frame;
    const context = `${generation}/${frame.identity.sessionId}/${epochOf(frame)}`;
    if (this.context !== context) this.clear();
    this.context = context;
    this.frame = frame;
    if (this.state.status === "ready" && previous?.identity.publicationRevision !== frame.identity.publicationRevision) {
      this.onFrame(this.state, frame);
    }
    if (!this.selectedUid) return;
    if (!frame.snapshot.agents.some((agent) => agentUid(agent) === this.selectedUid)) {
      this.cancel();
      if (this.state.status !== "missing" || this.state.reason !== "publication") {
        this.emit({ status: "missing", reason: "publication", uid: this.selectedUid, detail: null });
      }
      return;
    }
    void this.refresh();
  }

  select(uid) {
    if (uid === null) { this.clear(); return; }
    exactIdentity(uid);
    if (!this.frame) throw new Error("Wait for a presented world before selecting a creature");
    this.cancel();
    this.selectedUid = uid;
    this.lastRequested = null;
    this.lastReadAt = -Infinity;
    if (!this.frame.snapshot.agents.some((agent) => agentUid(agent) === uid)) {
      this.emit({ status: "missing", reason: "publication", uid, detail: null });
      return;
    }
    this.emit({ status: "loading", uid, detail: null });
    void this.refresh(true);
  }

  async refresh(force = false) {
    const frame = this.frame;
    const uid = this.selectedUid;
    if (!frame || !uid || this.pending
        || !frame.snapshot.agents.some((agent) => agentUid(agent) === uid)) return;
    const key = `${this.context}/${uid}/${frame.identity.publicationRevision}`;
    if (!force && (this.lastRequested === key || this.now() - this.lastReadAt < 250)) return;
    const request = { uid, sessionId: frame.identity.sessionId, epoch: epochOf(frame),
      identity: { ...frame.identity }, context: this.context, abort: new AbortController() };
    this.pending = request;
    this.lastRequested = key;
    this.lastReadAt = this.now();
    try {
      const received = await this.read(uid, request, request.abort.signal);
      if (this.pending !== request || this.context !== request.context || this.selectedUid !== uid) return;
      const detail = validateInspection(received, request);
      const previous = this.state.detail?.source;
      if (detail && previous && (BigInt(detail.source.tick) < BigInt(previous.tick)
          || REVISION_FIELDS.some((field) => BigInt(detail.source.inspected[field]) < BigInt(previous.inspected[field])))) {
        throw new Error("Creature inspection regressed behind the previous observation");
      }
      this.emit({ status: detail ? "ready" : "missing", reason: detail ? undefined : "owner", uid, detail });
    } catch (error) {
      if (this.pending === request && this.context === request.context && this.selectedUid === uid) {
        this.emit({ status: "error", uid, detail: null, message: error.message || String(error) });
      }
    } finally {
      if (this.pending === request) this.pending = null;
    }
  }
}

function displayNumber(value) {
  return Number.isFinite(value) ? Number(value.toPrecision(5)).toString() : "—";
}

/** A detail may remain useful while its owner sample trails the newest painted frame. */
export function inspectionStatus(detail, frame) {
  const source = detail.source;
  const same = frame && source.tick === frame.identity.tick
    && source.publicationRevision === frame.identity.publicationRevision
    && REVISION_FIELDS.every((field) => source.inspected[field] === frame.identity[`${field}Revision`]
      && source.published[field] === frame.identity[`${field}Revision`]);
  return `Observed tick ${source.tick} · ${same ? "matches displayed frame" : `displayed tick ${frame?.identity.tick ?? "unavailable"}`}.`;
}

/** Update only the provenance sentence when the canvas advances between bounded reads. */
export function renderInspectorSource(panel, state, frame) {
  if (panel && state.status === "ready") panel.querySelector("#inspector-status").textContent = inspectionStatus(state.detail, frame);
}

/** Update text-only accessible scientific detail; never interpret owner strings as markup. */
export function renderInspector(panel, state, frame) {
  if (!panel) return;
  panel.dataset.state = state.status;
  const status = panel.querySelector("#inspector-status");
  const summary = panel.querySelector("#inspector-summary");
  const signals = panel.querySelector("#inspector-signals");
  const input = panel.querySelector("#inspector-uid");
  panel.querySelector("#inspector-clear").disabled = state.uid === null;
  panel.querySelector("#inspector-refresh").disabled = state.uid === null || state.status === "missing";
  const selected = state.uid ?? "";
  if (panel.dataset.selectedUid !== selected) input.value = selected;
  panel.dataset.selectedUid = selected;
  summary.replaceChildren();
  signals.replaceChildren();
  for (const field of ["sessionId", "epoch", "tick", "publicationRevision", "uid"]) delete panel.dataset[field];
  const messages = {
    empty: "Click a creature in the world, or enter its stable UID.",
    loading: `Reading creature ${state.uid} from its owner…`,
    missing: state.reason === "owner"
      ? `Creature ${state.uid} was absent when the owner answered; the displayed frame may be older.`
      : `Creature ${state.uid} is not in the displayed population (it may have died or moved).`,
    error: `Inspection unavailable: ${state.message}. Refresh to retry.`,
  };
  if (state.status !== "ready") {
    status.textContent = messages[state.status];
    return;
  }
  const detail = state.detail;
  const source = detail.source;
  for (const field of ["sessionId", "epoch", "tick", "publicationRevision"]) panel.dataset[field] = source[field];
  panel.dataset.uid = detail.uid;
  renderInspectorSource(panel, state, frame);
  const row = (name, value) => {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = name;
    description.textContent = value;
    summary.append(term, description);
  };
  row("UID", detail.uid);
  row("Brain", detail.brainBound ? detail.brain : "Unbound");
  row("Health", displayNumber(detail.health));
  row("Age (ticks)", String(detail.age));
  row("Generation", String(detail.generation));
  row("Energy", displayNumber(detail.energy));
  row("Lifetime food balance", displayNumber(detail.foodBalanceTotal));
  row("Reproduction", displayNumber(detail.reproductionCounter));
  row("Diet (herbivore)", displayNumber(detail.herbivoreTendency));
  row("Temperature preference", displayNumber(detail.temperaturePreference));
  row("Emitted sound", displayNumber(detail.soundOutput));
  row("Sharing intent", displayNumber(detail.giveIntent));
  row("Parents", detail.lineage.filter((uid) => uid !== null).join(" + ") || "Founder");
  row("Two-parent offspring", detail.hybrid ? "Yes" : "No");
  row("Clock divisors", detail.clockScales.map(displayNumber).join(" / "));
  row("Mutation gate / scale", [detail.mutationRates.primary, detail.mutationRates.secondary].map(displayNumber).join(" / "));
  row("Sense multipliers", Object.entries(detail.traitModifiers).map(([name, value]) => `${name}: ${displayNumber(value)}`).join("; "));
  for (const [heading, channels] of [["Recorded senses", detail.sensors], ["Recorded outputs", detail.outputs]]) {
    const title = document.createElement("h3");
    title.textContent = heading;
    const table = document.createElement("table");
    table.setAttribute("aria-label", heading);
    for (const channel of channels) {
      const tr = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = channel.name.replaceAll("_", " ");
      const value = document.createElement("td");
      value.textContent = displayNumber(channel.value);
      tr.append(name, value);
      table.append(tr);
    }
    signals.append(title, table);
  }
}
