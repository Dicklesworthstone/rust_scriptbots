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

function validateInspectionSource(detail, request) {
  if (detail.uid !== request.uid) throw new Error("Inspection subject changed");
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
}

/** Refuse malformed, oversized or differently sourced inspector responses. */
export function validateInspection(detail, request) {
  if (detail == null) return null;
  if (detail.schema !== "scriptbots.agent-inspection.v1"
      || typeof detail.brain !== "string" || [...detail.brain].length > 256
      || typeof detail.brainBound !== "boolean" || typeof detail.hybrid !== "boolean"
      || [detail.age, detail.generation].some((value) => !Number.isInteger(value) || value < 0 || value > 0xffff_ffff)) {
    throw new Error("Creature inspection has an invalid schema or subject");
  }
  validateInspectionSource(detail, request);
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

// Mirrors the independent hard ceilings declared by core's BrainInspectionLimits.
const BRAIN_LIMITS = { max_layers: 64, max_name_bytes: 1024, max_values: 4096,
  max_edges: 1024, max_payload_bytes: 65536, max_source_scalars: 65536 };
const BRAIN_UNAVAILABLE = {
  missing_agent: "The creature was absent when its owner answered.",
  unbound_brain: "This creature has no bound brain; there is no internal activity to inspect.",
  unsupported: "This brain family does not expose activity inspection.",
  no_recorded_evaluation: "No distinguishable recorded signal pair is available for this reconstruction. Step the world to obtain another sample.",
  recorded_response_mismatch: "The current brain cannot reproduce the recorded response; activity is unavailable at this boundary.",
};
const ATTRIBUTION_UNAVAILABLE = new Set(["no_activations", "no_connections", "identity_passthrough",
  "unsupported_topology", "truncated_topology", "dynamic_self_modifying_program", "placeholder_no_model", "numerical_overflow", "work_limit_exceeded"]);
const isCount = (value, max = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(value) && value >= 0 && value <= max;
const isName = (value) => typeof value === "string" && /^[a-z0-9_]{1,32}$/.test(value);
const oneKey = (value, key) => value && typeof value === "object" && !Array.isArray(value)
  && Object.keys(value).length === 1 && Object.hasOwn(value, key);

/** Validate all nested graph sizes and numbers before any canvas or table allocation. */
export function validateBrainInspection(detail, request) {
  if (!detail || detail.schema !== "scriptbots.brain-activity.v1") throw new Error("Invalid brain activity schema");
  validateInspectionSource(detail, request);
  for (const [field, hard] of Object.entries(BRAIN_LIMITS)) {
    if (!isCount(detail.limits?.[field], hard)) throw new Error("Brain activity limits exceed the core contract");
  }
  if (detail.status === "unavailable") {
    if (!Object.hasOwn(BRAIN_UNAVAILABLE, detail.reason)) throw new Error("Unknown brain unavailability reason");
    if (detail.activations !== undefined || detail.outputs !== undefined) throw new Error("Unavailable brain has a contradictory payload");
    return detail;
  }
  if (detail.status !== "ready" || !["evaluator_state", "reconstructed_feed_forward"].includes(detail.activitySource)) {
    throw new Error("Invalid brain activity status or observation method");
  }
  const { activations: activity, build, limits } = detail;
  if (!activity || !Array.isArray(activity.layers) || activity.layers.length > limits.max_layers
      || !Array.isArray(activity.connections) || activity.connections.length > limits.max_edges
      || !Array.isArray(activity.output_slots) || activity.output_slots.length > 9
      || typeof activity.truncated !== "boolean") throw new Error("Invalid bounded brain graph");
  let values = 0;
  let names = 0;
  let maxCells = 0;
  for (const layer of activity.layers) {
    if (!layer || typeof layer.name !== "string" || layer.name.length > limits.max_name_bytes
        || !isCount(layer.width, BRAIN_LIMITS.max_values) || !isCount(layer.height, BRAIN_LIMITS.max_values)
        || layer.width * layer.height > BRAIN_LIMITS.max_values || !Array.isArray(layer.values)
        || layer.values.length > layer.width * layer.height || layer.values.some((value) => !Number.isFinite(value))) {
      throw new Error("Invalid brain layer geometry or values");
    }
    values += layer.values.length;
    names += new TextEncoder().encode(layer.name).length;
    maxCells = Math.max(maxCells, layer.width * layer.height);
  }
  if (values > limits.max_values || names > limits.max_name_bytes) throw new Error("Brain layers exceed their value/name budgets");
  if (activity.connections.some((edge) => !isCount(edge?.from, maxCells - 1)
      || !isCount(edge.to, maxCells - 1) || !Number.isFinite(edge.weight))
      || activity.output_slots.some((slot) => !isCount(slot, maxCells - 1))) throw new Error("Invalid brain connection or output slot");
  if (activity.truncated && activity.output_slots.length) throw new Error("Clipped brain cannot retain an output mapping");
  for (const field of ["source_scalars", "source_layers", "source_name_bytes", "source_values", "source_edges",
    "retained_layers", "retained_name_bytes", "retained_values", "retained_edges", "retained_payload_bytes"]) {
    if (!isCount(build?.[field])) throw new Error("Invalid brain work accounting");
  }
  if (build.retained_layers !== activity.layers.length || build.retained_values !== values
      || build.retained_name_bytes !== names || build.retained_edges !== activity.connections.length
      || build.retained_payload_bytes > limits.max_payload_bytes || build.source_scalars > limits.max_source_scalars
      || build.source_layers < build.retained_layers || build.source_values < values || build.source_edges < build.retained_edges
      || build.truncated !== activity.truncated) throw new Error("Brain graph disagrees with its retained work evidence");
  if (!Array.isArray(detail.outputs) || detail.outputs.length !== 9) throw new Error("Brain output explanations have the wrong width");
  for (const [index, output] of detail.outputs.entries()) {
    if (output?.output_index !== index || !isName(output.output_name) || !Number.isFinite(output.raw_value)
        || !isCount(output.non_finite_skipped, 0xffff_ffff) || !Array.isArray(output.inputs) || output.inputs.length > 3) {
      throw new Error("Invalid bounded output explanation");
    }
    const effective = output.effective;
    if (!(oneKey(effective, "continuous") && Number.isFinite(effective.continuous))
        && !(oneKey(effective, "clamped") && Number.isFinite(effective.clamped?.raw)
          && Number.isFinite(effective.clamped.applied) && effective.clamped.raw === output.raw_value)
        && !(oneKey(effective, "thresholded") && Number.isFinite(effective.thresholded?.raw)
          && Number.isFinite(effective.thresholded.threshold) && typeof effective.thresholded.active === "boolean"
          && effective.thresholded.raw === output.raw_value)) throw new Error("Invalid effective actuator value");
    const method = output.method;
    const unavailable = oneKey(method, "unavailable") && ATTRIBUTION_UNAVAILABLE.has(method.unavailable);
    if (method !== "one_hop_weight_activation"
        && !(oneKey(method, "path_product") && isCount(method.path_product?.depth, 4) && method.path_product.depth > 0)
        && !unavailable) throw new Error("Unknown output explanation method");
    if ((unavailable && output.inputs.length) || (activity.truncated && !unavailable)) {
      throw new Error("Unavailable or clipped topology cannot claim input contributions");
    }
    if (output.inputs.some((input) => !isCount(input?.input_index, 24) || !isName(input.sensor_name)
        || [input.contribution, input.weight, input.activation].some((value) => !Number.isFinite(value)))
        || new Set(output.inputs.map((input) => input.input_index)).size !== output.inputs.length) {
      throw new Error("Invalid named input contribution");
    }
  }
  return detail;
}

/** One selected UID and at most one in-flight read. No history queue or timers drive science. */
export class InspectionController {
  constructor({ read, validate = validateInspection, enabled = true,
    onChange = () => {}, onFrame = () => {}, now = () => performance.now() }) {
    this.read = read;
    this.validate = validate;
    this.enabled = enabled;
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

  setEnabled(enabled) {
    if (this.enabled === enabled) return;
    this.enabled = enabled;
    this.cancel();
    this.lastRequested = null;
    this.lastReadAt = -Infinity;
    this.emit({ status: enabled ? (this.selectedUid ? "loading" : "empty") : "inactive",
      uid: this.selectedUid, detail: null });
    if (enabled) void this.refresh(true);
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
    this.emit({ status: this.enabled ? "loading" : "inactive", uid, detail: null });
    void this.refresh(true);
  }

  async refresh(force = false) {
    const frame = this.frame;
    const uid = this.selectedUid;
    if (!this.enabled || !frame || !uid || this.pending
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
      const detail = this.validate(received, request);
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

const brainPanels = new WeakMap();
const EDGE_PAGE_SIZE = 32;

/** Signed display color only; the selected cell always reports the original scalar. */
export function brainActivityColor(value) {
  const strength = Math.abs(Math.tanh(value));
  const target = value < 0 ? [247, 168, 54] : [72, 160, 255];
  return [25, 34, 52].map((base, index) => Math.round(base + (target[index] - base) * strength));
}

/** Compact row-major display stays within 64×64 cells even for a 4096×1 or 1×4096 layer. */
export function brainHeatmapShape(layer) {
  const count = layer?.values?.length ?? 0;
  if (!isCount(count, BRAIN_LIMITS.max_values)) throw new Error("Brain heatmap exceeds its value bound");
  const columns = Math.min(64, Math.max(1, count), Math.max(1, layer?.width ?? 1, Math.ceil(Math.sqrt(count))));
  return { columns, rows: Math.max(1, Math.ceil(count / columns)) };
}

function paintBrainDetail(panel, view) {
  const detail = view.detail;
  if (!detail || detail.status !== "ready") return;
  const layerSelect = panel.querySelector("#brain-layer");
  const nodeInput = panel.querySelector("#brain-node");
  const layerIndex = Math.min(Number(layerSelect.value) || 0, Math.max(0, detail.activations.layers.length - 1));
  const layer = detail.activations.layers[layerIndex];
  const node = Math.min(Math.max(0, Math.trunc(Number(nodeInput.value) || 0)), Math.max(0, (layer?.values.length ?? 0) - 1));
  if (Number(nodeInput.value) !== node) nodeInput.value = String(node);
  nodeInput.max = String(Math.max(0, (layer?.values.length ?? 0) - 1));
  nodeInput.disabled = !layer?.values.length;
  const canvas = panel.querySelector("#brain-heatmap");
  const { columns, rows } = brainHeatmapShape(layer);
  canvas.width = columns * 8;
  canvas.height = rows * 8;
  const context = canvas.getContext("2d");
  context.fillStyle = "#192234";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (const [index, value] of (layer?.values ?? []).entries()) {
    context.fillStyle = `rgb(${brainActivityColor(value).join(",")})`;
    context.fillRect(index % columns * 8, Math.floor(index / columns) * 8, 8, 8);
  }
  if (layer?.values.length) {
    context.strokeStyle = "#ffffff";
    context.lineWidth = 1;
    context.strokeRect(node % columns * 8 + 0.5, Math.floor(node / columns) * 8 + 0.5, 7, 7);
  }
  view.columns = columns;
  view.layer = layerIndex;
  canvas.setAttribute("aria-label", layer
    ? `${layer.name}: ${layer.values.length} retained activation values; select a node below for its exact value`
    : "No activation values retained");
  panel.querySelector("#brain-node-value").textContent = layer?.values.length
    ? `${layer.name} · node ${node} = ${String(layer.values[node])}. Retained ${layer.values.length} of ${layer.width * layer.height} declared cells; compact row-major display.`
    : "No values were retained for this layer.";

  const edges = detail.activations.connections;
  const pages = Math.max(1, Math.ceil(edges.length / EDGE_PAGE_SIZE));
  view.edgePage = Math.min(view.edgePage, pages - 1);
  const start = view.edgePage * EDGE_PAGE_SIZE;
  panel.querySelector("#brain-edge-page").textContent = edges.length
    ? `Connections ${start + 1}–${Math.min(start + EDGE_PAGE_SIZE, edges.length)} of ${edges.length} retained · flat node indices`
    : "No weighted connections were retained.";
  panel.querySelector("#brain-edge-prev").disabled = view.edgePage === 0;
  panel.querySelector("#brain-edge-next").disabled = view.edgePage + 1 >= pages;
  const edgeRows = panel.querySelector("#brain-edges");
  edgeRows.replaceChildren();
  for (const edge of edges.slice(start, start + EDGE_PAGE_SIZE)) {
    const row = document.createElement("tr");
    for (const value of [edge.from, edge.to, String(edge.weight)]) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    }
    edgeRows.append(row);
  }

  const outputIndex = Math.min(8, Number(panel.querySelector("#brain-output").value) || 0);
  const output = detail.outputs[outputIndex];
  const method = output.method;
  const methodText = method === "one_hop_weight_activation" ? "Direct weight × activation estimate"
    : method.path_product ? `Weight-product path estimate, depth ${method.path_product.depth}`
      : `Attribution unavailable: ${method.unavailable.replaceAll("_", " ")}`;
  const effective = output.effective;
  const effectiveText = effective.thresholded
    ? `${effective.thresholded.active ? "ON" : "OFF"} at threshold ${String(effective.thresholded.threshold)}`
    : String(effective.clamped?.applied ?? effective.continuous);
  panel.querySelector("#brain-output-value").textContent = `Recorded ${String(output.raw_value)} · actuator ${effectiveText}. ${methodText}.`;
  const inputs = panel.querySelector("#brain-contributions");
  inputs.replaceChildren();
  for (const input of output.inputs) {
    const row = document.createElement("tr");
    for (const value of [input.sensor_name.replaceAll("_", " "), String(input.contribution),
      String(input.weight), String(input.activation)]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    inputs.append(row);
  }
  panel.querySelector("#brain-attribution-note").textContent = output.inputs.length
    ? "The shared core ranks up to three input contributions. These bounded weight estimates describe the exposed graph, not a causal intervention or full recurrent history."
    : "No input contribution is claimed for this output at this observation.";
  if (output.non_finite_skipped) panel.querySelector("#brain-attribution-note").textContent += ` Core reported ${output.non_finite_skipped} non-finite input or arithmetic events.`;
}

export function renderBrainSource(panel, state, frame) {
  if (panel && state.status === "ready") panel.querySelector("#brain-source").textContent = inspectionStatus(state.detail, frame);
}

/** Render a bounded activity graph without fetching or changing science from UI callbacks. */
export function renderBrainInspector(panel, state, frame) {
  if (!panel) return;
  let view = brainPanels.get(panel);
  if (!view) {
    view = { detail: null, shape: "", edgePage: 0, columns: 1, layer: 0 };
    brainPanels.set(panel, view);
    for (const id of ["brain-layer", "brain-node", "brain-output"]) {
      panel.querySelector(`#${id}`).addEventListener("input", () => paintBrainDetail(panel, view));
    }
    for (const [id, change] of [["brain-edge-prev", -1], ["brain-edge-next", 1]]) {
      panel.querySelector(`#${id}`).addEventListener("click", () => {
        view.edgePage = Math.max(0, view.edgePage + change);
        paintBrainDetail(panel, view);
      });
    }
    panel.querySelector("#brain-heatmap").addEventListener("click", (event) => {
      const layer = view.detail?.activations?.layers[view.layer];
      if (!layer?.values.length) return;
      const canvas = event.currentTarget;
      const rect = canvas.getBoundingClientRect();
      const x = Math.floor((event.clientX - rect.left) / rect.width * canvas.width / 8);
      const y = Math.floor((event.clientY - rect.top) / rect.height * canvas.height / 8);
      const node = y * view.columns + x;
      if (x < 0 || x >= view.columns || y < 0 || node >= layer.values.length) return;
      panel.querySelector("#brain-node").value = String(node);
      paintBrainDetail(panel, view);
    });
  }
  panel.dataset.state = state.status === "ready" ? state.detail.status : state.status;
  const content = panel.querySelector("#brain-content");
  const status = panel.querySelector("#brain-status");
  panel.querySelector("#brain-refresh").disabled = !state.uid || ["missing", "inactive"].includes(state.status);
  content.hidden = true;
  view.detail = null;
  panel.querySelector("#brain-source").textContent = "";
  if (state.status !== "ready") {
    const messages = { empty: "Select a creature to inspect its brain.", inactive: "Brain reads are stopped while this panel or page is hidden.",
      loading: "Reading bounded activity from the creature's owner…", missing: "The selected creature is no longer available in the displayed population.",
      error: `Brain observation unavailable: ${state.message ?? "unknown error"}. Refresh to retry.` };
    status.textContent = messages[state.status];
    return;
  }
  const detail = state.detail;
  view.detail = detail;
  renderBrainSource(panel, state, frame);
  if (detail.status === "unavailable") {
    status.textContent = BRAIN_UNAVAILABLE[detail.reason];
    return;
  }
  const build = detail.build;
  status.textContent = `${detail.activitySource === "reconstructed_feed_forward"
    ? "Feed-forward activity reconstructed from recorded inputs; the family verified the recorded outputs"
    : "Retained evaluator activity"}. ${build.retained_layers} layers, ${build.retained_values} values, ${build.retained_edges} weighted connections. ${build.truncated
      ? "CLIPPED by inspection bounds; the network is incomplete and input attribution is unavailable."
      : "No activity clipping reported."}`;
  panel.querySelector("#brain-budget").textContent = `Owner work: ${build.source_scalars} source scalars; ${build.retained_payload_bytes} retained bytes. Limits: ${detail.limits.max_values} values, ${detail.limits.max_edges} connections, ${detail.limits.max_payload_bytes} bytes.`;
  const shape = JSON.stringify([detail.uid, detail.source.sessionId, detail.source.epoch,
    detail.activations.layers.map((layer) => [layer.name, layer.width, layer.height]), detail.outputs.map((output) => output.output_name)]);
  if (view.shape !== shape) {
    view.shape = shape;
    view.edgePage = 0;
    for (const [id, names] of [["brain-layer", detail.activations.layers.map((layer) => layer.name)],
      ["brain-output", detail.outputs.map((output) => output.output_name.replaceAll("_", " "))]]) {
      const select = panel.querySelector(`#${id}`);
      select.replaceChildren();
      names.forEach((name, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = name;
        select.append(option);
      });
      select.disabled = names.length === 0;
    }
    panel.querySelector("#brain-node").value = "0";
  }
  content.hidden = false;
  paintBrainDetail(panel, view);
}

