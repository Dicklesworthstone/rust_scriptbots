import { STUDY_TRAJECTORY_METRICS } from "./study-trajectories.js";

// Saved-study presentation consumes schema 4 reports after the transport verifies
// the downloaded artifact. It never executes report content or reconstructs a world.
export const MAX_STUDY_REPORT_BYTES = 1024 * 1024;
const MAX_REPORT_NODES = 8192;
const MAX_REPORT_DEPTH = 24;
const MAX_HORIZON_TICKS = 10000;
const ID = /^[A-Za-z0-9_-]{1,128}$/;
const WORLD_DIGEST = /^[0-9a-f]{16}$/;
const CHECKSUM = /^[0-9a-f]{64}$/;
const METRICS = [
  ["population", "Population", true],
  ["food_total", "Ground food", false],
  ["total_energy", "Total energy", false],
  ["average_health", "Average health", false],
  ["births", "Births in final tick", true],
  ["deaths", "Deaths in final tick", true],
  ["spike_hits", "Spike hits in final tick", true],
];
const EFFECT_METRICS = ["population", "food_total", "total_energy"];

function refuse(message) { throw new Error("Study report refused: " + message); }
const object = (value) => value !== null && typeof value === "object"
  && !Array.isArray(value) && !(value instanceof JsonNumber);
const safeInteger = (value) => Number.isSafeInteger(value) && value >= 0;
function string(value, label, limit = 4096) {
  if (typeof value !== "string" || !value.length || value.length > limit) refuse(label + " is invalid");
  return value;
}
function identifier(value, label) {
  if (typeof value !== "string" || !ID.test(value)) refuse(label + " is invalid");
  return value;
}
function worldDigest(value, label) {
  if (typeof value !== "string" || !WORLD_DIGEST.test(value)) refuse(label + " is invalid");
  return value;
}
function checksum(value, label) {
  if (typeof value !== "string" || !CHECKSUM.test(value)) refuse(label + " is invalid");
  return value;
}
function shape(value, required, optional = [], label = "record") {
  if (!object(value) || required.some((key) => !Object.hasOwn(value, key))
      || Object.keys(value).some((key) => !required.includes(key) && !optional.includes(key))) {
    refuse(label + " has missing or unsupported fields");
  }
}
function boolean(value, label) {
  if (typeof value !== "boolean") refuse(label + " is invalid");
  return value;
}
function freeze(value) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

// JSON.parse rounds a u64 seed before a reviver can inspect it in many browsers.
// This bounded reader preserves numeric tokens, rejects duplicate object keys,
// and delegates only string escape decoding to the platform JSON parser.
class JsonNumber {
  constructor(raw) { this.raw = raw; this.value = Number(raw); }
}

function readJson(text) {
  let position = 0, nodes = 0;
  const number = /-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/y;
  const whitespace = () => {
    while (position < text.length && " \t\r\n".includes(text[position])) position += 1;
  };
  const quoted = () => {
    const start = position++;
    while (position < text.length) {
      const next = text[position++];
      if (next === "\\") { position += 1; continue; }
      if (next === '"') {
        try { return JSON.parse(text.slice(start, position)); }
        catch (_) { refuse("invalid JSON string"); }
      }
    }
    refuse("unterminated JSON string");
  };
  const read = (depth) => {
    whitespace();
    if (depth > MAX_REPORT_DEPTH || ++nodes > MAX_REPORT_NODES) refuse("JSON structure exceeds its bound");
    const next = text[position];
    if (next === '"') return quoted();
    if (next === "{") {
      position += 1;
      const result = Object.create(null);
      whitespace();
      if (text[position] === "}") { position += 1; return result; }
      for (;;) {
        whitespace();
        if (text[position] !== '"') refuse("invalid JSON object key");
        const key = quoted();
        if (Object.hasOwn(result, key)) refuse("duplicate JSON field " + key.slice(0, 128));
        whitespace();
        if (text[position++] !== ":") refuse("missing JSON colon");
        result[key] = read(depth + 1);
        whitespace();
        const separator = text[position++];
        if (separator === "}") return result;
        if (separator !== ",") refuse("invalid JSON object separator");
      }
    }
    if (next === "[") {
      position += 1;
      const result = [];
      whitespace();
      if (text[position] === "]") { position += 1; return result; }
      for (;;) {
        result.push(read(depth + 1));
        whitespace();
        const separator = text[position++];
        if (separator === "]") return result;
        if (separator !== ",") refuse("invalid JSON array separator");
      }
    }
    for (const [token, value] of [["true", true], ["false", false], ["null", null]]) {
      if (text.startsWith(token, position)) { position += token.length; return value; }
    }
    number.lastIndex = position;
    const matched = number.exec(text);
    if (!matched) refuse("invalid JSON value");
    position = number.lastIndex;
    return new JsonNumber(matched[0]);
  };
  const result = read(0);
  whitespace();
  if (position !== text.length) refuse("unexpected data after JSON");
  return result;
}

function unsigned(value, label, maximum = BigInt(Number.MAX_SAFE_INTEGER)) {
  if (!(value instanceof JsonNumber) || !/^(0|[1-9][0-9]*)$/.test(value.raw)
      || value.raw.length > 20 || BigInt(value.raw) > maximum) refuse(label + " is not an exact bounded integer");
  return value.raw;
}
function count(value, label) { return Number(unsigned(value, label)); }
function seed(value, label) { return unsigned(value, label, (1n << 64n) - 1n); }
function finite(value, label, minimum = -Infinity, maximum = Infinity) {
  if (!(value instanceof JsonNumber) || !Number.isFinite(value.value)
      || value.value < minimum || value.value > maximum) refuse(label + " is not a finite bounded number");
  return value.value;
}
function tick(value, label, low, high) {
  const result = count(value, label);
  if (result < low || result > high) refuse(label + " falls outside the observed study");
  return result;
}
function optionalTick(value, label, low, high) {
  return value === null ? null : tick(value, label, low, high);
}

function contextIdentity(context) {
  const status = context?.status;
  if (!object(status) || status.status !== "completed" || !safeInteger(status.source_tick)
      || !safeInteger(status.target_tick) || status.target_tick <= status.source_tick
      || status.target_tick - status.source_tick > MAX_HORIZON_TICKS) refuse("a completed, exact study receipt is required");
  identifier(status.branch_id, "study identity");
  identifier(status.checkpoint_id, "checkpoint identity");
  identifier(status.artifact_id, "report artifact identity");
  worldDigest(status.world_digest_v1, "saved world digest");
  return status;
}

function readArm(value, status, rootSeed) {
  shape(value, ["branch_id", "description", "command_applied_tick", "first_divergence", "extinction",
    "recovery", "hysteresis", "effect_sizes", "final_digest", "terminal_sample", "bundle_path", "bundle_verified"], [], "branch");
  if (!["control", "treatment"].includes(value.branch_id)) refuse("unsupported managed branch");
  const isControl = value.branch_id === "control", low = status.source_tick, high = status.target_tick;
  string(value.description, "branch description");
  const applied = optionalTick(value.command_applied_tick, "intervention boundary", low, high - 1);
  if (isControl ? applied !== null : applied === null) refuse("branch intervention boundary disagrees with its role");
  const sample = value.terminal_sample;
  shape(sample, ["tick", "population", "food_total", "births", "deaths", "total_energy", "average_health", "spike_hits", "digest_overall"], [], "terminal sample");
  if (tick(sample.tick, "terminal tick", low, high) !== high) refuse("terminal sample does not reach the admitted target");
  const terminal = { tick: high };
  for (const [key, label, integral] of METRICS) terminal[key] = integral ? count(sample[key], label) : finite(sample[key], label, 0);
  terminal.digest = worldDigest(sample.digest_overall, "terminal sample digest");
  if (worldDigest(value.final_digest, "final digest") !== terminal.digest) refuse("final digest differs from the terminal sample");

  const extinction = value.extinction;
  shape(extinction, ["extinct", "extinction_tick", "min_population", "final_population"], [], "extinction");
  const baselineTick = applied ?? low;
  const firstZero = optionalTick(extinction.extinction_tick, "first zero population tick", baselineTick, high);
  const minimum = count(extinction.min_population, "minimum population");
  if (count(extinction.final_population, "extinction final population") !== terminal.population
      || minimum > terminal.population || boolean(extinction.extinct, "extinction flag") !== (minimum === 0)
      || (extinction.extinct ? firstZero === null : firstZero !== null)) refuse("extinction observations disagree");

  const recovery = value.recovery;
  shape(recovery, ["baseline_tick", "pre_intervention_population", "trough_population", "trough_tick",
    "final_population", "recovered", "recovery_ratio", "recovery_ticks"], [], "recovery");
  if (tick(recovery.baseline_tick, "recovery baseline", low, high) !== baselineTick
      || count(recovery.final_population, "recovery final population") !== terminal.population
      || count(recovery.trough_population, "trough population") !== minimum) refuse("recovery observations disagree");
  const baselinePopulation = count(recovery.pre_intervention_population, "baseline population");
  const troughTick = tick(recovery.trough_tick, "trough tick", baselineTick, high);
  if (minimum > baselinePopulation) refuse("trough exceeds its population baseline");
  const ratio = finite(recovery.recovery_ratio, "recovery ratio", 0);
  const expectedRatio = terminal.population / Math.max(1, baselinePopulation);
  if (Math.abs(ratio - expectedRatio) > 1e-12 * Math.max(1, expectedRatio)) refuse("recovery ratio disagrees with its populations");
  const recovered = boolean(recovery.recovered, "recovery flag");
  const recoveryTicks = optionalTick(recovery.recovery_ticks, "recovery duration", 1, high - troughTick);
  if (!recovered && recoveryTicks !== null) refuse("unrecovered trajectory declares a recovery duration");

  const hysteresis = value.hysteresis;
  shape(hysteresis, ["integrated_trajectory_deficit", "normalized_trajectory_difference",
    "terminal_trajectory_gap", "hysteresis_index", "path_dependence_detected"], [], "trajectory assessment");
  for (const key of ["integrated_trajectory_deficit", "normalized_trajectory_difference", "terminal_trajectory_gap", "hysteresis_index"]) {
    finite(hysteresis[key], key);
  }
  boolean(hysteresis.path_dependence_detected, "path-dependence flag");

  let divergence = null;
  if (value.first_divergence !== null) {
    if (isControl) refuse("control cannot diverge from itself");
    const first = value.first_divergence;
    shape(first, ["branch_id", "tick", "stage", "control_digest", "branch_digest", "command_applied_tick", "details"], [], "first divergence");
    if (first.branch_id !== value.branch_id || tick(first.command_applied_tick, "divergence command boundary", low, high - 1) !== applied) {
      refuse("divergence belongs to another branch or intervention");
    }
    // Most lanes carry hashes; hydrology can say "absent" and RNG uses its
    // canonical debug observation. Keep the actual bounded lane evidence.
    const controlEvidence = string(first.control_digest, "control divergence evidence");
    const branchEvidence = string(first.branch_digest, "treatment divergence evidence");
    if (controlEvidence === branchEvidence) refuse("divergence evidence is identical");
    string(first.details, "divergence details");
    divergence = { tick: tick(first.tick, "first divergence tick", applied + 1, high),
      stage: string(first.stage, "digest lane", 128), controlEvidence, branchEvidence };
  }

  if (!object(value.effect_sizes) || Object.keys(value.effect_sizes).some((key) => !EFFECT_METRICS.includes(key))) {
    refuse("unsupported trajectory comparison");
  }
  if (isControl && Object.keys(value.effect_sizes).length) refuse("control declares a treatment comparison");
  const effects = [];
  for (const key of EFFECT_METRICS) {
    if (!Object.hasOwn(value.effect_sizes, key)) continue;
    const effect = value.effect_sizes[key];
    shape(effect, ["metric", "analysis_basis", "n_samples", "mean_diff", "sd_diff"],
      ["cohens_d", "hedges_g", "ci_95", "p_value"], "effect");
    shape(effect.analysis_basis, ["kind", "seed"], [], "analysis basis");
    if (effect.metric !== key || effect.analysis_basis.kind !== "single_seed_trajectory"
        || seed(effect.analysis_basis.seed, "comparison seed") !== rootSeed
        || ["cohens_d", "hedges_g", "ci_95", "p_value"].some((field) => effect[field] != null)) {
      refuse("a saved trajectory cannot supply independent-seed statistical inference");
    }
    const observations = count(effect.n_samples, "paired observation count");
    if (observations !== high - low + 1) refuse("comparison observation count differs from the recorded horizon");
    effects.push({ metric: key, label: METRICS.find(([metric]) => metric === key)[1], observations,
      meanDifference: finite(effect.mean_diff, "mean trajectory difference"),
      standardDeviation: finite(effect.sd_diff, "trajectory difference standard deviation", 0) });
  }
  if (value.bundle_path !== null) string(value.bundle_path, "bundle path", 8192);
  if (value.bundle_verified !== true) refuse("managed branch bundle verification is absent");
  return { id: value.branch_id, label: isControl ? "Control" : "Treatment", interventionTick: applied,
    terminal, minimumPopulation: minimum, firstZeroTick: firstZero, baselineTick, baselinePopulation,
    troughTick, recoveryRatio: baselinePopulation === 0 ? null : ratio, divergence, effects };
}

/** Parse only a verified download associated with the currently selected completed receipt.
 * The caller must retain its source-generation guard across this asynchronous operation.
 */
export async function parseSavedStudyReport(blob, context) {
  const status = contextIdentity(context);
  // History entries do not retain the original browser request. When available,
  // capture its expected arm and application boundary before reading any bytes.
  let expectedTreatmentTick;
  if (context.request != null) {
    const request = context.request;
    if (!object(request)) refuse("submitted request differs from the retained study");
    const delay = request.intervention_delay_ticks === undefined ? 0 : request.intervention_delay_ticks;
    if (request.checkpoint_id !== status.checkpoint_id
        || typeof request.branch_name !== "string" || request.branch_name !== status.branch_name
        || request.ticks !== status.target_tick - status.source_tick
        || !Object.hasOwn(request, "intervention")
        || (request.intervention !== null && !object(request.intervention))
        || !safeInteger(delay) || delay >= request.ticks
        || (request.intervention === null && delay !== 0)) {
      refuse("submitted request differs from the retained study");
    }
    expectedTreatmentTick = request.intervention === null ? null : status.source_tick + delay;
  }
  if (!(blob instanceof Blob) || blob.size < 1 || blob.size > MAX_STUDY_REPORT_BYTES) refuse("JSON exceeds the 1 MiB report preview bound");
  const buffer = await blob.arrayBuffer();
  if (buffer.byteLength !== blob.size || buffer.byteLength > MAX_STUDY_REPORT_BYTES) refuse("report byte count changed");
  let decoded;
  try { decoded = new TextDecoder("utf-8", { fatal: true }).decode(buffer); }
  catch (_) { refuse("report is not valid UTF-8"); }
  const report = readJson(decoded);
  shape(report, ["study_id", "schema_version", "base_seed", "source", "checkpoint_tick", "horizon_ticks",
    "pre_intervention_digest", "pre_intervention_digest_verified_for_all_branches", "control_branch_id",
    "branch_reports", "summary_markdown"], [], "report");
  if (count(report.schema_version, "schema version") !== 4) refuse("unsupported schema; this viewer requires schema 4");
  if (report.study_id !== status.branch_id || report.control_branch_id !== "control"
      || count(report.checkpoint_tick, "saved tick") !== status.source_tick
      || count(report.horizon_ticks, "horizon") !== status.target_tick - status.source_tick
      || worldDigest(report.pre_intervention_digest, "initial digest") !== status.world_digest_v1
      || report.pre_intervention_digest_verified_for_all_branches !== true) refuse("report does not match the selected saved-world study");
  const rootSeed = seed(report.base_seed, "starting seed");
  shape(report.source, ["kind", "artifact_path", "byte_size", "checksum_blake3"], [], "saved source");
  const sourceBytes = count(report.source.byte_size, "source byte count");
  if (report.source.kind !== "saved_checkpoint" || report.source.artifact_path !== "source_checkpoint.bin" || sourceBytes < 1) {
    refuse("report is not a retained saved-checkpoint study");
  }
  const sourceChecksum = checksum(report.source.checksum_blake3, "saved source checksum");
  const checkpoint = context.checkpoint;
  if (checkpoint != null && (!object(checkpoint) || checkpoint.checkpoint_id !== status.checkpoint_id
      || checkpoint.tick !== status.source_tick || checkpoint.world_digest_v1 !== status.world_digest_v1
      || !safeInteger(checkpoint.agent_count) || checkpoint.byte_size !== sourceBytes
      || checkpoint.checksum_blake3 !== sourceChecksum)) refuse("report source differs from the selected checkpoint");
  if (!Array.isArray(report.branch_reports) || report.branch_reports.length < 1 || report.branch_reports.length > 2) refuse("managed study needs one control and at most one treatment");
  const arms = report.branch_reports.map((arm) => readArm(arm, status, rootSeed));
  const control = arms.find((arm) => arm.id === "control"), treatment = arms.find((arm) => arm.id === "treatment");
  if (!control || new Set(arms.map((arm) => arm.id)).size !== arms.length) refuse("missing or duplicate study arm");
  if (expectedTreatmentTick !== undefined && (treatment?.interventionTick ?? null) !== expectedTreatmentTick) {
    refuse("report treatment and application boundary differ from the submitted request");
  }
  if (checkpoint != null && (control.baselinePopulation !== checkpoint.agent_count
      || (treatment?.interventionTick === status.source_tick && treatment.baselinePopulation !== checkpoint.agent_count))) {
    refuse("study population baseline differs from the selected checkpoint");
  }
  if (treatment && !treatment.divergence && (treatment.terminal.digest !== control.terminal.digest
      || METRICS.some(([key]) => treatment.terminal[key] !== control.terminal[key])
      || treatment.effects.some((effect) => effect.meanDifference !== 0 || effect.standardDeviation !== 0))) {
    refuse("changed outcomes lack a reported first divergence");
  }
  if (typeof report.summary_markdown !== "string" || report.summary_markdown.length > 128 * 1024) refuse("summary exceeds its bound");
  let artifact = null;
  if (context.artifact != null) {
    const value = context.artifact;
    if (!object(value) || value.artifact_id !== status.artifact_id || value.byte_size !== blob.size
        || typeof value.relative_path !== "string" || !value.relative_path.startsWith("branches/" + status.branch_id + "/")) {
      refuse("report download differs from the selected artifact receipt");
    }
    artifact = { id: value.artifact_id, byteSize: value.byte_size,
      checksumSha256: checksum(value.checksum_sha256, "downloaded report checksum") };
  }
  const endpoints = METRICS.map(([metric, label, integral]) => ({ metric, label, integral,
    control: control.terminal[metric], treatment: treatment ? treatment.terminal[metric] : null,
    difference: treatment ? treatment.terminal[metric] - control.terminal[metric] : null }));
  return freeze({ studyId: status.branch_id, checkpointId: status.checkpoint_id, seed: rootSeed,
    sourceTick: status.source_tick, targetTick: status.target_tick,
    horizonTicks: status.target_tick - status.source_tick, worldDigest: status.world_digest_v1,
    source: { byteSize: sourceBytes, checksumBlake3: sourceChecksum, artifactPath: report.source.artifact_path },
    artifact, arms: treatment ? [control, treatment] : [control], endpoints,
    trajectoryComparisons: treatment?.effects ?? [], hasTreatment: Boolean(treatment) });
}

function displayNumber(value, integral = false, signed = false) {
  const prefix = signed && value > 0 ? "+" : "";
  return prefix + new Intl.NumberFormat(undefined, integral
    ? { maximumFractionDigits: 0 } : { maximumSignificantDigits: 7 }).format(value);
}

/** Render bounded plain text and tables; report Markdown and filesystem paths are never executed. */
export function renderStudyReport(container, model) {
  if (!container) return;
  if (!model) { container.replaceChildren(); container.hidden = true; return; }
  const doc = container.ownerDocument;
  const element = (tag, text) => {
    const node = doc.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const table = (caption, headings, rows) => {
    const node = element("table");
    node.append(element("caption", caption));
    const head = element("thead"), heading = element("tr");
    for (const text of headings) { const cell = element("th", text); cell.scope = "col"; heading.append(cell); }
    head.append(heading);
    const body = element("tbody");
    for (const row of rows) {
      const tr = element("tr");
      row.forEach((value, index) => { const cell = element(index === 0 ? "th" : "td", value);
        if (index === 0) cell.scope = "row"; tr.append(cell); });
      body.append(tr);
    }
    node.append(head, body);
    return node;
  };
  const nodes = [
    element("h3", "Observed study outcomes"),
    element("p", "Saved tick " + model.sourceTick + " → observed terminal tick " + model.targetTick
      + " (" + model.horizonTicks + " future ticks). Starting seed " + model.seed + "."),
    element("p", model.hasTreatment
      ? "Control and treatment share one saved starting state. Differences describe this seed's trajectories; independent-seed confidence intervals and significance tests are unavailable."
      : "This report contains one unchanged future. Variability across independent starting seeds is not estimated."),
  ];
  const headers = model.hasTreatment ? ["Metric", "Control", "Treatment", "Δ treatment − control"] : ["Metric", "Control"];
  const rows = model.endpoints.map((row) => [row.label, displayNumber(row.control, row.integral),
    ...(model.hasTreatment ? [displayNumber(row.treatment, row.integral), displayNumber(row.difference, row.integral, true)] : [])]);
  nodes.push(table("Terminal boundary observations", headers, rows));
  nodes.push(element("p", "Births, deaths and spike hits describe the final tick only. Decimal values are rounded for display; the downloadable report retains full values."));
  if (model.trajectoryComparisons.length) {
    nodes.push(table("Recorded differences across the whole trajectory",
      ["Metric", "Mean treatment − control", "Standard deviation of differences", "Paired timepoints"],
      model.trajectoryComparisons.map((row) => [row.label, displayNumber(row.meanDifference, false, true),
        displayNumber(row.standardDeviation), displayNumber(row.observations, true)])));
    nodes.push(element("p", "Paired timepoints include the saved boundary and every observed tick. These correlated observations represent one starting seed."));
  }
  for (const arm of model.arms) {
    nodes.push(element("h4", arm.label + " trajectory"));
    nodes.push(element("p", arm.firstZeroTick === null
      ? "Population stayed above zero; minimum " + arm.minimumPopulation + " agents."
      : "Population reached zero at tick " + arm.firstZeroTick + "; the final population is " + arm.terminal.population + "."));
    nodes.push(element("p", "Baseline " + arm.baselinePopulation + " agents at tick " + arm.baselineTick
      + "; trough " + arm.minimumPopulation + " at tick " + arm.troughTick + ". "
      + (arm.recoveryRatio === null ? "A recovery percentage is undefined for a zero-population baseline."
        : "Final population is " + displayNumber(100 * arm.recoveryRatio) + "% of that baseline.")));
    if (arm.id === "treatment") {
      nodes.push(element("p", "Intervention boundary: tick " + arm.interventionTick + ". "
        + (arm.divergence ? "First observed digest difference: tick " + arm.divergence.tick + ", lane " + arm.divergence.stage + "."
          : "No digest difference was observed during this horizon.")));
    }
    nodes.push(element("p", "Terminal world digest: " + arm.terminal.digest));
  }
  nodes.push(element("h4", "Exact source provenance"));
  nodes.push(element("p", "Study " + model.studyId + "; checkpoint " + model.checkpointId + "."));
  nodes.push(element("p", "Shared initial world digest: " + model.worldDigest));
  nodes.push(element("p", "Saved source: " + model.source.byteSize + " bytes; BLAKE3 " + model.source.checksumBlake3 + "."));
  if (model.artifact) nodes.push(element("p", "Downloaded report " + model.artifact.id + ": "
    + model.artifact.byteSize + " bytes; SHA-256 " + model.artifact.checksumSha256 + "."));
  nodes.push(element("p", "Bundle verification establishes retained artifact integrity. A saved scientific study does not restore the source host's session or scheduled commands."));
  container.replaceChildren(...nodes);
  container.hidden = false;
}

const trajectoryViews = new WeakMap();

/** Inspect recorded timepoints; chart interactions never mutate or seek a live world. */
export function renderStudyTrajectories(container, model) {
  if (!container) return;
  if (!model) {
    trajectoryViews.delete(container);
    container.replaceChildren();
    container.hidden = true;
    return;
  }
  if (trajectoryViews.get(container) === model) return;
  const doc = container.ownerDocument;
  const element = (tag, text) => {
    const node = doc.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const svg = (tag, attributes = {}, text) => {
    const node = doc.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const button = (text, action) => {
    const node = element("button", text);
    node.type = "button";
    node.addEventListener("click", action);
    return node;
  };
  const number = (value, integral = false, signed = false) =>
    (signed && value > 0 ? "+" : "") + new Intl.NumberFormat(undefined, integral
      ? { maximumFractionDigits: 0 } : { minimumFractionDigits: 4, maximumFractionDigits: 4 }).format(value);
  const axisNumber = (value) => new Intl.NumberFormat(undefined, {
    notation: value >= 100000 || (value > 0 && value < 0.001) ? "scientific" : "standard",
    maximumSignificantDigits: 4,
  }).format(value);
  const metricLabel = element("label", "Compare a metric");
  const metricSelect = element("select");
  metricSelect.setAttribute("aria-label", "Trajectory metric");
  for (const metric of STUDY_TRAJECTORY_METRICS) {
    const option = element("option", metric.label);
    option.value = metric.key;
    metricSelect.append(option);
  }
  metricSelect.value = STUDY_TRAJECTORY_METRICS[0].key;
  metricLabel.append(metricSelect);
  const chart = svg("svg", { viewBox: "0 0 640 264", role: "img", class: "trajectory-chart" });
  const legend = element("p", model.hasTreatment
    ? "Control: solid blue · Treatment: dashed orange · Intervention: vertical amber line"
    : "Control: solid blue");
  const tickLabel = element("p");
  tickLabel.setAttribute("aria-live", "polite");
  const slider = element("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = String(model.horizonTicks);
  slider.step = "1";
  slider.setAttribute("aria-label", "Recorded tick offset");
  const tickInput = element("input");
  tickInput.type = "number";
  tickInput.min = String(model.sourceTick);
  tickInput.max = String(model.targetTick);
  tickInput.step = "1";
  tickInput.setAttribute("aria-label", "Recorded tick");
  const tickControls = element("div");
  tickControls.className = "trajectory-tick-controls";
  let selectedIndex = model.horizonTicks, selectedMetric = metricSelect.value;
  let selectionLine, selectionDots = [], yPosition;
  const xPosition = (offset) => 58 + (offset / model.horizonTicks) * 562;
  const previous = button("Previous tick", () => inspect(selectedIndex - 1));
  const next = button("Next tick", () => inspect(selectedIndex + 1));
  tickControls.append(previous, tickInput, next);
  const table = element("table");
  table.append(element("caption", "Recorded values at the selected tick"));
  const head = element("thead"), headings = element("tr");
  for (const text of ["Metric", "Control", ...(model.hasTreatment ? ["Treatment", "Δ treatment − control"] : [])]) {
    const cell = element("th", text);
    cell.scope = "col";
    headings.append(cell);
  }
  head.append(headings);
  const body = element("tbody");
  table.append(head, body);
  const digests = element("p");
  const inspect = (index) => {
    if (!Number.isSafeInteger(index) || index < 0 || index > model.horizonTicks) return;
    selectedIndex = index;
    const tick = model.sourceTick + index;
    slider.value = String(index);
    tickInput.value = String(tick);
    tickInput.setCustomValidity("");
    previous.disabled = index === 0;
    next.disabled = index === model.horizonTicks;
    tickLabel.textContent = "Recorded tick " + tick + " · " + index + " ticks after the saved source"
      + (tick === model.interventionTick ? " · intervention boundary" : "");
    selectionLine.setAttribute("x1", String(xPosition(index)));
    selectionLine.setAttribute("x2", String(xPosition(index)));
    selectionDots.forEach((dot, armIndex) => {
      dot.setAttribute("cx", String(xPosition(index)));
      dot.setAttribute("cy", String(yPosition(model.arms[armIndex].samples[index][selectedMetric])));
    });
    body.replaceChildren(...STUDY_TRAJECTORY_METRICS.map((metric) => {
      const control = model.arms[0].samples[index][metric.key];
      const treatment = model.hasTreatment ? model.arms[1].samples[index][metric.key] : null;
      const row = element("tr"), title = element("th", metric.label);
      title.scope = "row";
      row.append(title, element("td", number(control, metric.integral)));
      if (model.hasTreatment) row.append(element("td", number(treatment, metric.integral)),
        element("td", number(treatment - control, metric.integral, true)));
      return row;
    }));
    digests.textContent = model.arms.map((arm) => arm.label + " digest: " + arm.samples[index].digest).join(" · ");
  };
  const draw = () => {
    const metric = STUDY_TRAJECTORY_METRICS.find((item) => item.key === selectedMetric);
    let maximum = 0;
    for (const arm of model.arms) for (const sample of arm.samples) maximum = Math.max(maximum, sample[selectedMetric]);
    const scale = maximum || 1;
    yPosition = (value) => 222 - (value / scale) * 196;
    const nodes = [svg("title", {}, metric.label + " from tick " + model.sourceTick + " to " + model.targetTick)];
    chart.setAttribute("aria-label", metric.label + " recorded at every tick; exact values follow the chart");
    for (const fraction of [0, 0.5, 1]) {
      const y = yPosition(scale * fraction);
      nodes.push(svg("line", { x1: 58, x2: 620, y1: y, y2: y, stroke: "#334155" }),
        svg("text", { x: 50, y: y + 4, "text-anchor": "end", fill: "#cbd5e1", "font-size": 12 }, axisNumber(scale * fraction)));
    }
    nodes.push(svg("text", { x: 58, y: 247, fill: "#cbd5e1", "font-size": 12 }, String(model.sourceTick)),
      svg("text", { x: 620, y: 247, "text-anchor": "end", fill: "#cbd5e1", "font-size": 12 }, String(model.targetTick)));
    model.arms.forEach((arm, index) => {
      // The bounded study has at most 10,001 points; retain all peaks and troughs.
      const path = arm.samples.map((sample, offset) => (offset === 0 ? "M" : "L")
        + xPosition(offset).toFixed(2) + " " + yPosition(sample[selectedMetric]).toFixed(2)).join(" ");
      nodes.push(svg("path", { d: path, fill: "none", stroke: index === 0 ? "#38bdf8" : "#fb923c",
        "stroke-width": 2, "stroke-dasharray": index === 0 ? "none" : "6 4", class: "trajectory-curve" }));
    });
    if (model.interventionTick !== null) {
      const x = xPosition(model.interventionTick - model.sourceTick);
      nodes.push(svg("line", { x1: x, x2: x, y1: 26, y2: 222, stroke: "#fbbf24",
        "stroke-dasharray": "3 4", class: "trajectory-intervention" }));
    }
    selectionLine = svg("line", { y1: 26, y2: 222, stroke: "#e2e8f0", "stroke-opacity": 0.5,
      class: "trajectory-selected-tick" });
    selectionDots = model.arms.map((_, index) => svg("circle", { r: 4, fill: index === 0 ? "#38bdf8" : "#fb923c" }));
    nodes.push(selectionLine, ...selectionDots);
    chart.replaceChildren(...nodes);
    inspect(selectedIndex);
  };
  metricSelect.addEventListener("change", () => {
    if (!STUDY_TRAJECTORY_METRICS.some((metric) => metric.key === metricSelect.value)) {
      metricSelect.value = selectedMetric;
      return;
    }
    selectedMetric = metricSelect.value;
    draw();
  });
  slider.addEventListener("input", () => inspect(Number(slider.value)));
  tickInput.addEventListener("change", () => {
    const tick = /^(0|[1-9][0-9]*)$/.test(tickInput.value) ? Number(tickInput.value) : NaN;
    if (!Number.isSafeInteger(tick) || tick < model.sourceTick || tick > model.targetTick) {
      tickInput.setCustomValidity("Choose a whole recorded tick from " + model.sourceTick + " through " + model.targetTick);
      return;
    }
    inspect(tick - model.sourceTick);
  });
  const controls = element("div");
  controls.append(tickLabel, slider, tickControls);
  if (model.interventionTick !== null) controls.append(button("Intervention tick",
    () => inspect(model.interventionTick - model.sourceTick)));
  const provenance = element("details");
  provenance.append(element("summary", "Recorded data"));
  for (const arm of model.arms) provenance.append(element("p", arm.label + ": " + arm.artifact.relativePath
    + " · " + arm.artifact.byteSize + " bytes · SHA-256 " + arm.artifact.checksumSha256));
  draw();
  container.replaceChildren(element("h3", "Explore recorded trajectories"), metricLabel, legend, chart,
    controls, table, digests,
    element("p", "Every completed tick is shown. Decimal CSV values were recorded to four decimal places; differences use those recorded values. These trajectories share one saved source and do not estimate variation across independent seeds."),
    provenance);
  container.hidden = false;
  trajectoryViews.set(container, model);
}
