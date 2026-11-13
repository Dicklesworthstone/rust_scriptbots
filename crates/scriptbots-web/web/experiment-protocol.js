// Browser boundaries for the existing matched-seed experiment owner. These
// checks validate observations; they never run science or recalculate inference.
export const EXPERIMENT_LIMITS = Object.freeze({
  maxVariants: 8,
  maxSeeds: 32,
  maxRuns: 256,
  maxConcurrency: 8,
  maxTicks: 10_000_000,
  maxRequestBytes: 64 * 1024,
  maxResponseBytes: 1024 * 1024,
  maxStatusRuns: 4096,
  maxStatusVariants: 64,
  maxStatusSeeds: 128,
  maxCatalogRows: 100,
  maxCatalogExperiments: 1024,
  maxInterventions: 64,
  maxInterventionBytes: 4096,
});

export const EXPERIMENT_ENDPOINT_METRICS = Object.freeze([
  "alive_agents", "total_energy", "average_energy", "average_health",
  "mean_age", "max_age", "max_generation", "total_food",
]);

const MAX_JSON_NODES = 100_000;
const MAX_JSON_DEPTH = 24;
const U64_MAX = (1n << 64n) - 1n;
const DIGEST = /^[0-9a-f]{64}$/;
const WORLD_DIGEST = /^[0-9a-f]{16}$/;
const STATES = ["pending", "running", "cancelling", "cancelled", "stopped", "completed", "failed"];
const FAMILIES = Object.freeze({
  mlp: "mlp.baseline", "mlp.baseline": "mlp.baseline",
  dwraon: "dwraon.baseline", "dwraon.baseline": "dwraon.baseline",
  assembly: "assembly.experimental", "assembly.experimental": "assembly.experimental",
  neuro: "ml.neuroflow", neuroflow: "ml.neuroflow", "ml.neuroflow": "ml.neuroflow",
  ft: "ft", frankentorch: "ft",
});
const CANONICAL_FAMILIES = new Set(Object.values(FAMILIES));
const utf8 = new TextEncoder();

function refuse(message) { throw new Error("Experiment refused: " + message); }
function object(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    && [Object.prototype, null].includes(Object.getPrototypeOf(value));
}
function shape(value, required, optional = [], label = "record") {
  if (!object(value) || required.some((key) => !Object.hasOwn(value, key))
      || Object.keys(value).some((key) => !required.includes(key) && !optional.includes(key))) {
    refuse(label + " has missing or unsupported fields");
  }
}
function freeze(value) {
  if (value !== null && typeof value === "object") {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}
function text(value, label, maximum = 4096, allowEmpty = false) {
  if (typeof value !== "string" || (!allowEmpty && !value.trim())
      || value.length > maximum || utf8.encode(value).length > maximum) refuse(label + " is invalid or too long");
  return value;
}
function nullableText(value, label) { return value === null ? null : text(value, label); }
function integer(value, label, low = 0, high = Number.MAX_SAFE_INTEGER) {
  if (!Number.isSafeInteger(value) || value < low || value > high) refuse(label + " is not an exact bounded integer");
  return value;
}
function finite(value, label, low = -Infinity, high = Infinity) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < low || value > high) refuse(label + " is not a finite bounded number");
  return value;
}
function boolean(value, label) {
  if (typeof value !== "boolean") refuse(label + " is not a boolean");
  return value;
}
function digest(value, label) {
  if (typeof value !== "string" || !DIGEST.test(value)) refuse(label + " is not a canonical digest");
  return value;
}
function worldDigest(value, label) {
  if (typeof value !== "string" || !WORLD_DIGEST.test(value)) refuse(label + " is not a canonical scientific digest");
  return value;
}
function unsigned(value, label) {
  if (typeof value === "number") {
    integer(value, label);
    value = String(value);
  } else if (typeof value === "bigint") value = value.toString();
  if (typeof value !== "string" || !/^(0|[1-9][0-9]{0,19})$/.test(value)
      || BigInt(value) > U64_MAX) refuse(label + " is not an exact u64 decimal integer");
  return value;
}
function timestamp(value, label) {
  if (typeof value !== "string") refuse(label + " is not an epoch-seconds string");
  return unsigned(value, label);
}
function family(value) {
  if (typeof value !== "string" || !Object.hasOwn(FAMILIES, value)) refuse("unknown experiment brain family");
  return FAMILIES[value];
}
function canonicalFamily(value) {
  if (!CANONICAL_FAMILIES.has(value)) refuse("run does not name a canonical installed-family identity");
  return value;
}
export function validateExperimentIdentifier(value, label = "experiment identity", maximum = 128) {
  if (typeof value !== "string" || value.length > maximum
      || !/^[A-Za-z0-9_.-]+$/.test(value) || value === "." || value.includes("..")) refuse(label + " is invalid");
  return value;
}

// A reviver cannot undo JSON.parse's rounding of a full-width u64. This bounded
// reader retains only seed lanes as decimal strings and refuses unsafe integers
// elsewhere. Duplicate fields cannot replace identities before validation.
export function parseExperimentJson(source) {
  if (typeof source !== "string" || source.length > EXPERIMENT_LIMITS.maxResponseBytes
      || utf8.encode(source).length > EXPERIMENT_LIMITS.maxResponseBytes) refuse("response exceeds the 1 MiB browser bound");
  let position = 0, nodes = 0;
  const number = /-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/y;
  const whitespace = () => {
    while (position < source.length && " \t\r\n".includes(source[position])) position += 1;
  };
  const quoted = () => {
    const start = position++;
    while (position < source.length) {
      const next = source[position++];
      if (next === "\\") { position += 1; continue; }
      if (next === '"') {
        try { return JSON.parse(source.slice(start, position)); }
        catch (_) { refuse("invalid JSON string"); }
      }
    }
    refuse("unterminated JSON string");
  };
  const read = (depth, isSeed = false, seedArray = false) => {
    whitespace();
    if (++nodes > MAX_JSON_NODES || depth > MAX_JSON_DEPTH) refuse("JSON structure exceeds the browser bound");
    const next = source[position];
    if (isSeed && !(next === "0" || (next >= "1" && next <= "9"))) refuse("seed wire values must be unsigned JSON integers");
    if (next === '"') return quoted();
    if (next === "{") {
      position += 1;
      const result = Object.create(null);
      whitespace();
      if (source[position] === "}") { position += 1; return result; }
      for (;;) {
        whitespace();
        if (source[position] !== '"') refuse("invalid JSON object key");
        const key = quoted();
        if (["__proto__", "prototype", "constructor"].includes(key)) refuse("unsupported JSON object key");
        if (Object.hasOwn(result, key)) refuse("duplicate JSON field " + key.slice(0, 128));
        whitespace();
        if (source[position++] !== ":") refuse("missing JSON colon");
        result[key] = read(depth + 1, key === "seed", key === "matched_seeds" || key === "seeds");
        whitespace();
        const separator = source[position++];
        if (separator === "}") return result;
        if (separator !== ",") refuse("invalid JSON object separator");
      }
    }
    if (next === "[") {
      position += 1;
      const result = [];
      whitespace();
      if (source[position] === "]") { position += 1; return result; }
      for (;;) {
        result.push(read(depth + 1, seedArray));
        whitespace();
        const separator = source[position++];
        if (separator === "]") return result;
        if (separator !== ",") refuse("invalid JSON array separator");
      }
    }
    for (const [token, value] of [["true", true], ["false", false], ["null", null]]) {
      if (source.startsWith(token, position)) { position += token.length; return value; }
    }
    number.lastIndex = position;
    const matched = number.exec(source);
    if (!matched) refuse("invalid JSON value");
    position = number.lastIndex;
    if (isSeed) return unsigned(matched[0], "seed");
    const value = Number(matched[0]);
    if (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value))) refuse("nonfinite or inexact JSON number outside a seed field");
    return value;
  };
  const result = read(0);
  whitespace();
  if (position !== source.length) refuse("unexpected data after JSON");
  return result;
}

// Config semantics remain with the owner. The browser only admits bounded,
// finite JSON and preserves integer-vs-float values without unsafe coercions.
function configObject(value, label, numericOnly = false) {
  if (!object(value)) refuse(label + " must be an object");
  let nodes = 0;
  const visit = (entry, depth) => {
    if (++nodes > 2048 || depth > 12) refuse(label + " exceeds the configuration bound");
    if (typeof entry === "number") {
      finite(entry, label);
      if (Number.isInteger(entry) && !Number.isSafeInteger(entry)) refuse(label + " contains an inexact integer");
      return entry;
    }
    if (object(entry)) {
      const keys = Object.keys(entry).sort();
      if (numericOnly && !keys.length) refuse(label + " contains an empty numeric patch");
      const result = {};
      for (const key of keys) {
        if (!/^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$/.test(key)
            || key.length > 128 || key.split(".").some((part) => ["__proto__", "prototype", "constructor"].includes(part))) refuse(label + " contains an invalid config path");
        result[key] = visit(entry[key], depth + 1);
      }
      return result;
    }
    if (numericOnly) refuse(label + " may contain only numeric live patches");
    if (entry === null || typeof entry === "boolean") return entry;
    if (typeof entry === "string") return text(entry, label, 4096, true);
    if (Array.isArray(entry) && entry.length <= 256) return entry.map((item) => visit(item, depth + 1));
    refuse(label + " contains an unsupported JSON value");
  };
  const result = visit(value, 0);
  if (utf8.encode(JSON.stringify(result)).length > (numericOnly ? EXPERIMENT_LIMITS.maxInterventionBytes : 16 * 1024)) refuse(label + " exceeds its byte bound");
  return result;
}

// Prepare once, retain after uncertain admission, and retry these exact bytes.
export function prepareExperimentRequest(input) {
  shape(input, ["variants", "seeds", "ticks_per_run", "max_concurrency", "idempotency_key"],
    ["experiment_id", "description"], "experiment request");
  const experimentId = input.experiment_id == null ? null : validateExperimentIdentifier(input.experiment_id);
  const description = input.description == null ? null : text(input.description, "description", 2048, true);
  const ticks = integer(input.ticks_per_run, "run horizon", 1, EXPERIMENT_LIMITS.maxTicks);
  const concurrency = integer(input.max_concurrency, "concurrency", 1, EXPERIMENT_LIMITS.maxConcurrency);
  const key = text(input.idempotency_key, "retry identity", 1024);
  if (/[\u0000-\u001f\u007f]/.test(key)) refuse("retry identity contains control characters");
  if (!Array.isArray(input.variants) || input.variants.length < 2 || input.variants.length > EXPERIMENT_LIMITS.maxVariants) refuse("create between 2 and 8 distinct experiment arms");
  if (!Array.isArray(input.seeds) || !input.seeds.length || input.seeds.length > EXPERIMENT_LIMITS.maxSeeds) refuse("provide between 1 and 32 explicit independent seeds");
  const seeds = input.seeds.map((value) => unsigned(value, "seed"));
  if (new Set(seeds).size !== seeds.length) refuse("duplicate seeds are not independent replicates");
  if (seeds.length * input.variants.length > EXPERIMENT_LIMITS.maxRuns) refuse("experiment exceeds the browser run bound");
  const identifiers = new Set(), arms = new Set();
  const variants = input.variants.map((variant) => {
    shape(variant, ["variant_id", "brain_family"], ["config_overrides", "interventions"], "experiment arm");
    const id = variant.variant_id;
    if (typeof id !== "string" || !/^[A-Za-z0-9_-]{1,64}$/.test(id) || identifiers.has(id)) refuse("arm identities must be unique ASCII names of at most 64 characters");
    identifiers.add(id);
    const brain = family(variant.brain_family);
    const config = configObject(variant.config_overrides ?? {}, "arm config");
    const schedule = variant.interventions ?? [];
    if (!Array.isArray(schedule) || schedule.length > EXPERIMENT_LIMITS.maxInterventions) refuse("intervention schedule exceeds 64 statements");
    let previous = 0;
    const interventions = schedule.map((statement) => {
      shape(statement, ["tick", "set"], [], "intervention");
      const tick = integer(statement.tick, "intervention tick", previous, ticks - 1);
      previous = tick;
      const set = configObject(statement.set, "intervention patch", true);
      return { tick, set };
    });
    const semantic = JSON.stringify([brain, config, interventions]);
    if (arms.has(semantic)) refuse("identical declared arms do not define a comparison");
    arms.add(semantic);
    if (experimentId !== null) {
      for (const seed of seeds) validateExperimentIdentifier(`${experimentId}-${id}-seed${seed}`, "derived run identity");
    }
    return { variant_id: id, brain_family: brain, config_overrides: config, interventions };
  });
  const request = { experiment_id: experimentId, description, variants, seeds,
    ticks_per_run: ticks, max_concurrency: concurrency, idempotency_key: key };
  const body = "{" + Object.entries(request).map(([name, value]) => JSON.stringify(name) + ":"
    + (name === "seeds" ? "[" + seeds.join(",") + "]" : JSON.stringify(value))).join(",") + "}";
  if (utf8.encode(body).length > EXPERIMENT_LIMITS.maxRequestBytes) refuse("creation request exceeds the 64 KiB browser bound");
  return freeze({ request, body });
}

function batchSummary(value) {
  const id = validateExperimentIdentifier(value.experiment_id);
  if (!STATES.includes(value.status)) refuse("unsupported experiment lifecycle state");
  const total = integer(value.total_runs, "total runs", 1, EXPERIMENT_LIMITS.maxStatusRuns);
  const completed = integer(value.completed_runs, "completed runs", 0, total);
  const failed = integer(value.failed_runs, "failed runs", 0, total - completed);
  if ((value.status === "completed" && completed !== total)
      || (value.status === "pending" && completed + failed !== 0)
      || (["stopped", "cancelled"].includes(value.status) && completed + failed === total)) refuse("lifecycle label contradicts the run counts");
  return { experiment_id: id, status: value.status, total_runs: total, completed_runs: completed,
    failed_runs: failed, created_at_utc: timestamp(value.created_at_utc, "creation time"),
    updated_at_utc: timestamp(value.updated_at_utc, "update time") };
}

export function validateExperimentCatalog(value, context = {}) {
  shape(value, ["items", "total", "limit", "offset", "has_more", "execution_capacity"], ["next_cursor"], "experiment catalog");
  const total = integer(value.total, "catalog size", 0, EXPERIMENT_LIMITS.maxCatalogExperiments);
  const limit = integer(value.limit, "catalog page limit", 1, EXPERIMENT_LIMITS.maxCatalogRows);
  const offset = integer(value.offset, "catalog offset");
  if (context.limit !== undefined && limit !== context.limit) refuse("catalog page limit changed");
  if (Object.hasOwn(context, "cursor")) {
    const cursor = context.cursor == null ? "0" : context.cursor;
    if (typeof cursor !== "string" || !/^(0|[1-9][0-9]*)$/.test(cursor)
        || !Number.isSafeInteger(Number(cursor)) || offset !== Number(cursor)) refuse("catalog offset does not match the requested cursor");
  }
  if (!Array.isArray(value.items) || value.items.length !== Math.min(limit, Math.max(0, total - offset))) refuse("catalog page is incomplete or exceeds its bound");
  const ids = new Set();
  const items = value.items.map((item) => {
    shape(item, ["experiment_id", "status", "total_runs", "completed_runs", "failed_runs", "created_at_utc", "updated_at_utc"], [], "catalog item");
    const row = batchSummary(item);
    if (ids.has(row.experiment_id)) refuse("catalog repeats an experiment identity");
    ids.add(row.experiment_id);
    return row;
  });
  const more = offset + items.length < total;
  if (boolean(value.has_more, "catalog continuation") !== more
      || (value.next_cursor ?? null) !== (more ? String(offset + items.length) : null)) refuse("catalog continuation disagrees with its rows");
  const capacity = value.execution_capacity;
  shape(capacity, ["max_run_slots", "reserved_run_slots", "max_grid_cells", "reserved_grid_cells", "reserved_batches"], [], "execution capacity");
  const maxSlots = integer(capacity.max_run_slots, "maximum run slots", 0);
  const maxCells = integer(capacity.max_grid_cells, "maximum grid cells", 1, 1_048_576);
  return freeze({ items, total, limit, offset, next_cursor: more ? String(offset + items.length) : null,
    has_more: more, execution_capacity: {
      max_run_slots: maxSlots, reserved_run_slots: integer(capacity.reserved_run_slots, "reserved run slots", 0, maxSlots),
      max_grid_cells: maxCells, reserved_grid_cells: integer(capacity.reserved_grid_cells, "reserved grid cells", 0, maxCells),
      reserved_batches: integer(capacity.reserved_batches, "reserved batches", 0, EXPERIMENT_LIMITS.maxCatalogExperiments),
    } });
}

function checkpointProgress(value, horizon) {
  if (!object(value)) refuse("checkpoint progress is missing");
  if (value.state === "observed") {
    shape(value, ["state", "tick", "checkpoint_digest", "source"], [], "observed checkpoint");
    if (!["published", "recovered"].includes(value.source)) refuse("checkpoint observation source is invalid");
    return { state: "observed", tick: integer(value.tick, "saved checkpoint tick", 0, horizon),
      checkpoint_digest: digest(value.checkpoint_digest, "checkpoint bytes digest"), source: value.source };
  }
  shape(value, ["state", "reason"], [], "checkpoint availability");
  const reasons = value.state === "unavailable"
    ? ["not_started", "awaiting_publication", "recovery_pending", "no_committed_checkpoint"]
    : value.state === "refused" ? ["invalid_checkpoint", "unreadable_checkpoint", "conflicting_observation"] : [];
  if (!reasons.includes(value.reason)) refuse("checkpoint availability is unsupported");
  return { state: value.state, reason: value.reason };
}

function runRecord(value, experimentId) {
  shape(value, ["run_id", "variant_id", "brain_family", "seed", "state", "total_ticks", "target_ticks",
    "checkpoint_progress", "final_digest", "bundle_path", "error_reason"], [], "experiment run");
  const variant = validateExperimentIdentifier(value.variant_id, "variant identity");
  const seed = unsigned(value.seed, "run seed");
  const id = validateExperimentIdentifier(value.run_id, "run identity");
  if (id !== `${experimentId}-${variant}-seed${seed}`) refuse("run identity differs from its experiment, arm or seed");
  const horizon = integer(value.target_ticks, "target tick", 1, EXPERIMENT_LIMITS.maxTicks);
  const total = integer(value.total_ticks, "terminal ticks", 0, horizon);
  const error = nullableText(value.error_reason, "run failure");
  const finalDigest = value.final_digest === null ? null : worldDigest(value.final_digest, "final world digest");
  const bundle = nullableText(value.bundle_path, "retained bundle path");
  if (!["pending", "running", "completed", "failed"].includes(value.state)) refuse("unsupported run lifecycle state");
  if (value.state === "completed") {
    if (total !== horizon || finalDigest === null || bundle === null || error !== null) refuse("completed run lacks consistent terminal evidence");
  } else if (total !== 0 || finalDigest !== null || bundle !== null
      || (value.state === "failed" ? error === null : error !== null)) refuse("unfinished or failed run carries inconsistent terminal evidence");
  const progress = checkpointProgress(value.checkpoint_progress, horizon);
  if (value.state === "pending" && !(progress.state === "unavailable" && progress.reason === "not_started")) refuse("pending run invents checkpoint progress");
  if (value.state !== "pending" && progress.state === "unavailable" && progress.reason === "not_started") refuse("started run is described as never started");
  if (value.state === "completed" && progress.state === "observed" && progress.tick !== total) refuse("completed run and observed checkpoint ticks differ");
  return { run_id: id, variant_id: variant, brain_family: canonicalFamily(value.brain_family), seed,
    state: value.state, total_ticks: total, target_ticks: horizon, checkpoint_progress: progress,
    final_digest: finalDigest, bundle_path: bundle, error_reason: error };
}

function sameSeeds(left, right) {
  return left.length === right.length && new Set(left).size === left.length
    && new Set(right).size === right.length && left.every((seed) => right.includes(seed));
}

export function validateExperimentStatus(value, context = {}) {
  shape(value, ["schema_version", "generation", "plan_digest", "experiment_id", "status", "total_runs",
    "completed_runs", "failed_runs", "runs", "error_reason", "created_at_utc", "updated_at_utc"], [], "experiment status");
  if (value.schema_version !== 2) refuse("unsupported experiment status schema");
  const summary = batchSummary(value);
  const generation = integer(value.generation, "status generation");
  const plan = digest(value.plan_digest, "experiment plan digest");
  if (context.experimentId !== undefined && summary.experiment_id !== context.experimentId) refuse("status belongs to a different experiment");
  if (context.planDigest !== undefined && plan !== context.planDigest) refuse("experiment plan changed");
  if (context.minimumGeneration !== undefined && generation < integer(context.minimumGeneration, "previous generation")) refuse("stale experiment status generation");
  if (!Array.isArray(value.runs) || value.runs.length !== summary.total_runs) refuse("status does not contain the complete declared run matrix");
  const ids = new Set(), variants = new Map();
  const runs = value.runs.map((record) => {
    const run = runRecord(record, summary.experiment_id);
    if (ids.has(run.run_id)) refuse("duplicate experiment run identity");
    ids.add(run.run_id);
    if (!variants.has(run.variant_id)) variants.set(run.variant_id, { family: run.brain_family, seeds: [] });
    const arm = variants.get(run.variant_id);
    if (arm.family !== run.brain_family || arm.seeds.includes(run.seed)) refuse("arm family or seed matrix is inconsistent");
    arm.seeds.push(run.seed);
    return run;
  });
  const seedSet = [...variants.values()][0].seeds;
  if (variants.size > EXPERIMENT_LIMITS.maxStatusVariants || seedSet.length > EXPERIMENT_LIMITS.maxStatusSeeds
      || [...variants.values()].some((arm) => !sameSeeds(seedSet, arm.seeds))) refuse("arms do not contain the same complete bounded seed schedule");
  if (runs.some((run) => run.target_ticks !== runs[0].target_ticks)) refuse("experiment run horizons disagree");
  if (runs.filter((run) => run.state === "completed").length !== summary.completed_runs
      || runs.filter((run) => run.state === "failed").length !== summary.failed_runs) refuse("status totals disagree with observed run states");
  const error = nullableText(value.error_reason, "batch failure");
  if (summary.status === "pending" && runs.some((run) => run.state !== "pending")) refuse("pending batch already contains started runs");
  if (summary.status === "running" && runs.every((run) => run.state === "pending")) refuse("running batch has no started run");
  if (summary.status === "completed" && error !== null) refuse("completed batch also declares a failure");
  if (summary.status === "failed" && error === null
      && (summary.failed_runs === 0 || summary.completed_runs + summary.failed_runs !== summary.total_runs)) refuse("failed batch has no observed terminal or batch failure");

  if (context.request !== undefined) {
    const request = prepareExperimentRequest(context.request.request ?? context.request).request;
    if (request.experiment_id !== null && request.experiment_id !== summary.experiment_id) refuse("admission returned another experiment identity");
    if (variants.size !== request.variants.length || !sameSeeds(seedSet, request.seeds)
        || runs[0].target_ticks !== request.ticks_per_run || request.variants.some((arm) => variants.get(arm.variant_id)?.family !== arm.brain_family)) refuse("admitted arm, family, seed or horizon differs from the immutable request");
  }
  if (context.previousStatus) {
    const previous = validateExperimentStatus(context.previousStatus);
    if (previous.experiment_id !== summary.experiment_id || previous.plan_digest !== plan
        || previous.created_at_utc !== summary.created_at_utc || generation < previous.generation
        || previous.runs.length !== runs.length) refuse("status identity or generation changed across observations");
    const byId = new Map(previous.runs.map((run) => [run.run_id, run]));
    for (const run of runs) {
      const old = byId.get(run.run_id);
      if (!old || old.variant_id !== run.variant_id || old.seed !== run.seed
          || old.brain_family !== run.brain_family || old.target_ticks !== run.target_ticks) refuse("immutable run matrix changed across observations");
      if ((old.state === "running" && run.state === "pending")
          || (["completed", "failed"].includes(old.state) && old.state !== run.state)) refuse("run lifecycle regressed");
      const terminalKeys = ["state", "total_ticks", "final_digest", "bundle_path", "error_reason"];
      if ((generation === previous.generation || ["completed", "failed"].includes(old.state))
          && terminalKeys.some((key) => old[key] !== run[key])) refuse("unchanged or terminal run evidence was replaced");
    }
  }
  return freeze({ schema_version: 2, generation, plan_digest: plan, ...summary, runs, error_reason: error });
}

export function experimentCanResume(status) {
  return !!status && ["stopped", "cancelled", "failed"].includes(status.status)
    && status.runs.some((run) => ["pending", "running"].includes(run.state));
}

export function validateExperimentComparison(value, context) {
  shape(value, ["experiment_id", "baseline_variant", "challenger_variant", "metric", "endpoint_tick",
    "matched_seeds", "mean_difference", "ci_95", "standardized_effect", "p_value", "adjusted_p_value",
    "significant", "descriptive_only", "underpowered_reason", "undefined_effect_reason", "plan_digest",
    "status_generation"], [], "experiment comparison");
  if (!context?.status) refuse("comparison requires the currently observed experiment status");
  const status = validateExperimentStatus(context.status);
  const baseline = validateExperimentIdentifier(value.baseline_variant, "baseline arm");
  const challenger = validateExperimentIdentifier(value.challenger_variant, "challenger arm");
  if (baseline === challenger || (context.baselineVariant !== undefined && baseline !== context.baselineVariant)
      || (context.challengerVariant !== undefined && challenger !== context.challengerVariant)) refuse("comparison arm identities disagree");
  if (!EXPERIMENT_ENDPOINT_METRICS.includes(value.metric)
      || (context.metric !== undefined && value.metric !== context.metric)) refuse("comparison endpoint metric disagrees");
  if (value.experiment_id !== status.experiment_id || digest(value.plan_digest, "comparison plan") !== status.plan_digest
      || integer(value.status_generation, "comparison generation") !== status.generation) refuse("comparison does not match the observed plan and generation");
  const arms = [baseline, challenger].map((id) => status.runs.filter((run) => run.variant_id === id));
  if (arms.some((arm) => !arm.length || arm.some((run) => run.state !== "completed"))) refuse("comparison requires every declared seed in both arms to complete");
  if (!Array.isArray(value.matched_seeds) || !value.matched_seeds.length
      || value.matched_seeds.length > EXPERIMENT_LIMITS.maxStatusSeeds) refuse("comparison seed schedule is missing or too large");
  const seeds = value.matched_seeds.map((seed) => unsigned(seed, "comparison seed"));
  if (arms.some((arm) => !sameSeeds(seeds, arm.map((run) => run.seed)))) refuse("comparison drops, duplicates or invents an independent seed pair");
  const endpoint = integer(value.endpoint_tick, "comparison endpoint", 1, EXPERIMENT_LIMITS.maxTicks);
  if (arms.some((arm) => arm.some((run) => run.total_ticks !== endpoint))) refuse("comparison endpoint differs from completed runs");
  const descriptive = boolean(value.descriptive_only, "descriptive-only flag");
  if (descriptive !== (seeds.length < 2)) refuse("comparison mislabels independent replication");
  const difference = finite(value.mean_difference, "challenger-minus-baseline difference");
  let interval = null;
  if (value.ci_95 !== null) {
    if (!Array.isArray(value.ci_95) || value.ci_95.length !== 2) refuse("confidence interval is invalid");
    interval = value.ci_95.map((bound) => finite(bound, "confidence bound"));
    if (interval[0] > interval[1]) refuse("confidence interval is reversed");
  }
  const effect = value.standardized_effect === null ? null : finite(value.standardized_effect, "paired standardized effect");
  const p = value.p_value === null ? null : finite(value.p_value, "p value", 0, 1);
  const adjusted = value.adjusted_p_value === null ? null : finite(value.adjusted_p_value, "adjusted p value", 0, 1);
  const significant = value.significant === null ? null : boolean(value.significant, "significance observation");
  const underpowered = nullableText(value.underpowered_reason, "underpowered reason");
  const undefinedEffect = nullableText(value.undefined_effect_reason, "undefined effect reason");
  if (descriptive && [interval, effect, p, adjusted, significant].some((field) => field !== null)) refuse("one independent seed pair cannot claim inferential statistics");
  if (!descriptive && [interval, p, adjusted, significant].some((field) => field === null)) refuse("replicated comparison omits its declared inference");
  if ((effect === null) !== (undefinedEffect !== null)) refuse("standardized effect availability lacks its matching explanation");
  return freeze({ experiment_id: status.experiment_id, baseline_variant: baseline, challenger_variant: challenger,
    metric: value.metric, endpoint_tick: endpoint, matched_seeds: seeds, mean_difference: difference,
    ci_95: interval, standardized_effect: effect, p_value: p, adjusted_p_value: adjusted,
    significant, descriptive_only: descriptive, underpowered_reason: underpowered,
    undefined_effect_reason: undefinedEffect, plan_digest: status.plan_digest, status_generation: status.generation });
}
