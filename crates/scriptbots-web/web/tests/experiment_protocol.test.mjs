import test from "node:test";
import assert from "node:assert/strict";
import {
  EXPERIMENT_LIMITS, EXPERIMENT_ENDPOINT_METRICS, prepareExperimentRequest,
  parseExperimentJson, validateExperimentStatus, validateExperimentCatalog,
  validateExperimentComparison, validateExperimentIdentifier, experimentCanResume,
} from "../experiment-protocol.js";

const hash = (letter) => letter.repeat(64);
const largestSeed = "18446744073709551615";
const clone = (value) => structuredClone(value);

// Contract fixtures follow the actual Rust DTOs, including completed-only
// total_ticks and independent checkpoint observations. They are not science runs.
function requestFixture() {
  return {
    experiment_id: "browser-cohort", description: "Food conditions at identical independent seeds",
    variants: [
      { variant_id: "control", brain_family: "mlp", config_overrides: { food_growth_rate: 0.01 } },
      { variant_id: "treatment", brain_family: "mlp.baseline", config_overrides: { food_growth_rate: 0.02 } },
    ],
    seeds: ["7", largestSeed], ticks_per_run: 200, max_concurrency: 1,
    idempotency_key: "browser-retry-1",
  };
}

function statusFixture({ seeds = ["7", largestSeed], state = "completed", arms = ["control", "treatment"] } = {}) {
  const runs = arms.flatMap((variant, arm) => seeds.map((seed) => ({
    run_id: `browser-cohort-${variant}-seed${seed}`, variant_id: variant,
    brain_family: "mlp.baseline", seed, state,
    total_ticks: state === "completed" ? 200 : 0, target_ticks: 200,
    checkpoint_progress: state === "pending" ? { state: "unavailable", reason: "not_started" }
      : { state: "observed", tick: state === "completed" ? 200 : 50, checkpoint_digest: hash("d"), source: "published" },
    final_digest: state === "completed" ? hash(arm === 0 ? "a" : "b") : null,
    bundle_path: state === "completed" ? `/retained/experiments/browser-cohort/browser-cohort-${variant}-seed${seed}` : null,
    error_reason: state === "failed" ? "owner construction refused" : null,
  })));
  return {
    schema_version: 2, generation: 9, plan_digest: hash("c"), experiment_id: "browser-cohort",
    status: state, total_runs: runs.length,
    completed_runs: state === "completed" ? runs.length : 0,
    failed_runs: state === "failed" ? runs.length : 0,
    runs, error_reason: null, created_at_utc: "1794456000", updated_at_utc: "1794456010",
  };
}

function comparisonFixture(status = statusFixture()) {
  return {
    experiment_id: status.experiment_id, baseline_variant: "control", challenger_variant: "treatment",
    metric: "total_food", endpoint_tick: 200,
    matched_seeds: status.runs.filter((run) => run.variant_id === "control").map((run) => run.seed),
    mean_difference: 12.5, ci_95: [10, 15], standardized_effect: 2.5,
    p_value: 0.01, adjusted_p_value: 0.01, significant: true, descriptive_only: false,
    underpowered_reason: "small_sample(have=2)", undefined_effect_reason: null,
    plan_digest: status.plan_digest, status_generation: status.generation,
  };
}

function catalogFixture() {
  const status = statusFixture();
  const { experiment_id, total_runs, completed_runs, failed_runs, created_at_utc, updated_at_utc } = status;
  return {
    items: [{ experiment_id, status: "completed", total_runs, completed_runs, failed_runs, created_at_utc, updated_at_utc }],
    total: 1, limit: 12, offset: 0, next_cursor: null, has_more: false,
    execution_capacity: { max_run_slots: 64, reserved_run_slots: 0,
      max_grid_cells: 1_048_576, reserved_grid_cells: 0, reserved_batches: 0 },
  };
}

test("prepared requests freeze exact full-width seeds and retain identical retry bytes", () => {
  const original = requestFixture();
  const prepared = prepareExperimentRequest(original);
  assert.match(prepared.body, /"seeds":\[7,18446744073709551615\]/);
  assert.deepEqual(prepared.request.seeds, ["7", largestSeed]);
  assert.equal(prepared.request.variants[0].brain_family, "mlp.baseline");
  assert.equal(prepareExperimentRequest(prepared.request).body, prepared.body);
  original.seeds[0] = "8";
  original.variants[0].config_overrides.food_growth_rate = 0.5;
  assert.equal(prepared.request.seeds[0], "7");
  assert.equal(prepared.request.variants[0].config_overrides.food_growth_rate, 0.01);
  assert.throws(() => { prepared.request.variants[0].config_overrides.food_growth_rate = 1; }, TypeError);
  assert.ok(Object.isFrozen(prepared));
  assert.equal(parseExperimentJson(prepared.body).seeds[1], largestSeed);
});

test("new requests reject ambiguous, unsafe or excessive independent seed schedules", () => {
  const cases = [[], ["7", "7"], [7, "7"], [Number.MAX_SAFE_INTEGER + 1],
    ["18446744073709551616"], ["-1"], ["01"], ["7.0"], ["7e0"], [" 7"],
    Array.from({ length: EXPERIMENT_LIMITS.maxSeeds + 1 }, (_, index) => String(index))];
  for (const seeds of cases) assert.throws(() => prepareExperimentRequest({ ...requestFixture(), seeds }), /Experiment refused/);
  const edge = prepareExperimentRequest({ ...requestFixture(), seeds: [0, 9007199254740991, largestSeed] });
  assert.deepEqual(edge.request.seeds, ["0", "9007199254740991", largestSeed]);
});

test("arm aliases and object key order cannot disguise identical declared comparisons", () => {
  const request = requestFixture();
  request.variants[0].config_overrides = { food_growth_rate: 0.01, population_minimum: 2 };
  request.variants[1].config_overrides = { population_minimum: 2, food_growth_rate: 0.01 };
  assert.throws(() => prepareExperimentRequest(request), /identical declared arms/);
  request.variants[1].brain_family = "dwraon";
  const different = prepareExperimentRequest(request);
  assert.equal(different.request.variants[1].brain_family, "dwraon.baseline");
  request.variants[1].variant_id = request.variants[0].variant_id;
  assert.throws(() => prepareExperimentRequest(request), /arm identities/);
});

test("request identities, counts and controls are bounded before admission", () => {
  const mutations = [
    (r) => { r.variants.length = 1; },
    (r) => { r.variants = Array.from({ length: 9 }, (_, i) => ({ variant_id: `arm${i}`, brain_family: "mlp", config_overrides: { food_growth_rate: i / 100 } })); },
    (r) => { r.experiment_id = "../other"; },
    (r) => { r.experiment_id = "a".repeat(128); },
    (r) => { r.variants[0].variant_id = "a".repeat(65); },
    (r) => { r.variants[0].brain_family = "neuroflow"; },
    (r) => { r.variants[0].ignored_setting = true; },
    (r) => { r.ticks_per_run = 0; },
    (r) => { r.ticks_per_run = EXPERIMENT_LIMITS.maxTicks + 1; },
    (r) => { r.max_concurrency = 9; },
    (r) => { r.max_concurrency = 1.5; },
    (r) => { r.idempotency_key = ""; },
    (r) => { r.idempotency_key = "line\nbreak"; },
    (r) => { r.description = "🦋".repeat(600); },
    (r) => { r.unrecognized = "cannot vanish silently"; },
  ];
  for (const mutate of mutations) {
    const request = requestFixture(); mutate(request);
    assert.throws(() => prepareExperimentRequest(request), /Experiment refused/);
  }
  assert.equal(validateExperimentIdentifier("experiment.v2"), "experiment.v2");
  assert.throws(() => validateExperimentIdentifier("experiment..v2"), /identity/);
});

test("same-tick interventions retain order and cannot move outside the scientific horizon", () => {
  const request = requestFixture();
  request.variants[1].interventions = [
    { tick: 0, set: { food_growth_rate: 0.03 } },
    { tick: 20, set: { food_growth_rate: 0.04 } },
    { tick: 20, set: { food_growth_rate: 0.01 } },
  ];
  const prepared = prepareExperimentRequest(request);
  assert.deepEqual(prepared.request.variants[1].interventions.map((row) => row.set.food_growth_rate), [0.03, 0.04, 0.01]);
  for (const schedule of [
    [{ tick: 200, set: { food_growth_rate: 0.01 } }],
    [{ tick: 20, set: { food_growth_rate: 0.01 } }, { tick: 19, set: { food_growth_rate: 0.02 } }],
    [{ tick: 0, set: {} }], [{ tick: 0, set: { terrain_enabled: false } }],
    [{ tick: 0, set: { food_growth_rate: NaN } }],
    [{ tick: 0, set: { food_growth_rate: 0.02 }, unexpected: 1 }],
    Array.from({ length: 65 }, () => ({ tick: 0, set: { food_growth_rate: 0.01 } })),
  ]) {
    request.variants[1].interventions = schedule;
    assert.throws(() => prepareExperimentRequest(request), /Experiment refused/);
  }
});

test("configuration values preserve booleans and nested JSON while unsafe values are refused", () => {
  const request = requestFixture();
  request.variants[0].config_overrides = { terrain_enabled: true, "terrain.climate": "temperate", nested: { choice: null, weights: [0.25, 0.75] } };
  assert.deepEqual(prepareExperimentRequest(request).request.variants[0].config_overrides, request.variants[0].config_overrides);
  for (const config of [{ number: Infinity }, { number: 9007199254740992 }, { bad: undefined },
    { bad: () => 1 }, JSON.parse('{"__proto__":{"polluted":true}}'), { "bad/path": 1 }]) {
    request.variants[0].config_overrides = config;
    assert.throws(() => prepareExperimentRequest(request), /Experiment refused/);
  }
  assert.equal({}.polluted, undefined);
});

test("lossless JSON keeps adjacent u64 seeds distinct and refuses unsafe numbers elsewhere", () => {
  const observed = parseExperimentJson('{"seed":18446744073709551615,"matched_seeds":[9007199254740992,9007199254740993],"mean_difference":-0.125}');
  assert.equal(observed.seed, largestSeed);
  assert.deepEqual(observed.matched_seeds, ["9007199254740992", "9007199254740993"]);
  assert.equal(observed.mean_difference, -0.125);
  for (const wire of [
    '{"seed":18446744073709551616}', '{"seed":7.0}', '{"seed":7e0}', '{"seed":"7"}',
    '{"matched_seeds":["7"]}', '{"generation":9007199254740992}', '{"tick":9.007199254740992e15}',
    '{"value":1e309}', '{"seed":1,"se\\u0065d":2}', '{"__proto__":{}}',
    '{"value":1} trailing', '{"value":01}', '{"value":1,}', '[1,]', '"unterminated',
  ]) assert.throws(() => parseExperimentJson(wire), /Experiment refused/);
});

test("JSON byte, depth and node limits apply independently before schema use", () => {
  assert.throws(() => parseExperimentJson('"' + "x".repeat(EXPERIMENT_LIMITS.maxResponseBytes) + '"'), /response exceeds/);
  assert.throws(() => parseExperimentJson('"' + "🦋".repeat(270000) + '"'), /response exceeds/);
  assert.throws(() => parseExperimentJson("[".repeat(26) + "0" + "]".repeat(26)), /structure exceeds/);
  assert.throws(() => parseExperimentJson("[" + Array(100001).fill("0").join(",") + "]"), /structure exceeds/);
});

test("status binds the complete canonical family, seed and run matrix to the request", () => {
  const status = validateExperimentStatus(statusFixture(), {
    experimentId: "browser-cohort", request: prepareExperimentRequest(requestFixture()),
    planDigest: hash("c"), minimumGeneration: 8,
  });
  assert.equal(status.runs[1].seed, largestSeed);
  assert.equal(status.runs[0].final_digest, hash("a"));
  assert.equal(status.runs[0].checkpoint_progress.checkpoint_digest, hash("d"));
  assert.ok(Object.isFrozen(status.runs[0].checkpoint_progress));
  assert.throws(() => { status.runs[0].seed = "8"; }, TypeError);
  assert.throws(() => validateExperimentStatus(statusFixture(), { experimentId: "other" }), /different experiment/);
  assert.throws(() => validateExperimentStatus(statusFixture(), { planDigest: hash("f") }), /plan changed/);
  assert.throws(() => validateExperimentStatus(statusFixture(), { minimumGeneration: 10 }), /stale/);
  const request = requestFixture(); request.seeds[0] = "8";
  assert.throws(() => validateExperimentStatus(statusFixture(), { request }), /immutable request/);
});

test("historical catalogs can inspect more arms than this browser admits for creation", () => {
  const fixture = statusFixture({ arms: Array.from({ length: 64 }, (_, index) => `arm${index}`), seeds: ["7"] });
  assert.equal(validateExperimentStatus(fixture).runs.length, 64);
  const excessive = statusFixture({ arms: Array.from({ length: 65 }, (_, index) => `arm${index}`), seeds: ["7"] });
  assert.throws(() => validateExperimentStatus(excessive), /bounded seed schedule/);
});

test("missing, duplicate, mismatched and falsely completed matrix cells cannot be displayed", () => {
  const mutations = [
    (s) => { s.runs.pop(); },
    (s) => { s.runs[1] = clone(s.runs[0]); },
    (s) => { s.runs[1].run_id = "unrelated"; },
    (s) => { s.runs[1].brain_family = "dwraon.baseline"; },
    (s) => { s.runs[1].seed = "8"; s.runs[1].run_id = "browser-cohort-control-seed8"; },
    (s) => { s.runs[1].target_ticks = 201; s.runs[1].total_ticks = 201; s.runs[1].checkpoint_progress.tick = 201; },
    (s) => { s.completed_runs = 3; },
    (s) => { s.runs[1].final_digest = null; },
    (s) => { s.runs[1].bundle_path = null; },
    (s) => { s.runs[1].total_ticks = 0; },
    (s) => { s.runs[1].error_reason = "failed despite the label"; },
    (s) => { s.runs[1].checkpoint_progress.tick = 199; },
    (s) => { s.schema_version = 1; },
    (s) => { s.unknown = true; },
  ];
  for (const mutate of mutations) {
    const status = statusFixture(); mutate(status);
    assert.throws(() => validateExperimentStatus(status), /Experiment refused/);
  }
});

test("running progress is a checkpoint observation and never invented terminal ticks", () => {
  const running = statusFixture({ state: "running" });
  const observed = validateExperimentStatus(running);
  assert.equal(observed.runs[0].total_ticks, 0);
  assert.equal(observed.runs[0].checkpoint_progress.tick, 50);
  running.runs[0].total_ticks = 50;
  assert.throws(() => validateExperimentStatus(running), /terminal evidence/);
  const pending = statusFixture({ state: "pending" });
  pending.runs[0].checkpoint_progress = { state: "observed", tick: 0, checkpoint_digest: hash("d"), source: "published" };
  assert.throws(() => validateExperimentStatus(pending), /pending run invents/);
  const refusal = statusFixture({ state: "running" });
  refusal.runs[0].checkpoint_progress = { state: "refused", reason: "invalid_checkpoint" };
  assert.equal(validateExperimentStatus(refusal).runs[0].checkpoint_progress.state, "refused");
});

test("only stopped incomplete batches offer resume and cancellation is not completion", () => {
  for (const state of ["pending", "running", "completed", "failed"]) {
    assert.equal(experimentCanResume(validateExperimentStatus(statusFixture({ state }))), false);
  }
  for (const status of ["stopped", "cancelled", "failed"]) {
    const fixture = statusFixture({ state: "running" }); fixture.status = status;
    if (status === "failed") fixture.error_reason = "worker exited during publication";
    assert.equal(experimentCanResume(validateExperimentStatus(fixture)), true);
  }
  const cancelling = statusFixture({ state: "running" }); cancelling.status = "cancelling";
  assert.equal(experimentCanResume(validateExperimentStatus(cancelling)), false);
  const fabricatedFailure = statusFixture({ state: "running" }); fabricatedFailure.status = "failed";
  assert.throws(() => validateExperimentStatus(fabricatedFailure), /no observed terminal or batch failure/);
  const noStartedRun = statusFixture({ state: "pending" }); noStartedRun.status = "running";
  assert.throws(() => validateExperimentStatus(noStartedRun), /no started run/);
});

test("generation binding preserves terminal evidence while allowing newer checkpoint observations", () => {
  const previous = statusFixture({ state: "running" });
  const next = clone(previous); next.runs[0].checkpoint_progress.tick = 60;
  assert.equal(validateExperimentStatus(next, { previousStatus: previous }).runs[0].checkpoint_progress.tick, 60);
  for (const change of [
    (s) => { s.generation = 8; },
    (s) => { s.plan_digest = hash("f"); },
    (s) => { s.created_at_utc = "1794455999"; },
    (s) => { s.runs[0].state = "pending"; s.runs[0].checkpoint_progress = { state: "unavailable", reason: "not_started" }; },
  ]) {
    const changed = clone(previous); change(changed);
    assert.throws(() => validateExperimentStatus(changed, { previousStatus: previous }), /Experiment refused/);
  }
  const completed = statusFixture();
  assert.throws(() => validateExperimentStatus(completed, { previousStatus: previous }), /unchanged or terminal/);
  completed.generation += 1;
  assert.equal(validateExperimentStatus(completed, { previousStatus: previous }).status, "completed");
  const tampered = clone(completed); tampered.generation += 1; tampered.runs[0].final_digest = hash("f");
  assert.throws(() => validateExperimentStatus(tampered, { previousStatus: completed }), /terminal run evidence/);
});

test("catalog pagination binds request cursor, complete rows and capacity observations", () => {
  const page = catalogFixture();
  const validated = validateExperimentCatalog(page, { cursor: null, limit: 12 });
  assert.equal(validated.items[0].experiment_id, "browser-cohort");
  assert.equal(validated.execution_capacity.reserved_run_slots, 0);
  assert.ok(Object.isFrozen(validated.items));
  for (const mutate of [
    (p) => { p.offset = 1; }, (p) => { p.total = 2; }, (p) => { p.has_more = true; },
    (p) => { p.next_cursor = "1"; }, (p) => { p.execution_capacity.reserved_run_slots = 65; },
    (p) => { p.execution_capacity.reserved_grid_cells = 1_048_577; },
    (p) => { p.items[0].completed_runs = 3; },
  ]) {
    const changed = clone(page); mutate(changed);
    assert.throws(() => validateExperimentCatalog(changed, { cursor: null, limit: 12 }), /Experiment refused/);
  }
  const beyond = { ...page, items: [], offset: 20 };
  assert.equal(validateExperimentCatalog(beyond, { cursor: "20", limit: 12 }).items.length, 0);
  const first = { ...page, total: 2, limit: 1, has_more: true, next_cursor: "1" };
  assert.equal(validateExperimentCatalog(first, { cursor: null, limit: 1 }).next_cursor, "1");
  const duplicate = { ...page, items: [page.items[0], page.items[0]], total: 2 };
  assert.throws(() => validateExperimentCatalog(duplicate), /repeats an experiment/);
});


test("catalog reports owner-configured disabled and enlarged shared capacity", () => {
  for (const [maximum, reserved] of [[0, 0], [128, 96], [Number.MAX_SAFE_INTEGER, 128]]) {
    const page = catalogFixture();
    Object.assign(page.execution_capacity, { max_run_slots: maximum, reserved_run_slots: reserved });
    const checked = validateExperimentCatalog(page);
    assert.equal(checked.execution_capacity.max_run_slots, maximum);
    assert.equal(checked.execution_capacity.reserved_run_slots, reserved);
  }
  for (const [maximum, reserved] of [[0, 1], [128, 129], [-1, 0], [Infinity, 0], [Number.MAX_SAFE_INTEGER + 1, 0]]) {
    const page = catalogFixture();
    Object.assign(page.execution_capacity, { max_run_slots: maximum, reserved_run_slots: reserved });
    assert.throws(() => validateExperimentCatalog(page), /Experiment refused/);
  }
});

test("paired comparison binds each independent seed and preserves signed endpoint observations", () => {
  const status = statusFixture(), value = comparisonFixture(status);
  value.matched_seeds.reverse();
  value.mean_difference = -12.5; value.ci_95 = [-15, -10]; value.standardized_effect = -2.5;
  const result = validateExperimentComparison(value, { status, baselineVariant: "control", challengerVariant: "treatment", metric: "total_food" });
  assert.equal(result.mean_difference, -12.5);
  assert.deepEqual(result.matched_seeds, [largestSeed, "7"]);
  assert.equal(result.descriptive_only, false);
  assert.ok(Object.isFrozen(result.ci_95));
  assert.equal(EXPERIMENT_ENDPOINT_METRICS.length, 8);
});

test("one pair stays descriptive and zero variance does not manufacture an effect size", () => {
  const status = statusFixture({ seeds: [largestSeed] }), value = comparisonFixture(status);
  Object.assign(value, { descriptive_only: true, ci_95: null, standardized_effect: null,
    p_value: null, adjusted_p_value: null, significant: null,
    underpowered_reason: "insufficient_pairs(have=1, need=2)", undefined_effect_reason: "insufficient_pairs(have=1, need=2)" });
  assert.equal(validateExperimentComparison(value, { status }).matched_seeds[0], largestSeed);
  for (const [key, field] of [["ci_95", [1, 2]], ["p_value", 0.01], ["standardized_effect", 4], ["significant", false]]) {
    assert.throws(() => validateExperimentComparison({ ...value, [key]: field }, { status }), /one independent seed pair/);
  }
  const repeated = statusFixture(), zero = comparisonFixture(repeated);
  Object.assign(zero, { mean_difference: 0, ci_95: [0, 0], standardized_effect: null,
    p_value: 1, adjusted_p_value: 1, significant: false, undefined_effect_reason: "zero_variance" });
  assert.equal(validateExperimentComparison(zero, { status: repeated }).standardized_effect, null);
});

test("comparison refuses stale provenance, unpaired seeds, incomplete arms and invented inference", () => {
  const status = statusFixture(), base = comparisonFixture(status);
  const changes = [
    (r) => { r.experiment_id = "other"; }, (r) => { r.plan_digest = hash("f"); },
    (r) => { r.status_generation -= 1; }, (r) => { r.endpoint_tick = 199; },
    (r) => { r.baseline_variant = "treatment"; }, (r) => { r.metric = "fitness"; },
    (r) => { r.matched_seeds = ["7"]; }, (r) => { r.matched_seeds = ["7", "7"]; },
    (r) => { r.matched_seeds = ["7", "8"]; }, (r) => { r.descriptive_only = true; },
    (r) => { r.mean_difference = NaN; }, (r) => { r.ci_95 = [15, 10]; },
    (r) => { r.p_value = 1.1; }, (r) => { r.significant = 1; },
    (r) => { r.standardized_effect = null; }, (r) => { r.extra = "unsupported"; },
  ];
  for (const change of changes) {
    const value = clone(base); change(value);
    assert.throws(() => validateExperimentComparison(value, { status }), /Experiment refused/);
  }
  assert.throws(() => validateExperimentComparison(base, { status, metric: "total_energy" }), /metric disagrees/);
  assert.throws(() => validateExperimentComparison(base, { status: statusFixture({ state: "running" }) }), /every declared seed/);
  assert.throws(() => validateExperimentComparison(base, {}), /currently observed/);
});

test("two completed arms remain comparable when a separate arm fails", () => {
  const status = statusFixture({ arms: ["control", "treatment", "third"] });
  for (const run of status.runs.filter((row) => row.variant_id === "third")) {
    Object.assign(run, { state: "failed", total_ticks: 0, final_digest: null,
      bundle_path: null, error_reason: "unsupported evaluated state" });
  }
  status.status = "failed"; status.completed_runs = 4; status.failed_runs = 2;
  assert.equal(validateExperimentComparison(comparisonFixture(status), { status }).matched_seeds.length, 2);
});
