import assert from "node:assert/strict";
import test from "node:test";
import { readExperimentForm, renderExperimentLab, experimentObservationExport } from "../experiment-view.js";

class Element {
  constructor(ownerDocument, tag = "div") {
    this.ownerDocument = ownerDocument;
    this.tagName = tag;
    this.dataset = {};
    this.children = [];
    this.value = "";
    this.disabled = false;
    this.hidden = false;
    this._text = "";
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(" "); }
  set innerHTML(_) { throw new Error("Untrusted experiment data must use textContent."); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text = ""; this.children = children; }
}

function panel() {
  const document = { createElement: (tag) => new Element(document, tag) };
  const root = new Element(document);
  const nodes = new Map();
  root.querySelector = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, new Element(document));
    return nodes.get(selector);
  };
  root.at = (name) => root.querySelector("#experiment-" + name);
  return root;
}

function form(changes = {}) {
  const values = {
    id: "", description: "",
    seeds: "11, 42, 137", ticks: "200", concurrency: "1",
    "baseline-id": "baseline", "baseline-family": "mlp",
    "baseline-config": '{"population_minimum":20}', "baseline-schedule": "[]",
    "challenger-id": "challenger", "challenger-family": "dwraon",
    "challenger-config": '{"population_minimum":20}', "challenger-schedule": "[]",
    ...changes,
  };
  return { querySelector: (selector) => ({ value: values[selector.replace("#experiment-", "")] }) };
}

function run(variant = "baseline", seed = "11", changes = {}) {
  return {
    run_id: variant + "-" + seed,
    variant_id: variant,
    brain_family: "mlp.baseline",
    seed, state: "completed", total_ticks: 200, target_ticks: 200,
    checkpoint_progress: { state: "unavailable", reason: "No saved checkpoint." },
    final_digest: "final-" + variant + "-" + seed, bundle_path: "/server/experiment/run", error_reason: null,
    ...changes,
  };
}

function status(runs = [run(), run("challenger")]) {
  return {
    schema_version: 2, generation: 3, plan_digest: "plan-abc", experiment_id: "experiment-a",
    status: "completed", total_runs: runs.length,
    completed_runs: runs.filter((entry) => entry.state === "completed").length,
    failed_runs: runs.filter((entry) => entry.state === "failed").length,
    runs, error_reason: null, created_at_utc: "1700000000", updated_at_utc: "1700000001",
  };
}

function model(changes = {}) {
  const selected = status();
  return {
    enabled: true, source: { sessionId: "session-a", token: "https://private.invalid/token" },
    busy: false, canOperate: true, canLaunch: true, canSelect: true,
    canCancel: false, canResume: false, canCompare: true, canRetry: false,
    page: null, cursor: null, selected, creation: null, comparison: null, notice: "",
    runPageIndex: 0, totalRunPages: 1, visibleRuns: selected.runs,
    ...changes,
  };
}

function comparison(changes = {}) {
  return {
    experiment_id: "experiment-a", baseline_variant: "baseline", challenger_variant: "challenger",
    metric: "alive_agents", endpoint_tick: 200, matched_seeds: ["11", "18446744073709551615"],
    mean_difference: 2, ci_95: [-1, 5], standardized_effect: null,
    p_value: 0.25, adjusted_p_value: 0.25, significant: false, descriptive_only: false,
    underpowered_reason: "Few independent seed pairs.", undefined_effect_reason: "Paired effect denominator is zero.",
    plan_digest: "plan-abc", status_generation: 3, ...changes,
  };
}

test("form retains exact u64 seeds, per-arm overrides, and ordered same-tick interventions", () => {
  const request = readExperimentForm(form({
    id: "two_arm_study",
    seeds: "0, 9007199254740993\n18446744073709551615",
    "baseline-config": '{"population_minimum":20,"food.maximum":0.5}',
    "challenger-family": "assembly",
    "challenger-schedule": '[{"tick":100,"set":{"food_growth_rate":0.001}},{"tick":100,"set":{"food_growth_rate":0.002}}]',
  }));
  assert.deepEqual(request.seeds, ["0", "9007199254740993", "18446744073709551615"]);
  assert.equal(request.experiment_id, "two_arm_study");
  assert.equal(request.variants[0].config_overrides["food.maximum"], 0.5);
  assert.equal(request.variants[1].brain_family, "assembly");
  assert.deepEqual(request.variants[1].interventions.map((entry) => entry.set.food_growth_rate), [0.001, 0.002]);
  assert.equal(Object.hasOwn(request, "idempotency_key"), false);
});

test("form refuses malformed seed identities, limits, duplicate arms, and invalid editors", () => {
  for (const value of ["", "1,1", "01", "-1", "1.5", "1e3", "18446744073709551616", Array.from({ length: 33 }, (_, i) => String(i)).join(",")]) {
    assert.throws(() => readExperimentForm(form({ seeds: value })), /seed/i);
  }
  for (const value of ["0", "-1", "1e3", "10000001", "1.5"]) {
    assert.throws(() => readExperimentForm(form({ ticks: value })), /Horizon/);
  }
  assert.throws(() => readExperimentForm(form({ concurrency: "9" })), /Concurrency/);
  assert.throws(() => readExperimentForm(form({ "challenger-id": "baseline" })), /IDs must differ/);
  assert.throws(() => readExperimentForm(form({ "baseline-config": "[]" })), /object/);
  assert.throws(() => readExperimentForm(form({ "baseline-config": "{" })), /valid JSON/);
  assert.throws(() => readExperimentForm(form({ "challenger-schedule": "{}" })), /array/);
  assert.throws(() => readExperimentForm(form({ "baseline-config": " ".repeat(16384) + "{}" })), /16 KiB/);
  assert.throws(() => readExperimentForm(form({ id: "bad/id" })), /Experiment ID/);
  assert.throws(() => readExperimentForm(form({ "baseline-family": "invented" })), /brain family/);
});

test("optional request metadata and blank editors produce explicit empty values", () => {
  const request = readExperimentForm(form({
    seeds: "42", "baseline-config": "", "baseline-schedule": "",
    "challenger-config": "", "challenger-schedule": "", "challenger-family": "ft",
  }));
  assert.equal(request.experiment_id, null);
  assert.equal(request.description, null);
  assert.deepEqual(request.variants[0].config_overrides, {});
  assert.deepEqual(request.variants[0].interventions, []);
  assert.equal(request.variants[1].brain_family, "ft");
  assert.deepEqual(request.seeds, ["42"]);
});

test("running progress is a saved observation rather than the zero terminal tick field", () => {
  const root = panel();
  const running = run("baseline", "18446744073709551615", {
    state: "running", total_ticks: 0, final_digest: null, bundle_path: null,
    checkpoint_progress: { state: "observed", tick: 73, checkpoint_digest: "checkpoint-73", source: "recovered" },
  });
  renderExperimentLab(root, model({ selected: status([running]), visibleRuns: [running], canCompare: false }));
  const text = root.at("runs").textContent;
  assert.match(text, /seed 18446744073709551615/);
  assert.match(text, /tick 73 \(recovered\)/);
  assert.match(text, /checkpoint-73/);
  assert.doesNotMatch(text, /terminal endpoint: 0/);
});

test("failed checkpoint and run reasons render as literal text", () => {
  const root = panel();
  const failed = run("challenger", "42", {
    state: "failed", total_ticks: 0,
    error_reason: "<script>alert('run')</script>",
    checkpoint_progress: { state: "refused", reason: "<img src=x onerror=bad()> conflicting digests" },
  });
  renderExperimentLab(root, model({ selected: status([failed]), visibleRuns: [failed], canCompare: false }));
  assert.match(root.at("runs").textContent, /<script>alert\('run'\)<\/script>/);
  assert.match(root.at("runs").textContent, /Saved progress refused/);
  assert.match(root.at("runs").textContent, /conflicting digests/);
});

test("catalog retains an out-of-page selected identity and run rendering stays paged", () => {
  const root = panel();
  const runs = Array.from({ length: 40 }, (_, i) => run(i < 20 ? "baseline" : "challenger", String(i % 20)));
  const controller = model({
    selected: status(runs), visibleRuns: runs.slice(16, 32), runPageIndex: 1, totalRunPages: 3,
    page: {
      items: [{ experiment_id: "another", status: "running", completed_runs: 0, total_runs: 6 }],
      total: 30, offset: 12, limit: 12, has_more: true, next_cursor: "opaque",
      execution_capacity: { max_run_slots: 8, reserved_run_slots: 2, max_grid_cells: 1000, reserved_grid_cells: 400, reserved_batches: 2 },
    },
  });
  renderExperimentLab(root, controller);
  assert.equal(root.at("select").value, "experiment-a");
  assert.match(root.at("select").textContent, /outside this catalog page/);
  assert.equal(root.at("runs").children.length, 16);
  assert.match(root.at("run-page").textContent, /Run page 2 of 3/);
  assert.equal(root.at("run-previous").disabled, false);
  assert.equal(root.at("run-next").disabled, false);
  assert.equal(root.at("catalog-first").disabled, false);
  assert.equal(root.at("catalog-next").disabled, false);
  assert.match(root.at("capacity").textContent, /2\/8 run slots/);
});

test("inactive source keeps observations but disables network actions; exact retry is explicit", () => {
  const root = panel();
  const controller = model({
    enabled: false, canOperate: false, canLaunch: false, canSelect: false, canCompare: false,
    canRetry: false, canCancel: false, canResume: false,
    creation: { state: "unconfirmed", prepared: { body: "{}" } },
  });
  renderExperimentLab(root, controller);
  assert.match(root.at("source").textContent, /inactive/);
  assert.match(root.at("status").textContent, /experiment-a/);
  assert.match(root.at("admission").textContent, /Retry exact request/);
  for (const id of ["launch", "retry", "refresh", "cancel", "resume", "compare", "catalog-refresh"]) {
    assert.equal(root.at(id).disabled, true, id);
  }
  assert.equal(root.at("export").disabled, false);
});

test("only complete cohorts appear in comparison controls and chosen arms survive refresh", () => {
  const root = panel();
  const runs = [run("a"), run("b"), run("c"), run("pending", "11", { state: "running", total_ticks: 0 })];
  const controller = model({ selected: status(runs), visibleRuns: runs });
  renderExperimentLab(root, controller);
  assert.equal(root.at("compare-baseline").value, "a");
  assert.equal(root.at("compare-challenger").value, "b");
  assert.doesNotMatch(root.at("compare-baseline").textContent, /pending/);
  root.at("compare-baseline").value = "c";
  root.at("compare-challenger").value = "a";
  renderExperimentLab(root, controller);
  assert.equal(root.at("compare-baseline").value, "c");
  assert.equal(root.at("compare-challenger").value, "a");
});

test("paired comparison shows direction, uncertainty, provenance, and seed replication scope", () => {
  const root = panel();
  renderExperimentLab(root, model({ comparison: comparison() }));
  const text = root.at("comparison").textContent;
  assert.equal(root.at("comparison").hidden, false);
  assert.match(text, /challenger minus baseline/);
  assert.match(text, /2 matched seed pairs/);
  assert.match(text, /95% interval: \[-1, 5\]/);
  assert.match(text, /Standardized paired effect: unavailable/);
  assert.match(text, /not a fitness verdict/);
  assert.match(text, /ticks are not independent replicates/);
  assert.match(text, /not constitute one jointly corrected analysis/);
  assert.match(text, /denominator is zero/);
  assert.match(text, /18446744073709551615/);
  assert.match(text, /plan-abc.*generation 3/);
});

test("large endpoint differences retain trailing integer zeroes", () => {
  const root = panel();
  renderExperimentLab(root, model({ comparison: comparison({ mean_difference: 100000, ci_95: [10000, 200000] }) }));
  assert.match(root.at("comparison").textContent, /Mean paired difference: 100000/);
  assert.match(root.at("comparison").textContent, /\[10000, 200000\]/);
});

test("single-pair comparison is descriptive and clearing it removes stale inference", () => {
  const root = panel();
  const one = comparison({
    matched_seeds: ["42"], descriptive_only: true, ci_95: null, p_value: null,
    adjusted_p_value: null, significant: null, standardized_effect: null,
    underpowered_reason: "Only one paired seed.", undefined_effect_reason: null,
  });
  renderExperimentLab(root, model({ comparison: one }));
  assert.match(root.at("comparison").textContent, /Descriptive only/);
  assert.doesNotMatch(root.at("comparison").textContent, /95% interval|p-value/);
  renderExperimentLab(root, model({ comparison: null }));
  assert.equal(root.at("comparison").hidden, true);
  assert.equal(root.at("comparison").textContent, "");
});

test("observation export retains exact admitted request bytes and excludes connection token", async () => {
  const body = '{"seeds":[9007199254740993,18446744073709551615],"idempotency_key":"request-a"}';
  const controller = model({
    creation: { state: "observed", experimentId: "experiment-a", prepared: { body } },
    comparison: comparison(),
  });
  const output = experimentObservationExport(controller);
  const text = await output.blob.text();
  const data = JSON.parse(text);
  assert.equal(output.filename, "scriptbots-experiment-observation.json");
  assert.equal(data.request_json, body);
  assert.equal(data.comparison.matched_seeds[1], "18446744073709551615");
  assert.match(data.scope, /not a server run bundle/);
  assert.equal(data.source_session_id, "session-a");
  assert.doesNotMatch(text, /private\.invalid|source_token/);
  assert.equal(Object.hasOwn(data, "token"), false);
});

test("historical selection cannot acquire an unrelated or unconfirmed creation request", async () => {
  for (const creation of [
    null,
    { state: "observed", experimentId: "another", prepared: { body: '{"unrelated":true}' } },
    { state: "unconfirmed", experimentId: "experiment-a", prepared: { body: '{"unconfirmed":true}' } },
  ]) {
    const output = experimentObservationExport(model({ creation }));
    const data = JSON.parse(await output.blob.text());
    assert.equal(data.request_json, null);
    assert.equal(data.admission_state, null);
    assert.match(data.request_context, /cannot be reconstructed/);
  }
  assert.throws(() => experimentObservationExport(model({ selected: null, comparison: null })), /Observe/);
});

test("a newly observed catalog keeps an explicit choice so its first experiment can be opened", () => {
  const root = panel();
  renderExperimentLab(root, model({
    selected: null, visibleRuns: [], canCompare: false,
    page: {
      items: [{ experiment_id: "first-study", status: "completed", completed_runs: 2, total_runs: 2 }],
      total: 1, offset: 0, limit: 12, has_more: false, next_cursor: null,
      execution_capacity: { max_run_slots: 8, reserved_run_slots: 0, max_grid_cells: 1000, reserved_grid_cells: 0, reserved_batches: 0 },
    },
  }));
  assert.equal(root.at("select").value, "");
  assert.equal(root.at("select").children[1].value, "first-study");
  assert.match(root.at("select").children[0].textContent, /Choose a retained experiment/);
});

test("an in-flight comparison locks its pair and metric controls until the observation settles", () => {
  const root = panel();
  const ready = model();
  renderExperimentLab(root, ready);
  assert.equal(root.at("compare-baseline").disabled, false);
  assert.equal(root.at("compare-challenger").disabled, false);
  assert.equal(root.at("metric").disabled, false);
  renderExperimentLab(root, model({
    busy: true, canOperate: false, canSelect: false, canLaunch: false, canCompare: false,
  }));
  assert.equal(root.at("compare-baseline").disabled, true);
  assert.equal(root.at("compare-challenger").disabled, true);
  assert.equal(root.at("metric").disabled, true);
  assert.equal(root.at("compare-baseline").value, "baseline");
  assert.equal(root.at("compare-challenger").value, "challenger");
  renderExperimentLab(root, ready);
  assert.equal(root.at("metric").disabled, false);
});
