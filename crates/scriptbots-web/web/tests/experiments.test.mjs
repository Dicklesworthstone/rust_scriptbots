import test from "node:test";
import assert from "node:assert/strict";
import { ExperimentController, EXPERIMENT_RUN_PAGE_SIZE } from "../experiments.js";
import { RemoteOwnerHttpError } from "../remote.js";

const SESSION = "18446744073709551610";
const SOURCE = "wss://lab.example/prefix/ws";
const PLAN = "a".repeat(64);
const DIGEST = "b".repeat(64);
const WORLD_DIGEST = "b".repeat(16);
const clone = (value) => structuredClone(value);
const flush = async () => { for (let i = 0; i < 8; i += 1) await Promise.resolve(); };

function request() {
  return {
    experiment_id: "cohort", description: "Compare independent worlds",
    variants: [
      { variant_id: "control", brain_family: "mlp" },
      { variant_id: "treatment", brain_family: "dwraon" },
    ],
    seeds: ["9007199254740993", "18446744073709551615"],
    ticks_per_run: 20, max_concurrency: 1,
  };
}

// DTO fixtures describe the owner contract. They are not observed scientific runs.
function status({ id = "cohort", seeds = request().seeds, state = "running", generation = 1 } = {}) {
  const completed = state === "completed";
  const runs = ["control", "treatment"].flatMap((variant, arm) => seeds.map((seed, index) => {
    const started = completed || (arm === 0 && index === 0);
    return {
      run_id: id + "-" + variant + "-seed" + seed,
      variant_id: variant, brain_family: arm === 0 ? "mlp.baseline" : "dwraon.baseline",
      seed, state: completed ? "completed" : started ? "running" : "pending",
      total_ticks: completed ? 20 : 0, target_ticks: 20,
      checkpoint_progress: completed
        ? { state: "observed", tick: 20, checkpoint_digest: DIGEST, source: "published" }
        : { state: "unavailable", reason: started ? "awaiting_publication" : "not_started" },
      final_digest: completed ? WORLD_DIGEST : null,
      bundle_path: completed ? "/retained/" + id + "/" + variant + "-" + seed : null,
      error_reason: null,
    };
  }));
  return {
    schema_version: 2, generation, plan_digest: PLAN, experiment_id: id, status: state,
    total_runs: runs.length, completed_runs: completed ? runs.length : 0, failed_runs: 0,
    runs, error_reason: null, created_at_utc: "1794456000", updated_at_utc: "1794456010",
  };
}

function catalog(statuses = []) {
  return {
    items: statuses.map(({ experiment_id, status, total_runs, completed_runs, failed_runs,
      created_at_utc, updated_at_utc }) => ({
      experiment_id, status, total_runs, completed_runs, failed_runs, created_at_utc, updated_at_utc,
    })),
    total: statuses.length, limit: 12, offset: 0, next_cursor: null, has_more: false,
    execution_capacity: { max_run_slots: 64, reserved_run_slots: 0,
      max_grid_cells: 1048576, reserved_grid_cells: 0, reserved_batches: 0 },
  };
}

function comparison(observed) {
  return {
    experiment_id: observed.experiment_id, plan_digest: observed.plan_digest,
    status_generation: observed.generation,
    baseline_variant: "control", challenger_variant: "treatment",
    metric: "alive_agents", endpoint_tick: 20,
    matched_seeds: observed.runs.filter((run) => run.variant_id === "control").map((run) => run.seed),
    mean_difference: -2, ci_95: [-5, 1], standardized_effect: -0.5,
    p_value: 0.5, adjusted_p_value: 0.5, significant: false, descriptive_only: false,
    underpowered_reason: "Very small cohort", undefined_effect_reason: null,
  };
}

async function harness(page = catalog()) {
  const calls = { list: [], create: [], status: [], cancel: [], resume: [], compare: [] };
  let clock = 0, sequence = 0;
  const services = {};
  for (const kind of ["create", "status", "cancel", "resume", "compare"]) {
    services[kind] = (request, signal) => new Promise((resolve, reject) => {
      calls[kind].push({ request, signal, resolve, reject });
    });
  }
  services.list = async (request, signal) => {
    calls.list.push({ request, signal });
    return clone(page);
  };
  const controller = new ExperimentController({
    ...services, id: () => "request-" + (++sequence), now: () => clock,
  });
  controller.observe({ identity: { sessionId: SESSION } }, SOURCE);
  controller.setEnabled(true);
  await flush();
  return { controller, calls, advance: (milliseconds) => { clock += milliseconds; } };
}

async function launch(h, observed = status(), input = request()) {
  const promise = h.controller.launch(input);
  h.calls.create.at(-1).resolve(clone(observed));
  assert.equal(await promise, true);
  return h.controller;
}

const pair = { baselineVariant: "control", challengerVariant: "treatment", metric: "alive_agents" };

test("lost admission retains one immutable exact request and retries its original identity", async () => {
  const h = await harness();
  const input = request();
  const promise = h.controller.launch(input);
  const first = h.calls.create[0];
  input.seeds[0] = "7";
  input.variants[0].brain_family = "assembly";
  first.reject(new TypeError("Connection disappeared after submission"));
  assert.equal(await promise, null);
  assert.equal(h.controller.creation.state, "unconfirmed");
  assert.equal(h.controller.canLaunch, false);
  assert.equal(h.controller.canSelect, false);
  assert.equal(h.controller.canRetry, true);
  assert.match(first.request.body, /"seeds":\[9007199254740993,18446744073709551615\]/);
  assert.ok(Object.isFrozen(first.request.request.variants));
  assert.equal(await h.controller.launch(request()), null);
  assert.equal(h.calls.create.length, 1);
  const retried = h.controller.retryLaunch();
  const second = h.calls.create[1];
  assert.deepEqual(second.request, first.request);
  second.resolve(status());
  assert.equal(await retried, true);
  assert.equal(h.controller.creation.state, "observed");
  assert.equal(h.controller.creation.experimentId, "cohort");
  assert.equal(h.controller.selected.runs[0].seed, "9007199254740993");
});

test("definite first refusal permits correction but later refusal cannot erase prior uncertainty", async () => {
  const h = await harness();
  const first = h.controller.launch(request());
  h.calls.create[0].reject(new RemoteOwnerHttpError(422, "invalid initial configuration"));
  await first;
  assert.equal(h.controller.creation.state, "refused");
  assert.equal(h.controller.canLaunch, true);
  const next = h.controller.launch(request());
  h.calls.create[1].reject(new RemoteOwnerHttpError(503, "publication response unavailable"));
  await next;
  assert.equal(h.controller.creation.state, "unconfirmed");
  const retry = h.controller.retryLaunch();
  h.calls.create[2].reject(new RemoteOwnerHttpError(409, "conflict after uncertain admission"));
  await retry;
  assert.equal(h.controller.creation.state, "unconfirmed");
  assert.equal(h.controller.canLaunch, false);
  assert.equal(h.calls.create[2].request.body, h.calls.create[1].request.body);
});

test("hiding the panel aborts admission and keeps its retry bytes without accepting late evidence", async () => {
  const h = await harness();
  const admission = h.controller.launch(request());
  const pending = h.calls.create[0];
  h.controller.setEnabled(false);
  assert.equal(pending.signal.aborted, true);
  assert.equal(h.controller.creation.state, "unconfirmed");
  pending.resolve(status());
  assert.equal(await admission, null);
  assert.equal(h.controller.selected, null);
  h.controller.setEnabled(true);
  await flush();
  assert.equal(h.calls.create.length, 1, "reopening must not submit scientific work");
  assert.equal(h.controller.canRetry, true);
  const retried = h.controller.retryLaunch();
  assert.equal(h.calls.create[1].request.body, pending.request.body);
  h.calls.create[1].resolve(status());
  await retried;
  assert.equal(h.controller.selected.experiment_id, "cohort");
});

test("source changes fence late receipts and never resubmit work to the replacement host", async () => {
  const h = await harness();
  const admission = h.controller.launch(request());
  const pending = h.calls.create[0];
  h.controller.observe({ identity: { sessionId: "99" } }, "wss://other.example/ws");
  assert.equal(pending.signal.aborted, true);
  pending.resolve(status());
  assert.equal(await admission, null);
  assert.equal(h.controller.selected, null);
  assert.equal(h.controller.creation, null);
  assert.equal(h.controller.source.sessionId, "99");
  h.controller.poll();
  await flush();
  assert.equal(h.calls.create.length, 1);
  assert.equal(h.calls.resume.length, 0);
  assert.equal(h.calls.list.at(-1).request.sessionId, "99");
});

test("opening retained experiments observes history and pages runs without executing or resuming", async () => {
  const observed = status({ id: "retained", seeds: Array.from({ length: 10 }, (_, i) => String(i)), state: "stopped" });
  const h = await harness(catalog([observed]));
  const opening = h.controller.open("retained");
  h.calls.status[0].resolve(observed);
  assert.equal(await opening, true);
  assert.equal(h.calls.create.length, 0);
  assert.equal(h.calls.resume.length, 0);
  assert.equal(h.controller.creation, null);
  assert.equal(h.controller.canResume, true);
  assert.equal(h.controller.visibleRuns.length, EXPERIMENT_RUN_PAGE_SIZE);
  const first = h.controller.visibleRuns.map((run) => run.run_id);
  h.controller.nextRuns();
  assert.equal(h.controller.visibleRuns.length, 4);
  assert.ok(h.controller.visibleRuns.every((run) => !first.includes(run.run_id)));
  h.controller.nextRuns();
  assert.equal(h.controller.runPageIndex, 1);
  h.controller.previousRuns();
  assert.equal(h.controller.runPageIndex, 0);
  assert.equal(await h.controller.open("unobserved"), null);
  assert.equal(h.calls.status.length, 1);
});

test("polling is bounded and refuses changed plans or regressing status generations", async () => {
  const h = await harness();
  await launch(h);
  h.controller.poll();
  assert.equal(h.calls.status.length, 0);
  h.advance(1999);
  h.controller.poll();
  assert.equal(h.calls.status.length, 0);
  h.advance(1);
  h.controller.poll();
  h.controller.poll();
  assert.equal(h.calls.status.length, 1, "one operation occupies the polling slot");
  const wrong = status({ generation: 2 });
  wrong.plan_digest = "c".repeat(64);
  h.calls.status[0].resolve(wrong);
  await flush();
  assert.equal(h.controller.selected.plan_digest, PLAN);
  assert.match(h.controller.notice, /plan changed/);
  const refreshed = h.controller.refreshStatus(true);
  h.calls.status[1].resolve(status({ state: "completed", generation: 2 }));
  await refreshed;
  assert.equal(h.controller.selected.status, "completed");
  const stale = h.controller.refreshStatus(true);
  h.calls.status[2].resolve(status({ generation: 1 }));
  await stale;
  assert.equal(h.controller.selected.generation, 2);
  assert.equal(h.controller.selected.status, "completed");
  assert.match(h.controller.notice, /stale/);
});

test("cancel and resume only change displayed lifecycle after observed server responses", async () => {
  const h = await harness();
  await launch(h);
  const cancelling = h.controller.cancel();
  assert.equal(h.controller.selected.status, "running");
  h.calls.cancel[0].resolve(status({ state: "cancelling", generation: 2 }));
  await cancelling;
  assert.equal(h.controller.selected.status, "cancelling");
  assert.equal(h.controller.canResume, false);
  const refresh = h.controller.refreshStatus(true);
  h.calls.status[0].resolve(status({ state: "cancelled", generation: 3 }));
  await refresh;
  assert.equal(h.controller.canResume, true);
  assert.ok(h.controller.selected.runs.every((run) => run.total_ticks === 0));
  const resumed = h.controller.resume();
  assert.equal(h.controller.selected.status, "cancelled");
  h.calls.resume[0].resolve(status({ generation: 4 }));
  await resumed;
  assert.equal(h.controller.selected.status, "running");
  assert.equal(h.calls.resume[0].request.experimentId, "cohort");
});

test("comparisons require complete cohorts and bind the exact selected pair and generation", async () => {
  const h = await harness();
  await launch(h);
  assert.equal(await h.controller.compare(pair), null);
  assert.equal(h.calls.compare.length, 0);
  const terminal = status({ state: "completed", generation: 2 });
  const refresh = h.controller.refreshStatus(true);
  h.calls.status[0].resolve(terminal);
  await refresh;
  assert.equal(h.controller.canCompare, true);
  const comparing = h.controller.compare(pair);
  assert.equal(h.controller.canCompare, false, "selection controls are disabled during the request");
  assert.deepEqual(h.calls.compare[0].request, {
    sessionId: SESSION, experimentId: "cohort",
    baseline_variant: "control", challenger_variant: "treatment", metric: "alive_agents",
  });
  h.calls.compare[0].resolve(comparison(terminal));
  await comparing;
  assert.equal(h.controller.comparison.mean_difference, -2);
  assert.equal(h.controller.comparison.matched_seeds[1], "18446744073709551615");
  const again = h.controller.compare(pair);
  const stale = comparison(terminal);
  stale.status_generation = 1;
  h.calls.compare[1].resolve(stale);
  await again;
  assert.equal(h.controller.comparison, null);
  assert.match(h.controller.notice, /observed plan and generation/);
  assert.equal(await h.controller.compare({ ...pair, challengerVariant: "control" }), null);
  assert.equal(h.calls.compare.length, 2);
});

test("one seed stays descriptive and source reset invalidates a late comparison", async () => {
  const input = request();
  input.seeds = ["18446744073709551615"];
  const terminal = status({ seeds: input.seeds, state: "completed" });
  const h = await harness();
  await launch(h, terminal, input);
  const reading = h.controller.compare(pair);
  const descriptive = comparison(terminal);
  Object.assign(descriptive, { descriptive_only: true, ci_95: null, standardized_effect: null,
    p_value: null, adjusted_p_value: null, significant: null,
    underpowered_reason: "Only one independent seed pair",
    undefined_effect_reason: "Only one independent seed pair" });
  h.calls.compare[0].resolve(descriptive);
  await reading;
  assert.equal(h.controller.comparison.significant, null);
  assert.match(h.controller.notice, /descriptive/);
  const late = h.controller.compare(pair);
  h.controller.reset("Disconnected from the host.");
  assert.equal(h.calls.compare[1].signal.aborted, true);
  h.calls.compare[1].resolve(descriptive);
  await late;
  assert.equal(h.controller.comparison, null);
  assert.equal(h.controller.selected, null);
});

test("invalid input never reaches the host and an inconsistent receipt remains unconfirmed", async () => {
  const h = await harness();
  const invalid = request();
  invalid.seeds = ["7", "7"];
  assert.equal(await h.controller.launch(invalid), null);
  assert.equal(h.calls.create.length, 0);
  assert.match(h.controller.notice, /duplicate seeds/);
  const admission = h.controller.launch(request());
  const malformed = status();
  malformed.total_runs += 1;
  h.calls.create[0].resolve(malformed);
  await admission;
  assert.equal(h.controller.creation.state, "unconfirmed");
  assert.equal(h.controller.selected, null);
  assert.equal(h.controller.canRetry, true);
});
