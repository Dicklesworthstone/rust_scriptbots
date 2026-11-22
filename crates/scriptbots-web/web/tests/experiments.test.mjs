import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
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

function failedStatus(id = "cohort") {
  const value = status({ id, state: "completed", generation: 9 });
  value.status = "failed";
  value.completed_runs -= 1;
  value.failed_runs = 1;
  Object.assign(value.runs[0], { state: "failed", total_ticks: 0, final_digest: null,
    bundle_path: null, error_reason: "checkpoint publication failed",
    checkpoint_progress: { state: "observed", tick: 5, checkpoint_digest: DIGEST, source: "published" } });
  return value;
}

function retriedStatus() {
  const value = failedStatus();
  value.generation += 1;
  value.status = "running";
  value.failed_runs = 0;
  Object.assign(value.runs[0], { state: "running", error_reason: null });
  return value;
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
  const calls = { list: [], create: [], status: [], cancel: [], resume: [], retryFailed: [], compare: [], artifacts: [], download: [] };
  let clock = 0, sequence = 0;
  const services = {};
  for (const kind of ["create", "status", "cancel", "resume", "retryFailed", "compare", "artifacts", "download"]) {
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

// Transfer fixtures use real SHA-256 bytes and a mutually consistent manifest
// index. They do not claim to be native checkpoints or scientific replay proof.
function bundleFixture(observed, runId, alterManifest = () => {}) {
  const run = observed.runs.find((run) => run.run_id === runId);
  const files = new Map();
  const artifacts = [];
  const add = (path, content) => {
    const bytes = new Uint8Array(typeof content === "string" ? Buffer.from(content) : content);
    const metadata = { artifact_id: "experiment-" + String(artifacts.length + 1).padStart(64, "0"),
      filename: path.split("/").at(-1), content_type: "application/octet-stream", byte_size: bytes.length,
      checksum_blake3: DIGEST, checksum_sha256: createHash("sha256").update(bytes).digest("hex"),
      created_at_utc: "1794456010", relative_path: `experiments/${observed.experiment_id}/${runId}/${path}` };
    artifacts.push(metadata);
    files.set(metadata.artifact_id, { blob: new Blob([bytes]), filename: metadata.filename, metadata, bytes });
  };
  add("exports/summary.csv", "123456789");
  add("evidence/run.json", '{"seed":18446744073709551615,"fixture":true}\n');
  add("checkpoints/final.checkpoint", new Uint8Array([0, 255, 128, 7, 0]));
  const manifest = { bundle_version: "scriptbots.run-bundle.v1", digests: { max_tick: run.total_ticks },
    artifacts: artifacts.map((artifact) => ({ relative_path: artifact.relative_path.split(runId + "/")[1],
      blake3_hex: artifact.checksum_blake3, bytes_len: artifact.byte_size, artifact_type: "fixture" })) };
  alterManifest(manifest);
  add("bundle_manifest.json", JSON.stringify(manifest));
  return { files, inventory: { experiment_id: observed.experiment_id, plan_digest: observed.plan_digest,
    status_generation: observed.generation, run: clone(run), artifacts,
    total_bytes: artifacts.reduce((total, artifact) => total + artifact.byte_size, 0) } };
}

async function loadBundleFixture(h, observed = failedStatus(), alterManifest) {
  await launch(h, observed);
  const fixture = bundleFixture(observed, h.controller.artifactRunId, alterManifest);
  const loading = h.controller.loadRunArtifacts();
  assert.equal(h.calls.artifacts.at(-1).request.runId, fixture.inventory.run.run_id);
  h.calls.artifacts.at(-1).resolve(clone(fixture.inventory));
  assert.ok(await loading);
  return fixture;
}

async function serveBundle(h, fixture, action = () => h.controller.downloadRunBundle(), mutate = () => {}) {
  const pending = action();
  let index = 0;
  const deadline = Date.now() + 5000;
  while (h.controller.busy) {
    const call = h.calls.download[index++];
    assert.ok(call, "each file is requested sequentially");
    const result = { ...fixture.files.get(call.request.artifactId) };
    mutate(result, index);
    call.resolve(result);
    // Hashing is asynchronous Web Crypto work, not just a microtask.
    while (h.controller.busy && h.calls.download.length < index + 1) {
      assert.ok(Date.now() < deadline, "bounded fixture download must settle");
      await new Promise((resolve) => setTimeout(resolve, 1));
    }
  }
  return pending;
}

test("a completed run downloads as a byte-identical portable ZIP despite a failed companion", async () => {
  const h = await harness();
  const fixture = await loadBundleFixture(h);
  assert.equal(h.controller.canDownloadBundle, true);
  assert.equal(h.controller.selectedArtifact.filename, "bundle_manifest.json");
  const output = await serveBundle(h, fixture);
  assert.equal(output.filename, fixture.inventory.run.run_id + ".zip");
  assert.equal(output.blob.type, "application/zip");
  assert.equal(h.calls.download.length, fixture.inventory.artifacts.length);
  for (const call of h.calls.download) {
    assert.equal(call.request.sessionId, SESSION);
    assert.ok(Object.isFrozen(call.request.expected));
  }
  const bytes = new Uint8Array(await output.blob.arrayBuffer()), view = new DataView(bytes.buffer);
  const entries = new Map();
  let position = 0;
  while (view.getUint32(position, true) === 0x04034b50) {
    assert.equal(view.getUint16(position + 8, true), 0, "original files are stored without compression");
    const size = view.getUint32(position + 18, true);
    const names = view.getUint16(position + 26, true);
    assert.equal(view.getUint16(position + 28, true), 0);
    const path = new TextDecoder().decode(bytes.slice(position + 30, position + 30 + names));
    const data = bytes.slice(position + 30 + names, position + 30 + names + size);
    entries.set(path, data);
    if (path === "exports/summary.csv") assert.equal(view.getUint32(position + 14, true), 0xcbf43926, "standard CRC32 check vector");
    position += 30 + names + size;
  }
  const centralOffset = position;
  for (const [path, data] of entries) {
    assert.equal(view.getUint32(position, true), 0x02014b50);
    const names = view.getUint16(position + 28, true);
    assert.equal(new TextDecoder().decode(bytes.slice(position + 46, position + 46 + names)), path);
    assert.equal(view.getUint32(position + 24, true), data.length);
    const localOffset = view.getUint32(position + 42, true);
    assert.equal(view.getUint32(localOffset, true), 0x04034b50);
    assert.equal(view.getUint32(localOffset + 14, true), view.getUint32(position + 16, true));
    position += 46 + names;
  }
  assert.equal(view.getUint32(position, true), 0x06054b50);
  assert.equal(view.getUint16(position + 10, true), entries.size);
  assert.equal(view.getUint32(position + 12, true), position - centralOffset);
  assert.equal(view.getUint32(position + 16, true), centralOffset);
  assert.equal(position + 22, bytes.length);
  for (const artifact of fixture.inventory.artifacts) {
    const path = artifact.relative_path.split(fixture.inventory.run.run_id + "/")[1];
    assert.deepEqual(entries.get(path), fixture.files.get(artifact.artifact_id).bytes);
  }
  assert.match(new TextDecoder().decode(entries.get("evidence/run.json")), /18446744073709551615/);
});

test("individual run files are verified and complete ZIP export refuses corruption or a truncated manifest index", async () => {
  const individual = await harness();
  const fixture = await loadBundleFixture(individual);
  const output = await serveBundle(individual, fixture, () => individual.controller.downloadSelectedArtifact());
  assert.equal(output.filename, "bundle_manifest.json");
  assert.equal(individual.calls.download.length, 1);
  for (const defect of ["bytes", "metadata", "manifest"]) {
    const h = await harness();
    const fixture = await loadBundleFixture(h, failedStatus(), (manifest) => {
      if (defect === "manifest") manifest.artifacts.pop();
    });
    const output = await serveBundle(h, fixture, undefined, (result, index) => {
      if (index !== 1) return;
      if (defect === "bytes") result.blob = new Blob(["987654321"]);
      if (defect === "metadata") result.metadata = { ...result.metadata, relative_path: "experiments/foreign/run/summary.csv" };
    });
    assert.equal(output, null, defect + " must never produce a downloadable ZIP");
    assert.match(h.controller.notice, /download unavailable/);
    if (defect !== "manifest") assert.equal(h.calls.download.length, 1);
    assert.equal(h.controller.canDownloadBundle, true, "failure permits an explicit fresh download");
  }
});

test("run inventory survives peer progress but is cleared on selection and source changes", async () => {
  const h = await harness();
  const fixture = await loadBundleFixture(h);
  const retained = h.controller.runArtifacts;
  const retry = h.controller.retryFailed();
  h.calls.retryFailed[0].resolve(retriedStatus());
  assert.ok(await retry);
  assert.equal(h.controller.runArtifacts, retained);
  const other = h.controller.completedRuns.find((run) => run.run_id !== fixture.inventory.run.run_id);
  h.controller.selectArtifactRun(other.run_id);
  assert.equal(h.controller.runArtifacts, null);
  assert.equal(h.controller.canDownloadBundle, false);
  const loading = h.controller.loadRunArtifacts();
  const pending = h.calls.artifacts.at(-1);
  h.controller.observe({ identity: { sessionId: "99" } }, "wss://another.example/ws");
  pending.resolve(fixture.inventory);
  assert.equal(await loading, null);
  assert.equal(pending.signal.aborted, true);
  assert.equal(h.controller.runArtifacts, null);
  assert.equal(h.controller.artifactRunId, null);
});

test("hiding the experiment panel aborts a bundle download without fetching further files or exporting late bytes", async () => {
  const h = await harness();
  const fixture = await loadBundleFixture(h);
  const pending = h.controller.downloadRunBundle();
  const first = h.calls.download[0];
  assert.equal(h.controller.selectArtifactRun(h.controller.completedRuns[1].run_id), false);
  h.controller.setEnabled(false);
  first.resolve(fixture.files.get(first.request.artifactId));
  assert.equal(await pending, null);
  assert.equal(first.signal.aborted, true);
  assert.equal(h.calls.download.length, 1);
  assert.equal(h.controller.busy, false);
});

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

test("failed-run retry captures one observed generation and retains completed peers", async () => {
  const h = await harness();
  const failed = failedStatus();
  await launch(h, failed);
  assert.equal(h.controller.canResume, false);
  assert.equal(h.controller.canRetryFailed, true);
  const pending = h.controller.retryFailed();
  assert.deepEqual(h.calls.retryFailed[0].request,
    { sessionId: SESSION, experimentId: "cohort", expectedGeneration: 9 });
  assert.equal(h.controller.canRetryFailed, false);
  assert.equal(await h.controller.retryFailed(), null);
  assert.equal(h.calls.retryFailed.length, 1);
  assert.equal(h.controller.selected.status, "failed", "admission is not an observed retry");
  h.calls.retryFailed[0].resolve(retriedStatus());
  assert.ok(await pending);
  assert.equal(h.controller.selected.status, "running");
  assert.equal(h.controller.selected.generation, 10);
  assert.deepEqual(h.controller.selected.runs.slice(1), failed.runs.slice(1));
  assert.equal(h.controller.canRetryFailed, false);
  assert.equal(h.calls.resume.length, 0, "retry never uses ordinary resume");
});

test("lost failed-run retry survives an unchanged refresh before its delayed generation arrives", async () => {
  const h = await harness();
  await launch(h, failedStatus());
  const pending = h.controller.retryFailed();
  h.calls.retryFailed[0].reject(new Error("response lost after admission"));
  assert.equal(await pending, null);
  assert.equal(h.controller.canRetryFailed, false);
  assert.match(h.controller.notice, /refresh status/);
  assert.equal(await h.controller.retryFailed(), null);
  assert.equal(h.calls.retryFailed.length, 1);
  const unchanged = h.controller.refreshStatus(true);
  h.calls.status.at(-1).resolve(failedStatus());
  assert.ok(await unchanged);
  assert.equal(h.controller.canRetryFailed, true);
  assert.equal(h.controller.failedRetry.generation, 9,
    "the original POST may still commit after this unchanged observation");
  const refresh = h.controller.refreshStatus(true);
  h.calls.status.at(-1).resolve(retriedStatus());
  assert.ok(await refresh);
  assert.equal(h.controller.selected.status, "running");
  assert.equal(h.controller.failedRetry, null);
  assert.equal(h.calls.retryFailed.length, 1, "observing delayed admission never resubmits work");
});

test("an unchanged failed-run refresh permits only an explicit reissue of the original generation", async () => {
  const h = await harness();
  await launch(h, failedStatus());
  const first = h.controller.retryFailed();
  h.calls.retryFailed[0].reject(new Error("response lost while retry was validating"));
  assert.equal(await first, null);
  const refresh = h.controller.refreshStatus(true);
  h.calls.status.at(-1).resolve(failedStatus());
  assert.ok(await refresh);
  assert.equal(h.calls.retryFailed.length, 1);

  const reissued = h.controller.retryFailed();
  assert.deepEqual(h.calls.retryFailed[1].request, h.calls.retryFailed[0].request);
  assert.equal(h.controller.failedRetry.refreshed, false);
  assert.equal(h.controller.canRetryFailed, false);
  assert.equal(await h.controller.retryFailed(), null);
  h.calls.retryFailed[1].reject(new RemoteOwnerHttpError(409, "original retry already committed"));
  assert.equal(await reissued, null);
  assert.equal(h.controller.canRetryFailed, false, "another read is required after reissue");

  const observed = h.controller.refreshStatus(true);
  h.calls.status.at(-1).resolve(retriedStatus());
  assert.ok(await observed);
  assert.equal(h.controller.selected.generation, 10);
  assert.equal(h.controller.failedRetry, null);
  assert.equal(h.calls.retryFailed.length, 2);
});

test("opening another failed experiment clears only the previous selection's retry authority", async () => {
  const first = failedStatus(), second = failedStatus("other-cohort");
  const h = await harness(catalog([first, second]));
  const opening = h.controller.open(first.experiment_id);
  h.calls.status.at(-1).resolve(first);
  assert.ok(await opening);
  const retry = h.controller.retryFailed();
  h.calls.retryFailed[0].reject(new Error("first experiment retry response lost"));
  assert.equal(await retry, null);
  assert.equal(h.controller.failedRetry.experimentId, first.experiment_id);

  const switching = h.controller.open(second.experiment_id);
  h.calls.status.at(-1).resolve(second);
  assert.ok(await switching);
  assert.equal(h.controller.selected.experiment_id, second.experiment_id);
  assert.equal(h.controller.failedRetry, null);
  assert.equal(h.controller.canRetryFailed, true);
  assert.equal(h.calls.retryFailed.length, 1, "selection never starts another attempt");
});

test("failed-run retry refuses unchanged acknowledgements and terminal peer substitution", async () => {
  const h = await harness();
  const failed = failedStatus();
  await launch(h, failed);
  const pending = h.controller.retryFailed();
  h.calls.retryFailed[0].resolve(failed);
  assert.equal(await pending, null);
  assert.equal(h.controller.canRetryFailed, false);
  assert.match(h.controller.notice, /newer retry generation/);
  const refresh = h.controller.refreshStatus(true);
  const forged = retriedStatus();
  forged.runs[1].final_digest = "c".repeat(16);
  h.calls.status.at(-1).resolve(forged);
  assert.equal(await refresh, null);
  assert.deepEqual(h.controller.selected, failed);
  assert.equal(h.controller.canRetryFailed, false);
  const unchanged = h.controller.refreshStatus(true);
  h.calls.status.at(-1).resolve(failed);
  await unchanged;
  assert.equal(h.controller.canRetryFailed, true,
    "a verified unchanged status establishes that the observed generation can be retried");
  assert.equal(h.controller.failedRetry.generation, failed.generation,
    "an unchanged observation cannot revoke authority for a delayed retry response");
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
