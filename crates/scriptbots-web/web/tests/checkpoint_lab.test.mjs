import assert from "node:assert/strict";
import { createHash, webcrypto } from "node:crypto";
import test from "node:test";
import {
  CHECKPOINT_LAB_PAGE_SIZE, MAX_LAB_ARTIFACT_BYTES, CheckpointLabController,
  renderCheckpointLab, validateLabArtifact, validateLabArtifactPage, validateLabBranchStatus,
  validateLabCheckpoint, validateLabCheckpointPage, validateLabIntervention,
} from "../checkpoint-lab.js";
import { RemoteOwnerHttpError, RemoteStreamClient } from "../remote.js";

const SESSION = "18446744073709551610";
const DIGEST = "a".repeat(64);
const SOURCE = "wss://lab.example/nested/api/ws/stream";
const flush = () => new Promise((resolve) => setImmediate(resolve));
const frame = (sessionId = SESSION) => ({ identity: { sessionId, tick: "120" }, snapshot: { epoch: "0" } });

function checkpoint(changes = {}) {
  return { checkpoint_id: "ckpt-saved-t100", tick: 100, agent_count: 12, byte_size: 1024,
    world_digest_v1: DIGEST, checksum_blake3: "b".repeat(64), checksum_sha256: "c".repeat(64),
    schema: "scriptbots.world-checkpoint.v1.3", created_at_utc: "1794000000", ...changes };
}

function page(items = [checkpoint()], changes = {}) {
  return { items, total: items.length, limit: CHECKPOINT_LAB_PAGE_SIZE, offset: 0, has_more: false,
    retention: { max_count: 128, max_bytes: 64 * 1024 * 1024, file_count: items.length,
      file_bytes: items.reduce((sum, row) => sum + row.byte_size, 0) }, ...changes };
}

function status(request, source = checkpoint(), changes = {}) {
  return { branch_id: "branch-one", checkpoint_id: request.checkpoint_id, branch_name: request.branch_name,
    status: "running", source_tick: source.tick, target_tick: source.tick + request.ticks,
    world_digest_v1: source.world_digest_v1, ...changes };
}

function artifact(changes = {}) {
  return { artifact_id: "checkpoint-branch-report", filename: "study_report.json", content_type: "application/json",
    byte_size: 2, checksum_blake3: DIGEST, checksum_sha256: createHash("sha256").update("{}").digest("hex"),
    created_at_utc: "1794000000", relative_path: "branches/branch-one/study_report.json", ...changes };
}

function artifactPage(items = [artifact()], changes = {}) {
  return { items, total: items.length, limit: CHECKPOINT_LAB_PAGE_SIZE, offset: 0, has_more: false, ...changes };
}

async function harness() {
  let time = 0, nextId = 0;
  const queues = Object.fromEntries(["save", "launch", "status", "cancel", "artifacts", "download"].map((kind) => [kind, []]));
  const services = Object.fromEntries(Object.keys(queues).map((kind) => [kind, (request, signal) => new Promise((resolve, reject) => {
    queues[kind].push({ request, signal, resolve, reject });
  })]));
  const reads = [];
  const controller = new CheckpointLabController({ ...services,
    list: async (request) => { reads.push(request); return page(); },
    id: () => `browser-lab-${++nextId}`, now: () => time });
  controller.observe(frame(), SOURCE);
  controller.setEnabled(true);
  await flush();
  controller.select(checkpoint().checkpoint_id);
  return { controller, queues, reads, advance: (ms = 2000) => { time += ms; }, identities: () => nextId };
}

async function launch(h, intervention = null) {
  const promise = h.controller.launch({ branchName: "future", ticks: 120, intervention });
  const pending = h.queues.launch.at(-1);
  pending.resolve(status(pending.request));
  await promise;
  return h.controller.branch;
}

test("checkpoint selection requires exact metadata and complete bounded page cursors", () => {
  assert.equal(validateLabCheckpoint(checkpoint()).tick, 100);
  assert.equal(validateLabCheckpointPage(page()).items.length, 1);
  for (const changes of [{ tick: Number.MAX_SAFE_INTEGER + 1 }, { byte_size: Number.MAX_SAFE_INTEGER + 1 },
    { byte_size: 0 }, { checkpoint_id: undefined }, { checkpoint_id: ["saved-cut"] },
    { world_digest_v1: [DIGEST] }, { checksum_blake3: [DIGEST] }, { checksum_sha256: [DIGEST] },
    { agent_count: 1025 }, { world_digest_v1: "not-a-digest" }, { checkpoint_id: "../wrong" },
    { schema: "render-frame" }, { checksum_sha256: "C".repeat(64) }]) {
    assert.throws(() => validateLabCheckpoint(checkpoint(changes)));
  }
  const rows = Array.from({ length: 12 }, (_, index) => checkpoint({ checkpoint_id: `ckpt-${index}` }));
  assert.equal(validateLabCheckpointPage(page(rows, { total: 13, has_more: true, next_cursor: "12" })).next_cursor, "12");
  for (const invalid of [page([checkpoint(), checkpoint()]), page([], { total: 1, has_more: true, next_cursor: "1" }),
    page(rows, { total: 13, has_more: true, next_cursor: "13" }), page([checkpoint()], { offset: 1 }),
    page([checkpoint()], { total: Number.MAX_SAFE_INTEGER + 1 }), page([], { retention: null })]) {
    assert.throws(() => validateLabCheckpointPage(invalid));
  }
});

test("study treatments match real PascalCase server enums and reject silent substitutions", () => {
  const treatments = [null,
    { type: "Drought", params: { region: null, duration_ticks: 12, growth_scale: 0.1 } },
    { type: "ResourceShock", params: { region: null, bloom_amount: 0.2, scorch: 0 } },
    { type: "TemperatureShift", params: { discomfort_rate: 0.2, comfort_band: 0.1 } },
    { type: "ClosedWorld", params: { closed: true } },
    { type: "Core", params: { kind: "embargo", region: { shape: "all" }, ticks: 3 } }];
  for (const value of treatments) assert.equal(validateLabIntervention(value), value);
  for (const value of [
    { type: "drought", params: treatments[1].params },
    { ...treatments[1], ignored: "wrong" },
    { type: "Drought", params: { ...treatments[1].params, duration_ticks: 0 } },
    { type: "ResourceShock", params: { region: null, bloom_amount: 0, scorch: 0 } },
    { type: "ResourceShock", params: { region: null, bloom_amount: Infinity, scorch: 0 } },
    { type: "TemperatureShift", params: { discomfort_rate: -1, comfort_band: 1 } },
    { type: "ClosedWorld", params: { closed: "true" } },
    { type: "Core", params: { kind: "spawn_from_archive" } },
    { type: "ConfigPatch", params: { uncontrolled: true } },
  ]) assert.throws(() => validateLabIntervention(value));
});

test("branch observations bind the saved digest and horizon and cannot regress a terminal state", () => {
  const request = { checkpoint_id: checkpoint().checkpoint_id, branch_name: "future", ticks: 120 };
  const running = status(request);
  assert.equal(validateLabBranchStatus(running, request, checkpoint()), running);
  const complete = status(request, checkpoint(), { status: "completed", artifact_id: artifact().artifact_id });
  validateLabBranchStatus(complete, request, checkpoint(), running);
  for (const changes of [{ checkpoint_id: "ckpt-other" }, { world_digest_v1: "d".repeat(64) },
    { target_tick: 221 }, { source_tick: 101 }, { branch_name: "different" }, { status: "completed" },
    { artifact_id: "premature-report" }, { status: "fake" }, { branch_id: undefined }, { branch_id: ["branch-one"] }]) {
    assert.throws(() => validateLabBranchStatus({ ...running, ...changes }, request, checkpoint()));
  }
  assert.throws(() => validateLabBranchStatus(running, request, checkpoint(), complete));
  assert.throws(() => validateLabBranchStatus({ ...complete, artifact_id: "different" }, request, checkpoint(), complete));
  assert.throws(() => validateLabBranchStatus(running, request, checkpoint(), { ...running, status: "cancelling" }));
  assert.throws(() => validateLabBranchStatus({ ...complete, artifact_id: [artifact().artifact_id] }, request, checkpoint()));
});

test("recovered interrupted studies remain terminal without automatic reruns and permit an explicit new study", async () => {
  const h = await harness();
  const branch = await launch(h);
  const stopped = status(branch.request, checkpoint(), { status: "stopped", error_reason: "worker interrupted by host restart" });
  const observed = h.controller.refreshBranch(true);
  h.queues.status[0].resolve(stopped);
  await observed;
  assert.equal(h.controller.branch.status.status, "stopped");
  assert.equal(h.controller.branchState, "observed");
  assert.equal(h.controller.canLaunch, true);
  assert.match(h.controller.notice, /host interruption.*not rerun automatically/);
  assert.equal(await h.controller.retryLaunch(), null);
  assert.equal(await h.controller.cancelBranch(), null);
  assert.equal(await h.controller.downloadReport(), null);
  assert.equal(await h.controller.refreshArtifacts(true), null);
  h.advance();
  for (let i = 0; i < 20; i += 1) h.controller.observe(frame(), SOURCE);
  assert.equal(h.queues.status.length, 1);
  assert.equal(h.queues.launch.length, 1);
  assert.equal(h.queues.cancel.length, 0);
  assert.equal(h.queues.download.length, 0);
  assert.equal(h.queues.artifacts.length, 0);
  validateLabBranchStatus(stopped, branch.request, branch.checkpoint, stopped);
  assert.throws(() => validateLabBranchStatus(status(branch.request), branch.request, branch.checkpoint, stopped));
  assert.throws(() => validateLabBranchStatus(status(branch.request, checkpoint(),
    { status: "completed", artifact_id: artifact().artifact_id }), branch.request, branch.checkpoint, stopped));
  const explicit = h.controller.launch({ branchName: "new_after_interruption", ticks: 25 });
  assert.notEqual(h.queues.launch[1].request.idempotency_key, branch.request.idempotency_key);
  h.queues.launch[1].resolve(status(h.queues.launch[1].request, checkpoint(), { branch_id: "branch-new" }));
  await explicit;
  assert.equal(h.controller.branch.status.branch_id, "branch-new");
  assert.equal(h.controller.branch.status.status, "running");
});

test("a saved-cut receipt, not submission, selects the new checkpoint", async () => {
  const h = await harness(), before = h.controller.selected;
  const pending = h.controller.save();
  assert.equal(h.controller.saveState, "pending");
  assert.equal(h.controller.selected, before);
  assert.equal(h.controller.canLaunch, false);
  assert.equal(h.controller.busy, true);
  const saved = checkpoint({ checkpoint_id: "ckpt-new-t125", tick: 125, agent_count: 18 });
  h.queues.save[0].resolve(saved);
  await pending;
  assert.equal(h.controller.saveState, "saved");
  assert.equal(h.controller.selected.checkpoint_id, saved.checkpoint_id);
  assert.equal(h.controller.selected.tick, 125);
  assert.notEqual(h.controller.selected, saved);
  assert.ok(Object.isFrozen(h.controller.saveRequest));
  assert.equal(h.controller.saveRequest.sessionId, SESSION);
  assert.equal(h.controller.canLaunch, true);
});

test("an uncertain save retains one immutable retry identity and forbids a fresh duplicate save", async () => {
  const h = await harness();
  const first = h.controller.save();
  h.queues.save[0].reject(new Error("connection lost after possible admission"));
  await first;
  assert.equal(h.controller.saveState, "unconfirmed");
  assert.equal(await h.controller.save(), null);
  assert.equal(h.identities(), 1);
  const retry = h.controller.retrySave();
  assert.equal(h.queues.save[1].request, h.queues.save[0].request);
  h.queues.save[1].resolve(checkpoint({ checkpoint_id: "ckpt-retried" }));
  await retry;
  assert.equal(h.controller.saveState, "saved");
  assert.equal(h.identities(), 1);
});

test("study retries preserve the exact checkpoint and detached treatment after a lost admission response", async () => {
  const h = await harness();
  const treatment = { type: "Drought", params: { region: null, duration_ticks: 10, growth_scale: 0.1 } };
  const first = h.controller.launch({ branchName: "dry", ticks: 25, intervention: treatment });
  treatment.params.growth_scale = 0.8;
  const original = h.queues.launch[0].request;
  assert.equal(original.intervention.params.growth_scale, 0.1);
  assert.ok(Object.isFrozen(original.intervention.params));
  h.queues.launch[0].reject(new Error("admission reply timed out"));
  await first;
  assert.equal(h.controller.branchState, "unconfirmed");
  assert.equal(await h.controller.launch({ branchName: "different", ticks: 5 }), null);
  assert.equal(h.identities(), 1);
  const retry = h.controller.retryLaunch();
  assert.equal(h.queues.launch[1].request, original);
  h.queues.launch[1].resolve(status(original));
  await retry;
  assert.equal(h.controller.branch.status.status, "running");
  assert.equal(h.controller.canLaunch, false);
});

test("definite first-attempt validation refusals allow corrected study and save requests", async () => {
  const h = await harness();
  const rejected = h.controller.launch({ branchName: "invalid_config", ticks: 120,
    intervention: { type: "TemperatureShift", params: { discomfort_rate: 1000, comfort_band: 0 } } });
  h.queues.launch[0].reject(new RemoteOwnerHttpError(400, "saved configuration refuses this treatment"));
  await rejected;
  assert.equal(h.controller.branchState, "refused");
  assert.equal(h.controller.canLaunch, true);
  const corrected = h.controller.launch({ branchName: "corrected", ticks: 5 });
  assert.notEqual(h.queues.launch[1].request.idempotency_key, h.queues.launch[0].request.idempotency_key);
  h.queues.launch[1].resolve(status(h.queues.launch[1].request));
  await corrected;
  assert.equal(h.controller.branch.status.target_tick, 105);

  const save = h.controller.save();
  h.queues.save[0].reject(new RemoteOwnerHttpError(400, "save validation refused"));
  await save;
  assert.equal(h.controller.saveState, "refused");
  assert.equal(h.controller.canSave, true);
  const anotherSave = h.controller.save();
  h.queues.save[1].resolve(checkpoint());
  await anotherSave;
  assert.notEqual(h.queues.save[1].request.idempotency_key, h.queues.save[0].request.idempotency_key);
});

test("capacity and storage ambiguity keep exact retries, and later refusal cannot erase prior uncertainty", async () => {
  for (const httpStatus of [503, 507]) {
    const h = await harness();
    const first = h.controller.launch({ branchName: "capacity", ticks: 120 });
    h.queues.launch[0].reject(new RemoteOwnerHttpError(httpStatus, "capacity currently unavailable"));
    await first;
    assert.equal(h.controller.branchState, "unconfirmed");
    assert.equal(h.controller.canLaunch, false);
    const rejectedRetry = h.controller.retryLaunch();
    h.queues.launch[1].reject(new RemoteOwnerHttpError(409, "source no longer matches"));
    await rejectedRetry;
    assert.equal(h.controller.branchState, "unconfirmed");
    const retry = h.controller.retryLaunch();
    h.queues.launch[2].resolve(status(h.queues.launch[2].request));
    await retry;
    assert.equal(h.queues.launch[2].request, h.queues.launch[0].request);
    assert.equal(h.controller.branch.status.status, "running");
  }
});

test("disconnect aborts a pending mutation, keeps its slot until settlement, and permits exact reconnect retry", async () => {
  const h = await harness();
  const pending = h.controller.save(), request = h.queues.save[0];
  h.controller.setEnabled(false);
  assert.equal(request.signal.aborted, true);
  assert.equal(h.controller.saveState, "unconfirmed");
  h.controller.observe(frame(), SOURCE);
  h.controller.setEnabled(true);
  assert.equal(h.controller.busy, true);
  assert.equal(await h.controller.retrySave(), null);
  request.resolve(checkpoint({ checkpoint_id: "ckpt-late" }));
  await pending;
  assert.equal(h.controller.saveReceipt, null);
  const retry = h.controller.retrySave();
  assert.equal(h.queues.save[1].request, request.request);
  h.queues.save[1].resolve(checkpoint({ checkpoint_id: "ckpt-late" }));
  await retry;
  assert.equal(h.controller.saveReceipt.checkpoint_id, "ckpt-late");
});

test("late source responses cannot become another host's selected checkpoint or study", async () => {
  for (const change of [() => frame("7"), () => frame()]) {
    const h = await harness();
    const pending = h.controller.launch({ branchName: "future", ticks: 120 });
    const request = h.queues.launch[0];
    h.controller.observe(change(), "wss://other.example/api/ws/stream");
    assert.equal(request.signal.aborted, true);
    request.resolve(status(request.request));
    await pending;
    assert.equal(h.controller.branch, null);
    assert.equal(h.controller.selected, null);
    assert.match(h.controller.notice, /previous host/);
  }
});

test("source replacement and hidden-panel cancellation do not invent a stopped remote job", async () => {
  const h = await harness();
  await launch(h);
  h.advance();
  h.controller.poll();
  assert.equal(h.queues.status.length, 1);
  h.controller.setEnabled(false);
  assert.equal(h.queues.status[0].signal.aborted, true);
  assert.equal(h.controller.branch.status.status, "running");
  h.queues.status[0].reject(new Error("aborted"));
  await flush();
  assert.equal(h.queues.cancel.length, 0);
  h.controller.reset();
  assert.equal(h.controller.branch, null);
  assert.equal(h.controller.source, null);
});

test("status polling is bounded and refuses changed evidence without replacing the last observed state", async () => {
  const h = await harness();
  const branch = await launch(h);
  for (let i = 0; i < 100; i += 1) h.controller.observe(frame(), SOURCE);
  assert.equal(h.queues.status.length, 0);
  h.advance();
  for (let i = 0; i < 100; i += 1) h.controller.observe(frame(), SOURCE);
  assert.equal(h.queues.status.length, 1);
  h.queues.status[0].resolve(status(branch.request, checkpoint(), { target_tick: 500 }));
  await flush();
  assert.equal(h.controller.branch.status.target_tick, 220);
  assert.match(h.controller.notice, /does not match/);
  h.controller.poll();
  assert.equal(h.queues.status.length, 1);
});

test("cancellation retains its observed outcome and completion can win the race", async () => {
  const h = await harness();
  const branch = await launch(h);
  const cancelling = h.controller.cancelBranch();
  assert.equal(h.controller.branch.status.status, "running");
  h.queues.cancel[0].resolve(status(branch.request, checkpoint(), { status: "completed", artifact_id: artifact().artifact_id }));
  await cancelling;
  assert.equal(h.controller.branch.status.status, "completed");
  assert.equal(await h.controller.cancelBranch(), null);
  assert.equal(h.queues.cancel.length, 1);
  h.controller.poll();
  assert.equal(h.queues.artifacts.length, 1);
  h.queues.artifacts[0].resolve(artifactPage());
  await flush();
  assert.equal(h.controller.artifactsPage.items[0].artifact_id, artifact().artifact_id);
});

test("artifact catalog pages must contain actual identities under the completed study's path", () => {
  assert.equal(validateLabArtifactPage(artifactPage(), "branch-one").total, 1);
  for (const changes of [{ relative_path: "branches/branch-other/study_report.json" },
    { relative_path: "branches/branch-one/../study_report.json" }, { filename: "../../report" },
    { byte_size: Number.MAX_SAFE_INTEGER + 1 }, { byte_size: -1 }, { checksum_sha256: "wrong" },
    { artifact_id: "../escape" }, { artifact_id: undefined }, { artifact_id: ["artifact-one"] },
    { checksum_blake3: [DIGEST] }, { checksum_sha256: [DIGEST] }]) {
    assert.throws(() => validateLabArtifactPage(artifactPage([artifact(changes)]), "branch-one"));
  }
  assert.throws(() => validateLabArtifact(artifact({ content_type: "text/html\r\nX-Foo: injected" })));
  assert.throws(() => validateLabArtifactPage(artifactPage(), ["branch-one"]));
});

test("large retained scientific cuts remain selectable study sources independently of browser download bounds", async () => {
  const h = await harness();
  const source = checkpoint({ checkpoint_id: "ckpt-large", byte_size: MAX_LAB_ARTIFACT_BYTES + 1 });
  h.controller.services.list = async () => page([source]);
  await h.controller.refresh(true);
  h.controller.select(source.checkpoint_id);
  assert.equal(h.controller.selected.byte_size, source.byte_size);
  assert.equal(h.controller.canLaunch, true);
  const pending = h.controller.launch({ branchName: "large_source", ticks: 25 });
  h.queues.launch[0].resolve(status(h.queues.launch[0].request, source));
  await pending;
  assert.equal(h.controller.branch.status.source_tick, source.tick);
  const largeArtifact = artifact({ byte_size: MAX_LAB_ARTIFACT_BYTES + 1 });
  assert.equal(validateLabArtifactPage(artifactPage([largeArtifact]), "branch-one").items[0], largeArtifact);
});

test("unrelated or changed downloads are refused before becoming downloadable files", async () => {
  const h = await harness();
  const pending = h.controller.downloadCheckpoint();
  assert.equal(h.queues.download[0].request.expected, h.controller.selected);
  h.queues.download[0].resolve({ blob: new Blob(["{}"]), filename: "study_report.json", metadata: artifact() });
  assert.equal(await pending, null);
  assert.match(h.controller.notice, /does not match/);
  const branch = await launch(h);
  const statusRead = h.controller.refreshBranch(true);
  h.queues.status[0].resolve(status(branch.request, checkpoint(), { status: "completed", artifact_id: artifact().artifact_id }));
  await statusRead;
  const report = h.controller.downloadReport();
  h.queues.download[1].resolve({ blob: new Blob(["{}"]), filename: "study_report.json",
    metadata: artifact({ relative_path: "branches/branch-other/study_report.json" }) });
  assert.equal(await report, null);
  const valid = h.controller.downloadReport();
  h.queues.download[2].resolve({ blob: new Blob(["{}"]), filename: "study_report.json", metadata: artifact() });
  assert.equal((await valid).blob.size, 2);
});

test("an existing catalog row cannot replace a selected cut with different scientific bytes", async () => {
  const h = await harness();
  const original = h.controller.selected;
  h.controller.services.list = async () => page([checkpoint({ world_digest_v1: "d".repeat(64) })]);
  assert.equal(await h.controller.refresh(true), null);
  assert.equal(h.controller.selected, original);
  assert.match(h.controller.notice, /changed its scientific identity/);
});

function client() {
  const value = new RemoteStreamClient({ url: `${SOURCE}?old=query#fragment`, decodeFrame() {} });
  value.state = "connected";
  value.tracker.acknowledge(SESSION);
  return value;
}

test("Lab transport preserves reverse-proxy prefixes and exact server-checked session headers", async () => {
  const previous = globalThis.fetch, requests = [];
  globalThis.fetch = async (url, options) => {
    requests.push({ url: new URL(url), ...options });
    return new Response("{}", { headers: { "Content-Type": "application/json" } });
  };
  try {
    const value = client();
    const save = { sessionId: SESSION, description: "Saved source", idempotency_key: "save-one" };
    const branch = { sessionId: SESSION, checkpoint_id: checkpoint().checkpoint_id, branch_name: "dry", ticks: 120,
      intervention: { type: "Drought", params: { region: null, duration_ticks: 20, growth_scale: 0.1 } }, idempotency_key: "branch-one" };
    await value.listCheckpoints({ sessionId: SESSION });
    await value.saveCheckpoint(save);
    await value.saveCheckpoint(save);
    await value.launchCheckpointBranch(branch);
    await value.launchCheckpointBranch(branch);
    await value.checkpointBranchStatus("branch-one", { sessionId: SESSION });
    await value.cancelCheckpointBranch("branch-one", { sessionId: SESSION });
    await value.checkpointBranchArtifacts("branch-one", { sessionId: SESSION, cursor: "12" });
    assert.deepEqual(requests.map((row) => row.url.pathname), [
      "/nested/api/v1/checkpoints", "/nested/api/v1/checkpoints", "/nested/api/v1/checkpoints",
      "/nested/api/v1/checkpoints/branches", "/nested/api/v1/checkpoints/branches",
      "/nested/api/v1/checkpoints/branches/branch-one", "/nested/api/v1/checkpoints/branches/branch-one/cancel",
      "/nested/api/v1/checkpoints/branches/branch-one/artifacts" ]);
    for (const request of requests) {
      assert.equal(request.url.protocol, "https:");
      assert.equal(request.url.hash, "");
      assert.equal(request.headers["X-ScriptBots-Session"], SESSION);
      assert.equal(request.credentials, "omit");
      assert.equal(request.cache, "no-store");
    }
    assert.equal(requests[1].body, requests[2].body);
    assert.equal(requests[3].body, requests[4].body);
    assert.equal(JSON.parse(requests[3].body).sessionId, undefined);
    assert.equal(JSON.parse(requests[3].body).idempotency_key, "branch-one");
    assert.equal(requests[7].url.search, "?limit=12&cursor=12");
    const count = requests.length;
    await assert.rejects(value.listCheckpoints({ sessionId: "7" }));
    await assert.rejects(value.listCheckpoints({ sessionId: SESSION, cursor: "01" }));
    await assert.rejects(value.checkpointBranchStatus("../escape", { sessionId: SESSION }));
    await assert.rejects(value.launchCheckpointBranch({ ...branch, ticks: 10001 }));
    assert.equal(requests.length, count);
  } finally { globalThis.fetch = previous; }
});

test("late HTTP responses after disconnect or reconnect are never current Lab receipts", async () => {
  const previous = globalThis.fetch;
  let respond;
  globalThis.fetch = () => new Promise((resolve) => { respond = resolve; });
  try {
    for (const disconnected of [true, false]) {
      const value = client();
      const pending = value.saveCheckpoint({ sessionId: SESSION, idempotency_key: "save-once" });
      if (disconnected) value.state = "disconnected";
      else value.generation += 1;
      respond(new Response(JSON.stringify(checkpoint())));
      await assert.rejects(pending, /connection changed/);
    }
  } finally { globalThis.fetch = previous; }
});

test("HTTP errors retain actual status and conservative admission semantics, including plain extractor refusals", async () => {
  const previous = globalThis.fetch;
  try {
    for (const statusCode of [400, 404, 409, 413, 415, 422, 503, 507]) {
      globalThis.fetch = async () => new Response(statusCode === 413 ? "body too large" : JSON.stringify({ message: "refused" }), { status: statusCode });
      await assert.rejects(client().saveCheckpoint({ sessionId: SESSION, idempotency_key: "request" }), (error) => {
        assert.ok(error instanceof RemoteOwnerHttpError);
        assert.equal(error.status, statusCode);
        assert.equal(error.admission, statusCode < 500 ? "not_admitted" : "unknown");
        return true;
      });
    }
  } finally { globalThis.fetch = previous; }
});

test("artifact downloads verify bytes against catalog SHA-256 and retained source before yielding a Blob", async () => {
  const previous = globalThis.fetch;
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
  const requests = [];
  let payload = "{}", metadata = artifact();
  globalThis.fetch = async (url, options) => {
    requests.push({ url: String(url), ...options });
    return String(url).endsWith("/download")
      ? new Response(payload, { headers: { "Content-Length": String(payload.length) } })
      : new Response(JSON.stringify(metadata));
  };
  try {
    const value = client();
    const result = await value.downloadArtifact(metadata.artifact_id, { sessionId: SESSION, expected: metadata });
    assert.equal(await result.blob.text(), "{}");
    assert.equal(result.filename, metadata.filename);
    assert.equal(requests[1].headers["X-ScriptBots-Session"], SESSION);
    payload = "[]";
    await assert.rejects(value.downloadArtifact(metadata.artifact_id, { sessionId: SESSION }), /SHA-256/);
    payload = "x";
    await assert.rejects(value.downloadArtifact(metadata.artifact_id, { sessionId: SESSION }), /size differs/);
    metadata = artifact({ byte_size: MAX_LAB_ARTIFACT_BYTES + 1 });
    const before = requests.length;
    await assert.rejects(value.downloadArtifact(metadata.artifact_id, { sessionId: SESSION }), /browser download limit/);
    assert.equal(requests.length, before + 1, "oversized metadata refuses before the binary request");
    metadata = artifact();
    await assert.rejects(value.downloadArtifact(metadata.artifact_id,
      { sessionId: SESSION, expected: { ...metadata, checksum_sha256: "d".repeat(64) } }), /selected saved source/);
  } finally {
    globalThis.fetch = previous;
    if (descriptor) Object.defineProperty(globalThis, "crypto", descriptor);
    else Reflect.deleteProperty(globalThis, "crypto");
  }
});

test("empty retained bundle logs are catalogued and verified before becoming downloadable files", async () => {
  const previous = globalThis.fetch;
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
  const metadata = artifact({ filename: "events.jsonl", relative_path: "branches/branch-one/bundle_control/events.jsonl",
    content_type: "application/octet-stream", byte_size: 0, checksum_sha256: createHash("sha256").update("").digest("hex") });
  globalThis.fetch = async (url) => String(url).endsWith("/download")
    ? new Response("", { headers: { "Content-Length": "0" } }) : new Response(JSON.stringify(metadata));
  try {
    assert.equal(validateLabArtifactPage(artifactPage([metadata]), "branch-one").items[0].byte_size, 0);
    const result = await client().downloadArtifact(metadata.artifact_id, { sessionId: SESSION, expected: metadata });
    assert.equal(result.blob.size, 0);
    assert.equal(await result.blob.text(), "");
    assert.equal(result.filename, "events.jsonl");
    assert.throws(() => validateLabCheckpoint(checkpoint({ byte_size: 0 })));
  } finally {
    globalThis.fetch = previous;
    if (descriptor) Object.defineProperty(globalThis, "crypto", descriptor);
    else Reflect.deleteProperty(globalThis, "crypto");
  }
});

test("actual download stream bounds catch missing or dishonest Content-Length", async () => {
  const previous = globalThis.fetch;
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
  let cancelled = false, short = false;
  globalThis.fetch = async (url) => String(url).endsWith("/download")
    ? new Response(new ReadableStream({
      start(controller) { controller.enqueue(new TextEncoder().encode(short ? "x" : "too many bytes")); if (short) controller.close(); },
      cancel() { cancelled = true; },
    })) : new Response(JSON.stringify(artifact()));
  try {
    await assert.rejects(client().downloadArtifact(artifact().artifact_id, { sessionId: SESSION }), /byte bound/);
    assert.equal(cancelled, true);
    short = true;
    await assert.rejects(client().downloadArtifact(artifact().artifact_id, { sessionId: SESSION }), /truncated/);
  } finally {
    globalThis.fetch = previous;
    if (descriptor) Object.defineProperty(globalThis, "crypto", descriptor);
    else Reflect.deleteProperty(globalThis, "crypto");
  }
});

test("visible Lab state distinguishes uncertain admission, observed running and verified report availability", async () => {
  const nodes = new Map();
  const document = { createElement: () => ({ value: "", textContent: "" }) };
  const panel = { dataset: {}, querySelector: (id) => {
    if (!nodes.has(id)) nodes.set(id, { dataset: {}, ownerDocument: document, textContent: "", disabled: false,
      replaceChildren(...options) { this.options = options; } });
    return nodes.get(id);
  } };
  const h = await harness();
  renderCheckpointLab(panel, h.controller);
  assert.equal(nodes.get("#lab-branch-launch").disabled, false);
  assert.equal(nodes.get("#lab-report-download").disabled, true);
  const pending = h.controller.launch({ branchName: "study", ticks: 120 });
  h.queues.launch[0].reject(new Error("response lost"));
  await pending;
  renderCheckpointLab(panel, h.controller);
  assert.match(nodes.get("#lab-branch-detail").textContent, /admission is unconfirmed/);
  assert.equal(nodes.get("#lab-branch-retry").disabled, false);
  assert.equal(nodes.get("#lab-branch-launch").disabled, true);
  assert.equal(nodes.get("#lab-report-download").disabled, true);
  const retry = h.controller.retryLaunch();
  h.queues.launch[1].resolve(status(h.queues.launch[1].request, checkpoint(), { status: "completed", artifact_id: artifact().artifact_id }));
  await retry;
  renderCheckpointLab(panel, h.controller);
  assert.equal(nodes.get("#lab-report-download").disabled, false);
  assert.match(nodes.get("#lab-branch-detail").textContent, /descriptive trajectory comparison/);
  h.controller.setEnabled(false);
  renderCheckpointLab(panel, h.controller);
  assert.equal(nodes.get("#lab-report-download").disabled, true);
});
