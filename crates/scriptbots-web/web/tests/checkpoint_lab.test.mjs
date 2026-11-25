import assert from "node:assert/strict";
import { createHash, webcrypto } from "node:crypto";
import test from "node:test";
import {
  CHECKPOINT_LAB_PAGE_SIZE, MAX_LAB_ARTIFACT_BYTES, MAX_LAB_TRAJECTORY_CATALOG_ARTIFACTS, CheckpointLabController,
  renderCheckpointLab, validateLabArtifact, validateLabArtifactPage, validateLabBranchObservation, validateLabBranchPage, validateLabBranchStatus,
  validateLabCheckpoint, validateLabCheckpointPage, validateLabIntervention, validateCheckpointCanvas,
  parseReplayTarget, validateCheckpointReplay, archiveStudyTreatment,
} from "../checkpoint-lab.js";
import { RemoteOwnerHttpError, RemoteStreamClient } from "../remote.js";
import { MAX_STUDY_REPORT_BYTES } from "../study-report.js";
import { MAX_STUDY_TRAJECTORY_BYTES } from "../study-trajectories.js";

const SESSION = "340282366920938463463374607431768211455";
const DIGEST = "a".repeat(64);
const WORLD_DIGEST = "a".repeat(16);
const SOURCE = "wss://lab.example/nested/api/ws/stream";
const flush = () => new Promise((resolve) => setImmediate(resolve));
const frame = (sessionId = SESSION) => ({ identity: { sessionId, tick: "120" }, snapshot: { epoch: "0" } });

function checkpoint(changes = {}) {
  return { checkpoint_id: "ckpt-saved-t100", tick: 100, agent_count: 12, byte_size: 1024,
    world_digest_v1: WORLD_DIGEST, checksum_blake3: "b".repeat(64), checksum_sha256: "c".repeat(64),
    schema: "scriptbots.world-checkpoint.v1.3", created_at_utc: "1794000000", ...changes };
}

function savedCanvas(source = checkpoint()) {
  return { schema: "scriptbots.checkpoint-canvas.v1", sourceSessionId: SESSION, checkpoint: source,
    frame: { schema: "scriptbots.browser-frame.v1",
      identity: { sessionId: SESSION, tick: String(source.tick), publicationRevision: "1",
        scientificRevision: "0", controlRevision: "0", configRevision: "0" },
      snapshot: { tick: source.tick, epoch: "0", world: { width: 100, height: 100 },
        summary: { agentCount: source.agent_count, averageEnergy: 1, averageHealth: 1, births: 0, deaths: 0 },
        agents: Array.from({ length: source.agent_count }, (_, index) => ({ uid: String(index + 1) })) },
      environment: { width: 1, height: 1, sourceWidth: 1, sourceHeight: 1, cellSize: 100,
        terrainRevision: "1", foodRevision: "1", hydrologyRevision: "0", foodMax: 1, hydrologyEnabled: false,
        cells: [{ terrain: "Grass", elevation: 0, food: 1, waterDepth: 0 }] },
      playback: { paused: true, speedMultiplier: 1 } } };
}

function replayCanvas(source = checkpoint(), targetTick = "112") {
  const value = savedCanvas(source);
  const savedCut = targetTick === String(source.tick);
  value.schema = "scriptbots.checkpoint-replay.v1";
  value.targetTick = targetTick;
  value.maxVerificationTicks = "1024";
  value.verification = savedCut ? { kind: "saved_cut" }
    : { kind: "captured_interval", verifiedThroughTick: "120", capturedCurrentDigest: "e".repeat(16) };
  value.frame.identity.tick = targetTick;
  value.frame.snapshot.tick = Number(targetTick);
  // A reconstructed later population need not have the checkpoint's size.
  if (!savedCut) value.frame.snapshot.agents.splice(0, 4);
  value.frame.snapshot.summary.agentCount = value.frame.snapshot.agents.length;
  return value;
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

function study(changes = {}) {
  return status({ checkpoint_id: checkpoint().checkpoint_id, branch_name: "retained", ticks: 120 },
    checkpoint(), changes);
}

function historyPage(items = [study()], changes = {}) {
  return { items, total: items.length, limit: CHECKPOINT_LAB_PAGE_SIZE, offset: 0, has_more: false, ...changes };
}

function labPanel() {
  const nodes = new Map();
  const document = { createElement: () => ({ value: "", textContent: "" }) };
  const panel = { dataset: {}, querySelector: (id) => {
    if (!nodes.has(id)) nodes.set(id, { dataset: {}, ownerDocument: document, textContent: "", disabled: false,
      replaceChildren(...options) { this.options = options; } });
    return nodes.get(id);
  } };
  return { panel, nodes };
}

function artifact(changes = {}) {
  return { artifact_id: "checkpoint-branch-report", filename: "study_report.json", content_type: "application/json",
    byte_size: 2, checksum_blake3: DIGEST, checksum_sha256: createHash("sha256").update("{}").digest("hex"),
    created_at_utc: "1794000000", relative_path: "branches/branch-one/study_report.json", ...changes };
}

function artifactPage(items = [artifact()], changes = {}) {
  return { items, total: items.length, limit: CHECKPOINT_LAB_PAGE_SIZE, offset: 0, has_more: false, ...changes };
}

async function harness({ listStudies, parseReport, parseTrajectories } = {}) {
  let time = 0, nextId = 0;
  const queues = Object.fromEntries(["save", "review", "replay", "launch", "status", "cancel", "artifacts", "download"].map((kind) => [kind, []]));
  const services = Object.fromEntries(Object.keys(queues).map((kind) => [kind, (request, signal) => new Promise((resolve, reject) => {
    queues[kind].push({ request, signal, resolve, reject });
  })]));
  const reads = [];
  const controller = new CheckpointLabController({ ...services, listStudies, parseReport, parseTrajectories,
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

test("saved canvas binds the exact host, metadata and reconstructed scientific cut", () => {
  const request = { sessionId: SESSION, checkpoint: checkpoint() };
  assert.equal(validateCheckpointCanvas(savedCanvas(), request).frame.identity.sessionId, SESSION);
  for (const change of [
    (value) => { value.sourceSessionId = "7"; },
    (value) => { value.frame.identity.sessionId = Number(SESSION); },
    (value) => { value.frame.identity.sessionId = "340282366920938463463374607431768211456"; },
    (value) => { value.frame.identity.publicationRevision = "18446744073709551616"; },
    (value) => { value.frame.identity.scientificRevision = "18446744073709551616"; },
    (value) => { value.frame.identity.controlRevision = "18446744073709551616"; },
    (value) => { value.frame.identity.configRevision = "18446744073709551616"; },
    (value) => { value.frame.identity.tick = "101"; },
    (value) => { value.frame.snapshot.tick = 101; },
    (value) => { value.frame.snapshot.summary.agentCount = 11; },
    (value) => { value.frame.snapshot.agents.pop(); },
    (value) => { value.frame.playback.paused = false; },
    (value) => { value.frame.playback.speedMultiplier = 2; },
    (value) => { value.frame.environment = null; },
    (value) => { value.checkpoint.checksum_sha256 = "d".repeat(64); },
    (value) => { value.checkpoint.checksum_blake3 = "d".repeat(64); },
    (value) => { value.checkpoint.world_digest_v1 = "d".repeat(16); },
  ]) {
    const value = savedCanvas();
    change(value);
    assert.throws(() => validateCheckpointCanvas(value, request));
  }
});

test("checkpoint observations refuse out-of-range sessions without widening epoch or tick", async () => {
  const h = await harness();
  assert.equal(h.controller.source.sessionId, SESSION);
  for (const sessionId of [Number(SESSION), "01", "-1", "340282366920938463463374607431768211456"]) {
    assert.throws(() => h.controller.observe(frame(sessionId), SOURCE), /identities/);
    assert.equal(h.controller.source.sessionId, SESSION);
  }
  for (const field of ["epoch", "tick"]) {
    const invalid = frame();
    (field === "epoch" ? invalid.snapshot : invalid.identity)[field] = "18446744073709551616";
    assert.throws(() => h.controller.observe(invalid, SOURCE), /identities/);
    assert.equal(h.controller.source.sessionId, SESSION);
  }
  assert.equal(h.reads.length, 1, "invalid observations cannot schedule checkpoint reads");
});

test("explicit saved-world review is immutable and read-only until return to live", async () => {
  const h = await harness();
  const pending = h.controller.reviewSelected();
  assert.equal(h.controller.canvasPreview, null, "live presentation remains until reconstruction succeeds");
  assert.equal(await h.controller.reviewSelected(), null, "one request owns the slot");
  assert.deepEqual(h.queues.review[0].request, { sessionId: SESSION, checkpoint: checkpoint() });
  h.queues.review[0].resolve(savedCanvas());
  const preview = await pending;
  assert.equal(h.controller.canvasPreview, preview);
  assert.equal(Object.isFrozen(preview.frame.snapshot.agents), true);
  assert.equal(h.controller.canSave, false);
  assert.equal(h.controller.canLaunch, false);
  assert.equal(await h.controller.save(), null);
  assert.equal(await h.controller.launch({ branchName: "blocked", ticks: 120 }), null);
  assert.equal(h.queues.save.length, 0);
  assert.equal(h.queues.launch.length, 0);
  const { panel, nodes } = labPanel();
  renderCheckpointLab(panel, h.controller);
  assert.equal(nodes.get("#lab-checkpoint-save").disabled, true);
  assert.equal(nodes.get("#lab-checkpoint-review").disabled, false);
  assert.equal(nodes.get("#lab-branch-launch").disabled, true);
  h.controller.returnToLive();
  assert.equal(h.controller.canvasPreview, null);
  assert.equal(h.controller.canSave, true);
  assert.equal(h.controller.canLaunch, true);
  assert.equal(h.queues.review.length, 1);
});

test("return to live and source changes reject late reconstruction without freeing its slot early", async () => {
  for (const cancel of [
    (controller) => controller.returnToLive(),
    (controller) => controller.observe(frame("7"), SOURCE),
    (controller) => controller.setEnabled(false),
  ]) {
    const h = await harness();
    const pending = h.controller.reviewSelected();
    cancel(h.controller);
    assert.equal(h.queues.review[0].signal.aborted, true);
    assert.equal(h.controller.busy, true);
    assert.equal(await h.controller.reviewSelected(), null);
    h.queues.review[0].resolve(savedCanvas());
    assert.equal(await pending, null);
    assert.equal(h.controller.canvasPreview, null);
    assert.equal(h.controller.busy, false);
  }
});

test("a failed review retains the prior saved canvas and changing selection returns to live", async () => {
  const h = await harness();
  const pending = h.controller.reviewSelected();
  h.queues.review[0].resolve(savedCanvas());
  const prior = await pending;
  const refused = h.controller.reviewSelected();
  h.queues.review[1].reject(new Error("reconstruction busy"));
  await refused;
  assert.equal(h.controller.canvasPreview, prior);
  assert.match(h.controller.notice, /reconstruction busy/);
  h.controller.page = page([checkpoint(), checkpoint({ checkpoint_id: "ckpt-second" })]);
  h.controller.select("ckpt-second");
  assert.equal(h.controller.canvasPreview, null);
});

test("historical replay binds its target to the complete captured interval and actual population", () => {
  const request = { sessionId: SESSION, checkpoint: checkpoint(), targetTick: "112" };
  const value = replayCanvas();
  assert.equal(validateCheckpointReplay(value, request), value);
  assert.equal(value.frame.snapshot.summary.agentCount, 8);
  assert.equal(value.checkpoint.agent_count, 12);
  assert.deepEqual(value.verification, { kind: "captured_interval", verifiedThroughTick: "120", capturedCurrentDigest: "e".repeat(16) });
  value.maxVerificationTicks = "20";
  assert.equal(validateCheckpointReplay(value, request), value);
  value.maxVerificationTicks = "19";
  assert.throws(() => validateCheckpointReplay(value, request), /complete replay verification interval/,
    "source-to-target fits, but the complete captured interval exceeds the returned bound");
  const saved = replayCanvas(checkpoint(), "100");
  assert.equal(validateCheckpointReplay(saved, { ...request, targetTick: "100" }), saved);
  saved.verification = value.verification;
  assert.throws(() => validateCheckpointReplay(saved, { ...request, targetTick: "100" }), /saved-cut verification/);
  saved.verification = { kind: "saved_cut" };
  saved.frame.snapshot.agents.pop();
  saved.frame.snapshot.summary.agentCount -= 1;
  assert.throws(() => validateCheckpointReplay(saved, { ...request, targetTick: "100" }), /historical target/);
  const largeSource = checkpoint({ tick: Number.MAX_SAFE_INTEGER - 20 });
  const largeTarget = String(BigInt(Number.MAX_SAFE_INTEGER) - 10n);
  const large = replayCanvas(largeSource, largeTarget);
  large.verification.verifiedThroughTick = String(Number.MAX_SAFE_INTEGER);
  assert.equal(validateCheckpointReplay(large, { ...request, checkpoint: largeSource, targetTick: largeTarget }), large);
});

test("historical replay refuses inconsistent descriptors, interval evidence and reconstructed frames", () => {
  const request = { sessionId: SESSION, checkpoint: checkpoint(), targetTick: "112" };
  for (const change of [
    (value) => { value.schema = "scriptbots.checkpoint-canvas.v1"; },
    (value) => { value.sourceSessionId = "7"; },
    (value) => { value.checkpoint.checkpoint_id = "other-source"; },
    (value) => { value.checkpoint.tick = 101; },
    (value) => { value.checkpoint.agent_count = 11; },
    (value) => { value.checkpoint.byte_size = 2048; },
    (value) => { value.checkpoint.world_digest_v1 = "b".repeat(16); },
    (value) => { value.checkpoint.checksum_sha256 = "d".repeat(64); },
    (value) => { value.checkpoint.checksum_blake3 = "d".repeat(64); },
    (value) => { value.checkpoint.schema = "scriptbots.world-checkpoint.v1.4"; },
    (value) => { value.checkpoint.created_at_utc = "1794000001"; },
    (value) => { value.checkpoint.description = "substituted descriptor"; },
    (value) => { value.targetTick = 112; },
    (value) => { value.targetTick = "0112"; },
    (value) => { value.targetTick = "113"; },
    (value) => { value.maxVerificationTicks = 1024; },
    (value) => { value.maxVerificationTicks = "0"; },
    (value) => { value.maxVerificationTicks = "18446744073709551616"; },
    (value) => { value.verification = { kind: "saved_cut" }; },
    (value) => { value.verification = null; },
    (value) => { value.verification.verifiedThroughTick = "111"; },
    (value) => { value.verification.verifiedThroughTick = 120; },
    (value) => { value.verification.verifiedThroughTick = "18446744073709551616"; },
    (value) => { value.verification.capturedCurrentDigest = "e".repeat(64); },
    (value) => { delete value.verification.capturedCurrentDigest; },
    (value) => { value.verification.targetDigest = "a".repeat(16); },
    (value) => { value.frame.identity.sessionId = "7"; },
    (value) => { value.frame.identity.tick = "113"; },
    (value) => { value.frame.identity.publicationRevision = "18446744073709551616"; },
    (value) => { value.frame.snapshot.tick = 113; },
    (value) => { value.frame.snapshot.summary.agentCount = 12; },
    (value) => { value.frame.snapshot.agents.pop(); },
    (value) => { value.frame.playback.paused = false; },
    (value) => { value.frame.playback.speedMultiplier = 2; },
    (value) => { value.frame.environment = null; },
  ]) {
    const value = replayCanvas();
    change(value);
    assert.throws(() => validateCheckpointReplay(value, request));
  }
  assert.throws(() => validateCheckpointReplay(replayCanvas(checkpoint(), "99"), { ...request, targetTick: "99" }));
  assert.throws(() => validateCheckpointReplay(replayCanvas(), { ...request, targetTick: "9007199254740992" }), /cannot display/);
});

test("replay target drafts and checked steps preserve exact decimal identities", async () => {
  assert.equal(parseReplayTarget("0"), 0n);
  assert.equal(parseReplayTarget("9007199254740991"), 9007199254740991n);
  for (const value of ["", "01", "1e3", "1.0", " 112", "112 ", "-1", "+112", 112,
    "9007199254740992", "18446744073709551615", "18446744073709551616"]) {
    assert.throws(() => parseReplayTarget(value));
  }
  const h = await harness();
  assert.equal(h.controller.replayTickText, "100");
  h.controller.stepReplayTick(-1);
  assert.equal(h.controller.replayTickText, "100", "cannot step before the source");
  h.controller.stepReplayTick(10);
  h.controller.stepReplayTick(1);
  h.controller.stepReplayTick(-10);
  assert.equal(h.controller.replayTickText, "101");
  h.controller.editReplayTick("9007199254740991");
  h.controller.stepReplayTick(1);
  assert.equal(h.controller.replayTickText, "9007199254740991", "overflow cannot round to a neighboring tick");
  h.controller.stepReplayTick(-1);
  assert.equal(h.controller.replayTickText, "9007199254740990");
  const { panel, nodes } = labPanel();
  for (const invalid of ["99", "01", "1e3", "9007199254740992", "1".repeat(80)]) {
    h.controller.editReplayTick(invalid);
    renderCheckpointLab(panel, h.controller);
    assert.equal(h.controller.canSeekReplay, false);
    assert.equal(nodes.get("#lab-replay-seek").disabled, true);
    assert.equal(nodes.get("#lab-replay-tick").value, invalid.slice(0, 64));
    assert.equal(await h.controller.seekReplay(), null);
  }
  assert.equal(h.queues.replay.length, 0);
  const save = h.controller.save();
  h.queues.save[0].resolve(checkpoint({ tick: 120, checkpoint_id: "saved-new" }));
  await save;
  assert.equal(h.controller.replayTickText, "120", "saving also selects and initializes the new replay source");
});

test("explicit historical seeks freeze source and target and share the occupied Lab slot", async () => {
  const h = await harness();
  h.controller.editReplayTick("112");
  const pending = h.controller.seekReplay();
  const queued = h.queues.replay[0];
  assert.deepEqual(queued.request, { sessionId: SESSION, checkpoint: checkpoint(), targetTick: "112" });
  assert.equal(Object.isFrozen(queued.request), true);
  assert.equal(Object.isFrozen(queued.request.checkpoint), true);
  assert.equal(h.controller.canvasPreview, null);
  assert.equal(h.controller.reviewPending, true);
  assert.equal(await h.controller.seekReplay(), null);
  assert.equal(await h.controller.reviewSelected(), null);
  assert.equal(await h.controller.save(), null);
  const { panel, nodes } = labPanel();
  renderCheckpointLab(panel, h.controller);
  assert.equal(nodes.get("#lab-replay-tick").disabled, false, "a draft edit can supersede a seek");
  assert.equal(nodes.get("#lab-replay-seek").disabled, true, "the worker slot remains occupied");
  queued.resolve(replayCanvas());
  const preview = await pending;
  assert.equal(h.controller.canvasPreview, preview);
  assert.equal(Object.isFrozen(preview.verification), true);
  assert.equal(Object.isFrozen(preview.frame.snapshot.agents), true);
  assert.equal(h.controller.canSave, false);
  assert.equal(h.controller.canLaunch, false);
  assert.equal(h.controller.canSeekReplay, true);
  assert.equal(h.identities(), 0, "historical reads allocate no mutation idempotency key");
  assert.match(h.controller.notice, /through captured current tick 120/);
  assert.match(h.controller.notice, /not a separate target digest/);
});

test("edits, return to live, navigation and source changes reject late replay without releasing its slot", async () => {
  for (const [supersede, retain] of [
    [(controller) => controller.editReplayTick("114"), true],
    [(controller) => controller.returnToLive(), false],
    [(controller) => controller.setEnabled(false), true],
    [(controller) => controller.observe(frame("7"), SOURCE), false],
    [(controller) => controller.observe(frame(), "wss://other.example/api/ws/stream"), false],
  ]) {
    const h = await harness();
    const saved = h.controller.reviewSelected();
    h.queues.review[0].resolve(savedCanvas());
    const prior = await saved;
    h.controller.editReplayTick("113");
    const pending = h.controller.seekReplay();
    supersede(h.controller);
    assert.equal(h.queues.replay[0].signal.aborted, true);
    assert.equal(h.controller.busy, true);
    assert.equal(await h.controller.seekReplay(), null);
    h.queues.replay[0].resolve(replayCanvas(checkpoint(), "113"));
    assert.equal(await pending, null);
    assert.equal(h.controller.canvasPreview, retain ? prior : null);
    assert.equal(h.controller.busy, false);
    assert.equal(h.queues.replay.length, 1);
  }
});

test("a failed or edited replay preserves the previously verified historical canvas", async () => {
  const h = await harness();
  h.controller.editReplayTick("112");
  const first = h.controller.seekReplay();
  h.queues.replay[0].resolve(replayCanvas());
  const prior = await first;
  h.controller.editReplayTick("113");
  assert.equal(h.controller.canvasPreview, prior);
  const refused = h.controller.seekReplay();
  h.queues.replay[1].reject(new Error("captured interval exceeds the server replay bound"));
  assert.equal(await refused, null);
  assert.equal(h.controller.canvasPreview, prior);
  assert.match(h.controller.notice, /server replay bound/);
  h.controller.editReplayTick("114");
  const invalid = h.controller.seekReplay();
  h.queues.replay[2].resolve(replayCanvas(checkpoint(), "115"));
  assert.equal(await invalid, null);
  assert.equal(h.controller.canvasPreview, prior);
  h.controller.page = page([checkpoint(), checkpoint({ checkpoint_id: "second-source", tick: 110 })]);
  h.controller.select("second-source");
  assert.equal(h.controller.canvasPreview, null);
  assert.equal(h.controller.replayTickText, "110");
});

test("checkpoint selection requires exact metadata and complete bounded page cursors", () => {
  assert.equal(validateLabCheckpoint(checkpoint()).tick, 100);
  assert.equal(validateLabCheckpointPage(page()).items.length, 1);
  for (const changes of [{ tick: Number.MAX_SAFE_INTEGER + 1 }, { byte_size: Number.MAX_SAFE_INTEGER + 1 },
    { byte_size: 0 }, { checkpoint_id: undefined }, { checkpoint_id: ["saved-cut"] },
    { world_digest_v1: [DIGEST] }, { checksum_blake3: [DIGEST] }, { checksum_sha256: [DIGEST] },
    { world_digest_v1: DIGEST }, { world_digest_v1: "a".repeat(15) }, { world_digest_v1: "a".repeat(17) },
    { world_digest_v1: "A".repeat(16) }, { world_digest_v1: "g".repeat(16) },
    { checksum_blake3: WORLD_DIGEST }, { checksum_sha256: WORLD_DIGEST },
    { agent_count: 1025 }, { world_digest_v1: "not-a-digest" }, { checkpoint_id: "../wrong" },
    { schema: "render-frame" }, { schema: "scriptbots.world-checkpoint.v1.4" },
    { schema: "scriptbots.world-checkpoint." }, { schema: "scriptbots.host-checkpoint.v2" },
    { schema: ["scriptbots.host-checkpoint.v1"] }, { checksum_sha256: "C".repeat(64) }]) {
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

test("archive study treatments preserve the native wire format and bounded population request", () => {
  assert.deepEqual(archiveStudyTreatment(2, 42), { type: "Core", params: {
    kind: "spawn_from_archive", selector: { TopKByQuality: 2 },
    placement: { placement: "seeded", region: { shape: "all" }, seed: 42 },
    headroom: 2, herbivore_tendency: null,
  } });
  assert.equal(archiveStudyTreatment(1000, 0xffff_ffff).params.headroom, 1000);
  for (const change of [
    (p) => { p.selector.TopKByQuality = 0; },
    (p) => { p.selector.TopKByQuality = 1001; },
    (p) => { p.selector.TopKByQuality = 1.5; },
    (p) => { p.selector = "All"; },
    (p) => { p.selector.All = true; },
    (p) => { p.placement = { kind: "seeded", region: { shape: "all" }, seed: 42 }; },
    (p) => { p.placement.region.shape = "disc"; },
    (p) => { p.placement.seed = 0x1_0000_0000; },
    (p) => { p.placement.seed = -1; },
    (p) => { p.placement.seed = "42"; },
    (p) => { p.headroom = 0; },
    (p) => { p.headroom = 1001; },
    (p) => { p.herbivore_tendency = Infinity; },
    (p) => { p.herbivore_tendency = -0.1; },
    (p) => { p.ignored = true; },
  ]) {
    const value = archiveStudyTreatment(1, 42);
    change(value.params);
    assert.throws(() => validateLabIntervention(value), /Archive studies/);
  }
  assert.throws(() => archiveStudyTreatment(2, NaN), /Archive studies/);
});

test("archive studies require a complete source and retain the exact request across uncertain admission", async () => {
  const h = await harness();
  const treatment = archiveStudyTreatment(1, 42);
  assert.equal(await h.controller.launch({ branchName: "archive", ticks: 3, intervention: treatment }), null);
  assert.equal(h.queues.launch.length, 0);
  assert.equal(h.identities(), 0, "raw checkpoint refusal happens before a new retry identity");
  assert.match(h.controller.notice, /complete checkpoint.*behavioral archive/);

  const source = checkpoint({ checkpoint_id: "archive-owner", schema: "scriptbots.host-checkpoint.v1" });
  h.controller.services.list = async () => page([checkpoint(), source]);
  await h.controller.refresh(true);
  h.controller.select(source.checkpoint_id);
  const pending = h.controller.launch({ branchName: "archive", ticks: 3, intervention: treatment });
  const request = h.queues.launch[0].request;
  assert.equal(request.checkpoint_id, source.checkpoint_id);
  assert.deepEqual(request.intervention, archiveStudyTreatment(1, 42));
  treatment.params.selector.TopKByQuality = 999;
  treatment.params.placement.seed = 7;
  assert.deepEqual(request.intervention, archiveStudyTreatment(1, 42));
  assert.equal(Object.isFrozen(request.intervention.params.placement), true);
  h.queues.launch[0].reject(new Error("connection lost before admission was observed"));
  await pending;
  assert.equal(h.controller.branchState, "unconfirmed");
  h.controller.select(checkpoint().checkpoint_id);
  const retry = h.controller.retryLaunch();
  assert.equal(h.queues.launch[1].request, request, "retry uses the original complete source and treatment");
  h.queues.launch[1].resolve(status(request, source));
  await retry;
  assert.equal(h.controller.branch.status.status, "running");
  assert.equal(h.controller.branch.checkpoint.checkpoint_id, source.checkpoint_id);
});

test("treatment timing refuses invalid delays before request identity or admission", async () => {
  const h = await harness();
  const intervention = { type: "Drought", params: { region: null, duration_ticks: 3, growth_scale: 0.1 } };
  for (const interventionDelayTicks of [-1, 0.5, "1", null, NaN, Infinity, 25, 26, Number.MAX_SAFE_INTEGER + 1]) {
    assert.equal(await h.controller.launch({ branchName: "delayed", ticks: 25, intervention, interventionDelayTicks }), null);
    assert.match(h.controller.notice, /Treatment delay/);
    assert.equal(h.controller.branch, null);
  }
  assert.equal(await h.controller.launch({ branchName: "unchanged", ticks: 25, interventionDelayTicks: 1 }), null);
  assert.match(h.controller.notice, /Choose a treatment/);
  assert.equal(h.identities(), 0);
  assert.equal(h.queues.launch.length, 0);
});

test("zero and last-transition treatment delays preserve their exact bounded requests", async () => {
  const intervention = { type: "ClosedWorld", params: { closed: true } };
  for (const [ticks, interventionDelayTicks] of [[1, 0], [25, 0], [25, 24], [10000, 9999]]) {
    const h = await harness();
    const pending = h.controller.launch({ branchName: "timed", ticks, intervention, interventionDelayTicks });
    const request = h.queues.launch[0].request;
    assert.equal(request.ticks, ticks);
    assert.equal(request.intervention_delay_ticks, interventionDelayTicks === 0 ? undefined : interventionDelayTicks);
    assert.equal(Object.hasOwn(request, "intervention_delay_ticks"), interventionDelayTicks > 0,
      "immediate requests retain their original canonical representation");
    h.queues.launch[0].resolve(status(request));
    await pending;
    const { panel, nodes } = labPanel();
    renderCheckpointLab(panel, h.controller);
    assert.ok(nodes.get("#lab-branch-detail").textContent.includes(
      `Treatment requested at tick ${100 + interventionDelayTicks}, after ${interventionDelayTicks} future ticks.`));
    assert.equal(h.controller.branch.status.target_tick, 100 + ticks);
  }
});

test("a delayed study retry retains its original timing and source after form and selection changes", async () => {
  const h = await harness();
  const draft = { branchName: "delayed_drought", ticks: 25, interventionDelayTicks: 10,
    intervention: { type: "Drought", params: { region: null, duration_ticks: 3, growth_scale: 0.1 } } };
  const pending = h.controller.launch(draft);
  const request = h.queues.launch[0].request;
  draft.interventionDelayTicks = 20;
  draft.intervention.params.duration_ticks = 8;
  assert.equal(request.intervention_delay_ticks, 10);
  assert.equal(request.intervention.params.duration_ticks, 3);
  assert.equal(Object.isFrozen(request), true);
  h.queues.launch[0].reject(new Error("admission reply lost"));
  await pending;
  assert.equal(h.controller.branchState, "unconfirmed");
  assert.equal(await h.controller.launch(draft), null);
  h.controller.page = page([checkpoint({ checkpoint_id: "later-source", tick: 200 })]);
  h.controller.select("later-source");
  const retry = h.controller.retryLaunch();
  assert.equal(h.queues.launch[1].request, request);
  h.queues.launch[1].resolve(status(request));
  await retry;
  assert.equal(h.controller.branch.checkpoint.tick, 100);
  assert.equal(h.identities(), 1);
  const { panel, nodes } = labPanel();
  renderCheckpointLab(panel, h.controller);
  assert.match(nodes.get("#lab-branch-detail").textContent, /Treatment requested at tick 110, after 10 future ticks/);
  assert.match(nodes.get("#lab-branch-detail").textContent, /target 125/);
});

test("branch observations bind the saved digest and horizon and cannot regress a terminal state", () => {
  const request = { checkpoint_id: checkpoint().checkpoint_id, branch_name: "future", ticks: 120 };
  const running = status(request);
  assert.equal(validateLabBranchStatus(running, request, checkpoint()), running);
  const complete = status(request, checkpoint(), { status: "completed", artifact_id: artifact().artifact_id });
  validateLabBranchStatus(complete, request, checkpoint(), running);
  for (const changes of [{ checkpoint_id: "ckpt-other" }, { world_digest_v1: "d".repeat(16) },
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

test("owner health gates live checkpoint writes while saved-cut review and local transfer remain usable", async () => {
  const h = await harness(), selected = h.controller.selected;
  const saved = h.controller.save(), original = h.queues.save[0].request;
  h.controller.setOwnerWritable(false);
  assert.equal(h.queues.save[0].signal.aborted, false, "retain the outcome of an already submitted save");
  h.queues.save[0].reject(new Error("connection lost after possible admission"));
  await saved;
  assert.equal(h.controller.saveState, "unconfirmed");
  assert.equal(h.controller.canSave, false);
  assert.equal(await h.controller.save(), null);
  assert.equal(await h.controller.retrySave(), null);
  assert.equal(h.queues.save.length, 1);
  assert.equal(h.controller.selected, selected);

  const { panel, nodes } = labPanel();
  renderCheckpointLab(panel, h.controller);
  assert.equal(nodes.get("#lab-checkpoint-save").disabled, true);
  assert.equal(nodes.get("#lab-checkpoint-retry").disabled, true);
  assert.equal(nodes.get("#lab-checkpoint-review").disabled, false);
  assert.equal(nodes.get("#lab-checkpoint-local").disabled, false);
  const reviewed = h.controller.reviewSelected();
  h.queues.review[0].resolve(savedCanvas());
  await reviewed;
  const transfer = h.controller.prepareLocalImport(2048);
  h.queues.download[0].resolve(scientificDownload(selected));
  const imported = await transfer;
  assert.equal(imported.checkpoint, selected);
  assert.equal(imported.bytes.byteLength, selected.byte_size);
  assert.equal(h.identities(), 1, "read-only work cannot allocate another save identity");
  h.controller.returnToLive();
  assert.equal(h.controller.canLaunch, true, "isolated saved-cut studies have their own lifecycle");

  h.controller.setOwnerWritable(true);
  const retry = h.controller.retrySave();
  assert.equal(h.queues.save[1].request, original);
  h.queues.save[1].resolve(checkpoint({ checkpoint_id: "ckpt-recovered" }));
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

function scientificDownload(source = checkpoint(), blob = new Blob([new Uint8Array(source.byte_size)])) {
  return { blob, filename: "world.sbworld", metadata: artifact({ artifact_id: source.checkpoint_id,
    filename: "world.sbworld", content_type: "application/octet-stream", relative_path: "checkpoints/world.sbworld",
    byte_size: source.byte_size, checksum_blake3: source.checksum_blake3, checksum_sha256: source.checksum_sha256 }) };
}

test("complete owner checkpoints remain usable through the saved-world workflow", async () => {
  const source = checkpoint({ checkpoint_id: "owner-saved-t100", schema: "scriptbots.host-checkpoint.v1" });
  const h = await harness();
  h.controller.services.list = async () => page([checkpoint(), source]);
  await h.controller.refresh(true);
  h.controller.select(source.checkpoint_id);
  assert.equal(h.controller.selected.schema, source.schema);
  assert.equal(h.controller.canReview, true);
  assert.equal(h.controller.canLaunch, true);
  assert.equal(h.controller.canImportLocally, false);
  const { panel, nodes } = labPanel();
  renderCheckpointLab(panel, h.controller);
  assert.equal(nodes.get("#lab-checkpoint-local").disabled, true);
  assert.equal(nodes.get("#lab-checkpoint-review").disabled, false);
  assert.equal(nodes.get("#lab-checkpoint-download").disabled, false);
  assert.match(nodes.get("#lab-checkpoint-detail").textContent, /Complete owner checkpoint/);
  assert.equal(await h.controller.prepareLocalImport(2048), null);
  assert.equal(h.queues.download.length, 0, "unsupported local restore refuses before downloading");
  assert.match(h.controller.notice, /standalone world checkpoints/);

  const review = h.controller.reviewSelected();
  h.queues.review[0].resolve(savedCanvas(source));
  assert.equal((await review).checkpoint.schema, source.schema);
  h.controller.editReplayTick("112");
  const replay = h.controller.seekReplay();
  h.queues.replay[0].resolve(replayCanvas(source));
  assert.equal((await replay).targetTick, "112");
  const download = h.controller.downloadCheckpoint();
  h.queues.download[0].resolve(scientificDownload(source));
  assert.equal((await download).blob.size, source.byte_size);
  h.controller.returnToLive();
  const branch = h.controller.launch({ branchName: "complete_source", ticks: 120 });
  h.queues.launch[0].resolve(status(h.queues.launch[0].request, source));
  await branch;
  assert.equal(h.controller.branch.checkpoint.schema, source.schema);
  assert.equal(h.controller.branch.status.status, "running");
  assert.equal(h.queues.save.length, 0, "reviewing and studying do not replace the source owner");
});

test("a complete owner save receipt is accepted and retains its exact schema", async () => {
  const source = checkpoint({ checkpoint_id: "owner-save-receipt", schema: "scriptbots.host-checkpoint.v1" });
  const h = await harness();
  const pending = h.controller.save();
  h.queues.save[0].resolve(source);
  await pending;
  assert.equal(h.controller.saveState, "saved");
  assert.deepEqual(h.controller.saveReceipt, source);
  assert.equal(h.controller.selected.checkpoint_id, source.checkpoint_id);
  assert.equal(h.controller.canImportLocally, false);
  assert.throws(() => validateCheckpointCanvas(savedCanvas(checkpoint({ checkpoint_id: source.checkpoint_id })),
    { sessionId: SESSION, checkpoint: source }), /does not match/);
});

test("local transfer reads the selected scientific bytes once without changing either owner", async () => {
  const h = await harness();
  const source = h.controller.selected;
  const preview = h.controller.reviewSelected();
  h.queues.review[0].resolve(savedCanvas());
  await preview;
  const retainedPreview = h.controller.canvasPreview;
  const pending = h.controller.prepareLocalImport(2048);
  assert.equal(h.controller.busy, true);
  assert.equal(await h.controller.prepareLocalImport(2048), null);
  assert.deepEqual(h.queues.download[0].request, { sessionId: SESSION, artifactId: source.checkpoint_id,
    expected: source, maxBytes: 2048 });
  const body = new Uint8Array(source.byte_size);
  body[17] = 91;
  h.queues.download[0].resolve(scientificDownload(source, new Blob([body])));
  const transfer = await pending;
  assert.deepEqual(transfer.bytes, body);
  assert.equal(transfer.checkpoint, source);
  assert.equal(transfer.sourceSessionId, SESSION);
  assert.equal(transfer.sourceToken, SOURCE);
  assert.equal(h.controller.canvasPreview, retainedPreview);
  assert.equal(h.controller.selected, source);
  assert.equal(h.controller.busy, false);
  for (const kind of ["save", "launch", "cancel"]) assert.equal(h.queues[kind].length, 0);
  assert.equal(h.identities(), 0, "transferring a file cannot mint an owner mutation identity");
});

test("local transfer refuses size before fetching and preserves download transport limits", async () => {
  const h = await harness();
  for (const maxBytes of [0, -1, 0.5, Infinity, NaN, "1024"]) {
    assert.throws(() => h.controller.prepareLocalImport(maxBytes), /valid scientific checkpoint limit/);
  }
  assert.equal(await h.controller.prepareLocalImport(1023), null);
  assert.match(h.controller.notice, /remains available for server studies/);
  assert.equal(h.queues.download.length, 0);
  const pending = h.controller.prepareLocalImport(128 * 1024 * 1024);
  assert.equal(h.queues.download[0].request.maxBytes, MAX_LAB_ARTIFACT_BYTES);
  h.queues.download[0].resolve(scientificDownload());
  assert.equal((await pending).bytes.length, 1024);
});

test("local transfer rejects changed artifact metadata and corrupt read lengths", async () => {
  for (const change of [
    (result) => { result.metadata.checksum_sha256 = "d".repeat(64); },
    (result) => { result.metadata.checksum_blake3 = "d".repeat(64); },
    (result) => { result.metadata.artifact_id = "different-world"; },
    (result) => { result.blob = new Blob([new Uint8Array(1025)]); },
    (result) => { result.blob.arrayBuffer = async () => new ArrayBuffer(1023); },
  ]) {
    const h = await harness();
    const source = h.controller.selected;
    const pending = h.controller.prepareLocalImport(2048);
    const response = scientificDownload();
    change(response);
    h.queues.download[0].resolve(response);
    assert.equal(await pending, null);
    assert.equal(h.controller.selected, source);
    assert.match(h.controller.notice, /Local checkpoint transfer unavailable/);
  }
});

test("source changes and hidden panels cancel even a late scientific Blob read", async () => {
  for (const invalidate of [(controller) => controller.setEnabled(false),
    (controller) => controller.observe(frame("7"), SOURCE),
    (controller) => controller.observe(frame(), "wss://another.example/api/ws/stream"),
    (controller) => controller.reset()]) {
    const h = await harness();
    const pending = h.controller.prepareLocalImport(2048);
    const result = scientificDownload();
    let resolveRead;
    result.blob.arrayBuffer = () => new Promise((resolve) => { resolveRead = resolve; });
    h.queues.download[0].resolve(result);
    await flush();
    assert.equal(typeof resolveRead, "function");
    assert.equal(h.controller.busy, true);
    invalidate(h.controller);
    assert.equal(h.queues.download[0].signal.aborted, true);
    assert.equal(await h.controller.prepareLocalImport(2048), null, "a cancelled uncompleted read still holds its slot");
    resolveRead(new ArrayBuffer(1024));
    assert.equal(await pending, null);
    assert.equal(h.controller.busy, false);
    assert.equal(h.queues.download.length, 1);
  }
});

test("an existing catalog row cannot replace a selected cut with different scientific bytes", async () => {
  const h = await harness();
  const original = h.controller.selected;
  h.controller.services.list = async () => page([checkpoint({ world_digest_v1: "d".repeat(16) })]);
  assert.equal(await h.controller.refresh(true), null);
  assert.equal(h.controller.selected, original);
  assert.match(h.controller.notice, /changed its scientific identity/);
});

test("retained study pages identify each branch independently of its shared checkpoint", () => {
  const first = study(), second = study({ branch_id: "branch-two" });
  assert.equal(validateLabBranchPage(historyPage([first, second])).items.length, 2);
  for (const changes of [
    { branch_id: undefined }, { checkpoint_id: ["checkpoint"] }, { branch_name: "unsafe name" },
    { source_tick: -1 }, { target_tick: 100 }, { target_tick: 10101 },
    { target_tick: Number.MAX_SAFE_INTEGER + 1 }, { world_digest_v1: [DIGEST] },
    { status: "pending" }, { status: "completed" }, { artifact_id: "premature" },
  ]) assert.throws(() => validateLabBranchObservation(study(changes)));
  assert.throws(() => validateLabBranchPage(historyPage([first, first])));
  const rows = Array.from({ length: 12 }, (_, index) => study({ branch_id: `branch-${index}` }));
  assert.equal(validateLabBranchPage(historyPage(rows, { total: 13, has_more: true, next_cursor: "12" })).next_cursor, "12");
  assert.throws(() => validateLabBranchPage(historyPage(rows, { total: 13, has_more: true, next_cursor: "13" })));
  assert.throws(() => validateLabBranchObservation({ ...first, checkpoint_id: "different" }, first));
  assert.throws(() => validateLabBranchObservation({ ...first, target_tick: 225 }, first));
  assert.throws(() => validateLabBranchObservation({ ...first, world_digest_v1: "d".repeat(16) }, first));
  for (const world_digest_v1 of [DIGEST, "a".repeat(15), "a".repeat(17), "A".repeat(16)]) {
    assert.throws(() => validateLabBranchObservation({ ...first, world_digest_v1 }));
  }
});

test("selecting completed history reads exact retained status and downloads without reconstructing or submitting a study", async () => {
  const receipt = study({ status: "completed", artifact_id: artifact().artifact_id });
  const parsed = Object.freeze({ studyId: "retained-study", sourceTick: 100 });
  let parsedContext;
  const h = await harness({ listStudies: async () => historyPage([receipt]),
    parseReport: async (_blob, context) => { parsedContext = context; return parsed; } });
  await h.controller.refreshHistory(true);
  const selected = h.controller.selectStudy(receipt.branch_id);
  assert.equal(h.controller.branch, null);
  assert.equal(h.queues.status[0].request.branchId, receipt.branch_id);
  h.queues.status[0].resolve(receipt);
  await selected;
  assert.equal(h.controller.branch.origin, "history");
  assert.equal(h.controller.branch.request, null);
  assert.equal(h.controller.branch.checkpoint, null);
  assert.ok(Object.isFrozen(h.controller.branch.status));
  assert.equal(h.queues.launch.length, 0);
  assert.equal(h.queues.save.length, 0);
  assert.equal(h.identities(), 0);
  assert.equal(await h.controller.retryLaunch(), null);
  assert.equal(await h.controller.submitBranch(), null);
  const { panel, nodes } = labPanel();
  renderCheckpointLab(panel, h.controller);
  const detail = nodes.get("#lab-branch-detail").textContent;
  assert.match(detail, /original treatment is unknown/);
  assert.ok(detail.includes(receipt.checkpoint_id) && detail.includes(receipt.world_digest_v1));
  assert.ok(!detail.includes("One unchanged scientific future."));
  assert.equal(nodes.get("#lab-report-download").disabled, false);
  assert.equal(nodes.get("#lab-report-preview").disabled, false);
  const download = h.controller.downloadReport();
  h.queues.download[0].resolve({ blob: new Blob(["{}"]), filename: "study_report.json", metadata: artifact() });
  assert.equal((await download).blob.size, 2);
  const preview = h.controller.previewReport();
  h.queues.download[1].resolve({ blob: new Blob(["{}"]), filename: "study_report.json", metadata: artifact() });
  assert.equal(await preview, parsed);
  assert.equal(parsedContext.checkpoint, null);
  assert.equal(parsedContext.request, null);
  assert.equal(parsedContext.status.branch_id, receipt.branch_id);
  assert.equal(parsedContext.artifact.artifact_id, receipt.artifact_id);
  assert.equal(h.queues.launch.length, 0);
});

test("history pagination preserves the observed selection and uses bounded exact cursors", async () => {
  const rows = Array.from({ length: 12 }, (_, index) => study({ branch_id: `branch-${index}` }));
  const reads = [];
  const h = await harness({ listStudies: async (request) => {
    reads.push(request);
    return request.cursor === "12" ? historyPage([study({ branch_id: "branch-last" })], { total: 13, offset: 12 })
      : historyPage(rows, { total: 13, has_more: true, next_cursor: "12" });
  } });
  await h.controller.refreshHistory(true);
  const selecting = h.controller.selectStudy("branch-0");
  h.queues.status[0].resolve(rows[0]);
  await selecting;
  const selected = h.controller.branch;
  await h.controller.nextHistoryPage();
  assert.equal(h.controller.historyCursor, "12");
  assert.equal(h.controller.historyPage.items[0].branch_id, "branch-last");
  assert.equal(h.controller.branch, selected);
  assert.equal(await h.controller.nextHistoryPage(), null);
  await h.controller.firstHistoryPage();
  assert.equal(h.controller.historyCursor, null);
  assert.deepEqual(reads.map((request) => request.cursor), [null, "12", null]);
  assert.ok(reads.every((request) => request.limit === 12 && request.sessionId === SESSION));
});

test("history selection refuses changed scientific evidence and never regresses a newer terminal receipt", async () => {
  const running = study();
  const complete = { ...running, status: "completed", artifact_id: artifact().artifact_id };
  const h = await harness({ listStudies: async () => historyPage([running]) });
  await h.controller.refreshHistory(true);
  const first = h.controller.selectStudy(running.branch_id);
  h.queues.status[0].resolve(complete);
  await first;
  const observed = h.controller.branch;
  const changed = h.controller.selectStudy(running.branch_id);
  h.queues.status[1].resolve({ ...complete, world_digest_v1: "d".repeat(16) });
  assert.equal(await changed, null);
  assert.equal(h.controller.branch, observed);
  assert.match(h.controller.notice, /does not match/);
  const regressed = h.controller.selectStudy(running.branch_id);
  h.queues.status[2].resolve(running);
  assert.equal(await regressed, null);
  assert.equal(h.controller.branch.status.status, "completed");
  const oldPage = h.controller.historyPage;
  h.controller.services.listStudies = async () => historyPage([{ ...running, source_tick: 101 }]);
  assert.equal(await h.controller.refreshHistory(true), null);
  assert.equal(h.controller.historyPage, oldPage);
  assert.equal(h.queues.launch.length, 0);
});

test("history cannot discard uncertain admission and late selection after disconnect does not replace an observed study", async () => {
  const receipt = study({ branch_id: "branch-history" });
  const h = await harness({ listStudies: async () => historyPage([receipt]) });
  await h.controller.refreshHistory(true);
  const admission = h.controller.launch({ branchName: "new_request", ticks: 25 });
  const original = h.queues.launch[0].request;
  h.queues.launch[0].reject(new Error("reply lost"));
  await admission;
  assert.equal(h.controller.canSelectStudy, false);
  assert.equal(await h.controller.selectStudy(receipt.branch_id), null);
  assert.equal(h.queues.status.length, 0);
  assert.equal(h.controller.branch.request, original);
  const retry = h.controller.retryLaunch();
  h.queues.launch[1].resolve(status(original));
  await retry;
  const current = h.controller.branch;
  const selecting = h.controller.selectStudy(receipt.branch_id);
  h.controller.setEnabled(false);
  assert.equal(h.queues.status[0].signal.aborted, true);
  h.queues.status[0].resolve(receipt);
  assert.equal(await selecting, null);
  assert.equal(h.controller.branch, current);
  h.controller.setEnabled(true);
  const reselecting = h.controller.selectStudy(receipt.branch_id);
  h.queues.status[1].resolve(receipt);
  await reselecting;
  assert.equal(h.controller.branch.origin, "history");
  assert.equal(h.controller.branch.status.branch_id, receipt.branch_id);
  assert.equal(h.queues.launch.length, 2, "only the original immutable admission retry was submitted");
  assert.equal(h.identities(), 1);
});

test("a new host session discovers interrupted retained studies afresh without rerunning their science", async () => {
  let receipt = study();
  const sessions = [];
  const h = await harness({ listStudies: async (request) => { sessions.push(request.sessionId); return historyPage([receipt]); } });
  await h.controller.refreshHistory(true);
  const selecting = h.controller.selectStudy(receipt.branch_id);
  h.queues.status[0].resolve(receipt);
  await selecting;
  h.controller.observe(frame("7"), SOURCE);
  await flush();
  assert.equal(h.controller.branch, null);
  assert.equal(h.controller.historyPage, null);
  assert.equal(h.controller.selected, null);
  receipt = { ...receipt, status: "stopped", error_reason: "interrupted by restart" };
  await h.controller.refreshHistory(true);
  const recovered = h.controller.selectStudy(receipt.branch_id);
  assert.equal(h.queues.status[1].request.sessionId, "7");
  h.queues.status[1].resolve(receipt);
  await recovered;
  assert.deepEqual(sessions, [SESSION, "7"]);
  assert.equal(h.controller.branch.status.status, "stopped");
  assert.equal(h.controller.branch.request, null);
  assert.equal(h.controller.branch.checkpoint, null);
  assert.match(h.controller.notice, /not rerun automatically/);
  h.advance();
  h.controller.poll();
  assert.equal(h.queues.launch.length, 0);
  assert.equal(h.queues.cancel.length, 0);
  assert.equal(h.identities(), 0);
});

test("report preview binds verified bytes to the exact source and clears when a new study is requested", async () => {
  let context;
  const model = Object.freeze({ studyId: "browser-study", sourceTick: 100 });
  const h = await harness({ parseReport: async (_blob, value) => { context = value; return model; } });
  const branch = await launch(h);
  const finished = h.controller.refreshBranch(true);
  h.queues.status[0].resolve(status(branch.request, checkpoint(), { status: "completed", artifact_id: artifact().artifact_id }));
  await finished;
  const preview = h.controller.previewReport();
  assert.equal(h.queues.download[0].request.maxBytes, MAX_STUDY_REPORT_BYTES);
  h.queues.download[0].resolve({ blob: new Blob(["{}"]), filename: "study_report.json", metadata: artifact() });
  assert.equal(await preview, model);
  assert.equal(h.controller.reportModel, model);
  assert.equal(context.status, branch.status);
  assert.equal(context.checkpoint, branch.checkpoint);
  assert.equal(context.artifact.checksum_sha256, artifact().checksum_sha256);
  const next = h.controller.launch({ branchName: "next", ticks: 10 });
  assert.equal(h.controller.reportModel, null);
  h.queues.launch[1].resolve(status(h.queues.launch[1].request, checkpoint(), { branch_id: "branch-next" }));
  await next;
});

test("report preview retains the immutable submitted request while its download is pending", async () => {
  let context;
  const model = Object.freeze({ studyId: "browser-study", sourceTick: 100 });
  const h = await harness({ parseReport: async (_blob, value) => { context = value; return model; } });
  const draft = { branchName: "delayed", ticks: 120, interventionDelayTicks: 30,
    intervention: { type: "Drought", params: { region: null, duration_ticks: 10, growth_scale: 0.1 } } };
  const launching = h.controller.launch(draft);
  const submitted = h.queues.launch[0].request;
  h.queues.launch[0].resolve(status(submitted, checkpoint(), { status: "completed", artifact_id: artifact().artifact_id }));
  await launching;
  const preview = h.controller.previewReport();
  draft.interventionDelayTicks = 60;
  draft.intervention.params.growth_scale = 0.8;
  h.controller.branch.request = { ...submitted, intervention_delay_ticks: 60, intervention: draft.intervention };
  h.queues.download[0].resolve({ blob: new Blob(["{}"]), filename: "study_report.json", metadata: artifact() });
  assert.equal(await preview, model);
  assert.deepEqual(context.request, submitted);
  assert.equal(context.request.intervention_delay_ticks, 30);
  assert.equal(context.request.intervention.params.growth_scale, 0.1);
  assert.ok(Object.isFrozen(context.request));
  assert.ok(Object.isFrozen(context.request.intervention.params));
  assert.throws(() => { context.request.intervention_delay_ticks = 60; }, TypeError);
});

test("report decoding that finishes after disconnection cannot publish a stale preview", async () => {
  let finishParsing;
  const h = await harness({ parseReport: () => new Promise((resolve) => { finishParsing = resolve; }) });
  const branch = await launch(h);
  const finished = h.controller.refreshBranch(true);
  h.queues.status[0].resolve(status(branch.request, checkpoint(), { status: "completed", artifact_id: artifact().artifact_id }));
  await finished;
  const preview = h.controller.previewReport();
  h.queues.download[0].resolve({ blob: new Blob(["{}"]), filename: "study_report.json", metadata: artifact() });
  await flush();
  assert.equal(typeof finishParsing, "function");
  h.controller.setEnabled(false);
  finishParsing(Object.freeze({ studyId: "stale" }));
  assert.equal(await preview, null);
  assert.equal(h.controller.reportModel, null);
});

test("report preview rejects unrelated bytes, oversized reports and parser failures without displaying a model", async () => {
  let parseCalls = 0;
  const h = await harness({ parseReport: async () => { parseCalls += 1; throw new Error("report schema mismatch"); } });
  const branch = await launch(h);
  const finished = h.controller.refreshBranch(true);
  h.queues.status[0].resolve(status(branch.request, checkpoint(), { status: "completed", artifact_id: artifact().artifact_id }));
  await finished;
  const unrelated = h.controller.previewReport();
  h.queues.download[0].resolve({ blob: new Blob(["{}"]), filename: "study_report.json",
    metadata: artifact({ relative_path: "branches/unrelated/study_report.json" }) });
  assert.equal(await unrelated, null);
  const large = h.controller.previewReport();
  const blob = new Blob([new Uint8Array(MAX_STUDY_REPORT_BYTES + 1)]);
  h.queues.download[1].resolve({ blob, filename: "study_report.json", metadata: artifact({ byte_size: blob.size }) });
  assert.equal(await large, null);
  assert.equal(parseCalls, 0);
  const malformed = h.controller.previewReport();
  h.queues.download[2].resolve({ blob: new Blob(["{}"]), filename: "study_report.json", metadata: artifact() });
  assert.equal(await malformed, null);
  assert.equal(parseCalls, 1);
  assert.match(h.controller.notice, /report schema mismatch/);
  assert.equal(h.controller.reportModel, null);
});

// Controller contract fixtures; scientific CSV validation belongs to study-trajectories tests.
const trajectoryBytes = "controller trajectory fixture\n";
function trajectoryArtifact(armId, changes = {}) {
  return artifact({ artifact_id: "summary-" + armId, filename: "summary.csv", content_type: "text/csv",
    relative_path: "branches/branch-one/study/bundle_" + armId + "/exports/summary.csv",
    byte_size: new Blob([trajectoryBytes]).size,
    checksum_sha256: createHash("sha256").update(trajectoryBytes).digest("hex"), ...changes });
}
function trajectoryDownload(metadata) {
  return { blob: new Blob([trajectoryBytes]), filename: metadata.filename, metadata };
}
function trajectoryFillers(count = CHECKPOINT_LAB_PAGE_SIZE) {
  return Array.from({ length: count }, (_, index) => artifact({ artifact_id: "retained-file-" + index,
    filename: "evidence-" + index + ".json", relative_path: "branches/branch-one/study/evidence-" + index + ".json" }));
}
async function trajectoryHarness({ hasTreatment = true, parseTrajectories = async () => Object.freeze({ loaded: true }) } = {}) {
  const report = Object.freeze({ studyId: "branch-one", checkpointId: checkpoint().checkpoint_id,
    sourceTick: 100, targetTick: 220, horizonTicks: 120, worldDigest: WORLD_DIGEST, hasTreatment,
    arms: Object.freeze([Object.freeze({ id: "control", label: "Control" }),
      ...(hasTreatment ? [Object.freeze({ id: "treatment", label: "Treatment" })] : [])]) });
  const h = await harness({ parseReport: async () => report, parseTrajectories });
  const launching = h.controller.launch({ branchName: "future", ticks: 120,
    intervention: hasTreatment ? { type: "Drought", params: { region: null, duration_ticks: 10, growth_scale: 0.1 } } : null });
  h.queues.launch[0].resolve(status(h.queues.launch[0].request, checkpoint(),
    { status: "completed", artifact_id: artifact().artifact_id }));
  await launching;
  const preview = h.controller.previewReport();
  h.queues.download[0].resolve({ blob: new Blob(["{}"]), filename: "study_report.json", metadata: artifact() });
  assert.equal(await preview, report);
  return { ...h, report };
}
function serveTrajectories(h, rows, transform = (value) => value) {
  const downloads = [];
  h.controller.services.artifacts = async () => artifactPage(rows);
  h.controller.services.download = async (request, signal) => {
    downloads.push({ request, signal });
    return transform(trajectoryDownload(request.expected));
  };
  return downloads;
}

test("trajectory preview requires a loaded completed report and disables its action while busy", async () => {
  const h = await harness();
  const { panel, nodes } = labPanel();
  assert.equal(await h.controller.previewTrajectories(), null);
  await launch(h);
  assert.equal(await h.controller.previewTrajectories(), null);
  const finished = h.controller.refreshBranch(true);
  h.queues.status[0].resolve(status(h.controller.branch.request, checkpoint(),
    { status: "completed", artifact_id: artifact().artifact_id }));
  await finished;
  assert.equal(await h.controller.previewTrajectories(), null);
  renderCheckpointLab(panel, h.controller);
  assert.equal(nodes.get("#lab-trajectories-preview").disabled, true);
  assert.equal(h.queues.artifacts.length, 0);
  assert.equal(h.queues.download.length, 0);
  const ready = await trajectoryHarness();
  renderCheckpointLab(panel, ready.controller);
  assert.equal(nodes.get("#lab-trajectories-preview").disabled, false);
  const preview = ready.controller.previewTrajectories();
  renderCheckpointLab(panel, ready.controller);
  assert.equal(nodes.get("#lab-trajectories-preview").disabled, true);
  ready.queues.artifacts[0].reject(new Error("catalog temporarily unavailable"));
  assert.equal(await preview, null);
  renderCheckpointLab(panel, ready.controller);
  assert.equal(nodes.get("#lab-trajectories-preview").disabled, false);
});

test("trajectory discovery follows every catalog page and leaves the user's file selection intact", async () => {
  let parsedReport, parsedDownloads;
  const model = Object.freeze({ studyId: "branch-one", loaded: true });
  const h = await trajectoryHarness({ parseTrajectories: async (report, downloads) => {
    parsedReport = report; parsedDownloads = downloads; return model;
  } });
  const fillers = trajectoryFillers();
  const listing = h.controller.refreshArtifacts(true);
  h.queues.artifacts[0].resolve(artifactPage(fillers, { total: 13, has_more: true, next_cursor: "12" }));
  await listing;
  const next = h.controller.nextArtifactPage();
  h.queues.artifacts[1].resolve(artifactPage([artifact()], { total: 13, offset: 12 }));
  await next;
  h.controller.selectArtifact(artifact().artifact_id);
  const selected = h.controller.selectedArtifact, userPage = h.controller.artifactsPage;
  const control = trajectoryArtifact("control"), treatment = trajectoryArtifact("treatment");
  const downloads = [];
  h.controller.services.download = async (request, signal) => {
    downloads.push({ request, signal }); return trajectoryDownload(request.expected);
  };
  const preview = h.controller.previewTrajectories();
  assert.deepEqual(h.queues.artifacts[2].request,
    { sessionId: SESSION, branchId: "branch-one", cursor: null, limit: CHECKPOINT_LAB_PAGE_SIZE });
  h.queues.artifacts[2].resolve(artifactPage(fillers, { total: 14, has_more: true, next_cursor: "12" }));
  await flush();
  assert.equal(h.queues.artifacts[3].request.cursor, "12");
  h.queues.artifacts[3].resolve(artifactPage([treatment, control], { total: 14, offset: 12 }));
  assert.equal(await preview, model);
  assert.deepEqual(parsedReport, h.report);
  assert.ok(Object.isFrozen(parsedReport) && Object.isFrozen(parsedReport.arms));
  assert.deepEqual(parsedDownloads.map((row) => row.armId), ["control", "treatment"]);
  assert.deepEqual(downloads.map(({ request }) => request), [control, treatment].map((metadata) => ({
    sessionId: SESSION, artifactId: metadata.artifact_id, expected: metadata, maxBytes: MAX_STUDY_TRAJECTORY_BYTES,
  })));
  assert.equal(h.controller.trajectoryModel, model);
  assert.equal(h.controller.reportModel, h.report);
  assert.equal(h.controller.artifactsPage, userPage);
  assert.equal(h.controller.artifactCursor, "12");
  assert.equal(h.controller.selectedArtifact, selected);
  assert.equal(h.queues.launch.length, 1);
  assert.equal(h.queues.save.length, 0);
});

test("unchanged studies request only their control trajectory", async () => {
  let armIds;
  const h = await trajectoryHarness({ hasTreatment: false, parseTrajectories: async (_report, downloads) => {
    armIds = downloads.map((row) => row.armId); return Object.freeze({ loaded: true });
  } });
  const downloads = serveTrajectories(h, [trajectoryArtifact("control")]);
  assert.ok(await h.controller.previewTrajectories());
  assert.deepEqual(armIds, ["control"]);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].request.artifactId, "summary-control");
});

test("trajectory discovery refuses missing, duplicate, changing and oversized catalogs before downloading", async () => {
  const control = trajectoryArtifact("control"), treatment = trajectoryArtifact("treatment");
  const full = [control, ...trajectoryFillers(CHECKPOINT_LAB_PAGE_SIZE - 1)];
  const first = artifactPage(full, { total: 13, has_more: true, next_cursor: "12" });
  const cases = [
    ["missing treatment", [artifactPage([control])]],
    ["duplicate summary path", [artifactPage([control, { ...control, artifact_id: "duplicate-control" }, treatment])]],
    ["duplicate identity across pages", [first,
      artifactPage([{ ...treatment, artifact_id: control.artifact_id }], { total: 13, offset: 12 })]],
    ["changing catalog total", [first, artifactPage([treatment, artifact()], { total: 14, offset: 12 })]],
    ["catalog scan budget", [artifactPage(full, { total: MAX_LAB_TRAJECTORY_CATALOG_ARTIFACTS + 1,
      has_more: true, next_cursor: "12" })]],
  ];
  for (const [label, pages] of cases) {
    let parses = 0, reads = 0;
    const h = await trajectoryHarness({ parseTrajectories: async () => { parses += 1; return {}; } });
    const downloads = serveTrajectories(h, []);
    h.controller.services.artifacts = async () => {
      assert.ok(reads < pages.length, label + ": unexpected catalog request");
      return pages[reads++];
    };
    assert.equal(await h.controller.previewTrajectories(), null, label);
    assert.equal(parses, 0, label);
    assert.equal(downloads.length, 0, label);
    assert.equal(h.controller.trajectoryModel, null, label);
    assert.equal(h.controller.reportModel, h.report, label);
  }
});

test("trajectory preview refuses oversized summaries and substituted download metadata", async () => {
  const oversized = await trajectoryHarness({ hasTreatment: false });
  const unused = serveTrajectories(oversized,
    [trajectoryArtifact("control", { byte_size: MAX_STUDY_TRAJECTORY_BYTES + 1 })]);
  assert.equal(await oversized.controller.previewTrajectories(), null);
  assert.equal(unused.length, 0, "the byte budget is checked before requesting summary bytes");
  for (const change of [
    { artifact_id: "another-summary" },
    { relative_path: "branches/branch-one/study/bundle_treatment/exports/summary.csv" },
    { filename: "other.csv", relative_path: "branches/branch-one/study/bundle_control/exports/other.csv" },
    { content_type: "application/json" },
    { checksum_sha256: "e".repeat(64) },
    { checksum_blake3: "f".repeat(64) },
    { created_at_utc: "1794000001" },
  ]) {
    let parses = 0;
    const h = await trajectoryHarness({ hasTreatment: false,
      parseTrajectories: async () => { parses += 1; return {}; } });
    serveTrajectories(h, [trajectoryArtifact("control")], (value) => ({ ...value,
      filename: change.filename ?? value.filename, metadata: { ...value.metadata, ...change } }));
    assert.equal(await h.controller.previewTrajectories(), null, JSON.stringify(change));
    assert.equal(parses, 0);
    assert.equal(h.controller.trajectoryModel, null);
    assert.equal(h.controller.reportModel, h.report);
  }
});

test("trajectory parser failures preserve the readable study report and permit another attempt", async () => {
  const h = await trajectoryHarness({ parseTrajectories: async () => { throw new Error("trajectory endpoint differs"); } });
  serveTrajectories(h, [trajectoryArtifact("control"), trajectoryArtifact("treatment")]);
  assert.equal(await h.controller.previewTrajectories(), null);
  assert.match(h.controller.notice, /trajectory endpoint differs/);
  assert.equal(h.controller.reportModel, h.report);
  assert.equal(h.controller.trajectoryModel, null);
  assert.equal(h.controller.canOperate, true);
});

test("closing the Lab during trajectory discovery prevents late downloads", async () => {
  let parses = 0;
  const h = await trajectoryHarness({ parseTrajectories: async () => { parses += 1; return {}; } });
  const preview = h.controller.previewTrajectories();
  h.controller.setEnabled(false);
  assert.equal(h.queues.artifacts[0].signal.aborted, true);
  h.queues.artifacts[0].resolve(artifactPage([trajectoryArtifact("control"), trajectoryArtifact("treatment")]));
  assert.equal(await preview, null);
  assert.equal(h.queues.download.length, 1, "only the earlier report download occurred");
  assert.equal(parses, 0);
  assert.equal(h.controller.trajectoryModel, null);
  assert.equal(h.controller.reportModel, h.report);
});

test("a new source during trajectory download aborts the old read before another arm or parser runs", async () => {
  let parses = 0;
  const h = await trajectoryHarness({ parseTrajectories: async () => { parses += 1; return {}; } });
  const preview = h.controller.previewTrajectories();
  h.queues.artifacts[0].resolve(artifactPage([trajectoryArtifact("control"), trajectoryArtifact("treatment")]));
  await flush();
  const pending = h.queues.download[1];
  assert.equal(pending.request.artifactId, "summary-control");
  h.controller.observe(frame("7"), SOURCE);
  assert.equal(pending.signal.aborted, true);
  pending.resolve(trajectoryDownload(pending.request.expected));
  assert.equal(await preview, null);
  assert.equal(h.queues.download.length, 2, "the treatment was never fetched after source replacement");
  assert.equal(parses, 0);
  assert.equal(h.controller.source.sessionId, "7");
  assert.equal(h.controller.reportModel, null);
  assert.equal(h.controller.trajectoryModel, null);
});

test("trajectory decoding cannot publish after the Lab is disabled", async () => {
  let finishParsing;
  const h = await trajectoryHarness({ parseTrajectories: () => new Promise((resolve) => { finishParsing = resolve; }) });
  const downloads = serveTrajectories(h, [trajectoryArtifact("control"), trajectoryArtifact("treatment")]);
  const preview = h.controller.previewTrajectories();
  await flush();
  assert.equal(typeof finishParsing, "function");
  assert.equal(downloads.length, 2);
  h.controller.setEnabled(false);
  finishParsing(Object.freeze({ studyId: "stale-study" }));
  assert.equal(await preview, null);
  assert.equal(h.controller.trajectoryModel, null);
  assert.equal(h.controller.reportModel, h.report);
});

test("replacing a report or starting another study clears its previously loaded trajectories", async () => {
  const h = await trajectoryHarness();
  serveTrajectories(h, [trajectoryArtifact("control"), trajectoryArtifact("treatment")]);
  assert.ok(await h.controller.previewTrajectories());
  assert.ok(h.controller.trajectoryModel);
  h.controller.services.download = async () => ({ blob: new Blob(["{}"]), filename: "study_report.json", metadata: artifact() });
  assert.equal(await h.controller.previewReport(), h.report);
  assert.equal(h.controller.trajectoryModel, null);
  serveTrajectories(h, [trajectoryArtifact("control"), trajectoryArtifact("treatment")]);
  assert.ok(await h.controller.previewTrajectories());
  const next = h.controller.launch({ branchName: "next", ticks: 10 });
  assert.equal(h.controller.reportModel, null);
  assert.equal(h.controller.trajectoryModel, null);
  h.queues.launch[1].resolve(status(h.queues.launch[1].request, checkpoint(), { branch_id: "branch-next" }));
  await next;
});

test("selecting another retained study clears the previous study's trajectories without submitting science", async () => {
  const h = await trajectoryHarness();
  serveTrajectories(h, [trajectoryArtifact("control"), trajectoryArtifact("treatment")]);
  assert.ok(await h.controller.previewTrajectories());
  const receipt = study({ branch_id: "branch-two", status: "completed", artifact_id: "retained-report-two" });
  h.controller.services.listStudies = async () => historyPage([receipt]);
  await h.controller.refreshHistory(true);
  const selecting = h.controller.selectStudy(receipt.branch_id);
  h.queues.status[0].resolve(receipt);
  await selecting;
  assert.equal(h.controller.branch.status.branch_id, receipt.branch_id);
  assert.equal(h.controller.branch.origin, "history");
  assert.equal(h.controller.reportModel, null);
  assert.equal(h.controller.trajectoryModel, null);
  assert.equal(h.queues.launch.length, 1);
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
      intervention: { type: "Drought", params: { region: null, duration_ticks: 20, growth_scale: 0.1 } },
      intervention_delay_ticks: 30, idempotency_key: "branch-one" };
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
    assert.equal(JSON.parse(requests[3].body).intervention_delay_ticks, 30);
    assert.equal(requests[7].url.search, "?limit=12&cursor=12");
    const count = requests.length;
    await assert.rejects(value.listCheckpoints({ sessionId: "7" }));
    await assert.rejects(value.listCheckpoints({ sessionId: SESSION, cursor: "01" }));
    await assert.rejects(value.checkpointBranchStatus("../escape", { sessionId: SESSION }));
    await assert.rejects(value.launchCheckpointBranch({ ...branch, ticks: 10001 }));
    for (const intervention_delay_ticks of [-1, 0.5, "30", null, NaN, Infinity, 120, 121]) {
      await assert.rejects(value.launchCheckpointBranch({ ...branch, intervention_delay_ticks }), /Treatment delay/);
    }
    await assert.rejects(value.launchCheckpointBranch({ ...branch, intervention: null }), /Choose a treatment/);
    assert.equal(requests.length, count, "invalid timing must be refused before any HTTP request");
  } finally { globalThis.fetch = previous; }
});

test("saved-world transport decodes an isolated bounded frame without moving live cursors", async () => {
  const previous = globalThis.fetch, requests = [];
  const preview = savedCanvas();
  const wire = { schema: "scriptbots.remote-frame.v2", kind: "canvas", identity: preview.frame.identity,
    payload: preview.frame.snapshot, environment: preview.frame.environment, playback: preview.frame.playback,
    coalescedPublications: "0" };
  const envelope = { schema: preview.schema, sourceSessionId: SESSION, checkpoint: preview.checkpoint, frameBase64: "U0JXUw==" };
  globalThis.fetch = async (url, options) => {
    requests.push({ url: new URL(url), ...options });
    return new Response(JSON.stringify(envelope));
  };
  try {
    const value = client();
    const live = { ...wire, identity: { ...wire.identity, tick: "120", publicationRevision: "100" },
      payload: { ...wire.payload, tick: 120 } };
    value.tracker.accept(live, () => true);
    value.tracker.accept({ ...live, kind: "telemetry" }, () => true);
    const cursor = structuredClone(value.resume);
    const observed = [...value.tracker.observed];
    let decodes = 0;
    value.decodeFrame = (bytes) => { decodes += 1; assert.deepEqual([...bytes], [83, 66, 87, 83]); return wire; };
    value.onFrame = () => { throw new Error("A saved frame must not enter the stream"); };
    const result = await value.checkpointCanvas(checkpoint().checkpoint_id, { sessionId: SESSION, expected: checkpoint() });
    assert.deepEqual(result, preview);
    assert.equal(decodes, 1);
    assert.deepEqual(value.resume, cursor);
    assert.deepEqual([...value.tracker.observed], observed);
    assert.equal(requests[0].method, "GET");
    assert.equal(requests[0].url.pathname, "/nested/api/v1/checkpoints/ckpt-saved-t100/canvas");
    assert.equal(requests[0].url.search, "?session_id=" + SESSION);
    assert.equal(requests[0].headers["X-ScriptBots-Session"], SESSION);
    assert.equal(requests[0].body, undefined);
    assert.equal(requests[0].credentials, "omit");
    assert.equal(requests[0].cache, "no-store");
    await assert.rejects(value.checkpointCanvas("ckpt-different", { sessionId: SESSION, expected: checkpoint() }));
    assert.equal(requests.length, 1);
    envelope.checkpoint = checkpoint({ checksum_blake3: "d".repeat(64) });
    await assert.rejects(value.checkpointCanvas(checkpoint().checkpoint_id, { sessionId: SESSION, expected: checkpoint() }), /selected checkpoint/);
    assert.deepEqual(value.resume, cursor);
  } finally { globalThis.fetch = previous; }
});

test("saved-world transport refuses malformed, oversized, incomplete and stale responses", async () => {
  const previous = globalThis.fetch;
  const preview = savedCanvas();
  const wire = { schema: "scriptbots.remote-frame.v2", kind: "canvas", identity: preview.frame.identity,
    payload: preview.frame.snapshot, environment: preview.frame.environment, playback: preview.frame.playback,
    coalescedPublications: "0" };
  let envelope = { schema: preview.schema, sourceSessionId: SESSION, checkpoint: preview.checkpoint, frameBase64: "U0JXUw==" };
  globalThis.fetch = async () => new Response(JSON.stringify(envelope));
  try {
    const value = client();
    let decodes = 0;
    value.decodeFrame = () => { decodes += 1; return wire; };
    for (const encoded of ["", "U0JXUw", "U0JXUw==\n", "U0JXUx==", "A".repeat(Math.ceil(8 * 1024 * 1024 / 3) * 4 + 4)]) {
      envelope.frameBase64 = encoded;
      await assert.rejects(value.checkpointCanvas(checkpoint().checkpoint_id, { sessionId: SESSION, expected: checkpoint() }));
    }
    assert.equal(decodes, 0, "invalid base64 and byte counts refuse before WASM");
    envelope.frameBase64 = "U0JXUw==";
    wire.coalescedPublications = "1";
    await assert.rejects(value.checkpointCanvas(checkpoint().checkpoint_id, { sessionId: SESSION, expected: checkpoint() }), /complete reconstructed canvas/);
    wire.coalescedPublications = "0";
    wire.environment = null;
    await assert.rejects(value.checkpointCanvas(checkpoint().checkpoint_id, { sessionId: SESSION, expected: checkpoint() }), /scientific cut/);
    let respond;
    globalThis.fetch = () => new Promise((resolve) => { respond = resolve; });
    const pending = value.checkpointCanvas(checkpoint().checkpoint_id, { sessionId: SESSION, expected: checkpoint() });
    const before = decodes;
    value.generation += 1;
    respond(new Response(JSON.stringify(envelope)));
    await assert.rejects(pending, /connection changed/);
    assert.equal(decodes, before);
  } finally { globalThis.fetch = previous; }
});

test("historical replay transport isolates exact target reads from all live stream cursors", async () => {
  const previous = globalThis.fetch, requests = [];
  const preview = replayCanvas();
  const wire = { schema: "scriptbots.remote-frame.v2", kind: "canvas", identity: preview.frame.identity,
    payload: preview.frame.snapshot, environment: preview.frame.environment, playback: preview.frame.playback,
    coalescedPublications: "0" };
  const { frame: _frame, ...descriptor } = preview;
  const envelope = { ...descriptor, frameBase64: "U0JXUw==" };
  globalThis.fetch = async (url, options) => {
    requests.push({ url: new URL(url), ...options });
    return new Response(JSON.stringify(envelope));
  };
  try {
    const value = client();
    const live = { ...wire, identity: { ...wire.identity, tick: "120", publicationRevision: "100" },
      payload: { ...wire.payload, tick: 120 } };
    value.tracker.accept(live, () => true);
    value.tracker.accept({ ...live, kind: "telemetry" }, () => true);
    const cursor = structuredClone(value.resume), observed = [...value.tracker.observed];
    let decodes = 0;
    value.decodeFrame = (bytes) => { decodes += 1; assert.deepEqual([...bytes], [83, 66, 87, 83]); return wire; };
    value.onFrame = () => { throw new Error("A replay must not enter the live stream"); };
    const options = { sessionId: SESSION, expected: checkpoint(), targetTick: "112" };
    assert.deepEqual(await value.checkpointReplay(checkpoint().checkpoint_id, options), preview);
    assert.equal(decodes, 1);
    assert.deepEqual(value.resume, cursor);
    assert.deepEqual([...value.tracker.observed], observed);
    assert.equal(requests[0].method, "GET");
    assert.equal(requests[0].url.pathname, "/nested/api/v1/checkpoints/ckpt-saved-t100/replay");
    assert.equal(requests[0].url.search, `?session_id=${SESSION}&target_tick=112`);
    assert.equal(requests[0].headers["X-ScriptBots-Session"], SESSION);
    assert.equal(requests[0].body, undefined);
    assert.equal(requests[0].credentials, "omit");
    assert.equal(requests[0].cache, "no-store");
    for (const targetTick of ["99", "0112", "1e3", 112, "9007199254740992", "18446744073709551615"]) {
      await assert.rejects(value.checkpointReplay(checkpoint().checkpoint_id, { ...options, targetTick }));
    }
    await assert.rejects(value.checkpointReplay("different-source", options));
    assert.equal(requests.length, 1, "unsafe or ambiguous targets never reach the host");
    envelope.verification = { ...preview.verification, verifiedThroughTick: "1125" };
    await assert.rejects(value.checkpointReplay(checkpoint().checkpoint_id, options), /complete replay verification interval/);
    envelope.verification = preview.verification;
    envelope.checkpoint = checkpoint({ created_at_utc: "1794000001" });
    await assert.rejects(value.checkpointReplay(checkpoint().checkpoint_id, options), /selected checkpoint/);
    assert.deepEqual(value.resume, cursor);
    assert.deepEqual([...value.tracker.observed], observed);
  } finally { globalThis.fetch = previous; }
});

test("historical replay transport rejects malformed frames and stale connections before publishing", async () => {
  const previous = globalThis.fetch;
  const preview = replayCanvas();
  const wire = { schema: "scriptbots.remote-frame.v2", kind: "canvas", identity: preview.frame.identity,
    payload: preview.frame.snapshot, environment: preview.frame.environment, playback: preview.frame.playback,
    coalescedPublications: "0" };
  const { frame: _frame, ...descriptor } = preview;
  const envelope = { ...descriptor, frameBase64: "U0JXUw==" };
  globalThis.fetch = async () => new Response(JSON.stringify(envelope));
  try {
    const value = client();
    const options = { sessionId: SESSION, expected: checkpoint(), targetTick: "112" };
    let decodes = 0;
    value.decodeFrame = () => { decodes += 1; return wire; };
    for (const encoded of ["", "U0JXUw", "U0JXUw==\n", "U0JXUx=="]) {
      envelope.frameBase64 = encoded;
      await assert.rejects(value.checkpointReplay(checkpoint().checkpoint_id, options));
    }
    assert.equal(decodes, 0);
    envelope.frameBase64 = "U0JXUw==";
    wire.coalescedPublications = "1";
    await assert.rejects(value.checkpointReplay(checkpoint().checkpoint_id, options), /complete reconstructed canvas/);
    wire.coalescedPublications = "0";
    wire.environment = null;
    await assert.rejects(value.checkpointReplay(checkpoint().checkpoint_id, options), /historical target/);
    let respond;
    globalThis.fetch = () => new Promise((resolve) => { respond = resolve; });
    const pending = value.checkpointReplay(checkpoint().checkpoint_id, options);
    const before = decodes;
    value.generation += 1;
    respond(new Response(JSON.stringify(envelope)));
    await assert.rejects(pending, /connection changed/);
    assert.equal(decodes, before);
  } finally { globalThis.fetch = previous; }
});

test("saved-world and replay response bounds use actual bytes even with a dishonest Content-Length", async () => {
  const previous = globalThis.fetch;
  let cancelled = false, decoded = false;
  globalThis.fetch = async () => new Response(new ReadableStream({
    start(controller) { controller.enqueue(new Uint8Array(12 * 1024 * 1024 + 1)); },
    cancel() { cancelled = true; },
  }), { headers: { "Content-Length": "1" } });
  try {
    const value = client();
    value.decodeFrame = () => { decoded = true; };
    for (const method of ["checkpointCanvas", "checkpointReplay"]) {
      cancelled = false;
      await assert.rejects(value[method](checkpoint().checkpoint_id,
        { sessionId: SESSION, expected: checkpoint(), targetTick: "112" }), /bound/);
      assert.equal(cancelled, true);
      assert.equal(decoded, false);
    }
  } finally { globalThis.fetch = previous; }
});

test("study history transport uses GET pagination and the current exact host session", async () => {
  const previous = globalThis.fetch, requests = [];
  globalThis.fetch = async (url, options) => {
    requests.push({ url: new URL(url), ...options });
    return new Response(JSON.stringify(historyPage()));
  };
  try {
    const value = client();
    await value.listCheckpointBranches({ sessionId: SESSION, cursor: "12", limit: 12 });
    assert.equal(requests[0].method, "GET");
    assert.equal(requests[0].url.pathname, "/nested/api/v1/checkpoints/branches");
    assert.equal(requests[0].url.search, "?limit=12&cursor=12");
    assert.equal(requests[0].headers["X-ScriptBots-Session"], SESSION);
    assert.equal(requests[0].body, undefined);
    await assert.rejects(value.listCheckpointBranches({ sessionId: SESSION, cursor: "01" }));
    await assert.rejects(value.listCheckpointBranches({ sessionId: "7" }));
    assert.equal(requests.length, 1);
  } finally { globalThis.fetch = previous; }
});

test("report preview byte limits refuse oversized metadata before requesting binary content", async () => {
  const previous = globalThis.fetch, requests = [];
  const metadata = artifact({ byte_size: MAX_STUDY_REPORT_BYTES + 1 });
  globalThis.fetch = async (url) => {
    requests.push(String(url));
    return new Response(JSON.stringify(metadata));
  };
  try {
    const value = client();
    await assert.rejects(value.downloadArtifact(metadata.artifact_id,
      { sessionId: SESSION, maxBytes: MAX_STUDY_REPORT_BYTES }), /browser download limit/);
    assert.equal(requests.length, 1);
    assert.ok(!requests[0].endsWith("/download"));
    for (const maxBytes of [0, -1, 0.5, Infinity, MAX_LAB_ARTIFACT_BYTES + 1]) {
      await assert.rejects(value.downloadArtifact(metadata.artifact_id, { sessionId: SESSION, maxBytes }));
    }
    assert.equal(requests.length, 1);
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
