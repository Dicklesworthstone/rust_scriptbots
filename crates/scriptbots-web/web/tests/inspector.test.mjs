import assert from "node:assert/strict";
import test from "node:test";
import { InspectionController, agentUid, exactIdentity, inspectionStatus, pickAgent, validateInspection } from "../inspector.js";
import { CanvasRenderer } from "../renderers/canvas.js";
import { WebGpuRenderer } from "../renderers/webgpu.js";
import { RemoteStreamClient } from "../remote.js";

const HUGE_UID = "18446744073709551615";
const flush = () => new Promise((resolve) => setImmediate(resolve));

function agent(uid = HUGE_UID, position = [50, 50]) {
  return { uid, id: "4294967297", position, health: 1, color: [0.2, 0.8, 0.2], boost: false };
}

function frame(revision = 1, agents = [agent()], epoch = "0", sessionId = "9007199254740993") {
  return { identity: { sessionId, publicationRevision: String(revision), scientificRevision: String(revision),
    controlRevision: "0", configRevision: "0", tick: String(revision) },
  snapshot: { tick: revision, epoch, world: { width: 100, height: 100 }, agents } };
}

function detail(request) {
  const revisions = { scientific: request.identity.scientificRevision,
    control: request.identity.controlRevision, config: request.identity.configRevision };
  return { schema: "scriptbots.agent-inspection.v1", uid: request.uid,
    source: { sessionId: request.sessionId, epoch: request.epoch,
      publicationRevision: request.identity.publicationRevision, tick: request.identity.tick,
      published: { ...revisions }, inspected: { ...revisions } },
    brain: "FtBrain", brainBound: true, health: 1, age: 9, generation: 3, energy: 1.25, foodBalanceTotal: 2.5,
    herbivoreTendency: 0.8, temperaturePreference: 0.5, soundOutput: 0.2, giveIntent: 0.1,
    reproductionCounter: 0.4, clockScales: [50, 60], hybrid: false, lineage: ["9007199254740993", null],
    mutationRates: { primary: 0.003, secondary: 0.05 },
    traitModifiers: { smell: 0.3, sound: 0.4, hearing: 1, eye: 1.5, blood: 1.5 },
    sensors: Array.from({ length: 25 }, (_, index) => ({ name: `sense${index}`, value: index / 25 })),
    outputs: Array.from({ length: 9 }, (_, index) => ({ name: `output${index}`, value: index / 9 })) };
}

function deferredController() {
  const reads = [];
  const changes = [];
  const painted = [];
  let time = 0;
  const controller = new InspectionController({ now: () => time,
    read: (uid, request, signal) => new Promise((resolve, reject) => reads.push({ uid, request, signal, resolve, reject })),
    onChange: (state) => changes.push(state), onFrame: (state, paintedFrame) => painted.push(inspectionStatus(state.detail, paintedFrame)) });
  return { controller, reads, changes, painted, advance: (ms) => { time += ms; } };
}

test("picking uses stable exact UIDs and CSS-scaled renderer positions", () => {
  assert.equal(exactIdentity(HUGE_UID), HUGE_UID);
  assert.equal(agentUid(agent(Number.MAX_SAFE_INTEGER + 2)), null);
  for (const value of ["01", "+1", "1e2", "18446744073709551616", 9_007_199_254_740_992]) {
    assert.throws(() => exactIdentity(value));
  }
  const snapshot = frame(1, [agent("9007199254740993", [20, 10]), agent(HUGE_UID, [80, 70])]).snapshot;
  const viewport = { width: 300, height: 200, canvasWidth: 1200, canvasHeight: 800 };
  assert.equal(pickAgent(snapshot, 62, 21, viewport), "9007199254740993");
  assert.equal(pickAgent(snapshot, 240, 140, viewport), HUGE_UID);
  assert.equal(pickAgent(snapshot, 150, 100, viewport), null);
  assert.equal(pickAgent(snapshot, -1, 20, viewport), null);
  assert.equal(pickAgent({ ...snapshot, agents: [agent("8"), agent("7")] }, 150, 100, viewport), "7");
});

test("selection is bounded to one read, throttles advancing publications, and labels retained samples immediately", async () => {
  const { controller, reads, painted, advance } = deferredController();
  controller.observe(frame(), 1);
  controller.select(HUGE_UID);
  controller.observe(frame(2), 1);
  assert.equal(reads.length, 1);
  reads[0].resolve(detail(reads[0].request));
  await flush();
  assert.equal(controller.state.status, "ready");
  assert.match(inspectionStatus(controller.state.detail, controller.frame), /Observed tick 1 · displayed tick 2/);
  controller.observe(frame(3), 1);
  assert.equal(reads.length, 1);
  assert.match(painted.at(-1), /displayed tick 3/);
  advance(250);
  controller.observe(frame(3), 1);
  assert.equal(reads.length, 2);
  controller.observe(frame(4), 1);
  assert.equal(reads.length, 2, "a slow owner must not accumulate reads");
  reads[1].resolve(detail(reads[1].request));
  await flush();
  advance(250);
  controller.observe(frame(4), 1);
  reads[2].resolve(detail(reads[2].request));
  await flush();
  assert.match(inspectionStatus(controller.state.detail, controller.frame), /matches displayed frame/);
  advance(10_000);
  controller.observe(frame(4), 1);
  assert.equal(reads.length, 3, "unchanged paused publication needs no polling");
});

test("late replies cannot replace a different selected UID, epoch, session or same-seed owner", async () => {
  const { controller, reads } = deferredController();
  const population = [agent(HUGE_UID), agent("9007199254740993")];
  controller.observe(frame(1, population), 1);
  controller.select(HUGE_UID);
  controller.select("9007199254740993");
  assert.equal(reads[0].signal.aborted, true);
  reads[0].resolve(detail(reads[0].request));
  await flush();
  assert.equal(controller.state.uid, "9007199254740993");
  assert.equal(controller.state.detail, null);
  reads[1].resolve(detail(reads[1].request));
  await flush();
  assert.equal(controller.state.detail.uid, "9007199254740993");
  for (const [nextFrame, generation] of [[frame(2, population, "1"), 1],
    [frame(2, population, "1", "88"), 1], [frame(2, population, "1", "88"), 2]]) {
    void controller.refresh(true);
    const pending = reads.at(-1);
    controller.observe(nextFrame, generation);
    assert.equal(pending.signal.aborted, true);
    pending.resolve(detail(pending.request));
    await flush();
    assert.equal(controller.selectedUid, null);
    assert.equal(controller.state.status, "empty");
    controller.select(HUGE_UID);
    reads.at(-1).resolve(detail(reads.at(-1).request));
    await flush();
  }
  void controller.refresh(true);
  const pending = reads.at(-1);
  controller.reset();
  pending.resolve(detail(pending.request));
  await flush();
  assert.equal(controller.frame, null);
  assert.equal(controller.state.status, "empty");
});

test("death removes stale detail and a recycled arena handle cannot inherit selection", async () => {
  const { controller, reads } = deferredController();
  controller.observe(frame(), 1);
  controller.select(HUGE_UID);
  controller.observe(frame(2, [agent("123")]), 1);
  assert.equal(reads[0].signal.aborted, true);
  reads[0].resolve(detail(reads[0].request));
  await flush();
  assert.equal(controller.state.status, "missing");
  assert.equal(controller.state.uid, HUGE_UID);
  assert.equal(controller.state.detail, null);
  await controller.refresh(true);
  assert.equal(reads.length, 1);
  controller.select(null);
  assert.equal(controller.state.status, "empty");
});

test("inspection validation rejects stale source revisions, wrong subjects and unbounded or nonfinite values", () => {
  const source = frame(9);
  const request = { uid: HUGE_UID, sessionId: source.identity.sessionId, epoch: "0", identity: source.identity };
  assert.equal(validateInspection(detail(request), request).uid, HUGE_UID);
  const changes = [
    (value) => { value.uid = "7"; },
    (value) => { value.source.epoch = "1"; },
    (value) => { value.source.sessionId = "8"; },
    (value) => { value.source.publicationRevision = "8"; },
    (value) => { value.source.published.scientific = "8"; },
    (value) => { value.source.inspected.scientific = "8"; },
    (value) => { value.sensors.push(value.sensors[0]); },
    (value) => { value.outputs[6].value = Infinity; },
    (value) => { value.brain = "x".repeat(257); },
    (value) => { value.lineage[0] = 9_007_199_254_740_992; },
  ];
  for (const change of changes) {
    const value = detail(request);
    change(value);
    assert.throws(() => validateInspection(value, request));
  }
});

test("Canvas highlights only the exact selected creature and clears without changing snapshot values", () => {
  const arcs = [];
  const strokes = [];
  const context = { fillRect() {}, beginPath() {}, fill() {},
    arc(...args) { arcs.push(args); }, stroke() { strokes.push({ style: this.strokeStyle, arc: arcs.at(-1) }); } };
  const canvas = { width: 100, height: 100, getContext: () => context };
  const renderer = new CanvasRenderer(canvas);
  const snapshot = frame(1, [agent(HUGE_UID, [20, 20]), agent("9007199254740993", [80, 80])]).snapshot;
  const before = JSON.stringify(snapshot);
  renderer.render(snapshot, { selectedUid: HUGE_UID });
  assert.equal(strokes.length, 1);
  assert.equal(strokes[0].style, "rgb(251, 191, 36)");
  assert.deepEqual(strokes[0].arc.slice(0, 2), [20, 20]);
  strokes.length = 0;
  renderer.render(snapshot, { selectedUid: null });
  assert.equal(strokes.length, 0);
  renderer.render({ ...snapshot, agents: [{ ...agent(), uid: undefined }] }, { selectedUid: null });
  assert.equal(strokes.length, 0);
  assert.equal(JSON.stringify(snapshot), before);
});

test("WebGPU uploads exact UID selection in the existing bounded instance buffer", async () => {
  const original = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };
  const draws = [];
  const buffer = () => ({ destroy() {} });
  const pass = { setPipeline() {}, setBindGroup() {}, draw(...args) { draws.push(args); }, end() {} };
  const device = { lost: new Promise(() => {}), addEventListener() {},
    limits: { maxBufferSize: 1_048_576, maxStorageBufferBindingSize: 1_048_576 },
    createBuffer: buffer, createBindGroup: () => ({}),
    createCommandEncoder: () => ({ beginRenderPass: () => pass, finish: () => ({}) }),
    queue: { writeBuffer() {}, submit() {}, onSubmittedWorkDone: () => Promise.resolve() }, destroy() {} };
  const context = { getCurrentTexture: () => ({ createView: () => ({}) }), unconfigure() {} };
  try {
    const renderer = new WebGpuRenderer({ width: 100, height: 100 }, device, context,
      { getBindGroupLayout: () => ({}) }, buffer(), "rgba8unorm", assert.fail, {});
    const snapshot = frame(1, [agent(HUGE_UID), agent("9007199254740993")]).snapshot;
    renderer.render(snapshot, { selectedUid: "9007199254740993" });
    assert.equal(renderer.agentArray[3], 0);
    assert.equal(renderer.agentArray[11], 1);
    assert.deepEqual(draws, [[6, 2, 0, 0]]);
    await flush();
    renderer.render(snapshot, { selectedUid: null });
    assert.equal(renderer.agentArray[3], 0);
    assert.equal(renderer.agentArray[11], 0);
    renderer.destroy();
  } finally { globalThis.GPUBufferUsage = original; }
});

test("remote inspection preserves exact query identities, proxy prefix and bounded responses", async () => {
  const original = globalThis.fetch;
  const client = new RemoteStreamClient({ url: "wss://host.test/lab/api/ws/stream", decodeFrame: () => {} });
  client.state = "connected";
  client.tracker.acknowledge("9007199254740993");
  const options = { sessionId: "9007199254740993", epoch: "18446744073709551615" };
  try {
    globalThis.fetch = async (url, request) => {
      assert.equal(url.origin, "https://host.test");
      assert.equal(url.pathname, "/lab/api/agents/inspect");
      assert.equal(url.searchParams.get("uid"), HUGE_UID);
      assert.equal(url.searchParams.get("session_id"), options.sessionId);
      assert.equal(url.searchParams.get("epoch"), options.epoch);
      assert.equal(request.credentials, "omit");
      assert.equal(request.cache, "no-store");
      return new Response(JSON.stringify({ uid: HUGE_UID }), { headers: { "content-type": "application/json" } });
    };
    assert.deepEqual(await client.inspectAgent(HUGE_UID, options), { uid: HUGE_UID });
    globalThis.fetch = async () => new Response(JSON.stringify({ message: "creature is no longer present in this host" }), { status: 404 });
    assert.equal(await client.inspectAgent(HUGE_UID, options), null);
    globalThis.fetch = async () => new Response(JSON.stringify({ message: "host session changed" }), { status: 409 });
    await assert.rejects(client.inspectAgent(HUGE_UID, options), /session changed/);
    globalThis.fetch = async () => new Response("x".repeat(16 * 1024 + 1));
    await assert.rejects(client.inspectAgent(HUGE_UID, options), /16 KiB/);
    globalThis.fetch = async () => new Response("{}", { headers: { "content-length": "16385" } });
    await assert.rejects(client.inspectAgent(HUGE_UID, options), /16 KiB/);
    await assert.rejects(client.inspectAgent(HUGE_UID, { ...options, sessionId: "8" }), /selected creature's host/);
  } finally { globalThis.fetch = original; }
});
