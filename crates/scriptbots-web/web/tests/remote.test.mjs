// Consumer-state tests. The Rust codec tests exercise actual HostCore bytes;
// these tests exercise the browser's cursor commits and request correlation.
import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import test from "node:test";
import { RemoteFrameTracker, RemoteStreamClient, RemoteStepBatchController, RemoteOwnerHttpError, STEP_BATCH_LIMITS,
  REMOTE_CONTROL_RECOVERY_LIMITS, parseRemoteControlRecovery, consumeRemoteFrame, validateRemoteOwnerStatus,
  validateRemoteStepBatch, remoteCommandHistoryView, describeRemoteCommandReceipt, describeRemoteOwnerFault } from "../remote.js";
import { EXPERIMENT_LIMITS, prepareExperimentRequest } from "../experiment-protocol.js";
import { EnvironmentRaster, environmentDescription } from "../renderers/environment.js";

const SESSION = "340282366920938463463374607431768211455";
const CONTROL_SESSION = "71";
const COMMAND = "1234567890abcdef1234567890abcdef";
if (!globalThis.crypto) Object.defineProperty(globalThis, "crypto", { value: webcrypto, configurable: true });

test("command receipt text preserves applied boundaries and independent failure axes", () => {
  const applied = {
    command_id: COMMAND,
    application_state: "applied",
    journal_state: "failed",
    applied_tick: 12,
    control_revision: 7,
    scientific_revision: 5,
    config_revision: 3,
    journal_failure: { code: "disk_full", message: "retained write failed", diagnostic_truncated: false },
  };
  const original = structuredClone(applied);
  const text = describeRemoteCommandReceipt(applied);
  assert.match(text, /application applied; journal failed/);
  assert.match(text, /Applied at tick 12; control revision 7, scientific revision 5, config revision 3/);
  assert.match(text, /Journal failure \(disk_full\): retained write failed/);
  assert.doesNotMatch(text, /Application failure|diagnostic truncated/);
  assert.deepEqual(applied, original);

  const failed = {
    ...applied,
    application_state: "failed",
    application_failure: { code: "world_step", message: "<img src=x> failed sensing", diagnostic_truncated: true },
  };
  const failedText = describeRemoteCommandReceipt(failed);
  assert.match(failedText, /application failed; journal failed/);
  assert.match(failedText, /Application failure \(world_step\): <img src=x> failed sensing \[diagnostic truncated\]/);
  assert.match(failedText, /Journal failure \(disk_full\): retained write failed/);
  assert.doesNotMatch(failedText, /Applied at tick|control revision/,
    "latest published revisions on a failed command do not become an applied boundary");

  const rejected = {
    command_id: COMMAND,
    application_state: "rejected",
    journal_state: "pending",
    rejection_reason: { stale_revision: { expected: 9, actual: 10 } },
  };
  assert.match(describeRemoteCommandReceipt(rejected), /Rejection: .*stale_revision/);
  assert.doesNotMatch(describeRemoteCommandReceipt(rejected), /Applied at tick|failure/);
  assert.doesNotMatch(describeRemoteCommandReceipt({ ...applied, application_state: "queued" }), /Applied at tick/);
});

test("command receipt text never reports rounded numeric fields as an exact applied boundary", () => {
  const receipt = {
    command_id: COMMAND,
    application_state: "applied",
    journal_state: "durable",
    applied_tick: 0,
    control_revision: 0,
    scientific_revision: 0,
    config_revision: 0,
  };
  assert.match(describeRemoteCommandReceipt(receipt), /Applied at tick 0; control revision 0, scientific revision 0, config revision 0/);
  for (const [field, label] of [["applied_tick", "tick"], ["control_revision", "control revision"],
    ["scientific_revision", "scientific revision"], ["config_revision", "config revision"]]) {
    for (const value of [undefined, null, -1, 1.5, NaN, Infinity, "7", Number.MAX_SAFE_INTEGER + 1]) {
      assert.ok(describeRemoteCommandReceipt({ ...receipt, [field]: value })
        .includes(`${label} unavailable (not an exact browser integer)`));
    }
    assert.ok(describeRemoteCommandReceipt({ ...receipt, [field]: Number.MAX_SAFE_INTEGER })
      .includes(`${label} ${Number.MAX_SAFE_INTEGER}`));
  }
});

function environment() {
  return { width: 2, height: 2, sourceWidth: 2, sourceHeight: 2, cellSize: 40,
    terrainRevision: "1", foodRevision: "2", hydrologyRevision: "3", foodMax: 0.5,
    hydrologyEnabled: true, cells: [
      { terrain: "Grass", elevation: 0, food: 0, waterDepth: 0 },
      { terrain: "Sand", elevation: 1, food: 0.5, waterDepth: 0 },
      { terrain: "DeepWater", elevation: 0, food: 0, waterDepth: 0.15 },
      { terrain: "Rock", elevation: 0.5, food: 0.25, waterDepth: 1 },
    ] };
}

test("environment modes show actual food and water while retaining the full source extent", () => {
  const raster = new EnvironmentRaster();
  const env = environment();
  const world = { width: 80, height: 80 };
  const identity = { sessionId: SESSION };
  const food = raster.update(env, identity, world, "food");
  assert.deepEqual(Array.from(food.pixels.slice(0, 8)), [18, 27, 37, 255, 120, 180, 48, 255]);
  assert.deepEqual(food.coverage, [1, 1]);
  const water = raster.update(env, identity, world, "water");
  assert.deepEqual(Array.from(water.pixels.slice(8, 12)), [29, 80, 116, 255]);
  const terrain = raster.update(env, identity, world, "terrain");
  assert.deepEqual(Array.from(terrain.pixels.slice(0, 4)), [44, 65, 34, 255]);
  assert.notDeepEqual(terrain.pixels, raster.update(env, identity, world, "combined").pixels);
  assert.match(environmentDescription(env), /exact cell view/);
});

test("environment cache is bounded to one revisioned view and cannot leak another session", () => {
  const raster = new EnvironmentRaster();
  const env = environment();
  const world = { width: 80, height: 80 };
  const identity = { sessionId: SESSION };
  const first = raster.update(env, identity, world);
  assert.equal(raster.update(env, { ...identity, publicationRevision: "200" }, world), first);
  const next = { ...env, foodRevision: "4", cells: env.cells.map((cell) => ({ ...cell, food: 0 })) };
  const changed = raster.update(next, identity, world);
  assert.notEqual(changed, first);
  assert.notDeepEqual(changed.pixels, first.pixels);
  const session = raster.update(next, { sessionId: "1" }, world);
  assert.notEqual(session, changed);
  assert.notEqual(raster.update(next, { sessionId: "1" }, world, "combined", 1), session);
  assert.equal(raster.update(null, identity, world), null);
  assert.equal(raster.current, null);
  assert.equal(raster.update(env, identity, world, "agents"), null);
  assert.match(environmentDescription(null), /unavailable/);
});

test("replacing a same-seed owner invalidates environment despite repeated revisions", () => {
  const raster = new EnvironmentRaster();
  const identity = { sessionId: "11" };
  const world = { width: 80, height: 80 };
  const first = raster.update(environment(), identity, world, "food", 0, 1);
  const replacement = environment();
  replacement.cells[1].food = 0;
  const next = raster.update(replacement, identity, world, "food", 0, 2);
  assert.notEqual(next, first);
  assert.deepEqual(Array.from(next.pixels.slice(4, 8)), [18, 27, 37, 255]);
  assert.equal(raster.update(replacement, identity, world, "food", 0, 2), next);
});

test("environment bounds reject shortened worlds, hostile cardinalities and invalid source values", () => {
  const identity = { sessionId: SESSION };
  const world = { width: 80, height: 80 };
  const paint = (env) => new EnvironmentRaster().update(env, identity, world);
  assert.throws(() => paint({ ...environment(), width: 1, sourceWidth: 1,
    cells: environment().cells.slice(0, 2) }), /disagrees/);
  assert.throws(() => paint({ ...environment(), cells: Array(128 * 128 + 1).fill(environment().cells[0]) }), /bounds/);
  for (const value of [NaN, Infinity, -1]) {
    const invalid = environment();
    invalid.cells[0].food = value;
    assert.throws(() => paint(invalid), /invalid source/);
  }
  assert.throws(() => paint({ ...environment(), terrainRevision: "18446744073709551616" }), /bounds/);
  assert.throws(() => paint({ ...environment(), hydrologyEnabled: false }), /invalid source/);
  const sampled = { ...environment(), width: 128, height: 128, sourceWidth: 1000, sourceHeight: 500 };
  assert.match(environmentDescription(sampled), /center-sampled overview/);
});

function frame(kind, revision, { tick = 20, paused = true, payload } = {}) {
  return {
    schema: "scriptbots.remote-frame.v2",
    kind,
    identity: { sessionId: SESSION, publicationRevision: String(revision),
      scientificRevision: String(tick), controlRevision: "4", configRevision: "2", tick: String(tick) },
    playback: { paused, speedMultiplier: 1 },
    coalescedPublications: "0",
    payload: payload ?? { tick },
  };
}

function eventPage(after, earliest, latest, first, count, revision = "7") {
  return frame("events", revision, {
    payload: {
      afterEvent: String(after), earliestRetained: String(earliest), latestEvent: String(latest),
      gap: BigInt(after) + 1n < BigInt(earliest)
        ? { firstMissing: String(BigInt(after) + 1n), lastMissing: String(BigInt(earliest) - 1n) } : null,
      events: Array.from({ length: count }, (_, index) => ({
        sequence: String(BigInt(first) + BigInt(index)),
        record: { schemaVersion: 1, tick: "12", kind: "population_boom", severity: 0.5, magnitude: 1,
          window: ["10", "12"], metric: "population", before: 1, after: 2, score: 1,
          subject: "agent:18446744073709551615", humanText: "Canonical server narrative." },
      })),
    },
  });
}

function tracker(channels) {
  const value = new RemoteFrameTracker(channels);
  value.acknowledge(SESSION);
  return value;
}

test("reconnect before the first payload requests publication zero, including a missing sibling channel", () => {
  const value = tracker(["canvas", "telemetry", "events"]);
  assert.deepEqual(value.resume(), { session_id: SESSION, after_event: "0" });
  value.accept(frame("telemetry", "0", { tick: 0 }), () => {});
  assert.deepEqual(value.resume(), { session_id: SESSION, after_event: "0" });
  value.accept(frame("canvas", "0", { tick: 0 }), () => {});
  assert.deepEqual(value.resume(), { session_id: SESSION, after_event: "0", after_revision: "0" });
});

test("resume uses the minimum independently consumed channel and preserves large exact revisions", () => {
  const value = tracker(["canvas", "telemetry"]);
  const revision = 9007199254740993n;
  value.accept(frame("canvas", revision), () => {});
  value.accept(frame("telemetry", revision + 1n), () => {});
  assert.equal(value.resume().after_revision, String(revision));
  value.accept(frame("canvas", revision + 2n), () => {});
  assert.equal(value.resume().after_revision, String(revision + 1n));
  assert.equal(value.resume(["canvas"]).after_revision, String(revision + 2n));
  value.acknowledge(SESSION, ["canvas", "events"]);
  assert.equal(value.resume().after_revision, String(revision + 2n));
  assert.throws(() => value.accept(frame("telemetry", revision + 3n), () => {}), /unnegotiated/);
});

test("consumer failure or explicit refusal cannot advance the presentation cursor", () => {
  const value = tracker(["canvas"]);
  const publication = frame("canvas", "8");
  assert.throws(() => value.accept(publication, () => { throw new Error("renderer failed"); }), /renderer failed/);
  assert.equal(value.resume().after_revision, undefined);
  assert.throws(() => value.accept(publication, () => false), /consumer refused/);
  assert.equal(value.resume().after_revision, undefined);
  assert.throws(() => value.accept(publication, () => Promise.resolve()), /synchronously/);
  assert.equal(value.resume().after_revision, undefined);
  let count = 0;
  assert.equal(value.accept(publication, () => { count += 1; }), true);
  assert.equal(value.accept(publication, () => { count += 1; }), false);
  assert.equal(count, 1);
  assert.equal(value.resume().after_revision, "8");
});

test("event pages resume after an explicit retention gap, paginate, and suppress duplicates", () => {
  const value = tracker(["events"]);
  const first = eventPage(0, 5, 70, 5, 64);
  const received = [];
  value.accept(first, (page) => received.push(page.payload));
  assert.equal(value.resume().after_event, "68");
  assert.deepEqual(received[0].gap, { firstMissing: "1", lastMissing: "4" });
  assert.equal(received[0].events.length, 64);
  assert.equal(value.accept(first, () => assert.fail("duplicate event page was redelivered")), false);
  value.accept(eventPage(68, 5, 70, 69, 2), (page) => received.push(page.payload));
  assert.equal(value.resume().after_event, "70");
  assert.equal(received.flatMap((page) => page.events).length, 66);
  assert.equal(received[1].events[1].sequence, "70");
  assert.equal(value.resume().after_revision, undefined);
});

test("consumer credits follow synchronous consumption and safe duplicates, never failed presentation", () => {
  const value = tracker(["canvas"]);
  const publication = frame("canvas", "8");
  const order = [];
  const acknowledge = (ack) => {
    assert.equal(value.resume().after_revision, "8");
    order.push(ack);
  };
  assert.throws(() => consumeRemoteFrame(value, publication,
    () => { throw new Error("renderer failed"); }, acknowledge), /renderer failed/);
  assert.throws(() => consumeRemoteFrame(value, publication, () => false, acknowledge), /consumer refused/);
  assert.deepEqual(order, []);
  assert.equal(consumeRemoteFrame(value, publication, () => { order.push("paint"); }, acknowledge), true);
  const ack = { protocol_version: 1, command: "stream_ack", session_id: SESSION,
    channel: "canvas", publication_revision: "8" };
  assert.deepEqual(order, ["paint", ack]);
  assert.equal(consumeRemoteFrame(value, publication, () => assert.fail("duplicate painted"), acknowledge), false);
  assert.deepEqual(order, ["paint", ack, ack]);
});

test("event credits identify the delivered page, including gaps and a failed acknowledgement send", () => {
  const value = tracker(["events"]);
  const acknowledgements = [];
  const acknowledge = (ack) => acknowledgements.push(ack);
  consumeRemoteFrame(value, eventPage(0, 5, 4, 5, 0), () => {}, acknowledge);
  assert.equal(acknowledgements[0].after_event, "4");
  const page = eventPage(4, 5, 6, 5, 2, "8");
  assert.throws(() => consumeRemoteFrame(value, page, () => {}, () => {
    throw new Error("connection closed before credit write");
  }), /connection closed/);
  assert.equal(value.resume().after_event, "6");
  consumeRemoteFrame(value, page, () => assert.fail("consumed page repeated"), acknowledge);
  assert.equal(acknowledgements[1].after_event, "6");
  consumeRemoteFrame(value, eventPage(6, 5, 6, 7, 0, "8"), () => {}, acknowledge);
  assert.equal(acknowledgements[2].after_event, "6");
  assert.equal(acknowledgements[2].channel, "events");
  assert.equal(acknowledgements[2].publication_revision, "8");
});

test("failed event delivery preserves its cursor and malformed loss or future ordinals fail closed", () => {
  const value = tracker(["events"]);
  const first = eventPage(0, 3, 5, 3, 3);
  assert.throws(() => value.accept(first, () => { throw new Error("event panel failed"); }), /event panel failed/);
  assert.equal(value.resume().after_event, "0");
  const withoutGap = structuredClone(first);
  withoutGap.payload.gap = null;
  assert.throws(() => value.accept(withoutGap, () => assert.fail()), /exact gap/);
  const discontinuous = structuredClone(first);
  discontinuous.payload.events[1].sequence = "5";
  assert.throws(() => value.accept(discontinuous, () => assert.fail()), /contiguous/);
  assert.throws(() => value.accept(eventPage(1, 3, 5, 3, 3), () => assert.fail()), /consumed cursor/);
  assert.throws(() => value.accept(eventPage(0, 1, 65, 1, 65), () => assert.fail()), /retention bounds/);
  const future = structuredClone(first);
  future.payload.events[0].record.tick = "21";
  assert.throws(() => value.accept(future, () => assert.fail()), /source-bound/);
  assert.equal(value.resume().after_event, "0");
  value.accept(first, () => {});
  assert.equal(value.resume().after_event, "5");
});

test("publication identities and observed playback must remain coherent across selected channels", () => {
  const value = tracker(["canvas", "telemetry"]);
  value.accept(frame("canvas", "10"), () => {});
  assert.throws(() => value.accept(frame("telemetry", "10", { paused: false }), () => assert.fail()), /disagree/);
  assert.throws(() => value.accept(frame("canvas", "9"), () => assert.fail()), /regressed/);
  assert.throws(() => value.accept(frame("canvas", "11", { tick: 19 }), () => assert.fail()), /regressed/);
  const mismatchedTick = frame("telemetry", "11");
  mismatchedTick.payload.tick = 19;
  assert.throws(() => value.accept(mismatchedTick, () => assert.fail()), /payload tick/);
  const wrongSchema = frame("canvas", "11");
  wrongSchema.schema = "scriptbots.remote-frame.v1";
  assert.throws(() => value.accept(wrongSchema, () => assert.fail()), /unnegotiated/);
  const rounded = frame("canvas", "11");
  rounded.identity.publicationRevision = 11;
  assert.throws(() => value.accept(rounded, () => assert.fail()), /decimal string/);
  const oversized = frame("canvas", "18446744073709551616");
  assert.throws(() => value.accept(oversized, () => assert.fail()), /integer range/);
  value.accept(frame("telemetry", "10"), () => {});
  assert.equal(value.resume().after_revision, "10");
});

test("a changed server session never resets or reuses an existing consumer cursor implicitly", () => {
  const value = tracker(["canvas", "events"]);
  value.accept(frame("canvas", "12"), () => {});
  value.accept(eventPage(0, 1, 2, 1, 2, "12"), () => {});
  const prior = value.resume();
  assert.throws(() => value.acknowledge("100"), { code: "session_changed" });
  assert.deepEqual(value.resume(), prior);
  const otherSession = frame("canvas", "13");
  otherSession.identity.sessionId = "100";
  assert.throws(() => value.accept(otherSession, () => assert.fail()), /unnegotiated/);
  value.acknowledge(SESSION);
  assert.deepEqual(value.resume(), prior);
});

function client(options = {}) {
  return new RemoteStreamClient({ url: "ws://127.0.0.1:8787/api/ws/stream",
    decodeFrame: () => assert.fail("Unexpected binary decoding in a control-state test"), ...options });
}

function retainUnconfirmed(value, requestId) {
  // A previously connected browser may lose its socket before its close
  // callback is delivered. Neither the send error nor a missing receipt proves
  // whether the owner admitted the attempted mutation.
  value.state = "connected";
  if (value.tracker.sessionId === null) value.tracker.acknowledge(CONTROL_SESSION);
  assert.throws(() => value.sendControl("step", { requestId }), (error) =>
    /socket is not open/.test(error.message) && error.requestId === requestId);
}

function connectedControlClient(options = {}) {
  const value = client(options), sent = [];
  value.state = "connected";
  value.tracker.acknowledge(CONTROL_SESSION);
  value.generation = 1;
  value.socket = { readyState: 1, bufferedAmount: 0, send: (text) => sent.push(JSON.parse(text)), close() {} };
  return { value, sent };
}

function commandReceipt(requestId, commandId = COMMAND, fields = {}) {
  return { protocol_version: 1, request_id: requestId, type: "receipt",
    receipt: { command_id: commandId, application_state: "admitted", journal_state: "pending", ...fields } };
}

function receiveControl(value, reply) { value.receive(JSON.stringify(reply)); }

test("command recovery captures the original before send and follows only observed owner settlement", () => {
  const saved = [], onWire = [];
  const { value } = connectedControlClient({ onControlHistoryChange: (owner) => saved.push(owner.exportControlRecovery()) });
  value.socket.send = (text) => onWire.push({ message: JSON.parse(text), saved: JSON.parse(saved.at(-1)) });
  value.sendControl("step", { requestId: "lost-first-reply" });
  assert.deepEqual(onWire[0].saved.commands, [{ requestId: "lost-first-reply", sessionId: CONTROL_SESSION,
    command: "step", speed: null, commandId: null }], "the reference is available before WebSocket.send can deliver the mutation");
  receiveControl(value, commandReceipt("lost-first-reply", COMMAND,
    { application_state: "applied", journal_state: "committed_volatile", settled: false }));
  assert.equal(JSON.parse(saved.at(-1)).commands[0].commandId, COMMAND);
  assert.equal(JSON.parse(saved.at(-1)).commands.length, 1, "a volatile intermediate observation remains recoverable");
  assert.ok(!saved.at(-1).includes("receipt") && !saved.at(-1).includes("settled"), "stored references are not owner evidence");
  const beforeLookup = saved.length;
  const lookup = value.retryControl("lost-first-reply");
  assert.equal(saved.length, beforeLookup, "read correlation slots never become saved mutations");
  receiveControl(value, { protocol_version: 1, request_id: lookup, type: "error", status: 404, message: "No retained receipt" });
  assert.equal(JSON.parse(saved.at(-1)).commands.length, 1, "a missing receipt cannot erase uncertain work");
  const secondLookup = value.retryControl("lost-first-reply");
  receiveControl(value, commandReceipt(secondLookup, COMMAND,
    { application_state: "applied", journal_state: "durable", settled: true }));
  assert.deepEqual(JSON.parse(saved.at(-1)).commands, [], "only observed completion releases a saved reference");
  assert.equal(onWire.filter((entry) => entry.message.command === "step").length, 1);

  value.socket.send = () => { throw new Error("socket write failed"); };
  assert.throws(() => value.sendControl("resume", { requestId: "uncertain-send" }), (error) =>
    error.requestId === "uncertain-send" && /socket write failed/.test(error.message));
  assert.equal(JSON.parse(saved.at(-1)).commands[0].requestId, "uncertain-send");
});

test("reloaded command recovery preserves exact requests and performs only explicit same-owner reads", () => {
  const { value: original } = connectedControlClient();
  original.sendControl("step", { requestId: "unknown-step" });
  original.sendControl("pause", { requestId: "known-pause" });
  receiveControl(original, commandReceipt("known-pause", COMMAND,
    { application_state: "applied", journal_state: "committed_volatile", settled: false }));
  original.sendControl("speed", { requestId: "known-speed", speed: 0.375 });
  const raw = original.exportControlRecovery(), parsed = parseRemoteControlRecovery(raw);
  assert.ok(Object.isFrozen(parsed) && Object.isFrozen(parsed.commands) && parsed.commands.every(Object.isFrozen));
  assert.throws(() => { parsed.commands[0].command = "resume"; }, TypeError);
  let changes = 0;
  const restored = client({ onControlHistoryChange: () => { changes += 1; } });
  assert.equal(restored.restoreControlRecovery(raw), 3);
  assert.equal(changes, 0);
  assert.equal(restored.socket, null);
  assert.equal(restored.statusRequest, null);
  assert.equal(restored.tracker.sessionId, null);
  assert.equal(restored.resume, undefined);
  assert.equal(restored.ownerAllowsMutation, false);
  for (const [key, request] of restored.requests) {
    assert.equal(request.text, original.requests.get(key).text);
    assert.equal(request.received, false);
  }
  assert.ok(restored.controlRequests().every((entry) => !entry.settled && entry.receipt === null && entry.error === null));
  assert.equal(restored.controlRequests()[1].commandId, COMMAND);
  assert.equal(remoteCommandHistoryView(restored).canRecover, false);
  assert.throws(() => restored.retryControl("unknown-step"), /Reconnect and subscribe/);

  const sent = [];
  restored.state = "connected";
  restored.generation = 1;
  restored.tracker.acknowledge(CONTROL_SESSION);
  restored.socket = { readyState: 1, bufferedAmount: 0, send: (text) => sent.push(JSON.parse(text)), close() {} };
  const byKey = restored.retryControl("unknown-step");
  assert.deepEqual(sent[0], { protocol_version: 1, request_id: byKey, command: "status_by_key",
    idempotency_key: "unknown-step", session_id: CONTROL_SESSION });
  receiveControl(restored, { protocol_version: 1, request_id: byKey, type: "error", status: 404, message: "History evicted" });
  assert.equal(restored.controlRequests()[0].settled, false);
  const byId = restored.retryControl("known-pause");
  assert.deepEqual(sent[1], { protocol_version: 1, request_id: byId, command: "status", command_id: COMMAND });
  receiveControl(restored, commandReceipt(byId, COMMAND,
    { application_state: "applied", journal_state: "durable", settled: true }));
  assert.equal(restored.controlRequests()[1].settled, true);
  const repeated = restored.sendControl("step", { requestId: "unknown-step" });
  assert.equal(sent[2].command, "status_by_key", "an explicit duplicate mutation request is still a read after reload");
  receiveControl(restored, { protocol_version: 1, request_id: repeated, type: "error", status: 404, message: "History evicted" });
  assert.throws(() => restored.sendControl("speed", { requestId: "known-speed", speed: 0.5 }), /different command/);
  assert.ok(sent.every((message) => ["status", "status_by_key"].includes(message.command)));
  restored.tracker.sessionId = "72";
  assert.throws(() => restored.retryControl("known-pause"), /another owner session/);
  assert.equal(sent.length, 3);
});

test("restored full command histories retain old sessions and preserve independent lookup capacity", () => {
  const { value: original } = connectedControlClient();
  original.sendControl("step", { requestId: "previous-owner" });
  original.tracker.sessionId = "9007199254740993";
  for (let index = 1; index < REMOTE_CONTROL_RECOVERY_LIMITS.entries; index += 1) {
    original.sendControl("step", { requestId: `retained-${index}` });
  }
  const raw = original.exportControlRecovery(), restored = client(), sent = [];
  assert.equal(restored.restoreControlRecovery(raw), 32);
  restored.state = "connected";
  restored.tracker.acknowledge("9007199254740993");
  restored.socket = { readyState: 1, bufferedAmount: 0, send: (text) => sent.push(JSON.parse(text)), close() {} };
  assert.throws(() => restored.sendControl("step", { requestId: "overflow" }), /32 control outcomes/);
  assert.throws(() => restored.retryControl("previous-owner"), /another owner session/);
  assert.deepEqual(sent, []);
  const lookup = restored.retryControl("retained-1");
  assert.equal(sent[0].session_id, "9007199254740993");
  receiveControl(restored, commandReceipt(lookup, COMMAND,
    { application_state: "applied", journal_state: "committed_volatile", settled: false }));
  assert.throws(() => restored.sendControl("step", { requestId: "overflow" }), /32 control outcomes/);
  assert.equal(JSON.parse(restored.exportControlRecovery()).commands.length, 32);
  const settledLookup = restored.retryControl("retained-1");
  receiveControl(restored, commandReceipt(settledLookup, COMMAND,
    { application_state: "applied", journal_state: "committed_volatile", settled: true }));
  assert.equal(JSON.parse(restored.exportControlRecovery()).commands.length, 31);
  restored.sendControl("step", { requestId: "next-original" });
  assert.equal(restored.requests.size, 32);
  assert.equal(restored.requests.has("retained-1"), false);
  assert.equal(restored.requests.get("previous-owner").sessionId, CONTROL_SESSION);
});

test("saved command references reject hostile state and duplicate identities atomically", () => {
  const { value: original } = connectedControlClient();
  original.sendControl("step", { requestId: "original" });
  const source = JSON.parse(original.exportControlRecovery());
  const invalid = (change, pattern = /saved|Saved|Invalid|Unsupported|original mutation|exact|bound/) => {
    const record = structuredClone(source);
    change(record);
    const fresh = client();
    assert.throws(() => fresh.restoreControlRecovery(JSON.stringify(record)), pattern);
    assert.equal(fresh.requests.size, 0, "a malformed tail cannot install a valid prefix");
    assert.equal(fresh.tracker.sessionId, null);
    assert.equal(fresh.socket, null);
  };
  invalid((record) => { record.schema = "scriptbots.remote-command-recovery.v2"; });
  invalid((record) => { record.ownerAllowsMutation = true; });
  invalid((record) => { record.commands[0].settled = true; });
  invalid((record) => { record.commands[0].receipt = { application_state: "applied", journal_state: "durable" }; });
  invalid((record) => { record.commands[0].sessionId = "18446744073709551616"; }, /integer range/);
  invalid((record) => { record.commands[0].sessionId = "01"; }, /canonical decimal/);
  for (const requestId of [null, "", 0, false, {}, []]) {
    invalid((record) => { record.commands[0].requestId = requestId; });
  }
  invalid((record) => { delete record.commands[0].requestId; });
  invalid((record) => { record.commands[0].requestId = " "; });
  invalid((record) => { record.commands[0].requestId = "x".repeat(1025); });
  invalid((record) => { record.commands[0].command = "status_by_key"; });
  invalid((record) => { record.commands[0].speed = 1; });
  invalid((record) => { record.commands[0].command = "speed"; record.commands[0].speed = -1; });
  invalid((record) => { record.commands[0].commandId = COMMAND.toUpperCase(); });
  invalid((record) => { record.commands.push({ ...record.commands[0], command: "pause" }); }, /duplicate/);
  invalid((record) => {
    record.commands[0].commandId = COMMAND;
    record.commands.push({ ...record.commands[0], requestId: "another-key" });
  }, /duplicate/);
  invalid((record) => { record.commands.push({ ...record.commands[0], requestId: "another-key", commandId: "invalid" }); });
  invalid((record) => { record.commands = [null]; });
  invalid((record) => { record.commands = Array.from({ length: 33 }, (_, index) => ({ ...source.commands[0], requestId: `key-${index}` })); });
  assert.throws(() => parseRemoteControlRecovery("{"), SyntaxError);
  for (const value of [null, [], {}, 1]) assert.throws(() => parseRemoteControlRecovery(value), /byte bound/);
  const limit = REMOTE_CONTROL_RECOVERY_LIMITS.bytes, raw = JSON.stringify(source);
  assert.deepEqual(parseRemoteControlRecovery(raw.padEnd(limit)), source);
  assert.throws(() => parseRemoteControlRecovery(raw.padEnd(limit + 1)), /byte bound/);
  assert.throws(() => parseRemoteControlRecovery(JSON.stringify({ value: "é".repeat(limit / 2) })), /byte bound/);
  const unicode = structuredClone(source);
  unicode.commands[0].requestId = "é".repeat(512);
  assert.equal(parseRemoteControlRecovery(JSON.stringify(unicode)).commands[0].requestId, unicode.commands[0].requestId);
  unicode.commands[0].requestId += "é";
  assert.throws(() => parseRemoteControlRecovery(JSON.stringify(unicode)), /Invalid remote request/);
});

test("command recovery is endpoint-bound and cannot replace an active or populated client", () => {
  const { value: original } = connectedControlClient();
  original.sendControl("step", { requestId: "endpoint-command" });
  const raw = original.exportControlRecovery();
  const another = client({ url: "wss://another.example/api/ws/stream" });
  assert.throws(() => another.restoreControlRecovery(raw), /another endpoint/);
  assert.equal(another.requests.size, 0);
  const populated = client();
  populated.restoreControlRecovery(raw);
  assert.throws(() => populated.restoreControlRecovery(raw), /fresh disconnected client/);
  assert.equal(populated.requests.size, 1);
  assert.throws(() => original.restoreControlRecovery(raw), /fresh disconnected client/);
  const previouslyConnected = client();
  previouslyConnected.tracker.acknowledge(CONTROL_SESSION);
  assert.throws(() => previouslyConnected.restoreControlRecovery(raw), /fresh disconnected client/);
  for (const url of ["ws://127.0.0.1:8787/api/ws/stream?token=private", "ws://127.0.0.1:8787/api/ws/stream#private"]) {
    const privateClient = client({ url });
    assert.throws(() => privateClient.exportControlRecovery(), /private parameters/);
    const record = JSON.parse(raw);
    record.url = url;
    assert.throws(() => parseRemoteControlRecovery(JSON.stringify(record)), /private parameters/);
  }
  for (const url of ["ws://user:secret@127.0.0.1:8787/api/ws/stream", "https://example.com/api/ws/stream", "ws://EXAMPLE.com/api/ws/stream"]) {
    const record = JSON.parse(raw);
    record.url = url;
    assert.throws(() => parseRemoteControlRecovery(JSON.stringify(record)), /cannot be saved/);
  }
});

test("same-endpoint replacement keeps trusted private-URL history without inheriting live authority", () => {
  const url = "wss://lab.example/api/ws/stream?token=not-persisted";
  const { value: previous, sent: previousSent } = connectedControlClient({ url });
  previous.sendControl("step", { requestId: "known-original" });
  receiveControl(previous, commandReceipt("known-original", COMMAND,
    { application_state: "applied", journal_state: "pending", settled: false, applied_tick: 8 }));
  previous.sendControl("pause", { requestId: "unknown-original" });
  const oldLookup = previous.retryControl("unknown-original");
  const observations = previous.controlRequests(), sentBefore = previousSent.length;
  let changes = 0;
  const next = client({ url, onControlHistoryChange: () => { changes += 1; } });
  assert.equal(next.inheritControlHistory(previous), 2);
  assert.deepEqual(next.controlRequests(), observations);
  assert.equal(next.controlRequests()[0].receipt, observations[0].receipt, "immutable observations retain their exact known boundary");
  assert.notEqual(next.requests.get("known-original"), previous.requests.get("known-original"));
  assert.equal(next.requests.get("unknown-original").text, previous.requests.get("unknown-original").text);
  assert.equal(next.socket, null);
  assert.equal(next.tracker.sessionId, null);
  assert.equal(next.ownerStatus, null);
  assert.equal(next.ownerStatusAvailable, false);
  assert.equal(next.ownerAllowsMutation, false);
  assert.equal(next.statusRequest, null, "an old pending read never occupies the new connection's lookup slot");
  assert.equal(next.resume, undefined);
  assert.equal(next.generation, 0);
  assert.equal(changes, 0);
  assert.equal(previousSent.length, sentBefore);
  assert.throws(() => next.exportControlRecovery(), /private parameters/, "memory inheritance cannot enable storage of a private URL");
  assert.throws(() => next.retryControl("unknown-original"), /Reconnect and subscribe/);

  receiveControl(previous, commandReceipt("known-original", COMMAND,
    { application_state: "applied", journal_state: "durable", settled: true, applied_tick: 8 }));
  receiveControl(previous, { protocol_version: 1, request_id: oldLookup, type: "error", status: 404, message: "old read failed" });
  assert.equal(previous.controlRequests()[0].settled, true);
  assert.deepEqual(next.controlRequests(), observations, "late old-connection updates cannot mutate copied observations");
  const sent = [];
  next.state = "connected";
  next.tracker.acknowledge(CONTROL_SESSION);
  next.socket = { readyState: 1, bufferedAmount: 0, send: (text) => sent.push(JSON.parse(text)), close() {} };
  const lookup = next.retryControl("unknown-original");
  assert.equal(sent[0].command, "status_by_key");
  receiveControl(next, commandReceipt(lookup, "2".repeat(32), { application_state: "rejected", journal_state: "not_required" }));
  const knownLookup = next.retryControl("known-original");
  assert.equal(sent[1].command, "status");
  receiveControl(next, commandReceipt(knownLookup, COMMAND,
    { application_state: "applied", journal_state: "durable", settled: true, applied_tick: 8 }));
  assert.ok(next.controlRequests().every((entry) => entry.settled));
  assert.ok(sent.every((request) => ["status", "status_by_key"].includes(request.command)));
  next.tracker.sessionId = "72";
  assert.throws(() => next.retryControl("known-original"), /another owner session/);
  assert.equal(sent.length, 2);
});

test("memory inheritance refuses wrong endpoints, active clients and replacement of retained history", () => {
  const { value: previous } = connectedControlClient();
  previous.sendControl("step", { requestId: "keep-original" });
  const fresh = client();
  assert.throws(() => fresh.inheritControlHistory({ url: previous.url, requests: previous.requests }), /another client/);
  assert.throws(() => fresh.inheritControlHistory(fresh), /another client/);
  assert.equal(fresh.requests.size, 0);
  const different = client({ url: "wss://other.example/api/ws/stream" });
  assert.throws(() => different.inheritControlHistory(previous), /same endpoint/);
  assert.equal(different.requests.size, 0);
  assert.throws(() => previous.inheritControlHistory(fresh), /fresh disconnected client/);
  const negotiated = client();
  negotiated.tracker.acknowledge(CONTROL_SESSION);
  assert.throws(() => negotiated.inheritControlHistory(previous), /fresh disconnected client/);
  fresh.inheritControlHistory(previous);
  const original = fresh.requests.get("keep-original");
  assert.throws(() => fresh.inheritControlHistory(previous), /fresh disconnected client/);
  assert.equal(fresh.requests.get("keep-original"), original);
});

test("failed send retains one recovery identity, refuses conflicting reuse, and bounds history", () => {
  const value = client();
  retainUnconfirmed(value, "step-1");
  const retained = value.requests.get("step-1").text;
  assert.deepEqual(JSON.parse(retained), { protocol_version: 1, request_id: "step-1", command: "step" });
  assert.throws(() => value.sendControl("step", { requestId: "step-1" }), (error) =>
    /socket is not open/.test(error.message) && error.requestId !== "step-1");
  assert.equal(value.statusRequest.command, "status_by_key", "repeating an attempted mutation only reads its outcome");
  assert.equal(value.statusRequest.idempotencyKey, "step-1");
  assert.equal(value.requests.size, 1);
  assert.equal(value.requests.get("step-1").text, retained);
  assert.throws(() => value.sendControl("resume", { requestId: "step-1" }), /different command/);
  assert.throws(() => value.retryControl("step-1"), /socket is not open/,
    "a failed read must release its lookup slot for another safe read");
  assert.equal(value.requests.get("step-1").text, retained);
  for (let index = 2; index <= 32; index += 1) retainUnconfirmed(value, `step-${index}`);
  assert.equal(value.requests.size, 32);
  assert.throws(() => value.sendControl("step", { requestId: "step-33" }), /32 control outcomes/);
  receiveControl(value, commandReceipt("step-1", COMMAND, { application_state: "applied", journal_state: "not_required" }));
  retainUnconfirmed(value, "step-33");
  assert.equal(value.requests.size, 32);
  assert.equal(value.requests.has("step-1"), false);
  assert.throws(() => value.retryControl("step-1"), /no longer retained/);
});

test("receipt correlation preserves command identity and exposes server errors to the UI", () => {
  const receipts = [];
  const value = client({ onReceipt: (reply) => receipts.push(reply) });
  retainUnconfirmed(value, "step-1");
  const receipt = commandReceipt("step-1");
  receiveControl(value, receipt);
  assert.deepEqual(receipts[0], receipt);
  const changed = structuredClone(receipt);
  changed.receipt.command_id = "ffffffffffffffffffffffffffffffff";
  assert.throws(() => receiveControl(value, changed), /changed its canonical command identity/);
  assert.equal(receipts.length, 1);
  const error = { protocol_version: 1, request_id: "step-1", type: "error", status: 409, message: "Request cannot be re-admitted" };
  receiveControl(value, error);
  assert.deepEqual(receipts[1], error);
  assert.equal(value.controlRequests()[0].settled, false, "an error is not an owner outcome");
  assert.throws(() => receiveControl(value, { ...receipt, request_id: "unknown" }), /Uncorrelated/);
});

test("a full mutation history preserves both pending axes while a separate status lookup can settle an original", () => {
  const { value, sent } = connectedControlClient();
  for (let index = 0; index < 32; index += 1) {
    const key = `step-${index}`, id = BigInt(index + 1).toString(16).padStart(32, "0");
    value.sendControl("step", { requestId: key });
    receiveControl(value, commandReceipt(key, id,
      { application_state: index % 2 ? "applied" : "admitted", journal_state: "pending" }));
  }
  assert.ok(value.controlRequests().every((request) => !request.settled));
  assert.throws(() => value.sendControl("step", { requestId: "step-32" }), /32 control outcomes/);
  assert.equal(sent.length, 32, "a refused mutation must not reach the socket");
  const lookup = value.retryControl("step-0");
  assert.equal(sent.at(-1).command, "status");
  assert.equal(value.requests.size, 32);
  assert.equal(value.retryControl("step-0"), lookup, "repeated pending reads share the single lookup slot");
  assert.equal(sent.length, 33);
  assert.throws(() => value.retryControl("step-1"), /lookup is pending/);
  receiveControl(value, commandReceipt(lookup, "00000000000000000000000000000001",
    { application_state: "applied", journal_state: "committed_volatile", settled: true }));
  assert.equal(value.controlRequests()[0].settled, true);
  value.sendControl("step", { requestId: "step-32" });
  assert.equal(value.requests.size, 32);
  assert.equal(value.requests.has("step-0"), false);
  assert.ok(Array.from({ length: 31 }, (_, index) => value.requests.has(`step-${index + 1}`)).every(Boolean),
    "only the original whose two axes settled may be evicted");
});

test("a durable owner's full history retains intermediate volatile receipts until a failure or durable completion settles them", () => {
  const { value, sent } = connectedControlClient();
  for (let index = 0; index < 32; index += 1) {
    const key = `durable-${index}`, id = BigInt(index + 1).toString(16).padStart(32, "0");
    value.sendControl("step", { requestId: key });
    receiveControl(value, commandReceipt(key, id,
      { application_state: "applied", journal_state: "committed_volatile", applied_tick: index + 1, settled: false }));
  }
  assert.equal(remoteCommandHistoryView(value).unresolvedCount, 32);
  assert.ok(value.controlRequests().every((entry) => !entry.settled && entry.receipt.settled === false));
  assert.throws(() => value.sendControl("step", { requestId: "overflow" }), /32 control outcomes/);
  assert.equal(sent.length, 32);
  const failedLookup = value.retryControl("durable-0");
  const failure = { code: "fsync_failed", message: "Required durable journal write failed", diagnostic_truncated: false };
  const failed = commandReceipt(failedLookup, "00000000000000000000000000000001",
    { application_state: "applied", journal_state: "failed", applied_tick: 1, journal_failure: failure, settled: true });
  receiveControl(value, failed);
  let original = value.controlRequests().find((entry) => entry.requestId === "durable-0");
  assert.equal(original.settled, true);
  assert.equal(original.receipt.application_state, "applied");
  assert.equal(original.receipt.journal_state, "failed");
  assert.equal(original.receipt.applied_tick, 1);
  assert.ok(Object.isFrozen(original.receipt.journal_failure));
  failure.message = "changed outside the client";
  assert.equal(original.receipt.journal_failure.message, "Required durable journal write failed");
  assert.match(describeRemoteCommandReceipt(original.receipt), /Journal failure \(fsync_failed\)/);
  assert.equal(remoteCommandHistoryView(value).unresolvedCount, 31);

  receiveControl(value, commandReceipt("durable-0", "00000000000000000000000000000001",
    { application_state: "applied", journal_state: "committed_volatile", applied_tick: 1, settled: false }));
  original = value.controlRequests().find((entry) => entry.requestId === "durable-0");
  assert.equal(original.receipt.journal_state, "failed", "a delayed volatile observation cannot hide the later failure");
  assert.equal(original.receipt.journal_failure.message, "Required durable journal write failed");
  assert.equal(original.settled, true);
  value.sendControl("step", { requestId: "after-failure" });
  assert.equal(value.requests.has("durable-0"), false);
  assert.equal(value.requests.size, 32);
  assert.ok(Array.from({ length: 31 }, (_, index) => value.requests.has(`durable-${index + 1}`)).every(Boolean));

  const durableLookup = value.retryControl("durable-1");
  receiveControl(value, commandReceipt(durableLookup, "00000000000000000000000000000002",
    { application_state: "applied", journal_state: "durable", applied_tick: 2, settled: true }));
  value.sendControl("step", { requestId: "after-durability" });
  assert.equal(value.requests.has("durable-1"), false);
  assert.equal(value.requests.size, 32);
  assert.ok(Array.from({ length: 30 }, (_, index) => value.requests.has(`durable-${index + 2}`)).every(Boolean),
    "every other intermediate volatile receipt remains retained");
  assert.equal(sent.filter((request) => request.command === "step").length, 34, "only new explicit mutations advance science");
});

test("explicit volatile completion settles and same-state metadata enrichment preserves independent failure diagnosis", () => {
  const { value, sent } = connectedControlClient();
  value.sendControl("step", { requestId: "volatile-owner" });
  receiveControl(value, commandReceipt("volatile-owner", COMMAND,
    { application_state: "applied", journal_state: "committed_volatile", applied_tick: 7, settled: true }));
  assert.equal(value.controlRequests()[0].settled, true);
  assert.equal(remoteCommandHistoryView(value).unresolvedCount, 0);
  assert.doesNotMatch(describeRemoteCommandReceipt(value.controlRequests()[0].receipt), /completion.*pending|completion.*unconfirmed/);
  receiveControl(value, commandReceipt("volatile-owner", COMMAND,
    { application_state: "applied", journal_state: "durable", applied_tick: 7 }));
  assert.equal(value.controlRequests()[0].receipt.settled, true, "known completion survives stronger evidence without metadata");
  receiveControl(value, commandReceipt("volatile-owner", COMMAND, { settled: false }));
  assert.equal(value.controlRequests()[0].receipt.journal_state, "durable");
  assert.equal(value.controlRequests()[0].receipt.settled, true, "late admission cannot erase completion");

  const failedId = "00000000000000000000000000000002";
  value.sendControl("step", { requestId: "application-failed" });
  const diagnosis = { code: "world_step", message: "Retained scientific failure", diagnostic_truncated: true };
  const fields = { application_state: "failed", journal_state: "committed_volatile", application_failure: diagnosis };
  receiveControl(value, commandReceipt("application-failed", failedId, fields));
  assert.equal(value.controlRequests()[1].settled, false, "an old server's volatile outcome does not prove required completion");
  receiveControl(value, commandReceipt("application-failed", failedId, { ...fields, settled: false }));
  assert.equal(value.controlRequests()[1].receipt.settled, false);
  receiveControl(value, commandReceipt("application-failed", failedId, fields));
  assert.equal(value.controlRequests()[1].receipt.settled, false, "missing same-state metadata cannot erase known incompletion");
  receiveControl(value, commandReceipt("application-failed", failedId, { ...fields, settled: true,
    application_failure: { ...diagnosis, message: "A later payload must not replace the retained diagnosis" } }));
  const completed = value.controlRequests()[1].receipt;
  assert.equal(value.controlRequests()[1].settled, true);
  assert.equal(completed.settled, true);
  assert.deepEqual(completed.application_failure, diagnosis);
  assert.ok(Object.isFrozen(completed.application_failure));
  for (const receipt of [fields, { ...fields, settled: false }]) {
    receiveControl(value, commandReceipt("application-failed", failedId, receipt));
    assert.equal(value.controlRequests()[1].receipt, completed);
    assert.equal(value.controlRequests()[1].settled, true);
  }
  assert.equal(sent.length, 2, "metadata reconciliation never submits another mutation");
});

test("legacy terminal journal outcomes settle conservatively while explicit incomplete metadata remains authoritative", () => {
  const { value } = connectedControlClient();
  value.sendControl("step", { requestId: "legacy-volatile" });
  receiveControl(value, commandReceipt("legacy-volatile", COMMAND,
    { application_state: "applied", journal_state: "committed_volatile" }));
  assert.equal(value.controlRequests()[0].settled, false);
  assert.match(describeRemoteCommandReceipt(value.controlRequests()[0].receipt), /Required journal completion is unconfirmed/);
  const lookup = value.retryControl("legacy-volatile");
  receiveControl(value, commandReceipt(lookup, COMMAND, { application_state: "applied", journal_state: "durable" }));
  assert.equal(value.controlRequests()[0].settled, true);
  for (const [index, journal] of ["not_required", "failed"].entries()) {
    const key = `legacy-${journal}`, id = BigInt(index + 2).toString(16).padStart(32, "0");
    value.sendControl("step", { requestId: key });
    receiveControl(value, commandReceipt(key, id, { application_state: "applied", journal_state: journal,
      ...(journal === "failed" ? { journal_failure: { code: "write_failed", message: "No committed write", diagnostic_truncated: false } } : {}) }));
    assert.equal(value.controlRequests().find((entry) => entry.requestId === key).settled, true);
  }
  value.sendControl("step", { requestId: "explicit-incomplete" });
  const explicitId = "00000000000000000000000000000004";
  receiveControl(value, commandReceipt("explicit-incomplete", explicitId,
    { application_state: "applied", journal_state: "durable", settled: false }));
  let entry = value.controlRequests().find((item) => item.requestId === "explicit-incomplete");
  assert.equal(entry.settled, false);
  assert.match(describeRemoteCommandReceipt(entry.receipt), /Owner completion remains pending/);
  receiveControl(value, commandReceipt("explicit-incomplete", explicitId,
    { application_state: "applied", journal_state: "durable" }));
  assert.equal(value.controlRequests().find((item) => item.requestId === "explicit-incomplete").settled, false);
  receiveControl(value, commandReceipt("explicit-incomplete", explicitId,
    { application_state: "applied", journal_state: "durable", settled: true }));
  assert.equal(value.controlRequests().find((item) => item.requestId === "explicit-incomplete").settled, true);

  value.sendControl("step", { requestId: "legacy-after-admission" });
  const admittedId = "00000000000000000000000000000005";
  receiveControl(value, commandReceipt("legacy-after-admission", admittedId, { settled: false }));
  receiveControl(value, commandReceipt("legacy-after-admission", admittedId,
    { application_state: "applied", journal_state: "durable" }));
  assert.equal(value.controlRequests().find((item) => item.requestId === "legacy-after-admission").settled, true,
    "an earlier admitted receipt's false flag cannot override a stronger terminal legacy observation");
});

test("invalid completion metadata cannot resolve a retained command or consume its pending lookup", () => {
  const callbacks = [], { value } = connectedControlClient({ onReceipt: (reply) => callbacks.push(reply) });
  value.sendControl("step", { requestId: "completion-checked" });
  const lookup = value.retryControl("completion-checked");
  for (const fields of [
    ...[null, 0, 1, "true", {}, []].map((settled) => ({ application_state: "applied", journal_state: "durable", settled })),
    { application_state: "admitted", journal_state: "pending", settled: true },
    { application_state: "admitted", journal_state: "not_required", settled: true },
    { application_state: "applied", journal_state: "pending", settled: true },
  ]) assert.throws(() => receiveControl(value, commandReceipt(lookup, COMMAND, fields)), /Invalid remote command receipt outcome/);
  assert.equal(callbacks.length, 0);
  assert.equal(value.statusRequest.received, false);
  assert.equal(value.controlRequests()[0].commandId, null);
  assert.equal(value.controlRequests()[0].settled, false);
  receiveControl(value, commandReceipt(lookup, COMMAND,
    { application_state: "applied", journal_state: "committed_volatile", settled: false }));
  assert.equal(value.statusRequest.received, true);
  assert.equal(value.controlRequests()[0].settled, false);
});

test("lost admission recovery is read-only even after errors, full retention, and missing native history", () => {
  const { value, sent } = connectedControlClient();
  for (let index = 0; index < 32; index += 1) {
    const key = `unknown-${index}`;
    value.sendControl("step", { requestId: key });
    receiveControl(value, { protocol_version: 1, request_id: key, type: "error",
      status: 503, message: "Owner reply unavailable" });
  }
  assert.throws(() => value.sendControl("step", { requestId: "overflow" }), /32 control outcomes/);
  const lookup = value.retryControl("unknown-0");
  assert.deepEqual(sent.at(-1), { protocol_version: 1, request_id: lookup, command: "status_by_key",
    idempotency_key: "unknown-0", session_id: CONTROL_SESSION });
  assert.notEqual(lookup, "unknown-0");
  receiveControl(value, { protocol_version: 1, request_id: lookup, type: "error",
    status: 404, message: "No retained command receipt" });
  assert.equal(value.controlRequests()[0].settled, false);
  assert.deepEqual(value.controlRequests()[0].error,
    { status: 404, message: "No retained command receipt", operation: "status_by_key" });
  assert.throws(() => value.sendControl("step", { requestId: "overflow" }), /32 control outcomes/);
  const secondLookup = value.sendControl("step", { requestId: "unknown-0" });
  assert.notEqual(secondLookup, lookup, "a completed read gets a fresh correlation identity");
  assert.equal(sent.at(-1).command, "status_by_key");
  receiveControl(value, commandReceipt(secondLookup, COMMAND,
    { application_state: "applied", journal_state: "durable" }));
  assert.equal(value.controlRequests()[0].commandId, COMMAND);
  assert.equal(value.controlRequests()[0].error, null);
  assert.equal(value.controlRequests()[0].settled, true);
  const knownLookup = value.retryControl("unknown-0");
  assert.deepEqual(sent.at(-1), { protocol_version: 1, request_id: knownLookup, command: "status", command_id: COMMAND });
  receiveControl(value, { protocol_version: 1, request_id: knownLookup, type: "error",
    status: 404, message: "Volatile receipt history was evicted" });
  assert.equal(value.controlRequests()[0].receipt.journal_state, "durable", "missing later history cannot erase observed evidence");
  assert.equal(value.controlRequests()[0].settled, true);
  assert.equal(sent.filter((message) => message.command === "step").length, 32,
    "neither unknown admission nor known admission may be resubmitted after native history eviction");
  value.sendControl("step", { requestId: "next" });
  assert.equal(value.requests.has("unknown-0"), false);
  assert.equal(value.requests.size, 32);
});

test("terminal application and journal failures settle independently without becoming successful or mutable", () => {
  const { value } = connectedControlClient();
  const failure = { code: "disk_full", message: "Journal could not commit", diagnostic_truncated: false };
  value.sendControl("step", { requestId: "failed-step" });
  receiveControl(value, commandReceipt("failed-step", COMMAND,
    { application_state: "failed", application_failure: { ...failure, code: "step_failed" } }));
  assert.equal(value.controlRequests()[0].settled, false, "a terminal application does not settle a pending journal");
  const lookup = value.retryControl("failed-step");
  const reply = commandReceipt(lookup, COMMAND, { application_state: "failed",
    application_failure: { ...failure, code: "step_failed" }, journal_state: "failed", journal_failure: failure });
  receiveControl(value, reply);
  const history = value.controlRequests(), saved = history[0];
  assert.equal(saved.settled, true);
  assert.equal(saved.receipt.application_state, "failed");
  assert.equal(saved.receipt.journal_state, "failed");
  assert.ok(Object.isFrozen(history) && Object.isFrozen(saved) && Object.isFrozen(saved.receipt)
    && Object.isFrozen(saved.receipt.application_failure) && Object.isFrozen(saved.receipt.journal_failure));
  assert.throws(() => { saved.receipt.journal_failure.message = "success"; }, TypeError);
  reply.receipt.journal_failure.message = "changed outside the client";
  assert.equal(value.controlRequests()[0].receipt.journal_failure.message, "Journal could not commit");
  assert.deepEqual(Object.keys(saved).sort(),
    ["requestId", "command", "sessionId", "speed", "commandId", "receipt", "error", "settled"].sort());
});

test("delayed admission cannot regress settled receipt evidence or callbacks and contradictory outcomes are rejected atomically", () => {
  const callbacks = [], { value } = connectedControlClient({ onReceipt: (reply) => callbacks.push(reply) });
  const boundary = { application_state: "applied", journal_state: "pending",
    applied_tick: 12, control_revision: 7, scientific_revision: 5, config_revision: 3 };
  value.sendControl("step", { requestId: "observed" });
  receiveControl(value, commandReceipt("observed", COMMAND, boundary));
  const lookup = value.retryControl("observed");
  receiveControl(value, commandReceipt(lookup, COMMAND, { ...boundary, journal_state: "durable" }));
  const terminal = value.controlRequests()[0].receipt;
  receiveControl(value, commandReceipt("observed"));
  assert.equal(value.controlRequests()[0].receipt, terminal);
  assert.equal(callbacks.at(-1).receipt, terminal);
  assert.equal(value.controlRequests()[0].settled, true);
  receiveControl(value, commandReceipt("observed", COMMAND, { ...boundary, journal_state: "committed_volatile" }));
  assert.equal(value.controlRequests()[0].receipt, terminal, "a weaker journal observation cannot hide durability");
  const count = callbacks.length;
  for (const fields of [
    { application_state: "rejected", journal_state: "not_required" },
    { ...boundary, journal_state: "not_required" },
    { ...boundary, applied_tick: 13, journal_state: "durable" },
  ]) assert.throws(() => receiveControl(value, commandReceipt("observed", COMMAND, fields)), /retained/);
  assert.equal(callbacks.length, count);
  assert.equal(value.controlRequests()[0].receipt, terminal);
});

test("outcome lookup slots recover across reconnects and refuse changed owner sessions without replay", () => {
  const { value, sent } = connectedControlClient();
  value.sendControl("step", { requestId: "first" });
  value.sendControl("step", { requestId: "second" });
  const lookup = value.retryControl("first");
  assert.equal(value.retryControl("first"), lookup);
  assert.throws(() => value.retryControl("second"), /lookup is pending/);
  const beforeReconnect = sent.length;
  value.disconnect();
  value.state = "connected";
  value.socket = { readyState: 1, bufferedAmount: 0, send: (text) => sent.push(JSON.parse(text)), close() {} };
  const resumedLookup = value.retryControl("second");
  assert.notEqual(resumedLookup, lookup);
  assert.equal(sent.length, beforeReconnect + 1, "reconnect preserves envelopes without sending mutations");
  assert.equal(sent.at(-1).command, "status_by_key");
  assert.equal(value.requests.size, 2);
  value.tracker.sessionId = "72";
  const sourceChanged = sent.length;
  assert.throws(() => value.retryControl("first"), /another owner session/);
  assert.throws(() => receiveControl(value, commandReceipt("first")), /another retained owner session/);
  assert.throws(() => value.sendControl("status_by_key",
    { idempotencyKey: "first", sessionId: CONTROL_SESSION }), /another owner session/);
  assert.equal(sent.length, sourceChanged);
  assert.ok(value.controlRequests().every((request) => !request.settled && request.receipt === null));
});

test("malformed or miscorrelated receipts cannot resolve or evict a retained mutation", () => {
  const { value } = connectedControlClient();
  value.sendControl("step", { requestId: "checked" });
  for (const fields of [
    { application_state: "queued" },
    { journal_state: "disabled" },
    { journal_state: "durable" },
    { application_state: "failed", application_failure: { code: "failure", message: "missing truncation marker" } },
  ]) assert.throws(() => receiveControl(value, commandReceipt("checked", COMMAND, fields)), /Invalid remote command/);
  assert.equal(value.requests.get("checked").received, false);
  assert.equal(value.controlRequests()[0].commandId, null);
  receiveControl(value, commandReceipt("checked"));
  const lookup = value.retryControl("checked");
  assert.throws(() => receiveControl(value, commandReceipt(lookup, "ffffffffffffffffffffffffffffffff")),
    /changed its canonical command identity/);
  value.sendControl("step", { requestId: "distinct" });
  assert.throws(() => receiveControl(value, commandReceipt("distinct")), /different requests/);
  assert.ok(value.controlRequests().every((request) => !request.settled));
  assert.equal(value.controlRequests()[1].commandId, null);
  assert.throws(() => value.sendControl("status_by_key", {
    requestId: "same-key", idempotencyKey: "same-key", sessionId: CONTROL_SESSION,
  }), /Invalid remote recovery key/);
});

test("command history selects an existing original, otherwise the oldest unresolved or latest settled command", () => {
  const empty = remoteCommandHistoryView(null, "missing");
  assert.deepEqual(empty.entries, []);
  assert.equal(empty.selected, null);
  assert.equal(empty.selectedRequestId, null);
  assert.equal(empty.unresolvedCount, 0);
  assert.equal(empty.canRecover, false);
  assert.equal(empty.canRefresh, false);
  assert.ok(Object.isFrozen(empty) && Object.isFrozen(empty.entries));
  const { value, sent } = connectedControlClient();
  value.sendControl("step", { requestId: "earliest" });
  value.sendControl("pause", { requestId: "middle" });
  value.sendControl("step", { requestId: "latest" });
  receiveControl(value, commandReceipt("middle", "00000000000000000000000000000002",
    { application_state: "applied", journal_state: "not_required" }));
  let view = remoteCommandHistoryView(value);
  assert.equal(view.selectedRequestId, "earliest");
  assert.equal(view.unresolvedCount, 2);
  assert.equal(view.canRecover, true);
  assert.equal(view.canRefresh, false, "a lost first receipt recovers by key, not an invented command ID");
  view = remoteCommandHistoryView(value, "middle");
  assert.equal(view.selectedRequestId, "middle", "an explicitly selected settled outcome remains selected");
  assert.equal(view.canRefresh, true);
  assert.ok(Object.isFrozen(view) && Object.isFrozen(view.entries) && Object.isFrozen(view.selected));
  assert.throws(() => { view.selected.commandId = null; }, TypeError);
  assert.throws(() => { view.entries.push(null); }, TypeError);
  receiveControl(value, commandReceipt("earliest", "00000000000000000000000000000001",
    { application_state: "applied", journal_state: "not_required" }));
  assert.equal(remoteCommandHistoryView(value, "missing").selectedRequestId, "latest");
  receiveControl(value, commandReceipt("latest", "00000000000000000000000000000003",
    { application_state: "applied", journal_state: "not_required" }));
  view = remoteCommandHistoryView(value, "missing");
  assert.equal(view.selectedRequestId, "latest");
  assert.equal(view.unresolvedCount, 0);
  assert.equal(sent.length, 3, "selection and fallback must never query or mutate the owner");
});

test("selecting older outcomes and receiving unrelated replies never replaces the selected command's evidence", () => {
  const { value, sent } = connectedControlClient();
  value.sendControl("step", { requestId: "lost-step" });
  value.sendControl("pause", { requestId: "later-pause" });
  const pauseId = "00000000000000000000000000000002";
  receiveControl(value, commandReceipt("later-pause", pauseId,
    { application_state: "applied", journal_state: "not_required", applied_tick: 12 }));
  const pause = remoteCommandHistoryView(value, "later-pause").selected.receipt;
  let view = remoteCommandHistoryView(value, "lost-step");
  assert.equal(view.selected.receipt, null);
  assert.equal(sent.length, 2, "selecting an earlier unresolved command is read-only local navigation");
  const lookup = value.retryControl(view.selectedRequestId);
  view = remoteCommandHistoryView(value, "lost-step");
  assert.equal(view.lookupPending, true);
  assert.equal(view.lookupRequestId, lookup);
  assert.equal(view.lookupTargetRequestId, "lost-step");
  assert.equal(view.canRecover, false);
  assert.equal(view.canRefresh, false);
  view = remoteCommandHistoryView(value, "later-pause");
  assert.equal(view.selected.receipt, pause);
  assert.equal(view.canRecover, false, "another selected command cannot consume a second concurrent lookup slot");
  receiveControl(value, commandReceipt(lookup, COMMAND,
    { application_state: "applied", journal_state: "not_required", applied_tick: 11 }));
  view = remoteCommandHistoryView(value, "later-pause");
  assert.equal(view.selected.receipt, pause, "the older lookup's result must not replace the selected pause");
  assert.equal(view.lookupPending, false);
  assert.equal(view.canRefresh, true);
  const step = remoteCommandHistoryView(value, "lost-step").selected.receipt;
  receiveControl(value, commandReceipt("later-pause", pauseId));
  view = remoteCommandHistoryView(value, "lost-step");
  assert.equal(view.selectedRequestId, "lost-step");
  assert.equal(view.selected.receipt, step, "an unrelated delayed receipt must not alter the selected original");
  assert.equal(view.selected.receipt.applied_tick, 11);
  assert.equal(sent.length, 3);
  assert.equal(sent.at(-1).command, "status_by_key");
  assert.equal(sent.at(-1).idempotency_key, "lost-step");
});

test("full history keeps older unresolved commands selectable and falls back safely when a settled selection is evicted", () => {
  const { value, sent } = connectedControlClient();
  for (let index = 0; index < 32; index += 1) value.sendControl("step", { requestId: `history-${index}` });
  let view = remoteCommandHistoryView(value, "history-0");
  assert.equal(view.entries.length, 32);
  assert.equal(view.unresolvedCount, 32);
  assert.equal(view.canRecover, true);
  const lookup = value.retryControl(view.selectedRequestId);
  receiveControl(value, { protocol_version: 1, request_id: lookup, type: "error",
    status: 404, message: "No retained receipt" });
  view = remoteCommandHistoryView(value, view.selectedRequestId);
  assert.equal(view.selectedRequestId, "history-0");
  assert.equal(view.unresolvedCount, 32);
  assert.equal(view.selected.error.status, 404);
  assert.equal(view.canRecover, true);
  const second = value.retryControl(view.selectedRequestId);
  receiveControl(value, commandReceipt(second, COMMAND, { application_state: "applied", journal_state: "durable" }));
  view = remoteCommandHistoryView(value, "history-0");
  assert.equal(view.unresolvedCount, 31);
  assert.equal(view.selected.settled, true);
  value.sendControl("step", { requestId: "history-32" });
  view = remoteCommandHistoryView(value, "history-0");
  assert.equal(view.entries.length, 32);
  assert.equal(view.unresolvedCount, 32);
  assert.equal(view.selectedRequestId, "history-1", "eviction of one settled entry must select an available unresolved original");
  assert.equal(view.entries.some((entry) => entry.requestId === "history-0"), false);
  assert.equal(sent.filter((request) => request.command === "step").length, 33);
});

test("a connected pending outcome lookup can reconnect and recover the same original through a fresh read only", () => {
  const originalSocket = globalThis.WebSocket, sockets = [];
  class ControlledSocket {
    constructor() {
      this.readyState = 0; this.bufferedAmount = 0;
      this.listeners = new Map(); this.sent = []; sockets.push(this);
    }
    addEventListener(kind, listener) { this.listeners.set(kind, listener); }
    send(text) { this.sent.push(JSON.parse(text)); }
    close() { this.readyState = 3; }
    open() { this.readyState = 1; this.listeners.get("open")(); }
    message(reply) { this.listeners.get("message")({ data: JSON.stringify(reply) }); }
  }
  globalThis.WebSocket = ControlledSocket;
  const value = client();
  const subscribe = () => {
    const socket = sockets.at(-1);
    socket.open();
    const request = socket.sent.find((message) => message.command === "subscribe");
    socket.message({ protocol_version: 1, stream_protocol_version: 2, type: "subscribed",
      request_id: request.request_id, session_id: CONTROL_SESSION, channels: request.channels, max_fps: request.max_fps });
    return socket;
  };
  try {
    value.connect();
    const first = subscribe();
    value.sendControl("step", { requestId: "lost-step-before-reconnect" });
    const selectedRequestId = remoteCommandHistoryView(value).selectedRequestId;
    const retained = value.requests.get(selectedRequestId).text;
    const lostLookup = value.retryControl(selectedRequestId);
    let view = remoteCommandHistoryView(value, selectedRequestId);
    assert.equal(value.state, "connected");
    assert.equal(view.lookupPending, true, "a lost lookup reply can leave a healthy socket connected");
    assert.equal(view.canRecover, false);
    assert.equal(view.lookupTargetRequestId, selectedRequestId);

    value.reconnect();
    assert.equal(first.readyState, 3);
    view = remoteCommandHistoryView(value, selectedRequestId);
    assert.equal(view.selectedRequestId, selectedRequestId);
    assert.equal(view.lookupPending, false, "the old connection no longer owns the read slot");
    assert.equal(view.canRecover, false, "reads wait for the new subscription");
    const second = subscribe();
    view = remoteCommandHistoryView(value, selectedRequestId);
    assert.equal(view.canRecover, true);
    assert.equal(view.selected.sessionId, CONTROL_SESSION);
    assert.equal(value.requests.get(selectedRequestId).text, retained);
    assert.deepEqual(second.sent.map((request) => request.command), ["subscribe"],
      "reconnect never automatically resends a mutation or lookup");
    assert.deepEqual(second.sent[0].resume, { session_id: CONTROL_SESSION, after_event: "0" });

    const freshLookup = value.retryControl(selectedRequestId);
    assert.notEqual(freshLookup, lostLookup);
    assert.deepEqual(second.sent.at(-1), { protocol_version: 1, request_id: freshLookup, command: "status_by_key",
      idempotency_key: selectedRequestId, session_id: CONTROL_SESSION });
    first.message(commandReceipt(lostLookup, COMMAND, { application_state: "applied", journal_state: "durable" }));
    assert.equal(remoteCommandHistoryView(value, selectedRequestId).selected.receipt, null,
      "a late reply from the replaced socket cannot consume the new lookup");
    second.message(commandReceipt(freshLookup, COMMAND, { application_state: "applied", journal_state: "durable" }));
    view = remoteCommandHistoryView(value, selectedRequestId);
    assert.equal(view.selectedRequestId, selectedRequestId);
    assert.equal(view.selected.commandId, COMMAND);
    assert.equal(view.selected.settled, true);
    assert.equal(view.lookupPending, false);
    assert.deepEqual(sockets.flatMap((socket) => socket.sent).map((request) => request.command),
      ["subscribe", "step", "status_by_key", "subscribe", "status_by_key"]);
  } finally {
    value.disconnect();
    globalThis.WebSocket = originalSocket;
  }
});

test("retained command reads remain available on faulted owners and preserve selection across disconnect and source fences", () => {
  const { value } = connectedControlClient();
  value.sendControl("step", { requestId: "retained" });
  receiveControl(value, commandReceipt("retained", COMMAND,
    { application_state: "applied", journal_state: "pending", applied_tick: 12 }));
  value.ownerStatusEnabled = true;
  value.ownerStatusAvailable = true;
  value.ownerStatus = { identity: { sessionId: CONTROL_SESSION }, lifecycle: "stopping", health: "faulted" };
  assert.equal(value.ownerAllowsMutation, false);
  let view = remoteCommandHistoryView(value, "retained");
  assert.equal(view.canRecover, true, "outcome reads do not require mutation authority");
  assert.equal(view.canRefresh, true);
  const receipt = view.selected.receipt;
  value.disconnect();
  view = remoteCommandHistoryView(value, "retained");
  assert.equal(view.selectedRequestId, "retained");
  assert.equal(view.selected.receipt, receipt);
  assert.equal(view.canRecover, false);
  assert.equal(view.canRefresh, false);
  value.state = "connected";
  value.tracker.sessionId = "72";
  view = remoteCommandHistoryView(value, "retained");
  assert.equal(view.selected.receipt, receipt);
  assert.equal(view.canRecover, false);
  assert.equal(view.canRefresh, false);
  value.tracker.sessionId = CONTROL_SESSION;
  view = remoteCommandHistoryView(value, "retained");
  assert.equal(view.canRecover, true);
  assert.equal(view.canRefresh, true);
});

test("subscription acknowledgement gates all binary decoding and refuses unrelated acknowledgements", () => {
  const states = [];
  const value = client({ onState: (state) => states.push(state) });
  value.state = "subscribing";
  value.subscription = { request_id: "subscription-1", channels: ["canvas", "events"], max_fps: 60 };
  value.receive(new Uint8Array([1, 2, 3]).buffer);
  value.receive(JSON.stringify({ protocol_version: 1, type: "stream_rate",
    requested_interval_us: 16667, effective_interval_us: 66668, reason: "slow_consumer" }));
  assert.equal(value.state, "subscribing");
  const ack = { protocol_version: 1, stream_protocol_version: 2, type: "subscribed",
    request_id: "subscription-1", session_id: SESSION, channels: ["canvas", "events"], max_fps: 60 };
  assert.throws(() => value.receive(JSON.stringify({ ...ack, request_id: "unknown" })), /does not match/);
  assert.equal(value.state, "subscribing");
  value.receive(JSON.stringify(ack));
  assert.equal(value.state, "connected");
  assert.equal(states[0].sessionId, SESSION);
  assert.deepEqual(value.resume, { session_id: SESSION, after_event: "0" });
  assert.throws(() => value.receive(new ArrayBuffer(8 * 1024 * 1024 + 1)), /size or type bound/);
});

test("rate observations keep the connection usable and invalid mask changes are atomic", () => {
  const states = [];
  const value = client({ channels: ["canvas"], onState: (state) => states.push(state) });
  value.state = "connected";
  value.receive(JSON.stringify({ protocol_version: 1, type: "stream_rate",
    requested_interval_us: 16667, effective_interval_us: 66668, reason: "slow_consumer" }));
  assert.equal(value.state, "connected");
  assert.equal(states[0].effectiveIntervalUs, 66668);
  assert.equal(states[0].reason, "slow_consumer");
  assert.throws(() => value.setSubscriptions({ channels: ["events"], maxFps: 121 }), /1 through 120/);
  assert.deepEqual(value.channels, ["canvas"]);
  assert.equal(value.maxFps, 60);
  assert.throws(() => value.setSubscriptions({ channels: ["events", "events"] }), /distinct/);
  assert.deepEqual(value.channels, ["canvas"]);
});

test("a paused owner reconnect refreshes unchanged snapshots once while preserving cursors and refusing rewinds", () => {
  const originalSocket = globalThis.WebSocket;
  const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  if (!globalThis.crypto) Object.defineProperty(globalThis, "crypto", { value: webcrypto, configurable: true });
  const sockets = [], delivered = [];
  let decoded;
  class ControlledSocket {
    constructor(url) {
      this.url = url; this.readyState = 0; this.bufferedAmount = 0;
      this.listeners = new Map(); this.sent = []; sockets.push(this);
    }
    addEventListener(kind, listener) { this.listeners.set(kind, listener); }
    send(text) { this.sent.push(JSON.parse(text)); }
    close() { this.readyState = 3; }
    open() { this.readyState = 1; this.listeners.get("open")(); }
    message(data) { this.listeners.get("message")({ data }); }
  }
  globalThis.WebSocket = ControlledSocket;
  const value = new RemoteStreamClient({ url: "wss://lab.example/api/ws/stream", channels: ["canvas", "telemetry", "events"],
    decodeFrame: () => decoded, onFrame: (source) => delivered.push(source) });
  const subscribe = () => {
    const socket = sockets.at(-1);
    socket.open();
    const request = socket.sent.find((message) => message.command === "subscribe");
    socket.message(JSON.stringify({ protocol_version: 1, stream_protocol_version: 2, type: "subscribed",
      request_id: request.request_id, session_id: SESSION, channels: request.channels, max_fps: request.max_fps }));
    return { socket, request };
  };
  const receive = (socket, source) => { decoded = source; socket.message(new ArrayBuffer(1)); };
  try {
    value.connect();
    const first = subscribe();
    assert.equal(first.request.resume, undefined);
    receive(first.socket, frame("canvas", "10"));
    receive(first.socket, frame("telemetry", "10"));
    receive(first.socket, eventPage("0", "1", "1", "1", 1, "10"));
    assert.deepEqual(value.resume, { session_id: SESSION, after_event: "1", after_revision: "10" });
    value.sendControl("pause", { requestId: "retained-pause" });
    const retained = value.requests.get("retained-pause").text;

    value.reconnect();
    const second = subscribe();
    assert.deepEqual(second.request.resume, { session_id: SESSION, after_event: "1" },
      "omit only the snapshot cursor so an unchanged paused host can answer");
    assert.equal(delivered.length, 3, "subscription acknowledgement is not a source observation");
    receive(second.socket, frame("canvas", "10"));
    assert.equal(delivered.length, 4, "one fresh equal-revision canvas must reach the consumer");
    receive(second.socket, frame("canvas", "10"));
    assert.equal(delivered.length, 4, "same-connection duplicate canvas delivery stays suppressed");
    receive(second.socket, frame("telemetry", "10"));
    receive(second.socket, frame("telemetry", "10"));
    assert.equal(delivered.length, 5, "each subscribed snapshot channel has exactly one refresh");
    assert.deepEqual(value.resume, { session_id: SESSION, after_event: "1", after_revision: "10" });
    assert.equal(value.requests.get("retained-pause").text, retained);
    assert.ok(second.socket.sent.every((message) => ["subscribe", "stream_ack"].includes(message.command)),
      "reconnect never replays a retained command");

    value.reconnect();
    const third = subscribe();
    receive(third.socket, frame("canvas", "9"));
    assert.equal(value.state, "error", "refresh permission cannot admit an older publication");
    assert.equal(delivered.length, 5);
    assert.equal(value.tracker.observed.get("canvas").identity.publicationRevision, "10");
    value.reconnect();
    const fourth = subscribe();
    const changed = frame("canvas", "10"); changed.identity.controlRevision = "5";
    receive(fourth.socket, changed);
    assert.equal(value.state, "error", "the same publication cannot acquire a different identity on reconnect");
    assert.equal(delivered.length, 5);
  } finally {
    value.disconnect();
    globalThis.WebSocket = originalSocket;
    if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
    else delete globalThis.crypto;
  }
});

test("a synchronous reconnect during consumption retains the new connection's fresh-snapshot allowance", () => {
  const value = tracker(["canvas"]), source = frame("canvas", "10");
  value.accept(source, () => {});
  value.beginConnection();
  assert.equal(value.accept(source, () => { value.beginConnection(); }), true);
  assert.equal(value.resume().after_revision, undefined,
    "consuming the previous connection's frame cannot spend the replacement connection's refresh");
  assert.equal(value.accept(source, () => {}), true);
  assert.equal(value.accept(source, () => {}), false);
  assert.equal(value.resume().after_revision, "10");
});

function experimentTransport(sessionId = SESSION) {
  const client = new RemoteStreamClient({ url: "wss://lab.example/prefix/api/ws/stream?discard=yes#fragment", decodeFrame() {} });
  client.state = "connected";
  client.tracker.acknowledge(sessionId);
  const prepared = prepareExperimentRequest({ experiment_id: "cohort.example", seeds: ["9007199254740993", "18446744073709551615"],
    variants: [{ variant_id: "control", brain_family: "mlp" }, { variant_id: "treatment", brain_family: "dwraon" }],
    ticks_per_run: 20, max_concurrency: 1, idempotency_key: "one-exact-cohort" });
  return { client, prepared, sessionId: client.tracker.sessionId };
}

const OWNER_SESSION = "18446744073709551615";
function stepBatchFixture(overrides = {}) {
  return { batch_id: "retained-batch-a", session_id: OWNER_SESSION, requested_count: 3,
    attempted_count: 0, admitted_count: 0, applied_count: 0, settled_count: 0,
    phase: "reserved", advancing: false, offset: 0, entries: [], ...overrides };
}
function stepBatchEntry(index, receipt = {}) {
  const command_id = `exact-command-${index}`;
  return { index, command_id, submission_state: "observed", receipt: {
    command_id, application_state: "applied", journal_state: "committed_volatile", applied_tick: index + 1,
    control_revision: index + 1, scientific_revision: index + 1, config_revision: 0, ...receipt,
  } };
}
function stepBatchClient(methods = {}) {
  return { url: "wss://batch.example/api/ws/stream", state: "connected", tracker: { sessionId: OWNER_SESSION }, ...methods };
}
function stepBatchCrypto(t) {
  if (!globalThis.crypto) {
    Object.defineProperty(globalThis, "crypto", { value: webcrypto, configurable: true });
    t.after(() => { delete globalThis.crypto; });
  }
}

test("step batch pages retain independent outcomes and refuse foreign, oversized or fabricated progress", () => {
  const failure = { code: "disk_full", message: "retained journal write failed", diagnostic_truncated: false };
  const stopped = stepBatchFixture({ attempted_count: 1, admitted_count: 1, applied_count: 1, settled_count: 1,
    phase: "stopped", blocking_index: 0, entries: [stepBatchEntry(0, { journal_state: "failed", journal_failure: failure })] });
  const page = validateRemoteStepBatch(stopped, { sessionId: OWNER_SESSION, batchId: stopped.batch_id, count: 3 });
  assert.match(describeRemoteCommandReceipt(page.entries[0].receipt), /application applied; journal failed/);
  assert.match(describeRemoteCommandReceipt(page.entries[0].receipt), /Journal failure \(disk_full\)/);
  stopped.entries[0].receipt.journal_failure.message = "changed input";
  assert.equal(page.entries[0].receipt.journal_failure.message, "retained journal write failed");
  assert.equal(Object.isFrozen(page.entries[0].receipt.journal_failure), true);
  const rejected = stepBatchFixture({ attempted_count: 1, admitted_count: 0, settled_count: 1, phase: "stopped",
    entries: [stepBatchEntry(0, { application_state: "rejected", journal_state: "not_required" })] });
  assert.equal(validateRemoteStepBatch(rejected, { sessionId: OWNER_SESSION }).settled_count, 1,
    "a settled pre-admission rejection need not count as admitted");
  const projectionLost = stepBatchFixture({ attempted_count: 1, admitted_count: 1, phase: "indeterminate",
    entries: [{ index: 0, command_id: "kept-command", submission_state: "observed", observation_error: failure }] });
  assert.equal(validateRemoteStepBatch(projectionLost, { sessionId: OWNER_SESSION }).entries[0].receipt, undefined);
  for (const invalid of [
    { session_id: "1" }, { batch_id: "other-batch" }, { requested_count: 100001 }, { attempted_count: 4 },
    { admitted_count: 2 }, { settled_count: -1 }, { offset: 1.5 }, { phase: "complete" },
    { next_offset: 0 }, { entries: Array(17).fill(stepBatchEntry(0)) },
    { entries: [stepBatchEntry(0, { command_id: "different-command" })] },
    { entries: [{ ...stepBatchEntry(0), submission_state: "not_admitted",
      observation_error: { ...failure, code: "step_submission_not_admitted" } }] },
    { entries: [{ index: 0, command_id: "exact-command-0", submission_state: "not_admitted" }] },
    { entries: [{ ...stepBatchEntry(0), observation_error: { ...failure, message: "é".repeat(257) } }] },
  ]) {
    assert.throws(() => validateRemoteStepBatch({ ...rejected, ...invalid },
      { sessionId: OWNER_SESSION, batchId: rejected.batch_id, count: 3 }));
  }
});

test("step batch settlement metadata preserves pending durability and rejects contradictory completed pages", () => {
  const waiting = stepBatchFixture({ requested_count: 1, attempted_count: 1, admitted_count: 1, applied_count: 1,
    settled_count: 0, phase: "waiting", entries: [stepBatchEntry(0, { settled: false })] });
  const pending = validateRemoteStepBatch(waiting, { sessionId: OWNER_SESSION });
  assert.equal(pending.settled_count, 0);
  assert.equal(pending.entries[0].receipt.journal_state, "committed_volatile");
  assert.equal(pending.entries[0].receipt.settled, false);
  assert.ok(Object.isFrozen(pending.entries[0].receipt));
  assert.match(describeRemoteCommandReceipt(pending.entries[0].receipt), /Owner completion remains pending/);
  for (const fields of [
    ...[null, 0, 1, "true", {}, []].map((settled) => ({ settled })),
    { application_state: "admitted", journal_state: "pending", settled: true },
    { application_state: "admitted", journal_state: "not_required", settled: true },
    { application_state: "applied", journal_state: "pending", settled: true },
  ]) assert.throws(() => validateRemoteStepBatch({ ...waiting, entries: [stepBatchEntry(0, fields)] },
    { sessionId: OWNER_SESSION }), /Step batch receipt/);

  const completed = { ...waiting, phase: "complete", settled_count: 1, entries: [stepBatchEntry(0, { settled: true })] };
  assert.equal(validateRemoteStepBatch(completed, { sessionId: OWNER_SESSION }).entries[0].receipt.settled, true,
    "the native owner may explicitly finish at volatile commitment");
  assert.throws(() => validateRemoteStepBatch({ ...completed, entries: [stepBatchEntry(0, { settled: false })] },
    { sessionId: OWNER_SESSION }), /completion contradicts an unsettled/);
  assert.equal(validateRemoteStepBatch({ ...completed, entries: [stepBatchEntry(0)] },
    { sessionId: OWNER_SESSION }).phase, "complete", "older batch protocols retain their authoritative aggregate contract");
  assert.equal(waiting.entries[0].receipt.settled, false, "validation does not rewrite source progress or receipts");
});

test("step batch HTTP uses exact creation identity, bodyless advance and bounded session-bound receipt pages", async () => {
  const original = globalThis.fetch, calls = [];
  const { client } = experimentTransport(OWNER_SESSION);
  globalThis.fetch = async (url, options) => {
    calls.push({ url: new URL(url), options });
    const offset = options.method === "GET" ? Number(new URL(url).searchParams.get("offset")) : 0;
    return new Response(JSON.stringify(offset === 2 ? stepBatchFixture({ attempted_count: 3, admitted_count: 3,
      applied_count: 3, settled_count: 3, phase: "complete", offset: 2, entries: [stepBatchEntry(2)] }) : stepBatchFixture()));
  };
  try {
    const key = ` ${"\u0001".repeat(1022)} `;
    await client.createStepBatch({ count: 3, idempotency_key: key }, { sessionId: OWNER_SESSION });
    assert.equal(calls[0].url.pathname, "/prefix/api/control/step-batches");
    assert.equal(calls[0].options.method, "POST");
    assert.deepEqual(JSON.parse(calls[0].options.body), { count: 3, idempotency_key: key });
    assert.ok(calls[0].options.body.length > 2048, "valid escaped keys are not silently narrowed to 2 KiB");
    await client.advanceStepBatch("retained-batch-a", { sessionId: OWNER_SESSION });
    assert.equal(calls[1].url.pathname, "/prefix/api/control/step-batches/retained-batch-a/advance");
    assert.equal(calls[1].options.method, "POST");
    assert.equal(calls[1].options.body, undefined);
    const page = await client.stepBatchStatus("retained-batch-a", { sessionId: OWNER_SESSION, offset: 2, limit: 1 });
    assert.equal(page.entries[0].index, 2);
    assert.equal(calls[2].url.search, "?offset=2&limit=1");
    for (const call of calls) assert.equal(call.options.headers["X-ScriptBots-Session"], OWNER_SESSION);
    const count = calls.length;
    for (const invalid of [0, -1, 1.5, 100001, Number.MAX_SAFE_INTEGER + 1, "3"]) {
      await assert.rejects(client.createStepBatch({ count: invalid }, { sessionId: OWNER_SESSION }));
    }
    for (const limit of [0, 17, 1.5]) {
      await assert.rejects(client.stepBatchStatus("retained-batch-a", { sessionId: OWNER_SESSION, limit }));
    }
    await assert.rejects(client.createStepBatch({ count: 3, idempotency_key: "é".repeat(513) }, { sessionId: OWNER_SESSION }));
    assert.equal(calls.length, count, "malformed requests are refused before HTTP admission");
  } finally { globalThis.fetch = original; }
});

test("step batch HTTP enforces actual response bytes and ignores a response from a replaced connection", async () => {
  const original = globalThis.fetch;
  const { client } = experimentTransport(OWNER_SESSION);
  try {
    globalThis.fetch = async () => new Response(" ".repeat(STEP_BATCH_LIMITS.bytes) + JSON.stringify(stepBatchFixture()));
    await assert.rejects(client.stepBatchStatus("retained-batch-a", { sessionId: OWNER_SESSION }), /9 KiB/);
    let resolve;
    globalThis.fetch = () => new Promise((done) => { resolve = done; });
    const pending = client.createStepBatch({ count: 3, idempotency_key: "retained-key" }, { sessionId: OWNER_SESSION });
    client.generation += 1;
    resolve(new Response(JSON.stringify(stepBatchFixture())));
    await assert.rejects(pending, /connection changed/);
  } finally { globalThis.fetch = original; }
});

test("batch creation and advancement uncertainty preserve the exact reservation and stop later science", async (t) => {
  stepBatchCrypto(t);
  const creations = [], snapshots = [];
  let advances = 0, reads = 0, controller;
  const client = stepBatchClient({
    async createStepBatch(request) {
      creations.push(request);
      if (creations.length === 1) throw new Error("creation reply lost");
      return stepBatchFixture();
    },
    async advanceStepBatch(id) {
      assert.equal(id, "retained-batch-a");
      assert.equal(controller.batchId, id, "the ID is retained before any advancing request");
      assert.ok(snapshots.includes(id), "the ID has already been exposed to the UI");
      if (++advances === 1) throw new Error("advancing reply lost");
      return stepBatchFixture({ attempted_count: 3, admitted_count: 3, applied_count: 3, settled_count: 3,
        phase: "complete", offset: 2, entries: [stepBatchEntry(2)] });
    },
    async stepBatchStatus() {
      reads += 1;
      return stepBatchFixture({ attempted_count: 1, admitted_count: 1, phase: reads === 1 ? "indeterminate" : "ready",
        applied_count: reads === 1 ? 0 : 1, settled_count: reads === 1 ? 0 : 1,
        entries: reads === 1 ? [{ index: 0, command_id: "exact-command-0", submission_state: "indeterminate" }] : [stepBatchEntry(0)] });
    },
  });
  controller = new RemoteStepBatchController({ getClient: () => client, onChange: (state) => snapshots.push(state.batchId) });
  assert.equal(await controller.start(3), false);
  assert.equal(controller.batchId, null);
  assert.equal(Object.isFrozen(controller.creation), true);
  await assert.rejects(controller.start(2), /retained batch/);
  assert.equal(await controller.resume(), false);
  assert.equal(creations[0], creations[1], "creation retry reuses the identical frozen request");
  assert.equal(controller.batchId, "retained-batch-a");
  assert.equal(await controller.resume(), true);
  assert.equal(controller.page.phase, "indeterminate");
  assert.equal(advances, 1, "an unknown attempted outcome cannot cause another advance");
  assert.equal(await controller.resume(), true);
  assert.equal(controller.page.phase, "complete");
  assert.equal(advances, 2);
  assert.equal(creations.length, 2, "after the ID is known, recovery only addresses that ID");
});

test("an unknown retained batch and a different owner never create replacement science", async () => {
  let creates = 0, advances = 0, reads = 0;
  const client = stepBatchClient({
    async createStepBatch() { creates += 1; throw new Error("unexpected create"); },
    async advanceStepBatch() { advances += 1; throw new Error("unexpected advance"); },
    async stepBatchStatus(id) { reads += 1; assert.equal(id, "old-process-batch"); throw new RemoteOwnerHttpError(404, "unknown batch"); },
  });
  const controller = new RemoteStepBatchController({ getClient: () => client });
  assert.equal(await controller.open("old-process-batch"), false);
  assert.equal(controller.missing, true);
  assert.equal(controller.batchId, "old-process-batch");
  assert.equal(await controller.resume(), false);
  await assert.rejects(controller.start(3), /retained batch/);
  client.tracker.sessionId = "1";
  assert.equal(await controller.resume(), false);
  assert.equal(reads, 2, "a different owner cannot answer for the retained source");
  assert.equal(creates, 0);
  assert.equal(advances, 0);
  assert.equal(controller.batchId, "old-process-batch");
});

test("proven pre-admission backpressure continues the retained batch and observes the same command identity", async (t) => {
  stepBatchCrypto(t);
  let creates = 0, advances = 0;
  const observations = [];
  const client = stepBatchClient({
    async createStepBatch() { creates += 1; return stepBatchFixture({ requested_count: 1 }); },
    async advanceStepBatch(id) {
      assert.equal(id, "retained-batch-a");
      advances += 1;
      return advances === 1 ? stepBatchFixture({ requested_count: 1, attempted_count: 1, phase: "ready",
        entries: [{ index: 0, command_id: "exact-command-0", submission_state: "not_admitted",
          observation_error: { code: "step_submission_not_admitted", message: "No command was admitted", diagnostic_truncated: false } }] })
        : stepBatchFixture({ requested_count: 1, attempted_count: 1, admitted_count: 1, applied_count: 1,
          settled_count: 1, phase: "complete", entries: [stepBatchEntry(0)] });
    },
  });
  const controller = new RemoteStepBatchController({ getClient: () => client, wait: async () => {},
    onChange: (state) => { if (state.page) observations.push(state.page); } });
  assert.equal(await controller.start(1), true);
  const refused = observations.find((page) => page.entries[0]?.submission_state === "not_admitted");
  assert.ok(refused);
  assert.equal(refused.entries[0].receipt, undefined);
  assert.equal(refused.attempted_count, 1);
  assert.equal(refused.admitted_count, 0);
  assert.equal(controller.page.entries[0].command_id, refused.entries[0].command_id);
  assert.equal(controller.page.attempted_count, 1);
  assert.equal(controller.page.phase, "complete");
  assert.equal(creates, 1);
  assert.equal(advances, 2);
});

test("batch continuation yields at a finite budget and an aborted reservation never auto-advances", async (t) => {
  stepBatchCrypto(t);
  let clock = 0, advances = 0;
  const waiting = stepBatchFixture({ requested_count: 100000, attempted_count: 1, admitted_count: 1,
    phase: "waiting", blocking_index: 0, entries: [stepBatchEntry(0, { application_state: "admitted", journal_state: "pending" })] });
  const client = stepBatchClient({
    async createStepBatch() { return stepBatchFixture({ requested_count: 100000 }); },
    async advanceStepBatch() { advances += 1; return waiting; },
  });
  const controller = new RemoteStepBatchController({ getClient: () => client,
    now: () => clock, wait: async (milliseconds) => { clock += milliseconds; } });
  assert.equal(await controller.start(100000), true);
  assert.equal(advances, 100);
  assert.equal(clock, 10000);
  assert.equal(controller.page.phase, "waiting");
  assert.match(controller.notice, /continuation window finished/);

  let resolve, abort;
  client.createStepBatch = (_request, options) => { abort = options.signal; return new Promise((done) => { resolve = done; }); };
  const paused = new RemoteStepBatchController({ getClient: () => client });
  const pending = paused.start(3);
  paused.pause("Changed presentation source; retain this creation request.");
  assert.equal(abort.aborted, true);
  resolve(stepBatchFixture());
  await pending;
  assert.equal(paused.batchId, "retained-batch-a");
  assert.equal(advances, 100, "a late creation response cannot start science after the view was paused");
});

test("batch authority changes preserve read-only outcome inspection while pausing continuation", async (t) => {
  stepBatchCrypto(t);
  let allowed = false, resolveRead, resolveCreate, readSignal, createSignal, advances = 0;
  const client = stepBatchClient({
    stepBatchStatus(_id, { signal }) {
      readSignal = signal;
      return new Promise((resolve) => { resolveRead = resolve; });
    },
    createStepBatch(_request, { signal }) {
      createSignal = signal;
      return new Promise((resolve) => { resolveCreate = resolve; });
    },
    async advanceStepBatch() { advances += 1; throw new Error("unexpected continuation"); },
  });
  const reader = new RemoteStepBatchController({ getClient: () => client, canAdvance: () => allowed });
  const reading = reader.open("retained-batch-a");
  assert.equal(reader.pending.continuation, false);
  reader.syncAuthority(); // The same production hook called by canvas and owner-status updates.
  assert.equal(readSignal.aborted, false, "a faulted owner or historical view must still allow receipt reads");
  resolveRead(stepBatchFixture({ attempted_count: 1, admitted_count: 1, applied_count: 1, settled_count: 1,
    phase: "stopped", entries: [stepBatchEntry(0, { journal_state: "failed",
      journal_failure: { code: "disk_full", message: "retained fault evidence", diagnostic_truncated: false } })] }));
  assert.equal(await reading, true);
  assert.equal(reader.page.entries[0].receipt.journal_failure.code, "disk_full");

  allowed = true;
  const runner = new RemoteStepBatchController({ getClient: () => client, canAdvance: () => allowed });
  const creating = runner.start(3);
  assert.equal(runner.pending.continuation, true);
  allowed = false;
  runner.syncAuthority();
  assert.equal(createSignal.aborted, true, "live continuation must pause when mutation authority is lost");
  resolveCreate(stepBatchFixture());
  await creating;
  assert.equal(runner.batchId, "retained-batch-a");
  assert.equal(advances, 0);
});

test("clearing a batch reference makes no native calls and cannot discard a pending request", async (t) => {
  stepBatchCrypto(t);
  let creates = 0, reads = 0, advances = 0, resolveRead;
  const client = stepBatchClient({
    async createStepBatch() { creates += 1; throw new Error("reservation reply lost"); },
    async advanceStepBatch() { advances += 1; throw new Error("unexpected advance"); },
    stepBatchStatus(id) {
      reads += 1;
      if (id === "missing-batch") return Promise.reject(new RemoteOwnerHttpError(404, "unknown batch"));
      return new Promise((resolve) => { resolveRead = resolve; });
    },
  });
  const controller = new RemoteStepBatchController({ getClient: () => client });
  assert.equal(await controller.start(3), false);
  assert.ok(controller.creation);
  assert.equal(controller.canStart, false);
  controller.clearReference();
  assert.equal(controller.creation, null);
  assert.equal(controller.source, null);
  assert.equal(controller.canStart, true);
  assert.deepEqual([creates, reads, advances], [1, 0, 0]);

  assert.equal(await controller.open("missing-batch"), false);
  assert.equal(controller.missing, true);
  assert.equal(controller.canStart, false, "404 alone never permits a replacement run");
  controller.clearReference();
  assert.equal(controller.batchId, null);
  assert.equal(controller.page, null);
  assert.equal(controller.source, null);
  assert.equal(controller.creation, null);
  assert.equal(controller.missing, false);
  assert.equal(controller.canStart, true, "only the explicit local clear releases the selection");
  assert.match(controller.notice, /Earlier server work is unchanged/);
  assert.deepEqual([creates, reads, advances], [1, 1, 0], "forgetting a reference is purely local");

  const pending = controller.open("pending-batch");
  assert.throws(() => controller.clearReference(), /pending batch request/);
  assert.equal(controller.batchId, "pending-batch");
  assert.ok(controller.pending);
  resolveRead(stepBatchFixture({ batch_id: "pending-batch" }));
  assert.equal(await pending, true);
  assert.deepEqual([creates, reads, advances], [1, 2, 0]);
});

test("reloading a tab retains the reservation before POST and the batch ID before any steps", async (t) => {
  stepBatchCrypto(t);
  const records = new Map();
  const storage = { getItem: (key) => records.get(key) ?? null,
    setItem: (key, value) => records.set(key, value), removeItem: (key) => records.delete(key) };
  const reservations = [], submissions = [], lookups = [];
  const client = stepBatchClient({
    async createStepBatch(request) {
      reservations.push({ request, saved: JSON.parse([...records.values()][0]) });
      if (reservations.length === 1) throw new Error("reservation reply lost");
      return stepBatchFixture();
    },
    async advanceStepBatch(id) {
      submissions.push({ id, saved: JSON.parse([...records.values()][0]) });
      throw new Error("advance reply lost");
    },
    async stepBatchStatus(id) {
      lookups.push(id);
      return stepBatchFixture({ attempted_count: 3, applied_count: 3, admitted_count: 3,
        settled_count: 3, phase: "complete", entries: [0, 1, 2].map((index) => stepBatchEntry(index)) });
    },
  });
  let live = true;
  const reload = () => new RemoteStepBatchController({ getClient: () => client, canAdvance: () => live,
    tabStorage: () => storage });
  assert.equal(await reload().start(3), false);
  const reserved = reload();
  assert.equal(reserved.canStart, false);
  assert.equal(reservations.length, 1);
  assert.deepEqual(reservations[0].saved.creation, reservations[0].request);
  assert.equal(reservations[0].saved.batchId, null);
  assert.deepEqual(submissions, []);
  assert.deepEqual(lookups, []);
  live = false;
  assert.equal(await reserved.resume(), false);
  live = true;
  assert.equal(await reserved.resume(), false);
  const afterAdvance = reload();
  assert.equal(afterAdvance.page, null);
  assert.equal(afterAdvance.batchId, "retained-batch-a");
  assert.deepEqual(reservations[1].request, reservations[0].request);
  assert.equal(submissions[0].id, "retained-batch-a");
  assert.equal(submissions[0].saved.batchId, submissions[0].id);
  assert.deepEqual(lookups, []);
  client.tracker.sessionId = "42";
  assert.equal(await afterAdvance.resume(), false);
  client.tracker.sessionId = OWNER_SESSION;
  assert.equal(await afterAdvance.refresh(), true);
  assert.equal(afterAdvance.terminal, true);
  afterAdvance.clearReference();
  assert.equal(reload().source, null);
  assert.equal(reservations.length, 2);
  assert.equal(submissions.length, 1);
  assert.deepEqual(lookups, ["retained-batch-a"]);
});

test("unavailable tab storage and private endpoint parameters keep batch inspection usable", async () => {
  let writes = 0;
  const client = stepBatchClient({ async stepBatchStatus() { return stepBatchFixture(); } });
  const inspect = async (tabStorage) => {
    const controller = new RemoteStepBatchController({ getClient: () => client, tabStorage });
    assert.equal(await controller.open("retained-batch-a"), true);
    assert.equal(controller.page.batch_id, "retained-batch-a");
    assert.match(controller.recoveryWarning, /in memory only/);
    return controller;
  };
  await inspect(() => { throw new Error("storage denied"); });
  client.url += "?token=private";
  const storage = { getItem: () => "{", setItem: () => { writes++; }, removeItem() {} };
  const controller = await inspect(() => storage);
  assert.equal(writes, 0);
  controller.clearReference();
  assert.equal(controller.recoveryWarning, "");
});

function ownerStatusFixture(sessionId = OWNER_SESSION, revision = "9007199254740993") {
  return { schema: "scriptbots.remote-owner-status.v1",
    identity: { sessionId, publicationRevision: revision, scientificRevision: "9007199254740993",
      controlRevision: "9007199254740992", configRevision: "18446744073709551615", tick: "9007199254740993" },
    lifecycle: "running", health: "healthy", blocker: null, fault: null,
    playback: { paused: true, speedMultiplier: 1 } };
}
const settleOwnerStatus = () => new Promise((resolve) => setImmediate(resolve));

test("owner fault descriptions retain immutable diagnoses separately from published science", () => {
  const response = ownerStatusFixture();
  response.identity.tick = "3";
  response.health = "faulted";
  response.fault = {
    kind: "scientific",
    tick: "5",
    code: "world_step",
    message: "original sensing failure",
    diagnostic_truncated: false,
  };
  const observed = validateRemoteOwnerStatus(response, OWNER_SESSION);
  assert.match(describeRemoteOwnerFault(observed), /scientific fault at owner boundary tick 5 \(world_step\): original sensing failure/);
  assert.match(describeRemoteOwnerFault(observed), /Published world tick 3\./);
  for (const value of [observed, observed.identity, observed.playback, observed.fault]) {
    assert.equal(Object.isFrozen(value), true);
  }
  assert.throws(() => { observed.fault.tick = "3"; }, TypeError);
  response.identity.tick = "99";
  response.fault.message = "replaced input";
  assert.equal(observed.identity.tick, "3");
  assert.equal(observed.fault.message, "original sensing failure");

  const journal = ownerStatusFixture();
  journal.identity.tick = "3";
  journal.health = "faulted";
  journal.fault = {
    kind: "journal",
    session_id: OWNER_SESSION,
    batch_sequence: "18446744073709551615",
    code: "disk_full",
    message: "bounded diagnostic",
    diagnostic_truncated: true,
  };
  const journalText = describeRemoteOwnerFault(validateRemoteOwnerStatus(journal, OWNER_SESSION));
  assert.ok(journalText.includes(`journal batch ${OWNER_SESSION}/18446744073709551615`));
  assert.match(journalText, /bounded diagnostic \[diagnostic truncated\]/);
  assert.match(journalText, /Published world tick 3\./);
  assert.equal(describeRemoteOwnerFault(validateRemoteOwnerStatus(ownerStatusFixture(), OWNER_SESSION)), "");
});

test("remote owner status preserves exact fault identities and refuses contradictory publications", () => {
  const source = ownerStatusFixture();
  const healthy = validateRemoteOwnerStatus(source, OWNER_SESSION);
  assert.equal(healthy.identity.sessionId, OWNER_SESSION);
  assert.equal(healthy.identity.scientificRevision, "9007199254740993");
  assert.equal(healthy.identity.controlRevision, "9007199254740992");
  assert.equal(healthy.identity.configRevision, "18446744073709551615");
  const fastOwner = structuredClone(source); fastOwner.playback.speedMultiplier = 99;
  assert.equal(validateRemoteOwnerStatus(fastOwner, OWNER_SESSION).playback.speedMultiplier, 99,
    "remote status follows the canonical finite nonnegative speed contract");
  assert.throws(() => validateRemoteOwnerStatus(ownerStatusFixture("18446744073709551616"),
    "18446744073709551616"), /identity/, "owner sessions follow the canonical u64 REST header contract");
  const failed = ownerStatusFixture(OWNER_SESSION, "9007199254740994");
  failed.health = "faulted";
  failed.fault = { kind: "scientific", tick: failed.identity.tick, code: "brain_spawn",
    message: "<img src=x onerror=bad()> real diagnostic", diagnostic_truncated: false };
  const faulted = validateRemoteOwnerStatus(failed, OWNER_SESSION, healthy);
  assert.equal(faulted.fault.tick, failed.identity.tick);
  const journal = structuredClone(failed);
  journal.fault = { kind: "journal", session_id: OWNER_SESSION, batch_sequence: "18446744073709551615",
    code: "disk", message: "🦋".repeat(512), diagnostic_truncated: true };
  assert.equal(validateRemoteOwnerStatus(journal, OWNER_SESSION).fault.batch_sequence, "18446744073709551615");
  for (const change of [
    (status) => { status.schema = "unknown"; },
    (status) => { status.identity.sessionId = "8"; },
    (status) => { status.identity.tick = 9007199254740992; },
    (status) => { status.identity.configRevision = "18446744073709551616"; },
    (status) => { status.playback = { paused: true, speed_multiplier: 1 }; },
    (status) => { status.health = "faulted"; },
    (status) => { status.blocker = "unexpected"; },
  ]) {
    const invalid = structuredClone(source); change(invalid);
    assert.throws(() => validateRemoteOwnerStatus(invalid, OWNER_SESSION));
  }
  for (const change of [
    (status) => { status.fault.message += "🦋"; },
    (status) => { status.fault.session_id = "1"; },
    (status) => { status.fault.session_id = "18446744073709551616"; },
    (status) => { delete status.fault.diagnostic_truncated; },
  ]) {
    const invalid = structuredClone(journal); change(invalid);
    assert.throws(() => validateRemoteOwnerStatus(invalid, OWNER_SESSION));
  }
  assert.throws(() => validateRemoteOwnerStatus(source, OWNER_SESSION, faulted), /regressed/);
  const inventedRecovery = ownerStatusFixture(OWNER_SESSION, "9007199254740995");
  assert.throws(() => validateRemoteOwnerStatus(inventedRecovery, OWNER_SESSION, faulted), /contradicts/);
  const changedPublication = structuredClone(source); changedPublication.playback.paused = false;
  assert.throws(() => validateRemoteOwnerStatus(changedPublication, OWNER_SESSION, healthy), /contradicts/);
  const stopped = structuredClone(healthy); stopped.lifecycle = "stopped";
  assert.throws(() => validateRemoteOwnerStatus(inventedRecovery, OWNER_SESSION, stopped), /contradicts/);
});

test("scientific faults retain their exact owner tick beside an older coherent publication", () => {
  const source = ownerStatusFixture(OWNER_SESSION, "1");
  source.identity.tick = "0";
  source.identity.scientificRevision = "0";
  source.identity.controlRevision = "0";
  source.identity.configRevision = "0";
  const healthy = validateRemoteOwnerStatus(source, OWNER_SESSION);
  for (const tick of ["0", "1", "9007199254740993", "18446744073709551615"]) {
    const response = structuredClone(source);
    response.identity.publicationRevision = "2";
    response.health = "faulted";
    response.fault = { kind: "scientific", tick, code: "world_step",
      message: "sense provider failed during an incomplete transition", diagnostic_truncated: false };
    const faulted = validateRemoteOwnerStatus(response, OWNER_SESSION, healthy);
    assert.equal(faulted.fault.tick, tick, "the fault observation must not be clamped to the retained canvas");
    assert.equal(faulted.identity.tick, "0", "fault reporting must not invent a completed scientific publication");
    assert.equal(faulted.identity.scientificRevision, "0");

    const contradiction = structuredClone(response);
    contradiction.fault.tick = tick === "0" ? "1" : "0";
    assert.throws(() => validateRemoteOwnerStatus(contradiction, OWNER_SESSION, faulted), /contradicts/,
      "the same publication still cannot change its fault identity");
    const recovery = structuredClone(source);
    recovery.identity.publicationRevision = "3";
    assert.throws(() => validateRemoteOwnerStatus(recovery, OWNER_SESSION, faulted), /contradicts/);
    const stopped = structuredClone(response);
    stopped.identity.publicationRevision = "3";
    stopped.lifecycle = "stopped";
    const terminal = validateRemoteOwnerStatus(stopped, OWNER_SESSION, faulted);
    assert.equal(terminal.fault.tick, tick);
    const restarted = structuredClone(response);
    restarted.identity.publicationRevision = "4";
    assert.throws(() => validateRemoteOwnerStatus(restarted, OWNER_SESSION, terminal), /contradicts/);

    response.fault.tick = "malformed";
    assert.equal(faulted.fault.tick, tick, "validated fault evidence is independent of the response object");
  }
  for (const tick of [undefined, null, 1, -1, "", "01", "-1", "1.5", "1e3", "18446744073709551616"]) {
    const invalid = structuredClone(source);
    invalid.identity.publicationRevision = "2";
    invalid.health = "faulted";
    invalid.fault = { kind: "scientific", tick, code: "world_step", message: "invalid tick", diagnostic_truncated: false };
    assert.throws(() => validateRemoteOwnerStatus(invalid, OWNER_SESSION, healthy), /identity/,
      "separate owner and canvas ticks must still be exact canonical u64 values");
  }
});

test("remote polling exposes a sparse-publication fault and disables controls without losing the canvas", async () => {
  const original = globalThis.fetch, requests = [], observations = [];
  const { client, sessionId } = experimentTransport(OWNER_SESSION);
  client.onOwnerStatus = (current) => observations.push({ available: current.ownerStatusAvailable,
    mutation: current.ownerAllowsMutation, status: current.ownerStatus });
  globalThis.fetch = (_url, options) => new Promise((resolve) => requests.push({ options, resolve }));
  try {
    const source = ownerStatusFixture(sessionId, "1");
    source.identity.tick = "0";
    source.identity.scientificRevision = "0";
    source.identity.controlRevision = "0";
    source.identity.configRevision = "0";
    const consumed = [];
    for (const channel of ["canvas", "telemetry"]) {
      const publication = frame(channel, "1", { tick: 0 });
      publication.identity = { ...source.identity };
      assert.equal(client.tracker.accept(publication, (value) => { consumed.push(value.kind); }), true);
    }
    const narrative = eventPage(0, 1, 1, 1, 1, "1");
    narrative.identity = { ...source.identity };
    narrative.payload.events[0].record.tick = "0";
    narrative.payload.events[0].record.window = ["0", "0"];
    assert.equal(client.tracker.accept(narrative, (value) => { consumed.push(value.kind); }), true);
    assert.deepEqual(consumed, ["canvas", "telemetry", "events"]);
    const retained = new Map(client.tracker.observed);
    const retainedMetadata = structuredClone([...client.tracker.observed]);
    assert.equal(retained.size, 3);
    assert.equal(client.tracker.afterEvent, 1n);
    client.setOwnerStatusEnabled(true);
    requests[0].resolve(new Response(JSON.stringify(source)));
    await settleOwnerStatus();
    assert.equal(client.ownerAllowsMutation, true);
    const resume = structuredClone(client.resume);
    assert.deepEqual(resume, { session_id: sessionId, after_event: "1", after_revision: "1" });

    const pending = client.pollOwnerStatus(client.ownerStatusAttemptAt + 1000);
    assert.equal(requests.length, 2);
    const fault = structuredClone(source);
    fault.identity.publicationRevision = "2";
    fault.health = "faulted";
    fault.fault = { kind: "scientific", tick: "1", code: "world_step",
      message: "sense provider failed after the last published canvas", diagnostic_truncated: false };
    requests[1].resolve(new Response(JSON.stringify(fault)));
    assert.equal(await pending, true, "a genuine later fault is an available owner observation");
    assert.equal(client.ownerStatusAvailable, true);
    assert.equal(client.ownerStatusError, null);
    assert.equal(client.ownerStatus.health, "faulted");
    assert.equal(client.ownerStatus.fault.tick, "1");
    assert.equal(client.ownerStatus.identity.tick, "0");
    assert.equal(client.ownerAllowsMutation, false);
    assert.equal(client.state, "connected", "the scientific fault is distinct from a transport failure");
    assert.deepEqual(client.resume, resume, "health reads cannot advance the retained canvas or event cursors");
    assert.equal(client.tracker.afterEvent, 1n);
    for (const [channel, metadata] of retained) {
      assert.equal(client.tracker.observed.get(channel), metadata,
        "health polling cannot replace a consumed channel's publication metadata");
    }
    assert.deepEqual([...client.tracker.observed], retainedMetadata,
      "the consumed scientific/event metadata remains unchanged");
    assert.deepEqual(observations.at(-1), { available: true, mutation: false, status: client.ownerStatus });
  } finally { client.setOwnerStatusEnabled(false); globalThis.fetch = original; }
});

test("remote health polling requires a fresh session-bound read and never overlaps requests", async () => {
  const original = globalThis.fetch, requests = [];
  const { client, sessionId } = experimentTransport(OWNER_SESSION);
  globalThis.fetch = (url, options) => new Promise((resolve) => requests.push({ url: new URL(url), options, resolve }));
  try {
    assert.equal(client.ownerAllowsMutation, false, "a connected socket is not an owner-health observation");
    client.setOwnerStatusEnabled(true);
    for (let count = 0; count < 8; count += 1) assert.equal(await client.pollOwnerStatus(), false);
    assert.equal(requests.length, 1);
    assert.equal(requests[0].url.href, "https://lab.example/prefix/api/v1/owner/status");
    assert.equal(requests[0].options.headers["X-ScriptBots-Session"], sessionId);
    assert.equal(requests[0].options.method, "GET");
    requests[0].resolve(new Response(JSON.stringify(ownerStatusFixture(sessionId))));
    await settleOwnerStatus();
    assert.equal(client.ownerAllowsMutation, true);
    // A recorded fractional performance clock can cross a double-precision
    // boundary on its next exact deadline; elapsed-time subtraction loses it.
    client.ownerStatusAttemptAt = 2010.2;
    assert.equal(await client.pollOwnerStatus(client.ownerStatusAttemptAt + 999), false);
    const pending = client.pollOwnerStatus(client.ownerStatusAttemptAt + 1000);
    assert.equal(requests.length, 2);
    const fault = ownerStatusFixture(sessionId, "9007199254740994");
    fault.health = "faulted";
    fault.fault = { kind: "scientific", tick: fault.identity.tick, code: "brain_spawn",
      message: "completed fault", diagnostic_truncated: false };
    requests[1].resolve(new Response(JSON.stringify(fault)));
    assert.equal(await pending, true);
    assert.equal(client.ownerAllowsMutation, false);
    assert.equal(client.state, "connected", "owner fault does not impersonate a socket failure");
    const older = client.pollOwnerStatus(client.ownerStatusAttemptAt + 1000);
    requests[2].resolve(new Response(JSON.stringify(ownerStatusFixture(sessionId))));
    assert.equal(await older, false);
    assert.equal(client.ownerStatus.fault.code, "brain_spawn");
    assert.equal(client.ownerStatusAvailable, false);
    assert.match(client.ownerStatusError, /regressed/);
  } finally { client.setOwnerStatusEnabled(false); globalThis.fetch = original; }
});

test("hidden and replaced sources abort health reads without admitting stale responses", async () => {
  const original = globalThis.fetch, requests = [];
  const { client, sessionId } = experimentTransport(OWNER_SESSION);
  globalThis.fetch = (url, options) => new Promise((resolve) => requests.push({ options, resolve }));
  try {
    client.setOwnerStatusEnabled(true);
    client.setOwnerStatusEnabled(false);
    assert.equal(requests[0].options.signal.aborted, true);
    client.setOwnerStatusEnabled(true);
    assert.equal(requests.length, 1, "wait for even a cancelled request to release its single slot");
    requests[0].resolve(new Response(JSON.stringify(ownerStatusFixture(sessionId))));
    await settleOwnerStatus();
    assert.equal(client.ownerStatus, null);
    assert.equal(client.ownerAllowsMutation, false);
    const visible = client.pollOwnerStatus();
    assert.equal(requests.length, 2);
    client.disconnect();
    assert.equal(requests[1].options.signal.aborted, true);
    client.tracker = new RemoteFrameTracker(["canvas"]);
    client.tracker.acknowledge("9007199254740993");
    client.report("connected", "replacement subscription fixture");
    client.setOwnerStatusEnabled(true);
    requests[1].resolve(new Response(JSON.stringify(ownerStatusFixture(sessionId))));
    assert.equal(await visible, false);
    assert.equal(client.ownerStatus, null);
    const fresh = client.pollOwnerStatus();
    assert.equal(requests[2].options.headers["X-ScriptBots-Session"], "9007199254740993");
    const blocked = ownerStatusFixture("9007199254740993"); blocked.health = "blocked"; blocked.blocker = "JournalFull";
    requests[2].resolve(new Response(JSON.stringify(blocked)));
    assert.equal(await fresh, true);
    assert.equal(client.ownerAllowsMutation, true, "a running blocked owner retains canonical control handling");
  } finally { client.setOwnerStatusEnabled(false); globalThis.fetch = original; }
});

test("status HTTP failure and actual byte overflow disable writes while retaining readable observations", async () => {
  const original = globalThis.fetch;
  const { client, sessionId } = experimentTransport(OWNER_SESSION);
  try {
    globalThis.fetch = async () => new Response(JSON.stringify(ownerStatusFixture(sessionId)));
    client.setOwnerStatusEnabled(true);
    await settleOwnerStatus();
    const observed = structuredClone(client.ownerStatus);
    assert.equal(client.ownerAllowsMutation, true);
    globalThis.fetch = async () => new Response('{"message":"temporarily unavailable"}', { status: 503 });
    assert.equal(await client.pollOwnerStatus(client.ownerStatusAttemptAt + 1000), false);
    assert.deepEqual(client.ownerStatus, observed);
    assert.equal(client.ownerAllowsMutation, false);
    globalThis.fetch = async () => new Response(JSON.stringify(ownerStatusFixture(sessionId)));
    assert.equal(await client.pollOwnerStatus(client.ownerStatusAttemptAt + 1000), true);
    assert.equal(client.ownerAllowsMutation, true, "a new successful observation restores a running owner's controls");
    assert.equal(client.ownerStatusError, null);
    let cancelled = false;
    globalThis.fetch = async () => new Response(new ReadableStream({
      start(controller) { controller.enqueue(new Uint8Array(16 * 1024 + 1)); },
      cancel() { cancelled = true; },
    }));
    assert.equal(await client.pollOwnerStatus(client.ownerStatusAttemptAt + 1000), false);
    assert.equal(cancelled, true);
    assert.match(client.ownerStatusError, /exceeds/);
    const stopping = ownerStatusFixture(sessionId, "9007199254740994"); stopping.lifecycle = "stopping";
    globalThis.fetch = async () => new Response(JSON.stringify(stopping));
    assert.equal(await client.pollOwnerStatus(client.ownerStatusAttemptAt + 1000), true);
    assert.equal(client.ownerStatusAvailable, true);
    assert.equal(client.ownerAllowsMutation, false);
  } finally { client.setOwnerStatusEnabled(false); globalThis.fetch = original; }
});

test("run artifact inventory uses the same host and session, exact seeds and a bounded nonpaginated response", async () => {
  const original = globalThis.fetch, requests = [];
  try {
    const { client, sessionId } = experimentTransport();
    globalThis.fetch = async (url, options) => {
      requests.push({ url: new URL(url), ...options });
      return new Response('{"run":{"seed":18446744073709551615}}');
    };
    const result = await client.experimentRunArtifacts("cohort.example", "cohort.example-control-seed18446744073709551615", { sessionId });
    assert.equal(result.run.seed, "18446744073709551615");
    assert.equal(requests[0].url.href, "https://lab.example/prefix/api/v1/experiments/cohort.example/runs/cohort.example-control-seed18446744073709551615/artifacts");
    assert.equal(requests[0].method, "GET");
    assert.equal(requests[0].headers["X-ScriptBots-Session"], sessionId);
    assert.equal(requests[0].credentials, "omit");
    for (const id of ["../other", ".", "run/escape", "x".repeat(129)]) {
      await assert.rejects(client.experimentRunArtifacts("cohort.example", id, { sessionId }));
    }
    await assert.rejects(client.experimentRunArtifacts("cohort.example", "run", { sessionId: "7" }));
    assert.equal(requests.length, 1);
    let cancelled = false;
    globalThis.fetch = async () => new Response(new ReadableStream({
      start(controller) { controller.enqueue(new Uint8Array(EXPERIMENT_LIMITS.maxRunArtifactResponseBytes + 1)); },
      cancel() { cancelled = true; },
    }));
    await assert.rejects(client.experimentRunArtifacts("cohort.example", "run", { sessionId }), /exceeds/);
    assert.equal(cancelled, true);
    let respond;
    globalThis.fetch = () => new Promise((resolve) => { respond = resolve; });
    const pending = client.experimentRunArtifacts("cohort.example", "run", { sessionId });
    client.generation += 1;
    respond(new Response("{}"));
    await assert.rejects(pending, /connection changed/);
  } finally { globalThis.fetch = original; }
});

test("artifact download refuses moved paths or changed provenance before requesting any file bytes", async () => {
  const original = globalThis.fetch;
  const expected = { artifact_id: "experiment-" + "a".repeat(64), filename: "bundle_manifest.json",
    content_type: "application/json", byte_size: 2, checksum_blake3: "b".repeat(64),
    checksum_sha256: "c".repeat(64), created_at_utc: "1794456010",
    relative_path: "experiments/cohort/run/bundle_manifest.json" };
  try {
    const { client, sessionId } = experimentTransport();
    for (const [key, value] of [["relative_path", "experiments/foreign/run/bundle_manifest.json"],
      ["content_type", "application/octet-stream"], ["created_at_utc", "1794456011"]]) {
      let calls = 0;
      globalThis.fetch = async () => {
        calls += 1;
        return new Response(JSON.stringify({ ...expected, [key]: value }));
      };
      await assert.rejects(client.downloadArtifact(expected.artifact_id, { sessionId, expected }), /differs/);
      assert.equal(calls, 1, "a replaced catalog entry cannot trigger a byte download");
    }
  } finally { globalThis.fetch = original; }
});

test("failed-run retry transport binds the generation and source before HTTP admission", async () => {
  const original = globalThis.fetch, requests = [];
  globalThis.fetch = async (url, options) => {
    requests.push({ url: new URL(url), ...options });
    return new Response('{}');
  };
  try {
    const { client, sessionId } = experimentTransport();
    await client.retryFailedExperiment("cohort.example", { sessionId, expectedGeneration: Number.MAX_SAFE_INTEGER });
    const request = requests[0];
    assert.equal(request.method, "POST");
    assert.equal(request.url.pathname, "/prefix/api/v1/experiments/cohort.example/retry-failed");
    assert.equal(request.headers["X-ScriptBots-Session"], sessionId);
    assert.deepEqual(JSON.parse(request.body), { expected_generation: Number.MAX_SAFE_INTEGER });
    assert.equal(request.credentials, "omit");
    for (const expectedGeneration of [undefined, -1, 1.5, "7", NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      await assert.rejects(client.retryFailedExperiment("cohort.example", { sessionId, expectedGeneration }), /exact observed/);
    }
    await assert.rejects(client.retryFailedExperiment("cohort.example", { sessionId: "7", expectedGeneration: 9 }));
    await assert.rejects(client.retryFailedExperiment("../other", { sessionId, expectedGeneration: 9 }));
    assert.equal(requests.length, 1, "invalid generation, source or path must never reach HTTP");
  } finally { globalThis.fetch = original; }
});

test("cohort transport keeps exact u64 seeds, unchanged retry bytes and the full u128 negotiated source", async () => {
  const original = globalThis.fetch, requests = [];
  globalThis.fetch = async (url, options) => {
    requests.push({ url: new URL(url), ...options });
    return new Response('{"seed":18446744073709551615,"matched_seeds":[9007199254740993,18446744073709551615]}');
  };
  try {
    const { client, prepared, sessionId } = experimentTransport();
    const create = { sessionId, ...prepared };
    const result = await client.createExperiment(create);
    assert.equal(result.seed, "18446744073709551615");
    assert.deepEqual(result.matched_seeds, ["9007199254740993", "18446744073709551615"]);
    await client.createExperiment(create);
    await client.listExperiments({ sessionId, cursor: "12" });
    await client.experimentStatus("cohort.example", { sessionId });
    await client.cancelExperiment("cohort.example", { sessionId });
    await client.resumeExperiment("cohort.example", { sessionId });
    await client.compareExperiment("cohort.example", { sessionId, baseline_variant: "control",
      challenger_variant: "treatment", metric: "alive_agents" });
    assert.deepEqual(requests.map((request) => [request.method, request.url.pathname]), [
      ["POST", "/prefix/api/v1/experiments"], ["POST", "/prefix/api/v1/experiments"],
      ["GET", "/prefix/api/v1/experiments"], ["GET", "/prefix/api/v1/experiments/cohort.example"],
      ["POST", "/prefix/api/v1/experiments/cohort.example/cancel"],
      ["POST", "/prefix/api/v1/experiments/cohort.example/resume"],
      ["POST", "/prefix/api/v1/experiments/cohort.example/compare"],
    ]);
    for (const request of requests) {
      assert.equal(request.headers["X-ScriptBots-Session"], sessionId);
      assert.equal(request.credentials, "omit");
      assert.equal(request.cache, "no-store");
      assert.equal(request.url.protocol, "https:");
      assert.equal(request.url.hash, "");
    }
    assert.equal(requests[0].body, requests[1].body);
    assert.match(requests[0].body, /"seeds":\[9007199254740993,18446744073709551615\]/);
    assert.ok(!requests[0].body.includes("sessionId"));
    assert.equal(requests[2].url.search, "?limit=12&cursor=12");
    assert.deepEqual(JSON.parse(requests[6].body), { baseline_variant: "control", challenger_variant: "treatment", metric: "alive_agents" });
    const count = requests.length;
    await assert.rejects(client.createExperiment({ ...create, body: create.body + " " }), /payload changed/);
    await assert.rejects(client.experimentStatus("../escape", { sessionId }));
    await assert.rejects(client.experimentStatus(".", { sessionId }));
    await assert.rejects(client.cancelExperiment("cohort.example", { sessionId: "7" }));
    await assert.rejects(client.listExperiments({ sessionId, cursor: "01" }));
    await assert.rejects(client.compareExperiment("cohort.example", { sessionId, baseline_variant: "control", challenger_variant: "control", metric: "alive_agents" }));
    await assert.rejects(client.compareExperiment("cohort.example", { sessionId, baseline_variant: "control", challenger_variant: "treatment", metric: "invented" }));
    assert.equal(requests.length, count, "invalid source or payload never reached HTTP");
  } finally { globalThis.fetch = original; }
});

test("owner HTTP requests refuse malformed or overflowing sessions before any fetch", async () => {
  const original = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => { calls += 1; return new Response("{}"); };
  try {
    const { client, sessionId } = experimentTransport();
    assert.equal(sessionId, SESSION);
    for (const invalid of [undefined, null, 7, "", "-1", "+1", "01", "1.0", "1e2", " 1", "1 ",
      "340282366920938463463374607431768211456", "9".repeat(40)]) {
      // Pin the same invalid value on both sides so equality alone cannot stand
      // in for the declared u128 grammar and range check.
      client.tracker.sessionId = invalid;
      await assert.rejects(client.listExperiments({ sessionId: invalid }), /identity/);
      await assert.rejects(client.listCheckpoints({ sessionId: invalid }), /identity/);
      await assert.rejects(client.interventionStatus(COMMAND, { sessionId: invalid }), /identity/);
      await assert.rejects(client.downloadArtifact("artifact-one", { sessionId: invalid }), /identity/);
    }
    client.tracker.sessionId = sessionId;
    for (const cursor of ["18446744073709551616", SESSION]) {
      await assert.rejects(client.listExperiments({ sessionId, cursor }), /identity/);
    }
    assert.equal(calls, 0, "invalid source or a widened u64 cursor cannot reach HTTP");
  } finally { globalThis.fetch = original; }
});

test("experiment HTTP refusal differs from uncertain admission and cannot survive a changed connection", async () => {
  const original = globalThis.fetch;
  try {
    const { client, prepared, sessionId } = experimentTransport();
    for (const code of [400, 409, 422, 503, 500]) {
      globalThis.fetch = async () => new Response('{"message":"bounded refusal"}', { status: code });
      await assert.rejects(client.createExperiment({ sessionId, ...prepared }), (error) => {
        assert.equal(error.status, code);
        assert.equal(error.admission, [400, 409, 422].includes(code) ? "not_admitted" : "unknown");
        return true;
      });
    }
    let respond;
    globalThis.fetch = () => new Promise((resolve) => { respond = resolve; });
    const pending = client.createExperiment({ sessionId, ...prepared });
    client.generation += 1;
    respond(new Response("{}"));
    await assert.rejects(pending, /connection changed.*unconfirmed/);
  } finally { globalThis.fetch = original; }
});

test("experiment owner responses are actually byte-bounded and ambiguous numeric identities refuse", async () => {
  const original = globalThis.fetch;
  try {
    const { client, sessionId } = experimentTransport();
    let cancelled = false;
    globalThis.fetch = async () => new Response(new ReadableStream({
      start(controller) { controller.enqueue(new Uint8Array(EXPERIMENT_LIMITS.maxResponseBytes + 1)); },
      cancel() { cancelled = true; },
    }));
    await assert.rejects(client.experimentStatus("cohort.example", { sessionId }), /exceeds/);
    assert.equal(cancelled, true);
    for (const wire of ['{"seed":18446744073709551616}', '{"generation":9007199254740993}',
      '{"seed":1,"seed":2}', '{"seed":1e2}']) {
      globalThis.fetch = async () => new Response(wire);
      await assert.rejects(client.experimentStatus("cohort.example", { sessionId }));
    }
  } finally { globalThis.fetch = original; }
});
