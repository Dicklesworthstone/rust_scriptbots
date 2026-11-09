// Consumer-state tests. The Rust codec tests exercise actual HostCore bytes;
// these tests exercise the browser's cursor commits and request correlation.
import assert from "node:assert/strict";
import test from "node:test";
import { RemoteFrameTracker, RemoteStreamClient, consumeRemoteFrame } from "../remote.js";
import { EnvironmentRaster, environmentDescription } from "../renderers/environment.js";

const SESSION = "340282366920938463463374607431768211455";
const COMMAND = "1234567890abcdef1234567890abcdef";

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
  // Exercise the actual failed-send admission path: a previously connected
  // browser may lose its socket before its close callback is delivered.
  value.state = "connected";
  assert.throws(() => value.sendControl("step", { requestId }), (error) =>
    /socket is not open/.test(error.message) && error.requestId === requestId);
}

test("failed send retains exactly one retry identity, refuses conflicting reuse, and bounds history", () => {
  const value = client();
  retainUnconfirmed(value, "step-1");
  const retained = value.requests.get("step-1").text;
  assert.deepEqual(JSON.parse(retained), { protocol_version: 1, request_id: "step-1", command: "step" });
  retainUnconfirmed(value, "step-1");
  assert.equal(value.requests.size, 1);
  assert.equal(value.requests.get("step-1").text, retained);
  assert.throws(() => value.sendControl("resume", { requestId: "step-1" }), /different command/);
  assert.throws(() => value.retryControl("step-1"), /socket is not open/);
  assert.equal(value.requests.get("step-1").text, retained);
  for (let index = 2; index <= 32; index += 1) retainUnconfirmed(value, `step-${index}`);
  assert.equal(value.requests.size, 32);
  assert.throws(() => value.sendControl("step", { requestId: "step-33" }), /32 control acknowledgements/);
  value.receive(JSON.stringify({ protocol_version: 1, request_id: "step-1", type: "receipt",
    receipt: { command_id: COMMAND, application_state: "applied", journal_state: "disabled" } }));
  retainUnconfirmed(value, "step-33");
  assert.equal(value.requests.size, 32);
  assert.equal(value.requests.has("step-1"), false);
  assert.throws(() => value.retryControl("step-1"), /no longer retained/);
});

test("receipt correlation preserves command identity and exposes server errors to the UI", () => {
  const receipts = [];
  const value = client({ onReceipt: (reply) => receipts.push(reply) });
  retainUnconfirmed(value, "step-1");
  const receipt = { protocol_version: 1, request_id: "step-1", type: "receipt",
    receipt: { command_id: COMMAND, application_state: "queued", journal_state: "pending" } };
  value.receive(JSON.stringify(receipt));
  assert.deepEqual(receipts[0], receipt);
  const changed = structuredClone(receipt);
  changed.receipt.command_id = "ffffffffffffffffffffffffffffffff";
  assert.throws(() => value.receive(JSON.stringify(changed)), /changed its canonical command identity/);
  assert.equal(receipts.length, 1);
  const error = { protocol_version: 1, request_id: "step-1", type: "error", status: 409, message: "Request cannot be re-admitted" };
  value.receive(JSON.stringify(error));
  assert.deepEqual(receipts[1], error);
  assert.throws(() => value.receive(JSON.stringify({ ...receipt, request_id: "unknown" })), /Uncorrelated/);
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
