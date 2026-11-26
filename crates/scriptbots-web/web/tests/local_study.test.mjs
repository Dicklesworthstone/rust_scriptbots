import assert from "node:assert/strict";
import test from "node:test";
import { webcrypto } from "node:crypto";
import { captureLocalStudy, runLocalStudy } from "../local-study.js";

if (!globalThis.crypto) Object.defineProperty(globalThis, "crypto", { value: webcrypto });

const action = { kind: "drought", region: { shape: "all" }, ticks: 5, growth_scale: 0.1 };

function fixture() {
  const bytes = Uint8Array.of(9, 2);
  let checkpointCalls = 0;
  const released = [];
  function owner() {
    let tick = 71, control = 3, staged = false, committed = false, energy = 1;
    const identity = () => ({ sessionId: "340282366920938463463374607431768211455", tick: String(tick),
      scientificRevision: String(tick), controlRevision: String(control), configRevision: "0",
      publicationRevision: String(tick + control) });
    const receipt = () => ({ schema: "scriptbots.browser-intervention-receipt.v1", commandId: "d".repeat(32),
      applicationState: "applied", journalState: committed ? "committed_volatile" : "pending",
      admissionSequence: "4", applicationBoundary: { tick: "71", scientificRevision: "71", controlRevision: "4", configRevision: "0" },
      source: { ...identity(), epoch: "2", paused: true }, ownerError: null, message: null, messageTruncated: false });
    return {
      frame: () => ({ identity: identity(), playback: { paused: true }, snapshot: { epoch: "2", tick,
        summary: { agentCount: 12, averageEnergy: energy, averageHealth: 0.5 } } }),
      ownerStatus: () => ({ health: "healthy", lifecycle: "running", identity: identity() }),
      checkpoint() { checkpointCalls++; return bytes; },
      intervene(request) {
        assert.deepEqual(request.intervention, action);
        assert.equal(request.expectedControlRevision, "3");
        assert.equal(request.sessionId, identity().sessionId);
        control++; staged = true; return receipt();
      },
      drive(elapsed) { assert.equal(elapsed, 0); committed = true; },
      interventionStatus(commandId) { assert.equal(commandId, receipt().commandId); return receipt(); },
      tick(count) {
        assert.ok(count > 0 && count <= 16);
        assert.ok(!staged || committed);
        tick += count;
        energy += (staged ? -1 : 1) * count;
      },
      free() { released.push(tick); },
    };
  }
  const live = owner();
  return { bytes, live, owner, released, captures: () => checkpointCalls };
}

test("the same paused cut feeds independent arms and only the treatment receives a command", () => {
  const f = fixture(), progress = [];
  const job = captureLocalStudy(f.live, action, 35, 65536);
  let arms = 0;
  const result = runLocalStudy(job, bytes => {
    assert.equal(bytes, f.bytes);
    assert.equal(arms++, f.released.length, "only one restored owner exists at a time");
    return f.owner();
  }, (...value) => progress.push(value));
  assert.equal(f.captures(), 1);
  assert.deepEqual(f.released, [106, 106]);
  assert.equal(f.live.frame().identity.tick, "71");
  assert.equal(f.live.frame().identity.controlRevision, "3");
  const ticks = result.control.samples.map(row => row.tick);
  assert.deepEqual(ticks, result.treatment.samples.map(row => row.tick));
  assert.deepEqual(ticks, ["71", ...Array.from({ length: 17 }, (_, i) => String(73 + 2 * i)), "106"]);
  assert.equal(result.treatment.samples[1].averageEnergy - result.control.samples[1].averageEnergy, -4);
  assert.equal(result.control.receipt, null);
  assert.equal(result.treatment.receipt.journalState, "committed_volatile");
  assert.deepEqual(result.difference, { population: 0, averageEnergy: -70, averageHealth: 0 });
  assert.deepEqual(progress.at(-1), ["treatment", 35, 35]);
});

test("invalid horizon, active playback and oversized continuations cannot launch a study", () => {
  const f = fixture();
  for (const n of [0, 1.5, 513]) assert.throws(() => captureLocalStudy(f.live, action, n, 10), /horizon/);
  const moving = { ...f.live, frame: () => ({ ...f.live.frame(), playback: { paused: false } }) };
  assert.throws(() => captureLocalStudy(moving, action, 1, 10), /paused/);
  assert.equal(f.captures(), 0);
  assert.throws(() => captureLocalStudy(f.live, action, 1, 1), /continuation/);
});

test("source mismatch, incomplete steps and unconfirmed treatments refuse a result and release owners", () => {
  for (const fault of ["source", "step", "receipt", "journal"]) {
    const f = fixture();
    const job = captureLocalStudy(f.live, action, 1, 65536);
    let arm = 0;
    assert.throws(() => runLocalStudy(job, () => {
      const cloned = f.owner();
      arm++;
      if (fault === "source") cloned.tick(1);
      if (fault === "step") cloned.tick = () => {};
      if (fault === "receipt") cloned.intervene = () => ({});
      if (fault === "journal") cloned.drive = () => {};
      return cloned;
    }));
    assert.equal(f.released.length, arm);
    assert.equal(f.live.frame().identity.tick, "71");
  }
});

test("extinction retains a population endpoint and leaves means and their differences unavailable", () => {
  const f = fixture();
  const job = captureLocalStudy(f.live, action, 1, 1024);
  const result = runLocalStudy(job, () => {
    const copy = f.owner(), originalFrame = copy.frame;
    if (f.released.length) copy.frame = () => {
      const frame = originalFrame();
      if (frame.snapshot.tick > 71) frame.snapshot.summary = { agentCount: 0, averageEnergy: 0, averageHealth: 0 };
      return frame;
    };
    return copy;
  });
  assert.deepEqual(result.treatment.end, { population: 0, averageEnergy: null, averageHealth: null });
  assert.deepEqual(result.difference, { population: -12, averageEnergy: null, averageHealth: null });
});

test("long horizons include their endpoint and stay within thirty-three observations per arm", () => {
  for (const horizon of [1, 31, 32, 33, 511, 512]) {
    const f = fixture();
    const result = runLocalStudy(captureLocalStudy(f.live, action, horizon, 1024), f.owner);
    const rows = result.control.samples;
    assert.ok(rows.length >= 2 && rows.length <= 33);
    assert.equal(rows.at(-1).tick, String(71 + horizon));
    assert.deepEqual(rows.map(row => row.tick), result.treatment.samples.map(row => row.tick));
    assert.ok(rows.slice(1).every((row, i) => BigInt(row.tick) > BigInt(rows[i].tick)));
  }
});
