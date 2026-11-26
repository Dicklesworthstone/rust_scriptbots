import assert from "node:assert/strict";
import test from "node:test";
import { webcrypto } from "node:crypto";
import { captureLocalStudy, runLocalStudy, createLocalStudyFile, readLocalStudyFile,
  validateLocalStudyJob, validateLocalStudyResult, MAX_LOCAL_STUDY_FILE_BYTES } from "../local-study.js";
import { attachLocalStudy } from "../local-study-view.js";

if (!globalThis.crypto) Object.defineProperty(globalThis, "crypto", { value: webcrypto });

const action = { kind: "drought", region: { shape: "all" }, ticks: 5, growth_scale: 0.1 };

function fixture(maxBatch = 16) {
  const bytes = Uint8Array.of(9, 2);
  let checkpointCalls = 0;
  const released = [];
  function owner() {
    let tick = 71, control = 3, staged = false, committed = false, energy = 1;
    let appliedAt = null;
    const identity = () => ({ sessionId: "340282366920938463463374607431768211455", tick: String(tick),
      scientificRevision: String(tick), controlRevision: String(control), configRevision: "0",
      publicationRevision: String(tick + control) });
    const receipt = () => ({ schema: "scriptbots.browser-intervention-receipt.v1", commandId: "d".repeat(32),
      applicationState: "applied", journalState: committed ? "committed_volatile" : "pending",
      admissionSequence: "4", applicationBoundary: appliedAt,
      source: { ...identity(), epoch: "2", paused: true }, ownerError: null, message: null, messageTruncated: false });
    return {
      frame: () => ({ identity: identity(), playback: { paused: true }, snapshot: { epoch: "2", tick,
        summary: { agentCount: 12, averageEnergy: energy, averageHealth: 0.5 } } }),
      ownerStatus: () => ({ health: "healthy", lifecycle: "running", identity: identity() }),
      checkpoint() { checkpointCalls++; return bytes; },
      intervene(request) {
        assert.deepEqual(request.intervention, action);
        assert.equal(request.expectedControlRevision, String(control));
        assert.equal(request.sessionId, identity().sessionId);
        control++; staged = true;
        appliedAt = { tick: String(tick), scientificRevision: String(tick), controlRevision: String(control), configRevision: "0" };
        return receipt();
      },
      drive(elapsed) { assert.equal(elapsed, 0); committed = true; },
      interventionStatus(commandId) { assert.equal(commandId, receipt().commandId); return receipt(); },
      tick(count) {
        assert.ok(count > 0 && count <= maxBatch);
        assert.ok(!staged || committed);
        tick += count;
        control += count;
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

test("delayed treatments split both arms at the same exact boundary, including the last possible tick", () => {
  for (const [ticks, delay] of [[35, 7], [512, 1], [512, 511]]) {
    const f = fixture(17);
    const job = captureLocalStudy(f.live, action, ticks, 1024, delay);
    const result = runLocalStudy(job, f.owner);
    const boundary = String(71 + delay);
    assert.equal(result.delayTicks, delay);
    assert.equal(result.treatment.receipt.applicationBoundary.tick, boundary);
    assert.equal(result.treatment.receipt.applicationBoundary.controlRevision, String(4 + delay));
    const { control, treatment } = result;
    assert.deepEqual(control.samples.map(row => row.tick), treatment.samples.map(row => row.tick));
    const at = control.samples.findIndex(row => row.tick === boundary);
    assert.ok(at > 0 && at < control.samples.length - 1 && control.samples.length <= 33);
    assert.deepEqual(control.samples.slice(0, at + 1), treatment.samples.slice(0, at + 1));
    assert.equal(result.difference.averageEnergy, -2 * (ticks - delay));
    assert.equal(job.source.tick, f.live.frame().identity.tick);
    assert.deepEqual(f.released, [71 + ticks, 71 + ticks]);
    assert.ok(Object.isFrozen(validateLocalStudyResult(result, job)));
  }
});

test("impossible delays are refused by both capture and worker before allocating an owner", () => {
  const f = fixture(), job = captureLocalStudy(f.live, action, 4, 1024);
  for (const delay of [-1, 0.5, 4, 5, NaN, "1", null]) {
    assert.throws(() => captureLocalStudy(f.live, action, 4, 1024, delay), /delay/);
    assert.throws(() => runLocalStudy({ ...job, delayTicks: delay }, () => assert.fail("unexpected restore")), /delay/);
  }
  assert.equal(f.captures(), 1);
});

test("wrong queue timing and divergent pretreatment samples cannot produce a paired result", () => {
  for (const fault of ["early receipt", "divergent baseline"]) {
    const f = fixture();
    const job = captureLocalStudy(f.live, action, 8, 1024, 4);
    assert.throws(() => runLocalStudy(job, () => {
      const copy = f.owner();
      if (f.released.length && fault === "early receipt") {
        const observe = copy.interventionStatus;
        const earlier = r => ({ ...r, applicationBoundary: { ...r.applicationBoundary, tick: "71" } });
        const submit = copy.intervene;
        copy.intervene = request => earlier(submit(request));
        copy.interventionStatus = (...args) => earlier(observe(...args));
      }
      if (f.released.length && fault === "divergent baseline") {
        const read = copy.frame;
        copy.frame = () => {
          const frame = read();
          if (frame.identity.tick === "73") frame.snapshot.summary.averageEnergy++;
          return frame;
        };
      }
      return copy;
    }), fault === "early receipt" ? /requested scientific tick/ : /unchanged trajectories/);
    assert.equal(f.released.length, 2);
  }
});

async function sections(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const length = new DataView(bytes.buffer).getUint32(8, true);
  return { bytes, text: new TextDecoder().decode(bytes.subarray(44, 44 + length)), payload: bytes.slice(44 + length) };
}

// Independent wire construction lets malformed but correctly checksummed metadata
// reach the reader's semantic gates rather than merely failing checksum validation.
async function wire(text, payload) {
  const metadata = typeof text === "string" ? new TextEncoder().encode(text) : text;
  const header = new Uint8Array(44);
  header.set(new TextEncoder().encode("SBSTUDY1"));
  new DataView(header.buffer).setUint32(8, metadata.length, true);
  header.set(new Uint8Array(await webcrypto.subtle.digest("SHA-256", metadata)), 12);
  return new Blob([header, metadata, payload]);
}

test("portable studies retain the exact checkpoint, delayed treatment and full-width source for independent repeated science", async () => {
  const f = fixture(), job = captureLocalStudy(f.live, action, 35, 1024, 7);
  const prepared = await createLocalStudyFile(job, 1024);
  const raw = await sections(prepared.file), metadata = JSON.parse(raw.text);
  assert.equal(new TextDecoder().decode(raw.bytes.subarray(0, 8)), "SBSTUDY1");
  assert.deepEqual(raw.payload, Uint8Array.of(9, 2));
  assert.equal(metadata.schema, "scriptbots.local-paired-study-file.v1");
  assert.equal(metadata.source.sessionId, "340282366920938463463374607431768211455");
  assert.deepEqual(metadata.action, action);
  assert.equal(metadata.ticks, 35);
  assert.equal(metadata.delayTicks, 7);
  const digest = Buffer.from(await webcrypto.subtle.digest("SHA-256", raw.payload)).toString("hex");
  assert.equal(metadata.checkpoint.sha256, digest);
  const loaded = await readLocalStudyFile(prepared.file, 1024);
  assert.deepEqual(loaded.job, job);
  assert.deepEqual(loaded.checkpoint, { bytes: 2, sha256: digest });
  const original = runLocalStudy(job, f.owner), repeated = runLocalStudy(loaded.job, f.owner);
  assert.deepEqual(repeated.control.samples, original.control.samples);
  assert.deepEqual(repeated.treatment.samples, original.treatment.samples);
  assert.deepEqual(repeated.difference, original.difference);
  assert.equal(validateLocalStudyResult(repeated, loaded.job).source.tick, "71");
  assert.equal(f.live.frame().identity.tick, "71");
  assert.equal(f.captures(), 1, "import and repetition never capture a different live owner");
});

test("async file preparation snapshots metadata and only the exact checkpoint view before hashing", async () => {
  const f = fixture(), captured = captureLocalStudy(f.live, action, 4, 1024, 2);
  const backing = Uint8Array.of(99, 9, 2, 88);
  const job = { ...structuredClone(captured), bytes: backing.subarray(1, 3) };
  const pending = createLocalStudyFile(job, 1024);
  job.source.tick = "72";
  job.action.growth_scale = 0.9;
  job.ticks = 512;
  job.delayTicks = 511;
  backing.fill(0);
  const prepared = await pending;
  assert.deepEqual(prepared.job.bytes, Uint8Array.of(9, 2));
  assert.equal(prepared.job.source.tick, "71");
  assert.equal(prepared.job.action.growth_scale, 0.1);
  assert.equal(prepared.job.ticks, 4);
  assert.equal(prepared.job.delayTicks, 2);
  assert.deepEqual((await readLocalStudyFile(prepared.file, 1024)).job, captured);
});

test("file size, header, metadata and payload corruption refuse without restoring any owner", async () => {
  const f = fixture(), prepared = await createLocalStudyFile(captureLocalStudy(f.live, action, 1, 1024), 1024);
  const { bytes } = await sections(prepared.file);
  const mutations = [
    [0, bytes[0] ^ 1, /version/],
    [12, bytes[12] ^ 1, /metadata checksum/],
    [44, bytes[44] ^ 1, /metadata checksum/],
    [bytes.length - 1, bytes.at(-1) ^ 1, /checkpoint checksum/],
  ];
  for (const [offset, value, error] of mutations) {
    const changed = bytes.slice(); changed[offset] = value;
    await assert.rejects(readLocalStudyFile(new Blob([changed]), 1024), error);
  }
  for (const length of [0, 4097, 0xffff_ffff]) {
    const changed = bytes.slice(); new DataView(changed.buffer).setUint32(8, length, true);
    await assert.rejects(readLocalStudyFile(new Blob([changed]), 1024), /metadata length/);
  }
  await assert.rejects(readLocalStudyFile(new Blob([bytes.slice(0, 12)]), 1024), /empty|size/);
  await assert.rejects(readLocalStudyFile(new Blob([bytes.slice(0, -1)]), 1024), /size/);
  await assert.rejects(readLocalStudyFile(new Blob([bytes, Uint8Array.of(0)]), 1024), /size/);
  await assert.rejects(readLocalStudyFile(prepared.file, 1), /size/);
  class TooLarge extends Blob {
    get size() { return MAX_LOCAL_STUDY_FILE_BYTES + 1; }
    slice() { assert.fail("oversized file must be refused before any read"); }
  }
  await assert.rejects(readLocalStudyFile(new TooLarge(), 64 * 1024 * 1024), /size/);
  assert.deepEqual(f.released, []);
});

test("correct checksums cannot bless invalid plans, ambiguous JSON or unsupported file metadata", async () => {
  const f = fixture(), prepared = await createLocalStudyFile(captureLocalStudy(f.live, action, 1, 1024), 1024);
  const { text, payload } = await sections(prepared.file);
  for (const [changed, reason] of [
    [text.replace('"ticks":1}', '"ticks":0}'), /horizon/],
    [text.replace('"ticks":1}', '"ticks":513}'), /horizon/],
    [text.replace('"delayTicks":0', '"delayTicks":1'), /delay/],
    [text.replace('"delayTicks":0', '"delayTicks":null'), /delay/],
    [text.replace('"delayTicks":0', '"delayTicks":"0"'), /delay/],
    [text.replace('"delayTicks":0,', ''), /fields/],
    [text.replace('"tick":"71"', '"tick":"9007199254740991"'), /exact browser tick/],
    [text.replace('"sessionId":"340282366920938463463374607431768211455"', '"sessionId":1'), /identities/],
    [text.replace('"growth_scale":0.1', '"growth_scale":2'), /Growth scale/],
    [text.replace('"ticks":1}', '"ticks":1,"ticks":1}'), /canonical|duplicate/],
    [text.replace('"ticks":1}', '"ticks":1,"ignored":true}'), /fields/],
    [text.replace('"schema":"scriptbots.local-paired-study-file.v1"', '"schema":"unknown"'), /schema/],
    [text.replace('"bytes":2', '"bytes":3'), /size/],
    [text.replace('"ticks":1}', '"ticks":1.0}'), /canonical/],
  ]) {
    assert.notEqual(changed, text);
    await assert.rejects(readLocalStudyFile(await wire(changed, payload), 1024), reason);
  }
  await assert.rejects(readLocalStudyFile(await wire(Uint8Array.of(0xff), payload), 1024), /UTF-8/);
});

test("file metadata failures are rejected before reading the larger checkpoint section", async () => {
  const f = fixture(), prepared = await createLocalStudyFile(captureLocalStudy(f.live, action, 1, 1024), 1024);
  const { text, payload } = await sections(prepared.file);
  const invalid = await wire(text.replace('"bytes":2', '"bytes":100'), payload);
  let reads = 0;
  class Counted extends Blob {
    slice(...args) { reads++; assert.ok(reads <= 2, "only header and bounded metadata may be read"); return super.slice(...args); }
  }
  await assert.rejects(readLocalStudyFile(new Counted([invalid]), 1024), /size/);
  assert.equal(reads, 2);
});

test("capture and worker admission refuse malformed jobs before restoring a world", () => {
  const f = fixture(), job = captureLocalStudy(f.live, action, 1, 1024);
  for (const mutate of [
    value => { value.bytes = new Uint8Array(); },
    value => { value.source.epoch = "01"; },
    value => { value.source.sessionId = "340282366920938463463374607431768211456"; },
    value => { value.source.extra = "unobserved"; },
    value => { value.ticks = 513; },
    value => { value.delayTicks = 1; },
    value => { value.action.region.radius = 12; },
  ]) {
    const changed = structuredClone(job); mutate(changed);
    assert.throws(() => runLocalStudy(changed, () => assert.fail("invalid worker input must not restore")));
  }
  for (const limit of [0, -1, 1.5, NaN]) assert.throws(() => validateLocalStudyJob(job, limit), /limit/);
  assert.ok(Object.isFrozen(job.action));
  assert.ok(Object.isFrozen(job.action.region));
});

test("result admission binds exact inputs, ordered samples and independently observed endpoints", () => {
  const f = fixture(), job = captureLocalStudy(f.live, action, 35, 1024, 7);
  const result = runLocalStudy(job, f.owner);
  assert.ok(Object.isFrozen(validateLocalStudyResult(result, job)));
  for (const mutate of [
    value => { value.source.tick = "72"; },
    value => { value.action.growth_scale = 0.2; },
    value => { value.ticks = 34; },
    value => { value.delayTicks = 8; },
    value => { value.treatment.samples[1].tick = "74"; },
    value => { value.control.samples.pop(); },
    value => { value.control.samples[0].population = 0; },
    value => { value.treatment.end.averageEnergy = Infinity; },
    value => { value.control.finalTick = "107"; },
    value => { value.difference.population = 1; },
    value => { value.control.receipt = value.treatment.receipt; },
    value => { value.treatment.receipt.journalState = "pending"; },
    value => { value.treatment.receipt.applicationBoundary.tick = "77"; },
    value => { value.treatment.samples[1].averageEnergy++; },
  ]) {
    const changed = structuredClone(result); mutate(changed);
    assert.throws(() => validateLocalStudyResult(changed, job));
  }
});

function viewFixture() {
  class Element {
    constructor(document) { this.ownerDocument = document; this.events = {}; this.children = []; this.value = ""; this.textContent = ""; this.hidden = false; }
    get valueAsNumber() { return this.value.trim() === "" ? NaN : Number(this.value); }
    get validity() { return { valid: Number.isSafeInteger(this.valueAsNumber) && this.valueAsNumber >= 1 && this.valueAsNumber <= 512 }; }
    addEventListener(name, fn) { this.events[name] = fn; }
    async dispatch(name) { return this.events[name]?.(); }
    removeAttribute(name) { delete this[name]; }
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    insertRow() { const row = new Element(this.ownerDocument); this.append(row); return row; }
  }
  const document = { createElement: () => new Element(document) };
  const panel = new Element(document), nodes = new Map(), workers = [];
  panel.querySelector = selector => {
    if (!nodes.has(selector)) nodes.set(selector, new Element(document));
    return nodes.get(selector);
  };
  const field = name => panel.querySelector("#local-study-" + name);
  field("ticks").value = "1";
  field("delay").value = "0";
  let ownerReads = 0, actionReads = 0;
  const f = fixture();
  const controller = attachLocalStudy(panel, {
    getOwner() { ownerReads++; return f.live; },
    getAction() { actionReads++; return action; }, getLimit: () => 1024,
    createWorker() {
      const worker = { terminated: false,
        terminate() { this.terminated = true; },
        postMessage(job, transfer) { this.job = structuredClone(job, { transfer }); },
        finish(result = runLocalStudy(this.job, f.owner)) { this.onmessage({ data: { result } }); },
      };
      workers.push(worker);
      return worker;
    },
  });
  controller.setAvailable(true, true);
  return { f, controller, field, workers, ownerReads: () => ownerReads, actionReads: () => actionReads };
}

test("opening a portable study never restores or captures; explicit replay ignores live form inputs", async () => {
  const v = viewFixture(), f = fixture();
  const prepared = await createLocalStudyFile(captureLocalStudy(f.live, action, 35, 1024, 7), 1024);
  v.controller.setAvailable(false, true); // Independent repetition works with no capture authority.
  v.field("import").files = [prepared.file];
  await v.field("import").dispatch("change");
  assert.equal(v.workers.length, 0);
  assert.equal(v.ownerReads(), 0);
  assert.equal(v.actionReads(), 0);
  assert.equal(v.field("run").disabled, true);
  assert.equal(v.field("replay").disabled, false);
  assert.match(v.field("file-status").textContent, /source tick 71.*35 ticks.*queued at tick 78 after 7 ticks.*no world has been restored/);
  v.field("ticks").value = "512";
  v.field("delay").value = "511";
  await v.field("replay").dispatch("click");
  assert.equal(v.workers[0].job.ticks, 35);
  assert.equal(v.workers[0].job.delayTicks, 7);
  assert.deepEqual(v.workers[0].job.action, action);
  v.workers[0].finish();
  assert.match(v.field("status").textContent, /queued at tick 78/);
  assert.match(v.field("sample-label").textContent, /after treatment/);
  v.field("sample").value = "0";
  await v.field("sample").dispatch("input");
  assert.match(v.field("sample-label").textContent, /before treatment/);
  v.field("sample").value = "4";
  await v.field("sample").dispatch("input");
  assert.match(v.field("sample-label").textContent, /tick 78.*treatment queued here/);
  assert.equal(v.field("file-export").hidden, false);
  assert.equal(v.field("export").hidden, false);
  assert.equal(v.workers[0].terminated, true);
  const output = await (await fetch(v.field("export").href)).json();
  assert.deepEqual(output.checkpoint, prepared.checkpoint);
  const exported = await (await fetch(v.field("file-export").href)).blob();
  assert.deepEqual(new Uint8Array(await exported.arrayBuffer()), new Uint8Array(await prepared.file.arrayBuffer()));
  assert.equal(v.f.live.frame().identity.tick, "71");
  assert.equal(v.ownerReads(), 0);
  assert.equal(v.actionReads(), 0);
  // A second repetition rereads the immutable source file after the first buffer was transferred.
  await v.field("replay").dispatch("click");
  assert.deepEqual(v.workers[1].job.bytes, Uint8Array.of(9, 2));
  v.controller.reset();
});

test("live capture retains the requested delay and refuses invalid timing before checkpoint work", async () => {
  const v = viewFixture();
  v.field("ticks").value = "8";
  await v.field("ticks").dispatch("input");
  assert.equal(v.field("delay").max, "7");
  v.field("delay").value = "3";
  await v.field("run").dispatch("click");
  assert.equal(v.workers[0].job.delayTicks, 3);
  v.workers[0].finish();
  assert.match(v.field("status").textContent, /queued at tick 74/);
  for (const invalid of ["", "0.5", "8"]) {
    v.field("delay").value = invalid;
    await v.field("run").dispatch("click");
    assert.match(v.field("status").textContent, /unavailable.*delay/);
    assert.equal(v.workers.length, 1);
    assert.equal(v.f.captures(), 1);
  }
  v.controller.reset();
});

test("cancellation and reset fence asynchronous import and export preparation", async () => {
  const v = viewFixture(), f = fixture();
  const prepared = await createLocalStudyFile(captureLocalStudy(f.live, action, 1, 1024), 1024);
  let release;
  class Delayed extends Blob {
    slice(...args) {
      const section = super.slice(...args);
      return { arrayBuffer: () => new Promise(resolve => { release = async () => resolve(await section.arrayBuffer()); }) };
    }
  }
  v.field("import").files = [new Delayed([prepared.file])];
  const reading = v.field("import").dispatch("change");
  v.controller.reset();
  await release();
  await reading;
  assert.equal(v.field("replay").disabled, true);
  assert.equal(v.workers.length, 0);
  assert.match(v.field("file-status").textContent, /Choose a saved study/);
  const preparing = v.field("run").dispatch("click");
  await v.field("cancel").dispatch("click");
  await preparing;
  assert.equal(v.workers.length, 0);
  assert.equal(v.field("status").textContent, "Study cancelled.");
  assert.equal(v.field("file-export").hidden, true);
});

test("cancelled worker messages and mismatched results cannot replace the selected comparison", async () => {
  const v = viewFixture();
  await v.field("run").dispatch("click");
  const old = v.workers[0], oldResult = runLocalStudy(old.job, v.f.owner);
  await v.field("cancel").dispatch("click");
  await v.field("run").dispatch("click");
  old.finish(oldResult);
  assert.match(v.field("status").textContent, /Restoring the control/);
  const current = v.workers[1], wrong = runLocalStudy(current.job, v.f.owner);
  wrong.ticks = 2;
  current.finish(wrong);
  assert.match(v.field("status").textContent, /Study failed.*selected source/);
  assert.equal(v.field("file-export").hidden, true);
  assert.equal(v.field("export").hidden, true);
  assert.equal(current.terminated, true);
  v.controller.reset();
});

test("file cancellation permits the same selection again and malformed worker envelopes release the task", async () => {
  const v = viewFixture(), f = fixture();
  const prepared = await createLocalStudyFile(captureLocalStudy(f.live, action, 1, 1024), 1024);
  let release;
  class Delayed extends Blob {
    slice(...args) {
      const section = super.slice(...args);
      return { arrayBuffer: () => new Promise(resolve => { release = async () => resolve(await section.arrayBuffer()); }) };
    }
  }
  v.field("import").value = "same-file.sbstudy";
  v.field("import").files = [new Delayed([prepared.file])];
  const reading = v.field("import").dispatch("change");
  assert.equal(v.field("import").value, "", "the same file must trigger another native change event");
  await v.field("cancel").dispatch("click");
  assert.match(v.field("file-status").textContent, /cancelled.*again/);
  await release(); await reading;
  assert.match(v.field("file-status").textContent, /cancelled.*again/);
  v.field("import").files = [prepared.file];
  await v.field("import").dispatch("change");
  await v.field("replay").dispatch("click");
  v.workers[0].onmessage({ data: null });
  assert.equal(v.workers[0].terminated, true);
  assert.match(v.field("status").textContent, /Study failed/);
  assert.equal(v.field("replay").disabled, false);
  v.controller.reset();
});
