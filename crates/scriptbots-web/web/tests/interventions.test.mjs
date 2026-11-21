import assert from "node:assert/strict";
import test from "node:test";
import {
  InterventionController, interventionReceiptText, makeInterventionRequest,
  validateInterventionAction, validateInterventionHistory, validateInterventionReceipt,
} from "../interventions.js";
import { RemoteStreamClient } from "../remote.js";

const COMMAND = "d78bce3980d94a5bb851b15b0f7b0471";
const SESSION = "18446744073709551610";
const flush = () => new Promise((resolve) => setImmediate(resolve));
const add = (value, delta = 1n) => (BigInt(value) + delta).toString();
const action = () => ({ kind: "drought", region: { shape: "all" }, ticks: 5, growth_scale: 0.1 });

function frame(changes = {}) {
  return { identity: { sessionId: SESSION, publicationRevision: "20", scientificRevision: "12",
    controlRevision: "7", configRevision: "3", tick: "10", ...changes.identity },
  snapshot: { epoch: "0", ...changes.snapshot }, playback: { paused: true, speedMultiplier: 0, ...changes.playback } };
}

function request(source = frame()) {
  return { ...makeInterventionRequest(source, action(), "web-intervention-test"), identity: source.identity };
}

function sourceFor(source = frame(), changes = {}) {
  return { ...source.identity, epoch: source.snapshot.epoch, paused: source.playback.paused, ...changes };
}

function receipt(req, { source = frame(), state = "applied", journal = "pending", changes = {} } = {}) {
  const applied = state === "applied";
  const observed = sourceFor(source, {
    publicationRevision: add(source.identity.publicationRevision, applied ? 2n : 1n),
    controlRevision: add(source.identity.controlRevision, applied ? 1n : 0n),
    scientificRevision: add(source.identity.scientificRevision, applied ? 1n : 0n),
  });
  return { schema: "scriptbots.browser-intervention-receipt.v1", commandId: COMMAND,
    admissionSequence: "9007199254740994", applicationState: state, journalState: journal,
    applicationBoundary: applied ? { tick: source.identity.tick, controlRevision: observed.controlRevision,
      scientificRevision: observed.scientificRevision, configRevision: observed.configRevision } : null,
    message: null, messageTruncated: false, ownerError: null, source: observed, ...changes };
}

function historyRequest(source = frame(), changes = {}) {
  return { sessionId: source.identity.sessionId, epoch: source.snapshot.epoch, identity: source.identity,
    afterSeq: "0", limit: 32, ...changes };
}

function record(seq, changes = {}) {
  return { seq: String(seq), tick: "1", kind: "bloom", region: { shape: "all" },
    agentsAffected: "0", cellsAffected: "4", expiresAt: null, expired: false, ...changes };
}

function history(req = historyRequest(), changes = {}) {
  return { schema: "scriptbots.browser-intervention-history.v1", source: sourceFor(frame()), closed: true,
    afterSeq: req.afterSeq, earliestRetained: "1", latestSeq: "2", nextAfterSeq: "2",
    hasMore: false, gap: null, records: [record(1), record(2)], ...changes };
}

function harness() {
  let time = 0, nextId = 0;
  const sends = [], statuses = [], histories = [];
  const controller = new InterventionController({
    now: () => time,
    id: () => `web-intervention-request-${++nextId}`,
    send: (request, signal) => new Promise((resolve, reject) => sends.push({ request, signal, resolve, reject })),
    status: (commandId, sessionId, signal) => new Promise((resolve, reject) => statuses.push({ commandId, sessionId, signal, resolve, reject })),
    history: (request, signal) => new Promise((resolve, reject) => histories.push({ request, signal, resolve, reject })),
  });
  const current = frame();
  controller.observe(current, 1);
  controller.setEnabled(true);
  return { controller, current, sends, statuses, histories,
    advance: (milliseconds) => { time += milliseconds; }, identities: () => nextId };
}

test("toolbox actions preserve canonical scalar payloads and reject unsupported or nonfinite input", () => {
  for (const value of [
    action(), { kind: "embargo", region: { shape: "disc", x: -5, y: 3, radius: 4 }, ticks: 0xffff_ffff },
    { kind: "bloom", region: { shape: "rect", x: 0, y: 2, w: 4, h: 5 }, amount: 0 },
    { kind: "meteor", region: { shape: "all" }, lethality: 2, scorch: 0.3 },
    { kind: "set_closed_world", closed: false },
  ]) assert.equal(validateInterventionAction(value), value);
  for (const value of [
    { ...action(), ticks: 0 }, { ...action(), ticks: 0x1_0000_0000 }, { ...action(), ticks: 1.5 },
    { ...action(), growth_scale: NaN }, { ...action(), growth_scale: 1.1 },
    { ...action(), extra: "silently ignored input" }, { ...action(), region: { shape: "all", radius: 5 } },
    { ...action(), region: { shape: "disc", x: 0, y: 0, radius: -1 } },
    { kind: "bloom", region: { shape: "all" }, amount: 1e300 },
    { kind: "meteor", region: { shape: "all" }, lethality: -1, scorch: 0.3 },
    { kind: "set_closed_world", closed: "false" },
    { kind: "inject_cohort", region: { shape: "all" }, count: 10 },
  ]) assert.throws(() => validateInterventionAction(value));
});

test("new requests freeze the exact paused source and a detached payload beyond JavaScript integer precision", () => {
  const value = { kind: "meteor", region: { shape: "disc", x: 1, y: 2, radius: 3 }, lethality: 0.5, scorch: 0.25 };
  const source = frame({ identity: { controlRevision: "9007199254740993" }, snapshot: { epoch: "9007199254740995" } });
  const frozen = makeInterventionRequest(source, value, "web-intervention-wide");
  assert.equal(frozen.sessionId, SESSION);
  assert.equal(frozen.epoch, "9007199254740995");
  assert.equal(frozen.expectedControlRevision, "9007199254740993");
  assert.ok(Object.isFrozen(frozen));
  assert.ok(Object.isFrozen(frozen.intervention));
  assert.ok(Object.isFrozen(frozen.intervention.region));
  value.region.radius = 99;
  value.lethality = 8;
  source.identity.controlRevision = "9007199254740999";
  assert.equal(frozen.intervention.region.radius, 3);
  assert.equal(frozen.intervention.lethality, 0.5);
  assert.equal(frozen.expectedControlRevision, "9007199254740993");
  assert.throws(() => { frozen.intervention.region.radius = 4; }, TypeError);
  assert.throws(() => makeInterventionRequest(frame({ playback: { paused: false } }), action(), "valid-id"), /Pause/);
  assert.throws(() => makeInterventionRequest(frame(), action(), "x".repeat(129)));
  assert.throws(() => makeInterventionRequest(frame({ identity: { controlRevision: 7 } }), action(), "valid-id"));
});

test("receipts retain canonical hexadecimal command IDs, exact large boundaries and separate lifecycle axes", () => {
  const source = frame({ identity: { tick: "9007199254740993", controlRevision: "9007199254740994",
    scientificRevision: "9007199254740995", publicationRevision: "9007199254740996", configRevision: "9007199254740992" } });
  const req = request(source), applied = receipt(req, { source, journal: "committed_volatile" });
  assert.equal(validateInterventionReceipt(applied, req), applied);
  assert.equal(applied.applicationBoundary.tick, "9007199254740993");
  assert.match(interventionReceiptText(applied), /Queued at completed tick 9007199254740993/);
  assert.match(interventionReceiptText(applied), /does not identify an effect-history row/);
  assert.match(interventionReceiptText(applied), /committed volatile/);
  const admitted = receipt(req, { source, state: "admitted", journal: "pending" });
  assert.equal(validateInterventionReceipt(admitted, req), admitted);
  assert.match(interventionReceiptText(admitted), /queue application has not yet been observed/);
  const afterEpoch = structuredClone(applied);
  afterEpoch.source.epoch = "1";
  afterEpoch.source.publicationRevision = add(afterEpoch.source.publicationRevision);
  assert.equal(validateInterventionReceipt(afterEpoch, req, applied), afterEpoch, "a known command survives a later epoch");
  const malformed = structuredClone(applied);
  malformed.commandId = COMMAND.toUpperCase();
  assert.throws(() => validateInterventionReceipt(malformed, req));
  malformed.commandId = "18446744073709551615";
  assert.throws(() => validateInterventionReceipt(malformed, req));
});

test("receipt validation rejects impossible admission and journal combinations from the owner protocol", () => {
  const req = request();
  for (const state of ["admitted", "applied", "failed"]) {
    const invalid = receipt(req, { state }); invalid.admissionSequence = null;
    assert.throws(() => validateInterventionReceipt(invalid, req), `${state} must have admission identity`);
  }
  for (const journal of ["committed_volatile", "durable", "failed"]) {
    assert.throws(() => validateInterventionReceipt(receipt(req, { state: "admitted", journal }), req));
  }
  for (const journal of ["not_required", "pending"]) {
    assert.doesNotThrow(() => validateInterventionReceipt(receipt(req, { state: "admitted", journal }), req));
  }
  const refusal = receipt(req, { state: "rejected", journal: "pending", changes: { admissionSequence: null } });
  assert.equal(validateInterventionReceipt(refusal, req), refusal, "pre-admission rejection may still have a pending audit");
});

test("retained receipts refuse source, identity, application and journal regressions", () => {
  const req = request();
  const previous = receipt(req, { journal: "committed_volatile" });
  for (const mutate of [
    (r) => { r.commandId = "a".repeat(32); },
    (r) => { r.admissionSequence = "1"; },
    (r) => { r.applicationState = "admitted"; r.applicationBoundary = null; r.journalState = "pending"; },
    (r) => { r.applicationBoundary.tick = "9"; },
    (r) => { r.source.sessionId = "1"; },
    (r) => { r.source.tick = "9"; },
    (r) => { r.source.publicationRevision = "19"; },
    (r) => { r.source.scientificRevision = "12"; r.applicationBoundary.scientificRevision = "12"; },
    (r) => { r.source.controlRevision = "7"; r.applicationBoundary.controlRevision = "7"; },
    (r) => { r.source.configRevision = "2"; r.applicationBoundary.configRevision = "2"; },
    (r) => { r.journalState = "pending"; },
    (r) => { r.journalState = "not_required"; },
    (r) => { r.ownerError = "x".repeat(1025); },
  ]) {
    const invalid = structuredClone(previous); mutate(invalid);
    assert.throws(() => validateInterventionReceipt(invalid, req, previous));
  }
  const durable = structuredClone(previous); durable.journalState = "durable";
  assert.doesNotThrow(() => validateInterventionReceipt(durable, req, previous));
  const failed = structuredClone(previous); failed.journalState = "failed";
  assert.doesNotThrow(() => validateInterventionReceipt(failed, req, previous));
  for (const terminal of [durable, failed]) {
    for (const state of ["pending", "committed_volatile"]) {
      const invalid = structuredClone(terminal); invalid.journalState = state;
      assert.throws(() => validateInterventionReceipt(invalid, req, terminal));
    }
  }
  const pending = receipt(req, { state: "admitted" });
  for (const field of ["tick", "scientificRevision", "controlRevision", "configRevision"]) {
    const newer = structuredClone(pending);
    newer.source[field] = add(newer.source[field]);
    const regressed = structuredClone(newer);
    regressed.source.publicationRevision = add(newer.source.publicationRevision);
    regressed.source[field] = pending.source[field];
    assert.throws(() => validateInterventionReceipt(regressed, req, newer),
      `${field} cannot regress merely because the publication advanced`);
  }
});

test("history accepts exact missing prefixes, true expiry records and independent global effects", () => {
  const req = historyRequest(frame(), { afterSeq: "1", limit: 2 });
  const page = history(req, { earliestRetained: "4", latestSeq: "6", nextAfterSeq: "5", hasMore: true,
    gap: { firstMissing: "2", lastMissing: "3" }, records: [
      record(4, { kind: "drought", expiresAt: "6" }),
      record(5, { kind: "expiry:drought", tick: "6", expired: true, cellsAffected: "0" }),
    ] });
  assert.equal(validateInterventionHistory(page, req), page);
  assert.ok(!Object.hasOwn(page.records[0], "commandId"), "effects must not invent a per-command relationship");
  page.records[1].kind = "expiry:embargo";
  assert.equal(validateInterventionHistory(page, req), page);
  const following = historyRequest(frame(), { afterSeq: "5", limit: 2 });
  const next = history(following, { earliestRetained: "4", latestSeq: "6", nextAfterSeq: "6",
    records: [record(6, { kind: "set_closed_world" })] });
  assert.equal(validateInterventionHistory(next, following), next);
});

test("empty and maximum-u64 history cursors remain exact without overflow", () => {
  const emptyRequest = historyRequest();
  const empty = history(emptyRequest, { latestSeq: "0", nextAfterSeq: "0", records: [] });
  assert.equal(validateInterventionHistory(empty, emptyRequest), empty);
  const max = "18446744073709551615", prior = "18446744073709551614";
  const maxRequest = historyRequest(frame(), { afterSeq: prior, limit: 1 });
  const last = history(maxRequest, { earliestRetained: max, latestSeq: max, nextAfterSeq: max, records: [record(max)] });
  assert.equal(validateInterventionHistory(last, maxRequest), last);
});

test("history rejects hidden loss, invented bounds, invalid records and discontinuous cursors", () => {
  const req = historyRequest(frame(), { afterSeq: "1", limit: 2 });
  const valid = history(req, { earliestRetained: "4", latestSeq: "6", nextAfterSeq: "5", hasMore: true,
    gap: { firstMissing: "2", lastMissing: "3" }, records: [record(4), record(5)] });
  for (const mutate of [
    (p) => { p.gap = null; }, (p) => { p.gap.firstMissing = "3"; }, (p) => { p.gap.lastMissing = "4"; },
    (p) => { p.afterSeq = "0"; }, (p) => { p.earliestRetained = "0"; }, (p) => { p.earliestRetained = "8"; },
    (p) => { p.nextAfterSeq = "6"; }, (p) => { p.hasMore = false; }, (p) => { p.records.pop(); },
    (p) => { p.records.push(record(6)); }, (p) => { p.records[0].seq = "5"; },
    (p) => { p.records[1].seq = "18446744073709551616"; },
    (p) => { p.records[0].tick = "11"; }, (p) => { p.records[0].cellsAffected = 4; },
    (p) => { p.records[0].kind = "expiry:made_up"; },
    (p) => { p.records[0].kind = "expiry:drought"; p.records[0].expired = false; },
    (p) => { p.records[0].expired = true; },
    (p) => { p.records[0].region = { shape: "disc", x: 0, y: 0, radius: Infinity }; },
    (p) => { p.records[0].expiresAt = "00"; },
    (p) => { p.source.sessionId = "1"; }, (p) => { p.source.epoch = "1"; },
    (p) => { p.source.publicationRevision = "19"; }, (p) => { p.source.tick = "9"; },
  ]) {
    const invalid = structuredClone(valid); mutate(invalid);
    assert.throws(() => validateInterventionHistory(invalid, req));
  }
  const ordinary = history(); ordinary.gap = { firstMissing: "1", lastMissing: "1" };
  assert.throws(() => validateInterventionHistory(ordinary, historyRequest()));
  const oversizedRequest = historyRequest(frame(), { limit: 65 });
  const oversized = history(oversizedRequest, { latestSeq: "65", nextAfterSeq: "65",
    records: Array.from({ length: 65 }, (_, index) => record(index + 1)) });
  assert.throws(() => validateInterventionHistory(oversized, oversizedRequest));
});

test("a lost delivery preserves one frozen request identity and source across explicit retry", async () => {
  const h = harness(), mutable = action();
  const staged = h.controller.stage(mutable);
  assert.equal(h.sends.length, 1);
  const sent = h.sends[0].request;
  mutable.ticks = 999;
  h.sends[0].reject(new Error("connection closed after admission may have occurred"));
  await staged;
  assert.equal(h.controller.receipt, null);
  assert.match(h.controller.commandError, /unconfirmed/);
  assert.equal(h.controller.canStage, false);
  await assert.rejects(h.controller.stage(action()));
  assert.equal(h.identities(), 1);
  h.controller.observe(frame({ identity: { publicationRevision: "22", controlRevision: "8" } }), 1);
  const retry = h.controller.retry();
  assert.equal(h.sends.length, 2);
  assert.equal(h.sends[1].request, sent);
  assert.equal(sent.expectedControlRevision, "7");
  assert.equal(sent.intervention.ticks, 5);
  h.sends[1].resolve(receipt(sent));
  await retry;
  assert.equal(h.controller.receipt.commandId, COMMAND);
  assert.equal(h.identities(), 1);
});

test("admission and queue application do not permit another experiment before a subsequent science tick", async () => {
  const h = harness();
  const staged = h.controller.stage(action());
  const req = h.sends[0].request;
  h.sends[0].resolve(receipt(req, { state: "admitted" }));
  await staged;
  assert.equal(h.controller.canStage, false);
  const refresh = h.controller.refreshReceipt();
  h.statuses[0].resolve(receipt(req, { journal: "committed_volatile" }));
  await refresh;
  assert.equal(h.controller.receipt.applicationState, "applied");
  assert.equal(h.controller.canStage, false, "queue application has not advanced scientific time");
  h.controller.observe(frame({ identity: { publicationRevision: "23", controlRevision: "9", scientificRevision: "14", tick: "11" } }), 1);
  assert.equal(h.controller.canStage, true);
});

test("an actual stale-control rejection remains a refusal and permits a fresh paused experiment", async () => {
  const h = harness();
  const staged = h.controller.stage(action());
  const refused = receipt(h.sends[0].request, { state: "rejected", journal: "committed_volatile",
    changes: { message: "ControlRevisionConflict { expected: 7, actual: 8 }" } });
  refused.source.controlRevision = "8";
  h.sends[0].resolve(refused);
  await staged;
  assert.equal(h.controller.receipt.applicationState, "rejected");
  assert.equal(h.controller.receipt.applicationBoundary, null);
  assert.match(interventionReceiptText(h.controller.receipt), /rejected/);
  assert.equal(h.controller.canStage, true);
});

test("unknown status never fabricates non-admission or allocates a fresh retry identity", async () => {
  const h = harness(), staged = h.controller.stage(action());
  const original = h.sends[0].request;
  h.sends[0].resolve(receipt(original, { state: "admitted" }));
  await staged;
  const refresh = h.controller.refreshReceipt();
  assert.equal(h.statuses[0].commandId, COMMAND);
  assert.equal(h.statuses[0].sessionId, SESSION);
  h.statuses[0].resolve(null);
  await refresh;
  assert.equal(h.controller.receipt.applicationState, "admitted");
  assert.match(h.controller.commandError, /Absence does not prove/);
  assert.equal(h.controller.canStage, false);
  const retried = h.controller.retry();
  assert.equal(h.sends[1].request, original);
  h.sends[1].resolve(receipt(original));
  await retried;
  assert.equal(h.identities(), 1);
});

test("owner generation changes abort a pending post and discard its late outcome", async () => {
  const h = harness(), staged = h.controller.stage(action());
  const old = h.sends[0];
  h.controller.observe(h.current, 2);
  assert.equal(old.signal.aborted, true);
  assert.equal(h.controller.retained, null);
  old.resolve(receipt(old.request));
  await staged;
  assert.equal(h.controller.receipt, null);
  assert.equal(h.controller.commandError, null);
  const next = h.controller.stage(action());
  assert.notEqual(h.sends[1].request.requestId, old.request.requestId);
  h.sends[1].resolve(receipt(h.sends[1].request));
  await next;
});

test("closing aborts in-flight work but retains the exact uncertain command for explicit retry", async () => {
  const h = harness(), staged = h.controller.stage(action());
  const old = h.sends[0];
  h.controller.setEnabled(false);
  assert.equal(old.signal.aborted, true);
  assert.equal(h.histories[0].signal.aborted, true);
  assert.equal(h.controller.retained.request, old.request);
  assert.match(h.controller.commandError, /original request is retained/);
  old.resolve(receipt(old.request));
  await staged;
  assert.equal(h.controller.receipt, null, "closed-panel responses cannot overwrite state");
  h.controller.setEnabled(true);
  const retry = h.controller.retry();
  assert.equal(h.sends[1].request, old.request);
  h.sends[1].resolve(receipt(old.request));
  await retry;
  assert.equal(h.identities(), 1);
});

test("returning to a saved owner preserves its queued experiment and history cursor without replaying it", async () => {
  const h = harness();
  const pageRead = h.controller.historyPageAfter("2");
  const pageRequest = h.histories[1].request;
  const savedPage = history(pageRequest, { records: [] });
  h.histories[1].resolve(savedPage);
  await pageRead;
  const staged = h.controller.stage(action());
  const original = h.sends[0].request;
  const originalIdentity = h.controller.retained.identity;
  const queued = receipt(original, { journal: "durable" });
  h.sends[0].resolve(queued);
  await staged;
  h.controller.setEnabled(false);

  // Other playback commands may publish while science remains paused at tick 10.
  const returned = frame({ identity: { publicationRevision: "30", controlRevision: "12", scientificRevision: "17" } });
  h.controller.reattachOwner(returned, 3);
  assert.equal(h.controller.enabled, false, "reattachment cannot start work before the view is enabled");
  assert.equal(h.controller.retained.request, original);
  assert.equal(h.controller.retained.identity, originalIdentity);
  assert.equal(h.controller.retained.context, `3/${SESSION}`);
  assert.equal(h.controller.receipt, queued);
  assert.equal(h.controller.historyPage, savedPage);
  assert.equal(h.controller.historyAfter, "2");
  assert.equal(h.histories.length, 2, "reattachment alone performs no reads");
  assert.equal(h.sends.length, 1, "restoring the view never replays its submission");
  h.controller.setEnabled(true);
  assert.equal(h.histories[2].request.afterSeq, "2");
  assert.equal(h.histories[2].request.identity.controlRevision, "12");
  assert.equal(original.expectedControlRevision, "7", "the original command guard cannot follow the current frame");
  assert.equal(h.controller.canStage, false, "the first ecological effect is still queued at this same science tick");
  await assert.rejects(h.controller.stage(action()));
  h.controller.observe(frame({ identity: { publicationRevision: "31", controlRevision: "12", scientificRevision: "18", tick: "11" } }), 3);
  assert.equal(h.controller.canStage, true, "the next observed science tick releases the experiment slot");
  assert.equal(h.identities(), 1);
});

test("reattaching an uncertain owner aborts old I/O and only an explicit retry resends its frozen request", async () => {
  const h = harness(), staged = h.controller.stage(action());
  const old = h.sends[0], originalIdentity = h.controller.retained.identity;
  h.controller.reattachOwner(frame({ identity: { publicationRevision: "23", controlRevision: "8", scientificRevision: "13" } }), 5);
  assert.equal(old.signal.aborted, true);
  assert.equal(h.histories[0].signal.aborted, true);
  assert.equal(h.controller.retained.request, old.request);
  assert.equal(h.controller.retained.identity, originalIdentity);
  assert.match(h.controller.commandError, /original request is retained/);
  assert.equal(h.controller.enabled, false);
  h.controller.setEnabled(true);
  assert.equal(h.sends.length, 1, "a reconnect must never automatically submit an uncertain command");
  assert.equal(h.controller.canStage, false);
  old.resolve(receipt(old.request));
  await staged;
  assert.equal(h.controller.receipt, null, "a detached response cannot replace the retained uncertain outcome");

  const retried = h.controller.retry();
  assert.equal(h.sends[1].request, old.request);
  assert.equal(h.sends[1].request.expectedControlRevision, "7");
  h.sends[1].resolve(receipt(old.request));
  await retried;
  assert.equal(h.controller.receipt.commandId, COMMAND);
  assert.equal(h.identities(), 1);
});

test("reattaching a different owner session clears old receipts and ignores its late submission result", async () => {
  const h = harness(), staged = h.controller.stage(action());
  const old = h.sends[0];
  const replacement = frame({ identity: { sessionId: "18446744073709551611" } });
  h.controller.reattachOwner(replacement, 2);
  assert.equal(old.signal.aborted, true);
  assert.equal(h.controller.retained, null);
  assert.equal(h.controller.receipt, null);
  assert.equal(h.controller.historyPage, null);
  assert.equal(h.controller.historyAfter, "0");
  assert.equal(h.controller.enabled, false);
  old.resolve(receipt(old.request));
  await staged;
  assert.equal(h.controller.receipt, null);
  h.controller.setEnabled(true);
  const next = h.controller.stage(action());
  assert.equal(h.sends[1].request.sessionId, replacement.identity.sessionId);
  assert.notEqual(h.sends[1].request.requestId, old.request.requestId);
  h.sends[1].resolve(receipt(h.sends[1].request, { source: replacement }));
  await next;
  assert.equal(h.controller.receipt.source.sessionId, replacement.identity.sessionId);
});

test("an explicit owner replacement cannot restore a request merely because its checkpoint reused the session", async () => {
  const h = harness(), staged = h.controller.stage(action());
  const old = h.sends[0];
  h.sends[0].resolve(receipt(old.request, { journal: "durable" }));
  await staged;
  h.controller.setEnabled(false);
  h.controller.reset();
  h.controller.reattachOwner(h.current, 4);
  h.controller.setEnabled(true);
  assert.equal(h.controller.retained, null);
  assert.equal(h.controller.receipt, null);
  assert.equal(h.controller.canStage, true);
  const next = h.controller.stage(action());
  assert.notEqual(h.sends[1].request.requestId, old.request.requestId);
  h.sends[1].resolve(receipt(h.sends[1].request));
  await next;
});

test("same-session source rewinds block staging and retry while preserving the uncertain original request", async () => {
  const h = harness();
  const boundary = frame({ identity: { publicationRevision: "40", scientificRevision: "30", controlRevision: "10",
    configRevision: "4", tick: "100" }, snapshot: { epoch: "2" } });
  h.controller.observe(boundary, 1);
  const staged = h.controller.stage(action());
  const original = h.sends[0].request;
  h.sends[0].reject(new Error("response lost after possible admission"));
  await staged;
  for (const field of ["epoch", "publicationRevision", "scientificRevision", "controlRevision", "configRevision", "tick"]) {
    const rewound = structuredClone(boundary);
    const counters = field === "epoch" ? rewound.snapshot : rewound.identity;
    counters[field] = (BigInt(counters[field]) - 1n).toString();
    assert.equal(h.controller.reattachOwner(rewound, 3), false, `${field} regression must refuse reattachment`);
    assert.equal(h.controller.retained.request, original);
    assert.equal(h.controller.frame, boundary, "keep the earlier evidence needed to detect a rewind");
    assert.match(h.controller.attachmentError, new RegExp(field));
    h.controller.setEnabled(true);
    assert.equal(h.controller.enabled, false, "opening the toolbox cannot bypass source refusal");
    assert.equal(h.controller.canStage, false);
    await h.controller.retry();
    assert.equal(h.sends.length, 1, "an unsafe source cannot receive even an explicit retry");
  }
  assert.equal(h.controller.reattachOwner(boundary, 3), true);
  assert.equal(h.controller.attachmentError, null);
  assert.equal(h.sends.length, 1, "returning to a coherent boundary never resubmits automatically");
  h.controller.setEnabled(true);
  const retried = h.controller.retry();
  assert.equal(h.sends[1].request, original);
  h.sends[1].resolve(receipt(original, { source: boundary }));
  await retried;
  assert.equal(h.identities(), 1);
});

test("reattachment also guards the newer source already observed by an owner receipt", async () => {
  const h = harness(), staged = h.controller.stage(action());
  const original = h.sends[0].request;
  const accepted = receipt(original, { journal: "durable" });
  h.sends[0].resolve(accepted);
  await staged;
  const newerThanCanvas = frame({ identity: { publicationRevision: "21" } });
  assert.equal(h.controller.reattachOwner(newerThanCanvas, 3), false,
    "a fresh connection frame may still be older than its previous HTTP observation");
  assert.equal(h.controller.receipt, accepted);
  assert.equal(h.controller.retained.request, original);
  const coherent = frame({ identity: { publicationRevision: "22", controlRevision: "8", scientificRevision: "13" } });
  assert.equal(h.controller.reattachOwner(coherent, 3), true);
  h.controller.setEnabled(true);
  assert.equal(h.controller.canStage, false, "coherence does not turn a queued action into an ecological effect");
  assert.equal(h.sends.length, 1);
});

test("history requests are single-flight, publication-driven and stop when disabled", async () => {
  const h = harness();
  assert.equal(h.histories.length, 1);
  h.controller.observe(frame({ identity: { publicationRevision: "21" } }), 1);
  h.advance(300);
  h.controller.observe(frame({ identity: { publicationRevision: "22" } }), 1);
  assert.equal(h.histories.length, 1);
  h.histories[0].resolve(history(h.histories[0].request));
  await flush();
  h.controller.observe(frame({ identity: { publicationRevision: "22" } }), 1);
  assert.equal(h.histories.length, 2);
  const second = h.histories[1];
  second.resolve(history(second.request, { source: sourceFor(frame({ identity: { publicationRevision: "22" } })) }));
  await flush();
  h.controller.observe(frame({ identity: { publicationRevision: "23" } }), 1);
  assert.equal(h.histories.length, 2, "new publications still obey the four-Hz ceiling");
  h.advance(250);
  h.controller.observe(frame({ identity: { publicationRevision: "23" } }), 1);
  assert.equal(h.histories.length, 3);
  h.controller.setEnabled(false);
  assert.equal(h.histories[2].signal.aborted, true);
  h.advance(1000);
  h.controller.observe(frame({ identity: { publicationRevision: "24" } }), 1);
  assert.equal(h.histories.length, 3);
});

test("applied commands continue bounded receipt refresh while their journal is pending", async () => {
  const h = harness(), staged = h.controller.stage(action());
  const req = h.sends[0].request;
  h.sends[0].resolve(receipt(req, { journal: "pending" }));
  await staged;
  h.advance(250);
  h.controller.observe(frame({ identity: { publicationRevision: "22", controlRevision: "8", scientificRevision: "13" } }), 1);
  assert.equal(h.statuses.length, 1, "queue application must not stop journal observation");
  const committed = receipt(req, { journal: "committed_volatile" });
  committed.source.publicationRevision = "22";
  h.statuses[0].resolve(committed);
  await flush();
  assert.equal(h.controller.receipt.journalState, "committed_volatile");
  h.advance(250);
  h.controller.observe(frame({ identity: { publicationRevision: "23", controlRevision: "8", scientificRevision: "13" } }), 1);
  assert.equal(h.statuses.length, 2, "a real durable journal may upgrade its volatile receipt");
  const durable = structuredClone(committed); durable.journalState = "durable"; durable.source.publicationRevision = "23";
  h.statuses[1].resolve(durable);
  await flush();
  h.advance(1000);
  h.controller.observe(frame({ identity: { publicationRevision: "24", controlRevision: "8", scientificRevision: "13" } }), 1);
  assert.equal(h.statuses.length, 2, "a durable lifecycle needs no further polling");
});

test("manual history navigation retains its exact cursor and discards superseded page responses", async () => {
  const h = harness(), first = h.histories[0];
  const following = h.controller.historyPageAfter("9007199254740993");
  assert.equal(first.signal.aborted, true);
  assert.equal(h.histories[1].request.afterSeq, "9007199254740993");
  first.resolve(history(first.request));
  await flush();
  assert.equal(h.controller.historyPage, null);
  const next = h.histories[1];
  next.resolve(history(next.request, { earliestRetained: "9007199254740994", latestSeq: "9007199254740994",
    nextAfterSeq: "9007199254740994", records: [record("9007199254740994")] }));
  await following;
  assert.equal(h.controller.historyPage.nextAfterSeq, "9007199254740994");
  assert.throws(() => h.controller.historyPageAfter(9007199254740994));
});

function remoteClient(url = "wss://lab.example/nested/api/ws/stream?old=query#fragment") {
  const client = new RemoteStreamClient({ url, decodeFrame: () => {} });
  client.tracker.acknowledge(SESSION);
  client.state = "connected";
  return client;
}

// These are actual transport-module consumer tests with controlled fetch responses.
// Real HTTP listener/owner behavior is covered separately by the Rust server test.
test("remote intervention retries send identical JSON and use the negotiated server prefix", async () => {
  const client = remoteClient(), frozen = makeInterventionRequest(frame(), action(), "web-intervention-transport");
  const original = globalThis.fetch, requests = [];
  globalThis.fetch = async (url, options) => {
    requests.push({ url: String(url), ...options });
    if (requests.length === 1) throw new Error("lost response after possible admission");
    return new Response(JSON.stringify(receipt(frozen)), { headers: { "content-type": "application/json" } });
  };
  try {
    await assert.rejects(client.intervene(frozen), /possible admission/);
    const accepted = await client.intervene(frozen);
    assert.equal(accepted.commandId, COMMAND);
    assert.equal(requests.length, 2);
    assert.equal(requests[0].body, requests[1].body);
    assert.deepEqual(JSON.parse(requests[1].body), frozen);
    for (const call of requests) {
      assert.equal(call.url, "https://lab.example/nested/api/control/browser-intervention");
      assert.equal(call.method, "POST");
      assert.equal(call.cache, "no-store");
      assert.equal(call.credentials, "omit");
      assert.equal(call.headers["Content-Type"], "application/json");
      assert.ok(call.signal instanceof AbortSignal);
    }
  } finally { globalThis.fetch = original; }
});

test("remote submission bounds actual UTF-8 bytes at 4096 before fetching", async () => {
  const client = remoteClient(), frozen = makeInterventionRequest(frame(), action(), "web-intervention-size");
  const original = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return new Response("{}");
  };
  try {
    // Padding probes only the transport ceiling; the real strict endpoint separately
    // refuses unknown body fields rather than accepting this synthetic action.
    const sized = { ...frozen, padding: "" };
    const overhead = new TextEncoder().encode(JSON.stringify(sized)).length;
    sized.padding = "x".repeat(4096 - overhead);
    assert.equal(new TextEncoder().encode(JSON.stringify(sized)).length, 4096);
    await client.intervene(sized);
    assert.equal(calls, 1);
    sized.padding += "x";
    await assert.rejects(client.intervene(sized), /4 KiB/);
    assert.equal(calls, 1, "oversized requests cannot reach fetch");
    const unicode = { ...frozen, padding: "😀".repeat(1000) };
    assert.ok(JSON.stringify(unicode).length < 4096);
    assert.ok(new TextEncoder().encode(JSON.stringify(unicode)).length > 4096);
    await assert.rejects(client.intervene(unicode), /4 KiB/);
    assert.equal(calls, 1);
  } finally { globalThis.fetch = original; }
});

test("remote intervention reads preserve exact query identities and reject unnegotiated sources before fetch", async () => {
  const client = remoteClient(), original = globalThis.fetch, calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url: new URL(url), options });
    return new Response("{}");
  };
  try {
    await client.interventionStatus(COMMAND, { sessionId: SESSION });
    await client.interventionHistory({ sessionId: SESSION, epoch: "9007199254740993",
      afterSeq: "18446744073709551615", limit: 64 });
    assert.equal(calls[0].url.pathname, `/nested/api/control/browser-intervention/${COMMAND}`);
    assert.equal(calls[0].url.searchParams.get("session_id"), SESSION);
    assert.equal(calls[1].url.pathname, "/nested/api/interventions/browser");
    assert.equal(calls[1].url.searchParams.get("epoch"), "9007199254740993");
    assert.equal(calls[1].url.searchParams.get("after_seq"), "18446744073709551615");
    assert.equal(calls[1].url.searchParams.get("limit"), "64");
    assert.equal(calls[1].options.method, "GET");
    assert.equal(calls[1].options.body, undefined);
    assert.equal(calls[1].url.hash, "");
    await assert.rejects(client.interventionStatus(COMMAND.toUpperCase(), { sessionId: SESSION }));
    await assert.rejects(client.interventionStatus(COMMAND, { sessionId: "1" }));
    await assert.rejects(client.interventionHistory({ sessionId: SESSION, epoch: "00", afterSeq: "0" }));
    await assert.rejects(client.interventionHistory({ sessionId: SESSION, epoch: "0", afterSeq: 0 }));
    await assert.rejects(client.interventionHistory({ sessionId: SESSION, epoch: "0", afterSeq: "0", limit: 65 }));
    const wrongPath = remoteClient("wss://lab.example/nested/another-stream");
    await assert.rejects(wrongPath.intervene(makeInterventionRequest(frame(), action(), "valid-id")), /api\/ws\/stream/);
    client.state = "disconnected";
    await assert.rejects(client.interventionStatus(COMMAND, { sessionId: SESSION }), /Connect/);
    assert.equal(calls.length, 2);
  } finally { globalThis.fetch = original; }
});

test("only the exact missing-receipt 404 maps to unknown status", async () => {
  const client = remoteClient(), original = globalThis.fetch;
  const missing = "no retained command receipt; admission remains unknown";
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({ message: missing }), { status: 404 });
    assert.equal(await client.interventionStatus(COMMAND, { sessionId: SESSION }), null);
    await assert.rejects(client.intervene(makeInterventionRequest(frame(), action(), "valid-id")), /admission remains unknown/);
    await assert.rejects(client.interventionHistory({ sessionId: SESSION, epoch: "0", afterSeq: "0" }), /admission remains unknown/);
    globalThis.fetch = async () => new Response(JSON.stringify({ message: "proxy route is missing" }), { status: 404 });
    await assert.rejects(client.interventionStatus(COMMAND, { sessionId: SESSION }), /proxy route/);
    globalThis.fetch = async () => new Response(JSON.stringify({ message: missing }), { status: 503 });
    await assert.rejects(client.interventionStatus(COMMAND, { sessionId: SESSION }), /admission remains unknown/);
    globalThis.fetch = async () => new Response("not JSON", { status: 404 });
    await assert.rejects(client.interventionStatus(COMMAND, { sessionId: SESSION }));
  } finally { globalThis.fetch = original; }
});

test("receipt and history responses enforce streamed byte ceilings even when Content-Length lies", async () => {
  const client = remoteClient(), original = globalThis.fetch;
  const frozen = makeInterventionRequest(frame(), action(), "web-intervention-streamed");
  try {
    for (const [limit, operation] of [
      [16 * 1024, () => client.intervene(frozen)],
      [16 * 1024, () => client.interventionStatus(COMMAND, { sessionId: SESSION })],
      [64 * 1024, () => client.interventionHistory({ sessionId: SESSION, epoch: "0", afterSeq: "0" })],
    ]) {
      let cancelled = 0;
      globalThis.fetch = async () => new Response(new ReadableStream({
        start(controller) { controller.enqueue(new Uint8Array(limit)); controller.enqueue(new Uint8Array(1)); },
        cancel() { cancelled += 1; },
      }), { headers: { "content-length": "1" } });
      await assert.rejects(operation(), new RegExp(`${limit / 1024} KiB bound`));
      assert.equal(cancelled, 1);
      globalThis.fetch = async () => new Response(new ReadableStream({
        start(controller) { controller.enqueue(new TextEncoder().encode("{}")); },
        cancel() { cancelled += 1; },
      }), { headers: { "content-length": String(limit + 1) } });
      await assert.rejects(operation(), new RegExp(`${limit / 1024} KiB bound`));
      assert.equal(cancelled, 2, "an oversized advertised response is cancelled before reading");
    }
  } finally { globalThis.fetch = original; }
});

test("remote delivery forwards caller cancellation without changing the retained request", async () => {
  const client = remoteClient(), original = globalThis.fetch;
  const frozen = makeInterventionRequest(frame(), action(), "web-intervention-cancel");
  const abort = new AbortController();
  let observed;
  globalThis.fetch = async (_url, options) => {
    observed = options;
    return new Promise((_resolve, reject) => {
      const rejectAbort = () => reject(new DOMException("cancelled", "AbortError"));
      if (options.signal.aborted) rejectAbort();
      else options.signal.addEventListener("abort", rejectAbort, { once: true });
    });
  };
  try {
    const pending = client.intervene(frozen, { signal: abort.signal });
    assert.ok(observed);
    abort.abort();
    await assert.rejects(pending, { name: "AbortError" });
    assert.equal(observed.signal.aborted, true);
    assert.equal(observed.body, JSON.stringify(frozen));
    assert.ok(Object.isFrozen(frozen));
  } finally { globalThis.fetch = original; }
});
