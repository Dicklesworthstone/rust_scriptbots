import assert from "node:assert/strict";
import test from "node:test";
import { NarrativeController, NARRATIVE_ROW_LIMIT, renderNarrative, validateNarrativePage } from "../narrative.js";

const SESSION = "340282366920938463463374607431768211450";
const flush = () => new Promise((resolve) => setImmediate(resolve));

function identity(changes = {}) {
  return { sessionId: SESSION, publicationRevision: "24", scientificRevision: "12",
    controlRevision: "5", configRevision: "2", tick: "10", ...changes };
}

function record(sequence) {
  return { sequence: String(sequence), record: { schemaVersion: 1, tick: "10", kind: "extinction",
    humanText: "The last creature died.", metric: "population", severity: 1, magnitude: -2,
    window: ["1", "10"], before: 2, after: 0, score: 3.5, subject: null } };
}

function page({ after = "0", earliest = "1", latest = "2", source = identity() } = {}) {
  const start = BigInt(after) + 1n > BigInt(earliest) ? BigInt(after) + 1n : BigInt(earliest);
  const end = start + 64n < BigInt(latest) + 1n ? start + 64n : BigInt(latest) + 1n;
  const events = [];
  for (let ordinal = start; ordinal < end; ordinal++) events.push(record(ordinal));
  return { schema: "scriptbots.remote-frame.v2", kind: "events", identity: source,
    playback: { paused: true, speedMultiplier: 0 }, coalescedPublications: "0", environment: null,
    payload: { afterEvent: after, earliestRetained: earliest, latestEvent: latest,
      gap: BigInt(after) + 1n < BigInt(earliest)
        ? { firstMissing: String(BigInt(after) + 1n), lastMissing: String(BigInt(earliest) - 1n) } : null, events } };
}

function harness() {
  let time = 0;
  const requests = [];
  const controller = new NarrativeController({ now: () => time,
    read: (afterEvent, sessionId) => new Promise((resolve, reject) => requests.push({ afterEvent, sessionId, resolve, reject })) });
  return { controller, requests, advance: (delta = 250) => { time += delta; } };
}

test("local history pages preserve canonical scientific values and full-width identities", () => {
  const source = identity({ tick: "9007199254740993", publicationRevision: "9007199254741000" });
  const value = page({ source });
  value.payload.events[0].record.tick = source.tick;
  value.payload.events[0].record.window = ["9007199254740992", source.tick];
  assert.equal(validateNarrativePage(value), value);
  const controller = new NarrativeController();
  controller.accept(value);
  assert.equal(controller.sessionId, SESSION);
  assert.equal(controller.pageIdentity.tick, source.tick);
  assert.equal(controller.rows[0].record.tick, source.tick);
  assert.equal(controller.rows[0].record.humanText, "The last creature died.");
  value.payload.events[0].record.humanText = "changed after receipt";
  value.payload.events[0].record.window[0] = "0";
  assert.equal(controller.rows[0].record.humanText, "The last creature died.");
  assert.equal(controller.rows[0].record.window[0], "9007199254740992");
});

test("malformed narrative pages refuse atomically without adopting foreign authority or fabricated history", () => {
  for (const mutate of [
    (value) => { value.schema = "unknown"; },
    (value) => { value.identity.sessionId = "340282366920938463463374607431768211456"; },
    (value) => { value.identity.tick = 10; },
    (value) => { value.payload.earliestRetained = "01"; },
    (value) => { value.payload.latestEvent = "18446744073709551616"; },
    (value) => { value.payload.events[0].sequence = "2"; },
    (value) => { value.payload.events[0].record.tick = "11"; },
    (value) => { value.payload.events[0].record.humanText = "字".repeat(1366); },
    (value) => { value.payload.events[0].record.severity = NaN; },
    (value) => { value.payload.events[0].record.score = Infinity; },
    (value) => { value.payload.events[0].record.window = ["10", "9"]; },
    (value) => { value.payload.events.pop(); },
    (value) => { value.payload.gap = { firstMissing: "1", lastMissing: "1" }; },
  ]) {
    const controller = new NarrativeController();
    const invalid = page(); mutate(invalid);
    assert.throws(() => controller.accept(invalid));
    assert.equal(controller.sessionId, null);
    assert.equal(controller.afterEvent, "0");
    assert.deepEqual(controller.rows, []);
    controller.accept(page());
    assert.equal(controller.afterEvent, "2", "a rejected candidate must not poison the next valid source");
  }
});

test("a paused owner's backlog is paged four times per second and retained display stays bounded", async () => {
  const { controller, requests, advance } = harness();
  controller.observe({ identity: identity() }, 1);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].sessionId, SESSION);
  requests[0].resolve(page({ latest: "70" }));
  await flush();
  assert.equal(controller.afterEvent, "64");
  assert.equal(controller.rows.length, NARRATIVE_ROW_LIMIT);
  assert.equal(controller.rows[0].sequence, "17");
  for (let i = 0; i < 20; i++) controller.observe({ identity: identity() }, 1);
  assert.equal(requests.length, 1, "extra RAF presentations must not drain the backlog eagerly");
  advance(249);
  controller.observe({ identity: identity() }, 1);
  assert.equal(requests.length, 1);
  advance(1);
  controller.observe({ identity: identity() }, 1);
  assert.equal(requests.length, 2);
  assert.equal(requests[1].afterEvent, "64");
  requests[1].resolve(page({ after: "64", latest: "70" }));
  await flush();
  assert.equal(controller.afterEvent, "70");
  assert.equal(controller.rows[0].sequence, "23");
  assert.equal(controller.omittedRows, 22n);
  advance(10_000);
  controller.observe({ identity: identity() }, 1);
  assert.equal(requests.length, 2, "a completed unchanged publication is read only once");
});

test("explicit retention gaps retain exact large cursors and duplicate remote delivery adds nothing", () => {
  const controller = new NarrativeController();
  controller.reset("remote");
  const value = page({ earliest: "9007199254740994", latest: "9007199254740995" });
  assert.equal(controller.accept(value), true);
  assert.equal(controller.afterEvent, "9007199254740995");
  assert.equal(controller.missingEvents, 9007199254740993n);
  assert.deepEqual(controller.lastGap, { firstMissing: "1", lastMissing: "9007199254740993" });
  assert.equal(controller.accept(value), false);
  assert.equal(controller.rows.length, 2);
  assert.equal(controller.missingEvents, 9007199254740993n);
  const invalidGap = page({ after: controller.afterEvent, earliest: "9007199254741000", latest: "9007199254741001" });
  invalidGap.payload.gap.firstMissing = "9007199254740995";
  assert.throws(() => controller.accept(invalidGap), /exact gap/);
  assert.equal(controller.afterEvent, "9007199254740995");
});

test("hidden pages issue no local reads and retain only one actual request across replacement owners", async () => {
  const { controller, requests, advance } = harness();
  controller.setEnabled(false);
  controller.observe({ identity: identity() }, 1);
  assert.equal(requests.length, 0);
  controller.setEnabled(true);
  controller.observe({ identity: identity() }, 1);
  assert.equal(requests.length, 1);
  advance(5_000);
  controller.observe({ identity: identity({ publicationRevision: "25" }) }, 1);
  assert.equal(requests.length, 1, "an unresolved read cannot accumulate more requests");
  controller.reset("local");
  controller.observe({ identity: identity() }, 2);
  assert.equal(requests.length, 1, "owner replacement retains the in-flight request bound");
  requests[0].resolve(page());
  await flush();
  assert.equal(controller.rows.length, 0, "same-session continuation replacement rejects the old reply");
  controller.observe({ identity: identity() }, 2);
  assert.equal(requests.length, 2);
  requests[1].resolve(page());
  await flush();
  assert.equal(controller.rows.length, 2);
});

test("local query refusals stop repeated reads of one publication and explicit retry preserves cursor", async () => {
  const { controller, requests, advance } = harness();
  controller.observe({ identity: identity() }, 1);
  requests[0].reject(new Error("owner history unavailable"));
  await flush();
  advance(1_000);
  controller.observe({ identity: identity() }, 1);
  assert.equal(requests.length, 1);
  assert.match(controller.error, /owner history unavailable/);
  const retry = controller.refresh(true);
  assert.equal(requests.length, 2);
  assert.equal(requests[1].afterEvent, "0");
  requests[1].resolve(page());
  await retry;
  assert.equal(controller.error, null);
  assert.equal(controller.afterEvent, "2");
});

test("foreign or stale local responses leave scientific history unchanged", async () => {
  for (const source of [identity({ sessionId: "2" }), identity({ publicationRevision: "23" }), identity({ tick: "9" })]) {
    const { controller, requests } = harness();
    controller.observe({ identity: identity() }, 1);
    requests[0].resolve(page({ source }));
    await flush();
    assert.ok(controller.error);
    assert.equal(controller.afterEvent, "0");
    assert.equal(controller.rows.length, 0);
  }
});

test("a newer completed read does not poll again against an older presentation", async () => {
  const { controller, requests, advance } = harness();
  controller.observe({ identity: identity() }, 1);
  requests[0].resolve(page({ source: identity({ publicationRevision: "25" }) }));
  await flush();
  advance(1_000);
  controller.observe({ identity: identity() }, 1);
  assert.equal(requests.length, 1);
  assert.equal(controller.pageIdentity.publicationRevision, "25");
  assert.equal(controller.frameIdentity.publicationRevision, "24");
});

test("remote reconnect preserves consumed history while actual source switches clear it", () => {
  let reads = 0;
  const controller = new NarrativeController({ read: () => { reads += 1; } });
  controller.reset("remote");
  controller.accept(page());
  controller.observe({ identity: identity() }, 1);
  controller.setConnection("Server disconnected; retained history only.");
  assert.equal(controller.rows.length, 2);
  controller.setConnection(null);
  controller.accept(page({ after: "2", latest: "3", source: identity({ publicationRevision: "25" }) }));
  assert.equal(controller.afterEvent, "3");
  assert.equal(reads, 0, "remote history comes exclusively from its existing credited stream");
  controller.reset("local");
  assert.equal(controller.afterEvent, "0");
  assert.equal(controller.rows.length, 0);
  controller.setEnabled(false);
  controller.observe({ identity: identity({ sessionId: "7" }) }, 2);
  assert.equal(controller.sessionId, "7");
  assert.equal(reads, 0);
});

function panel() {
  function element() {
    return { textContent: "", dataset: {}, children: [], scrollHeight: 0, scrollTop: 0, clientHeight: 0,
      get lastElementChild() { return this.children.at(-1); },
      replaceChildren() { this.children = []; }, append(child) { this.children.push(child); } };
  }
  const fields = new Map(["status", "source", "retention", "gap", "refresh"].map((id) => [`#narrative-${id}`, element()]));
  fields.set("#remote-events", element());
  return { dataset: {}, ownerDocument: { createElement: element }, querySelector: (key) => fields.get(key) };
}

test("timeline renderer names its exact observation, preserves canonical text, and distinguishes gaps from display retention", () => {
  const view = panel();
  const controller = new NarrativeController();
  const value = page({ earliest: "3", latest: "52" });
  value.payload.events.at(-1).record.humanText = "<script>canonical text is never HTML</script>";
  controller.accept(value);
  controller.frameIdentity = identity({ publicationRevision: "26", tick: "12" });
  renderNarrative(view, controller);
  const list = view.querySelector("#remote-events");
  assert.equal(list.children.length, 48);
  assert.equal(list.children.at(-1).textContent, "#52 · tick 10: <script>canonical text is never HTML</script>");
  assert.match(view.querySelector("#narrative-source").textContent, /observed tick 10, publication 24/);
  assert.match(view.querySelector("#narrative-source").textContent, /Latest world observation: tick 12, publication 26/);
  assert.match(view.querySelector("#narrative-gap").textContent, /events 1–2 are no longer retained/);
  assert.match(view.querySelector("#narrative-retention").textContent, /2 older viewed events/);
  assert.match(view.querySelector("#narrative-retention").textContent, /does not confirm durable/);
  const last = list.lastElementChild;
  renderNarrative(view, controller);
  assert.equal(list.lastElementChild, last, "status updates must not rebuild unchanged scientific rows");
  controller.reset();
  controller.accept(page({ latest: "0" }));
  renderNarrative(view, controller);
  assert.equal(list.children.length, 0);
  assert.match(view.querySelector("#narrative-status").textContent, /No scientific narrative events/);
  assert.match(view.querySelector("#narrative-retention").textContent, /Source retains no events/);
});

test("identical event ordinals from a replacement session never reuse old-owner prose", () => {
  const view = panel();
  const controller = new NarrativeController();
  controller.accept(page());
  renderNarrative(view, controller);
  const prior = view.querySelector("#remote-events").lastElementChild;
  // Do not render the empty reset in this test: source identity itself must
  // invalidate reuse even when a render consumer coalesces the intermediate state.
  controller.reset();
  const next = page({ source: identity({ sessionId: "2" }) });
  next.payload.events.at(-1).record.humanText = "A different world's canonical narrative.";
  controller.accept(next);
  renderNarrative(view, controller);
  assert.notEqual(view.querySelector("#remote-events").lastElementChild, prior);
  assert.match(view.querySelector("#remote-events").lastElementChild.textContent, /different world's/);
  assert.equal(view.dataset.sessionId, "2");
});
