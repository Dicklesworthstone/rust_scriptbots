// Production storage adapter against a controlled IndexedDB event double.
// These tests prove acknowledgement/error/bounds logic, not actual browser
// storage or checkpoint validity. A separate real-WASM Playwright journey
// exercises reload and checkpoint validation through actual browser storage.
import assert from "node:assert/strict";
import test from "node:test";
import { BrowserContinuationStore, BrowserRecoveryController, ContinuationStoreError } from "../continuation-store.js";

const SAVED_AT = "2026-11-06T00:00:00.000Z";
const record = (overrides = {}) => ({ schema: "scriptbots.browser-continuation.v1",
  tick: "7", savedAt: SAVED_AT, blob: new Blob([new Uint8Array([1, 2, 3])]), ...overrides });
const codeIs = (code) => (error) => error instanceof ContinuationStoreError && error.code === code;

class IndexedDbEvents {
  constructor(stored) {
    this.stored = stored;
    this.recovery = undefined;
    this.opens = 0;
    this.closes = 0;
    this.transactions = [];
    this.waiters = [];
    this.ready = [];
  }

  open() {
    this.opens += 1;
    if (this.openError) throw this.openError;
    const factory = this;
    const database = {
      createObjectStore(name) { assert.equal(name, "continuations"); },
      close() { factory.closes += 1; },
      transaction(name, mode) {
        assert.equal(name, "continuations");
        const transaction = {
          error: null,
          candidate: undefined,
          slot: null,
          request: null,
          objectStore() {
            const issue = (result, candidate) => {
              if (factory.operationError) throw factory.operationError;
              transaction.candidate = candidate;
              transaction.request = { result, error: null };
              const waiter = factory.waiters.shift();
              if (waiter) waiter(transaction);
              else factory.ready.push(transaction);
              return transaction.request;
            };
            return {
              get(key) {
                assert.ok(["latest", "recovery"].includes(key));
                return issue(key === "latest" ? factory.stored : factory.recovery);
              },
              put(value, key) {
                assert.ok(["latest", "recovery"].includes(key));
                transaction.slot = key;
                return issue(key, structuredClone(value));
              },
            };
          },
          succeed() { transaction.request.onsuccess(); },
          complete() {
            if (mode === "readwrite") {
              if (transaction.slot === "latest") factory.stored = transaction.candidate;
              else factory.recovery = transaction.candidate;
            }
            transaction.oncomplete();
          },
          abort(error = new DOMException("Controlled abort", "AbortError")) {
            transaction.error = error;
            queueMicrotask(() => { transaction.onerror?.(); transaction.onabort(); });
          },
        };
        factory.transactions.push(transaction);
        return transaction;
      },
    };
    const request = { result: database, error: null, transaction: { abort() { factory.upgradeAborted = true; } } };
    this.lastOpen = request;
    queueMicrotask(() => {
      if (this.stallOpen) return;
      if (this.blocked) request.onblocked();
      else {
        request.onupgradeneeded();
        request.onsuccess();
      }
    });
    return request;
  }

  nextTransaction() {
    return this.ready.length ? Promise.resolve(this.ready.shift())
      : new Promise((resolve) => this.waiters.push(resolve));
  }
}

async function finishRead(store, factory) {
  const promise = store.read();
  const transaction = await factory.nextTransaction();
  transaction.succeed();
  transaction.complete();
  return promise;
}

test("saving waits for transaction completion and retains exact large tick metadata", async () => {
  const previous = record();
  const factory = new IndexedDbEvents(previous);
  const store = new BrowserContinuationStore(8, factory);
  let acknowledged = false;
  const save = store.save(new Blob([new Uint8Array([9, 8])]), "9007199254740993", SAVED_AT)
    .then((metadata) => { acknowledged = true; return metadata; });
  const transaction = await factory.nextTransaction();
  transaction.succeed();
  await Promise.resolve();
  assert.equal(acknowledged, false, "request success is not a save acknowledgement");
  assert.equal(factory.stored, previous);
  transaction.complete();
  assert.deepEqual(await save, { tick: "9007199254740993", savedAt: SAVED_AT, byteLength: 2 });
  assert.equal(factory.closes, 1);
  const restored = await finishRead(new BrowserContinuationStore(8, factory), factory);
  assert.equal(restored.tick, "9007199254740993");
  assert.deepEqual(new Uint8Array(await restored.blob.arrayBuffer()), new Uint8Array([9, 8]));
  assert.equal(factory.closes, 2);
});

test("quota failure after request success preserves the prior slot and permits retry", async () => {
  const previous = record();
  const factory = new IndexedDbEvents(previous);
  const store = new BrowserContinuationStore(8, factory);
  const save = store.save(new Blob(["new"]), "8", SAVED_AT);
  const refused = assert.rejects(save, codeIs("quota"));
  const transaction = await factory.nextTransaction();
  transaction.succeed();
  transaction.abort(new DOMException("Controlled quota failure", "QuotaExceededError"));
  await refused;
  assert.equal(factory.stored, previous);
  assert.equal(factory.closes, 1);
  assert.equal(await finishRead(store, factory), previous);
});

test("one operation is admitted until abort settles, without a hidden work queue", async () => {
  const factory = new IndexedDbEvents(record());
  const store = new BrowserContinuationStore(8, factory);
  const read = store.read();
  const refused = assert.rejects(read, codeIs("aborted"));
  const transaction = await factory.nextTransaction();
  await assert.rejects(store.save(new Blob(["x"]), "9", SAVED_AT), codeIs("busy"));
  assert.equal(factory.opens, 1);
  transaction.abort();
  await refused;
  assert.equal((await finishRead(store, factory)).tick, "7");
});

test("metadata discovery does not materialize checkpoint bytes and absent records stay absent", async () => {
  const factory = new IndexedDbEvents(record());
  const store = new BrowserContinuationStore(8, factory);
  const arrayBuffer = Blob.prototype.arrayBuffer;
  Blob.prototype.arrayBuffer = () => { throw new Error("metadata discovery read the payload"); };
  try {
    const found = await finishRead(store, factory);
    assert.equal(found.tick, "7");
    assert.equal(found.blob.size, 3);
    factory.stored = undefined;
    assert.equal(await finishRead(store, factory), null);
  } finally {
    Blob.prototype.arrayBuffer = arrayBuffer;
  }
});

test("stored schema, exact integer, timestamp, Blob type and byte bounds fail closed", async () => {
  for (const [mutation, code] of [
    [{ schema: "scriptbots.browser-continuation.v2" }, "schema"],
    [{ tick: "07" }, "metadata"], [{ tick: "18446744073709551616" }, "metadata"],
    [{ tick: 7 }, "metadata"], [{ savedAt: "2026-02-30T00:00:00.000Z" }, "metadata"],
    [{ blob: new Uint8Array([1, 2]) }, "payload"], [{ blob: new Blob([]) }, "payload"],
    [{ blob: new Blob([new Uint8Array(9)]) }, "oversized"],
  ]) {
    const factory = new IndexedDbEvents(record(mutation));
    await assert.rejects(finishRead(new BrowserContinuationStore(8, factory), factory), codeIs(code));
    assert.equal(factory.transactions.length, 1);
    assert.equal(factory.closes, 1);
    assert.deepEqual(factory.stored, record(mutation), "refused records are not silently repaired");
  }
});

test("invalid saves are refused before opening storage", async () => {
  const factory = new IndexedDbEvents(record());
  const store = new BrowserContinuationStore(8, factory);
  await assert.rejects(store.save(new Blob([new Uint8Array(9)]), "7", SAVED_AT), codeIs("oversized"));
  await assert.rejects(store.save(new Blob(["x"]), "-1", SAVED_AT), codeIs("metadata"));
  assert.equal(factory.opens, 0);
  assert.throws(() => new BrowserContinuationStore(0, factory), codeIs("bounds"));
});

test("blocked opens refuse immediately and close or abort a late connection", async () => {
  const factory = new IndexedDbEvents(record());
  factory.blocked = true;
  const store = new BrowserContinuationStore(8, factory);
  await assert.rejects(store.read(), codeIs("blocked"));
  await assert.rejects(store.read(), codeIs("blocked"));
  assert.equal(factory.opens, 1, "retries cannot accumulate uncancellable blocked open requests");
  factory.lastOpen.onupgradeneeded();
  assert.equal(factory.upgradeAborted, true);
  factory.lastOpen.onsuccess();
  assert.equal(factory.closes, 1);
  assert.equal(factory.transactions.length, 0);
  factory.blocked = false;
  assert.equal((await finishRead(store, factory)).tick, "7");
});

test("denied storage and synchronous operation errors release ownership", async () => {
  const factory = new IndexedDbEvents(record());
  factory.openError = new DOMException("Controlled privacy refusal", "SecurityError");
  const store = new BrowserContinuationStore(8, factory);
  await assert.rejects(store.read(), codeIs("unavailable"));
  factory.openError = null;
  factory.operationError = new DOMException("Controlled clone refusal", "DataCloneError");
  await assert.rejects(store.save(new Blob(["x"]), "9", SAVED_AT), codeIs("storage"));
  assert.equal(factory.stored.tick, "7");
  assert.equal(factory.closes, 1);
  factory.operationError = null;
  assert.equal((await finishRead(store, factory)).tick, "7");
});

const source = (overrides = {}) => ({ sessionId: "340282366920938463463374607431768211455",
  scientificRevision: "9007199254740993", controlRevision: "21", configRevision: "5", tick: "9007199254740993", ...overrides });

async function finishRecoveryRead(store, factory) {
  const read = store.readRecovery();
  const transaction = await factory.nextTransaction();
  transaction.succeed();
  transaction.complete();
  return read;
}

test("recovery is a separately versioned slot and waits for actual commit", async () => {
  const manual = record();
  const factory = new IndexedDbEvents(manual);
  const store = new BrowserContinuationStore(8, factory);
  assert.equal(await finishRecoveryRead(store, factory), null);
  let acknowledged = false;
  const save = store.saveRecovery(new Blob(["next"]), source(), { savedAt: SAVED_AT })
    .then((value) => { acknowledged = true; return value; });
  const transaction = await factory.nextTransaction();
  transaction.succeed();
  await Promise.resolve();
  assert.equal(acknowledged, false);
  assert.equal(factory.recovery, undefined);
  transaction.complete();
  assert.deepEqual(await save, { tick: source().tick, savedAt: SAVED_AT, byteLength: 4, source: source() });
  assert.equal(factory.stored, manual, "automatic recovery must not replace the manual checkpoint");
  const found = await finishRecoveryRead(store, factory);
  assert.equal(found.schema, "scriptbots.browser-recovery.v1");
  assert.equal(found.source.sessionId, source().sessionId);
  assert.equal(await found.blob.text(), "next");

  const failed = store.saveRecovery(new Blob(["bad"]), source({ tick: "9007199254740994" }));
  const refused = assert.rejects(failed, codeIs("quota"));
  const quota = await factory.nextTransaction();
  quota.succeed();
  quota.abort(new DOMException("Controlled quota failure", "QuotaExceededError"));
  await refused;
  assert.equal(factory.recovery, found);
  assert.equal(factory.stored, manual);
});

test("recovery validates source integers and retained bytes before restore allocation", async () => {
  const factory = new IndexedDbEvents(record());
  const store = new BrowserContinuationStore(8, factory);
  for (const mutation of [{ tick: 1 }, { sessionId: "340282366920938463463374607431768211456" },
    { scientificRevision: "01" }, { controlRevision: "18446744073709551616" }, { configRevision: "-1" }]) {
    await assert.rejects(store.saveRecovery(new Blob(["x"]), source(mutation)), codeIs("source"));
  }
  assert.equal(factory.opens, 0);
  const base = { schema: "scriptbots.browser-recovery.v1", tick: source().tick, savedAt: SAVED_AT,
    blob: new Blob(["x"]), source: source() };
  for (const [mutation, code] of [[{ schema: "scriptbots.browser-recovery.v2" }, "schema"],
    [{ tick: "7" }, "source"], [{ source: null }, "source"], [{ blob: new Blob([new Uint8Array(9)]) }, "oversized"]]) {
    factory.recovery = { ...base, ...mutation };
    await assert.rejects(finishRecoveryRead(store, factory), codeIs(code));
    assert.equal(factory.stored.tick, "7");
  }
});

test("cancelling an opening recovery releases the operation and shares its single pending open", async () => {
  const manual = record();
  const factory = new IndexedDbEvents(manual);
  factory.stallOpen = true;
  const store = new BrowserContinuationStore(8, factory);
  const abort = new AbortController();
  const save = store.saveRecovery(new Blob(["new"]), source(), { signal: abort.signal });
  const refused = assert.rejects(save, codeIs("aborted"));
  abort.abort();
  await refused;
  assert.equal(factory.transactions.length, 0);
  const read = store.read();
  assert.equal(factory.opens, 1, "manual retry reuses the pending native open instead of leaking another request");
  factory.lastOpen.onsuccess();
  const transaction = await factory.nextTransaction();
  transaction.succeed();
  transaction.complete();
  assert.equal(await read, manual);
  assert.equal(factory.recovery, undefined);
  assert.equal(factory.closes, 1);
});

test("a cancelled open with no waiter closes its late result and admits later work", async () => {
  const factory = new IndexedDbEvents(record());
  factory.stallOpen = true;
  const store = new BrowserContinuationStore(8, factory);
  const abort = new AbortController();
  const save = store.saveRecovery(new Blob(["new"]), source(), { signal: abort.signal });
  const refused = assert.rejects(save, codeIs("aborted"));
  abort.abort();
  await refused;
  factory.lastOpen.onsuccess();
  assert.equal(factory.closes, 1);
  assert.equal(factory.transactions.length, 0);
  factory.stallOpen = false;
  assert.equal((await finishRead(store, factory)).tick, "7");
  assert.equal(factory.opens, 2);
});

test("cancellation after request success aborts recovery before acknowledgement", async () => {
  const factory = new IndexedDbEvents(record());
  const store = new BrowserContinuationStore(8, factory);
  const abort = new AbortController();
  const save = store.saveRecovery(new Blob(["new"]), source(), { signal: abort.signal });
  const refused = assert.rejects(save, codeIs("aborted"));
  const transaction = await factory.nextTransaction();
  transaction.succeed();
  abort.abort();
  await refused;
  assert.equal(factory.recovery, undefined);
  assert.equal(factory.stored.tick, "7");
  assert.equal((await finishRead(store, factory)).tick, "7");
});

function recoveryHarness() {
  const factory = new IndexedDbEvents(record());
  const saved = [];
  let now = 0;
  let current = source();
  let captures = 0;
  const controller = new BrowserRecoveryController(new BrowserContinuationStore(16, factory), {
    now: () => now, onSaved: (value) => saved.push(value),
  });
  const owner = {};
  controller.attach(owner);
  return { factory, saved, controller, owner,
    time: (value) => { now = value; },
    source: (value) => { current = value; },
    captures: () => captures,
    poll: () => controller.poll(() => current, () => { captures += 1; return new Blob([String(captures)]); }),
    finish: async () => {
      const transaction = await factory.nextTransaction();
      transaction.succeed();
      transaction.complete();
    },
  };
}

test("recovery requires opt-in, bounds in-flight work and coalesces to the newest owner cut", async () => {
  const h = recoveryHarness();
  assert.equal(await h.poll(), false);
  assert.equal(h.captures(), 0);
  assert.equal(h.factory.opens, 0);
  h.controller.setEnabled(true);
  const first = h.poll();
  assert.equal(h.controller.pending, true);
  h.source(source({ tick: "9007199254740998", scientificRevision: "9007199254740998" }));
  h.time(60_000);
  for (let index = 0; index < 100; index += 1) assert.equal(await h.poll(), false);
  assert.equal(h.captures(), 1);
  assert.equal(h.saved.length, 0);
  await h.finish();
  assert.equal(await first, true);
  assert.equal(h.saved[0].tick, "9007199254740993");
  const latest = h.poll();
  await h.finish();
  assert.equal(await latest, true);
  assert.equal(h.saved[1].tick, "9007199254740998");
  assert.equal(h.captures(), 2);
  assert.equal(h.factory.stored.tick, "7");
});

test("unchanged paused states skip writes but same-tick configuration changes are recovered", async () => {
  const h = recoveryHarness();
  h.controller.setEnabled(true);
  const first = h.poll();
  await h.finish();
  await first;
  h.time(29_999);
  assert.equal(await h.poll(), false);
  h.time(30_000);
  assert.equal(await h.poll(), false, "paused publication did not change any restorable state");
  assert.equal(h.captures(), 1);
  h.source(source({ configRevision: "6", scientificRevision: "9007199254740994", controlRevision: "22" }));
  h.time(60_000);
  h.controller.setAvailable(false);
  assert.equal(await h.poll(), false, "hidden pages and manual storage defer new capture");
  h.controller.setAvailable(true);
  const changed = h.poll();
  await h.finish();
  assert.equal(await changed, true);
  assert.equal(h.saved[1].tick, h.saved[0].tick);
  assert.equal(h.saved[1].source.configRevision, "6");
  assert.equal(h.saved[1].source.controlRevision, "22");
});

test("owner replacement cancels a pending recovery and requires explicit opt-in again", async () => {
  const h = recoveryHarness();
  h.controller.setEnabled(true);
  const pending = h.poll();
  const transaction = await h.factory.nextTransaction();
  transaction.succeed();
  h.controller.attach({});
  assert.equal(h.controller.enabled, false);
  assert.equal(await h.poll(), false);
  assert.equal(await pending, false);
  assert.equal(h.saved.length, 0);
  assert.equal(h.factory.recovery, undefined);
  assert.equal(h.controller.error, null);
  h.controller.setEnabled(true);
  h.source(source({ sessionId: "7", tick: "0", scientificRevision: "0", controlRevision: "0", configRevision: "0" }));
  const restarted = h.poll();
  await h.finish();
  assert.equal(await restarted, true);
  assert.equal(h.saved[0].source.sessionId, "7");
  h.controller.attach(null);
  assert.throws(() => h.controller.setEnabled(true), codeIs("source"));
});

test("recovery storage faults latch without recapturing until the user retries", async () => {
  const h = recoveryHarness();
  h.controller.setEnabled(true);
  const failed = h.poll();
  const transaction = await h.factory.nextTransaction();
  transaction.succeed();
  transaction.abort(new DOMException("Controlled quota failure", "QuotaExceededError"));
  assert.equal(await failed, false);
  assert.equal(h.controller.error.code, "quota");
  h.time(90_000);
  assert.equal(await h.poll(), false);
  assert.equal(h.captures(), 1);
  assert.equal(h.saved.length, 0);
  h.controller.retry();
  const retried = h.poll();
  await h.finish();
  assert.equal(await retried, true);
  assert.equal(h.saved.length, 1);
  h.source(source({ controlRevision: "20" }));
  h.time(120_000);
  assert.equal(await h.poll(), false);
  assert.equal(h.controller.error.code, "source");
  assert.equal(h.captures(), 2, "regressing identity is refused before encoding another checkpoint");
});

test("capture must retain the same source and disabled recovery cannot acknowledge a late commit", async () => {
  let current = source();
  let writes = 0;
  let committed;
  const saved = [];
  const controller = new BrowserRecoveryController({ saveRecovery: (_blob, observed) => {
    writes += 1;
    // Models a transaction already beyond the cancellable stage. Completion
    // still must not be presented as a save for a newly selected owner.
    return new Promise((resolve) => { committed = () => resolve({ tick: observed.tick, source: observed }); });
  } }, { onSaved: (value) => saved.push(value) });
  controller.attach({});
  controller.setEnabled(true);
  assert.equal(await controller.poll(() => current, () => {
    current = source({ controlRevision: "22" });
    return new Blob(["x"]);
  }), false);
  assert.equal(controller.error.code, "source");
  assert.equal(writes, 0);
  controller.retry();
  const pending = controller.poll(() => current, () => new Blob(["x"]));
  controller.setEnabled(false);
  committed();
  assert.equal(await pending, false);
  assert.deepEqual(saved, []);
  assert.equal(controller.pending, false);
});
