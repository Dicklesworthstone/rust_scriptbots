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
    this.creates = 0;
    this.storeExists = stored !== undefined;
    this.transactions = [];
    this.waiters = [];
    this.ready = [];
    this.active = null;
    this.queued = [];
  }

  open(name, version) {
    assert.equal(name, "scriptbots-continuations");
    assert.equal(version, 2, "updated clients must fence old unconditional version-one writers");
    this.opens += 1;
    if (this.openError) throw this.openError;
    const factory = this;
    const database = {
      objectStoreNames: { contains(name) { assert.equal(name, "continuations"); return factory.storeExists; } },
      createObjectStore(name) {
        assert.equal(name, "continuations");
        assert.equal(factory.storeExists, false, "upgrades preserve the existing object store");
        factory.storeExists = true;
        factory.creates += 1;
      },
      close() { factory.closes += 1; },
      transaction(name, mode) {
        assert.equal(name, "continuations");
        const transaction = {
          mode,
          error: null,
          candidates: new Map(),
          requests: [],
          notified: false,
          finished: false,
          request: null,
          objectStore() {
            const issue = (operation) => {
              if (factory.operationError) throw factory.operationError;
              transaction.request = { result: undefined, error: null, operation };
              transaction.requests.push(transaction.request);
              factory.admit();
              return transaction.request;
            };
            return {
              get(key) {
                assert.ok(["latest", "recovery"].includes(key));
                return issue(() => transaction.candidates.has(key) ? transaction.candidates.get(key)
                  : key === "latest" ? factory.stored : factory.recovery);
              },
              put(value, key) {
                assert.ok(["latest", "recovery"].includes(key));
                assert.equal(mode, "readwrite");
                if (factory.writeError) throw factory.writeError;
                const copied = structuredClone(value);
                return issue(() => { transaction.candidates.set(key, copied); return key; });
              },
            };
          },
          succeed() {
            assert.equal(factory.active, transaction, "overlapping transactions cannot read stale committed state");
            while (transaction.requests.length && !transaction.finished) {
              const request = transaction.requests.shift();
              request.result = request.operation();
              request.onsuccess();
            }
          },
          complete() {
            assert.equal(factory.active, transaction);
            assert.equal(transaction.finished, false);
            assert.equal(transaction.requests.length, 0, "commit must wait for every request");
            for (const [slot, candidate] of transaction.candidates) {
              if (slot === "latest") factory.stored = candidate;
              else factory.recovery = candidate;
            }
            transaction.finished = true;
            transaction.oncomplete();
            factory.active = null;
            factory.admit();
          },
          abort(error = new DOMException("Controlled abort", "AbortError")) {
            if (transaction.finished) throw new DOMException("Transaction is finished", "InvalidStateError");
            transaction.finished = true;
            transaction.error = error;
            transaction.deliverAbort = () => {
              transaction.onerror?.();
              transaction.onabort();
              if (factory.active === transaction) factory.active = null;
              factory.queued = factory.queued.filter((pending) => pending !== transaction);
              factory.admit();
            };
            if (!factory.stallAbort) queueMicrotask(transaction.deliverAbort);
          },
        };
        factory.transactions.push(transaction);
        factory.queued.push(transaction);
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

  // Serialize this single-store test backend. In IndexedDB, overlapping
  // readwrite transactions wait for earlier transactions to finish before
  // their requests can observe or modify the store.
  admit() {
    this.active ??= this.queued.shift() ?? null;
    if (!this.active || this.active.notified || this.active.requests.length === 0) return;
    this.active.notified = true;
    const waiter = this.waiters.shift();
    if (waiter) waiter(this.active);
    else this.ready.push(this.active);
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

test("storage version refusal requests reload and upgrades preserve existing records", async () => {
  const previous = record();
  const factory = new IndexedDbEvents(previous);
  const store = new BrowserContinuationStore(8, factory);
  factory.openError = new DOMException("Requested version is lower than the current version", "VersionError");
  await assert.rejects(store.read(), (error) => codeIs("version")(error) && error.message.includes("Reload"));
  assert.equal(factory.stored, previous);
  factory.openError = null;
  assert.equal(await finishRead(store, factory), previous);
  assert.equal(factory.creates, 0, "an existing store is never recreated during upgrade");
  const empty = new IndexedDbEvents(undefined);
  assert.equal(await finishRead(new BrowserContinuationStore(8, empty), empty), null);
  assert.equal(empty.creates, 1);
});

const source = (overrides = {}) => ({ sessionId: "340282366920938463463374607431768211455",
  scientificRevision: "9007199254740993", controlRevision: "21", configRevision: "5", tick: "9007199254740993", ...overrides });

async function finishRecoveryRead(store, factory, options) {
  const read = store.readRecovery(options);
  const transaction = await factory.nextTransaction();
  transaction.succeed();
  if (!transaction.finished) transaction.complete();
  return read;
}

async function finishRecoverySave(store, factory, blob, observed = source(), options = {}) {
  const saved = store.saveRecovery(blob, observed, { savedAt: SAVED_AT, ...options });
  const transaction = await factory.nextTransaction();
  transaction.succeed();
  if (!transaction.finished) transaction.complete();
  return saved;
}

test("recovery is a separately versioned slot and waits for actual commit", async () => {
  const manual = record();
  const factory = new IndexedDbEvents(manual);
  const store = new BrowserContinuationStore(8, factory);
  assert.equal(store.recoveryObserved, false);
  assert.equal(await finishRecoveryRead(store, factory), null);
  assert.equal(store.recoveryObserved, true);
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
    [{ tick: "7" }, "source"], [{ source: null }, "source"], [{ blob: new Blob([new Uint8Array(9)]) }, "oversized"],
    [{ writeToken: null }, "metadata"], [{ writeToken: "0".repeat(31) }, "metadata"],
    [{ writeToken: "A".repeat(32) }, "metadata"]]) {
    factory.recovery = { ...base, ...mutation };
    await assert.rejects(finishRecoveryRead(store, factory), codeIs(code));
    assert.equal(factory.stored.tick, "7");
  }
});

test("two tabs cannot replace a recovery slot they have not observed or no longer own", async () => {
  const factory = new IndexedDbEvents(record());
  const first = new BrowserContinuationStore(16, factory);
  const second = new BrowserContinuationStore(16, factory);
  await assert.rejects(first.saveRecovery(new Blob(["first"]), source()), codeIs("unobserved"));
  assert.equal(factory.opens, 0, "an unobserved save cannot allocate a storage operation");
  await finishRecoveryRead(first, factory);
  await finishRecoveryRead(second, factory);

  const firstSave = first.saveRecovery(new Blob(["first"]), source(), { savedAt: SAVED_AT });
  const secondSave = second.saveRecovery(new Blob(["second"]), source({ sessionId: "7" }), { savedAt: SAVED_AT });
  const secondRefusal = assert.rejects(secondSave, codeIs("conflict"));
  const winning = await factory.nextTransaction();
  winning.succeed();
  assert.equal(factory.recovery, undefined, "request success cannot claim the shared slot");
  winning.complete();
  await firstSave;
  const retained = factory.recovery;
  assert.match(retained.writeToken, /^[0-9a-f]{32}$/);

  const stale = await factory.nextTransaction();
  stale.succeed();
  await secondRefusal;
  assert.equal(factory.recovery, retained);
  assert.equal(await retained.blob.text(), "first");
  assert.equal(first.recoveryObserved, true);
  assert.equal(second.recoveryObserved, false);
  const transactions = factory.transactions.length;
  await assert.rejects(second.saveRecovery(new Blob(["second"]), source({ sessionId: "7" })), codeIs("unobserved"));
  assert.equal(factory.transactions.length, transactions, "a conflict requires a separate explicit read");

  assert.equal((await finishRecoveryRead(second, factory)).writeToken, retained.writeToken);
  await finishRecoverySave(second, factory, new Blob(["second"]), source({ sessionId: "7" }));
  const replacement = factory.recovery;
  assert.notEqual(replacement.writeToken, retained.writeToken);
  assert.equal(await replacement.blob.text(), "second");
  await assert.rejects(finishRecoverySave(first, factory, new Blob(["stale-first"])), codeIs("conflict"));
  assert.equal(factory.recovery, replacement, "an explicit takeover also fences the former writer");
  assert.equal(factory.stored.tick, "7", "the manual save is outside the recovery CAS");
});

test("recovery write identities reject ABA even when every previous metadata field and byte returns", async () => {
  const factory = new IndexedDbEvents(record());
  const writer = new BrowserContinuationStore(16, factory);
  const observer = new BrowserContinuationStore(16, factory);
  await finishRecoveryRead(writer, factory);
  await finishRecoverySave(writer, factory, new Blob(["original"]));
  const observed = await finishRecoveryRead(observer, factory);
  await finishRecoverySave(writer, factory, new Blob(["intervening"]), source({ sessionId: "7" }));
  await finishRecoverySave(writer, factory, new Blob(["original"]));
  const current = factory.recovery;
  assert.deepEqual(current.source, observed.source);
  assert.equal(current.schema, observed.schema);
  assert.equal(current.savedAt, observed.savedAt);
  assert.equal(current.tick, observed.tick);
  assert.equal(await current.blob.text(), await observed.blob.text());
  assert.notEqual(current.writeToken, observed.writeToken);
  await assert.rejects(finishRecoverySave(observer, factory, new Blob(["stale"])), codeIs("conflict"));
  assert.equal(factory.recovery, current);
});

test("legacy recovery adoption preserves bytes and metadata and shares one atomic identity", async () => {
  const factory = new IndexedDbEvents(record());
  const legacy = { schema: "scriptbots.browser-recovery.v1", tick: source().tick,
    savedAt: SAVED_AT, blob: new Blob([new Uint8Array([0, 255, 1, 128])]), source: source(), retained: "metadata" };
  factory.recovery = legacy;
  const first = new BrowserContinuationStore(16, factory);
  const second = new BrowserContinuationStore(16, factory);
  let acknowledged = false;
  const read = first.readRecovery().then((value) => { acknowledged = true; return value; });
  const otherRead = second.readRecovery();
  const adoption = await factory.nextTransaction();
  adoption.succeed();
  await Promise.resolve();
  assert.equal(acknowledged, false);
  assert.equal(first.recoveryObserved, false);
  assert.equal(factory.recovery, legacy);
  adoption.complete();
  const adopted = await read;
  const follower = await factory.nextTransaction();
  follower.succeed();
  follower.complete();
  const alsoObserved = await otherRead;
  assert.equal(alsoObserved.writeToken, adopted.writeToken);
  assert.match(adopted.writeToken, /^[0-9a-f]{32}$/);
  assert.deepEqual(Object.keys(adopted).sort(), [...Object.keys(legacy), "writeToken"].sort());
  for (const field of ["schema", "tick", "savedAt", "retained"]) assert.equal(adopted[field], legacy[field]);
  assert.deepEqual(adopted.source, legacy.source);
  assert.deepEqual(new Uint8Array(await adopted.blob.arrayBuffer()), new Uint8Array(await legacy.blob.arrayBuffer()));
  assert.equal(first.recoveryObserved, true);
  assert.equal(second.recoveryObserved, true);
  assert.equal(follower.candidates.size, 0, "an observed current record does not receive a new token");
});

test("cancelled or quota-refused legacy adoption leaves the original recovery exactly intact", async () => {
  for (const outcome of ["cancelled", "quota"]) {
    const factory = new IndexedDbEvents(record());
    const legacy = { schema: "scriptbots.browser-recovery.v1", tick: source().tick,
      savedAt: SAVED_AT, blob: new Blob(["legacy"]), source: source() };
    factory.recovery = legacy;
    const store = new BrowserContinuationStore(16, factory);
    const abort = new AbortController();
    const read = store.readRecovery({ signal: abort.signal });
    const refused = assert.rejects(read, codeIs(outcome === "cancelled" ? "aborted" : "quota"));
    const transaction = await factory.nextTransaction();
    transaction.succeed();
    if (outcome === "cancelled") abort.abort();
    else transaction.abort(new DOMException("Controlled quota failure", "QuotaExceededError"));
    await refused;
    assert.equal(store.recoveryObserved, false);
    assert.equal(factory.recovery, legacy);
    assert.equal(Object.hasOwn(legacy, "writeToken"), false);
    const adopted = await finishRecoveryRead(store, factory);
    assert.equal(await adopted.blob.text(), "legacy");
    assert.match(adopted.writeToken, /^[0-9a-f]{32}$/);
  }
});

test("write-refused legacy adoption still permits a readonly restore without granting write authority", async () => {
  const factory = new IndexedDbEvents(record());
  const legacy = { schema: "scriptbots.browser-recovery.v1", tick: source().tick,
    savedAt: SAVED_AT, blob: new Blob(["legacy"]), source: source() };
  factory.recovery = legacy;
  factory.writeError = new DOMException("Controlled write quota refusal", "QuotaExceededError");
  const store = new BrowserContinuationStore(16, factory);
  await assert.rejects(finishRecoveryRead(store, factory), codeIs("quota"));
  assert.equal(store.recoveryObserved, false);
  assert.equal(factory.recovery, legacy);
  const readable = await finishRecoveryRead(store, factory, { observe: false });
  assert.equal(readable, legacy);
  assert.equal(await readable.blob.text(), "legacy");
  assert.equal(factory.transactions.at(-1).mode, "readonly");
  assert.equal(factory.transactions.at(-1).candidates.size, 0);
  assert.equal(store.recoveryObserved, false);
  await assert.rejects(store.saveRecovery(new Blob(["replacement"]), source()), codeIs("unobserved"));
});

test("an aborted replacement retains the prior compare identity for a safe retry", async () => {
  const factory = new IndexedDbEvents(record());
  const store = new BrowserContinuationStore(16, factory);
  await finishRecoveryRead(store, factory);
  await finishRecoverySave(store, factory, new Blob(["retained"]));
  const retained = factory.recovery;
  const save = store.saveRecovery(new Blob(["uncommitted"]), source());
  const refused = assert.rejects(save, codeIs("quota"));
  const transaction = await factory.nextTransaction();
  transaction.succeed();
  transaction.abort(new DOMException("Controlled quota failure", "QuotaExceededError"));
  await refused;
  assert.equal(factory.recovery, retained);
  assert.equal(store.recoveryObserved, true);
  await finishRecoverySave(store, factory, new Blob(["retried"]));
  assert.equal(await factory.recovery.blob.text(), "retried");
  assert.notEqual(factory.recovery.writeToken, retained.writeToken);
});

test("a legacy replacement cannot bypass an already observed recovery token", async () => {
  const factory = new IndexedDbEvents(record());
  const store = new BrowserContinuationStore(16, factory);
  await finishRecoveryRead(store, factory);
  await finishRecoverySave(store, factory, new Blob(["current"]));
  const { writeToken: _token, ...legacy } = factory.recovery;
  factory.recovery = legacy; // An older page does not implement the new protocol.
  await assert.rejects(finishRecoverySave(store, factory, new Blob(["stale"])), codeIs("conflict"));
  assert.equal(factory.recovery, legacy);
  assert.equal(Object.hasOwn(factory.recovery, "writeToken"), false, "a failed save never silently adopts a foreign record");
});

test("cancelling an opening recovery releases the operation and shares its single pending open", async () => {
  const manual = record();
  const factory = new IndexedDbEvents(manual);
  factory.stallOpen = true;
  const store = new BrowserContinuationStore(8, factory);
  const abort = new AbortController();
  const observed = store.readRecovery({ signal: abort.signal });
  const refused = assert.rejects(observed, codeIs("aborted"));
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
  assert.equal(store.recoveryObserved, false);
});

test("a cancelled open with no waiter closes its late result and admits later work", async () => {
  const factory = new IndexedDbEvents(record());
  factory.stallOpen = true;
  const store = new BrowserContinuationStore(8, factory);
  const abort = new AbortController();
  const observed = store.readRecovery({ signal: abort.signal });
  const refused = assert.rejects(observed, codeIs("aborted"));
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
  await finishRecoveryRead(store, factory);
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

async function recoveryHarness() {
  const factory = new IndexedDbEvents(record());
  const saved = [];
  let now = 0;
  let current = source();
  let captures = 0;
  const store = new BrowserContinuationStore(16, factory);
  await finishRecoveryRead(store, factory);
  const controller = new BrowserRecoveryController(store, {
    now: () => now, onSaved: (value) => saved.push(value),
  });
  const owner = {};
  controller.attach(owner);
  return { factory, store, saved, controller, owner,
    time: (value) => { now = value; },
    source: (value) => { current = value; },
    captures: () => captures,
    poll: () => controller.poll(() => current, () => { captures += 1; return new Blob([String(captures)]); }),
    finish: async () => {
      const transaction = await factory.nextTransaction();
      transaction.succeed();
      if (!transaction.finished) transaction.complete();
    },
  };
}

test("recovery requires opt-in, bounds in-flight work and coalesces to the newest owner cut", async () => {
  const h = await recoveryHarness();
  assert.equal(await h.poll(), false);
  assert.equal(h.captures(), 0);
  assert.equal(h.factory.opens, 1, "only the explicit recovery observation has opened storage");
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
  const h = await recoveryHarness();
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
  const h = await recoveryHarness();
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
  const h = await recoveryHarness();
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

test("a cross-tab conflict latches recovery until an explicit reread and retry", async () => {
  const h = await recoveryHarness();
  const other = new BrowserContinuationStore(16, h.factory);
  await finishRecoveryRead(other, h.factory);
  await finishRecoverySave(other, h.factory, new Blob(["other"]), source({ sessionId: "7" }));
  const foreign = h.factory.recovery;
  h.controller.setEnabled(true);
  const conflicted = h.poll();
  await h.finish();
  assert.equal(await conflicted, false);
  assert.equal(h.controller.error.code, "conflict");
  h.time(90_000);
  assert.equal(await h.poll(), false);
  assert.equal(h.captures(), 1);
  h.controller.retry();
  assert.equal(await h.poll(), false);
  assert.equal(h.controller.error.code, "unobserved");
  assert.equal(h.factory.recovery, foreign, "retry alone never authorizes replacing a different tab's world");
  await finishRecoveryRead(h.store, h.factory);
  h.controller.retry();
  const retried = h.poll();
  await h.finish();
  assert.equal(await retried, true);
  assert.equal(h.saved.length, 1);
  assert.equal(h.factory.recovery.source.sessionId, source().sessionId);
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

class StorageClock {
  time = 0;
  nextTimer = 0;
  timers = new Map();

  options(timeoutMs = 10) {
    return { timeoutMs, now: () => this.time,
      setTimer: (callback, delay) => {
        const id = ++this.nextTimer;
        this.timers.set(id, { callback, deadline: this.time + delay });
        return id;
      },
      clearTimer: (id) => this.timers.delete(id) };
  }

  advance(milliseconds, dispatch = true) {
    this.time += milliseconds;
    if (dispatch) this.fire();
  }

  fire() {
    for (const [id, timer] of [...this.timers]) {
      if (timer.deadline <= this.time && this.timers.delete(id)) timer.callback();
    }
  }
}

const timeoutIs = (outcome) => (error) => codeIs("timeout")(error) && error.outcome === outcome;

test("storage uses a finite ten-second default deadline and rejects invalid budgets", async () => {
  const factory = new IndexedDbEvents(record());
  factory.stallOpen = true;
  for (const timeoutMs of [0, -1, 0.5, NaN, Infinity, 2_147_483_648]) {
    assert.throws(() => new BrowserContinuationStore(8, factory, { timeoutMs }), codeIs("bounds"));
  }
  const clock = new StorageClock(), options = clock.options();
  delete options.timeoutMs;
  const store = new BrowserContinuationStore(8, factory, options);
  const refused = assert.rejects(store.read(), timeoutIs("not_written"));
  clock.advance(9_999);
  assert.equal(clock.timers.size, 1);
  assert.equal(factory.transactions.length, 0);
  clock.advance(1);
  await refused;
  assert.equal(clock.timers.size, 0);
});

test("stalled manual reads and saves time out without accumulating native opens", async () => {
  const previous = record(), factory = new IndexedDbEvents(previous), clock = new StorageClock();
  factory.stallOpen = true;
  const store = new BrowserContinuationStore(8, factory, clock.options());
  for (const operation of [() => store.read(), () => store.save(new Blob(["new"]), "8", SAVED_AT)]) {
    const refused = assert.rejects(operation(), timeoutIs("not_written"));
    clock.advance(10);
    await refused;
    assert.equal(factory.opens, 1, "a retry shares the original uncancellable IndexedDB open");
    assert.equal(factory.transactions.length, 0);
    assert.equal(factory.stored, previous);
    assert.equal(clock.timers.size, 0);
  }
  factory.lastOpen.onsuccess();
  assert.equal(factory.closes, 1, "an unobserved late connection is closed");
  factory.stallOpen = false;
  assert.equal(await finishRead(store, factory), previous);
  assert.equal(factory.opens, 2);
});

test("opening consumes the same operation deadline as the transaction", async () => {
  const previous = record(), factory = new IndexedDbEvents(previous), clock = new StorageClock();
  factory.stallOpen = true;
  const store = new BrowserContinuationStore(8, factory, clock.options());
  const refused = assert.rejects(store.save(new Blob(["new"]), "8", SAVED_AT), timeoutIs("unconfirmed"));
  clock.advance(6);
  factory.lastOpen.onsuccess();
  const transaction = await factory.nextTransaction();
  transaction.succeed();
  assert.equal(factory.stored, previous, "request success has not committed the replacement");
  clock.advance(4);
  await refused;
  assert.equal(factory.stored, previous);
  assert.equal(clock.timers.size, 0, "transaction admission never starts a fresh budget");
  factory.stallOpen = false;
  assert.equal(await finishRead(store, factory), previous);
});

test("a timed-out transaction fences new work until its actual abort and ignores late requests", async () => {
  const factory = new IndexedDbEvents(record()), clock = new StorageClock();
  const store = new BrowserContinuationStore(16, factory, clock.options());
  await finishRecoveryRead(store, factory);
  factory.stallAbort = true;
  const refused = assert.rejects(store.saveRecovery(new Blob(["expired"]), source(), { savedAt: SAVED_AT }),
    (error) => timeoutIs("unconfirmed")(error) && error.recoveryObservationRequired);
  const transaction = await factory.nextTransaction(), lateGet = transaction.request;
  clock.advance(10);
  await refused;
  assert.equal(store.recoveryObserved, false);
  for (let attempt = 0; attempt < 3; attempt += 1) {
    await assert.rejects(store.read(), codeIs("busy"));
    await assert.rejects(store.save(new Blob(["other"]), "9", SAVED_AT), codeIs("busy"));
  }
  assert.equal(factory.opens, 2);
  assert.equal(factory.transactions.length, 2, "an unreported abort cannot leak more native work");
  lateGet.result = undefined;
  lateGet.onsuccess();
  assert.equal(transaction.candidates.size, 0, "a late get cannot enqueue the recovery put");
  assert.equal(factory.recovery, undefined);
  transaction.deliverAbort();
  factory.stallAbort = false;
  assert.equal(await finishRecoveryRead(store, factory), null);
  await finishRecoverySave(store, factory, new Blob(["reconciled"]));
  assert.equal(await factory.recovery.blob.text(), "reconciled");
});

test("a recovery read arriving after its deadline cannot adopt legacy write authority", async () => {
  const factory = new IndexedDbEvents(record()), clock = new StorageClock();
  const legacy = { schema: "scriptbots.browser-recovery.v1", tick: source().tick,
    savedAt: SAVED_AT, blob: new Blob(["legacy"]), source: source() };
  factory.recovery = legacy;
  const store = new BrowserContinuationStore(16, factory, clock.options());
  const refused = assert.rejects(store.readRecovery(), timeoutIs("unconfirmed"));
  const transaction = await factory.nextTransaction();
  clock.advance(10, false); // Native request delivery wins the race with the timer callback.
  transaction.succeed();
  await refused;
  assert.equal(factory.recovery, legacy);
  assert.equal(Object.hasOwn(legacy, "writeToken"), false);
  assert.equal(transaction.candidates.size, 0);
  assert.equal(store.recoveryObserved, false);
});

test("a commit racing the deadline remains unconfirmed and requires explicit recovery reread", async () => {
  for (const timerFirst of [false, true]) {
    const factory = new IndexedDbEvents(record()), clock = new StorageClock();
    const store = new BrowserContinuationStore(16, factory, clock.options());
    await finishRecoveryRead(store, factory);
    const refused = assert.rejects(store.saveRecovery(new Blob(["committed"]), source(), { savedAt: SAVED_AT }),
      (error) => timeoutIs("unconfirmed")(error) && error.recoveryObservationRequired
        && !error.message.includes("unchanged"));
    const transaction = await factory.nextTransaction();
    transaction.succeed();
    // Model the commit boundary having passed before the completion event is delivered.
    transaction.abort = () => { throw new DOMException("Already committing", "InvalidStateError"); };
    clock.advance(10, false);
    if (timerFirst) clock.fire();
    transaction.complete();
    await refused;
    assert.equal(await factory.recovery.blob.text(), "committed", "timeout cannot promise the previous bytes survived");
    assert.equal(store.recoveryObserved, false, "a late completion cannot grant replacement authority");
    const opens = factory.opens;
    await assert.rejects(store.saveRecovery(new Blob(["retry"]), source()), codeIs("unobserved"));
    assert.equal(factory.opens, opens, "uncertain writes are not retried under a new identity");
    assert.equal(await finishRecoveryRead(store, factory), factory.recovery);
    assert.equal(store.recoveryObserved, true);
    assert.equal(clock.timers.size, 0);
  }
});

test("recovery deadlines release the controller and latch until explicit reconciliation", async () => {
  const factory = new IndexedDbEvents(record()), clock = new StorageClock();
  const store = new BrowserContinuationStore(16, factory, clock.options()), saved = [];
  await finishRecoveryRead(store, factory);
  let captures = 0;
  const controller = new BrowserRecoveryController(store, { now: () => clock.time,
    intervalMs: 20, onSaved: (value) => saved.push(value) });
  controller.attach({});
  controller.setEnabled(true);
  const poll = () => controller.poll(source, () => { captures += 1; return new Blob(["world"]); });
  const pending = poll();
  await factory.nextTransaction();
  clock.advance(10);
  assert.equal(await pending, false);
  assert.equal(controller.pending, false);
  assert.equal(controller.error.code, "timeout");
  assert.equal(controller.error.recoveryObservationRequired, true);
  for (let attempt = 0; attempt < 3; attempt += 1) {
    clock.advance(100);
    assert.equal(await poll(), false);
  }
  assert.equal(captures, 1);
  assert.deepEqual(saved, []);
  await finishRecoveryRead(store, factory);
  controller.retry();
  const resumed = poll(), transaction = await factory.nextTransaction();
  transaction.succeed();
  transaction.complete();
  assert.equal(await resumed, true);
  assert.equal(captures, 2);
  assert.equal(saved.length, 1);
});
