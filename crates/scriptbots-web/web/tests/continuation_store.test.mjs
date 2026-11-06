// Production storage adapter against a controlled IndexedDB event double.
// These tests prove acknowledgement/error/bounds logic, not actual browser
// storage or checkpoint validity. A separate real-WASM Playwright journey
// exercises reload and checkpoint validation through actual browser storage.
import assert from "node:assert/strict";
import test from "node:test";
import { BrowserContinuationStore, ContinuationStoreError } from "../continuation-store.js";

const SAVED_AT = "2026-11-06T00:00:00.000Z";
const record = (overrides = {}) => ({ schema: "scriptbots.browser-continuation.v1",
  tick: "7", savedAt: SAVED_AT, blob: new Blob([new Uint8Array([1, 2, 3])]), ...overrides });
const codeIs = (code) => (error) => error instanceof ContinuationStoreError && error.code === code;

class IndexedDbEvents {
  constructor(stored) {
    this.stored = stored;
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
              get(key) { assert.equal(key, "latest"); return issue(factory.stored); },
              put(value, key) { assert.equal(key, "latest"); return issue(key, structuredClone(value)); },
            };
          },
          succeed() { transaction.request.onsuccess(); },
          complete() {
            if (mode === "readwrite") factory.stored = transaction.candidate;
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
