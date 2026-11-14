// Manual and recovery copies of the canonical owner checkpoint. This does not
// turn the checkpoint's volatile journal into a durable SQL/batch journal.
const DATABASE = "scriptbots-continuations";
const STORE = "continuations";
const SLOT = "latest";
const SCHEMA = "scriptbots.browser-continuation.v1";
const RECOVERY_SLOT = "recovery";
const RECOVERY_SCHEMA = "scriptbots.browser-recovery.v1";
const SOURCE_FIELDS = ["sessionId", "scientificRevision", "controlRevision", "configRevision", "tick"];

function recoverySource(source) {
  if (!source || SOURCE_FIELDS.some((field) => {
    const value = source[field];
    const bits = field === "sessionId" ? 128 : 64;
    return typeof value !== "string" || value.length > (bits === 128 ? 39 : 20)
      || !/^(0|[1-9][0-9]*)$/.test(value) || BigInt(value) >= (1n << BigInt(bits));
  })) {
    throw new ContinuationStoreError("source", "Recovery needs the exact observed owner, revisions and tick.");
  }
  return Object.fromEntries(SOURCE_FIELDS.map((field) => [field, source[field]]));
}

function sameSource(left, right) {
  return SOURCE_FIELDS.every((field) => left[field] === right[field]);
}

export class ContinuationStoreError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "ContinuationStoreError";
    this.code = code;
  }
}

function storageError(error) {
  if (error instanceof ContinuationStoreError) return error;
  if (error?.name === "QuotaExceededError") {
    return new ContinuationStoreError("quota", "Browser storage quota was exceeded; the previous saved continuation is unchanged.");
  }
  if (error?.name === "SecurityError" || error?.name === "NotAllowedError") {
    return new ContinuationStoreError("unavailable", "Browser storage is unavailable or permission was refused.");
  }
  if (error?.name === "AbortError") {
    return new ContinuationStoreError("aborted", "The browser storage transaction was aborted; the previous saved continuation is unchanged.");
  }
  return new ContinuationStoreError("storage", `Browser storage failed: ${error?.message || error || "unknown error"}`);
}

export class BrowserContinuationStore {
  #maxBytes;
  #factory;
  #busy = false;
  #blockedOpen = false;
  #opening = null;

  constructor(maxBytes, factory) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
      throw new ContinuationStoreError("bounds", "The owner checkpoint size limit is invalid.");
    }
    this.#maxBytes = maxBytes;
    this.#factory = factory;
  }

  #validate(record, schema = SCHEMA) {
    if (!record || record.schema !== schema) {
      throw new ContinuationStoreError("schema", "The saved continuation has an unsupported record version.");
    }
    if (typeof record.tick !== "string" || !/^(0|[1-9][0-9]{0,19})$/.test(record.tick)
        || BigInt(record.tick) > 18446744073709551615n
        || typeof record.savedAt !== "string" || record.savedAt.length !== 24
        || !Number.isFinite(Date.parse(record.savedAt))
        || new Date(record.savedAt).toISOString() !== record.savedAt) {
      throw new ContinuationStoreError("metadata", "The saved continuation metadata is invalid.");
    }
    if (!(record.blob instanceof Blob) || record.blob.size === 0) {
      throw new ContinuationStoreError("payload", "The saved continuation does not contain checkpoint bytes.");
    }
    // IndexedDB returns a Blob without materializing its contents. Check its
    // size before the consumer may allocate an ArrayBuffer or invoke WASM.
    if (record.blob.size > this.#maxBytes) {
      throw new ContinuationStoreError("oversized", "The saved continuation exceeds the browser size limit.");
    }
    return record;
  }

  async read() {
    const record = await this.#transaction("readonly", (store) => store.get(SLOT));
    return record === undefined ? null : this.#validate(record);
  }

  async save(blob, tick, savedAt = new Date().toISOString()) {
    const record = this.#validate({ schema: SCHEMA, tick, savedAt, blob });
    await this.#transaction("readwrite", (store) => store.put(record, SLOT));
    return { tick: record.tick, savedAt: record.savedAt, byteLength: record.blob.size };
  }

  async readRecovery() {
    const record = await this.#transaction("readonly", (store) => store.get(RECOVERY_SLOT));
    if (record === undefined) return null;
    this.#validate(record, RECOVERY_SCHEMA);
    const source = recoverySource(record.source);
    if (source.tick !== record.tick) {
      throw new ContinuationStoreError("source", "The recovery tick differs from its owner observation.");
    }
    return record;
  }

  async saveRecovery(blob, source, { savedAt = new Date().toISOString(), signal } = {}) {
    source = recoverySource(source);
    const record = this.#validate({ schema: RECOVERY_SCHEMA, tick: source.tick, savedAt, blob, source }, RECOVERY_SCHEMA);
    await this.#transaction("readwrite", (store) => store.put(record, RECOVERY_SLOT), signal);
    return { tick: record.tick, savedAt: record.savedAt, byteLength: record.blob.size, source };
  }

  #open(signal) {
    return new Promise((resolve, reject) => {
      let opening = this.#opening;
      const start = opening === null;
      if (start) {
        opening = { waiters: new Set(), refused: false };
        this.#opening = opening;
      }
      const finish = (callback, value) => {
        opening.waiters.delete(waiter);
        signal?.removeEventListener("abort", cancel);
        callback(value);
      };
      const cancel = () => finish(reject, storageError(new DOMException("Recovery cancelled while opening storage", "AbortError")));
      const waiter = { resolve: (value) => finish(resolve, value), reject: (error) => finish(reject, error) };
      opening.waiters.add(waiter);
      signal?.addEventListener("abort", cancel, { once: true });
      if (signal?.aborted) cancel();
      // IDBOpenDBRequest cannot itself be cancelled. One shared request remains
      // until it settles; a later manual operation can await it without opening
      // another connection. A late unobserved connection is closed immediately.
      if (!start) return;
      const refuse = (error) => {
        for (const pending of [...opening.waiters]) pending.reject(storageError(error));
      };
      let request;
      try {
        const factory = this.#factory ?? globalThis.indexedDB;
        if (!factory) throw new ContinuationStoreError("unavailable", "This browser does not provide IndexedDB storage.");
        request = factory.open(DATABASE, 1);
      } catch (error) {
        this.#opening = null;
        refuse(error);
        return;
      }
      request.onblocked = () => {
        opening.refused = true;
        this.#blockedOpen = true;
        refuse(new ContinuationStoreError("blocked", "Browser storage is blocked by another tab; close that tab and retry."));
      };
      request.onupgradeneeded = () => {
        if (opening.refused || opening.waiters.size === 0) {
          opening.refused = true;
          request.transaction.abort();
          return;
        }
        try {
          request.result.createObjectStore(STORE);
        } catch (error) {
          opening.refused = true;
          request.transaction.abort();
          refuse(error);
        }
      };
      request.onerror = () => {
        this.#blockedOpen = false;
        this.#opening = null;
        refuse(request.error);
      };
      request.onsuccess = () => {
        this.#blockedOpen = false;
        this.#opening = null;
        const database = request.result;
        database.onversionchange = () => database.close();
        if (opening.refused || opening.waiters.size === 0) database.close();
        else for (const pending of [...opening.waiters]) pending.resolve(database);
      };
    });
  }

  async #transaction(mode, operation, signal) {
    if (signal?.aborted) throw storageError(new DOMException("Recovery cancelled", "AbortError"));
    if (this.#busy) throw new ContinuationStoreError("busy", "A browser continuation operation is already in progress.");
    if (this.#blockedOpen) throw new ContinuationStoreError("blocked", "Browser storage is still blocked by another tab; close that tab and retry.");
    this.#busy = true;
    let database;
    try {
      database = await this.#open(signal);
      if (signal?.aborted) throw new DOMException("Recovery cancelled", "AbortError");
      return await new Promise((resolve, reject) => {
        let transaction;
        let request;
        let result;
        let failure;
        const cancel = () => {
          failure = new DOMException("Recovery cancelled because its owner or selection changed", "AbortError");
          // Once commit completed, cancellation cannot undo it. The controller
          // still suppresses an acknowledgement for a replaced owner.
          try { transaction?.abort(); } catch (_) { /* Already committed or aborted. */ }
        };
        const settle = (callback, value) => {
          signal?.removeEventListener("abort", cancel);
          callback(value);
        };
        try {
          transaction = database.transaction(STORE, mode);
          transaction.onabort = () => settle(reject, storageError(failure || transaction.error
            || new DOMException("Transaction aborted", "AbortError")));
          transaction.onerror = () => { failure ||= transaction.error || request?.error; };
          // A request can succeed and its transaction still fail at commit.
          // Only the transaction's complete event acknowledges a saved record.
          transaction.oncomplete = () => settle(resolve, result);
          signal?.addEventListener("abort", cancel, { once: true });
          request = operation(transaction.objectStore(STORE));
          request.onsuccess = () => { result = request.result; };
          request.onerror = () => { failure = request.error; };
        } catch (error) {
          failure = error;
          if (transaction) {
            try { transaction.abort(); } catch (_) { settle(reject, storageError(error)); }
          } else settle(reject, storageError(error));
        }
      });
    } catch (error) {
      throw storageError(error);
    } finally {
      database?.close();
      this.#busy = false;
    }
  }
}

/** Opt-in recovery for one live owner. poll() belongs to the owner timer, never RAF. */
export class BrowserRecoveryController {
  #store;
  #now;
  #intervalMs;
  #onChange;
  #onSaved;
  #owner = null;
  #enabled = false;
  #available = true;
  #pending = null;
  #lastAttempt = null;
  #lastSource = null;
  #error = null;

  constructor(store, { now = () => performance.now(), intervalMs = 30_000,
    onChange = () => {}, onSaved = () => {} } = {}) {
    if (!Number.isFinite(intervalMs) || intervalMs <= 0) {
      throw new ContinuationStoreError("bounds", "The recovery interval must be positive and finite.");
    }
    this.#store = store;
    this.#now = now;
    this.#intervalMs = intervalMs;
    this.#onChange = onChange;
    this.#onSaved = onSaved;
  }

  get enabled() { return this.#enabled; }
  get pending() { return this.#pending !== null; }
  get available() { return this.#available; }
  get error() { return this.#error; }

  /** New worlds, restores and remote mode require a fresh explicit opt-in. */
  attach(owner) {
    if (this.#owner === owner) return;
    this.#pending?.abort.abort();
    this.#owner = owner;
    this.#enabled = false;
    this.#lastAttempt = null;
    this.#lastSource = null;
    this.#error = null;
    this.#onChange(this);
  }

  setEnabled(enabled) {
    if (enabled && this.#owner === null) {
      throw new ContinuationStoreError("source", "Choose a local world before enabling recovery.");
    }
    this.#enabled = Boolean(enabled);
    this.#error = null;
    this.#lastAttempt = null;
    if (!enabled) this.#pending?.abort.abort();
    this.#onChange(this);
  }

  /** Hidden tabs and manual storage operations suspend capture, not science. */
  setAvailable(available) {
    available = Boolean(available);
    if (this.#available === available) return;
    this.#available = available;
    this.#onChange(this);
  }

  retry() {
    this.#error = null;
    this.#lastAttempt = null;
    this.#onChange(this);
  }

  async poll(readSource, capture) {
    if (!this.#enabled || !this.#available || this.#owner === null || this.#pending || this.#error) return false;
    const now = this.#now();
    if (this.#lastAttempt !== null && now - this.#lastAttempt < this.#intervalMs) return false;
    this.#lastAttempt = now;
    const owner = this.#owner;
    const attempt = { owner, abort: new AbortController() };
    try {
      const source = recoverySource(readSource());
      if (this.#lastSource) {
        if (source.sessionId !== this.#lastSource.sessionId
            || SOURCE_FIELDS.some((field) => field !== "sessionId" && BigInt(source[field]) < BigInt(this.#lastSource[field]))) {
          throw new ContinuationStoreError("source", "The recovery owner or its observed revisions regressed.");
        }
        if (sameSource(source, this.#lastSource)) return false;
      }
      const blob = capture();
      if (!sameSource(source, recoverySource(readSource()))) {
        throw new ContinuationStoreError("source", "The owner changed while capturing recovery; the previous save is unchanged.");
      }
      this.#pending = attempt;
      this.#onChange(this);
      const saved = await this.#store.saveRecovery(blob, source, { signal: attempt.abort.signal });
      if (attempt.abort.signal.aborted || owner !== this.#owner || !this.#enabled) return false;
      this.#lastSource = source;
      this.#onSaved(saved);
      return true;
    } catch (error) {
      if (!attempt.abort.signal.aborted && owner === this.#owner && this.#enabled) {
        // A quota/storage failure is latched until an explicit retry. Automatic
        // retries must not repeatedly encode large worlds or flood IndexedDB.
        this.#error = storageError(error);
      }
      return false;
    } finally {
      if (this.#pending === attempt) this.#pending = null;
      this.#onChange(this);
    }
  }
}
