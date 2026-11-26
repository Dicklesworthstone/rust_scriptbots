// Manual and recovery copies of the canonical owner checkpoint. This does not
// turn the checkpoint's volatile journal into a durable SQL/batch journal.
const DATABASE = "scriptbots-continuations";
// Older pages open version 1 and write recovery unconditionally. Advancing the
// database version fences those writers without replacing any retained record.
const DATABASE_VERSION = 2;
const STORE = "continuations";
const SLOT = "latest";
const SCHEMA = "scriptbots.browser-continuation.v1";
const RECOVERY_SLOT = "recovery";
const RECOVERY_SCHEMA = "scriptbots.browser-recovery.v1";
const SOURCE_FIELDS = ["sessionId", "scientificRevision", "controlRevision", "configRevision", "tick"];
const WRITE_TOKEN = /^[0-9a-f]{32}$/;
const OPERATION_TIMEOUT_MS = 10_000;

function newWriteToken() {
  if (typeof globalThis.crypto?.getRandomValues !== "function") {
    throw new ContinuationStoreError("unavailable", "Secure browser randomness is unavailable; recovery storage was not changed.");
  }
  const bytes = globalThis.crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

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
  if (error?.name === "VersionError") {
    return new ContinuationStoreError("version", "A newer ScriptBots version has updated browser storage. Reload this tab before saving or restoring.");
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
  // A deadline releases its caller, not an unsettled native transaction.
  // Retain this fence until an actual complete/abort event retires its work.
  #unsettledTransaction = null;
  #timeoutMs;
  #now;
  #setTimer;
  #clearTimer;
  #blockedOpen = false;
  #opening = null;
  // Undefined is unobserved, null is an observed empty slot. Only a completed
  // read or write establishes authority to replace the shared recovery slot.
  #recoveryToken;

  constructor(maxBytes, factory, { timeoutMs = OPERATION_TIMEOUT_MS,
    now = () => performance.now(), setTimer = (callback, delay) => setTimeout(callback, delay),
    clearTimer = (timer) => clearTimeout(timer) } = {}) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
      throw new ContinuationStoreError("bounds", "The owner checkpoint size limit is invalid.");
    }
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0 || timeoutMs > 2_147_483_647
        || typeof now !== "function" || typeof setTimer !== "function" || typeof clearTimer !== "function") {
      throw new ContinuationStoreError("bounds", "The browser storage operation deadline is invalid.");
    }
    this.#maxBytes = maxBytes;
    this.#factory = factory;
    this.#timeoutMs = timeoutMs;
    this.#now = now;
    this.#setTimer = setTimer;
    this.#clearTimer = clearTimer;
  }

  get recoveryObserved() { return this.#recoveryToken !== undefined; }

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
    const record = await this.#transaction("readonly", (store, request) => request(store.get(SLOT)));
    return record === undefined ? null : this.#validate(record);
  }

  async save(blob, tick, savedAt = new Date().toISOString()) {
    const record = this.#validate({ schema: SCHEMA, tick, savedAt, blob });
    await this.#transaction("readwrite", (store, request) => request(store.put(record, SLOT)));
    return { tick: record.tick, savedAt: record.savedAt, byteLength: record.blob.size };
  }

  #validateRecovery(record) {
    this.#validate(record, RECOVERY_SCHEMA);
    const source = recoverySource(record.source);
    if (source.tick !== record.tick) {
      throw new ContinuationStoreError("source", "The recovery tick differs from its owner observation.");
    }
    if (Object.hasOwn(record, "writeToken")
        && (typeof record.writeToken !== "string" || !WRITE_TOKEN.test(record.writeToken))) {
      throw new ContinuationStoreError("metadata", "The saved recovery has an invalid write identity.");
    }
    return record;
  }

  async readRecovery({ signal, observe = true } = {}) {
    if (!observe) {
      // Restoring retained bytes must work even when quota or a read-only
      // browser policy prevents adopting legacy metadata for future writes.
      const record = await this.#transaction("readonly", (store, request) => request(store.get(RECOVERY_SLOT)), signal);
      return record === undefined ? null : this.#validateRecovery(record);
    }
    const record = await this.#transaction("readwrite", (store, request) => {
      // Invalidate an older observation only after this operation is admitted.
      // A refused concurrent read must not revoke a write already in flight.
      this.#recoveryToken = undefined;
      request(store.get(RECOVERY_SLOT), (current) => {
        if (current === undefined) return null;
        this.#validateRecovery(current);
        if (Object.hasOwn(current, "writeToken")) return current;

        // Earlier v1 records have no compare token. Adopt one atomically while
        // retaining every original checkpoint byte and metadata field. Comparing
        // only their source/time/size would permit an ABA overwrite.
        const adopted = { ...current, writeToken: newWriteToken() };
        request(store.put(adopted, RECOVERY_SLOT), () => adopted);
        return undefined;
      });
    }, signal, true);
    this.#recoveryToken = record === null ? null : record.writeToken;
    return record;
  }

  async saveRecovery(blob, source, { savedAt = new Date().toISOString(), signal } = {}) {
    source = recoverySource(source);
    const record = this.#validate({ schema: RECOVERY_SCHEMA, tick: source.tick, savedAt, blob, source }, RECOVERY_SCHEMA);
    const expected = this.#recoveryToken;
    if (expected === undefined) {
      throw new ContinuationStoreError("unobserved", "Read the saved recovery before enabling or retrying recovery; no checkpoint was replaced.");
    }
    record.writeToken = newWriteToken();
    if (record.writeToken === expected) {
      throw new ContinuationStoreError("unavailable", "Recovery could not obtain a new write identity; the previous checkpoint is unchanged.");
    }
    await this.#transaction("readwrite", (store, request) => request(store.get(RECOVERY_SLOT), (current) => {
      if (current === undefined ? expected !== null
        : current?.writeToken !== expected || typeof expected !== "string") {
        this.#recoveryToken = undefined;
        throw new ContinuationStoreError("conflict", "Another tab changed the saved recovery. Inspect the current save and explicitly retry; no checkpoint was replaced.");
      }
      if (current !== undefined) this.#validateRecovery(current);
      request(store.put(record, RECOVERY_SLOT));
    }), signal, true);
    this.#recoveryToken = record.writeToken;
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
      const cancel = () => finish(reject, storageError(signal?.reason instanceof ContinuationStoreError
        ? signal.reason : new DOMException("Recovery cancelled while opening storage", "AbortError")));
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
        request = factory.open(DATABASE, DATABASE_VERSION);
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
          if (!request.result.objectStoreNames.contains(STORE)) request.result.createObjectStore(STORE);
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

  async #transaction(mode, operation, signal, recoveryWrite = false) {
    if (signal?.aborted) throw storageError(new DOMException("Recovery cancelled", "AbortError"));
    if (this.#busy) throw new ContinuationStoreError("busy", "A browser continuation operation is already in progress.");
    if (this.#unsettledTransaction) {
      throw new ContinuationStoreError("busy", "A previous browser storage transaction is still settling; wait before retrying.");
    }
    if (this.#blockedOpen) throw new ContinuationStoreError("blocked", "Browser storage is still blocked by another tab; close that tab and retry.");
    const deadline = this.#now() + this.#timeoutMs;
    if (!Number.isFinite(deadline)) throw new ContinuationStoreError("bounds", "The browser storage clock is unavailable.");
    this.#busy = true;
    let database;
    let transaction;
    let transactionOutcome = null;
    let timer;
    let failure;
    let timeoutError;
    let rejectActive;
    const budgetAbort = new AbortController();
    const abortTransaction = () => {
      if (!transaction || transactionOutcome !== null) return;
      // abort() can lose a race with commit. Its terminal event, not this call,
      // releases the native-work fence and establishes whether a write finished.
      try { transaction.abort(); } catch (_) { /* Await the actual outcome. */ }
    };
    const timeout = () => {
      if (timeoutError) return timeoutError;
      const uncertainWrite = transaction !== undefined && mode === "readwrite" && transactionOutcome !== "aborted";
      timeoutError = new ContinuationStoreError("timeout", uncertainWrite
        ? "Browser storage timed out; the write outcome is unconfirmed. Read the saved checkpoint before retrying."
        : transaction === undefined
          ? "Browser storage timed out before a transaction began; no checkpoint was changed."
          : "Browser storage timed out; no checkpoint write completed.");
      timeoutError.outcome = uncertainWrite ? "unconfirmed" : "not_written";
      timeoutError.recoveryObservationRequired = recoveryWrite && uncertainWrite;
      if (timeoutError.recoveryObservationRequired) this.#recoveryToken = undefined;
      failure = timeoutError;
      budgetAbort.abort(timeoutError);
      abortTransaction();
      // Do not wait indefinitely for an abort event. Further work remains
      // fenced until IndexedDB reports that this transaction actually ended.
      rejectActive?.(timeoutError);
      return timeoutError;
    };
    const expired = () => {
      if (timeoutError) return true;
      if (this.#now() < deadline) return false;
      timeout();
      return true;
    };
    const cancel = () => {
      if (timeoutError) return;
      failure = new DOMException("Recovery cancelled because its owner or selection changed", "AbortError");
      budgetAbort.abort(failure);
      abortTransaction();
    };
    try {
      timer = this.#setTimer(timeout, Math.max(0, deadline - this.#now()));
      signal?.addEventListener("abort", cancel, { once: true });
      if (signal?.aborted) cancel();
      if (expired()) throw timeoutError;
      if (budgetAbort.signal.aborted) throw failure;
      database = await this.#open(budgetAbort.signal);
      if (expired()) throw timeoutError;
      if (signal?.aborted) throw failure;
      return await new Promise((resolve, reject) => {
        let lastRequest;
        let result;
        let settled = false;
        const settle = (callback, value) => {
          if (settled) return;
          settled = true;
          callback(value);
        };
        rejectActive = (error) => settle(reject, error);
        const finished = (outcome) => {
          transactionOutcome = outcome;
          if (this.#unsettledTransaction === transaction) this.#unsettledTransaction = null;
        };
        try {
          transaction = database.transaction(STORE, mode);
          this.#unsettledTransaction = transaction;
          transaction.onabort = () => {
            finished("aborted");
            if (settled || expired()) return;
            settle(reject, storageError(failure || transaction.error || new DOMException("Transaction aborted", "AbortError")));
          };
          transaction.onerror = () => { if (!settled) failure ||= transaction.error || lastRequest?.error; };
          // A request can succeed and its transaction still fail at commit.
          // Completion after the original deadline cannot acknowledge success.
          transaction.oncomplete = () => {
            finished("committed");
            if (settled || expired()) return;
            settle(resolve, result);
          };
          const request = (next, observe = (value) => value) => {
            lastRequest = next;
            next.onsuccess = () => {
              if (settled || transactionOutcome !== null || expired() || budgetAbort.signal.aborted) return;
              try {
                result = observe(next.result);
              } catch (error) {
                failure = error;
                try { transaction.abort(); } catch (_) { settle(reject, storageError(error)); }
              }
            };
            next.onerror = () => { if (!settled) failure = next.error; };
          };
          if (expired()) return;
          operation(transaction.objectStore(STORE), request);
        } catch (error) {
          failure = error;
          if (transaction) {
            try { transaction.abort(); } catch (_) { settle(reject, storageError(error)); }
          } else settle(reject, storageError(error));
        }
      });
    } catch (error) {
      throw storageError(timeoutError || error);
    } finally {
      this.#clearTimer(timer);
      signal?.removeEventListener("abort", cancel);
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
