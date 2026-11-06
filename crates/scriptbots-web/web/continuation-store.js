// One browser-managed copy of the canonical owner checkpoint. This does not
// turn the checkpoint's volatile journal into a durable SQL/batch journal.
const DATABASE = "scriptbots-continuations";
const STORE = "continuations";
const SLOT = "latest";
const SCHEMA = "scriptbots.browser-continuation.v1";

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

  constructor(maxBytes, factory) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
      throw new ContinuationStoreError("bounds", "The owner checkpoint size limit is invalid.");
    }
    this.#maxBytes = maxBytes;
    this.#factory = factory;
  }

  #validate(record) {
    if (!record || record.schema !== SCHEMA) {
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

  #open() {
    return new Promise((resolve, reject) => {
      let request;
      let refused = false;
      try {
        const factory = this.#factory ?? globalThis.indexedDB;
        if (!factory) throw new ContinuationStoreError("unavailable", "This browser does not provide IndexedDB storage.");
        request = factory.open(DATABASE, 1);
      } catch (error) {
        reject(storageError(error));
        return;
      }
      request.onblocked = () => {
        refused = true;
        this.#blockedOpen = true;
        reject(new ContinuationStoreError("blocked", "Browser storage is blocked by another tab; close that tab and retry."));
      };
      request.onupgradeneeded = () => {
        if (refused) {
          request.transaction.abort();
          return;
        }
        try {
          request.result.createObjectStore(STORE);
        } catch (error) {
          refused = true;
          request.transaction.abort();
          reject(storageError(error));
        }
      };
      request.onerror = () => {
        this.#blockedOpen = false;
        reject(storageError(request.error));
      };
      request.onsuccess = () => {
        this.#blockedOpen = false;
        const database = request.result;
        database.onversionchange = () => database.close();
        if (refused) database.close();
        else resolve(database);
      };
    });
  }

  async #transaction(mode, operation) {
    if (this.#busy) throw new ContinuationStoreError("busy", "A browser continuation operation is already in progress.");
    if (this.#blockedOpen) throw new ContinuationStoreError("blocked", "Browser storage is still blocked by another tab; close that tab and retry.");
    this.#busy = true;
    let database;
    try {
      database = await this.#open();
      return await new Promise((resolve, reject) => {
        let transaction;
        let request;
        let result;
        let failure;
        try {
          transaction = database.transaction(STORE, mode);
          transaction.onabort = () => reject(storageError(failure || transaction.error
            || new DOMException("Transaction aborted", "AbortError")));
          transaction.onerror = () => { failure ||= transaction.error || request?.error; };
          // A request can succeed and its transaction still fail at commit.
          // Only the transaction's complete event acknowledges a saved record.
          transaction.oncomplete = () => resolve(result);
          request = operation(transaction.objectStore(STORE));
          request.onsuccess = () => { result = request.result; };
          request.onerror = () => { failure = request.error; };
        } catch (error) {
          failure = error;
          if (transaction) {
            try { transaction.abort(); } catch (_) { reject(storageError(error)); }
          } else reject(storageError(error));
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
