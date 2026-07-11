# ADR-002: Browser-Side Persistence Strategy

- **Status:** Accepted for implementation (revised 2026-07-11)
- **Context owner:** Storage Lead
- **Related plan items:** `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md` §1.4, §4

## Context

FrankenSQLite is ScriptBots' only SQL/database engine. Native builds use the pinned `fsqlite` facade and a thread-confined file-backed connection. The browser target needs the same schema and query semantics without pretending that the current WASM build already provides durable filesystem storage.

At pinned revision `cd9990bb16291d8c7c247b75b47faae8d7701adb`, `fsqlite-wasm` supports the in-memory `MemoryVfs` path. Non-memory database paths return `NotImplemented`. OPFS and IndexedDB VFS adapters are planned but are not implemented. With the optional `backup` feature, the WASM facade can import and export a complete standard-SQLite image as bytes.

The published TypeScript worker/SDK is also not yet a proven integration surface for ScriptBots: its declarations assume backup, batch execution, prepared statements, row arrays, and diagnostics that the default WASM artifact does not enable. ScriptBots therefore cannot claim browser durability or SDK readiness until a qualified feature matrix passes a real browser test.

## Decision

Use FrankenSQLite in a dedicated browser storage Worker and preserve the native storage protocol:

1. The simulation emits canonical, ordered persistence batches over a direct `MessageChannel`.
2. The channel has a bounded in-flight window, sequence numbers, idempotency keys, and explicit acknowledgements.
3. The Worker applies each batch transactionally to `fsqlite-wasm` at `:memory:` and publishes immutable analytics snapshots. Rendering never runs SQL.
4. A committed in-memory transaction is reported as `CommittedVolatile`, never `Durable`.
5. Browser durability initially consists of an opaque full SQLite-image checkpoint plus an ordered batch journal stored in IndexedDB. IndexedDB is a byte/journal substrate, not a second query engine.
6. Reload restores the newest valid image, replays later idempotent journal batches through FrankenSQLite, verifies the resulting sequence and digest, and only then publishes `Durable`.
7. When upstream ships and qualifies an OPFS or IndexedDB VFS, replace the checkpoint adapter without changing the ScriptBots schema, batch protocol, or renderer read model.

Remote synchronization may upload the same run bundle, but it is not required for offline operation and does not become an alternate database backend.

## Required Proof Before Shipping Browser Persistence

- Build a ScriptBots-owned `fsqlite-wasm` artifact with an explicit, minimal feature set; do not consume an unqualified default package.
- Run real `wasm-pack` browser tests for schema creation, transactions, rollback, full-image export/import, journal replay, quota failure, worker restart, and corrupted-checkpoint rejection.
- Prove bounded backpressure and exactly-once logical replay across Worker termination.
- Surface `CommittedVolatile`, `Durable`, lag, quota, and recovery state in the browser UI.
- Verify that the downloadable database opens through native `scriptbots-storage` and produces matching tick/replay digests.

## Current Limitations

- No direct durable OPFS/IndexedDB FrankenSQLite connection.
- No multi-tab writer or concurrent browser-writer guarantee.
- No resumable `WorldState` checkpoint in the current `scriptbots-web` snapshot; its snapshot is render-oriented.
- No production claim for the upstream TypeScript worker/SDK until its declared API matches the built WASM exports and browser E2E passes.
