# ADR-002: Browser-Side Persistence Strategy

- **Status:** Draft (2025-10-22)
- **Context owner:** Storage Lead (TBD)
- **Related plan items:** `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md` §1.4, §4

## Context
Native builds rely on the Rust `duckdb` crate compiled with bundled C++ sources. This approach cannot run inside browsers due to reliance on native threads, filesystem access, and dynamic libraries. We need a browser-friendly persistence strategy that preserves deterministic logging and analytics without changing existing native code paths.

## Options Considered

### Option A — Integrate `duckdb-wasm`
- **Description:** Use the official DuckDB WebAssembly distribution (`@duckdb/duckdb-wasm`) loaded via JS/TS; connect WASM simulation via bindings for inserts/queries.
- **Pros:** Maintains SQL parity with native DuckDB; supports OPFS-backed persistence; DuckDB team actively maintains WebAssembly build.citeturn0search0
- **Cons:** Bundle size (multi-megabyte); limited to single-threaded execution; recent npm supply-chain incident (malicious 1.29.2/1.3.3 packages) requires strict version pinning and integrity checks.citeturn0search2turn0search3turn0search7
- **Operational Notes:** Serve pre-compressed `.wasm`; enforce Subresource Integrity (SRI); monitor DuckDB security advisories; prefer self-hosted binaries rather than CDN when possible.citeturn0search10turn0search2

### Option B — IndexedDB / OPFS Snapshot Log
- **Description:** Serialize simulation metrics/snapshots into IndexedDB or Origin Private File System (OPFS) using structured clones or binary blobs.
- **Pros:** Zero third-party runtime; small bundle footprint; straightforward permission model.citeturn0search1
- **Cons:** Requires custom query tooling; harder to maintain parity with native SQL workflows; potential performance overhead for large binary snapshots.

### Option C — Headless Remote Persistence
- **Description:** Stream metrics to a native or cloud-hosted DuckDB instance via WebTransport/WebSocket.
- **Pros:** Re-uses existing storage crate logic; offloads heavy work from browser.
- **Cons:** Breaks offline/local-first requirement; introduces latency, network dependency, and security concerns. Not aligned with "browser-only" goal.

## Decision (Interim)
Adopt a staged approach:
1. **Phase 4.1:** Implement in-memory/IndexedDB snapshot ring buffer to unblock replay and analytics basics.
2. **Phase 4.2:** Integrate `duckdb-wasm` as an optional module once supply-chain safeguards are in place:
   - Pin to vetted versions (≥1.30.0) and verify via SRI.citeturn0search2turn0search7
   - Load from self-hosted origin under COEP to stay cross-origin isolated.
   - Map core persistence APIs to DuckDB transactions; batch writes to mitigate single-thread limits.
3. Evaluate remote persistence only after in-browser path stabilizes.

## Open Questions
- How do we share schema definitions between Rust and the `duckdb-wasm` JS API to avoid drift?
- Can we reuse native `scriptbots-storage` logic through WASM bindings instead of re-implementing in JS?
- What telemetry do we collect to detect IndexedDB/OPFS quota exhaustion?

## Next Steps
- Prototype IndexedDB snapshot writer (Phase 4.1).
- Draft security checklist for `duckdb-wasm` bundle ingestion (SRI, CSP, lockdown).
- Monitor DuckDB release announcements for wasm artifacts post-September 2025 supply-chain incident.citeturn0search2turn0search7
