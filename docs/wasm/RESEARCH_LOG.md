# WebAssembly Port Research Log

_Started: 2025-10-22 (UTC). Append new entries chronologically; include links and citations._

## 2025-10-22
- **WebGPU availability snapshot:** Chrome 139 stable continues to expand WebGPU features (compressed 3D textures, compatibility mode), Firefox 141 enables WebGPU on Windows-only builds, Safari 26 beta brings WebGPU to macOS/iOS betas.citeturn2search0turn2search2turn2search3turn3search4
- **DuckDB WebAssembly path:** Official DuckDB WebAssembly builds support OPFS-backed persistence but recently faced a malicious npm publication, so pinning and SRI are required.citeturn0search0turn0search2turn0search3turn0search7
- **IndexedDB/OPFS option:** Workers and OPFS allow in-browser persistence without third-party runtimes; OPFS offers quotas suitable for offline apps.citeturn4search4turn4search5
- **Audio constraints:** `kira` 0.10.8 WebAssembly guide confirms lack of streaming support and file loading; Web Audio AudioWorklet tutorials illustrate alternatives.citeturn5search0turn2search1
- **SharedArrayBuffer & wasm threads:** Browsers mandate COOP/COEP and crossOriginIsolated environments; `wasm-bindgen-rayon` provides worker pool setup helpers; `coi-serviceworker` aids local dev.citeturn0search0turn0search1turn1search0turn3search4turn3search0
- **Component model outlook:** Bytecode Alliance roadmap and Rust RFC updates outline WASI Preview 2 progress and `cargo component` tooling; browsers lack component loaders today.citeturn4search0turn4search1turn4search2turn4search3
- **`getrandom` WebAssembly support:** Requires enabling the `js` feature (pulling in `wasm-bindgen` helpers); otherwise builds fail on `wasm32-unknown-unknown`.citeturn5search7
- **Cargo target-specific flags:** Use `.cargo/config.toml` or target env vars to override `RUSTFLAGS` per target without touching project manifests.citeturn2search3turn2search4
- **WebGPU spike crate:** Minimal `wgpu`/`winit` renderer built outside the repo (`/tmp/scriptbots-webgpu-proto`) renders 10k point sprites; release wasm bundle size ≈616 KiB before optimization. Metrics capture still pending.
- **Canvas fallback spike:** Vite-based prototype (`/tmp/canvas-baseline`) draws 10k agents via Canvas2D with deterministic `seedrandom`; browser metrics will be logged once run on GUI workstation.
- **WebGPU spike prep:** `wgpu`’s wasm examples (`cargo xtask run-wasm`), Learn WGPU tutorials, and Chrome’s WebGPU guide outline end-to-end browser setup and profiling strategies.citeturn0search0turn0search3turn0search8
- **Canvas baseline prep:** MDN and Chrome performance docs plus OffscreenCanvas guidance inform batching strategies and worker handoff for Canvas-rendered agents.citeturn1search0turn1search1turn1search2turn1search3

## 2026-07-11
- **Database decision:** The earlier DuckDB WebAssembly option is rejected. FrankenSQLite is the only ScriptBots SQL/database engine on native and browser targets.
- **Pinned FrankenSQLite WASM boundary:** At revision `cd9990bb16291d8c7c247b75b47faae8d7701adb`, `fsqlite-wasm` executes against `MemoryVfs`; non-memory paths are not implemented. The optional `backup` feature can import/export a complete standard-SQLite image.
- **Durability architecture:** Run FrankenSQLite in a dedicated browser storage Worker. Use bounded, sequenced persistence batches; store opaque SQLite-image checkpoints plus an idempotent batch journal in IndexedDB until a native FrankenSQLite OPFS/IndexedDB VFS is implemented and qualified.
- **Honest acknowledgement states:** An in-memory SQL transaction is `CommittedVolatile`. Only a successfully stored checkpoint or journal record is `Durable`.
- **Integration gate:** The upstream TypeScript Worker/SDK assumes feature-gated APIs absent from the default WASM artifact, and upstream CI does not yet run the embedded browser tests. ScriptBots must build an explicit feature matrix and pass real browser recovery tests before advertising persistent browser storage.
