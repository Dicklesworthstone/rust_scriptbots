# WebAssembly Port Research Log

_Started: 2025-10-22 (UTC). Append new entries chronologically; include links and citations._

## 2025-10-22
- **WebGPU availability snapshot:** Chrome 139 stable continues to expand WebGPU features (compressed 3D textures, compatibility mode), Firefox 141 enables WebGPU on Windows-only builds, Safari 26 beta brings WebGPU to macOS/iOS betas.citeturn2search0turn2search2turn2search3turn3search4
- **DuckDB WebAssembly path:** Official DuckDB WebAssembly builds support OPFS-backed persistence but recently faced a malicious npm publication, so pinning and SRI are required.citeturn0search0turn0search2turn0search3turn0search7
- **IndexedDB/OPFS option:** Workers and OPFS allow in-browser persistence without third-party runtimes; OPFS offers quotas suitable for offline apps.citeturn1search0turn1search2
- **Audio constraints:** `kira` 0.10.8 WebAssembly guide confirms lack of streaming support and file loading; Web Audio AudioWorklet tutorials illustrate alternatives.citeturn3search0turn2search1
- **SharedArrayBuffer & wasm threads:** Browsers mandate COOP/COEP and crossOriginIsolated environments; `wasm-bindgen-rayon` provides worker pool setup helpers; `coi-serviceworker` aids local dev.citeturn3search0turn3search1turn3search3turn3search5turn3search6
- **Component model outlook:** Bytecode Alliance roadmap and Rust RFC updates outline WASI Preview 2 progress and `cargo component` tooling; browsers lack component loaders today.citeturn4search0turn4search1turn4search2turn4search3
- **`getrandom` WebAssembly support:** Requires enabling the `js` feature (pulling in `wasm-bindgen` helpers); otherwise builds fail on `wasm32-unknown-unknown`.citeturn5search7
- **Cargo target-specific flags:** Use `.cargo/config.toml` or target env vars to override `RUSTFLAGS` per target without touching project manifests.citeturn0search0turn0search1
