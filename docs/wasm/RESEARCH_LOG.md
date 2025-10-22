# WebAssembly Port Research Log

_Started: 2025-10-22 (UTC). Append new entries chronologically; include links and citations._

## 2025-10-22
- **WebGPU availability snapshot:** Chrome 139 stable continues to expand WebGPU features; Linux remains behind a flag. Firefox 141 ships WebGPU on Windows only; Safari 26 beta introduces WebGPU across macOS/iOS betas.citeturn0search2turn2search2turn3search1turn2search0
- **SharedArrayBuffer requirements:** Cross-origin isolation (COOP + COEP) still mandatory for enabling SharedArrayBuffer and wasm threads across Chromium, Firefox, Safari.citeturn1search0turn1search1
- **WebAssembly SIMD status:** All evergreen browsers support baseline SIMD; relaxed SIMD still rolling out (Chrome stable, others partial).citeturn4search7
- **WASM runtime changes:** rustwasm GitHub org sunsetting; wasm-bindgen maintained by Bytecode Alliance, requiring pinning and migration plan away from wasm-pack eventually.citeturn4search9
