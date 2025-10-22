# WebGPU Spike Notes

_Created: 2025-10-22 (UTC)_

## Goal
- Stand up a minimal WebGPU renderer in Rust/Wasm to benchmark simulation-scale workloads (≥10k agents) and establish a reference implementation for the future `scriptbots-web` crate.

## Reference Implementations
- `wgpu`’s official examples ship with a WASM harness (`cargo xtask run-wasm`) that runs directly in supported browsers (Chrome 139+, Edge 139+, Safari 26 beta).citeturn0search0
- Learn WGPU’s tutorials outline wasm-specific dependencies (`wasm-bindgen`, `web-sys`, `console_log`) and feature flags needed for the browser pipeline.citeturn0search2turn0search3
- Chrome’s “Build an app with WebGPU” guide demonstrates cross-platform builds (desktop + browser) with Dawn/webgpu.h; useful for validating interop assumptions and profiling strategies.citeturn0search8
- `wgpu-rust-renderer` provides a small WebGPU renderer targeting wasm-bindgen (good baseline for layout + asset structure).citeturn0search5

## Environment Prep
1. Install required targets/tools  
   ```bash
   rustup target add wasm32-unknown-unknown
   cargo install wasm-bindgen-cli@0.2.98
   npm install -g serve # or use python3 -m http.server
   ```
2. Clone `gfx-rs/wgpu` and run `cargo xtask run-wasm --example boids -- --backend webgpu`. Capture build output and confirm example loads in Chrome 139 (Desktop) with WebGPU enabled.citeturn0search0turn0search8
3. Enable Chrome WebGPU telemetry flags if required (`chrome://flags/#enable-unsafe-webgpu`). Safari 26 beta requires WebGPU-enabled OS builds; track compatibility matrix (`docs/wasm/browser_matrix.csv`).

## Prototype Steps
1. Fork or copy the `boids` example into a dedicated `spikes/webgpu-boids` workspace (kept outside scriptbots core crates to honor “no changes to main code” rule). Use the existing simulation parameters as placeholder input volume (~16k agents).  
2. Instrument frame timing:  
   - Wrap render loop with `web_sys::window().performance().now()` to record ms/frame.  
   - Emit stats into console (use `wasm_bindgen::console::log_1`).  
3. Gather GPU counters:  
   - Capture Chrome DevTools Performance recordings; note frame time, GPU main-thread utilization.  
   - Use `chrome://tracing` or WebGPU capture (available in Chrome 139) to inspect draw call batching.  
4. Stress-test steps:  
   - Increase instance buffer to 25k and 50k agents; measure FPS on Apple M2 (Safari 26 beta) and Windows 11 RTX 3060 (Chrome 139).  
   - Record load time (ms) and wasm bundle size after release build (`wasm-bindgen --out-dir pkg`).

## Deliverables
- Metrics table stored in `docs/wasm/adrs/ADR-001-wasm-rendering.md` (append under “Research Tasks”).  
- Raw captures stored in `spikes/webgpu-boids/metrics/*.json` (kept out of repo unless approved).  
- Update `docs/wasm/RESEARCH_LOG.md` with findings (date, hardware, browser, FPS).

## Risk Notes
- WebGPU availability still gated behind browser rollout (Safari stable lacks support); ensure plan accounts for WebGL fallback (Canvas spike).citeturn0search8turn0search5
- `wasm-bindgen` CLI version must match crate version or wasm build will fail (align with tutorial guidance).citeturn0search3
