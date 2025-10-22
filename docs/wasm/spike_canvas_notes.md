# Canvas Baseline Spike Notes

_Created: 2025-10-22 (UTC)_

## Goal
- Prototype a JS/Canvas2D renderer fed by wasm snapshots to establish a fallback path when WebGPU is unavailable (Safari stable, older hardware, embedded contexts).

## Reference Guidance
- MDN’s “Optimizing canvas” article covers batching, `requestAnimationFrame`, avoiding state thrashing, and typed-array usage.citeturn1search0
- Chrome’s “Rendering performance” post highlights draw-call minimization, compositing layers, and avoiding layout thrash for canvas-heavy apps.citeturn1search1
- OffscreenCanvas + Web Worker patterns allow decoupling simulation and rendering threads, improving responsiveness.citeturn1search2turn1search3

## Prototype Steps
1. Create `spikes/canvas-baseline` directory with Vite (or plain Rollup) scaffold. Keep outside main crates to respect “no main-code changes” rule.  
   ```bash
   npm create vite@latest canvas-baseline -- --template vanilla-ts
   ```
2. Integrate wasm snapshot loader:  
   - Use existing `scriptbots-core` JSON snapshots (or synthetic data) serialized as arrays of `{x,y,color}`.  
   - In MVP, fetch snapshots from static JSON to emulate 5k/10k/20k agents.
3. Rendering loop:  
   - Acquire `CanvasRenderingContext2D`; call `requestAnimationFrame` to render each frame.  
   - Batch draws via `ctx.beginPath()` + `ctx.arc()` loops; flush with `ctx.fill()`.  
   - Measure cost of `ctx.fillRect` vs `ctx.arc`. Record ms/frame.
4. Performance instrumentation:  
   - Use `performance.now()` to log render time and memory usage.  
   - Capture Chrome DevTools Performance traces and Lighthouse reports.
5. OffscreenCanvas experiment:  
   - Move canvas drawing into a worker using `OffscreenCanvas` to evaluate main-thread responsiveness.  
   - Compare FPS with and without worker for 10k agents.
6. Output metrics:  
   - Record FPS (5k/10k/20k agents) for Chrome 139, Edge 139, Safari 18 (WebGL fallback).  
   - Note CPU usage and frame budget (goal ≤16.6ms).

## Current Status (2025-10-22)
- ✅ Prototype scaffolded at `/tmp/canvas-baseline` (vanilla Vite app). Rendering loop draws 10k agents with deterministic `seedrandom` and HUD displaying current FPS.
- ✅ Local development dependencies installed (`npm install`, `npm install seedrandom`).
- 🚧 Browser-based profiling (Chrome/Edge/Safari) still required to populate `docs/wasm/rendering_metrics_template.csv` and ADR-001. Blocked in headless CLI environment; collect metrics manually on a desktop with the spike served via `npm run dev`.
- 🚧 Scale tests (20k agents, OffscreenCanvas worker) pending manual execution.

## Deliverables
- Update `docs/wasm/adrs/ADR-001-wasm-rendering.md` with Canvas metrics alongside WebGPU results.
- Store benchmark scripts, config, and captured traces under `spikes/canvas-baseline/metrics/` (outside repo unless approved).
- Log findings (date, browser, agent counts, frame times) in `docs/wasm/RESEARCH_LOG.md`.

## Risks & Mitigations
- Canvas rendering may struggle beyond ~20k agents; investigate WebGL fallback if FPS < 30 by repurposing existing WebGL frameworks (e.g., regl).citeturn1search0turn1search1
- OffscreenCanvas lacks support in older Safari; detect availability before enabling worker path.citeturn1search2turn1search3
