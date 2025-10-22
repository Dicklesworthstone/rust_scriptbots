# Rendering Spike Plan (Phase 3.1)

_Created: 2025-10-22 (UTC). Update as experiments progress._

## Goals
- Evaluate practical viability of WebGPU (`wgpu`) vs. Canvas/WebGL fallback for the `scriptbots-web` renderer.
- Capture performance metrics (FPS, CPU/GPU usage, memory footprint) across target browsers.
- Inform ADR-001 decision with real prototype data by end of Week 2.

## Timeline

| Week | Task | Owner | Deliverables |
|------|------|-------|--------------|
| Week 1 (Oct 27–31) | WebGPU spike: render 10k agents + food grid using `wgpu` (wasm32) | Rendering Lead (TBD) | Demo branch, metrics (Chrome 139 Win, Safari 26 beta Mac), notes in ADR-001 |
| Week 1 (Oct 27–31) | Canvas2D baseline: JS renderer driven by serialized snapshots | Contributor TBD | Performance log, CPU utilization data, serialization cost analysis |
| Week 2 (Nov 3–7) | Bundle size + load time comparison (gzip + brotli) | Web Crate Owner | Size table appended to ADR-001 |
| Week 2 (Nov 3–7) | Input latency & UX evaluation (pan/zoom responsiveness) | QA Lead | Report in ADR-001 annex |

## Metrics to Capture
- FPS at 5k and 10k agents.
- CPU and GPU utilization (Chrome DevTools, Safari Web Inspector).
- Frame time breakdown (simulation vs. rendering) if available.
- Wasm binary size + JS bundle size (pre/post minification).
- Input latency (ms) for pan/zoom actions.

## Tooling
- `wasm-pack` or `trunk` for rapid prototyping (not final choice).
- Chrome DevTools Performance panel, Firefox Profiler, Safari Web Inspector.
- `wasm-bindgen-test` for headless benchmarking (optional).

## Reporting
- Log raw observations in `docs/wasm/RESEARCH_LOG.md`.
- Summaries + charts embedded into `docs/wasm/adrs/ADR-001-wasm-rendering.md`.
- Update plan (`PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md`) when spikes complete.
