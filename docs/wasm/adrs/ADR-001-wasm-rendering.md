# ADR-001: Rendering Strategy for `scriptbots-web`

- **Status:** Draft (2025-10-22) — research phase
- **Context owner:** Rendering Lead (TBD)
- **Related plan items:** `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md` §1.3, §3.1

## Context
The native application uses GPUI, which targets desktop platforms via Metal/Vulkan and lacks a browser backend. We must select a rendering stack for the WebAssembly sibling crate that:

1. Preserves simulation readability (thousands of agents, food grid overlays, HUD).
2. Achieves ≥60 FPS on mid-range hardware.
3. Integrates cleanly with wasm-bindgen bindings and the forthcoming input/event pipeline.
4. Supports deterministic presentation (consistent random seeds yield identical visuals aside from floating-point tolerances).

## Options Under Evaluation

### Option A — WebGPU via `wgpu`
- **Pros:** GPU acceleration, conceptually close to GPUI rendering model, future-proof (aligns with WebGPU roadmap).
- **Cons:** Requires WebGPU availability (Chrome stable, Safari 26 beta, Firefox Windows). Larger bundle size, more complex setup.
- **Open Questions:** Shader toolchain (WGSL vs. precompiled), fallback path when WebGPU unavailable.

### Option B — Canvas/WebGL with Rust-side renderer
- **Pros:** Works on wider browser set today, simpler setup, integrates with `wgpu` fallback via `wgpu-core` WebGL backend.
- **Cons:** Limited compute headroom, manual batching required for performance, potential duplicate logic vs. native renderer.
- **Open Questions:** How to maintain determinism with JS-driven draw loops; ease of sharing shaders/assets.

### Option C — JS/TS Renderer with Rust simulation snapshots
- **Pros:** Keeps wasm module simulation-only; leverage mature JS rendering frameworks (PixiJS, regl).
- **Cons:** Requires higher-bandwidth data transfers each frame; more divergent codebase; potential determinism drift from JS floats.
- **Open Questions:** Acceptable snapshot serialization format and cost of per-frame marshaling.

## Research Tasks
- [ ] Benchmark minimal prototypes for each option (agent draw, food grid, HUD) and record FPS/CPU usage.
- [ ] Measure payload sizes (wasm binary + JS) after tree shaking/minification.
- [ ] Evaluate developer ergonomics (hot reload, debugging, shader tooling).
- [ ] Confirm availability of text rendering, gradient fills, and blending modes needed for HUD/overlays.

## Decision Criteria (proposed)
1. Browser coverage (WebGPU availability with fallbacks).
2. Performance headroom (≥60 FPS target; 30 FPS minimum).
3. Implementation complexity and maintainability.
4. Ability to reuse assets/logic between native and web builds.
5. Size impact on wasm bundle (<10 MiB compressed target).

## Next Steps
- Populate `docs/wasm/RESEARCH_LOG.md` with prototype findings.
- Draft comparison table summarizing experimental metrics.
- Reach consensus in architecture review before Phase 3 begins execution.

## References
- Chromium WebGPU overview and origin trials.
- Apple Safari 26 beta release notes (WebGPU introduction).
- Firefox 141 release notes (WebGPU on Windows).

_Update this ADR as experiments progress; once a direction is chosen, move Status to "Accepted" and capture rationale._***
