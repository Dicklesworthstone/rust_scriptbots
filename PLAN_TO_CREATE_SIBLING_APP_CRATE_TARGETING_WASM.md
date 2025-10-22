# Plan to Create a Sibling App Crate Targeting WebAssembly

## Purpose
- Deliver a **parallel** (non-invasive) WebAssembly target for the ScriptBots workspace without modifying existing native crates.
- Establish a documented, deterministic pathway to run the simulation inside browsers while preserving native builds.
- Define milestones, required artifacts, ownership notes, and validation strategy so multiple agents can collaborate safely.

## Guiding Constraints
- **Zero modifications** to current crates until design sign-off; the wasm path lives in a sibling crate (`scriptbots-web` working title).
- **Determinism parity** must be maintained; wasm builds must produce identical simulation outcomes as native reference runs given the same seed.
- **Feature gates over forks**: never duplicate logic; reuse existing core/brain/index crates via feature flags or thin adapters.
- **Browser portability first**: initial target is Chromium-based browsers with WebAssembly GC disabled (baseline October 2025), then widen support.
- **Documentation-first**: every task outputs notes (in-repo or in issue tracker) so newcomers can follow context mid-stream.

---

## Phase 0 · Project Setup & Governance

### 0.1 Stakeholders and Roles
- **Architect** (initially: requesting user): approves scope decisions, accepts PRs altering core crates.
- **Web Crate Owner**: responsible for `scriptbots-web` crate structure, wasm toolchain upgrades.
- **Rendering Lead**: evaluates browser rendering stack, owns canvas/WebGPU integration.
- **Storage Lead**: researches browser persistence (IndexedDB, `duckdb-wasm`).
- **QA & Determinism Lead**: maintains regression suite comparing native vs wasm outputs.

### 0.2 Communication Primitives
- Planning docs (`PLAN_TO_PORT_...`, this file) updated inline with `[Currently In Progress]` markers.
- Daily standup notes appended to `codex_log.txt` (if mandated) referencing section numbers here.
- Issue tracker: create labeled issues per milestone (`wasm-phase1`, `wasm-rendering`, etc.).

### 0.3 Toolchain Baseline
- Rust toolchain remains 1.85 (per workspace).
- Add `wasm32-unknown-unknown` and `wasm32-wasip1-preview1` targets via `rustup target add` (documented only; actual install deferred).
- Introduce `wasm-bindgen-cli` and `wasm-pack` as optional developer tools (no enforced dependency yet).
- Confirm node.js ≥ 20 for bundling scripts; document installation instructions.

---

## Phase 1 · Feasibility Validation [Currently In Progress]

### 1.1 Build Dry Run (No Code Changes) [Completed 2025-10-22 — see docs/wasm/PHASE1_FINDINGS.md]
- Command: `cargo check --target wasm32-unknown-unknown -p scriptbots-core`.
- Capture and catalogue all compilation blockers (expected: Rayon threading, OS-specific APIs).
- Produce `PHASE1_FINDINGS.md` summarizing blockers and severity.
- **Follow-up:** enable `getrandom`’s JS backend and gate Rayon before rerunning check (tracked in `docs/wasm/TODO_PHASE1.md`). CPU-specific `RUSTFLAGS` guidance lives in `docs/wasm/cargo_rustflags_notes.md`.

### 1.2 Dependency Audit [Completed 2025-10-22 — see docs/wasm/dependency_audit.csv]
- Inventory every crate transitively pulled into `scriptbots-web`; flag categories:
  - ✅ **Pure Rust / wasm-ready**: `serde`, `rand`, `slotmap`, etc.
  - ⚠️ **Requires adaptation**: `rayon`, `tracing` (subscriber support in wasm), `duckdb`.
  - ❌ **Native-only**: `tch`, `kira` advanced modes, GPUI.
- Output `dependency_audit.csv` capturing crate name, version, wasm status, mitigation plan.

### 1.3 Rendering Stack Decision
- Evaluate candidate approaches (deliverables listed in §3):
  - Option A: WebGPU via `wgpu` + `web-sys` (closest to GPUI GPU semantics).
  - Option B: `egui` web backend with custom painting for agents.
  - Option C: Headless simulation + JS/Canvas renderer (JS handles drawing).
- Criteria: wasm support maturity, determinism, performance, developer ergonomics, theming parity.
- Produce decision record (`ADR-001-wasm-rendering.md`) with trade-offs and chosen path.

### 1.4 Storage Strategy Draft [Completed 2025-10-22 — see docs/wasm/adrs/ADR-002-browser-persistence.md]
- Research `duckdb-wasm` integration patterns; outline JS glue requirements.
- Explore IndexedDB or WebAssembly linear-memory journaling as interim solution.
- Deliverable: `ADR-002-browser-persistence.md`.

### 1.5 Audio Strategy Draft [Completed 2025-10-22 — see docs/wasm/adrs/ADR-003-wasm-audio.md]
- Review `kira` wasm capabilities; map gaps (streaming, mixing).
- Decide interim plan: disable audio initially vs integrate alternative (e.g., Web Audio API via `web-sys`).
- Document in `ADR-003-wasm-audio.md`.

### 1.6 Multithreading Requirements [Completed 2025-10-22 — see docs/wasm/multithreading_notes.md and docs/wasm/rayon_fallback_plan.md]
- Document prerequisites for enabling `wasm-bindgen-rayon` (SharedArrayBuffer gates, COOP/COEP headers).
- Decide MVP posture (single-thread fallback vs. multithread-first) and capture rationale.
- Record hosting requirements (HTTP response headers, service worker interplay).
- **Implementation status:** `scriptbots-core` now exposes a `parallel` feature (enabled by default) with sequential fallbacks, enabling wasm builds to disable Rayon cleanly.

### 1.7 Component Model & WASI Preview Assessment [Completed 2025-10-22 — see docs/wasm/adrs/ADR-004-component-model.md]
- Investigate `cargo component` support for future interoperability with WASI Preview 2.
- Summarize trade-offs vs. classic `wasm-bindgen` flow in `ADR-004-component-model.md`.
- Track upstream stabilization timelines for rustc and browser runtime support.

### 1.8 Browser Capability Matrix [Completed 2025-10-22 — see docs/wasm/browser_matrix.csv; review quarterly]
- Assemble matrix of WebGPU, SharedArrayBuffer, WebAssembly GC, and WASM SIMD availability across target browsers/versions.
- Capture references (browser release notes, caniuse, WPT dashboards) in `docs/wasm/browser_matrix.csv`.
- Update matrix quarterly or upon major browser releases; annotate blockers impacting roadmap.

---

## Phase 2 · Sibling Crate Scaffolding

### 2.1 Workspace Updates [Currently In Progress — 2025-10-22 Codex]
- Add new member: `crates/scriptbots-web` (binary or lib crate exporting wasm bindings).
- Provide README within crate describing build/run steps.
- Ensure crate uses feature flags to opt out of native-only deps (`default-features = false` when importing).
  - [2025-10-22 Codex] Added `scriptbots-web` crate skeleton exporting wasm-bindgen bindings; gated to `wasm32` so native checks pass while scaffolding evolves.

### 2.2 `wasm-bindgen` Interface Design [Currently In Progress — 2025-10-22 Codex]
- Define minimal API surface:
  - `init_sim(config_json: JsValue) -> Result<SimHandle>`
  - `tick(handle: SimHandle, delta_ticks: u32) -> JsValue /* snapshot */`
  - `reset(handle, seed)`
- Outline serialization format (likely `serde_wasm_bindgen` with compact binary option later).
- Document error handling expectations (convert `anyhow` to `JsError`).
  - [2025-10-22 Codex] Implemented initial bindings in `scriptbots-web`: `SimHandle` wraps the core world, exposes `init_sim`, `tick`, `snapshot`, and `reset`, returns structured snapshots (agents + summary + world metadata), validates seeds/population, and maps errors to `JsError`.
  - [2025-10-22 Codex] Added optional config overrides in `InitOptions`, `snapshot_format` toggle (`json`/`binary`), seeded agents with a lightweight wander brain for browser motion, exposed `default_init_options()` for JS bootstrap, and enabled runtime brain switching via `registerBrain` (`wander` | `mlp` | `none`).
  - [2025-10-22 Codex] Introduced a deterministic wasm-vs-native parity test (`wasm_bindgen_test`), comparing tick snapshots to guard against divergence before integrating with JS runtimes.

### 2.3 Threading Strategy
- Path A (initial): run core simulation single-threaded by feature gating Rayon (e.g., `cfg(not(target_arch = "wasm32"))` in consumer crate only).
- Path B (later): integrate `wasm-bindgen-rayon` for multithreaded wasm with thread pool initialization from JS.
- Document gating macros and environment initialization sequence.
  - [2025-10-22 Codex] Documented seed strategy toggle (`wander` vs `none`) to keep wasm builds deterministic while leaving room for future brain registrations.

### 2.4 Determinism Harness
- Design comparison test:
  1. Run N ticks native, capture snapshot JSON.
  2. Run same ticks via wasm (in node or headless browser) with identical seed.
  3. Diff snapshots; acceptable tolerance for floating point defined (e.g., ULP <= 2).
- Implement as `cargo test --test wasm_parity` once infrastructure exists.

### 2.5 CI Additions (future)
- Add GitHub Actions job (or equivalent) building wasm target with `wasm-pack build`.
- Run parity test in headless Chromium via `wasm-pack test --headless --chrome`.
- Cache `wasm-pack` artifacts to minimize build times.
  - [2025-10-22 Codex] Added `wasm` job to CI: installs wasm-pack, provisions Playwright Chromium, runs `wasm-pack build` and headless Chrome parity tests on Ubuntu.

---

## Phase 3 · Rendering Implementation [Currently In Progress — planning]

### 3.1 Rendering MVP [Currently In Progress — spike scheduling; see docs/wasm/rendering_spike_plan.md, docs/wasm/spike_webgpu_notes.md, docs/wasm/spike_canvas_notes.md]
- Chosen stack (from ADR-001) built to render:
  - Food grid heatmap
  - Agents (circles with spike indicator)
  - Basic HUD (population counts)
- Data flow: `scriptbots-web` exports snapshot structure; JS/WASM renderer paints to `<canvas>` or WebGPU surface.
- 2025-10-22: WebGPU spike crate compiled outside the repo (`/tmp/scriptbots-webgpu-proto`); ready for browser benchmarking once hosted locally. FPS/GPU metrics still pending (blocked in headless CLI—needs manual run on a workstation with Chrome 139 / Safari 26 beta).
- 2025-10-22: Canvas fallback spike scaffolded at `/tmp/canvas-baseline`; renders 10k agents via Canvas2D. Performance profiling (FPS/CPU) still requires manual browser execution (Chrome 139 / Edge 139 / Safari 18).
- [2025-10-22 Codex] Added `crates/scriptbots-web/web/` demo harness (Canvas/WebGPU) consuming the new WASM snapshot shape, with live metrics, renderer/snapshot toggles, and reset controls for quick benchmarking; pairs with the wasm parity test for verification.
- **Prototype plan (2025-10-22):**
  - Week 1: WebGPU `wgpu` spike — draw 10k agents + grid, measure FPS on Chrome 139 (Windows) and Safari 26 beta (macOS).
  - Week 1: Canvas2D baseline — JS renderer consuming serialized positions to benchmark fallback CPU cost.
  - Week 2: Compare bundle sizes, shader complexity, and input latency; capture in ADR-001.
- Assignments pending: Rendering Lead to own WebGPU spike; secondary contributor for Canvas baseline.

### 3.2 Input & Camera Controls
- Map existing controls (pan, zoom, pause) to browser events:
  - Pointer events for drag
  - Wheel for zoom (with smooth scaling)
  - Keyboard shortcuts (space = pause, +/- = speed)
- Implement `InputState` bridging JS events to wasm simulation commands.

### 3.3 Performance & Frame Throttling
- Target 60 FPS with 5k agents baseline.
- Implement requestAnimationFrame loop that batches multiple simulation ticks per render.
- Provide adaptive tick stepping when tab is throttled (visibility API).

### 3.4 Diagnostics Overlay
- Port essential metrics (tick rate, population counts) to web UI.
- Provide debug panel toggled via GUI (e.g., `?` key) showing memory usage, WASM heap.

---

## Phase 4 · Persistence & Analytics

### 4.1 Short-Term Persistence
- Implement in-memory ring buffer of snapshots for replay within single session.
- Allow exporting snapshots as downloadable JSON/Parquet via browser download APIs.

### 4.2 DuckDB Integration Pilot
- Connect to `duckdb-wasm` (bundled or CDN).
- Bridge `record_tick` calls to the JS DuckDB instance; ensure deterministic ordering.
- Prototype ingestion of limited schema (ticks table) before full parity.

### 4.3 Analytics UI
- Optional: integrate browser-based charts (D3.js or Plotly) reading from exported data.
- Ensure analytics remain opt-in to avoid performance hit.

---

## Phase 5 · Feature Parity & Polish

### 5.1 Feature Flags Audit
- Verify `ml`, `neuro` features compile or are gracefully disabled on wasm.
- Document support matrix in workspace README (native vs wasm feature availability).

### 5.2 Accessibility & UX
- Ensure color palettes meet WCAG AA contrast.
- Provide keyboard-only navigation and adjustable simulation speed slider.
- Implement persistent user settings stored in LocalStorage (zoom level, palettes).

### 5.3 Documentation
- Create `docs/wasm/` subtree with:
  - Build guide (`BUILD_WEB.md`)
  - Troubleshooting (`WASM_TROUBLESHOOTING.md`)
  - Architecture diagram (Mermaid) showing JS ↔ wasm interactions.

### 5.4 Release Packaging
- Configure bundling via `wasm-pack` + `vite` or equivalent.
- Provide GitHub Pages deployment workflow (manual or CI-driven).
- Offer downloadable zip containing wasm bundle for offline hosting.

---

## Phase 6 · Validation & Launch

### 6.1 Cross-Platform Testing
- Test in latest Chrome, Edge, Firefox Nightly, Safari Technology Preview.
- Capture compatibility matrix including GPU feature availability and performance.

### 6.2 Determinism Sign-off
- Run scripted parity suite (Phase 2.4) across representative scenarios (herbivores only, carnivores heavy, reproduction stress).
- Document variances and sign-off criteria; ensure any floating point drift stays within threshold or justify otherwise.

### 6.3 Performance Benchmarks
- Establish target metrics:
  - 10k agents @ 30 FPS on Apple M2 Safari
  - 5k agents @ 60 FPS on mid-range Windows laptop (Chrome)
- Record results and optimization notes.

### 6.4 Launch Checklist
- All ADRs resolved and merged.
- Documentation complete.
- CI passing for native + wasm pipelines.
- Governance sign-off (architect + crate owners).

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
| ---- | ------ | ---------- | ---------- |
| GPUI parity gap | Medium | High | Decouple UI; reuse data models but acknowledge cosmetic divergence. |
| Rayon on wasm blocking execution | High | Medium | Start single-threaded; introduce wasm thread pool after MVP; guard with compile-time features. |
| DuckDB port complexity | Medium | Medium | Ship without persistence initially; stage `duckdb-wasm` integration as optional add-on. |
| Browser perf shortfalls | High | Medium | Profile early, optimize data transfer (shared memory, binary snapshots), and cap agent count if necessary. |
| Feature drift between native & web | Medium | Medium | Maintain shared test suites and documentation; gate unsupported features to fail fast. |
| Toolchain churn (wasm-bindgen breaking changes) | Low | Medium | Pin versions, monitor release notes, add coverage tests in CI. |

---

## Open Questions
- Should the wasm renderer aim for visual parity with GPUI or adopt a browser-native aesthetic?
- Does the team want web builds to support multiplayer/remote observers out of the gate?
- Are we comfortable relying on WebGPU (still guarded behind flags in some browsers) or do we need a Canvas2D fallback?
- How should analytics data sync back to native storage (e.g., upload DuckDB snapshots)?

---

## Appendix A · Draft Directory Layout

```
crates/
  scriptbots-web/
    Cargo.toml
    src/
      lib.rs            # wasm-bindgen exports, simulation harness
      renderer.rs       # chosen rendering backend bridge
      input.rs          # browser input handling pipeline
    web/
      index.html
      main.ts           # JS/TS bootstrap
      wasm_worker.js    # (optional) thread pool setup
    README.md
docs/
  wasm/
    BUILD_WEB.md
    WASM_TROUBLESHOOTING.md
    architecture.mmd
```

---

## Appendix B · CPU Feature Matrix (Initial Assumptions)
- No SIMD for MVP (`-C target-feature=+simd128` opt-in later).
- Linear memory size capped at 512 MiB for typical browsers; simulation must enforce heuristics (e.g., agent cap, snapshot pooling).
- WASM GC not required; stick to reference types supported by `wasm-bindgen`.

---

Last updated: 2025-10-22 (UTC). Add `[Currently In Progress]` markers inline when phases begin.***
