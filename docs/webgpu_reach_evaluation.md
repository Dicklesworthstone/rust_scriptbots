# Cross-Platform Reach Evaluation: WebGPU Path for scriptbots-web (bd-2z0.14.3.7)

**Status:** Verdict (2026-07-18, BlueDog) — P3 exploration, no production commitment.
**Scope:** what a GPU web path for the browser frontend would cost, and whether to adopt one.
**Inputs:** `docs/wasm/adrs/ADR-001-wasm-rendering.md`, `docs/wasm/browser_matrix.csv` (2025-10-22 refresh), `docs/wasm/spike_webgpu_notes.md`, `crates/scriptbots-web/web/` (live harness), `crates/scriptbots-web/src/lib.rs` (snapshot formats).

---

## 1. Current state (what already exists)

| Component | State |
|---|---|
| `web/main.js` + 2D canvas | **Live.** Draws agents as 2D arcs with health-sized radii, boost/spike glyphs, HUD metrics, FPS/TPS counters. This is the shipped browser experience. |
| `web/renderers/webgpu.js` (252 lines) | **Unwired prototype.** `WebGpuRenderer.isSupported()` + `create(canvas)`, WGSL instanced-agent shader, view uniforms, agent storage buffer, per-frame `getCurrentTexture()` render pass. Not imported by `main.js`; never measured. |
| `web/renderers/canvas.js` (43 lines) | Renderer-class wrapper around the same 2D canvas path (also not used by `main.js`, which inlines its own canvas code). |
| Snapshot transport | `json` (default) or `binary` (Postcard `Uint8Array`) — the bandwidth-efficient path for a GPU renderer already exists. |
| Threading | Single-threaded wasm (Rayon disabled). No COOP/COEP requirement today. |
| `docs/wasm/browser_matrix.csv` | Chrome/Edge 139: WebGPU default-on. Safari 26 beta: WebGPU available (macOS/iOS 26 only). Firefox 141: WebGPU **Windows-only**. Safari stable 18.6: none. |

## 2. Options analysis

### Option A — Finish wiring `webgpu.js` (incremental, recommended near-term)
- **What:** import `WebGpuRenderer` in `main.js`, add `?renderer=webgpu|canvas|auto` selection with `isSupported()` capability detection and automatic canvas fallback, emit frame-time stats into the existing metrics rail, keep canvas as the default for unsupported browsers.
- **Cost:** ~2–3 days including a browser smoke matrix (Chrome, Safari 26 beta, Firefox-Windows) logged into `docs/wasm/rendering_metrics_template.csv`.
- **Dependencies:** **zero new Rust crates** — the prototype talks to the WebGPU web API directly from JS. No wasm-graph change, so the `ci/check_wasm_graph.sh` denylist is untouched.
- **Determinism:** presentation-only change; snapshot bytes identical. Visual parity vs canvas is a *styling* question, not a science one.
- **Risk:** low. The unknowns are shader robustness across drivers (ANGLE/Metal/D3D12) and alpha/blending polish, both contained by the canvas fallback.

### Option B — `wgpu`-native wasm renderer (Rust-side, shares WGSL with native)
- **What:** compile a `wgpu` renderer to `wasm32-unknown-unknown` (`wgpu` has a WebGPU backend; our native `scriptbots-world-gfx` crate is prior art for instanced sprites + post chain).
- **Cost:** weeks. Adds `wgpu` + `naga` + JS glue to the wasm graph — **`scriptbots-web` is denylisted against new heavy deps by `ci/check_wasm_graph.sh` (bd-2z0.8.16)**, so this needs a serialized dependency-lane admission, bundle-size budget (<10 MiB compressed per ADR-001), and a second shader maintenance story or a shared WGSL library with world-gfx.
- **Value:** shader/code reuse with the native 2D wgpu lane. But the *native flagship* is Bevy (bd-2z0.14 A-side), so world-gfx's long-term role is the 2D capture lane — investing in a wasm twin of a secondary lane is poor leverage.

### Option C — Stay canvas-only
- Honest and cheap. At 1–5k agents the 2D canvas path is adequate for the science-dashboard use case (this is the current product). The cost is visual: no GPU effects, and large populations draw slowly.

### Option D — `ftui-web` terminal-in-browser
- FrankenTUI has a wasm target (`ftui-web`). Once the B-side (bd-2z0.14.2) braille canvas lands, an `ftui-web` build of the TUI gives a **zero-new-renderer** rich browser surface sharing the TUI's exact semantics and themes. This is arguably the most interesting long-term web play because it reuses one codebase's visual semantics three ways (native TUI, browser, CI evidence). Not evaluated for performance here; worth a follow-up spike after B1/B2 land.
- Related: `bd-2z0.12.4` already evaluates asupersync's BrowserRuntime for the *runtime* side; the two evaluations compose cleanly.

### Option E — JS framework renderer (PixiJS/regl) — rejected
- Divergent codebase, determinism drift via JS float paths, no reuse. ADR-001's Option C; rejected there and here.

## 3. Verdict

| Question | Answer |
|---|---|
| Adopt WebGPU now? | **Yes — incrementally, via Option A.** Wire the existing `webgpu.js` prototype behind a query flag with automatic canvas fallback, measure it on the browser matrix, keep canvas the default until evidence says otherwise. |
| Build a `wgpu`-native wasm renderer? | **Defer.** Revisit only if Option A's measurements show the JS-side prototype hitting a wall that Rust-side batching would solve, AND the wasm-graph admission lane accepts the dep set. The Bevy native flagship makes a wasm twin of world-gfx low-leverage today. |
| Long-term rich web surface? | **`ftui-web` (Option D) after the B-side canvas lands** — one visual-semantics codebase, three surfaces. Schedule a dedicated spike post-B1/B2. |
| Spike measurements | **Deferred honestly:** the live-tree core refactor (Region/ActiveEffectKind, in flight by another agent at evaluation time) blocks compiling `scriptbots-web` for a measured spike in this window. Option A's acceptance evidence = the browser-matrix frame-time log; it MUST be captured before WebGPU becomes the default. Canvas stays default until then. |

## 4. Constraints recorded

- **No COOP/COEP needed** (single-threaded wasm; threads remain roadmap).
- **Snapshot format:** GPU renderers should consume `snapshot_format: "binary"` (Postcard) — already implemented; JSON remains the debug path.
- **License/dependency impact:** Option A adds nothing; Option B would require the bd-2z0.8 serialized lane + `docs/licenses.md` rows.
- **Non-goal reaffirmed (bd-2z0.14):** a full browser renderer overhaul stays out of the core program; this document is the committed evaluation that closes the question for now.
