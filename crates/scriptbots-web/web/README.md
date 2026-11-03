# ScriptBots Web Demo Harness

Browser harness for exercising the `scriptbots-web` WASM bindings with Canvas
2D or WebGPU. The setup
uses the produced browser package and a static server. It requires no bundler or
package-manager installation.

## Build steps

```bash
# Run packaging inside the source-pinned DSR profile. This path and its
# .wasm-pack sibling must be absent in the fresh checkout.
bash scripts/package_wasm.sh "$PWD/crates/scriptbots-web/web/pkg"

# Serve the demo and its produced package with the existing Python server.
cd crates/scriptbots-web/web
python -m http.server 8000
```

Open `http://localhost:8000/` and the harness will auto-load the wasm module, display build information, and begin ticking.

The producer pins wasm-pack 0.15.0, preserves its original generated output in
the `.wasm-pack` sibling, and emits package metadata with module/type/WASM exports
plus license notices. For an external DSR package directory, the existing browser
smoke serves that exact directory at `/pkg`:

```bash
SCRIPTBOTS_WASM_PACKAGE_DIR=/absolute/fresh/scriptbots-web-package \
  SCRIPTBOTS_WASM_PROOF_DIR=/absolute/existing/dsr-proof \
  bash scripts/e2e_wasm_browser_stats_dom.sh
```

The smoke retains the three injected DOM controls and adds a real-package case
with exact loaded-file hashes, progressing DOM metrics, public API calls and an
isolated corrupt-WASM refusal. It uses already provisioned Bun/Playwright and
retains raw logs. Packaging/browser acceptance is DSR-only and remains pending
under bd-azi3 until actual qualification; this setup does not implement Vite.
The pinned DSR profile can select its provisioned Chrome binary with
`SCRIPTBOTS_CHROMIUM_EXECUTABLE`; the smoke records the actual browser version.

### Controls & metrics

- **Simulation speed**: choose the host playback multiplier (default `2×`;
  `0` pauses). An independent timer supplies monotonic elapsed time to the Rust
  host scheduler. RAF reads the latest snapshot and never authorizes a tick.
  Hidden pages pause; becoming visible resumes the selected speed without
  replaying the hidden interval. A late timer retains the host's bounded catch-up
  policy and logs actual skipped tick opportunities. This main-thread adapter
  cannot guarantee deadlines when the browser suspends or blocks JavaScript.
- **Population**: select the spawn count used the next time you reset. The wander brain offers immediate motion for visual feedback.
- **Renderer**: Automatic prefers WebGPU and falls back to Canvas 2D. You can
  also select either backend explicitly. Canvas presents immediately while
  WebGPU initializes, so a delayed device cannot block simulation startup.
  Missing support, adapter/device
  creation failure and device loss fall back visibly to Canvas; the status and
  log retain the reason. Changing renderer keeps the current simulation.
  Both backends scale world positions to the canvas and draw circular agents
  with boost rings. The canvas keeps its intrinsic aspect ratio when displayed
  beside the controls. The demo consumes one latest JSON snapshot per paint;
  binary bindings are separate from these controls.
- **Reset Simulation**: reseeds with a fresh RNG seed and reuses the slider settings. A rolling 5-second performance log is appended to the console pane (`FPS`, `TPS`, population).
- Runtime brain presets can be toggled programmatically via `registerBrain("wander" | "mlp" | "none")` on the `SimHandle`.

Snapshots returned to JS follow the selected `snapshot_format` (defaults to JSON). To switch to binary snapshots (Postcard-encoded `Uint8Array`), pass `snapshot_format: "binary"` to `init_sim`.

The log pane (bottom of the sidebar) captures bootstrap and performance events so that ad-hoc benchmarking runs can be recorded while iterating on the renderer.

The renderer status exposes `data-snapshot-tick` for the snapshot submitted to
the current backend. WebGPU admits one outstanding frame and refuses later
snapshots until its queue completion is observed. Refused snapshots are neither
retained nor uploaded; the next admitted paint uses the latest snapshot. The
status tick stays on the last admitted frame while science and live metrics
continue. Queue completion is not a GPU presentation acknowledgement. The active
backend is in `data-renderer`; the selector retains the user's preference when
a device failure causes fallback.

Inside a pinned DSR profile, the existing stats test can separately exercise
real WebGPU/Canvas geometry, resizing, explicit capability failures and actual
device destruction. It also holds delivery of real queue completions to check
submission/write counts, latest-input admission and the submitted-snapshot stamp
while the actual WASM science timer advances. This is a controlled producer
test, not a measurement of physical GPU latency:

```bash
SCRIPTBOTS_WEB_RENDERER_TESTS=1 \
  SCRIPTBOTS_WASM_PACKAGE_DIR=/absolute/fresh/scriptbots-web-package \
  SCRIPTBOTS_WASM_PROOF_DIR=/absolute/existing/dsr-proof \
  bun crates/scriptbots-web/web/tests/stats_dom.test.js
```

This lane requires WebGPU rather than skipping when it is unavailable. It uses
Chromium's SwiftShader flags, records the actual adapter and retains screenshots.
Known geometry is rendering proof; the same packaged app also checks that
backend changes/device loss preserve its paused exported snapshot. This is
software-adapter coverage, not hardware qualification or a performance result.
The visible-frame guard inspects compositor PNG pixels, with transparent and
opaque empty-frame negatives and a one-pixel positive control. A separate
canvas-copy count is diagnostic; it does not qualify GPU image export.
The ordinary stats mode retains its separate four-case contract.

For determinism validation, run the accompanying WASM parity test:

```bash
wasm-pack test --headless --chrome crates/scriptbots-web \
  --locked --features native-parity-fixture
```

The cross-architecture test compares the browser's scalar, single-thread WASM
results at ticks 1, 8, 64, and 150 against a committed fixture generated by the
native production `parallel + simd_wide` core lane. The fixture records those
source features and includes the internal food-state digest, so a scalar native
fixture or an unobserved food-grid divergence fails closed. Exact structural
fields must match, while documented floating-point fields use the fixture's
pinned tolerance policy.
