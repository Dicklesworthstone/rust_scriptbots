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

The same test has a live remote mode. Inside a source-pinned DSR profile, build
the native `scriptbots-app` binary and the WASM package from that source, then run:

```bash
SCRIPTBOTS_WEB_REMOTE_TESTS=1 \
  SCRIPTBOTS_APP_BINARY=/absolute/source-bound/scriptbots-app \
  SCRIPTBOTS_WASM_PACKAGE_DIR=/absolute/fresh/scriptbots-web-package \
  SCRIPTBOTS_WASM_PROOF_DIR=/absolute/existing/dsr-proof \
  bun crates/scriptbots-web/web/tests/stats_dom.test.js
```

This starts the production server with a fresh file-backed run, one simulation
thread and the static server's exact allowed Origin. It checks actual native
binary frames through the packaged WASM decoder, per-channel credits, browser
pause/step/resume controls against independent HTTP reads, durable command
receipts, same-ID retry after a deliberately dropped receipt and cursor reconnect,
Canvas/software-WebGPU remote PNG pixels/identity and saved
local-owner freeze/return. Truncated/version-mismatched native frames and an
unlisted browser Origin must fail. The test retains the database, native log,
wire samples, PNG and structured observations; it requests owner shutdown on
exit. This bounded software journey does not qualify hardware, stream latency,
saturation, long-run persistence or the complete remote/browser acceptance. The
initial event subscription records how many native narrative records were
observed and verifies their displayed text. An empty page qualifies only empty
history/credit handling; nonempty narrative delivery requires observed records.

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
- **World layers**: choose the combined view, terrain, food, water, or agents
  only. Canvas and WebGPU use the same overview captured with the agents'
  committed publication, both locally and when connected to a server. Source
  grids up to 128 cells per side retain every cell; larger grids use at most
  128 × 128 center-cell samples covering the whole world. These samples are not
  scientific averages. Food uses the configured food maximum and water uses
  observed depth. The layer status identifies sampling and disabled hydrology.
  Changing the view never submits a simulation command or advances a tick.
- **Reset Simulation**: reseeds with a fresh RNG seed and reuses the slider settings. A rolling 5-second performance log is appended to the console pane (`FPS`, `TPS`, population).
- **Saved continuations**: choose MLP or Unbound for a new world, then Reset.
  **Save Continuation File** downloads a portable full-owner checkpoint; **Load
  Continuation File** validates and restores it paused. **Save in This Browser**
  replaces one IndexedDB slot for this site and browser profile. The save is
  acknowledged only when the transaction completes. A reload displays the saved
  tick and time; **Restore Saved World** is always explicit and starts paused,
  without adding ticks for offline time. There is no autosave. Browser storage
  clearing or eviction can remove the slot, so download a file for a portable
  copy. Quota/abort errors keep the previous save, and invalid or stale restores
  preserve the active world. Both local restore paths work after an invalid
  startup permalink; all local continuation controls are disabled in remote mode.
- Runtime brain presets can be toggled programmatically via `registerBrain("wander" | "mlp" | "none")` on the `SimHandle`.

Snapshots returned to JS follow the selected `snapshot_format` (defaults to JSON). To switch to binary snapshots (Postcard-encoded `Uint8Array`), pass `snapshot_format: "binary"` to `init_sim`.

The log pane (bottom of the sidebar) captures bootstrap and performance events so that ad-hoc benchmarking runs can be recorded while iterating on the renderer.

The renderer status exposes `data-snapshot-tick` for the snapshot submitted to
the current backend. Canvas remains the default under the committed browser
renderer decision; WebGPU and automatic preference are explicit choices until
the required browser-matrix measurements justify promotion. WebGPU admits one
outstanding frame and refuses later
snapshots until its queue completion is observed. Refused snapshots are neither
retained nor uploaded; the next admitted paint uses the latest snapshot. The
status tick stays on the last admitted frame while science and live metrics
continue. Queue completion is not a GPU presentation acknowledgement. The active
backend is in `data-renderer`; the selector retains the user's preference when
a device failure causes fallback.

`SimHandle.frame()` returns `scriptbots.browser-frame.v1`: the existing JSON
world projection in `snapshot`, and an `identity` from the same immutable
HostCore publication. Session, publication, scientific, control and config
revisions and tick are decimal strings, preserving all integer bits. This
read does not advance science. The separate `snapshot()` JSON/postcard codec
and its positional world schema are unchanged. Renderer status publishes the
identity only when that frame is admitted; publication/science/control
revisions are distinct and must not be inferred from tick.

**Save Frame as PNG** waits for the next admitted frame and invokes canvas
serialization in that submitting callback. The filename contains its session,
publication/science/control/config revisions, tick and actual renderer. The
ready status describes that captured frame even if live science has since
advanced. Identity is in the filename/status, not embedded PNG metadata, and
is not a canonical `WorldDigest` or a GPU presentation acknowledgement.
The browser controls the final filesystem save; “PNG ready” means the download
was offered, not that a disk write was observed.

Only one request may wait or encode. Its wait times out after ten seconds;
that cannot cancel the browser's encoder. If encoding is still pending, the
save button remains disabled until its callback returns, so retry cannot
accumulate native encoders. Late callbacks cannot trigger a stale download.
A reset before serialization
completes refuses the old-world export. Serialization failure stays visible
and permits retry. At most one generated object URL is retained, released on
replacement or page exit. Capture never submits a science command or changes
playback speed.
Unchanged publications reuse their painted frame rather than repeatedly drawing
a paused world. Renderer replacement, a new simulation or canvas resize
invalidates that reuse; a waiting capture always requests a fresh paint. The
cache retains only revision/session/size/generation metadata, not agent snapshots
or a previous simulation handle.

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

The ordinary real-package checkpoint journey also saves a nonempty MLP owner in
actual IndexedDB, aborts a replacement after request success, reloads the page,
and explicitly restores the original snapshot and uninterrupted next tick. It
checks stored-byte corruption and recovery from an ownerless startup. The
permalink lifecycle harness separately checks stored tick mismatches, bounds,
quota refusal, and delayed restore races with reset/remote mode using its explicit
WASM double. `tests/continuation_store.test.mjs` can run under Node through DSR;
its controlled IndexedDB events test adapter ordering, error handling and bounds,
not real browser persistence or the Rust codec.

The saved slot contains an opaque full-owner checkpoint Blob, not a database
image or a batch journal. It leaves internal `CommittedVolatile` receipts intact
and does not qualify the full SQLite-image/journal `Durable` path in `bd-ywtv`.
Multiple tabs share the slot; the last completed manual save wins.

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
