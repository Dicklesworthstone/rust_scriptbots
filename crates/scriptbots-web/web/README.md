# ScriptBots Web Demo Harness

Thin Canvas harness for exercising the `scriptbots-web` WASM bindings. The setup intentionally avoids a bundler—build the WASM package with `wasm-pack` and open the page via a static file server.

## Build steps

```bash
wasm-pack build crates/scriptbots-web --target web --out-dir crates/scriptbots-web/web/pkg

# Option A: Python http.server
cd crates/scriptbots-web/web
python -m http.server 8000

# Option B: Node (requires npm)
# npm install --global http-server
# http-server
```

Open `http://localhost:8000/` and the harness will auto-load the wasm module, display build information, and begin ticking.

### Controls & metrics

- **Steps / frame**: choose how many simulation ticks to execute per animation frame (default `2`).
- **Population**: select the spawn count used the next time you reset. The wander brain offers immediate motion for visual feedback.
- **Snapshot format**: toggle between JSON and Postcard binary snapshots; the chosen format applies at the next reset and updates the status chip.
- **Renderer**: switch between Canvas 2D and WebGPU (if supported). WebGPU falls back gracefully when unavailable.
- **Reset Simulation**: reseeds with a fresh RNG seed and reuses the slider settings. A rolling 5-second performance log is appended to the console pane (`FPS`, `TPS`, population).
- Runtime brain presets can be toggled programmatically via `registerBrain("wander" | "mlp" | "none")` on the `SimHandle`.

Snapshots returned to JS follow the selected `snapshot_format` (defaults to JSON). To switch to binary snapshots (Postcard-encoded `Uint8Array`), pass `snapshot_format: "binary"` to `init_sim`.

The log pane (bottom of the sidebar) captures bootstrap and performance events so that ad-hoc benchmarking runs can be recorded while iterating on the renderer.

For determinism validation, run the accompanying WASM parity test:

```bash
wasm-pack test --headless --chrome crates/scriptbots-web
```

The test compares snapshots produced by the WASM harness with native `WorldState` execution to ensure the exported bindings preserve simulation state exactly.
