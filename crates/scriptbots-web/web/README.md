# ScriptBots Web Demo Harness

Thin Canvas harness for exercising the `scriptbots-web` WASM bindings. The setup intentionally avoids a bundler—build the WASM package with `wasm-pack` and open the page via a static file server.

## Build steps

```bash
wasm-pack build crates/scriptbots-web --target web --out-dir crates/scriptbots-web/web/pkg
```

Serve the directory (using `python -m http.server` or your preferred static server) from `crates/scriptbots-web/web` and open `http://localhost:8000/`.

The demo exposes live metrics (FPS, ticks/sec, births/deaths, energy/health averages) while rendering agents with color-coded rings for boosts. Use the sliders to tweak simulation speed and population; press **Reset Simulation** to reseed with a new deterministic RNG seed.

For determinism validation, run the accompanying WASM parity test:

```bash
wasm-pack test crates/scriptbots-web --headless --chrome
```

The test compares snapshots produced by the WASM harness with native `WorldState` execution to ensure the exported bindings preserve simulation state exactly.
