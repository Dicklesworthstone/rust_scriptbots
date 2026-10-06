# scriptbots-web

WebAssembly harness for running ScriptBots simulation logic inside browsers. The crate exposes wasm-bindgen bindings that initialize a world, advance ticks, and surface simulation snapshots for rendering layers implemented in JS/WebGPU/Canvas.

## Building

```bash
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown -p scriptbots-web
```

To produce the browser package the demo in `web/` loads (`web/pkg/scriptbots_web.js`), use
either route:

```bash
# wasm-pack (the route web/README.md documents)
wasm-pack build crates/scriptbots-web --target web --out-dir web/pkg -- --locked

# or plain cargo + wasm-bindgen-cli (version must match the locked wasm-bindgen, 0.2.129)
cargo build --locked --release --target wasm32-unknown-unknown -p scriptbots-web
wasm-bindgen --target web --out-dir crates/scriptbots-web/web/pkg \
  target/wasm32-unknown-unknown/release/scriptbots_web.wasm
```

`scripts/e2e_wasm_browser_stats_dom.sh` then drives the demo page in headless Chromium through
Playwright and checks the live stats DOM (needs `bun` and the `playwright` module). Release-grade
WASM evidence is DSR-only; an npm package layout is still open (bd-azi3). See `docs/wasm/` for
the roadmap and research artifacts.

## Bindings overview

The crate exposes the following wasm-bindgen surface:

- `default_init_options() -> JsValue` — returns the JSON-serialisable `InitOptions` structure with defaults applied.
- `init_sim(options: JsValue) -> SimHandle` — constructs a simulation using the supplied options. Recognised fields include:
  - `population` (usize): initial spawn count (defaults to 64).
  - `seed` (u64): explicit RNG seed. It overrides `config.rng_seed`; when omitted,
    the nested config seed is preserved. If both are omitted, core seeds the world
    from entropy.
  - `world_width` / `world_height`: override the world dimensions (pixels).
  - `config`: optional full `ScriptBotsConfig` override (rarely needed; defaults are usually sufficient).
  - `snapshot_format`: `"json"` (default) or `"binary"` (Postcard-encoded `Uint8Array`).
  - `seed_strategy`: `"wander"` (default) to attach lightweight wander brains or `"none"` to start with passive agents for custom registry wiring.
  - `default_brain`: `"mlp"` to pre-bind agents to the baseline MLP implementation during seeding.
- `SimHandle::tick(steps: u32) -> JsValue` — advances the simulation and returns either JSON or a `Uint8Array` depending on `snapshot_format`.
- `SimHandle::snapshot() -> JsValue` — builds a snapshot without ticking (uses the same format toggle).
- `SimHandle::reset(seed?: number)` — rebuilds the world. An explicit seed
  overrides the nested config for that reset; omitting it falls back to the
  original `config.rng_seed` (or entropy when the nested seed is also absent).
- `SimHandle::registerBrain(kind: string)` — installs a brain preset for all agents (`"wander"`, `"mlp"`, or `"none"`).
- `decode_snapshot_binary(bytes: &[u8]) -> JsValue` — helper exposed for JS callers to convert binary snapshots back into structured data.

Snapshots are deterministic and use camelCase field names at the JavaScript boundary. They include:

- `tick`, `epoch`, and `world` metadata (dimensions, closed/open flag)
- Per-tick summary metrics (`agentCount`, `births`, `deaths`, `averageEnergy`, `averageHealth`, etc.)
- Per-agent state (`position`, `velocity`, `heading`, `energy`, `health`, `color`, `spikeLength`, `boost` flag)

The wasm crate depends on `scriptbots-core` with `default-features = false`, disabling Rayon on wasm targets. Native builds can re-enable parallelism via the `parallel` feature if needed.
