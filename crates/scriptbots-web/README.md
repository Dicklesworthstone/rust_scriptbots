# scriptbots-web

WebAssembly harness for running ScriptBots simulation logic inside browsers. The crate exposes wasm-bindgen bindings that initialize a world, advance ticks, and surface simulation snapshots for rendering layers implemented in JS/WebGPU/Canvas.

## Building

```bash
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown -p scriptbots-web
```

`wasm-pack` integration will land in a later phase; see `docs/wasm/` for the roadmap and research artifacts.

## Bindings overview

The crate exposes the following wasm-bindgen surface:

- `default_init_options() -> JsValue` — returns the JSON-serialisable `InitOptions` structure with defaults applied.
- `init_sim(options: JsValue) -> SimHandle` — constructs a simulation using the supplied options. Recognised fields include:
  - `population` (usize): initial spawn count (defaults to 64).
  - `seed` (u64): RNG seed; when omitted a random seed is selected on the JS side.
  - `world_width` / `world_height`: override the world dimensions (pixels).
  - `config`: optional full `ScriptBotsConfig` override (rarely needed; defaults are usually sufficient).
  - `snapshot_format`: `"json"` (default) or `"binary"` (Postcard-encoded `Uint8Array`).
  - `seed_strategy`: `"wander"` (default) to attach lightweight wander brains or `"none"` to start with passive agents for custom registry wiring.
- `SimHandle::tick(steps: u32) -> JsValue` — advances the simulation and returns either JSON or a `Uint8Array` depending on `snapshot_format`.
- `SimHandle::snapshot() -> JsValue` — builds a snapshot without ticking (uses the same format toggle).
- `SimHandle::reset(seed?: number)` — rebuilds the world with an optional seed.
- `SimHandle::registerBrain(kind: string)` — placeholder for future integration with `scriptbots-brain`; currently returns an error to signal unimplemented functionality.

Snapshots are deterministic and include:

- `tick`, `epoch`, and `world` metadata (dimensions, closed/open flag)
- Per-tick summary metrics (`agent_count`, `births`, `deaths`, `average_energy`, `average_health`, etc.)
- Per-agent state (`position`, `velocity`, `heading`, `energy`, `health`, `color`, `spike_length`, `boost` flag)

The wasm crate depends on `scriptbots-core` with `default-features = false`, disabling Rayon on wasm targets. Native builds can re-enable parallelism via the `parallel` feature if needed.
