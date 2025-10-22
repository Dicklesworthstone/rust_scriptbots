# scriptbots-web

WebAssembly harness for running ScriptBots simulation logic inside browsers. The crate exposes wasm-bindgen bindings that initialize a world, advance ticks, and surface simulation snapshots for rendering layers implemented in JS/WebGPU/Canvas.

## Building

```bash
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown -p scriptbots-web
```

`wasm-pack` integration will land in a later phase; see `docs/wasm/` for the roadmap and research artifacts.
