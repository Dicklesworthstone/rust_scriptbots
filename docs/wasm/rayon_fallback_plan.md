# Rayon Fallback & Thread Pool Strategy (Phase 1 Task)

_Created: 2025-10-22 (UTC)_

## Objectives
- Ensure the workspace builds on `wasm32-unknown-unknown` even when Rayon is unavailable.
- Provide a deterministic toggle for parallelism: enable Rayon on native targets, degrade gracefully to sequential iteration on wasm.
- Document hosting prerequisites (SharedArrayBuffer, COOP/COEP) so we can later re-enable threads using `wasm-bindgen-rayon`.

## Proposed Implementation
1. **Feature flag:** Introduce a workspace feature `parallel` enabled by default. Rayon-dependent crates (`scriptbots-core`, `scriptbots-brain`, etc.) guard their Rayon imports behind `cfg(feature = "parallel")`. For wasm builds, set `default-features = false` (disabling `parallel`) while opting into `parallel` for native builds.
2. **Conditional compilation:** Add `#[cfg(feature = "parallel")]` around Rayon-specific struct fields and method implementations. Provide sequential fallbacks (`iter()` instead of `par_iter()`) when the feature is off.
3. **Build integration:** Update `Cargo.toml` in the future `scriptbots-web` crate to disable `parallel`:
   ```toml
   [dependencies.scriptbots-core]
   version = "0.1.0"
   default-features = false
   features = []
   ```
   Use the same pattern for dependent crates relying on Rayon.
4. **Threaded wasm path:** Once hosting is cross-origin isolated, optionally re-enable threads with `wasm-bindgen-rayon` by calling `init_thread_pool`. This requires bundling the generated JS glue and ensuring browsers allow SharedArrayBuffer.citeturn1search0turn3search0turn0search1turn0search2

## Testing Plan
- Unit test: run existing deterministic tests with `--no-default-features --features ""` to ensure sequential path compiles.
- Wasm check: `cargo check --target wasm32-unknown-unknown -p scriptbots-core --no-default-features`.
- Native regression: ensure `parallel` remains on by default for desktop builds (`cargo test --workspace`).

## Outstanding Tasks
- Audit crates for direct `rayon` usage (e.g., `par_iter`, `into_par_iter`) and plan sequential equivalents.
- Decide whether to re-export a wrapper (e.g., `ParallelIter`) to keep call sites uniform.
- Update documentation once feature flag lands (Plan doc + README).
