# Phase 1 Findings — Initial wasm32 Dry Run

_Date: 2025-10-22 (UTC)_

## Command
- `cargo check --target wasm32-unknown-unknown -p scriptbots-core`

## Outcome
- **Status:** Failed
- **Primary blocker:** `getrandom` 0.3.4 emits `compile_error!` for `wasm32-unknown-unknown` without the `js`/`wasm_js` configuration, causing dependent crates (`rand`, `rayon`, `slotmap`, etc.) to fail.citeturn5search7
- **Secondary noise:** The workspace inherits CPU-specific `RUSTFLAGS` (e.g., `-C target-cpu=znver3`) from host environment which are ignored under wasm targets; warnings are benign but noisy.

## Next Actions
1. Introduce wasm-friendly RNG configuration:
   - Enable `getrandom`'s `js` feature via `rand` or per-crate dependency overrides.
   - Alternatively, switch to `fastrand` or `rand`'s `wasm-bindgen` feature for wasm builds.
2. Capture CPU flag noise suppression strategy (`RUSTFLAGS` override or target-specific `.cargo/config.toml`).
3. Re-run `cargo check` after gating `rayon` (single-thread fallback) to surface further blockers.

## Notes
- No modifications were made to source files; this run was purely diagnostic.
- Findings should feed into `dependency_audit.csv` mitigation notes and ADR authoring.
