# Phase 1 Findings — Initial wasm32 Dry Run

_Date: 2025-10-22 (UTC)_

## Command
- `cargo check --target wasm32-unknown-unknown -p scriptbots-core`

## Outcome (Initial Run)
- **Status:** Failed
- **Primary blocker:** `getrandom` 0.3.4 emits `compile_error!` for `wasm32-unknown-unknown` without the `wasm_js` configuration, causing dependent crates (`rand`, `rayon`, `slotmap`, etc.) to fail.citeturn5search7
- **Secondary noise:** The workspace inherits CPU-specific `RUSTFLAGS` (e.g., `-C target-cpu=znver3`) from host environment which are ignored under wasm targets; warnings are benign but noisy.

## Remediation
1. Enable `getrandom`'s WebAssembly backend via workspace dependency and add a direct dependency in `scriptbots-core` to ensure feature propagation. ✔ (2025-10-22)
2. Capture CPU flag noise suppression strategy (`RUSTFLAGS` override or target-specific `.cargo/config.toml`). _Pending._
3. Gate Rayon behind feature flag or document single-thread fallback prior to runtime testing. _Pending._

## Follow-up Run (2025-10-22)
- Command: `cargo check --target wasm32-unknown-unknown -p scriptbots-core`
- Result: **Success** after Step 1 above; no additional blockers surfaced during compilation.

## Notes
- No source files were modified; only `Cargo.toml` metadata was adjusted.
- Findings feed into `dependency_audit.csv` mitigation notes and ADR authoring.
