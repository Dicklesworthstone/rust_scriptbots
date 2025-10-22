# Target-Specific `RUSTFLAGS` Notes for wasm32 Builds

_Created: 2025-10-22 (UTC)_

## Context
- Host developers set aggressive CPU-specific `RUSTFLAGS` (e.g., `-C target-cpu=znver3`). These flags show up as warnings (`not a recognized feature`) when cross-compiling to `wasm32-unknown-unknown`.
- We want a deterministic way to suppress non-applicable flags for wasm builds without rewriting developer environment variables.

## Recommended Approach
1. Use `.cargo/config.toml` to scope Rust flags by target. Cargo expects per-target `rustflags` to live in `.cargo/config.toml`, not in `Cargo.toml`.citeturn0search0turn0search1
2. Create a target stanza disabling host-specific CPU options and optionally enabling wasm-specific cfg flags:
   ```toml
   # .cargo/config.toml
   [target.wasm32-unknown-unknown]
   rustflags = [
     "--cfg", "getrandom_backend=\"wasm_js\"",
     "-C", "target-feature=+bulk-memory,+mutable-globals",
   ]
   ```
   - Remove host-only `-C target-cpu=…` flags by **not** inheriting `RUSTFLAGS` when building wasm (documented for developers via env wrapper or build script).
   - Optional: add `-C target-feature` entries only if required; otherwise keep minimal.
3. For CI, set `CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS` to override per-job without touching global environment. Cargo respects this environment variable ahead of config files.citeturn0search1

## Next Steps
- Add `.cargo/config.toml` guidance to `docs/wasm/BUILD_WEB.md` once created.
- Provide a shell helper (`scripts/set-wasm-env.sh`) in Phase 2 if developers need convenience commands.
