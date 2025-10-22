# ADR-004: Tracking WASI Preview 2 & Component Model Adoption

- **Status:** Draft (2025-10-22)
- **Context owner:** Web Crate Owner (TBD)
- **Related plan items:** `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md` §1.7, §8.1

## Context
Rust’s `wasm-bindgen` workflow targets the classic WebAssembly MVP ABI. The WebAssembly Component Model and WASI Preview 2 are maturing, promising better interoperability between wasm modules and hosts. The Bytecode Alliance and Rust project announced a roadmap: WASI Preview 2 stabilized in 2024, with `cargo component` offering tooling around components and adapters.citeturn4search0turn4search1

## Current State (Oct 2025)
- `cargo component` 0.5+ supports building Rust crates into components with bindings for WASI Preview 2.citeturn4search2
- The Rust compiler is adding Tier 3 support for `wasm32-wasip2-preview1` targets, but web browsers do not yet execute components directly; they still expect classic Wasm + JS glue.citeturn4search3
- Component model adoption on the web requires host support (e.g., browsers implementing component loaders) which is not yet available.

## Decision
- Continue shipping the WebAssembly MVP (`wasm32-unknown-unknown`) build for browser targets using `wasm-bindgen`.
- Track WASI Preview 2 and component tooling in Phase 8; only adopt once:
  1. Rust stabilizes the `wasm32-wasip2` target for production.
  2. There is a clear browser story (component loader or adapter) or alternative host (e.g., Wasmtime, Spin) we need to support.
- Maintain compatibility by structuring code so simulation core can compile under both `wasm-bindgen` and `wit-bindgen`/`cargo component` flows in the future.

## Next Steps
- Monitor `cargo component` releases; note breaking changes in `docs/wasm/RESEARCH_LOG.md`.
- Evaluate whether to produce a WASI Preview 2 build artifact for headless/server runtimes once the browser crate stabilizes.
- Revisit by Phase 8 (Future Enhancements) or earlier if browser vendors announce component-model adoption.

## Open Questions
- Should simulation data APIs be described using WIT interfaces to ease future component migration?
- Do we want to support non-browser WASI hosts (e.g., CLI or Spin apps) sharing the same wasm binary?
