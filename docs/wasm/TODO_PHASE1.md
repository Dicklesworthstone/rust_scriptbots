# Phase 1 WebAssembly Sibling Crate TODOs

_Created: 2025-10-22 (UTC). Keep this list exhaustive; mark completion inline with `[x]`._

## Legend
- `[ ]` Task not started
- `[~]` Task in progress
- `[x]` Task complete
- Indent subtasks beneath parent tasks; avoid deleting completed items (append completion note instead).

## Top-Level Objectives
1. Produce baseline research artifacts for Phase 1 (dependency audit, browser matrix, ADR scaffolding).
2. Establish repeatable research workflow (logs, citations) to feed future ADRs.
3. Capture security requirements (COOP/COEP, CSP) before any code lands.

---

## Tasks

- [x] Maintain planning doc status markers
  - [x] Update `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md` with `[Currently In Progress]` tags for active subsections.
  - [ ] Refresh markers when tasks exit Phase 1.

- [x] Dependency audit deliverables
  - [x] Run `cargo tree --workspace --edges normal,build,dev` to gather dependency graph.
  - [x] Parse and classify dependencies into `dependency_audit.csv`.
  - [x] Document blockers/owners in `PHASE1_FINDINGS.md`.

- [x] Browser capability matrix
  - [x] Collect WebGPU / SharedArrayBuffer / SIMD availability data with citations.
  - [x] Populate `docs/wasm/browser_matrix.csv`.
  - [x] Note refresh cadence and triggers in the matrix header.

- [~] ADR-001 rendering experiments
  - [x] Create ADR scaffold at `docs/wasm/adrs/ADR-001-wasm-rendering.md`.
  - [x] Document evaluation criteria sourced from latest WebGPU/WebGL guidance.
  - [x] Schedule prototype spikes for WebGPU and Canvas paths (see `docs/wasm/rendering_spike_plan.md`).
  - [ ] Build WebGPU spike prototype and capture metrics.
  - [ ] Build Canvas baseline prototype and capture metrics.

- [ ] Wasm RNG + Rayon gating
  - [x] Enable `getrandom` WebAssembly backend via workspace feature flags (Cargo.toml updated).
  - [ ] Gate Rayon behind feature flag or `cfg` for wasm, documenting fallback path.
  - [x] Re-run `cargo check --target wasm32-unknown-unknown` and append findings.
  - [ ] Document approach for suppressing CPU-specific `RUSTFLAGS` noise during wasm builds.

- [x] Security baseline
  - [x] Draft `docs/wasm/SECURITY_NOTES_PHASE1.md` covering COOP/COEP, CSP, SAB requirements.
  - [x] Capture references (Chrome, Mozilla, W3C) for header guidance.

- [x] Research log
  - [x] Initialize `docs/wasm/RESEARCH_LOG.md` with date-stamped entries.
  - [x] Record links and summaries for all findings generated today.

- [ ] Reporting
  - [ ] Summarize completed work and outstanding items in final agent response for traceability.

---

## Parking Lot (Revisit Later)
- Evaluate `cargo component` readiness (Phase 1.7) once dependency audit is complete.
- Determine whether to integrate automated tools (`carton`, `wasm-opt`) into CI during Phase 2.
