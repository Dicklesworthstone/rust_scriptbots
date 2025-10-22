# ADR-003: Audio Strategy for Browser Target

- **Status:** Draft (2025-10-22)
- **Context owner:** Rendering/Audio Lead (TBD)
- **Related plan items:** `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md` §1.5, §3.8, §4

## Context
The desktop build relies on the `kira` crate for expressive audio. Browser environments impose additional constraints:

1. Audio must be triggered from user gestures (autoplay restrictions).
2. WebAssembly threading support is limited; `kira`’s streaming audio backend requires threads.
3. Low-latency mixing is best handled via the Web Audio API (AudioWorklet).

Kira added WebAssembly support but still has limitations. Documentation for 0.10.8 (June 2025) notes that static sounds cannot be loaded from files, and streaming sounds are unsupported because they require threads.citeturn3search0

## Options

### Option A — Keep `kira` (muted or limited) on web
- Disable streaming features; preload short clips into memory at build time.
- Introduce a browser-specific facade that maps high-level audio events to simple one-shot sounds.
- Pros: Minimal architectural change, reuse existing asset pipeline.
- Cons: No streaming; limited dynamic audio. Requires custom preloading step to embed samples.

### Option B — Web Audio API via JavaScript/AudioWorklet
- Use `web-sys` or `wasm-bindgen` to talk directly to the Web Audio API; optionally implement AudioWorklet nodes in JS or Wasm.
- Allows granular control, streaming via `AudioWorkletProcessor`, and advanced effects.
- Examples of Rust + Web Audio + Wasm workflows (e.g., tutorial building a tuner with AudioWorklet) demonstrate real-time viability.citeturn2search1
- Pros: Full browser feature set; integrates with existing audio ecosystem.
- Cons: Requires additional JS glue; duplicative logic relative to desktop `kira`.

### Option C — Hybrid Layer
- Abstract audio events behind a trait implemented by native (`kira`) and web (`WebAudio`) backends.
- Start with Option A (limited Kira) for MVP, plan migration to Option B once requirements expand.
- Pros: Maintains shared event semantics; allows gradual evolution.
- Cons: Higher upfront abstraction cost.

## Interim Decision
- **Phase 3 / MVP:** Ship with audio muted by default; provide toggle to enable limited `kira` playback using embedded static sounds (Option A).
- **Phase 4:** Develop Web Audio backend (Option B) with AudioWorklet for looped ambience and effects. Expose cross-platform audio events through a thin trait so the runtime can choose backend dynamically.

## Implementation Notes
- Browser entry point must await a user gesture before unlocking audio context.
- Investigate existing Rust crates wrapping Web Audio (e.g., `web-audio-api-rs` inspiration) for design ideas, though they target native contexts today.citeturn2search4
- Provide fallbacks if AudioWorklet unavailable (e.g., use ScriptProcessor only during development).
- Document feature flags (`audio-web`) to avoid bundling audio assets for headless builds.

## Open Questions
- Do we require positional audio overlays in browser MVP?
- Should audio assets use compressed formats (Ogg/Opus) with on-the-fly decode vs. pre-encoded PCM?
- How do we synchronize audio with simulation events while keeping deterministic playback across native/web builds?

## Next Steps
- Prototype one-shot sound playback via Web Audio triggered from wasm-bindgen.
- Define asset packaging pipeline for browser (e.g., embedding via include_bytes! or fetching over HTTP).
- Revisit decision once Web GPU renderer is stable, ensuring audio integration doesn’t block rendering milestones.
