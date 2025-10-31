# Bevy Render Polish Breakdown — 2025-10-31

_Prepared by WhiteCastle (codex-cli)_

## Immediate Coordination

- **Quick huddle:** 2025-10-31 @ 19:45 UTC (target 15–20 minutes).
  - Confirm lane assignments & blockers.
  - Establish first deliverables for the next 6-hour window.
- **Review matrix**
  - Automation/parity patches → BrownSnow primary, WhiteCastle secondary.
  - Render polish branches → OrangeCreek primary, BrownSnow secondary when perf-sensitive.
  - Benchmark/stability artifacts → WhiteCastle primary, OrangeCreek secondary.

## Lane Ownership (effective immediately)

1. **Benchmarks & Stability — BrownSnow (primary), WhiteCastle (backup)**
   - Secure Vulkan-capable host access; rerun Linux `default` scenario first.
   - Prepare soak-test harness + scripts so long runs can launch on-demand.
   - Capture logs + summary tables in `docs/perf/bevy_vs_gpui.md`.

2. **Automation & Parity — OrangeCreek (primary), BrownSnow (secondary)**
   - Finalize CI workflow patches (headless Bevy snapshot + `--include-ignored` test lane).
   - Rebuild Bevy golden PNG once lighting polish lands; coordinate asset freeze.
   - Execute HUD readability sweeps (1080p/1440p/4K) and document findings.

3. **Render Polish Backlog — WhiteCastle (primary), OrangeCreek (secondary)**
   - Deliverables summarised below; each sub-bucket includes owners & acceptance checks.

## Deliverable Buckets

| Bucket | Scope | Owner(s) | Acceptance |
| --- | --- | --- | --- |
| Lighting presets | DLSS/FSR3 toggle prototype, ray-traced area light preset, reflection probe bake guidance | WhiteCastle ↦ OrangeCreek | 1080p/4K captures showing preset deltas, perf trace <10% regression, toggles in `render.config.toml`. |
| Tone mapping & exposure | ACES/AgX/TonyMcMapface UI toggles, per-camera auto exposure curves, config parity with GPUI | WhiteCastle ↦ OrangeCreek | Snapshot diff within tolerance, QA checklist for dark/bright scenes, config docs updated. |
| Atmosphere & fog | Biome-specific fog/sky profiles with volumetric volumes + time-of-day hooks | WhiteCastle ↦ BrownSnow | Captures across three biomes, perf delta <5%, HUD readability unchanged. |
| Terrain shading/materials | GPU-driven heightfield meshing port, slope/curvature/moisture textures, decal overlays | WhiteCastle ↦ BrownSnow | Mesh rebuild latency targets met, before/after screenshots, asset-loading test. |
| Particles & FX | Hanabi library (ambient/weather/combat), tint sync with atmospheric parameters | WhiteCastle ↦ OrangeCreek | Presets catalogued, global toggle for perf checks, snapshot comparisons recorded. |
| Camera rigs | Orbit/inspector/cinematic rigs, snapshot presets, QA checklist | WhiteCastle ↦ OrangeCreek | `docs/rendering_reference/camera_checklist.md` updated, harness outputs per rig, shortcuts documented. |
| Agent styling | Rigged glTF import, animation graph states, decal-driven vision cones | WhiteCastle ↦ BrownSnow | Animation coverage tests, GPU profiler capture hitting batching goals, design sign-off attached. |
| Spike FX | Spline mesh spikes, Hanabi impacts, decal scorch marks | WhiteCastle ↦ OrangeCreek | Visual diff vs GPUI references, asset load sanity test, perf impact logged. |
| Weather & hydrology | `WeatherState` resource, puddle decals, splash particles bound to hydrology events | WhiteCastle ↦ BrownSnow | Scenario logs with weather transitions, screencaps per weather state, 30-minute soak metrics captured. |
| Debug tooling | Overlay extensions (weather/hydrology/behavior) + HDR/PNG snapshot automation in CI | WhiteCastle ↦ OrangeCreek | Debug panels accessible, CI HDR diff job green, documentation added to plan. |

## Next Checkpoints

- **Status echo:** Post updates in `PLAN_TO_INTEGRATE_BEVY_ENGINE_TO_SCRIPTBOTS.md` after the 19:45 UTC huddle.
- **Daily cadence:** Tentative follow-up 2025-11-01 @ 19:00 UTC unless the team prefers an alternate slot.
- **Escalation:** Flag hardware or scheduling blockers immediately via Agent Mail so we can reshuffle coverage without losing cycle time.
