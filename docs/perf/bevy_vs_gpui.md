# Bevy vs. GPUI Performance & QA Baselines

_Maintained by BrownLake – 2025-10-31_

This document tracks the methodology and captured measurements for Phase 5 of the Bevy integration plan. Results should be updated whenever renderer behaviour changes (camera polish, HUD/UI updates, terrain materials, etc.).

## 1. Scenarios

| Scenario | Seed / Config | Description | Notes |
| --- | --- | --- | --- |
| `default` | `--rng-seed 424242` | Baseline world with standard population, mirrors existing golden capture. | Use for sanity/visual parity. |
| `dense_agents` | `--config docs/rendering_reference/configs/dense_agents.toml` | High-population stress test with clustered agents. | Ensures physics + HUD scales. |
| `storm_event` | `--config docs/rendering_reference/configs/storm_event.toml` | Weather-heavy scene with ambient FX enabled. | Validates post-processing + fog. |

> **TODO:** Create the `dense_agents.toml` and `storm_event.toml` configs (or update references) before the first benchmarking run if they do not already exist.

## 2. Measurement Procedure

1. **Build targets**
   - GPUI: `cargo run -p scriptbots-app --bin scriptbots-app --release --features gui`
   - Bevy: `cargo run -p scriptbots-app --bin scriptbots-app --release --features bevy_render`
2. **Environment hints**
   - Linux: export `WGPU_BACKEND=vulkan` (fallback `gl` if needed).
   - Windows: use the helper scripts (`run_windows_version_with_gui.bat`, `run_windows_version_with_bevy.bat`).
   - macOS: prefer `WGPU_BACKEND=metal`.
3. **Diagnostics**
   - Export `SB_DIAGNOSTICS=1` to activate the periodic frame summaries.
   - Optional: `RUST_LOG=info,scriptbots::bevy::diagnostics=info` for focused output.
4. **Runtime invocation**
   ```bash
   # GPUI
   SB_DIAGNOSTICS=1 RUST_LOG=info cargo run -p scriptbots-app --features gui -- --mode gui --rng-seed 424242

   # Bevy
   SB_DIAGNOSTICS=1 RUST_LOG=info,sb::bevy::diagnostics=info \
     cargo run -p scriptbots-app --features bevy_render -- --mode bevy --rng-seed 424242
   ```
   Adjust `--config` to swap scenarios and `--threads` to match target hardware (default 8 for parity with helper scripts).
   Capture logs under `logs/perf/<scenario>_<renderer>.log`; the helper commands in this doc assume that directory exists.
5. **Sampling window**
   - Allow at least 5 minutes of runtime per scenario.
   - Record the console metrics every ~300 frames (default ticker) and compute percentiles offline.

## 3. Data Capture Template

| Platform | Renderer | Scenario | FPS Mean | FPS P95 | Frame ms Mean | Frame ms P95 | Sim tick/sec | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linux (Vulkan) | GPUI | `default` |  |  |  |  |  |  |
| Linux (Vulkan) | Bevy | `default` |  |  |  |  |  |  |
| Windows (Vulkan) | GPUI | `default` |  |  |  |  |  |  |
| Windows (Vulkan) | Bevy | `default` |  |  |  |  |  |  |
| … | … | `dense_agents` |  |  |  |  |  |  |
| … | … | `storm_event` |  |  |  |  |  |  |

> Populate the table with raw numbers and link to the relevant log excerpts (`logs/perf/YYYYMMDD_<scenario>.log`) once captured.

### 3.1 Initial Golden Comparison (GPU-less sanity check)

Using the existing reference PNGs, the difference between the GPUI and Bevy default renders is:

- Mean absolute error: **39.64**
- RMSE: **59.37**
- Channel MAE (R,G,B): **39.08**, **32.36**, **47.28**

These values provide a pre-flight baseline before collecting live runtime numbers. Update this section after regenerating goldens or once new lighting passes land.

> **Headless environment note (2025-10-31):** Attempting to run the GPUI renderer under `WGPU_BACKEND=gl` in this CI/container environment stalls before diagnostics because no surface/device is available. Metrics capture is therefore blocked until we can run on a workstation with a real graphics backend.

## 4. Follow-up Checklist

- [ ] Automate log parsing into CSV summaries (TBD).
- [ ] Attach representative PNGs per scenario (GPUI + Bevy) for qualitative review.
- [ ] Update `docs/rendering_reference/checksums.txt` after regenerating goldens.
- [ ] Execute 30-minute soak tests (Linux Vulkan, Windows Vulkan/D3D12) while capturing diagnostic logs (`SB_DIAGNOSTICS=1`) and memory charts.
