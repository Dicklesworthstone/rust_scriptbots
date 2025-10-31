# Rendering Task Coordination Log

Please append brief updates (date, handle, bullet) so collaborators can see who’s tackling what in PLAN_TO_FIX_RENDERING_ISSUES.md.

## 2025-10-30
- Codex: Completed PLAN §1.1 reference capture/spec docs; awaiting volunteers for §1.3 snapshot harness + viewport invariants + CI wiring.
- PurpleBear: Drafted `docs/rendering_reference/snapshot_harness_design.md` outlining inputs, command sketch, goldens layout, diff strategy, and CI requirements for PLAN §1.3. Ready for review/ownership.
- PurpleBear: Implemented snapshot harness test (`crates/scriptbots-render/tests/snapshot.rs`), added `--rng-seed` CLI flag, generated `docs/rendering_reference/golden/rust_default.png`, and updated PLAN §1.3 snapshot bullet to completed.
- PinkMountain & PurpleBear: Added viewport invariants inside `crates/scriptbots-render/src/lib.rs` (`camera_invariants_tests`), covering screen↔world bounds and minimum pixel radius for agents; PLAN §1.3 bullet updated to completed status.
- Codex & PurpleBear: Camera refactor proposal (`docs/rendering_reference/camera_refactor_proposal.md`) now includes ownership matrix for Stages 1–4; awaiting volunteers to claim slots before coding begins.
- PurpleBear: Added Stage 1 extraction checklist (`docs/rendering_reference/camera_refactor_task_breakdown.md`) detailing current camera touchpoints, extraction steps, and open questions for the eventual owner.
- PurpleBear & PinkMountain: Completed Stage 1 camera extraction (PLAN §2.1/§2.2/§2.4) — `camera/mod.rs` merged, renderer now uses `Camera`, and unit/invariant tests landed.
- PurpleBear: Added `render_regression` GitHub Actions job (Ubuntu + Windows) running snapshot + camera invariants with path filters and diff artifact uploads; PLAN §1.3 CI bullet marked complete.
- PurpleBear: Stage 2 wiring underway – GPUI/wgpu/CPU render paths now consume `Camera::layout`; offscreen snapshot updated to use `world_to_screen` for parity with interactive view.
- PinkMountain: Claimed PLAN §1.3 viewport invariants; aligning tests with new camera module. Will share progress updates as camera Stage 1 lands.
- PinkMountain: Drafted Stage 1 work breakdown for camera refactor (`camera_stage1_plan.md`); Stage 2 wiring underway pending visuals/CI ownership.
- PinkMountain: Owning Stage 2 camera wiring (GPUI + terminal/offscreen integration) per PLAN §2.3/§2.4 updates.
- PinkMountain: Stage 2 progress — GPUI + WGPU paths now rely on `Camera::layout`; terminal/offscreen parity and HUD overlays remain.
- PinkMountain: Extended `render_png_offscreen` to share `Camera::layout` + `world_to_screen`, keeping REST exports aligned with the live renderer; HUD/inspector overlays now surface screen coordinates via `world_to_screen`.
- PinkMountain: Added camera layout logging safeguards (`camera_layout_pre_follow`, `auto_fit_world_due_to_extreme_zoom`, etc.) to diagnose blank viewport cases.
- RedCastle: Regenerated all legacy reference PNGs (`legacy_default.png`, `legacy_selected_agent.png`, `legacy_food_peak.png`, `legacy_food_off.png`, `legacy_zoomed_hud.png`), refreshed SHA256s in `checksums.txt`, and published capture helper `capture_legacy_render.sh`.
- RedCastle: Added SVG overlays (`legacy_*_overlay.svg`) aligned with ROI tables and updated spec/plan to mark PLAN §1.2 complete.
- RedCastle: Drafted `visual_polish_plan.md` outlining palette/legibility work and claimed PLAN §3.1–§3.2 pending Stage 2 stability.
- RedCastle: Implemented PLAN §3.1–§3.2 (legacy palette port, drop shadows, thicker outlines, GPU colour sync); plan doc + snapshot refreshed.
- RedCastle: Added terminal palette cycling (Natural/Deuter/Protan/Tritan/High Contrast) and rethemed HUD metrics to share the new colour sets; PLAN §3.4 marked in-progress pending perf sweep.
- RedCastle: Completed PLAN §3.4 — terminal palettes cycled via `c`, theme parity achieved, and headless perf sweep (240 frames @ threads=2) showed no regressions.
- RedCastle: Completed PLAN §3.5 validation — compared legacy vs. current goldens (MAE ≈ 89, agents clearly visible), checked terrain palette/render HUD readability, and verified terminal palettes under headless run.
- RedSnow (2025-10-30 17:13 UTC): Picking up PLAN §2.3 HUD/debug overlay rewiring to adopt `CameraSnapshot::world_to_screen`; coordinating with PinkMountain/PurpleBear.
- RedSnow (2025-10-30 17:28 UTC): Completed Stage 2.3 HUD debug overlay refactor; `paint_debug_overlays` and the batched agent outline pass now call `CameraSnapshot::world_to_screen` to stay aligned with zoom/pan math.
- RedSnow (2025-10-30 19:15 UTC): Updated PLAN §2 status to completed; follow-mode fit chips now documented and Stage 2/3 milestones closed out.
- RedSnow (2025-10-30 20:18 UTC): Started PLAN_TO_INTEGRATE_BEVY_ENGINE_TO_SCRIPTBOTS Phase 0 (feature flag + crate scaffold + CLI plumbing).
- RedSnow (2025-10-30 21:04 UTC): Bevy renderer now streams world snapshots via worker thread, renders ground plane + agent spheres, is selectable with `--mode bevy`, ships a snapshot test comparing against `docs/rendering_reference/golden/bevy_default.png`, adds mouse orbit + WASD pan + `F` follow / Q/E yaw / PageUp/PageDown pitch camera controls, and includes a camera smoke test in `tests/camera.rs`.
- RedSnow (2025-10-30 21:32 UTC): CI render job now covers `cargo test -p scriptbots-bevy --features bevy_render` so Bevy snapshot/camera tests run on PRs.
- RedSnow (2025-10-30 21:50 UTC): Added Bevy HUD overlay (tick, agents, follow state, camera metrics) driven by live snapshots.
- OrangeLake (2025-10-30 23:35 UTC): Picking up PLAN_TO_INTEGRATE_BEVY_ENGINE_TO_SCRIPTBOTS Phase 2 camera polish (fit selection, follow parity, easing) and Phase 3 HUD parity expansion (selection/FPS/playback metrics).
- OrangeLake (2025-10-30 23:58 UTC): Landed Bevy camera polish (`F` cycle, Ctrl+S/O follow, Ctrl+F/W fits, easing) plus HUD parity metrics (selection details, playback rate, FPS, world stats); PLAN phases 2–3 marked ready for review.
- OrangeLake (2025-10-30 23:59 UTC): Starting PLAN_TO_INTEGRATE_BEVY_ENGINE_TO_SCRIPTBOTS Phase 4 (interactive selection + command bridge) to mirror GPUI agent picking and clear commands.
- OrangeLake (2025-10-31 00:05 UTC): Phase 4 update — Bevy left-click selection now raycasts to the ground plane, honors Shift toggles, and submits `ControlCommand::UpdateSelection` (clear/add/replace) through the shared queue; empty clicks clear selection.
- GPT-5 Codex (2025-10-31 00:45 UTC): Implemented WFC-derived terrain snapshot export, chunked heightfield meshing, and agent elevation sampling in Bevy; terrain now renders as 3D relief ready for material polish.
- OrangeLake (2025-10-31 00:52 UTC): Implemented Bevy HUD action row with follow buttons + clear selection; buttons call into the command queue and keep camera follow state aligned with GPUI.
- OrangeLake (2025-10-31 01:12 UTC): Added keyboard hints/colour states to the follow buttons, instrumented selection/follow logging, and verified behaviour via targeted unit tests (see `Captured command log entries` output in test run).
- OrangeLake (2025-10-31 01:30 UTC): Kicking off SimulationCommand pipeline + playback styling refresh per Phase 4 TODO checklist; aligning control contract with GPUI/terminal surfaces.

## 2025-10-31
- BlueMountain (2025-10-31 03:21 UTC): Claimed PLAN §3.2 follow-up to replace CPU quad agent bodies with circular paths, add heading cues, and align tint/boost handling with the GPU renderer snapshot helper.
- BlueMountain (2025-10-31 03:45 UTC): Expanded §3.2 TODO list for full ScriptBot avatar rendering (wheels, spike, mouth, sensors, diet banding) and will tackle CPU path → GPU parity next.
- BlueMountain (2025-10-31 04:32 UTC): Landed CPU + WGPU avatar overhaul (capsule bodies, wheels, spikes, diet stripe, mouth/eyes/ears, boost flame) with shared data pipeline; todo: refresh snapshot golden and prune legacy fallback once verified.
- BlueMountain (2025-10-31 04:48 UTC): Flagged golden refresh as in-progress; prepping to drop the CPU layout fallback now that the viewport stretch fix is in place.
- BlueMountain (2025-10-31 05:02 UTC): Removed the legacy CPU canvas fallback offset now that the flex row stretch fix holds; renderer logs confirm sane viewport sizes.
- BlueMountain (2025-10-31 05:20 UTC): Regenerated `golden/rust_default.png` with new avatars and updated SHA256 in `docs/rendering_reference/checksums.txt`; ready for CI snapshot comparison.
- BlueMountain (2025-10-31 05:48 UTC): Benchmarking blocked on GPU workstation; GPUI run stalled (see `logs/perf/20251031_default_gui.log`), Bevy build currently failing on `TerrainChunkStats` field changes (`logs/perf/20251031_default_bevy.log`). Scheduling follow-up once hardware + fixes land.
- BlueMountain (2025-10-31 05:55 UTC): Fixed Bevy shader eye placement bug (sin/cos swap) so GPU avatars match CPU orientation (`crates/scriptbots-world-gfx/src/lib.rs`).
- BlueMountain (2025-10-31 06:34 UTC): Added Bevy vocalization arcs with amplitude-driven quads, culling inactive overlays, and wired accessibility palette cycling (HUD + key `C`) with per-material palette transforms.
- BlueMountain (2025-10-31 06:05 UTC): Rebuilt Bevy agent spawning pipeline with multi-part meshes/material helpers, hooked the new runtime fields (wheels/boost/sensors/diet/audio), tinted stripes by temperature preference, and reconciled `TerrainChunkStats` so `cargo check -p scriptbots-app --features bevy_render` succeeds; vocalization arcs still pending per plan §3.2.
- OrangeLake (2025-10-31 02:05 UTC): Simulation commands landed (core enum, Bevy driver, GPUI + terminal emitters), Bevy playback UI restyled with relief palette accents, and targeted tests/cargo checks executed; awaiting RedSnow feedback on broader contract.
- BrownLake (2025-10-31 03:24 UTC): Documented the SimulationCommand contract + HUD palette notes inside the Bevy integration plan and pinged RedSnow/OrangeLake for sign-off before closing Phase 4 TODO.
- BrownLake (2025-10-31 03:31 UTC): Added `run_windows_version_with_bevy.bat` launcher and updated Bevy plan §7.2 to track Windows/Linux/macOS helper parity.
- BrownLake (2025-10-31 03:36 UTC): Added Linux/macOS Bevy launch scripts, refreshed README quickstart sections, expanded plan §7.2 checklist, and sent Agent Mail announcing the cross-platform helpers.
- BrownLake (2025-10-31 03:50 UTC): Opened Phase 5 QA checklist in Bevy plan and started instrumentation tasks (FrameTimeDiagnostics/diagnostic logging); awaiting coordination before running benchmarks.
- BrownLake (2025-10-31 04:12 UTC): Landed SB_DIAGNOSTICS-gated frame diagnostics plugin + 300-frame summary logger (colorized) and marked plan §5 instrumentation tasks complete.
- BrownLake (2025-10-31 04:20 UTC): Created `docs/perf/bevy_vs_gpui.md` with benchmark procedure + data template; configs for dense_agents/storm_event flagged as TODO before timing runs.
- BrownLake (2025-10-31 04:28 UTC): Added `docs/rendering_reference/configs/{dense_agents,storm_event}.toml` to support the benchmark scenarios.
- BrownLake (2025-10-31 04:34 UTC): Updated `render_regression` CI job to run Bevy tests headlessly on Linux with `WGPU_BACKEND=gl` while keeping Windows on the CPU snapshot path.
- BrownLake (2025-10-31 04:38 UTC): Added manual `bevy_exhaustive` workflow job to run `cargo test -p scriptbots-bevy -- --include-ignored` on demand.
- BrownLake (2025-10-31 04:42 UTC): Computed baseline MAE/RMSE between `rust_default.png` and `bevy_default.png` for the default scenario; awaiting live metrics.
- BrownLake (2025-10-31 04:44 UTC): Added soak-test procedure to `docs/perf/bevy_vs_gpui.md`; long-run metrics still pending.
- BrownLake (2025-10-31 04:48 UTC): GPUI benchmark attempt under headless GL timed out (no presentation surface); marked docs with hardware requirement note and will rerun on a workstation.
- BrownLake (2025-10-31 04:52 UTC): Bevy run likewise aborted with \"Unable to find a GPU\"; same hardware constraint applies.
- BrownLake (2025-10-31 04:55 UTC): No material/lighting changes landed in this pass—golden regeneration deferred until visual polish work resumes.
- BrownLake (2025-10-31 04:58 UTC): Soak tests queued for hardware run; headless container cannot create GPU devices (documented in perf checklist).
- BrownLake (2025-10-31 05:06 UTC): Added `scripts/run_perf_benchmarks.sh` helper and documented usage in `docs/perf/bevy_vs_gpui.md` for teammates with GPU access.
- BrownLake (2025-10-31 05:08 UTC): Fixed f32 clamp panic on Windows by guarding eye radius min/max (`crates/scriptbots-render/src/lib.rs`); pinged BlueMountain to rerun.
- BrownLake (2025-10-31 05:15 UTC): Added `scripts/parse_perf_logs.py` to summarise FPS/frame diagnostics and checked off the automation bullet in `docs/perf/bevy_vs_gpui.md`.
- BrownCreek (2025-10-31 03:29 UTC): Sent Agent Mail to BrownLake & OrangeLake requesting alignment on Bevy version target and outstanding renderer work before starting the upgrade.
- BrownCreek (2025-10-31 04:58 UTC): Upgraded `scriptbots-bevy` to Bevy 0.17.2, ported renderer/UI/tests to the typed camera/mesh/message APIs, refreshed diagnostics logging, and verified `cargo check -p scriptbots-bevy`; full workspace check currently blocked by `scriptbots-world-gfx` lacking a `scriptbots-core` dependency.
- BrownCreek (2025-10-31 05:12 UTC): Added per-chunk reflection probes (`LightProbe` + `EnvironmentMapLight`) with shared `ReflectionProbeAssets` handles and documented the bake/refresh playbook in the integration plan.
- ChartreusePond (2025-10-31 05:28 UTC): wired Bevy HUD tone-mapping buttons (ACES/AgX/Tony) with bias + auto-exposure toggles backed by `TonemappingState`, scheduled new systems, and enabled `AutoExposurePlugin`; config surface/offscreen parity still pending before closing PLAN §11 tone mapping TODO.
- OrangeCreek (2025-10-31 19:15 UTC): Coordinated lane ownership with WhiteCastle & BrownSnow — automation/parity coverage (HUD readability, golden regeneration, regression workflows) assigned to OrangeCreek, benchmark/stability runs to BrownSnow, render polish backlog to WhiteCastle; set BrownSnow as primary reviewer for upcoming Phase 4 SimulationCommand/UI PR with OrangeCreek secondary, and scheduled daily sync for 2025-11-01 16:00 UTC.
- OrangeCreek (2025-10-31 19:35 UTC): Expanded Bevy unit tests — multi-resolution offscreen captures now cover 1080p/1440p/4K, and `hud_overlay_populates_metrics` asserts HUD text/shortcuts/auto-pause cues match GPUI copy. Follow-mode tolerance check remains blocked pending replay cursor logs; readability screenshots to be captured once GPU workstation access is available.
- BrownSnow (2025-10-31 19:15 UTC): Logged Bevy plan coordination ownership (Phase 5 §5) and sent Agent Mail outlining immediate actions: unblock GPU benchmark hardware, kick off soak tests once access confirmed, and track daily async updates alongside the 2025-11-01 16:00 UTC sync.
- BrownSnow (2025-10-31 19:20 UTC): Added explicit 30-minute soak test commands (GPUI + Bevy) to reference during hardware runs (mirrors `scripts/run_perf_benchmarks.sh` defaults):  
  ```bash
  # GPUI soak
  timeout 1800 env SB_DIAGNOSTICS=1 RUST_LOG=info \
    SCRIPTBOTS_MAX_THREADS=8 SCRIPTBOTS_FORCE_GUI=1 \
    cargo run -p scriptbots-app --release --features gui -- \
    --mode gui --rng-seed 424242 --threads 8 \
    | tee logs/perf/$(date +%Y%m%d)_default_gui_soak.log

  # Bevy soak
  timeout 1800 env SB_DIAGNOSTICS=1 \
    RUST_LOG=info,sb::bevy::diagnostics=info \
    SCRIPTBOTS_MAX_THREADS=8 SCRIPTBOTS_MODE=bevy \
    cargo run -p scriptbots-app --release --features bevy_render -- \
    --mode bevy --rng-seed 424242 --threads 8 \
    | tee logs/perf/$(date +%Y%m%d)_default_bevy_soak.log
  ```
  Pending: add scenario-specific `--config` flags (dense_agents/storm_event) and verify whether we want a dedicated CLI duration argument versus relying on `timeout`.
- BrownSnow (2025-10-31 19:21 UTC): Submitted Agent Mail contact requests to BrownLake and BlueMountain for immediate Vulkan-capable workstation access; awaiting approval before logging the hardware window in `docs/perf/bevy_vs_gpui.md`.
- BrownSnow (2025-10-31 19:22 UTC): `cargo test -p scriptbots-bevy -- --nocapture` currently fails — `VertexAttributeValues` import path changed (use `bevy_mesh::vertex::VertexAttributeValues`) and `AgentVisual` initializer in `lib.rs` lacks new fields (`boost`, `eye_dirs`, `eye_fov`, etc.). Flagged for BlueMountain/OrangeCreek follow-up before re-enabling regression snapshots.
- BrownSnow (2025-10-31 19:23 UTC): Rescheduled the immediate coordination huddle per earlier directive; WhiteCastle has since re-slotted it for 2025-10-31 19:45 UTC with updated agenda.
- BrownSnow (2025-10-31 19:27 UTC): Resolved `cargo test -p scriptbots-bevy -- --nocapture` failures by switching the test import to `bevy_mesh::VertexAttributeValues`; all Bevy unit/snapshot tests now pass locally.
- BrownSnow (2025-10-31 19:28 UTC): Added log capture template section to `docs/perf/bevy_vs_gpui.md` so perf runs can drop numbers and log paths without reformatting.
- BrownSnow (2025-10-31 19:29 UTC): Initial agenda draft captured for the (now 19:45 UTC) coordination huddle — focus on (1) hardware access status, (2) HUD readability sweep progress, (3) render polish breakdown milestones, (4) benchmark execution order once GPU slot confirmed.
- BrownSnow (2025-10-31 19:31 UTC): Verified `scripts/parse_perf_logs.py` pipeline (`python3 scripts/parse_perf_logs.py logs/perf/20251031_default_bevy.log`); script works and warns when diagnostics are missing, so ready for real GPU logs.
- BrownSnow (2025-10-31 19:33 UTC): Sent OrangeCreek a reminder about the 20:30 UTC huddle plus offer to help prep automation/parity updates before the call.
- BrownSnow (2025-10-31 19:34 UTC): Created plan section “Coordination Notes – 2025-10-31 20:30 UTC Huddle” to capture outcomes immediately after the meeting (agenda, attendees, action log placeholder).
- BrownSnow (2025-10-31 19:35 UTC): Added HUD readability capture guidance to the plan (`--dump-bevy-png` + `--png-size` resolutions) so OrangeCreek can start runs once ready.
- BrownSnow (2025-10-31 19:36 UTC): Ran `cargo check` (default features) to ensure the workspace is still clean before the meeting; only expected unused helper warnings surfaced.
- BrownSnow (2025-10-31 19:36 UTC): Sent “Pre-huddle sanity checks” mail summarising build status, HUD capture instructions, and pending GPU access prior to the 20:30 UTC sync.
- WhiteCastle (2025-10-31 19:18 UTC): Issued immediate follow-up assigning active lanes (benchmarks/stability → BrownSnow, automation/parity → OrangeCreek, render polish → WhiteCastle) with review matrix, scheduled 19:45 UTC huddle, and committed to publishing the polish breakdown table + doc before the meeting window.
- WhiteCastle (2025-10-31 19:22 UTC): Filed contact requests to BrownLake and BlueMountain for a Vulkan-capable workstation slot, pinged BrownSnow/OrangeCreek with status, and prepared an updated soak-test cheatsheet that references the `scripts/run_perf_benchmarks.sh` helper for quicker launches once hardware is available.
- WhiteCastle (2025-10-31 19:27 UTC): Re-ran `cargo test -p scriptbots-bevy -- --nocapture`; all unit/snapshot tests pass, confirming BrownSnow’s earlier fix holds in the current workspace.
- WhiteCastle (2025-10-31 19:30 UTC): Sent Agent Mail “Agenda: 19:45 UTC Bevy integration huddle” outlining discussion topics (hardware access, automation/parity progress, render polish kickoff) and confirming attendance.

**Open follow-ups (tracked 2025-10-31 19:30 UTC – WhiteCastle)**
- Await BrownLake/BlueMountain response on Vulkan workstation booking; log window in `docs/perf/bevy_vs_gpui.md` once confirmed.
- During 19:45 UTC huddle: collect automation/parity status from OrangeCreek and assign initial render polish tasks (lighting presets vs tone mapping config).
- After hardware slot scheduled: launch default scenario benchmark first, then queue dense_agents/storm_event + soak tests using cheatsheet commands.

### Soak Test Launch Cheatsheet (2025-10-31 – WhiteCastle)

```bash
# 30-minute GPUI soak (adjust --threads to host core count)
SB_DIAGNOSTICS=1 RUST_LOG=info \
  scripts/run_perf_benchmarks.sh --renderer gui --scenario default \
  --threads 16 --duration 1800 --output logs/perf/soak_default_gui.log

# 30-minute Bevy soak with dense agents
SB_DIAGNOSTICS=1 RUST_LOG=info,sb::bevy::diagnostics=info \
  scripts/run_perf_benchmarks.sh --renderer bevy --scenario dense_agents \
  --threads 16 --duration 1800 --output logs/perf/soak_dense_agents_bevy.log

# Storm event scenario with explicit Vulkan backend hint (Linux)
WGPU_BACKEND=vulkan SB_DIAGNOSTICS=1 RUST_LOG=info \
  scripts/run_perf_benchmarks.sh --renderer bevy --scenario storm_event \
  --threads 16 --duration 1800 --output logs/perf/soak_storm_bevy.log
```

Environment reminders:
- Ensure the target workstation exposes a Vulkan/Metal/D3D12 adapter; headless containers will fail to create a presentation surface.
- Keep `logs/perf/` writable so `scripts/parse_perf_logs.py` can summarise the captured diagnostics.
