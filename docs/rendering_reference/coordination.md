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
