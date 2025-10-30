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
- PinkMountain: Claimed PLAN §1.3 viewport invariants; aligning tests with new camera module. Will share progress updates as camera Stage 1 lands.
- PinkMountain: Drafted Stage 1 work breakdown for camera refactor (`camera_stage1_plan.md`); Stage 2 wiring underway pending visuals/CI ownership.
- PinkMountain: Owning Stage 2 camera wiring (GPUI + terminal/offscreen integration) per PLAN §2.3/§2.4 updates.
- RedCastle: Regenerated all legacy reference PNGs (`legacy_default.png`, `legacy_selected_agent.png`, `legacy_food_peak.png`, `legacy_food_off.png`, `legacy_zoomed_hud.png`), refreshed SHA256s in `checksums.txt`, and published capture helper `capture_legacy_render.sh`.
- RedCastle: Added SVG overlays (`legacy_*_overlay.svg`) aligned with ROI tables and updated spec/plan to mark PLAN §1.2 complete.
- RedCastle: Drafted `visual_polish_plan.md` outlining palette/legibility work and claimed PLAN §3.1–§3.2 pending Stage 2 stability.
