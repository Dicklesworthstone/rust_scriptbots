# Snapshot Harness Design (PLAN §1.3)

**Status:** Drafted by PurpleBear on 2025-10-30. Looking for co-owners to review, edit, and execute.

## Goals

1. Provide a deterministic command that renders the GPUI scene to PNG and exits without user interaction.
2. Persist golden PNGs (and ancillary metadata) so regressions can be detected via bitwise or perceptual diffing.
3. Integrate the harness into both local development (`cargo test` / simple script) and CI (Linux + Windows matrices).
4. Keep runtime under 30 seconds per job to avoid slowing down the pipeline.

## Deterministic Inputs

| Component | Choice | Notes |
| --- | --- | --- |
| Config file | `ci/configs/render_snapshot.toml` (new) | Derive from `replay_ci.toml`, but keep legacy world dimensions `6000×3000` and seed `424242` to match captures. |
| RNG seed | `424242` | Matches existing CI config; ensures agent placements are stable. |
| Renderer | CPU fallback (`render_png_offscreen`) for baseline; optional WGPU path behind `SB_WGPU_DUMP=1`. | GPU path will need GPU driver on CI; start with CPU to unblock testing. |
| Output size | `1600×900` | Aligns with legacy GLUT captures. |
| World warm-up | Simulate N ticks (e.g., 120) before snapshot so agents disperse slightly; implement as `--profile-steps` + snapshot or extend CLI to accept `--dump-png-after <ticks>`. |

## Command Sketch

```bash
cargo run -p scriptbots-app --features gui -- \
  --config ci/configs/render_snapshot.toml \
  --dump-png docs/rendering_reference/golden/rust_default.png \
  --png-size 1600x900 \
  --threads 1 \
  --low-power \
  --seed 424242  # (new CLI flag? or embedded in config)
```

### Required Enhancements

- `ScriptBotsConfig` already supports `rng_seed`; ensure the CLI honors it during bootstrap.
- Consider a new flag `--warmup-ticks <N>` to step the world before snapshotting (alternatively, run `cargo run ... --profile-steps` then reuse persisted world, but CLI restart cost is low).
- Ensure `render_png_offscreen` uses the same palette/camera defaults as live GPUI to avoid mismatch.

## Golden Assets Layout

```
docs/rendering_reference/
  golden/
    rust_default.png      # baseline snapshot
    legacy_default.png    # legacy GLUT reference (already captured)
  checksums.txt           # extend with SHA256 for rust_default.png
  diff/                   # optional: store latest diff artifacts for investigation
```

Metadata to record alongside PNG:
- SHA256 hash (append to `checksums.txt`).
- Render metadata JSON (proposal): `rust_default.meta.json` capturing RNG seed, tick count, camera zoom, etc., to aid debugging mismatches.

## Diff Strategy

1. **Primary:** Exact byte comparison vs. committed PNG (`cmp` or sha256). Suitable because harness uses fixed seed and offscreen renderer is deterministic.
2. **Secondary:** SSIM/perceptual diff (via `cargo install cargo-ssim` or embed `imgcmp` crate) to provide fuzzy match / visual heatmap. Consider enabling only when byte diff fails to keep runtime low.

Implementation idea:
- Write a Rust integration test under `crates/scriptbots-render/tests/snapshot.rs` that runs the harness command (or calls `render_png_offscreen`) and compares against `golden/rust_default.png`.
- On failure, emit diff metrics and optionally produce a diff PNG in `target/snapshot-failures/`.

## CI Wiring

1. Extend `.github/workflows/ci.yml` (render job) with a new step:
   ```yaml
   - name: Render snapshot harness
     run: cargo test -p scriptbots-render snapshot_golden -- --nocapture
   ```
2. Upload diff artifacts on failure using `actions/upload-artifact`.
3. Use path filters so changes under `crates/scriptbots-render`, `crates/scriptbots-world-gfx`, or `docs/rendering_reference` trigger the job.
4. Mirror step on Windows runner; ensure ImageMagick or alternative diff tool is available (or stick to Rust-only comparison to avoid external deps).

## Open Questions / Follow-ups

1. Should we also snapshot the terminal renderer (`SCRIPTBOTS_MODE=terminal`) for parity? (Nice-to-have.)
2. How do we inject camera HUD overlays to confirm they render correctly? Maybe capture multiple scenes (`rust_default.png`, `rust_selected_agent.png`).
3. If WGPU offscreen path is enabled in future, we must decide whether to store separate goldens (CPU vs GPU).
4. Coordinate with whoever implements viewport invariants to reuse the same config/seed for test inputs.

Please comment inline or edit this document; once we agree, we can start implementing and mark PLAN §1.3 bullets as `[Currently In Progress – <handle>]`.
