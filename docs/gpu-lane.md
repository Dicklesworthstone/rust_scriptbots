# GPU Compute Sense Lane & Parity Gate

> Status, hardware target matrix, determinism guarantees, and parity gate verification for GPU-accelerated sensing (bd-16g.15.2, bd-16g.15.3).

---

## 1. Executive Summary

GPU sensing provides an order-independent, fixed-point fast lane for agent perception in `rust_scriptbots`.
Floating-point sensor accumulation on GPUs is fundamentally non-deterministic across hardware vendors and workgroups due to non-associative floating-point addition and driver-dependent transcendental approximations (`sin`, `cos`, `atan2`, `acos`).

To prevent silent determinism drift:
1. **CPU Remains the Default**: Simulations run on the CPU reference implementation unless explicitly configured otherwise.
2. **Bit-Identical Fixed-Point Math**: Geometry is evaluated via dot products and Abramowitz & Stegun polynomial `acos` matching `crates/scriptbots-core/src/sense_fixed.rs` verbatim. Sensor terms are converted into 20-bit fixed-point integers (`to_fixed`) and accumulated into 64-bit integers using tree reductions in shared memory.
3. **Parity Gate Per Target**: Every supported adapter and driver combination must pass an end-to-end 1,000-tick CPU-vs-GPU parity verification run.
4. **Honest Reproducibility Labels**: Uncertified or approximate runs are marked `reproducible = false` in `RunManifestV3`, are tagged with startup and exit warnings, and are excluded from replay certification and competitive leaderboard rankings.

---

## 2. Hardware & Target Matrix

| Target Architecture | Graphics API | Candidate Adapters | Hardware Class | Verification Status | Determinism Verdict |
|----------------------|--------------|-----------------|----------------|---------------------|---------------------|
| `x86_64-unknown-linux-gnu` | Vulkan | AMD Radeon RX 7900 XTX | Discrete GPU | Unverified target | Unestablished |
| `x86_64-unknown-linux-gnu` | Vulkan | NVIDIA GeForce RTX 4090 | Discrete GPU | Unverified target | Unestablished |
| `x86_64-unknown-linux-gnu` | Vulkan | llvmpipe (Mesa) | Software CPU | Emulated (Excluded from Performance Claims) | Approximate |
| `aarch64-apple-darwin` | Metal | Apple M1/M2/M3/M4 (Family 7/8/9) | Integrated Apple Silicon | Unverified target | Unestablished |
| `x86_64-pc-windows-msvc` | DirectX 12 | NVIDIA GeForce RTX 3080/4080 | Discrete GPU | Unverified target | Unestablished |

No hardware certification is established by this table. The retained fixture is a
software-adapter report; a table row is not parity evidence. Runtime certification
requires a matching full adapter/driver/backend tuple, current shader, canonical
fixture, at least 1,000 ticks with actual sensor comparisons, equal nonzero science
digests, zero delta/no divergence, and the recomputed versioned report checksum.
Missing adapter/driver identity strings refuse certification even when both
the report and live adapter expose the same blank value.
That checksum detects corruption; it does not authenticate the report's author.

Builds seeking certification must embed `SCRIPTBOTS_SOURCE_TREE_CLEAN=true`, an
exact `SCRIPTBOTS_SOURCE_REVISION` (or plain-commit `RCH_BUILD_SOURCE`),
`SCRIPTBOTS_BUILD_TARGET`, and complete `SCRIPTBOTS_RUSTC_VV` output while compiling.
The identity also binds Cargo manifests/lockfile and local feature/code-generation
settings. Missing inputs remain `unavailable` in new reports and refuse certification.
These build-time declarations require independent build provenance review; runtime
environment variables cannot replace them. Legacy v0 checksums are not accepted.

> [!NOTE]
> **Software Adapter Exclusion**:
> Software fallback adapters like `llvmpipe` run on CPU threads emulating GPU pipelines. They are visibly classified as software adapters and excluded from all official GPU scaling and performance benchmarks.

---

## 3. CLI Policy & Flag Semantics

| Flag | Values | Default | Purpose |
|------|--------|---------|---------|
| `--sense-backend` | `cpu`, `gpu`, `auto` | `cpu` | Selects sensory execution backend. `cpu` is default. `gpu` requires compatible GPU. `auto` chooses verified GPU or falls back honestly to CPU. |
| `--allow-approximate-sense` | Flag (bool) | `false` | Explicit opt-in required to execute on uncertified GPU targets or approximate adapters. |

### Failure Modes Before Storage Writes:
- **No Adapter**: Passing `--sense-backend gpu` on a machine without a supported GPU returns a typed pre-storage error. Silent fallback to CPU is forbidden.
- **Uncertified Adapter**: Passing `--sense-backend gpu` on an uncertified target without `--allow-approximate-sense` returns a typed pre-storage error detailing required guidance.
- **Auto Selection**: `--sense-backend auto` probes for certified bit-exact GPU hardware. If certified, it selects GPU; otherwise, it honestly logs and selects the CPU lane.

---

## 4. Manifest & Downstream Propagation

When GPU sensing is classified as `Approximate`:
1. `RunManifestV3.reproducible` is forced to `false`.
2. `RunManifestV3.warnings` retains `"gpu sensing is approximate; run is not certified as reproducible"`.
3. `RunManifestV3.sense_policy` records `SensePolicyV0` with `SenseDeterminism::Approximate` and gate evidence.
4. **Replay Certification**: Replay verification checks refuse to certify approximate runs.
5. **Leaderboards**: Tournament rankings exclude approximate runs from competitive leaderboards.
6. **Exploration Export**: Raw data export remains permitted for scientific exploration.

---

## 5. Running the Parity Verification Gate

To run the current focused provider, policy and documentation checks:

```bash
# Execute the E2E parity runner
./scripts/e2e_gpu_sense_parity.sh
```

The helper:
1. Runs selected parity tests, including a five-tick accumulator-perturbation
   diagnostic and a one-tick empty-population case when an adapter is available.
   The provider test can skip execution when no adapter/device is available.
2. Checks application policy/manifest tests and the retained software-adapter
   fixture. It does not produce a fresh 1,000-tick report from that fixture.
3. Verifies this documentation against its generator.

The historical `DriverAcos` and `F32Accumulator` mode names currently select
integer accumulator perturbations, not altered shader implementations. Actual
driver-`acos` and f32-shader fault qualification remains open, along with the
1,000-tick hardware run and DSR scaling requirements of `bd-16g.15.3`.
`run_sense_parity_check` defaults to 1,000 ticks for callers that explicitly run
it; the helper's smaller diagnostic does not establish that acceptance.
