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

| Target Architecture | Graphics API | Tested Adapters | Hardware Class | Verification Status | Determinism Verdict |
|----------------------|--------------|-----------------|----------------|---------------------|---------------------|
| `x86_64-unknown-linux-gnu` | Vulkan | AMD Radeon RX 7900 XTX | Discrete GPU | Certified | Exact |
| `x86_64-unknown-linux-gnu` | Vulkan | NVIDIA GeForce RTX 4090 | Discrete GPU | Certified | Exact |
| `x86_64-unknown-linux-gnu` | Vulkan | llvmpipe (Mesa) | Software CPU | Emulated (Excluded from Performance Claims) | Approximate |
| `aarch64-apple-darwin` | Metal | Apple M1/M2/M3/M4 (Family 7/8/9) | Integrated Apple Silicon | Certified | Exact |
| `x86_64-pc-windows-msvc` | DirectX 12 | NVIDIA GeForce RTX 3080/4080 | Discrete GPU | Certified | Exact |

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

To execute the 1,000-tick CPU-vs-GPU parity verification suite:

```bash
# Execute the E2E parity runner
./scripts/e2e_gpu_sense_parity.sh
```

The runner:
1. Executes 1,000 ticks comparing CPU reference against GPU compute sensing.
2. Injects driver `acos` and `f32` accumulator faults to prove negative gate behavior (divergence detection).
3. Validates the resulting `sense_lane_parity.json` report artifact.
4. Verifies this documentation against the code generator to prevent doc drift.
