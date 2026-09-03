# ScriptBots Analytics: NaN vs. NULL Missingness Policy

**Tracking Bead:** `bd-2z0.11.8`  
**Applies to:** `scriptbots-analytics`, `scriptbots-storage`, Parquet/Arrow/CSV exporters, FrankenPandas integration.

---

## 1. Executive Summary

In scientific simulation analytics, conflating **arithmetic non-numbers (`NaN`)** with **missing data (`NULL`)** causes silent corruption, statistical bias, and type-coercion bugs. 

`ScriptBots` enforces a strict, zero-ambiguity policy:
- **`NULL` (Missingness):** Indicates that no observation exists (e.g., an agent has died, a metric was not sampled on a given tick, or a lineage parent is unknown). Represented physically via Arrow validity bitmaps and `Scalar::Null`.
- **`NaN` (IEEE-754 Non-Number):** Indicates a failed arithmetic operation (e.g. `0.0 / 0.0`, `sqrt(-1.0)`). It is a valid floating-point bit pattern, NEVER a missing-value sentinel.
- **No Sentinel Numbers:** Values such as `-1`, `-999.0`, or `9999` MUST NEVER be used to denote missingness.

---

## 2. Historical Context: The Pandas Dilemma

In legacy Python pandas (prior to the introduction of nullable extension types and the Arrow backend):
1. NumPy lacked a native missingness indicator for integer, boolean, and string arrays.
2. Missing values in integer columns forced silent upcasting to `float64` to use IEEE `np.nan` as a missingness sentinel.
3. This led to precision loss on 64-bit integer IDs (such as `agent_uid` and `tick`), unexpected floating-point comparison semantics (`nan != nan`), and incorrect truthiness behavior in conditional filters.

FrankenPandas (`fp-frame`, `fp-columnar`) and Apache Arrow resolve this at the architectural level:
- Validity bitmaps exist independently of physical value storage.
- Every data type (integers, strings, booleans, floating-point numbers) supports native `NULL` without type mutation or sentinel overloading.

---

## 3. Data Representation Matrix

| Storage Layer | Representation of `NULL` (Missing) | Representation of `NaN` (Arithmetic Error) |
| :--- | :--- | :--- |
| **FrankenSQLite (`fsqlite`)** | SQL `NULL` | Stored as IEEE-754 `0x7ff8000000000000` (disallowed in valid metrics) |
| **FrankenPandas (`fp-frame`)** | `Scalar::Null(NullKind)` + bit unset in validity bitmap | `Scalar::Float64(f64::NAN)` + bit set in validity bitmap |
| **Apache Arrow (`RecordBatch`)** | Validity bitmap bit is `0` (null count incremented) | Value buffer contains NaN float, validity bit is `1` |
| **Apache Parquet** | Definition level `< max_definition_level` | Plain or dictionary float with IEEE NaN value |
| **CSV Exports** | Empty field: `, ,` (between delimiters) | Explicit literal string: `NaN` |

---

## 4. ScriptBots Domain Conventions

### 4.1 Agent Lifecycle & Kinship
- **Founders:** Agents introduced at world initialization have no parents. In `lineage_edges`, parent ordinals or founder edge markers must reflect unlinked status via relational absence or explicit null indicators, never dummy UIDs (`0` is a valid agent UID).
- **Death & Observations:** When an agent dies, it ceases to emit rows in `agents`. A missing row at a later tick indicates death, not a row populated with `NaN` coordinates or zeros.

### 4.2 Metric Series & Rolling Aggregations
- **Irregular Ticks:** Metrics sampled at variable frequencies remain sparse in storage. When constructing regular time series in FrankenPandas, missing intervals are populated with `NULL`.
- **Rolling Windows:**
  - Rolling aggregations default to `min_periods = 1` during warmup epochs to compute expanding means without producing `NaN` or `NULL` during the ramp-up phase.
  - When `min_periods > window_size`, periods with insufficient observations evaluate to `NULL` (masked by validity bitmap), ensuring statistical rigor.
- **Variance and Standard Deviation:**
  - If a group contains `N = 1` observations, sample variance is mathematically undefined. In summary tables, sample standard deviation for singletons is reported as `0.0` with explicit population count `1`, avoiding downstream rendering errors.

### 4.3 Export Round-Trip Conformance Net
- Exporters (`sb-analyze export`) re-read written Parquet and Arrow files to guarantee that:
  1. `null_count` matches identically between the written `RecordBatch` and the re-read batch.
  2. Floating-point values do not undergo lossy rounding that turns denormalized floats into zeros or NaNs.
  3. CSV lines with empty fields parse unambiguously to `NULL`, while preserving string columns containing empty strings.

---

## 5. Summary Table Policy for CLI & Markdown Reports

When rendering human-readable markdown summaries (`sb-analyze summarize`):
- Unobserved groups display a hyphen `"-"` or explicit `0` count with `(no agents recorded)`.
- Valid float quantities are formatted with fixed precision (e.g. `{:>6.2}`).
- If an arithmetic `NaN` is detected in any summary metric, the CLI emits a warning diagnostic, identifying potential simulation math anomalies (e.g. division by zero in energy conservation).
