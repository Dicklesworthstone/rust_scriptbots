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
  - `sb-analyze summarize --rolling-window N` uses the last `N` recorded observations of each metric, including the current observation. It does not fill gaps or interpret `N` as elapsed ticks; sparse series remain sparse.
  - Rolling aggregations default to `min_periods = 1` during warmup epochs to compute expanding means without producing `NaN` or `NULL` during the ramp-up phase.
  - When `min_periods > window_size`, periods with insufficient observations evaluate to `NULL` (masked by validity bitmap), ensuring statistical rigor.
- **Variance and Standard Deviation:**
  - If a group contains `N = 1` observations, sample variance is mathematically undefined. In summary tables, sample standard deviation for singletons is reported as `0.0` with explicit population count `1`, avoiding downstream rendering errors.

### 4.3 Export Round-Trip Conformance Net

Use `--verify` to compare the complete exported artifact with a second traversal
of the same finished database:

```bash
sb-analyze runs/example.sqlite export --format parquet --out-dir exports --verify
```

All five canonical tables use keyset pages containing at most 1,024 rows. Arrow
files contain bounded record batches; Parquet seals a row group after each page.
Verification walks source and decoded rows together even when their batch
boundaries differ. It checks column order, types, nullability, provenance metadata,
every value and null position, and complete decoded-stream exhaustion. Floating
values are compared by their IEEE bit patterns, including signed zero. Missing
rows, extra rows and decoder errors fail verification.

CSV verification compares every encoded byte with a fresh source traversal,
including the header, escaped rows and final completion receipt. It reuses the
original timestamps and refuses bytes after the receipt. A self-consistent
checksum on altered rows is insufficient. This verifies the canonical CSV text;
the existing text encoding does not distinguish an absent optional string from
an empty string. Use Arrow or Parquet when typed null semantics are required.

The finished-reader lease keeps the source fixed throughout export and readback.
The page limit bounds retained row buffers, not individual variable-length
payloads, SQL engine working memory, file footer metadata or wall-clock duration.

---

## 5. Summary Table Policy for CLI & Markdown Reports

When rendering human-readable markdown summaries (`sb-analyze summarize`):
- Unobserved groups display a hyphen `"-"` or explicit `0` count with `(no agents recorded)`.
- Valid float quantities are formatted with fixed precision (e.g. `{:>6.2}`).
- Non-finite mandatory agent fields or metric observations are refused before aggregation. They are not converted to missing observations or silently dropped.

Summary inputs use the same finished-reader lease and 1,024-row export pages.
Per-epoch `population` counts agent observations within that epoch, including
repeat observations of the same agent; it is not a distinct-agent count. Two
agent traversals preserve the original source-order means and sample standard
deviations while retaining group accumulators instead of every input sample.

Rolling rows are collected directly into the report and ordered by metric name
and tick. Each metric is passed to the pinned upstream FrankenPandas rolling
implementation as one complete series, so page boundaries cannot reset its
compensated arithmetic. Lineage pages feed the existing descendant graph directly.
The report, lineage graph and one metric's rolling workspace still grow with their
respective sizes; paging bounds source-row buffers, not total report memory or
SQL execution time.
