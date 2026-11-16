# Browser brain families and event persistence: validation

Date: 2026-11-16 UTC.

## Published implementation

| Commit | Capability |
|---|---|
| [eac2c9c](https://github.com/Dicklesworthstone/rust_scriptbots/commit/eac2c9c847cc78d36fc5b674597e7dada639455a) | Bounded canonical domain-event inserts with checked identities, exact returned-row counts and transactional rollback. |
| [4433f24](https://github.com/Dicklesworthstone/rust_scriptbots/commit/4433f244de971c5a6b310551044a62043f9d7ff6) | Browser continuation of supported mixed-family registries, including empty/extinct families and explicit reset guards. |
| [9486777](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9486777ae6af272a332894ed0c971053804f20e0) | Reproducible mixed founders, version-two initial recipes, sharing/forks and browser creation controls. |

All current checks used the exact final functional commit, `9486777ae6af272a332894ed0c971053804f20e0`, in an isolated checkout. The subsequent progress/evidence commit adds documentation and tracker records only. Independent source review covered all production changes and authored regressions.

## Observed results

| Check | Actual result | Evidence |
|---|---|---|
| JavaScript recipe unit tests | **10 passed, 0 failed, 0 skipped.** Includes explicit version-two mixed-recipe admission and rejection. | [Log](current-fast/check-1.log), [receipt](current-fast/receipt.json) |
| `cargo fmt --check` | Process exit 0 and an aggregate DSR success, **but the log prints eight formatting diffs** in unchanged `crates/scriptbots-app/src/terminal/frankentui_shell.rs`. No web/storage diffs were emitted. Clean workspace formatting is not established. | [Log](current-fast/check-2.log) |
| Real Chromium DOM lifecycle file | Exit 1 before page creation. The verified browser executable aborts on `socket() failed: Operation not permitted`, followed by denied ptrace and SIGABRT. **Zero DOM assertions ran.** | [Log](current-dom/check-1.log), [receipt](current-dom/receipt.json) |
| Native `browser_mixed` test/build probe | Exit 124 with a configured 300-second deadline. Last recorded progress is acquisition of pinned dependencies; no compiler/test result was observed. | [Log](current-native/check-1.log), [receipt](current-native/receipt.json) |
| Workspace check, Clippy, storage Rust tests, generated WASM and scientific browser execution | Unreached/unpassed. | No success claimed. |
| Throughput benchmark | Not executed. | No speedup claimed. |

The aggregate status in the fast-lane receipt must not be interpreted as clean workspace formatting: the raw formatter diagnostics contradict that conclusion. The positive executable evidence is the ten recipe unit tests.

The configured Rust toolchain was `nightly-2026-08-31-x86_64-unknown-linux-gnu`; execution used DSR 0.2.3 directly. No GitHub Actions were used. The raw receipts record the exact commands, timeouts, configuration hashes and source signatures. All three current receipts have identical before/after source signatures:

```text
9486777ae6af272a332894ed0c971053804f20e0
273da66f0fb0341d516c63b7fa21bbcebc963efb2b1124867d4ef0c017eeb88f
e82509b8f8cf18ccdc4a91637bb6e8be314232f9347b804b162b7f17966ad73d
```

## Earlier attempts retained

The [first earlier receipt](earlier-fast-1/receipt.json) and [second earlier receipt](earlier-fast-2/receipt.json) concern `4433f244`. They retain the toolchain-bootstrap/format timeouts and missing default Playwright browser prerequisite. Their recipe observations are superseded by the current ten-test result and are not added to its count.

Each run directory contains its original receipt, frozen check configuration and complete command logs. The eight log hashes and five configuration snapshots were checked against their receipts before preservation. Files retain their original bytes, including unsuccessful outcomes and the formatter-status discrepancy.

## Qualification boundary

The implementation includes authored real-owner/archive regressions for mixed-family continuation, actual per-family offspring, genome/evaluator state, version-one recipe compatibility, bounded insertion chunks and late rollback/retry. Their Rust execution remains pending. The DOM fixture exercises the real page with a WASM test double; even successful DOM execution would be separate from Rust codec and scientific qualification.

The original `bd-ywtv` browser journal, SQLite-image recovery, durable-watermark and crash-restart acceptance remains open. The original `bd-50bg` reliability and same-invocation throughput acceptance remains open. Progress was recorded through `br` without changing either bead's criteria, ownership or in-progress status.
