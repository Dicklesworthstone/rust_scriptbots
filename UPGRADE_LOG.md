# Dependency Upgrade Log

This ledger records every dependency-source, version, feature, and toolchain
mutation in the ScriptBots workspace. Each entry is intentionally narrow: one
package or one inseparable family, one reviewed lockfile delta, and one explicit
rollback boundary. Research can happen in parallel; changes to manifests and
`Cargo.lock` cannot.

For each entry, record:

- the owning Bead;
- the old declaration and resolved version/source;
- the target declaration and resolved version/source;
- primary release, migration, security, MSRV, and licensing references;
- the exact allowed manifest and lockfile delta;
- focused and workspace verification evidence;
- build, binary-size, feature, and behavior impact where relevant;
- the result and a manual, path-scoped rollback procedure.

Do not combine repair work for an existing red baseline with a dependency
mutation. Stop an update when unrelated packages move, the pinned toolchain is
incompatible, scientific output changes without an approved version boundary,
licensing is unclear, or the migration crosses the review circuit breakers in
the rearchitecture plan.

## 2026-07-11 — Freeze the existing GPUI source resolution

- **Bead:** `bd-2z0.1.2`
- **Change class:** reproducibility source pin; not a GPUI version upgrade
- **Manifest:** root `Cargo.toml`
- **Before:** `gpui` used mutable Zed `branch = "main"`
- **Before resolution:** Zed commit
  `5f8a7413a31769e0882357f90dc424b3962ac72d`
- **Before lock SHA-256:**
  `789b53e73ba975a8482b14c60dc47f10648a3033c577de1d6fcf75b541ee5136`
- **Target:** the same Zed commit selected with exact
  `rev = "5f8a7413a31769e0882357f90dc424b3962ac72d"`
- **Allowed lock delta:** query normalization from `branch=main` to the exact
  `rev` for the 16 packages already resolved from that Zed commit; no registry
  version, checksum, or unrelated Git revision may change
- **Primary source:**
  <https://github.com/zed-industries/zed/commit/5f8a7413a31769e0882357f90dc424b3962ac72d>
- **Known-red before baseline:** `scriptbots-render` does not compile against
  this already-resolved commit because `flex_grow` now requires an `f32` and
  `Application::new` is absent. This pin preserves that red fingerprint; it does
  not repair or bless it.
- **Out of scope:** dated nightly selection, `rust-version`, GPUI API repair,
  removal of the unused `num-bigint-dig` patch, registry updates, and CI changes
- **Dry-run circuit breaker:** `cargo update --dry-run -p gpui --precise
  5f8a7413a31769e0882357f90dc424b3962ac72d` proposed the expected 16
  source-ID replacements but also tried to prune `windows 0.57.0` and its three
  companion registry packages. The write was stopped. Only the 16 source IDs
  were then normalized; the unrelated registry entries were preserved.
- **After lock SHA-256:**
  `c09eb8cb5b51d77ce1a5ae1e27c68ab31265d97cf43d04332d686cfbf4104337`
- **Verification:** two `cargo metadata --locked --no-deps --format-version 1`
  resolutions accepted the same lock/hash; all 16 Zed sources use the exact
  revision; no mutable Git query remains; `cargo fmt --all --check` passed; the
  default `scriptbots-app` host-target check passed with its pre-existing
  warning; the locked workspace/all-targets check reproduced the same known
  `scriptbots-render` GPUI/test-initializer failures.
- **Result:** accepted as a behavior-preserving source pin with no dependency
  version, checksum, feature, or registry-package delta
- **Rollback:** manually restore only the prior `gpui` declaration and the 16
  reviewed Zed source lines from this patch. Do not use checkout, reset, clean,
  or any operation that could overwrite unrelated work.
