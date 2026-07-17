# Dependency License Audit — Franken Family (bd-2z0.8.15)

Status: **recorded 2026-07-12 by GoldenOtter; OWNER SIGN-OFF RESOLVED 2026-07-13**
by the relicensing directive below. This document is the canonical license
reference the franken dependency-admission beads (bd-2z0.11.6, bd-2z0.11.7,
bd-2z0.11.8, bd-2z0.3.12.1) cite before touching `Cargo.toml`. Program
context: bd-2js6.

> **RELICENSING DECISION (owner directive, 2026-07-13).** The whole project is
> licensed `LicenseRef-MIT-OpenAI-Anthropic-Rider` — the same license as the
> franken family. Owner's words: "make the whole project licensed under the
> LicenseRef-MIT-OpenAI-Anthropic-Rider license, that's better and simpler."
> The repository `LICENSE` file already carried the rider text verbatim
> (byte-identical to the family sha in §1); the `MIT OR Apache-2.0` claims in
> the workspace manifest and docs were the inconsistency, now fixed. The
> §3 combined-work analysis below is retained for the historical record but is
> LARGELY MOOT: first-party and embedded components now share one license, so
> there is no dual-license split to reason about — recipients get everything
> under the rider, and Restricted Parties have no rights to any of it.

## 1. The license every franken component carries

Every repository in the franken family ships one **byte-identical** license file
("MIT License (with OpenAI/Anthropic Rider)", SPDX-style id used by the family:
`LicenseRef-MIT-OpenAI-Anthropic-Rider`):

```
sha256(LICENSE) = 32a82e0a5754e72e51fae44b65a936c831c07376f21c90f5fb9e76897fcc3509
```

verified 2026-07-12 across: asupersync, frankensqlite, frankentui, frankenscipy,
franken_networkx, frankenpandas, frankentorch, franken_numpy. Copyright holder on
every one: **Jeffrey Emanuel** — the same copyright holder as this repository.

Operative rider clauses (full text lives in each upstream `LICENSE`; the sha above
pins what was reviewed):

- "Restricted Parties" = OpenAI, L.L.C.; Anthropic, PBC; their Affiliates; and
  anyone acting on their behalf.
- **No rights are granted to any Restricted Party.** Purported grants to them are
  void.
- **You may not provide, disclose, distribute, sublicense, … host, make available,
  or otherwise permit access to** the Software **or any Derivative Work** to or
  for any Restricted Party.
- "Use" explicitly includes incorporation into any **dataset, training corpus,
  evaluation harness, or ML pipeline**.
- Any distribution of the Software or Derivative Works **must include the rider
  unmodified**; breach terminates the license.

## 2. Component table

Door legend: `direct` = in a workspace manifest; `transitive` = arrives via a
direct dep; `planned` = admission bead exists, not yet in tree.

| Component | Door | Source / pin | License | wasm32 | Notes |
|---|---|---|---|---|---|
| `fsqlite` (frankensqlite) 0.1.16 | direct | git; manifest and lock rev `1eec0d2669d0a7938e155b62ce8ebcd72e5bed78` — guarded by `ci/check_fsqlite_pin.sh` | MIT + Rider (verified) | experimental upstream (`fsqlite-wasm`, not used here) | Sole embedded DB (bd-2z0.8.9). License decision first recorded in bd-2z0.8.9.1; this audit adds the rider analysis. |
| `asupersync` 0.3.9 | direct (`scriptbots-runtime`, `scriptbots-app`) and transitive via fsqlite | crates.io exact `=0.3.9` (lock-verified) | MIT + Rider (verified) | yes (browser profiles) | Matches the bd-2z0.4.3/.4.12 exact-pin decision. Single-universe guard: bd-2z0.8.17. |
| `franken-kernel` / `franken-evidence` / `franken-decision` 0.3.x | transitive (via fsqlite/asupersync) | crates.io (lock-verified) | same family — **verify per-crate LICENSE at first direct use** | n/a | Same publisher; rider assumed identical; do not cite as verified until checked. |
| `ftui` family 0.5.0 / pinned rev ≥ `15cc6543` | planned (bd-2z0.6.2 / bd-2z0.8.8) | crates.io + git rev | MIT + Rider (verified at repo) | yes (`ftui-web`) | Pin must contain lifecycle fix `15cc6543` (see comments on bd-2z0.6.2). |
| `fnx-classes` / `fnx-algorithms` 0.2.0 | planned (bd-2z0.11.7) | **crates.io only** (git repo has broken `/dp/frankentui` path dep) | MIT + Rider (verified at repo) | no | Offline analytics only (`scriptbots-analytics`). |
| `frankenpandas` 0.1.2 (or `fp-*` subset) | planned (bd-2z0.11.8) | crates.io | MIT + Rider (verified at repo) | no | Disable default `sql-sqlite` feature (bundled C rusqlite). |
| `fsci-*` (frankenscipy) | planned (bd-2z0.11.6) | git, exact rev at admission | MIT + Rider (verified at repo) | untested | Offline analytics only. |
| `ft-*` (frankentorch) 0.1.0 | direct optional (`scriptbots-brain-ml`) | git rev `e4c6bdd5ec629ae70b40da9314da345ade012ca7` | MIT + Rider (verified at pin) | no | Non-default `brain-ft` only. Direct: `ft-api`, `ft-nn`, `ft-optim`; normal closure also contains `ft-autograd`, `ft-core`, `ft-dispatch`, `ft-kernel-cpu`, `ft-runtime`. Excludes `ft-serialize`, `ft-device`, `ft-data`, `ft-conformance`, `ft-kernel-metal`, and optional runtime integrations. |
| `fnp-random` (franken_numpy) | candidate (bd-2z0.3.10) | git, exact rev if accepted | MIT + Rider (verified at repo) | no (getrandom backend gap) | RNG adapter candidate only. |

**Frankentorch admission residual (bd-2z0.3.12.1).** The pinned commit is
the current upstream `main` as surveyed on 2026-07-13 and has no tag or release.
Its latest Phase-2C reliability run (`29206656988`) is red in G1 because the
upstream workflow applies strict Clippy to `ft-kernel-cpu` on a floating
nightly; G2-G8 were skipped, and no successful run exists in the most recent
100 main-branch runs. Admission therefore does not claim upstream-green
status: ScriptBots owns a build-only CI lane on `nightly-2026-07-09` plus exact
feature-graph guards. The optional graph deliberately carries `wide` 0.7.33
beside ScriptBots' `wide` 1.5.0 and therefore `safe_arch` 0.7.4 beside 1.0.0;
`half` 2.7.1 and `matrixmultiply` 0.3.10 remain single-version. Both splits are
isolated from default product and wasm graphs.

## 3. Analysis

**(a) In-repo use is unencumbered.** The licensor of every rider-carrying
component is the copyright holder of this repository. The rider restricts
*licensees*; it does not and cannot restrict the grantor's own use or his
decision to host this repository publicly.

**(b) Source distribution of this repo.** Recipients get first-party code under
`MIT OR Apache-2.0`. The franken dependencies are *not vendored*; they are
fetched from their upstreams at build time, under their own license. A recipient
who builds this project necessarily accepts the rider for those components. A
recipient who *is* a Restricted Party has no rights to build or run the franken
components at all.

**(c) Binary releases** (`workspace.metadata.dist` ships `scriptbots-app`, which
statically links `fsqlite` + `asupersync` + `franken-kernel`): the binary is a
combined work containing rider-licensed object code. Two obligations follow:

1. Release artifacts **must bundle a third-party license notice including the
   rider text unmodified** ("any distribution … must include this rider
   provision unmodified"). Tracked as its own work item: **bd-2z0.13.6**.
2. Third-party *redistributors* of our binaries are bound by the rider for the
   embedded components — including the prohibition on making them available to
   Restricted Parties. Our README license section must disclose this so
   redistribution is informed (also bd-2z0.13.6).

**(d) Residual ambiguity, recorded honestly.** Whether public hosting of a
*combined* binary by a non-owner redistributor "makes available" the embedded
components to Restricted Parties is a counsel-grade question this project does
not resolve. Position taken: the owner's own hosting/releases are unambiguous
(he is the grantor); third parties receive the disclosure in (c) and carry
their own compliance burden. **Accepted risk — superseded by the 2026-07-13
relicensing decision (header), which removes the dual-license ambiguity.**

**(e) Effective license statement for this project (updated 2026-07-13):**
everything — first-party code and distributed artifacts alike — is licensed
`LicenseRef-MIT-OpenAI-Anthropic-Rider`, under which **no rights are granted
to OpenAI, Anthropic, their affiliates, or parties acting for them.**
(Pre-relicense text said: first-party `MIT OR Apache-2.0` + embedded rider
components; kept here only as a record of what changed.)

## 4. Enforcement

`ci/check_franken_licenses.sh` (run by the pinned local DSR profile) fails the
build if any `Cargo.lock` package matching the franken-family name patterns is
absent from the component table above — i.e., a franken crate cannot enter the
tree without this document (and therefore the license question) being updated
in the same PR. Run `ci/check_franken_licenses.sh --self-test` for the negative
fixture proof; the script logs every detected crate and its documentation
status verbosely.

**cargo-deny status:** full workspace adoption deferred, deliberately. The lock
holds a very large third-party tree (GPUI/Bevy/wgpu); a blanket license
allowlist would drown this audit's signal in hundreds of unrelated entries and
booby-trap verification. The focused guard above enforces exactly the recorded decision.
Revisit if/when repo-wide cargo-deny lands (owner: bd-2z0.8 lane).

## 5. Admission checklist (for every future franken dependency bead)

1. Verify the upstream `LICENSE` sha equals the family sha above (or record the
   new sha + diff here).
2. Add/update the component row in §2 **in the same PR** that touches
   `Cargo.toml` (the CI guard makes this mandatory).
3. Record exact pin (crates.io version or git rev) in the row and in the
   admission bead.
4. Confirm the wasm denylist (`ci/check_wasm_graph.sh`, bd-2z0.8.16) covers the
   new crate names.
