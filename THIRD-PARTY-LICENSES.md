# Third-Party License Notices for ScriptBots Release Artifacts

This file ships inside every ScriptBots release archive (enforced by
`.github/workflows/release.yml`; kept current by
`ci/check_franken_licenses.sh --third-party` in CI — see
`docs/licenses.md` for the full audit this file operationalizes,
tracked as bd-2z0.13.6).

## 1. First-party code

ScriptBots itself (all `scriptbots-*` crates) is licensed under
**`LicenseRef-MIT-OpenAI-Anthropic-Rider`** — the same license reproduced in
§2 below; the repository `LICENSE` file carries it verbatim (owner
relicensing decision, 2026-07-13; formerly declared `MIT OR Apache-2.0`).

## 2. Franken-family components — MIT with OpenAI/Anthropic Rider

Default ScriptBots binaries statically link the following components, which are
**not** plain MIT/Apache and whose license must accompany any distribution
of this software or derivative works, unmodified, per its own terms:

- `fsqlite` and its `fsqlite-*` member crates (FrankenSQLite)
- `asupersync`
- `franken-kernel`, `franken-evidence`, `franken-decision`

Builds that explicitly enable the non-default `brain-ft` feature additionally
admit and compile the pinned Frankentorch family: `ft-api`, `ft-nn`, `ft-optim`,
`ft-autograd`, `ft-core`, `ft-dispatch`, `ft-kernel-cpu`, and `ft-runtime`.
The future `FtBrain` adapter (`bd-2z0.3.12.3`) will link that family into the
application. Default release products do not embed these optional crates.
`ft-serialize` and its runtime integrations are deliberately excluded.

Each listed component carries the byte-identical license reproduced in full below
(canonical upstream sha256
`32a82e0a5754e72e51fae44b65a936c831c07376f21c90f5fb9e76897fcc3509`).

**Practical consequences for recipients and redistributors:**
1. No rights to these components are granted to OpenAI, Anthropic, their
   affiliates, or parties acting for them ("Restricted Parties").
2. If you redistribute ScriptBots binaries or derivatives, you are bound by
   the rider for the embedded components — including the obligation to keep
   this notice intact and the prohibition on providing the software to or
   for Restricted Parties.

### License text (verbatim)

BEGIN LICENSE TEXT
MIT License (with OpenAI/Anthropic Rider)

Copyright (c) 2026 Jeffrey Emanuel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

ADDITIONAL RIDER / RESTRICTION (OpenAI / Anthropic):

This rider is part of the "conditions" of this License. In the event of any
conflict between this rider and any other portion of this License, this rider
controls.

"Restricted Parties" means OpenAI, L.L.C.; Anthropic, PBC; any of their
respective Affiliates; and any person or entity acting directly or indirectly
on behalf of, for the benefit of, or under the direction of any of the
foregoing (including any officer, director, employee, contractor, agent,
consultant, service provider, or representative).

Notwithstanding any other provision of this License, no rights are granted to
any Restricted Party. Any purported license, sublicense, assignment, transfer,
or other permission to any Restricted Party is null and void absent the
express prior written permission of Jeffrey Emanuel.

You may not provide, disclose, distribute, sublicense, sell, lease, lend,
host, make available, or otherwise permit access to the Software or any
derivative work of the Software (as defined in applicable copyright law)
(collectively, "Derivative Works") to or for any Restricted Party.

For purposes of this rider, "use" includes, without limitation: copying,
modifying, merging, publishing, distributing, sublicensing, selling,
transferring, making available, hosting, deploying, executing, benchmarking,
testing, analyzing, indexing, or incorporating the Software or any Derivative
Works into any dataset, training corpus, evaluation harness, or pipeline for
machine learning or other automated systems.

This rider applies to the Software and all Derivative Works. As a condition of
use, you agree that this rider is a precondition to exercising any rights
under this License, and you agree that any distribution of the Software or any
Derivative Works must include this rider provision unmodified.

Any breach of this rider automatically and immediately terminates the
permissions granted by this License. Upon termination, you must immediately
cease all use and distribution of the Software and any Derivative Works and
destroy all copies under your control.

You agree that a breach of this rider would cause irreparable harm and that
Jeffrey Emanuel may seek injunctive or other equitable relief to enforce this
rider, in addition to any other remedies available at law. To the maximum
extent permitted by applicable law, the prevailing party in any action to
enforce this rider shall be entitled to recover reasonable attorneys' fees and
costs.

For purposes of this rider, "Affiliate" means any entity that directly or
indirectly controls, is controlled by, or is under common control with the
specified party. "Control" means ownership of more than 50% of the voting
securities or other ownership interest, or the power to direct management or
policies by contract or otherwise.

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
END LICENSE TEXT

## 3. Other third-party dependencies

ScriptBots additionally links many community crates (GPUI, Bevy, wgpu,
tokio/axum, rayon, serde, and others), each under its own license
(predominantly MIT and/or Apache-2.0). Those licenses are unmodified and
unaffected by §2. A complete machine-readable inventory can be produced from
the exact locked dependency set with `cargo license` (or `cargo about`) at
the pinned workspace revision. This file exists specifically to satisfy the
§2 rider's include-with-distribution obligation; it does not replace those
crates' own notices.

MAINTENANCE. Decision recorded (bd-2z0.13.6): checked-in notice file + CI
staleness guard, rather than build-time generation — cargo-about would add a
toolchain dependency to the release path for a file whose legally-operative
section is static. `ci/check_franken_licenses.sh --third-party` fails CI if a
franken-family crate appears in Cargo.lock without being named here, or if
the embedded license text drifts from the canonical sha.
