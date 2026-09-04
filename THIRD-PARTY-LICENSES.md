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
- `fnx-` family crates (`fnx-classes`, `fnx-algorithms`, `fnx-readwrite`, `fnx-runtime`, `fnx-dispatch`, `fnx-cgse`) (franken_networkx analytics graph library)
- `fp-` family crates (`fp-columnar`, `fp-frame`, `fp-groupby`, `fp-index`, `fp-runtime`, `fp-types`, `frankenpandas`) (frankenpandas analytics dataframe library)

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

## 3. Community dependencies and licenses

ScriptBots statically and dynamically links community crates from the Rust ecosystem.
Each of these dependencies is distributed under its own license, predominantly standard
permissive open-source licenses (MIT, Apache-2.0, BSD, ISC, Zlib, and Unicode).
The franken rider in §2 applies strictly to first-party code and franken-family components;
it does not alter or restrict the terms of third-party open-source components.

### 3.1 License category inventory

The following inventory categorizes the locked dependency graph (`Cargo.lock`) by license type:

| License | Crate Count | Notable Dependencies |
|---|---|---|
| **MIT OR Apache-2.0** (Dual) | ~600 | `tokio`, `serde`, `rayon`, `wgpu`, `bevy`, `fastmcp-rust`, `tracing`, `syn`, `quote`, `bitflags`, `clap`, `image` |
| **MIT** | ~280 | `gpui`, `anyhow`, `axum`, `bincode`, `num_cpus`, `ordered-float`, `parking_lot`, `crossbeam`, `futures` |
| **Apache-2.0** | ~75 | `arrow`, `parquet`, `utoipa`, `ab_glyph`, `approx`, `winit`, `wayland-client` |
| **BSD-3-Clause** / **BSD-2-Clause** | ~20 | `tiny-skia`, `ravif`, `curve25519-dalek`, `ed25519-dalek`, `exr`, `subtle`, `snap`, `rav1e` |
| **Zlib** / **Zlib-variant** | ~25 | `miniz_oxide`, `slotmap`, `bytemuck`, `wide`, `safe_arch`, `tinyvec`, `zlib-rs` |
| **ISC** | ~7 | `rustls-webpki`, `libloading`, `inotify`, `untrusted`, `ring` |
| **Unicode-3.0** / **Unicode-DFS** | ~20 | `unicode-ident`, `unicode-width`, `icu_*` family, `tinystr` |
| **Unlicense OR MIT** | ~12 | `memchr`, `aho-corasick`, `byteorder`, `csv`, `walkdir` |
| **MPL-2.0** | 3 | `cbindgen`, `dwrote`, `option-ext` |
| **GPL-3.0-or-later** (Transitive tooling) | 3 | `zlog`, `ztracing`, `ztracing_macro` (transitive internal components from Zed GPUI platform) |

### 3.2 Standard community license texts

#### MIT License

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

#### Apache License, Version 2.0

```text
                              Apache License
                        Version 2.0, January 2004
                     http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.
   "License" shall mean the terms and conditions for use, reproduction,
   and distribution as defined by Sections 1 through 9 of this document.
   "Licensor" shall mean the copyright owner or entity authorized by
   the copyright owner that is granting the License.
   "Legal Entity" shall mean the union of the acting entity and all
   other entities that control, are controlled by, or are under common
   control with that entity.
   "You" (or "Your") shall mean an individual or Legal Entity
   exercising permissions granted by this License.
   "Source" form shall mean the preferred form for making modifications,
   including but not limited to software source code, documentation
   source, and configuration files.
   "Object" form shall mean any form resulting from mechanical
   transformation or translation of a Source form, including but
   not limited to compiled object code, generated documentation,
   and conversions to other media types.
   "Work" shall mean the work of authorship, whether in Source or
   Object form, made available under the License.
   "Derivative Works" shall mean any work, whether in Source or Object
   form, that is based on (or derived from) the Work.
   "Contribution" shall mean any work of authorship that is intentionally
   submitted to Licensor for inclusion in the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   copyright license to reproduce, prepare Derivative Works of,
   publicly display, publicly perform, sublicense, and distribute the
   Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   patent license to make, have made, use, offer to sell, sell, import,
   and otherwise transfer the Work.

4. Redistribution. You may reproduce and distribute copies of the
   Work or Derivative Works thereof in any medium, with or without
   modifications, and in Source or Object form, provided that You
   meet the following conditions:
   (a) You must give any other recipients of the Work or Derivative
       Works a copy of this License; and
   (b) You must cause any modified files to carry prominent notices
       stating that You changed the files; and
   (c) You must retain, in the Source form of any Derivative Works
       that You distribute, all copyright, patent, trademark, and
       attribution notices from the Source form of the Work; and
   (d) If the Work includes a "NOTICE" text file as part of its
       distribution, You must include a readable copy of the attribution
       notices contained within such NOTICE file.

5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.

6. Trademarks. This License does not grant permission to use the trade
   names, trademarks, service marks, or product names of the Licensor.

7. Disclaimer of Warranty. Unless required by applicable law or
   agreed to in writing, Licensor provides the Work (and each
   Contributor provides its Contributions) on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied, including, without limitation, any warranties or conditions
   of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
   PARTICULAR PURPOSE.

8. Limitation of Liability. In no event and under no legal theory,
   whether in tort, contract, or otherwise, shall any Contributor be
   liable to You for damages, including any direct, indirect, special,
   incidental, or consequential damages of any character arising as a
   result of this License or out of the use or inability to use the
   Work.

9. Accepting Warranty or Additional Liability. While redistributing
   the Work or Derivative Works thereof, You may choose to offer,
   and charge a fee for, acceptance of support, warranty, indemnity,
   or other liability obligations.
```

#### BSD 3-Clause License

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

#### BSD 2-Clause License

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

#### ISC License

```text
Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```

#### Zlib License

```text
This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
```

#### Unicode License Agreement (Unicode-3.0)

```text
UNICODE, INC. LICENSE AGREEMENT - DATA FILES AND SOFTWARE

See Terms of Use <https://www.unicode.org/copyright.html>
for definitions of terms used in this license.

Permission is hereby granted, free of charge, to any person obtaining a copy
of the Unicode data files and any associated documentation (the "Data Files")
or Unicode software and any associated documentation (the "Software") to deal
in the Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the Data
Files or Software are furnished to do so, provided that either
(a) this copyright and permission notice appear with all copies of the Data
    Files or Software, or
(b) this copyright and permission notice appear in associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED
IN THIS NOTICE BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR
CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
OF THE DATA FILES OR SOFTWARE.
```

## 4. Maintenance and CI enforcement

- **Checked-in notice**: Maintained as `THIRD-PARTY-LICENSES.md` in the workspace root, packed into every release archive via `[workspace.metadata.dist] include = ["THIRD-PARTY-LICENSES.md"]` in `Cargo.toml`.
- **Rider integrity guard**: `ci/check_franken_licenses.sh --third-party` verifies that the embedded rider block in §2 matches canonical SHA-256 `32a82e0a5754e72e51fae44b65a936c831c07376f21c90f5fb9e76897fcc3509` exactly.
- **Dependency coverage guard**: `ci/check_franken_licenses.sh --third-party` verifies that every franken-family crate in `Cargo.lock` is named in §2.
- **Release archive verification**: Release pipelines extract all generated release archives (`.tar.xz`, `.tar.gz`, `.zip`) and verify that `THIRD-PARTY-LICENSES.md` is present and that the embedded rider block SHA-256 matches the canonical hash.
- **README link guard**: `ci/check_franken_licenses.sh --readme-guard` verifies that README.md license links point to valid existing files.
