#!/usr/bin/env python3
"""Conservative audit of declaration-only public Rust enum variants.

The first-pass production signal is intentionally broad: a public variant with
no textual value-production site in non-test crate source. Match/let patterns
and comparisons are consumption rather than production. Inline and integration
tests are counted separately; the publishable raw set has no constructor in
either stratum, while test-only constructors remain visible in the JSON. Neither
signal is a finding. The refined candidate set removes measured
generated-construction classes and conservatively treats ``Self::Variant`` and
variant imports as references:

* a variant field carrying ``#[from]`` (``thiserror`` generates construction);
* a ``#[default]`` variant (``derive(Default)`` generates construction);
* an enum deriving ``Deserialize`` (Serde constructs from input);
* ``Self::Variant``, explicit/group imports, or wildcard imports.

The remaining rows still require human adjudication. Rust macro expansion and
full name resolution are deliberately outside this small source audit. The
mechanical rules report evidence; the small path-qualified disposition table
records reviewed decisions and their rationale. The tool never edits source.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
VariantIdentity = tuple[str, str, str]
VariantPair = tuple[str, str]
PUBLIC_ENUM_RE = re.compile(
    rf"(?P<attrs>(?:\s*#\s*\[[^\]]*\]\s*)*)"
    rf"\bpub\s+enum\s+(?P<name>{IDENTIFIER})"
    rf"(?:\s*<[^{{;]*>)?(?:\s+where[^{{]*)?\s*\{{",
    re.MULTILINE,
)
VARIANT_RE = re.compile(
    rf"^(?P<attrs>(?:\s*#\s*\[[^\]]*\]\s*)*)\s*(?P<name>{IDENTIFIER})\b",
    re.DOTALL,
)
RAW_STRING_RE = re.compile(r'(?:br|r)(?P<hashes>#{0,255})"')
LIFETIME_RE = re.compile(r"'[A-Za-z_][A-Za-z0-9_]*")
KNOWN_NON_ORPHANS = (
    ("crates/scriptbots-analytics/src/lib.rs", "AnalyticsError", "Json"),
    ("crates/scriptbots-analytics/src/lib.rs", "AnalyticsError", "Io"),
    ("crates/scriptbots-app/src/control.rs", "KnobKind", "String"),
    ("crates/scriptbots-app/src/control.rs", "KnobKind", "Array"),
    ("crates/scriptbots-app/src/control.rs", "KnobKind", "Object"),
    ("crates/scriptbots-app/src/control.rs", "KnobKind", "Null"),
    (
        "crates/scriptbots-bevy/src/particles.rs",
        "OverflowPolicy",
        "DropOldestAmbient",
    ),
    ("crates/scriptbots-world-gfx/src/lib.rs", "ReadbackError", "Resize"),
)
KNOWN_REFINED_CANDIDATES = (
    (
        "crates/scriptbots-analytics/src/stats.rs",
        "EffectSizeEstimator",
        "HedgesG",
    ),
    ("crates/scriptbots-app/src/lib.rs", "ScenarioError", "NotATable"),
)
MANUAL_DISPOSITIONS = {
    (
        "crates/scriptbots-analytics/src/stats.rs",
        "EffectSizeEstimator",
        "HedgesG",
    ): {
        "classification": "required_unreached",
        "decision": "keep_and_construct_when_corrected_estimator_is_selected",
        "rationale": (
            "The public identity was added by bd-k3f3 so corrected effect results can say "
            "which estimator produced them. Certification deliberately still selects and "
            "labels CohensD; constructing HedgesG only in a test would fake production reachability."
        ),
    },
    (
        "crates/scriptbots-app/src/lab/notebook.rs",
        "Support",
        "Descriptive",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_superseded_lab_notebook_module",
        "rationale": (
            "No code outside lab/notebook.rs consumes NotebookRenderer or constructs this "
            "support form. bd-16g.1.5 actually landed the live notebook implementation in "
            "lab_assistant.rs, leaving this parallel module superseded."
        ),
    },
    (
        "crates/scriptbots-app/src/lab/notebook.rs",
        "Support",
        "Effect",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_superseded_lab_notebook_module",
        "rationale": (
            "Only inline tests construct this sibling of Descriptive; no code outside "
            "lab/notebook.rs consumes NotebookRenderer or Support. bd-16g.1.5 landed the live "
            "notebook implementation in lab_assistant.rs, leaving this module superseded."
        ),
    },
    (
        "crates/scriptbots-app/src/lab/stats.rs",
        "Correction",
        "None",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_owner_authorized_bd_7453_supersession",
        "rationale": (
            "The comparison reference does not make the never-wired lab/stats layer live. "
            "bd-7453 establishes that the layer is superseded; removal belongs to its "
            "owner-authorized module-level resolution."
        ),
    },
    (
        "crates/scriptbots-app/src/lab/stats.rs",
        "Correction",
        "HolmBonferroni",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_owner_authorized_bd_7453_supersession",
        "rationale": (
            "Serde can construct the value, but bd-7453 establishes that the whole lab/stats "
            "layer was never wired and is superseded by scriptbots-analytics. Removal belongs "
            "to the owner-authorized module-level resolution."
        ),
    },
    (
        "crates/scriptbots-app/src/lab/stats.rs",
        "Correction",
        "BenjaminiHochberg",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_owner_authorized_bd_7453_supersession",
        "rationale": (
            "Serde can construct the value, but bd-7453 establishes that the whole lab/stats "
            "layer was never wired and is superseded by scriptbots-analytics. Removal belongs "
            "to the owner-authorized module-level resolution."
        ),
    },
    (
        "crates/scriptbots-app/src/lab/stats.rs",
        "TestName",
        "SpearmanRank",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_owner_authorized_bd_7453_supersession",
        "rationale": (
            "This identifier has no producer in the never-wired lab/stats layer. bd-7453 "
            "establishes that the layer is superseded; remove it only with that owner-authorized "
            "module-level resolution."
        ),
    },
    (
        "crates/scriptbots-app/src/lab/stats.rs",
        "TestName",
        "BootstrapCi",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_owner_authorized_bd_7453_supersession",
        "rationale": (
            "This identifier has no producer in the never-wired lab/stats layer. bd-7453 "
            "establishes that the layer is superseded; remove it only with that owner-authorized "
            "module-level resolution."
        ),
    },
    (
        "crates/scriptbots-app/src/lib.rs",
        "ScenarioError",
        "NotATable",
    ): {
        "classification": "required_unreached",
        "decision": "construct_on_non_table_scenario_input",
        "rationale": (
            "The public error and its documentation promise a typed top-level-table refusal, "
            "but TOML/RON parsing currently maps that condition into the generic Parse variant. "
            "The branch should be wired, not deleted."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/frankentui_shell.rs",
        "ShellRoute",
        "WorldCanvas",
    ): {
        "classification": "required_unreached",
        "decision": "construct_from_live_navigation_in_bd_2z0_6_8",
        "rationale": (
            "Only shell-model tests construct this route today. bd-2z0.6.8 owns connecting the "
            "standalone shell to HostClient, where live navigation must select the world canvas."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/frankentui_shell.rs",
        "ShellRoute",
        "Inspector",
    ): {
        "classification": "required_unreached",
        "decision": "construct_from_live_navigation_in_bd_2z0_6_8",
        "rationale": (
            "Only shell-model tests construct this route today. bd-2z0.6.8 owns connecting the "
            "standalone shell to HostClient, where live navigation must select the inspector."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/frankentui_shell.rs",
        "ShellRoute",
        "HelpOverlay",
    ): {
        "classification": "dead_removable",
        "decision": "remove_redundant_route_variant",
        "rationale": (
            "The shell models help with ToggleHelp plus help_visible; navigation never constructs "
            "or consumes a HelpOverlay route. Serde reachability does not justify duplicate state."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/frankentui_shell.rs",
        "ShellMessage",
        "Navigate",
    ): {
        "classification": "required_unreached",
        "decision": "construct_from_live_shell_actions_in_bd_2z0_6_8",
        "rationale": (
            "Only shell-model tests construct this handled message. bd-2z0.6.8 owns routing "
            "real terminal navigation through the standalone HostClient-backed shell."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/frankentui_shell.rs",
        "ShellMessage",
        "ToggleHelp",
    ): {
        "classification": "required_unreached",
        "decision": "construct_from_live_shell_actions_in_bd_2z0_6_8",
        "rationale": (
            "Only shell-model tests construct this handled message. bd-2z0.6.8 owns routing "
            "real terminal help actions through the standalone HostClient-backed shell."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/frankentui_shell.rs",
        "ShellMessage",
        "SubmitCommand",
    ): {
        "classification": "required_unreached",
        "decision": "construct_from_live_host_actions_in_bd_2z0_6_8",
        "rationale": (
            "The standalone shell handles the message correctly, but no live terminal producer "
            "constructs it. bd-2z0.6.8 owns integrating this existing shell with HostClient."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/frankentui_shell.rs",
        "ShellMessage",
        "CommandReceipt",
    ): {
        "classification": "required_unreached",
        "decision": "construct_from_live_host_receipts_in_bd_2z0_6_8",
        "rationale": (
            "Only shell-model tests construct this handled receipt. bd-2z0.6.8 owns returning "
            "real HostClient command receipts through the standalone shell."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/frankentui_shell.rs",
        "ShellMessage",
        "UpdateSnapshot",
    ): {
        "classification": "required_unreached",
        "decision": "construct_from_live_host_snapshots_in_bd_2z0_6_8",
        "rationale": (
            "Only shell-model tests construct this handled update. bd-2z0.6.8 owns feeding live "
            "HostClient snapshots into the standalone shell."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/frankentui_shell.rs",
        "ShellMessage",
        "SetStatus",
    ): {
        "classification": "required_unreached",
        "decision": "construct_from_live_shell_diagnostics_in_bd_2z0_6_8",
        "rationale": (
            "The standalone shell handles status messages, but the live terminal never produces "
            "one through this model. bd-2z0.6.8 owns the real shell integration."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/paint.rs",
        "SubCellMode",
        "Braille",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_owner_authorized_bd_c1z8_resolution",
        "rationale": (
            "Only tests construct this value in the dead duplicate paint.rs engine confirmed by "
            "bd-c1z8. Remove it with that module after the owner authorizes file deletion."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/paint.rs",
        "SubCellMode",
        "HalfBlock",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_owner_authorized_bd_c1z8_resolution",
        "rationale": (
            "Only tests construct this value in the dead duplicate paint.rs engine confirmed by "
            "bd-c1z8. Remove it with that module after the owner authorizes file deletion."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/paint.rs",
        "SubCellMode",
        "Quadrant",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_owner_authorized_bd_c1z8_resolution",
        "rationale": (
            "Only tests construct this value in the dead duplicate paint.rs engine confirmed by "
            "bd-c1z8. Remove it with that module after the owner authorizes file deletion."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/paint.rs",
        "ColorDepth",
        "TrueColor",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_owner_authorized_bd_c1z8_resolution",
        "rationale": (
            "Only tests construct this value in the dead duplicate paint.rs engine confirmed by "
            "bd-c1z8. Remove it with that module after the owner authorizes file deletion."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/paint.rs",
        "ColorDepth",
        "Palette256",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_owner_authorized_bd_c1z8_resolution",
        "rationale": (
            "Only tests construct this value in the dead duplicate paint.rs engine confirmed by "
            "bd-c1z8. Remove it with that module after the owner authorizes file deletion."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/paint.rs",
        "ColorDepth",
        "Basic16",
    ): {
        "classification": "dead_removable",
        "decision": "retain_until_owner_authorizes_bd_c1z8_resolution",
        "rationale": (
            "The variant belongs to the dead duplicate paint.rs engine confirmed by bd-c1z8. "
            "It is removable with that engine, but AGENTS.md forbids file deletion without "
            "the owner's express written permission."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/paint.rs",
        "DitherMode",
        "None",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_owner_authorized_bd_c1z8_resolution",
        "rationale": (
            "Only tests construct this value in the dead duplicate paint.rs engine confirmed by "
            "bd-c1z8. Remove it with that module after the owner authorizes file deletion."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/paint.rs",
        "DitherMode",
        "Ordered2x2",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_owner_authorized_bd_c1z8_resolution",
        "rationale": (
            "Only tests construct this value in the dead duplicate paint.rs engine confirmed by "
            "bd-c1z8. Remove it with that module after the owner authorizes file deletion."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/subcell.rs",
        "SubCellMode",
        "HalfBlock",
    ): {
        "classification": "required_unreached",
        "decision": "construct_from_capability_fallback_selection",
        "rationale": (
            "subcell.rs is the integrated live canvas, and bd-2z0.14.2.1 retained half-block as "
            "the 1x2 fallback. Capability detection currently selects only Ascii, Quadrant, or "
            "Braille, so the promised fallback still needs a real producer."
        ),
    },
    (
        "crates/scriptbots-app/src/terminal/subcell.rs",
        "Layer",
        "Water",
    ): {
        "classification": "required_unreached",
        "decision": "construct_from_hydrology_depth_ramp_in_bd_sk55",
        "rationale": (
            "subcell.rs is the integrated live canvas. Open bd-sk55 owns carrying normalized "
            "hydrology depth into it; that path must produce Water rather than folding depth into "
            "the existing Terrain layer."
        ),
    },
    (
        "crates/scriptbots-core/src/attribution.rs",
        "EffectiveOutput",
        "Continuous",
    ): {
        "classification": "dead_removable",
        "decision": "remove_variant_and_stale_renderer_arms",
        "rationale": (
            "effective_for produces Thresholded for Boost and Clamped for every other output. "
            "Continuous has only renderer/TUI handler arms, so the documented direct-use state "
            "cannot occur."
        ),
    },
    (
        "crates/scriptbots-core/src/interventions.rs",
        "ToroidalRegion",
        "All",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_superseded_interventions_module",
        "rationale": (
            "Nothing outside interventions.rs uses this parallel API. The live Region and "
            "Intervention pipeline is defined, queued, and applied in core/lib.rs; Serde never "
            "deserializes this superseded module."
        ),
    },
    (
        "crates/scriptbots-core/src/interventions.rs",
        "ToroidalRegion",
        "Rect",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_superseded_interventions_module",
        "rationale": (
            "Nothing outside interventions.rs uses this parallel API. The live Region and "
            "Intervention pipeline is defined, queued, and applied in core/lib.rs; Serde never "
            "deserializes this superseded module."
        ),
    },
    (
        "crates/scriptbots-core/src/interventions.rs",
        "ToroidalRegion",
        "Disc",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_superseded_interventions_module",
        "rationale": (
            "Only tests construct this value, and nothing outside interventions.rs uses the "
            "parallel API. The live Region and Intervention pipeline in core/lib.rs supersedes it."
        ),
    },
    (
        "crates/scriptbots-core/src/interventions.rs",
        "InterventionAction",
        "Drought",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_superseded_interventions_module",
        "rationale": (
            "Nothing outside interventions.rs uses this parallel API. The live Region and "
            "Intervention pipeline is defined, queued, and applied in core/lib.rs; Serde never "
            "deserializes this superseded module."
        ),
    },
    (
        "crates/scriptbots-core/src/interventions.rs",
        "InterventionAction",
        "PredatorInjection",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_superseded_interventions_module",
        "rationale": (
            "Nothing outside interventions.rs uses this parallel API. The live Region and "
            "Intervention pipeline is defined, queued, and applied in core/lib.rs; Serde never "
            "deserializes this superseded module."
        ),
    },
    (
        "crates/scriptbots-core/src/interventions.rs",
        "InterventionAction",
        "Meteor",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_superseded_interventions_module",
        "rationale": (
            "Only tests construct this value, and nothing outside interventions.rs uses the "
            "parallel API. The live Region and Intervention pipeline in core/lib.rs supersedes it."
        ),
    },
    (
        "crates/scriptbots-core/src/interventions.rs",
        "InterventionAction",
        "TerrainPaint",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_superseded_interventions_module",
        "rationale": (
            "Nothing outside interventions.rs uses this parallel API. The live Region and "
            "Intervention pipeline is defined, queued, and applied in core/lib.rs; Serde never "
            "deserializes this superseded module."
        ),
    },
    (
        "crates/scriptbots-core/src/interventions.rs",
        "InterventionAction",
        "FoodEmbargo",
    ): {
        "classification": "dead_removable",
        "decision": "remove_with_superseded_interventions_module",
        "rationale": (
            "Nothing outside interventions.rs uses this parallel API. The live Region and "
            "Intervention pipeline is defined, queued, and applied in core/lib.rs; Serde never "
            "deserializes this superseded module."
        ),
    },
    (
        "crates/scriptbots-core/src/lib.rs",
        "RenderAntiAliasing",
        "Smaa",
    ): {
        "classification": "required_unreached",
        "decision": "construct_when_bd_2z0_14_1_6_materializes_smaa",
        "rationale": (
            "No renderer or CLI path selects SMAA today, so Serde acceptance alone is not proof. "
            "Open bd-2z0.14.1.6 explicitly owns SMAA as the documented alternative in the real "
            "post-processing stack; that implementation must become its producer."
        ),
    },
    (
        "crates/scriptbots-core/src/lib.rs",
        "TuiThemeId",
        "CyberpunkAurora",
    ): {
        "classification": "required_unreached",
        "decision": "wire_render_settings_theme_to_live_curated_theme",
        "rationale": (
            "RenderSettings documents this as the terminal chrome theme, but the field is never "
            "read and terminal mode uses the parallel CuratedThemeId. Map the scenario/config "
            "value into that live selector in open bd-2z0.14.2.2."
        ),
    },
    (
        "crates/scriptbots-core/src/lib.rs",
        "TuiThemeId",
        "Darcula",
    ): {
        "classification": "required_unreached",
        "decision": "wire_render_settings_theme_to_live_curated_theme",
        "rationale": (
            "RenderSettings documents this as the terminal chrome theme, but the field is never "
            "read and terminal mode uses the parallel CuratedThemeId. Map the scenario/config "
            "value into that live selector in open bd-2z0.14.2.2."
        ),
    },
    (
        "crates/scriptbots-core/src/lib.rs",
        "TuiThemeId",
        "LumenLight",
    ): {
        "classification": "required_unreached",
        "decision": "wire_render_settings_theme_to_live_curated_theme",
        "rationale": (
            "RenderSettings documents this as the terminal chrome theme, but the field is never "
            "read and terminal mode uses the parallel CuratedThemeId. Map the scenario/config "
            "value into that live selector in open bd-2z0.14.2.2."
        ),
    },
    (
        "crates/scriptbots-core/src/lib.rs",
        "TuiThemeId",
        "NordicFrost",
    ): {
        "classification": "required_unreached",
        "decision": "wire_render_settings_theme_to_live_curated_theme",
        "rationale": (
            "Only a Serde round-trip test constructs this value. RenderSettings documents it as "
            "the terminal chrome theme, but the field is never read and terminal mode uses the "
            "parallel CuratedThemeId; open bd-2z0.14.2.2 owns wiring that mapping."
        ),
    },
    (
        "crates/scriptbots-core/src/lib.rs",
        "TuiThemeId",
        "HighContrast",
    ): {
        "classification": "required_unreached",
        "decision": "wire_render_settings_theme_to_live_curated_theme",
        "rationale": (
            "RenderSettings documents this as the terminal chrome theme, but the field is never "
            "read and terminal mode uses the parallel CuratedThemeId. Map the scenario/config "
            "value into that live selector in open bd-2z0.14.2.2."
        ),
    },
    (
        "crates/scriptbots-core/src/lib.rs",
        "WorldContinuationBlocker",
        "PersistenceFault",
    ): {
        "classification": "dead_removable",
        "decision": "remove_stale_blocker_and_wire_encoding",
        "rationale": (
            "Commit e8c9d3285 externalized persistence admission and removed the last producer. "
            "The world now reports RetainedPersistenceBatch for its persistence boundary; only "
            "Display, Serde, and the trace encoder preserve this obsolete state."
        ),
    },
    (
        "crates/scriptbots-core/src/map_elites.rs",
        "SelectionMode",
        "Fitness",
    ): {
        "classification": "dead_removable",
        "decision": "remove_unused_map_elites_selection_enum",
        "rationale": (
            "The MAP-Elites archive always replaces by fitness and never stores or reads this "
            "selection enum. It is unrelated to the live selection-control enum in core/lib.rs."
        ),
    },
    (
        "crates/scriptbots-core/src/genome_diff.rs",
        "Locus",
        "Hyper",
    ): {
        "classification": "dead_removable",
        "decision": "remove_reserved_locus_until_a_family_emits_it",
        "rationale": (
            "No BrainFamilyCodec emits a hyperparameter locus: MLP/DWRAON emit node loci and "
            "Assembly emits cells. bd-16g.13.1 closed without the promised producer or unit case, "
            "so this reserved future placeholder is removable under the no-shims policy."
        ),
    },
    (
        "crates/scriptbots-core/src/map_elites.rs",
        "SelectionMode",
        "Novelty",
    ): {
        "classification": "dead_removable",
        "decision": "remove_unused_map_elites_selection_enum",
        "rationale": (
            "The MAP-Elites archive always replaces by fitness and never stores or reads this "
            "selection enum. It is unrelated to the live selection-control enum in core/lib.rs."
        ),
    },
    (
        "crates/scriptbots-core/src/map_elites.rs",
        "SelectionMode",
        "CuriosityHybrid",
    ): {
        "classification": "dead_removable",
        "decision": "remove_unused_map_elites_selection_enum",
        "rationale": (
            "The MAP-Elites archive always replaces by fitness and never stores or reads this "
            "selection enum. It is unrelated to the live selection-control enum in core/lib.rs."
        ),
    },
    (
        "crates/scriptbots-core/src/phylo.rs",
        "PhyloKey",
        "Agent",
    ): {
        "classification": "required_unreached",
        "decision": "construct_from_on_demand_agent_subtree_materialization",
        "rationale": (
            "bd-16g.3.4 promises agent-scale subtrees materialized from AgentUid on demand, but "
            "HEAD only creates Species keys and expand merely uncollapses existing nodes. Keep "
            "this key and construct it when that bounded materialization path lands."
        ),
    },
    (
        "crates/scriptbots-runtime/src/migrator.rs",
        "MigrationError",
        "InvalidEdge",
    ): {
        "classification": "required_unreached",
        "decision": "construct_when_custom_topology_contains_an_invalid_edge",
        "rationale": (
            "Custom topology currently drops invalid edges silently even though this typed error "
            "promises rejection. Reopened bd-16g.5.2 explicitly requires self/out-of-range edge "
            "validation, so that production barrier must construct the error rather than delete it."
        ),
    },
    (
        "crates/scriptbots-storage/src/journal.rs",
        "DomainEventExpectation",
        "RequireNonEmpty",
    ): {
        "classification": "required_test_surface",
        "decision": "keep_as_explicit_conformance_evidence_contract",
        "rationale": (
            "Production code consumes the value and two integration tests construct it to demand "
            "non-empty durable evidence. It is an intentional test/evidence API, not dead state."
        ),
    },
}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative: str
    raw: str
    code: str
    test_code: str = ""


@dataclass(frozen=True)
class Variant:
    path: str
    line: int
    enum: str
    variant: str
    enum_derives_deserialize: bool
    has_from: bool
    has_default: bool


@dataclass(frozen=True)
class Verdict:
    path: str
    line: int
    enum: str
    variant: str
    construction_references: int
    qualified_references: int
    self_references: int
    import_references: int
    ambiguous_references: int
    test_construction_references: int
    test_references: int
    doc_mentions: int
    enum_derives_deserialize: bool
    has_from: bool
    has_default: bool
    production_candidate: bool
    raw_candidate: bool
    refined_candidate: bool
    excluded_by: tuple[str, ...]


def strip_rust_comments_and_literals(source: str) -> str:
    """Replace comments and string/character contents while preserving offsets."""

    output = list(source)
    index = 0
    length = len(source)

    def blank(start: int, end: int) -> None:
        for position in range(start, end):
            if output[position] != "\n":
                output[position] = " "

    while index < length:
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            end = length if end == -1 else end
            blank(index, end)
            index = end
            continue
        if source.startswith("/*", index):
            start = index
            index += 2
            depth = 1
            while index < length and depth:
                if source.startswith("/*", index):
                    depth += 1
                    index += 2
                elif source.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            blank(start, index)
            continue

        raw_match = RAW_STRING_RE.match(source, index)
        if raw_match:
            start = index
            hashes = raw_match.group("hashes")
            index = raw_match.end()
            terminator = '"' + hashes
            end = source.find(terminator, index)
            index = length if end == -1 else end + len(terminator)
            blank(start, index)
            continue

        string_prefix = (
            1 if source[index] == '"' else 2 if source.startswith('b"', index) else 0
        )
        if string_prefix:
            start = index
            index += string_prefix
            escaped = False
            while index < length:
                character = source[index]
                index += 1
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    break
            blank(start, index)
            continue

        if source[index] == "'":
            lifetime = LIFETIME_RE.match(source, index)
            if lifetime and not source.startswith("'", lifetime.end()):
                index = lifetime.end()
                continue
            start = index
            index += 1
            escaped = False
            while index < length:
                character = source[index]
                index += 1
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == "'":
                    break
            blank(start, index)
            continue
        index += 1

    return "".join(output)


CFG_TEST_RE = re.compile(
    r"#\s*\[\s*cfg\s*\((?P<expression>[^\]]*\btest\b[^\]]*)\)\s*\]"
)


def split_cfg_arguments(expression: str) -> list[str]:
    """Split one cfg predicate's arguments at top-level commas."""

    arguments: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(expression):
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            arguments.append(expression[start:index].strip())
            start = index + 1
    arguments.append(expression[start:].strip())
    return [argument for argument in arguments if argument]


def cfg_requires_test(expression: str) -> bool:
    """Return true only when the cfg predicate logically requires `test`."""

    expression = expression.strip()
    if expression == "test":
        return True
    predicate = re.fullmatch(
        rf"(?P<operator>{IDENTIFIER})\s*\((?P<arguments>.*)\)",
        expression,
        re.DOTALL,
    )
    if predicate is None:
        return False
    arguments = split_cfg_arguments(predicate.group("arguments"))
    operator = predicate.group("operator")
    if operator == "all":
        return any(cfg_requires_test(argument) for argument in arguments)
    if operator == "any":
        return bool(arguments) and all(
            cfg_requires_test(argument) for argument in arguments
        )
    # `not(test)` and unknown predicates can both be enabled outside tests.
    return False


def cfg_item_end(source: str, cursor: int) -> int | None:
    """Bound one cfg-gated item, field, variant, arm, or statement."""

    item_prefix = re.compile(
        r"(?:pub(?:\s*\([^)]*\))?\s+)?"
        r"(?:(?:async|unsafe|const|default)\s+)*"
        r"(?:(?:fn|mod|impl|trait|enum|struct|union|extern)\b|macro_rules\s*!)"
    )
    starts_block_item = item_prefix.match(source, cursor) is not None
    control = re.match(
        r"(?:'[A-Za-z_][A-Za-z0-9_]*\s*:\s*)?(if|while|for|loop|match)\b",
        source[cursor:],
    )
    starts_control_block = control is not None
    parenthesis_depth = 0
    bracket_depth = 0
    angle_depth = 0
    saw_match_arrow = False
    index = cursor
    while index < len(source):
        character = source[index]
        if character == "(":
            parenthesis_depth += 1
        elif character == ")":
            parenthesis_depth = max(0, parenthesis_depth - 1)
        elif character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif (
            character == "<"
            and not source.startswith("<<", index)
            and (index == 0 or source[index - 1] != "<")
        ):
            angle_depth += 1
        elif character == ">" and angle_depth:
            angle_depth -= 1
        elif parenthesis_depth == 0 and bracket_depth == 0 and angle_depth == 0:
            if source.startswith("=>", index):
                saw_match_arrow = True
                index += 2
                continue
            if character in ",;":
                return index + 1
            if character == "{":
                closing = matching_brace(source, index)
                if starts_block_item or saw_match_arrow or starts_control_block:
                    end = closing + 1
                    if control is not None and control.group(1) == "if":
                        while True:
                            else_start = end
                            while (
                                else_start < len(source)
                                and source[else_start].isspace()
                            ):
                                else_start += 1
                            if not re.match(r"else\b", source[else_start:]):
                                break
                            next_opening = source.find("{", else_start + 4)
                            if next_opening == -1:
                                break
                            end = matching_brace(source, next_opening) + 1
                    while end < len(source) and source[end].isspace():
                        end += 1
                    if end < len(source) and source[end] in ",;":
                        end += 1
                    return end
                index = closing
        index += 1
    return None


def split_inline_test_code(source: str) -> tuple[str, str]:
    """Return production/test-only views while preserving source offsets."""

    production = list(source)
    tests = ["\n" if character == "\n" else " " for character in source]

    def blank_production(start: int, end: int) -> None:
        for position in range(start, end):
            if production[position] != "\n":
                production[position] = "~"
            tests[position] = source[position]

    for cfg_match in CFG_TEST_RE.finditer(source):
        expression = cfg_match.group("expression")
        if not cfg_requires_test(expression):
            continue
        cursor = cfg_match.end()
        while True:
            while cursor < len(source) and source[cursor].isspace():
                cursor += 1
            if not source.startswith("#[", cursor):
                break
            closing_attribute = source.find("]", cursor + 2)
            if closing_attribute == -1:
                break
            cursor = closing_attribute + 1
        end = cfg_item_end(source, cursor)
        if end is None:
            continue
        blank_production(cfg_match.start(), end)
    return "".join(production), "".join(tests)


def matching_brace(source: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unclosed enum body at byte {opening}")


def top_level_segments(body: str, body_offset: int) -> Iterable[tuple[str, int]]:
    start = 0
    round_depth = 0
    square_depth = 0
    brace_depth = 0
    for index, character in enumerate(body):
        if character == "(":
            round_depth += 1
        elif character == ")":
            round_depth = max(0, round_depth - 1)
        elif character == "[":
            square_depth += 1
        elif character == "]":
            square_depth = max(0, square_depth - 1)
        elif character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth = max(0, brace_depth - 1)
        elif (
            character == ","
            and round_depth == 0
            and square_depth == 0
            and brace_depth == 0
        ):
            yield body[start:index], body_offset + start
            start = index + 1
    if body[start:].strip():
        yield body[start:], body_offset + start


def is_test_source(relative: str) -> bool:
    return "/tests/" in relative or "/benches/" in relative


def discover_source(
    root: Path,
    revision: str | None = None,
    *,
    include_tests: bool = False,
) -> list[SourceFile]:
    files = []
    if revision is None:
        listing = subprocess.run(
            ["git", "ls-files", "*.rs"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        source_rows = [
            (relative, (root / relative).read_text(encoding="utf-8"))
            for relative in sorted(listing)
            if relative.startswith("crates/")
            and (include_tests or not is_test_source(relative))
        ]
    else:
        listing = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", revision, "--", "crates"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        source_rows = []
        for relative in sorted(
            path
            for path in listing
            if path.endswith(".rs") and (include_tests or not is_test_source(path))
        ):
            raw = subprocess.run(
                ["git", "show", f"{revision}:{relative}"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            source_rows.append((relative, raw))
    for relative, raw in source_rows:
        path = root / relative
        stripped = strip_rust_comments_and_literals(raw)
        if is_test_source(relative):
            production_code = stripped
            inline_test_code = ""
        else:
            production_code, inline_test_code = split_inline_test_code(stripped)
        files.append(
            SourceFile(
                path=path,
                relative=relative,
                raw=raw,
                code=production_code,
                test_code=inline_test_code,
            )
        )
    return files


def discover_variants(files: Sequence[SourceFile]) -> list[Variant]:
    variants = []
    for source_file in files:
        for enum_match in PUBLIC_ENUM_RE.finditer(source_file.code):
            opening = enum_match.end() - 1
            closing = matching_brace(source_file.code, opening)
            body_offset = opening + 1
            body = source_file.code[body_offset:closing]
            enum_attrs = enum_match.group("attrs")
            derives_deserialize = bool(
                re.search(
                    r"#\s*\[\s*derive\s*\([^\]]*\bDeserialize\b",
                    enum_attrs,
                    re.DOTALL,
                )
            )
            for segment, segment_offset in top_level_segments(body, body_offset):
                variant_match = VARIANT_RE.match(segment)
                if not variant_match:
                    continue
                variant_offset = segment_offset + variant_match.start("name")
                variants.append(
                    Variant(
                        path=source_file.relative,
                        line=source_file.code.count("\n", 0, variant_offset) + 1,
                        enum=enum_match.group("name"),
                        variant=variant_match.group("name"),
                        enum_derives_deserialize=derives_deserialize,
                        has_from=bool(re.search(r"#\s*\[\s*from\s*\]", segment)),
                        has_default=bool(re.search(r"#\s*\[\s*default\s*\]", segment)),
                    )
                )
    return variants


def find_match_body_opening(source: str, cursor: int) -> int | None:
    """Find the top-level body brace after a `match` token."""

    stack: list[str] = []
    closing_for = {"(": ")", "[": "]"}
    while cursor < len(source):
        character = source[cursor]
        if character in closing_for:
            stack.append(closing_for[character])
        elif stack and character == stack[-1]:
            stack.pop()
        elif not stack:
            if character == "{":
                return cursor
            if character == ";":
                return None
        cursor += 1
    return None


def find_top_level_fat_arrow(
    source: str,
    start: int,
    end: int,
) -> int | None:
    """Find a match-arm arrow while ignoring nested pattern delimiters."""

    stack: list[str] = []
    closing_for = {"(": ")", "[": "]", "{": "}"}
    cursor = start
    while cursor < end:
        if not stack and source.startswith("=>", cursor):
            return cursor
        character = source[cursor]
        if character in closing_for:
            stack.append(closing_for[character])
        elif stack and character == stack[-1]:
            stack.pop()
        cursor += 1
    return None


def find_top_level_guard(
    source: str,
    start: int,
    end: int,
) -> int | None:
    """Find the optional top-level `if` guard in one match-arm pattern."""

    stack: list[str] = []
    closing_for = {"(": ")", "[": "]", "{": "}"}
    cursor = start
    while cursor < end:
        character = source[cursor]
        if character in closing_for:
            stack.append(closing_for[character])
        elif stack and character == stack[-1]:
            stack.pop()
        elif (
            not stack
            and source.startswith("if", cursor)
            and (cursor == 0 or not re.match(r"[A-Za-z0-9_]", source[cursor - 1]))
            and (
                cursor + 2 >= len(source)
                or not re.match(r"[A-Za-z0-9_]", source[cursor + 2])
            )
        ):
            return cursor
        cursor += 1
    return None


def match_arm_body_end(source: str, start: int, limit: int) -> int:
    """Return the start of the next arm after one arm expression."""

    cursor = start
    while cursor < limit and source[cursor].isspace():
        cursor += 1
    if cursor >= limit:
        return limit

    block_like = re.match(
        r"(?:if|match|loop|while|for|unsafe|async|const)\b",
        source[cursor:],
    )
    if source[cursor] == "{":
        cursor = matching_brace(source, cursor) + 1
    elif block_like is not None:
        bounded = cfg_item_end(source, cursor)
        if bounded is not None and bounded <= limit:
            cursor = bounded
    else:
        stack: list[str] = []
        closing_for = {"(": ")", "[": "]", "{": "}"}
        while cursor < limit:
            character = source[cursor]
            if character in closing_for:
                stack.append(closing_for[character])
            elif stack and character == stack[-1]:
                stack.pop()
            elif not stack and character == ",":
                return cursor + 1
            cursor += 1
        return limit

    while cursor < limit and source[cursor].isspace():
        cursor += 1
    if cursor < limit and source[cursor] == ",":
        cursor += 1
    return cursor


def rust_pattern_spans(source: str) -> tuple[tuple[int, int], ...]:
    """Locate match-arm and let-binding patterns in stripped Rust source."""

    spans: list[tuple[int, int]] = []
    for match_token in re.finditer(r"\bmatch\b", source):
        opening = find_match_body_opening(source, match_token.end())
        if opening is None:
            continue
        closing = matching_brace(source, opening)
        cursor = opening + 1
        while cursor < closing:
            while cursor < closing and (
                source[cursor].isspace() or source[cursor] == ","
            ):
                cursor += 1
            if cursor >= closing:
                break
            arrow = find_top_level_fat_arrow(source, cursor, closing)
            if arrow is None:
                break
            guard = find_top_level_guard(source, cursor, arrow)
            spans.append((cursor, arrow if guard is None else guard))
            next_cursor = match_arm_body_end(source, arrow + 2, closing)
            if next_cursor <= cursor:
                break
            cursor = next_cursor

    for let_token in re.finditer(r"\blet\b", source):
        stack: list[str] = []
        closing_for = {"(": ")", "[": "]", "{": "}"}
        cursor = let_token.end()
        while cursor < len(source):
            character = source[cursor]
            if character in closing_for:
                stack.append(closing_for[character])
            elif stack and character == stack[-1]:
                stack.pop()
            elif not stack and character == ";":
                break
            elif (
                not stack
                and character == "="
                and not source.startswith(("==", "=>"), cursor)
                and (cursor == 0 or source[cursor - 1] not in "!<>=")
            ):
                spans.append((let_token.end(), cursor))
                break
            cursor += 1
    return tuple(spans)


def inside_matches_macro(source: str, start: int) -> bool:
    search_start = max(0, start - 2_048)
    macro = source.rfind("matches!", search_start, start)
    if macro == -1:
        return False
    opening = source.find("(", macro + len("matches!"), start)
    if opening == -1:
        return False
    depth = 0
    saw_separator = False
    for character in source[opening:start]:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return False
        elif character == "," and depth == 1:
            saw_separator = True
    return depth > 0 and saw_separator


def is_construction_reference(
    source: str,
    start: int,
    end: int,
    pattern_spans: Sequence[tuple[int, int]] = (),
) -> bool:
    """Distinguish value production from match/let consumption conservatively."""

    if any(
        pattern_start <= start < pattern_end
        for pattern_start, pattern_end in pattern_spans
    ):
        return False
    prefix = source[max(0, start - 2_048) : start]
    suffix = source[end:]
    stripped_prefix = prefix.rstrip()
    stripped_suffix = suffix.lstrip()
    if stripped_prefix.endswith(("==", "!=")) or stripped_suffix.startswith(
        ("==", "!=")
    ):
        return False
    if inside_matches_macro(source, start):
        return False
    return True


def reference_counts(
    variants: Sequence[Variant],
    files: Sequence[SourceFile],
) -> tuple[
    Counter[VariantIdentity],
    Counter[VariantIdentity],
    Counter[VariantIdentity],
    Counter[VariantIdentity],
    Counter[VariantIdentity],
    Counter[VariantIdentity],
]:
    identities_by_pair: dict[VariantPair, list[VariantIdentity]] = defaultdict(list)
    variants_by_enum: dict[str, set[str]] = defaultdict(set)
    definition_paths_by_enum: dict[str, set[str]] = defaultdict(set)
    for item in variants:
        identity = (item.path, item.enum, item.variant)
        identities_by_pair[(item.enum, item.variant)].append(identity)
        variants_by_enum[item.enum].add(item.variant)
        definition_paths_by_enum[item.enum].add(item.path)
    enum_alternation = "|".join(
        re.escape(name)
        for name in sorted(variants_by_enum, key=lambda name: (-len(name), name))
    )
    qualified = re.compile(
        rf"\b(?P<enum>{enum_alternation})\s*::\s*(?P<variant>{IDENTIFIER})\b"
    )
    grouped_import = re.compile(
        rf"\b(?P<enum>{enum_alternation})\s*::\s*\{{(?P<body>[^}}]*)\}}",
        re.DOTALL,
    )
    wildcard_import = re.compile(rf"\b(?P<enum>{enum_alternation})\s*::\s*\*")
    impl_token = re.compile(r"\bimpl\b")
    impl_owner = re.compile(rf"\b(?P<enum>{enum_alternation})\b")
    self_reference = re.compile(rf"\bSelf\s*::\s*(?P<variant>{IDENTIFIER})\b")
    enum_group_import = re.compile(
        rf"\buse\s+(?P<module>(?:{IDENTIFIER}\s*::\s*)+)"
        rf"\{{(?P<body>[^}}]*)\}}\s*;",
        re.DOTALL,
    )
    enum_direct_import = re.compile(
        rf"\buse\s+(?P<module>(?:{IDENTIFIER}\s*::\s*)+)"
        rf"(?P<enum>{enum_alternation})(?:\s+as\s+{IDENTIFIER})?\s*;"
    )
    qualified_code: Counter[VariantIdentity] = Counter()
    qualified_raw: Counter[VariantIdentity] = Counter()
    qualified_all_code: Counter[VariantIdentity] = Counter()
    constructions: Counter[VariantIdentity] = Counter()
    self_code: Counter[VariantIdentity] = Counter()
    import_code: Counter[VariantIdentity] = Counter()
    ambiguous_code: Counter[VariantIdentity] = Counter()
    import_origins: dict[str, dict[str, str]] = defaultdict(dict)

    def crate_prefix(path: str) -> str:
        marker = "/src/"
        return path.split(marker, maxsplit=1)[0] if marker in path else ""

    def record_import_origin(
        source_file: SourceFile,
        module: str,
        enum: str,
    ) -> None:
        module_parts = re.findall(IDENTIFIER, module)
        if not module_parts:
            return
        module_leaf = module_parts[-1]
        source_crate = crate_prefix(source_file.relative)
        matching_paths = [
            path
            for path in definition_paths_by_enum[enum]
            if crate_prefix(path) == source_crate
            and (
                path.endswith(f"/{module_leaf}.rs")
                or path.endswith(f"/{module_leaf}/mod.rs")
            )
        ]
        if len(matching_paths) == 1:
            import_origins[source_file.relative][enum] = matching_paths[0]

    for source_file in files:
        for match in enum_group_import.finditer(source_file.code):
            imported_names = set(re.findall(rf"\b{IDENTIFIER}\b", match.group("body")))
            for enum in variants_by_enum.keys() & imported_names:
                record_import_origin(source_file, match.group("module"), enum)
        for match in enum_direct_import.finditer(source_file.code):
            record_import_origin(
                source_file,
                match.group("module"),
                match.group("enum"),
            )

    def resolve(
        pair: VariantPair,
        source_path: str,
    ) -> tuple[VariantIdentity | None, tuple[VariantIdentity, ...]]:
        candidates = identities_by_pair.get(pair, [])
        local = [identity for identity in candidates if identity[0] == source_path]
        if len(local) == 1:
            return local[0], ()
        imported_path = import_origins[source_path].get(pair[0])
        imported = [identity for identity in candidates if identity[0] == imported_path]
        if len(imported) == 1:
            return imported[0], ()
        if len(candidates) == 1:
            return candidates[0], ()
        return None, tuple(candidates)

    def collect_qualified(
        source_file: SourceFile,
        source: str,
        destination: Counter[VariantIdentity],
        construction_destination: Counter[VariantIdentity] | None = None,
        ambiguous_destination: Counter[VariantIdentity] | None = None,
    ) -> None:
        pattern_spans = (
            rust_pattern_spans(source) if construction_destination is not None else ()
        )
        for match in qualified.finditer(source):
            pair = (match.group("enum"), match.group("variant"))
            identity, ambiguous = resolve(pair, source_file.relative)
            if identity is not None:
                destination[identity] += 1
                if construction_destination is not None:
                    if is_construction_reference(
                        source,
                        match.start(),
                        match.end(),
                        pattern_spans,
                    ):
                        construction_destination[identity] += 1
            elif ambiguous_destination is not None:
                for candidate in ambiguous:
                    ambiguous_destination[candidate] += 1

    for source_file in files:
        collect_qualified(
            source_file,
            source_file.code,
            qualified_code,
            constructions,
            ambiguous_code,
        )
        collect_qualified(source_file, source_file.raw, qualified_raw)
        collect_qualified(
            source_file,
            strip_rust_comments_and_literals(source_file.raw),
            qualified_all_code,
        )

        for match in grouped_import.finditer(source_file.code):
            enum = match.group("enum")
            imported = set(re.findall(rf"\b{IDENTIFIER}\b", match.group("body")))
            for variant in variants_by_enum[enum].intersection(imported):
                identity, ambiguous = resolve(
                    (enum, variant),
                    source_file.relative,
                )
                if identity is not None:
                    import_code[identity] += 1
                else:
                    for candidate in ambiguous:
                        ambiguous_code[candidate] += 1
        for match in wildcard_import.finditer(source_file.code):
            enum = match.group("enum")
            for variant in variants_by_enum[enum]:
                identity, ambiguous = resolve(
                    (enum, variant),
                    source_file.relative,
                )
                if identity is not None:
                    import_code[identity] += 1
                else:
                    for candidate in ambiguous:
                        ambiguous_code[candidate] += 1

        for impl_match in impl_token.finditer(source_file.code):
            opening = source_file.code.find("{", impl_match.end())
            if opening == -1:
                continue
            header = source_file.code[impl_match.end() : opening]
            if ";" in header:
                continue
            header = re.split(r"\bwhere\b", header, maxsplit=1)[0]
            trait_separator = list(re.finditer(r"\bfor\b", header))
            if trait_separator:
                header = header[trait_separator[-1].end() :]
            owners = list(impl_owner.finditer(header))
            if not owners:
                continue
            enum = owners[-1].group("enum")
            closing = matching_brace(source_file.code, opening)
            body = source_file.code[opening + 1 : closing]
            body_pattern_spans = rust_pattern_spans(body)
            for match in self_reference.finditer(body):
                variant = match.group("variant")
                if variant not in variants_by_enum[enum]:
                    continue
                identity, ambiguous = resolve(
                    (enum, variant),
                    source_file.relative,
                )
                if identity is not None:
                    self_code[identity] += 1
                    if is_construction_reference(
                        body,
                        match.start(),
                        match.end(),
                        body_pattern_spans,
                    ):
                        constructions[identity] += 1
                else:
                    for candidate in ambiguous:
                        ambiguous_code[candidate] += 1
    documentation_mentions = Counter(
        {
            identity: max(0, count - qualified_all_code[identity])
            for identity, count in qualified_raw.items()
        }
    )
    return (
        qualified_code,
        documentation_mentions,
        constructions,
        self_code,
        import_code,
        ambiguous_code,
    )


def adjudicate(
    variants: Sequence[Variant],
    files: Sequence[SourceFile],
    test_files: Sequence[SourceFile] = (),
) -> list[Verdict]:
    (
        qualified_code,
        documentation_mentions,
        constructions,
        self_code,
        import_code,
        ambiguous_code,
    ) = reference_counts(variants, files)
    (
        test_qualified,
        _test_raw,
        test_constructions,
        test_self,
        test_imports,
        test_ambiguous,
    ) = reference_counts(variants, test_files)
    verdicts = []
    for variant in variants:
        identity = (variant.path, variant.enum, variant.variant)
        qualified_references = qualified_code[identity]
        construction_references = constructions[identity]
        self_references = self_code[identity]
        import_references = import_code[identity]
        ambiguous_references = ambiguous_code[identity]
        test_construction_references = test_constructions[identity]
        test_references = (
            test_qualified[identity]
            + test_self[identity]
            + test_imports[identity]
            + test_ambiguous[identity]
        )
        doc_mentions = documentation_mentions[identity]
        production_candidate = construction_references == 0
        raw_candidate = production_candidate and test_construction_references == 0
        excluded_by = []
        if variant.enum_derives_deserialize:
            excluded_by.append("enum_derives_deserialize")
        if variant.has_from:
            excluded_by.append("variant_has_from")
        if variant.has_default:
            excluded_by.append("variant_is_default")
        if qualified_references:
            excluded_by.append("qualified_reference")
        if self_references:
            excluded_by.append("self_reference")
        if import_references:
            excluded_by.append("variant_import")
        if ambiguous_references:
            excluded_by.append("ambiguous_same_named_enum_reference")
        if test_references:
            excluded_by.append("test_reference")
        refined_candidate = raw_candidate and not excluded_by
        verdicts.append(
            Verdict(
                path=variant.path,
                line=variant.line,
                enum=variant.enum,
                variant=variant.variant,
                construction_references=construction_references,
                qualified_references=qualified_references,
                self_references=self_references,
                import_references=import_references,
                ambiguous_references=ambiguous_references,
                test_construction_references=test_construction_references,
                test_references=test_references,
                doc_mentions=doc_mentions,
                enum_derives_deserialize=variant.enum_derives_deserialize,
                has_from=variant.has_from,
                has_default=variant.has_default,
                production_candidate=production_candidate,
                raw_candidate=raw_candidate,
                refined_candidate=refined_candidate,
                excluded_by=tuple(excluded_by),
            )
        )
    return verdicts


def calibration(verdicts: Sequence[Verdict]) -> dict[str, object]:
    by_identity = {
        (verdict.path, verdict.enum, verdict.variant): verdict for verdict in verdicts
    }
    rows = []
    false_positives = 0
    missing = 0
    for identity in KNOWN_NON_ORPHANS:
        verdict = by_identity.get(identity)
        if verdict is None:
            missing += 1
            rows.append({"identity": identity, "present": False})
            continue
        misclassified = verdict.refined_candidate
        false_positives += int(misclassified)
        rows.append(
            {
                "identity": identity,
                "present": True,
                "refined_candidate": verdict.refined_candidate,
                "excluded_by": verdict.excluded_by,
                "qualified_references": verdict.qualified_references,
            }
        )
    denominator = len(KNOWN_NON_ORPHANS) - missing
    return {
        "known_non_orphans": len(KNOWN_NON_ORPHANS),
        "present": denominator,
        "missing": missing,
        "false_positives": false_positives,
        "false_positive_rate": (
            None if denominator == 0 else false_positives / denominator
        ),
        "rows": rows,
        "known_refined_candidates": len(KNOWN_REFINED_CANDIDATES),
        "false_negatives": sum(
            1
            for identity in KNOWN_REFINED_CANDIDATES
            if identity not in by_identity
            or not by_identity[identity].refined_candidate
        ),
    }


def disposition(verdict: Verdict) -> dict[str, str]:
    identity = (verdict.path, verdict.enum, verdict.variant)
    if identity in MANUAL_DISPOSITIONS:
        return MANUAL_DISPOSITIONS[identity]
    if verdict.has_from or verdict.has_default:
        mechanism = (
            "thiserror generates From construction for the #[from] field"
            if verdict.has_from
            else "derive(Default) constructs the #[default] variant"
        )
        return {
            "classification": "macro_constructible_required",
            "decision": "keep",
            "rationale": mechanism,
        }
    if verdict.enum_derives_deserialize:
        return {
            "classification": "serde_constructible_required",
            "decision": "keep",
            "rationale": "Serde can construct the variant from external input",
        }
    if (
        verdict.qualified_references
        or verdict.self_references
        or verdict.import_references
    ):
        return {
            "classification": "referenced_or_handled",
            "decision": "keep",
            "rationale": "the variant has a source reference outside its declaration",
        }
    return {
        "classification": "unadjudicated",
        "decision": "manual_review_required",
        "rationale": "the refined mechanical rules cannot decide intent",
    }


def synthetic_self_test() -> None:
    fixture = """\
pub enum Calibrated {
    Orphan,
    Wrapped(#[from] std::io::Error),
    Constructed,
    Handled,
    ViaSelf,
    ViaImport,
}
#[derive(Default)]
pub enum DefaultWire {
    #[default]
    Generated,
}
pub enum TraitBuilt {
    Generated,
}
pub enum Flags {
    A = 1 << 0,
    B,
    C,
}
pub enum PatternUse {
    IfLet(u8),
    Matches(u8),
    Guard(u8),
    Tuple(u8),
    GuardBuilt,
    Compared,
    Built,
}
pub enum ClosedMatches {
    Pattern(u8),
    Constructed,
}
pub enum InlineOnly {
    BuiltInTest,
}
pub enum IntegrationOnly {
    BuiltInTest,
}
pub enum FeatureOrTest {
    BuiltByFeature,
}
pub enum TestAndFeature {
    BuiltOnlyInTest,
}
pub enum AfterTestField {
    Live,
}
pub enum AfterTestArm {
    Live,
}
pub enum AfterTestControl {
    Live,
}
pub enum ArmConstructed {
    First,
    Second,
}
pub enum ArmPattern {
    First,
    Second,
}
pub enum TupleArmPattern {
    First,
    Second,
}
pub enum NestedPattern {
    Match,
    Let,
    Built,
}
struct PatternWrapper {
    value: NestedPattern,
}
enum ArmInput {
    Test,
    Production,
}
// Calibrated::Orphan is intentionally documented but never constructed.
fn construct() { let _ = Calibrated::Constructed; }
fn construct_flags() { let _ = [Flags::A, Flags::B, Flags::C]; }
fn pattern_uses(value: PatternUse) {
    if let PatternUse::IfLet(_) = value {}
    let _ = matches!(value, PatternUse::Matches(_));
    match value {
        PatternUse::Guard(number) if number > 0 => {}
        _ => {}
    }
    match (value, true) {
        (PatternUse::Tuple(_), true) => {}
        (_, _) if consume(PatternUse::GuardBuilt) => {}
        _ => {}
    }
    let _ = value == PatternUse::Compared;
    let _ = PatternUse::Built;
}
fn closed_matches_then_construct(value: ClosedMatches) {
    let _ = matches!(value, ClosedMatches::Pattern(_));
    consume(ClosedMatches::Constructed);
}
fn handle(value: Calibrated) {
    match value { Calibrated::Handled => {}, _ => {} }
}
impl Calibrated { fn self_value() -> Self { Self::ViaSelf } }
impl From<()> for TraitBuilt {
    fn from(_: ()) -> Self { Self::Generated }
}
use Calibrated::{ViaImport};
#[derive(Deserialize)]
pub enum Wire {
    FromInput,
}
#[cfg(test)]
mod tests {
    use super::InlineOnly;
    fn construct_inline_only() { let _ = InlineOnly::BuiltInTest; }
}
#[cfg(any(test, feature = "audit-feature"))]
fn construct_feature_or_test() { let _ = FeatureOrTest::BuiltByFeature; }
#[cfg(all(test, feature = "audit-feature"))]
fn construct_test_and_feature() { let _ = TestAndFeature::BuiltOnlyInTest; }
struct TestFieldBoundary {
    #[cfg(test)]
    generic_field: Result<u8, u16>,
}
fn after_test_field() { let _ = AfterTestField::Live; }
fn after_test_arm(value: ArmInput) {
    match value {
        #[cfg(test)]
        ArmInput::Test => { consume_test_arm() }
        ArmInput::Production => { let _ = AfterTestArm::Live; }
    }
}
fn after_test_control() {
    #[cfg(test)]
    if test_condition() {
        test_path();
    } else if second_test_condition() {
        second_test_path();
    } else {
        final_test_path();
    }
    let _ = AfterTestControl::Live;
}
fn construct_in_unbraced_arms(value: u8) {
    match value {
        0 => consume(ArmConstructed::First),
        _ => consume(ArmConstructed::Second),
    }
}
fn consume_unbraced_patterns(value: ArmPattern) -> u8 {
    match value {
        ArmPattern::First => 1,
        ArmPattern::Second => 2,
    }
}
fn consume_tuple_patterns(value: TupleArmPattern) -> u8 {
    match (value, true) {
        (TupleArmPattern::First, true) => 1,
        (TupleArmPattern::Second, false) => 2,
        _ => 3,
    }
}
fn consume_nested_patterns(value: PatternWrapper, branch: u8) {
    match value {
        PatternWrapper { value: NestedPattern::Match } => {}
        _ => {}
    }
    if let PatternWrapper { value: NestedPattern::Let } = value {}
    match branch {
        0 => consume(PatternWrapper { value: NestedPattern::Built }),
        _ => {}
    }
}
"""
    duplicate_a = """\
pub enum Same {
    Shared,
}
"""
    duplicate_b = """\
pub enum Same {
    Shared,
}
fn construct_same() { let _ = Same::Shared; }
"""
    imported_a = """\
pub enum ImportedSame {
    Shared,
}
"""
    imported_b = """\
pub enum ImportedSame {
    Shared,
}
"""
    import_consumer = """\
use first::{ImportedSame};
fn construct_imported_same() { let _ = ImportedSame::Shared; }
"""
    integration_test = """\
use fixture::IntegrationOnly;
fn construct_integration_only() { let _ = IntegrationOnly::BuiltInTest; }
"""

    def fixture_source(relative: str, raw: str) -> SourceFile:
        stripped = strip_rust_comments_and_literals(raw)
        code, test_code = split_inline_test_code(stripped)
        return SourceFile(
            path=Path(relative),
            relative=relative,
            raw=raw,
            code=code,
            test_code=test_code,
        )

    source_files = [
        fixture_source("fixture.rs", fixture),
        fixture_source("duplicate_a.rs", duplicate_a),
        fixture_source("duplicate_b.rs", duplicate_b),
        fixture_source("crate/src/first.rs", imported_a),
        fixture_source("crate/src/second.rs", imported_b),
        fixture_source("crate/src/lib.rs", import_consumer),
    ]
    fixture_file = source_files[0]
    if (
        "generic_field" in fixture_file.code
        or "generic_field" not in fixture_file.test_code
    ):
        raise AssertionError("cfg(test) generic field was not isolated exactly")
    if (
        "test_path" in fixture_file.code
        or "final_test_path" in fixture_file.code
        or "final_test_path" not in fixture_file.test_code
        or "AfterTestControl::Live" not in fixture_file.code
    ):
        raise AssertionError("cfg(test) control flow was not isolated exactly")
    test_files = [
        SourceFile(
            path=source.path,
            relative=source.relative,
            raw=source.test_code,
            code=source.test_code,
        )
        for source in source_files
        if source.test_code.strip()
    ]
    test_files.append(
        SourceFile(
            path=Path("tests/integration.rs"),
            relative="tests/integration.rs",
            raw=integration_test,
            code=strip_rust_comments_and_literals(integration_test),
        )
    )
    variants = discover_variants(source_files)
    discovered_flags = {
        variant.variant for variant in variants if variant.enum == "Flags"
    }
    if discovered_flags != {"A", "B", "C"}:
        raise AssertionError(f"discriminant fixture mismatch: {discovered_flags!r}")
    verdicts = adjudicate(variants, source_files, test_files)
    pattern_constructions = {
        verdict.variant: verdict.construction_references
        for verdict in verdicts
        if verdict.enum == "PatternUse"
    }
    expected_pattern_constructions = {
        "IfLet": 0,
        "Matches": 0,
        "Guard": 0,
        "Tuple": 0,
        "GuardBuilt": 1,
        "Compared": 0,
        "Built": 1,
    }
    if pattern_constructions != expected_pattern_constructions:
        raise AssertionError(
            "pattern/construction fixture mismatch: "
            f"{pattern_constructions!r} != {expected_pattern_constructions!r}"
        )
    arm_constructions = {
        verdict.variant: verdict.construction_references
        for verdict in verdicts
        if verdict.enum == "ArmConstructed"
    }
    if arm_constructions != {"First": 1, "Second": 1}:
        raise AssertionError(
            f"unbraced arm constructor fixture mismatch: {arm_constructions!r}"
        )
    arm_patterns = {
        verdict.variant: verdict.construction_references
        for verdict in verdicts
        if verdict.enum == "ArmPattern"
    }
    if arm_patterns != {"First": 0, "Second": 0}:
        raise AssertionError(f"unbraced arm pattern fixture mismatch: {arm_patterns!r}")
    tuple_arm_patterns = {
        verdict.variant: verdict.construction_references
        for verdict in verdicts
        if verdict.enum == "TupleArmPattern"
    }
    if tuple_arm_patterns != {"First": 0, "Second": 0}:
        raise AssertionError(
            f"tuple arm pattern fixture mismatch: {tuple_arm_patterns!r}"
        )
    nested_patterns = {
        verdict.variant: verdict.construction_references
        for verdict in verdicts
        if verdict.enum == "NestedPattern"
    }
    if nested_patterns != {"Match": 0, "Let": 0, "Built": 1}:
        raise AssertionError(f"nested pattern fixture mismatch: {nested_patterns!r}")
    closed_matches_constructions = {
        verdict.variant: verdict.construction_references
        for verdict in verdicts
        if verdict.enum == "ClosedMatches"
    }
    if closed_matches_constructions != {"Pattern": 0, "Constructed": 1}:
        raise AssertionError(
            f"closed matches fixture mismatch: {closed_matches_constructions!r}"
        )
    test_only_counts = {
        verdict.enum: (
            verdict.construction_references,
            verdict.test_construction_references,
        )
        for verdict in verdicts
        if verdict.enum
        in {
            "InlineOnly",
            "IntegrationOnly",
            "FeatureOrTest",
            "TestAndFeature",
        }
    }
    expected_test_only_counts = {
        "InlineOnly": (0, 1),
        "IntegrationOnly": (0, 1),
        "FeatureOrTest": (1, 0),
        "TestAndFeature": (0, 1),
    }
    if test_only_counts != expected_test_only_counts:
        raise AssertionError(
            f"test-only fixture mismatch: {test_only_counts!r} "
            f"!= {expected_test_only_counts!r}"
        )
    doc_counts = {
        (verdict.enum, verdict.variant): verdict.doc_mentions
        for verdict in verdicts
        if (verdict.enum, verdict.variant)
        in {
            ("Calibrated", "Orphan"),
            ("InlineOnly", "BuiltInTest"),
        }
    }
    expected_doc_counts = {
        ("Calibrated", "Orphan"): 1,
        ("InlineOnly", "BuiltInTest"): 0,
    }
    if doc_counts != expected_doc_counts:
        raise AssertionError(
            f"documentation fixture mismatch: {doc_counts!r} != {expected_doc_counts!r}"
        )
    refined = {
        (verdict.path, verdict.enum, verdict.variant)
        for verdict in verdicts
        if verdict.refined_candidate
    }
    expected = {
        ("fixture.rs", "Calibrated", "Orphan"),
        ("duplicate_a.rs", "Same", "Shared"),
        ("crate/src/second.rs", "ImportedSame", "Shared"),
    }
    if refined != expected:
        raise AssertionError(f"refined fixture mismatch: {refined!r} != {expected!r}")


def render_human(
    verdicts: Sequence[Verdict],
    calibration_result: dict[str, object],
) -> None:
    production = [verdict for verdict in verdicts if verdict.production_candidate]
    raw = [verdict for verdict in verdicts if verdict.raw_candidate]
    refined = [verdict for verdict in verdicts if verdict.refined_candidate]
    rate = calibration_result["false_positive_rate"]
    rendered_rate = "n/a" if rate is None else f"{float(rate):.1%}"
    print(
        f"public variants={len(verdicts)} "
        f"production_zero_construction={len(production)} "
        f"workspace_zero_construction={len(raw)} "
        f"refined_candidates={len(refined)}"
    )
    print(
        "known-verdict calibration: "
        f"{calibration_result['false_positives']}/"
        f"{calibration_result['present']} false positives ({rendered_rate}), "
        f"{calibration_result['missing']} missing; "
        f"{calibration_result['false_negatives']}/"
        f"{calibration_result['known_refined_candidates']} known candidates missed"
    )
    print("\nRAW ZERO-CONSTRUCTION SET")
    for verdict in raw:
        exclusions = ",".join(verdict.excluded_by) or "-"
        classification = disposition(verdict)["classification"]
        print(
            f"{verdict.path}:{verdict.line} "
            f"{verdict.enum}::{verdict.variant} "
            f"constructions={verdict.construction_references} "
            f"self={verdict.self_references} imports={verdict.import_references} "
            f"tests={verdict.test_construction_references}/"
            f"{verdict.test_references} docs={verdict.doc_mentions} "
            f"excluded={exclusions} "
            f"class={classification} decision={disposition(verdict)['decision']}"
        )
    print("\nREFINED CANDIDATES — HUMAN ADJUDICATION REQUIRED")
    for verdict in refined:
        adjudication = disposition(verdict)
        print(
            f"{verdict.path}:{verdict.line} "
            f"{verdict.enum}::{verdict.variant} docs={verdict.doc_mentions} "
            f"class={adjudication['classification']} "
            f"decision={adjudication['decision']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conservatively audit declaration-only public Rust enum variants."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to this script's parent repository)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    parser.add_argument(
        "--revision",
        help="scan Rust sources from a Git revision without checking it out",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the embedded parser/refinement fixture before scanning",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail on calibration drift or any unadjudicated zero-production candidate",
    )
    args = parser.parse_args()

    if args.self_test:
        synthetic_self_test()

    root = args.root.resolve()
    all_files = discover_source(
        root,
        args.revision,
        include_tests=True,
    )
    files = [
        source_file
        for source_file in all_files
        if not is_test_source(source_file.relative)
    ]
    test_files = [
        source_file for source_file in all_files if is_test_source(source_file.relative)
    ]
    integration_test_files = len(test_files)
    test_files.extend(
        SourceFile(
            path=source_file.path,
            relative=source_file.relative,
            raw=source_file.test_code,
            code=source_file.test_code,
        )
        for source_file in files
        if source_file.test_code.strip()
    )
    inline_test_sources = len(test_files) - integration_test_files
    verdicts = adjudicate(discover_variants(files), files, test_files)
    calibration_result = calibration(verdicts)
    if args.json:
        print(
            json.dumps(
                {
                    "schema": "scriptbots.orphan-variants.v1",
                    "root": str(root),
                    "revision": args.revision,
                    "source_files": len(files),
                    "integration_test_files": integration_test_files,
                    "inline_test_sources": inline_test_sources,
                    "test_reference_sources": len(test_files),
                    "public_variants": len(verdicts),
                    "production_only_candidates": [
                        {
                            **asdict(verdict),
                            "disposition": disposition(verdict),
                        }
                        for verdict in verdicts
                        if verdict.production_candidate and not verdict.raw_candidate
                    ],
                    "raw_candidates": [
                        {
                            **asdict(verdict),
                            "disposition": disposition(verdict),
                        }
                        for verdict in verdicts
                        if verdict.raw_candidate
                    ],
                    "refined_candidates": [
                        {
                            **asdict(verdict),
                            "disposition": disposition(verdict),
                        }
                        for verdict in verdicts
                        if verdict.refined_candidate
                    ],
                    "calibration": calibration_result,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        render_human(verdicts, calibration_result)
    if args.check:
        unadjudicated = [
            verdict
            for verdict in verdicts
            if verdict.production_candidate
            and disposition(verdict)["classification"] == "unadjudicated"
        ]
        calibration_failed = any(
            calibration_result[key]
            for key in ("missing", "false_positives", "false_negatives")
        )
        if unadjudicated or calibration_failed:
            print(
                "orphan-variant check failed: "
                f"unadjudicated={len(unadjudicated)} "
                f"missing_calibration={calibration_result['missing']} "
                f"false_positives={calibration_result['false_positives']} "
                f"false_negatives={calibration_result['false_negatives']}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
