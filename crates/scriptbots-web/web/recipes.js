// Browser launch recipes describe initial conditions. Saved-owner continuations
// and legacy gallery references carry different, explicitly bounded metadata.
export function validateLaunchInfo(info) {
  const kinds = { browser_launch: "initial_conditions_restart", gallery_reference: "gallery_configuration_restart",
    checkpoint_continuation: "saved_owner_continuation" };
  if (!info || Array.isArray(info) || !Object.hasOwn(kinds, info.kind) || info.semantics !== kinds[info.kind]
      || typeof info.shareable !== "boolean" || (info.reason !== null && (typeof info.reason !== "string" || info.reason.length > 2048))) {
    throw new Error("The browser package returned invalid launch metadata");
  }
  if (info.kind === "checkpoint_continuation") {
    if (info.shareable || [info.seed, info.initial_population, info.brain_preset, info.recipe_version].some((value) => value !== null)) {
      throw new Error("A continuation cannot claim an unrecorded launch recipe");
    }
  } else {
    if (typeof info.seed !== "string" || !/^(0|[1-9][0-9]{0,19})$/.test(info.seed)
        || BigInt(info.seed) > 18446744073709551615n) throw new Error("Recipe seed must be an exact u64 decimal string");
    if (info.kind === "browser_launch") {
      if (!Number.isSafeInteger(info.initial_population) || info.initial_population < 0 || info.initial_population > 50_000
          || !["mlp", "dwraon", "assembly", "wander", "none"].includes(info.brain_preset) || info.recipe_version !== 1) {
        throw new Error("Browser launch metadata has an unsupported founder recipe");
      }
    } else if (!info.shareable || [info.initial_population, info.brain_preset, info.recipe_version].some((value) => value !== null)) {
      throw new Error("A gallery reference cannot claim a browser founder recipe");
    }
  }
  return { ...info };
}

export function launchInfoText(info) {
  if (!info) return "Initial-condition metadata is unavailable in this browser package.";
  if (info.kind === "browser_launch") {
    const recipe = `Initial seed ${info.seed} · ${info.initial_population} founders · ${info.brain_preset} brains. Recipe v${info.recipe_version}.`;
    return info.shareable ? `${recipe} Links restart these initial conditions at tick zero.`
      : `${recipe} Recipe sharing unavailable: ${info.reason || "this recipe exceeds the link size limit"} Save a continuation to retain this world.`;
  }
  if (info.kind === "gallery_reference") return `Gallery reference · seed ${info.seed}. This link does not encode a browser founder population or brain recipe.`;
  return "Saved owner continuation. The original launch recipe is unavailable; save another continuation to retain this evolved world.";
}

function line(parent, text, className = "") {
  const element = parent.ownerDocument.createElement("div");
  element.className = className;
  element.textContent = text;
  parent.append(element);
  return element;
}

export function renderBuildComparison(banner, matchResult) {
  if (!banner) return;
  banner.replaceChildren();
  banner.className = "";
  if (!matchResult || (matchResult.status === "exact" && matchResult.identity_scope !== "legacy_gallery_reference")) {
    banner.style.display = "none";
    return;
  }

  const recorded = matchResult.identity_scope === "recorded_browser_build_inputs";
  const legacy = matchResult.identity_scope === "legacy_gallery_reference";
  const unknown = matchResult.status === "unavailable";
  const recordedMatch = recorded && matchResult.status === "recorded_match";
  const legacyExact = matchResult.status === "exact" && matchResult.identity_scope === "legacy_gallery_reference";
  banner.style.display = "block";
  if (!unknown && !recordedMatch && !legacyExact && matchResult.status !== "compatible" && matchResult.status !== "mismatch") {
    banner.className = "banner-compatible";
    line(banner, "Build comparison metadata is unavailable.");
    return;
  }
  const compatible = matchResult.status === "compatible";
  const informational = compatible || recordedMatch || unknown || legacyExact;
  banner.className = informational ? "banner-compatible" : "banner-mismatch";
  const title = line(banner, "", "banner-title");
  const icon = banner.ownerDocument.createElement("span");
  icon.textContent = informational ? "ⓘ" : "⚠";
  const titleText = banner.ownerDocument.createElement("span");
  titleText.textContent = unknown ? "Build Identity Unavailable" : recordedMatch ? "Recorded Build Inputs Match"
    : legacyExact ? "Gallery Build Reference" : legacy ? compatible ? "Gallery Reference Difference" : "Gallery Reference Mismatch"
    : compatible ? "Compatible Build Note" : "Build Mismatch Warning";
  title.append(icon, titleText);
  line(banner, unknown ? "The recorded build inputs cannot be compared completely; build equality is unknown."
    : recordedMatch ? "The recorded build input fingerprints agree. This does not establish byte-identical binaries or cross-target scientific equivalence."
    : legacyExact ? "The gallery's build references agree. They do not establish measured compiler or source identity."
    : legacy ? compatible
      ? "The gallery core reference agrees, while toolchain or lockfile references differ. These labels do not establish measured build identity."
      : "The gallery build references differ. These labels do not establish measured compiler or source identity."
    : recorded ? "This recipe records different build inputs. Starting it does not establish the same scientific trajectory."
    : compatible
    ? "This world was recorded on a compatible build. Toolchain or lockfile differ, but core simulation digest matches."
    : `This world was recorded on a different build (core digest ${matchResult.link_core_digest}). You are on build (core digest ${matchResult.local_core_digest}). Trajectories may diverge after tick 0.`);
  const details = line(banner, "", "banner-details");
  line(details, `Link Build: toolchain=${matchResult.link_toolchain_digest}, lockfile=${matchResult.link_lockfile_digest}, core=${matchResult.link_core_digest}`);
  line(details, `Running Build: toolchain=${matchResult.local_toolchain_digest}, lockfile=${matchResult.local_lockfile_digest}, core=${matchResult.local_core_digest}`);
  if (typeof matchResult.evidence === "string") line(details, matchResult.evidence.slice(0, 2048));
  if (!informational) {
    const note = line(details, "Cross-target floating point equivalence between native and WebAssembly is an open research area (bd-2wj).");
    note.style.cssText = "margin-top: 0.35rem; font-style: italic; opacity: 0.85;";
  }
}
