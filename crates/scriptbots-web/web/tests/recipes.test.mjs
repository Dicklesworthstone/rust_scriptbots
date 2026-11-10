import assert from "node:assert/strict";
import test from "node:test";
import { validateLaunchInfo, launchInfoText, renderBuildComparison } from "../recipes.js";

const browserRecipe = (overrides = {}) => ({
  kind: "browser_launch", semantics: "initial_conditions_restart", shareable: true,
  seed: "18446744073709551615", initial_population: 151, brain_preset: "dwraon", recipe_version: 1, reason: null,
  ...overrides,
});
const galleryReference = () => ({
  kind: "gallery_reference", semantics: "gallery_configuration_restart", shareable: true,
  seed: "9007199254740993", initial_population: null, brain_preset: null, recipe_version: null,
  reason: "No browser founder recipe is encoded.",
});
const continuation = () => ({
  kind: "checkpoint_continuation", semantics: "saved_owner_continuation", shareable: false,
  seed: null, initial_population: null, brain_preset: null, recipe_version: null,
  reason: "Original launch recipe is unavailable.",
});

test("browser recipes preserve exact full-width seeds and registered empty-world families", () => {
  const decoded = validateLaunchInfo(browserRecipe());
  assert.equal(decoded.seed, "18446744073709551615");
  const text = launchInfoText(decoded);
  assert.ok(text.includes("18446744073709551615"));
  assert.ok(text.includes("151 founders"));
  assert.ok(text.includes("dwraon brains"));
  assert.ok(text.includes("tick zero"));
  for (const brain_preset of ["mlp", "dwraon", "assembly", "wander", "none"]) {
    for (const initial_population of [0, 50_000]) {
      assert.equal(validateLaunchInfo(browserRecipe({ brain_preset, initial_population })).initial_population, initial_population);
    }
  }
});

test("malformed metadata cannot round seeds or claim a different continuation contract", () => {
  for (const metadata of [null, [], {}, browserRecipe({ semantics: "saved_owner_continuation" }),
    browserRecipe({ shareable: "true" }), browserRecipe({ seed: 9007199254740992 }),
    browserRecipe({ seed: "18446744073709551616" }), browserRecipe({ seed: "01" }), browserRecipe({ seed: "-1" }),
    browserRecipe({ seed: "1e3" }), browserRecipe({ seed: "9".repeat(1000) }),
    browserRecipe({ initial_population: -1 }), browserRecipe({ initial_population: 50_001 }),
    browserRecipe({ initial_population: 1.5 }), browserRecipe({ brain_preset: "unknown" }),
    browserRecipe({ recipe_version: 2 }), browserRecipe({ reason: "x".repeat(2049) })]) {
    assert.throws(() => validateLaunchInfo(metadata));
  }
});

test("gallery references and saved owners retain their narrower recorded evidence", () => {
  const gallery = validateLaunchInfo(galleryReference());
  assert.equal(gallery.seed, "9007199254740993");
  assert.ok(launchInfoText(gallery).includes("does not encode a browser founder population or brain recipe"));
  const saved = validateLaunchInfo(continuation());
  assert.equal(saved.shareable, false);
  assert.ok(launchInfoText(saved).includes("original launch recipe is unavailable"));
  assert.ok(launchInfoText(null).includes("metadata is unavailable"));
  for (const metadata of [{ ...gallery, initial_population: 0 }, { ...gallery, brain_preset: "none" },
    { ...saved, shareable: true }, { ...saved, seed: "0" }, { ...saved, recipe_version: 1 }]) {
    assert.throws(() => validateLaunchInfo(metadata));
  }
});

test("a usable owner with an oversized recipe keeps its known launch metadata", () => {
  const decoded = validateLaunchInfo(browserRecipe({ shareable: false, reason: "Recipe exceeds the encoded byte limit." }));
  assert.equal(decoded.shareable, false);
  const text = launchInfoText(decoded);
  assert.ok(text.includes("18446744073709551615"));
  assert.ok(text.includes("151 founders"));
  assert.ok(text.includes("Recipe exceeds the encoded byte limit."));
  assert.ok(text.includes("Save a continuation"));
});

// This small DOM fixture observes text-node writes and state transitions. The
// production-page integration lives in permalink_fork_dom.test.js.
function bannerFixture() {
  class Element {
    constructor(tagName, ownerDocument) {
      Object.assign(this, { tagName, ownerDocument, children: [], style: {}, className: "", ownText: "" });
    }
    set textContent(text) { this.ownText = String(text); this.children = []; }
    get textContent() { return this.ownText + this.children.map((child) => child.textContent).join(""); }
    set innerHTML(_) { throw new Error("Build evidence must be written as text"); }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.ownText = ""; this.children = children; }
  }
  const document = { createElement: (tagName) => new Element(tagName, document) };
  return document.createElement("div");
}

const build = (overrides = {}) => ({
  status: "recorded_match", identity_scope: "recorded_browser_build_inputs",
  link_toolchain_digest: "0x1111", link_lockfile_digest: "0x2222", link_core_digest: "0x3333",
  local_toolchain_digest: "0x1111", local_lockfile_digest: "0x2222", local_core_digest: "0x3333",
  evidence: "<img src=x onerror=example()> recorded input evidence", ...overrides,
});

test("matching recorded inputs are displayed without promoting them to binary identity", () => {
  const banner = bannerFixture();
  renderBuildComparison(banner, build());
  assert.equal(banner.style.display, "block");
  assert.equal(banner.className, "banner-compatible");
  assert.ok(banner.textContent.includes("Recorded Build Inputs Match"));
  assert.ok(banner.textContent.includes("does not establish byte-identical binaries"));
  assert.ok(banner.textContent.includes("<img src=x onerror=example()> recorded input evidence"));
  assert.equal(banner.children[0].children[1].tagName, "span");
});

test("unavailable identity clears earlier mismatch state and remains informational", () => {
  const banner = bannerFixture();
  renderBuildComparison(banner, build({ status: "mismatch", link_core_digest: "old-mismatch-input" }));
  assert.equal(banner.className, "banner-mismatch");
  assert.ok(banner.textContent.includes("Build Mismatch Warning"));
  renderBuildComparison(banner, build({ status: "unavailable", link_core_digest: "unavailable" }));
  assert.equal(banner.className, "banner-compatible");
  assert.ok(banner.textContent.includes("Build Identity Unavailable"));
  assert.ok(banner.textContent.includes("equality is unknown"));
  assert.ok(!banner.textContent.includes("old-mismatch-input"));
  assert.ok(!banner.textContent.includes("Build Mismatch Warning"));
});

test("legacy gallery references stay labelled and unknown status cannot leave a stale comparison", () => {
  const banner = bannerFixture();
  renderBuildComparison(banner, build({ status: "exact", identity_scope: "legacy_gallery_reference" }));
  assert.equal(banner.style.display, "block");
  assert.ok(banner.textContent.includes("Gallery Build Reference"));
  assert.ok(banner.textContent.includes("do not establish measured compiler or source identity"));
  for (const status of ["compatible", "mismatch"]) {
    renderBuildComparison(banner, build({ status, identity_scope: "legacy_gallery_reference" }));
    assert.ok(banner.textContent.includes("Gallery Reference"));
    assert.ok(banner.textContent.includes("do not establish measured"));
    assert.ok(!banner.textContent.includes("This world was recorded on"));
  }
  renderBuildComparison(banner, build({ status: "future-status" }));
  assert.equal(banner.className, "banner-compatible");
  assert.equal(banner.textContent, "Build comparison metadata is unavailable.");
  renderBuildComparison(banner, null);
  assert.equal(banner.style.display, "none");
  assert.equal(banner.textContent, "");
  renderBuildComparison(null, build());
});
