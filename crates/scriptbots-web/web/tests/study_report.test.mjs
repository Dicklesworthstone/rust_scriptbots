import test from "node:test";
import assert from "node:assert/strict";
import { MAX_STUDY_REPORT_BYTES, parseSavedStudyReport, renderStudyReport } from "../study-report.js";

const hash = (letter) => letter.repeat(64);
const status = Object.freeze({
  branch_id: "branch-browser-study", checkpoint_id: "ckpt-browser-source",
  branch_name: "browser-study", status: "completed", source_tick: 10, target_tick: 12,
  world_digest_v1: hash("b"), artifact_id: "checkpoint-branch-report",
});
const checkpoint = Object.freeze({
  checkpoint_id: status.checkpoint_id, tick: 10, byte_size: 128, agent_count: 3,
  world_digest_v1: status.world_digest_v1, checksum_blake3: hash("a"),
});

// Explicit contract fixture matching BranchStudyReport schema 4 and its serde
// field names. These hand-computed observations are not real-run science evidence.
function reportFixture() {
  const sample = { tick: 12, population: 3, food_total: 10, births: 0, deaths: 0,
    total_energy: 3, average_health: 1, spike_hits: 0, digest_overall: hash("c") };
  const recovery = { baseline_tick: 10, pre_intervention_population: 3, trough_population: 3,
    trough_tick: 10, final_population: 3, recovered: true, recovery_ratio: 1, recovery_ticks: 1 };
  const control = {
    branch_id: "control", description: "browser-study: unchanged scientific future",
    command_applied_tick: null, first_divergence: null,
    extinction: { extinct: false, extinction_tick: null, min_population: 3, final_population: 3 },
    recovery,
    hysteresis: { integrated_trajectory_deficit: 0, normalized_trajectory_difference: 0,
      terminal_trajectory_gap: 0, hysteresis_index: 0, path_dependence_detected: false },
    effect_sizes: {}, final_digest: hash("c"), terminal_sample: sample,
    bundle_path: "/retained/branches/branch-browser-study/study/bundle_control", bundle_verified: true,
  };
  const effect = (metric, mean, sd) => ({ metric,
    analysis_basis: { kind: "single_seed_trajectory", seed: 42 }, n_samples: 3, mean_diff: mean, sd_diff: sd });
  const treatment = {
    branch_id: "treatment", description: "browser-study: requested intervention",
    command_applied_tick: 10,
    first_divergence: { branch_id: "treatment", tick: 11, stage: "food", control_digest: hash("e"),
      branch_digest: hash("f"), command_applied_tick: 10, details: "first differing food lane" },
    extinction: { extinct: false, extinction_tick: null, min_population: 2, final_population: 2 },
    recovery: { ...recovery, trough_population: 2, trough_tick: 11, final_population: 2,
      recovered: false, recovery_ratio: 2 / 3, recovery_ticks: null },
    hysteresis: { integrated_trajectory_deficit: 2, normalized_trajectory_difference: 2 / 9,
      terminal_trajectory_gap: 1 / 3, hysteresis_index: 1, path_dependence_detected: true },
    effect_sizes: { population: effect("population", -2 / 3, Math.sqrt(1 / 3)),
      food_total: effect("food_total", -8 / 3, Math.sqrt(16 / 3)),
      total_energy: effect("total_energy", -5 / 6, Math.sqrt(7 / 12)) },
    final_digest: hash("d"), terminal_sample: { ...sample, population: 2, food_total: 6,
      total_energy: 1.5, average_health: 0.75, deaths: 1, spike_hits: 1, digest_overall: hash("d") },
    bundle_path: "/retained/branches/branch-browser-study/study/bundle_treatment", bundle_verified: true,
  };
  return {
    study_id: status.branch_id, schema_version: 4, base_seed: 42,
    source: { kind: "saved_checkpoint", artifact_path: "source_checkpoint.bin", byte_size: 128, checksum_blake3: hash("a") },
    checkpoint_tick: 10, horizon_ticks: 2, pre_intervention_digest: hash("b"),
    pre_intervention_digest_verified_for_all_branches: true, control_branch_id: "control",
    branch_reports: [control, treatment], summary_markdown: "# Saved study\nDescriptive trajectory comparison.",
  };
}

function input(value = reportFixture(), context = {}) {
  const blob = value instanceof Blob ? value : new Blob([typeof value === "string" ? value : JSON.stringify(value)]);
  return { blob, context: { status, checkpoint,
    artifact: { artifact_id: status.artifact_id, byte_size: blob.size, checksum_sha256: hash("e"),
      relative_path: "branches/" + status.branch_id + "/study/study_report.json" }, ...context } };
}
async function parse(value, context) {
  const prepared = input(value, context);
  return parseSavedStudyReport(prepared.blob, prepared.context);
}

test("schema 4 saved report binds source and separates endpoint deltas from trajectory means", async () => {
  const model = await parse(reportFixture());
  assert.equal(model.studyId, status.branch_id);
  assert.equal(model.checkpointId, checkpoint.checkpoint_id);
  assert.equal(model.seed, "42");
  assert.equal(model.sourceTick, 10);
  assert.equal(model.targetTick, 12);
  assert.equal(model.source.checksumBlake3, checkpoint.checksum_blake3);
  assert.deepEqual(model.endpoints.find((row) => row.metric === "population"), {
    metric: "population", label: "Population", integral: true, control: 3, treatment: 2, difference: -1,
  });
  assert.equal(model.trajectoryComparisons.find((row) => row.metric === "population").meanDifference, -2 / 3);
  assert.equal(model.trajectoryComparisons[0].observations, 3);
  assert.equal(model.arms[1].divergence.tick, 11);
  assert.ok(Object.isFrozen(model));
  assert.ok(Object.isFrozen(model.arms[1].terminal));
  assert.throws(() => { model.endpoints[0].difference = 900; }, TypeError);
});

test("exact full-width u64 seed survives JSON reading and binds every comparison", async () => {
  const wire = JSON.stringify(reportFixture()).replace('"base_seed":42', '"base_seed":18446744073709551615')
    .split('"seed":42').join('"seed":18446744073709551615');
  assert.equal((await parse(wire)).seed, "18446744073709551615");
  await assert.rejects(parse(wire.replace('"base_seed":18446744073709551615', '"base_seed":18446744073709551616')), /exact bounded integer/);
  await assert.rejects(parse(wire.replace('"seed":18446744073709551615', '"seed":18446744073709551614')), /independent-seed statistical inference/);
});

test("retained history can validate a report from its exact completed status without a live checkpoint row", async () => {
  const model = await parse(reportFixture(), { checkpoint: null });
  assert.equal(model.checkpointId, status.checkpoint_id);
  assert.equal(model.worldDigest, status.world_digest_v1);
  await assert.rejects(parse(reportFixture(), { checkpoint: null, status: { ...status, world_digest_v1: hash("f") } }), /selected saved-world study/);
  await assert.rejects(parse(reportFixture(), { checkpoint: { ...checkpoint, agent_count: 4 } }), /population baseline/);
});

test("control-only report produces no invented treatment, differences, or significance", async () => {
  const report = reportFixture();
  report.branch_reports.length = 1;
  const model = await parse(report);
  assert.equal(model.hasTreatment, false);
  assert.equal(model.arms.length, 1);
  assert.deepEqual(model.trajectoryComparisons, []);
  assert.ok(model.endpoints.every((row) => row.treatment === null && row.difference === null));
});

test("arm order is immaterial, while duplicates and absent controls are refused", async () => {
  const report = reportFixture();
  report.branch_reports.reverse();
  assert.deepEqual((await parse(report)).arms.map((arm) => arm.id), ["control", "treatment"]);
  report.branch_reports = [report.branch_reports[0], report.branch_reports[0]];
  await assert.rejects(parse(report), /missing or duplicate study arm/);
  report.branch_reports.length = 1;
  await assert.rejects(parse(report), /missing or duplicate study arm/);
});

test("report identities, bounds, source bytes and terminal digests are never substituted", async (t) => {
  const cases = [
    ["another study", (r) => { r.study_id = "branch-another"; }],
    ["unsupported schema", (r) => { r.schema_version = 3; }],
    ["different saved tick", (r) => { r.checkpoint_tick = 11; }],
    ["different horizon", (r) => { r.horizon_ticks = 3; }],
    ["unverified initial state", (r) => { r.pre_intervention_digest_verified_for_all_branches = false; }],
    ["different source digest", (r) => { r.pre_intervention_digest = hash("f"); }],
    ["different source bytes", (r) => { r.source.byte_size = 129; }],
    ["different source checksum", (r) => { r.source.checksum_blake3 = hash("e"); }],
    ["warmup cannot replace saved source", (r) => { r.source = { kind: "seeded_warmup" }; }],
    ["source traversal", (r) => { r.source.artifact_path = "../source_checkpoint.bin"; }],
    ["wrong final boundary", (r) => { r.branch_reports[1].terminal_sample.tick = 11; }],
    ["digest disagreement", (r) => { r.branch_reports[1].final_digest = hash("f"); }],
    ["unverified bundle", (r) => { r.branch_reports[0].bundle_verified = false; }],
    ["unrelated arm", (r) => { r.branch_reports[1].branch_id = "another"; }],
    ["too many arms", (r) => { r.branch_reports.push(structuredClone(r.branch_reports[1])); }],
  ];
  for (const [label, change] of cases) await t.test(label, async () => {
    const report = reportFixture(); change(report);
    await assert.rejects(parse(report), /Study report refused/);
  });
});

test("status and downloaded artifact must identify the currently selected completed study", async () => {
  const { blob, context } = input();
  for (const wrongStatus of [
    { ...status, status: "running" }, { ...status, status: "stopped" },
    { ...status, source_tick: Number.MAX_SAFE_INTEGER + 1 },
  ]) await assert.rejects(parseSavedStudyReport(blob, { ...context, status: wrongStatus }), /completed, exact study receipt/);
  await assert.rejects(parseSavedStudyReport(blob, { ...context, artifact: { ...context.artifact, artifact_id: "wrong" } }), /selected artifact receipt/);
  await assert.rejects(parseSavedStudyReport(blob, { ...context, artifact: { ...context.artifact, byte_size: blob.size + 1 } }), /selected artifact receipt/);
  await assert.rejects(parseSavedStudyReport(blob, { ...context, artifact: { ...context.artifact, relative_path: "branches/another/study_report.json" } }), /selected artifact receipt/);
});

test("nonfinite, negative, inexact or wrong-type observations are refused", async () => {
  const cases = [
    (r) => { r.branch_reports[0].terminal_sample.food_total = -1; },
    (r) => { r.branch_reports[0].terminal_sample.total_energy = null; },
    (r) => { r.branch_reports[0].terminal_sample.population = 1.5; },
    (r) => { r.branch_reports[0].terminal_sample.population = Number.MAX_SAFE_INTEGER + 1; },
    (r) => { r.branch_reports[0].terminal_sample.average_health = "1"; },
    (r) => { r.schema_version = "4"; },
    (r) => { r.base_seed = "42"; },
  ];
  for (const change of cases) {
    const report = reportFixture(); change(report);
    await assert.rejects(parse(report), /Study report refused/);
  }
  await assert.rejects(parse(JSON.stringify(reportFixture()).replace('"food_total":10', '"food_total":1e309')), /finite bounded number/);
  await assert.rejects(parse(JSON.stringify(reportFixture()).replace('"checkpoint_tick":10', '"checkpoint_tick":10.0')), /exact bounded integer/);
});

test("a single trajectory cannot smuggle replicated confidence intervals or mismatched samples", async () => {
  for (const [field, value] of [["ci_95", [-2, -1]], ["p_value", 0.01], ["cohens_d", 2], ["hedges_g", 2]]) {
    const report = reportFixture();
    report.branch_reports[1].effect_sizes.population[field] = value;
    await assert.rejects(parse(report), /independent-seed statistical inference/);
  }
  const report = reportFixture(), effect = report.branch_reports[1].effect_sizes.population;
  effect.n_samples = 2;
  await assert.rejects(parse(report), /observation count/);
  effect.n_samples = 3;
  effect.analysis_basis = { kind: "independent_paired_replicates", pairs: 3 };
  await assert.rejects(parse(report), /analysis basis/);
});

test("missing recorded effects stay absent rather than becoming zero-valued claims", async () => {
  const report = reportFixture();
  delete report.branch_reports[1].effect_sizes.population;
  const model = await parse(report);
  assert.equal(model.trajectoryComparisons.length, 2);
  assert.ok(model.trajectoryComparisons.every((effect) => effect.metric !== "population"));
  assert.equal(model.endpoints[0].difference, -1);
});

test("first divergence and extinction bind their actual branch and observation boundaries", async () => {
  const changes = [
    (arm) => { arm.first_divergence.branch_id = "control"; },
    (arm) => { arm.first_divergence.tick = 10; },
    (arm) => { arm.first_divergence.control_digest = arm.first_divergence.branch_digest; },
    (arm) => { arm.first_divergence = null; },
    (arm) => { arm.extinction.extinct = true; },
    (arm) => { arm.extinction.final_population = 100; },
    (arm) => { arm.recovery.recovery_ratio = 10; },
  ];
  for (const change of changes) {
    const report = reportFixture(); change(report.branch_reports[1]);
    await assert.rejects(parse(report), /Study report refused/);
  }
});

test("canonical non-hash RNG and absent hydrology divergence evidence remains inspectable", async () => {
  for (const [stage, controlEvidence, branchEvidence] of [
    ["rng", "RngProbe { words: [1, 2, 3, 4] }", "RngProbe { words: [1, 2, 3, 5] }"],
    ["hydrology", "absent", hash("f")],
    ["hydrology", hash("e"), "absent"],
  ]) {
    const report = reportFixture(), first = report.branch_reports[1].first_divergence;
    first.stage = stage;
    first.control_digest = controlEvidence;
    first.branch_digest = branchEvidence;
    const model = await parse(report);
    assert.equal(model.arms[1].divergence.controlEvidence, controlEvidence);
    assert.equal(model.arms[1].divergence.branchEvidence, branchEvidence);
    assert.equal(model.arms[1].divergence.stage, stage);
  }
  for (const invalid of ["", "x".repeat(4097), ["absent"]]) {
    const report = reportFixture();
    report.branch_reports[1].first_divergence.control_digest = invalid;
    await assert.rejects(parse(report), /divergence evidence/);
  }
  const report = reportFixture();
  report.branch_reports[1].first_divergence.control_digest = "absent";
  report.branch_reports[1].first_divergence.branch_digest = "absent";
  await assert.rejects(parse(report), /evidence is identical/);
});

test("a zero-population baseline never becomes a fabricated recovery percentage", async () => {
  const report = reportFixture();
  report.branch_reports.length = 1;
  const arm = report.branch_reports[0];
  arm.extinction = { extinct: true, extinction_tick: 10, min_population: 0, final_population: 3 };
  arm.recovery = { baseline_tick: 10, pre_intervention_population: 0, trough_population: 0,
    trough_tick: 10, final_population: 3, recovered: true, recovery_ratio: 3, recovery_ticks: 1 };
  const model = await parse(report, { checkpoint: null });
  assert.equal(model.arms[0].recoveryRatio, null);
  assert.equal(model.arms[0].firstZeroTick, 10);
  assert.equal(model.arms[0].terminal.population, 3);
  const container = containerFixture();
  renderStudyReport(container, model);
  assert.match(container.textContent, /recovery percentage is undefined/);
  assert.ok(!container.textContent.includes("300%"));
});

test("oversized, malformed, duplicate-field and excessively nested reports fail before presentation", async () => {
  const oversized = new Blob([new Uint8Array(MAX_STUDY_REPORT_BYTES + 1)]);
  let read = false;
  Object.defineProperty(oversized, "arrayBuffer", { value: async () => { read = true; return new ArrayBuffer(0); } });
  await assert.rejects(parse(oversized), /1 MiB/);
  assert.equal(read, false);
  await assert.rejects(parse(new Blob([new Uint8Array([0xff])])), /UTF-8/);
  await assert.rejects(parse("{"), /JSON/);
  await assert.rejects(parse(JSON.stringify(reportFixture()) + "garbage"), /unexpected data/);
  await assert.rejects(parse(JSON.stringify(reportFixture()).replace('{"study_id":', '{"study_id":"duplicate","study_id":')), /duplicate JSON field/);
  await assert.rejects(parse("[".repeat(30) + "0" + "]".repeat(30)), /structure exceeds/);
  await assert.rejects(parse(JSON.stringify(Array(8193).fill(0))), /structure exceeds/);
  await assert.rejects(parse('{"__proto__":{"polluted":true}}'), /missing or unsupported fields/);
  assert.equal({}.polluted, undefined);
});

class Element {
  constructor(tag, ownerDocument) { this.tagName = tag; this.ownerDocument = ownerDocument; this.children = []; this.value = ""; this.hidden = false; }
  set textContent(value) { this.value = String(value); this.children = []; }
  get textContent() { return this.value + this.children.map((child) => child.textContent).join(" "); }
  set innerHTML(_) { throw new Error("Report renderer must not execute markup"); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.value = ""; this.children = [...children]; }
}
function containerFixture() {
  const doc = { createElement(tag) { return new Element(tag, doc); } };
  return new Element("section", doc);
}
function descendants(node) { return [node, ...node.children.flatMap(descendants)]; }

test("report rendering uses safe text, measured endpoint labels and exact provenance", async () => {
  const report = reportFixture();
  report.summary_markdown = "<img src=x onerror=alert(1)>";
  report.branch_reports[1].first_divergence.stage = "<script>untrusted lane</script>";
  const model = await parse(report), container = containerFixture();
  renderStudyReport(container, model);
  const nodes = descendants(container), text = container.textContent;
  assert.equal(container.hidden, false);
  assert.equal(nodes.filter((node) => node.tagName === "table").length, 2);
  assert.ok(nodes.every((node) => !["img", "script", "a"].includes(node.tagName)));
  assert.match(text, /Δ treatment − control/);
  assert.match(text, /Births in final tick/);
  assert.match(text, /one starting seed/);
  assert.match(text, /independent-seed confidence intervals and significance tests are unavailable/);
  assert.ok(text.includes(model.worldDigest));
  assert.ok(text.includes(model.artifact.checksumSha256));
  assert.ok(text.includes("<script>untrusted lane</script>"));
  assert.ok(!text.includes("onerror"));
  renderStudyReport(container, null);
  assert.equal(container.hidden, true);
  assert.equal(container.children.length, 0);
});

test("control-only presentation omits the treatment column", async () => {
  const report = reportFixture(); report.branch_reports.length = 1;
  const container = containerFixture();
  renderStudyReport(container, await parse(report));
  assert.equal(descendants(container).filter((node) => node.tagName === "table").length, 1);
  assert.ok(!container.textContent.includes("Δ treatment"));
  assert.match(container.textContent, /one unchanged future/);
});
