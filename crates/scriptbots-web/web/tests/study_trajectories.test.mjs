import assert from "node:assert/strict";
import test from "node:test";
import {
  MAX_STUDY_TRAJECTORY_BYTES, STUDY_TRAJECTORY_METRICS,
  parseSavedStudyTrajectories, studyTrajectorySources,
} from "../study-trajectories.js";

const HEADER = "tick,population,food_total,births,deaths,total_energy,average_health,spike_hits,digest";
const digest = (letter) => letter.repeat(16);
const checksum = "a".repeat(64);
const sample = (tick, changes = {}) => ({ tick, population: 3, food_total: 10,
  births: 0, deaths: 0, total_energy: 3, average_health: 1, spike_hits: 0,
  digest: digest("a"), ...changes });

// Contract observations, not real simulation evidence. Treatment starts after tick 12.
function fixture() {
  const control = [
    sample(10),
    sample(11, { population: 4, food_total: 11, births: 1, total_energy: 4, digest: digest("b") }),
    sample(12, { population: 4, food_total: 12, total_energy: 4, digest: digest("c") }),
    sample(13, { population: 4, food_total: 13, total_energy: 4, digest: digest("d") }),
  ];
  const treatment = control.map((row) => ({ ...row }));
  treatment[3] = sample(13, { population: 2, food_total: 6, deaths: 2,
    total_energy: 1.5, average_health: 0.75, spike_hits: 1, digest: digest("e") });
  return { report: {
    studyId: "branch-trajectory", checkpointId: "ckpt-source", sourceTick: 10, targetTick: 13,
    horizonTicks: 3, worldDigest: digest("a"), hasTreatment: true,
    arms: [
      { id: "control", label: "Control", interventionTick: null, baselineTick: 10,
        baselinePopulation: 3, minimumPopulation: 3, troughTick: 10, firstZeroTick: null,
        divergence: null, terminal: { ...control[3] } },
      { id: "treatment", label: "Treatment", interventionTick: 12, baselineTick: 12,
        baselinePopulation: 4, minimumPopulation: 2, troughTick: 13, firstZeroTick: null,
        divergence: { tick: 13 }, terminal: { ...treatment[3] } },
    ],
  }, rows: { control, treatment } };
}

function csv(rows) {
  return HEADER + "\n" + rows.map((row) => [
    row.tick, row.population, row.food_total.toFixed(4), row.births, row.deaths,
    row.total_energy.toFixed(4), row.average_health.toFixed(4), row.spike_hits, row.digest,
  ].join(",")).join("\n") + "\n";
}
function download(report, armId, value) {
  const blob = value instanceof Blob ? value : new Blob([value]);
  return { armId, blob, metadata: { artifact_id: "summary-" + armId, byte_size: blob.size,
    relative_path: "branches/" + report.studyId + "/study/bundle_" + armId + "/exports/summary.csv",
    checksum_sha256: checksum } };
}
function input(value = fixture()) {
  return { report: value.report, downloads: value.report.arms.map((arm) =>
    download(value.report, arm.id, csv(value.rows[arm.id]))) };
}
async function parse(value = fixture()) {
  const { report, downloads } = input(value);
  return parseSavedStudyTrajectories(report, downloads);
}

test("retained summaries yield immutable exact ticks, observations and artifact provenance", async () => {
  const model = await parse();
  assert.equal(model.sourceTick, 10);
  assert.equal(model.targetTick, 13);
  assert.equal(model.interventionTick, 12);
  assert.equal(model.decimalPlaces, 4);
  assert.equal(model.hasTreatment, true);
  assert.deepEqual(model.arms.map((arm) => arm.id), ["control", "treatment"]);
  assert.deepEqual(model.arms[0].samples.map((row) => row.tick), [10, 11, 12, 13]);
  assert.equal(model.arms[1].samples[3].population - model.arms[0].samples[3].population, -2);
  assert.equal(model.arms[0].samples[1].births, 1, "births are per-tick observations");
  assert.equal(model.arms[0].samples[2].births, 0, "counts are not accumulated");
  assert.equal(model.arms[0].artifact.checksumSha256, checksum);
  assert.equal(model.arms[0].artifact.relativePath, "branches/branch-trajectory/study/bundle_control/exports/summary.csv");
  assert.ok(Object.isFrozen(model.arms[0].samples[0]));
  assert.ok(Object.isFrozen(model.arms[1].artifact));
  assert.throws(() => { model.arms[0].samples[0].population = 99; }, TypeError);
  assert.deepEqual(STUDY_TRAJECTORY_METRICS.map((metric) => metric.key),
    ["population", "food_total", "total_energy", "average_health"]);
});

test("catalog discovery derives safe paths and normalizes arm order without trusting bundle paths", async () => {
  const value = fixture();
  value.report.arms.reverse();
  value.report.arms[0].bundle_path = "https://unrelated.example/data.csv";
  assert.deepEqual(studyTrajectorySources(value.report).map((source) => source.armId), ["control", "treatment"]);
  const prepared = input(value);
  assert.deepEqual((await parseSavedStudyTrajectories(prepared.report, prepared.downloads)).arms.map((arm) => arm.id),
    ["control", "treatment"]);
  for (const studyId of [undefined, 123, "../escape", "branch/other", ""]) {
    assert.throws(() => studyTrajectorySources({ ...value.report, studyId }), /validated saved-study/);
  }
  assert.throws(() => studyTrajectorySources({ ...value.report, arms: [value.report.arms[0]] }), /validated saved-study/);
  assert.throws(() => studyTrajectorySources({ ...value.report, arms: [value.report.arms[1], value.report.arms[1]] }), /validated saved-study/);
});

test("control-only and unchanged delayed treatments retain observed trajectories without invented effects", async () => {
  const controlOnly = fixture();
  controlOnly.report.arms.length = 1;
  controlOnly.report.hasTreatment = false;
  const control = await parse(controlOnly);
  assert.equal(control.hasTreatment, false);
  assert.equal(control.interventionTick, null);
  assert.equal(control.arms.length, 1);

  const unchanged = fixture(), arm = unchanged.report.arms[1];
  unchanged.rows.treatment = unchanged.rows.control.map((row) => ({ ...row }));
  arm.terminal = { ...unchanged.rows.control[3] };
  arm.divergence = null;
  arm.minimumPopulation = 4;
  arm.troughTick = 12;
  const model = await parse(unchanged);
  assert.equal(model.arms[1].samples[0].population, 3);
  assert.equal(model.arms[1].samples[2].population, 4);
  assert.equal(model.arms[1].samples[3].population, 4);
});

test("CSV precision matches f64 food and shortest-serialized f32 energy and health", async () => {
  const value = fixture();
  value.report.arms.length = 1;
  value.report.hasTreatment = false;
  const terminal = value.report.arms[0].terminal;
  terminal.food_total = 10.123456789;
  terminal.total_energy = 123456.79;
  terminal.average_health = 0.03125;
  const last = value.rows.control[3];
  last.food_total = 10.1235;
  last.total_energy = 123456.7891;
  last.average_health = 0.0312; // Rust's half-even fixed formatting.
  assert.equal((await parse(value)).arms[0].samples[3].total_energy, 123456.7891);
  last.food_total = 10.1236;
  await assert.rejects(parse(value), /recorded precision/);
  last.food_total = 10.1235;
  last.total_energy = 123456.7892;
  await assert.rejects(parse(value), /recorded precision/);
});

test("source and terminal observations are bound to report digests and metrics", async () => {
  for (const change of [
    (value) => { value.rows.control[0].digest = digest("f"); },
    (value) => { value.rows.control[0].population = 4; },
    (value) => { value.rows.control[0].births = 1; },
    (value) => { value.rows.treatment[3].digest = digest("f"); },
    (value) => { value.rows.treatment[3].deaths = 1; },
    (value) => { value.rows.treatment[3].population = 3; },
    (value) => { value.rows.treatment[3].average_health = 0.5; },
  ]) {
    const value = fixture(); change(value);
    await assert.rejects(parse(value), /source observation|terminal observation/);
  }
});

test("observed population trough and extinction use the actual treatment baseline", async () => {
  const value = fixture();
  value.rows.control[1].population = 1;
  value.rows.treatment[1].population = 1;
  value.report.arms[0].minimumPopulation = 1;
  value.report.arms[0].troughTick = 11;
  assert.equal((await parse(value)).arms[1].samples[2].population, 4);
  for (const change of [
    (report) => { report.arms[1].baselinePopulation = 3; },
    (report) => { report.arms[1].minimumPopulation = 1; },
    (report) => { report.arms[1].troughTick = 12; },
    (report) => { report.arms[1].firstZeroTick = 13; },
  ]) {
    const next = fixture(); change(next.report);
    await assert.rejects(parse(next), /baseline population|population observations/);
  }
});

test("paired observations cannot diverge before the command or substitute reported first divergence", async () => {
  for (const key of ["digest", "food_total"]) {
    const value = fixture();
    value.rows.treatment[1][key] = key === "digest" ? digest("f") : 30;
    await assert.rejects(parse(value), /before their scientific divergence/);
  }
  const early = fixture(), arm = early.report.arms[1];
  arm.interventionTick = 10;
  arm.baselineTick = 10;
  arm.baselinePopulation = 3;
  early.rows.treatment[1].digest = digest("f");
  await assert.rejects(parse(early), /first divergence differs/);
  const absent = fixture();
  absent.report.arms[1].divergence = null;
  await assert.rejects(parse(absent), /first divergence differs/);
});

test("equal scientific digests after treatment do not invent equality of per-tick event counts", async () => {
  const value = fixture(), arm = value.report.arms[1];
  value.rows.treatment = value.rows.control.map((row) => ({ ...row }));
  value.rows.treatment[1].digest = digest("f");
  value.rows.treatment[3].spike_hits = 1;
  Object.assign(arm, { interventionTick: 10, baselineTick: 10, baselinePopulation: 3,
    minimumPopulation: 3, troughTick: 10, divergence: { tick: 11 },
    terminal: { ...value.rows.treatment[3] } });
  const model = await parse(value);
  assert.equal(model.arms[0].samples[3].digest, model.arms[1].samples[3].digest);
  assert.equal(model.arms[0].samples[3].spike_hits, 0);
  assert.equal(model.arms[1].samples[3].spike_hits, 1);
});

test("every recorded tick and exactly nine numeric columns are required", async (t) => {
  const cases = [
    ["missing tick", (lines) => { lines.splice(2, 1); }],
    ["extra tick", (lines) => { lines.push(lines.at(-1)); }],
    ["duplicate tick", (lines) => { lines[2] = lines[1]; }],
    ["wrong tick", (lines) => { lines[2] = lines[2].replace(/^11,/, "15,"); }],
    ["blank line", (lines) => { lines[2] = ""; }],
    ["extra column", (lines) => { lines[2] += ",0"; }],
    ["missing column", (lines) => { lines[2] = lines[2].slice(0, lines[2].lastIndexOf(",")); }],
    ["wrong header", (lines) => { lines[0] = lines[0].replace("digest", "overall"); }],
    ["quoted count", (lines) => { lines[2] = lines[2].replace("11,4,", '11,"4",'); }],
    ["fractional count", (lines) => { lines[2] = lines[2].replace("11,4,", "11,4.5,"); }],
    ["unsafe count", (lines) => { lines[2] = lines[2].replace("11,4,", "11,9007199254740992,"); }],
    ["noncanonical count", (lines) => { lines[2] = lines[2].replace("11,4,", "11,04,"); }],
    ["negative float", (lines) => { lines[2] = lines[2].replace("11.0000", "-1.0000"); }],
    ["nonfinite float", (lines) => { lines[2] = lines[2].replace("11.0000", "Infinity"); }],
    ["extra precision", (lines) => { lines[2] = lines[2].replace("11.0000", "11.00000"); }],
    ["row bound", (lines) => { lines[2] = "0".repeat(1025); }],
    ["uppercase digest", (lines) => { lines[2] = lines[2].replace(digest("b"), digest("B")); }],
  ];
  for (const [label, change] of cases) await t.test(label, async () => {
    const value = fixture(), prepared = input(value);
    const lines = csv(value.rows.control).trimEnd().split("\n");
    change(lines);
    prepared.downloads[0] = download(value.report, "control", lines.join("\n") + "\n");
    await assert.rejects(parseSavedStudyTrajectories(prepared.report, prepared.downloads), /Study trajectories refused/);
  });
  const value = fixture(), prepared = input(value);
  prepared.downloads[0] = download(value.report, "control", csv(value.rows.control).replaceAll("\n", "\r\n"));
  assert.equal((await parseSavedStudyTrajectories(prepared.report, prepared.downloads)).arms[0].samples.length, 4);
  prepared.downloads[0] = download(value.report, "control", csv(value.rows.control) + "\n");
  await assert.rejects(parseSavedStudyTrajectories(prepared.report, prepared.downloads), /exactly once/);
});

test("summary receipts cannot substitute paths, arms, sizes or artifact identities", async () => {
  for (const change of [
    (p) => { p.downloads[0].metadata.relative_path = "branches/another/study/bundle_control/exports/summary.csv"; },
    (p) => { p.downloads[0].metadata.relative_path = "branches/branch-trajectory/study/bundle_treatment/exports/summary.csv"; },
    (p) => { p.downloads[0].metadata.byte_size += 1; },
    (p) => { p.downloads[0].metadata.checksum_sha256 = "bad"; },
    (p) => { p.downloads[0].metadata.artifact_id = undefined; },
    (p) => { p.downloads[1].metadata.artifact_id = p.downloads[0].metadata.artifact_id; },
    (p) => { p.downloads[1].armId = "control"; },
    (p) => { p.downloads.pop(); },
    (p) => { p.downloads.push(p.downloads[0]); },
  ]) {
    const prepared = input(); change(prepared);
    await assert.rejects(parseSavedStudyTrajectories(prepared.report, prepared.downloads), /Study trajectories refused/);
  }
});

test("byte, UTF-8 and horizon limits refuse before unbounded parsing", async () => {
  const oversized = new Blob([new Uint8Array(MAX_STUDY_TRAJECTORY_BYTES + 1)]);
  let read = false;
  Object.defineProperty(oversized, "arrayBuffer", { value: async () => { read = true; return new ArrayBuffer(0); } });
  const prepared = input();
  prepared.downloads[0] = download(prepared.report, "control", oversized);
  await assert.rejects(parseSavedStudyTrajectories(prepared.report, prepared.downloads), /bounded catalog receipt/);
  assert.equal(read, false);
  for (const blob of [new Blob([new Uint8Array([0xff])]), new Blob(["\uFEFF" + csv(fixture().rows.control)])]) {
    const next = input();
    next.downloads[0] = download(next.report, "control", blob);
    await assert.rejects(parseSavedStudyTrajectories(next.report, next.downloads), /UTF-8|columns differ/);
  }
  const wrongBytes = input();
  Object.defineProperty(wrongBytes.downloads[0].blob, "arrayBuffer", { value: async () => new ArrayBuffer(1) });
  await assert.rejects(parseSavedStudyTrajectories(wrongBytes.report, wrongBytes.downloads), /byte count changed/);
  for (const horizonTicks of [0, 10001, 0.5, "3"]) {
    const next = input();
    next.report.horizonTicks = horizonTicks;
    await assert.rejects(parseSavedStudyTrajectories(next.report, next.downloads), /source or horizon/);
  }
});

test("maximum supported horizon retains all observations without sampling away inspection ticks", async () => {
  const rows = Array.from({ length: 10001 }, (_, tick) =>
    sample(tick, { digest: tick.toString(16).padStart(16, "0") }));
  const value = fixture();
  value.report.sourceTick = 0;
  value.report.targetTick = 10000;
  value.report.horizonTicks = 10000;
  value.report.worldDigest = rows[0].digest;
  value.report.hasTreatment = false;
  value.report.arms.length = 1;
  Object.assign(value.report.arms[0], { baselineTick: 0, troughTick: 0, terminal: { ...rows.at(-1) } });
  value.rows.control = rows;
  const model = await parse(value);
  assert.equal(model.arms[0].samples.length, 10001);
  assert.equal(model.arms[0].samples[4321].tick, 4321);
});

test("asynchronous byte reads retain the report and metadata captured at admission", async () => {
  const prepared = input(), blob = prepared.downloads[0].blob;
  const bytes = await blob.arrayBuffer();
  let release;
  Object.defineProperty(blob, "arrayBuffer", { value: () => new Promise((resolve) => { release = resolve; }) });
  const pending = parseSavedStudyTrajectories(prepared.report, prepared.downloads);
  assert.equal(typeof release, "function");
  prepared.report.worldDigest = digest("f");
  prepared.report.arms[1].terminal.population = 999;
  prepared.downloads[1].metadata.relative_path = "branches/another/summary.csv";
  release(bytes);
  const model = await pending;
  assert.equal(model.arms[1].samples[3].population, 2);
  assert.equal(model.arms[1].artifact.relativePath, "branches/branch-trajectory/study/bundle_treatment/exports/summary.csv");
});
