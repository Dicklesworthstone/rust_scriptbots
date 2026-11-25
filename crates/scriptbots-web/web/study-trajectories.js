// Retained branch summaries are descriptive observations, not restorable worlds.
// Call this reader only after the artifact transport verifies the downloaded bytes.
export const MAX_STUDY_TRAJECTORY_BYTES = 4 * 1024 * 1024;
const MAX_HORIZON_TICKS = 10000;
const HEADER = "tick,population,food_total,births,deaths,total_energy,average_health,spike_hits,digest";
const ID = /^[A-Za-z0-9_-]{1,128}$/;
const DIGEST = /^[0-9a-f]{16}$/;
const CHECKSUM = /^[0-9a-f]{64}$/;
const COUNT = /^(0|[1-9][0-9]*)$/;
const DECIMAL = /^(?:(?:0|[1-9][0-9]*)\.[0-9]{4}|-0\.0000)$/;
const COUNT_KEYS = ["population", "births", "deaths", "spike_hits"];
const FLOAT_KEYS = ["food_total", "total_energy", "average_health"];
const OBSERVATION_KEYS = [...COUNT_KEYS, ...FLOAT_KEYS];
const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const integer = (value) => Number.isSafeInteger(value) && value >= 0;
const finite = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
const matchesText = (value, expression) => typeof value === "string" && expression.test(value);
function refuse(message) { throw new Error("Study trajectories refused: " + message); }
function freeze(value) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

export const STUDY_TRAJECTORY_METRICS = freeze([
  { key: "population", label: "Population", integral: true },
  { key: "food_total", label: "Ground food", integral: false },
  { key: "total_energy", label: "Total energy", integral: false },
  { key: "average_health", label: "Average health", integral: false },
]);

/** Catalog paths are derived from managed identities, never report filesystem paths. */
export function studyTrajectorySources(report) {
  if (!object(report) || !matchesText(report.studyId, ID) || !Array.isArray(report.arms)
      || report.arms.length < 1 || report.arms.length > 2
      || report.arms.some((arm) => !object(arm) || !["control", "treatment"].includes(arm.id))
      || new Set(report.arms.map((arm) => arm.id)).size !== report.arms.length
      || !report.arms.some((arm) => arm.id === "control")) {
    refuse("a validated saved-study report is required");
  }
  return freeze(["control", "treatment"].filter((id) => report.arms.some((arm) => arm.id === id))
    .map((armId) => ({ armId, label: armId === "control" ? "Control" : "Treatment",
      relativePath: "branches/" + report.studyId + "/study/bundle_" + armId + "/exports/summary.csv" })));
}

function captureReport(report, sources) {
  if (!matchesText(report.checkpointId, ID) || !integer(report.sourceTick) || !integer(report.targetTick)
      || !integer(report.horizonTicks) || report.horizonTicks < 1 || report.horizonTicks > MAX_HORIZON_TICKS
      || report.targetTick - report.sourceTick !== report.horizonTicks || !matchesText(report.worldDigest, DIGEST)
      || report.hasTreatment !== (sources.length === 2)) refuse("report source or horizon is invalid");
  const arms = sources.map((source) => {
    const arm = report.arms.find((value) => value.id === source.armId);
    const due = arm.interventionTick;
    const baselineTick = source.armId === "control" ? report.sourceTick : due;
    if (source.armId === "control" ? due !== null
      : !integer(due) || due < report.sourceTick || due >= report.targetTick) refuse("report intervention boundary is invalid");
    const terminal = arm.terminal;
    if (!object(terminal) || terminal.tick !== report.targetTick || !matchesText(terminal.digest, DIGEST)
        || COUNT_KEYS.some((key) => !integer(terminal[key]))
        || FLOAT_KEYS.some((key) => !finite(terminal[key]))
        || !Number.isFinite(Math.fround(terminal.total_energy)) || !Number.isFinite(Math.fround(terminal.average_health))
        || arm.baselineTick !== baselineTick || !integer(arm.baselinePopulation)
        || !integer(arm.minimumPopulation) || !integer(arm.troughTick)
        || arm.troughTick < baselineTick || arm.troughTick > report.targetTick
        || (arm.firstZeroTick !== null && (!integer(arm.firstZeroTick)
          || arm.firstZeroTick < baselineTick || arm.firstZeroTick > report.targetTick))) {
      refuse("report endpoint or population baseline is invalid");
    }
    const divergenceTick = arm.divergence === null ? null : arm.divergence?.tick;
    if (divergenceTick !== null && (source.armId === "control" || !integer(divergenceTick)
        || divergenceTick <= due || divergenceTick > report.targetTick)) refuse("report divergence boundary is invalid");
    return { id: source.armId, label: source.label, interventionTick: due, baselineTick,
      baselinePopulation: arm.baselinePopulation, minimumPopulation: arm.minimumPopulation,
      firstZeroTick: arm.firstZeroTick, troughTick: arm.troughTick, divergenceTick,
      terminal: { tick: terminal.tick, digest: terminal.digest,
        ...Object.fromEntries(OBSERVATION_KEYS.map((key) => [key, terminal[key]])) } };
  });
  return { studyId: report.studyId, checkpointId: report.checkpointId,
    sourceTick: report.sourceTick, targetTick: report.targetTick, horizonTicks: report.horizonTicks,
    worldDigest: report.worldDigest, hasTreatment: report.hasTreatment, arms };
}

function readCount(value) {
  if (!COUNT.test(value) || !integer(Number(value))) refuse("CSV contains an inexact count or tick");
  return Number(value);
}
function readDecimal(value) {
  if (!DECIMAL.test(value) || !finite(Number(value))) refuse("CSV requires finite nonnegative observations at four decimal places");
  return Number(value);
}
function endpointMatches(observed, terminal, key) {
  // serde serializes energy/health as f32; recovering that binary32 value avoids
  // confusing its shortest JSON spelling with the CSV's fixed four-place value.
  const expected = key === "food_total" ? terminal[key] : Math.fround(terminal[key]);
  const rounding = 0.00005 + 4 * Number.EPSILON * Math.max(1, Math.abs(expected));
  return Math.abs(observed - expected) <= rounding;
}

async function readArm(receipt, report) {
  const buffer = await receipt.blob.arrayBuffer();
  if (buffer.byteLength !== receipt.artifact.byteSize || buffer.byteLength > MAX_STUDY_TRAJECTORY_BYTES) {
    refuse("CSV byte count changed during reading");
  }
  let decoded;
  try { decoded = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(buffer); }
  catch (_) { refuse("CSV is not valid UTF-8"); }
  const lines = decoded.split("\n");
  if (lines.at(-1) === "") lines.pop();
  if (lines.length !== report.horizonTicks + 2) refuse("CSV must contain every observed tick exactly once");
  const line = (value) => value.endsWith("\r") ? value.slice(0, -1) : value;
  if (line(lines[0]) !== HEADER) refuse("CSV columns differ from the retained summary schema");
  const samples = lines.slice(1).map((text, index) => {
    if (text.length > 1024) refuse("CSV row exceeds its bounded numeric schema");
    const fields = line(text).split(",");
    if (fields.length !== 9) refuse("CSV row has missing or extra columns");
    const tick = readCount(fields[0]);
    if (tick !== report.sourceTick + index) refuse("CSV ticks are not contiguous from the saved source");
    if (!DIGEST.test(fields[8])) refuse("CSV scientific digest is invalid");
    return { tick, population: readCount(fields[1]), food_total: readDecimal(fields[2]),
      births: readCount(fields[3]), deaths: readCount(fields[4]), total_energy: readDecimal(fields[5]),
      average_health: readDecimal(fields[6]), spike_hits: readCount(fields[7]), digest: fields[8] };
  });
  const arm = receipt.arm, first = samples[0], last = samples.at(-1);
  if (first.digest !== report.worldDigest || first.population !== report.arms[0].baselinePopulation
      || first.births !== 0 || first.deaths !== 0 || first.spike_hits !== 0) refuse("CSV source observation differs from the saved report");
  if (last.digest !== arm.terminal.digest || COUNT_KEYS.some((key) => last[key] !== arm.terminal[key])
      || FLOAT_KEYS.some((key) => !endpointMatches(last[key], arm.terminal, key))) {
    refuse("CSV terminal observation differs from the report at its recorded precision");
  }
  const baselineIndex = arm.baselineTick - report.sourceTick;
  if (samples[baselineIndex].population !== arm.baselinePopulation) refuse("CSV treatment baseline population differs from the report");
  let minimum = arm.baselinePopulation, troughTick = arm.baselineTick, firstZeroTick = null;
  for (let index = baselineIndex; index < samples.length; index += 1) {
    const sample = samples[index];
    if (sample.population < minimum) { minimum = sample.population; troughTick = sample.tick; }
    if (sample.population === 0 && firstZeroTick === null) firstZeroTick = sample.tick;
  }
  if (minimum !== arm.minimumPopulation || troughTick !== arm.troughTick || firstZeroTick !== arm.firstZeroTick) {
    refuse("CSV population observations differ from the report's treatment interval");
  }
  return { id: arm.id, label: arm.label, artifact: receipt.artifact, samples };
}

/** Join verified catalog downloads to one parsed report without reconstructing simulation state.
 * Keep the caller's source-generation guard across this asynchronous operation.
 */
export async function parseSavedStudyTrajectories(report, downloads) {
  const sources = studyTrajectorySources(report), expected = captureReport(report, sources);
  if (!Array.isArray(downloads) || downloads.length !== sources.length) refuse("one summary is required for every reported arm");
  // Capture every receipt and expected observation before the first byte read.
  const receipts = sources.map((source, index) => {
    const matches = downloads.filter((download) => download?.armId === source.armId);
    if (matches.length !== 1) refuse("missing or duplicate branch summary");
    const { blob, metadata } = matches[0];
    if (!(blob instanceof Blob) || blob.size < 1 || blob.size > MAX_STUDY_TRAJECTORY_BYTES
        || !object(metadata) || !matchesText(metadata.artifact_id, ID) || metadata.byte_size !== blob.size
        || metadata.relative_path !== source.relativePath || !matchesText(metadata.checksum_sha256, CHECKSUM)) {
      refuse("summary download differs from its bounded catalog receipt");
    }
    return { blob, arm: expected.arms[index], artifact: { id: metadata.artifact_id,
      relativePath: metadata.relative_path, checksumSha256: metadata.checksum_sha256, byteSize: metadata.byte_size } };
  });
  if (new Set(receipts.map((receipt) => receipt.artifact.id)).size !== receipts.length) refuse("branch summaries reuse an artifact identity");
  const arms = [];
  for (const receipt of receipts) arms.push(await readArm(receipt, expected));
  const control = arms[0], treatment = arms[1], treatmentReport = expected.arms[1];
  if (treatment) {
    let firstDivergenceTick = null;
    for (let index = 0; index < control.samples.length; index += 1) {
      const left = control.samples[index], right = treatment.samples[index];
      const sameDigest = left.digest === right.digest;
      const sameObservations = OBSERVATION_KEYS.every((key) => left[key] === right[key]);
      if (left.tick <= treatmentReport.interventionTick && (!sameDigest || !sameObservations)) {
        refuse("paired CSV observations disagree before their scientific divergence");
      }
      // Per-tick event counts are not digest lanes. Reconverged scientific state
      // does not establish identical counts after the intervention.
      if (!sameDigest && firstDivergenceTick === null) firstDivergenceTick = left.tick;
    }
    if (firstDivergenceTick !== treatmentReport.divergenceTick) refuse("CSV first divergence differs from the report");
  }
  return freeze({ studyId: expected.studyId, checkpointId: expected.checkpointId,
    sourceTick: expected.sourceTick, targetTick: expected.targetTick, horizonTicks: expected.horizonTicks,
    interventionTick: treatmentReport?.interventionTick ?? null,
    hasTreatment: expected.hasTreatment, decimalPlaces: 4, arms });
}
