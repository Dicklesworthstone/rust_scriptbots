import { makeInterventionRequest, validateInterventionAction, validateInterventionReceipt } from "./interventions.js";
import { exactIdentity } from "./inspector.js";

export const MAX_STUDY_TICKS = 512;
const MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024;
const FILE_SCHEMA = "scriptbots.local-paired-study-file.v1";
const FILE_MAGIC = new TextEncoder().encode("SBSTUDY1");
const FILE_HEADER_BYTES = 44;
const MAX_METADATA_BYTES = 4096;
export const MAX_LOCAL_STUDY_FILE_BYTES = FILE_HEADER_BYTES + MAX_METADATA_BYTES + MAX_CHECKPOINT_BYTES;
const revisions = ["sessionId", "tick", "scientificRevision", "configRevision", "controlRevision"];
const sourceFields = [...revisions, "publicationRevision", "epoch"];
const metricFields = ["population", "averageEnergy", "averageHealth"];

function shape(value, keys) {
  if (!value || typeof value !== "object" || Array.isArray(value)
      || Object.keys(value).length !== keys.length || keys.some(key => !Object.hasOwn(value, key))) {
    throw new Error("The study contains missing or unsupported fields");
  }
}

function freeze(value) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

function canonical(value) {
  if (value && typeof value === "object") {
    return "{" + Object.keys(value).sort().map(key => JSON.stringify(key) + ":" + canonical(value[key])).join(",") + "}";
  }
  return JSON.stringify(value);
}

function plan({ source, ticks, delayTicks, action }) {
  checkTiming(ticks, delayTicks);
  shape(source, sourceFields);
  for (const key of sourceFields) exactIdentity(source[key], key === "sessionId" ? 128 : 64);
  if (BigInt(source.tick) + BigInt(ticks) > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error("Study would exceed the exact browser tick range");
  }
  validateInterventionAction(action);
  return freeze(structuredClone({ source, ticks, delayTicks, action }));
}

function checkpointLimit(codecLimit) {
  if (!Number.isSafeInteger(codecLimit) || codecLimit <= 0) throw new Error("The study checkpoint limit is invalid");
  return Math.min(codecLimit, MAX_CHECKPOINT_BYTES);
}

/** Shared capture, file and worker admission. Metadata is detached before any asynchronous work. */
export function validateLocalStudyJob(job, codecLimit = MAX_CHECKPOINT_BYTES) {
  shape(job, ["bytes", "source", "ticks", "delayTicks", "action"]);
  const limit = checkpointLimit(codecLimit);
  if (!(job.bytes instanceof Uint8Array) || !job.bytes.byteLength || job.bytes.byteLength > limit) {
    throw new Error("The study needs a valid local continuation of at most 64 MiB");
  }
  return { ...plan(job), bytes: job.bytes };
}

function active(signal) {
  if (signal?.aborted) throw new Error("Study file operation was cancelled");
}

async function sha256(bytes, signal) {
  active(signal);
  if (!globalThis.crypto?.subtle) throw new Error("Portable study files require Web Crypto on a secure connection");
  const digest = new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", bytes));
  active(signal);
  return digest;
}

const hex = bytes => Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");

/** The metadata hash binds the entire canonical plan, including the checkpoint's SHA-256.
 * Original checkpoint bytes are embedded unchanged; this is integrity, not a replay attestation.
 */
export async function createLocalStudyFile(input, codecLimit, { signal } = {}) {
  active(signal);
  const job = validateLocalStudyJob(input, codecLimit);
  // Snapshot the exact view before the first await, including nonzero byte offsets.
  const payload = new Blob([job.bytes]);
  const bytes = new Uint8Array(await payload.arrayBuffer());
  const checkpoint = freeze({ bytes: bytes.length, sha256: hex(await sha256(bytes, signal)) });
  const metadata = new TextEncoder().encode(canonical({ schema: FILE_SCHEMA,
    source: job.source, ticks: job.ticks, delayTicks: job.delayTicks, action: job.action, checkpoint }));
  if (metadata.length > MAX_METADATA_BYTES) throw new Error("Study metadata exceeds the file limit");
  const header = new Uint8Array(FILE_HEADER_BYTES);
  header.set(FILE_MAGIC);
  new DataView(header.buffer).setUint32(8, metadata.length, true);
  header.set(await sha256(metadata, signal), 12);
  return { job: { ...job, bytes }, checkpoint,
    file: new Blob([header, metadata, payload], { type: "application/octet-stream" }) };
}

/** Verify bounded sections before allocating checkpoint bytes or invoking any owner decoder. */
export async function readLocalStudyFile(file, codecLimit, { signal } = {}) {
  active(signal);
  const limit = checkpointLimit(codecLimit);
  if (!(file instanceof Blob) || file.size <= FILE_HEADER_BYTES
      || file.size > FILE_HEADER_BYTES + MAX_METADATA_BYTES + limit) {
    throw new Error("Study file is empty or exceeds the bounded continuation size");
  }
  const header = new Uint8Array(await file.slice(0, FILE_HEADER_BYTES).arrayBuffer());
  active(signal);
  if (FILE_MAGIC.some((byte, index) => header[index] !== byte)) throw new Error("Unsupported study file version");
  const length = new DataView(header.buffer, header.byteOffset, header.byteLength).getUint32(8, true);
  if (length < 1 || length > MAX_METADATA_BYTES || FILE_HEADER_BYTES + length >= file.size) {
    throw new Error("Study file has an invalid metadata length");
  }
  const metadata = new Uint8Array(await file.slice(FILE_HEADER_BYTES, FILE_HEADER_BYTES + length).arrayBuffer());
  if (hex(await sha256(metadata, signal)) !== hex(header.subarray(12))) throw new Error("Study metadata checksum does not match");
  let text, decoded;
  try {
    text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(metadata);
    decoded = JSON.parse(text);
  } catch (_) { throw new Error("Study metadata is not valid UTF-8 JSON"); }
  shape(decoded, ["schema", "source", "ticks", "delayTicks", "action", "checkpoint"]);
  if (decoded.schema !== FILE_SCHEMA) throw new Error("Unsupported study metadata schema");
  const expected = plan(decoded);
  shape(decoded.checkpoint, ["bytes", "sha256"]);
  const checkpoint = decoded.checkpoint;
  if (!Number.isSafeInteger(checkpoint.bytes) || checkpoint.bytes < 1 || checkpoint.bytes > limit
      || typeof checkpoint.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(checkpoint.sha256)
      || file.size !== FILE_HEADER_BYTES + length + checkpoint.bytes) {
    throw new Error("Study checkpoint size or checksum declaration is invalid");
  }
  // This also refuses duplicate fields, alternate number spellings and hidden extra syntax.
  if (canonical({ schema: FILE_SCHEMA, ...expected, checkpoint }) !== text) {
    throw new Error("Study metadata is not canonical or contains duplicate fields");
  }
  active(signal);
  const bytes = new Uint8Array(await file.slice(FILE_HEADER_BYTES + length).arrayBuffer());
  if (hex(await sha256(bytes, signal)) !== checkpoint.sha256) throw new Error("Study checkpoint checksum does not match");
  return { job: validateLocalStudyJob({ ...expected, bytes }, limit), checkpoint: freeze(checkpoint), file };
}

function checkTiming(ticks, delayTicks) {
  if (!Number.isSafeInteger(ticks) || ticks < 1 || ticks > MAX_STUDY_TICKS) {
    throw new Error(`Choose a study horizon of 1 to ${MAX_STUDY_TICKS} ticks`);
  }
  if (!Number.isSafeInteger(delayTicks) || delayTicks < 0 || delayTicks >= ticks) {
    throw new Error("Intervention delay must be a whole number from zero to one less than the study horizon");
  }
}

function observation(owner) {
  const frame = owner.frame();
  const status = owner.ownerStatus();
  if (!frame.playback?.paused || status.health !== "healthy" || status.lifecycle !== "running"
      || revisions.some(key => status.identity?.[key] !== frame.identity[key])) {
    throw new Error("The study requires a healthy owner at an observed paused boundary");
  }
  const { tick, summary } = frame.snapshot;
  if (!Number.isSafeInteger(tick) || tick < 0 || String(tick) !== frame.identity.tick
      || !Number.isSafeInteger(summary.agentCount) || summary.agentCount < 0
      || !Number.isFinite(summary.averageEnergy) || !Number.isFinite(summary.averageHealth)) {
    throw new Error("The owner returned an inexact tick or invalid scientific summary");
  }
  return frame;
}

function metrics(frame) {
  const { agentCount, averageEnergy, averageHealth } = frame.snapshot.summary;
  return { population: agentCount, averageEnergy: agentCount ? averageEnergy : null,
    averageHealth: agentCount ? averageHealth : null };
}

function sameCut(frame, source) {
  return frame.snapshot.epoch === source.epoch && revisions.every(key => frame.identity[key] === source[key]);
}

/** Capture once, before dispatch. Capturing can settle a deferred journal receipt. */
export function captureLocalStudy(owner, action, ticks, codecLimit, delayTicks = 0) {
  checkTiming(ticks, delayTicks);
  validateInterventionAction(action);
  const before = observation(owner);
  if (!Number.isSafeInteger(before.snapshot.tick + ticks)) throw new Error("Study would exceed the exact browser tick range");
  const source = { ...before.identity, epoch: before.snapshot.epoch };
  const bytes = owner.checkpoint();
  if (!(bytes instanceof Uint8Array) || !bytes.length || !Number.isSafeInteger(codecLimit)
      || bytes.length > Math.min(codecLimit, MAX_CHECKPOINT_BYTES)) {
    throw new Error("The study needs a valid local continuation of at most 64 MiB");
  }
  const captured = observation(owner);
  if (!sameCut(captured, source)) throw new Error("The scientific source changed during capture");
  return validateLocalStudyJob({ bytes, source: { ...captured.identity, epoch: captured.snapshot.epoch }, ticks, delayTicks, action }, codecLimit);
}

function queueTreatment(owner, frame, action) {
  const request = makeInterventionRequest(frame, action);
  const expected = { ...request, identity: frame.identity };
  let receipt = validateInterventionReceipt(owner.intervene(request), expected);
  // Both restored owners use a paused clock at its origin for the entire study.
  if (receipt.journalState === "pending" || receipt.applicationState === "admitted") {
    owner.drive(0);
    receipt = validateInterventionReceipt(owner.interventionStatus(receipt.commandId, request.sessionId), expected, receipt);
  }
  if (receipt.ownerError || receipt.applicationState !== "applied" || receipt.applicationBoundary.tick !== frame.identity.tick
      || !["committed_volatile", "durable"].includes(receipt.journalState)) {
    throw new Error("The treatment was not confirmed at its requested scientific tick");
  }
  return receipt;
}

/** Called in a dedicated worker. Release the control owner before restoring the treatment. */
export function runLocalStudy(input, restore, onProgress = () => {}) {
  const { bytes, source, ticks, delayTicks, action } = validateLocalStudyJob(input);
  function runArm(name) {
    const owner = restore(bytes);
    try {
      const start = observation(owner);
      if (!sameCut(start, source)) throw new Error("Restored owner does not match the captured source");
      let receipt = null;
      const samples = [{ tick: start.identity.tick, ...metrics(start) }];
      // A delayed treatment adds a sample at the shared pretreatment boundary.
      // Reserving one interval keeps both trajectories within 33 observations.
      const stride = Math.ceil(ticks / (delayTicks === 0 ? 32 : 31));
      let end = start;
      for (let done = 0; done < ticks;) {
        if (name === "treatment" && done === delayTicks) receipt = queueTreatment(owner, end, action);
        const nextBoundary = done < delayTicks ? delayTicks : ticks;
        const batch = Math.min(stride, nextBoundary - done);
        owner.tick(batch);
        done += batch;
        end = observation(owner);
        if (end.identity.sessionId !== source.sessionId || end.snapshot.tick !== start.snapshot.tick + done) {
          throw new Error(`The ${name} arm did not reach its requested scientific tick`);
        }
        samples.push({ tick: end.identity.tick, ...metrics(end) });
        onProgress(name, done, ticks);
      }
      return { start: metrics(start), end: metrics(end), finalTick: end.identity.tick, receipt, samples };
    } finally {
      owner.free();
    }
  }
  const control = runArm("control");
  const treatment = runArm("treatment");
  const keys = Object.keys(control.end);
  const queueTick = BigInt(treatment.receipt.applicationBoundary.tick);
  for (const [i, sample] of control.samples.entries()) {
    if (BigInt(sample.tick) > queueTick) break;
    if (keys.some(key => sample[key] !== treatment.samples[i][key])) {
      throw new Error("The two unchanged trajectories disagree before treatment");
    }
  }
  const difference = {};
  for (const key of keys) {
    difference[key] = treatment.end[key] === null || control.end[key] === null
      ? null : treatment.end[key] - control.end[key];
  }
  return { schema: "scriptbots.local-paired-study.v1", source, ticks, delayTicks, action, control, treatment, difference };
}

/** Refuse a stale/malformed worker response before presenting it as the selected study. */
export function validateLocalStudyResult(result, expected) {
  shape(result, ["schema", "source", "ticks", "delayTicks", "action", "control", "treatment", "difference"]);
  const selected = plan(expected);
  if (result.schema !== "scriptbots.local-paired-study.v1" || canonical(plan(result)) !== canonical(selected)) {
    throw new Error("Study result differs from the selected source or intervention");
  }
  const start = Number(selected.source.tick), stride = Math.ceil(selected.ticks / (selected.delayTicks === 0 ? 32 : 31));
  const ticks = [start];
  for (let done = 0; done < selected.ticks;) {
    const nextBoundary = done < selected.delayTicks ? selected.delayTicks : selected.ticks;
    done += Math.min(stride, nextBoundary - done);
    ticks.push(start + done);
  }
  const validMetrics = value => {
    if (!Number.isSafeInteger(value.population) || value.population < 0
        || metricFields.slice(1).some(key => value.population === 0 ? value[key] !== null
          : typeof value[key] !== "number" || !Number.isFinite(value[key]))) throw new Error("Study result contains invalid measurements");
  };
  for (const name of ["control", "treatment"]) {
    const arm = result[name];
    shape(arm, ["start", "end", "finalTick", "receipt", "samples"]);
    for (const boundary of [arm.start, arm.end]) { shape(boundary, metricFields); validMetrics(boundary); }
    if (arm.finalTick !== String(start + selected.ticks) || !Array.isArray(arm.samples) || arm.samples.length !== ticks.length) {
      throw new Error("Study result does not contain the requested scientific horizon");
    }
    arm.samples.forEach((sample, index) => {
      shape(sample, ["tick", ...metricFields]);
      validMetrics(sample);
      if (sample.tick !== String(ticks[index])) throw new Error("Study observations do not match their scheduled ticks");
    });
    if (metricFields.some(key => arm.start[key] !== arm.samples[0][key] || arm.end[key] !== arm.samples.at(-1)[key])) {
      throw new Error("Study endpoint measurements disagree with the sampled series");
    }
  }
  if (result.control.receipt !== null) throw new Error("Control arm unexpectedly received an intervention");
  const receipt = validateInterventionReceipt(result.treatment.receipt,
    { sessionId: selected.source.sessionId, epoch: selected.source.epoch, identity: selected.source });
  const queueTick = BigInt(selected.source.tick) + BigInt(selected.delayTicks);
  if (receipt.ownerError || receipt.applicationState !== "applied" || receipt.applicationBoundary.tick !== String(queueTick)
      || !["committed_volatile", "durable"].includes(receipt.journalState)) throw new Error("Treatment was not confirmed at its requested scientific tick");
  for (const [index, sample] of result.control.samples.entries()) {
    if (BigInt(sample.tick) > queueTick) break;
    if (metricFields.some(key => sample[key] !== result.treatment.samples[index][key])) {
      throw new Error("Study comparison contains divergent pretreatment measurements");
    }
  }
  shape(result.difference, metricFields);
  for (const key of metricFields) {
    const left = result.control.end[key], right = result.treatment.end[key];
    const difference = left === null || right === null ? null : right - left;
    if (result.control.start[key] !== result.treatment.start[key] || result.difference[key] !== difference
        || (difference !== null && !Number.isFinite(difference))) throw new Error("Study comparison does not match its two independent arms");
  }
  return freeze(structuredClone(result));
}
