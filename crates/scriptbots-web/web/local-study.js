import { makeInterventionRequest, validateInterventionAction, validateInterventionReceipt } from "./interventions.js";

export const MAX_STUDY_TICKS = 512;
const MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024;
const revisions = ["sessionId", "tick", "scientificRevision", "configRevision", "controlRevision"];

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
  return { bytes, source: { ...captured.identity, epoch: captured.snapshot.epoch }, ticks, delayTicks, action };
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
export function runLocalStudy({ bytes, source, ticks, action, delayTicks = 0 }, restore, onProgress = () => {}) {
  checkTiming(ticks, delayTicks);
  validateInterventionAction(action);
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
