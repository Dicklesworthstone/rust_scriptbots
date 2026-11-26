import { makeInterventionRequest, validateInterventionAction, validateInterventionReceipt } from "./interventions.js";

export const MAX_STUDY_TICKS = 512;
const MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024;
const revisions = ["sessionId", "tick", "scientificRevision", "configRevision", "controlRevision"];

function checkHorizon(ticks) {
  if (!Number.isSafeInteger(ticks) || ticks < 1 || ticks > MAX_STUDY_TICKS) {
    throw new Error(`Choose a study horizon of 1 to ${MAX_STUDY_TICKS} ticks`);
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
export function captureLocalStudy(owner, action, ticks, codecLimit) {
  checkHorizon(ticks);
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
  return { bytes, source: { ...captured.identity, epoch: captured.snapshot.epoch }, ticks, action };
}

/** Called in a dedicated worker. Release the control owner before restoring the treatment. */
export function runLocalStudy({ bytes, source, ticks, action }, restore, onProgress = () => {}) {
  checkHorizon(ticks);
  validateInterventionAction(action);
  function runArm(name) {
    const owner = restore(bytes);
    try {
      const start = observation(owner);
      if (!sameCut(start, source)) throw new Error("Restored owner does not match the captured source");
      let receipt = null;
      if (name === "treatment") {
        const request = makeInterventionRequest(start, action);
        const expected = { ...request, identity: start.identity };
        receipt = validateInterventionReceipt(owner.intervene(request), expected);
        // Drain a deferred receipt at the restored clock origin while still paused.
        if (receipt.journalState === "pending" || receipt.applicationState === "admitted") {
          owner.drive(0);
          receipt = validateInterventionReceipt(owner.interventionStatus(receipt.commandId, request.sessionId), expected, receipt);
        }
        if (receipt.ownerError || receipt.applicationState !== "applied" || receipt.applicationBoundary.tick !== source.tick
            || !["committed_volatile", "durable"].includes(receipt.journalState)) {
          throw new Error("The treatment was not confirmed at the saved tick");
        }
      }
      let end;
      for (let done = 0; done < ticks;) {
        const batch = Math.min(16, ticks - done);
        owner.tick(batch);
        done += batch;
        end = observation(owner);
        if (end.identity.sessionId !== source.sessionId || end.snapshot.tick !== start.snapshot.tick + done) {
          throw new Error(`The ${name} arm did not reach its requested scientific tick`);
        }
        onProgress(name, done, ticks);
      }
      return { start: metrics(start), end: metrics(end), finalTick: end.identity.tick, receipt };
    } finally {
      owner.free();
    }
  }
  const control = runArm("control");
  const treatment = runArm("treatment");
  const difference = {};
  for (const key of Object.keys(control.end)) {
    if (control.start[key] !== treatment.start[key]) throw new Error("The two restored starting summaries disagree");
    difference[key] = treatment.end[key] === null || control.end[key] === null
      ? null : treatment.end[key] - control.end[key];
  }
  return { schema: "scriptbots.local-paired-study.v1", source, ticks, action, control, treatment, difference };
}
