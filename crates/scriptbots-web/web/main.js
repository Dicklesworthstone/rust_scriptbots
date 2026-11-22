import initWasm, * as wasm from "./pkg/scriptbots_web.js";
import { CanvasRenderer } from "./renderers/canvas.js";
import { WebGpuRenderer } from "./renderers/webgpu.js";
import { environmentDescription } from "./renderers/environment.js";
import { RemoteStreamClient } from "./remote.js";
import { BrowserContinuationStore, BrowserRecoveryController } from "./continuation-store.js";
import { InspectionController, pickAgent, renderInspector, renderInspectorSource,
  validateBrainInspection, renderBrainInspector, renderBrainSource } from "./inspector.js";
import { GenomeInspectionController, renderGenomeInspector, renderGenomeSource } from "./genome.js";
import { InterventionController, renderInterventions, validateInterventionAction } from "./interventions.js";
import { NarrativeController, renderNarrative, renderNarrativeSource } from "./narrative.js";
import { validateContinuationBrains, validateLaunchInfo, launchInfoText, renderBuildComparison } from "./recipes.js";
import { CheckpointLabController, renderCheckpointLab } from "./checkpoint-lab.js";
import { ExperimentController } from "./experiments.js";
import { readExperimentForm, renderExperimentLab, experimentObservationExport } from "./experiment-view.js";
import { renderStudyReport } from "./study-report.js";
import { WorldViewport, WorldDrag, MIN_WORLD_ZOOM, MAX_WORLD_ZOOM } from "./world-view.js";

const {
  init_sim,
  default_init_options,
  version,
  init_from_permalink,
  permalink_of,
  fork,
  permalink_diff,
  build_identity,
  check_build_match,
} = wasm;

let canvas = document.getElementById("sim-canvas");
const rendererSelect = document.getElementById("renderer");
const rendererStatus = document.getElementById("renderer-status");
const environmentLayer = document.getElementById("environment-layer");
const environmentStatus = document.getElementById("environment-status");
const worldViewport = new WorldViewport();
const worldDrag = new WorldDrag();
const worldViewStatus = document.getElementById("world-view-status");
const worldZoomIn = document.getElementById("world-zoom-in");
const worldZoomOut = document.getElementById("world-zoom-out");
const worldFit = document.getElementById("world-fit");
const worldFollow = document.getElementById("world-follow");
const captureButton = document.getElementById("capture-btn");
const captureStatus = document.getElementById("capture-status");
let captureRequest = null;
let encodingRequest = null;
let captureUrl = null;
let renderer = null;
let rendererGeneration = 0;
let lastPaintedFrame = null;
let lastPaintedLiveFrame = null;
let presentationGeneration = 0;
let reviewedCheckpoint = null;
const checkpointView = document.getElementById("checkpoint-view");
const checkpointViewStatus = document.getElementById("checkpoint-view-status");
const checkpointReturnLive = document.getElementById("checkpoint-return-live");
let presentationFailure = null;
const fpsEl = document.getElementById("metric-fps");
const tpsEl = document.getElementById("metric-tps");
const tickEl = document.getElementById("metric-tick");
const popEl = document.getElementById("metric-population");
const birthsEl = document.getElementById("metric-births");
const deathsEl = document.getElementById("metric-deaths");
const energyEl = document.getElementById("metric-energy");
const healthEl = document.getElementById("metric-health");
const resetButton = document.getElementById("reset-btn");
const brainPresetSelect = document.getElementById("brain-preset");
const checkpointSaveButton = document.getElementById("checkpoint-save");
const checkpointLoadInput = document.getElementById("checkpoint-load");
const checkpointStatus = document.getElementById("checkpoint-status");
const checkpointCapability = document.getElementById("checkpoint-capability");
const worldCheckpointExport = document.getElementById("world-checkpoint-export");
const worldCheckpointImport = document.getElementById("world-checkpoint-import");
const worldCheckpointStatus = document.getElementById("world-checkpoint-status");
let activeBrainPreset = null;
let activeBrainFamilies = [];
let checkpointLoadPending = false;
const browserSaveButton = document.getElementById("checkpoint-browser-save");
const browserRestoreButton = document.getElementById("checkpoint-browser-restore");
const browserCheckpointStatus = document.getElementById("checkpoint-browser-status");
let browserContinuationStore = null;
let browserStorePending = false;
const recoveryEnabled = document.getElementById("checkpoint-recovery-enabled");
const recoveryRestoreButton = document.getElementById("checkpoint-recovery-restore");
const recoveryRetryButton = document.getElementById("checkpoint-recovery-retry");
const recoveryStatus = document.getElementById("checkpoint-recovery-status");
let browserRecovery = null;
let recoveryRecord = null;
let recoveryReadStatus = "Checking for a recovery checkpoint…";
const stepsSlider = document.getElementById("steps-per-frame");
const populationSlider = document.getElementById("population");
const logView = document.getElementById("log");
const versionEl = document.getElementById("version");
const remoteUrl = document.getElementById("remote-url");
const remoteConnect = document.getElementById("remote-connect");
const remoteReconnect = document.getElementById("remote-reconnect");
const remoteLocal = document.getElementById("remote-local");
const remoteStatus = document.getElementById("remote-status");
const remotePlayback = document.getElementById("remote-playback");
const remoteReceipt = document.getElementById("remote-receipt");
const remoteOverlay = document.getElementById("remote-overlay");
const remotePause = document.getElementById("remote-pause");
const remoteResume = document.getElementById("remote-resume");
const remoteStep = document.getElementById("remote-step");
const remoteRefresh = document.getElementById("remote-refresh");
const remoteRetry = document.getElementById("remote-retry");

const mismatchBannerEl = document.getElementById("mismatch-banner");
const permalinkInput = document.getElementById("permalink-input");
const loadPermalinkBtn = document.getElementById("load-permalink-btn");
const copyPermalinkBtn = document.getElementById("copy-permalink-btn");
const permalinkStatus = document.getElementById("permalink-status");
const forkPatchInput = document.getElementById("fork-patch-input");
const forkBtn = document.getElementById("fork-btn");
const forkOutput = document.getElementById("fork-output");
const forkChildLink = document.getElementById("fork-child-link");
const launchInfoElement = document.getElementById("launch-info");
const forkRecipeInfo = document.getElementById("fork-recipe-info");
const loadForkedBtn = document.getElementById("load-forked-btn");
const copyForkBtn = document.getElementById("copy-fork-btn");
const diffContainer = document.getElementById("diff-container");
const diffEmpty = document.getElementById("diff-empty");
const diffTable = document.getElementById("diff-table");
const diffTbody = document.getElementById("diff-tbody");

let currentPermalink = null;
let latestForkedLink = null;
let activeLaunchInfo = null;

function updateRecipeControls() {
  const unavailable = remoteMode || !simHandle || !wasmInitialized || activeLaunchInfo?.shareable === false;
  for (const control of [copyPermalinkBtn, forkPatchInput, forkBtn]) if (control) control.disabled = unavailable;
  for (const control of [loadForkedBtn, copyForkBtn]) if (control) control.disabled = remoteMode || !latestForkedLink;
  if (launchInfoElement) {
    launchInfoElement.dataset.kind = remoteMode ? "remote" : activeLaunchInfo?.kind ?? "unknown";
    launchInfoElement.dataset.shareable = remoteMode ? "false" : String(activeLaunchInfo?.shareable ?? "unknown");
    launchInfoElement.textContent = remoteMode
      ? "Viewing a remote world. Local recipe details return with your local simulation."
      : launchInfoText(activeLaunchInfo);
  }
}

const metrics = {
  lastFrameTs: performance.now(),
  lastStatsTs: performance.now(),
  previousTick: null,
  fps: 0,
  tps: 0,
  tick: 0,
  statsTickAcc: 0,
};

const perfWindow = {
  frameAcc: 0,
  tickAcc: 0,
  lastLog: performance.now(),
};

let simHandle = null;
let wasmInitialized = false;
let simulationGeneration = 0;
let speedMultiplier = Number(stepsSlider.value);
let simulationStartedAt = null;
let simulationTimer = null;
let queuedReset = false;
let population = Number(populationSlider.value);
let remoteMode = false;
let remoteClient = null;
let remoteFrame = null;
let remoteConnected = false;
let lastRemoteRequestId = null;
let lastRemoteCommandId = null;
let lastRemoteStatusRequestId = null;

const experimentPanel = document.getElementById("experiment-lab");
const experimentForm = document.getElementById("experiment-form");
const experimentFormError = document.getElementById("experiment-form-error");
let experimentExportUrl = null;
const experiments = new ExperimentController({
  list: (request, signal) => remoteClient.listExperiments({ ...request, signal }),
  create: (request, signal) => remoteClient.createExperiment(request, { signal }),
  status: (request, signal) => remoteClient.experimentStatus(request.experimentId, { sessionId: request.sessionId, signal }),
  cancel: (request, signal) => remoteClient.cancelExperiment(request.experimentId, { sessionId: request.sessionId, signal }),
  resume: (request, signal) => remoteClient.resumeExperiment(request.experimentId, { sessionId: request.sessionId, signal }),
  retryFailed: (request, signal) => remoteClient.retryFailedExperiment(request.experimentId, { ...request, signal }),
  compare: (request, signal) => remoteClient.compareExperiment(request.experimentId, { ...request, signal }),
  onChange: (controller) => renderExperimentLab(experimentPanel, controller),
});

function updateExperimentVisibility() {
  const observed = remoteFrame && lastPaintedFrame?.sessionId === remoteFrame.identity.sessionId
    && lastPaintedFrame.simulationGeneration === simulationGeneration;
  experiments.setEnabled(remoteMode && remoteConnected && !reviewedCheckpoint && !!observed && !!experimentPanel?.open && !document.hidden);
}

function observeExperiments(frame) {
  if (remoteMode && remoteConnected && remoteClient) {
    experiments.observe(frame, remoteClient.url);
    updateExperimentVisibility();
  }
}

async function experimentAction(action) {
  try {
    if (experimentFormError) experimentFormError.textContent = "";
    await action();
  } catch (error) {
    if (experimentFormError) experimentFormError.textContent = String(error.message || error).slice(0, 512);
  }
}

experimentPanel?.addEventListener("toggle", updateExperimentVisibility);
document.addEventListener("visibilitychange", updateExperimentVisibility);
experimentForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!experimentForm.reportValidity()) return;
  void experimentAction(() => experiments.launch(readExperimentForm(experimentForm)));
});
document.getElementById("experiment-select")?.addEventListener("change", (event) => {
  if (event.target.value) void experimentAction(() => experiments.open(event.target.value));
});
for (const [id, action] of [
  ["catalog-refresh", () => experiments.refreshCatalog(true)],
  ["catalog-first", () => experiments.firstPage()],
  ["catalog-next", () => experiments.nextPage()],
  ["retry", () => experiments.retryLaunch()],
  ["refresh", () => experiments.refreshStatus(true)],
  ["cancel", () => experiments.cancel()],
  ["resume", () => experiments.resume()],
  ["retry-failed", () => experiments.retryFailed()],
  ["run-previous", () => experiments.previousRuns()],
  ["run-next", () => experiments.nextRuns()],
  ["compare", () => experiments.compare({
    baselineVariant: document.getElementById("experiment-compare-baseline").value,
    challengerVariant: document.getElementById("experiment-compare-challenger").value,
    metric: document.getElementById("experiment-metric").value,
  })],
]) {
  document.getElementById("experiment-" + id)?.addEventListener("click", () => { void experimentAction(action); });
}
for (const id of ["compare-baseline", "compare-challenger", "metric"]) {
  document.getElementById("experiment-" + id)?.addEventListener("change", () => experiments.clearComparison());
}
document.getElementById("experiment-export")?.addEventListener("click", () => {
  const status = document.getElementById("experiment-export-status");
  try {
    const result = experimentObservationExport(experiments);
    if (experimentExportUrl) URL.revokeObjectURL(experimentExportUrl);
    experimentExportUrl = URL.createObjectURL(result.blob);
    const anchor = document.createElement("a");
    anchor.href = experimentExportUrl;
    anchor.download = result.filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    if (status) status.textContent = "Browser observation JSON is ready to save. Your browser manages the final file save; server run bundles remain separate.";
  } catch (error) {
    if (status) status.textContent = "Export unavailable: " + String(error.message || error).slice(0, 512);
  }
});
renderExperimentLab(experimentPanel, experiments);

const checkpointLabPanel = document.getElementById("checkpoint-lab");
const labForm = document.getElementById("lab-branch-form");
const labFormError = document.getElementById("lab-form-error");
const labDownloadStatus = document.getElementById("lab-download-status");
const labReportPanel = document.getElementById("lab-study-report");
let labDownloadUrl = null;
let labPollTimer = null;
const checkpointLab = new CheckpointLabController({
  list: (request, signal) => remoteClient.listCheckpoints({ ...request, signal }),
  listStudies: (request, signal) => remoteClient.listCheckpointBranches({ ...request, signal }),
  save: (request, signal) => remoteClient.saveCheckpoint(request, { signal }),
  review: (request, signal) => remoteClient.checkpointCanvas(request.checkpoint.checkpoint_id, {
    sessionId: request.sessionId, expected: request.checkpoint, signal,
  }),
  launch: (request, signal) => remoteClient.launchCheckpointBranch(request, { signal }),
  status: (request, signal) => remoteClient.checkpointBranchStatus(request.branchId, { sessionId: request.sessionId, signal }),
  cancel: (request, signal) => remoteClient.cancelCheckpointBranch(request.branchId, { sessionId: request.sessionId, signal }),
  artifacts: (request, signal) => remoteClient.checkpointBranchArtifacts(request.branchId, { ...request, signal }),
  download: (request, signal) => remoteClient.downloadArtifact(request.artifactId, {
    sessionId: request.sessionId, expected: request.expected, maxBytes: request.maxBytes, signal,
  }),
  onChange: (controller) => {
    renderCheckpointLab(checkpointLabPanel, controller);
    updateLocalTransferControl(controller);
    renderStudyReport(labReportPanel, controller.reportModel);
    synchronizeCheckpointView(controller);
  },
});

function updateLocalTransferControl(controller = checkpointLab) {
  const button = document.getElementById("lab-checkpoint-local");
  if (button) button.disabled = !controller.canOperate || !controller.selected || checkpointLoadPending
    || !remoteMode || !remoteConnected || !wasmInitialized || document.hidden
    || typeof wasm.import_world_checkpoint !== "function" || typeof wasm.world_checkpoint_byte_limit !== "function";
}

function updateCheckpointLabVisibility() {
  const observed = remoteFrame && lastPaintedLiveFrame?.sessionId === remoteFrame.identity.sessionId
    && lastPaintedLiveFrame.simulationGeneration === simulationGeneration;
  checkpointLab.setEnabled(remoteMode && remoteConnected && !!observed && !!checkpointLabPanel?.open && !document.hidden);
}

function observeCheckpointLab(frame) {
  if (remoteMode && remoteConnected && remoteClient) {
    checkpointLab.observe(frame, remoteClient.url);
    updateCheckpointLabVisibility();
  }
}

function renderCheckpointView() {
  const pending = checkpointLab.operation?.kind === "Saved-world review unavailable"
    && !checkpointLab.operation.abort.signal.aborted;
  if (checkpointView) checkpointView.hidden = !reviewedCheckpoint && !pending;
  if (checkpointReturnLive) checkpointReturnLive.textContent = reviewedCheckpoint ? "Return to live" : "Cancel saved-world review";
  if (!checkpointViewStatus) return;
  const checkpoint = reviewedCheckpoint?.checkpoint;
  const painted = checkpoint && lastPaintedFrame?.presentationGeneration === presentationGeneration;
  const message = checkpoint
    ? `${painted ? "Reviewing" : "Preparing"} saved checkpoint ${checkpoint.checkpoint_id} at tick ${checkpoint.tick}; world digest ${checkpoint.world_digest_v1}. This is a reconstructed scientific state. Live controls and creature inspection are unavailable here. ${remoteConnected ? "The server and its event timeline continue independently." : "The server is disconnected; this saved view is retained."}`
    : "Reconstructing the selected saved world. The current canvas stays visible until the saved state is ready.";
  if (checkpointViewStatus.textContent !== message) checkpointViewStatus.textContent = message;
  Object.assign(checkpointViewStatus.dataset, { state: checkpoint ? painted ? "ready" : "preparing" : "loading",
    checkpointId: checkpoint?.checkpoint_id ?? "", tick: checkpoint ? String(checkpoint.tick) : "",
    sourceSessionId: reviewedCheckpoint?.sourceSessionId ?? "", worldDigestV1: checkpoint?.world_digest_v1 ?? "" });
}

function synchronizeCheckpointView(controller) {
  if (reviewedCheckpoint !== controller.canvasPreview) {
    reviewedCheckpoint = controller.canvasPreview;
    // Restored publications and environment revisions are newly reconstructed.
    // They cannot share renderer caches or outstanding PNGs with the live owner.
    presentationGeneration += 1;
    lastPaintedFrame = null;
    if (!reviewedCheckpoint) lastPaintedLiveFrame = null;
    if (captureRequest) finishCapture(captureRequest, new Error("Displayed view changed"));
    inspector.reset();
    brainInspector.reset();
    genomeInspector.reset();
    updateInspectorVisibility();
    updateBrainVisibility();
    updateGenomeVisibility();
    for (const input of inspectorPanel?.querySelectorAll("input, button") ?? []) input.disabled = !!reviewedCheckpoint;
    if (reviewedCheckpoint) {
      for (const id of ["inspector-status", "brain-status", "genome-status"]) {
        const status = document.getElementById(id);
        if (status) status.textContent = "Saved-world review: return to live to inspect the current population.";
      }
    }
    document.getElementById("sim-container").dataset.view = reviewedCheckpoint ? "checkpoint" : "live";
    canvas.setAttribute("aria-label", reviewedCheckpoint
      ? "Saved checkpoint canvas. Drag to pan, scroll to zoom, Home fits the world. Return to live to inspect creatures."
      : "Simulation canvas. Click a creature to inspect; drag to pan; scroll or plus/minus to zoom; arrows pan; Home fits; F follows; Escape clears selection.");
    const metricsHeading = document.getElementById("metrics-heading");
    if (metricsHeading) metricsHeading.textContent = reviewedCheckpoint ? "Saved World Metrics" : "Live Metrics";
    resetPerformanceWindows(reviewedCheckpoint?.frame.snapshot ?? remoteFrame?.snapshot);
    updateRemoteControls();
  }
  renderCheckpointView();
}

function updateLabTreatmentFields() {
  const kind = document.getElementById("lab-branch-kind")?.value;
  for (const fields of labForm?.querySelectorAll("[data-lab-kinds]") ?? []) {
    const active = fields.dataset.labKinds.split(" ").includes(kind);
    fields.hidden = !active;
    fields.disabled = !active;
  }
  if (labFormError) labFormError.textContent = "";
}

function readLabTreatment() {
  const number = (id) => Number(document.getElementById(id).value);
  switch (document.getElementById("lab-branch-kind").value) {
    case "unchanged": return null;
    case "drought": return { type: "Drought", params: {
      region: null, duration_ticks: number("lab-drought-ticks"), growth_scale: number("lab-growth-scale"),
    } };
    case "resource_shock": return { type: "ResourceShock", params: {
      region: null, bloom_amount: number("lab-bloom"), scorch: number("lab-scorch"),
    } };
    case "temperature": return { type: "TemperatureShift", params: {
      discomfort_rate: number("lab-discomfort"), comfort_band: number("lab-comfort-band"),
    } };
    case "closed_world": return { type: "ClosedWorld", params: {
      closed: document.getElementById("lab-closed").value === "true",
    } };
    default: throw new Error("Choose a supported study treatment");
  }
}

async function offerLabDownload(operation) {
  try {
    const result = await operation();
    if (!result) return;
    if (!(result.blob instanceof Blob)) throw new Error("The server did not return a downloadable file");
    if (labDownloadUrl) URL.revokeObjectURL(labDownloadUrl);
    labDownloadUrl = URL.createObjectURL(result.blob);
    const anchor = document.createElement("a");
    anchor.href = labDownloadUrl;
    anchor.download = result.filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    if (labDownloadStatus) labDownloadStatus.textContent = `${result.filename} is ready to save (${result.blob.size.toLocaleString()} bytes). Your browser manages the final file save.`;
  } catch (error) {
    if (labDownloadStatus) labDownloadStatus.textContent = `Download unavailable: ${String(error.message || error).slice(0, 512)}`;
  }
}

checkpointReturnLive?.addEventListener("click", () => checkpointLab.returnToLive());
document.getElementById("lab-checkpoint-local")?.addEventListener("click", async () => {
  if (!remoteMode || !remoteConnected || !checkpointLab.canOperate || !checkpointLab.selected
      || checkpointLoadPending || !wasmInitialized || document.hidden
      || typeof wasm.import_world_checkpoint !== "function" || typeof wasm.world_checkpoint_byte_limit !== "function") return;
  const client = remoteClient;
  const sourceSessionId = checkpointLab.source.sessionId;
  const generation = simulationGeneration;
  const original = simHandle;
  checkpointLoadPending = true;
  updateCheckpointControls();
  try {
    const result = await checkpointLab.prepareLocalImport(wasm.world_checkpoint_byte_limit());
    if (!result) return;
    if (!remoteMode || !remoteConnected || document.hidden || client !== remoteClient
        || generation !== simulationGeneration || original !== simHandle
        || sourceSessionId !== remoteFrame?.identity.sessionId || sourceSessionId !== result.sourceSessionId
        || result.sourceToken !== remoteClient.url || !checkpointLab.enabled) {
      throw new Error("The displayed source changed during checkpoint transfer; select the saved world again");
    }
    // Restore and validate every candidate read before releasing the old local
    // owner or disconnecting the server. Rejection leaves both sources usable.
    const installed = installContinuation(result.bytes, String(result.checkpoint.tick), true);
    returnToLocalSimulation();
    worldCheckpointStatus.textContent = `Imported server checkpoint ${result.checkpoint.checkpoint_id} at tick ${installed.frame.identity.tick}; this new browser session is paused. The server continues independently; its commands, schedule and durable history are not transferred.`;
    document.getElementById("world-checkpoint-transfer").open = true;
  } catch (error) {
    checkpointLab.notice = `Local checkpoint transfer unavailable: ${String(error.message || error).slice(0, 1024)}`;
    checkpointLab.emit();
  } finally {
    checkpointLoadPending = false;
    updateCheckpointControls();
  }
});
checkpointLabPanel?.addEventListener("toggle", updateCheckpointLabVisibility);
document.addEventListener("visibilitychange", updateCheckpointLabVisibility);
document.getElementById("lab-branch-kind")?.addEventListener("change", updateLabTreatmentFields);
labForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!labForm.reportValidity()) return;
  try {
    labFormError.textContent = "";
    await checkpointLab.launch({ branchName: document.getElementById("lab-branch-name").value,
      ticks: Number(document.getElementById("lab-branch-ticks").value), intervention: readLabTreatment() });
  } catch (error) {
    labFormError.textContent = String(error.message || error).slice(0, 512);
  }
});
for (const [id, action] of [
  ["lab-history-refresh", () => checkpointLab.refreshHistory(true)],
  ["lab-history-first", () => checkpointLab.firstHistoryPage()],
  ["lab-history-next", () => checkpointLab.nextHistoryPage()],
  ["lab-checkpoint-save", () => checkpointLab.save()],
  ["lab-checkpoint-retry", () => checkpointLab.retrySave()],
  ["lab-checkpoint-review", () => checkpointLab.reviewSelected()],
  ["lab-checkpoint-refresh", () => checkpointLab.refresh(true)],
  ["lab-checkpoint-first", () => checkpointLab.firstPage()],
  ["lab-checkpoint-next", () => checkpointLab.nextPage()],
  ["lab-branch-retry", () => checkpointLab.retryLaunch()],
  ["lab-branch-refresh", () => checkpointLab.refreshBranch(true)],
  ["lab-branch-cancel", () => checkpointLab.cancelBranch()],
  ["lab-report-preview", () => checkpointLab.previewReport()],
  ["lab-artifact-refresh", () => checkpointLab.refreshArtifacts(true)],
  ["lab-artifact-first", () => checkpointLab.firstArtifactPage()],
  ["lab-artifact-next", () => checkpointLab.nextArtifactPage()],
]) {
  document.getElementById(id)?.addEventListener("click", async () => {
    try { await action(); }
    catch (error) { if (labFormError) labFormError.textContent = String(error.message || error).slice(0, 512); }
  });
}
document.getElementById("lab-checkpoint-select")?.addEventListener("change", (event) => checkpointLab.select(event.target.value));
document.getElementById("lab-history-select")?.addEventListener("change", async (event) => {
  try { await checkpointLab.selectStudy(event.target.value); }
  catch (error) { if (labFormError) labFormError.textContent = String(error.message || error).slice(0, 512); }
});
document.getElementById("lab-artifact-select")?.addEventListener("change", (event) => checkpointLab.selectArtifact(event.target.value));
document.getElementById("lab-checkpoint-download")?.addEventListener("click", () => { void offerLabDownload(() => checkpointLab.downloadCheckpoint()); });
document.getElementById("lab-report-download")?.addEventListener("click", () => { void offerLabDownload(() => checkpointLab.downloadReport()); });
document.getElementById("lab-artifact-download")?.addEventListener("click", () => { void offerLabDownload(() => checkpointLab.downloadSelectedArtifact()); });
function startLabPolling() {
  if (labPollTimer === null) labPollTimer = setInterval(() => { void checkpointLab.poll(); void experiments.poll(); }, 1000);
}
window.addEventListener("pageshow", () => { updateCheckpointLabVisibility(); updateExperimentVisibility(); startLabPolling(); });
updateLabTreatmentFields();
renderCheckpointLab(checkpointLabPanel, checkpointLab);
renderStudyReport(labReportPanel, checkpointLab.reportModel);
startLabPolling();

const narrativePanel = document.getElementById("narrative-timeline");
const narrative = new NarrativeController({
  read: (afterEvent, sessionId) => {
    if (!simHandle?.narrativePage) throw new Error("This WASM package does not expose scientific narrative history; rebuild the browser package");
    return simHandle.narrativePage(afterEvent, sessionId);
  },
  onChange: (controller) => renderNarrative(narrativePanel, controller),
  onFrame: (controller) => renderNarrativeSource(narrativePanel, controller),
});
const updateNarrativeVisibility = () => narrative.setEnabled(!document.hidden);
updateNarrativeVisibility();
document.addEventListener("visibilitychange", updateNarrativeVisibility);
document.getElementById("narrative-refresh")?.addEventListener("click", () => { void narrative.refresh(true); });

const inspectorPanel = document.getElementById("agent-inspector");
const inspector = new InspectionController({
  read: (uid, request, signal) => {
    if (remoteMode) return remoteClient.inspectAgent(uid, { sessionId: request.sessionId, epoch: request.epoch, signal });
    if (!simHandle?.inspectAgent) throw new Error("This WASM package does not expose creature inspection; rebuild the browser package");
    return simHandle.inspectAgent(uid);
  },
  onChange: (state, frame) => renderInspector(inspectorPanel, state, frame),
  onFrame: (state, frame) => renderInspectorSource(inspectorPanel, state, frame),
});
const updateInspectorVisibility = () => inspector.setEnabled(!reviewedCheckpoint && !document.hidden);
document.addEventListener("visibilitychange", updateInspectorVisibility);
updateInspectorVisibility();
const brainPanel = document.getElementById("brain-inspector");
const brainInspector = new InspectionController({
  enabled: false,
  validate: validateBrainInspection,
  read: (uid, request, signal) => {
    if (remoteMode) return remoteClient.inspectBrain(uid, { sessionId: request.sessionId, epoch: request.epoch, signal });
    if (!simHandle?.inspectBrain) throw new Error("This WASM package does not expose brain activity; rebuild the browser package");
    return simHandle.inspectBrain(uid);
  },
  onChange: (state, frame) => renderBrainInspector(brainPanel, state, frame),
  onFrame: (state, frame) => renderBrainSource(brainPanel, state, frame),
});
const updateBrainVisibility = () => brainInspector.setEnabled(!reviewedCheckpoint && Boolean(brainPanel?.open) && !document.hidden);
brainPanel?.addEventListener("toggle", updateBrainVisibility);
document.addEventListener("visibilitychange", updateBrainVisibility);
document.getElementById("brain-refresh")?.addEventListener("click", () => { void brainInspector.refresh(true); });

const genomePanel = document.getElementById("genome-inspector");
const genomeInspector = new GenomeInspectionController({
  read: (uid, request, signal) => {
    if (remoteMode) return remoteClient.inspectGenome(uid, { sessionId: request.sessionId,
      epoch: request.epoch, locusOffset: request.locusOffset, deltaOffset: request.deltaOffset, signal });
    if (!simHandle?.inspectGenome) throw new Error("This WASM package does not expose genome pages; rebuild the browser package");
    return simHandle.inspectGenome(uid, request.locusOffset, request.deltaOffset);
  },
  onChange: (state, frame) => renderGenomeInspector(genomePanel, state, frame),
  onFrame: (state, frame) => renderGenomeSource(genomePanel, state, frame),
});
const updateGenomeVisibility = () => genomeInspector.setEnabled(!reviewedCheckpoint && Boolean(genomePanel?.open) && !document.hidden);
genomePanel?.addEventListener("toggle", updateGenomeVisibility);
document.addEventListener("visibilitychange", updateGenomeVisibility);
document.getElementById("genome-refresh")?.addEventListener("click", () => { void genomeInspector.restart(); });
for (const [id, kind, direction] of [["locus-prev", "locus", -1], ["locus-next", "locus", 1],
  ["delta-prev", "delta", -1], ["delta-next", "delta", 1]]) {
  document.getElementById(`genome-${id}`)?.addEventListener("click", () => genomeInspector.navigate(kind, direction));
}

const interventionPanel = document.getElementById("intervention-toolbox");
const interventionField = (id) => document.getElementById(`intervention-${id}`);
let interventionRegionContext = null;
let interventionResumeSpeed = 1;
let interventions;
function createInterventionController(remote) {
  return new InterventionController({
    send: (request, signal) => {
      if (remote) return remoteClient.intervene(request, { signal });
      if (!simHandle?.intervene) throw new Error("This WASM package does not expose interventions; rebuild the browser package");
      return simHandle.intervene(request);
    },
    status: (commandId, sessionId, signal) => {
      if (remote) return remoteClient.interventionStatus(commandId, { sessionId, signal });
      if (!simHandle?.interventionStatus) throw new Error("This WASM package does not expose intervention receipts");
      return simHandle.interventionStatus(commandId, sessionId);
    },
    history: (request, signal) => {
      if (remote) return remoteClient.interventionHistory({ ...request, signal });
      if (!simHandle?.interventionHistory) throw new Error("This WASM package does not expose intervention history");
      return simHandle.interventionHistory(request.sessionId, request.epoch, request.afterSeq, request.limit);
    },
    onChange: (controller) => {
      if (controller !== interventions) return;
      if (controller.frame && interventionRegionContext !== controller.context) {
        interventionRegionContext = controller.context;
        const world = controller.frame.snapshot.world;
        interventionField("x").value = String(world.width / 2);
        interventionField("y").value = String(world.height / 2);
        interventionField("radius").value = String(Math.min(100, world.width / 4, world.height / 4));
      }
      renderInterventions(interventionPanel, controller);
    },
  });
}
// Only the retained local owner and the current remote endpoint have saved views.
const localInterventions = createInterventionController(false);
const remoteInterventions = createInterventionController(true);
interventions = localInterventions;
let localInterventionOwner = null;
let remoteInterventionEndpoint = null;
let interventionNeedsAttach = false;
const updateInterventionVisibility = () => interventions.setEnabled(!reviewedCheckpoint && Boolean(interventionPanel?.open)
  && !document.hidden && !interventionNeedsAttach && (!remoteMode || (remoteConnected && remoteFrame !== null)));
function observeInterventions(frame) {
  if (interventionNeedsAttach) {
    // A verified source must also have painted before it can stage an experiment.
    if (!interventions.reattachOwner(frame, simulationGeneration)) return;
    interventionNeedsAttach = false;
    updateInterventionVisibility();
  }
  interventions.observe(frame, simulationGeneration);
}
interventionPanel?.addEventListener("toggle", updateInterventionVisibility);
document.addEventListener("visibilitychange", updateInterventionVisibility);
function updateInterventionFields() {
  const kind = interventionField("kind").value;
  for (const group of interventionPanel.querySelectorAll("[data-intervention-kinds]")) {
    group.hidden = !group.dataset.interventionKinds.split(" ").includes(kind);
    for (const input of group.querySelectorAll("input, select")) input.disabled = group.hidden;
  }
  const disc = kind !== "set_closed_world" && interventionField("region").value === "disc";
  interventionField("disc").hidden = !disc;
  for (const input of interventionField("disc").querySelectorAll("input")) input.disabled = !disc;
}
interventionField("kind")?.addEventListener("change", updateInterventionFields);
interventionField("region")?.addEventListener("change", updateInterventionFields);
if (interventionPanel) updateInterventionFields();
function interventionNumber(id) {
  const text = interventionField(id).value.trim();
  if (!text || !Number.isFinite(Number(text))) throw new Error(`Enter a finite ${id.replaceAll("-", " ")}`);
  return Number(text);
}
interventionField("use-selected")?.addEventListener("click", () => {
  if (reviewedCheckpoint) return;
  const creature = interventions.frame?.snapshot.agents.find((agent) => agent.uid === inspector.selectedUid);
  if (!creature) { interventionField("form-error").textContent = "Select a creature in the displayed world first."; return; }
  interventionField("x").value = String(creature.position[0]);
  interventionField("y").value = String(creature.position[1]);
  interventionField("form-error").textContent = "";
});
interventionField("form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  interventionField("form-error").textContent = "";
  try {
    const kind = interventionField("kind").value;
    let action;
    if (kind === "set_closed_world") action = { kind, closed: interventionField("closed").value === "true" };
    else {
      const region = interventionField("region").value === "all" ? { shape: "all" }
        : { shape: "disc", x: interventionNumber("x"), y: interventionNumber("y"), radius: interventionNumber("radius") };
      const world = interventions.frame?.snapshot.world;
      if (region.shape === "disc" && region.radius > Math.min(world.width, world.height) / 2) {
        throw new Error("Disc radius cannot exceed half the smaller world dimension");
      }
      action = { kind, region };
      if (["drought", "embargo"].includes(kind)) action.ticks = interventionNumber("ticks");
      if (kind === "drought") action.growth_scale = interventionNumber("growth-scale");
      if (kind === "bloom") action.amount = interventionNumber("amount");
      if (kind === "meteor") Object.assign(action, { lethality: interventionNumber("lethality"), scorch: interventionNumber("scorch") });
    }
    await interventions.stage(validateInterventionAction(action));
  } catch (error) { interventionField("form-error").textContent = `Action was not submitted: ${error.message || error}`; }
});
interventionField("retry")?.addEventListener("click", () => { void interventions.retry(); });
interventionField("status-refresh")?.addEventListener("click", () => { void interventions.refreshReceipt(); });
interventionField("history-refresh")?.addEventListener("click", () => { void interventions.refreshHistory(true); });
interventionField("history-first")?.addEventListener("click", () => { void interventions.historyPageAfter("0"); });
interventionField("history-next")?.addEventListener("click", () => {
  if (interventions.historyPage?.hasMore) void interventions.historyPageAfter(interventions.historyPage.nextAfterSeq);
});
for (const action of ["pause", "step", "resume"]) {
  interventionField(action)?.addEventListener("click", () => {
    interventionField("form-error").textContent = "";
    if (remoteMode) { sendRemoteControl(action); return; }
    try {
      if (!simHandle) throw new Error("Wait for the local owner to start");
      if (action === "step") {
        if (simHandle.frame().playback?.paused !== true) throw new Error("Pause the local owner before stepping this experiment");
        simHandle.tick(1);
      } else {
        if (action === "pause" && speedMultiplier > 0) interventionResumeSpeed = speedMultiplier;
        speedMultiplier = action === "pause" ? 0 : interventionResumeSpeed;
        stepsSlider.value = String(speedMultiplier);
        applyPlaybackSpeed();
      }
      lastPaintedFrame = null;
    } catch (error) { interventionField("form-error").textContent = `Playback request failed: ${error.message || error}`; }
  });
}

function selectCreature(uid) {
  if (reviewedCheckpoint) return;
  if (uid === null) cancelWorldDrag();
  const following = worldViewport.followedUid !== null;
  inspector.select(uid);
  brainInspector.select(uid);
  genomeInspector.select(uid);
  if (following) {
    worldViewport.stopFollowing();
    if (uid !== null && paintedWorld()?.frame.snapshot.agents.some((agent) => agent.uid === uid)) {
      worldViewport.follow(uid, paintedWorld().frame.snapshot);
    }
  }
  // Selection and camera revision participate in the paused-frame paint key.
  // Retain the previous accepted transform until the renderer accepts a repaint.
  updateWorldNavigation();
}

document.getElementById("sim-container")?.addEventListener("click", (event) => {
  if (event.target !== canvas || event.button !== 0) return;
  if (worldDrag.consumeClick()) { event.preventDefault(); return; }
  const painted = paintedWorld();
  if (reviewedCheckpoint || !painted || !inspector.frame) return;
  const bounds = canvas.getBoundingClientRect();
  const uid = pickAgent(painted.frame.snapshot, event.clientX - bounds.left, event.clientY - bounds.top,
    { width: bounds.width, height: bounds.height, canvasWidth: canvas.width, canvasHeight: canvas.height, view: painted.view });
  selectCreature(uid);
  canvas.focus({ preventScroll: true });
});
document.getElementById("inspector-form")?.addEventListener("submit", (event) => {
  event.preventDefault();
  const input = document.getElementById("inspector-uid");
  input.setCustomValidity("");
  try { selectCreature(input.value.trim()); }
  catch (error) { input.setCustomValidity(error.message); input.reportValidity(); }
});
document.getElementById("inspector-uid")?.addEventListener("input", (event) => event.target.setCustomValidity(""));
document.getElementById("inspector-clear")?.addEventListener("click", () => selectCreature(null));
document.getElementById("inspector-refresh")?.addEventListener("click", () => { void inspector.refresh(true); });
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && (event.target === canvas || inspectorPanel?.contains(event.target))) {
    selectCreature(null);
  }
});

function paintedWorld() {
  return lastPaintedFrame?.simulationGeneration === simulationGeneration
    && lastPaintedFrame.presentationGeneration === presentationGeneration && lastPaintedFrame.view
    && lastPaintedFrame.viewSource === worldViewport.source
    && lastPaintedFrame.width === canvas.width && lastPaintedFrame.height === canvas.height
    ? lastPaintedFrame : null;
}

function updateWorldNavigation() {
  const painted = paintedWorld();
  const selected = painted && !reviewedCheckpoint ? inspector.selectionFor(painted.frame, simulationGeneration) : null;
  if (worldZoomIn) worldZoomIn.disabled = !painted || worldViewport.view.zoom >= MAX_WORLD_ZOOM;
  if (worldZoomOut) worldZoomOut.disabled = !painted || worldViewport.view.zoom <= MIN_WORLD_ZOOM;
  if (worldFit) worldFit.disabled = !painted;
  if (worldFollow) {
    worldFollow.disabled = !painted || !!reviewedCheckpoint || (!selected && worldViewport.followedUid === null);
    worldFollow.textContent = worldViewport.followedUid === null ? "Follow selected" : "Stop following";
    worldFollow.setAttribute("aria-pressed", String(worldViewport.followedUid !== null));
  }
  if (!worldViewStatus) return;
  let message = "Waiting for a world view.";
  if (painted) {
    const { view, frame } = painted;
    const scale = `${Number(view.zoom.toFixed(2))}×`;
    message = worldViewport.followedUid !== null ? `${scale} · following creature ${worldViewport.followedUid}.`
      : `${scale} · center ${(view.centerX * frame.snapshot.world.width).toFixed(1)}, ${(view.centerY * frame.snapshot.world.height).toFixed(1)}. World edges wrap.`;
    if (view !== worldViewport.view) message += " Updating view…";
    if (worldViewport.notice) message += " " + worldViewport.notice;
    Object.assign(worldViewStatus.dataset, { zoom: String(view.zoom), centerX: String(view.centerX), centerY: String(view.centerY),
      sessionId: frame.identity.sessionId, tick: frame.identity.tick, followedUid: worldViewport.followedUid ?? "" });
  } else {
    for (const key of Object.keys(worldViewStatus.dataset)) delete worldViewStatus.dataset[key];
  }
  if (worldViewStatus.textContent !== message) worldViewStatus.textContent = message;
}

function cancelWorldDrag() {
  const pointerId = worldDrag.active?.pointerId;
  if (pointerId !== undefined) {
    worldDrag.cancel(true);
    if (canvas.hasPointerCapture?.(pointerId)) canvas.releasePointerCapture(pointerId);
  }
  delete canvas.dataset.dragging;
}

function zoomWorld(factor, x = 0.5, y = 0.5, paintedView = worldViewport.view) {
  if (!paintedWorld()) return;
  cancelWorldDrag();
  worldViewport.zoomBy(factor, x, y, paintedView);
  updateWorldNavigation();
}

function followSelectedCreature() {
  const painted = paintedWorld();
  if (!painted || reviewedCheckpoint) return;
  cancelWorldDrag();
  if (worldViewport.followedUid !== null) worldViewport.stopFollowing();
  else {
    const selected = inspector.selectionFor(painted.frame, simulationGeneration);
    if (selected === null) return;
    worldViewport.follow(selected, painted.frame.snapshot);
  }
  updateWorldNavigation();
}

worldZoomIn?.addEventListener("click", () => zoomWorld(1.5));
worldZoomOut?.addEventListener("click", () => zoomWorld(1 / 1.5));
worldFit?.addEventListener("click", () => {
  if (paintedWorld()) { cancelWorldDrag(); worldViewport.fit(); updateWorldNavigation(); }
});
worldFollow?.addEventListener("click", followSelectedCreature);

const worldContainer = document.getElementById("sim-container");
worldContainer?.addEventListener("wheel", (event) => {
  const painted = paintedWorld();
  if (event.target !== canvas || !painted) return;
  const bounds = canvas.getBoundingClientRect();
  if (!(bounds.width > 0 && bounds.height > 0)) return;
  event.preventDefault();
  const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? bounds.height : 1;
  const delta = Math.max(-600, Math.min(600, event.deltaY * unit));
  zoomWorld(Math.exp(-delta * 0.002), (event.clientX - bounds.left) / bounds.width,
    (event.clientY - bounds.top) / bounds.height, painted.view);
}, { passive: false });
worldContainer?.addEventListener("pointerdown", (event) => {
  const painted = paintedWorld();
  if (event.target !== canvas || !painted || event.button !== 0 || event.isPrimary === false) return;
  const bounds = canvas.getBoundingClientRect();
  if (worldDrag.begin(event.pointerId, event.clientX, event.clientY, painted.view, bounds.width, bounds.height)) {
    canvas.setPointerCapture(event.pointerId);
    canvas.focus({ preventScroll: true });
  }
});
worldContainer?.addEventListener("pointermove", (event) => {
  if (event.target !== canvas || !paintedWorld()) return;
  const bounds = canvas.getBoundingClientRect();
  if (!(bounds.width > 0 && bounds.height > 0)) return;
  if (worldDrag.active && (worldDrag.active.width !== bounds.width || worldDrag.active.height !== bounds.height)) {
    cancelWorldDrag();
    return;
  }
  const movement = worldDrag.move(event.pointerId, event.clientX, event.clientY, bounds.width, bounds.height);
  if (!movement) return;
  event.preventDefault();
  worldViewport.panFrom(movement.view, movement.deltaX / bounds.width, movement.deltaY / bounds.height);
  canvas.dataset.dragging = "true";
  updateWorldNavigation();
});
worldContainer?.addEventListener("pointerup", (event) => {
  if (event.target !== canvas || !worldDrag.end(event.pointerId)) return;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  delete canvas.dataset.dragging;
});
for (const kind of ["pointercancel", "lostpointercapture"]) {
  worldContainer?.addEventListener(kind, (event) => {
    if (worldDrag.active?.pointerId === event.pointerId) cancelWorldDrag();
  });
}
document.addEventListener("keydown", (event) => {
  if (event.target !== canvas || event.ctrlKey || event.metaKey || event.altKey || !paintedWorld()) return;
  const pans = { ArrowLeft: [0.1, 0], ArrowRight: [-0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] };
  if (pans[event.key]) {
    cancelWorldDrag();
    worldViewport.panFrom(worldViewport.view, ...pans[event.key].map((value) => value * (event.shiftKey ? 2.5 : 1)));
  } else if (event.key === "+" || event.key === "=") zoomWorld(1.5);
  else if (event.key === "-") zoomWorld(1 / 1.5);
  else if (event.key === "Home" || event.key === "0") { cancelWorldDrag(); worldViewport.fit(); }
  else if (event.key.toLowerCase() === "f") followSelectedCreature();
  else return;
  event.preventDefault();
  updateWorldNavigation();
});

stepsSlider.addEventListener("input", () => {
  if (remoteMode) return;
  speedMultiplier = Number(stepsSlider.value);
  applyPlaybackSpeed();
  appendLog(`Simulation speed set to ${speedMultiplier}×`);
});

document.addEventListener("visibilitychange", () => {
  applyPlaybackSpeed();
  updateCheckpointControls();
});

populationSlider.addEventListener("input", () => {
  population = Number(populationSlider.value);
});

brainPresetSelect?.addEventListener("change", updateCheckpointControls);

resetButton.addEventListener("click", () => {
  if (remoteMode || brainPresetSelect?.value === "mixed_continuation") return;
  if (permalinkStatus) {
    permalinkStatus.textContent = "";
  }
  appendLog("Reset requested");
  if (wasmInitialized && simulationTimer === null) {
    // A failed startup has no owner to drive, and a stopped owner's clock must
    // stay stopped if its replacement is refused. Only a successful install
    // schedules a new pump in either case.
    queuedReset = false;
    resetSimulation(population).catch((err) => {
      console.error(err);
      appendLog(`Reset failed: ${err}`);
    });
  } else {
    queuedReset = true;
  }
});

function appendLog(message) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}`;
  logView.textContent = `${line}\n${logView.textContent}`.slice(0, 4096);
}

function hasPresentationFrame() {
  return remoteMode ? reviewedCheckpoint !== null || remoteFrame !== null : simHandle !== null;
}

function updateRemoteControls() {
  updateInterventionVisibility();
  updateCheckpointLabVisibility();
  updateExperimentVisibility();
  for (const input of [stepsSlider, populationSlider, resetButton, permalinkInput,
    loadPermalinkBtn, copyPermalinkBtn, forkPatchInput, forkBtn, loadForkedBtn, copyForkBtn]) {
    if (input) input.disabled = remoteMode;
  }
  remoteLocal.disabled = !remoteMode;
  remoteReconnect.disabled = !remoteMode || !remoteClient || remoteConnected;
  for (const button of [remotePause, remoteResume, remoteStep]) button.disabled = !remoteConnected || !!reviewedCheckpoint;
  remoteRefresh.disabled = !remoteConnected || !lastRemoteCommandId;
  remoteRetry.disabled = !remoteConnected || !!reviewedCheckpoint || !lastRemoteRequestId;
  captureButton.disabled = !hasPresentationFrame() || !!captureRequest || !!encodingRequest;
  remoteOverlay.hidden = !remoteMode || !!remoteFrame || !!reviewedCheckpoint;
  renderCheckpointView();
  updateCheckpointControls();
  updateWorldNavigation();
}

function updateCheckpointControls() {
  updateLocalTransferControl();
  const supportsContinuation = activeBrainPreset !== "wander";
  const recoveryBusy = browserRecovery?.pending === true;
  if (brainPresetSelect) brainPresetSelect.disabled = remoteMode;
  resetButton.disabled = remoteMode || brainPresetSelect?.value === "mixed_continuation";
  if (checkpointSaveButton) checkpointSaveButton.disabled = remoteMode || !simHandle || checkpointLoadPending || !supportsContinuation;
  if (checkpointLoadInput) checkpointLoadInput.disabled = remoteMode || !wasmInitialized || checkpointLoadPending || browserStorePending || recoveryBusy;
  if (worldCheckpointExport) worldCheckpointExport.disabled = remoteMode || !simHandle || !supportsContinuation
    || checkpointLoadPending || browserStorePending || recoveryBusy || typeof simHandle.worldCheckpoint !== "function"
    || typeof wasm.world_checkpoint_byte_limit !== "function";
  if (worldCheckpointImport) worldCheckpointImport.disabled = remoteMode || !wasmInitialized
    || checkpointLoadPending || browserStorePending || recoveryBusy || typeof wasm.import_world_checkpoint !== "function"
    || typeof wasm.world_checkpoint_byte_limit !== "function";
  const storageBusy = remoteMode || !browserContinuationStore || checkpointLoadPending || browserStorePending || recoveryBusy;
  if (browserSaveButton) browserSaveButton.disabled = storageBusy || !simHandle || !supportsContinuation;
  if (browserRestoreButton) browserRestoreButton.disabled = storageBusy;
  if (checkpointCapability) checkpointCapability.textContent = remoteMode
    ? "These continuation controls apply to your local browser world. Return to Local Simulation to use them."
    : !supportsContinuation
      ? "This Wander world is a movement demo without inherited genomes or saved continuations. Choose MLP, DWRAON or Assembly, then Reset Simulation to start an evolving world."
      : activeBrainPreset === "mixed"
        ? activeLaunchInfo?.kind === "browser_launch"
          ? "This world starts with MLP, DWRAON and Assembly founders in repeating order. Share its recipe to restart or save a continuation to retain its evolution."
          : `This saved world retains ${activeBrainFamilies.join(" + ")} brain families, including any extinct family. Resume or save it to continue; choose a new-world preset before resetting.`
        : "This local world supports saved continuations. Loading a continuation preserves its brain family and resumes paused.";
  browserRecovery?.setAvailable(!remoteMode && !!simHandle && supportsContinuation && !document.hidden
    && !queuedReset && !checkpointLoadPending && !browserStorePending);
  renderRecoveryControls();
  updateRecipeControls();
}

function renderRecoveryControls() {
  const busy = browserStorePending || checkpointLoadPending;
  if (recoveryEnabled) {
    recoveryEnabled.checked = browserRecovery?.enabled === true;
    recoveryEnabled.disabled = !browserRecovery || remoteMode || !simHandle || activeBrainPreset === "wander" || busy;
  }
  if (recoveryRestoreButton) recoveryRestoreButton.disabled = remoteMode || !wasmInitialized
    || !browserContinuationStore || busy || browserRecovery?.pending === true;
  if (recoveryRetryButton) recoveryRetryButton.disabled = remoteMode || !browserRecovery?.enabled
    || !browserRecovery.error || busy || browserRecovery.pending;
  if (!recoveryStatus) return;
  let state = "Recovery is off for this world.";
  if (browserRecovery?.pending) state = browserRecovery.enabled ? "Saving a recovery checkpoint…" : "Cancelling the previous world's recovery write…";
  else if (browserRecovery?.error) state = `Recovery stopped: ${browserRecovery.error.message} Use Retry Recovery after resolving the error.`;
  else if (browserRecovery?.enabled) state = browserRecovery.available
    ? "Recovery is on: saving changed owner state at most once every 30 seconds."
    : "Recovery is on, with new saves suspended while the page is hidden or another save/load is active.";
  const saved = recoveryRecord ? ` Last observed recovery: ${storedContinuationDescription(recoveryRecord)}.` : "";
  recoveryStatus.dataset.state = browserRecovery?.pending ? "saving" : browserRecovery?.error ? "error"
    : browserRecovery?.enabled ? "enabled" : "off";
  recoveryStatus.dataset.savedTick = recoveryRecord?.tick ?? "";
  recoveryStatus.textContent = `${state}${saved}${recoveryReadStatus ? ` ${recoveryReadStatus}` : ""}`;
}

function receiveRemoteFrame(frame) {
  if (!remoteMode) return;
  remotePlayback.textContent = `Observed server: ${frame.playback.paused ? "paused" : "running"}, ${frame.playback.speedMultiplier}× — tick ${frame.identity.tick}`;
  if (frame.kind === "canvas") {
    const browserFrame = { schema: "scriptbots.browser-frame.v1", identity: frame.identity,
      snapshot: frame.payload, environment: frame.environment, playback: frame.playback };
    validateFrameIdentity(browserFrame);
    if (reviewedCheckpoint && reviewedCheckpoint.sourceSessionId !== frame.identity.sessionId) {
      checkpointLab.reset("The connected host changed; choose a checkpoint from the new session.");
    }
    remoteFrame = browserFrame;
    if (frame.coalescedPublications !== "0") {
      remoteStatus.textContent = `Connected — latest publication ${frame.identity.publicationRevision}; skipped ${frame.coalescedPublications} intermediate publications.`;
    }
    updateRemoteControls();
  } else if (frame.kind === "events") {
    narrative.accept(frame);
  }
}

function receiveRemoteState(state) {
  if (!remoteMode) return;
  const hadFrame = remoteFrame !== null;
  remoteConnected = state.state === "connected";
  narrative.setConnection(remoteConnected ? null
    : `Server history is not live: ${String(state.message || state.state).slice(0, 512)}. Retained events describe the earlier observation.`);
  if (!remoteConnected) {
    // The canvas pixels may remain visible, but neither a stale RAF nor the next
    // subscription acknowledgement can admit experiments before a fresh paint.
    remoteFrame = null;
    interventionNeedsAttach = true;
    lastPaintedFrame = null;
    lastPaintedLiveFrame = null;
  }
  remoteStatus.dataset.state = state.state;
  remoteStatus.textContent = state.message || state.state;
  if (["disconnected", "error", "session_changed"].includes(state.state) && hadFrame) {
    remoteStatus.textContent += " Showing the last received frame.";
  }
  if (state.state === "session_changed") {
    checkpointLab.reset("The connected host changed; reconnect before reviewing a saved world.");
    remoteStatus.textContent += " Connect to Server starts a fresh subscription; old cursors are not reused.";
  }
  updateRemoteControls();
}

function receiveRemoteReceipt(reply) {
  if (!remoteMode) return;
  if (reply.request_id !== lastRemoteRequestId && reply.request_id !== lastRemoteStatusRequestId) {
    appendLog(`Earlier remote request ${reply.request_id}: ${reply.type === "receipt"
      ? `${reply.receipt.application_state}; journal ${reply.receipt.journal_state}`
      : `${reply.status}: ${reply.message}`}`);
    return;
  }
  if (reply.type === "receipt") {
    const receipt = reply.receipt;
    lastRemoteCommandId = receipt.command_id;
    remoteReceipt.textContent = `Command ${receipt.command_id}: application ${receipt.application_state}; journal ${receipt.journal_state}.`;
    if (receipt.rejection_reason) remoteReceipt.textContent += ` ${JSON.stringify(receipt.rejection_reason)}`;
  } else {
    remoteReceipt.textContent = `Command response ${reply.status}: ${reply.message}`;
  }
  appendLog(remoteReceipt.textContent);
  updateRemoteControls();
}

function connectRemote() {
  let installedClient = false;
  try {
    // Refuse invalid connection options before disturbing either retained owner.
    const nextClient = new RemoteStreamClient({
      url: remoteUrl.value.trim(),
      decodeFrame: wasm.decode_remote_frame_binary,
      channels: ["canvas", "telemetry", "events"],
      maxFps: 60,
      onFrame: receiveRemoteFrame,
      onState: receiveRemoteState,
      onReceipt: receiveRemoteReceipt,
    });
    // Freeze local science before changing the active source. The saved local
    // owner is resumed only by the user's explicit Return to Local action.
    if (!remoteMode && simHandle) {
      driveSimulation();
      simHandle.setSpeed(0);
    }
    if (simulationTimer !== null) clearTimeout(simulationTimer);
    simulationTimer = null;
    queuedReset = false;
    interventions.setEnabled(false);
    checkpointLab.returnToLive();
    remoteClient?.disconnect();
    if (remoteClient?.url !== nextClient.url) {
      checkpointLab.reset("Connected endpoint changed; choose a checkpoint from the new server.");
      experiments.reset("Connected endpoint changed; choose an experiment from the new server.");
    }
    remoteClient = nextClient;
    installedClient = true;
    remoteMode = true;
    recoveryReadStatus = "";
    browserRecovery?.attach(null);
    remoteConnected = false;
    remoteFrame = null;
    lastRemoteRequestId = null;
    lastRemoteCommandId = null;
    lastRemoteStatusRequestId = null;
    simulationGeneration += 1;
    presentationGeneration += 1;
    lastPaintedLiveFrame = null;
    narrative.reset("remote");
    inspector.reset();
    brainInspector.reset();
    genomeInspector.reset();
    interventions = remoteInterventions;
    if (remoteInterventionEndpoint !== nextClient.url) remoteInterventions.reset();
    remoteInterventionEndpoint = nextClient.url;
    interventionNeedsAttach = true;
    remoteInterventions.emit();
    lastPaintedFrame = null;
    if (captureRequest) finishCapture(captureRequest, new Error("Presentation source changed"));
    remoteReceipt.textContent = "";
    remotePlayback.textContent = "Waiting for observed server state…";
    updateMismatchBanner(null);
    resetPerformanceWindows();
    updateRemoteControls();
    remoteClient.connect();
  } catch (error) {
    const message = `Connection failed: ${error.message}`;
    if (installedClient) receiveRemoteState({ state: "error", message });
    else { remoteStatus.dataset.state = "error"; remoteStatus.textContent = message; }
  }
}

remoteConnect.addEventListener("click", connectRemote);
remoteReconnect.addEventListener("click", () => {
  try { remoteClient?.reconnect(); }
  catch (error) { receiveRemoteState({ state: "error", message: error.message }); }
});
function returnToLocalSimulation() {
  checkpointLab.returnToLive();
  interventions.setEnabled(false);
  checkpointLab.setEnabled(false);
  experiments.setEnabled(false);
  remoteClient?.disconnect();
  remoteMode = false;
  browserRecovery?.attach(simHandle);
  remoteConnected = false;
  remoteClient = null;
  remoteFrame = null;
  simulationGeneration += 1;
  presentationGeneration += 1;
  lastPaintedLiveFrame = null;
  narrative.reset("local");
  inspector.reset();
  brainInspector.reset();
  genomeInspector.reset();
  interventions = localInterventions;
  interventionNeedsAttach = true;
  if (localInterventionOwner !== simHandle) {
    localInterventions.reset();
    localInterventionOwner = simHandle;
  }
  lastPaintedFrame = null;
  if (captureRequest) finishCapture(captureRequest, new Error("Presentation source changed"));
  remoteStatus.textContent = "Running locally in this browser.";
  remoteStatus.dataset.state = "local";
  remotePlayback.textContent = "";
  remoteReceipt.textContent = "";
  resetPerformanceWindows();
  applyPlaybackSpeed();
  localInterventions.emit();
  updateRemoteControls();
  scheduleSimulation();
}
remoteLocal.addEventListener("click", returnToLocalSimulation);

function sendRemoteControl(command) {
  try {
    if (reviewedCheckpoint) throw new Error("Return to live before controlling the server");
    lastRemoteCommandId = null;
    lastRemoteStatusRequestId = null;
    lastRemoteRequestId = remoteClient.sendControl(command);
    remoteReceipt.textContent = `Sent ${command} (${lastRemoteRequestId}); awaiting the server's receipt.`;
    updateRemoteControls();
  } catch (error) {
    if (error.requestId) {
      lastRemoteRequestId = error.requestId;
      remoteReceipt.textContent = `Command delivery unconfirmed (${error.requestId}): ${error.message}. Reconnect and retry with the same ID.`;
    } else {
      remoteReceipt.textContent = `Command was not sent: ${error.message}`;
    }
    updateRemoteControls();
  }
}

remotePause.addEventListener("click", () => sendRemoteControl("pause"));
remoteResume.addEventListener("click", () => sendRemoteControl("resume"));
remoteStep.addEventListener("click", () => sendRemoteControl("step"));
remoteRefresh.addEventListener("click", () => {
  try { lastRemoteStatusRequestId = remoteClient.sendControl("status", { commandId: lastRemoteCommandId }); }
  catch (error) { remoteReceipt.textContent = `Outcome lookup failed: ${error.message}`; }
});
remoteRetry.addEventListener("click", () => {
  try {
    if (reviewedCheckpoint) throw new Error("Return to live before retrying a server command");
    remoteClient.retryControl(lastRemoteRequestId);
    remoteReceipt.textContent = `Retried the same command identity (${lastRemoteRequestId}); awaiting the server's receipt.`;
  } catch (error) { remoteReceipt.textContent = `Retry failed: ${error.message}`; }
});

function resetPerformanceWindows(snapshot = null) {
  const now = performance.now();
  metrics.lastFrameTs = now;
  metrics.lastStatsTs = now;
  metrics.previousTick = snapshot?.tick ?? (remoteMode ? remoteFrame?.snapshot.tick ?? null : simHandle ? simHandle.snapshot().tick : null);
  metrics.fps = 0;
  metrics.tps = 0;
  metrics.tick = 0;
  metrics.statsTickAcc = 0;
  perfWindow.frameAcc = 0;
  perfWindow.tickAcc = 0;
  perfWindow.lastLog = now;
}

function reportRendererFailure(error) {
  rendererStatus.textContent = `Rendering unavailable: ${error.message}`;
  appendLog(rendererStatus.textContent);
}

async function selectRenderer(preference, fallbackReason = null) {
  const generation = ++rendererGeneration;
  // A canvas cannot switch context kinds. Prepare a replacement while the
  // current renderer keeps presenting the latest snapshot.
  let nextCanvas = canvas.cloneNode(false);
  let nextRenderer = null;
  let backend = "canvas";
  if (preference !== "canvas") {
    rendererStatus.textContent = `${rendererStatus.dataset.renderer === "webgpu" ? "WebGPU" : "Canvas 2D"} — starting WebGPU…`;
    try {
      nextRenderer = await WebGpuRenderer.create(nextCanvas, (reason) => {
        if (renderer === nextRenderer) {
          void selectRenderer("canvas", reason).catch(reportRendererFailure);
        }
      });
      backend = "webgpu";
    } catch (error) {
      fallbackReason = `WebGPU unavailable: ${error.message}`;
      // Creation may already have acquired a WebGPU context.
      nextCanvas = canvas.cloneNode(false);
    }
  }
  if (!nextRenderer) nextRenderer = new CanvasRenderer(nextCanvas);
  if (generation !== rendererGeneration) {
    nextRenderer.destroy();
    return;
  }
  cancelWorldDrag();
  renderer?.destroy();
  canvas.replaceWith(nextCanvas);
  canvas = nextCanvas;
  renderer = nextRenderer;
  lastPaintedFrame = null;
  updateWorldNavigation();
  rendererStatus.dataset.renderer = backend;
  for (const key of Object.keys(rendererStatus.dataset)) {
    if (key !== "renderer") delete rendererStatus.dataset[key];
  }
  rendererStatus.textContent = backend === "webgpu" ? "WebGPU" : "Canvas 2D";
  if (fallbackReason) {
    rendererStatus.textContent += ` — ${fallbackReason}`;
    appendLog(`Renderer fallback: ${fallbackReason}`);
  }
}

rendererSelect.addEventListener("change", () => {
  void selectRenderer(rendererSelect.value).catch(reportRendererFailure);
});

environmentLayer?.addEventListener("change", () => {
  // Presentation-only: redraw the same committed frame, including while paused.
  lastPaintedFrame = null;
});

function validateFrameIdentity(frame) {
  const identity = frame?.identity;
  if (frame?.schema !== "scriptbots.browser-frame.v1"
      || !Number.isSafeInteger(frame.snapshot.tick) || identity?.tick !== String(frame.snapshot.tick)
      || ![identity.sessionId, identity.publicationRevision, identity.scientificRevision,
        identity.controlRevision, identity.configRevision, identity.tick]
        .every((value) => typeof value === "string" && /^(0|[1-9][0-9]*)$/.test(value))) {
    throw new Error("Frame identity does not match the rendered snapshot");
  }
}

function observePresentedLiveFrame(frame) {
  lastPaintedLiveFrame = { simulationGeneration, sessionId: frame.identity.sessionId };
  inspector.observe(frame, simulationGeneration);
  brainInspector.observe(frame, simulationGeneration);
  genomeInspector.observe(frame, simulationGeneration);
  observeInterventions(frame);
  observeCheckpointLab(frame);
  observeExperiments(frame);
}

function drawSnapshot(frame, historical = false) {
  if (!renderer) return false;
  let view, selectedUid;
  try {
    validateFrameIdentity(frame);
    const previousSource = worldViewport.source;
    view = worldViewport.observe(frame, presentationGeneration);
    if (worldViewport.source !== previousSource) {
      cancelWorldDrag();
      // A busy renderer may retain old pixels; they cannot authorize a new owner's camera.
      updateWorldNavigation();
    }
    if (worldDrag.active) {
      const bounds = canvas.getBoundingClientRect();
      if (worldDrag.active.width !== bounds.width || worldDrag.active.height !== bounds.height
          || (lastPaintedFrame && (lastPaintedFrame.width !== canvas.width || lastPaintedFrame.height !== canvas.height))) {
        cancelWorldDrag();
      }
    }
    selectedUid = historical ? null : inspector.selectionFor(frame, simulationGeneration);
  } catch (error) {
    if (captureRequest) finishCapture(captureRequest, error);
    else showCaptureError(`Frame refused: ${error.message}`);
    return false;
  }

  if (lastPaintedFrame?.simulationGeneration === simulationGeneration
      && lastPaintedFrame.presentationGeneration === presentationGeneration
      && lastPaintedFrame.publicationRevision === frame.identity.publicationRevision
      && lastPaintedFrame.sessionId === frame.identity.sessionId
      && lastPaintedFrame.viewSource === worldViewport.source
      && lastPaintedFrame.viewRevision === worldViewport.revision
      && lastPaintedFrame.selectedUid === selectedUid
      && lastPaintedFrame.width === canvas.width && lastPaintedFrame.height === canvas.height
      && (!captureRequest || captureRequest.encoding)) {
    if (!historical) observePresentedLiveFrame(frame);
    updateWorldNavigation();
    return true;
  }
  try {
    if (renderer.render(frame.snapshot, { environment: frame.environment, identity: frame.identity,
      layer: environmentLayer?.value ?? "combined", generation: presentationGeneration,
      selectedUid, view })) {
      lastPaintedFrame = { simulationGeneration, presentationGeneration, publicationRevision: frame.identity.publicationRevision,
        sessionId: frame.identity.sessionId, width: canvas.width, height: canvas.height,
        view, viewSource: worldViewport.source, viewRevision: worldViewport.revision, selectedUid, frame };
      for (const [key, value] of Object.entries(frame.identity)) {
        rendererStatus.dataset[key === "tick" ? "snapshotTick" : key] = value;
      }
      if (environmentStatus) {
        environmentStatus.textContent = environmentDescription(frame.environment);
        environmentStatus.dataset.state = frame.environment ? "observed" : "unavailable";
        environmentStatus.dataset.sessionId = frame.identity.sessionId;
        environmentStatus.dataset.publicationRevision = frame.identity.publicationRevision;
        environmentStatus.dataset.layer = environmentLayer?.value ?? "combined";
      }
      rendererStatus.dataset.view = historical ? "checkpoint" : "live";
      rendererStatus.dataset.checkpointId = historical ? reviewedCheckpoint.checkpoint.checkpoint_id : "";
      renderCheckpointView();
      captureAdmittedFrame(frame.identity);
      if (!historical) observePresentedLiveFrame(frame);
      updateWorldNavigation();
      return true;
    }
  } catch (error) {
    if (rendererStatus.dataset.renderer === "webgpu") {
      void selectRenderer("canvas", `WebGPU stopped: ${error.message}`).catch(reportRendererFailure);
    } else {
      throw error;
    }
  }
  return false;
}

function showCaptureError(message) {
  for (const key of Object.keys(captureStatus.dataset)) delete captureStatus.dataset[key];
  captureStatus.dataset.state = "error";
  if (captureStatus.textContent !== message) appendLog(message);
  captureStatus.textContent = message;
}

function finishCapture(request, error = null) {
  if (captureRequest !== request) return;
  clearTimeout(request.timeout);
  captureRequest = null;
  captureButton.disabled = !hasPresentationFrame() || !!encodingRequest;
  if (error) {
    showCaptureError(`PNG export failed: ${error.message}`);
  }
}

captureButton.addEventListener("click", () => {
  if (!hasPresentationFrame() || captureRequest || encodingRequest) return;
  const request = { simulationGeneration, presentationGeneration, encoding: false, timeout: null };
  request.timeout = setTimeout(() => finishCapture(request, new Error(encodingRequest === request
    ? "No PNG completed within 10 seconds; encoding is still finishing"
    : "No frame admitted within 10 seconds; try again")), 10_000);
  captureRequest = request;
  captureButton.disabled = true;
  for (const key of Object.keys(captureStatus.dataset)) delete captureStatus.dataset[key];
  captureStatus.dataset.state = "waiting";
  captureStatus.textContent = "Waiting for the next rendered frame…";
});

function captureAdmittedFrame(frameIdentity) {
  const request = captureRequest;
  if (!request || request.encoding) return;
  try {
    if (request.simulationGeneration !== simulationGeneration || request.presentationGeneration !== presentationGeneration) {
      throw new Error("Displayed world changed before capture; try again");
    }
    const identity = { ...frameIdentity };
    const capturedView = lastPaintedFrame.view;
    const checkpoint = reviewedCheckpoint?.checkpoint;
    const backend = rendererStatus.dataset.renderer;
    const filename = checkpoint
      ? `scriptbots-checkpoint-${checkpoint.checkpoint_id}-t${checkpoint.tick}-${checkpoint.world_digest_v1}-${backend}.png`
      : `scriptbots-session-${identity.sessionId}-p${identity.publicationRevision}-s${identity.scientificRevision}-c${identity.controlRevision}-g${identity.configRevision}-t${identity.tick}-${backend}.png`;
    request.encoding = true;
    encodingRequest = request;
    captureStatus.dataset.state = "encoding";
    captureStatus.textContent = `Encoding tick ${identity.tick}…`;
    // Invoke serialization in the submitting callback, before another frame
    // can replace the drawing buffer. The callback retains identity, not agents.
    canvas.toBlob((blob) => {
      const completedEncoding = encodingRequest === request;
      if (completedEncoding) encodingRequest = null;
      if (captureRequest !== request) {
        if (completedEncoding && !captureRequest) {
          showCaptureError("PNG encoding finished after its request expired; try again");
        }
        captureButton.disabled = !hasPresentationFrame() || !!captureRequest || !!encodingRequest;
        return;
      }
      try {
        if (request.simulationGeneration !== simulationGeneration || request.presentationGeneration !== presentationGeneration) {
          throw new Error("Displayed world changed during capture; try again");
        }
        if (!blob || blob.type !== "image/png") throw new Error("The browser did not produce a PNG");
        if (captureUrl) URL.revokeObjectURL(captureUrl);
        captureUrl = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = captureUrl;
        link.download = filename;
        link.click();
        Object.assign(captureStatus.dataset, identity, { renderer: backend, state: "ready", filename,
          view: checkpoint ? "checkpoint" : "live", checkpointId: checkpoint?.checkpoint_id ?? "",
          worldDigestV1: checkpoint?.world_digest_v1 ?? "", cameraZoom: String(capturedView.zoom),
          cameraCenterX: String(capturedView.centerX), cameraCenterY: String(capturedView.centerY) });
        captureStatus.textContent = checkpoint
          ? `Saved-world PNG ready — ${checkpoint.checkpoint_id}, tick ${checkpoint.tick}, world digest ${checkpoint.world_digest_v1} (${backend === "webgpu" ? "WebGPU" : "Canvas 2D"})`
          : `PNG ready — tick ${identity.tick}, science revision ${identity.scientificRevision}, publication ${identity.publicationRevision} (${backend === "webgpu" ? "WebGPU" : "Canvas 2D"})`;
        finishCapture(request);
      } catch (error) {
        finishCapture(request, error);
      }
    }, "image/png");
  } catch (error) {
    if (encodingRequest === request) encodingRequest = null;
    finishCapture(request, error);
  }
}

window.addEventListener("pagehide", () => {
  cancelWorldDrag();
  browserRecovery?.setEnabled(false);
  narrative.setEnabled(false);
  checkpointLab.setEnabled(false);
  experiments.setEnabled(false);
  if (experimentExportUrl) URL.revokeObjectURL(experimentExportUrl);
  experimentExportUrl = null;
  if (labPollTimer !== null) clearInterval(labPollTimer);
  labPollTimer = null;
  if (labDownloadUrl) URL.revokeObjectURL(labDownloadUrl);
  labDownloadUrl = null;
  remoteClient?.disconnect();
  if (captureRequest) finishCapture(captureRequest, new Error("Page left during capture"));
  if (captureUrl) URL.revokeObjectURL(captureUrl);
  captureUrl = null;
});

function updateStats(snapshot, now, historical = false) {
  const frameDt = now - metrics.lastFrameTs;
  metrics.lastFrameTs = now;

  if (frameDt > 0) {
    const instantaneous = 1000 / frameDt;
    metrics.fps = metrics.fps * 0.85 + instantaneous * 0.15;
  }

  if (historical) {
    fpsEl.textContent = metrics.fps.toFixed(1);
    tpsEl.textContent = "—";
    tickEl.textContent = snapshot.tick.toLocaleString();
    popEl.textContent = snapshot.summary.agentCount.toLocaleString();
    // Checkpoints retain scientific state, not the original per-tick activity.
    birthsEl.textContent = "—";
    deathsEl.textContent = "—";
    energyEl.textContent = snapshot.summary.averageEnergy.toFixed(3);
    healthEl.textContent = snapshot.summary.averageHealth.toFixed(3);
    return;
  }

  const statsDt = now - metrics.lastStatsTs;
  const previousTick = metrics.previousTick ?? snapshot.tick;
  const deltaTicks = snapshot.tick - previousTick;
  metrics.previousTick = snapshot.tick;
  metrics.tick = snapshot.tick;
  metrics.statsTickAcc += deltaTicks;

  if (statsDt >= 500) {
    metrics.tps = metrics.statsTickAcc * (1000 / statsDt);
    metrics.statsTickAcc = 0;
    metrics.lastStatsTs = now;

    fpsEl.textContent = metrics.fps.toFixed(1);
    tpsEl.textContent = metrics.tps.toFixed(1);
    tickEl.textContent = snapshot.tick.toLocaleString();
    popEl.textContent = snapshot.summary.agentCount.toLocaleString();
    birthsEl.textContent = snapshot.summary.births.toLocaleString();
    deathsEl.textContent = snapshot.summary.deaths.toLocaleString();
    energyEl.textContent = snapshot.summary.averageEnergy.toFixed(3);
    healthEl.textContent = snapshot.summary.averageHealth.toFixed(3);
  }

  perfWindow.frameAcc += 1;
  perfWindow.tickAcc += deltaTicks;
  const logDt = now - perfWindow.lastLog;
  if (logDt >= 5000) {
    const durationSeconds = logDt / 1000;
    const avgFps = perfWindow.frameAcc / durationSeconds;
    const avgTps = perfWindow.tickAcc / durationSeconds;
    appendLog(
      `5s avg — FPS: ${avgFps.toFixed(1)} | TPS: ${avgTps.toFixed(1)} | Population: ${snapshot.summary.agentCount}`,
    );
    perfWindow.frameAcc = 0;
    perfWindow.tickAcc = 0;
    perfWindow.lastLog = now;
  }
}

function presentSimulation(now) {
  try {
    if (!hasPresentationFrame()) return;

    const liveFrame = remoteMode ? remoteFrame : simHandle.frame();
    if (liveFrame) narrative.observe(liveFrame, simulationGeneration);
    const frame = reviewedCheckpoint?.frame ?? liveFrame;
    const presented = drawSnapshot(frame, !!reviewedCheckpoint);
    updateStats(frame.snapshot, now, !!reviewedCheckpoint);
    if (presented && presentationFailure) {
      if (rendererStatus.textContent === presentationFailure.message) {
        rendererStatus.textContent = presentationFailure.previousStatus;
      }
      delete rendererStatus.dataset.presentationState;
      presentationFailure = null;
      appendLog("Presentation resumed");
    }
  } catch (error) {
    // A renderer may have changed part of the canvas before throwing. Even a
    // paused, unchanged publication must be painted again after recovery.
    lastPaintedFrame = null;
    const detail = String(error?.message || error).slice(0, 512);
    if (captureRequest) finishCapture(captureRequest, new Error(detail));
    if (!presentationFailure) {
      presentationFailure = {
        message: `Presentation interrupted: ${detail}. Retrying…`,
        previousStatus: rendererStatus.textContent,
      };
      console.error("Presentation interrupted", error);
      appendLog(presentationFailure.message);
    }
    // Report once per continuous failure, including when the failing frame
    // changes every callback. A renderer selection can replace the status while
    // recovery is pending; retain that new backend's label for successful paint.
    if (rendererStatus.textContent !== presentationFailure.message) {
      presentationFailure.previousStatus = rendererStatus.textContent;
      rendererStatus.textContent = presentationFailure.message;
    }
    if (rendererStatus.dataset.presentationState !== "error") {
      rendererStatus.dataset.presentationState = "error";
    }
  } finally {
    // Presentation owns one RAF chain, independently of the scientific timer.
    // A failed frame must not kill that chain or restart a stopped owner clock.
    requestAnimationFrame(presentSimulation);
  }
}

function driveSimulation() {
  if (remoteMode || !simHandle || simulationStartedAt === null) return;
  const receipt = simHandle.drive(performance.now() - simulationStartedAt);
  if (receipt.automatic_steps_skipped > 0) {
    appendLog(
      `Simulation timer missed ${receipt.automatic_steps_skipped} tick opportunities; completed ${receipt.scientific_steps} bounded catch-up ticks`,
    );
  }
}

function applyPlaybackSpeed() {
  if (remoteMode || !simHandle) return;
  // Settle the prior speed up to this input boundary before applying the new
  // canonical command. Resuming never accrues science for a hidden interval.
  driveSimulation();
  simHandle.setSpeed(document.hidden ? 0 : speedMultiplier);
}

function releaseSimulation(handle) {
  try {
    handle.free();
  } catch (error) {
    console.error("Failed to release simulation", error);
    appendLog(`Failed to release simulation: ${error.message || error}`);
  }
}

function replaceSimulation(next, { continuation = false, permalink = null, expectedTick = null, expectedBrainPreset = null, expectedLaunchKind = null } = {}) {
  const nextSpeed = continuation ? 0 : speedMultiplier;
  let frame;
  let brainPreset;
  let brainFamilies = [];
  let launchInfo;
  try {
    // All fallible owner reads happen before the active world, clock, or UI is
    // changed. Refusing a candidate must leave the previous owner usable.
    next.setSpeed(document.hidden ? 0 : nextSpeed);
    frame = next.frame();
    validateFrameIdentity(frame);
    if (expectedTick !== null && frame.identity.tick !== expectedTick) {
      throw new Error("Saved continuation tick does not match its checkpoint bytes");
    }
    brainPreset = next.brainPreset?.() ?? null;
    if (expectedBrainPreset !== null && brainPreset !== expectedBrainPreset) {
      throw new Error("The browser package did not create the requested brain preset; reload an updated package");
    }
    launchInfo = typeof next.launchInfo === "function" ? validateLaunchInfo(next.launchInfo()) : continuation ? {
      kind: "checkpoint_continuation", semantics: "saved_owner_continuation", shareable: false,
      seed: null, initial_population: null, brain_preset: null, recipe_version: null,
      reason: "This package did not record the original launch recipe for the saved owner.",
    } : null;
    if (expectedLaunchKind !== null && launchInfo?.kind !== expectedLaunchKind) {
      throw new Error("The browser package did not identify the imported scientific checkpoint; reload an updated package");
    }
    if (continuation || brainPreset === "mixed") {
      brainFamilies = validateContinuationBrains(brainPreset, next.brainFamilies?.() ?? null);
    }
    if (!continuation && brainPreset === "mixed"
        && (launchInfo?.kind !== "browser_launch" || launchInfo.brain_preset !== "mixed"
          || launchInfo.recipe_version !== 2 || brainFamilies.join(",") !== "mlp,dwraon,assembly")) {
      throw new Error("Mixed founder recipe has an inconsistent family roster");
    }
    if (launchInfo?.shareable === false) {
      permalink = null;
    } else if (!continuation && permalink_of) {
      permalink = permalink_of(next);
    } else if (!continuation && next.permalinkOf) {
      permalink = next.permalinkOf();
    }
  } catch (error) {
    releaseSimulation(next);
    throw error;
  }

  const previous = simHandle;
  simHandle = next;
  recoveryReadStatus = "";
  browserRecovery?.attach(next);
  activeBrainPreset = brainPreset;
  activeBrainFamilies = brainFamilies;
  activeLaunchInfo = launchInfo;
  latestForkedLink = null;
  if (forkOutput) forkOutput.style.display = "none";
  if (forkRecipeInfo) forkRecipeInfo.textContent = "";
  simulationGeneration += 1;
  presentationGeneration += 1;
  lastPaintedLiveFrame = null;
  narrative.reset("local");
  inspector.reset();
  brainInspector.reset();
  genomeInspector.reset();
  // Checkpoint restoration may retain the canonical session ID, but it replaces
  // the actual local owner and must never inherit the prior owner's requests.
  localInterventions.reset();
  localInterventionOwner = next;
  if (interventions === localInterventions) interventionNeedsAttach = true;
  simulationStartedAt = performance.now();
  lastPaintedFrame = null;
  if (continuation) {
    speedMultiplier = 0;
    stepsSlider.value = "0";
  }
  if (brainPresetSelect && brainPreset !== null) {
    brainPresetSelect.value = continuation && brainPreset === "mixed" ? "mixed_continuation" : brainPreset;
  }
  resetPerformanceWindows(frame.snapshot);
  if (captureRequest) finishCapture(captureRequest, new Error("World changed during capture; try again"));
  captureButton.disabled = !!captureRequest || !!encodingRequest;
  updateCheckpointControls();
  if (previous) releaseSimulation(previous);
  scheduleSimulation();
  return { frame, permalink, launchInfo };
}

function scheduleSimulation() {
  if (!remoteMode && simHandle && simulationTimer === null) {
    simulationTimer = setTimeout(pumpSimulation, 1000 / 60);
  }
}

function pumpSimulation() {
  simulationTimer = null;
  if (remoteMode) return;

  if (queuedReset) {
    queuedReset = false;
    resetSimulation(population).catch((err) => {
      console.error(err);
      appendLog(`Reset failed: ${err}`);
    });
  }

  try {
    driveSimulation();
  } catch (err) {
    console.error("Simulation clock stopped", err);
    appendLog(`Simulation clock stopped: ${err}`);
    if (simulationTimer !== null) clearTimeout(simulationTimer);
    simulationTimer = null;
    return;
  }
  if (browserRecovery?.enabled) {
    browserRecovery.setAvailable(!document.hidden && !checkpointLoadPending && !browserStorePending
      && !queuedReset && activeBrainPreset !== "wander");
    // Recovery observes the completed owner boundary. Capturing never drives
    // elapsed time, and a stalled storage write cannot enqueue more snapshots.
    void browserRecovery.poll(() => simHandle.frame().identity,
      () => new Blob([simHandle.checkpoint()], { type: "application/octet-stream" }));
  }
  // One outstanding callback, no snapshot queue. HostCore owns deadlines and
  // its unchanged catch-up bound; timer jitter never becomes a render tick.
  scheduleSimulation();
}

function updateMismatchBanner(matchResult) {
  renderBuildComparison(mismatchBannerEl, matchResult);
}

function updateDiffTable(linkStr) {
  if (!diffTable || !diffTbody || !diffEmpty) return;

  if (!linkStr || !permalink_diff) {
    diffTable.style.display = "none";
    diffEmpty.style.display = "block";
    diffEmpty.textContent = "No recorded configuration changes.";
    return;
  }

  try {
    const rows = permalink_diff(linkStr);
    if (!rows || rows.length === 0) {
      diffTable.style.display = "none";
      diffEmpty.style.display = "block";
      diffEmpty.textContent = "No configuration changes from the original recipe or gallery base (0 diffs).";
      diffTbody.innerHTML = "";
      return;
    }

    diffEmpty.style.display = "none";
    diffTable.style.display = "table";
    diffTbody.innerHTML = "";

    for (const row of rows) {
      const tr = document.createElement("tr");
      tr.style.borderBottom = "1px solid rgba(255,255,255,0.05)";

      const tdKnob = document.createElement("td");
      tdKnob.style.padding = "0.25rem 0.4rem";
      tdKnob.style.fontFamily = "monospace";
      tdKnob.textContent = row.knob;

      const tdParent = document.createElement("td");
      tdParent.style.padding = "0.25rem 0.4rem";
      tdParent.textContent = (row.parent_value ?? row.parentValue ?? 0).toString();

      const tdThis = document.createElement("td");
      tdThis.style.padding = "0.25rem 0.4rem";
      tdThis.style.fontWeight = "600";
      tdThis.style.color = "#93c5fd";
      tdThis.textContent = (row.this_value ?? row.thisValue ?? 0).toString();

      tr.appendChild(tdKnob);
      tr.appendChild(tdParent);
      tr.appendChild(tdThis);
      diffTbody.appendChild(tr);
    }
  } catch (err) {
    console.warn("Could not compute permalink diff:", err);
    diffTable.style.display = "none";
    diffEmpty.style.display = "block";
    diffEmpty.textContent = `Could not compute diff: ${err}`;
  }
}

async function loadWorldFromPermalink(linkStr) {
  if (remoteMode || !linkStr) return;
  try {
    if (permalinkStatus) {
      permalinkStatus.textContent = "Loading permalink...";
      permalinkStatus.style.color = "#93c5fd";
    }

    let matchResult = null;
    if (check_build_match) {
      try {
        matchResult = check_build_match(linkStr);
      } catch (e) {
        console.warn("Failed to check build match:", e);
      }
    }

    if (!init_from_permalink) {
      throw new Error("init_from_permalink is not exported by wasm module");
    }

    const installed = replaceSimulation(init_from_permalink(linkStr), { permalink: linkStr });
    currentPermalink = installed.permalink;
    if (permalinkInput) {
      permalinkInput.value = linkStr;
    }

    updateMismatchBanner(matchResult);
    updateDiffTable(linkStr);

    if (permalinkStatus) {
      permalinkStatus.textContent = installed.launchInfo?.kind === "gallery_reference"
        ? "Gallery configuration loaded at tick zero; no browser founder recipe is encoded"
        : "Initial-condition recipe loaded; world restarted at tick zero";
      permalinkStatus.style.color = "#34d399";
    }
    appendLog(`Restarted from shared initial-condition recipe (${linkStr.slice(0, 20)}…)`);

    try {
      const url = new URL(window.location.href);
      url.searchParams.set("world", linkStr);
      window.history.replaceState({}, "", url.toString());
    } catch (_) {}
  } catch (err) {
    console.error("Failed to load permalink:", err);
    if (permalinkStatus) {
      permalinkStatus.textContent = `Error: ${err}`;
      permalinkStatus.style.color = "#f87171";
    }
    appendLog(`Failed to load permalink: ${err}`);
  }
}

async function handleFork() {
  if (remoteMode || !simHandle || activeLaunchInfo?.shareable === false) {
    appendLog("Cannot fork: simulation not running");
    return;
  }

  let patch = {};
  const patchText = forkPatchInput ? forkPatchInput.value.trim() : "";
  if (patchText) {
    try {
      patch = JSON.parse(patchText);
    } catch (e) {
      if (permalinkStatus) {
        permalinkStatus.textContent = "Invalid JSON in fork knob patch";
        permalinkStatus.style.color = "#f87171";
      }
      appendLog(`Fork error: invalid JSON: ${e.message}`);
      return;
    }
  }

  try {
    let childLink = null;
    if (fork) {
      childLink = fork(simHandle, patch);
    } else if (simHandle.fork) {
      childLink = simHandle.fork(patch);
    } else {
      throw new Error("fork is not exported by wasm module");
    }

    const childInfo = typeof wasm.permalink_info === "function" ? validateLaunchInfo(wasm.permalink_info(childLink)) : null;
    latestForkedLink = childLink;
    if (forkChildLink) {
      forkChildLink.textContent = childLink;
    }
    if (forkOutput) {
      forkOutput.style.display = "block";
    }
    if (forkRecipeInfo) forkRecipeInfo.textContent = launchInfoText(childInfo);
    updateRecipeControls();
    if (permalinkStatus) {
      permalinkStatus.textContent = "Variant recipe created; loading starts a new world at tick zero";
      permalinkStatus.style.color = "#34d399";
    }
    appendLog(`Created initial-condition variant link (${childLink.slice(0, 20)}…)`);
  } catch (err) {
    console.error("Fork failed:", err);
    if (permalinkStatus) {
      permalinkStatus.textContent = `Fork error: ${err}`;
      permalinkStatus.style.color = "#f87171";
    }
    appendLog(`Fork failed: ${err}`);
  }
}

if (loadPermalinkBtn) {
  loadPermalinkBtn.addEventListener("click", () => {
    const link = permalinkInput ? permalinkInput.value.trim() : "";
    if (link) {
      loadWorldFromPermalink(link);
    }
  });
}

if (copyPermalinkBtn) {
  copyPermalinkBtn.addEventListener("click", () => {
    if (remoteMode || activeLaunchInfo?.shareable === false) return;
    if (!currentPermalink && simHandle) {
      try {
        if (permalink_of) {
          currentPermalink = permalink_of(simHandle);
        } else if (simHandle.permalinkOf) {
          currentPermalink = simHandle.permalinkOf();
        }
      } catch (error) {
        if (permalinkStatus) {
          permalinkStatus.textContent = `Recipe export unavailable: ${error.message || error}`;
          permalinkStatus.style.color = "#f87171";
        }
        return;
      }
    }
    if (currentPermalink) {
      navigator.clipboard
        .writeText(currentPermalink)
        .then(() => {
          if (permalinkStatus) {
            permalinkStatus.textContent = "Initial-condition recipe copied; this link restarts at tick zero";
            permalinkStatus.style.color = "#34d399";
          }
        })
        .catch(() => {
          if (permalinkInput) {
            permalinkInput.select();
          }
        });
    }
  });
}

if (forkBtn) {
  forkBtn.addEventListener("click", handleFork);
}

if (loadForkedBtn) {
  loadForkedBtn.addEventListener("click", () => {
    if (latestForkedLink) {
      loadWorldFromPermalink(latestForkedLink);
    }
  });
}

if (copyForkBtn) {
  copyForkBtn.addEventListener("click", () => {
    if (latestForkedLink) {
      navigator.clipboard.writeText(latestForkedLink).then(() => {
        if (permalinkStatus) {
          permalinkStatus.textContent = "Variant recipe link copied; this link restarts at tick zero";
          permalinkStatus.style.color = "#34d399";
        }
      });
    }
  });
}

async function resetSimulation(populationOverride) {
  if (remoteMode) return;
  const defaults = default_init_options ? default_init_options() : {};
  const seed = Math.floor(Math.random() * 1_000_000);
  const preset = brainPresetSelect?.value || "mlp";
  if (preset === "mixed_continuation") throw new Error("Choose a new-world brain preset before resetting this mixed continuation");
  const evolutionary = ["mlp", "dwraon", "assembly"].includes(preset);
  const mixed = preset === "mixed";
  if (!evolutionary && !mixed && !["wander", "none"].includes(preset)) throw new Error("Choose a supported brain family");
  const options = {
    ...defaults,
    seed,
    population: populationOverride,
    default_brain: evolutionary ? preset : null,
    mixed_brains: mixed,
    seed_strategy: preset === "wander" ? "wander" : "none",
  };

  const installed = replaceSimulation(init_sim(options), { expectedBrainPreset: mixed ? "mixed" : null });
  currentPermalink = installed.permalink;
  if (permalinkInput) permalinkInput.value = currentPermalink ?? "";

  updateMismatchBanner(null);
  updateDiffTable(currentPermalink);

  if (permalinkStatus) {
    permalinkStatus.textContent = "";
  }
  if (forkOutput) {
    forkOutput.style.display = "none";
  }

  try {
    const url = new URL(window.location.href);
    url.searchParams.delete("world");
    url.searchParams.delete("permalink");
    window.history.replaceState({}, "", url.toString());
  } catch (_) {}

  appendLog(`Simulation reset (seed=${seed}, population=${populationOverride})`);
}

checkpointSaveButton?.addEventListener("click", () => {
  if (remoteMode || !simHandle || checkpointLoadPending) return;
  try {
    driveSimulation();
    const bytes = simHandle.checkpoint();
    const tick = simHandle.frame().identity.tick;
    const url = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `scriptbots-tick-${tick}.sbcheckpoint`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    checkpointStatus.textContent = `Continuation file prepared at tick ${tick}.`;
  } catch (error) {
    checkpointStatus.textContent = `Save failed: ${error.message || error}`;
  }
});

function installContinuation(bytes, expectedTick = null, scientificCheckpoint = false) {
  const next = scientificCheckpoint ? wasm.import_world_checkpoint(bytes) : wasm.restore_checkpoint(bytes);
  const installed = replaceSimulation(next, { continuation: true, expectedTick,
    expectedLaunchKind: scientificCheckpoint ? "scientific_checkpoint" : null });
  currentPermalink = null;
  if (permalinkInput) permalinkInput.value = "";
  latestForkedLink = null;
  if (forkOutput) forkOutput.style.display = "none";
  updateMismatchBanner(null);
  updateDiffTable(null);
  try {
    const url = new URL(window.location.href);
    url.searchParams.delete("world");
    url.searchParams.delete("permalink");
    window.history.replaceState({}, "", url.toString());
  } catch (_) {}
  return installed;
}

worldCheckpointExport?.addEventListener("click", () => {
  if (remoteMode || !simHandle || checkpointLoadPending || browserStorePending || browserRecovery?.pending
      || typeof simHandle.worldCheckpoint !== "function" || typeof wasm.world_checkpoint_byte_limit !== "function") return;
  try {
    // Export observes the current completed cut. It does not drive the owner or
    // change playback; the explicit import path creates a separate paused owner.
    const bytes = simHandle.worldCheckpoint();
    if (!(bytes instanceof Uint8Array) || bytes.length === 0 || bytes.length > wasm.world_checkpoint_byte_limit()) {
      throw new Error("The package returned an invalid or oversized scientific checkpoint");
    }
    const frame = simHandle.frame();
    validateFrameIdentity(frame);
    const url = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `scriptbots-scientific-tick-${frame.identity.tick}.sbworld`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    worldCheckpointStatus.textContent = `Scientific checkpoint prepared at tick ${frame.identity.tick}. It contains saved science for a new session or compatible native tools, without the current host's journal or scheduled commands.`;
  } catch (error) {
    worldCheckpointStatus.textContent = `Scientific export failed: ${error.message || error}`;
  }
});

worldCheckpointImport?.addEventListener("change", async () => {
  const file = worldCheckpointImport.files?.[0];
  worldCheckpointImport.value = "";
  if (!file || remoteMode || !wasmInitialized || checkpointLoadPending || browserStorePending || browserRecovery?.pending
      || typeof wasm.import_world_checkpoint !== "function" || typeof wasm.world_checkpoint_byte_limit !== "function") return;
  const generation = simulationGeneration;
  const original = simHandle;
  checkpointLoadPending = true;
  worldCheckpointStatus.textContent = "Reading scientific checkpoint…";
  updateCheckpointControls();
  try {
    if (file.size === 0 || file.size > wasm.world_checkpoint_byte_limit()) {
      throw new Error("Scientific checkpoint is empty or exceeds the browser size limit");
    }
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (continuationReadIsStale(generation, original)) {
      worldCheckpointStatus.textContent = "Scientific import cancelled because the active world changed.";
      return;
    }
    const installed = installContinuation(bytes, null, true);
    worldCheckpointStatus.textContent = `Imported scientific state at tick ${installed.frame.identity.tick}; the new browser session is paused. Original host commands, scheduled changes and durable history are not continued.`;
  } catch (error) {
    worldCheckpointStatus.textContent = `Scientific import failed: ${error.message || error}`;
  } finally {
    checkpointLoadPending = false;
    updateCheckpointControls();
  }
});

function continuationReadIsStale(generation, original) {
  return remoteMode || queuedReset || simulationGeneration !== generation || simHandle !== original;
}

function storedContinuationDescription(record) {
  return `tick ${record.tick}, saved ${new Date(record.savedAt).toLocaleString()}`;
}

async function discoverBrowserContinuation() {
  if (browserStorePending || checkpointLoadPending || !browserContinuationStore) return;
  browserStorePending = true;
  browserCheckpointStatus.textContent = "Checking for a saved browser continuation…";
  updateCheckpointControls();
  try {
    const record = await browserContinuationStore.read();
    browserCheckpointStatus.textContent = record
      ? `Saved continuation found: ${storedContinuationDescription(record)}. Restore validates and pauses this world.`
      : "No continuation saved in this browser. Save replaces the one saved slot.";
  } catch (error) {
    browserCheckpointStatus.textContent = `Browser save discovery failed: ${error.message || error}`;
  }
  try {
    const record = await browserContinuationStore.readRecovery();
    recoveryRecord = record ? { tick: record.tick, savedAt: record.savedAt,
      byteLength: record.blob.size, source: record.source } : null;
    recoveryReadStatus = record ? "Restore Recovery World validates this saved state and starts paused."
      : "No recovery checkpoint is stored yet.";
  } catch (error) {
    recoveryReadStatus = `Recovery discovery failed: ${error.message || error}`;
  } finally {
    browserStorePending = false;
    updateCheckpointControls();
  }
}

recoveryEnabled?.addEventListener("change", () => {
  if (!browserRecovery || remoteMode || !simHandle || activeBrainPreset === "wander"
      || browserStorePending || checkpointLoadPending) {
    renderRecoveryControls();
    return;
  }
  recoveryReadStatus = "";
  browserRecovery.setEnabled(recoveryEnabled.checked);
});

recoveryRetryButton?.addEventListener("click", () => {
  if (!browserRecovery?.enabled || remoteMode || browserStorePending || checkpointLoadPending || browserRecovery.pending) return;
  recoveryReadStatus = "";
  browserRecovery.retry();
});

recoveryRestoreButton?.addEventListener("click", async () => {
  if (remoteMode || !wasmInitialized || !browserContinuationStore || browserStorePending || checkpointLoadPending || browserRecovery?.pending) return;
  const generation = simulationGeneration;
  const original = simHandle;
  browserRecovery?.setEnabled(false);
  browserStorePending = true;
  checkpointLoadPending = true;
  recoveryReadStatus = "Reading the recovery checkpoint…";
  updateCheckpointControls();
  try {
    const record = await browserContinuationStore.readRecovery();
    if (continuationReadIsStale(generation, original)) {
      recoveryReadStatus = "Recovery restore cancelled because the active world changed.";
      return;
    }
    if (!record) {
      recoveryRecord = null;
      recoveryReadStatus = "No recovery checkpoint is stored yet.";
      return;
    }
    recoveryRecord = { tick: record.tick, savedAt: record.savedAt, byteLength: record.blob.size, source: record.source };
    const bytes = new Uint8Array(await record.blob.arrayBuffer());
    if (continuationReadIsStale(generation, original)) {
      recoveryReadStatus = "Recovery restore cancelled because the active world changed.";
      return;
    }
    const installed = installContinuation(bytes, record.tick);
    recoveryReadStatus = `Restored recovery at tick ${installed.frame.identity.tick}; paused. Enable recovery explicitly for this restored world.`;
  } catch (error) {
    recoveryReadStatus = `Recovery restore failed: ${error.message || error}`;
  } finally {
    browserStorePending = false;
    checkpointLoadPending = false;
    updateCheckpointControls();
  }
});

browserSaveButton?.addEventListener("click", async () => {
  if (remoteMode || !simHandle || !browserContinuationStore || browserStorePending || checkpointLoadPending || browserRecovery?.pending) return;
  const generation = simulationGeneration;
  browserStorePending = true;
  browserCheckpointStatus.textContent = "Saving continuation in this browser…";
  updateCheckpointControls();
  try {
    driveSimulation();
    const bytes = simHandle.checkpoint();
    const frame = simHandle.frame();
    validateFrameIdentity(frame);
    const record = await browserContinuationStore.save(
      new Blob([bytes], { type: "application/octet-stream" }), frame.identity.tick,
    );
    browserCheckpointStatus.textContent = `Saved continuation in this browser: ${storedContinuationDescription(record)}.`
      + (generation !== simulationGeneration || remoteMode ? " The displayed world has since changed." : "");
  } catch (error) {
    browserCheckpointStatus.textContent = `Browser save failed: ${error.message || error}`;
  } finally {
    browserStorePending = false;
    updateCheckpointControls();
  }
});

browserRestoreButton?.addEventListener("click", async () => {
  if (remoteMode || !wasmInitialized || !browserContinuationStore || browserStorePending || checkpointLoadPending || browserRecovery?.pending) return;
  const generation = simulationGeneration;
  const original = simHandle;
  browserStorePending = true;
  checkpointLoadPending = true;
  browserCheckpointStatus.textContent = "Reading the saved browser continuation…";
  updateCheckpointControls();
  try {
    const record = await browserContinuationStore.read();
    if (continuationReadIsStale(generation, original)) {
      browserCheckpointStatus.textContent = "Browser restore cancelled because the active world changed.";
      return;
    }
    if (!record) {
      browserCheckpointStatus.textContent = "No continuation saved in this browser.";
      return;
    }
    // read() checks Blob.size before allocating its contents. A second identity
    // check after the read prevents reset or remote mode losing their new owner.
    const bytes = new Uint8Array(await record.blob.arrayBuffer());
    if (continuationReadIsStale(generation, original)) {
      browserCheckpointStatus.textContent = "Browser restore cancelled because the active world changed.";
      return;
    }
    const installed = installContinuation(bytes, record.tick);
    browserCheckpointStatus.textContent = `Restored browser continuation at tick ${installed.frame.identity.tick}; paused.`;
  } catch (error) {
    browserCheckpointStatus.textContent = `Browser restore failed: ${error.message || error}`;
  } finally {
    browserStorePending = false;
    checkpointLoadPending = false;
    updateCheckpointControls();
  }
});

checkpointLoadInput?.addEventListener("change", async () => {
  const file = checkpointLoadInput.files?.[0];
  checkpointLoadInput.value = "";
  if (!file || remoteMode || !wasmInitialized || checkpointLoadPending || browserStorePending || browserRecovery?.pending) return;
  const generation = simulationGeneration;
  const original = simHandle;
  checkpointLoadPending = true;
  checkpointStatus.textContent = "Reading continuation file…";
  updateCheckpointControls();
  try {
    if (file.size > wasm.checkpoint_byte_limit()) throw new Error("Continuation file exceeds the browser size limit");
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (continuationReadIsStale(generation, original)) {
      checkpointStatus.textContent = "Load cancelled because the active world changed.";
      return;
    }
    const installed = installContinuation(bytes);
    checkpointStatus.textContent = `Loaded continuation at tick ${installed.frame.identity.tick}; paused.`;
  } catch (error) {
    checkpointStatus.textContent = `Load failed: ${error.message || error}`;
  } finally {
    checkpointLoadPending = false;
    updateCheckpointControls();
  }
});

async function bootstrap() {
  try {
    await initWasm();
    wasmInitialized = true;
    browserContinuationStore = new BrowserContinuationStore(wasm.checkpoint_byte_limit());
    browserRecovery = new BrowserRecoveryController(browserContinuationStore, {
      onChange: updateCheckpointControls,
      onSaved: (record) => { recoveryRecord = record; recoveryReadStatus = "Recovery transaction completed in this browser."; },
    });
    // GPU initialization must not hold up science or the first visible frame.
    await selectRenderer("canvas");
    if (rendererSelect.value !== "canvas") {
      void selectRenderer(rendererSelect.value).catch(reportRendererFailure);
    }
    const tag = version ? version() : "unknown";
    if (versionEl) {
      versionEl.textContent = tag;
    }
    appendLog(`Loaded ${tag}`);

    const urlParams = new URLSearchParams(window.location.search);
    const worldParam = urlParams.get("world") || urlParams.get("permalink");
    if (worldParam && init_from_permalink && !queuedReset) {
      await loadWorldFromPermalink(worldParam);
    } else {
      queuedReset = false;
      await resetSimulation(population);
    }

    scheduleSimulation();
    remoteConnect.disabled = false;
    requestAnimationFrame(presentSimulation);
    void discoverBrowserContinuation();
  } catch (err) {
    console.error("Failed to initialise wasm module", err);
    appendLog(`Bootstrap failed: ${err}`);
  }
}

bootstrap();
