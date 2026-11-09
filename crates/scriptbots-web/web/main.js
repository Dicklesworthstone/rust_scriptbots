import initWasm, * as wasm from "./pkg/scriptbots_web.js";
import { CanvasRenderer } from "./renderers/canvas.js";
import { WebGpuRenderer } from "./renderers/webgpu.js";
import { environmentDescription } from "./renderers/environment.js";
import { RemoteStreamClient } from "./remote.js";
import { BrowserContinuationStore } from "./continuation-store.js";
import { InspectionController, pickAgent, renderInspector, renderInspectorSource } from "./inspector.js";

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
const captureButton = document.getElementById("capture-btn");
const captureStatus = document.getElementById("capture-status");
let captureRequest = null;
let encodingRequest = null;
let captureUrl = null;
let renderer = null;
let rendererGeneration = 0;
let lastPaintedFrame = null;
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
let checkpointLoadPending = false;
const browserSaveButton = document.getElementById("checkpoint-browser-save");
const browserRestoreButton = document.getElementById("checkpoint-browser-restore");
const browserCheckpointStatus = document.getElementById("checkpoint-browser-status");
let browserContinuationStore = null;
let browserStorePending = false;
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
const remoteEvents = document.getElementById("remote-events");
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
const loadForkedBtn = document.getElementById("load-forked-btn");
const copyForkBtn = document.getElementById("copy-fork-btn");
const diffContainer = document.getElementById("diff-container");
const diffEmpty = document.getElementById("diff-empty");
const diffTable = document.getElementById("diff-table");
const diffTbody = document.getElementById("diff-tbody");

let currentPermalink = null;
let latestForkedLink = null;

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

function selectCreature(uid) {
  inspector.select(uid);
  // Selection is presentation-only, so paused frames also need repainting.
  lastPaintedFrame = null;
}

document.getElementById("sim-container")?.addEventListener("click", (event) => {
  if (event.target !== canvas || event.button !== 0 || !inspector.frame) return;
  const bounds = canvas.getBoundingClientRect();
  const uid = pickAgent(inspector.frame.snapshot, event.clientX - bounds.left, event.clientY - bounds.top,
    { width: bounds.width, height: bounds.height, canvasWidth: canvas.width, canvasHeight: canvas.height });
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

stepsSlider.addEventListener("input", () => {
  if (remoteMode) return;
  speedMultiplier = Number(stepsSlider.value);
  applyPlaybackSpeed();
  appendLog(`Simulation speed set to ${speedMultiplier}×`);
});

document.addEventListener("visibilitychange", applyPlaybackSpeed);

populationSlider.addEventListener("input", () => {
  population = Number(populationSlider.value);
});

resetButton.addEventListener("click", () => {
  if (remoteMode) return;
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
  return remoteMode ? remoteFrame !== null : simHandle !== null;
}

function updateRemoteControls() {
  for (const input of [stepsSlider, populationSlider, resetButton, permalinkInput,
    loadPermalinkBtn, copyPermalinkBtn, forkPatchInput, forkBtn, loadForkedBtn, copyForkBtn]) {
    if (input) input.disabled = remoteMode;
  }
  remoteLocal.disabled = !remoteMode;
  remoteReconnect.disabled = !remoteMode || !remoteClient || remoteConnected;
  for (const button of [remotePause, remoteResume, remoteStep]) button.disabled = !remoteConnected;
  remoteRefresh.disabled = !remoteConnected || !lastRemoteCommandId;
  remoteRetry.disabled = !remoteConnected || !lastRemoteRequestId;
  captureButton.disabled = !hasPresentationFrame() || !!captureRequest || !!encodingRequest;
  remoteOverlay.hidden = !remoteMode || !!remoteFrame;
  updateCheckpointControls();
}

function updateCheckpointControls() {
  if (brainPresetSelect) brainPresetSelect.disabled = remoteMode;
  if (checkpointSaveButton) checkpointSaveButton.disabled = remoteMode || !simHandle || checkpointLoadPending;
  if (checkpointLoadInput) checkpointLoadInput.disabled = remoteMode || !wasmInitialized || checkpointLoadPending || browserStorePending;
  const storageBusy = remoteMode || !browserContinuationStore || checkpointLoadPending || browserStorePending;
  if (browserSaveButton) browserSaveButton.disabled = storageBusy || !simHandle;
  if (browserRestoreButton) browserRestoreButton.disabled = storageBusy;
}

function remoteEventLine(text) {
  const item = document.createElement("li");
  item.textContent = text;
  remoteEvents.append(item);
  while (remoteEvents.children.length > 48) remoteEvents.firstElementChild.remove();
  remoteEvents.scrollTop = remoteEvents.scrollHeight;
}

function receiveRemoteFrame(frame) {
  if (!remoteMode) return;
  remotePlayback.textContent = `Observed server: ${frame.playback.paused ? "paused" : "running"}, ${frame.playback.speedMultiplier}× — tick ${frame.identity.tick}`;
  if (frame.kind === "canvas") {
    const browserFrame = { schema: "scriptbots.browser-frame.v1", identity: frame.identity,
      snapshot: frame.payload, environment: frame.environment };
    validateFrameIdentity(browserFrame);
    remoteFrame = browserFrame;
    if (frame.coalescedPublications !== "0") {
      remoteStatus.textContent = `Connected — latest publication ${frame.identity.publicationRevision}; skipped ${frame.coalescedPublications} intermediate publications.`;
    }
    updateRemoteControls();
  } else if (frame.kind === "events") {
    const page = frame.payload;
    if (page.gap) {
      remoteEventLine(`History gap: narrative events ${page.gap.firstMissing}–${page.gap.lastMissing} are no longer retained by this server.`);
    }
    for (const event of page.events) {
      remoteEventLine(`#${event.sequence} · tick ${event.record.tick}: ${event.record.humanText}`);
    }
  }
}

function receiveRemoteState(state) {
  if (!remoteMode) return;
  remoteConnected = state.state === "connected";
  remoteStatus.dataset.state = state.state;
  remoteStatus.textContent = state.message || state.state;
  if (["disconnected", "error", "session_changed"].includes(state.state) && remoteFrame) {
    remoteStatus.textContent += " Showing the last received frame.";
  }
  if (state.state === "session_changed") {
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
  try {
    // Freeze local science before changing the active source. The saved local
    // owner is resumed only by the user's explicit Return to Local action.
    if (!remoteMode && simHandle) {
      driveSimulation();
      simHandle.setSpeed(0);
    }
    if (simulationTimer !== null) clearTimeout(simulationTimer);
    simulationTimer = null;
    queuedReset = false;
    remoteClient?.disconnect();
    remoteClient = null;
    remoteMode = true;
    remoteConnected = false;
    remoteFrame = null;
    lastRemoteRequestId = null;
    lastRemoteCommandId = null;
    lastRemoteStatusRequestId = null;
    simulationGeneration += 1;
    inspector.reset();
    lastPaintedFrame = null;
    if (captureRequest) finishCapture(captureRequest, new Error("Presentation source changed"));
    remoteEvents.replaceChildren();
    remoteReceipt.textContent = "";
    remotePlayback.textContent = "Waiting for observed server state…";
    remoteClient = new RemoteStreamClient({
      url: remoteUrl.value.trim(),
      decodeFrame: wasm.decode_remote_frame_binary,
      channels: ["canvas", "telemetry", "events"],
      maxFps: 60,
      onFrame: receiveRemoteFrame,
      onState: receiveRemoteState,
      onReceipt: receiveRemoteReceipt,
    });
    updateMismatchBanner(null);
    resetPerformanceWindows();
    updateRemoteControls();
    remoteClient.connect();
  } catch (error) {
    receiveRemoteState({ state: "error", message: `Connection failed: ${error.message}` });
  }
}

remoteConnect.addEventListener("click", connectRemote);
remoteReconnect.addEventListener("click", () => {
  try { remoteClient?.reconnect(); }
  catch (error) { receiveRemoteState({ state: "error", message: error.message }); }
});
remoteLocal.addEventListener("click", () => {
  remoteClient?.disconnect();
  remoteMode = false;
  remoteConnected = false;
  remoteClient = null;
  remoteFrame = null;
  simulationGeneration += 1;
  inspector.reset();
  lastPaintedFrame = null;
  if (captureRequest) finishCapture(captureRequest, new Error("Presentation source changed"));
  remoteStatus.textContent = "Running locally in this browser.";
  remoteStatus.dataset.state = "local";
  remotePlayback.textContent = "";
  remoteReceipt.textContent = "";
  resetPerformanceWindows();
  updateRemoteControls();
  applyPlaybackSpeed();
  scheduleSimulation();
});

function sendRemoteControl(command) {
  try {
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
  renderer?.destroy();
  canvas.replaceWith(nextCanvas);
  canvas = nextCanvas;
  renderer = nextRenderer;
  lastPaintedFrame = null;
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

function drawSnapshot(frame) {
  if (!renderer) return false;
  try {
    validateFrameIdentity(frame);
  } catch (error) {
    if (captureRequest) finishCapture(captureRequest, error);
    else showCaptureError(`Frame refused: ${error.message}`);
    return false;
  }
  if (lastPaintedFrame?.simulationGeneration === simulationGeneration
      && lastPaintedFrame.publicationRevision === frame.identity.publicationRevision
      && lastPaintedFrame.sessionId === frame.identity.sessionId
      && lastPaintedFrame.width === canvas.width && lastPaintedFrame.height === canvas.height
      && (!captureRequest || captureRequest.encoding)) {
    inspector.observe(frame, simulationGeneration);
    return true;
  }
  try {
    if (renderer.render(frame.snapshot, { environment: frame.environment, identity: frame.identity,
      layer: environmentLayer?.value ?? "combined", generation: simulationGeneration,
      selectedUid: inspector.selectionFor(frame, simulationGeneration) })) {
      lastPaintedFrame = { simulationGeneration, publicationRevision: frame.identity.publicationRevision,
        sessionId: frame.identity.sessionId, width: canvas.width, height: canvas.height };
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
      captureAdmittedFrame(frame.identity);
      inspector.observe(frame, simulationGeneration);
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
  const request = { simulationGeneration, encoding: false, timeout: null };
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
    if (request.simulationGeneration !== simulationGeneration) throw new Error("World changed before capture; try again");
    const identity = { ...frameIdentity };
    const backend = rendererStatus.dataset.renderer;
    const filename = `scriptbots-session-${identity.sessionId}-p${identity.publicationRevision}-s${identity.scientificRevision}-c${identity.controlRevision}-g${identity.configRevision}-t${identity.tick}-${backend}.png`;
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
        if (request.simulationGeneration !== simulationGeneration) throw new Error("World changed during capture; try again");
        if (!blob || blob.type !== "image/png") throw new Error("The browser did not produce a PNG");
        if (captureUrl) URL.revokeObjectURL(captureUrl);
        captureUrl = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = captureUrl;
        link.download = filename;
        link.click();
        Object.assign(captureStatus.dataset, identity, { renderer: backend, state: "ready", filename });
        captureStatus.textContent = `PNG ready — tick ${identity.tick}, science revision ${identity.scientificRevision}, publication ${identity.publicationRevision} (${backend === "webgpu" ? "WebGPU" : "Canvas 2D"})`;
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
  remoteClient?.disconnect();
  if (captureRequest) finishCapture(captureRequest, new Error("Page left during capture"));
  if (captureUrl) URL.revokeObjectURL(captureUrl);
  captureUrl = null;
});

function updateStats(snapshot, now) {
  const frameDt = now - metrics.lastFrameTs;
  metrics.lastFrameTs = now;

  if (frameDt > 0) {
    const instantaneous = 1000 / frameDt;
    metrics.fps = metrics.fps * 0.85 + instantaneous * 0.15;
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

    const frame = remoteMode ? remoteFrame : simHandle.frame();
    const presented = drawSnapshot(frame);
    updateStats(frame.snapshot, now);
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

function replaceSimulation(next, { continuation = false, permalink = null, expectedTick = null } = {}) {
  const nextSpeed = continuation ? 0 : speedMultiplier;
  let frame;
  let brainPreset;
  try {
    // All fallible owner reads happen before the active world, clock, or UI is
    // changed. Refusing a candidate must leave the previous owner usable.
    next.setSpeed(document.hidden ? 0 : nextSpeed);
    frame = next.frame();
    validateFrameIdentity(frame);
    if (expectedTick !== null && frame.identity.tick !== expectedTick) {
      throw new Error("Saved continuation tick does not match its checkpoint bytes");
    }
    if (continuation) {
      brainPreset = next.brainPreset();
    } else if (permalink_of) {
      permalink = permalink_of(next);
    } else if (next.permalinkOf) {
      permalink = next.permalinkOf();
    }
  } catch (error) {
    releaseSimulation(next);
    throw error;
  }

  const previous = simHandle;
  simHandle = next;
  simulationGeneration += 1;
  inspector.reset();
  simulationStartedAt = performance.now();
  lastPaintedFrame = null;
  if (continuation) {
    speedMultiplier = 0;
    stepsSlider.value = "0";
    if (brainPresetSelect) brainPresetSelect.value = brainPreset;
  }
  resetPerformanceWindows(frame.snapshot);
  if (captureRequest) finishCapture(captureRequest, new Error("World changed during capture; try again"));
  captureButton.disabled = !!captureRequest || !!encodingRequest;
  updateCheckpointControls();
  if (previous) releaseSimulation(previous);
  scheduleSimulation();
  return { frame, permalink };
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
  // One outstanding callback, no snapshot queue. HostCore owns deadlines and
  // its unchanged catch-up bound; timer jitter never becomes a render tick.
  scheduleSimulation();
}

function bannerLine(parent, text, className = "") {
  const line = document.createElement("div");
  line.className = className;
  line.textContent = text;
  parent.append(line);
  return line;
}

function updateMismatchBanner(matchResult) {
  if (!mismatchBannerEl) return;

  if (!matchResult || matchResult.status === "exact") {
    mismatchBannerEl.className = "";
    mismatchBannerEl.style.display = "none";
    mismatchBannerEl.replaceChildren();
    return;
  }

  if (matchResult.status !== "compatible" && matchResult.status !== "mismatch") return;
  const compatible = matchResult.status === "compatible";
  mismatchBannerEl.replaceChildren();
  mismatchBannerEl.className = compatible ? "banner-compatible" : "banner-mismatch";
  mismatchBannerEl.style.display = "block";
  const title = bannerLine(mismatchBannerEl, "", "banner-title");
  const icon = document.createElement("span");
  icon.textContent = compatible ? "ⓘ" : "⚠";
  const titleText = document.createElement("span");
  titleText.textContent = compatible ? "Compatible Build Note" : "Build Mismatch Warning";
  title.append(icon, titleText);
  bannerLine(mismatchBannerEl, compatible
    ? "This world was recorded on a compatible build. Toolchain or lockfile differ, but core simulation digest matches."
    : `This world was recorded on a different build (core digest ${matchResult.link_core_digest}). You are on build (core digest ${matchResult.local_core_digest}). Trajectories may diverge after tick 0.`);
  const details = bannerLine(mismatchBannerEl, "", "banner-details");
  bannerLine(details, `Link Build: toolchain=${matchResult.link_toolchain_digest}, lockfile=${matchResult.link_lockfile_digest}, core=${matchResult.link_core_digest}`);
  bannerLine(details, `Running Build: toolchain=${matchResult.local_toolchain_digest}, lockfile=${matchResult.local_lockfile_digest}, core=${matchResult.local_core_digest}`);
  if (!compatible) {
    const note = bannerLine(details, "Cross-target floating point equivalence between native and WebAssembly is an open research area (bd-2wj).");
    note.style.cssText = "margin-top: 0.35rem; font-style: italic; opacity: 0.85;";
  }
}

function updateDiffTable(linkStr) {
  if (!diffTable || !diffTbody || !diffEmpty) return;

  if (!linkStr || !permalink_diff) {
    diffTable.style.display = "none";
    diffEmpty.style.display = "block";
    diffEmpty.textContent = "No knob differences from base scenario.";
    return;
  }

  try {
    const rows = permalink_diff(linkStr);
    if (!rows || rows.length === 0) {
      diffTable.style.display = "none";
      diffEmpty.style.display = "block";
      diffEmpty.textContent = "No knob differences from base scenario (0 diffs).";
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
      permalinkStatus.textContent = "World loaded from permalink";
      permalinkStatus.style.color = "#34d399";
    }
    appendLog(`Loaded permalink world (${linkStr.slice(0, 20)}…)`);

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
  if (remoteMode || !simHandle) {
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

    latestForkedLink = childLink;
    if (forkChildLink) {
      forkChildLink.textContent = childLink;
    }
    if (forkOutput) {
      forkOutput.style.display = "block";
    }
    if (permalinkStatus) {
      permalinkStatus.textContent = "Fork created successfully";
      permalinkStatus.style.color = "#34d399";
    }
    appendLog(`Created fork child link (${childLink.slice(0, 20)}…)`);
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
    if (remoteMode) return;
    if (!currentPermalink && simHandle) {
      try {
        if (permalink_of) {
          currentPermalink = permalink_of(simHandle);
        } else if (simHandle.permalinkOf) {
          currentPermalink = simHandle.permalinkOf();
        }
      } catch (_) {}
    }
    if (currentPermalink) {
      navigator.clipboard
        .writeText(currentPermalink)
        .then(() => {
          if (permalinkStatus) {
            permalinkStatus.textContent = "Permalink copied to clipboard";
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
          permalinkStatus.textContent = "Fork link copied to clipboard";
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
  const options = {
    ...defaults,
    seed,
    population: populationOverride,
    default_brain: brainPresetSelect?.value === "mlp" ? "mlp" : null,
    seed_strategy: brainPresetSelect?.value === "none" || brainPresetSelect?.value === "mlp" ? "none" : "wander",
  };

  const installed = replaceSimulation(init_sim(options));
  currentPermalink = installed.permalink;
  if (currentPermalink && permalinkInput) {
    permalinkInput.value = currentPermalink;
  }

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

function installContinuation(bytes, expectedTick = null) {
  const installed = replaceSimulation(wasm.restore_checkpoint(bytes), { continuation: true, expectedTick });
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
  } finally {
    browserStorePending = false;
    updateCheckpointControls();
  }
}

browserSaveButton?.addEventListener("click", async () => {
  if (remoteMode || !simHandle || !browserContinuationStore || browserStorePending || checkpointLoadPending) return;
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
  if (remoteMode || !wasmInitialized || !browserContinuationStore || browserStorePending || checkpointLoadPending) return;
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
  if (!file || remoteMode || !wasmInitialized || checkpointLoadPending || browserStorePending) return;
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
